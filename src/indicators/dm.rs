//! # Directional Movement (DM)
//!
//! Measures the strength of upward and downward price movements based on changes
//! between consecutive high and low values. +DM is computed when the positive
//! range (current high minus previous high) exceeds the negative range (previous
//! low minus current low), while -DM is computed in the opposite case. Both
//! values can be optionally smoothed over the specified `period`.
//!
//! ## Parameters
//! - **period**: The smoothing window size (number of data points). Defaults to 14.
//!
//! ## Errors
//! - **AllValuesNaN**: dm: All input data values are `NaN`.
//! - **EmptyData**: dm: Input high/low slices are empty or mismatched in length.
//! - **InvalidPeriod**: dm: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: dm: Not enough valid data points for the requested `period`.
//!
//! ## Returns
//! - **`Ok(DmOutput)`** on success, containing two `Vec<f64>` matching the input length.
//! - **`Err(DmError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum DmData<'a> {
	Candles { candles: &'a Candles },
	Slices { high: &'a [f64], low: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct DmOutput {
	pub plus: Vec<f64>,
	pub minus: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct DmParams {
	pub period: Option<usize>,
}

impl Default for DmParams {
	fn default() -> Self {
		Self { period: Some(14) }
	}
}

#[derive(Debug, Clone)]
pub struct DmInput<'a> {
	pub data: DmData<'a>,
	pub params: DmParams,
}

impl<'a> DmInput<'a> {
	#[inline]
	pub fn from_candles(candles: &'a Candles, params: DmParams) -> Self {
		Self {
			data: DmData::Candles { candles },
			params,
		}
	}
	#[inline]
	pub fn from_slices(high: &'a [f64], low: &'a [f64], params: DmParams) -> Self {
		Self {
			data: DmData::Slices { high, low },
			params,
		}
	}
	#[inline]
	pub fn with_default_candles(candles: &'a Candles) -> Self {
		Self {
			data: DmData::Candles { candles },
			params: DmParams::default(),
		}
	}
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params
			.period
			.unwrap_or_else(|| DmParams::default().period.unwrap())
	}
}

#[derive(Copy, Clone, Debug)]
pub struct DmBuilder {
	period: Option<usize>,
	kernel: Kernel,
}

impl Default for DmBuilder {
	fn default() -> Self {
		Self {
			period: None,
			kernel: Kernel::Auto,
		}
	}
}

impl DmBuilder {
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
	pub fn apply(self, candles: &Candles) -> Result<DmOutput, DmError> {
		let p = DmParams { period: self.period };
		let i = DmInput::from_candles(candles, p);
		dm_with_kernel(&i, self.kernel)
	}

	#[inline(always)]
	pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<DmOutput, DmError> {
		let p = DmParams { period: self.period };
		let i = DmInput::from_slices(high, low, p);
		dm_with_kernel(&i, self.kernel)
	}

	#[inline(always)]
	pub fn into_stream(self) -> Result<DmStream, DmError> {
		let p = DmParams { period: self.period };
		DmStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum DmError {
	#[error("dm: Empty data provided or mismatched high/low lengths.")]
	EmptyData,
	#[error("dm: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("dm: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("dm: All values are NaN.")]
	AllValuesNaN,
}

#[inline]
pub fn dm(input: &DmInput) -> Result<DmOutput, DmError> {
	dm_with_kernel(input, Kernel::Auto)
}

pub fn dm_with_kernel(input: &DmInput, kernel: Kernel) -> Result<DmOutput, DmError> {
	let (high, low) = match &input.data {
		DmData::Candles { candles } => {
			let high = candles.select_candle_field("high").map_err(|_| DmError::EmptyData)?;
			let low = candles.select_candle_field("low").map_err(|_| DmError::EmptyData)?;
			(high, low)
		}
		DmData::Slices { high, low } => (*high, *low),
	};

	if high.is_empty() || low.is_empty() || high.len() != low.len() {
		return Err(DmError::EmptyData);
	}

	let period = input.get_period();
	if period == 0 || period > high.len() {
		return Err(DmError::InvalidPeriod {
			period,
			data_len: high.len(),
		});
	}

	let first_valid_idx = high
		.iter()
		.zip(low.iter())
		.position(|(&h, &l)| !h.is_nan() && !l.is_nan())
		.ok_or(DmError::AllValuesNaN)?;

	if (high.len() - first_valid_idx) < period {
		return Err(DmError::NotEnoughValidData {
			needed: period,
			valid: high.len() - first_valid_idx,
		});
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => dm_scalar(high, low, period, first_valid_idx),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => dm_avx2(high, low, period, first_valid_idx),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => dm_avx512(high, low, period, first_valid_idx),
			_ => unreachable!(),
		}
	}
}

#[inline]
pub unsafe fn dm_scalar(high: &[f64], low: &[f64], period: usize, first_valid_idx: usize) -> Result<DmOutput, DmError> {
	let mut plus_dm = vec![f64::NAN; high.len()];
	let mut minus_dm = vec![f64::NAN; high.len()];

	let mut prev_high = high[first_valid_idx];
	let mut prev_low = low[first_valid_idx];
	let mut sum_plus = 0.0;
	let mut sum_minus = 0.0;

	let end_init = first_valid_idx + period - 1;
	for i in (first_valid_idx + 1)..=end_init {
		let diff_p = high[i] - prev_high;
		let diff_m = prev_low - low[i];
		prev_high = high[i];
		prev_low = low[i];

		let plus_val = if diff_p > 0.0 && diff_p > diff_m { diff_p } else { 0.0 };
		let minus_val = if diff_m > 0.0 && diff_m > diff_p { diff_m } else { 0.0 };

		sum_plus += plus_val;
		sum_minus += minus_val;
	}

	plus_dm[end_init] = sum_plus;
	minus_dm[end_init] = sum_minus;

	let inv_period = 1.0 / (period as f64);

	for i in (end_init + 1)..high.len() {
		let diff_p = high[i] - prev_high;
		let diff_m = prev_low - low[i];
		prev_high = high[i];
		prev_low = low[i];

		let plus_val = if diff_p > 0.0 && diff_p > diff_m { diff_p } else { 0.0 };
		let minus_val = if diff_m > 0.0 && diff_m > diff_p { diff_m } else { 0.0 };

		sum_plus = sum_plus - (sum_plus * inv_period) + plus_val;
		sum_minus = sum_minus - (sum_minus * inv_period) + minus_val;

		plus_dm[i] = sum_plus;
		minus_dm[i] = sum_minus;
	}

	Ok(DmOutput {
		plus: plus_dm,
		minus: minus_dm,
	})
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn dm_avx2(high: &[f64], low: &[f64], period: usize, first_valid_idx: usize) -> Result<DmOutput, DmError> {
	dm_scalar(high, low, period, first_valid_idx)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn dm_avx512(high: &[f64], low: &[f64], period: usize, first_valid_idx: usize) -> Result<DmOutput, DmError> {
	dm_scalar(high, low, period, first_valid_idx)
}

// Long and short variants for AVX512, required by API parity
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn dm_avx512_short(
	high: &[f64],
	low: &[f64],
	period: usize,
	first_valid_idx: usize,
) -> Result<DmOutput, DmError> {
	dm_avx512(high, low, period, first_valid_idx)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn dm_avx512_long(
	high: &[f64],
	low: &[f64],
	period: usize,
	first_valid_idx: usize,
) -> Result<DmOutput, DmError> {
	dm_avx512(high, low, period, first_valid_idx)
}

#[derive(Debug, Clone)]
pub struct DmStream {
	period: usize,
	buf_high: Vec<f64>,
	buf_low: Vec<f64>,
	idx: usize,
	filled: bool,
	sum_plus: f64,
	sum_minus: f64,
	prev_high: f64,
	prev_low: f64,
	count: usize,
}

impl DmStream {
	pub fn try_new(params: DmParams) -> Result<Self, DmError> {
		let period = params.period.unwrap_or(14);
		if period == 0 {
			return Err(DmError::InvalidPeriod { period, data_len: 0 });
		}
		Ok(Self {
			period,
			buf_high: vec![f64::NAN; period],
			buf_low: vec![f64::NAN; period],
			idx: 0,
			filled: false,
			sum_plus: 0.0,
			sum_minus: 0.0,
			prev_high: f64::NAN,
			prev_low: f64::NAN,
			count: 0,
		})
	}

	#[inline(always)]
	pub fn update(&mut self, high: f64, low: f64) -> Option<(f64, f64)> {
		if self.count == 0 {
			self.prev_high = high;
			self.prev_low = low;
		}

		let diff_p = high - self.prev_high;
		let diff_m = self.prev_low - low;
		self.prev_high = high;
		self.prev_low = low;

		let plus_val = if diff_p > 0.0 && diff_p > diff_m { diff_p } else { 0.0 };
		let minus_val = if diff_m > 0.0 && diff_m > diff_p { diff_m } else { 0.0 };

		if self.count < self.period - 1 {
			self.sum_plus += plus_val;
			self.sum_minus += minus_val;
			self.count += 1;
			return None;
		} else if self.count == self.period - 1 {
			self.sum_plus += plus_val;
			self.sum_minus += minus_val;
			self.count += 1;
			return Some((self.sum_plus, self.sum_minus));
		}

		let inv_period = 1.0 / (self.period as f64);
		self.sum_plus = self.sum_plus - (self.sum_plus * inv_period) + plus_val;
		self.sum_minus = self.sum_minus - (self.sum_minus * inv_period) + minus_val;
		Some((self.sum_plus, self.sum_minus))
	}
}

#[derive(Clone, Debug)]
pub struct DmBatchRange {
	pub period: (usize, usize, usize),
}

impl Default for DmBatchRange {
	fn default() -> Self {
		Self { period: (14, 14, 0) }
	}
}

#[derive(Clone, Debug, Default)]
pub struct DmBatchBuilder {
	range: DmBatchRange,
	kernel: Kernel,
}

impl DmBatchBuilder {
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
	pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<DmBatchOutput, DmError> {
		dm_batch_with_kernel(high, low, &self.range, self.kernel)
	}
	pub fn apply_candles(self, c: &Candles) -> Result<DmBatchOutput, DmError> {
		let high = c.select_candle_field("high").map_err(|_| DmError::EmptyData)?;
		let low = c.select_candle_field("low").map_err(|_| DmError::EmptyData)?;
		self.apply_slices(high, low)
	}
	pub fn with_default_candles(c: &Candles) -> Result<DmBatchOutput, DmError> {
		DmBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c)
	}
}

#[derive(Clone, Debug)]
pub struct DmBatchOutput {
	pub plus: Vec<f64>,
	pub minus: Vec<f64>,
	pub combos: Vec<DmParams>,
	pub rows: usize,
	pub cols: usize,
}
impl DmBatchOutput {
	pub fn row_for_params(&self, p: &DmParams) -> Option<usize> {
		self.combos
			.iter()
			.position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
	}
	pub fn values_for(&self, p: &DmParams) -> Option<(&[f64], &[f64])> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			(
				&self.plus[start..start + self.cols],
				&self.minus[start..start + self.cols],
			)
		})
	}
}

#[inline(always)]
fn expand_grid(r: &DmBatchRange) -> Vec<DmParams> {
	let (start, end, step) = r.period;
	let periods = if step == 0 || start == end {
		vec![start]
	} else {
		(start..=end).step_by(step).collect()
	};
	periods.into_iter().map(|p| DmParams { period: Some(p) }).collect()
}

pub fn dm_batch_with_kernel(
	high: &[f64],
	low: &[f64],
	sweep: &DmBatchRange,
	k: Kernel,
) -> Result<DmBatchOutput, DmError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(DmError::InvalidPeriod { period: 0, data_len: 0 }),
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	dm_batch_par_slice(high, low, sweep, simd)
}

#[inline(always)]
pub fn dm_batch_slice(high: &[f64], low: &[f64], sweep: &DmBatchRange, kern: Kernel) -> Result<DmBatchOutput, DmError> {
	dm_batch_inner(high, low, sweep, kern, false)
}

#[inline(always)]
pub fn dm_batch_par_slice(
	high: &[f64],
	low: &[f64],
	sweep: &DmBatchRange,
	kern: Kernel,
) -> Result<DmBatchOutput, DmError> {
	dm_batch_inner(high, low, sweep, kern, true)
}

#[inline(always)]
fn dm_batch_inner(
	high: &[f64],
	low: &[f64],
	sweep: &DmBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<DmBatchOutput, DmError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(DmError::InvalidPeriod { period: 0, data_len: 0 });
	}

	let first = high
		.iter()
		.zip(low.iter())
		.position(|(&h, &l)| !h.is_nan() && !l.is_nan())
		.ok_or(DmError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if high.len() - first < max_p {
		return Err(DmError::NotEnoughValidData {
			needed: max_p,
			valid: high.len() - first,
		});
	}

	let rows = combos.len();
	let cols = high.len();
	let mut plus = vec![f64::NAN; rows * cols];
	let mut minus = vec![f64::NAN; rows * cols];

	let do_row = |row: usize, plus_row: &mut [f64], minus_row: &mut [f64]| unsafe {
		let period = combos[row].period.unwrap();
		match kern {
			Kernel::Scalar => dm_row_scalar(high, low, first, period, plus_row, minus_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => dm_row_avx2(high, low, first, period, plus_row, minus_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => dm_row_avx512(high, low, first, period, plus_row, minus_row),
			_ => unreachable!(),
		}
	};

	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			plus.par_chunks_mut(cols)
				.zip(minus.par_chunks_mut(cols))
				.enumerate()
				.for_each(|(row, (plus_row, minus_row))| do_row(row, plus_row, minus_row));
		}

		#[cfg(target_arch = "wasm32")]
		{
			for (row, (plus_row, minus_row)) in plus.chunks_mut(cols).zip(minus.chunks_mut(cols)).enumerate() {
				do_row(row, plus_row, minus_row);
			}
		}
	} else {
		for (row, (plus_row, minus_row)) in plus.chunks_mut(cols).zip(minus.chunks_mut(cols)).enumerate() {
			do_row(row, plus_row, minus_row);
		}
	}

	Ok(DmBatchOutput {
		plus,
		minus,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
unsafe fn dm_row_scalar(high: &[f64], low: &[f64], first: usize, period: usize, plus: &mut [f64], minus: &mut [f64]) {
	let mut prev_high = high[first];
	let mut prev_low = low[first];
	let mut sum_plus = 0.0;
	let mut sum_minus = 0.0;

	let end_init = first + period - 1;
	for i in (first + 1)..=end_init {
		let diff_p = high[i] - prev_high;
		let diff_m = prev_low - low[i];
		prev_high = high[i];
		prev_low = low[i];

		let plus_val = if diff_p > 0.0 && diff_p > diff_m { diff_p } else { 0.0 };
		let minus_val = if diff_m > 0.0 && diff_m > diff_p { diff_m } else { 0.0 };

		sum_plus += plus_val;
		sum_minus += minus_val;
	}

	plus[end_init] = sum_plus;
	minus[end_init] = sum_minus;

	let inv_period = 1.0 / (period as f64);

	for i in (end_init + 1)..high.len() {
		let diff_p = high[i] - prev_high;
		let diff_m = prev_low - low[i];
		prev_high = high[i];
		prev_low = low[i];

		let plus_val = if diff_p > 0.0 && diff_p > diff_m { diff_p } else { 0.0 };
		let minus_val = if diff_m > 0.0 && diff_m > diff_p { diff_m } else { 0.0 };

		sum_plus = sum_plus - (sum_plus * inv_period) + plus_val;
		sum_minus = sum_minus - (sum_minus * inv_period) + minus_val;

		plus[i] = sum_plus;
		minus[i] = sum_minus;
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn dm_row_avx2(high: &[f64], low: &[f64], first: usize, period: usize, plus: &mut [f64], minus: &mut [f64]) {
	dm_row_scalar(high, low, first, period, plus, minus)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn dm_row_avx512(high: &[f64], low: &[f64], first: usize, period: usize, plus: &mut [f64], minus: &mut [f64]) {
	dm_row_scalar(high, low, first, period, plus, minus)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn dm_row_avx512_short(
	high: &[f64],
	low: &[f64],
	first: usize,
	period: usize,
	plus: &mut [f64],
	minus: &mut [f64],
) {
	dm_row_avx512(high, low, first, period, plus, minus)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn dm_row_avx512_long(
	high: &[f64],
	low: &[f64],
	first: usize,
	period: usize,
	plus: &mut [f64],
	minus: &mut [f64],
) {
	dm_row_avx512(high, low, first, period, plus, minus)
}

#[inline(always)]
fn expand_grid_dm(_r: &DmBatchRange) -> Vec<DmParams> {
	expand_grid(_r)
}

//------------------ TESTS ----------------------------

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_dm_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let default_params = DmParams { period: None };
		let input_default = DmInput::from_candles(&candles, default_params);
		let output_default = dm_with_kernel(&input_default, kernel)?;
		assert_eq!(output_default.plus.len(), candles.high.len());
		assert_eq!(output_default.minus.len(), candles.high.len());

		let params_custom = DmParams { period: Some(10) };
		let input_custom = DmInput::from_candles(&candles, params_custom);
		let output_custom = dm_with_kernel(&input_custom, kernel)?;
		assert_eq!(output_custom.plus.len(), candles.high.len());
		assert_eq!(output_custom.minus.len(), candles.high.len());
		Ok(())
	}

	fn check_dm_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = DmInput::with_default_candles(&candles);
		let result = dm_with_kernel(&input, kernel)?;
		assert_eq!(result.plus.len(), candles.high.len());
		assert_eq!(result.minus.len(), candles.high.len());
		Ok(())
	}

	fn check_dm_with_slice_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high_values = [8000.0, 8050.0, 8100.0, 8075.0, 8110.0, 8050.0];
		let low_values = [7800.0, 7900.0, 7950.0, 7950.0, 8000.0, 7950.0];
		let params = DmParams { period: Some(3) };
		let input = DmInput::from_slices(&high_values, &low_values, params);
		let result = dm_with_kernel(&input, kernel)?;
		assert_eq!(result.plus.len(), 6);
		assert_eq!(result.minus.len(), 6);

		for i in 0..2 {
			assert!(result.plus[i].is_nan());
			assert!(result.minus[i].is_nan());
		}
		Ok(())
	}

	fn check_dm_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high_values = [100.0, 110.0, 120.0];
		let low_values = [90.0, 100.0, 110.0];
		let params = DmParams { period: Some(0) };
		let input = DmInput::from_slices(&high_values, &low_values, params);
		let result = dm_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}

	fn check_dm_period_exceeds_data_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high_values = [100.0, 110.0, 120.0];
		let low_values = [90.0, 100.0, 110.0];
		let params = DmParams { period: Some(10) };
		let input = DmInput::from_slices(&high_values, &low_values, params);
		let result = dm_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}

	fn check_dm_not_enough_valid_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high_values = [f64::NAN, f64::NAN, 100.0, 101.0, 102.0];
		let low_values = [f64::NAN, f64::NAN, 90.0, 89.0, 88.0];
		let params = DmParams { period: Some(5) };
		let input = DmInput::from_slices(&high_values, &low_values, params);
		let result = dm_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}

	fn check_dm_all_values_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high_values = [f64::NAN, f64::NAN, f64::NAN];
		let low_values = [f64::NAN, f64::NAN, f64::NAN];
		let params = DmParams { period: Some(3) };
		let input = DmInput::from_slices(&high_values, &low_values, params);
		let result = dm_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}

	fn check_dm_with_slice_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high_values = [9000.0, 9100.0, 9050.0, 9200.0, 9150.0, 9300.0];
		let low_values = [8900.0, 9000.0, 8950.0, 9000.0, 9050.0, 9100.0];
		let params = DmParams { period: Some(2) };
		let input_first = DmInput::from_slices(&high_values, &low_values, params.clone());
		let result_first = dm_with_kernel(&input_first, kernel)?;
		let input_second = DmInput::from_slices(&result_first.plus, &result_first.minus, params);
		let result_second = dm_with_kernel(&input_second, kernel)?;
		assert_eq!(result_second.plus.len(), high_values.len());
		assert_eq!(result_second.minus.len(), high_values.len());
		Ok(())
	}

	fn check_dm_known_values(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = DmParams { period: Some(14) };
		let input = DmInput::from_candles(&candles, params);
		let output = dm_with_kernel(&input, kernel)?;

		let slice_size = 5;
		let last_plus_slice = &output.plus[output.plus.len() - slice_size..];
		let last_minus_slice = &output.minus[output.minus.len() - slice_size..];

		let expected_plus = [
			1410.819956368491,
			1384.04710234217,
			1285.186595032015,
			1199.3875525297283,
			1113.7170130633192,
		];
		let expected_minus = [
			3602.8631384045057,
			3345.5157713756125,
			3258.5503591344973,
			3025.796762053462,
			3493.668421906786,
		];

		for i in 0..slice_size {
			let diff_plus = (last_plus_slice[i] - expected_plus[i]).abs();
			let diff_minus = (last_minus_slice[i] - expected_minus[i]).abs();
			assert!(diff_plus < 1e-6);
			assert!(diff_minus < 1e-6);
		}
		Ok(())
	}

	macro_rules! generate_all_dm_tests {
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

	#[cfg(debug_assertions)]
	fn check_dm_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		// Define comprehensive parameter combinations
		let test_params = vec![
			// Default parameters
			DmParams::default(),
			// Minimum viable period
			DmParams { period: Some(2) },
			// Small periods
			DmParams { period: Some(3) },
			DmParams { period: Some(5) },
			DmParams { period: Some(7) },
			// Medium periods
			DmParams { period: Some(10) },
			DmParams { period: Some(14) }, // default value
			DmParams { period: Some(20) },
			DmParams { period: Some(30) },
			// Large periods
			DmParams { period: Some(50) },
			DmParams { period: Some(100) },
			DmParams { period: Some(200) },
			// Edge case close to common usage
			DmParams { period: Some(25) },
		];

		for (param_idx, params) in test_params.iter().enumerate() {
			let input = DmInput::from_candles(&candles, params.clone());
			let output = dm_with_kernel(&input, kernel)?;

			// Check plus array
			for (i, &val) in output.plus.iter().enumerate() {
				if val.is_nan() {
					continue; // NaN values are expected during warmup
				}

				let bits = val.to_bits();

				// Check all three poison patterns
				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in plus array \
						 with params: period={} (param set {})",
						test_name, val, bits, i, 
						params.period.unwrap_or(14), param_idx
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in plus array \
						 with params: period={} (param set {})",
						test_name, val, bits, i, 
						params.period.unwrap_or(14), param_idx
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in plus array \
						 with params: period={} (param set {})",
						test_name, val, bits, i, 
						params.period.unwrap_or(14), param_idx
					);
				}
			}

			// Check minus array
			for (i, &val) in output.minus.iter().enumerate() {
				if val.is_nan() {
					continue; // NaN values are expected during warmup
				}

				let bits = val.to_bits();

				// Check all three poison patterns
				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in minus array \
						 with params: period={} (param set {})",
						test_name, val, bits, i, 
						params.period.unwrap_or(14), param_idx
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in minus array \
						 with params: period={} (param set {})",
						test_name, val, bits, i, 
						params.period.unwrap_or(14), param_idx
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in minus array \
						 with params: period={} (param set {})",
						test_name, val, bits, i, 
						params.period.unwrap_or(14), param_idx
					);
				}
			}
		}

		Ok(())
	}

	#[cfg(not(debug_assertions))]
	fn check_dm_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		Ok(()) // No-op in release builds
	}

	generate_all_dm_tests!(
		check_dm_partial_params,
		check_dm_default_candles,
		check_dm_with_slice_data,
		check_dm_zero_period,
		check_dm_period_exceeds_data_length,
		check_dm_not_enough_valid_data,
		check_dm_all_values_nan,
		check_dm_with_slice_reinput,
		check_dm_known_values,
		check_dm_no_poison
	);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let output = DmBatchBuilder::new().kernel(kernel).apply_candles(&c)?;

		let def = DmParams::default();
		let (row_plus, row_minus) = output.values_for(&def).expect("default row missing");

		assert_eq!(row_plus.len(), c.high.len());
		assert_eq!(row_minus.len(), c.high.len());

		let expected_plus = [
			1410.819956368491,
			1384.04710234217,
			1285.186595032015,
			1199.3875525297283,
			1113.7170130633192,
		];
		let expected_minus = [
			3602.8631384045057,
			3345.5157713756125,
			3258.5503591344973,
			3025.796762053462,
			3493.668421906786,
		];
		let start = row_plus.len() - 5;
		for (i, &v) in row_plus[start..].iter().enumerate() {
			assert!((v - expected_plus[i]).abs() < 1e-6);
		}
		for (i, &v) in row_minus[start..].iter().enumerate() {
			assert!((v - expected_minus[i]).abs() < 1e-6);
		}
		Ok(())
	}

	#[cfg(debug_assertions)]
	fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		// Test various parameter sweep configurations
		let test_configs = vec![
			// (period_start, period_end, period_step)
			// Small periods
			(2, 10, 2),      // 2, 4, 6, 8, 10
			// Medium periods
			(5, 25, 5),      // 5, 10, 15, 20, 25
			// Large periods
			(30, 60, 15),    // 30, 45, 60
			// Dense small range
			(2, 5, 1),       // 2, 3, 4, 5
			// Single value (no sweep)
			(14, 14, 0),     // Just 14 (default)
			// Wide range
			(10, 100, 10),   // 10, 20, 30, ..., 100
			// Very large periods
			(100, 200, 50),  // 100, 150, 200
		];

		for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
			let output = DmBatchBuilder::new()
				.kernel(kernel)
				.period_range(p_start, p_end, p_step)
				.apply_candles(&c)?;

			// Check plus matrix
			for (idx, &val) in output.plus.iter().enumerate() {
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
						"[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) in plus \
						 at row {} col {} (flat index {}) with params: period={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.period.unwrap_or(14)
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) in plus \
						 at row {} col {} (flat index {}) with params: period={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.period.unwrap_or(14)
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) in plus \
						 at row {} col {} (flat index {}) with params: period={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.period.unwrap_or(14)
					);
				}
			}

			// Check minus matrix
			for (idx, &val) in output.minus.iter().enumerate() {
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
						"[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) in minus \
						 at row {} col {} (flat index {}) with params: period={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.period.unwrap_or(14)
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) in minus \
						 at row {} col {} (flat index {}) with params: period={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.period.unwrap_or(14)
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) in minus \
						 at row {} col {} (flat index {}) with params: period={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.period.unwrap_or(14)
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
