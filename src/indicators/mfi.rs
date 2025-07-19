//! # Money Flow Index (MFI)
//!
//! MFI is a momentum indicator that measures the inflow and outflow of money into an asset
//! over a specified period. It uses price and volume to identify overbought or oversold
//! conditions by comparing the "typical price" movement and volume flow.
//!
//! ## Parameters
//! - **period**: The window size. Defaults to 14.
//!
//! ## Errors
//! - **EmptyData**: mfi: Input data slices or candle fields are empty.
//! - **InvalidPeriod**: mfi: `period` is zero, or exceeds the data length.
//! - **NotEnoughValidData**: mfi: Fewer than `period` valid (non-`NaN`) data points remain after the first valid index.
//! - **AllValuesNaN**: mfi: All computed typical prices or volumes are `NaN`.
//!
//! ## Returns
//! - **`Ok(MfiOutput)`** on success, containing a `Vec<f64>` matching the input length, with leading `NaN`s until the MFI window is filled.
//! - **`Err(MfiError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum MfiData<'a> {
	Candles {
		candles: &'a Candles,
	},
	Slices {
		high: &'a [f64],
		low: &'a [f64],
		close: &'a [f64],
		volume: &'a [f64],
	},
}

#[derive(Debug, Clone)]
pub struct MfiOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct MfiParams {
	pub period: Option<usize>,
}

impl Default for MfiParams {
	fn default() -> Self {
		Self { period: Some(14) }
	}
}

#[derive(Debug, Clone)]
pub struct MfiInput<'a> {
	pub data: MfiData<'a>,
	pub params: MfiParams,
}

impl<'a> MfiInput<'a> {
	#[inline]
	pub fn from_candles(candles: &'a Candles, params: MfiParams) -> Self {
		Self {
			data: MfiData::Candles { candles },
			params,
		}
	}

	#[inline]
	pub fn from_slices(
		high: &'a [f64],
		low: &'a [f64],
		close: &'a [f64],
		volume: &'a [f64],
		params: MfiParams,
	) -> Self {
		Self {
			data: MfiData::Slices {
				high,
				low,
				close,
				volume,
			},
			params,
		}
	}

	#[inline]
	pub fn with_default_candles(candles: &'a Candles) -> Self {
		Self::from_candles(candles, MfiParams::default())
	}

	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(14)
	}
}

#[derive(Debug, Error)]
pub enum MfiError {
	#[error("mfi: Empty data provided.")]
	EmptyData,
	#[error("mfi: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("mfi: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("mfi: All values are NaN.")]
	AllValuesNaN,
}

#[inline]
pub fn mfi(input: &MfiInput) -> Result<MfiOutput, MfiError> {
	mfi_with_kernel(input, Kernel::Auto)
}

pub fn mfi_with_kernel(input: &MfiInput, kernel: Kernel) -> Result<MfiOutput, MfiError> {
	let (high, low, close, volume): (&[f64], &[f64], &[f64], &[f64]) = match &input.data {
		MfiData::Candles { candles } => (
			candles.high.as_slice(),
			candles.low.as_slice(),
			candles.close.as_slice(),
			candles.volume.as_slice(),
		),
		MfiData::Slices {
			high,
			low,
			close,
			volume,
		} => (*high, *low, *close, *volume),
	};

	let length = high.len();
	if length == 0 || low.len() != length || close.len() != length || volume.len() != length {
		return Err(MfiError::EmptyData);
	}

	let period = input.get_period();
	let first_valid_idx =
		(0..length).find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan() && !volume[i].is_nan());
	let first_valid_idx = match first_valid_idx {
		Some(idx) => idx,
		None => return Err(MfiError::AllValuesNaN),
	};

	if period == 0 || period > length {
		return Err(MfiError::InvalidPeriod {
			period,
			data_len: length,
		});
	}
	if (length - first_valid_idx) < period {
		return Err(MfiError::NotEnoughValidData {
			needed: period,
			valid: length - first_valid_idx,
		});
	}

	let mut out = vec![f64::NAN; length];

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => {
				mfi_scalar(high, low, close, volume, period, first_valid_idx, &mut out)
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => mfi_avx2(high, low, close, volume, period, first_valid_idx, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => mfi_avx512(high, low, close, volume, period, first_valid_idx, &mut out),
			_ => unreachable!(),
		}
	}
	Ok(MfiOutput { values: out })
}

#[inline]
pub unsafe fn mfi_scalar(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	period: usize,
	first: usize,
	out: &mut [f64],
) {
	let len = high.len();
	let mut typical = vec![f64::NAN; len];
	for i in first..len {
		if !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan() {
			typical[i] = (high[i] + low[i] + close[i]) / 3.0;
		}
	}

	let mut pos_buf = vec![0.0; period];
	let mut neg_buf = vec![0.0; period];
	let mut pos_sum = 0.0;
	let mut neg_sum = 0.0;

	let mut prev_typical = typical[first];
	let mut ring_idx = 0;

	for i in (first + 1)..(first + period) {
		let diff = typical[i] - prev_typical;
		prev_typical = typical[i];
		let flow = typical[i] * volume[i];
		if diff > 0.0 {
			pos_buf[ring_idx] = flow;
			neg_buf[ring_idx] = 0.0;
			pos_sum += flow;
		} else if diff < 0.0 {
			neg_buf[ring_idx] = flow;
			pos_buf[ring_idx] = 0.0;
			neg_sum += flow;
		} else {
			pos_buf[ring_idx] = 0.0;
			neg_buf[ring_idx] = 0.0;
		}
		ring_idx = (ring_idx + 1) % period;
	}

	let idx_mfi_start = first + period - 1;
	if idx_mfi_start < len {
		let total = pos_sum + neg_sum;
		out[idx_mfi_start] = if total < 1e-14 { 0.0 } else { 100.0 * (pos_sum / total) };
	}

	for i in (first + period)..len {
		let old_pos = pos_buf[ring_idx];
		let old_neg = neg_buf[ring_idx];
		pos_sum -= old_pos;
		neg_sum -= old_neg;

		let diff = typical[i] - prev_typical;
		prev_typical = typical[i];
		let flow = typical[i] * volume[i];

		if diff > 0.0 {
			pos_buf[ring_idx] = flow;
			neg_buf[ring_idx] = 0.0;
			pos_sum += flow;
		} else if diff < 0.0 {
			neg_buf[ring_idx] = flow;
			pos_buf[ring_idx] = 0.0;
			neg_sum += flow;
		} else {
			pos_buf[ring_idx] = 0.0;
			neg_buf[ring_idx] = 0.0;
		}

		ring_idx = (ring_idx + 1) % period;

		let total = pos_sum + neg_sum;
		out[i] = if total < 1e-14 { 0.0 } else { 100.0 * (pos_sum / total) };
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn mfi_avx2(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	period: usize,
	first: usize,
	out: &mut [f64],
) {
	mfi_scalar(high, low, close, volume, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn mfi_avx512(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	period: usize,
	first: usize,
	out: &mut [f64],
) {
	unsafe {
		if period <= 32 {
			mfi_avx512_short(high, low, close, volume, period, first, out)
		} else {
			mfi_avx512_long(high, low, close, volume, period, first, out)
		}
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn mfi_avx512_short(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	period: usize,
	first: usize,
	out: &mut [f64],
) {
	mfi_scalar(high, low, close, volume, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn mfi_avx512_long(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	period: usize,
	first: usize,
	out: &mut [f64],
) {
	mfi_scalar(high, low, close, volume, period, first, out)
}

#[derive(Copy, Clone, Debug)]
pub struct MfiBuilder {
	period: Option<usize>,
	kernel: Kernel,
}

impl Default for MfiBuilder {
	fn default() -> Self {
		Self {
			period: None,
			kernel: Kernel::Auto,
		}
	}
}

impl MfiBuilder {
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
	pub fn apply(self, c: &Candles) -> Result<MfiOutput, MfiError> {
		let p = MfiParams { period: self.period };
		let i = MfiInput::from_candles(c, p);
		mfi_with_kernel(&i, self.kernel)
	}

	#[inline(always)]
	pub fn apply_slices(self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Result<MfiOutput, MfiError> {
		let p = MfiParams { period: self.period };
		let i = MfiInput::from_slices(high, low, close, volume, p);
		mfi_with_kernel(&i, self.kernel)
	}

	#[inline(always)]
	pub fn into_stream(self) -> Result<MfiStream, MfiError> {
		let p = MfiParams { period: self.period };
		MfiStream::try_new(p)
	}
}

#[derive(Debug, Clone)]
pub struct MfiStream {
	period: usize,
	pos_buf: Vec<f64>,
	neg_buf: Vec<f64>,
	typical_buf: Vec<f64>,
	volume_buf: Vec<f64>,
	head: usize,
	filled: bool,
	pos_sum: f64,
	neg_sum: f64,
	prev_typical: f64,
	index: usize,
}

impl MfiStream {
	pub fn try_new(params: MfiParams) -> Result<Self, MfiError> {
		let period = params.period.unwrap_or(14);
		if period == 0 {
			return Err(MfiError::InvalidPeriod { period, data_len: 0 });
		}
		Ok(Self {
			period,
			pos_buf: vec![0.0; period],
			neg_buf: vec![0.0; period],
			typical_buf: Vec::with_capacity(period),
			volume_buf: Vec::with_capacity(period),
			head: 0,
			filled: false,
			pos_sum: 0.0,
			neg_sum: 0.0,
			prev_typical: f64::NAN,
			index: 0,
		})
	}

	#[inline(always)]
	pub fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> Option<f64> {
		let typical = (high + low + close) / 3.0;
		if self.index == 0 {
			self.prev_typical = typical;
			self.typical_buf.clear();
			self.volume_buf.clear();
			self.index += 1;
			return None;
		}
		let diff = typical - self.prev_typical;
		self.prev_typical = typical;
		let flow = typical * volume;
		if diff > 0.0 {
			self.pos_sum += flow - self.pos_buf[self.head];
			self.neg_sum -= self.neg_buf[self.head];
			self.pos_buf[self.head] = flow;
			self.neg_buf[self.head] = 0.0;
		} else if diff < 0.0 {
			self.neg_sum += flow - self.neg_buf[self.head];
			self.pos_sum -= self.pos_buf[self.head];
			self.neg_buf[self.head] = flow;
			self.pos_buf[self.head] = 0.0;
		} else {
			self.pos_sum -= self.pos_buf[self.head];
			self.neg_sum -= self.neg_buf[self.head];
			self.pos_buf[self.head] = 0.0;
			self.neg_buf[self.head] = 0.0;
		}
		self.head = (self.head + 1) % self.period;
		self.index += 1;
		if !self.filled && self.head == 0 {
			self.filled = true;
		}
		if !self.filled {
			return None;
		}
		let total = self.pos_sum + self.neg_sum;
		Some(if total < 1e-14 {
			0.0
		} else {
			100.0 * (self.pos_sum / total)
		})
	}
}

#[derive(Clone, Debug)]
pub struct MfiBatchRange {
	pub period: (usize, usize, usize),
}

impl Default for MfiBatchRange {
	fn default() -> Self {
		Self { period: (14, 14, 0) }
	}
}

#[derive(Clone, Debug, Default)]
pub struct MfiBatchBuilder {
	range: MfiBatchRange,
	kernel: Kernel,
}

impl MfiBatchBuilder {
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

	pub fn apply_slices(
		self,
		high: &[f64],
		low: &[f64],
		close: &[f64],
		volume: &[f64],
	) -> Result<MfiBatchOutput, MfiError> {
		mfi_batch_with_kernel(high, low, close, volume, &self.range, self.kernel)
	}

	pub fn apply_candles(self, c: &Candles) -> Result<MfiBatchOutput, MfiError> {
		self.apply_slices(&c.high, &c.low, &c.close, &c.volume)
	}

	pub fn with_default_candles(c: &Candles, k: Kernel) -> Result<MfiBatchOutput, MfiError> {
		MfiBatchBuilder::new().kernel(k).apply_candles(c)
	}
}

#[derive(Clone, Debug)]
pub struct MfiBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<MfiParams>,
	pub rows: usize,
	pub cols: usize,
}
impl MfiBatchOutput {
	pub fn row_for_params(&self, p: &MfiParams) -> Option<usize> {
		self.combos
			.iter()
			.position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
	}
	pub fn values_for(&self, p: &MfiParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &MfiBatchRange) -> Vec<MfiParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let periods = axis_usize(r.period);
	let mut out = Vec::with_capacity(periods.len());
	for &p in &periods {
		out.push(MfiParams { period: Some(p) });
	}
	out
}

pub fn mfi_batch_with_kernel(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	sweep: &MfiBatchRange,
	k: Kernel,
) -> Result<MfiBatchOutput, MfiError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(MfiError::InvalidPeriod { period: 0, data_len: 0 }),
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	mfi_batch_par_slice(high, low, close, volume, sweep, simd)
}

#[inline(always)]
pub fn mfi_batch_slice(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	sweep: &MfiBatchRange,
	kern: Kernel,
) -> Result<MfiBatchOutput, MfiError> {
	mfi_batch_inner(high, low, close, volume, sweep, kern, false)
}

#[inline(always)]
pub fn mfi_batch_par_slice(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	sweep: &MfiBatchRange,
	kern: Kernel,
) -> Result<MfiBatchOutput, MfiError> {
	mfi_batch_inner(high, low, close, volume, sweep, kern, true)
}

fn round_up8(x: usize) -> usize {
	(x + 7) & !7
}

#[inline(always)]
fn mfi_batch_inner(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	sweep: &MfiBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<MfiBatchOutput, MfiError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(MfiError::InvalidPeriod { period: 0, data_len: 0 });
	}
	let length = high.len();
	let first = (0..length)
		.find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan() && !volume[i].is_nan())
		.ok_or(MfiError::AllValuesNaN)?;

	let max_p = combos.iter().map(|c| round_up8(c.period.unwrap())).max().unwrap();
	if length - first < max_p {
		return Err(MfiError::NotEnoughValidData {
			needed: max_p,
			valid: length - first,
		});
	}

	let rows = combos.len();
	let cols = length;
	let mut values = vec![f64::NAN; rows * cols];
	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let period = combos[row].period.unwrap();
		match kern {
			Kernel::Scalar => mfi_row_scalar(high, low, close, volume, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => mfi_row_avx2(high, low, close, volume, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => mfi_row_avx512(high, low, close, volume, first, period, out_row),
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

	Ok(MfiBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
unsafe fn mfi_row_scalar(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	first: usize,
	period: usize,
	out: &mut [f64],
) {
	mfi_scalar(high, low, close, volume, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn mfi_row_avx2(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	first: usize,
	period: usize,
	out: &mut [f64],
) {
	mfi_scalar(high, low, close, volume, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn mfi_row_avx512(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	first: usize,
	period: usize,
	out: &mut [f64],
) {
	if period <= 32 {
		mfi_row_avx512_short(high, low, close, volume, first, period, out)
	} else {
		mfi_row_avx512_long(high, low, close, volume, first, period, out)
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn mfi_row_avx512_short(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	first: usize,
	period: usize,
	out: &mut [f64],
) {
	mfi_scalar(high, low, close, volume, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn mfi_row_avx512_long(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	first: usize,
	period: usize,
	out: &mut [f64],
) {
	mfi_scalar(high, low, close, volume, period, first, out)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;
	use paste::paste;

	fn check_mfi_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let default_params = MfiParams { period: None };
		let input = MfiInput::from_candles(&candles, default_params);
		let output = mfi_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_mfi_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = MfiParams { period: Some(14) };
		let input = MfiInput::from_candles(&candles, params);
		let mfi_result = mfi_with_kernel(&input, kernel)?;
		let expected_last_five_mfi = [
			38.13874339324763,
			37.44139770113819,
			31.02039511395131,
			28.092605898618896,
			25.905204729397813,
		];
		let start_index = mfi_result.values.len() - 5;
		for (i, &value) in mfi_result.values[start_index..].iter().enumerate() {
			let expected_value = expected_last_five_mfi[i];
			let diff = (value - expected_value).abs();
			assert!(
				diff < 1e-1,
				"MFI mismatch at index {}: expected {}, got {}",
				i,
				expected_value,
				value
			);
		}
		Ok(())
	}

	fn check_mfi_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = MfiInput::with_default_candles(&candles);
		let output = mfi_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_mfi_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = MfiParams { period: Some(0) };
		let input = MfiInput::from_candles(&candles, params);
		let result = mfi_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}

	fn check_mfi_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let input_high = [1.0, 2.0, 3.0];
		let input_low = [0.5, 1.5, 2.5];
		let input_close = [0.8, 1.8, 2.8];
		let input_volume = [100.0, 200.0, 300.0];
		let params = MfiParams { period: Some(10) };
		let input = MfiInput::from_slices(&input_high, &input_low, &input_close, &input_volume, params);
		let result = mfi_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}

	fn check_mfi_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let input_high = [1.0];
		let input_low = [0.5];
		let input_close = [0.8];
		let input_volume = [100.0];
		let params = MfiParams { period: Some(14) };
		let input = MfiInput::from_slices(&input_high, &input_low, &input_close, &input_volume, params);
		let result = mfi_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}

	fn check_mfi_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let first_params = MfiParams { period: Some(7) };
		let first_input = MfiInput::from_candles(&candles, first_params);
		let first_result = mfi_with_kernel(&first_input, kernel)?;
		let second_params = MfiParams { period: Some(7) };
		let high_values: Vec<f64> = first_result.values.clone();
		let low_values: Vec<f64> = first_result.values.clone();
		let close_values: Vec<f64> = first_result.values.clone();
		let volume_values: Vec<f64> = vec![10_000.0; first_result.values.len()];
		let second_input =
			MfiInput::from_slices(&high_values, &low_values, &close_values, &volume_values, second_params);
		let second_result = mfi_with_kernel(&second_input, kernel)?;
		assert_eq!(second_result.values.len(), first_result.values.len());
		Ok(())
	}

	macro_rules! generate_all_mfi_tests {
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
	generate_all_mfi_tests!(
		check_mfi_partial_params,
		check_mfi_accuracy,
		check_mfi_default_candles,
		check_mfi_zero_period,
		check_mfi_period_exceeds_length,
		check_mfi_very_small_dataset,
		check_mfi_reinput
	);
	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let output = MfiBatchBuilder::new().kernel(kernel).apply_candles(&c)?;

		let def = MfiParams::default();
		let row = output.values_for(&def).expect("default row missing");

		assert_eq!(row.len(), c.close.len());

		// The following block assumes you have a known correct output for your default test file
		// If you want to check real values, insert expected_last_five_mfi as needed:
		let expected = [
			38.13874339324763,
			37.44139770113819,
			31.02039511395131,
			28.092605898618896,
			25.905204729397813,
		];
		let start = row.len().saturating_sub(5);
		for (i, &v) in row[start..].iter().enumerate() {
			assert!(
				(v - expected[i]).abs() < 1e-1,
				"[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
			);
		}
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
}
