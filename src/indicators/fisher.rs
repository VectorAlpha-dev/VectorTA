//! # Fisher Transform
//!
//! The Fisher Transform identifies potential price reversals by normalizing price extremes
//! using a Fisher Transform function. Takes `period` as parameter.
//!
//! ## Parameters
//! - **period**: Window size (number of data points).
//!
//! ## Errors
//! - **EmptyData**: fisher: Input data is empty.
//! - **InvalidPeriod**: fisher: `period` is zero or exceeds data length.
//! - **NotEnoughValidData**: fisher: Not enough valid (non-NaN) data points after the first valid index.
//! - **AllValuesNaN**: fisher: All input data values are `NaN`.
//!
//! ## Returns
//! - **`Ok(FisherOutput)`** on success, containing `fisher` and `signal` vectors matching input length.
//! - **`Err(FisherError)`** otherwise.

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

impl<'a> FisherInput<'a> {
	#[inline(always)]
	pub fn as_ref(&self) -> (&'a [f64], &'a [f64]) {
		match &self.data {
			FisherData::Candles { candles } => (source_type(candles, "high"), source_type(candles, "low")),
			FisherData::Slices { high, low } => (*high, *low),
		}
	}
}

#[derive(Debug, Clone)]
pub enum FisherData<'a> {
	Candles { candles: &'a Candles },
	Slices { high: &'a [f64], low: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct FisherOutput {
	pub fisher: Vec<f64>,
	pub signal: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct FisherParams {
	pub period: Option<usize>,
}

impl Default for FisherParams {
	fn default() -> Self {
		Self { period: Some(9) }
	}
}

#[derive(Debug, Clone)]
pub struct FisherInput<'a> {
	pub data: FisherData<'a>,
	pub params: FisherParams,
}

impl<'a> FisherInput<'a> {
	#[inline]
	pub fn from_candles(candles: &'a Candles, params: FisherParams) -> Self {
		Self {
			data: FisherData::Candles { candles },
			params,
		}
	}

	#[inline(always)]
	pub fn get_high_low(&self) -> (&'a [f64], &'a [f64]) {
		match &self.data {
			FisherData::Candles { candles } => {
				let high = source_type(candles, "high");
				let low = source_type(candles, "low");
				(high, low)
			}
			FisherData::Slices { high, low } => (*high, *low),
		}
	}
	#[inline]
	pub fn from_slices(high: &'a [f64], low: &'a [f64], params: FisherParams) -> Self {
		Self {
			data: FisherData::Slices { high, low },
			params,
		}
	}
	#[inline]
	pub fn with_default_candles(candles: &'a Candles) -> Self {
		Self::from_candles(candles, FisherParams::default())
	}
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(9)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct FisherBuilder {
	period: Option<usize>,
	kernel: Kernel,
}

impl Default for FisherBuilder {
	fn default() -> Self {
		Self {
			period: None,
			kernel: Kernel::Auto,
		}
	}
}

impl FisherBuilder {
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
	pub fn apply(self, c: &Candles) -> Result<FisherOutput, FisherError> {
		let p = FisherParams { period: self.period };
		let i = FisherInput::from_candles(c, p);
		fisher_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<FisherOutput, FisherError> {
		let p = FisherParams { period: self.period };
		let i = FisherInput::from_slices(high, low, p);
		fisher_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<FisherStream, FisherError> {
		let p = FisherParams { period: self.period };
		FisherStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum FisherError {
	#[error("fisher: Empty data provided.")]
	EmptyData,
	#[error("fisher: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("fisher: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("fisher: All values are NaN.")]
	AllValuesNaN,
}

#[inline]
pub fn fisher(input: &FisherInput) -> Result<FisherOutput, FisherError> {
	fisher_with_kernel(input, Kernel::Auto)
}

pub fn fisher_with_kernel(input: &FisherInput, kernel: Kernel) -> Result<FisherOutput, FisherError> {
	let (high, low) = input.get_high_low();

	if high.is_empty() || low.is_empty() {
		return Err(FisherError::EmptyData);
	}

	let period = input.get_period();
	let data_len = high.len().min(low.len());
	if period == 0 || period > data_len {
		return Err(FisherError::InvalidPeriod { period, data_len });
	}

	let mut merged = vec![f64::NAN; data_len];
	for i in 0..data_len {
		merged[i] = 0.5 * (high[i] + low[i]);
	}

	let first = merged
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(FisherError::AllValuesNaN)?;
	if (data_len - first) < period {
		return Err(FisherError::NotEnoughValidData {
			needed: period,
			valid: data_len - first,
		});
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => fisher_scalar(&merged, period, first, data_len),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => fisher_avx2(&merged, period, first, data_len),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => fisher_avx512(&merged, period, first, data_len),
			_ => unreachable!(),
		}
	}
}

#[inline]
pub fn fisher_scalar(
	merged: &[f64],
	period: usize,
	first: usize,
	data_len: usize,
) -> Result<FisherOutput, FisherError> {
	let mut fisher_vals = vec![f64::NAN; data_len];
	let mut signal_vals = vec![f64::NAN; data_len];
	let mut prev_fish = 0.0;
	let mut val1 = 0.0;

	for i in first..data_len {
		if i < first + period - 1 {
			continue;
		}
		let start = i + 1 - period;
		let window = &merged[start..=i];
		let (mut min_val, mut max_val) = (f64::MAX, f64::MIN);
		for &v in window {
			if v < min_val {
				min_val = v;
			}
			if v > max_val {
				max_val = v;
			}
		}
		let range = (max_val - min_val).max(0.001);
		let current_hl = merged[i];
		val1 = 0.33 * 2.0 * ((current_hl - min_val) / range - 0.5) + 0.67 * val1;
		if val1 > 0.99 {
			val1 = 0.999;
		} else if val1 < -0.99 {
			val1 = -0.999;
		}
		signal_vals[i] = prev_fish;
		let new_fish = 0.5 * ((1.0 + val1) / (1.0 - val1)).ln() + 0.5 * prev_fish;
		fisher_vals[i] = new_fish;
		prev_fish = new_fish;
	}

	Ok(FisherOutput {
		fisher: fisher_vals,
		signal: signal_vals,
	})
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn fisher_avx512(
	merged: &[f64],
	period: usize,
	first: usize,
	data_len: usize,
) -> Result<FisherOutput, FisherError> {
	// AVX512 stub: fallback to scalar
	fisher_scalar(merged, period, first, data_len)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn fisher_avx2(merged: &[f64], period: usize, first: usize, data_len: usize) -> Result<FisherOutput, FisherError> {
	// AVX2 stub: fallback to scalar
	fisher_scalar(merged, period, first, data_len)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn fisher_avx512_short(
	merged: &[f64],
	period: usize,
	first: usize,
	data_len: usize,
) -> Result<FisherOutput, FisherError> {
	fisher_avx512(merged, period, first, data_len)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn fisher_avx512_long(
	merged: &[f64],
	period: usize,
	first: usize,
	data_len: usize,
) -> Result<FisherOutput, FisherError> {
	fisher_avx512(merged, period, first, data_len)
}

#[inline]
pub fn fisher_batch_with_kernel(
	high: &[f64],
	low: &[f64],
	sweep: &FisherBatchRange,
	k: Kernel,
) -> Result<FisherBatchOutput, FisherError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(FisherError::InvalidPeriod { period: 0, data_len: 0 }),
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	fisher_batch_par_slice(high, low, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct FisherBatchRange {
	pub period: (usize, usize, usize),
}

impl Default for FisherBatchRange {
	fn default() -> Self {
		Self { period: (9, 240, 1) }
	}
}

#[derive(Clone, Debug, Default)]
pub struct FisherBatchBuilder {
	range: FisherBatchRange,
	kernel: Kernel,
}

impl FisherBatchBuilder {
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
	pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<FisherBatchOutput, FisherError> {
		fisher_batch_with_kernel(high, low, &self.range, self.kernel)
	}
	pub fn with_default_slices(high: &[f64], low: &[f64], k: Kernel) -> Result<FisherBatchOutput, FisherError> {
		FisherBatchBuilder::new().kernel(k).apply_slices(high, low)
	}
	pub fn apply_candles(self, c: &Candles) -> Result<FisherBatchOutput, FisherError> {
		let high = source_type(c, "high");
		let low = source_type(c, "low");
		self.apply_slices(high, low)
	}
	pub fn with_default_candles(c: &Candles) -> Result<FisherBatchOutput, FisherError> {
		FisherBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c)
	}
}

#[derive(Clone, Debug)]
pub struct FisherBatchOutput {
	pub fisher: Vec<f64>,
	pub signal: Vec<f64>,
	pub combos: Vec<FisherParams>,
	pub rows: usize,
	pub cols: usize,
}
impl FisherBatchOutput {
	pub fn row_for_params(&self, p: &FisherParams) -> Option<usize> {
		self.combos
			.iter()
			.position(|c| c.period.unwrap_or(9) == p.period.unwrap_or(9))
	}
	pub fn fisher_for(&self, p: &FisherParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.fisher[start..start + self.cols]
		})
	}
	pub fn signal_for(&self, p: &FisherParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.signal[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &FisherBatchRange) -> Vec<FisherParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let periods = axis_usize(r.period);
	let mut out = Vec::with_capacity(periods.len());
	for &p in &periods {
		out.push(FisherParams { period: Some(p) });
	}
	out
}

#[inline(always)]
pub fn fisher_batch_slice(
	high: &[f64],
	low: &[f64],
	sweep: &FisherBatchRange,
	kern: Kernel,
) -> Result<FisherBatchOutput, FisherError> {
	fisher_batch_inner(high, low, sweep, kern, false)
}
#[inline(always)]
pub fn fisher_batch_par_slice(
	high: &[f64],
	low: &[f64],
	sweep: &FisherBatchRange,
	kern: Kernel,
) -> Result<FisherBatchOutput, FisherError> {
	fisher_batch_inner(high, low, sweep, kern, true)
}

#[inline(always)]
fn fisher_batch_inner(
	high: &[f64],
	low: &[f64],
	sweep: &FisherBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<FisherBatchOutput, FisherError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(FisherError::InvalidPeriod { period: 0, data_len: 0 });
	}

	let data_len = high.len().min(low.len());
	let mut merged = vec![f64::NAN; data_len];
	for i in 0..data_len {
		merged[i] = 0.5 * (high[i] + low[i]);
	}
	let first = merged
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(FisherError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if data_len - first < max_p {
		return Err(FisherError::NotEnoughValidData {
			needed: max_p,
			valid: data_len - first,
		});
	}
	let rows = combos.len();
	let cols = data_len;

	let mut fisher = vec![f64::NAN; rows * cols];
	let mut signal = vec![f64::NAN; rows * cols];
	let do_row = |row: usize, out_fish: &mut [f64], out_signal: &mut [f64]| unsafe {
		let period = combos[row].period.unwrap();
		fisher_row_scalar(&merged, first, period, out_fish, out_signal)
	};

	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			fisher
				.par_chunks_mut(cols)
				.zip(signal.par_chunks_mut(cols))
				.enumerate()
				.for_each(|(row, (fish, sig))| do_row(row, fish, sig));
		}

		#[cfg(target_arch = "wasm32")]
		{
			for (row, (fish, sig)) in fisher.chunks_mut(cols).zip(signal.chunks_mut(cols)).enumerate() {
				do_row(row, fish, sig);
			}
		}
	} else {
		for (row, (fish, sig)) in fisher.chunks_mut(cols).zip(signal.chunks_mut(cols)).enumerate() {
			do_row(row, fish, sig);
		}
	}
	Ok(FisherBatchOutput {
		fisher,
		signal,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
unsafe fn fisher_row_scalar(merged: &[f64], first: usize, period: usize, out_fish: &mut [f64], out_signal: &mut [f64]) {
	let data_len = merged.len();
	let mut prev_fish = 0.0;
	let mut val1 = 0.0;
	for i in first..data_len {
		if i < first + period - 1 {
			continue;
		}
		let start = i + 1 - period;
		let window = &merged[start..=i];
		let (mut min_val, mut max_val) = (f64::MAX, f64::MIN);
		for &v in window {
			if v < min_val {
				min_val = v;
			}
			if v > max_val {
				max_val = v;
			}
		}
		let range = (max_val - min_val).max(0.001);
		let current_hl = merged[i];
		val1 = 0.33 * 2.0 * ((current_hl - min_val) / range - 0.5) + 0.67 * val1;
		if val1 > 0.99 {
			val1 = 0.999;
		} else if val1 < -0.99 {
			val1 = -0.999;
		}
		out_signal[i] = prev_fish;
		let new_fish = 0.5 * ((1.0 + val1) / (1.0 - val1)).ln() + 0.5 * prev_fish;
		out_fish[i] = new_fish;
		prev_fish = new_fish;
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn fisher_row_avx2(merged: &[f64], first: usize, period: usize, out_fish: &mut [f64], out_signal: &mut [f64]) {
	unsafe { fisher_row_scalar(merged, first, period, out_fish, out_signal) }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn fisher_row_avx512(merged: &[f64], first: usize, period: usize, out_fish: &mut [f64], out_signal: &mut [f64]) {
	unsafe { fisher_row_scalar(merged, first, period, out_fish, out_signal) }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn fisher_row_avx512_short(
	merged: &[f64],
	first: usize,
	period: usize,
	out_fish: &mut [f64],
	out_signal: &mut [f64],
) {
	fisher_row_avx512(merged, first, period, out_fish, out_signal)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn fisher_row_avx512_long(
	merged: &[f64],
	first: usize,
	period: usize,
	out_fish: &mut [f64],
	out_signal: &mut [f64],
) {
	fisher_row_avx512(merged, first, period, out_fish, out_signal)
}

#[derive(Debug, Clone)]
pub struct FisherStream {
	period: usize,
	buffer: Vec<f64>,
	head: usize,
	filled: bool,
	prev_fish: f64,
	val1: f64,
}
impl FisherStream {
	pub fn try_new(params: FisherParams) -> Result<Self, FisherError> {
		let period = params.period.unwrap_or(9);
		if period == 0 {
			return Err(FisherError::InvalidPeriod { period, data_len: 0 });
		}
		Ok(Self {
			period,
			buffer: vec![f64::NAN; period],
			head: 0,
			filled: false,
			prev_fish: 0.0,
			val1: 0.0,
		})
	}
	#[inline(always)]
	pub fn update(&mut self, high: f64, low: f64) -> Option<(f64, f64)> {
		let merged = 0.5 * (high + low);
		self.buffer[self.head] = merged;
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
	fn dot_ring(&mut self) -> (f64, f64) {
		let mut min_val = f64::MAX;
		let mut max_val = f64::MIN;
		let mut idx = self.head;
		for _ in 0..self.period {
			let v = self.buffer[idx];
			if v < min_val {
				min_val = v;
			}
			if v > max_val {
				max_val = v;
			}
			idx = (idx + 1) % self.period;
		}
		let range = (max_val - min_val).max(0.001);
		let current_hl = self.buffer[(self.head + self.period - 1) % self.period];
		self.val1 = 0.33 * 2.0 * ((current_hl - min_val) / range - 0.5) + 0.67 * self.val1;
		if self.val1 > 0.99 {
			self.val1 = 0.999;
		} else if self.val1 < -0.99 {
			self.val1 = -0.999;
		}
		let new_signal = self.prev_fish;
		let new_fish = 0.5 * ((1.0 + self.val1) / (1.0 - self.val1)).ln() + 0.5 * self.prev_fish;
		self.prev_fish = new_fish;
		(new_fish, new_signal)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_fisher_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let default_params = FisherParams { period: None };
		let input = FisherInput::from_candles(&candles, default_params);
		let output = fisher_with_kernel(&input, kernel)?;
		assert_eq!(output.fisher.len(), candles.close.len());
		Ok(())
	}

	fn check_fisher_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = FisherInput::from_candles(&candles, FisherParams::default());
		let result = fisher_with_kernel(&input, kernel)?;
		let expected_last_five_fisher = [
			-0.4720164683904261,
			-0.23467530106650444,
			-0.14879388501136784,
			-0.026651419122953053,
			-0.2569225042442664,
		];
		let start = result.fisher.len().saturating_sub(5);
		for (i, &val) in result.fisher[start..].iter().enumerate() {
			let diff = (val - expected_last_five_fisher[i]).abs();
			assert!(
				diff < 1e-1,
				"[{}] Fisher {:?} mismatch at idx {}: got {}, expected {}",
				test_name,
				kernel,
				i,
				val,
				expected_last_five_fisher[i]
			);
		}
		Ok(())
	}

	fn check_fisher_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [10.0, 20.0, 30.0];
		let low = [5.0, 15.0, 25.0];
		let params = FisherParams { period: Some(0) };
		let input = FisherInput::from_slices(&high, &low, params);
		let res = fisher_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] Fisher should fail with zero period", test_name);
		Ok(())
	}

	fn check_fisher_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [10.0, 20.0, 30.0];
		let low = [5.0, 15.0, 25.0];
		let params = FisherParams { period: Some(10) };
		let input = FisherInput::from_slices(&high, &low, params);
		let res = fisher_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] Fisher should fail with period exceeding length",
			test_name
		);
		Ok(())
	}

	fn check_fisher_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [10.0];
		let low = [5.0];
		let params = FisherParams { period: Some(9) };
		let input = FisherInput::from_slices(&high, &low, params);
		let res = fisher_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] Fisher should fail with insufficient data",
			test_name
		);
		Ok(())
	}

	fn check_fisher_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0];
		let low = [5.0, 7.0, 9.0, 10.0, 13.0, 15.0];
		let first_params = FisherParams { period: Some(3) };
		let first_input = FisherInput::from_slices(&high, &low, first_params);
		let first_result = fisher_with_kernel(&first_input, kernel)?;
		let second_params = FisherParams { period: Some(3) };
		let second_input = FisherInput::from_slices(&first_result.fisher, &first_result.signal, second_params);
		let second_result = fisher_with_kernel(&second_input, kernel)?;
		assert_eq!(first_result.fisher.len(), second_result.fisher.len());
		assert_eq!(first_result.signal.len(), second_result.signal.len());
		Ok(())
	}

	fn check_fisher_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = FisherInput::from_candles(&candles, FisherParams::default());
		let res = fisher_with_kernel(&input, kernel)?;
		assert_eq!(res.fisher.len(), candles.close.len());
		if res.fisher.len() > 240 {
			for (i, &val) in res.fisher[240..].iter().enumerate() {
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

	fn check_fisher_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let period = 9;
		let input = FisherInput::from_candles(&candles, FisherParams { period: Some(period) });
		let batch_output = fisher_with_kernel(&input, kernel)?.fisher;

		let highs = source_type(&candles, "high");
		let lows = source_type(&candles, "low");

		let mut stream = FisherStream::try_new(FisherParams { period: Some(period) })?;
		let mut stream_fisher = Vec::with_capacity(highs.len());
		for (&h, &l) in highs.iter().zip(lows.iter()) {
			match stream.update(h, l) {
				Some((fish, _sig)) => stream_fisher.push(fish),
				None => stream_fisher.push(f64::NAN),
			}
		}

		assert_eq!(batch_output.len(), stream_fisher.len());
		for (i, (&b, &s)) in batch_output.iter().zip(stream_fisher.iter()).enumerate() {
			if b.is_nan() && s.is_nan() {
				continue;
			}
			let diff = (b - s).abs();
			assert!(
				diff < 1e-9,
				"[{}] Fisher streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
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
	fn check_fisher_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		// Define comprehensive parameter combinations
		let test_params = vec![
			FisherParams::default(), // period: 9
			FisherParams { period: Some(1) }, // minimum period
			FisherParams { period: Some(2) }, // very small period
			FisherParams { period: Some(3) }, // small period
			FisherParams { period: Some(5) }, // small period
			FisherParams { period: Some(10) }, // medium period
			FisherParams { period: Some(20) }, // medium period
			FisherParams { period: Some(30) }, // medium-large period
			FisherParams { period: Some(50) }, // large period
			FisherParams { period: Some(100) }, // very large period
			FisherParams { period: Some(200) }, // extra large period
			FisherParams { period: Some(240) }, // edge case from default batch range
		];

		for (param_idx, params) in test_params.iter().enumerate() {
			let input = FisherInput::from_candles(&candles, params.clone());
			let output = fisher_with_kernel(&input, kernel)?;

			// Check fisher values
			for (i, &val) in output.fisher.iter().enumerate() {
				if val.is_nan() {
					continue; // NaN values are expected during warmup
				}

				let bits = val.to_bits();

				// Check all three poison patterns
				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 in fisher output with params: period={} (param set {})",
						test_name,
						val,
						bits,
						i,
						params.period.unwrap_or(9),
						param_idx
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 in fisher output with params: period={} (param set {})",
						test_name,
						val,
						bits,
						i,
						params.period.unwrap_or(9),
						param_idx
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 in fisher output with params: period={} (param set {})",
						test_name,
						val,
						bits,
						i,
						params.period.unwrap_or(9),
						param_idx
					);
				}
			}

			// Check signal values
			for (i, &val) in output.signal.iter().enumerate() {
				if val.is_nan() {
					continue; // NaN values are expected during warmup
				}

				let bits = val.to_bits();

				// Check all three poison patterns
				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 in signal output with params: period={} (param set {})",
						test_name,
						val,
						bits,
						i,
						params.period.unwrap_or(9),
						param_idx
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 in signal output with params: period={} (param set {})",
						test_name,
						val,
						bits,
						i,
						params.period.unwrap_or(9),
						param_idx
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 in signal output with params: period={} (param set {})",
						test_name,
						val,
						bits,
						i,
						params.period.unwrap_or(9),
						param_idx
					);
				}
			}
		}

		Ok(())
	}

	#[cfg(not(debug_assertions))]
	fn check_fisher_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(()) // No-op in release builds
	}

	macro_rules! generate_all_fisher_tests {
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

	generate_all_fisher_tests!(
		check_fisher_partial_params,
		check_fisher_accuracy,
		check_fisher_zero_period,
		check_fisher_period_exceeds_length,
		check_fisher_very_small_dataset,
		check_fisher_reinput,
		check_fisher_nan_handling,
		check_fisher_streaming,
		check_fisher_no_poison
	);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let output = FisherBatchBuilder::new().kernel(kernel).apply_candles(&c)?;

		let def = FisherParams::default();
		let row = output.fisher_for(&def).expect("default row missing");

		assert_eq!(row.len(), c.close.len());

		// Optional: If you have reference values for the last few outputs,
		// you could add accuracy checks like ALMA does.
		// For demonstration, let's assume the expected values for the last five Fisher are:
		let expected_last_five = [
			-0.4720164683904261,
			-0.23467530106650444,
			-0.14879388501136784,
			-0.026651419122953053,
			-0.2569225042442664,
		];
		let start = row.len().saturating_sub(5);
		for (i, &val) in row[start..].iter().enumerate() {
			let diff = (val - expected_last_five[i]).abs();
			assert!(
				diff < 1e-1,
				"[{test}] default-row mismatch at idx {i}: {val} vs {expected_last_five:?}"
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
			(1, 10, 1),        // Small periods, every value
			(2, 20, 2),        // Small to medium, even values
			(5, 50, 5),        // Medium range, step 5
			(10, 100, 10),     // Medium to large, step 10
			(20, 240, 20),     // Large range, step 20
			(9, 9, 0),         // Static default period
			(50, 200, 50),     // Large periods only
			(1, 5, 1),         // Very small periods only
			(100, 240, 40),    // Very large periods
			(3, 30, 3),        // Multiples of 3
		];

		for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
			let output = FisherBatchBuilder::new()
				.kernel(kernel)
				.period_range(p_start, p_end, p_step)
				.apply_candles(&c)?;

			// Check fisher values
			for (idx, &val) in output.fisher.iter().enumerate() {
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
						 at row {} col {} (flat index {}) in fisher output with params: period={}",
						test,
						cfg_idx,
						val,
						bits,
						row,
						col,
						idx,
						combo.period.unwrap_or(9)
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in fisher output with params: period={}",
						test,
						cfg_idx,
						val,
						bits,
						row,
						col,
						idx,
						combo.period.unwrap_or(9)
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in fisher output with params: period={}",
						test,
						cfg_idx,
						val,
						bits,
						row,
						col,
						idx,
						combo.period.unwrap_or(9)
					);
				}
			}

			// Check signal values
			for (idx, &val) in output.signal.iter().enumerate() {
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
						 at row {} col {} (flat index {}) in signal output with params: period={}",
						test,
						cfg_idx,
						val,
						bits,
						row,
						col,
						idx,
						combo.period.unwrap_or(9)
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in signal output with params: period={}",
						test,
						cfg_idx,
						val,
						bits,
						row,
						col,
						idx,
						combo.period.unwrap_or(9)
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in signal output with params: period={}",
						test,
						cfg_idx,
						val,
						bits,
						row,
						col,
						idx,
						combo.period.unwrap_or(9)
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
