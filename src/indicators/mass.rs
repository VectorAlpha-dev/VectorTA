//! # Mass Index (MASS)
//!
//! The Mass Index is an indicator that uses the ratio of two exponential moving averages
//! (both using period=9) of the range (high - low) and sums these ratios over `period` bars.
//! This implementation follows the Tulip Indicators reference for MASS, with a default period of 5.
//!
//! ## Parameters
//! - **period**: The summation window size. Defaults to 5.
//!
//! ## Errors
//! - **EmptyData**: mass: Input data slices are empty.
//! - **DifferentLengthHL**: mass: `high` and `low` slices have different lengths.
//! - **InvalidPeriod**: mass: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: mass: Fewer than `16 + period - 1` valid (non-`NaN`) data points remain
//!   after the first valid index.
//! - **AllValuesNaN**: mass: All input data values are `NaN`.
//!
//! ## Returns
//! - **`Ok(MassOutput)`** on success, containing a `Vec<f64>` matching the input length,
//!   with leading `NaN`s until enough data is accumulated for the Mass Index calculation.
//! - **`Err(MassError)`** otherwise.

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
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum MassData<'a> {
	Candles {
		candles: &'a Candles,
		high_source: &'a str,
		low_source: &'a str,
	},
	Slices {
		high: &'a [f64],
		low: &'a [f64],
	},
}

#[derive(Debug, Clone)]
pub struct MassOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct MassParams {
	pub period: Option<usize>,
}

impl Default for MassParams {
	fn default() -> Self {
		Self { period: Some(5) }
	}
}

#[derive(Debug, Clone)]
pub struct MassInput<'a> {
	pub data: MassData<'a>,
	pub params: MassParams,
}

impl<'a> MassInput<'a> {
	#[inline]
	pub fn from_candles(candles: &'a Candles, high_source: &'a str, low_source: &'a str, params: MassParams) -> Self {
		Self {
			data: MassData::Candles {
				candles,
				high_source,
				low_source,
			},
			params,
		}
	}

	#[inline]
	pub fn from_slices(high: &'a [f64], low: &'a [f64], params: MassParams) -> Self {
		Self {
			data: MassData::Slices { high, low },
			params,
		}
	}

	#[inline]
	pub fn with_default_candles(candles: &'a Candles) -> Self {
		Self {
			data: MassData::Candles {
				candles,
				high_source: "high",
				low_source: "low",
			},
			params: MassParams::default(),
		}
	}

	#[inline]
	pub fn get_period(&self) -> usize {
		self.params
			.period
			.unwrap_or_else(|| MassParams::default().period.unwrap())
	}
}

#[derive(Debug, Error)]
pub enum MassError {
	#[error("mass: Empty data provided.")]
	EmptyData,
	#[error("mass: High and low slices must have the same length.")]
	DifferentLengthHL,
	#[error("mass: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("mass: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("mass: All values are NaN.")]
	AllValuesNaN,
}

#[derive(Clone, Debug)]
pub struct MassBuilder {
	period: Option<usize>,
	kernel: Kernel,
}

impl Default for MassBuilder {
	fn default() -> Self {
		Self {
			period: None,
			kernel: Kernel::Auto,
		}
	}
}

impl MassBuilder {
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
	pub fn apply(self, c: &Candles) -> Result<MassOutput, MassError> {
		let p = MassParams { period: self.period };
		let i = MassInput::with_default_candles(c);
		mass_with_kernel(&i, self.kernel)
	}

	#[inline(always)]
	pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<MassOutput, MassError> {
		let p = MassParams { period: self.period };
		let i = MassInput::from_slices(high, low, p);
		mass_with_kernel(&i, self.kernel)
	}

	#[inline(always)]
	pub fn into_stream(self) -> Result<MassStream, MassError> {
		let p = MassParams { period: self.period };
		MassStream::try_new(p)
	}
}

#[inline]
pub fn mass(input: &MassInput) -> Result<MassOutput, MassError> {
	mass_with_kernel(input, Kernel::Auto)
}

pub fn mass_with_kernel(input: &MassInput, kernel: Kernel) -> Result<MassOutput, MassError> {
	let (high, low) = match &input.data {
		MassData::Candles {
			candles,
			high_source,
			low_source,
		} => (source_type(candles, high_source), source_type(candles, low_source)),
		MassData::Slices { high, low } => (*high, *low),
	};

	if high.is_empty() || low.is_empty() {
		return Err(MassError::EmptyData);
	}
	if high.len() != low.len() {
		return Err(MassError::DifferentLengthHL);
	}

	let period = input.get_period();
	if period == 0 || period > high.len() {
		return Err(MassError::InvalidPeriod {
			period,
			data_len: high.len(),
		});
	}

	let first_valid_idx = match (0..high.len()).find(|&i| !high[i].is_nan() && !low[i].is_nan()) {
		Some(idx) => idx,
		None => return Err(MassError::AllValuesNaN),
	};

	let needed_bars = 16 + period - 1;
	if (high.len() - first_valid_idx) < needed_bars {
		return Err(MassError::NotEnoughValidData {
			needed: needed_bars,
			valid: high.len() - first_valid_idx,
		});
	}

	let warmup_period = first_valid_idx + 16 + period - 1;
	let mut out = alloc_with_nan_prefix(high.len(), warmup_period);
	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => mass_scalar(high, low, period, first_valid_idx, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => mass_avx2(high, low, period, first_valid_idx, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => mass_avx512(high, low, period, first_valid_idx, &mut out),
			_ => unreachable!(),
		}
	}
	Ok(MassOutput { values: out })
}

#[inline]
pub fn mass_into_slice(dst: &mut [f64], input: &MassInput, kern: Kernel) -> Result<(), MassError> {
	let (high, low) = match &input.data {
		MassData::Candles {
			candles,
			high_source,
			low_source,
		} => (source_type(candles, high_source), source_type(candles, low_source)),
		MassData::Slices { high, low } => (*high, *low),
	};

	if high.is_empty() || low.is_empty() {
		return Err(MassError::EmptyData);
	}
	if high.len() != low.len() {
		return Err(MassError::DifferentLengthHL);
	}
	if dst.len() != high.len() {
		return Err(MassError::InvalidPeriod {
			period: dst.len(),
			data_len: high.len(),
		});
	}

	let period = input.get_period();
	if period == 0 || period > high.len() {
		return Err(MassError::InvalidPeriod {
			period,
			data_len: high.len(),
		});
	}

	let first_valid_idx = match (0..high.len()).find(|&i| !high[i].is_nan() && !low[i].is_nan()) {
		Some(idx) => idx,
		None => return Err(MassError::AllValuesNaN),
	};

	let needed_bars = 16 + period - 1;
	if (high.len() - first_valid_idx) < needed_bars {
		return Err(MassError::NotEnoughValidData {
			needed: needed_bars,
			valid: high.len() - first_valid_idx,
		});
	}

	let chosen = match kern {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => mass_scalar(high, low, period, first_valid_idx, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => mass_avx2(high, low, period, first_valid_idx, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => mass_avx512(high, low, period, first_valid_idx, dst),
			_ => unreachable!(),
		}
	}

	Ok(())
}

#[inline]
pub fn mass_scalar(high: &[f64], low: &[f64], period: usize, first_valid_idx: usize, out: &mut [f64]) {
	let alpha = 2.0 / 10.0;
	let inv_alpha = 1.0 - alpha;
	let mut ema1 = high[first_valid_idx] - low[first_valid_idx];
	let mut ema2 = ema1;
	let mut ring = vec![0.0; period];
	let mut ring_index = 0;
	let mut sum_ratio = 0.0;

	for i in first_valid_idx..high.len() {
		let hl = high[i] - low[i];
		ema1 = ema1.mul_add(inv_alpha, hl * alpha);

		if i == first_valid_idx + 8 {
			ema2 = ema1;
		}

		if i >= first_valid_idx + 8 {
			ema2 = ema2.mul_add(inv_alpha, ema1 * alpha);
		}

		if i >= first_valid_idx + 16 {
			let ratio = ema1 / ema2;
			sum_ratio -= ring[ring_index];
			ring[ring_index] = ratio;
			sum_ratio += ratio;
			ring_index = (ring_index + 1) % period;

			if i >= first_valid_idx + 16 + (period - 1) {
				out[i] = sum_ratio;
			}
		}
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn mass_avx512(high: &[f64], low: &[f64], period: usize, first_valid_idx: usize, out: &mut [f64]) {
	if period <= 32 {
		unsafe { mass_avx512_short(high, low, period, first_valid_idx, out) }
	} else {
		unsafe { mass_avx512_long(high, low, period, first_valid_idx, out) }
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn mass_avx2(high: &[f64], low: &[f64], period: usize, first_valid_idx: usize, out: &mut [f64]) {
	mass_scalar(high, low, period, first_valid_idx, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn mass_avx512_short(high: &[f64], low: &[f64], period: usize, first_valid_idx: usize, out: &mut [f64]) {
	mass_scalar(high, low, period, first_valid_idx, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn mass_avx512_long(high: &[f64], low: &[f64], period: usize, first_valid_idx: usize, out: &mut [f64]) {
	mass_scalar(high, low, period, first_valid_idx, out);
}

#[derive(Debug, Clone)]
pub struct MassStream {
	period: usize,
	ring: Vec<f64>,
	ring_index: usize,
	sum_ratio: f64,
	alpha: f64,
	inv_alpha: f64,
	ema1: f64,
	ema2: f64,
	bar: usize,
	filled: bool,
}

impl MassStream {
	pub fn try_new(params: MassParams) -> Result<Self, MassError> {
		let period = params.period.unwrap_or(5);
		if period == 0 {
			return Err(MassError::InvalidPeriod { period, data_len: 0 });
		}

		Ok(Self {
			period,
			ring: vec![0.0; period],
			ring_index: 0,
			sum_ratio: 0.0,
			alpha: 2.0 / 10.0,
			inv_alpha: 1.0 - (2.0 / 10.0),
			ema1: f64::NAN,
			ema2: f64::NAN,
			bar: 0,
			filled: false,
		})
	}

	#[inline(always)]
	pub fn update(&mut self, high: f64, low: f64) -> Option<f64> {
		let hl = high - low;
		if self.bar == 0 {
			self.ema1 = hl;
			self.ema2 = hl;
			self.bar += 1;
			return None;
		}
		self.ema1 = self.ema1.mul_add(self.inv_alpha, hl * self.alpha);

		if self.bar == 8 {
			self.ema2 = self.ema1;
		}
		if self.bar >= 8 {
			self.ema2 = self.ema2.mul_add(self.inv_alpha, self.ema1 * self.alpha);
		}

		if self.bar >= 16 {
			let ratio = self.ema1 / self.ema2;
			self.sum_ratio -= self.ring[self.ring_index];
			self.ring[self.ring_index] = ratio;
			self.sum_ratio += ratio;
			self.ring_index = (self.ring_index + 1) % self.period;

			if self.bar >= 16 + (self.period - 1) {
				self.bar += 1;
				return Some(self.sum_ratio);
			}
		}
		self.bar += 1;
		None
	}
}

#[derive(Clone, Debug)]
pub struct MassBatchRange {
	pub period: (usize, usize, usize),
}

impl Default for MassBatchRange {
	fn default() -> Self {
		Self { period: (5, 20, 1) }
	}
}

#[derive(Clone, Debug, Default)]
pub struct MassBatchBuilder {
	range: MassBatchRange,
	kernel: Kernel,
}

impl MassBatchBuilder {
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
	pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<MassBatchOutput, MassError> {
		mass_batch_with_kernel(high, low, &self.range, self.kernel)
	}
	pub fn with_default_slices(high: &[f64], low: &[f64], k: Kernel) -> Result<MassBatchOutput, MassError> {
		MassBatchBuilder::new().kernel(k).apply_slices(high, low)
	}
	pub fn apply_candles(self, c: &Candles) -> Result<MassBatchOutput, MassError> {
		let high = source_type(c, "high");
		let low = source_type(c, "low");
		self.apply_slices(high, low)
	}
	pub fn with_default_candles(c: &Candles) -> Result<MassBatchOutput, MassError> {
		MassBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c)
	}
}

pub fn mass_batch_with_kernel(
	high: &[f64],
	low: &[f64],
	sweep: &MassBatchRange,
	k: Kernel,
) -> Result<MassBatchOutput, MassError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(MassError::InvalidPeriod { period: 0, data_len: 0 }),
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	mass_batch_par_slice(high, low, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct MassBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<MassParams>,
	pub rows: usize,
	pub cols: usize,
}

impl MassBatchOutput {
	pub fn row_for_params(&self, p: &MassParams) -> Option<usize> {
		self.combos
			.iter()
			.position(|c| c.period.unwrap_or(5) == p.period.unwrap_or(5))
	}
	pub fn values_for(&self, p: &MassParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct MassBatchConfig {
	pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct MassBatchJsOutput {
	pub values: Vec<f64>,
	pub combos: Vec<MassParams>,
	pub rows: usize,
	pub cols: usize,
}

#[inline(always)]
fn expand_grid_mass(r: &MassBatchRange) -> Vec<MassParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let periods = axis_usize(r.period);
	let mut out = Vec::with_capacity(periods.len());
	for &p in &periods {
		out.push(MassParams { period: Some(p) });
	}
	out
}

#[inline(always)]
pub fn mass_batch_slice(
	high: &[f64],
	low: &[f64],
	sweep: &MassBatchRange,
	kern: Kernel,
) -> Result<MassBatchOutput, MassError> {
	mass_batch_inner(high, low, sweep, kern, false)
}

#[inline(always)]
pub fn mass_batch_par_slice(
	high: &[f64],
	low: &[f64],
	sweep: &MassBatchRange,
	kern: Kernel,
) -> Result<MassBatchOutput, MassError> {
	mass_batch_inner(high, low, sweep, kern, true)
}

#[inline(always)]
fn mass_batch_inner(
	high: &[f64],
	low: &[f64],
	sweep: &MassBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<MassBatchOutput, MassError> {
	let combos = expand_grid_mass(sweep);
	if combos.is_empty() {
		return Err(MassError::InvalidPeriod { period: 0, data_len: 0 });
	}

	if high.is_empty() || low.is_empty() || high.len() != low.len() {
		return Err(MassError::DifferentLengthHL);
	}

	let first = (0..high.len())
		.find(|&i| !high[i].is_nan() && !low[i].is_nan())
		.ok_or(MassError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	let needed_bars = 16 + max_p - 1;
	if high.len() - first < needed_bars {
		return Err(MassError::NotEnoughValidData {
			needed: needed_bars,
			valid: high.len() - first,
		});
	}

	let rows = combos.len();
	let cols = high.len();
	
	// Use uninitialized memory with proper warmup calculation
	let mut buf_mu = make_uninit_matrix(rows, cols);
	
	// Calculate warmup periods for each parameter combination
	let warm: Vec<usize> = combos
		.iter()
		.map(|c| first + 16 + c.period.unwrap() - 1)
		.collect();
	init_matrix_prefixes(&mut buf_mu, cols, &warm);
	
	// Convert to mutable slice for computation
	let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
	let values_slice: &mut [f64] =
		unsafe { core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len()) };

	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let period = combos[row].period.unwrap();
		match kern {
			Kernel::Scalar => mass_row_scalar(high, low, period, first, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => mass_row_avx2(high, low, period, first, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => mass_row_avx512(high, low, period, first, out_row),
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

	// Convert ManuallyDrop buffer to Vec
	let values = unsafe {
		Vec::from_raw_parts(
			buf_guard.as_mut_ptr() as *mut f64,
			buf_guard.len(),
			buf_guard.capacity(),
		)
	};

	Ok(MassBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
unsafe fn mass_row_scalar(high: &[f64], low: &[f64], period: usize, first: usize, out: &mut [f64]) {
	mass_scalar(high, low, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn mass_row_avx2(high: &[f64], low: &[f64], period: usize, first: usize, out: &mut [f64]) {
	mass_scalar(high, low, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn mass_row_avx512(high: &[f64], low: &[f64], period: usize, first: usize, out: &mut [f64]) {
	if period <= 32 {
		mass_row_avx512_short(high, low, period, first, out);
	} else {
		mass_row_avx512_long(high, low, period, first, out);
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn mass_row_avx512_short(high: &[f64], low: &[f64], period: usize, first: usize, out: &mut [f64]) {
	mass_scalar(high, low, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn mass_row_avx512_long(high: &[f64], low: &[f64], period: usize, first: usize, out: &mut [f64]) {
	mass_scalar(high, low, period, first, out);
}

// Tests
#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_mass_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let default_params = MassParams { period: None };
		let input_default = MassInput::from_candles(&candles, "high", "low", default_params);
		let output_default = mass_with_kernel(&input_default, kernel)?;
		assert_eq!(output_default.values.len(), candles.high.len());
		Ok(())
	}

	fn check_mass_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = MassParams { period: Some(5) };
		let input = MassInput::from_candles(&candles, "high", "low", params);
		let mass_result = mass_with_kernel(&input, kernel)?;
		assert_eq!(mass_result.values.len(), candles.high.len(), "MASS length mismatch");
		let expected_last_five = [
			4.512263952194651,
			4.126178935431121,
			3.838738456245828,
			3.6450956734739375,
			3.6748009093527125,
		];
		let result_len = mass_result.values.len();
		assert!(result_len >= 5, "MASS output length is too short for comparison");
		let start_idx = result_len - 5;
		let result_slice = &mass_result.values[start_idx..];
		for (i, &value) in result_slice.iter().enumerate() {
			let expected = expected_last_five[i];
			assert!(
				(value - expected).abs() < 1e-7,
				"MASS mismatch at index {}: expected {}, got {}",
				start_idx + i,
				expected,
				value
			);
		}
		Ok(())
	}

	fn check_mass_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = MassInput::with_default_candles(&candles);
		match input.data {
			MassData::Candles {
				high_source,
				low_source,
				..
			} => {
				assert_eq!(high_source, "high");
				assert_eq!(low_source, "low");
			}
			_ => panic!("Expected MassData::Candles variant"),
		}
		let output = mass_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.high.len());
		Ok(())
	}

	fn check_mass_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high_data = [10.0, 15.0, 20.0];
		let low_data = [5.0, 10.0, 12.0];
		let params = MassParams { period: Some(0) };
		let input = MassInput::from_slices(&high_data, &low_data, params);
		let result = mass_with_kernel(&input, kernel);
		assert!(result.is_err(), "Expected an error for zero period");
		Ok(())
	}

	fn check_mass_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high_data = [10.0, 15.0, 20.0];
		let low_data = [5.0, 10.0, 12.0];
		let params = MassParams { period: Some(10) };
		let input = MassInput::from_slices(&high_data, &low_data, params);
		let result = mass_with_kernel(&input, kernel);
		assert!(result.is_err(), "Expected an error for period > data.len()");
		Ok(())
	}

	fn check_mass_very_small_data_set(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high_data = [10.0];
		let low_data = [5.0];
		let params = MassParams { period: Some(5) };
		let input = MassInput::from_slices(&high_data, &low_data, params);
		let result = mass_with_kernel(&input, kernel);
		assert!(result.is_err(), "Expected error for data smaller than needed bars");
		Ok(())
	}

	fn check_mass_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let first_params = MassParams { period: Some(5) };
		let first_input = MassInput::from_candles(&candles, "high", "low", first_params);
		let first_result = mass_with_kernel(&first_input, kernel)?;
		let second_params = MassParams { period: Some(5) };
		let second_input = MassInput::from_slices(&first_result.values, &first_result.values, second_params);
		let second_result = mass_with_kernel(&second_input, kernel)?;
		assert_eq!(
			second_result.values.len(),
			first_result.values.len(),
			"Second MASS output length mismatch"
		);
		Ok(())
	}

	fn check_mass_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let period = 5;
		let params = MassParams { period: Some(period) };
		let input = MassInput::from_candles(&candles, "high", "low", params);
		let mass_result = mass_with_kernel(&input, kernel)?;
		assert_eq!(mass_result.values.len(), candles.high.len(), "MASS length mismatch");
		if mass_result.values.len() > 240 {
			for i in 240..mass_result.values.len() {
				assert!(
					!mass_result.values[i].is_nan(),
					"Expected no NaN after index 240, but found NaN at index {}",
					i
				);
			}
		}
		Ok(())
	}

	#[cfg(debug_assertions)]
	fn check_mass_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		// Comprehensive parameter combinations for mass
		let test_params = vec![
			MassParams::default(),           // period: 5
			MassParams { period: Some(2) },  // minimum period
			MassParams { period: Some(3) },  // small
			MassParams { period: Some(4) },  // small
			MassParams { period: Some(5) },  // default
			MassParams { period: Some(10) }, // medium
			MassParams { period: Some(20) }, // medium-large
			MassParams { period: Some(30) }, // large
			MassParams { period: Some(50) }, // large
			MassParams { period: Some(100) }, // very large
			MassParams { period: Some(200) }, // very large
			MassParams { period: Some(255) }, // edge case
		];

		for (param_idx, params) in test_params.iter().enumerate() {
			let input = MassInput::from_candles(&candles, "high", "low", params.clone());
			let output = mass_with_kernel(&input, kernel)?;

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
						test_name, val, bits, i, params.period.unwrap_or(5), param_idx
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
						test_name, val, bits, i, params.period.unwrap_or(5), param_idx
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
						test_name, val, bits, i, params.period.unwrap_or(5), param_idx
					);
				}
			}
		}

		Ok(())
	}

	#[cfg(not(debug_assertions))]
	fn check_mass_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		Ok(()) // No-op in release builds
	}

	macro_rules! generate_all_mass_tests {
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

	generate_all_mass_tests!(
		check_mass_partial_params,
		check_mass_accuracy,
		check_mass_default_candles,
		check_mass_zero_period,
		check_mass_period_exceeds_length,
		check_mass_very_small_data_set,
		check_mass_reinput,
		check_mass_nan_handling,
		check_mass_no_poison
	);
	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file)?;
		let output = MassBatchBuilder::new().kernel(kernel).apply_candles(&candles)?;
		let def = MassParams::default();
		let row = output.values_for(&def).expect("default row missing");
		assert_eq!(row.len(), candles.high.len());

		let expected = [
			4.512263952194651,
			4.126178935431121,
			3.838738456245828,
			3.6450956734739375,
			3.6748009093527125,
		];
		let start = row.len().saturating_sub(5);
		for (i, &v) in row[start..].iter().enumerate() {
			assert!(
				(v - expected[i]).abs() < 1e-7,
				"[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
			);
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
			(2, 10, 2),    // Small periods with step 2
			(5, 25, 5),    // Medium periods with step 5
			(30, 60, 15),  // Large periods with step 15
			(2, 5, 1),     // Dense small range
			(10, 10, 0),   // Static period
			(50, 100, 25), // Very large periods
			(3, 15, 3),    // Another small/medium range
			(20, 40, 10),  // Medium range
		];

		for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
			let output = MassBatchBuilder::new()
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
						test, cfg_idx, val, bits, row, col, idx, combo.period.unwrap_or(5)
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}",
						test, cfg_idx, val, bits, row, col, idx, combo.period.unwrap_or(5)
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}",
						test, cfg_idx, val, bits, row, col, idx, combo.period.unwrap_or(5)
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

	// Macro for batch testing like alma.rs
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

// Python bindings
#[cfg(feature = "python")]
#[pyfunction(name = "mass")]
#[pyo3(signature = (high, low, period, kernel=None))]
pub fn mass_py<'py>(
	py: Python<'py>,
	high: PyReadonlyArray1<'py, f64>,
	low: PyReadonlyArray1<'py, f64>,
	period: usize,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
	use numpy::{IntoPyArray, PyArrayMethods};

	let high_slice = high.as_slice()?;
	let low_slice = low.as_slice()?;
	let kern = validate_kernel(kernel, false)?;

	let params = MassParams { period: Some(period) };
	let input = MassInput::from_slices(high_slice, low_slice, params);

	let result_vec: Vec<f64> = py
		.allow_threads(|| mass_with_kernel(&input, kern).map(|o| o.values))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "MassStream")]
pub struct MassStreamPy {
	stream: MassStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl MassStreamPy {
	#[new]
	fn new(period: usize) -> PyResult<Self> {
		let params = MassParams { period: Some(period) };
		let stream = MassStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(MassStreamPy { stream })
	}

	fn update(&mut self, high: f64, low: f64) -> Option<f64> {
		self.stream.update(high, low)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "mass_batch")]
#[pyo3(signature = (high, low, period_range, kernel=None))]
pub fn mass_batch_py<'py>(
	py: Python<'py>,
	high: PyReadonlyArray1<'py, f64>,
	low: PyReadonlyArray1<'py, f64>,
	period_range: (usize, usize, usize),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;

	let high_slice = high.as_slice()?;
	let low_slice = low.as_slice()?;

	let sweep = MassBatchRange { period: period_range };

	let combos = expand_grid_mass(&sweep);
	let rows = combos.len();
	let cols = high_slice.len();

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
			mass_batch_inner_into(high_slice, low_slice, &sweep, simd, true, slice_out)
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

#[cfg(any(feature = "python", feature = "wasm"))]
#[inline(always)]
fn mass_batch_inner_into(
	high: &[f64],
	low: &[f64],
	sweep: &MassBatchRange,
	kern: Kernel,
	parallel: bool,
	out: &mut [f64],
) -> Result<Vec<MassParams>, MassError> {
	let combos = expand_grid_mass(sweep);
	if combos.is_empty() {
		return Err(MassError::InvalidPeriod { period: 0, data_len: 0 });
	}

	if high.is_empty() || low.is_empty() || high.len() != low.len() {
		return Err(MassError::DifferentLengthHL);
	}

	let first = (0..high.len())
		.find(|&i| !high[i].is_nan() && !low[i].is_nan())
		.ok_or(MassError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	let needed_bars = 16 + max_p - 1;
	if high.len() - first < needed_bars {
		return Err(MassError::NotEnoughValidData {
			needed: needed_bars,
			valid: high.len() - first,
		});
	}

	let cols = high.len();

	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let period = combos[row].period.unwrap();
		match kern {
			Kernel::Scalar => mass_row_scalar(high, low, period, first, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => mass_row_avx2(high, low, period, first, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => mass_row_avx512(high, low, period, first, out_row),
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

// WASM bindings
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn mass_js(high: &[f64], low: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
	let params = MassParams { period: Some(period) };
	let input = MassInput::from_slices(high, low, params);

	let mut output = vec![0.0; high.len()];

	mass_into_slice(&mut output, &input, Kernel::Auto).map_err(|e| JsValue::from_str(&e.to_string()))?;

	Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn mass_into(
	high_ptr: *const f64,
	low_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period: usize,
) -> Result<(), JsValue> {
	if high_ptr.is_null() || low_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("null pointer passed to mass_into"));
	}

	unsafe {
		let high = std::slice::from_raw_parts(high_ptr, len);
		let low = std::slice::from_raw_parts(low_ptr, len);

		if period == 0 || period > len {
			return Err(JsValue::from_str("Invalid period"));
		}

		let params = MassParams { period: Some(period) };
		let input = MassInput::from_slices(high, low, params);

		// Check for aliasing with either input
		if high_ptr == out_ptr || low_ptr == out_ptr {
			let mut temp = vec![0.0; len];
			mass_into_slice(&mut temp, &input, Kernel::Auto).map_err(|e| JsValue::from_str(&e.to_string()))?;
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			out.copy_from_slice(&temp);
		} else {
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			mass_into_slice(out, &input, Kernel::Auto).map_err(|e| JsValue::from_str(&e.to_string()))?;
		}

		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn mass_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn mass_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe {
			let _ = Vec::from_raw_parts(ptr, len, len);
		}
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = mass_batch)]
pub fn mass_batch_unified_js(high: &[f64], low: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	let config: MassBatchConfig =
		serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

	let sweep = MassBatchRange {
		period: config.period_range,
	};

	let output = mass_batch_inner(high, low, &sweep, Kernel::Auto, false).map_err(|e| JsValue::from_str(&e.to_string()))?;

	let js_output = MassBatchJsOutput {
		values: output.values,
		combos: output.combos,
		rows: output.rows,
		cols: output.cols,
	};

	serde_wasm_bindgen::to_value(&js_output).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn mass_batch_into(
	high_ptr: *const f64,
	low_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period_start: usize,
	period_end: usize,
	period_step: usize,
) -> Result<usize, JsValue> {
	if high_ptr.is_null() || low_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("null pointer passed to mass_batch_into"));
	}

	unsafe {
		let high = std::slice::from_raw_parts(high_ptr, len);
		let low = std::slice::from_raw_parts(low_ptr, len);

		let sweep = MassBatchRange {
			period: (period_start, period_end, period_step),
		};

		let combos = expand_grid_mass(&sweep);
		let rows = combos.len();
		let cols = len;

		let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);

		mass_batch_inner_into(high, low, &sweep, Kernel::Auto, false, out).map_err(|e| JsValue::from_str(&e.to_string()))?;

		Ok(rows)
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct MassStreamWasm {
	stream: MassStream,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl MassStreamWasm {
	#[wasm_bindgen(constructor)]
	pub fn new(period: usize) -> Result<MassStreamWasm, JsValue> {
		let params = MassParams { period: Some(period) };
		let stream = MassStream::try_new(params).map_err(|e| JsValue::from_str(&e.to_string()))?;
		Ok(MassStreamWasm { stream })
	}

	pub fn update(&mut self, high: f64, low: f64) -> Option<f64> {
		self.stream.update(high, low)
	}
}
