//! # Aroon Indicator
//!
//! A trend-following indicator that measures the strength and potential direction of a market trend
//! based on the recent highs and lows over a specified window. Provides two outputs:
//! - **aroon_up**: How close the most recent highest high is to the current bar (percentage).
//! - **aroon_down**: How close the most recent lowest low is to the current bar (percentage).
//!
//! ## Parameters
//! - **length**: Lookback window (default: 14)
//!
//! ## Errors
//! - **AllValuesNaN**: aroon: All input data values are `NaN`.
//! - **InvalidLength**: aroon: `length` is zero or exceeds the data length.
//! - **NotEnoughValidData**: aroon: Not enough valid data points for the requested `length`.
//! - **MismatchSliceLength**: aroon: `high` and `low` slices differ in length.
//!
//! ## Returns
//! - **`Ok(AroonOutput)`** on success, containing vectors for aroon_up and aroon_down.
//! - **`Err(AroonError)`** otherwise.

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};

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
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::mem::ManuallyDrop;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum AroonData<'a> {
	Candles { candles: &'a Candles },
	SlicesHL { high: &'a [f64], low: &'a [f64] },
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct AroonParams {
	pub length: Option<usize>,
}

impl Default for AroonParams {
	fn default() -> Self {
		Self { length: Some(14) }
	}
}

#[derive(Debug, Clone)]
pub struct AroonInput<'a> {
	pub data: AroonData<'a>,
	pub params: AroonParams,
}

impl<'a> AroonInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, p: AroonParams) -> Self {
		Self {
			data: AroonData::Candles { candles: c },
			params: p,
		}
	}
	#[inline]
	pub fn from_slices_hl(high: &'a [f64], low: &'a [f64], p: AroonParams) -> Self {
		Self {
			data: AroonData::SlicesHL { high, low },
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, AroonParams::default())
	}
	#[inline]
	pub fn get_length(&self) -> usize {
		self.params.length.unwrap_or(14)
	}
}

#[derive(Debug, Clone)]
pub struct AroonOutput {
	pub aroon_up: Vec<f64>,
	pub aroon_down: Vec<f64>,
}

#[derive(Copy, Clone, Debug)]
pub struct AroonBuilder {
	length: Option<usize>,
	kernel: Kernel,
}

impl Default for AroonBuilder {
	fn default() -> Self {
		Self {
			length: None,
			kernel: Kernel::Auto,
		}
	}
}
impl AroonBuilder {
	#[inline(always)]
	pub fn new() -> Self {
		Self::default()
	}
	#[inline(always)]
	pub fn length(mut self, n: usize) -> Self {
		self.length = Some(n);
		self
	}
	#[inline(always)]
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline(always)]
	pub fn apply(self, c: &Candles) -> Result<AroonOutput, AroonError> {
		let p = AroonParams { length: self.length };
		let i = AroonInput::from_candles(c, p);
		aroon_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<AroonOutput, AroonError> {
		let p = AroonParams { length: self.length };
		let i = AroonInput::from_slices_hl(high, low, p);
		aroon_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<AroonStream, AroonError> {
		let p = AroonParams { length: self.length };
		AroonStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum AroonError {
	#[error("aroon: All values are NaN.")]
	AllValuesNaN,
	#[error("aroon: Invalid length: length = {length}, data length = {data_len}")]
	InvalidLength { length: usize, data_len: usize },
	#[error("aroon: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("aroon: Mismatch in high/low slice length: high_len={high_len}, low_len={low_len}")]
	MismatchSliceLength { high_len: usize, low_len: usize },
}

#[inline]
pub fn aroon(input: &AroonInput) -> Result<AroonOutput, AroonError> {
	aroon_with_kernel(input, Kernel::Auto)
}

pub fn aroon_with_kernel(input: &AroonInput, kernel: Kernel) -> Result<AroonOutput, AroonError> {
	let (high, low): (&[f64], &[f64]) = match &input.data {
		AroonData::Candles { candles } => (source_type(candles, "high"), source_type(candles, "low")),
		AroonData::SlicesHL { high, low } => (*high, *low),
	};
	if high.len() != low.len() {
		return Err(AroonError::MismatchSliceLength {
			high_len: high.len(),
			low_len: low.len(),
		});
	}
	let len = high.len();
	let length = input.get_length();

	if length == 0 || length > len {
		return Err(AroonError::InvalidLength { length, data_len: len });
	}
	if len < length {
		return Err(AroonError::NotEnoughValidData {
			needed: length,
			valid: len,
		});
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	// Calculate warmup period: Aroon needs 'length' bars before producing valid values
	let warmup_period = length;
	let mut up = alloc_with_nan_prefix(len, warmup_period);
	let mut down = alloc_with_nan_prefix(len, warmup_period);

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => aroon_scalar(high, low, length, &mut up, &mut down),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => aroon_avx2(high, low, length, &mut up, &mut down),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => aroon_avx512(high, low, length, &mut up, &mut down),
			_ => unreachable!(),
		}
	}
	Ok(AroonOutput {
		aroon_up: up,
		aroon_down: down,
	})
}

#[inline]
pub fn aroon_scalar(high: &[f64], low: &[f64], length: usize, up: &mut [f64], down: &mut [f64]) {
	let len = high.len();
	assert!(
		length >= 1 && length <= len,
		"Invalid length: {} for data of size {}",
		length,
		len
	);
	assert!(
		low.len() == len && up.len() == len && down.len() == len,
		"Slice lengths must match"
	);

	let inv_length = 1.0 / (length as f64);

	// Note: The first `length` entries are already filled with NaN by alloc_with_nan_prefix

	// 2) For each bar i from `length` up to `len - 1`, scan a window of size `length + 1`.
	for i in length..len {
		let start = i - length;
		// Initialize with the first bar in [start..=i]
		let mut max_val = high[start];
		let mut min_val = low[start];
		let mut max_idx = start;
		let mut min_idx = start;

		// Find indices of highest high / lowest low in [start..=i]
		for j in (start + 1)..=i {
			let h = high[j];
			if h > max_val {
				max_val = h;
				max_idx = j;
			}
			let l = low[j];
			if l < min_val {
				min_val = l;
				min_idx = j;
			}
		}

		// periods_hi = how many bars ago the highest high was (0..=length)
		let periods_hi = i - max_idx;
		let periods_lo = i - min_idx;

		// Aroon up/down = (length - periods)/length * 100
		up[i] = (length as f64 - periods_hi as f64) * inv_length * 100.0;
		down[i] = (length as f64 - periods_lo as f64) * inv_length * 100.0;
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn aroon_avx512(high: &[f64], low: &[f64], length: usize, up: &mut [f64], down: &mut [f64]) {
	unsafe {
		aroon_scalar(high, low, length, up, down);
	}
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn aroon_avx2(high: &[f64], low: &[f64], length: usize, up: &mut [f64], down: &mut [f64]) {
	unsafe {
		aroon_scalar(high, low, length, up, down);
	}
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn aroon_avx512_short(high: &[f64], low: &[f64], length: usize, up: &mut [f64], down: &mut [f64]) {
	aroon_avx512(high, low, length, up, down)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn aroon_avx512_long(high: &[f64], low: &[f64], length: usize, up: &mut [f64], down: &mut [f64]) {
	aroon_avx512(high, low, length, up, down)
}

#[derive(Debug)]
pub struct AroonStream {
	length: usize,
	buf_size: usize, // = length + 1
	buffer_high: Vec<f64>,
	buffer_low: Vec<f64>,
	head: usize,  // next write position in [0..buf_size)
	count: usize, // how many total bars have been pushed
}

impl AroonStream {
	/// Create a new streaming Aroon from `params`.  Extracts `length = params.length.unwrap_or(14)`.
	/// Fails if `length == 0`.  Allocates two Vecs of size `length + 1`, each pre‐filled with NaN.
	pub fn try_new(params: AroonParams) -> Result<Self, AroonError> {
		let length = params.length.unwrap_or(14);
		if length == 0 {
			return Err(AroonError::InvalidLength { length: 0, data_len: 0 });
		}
		let buf_size = length + 1;
		Ok(AroonStream {
			length,
			buf_size,
			buffer_high: alloc_with_nan_prefix(buf_size, buf_size), // All NaN for circular buffer
			buffer_low: alloc_with_nan_prefix(buf_size, buf_size),  // All NaN for circular buffer
			head: 0,
			count: 0,
		})
	}

	/// Push a new (high, low).  Until we have seen at least `length+1` bars, this returns `None`.
	/// Once `count >= length+1`, each call returns `Some((aroon_up, aroon_down))`.
	#[inline(always)]
	pub fn update(&mut self, high: f64, low: f64) -> Option<(f64, f64)> {
		// 1) Overwrite the “head” slot
		self.buffer_high[self.head] = high;
		self.buffer_low[self.head] = low;

		// 2) Advance head mod buf_size
		self.head = (self.head + 1) % self.buf_size;

		// 3) Increment count until we reach buf_size
		if self.count < self.buf_size {
			self.count += 1;
		}

		// 4) If we haven’t yet filled `length+1` bars, return None
		if self.count < self.buf_size {
			return None;
		}

		// 5) Compute “current index” = the slot we just wrote was (head + buf_size − 1) % buf_size
		let cur_idx = (self.head + self.buf_size - 1) % self.buf_size;

		// 6) Scan exactly the last (length+1) bars in chronological order:
		//    - “oldest_idx” is (cur_idx - length) mod buf_size  ≡  (cur_idx + 1) % buf_size
		let oldest_idx = (cur_idx + 1) % self.buf_size;
		// Initialize to the oldest bar in the window:
		let mut max_idx = oldest_idx;
		let mut min_idx = oldest_idx;
		let mut max_h = self.buffer_high[oldest_idx];
		let mut min_l = self.buffer_low[oldest_idx];
		// Walk forward k = 1..=length (so that:
		//    (oldest_idx + length) % buf_size == cur_idx,
		// covering every bar from oldest → current in order)
		for k in 1..=self.length {
			let idx = (oldest_idx + k) % self.buf_size;
			let hv = self.buffer_high[idx];
			if hv > max_h {
				max_h = hv;
				max_idx = idx;
			}
			let lv = self.buffer_low[idx];
			if lv < min_l {
				min_l = lv;
				min_idx = idx;
			}
		}

		// 7) “Bars ago” for that max:  dist_hi = (cur_idx − max_idx) mod buf_size
		let dist_hi = ((cur_idx as isize - max_idx as isize).rem_euclid(self.buf_size as isize)) as usize;
		let dist_lo = ((cur_idx as isize - min_idx as isize).rem_euclid(self.buf_size as isize)) as usize;

		// 8) Aroon formula: up = (length − dist_hi)/length * 100
		let inv_len = 1.0 / (self.length as f64);
		let up = (self.length as f64 - dist_hi as f64) * inv_len * 100.0;
		let down = (self.length as f64 - dist_lo as f64) * inv_len * 100.0;

		Some((up, down))
	}
}

#[derive(Clone, Debug)]
pub struct AroonBatchRange {
	pub length: (usize, usize, usize),
}
impl Default for AroonBatchRange {
	fn default() -> Self {
		Self { length: (14, 50, 1) }
	}
}

#[derive(Clone, Debug, Default)]
pub struct AroonBatchBuilder {
	range: AroonBatchRange,
	kernel: Kernel,
}
impl AroonBatchBuilder {
	pub fn new() -> Self {
		Self::default()
	}
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline]
	pub fn length_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.length = (start, end, step);
		self
	}
	#[inline]
	pub fn length_static(mut self, x: usize) -> Self {
		self.range.length = (x, x, 0);
		self
	}
	pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<AroonBatchOutput, AroonError> {
		aroon_batch_with_kernel(high, low, &self.range, self.kernel)
	}
	pub fn with_default_slices(high: &[f64], low: &[f64], k: Kernel) -> Result<AroonBatchOutput, AroonError> {
		AroonBatchBuilder::new().kernel(k).apply_slices(high, low)
	}
	pub fn apply_candles(self, c: &Candles) -> Result<AroonBatchOutput, AroonError> {
		self.apply_slices(source_type(c, "high"), source_type(c, "low"))
	}
	pub fn with_default_candles(c: &Candles) -> Result<AroonBatchOutput, AroonError> {
		AroonBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c)
	}
}

pub struct AroonBatchOutput {
	pub up: Vec<f64>,
	pub down: Vec<f64>,
	pub combos: Vec<AroonParams>,
	pub rows: usize,
	pub cols: usize,
}
impl AroonBatchOutput {
	pub fn row_for_params(&self, p: &AroonParams) -> Option<usize> {
		self.combos
			.iter()
			.position(|c| c.length.unwrap_or(14) == p.length.unwrap_or(14))
	}
	pub fn up_for(&self, p: &AroonParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.up[start..start + self.cols]
		})
	}
	pub fn down_for(&self, p: &AroonParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.down[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &AroonBatchRange) -> Vec<AroonParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let lengths = axis_usize(r.length);
	let mut out = Vec::with_capacity(lengths.len());
	for &l in &lengths {
		out.push(AroonParams { length: Some(l) });
	}
	out
}

pub fn aroon_batch_with_kernel(
	high: &[f64],
	low: &[f64],
	sweep: &AroonBatchRange,
	k: Kernel,
) -> Result<AroonBatchOutput, AroonError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(AroonError::InvalidLength { length: 0, data_len: 0 }),
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	aroon_batch_par_slice(high, low, sweep, simd)
}

#[inline(always)]
pub fn aroon_batch_slice(
	high: &[f64],
	low: &[f64],
	sweep: &AroonBatchRange,
	kern: Kernel,
) -> Result<AroonBatchOutput, AroonError> {
	aroon_batch_inner(high, low, sweep, kern, false)
}
#[inline(always)]
pub fn aroon_batch_par_slice(
	high: &[f64],
	low: &[f64],
	sweep: &AroonBatchRange,
	kern: Kernel,
) -> Result<AroonBatchOutput, AroonError> {
	aroon_batch_inner(high, low, sweep, kern, true)
}
#[inline(always)]
fn aroon_batch_inner(
	high: &[f64],
	low: &[f64],
	sweep: &AroonBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<AroonBatchOutput, AroonError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(AroonError::InvalidLength { length: 0, data_len: 0 });
	}
	if high.len() != low.len() {
		return Err(AroonError::MismatchSliceLength {
			high_len: high.len(),
			low_len: low.len(),
		});
	}
	let len = high.len();
	let max_l = combos.iter().map(|c| c.length.unwrap()).max().unwrap();
	if len < max_l {
		return Err(AroonError::NotEnoughValidData {
			needed: max_l,
			valid: len,
		});
	}
	let rows = combos.len();
	let cols = len;

	// Step 1: Allocate uninitialized matrices
	let mut buf_up_mu = make_uninit_matrix(rows, cols);
	let mut buf_down_mu = make_uninit_matrix(rows, cols);

	// Step 2: Calculate warmup periods for each row
	let warmup_periods: Vec<usize> = combos.iter().map(|c| c.length.unwrap()).collect();

	// Step 3: Initialize NaN prefixes for each row
	init_matrix_prefixes(&mut buf_up_mu, cols, &warmup_periods);
	init_matrix_prefixes(&mut buf_down_mu, cols, &warmup_periods);

	// Step 4: Convert to mutable slices for computation
	let mut buf_up_guard = ManuallyDrop::new(buf_up_mu);
	let mut buf_down_guard = ManuallyDrop::new(buf_down_mu);
	let up: &mut [f64] =
		unsafe { core::slice::from_raw_parts_mut(buf_up_guard.as_mut_ptr() as *mut f64, buf_up_guard.len()) };
	let down: &mut [f64] =
		unsafe { core::slice::from_raw_parts_mut(buf_down_guard.as_mut_ptr() as *mut f64, buf_down_guard.len()) };

	let do_row = |row: usize, out_up: &mut [f64], out_down: &mut [f64]| unsafe {
		let length = combos[row].length.unwrap();
		match kern {
			Kernel::Scalar => aroon_row_scalar(high, low, length, out_up, out_down),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => aroon_row_avx2(high, low, length, out_up, out_down),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => aroon_row_avx512(high, low, length, out_up, out_down),
			_ => unreachable!(),
		}
	};
	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			up.par_chunks_mut(cols)
				.zip(down.par_chunks_mut(cols))
				.enumerate()
				.for_each(|(row, (u, d))| do_row(row, u, d));
		}

		#[cfg(target_arch = "wasm32")]
		{
			for (row, (u, d)) in up.chunks_mut(cols).zip(down.chunks_mut(cols)).enumerate() {
				do_row(row, u, d);
			}
		}
	} else {
		for (row, (u, d)) in up.chunks_mut(cols).zip(down.chunks_mut(cols)).enumerate() {
			do_row(row, u, d);
		}
	}
	// Step 6: Reclaim as Vec<f64>
	let up_values = unsafe {
		Vec::from_raw_parts(
			buf_up_guard.as_mut_ptr() as *mut f64,
			buf_up_guard.len(),
			buf_up_guard.capacity(),
		)
	};
	let down_values = unsafe {
		Vec::from_raw_parts(
			buf_down_guard.as_mut_ptr() as *mut f64,
			buf_down_guard.len(),
			buf_down_guard.capacity(),
		)
	};

	Ok(AroonBatchOutput {
		up: up_values,
		down: down_values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
pub unsafe fn aroon_row_scalar(high: &[f64], low: &[f64], length: usize, out_up: &mut [f64], out_down: &mut [f64]) {
	aroon_scalar(high, low, length, out_up, out_down)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn aroon_row_avx2(high: &[f64], low: &[f64], length: usize, out_up: &mut [f64], out_down: &mut [f64]) {
	aroon_row_scalar(high, low, length, out_up, out_down)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn aroon_row_avx512(high: &[f64], low: &[f64], length: usize, out_up: &mut [f64], out_down: &mut [f64]) {
	if length <= 32 {
		aroon_row_avx512_short(high, low, length, out_up, out_down)
	} else {
		aroon_row_avx512_long(high, low, length, out_up, out_down)
	}
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn aroon_row_avx512_short(
	high: &[f64],
	low: &[f64],
	length: usize,
	out_up: &mut [f64],
	out_down: &mut [f64],
) {
	aroon_row_scalar(high, low, length, out_up, out_down)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn aroon_row_avx512_long(
	high: &[f64],
	low: &[f64],
	length: usize,
	out_up: &mut [f64],
	out_down: &mut [f64],
) {
	aroon_row_scalar(high, low, length, out_up, out_down)
}
#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;
	use crate::utilities::enums::Kernel;

	fn check_aroon_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let partial_params = AroonParams { length: None };
		let input = AroonInput::from_candles(&candles, partial_params);
		let result = aroon_with_kernel(&input, kernel)?;
		assert_eq!(result.aroon_up.len(), candles.close.len());
		assert_eq!(result.aroon_down.len(), candles.close.len());
		Ok(())
	}

	fn check_aroon_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = AroonInput::with_default_candles(&candles);
		let result = aroon_with_kernel(&input, kernel)?;

		let expected_up_last_five = [21.43, 14.29, 7.14, 0.0, 0.0];
		let expected_down_last_five = [71.43, 64.29, 57.14, 50.0, 42.86];

		assert!(
			result.aroon_up.len() >= 5 && result.aroon_down.len() >= 5,
			"Not enough Aroon values"
		);

		let start_index = result.aroon_up.len().saturating_sub(5);

		let up_last_five = &result.aroon_up[start_index..];
		let down_last_five = &result.aroon_down[start_index..];

		for (i, &value) in up_last_five.iter().enumerate() {
			assert!(
				(value - expected_up_last_five[i]).abs() < 1e-2,
				"Aroon Up mismatch at index {}: expected {}, got {}",
				i,
				expected_up_last_five[i],
				value
			);
		}

		for (i, &value) in down_last_five.iter().enumerate() {
			assert!(
				(value - expected_down_last_five[i]).abs() < 1e-2,
				"Aroon Down mismatch at index {}: expected {}, got {}",
				i,
				expected_down_last_five[i],
				value
			);
		}

		Ok(())
	}

	fn check_aroon_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = AroonInput::with_default_candles(&candles);
		match input.data {
			AroonData::Candles { .. } => {}
			_ => panic!("Expected AroonData::Candles variant"),
		}
		let result = aroon_with_kernel(&input, kernel)?;
		assert_eq!(result.aroon_up.len(), candles.close.len());
		assert_eq!(result.aroon_down.len(), candles.close.len());
		Ok(())
	}

	fn check_aroon_zero_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [10.0, 11.0, 12.0];
		let low = [9.0, 10.0, 11.0];
		let params = AroonParams { length: Some(0) };
		let input = AroonInput::from_slices_hl(&high, &low, params);
		let result = aroon_with_kernel(&input, kernel);
		assert!(result.is_err(), "Expected error for zero length");
		Ok(())
	}

	fn check_aroon_length_exceeds_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [10.0, 11.0, 12.0];
		let low = [9.0, 10.0, 11.0];
		let params = AroonParams { length: Some(14) };
		let input = AroonInput::from_slices_hl(&high, &low, params);
		let result = aroon_with_kernel(&input, kernel);
		assert!(result.is_err(), "Expected error for length > data.len()");
		Ok(())
	}

	fn check_aroon_very_small_data_set(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [100.0];
		let low = [99.5];
		let params = AroonParams { length: Some(14) };
		let input = AroonInput::from_slices_hl(&high, &low, params);
		let result = aroon_with_kernel(&input, kernel);
		assert!(result.is_err(), "Expected error for data smaller than length");
		Ok(())
	}

	fn check_aroon_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let first_params = AroonParams { length: Some(14) };
		let first_input = AroonInput::from_candles(&candles, first_params);
		let first_result = aroon_with_kernel(&first_input, kernel)?;
		assert_eq!(first_result.aroon_up.len(), candles.close.len());
		assert_eq!(first_result.aroon_down.len(), candles.close.len());
		let second_params = AroonParams { length: Some(5) };
		let second_input = AroonInput::from_slices_hl(&candles.high, &candles.low, second_params);
		let second_result = aroon_with_kernel(&second_input, kernel)?;
		assert_eq!(second_result.aroon_up.len(), candles.close.len());
		assert_eq!(second_result.aroon_down.len(), candles.close.len());
		Ok(())
	}

	fn check_aroon_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = AroonParams { length: Some(14) };
		let input = AroonInput::from_candles(&candles, params);
		let result = aroon_with_kernel(&input, kernel)?;
		assert_eq!(result.aroon_up.len(), candles.close.len());
		assert_eq!(result.aroon_down.len(), candles.close.len());
		if result.aroon_up.len() > 240 {
			for i in 240..result.aroon_up.len() {
				assert!(!result.aroon_up[i].is_nan(), "Found NaN in aroon_up at {}", i);
				assert!(!result.aroon_down[i].is_nan(), "Found NaN in aroon_down at {}", i);
			}
		}
		Ok(())
	}

	fn check_aroon_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let length = 14;

		let input = AroonInput::from_candles(&candles, AroonParams { length: Some(length) });
		let batch_output = aroon_with_kernel(&input, kernel)?;

		let mut stream = AroonStream::try_new(AroonParams { length: Some(length) })?;
		let mut stream_up = Vec::with_capacity(candles.close.len());
		let mut stream_down = Vec::with_capacity(candles.close.len());
		for (&h, &l) in candles.high.iter().zip(&candles.low) {
			match stream.update(h, l) {
				Some((up, down)) => {
					stream_up.push(up);
					stream_down.push(down);
				}
				None => {
					stream_up.push(f64::NAN);
					stream_down.push(f64::NAN);
				}
			}
		}
		assert_eq!(batch_output.aroon_up.len(), stream_up.len());
		assert_eq!(batch_output.aroon_down.len(), stream_down.len());
		for (i, (&b, &s)) in batch_output.aroon_up.iter().zip(&stream_up).enumerate() {
			if b.is_nan() && s.is_nan() {
				continue;
			}
			let diff = (b - s).abs();
			assert!(
				diff < 1e-8,
				"[{}] Aroon streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
				test_name,
				i,
				b,
				s,
				diff
			);
		}
		for (i, (&b, &s)) in batch_output.aroon_down.iter().zip(&stream_down).enumerate() {
			if b.is_nan() && s.is_nan() {
				continue;
			}
			let diff = (b - s).abs();
			assert!(
				diff < 1e-8,
				"[{}] Aroon streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
				test_name,
				i,
				b,
				s,
				diff
			);
		}
		Ok(())
	}

	macro_rules! generate_all_aroon_tests {
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

	generate_all_aroon_tests!(
		check_aroon_partial_params,
		check_aroon_accuracy,
		check_aroon_default_candles,
		check_aroon_zero_length,
		check_aroon_length_exceeds_data,
		check_aroon_very_small_data_set,
		check_aroon_reinput,
		check_aroon_nan_handling,
		check_aroon_streaming
	);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let output = AroonBatchBuilder::new().kernel(kernel).apply_candles(&c)?;

		let def = AroonParams::default();
		let row = output.up_for(&def).expect("default up row missing");
		assert_eq!(row.len(), c.close.len());

		let expected = [21.43, 14.29, 7.14, 0.0, 0.0];
		let start = row.len() - 5;
		for (i, &v) in row[start..].iter().enumerate() {
			assert!(
				(v - expected[i]).abs() < 1e-2,
				"[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
			);
		}
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
}

#[cfg(feature = "python")]
#[pyfunction(name = "aroon")]
#[pyo3(signature = (high, low, length, kernel=None))]
pub fn aroon_py<'py>(
	py: Python<'py>,
	high: numpy::PyReadonlyArray1<'py, f64>,
	low: numpy::PyReadonlyArray1<'py, f64>,
	length: usize,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
	use numpy::{PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;

	let high_slice = high.as_slice()?; // zero-copy, read-only view
	let low_slice = low.as_slice()?; // zero-copy, read-only view

	// Validate input lengths
	if high_slice.len() != low_slice.len() {
		return Err(PyValueError::new_err(format!(
			"High/low data length mismatch: high={}, low={}",
			high_slice.len(),
			low_slice.len()
		)));
	}

	// Use kernel validation for safety
	let kern = validate_kernel(kernel, false)?;

	// ---------- build input struct -------------------------------------------------
	let params = AroonParams { length: Some(length) };
	let aroon_in = AroonInput::from_slices_hl(high_slice, low_slice, params);

	// ---------- allocate uninitialized NumPy output buffers ------------------------
	// NOTE: PyArray1::new() creates uninitialized memory, not zero-initialized
	// SAFETY: We MUST write to ALL elements before returning these arrays to Python.
	// Python/NumPy's memory model requires that all array elements are initialized.
	// Returning uninitialized memory to Python is undefined behavior and can cause
	// crashes or expose sensitive data from previous memory contents.
	let out_up = unsafe { PyArray1::<f64>::new(py, [high_slice.len()], false) };
	let out_down = unsafe { PyArray1::<f64>::new(py, [high_slice.len()], false) };
	let slice_up = unsafe { out_up.as_slice_mut()? };
	let slice_down = unsafe { out_down.as_slice_mut()? };

	// ---------- heavy lifting without the GIL --------------------------------------
	py.allow_threads(|| -> Result<(), AroonError> {
		// Get the appropriate kernel
		let chosen = match kern {
			Kernel::Auto => detect_best_kernel(),
			k => k,
		};

		// Validate parameters
		if length == 0 || length > high_slice.len() {
			return Err(AroonError::InvalidLength {
				length,
				data_len: high_slice.len(),
			});
		}

		// SAFETY: We must write to ALL elements before returning to Python
		// 1. Fill the warmup period (first `length` elements) with NaN
		// This is required because Aroon needs at least `length` data points
		// to compute the first valid value.
		if length > 0 {
			slice_up[..length].fill(f64::NAN);
			slice_down[..length].fill(f64::NAN);
		}

		// 2. aroon_scalar MUST write to all elements from index `length` onwards
		// This is guaranteed by the Aroon algorithm implementation which processes
		// every element from `length` to `len` in the main loop.
		unsafe {
			match chosen {
				Kernel::Scalar | Kernel::ScalarBatch => {
					aroon_scalar(high_slice, low_slice, length, slice_up, slice_down)
				}
				#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
				Kernel::Avx2 | Kernel::Avx2Batch => aroon_avx2(high_slice, low_slice, length, slice_up, slice_down),
				#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
				Kernel::Avx512 | Kernel::Avx512Batch => aroon_avx512(high_slice, low_slice, length, slice_up, slice_down),
				#[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
				Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
					// Fallback to scalar when AVX is not available
					aroon_scalar(high_slice, low_slice, length, slice_up, slice_down)
				}
				_ => unreachable!(),
			}
		}

		Ok(())
	})
	.map_err(|e| PyValueError::new_err(e.to_string()))?;

	// Build output dictionary
	let dict = PyDict::new(py);
	dict.set_item("up", out_up)?;
	dict.set_item("down", out_down)?;

	Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "AroonStream")]
pub struct AroonStreamPy {
	stream: AroonStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl AroonStreamPy {
	#[new]
	fn new(length: usize) -> PyResult<Self> {
		let params = AroonParams { length: Some(length) };
		let stream = AroonStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(AroonStreamPy { stream })
	}

	/// Updates the stream with new high and low values and returns the calculated Aroon values.
	/// Returns `None` if the buffer is not yet full, otherwise returns a tuple of (aroon_up, aroon_down).
	fn update(&mut self, high: f64, low: f64) -> Option<(f64, f64)> {
		self.stream.update(high, low)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "aroon_batch")]
#[pyo3(signature = (high, low, length_range, kernel=None))]
/// Batch Aroon calculation across multiple lengths.
pub fn aroon_batch_py<'py>(
	py: Python<'py>,
	high: numpy::PyReadonlyArray1<'py, f64>,
	low: numpy::PyReadonlyArray1<'py, f64>,
	length_range: (usize, usize, usize),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;

	let high_slice = high.as_slice()?;
	let low_slice = low.as_slice()?;

	// Validate input lengths
	if high_slice.len() != low_slice.len() {
		return Err(PyValueError::new_err(format!(
			"High/low data length mismatch: high={}, low={}",
			high_slice.len(),
			low_slice.len()
		)));
	}

	let sweep = AroonBatchRange { length: length_range };

	// 1. Expand grid once to know rows*cols
	let combos = expand_grid(&sweep);
	let rows = combos.len();
	let cols = high_slice.len();

	// 2. Pre-allocate uninitialized NumPy arrays (1-D, will reshape later)
	// NOTE: PyArray1::new() creates uninitialized memory, not zero-initialized
	// SAFETY: We must write to ALL elements before returning to Python
	let out_up = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let out_down = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let slice_up = unsafe { out_up.as_slice_mut()? };
	let slice_down = unsafe { out_down.as_slice_mut()? };

	// Use kernel validation for safety
	let kern = validate_kernel(kernel, true)?;

	// 3. Heavy work without the GIL
	let combos = py
		.allow_threads(|| -> Result<Vec<AroonParams>, AroonError> {
			// Resolve Kernel::Auto to a specific kernel
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

			// Validate data
			let max_l = combos.iter().map(|c| c.length.unwrap()).max().unwrap();
			if high_slice.len() < max_l {
				return Err(AroonError::NotEnoughValidData {
					needed: max_l,
					valid: high_slice.len(),
				});
			}

			// Process each row
			let do_row = |row: usize, out_up: &mut [f64], out_down: &mut [f64]| unsafe {
				let length = combos[row].length.unwrap();

				// SAFETY: Fill prefix with NaN since Aroon only writes from index `length` onwards
				if length > 0 {
					out_up[..length].fill(f64::NAN);
					out_down[..length].fill(f64::NAN);
				}

				match simd {
					Kernel::Scalar => aroon_row_scalar(high_slice, low_slice, length, out_up, out_down),
					#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
					Kernel::Avx2 => aroon_row_avx2(high_slice, low_slice, length, out_up, out_down),
					#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
					Kernel::Avx512 => aroon_row_avx512(high_slice, low_slice, length, out_up, out_down),
					#[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
					Kernel::Avx2 | Kernel::Avx512 => aroon_row_scalar(high_slice, low_slice, length, out_up, out_down),
					_ => unreachable!(),
				}
			};

			// Process all rows in parallel
			#[cfg(not(target_arch = "wasm32"))]
			{
				slice_up
					.par_chunks_mut(cols)
					.zip(slice_down.par_chunks_mut(cols))
					.enumerate()
					.for_each(|(row, (up_slice, down_slice))| do_row(row, up_slice, down_slice));
			}

			#[cfg(target_arch = "wasm32")]
			{
				for (row, (up_slice, down_slice)) in
					slice_up.chunks_mut(cols).zip(slice_down.chunks_mut(cols)).enumerate()
				{
					do_row(row, up_slice, down_slice);
				}
			}

			Ok(combos)
		})
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	// 4. Build dict with the GIL
	let dict = PyDict::new(py);
	dict.set_item("up", out_up.reshape((rows, cols))?)?;
	dict.set_item("down", out_down.reshape((rows, cols))?)?;
	dict.set_item(
		"lengths",
		combos
			.iter()
			.map(|p| p.length.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;

	Ok(dict)
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct AroonJsOutput {
	pub up: Vec<f64>,
	pub down: Vec<f64>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn aroon_js(high: &[f64], low: &[f64], length: usize) -> Result<JsValue, JsValue> {
	let params = AroonParams { length: Some(length) };
	let input = AroonInput::from_slices_hl(high, low, params);

	let output = aroon_with_kernel(&input, Kernel::Scalar).map_err(|e| JsValue::from_str(&e.to_string()))?;

	// Create the structured output
	let js_output = AroonJsOutput {
		up: output.aroon_up,
		down: output.aroon_down,
	};

	// Serialize the output struct into a JavaScript object
	serde_wasm_bindgen::to_value(&js_output).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn aroon_batch_js(
	high: &[f64],
	low: &[f64],
	length_start: usize,
	length_end: usize,
	length_step: usize,
) -> Result<JsValue, JsValue> {
	let sweep = AroonBatchRange {
		length: (length_start, length_end, length_step),
	};

	// Use the existing batch function with parallel=false for WASM
	let output =
		aroon_batch_inner(high, low, &sweep, Kernel::Scalar, false).map_err(|e| JsValue::from_str(&e.to_string()))?;

	// Create the structured output
	let js_output = AroonBatchJsOutput {
		up: output.up,
		down: output.down,
		combos: output.combos,
		rows: output.rows,
		cols: output.cols,
	};

	// Serialize the output struct into a JavaScript object
	serde_wasm_bindgen::to_value(&js_output).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn aroon_batch_metadata_js(
	length_start: usize,
	length_end: usize,
	length_step: usize,
) -> Result<Vec<f64>, JsValue> {
	let sweep = AroonBatchRange {
		length: (length_start, length_end, length_step),
	};

	let combos = expand_grid(&sweep);
	let metadata = combos.iter().map(|combo| combo.length.unwrap() as f64).collect();

	Ok(metadata)
}

// New ergonomic WASM API
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct AroonBatchConfig {
	pub length_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct AroonBatchJsOutput {
	pub up: Vec<f64>,
	pub down: Vec<f64>,
	pub combos: Vec<AroonParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = aroon_batch)]
pub fn aroon_batch_unified_js(high: &[f64], low: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	// 1. Deserialize the configuration object from JavaScript
	let config: AroonBatchConfig =
		serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

	let sweep = AroonBatchRange {
		length: config.length_range,
	};

	// 2. Run the existing core logic
	let output =
		aroon_batch_inner(high, low, &sweep, Kernel::Scalar, false).map_err(|e| JsValue::from_str(&e.to_string()))?;

	// 3. Create the structured output
	let js_output = AroonBatchJsOutput {
		up: output.up,
		down: output.down,
		combos: output.combos,
		rows: output.rows,
		cols: output.cols,
	};

	// 4. Serialize the output struct into a JavaScript object
	serde_wasm_bindgen::to_value(&js_output).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}
