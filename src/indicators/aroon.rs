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

impl<'a> AsRef<[f64]> for AroonInput<'a> {
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			AroonData::Candles { candles } => &candles.high,
			AroonData::SlicesHL { high, .. } => high,
		}
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
	#[error("aroon: Input data slice is empty.")]
	EmptyInputData,
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
	if high.is_empty() || low.is_empty() {
		return Err(AroonError::EmptyInputData);
	}
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

	#[cfg(debug_assertions)]
	fn check_aroon_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		// Define comprehensive parameter combinations
		let test_params = vec![
			AroonParams::default(), // length: 14
			AroonParams { length: Some(1) }, // minimum length
			AroonParams { length: Some(2) }, // very small length
			AroonParams { length: Some(5) }, // small length
			AroonParams { length: Some(10) }, // medium length
			AroonParams { length: Some(20) }, // medium length
			AroonParams { length: Some(50) }, // large length
			AroonParams { length: Some(100) }, // very large length
			AroonParams { length: Some(200) }, // extra large length
		];

		for (param_idx, params) in test_params.iter().enumerate() {
			let input = AroonInput::from_candles(&candles, params.clone());
			let output = aroon_with_kernel(&input, kernel)?;

			// Check aroon_up values
			for (i, &val) in output.aroon_up.iter().enumerate() {
				if val.is_nan() {
					continue; // NaN values are expected during warmup
				}

				let bits = val.to_bits();

				// Check all three poison patterns
				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 in aroon_up output with params: length={} (param set {})",
						test_name,
						val,
						bits,
						i,
						params.length.unwrap_or(14),
						param_idx
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 in aroon_up output with params: length={} (param set {})",
						test_name,
						val,
						bits,
						i,
						params.length.unwrap_or(14),
						param_idx
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 in aroon_up output with params: length={} (param set {})",
						test_name,
						val,
						bits,
						i,
						params.length.unwrap_or(14),
						param_idx
					);
				}
			}

			// Check aroon_down values
			for (i, &val) in output.aroon_down.iter().enumerate() {
				if val.is_nan() {
					continue; // NaN values are expected during warmup
				}

				let bits = val.to_bits();

				// Check all three poison patterns
				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 in aroon_down output with params: length={} (param set {})",
						test_name,
						val,
						bits,
						i,
						params.length.unwrap_or(14),
						param_idx
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 in aroon_down output with params: length={} (param set {})",
						test_name,
						val,
						bits,
						i,
						params.length.unwrap_or(14),
						param_idx
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 in aroon_down output with params: length={} (param set {})",
						test_name,
						val,
						bits,
						i,
						params.length.unwrap_or(14),
						param_idx
					);
				}
			}
		}

		Ok(())
	}

	#[cfg(not(debug_assertions))]
	fn check_aroon_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		Ok(()) // No-op in release builds
	}

	#[cfg(feature = "proptest")]
	#[allow(clippy::float_cmp)]
	fn check_aroon_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		use proptest::prelude::*;
		skip_if_unsupported!(kernel, test_name);
		
		// Test strategy: generate length and OHLC data
		let strat = (1usize..=100).prop_flat_map(|length| {
			(
				prop::collection::vec(
					// Generate OHLC bars with realistic constraints
					(-1e6f64..1e6f64)
						.prop_filter("finite", |x| x.is_finite())
						.prop_flat_map(|base| {
							(0.0f64..0.3f64).prop_map(move |volatility| {
								let range = base.abs() * volatility + 0.01;
								let mid = base;
								let high = mid + range;
								let low = mid - range;
								(high, low)
							})
						}),
					length..400,
				),
				Just(length),
			)
		});
		
		proptest::test_runner::TestRunner::default()
			.run(&strat, |(bars, length)| {
				let (highs, lows): (Vec<f64>, Vec<f64>) = bars.into_iter().unzip();
				
				let params = AroonParams { length: Some(length) };
				let input = AroonInput::from_slices_hl(&highs, &lows, params.clone());
				
				let AroonOutput { aroon_up: out_up, aroon_down: out_down } = 
					aroon_with_kernel(&input, kernel).unwrap();
				let AroonOutput { aroon_up: ref_up, aroon_down: ref_down } = 
					aroon_with_kernel(&input, Kernel::Scalar).unwrap();
				
				// Property 1: Output structure
				prop_assert_eq!(out_up.len(), highs.len());
				prop_assert_eq!(out_down.len(), lows.len());
				
				// Property 2: Warmup period (first `length` values are NaN)
				for i in 0..length.min(out_up.len()) {
					prop_assert!(out_up[i].is_nan());
					prop_assert!(out_down[i].is_nan());
				}
				
				// Property 3: Valid values after warmup
				for i in length..out_up.len() {
					prop_assert!(!out_up[i].is_nan());
					prop_assert!(!out_down[i].is_nan());
				}
				
				// Property 4: Range bounds [0, 100]
				for i in length..out_up.len() {
					prop_assert!(
						out_up[i] >= 0.0 && out_up[i] <= 100.0,
						"Aroon up at {} = {}, outside [0,100]", i, out_up[i]
					);
					prop_assert!(
						out_down[i] >= 0.0 && out_down[i] <= 100.0,
						"Aroon down at {} = {}, outside [0,100]", i, out_down[i]
					);
				}
				
				// Property 5: Mathematical formula verification
				// Spot check a few calculated values
				for i in length..out_up.len().min(length + 5) {
					// Find the position of highest high in window
					let window_start = i - length;
					let mut max_val = highs[window_start];
					let mut max_idx = window_start;
					let mut min_val = lows[window_start];
					let mut min_idx = window_start;
					
					for j in (window_start + 1)..=i {
						if highs[j] > max_val {
							max_val = highs[j];
							max_idx = j;
						}
						if lows[j] < min_val {
							min_val = lows[j];
							min_idx = j;
						}
					}
					
					let periods_since_high = i - max_idx;
					let periods_since_low = i - min_idx;
					let expected_up = ((length as f64 - periods_since_high as f64) / length as f64) * 100.0;
					let expected_down = ((length as f64 - periods_since_low as f64) / length as f64) * 100.0;
					
					prop_assert!(
						(out_up[i] - expected_up).abs() < 1e-9,
						"Formula mismatch for aroon_up at {}: expected {}, got {}",
						i, expected_up, out_up[i]
					);
					prop_assert!(
						(out_down[i] - expected_down).abs() < 1e-9,
						"Formula mismatch for aroon_down at {}: expected {}, got {}",
						i, expected_down, out_down[i]
					);
				}
				
				// Property 6: Edge case - length = 1
				if length == 1 {
					// With length=1, the window size is actually 2 (indices [i-1, i])
					// The value depends on whether current bar's high/low is strictly greater/less
					// than the previous bar
					for i in 1..out_up.len().min(10) {
						// Aroon values with length=1 can only be 0 or 100
						prop_assert!(
							out_up[i] == 0.0 || out_up[i] == 100.0,
							"With length=1, aroon_up must be exactly 0 or 100, got {} at {}",
							out_up[i], i
						);
						prop_assert!(
							out_down[i] == 0.0 || out_down[i] == 100.0,
							"With length=1, aroon_down must be exactly 0 or 100, got {} at {}",
							out_down[i], i
						);
						
						// Additional check: verify the logic
						if i > 0 && i < highs.len() {
							// If current high > previous high, aroon_up should be 100
							if highs[i] > highs[i-1] {
								prop_assert_eq!(out_up[i], 100.0,
									"When high[{}]={} > high[{}]={}, aroon_up should be 100",
									i, highs[i], i-1, highs[i-1]
								);
							}
							// If current low < previous low, aroon_down should be 100
							if lows[i] < lows[i-1] {
								prop_assert_eq!(out_down[i], 100.0,
									"When low[{}]={} < low[{}]={}, aroon_down should be 100",
									i, lows[i], i-1, lows[i-1]
								);
							}
						}
					}
				}
				
				// Property 7: Constant data behavior
				let is_constant = highs.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10) &&
								 lows.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10);
				
				if is_constant && length > 1 {
					// With constant prices, all positions are equally "recent"
					// Aroon should decay to 0 as we get past the initial period
					for i in (length * 2).min(out_up.len())..(length * 3).min(out_up.len()) {
						prop_assert!(
							out_up[i] <= 100.0 / length as f64 + 1e-9,
							"With constant prices, aroon_up should approach 0, got {} at {}",
							out_up[i], i
						);
						prop_assert!(
							out_down[i] <= 100.0 / length as f64 + 1e-9,
							"With constant prices, aroon_down should approach 0, got {} at {}",
							out_down[i], i
						);
					}
				}
				
				// Property 8: Cross-kernel validation
				prop_assert_eq!(out_up.len(), ref_up.len());
				prop_assert_eq!(out_down.len(), ref_down.len());
				
				for i in 0..out_up.len() {
					let y_up = out_up[i];
					let r_up = ref_up[i];
					let y_down = out_down[i];
					let r_down = ref_down[i];
					
					// Check NaN/finite consistency
					if !y_up.is_finite() || !r_up.is_finite() {
						prop_assert_eq!(y_up.to_bits(), r_up.to_bits());
					} else {
						let ulp_diff = y_up.to_bits().abs_diff(r_up.to_bits());
						prop_assert!(
							(y_up - r_up).abs() <= 1e-9 || ulp_diff <= 4,
							"Kernel mismatch for aroon_up at {}: {} vs {} (ULP={})",
							i, y_up, r_up, ulp_diff
						);
					}
					
					if !y_down.is_finite() || !r_down.is_finite() {
						prop_assert_eq!(y_down.to_bits(), r_down.to_bits());
					} else {
						let ulp_diff = y_down.to_bits().abs_diff(r_down.to_bits());
						prop_assert!(
							(y_down - r_down).abs() <= 1e-9 || ulp_diff <= 4,
							"Kernel mismatch for aroon_down at {}: {} vs {} (ULP={})",
							i, y_down, r_down, ulp_diff
						);
					}
				}
				
				// Property 9: Monotonicity - Aroon decreases as distance from extreme increases
				// Test a few windows to verify this property
				for i in (length + 10)..(out_up.len().min(length + 15)) {
					let window_start = i - length;
					
					// Find position of highest high
					let mut max_idx = window_start;
					for j in (window_start + 1)..=i {
						if highs[j] > highs[max_idx] {
							max_idx = j;
						}
					}
					
					// If the high is getting older (further from current), Aroon up should decrease
					if i + 1 < out_up.len() && max_idx < i {
						// Next window: if same high is still max but now older
						let next_window_start = i + 1 - length;
						let mut next_max_idx = next_window_start;
						for j in (next_window_start + 1)..=i+1 {
							if j < highs.len() && highs[j] > highs[next_max_idx] {
								next_max_idx = j;
							}
						}
						
						// If the same extreme is still the max but older, Aroon should decrease
						if next_max_idx == max_idx {
							prop_assert!(
								out_up[i + 1] <= out_up[i] + 1e-9,
								"Monotonicity: Aroon up should decrease as extreme ages: {} -> {}",
								out_up[i], out_up[i + 1]
							);
						}
					}
				}
				
				// Property 10: High/Low relationship integrity
				for i in 0..highs.len() {
					prop_assert!(
						highs[i] >= lows[i],
						"Data integrity: High {} < Low {} at index {}",
						highs[i], lows[i], i
					);
				}
				
				// Property 11: Poison value detection (debug mode only)
				#[cfg(debug_assertions)]
				{
					for (i, &val) in out_up.iter().enumerate() {
						if val.is_finite() {
							let bits = val.to_bits();
							prop_assert!(
								bits != 0x11111111_11111111 && 
								bits != 0x22222222_22222222 && 
								bits != 0x33333333_33333333,
								"Found poison value {} (0x{:016X}) at {} in aroon_up",
								val, bits, i
							);
						}
					}
					for (i, &val) in out_down.iter().enumerate() {
						if val.is_finite() {
							let bits = val.to_bits();
							prop_assert!(
								bits != 0x11111111_11111111 && 
								bits != 0x22222222_22222222 && 
								bits != 0x33333333_33333333,
								"Found poison value {} (0x{:016X}) at {} in aroon_down",
								val, bits, i
							);
						}
					}
				}
				
				Ok(())
			})
			.unwrap();
		
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
		check_aroon_streaming,
		check_aroon_no_poison
	);
	
	#[cfg(feature = "proptest")]
	generate_all_aroon_tests!(check_aroon_property);

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

	#[cfg(debug_assertions)]
	fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		// Test various parameter sweep configurations
		let test_configs = vec![
			// (length_start, length_end, length_step)
			(1, 10, 1),        // Small lengths, every value
			(2, 20, 2),        // Small to medium, even values
			(5, 50, 5),        // Medium range, step 5
			(10, 100, 10),     // Medium to large, step 10
			(14, 14, 0),       // Static default length
			(50, 200, 50),     // Large lengths only
			(1, 5, 1),         // Very small lengths only
			(100, 200, 50),    // Very large lengths
			(3, 30, 3),        // Multiples of 3
		];

		for (cfg_idx, &(l_start, l_end, l_step)) in test_configs.iter().enumerate() {
			let output = AroonBatchBuilder::new()
				.kernel(kernel)
				.length_range(l_start, l_end, l_step)
				.apply_candles(&c)?;

			// Check aroon_up values
			for (idx, &val) in output.up.iter().enumerate() {
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
						 at row {} col {} (flat index {}) in aroon_up output with params: length={}",
						test,
						cfg_idx,
						val,
						bits,
						row,
						col,
						idx,
						combo.length.unwrap_or(14)
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in aroon_up output with params: length={}",
						test,
						cfg_idx,
						val,
						bits,
						row,
						col,
						idx,
						combo.length.unwrap_or(14)
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in aroon_up output with params: length={}",
						test,
						cfg_idx,
						val,
						bits,
						row,
						col,
						idx,
						combo.length.unwrap_or(14)
					);
				}
			}

			// Check aroon_down values
			for (idx, &val) in output.down.iter().enumerate() {
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
						 at row {} col {} (flat index {}) in aroon_down output with params: length={}",
						test,
						cfg_idx,
						val,
						bits,
						row,
						col,
						idx,
						combo.length.unwrap_or(14)
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in aroon_down output with params: length={}",
						test,
						cfg_idx,
						val,
						bits,
						row,
						col,
						idx,
						combo.length.unwrap_or(14)
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in aroon_down output with params: length={}",
						test,
						cfg_idx,
						val,
						bits,
						row,
						col,
						idx,
						combo.length.unwrap_or(14)
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
	use numpy::{IntoPyArray, PyArrayMethods};
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

	// Validate kernel before allow_threads
	let kern = validate_kernel(kernel, false)?;

	// Build input struct
	let params = AroonParams { length: Some(length) };
	let aroon_in = AroonInput::from_slices_hl(high_slice, low_slice, params);

	// GOOD: Get AroonOutput struct containing Vec<f64> from Rust function
	let output = py
		.allow_threads(|| aroon_with_kernel(&aroon_in, kern))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	// Build output dictionary with zero-copy transfer
	let dict = PyDict::new(py);
	dict.set_item("up", output.aroon_up.into_pyarray(py))?;
	dict.set_item("down", output.aroon_down.into_pyarray(py))?;

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

	// Validate kernel before allow_threads
	let kern = validate_kernel(kernel, true)?;

	let sweep = AroonBatchRange { length: length_range };

	// Heavy work without the GIL
	let output = py
		.allow_threads(|| {
			let kernel = match kern {
				Kernel::Auto => detect_best_batch_kernel(),
				k => k,
			};
			aroon_batch_with_kernel(high_slice, low_slice, &sweep, kernel)
		})
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	// Build dict with zero-copy transfers
	let dict = PyDict::new(py);
	dict.set_item("up", output.up.into_pyarray(py).reshape((output.rows, output.cols))?)?;
	dict.set_item("down", output.down.into_pyarray(py).reshape((output.rows, output.cols))?)?;
	dict.set_item(
		"lengths",
		output
			.combos
			.iter()
			.map(|p| p.length.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;

	Ok(dict)
}

/// Write directly to output slices - no allocations
#[inline]
pub fn aroon_into_slice(
	up_dst: &mut [f64],
	down_dst: &mut [f64],
	input: &AroonInput,
	kern: Kernel,
) -> Result<(), AroonError> {
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
	
	if up_dst.len() != len || down_dst.len() != len {
		return Err(AroonError::InvalidLength { 
			length: up_dst.len(), 
			data_len: len 
		});
	}
	
	if length == 0 || length > len {
		return Err(AroonError::InvalidLength { length, data_len: len });
	}
	
	let chosen = match kern {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};
	
	// Calculate warmup period
	let warmup_period = length;
	
	// Fill warmup with NaN
	for v in &mut up_dst[..warmup_period] {
		*v = f64::NAN;
	}
	for v in &mut down_dst[..warmup_period] {
		*v = f64::NAN;
	}
	
	// Compute directly into the output slices
	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => aroon_scalar(high, low, length, up_dst, down_dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => aroon_avx2(high, low, length, up_dst, down_dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => aroon_avx512(high, low, length, up_dst, down_dst),
			_ => unreachable!(),
		}
	}
	
	Ok(())
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

	// Use the proper aroon function which uses alloc_with_nan_prefix internally
	let output = aroon_with_kernel(&input, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;

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
	// In WASM, we should always use Scalar kernel
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
	// In WASM, we should always use Scalar kernel
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

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn aroon_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn aroon_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe {
			let _ = Vec::from_raw_parts(ptr, len, len);
		}
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn aroon_into(
	high_ptr: *const f64,
	low_ptr: *const f64,
	up_ptr: *mut f64,
	down_ptr: *mut f64,
	len: usize,
	length: usize,
) -> Result<(), JsValue> {
	if high_ptr.is_null() || low_ptr.is_null() || up_ptr.is_null() || down_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}
	
	unsafe {
		let high = std::slice::from_raw_parts(high_ptr, len);
		let low = std::slice::from_raw_parts(low_ptr, len);
		let params = AroonParams { length: Some(length) };
		let input = AroonInput::from_slices_hl(high, low, params);
		
		// Check for aliasing - multiple scenarios since we have 2 inputs and 2 outputs
		let needs_temp = high_ptr == up_ptr.cast()
			|| high_ptr == down_ptr.cast()
			|| low_ptr == up_ptr.cast()
			|| low_ptr == down_ptr.cast()
			|| up_ptr == down_ptr;
		
		if needs_temp {
			// Allocate temporary buffers for outputs
			let mut temp_up = vec![0.0; len];
			let mut temp_down = vec![0.0; len];
			aroon_into_slice(&mut temp_up, &mut temp_down, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			
			// Copy to output pointers
			let up_out = std::slice::from_raw_parts_mut(up_ptr, len);
			let down_out = std::slice::from_raw_parts_mut(down_ptr, len);
			up_out.copy_from_slice(&temp_up);
			down_out.copy_from_slice(&temp_down);
		} else {
			// Direct computation into output slices
			let up_out = std::slice::from_raw_parts_mut(up_ptr, len);
			let down_out = std::slice::from_raw_parts_mut(down_ptr, len);
			aroon_into_slice(up_out, down_out, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		
		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn aroon_batch_into(
	high_ptr: *const f64,
	low_ptr: *const f64,
	up_ptr: *mut f64,
	down_ptr: *mut f64,
	len: usize,
	length_start: usize,
	length_end: usize,
	length_step: usize,
) -> Result<usize, JsValue> {
	if high_ptr.is_null() || low_ptr.is_null() || up_ptr.is_null() || down_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}
	
	unsafe {
		let high = std::slice::from_raw_parts(high_ptr, len);
		let low = std::slice::from_raw_parts(low_ptr, len);
		
		let sweep = AroonBatchRange {
			length: (length_start, length_end, length_step),
		};
		
		let combos = expand_grid(&sweep);
		let rows = combos.len();
		let total_size = rows * len;
		
		// Prepare output slices
		let up_out = std::slice::from_raw_parts_mut(up_ptr, total_size);
		let down_out = std::slice::from_raw_parts_mut(down_ptr, total_size);
		
		// Compute each parameter combination
		for (i, params) in combos.iter().enumerate() {
			let row_start = i * len;
			let row_end = row_start + len;
			
			let input = AroonInput::from_slices_hl(high, low, params.clone());
			
			aroon_into_slice(
				&mut up_out[row_start..row_end],
				&mut down_out[row_start..row_end],
				&input,
				Kernel::Auto,
			)
			.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		
		Ok(rows)
	}
}
