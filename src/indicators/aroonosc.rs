//! # Aroon Oscillator
//!
//! The Aroon Oscillator measures the relative time since the most recent highest
//! high and lowest low within a specified `length`. It oscillates between -100
//! and +100, providing insights into the strength and direction of a price trend.
//! Higher positive values indicate a stronger uptrend, while negative values
//! signify a more dominant downtrend.
//!
//! ## Parameters
//! - **length**: The number of recent bars to look back when identifying the highest
//!   high and lowest low (defaults to 14).
//!
//! ## Errors
//! - **InvalidLength**: aroon_osc: The specified `length` is zero.
//! - **NoCandles**: aroon_osc: No candle data available.
//! - **EmptySlices**: aroon_osc: One or both high/low slices are empty.
//! - **SlicesLengthMismatch**: aroon_osc: High/low slices have different lengths.
//! - **NotEnoughData**: aroon_osc: Not enough data points to compute the Aroon Oscillator.
//!
//! ## Returns
//! - **`Ok(AroonOscOutput)`** on success, containing a `Vec<f64>` of the oscillator values.
//! - **`Err(AroonOscError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
	alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use paste::paste;
#[cfg(not(target_arch = "wasm32"))]
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::mem::ManuallyDrop;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum AroonOscData<'a> {
	Candles { candles: &'a Candles },
	SlicesHL { high: &'a [f64], low: &'a [f64] },
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(serde::Serialize, serde::Deserialize))]
pub struct AroonOscParams {
	pub length: Option<usize>,
}

impl Default for AroonOscParams {
	fn default() -> Self {
		Self { length: Some(14) }
	}
}

#[derive(Debug, Clone)]
pub struct AroonOscInput<'a> {
	pub data: AroonOscData<'a>,
	pub params: AroonOscParams,
}

impl<'a> AroonOscInput<'a> {
	#[inline]
	pub fn from_candles(candles: &'a Candles, params: AroonOscParams) -> Self {
		Self {
			data: AroonOscData::Candles { candles },
			params,
		}
	}
	#[inline]
	pub fn from_slices_hl(high: &'a [f64], low: &'a [f64], params: AroonOscParams) -> Self {
		Self {
			data: AroonOscData::SlicesHL { high, low },
			params,
		}
	}
	#[inline]
	pub fn with_default_candles(candles: &'a Candles) -> Self {
		Self {
			data: AroonOscData::Candles { candles },
			params: AroonOscParams::default(),
		}
	}
	#[inline]
	pub fn get_length(&self) -> usize {
		self.params.length.unwrap_or(14)
	}
	
	#[inline]
	pub fn data_len(&self) -> usize {
		match &self.data {
			AroonOscData::Candles { candles } => candles.close.len(),
			AroonOscData::SlicesHL { high, .. } => high.len(),
		}
	}
	
	#[inline]
	pub fn get_high(&self) -> &'a [f64] {
		match &self.data {
			AroonOscData::Candles { candles } => &candles.high,
			AroonOscData::SlicesHL { high, .. } => high,
		}
	}
	
	#[inline]
	pub fn get_low(&self) -> &'a [f64] {
		match &self.data {
			AroonOscData::Candles { candles } => &candles.low,
			AroonOscData::SlicesHL { low, .. } => low,
		}
	}
}

#[derive(Debug, Clone)]
pub struct AroonOscOutput {
	pub values: Vec<f64>,
}

#[derive(Copy, Clone, Debug)]
pub struct AroonOscBuilder {
	length: Option<usize>,
	kernel: Kernel,
}

impl Default for AroonOscBuilder {
	fn default() -> Self {
		Self {
			length: None,
			kernel: Kernel::Auto,
		}
	}
}

impl AroonOscBuilder {
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
	pub fn apply(self, c: &Candles) -> Result<AroonOscOutput, AroonOscError> {
		let p = AroonOscParams { length: self.length };
		let i = AroonOscInput::from_candles(c, p);
		aroon_osc_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slice(self, high: &[f64], low: &[f64]) -> Result<AroonOscOutput, AroonOscError> {
		let p = AroonOscParams { length: self.length };
		let i = AroonOscInput::from_slices_hl(high, low, p);
		aroon_osc_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<AroonOscStream, AroonOscError> {
		let p = AroonOscParams { length: self.length };
		AroonOscStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum AroonOscError {
	#[error("aroonosc: Candle field error: {0}")]
	CandleFieldError(String),

	#[error("aroonosc: Invalid length specified for Aroon Osc calculation. length={length}")]
	InvalidLength { length: usize },

	#[error("aroonosc: No candles available.")]
	NoCandles,

	#[error("aroonosc: One or both of the slices for AroonOsc are empty.")]
	EmptySlices,

	#[error("aroonosc: Mismatch in high/low slice length. high_len={high_len}, low_len={low_len}")]
	SlicesLengthMismatch { high_len: usize, low_len: usize },

	#[error("aroonosc: Not enough data points for Aroon Osc: required={required}, found={found}")]
	NotEnoughData { required: usize, found: usize },
	
	#[error("aroonosc: Empty data")]
	EmptyData,
	
	#[error("aroonosc: Mismatched array lengths: high_len={high_len}, low_len={low_len}")]
	MismatchedArrayLengths { high_len: usize, low_len: usize },
}

#[inline]
pub fn aroon_osc(input: &AroonOscInput) -> Result<AroonOscOutput, AroonOscError> {
	aroon_osc_with_kernel(input, Kernel::Auto)
}

pub fn aroon_osc_with_kernel(input: &AroonOscInput, kernel: Kernel) -> Result<AroonOscOutput, AroonOscError> {
	let length = input.get_length();
	if length == 0 {
		return Err(AroonOscError::InvalidLength { length });
	}
	let (high, low) = match &input.data {
		AroonOscData::Candles { candles } => {
			if candles.close.is_empty() {
				return Err(AroonOscError::NoCandles);
			}
			let high = candles
				.select_candle_field("high")
				.map_err(|e| AroonOscError::CandleFieldError(e.to_string()))?;
			let low = candles
				.select_candle_field("low")
				.map_err(|e| AroonOscError::CandleFieldError(e.to_string()))?;
			(high, low)
		}
		AroonOscData::SlicesHL { high, low } => {
			if high.is_empty() || low.is_empty() {
				return Err(AroonOscError::EmptySlices);
			}
			if high.len() != low.len() {
				return Err(AroonOscError::SlicesLengthMismatch {
					high_len: high.len(),
					low_len: low.len(),
				});
			}
			(*high, *low)
		}
	};
	let len = low.len();
	if len < length {
		return Err(AroonOscError::NotEnoughData {
			required: length,
			found: len,
		});
	}

	// Calculate warmup period for Aroon Oscillator
	// Aroon needs (length + 1) data points to start producing values
	let warmup_period = length; // Values before index (length) are NaN
	let mut out = alloc_with_nan_prefix(len, warmup_period);

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => aroon_osc_scalar(high, low, length, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => aroon_osc_avx2(high, low, length, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => aroon_osc_avx512(high, low, length, &mut out),
			_ => unreachable!(),
		}
	}
	Ok(AroonOscOutput { values: out })
}

#[inline]
pub fn aroon_osc_scalar(high: &[f64], low: &[f64], length: usize, out: &mut [f64]) {
	let len = low.len();
	let window = length + 1;
	let inv_length = 1.0 / length as f64;
	for i in (window - 1)..len {
		let start = i + 1 - window;
		let mut highest_val = high[start];
		let mut lowest_val = low[start];
		let mut highest_idx = start;
		let mut lowest_idx = start;
		for j in (start + 1)..=i {
			let h_val = high[j];
			if h_val > highest_val {
				highest_val = h_val;
				highest_idx = j;
			}
			let l_val = low[j];
			if l_val < lowest_val {
				lowest_val = l_val;
				lowest_idx = j;
			}
		}
		let offset_highest = i - highest_idx;
		let offset_lowest = i - lowest_idx;
		let up = (length as f64 - offset_highest as f64) * inv_length * 100.0;
		let down = (length as f64 - offset_lowest as f64) * inv_length * 100.0;
		out[i] = up - down;
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn aroon_osc_avx512(high: &[f64], low: &[f64], length: usize, out: &mut [f64]) {
	unsafe { aroon_osc_scalar(high, low, length, out) }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn aroon_osc_avx2(high: &[f64], low: &[f64], length: usize, out: &mut [f64]) {
	unsafe { aroon_osc_scalar(high, low, length, out) }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn aroon_osc_avx512_short(high: &[f64], low: &[f64], length: usize, out: &mut [f64]) {
	aroon_osc_scalar(high, low, length, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn aroon_osc_avx512_long(high: &[f64], low: &[f64], length: usize, out: &mut [f64]) {
	aroon_osc_scalar(high, low, length, out)
}

/// Write Aroon Oscillator directly to output slice - no allocations
#[inline]
pub fn aroon_osc_into_slice(dst: &mut [f64], input: &AroonOscInput, kern: Kernel) -> Result<(), AroonOscError> {
	let length = input.get_length();
	if length == 0 {
		return Err(AroonOscError::InvalidLength { length: 0 });
	}

	let data_len = input.data_len();
	if data_len == 0 {
		return Err(AroonOscError::EmptyData);
	}
	if length > data_len {
		return Err(AroonOscError::NotEnoughData {
			required: length,
			found: data_len,
		});
	}
	if dst.len() != data_len {
		return Err(AroonOscError::InvalidLength { 
			length: dst.len(),
		});
	}

	let high = input.get_high();
	let low = input.get_low();
	
	if high.len() != low.len() {
		return Err(AroonOscError::MismatchedArrayLengths {
			high_len: high.len(),
			low_len: low.len(),
		});
	}

	// Fill warmup with NaN first
	let warmup = length;
	for v in &mut dst[..warmup] {
		*v = f64::NAN;
	}

	// Select kernel
	let chosen = match kern {
		Kernel::Auto => detect_best_kernel(),
		k => k,
	};

	// Compute Aroon Oscillator values
	match chosen {
		#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
		Kernel::Avx512 => {
			#[target_feature(enable = "avx512f")]
			unsafe fn aroon_osc_avx512_wrapper(high: &[f64], low: &[f64], length: usize, out: &mut [f64]) {
				aroon_osc_avx512(high, low, length, out)
			}
			unsafe { aroon_osc_avx512_wrapper(high, low, length, dst) }
		}
		#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
		Kernel::Avx2 => {
			#[target_feature(enable = "avx2,fma")]
			unsafe fn aroon_osc_avx2_wrapper(high: &[f64], low: &[f64], length: usize, out: &mut [f64]) {
				aroon_osc_avx2(high, low, length, out)
			}
			unsafe { aroon_osc_avx2_wrapper(high, low, length, dst) }
		}
		_ => aroon_osc_scalar(high, low, length, dst),
	}

	Ok(())
}

#[inline(always)]
pub fn aroon_osc_batch_with_kernel(
	high: &[f64],
	low: &[f64],
	sweep: &AroonOscBatchRange,
	k: Kernel,
) -> Result<AroonOscBatchOutput, AroonOscError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => {
			return Err(AroonOscError::InvalidLength { length: 0 });
		}
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	aroon_osc_batch_par_slice(high, low, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct AroonOscBatchRange {
	pub length: (usize, usize, usize),
}

impl Default for AroonOscBatchRange {
	fn default() -> Self {
		Self { length: (14, 30, 1) }
	}
}

#[derive(Clone, Debug)]
pub struct AroonOscBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<AroonOscParams>,
	pub rows: usize,
	pub cols: usize,
}
impl AroonOscBatchOutput {
	pub fn row_for_params(&self, p: &AroonOscParams) -> Option<usize> {
		self.combos
			.iter()
			.position(|c| c.length.unwrap_or(14) == p.length.unwrap_or(14))
	}
	pub fn values_for(&self, p: &AroonOscParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[derive(Clone, Debug, Default)]
pub struct AroonOscBatchBuilder {
	range: AroonOscBatchRange,
	kernel: Kernel,
}
impl AroonOscBatchBuilder {
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
	pub fn length_static(mut self, l: usize) -> Self {
		self.range.length = (l, l, 0);
		self
	}
	pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<AroonOscBatchOutput, AroonOscError> {
		aroon_osc_batch_with_kernel(high, low, &self.range, self.kernel)
	}
	pub fn with_default_slices(high: &[f64], low: &[f64], k: Kernel) -> Result<AroonOscBatchOutput, AroonOscError> {
		AroonOscBatchBuilder::new().kernel(k).apply_slices(high, low)
	}
	pub fn apply_candles(self, c: &Candles) -> Result<AroonOscBatchOutput, AroonOscError> {
		let high = c
			.select_candle_field("high")
			.map_err(|e| AroonOscError::CandleFieldError(e.to_string()))?;
		let low = c
			.select_candle_field("low")
			.map_err(|e| AroonOscError::CandleFieldError(e.to_string()))?;
		self.apply_slices(high, low)
	}
	pub fn with_default_candles(c: &Candles) -> Result<AroonOscBatchOutput, AroonOscError> {
		AroonOscBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c)
	}
}

#[inline(always)]
fn expand_grid(r: &AroonOscBatchRange) -> Vec<AroonOscParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let lengths = axis_usize(r.length);
	let mut out = Vec::with_capacity(lengths.len());
	for &l in &lengths {
		out.push(AroonOscParams { length: Some(l) });
	}
	out
}

#[inline(always)]
pub fn aroon_osc_batch_slice(
	high: &[f64],
	low: &[f64],
	sweep: &AroonOscBatchRange,
	kern: Kernel,
) -> Result<AroonOscBatchOutput, AroonOscError> {
	aroon_osc_batch_inner(high, low, sweep, kern, false)
}
#[inline(always)]
pub fn aroon_osc_batch_par_slice(
	high: &[f64],
	low: &[f64],
	sweep: &AroonOscBatchRange,
	kern: Kernel,
) -> Result<AroonOscBatchOutput, AroonOscError> {
	aroon_osc_batch_inner(high, low, sweep, kern, true)
}

#[inline(always)]
fn aroon_osc_batch_inner(
	high: &[f64],
	low: &[f64],
	sweep: &AroonOscBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<AroonOscBatchOutput, AroonOscError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(AroonOscError::InvalidLength { length: 0 });
	}
	let len = high.len();
	if high.len() != low.len() {
		return Err(AroonOscError::SlicesLengthMismatch {
			high_len: high.len(),
			low_len: low.len(),
		});
	}
	let max_l = combos.iter().map(|c| c.length.unwrap()).max().unwrap();
	if len < max_l {
		return Err(AroonOscError::NotEnoughData {
			required: max_l,
			found: len,
		});
	}
	let rows = combos.len();
	let cols = len;

	// Step 1: Allocate uninitialized matrix
	let mut buf_mu = make_uninit_matrix(rows, cols);

	// Step 2: Calculate warmup periods for each row
	let warmup_periods: Vec<usize> = combos
		.iter()
		.map(|c| c.length.unwrap()) // Each row needs length warmup period
		.collect();

	// Step 3: Initialize NaN prefixes for each row
	init_matrix_prefixes(&mut buf_mu, cols, &warmup_periods);

	// Step 4: Convert to mutable slice for computation
	let mut buf_guard = ManuallyDrop::new(buf_mu);
	let values: &mut [f64] =
		unsafe { core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len()) };

	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let length = combos[row].length.unwrap();
		match kern {
			Kernel::Scalar => aroon_osc_row_scalar(high, low, length, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => aroon_osc_row_avx2(high, low, length, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => aroon_osc_row_avx512(high, low, length, out_row),
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

	// Step 5: Reclaim as Vec<f64>
	let values = unsafe {
		Vec::from_raw_parts(
			buf_guard.as_mut_ptr() as *mut f64,
			buf_guard.len(),
			buf_guard.capacity(),
		)
	};

	Ok(AroonOscBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
fn aroon_osc_batch_inner_into(
	high: &[f64],
	low: &[f64],
	sweep: &AroonOscBatchRange,
	kern: Kernel,
	parallel: bool,
	out: &mut [f64],
) -> Result<Vec<AroonOscParams>, AroonOscError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(AroonOscError::InvalidLength { length: 0 });
	}
	let len = high.len();
	if high.len() != low.len() {
		return Err(AroonOscError::SlicesLengthMismatch {
			high_len: high.len(),
			low_len: low.len(),
		});
	}
	let max_l = combos.iter().map(|c| c.length.unwrap()).max().unwrap();
	if len < max_l {
		return Err(AroonOscError::NotEnoughData {
			required: max_l,
			found: len,
		});
	}
	let rows = combos.len();
	let cols = len;

	// Initialize NaN prefixes for each row
	for (row, combo) in combos.iter().enumerate() {
		let warmup = combo.length.unwrap();
		let row_start = row * cols;
		out[row_start..row_start + warmup].fill(f64::NAN);
	}

	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let length = combos[row].length.unwrap();
		match kern {
			Kernel::Scalar => aroon_osc_row_scalar(high, low, length, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => aroon_osc_row_avx2(high, low, length, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => aroon_osc_row_avx512(high, low, length, out_row),
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

#[inline(always)]
pub unsafe fn aroon_osc_row_scalar(high: &[f64], low: &[f64], length: usize, out: &mut [f64]) {
	aroon_osc_scalar(high, low, length, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn aroon_osc_row_avx2(high: &[f64], low: &[f64], length: usize, out: &mut [f64]) {
	aroon_osc_scalar(high, low, length, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn aroon_osc_row_avx512(high: &[f64], low: &[f64], length: usize, out: &mut [f64]) {
	if length <= 32 {
		aroon_osc_avx512_short(high, low, length, out);
	} else {
		aroon_osc_avx512_long(high, low, length, out);
	}
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn aroon_osc_row_avx512_short(high: &[f64], low: &[f64], length: usize, out: &mut [f64]) {
	aroon_osc_scalar(high, low, length, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn aroon_osc_row_avx512_long(high: &[f64], low: &[f64], length: usize, out: &mut [f64]) {
	aroon_osc_scalar(high, low, length, out)
}

/// Write batch Aroon Oscillator directly to output slice - no allocations
#[inline]
pub fn aroon_osc_batch_into_slice(
	high: &[f64],
	low: &[f64],
	sweep: &AroonOscBatchRange,
	kern: Kernel,
	parallel: bool,
	out: &mut [f64],
) -> Result<Vec<AroonOscParams>, AroonOscError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(AroonOscError::InvalidLength { length: 0 });
	}
	
	let len = high.len();
	if high.len() != low.len() {
		return Err(AroonOscError::SlicesLengthMismatch {
			high_len: high.len(),
			low_len: low.len(),
		});
	}
	
	let expected_len = combos.len() * len;
	if out.len() != expected_len {
		return Err(AroonOscError::InvalidLength { 
			length: out.len(),
		});
	}
	
	// Use the existing inner function
	aroon_osc_batch_inner_into(high, low, sweep, kern, parallel, out)
}

#[derive(Debug, Clone)]
pub struct AroonOscStream {
	length: usize,
	high_buffer: Vec<f64>,
	low_buffer: Vec<f64>,
	head: usize,
	filled: bool,
}

impl AroonOscStream {
	pub fn try_new(params: AroonOscParams) -> Result<Self, AroonOscError> {
		let length = params.length.unwrap_or(14);
		if length == 0 {
			return Err(AroonOscError::InvalidLength { length });
		}
		Ok(Self {
			length,
			high_buffer: vec![f64::NAN; length + 1],
			low_buffer: vec![f64::NAN; length + 1],
			head: 0,
			filled: false,
		})
	}
	#[inline(always)]
	pub fn update(&mut self, high: f64, low: f64) -> Option<f64> {
		let window = self.length + 1;
		self.high_buffer[self.head] = high;
		self.low_buffer[self.head] = low;
		self.head = (self.head + 1) % window;
		if !self.filled && self.head == 0 {
			self.filled = true;
		}
		if !self.filled {
			return None;
		}
		Some(self.calc_ring())
	}
	#[inline(always)]
	fn calc_ring(&self) -> f64 {
		let window = self.length + 1;
		let mut highest_val = self.high_buffer[0];
		let mut lowest_val = self.low_buffer[0];
		let mut highest_idx = 0;
		let mut lowest_idx = 0;
		for i in 1..window {
			let idx = (self.head + i) % window;
			let h_val = self.high_buffer[idx];
			if h_val > highest_val {
				highest_val = h_val;
				highest_idx = i;
			}
			let l_val = self.low_buffer[idx];
			if l_val < lowest_val {
				lowest_val = l_val;
				lowest_idx = i;
			}
		}
		let offset_highest = self.length - highest_idx;
		let offset_lowest = self.length - lowest_idx;
		let inv_length = 1.0 / self.length as f64;
		let up = (self.length as f64 - offset_highest as f64) * inv_length * 100.0;
		let down = (self.length as f64 - offset_lowest as f64) * inv_length * 100.0;
		up - down
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;
	fn check_aroonosc_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let partial_params = AroonOscParams { length: Some(20) };
		let input = AroonOscInput::from_candles(&candles, partial_params);
		let result = aroon_osc_with_kernel(&input, kernel)?;
		assert_eq!(result.values.len(), candles.close.len());
		Ok(())
	}
	fn check_aroonosc_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = AroonOscInput::with_default_candles(&candles);
		let result = aroon_osc_with_kernel(&input, kernel)?;
		let expected_last_five = [-50.0, -50.0, -50.0, -50.0, -42.8571];
		assert!(result.values.len() >= 5, "Not enough Aroon Osc values");
		assert_eq!(result.values.len(), candles.close.len());
		let start_index = result.values.len().saturating_sub(5);
		let last_five = &result.values[start_index..];
		for (i, &value) in last_five.iter().enumerate() {
			assert!(
				(value - expected_last_five[i]).abs() < 1e-2,
				"Aroon Osc mismatch at index {}: expected {}, got {}",
				i,
				expected_last_five[i],
				value
			);
		}
		let length = 14;
		for val in result.values.iter().skip(length) {
			if !val.is_nan() {
				assert!(val.is_finite(), "Aroon Osc should be finite after enough data");
			}
		}
		Ok(())
	}
	fn check_aroonosc_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = AroonOscInput::with_default_candles(&candles);
		match input.data {
			AroonOscData::Candles { .. } => {}
			_ => panic!("Expected AroonOscData::Candles variant"),
		}
		assert!(input.params.length.is_some());
		Ok(())
	}
	fn check_aroonosc_with_slices_data_reinput(
		test_name: &str,
		kernel: Kernel,
	) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let first_params = AroonOscParams { length: Some(10) };
		let first_input = AroonOscInput::from_candles(&candles, first_params);
		let first_result = aroon_osc_with_kernel(&first_input, kernel)?;
		let second_params = AroonOscParams { length: Some(5) };
		let second_input = AroonOscInput::from_slices_hl(&first_result.values, &first_result.values, second_params);
		let second_result = aroon_osc_with_kernel(&second_input, kernel)?;
		assert_eq!(second_result.values.len(), first_result.values.len());
		for i in 20..second_result.values.len() {
			assert!(!second_result.values[i].is_nan());
		}
		Ok(())
	}
	fn check_aroonosc_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = AroonOscInput::with_default_candles(&candles);
		let result = aroon_osc_with_kernel(&input, kernel)?;
		if result.values.len() > 50 {
			for i in 50..result.values.len() {
				assert!(
					!result.values[i].is_nan(),
					"Expected no NaN after index {}, but found NaN",
					i
				);
			}
		}
		Ok(())
	}

	// Check for poison values in single output - only runs in debug mode
	#[cfg(debug_assertions)]
	fn check_aroonosc_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		// Test with multiple parameter combinations to catch edge cases
		let test_lengths = vec![
			5,   // Very small period
			14,  // Default period
			25,  // Medium period
			50,  // Large period
			100, // Very large period
			200, // Extra large period
		];

		for length in test_lengths {
			let params = AroonOscParams { length: Some(length) };
			let input = AroonOscInput::from_candles(&candles, params);

			// Skip if not enough data for this length
			if candles.close.len() < length {
				continue;
			}

			let output = aroon_osc_with_kernel(&input, kernel)?;

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
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} with length {}",
						test_name, val, bits, i, length
					);
				}

				// Check for init_matrix_prefixes poison (0x22222222_22222222)
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} with length {}",
						test_name, val, bits, i, length
					);
				}

				// Check for make_uninit_matrix poison (0x33333333_33333333)
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} with length {}",
						test_name, val, bits, i, length
					);
				}
			}
		}

		Ok(())
	}

	// Release mode stub - does nothing
	#[cfg(not(debug_assertions))]
	fn check_aroonosc_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		Ok(())
	}

	macro_rules! generate_all_aroonosc_tests {
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
	generate_all_aroonosc_tests!(
		check_aroonosc_partial_params,
		check_aroonosc_accuracy,
		check_aroonosc_default_candles,
		check_aroonosc_with_slices_data_reinput,
		check_aroonosc_nan_handling,
		check_aroonosc_no_poison
	);
	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let output = AroonOscBatchBuilder::new().kernel(kernel).apply_candles(&c)?;
		let def = AroonOscParams::default();
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

		// Test multiple batch configurations to catch edge cases
		let test_configs = vec![
			// (start, end, step) for length ranges
			(2, 10, 2),    // Small lengths: 2, 4, 6, 8, 10
			(5, 25, 5),    // Medium lengths: 5, 10, 15, 20, 25
			(10, 100, 10), // Large lengths: 10, 20, 30, ..., 100
			(50, 200, 50), // Very large lengths: 50, 100, 150, 200
			(14, 14, 0),   // Single parameter (default)
			(1, 5, 1),     // Edge case: very small lengths 1, 2, 3, 4, 5
		];

		for (start, end, step) in test_configs {
			// Skip if data is not sufficient for the largest length
			if c.close.len() < end {
				continue;
			}

			let output = AroonOscBatchBuilder::new()
				.kernel(kernel)
				.length_range(start, end, step)
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
				let length = output.combos[row].length.unwrap_or(14);

				// Check for alloc_with_nan_prefix poison (0x11111111_11111111)
				if bits == 0x11111111_11111111 {
					panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with length {} in range ({}, {}, {})",
                        test, val, bits, row, col, idx, length, start, end, step
                    );
				}

				// Check for init_matrix_prefixes poison (0x22222222_22222222)
				if bits == 0x22222222_22222222 {
					panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {}) with length {} in range ({}, {}, {})",
                        test, val, bits, row, col, idx, length, start, end, step
                    );
				}

				// Check for make_uninit_matrix poison (0x33333333_33333333)
				if bits == 0x33333333_33333333 {
					panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with length {} in range ({}, {}, {})",
                        test, val, bits, row, col, idx, length, start, end, step
                    );
				}
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
            paste! {
                #[test] fn [<$fn_name _scalar>]() { let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch); }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx2>]()   { let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch); }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx512>]() { let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch); }
            }
        };
    }
	gen_batch_tests!(check_batch_default_row);
	gen_batch_tests!(check_batch_no_poison);
}

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

#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;

#[cfg(feature = "python")]
#[pyfunction(name = "aroonosc")]
#[pyo3(signature = (high, low, length=14, kernel=None))]
pub fn aroon_osc_py<'py>(
	py: Python<'py>,
	high: numpy::PyReadonlyArray1<'py, f64>,
	low: numpy::PyReadonlyArray1<'py, f64>,
	length: usize,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
	use numpy::{IntoPyArray, PyArrayMethods};

	// Get slices - as_slice() will fail if array is not contiguous
	let high_slice = high.as_slice()?;
	let low_slice = low.as_slice()?;

	// Validate inputs have same length
	if high_slice.len() != low_slice.len() {
		return Err(PyValueError::new_err(format!(
			"High and low arrays must have same length. Got high: {}, low: {}",
			high_slice.len(),
			low_slice.len()
		)));
	}

	// Check length validity early
	if length == 0 {
		return Err(PyValueError::new_err("Invalid length: length must be greater than 0"));
	}

	// Validate kernel parameter before entering allow_threads
	let kern = validate_kernel(kernel, false)?;

	// Build input struct
	let params = AroonOscParams { length: Some(length) };
	let aroon_in = AroonOscInput::from_slices_hl(high_slice, low_slice, params);

	// Get Vec<f64> from Rust function with zero-copy transfer
	let result_vec: Vec<f64> = py
		.allow_threads(|| aroon_osc_with_kernel(&aroon_in, kern).map(|o| o.values))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	// Zero-copy transfer to NumPy
	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "AroonOscStream")]
pub struct AroonOscStreamPy {
	stream: AroonOscStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl AroonOscStreamPy {
	#[new]
	fn new(length: usize) -> PyResult<Self> {
		let params = AroonOscParams { length: Some(length) };
		let stream = AroonOscStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(AroonOscStreamPy { stream })
	}

	/// Updates the stream with new high/low values and returns the calculated Aroon Oscillator value.
	/// Returns `None` if the buffer is not yet full.
	fn update(&mut self, high: f64, low: f64) -> Option<f64> {
		self.stream.update(high, low)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "aroonosc_batch")]
#[pyo3(signature = (high, low, length_range, kernel=None))]
pub fn aroon_osc_batch_py<'py>(
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

	// Validate inputs have same length
	if high_slice.len() != low_slice.len() {
		return Err(PyValueError::new_err(format!(
			"High and low arrays must have same length. Got high: {}, low: {}",
			high_slice.len(),
			low_slice.len()
		)));
	}

	// Validate kernel parameter before entering allow_threads
	let kern = validate_kernel(kernel, true)?;

	let sweep = AroonOscBatchRange { length: length_range };

	// Calculate dimensions for pre-allocation
	let combos = expand_grid(&sweep);
	let rows = combos.len();
	let cols = high_slice.len();

	// Pre-allocate uninitialized NumPy array for maximum efficiency
	let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let slice_out = unsafe { out_arr.as_slice_mut()? };

	// Compute directly into pre-allocated buffer
	let combos = py
		.allow_threads(|| -> Result<Vec<AroonOscParams>, AroonOscError> {
			// Resolve kernel for batch operations
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

			aroon_osc_batch_inner_into(high_slice, low_slice, &sweep, simd, true, slice_out)
		})
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	// Build result dictionary with zero-copy transfers
	let dict = PyDict::new(py);
	dict.set_item("values", out_arr.reshape((rows, cols))?)?;
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
#[wasm_bindgen]
pub fn aroonosc_js(high: &[f64], low: &[f64], length: usize) -> Result<Vec<f64>, JsValue> {
	if high.len() != low.len() {
		return Err(JsValue::from_str(&format!(
			"High and low arrays must have same length. Got high: {}, low: {}",
			high.len(),
			low.len()
		)));
	}

	let params = AroonOscParams { length: Some(length) };
	let input = AroonOscInput::from_slices_hl(high, low, params);
	
	// Use aroon_osc_with_kernel to properly allocate with NaN prefix
	aroon_osc_with_kernel(&input, Kernel::Auto)
		.map(|output| output.values)
		.map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn aroonosc_batch_js(
	high: &[f64],
	low: &[f64],
	length_start: usize,
	length_end: usize,
	length_step: usize,
) -> Result<Vec<f64>, JsValue> {
	if high.len() != low.len() {
		return Err(JsValue::from_str(&format!(
			"High and low arrays must have same length. Got high: {}, low: {}",
			high.len(),
			low.len()
		)));
	}

	let sweep = AroonOscBatchRange {
		length: (length_start, length_end, length_step),
	};

	// Use the existing batch function with parallel=false for WASM
	aroon_osc_batch_slice(high, low, &sweep, Kernel::Auto)
		.map(|output| output.values)
		.map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn aroonosc_batch_metadata_js(
	length_start: usize,
	length_end: usize,
	length_step: usize,
) -> Result<Vec<f64>, JsValue> {
	let sweep = AroonOscBatchRange {
		length: (length_start, length_end, length_step),
	};

	let combos = expand_grid(&sweep);
	let mut metadata = Vec::with_capacity(combos.len());

	for combo in combos {
		metadata.push(combo.length.unwrap() as f64);
	}

	Ok(metadata)
}

// New ergonomic WASM API
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct AroonOscBatchConfig {
	pub length_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct AroonOscBatchJsOutput {
	pub values: Vec<f64>,
	pub combos: Vec<AroonOscParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = aroonosc_batch)]
pub fn aroon_osc_batch_unified_js(high: &[f64], low: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	if high.len() != low.len() {
		return Err(JsValue::from_str(&format!(
			"High and low arrays must have same length. Got high: {}, low: {}",
			high.len(),
			low.len()
		)));
	}

	// 1. Deserialize the configuration object from JavaScript
	let config: AroonOscBatchConfig =
		serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

	let sweep = AroonOscBatchRange {
		length: config.length_range,
	};

	// 2. Run the existing core logic
	let output =
		aroon_osc_batch_slice(high, low, &sweep, Kernel::Auto).map_err(|e| JsValue::from_str(&e.to_string()))?;

	// 3. Create the structured output
	let js_output = AroonOscBatchJsOutput {
		values: output.values,
		combos: output.combos,
		rows: output.rows,
		cols: output.cols,
	};

	// 4. Serialize the output struct into a JavaScript object
	serde_wasm_bindgen::to_value(&js_output).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn aroonosc_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn aroonosc_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe {
			let _ = Vec::from_raw_parts(ptr, len, len);
		}
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn aroonosc_into(
	high_ptr: *const f64,
	low_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	length: usize,
) -> Result<(), JsValue> {
	if high_ptr.is_null() || low_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}
	
	unsafe {
		let high = std::slice::from_raw_parts(high_ptr, len);
		let low = std::slice::from_raw_parts(low_ptr, len);
		
		let params = AroonOscParams { length: Some(length) };
		let input = AroonOscInput::from_slices_hl(high, low, params);
		
		// Check for aliasing - if any input overlaps with output, use temp buffer
		if high_ptr == out_ptr || low_ptr == out_ptr {
			let mut temp = vec![0.0; len];
			aroon_osc_into_slice(&mut temp, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			out.copy_from_slice(&temp);
		} else {
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			aroon_osc_into_slice(out, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		
		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn aroonosc_batch_into(
	high_ptr: *const f64,
	low_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	length_start: usize,
	length_end: usize,
	length_step: usize,
) -> Result<usize, JsValue> {
	if high_ptr.is_null() || low_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}
	
	unsafe {
		let high = std::slice::from_raw_parts(high_ptr, len);
		let low = std::slice::from_raw_parts(low_ptr, len);
		
		let sweep = AroonOscBatchRange {
			length: (length_start, length_end, length_step),
		};
		
		let combos = expand_grid(&sweep);
		let rows = combos.len();
		let expected_len = rows * len;
		let out = std::slice::from_raw_parts_mut(out_ptr, expected_len);
		
		// Check for aliasing - if any input overlaps with output, use temp buffer
		let high_overlaps = (high_ptr as usize) < (out_ptr as usize + expected_len * 8) &&
						   (high_ptr as usize + len * 8) > (out_ptr as usize);
		let low_overlaps = (low_ptr as usize) < (out_ptr as usize + expected_len * 8) &&
						  (low_ptr as usize + len * 8) > (out_ptr as usize);
		
		if high_overlaps || low_overlaps {
			let mut temp = vec![0.0; expected_len];
			aroon_osc_batch_into_slice(high, low, &sweep, Kernel::Auto, false, &mut temp)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			out.copy_from_slice(&temp);
		} else {
			aroon_osc_batch_into_slice(high, low, &sweep, Kernel::Auto, false, out)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		
		Ok(rows)
	}
}
