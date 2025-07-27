//! # MinMax (Local Extrema)
//!
//! Identifies local minima and maxima over a specified `order` range.
//! Provides SIMD and batch APIs, builder pattern, streaming, and robust input validation.
//!
//! ## Parameters
//! - **order**: Neighborhood range (defaults to 3)
//!
//! ## Errors
//! - **EmptyData**: minmax: Input data slice is empty.
//! - **InvalidOrder**: minmax: `order` is zero or exceeds the data length.
//! - **NotEnoughValidData**: minmax: Not enough valid data points for the requested `order`.
//! - **AllValuesNaN**: minmax: All input data values are `NaN`.
//!
//! ## Returns
//! - **`Ok(MinmaxOutput)`** on success, containing local extrema and forward-filled values.
//! - **`Err(MinmaxError)`** otherwise.

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
use std::error::Error;
use thiserror::Error;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

// --- DATA TYPES ---

#[derive(Debug, Clone)]
pub enum MinmaxData<'a> {
	Candles {
		candles: &'a Candles,
		high_src: &'a str,
		low_src: &'a str,
	},
	Slices {
		high: &'a [f64],
		low: &'a [f64],
	},
}

#[derive(Debug, Clone)]
pub struct MinmaxOutput {
	pub is_min: Vec<f64>,
	pub is_max: Vec<f64>,
	pub last_min: Vec<f64>,
	pub last_max: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct MinmaxParams {
	pub order: Option<usize>,
}

impl Default for MinmaxParams {
	fn default() -> Self {
		Self { order: Some(3) }
	}
}

#[derive(Debug, Clone)]
pub struct MinmaxInput<'a> {
	pub data: MinmaxData<'a>,
	pub params: MinmaxParams,
}

impl<'a> MinmaxInput<'a> {
	pub fn from_candles(candles: &'a Candles, high_src: &'a str, low_src: &'a str, params: MinmaxParams) -> Self {
		Self {
			data: MinmaxData::Candles {
				candles,
				high_src,
				low_src,
			},
			params,
		}
	}
	pub fn from_slices(high: &'a [f64], low: &'a [f64], params: MinmaxParams) -> Self {
		Self {
			data: MinmaxData::Slices { high, low },
			params,
		}
	}
	pub fn with_default_candles(candles: &'a Candles) -> Self {
		Self::from_candles(candles, "high", "low", MinmaxParams::default())
	}
	pub fn get_order(&self) -> usize {
		self.params.order.unwrap_or(3)
	}
}

// --- BUILDER ---

#[derive(Copy, Clone, Debug)]
pub struct MinmaxBuilder {
	order: Option<usize>,
	kernel: Kernel,
}

impl Default for MinmaxBuilder {
	fn default() -> Self {
		Self {
			order: None,
			kernel: Kernel::Auto,
		}
	}
}

impl MinmaxBuilder {
	pub fn new() -> Self {
		Self::default()
	}
	pub fn order(mut self, n: usize) -> Self {
		self.order = Some(n);
		self
	}
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	pub fn apply(self, candles: &Candles) -> Result<MinmaxOutput, MinmaxError> {
		let params = MinmaxParams { order: self.order };
		let input = MinmaxInput::from_candles(candles, "high", "low", params);
		minmax_with_kernel(&input, self.kernel)
	}
	pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<MinmaxOutput, MinmaxError> {
		let params = MinmaxParams { order: self.order };
		let input = MinmaxInput::from_slices(high, low, params);
		minmax_with_kernel(&input, self.kernel)
	}
	pub fn into_stream(self) -> Result<MinmaxStream, MinmaxError> {
		let params = MinmaxParams { order: self.order };
		MinmaxStream::try_new(params)
	}
}

// --- ERRORS ---

#[derive(Debug, Error)]
pub enum MinmaxError {
	#[error("minmax: Empty data provided.")]
	EmptyData,
	#[error("minmax: Invalid order: order = {order}, data length = {data_len}")]
	InvalidOrder { order: usize, data_len: usize },
	#[error("minmax: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("minmax: All values are NaN.")]
	AllValuesNaN,
}

// --- KERNEL API ---

#[inline]
pub fn minmax(input: &MinmaxInput) -> Result<MinmaxOutput, MinmaxError> {
	minmax_with_kernel(input, Kernel::Auto)
}

/// Write directly to output slices - no allocations
#[inline]
pub fn minmax_into_slice(
	is_min_dst: &mut [f64],
	is_max_dst: &mut [f64],
	last_min_dst: &mut [f64],
	last_max_dst: &mut [f64],
	input: &MinmaxInput,
	kern: Kernel,
) -> Result<(), MinmaxError> {
	let (high, low) = match &input.data {
		MinmaxData::Candles {
			candles,
			high_src,
			low_src,
		} => {
			let h = source_type(candles, high_src);
			let l = source_type(candles, low_src);
			(h, l)
		}
		MinmaxData::Slices { high, low } => (*high, *low),
	};

	if high.is_empty() || low.is_empty() {
		return Err(MinmaxError::EmptyData);
	}
	
	let order = input.get_order();
	let len_data = high.len();
	
	// Validate lengths
	if is_min_dst.len() != len_data || is_max_dst.len() != len_data 
		|| last_min_dst.len() != len_data || last_max_dst.len() != len_data {
		return Err(MinmaxError::InvalidOrder {
			order: is_min_dst.len(),
			data_len: len_data,
		});
	}
	
	if order == 0 || order > len_data {
		return Err(MinmaxError::InvalidOrder {
			order,
			data_len: len_data,
		});
	}
	
	let first_valid_idx = high
		.iter()
		.zip(low.iter())
		.position(|(&h, &l)| !(h.is_nan() || l.is_nan()))
		.ok_or(MinmaxError::AllValuesNaN)?;

	if (len_data - first_valid_idx) < order {
		return Err(MinmaxError::NotEnoughValidData {
			needed: order,
			valid: len_data - first_valid_idx,
		});
	}
	
	let chosen = match kern {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => minmax_scalar(
				high,
				low,
				order,
				first_valid_idx,
				is_min_dst,
				is_max_dst,
				last_min_dst,
				last_max_dst,
			),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => minmax_avx2(
				high,
				low,
				order,
				first_valid_idx,
				is_min_dst,
				is_max_dst,
				last_min_dst,
				last_max_dst,
			),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => minmax_avx512(
				high,
				low,
				order,
				first_valid_idx,
				is_min_dst,
				is_max_dst,
				last_min_dst,
				last_max_dst,
			),
			_ => minmax_scalar(
				high,
				low,
				order,
				first_valid_idx,
				is_min_dst,
				is_max_dst,
				last_min_dst,
				last_max_dst,
			),
		}
	}
	
	// Fill warmup with NaN
	let warmup_period = order;
	for v in &mut is_min_dst[..warmup_period] {
		*v = f64::NAN;
	}
	for v in &mut is_max_dst[..warmup_period] {
		*v = f64::NAN;
	}
	for v in &mut last_min_dst[..first_valid_idx] {
		*v = f64::NAN;
	}
	for v in &mut last_max_dst[..first_valid_idx] {
		*v = f64::NAN;
	}
	
	Ok(())
}

pub fn minmax_with_kernel(input: &MinmaxInput, kernel: Kernel) -> Result<MinmaxOutput, MinmaxError> {
	let (high, low) = match &input.data {
		MinmaxData::Candles {
			candles,
			high_src,
			low_src,
		} => {
			let h = source_type(candles, high_src);
			let l = source_type(candles, low_src);
			(h, l)
		}
		MinmaxData::Slices { high, low } => (*high, *low),
	};

	if high.is_empty() || low.is_empty() {
		return Err(MinmaxError::EmptyData);
	}
	let order = input.get_order();
	let len_data = high.len();
	if order == 0 || order > len_data {
		return Err(MinmaxError::InvalidOrder {
			order,
			data_len: len_data,
		});
	}
	let first_valid_idx = high
		.iter()
		.zip(low.iter())
		.position(|(&h, &l)| !(h.is_nan() || l.is_nan()))
		.ok_or(MinmaxError::AllValuesNaN)?;

	if (len_data - first_valid_idx) < order {
		return Err(MinmaxError::NotEnoughValidData {
			needed: order,
			valid: len_data - first_valid_idx,
		});
	}
	// Use zero-copy allocation for outputs
	let warmup_period = order;
	let mut is_min = alloc_with_nan_prefix(len_data, warmup_period);
	let mut is_max = alloc_with_nan_prefix(len_data, warmup_period);
	let mut last_min = alloc_with_nan_prefix(len_data, first_valid_idx);
	let mut last_max = alloc_with_nan_prefix(len_data, first_valid_idx);

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => minmax_scalar(
				high,
				low,
				order,
				first_valid_idx,
				&mut is_min,
				&mut is_max,
				&mut last_min,
				&mut last_max,
			),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => minmax_avx2(
				high,
				low,
				order,
				first_valid_idx,
				&mut is_min,
				&mut is_max,
				&mut last_min,
				&mut last_max,
			),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => minmax_avx512(
				high,
				low,
				order,
				first_valid_idx,
				&mut is_min,
				&mut is_max,
				&mut last_min,
				&mut last_max,
			),
			_ => unreachable!(),
		}
	}
	Ok(MinmaxOutput {
		is_min,
		is_max,
		last_min,
		last_max,
	})
}

// --- SCALAR LOGIC ---

#[inline]
pub fn minmax_scalar(
	high: &[f64],
	low: &[f64],
	order: usize,
	first_valid_idx: usize,
	is_min: &mut [f64],
	is_max: &mut [f64],
	last_min: &mut [f64],
	last_max: &mut [f64],
) {
	let len = high.len();
	let mut last_min_val = f64::NAN;
	let mut last_max_val = f64::NAN;

	for i in first_valid_idx..len {
		let center_low = low[i];
		let center_high = high[i];
		if i >= order && i + order < len && !center_low.is_nan() && !center_high.is_nan() {
			let mut less_than_neighbors = true;
			let mut greater_than_neighbors = true;

			for o in 1..=order {
				if center_low >= low[i - o] || center_low >= low[i + o] {
					less_than_neighbors = false;
				}
				if center_high <= high[i - o] || center_high <= high[i + o] {
					greater_than_neighbors = false;
				}
				if !less_than_neighbors && !greater_than_neighbors {
					break;
				}
			}
			if less_than_neighbors {
				is_min[i] = center_low;
			}
			if greater_than_neighbors {
				is_max[i] = center_high;
			}
		}
		if i == first_valid_idx {
			last_min_val = is_min[i];
			last_max_val = is_max[i];
		} else {
			if !is_min[i].is_nan() {
				last_min_val = is_min[i];
			}
			if !is_max[i].is_nan() {
				last_max_val = is_max[i];
			}
		}
		last_min[i] = last_min_val;
		last_max[i] = last_max_val;
	}
}

// --- AVX2/AVX512 STUBS ---

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn minmax_avx2(
	high: &[f64],
	low: &[f64],
	order: usize,
	first_valid_idx: usize,
	is_min: &mut [f64],
	is_max: &mut [f64],
	last_min: &mut [f64],
	last_max: &mut [f64],
) {
	minmax_scalar(high, low, order, first_valid_idx, is_min, is_max, last_min, last_max)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn minmax_avx512(
	high: &[f64],
	low: &[f64],
	order: usize,
	first_valid_idx: usize,
	is_min: &mut [f64],
	is_max: &mut [f64],
	last_min: &mut [f64],
	last_max: &mut [f64],
) {
	if order <= 16 {
		minmax_avx512_short(high, low, order, first_valid_idx, is_min, is_max, last_min, last_max)
	} else {
		minmax_avx512_long(high, low, order, first_valid_idx, is_min, is_max, last_min, last_max)
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn minmax_avx512_short(
	high: &[f64],
	low: &[f64],
	order: usize,
	first_valid_idx: usize,
	is_min: &mut [f64],
	is_max: &mut [f64],
	last_min: &mut [f64],
	last_max: &mut [f64],
) {
	minmax_scalar(high, low, order, first_valid_idx, is_min, is_max, last_min, last_max)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn minmax_avx512_long(
	high: &[f64],
	low: &[f64],
	order: usize,
	first_valid_idx: usize,
	is_min: &mut [f64],
	is_max: &mut [f64],
	last_min: &mut [f64],
	last_max: &mut [f64],
) {
	minmax_scalar(high, low, order, first_valid_idx, is_min, is_max, last_min, last_max)
}

// --- STREAMING ---

#[derive(Debug, Clone)]
pub struct MinmaxStream {
	order: usize,
	high_buf: Vec<f64>,
	low_buf: Vec<f64>,
	idx: usize,
	filled: bool,
	len: usize,
	last_min: f64,
	last_max: f64,
}

impl MinmaxStream {
	pub fn try_new(params: MinmaxParams) -> Result<Self, MinmaxError> {
		let order = params.order.unwrap_or(3);
		if order == 0 {
			return Err(MinmaxError::InvalidOrder { order, data_len: 0 });
		}
		Ok(Self {
			order,
			high_buf: vec![f64::NAN; order * 2 + 1],
			low_buf: vec![f64::NAN; order * 2 + 1],
			idx: 0,
			filled: false,
			len: order * 2 + 1,
			last_min: f64::NAN,
			last_max: f64::NAN,
		})
	}
	pub fn update(&mut self, high: f64, low: f64) -> (Option<f64>, Option<f64>, f64, f64) {
		self.high_buf[self.idx] = high;
		self.low_buf[self.idx] = low;
		self.idx = (self.idx + 1) % self.len;

		if !self.filled && self.idx == 0 {
			self.filled = true;
		}
		if !self.filled {
			return (None, None, self.last_min, self.last_max);
		}
		let center = (self.idx + self.len - self.order) % self.len;
		let center_high = self.high_buf[center];
		let center_low = self.low_buf[center];

		let mut is_min = true;
		let mut is_max = true;

		for o in 1..=self.order {
			let l_idx = (center + self.len - o) % self.len;
			let r_idx = (center + o) % self.len;
			if center_low >= self.low_buf[l_idx] || center_low >= self.low_buf[r_idx] {
				is_min = false;
			}
			if center_high <= self.high_buf[l_idx] || center_high <= self.high_buf[r_idx] {
				is_max = false;
			}
			if !is_min && !is_max {
				break;
			}
		}
		let min_val = if is_min { center_low } else { f64::NAN };
		let max_val = if is_max { center_high } else { f64::NAN };

		if !min_val.is_nan() {
			self.last_min = min_val;
		}
		if !max_val.is_nan() {
			self.last_max = max_val;
		}
		(
			if is_min { Some(center_low) } else { None },
			if is_max { Some(center_high) } else { None },
			self.last_min,
			self.last_max,
		)
	}
}

// --- BATCH API ---

#[derive(Clone, Debug)]
pub struct MinmaxBatchRange {
	pub order: (usize, usize, usize),
}

impl Default for MinmaxBatchRange {
	fn default() -> Self {
		Self { order: (3, 20, 1) }
	}
}

#[derive(Clone, Debug, Default)]
pub struct MinmaxBatchBuilder {
	range: MinmaxBatchRange,
	kernel: Kernel,
}

impl MinmaxBatchBuilder {
	pub fn new() -> Self {
		Self::default()
	}
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	pub fn order_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.order = (start, end, step);
		self
	}
	pub fn order_static(mut self, o: usize) -> Self {
		self.range.order = (o, o, 0);
		self
	}
	pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<MinmaxBatchOutput, MinmaxError> {
		minmax_batch_with_kernel(high, low, &self.range, self.kernel)
	}
	pub fn with_default_slices(high: &[f64], low: &[f64], k: Kernel) -> Result<MinmaxBatchOutput, MinmaxError> {
		MinmaxBatchBuilder::new().kernel(k).apply_slices(high, low)
	}
	pub fn apply_candles(self, c: &Candles) -> Result<MinmaxBatchOutput, MinmaxError> {
		let high = source_type(c, "high");
		let low = source_type(c, "low");
		self.apply_slices(high, low)
	}
	pub fn with_default_candles(c: &Candles) -> Result<MinmaxBatchOutput, MinmaxError> {
		MinmaxBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c)
	}
}

pub fn minmax_batch_with_kernel(
	high: &[f64],
	low: &[f64],
	sweep: &MinmaxBatchRange,
	k: Kernel,
) -> Result<MinmaxBatchOutput, MinmaxError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(MinmaxError::InvalidOrder { order: 0, data_len: 0 }),
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	minmax_batch_par_slice(high, low, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct MinmaxBatchOutput {
	pub is_min: Vec<f64>,
	pub is_max: Vec<f64>,
	pub last_min: Vec<f64>,
	pub last_max: Vec<f64>,
	pub combos: Vec<MinmaxParams>,
	pub rows: usize,
	pub cols: usize,
}

impl MinmaxBatchOutput {
	pub fn row_for_params(&self, p: &MinmaxParams) -> Option<usize> {
		self.combos
			.iter()
			.position(|c| c.order.unwrap_or(3) == p.order.unwrap_or(3))
	}
	pub fn is_min_for(&self, p: &MinmaxParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.is_min[start..start + self.cols]
		})
	}
	pub fn is_max_for(&self, p: &MinmaxParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.is_max[start..start + self.cols]
		})
	}
	pub fn last_min_for(&self, p: &MinmaxParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.last_min[start..start + self.cols]
		})
	}
	pub fn last_max_for(&self, p: &MinmaxParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.last_max[start..start + self.cols]
		})
	}
}

// --- BATCH EXPANSION ---

#[inline(always)]
fn expand_grid(r: &MinmaxBatchRange) -> Vec<MinmaxParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let orders = axis_usize(r.order);
	let mut out = Vec::with_capacity(orders.len());
	for &o in &orders {
		out.push(MinmaxParams { order: Some(o) });
	}
	out
}

#[inline(always)]
pub fn minmax_batch_slice(
	high: &[f64],
	low: &[f64],
	sweep: &MinmaxBatchRange,
	kern: Kernel,
) -> Result<MinmaxBatchOutput, MinmaxError> {
	minmax_batch_inner(high, low, sweep, kern, false)
}

#[inline(always)]
pub fn minmax_batch_par_slice(
	high: &[f64],
	low: &[f64],
	sweep: &MinmaxBatchRange,
	kern: Kernel,
) -> Result<MinmaxBatchOutput, MinmaxError> {
	minmax_batch_inner(high, low, sweep, kern, true)
}

#[inline(always)]
fn minmax_batch_inner(
	high: &[f64],
	low: &[f64],
	sweep: &MinmaxBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<MinmaxBatchOutput, MinmaxError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(MinmaxError::InvalidOrder { order: 0, data_len: 0 });
	}
	let len_data = high.len();
	let first_valid_idx = high
		.iter()
		.zip(low.iter())
		.position(|(&h, &l)| !(h.is_nan() || l.is_nan()))
		.ok_or(MinmaxError::AllValuesNaN)?;
	let max_o = combos.iter().map(|c| c.order.unwrap()).max().unwrap();
	if (len_data - first_valid_idx) < max_o {
		return Err(MinmaxError::NotEnoughValidData {
			needed: max_o,
			valid: len_data - first_valid_idx,
		});
	}
	let rows = combos.len();
	let cols = len_data;
	
	// Calculate warmup periods for each row
	let warmup_periods: Vec<usize> = combos.iter()
		.map(|c| c.order.unwrap() + first_valid_idx)
		.collect();
	
	// Use uninitialized memory helpers for batch outputs
	let mut is_min_buf = make_uninit_matrix(rows, cols);
	let mut is_max_buf = make_uninit_matrix(rows, cols);
	let mut last_min_buf = make_uninit_matrix(rows, cols);
	let mut last_max_buf = make_uninit_matrix(rows, cols);
	
	// Initialize NaN prefixes for each row
	init_matrix_prefixes(&mut is_min_buf, cols, &warmup_periods);
	init_matrix_prefixes(&mut is_max_buf, cols, &warmup_periods);
	
	let first_valid_periods: Vec<usize> = vec![first_valid_idx; rows];
	init_matrix_prefixes(&mut last_min_buf, cols, &first_valid_periods);
	init_matrix_prefixes(&mut last_max_buf, cols, &first_valid_periods);
	
	// Convert to mutable slices for computation
	let mut is_min_guard = core::mem::ManuallyDrop::new(is_min_buf);
	let mut is_max_guard = core::mem::ManuallyDrop::new(is_max_buf);
	let mut last_min_guard = core::mem::ManuallyDrop::new(last_min_buf);
	let mut last_max_guard = core::mem::ManuallyDrop::new(last_max_buf);
	
	let is_min: &mut [f64] = unsafe { core::slice::from_raw_parts_mut(is_min_guard.as_mut_ptr() as *mut f64, is_min_guard.len()) };
	let is_max: &mut [f64] = unsafe { core::slice::from_raw_parts_mut(is_max_guard.as_mut_ptr() as *mut f64, is_max_guard.len()) };
	let last_min: &mut [f64] = unsafe { core::slice::from_raw_parts_mut(last_min_guard.as_mut_ptr() as *mut f64, last_min_guard.len()) };
	let last_max: &mut [f64] = unsafe { core::slice::from_raw_parts_mut(last_max_guard.as_mut_ptr() as *mut f64, last_max_guard.len()) };

	let do_row = |row: usize,
	              out_min: &mut [f64],
	              out_max: &mut [f64],
	              out_last_min: &mut [f64],
	              out_last_max: &mut [f64]| unsafe {
		let order = combos[row].order.unwrap();
		match kern {
			Kernel::Scalar => minmax_row_scalar(
				high,
				low,
				first_valid_idx,
				order,
				out_min,
				out_max,
				out_last_min,
				out_last_max,
			),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => minmax_row_avx2(
				high,
				low,
				first_valid_idx,
				order,
				out_min,
				out_max,
				out_last_min,
				out_last_max,
			),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => minmax_row_avx512(
				high,
				low,
				first_valid_idx,
				order,
				out_min,
				out_max,
				out_last_min,
				out_last_max,
			),
			_ => unreachable!(),
		}
	};

	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			is_min
				.par_chunks_mut(cols)
				.zip(is_max.par_chunks_mut(cols))
				.zip(last_min.par_chunks_mut(cols).zip(last_max.par_chunks_mut(cols)))
				.enumerate()
				.for_each(|(row, ((min, max), (lmin, lmax)))| do_row(row, min, max, lmin, lmax));
		}
		#[cfg(target_arch = "wasm32")]
		{
			for (row, ((min, max), (lmin, lmax))) in is_min
				.chunks_mut(cols)
				.zip(is_max.chunks_mut(cols))
				.zip(last_min.chunks_mut(cols).zip(last_max.chunks_mut(cols)))
				.enumerate()
			{
				do_row(row, min, max, lmin, lmax)
			}
		}
	} else {
		for (row, ((min, max), (lmin, lmax))) in is_min
			.chunks_mut(cols)
			.zip(is_max.chunks_mut(cols))
			.zip(last_min.chunks_mut(cols).zip(last_max.chunks_mut(cols)))
			.enumerate()
		{
			do_row(row, min, max, lmin, lmax)
		}
	}
	
	// Convert back to Vecs
	let is_min = unsafe {
		Vec::from_raw_parts(
			is_min_guard.as_mut_ptr() as *mut f64,
			is_min_guard.len(),
			is_min_guard.capacity(),
		)
	};
	let is_max = unsafe {
		Vec::from_raw_parts(
			is_max_guard.as_mut_ptr() as *mut f64,
			is_max_guard.len(),
			is_max_guard.capacity(),
		)
	};
	let last_min = unsafe {
		Vec::from_raw_parts(
			last_min_guard.as_mut_ptr() as *mut f64,
			last_min_guard.len(),
			last_min_guard.capacity(),
		)
	};
	let last_max = unsafe {
		Vec::from_raw_parts(
			last_max_guard.as_mut_ptr() as *mut f64,
			last_max_guard.len(),
			last_max_guard.capacity(),
		)
	};
	
	// Forget the guards since we've taken ownership
	core::mem::forget(is_min_guard);
	core::mem::forget(is_max_guard);
	core::mem::forget(last_min_guard);
	core::mem::forget(last_max_guard);
	
	Ok(MinmaxBatchOutput {
		is_min,
		is_max,
		last_min,
		last_max,
		combos,
		rows,
		cols,
	})
}

// Direct buffer writing variant for Python/WASM bindings
#[inline(always)]
fn minmax_batch_inner_into(
	high: &[f64],
	low: &[f64],
	sweep: &MinmaxBatchRange,
	kern: Kernel,
	parallel: bool,
	is_min_out: &mut [f64],
	is_max_out: &mut [f64],
	last_min_out: &mut [f64],
	last_max_out: &mut [f64],
) -> Result<Vec<MinmaxParams>, MinmaxError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(MinmaxError::InvalidOrder { order: 0, data_len: 0 });
	}
	
	let len_data = high.len();
	let first_valid_idx = high
		.iter()
		.zip(low.iter())
		.position(|(&h, &l)| !(h.is_nan() || l.is_nan()))
		.ok_or(MinmaxError::AllValuesNaN)?;
	
	let max_o = combos.iter().map(|c| c.order.unwrap()).max().unwrap();
	if (len_data - first_valid_idx) < max_o {
		return Err(MinmaxError::NotEnoughValidData {
			needed: max_o,
			valid: len_data - first_valid_idx,
		});
	}
	
	let rows = combos.len();
	let cols = len_data;
	
	let do_row = |row: usize,
	              out_min: &mut [f64],
	              out_max: &mut [f64],
	              out_last_min: &mut [f64],
	              out_last_max: &mut [f64]| unsafe {
		let order = combos[row].order.unwrap();
		match kern {
			Kernel::Scalar => minmax_row_scalar(
				high,
				low,
				first_valid_idx,
				order,
				out_min,
				out_max,
				out_last_min,
				out_last_max,
			),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => minmax_row_avx2(
				high,
				low,
				first_valid_idx,
				order,
				out_min,
				out_max,
				out_last_min,
				out_last_max,
			),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => minmax_row_avx512(
				high,
				low,
				first_valid_idx,
				order,
				out_min,
				out_max,
				out_last_min,
				out_last_max,
			),
			_ => unreachable!(),
		}
	};
	
	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			is_min_out
				.par_chunks_mut(cols)
				.zip(is_max_out.par_chunks_mut(cols))
				.zip(last_min_out.par_chunks_mut(cols).zip(last_max_out.par_chunks_mut(cols)))
				.enumerate()
				.for_each(|(row, ((min, max), (lmin, lmax)))| do_row(row, min, max, lmin, lmax));
		}
		#[cfg(target_arch = "wasm32")]
		{
			for (row, ((min, max), (lmin, lmax))) in is_min_out
				.chunks_mut(cols)
				.zip(is_max_out.chunks_mut(cols))
				.zip(last_min_out.chunks_mut(cols).zip(last_max_out.chunks_mut(cols)))
				.enumerate()
			{
				do_row(row, min, max, lmin, lmax)
			}
		}
	} else {
		for (row, ((min, max), (lmin, lmax))) in is_min_out
			.chunks_mut(cols)
			.zip(is_max_out.chunks_mut(cols))
			.zip(last_min_out.chunks_mut(cols).zip(last_max_out.chunks_mut(cols)))
			.enumerate()
		{
			do_row(row, min, max, lmin, lmax)
		}
	}
	
	Ok(combos)
}

// --- BATCH ROW VARIANTS ---

#[inline(always)]
pub unsafe fn minmax_row_scalar(
	high: &[f64],
	low: &[f64],
	first_valid: usize,
	order: usize,
	is_min: &mut [f64],
	is_max: &mut [f64],
	last_min: &mut [f64],
	last_max: &mut [f64],
) {
	minmax_scalar(high, low, order, first_valid, is_min, is_max, last_min, last_max)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn minmax_row_avx2(
	high: &[f64],
	low: &[f64],
	first_valid: usize,
	order: usize,
	is_min: &mut [f64],
	is_max: &mut [f64],
	last_min: &mut [f64],
	last_max: &mut [f64],
) {
	minmax_row_scalar(high, low, first_valid, order, is_min, is_max, last_min, last_max)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn minmax_row_avx512(
	high: &[f64],
	low: &[f64],
	first_valid: usize,
	order: usize,
	is_min: &mut [f64],
	is_max: &mut [f64],
	last_min: &mut [f64],
	last_max: &mut [f64],
) {
	if order <= 16 {
		minmax_row_avx512_short(high, low, first_valid, order, is_min, is_max, last_min, last_max)
	} else {
		minmax_row_avx512_long(high, low, first_valid, order, is_min, is_max, last_min, last_max)
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn minmax_row_avx512_short(
	high: &[f64],
	low: &[f64],
	first_valid: usize,
	order: usize,
	is_min: &mut [f64],
	is_max: &mut [f64],
	last_min: &mut [f64],
	last_max: &mut [f64],
) {
	minmax_row_scalar(high, low, first_valid, order, is_min, is_max, last_min, last_max)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn minmax_row_avx512_long(
	high: &[f64],
	low: &[f64],
	first_valid: usize,
	order: usize,
	is_min: &mut [f64],
	is_max: &mut [f64],
	last_min: &mut [f64],
	last_max: &mut [f64],
) {
	minmax_row_scalar(high, low, first_valid, order, is_min, is_max, last_min, last_max)
}

// --- PYTHON BINDINGS ---

#[cfg(feature = "python")]
#[pyfunction(name = "minmax")]
#[pyo3(signature = (high, low, order, kernel=None))]
pub fn minmax_py<'py>(
	py: Python<'py>,
	high: numpy::PyReadonlyArray1<'py, f64>,
	low: numpy::PyReadonlyArray1<'py, f64>,
	order: usize,
	kernel: Option<&str>,
) -> PyResult<(Bound<'py, numpy::PyArray1<f64>>, Bound<'py, numpy::PyArray1<f64>>, 
              Bound<'py, numpy::PyArray1<f64>>, Bound<'py, numpy::PyArray1<f64>>)> {
	use numpy::{IntoPyArray, PyArrayMethods};
	
	let high_slice = high.as_slice()?;
	let low_slice = low.as_slice()?;
	let kern = validate_kernel(kernel, false)?;
	
	let params = MinmaxParams { order: Some(order) };
	let input = MinmaxInput::from_slices(high_slice, low_slice, params);
	
	let output = py.allow_threads(|| minmax_with_kernel(&input, kern))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;
	
	Ok((
		output.is_min.into_pyarray(py),
		output.is_max.into_pyarray(py),
		output.last_min.into_pyarray(py),
		output.last_max.into_pyarray(py),
	))
}

#[cfg(feature = "python")]
#[pyclass(name = "MinmaxStream")]
pub struct MinmaxStreamPy {
	stream: MinmaxStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl MinmaxStreamPy {
	#[new]
	fn new(order: usize) -> PyResult<Self> {
		let params = MinmaxParams { order: Some(order) };
		let stream = MinmaxStream::try_new(params)
			.map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(MinmaxStreamPy { stream })
	}
	
	fn update(&mut self, high: f64, low: f64) -> (Option<f64>, Option<f64>, f64, f64) {
		self.stream.update(high, low)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "minmax_batch")]
#[pyo3(signature = (high, low, order_range, kernel=None))]
pub fn minmax_batch_py<'py>(
	py: Python<'py>,
	high: numpy::PyReadonlyArray1<'py, f64>,
	low: numpy::PyReadonlyArray1<'py, f64>,
	order_range: (usize, usize, usize),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;
	
	let high_slice = high.as_slice()?;
	let low_slice = low.as_slice()?;
	let kern = validate_kernel(kernel, true)?;
	
	let sweep = MinmaxBatchRange { order: order_range };
	let combos = expand_grid(&sweep);
	let rows = combos.len();
	let cols = high_slice.len();
	
	// Pre-allocate arrays for batch operation
	let is_min_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let is_max_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let last_min_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let last_max_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	
	let is_min_slice = unsafe { is_min_arr.as_slice_mut()? };
	let is_max_slice = unsafe { is_max_arr.as_slice_mut()? };
	let last_min_slice = unsafe { last_min_arr.as_slice_mut()? };
	let last_max_slice = unsafe { last_max_arr.as_slice_mut()? };
	
	let combos = py.allow_threads(|| {
		let kernel = match kern {
			Kernel::Auto => detect_best_batch_kernel(),
			k => k,
		};
		
		// Map batch kernels to regular SIMD kernels
		let simd = match kernel {
			Kernel::Avx512Batch => Kernel::Avx512,
			Kernel::Avx2Batch => Kernel::Avx2,
			Kernel::ScalarBatch => Kernel::Scalar,
			_ => kernel,
		};
		
		// Write directly to pre-allocated buffers
		minmax_batch_inner_into(
			high_slice, 
			low_slice, 
			&sweep, 
			simd, 
			true, 
			is_min_slice, 
			is_max_slice,
			last_min_slice,
			last_max_slice
		)
	}).map_err(|e| PyValueError::new_err(e.to_string()))?;
	
	let dict = PyDict::new(py);
	dict.set_item("is_min", is_min_arr.reshape((rows, cols))?)?;
	dict.set_item("is_max", is_max_arr.reshape((rows, cols))?)?;
	dict.set_item("last_min", last_min_arr.reshape((rows, cols))?)?;
	dict.set_item("last_max", last_max_arr.reshape((rows, cols))?)?;
	dict.set_item(
		"orders",
		combos.iter()
			.map(|p| p.order.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py)
	)?;
	
	Ok(dict)
}

// --- WASM BINDINGS ---

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn minmax_js(
	high: &[f64],
	low: &[f64],
	order: usize,
) -> Result<JsValue, JsValue> {
	let params = MinmaxParams { order: Some(order) };
	let input = MinmaxInput::from_slices(high, low, params);
	
	// Single allocation for all outputs
	let len = high.len();
	let mut is_min = vec![0.0; len];
	let mut is_max = vec![0.0; len];
	let mut last_min = vec![0.0; len];
	let mut last_max = vec![0.0; len];
	
	minmax_into_slice(&mut is_min, &mut is_max, &mut last_min, &mut last_max, &input, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	let result = js_sys::Object::new();
	js_sys::Reflect::set(&result, &"is_min".into(), &serde_wasm_bindgen::to_value(&is_min).unwrap()).unwrap();
	js_sys::Reflect::set(&result, &"is_max".into(), &serde_wasm_bindgen::to_value(&is_max).unwrap()).unwrap();
	js_sys::Reflect::set(&result, &"last_min".into(), &serde_wasm_bindgen::to_value(&last_min).unwrap()).unwrap();
	js_sys::Reflect::set(&result, &"last_max".into(), &serde_wasm_bindgen::to_value(&last_max).unwrap()).unwrap();
	Ok(result.into())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn minmax_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn minmax_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe {
			let _ = Vec::from_raw_parts(ptr, len, len);
		}
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn minmax_into(
	high_ptr: *const f64,
	low_ptr: *const f64,
	is_min_ptr: *mut f64,
	is_max_ptr: *mut f64,
	last_min_ptr: *mut f64,
	last_max_ptr: *mut f64,
	len: usize,
	order: usize,
) -> Result<(), JsValue> {
	if high_ptr.is_null() || low_ptr.is_null() || is_min_ptr.is_null() 
		|| is_max_ptr.is_null() || last_min_ptr.is_null() || last_max_ptr.is_null() {
		return Err(JsValue::from_str("null pointer passed to minmax_into"));
	}

	unsafe {
		let high = std::slice::from_raw_parts(high_ptr, len);
		let low = std::slice::from_raw_parts(low_ptr, len);

		if order == 0 || order > len {
			return Err(JsValue::from_str("Invalid order"));
		}

		let params = MinmaxParams { order: Some(order) };
		let input = MinmaxInput::from_slices(high, low, params);

		// Check for aliasing between any input and output pointers
		let input_ptrs = [high_ptr as *const u8, low_ptr as *const u8];
		let output_ptrs = [
			is_min_ptr as *mut u8,
			is_max_ptr as *mut u8,
			last_min_ptr as *mut u8,
			last_max_ptr as *mut u8,
		];
		
		let mut needs_temp = false;
		for &inp in &input_ptrs {
			for &out in &output_ptrs {
				if inp == out {
					needs_temp = true;
					break;
				}
			}
			if needs_temp {
				break;
			}
		}

		if needs_temp {
			// Use temporary buffers
			let mut temp_is_min = vec![0.0; len];
			let mut temp_is_max = vec![0.0; len];
			let mut temp_last_min = vec![0.0; len];
			let mut temp_last_max = vec![0.0; len];
			
			minmax_into_slice(
				&mut temp_is_min, 
				&mut temp_is_max, 
				&mut temp_last_min, 
				&mut temp_last_max, 
				&input, 
				Kernel::Auto
			).map_err(|e| JsValue::from_str(&e.to_string()))?;
			
			// Copy results to output pointers
			let is_min_out = std::slice::from_raw_parts_mut(is_min_ptr, len);
			let is_max_out = std::slice::from_raw_parts_mut(is_max_ptr, len);
			let last_min_out = std::slice::from_raw_parts_mut(last_min_ptr, len);
			let last_max_out = std::slice::from_raw_parts_mut(last_max_ptr, len);
			
			is_min_out.copy_from_slice(&temp_is_min);
			is_max_out.copy_from_slice(&temp_is_max);
			last_min_out.copy_from_slice(&temp_last_min);
			last_max_out.copy_from_slice(&temp_last_max);
		} else {
			// Direct output
			let is_min_out = std::slice::from_raw_parts_mut(is_min_ptr, len);
			let is_max_out = std::slice::from_raw_parts_mut(is_max_ptr, len);
			let last_min_out = std::slice::from_raw_parts_mut(last_min_ptr, len);
			let last_max_out = std::slice::from_raw_parts_mut(last_max_ptr, len);
			
			minmax_into_slice(
				is_min_out, 
				is_max_out, 
				last_min_out, 
				last_max_out, 
				&input, 
				Kernel::Auto
			).map_err(|e| JsValue::from_str(&e.to_string()))?;
		}

		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct MinmaxBatchConfig {
	pub order_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = minmax_batch)]
pub fn minmax_batch_unified_js(
	high: &[f64],
	low: &[f64],
	config: JsValue,
) -> Result<JsValue, JsValue> {
	let config: MinmaxBatchConfig = serde_wasm_bindgen::from_value(config)
		.map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
	
	let sweep = MinmaxBatchRange { order: config.order_range };
	
	minmax_batch_with_kernel(high, low, &sweep, Kernel::Auto)
		.map(|output| {
			let result = js_sys::Object::new();
			let rows = output.rows;
			let cols = output.cols;
			
			// Create 2D arrays for each output
			let is_min_2d = js_sys::Array::new();
			let is_max_2d = js_sys::Array::new();
			let last_min_2d = js_sys::Array::new();
			let last_max_2d = js_sys::Array::new();
			
			for i in 0..rows {
				let start = i * cols;
				let end = start + cols;
				
				is_min_2d.push(&serde_wasm_bindgen::to_value(&output.is_min[start..end]).unwrap());
				is_max_2d.push(&serde_wasm_bindgen::to_value(&output.is_max[start..end]).unwrap());
				last_min_2d.push(&serde_wasm_bindgen::to_value(&output.last_min[start..end]).unwrap());
				last_max_2d.push(&serde_wasm_bindgen::to_value(&output.last_max[start..end]).unwrap());
			}
			
			js_sys::Reflect::set(&result, &"is_min".into(), &is_min_2d).unwrap();
			js_sys::Reflect::set(&result, &"is_max".into(), &is_max_2d).unwrap();
			js_sys::Reflect::set(&result, &"last_min".into(), &last_min_2d).unwrap();
			js_sys::Reflect::set(&result, &"last_max".into(), &last_max_2d).unwrap();
			js_sys::Reflect::set(&result, &"combos".into(), &serde_wasm_bindgen::to_value(&output.combos).unwrap()).unwrap();
			js_sys::Reflect::set(&result, &"rows".into(), &JsValue::from_f64(rows as f64)).unwrap();
			js_sys::Reflect::set(&result, &"cols".into(), &JsValue::from_f64(cols as f64)).unwrap();
			
			result.into()
		})
		.map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn minmax_batch_into(
	high_ptr: *const f64,
	low_ptr: *const f64,
	is_min_ptr: *mut f64,
	is_max_ptr: *mut f64,
	last_min_ptr: *mut f64,
	last_max_ptr: *mut f64,
	len: usize,
	order_start: usize,
	order_end: usize,
	order_step: usize,
) -> Result<usize, JsValue> {
	if high_ptr.is_null() || low_ptr.is_null() || is_min_ptr.is_null() 
		|| is_max_ptr.is_null() || last_min_ptr.is_null() || last_max_ptr.is_null() {
		return Err(JsValue::from_str("null pointer passed to minmax_batch_into"));
	}

	unsafe {
		let high = std::slice::from_raw_parts(high_ptr, len);
		let low = std::slice::from_raw_parts(low_ptr, len);

		let sweep = MinmaxBatchRange {
			order: (order_start, order_end, order_step),
		};

		let combos = expand_grid(&sweep);
		let rows = combos.len();
		let cols = len;

		let is_min_out = std::slice::from_raw_parts_mut(is_min_ptr, rows * cols);
		let is_max_out = std::slice::from_raw_parts_mut(is_max_ptr, rows * cols);
		let last_min_out = std::slice::from_raw_parts_mut(last_min_ptr, rows * cols);
		let last_max_out = std::slice::from_raw_parts_mut(last_max_ptr, rows * cols);

		minmax_batch_inner_into(
			high, 
			low, 
			&sweep, 
			Kernel::Auto, 
			false,
			is_min_out,
			is_max_out,
			last_min_out,
			last_max_out
		).map_err(|e| JsValue::from_str(&e.to_string()))?;

		Ok(rows)
	}
}

// --- TESTS ---

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_minmax_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = MinmaxParams { order: None };
		let input = MinmaxInput::from_candles(&candles, "high", "low", params);
		let output = minmax_with_kernel(&input, kernel)?;
		assert_eq!(output.is_min.len(), candles.close.len());
		Ok(())
	}

	fn check_minmax_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = MinmaxParams { order: Some(3) };
		let input = MinmaxInput::from_candles(&candles, "high", "low", params);
		let output = minmax_with_kernel(&input, kernel)?;
		assert_eq!(output.is_min.len(), candles.close.len());
		let count = output.is_min.len();
		assert!(count >= 5, "Not enough data to check last 5");
		let start_index = count - 5;
		for &val in &output.is_min[start_index..] {
			assert!(val.is_nan());
		}
		for &val in &output.is_max[start_index..] {
			assert!(val.is_nan());
		}
		let expected_last_five_min = [57876.0, 57876.0, 57876.0, 57876.0, 57876.0];
		let last_min_slice = &output.last_min[start_index..];
		for (i, &val) in last_min_slice.iter().enumerate() {
			let expected_val = expected_last_five_min[i];
			assert!(
				(val - expected_val).abs() < 1e-1,
				"Minmax last_min mismatch at idx {}: {} vs {}",
				i,
				val,
				expected_val
			);
		}
		let expected_last_five_max = [60102.0, 60102.0, 60102.0, 60102.0, 60102.0];
		let last_max_slice = &output.last_max[start_index..];
		for (i, &val) in last_max_slice.iter().enumerate() {
			let expected_val = expected_last_five_max[i];
			assert!(
				(val - expected_val).abs() < 1e-1,
				"Minmax last_max mismatch at idx {}: {} vs {}",
				i,
				val,
				expected_val
			);
		}
		Ok(())
	}

	fn check_minmax_zero_order(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [10.0, 20.0, 30.0];
		let low = [1.0, 2.0, 3.0];
		let params = MinmaxParams { order: Some(0) };
		let input = MinmaxInput::from_slices(&high, &low, params);
		let res = minmax_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] Minmax should fail with zero order", test_name);
		Ok(())
	}

	fn check_minmax_order_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [10.0, 20.0, 30.0];
		let low = [1.0, 2.0, 3.0];
		let params = MinmaxParams { order: Some(10) };
		let input = MinmaxInput::from_slices(&high, &low, params);
		let res = minmax_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] Minmax should fail with order > length", test_name);
		Ok(())
	}

	fn check_minmax_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [f64::NAN, f64::NAN, f64::NAN];
		let low = [f64::NAN, f64::NAN, f64::NAN];
		let params = MinmaxParams { order: Some(1) };
		let input = MinmaxInput::from_slices(&high, &low, params);
		let res = minmax_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] Minmax should fail with all NaN data", test_name);
		Ok(())
	}

	fn check_minmax_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [f64::NAN, 10.0];
		let low = [f64::NAN, 5.0];
		let params = MinmaxParams { order: Some(3) };
		let input = MinmaxInput::from_slices(&high, &low, params);
		let res = minmax_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] Minmax should fail with not enough valid data",
			test_name
		);
		Ok(())
	}

	fn check_minmax_basic_slices(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [50.0, 55.0, 60.0, 55.0, 50.0, 45.0, 50.0, 55.0];
		let low = [40.0, 38.0, 35.0, 38.0, 40.0, 42.0, 41.0, 39.0];
		let params = MinmaxParams { order: Some(2) };
		let input = MinmaxInput::from_slices(&high, &low, params);
		let output = minmax_with_kernel(&input, kernel)?;
		assert_eq!(output.is_min.len(), 8);
		assert_eq!(output.is_max.len(), 8);
		assert_eq!(output.last_min.len(), 8);
		assert_eq!(output.last_max.len(), 8);
		Ok(())
	}

	macro_rules! generate_all_minmax_tests {
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

	generate_all_minmax_tests!(
		check_minmax_partial_params,
		check_minmax_accuracy,
		check_minmax_zero_order,
		check_minmax_order_exceeds_length,
		check_minmax_nan_handling,
		check_minmax_very_small_dataset,
		check_minmax_basic_slices
	);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let output = MinmaxBatchBuilder::new().kernel(kernel).apply_candles(&c)?;
		let def = MinmaxParams::default();
		let row = output.is_min_for(&def).expect("default row missing");
		assert_eq!(row.len(), c.close.len());
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
