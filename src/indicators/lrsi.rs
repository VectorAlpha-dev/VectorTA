//! # Laguerre RSI (LRSI)
//!
//! A momentum oscillator using a Laguerre filter, similar to RSI, but with different
//! responsiveness and smoothness characteristics. This implementation matches the
//! structure and feature parity of alma.rs, including AVX stubs, batch/grid support,
//! builder and streaming API, and full input validation.
//!
//! ## Parameters
//! - **alpha**: Smoothing factor (0 < alpha < 1). Default: 0.2
//!
//! ## Errors
//! - **AllValuesNaN**: lrsi: All input data values are `NaN`.
//! - **InvalidAlpha**: lrsi: `alpha` not in (0, 1).
//! - **EmptyData**: lrsi: Empty input.
//! - **NotEnoughValidData**: lrsi: Not enough valid data.
//!
//! ## Returns
//! - **Ok(LrsiOutput)** with `Vec<f64>` matching input
//! - **Err(LrsiError)** otherwise

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
use crate::utilities::helpers::{alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes, make_uninit_matrix};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum LrsiData<'a> {
	Candles { candles: &'a Candles },
	Slices { high: &'a [f64], low: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct LrsiOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct LrsiParams {
	pub alpha: Option<f64>,
}

impl Default for LrsiParams {
	fn default() -> Self {
		Self { alpha: Some(0.2) }
	}
}

#[derive(Debug, Clone)]
pub struct LrsiInput<'a> {
	pub data: LrsiData<'a>,
	pub params: LrsiParams,
}

impl<'a> LrsiInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, p: LrsiParams) -> Self {
		Self {
			data: LrsiData::Candles { candles: c },
			params: p,
		}
	}
	#[inline]
	pub fn from_slices(high: &'a [f64], low: &'a [f64], p: LrsiParams) -> Self {
		Self {
			data: LrsiData::Slices { high, low },
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, LrsiParams::default())
	}
	#[inline]
	pub fn get_alpha(&self) -> f64 {
		self.params.alpha.unwrap_or(0.2)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct LrsiBuilder {
	alpha: Option<f64>,
	kernel: Kernel,
}

impl Default for LrsiBuilder {
	fn default() -> Self {
		Self {
			alpha: None,
			kernel: Kernel::Auto,
		}
	}
}

impl LrsiBuilder {
	#[inline(always)]
	pub fn new() -> Self {
		Self::default()
	}
	#[inline(always)]
	pub fn alpha(mut self, x: f64) -> Self {
		self.alpha = Some(x);
		self
	}
	#[inline(always)]
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}

	#[inline(always)]
	pub fn apply(self, c: &Candles) -> Result<LrsiOutput, LrsiError> {
		let p = LrsiParams { alpha: self.alpha };
		let i = LrsiInput::from_candles(c, p);
		lrsi_with_kernel(&i, self.kernel)
	}

	#[inline(always)]
	pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<LrsiOutput, LrsiError> {
		let p = LrsiParams { alpha: self.alpha };
		let i = LrsiInput::from_slices(high, low, p);
		lrsi_with_kernel(&i, self.kernel)
	}

	#[inline(always)]
	pub fn into_stream(self) -> Result<LrsiStream, LrsiError> {
		let p = LrsiParams { alpha: self.alpha };
		LrsiStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum LrsiError {
	#[error("lrsi: Empty data provided.")]
	EmptyData,
	#[error("lrsi: Invalid alpha: alpha = {alpha}. Must be between 0 and 1.")]
	InvalidAlpha { alpha: f64 },
	#[error("lrsi: All values are NaN.")]
	AllValuesNaN,
	#[error("lrsi: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn lrsi(input: &LrsiInput) -> Result<LrsiOutput, LrsiError> {
	lrsi_with_kernel(input, Kernel::Auto)
}

pub fn lrsi_with_kernel(input: &LrsiInput, kernel: Kernel) -> Result<LrsiOutput, LrsiError> {
	let (high, low) = match &input.data {
		LrsiData::Candles { candles } => {
			let high = candles.select_candle_field("high").unwrap();
			let low = candles.select_candle_field("low").unwrap();
			(high, low)
		}
		LrsiData::Slices { high, low } => (*high, *low),
	};

	if high.is_empty() || low.is_empty() {
		return Err(LrsiError::EmptyData);
	}

	let alpha = input.get_alpha();
	if !(0.0 < alpha && alpha < 1.0) {
		return Err(LrsiError::InvalidAlpha { alpha });
	}

	// Find first valid price without allocating
	let mut first_valid_idx = None;
	for i in 0..high.len() {
		let price = (high[i] + low[i]) / 2.0;
		if !price.is_nan() {
			first_valid_idx = Some(i);
			break;
		}
	}
	
	let first_valid_idx = first_valid_idx.ok_or(LrsiError::AllValuesNaN)?;
	let n = high.len();
	if n - first_valid_idx < 4 {
		return Err(LrsiError::NotEnoughValidData {
			needed: 4,
			valid: n - first_valid_idx,
		});
	}

	let warmup_period = first_valid_idx + 3;  // Needs at least 4 values
	let mut out = alloc_with_nan_prefix(n, warmup_period);

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => lrsi_scalar_hl(high, low, alpha, first_valid_idx, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => lrsi_avx2_hl(high, low, alpha, first_valid_idx, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => lrsi_avx512_hl(high, low, alpha, first_valid_idx, &mut out),
			_ => unreachable!(),
		}
	}

	Ok(LrsiOutput { values: out })
}

#[inline]
pub fn lrsi_into_slice(dst: &mut [f64], input: &LrsiInput, kern: Kernel) -> Result<(), LrsiError> {
	let (high, low) = match &input.data {
		LrsiData::Candles { candles } => {
			let high = candles.select_candle_field("high").unwrap();
			let low = candles.select_candle_field("low").unwrap();
			(high, low)
		}
		LrsiData::Slices { high, low } => (*high, *low),
	};

	let alpha = input.get_alpha();
	if !(0.0 < alpha && alpha < 1.0) {
		return Err(LrsiError::InvalidAlpha { alpha });
	}

	// Find first valid price without allocating
	let mut first_valid_idx = None;
	for i in 0..high.len() {
		let price = (high[i] + low[i]) / 2.0;
		if !price.is_nan() {
			first_valid_idx = Some(i);
			break;
		}
	}
	
	let first_valid_idx = first_valid_idx.ok_or(LrsiError::AllValuesNaN)?;
	let n = high.len();
	
	if dst.len() != n {
		return Err(LrsiError::NotEnoughValidData {
			needed: n,
			valid: dst.len(),
		});
	}
	
	if n - first_valid_idx < 4 {
		return Err(LrsiError::NotEnoughValidData {
			needed: 4,
			valid: n - first_valid_idx,
		});
	}

	let chosen = match kern {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => lrsi_scalar_hl(high, low, alpha, first_valid_idx, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => lrsi_avx2_hl(high, low, alpha, first_valid_idx, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => lrsi_avx512_hl(high, low, alpha, first_valid_idx, dst),
			_ => unreachable!(),
		}
	}
	
	// Fill warmup period with NaN
	let warmup_end = first_valid_idx + 3;
	for v in &mut dst[..warmup_end] {
		*v = f64::NAN;
	}

	Ok(())
}

#[inline]
pub fn lrsi_scalar_hl(high: &[f64], low: &[f64], alpha: f64, first: usize, out: &mut [f64]) {
	let gamma = 1.0 - alpha;
	
	// Initialize state variables with first valid price
	let first_price = (high[first] + low[first]) / 2.0;
	let mut l0 = first_price;
	let mut l1 = first_price;
	let mut l2 = first_price;
	let mut l3 = first_price;
	out[first] = 0.0;

	// Process remaining values with rolling state
	for i in (first + 1)..high.len() {
		let p = (high[i] + low[i]) / 2.0;
		if p.is_nan() {
			out[i] = f64::NAN;
			continue;
		}

		// Update Laguerre filter states
		let new_l0 = alpha * p + gamma * l0;
		let new_l1 = -gamma * new_l0 + l0 + gamma * l1;
		let new_l2 = -gamma * new_l1 + l1 + gamma * l2;
		let new_l3 = -gamma * new_l2 + l2 + gamma * l3;

		// Calculate RSI-like ratio
		let mut cu = 0.0;
		let mut cd = 0.0;
		if new_l0 >= new_l1 {
			cu += new_l0 - new_l1;
		} else {
			cd += new_l1 - new_l0;
		}
		if new_l1 >= new_l2 {
			cu += new_l1 - new_l2;
		} else {
			cd += new_l2 - new_l1;
		}
		if new_l2 >= new_l3 {
			cu += new_l2 - new_l3;
		} else {
			cd += new_l3 - new_l2;
		}

		out[i] = if (cu + cd).abs() < f64::EPSILON {
			0.0
		} else {
			cu / (cu + cd)
		};

		// Update state for next iteration
		l0 = new_l0;
		l1 = new_l1;
		l2 = new_l2;
		l3 = new_l3;
	}
}

// Keep old function for compatibility with row functions
#[inline]
pub fn lrsi_scalar(price: &[f64], alpha: f64, first: usize, out: &mut [f64]) {
	let gamma = 1.0 - alpha;
	
	// Initialize state variables (no allocation needed)
	let mut l0 = price[first];
	let mut l1 = price[first];
	let mut l2 = price[first];
	let mut l3 = price[first];
	out[first] = 0.0;

	// Process remaining values with rolling state
	for i in (first + 1)..price.len() {
		let p = price[i];
		if p.is_nan() {
			out[i] = f64::NAN;
			continue;
		}

		// Update Laguerre filter states
		let new_l0 = alpha * p + gamma * l0;
		let new_l1 = -gamma * new_l0 + l0 + gamma * l1;
		let new_l2 = -gamma * new_l1 + l1 + gamma * l2;
		let new_l3 = -gamma * new_l2 + l2 + gamma * l3;

		// Calculate RSI-like ratio
		let mut cu = 0.0;
		let mut cd = 0.0;
		if new_l0 >= new_l1 {
			cu += new_l0 - new_l1;
		} else {
			cd += new_l1 - new_l0;
		}
		if new_l1 >= new_l2 {
			cu += new_l1 - new_l2;
		} else {
			cd += new_l2 - new_l1;
		}
		if new_l2 >= new_l3 {
			cu += new_l2 - new_l3;
		} else {
			cd += new_l3 - new_l2;
		}

		out[i] = if (cu + cd).abs() < f64::EPSILON {
			0.0
		} else {
			cu / (cu + cd)
		};

		// Update state for next iteration
		l0 = new_l0;
		l1 = new_l1;
		l2 = new_l2;
		l3 = new_l3;
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn lrsi_avx2_hl(high: &[f64], low: &[f64], alpha: f64, first: usize, out: &mut [f64]) {
	lrsi_scalar_hl(high, low, alpha, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn lrsi_avx512_hl(high: &[f64], low: &[f64], alpha: f64, first: usize, out: &mut [f64]) {
	lrsi_scalar_hl(high, low, alpha, first, out)
}

// Keep old functions for compatibility with row functions
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn lrsi_avx2(price: &[f64], alpha: f64, first: usize, out: &mut [f64]) {
	lrsi_scalar(price, alpha, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn lrsi_avx512(price: &[f64], alpha: f64, first: usize, out: &mut [f64]) {
	lrsi_scalar(price, alpha, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn lrsi_avx512_short(price: &[f64], alpha: f64, first: usize, out: &mut [f64]) {
	lrsi_scalar(price, alpha, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn lrsi_avx512_long(price: &[f64], alpha: f64, first: usize, out: &mut [f64]) {
	lrsi_scalar(price, alpha, first, out)
}

// Streaming API
#[derive(Debug, Clone)]
pub struct LrsiStream {
	alpha: f64,
	gamma: f64,
	l0: f64,
	l1: f64,
	l2: f64,
	l3: f64,
	initialized: bool,
}

impl LrsiStream {
	pub fn try_new(params: LrsiParams) -> Result<Self, LrsiError> {
		let alpha = params.alpha.unwrap_or(0.2);
		if !(0.0 < alpha && alpha < 1.0) {
			return Err(LrsiError::InvalidAlpha { alpha });
		}
		Ok(Self {
			alpha,
			gamma: 1.0 - alpha,
			l0: f64::NAN,
			l1: f64::NAN,
			l2: f64::NAN,
			l3: f64::NAN,
			initialized: false,
		})
	}
	#[inline(always)]
	pub fn update(&mut self, price: f64) -> Option<f64> {
		if price.is_nan() {
			return None;
		}
		if !self.initialized {
			self.l0 = price;
			self.l1 = price;
			self.l2 = price;
			self.l3 = price;
			self.initialized = true;
			return Some(0.0);
		}
		let alpha = self.alpha;
		let gamma = self.gamma;

		let l0 = alpha * price + gamma * self.l0;
		let l1 = -gamma * l0 + self.l0 + gamma * self.l1;
		let l2 = -gamma * l1 + self.l1 + gamma * self.l2;
		let l3 = -gamma * l2 + self.l2 + gamma * self.l3;

		self.l0 = l0;
		self.l1 = l1;
		self.l2 = l2;
		self.l3 = l3;

		let mut cu = 0.0;
		let mut cd = 0.0;
		if l0 >= l1 {
			cu += l0 - l1;
		} else {
			cd += l1 - l0;
		}
		if l1 >= l2 {
			cu += l1 - l2;
		} else {
			cd += l2 - l1;
		}
		if l2 >= l3 {
			cu += l2 - l3;
		} else {
			cd += l3 - l2;
		}
		Some(if (cu + cd).abs() < f64::EPSILON {
			0.0
		} else {
			cu / (cu + cd)
		})
	}
}

#[derive(Clone, Debug)]
pub struct LrsiBatchRange {
	pub alpha: (f64, f64, f64),
}

impl Default for LrsiBatchRange {
	fn default() -> Self {
		Self { alpha: (0.2, 0.2, 0.0) }
	}
}

#[derive(Clone, Debug, Default)]
pub struct LrsiBatchBuilder {
	range: LrsiBatchRange,
	kernel: Kernel,
}

impl LrsiBatchBuilder {
	pub fn new() -> Self {
		Self::default()
	}
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}

	#[inline]
	pub fn alpha_range(mut self, start: f64, end: f64, step: f64) -> Self {
		self.range.alpha = (start, end, step);
		self
	}
	#[inline]
	pub fn alpha_static(mut self, x: f64) -> Self {
		self.range.alpha = (x, x, 0.0);
		self
	}

	pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<LrsiBatchOutput, LrsiError> {
		lrsi_batch_with_kernel(high, low, &self.range, self.kernel)
	}

	pub fn with_default_slices(high: &[f64], low: &[f64], k: Kernel) -> Result<LrsiBatchOutput, LrsiError> {
		LrsiBatchBuilder::new().kernel(k).apply_slices(high, low)
	}

	pub fn apply_candles(self, c: &Candles) -> Result<LrsiBatchOutput, LrsiError> {
		let high = c.select_candle_field("high").unwrap();
		let low = c.select_candle_field("low").unwrap();
		self.apply_slices(high, low)
	}

	pub fn with_default_candles(c: &Candles) -> Result<LrsiBatchOutput, LrsiError> {
		LrsiBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c)
	}
}

pub fn lrsi_batch_with_kernel(
	high: &[f64],
	low: &[f64],
	sweep: &LrsiBatchRange,
	k: Kernel,
) -> Result<LrsiBatchOutput, LrsiError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(LrsiError::EmptyData),
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	lrsi_batch_par_slice(high, low, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct LrsiBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<LrsiParams>,
	pub rows: usize,
	pub cols: usize,
}
impl LrsiBatchOutput {
	pub fn row_for_params(&self, p: &LrsiParams) -> Option<usize> {
		self.combos
			.iter()
			.position(|c| (c.alpha.unwrap_or(0.2) - p.alpha.unwrap_or(0.2)).abs() < 1e-12)
	}

	pub fn values_for(&self, p: &LrsiParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &LrsiBatchRange) -> Vec<LrsiParams> {
	fn axis_f64((start, end, step): (f64, f64, f64)) -> Vec<f64> {
		if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
			return vec![start];
		}
		let mut v = Vec::new();
		let mut x = start;
		while x <= end + 1e-12 {
			v.push(x);
			x += step;
		}
		v
	}

	let alphas = axis_f64(r.alpha);

	let mut out = Vec::with_capacity(alphas.len());
	for &a in &alphas {
		out.push(LrsiParams { alpha: Some(a) });
	}
	out
}

#[inline(always)]
pub fn lrsi_batch_slice(
	high: &[f64],
	low: &[f64],
	sweep: &LrsiBatchRange,
	kern: Kernel,
) -> Result<LrsiBatchOutput, LrsiError> {
	lrsi_batch_inner(high, low, sweep, kern, false)
}

#[inline(always)]
pub fn lrsi_batch_par_slice(
	high: &[f64],
	low: &[f64],
	sweep: &LrsiBatchRange,
	kern: Kernel,
) -> Result<LrsiBatchOutput, LrsiError> {
	lrsi_batch_inner(high, low, sweep, kern, true)
}

#[inline(always)]
fn lrsi_batch_inner(
	high: &[f64],
	low: &[f64],
	sweep: &LrsiBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<LrsiBatchOutput, LrsiError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(LrsiError::EmptyData);
	}
	if high.len() == 0 || low.len() == 0 {
		return Err(LrsiError::EmptyData);
	}
	
	// Find first valid price without allocating
	let mut first = None;
	for i in 0..high.len() {
		let price = (high[i] + low[i]) / 2.0;
		if !price.is_nan() {
			first = Some(i);
			break;
		}
	}
	let first = first.ok_or(LrsiError::AllValuesNaN)?;
	
	let rows = combos.len();
	let cols = high.len();
	if cols - first < 4 {
		return Err(LrsiError::NotEnoughValidData {
			needed: 4,
			valid: cols - first,
		});
	}
	
	// Use helper functions for batch allocation
	let mut buf_uninit = make_uninit_matrix(rows, cols);
	let warmup_periods = vec![first + 3; rows];  // Same warmup for all rows
	init_matrix_prefixes(&mut buf_uninit, cols, &warmup_periods);
	
	// Convert to initialized slice
	let values_ptr = buf_uninit.as_mut_ptr() as *mut f64;
	let mut values = unsafe { std::slice::from_raw_parts_mut(values_ptr, rows * cols) };
	
	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let alpha = combos[row].alpha.unwrap();
		match kern {
			Kernel::Scalar => lrsi_row_scalar_hl(high, low, first, alpha, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => lrsi_row_avx2_hl(high, low, first, alpha, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => lrsi_row_avx512_hl(high, low, first, alpha, out_row),
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

	// SAFETY: buf_uninit was properly initialized through kernel computations
	let values = unsafe {
		let ptr = buf_uninit.as_mut_ptr();
		let len = buf_uninit.len();
		let cap = buf_uninit.capacity();
		std::mem::forget(buf_uninit);
		Vec::from_raw_parts(ptr as *mut f64, len, cap)
	};

	Ok(LrsiBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
unsafe fn lrsi_row_scalar_hl(high: &[f64], low: &[f64], first: usize, alpha: f64, out: &mut [f64]) {
	lrsi_scalar_hl(high, low, alpha, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn lrsi_row_avx2_hl(high: &[f64], low: &[f64], first: usize, alpha: f64, out: &mut [f64]) {
	lrsi_scalar_hl(high, low, alpha, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn lrsi_row_avx512_hl(high: &[f64], low: &[f64], first: usize, alpha: f64, out: &mut [f64]) {
	lrsi_scalar_hl(high, low, alpha, first, out)
}

// Keep old functions for compatibility
#[inline(always)]
unsafe fn lrsi_row_scalar(price: &[f64], first: usize, alpha: f64, out: &mut [f64]) {
	lrsi_scalar(price, alpha, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn lrsi_row_avx2(price: &[f64], first: usize, alpha: f64, out: &mut [f64]) {
	lrsi_scalar(price, alpha, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn lrsi_row_avx512(price: &[f64], first: usize, alpha: f64, out: &mut [f64]) {
	lrsi_scalar(price, alpha, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn lrsi_row_avx512_short(price: &[f64], first: usize, alpha: f64, out: &mut [f64]) {
	lrsi_scalar(price, alpha, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn lrsi_row_avx512_long(price: &[f64], first: usize, alpha: f64, out: &mut [f64]) {
	lrsi_scalar(price, alpha, first, out)
}

// WASM Bindings

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn lrsi_js(high: &[f64], low: &[f64], alpha: f64) -> Result<Vec<f64>, JsValue> {
	let params = LrsiParams { alpha: Some(alpha) };
	let input = LrsiInput::from_slices(high, low, params);
	
	let mut output = vec![0.0; high.len()];
	lrsi_into_slice(&mut output, &input, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	Ok(output)
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct LrsiBatchConfig {
	pub alpha_range: (f64, f64, f64),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct LrsiBatchJsOutput {
	pub values: Vec<f64>,
	pub combos: Vec<LrsiParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = lrsi_batch)]
pub fn lrsi_batch_unified_js(high: &[f64], low: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	let config: LrsiBatchConfig =
		serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
	
	let sweep = LrsiBatchRange {
		alpha: config.alpha_range,
	};
	
	let result = lrsi_batch_with_kernel(high, low, &sweep, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	let output = LrsiBatchJsOutput {
		values: result.values,
		combos: result.combos,
		rows: result.rows,
		cols: result.cols,
	};
	
	serde_wasm_bindgen::to_value(&output).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn lrsi_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn lrsi_free(ptr: *mut f64, len: usize) {
	unsafe {
		let _ = Vec::from_raw_parts(ptr, len, len);
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn lrsi_into(
	high_ptr: *const f64,
	low_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	alpha: f64,
) -> Result<(), JsValue> {
	if high_ptr.is_null() || low_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("null pointer passed to lrsi_into"));
	}
	
	unsafe {
		let high = std::slice::from_raw_parts(high_ptr, len);
		let low = std::slice::from_raw_parts(low_ptr, len);
		let params = LrsiParams { alpha: Some(alpha) };
		let input = LrsiInput::from_slices(high, low, params);
		
		// Check if we need to handle aliasing (in-place operation)
		if high_ptr == out_ptr || low_ptr == out_ptr {
			let mut temp = vec![0.0; len];
			lrsi_into_slice(&mut temp, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			out.copy_from_slice(&temp);
		} else {
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			lrsi_into_slice(out, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_lrsi_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let default_params = LrsiParams { alpha: None };
		let input = LrsiInput::from_candles(&candles, default_params);
		let output = lrsi_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_lrsi_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = LrsiInput::from_candles(&candles, LrsiParams::default());
		let lrsi_result = lrsi_with_kernel(&input, kernel)?;
		assert_eq!(lrsi_result.values.len(), candles.close.len());
		let expected_last_five_lrsi = [0.0, 0.0, 0.0, 0.0, 0.0];
		let start_index = lrsi_result.values.len() - 5;
		let result_last_five_lrsi = &lrsi_result.values[start_index..];
		for (i, &value) in result_last_five_lrsi.iter().enumerate() {
			let expected_value = expected_last_five_lrsi[i];
			assert!(
				(value - expected_value).abs() < 1e-9,
				"LRSI mismatch at index {}: expected {}, got {}",
				i,
				expected_value,
				value
			);
		}
		Ok(())
	}

	fn check_lrsi_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = LrsiInput::with_default_candles(&candles);
		let output = lrsi_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_lrsi_invalid_alpha(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [1.0, 2.0];
		let low = [1.0, 2.0];
		let params = LrsiParams { alpha: Some(1.2) };
		let input = LrsiInput::from_slices(&high, &low, params);
		let result = lrsi_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}

	fn check_lrsi_empty_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high: [f64; 0] = [];
		let low: [f64; 0] = [];
		let params = LrsiParams::default();
		let input = LrsiInput::from_slices(&high, &low, params);
		let result = lrsi_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}

	fn check_lrsi_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [f64::NAN, f64::NAN, f64::NAN];
		let low = [f64::NAN, f64::NAN, f64::NAN];
		let params = LrsiParams::default();
		let input = LrsiInput::from_slices(&high, &low, params);
		let result = lrsi_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}

	fn check_lrsi_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [1.0, 1.0];
		let low = [1.0, 1.0];
		let params = LrsiParams::default();
		let input = LrsiInput::from_slices(&high, &low, params);
		let result = lrsi_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}

	fn check_lrsi_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let high = candles.select_candle_field("high").unwrap();
		let low = candles.select_candle_field("low").unwrap();

		let input = LrsiInput::from_slices(high, low, LrsiParams::default());
		let batch_output = lrsi_with_kernel(&input, kernel)?.values;

		let mut stream = LrsiStream::try_new(LrsiParams::default())?;
		let mut stream_values = Vec::with_capacity(high.len());
		for i in 0..high.len() {
			let price = (high[i] + low[i]) / 2.0;
			match stream.update(price) {
				Some(val) => stream_values.push(val),
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
				"[{}] LRSI streaming mismatch at idx {}: batch={}, stream={}, diff={}",
				test_name,
				i,
				b,
				s,
				diff
			);
		}
		Ok(())
	}

	macro_rules! generate_all_lrsi_tests {
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

	generate_all_lrsi_tests!(
		check_lrsi_partial_params,
		check_lrsi_accuracy,
		check_lrsi_default_candles,
		check_lrsi_invalid_alpha,
		check_lrsi_empty_data,
		check_lrsi_all_nan,
		check_lrsi_very_small_dataset,
		check_lrsi_streaming
	);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let output = LrsiBatchBuilder::new().kernel(kernel).apply_candles(&c)?;

		let def = LrsiParams::default();
		let row = output.values_for(&def).expect("default row missing");
		assert_eq!(row.len(), c.close.len());

		let expected = [0.0, 0.0, 0.0, 0.0, 0.0];
		let start = row.len() - 5;
		for (i, &v) in row[start..].iter().enumerate() {
			assert!(
				(v - expected[i]).abs() < 1e-9,
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

// Batch function that writes directly to output slice for Python bindings
#[inline(always)]
fn lrsi_batch_inner_into(
	high: &[f64],
	low: &[f64],
	sweep: &LrsiBatchRange,
	kern: Kernel,
	parallel: bool,
	out: &mut [f64],
) -> Result<Vec<LrsiParams>, LrsiError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(LrsiError::EmptyData);
	}
	if high.len() == 0 || low.len() == 0 {
		return Err(LrsiError::EmptyData);
	}
	
	// Find first valid price without allocating
	let mut first = None;
	for i in 0..high.len() {
		let price = (high[i] + low[i]) / 2.0;
		if !price.is_nan() {
			first = Some(i);
			break;
		}
	}
	let first = first.ok_or(LrsiError::AllValuesNaN)?;
	
	let rows = combos.len();
	let cols = high.len();
	if cols - first < 4 {
		return Err(LrsiError::NotEnoughValidData {
			needed: 4,
			valid: cols - first,
		});
	}

	// Fill with NaN prefix
	for row in 0..rows {
		let row_start = row * cols;
		for col in 0..(first + 3) {
			out[row_start + col] = f64::NAN;
		}
	}

	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let alpha = combos[row].alpha.unwrap();
		match kern {
			Kernel::Scalar => lrsi_row_scalar_hl(high, low, first, alpha, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => lrsi_row_avx2_hl(high, low, first, alpha, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => lrsi_row_avx512_hl(high, low, first, alpha, out_row),
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

#[cfg(feature = "python")]
#[pyfunction(name = "lrsi")]
#[pyo3(signature = (high, low, alpha, kernel=None))]
pub fn lrsi_py<'py>(
	py: Python<'py>,
	high: PyReadonlyArray1<'py, f64>,
	low: PyReadonlyArray1<'py, f64>,
	alpha: f64,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
	let high_slice = high.as_slice()?;
	let low_slice = low.as_slice()?;
	let kern = validate_kernel(kernel, false)?;
	
	let params = LrsiParams { alpha: Some(alpha) };
	let input = LrsiInput::from_slices(high_slice, low_slice, params);
	
	let result_vec: Vec<f64> = py
		.allow_threads(|| lrsi_with_kernel(&input, kern).map(|o| o.values))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;
	
	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "LrsiStream")]
pub struct LrsiStreamPy {
	stream: LrsiStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl LrsiStreamPy {
	#[new]
	fn new(alpha: f64) -> PyResult<Self> {
		let params = LrsiParams { alpha: Some(alpha) };
		let stream = LrsiStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(LrsiStreamPy { stream })
	}

	fn update(&mut self, high: f64, low: f64) -> Option<f64> {
		let price = (high + low) / 2.0;
		self.stream.update(price)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "lrsi_batch")]
#[pyo3(signature = (high, low, alpha_range, kernel=None))]
pub fn lrsi_batch_py<'py>(
	py: Python<'py>,
	high: PyReadonlyArray1<'py, f64>,
	low: PyReadonlyArray1<'py, f64>,
	alpha_range: (f64, f64, f64),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
	let high_slice = high.as_slice()?;
	let low_slice = low.as_slice()?;
	
	let sweep = LrsiBatchRange { alpha: alpha_range };
	
	let combos = expand_grid(&sweep);
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
			lrsi_batch_inner_into(high_slice, low_slice, &sweep, simd, true, slice_out)
		})
		.map_err(|e| PyValueError::new_err(e.to_string()))?;
	
	let dict = PyDict::new(py);
	dict.set_item("values", out_arr.reshape((rows, cols))?)?;
	dict.set_item(
		"alphas",
		combos
			.iter()
			.map(|p| p.alpha.unwrap())
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;
	
	Ok(dict)
}
