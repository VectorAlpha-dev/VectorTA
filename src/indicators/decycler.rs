//! # Ehlers Simple Decycler
//!
//! A noise-reduction filter based on John Ehlers's High-Pass Filter concept.
//! Removes high-frequency components from the original data, leaving a "decycled" output
//! that helps highlight underlying trends.
//!
//! ## Parameters
//! - **hp_period**: Window size used for the embedded high-pass filter (minimum of 2). Defaults to 125.
//! - **k**: Frequency coefficient for the high-pass filter. Defaults to 0.707.
//!
//! ## Errors
//! - **EmptyData**: decycler: Input data slice is empty.
//! - **InvalidPeriod**: decycler: `hp_period` is zero, less than 2, or exceeds the data length.
//! - **NotEnoughValidData**: decycler: Fewer than `hp_period` valid (non-`NaN`) data points remain after the first valid index.
//! - **AllValuesNaN**: decycler: All input data values are `NaN`.
//! - **InvalidK**: decycler: `k` is non-positive or NaN.
//!
//! ## Returns
//! - **`Ok(DecyclerOutput)`** on success, containing a `Vec<f64>` matching the input length.
//! - **`Err(DecyclerError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
	alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, 
	init_matrix_prefixes, make_uninit_matrix,
};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

impl<'a> AsRef<[f64]> for DecyclerInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			DecyclerData::Slice(slice) => slice,
			DecyclerData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

#[derive(Debug, Clone)]
pub enum DecyclerData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct DecyclerOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct DecyclerParams {
	pub hp_period: Option<usize>,
	pub k: Option<f64>,
}

impl Default for DecyclerParams {
	fn default() -> Self {
		Self {
			hp_period: Some(125),
			k: Some(0.707),
		}
	}
}

#[derive(Debug, Clone)]
pub struct DecyclerInput<'a> {
	pub data: DecyclerData<'a>,
	pub params: DecyclerParams,
}

impl<'a> DecyclerInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: DecyclerParams) -> Self {
		Self {
			data: DecyclerData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slice(sl: &'a [f64], p: DecyclerParams) -> Self {
		Self {
			data: DecyclerData::Slice(sl),
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", DecyclerParams::default())
	}
	#[inline]
	pub fn get_hp_period(&self) -> usize {
		self.params.hp_period.unwrap_or(125)
	}
	#[inline]
	pub fn get_k(&self) -> f64 {
		self.params.k.unwrap_or(0.707)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct DecyclerBuilder {
	hp_period: Option<usize>,
	k: Option<f64>,
	kernel: Kernel,
}

impl Default for DecyclerBuilder {
	fn default() -> Self {
		Self {
			hp_period: None,
			k: None,
			kernel: Kernel::Auto,
		}
	}
}

impl DecyclerBuilder {
	#[inline(always)]
	pub fn new() -> Self {
		Self::default()
	}
	#[inline(always)]
	pub fn hp_period(mut self, n: usize) -> Self {
		self.hp_period = Some(n);
		self
	}
	#[inline(always)]
	pub fn k(mut self, x: f64) -> Self {
		self.k = Some(x);
		self
	}
	#[inline(always)]
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline(always)]
	pub fn apply(self, c: &Candles) -> Result<DecyclerOutput, DecyclerError> {
		let p = DecyclerParams {
			hp_period: self.hp_period,
			k: self.k,
		};
		let i = DecyclerInput::from_candles(c, "close", p);
		decycler_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slice(self, d: &[f64]) -> Result<DecyclerOutput, DecyclerError> {
		let p = DecyclerParams {
			hp_period: self.hp_period,
			k: self.k,
		};
		let i = DecyclerInput::from_slice(d, p);
		decycler_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<DecyclerStream, DecyclerError> {
		let p = DecyclerParams {
			hp_period: self.hp_period,
			k: self.k,
		};
		DecyclerStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum DecyclerError {
	#[error("decycler: Empty data provided for Decycler.")]
	EmptyData,
	#[error("decycler: Invalid period: hp_period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("decycler: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("decycler: All values are NaN.")]
	AllValuesNaN,
	#[error("decycler: Invalid k: k = {k}")]
	InvalidK { k: f64 },
}

#[inline]
pub fn decycler(input: &DecyclerInput) -> Result<DecyclerOutput, DecyclerError> {
	decycler_with_kernel(input, Kernel::Auto)
}

/// Write directly to output slice - no allocations
#[cfg(any(feature = "python", feature = "wasm"))]
#[inline]
pub fn decycler_into_slice(
	out: &mut [f64],
	input: &DecyclerInput,
	kernel: Kernel,
) -> Result<(), DecyclerError> {
	let data: &[f64] = input.as_ref();
	
	if data.is_empty() {
		return Err(DecyclerError::EmptyData);
	}
	if out.len() != data.len() {
		return Err(DecyclerError::InvalidPeriod {
			period: out.len(),
			data_len: data.len(),
		});
	}
	
	let hp_period = input.get_hp_period();
	let k = input.get_k();
	
	if hp_period < 2 || hp_period > data.len() {
		return Err(DecyclerError::InvalidPeriod {
			period: hp_period,
			data_len: data.len(),
		});
	}
	if !(k.is_finite()) || k <= 0.0 {
		return Err(DecyclerError::InvalidK { k });
	}
	
	let first = data
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(DecyclerError::AllValuesNaN)?;
	if data.len() - first < hp_period {
		return Err(DecyclerError::NotEnoughValidData {
			needed: hp_period,
			valid: data.len() - first,
		});
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	// Compute directly into the output slice
	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => decycler_scalar_into(data, hp_period, k, first, out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => decycler_scalar_into(data, hp_period, k, first, out), // AVX2 not implemented, fallback
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => decycler_scalar_into(data, hp_period, k, first, out), // AVX512 not implemented, fallback
			_ => unreachable!(),
		}
	}?;

	// Fill warmup period with NaN
	let warmup_period = first + 2;
	for v in &mut out[..warmup_period] {
		*v = f64::NAN;
	}

	Ok(())
}

#[inline]
unsafe fn decycler_scalar_into(data: &[f64], hp_period: usize, k: f64, first: usize, out: &mut [f64]) -> Result<(), DecyclerError> {
	use std::f64::consts::PI;
	
	// Initialize NaN prefix
	for i in 0..first {
		out[i] = f64::NAN;
	}
	
	// We only need to keep track of the last 2 hp values
	let mut hp_prev2 = 0.0;
	let mut hp_prev1 = 0.0;
	
	let angle = 2.0 * PI * k / (hp_period as f64);
	let sin_val = angle.sin();
	let cos_val = angle.cos();
	let alpha = 1.0 + ((sin_val - 1.0) / cos_val);
	let one_minus_alpha_half = 1.0 - alpha / 2.0;
	let c = one_minus_alpha_half * one_minus_alpha_half;
	let one_minus_alpha = 1.0 - alpha;
	let one_minus_alpha_sq = one_minus_alpha * one_minus_alpha;

	if data.len() > first {
		let hp_val = data[first];
		hp_prev2 = hp_val;
		out[first] = data[first] - hp_val;
	}
	if data.len() > (first + 1) {
		let hp_val = data[first + 1];
		hp_prev1 = hp_val;
		out[first + 1] = data[first + 1] - hp_val;
	}
	for i in (first + 2)..data.len() {
		let current = data[i];
		let prev1 = data[i - 1];
		let prev2 = data[i - 2];
		
		let hp_val = c * current - 2.0 * c * prev1 + c * prev2 + 2.0 * one_minus_alpha * hp_prev1
			- one_minus_alpha_sq * hp_prev2;
		
		// Shift the hp values
		hp_prev2 = hp_prev1;
		hp_prev1 = hp_val;
		
		out[i] = current - hp_val;
	}
	Ok(())
}

pub fn decycler_with_kernel(input: &DecyclerInput, kernel: Kernel) -> Result<DecyclerOutput, DecyclerError> {
	let data: &[f64] = match &input.data {
		DecyclerData::Candles { candles, source } => source_type(candles, source),
		DecyclerData::Slice(sl) => sl,
	};

	if data.is_empty() {
		return Err(DecyclerError::EmptyData);
	}
	let hp_period = input.get_hp_period();
	let k = input.get_k();
	if hp_period < 2 || hp_period > data.len() {
		return Err(DecyclerError::InvalidPeriod {
			period: hp_period,
			data_len: data.len(),
		});
	}
	if !(k.is_finite()) || k <= 0.0 {
		return Err(DecyclerError::InvalidK { k });
	}
	let first = data
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(DecyclerError::AllValuesNaN)?;
	if data.len() - first < hp_period {
		return Err(DecyclerError::NotEnoughValidData {
			needed: hp_period,
			valid: data.len() - first,
		});
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => decycler_scalar(data, hp_period, k, first),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => decycler_avx2(data, hp_period, k, first),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => decycler_avx512(data, hp_period, k, first),
			_ => unreachable!(),
		}
	}
}

#[inline]
pub fn decycler_scalar(data: &[f64], hp_period: usize, k: f64, first: usize) -> Result<DecyclerOutput, DecyclerError> {
	use std::f64::consts::PI;
	
	// Use alloc_with_nan_prefix to avoid unnecessary initialization
	let warmup_period = first + 2; // Need at least 2 values for the calculation
	let mut out = alloc_with_nan_prefix(data.len(), warmup_period);
	
	// We only need to keep track of the last 2 hp values
	let mut hp_prev2 = 0.0;
	let mut hp_prev1 = 0.0;
	
	let angle = 2.0 * PI * k / (hp_period as f64);
	let sin_val = angle.sin();
	let cos_val = angle.cos();
	let alpha = 1.0 + ((sin_val - 1.0) / cos_val);
	let one_minus_alpha_half = 1.0 - alpha / 2.0;
	let c = one_minus_alpha_half * one_minus_alpha_half;
	let one_minus_alpha = 1.0 - alpha;
	let one_minus_alpha_sq = one_minus_alpha * one_minus_alpha;

	if data.len() > first {
		let hp_val = data[first];
		hp_prev2 = hp_val;
		out[first] = data[first] - hp_val;
	}
	if data.len() > (first + 1) {
		let hp_val = data[first + 1];
		hp_prev1 = hp_val;
		out[first + 1] = data[first + 1] - hp_val;
	}
	for i in (first + 2)..data.len() {
		let current = data[i];
		let prev1 = data[i - 1];
		let prev2 = data[i - 2];
		
		let hp_val = c * current - 2.0 * c * prev1 + c * prev2 + 2.0 * one_minus_alpha * hp_prev1
			- one_minus_alpha_sq * hp_prev2;
		
		// Shift the hp values
		hp_prev2 = hp_prev1;
		hp_prev1 = hp_val;
		
		out[i] = current - hp_val;
	}
	Ok(DecyclerOutput { values: out })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn decycler_avx512(data: &[f64], hp_period: usize, k: f64, first: usize) -> Result<DecyclerOutput, DecyclerError> {
	// STUB: AVX512 not implemented, fallback to scalar
	decycler_scalar(data, hp_period, k, first)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn decycler_avx2(data: &[f64], hp_period: usize, k: f64, first: usize) -> Result<DecyclerOutput, DecyclerError> {
	// STUB: AVX2 not implemented, fallback to scalar
	decycler_scalar(data, hp_period, k, first)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn decycler_avx512_short(
	data: &[f64],
	hp_period: usize,
	k: f64,
	first: usize,
) -> Result<DecyclerOutput, DecyclerError> {
	decycler_scalar(data, hp_period, k, first)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn decycler_avx512_long(
	data: &[f64],
	hp_period: usize,
	k: f64,
	first: usize,
) -> Result<DecyclerOutput, DecyclerError> {
	decycler_scalar(data, hp_period, k, first)
}

#[inline]
pub fn decycler_batch_with_kernel(
	data: &[f64],
	sweep: &DecyclerBatchRange,
	k: Kernel,
) -> Result<DecyclerBatchOutput, DecyclerError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(DecyclerError::InvalidPeriod { period: 0, data_len: 0 }),
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	decycler_batch_par_slice(data, sweep, simd)
}

#[derive(Debug, Clone)]
pub struct DecyclerStream {
	hp_period: usize,
	k: f64,
	buffer: Vec<f64>,
	hp: Vec<f64>,
	idx: usize,
	filled: bool,
	angle: f64,
	c: f64,
	one_minus_alpha: f64,
	one_minus_alpha_sq: f64,
}

impl DecyclerStream {
	pub fn try_new(params: DecyclerParams) -> Result<Self, DecyclerError> {
		let hp_period = params.hp_period.unwrap_or(125);
		let k = params.k.unwrap_or(0.707);
		if hp_period < 2 {
			return Err(DecyclerError::InvalidPeriod {
				period: hp_period,
				data_len: 0,
			});
		}
		if !(k.is_finite()) || k <= 0.0 {
			return Err(DecyclerError::InvalidK { k });
		}
		use std::f64::consts::PI;
		let angle = 2.0 * PI * k / (hp_period as f64);
		let sin_val = angle.sin();
		let cos_val = angle.cos();
		let alpha = 1.0 + ((sin_val - 1.0) / cos_val);
		let one_minus_alpha_half = 1.0 - alpha / 2.0;
		let c = one_minus_alpha_half * one_minus_alpha_half;
		let one_minus_alpha = 1.0 - alpha;
		let one_minus_alpha_sq = one_minus_alpha * one_minus_alpha;
		Ok(Self {
			hp_period,
			k,
			buffer: vec![f64::NAN; hp_period],
			hp: vec![0.0; hp_period],
			idx: 0,
			filled: false,
			angle,
			c,
			one_minus_alpha,
			one_minus_alpha_sq,
		})
	}
	#[inline(always)]
	pub fn update(&mut self, value: f64) -> Option<f64> {
		let idx0 = self.idx % self.hp_period;
		self.buffer[idx0] = value;
		if self.idx == 0 {
			self.hp[idx0] = value;
		} else if self.idx == 1 {
			self.hp[idx0] = value;
		} else {
			let prev1 = self.buffer[(self.idx + self.hp_period - 1) % self.hp_period];
			let prev2 = self.buffer[(self.idx + self.hp_period - 2) % self.hp_period];
			let hp_prev1 = self.hp[(self.idx + self.hp_period - 1) % self.hp_period];
			let hp_prev2 = self.hp[(self.idx + self.hp_period - 2) % self.hp_period];
			let val = self.c * value - 2.0 * self.c * prev1 + self.c * prev2 + 2.0 * self.one_minus_alpha * hp_prev1
				- self.one_minus_alpha_sq * hp_prev2;
			self.hp[idx0] = val;
		}
		let out = if self.idx < 2 {
			value - self.hp[idx0]
		} else {
			value - self.hp[idx0]
		};
		self.idx += 1;
		if self.idx >= self.hp_period {
			self.filled = true;
		}
		Some(out)
	}
}

#[derive(Clone, Debug)]
pub struct DecyclerBatchRange {
	pub hp_period: (usize, usize, usize),
	pub k: (f64, f64, f64),
}

impl Default for DecyclerBatchRange {
	fn default() -> Self {
		Self {
			hp_period: (125, 125, 0),
			k: (0.707, 0.707, 0.0),
		}
	}
}

#[derive(Clone, Debug, Default)]
pub struct DecyclerBatchBuilder {
	range: DecyclerBatchRange,
	kernel: Kernel,
}

impl DecyclerBatchBuilder {
	pub fn new() -> Self {
		Self::default()
	}
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline]
	pub fn hp_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.hp_period = (start, end, step);
		self
	}
	#[inline]
	pub fn hp_period_static(mut self, p: usize) -> Self {
		self.range.hp_period = (p, p, 0);
		self
	}
	#[inline]
	pub fn k_range(mut self, start: f64, end: f64, step: f64) -> Self {
		self.range.k = (start, end, step);
		self
	}
	#[inline]
	pub fn k_static(mut self, v: f64) -> Self {
		self.range.k = (v, v, 0.0);
		self
	}
	pub fn apply_slice(self, data: &[f64]) -> Result<DecyclerBatchOutput, DecyclerError> {
		decycler_batch_with_kernel(data, &self.range, self.kernel)
	}
	pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<DecyclerBatchOutput, DecyclerError> {
		DecyclerBatchBuilder::new().kernel(k).apply_slice(data)
	}
	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<DecyclerBatchOutput, DecyclerError> {
		let slice = source_type(c, src);
		self.apply_slice(slice)
	}
	pub fn with_default_candles(c: &Candles) -> Result<DecyclerBatchOutput, DecyclerError> {
		DecyclerBatchBuilder::new()
			.kernel(Kernel::Auto)
			.apply_candles(c, "close")
	}
}

#[derive(Clone, Debug)]
pub struct DecyclerBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<DecyclerParams>,
	pub rows: usize,
	pub cols: usize,
}
impl DecyclerBatchOutput {
	pub fn row_for_params(&self, p: &DecyclerParams) -> Option<usize> {
		self.combos.iter().position(|c| {
			c.hp_period.unwrap_or(125) == p.hp_period.unwrap_or(125)
				&& (c.k.unwrap_or(0.707) - p.k.unwrap_or(0.707)).abs() < 1e-12
		})
	}
	pub fn values_for(&self, p: &DecyclerParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &DecyclerBatchRange) -> Vec<DecyclerParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
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
	let hp_periods = axis_usize(r.hp_period);
	let ks = axis_f64(r.k);
	let mut out = Vec::with_capacity(hp_periods.len() * ks.len());
	for &p in &hp_periods {
		for &k in &ks {
			out.push(DecyclerParams {
				hp_period: Some(p),
				k: Some(k),
			});
		}
	}
	out
}

#[inline(always)]
pub fn decycler_batch_slice(
	data: &[f64],
	sweep: &DecyclerBatchRange,
	kern: Kernel,
) -> Result<DecyclerBatchOutput, DecyclerError> {
	decycler_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn decycler_batch_par_slice(
	data: &[f64],
	sweep: &DecyclerBatchRange,
	kern: Kernel,
) -> Result<DecyclerBatchOutput, DecyclerError> {
	decycler_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn decycler_batch_inner(
	data: &[f64],
	sweep: &DecyclerBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<DecyclerBatchOutput, DecyclerError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(DecyclerError::InvalidPeriod { period: 0, data_len: 0 });
	}
	let first = data
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(DecyclerError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.hp_period.unwrap()).max().unwrap();
	if data.len() - first < max_p {
		return Err(DecyclerError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}
	let rows = combos.len();
	let cols = data.len();
	
	// Use matrix helpers instead of direct allocation
	let mut buffer = make_uninit_matrix(rows, cols);
	
	// Calculate warmup periods for each row
	let warmup_periods: Vec<usize> = combos.iter()
		.map(|_| first + 2)  // Each row needs warmup of first + 2
		.collect();
	
	// Initialize NaN prefixes
	init_matrix_prefixes(&mut buffer, cols, &warmup_periods);
	
	// Convert to initialized memory for processing
	let mut values = unsafe {
		use std::mem::ManuallyDrop;
		let mut v = ManuallyDrop::new(buffer);
		Vec::from_raw_parts(v.as_mut_ptr() as *mut f64, v.len(), v.capacity())
	};

	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let hp_period = combos[row].hp_period.unwrap();
		let k = combos[row].k.unwrap();
		match kern {
			Kernel::Scalar => decycler_row_scalar(data, first, hp_period, k, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => decycler_row_avx2(data, first, hp_period, k, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => decycler_row_avx512(data, first, hp_period, k, out_row),
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

	Ok(DecyclerBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
unsafe fn decycler_row_scalar(data: &[f64], first: usize, hp_period: usize, k: f64, out: &mut [f64]) {
	use std::f64::consts::PI;
	
	// We only need to keep track of the last 2 hp values
	let mut hp_prev2 = 0.0;
	let mut hp_prev1 = 0.0;
	
	let angle = 2.0 * PI * k / (hp_period as f64);
	let sin_val = angle.sin();
	let cos_val = angle.cos();
	let alpha = 1.0 + ((sin_val - 1.0) / cos_val);
	let one_minus_alpha_half = 1.0 - alpha / 2.0;
	let c = one_minus_alpha_half * one_minus_alpha_half;
	let one_minus_alpha = 1.0 - alpha;
	let one_minus_alpha_sq = one_minus_alpha * one_minus_alpha;

	if data.len() > first {
		let hp_val = data[first];
		hp_prev2 = hp_val;
		out[first] = data[first] - hp_val;
	}
	if data.len() > (first + 1) {
		let hp_val = data[first + 1];
		hp_prev1 = hp_val;
		out[first + 1] = data[first + 1] - hp_val;
	}
	for i in (first + 2)..data.len() {
		let current = data[i];
		let prev1 = data[i - 1];
		let prev2 = data[i - 2];
		
		let hp_val = c * current - 2.0 * c * prev1 + c * prev2 + 2.0 * one_minus_alpha * hp_prev1
			- one_minus_alpha_sq * hp_prev2;
		
		// Shift the hp values
		hp_prev2 = hp_prev1;
		hp_prev1 = hp_val;
		
		out[i] = current - hp_val;
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
unsafe fn decycler_row_avx2(data: &[f64], first: usize, hp_period: usize, k: f64, out: &mut [f64]) {
	decycler_row_scalar(data, first, hp_period, k, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn decycler_row_avx512(data: &[f64], first: usize, hp_period: usize, k: f64, out: &mut [f64]) {
	if hp_period <= 32 {
		decycler_row_avx512_short(data, first, hp_period, k, out)
	} else {
		decycler_row_avx512_long(data, first, hp_period, k, out)
	}
	_mm_sfence();
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn decycler_row_avx512_short(data: &[f64], first: usize, hp_period: usize, k: f64, out: &mut [f64]) {
	decycler_row_scalar(data, first, hp_period, k, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn decycler_row_avx512_long(data: &[f64], first: usize, hp_period: usize, k: f64, out: &mut [f64]) {
	decycler_row_scalar(data, first, hp_period, k, out)
}

#[inline(always)]
pub fn expand_grid_decycler(r: &DecyclerBatchRange) -> Vec<DecyclerParams> {
	expand_grid(r)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;
	use crate::utilities::enums::Kernel;

	fn check_decycler_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let default_params = DecyclerParams {
			hp_period: None,
			k: None,
		};
		let input_default = DecyclerInput::from_candles(&candles, "close", default_params);
		let output_default = decycler_with_kernel(&input_default, kernel)?;
		assert_eq!(output_default.values.len(), candles.close.len());

		let params_hp_50 = DecyclerParams {
			hp_period: Some(50),
			k: None,
		};
		let input_hp_50 = DecyclerInput::from_candles(&candles, "hl2", params_hp_50);
		let output_hp_50 = decycler_with_kernel(&input_hp_50, kernel)?;
		assert_eq!(output_hp_50.values.len(), candles.close.len());

		let params_custom = DecyclerParams {
			hp_period: Some(30),
			k: None,
		};
		let input_custom = DecyclerInput::from_candles(&candles, "hlc3", params_custom);
		let output_custom = decycler_with_kernel(&input_custom, kernel)?;
		assert_eq!(output_custom.values.len(), candles.close.len());
		Ok(())
	}

	fn check_decycler_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let close_prices = candles
			.select_candle_field("close")
			.expect("Failed to extract close prices");
		let params = DecyclerParams {
			hp_period: Some(125),
			k: None,
		};
		let input = DecyclerInput::from_candles(&candles, "close", params);
		let decycler_result = decycler_with_kernel(&input, kernel)?;
		assert_eq!(decycler_result.values.len(), close_prices.len());
		let test_values = [
			60289.96384058519,
			60204.010366691065,
			60114.255563805666,
			60028.535266555904,
			59934.26876964316,
		];
		assert!(decycler_result.values.len() >= test_values.len());
		let start_index = decycler_result.values.len() - test_values.len();
		let result_last_values = &decycler_result.values[start_index..];
		for (i, &value) in result_last_values.iter().enumerate() {
			let expected_value = test_values[i];
			assert!(
				(value - expected_value).abs() < 1e-6,
				"Decycler mismatch at index {}: expected {}, got {}",
				i,
				expected_value,
				value
			);
		}
		Ok(())
	}

	fn check_decycler_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let input_data = [10.0, 20.0, 30.0];
		let params = DecyclerParams {
			hp_period: Some(0),
			k: None,
		};
		let input = DecyclerInput::from_slice(&input_data, params);
		let result = decycler_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}

	fn check_decycler_period_exceed_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let input_data = [10.0, 20.0, 30.0];
		let params = DecyclerParams {
			hp_period: Some(10),
			k: None,
		};
		let input = DecyclerInput::from_slice(&input_data, params);
		let result = decycler_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}

	fn check_decycler_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let input_data = [42.0];
		let params = DecyclerParams {
			hp_period: Some(2),
			k: None,
		};
		let input = DecyclerInput::from_slice(&input_data, params);
		let result = decycler_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}

	fn check_decycler_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let first_params = DecyclerParams {
			hp_period: Some(30),
			k: None,
		};
		let first_input = DecyclerInput::from_candles(&candles, "close", first_params);
		let first_result = decycler_with_kernel(&first_input, kernel)?;
		assert_eq!(first_result.values.len(), candles.close.len());
		let second_params = DecyclerParams {
			hp_period: Some(30),
			k: None,
		};
		let second_input = DecyclerInput::from_slice(&first_result.values, second_params);
		let second_result = decycler_with_kernel(&second_input, kernel)?;
		assert_eq!(second_result.values.len(), first_result.values.len());
		Ok(())
	}

	fn check_decycler_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let close_prices = &candles.close;
		let period = 125;
		let params = DecyclerParams {
			hp_period: Some(period),
			k: None,
		};
		let input = DecyclerInput::from_candles(&candles, "close", params);
		let decycler_result = decycler_with_kernel(&input, kernel)?;
		assert_eq!(decycler_result.values.len(), close_prices.len());
		if decycler_result.values.len() > 240 {
			for i in 240..decycler_result.values.len() {
				assert!(
					!decycler_result.values[i].is_nan(),
					"Expected no NaN after index 240, found NaN at {}",
					i
				);
			}
		}
		Ok(())
	}

	#[cfg(debug_assertions)]
	fn check_decycler_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let test_params = vec![
			DecyclerParams::default(),
			DecyclerParams {
				hp_period: Some(2),
				k: Some(0.1),
			},
			DecyclerParams {
				hp_period: Some(2),
				k: Some(0.707),
			},
			DecyclerParams {
				hp_period: Some(2),
				k: Some(2.0),
			},
			DecyclerParams {
				hp_period: Some(5),
				k: Some(0.5),
			},
			DecyclerParams {
				hp_period: Some(5),
				k: Some(1.0),
			},
			DecyclerParams {
				hp_period: Some(10),
				k: Some(0.707),
			},
			DecyclerParams {
				hp_period: Some(20),
				k: Some(0.3),
			},
			DecyclerParams {
				hp_period: Some(20),
				k: Some(0.707),
			},
			DecyclerParams {
				hp_period: Some(20),
				k: Some(1.5),
			},
			DecyclerParams {
				hp_period: Some(50),
				k: Some(0.707),
			},
			DecyclerParams {
				hp_period: Some(75),
				k: Some(0.9),
			},
			DecyclerParams {
				hp_period: Some(100),
				k: Some(0.707),
			},
			DecyclerParams {
				hp_period: Some(100),
				k: Some(2.5),
			},
			DecyclerParams {
				hp_period: Some(200),
				k: Some(0.707),
			},
			DecyclerParams {
				hp_period: Some(200),
				k: Some(5.0),
			},
			DecyclerParams {
				hp_period: Some(500),
				k: Some(0.707),
			},
		];

		for (param_idx, params) in test_params.iter().enumerate() {
			let input = DecyclerInput::from_candles(&candles, "close", params.clone());
			let output = decycler_with_kernel(&input, kernel)?;

			for (i, &val) in output.values.iter().enumerate() {
				if val.is_nan() {
					continue;
				}

				let bits = val.to_bits();

				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: hp_period={}, k={} (param set {})",
						test_name,
						val,
						bits,
						i,
						params.hp_period.unwrap_or(125),
						params.k.unwrap_or(0.707),
						param_idx
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: hp_period={}, k={} (param set {})",
						test_name,
						val,
						bits,
						i,
						params.hp_period.unwrap_or(125),
						params.k.unwrap_or(0.707),
						param_idx
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: hp_period={}, k={} (param set {})",
						test_name,
						val,
						bits,
						i,
						params.hp_period.unwrap_or(125),
						params.k.unwrap_or(0.707),
						param_idx
					);
				}
			}
		}

		Ok(())
	}

	#[cfg(not(debug_assertions))]
	fn check_decycler_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(())
	}

	macro_rules! generate_all_decycler_tests {
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
	generate_all_decycler_tests!(
		check_decycler_partial_params,
		check_decycler_accuracy,
		check_decycler_zero_period,
		check_decycler_period_exceed_length,
		check_decycler_very_small_dataset,
		check_decycler_reinput,
		check_decycler_nan_handling,
		check_decycler_no_poison
	);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let output = DecyclerBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;
		let def = DecyclerParams::default();
		let row = output.values_for(&def).expect("default row missing");
		assert_eq!(row.len(), c.close.len());
		let expected = [
			60289.96384058519,
			60204.010366691065,
			60114.255563805666,
			60028.535266555904,
			59934.26876964316,
		];
		let start = row.len() - 5;
		for (i, &v) in row[start..].iter().enumerate() {
			assert!(
				(v - expected[i]).abs() < 1e-6,
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
					let _ = $fn_name(stringify!([<$fn_name _auto_detect>]),
									 Kernel::Auto);
				}
			}
		};
	}
	#[cfg(debug_assertions)]
	fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let test_configs = vec![
			(2, 10, 2, 0.1, 1.0, 0.3),
			(5, 25, 5, 0.707, 0.707, 0.0),
			(10, 10, 0, 0.5, 2.0, 0.5),
			(2, 5, 1, 0.3, 0.9, 0.2),
			(30, 60, 15, 0.707, 0.707, 0.0),
			(50, 100, 25, 0.5, 1.5, 0.5),
			(75, 125, 25, 0.707, 2.0, 0.5),
			(100, 200, 50, 0.5, 1.0, 0.25),
			(200, 500, 100, 0.707, 0.707, 0.0),
		];

		for (cfg_idx, &(hp_start, hp_end, hp_step, k_start, k_end, k_step)) in
			test_configs.iter().enumerate()
		{
			let output = DecyclerBatchBuilder::new()
				.kernel(kernel)
				.hp_period_range(hp_start, hp_end, hp_step)
				.k_range(k_start, k_end, k_step)
				.apply_candles(&c, "close")?;

			for (idx, &val) in output.values.iter().enumerate() {
				if val.is_nan() {
					continue;
				}

				let bits = val.to_bits();
				let row = idx / output.cols;
				let col = idx % output.cols;
				let combo = &output.combos[row];

				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
						at row {} col {} (flat index {}) with params: hp_period={}, k={}",
						test,
						cfg_idx,
						val,
						bits,
						row,
						col,
						idx,
						combo.hp_period.unwrap_or(125),
						combo.k.unwrap_or(0.707)
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						at row {} col {} (flat index {}) with params: hp_period={}, k={}",
						test,
						cfg_idx,
						val,
						bits,
						row,
						col,
						idx,
						combo.hp_period.unwrap_or(125),
						combo.k.unwrap_or(0.707)
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						at row {} col {} (flat index {}) with params: hp_period={}, k={}",
						test,
						cfg_idx,
						val,
						bits,
						row,
						col,
						idx,
						combo.hp_period.unwrap_or(125),
						combo.k.unwrap_or(0.707)
					);
				}
			}
		}

		Ok(())
	}

	#[cfg(not(debug_assertions))]
	fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(())
	}

	gen_batch_tests!(check_batch_default_row);
	gen_batch_tests!(check_batch_no_poison);

	#[cfg(feature = "proptest")]
	#[allow(clippy::float_cmp)]
	fn check_decycler_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		use proptest::prelude::*;
		skip_if_unsupported!(kernel, test_name);

		// Generate realistic test data with valid parameters
		// hp_period range: 2-100 (must be >= 2)
		// k range: 0.1-5.0 (must be positive)
		let strat = (2usize..=100)
			.prop_flat_map(|period| {
				(
					prop::collection::vec(
						(-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
						period..400,
					),
					Just(period),
					0.1f64..5.0f64,  // k must be positive
				)
			});

		proptest::test_runner::TestRunner::default()
			.run(&strat, |(data, hp_period, k)| {
				let params = DecyclerParams {
					hp_period: Some(hp_period),
					k: Some(k),
				};
				let input = DecyclerInput::from_slice(&data, params);

				// Test with specified kernel
				let DecyclerOutput { values: out } = decycler_with_kernel(&input, kernel).unwrap();
				
				// Also test against scalar reference for kernel consistency
				let DecyclerOutput { values: ref_out } = decycler_with_kernel(&input, Kernel::Scalar).unwrap();

				// Find first non-NaN value
				let first = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
				let warmup_period = first + 2;

				// Property 1: Warmup period should have NaN values
				for i in 0..warmup_period.min(out.len()) {
					prop_assert!(
						out[i].is_nan(),
						"Expected NaN during warmup at index {}, got {}",
						i,
						out[i]
					);
				}

				// Property 2: After warmup, values should be finite
				for i in warmup_period..out.len() {
					prop_assert!(
						out[i].is_finite(),
						"Expected finite value after warmup at index {}, got {}",
						i,
						out[i]
					);
				}

				// Property 3: Kernel consistency - results should match across kernels
				for i in 0..out.len() {
					let y = out[i];
					let r = ref_out[i];
					
					if y.is_nan() && r.is_nan() {
						continue;  // Both NaN is ok
					}
					
					// Use ULP comparison for floating point equality
					let y_bits = y.to_bits();
					let r_bits = r.to_bits();
					let ulp_diff = if y_bits > r_bits {
						y_bits - r_bits
					} else {
						r_bits - y_bits
					};
					
					prop_assert!(
						ulp_diff <= 3,
						"Kernel mismatch at index {}: {} ({} bits) vs {} ({} bits), ULP diff: {}",
						i, y, y_bits, r, r_bits, ulp_diff
					);
				}

				// Property 4: For constant input, high-pass filter output should be close to 0
				// Decycler = input - high_pass, so for DC signal, high_pass → input, decycler → 0
				if data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-9) && !data.is_empty() {
					// After sufficient warmup for filter to stabilize
					if out.len() > warmup_period + hp_period * 2 {
						let check_idx = out.len() - 1;
						prop_assert!(
							out[check_idx].abs() <= 1e-6 || out[check_idx].abs() <= data[first].abs() * 1e-6,
							"Constant input {} should produce output close to 0 (high-pass filtered), got {}",
							data[first],
							out[check_idx]
						);
					}
				}

				// Property 5: Bounded output - decycler removes components, shouldn't amplify much
				// After warmup, output magnitude should be bounded relative to input
				if out.len() > warmup_period + hp_period {
					let start_check = warmup_period + hp_period;
					for i in start_check..out.len() {
						let input_mag = data[i].abs();
						let output_mag = out[i].abs();
						// Allow some tolerance for numerical precision and filter characteristics
						// Decycler can have transient responses but shouldn't amplify significantly
						prop_assert!(
							output_mag <= input_mag * 2.0 + 1.0,
							"Output magnitude {} exceeds reasonable bound for input {} at index {}",
							output_mag,
							input_mag,
							i
						);
					}
				}

				// Property 6: For monotonic trends, decycler should reduce the trend
				// Check if data has a strong linear trend
				if data.len() > warmup_period + 10 {
					let trend_start = warmup_period;
					let trend_end = data.len().min(warmup_period + 50);
					let is_monotonic_increasing = data[trend_start..trend_end].windows(2)
						.all(|w| w[1] >= w[0] - 1e-9);
					let is_monotonic_decreasing = data[trend_start..trend_end].windows(2)
						.all(|w| w[1] <= w[0] + 1e-9);
					
					if (is_monotonic_increasing || is_monotonic_decreasing) && trend_end > trend_start + 5 {
						// For strong trends, the decycler output should have less variance than input
						// as it removes the low-frequency trend component
						let input_range = data[trend_start..trend_end].iter()
							.fold(f64::INFINITY, |a, &b| a.min(b))
							..=data[trend_start..trend_end].iter()
							.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
						let output_range = out[trend_start..trend_end].iter()
							.fold(f64::INFINITY, |a, &b| a.min(b))
							..=out[trend_start..trend_end].iter()
							.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
						
						let input_span = input_range.end() - input_range.start();
						let output_span = output_range.end() - output_range.start();
						
						// Output should have reduced range for monotonic trends
						// Allow some tolerance for edge effects
						if input_span > 1e-9 {
							prop_assert!(
								output_span <= input_span * 1.5,
								"Decycler should reduce trend variation: input span {}, output span {}",
								input_span,
								output_span
							);
						}
					}
				}

				// Property 7: Check for poison values (debug builds only)
				#[cfg(debug_assertions)]
				for (i, &val) in out.iter().enumerate() {
					if val.is_nan() {
						continue;
					}
					
					let bits = val.to_bits();
					prop_assert!(
						bits != 0x11111111_11111111 && 
						bits != 0x22222222_22222222 && 
						bits != 0x33333333_33333333,
						"Found poison value at index {}: {} (0x{:016X})",
						i, val, bits
					);
				}

				// Property 8: Edge cases - minimum period and extreme k values
				if hp_period == 2 {
					// Should still produce valid output with correct length
					prop_assert!(
						out.len() == data.len(),
						"Output length mismatch for period=2"
					);
				}
				
				// For extreme k values, output should still be bounded
				if k < 0.2 || k > 4.5 {
					for i in warmup_period..out.len() {
						prop_assert!(
							out[i].is_finite(),
							"Extreme k={} produced non-finite value at index {}",
							k,
							i
						);
					}
				}

				Ok(())
			})?;

		Ok(())
	}

	#[cfg(feature = "proptest")]
	generate_all_decycler_tests!(check_decycler_property);
}

#[cfg(feature = "python")]
#[pyfunction(name = "decycler")]
#[pyo3(signature = (data, hp_period=None, k=None, kernel=None))]
pub fn decycler_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	hp_period: Option<usize>,
	k: Option<f64>,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, false)?;
	let params = DecyclerParams { hp_period, k };
	let decycler_in = DecyclerInput::from_slice(slice_in, params);

	let result_vec: Vec<f64> = py
		.allow_threads(|| decycler_with_kernel(&decycler_in, kern).map(|o| o.values))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "DecyclerStream")]
pub struct DecyclerStreamPy {
	stream: DecyclerStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl DecyclerStreamPy {
	#[new]
	fn new(hp_period: Option<usize>, k: Option<f64>) -> PyResult<Self> {
		let params = DecyclerParams { hp_period, k };
		let stream = DecyclerStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(DecyclerStreamPy { stream })
	}

	fn update(&mut self, value: f64) -> Option<f64> {
		self.stream.update(value)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "decycler_batch")]
#[pyo3(signature = (data, hp_period_range, k_range, kernel=None))]
pub fn decycler_batch_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	hp_period_range: (usize, usize, usize),
	k_range: (f64, f64, f64),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;

	let slice_in = data.as_slice()?;

	let sweep = DecyclerBatchRange {
		hp_period: hp_period_range,
		k: k_range,
	};

	let combos = expand_grid(&sweep);
	let rows = combos.len();
	let cols = slice_in.len();

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
			decycler_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
		})
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	let dict = PyDict::new(py);
	dict.set_item("values", out_arr.reshape((rows, cols))?)?;
	dict.set_item(
		"hp_periods",
		combos
			.iter()
			.map(|p| p.hp_period.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;
	dict.set_item(
		"ks",
		combos
			.iter()
			.map(|p| p.k.unwrap())
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;
	dict.set_item("rows", rows)?;
	dict.set_item("cols", cols)?;

	Ok(dict.to_owned())
}

#[inline(always)]
fn decycler_batch_inner_into(
	data: &[f64],
	sweep: &DecyclerBatchRange,
	kern: Kernel,
	parallel: bool,
	out: &mut [f64],
) -> Result<Vec<DecyclerParams>, DecyclerError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(DecyclerError::InvalidPeriod { period: 0, data_len: 0 });
	}

	let first = data
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(DecyclerError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.hp_period.unwrap()).max().unwrap();
	if data.len() - first < max_p {
		return Err(DecyclerError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}

	let rows = combos.len();
	let cols = data.len();

	// Initialize NaN prefixes for each row based on warmup period
	for (row, _combo) in combos.iter().enumerate() {
		let warmup = first + 2;  // Each row needs warmup of first + 2
		let row_start = row * cols;
		for i in 0..warmup.min(cols) {
			out[row_start + i] = f64::NAN;
		}
	}

	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let hp_period = combos[row].hp_period.unwrap();
		let k = combos[row].k.unwrap();
		match kern {
			Kernel::Scalar => decycler_row_scalar(data, first, hp_period, k, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => decycler_row_avx2(data, first, hp_period, k, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => decycler_row_avx512(data, first, hp_period, k, out_row),
			_ => unreachable!(),
		}
	};

	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			out
				.par_chunks_mut(cols)
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

// =============================================================================
// WASM Bindings
// =============================================================================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn decycler_js(data: &[f64], hp_period: usize, k: f64) -> Result<Vec<f64>, JsValue> {
	let params = DecyclerParams {
		hp_period: Some(hp_period),
		k: Some(k),
	};
	let input = DecyclerInput::from_slice(data, params);
	
	let mut output = vec![0.0; data.len()]; // Single allocation
	decycler_into_slice(&mut output, &input, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn decycler_into(
	in_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	hp_period: usize,
	k: f64,
) -> Result<(), JsValue> {
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}
	
	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);
		let params = DecyclerParams {
			hp_period: Some(hp_period),
			k: Some(k),
		};
		let input = DecyclerInput::from_slice(data, params);
		
		if in_ptr == out_ptr as *const f64 { // CRITICAL: Aliasing check
			let mut temp = vec![0.0; len];
			decycler_into_slice(&mut temp, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			out.copy_from_slice(&temp);
		} else {
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			decycler_into_slice(out, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn decycler_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn decycler_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe { 
			let _ = Vec::from_raw_parts(ptr, len, len); 
		}
	}
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct DecyclerBatchConfig {
	pub hp_period_range: (usize, usize, usize),
	pub k_range: (f64, f64, f64),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct DecyclerBatchJsOutput {
	pub values: Vec<f64>,
	pub combos: Vec<DecyclerParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = decycler_batch)]
pub fn decycler_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	let config: DecyclerBatchConfig = serde_wasm_bindgen::from_value(config)
		.map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
	
	let sweep = DecyclerBatchRange {
		hp_period: config.hp_period_range,
		k: config.k_range,
	};
	
	match decycler_batch_with_kernel(data, &sweep, Kernel::Auto) {
		Ok(result) => {
			let output = DecyclerBatchJsOutput {
				values: result.values,
				combos: result.combos,
				rows: result.rows,
				cols: result.cols,
			};
			serde_wasm_bindgen::to_value(&output)
				.map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
		}
		Err(e) => Err(JsValue::from_str(&e.to_string())),
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn decycler_batch_into(
	in_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	hp_period_start: usize,
	hp_period_end: usize,
	hp_period_step: usize,
	k_start: f64,
	k_end: f64,
	k_step: f64,
) -> Result<usize, JsValue> {
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}
	
	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);
		let sweep = DecyclerBatchRange {
			hp_period: (hp_period_start, hp_period_end, hp_period_step),
			k: (k_start, k_end, k_step),
		};
		
		let combos = expand_grid(&sweep);
		let rows = combos.len();
		let cols = len;
		let total_size = rows * cols;
		
		let out = std::slice::from_raw_parts_mut(out_ptr, total_size);
		
		// Check if we have aliasing
		let data_ptr_range = in_ptr as usize..(in_ptr as usize + len * std::mem::size_of::<f64>());
		let out_ptr_range = out_ptr as usize..(out_ptr as usize + total_size * std::mem::size_of::<f64>());
		
		if data_ptr_range.start < out_ptr_range.end && out_ptr_range.start < data_ptr_range.end {
			// Ranges overlap - need to use temporary buffer
			let result = decycler_batch_with_kernel(data, &sweep, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			out.copy_from_slice(&result.values);
		} else {
			// No overlap - can write directly
			decycler_batch_inner_into(data, &sweep, detect_best_kernel(), true, out)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		
		Ok(rows)
	}
}
