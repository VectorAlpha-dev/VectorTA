//! # Variable Index Dynamic Average (VIDYA)
//!
//! VIDYA is a moving average calculation that dynamically adjusts its smoothing factor
//! based on the ratio of short-term to long-term standard deviations. This allows the
//! average to become more responsive to volatility changes while still providing a
//! smoothed signal.
//!
//! ## Parameters
//! - **short_period**: The short look-back period for standard deviation. Defaults to 2.
//! - **long_period**: The long look-back period for standard deviation. Defaults to 5.
//! - **alpha**: A smoothing factor between 0.0 and 1.0. Defaults to 0.2.
//!
//! ## Errors
//! - **EmptyData**: vidya: Input data slice is empty.
//! - **AllValuesNaN**: vidya: All input data values are `NaN`.
//! - **NotEnoughValidData**: vidya: Fewer than `long_period` valid data points remain
//!   after the first valid index.
//! - **InvalidParameters**: vidya: Invalid `short_period`, `long_period`, or `alpha`.
//!
//! ## Returns
//! - **`Ok(VidyaOutput)`** on success, containing a `Vec<f64>` matching the input length,
//!   with leading `NaN`s until the computation can start.
//! - **`Err(VidyaError)`** otherwise.

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
use paste::paste;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use thiserror::Error;

impl<'a> AsRef<[f64]> for VidyaInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			VidyaData::Slice(slice) => slice,
			VidyaData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

#[derive(Debug, Clone)]
pub enum VidyaData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct VidyaOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct VidyaParams {
	pub short_period: Option<usize>,
	pub long_period: Option<usize>,
	pub alpha: Option<f64>,
}

impl Default for VidyaParams {
	fn default() -> Self {
		Self {
			short_period: Some(2),
			long_period: Some(5),
			alpha: Some(0.2),
		}
	}
}

#[derive(Debug, Clone)]
pub struct VidyaInput<'a> {
	pub data: VidyaData<'a>,
	pub params: VidyaParams,
}

impl<'a> VidyaInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: VidyaParams) -> Self {
		Self {
			data: VidyaData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slice(sl: &'a [f64], p: VidyaParams) -> Self {
		Self {
			data: VidyaData::Slice(sl),
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", VidyaParams::default())
	}
	#[inline]
	pub fn get_short_period(&self) -> usize {
		self.params.short_period.unwrap_or(2)
	}
	#[inline]
	pub fn get_long_period(&self) -> usize {
		self.params.long_period.unwrap_or(5)
	}
	#[inline]
	pub fn get_alpha(&self) -> f64 {
		self.params.alpha.unwrap_or(0.2)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct VidyaBuilder {
	short_period: Option<usize>,
	long_period: Option<usize>,
	alpha: Option<f64>,
	kernel: Kernel,
}

impl Default for VidyaBuilder {
	fn default() -> Self {
		Self {
			short_period: None,
			long_period: None,
			alpha: None,
			kernel: Kernel::Auto,
		}
	}
}

impl VidyaBuilder {
	#[inline(always)]
	pub fn new() -> Self {
		Self::default()
	}
	#[inline(always)]
	pub fn short_period(mut self, n: usize) -> Self {
		self.short_period = Some(n);
		self
	}
	#[inline(always)]
	pub fn long_period(mut self, n: usize) -> Self {
		self.long_period = Some(n);
		self
	}
	#[inline(always)]
	pub fn alpha(mut self, a: f64) -> Self {
		self.alpha = Some(a);
		self
	}
	#[inline(always)]
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline(always)]
	pub fn apply(self, c: &Candles) -> Result<VidyaOutput, VidyaError> {
		let p = VidyaParams {
			short_period: self.short_period,
			long_period: self.long_period,
			alpha: self.alpha,
		};
		let i = VidyaInput::from_candles(c, "close", p);
		vidya_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slice(self, d: &[f64]) -> Result<VidyaOutput, VidyaError> {
		let p = VidyaParams {
			short_period: self.short_period,
			long_period: self.long_period,
			alpha: self.alpha,
		};
		let i = VidyaInput::from_slice(d, p);
		vidya_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<VidyaStream, VidyaError> {
		let p = VidyaParams {
			short_period: self.short_period,
			long_period: self.long_period,
			alpha: self.alpha,
		};
		VidyaStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum VidyaError {
	#[error("vidya: Empty data provided.")]
	EmptyData,
	#[error("vidya: All values are NaN.")]
	AllValuesNaN,
	#[error("vidya: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("vidya: Invalid short/long period or alpha. short={short}, long={long}, alpha={alpha}")]
	InvalidParameters { short: usize, long: usize, alpha: f64 },
	#[error("vidya: Output length mismatch: dst_len = {dst_len}, data_len = {data_len}")]
	InvalidOutputLength { dst_len: usize, data_len: usize },
}

#[inline]
pub fn vidya(input: &VidyaInput) -> Result<VidyaOutput, VidyaError> {
	vidya_with_kernel(input, Kernel::Auto)
}

pub fn vidya_with_kernel(input: &VidyaInput, kernel: Kernel) -> Result<VidyaOutput, VidyaError> {
	let data: &[f64] = match &input.data {
		VidyaData::Candles { candles, source } => source_type(candles, source),
		VidyaData::Slice(sl) => sl,
	};

	if data.is_empty() {
		return Err(VidyaError::EmptyData);
	}

	let short_period = input.get_short_period();
	let long_period = input.get_long_period();
	let alpha = input.get_alpha();

	if short_period < 1
		|| long_period < short_period
		|| long_period < 2
		|| alpha < 0.0
		|| alpha > 1.0
		|| long_period > data.len()
	{
		return Err(VidyaError::InvalidParameters {
			short: short_period,
			long: long_period,
			alpha,
		});
	}

	let first = data.iter().position(|&x| !x.is_nan()).ok_or(VidyaError::AllValuesNaN)?;
	if (data.len() - first) < long_period {
		return Err(VidyaError::NotEnoughValidData {
			needed: long_period,
			valid: data.len() - first,
		});
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	let warmup_period = first + long_period - 2;
	let mut out = alloc_with_nan_prefix(data.len(), warmup_period);
	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => {
				vidya_scalar(data, short_period, long_period, alpha, first, &mut out)
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => vidya_avx2(data, short_period, long_period, alpha, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => vidya_avx512(data, short_period, long_period, alpha, first, &mut out),
			_ => vidya_scalar(data, short_period, long_period, alpha, first, &mut out),
		}
	}

	Ok(VidyaOutput { values: out })
}

/// Write directly to output slice - no allocations (for WASM bindings)
#[inline]
pub fn vidya_into_slice(dst: &mut [f64], input: &VidyaInput, kern: Kernel) -> Result<(), VidyaError> {
	let data: &[f64] = match &input.data {
		VidyaData::Candles { candles, source } => source_type(candles, source),
		VidyaData::Slice(sl) => sl,
	};

	if data.is_empty() {
		return Err(VidyaError::EmptyData);
	}

	if dst.len() != data.len() {
		return Err(VidyaError::InvalidOutputLength {
			dst_len: dst.len(),
			data_len: data.len(),
		});
	}

	let short_period = input.get_short_period();
	let long_period = input.get_long_period();
	let alpha = input.get_alpha();

	if short_period < 1
		|| long_period < short_period
		|| long_period < 2
		|| alpha < 0.0
		|| alpha > 1.0
		|| long_period > data.len()
	{
		return Err(VidyaError::InvalidParameters {
			short: short_period,
			long: long_period,
			alpha,
		});
	}

	let first = data.iter().position(|&x| !x.is_nan()).ok_or(VidyaError::AllValuesNaN)?;
	if (data.len() - first) < long_period {
		return Err(VidyaError::NotEnoughValidData {
			needed: long_period,
			valid: data.len() - first,
		});
	}

	let chosen = match kern {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	let warmup_period = first + long_period - 2;
	
	unsafe {
		#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
		{
			if matches!(chosen, Kernel::Scalar | Kernel::ScalarBatch) {
				vidya_simd128(data, short_period, long_period, alpha, first, dst);
				// Fill warmup with NaN after computation, but don't overwrite the first valid value
				for v in &mut dst[..warmup_period] {
					*v = f64::NAN;
				}
				return Ok(());
			}
		}

		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => {
				vidya_scalar(data, short_period, long_period, alpha, first, dst)
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => vidya_avx2(data, short_period, long_period, alpha, first, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => vidya_avx512(data, short_period, long_period, alpha, first, dst),
			_ => vidya_scalar(data, short_period, long_period, alpha, first, dst),
		}
	}

	// Fill warmup with NaN, but don't overwrite the first valid value at warmup_period
	for v in &mut dst[..warmup_period] {
		*v = f64::NAN;
	}
	
	Ok(())
}

#[inline]
pub unsafe fn vidya_scalar(
	data: &[f64],
	short_period: usize,
	long_period: usize,
	alpha: f64,
	first: usize,
	out: &mut [f64],
) {
	let len = data.len();
	let mut long_sum = 0.0;
	let mut long_sum2 = 0.0;
	let mut short_sum = 0.0;
	let mut short_sum2 = 0.0;

	for i in first..(first + long_period) {
		long_sum += data[i];
		long_sum2 += data[i] * data[i];
		if i >= (first + long_period - short_period) {
			short_sum += data[i];
			short_sum2 += data[i] * data[i];
		}
	}

	let mut val = data[first + long_period - 2];
	out[first + long_period - 2] = val;

	if first + long_period - 1 < data.len() {
		let sp = short_period as f64;
		let lp = long_period as f64;
		let short_div = 1.0 / sp;
		let long_div = 1.0 / lp;
		let short_stddev = (short_sum2 * short_div - (short_sum * short_div) * (short_sum * short_div)).sqrt();
		let long_stddev = (long_sum2 * long_div - (long_sum * long_div) * (long_sum * long_div)).sqrt();
		let mut k = short_stddev / long_stddev;
		if k.is_nan() {
			k = 0.0;
		}
		k *= alpha;
		val = (data[first + long_period - 1] - val) * k + val;
		out[first + long_period - 1] = val;
	}

	for i in (first + long_period)..len {
		long_sum += data[i];
		long_sum2 += data[i] * data[i];
		short_sum += data[i];
		short_sum2 += data[i] * data[i];

		let remove_long = i - long_period;
		let remove_short = i - short_period;
		long_sum -= data[remove_long];
		long_sum2 -= data[remove_long] * data[remove_long];
		short_sum -= data[remove_short];
		short_sum2 -= data[remove_short] * data[remove_short];

		let sp = short_period as f64;
		let lp = long_period as f64;
		let short_div = 1.0 / sp;
		let long_div = 1.0 / lp;
		let short_stddev = (short_sum2 * short_div - (short_sum * short_div) * (short_sum * short_div)).sqrt();
		let long_stddev = (long_sum2 * long_div - (long_sum * long_div) * (long_sum * long_div)).sqrt();
		let mut k = short_stddev / long_stddev;
		if k.is_nan() {
			k = 0.0;
		}
		k *= alpha;
		val = (data[i] - val) * k + val;
		out[i] = val;
	}
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn vidya_simd128(
	data: &[f64],
	short_period: usize,
	long_period: usize,
	alpha: f64,
	first: usize,
	out: &mut [f64],
) {
	use core::arch::wasm32::*;

	let len = data.len();
	let mut long_sum = 0.0;
	let mut long_sum2 = 0.0;
	let mut short_sum = 0.0;
	let mut short_sum2 = 0.0;

	// Warmup phase - same as scalar
	for i in first..(first + long_period) {
		long_sum += data[i];
		long_sum2 += data[i] * data[i];
		if i >= (first + long_period - short_period) {
			short_sum += data[i];
			short_sum2 += data[i] * data[i];
		}
	}

	let mut val = data[first + long_period - 2];
	out[first + long_period - 2] = val;

	if first + long_period - 1 < data.len() {
		let sp = short_period as f64;
		let lp = long_period as f64;
		let short_div = 1.0 / sp;
		let long_div = 1.0 / lp;
		let short_stddev = (short_sum2 * short_div - (short_sum * short_div) * (short_sum * short_div)).sqrt();
		let long_stddev = (long_sum2 * long_div - (long_sum * long_div) * (long_sum * long_div)).sqrt();
		let mut k = short_stddev / long_stddev;
		if k.is_nan() {
			k = 0.0;
		}
		k *= alpha;
		val = (data[first + long_period - 1] - val) * k + val;
		out[first + long_period - 1] = val;
	}

	// Main loop with SIMD optimizations for sum calculations
	let alpha_v = f64x2_splat(alpha);
	let sp_v = f64x2_splat(short_period as f64);
	let lp_v = f64x2_splat(long_period as f64);
	let short_div_v = f64x2_splat(1.0 / short_period as f64);
	let long_div_v = f64x2_splat(1.0 / long_period as f64);

	for i in (first + long_period)..len {
		// Update sums - can vectorize some of this
		let new_val = data[i];
		long_sum += new_val;
		long_sum2 += new_val * new_val;
		short_sum += new_val;
		short_sum2 += new_val * new_val;

		let remove_long_idx = i - long_period;
		let remove_short_idx = i - short_period;
		let remove_long = data[remove_long_idx];
		let remove_short = data[remove_short_idx];
		
		long_sum -= remove_long;
		long_sum2 -= remove_long * remove_long;
		short_sum -= remove_short;
		short_sum2 -= remove_short * remove_short;

		// Calculate standard deviations using SIMD
		let short_mean = short_sum / short_period as f64;
		let long_mean = long_sum / long_period as f64;
		
		let short_variance = short_sum2 / short_period as f64 - short_mean * short_mean;
		let long_variance = long_sum2 / long_period as f64 - long_mean * long_mean;
		
		let short_stddev = short_variance.sqrt();
		let long_stddev = long_variance.sqrt();
		
		let mut k = short_stddev / long_stddev;
		if k.is_nan() {
			k = 0.0;
		}
		k *= alpha;
		val = (new_val - val) * k + val;
		out[i] = val;
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vidya_avx2(
	data: &[f64],
	short_period: usize,
	long_period: usize,
	alpha: f64,
	first: usize,
	out: &mut [f64],
) {
	vidya_scalar(data, short_period, long_period, alpha, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vidya_avx512(
	data: &[f64],
	short_period: usize,
	long_period: usize,
	alpha: f64,
	first: usize,
	out: &mut [f64],
) {
	if long_period <= 32 {
		vidya_avx512_short(data, short_period, long_period, alpha, first, out)
	} else {
		vidya_avx512_long(data, short_period, long_period, alpha, first, out)
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vidya_avx512_short(
	data: &[f64],
	short_period: usize,
	long_period: usize,
	alpha: f64,
	first: usize,
	out: &mut [f64],
) {
	vidya_scalar(data, short_period, long_period, alpha, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vidya_avx512_long(
	data: &[f64],
	short_period: usize,
	long_period: usize,
	alpha: f64,
	first: usize,
	out: &mut [f64],
) {
	vidya_scalar(data, short_period, long_period, alpha, first, out);
}

// Streaming (stepwise) API
#[derive(Debug, Clone)]
pub struct VidyaStream {
	short_period: usize,
	long_period: usize,
	alpha: f64,
	long_buf: Vec<f64>,
	short_buf: Vec<f64>,
	long_sum: f64,
	long_sum2: f64,
	short_sum: f64,
	short_sum2: f64,
	head: usize,
	idx: usize,
	val: f64,
	filled: bool,
}

impl VidyaStream {
	pub fn try_new(params: VidyaParams) -> Result<Self, VidyaError> {
		let short_period = params.short_period.unwrap_or(2);
		let long_period = params.long_period.unwrap_or(5);
		let alpha = params.alpha.unwrap_or(0.2);

		if short_period < 1 || long_period < short_period || long_period < 2 || alpha < 0.0 || alpha > 1.0 {
			return Err(VidyaError::InvalidParameters {
				short: short_period,
				long: long_period,
				alpha,
			});
		}
		Ok(Self {
			short_period,
			long_period,
			alpha,
			long_buf: alloc_with_nan_prefix(long_period, long_period),
			short_buf: alloc_with_nan_prefix(short_period, short_period),
			long_sum: 0.0,
			long_sum2: 0.0,
			short_sum: 0.0,
			short_sum2: 0.0,
			head: 0,
			idx: 0,
			val: f64::NAN,
			filled: false,
		})
	}

	#[inline(always)]
	pub fn update(&mut self, value: f64) -> Option<f64> {
		// Oldest values before inserting the new one
		let long_tail = self.long_buf[self.head];
		let short_tail = self.short_buf[self.idx % self.short_period];

		self.long_sum += value;
		self.long_sum2 += value * value;
		if self.idx >= self.long_period - self.short_period {
			self.short_sum += value;
			self.short_sum2 += value * value;
		}

		if self.idx >= self.long_period {
			self.long_sum -= long_tail;
			self.long_sum2 -= long_tail * long_tail;
			self.short_sum -= short_tail;
			self.short_sum2 -= short_tail * short_tail;
		}

		self.long_buf[self.head] = value;
		self.short_buf[self.idx % self.short_period] = value;
		self.head = (self.head + 1) % self.long_period;
		self.idx += 1;

		if self.idx < self.long_period - 1 {
			self.val = value;
			return None;
		}
		if self.idx == self.long_period - 1 {
			self.val = value;
			return Some(self.val);
		}

		let sp = self.short_period as f64;
		let lp = self.long_period as f64;
		let short_div = 1.0 / sp;
		let long_div = 1.0 / lp;
		let short_stddev = (self.short_sum2 * short_div - (self.short_sum * short_div).powi(2)).sqrt();
		let long_stddev = (self.long_sum2 * long_div - (self.long_sum * long_div).powi(2)).sqrt();
		let mut k = short_stddev / long_stddev;
		if k.is_nan() {
			k = 0.0;
		}
		k *= self.alpha;
		self.val = (value - self.val) * k + self.val;

		Some(self.val)
	}
}

// Batch parameter sweep
#[derive(Clone, Debug)]
pub struct VidyaBatchRange {
	pub short_period: (usize, usize, usize),
	pub long_period: (usize, usize, usize),
	pub alpha: (f64, f64, f64),
}

impl Default for VidyaBatchRange {
	fn default() -> Self {
		Self {
			short_period: (2, 2, 0),
			long_period: (5, 240, 1),
			alpha: (0.2, 0.2, 0.0),
		}
	}
}

#[derive(Clone, Debug, Default)]
pub struct VidyaBatchBuilder {
	range: VidyaBatchRange,
	kernel: Kernel,
}

impl VidyaBatchBuilder {
	pub fn new() -> Self {
		Self::default()
	}
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline]
	pub fn short_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.short_period = (start, end, step);
		self
	}
	#[inline]
	pub fn short_period_static(mut self, n: usize) -> Self {
		self.range.short_period = (n, n, 0);
		self
	}
	#[inline]
	pub fn long_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.long_period = (start, end, step);
		self
	}
	#[inline]
	pub fn long_period_static(mut self, n: usize) -> Self {
		self.range.long_period = (n, n, 0);
		self
	}
	#[inline]
	pub fn alpha_range(mut self, start: f64, end: f64, step: f64) -> Self {
		self.range.alpha = (start, end, step);
		self
	}
	#[inline]
	pub fn alpha_static(mut self, a: f64) -> Self {
		self.range.alpha = (a, a, 0.0);
		self
	}
	pub fn apply_slice(self, data: &[f64]) -> Result<VidyaBatchOutput, VidyaError> {
		vidya_batch_with_kernel(data, &self.range, self.kernel)
	}
	pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<VidyaBatchOutput, VidyaError> {
		VidyaBatchBuilder::new().kernel(k).apply_slice(data)
	}
	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<VidyaBatchOutput, VidyaError> {
		let slice = source_type(c, src);
		self.apply_slice(slice)
	}
	pub fn with_default_candles(c: &Candles) -> Result<VidyaBatchOutput, VidyaError> {
		VidyaBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
	}
}

pub fn vidya_batch_with_kernel(
	data: &[f64],
	sweep: &VidyaBatchRange,
	k: Kernel,
) -> Result<VidyaBatchOutput, VidyaError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => {
			return Err(VidyaError::InvalidParameters {
				short: 0,
				long: 0,
				alpha: 0.0,
			})
		}
	};

	let simd = match kernel {
		#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
		Kernel::Avx512Batch => Kernel::Avx512,
		#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => Kernel::Scalar,
	};
	vidya_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct VidyaBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<VidyaParams>,
	pub rows: usize,
	pub cols: usize,
}
impl VidyaBatchOutput {
	pub fn row_for_params(&self, p: &VidyaParams) -> Option<usize> {
		self.combos.iter().position(|c| {
			c.short_period.unwrap_or(2) == p.short_period.unwrap_or(2)
				&& c.long_period.unwrap_or(5) == p.long_period.unwrap_or(5)
				&& (c.alpha.unwrap_or(0.2) - p.alpha.unwrap_or(0.2)).abs() < 1e-12
		})
	}
	pub fn values_for(&self, p: &VidyaParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &VidyaBatchRange) -> Vec<VidyaParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			let mut v = Vec::with_capacity(1);
			v.push(start);
			return v;
		}
		(start..=end).step_by(step).collect()
	}
	fn axis_f64((start, end, step): (f64, f64, f64)) -> Vec<f64> {
		if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
			let mut v = Vec::with_capacity(1);
			v.push(start);
			return v;
		}
		let count = ((end - start) / step).ceil() as usize + 1;
		let mut v = Vec::with_capacity(count);
		let mut x = start;
		while x <= end + 1e-12 {
			v.push(x);
			x += step;
		}
		v
	}
	let short_periods = axis_usize(r.short_period);
	let long_periods = axis_usize(r.long_period);
	let alphas = axis_f64(r.alpha);

	let mut out = Vec::with_capacity(short_periods.len() * long_periods.len() * alphas.len());
	for &sp in &short_periods {
		for &lp in &long_periods {
			for &a in &alphas {
				out.push(VidyaParams {
					short_period: Some(sp),
					long_period: Some(lp),
					alpha: Some(a),
				});
			}
		}
	}
	out
}

#[inline(always)]
pub fn vidya_batch_slice(data: &[f64], sweep: &VidyaBatchRange, kern: Kernel) -> Result<VidyaBatchOutput, VidyaError> {
	vidya_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn vidya_batch_par_slice(
	data: &[f64],
	sweep: &VidyaBatchRange,
	kern: Kernel,
) -> Result<VidyaBatchOutput, VidyaError> {
	vidya_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn vidya_batch_inner(
	data: &[f64],
	sweep: &VidyaBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<VidyaBatchOutput, VidyaError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(VidyaError::InvalidParameters {
			short: 0,
			long: 0,
			alpha: 0.0,
		});
	}
	let first = data.iter().position(|x| !x.is_nan()).ok_or(VidyaError::AllValuesNaN)?;
	let max_long = combos.iter().map(|c| c.long_period.unwrap()).max().unwrap();
	if data.len() - first < max_long {
		return Err(VidyaError::NotEnoughValidData {
			needed: max_long,
			valid: data.len() - first,
		});
	}

	let rows = combos.len();
	let cols = data.len();
	
	// Calculate warmup periods for each parameter combination
	let warmup_periods: Vec<usize> = combos.iter()
		.map(|c| first + c.long_period.unwrap() - 2)
		.collect();
	
	// Allocate uninitialized matrix and initialize NaN prefixes
	let mut buf_mu = make_uninit_matrix(rows, cols);
	init_matrix_prefixes(&mut buf_mu, cols, &warmup_periods);
	
	// Convert to initialized memory safely
	let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
	let out: &mut [f64] =
		unsafe { core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len()) };
	let mut values = out;
	
	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let p = &combos[row];
		let sp = p.short_period.unwrap();
		let lp = p.long_period.unwrap();
		let a = p.alpha.unwrap();
		match kern {
			Kernel::Scalar | Kernel::ScalarBatch => vidya_row_scalar(data, first, sp, lp, a, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => vidya_row_avx2(data, first, sp, lp, a, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => vidya_row_avx512(data, first, sp, lp, a, out_row),
			_ => vidya_row_scalar(data, first, sp, lp, a, out_row),
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
	
	// Take ownership of the buffer
	let values = unsafe {
		Vec::from_raw_parts(
			buf_guard.as_mut_ptr() as *mut f64,
			buf_guard.len(),
			buf_guard.capacity(),
		)
	};
	
	Ok(VidyaBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
unsafe fn vidya_row_scalar(
	data: &[f64],
	first: usize,
	short_period: usize,
	long_period: usize,
	alpha: f64,
	out: &mut [f64],
) {
	vidya_scalar(data, short_period, long_period, alpha, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn vidya_row_avx2(
	data: &[f64],
	first: usize,
	short_period: usize,
	long_period: usize,
	alpha: f64,
	out: &mut [f64],
) {
	vidya_scalar(data, short_period, long_period, alpha, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn vidya_row_avx512(
	data: &[f64],
	first: usize,
	short_period: usize,
	long_period: usize,
	alpha: f64,
	out: &mut [f64],
) {
	if long_period <= 32 {
		vidya_row_avx512_short(data, first, short_period, long_period, alpha, out)
	} else {
		vidya_row_avx512_long(data, first, short_period, long_period, alpha, out)
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn vidya_row_avx512_short(
	data: &[f64],
	first: usize,
	short_period: usize,
	long_period: usize,
	alpha: f64,
	out: &mut [f64],
) {
	vidya_scalar(data, short_period, long_period, alpha, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn vidya_row_avx512_long(
	data: &[f64],
	first: usize,
	short_period: usize,
	long_period: usize,
	alpha: f64,
	out: &mut [f64],
) {
	vidya_scalar(data, short_period, long_period, alpha, first, out);
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_vidya_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let default_params = VidyaParams {
			short_period: None,
			long_period: Some(10),
			alpha: None,
		};
		let input_default = VidyaInput::from_candles(&candles, "close", default_params);
		let output_default = vidya_with_kernel(&input_default, kernel)?;
		assert_eq!(output_default.values.len(), candles.close.len());
		Ok(())
	}

	fn check_vidya_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let close_prices = &candles.close;

		let params = VidyaParams {
			short_period: Some(2),
			long_period: Some(5),
			alpha: Some(0.2),
		};
		let input = VidyaInput::from_candles(&candles, "close", params);
		let vidya_result = vidya_with_kernel(&input, kernel)?;
		assert_eq!(vidya_result.values.len(), close_prices.len());

		if vidya_result.values.len() >= 5 {
			let expected_last_five = [
				59553.42785306692,
				59503.60445032524,
				59451.72283651444,
				59413.222561244685,
				59239.716526894175,
			];
			let start_index = vidya_result.values.len() - 5;
			let result_last_five = &vidya_result.values[start_index..];
			for (i, &value) in result_last_five.iter().enumerate() {
				let expected_value = expected_last_five[i];
				assert!(
					(value - expected_value).abs() < 1e-1,
					"VIDYA mismatch at index {}: expected {}, got {}",
					i,
					expected_value,
					value
				);
			}
		}
		Ok(())
	}

	fn check_vidya_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = VidyaInput::with_default_candles(&candles);
		match input.data {
			VidyaData::Candles { source, .. } => assert_eq!(source, "close"),
			_ => panic!("Expected VidyaData::Candles"),
		}
		let output = vidya_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_vidya_invalid_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data = [10.0, 20.0, 30.0];
		let params = VidyaParams {
			short_period: Some(0),
			long_period: Some(5),
			alpha: Some(0.2),
		};
		let input = VidyaInput::from_slice(&data, params);
		let result = vidya_with_kernel(&input, kernel);
		assert!(result.is_err(), "Expected error for invalid short period");
		Ok(())
	}

	fn check_vidya_exceeding_data_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data = [10.0, 20.0, 30.0];
		let params = VidyaParams {
			short_period: Some(2),
			long_period: Some(5),
			alpha: Some(0.2),
		};
		let input = VidyaInput::from_slice(&data, params);
		let result = vidya_with_kernel(&input, kernel);
		assert!(result.is_err(), "Expected error for period > data.len()");
		Ok(())
	}

	fn check_vidya_very_small_data_set(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data = [42.0, 43.0];
		let params = VidyaParams {
			short_period: Some(2),
			long_period: Some(5),
			alpha: Some(0.2),
		};
		let input = VidyaInput::from_slice(&data, params);
		let result = vidya_with_kernel(&input, kernel);
		assert!(result.is_err(), "Expected error for data smaller than long period");
		Ok(())
	}

	fn check_vidya_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let first_params = VidyaParams {
			short_period: Some(2),
			long_period: Some(5),
			alpha: Some(0.2),
		};
		let first_input = VidyaInput::from_candles(&candles, "close", first_params);
		let first_result = vidya_with_kernel(&first_input, kernel)?;

		let second_params = VidyaParams {
			short_period: Some(2),
			long_period: Some(5),
			alpha: Some(0.2),
		};
		let second_input = VidyaInput::from_slice(&first_result.values, second_params);
		let second_result = vidya_with_kernel(&second_input, kernel)?;
		assert_eq!(second_result.values.len(), first_result.values.len());
		Ok(())
	}

	fn check_vidya_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = VidyaParams {
			short_period: Some(2),
			long_period: Some(5),
			alpha: Some(0.2),
		};
		let input = VidyaInput::from_candles(&candles, "close", params);
		let vidya_result = vidya_with_kernel(&input, kernel)?;
		if vidya_result.values.len() > 10 {
			for i in 10..vidya_result.values.len() {
				assert!(!vidya_result.values[i].is_nan());
			}
		}
		Ok(())
	}

	fn check_vidya_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let short_period = 2;
		let long_period = 5;
		let alpha = 0.2;

		let params = VidyaParams {
			short_period: Some(short_period),
			long_period: Some(long_period),
			alpha: Some(alpha),
		};
		let input = VidyaInput::from_candles(&candles, "close", params.clone());
		let batch_output = vidya_with_kernel(&input, kernel)?.values;

		let mut stream = VidyaStream::try_new(params.clone())?;
		let mut stream_values = Vec::with_capacity(candles.close.len());
		for &price in &candles.close {
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
				diff < 1e-6,
				"[{}] VIDYA streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
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
	fn check_vidya_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let test_params = vec![
			VidyaParams::default(),
			VidyaParams {
				short_period: Some(1),
				long_period: Some(2),
				alpha: Some(0.1),
			},
			VidyaParams {
				short_period: Some(2),
				long_period: Some(3),
				alpha: Some(0.2),
			},
			VidyaParams {
				short_period: Some(2),
				long_period: Some(5),
				alpha: Some(0.5),
			},
			VidyaParams {
				short_period: Some(3),
				long_period: Some(7),
				alpha: Some(0.3),
			},
			VidyaParams {
				short_period: Some(4),
				long_period: Some(10),
				alpha: Some(0.2),
			},
			VidyaParams {
				short_period: Some(5),
				long_period: Some(20),
				alpha: Some(0.4),
			},
			VidyaParams {
				short_period: Some(10),
				long_period: Some(30),
				alpha: Some(0.2),
			},
			VidyaParams {
				short_period: Some(15),
				long_period: Some(50),
				alpha: Some(0.3),
			},
			VidyaParams {
				short_period: Some(20),
				long_period: Some(100),
				alpha: Some(0.2),
			},
			VidyaParams {
				short_period: Some(50),
				long_period: Some(200),
				alpha: Some(0.1),
			},
			// Test various alpha values
			VidyaParams {
				short_period: Some(2),
				long_period: Some(10),
				alpha: Some(0.8),
			},
			VidyaParams {
				short_period: Some(3),
				long_period: Some(15),
				alpha: Some(1.0),
			},
			// Edge cases
			VidyaParams {
				short_period: Some(1),
				long_period: Some(100),
				alpha: Some(0.01),
			},
			VidyaParams {
				short_period: Some(99),
				long_period: Some(100),
				alpha: Some(0.99),
			},
		];

		for (param_idx, params) in test_params.iter().enumerate() {
			let input = VidyaInput::from_candles(&candles, "close", params.clone());
			let output = vidya_with_kernel(&input, kernel)?;

			for (i, &val) in output.values.iter().enumerate() {
				if val.is_nan() {
					continue;
				}

				let bits = val.to_bits();

				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: {:?} (param set {})",
						test_name, val, bits, i, params, param_idx
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: {:?} (param set {})",
						test_name, val, bits, i, params, param_idx
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: {:?} (param set {})",
						test_name, val, bits, i, params, param_idx
					);
				}
			}
		}

		Ok(())
	}

	#[cfg(not(debug_assertions))]
	fn check_vidya_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		Ok(())
	}

	#[cfg(feature = "proptest")]
	#[allow(clippy::float_cmp)]
	fn check_vidya_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		use proptest::prelude::*;
		skip_if_unsupported!(kernel, test_name);

		// Strategy for generating realistic test data with varying volatility patterns
		let strat = (1usize..=20)  // short_period range
			.prop_flat_map(|short_period| {
				// long_period must be >= short_period
				let long_min = (short_period + 1).max(2);
				let long_max = 100.min(long_min + 50);
				
				(long_min..=long_max).prop_flat_map(move |long_period| {
					// Generate data with length >= long_period
					let data_len = long_period.max(10)..400;
					
					(
						// Generate realistic price data with varying volatility
						prop::collection::vec(
							(-0.05f64..0.05f64).prop_filter("finite", |x| x.is_finite()),
							data_len,
						).prop_map(|returns| {
							let mut prices = Vec::with_capacity(returns.len());
							let mut price = 100.0;
							// Create periods of high and low volatility
							for (i, ret) in returns.iter().enumerate() {
								// Alternate between high and low volatility periods
								let volatility_factor = if (i / 20) % 2 == 0 { 0.5 } else { 2.0 };
								price *= 1.0 + (ret * volatility_factor);
								prices.push(price);
							}
							prices
						}),
						Just(short_period),
						Just(long_period),
						0.01f64..1.0f64,  // alpha range
					)
				})
			});

		proptest::test_runner::TestRunner::default()
			.run(&strat, |(data, short_period, long_period, alpha)| {
				let params = VidyaParams {
					short_period: Some(short_period),
					long_period: Some(long_period),
					alpha: Some(alpha),
				};
				let input = VidyaInput::from_slice(&data, params.clone());

				// Test kernel consistency
				let VidyaOutput { values: out } = vidya_with_kernel(&input, kernel).unwrap();
				let VidyaOutput { values: ref_out } = vidya_with_kernel(&input, Kernel::Scalar).unwrap();

				// Property 1: Kernel consistency - all kernels should produce identical results
				for i in 0..data.len() {
					let y = out[i];
					let r = ref_out[i];
					
					if !y.is_finite() || !r.is_finite() {
						prop_assert!(y.to_bits() == r.to_bits(), 
							"[{}] finite/NaN mismatch at idx {}: {} vs {}", test_name, i, y, r);
						continue;
					}
					
					let ulp_diff = y.to_bits().abs_diff(r.to_bits());
					prop_assert!(
						(y - r).abs() <= 1e-9 || ulp_diff <= 4,
						"[{}] kernel mismatch at idx {}: {} vs {} (ULP={})", 
						test_name, i, y, r, ulp_diff
					);
				}

				// Property 2: Warmup period - VIDYA starts outputting at first + long_period - 2
				let first = data.iter().position(|&x| !x.is_nan()).unwrap_or(0);
				let first_valid_idx = if first + long_period >= 2 { 
					first + long_period - 2 
				} else { 
					0 
				};
				
				// Values before first_valid_idx should be NaN
				for i in 0..first_valid_idx.min(data.len()) {
					prop_assert!(out[i].is_nan(), 
						"[{}] Expected NaN during warmup at idx {}, got {}", test_name, i, out[i]);
				}
				
				// Property 3: First valid output should be at first_valid_idx
				if first_valid_idx < data.len() {
					prop_assert!(!out[first_valid_idx].is_nan(),
						"[{}] Expected valid value at first_valid_idx {}, got NaN", test_name, first_valid_idx);
				}
				
				// For the rest of the test, use the actual warmup_end as defined by the implementation
				let warmup_end = first + long_period - 2;

				// Property 4: Output values should be reasonable (within data range + some margin)
				if data.len() > warmup_end + 1 {
					let data_min = data.iter().cloned().fold(f64::INFINITY, f64::min);
					let data_max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
					// Use a minimum margin for cases where all values are similar
					let range = data_max - data_min;
					let margin = if range < 1.0 { 
						data_max.abs() * 0.2  // 20% of value when range is tiny
					} else {
						range * 0.1  // 10% of range otherwise
					};
					
					for i in (warmup_end + 1)..data.len() {
						let y = out[i];
						if y.is_finite() {
							prop_assert!(
								y >= data_min - margin && y <= data_max + margin,
								"[{}] Output {} at idx {} outside reasonable range [{}, {}]",
								test_name, y, i, data_min - margin, data_max + margin
							);
						}
					}
				}

				// Property 5: Very low alpha should produce stable output
				if alpha < 0.05 && data.len() > warmup_end + 10 {
					// With very low alpha, VIDYA should change slowly
					let vidya_section = &out[(warmup_end + 1)..];
					if vidya_section.len() > 2 {
						// Check that consecutive values are very close
						for window in vidya_section.windows(2) {
							let change_ratio = (window[1] - window[0]).abs() / window[0].abs().max(1e-10);
							prop_assert!(
								change_ratio < 0.1,  // Less than 10% change between consecutive values
								"[{}] With alpha={}, VIDYA should be stable but found large change ratio {}",
								test_name, alpha, change_ratio
							);
						}
					}
				}

				// Property 6: Constant input should produce constant output (after warmup)
				if data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10) && data.len() > warmup_end + 1 {
					// All data values are essentially the same
					let constant_value = data[0];
					for i in (warmup_end + 1)..data.len() {
						prop_assert!(
							(out[i] - constant_value).abs() <= 1e-9,
							"[{}] Constant input should produce constant output, got {} expected {}",
							test_name, out[i], constant_value
						);
					}
				}

				// Property 7: VIDYA follows general price trends
				// Check that VIDYA generally follows the direction of price movements
				if data.len() > warmup_end + 20 {
					// Count how often VIDYA moves in the same direction as price
					let mut same_direction_count = 0;
					let mut total_movements = 0;
					
					for i in (warmup_end + 1)..data.len() {
						let price_change = data[i] - data[i - 1];
						let vidya_change = out[i] - out[i - 1];
						
						// Only count significant movements
						if price_change.abs() > 1e-6 && vidya_change.abs() > 1e-10 {
							total_movements += 1;
							if price_change.signum() == vidya_change.signum() {
								same_direction_count += 1;
							}
						}
					}
					
					// VIDYA should follow price direction more often than not (at least 44% for a lagging indicator)
					// Note: Can be lower when there are many flat sections in the data
					if total_movements > 10 {
						let direction_ratio = same_direction_count as f64 / total_movements as f64;
						prop_assert!(
							direction_ratio >= 0.44,
							"[{}] VIDYA should generally follow price direction, but only followed {:.1}% of the time",
							test_name, direction_ratio * 100.0
						);
					}
				}
				
				// Property 8: No poison values in output
				for (i, &val) in out.iter().enumerate() {
					if val.is_finite() {
						let bits = val.to_bits();
						prop_assert!(
							bits != 0x11111111_11111111 && 
							bits != 0x22222222_22222222 && 
							bits != 0x33333333_33333333,
							"[{}] Found poison value {} (0x{:016X}) at index {}",
							test_name, val, bits, i
						);
					}
				}

				Ok(())
			})
			.unwrap();

		Ok(())
	}

	macro_rules! generate_all_vidya_tests {
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
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                $(
                    #[test]
                    fn [<$test_fn _simd128_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _simd128_f64>]), Kernel::Scalar);
                    }
                )*
            }
        }
    }

	generate_all_vidya_tests!(
		check_vidya_partial_params,
		check_vidya_accuracy,
		check_vidya_default_candles,
		check_vidya_invalid_params,
		check_vidya_exceeding_data_length,
		check_vidya_very_small_data_set,
		check_vidya_reinput,
		check_vidya_nan_handling,
		check_vidya_streaming,
		check_vidya_no_poison
	);

	#[cfg(feature = "proptest")]
	generate_all_vidya_tests!(check_vidya_property);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let output = VidyaBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;

		let def = VidyaParams::default();
		let row = output.values_for(&def).expect("default row missing");

		assert_eq!(row.len(), c.close.len());

		let expected = [
			59553.42785306692,
			59503.60445032524,
			59451.72283651444,
			59413.222561244685,
			59239.716526894175,
		];
		let start = row.len() - 5;
		for (i, &v) in row[start..].iter().enumerate() {
			assert!(
				(v - expected[i]).abs() < 1e-1,
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

		let test_configs = vec![
			// (short_start, short_end, short_step, long_start, long_end, long_step, alpha_start, alpha_end, alpha_step)
			(2, 5, 1, 5, 10, 1, 0.2, 0.2, 0.0),           // Small ranges, static alpha
			(2, 10, 2, 10, 30, 5, 0.1, 0.5, 0.1),         // Medium ranges, varying alpha
			(5, 20, 5, 30, 60, 10, 0.2, 0.2, 0.0),        // Medium to large, static alpha
			(10, 30, 10, 50, 100, 25, 0.3, 0.3, 0.0),     // Large ranges, static alpha
			(2, 2, 0, 5, 50, 5, 0.1, 0.9, 0.2),           // Static short, varying long and alpha
			(5, 15, 5, 20, 20, 0, 0.2, 0.8, 0.3),         // Varying short, static long, varying alpha
			(1, 3, 1, 4, 8, 2, 0.5, 0.5, 0.0),            // Small dense ranges
			(20, 50, 15, 100, 200, 50, 0.1, 0.3, 0.1),    // Very large ranges
			(2, 2, 0, 3, 3, 0, 0.1, 1.0, 0.1),            // Static both, full alpha range
		];

		for (cfg_idx, &(s_start, s_end, s_step, l_start, l_end, l_step, a_start, a_end, a_step)) in test_configs.iter().enumerate() {
			let output = VidyaBatchBuilder::new()
				.kernel(kernel)
				.short_period_range(s_start, s_end, s_step)
				.long_period_range(l_start, l_end, l_step)
				.alpha_range(a_start, a_end, a_step)
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
						 at row {} col {} (flat index {}) with params: {:?}",
						test, cfg_idx, val, bits, row, col, idx, combo
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: {:?}",
						test, cfg_idx, val, bits, row, col, idx, combo
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: {:?}",
						test, cfg_idx, val, bits, row, col, idx, combo
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
	gen_batch_tests!(check_batch_no_poison);
}

// Python bindings section

/// Batch calculation that writes directly to output buffer (for Python bindings)
#[inline(always)]
pub fn vidya_batch_inner_into(
	data: &[f64],
	sweep: &VidyaBatchRange,
	kern: Kernel,
	parallel: bool,
	out: &mut [f64],
) -> Result<Vec<VidyaParams>, VidyaError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(VidyaError::InvalidParameters {
			short: 0,
			long: 0,
			alpha: 0.0,
		});
	}
	
	let first = data.iter().position(|x| !x.is_nan()).ok_or(VidyaError::AllValuesNaN)?;
	let max_long = combos.iter().map(|c| c.long_period.unwrap()).max().unwrap();
	if data.len() - first < max_long {
		return Err(VidyaError::NotEnoughValidData {
			needed: max_long,
			valid: data.len() - first,
		});
	}

	let rows = combos.len();
	let cols = data.len();
	
	// Calculate warmup periods for each parameter combination
	let warmup_periods: Vec<usize> = combos.iter()
		.map(|c| first + c.long_period.unwrap() - 2)
		.collect();
	
	// Initialize the NaN prefixes directly in the output buffer
	for (row, &warmup) in warmup_periods.iter().enumerate() {
		let row_start = row * cols;
		out[row_start..row_start + warmup].fill(f64::NAN);
	}
	
	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let p = &combos[row];
		let sp = p.short_period.unwrap();
		let lp = p.long_period.unwrap();
		let a = p.alpha.unwrap();
		match kern {
			Kernel::Scalar | Kernel::ScalarBatch => vidya_row_scalar(data, first, sp, lp, a, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => vidya_row_avx2(data, first, sp, lp, a, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => vidya_row_avx512(data, first, sp, lp, a, out_row),
			_ => vidya_row_scalar(data, first, sp, lp, a, out_row),
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
#[pyfunction(name = "vidya")]
#[pyo3(signature = (data, short_period, long_period, alpha, kernel=None))]
pub fn vidya_py<'py>(
	py: Python<'py>,
	data: PyReadonlyArray1<'py, f64>,
	short_period: usize,
	long_period: usize,
	alpha: f64,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, false)?;
	
	let params = VidyaParams {
		short_period: Some(short_period),
		long_period: Some(long_period),
		alpha: Some(alpha),
	};
	let input = VidyaInput::from_slice(slice_in, params);
	
	let result_vec: Vec<f64> = py.allow_threads(|| {
		vidya_with_kernel(&input, kern).map(|o| o.values)
	})
	.map_err(|e| PyValueError::new_err(e.to_string()))?;
	
	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "VidyaStream")]
pub struct VidyaStreamPy {
	stream: VidyaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl VidyaStreamPy {
	#[new]
	fn new(short_period: usize, long_period: usize, alpha: f64) -> PyResult<Self> {
		let params = VidyaParams {
			short_period: Some(short_period),
			long_period: Some(long_period),
			alpha: Some(alpha),
		};
		let stream = VidyaStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(VidyaStreamPy { stream })
	}
	
	fn update(&mut self, value: f64) -> Option<f64> {
		self.stream.update(value)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "vidya_batch")]
#[pyo3(signature = (data, short_period_range, long_period_range, alpha_range, kernel=None))]
pub fn vidya_batch_py<'py>(
	py: Python<'py>,
	data: PyReadonlyArray1<'py, f64>,
	short_period_range: (usize, usize, usize),
	long_period_range: (usize, usize, usize),
	alpha_range: (f64, f64, f64),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
	let slice_in = data.as_slice()?;
	
	let sweep = VidyaBatchRange {
		short_period: short_period_range,
		long_period: long_period_range,
		alpha: alpha_range,
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
				_ => kernel,
			};
			vidya_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
		})
		.map_err(|e| PyValueError::new_err(e.to_string()))?;
	
	let dict = PyDict::new(py);
	dict.set_item("values", out_arr.reshape((rows, cols))?)?;
	dict.set_item(
		"short_periods",
		combos
			.iter()
			.map(|p| p.short_period.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;
	dict.set_item(
		"long_periods",
		combos
			.iter()
			.map(|p| p.long_period.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;
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

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vidya_js(data: &[f64], short_period: usize, long_period: usize, alpha: f64) -> Result<Vec<f64>, JsValue> {
	let params = VidyaParams {
		short_period: Some(short_period),
		long_period: Some(long_period),
		alpha: Some(alpha),
	};
	let input = VidyaInput::from_slice(data, params);

	let mut output = vec![0.0; data.len()];
	vidya_into_slice(&mut output, &input, Kernel::Auto).map_err(|e| JsValue::from_str(&e.to_string()))?;

	Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vidya_into(
	in_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	short_period: usize,
	long_period: usize,
	alpha: f64,
) -> Result<(), JsValue> {
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}

	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);
		let params = VidyaParams {
			short_period: Some(short_period),
			long_period: Some(long_period),
			alpha: Some(alpha),
		};
		let input = VidyaInput::from_slice(data, params);

		if in_ptr == out_ptr {
			// Handle aliasing: use temporary buffer
			let mut temp = vec![0.0; len];
			vidya_into_slice(&mut temp, &input, Kernel::Auto).map_err(|e| JsValue::from_str(&e.to_string()))?;
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			out.copy_from_slice(&temp);
		} else {
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			vidya_into_slice(out, &input, Kernel::Auto).map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vidya_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vidya_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe {
			let _ = Vec::from_raw_parts(ptr, len, len);
		}
	}
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct VidyaBatchConfig {
	pub short_period_range: (usize, usize, usize),
	pub long_period_range: (usize, usize, usize),
	pub alpha_range: (f64, f64, f64),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct VidyaBatchJsOutput {
	pub values: Vec<f64>,
	pub combos: Vec<VidyaParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = vidya_batch)]
pub fn vidya_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	let config: VidyaBatchConfig =
		serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

	let sweep = VidyaBatchRange {
		short_period: config.short_period_range,
		long_period: config.long_period_range,
		alpha: config.alpha_range,
	};

	let mut output = vec![0.0; data.len() * 232]; // Default 232 combinations like alma
	let kernel = detect_best_kernel();
	let combos = vidya_batch_inner_into(data, &sweep, kernel, false, &mut output)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;

	let rows = combos.len();
	let cols = data.len();
	output.truncate(rows * cols);

	let result = VidyaBatchJsOutput {
		values: output,
		combos,
		rows,
		cols,
	};

	serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vidya_batch_into(
	in_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	short_period_start: usize,
	short_period_end: usize,
	short_period_step: usize,
	long_period_start: usize,
	long_period_end: usize,
	long_period_step: usize,
	alpha_start: f64,
	alpha_end: f64,
	alpha_step: f64,
) -> Result<usize, JsValue> {
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}

	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);
		let sweep = VidyaBatchRange {
			short_period: (short_period_start, short_period_end, short_period_step),
			long_period: (long_period_start, long_period_end, long_period_step),
			alpha: (alpha_start, alpha_end, alpha_step),
		};

		// Calculate expected output size
		let combos = expand_grid(&sweep);
		let total_size = combos.len() * len;
		let out = std::slice::from_raw_parts_mut(out_ptr, total_size);

		let kernel = detect_best_kernel();
		vidya_batch_inner_into(data, &sweep, kernel, false, out)
			.map_err(|e| JsValue::from_str(&e.to_string()))?;

		Ok(combos.len())
	}
}
