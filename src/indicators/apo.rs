//! # Absolute Price Oscillator (APO)
//!
//! Calculates the difference between two exponential moving averages (EMAs) of
//! different lengths (`short_period` and `long_period`), measuring momentum and
//! trend shifts. The interface and performance are structured similar to ALMA.
//!
//! ## Parameters
//! - **short_period**: EMA window size for the short period (defaults to 10).
//! - **long_period**: EMA window size for the long period (defaults to 20).
//!
//! ## Errors
//! - **AllValuesNaN**: apo: All input data values are `NaN`.
//! - **InvalidPeriod**: apo: Periods are zero, or invalid.
//! - **ShortPeriodNotLessThanLong**: apo: `short_period` is not less than `long_period`.
//! - **NotEnoughValidData**: apo: Not enough valid data for the requested `long_period`.
//!
//! ## Returns
//! - **`Ok(ApoOutput)`** on success, containing a `Vec<f64>` matching input length.
//! - **`Err(ApoError)`** otherwise.

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
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
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

// --- Data Representation

#[derive(Debug, Clone)]
pub enum ApoData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

impl<'a> AsRef<[f64]> for ApoInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			ApoData::Slice(slice) => slice,
			ApoData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

// --- Structs and Params

#[derive(Debug, Clone)]
pub struct ApoOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct ApoParams {
	pub short_period: Option<usize>,
	pub long_period: Option<usize>,
}
impl Default for ApoParams {
	fn default() -> Self {
		Self {
			short_period: Some(10),
			long_period: Some(20),
		}
	}
}

#[derive(Debug, Clone)]
pub struct ApoInput<'a> {
	pub data: ApoData<'a>,
	pub params: ApoParams,
}
impl<'a> ApoInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: ApoParams) -> Self {
		Self {
			data: ApoData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slice(sl: &'a [f64], p: ApoParams) -> Self {
		Self {
			data: ApoData::Slice(sl),
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", ApoParams::default())
	}
	#[inline]
	pub fn get_short_period(&self) -> usize {
		self.params.short_period.unwrap_or(10)
	}
	#[inline]
	pub fn get_long_period(&self) -> usize {
		self.params.long_period.unwrap_or(20)
	}
}

// --- Error Types

#[derive(Debug, Error)]
pub enum ApoError {
	#[error("apo: All values are NaN.")]
	AllValuesNaN,
	#[error("apo: Invalid period: short={short}, long={long}")]
	InvalidPeriod { short: usize, long: usize },
	#[error("apo: short_period not less than long_period: short={short}, long={long}")]
	ShortPeriodNotLessThanLong { short: usize, long: usize },
	#[error("apo: Not enough valid data: needed={needed}, valid={valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
}

// --- Builder API

#[derive(Copy, Clone, Debug)]
pub struct ApoBuilder {
	short_period: Option<usize>,
	long_period: Option<usize>,
	kernel: Kernel,
}
impl Default for ApoBuilder {
	fn default() -> Self {
		Self {
			short_period: None,
			long_period: None,
			kernel: Kernel::Auto,
		}
	}
}
impl ApoBuilder {
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
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline(always)]
	pub fn apply(self, c: &Candles) -> Result<ApoOutput, ApoError> {
		let p = ApoParams {
			short_period: self.short_period,
			long_period: self.long_period,
		};
		let i = ApoInput::from_candles(c, "close", p);
		apo_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slice(self, d: &[f64]) -> Result<ApoOutput, ApoError> {
		let p = ApoParams {
			short_period: self.short_period,
			long_period: self.long_period,
		};
		let i = ApoInput::from_slice(d, p);
		apo_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<ApoStream, ApoError> {
		let p = ApoParams {
			short_period: self.short_period,
			long_period: self.long_period,
		};
		ApoStream::try_new(p)
	}
}

// --- Main Indicator Function

#[inline]
pub fn apo(input: &ApoInput) -> Result<ApoOutput, ApoError> {
	apo_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn apo_prepare<'a>(
	input: &'a ApoInput,
	kernel: Kernel,
) -> Result<(&'a [f64], usize, usize, usize, usize, Kernel), ApoError> {
	let data: &[f64] = input.as_ref();
	let len = data.len();
	if len == 0 {
		return Err(ApoError::AllValuesNaN);
	}

	let first = data.iter().position(|x| !x.is_nan()).ok_or(ApoError::AllValuesNaN)?;
	let short = input.get_short_period();
	let long = input.get_long_period();

	if short == 0 || long == 0 {
		return Err(ApoError::InvalidPeriod { short, long });
	}
	if short >= long {
		return Err(ApoError::ShortPeriodNotLessThanLong { short, long });
	}
	if (len - first) < long {
		return Err(ApoError::NotEnoughValidData {
			needed: long,
			valid: len - first,
		});
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	Ok((data, first, short, long, len, chosen))
}

#[inline(always)]
fn apo_compute_into(data: &[f64], first: usize, short: usize, long: usize, kernel: Kernel, out: &mut [f64]) {
	unsafe {
		#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
		{
			if matches!(kernel, Kernel::Scalar | Kernel::ScalarBatch) {
				apo_simd128(data, short, long, first, out);
				return;
			}
		}
		
		match kernel {
			Kernel::Scalar | Kernel::ScalarBatch => apo_scalar(data, short, long, first, out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => apo_avx2(data, short, long, first, out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => apo_avx512(data, short, long, first, out),
			_ => unreachable!(),
		}
	}
}

pub fn apo_with_kernel(input: &ApoInput, kernel: Kernel) -> Result<ApoOutput, ApoError> {
	let (data, first, short, long, len, chosen) = apo_prepare(input, kernel)?;

	// Calculate warmup period: first valid data point
	let warmup_period = first;

	// Use zero-copy allocation with NaN prefix
	let mut out = alloc_with_nan_prefix(len, warmup_period);

	apo_compute_into(data, first, short, long, chosen, &mut out);

	Ok(ApoOutput { values: out })
}

// --- Scalar Kernel

#[inline(always)]
pub fn apo_scalar(data: &[f64], short: usize, long: usize, first: usize, out: &mut [f64]) {
	let alpha_short = 2.0 / (short as f64 + 1.0);
	let alpha_long = 2.0 / (long as f64 + 1.0);

	let mut short_ema = data[first];
	let mut long_ema = data[first];

	// Start from first valid index - warmup region already has NaN values
	for i in first..data.len() {
		let price = data[i];
		if i == first {
			short_ema = price;
			long_ema = price;
			out[i] = short_ema - long_ema; // This will be 0.0
			continue;
		}
		short_ema = alpha_short * price + (1.0 - alpha_short) * short_ema;
		long_ema = alpha_long * price + (1.0 - alpha_long) * long_ema;
		out[i] = short_ema - long_ema;
	}
}

// --- AVX2/AVX512 Kernels: Stubs

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn apo_avx2(data: &[f64], short: usize, long: usize, first: usize, out: &mut [f64]) {
	let alpha_short = 2.0 / (short as f64 + 1.0);
	let alpha_long = 2.0 / (long as f64 + 1.0);
	
	// For vectorization, we need the complements
	let one_minus_alpha_short = 1.0 - alpha_short;
	let one_minus_alpha_long = 1.0 - alpha_long;

	let mut short_ema = data[first];
	let mut long_ema = data[first];

	// Initialize first value
	out[first] = 0.0; // short_ema - long_ema

	// Process remaining values
	for i in (first + 1)..data.len() {
		let price = data[i];
		short_ema = alpha_short * price + one_minus_alpha_short * short_ema;
		long_ema = alpha_long * price + one_minus_alpha_long * long_ema;
		out[i] = short_ema - long_ema;
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn apo_avx512(data: &[f64], short: usize, long: usize, first: usize, out: &mut [f64]) {
	// Choose between short/long variants based on period
	if long <= 32 {
		apo_avx512_short(data, short, long, first, out);
	} else {
		apo_avx512_long(data, short, long, first, out);
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn apo_avx512_short(data: &[f64], short: usize, long: usize, first: usize, out: &mut [f64]) {
	let alpha_short = 2.0 / (short as f64 + 1.0);
	let alpha_long = 2.0 / (long as f64 + 1.0);
	
	let one_minus_alpha_short = 1.0 - alpha_short;
	let one_minus_alpha_long = 1.0 - alpha_long;

	let mut short_ema = data[first];
	let mut long_ema = data[first];

	// Initialize first value
	out[first] = 0.0;

	// Process remaining values with AVX512 optimizations
	for i in (first + 1)..data.len() {
		let price = data[i];
		short_ema = alpha_short * price + one_minus_alpha_short * short_ema;
		long_ema = alpha_long * price + one_minus_alpha_long * long_ema;
		out[i] = short_ema - long_ema;
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn apo_avx512_long(data: &[f64], short: usize, long: usize, first: usize, out: &mut [f64]) {
	// For longer periods, use the same approach
	// In a real implementation, this might use different optimizations
	apo_avx512_short(data, short, long, first, out);
}

// --- SIMD128 for WASM

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
unsafe fn apo_simd128(data: &[f64], short: usize, long: usize, first: usize, out: &mut [f64]) {
	use core::arch::wasm32::*;
	
	let alpha_short = 2.0 / (short as f64 + 1.0);
	let alpha_long = 2.0 / (long as f64 + 1.0);
	
	let one_minus_alpha_short = 1.0 - alpha_short;
	let one_minus_alpha_long = 1.0 - alpha_long;
	
	// Initialize EMAs with the first valid value
	let mut short_ema = data[first];
	let mut long_ema = data[first];
	
	// First value is always 0.0 (difference between identical EMAs)
	out[first] = 0.0;
	
	// Process values using SIMD where possible
	let alpha_short_vec = f64x2_splat(alpha_short);
	let alpha_long_vec = f64x2_splat(alpha_long);
	let one_minus_alpha_short_vec = f64x2_splat(one_minus_alpha_short);
	let one_minus_alpha_long_vec = f64x2_splat(one_minus_alpha_long);
	
	let mut i = first + 1;
	
	// Process pairs of values
	while i + 1 < data.len() {
		// Load two consecutive values
		let price_vec = v128_load(&data[i] as *const f64 as *const v128);
		
		// Create EMA vectors
		let short_ema_vec = f64x2_splat(short_ema);
		let long_ema_vec = f64x2_splat(long_ema);
		
		// Calculate new EMAs for both values
		let new_short_ema_vec = f64x2_add(
			f64x2_mul(alpha_short_vec, price_vec),
			f64x2_mul(one_minus_alpha_short_vec, short_ema_vec)
		);
		
		let new_long_ema_vec = f64x2_add(
			f64x2_mul(alpha_long_vec, price_vec),
			f64x2_mul(one_minus_alpha_long_vec, long_ema_vec)
		);
		
		// Calculate APO (short_ema - long_ema)
		let apo_vec = f64x2_sub(new_short_ema_vec, new_long_ema_vec);
		
		// Store results
		v128_store(&mut out[i] as *mut f64 as *mut v128, apo_vec);
		
		// Update EMAs for next iteration (use second value)
		short_ema = f64x2_extract_lane::<1>(new_short_ema_vec);
		long_ema = f64x2_extract_lane::<1>(new_long_ema_vec);
		
		i += 2;
	}
	
	// Handle remaining value if any
	if i < data.len() {
		let price = data[i];
		short_ema = alpha_short * price + one_minus_alpha_short * short_ema;
		long_ema = alpha_long * price + one_minus_alpha_long * long_ema;
		out[i] = short_ema - long_ema;
	}
}

// --- Batch, Streaming, and Builder APIs

#[derive(Clone, Debug)]
pub struct ApoStream {
	short: usize,
	long: usize,
	alpha_short: f64,
	alpha_long: f64,
	short_ema: f64,
	long_ema: f64,
	filled: bool,
	nan_leading: usize,
	seen: usize,
}

impl ApoStream {
	pub fn try_new(params: ApoParams) -> Result<Self, ApoError> {
		let short = params.short_period.unwrap_or(10);
		let long = params.long_period.unwrap_or(20);
		if short == 0 || long == 0 {
			return Err(ApoError::InvalidPeriod { short, long });
		}
		if short >= long {
			return Err(ApoError::ShortPeriodNotLessThanLong { short, long });
		}
		Ok(Self {
			short,
			long,
			alpha_short: 2.0 / (short as f64 + 1.0),
			alpha_long: 2.0 / (long as f64 + 1.0),
			short_ema: f64::NAN,
			long_ema: f64::NAN,
			filled: false,
			nan_leading: 0,
			seen: 0,
		})
	}

	#[inline(always)]
	pub fn update(&mut self, price: f64) -> Option<f64> {
		if !self.filled && price.is_nan() {
			self.nan_leading += 1;
			return None;
		}
		if !self.filled {
			self.short_ema = price;
			self.long_ema = price;
			self.filled = true;
			self.seen = 1;
			return Some(0.0);
		}
		self.seen += 1;
		self.short_ema = self.alpha_short * price + (1.0 - self.alpha_short) * self.short_ema;
		self.long_ema = self.alpha_long * price + (1.0 - self.alpha_long) * self.long_ema;
		Some(self.short_ema - self.long_ema)
	}
}

/// Helper function for WASM bindings - writes directly to output slice with zero allocations
pub fn apo_into_slice(dst: &mut [f64], input: &ApoInput, kern: Kernel) -> Result<(), ApoError> {
	let (data, first, short, long, len, chosen) = apo_prepare(input, kern)?;
	
	if dst.len() != len {
		return Err(ApoError::InvalidPeriod {
			short: dst.len(),
			long: len,
		});
	}
	
	apo_compute_into(data, first, short, long, chosen, dst);
	
	// Fill warmup period with NaN
	for v in &mut dst[..first] {
		*v = f64::NAN;
	}
	
	Ok(())
}

// --- Batch Sweeping API

#[derive(Clone, Debug)]
pub struct ApoBatchRange {
	pub short: (usize, usize, usize),
	pub long: (usize, usize, usize),
}
impl Default for ApoBatchRange {
	fn default() -> Self {
		Self {
			short: (5, 20, 5),
			long: (15, 50, 5),
		}
	}
}

#[derive(Clone, Debug, Default)]
pub struct ApoBatchBuilder {
	range: ApoBatchRange,
	kernel: Kernel,
}
impl ApoBatchBuilder {
	pub fn new() -> Self {
		Self::default()
	}
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	pub fn short_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.short = (start, end, step);
		self
	}
	pub fn short_static(mut self, s: usize) -> Self {
		self.range.short = (s, s, 0);
		self
	}
	pub fn long_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.long = (start, end, step);
		self
	}
	pub fn long_static(mut self, s: usize) -> Self {
		self.range.long = (s, s, 0);
		self
	}
	pub fn apply_slice(self, data: &[f64]) -> Result<ApoBatchOutput, ApoError> {
		apo_batch_with_kernel(data, &self.range, self.kernel)
	}
	pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<ApoBatchOutput, ApoError> {
		ApoBatchBuilder::new().kernel(k).apply_slice(data)
	}
	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<ApoBatchOutput, ApoError> {
		let slice = source_type(c, src);
		self.apply_slice(slice)
	}
	pub fn with_default_candles(c: &Candles) -> Result<ApoBatchOutput, ApoError> {
		ApoBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
	}
}

#[derive(Clone, Debug)]
pub struct ApoBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<ApoParams>,
	pub rows: usize,
	pub cols: usize,
}
impl ApoBatchOutput {
	pub fn row_for_params(&self, p: &ApoParams) -> Option<usize> {
		self.combos.iter().position(|c| {
			c.short_period.unwrap_or(10) == p.short_period.unwrap_or(10)
				&& c.long_period.unwrap_or(20) == p.long_period.unwrap_or(20)
		})
	}
	pub fn values_for(&self, p: &ApoParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

// --- Grid Expansion

#[inline(always)]
fn expand_grid(r: &ApoBatchRange) -> Vec<ApoParams> {
	fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let shorts = axis(r.short);
	let longs = axis(r.long);
	let mut out = Vec::with_capacity(shorts.len() * longs.len());
	for &s in &shorts {
		for &l in &longs {
			if s < l && s > 0 && l > 0 {
				out.push(ApoParams {
					short_period: Some(s),
					long_period: Some(l),
				});
			}
		}
	}
	out
}

// --- Batch Slice API

#[inline(always)]
pub fn apo_batch_with_kernel(data: &[f64], sweep: &ApoBatchRange, k: Kernel) -> Result<ApoBatchOutput, ApoError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => {
			return Err(ApoError::InvalidPeriod { short: 0, long: 0 });
		}
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	apo_batch_par_slice(data, sweep, simd)
}

#[inline(always)]
pub fn apo_batch_slice(data: &[f64], sweep: &ApoBatchRange, kern: Kernel) -> Result<ApoBatchOutput, ApoError> {
	apo_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn apo_batch_par_slice(data: &[f64], sweep: &ApoBatchRange, kern: Kernel) -> Result<ApoBatchOutput, ApoError> {
	apo_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn apo_batch_inner(
	data: &[f64],
	sweep: &ApoBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<ApoBatchOutput, ApoError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(ApoError::InvalidPeriod { short: 0, long: 0 });
	}
	let first = data.iter().position(|x| !x.is_nan()).ok_or(ApoError::AllValuesNaN)?;
	let max_long = combos.iter().map(|c| c.long_period.unwrap()).max().unwrap();
	if data.len() - first < max_long {
		return Err(ApoError::NotEnoughValidData {
			needed: max_long,
			valid: data.len() - first,
		});
	}

	let rows = combos.len();
	let cols = data.len();

	// Step 1: Allocate uninitialized matrix
	let mut buf_mu = make_uninit_matrix(rows, cols);

	// Step 2: Calculate warmup periods for each row
	let warm: Vec<usize> = combos
		.iter()
		.map(|_c| {
			// For APO, warmup is simply the first valid data index
			first
		})
		.collect();

	// Step 3: Initialize NaN prefixes for each row
	init_matrix_prefixes(&mut buf_mu, cols, &warm);

	// Step 4: Convert to mutable slice for computation
	let mut buf_guard = ManuallyDrop::new(buf_mu);
	let values: &mut [f64] =
		unsafe { core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len()) };

	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let s = combos[row].short_period.unwrap();
		let l = combos[row].long_period.unwrap();
		match kern {
			Kernel::Scalar => apo_row_scalar(data, first, s, l, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => apo_row_avx2(data, first, s, l, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => apo_row_avx512(data, first, s, l, out_row),
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

	// Step 6: Reclaim as Vec<f64>
	let values = unsafe {
		Vec::from_raw_parts(
			buf_guard.as_mut_ptr() as *mut f64,
			buf_guard.len(),
			buf_guard.capacity(),
		)
	};

	Ok(ApoBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
fn apo_batch_inner_into(
	data: &[f64],
	sweep: &ApoBatchRange,
	kern: Kernel,
	parallel: bool,
	out: &mut [f64],
) -> Result<Vec<ApoParams>, ApoError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(ApoError::InvalidPeriod { short: 0, long: 0 });
	}
	
	let first = data.iter().position(|x| !x.is_nan()).ok_or(ApoError::AllValuesNaN)?;
	let max_long = combos.iter().map(|c| c.long_period.unwrap()).max().unwrap();
	if data.len() - first < max_long {
		return Err(ApoError::NotEnoughValidData {
			needed: max_long,
			valid: data.len() - first,
		});
	}

	let rows = combos.len();
	let cols = data.len();

	// Initialize NaN prefixes for each row
	for row in 0..rows {
		let warmup = first;
		for col in 0..warmup {
			out[row * cols + col] = f64::NAN;
		}
	}

	// Compute APO values for each parameter combination
	if parallel && !cfg!(target_arch = "wasm32") {
		#[cfg(not(target_arch = "wasm32"))]
		{
			use rayon::prelude::*;
			out.par_chunks_mut(cols)
				.enumerate()
				.for_each(|(row, out_row)| {
					let params = &combos[row];
					unsafe {
						match kern {
							Kernel::Scalar => apo_row_scalar(
								data,
								first,
								params.short_period.unwrap(),
								params.long_period.unwrap(),
								out_row
							),
							#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
							Kernel::Avx2 => apo_row_avx2(
								data,
								first,
								params.short_period.unwrap(),
								params.long_period.unwrap(),
								out_row
							),
							#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
							Kernel::Avx512 => apo_row_avx512(
								data,
								first,
								params.short_period.unwrap(),
								params.long_period.unwrap(),
								out_row
							),
							_ => unreachable!(),
						}
					}
				});
		}
	} else {
		for (row, out_row) in out.chunks_mut(cols).enumerate() {
			let params = &combos[row];
			unsafe {
				match kern {
					Kernel::Scalar => apo_row_scalar(
						data,
						first,
						params.short_period.unwrap(),
						params.long_period.unwrap(),
						out_row
					),
					#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
					Kernel::Avx2 => apo_row_avx2(
						data,
						first,
						params.short_period.unwrap(),
						params.long_period.unwrap(),
						out_row
					),
					#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
					Kernel::Avx512 => apo_row_avx512(
						data,
						first,
						params.short_period.unwrap(),
						params.long_period.unwrap(),
						out_row
					),
					_ => unreachable!(),
				}
			}
		}
	}

	Ok(combos)
}

// --- Row Kernels

#[inline(always)]
pub unsafe fn apo_row_scalar(data: &[f64], first: usize, short: usize, long: usize, out: &mut [f64]) {
	apo_scalar(data, short, long, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn apo_row_avx2(data: &[f64], first: usize, short: usize, long: usize, out: &mut [f64]) {
	apo_avx2(data, short, long, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn apo_row_avx512(data: &[f64], first: usize, short: usize, long: usize, out: &mut [f64]) {
	if long <= 32 {
		apo_row_avx512_short(data, first, short, long, out)
	} else {
		apo_row_avx512_long(data, first, short, long, out)
	}
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn apo_row_avx512_short(data: &[f64], first: usize, short: usize, long: usize, out: &mut [f64]) {
	apo_avx512_short(data, short, long, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn apo_row_avx512_long(data: &[f64], first: usize, short: usize, long: usize, out: &mut [f64]) {
	apo_avx512_long(data, short, long, first, out)
}

// --- Tests

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_apo_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let default_params = ApoParams {
			short_period: None,
			long_period: None,
		};
		let input = ApoInput::from_candles(&candles, "close", default_params);
		let output = apo_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_apo_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = ApoInput::with_default_candles(&candles);
		let result = apo_with_kernel(&input, kernel)?;
		let expected_last_five = [-429.8, -401.6, -386.1, -357.9, -374.1];
		let start_index = result.values.len().saturating_sub(5);
		let result_last_five = &result.values[start_index..];
		for (i, &value) in result_last_five.iter().enumerate() {
			assert!(
				(value - expected_last_five[i]).abs() < 1e-1,
				"[{}] APO value mismatch at index {}: expected {}, got {}",
				test_name,
				i,
				expected_last_five[i],
				value
			);
		}
		Ok(())
	}

	fn check_apo_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = ApoInput::with_default_candles(&candles);
		match input.data {
			ApoData::Candles { source, .. } => assert_eq!(source, "close"),
			_ => panic!("Expected ApoData::Candles"),
		}
		let output = apo_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_apo_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let input_data = [10.0, 20.0, 30.0];
		let params = ApoParams {
			short_period: Some(0),
			long_period: Some(20),
		};
		let input = ApoInput::from_slice(&input_data, params);
		let res = apo_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] APO should fail with zero period", test_name);
		Ok(())
	}

	fn check_apo_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let empty_data: Vec<f64> = vec![];
		let params = ApoParams::default();
		let input = ApoInput::from_slice(&empty_data, params);
		let result = apo_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}

	fn check_apo_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = ApoParams::default();
		
		// Batch calculation
		let input = ApoInput::from_candles(&candles, "close", params.clone());
		let batch_result = apo_with_kernel(&input, kernel)?;
		
		// Streaming calculation
		let mut stream = ApoStream::try_new(params)?;
		let mut streaming_results = vec![];
		
		for &close in &candles.close {
			if let Some(val) = stream.update(close) {
				streaming_results.push(val);
			} else {
				streaming_results.push(f64::NAN);
			}
		}
		
		// Compare results (allowing for small floating point differences)
		assert_eq!(batch_result.values.len(), streaming_results.len());
		let first_valid = candles.close.iter().position(|x| !x.is_nan()).unwrap_or(0);
		
		for i in first_valid..batch_result.values.len() {
			if !batch_result.values[i].is_nan() && !streaming_results[i].is_nan() {
				let diff = (batch_result.values[i] - streaming_results[i]).abs();
				assert!(diff < 1e-10, "Streaming mismatch at index {}: batch={}, stream={}", 
					i, batch_result.values[i], streaming_results[i]);
			}
		}
		Ok(())
	}

	fn check_apo_period_invalid(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data_small = [10.0, 20.0, 30.0];
		let params = ApoParams {
			short_period: Some(20),
			long_period: Some(10),
		};
		let input = ApoInput::from_slice(&data_small, params);
		let res = apo_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] APO should fail with invalid period", test_name);
		Ok(())
	}

	fn check_apo_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let single_point = [42.0];
		let params = ApoParams {
			short_period: Some(9),
			long_period: Some(10),
		};
		let input = ApoInput::from_slice(&single_point, params);
		let res = apo_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] APO should fail with insufficient data", test_name);
		Ok(())
	}

	fn check_apo_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let first_params = ApoParams {
			short_period: Some(10),
			long_period: Some(20),
		};
		let first_input = ApoInput::from_candles(&candles, "close", first_params);
		let first_result = apo_with_kernel(&first_input, kernel)?;
		let second_params = ApoParams {
			short_period: Some(5),
			long_period: Some(15),
		};
		let second_input = ApoInput::from_slice(&first_result.values, second_params);
		let second_result = apo_with_kernel(&second_input, kernel)?;
		assert_eq!(second_result.values.len(), first_result.values.len());
		Ok(())
	}

	fn check_apo_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = ApoInput::from_candles(
			&candles,
			"close",
			ApoParams {
				short_period: Some(10),
				long_period: Some(20),
			},
		);
		let res = apo_with_kernel(&input, kernel)?;
		assert_eq!(res.values.len(), candles.close.len());
		if res.values.len() > 30 {
			for (i, &val) in res.values[30..].iter().enumerate() {
				assert!(
					!val.is_nan(),
					"[{}] Found unexpected NaN at out-index {}",
					test_name,
					30 + i
				);
			}
		}
		Ok(())
	}

	#[cfg(debug_assertions)]
	fn check_apo_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let input = ApoInput::from_candles(&candles, "close", ApoParams::default());
		let output = apo_with_kernel(&input, kernel)?;

		for (i, &val) in output.values.iter().enumerate() {
			if val.is_nan() {
				continue;
			}

			let bits = val.to_bits();

			if bits == 0x11111111_11111111 {
				panic!(
					"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {}",
					test_name, val, bits, i
				);
			}

			if bits == 0x22222222_22222222 {
				panic!(
					"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {}",
					test_name, val, bits, i
				);
			}

			if bits == 0x33333333_33333333 {
				panic!(
					"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {}",
					test_name, val, bits, i
				);
			}
		}

		Ok(())
	}

	#[cfg(not(debug_assertions))]
	fn check_apo_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		Ok(())
	}

	#[cfg(feature = "proptest")]
	#[allow(clippy::float_cmp)]
	fn check_apo_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		use proptest::prelude::*;
		skip_if_unsupported!(kernel, test_name);

		// Strategy 1: Random price data with realistic period ranges
		let random_data_strat = (3usize..=20, 10usize..=50)
			.prop_filter("short < long", |(s, l)| s < l)
			.prop_flat_map(|(short_period, long_period)| {
				// Generate data length that's sufficient for the long period
				let len = long_period * 2..400;
				(
					prop::collection::vec(
						// Generate realistic price data
						(10f64..10000f64).prop_filter("finite", |x| x.is_finite()),
						len,
					),
					Just(short_period),
					Just(long_period),
					Just("random"),
				)
			});

		// Strategy 2: Constant data (APO should stay near 0)
		let constant_data_strat = (3usize..=20, 10usize..=50)
			.prop_filter("short < long", |(s, l)| s < l)
			.prop_flat_map(|(short_period, long_period)| {
				let len = long_period * 2..200;
				(
					prop::collection::vec(Just(100.0f64), len),
					Just(short_period),
					Just(long_period),
					Just("constant"),
				)
			});

		// Strategy 3: Trending data (monotonic increase/decrease)
		let trending_data_strat = (3usize..=20, 10usize..=50)
			.prop_filter("short < long", |(s, l)| s < l)
			.prop_flat_map(|(short_period, long_period)| {
				let len = long_period * 2..200;
				(
					(50..150usize).prop_flat_map(move |size| {
						(0.1f64..5.0).prop_map(move |slope| {
							(0..size).map(|i| 100.0 + slope * i as f64).collect::<Vec<f64>>()
						})
					}),
					Just(short_period),
					Just(long_period),
					Just("trending"),
				)
			});

		// Combine all strategies
		let strat = prop_oneof![
			random_data_strat,
			constant_data_strat,
			trending_data_strat,
		];

		proptest::test_runner::TestRunner::default()
			.run(&strat, |(data, short_period, long_period, data_type)| {
				let params = ApoParams {
					short_period: Some(short_period),
					long_period: Some(long_period),
				};
				let input = ApoInput::from_slice(&data, params.clone());

				let result = apo_with_kernel(&input, kernel);
				prop_assert!(result.is_ok(), "APO computation failed: {:?}", result);

				let ApoOutput { values: out } = result.unwrap();

				// Property 1: Output length matches input
				prop_assert_eq!(out.len(), data.len(), "Output length mismatch");

				// Property 2: First valid value should be 0.0 (both EMAs start with same value)
				let first_valid = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
				if first_valid < data.len() {
					prop_assert!(
						out[first_valid].abs() < 1e-10,
						"First APO value should be 0, got {} at index {}",
						out[first_valid],
						first_valid
					);
				}

				// Property 3: All values after warmup should be finite
				for i in first_valid..out.len() {
					prop_assert!(
						out[i].is_finite(),
						"APO output at index {} should be finite, got {}",
						i,
						out[i]
					);
				}

				// Property 4: Values should be bounded reasonably
				// APO is the difference between two EMAs, both bounded by data range
				// In practice, APO magnitude is much smaller than data range
				let data_min = data.iter().filter(|x| x.is_finite()).fold(f64::INFINITY, |a, &b| a.min(b));
				let data_max = data.iter().filter(|x| x.is_finite()).fold(f64::NEG_INFINITY, |a, &b| a.max(b));
				let data_range = data_max - data_min;
				
				// Tighter bound: APO typically doesn't exceed 30% of data range
				let apo_bound = data_range * 0.3;
				
				for i in first_valid..out.len() {
					prop_assert!(
						out[i].abs() <= apo_bound,
						"APO value at index {} exceeds expected bound: {} > {}",
						i,
						out[i].abs(),
						apo_bound
					);
				}

				// Property 5: Data-type specific validations
				match data_type {
					"constant" => {
						// For constant data, APO should remain very close to 0
						for i in first_valid..out.len() {
							prop_assert!(
								out[i].abs() < 1e-9,
								"APO should be ~0 for constant data, got {} at index {}",
								out[i],
								i
							);
						}
					}
					"trending" => {
						// For monotonically increasing data, APO should generally be positive after initial warmup
						// (short EMA responds faster, stays above long EMA)
						if data.len() > long_period * 2 {
							let check_start = first_valid + long_period;
							let check_end = out.len();
							if check_start < check_end {
								// Check if data is increasing or decreasing
								let is_increasing = data[first_valid] < data[data.len() - 1];
								
								// Count positive vs negative values in the latter part
								let positive_count = out[check_start..check_end]
									.iter()
									.filter(|&&v| v > 0.0)
									.count();
								let total_count = check_end - check_start;
								
								if is_increasing {
									// For uptrend, expect mostly positive APO values
									prop_assert!(
										positive_count > total_count / 2,
										"APO should be mostly positive for uptrend, got {} positive out of {}",
										positive_count,
										total_count
									);
								} else {
									// For downtrend, expect mostly negative APO values
									prop_assert!(
										positive_count < total_count / 2,
										"APO should be mostly negative for downtrend, got {} positive out of {}",
										positive_count,
										total_count
									);
								}
							}
						}
					}
					_ => {} // Random data, no specific pattern expected
				}

				// Property 6: Manual calculation verification for first few values
				if data.len() >= 3 && first_valid + 2 < data.len() {
					let alpha_short = 2.0 / (short_period as f64 + 1.0);
					let alpha_long = 2.0 / (long_period as f64 + 1.0);

					// First value after warmup
					let mut short_ema = data[first_valid];
					let mut long_ema = data[first_valid];
					let expected_first = 0.0; // Both EMAs start equal
					prop_assert!(
						(out[first_valid] - expected_first).abs() < 1e-9,
						"First value mismatch: expected {}, got {}",
						expected_first,
						out[first_valid]
					);

					// Second value
					if first_valid + 1 < data.len() {
						let price = data[first_valid + 1];
						short_ema = alpha_short * price + (1.0 - alpha_short) * short_ema;
						long_ema = alpha_long * price + (1.0 - alpha_long) * long_ema;
						let expected_second = short_ema - long_ema;
						prop_assert!(
							(out[first_valid + 1] - expected_second).abs() < 1e-9,
							"Second value mismatch: expected {}, got {}",
							expected_second,
							out[first_valid + 1]
						);
					}

					// Third value
					if first_valid + 2 < data.len() {
						let price = data[first_valid + 2];
						short_ema = alpha_short * price + (1.0 - alpha_short) * short_ema;
						long_ema = alpha_long * price + (1.0 - alpha_long) * long_ema;
						let expected_third = short_ema - long_ema;
						prop_assert!(
							(out[first_valid + 2] - expected_third).abs() < 1e-9,
							"Third value mismatch: expected {}, got {}",
							expected_third,
							out[first_valid + 2]
						);
					}
				}

				// Property 7: Kernel consistency - all kernels should produce the same result
				let ref_output = apo_with_kernel(&input, Kernel::Scalar);
				prop_assert!(ref_output.is_ok(), "Reference scalar computation failed");
				let ApoOutput { values: ref_out } = ref_output.unwrap();

				for (i, (&val, &ref_val)) in out.iter().zip(ref_out.iter()).enumerate() {
					if !val.is_finite() || !ref_val.is_finite() {
						// Both should be NaN at the same positions
						prop_assert_eq!(
							val.is_nan(), ref_val.is_nan(),
							"NaN mismatch at index {}: kernel={}, scalar={}",
							i, val, ref_val
						);
					} else {
						// Values should be nearly identical (allowing for small floating-point differences)
						let diff = (val - ref_val).abs();
						let ulp_diff = val.to_bits().abs_diff(ref_val.to_bits());
						prop_assert!(
							diff <= 1e-9 || ulp_diff <= 4,
							"Kernel mismatch at index {}: {} vs {} (diff: {}, ULP: {})",
							i, val, ref_val, diff, ulp_diff
						);
					}
				}

				// Property 8: Period relationship
				prop_assert!(short_period < long_period, "Short period must be less than long period");

				// Property 9: Streaming consistency
				let mut stream = ApoStream::try_new(params).unwrap();
				let mut stream_values = Vec::new();
				for &price in &data {
					if let Some(val) = stream.update(price) {
						stream_values.push(val);
					} else {
						stream_values.push(f64::NAN);
					}
				}
				
				// Compare streaming vs batch (allowing for floating-point differences)
				for i in first_valid..out.len() {
					if out[i].is_finite() && stream_values[i].is_finite() {
						let diff = (out[i] - stream_values[i]).abs();
						prop_assert!(
							diff < 1e-10,
							"Streaming mismatch at index {}: batch={}, stream={}, diff={}",
							i, out[i], stream_values[i], diff
						);
					}
				}

				Ok(())
			})
			.map_err(|e| e.into())
	}

	macro_rules! generate_all_apo_tests {
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

	generate_all_apo_tests!(
		check_apo_partial_params,
		check_apo_accuracy,
		check_apo_default_candles,
		check_apo_zero_period,
		check_apo_empty_input,
		check_apo_streaming,
		check_apo_period_invalid,
		check_apo_very_small_dataset,
		check_apo_reinput,
		check_apo_nan_handling,
		check_apo_no_poison,
		check_apo_property
	);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let output = ApoBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;
		let def = ApoParams::default();
		let row = output.values_for(&def).expect("default row missing");
		assert_eq!(row.len(), c.close.len());
		Ok(())
	}
	
	#[cfg(debug_assertions)]
	fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);
		
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		
		// Test various parameter sweep configurations
		let test_configs = vec![
			// (short_start, short_end, short_step, long_start, long_end, long_step)
			(2, 10, 2, 15, 30, 5),      // Small to medium periods
			(5, 25, 5, 30, 50, 10),     // Medium periods
			(10, 20, 5, 25, 45, 10),    // Common trading periods
			(12, 12, 0, 26, 26, 0),     // Static periods (default-like)
			(3, 9, 3, 10, 20, 5),       // Small range sweep
		];
		
		for (cfg_idx, &(s_start, s_end, s_step, l_start, l_end, l_step)) in test_configs.iter().enumerate() {
			let output = ApoBatchBuilder::new()
				.kernel(kernel)
				.short_range(s_start, s_end, s_step)
				.long_range(l_start, l_end, l_step)
				.apply_candles(&c, "close")?;
				
			for (idx, &val) in output.values.iter().enumerate() {
				if val.is_nan() {
					continue;
				}
				
				let bits = val.to_bits();
				let row = idx / output.cols;
				let col = idx % output.cols;
				let combo = &output.combos[row];
				
				// Check all three poison patterns
				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: short={}, long={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.short_period.unwrap_or(12),
						combo.long_period.unwrap_or(26)
					);
				}
				
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: short={}, long={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.short_period.unwrap_or(12),
						combo.long_period.unwrap_or(26)
					);
				}
				
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: short={}, long={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.short_period.unwrap_or(12),
						combo.long_period.unwrap_or(26)
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

	#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
	#[test]
	fn test_apo_simd128_correctness() {
		let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
		let short_period = 3;
		let long_period = 5;
		let params = ApoParams {
			short_period: Some(short_period),
			long_period: Some(long_period),
		};
		let input = ApoInput::from_slice(&data, params);

		// Force scalar kernel (which will use SIMD128 on WASM)
		let scalar_output = apo_with_kernel(&input, Kernel::Scalar).unwrap();

		// Create a pure scalar version for comparison
		let mut pure_scalar_output = vec![f64::NAN; data.len()];
		let first = 0; // APO starts from first data point
		unsafe {
			apo_scalar(&data, short_period, long_period, first, &mut pure_scalar_output);
		}

		// Compare results
		assert_eq!(scalar_output.values.len(), pure_scalar_output.len());
		for (i, (simd_val, scalar_val)) in scalar_output
			.values
			.iter()
			.zip(pure_scalar_output.iter())
			.enumerate()
		{
			if scalar_val.is_nan() {
				assert!(simd_val.is_nan(), "SIMD128 NaN mismatch at index {}", i);
			} else {
				assert!(
					(scalar_val - simd_val).abs() < 1e-10,
					"SIMD128 mismatch at index {}: scalar={}, simd128={}",
					i,
					scalar_val,
					simd_val
				);
			}
		}
	}
}

// ================================================================================================
// Python Bindings
// ================================================================================================

#[cfg(feature = "python")]
#[pyfunction(name = "apo")]
#[pyo3(signature = (data, short_period=10, long_period=20, kernel=None))]
pub fn apo_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	short_period: usize,
	long_period: usize,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
	use numpy::{IntoPyArray, PyArrayMethods};

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, false)?;

	let params = ApoParams {
		short_period: Some(short_period),
		long_period: Some(long_period),
	};
	let apo_in = ApoInput::from_slice(slice_in, params);

	// Get Vec<f64> from Rust function and convert to NumPy with zero-copy
	let result_vec: Vec<f64> = py
		.allow_threads(|| apo_with_kernel(&apo_in, kern).map(|o| o.values))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "ApoStream")]
pub struct ApoStreamPy {
	stream: ApoStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl ApoStreamPy {
	#[new]
	fn new(short_period: usize, long_period: usize) -> PyResult<Self> {
		let params = ApoParams {
			short_period: Some(short_period),
			long_period: Some(long_period),
		};
		let stream = ApoStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(ApoStreamPy { stream })
	}

	/// Updates the stream with a new value and returns the calculated APO value.
	/// Returns `None` if the buffer is not yet full.
	fn update(&mut self, value: f64) -> Option<f64> {
		self.stream.update(value)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "apo_batch")]
#[pyo3(signature = (data, short_period_range, long_period_range, kernel=None))]
pub fn apo_batch_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	short_period_range: (usize, usize, usize),
	long_period_range: (usize, usize, usize),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, true)?;

	let sweep = ApoBatchRange {
		short: short_period_range,
		long: long_period_range,
	};

	// Expand grid to know dimensions
	let combos = expand_grid(&sweep);
	if combos.is_empty() {
		return Err(PyValueError::new_err("No valid parameter combinations"));
	}
	let rows = combos.len();
	let cols = slice_in.len();

	// Pre-allocate uninitialized NumPy array (acceptable for batch operations)
	let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let slice_out = unsafe { out_arr.as_slice_mut()? };

	// Heavy work without the GIL
	let combos = py
		.allow_threads(|| -> Result<Vec<ApoParams>, ApoError> {
			// Resolve Kernel::Auto to a specific kernel
			let kernel = match kern {
				Kernel::Auto => detect_best_batch_kernel(),
				k => k,
			};

			let result = apo_batch_with_kernel(slice_in, &sweep, kernel)?;

			// Copy results to the pre-allocated buffer
			slice_out.copy_from_slice(&result.values);

			Ok(result.combos)
		})
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	// Build result dictionary
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

	Ok(dict)
}

// ================================================================================================
// WASM Bindings
// ================================================================================================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn apo_js(data: &[f64], short_period: usize, long_period: usize) -> Result<Vec<f64>, JsValue> {
	let params = ApoParams {
		short_period: Some(short_period),
		long_period: Some(long_period),
	};
	let input = ApoInput::from_slice(data, params);

	// Single allocation following WASM guide pattern
	let mut output = vec![0.0; data.len()];
	
	apo_into_slice(&mut output, &input, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn apo_into(
	in_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	short_period: usize,
	long_period: usize,
) -> Result<(), JsValue> {
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}
	
	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);
		let params = ApoParams {
			short_period: Some(short_period),
			long_period: Some(long_period),
		};
		let input = ApoInput::from_slice(data, params);
		
		if in_ptr == out_ptr as *const f64 {  // CRITICAL: Aliasing check
			let mut temp = vec![0.0; len];
			apo_into_slice(&mut temp, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			out.copy_from_slice(&temp);
		} else {
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			apo_into_slice(out, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn apo_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn apo_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe { let _ = Vec::from_raw_parts(ptr, len, len); }
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn apo_batch_js(
	data: &[f64],
	short_period_start: usize,
	short_period_end: usize,
	short_period_step: usize,
	long_period_start: usize,
	long_period_end: usize,
	long_period_step: usize,
) -> Result<Vec<f64>, JsValue> {
	let sweep = ApoBatchRange {
		short: (short_period_start, short_period_end, short_period_step),
		long: (long_period_start, long_period_end, long_period_step),
	};

	// Use the existing batch function with parallel=false for WASM
	apo_batch_inner(data, &sweep, Kernel::Scalar, false)
		.map(|output| output.values)
		.map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn apo_batch_metadata_js(
	short_period_start: usize,
	short_period_end: usize,
	short_period_step: usize,
	long_period_start: usize,
	long_period_end: usize,
	long_period_step: usize,
) -> Result<Vec<f64>, JsValue> {
	let sweep = ApoBatchRange {
		short: (short_period_start, short_period_end, short_period_step),
		long: (long_period_start, long_period_end, long_period_step),
	};

	let combos = expand_grid(&sweep);
	let mut metadata = Vec::with_capacity(combos.len() * 2);

	for combo in combos {
		metadata.push(combo.short_period.unwrap() as f64);
		metadata.push(combo.long_period.unwrap() as f64);
	}

	Ok(metadata)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn apo_batch_into(
	in_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	short_period_start: usize,
	short_period_end: usize,
	short_period_step: usize,
	long_period_start: usize,
	long_period_end: usize,
	long_period_step: usize,
) -> Result<usize, JsValue> {
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("null pointer passed to apo_batch_into"));
	}

	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);

		let sweep = ApoBatchRange {
			short: (short_period_start, short_period_end, short_period_step),
			long: (long_period_start, long_period_end, long_period_step),
		};

		let combos = expand_grid(&sweep);
		let rows = combos.len();
		let cols = len;

		let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);

		apo_batch_inner_into(data, &sweep, Kernel::Auto, false, out)
			.map_err(|e| JsValue::from_str(&e.to_string()))?;

		Ok(rows)
	}
}

// New ergonomic WASM API
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct ApoBatchConfig {
	pub short_period_range: (usize, usize, usize),
	pub long_period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct ApoBatchJsOutput {
	pub values: Vec<f64>,
	pub combos: Vec<ApoParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = apo_batch)]
pub fn apo_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	// 1. Deserialize the configuration object from JavaScript
	let config: ApoBatchConfig =
		serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

	let sweep = ApoBatchRange {
		short: config.short_period_range,
		long: config.long_period_range,
	};

	// 2. Run the existing core logic
	let output = apo_batch_inner(data, &sweep, Kernel::Scalar, false).map_err(|e| JsValue::from_str(&e.to_string()))?;

	// 3. Create the structured output
	let js_output = ApoBatchJsOutput {
		values: output.values,
		combos: output.combos,
		rows: output.rows,
		cols: output.cols,
	};

	// 4. Serialize the output struct into a JavaScript object
	serde_wasm_bindgen::to_value(&js_output).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}
