//! # Pivot Points (PIVOT)
//!
//! Support (S) and resistance (R) levels from High, Low, Close, Open prices.
//! Multiple calculation modes supported (Standard, Fibonacci, Demark, Camarilla, Woodie).
//!
//! ## Parameters
//! - **mode**: Calculation method. 0=Standard, 1=Fibonacci, 2=Demark, 3=Camarilla (default), 4=Woodie
//!
//! ## Errors
//! - **EmptyData**: Required field missing
//! - **AllValuesNaN**: All values are NaN
//! - **NotEnoughValidData**: Not enough valid data for calculation
//!
//! ## Returns
//! - **Ok(PivotOutput)** with 9 Vec<f64> levels, each input length
//! - **Err(PivotError)** on error

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
	alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel,
};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use thiserror::Error;

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

// ========== DATA/INPUT/OUTPUT STRUCTS ==========

#[derive(Debug, Clone)]
pub enum PivotData<'a> {
	Candles {
		candles: &'a Candles,
	},
	Slices {
		high: &'a [f64],
		low: &'a [f64],
		close: &'a [f64],
		open: &'a [f64],
	},
}

#[derive(Debug, Clone)]
pub struct PivotParams {
	pub mode: Option<usize>,
}
impl Default for PivotParams {
	fn default() -> Self {
		Self { mode: Some(3) }
	}
}

#[derive(Debug, Clone)]
pub struct PivotInput<'a> {
	pub data: PivotData<'a>,
	pub params: PivotParams,
}
impl<'a> PivotInput<'a> {
	#[inline]
	pub fn from_candles(candles: &'a Candles, params: PivotParams) -> Self {
		Self {
			data: PivotData::Candles { candles },
			params,
		}
	}
	#[inline]
	pub fn from_slices(
		high: &'a [f64],
		low: &'a [f64],
		close: &'a [f64],
		open: &'a [f64],
		params: PivotParams,
	) -> Self {
		Self {
			data: PivotData::Slices { high, low, close, open },
			params,
		}
	}
	#[inline]
	pub fn with_default_candles(candles: &'a Candles) -> Self {
		Self::from_candles(candles, PivotParams::default())
	}
	#[inline]
	pub fn get_mode(&self) -> usize {
		self.params.mode.unwrap_or_else(|| PivotParams::default().mode.unwrap())
	}
}
impl<'a> AsRef<PivotData<'a>> for PivotInput<'a> {
	fn as_ref(&self) -> &PivotData<'a> {
		&self.data
	}
}

#[derive(Debug, Clone)]
pub struct PivotOutput {
	pub r4: Vec<f64>,
	pub r3: Vec<f64>,
	pub r2: Vec<f64>,
	pub r1: Vec<f64>,
	pub pp: Vec<f64>,
	pub s1: Vec<f64>,
	pub s2: Vec<f64>,
	pub s3: Vec<f64>,
	pub s4: Vec<f64>,
}

#[derive(Debug, Error)]
pub enum PivotError {
	#[error("pivot: One or more required fields is empty.")]
	EmptyData,
	#[error("pivot: All values are NaN.")]
	AllValuesNaN,
	#[error("pivot: Not enough valid data after the first valid index.")]
	NotEnoughValidData,
}

// ========== BUILDER ==========

#[derive(Copy, Clone, Debug)]
pub struct PivotBuilder {
	mode: Option<usize>,
	kernel: Kernel,
}
impl Default for PivotBuilder {
	fn default() -> Self {
		Self {
			mode: None,
			kernel: Kernel::Auto,
		}
	}
}
impl PivotBuilder {
	#[inline(always)]
	pub fn new() -> Self {
		Self::default()
	}
	#[inline(always)]
	pub fn mode(mut self, mode: usize) -> Self {
		self.mode = Some(mode);
		self
	}
	#[inline(always)]
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline(always)]
	pub fn apply(self, candles: &Candles) -> Result<PivotOutput, PivotError> {
		let params = PivotParams { mode: self.mode };
		let input = PivotInput::from_candles(candles, params);
		pivot_with_kernel(&input, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slices(
		self,
		high: &[f64],
		low: &[f64],
		close: &[f64],
		open: &[f64],
	) -> Result<PivotOutput, PivotError> {
		let params = PivotParams { mode: self.mode };
		let input = PivotInput::from_slices(high, low, close, open, params);
		pivot_with_kernel(&input, self.kernel)
	}
}

// ========== MAIN INTERFACE FUNCTIONS ==========

#[inline]
pub fn pivot(input: &PivotInput) -> Result<PivotOutput, PivotError> {
	pivot_with_kernel(input, Kernel::Auto)
}

pub fn pivot_with_kernel(input: &PivotInput, kernel: Kernel) -> Result<PivotOutput, PivotError> {
	let (high, low, close, open) = match &input.data {
		PivotData::Candles { candles } => {
			let high = source_type(candles, "high");
			let low = source_type(candles, "low");
			let close = source_type(candles, "close");
			let open = source_type(candles, "open");
			(high, low, close, open)
		}
		PivotData::Slices { high, low, close, open } => (*high, *low, *close, *open),
	};
	let len = high.len();
	if high.is_empty() || low.is_empty() || close.is_empty() {
		return Err(PivotError::EmptyData);
	}
	if low.len() != len || close.len() != len || open.len() != len {
		return Err(PivotError::EmptyData);
	}
	let mode = input.get_mode();

	let mut first_valid_idx = None;
	for i in 0..len {
		let h = high[i];
		let l = low[i];
		let c = close[i];
		if !(h.is_nan() || l.is_nan() || c.is_nan()) {
			first_valid_idx = Some(i);
			break;
		}
	}
	let first_valid_idx = match first_valid_idx {
		Some(idx) => idx,
		None => return Err(PivotError::AllValuesNaN),
	};
	if first_valid_idx >= len {
		return Err(PivotError::NotEnoughValidData);
	}

	// Allocate output vectors with NaN prefix
	let mut r4 = alloc_with_nan_prefix(len, first_valid_idx);
	let mut r3 = alloc_with_nan_prefix(len, first_valid_idx);
	let mut r2 = alloc_with_nan_prefix(len, first_valid_idx);
	let mut r1 = alloc_with_nan_prefix(len, first_valid_idx);
	let mut pp = alloc_with_nan_prefix(len, first_valid_idx);
	let mut s1 = alloc_with_nan_prefix(len, first_valid_idx);
	let mut s2 = alloc_with_nan_prefix(len, first_valid_idx);
	let mut s3 = alloc_with_nan_prefix(len, first_valid_idx);
	let mut s4 = alloc_with_nan_prefix(len, first_valid_idx);

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};
	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => pivot_scalar(
				high,
				low,
				close,
				open,
				mode,
				first_valid_idx,
				&mut r4,
				&mut r3,
				&mut r2,
				&mut r1,
				&mut pp,
				&mut s1,
				&mut s2,
				&mut s3,
				&mut s4,
			),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => pivot_avx2(
				high,
				low,
				close,
				open,
				mode,
				first_valid_idx,
				&mut r4,
				&mut r3,
				&mut r2,
				&mut r1,
				&mut pp,
				&mut s1,
				&mut s2,
				&mut s3,
				&mut s4,
			),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => pivot_avx512(
				high,
				low,
				close,
				open,
				mode,
				first_valid_idx,
				&mut r4,
				&mut r3,
				&mut r2,
				&mut r1,
				&mut pp,
				&mut s1,
				&mut s2,
				&mut s3,
				&mut s4,
			),
			_ => unreachable!(),
		}
	}
	Ok(PivotOutput {
		r4,
		r3,
		r2,
		r1,
		pp,
		s1,
		s2,
		s3,
		s4,
	})
}

#[inline]
pub fn pivot_into_slices(
	r4: &mut [f64],
	r3: &mut [f64],
	r2: &mut [f64],
	r1: &mut [f64],
	pp: &mut [f64],
	s1: &mut [f64],
	s2: &mut [f64],
	s3: &mut [f64],
	s4: &mut [f64],
	input: &PivotInput,
	kern: Kernel,
) -> Result<(), PivotError> {
	let (high, low, close, open) = match &input.data {
		PivotData::Candles { candles } => {
			let high = source_type(candles, "high");
			let low = source_type(candles, "low");
			let close = source_type(candles, "close");
			let open = source_type(candles, "open");
			(high, low, close, open)
		}
		PivotData::Slices { high, low, close, open } => (*high, *low, *close, *open),
	};
	
	let len = high.len();
	if high.is_empty() || low.is_empty() || close.is_empty() {
		return Err(PivotError::EmptyData);
	}
	if low.len() != len || close.len() != len || open.len() != len {
		return Err(PivotError::EmptyData);
	}
	if r4.len() != len || r3.len() != len || r2.len() != len || r1.len() != len 
		|| pp.len() != len || s1.len() != len || s2.len() != len || s3.len() != len || s4.len() != len {
		return Err(PivotError::EmptyData);
	}
	
	let mode = input.get_mode();
	
	let mut first_valid_idx = None;
	for i in 0..len {
		let h = high[i];
		let l = low[i];
		let c = close[i];
		if !(h.is_nan() || l.is_nan() || c.is_nan()) {
			first_valid_idx = Some(i);
			break;
		}
	}
	let first_valid_idx = match first_valid_idx {
		Some(idx) => idx,
		None => return Err(PivotError::AllValuesNaN),
	};
	if first_valid_idx >= len {
		return Err(PivotError::NotEnoughValidData);
	}
	
	// Fill warmup with NaN
	for i in 0..first_valid_idx {
		r4[i] = f64::NAN;
		r3[i] = f64::NAN;
		r2[i] = f64::NAN;
		r1[i] = f64::NAN;
		pp[i] = f64::NAN;
		s1[i] = f64::NAN;
		s2[i] = f64::NAN;
		s3[i] = f64::NAN;
		s4[i] = f64::NAN;
	}
	
	let chosen = match kern {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};
	
	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => pivot_scalar(
				high, low, close, open, mode, first_valid_idx,
				r4, r3, r2, r1, pp, s1, s2, s3, s4,
			),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => pivot_avx2(
				high, low, close, open, mode, first_valid_idx,
				r4, r3, r2, r1, pp, s1, s2, s3, s4,
			),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => pivot_avx512(
				high, low, close, open, mode, first_valid_idx,
				r4, r3, r2, r1, pp, s1, s2, s3, s4,
			),
			_ => unreachable!(),
		}
	}
	
	Ok(())
}

#[inline]
pub unsafe fn pivot_scalar(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	open: &[f64],
	mode: usize,
	first: usize,
	r4: &mut [f64],
	r3: &mut [f64],
	r2: &mut [f64],
	r1: &mut [f64],
	pp: &mut [f64],
	s1: &mut [f64],
	s2: &mut [f64],
	s3: &mut [f64],
	s4: &mut [f64],
) {
	let len = high.len();
	for i in first..len {
		let h = high[i];
		let l = low[i];
		let c = close[i];
		let o = open[i];
		if h.is_nan() || l.is_nan() || c.is_nan() {
			continue;
		}
		let p = match mode {
			2 => {
				if c < o {
					(h + 2.0 * l + c) / 4.0
				} else if c > o {
					(2.0 * h + l + c) / 4.0
				} else {
					(h + l + 2.0 * c) / 4.0
				}
			}
			4 => (h + l + (2.0 * o)) / 4.0,
			_ => (h + l + c) / 3.0,
		};
		pp[i] = p;
		match mode {
			0 => {
				r1[i] = 2.0 * p - l;
				r2[i] = p + (h - l);
				s1[i] = 2.0 * p - h;
				s2[i] = p - (h - l);
			}
			1 => {
				r1[i] = p + 0.382 * (h - l);
				r2[i] = p + 0.618 * (h - l);
				r3[i] = p + 1.0 * (h - l);
				s1[i] = p - 0.382 * (h - l);
				s2[i] = p - 0.618 * (h - l);
				s3[i] = p - 1.0 * (h - l);
			}
			2 => {
				s1[i] = if c < o {
					(h + 2.0 * l + c) / 2.0 - h
				} else if c > o {
					(2.0 * h + l + c) / 2.0 - h
				} else {
					(h + l + 2.0 * c) / 2.0 - h
				};
				r1[i] = if c < o {
					(h + 2.0 * l + c) / 2.0 - l
				} else if c > o {
					(2.0 * h + l + c) / 2.0 - l
				} else {
					(h + l + 2.0 * c) / 2.0 - l
				};
			}
			3 => {
				r4[i] = (0.55 * (h - l)) + c;
				r3[i] = (0.275 * (h - l)) + c;
				r2[i] = (0.183 * (h - l)) + c;
				r1[i] = (0.0916 * (h - l)) + c;
				s1[i] = c - (0.0916 * (h - l));
				s2[i] = c - (0.183 * (h - l));
				s3[i] = c - (0.275 * (h - l));
				s4[i] = c - (0.55 * (h - l));
			}
			4 => {
				r3[i] = h + 2.0 * (p - l);
				r4[i] = r3[i] + (h - l);
				r2[i] = p + (h - l);
				r1[i] = 2.0 * p - l;
				s1[i] = 2.0 * p - h;
				s2[i] = p - (h - l);
				s3[i] = l - 2.0 * (h - p);
				s4[i] = s3[i] - (h - l);
			}
			_ => {}
		}
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn pivot_avx2(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	open: &[f64],
	mode: usize,
	first: usize,
	r4: &mut [f64],
	r3: &mut [f64],
	r2: &mut [f64],
	r1: &mut [f64],
	pp: &mut [f64],
	s1: &mut [f64],
	s2: &mut [f64],
	s3: &mut [f64],
	s4: &mut [f64],
) {
	// AVX2 stub fallback to scalar
	pivot_scalar(high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn pivot_avx512(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	open: &[f64],
	mode: usize,
	first: usize,
	r4: &mut [f64],
	r3: &mut [f64],
	r2: &mut [f64],
	r1: &mut [f64],
	pp: &mut [f64],
	s1: &mut [f64],
	s2: &mut [f64],
	s3: &mut [f64],
	s4: &mut [f64],
) {
	if high.len() <= 32 {
		pivot_avx512_short(high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4)
	} else {
		pivot_avx512_long(high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4)
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn pivot_avx512_short(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	open: &[f64],
	mode: usize,
	first: usize,
	r4: &mut [f64],
	r3: &mut [f64],
	r2: &mut [f64],
	r1: &mut [f64],
	pp: &mut [f64],
	s1: &mut [f64],
	s2: &mut [f64],
	s3: &mut [f64],
	s4: &mut [f64],
) {
	// AVX512 short stub fallback to scalar
	pivot_scalar(high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn pivot_avx512_long(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	open: &[f64],
	mode: usize,
	first: usize,
	r4: &mut [f64],
	r3: &mut [f64],
	r2: &mut [f64],
	r1: &mut [f64],
	pp: &mut [f64],
	s1: &mut [f64],
	s2: &mut [f64],
	s3: &mut [f64],
	s4: &mut [f64],
) {
	// AVX512 long stub fallback to scalar
	pivot_scalar(high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4)
}

// ========== ROW "BATCH" VECTORIZED API ==========

#[inline(always)]
pub unsafe fn pivot_row_scalar(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	open: &[f64],
	mode: usize,
	first: usize,
	r4: &mut [f64],
	r3: &mut [f64],
	r2: &mut [f64],
	r1: &mut [f64],
	pp: &mut [f64],
	s1: &mut [f64],
	s2: &mut [f64],
	s3: &mut [f64],
	s4: &mut [f64],
) {
	pivot_scalar(high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn pivot_row_avx2(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	open: &[f64],
	mode: usize,
	first: usize,
	r4: &mut [f64],
	r3: &mut [f64],
	r2: &mut [f64],
	r1: &mut [f64],
	pp: &mut [f64],
	s1: &mut [f64],
	s2: &mut [f64],
	s3: &mut [f64],
	s4: &mut [f64],
) {
	pivot_avx2(high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn pivot_row_avx512(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	open: &[f64],
	mode: usize,
	first: usize,
	r4: &mut [f64],
	r3: &mut [f64],
	r2: &mut [f64],
	r1: &mut [f64],
	pp: &mut [f64],
	s1: &mut [f64],
	s2: &mut [f64],
	s3: &mut [f64],
	s4: &mut [f64],
) {
	pivot_avx512(high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn pivot_row_avx512_short(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	open: &[f64],
	mode: usize,
	first: usize,
	r4: &mut [f64],
	r3: &mut [f64],
	r2: &mut [f64],
	r1: &mut [f64],
	pp: &mut [f64],
	s1: &mut [f64],
	s2: &mut [f64],
	s3: &mut [f64],
	s4: &mut [f64],
) {
	pivot_avx512_short(high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn pivot_row_avx512_long(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	open: &[f64],
	mode: usize,
	first: usize,
	r4: &mut [f64],
	r3: &mut [f64],
	r2: &mut [f64],
	r1: &mut [f64],
	pp: &mut [f64],
	s1: &mut [f64],
	s2: &mut [f64],
	s3: &mut [f64],
	s4: &mut [f64],
) {
	pivot_avx512_long(high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4)
}

// ========== BATCH (RANGE) API ==========

#[derive(Clone, Debug)]
pub struct PivotBatchRange {
	pub mode: (usize, usize, usize),
}
impl Default for PivotBatchRange {
	fn default() -> Self {
		Self { mode: (3, 3, 1) }
	}
}
#[derive(Clone, Debug, Default)]
pub struct PivotBatchBuilder {
	range: PivotBatchRange,
	kernel: Kernel,
}
impl PivotBatchBuilder {
	pub fn new() -> Self {
		Self::default()
	}
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline]
	pub fn mode_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.mode = (start, end, step);
		self
	}
	#[inline]
	pub fn mode_static(mut self, m: usize) -> Self {
		self.range.mode = (m, m, 1);
		self
	}
	pub fn apply_slice(
		self,
		high: &[f64],
		low: &[f64],
		close: &[f64],
		open: &[f64],
	) -> Result<PivotBatchOutput, PivotError> {
		pivot_batch_with_kernel(high, low, close, open, &self.range, self.kernel)
	}
	pub fn apply_candles(self, candles: &Candles) -> Result<PivotBatchOutput, PivotError> {
		let high = source_type(candles, "high");
		let low = source_type(candles, "low");
		let close = source_type(candles, "close");
		let open = source_type(candles, "open");
		self.apply_slice(high, low, close, open)
	}
	pub fn with_default_candles(candles: &Candles) -> Result<PivotBatchOutput, PivotError> {
		PivotBatchBuilder::new().kernel(Kernel::Auto).apply_candles(candles)
	}
}

pub fn pivot_batch_with_kernel(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	open: &[f64],
	sweep: &PivotBatchRange,
	k: Kernel,
) -> Result<PivotBatchOutput, PivotError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(PivotError::EmptyData),
	};
	pivot_batch_inner(high, low, close, open, sweep, kernel)
}

#[derive(Clone, Debug)]
pub struct PivotBatchOutput {
	pub levels: Vec<[Vec<f64>; 9]>,
	pub combos: Vec<PivotParams>,
	pub rows: usize,
	pub cols: usize,
}
fn expand_grid(r: &PivotBatchRange) -> Vec<PivotParams> {
	let (start, end, step) = r.mode;
	let mut v = Vec::new();
	let mut m = start;
	while m <= end {
		v.push(PivotParams { mode: Some(m) });
		m += step;
	}
	v
}
fn pivot_batch_inner(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	open: &[f64],
	sweep: &PivotBatchRange,
	kernel: Kernel,
) -> Result<PivotBatchOutput, PivotError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(PivotError::EmptyData);
	}
	let len = high.len();
	let mut levels = Vec::with_capacity(combos.len());
	for p in &combos {
		let mode = p.mode.unwrap_or(3);
		let mut first = None;
		for i in 0..len {
			if !(high[i].is_nan() || low[i].is_nan() || close[i].is_nan()) {
				first = Some(i);
				break;
			}
		}
		let first = first.unwrap_or(len);
		
		// Allocate output vectors with NaN prefix
		let mut r4 = alloc_with_nan_prefix(len, first);
		let mut r3 = alloc_with_nan_prefix(len, first);
		let mut r2 = alloc_with_nan_prefix(len, first);
		let mut r1 = alloc_with_nan_prefix(len, first);
		let mut pp = alloc_with_nan_prefix(len, first);
		let mut s1 = alloc_with_nan_prefix(len, first);
		let mut s2 = alloc_with_nan_prefix(len, first);
		let mut s3 = alloc_with_nan_prefix(len, first);
		let mut s4 = alloc_with_nan_prefix(len, first);
		unsafe {
			match kernel {
				Kernel::Scalar | Kernel::ScalarBatch => pivot_row_scalar(
					high, low, close, open, mode, first, &mut r4, &mut r3, &mut r2, &mut r1, &mut pp, &mut s1, &mut s2,
					&mut s3, &mut s4,
				),
				#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
				Kernel::Avx2 | Kernel::Avx2Batch => pivot_row_avx2(
					high, low, close, open, mode, first, &mut r4, &mut r3, &mut r2, &mut r1, &mut pp, &mut s1, &mut s2,
					&mut s3, &mut s4,
				),
				#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
				Kernel::Avx512 | Kernel::Avx512Batch => pivot_row_avx512(
					high, low, close, open, mode, first, &mut r4, &mut r3, &mut r2, &mut r1, &mut pp, &mut s1, &mut s2,
					&mut s3, &mut s4,
				),
				_ => unreachable!(),
			}
		}
		levels.push([r4, r3, r2, r1, pp, s1, s2, s3, s4]);
	}
	let rows = combos.len();
	let cols = high.len();
	Ok(PivotBatchOutput {
		levels,
		combos,
		rows,
		cols,
	})
}

// ========== STREAMING INTERFACE ==========

/// Streaming pivot calculation
/// Note: Pivot is not truly a streaming indicator as it requires complete period data.
/// This implementation maintains a single pivot level based on the most recent data point.
pub struct PivotStream {
	mode: usize,
}

impl PivotStream {
	pub fn new(mode: usize) -> Self {
		Self { mode }
	}

	pub fn try_new(params: PivotParams) -> Result<Self, PivotError> {
		let mode = params.mode.unwrap_or(3);
		if mode > 4 {
			return Err(PivotError::EmptyData); // Using existing error for invalid mode
		}
		Ok(Self { mode })
	}

	/// Update with new OHLC data and return pivot levels
	/// Returns tuple of (r4, r3, r2, r1, pp, s1, s2, s3, s4)
	pub fn update(&mut self, high: f64, low: f64, close: f64, open: f64) -> Option<(f64, f64, f64, f64, f64, f64, f64, f64, f64)> {
		if high.is_nan() || low.is_nan() || close.is_nan() || open.is_nan() {
			return None;
		}

		let p = match self.mode {
			2 => { // Demark
				if close < open {
					(high + 2.0 * low + close) / 4.0
				} else if close > open {
					(2.0 * high + low + close) / 4.0
				} else {
					(high + low + 2.0 * close) / 4.0
				}
			}
			4 => (high + low + 2.0 * open) / 4.0, // Woodie
			_ => (high + low + close) / 3.0, // Standard/Fibonacci/Camarilla
		};

		let (r4, r3, r2, r1, s1, s2, s3, s4) = match self.mode {
			0 => { // Standard
				let r1 = 2.0 * p - low;
				let r2 = p + (high - low);
				let s1 = 2.0 * p - high;
				let s2 = p - (high - low);
				(f64::NAN, f64::NAN, r2, r1, s1, s2, f64::NAN, f64::NAN)
			}
			1 => { // Fibonacci
				let r1 = p + 0.382 * (high - low);
				let r2 = p + 0.618 * (high - low);
				let r3 = p + 1.0 * (high - low);
				let s1 = p - 0.382 * (high - low);
				let s2 = p - 0.618 * (high - low);
				let s3 = p - 1.0 * (high - low);
				(f64::NAN, r3, r2, r1, s1, s2, s3, f64::NAN)
			}
			2 => { // Demark
				let r1 = if close < open {
					(high + 2.0 * low + close) / 2.0 - low
				} else if close > open {
					(2.0 * high + low + close) / 2.0 - low
				} else {
					(high + low + 2.0 * close) / 2.0 - low
				};
				let s1 = if close < open {
					(high + 2.0 * low + close) / 2.0 - high
				} else if close > open {
					(2.0 * high + low + close) / 2.0 - high
				} else {
					(high + low + 2.0 * close) / 2.0 - high
				};
				(f64::NAN, f64::NAN, f64::NAN, r1, s1, f64::NAN, f64::NAN, f64::NAN)
			}
			3 => { // Camarilla
				let r4 = (0.55 * (high - low)) + close;
				let r3 = (0.275 * (high - low)) + close;
				let r2 = (0.183 * (high - low)) + close;
				let r1 = (0.0916 * (high - low)) + close;
				let s1 = close - (0.0916 * (high - low));
				let s2 = close - (0.183 * (high - low));
				let s3 = close - (0.275 * (high - low));
				let s4 = close - (0.55 * (high - low));
				(r4, r3, r2, r1, s1, s2, s3, s4)
			}
			4 => { // Woodie
				let r3 = high + 2.0 * (p - low);
				let r4 = r3 + (high - low);
				let r2 = p + (high - low);
				let r1 = 2.0 * p - low;
				let s1 = 2.0 * p - high;
				let s2 = p - (high - low);
				let s3 = low - 2.0 * (high - p);
				let s4 = s3 - (high - low);
				(r4, r3, r2, r1, s1, s2, s3, s4)
			}
			_ => return None,
		};

		Some((r4, r3, r2, r1, p, s1, s2, s3, s4))
	}
}

// ========== PYTHON BINDINGS ==========

#[cfg(feature = "python")]
#[pyfunction(name = "pivot")]
#[pyo3(signature = (high, low, close, open, mode=3, kernel=None))]
pub fn pivot_py<'py>(
	py: Python<'py>,
	high: PyReadonlyArray1<'py, f64>,
	low: PyReadonlyArray1<'py, f64>,
	close: PyReadonlyArray1<'py, f64>,
	open: PyReadonlyArray1<'py, f64>,
	mode: usize,
	kernel: Option<&str>,
) -> PyResult<(
	Bound<'py, PyArray1<f64>>,
	Bound<'py, PyArray1<f64>>,
	Bound<'py, PyArray1<f64>>,
	Bound<'py, PyArray1<f64>>,
	Bound<'py, PyArray1<f64>>,
	Bound<'py, PyArray1<f64>>,
	Bound<'py, PyArray1<f64>>,
	Bound<'py, PyArray1<f64>>,
	Bound<'py, PyArray1<f64>>,
)> {
	use numpy::{IntoPyArray, PyArrayMethods};

	let high_slice = high.as_slice()?;
	let low_slice = low.as_slice()?;
	let close_slice = close.as_slice()?;
	let open_slice = open.as_slice()?;
	
	let kern = validate_kernel(kernel, false)?;
	
	let params = PivotParams { mode: Some(mode) };
	let input = PivotInput::from_slices(high_slice, low_slice, close_slice, open_slice, params);

	let result = py.allow_threads(|| pivot_with_kernel(&input, kern))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	Ok((
		result.r4.into_pyarray(py),
		result.r3.into_pyarray(py),
		result.r2.into_pyarray(py),
		result.r1.into_pyarray(py),
		result.pp.into_pyarray(py),
		result.s1.into_pyarray(py),
		result.s2.into_pyarray(py),
		result.s3.into_pyarray(py),
		result.s4.into_pyarray(py),
	))
}

#[cfg(feature = "python")]
#[pyclass(name = "PivotStream")]
pub struct PivotStreamPy {
	inner: PivotStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl PivotStreamPy {
	#[new]
	fn new(mode: Option<usize>) -> PyResult<Self> {
		let params = PivotParams { mode };
		let inner = PivotStream::try_new(params)
			.map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(PivotStreamPy { inner })
	}

	fn update(&mut self, high: f64, low: f64, close: f64, open: f64) -> Option<(f64, f64, f64, f64, f64, f64, f64, f64, f64)> {
		self.inner.update(high, low, close, open)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "pivot_batch")]
#[pyo3(signature = (high, low, close, open, mode_range, kernel=None))]
pub fn pivot_batch_py<'py>(
	py: Python<'py>,
	high: PyReadonlyArray1<'py, f64>,
	low: PyReadonlyArray1<'py, f64>,
	close: PyReadonlyArray1<'py, f64>,
	open: PyReadonlyArray1<'py, f64>,
	mode_range: (usize, usize, usize),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;

	let high_slice = high.as_slice()?;
	let low_slice = low.as_slice()?;
	let close_slice = close.as_slice()?;
	let open_slice = open.as_slice()?;
	
	let kern = validate_kernel(kernel, true)?;
	
	let sweep = PivotBatchRange { mode: mode_range };

	let output = py.allow_threads(|| {
		pivot_batch_with_kernel(high_slice, low_slice, close_slice, open_slice, &sweep, kern)
	})
	.map_err(|e| PyValueError::new_err(e.to_string()))?;

	let dict = PyDict::new(py);
	
	// For pivot, we need to handle 9 output arrays per parameter combination
	// We'll flatten them into a 2D array where each row contains all 9 levels
	let total_values = output.rows * output.cols * 9;
	let mut flat_values = Vec::with_capacity(total_values);
	
	for levels in &output.levels {
		// For each parameter combo, interleave the 9 arrays
		for i in 0..output.cols {
			flat_values.push(levels[0][i]); // r4
			flat_values.push(levels[1][i]); // r3
			flat_values.push(levels[2][i]); // r2
			flat_values.push(levels[3][i]); // r1
			flat_values.push(levels[4][i]); // pp
			flat_values.push(levels[5][i]); // s1
			flat_values.push(levels[6][i]); // s2
			flat_values.push(levels[7][i]); // s3
			flat_values.push(levels[8][i]); // s4
		}
	}
	
	// Create the values array and reshape it
	let values_arr = unsafe { PyArray1::<f64>::new(py, [total_values], false) };
	unsafe {
		let slice = values_arr.as_slice_mut()?;
		slice.copy_from_slice(&flat_values);
	}
	dict.set_item("values", values_arr.reshape((output.rows, output.cols * 9))?)?;
	
	// Add mode parameters
	dict.set_item(
		"modes",
		output.combos
			.iter()
			.map(|p| p.mode.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;
	
	// Add metadata
	dict.set_item("rows", output.rows)?;
	dict.set_item("cols", output.cols)?;
	dict.set_item("n_levels", 9)?;
	
	Ok(dict)
}

// ========== WASM BINDINGS ==========

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn pivot_js(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	open: &[f64],
	mode: usize,
) -> Result<Vec<f64>, JsValue> {
	let params = PivotParams { mode: Some(mode) };
	let input = PivotInput::from_slices(high, low, close, open, params);
	
	let len = high.len();
	
	// Single allocation for all 9 levels
	let mut output = vec![0.0; len * 9];
	
	// Create mutable slices for each level
	let (r4, rest) = output.split_at_mut(len);
	let (r3, rest) = rest.split_at_mut(len);
	let (r2, rest) = rest.split_at_mut(len);
	let (r1, rest) = rest.split_at_mut(len);
	let (pp, rest) = rest.split_at_mut(len);
	let (s1, rest) = rest.split_at_mut(len);
	let (s2, rest) = rest.split_at_mut(len);
	let (s3, s4) = rest.split_at_mut(len);
	
	// Compute into slices
	pivot_into_slices(
		r4, r3, r2, r1,
		pp, s1, s2, s3, s4,
		&input, Kernel::Auto
	).map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn pivot_into(
	high_ptr: *const f64,
	low_ptr: *const f64,
	close_ptr: *const f64,
	open_ptr: *const f64,
	r4_ptr: *mut f64,
	r3_ptr: *mut f64,
	r2_ptr: *mut f64,
	r1_ptr: *mut f64,
	pp_ptr: *mut f64,
	s1_ptr: *mut f64,
	s2_ptr: *mut f64,
	s3_ptr: *mut f64,
	s4_ptr: *mut f64,
	len: usize,
	mode: usize,
) -> Result<(), JsValue> {
	if high_ptr.is_null() || low_ptr.is_null() || close_ptr.is_null() || open_ptr.is_null() {
		return Err(JsValue::from_str("Null input pointer provided"));
	}
	
	if r4_ptr.is_null() || r3_ptr.is_null() || r2_ptr.is_null() || r1_ptr.is_null() 
		|| pp_ptr.is_null() || s1_ptr.is_null() || s2_ptr.is_null() || s3_ptr.is_null() || s4_ptr.is_null() {
		return Err(JsValue::from_str("Null output pointer provided"));
	}
	
	unsafe {
		let high = std::slice::from_raw_parts(high_ptr, len);
		let low = std::slice::from_raw_parts(low_ptr, len);
		let close = std::slice::from_raw_parts(close_ptr, len);
		let open = std::slice::from_raw_parts(open_ptr, len);
		
		let params = PivotParams { mode: Some(mode) };
		let input = PivotInput::from_slices(high, low, close, open, params);
		
		// Check for any aliasing between inputs and outputs
		let input_ptrs = [high_ptr as *const u8, low_ptr as *const u8, close_ptr as *const u8, open_ptr as *const u8];
		let output_ptrs = [
			r4_ptr as *const u8, r3_ptr as *const u8, r2_ptr as *const u8, r1_ptr as *const u8,
			pp_ptr as *const u8, s1_ptr as *const u8, s2_ptr as *const u8, s3_ptr as *const u8, s4_ptr as *const u8
		];
		
		let has_aliasing = input_ptrs.iter().any(|&inp| {
			output_ptrs.iter().any(|&out| inp == out)
		});
		
		if has_aliasing {
			// Use single temporary buffer if there's aliasing
			let mut temp = vec![0.0; len * 9];
			
			// Split into slices
			let (r4_temp, rest) = temp.split_at_mut(len);
			let (r3_temp, rest) = rest.split_at_mut(len);
			let (r2_temp, rest) = rest.split_at_mut(len);
			let (r1_temp, rest) = rest.split_at_mut(len);
			let (pp_temp, rest) = rest.split_at_mut(len);
			let (s1_temp, rest) = rest.split_at_mut(len);
			let (s2_temp, rest) = rest.split_at_mut(len);
			let (s3_temp, s4_temp) = rest.split_at_mut(len);
			
			pivot_into_slices(
				r4_temp, r3_temp, r2_temp, r1_temp,
				pp_temp, s1_temp, s2_temp, s3_temp, s4_temp,
				&input, Kernel::Auto
			).map_err(|e| JsValue::from_str(&e.to_string()))?;
			
			// Copy results to output pointers
			let r4_out = std::slice::from_raw_parts_mut(r4_ptr, len);
			let r3_out = std::slice::from_raw_parts_mut(r3_ptr, len);
			let r2_out = std::slice::from_raw_parts_mut(r2_ptr, len);
			let r1_out = std::slice::from_raw_parts_mut(r1_ptr, len);
			let pp_out = std::slice::from_raw_parts_mut(pp_ptr, len);
			let s1_out = std::slice::from_raw_parts_mut(s1_ptr, len);
			let s2_out = std::slice::from_raw_parts_mut(s2_ptr, len);
			let s3_out = std::slice::from_raw_parts_mut(s3_ptr, len);
			let s4_out = std::slice::from_raw_parts_mut(s4_ptr, len);
			
			r4_out.copy_from_slice(r4_temp);
			r3_out.copy_from_slice(r3_temp);
			r2_out.copy_from_slice(r2_temp);
			r1_out.copy_from_slice(r1_temp);
			pp_out.copy_from_slice(pp_temp);
			s1_out.copy_from_slice(s1_temp);
			s2_out.copy_from_slice(s2_temp);
			s3_out.copy_from_slice(s3_temp);
			s4_out.copy_from_slice(s4_temp);
		} else {
			// Direct computation into output slices
			let r4_out = std::slice::from_raw_parts_mut(r4_ptr, len);
			let r3_out = std::slice::from_raw_parts_mut(r3_ptr, len);
			let r2_out = std::slice::from_raw_parts_mut(r2_ptr, len);
			let r1_out = std::slice::from_raw_parts_mut(r1_ptr, len);
			let pp_out = std::slice::from_raw_parts_mut(pp_ptr, len);
			let s1_out = std::slice::from_raw_parts_mut(s1_ptr, len);
			let s2_out = std::slice::from_raw_parts_mut(s2_ptr, len);
			let s3_out = std::slice::from_raw_parts_mut(s3_ptr, len);
			let s4_out = std::slice::from_raw_parts_mut(s4_ptr, len);
			
			pivot_into_slices(
				r4_out, r3_out, r2_out, r1_out,
				pp_out, s1_out, s2_out, s3_out, s4_out,
				&input, Kernel::Auto
			).map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		
		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn pivot_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn pivot_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe {
			let _ = Vec::from_raw_parts(ptr, len, len);
		}
	}
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct PivotBatchConfig {
	pub mode_range: (usize, usize, usize), // (start, end, step)
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct PivotBatchJsOutput {
	pub values: Vec<f64>,  // Flattened array of all levels
	pub modes: Vec<usize>,
	pub rows: usize,       // Number of parameter combinations
	pub cols: usize,       // Data length
	pub n_levels: usize,   // Always 9 for pivot
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = pivot_batch)]
pub fn pivot_batch_js(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	open: &[f64],
	config: JsValue,
) -> Result<JsValue, JsValue> {
	let config: PivotBatchConfig = serde_wasm_bindgen::from_value(config)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	let sweep = PivotBatchRange {
		mode: config.mode_range,
	};
	
	let output = pivot_batch_inner(high, low, close, open, &sweep, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	// Flatten all 9 arrays per combination
	let mut flat_values = Vec::with_capacity(output.rows * output.cols * 9);
	for levels in &output.levels {
		// For each parameter combo, append all 9 levels
		for i in 0..output.cols {
			flat_values.push(levels[0][i]); // r4
			flat_values.push(levels[1][i]); // r3
			flat_values.push(levels[2][i]); // r2
			flat_values.push(levels[3][i]); // r1
			flat_values.push(levels[4][i]); // pp
			flat_values.push(levels[5][i]); // s1
			flat_values.push(levels[6][i]); // s2
			flat_values.push(levels[7][i]); // s3
			flat_values.push(levels[8][i]); // s4
		}
	}
	
	let modes: Vec<usize> = output.combos.iter()
		.map(|p| p.mode.unwrap())
		.collect();
	
	let result = PivotBatchJsOutput {
		values: flat_values,
		modes,
		rows: output.rows,
		cols: output.cols,
		n_levels: 9,
	};
	
	serde_wasm_bindgen::to_value(&result)
		.map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;
	use crate::utilities::enums::Kernel;
	use paste::paste;

	fn check_pivot_default_mode_camarilla(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let params = PivotParams { mode: None };
		let input = PivotInput::from_candles(&candles, params);
		let result = pivot_with_kernel(&input, kernel)?;

		assert_eq!(result.r4.len(), candles.close.len());
		assert_eq!(result.r3.len(), candles.close.len());
		assert_eq!(result.r2.len(), candles.close.len());
		assert_eq!(result.r1.len(), candles.close.len());
		assert_eq!(result.pp.len(), candles.close.len());
		assert_eq!(result.s1.len(), candles.close.len());
		assert_eq!(result.s2.len(), candles.close.len());
		assert_eq!(result.s3.len(), candles.close.len());
		assert_eq!(result.s4.len(), candles.close.len());

		// Spot-check Camarilla outputs for a few points
		let last_five_r4 = &result.r4[result.r4.len().saturating_sub(5)..];
		let expected_r4 = [59466.5, 59357.55, 59243.6, 59334.85, 59170.35];
		for (i, &val) in last_five_r4.iter().enumerate() {
			let exp = expected_r4[i];
			assert!(
				(val - exp).abs() < 1e-1,
				"Camarilla r4 mismatch at index {}, expected {}, got {}",
				i,
				exp,
				val
			);
		}
		Ok(())
	}

	fn check_pivot_nan_values(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [10.0, f64::NAN, 30.0];
		let low = [9.0, 8.5, f64::NAN];
		let close = [9.5, 9.0, 29.0];
		let open = [9.1, 8.8, 28.5];

		let params = PivotParams { mode: Some(3) };
		let input = PivotInput::from_slices(&high, &low, &close, &open, params);
		let result = pivot_with_kernel(&input, kernel)?;
		assert_eq!(result.pp.len(), high.len());
		Ok(())
	}

	fn check_pivot_no_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high: [f64; 0] = [];
		let low: [f64; 0] = [];
		let close: [f64; 0] = [];
		let open: [f64; 0] = [];
		let params = PivotParams { mode: Some(3) };
		let input = PivotInput::from_slices(&high, &low, &close, &open, params);
		let result = pivot_with_kernel(&input, kernel);
		assert!(result.is_err());
		if let Err(e) = result {
			assert!(
				e.to_string().contains("One or more required fields"),
				"Expected 'EmptyData' error, got: {}",
				e
			);
		}
		Ok(())
	}

	fn check_pivot_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [f64::NAN, f64::NAN];
		let low = [f64::NAN, f64::NAN];
		let close = [f64::NAN, f64::NAN];
		let open = [f64::NAN, f64::NAN];
		let params = PivotParams { mode: Some(3) };
		let input = PivotInput::from_slices(&high, &low, &close, &open, params);
		let result = pivot_with_kernel(&input, kernel);
		assert!(result.is_err());
		if let Err(e) = result {
			assert!(
				e.to_string().contains("All values are NaN"),
				"Expected 'AllValuesNaN' error, got: {}",
				e
			);
		}
		Ok(())
	}

	fn check_pivot_fibonacci_mode(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file)?;
		let params = PivotParams { mode: Some(1) };
		let input = PivotInput::from_candles(&candles, params);
		let output = pivot_with_kernel(&input, kernel)?;
		assert_eq!(output.r3.len(), candles.close.len());
		Ok(())
	}

	fn check_pivot_standard_mode(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file)?;
		let params = PivotParams { mode: Some(0) };
		let input = PivotInput::from_candles(&candles, params);
		let output = pivot_with_kernel(&input, kernel)?;
		assert_eq!(output.r2.len(), candles.close.len());
		Ok(())
	}

	fn check_pivot_demark_mode(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file)?;
		let params = PivotParams { mode: Some(2) };
		let input = PivotInput::from_candles(&candles, params);
		let output = pivot_with_kernel(&input, kernel)?;
		assert_eq!(output.r1.len(), candles.close.len());
		Ok(())
	}

	fn check_pivot_woodie_mode(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file)?;
		let params = PivotParams { mode: Some(4) };
		let input = PivotInput::from_candles(&candles, params);
		let output = pivot_with_kernel(&input, kernel)?;
		assert_eq!(output.r4.len(), candles.close.len());
		Ok(())
	}

	fn check_pivot_batch_default_row(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file)?;
		let output = PivotBatchBuilder::new().kernel(kernel).apply_candles(&candles)?;
		let default = PivotParams::default();
		let def_idx = output
			.combos
			.iter()
			.position(|p| p.mode == default.mode)
			.expect("default row missing");
		for arr in &output.levels[def_idx] {
			assert_eq!(arr.len(), candles.close.len());
		}
		Ok(())
	}

	// Macro for all kernel variants
	macro_rules! generate_all_pivot_tests {
        ($($test_fn:ident),*) => {
            paste! {
                $(
                    #[test]
                    fn [<$test_fn _scalar>]() { let _ = $test_fn(stringify!([<$test_fn _scalar>]), Kernel::Scalar); }
                )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(
                    #[test]
                    fn [<$test_fn _avx2>]() { let _ = $test_fn(stringify!([<$test_fn _avx2>]), Kernel::Avx2); }
                    #[test]
                    fn [<$test_fn _avx512>]() { let _ = $test_fn(stringify!([<$test_fn _avx512>]), Kernel::Avx512); }
                )*
                $(
                    #[test]
                    fn [<$test_fn _auto_detect>]() { let _ = $test_fn(stringify!([<$test_fn _auto_detect>]), Kernel::Auto); }
                )*
            }
        }
    }

	generate_all_pivot_tests!(
		check_pivot_default_mode_camarilla,
		check_pivot_nan_values,
		check_pivot_no_data,
		check_pivot_all_nan,
		check_pivot_fibonacci_mode,
		check_pivot_standard_mode,
		check_pivot_demark_mode,
		check_pivot_woodie_mode,
		check_pivot_batch_default_row
	);
	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file)?;

		let output = PivotBatchBuilder::new().kernel(kernel).apply_candles(&candles)?;

		let def = PivotParams::default();
		let row = output
			.combos
			.iter()
			.position(|p| p.mode == def.mode)
			.expect("default row missing");
		let levels = &output.levels[row];

		// Spot check: each level should be the right length
		for arr in levels.iter() {
			assert_eq!(arr.len(), candles.close.len());
		}

		// Optionally, spot-check some values (e.g. Camarilla r4)
		let expected_r4 = [59466.5, 59357.55, 59243.6, 59334.85, 59170.35];
		let r4 = &levels[0];
		let last_five_r4 = &r4[r4.len().saturating_sub(5)..];
		for (i, &val) in last_five_r4.iter().enumerate() {
			let exp = expected_r4[i];
			assert!(
				(val - exp).abs() < 1e-1,
				"[{test}] Camarilla r4 mismatch at idx {i}: {val} vs {exp:?}"
			);
		}
		Ok(())
	}

	// Kernel variant macro expansion (as in alma.rs)
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
}
