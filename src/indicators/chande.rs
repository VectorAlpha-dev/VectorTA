//! # Chande Exits (Chandelier Exits)
//!
//! Volatility-based trailing exit using ATR and rolling max/min, with builder, batch, and AVX/parallel support.
//! API/feature/test coverage parity with alma.rs.
//!
//! ## Parameters
//! - **period**: Window size for both ATR and rolling max/min (default: 22).
//! - **mult**: ATR multiplier (default: 3.0).
//! - **direction**: "long" or "short" (default: "long").
//!
//! ## Errors
//! - **AllValuesNaN**: chande: All input values are NaN.
//! - **InvalidPeriod**: chande: period is zero or exceeds length.
//! - **NotEnoughValidData**: chande: Not enough valid data for period.
//! - **InvalidDirection**: chande: direction must be "long" or "short".
//!
//! ## Returns
//! - `Ok(ChandeOutput)` on success, `Err(ChandeError)` on error.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
	alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes, make_uninit_matrix,
};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::mem::ManuallyDrop;
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
use wasm_bindgen::prelude::*;
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub enum ChandeData<'a> {
	Candles { candles: &'a Candles },
}

#[derive(Debug, Clone)]
pub struct ChandeOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct ChandeParams {
	pub period: Option<usize>,
	pub mult: Option<f64>,
	pub direction: Option<String>,
}

impl Default for ChandeParams {
	fn default() -> Self {
		Self {
			period: Some(22),
			mult: Some(3.0),
			direction: Some("long".into()),
		}
	}
}

#[derive(Debug, Clone)]
pub struct ChandeInput<'a> {
	pub data: ChandeData<'a>,
	pub params: ChandeParams,
}

impl<'a> ChandeInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, p: ChandeParams) -> Self {
		Self {
			data: ChandeData::Candles { candles: c },
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, ChandeParams::default())
	}
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(22)
	}
	#[inline]
	pub fn get_mult(&self) -> f64 {
		self.params.mult.unwrap_or(3.0)
	}
	#[inline]
	pub fn get_direction(&self) -> &str {
		self.params.direction.as_deref().unwrap_or("long")
	}
}

#[derive(Clone, Debug)]
pub struct ChandeBuilder {
	period: Option<usize>,
	mult: Option<f64>,
	direction: Option<String>,
	kernel: Kernel,
}

impl Default for ChandeBuilder {
	fn default() -> Self {
		Self {
			period: None,
			mult: None,
			direction: None,
			kernel: Kernel::Auto,
		}
	}
}
impl ChandeBuilder {
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
	pub fn mult(mut self, m: f64) -> Self {
		self.mult = Some(m);
		self
	}
	#[inline(always)]
	pub fn direction<S: Into<String>>(mut self, d: S) -> Self {
		self.direction = Some(d.into());
		self
	}
	#[inline(always)]
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}

	#[inline(always)]
	pub fn apply(self, c: &Candles) -> Result<ChandeOutput, ChandeError> {
		let p = ChandeParams {
			period: self.period,
			mult: self.mult,
			direction: self.direction,
		};
		let i = ChandeInput::from_candles(c, p);
		chande_with_kernel(&i, self.kernel)
	}

	#[inline(always)]
	pub fn into_stream(self) -> Result<ChandeStream, ChandeError> {
		let p = ChandeParams {
			period: self.period,
			mult: self.mult,
			direction: self.direction,
		};
		ChandeStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum ChandeError {
	#[error("chande: All values are NaN.")]
	AllValuesNaN,
	#[error("chande: Invalid period: period = {period}, data_len = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("chande: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("chande: Invalid direction: {direction}")]
	InvalidDirection { direction: String },
}

#[inline]
pub fn chande(input: &ChandeInput) -> Result<ChandeOutput, ChandeError> {
	chande_with_kernel(input, Kernel::Auto)
}

pub fn chande_with_kernel(input: &ChandeInput, kernel: Kernel) -> Result<ChandeOutput, ChandeError> {
	let ChandeData::Candles { candles } = &input.data;
	let high = source_type(candles, "high");
	let low = source_type(candles, "low");
	let close = source_type(candles, "close");
	let len = high.len();

	let first = close
		.iter()
		.position(|&x| !x.is_nan())
		.ok_or(ChandeError::AllValuesNaN)?;
	let period = input.get_period();
	let mult = input.get_mult();
	let dir = input.get_direction().to_lowercase();
	if dir != "long" && dir != "short" {
		return Err(ChandeError::InvalidDirection { direction: dir });
	}
	if period == 0 || period > len {
		return Err(ChandeError::InvalidPeriod { period, data_len: len });
	}
	if len - first < period {
		return Err(ChandeError::NotEnoughValidData {
			needed: period,
			valid: len - first,
		});
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	let warmup_period = first + period - 1;
	let mut out = alloc_with_nan_prefix(len, warmup_period);
	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => {
				chande_scalar(high, low, close, period, mult, &dir, first, &mut out)
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => chande_avx2(high, low, close, period, mult, &dir, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => chande_avx512(high, low, close, period, mult, &dir, first, &mut out),
			_ => unreachable!(),
		}
	}
	Ok(ChandeOutput { values: out })
}

/// Helper function to compute chande directly into a pre-allocated slice
#[inline]
pub fn chande_compute_into(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	period: usize,
	mult: f64,
	direction: &str,
	kernel: Kernel,
	out: &mut [f64],
) -> Result<(), ChandeError> {
	// Validate inputs
	let len = high.len();
	if len != low.len() || len != close.len() {
		return Err(ChandeError::AllValuesNaN);
	}
	
	let first = close
		.iter()
		.position(|&x| !x.is_nan())
		.ok_or(ChandeError::AllValuesNaN)?;
	
	let dir = direction.to_lowercase();
	if dir != "long" && dir != "short" {
		return Err(ChandeError::InvalidDirection { direction: dir });
	}
	
	if period == 0 || period > len {
		return Err(ChandeError::InvalidPeriod { period, data_len: len });
	}
	
	if len - first < period {
		return Err(ChandeError::NotEnoughValidData {
			needed: period,
			valid: len - first,
		});
	}
	
	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};
	
	// Fill warmup period with NaN
	let warmup_period = first + period - 1;
	for i in 0..warmup_period {
		out[i] = f64::NAN;
	}
	
	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => {
				chande_scalar(high, low, close, period, mult, &dir, first, out)
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => chande_avx2(high, low, close, period, mult, &dir, first, out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => chande_avx512(high, low, close, period, mult, &dir, first, out),
			_ => unreachable!(),
		}
	}
	
	Ok(())
}

/// Helper function for WASM to compute chande directly into a pre-allocated slice
/// This follows the pattern from alma_into_slice for zero-copy operations
#[inline]
pub fn chande_into_slice(
	dst: &mut [f64],
	input: &ChandeInput,
	kern: Kernel,
) -> Result<(), ChandeError> {
	let (high, low, close) = match &input.data {
		ChandeData::Candles { candles } => (&candles.high, &candles.low, &candles.close),
	};
	
	let params = &input.params;
	let period = params.period.unwrap_or(22);
	let mult = params.mult.unwrap_or(3.0);
	let direction = params.direction.as_deref().unwrap_or("long");
	
	// Validate inputs
	if high.is_empty() || low.is_empty() || close.is_empty() {
		return Err(ChandeError::AllValuesNaN);
	}
	
	if high.len() != low.len() || high.len() != close.len() {
		return Err(ChandeError::NotEnoughValidData {
			needed: high.len().max(low.len()).max(close.len()),
			valid: high.len().min(low.len()).min(close.len()),
		});
	}
	
	if dst.len() != high.len() {
		return Err(ChandeError::InvalidPeriod {
			period,
			data_len: high.len(),
		});
	}
	
	// Use the existing chande_compute_into function
	chande_compute_into(high, low, close, period, mult, direction, kern, dst)?;
	
	Ok(())
}

#[inline]
pub fn chande_scalar(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	period: usize,
	mult: f64,
	dir: &str,
	first: usize,
	out: &mut [f64],
) {
	let len = high.len();
	let alpha = 1.0 / period as f64;
	let mut sum_tr = 0.0;
	let mut rma = f64::NAN;
	for i in first..len {
		let tr = if i == first {
			high[i] - low[i]
		} else {
			let hl = high[i] - low[i];
			let hc = (high[i] - close[i - 1]).abs();
			let lc = (low[i] - close[i - 1]).abs();
			hl.max(hc).max(lc)
		};
		if i < first + period {
			sum_tr += tr;
			if i == first + period - 1 {
				rma = sum_tr / period as f64;
			}
		} else {
			rma += alpha * (tr - rma);
		}
		if i >= first + period - 1 && !rma.is_nan() {
			let start = i + 1 - period;
			if dir == "long" {
				let mut m = f64::MIN;
				for j in start..=i {
					if high[j] > m {
						m = high[j];
					}
				}
				out[i] = m - rma * mult;
			} else {
				let mut m = f64::MAX;
				for j in start..=i {
					if low[j] < m {
						m = low[j];
					}
				}
				out[i] = m + rma * mult;
			}
		}
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn chande_avx2(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	period: usize,
	mult: f64,
	dir: &str,
	first: usize,
	out: &mut [f64],
) {
	chande_scalar(high, low, close, period, mult, dir, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn chande_avx512(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	period: usize,
	mult: f64,
	dir: &str,
	first: usize,
	out: &mut [f64],
) {
	if period <= 32 {
		unsafe { chande_avx512_short(high, low, close, period, mult, dir, first, out) }
	} else {
		unsafe { chande_avx512_long(high, low, close, period, mult, dir, first, out) }
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn chande_avx512_short(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	period: usize,
	mult: f64,
	dir: &str,
	first: usize,
	out: &mut [f64],
) {
	chande_scalar(high, low, close, period, mult, dir, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn chande_avx512_long(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	period: usize,
	mult: f64,
	dir: &str,
	first: usize,
	out: &mut [f64],
) {
	chande_scalar(high, low, close, period, mult, dir, first, out)
}

#[derive(Debug, Clone)]
pub struct ChandeStream {
	period: usize,
	mult: f64,
	direction: String,
	high_buf: Vec<f64>,
	low_buf: Vec<f64>,
	close_prev: f64,
	atr: f64,
	buffer_filled: usize,
	filled: bool,
	buffer_idx: usize,  // Ring buffer index
}

impl ChandeStream {
	pub fn try_new(params: ChandeParams) -> Result<Self, ChandeError> {
		let period = params.period.unwrap_or(22);
		let mult = params.mult.unwrap_or(3.0);
		let direction = params.direction.unwrap_or_else(|| "long".into());
		if period == 0 {
			return Err(ChandeError::InvalidPeriod { period, data_len: 0 });
		}
		if direction != "long" && direction != "short" {
			return Err(ChandeError::InvalidDirection { direction });
		}
		let mut high_buf = vec![0.0; period];
		let mut low_buf = vec![0.0; period];
		
		Ok(Self {
			period,
			mult,
			direction,
			high_buf,
			low_buf,
			close_prev: f64::NAN,
			atr: 0.0,
			buffer_filled: 0,
			filled: false,
			buffer_idx: 0,
		})
	}

	pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
		// Calculate TR
		let tr = if self.buffer_filled == 0 {
			high - low
		} else {
			let hl = high - low;
			let hc = (high - self.close_prev).abs();
			let lc = (low - self.close_prev).abs();
			hl.max(hc).max(lc)
		};
		
		// Update ATR
		if self.buffer_filled < self.period {
			// Warmup period
			self.atr += tr;
			self.buffer_filled += 1;
			
			// Store in buffer during warmup
			self.high_buf[self.buffer_filled - 1] = high;
			self.low_buf[self.buffer_filled - 1] = low;
			
			if self.buffer_filled == self.period {
				self.atr /= self.period as f64;
				self.filled = true;
			}
		} else {
			// Normal operation - use RMA
			let alpha = 1.0 / self.period as f64;
			self.atr += alpha * (tr - self.atr);
			
			// Store in ring buffer
			self.high_buf[self.buffer_idx] = high;
			self.low_buf[self.buffer_idx] = low;
			self.buffer_idx = (self.buffer_idx + 1) % self.period;
		}
		
		self.close_prev = close;
		
		if self.filled {
			// Find max/min over the buffer
			if self.direction == "long" {
				let m = self.high_buf[..self.buffer_filled.min(self.period)]
					.iter()
					.cloned()
					.fold(f64::MIN, f64::max);
				Some(m - self.atr * self.mult)
			} else {
				let m = self.low_buf[..self.buffer_filled.min(self.period)]
					.iter()
					.cloned()
					.fold(f64::MAX, f64::min);
				Some(m + self.atr * self.mult)
			}
		} else {
			None
		}
	}
}

#[derive(Clone, Debug)]
pub struct ChandeBatchRange {
	pub period: (usize, usize, usize),
	pub mult: (f64, f64, f64),
}

impl Default for ChandeBatchRange {
	fn default() -> Self {
		Self {
			period: (22, 22, 0),
			mult: (3.0, 3.0, 0.0),
		}
	}
}

#[derive(Clone, Debug, Default)]
pub struct ChandeBatchBuilder {
	range: ChandeBatchRange,
	direction: String,
	kernel: Kernel,
}

impl ChandeBatchBuilder {
	pub fn new() -> Self {
		Self {
			range: ChandeBatchRange::default(),
			direction: "long".into(),
			kernel: Kernel::Auto,
		}
	}
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	pub fn direction<S: Into<String>>(mut self, d: S) -> Self {
		self.direction = d.into();
		self
	}

	pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.period = (start, end, step);
		self
	}
	pub fn period_static(mut self, p: usize) -> Self {
		self.range.period = (p, p, 0);
		self
	}
	pub fn mult_range(mut self, start: f64, end: f64, step: f64) -> Self {
		self.range.mult = (start, end, step);
		self
	}
	pub fn mult_static(mut self, m: f64) -> Self {
		self.range.mult = (m, m, 0.0);
		self
	}

	pub fn apply_candles(self, c: &Candles) -> Result<ChandeBatchOutput, ChandeError> {
		let high = source_type(c, "high");
		let low = source_type(c, "low");
		let close = source_type(c, "close");
		chande_batch_with_kernel(high, low, close, &self.range, &self.direction, self.kernel)
	}
}

pub fn chande_batch_with_kernel(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	sweep: &ChandeBatchRange,
	direction: &str,
	k: Kernel,
) -> Result<ChandeBatchOutput, ChandeError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(ChandeError::InvalidPeriod { period: 0, data_len: 0 }),
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	chande_batch_par_slice(high, low, close, sweep, direction, simd)
}

#[derive(Clone, Debug)]
pub struct ChandeBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<ChandeParams>,
	pub rows: usize,
	pub cols: usize,
}
impl ChandeBatchOutput {
	pub fn row_for_params(&self, p: &ChandeParams) -> Option<usize> {
		self.combos.iter().position(|c| {
			c.period.unwrap_or(22) == p.period.unwrap_or(22)
				&& (c.mult.unwrap_or(3.0) - p.mult.unwrap_or(3.0)).abs() < 1e-12
				&& c.direction.as_deref().unwrap_or("long") == p.direction.as_deref().unwrap_or("long")
		})
	}
	pub fn values_for(&self, p: &ChandeParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &ChandeBatchRange, dir: &str) -> Vec<ChandeParams> {
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
	let periods = axis_usize(r.period);
	let mults = axis_f64(r.mult);
	let mut out = Vec::with_capacity(periods.len() * mults.len());
	for &p in &periods {
		for &m in &mults {
			out.push(ChandeParams {
				period: Some(p),
				mult: Some(m),
				direction: Some(dir.to_string()),
			});
		}
	}
	out
}

#[inline(always)]
pub fn chande_batch_slice(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	sweep: &ChandeBatchRange,
	dir: &str,
	kern: Kernel,
) -> Result<ChandeBatchOutput, ChandeError> {
	chande_batch_inner(high, low, close, sweep, dir, kern, false)
}

#[inline(always)]
pub fn chande_batch_par_slice(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	sweep: &ChandeBatchRange,
	dir: &str,
	kern: Kernel,
) -> Result<ChandeBatchOutput, ChandeError> {
	chande_batch_inner(high, low, close, sweep, dir, kern, true)
}

#[inline(always)]
fn chande_batch_inner(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	sweep: &ChandeBatchRange,
	dir: &str,
	kern: Kernel,
	parallel: bool,
) -> Result<ChandeBatchOutput, ChandeError> {
	let combos = expand_grid(sweep, dir);
	if combos.is_empty() {
		return Err(ChandeError::InvalidPeriod { period: 0, data_len: 0 });
	}
	let first = close
		.iter()
		.position(|&x| !x.is_nan())
		.ok_or(ChandeError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if high.len() - first < max_p {
		return Err(ChandeError::NotEnoughValidData {
			needed: max_p,
			valid: high.len() - first,
		});
	}
	let rows = combos.len();
	let cols = high.len();

	// Calculate warmup periods for each row
	let warmup_periods: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap() - 1).collect();

	// Allocate uninitialized matrix and set NaN prefixes
	let mut buf_mu = make_uninit_matrix(rows, cols);
	init_matrix_prefixes(&mut buf_mu, cols, &warmup_periods);

	// Convert to mutable slice for computation
	let mut buf_guard = ManuallyDrop::new(buf_mu);
	let values_slice: &mut [f64] =
		unsafe { core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len()) };

	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let period = combos[row].period.unwrap();
		let mult = combos[row].mult.unwrap();
		let direction = combos[row].direction.as_deref().unwrap();
		match kern {
			Kernel::Scalar => chande_row_scalar(high, low, close, first, period, mult, direction, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => chande_row_avx2(high, low, close, first, period, mult, direction, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => chande_row_avx512(high, low, close, first, period, mult, direction, out_row),
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

	// Reclaim as Vec<f64>
	let values = unsafe {
		Vec::from_raw_parts(
			buf_guard.as_mut_ptr() as *mut f64,
			buf_guard.len(),
			buf_guard.capacity(),
		)
	};

	Ok(ChandeBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

/// Computes batch chande directly into pre-allocated output slice
#[inline(always)]
fn chande_batch_inner_into(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	sweep: &ChandeBatchRange,
	dir: &str,
	kern: Kernel,
	parallel: bool,
	out: &mut [f64],
) -> Result<Vec<ChandeParams>, ChandeError> {
	let combos = expand_grid(sweep, dir);
	if combos.is_empty() {
		return Err(ChandeError::InvalidPeriod { period: 0, data_len: 0 });
	}
	
	let first = close
		.iter()
		.position(|&x| !x.is_nan())
		.ok_or(ChandeError::AllValuesNaN)?;
	
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if high.len() - first < max_p {
		return Err(ChandeError::NotEnoughValidData {
			needed: max_p,
			valid: high.len() - first,
		});
	}
	
	let cols = high.len();
	
	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let period = combos[row].period.unwrap();
		let mult = combos[row].mult.unwrap();
		let direction = combos[row].direction.as_deref().unwrap();
		match kern {
			Kernel::Scalar => chande_row_scalar(high, low, close, first, period, mult, direction, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => chande_row_avx2(high, low, close, first, period, mult, direction, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => chande_row_avx512(high, low, close, first, period, mult, direction, out_row),
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
unsafe fn chande_row_scalar(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	first: usize,
	period: usize,
	mult: f64,
	dir: &str,
	out: &mut [f64],
) {
	chande_scalar(high, low, close, period, mult, dir, first, out);
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn chande_row_avx2(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	first: usize,
	period: usize,
	mult: f64,
	dir: &str,
	out: &mut [f64],
) {
	chande_row_scalar(high, low, close, first, period, mult, dir, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn chande_row_avx512(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	first: usize,
	period: usize,
	mult: f64,
	dir: &str,
	out: &mut [f64],
) {
	if period <= 32 {
		chande_row_avx512_short(high, low, close, first, period, mult, dir, out)
	} else {
		chande_row_avx512_long(high, low, close, first, period, mult, dir, out)
	}
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn chande_row_avx512_short(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	first: usize,
	period: usize,
	mult: f64,
	dir: &str,
	out: &mut [f64],
) {
	chande_row_scalar(high, low, close, first, period, mult, dir, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn chande_row_avx512_long(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	first: usize,
	period: usize,
	mult: f64,
	dir: &str,
	out: &mut [f64],
) {
	chande_row_scalar(high, low, close, first, period, mult, dir, out)
}
#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_chande_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let default_params = ChandeParams {
			period: None,
			mult: None,
			direction: None,
		};
		let input = ChandeInput::from_candles(&candles, default_params);
		let output = chande_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());

		Ok(())
	}

	fn check_chande_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let close_prices = &candles.close;

		let params = ChandeParams {
			period: Some(22),
			mult: Some(3.0),
			direction: Some("long".into()),
		};
		let input = ChandeInput::from_candles(&candles, params);
		let chande_result = chande_with_kernel(&input, kernel)?;

		assert_eq!(chande_result.values.len(), close_prices.len());

		let expected_last_five = [
			59444.14115983658,
			58576.49837984401,
			58649.1120898511,
			58724.56154031242,
			58713.39965211639,
		];

		assert!(chande_result.values.len() >= 5);
		let start_idx = chande_result.values.len() - 5;
		let actual_last_five = &chande_result.values[start_idx..];
		for (i, &val) in actual_last_five.iter().enumerate() {
			let exp = expected_last_five[i];
			assert!(
				(val - exp).abs() < 1e-4,
				"[{}] Chande Exits mismatch at index {}: expected {}, got {}",
				test_name,
				i,
				exp,
				val
			);
		}

		let period = 22;
		for i in 0..(period - 1) {
			assert!(chande_result.values[i].is_nan(), "Expected leading NaN at index {}", i);
		}

		let default_input = ChandeInput::with_default_candles(&candles);
		let default_output = chande_with_kernel(&default_input, kernel)?;
		assert_eq!(default_output.values.len(), close_prices.len());
		Ok(())
	}

	fn check_chande_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let params = ChandeParams {
			period: Some(0),
			mult: Some(3.0),
			direction: Some("long".into()),
		};
		let input = ChandeInput::from_candles(&candles, params);

		let res = chande_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] Chande should fail with zero period", test_name);
		Ok(())
	}

	fn check_chande_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let params = ChandeParams {
			period: Some(99999),
			mult: Some(3.0),
			direction: Some("long".into()),
		};
		let input = ChandeInput::from_candles(&candles, params);

		let res = chande_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] Chande should fail with period exceeding length",
			test_name
		);
		Ok(())
	}

	fn check_chande_bad_direction(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let params = ChandeParams {
			period: Some(22),
			mult: Some(3.0),
			direction: Some("bad".into()),
		};
		let input = ChandeInput::from_candles(&candles, params);

		let res = chande_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] Chande should fail with bad direction", test_name);
		Ok(())
	}

	fn check_chande_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let params = ChandeParams {
			period: Some(22),
			mult: Some(3.0),
			direction: Some("long".into()),
		};
		let input = ChandeInput::from_candles(&candles, params);
		let result = chande_with_kernel(&input, kernel)?;

		if result.values.len() > 240 {
			for i in 240..result.values.len() {
				assert!(
					!result.values[i].is_nan(),
					"[{}] Unexpected NaN at index {}",
					test_name,
					i
				);
			}
		}
		Ok(())
	}

	fn check_chande_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let params = ChandeParams {
			period: Some(22),
			mult: Some(3.0),
			direction: Some("long".into()),
		};
		let input = ChandeInput::from_candles(&candles, params.clone());
		let batch_output = chande_with_kernel(&input, kernel)?.values;

		let mut stream = ChandeStream::try_new(params)?;
		let mut stream_values = Vec::with_capacity(candles.close.len());
		for ((&h, &l), &c) in candles.high.iter().zip(&candles.low).zip(&candles.close) {
			match stream.update(h, l, c) {
				Some(chande_val) => stream_values.push(chande_val),
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
				diff < 1e-8,
				"[{}] Chande streaming mismatch at idx {}: batch={}, stream={}, diff={}",
				test_name,
				i,
				b,
				s,
				diff
			);
		}
		Ok(())
	}

	// Check for poison values in single output - only runs in debug mode
	#[cfg(debug_assertions)]
	fn check_chande_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		// Test with multiple parameter combinations to increase chance of catching bugs
		let param_combinations = vec![
			ChandeParams {
				period: Some(10),
				mult: Some(2.0),
				direction: Some("long".into()),
			},
			ChandeParams {
				period: Some(22),
				mult: Some(3.0),
				direction: Some("short".into()),
			},
			ChandeParams {
				period: Some(50),
				mult: Some(5.0),
				direction: Some("long".into()),
			},
		];

		for params in param_combinations {
			let input = ChandeInput::from_candles(&candles, params.clone());
			let output = chande_with_kernel(&input, kernel)?;

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
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} with params: period={}, mult={}, direction={}",
                        test_name, val, bits, i,
                        params.period.unwrap(), params.mult.unwrap(), params.direction.as_ref().unwrap()
                    );
				}

				// Check for init_matrix_prefixes poison (0x22222222_22222222)
				if bits == 0x22222222_22222222 {
					panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} with params: period={}, mult={}, direction={}",
                        test_name, val, bits, i,
                        params.period.unwrap(), params.mult.unwrap(), params.direction.as_ref().unwrap()
                    );
				}

				// Check for make_uninit_matrix poison (0x33333333_33333333)
				if bits == 0x33333333_33333333 {
					panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} with params: period={}, mult={}, direction={}",
                        test_name, val, bits, i,
                        params.period.unwrap(), params.mult.unwrap(), params.direction.as_ref().unwrap()
                    );
				}
			}
		}

		Ok(())
	}

	// Release mode stub - does nothing
	#[cfg(not(debug_assertions))]
	fn check_chande_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		Ok(())
	}

	macro_rules! generate_all_chande_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $( #[test] fn [<$test_fn _scalar_f64>]() {
                    let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar);
                })*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $( #[test] fn [<$test_fn _avx2_f64>]() {
                    let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2);
                })*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $( #[test] fn [<$test_fn _avx512_f64>]() {
                    let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512);
                })*
            }
        }
    }

	generate_all_chande_tests!(
		check_chande_partial_params,
		check_chande_accuracy,
		check_chande_zero_period,
		check_chande_period_exceeds_length,
		check_chande_bad_direction,
		check_chande_nan_handling,
		check_chande_streaming,
		check_chande_no_poison
	);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let output = ChandeBatchBuilder::new().kernel(kernel).apply_candles(&c)?;

		let def = ChandeParams::default();
		let row = output.values_for(&def).expect("default row missing");
		assert_eq!(row.len(), c.close.len());

		let expected = [
			59444.14115983658,
			58576.49837984401,
			58649.1120898511,
			58724.56154031242,
			58713.39965211639,
		];
		let start = row.len() - 5;
		for (i, &v) in row[start..].iter().enumerate() {
			assert!(
				(v - expected[i]).abs() < 1e-4,
				"[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
			);
		}
		Ok(())
	}

	// Check for poison values in batch output - only runs in debug mode
	#[cfg(debug_assertions)]
	fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		// Test batch with multiple parameter combinations
		let output = ChandeBatchBuilder::new()
			.kernel(kernel)
			.period_range(10, 30, 10) // Tests periods 10, 20, 30
			.mult_range(2.0, 5.0, 1.5) // Tests multipliers 2.0, 3.5, 5.0
			.direction("long")
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
			let params = &output.combos[row];

			// Check for alloc_with_nan_prefix poison (0x11111111_11111111)
			if bits == 0x11111111_11111111 {
				panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with params: period={}, mult={}, direction={}",
                    test, val, bits, row, col, idx,
                    params.period.unwrap(), params.mult.unwrap(), params.direction.as_ref().unwrap()
                );
			}

			// Check for init_matrix_prefixes poison (0x22222222_22222222)
			if bits == 0x22222222_22222222 {
				panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {}) with params: period={}, mult={}, direction={}",
                    test, val, bits, row, col, idx,
                    params.period.unwrap(), params.mult.unwrap(), params.direction.as_ref().unwrap()
                );
			}

			// Check for make_uninit_matrix poison (0x33333333_33333333)
			if bits == 0x33333333_33333333 {
				panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with params: period={}, mult={}, direction={}",
                    test, val, bits, row, col, idx,
                    params.period.unwrap(), params.mult.unwrap(), params.direction.as_ref().unwrap()
                );
			}
		}

		// Also test with "short" direction
		let output_short = ChandeBatchBuilder::new()
			.kernel(kernel)
			.period_range(15, 45, 15) // Tests periods 15, 30, 45
			.mult_range(1.0, 4.0, 1.5) // Tests multipliers 1.0, 2.5, 4.0
			.direction("short")
			.apply_candles(&c)?;

		for (idx, &val) in output_short.values.iter().enumerate() {
			if val.is_nan() {
				continue;
			}

			let bits = val.to_bits();
			let row = idx / output_short.cols;
			let col = idx % output_short.cols;
			let params = &output_short.combos[row];

			if bits == 0x11111111_11111111 {
				panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with params: period={}, mult={}, direction={}",
                    test, val, bits, row, col, idx,
                    params.period.unwrap(), params.mult.unwrap(), params.direction.as_ref().unwrap()
                );
			}

			if bits == 0x22222222_22222222 {
				panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {}) with params: period={}, mult={}, direction={}",
                    test, val, bits, row, col, idx,
                    params.period.unwrap(), params.mult.unwrap(), params.direction.as_ref().unwrap()
                );
			}

			if bits == 0x33333333_33333333 {
				panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with params: period={}, mult={}, direction={}",
                    test, val, bits, row, col, idx,
                    params.period.unwrap(), params.mult.unwrap(), params.direction.as_ref().unwrap()
                );
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

// ============================
// Python Bindings
// ============================

#[cfg(feature = "python")]
#[inline]
fn validate_kernel(kernel: Option<&str>, is_batch: bool) -> PyResult<Kernel> {
	let kernel_str = kernel.unwrap_or("auto");
	match kernel_str.to_lowercase().as_str() {
		"auto" => Ok(Kernel::Auto),
		"scalar" => Ok(if is_batch { Kernel::ScalarBatch } else { Kernel::Scalar }),
		"avx2" => Ok(if is_batch { Kernel::Avx2Batch } else { Kernel::Avx2 }),
		"avx512" => Ok(if is_batch { Kernel::Avx512Batch } else { Kernel::Avx512 }),
		_ => Err(PyValueError::new_err(format!("Invalid kernel: {}", kernel_str))),
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "chande")]
#[pyo3(signature = (high, low, close, period, mult, direction, kernel=None))]
pub fn chande_py<'py>(
	py: Python<'py>,
	high: PyReadonlyArray1<'py, f64>,
	low: PyReadonlyArray1<'py, f64>,
	close: PyReadonlyArray1<'py, f64>,
	period: usize,
	mult: f64,
	direction: &str,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
	let high_slice = high.as_slice()?;
	let low_slice = low.as_slice()?;
	let close_slice = close.as_slice()?;
	
	// Validate inputs have same length
	if high_slice.len() != low_slice.len() || high_slice.len() != close_slice.len() {
		return Err(PyValueError::new_err("Input arrays must have the same length"));
	}
	
	let kern = validate_kernel(kernel, false)?;
	
	// Build input and compute result
	let candles = Candles {
		high: high_slice.to_vec(),
		low: low_slice.to_vec(),
		close: close_slice.to_vec(),
		timestamp: vec![],
		open: vec![],
		volume: vec![],
		hl2: vec![],
		hlc3: vec![],
		ohlc4: vec![],
		hlcc4: vec![],
	};
	
	let params = ChandeParams {
		period: Some(period),
		mult: Some(mult),
		direction: Some(direction.to_string()),
	};
	
	let input = ChandeInput::from_candles(&candles, params);
	
	let result_vec: Vec<f64> = py
		.allow_threads(|| chande_with_kernel(&input, kern).map(|o| o.values))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;
	
	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "ChandeStream")]
pub struct ChandeStreamPy {
	stream: ChandeStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl ChandeStreamPy {
	#[new]
	fn new(period: usize, mult: f64, direction: &str) -> PyResult<Self> {
		let params = ChandeParams {
			period: Some(period),
			mult: Some(mult),
			direction: Some(direction.to_string()),
		};
		let stream = ChandeStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(ChandeStreamPy { stream })
	}
	
	fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
		self.stream.update(high, low, close)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "chande_batch")]
#[pyo3(signature = (high, low, close, period_range, mult_range, direction="long", kernel=None))]
pub fn chande_batch_py<'py>(
	py: Python<'py>,
	high: PyReadonlyArray1<'py, f64>,
	low: PyReadonlyArray1<'py, f64>,
	close: PyReadonlyArray1<'py, f64>,
	period_range: (usize, usize, usize),
	mult_range: (f64, f64, f64),
	direction: &str,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
	let high_slice = high.as_slice()?;
	let low_slice = low.as_slice()?;
	let close_slice = close.as_slice()?;
	
	// Validate inputs have same length
	if high_slice.len() != low_slice.len() || high_slice.len() != close_slice.len() {
		return Err(PyValueError::new_err("Input arrays must have the same length"));
	}
	
	let kern = validate_kernel(kernel, true)?;
	
	let sweep = ChandeBatchRange {
		period: period_range,
		mult: mult_range,
	};
	
	let combos = expand_grid(&sweep, direction);
	let rows = combos.len();
	let cols = high_slice.len();
	
	// Pre-allocate output array for batch operations
	let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let slice_out = unsafe { out_arr.as_slice_mut()? };
	
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
			
			// Compute directly into pre-allocated buffer without intermediate allocation
			chande_batch_inner_into(high_slice, low_slice, close_slice, &sweep, direction, simd, true, slice_out)
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
	dict.set_item(
		"mults",
		combos
			.iter()
			.map(|p| p.mult.unwrap())
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;
	dict.set_item(
		"directions",
		combos
			.iter()
			.map(|p| p.direction.as_ref().unwrap().as_str())
			.collect::<Vec<_>>(),
	)?;
	
	Ok(dict)
}

// ============================
// WASM Bindings
// ============================

/// Serializable batch result for WASM
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct ChandeBatchResult {
	pub values: Vec<f64>,
	pub periods: Vec<usize>,
	pub mults: Vec<f64>,
	pub directions: Vec<String>,
	pub rows: usize,
	pub cols: usize,
}

/// Safe API: Single calculation with automatic memory management
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn chande_js(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	period: usize,
	mult: f64,
	direction: &str,
) -> Result<Vec<f64>, JsValue> {
	let params = ChandeParams {
		period: Some(period),
		mult: Some(mult),
		direction: Some(direction.to_string()),
	};
	
	let candles = Candles {
		high: high.to_vec(),
		low: low.to_vec(),
		close: close.to_vec(),
		timestamp: vec![],
		open: vec![],
		volume: vec![],
		hl2: vec![],
		hlc3: vec![],
		ohlc4: vec![],
		hlcc4: vec![],
	};
	
	let input = ChandeInput::from_candles(&candles, params);
	let mut output = vec![0.0; high.len()];
	
	chande_into_slice(&mut output, &input, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	Ok(output)
}

/// Safe API: Batch processing with JavaScript-friendly output
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn chande_batch_js(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	period_start: usize,
	period_end: usize,
	period_step: usize,
	mult_start: f64,
	mult_end: f64,
	mult_step: f64,
	direction: &str,
) -> Result<JsValue, JsValue> {
	let period_range = (period_start, period_end, period_step);
	let mult_range = (mult_start, mult_end, mult_step);
	
	let candles = Candles {
		high: high.to_vec(),
		low: low.to_vec(),
		close: close.to_vec(),
		timestamp: vec![],
		open: vec![],
		volume: vec![],
		hl2: vec![],
		hlc3: vec![],
		ohlc4: vec![],
		hlcc4: vec![],
	};
	
	let sweep = ChandeBatchRange {
		period: period_range,
		mult: mult_range,
	};
	
	// Generate parameter combinations
	let combos = expand_grid(&sweep, direction);
	let rows = combos.len();
	let cols = high.len();
	
	// Allocate output matrix
	let mut out_flat = vec![0.0; rows * cols];
	
	// Compute batch
	let _ = chande_batch_inner_into(high, low, close, &sweep, direction, Kernel::Auto, false, &mut out_flat)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	let result = ChandeBatchResult {
		values: out_flat,
		periods: combos.iter().map(|p| p.period.unwrap()).collect(),
		mults: combos.iter().map(|p| p.mult.unwrap()).collect(),
		directions: combos.iter().map(|p| p.direction.as_ref().unwrap().clone()).collect(),
		rows,
		cols,
	};
	
	serde_wasm_bindgen::to_value(&result)
		.map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Memory allocation for WASM
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn chande_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

/// Memory deallocation for WASM
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn chande_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe {
			let _ = Vec::from_raw_parts(ptr, len, len);
		}
	}
}

/// Fast/Zero-copy API: Compute directly into pre-allocated buffer
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn chande_into(
	high_ptr: *const f64,
	low_ptr: *const f64,
	close_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period: usize,
	mult: f64,
	direction: &str,
) -> Result<(), JsValue> {
	if high_ptr.is_null() || low_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}
	
	unsafe {
		let high = std::slice::from_raw_parts(high_ptr, len);
		let low = std::slice::from_raw_parts(low_ptr, len);
		let close = std::slice::from_raw_parts(close_ptr, len);
		
		// Check for aliasing with any of the input pointers
		let needs_temp = high_ptr == out_ptr || low_ptr == out_ptr || close_ptr == out_ptr;
		
		if needs_temp {
			// Use temporary buffer to avoid aliasing issues
			let mut temp = vec![0.0; len];
			chande_compute_into(high, low, close, period, mult, direction, Kernel::Auto, &mut temp)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			out.copy_from_slice(&temp);
		} else {
			// Direct computation when no aliasing
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			chande_compute_into(high, low, close, period, mult, direction, Kernel::Auto, out)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		
		Ok(())
	}
}

/// Fast/Zero-copy API: Batch computation into pre-allocated buffer
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn chande_batch_into(
	high_ptr: *const f64,
	low_ptr: *const f64,
	close_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period_start: usize,
	period_end: usize,
	period_step: usize,
	mult_start: f64,
	mult_end: f64,
	mult_step: f64,
	direction: &str,
) -> Result<usize, JsValue> {
	if high_ptr.is_null() || low_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}
	
	unsafe {
		let high = std::slice::from_raw_parts(high_ptr, len);
		let low = std::slice::from_raw_parts(low_ptr, len);
		let close = std::slice::from_raw_parts(close_ptr, len);
		
		let sweep = ChandeBatchRange {
			period: (period_start, period_end, period_step),
			mult: (mult_start, mult_end, mult_step),
		};
		
		let param_combinations = expand_grid(&sweep, direction);
		let rows = param_combinations.len();
		let cols = len;
		
		// Check for aliasing
		let needs_temp = high_ptr == out_ptr || low_ptr == out_ptr || close_ptr == out_ptr;
		
		if needs_temp {
			// Use temporary buffer
			let mut temp = vec![0.0; rows * cols];
			chande_batch_inner_into(high, low, close, &sweep, direction, Kernel::Auto, false, &mut temp)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);
			out.copy_from_slice(&temp);
		} else {
			// Direct computation
			let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);
			chande_batch_inner_into(high, low, close, &sweep, direction, Kernel::Auto, false, out)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		
		Ok(rows)
	}
}
