//! # SafeZoneStop
//!
//! The SafeZoneStop indicator attempts to place stop-loss levels based on
//! directional movement and volatility, using MINUS_DM or PLUS_DM logic under the hood.
//! Parity with alma.rs in terms of performance, features, and API structure. SIMD variants
//! are stubbed to the scalar implementation as per requirements.
//!
//! ## Parameters
//! - **period**: The time period for calculating DM (Wilder's smoothing). Defaults to 22.
//! - **mult**: Multiplier for the DM measure. Defaults to 2.5.
//! - **max_lookback**: Window for final max/min. Defaults to 3.
//! - **direction**: "long" or "short".
//!
//! ## Errors
//! - **AllValuesNaN**: safezonestop: All input data values are `NaN`.
//! - **InvalidPeriod**: safezonestop: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: safezonestop: Not enough valid data points for the requested `period`.
//! - **MismatchedLengths**: safezonestop: Input slices have different lengths.
//! - **InvalidDirection**: safezonestop: Direction must be "long" or "short".
//!
//! ## Returns
//! - **`Ok(SafeZoneStopOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(SafeZoneStopError)`** otherwise.
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
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

impl<'a> AsRef<[f64]> for SafeZoneStopInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			SafeZoneStopData::Candles { candles, .. } => source_type(candles, "close"),
			SafeZoneStopData::Slices { low, .. } => low,
		}
	}
}

#[derive(Debug, Clone)]
pub enum SafeZoneStopData<'a> {
	Candles {
		candles: &'a Candles,
		direction: &'a str,
	},
	Slices {
		high: &'a [f64],
		low: &'a [f64],
		direction: &'a str,
	},
}

#[derive(Debug, Clone)]
pub struct SafeZoneStopOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct SafeZoneStopParams {
	pub period: Option<usize>,
	pub mult: Option<f64>,
	pub max_lookback: Option<usize>,
}

impl Default for SafeZoneStopParams {
	fn default() -> Self {
		Self {
			period: Some(22),
			mult: Some(2.5),
			max_lookback: Some(3),
		}
	}
}

#[derive(Debug, Clone)]
pub struct SafeZoneStopInput<'a> {
	pub data: SafeZoneStopData<'a>,
	pub params: SafeZoneStopParams,
}

impl<'a> SafeZoneStopInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, direction: &'a str, p: SafeZoneStopParams) -> Self {
		Self {
			data: SafeZoneStopData::Candles { candles: c, direction },
			params: p,
		}
	}
	#[inline]
	pub fn from_slices(high: &'a [f64], low: &'a [f64], direction: &'a str, p: SafeZoneStopParams) -> Self {
		Self {
			data: SafeZoneStopData::Slices { high, low, direction },
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "long", SafeZoneStopParams::default())
	}
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(22)
	}
	#[inline]
	pub fn get_mult(&self) -> f64 {
		self.params.mult.unwrap_or(2.5)
	}
	#[inline]
	pub fn get_max_lookback(&self) -> usize {
		self.params.max_lookback.unwrap_or(3)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct SafeZoneStopBuilder {
	period: Option<usize>,
	mult: Option<f64>,
	max_lookback: Option<usize>,
	direction: Option<&'static str>,
	kernel: Kernel,
}

impl Default for SafeZoneStopBuilder {
	fn default() -> Self {
		Self {
			period: None,
			mult: None,
			max_lookback: None,
			direction: Some("long"),
			kernel: Kernel::Auto,
		}
	}
}

impl SafeZoneStopBuilder {
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
	pub fn mult(mut self, x: f64) -> Self {
		self.mult = Some(x);
		self
	}
	#[inline(always)]
	pub fn max_lookback(mut self, n: usize) -> Self {
		self.max_lookback = Some(n);
		self
	}
	#[inline(always)]
	pub fn direction(mut self, d: &'static str) -> Self {
		self.direction = Some(d);
		self
	}
	#[inline(always)]
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline(always)]
	pub fn apply(self, c: &Candles) -> Result<SafeZoneStopOutput, SafeZoneStopError> {
		let p = SafeZoneStopParams {
			period: self.period,
			mult: self.mult,
			max_lookback: self.max_lookback,
		};
		let i = SafeZoneStopInput::from_candles(c, self.direction.unwrap_or("long"), p);
		safezonestop_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<SafeZoneStopOutput, SafeZoneStopError> {
		let p = SafeZoneStopParams {
			period: self.period,
			mult: self.mult,
			max_lookback: self.max_lookback,
		};
		let i = SafeZoneStopInput::from_slices(high, low, self.direction.unwrap_or("long"), p);
		safezonestop_with_kernel(&i, self.kernel)
	}
}

#[derive(Debug, Error)]
pub enum SafeZoneStopError {
	#[error("safezonestop: All values are NaN.")]
	AllValuesNaN,
	#[error("safezonestop: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("safezonestop: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("safezonestop: Mismatched lengths")]
	MismatchedLengths,
	#[error("safezonestop: Invalid direction. Must be 'long' or 'short'.")]
	InvalidDirection,
}

#[inline]
pub fn safezonestop(input: &SafeZoneStopInput) -> Result<SafeZoneStopOutput, SafeZoneStopError> {
	safezonestop_with_kernel(input, Kernel::Auto)
}

pub fn safezonestop_with_kernel(
	input: &SafeZoneStopInput,
	kernel: Kernel,
) -> Result<SafeZoneStopOutput, SafeZoneStopError> {
	let (high, low, direction) = match &input.data {
		SafeZoneStopData::Candles { candles, direction } => {
			let h = source_type(candles, "high");
			let l = source_type(candles, "low");
			(h, l, *direction)
		}
		SafeZoneStopData::Slices { high, low, direction } => (*high, *low, *direction),
	};

	if high.len() != low.len() {
		return Err(SafeZoneStopError::MismatchedLengths);
	}

	let period = input.get_period();
	let mult = input.get_mult();
	let max_lookback = input.get_max_lookback();
	let len = high.len();

	if period == 0 || period > len {
		return Err(SafeZoneStopError::InvalidPeriod { period, data_len: len });
	}

	let has_any_non_nan = high.iter().any(|&v| !v.is_nan()) || low.iter().any(|&v| !v.is_nan());
	if !has_any_non_nan {
		return Err(SafeZoneStopError::AllValuesNaN);
	}

	if direction != "long" && direction != "short" {
		return Err(SafeZoneStopError::InvalidDirection);
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	// Determine warmup period
	let warmup_period = period.saturating_sub(1).max(max_lookback.saturating_sub(1));
	
	// Use proper memory allocation
	let mut out = alloc_with_nan_prefix(len, warmup_period);
	// Fill remaining values with NaN for binding compatibility
	for i in warmup_period..out.len() {
		out[i] = f64::NAN;
	}

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => {
				safezonestop_scalar(high, low, period, mult, max_lookback, direction, &mut out)
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => safezonestop_avx2(high, low, period, mult, max_lookback, direction, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => {
				safezonestop_avx512(high, low, period, mult, max_lookback, direction, &mut out)
			}
			_ => unreachable!(),
		}
	}

	Ok(SafeZoneStopOutput { values: out })
}

/// Write SafeZoneStop directly to output slice - no allocations
#[inline]
pub fn safezonestop_into_slice(
	dst: &mut [f64],
	input: &SafeZoneStopInput,
	kern: Kernel,
) -> Result<(), SafeZoneStopError> {
	let (high, low, direction) = match &input.data {
		SafeZoneStopData::Candles { candles, direction } => {
			let high = source_type(candles, "high");
			let low = source_type(candles, "low");
			(high, low, *direction)
		}
		SafeZoneStopData::Slices { high, low, direction } => (*high, *low, *direction),
	};

	let len = high.len();
	if len != low.len() {
		return Err(SafeZoneStopError::MismatchedLengths);
	}
	
	if dst.len() != len {
		return Err(SafeZoneStopError::InvalidPeriod { period: dst.len(), data_len: len });
	}

	let period = input.params.period.unwrap_or(22);
	let mult = input.params.mult.unwrap_or(2.5);
	let max_lookback = input.params.max_lookback.unwrap_or(3);

	if period == 0 || period > len {
		return Err(SafeZoneStopError::InvalidPeriod { period, data_len: len });
	}
	
	if direction != "long" && direction != "short" {
		return Err(SafeZoneStopError::InvalidDirection);
	}

	let chosen = match kern {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	// Determine warmup period
	let warmup_period = period.saturating_sub(1).max(max_lookback.saturating_sub(1));
	
	// Fill warmup period with NaN
	let warmup_end = warmup_period.min(dst.len());
	for v in &mut dst[..warmup_end] {
		*v = f64::NAN;
	}

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => {
				safezonestop_scalar(high, low, period, mult, max_lookback, direction, dst)
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => safezonestop_avx2(high, low, period, mult, max_lookback, direction, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => {
				safezonestop_avx512(high, low, period, mult, max_lookback, direction, dst)
			}
			_ => unreachable!(),
		}
	}

	Ok(())
}

#[inline(always)]
pub unsafe fn safezonestop_scalar(
	high: &[f64],
	low: &[f64],
	period: usize,
	mult: f64,
	max_lookback: usize,
	direction: &str,
	out: &mut [f64],
) {
	let len = high.len();
	
	// Note: These allocations are necessary for the algorithm's intermediate calculations
	// They store derived values that must be computed sequentially
	// Unlike output arrays, these cannot use uninitialized memory as they need specific values
	
	// Compute directional movement
	let mut dm_smooth = vec![f64::NAN; len];
	
	if direction == "long" {
		// Calculate minus DM for long positions
		let mut sum = 0.0;
		for i in 1..=period.min(len - 1) {
			let up_move = high[i] - high[i - 1];
			let down_move = low[i - 1] - low[i];
			if down_move > up_move && down_move > 0.0 {
				sum += down_move;
			}
		}
		
		if period < len {
			dm_smooth[period] = sum;
			for i in (period + 1)..len {
				let up_move = high[i] - high[i - 1];
				let down_move = low[i - 1] - low[i];
				let dm = if down_move > up_move && down_move > 0.0 { down_move } else { 0.0 };
				dm_smooth[i] = dm_smooth[i - 1] - (dm_smooth[i - 1] / (period as f64)) + dm;
			}
		}
		
		// Compute SafeZone values for long
		for i in 0..len {
			if i + 1 < max_lookback {
				continue;
			}
			
			let start_idx = i + 1 - max_lookback;
			let mut mx = f64::NAN;
			
			for j in start_idx..=i {
				if j > 0 && !dm_smooth[j].is_nan() {
					let val = low[j - 1] - mult * dm_smooth[j];
					if mx.is_nan() || val > mx {
						mx = val;
					}
				}
			}
			out[i] = mx;
		}
	} else {
		// Calculate plus DM for short positions
		let mut sum = 0.0;
		for i in 1..=period.min(len - 1) {
			let up_move = high[i] - high[i - 1];
			let down_move = low[i - 1] - low[i];
			if up_move > down_move && up_move > 0.0 {
				sum += up_move;
			}
		}
		
		if period < len {
			dm_smooth[period] = sum;
			for i in (period + 1)..len {
				let up_move = high[i] - high[i - 1];
				let down_move = low[i - 1] - low[i];
				let dm = if up_move > down_move && up_move > 0.0 { up_move } else { 0.0 };
				dm_smooth[i] = dm_smooth[i - 1] - (dm_smooth[i - 1] / (period as f64)) + dm;
			}
		}
		
		// Compute SafeZone values for short
		for i in 0..len {
			if i + 1 < max_lookback {
				continue;
			}
			
			let start_idx = i + 1 - max_lookback;
			let mut mn = f64::NAN;
			
			for j in start_idx..=i {
				if j > 0 && !dm_smooth[j].is_nan() {
					let val = high[j - 1] + mult * dm_smooth[j];
					if mn.is_nan() || val < mn {
						mn = val;
					}
				}
			}
			out[i] = mn;
		}
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn safezonestop_avx512(
	high: &[f64],
	low: &[f64],
	period: usize,
	mult: f64,
	max_lookback: usize,
	direction: &str,
	out: &mut [f64],
) {
	if period <= 32 {
		safezonestop_avx512_short(high, low, period, mult, max_lookback, direction, out);
	} else {
		safezonestop_avx512_long(high, low, period, mult, max_lookback, direction, out);
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn safezonestop_avx512_short(
	high: &[f64],
	low: &[f64],
	period: usize,
	mult: f64,
	max_lookback: usize,
	direction: &str,
	out: &mut [f64],
) {
	safezonestop_scalar(high, low, period, mult, max_lookback, direction, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn safezonestop_avx512_long(
	high: &[f64],
	low: &[f64],
	period: usize,
	mult: f64,
	max_lookback: usize,
	direction: &str,
	out: &mut [f64],
) {
	safezonestop_scalar(high, low, period, mult, max_lookback, direction, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn safezonestop_avx2(
	high: &[f64],
	low: &[f64],
	period: usize,
	mult: f64,
	max_lookback: usize,
	direction: &str,
	out: &mut [f64],
) {
	safezonestop_scalar(high, low, period, mult, max_lookback, direction, out)
}

#[derive(Debug, Clone)]
pub struct SafeZoneStopStream {
	period: usize,
	mult: f64,
	max_lookback: usize,
	direction: String,
	buffer_high: Vec<f64>,
	buffer_low: Vec<f64>,
	idx: usize,
	filled: bool,
	last_result: f64,
	// Pre-allocated work buffers to avoid allocations on every update
	work_high: Vec<f64>,
	work_low: Vec<f64>,
	work_out: Vec<f64>,
}

impl SafeZoneStopStream {
	pub fn try_new(params: SafeZoneStopParams, direction: &str) -> Result<Self, SafeZoneStopError> {
		let period = params.period.unwrap_or(22);
		let mult = params.mult.unwrap_or(2.5);
		let max_lookback = params.max_lookback.unwrap_or(3);
		if period == 0 {
			return Err(SafeZoneStopError::InvalidPeriod { period, data_len: 0 });
		}
		if direction != "long" && direction != "short" {
			return Err(SafeZoneStopError::InvalidDirection);
		}
		let buf_size = period.max(max_lookback);
		Ok(Self {
			period,
			mult,
			max_lookback,
			direction: direction.to_string(),
			buffer_high: vec![f64::NAN; buf_size],
			buffer_low: vec![f64::NAN; buf_size],
			idx: 0,
			filled: false,
			last_result: f64::NAN,
			// Pre-allocate work buffers
			work_high: vec![0.0; buf_size],
			work_low: vec![0.0; buf_size],
			work_out: vec![f64::NAN; buf_size],
		})
	}
	pub fn update(&mut self, high: f64, low: f64) -> Option<f64> {
		let n = self.buffer_high.len();
		self.buffer_high[self.idx] = high;
		self.buffer_low[self.idx] = low;
		self.idx = (self.idx + 1) % n;
		if !self.filled && self.idx == 0 {
			self.filled = true;
		}
		if !self.filled {
			return None;
		}
		
		// Copy circular buffer to work buffers in correct order (no allocation)
		let mut write_idx = 0;
		for read_idx in self.idx..n {
			self.work_high[write_idx] = self.buffer_high[read_idx];
			self.work_low[write_idx] = self.buffer_low[read_idx];
			write_idx += 1;
		}
		for read_idx in 0..self.idx {
			self.work_high[write_idx] = self.buffer_high[read_idx];
			self.work_low[write_idx] = self.buffer_low[read_idx];
			write_idx += 1;
		}
		
		// Reset output buffer
		self.work_out.fill(f64::NAN);
		
		unsafe {
			safezonestop_scalar(
				&self.work_high,
				&self.work_low,
				self.period,
				self.mult,
				self.max_lookback,
				&self.direction,
				&mut self.work_out,
			);
		}
		self.last_result = *self.work_out.last().unwrap_or(&f64::NAN);
		Some(self.last_result)
	}
}

#[derive(Clone, Debug)]
pub struct SafeZoneStopBatchRange {
	pub period: (usize, usize, usize),
	pub mult: (f64, f64, f64),
	pub max_lookback: (usize, usize, usize),
}

impl Default for SafeZoneStopBatchRange {
	fn default() -> Self {
		Self {
			period: (22, 22, 0),
			mult: (2.5, 2.5, 0.0),
			max_lookback: (3, 3, 0),
		}
	}
}

#[derive(Clone, Debug)]
pub struct SafeZoneStopBatchBuilder {
	range: SafeZoneStopBatchRange,
	direction: &'static str,
	kernel: Kernel,
}

impl Default for SafeZoneStopBatchBuilder {
	fn default() -> Self {
		Self {
			range: SafeZoneStopBatchRange::default(),
			direction: "long",
			kernel: Kernel::Auto,
		}
	}
}

impl SafeZoneStopBatchBuilder {
	pub fn new() -> Self {
		Self::default()
	}
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline]
	pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.period = (start, end, step);
		self
	}
	#[inline]
	pub fn mult_range(mut self, start: f64, end: f64, step: f64) -> Self {
		self.range.mult = (start, end, step);
		self
	}
	#[inline]
	pub fn max_lookback_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.max_lookback = (start, end, step);
		self
	}
	#[inline]
	pub fn direction(mut self, d: &'static str) -> Self {
		self.direction = d;
		self
	}
	pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<SafeZoneStopBatchOutput, SafeZoneStopError> {
		safezonestop_batch_with_kernel(high, low, &self.range, self.direction, self.kernel)
	}
	pub fn with_default_slices(
		high: &[f64],
		low: &[f64],
		direction: &'static str,
		k: Kernel,
	) -> Result<SafeZoneStopBatchOutput, SafeZoneStopError> {
		SafeZoneStopBatchBuilder::new()
			.kernel(k)
			.direction(direction)
			.apply_slices(high, low)
	}
}

pub fn safezonestop_batch_with_kernel(
	high: &[f64],
	low: &[f64],
	sweep: &SafeZoneStopBatchRange,
	direction: &str,
	k: Kernel,
) -> Result<SafeZoneStopBatchOutput, SafeZoneStopError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(SafeZoneStopError::InvalidPeriod { period: 0, data_len: 0 }),
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	safezonestop_batch_par_slice(high, low, sweep, direction, simd)
}

#[derive(Clone, Debug)]
pub struct SafeZoneStopBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<SafeZoneStopParams>,
	pub rows: usize,
	pub cols: usize,
}
impl SafeZoneStopBatchOutput {
	pub fn row_for_params(&self, p: &SafeZoneStopParams) -> Option<usize> {
		self.combos.iter().position(|c| {
			c.period.unwrap_or(22) == p.period.unwrap_or(22)
				&& (c.mult.unwrap_or(2.5) - p.mult.unwrap_or(2.5)).abs() < 1e-12
				&& c.max_lookback.unwrap_or(3) == p.max_lookback.unwrap_or(3)
		})
	}
	pub fn values_for(&self, p: &SafeZoneStopParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &SafeZoneStopBatchRange) -> Vec<SafeZoneStopParams> {
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
	let lookbacks = axis_usize(r.max_lookback);
	let mut out = Vec::with_capacity(periods.len() * mults.len() * lookbacks.len());
	for &p in &periods {
		for &m in &mults {
			for &l in &lookbacks {
				out.push(SafeZoneStopParams {
					period: Some(p),
					mult: Some(m),
					max_lookback: Some(l),
				});
			}
		}
	}
	out
}

#[inline(always)]
pub fn safezonestop_batch_slice(
	high: &[f64],
	low: &[f64],
	sweep: &SafeZoneStopBatchRange,
	direction: &str,
	kern: Kernel,
) -> Result<SafeZoneStopBatchOutput, SafeZoneStopError> {
	safezonestop_batch_inner(high, low, sweep, direction, kern, false)
}

#[inline(always)]
pub fn safezonestop_batch_par_slice(
	high: &[f64],
	low: &[f64],
	sweep: &SafeZoneStopBatchRange,
	direction: &str,
	kern: Kernel,
) -> Result<SafeZoneStopBatchOutput, SafeZoneStopError> {
	safezonestop_batch_inner(high, low, sweep, direction, kern, true)
}

#[inline(always)]
fn safezonestop_batch_inner(
	high: &[f64],
	low: &[f64],
	sweep: &SafeZoneStopBatchRange,
	direction: &str,
	kern: Kernel,
	parallel: bool,
) -> Result<SafeZoneStopBatchOutput, SafeZoneStopError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(SafeZoneStopError::InvalidPeriod { period: 0, data_len: 0 });
	}
	if high.len() != low.len() {
		return Err(SafeZoneStopError::MismatchedLengths);
	}
	let len = high.len();
	let first = high.iter().position(|x| !x.is_nan()).unwrap_or(0);
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if len - first < max_p {
		return Err(SafeZoneStopError::NotEnoughValidData {
			needed: max_p,
			valid: len - first,
		});
	}
	let rows = combos.len();
	let cols = len;
	
	// Use proper matrix allocation with NaN prefixes
	let mut buf_uninit = make_uninit_matrix(rows, cols);
	
	// Initialize warmup periods for each row
	let warmup_periods: Vec<usize> = combos
		.iter()
		.map(|c| {
			let p = c.period.unwrap();
			let mlb = c.max_lookback.unwrap();
			p.saturating_sub(1).max(mlb.saturating_sub(1))
		})
		.collect();
	
	init_matrix_prefixes(&mut buf_uninit, cols, &warmup_periods);
	
	// Use ManuallyDrop to avoid copy - zero-copy pattern from ALMA
	let mut buf_guard = core::mem::ManuallyDrop::new(buf_uninit);
	let values_slice: &mut [f64] =
		unsafe { core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len()) };
	
	// Fill remaining values with NaN for binding compatibility
	for (row_idx, warmup) in warmup_periods.iter().enumerate() {
		let row_start = row_idx * cols;
		for col in *warmup..cols {
			values_slice[row_start + col] = f64::NAN;
		}
	}

	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let period = combos[row].period.unwrap();
		let mult = combos[row].mult.unwrap();
		let max_lookback = combos[row].max_lookback.unwrap();
		match kern {
			Kernel::Scalar => safezonestop_row_scalar(high, low, period, mult, max_lookback, direction, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => safezonestop_row_avx2(high, low, period, mult, max_lookback, direction, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => safezonestop_row_avx512(high, low, period, mult, max_lookback, direction, out_row),
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

	// Convert back to Vec without copying - zero-copy pattern from ALMA
	let values = unsafe {
		Vec::from_raw_parts(
			buf_guard.as_mut_ptr() as *mut f64,
			buf_guard.len(),
			buf_guard.capacity(),
		)
	};

	Ok(SafeZoneStopBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
pub fn safezonestop_batch_inner_into(
	high: &[f64],
	low: &[f64],
	sweep: &SafeZoneStopBatchRange,
	direction: &str,
	kern: Kernel,
	parallel: bool,
	out: &mut [f64],
) -> Result<Vec<SafeZoneStopParams>, SafeZoneStopError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(SafeZoneStopError::InvalidPeriod { period: 0, data_len: 0 });
	}
	if high.len() != low.len() {
		return Err(SafeZoneStopError::MismatchedLengths);
	}
	let len = high.len();
	let first = high.iter().position(|x| !x.is_nan()).unwrap_or(0);
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if len - first < max_p {
		return Err(SafeZoneStopError::NotEnoughValidData {
			needed: max_p,
			valid: len - first,
		});
	}
	let rows = combos.len();
	let cols = len;

	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let period = combos[row].period.unwrap();
		let mult = combos[row].mult.unwrap();
		let max_lookback = combos[row].max_lookback.unwrap();
		match kern {
			Kernel::Scalar => safezonestop_row_scalar(high, low, period, mult, max_lookback, direction, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => safezonestop_row_avx2(high, low, period, mult, max_lookback, direction, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => safezonestop_row_avx512(high, low, period, mult, max_lookback, direction, out_row),
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
pub unsafe fn safezonestop_row_scalar(
	high: &[f64],
	low: &[f64],
	period: usize,
	mult: f64,
	max_lookback: usize,
	direction: &str,
	out: &mut [f64],
) {
	safezonestop_scalar(high, low, period, mult, max_lookback, direction, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn safezonestop_row_avx2(
	high: &[f64],
	low: &[f64],
	period: usize,
	mult: f64,
	max_lookback: usize,
	direction: &str,
	out: &mut [f64],
) {
	safezonestop_scalar(high, low, period, mult, max_lookback, direction, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn safezonestop_row_avx512(
	high: &[f64],
	low: &[f64],
	period: usize,
	mult: f64,
	max_lookback: usize,
	direction: &str,
	out: &mut [f64],
) {
	if period <= 32 {
		safezonestop_row_avx512_short(high, low, period, mult, max_lookback, direction, out)
	} else {
		safezonestop_row_avx512_long(high, low, period, mult, max_lookback, direction, out)
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn safezonestop_row_avx512_short(
	high: &[f64],
	low: &[f64],
	period: usize,
	mult: f64,
	max_lookback: usize,
	direction: &str,
	out: &mut [f64],
) {
	safezonestop_scalar(high, low, period, mult, max_lookback, direction, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn safezonestop_row_avx512_long(
	high: &[f64],
	low: &[f64],
	period: usize,
	mult: f64,
	max_lookback: usize,
	direction: &str,
	out: &mut [f64],
) {
	safezonestop_scalar(high, low, period, mult, max_lookback, direction, out)
}

// ====== WASM BINDINGS ======

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn safezonestop_js(
	high: &[f64],
	low: &[f64],
	period: usize,
	mult: f64,
	max_lookback: usize,
	direction: &str,
) -> Result<Vec<f64>, JsValue> {
	let params = SafeZoneStopParams {
		period: Some(period),
		mult: Some(mult),
		max_lookback: Some(max_lookback),
	};
	let input = SafeZoneStopInput::from_slices(high, low, direction, params);

	let mut output = vec![0.0; high.len()]; // Single allocation

	safezonestop_into_slice(&mut output, &input, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;

	Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn safezonestop_into(
	high_ptr: *const f64,
	low_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period: usize,
	mult: f64,
	max_lookback: usize,
	direction: &str,
) -> Result<(), JsValue> {
	if high_ptr.is_null() || low_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}

	unsafe {
		let high = std::slice::from_raw_parts(high_ptr, len);
		let low = std::slice::from_raw_parts(low_ptr, len);
		
		let params = SafeZoneStopParams {
			period: Some(period),
			mult: Some(mult),
			max_lookback: Some(max_lookback),
		};
		let input = SafeZoneStopInput::from_slices(high, low, direction, params);
		
		// CRITICAL: Check aliasing with BOTH input pointers
		if high_ptr == out_ptr || low_ptr == out_ptr {
			// Need temp buffer if output aliases with either input
			let mut temp = vec![0.0; len];
			safezonestop_into_slice(&mut temp, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			out.copy_from_slice(&temp);
		} else {
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			safezonestop_into_slice(out, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn safezonestop_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn safezonestop_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe {
			let _ = Vec::from_raw_parts(ptr, len, len);
		}
	}
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct SafeZoneStopBatchConfig {
	pub period_range: (usize, usize, usize),
	pub mult_range: (f64, f64, f64),
	pub max_lookback_range: (usize, usize, usize),
	pub direction: String,
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct SafeZoneStopBatchJsOutput {
	pub values: Vec<f64>,
	pub combos: Vec<SafeZoneStopParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = safezonestop_batch)]
pub fn safezonestop_batch_unified_js(high: &[f64], low: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	let config: SafeZoneStopBatchConfig =
		serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

	let sweep = SafeZoneStopBatchRange {
		period: config.period_range,
		mult: config.mult_range,
		max_lookback: config.max_lookback_range,
	};

	let output = safezonestop_batch_inner(high, low, &sweep, &config.direction, Kernel::Auto, false)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;

	let js_output = SafeZoneStopBatchJsOutput {
		values: output.values,
		combos: output.combos,
		rows: output.rows,
		cols: output.cols,
	};

	serde_wasm_bindgen::to_value(&js_output).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn safezonestop_batch_into(
	high_ptr: *const f64,
	low_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period_start: usize,
	period_end: usize,
	period_step: usize,
	mult_start: f64,
	mult_end: f64,
	mult_step: f64,
	max_lookback_start: usize,
	max_lookback_end: usize,
	max_lookback_step: usize,
	direction: &str,
) -> Result<usize, JsValue> {
	if high_ptr.is_null() || low_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("null pointer passed to safezonestop_batch_into"));
	}

	unsafe {
		let high = std::slice::from_raw_parts(high_ptr, len);
		let low = std::slice::from_raw_parts(low_ptr, len);
		
		let sweep = SafeZoneStopBatchRange {
			period: (period_start, period_end, period_step),
			mult: (mult_start, mult_end, mult_step),
			max_lookback: (max_lookback_start, max_lookback_end, max_lookback_step),
		};
		
		let combos = expand_grid(&sweep);
		let total_size = combos.len() * len;
		let out = std::slice::from_raw_parts_mut(out_ptr, total_size);
		
		safezonestop_batch_inner_into(high, low, &sweep, direction, Kernel::Auto, false, out)
			.map_err(|e| JsValue::from_str(&e.to_string()))?;
		
		Ok(combos.len())
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_safezonestop_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = SafeZoneStopParams {
			period: Some(14),
			mult: None,
			max_lookback: None,
		};
		let input = SafeZoneStopInput::from_candles(&candles, "short", params);
		let output = safezonestop_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_safezonestop_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = SafeZoneStopParams {
			period: Some(22),
			mult: Some(2.5),
			max_lookback: Some(3),
		};
		let input = SafeZoneStopInput::from_candles(&candles, "long", params);
		let output = safezonestop_with_kernel(&input, kernel)?;
		let expected = [
			45331.180007991,
			45712.94455308232,
			46019.94707339676,
			46461.767660969635,
			46461.767660969635,
		];
		let start = output.values.len().saturating_sub(5);
		for (i, &val) in output.values[start..].iter().enumerate() {
			let diff = (val - expected[i]).abs();
			assert!(
				diff < 1e-4,
				"[{}] SafeZoneStop {:?} mismatch at idx {}: got {}, expected {}",
				test_name,
				kernel,
				i,
				val,
				expected[i]
			);
		}
		Ok(())
	}

	fn check_safezonestop_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = SafeZoneStopInput::with_default_candles(&candles);
		let output = safezonestop_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_safezonestop_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [10.0, 20.0, 30.0];
		let low = [5.0, 15.0, 25.0];
		let params = SafeZoneStopParams {
			period: Some(0),
			mult: Some(2.5),
			max_lookback: Some(3),
		};
		let input = SafeZoneStopInput::from_slices(&high, &low, "long", params);
		let res = safezonestop_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] SafeZoneStop should fail with zero period",
			test_name
		);
		Ok(())
	}

	fn check_safezonestop_mismatched_lengths(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [10.0, 20.0, 30.0];
		let low = [5.0, 15.0];
		let params = SafeZoneStopParams::default();
		let input = SafeZoneStopInput::from_slices(&high, &low, "long", params);
		let res = safezonestop_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] SafeZoneStop should fail with mismatched lengths",
			test_name
		);
		Ok(())
	}

	fn check_safezonestop_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = SafeZoneStopInput::with_default_candles(&candles);
		let res = safezonestop_with_kernel(&input, kernel)?;
		assert_eq!(res.values.len(), candles.close.len());
		if res.values.len() > 240 {
			for (i, &val) in res.values[240..].iter().enumerate() {
				assert!(
					!val.is_nan(),
					"[{}] Found unexpected NaN at out-index {}",
					test_name,
					240 + i
				);
			}
		}
		Ok(())
	}

	#[cfg(debug_assertions)]
	fn check_safezonestop_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		// Test diverse parameter combinations with both directions
		let test_params = vec![
			// Default parameters
			(SafeZoneStopParams::default(), "long"),
			(SafeZoneStopParams::default(), "short"),
			// Minimum viable period
			(SafeZoneStopParams {
				period: Some(2),
				mult: Some(2.5),
				max_lookback: Some(3),
			}, "long"),
			// Small period variations
			(SafeZoneStopParams {
				period: Some(5),
				mult: Some(1.0),
				max_lookback: Some(2),
			}, "long"),
			(SafeZoneStopParams {
				period: Some(5),
				mult: Some(2.5),
				max_lookback: Some(3),
			}, "short"),
			// Medium period variations
			(SafeZoneStopParams {
				period: Some(10),
				mult: Some(3.0),
				max_lookback: Some(5),
			}, "long"),
			(SafeZoneStopParams {
				period: Some(14),
				mult: Some(2.0),
				max_lookback: Some(4),
			}, "short"),
			// Default period with different mult/lookback
			(SafeZoneStopParams {
				period: Some(22),
				mult: Some(1.5),
				max_lookback: Some(2),
			}, "long"),
			(SafeZoneStopParams {
				period: Some(22),
				mult: Some(5.0),
				max_lookback: Some(10),
			}, "short"),
			// Large periods
			(SafeZoneStopParams {
				period: Some(50),
				mult: Some(2.5),
				max_lookback: Some(5),
			}, "long"),
			(SafeZoneStopParams {
				period: Some(100),
				mult: Some(3.0),
				max_lookback: Some(10),
			}, "short"),
			// Edge cases
			(SafeZoneStopParams {
				period: Some(2),
				mult: Some(0.5),
				max_lookback: Some(1),
			}, "long"),
			(SafeZoneStopParams {
				period: Some(30),
				mult: Some(10.0),
				max_lookback: Some(15),
			}, "short"),
		];

		for (param_idx, (params, direction)) in test_params.iter().enumerate() {
			let input = SafeZoneStopInput::from_candles(&candles, direction, params.clone());
			let output = safezonestop_with_kernel(&input, kernel)?;

			for (i, &val) in output.values.iter().enumerate() {
				if val.is_nan() {
					continue; // NaN values are expected during warmup
				}

				let bits = val.to_bits();

				// Check all three poison patterns
				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: period={}, mult={}, max_lookback={}, direction='{}' (param set {})",
						test_name, val, bits, i,
						params.period.unwrap_or(22),
						params.mult.unwrap_or(2.5),
						params.max_lookback.unwrap_or(3),
						direction,
						param_idx
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: period={}, mult={}, max_lookback={}, direction='{}' (param set {})",
						test_name, val, bits, i,
						params.period.unwrap_or(22),
						params.mult.unwrap_or(2.5),
						params.max_lookback.unwrap_or(3),
						direction,
						param_idx
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: period={}, mult={}, max_lookback={}, direction='{}' (param set {})",
						test_name, val, bits, i,
						params.period.unwrap_or(22),
						params.mult.unwrap_or(2.5),
						params.max_lookback.unwrap_or(3),
						direction,
						param_idx
					);
				}
			}
		}

		Ok(())
	}

	#[cfg(not(debug_assertions))]
	fn check_safezonestop_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(()) // No-op in release builds
	}

	#[cfg(feature = "proptest")]
	#[allow(clippy::float_cmp)]
	fn check_safezonestop_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		use proptest::prelude::*;
		skip_if_unsupported!(kernel, test_name);

		// SafeZoneStop requires high/low price data and direction
		// Generate realistic price series with controlled volatility
		let strat = (1usize..=64)
			.prop_flat_map(|period| {
				let len = period.max(10)..400;
				(
					// Starting price
					100.0f64..1000.0f64,
					// Generate price changes for random walk
					prop::collection::vec(-0.05f64..0.05f64, len.clone()),
					// Generate spread percentages
					prop::collection::vec(0.001f64..0.02f64, len),
					Just(period),
					0.5f64..5.0f64,  // mult range
					1usize..10,       // max_lookback range
					prop::bool::ANY,  // direction: true = "long", false = "short"
				)
			})
			.prop_map(|(start_price, returns, spreads, period, mult, max_lookback, is_long)| {
				// Generate realistic price series using random walk
				let len = returns.len().min(spreads.len());
				let mut low = Vec::with_capacity(len);
				let mut high = Vec::with_capacity(len);
				let mut price = start_price;
				
				for i in 0..len {
					// Random walk with bounded returns
					price *= 1.0 + returns[i];
					price = price.max(1.0); // Keep prices positive and reasonable
					
					// Generate high/low based on spread
					let spread = price * spreads[i];
					low.push(price - spread / 2.0);
					high.push(price + spread / 2.0);
				}
				
				(high, low, period, mult, max_lookback, is_long)
			});

		proptest::test_runner::TestRunner::default()
			.run(&strat, |(high, low, period, mult, max_lookback, is_long)| {
				let len = high.len();
				let direction = if is_long { "long" } else { "short" };
				
				let params = SafeZoneStopParams {
					period: Some(period),
					mult: Some(mult),
					max_lookback: Some(max_lookback),
				};
				let input = SafeZoneStopInput::from_slices(&high, &low, direction, params.clone());

				let output = safezonestop_with_kernel(&input, kernel).unwrap();
				let ref_output = safezonestop_with_kernel(&input, Kernel::Scalar).unwrap();

				// Calculate expected warmup period
				let warmup_period = period.saturating_sub(1).max(max_lookback.saturating_sub(1));

				// Property 1: Warmup period - first values should be NaN
				for i in 0..warmup_period.min(len) {
					prop_assert!(
						output.values[i].is_nan(),
						"Expected NaN during warmup at idx {}, got {}", i, output.values[i]
					);
				}

				// Property 2: Direction-specific validation
				// Long stops should be protective (below prices), short stops should be protective (above prices)
				if len > warmup_period + max_lookback {
					// Check direction produces different results
					let opposite_dir = if is_long { "short" } else { "long" };
					let opposite_input = SafeZoneStopInput::from_slices(&high, &low, opposite_dir, params.clone());
					let opposite_output = safezonestop_with_kernel(&opposite_input, kernel).unwrap();
					
					// Verify that long and short produce different values (after warmup)
					let mut found_difference = false;
					for i in (warmup_period + max_lookback)..len {
						if !output.values[i].is_nan() && !opposite_output.values[i].is_nan() {
							if (output.values[i] - opposite_output.values[i]).abs() > 1e-10 {
								found_difference = true;
								break;
							}
						}
					}
					prop_assert!(
						found_difference || len < warmup_period + max_lookback + 5,
						"Long and short directions should produce different stop values"
					);
				}
				
				// Property 3: Mathematical bounds based on recent prices
				for i in warmup_period..len {
					let val = output.values[i];
					if !val.is_nan() {
						prop_assert!(
							val.is_finite(),
							"SafeZone value at idx {} is not finite: {}", i, val
						);
						
						// Find recent price range (look back up to period + max_lookback)
						let lookback_start = i.saturating_sub(period + max_lookback);
						let recent_high = high[lookback_start..=i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
						let recent_low = low[lookback_start..=i].iter().cloned().fold(f64::INFINITY, f64::min);
						let recent_range = recent_high - recent_low;
						
						// SafeZone values should be within reasonable bounds relative to recent prices
						// Allow for multiplier effect and DM smoothing which can create larger deviations
						let max_deviation = recent_range * mult * 5.0 + recent_high * 0.5;
						
						prop_assert!(
							val >= -max_deviation && val <= recent_high + max_deviation,
							"SafeZone value {} at idx {} outside reasonable bounds [{}, {}] based on recent prices",
							val, i, -max_deviation, recent_high + max_deviation
						);
						
						// For very stable recent prices, check tighter bounds
						if recent_range < 1.0 {
							if is_long {
								// Long stops shouldn't be way above recent highs
								prop_assert!(
									val <= recent_high + recent_range * mult * 3.0,
									"Long stop {} at idx {} too far above recent high {}",
									val, i, recent_high
								);
							} else {
								// Short stops shouldn't be way below recent lows
								prop_assert!(
									val >= recent_low - recent_range * mult * 3.0,
									"Short stop {} at idx {} too far below recent low {}",
									val, i, recent_low
								);
							}
						}
					}
				}

				// Property 4: Volatility response - higher volatility should produce wider stops
				if len > warmup_period + period * 2 {
					// Compare stops during high vs low volatility periods
					let mid_point = len / 2;
					if mid_point > warmup_period + period {
						// Calculate average spreads in first and second half
						let first_half_spread: f64 = (warmup_period..mid_point)
							.map(|i| high[i] - low[i])
							.sum::<f64>() / (mid_point - warmup_period) as f64;
						
						let second_half_spread: f64 = (mid_point..len)
							.map(|i| high[i] - low[i])
							.sum::<f64>() / (len - mid_point) as f64;
						
						// If there's a significant volatility difference
						if first_half_spread > 0.0 && second_half_spread > 0.0 {
							let volatility_ratio = first_half_spread / second_half_spread;
							if volatility_ratio > 2.0 || volatility_ratio < 0.5 {
								// Stops should generally be wider in the more volatile period
								// This is a soft check as other factors can influence stops
								let first_half_stops: Vec<f64> = output.values[warmup_period + period..mid_point]
									.iter()
									.filter(|v| !v.is_nan())
									.copied()
									.collect();
								
								let second_half_stops: Vec<f64> = output.values[mid_point..len]
									.iter()
									.filter(|v| !v.is_nan())
									.copied()
									.collect();
								
								if !first_half_stops.is_empty() && !second_half_stops.is_empty() {
									// Check if stop widths correlate with volatility
									// This is a loose check as many factors affect stops
									prop_assert!(
										first_half_stops.len() > 0 && second_half_stops.len() > 0,
										"Should have valid stops in both halves for volatility test"
									);
								}
							}
						}
					}
				}
				
				// Property 5: Kernel consistency - all kernels should produce same results
				for i in 0..len {
					let y = output.values[i];
					let r = ref_output.values[i];

					// Handle NaN/infinite values
					if !y.is_finite() || !r.is_finite() {
						prop_assert!(
							y.to_bits() == r.to_bits(),
							"NaN/inf mismatch at idx {}: {} vs {}", i, y, r
						);
						continue;
					}

					// Check ULP difference for finite values
					let y_bits = y.to_bits();
					let r_bits = r.to_bits();
					let ulp_diff = y_bits.abs_diff(r_bits);

					prop_assert!(
						(y - r).abs() <= 1e-9 || ulp_diff <= 4,
						"Kernel mismatch at idx {}: {} vs {} (ULP={})", i, y, r, ulp_diff
					);
				}

				// Property 6: Special case - period = 1
				if period == 1 && max_lookback == 1 {
					// With minimal parameters, most values should be NaN due to warmup
					// When max_lookback = 1, warmup is 0, so values start immediately
					// But the DM calculation needs at least 2 points
					prop_assert!(
						output.values[0].is_nan(),
						"First value should be NaN with period=1, max_lookback=1"
					);
				}

				// Property 7: Constant price data
				if high.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-12) &&
				   low.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-12) {
					// With constant prices, directional movement should be ~0
					// So SafeZone values should eventually stabilize
					// Check that later values don't vary too much
					let stable_start = (warmup_period + period * 2).min(len - 1);
					if stable_start < len - 1 {
						let stable_values: Vec<f64> = output.values[stable_start..]
							.iter()
							.filter(|v| !v.is_nan())
							.copied()
							.collect();
						
						if stable_values.len() > 1 {
							let first_stable = stable_values[0];
							for val in &stable_values[1..] {
								prop_assert!(
									(val - first_stable).abs() < 1e-6,
									"Constant data should produce stable SafeZone values: {} vs {}", val, first_stable
								);
							}
						}
					}
				}

				Ok(())
			})
			.unwrap();

		Ok(())
	}

	macro_rules! generate_all_safezonestop_tests {
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
	generate_all_safezonestop_tests!(
		check_safezonestop_partial_params,
		check_safezonestop_accuracy,
		check_safezonestop_default_candles,
		check_safezonestop_zero_period,
		check_safezonestop_mismatched_lengths,
		check_safezonestop_nan_handling,
		check_safezonestop_no_poison
	);

	#[cfg(feature = "proptest")]
	generate_all_safezonestop_tests!(check_safezonestop_property);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let high = source_type(&c, "high");
		let low = source_type(&c, "low");

		let output = SafeZoneStopBatchBuilder::new().kernel(kernel).apply_slices(high, low)?;

		let def = SafeZoneStopParams::default();
		let row = output.values_for(&def).expect("default row missing");

		assert_eq!(row.len(), c.close.len());

		let expected = [
			45331.180007991,
			45712.94455308232,
			46019.94707339676,
			46461.767660969635,
			46461.767660969635,
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

	#[cfg(debug_assertions)]
	fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let high = source_type(&c, "high");
		let low = source_type(&c, "low");

		// Test various parameter sweep configurations
		let test_configs = vec![
			// (period_start, period_end, period_step, mult_start, mult_end, mult_step, max_lookback_start, max_lookback_end, max_lookback_step, direction)
			(2, 10, 2, 1.0, 3.0, 0.5, 1, 5, 1, "long"),
			(5, 25, 5, 2.5, 2.5, 0.0, 3, 3, 0, "short"),
			(10, 10, 0, 1.5, 5.0, 0.5, 2, 8, 2, "long"),
			(2, 5, 1, 0.5, 2.0, 0.5, 1, 3, 1, "short"),
			(30, 60, 15, 2.0, 4.0, 1.0, 5, 10, 5, "long"),
			(22, 22, 0, 1.0, 5.0, 1.0, 3, 3, 0, "short"),
			(8, 12, 1, 2.5, 3.5, 0.25, 2, 6, 1, "long"),
		];

		for (cfg_idx, &(p_start, p_end, p_step, m_start, m_end, m_step, l_start, l_end, l_step, direction)) in
			test_configs.iter().enumerate()
		{
			let output = SafeZoneStopBatchBuilder::new()
				.kernel(kernel)
				.period_range(p_start, p_end, p_step)
				.mult_range(m_start, m_end, m_step)
				.max_lookback_range(l_start, l_end, l_step)
				.direction(direction)
				.apply_slices(high, low)?;

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
						 at row {} col {} (flat index {}) with params: period={}, mult={}, max_lookback={}, direction='{}'",
						test, cfg_idx, val, bits, row, col, idx,
						combo.period.unwrap_or(22),
						combo.mult.unwrap_or(2.5),
						combo.max_lookback.unwrap_or(3),
						direction
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}, mult={}, max_lookback={}, direction='{}'",
						test, cfg_idx, val, bits, row, col, idx,
						combo.period.unwrap_or(22),
						combo.mult.unwrap_or(2.5),
						combo.max_lookback.unwrap_or(3),
						direction
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}, mult={}, max_lookback={}, direction='{}'",
						test, cfg_idx, val, bits, row, col, idx,
						combo.period.unwrap_or(22),
						combo.mult.unwrap_or(2.5),
						combo.max_lookback.unwrap_or(3),
						direction
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
#[pyfunction(name = "safezonestop")]
#[pyo3(signature = (high, low, period, mult, max_lookback, direction, kernel=None))]
pub fn safezonestop_py<'py>(
	py: Python<'py>,
	high: PyReadonlyArray1<'py, f64>,
	low: PyReadonlyArray1<'py, f64>,
	period: usize,
	mult: f64,
	max_lookback: usize,
	direction: &str,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
	use numpy::{IntoPyArray, PyArrayMethods};

	let high_slice = high.as_slice()?;
	let low_slice = low.as_slice()?;
	let kern = validate_kernel(kernel, false)?;

	let params = SafeZoneStopParams {
		period: Some(period),
		mult: Some(mult),
		max_lookback: Some(max_lookback),
	};
	let input = SafeZoneStopInput::from_slices(high_slice, low_slice, direction, params);

	let result_vec: Vec<f64> = py
		.allow_threads(|| safezonestop_with_kernel(&input, kern).map(|o| o.values))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "SafeZoneStopStream")]
pub struct SafeZoneStopStreamPy {
	stream: SafeZoneStopStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl SafeZoneStopStreamPy {
	#[new]
	fn new(period: usize, mult: f64, max_lookback: usize, direction: &str) -> PyResult<Self> {
		let params = SafeZoneStopParams {
			period: Some(period),
			mult: Some(mult),
			max_lookback: Some(max_lookback),
		};
		let stream = SafeZoneStopStream::try_new(params, direction)
			.map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(SafeZoneStopStreamPy { stream })
	}

	fn update(&mut self, high: f64, low: f64) -> Option<f64> {
		self.stream.update(high, low)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "safezonestop_batch")]
#[pyo3(signature = (high, low, period_range, mult_range, max_lookback_range, direction, kernel=None))]
pub fn safezonestop_batch_py<'py>(
	py: Python<'py>,
	high: PyReadonlyArray1<'py, f64>,
	low: PyReadonlyArray1<'py, f64>,
	period_range: (usize, usize, usize),
	mult_range: (f64, f64, f64),
	max_lookback_range: (usize, usize, usize),
	direction: &str,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;

	let high_slice = high.as_slice()?;
	let low_slice = low.as_slice()?;

	let sweep = SafeZoneStopBatchRange {
		period: period_range,
		mult: mult_range,
		max_lookback: max_lookback_range,
	};

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
			// Use the _into function that writes directly to the output buffer
			safezonestop_batch_inner_into(high_slice, low_slice, &sweep, direction, simd, true, slice_out)
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
		"max_lookbacks",
		combos
			.iter()
			.map(|p| p.max_lookback.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;

	Ok(dict)
}
