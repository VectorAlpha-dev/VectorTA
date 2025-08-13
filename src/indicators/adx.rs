//! # Average Directional Index (ADX)
//!
//! The Average Directional Index (ADX) is a technical indicator that measures the strength
//! of a prevailing trend by comparing consecutive bars’ highs and lows. The ADX uses
//! smoothed values of the directional movement (positive and negative) to arrive at a single
//! value that signals the intensity of price movement.
//!
//! ## Parameters
//! - **period**: Smoothing period (default 14).
//!
//! ## Errors
//! - **CandleFieldError**: adx: An error occurred while selecting fields from the `Candles`.
//! - **InvalidPeriod**: adx: The specified `period` is zero or exceeds the data length.
//! - **NotEnoughData**: adx: Not enough data points to compute ADX. Requires at least `period + 1` bars.
//!
//! ## Returns
//! - **`Ok(AdxOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(AdxError)`** otherwise.

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

use crate::utilities::data_loader::Candles;
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
use std::mem::ManuallyDrop;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum AdxData<'a> {
	Candles {
		candles: &'a Candles,
	},
	Slices {
		high: &'a [f64],
		low: &'a [f64],
		close: &'a [f64],
	},
}

#[derive(Debug, Clone)]
pub struct AdxOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct AdxParams {
	pub period: Option<usize>,
}

impl Default for AdxParams {
	fn default() -> Self {
		Self { period: Some(14) }
	}
}

#[derive(Debug, Clone)]
pub struct AdxInput<'a> {
	pub data: AdxData<'a>,
	pub params: AdxParams,
}

impl<'a> AdxInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, p: AdxParams) -> Self {
		Self {
			data: AdxData::Candles { candles: c },
			params: p,
		}
	}
	#[inline]
	pub fn from_slices(h: &'a [f64], l: &'a [f64], c: &'a [f64], p: AdxParams) -> Self {
		Self {
			data: AdxData::Slices {
				high: h,
				low: l,
				close: c,
			},
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, AdxParams::default())
	}
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(14)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct AdxBuilder {
	period: Option<usize>,
	kernel: Kernel,
}

impl Default for AdxBuilder {
	fn default() -> Self {
		Self {
			period: None,
			kernel: Kernel::Auto,
		}
	}
}

impl AdxBuilder {
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
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline(always)]
	pub fn apply(self, candles: &Candles) -> Result<AdxOutput, AdxError> {
		let p = AdxParams { period: self.period };
		let i = AdxInput::from_candles(candles, p);
		adx_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slices(self, high: &[f64], low: &[f64], close: &[f64]) -> Result<AdxOutput, AdxError> {
		let p = AdxParams { period: self.period };
		let i = AdxInput::from_slices(high, low, close, p);
		adx_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<AdxStream, AdxError> {
		let p = AdxParams { period: self.period };
		AdxStream::try_new(p)
	}
}
#[derive(Debug, thiserror::Error)]
pub enum AdxError {
	#[error("adx: All values are NaN.")]
	AllValuesNaN,

	#[error("adx: Invalid period: period = {period}, data_len = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },

	#[error("adx: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },

	#[error("adx: Candle field error: {field}")]
	CandleFieldError { field: &'static str },
}

#[inline]
pub fn adx(input: &AdxInput) -> Result<AdxOutput, AdxError> {
	adx_with_kernel(input, Kernel::Auto)
}

pub fn adx_with_kernel(input: &AdxInput, kernel: Kernel) -> Result<AdxOutput, AdxError> {
	let (high, low, close) = match &input.data {
		AdxData::Candles { candles } => {
			let high = candles
				.select_candle_field("high")
				.map_err(|_| AdxError::CandleFieldError { field: "high" })?;
			let low = candles
				.select_candle_field("low")
				.map_err(|_| AdxError::CandleFieldError { field: "low" })?;
			let close = candles
				.select_candle_field("close")
				.map_err(|_| AdxError::CandleFieldError { field: "close" })?;
			(high, low, close)
		}
		AdxData::Slices { high, low, close } => (*high, *low, *close),
	};

	let len = close.len();
	let period = input.get_period();

	if period == 0 || period > len {
		return Err(AdxError::InvalidPeriod { period, data_len: len });
	}
	if len < period + 1 {
		return Err(AdxError::NotEnoughValidData {
			needed: period + 1,
			valid: len,
		});
	}
	if high.iter().all(|x| x.is_nan()) || low.iter().all(|x| x.is_nan()) || close.iter().all(|x| x.is_nan()) {
		return Err(AdxError::AllValuesNaN);
	}

	// Calculate warmup period for ADX (2 * period - 1 is when ADX starts producing values)
	let warmup_period = 2 * period - 1;
	let mut out = alloc_with_nan_prefix(len, warmup_period);

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => adx_scalar(high, low, close, period, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => adx_avx2(high, low, close, period, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => adx_avx512(high, low, close, period, &mut out),
			_ => unreachable!(),
		}
	}

	Ok(AdxOutput { values: out })
}

#[inline]
pub fn adx_scalar(high: &[f64], low: &[f64], close: &[f64], period: usize, out: &mut [f64]) {
	let len = close.len();
	let mut tr_sum = 0.0;
	let mut plus_dm_sum = 0.0;
	let mut minus_dm_sum = 0.0;

	let period_f64 = period as f64;
	let reciprocal_period = 1.0 / period_f64;
	let one_minus_rp = 1.0 - reciprocal_period;
	let period_minus_one = period_f64 - 1.0;

	for i in 1..=period {
		let curr_high = high[i];
		let curr_low = low[i];
		let prev_close = close[i - 1];
		let prev_high = high[i - 1];
		let prev_low = low[i - 1];

		let tr = (curr_high - curr_low)
			.max((curr_high - prev_close).abs())
			.max((curr_low - prev_close).abs());

		let up_move = curr_high - prev_high;
		let down_move = prev_low - curr_low;

		if up_move > down_move && up_move > 0.0 {
			plus_dm_sum += up_move;
		}
		if down_move > up_move && down_move > 0.0 {
			minus_dm_sum += down_move;
		}

		tr_sum += tr;
	}

	let mut atr = tr_sum;
	let mut plus_dm_smooth = plus_dm_sum;
	let mut minus_dm_smooth = minus_dm_sum;

	let plus_di_prev = (plus_dm_smooth / atr) * 100.0;
	let minus_di_prev = (minus_dm_smooth / atr) * 100.0;

	let sum_di = plus_di_prev + minus_di_prev;
	let initial_dx = if sum_di != 0.0 {
		((plus_di_prev - minus_di_prev).abs() / sum_di) * 100.0
	} else {
		0.0
	};

	let mut dx_sum = initial_dx;
	let mut dx_count = 1;
	let mut last_adx = 0.0;
	let mut have_adx = false;

	for i in (period + 1)..len {
		let curr_high = high[i];
		let curr_low = low[i];
		let prev_close = close[i - 1];
		let prev_high = high[i - 1];
		let prev_low = low[i - 1];

		let tr = (curr_high - curr_low)
			.max((curr_high - prev_close).abs())
			.max((curr_low - prev_close).abs());

		let up_move = curr_high - prev_high;
		let down_move = prev_low - curr_low;

		let plus_dm = if up_move > down_move && up_move > 0.0 {
			up_move
		} else {
			0.0
		};
		let minus_dm = if down_move > up_move && down_move > 0.0 {
			down_move
		} else {
			0.0
		};

		atr = atr * one_minus_rp + tr;
		plus_dm_smooth = plus_dm_smooth * one_minus_rp + plus_dm;
		minus_dm_smooth = minus_dm_smooth * one_minus_rp + minus_dm;

		let plus_di_current = (plus_dm_smooth / atr) * 100.0;
		let minus_di_current = (minus_dm_smooth / atr) * 100.0;

		let sum_di_current = plus_di_current + minus_di_current;
		let dx = if sum_di_current != 0.0 {
			let diff = (plus_di_current - minus_di_current).abs();
			(diff / sum_di_current) * 100.0
		} else {
			0.0
		};

		if dx_count < period {
			dx_sum += dx;
			dx_count += 1;
			if dx_count == period {
				last_adx = dx_sum * reciprocal_period;
				out[i] = last_adx;
				have_adx = true;
			}
		} else if have_adx {
			let adx_current = ((last_adx * period_minus_one) + dx) * reciprocal_period;
			out[i] = adx_current;
			last_adx = adx_current;
		}
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn adx_avx2(high: &[f64], low: &[f64], close: &[f64], period: usize, out: &mut [f64]) {
	adx_scalar(high, low, close, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn adx_avx512(high: &[f64], low: &[f64], close: &[f64], period: usize, out: &mut [f64]) {
	adx_scalar(high, low, close, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn adx_avx512_short(high: &[f64], low: &[f64], close: &[f64], period: usize, out: &mut [f64]) {
	adx_scalar(high, low, close, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn adx_avx512_long(high: &[f64], low: &[f64], close: &[f64], period: usize, out: &mut [f64]) {
	adx_scalar(high, low, close, period, out)
}

#[inline]
pub fn adx_batch_with_kernel(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	sweep: &AdxBatchRange,
	k: Kernel,
) -> Result<AdxBatchOutput, AdxError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		Kernel::Scalar => Kernel::ScalarBatch,
		#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
		Kernel::Avx2 => Kernel::Avx2Batch,
		#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
		Kernel::Avx512 => Kernel::Avx512Batch,
		_ => return Err(AdxError::InvalidPeriod { period: 0, data_len: 0 }),
	};

	let simd = match kernel {
		#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
		Kernel::Avx512Batch => Kernel::Avx512,
		#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => Kernel::Scalar,
	};
	adx_batch_par_slice(high, low, close, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct AdxBatchRange {
	pub period: (usize, usize, usize),
}

impl Default for AdxBatchRange {
	fn default() -> Self {
		Self { period: (14, 50, 1) }
	}
}

#[derive(Clone, Debug, Default)]
pub struct AdxBatchBuilder {
	range: AdxBatchRange,
	kernel: Kernel,
}

impl AdxBatchBuilder {
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
	pub fn period_static(mut self, p: usize) -> Self {
		self.range.period = (p, p, 0);
		self
	}
	pub fn apply_slices(self, high: &[f64], low: &[f64], close: &[f64]) -> Result<AdxBatchOutput, AdxError> {
		adx_batch_with_kernel(high, low, close, &self.range, self.kernel)
	}
	pub fn apply_candles(self, candles: &Candles) -> Result<AdxBatchOutput, AdxError> {
		let high = candles
			.select_candle_field("high")
			.map_err(|_| AdxError::CandleFieldError { field: "high" })?;
		let low = candles
			.select_candle_field("low")
			.map_err(|_| AdxError::CandleFieldError { field: "low" })?;
		let close = candles
			.select_candle_field("close")
			.map_err(|_| AdxError::CandleFieldError { field: "close" })?;
		self.apply_slices(high, low, close)
	}
	pub fn with_default_candles(c: &Candles) -> Result<AdxBatchOutput, AdxError> {
		AdxBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c)
	}
}

#[derive(Clone, Debug)]
pub struct AdxBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<AdxParams>,
	pub rows: usize,
	pub cols: usize,
}

impl AdxBatchOutput {
	pub fn row_for_params(&self, p: &AdxParams) -> Option<usize> {
		self.combos
			.iter()
			.position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
	}

	pub fn values_for(&self, p: &AdxParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &AdxBatchRange) -> Vec<AdxParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let periods = axis_usize(r.period);
	let mut out = Vec::with_capacity(periods.len());
	for &p in &periods {
		out.push(AdxParams { period: Some(p) });
	}
	out
}

#[inline(always)]
pub fn adx_batch_slice(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	sweep: &AdxBatchRange,
	kern: Kernel,
) -> Result<AdxBatchOutput, AdxError> {
	adx_batch_inner(high, low, close, sweep, kern, false)
}

#[inline(always)]
pub fn adx_batch_par_slice(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	sweep: &AdxBatchRange,
	kern: Kernel,
) -> Result<AdxBatchOutput, AdxError> {
	adx_batch_inner(high, low, close, sweep, kern, true)
}

#[inline(always)]
fn adx_batch_inner(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	sweep: &AdxBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<AdxBatchOutput, AdxError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(AdxError::InvalidPeriod { period: 0, data_len: 0 });
	}
	let rows = combos.len();
	let cols = close.len();

	// Step 1: Allocate uninitialized matrix
	let mut buf_mu = make_uninit_matrix(rows, cols);

	// Step 2: Calculate warmup periods for each row (ADX warmup is 2 * period - 1)
	let warm: Vec<usize> = combos.iter().map(|c| 2 * c.period.unwrap() - 1).collect();

	// Step 3: Initialize NaN prefixes for each row
	init_matrix_prefixes(&mut buf_mu, cols, &warm);

	// Step 4: Convert to mutable slice for computation
	let mut buf_guard = ManuallyDrop::new(buf_mu);
	let values: &mut [f64] =
		unsafe { core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len()) };
	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let period = combos[row].period.unwrap();
		match kern {
			Kernel::Scalar => adx_row_scalar(high, low, close, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => adx_row_avx2(high, low, close, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => adx_row_avx512(high, low, close, period, out_row),
			_ => adx_row_scalar(high, low, close, period, out_row),
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

	Ok(AdxBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
unsafe fn adx_row_scalar(high: &[f64], low: &[f64], close: &[f64], period: usize, out: &mut [f64]) {
	adx_scalar(high, low, close, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn adx_row_avx2(high: &[f64], low: &[f64], close: &[f64], period: usize, out: &mut [f64]) {
	adx_row_scalar(high, low, close, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn adx_row_avx512(high: &[f64], low: &[f64], close: &[f64], period: usize, out: &mut [f64]) {
	adx_row_scalar(high, low, close, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn adx_row_avx512_short(high: &[f64], low: &[f64], close: &[f64], period: usize, out: &mut [f64]) {
	adx_row_scalar(high, low, close, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn adx_row_avx512_long(high: &[f64], low: &[f64], close: &[f64], period: usize, out: &mut [f64]) {
	adx_row_scalar(high, low, close, period, out)
}

#[derive(Debug, Clone)]
pub struct AdxStream {
	period: usize,
	atr: f64,
	plus_dm_smooth: f64,
	minus_dm_smooth: f64,
	dx_sum: f64,
	dx_count: usize,
	last_adx: f64,
	count: usize,
	prev_high: f64,
	prev_low: f64,
	prev_close: f64,
}

impl AdxStream {
	pub fn try_new(params: AdxParams) -> Result<Self, AdxError> {
		let period = params.period.unwrap_or(14);
		if period == 0 {
			return Err(AdxError::InvalidPeriod { period, data_len: 0 });
		}
		Ok(Self {
			period,
			atr: 0.0,
			plus_dm_smooth: 0.0,
			minus_dm_smooth: 0.0,
			dx_sum: 0.0,
			dx_count: 0,
			last_adx: 0.0,
			count: 0,
			prev_high: f64::NAN,
			prev_low: f64::NAN,
			prev_close: f64::NAN,
		})
	}

	#[inline(always)]
	pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
		if self.count == 0 {
			self.prev_high = high;
			self.prev_low = low;
			self.prev_close = close;
			self.count = 1;
			return None;
		}

		let tr = (high - low)
			.max((high - self.prev_close).abs())
			.max((low - self.prev_close).abs());
		let up_move = high - self.prev_high;
		let down_move = self.prev_low - low;
		let plus_dm = if up_move > down_move && up_move > 0.0 {
			up_move
		} else {
			0.0
		};
		let minus_dm = if down_move > up_move && down_move > 0.0 {
			down_move
		} else {
			0.0
		};

		self.count += 1;

		if self.count <= self.period + 1 {
			self.atr += tr;
			self.plus_dm_smooth += plus_dm;
			self.minus_dm_smooth += minus_dm;

			if self.count == self.period + 1 {
				let plus_di_prev = (self.plus_dm_smooth / self.atr) * 100.0;
				let minus_di_prev = (self.minus_dm_smooth / self.atr) * 100.0;
				let sum_di = plus_di_prev + minus_di_prev;
				let initial_dx = if sum_di != 0.0 {
					((plus_di_prev - minus_di_prev).abs() / sum_di) * 100.0
				} else {
					0.0
				};
				self.dx_sum = initial_dx;
				self.dx_count = 1;
			}

			self.prev_high = high;
			self.prev_low = low;
			self.prev_close = close;
			return None;
		}

		let rp = 1.0 / self.period as f64;
		let one_minus_rp = 1.0 - rp;
		let period_minus_one = self.period as f64 - 1.0;

		self.atr = self.atr * one_minus_rp + tr;
		self.plus_dm_smooth = self.plus_dm_smooth * one_minus_rp + plus_dm;
		self.minus_dm_smooth = self.minus_dm_smooth * one_minus_rp + minus_dm;

		let plus_di = (self.plus_dm_smooth / self.atr) * 100.0;
		let minus_di = (self.minus_dm_smooth / self.atr) * 100.0;
		let sum_di = plus_di + minus_di;
		let dx = if sum_di != 0.0 {
			((plus_di - minus_di).abs() / sum_di) * 100.0
		} else {
			0.0
		};

		let out = if self.dx_count < self.period {
			self.dx_sum += dx;
			self.dx_count += 1;
			if self.dx_count == self.period {
				self.last_adx = self.dx_sum * rp;
				Some(self.last_adx)
			} else {
				None
			}
		} else {
			self.last_adx = ((self.last_adx * period_minus_one) + dx) * rp;
			Some(self.last_adx)
		};

		self.prev_high = high;
		self.prev_low = low;
		self.prev_close = close;

		out
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_adx_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let default_params = AdxParams { period: None };
		let input = AdxInput::from_candles(&candles, default_params);
		let output = adx_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());

		Ok(())
	}

	fn check_adx_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let input = AdxInput::from_candles(&candles, AdxParams::default());
		let result = adx_with_kernel(&input, kernel)?;
		let expected_last_five = [36.14, 36.52, 37.01, 37.46, 38.47];
		let start = result.values.len().saturating_sub(5);
		for (i, &val) in result.values[start..].iter().enumerate() {
			let diff = (val - expected_last_five[i]).abs();
			assert!(
				diff < 1e-1,
				"[{}] ADX {:?} mismatch at idx {}: got {}, expected {}",
				test_name,
				kernel,
				i,
				val,
				expected_last_five[i]
			);
		}
		Ok(())
	}

	fn check_adx_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let input = AdxInput::with_default_candles(&candles);
		let output = adx_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());

		Ok(())
	}

	fn check_adx_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [10.0, 20.0, 30.0];
		let low = [5.0, 15.0, 25.0];
		let close = [9.0, 19.0, 29.0];
		let params = AdxParams { period: Some(0) };
		let input = AdxInput::from_slices(&high, &low, &close, params);
		let res = adx_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] ADX should fail with zero period", test_name);
		Ok(())
	}

	fn check_adx_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [10.0, 20.0, 30.0];
		let low = [5.0, 15.0, 25.0];
		let close = [9.0, 19.0, 29.0];
		let params = AdxParams { period: Some(10) };
		let input = AdxInput::from_slices(&high, &low, &close, params);
		let res = adx_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] ADX should fail with period exceeding length",
			test_name
		);
		Ok(())
	}

	fn check_adx_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [42.0];
		let low = [41.0];
		let close = [40.5];
		let params = AdxParams { period: Some(14) };
		let input = AdxInput::from_slices(&high, &low, &close, params);
		let res = adx_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] ADX should fail with insufficient data", test_name);
		Ok(())
	}

	fn check_adx_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let first_params = AdxParams { period: Some(14) };
		let first_input = AdxInput::from_candles(&candles, first_params);
		let first_result = adx_with_kernel(&first_input, kernel)?;

		let second_params = AdxParams { period: Some(5) };
		let second_input = AdxInput::from_slices(&candles.high, &candles.low, &first_result.values, second_params);
		let second_result = adx_with_kernel(&second_input, kernel)?;

		assert_eq!(second_result.values.len(), candles.close.len());
		for i in 100..second_result.values.len() {
			assert!(!second_result.values[i].is_nan());
		}
		Ok(())
	}

	fn check_adx_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let input = AdxInput::from_candles(&candles, AdxParams { period: Some(14) });
		let res = adx_with_kernel(&input, kernel)?;
		assert_eq!(res.values.len(), candles.close.len());
		if res.values.len() > 100 {
			for (i, &val) in res.values[100..].iter().enumerate() {
				assert!(
					!val.is_nan(),
					"[{}] Found unexpected NaN at out-index {}",
					test_name,
					100 + i
				);
			}
		}
		Ok(())
	}

	fn check_adx_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let period = 14;

		let input = AdxInput::from_candles(&candles, AdxParams { period: Some(period) });
		let batch_output = adx_with_kernel(&input, kernel)?.values;

		let mut stream = AdxStream::try_new(AdxParams { period: Some(period) })?;
		let mut stream_values = Vec::with_capacity(candles.close.len());
		for ((&h, &l), &c) in candles.high.iter().zip(&candles.low).zip(&candles.close) {
			match stream.update(h, l, c) {
				Some(adx_val) => stream_values.push(adx_val),
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
				"[{}] ADX streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
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
	fn check_adx_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let test_params = vec![
			AdxParams::default(),
			AdxParams { period: Some(5) },
			AdxParams { period: Some(10) },
			AdxParams { period: Some(20) },
			AdxParams { period: Some(50) },
		];

		for params in test_params {
			// Test with high/low/close slices
			let input = AdxInput::from_slices(&candles.high, &candles.low, &candles.close, params.clone());
			let output = adx_with_kernel(&input, kernel)?;

			// Check every value in the output
			for (idx, &val) in output.values.iter().enumerate() {
				if val.is_nan() || val.is_infinite() {
					continue;
				}

				let bits = val.to_bits();

				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						with params: period={}",
						test_name,
						val,
						bits,
						idx,
						params.period.unwrap_or(14)
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						with params: period={}",
						test_name,
						val,
						bits,
						idx,
						params.period.unwrap_or(14)
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						with params: period={}",
						test_name,
						val,
						bits,
						idx,
						params.period.unwrap_or(14)
					);
				}
			}
		}

		Ok(())
	}

	#[cfg(not(debug_assertions))]
	fn check_adx_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(())
	}

	macro_rules! generate_all_adx_tests {
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

	generate_all_adx_tests!(
		check_adx_partial_params,
		check_adx_accuracy,
		check_adx_default_candles,
		check_adx_zero_period,
		check_adx_period_exceeds_length,
		check_adx_very_small_dataset,
		check_adx_reinput,
		check_adx_nan_handling,
		check_adx_streaming,
		check_adx_no_poison
	);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let output = AdxBatchBuilder::new().kernel(kernel).apply_candles(&c)?;

		let def = AdxParams::default();
		let row = output
			.combos
			.iter()
			.position(|p| p.period == def.period)
			.expect("default row missing");
		let slice = &output.values[row * output.cols..][..output.cols];

		assert_eq!(slice.len(), c.close.len());
		let expected = [36.14, 36.52, 37.01, 37.46, 38.47];
		let start = slice.len().saturating_sub(5);
		for (i, &v) in slice[start..].iter().enumerate() {
			assert!(
				(v - expected[i]).abs() < 1e-1,
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

		let test_configs = vec![
			(5, 20, 5),   // period_start, period_end, period_step
			(10, 30, 10),
			(14, 14, 1),  // default period only
			(20, 50, 15),
			(2, 10, 2),
		];

		for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
			let output = AdxBatchBuilder::new()
				.kernel(kernel)
				.period_range(p_start, p_end, p_step)
				.apply_candles(&c)?;

			// Check every value in the flat output matrix
			for (idx, &val) in output.values.iter().enumerate() {
				if val.is_nan() || val.is_infinite() {
					continue;
				}

				let bits = val.to_bits();
				let row = idx / output.cols;
				let col = idx % output.cols;
				let combo = &output.combos[row];

				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
						at row {} col {} (flat index {}) with params: period={}",
						test,
						cfg_idx,
						val,
						bits,
						row,
						col,
						idx,
						combo.period.unwrap_or(14)
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						at row {} col {} (flat index {}) with params: period={}",
						test,
						cfg_idx,
						val,
						bits,
						row,
						col,
						idx,
						combo.period.unwrap_or(14)
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						at row {} col {} (flat index {}) with params: period={}",
						test,
						cfg_idx,
						val,
						bits,
						row,
						col,
						idx,
						combo.period.unwrap_or(14)
					);
				}
			}

			// Also check intermediate buffers if exposed via from_raw
			let params = expand_grid(&AdxBatchRange {
				period: (p_start, p_end, p_step),
			});

			// Test slicing back to individual outputs
			for p in &params {
				if let Some(slice) = output.values_for(p) {
					for (idx, &val) in slice.iter().enumerate() {
						if val.is_nan() || val.is_infinite() {
							continue;
						}

						let bits = val.to_bits();
						if bits == 0x11111111_11111111 || bits == 0x22222222_22222222 || bits == 0x33333333_33333333 {
							panic!(
								"[{}] Config {}: Found poison value {} (0x{:016X}) in sliced output \
								at index {} with params: period={}",
								test, cfg_idx, val, bits, idx, p.period.unwrap_or(14)
							);
						}
					}
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
#[pyfunction(name = "adx")]
#[pyo3(signature = (high, low, close, period, kernel=None))]
pub fn adx_py<'py>(
	py: Python<'py>,
	high: numpy::PyReadonlyArray1<'py, f64>,
	low: numpy::PyReadonlyArray1<'py, f64>,
	close: numpy::PyReadonlyArray1<'py, f64>,
	period: usize,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
	use numpy::{IntoPyArray, PyArrayMethods};

	let high_slice = high.as_slice()?;
	let low_slice = low.as_slice()?;
	let close_slice = close.as_slice()?;

	// Validate input lengths match
	if high_slice.len() != low_slice.len() || high_slice.len() != close_slice.len() {
		return Err(PyValueError::new_err("Input arrays must have the same length"));
	}

	// Validate kernel before allow_threads
	let kern = validate_kernel(kernel, false)?;

	// Build input struct
	let params = AdxParams { period: Some(period) };
	let adx_in = AdxInput::from_slices(high_slice, low_slice, close_slice, params);

	// Get Vec<f64> from Rust function
	let result_vec: Vec<f64> = py
		.allow_threads(|| adx_with_kernel(&adx_in, kern).map(|o| o.values))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	// Zero-copy transfer to NumPy
	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "AdxStream")]
pub struct AdxStreamPy {
	stream: AdxStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl AdxStreamPy {
	#[new]
	fn new(period: usize) -> PyResult<Self> {
		let params = AdxParams { period: Some(period) };
		let stream = AdxStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(AdxStreamPy { stream })
	}

	/// Updates the stream with new high, low, close values and returns the calculated ADX value.
	/// Returns `None` if the buffer is not yet full.
	fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
		self.stream.update(high, low, close)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "adx_batch")]
#[pyo3(signature = (high, low, close, period_range, kernel=None))]
pub fn adx_batch_py<'py>(
	py: Python<'py>,
	high: numpy::PyReadonlyArray1<'py, f64>,
	low: numpy::PyReadonlyArray1<'py, f64>,
	close: numpy::PyReadonlyArray1<'py, f64>,
	period_range: (usize, usize, usize),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;

	let high_slice = high.as_slice()?;
	let low_slice = low.as_slice()?;
	let close_slice = close.as_slice()?;

	// Validate input lengths match
	if high_slice.len() != low_slice.len() || high_slice.len() != close_slice.len() {
		return Err(PyValueError::new_err("Input arrays must have the same length"));
	}

	// Validate kernel before allow_threads
	let kern = validate_kernel(kernel, true)?;

	let sweep = AdxBatchRange { period: period_range };

	// Get batch output from Rust function
	let output = py
		.allow_threads(|| {
			// Use the high-level batch function that handles kernel conversion
			adx_batch_with_kernel(high_slice, low_slice, close_slice, &sweep, kern)
		})
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	// Build dict with zero-copy transfers
	let dict = PyDict::new(py);
	dict.set_item("values", output.values.into_pyarray(py).reshape((output.rows, output.cols))?)?;
	dict.set_item(
		"periods",
		output
			.combos
			.iter()
			.map(|p| p.period.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;

	Ok(dict)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn adx_js(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
	// Validate input lengths match
	if high.len() != low.len() || high.len() != close.len() {
		return Err(JsValue::from_str("Input arrays must have the same length"));
	}

	let params = AdxParams { period: Some(period) };
	let input = AdxInput::from_slices(high, low, close, params);
	
	let mut output = vec![0.0; high.len()];  // Single allocation
	#[cfg(target_arch = "wasm32")]
	let kernel = Kernel::Scalar;
	#[cfg(not(target_arch = "wasm32"))]
	let kernel = Kernel::Auto;
	
	adx_into_slice(&mut output, &input, kernel)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn adx_batch_js(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	period_start: usize,
	period_end: usize,
	period_step: usize,
) -> Result<Vec<f64>, JsValue> {
	// Validate input lengths match
	if high.len() != low.len() || high.len() != close.len() {
		return Err(JsValue::from_str("Input arrays must have the same length"));
	}

	let sweep = AdxBatchRange {
		period: (period_start, period_end, period_step),
	};

	// Use the existing batch function with parallel=false for WASM
	#[cfg(target_arch = "wasm32")]
	let kernel = Kernel::ScalarBatch;
	#[cfg(not(target_arch = "wasm32"))]
	let kernel = Kernel::Auto;
	
	adx_batch_inner(high, low, close, &sweep, kernel, false)
		.map(|output| output.values)
		.map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn adx_batch_metadata_js(period_start: usize, period_end: usize, period_step: usize) -> Result<Vec<f64>, JsValue> {
	let sweep = AdxBatchRange {
		period: (period_start, period_end, period_step),
	};

	let combos = expand_grid(&sweep);
	let metadata: Vec<f64> = combos.into_iter().map(|combo| combo.period.unwrap() as f64).collect();

	Ok(metadata)
}

// New ergonomic WASM API
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct AdxBatchConfig {
	pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct AdxBatchJsOutput {
	pub values: Vec<f64>,
	pub combos: Vec<AdxParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = adx_batch)]
pub fn adx_batch_unified_js(high: &[f64], low: &[f64], close: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	// Validate input lengths match
	if high.len() != low.len() || high.len() != close.len() {
		return Err(JsValue::from_str("Input arrays must have the same length"));
	}

	// 1. Deserialize the configuration object from JavaScript
	let config: AdxBatchConfig =
		serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

	let sweep = AdxBatchRange {
		period: config.period_range,
	};

	// 2. Run the existing core logic
	#[cfg(target_arch = "wasm32")]
	let kernel = Kernel::ScalarBatch;
	#[cfg(not(target_arch = "wasm32"))]
	let kernel = Kernel::Auto;
	
	let output = adx_batch_inner(high, low, close, &sweep, kernel, false)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;

	// 3. Create the structured output
	let js_output = AdxBatchJsOutput {
		values: output.values,
		combos: output.combos,
		rows: output.rows,
		cols: output.cols,
	};

	// 4. Serialize the output struct into a JavaScript object
	serde_wasm_bindgen::to_value(&js_output).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

// ====== Optimized WASM API following ALMA pattern ======

/// Core helper function that writes directly to output slice - no allocations
#[inline]
pub fn adx_into_slice(dst: &mut [f64], input: &AdxInput, kern: Kernel) -> Result<(), AdxError> {
	let (high, low, close) = match &input.data {
		AdxData::Candles { candles } => {
			let high = candles
				.select_candle_field("high")
				.map_err(|_| AdxError::CandleFieldError { field: "high" })?;
			let low = candles
				.select_candle_field("low")
				.map_err(|_| AdxError::CandleFieldError { field: "low" })?;
			let close = candles
				.select_candle_field("close")
				.map_err(|_| AdxError::CandleFieldError { field: "close" })?;
			(high, low, close)
		}
		AdxData::Slices { high, low, close } => (*high, *low, *close),
	};

	let len = close.len();
	let period = input.get_period();

	if period == 0 || period > len {
		return Err(AdxError::InvalidPeriod { period, data_len: len });
	}
	if len < period + 1 {
		return Err(AdxError::NotEnoughValidData {
			needed: period + 1,
			valid: len,
		});
	}
	if high.iter().all(|x| x.is_nan()) || low.iter().all(|x| x.is_nan()) || close.iter().all(|x| x.is_nan()) {
		return Err(AdxError::AllValuesNaN);
	}

	if dst.len() != len {
		return Err(AdxError::InvalidPeriod {
			period: dst.len(),
			data_len: len,
		});
	}

	let chosen = match kern {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	// Fill output with NaN prefix for warmup period
	let warmup_period = 2 * period - 1;
	for v in &mut dst[..warmup_period] {
		*v = f64::NAN;
	}

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => adx_scalar(high, low, close, period, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => adx_avx2(high, low, close, period, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => adx_avx512(high, low, close, period, dst),
			_ => unreachable!(),
		}
	}

	Ok(())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn adx_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn adx_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe {
			let _ = Vec::from_raw_parts(ptr, len, len);
		}
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn adx_into(
	high_ptr: *const f64,
	low_ptr: *const f64,
	close_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period: usize,
) -> Result<(), JsValue> {
	if high_ptr.is_null() || low_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}

	unsafe {
		let high = std::slice::from_raw_parts(high_ptr, len);
		let low = std::slice::from_raw_parts(low_ptr, len);
		let close = std::slice::from_raw_parts(close_ptr, len);
		
		let params = AdxParams { period: Some(period) };
		let input = AdxInput::from_slices(high, low, close, params);
		
		#[cfg(target_arch = "wasm32")]
		let kernel = Kernel::Scalar;
		#[cfg(not(target_arch = "wasm32"))]
		let kernel = Kernel::Auto;
		
		// Check for aliasing with any of the input pointers
		if high_ptr == out_ptr || low_ptr == out_ptr || close_ptr == out_ptr {
			// Need temporary buffer for aliased case
			let mut temp = vec![0.0; len];
			adx_into_slice(&mut temp, &input, kernel)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			out.copy_from_slice(&temp);
		} else {
			// No aliasing, can write directly
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			adx_into_slice(out, &input, kernel)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn adx_batch_into(
	high_ptr: *const f64,
	low_ptr: *const f64,
	close_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period_start: usize,
	period_end: usize,
	period_step: usize,
) -> Result<usize, JsValue> {
	if high_ptr.is_null() || low_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("null pointer passed to adx_batch_into"));
	}

	unsafe {
		let high = std::slice::from_raw_parts(high_ptr, len);
		let low = std::slice::from_raw_parts(low_ptr, len);
		let close = std::slice::from_raw_parts(close_ptr, len);

		let sweep = AdxBatchRange {
			period: (period_start, period_end, period_step),
		};

		// Calculate number of combinations
		let combos = expand_grid(&sweep);
		let rows = combos.len();
		let cols = len;

		let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);

		// Use batch function with direct output
		#[cfg(target_arch = "wasm32")]
		let kernel = Kernel::ScalarBatch;
		#[cfg(not(target_arch = "wasm32"))]
		let kernel = Kernel::Auto;
		
		let result = adx_batch_inner(high, low, close, &sweep, kernel, false)
			.map_err(|e| JsValue::from_str(&e.to_string()))?;
		
		// Copy results to output buffer
		out.copy_from_slice(&result.values);

		Ok(rows)
	}
}
