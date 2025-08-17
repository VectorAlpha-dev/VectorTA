//! # DX (Directional Movement Index)
//!
//! Measures trend strength by comparing smoothed +DI and -DI, using Welles Wilder’s logic.
//! - `period`: window size (typically 14).
//!
//! ## Errors
//! - **EmptyData**: All input slices empty.
//! - **SelectCandleFieldError**: Failed to select field from `Candles`.
//! - **InvalidPeriod**: `period` is zero or exceeds data length.
//! - **NotEnoughValidData**: Not enough valid data after first valid index.
//! - **AllValuesNaN**: All high, low, and close values are NaN.
//!
//! ## Returns
//! - **Ok(DxOutput)** on success, with output length matching input.
//! - **Err(DxError)** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
	alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, 
	init_matrix_prefixes, make_uninit_matrix
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
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
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


#[derive(Debug, Clone)]
pub enum DxData<'a> {
	Candles {
		candles: &'a Candles,
	},
	HlcSlices {
		high: &'a [f64],
		low: &'a [f64],
		close: &'a [f64],
	},
}

#[derive(Debug, Clone)]
pub struct DxOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct DxParams {
	pub period: Option<usize>,
}

impl Default for DxParams {
	fn default() -> Self {
		Self { period: Some(14) }
	}
}

impl DxParams {
	pub fn generate_batch_params(period_range: (usize, usize, usize)) -> Vec<Self> {
		let (start, end, step) = period_range;
		let step = if step == 0 { 1 } else { step };
		let mut params = Vec::new();
		
		let mut period = start;
		while period <= end {
			params.push(Self { period: Some(period) });
			period += step;
		}
		
		params
	}
}

#[derive(Debug, Clone)]
pub struct DxInput<'a> {
	pub data: DxData<'a>,
	pub params: DxParams,
}

impl<'a> DxInput<'a> {
	#[inline]
	pub fn from_candles(candles: &'a Candles, params: DxParams) -> Self {
		Self {
			data: DxData::Candles { candles },
			params,
		}
	}
	#[inline]
	pub fn from_hlc_slices(high: &'a [f64], low: &'a [f64], close: &'a [f64], params: DxParams) -> Self {
		Self {
			data: DxData::HlcSlices { high, low, close },
			params,
		}
	}
	#[inline]
	pub fn with_default_candles(candles: &'a Candles) -> Self {
		Self {
			data: DxData::Candles { candles },
			params: DxParams::default(),
		}
	}
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(14)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct DxBuilder {
	period: Option<usize>,
	kernel: Kernel,
}

impl Default for DxBuilder {
	fn default() -> Self {
		Self {
			period: None,
			kernel: Kernel::Auto,
		}
	}
}

impl DxBuilder {
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
	pub fn apply(self, c: &Candles) -> Result<DxOutput, DxError> {
		let p = DxParams { period: self.period };
		let i = DxInput::from_candles(c, p);
		dx_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_hlc(self, high: &[f64], low: &[f64], close: &[f64]) -> Result<DxOutput, DxError> {
		let p = DxParams { period: self.period };
		let i = DxInput::from_hlc_slices(high, low, close, p);
		dx_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<DxStream, DxError> {
		let p = DxParams { period: self.period };
		DxStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum DxError {
	#[error("dx: Empty data provided for DX.")]
	EmptyData,
	#[error("dx: Could not select candle field: {0}")]
	SelectCandleFieldError(String),
	#[error("dx: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("dx: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("dx: All high, low, and close values are NaN.")]
	AllValuesNaN,
}

#[inline]
pub fn dx(input: &DxInput) -> Result<DxOutput, DxError> {
	dx_with_kernel(input, Kernel::Auto)
}

pub fn dx_with_kernel(input: &DxInput, kernel: Kernel) -> Result<DxOutput, DxError> {
	let (high, low, close) = match &input.data {
		DxData::Candles { candles } => {
			let high = candles
				.select_candle_field("high")
				.map_err(|e| DxError::SelectCandleFieldError(e.to_string()))?;
			let low = candles
				.select_candle_field("low")
				.map_err(|e| DxError::SelectCandleFieldError(e.to_string()))?;
			let close = candles
				.select_candle_field("close")
				.map_err(|e| DxError::SelectCandleFieldError(e.to_string()))?;
			(high, low, close)
		}
		DxData::HlcSlices { high, low, close } => (*high, *low, *close),
	};
	if high.is_empty() || low.is_empty() || close.is_empty() {
		return Err(DxError::EmptyData);
	}
	let len = high.len().min(low.len()).min(close.len());
	let period = input.get_period();
	if period == 0 || period > len {
		return Err(DxError::InvalidPeriod { period, data_len: len });
	}
	let first_valid_idx = (0..len).find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan());
	let first_valid_idx = match first_valid_idx {
		Some(idx) => idx,
		None => return Err(DxError::AllValuesNaN),
	};
	if (len - first_valid_idx) < period {
		return Err(DxError::NotEnoughValidData {
			needed: period,
			valid: len - first_valid_idx,
		});
	}
	let warmup_period = first_valid_idx + period - 1;
	let mut out = alloc_with_nan_prefix(len, warmup_period);

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => dx_scalar(high, low, close, period, first_valid_idx, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => dx_avx2(high, low, close, period, first_valid_idx, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => dx_avx512(high, low, close, period, first_valid_idx, &mut out),
			_ => unreachable!(),
		}
	}

	Ok(DxOutput { values: out })
}

// Scalar implementation (original logic preserved)
#[inline]
pub fn dx_scalar(high: &[f64], low: &[f64], close: &[f64], period: usize, first_valid_idx: usize, out: &mut [f64]) {
	let len = high.len().min(low.len()).min(close.len());
	let mut prev_high = high[first_valid_idx];
	let mut prev_low = low[first_valid_idx];
	let mut prev_close = close[first_valid_idx];
	let mut plus_dm_sum = 0.0;
	let mut minus_dm_sum = 0.0;
	let mut tr_sum = 0.0;
	let mut initial_count = 0;
	for i in (first_valid_idx + 1)..len {
		if high[i].is_nan() || low[i].is_nan() || close[i].is_nan() {
			out[i] = if i > 0 { out[i - 1] } else { f64::NAN };
			prev_high = high[i];
			prev_low = low[i];
			prev_close = close[i];
			continue;
		}
		let up_move = high[i] - prev_high;
		let down_move = prev_low - low[i];
		let mut plus_dm = 0.0;
		let mut minus_dm = 0.0;
		if up_move > 0.0 && up_move > down_move {
			plus_dm = up_move;
		} else if down_move > 0.0 && down_move > up_move {
			minus_dm = down_move;
		}
		let tr1 = high[i] - low[i];
		let tr2 = (high[i] - prev_close).abs();
		let tr3 = (low[i] - prev_close).abs();
		let tr = tr1.max(tr2).max(tr3);
		if initial_count < (period - 1) {
			plus_dm_sum += plus_dm;
			minus_dm_sum += minus_dm;
			tr_sum += tr;
			initial_count += 1;
			if initial_count == (period - 1) {
				let plus_di = (plus_dm_sum / tr_sum) * 100.0;
				let minus_di = (minus_dm_sum / tr_sum) * 100.0;
				let sum_di = plus_di + minus_di;
				out[i] = if sum_di != 0.0 {
					100.0 * ((plus_di - minus_di).abs() / sum_di)
				} else {
					0.0
				};
			}
		} else {
			plus_dm_sum = plus_dm_sum - (plus_dm_sum / period as f64) + plus_dm;
			minus_dm_sum = minus_dm_sum - (minus_dm_sum / period as f64) + minus_dm;
			tr_sum = tr_sum - (tr_sum / period as f64) + tr;
			let plus_di = if tr_sum != 0.0 {
				(plus_dm_sum / tr_sum) * 100.0
			} else {
				0.0
			};
			let minus_di = if tr_sum != 0.0 {
				(minus_dm_sum / tr_sum) * 100.0
			} else {
				0.0
			};
			let sum_di = plus_di + minus_di;
			out[i] = if sum_di != 0.0 {
				100.0 * ((plus_di - minus_di).abs() / sum_di)
			} else {
				out[i - 1]
			};
		}
		prev_high = high[i];
		prev_low = low[i];
		prev_close = close[i];
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn dx_avx512(high: &[f64], low: &[f64], close: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	if period <= 32 {
		unsafe { dx_avx512_short(high, low, close, period, first_valid, out) }
	} else {
		unsafe { dx_avx512_long(high, low, close, period, first_valid, out) }
	}
}

#[inline]
pub fn dx_avx2(high: &[f64], low: &[f64], close: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	// Stub points to scalar for API parity
	dx_scalar(high, low, close, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn dx_avx512_short(high: &[f64], low: &[f64], close: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	dx_scalar(high, low, close, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn dx_avx512_long(high: &[f64], low: &[f64], close: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	dx_scalar(high, low, close, period, first_valid, out)
}

// Stream implementation (emulates alma.rs streaming)
#[derive(Debug, Clone)]
pub struct DxStream {
	period: usize,
	plus_dm_sum: f64,
	minus_dm_sum: f64,
	tr_sum: f64,
	prev_high: f64,
	prev_low: f64,
	prev_close: f64,
	initial_count: usize,
	filled: bool,
}

impl DxStream {
	pub fn try_new(params: DxParams) -> Result<Self, DxError> {
		let period = params.period.unwrap_or(14);
		if period == 0 {
			return Err(DxError::InvalidPeriod { period, data_len: 0 });
		}
		Ok(Self {
			period,
			plus_dm_sum: 0.0,
			minus_dm_sum: 0.0,
			tr_sum: 0.0,
			prev_high: f64::NAN,
			prev_low: f64::NAN,
			prev_close: f64::NAN,
			initial_count: 0,
			filled: false,
		})
	}

	#[inline(always)]
	pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
		if self.prev_high.is_nan() || self.prev_low.is_nan() || self.prev_close.is_nan() {
			self.prev_high = high;
			self.prev_low = low;
			self.prev_close = close;
			return None;
		}
		let up_move = high - self.prev_high;
		let down_move = self.prev_low - low;
		let plus_dm = if up_move > 0.0 && up_move > down_move {
			up_move
		} else {
			0.0
		};
		let minus_dm = if down_move > 0.0 && down_move > up_move {
			down_move
		} else {
			0.0
		};
		let tr1 = high - low;
		let tr2 = (high - self.prev_close).abs();
		let tr3 = (low - self.prev_close).abs();
		let tr = tr1.max(tr2).max(tr3);

		if self.initial_count < (self.period - 1) {
			self.plus_dm_sum += plus_dm;
			self.minus_dm_sum += minus_dm;
			self.tr_sum += tr;
			self.initial_count += 1;
			if self.initial_count == (self.period - 1) {
				let plus_di = (self.plus_dm_sum / self.tr_sum) * 100.0;
				let minus_di = (self.minus_dm_sum / self.tr_sum) * 100.0;
				let sum_di = plus_di + minus_di;
				self.filled = true;
				self.prev_high = high;
				self.prev_low = low;
				self.prev_close = close;
				return Some(if sum_di != 0.0 {
					100.0 * ((plus_di - minus_di).abs() / sum_di)
				} else {
					0.0
				});
			} else {
				self.prev_high = high;
				self.prev_low = low;
				self.prev_close = close;
				return None;
			}
		} else {
			self.plus_dm_sum = self.plus_dm_sum - (self.plus_dm_sum / self.period as f64) + plus_dm;
			self.minus_dm_sum = self.minus_dm_sum - (self.minus_dm_sum / self.period as f64) + minus_dm;
			self.tr_sum = self.tr_sum - (self.tr_sum / self.period as f64) + tr;
			let plus_di = if self.tr_sum != 0.0 {
				(self.plus_dm_sum / self.tr_sum) * 100.0
			} else {
				0.0
			};
			let minus_di = if self.tr_sum != 0.0 {
				(self.minus_dm_sum / self.tr_sum) * 100.0
			} else {
				0.0
			};
			let sum_di = plus_di + minus_di;
			self.prev_high = high;
			self.prev_low = low;
			self.prev_close = close;
			return Some(if sum_di != 0.0 {
				100.0 * ((plus_di - minus_di).abs() / sum_di)
			} else {
				0.0
			});
		}
	}
}

// Batch/grid sweep types
#[derive(Clone, Debug)]
pub struct DxBatchRange {
	pub period: (usize, usize, usize),
}

impl Default for DxBatchRange {
	fn default() -> Self {
		Self { period: (14, 14, 0) }
	}
}

impl DxBatchRange {
	pub fn from_tuple(period: (usize, usize, usize)) -> Self {
		Self { period }
	}
}

#[derive(Clone, Debug, Default)]
pub struct DxBatchBuilder {
	range: DxBatchRange,
	kernel: Kernel,
}

impl DxBatchBuilder {
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

	pub fn apply_hlc(self, high: &[f64], low: &[f64], close: &[f64]) -> Result<DxBatchOutput, DxError> {
		dx_batch_with_kernel(high, low, close, &self.range, self.kernel)
	}

	pub fn apply_candles(self, c: &Candles) -> Result<DxBatchOutput, DxError> {
		let high = source_type(c, "high");
		let low = source_type(c, "low");
		let close = source_type(c, "close");
		self.apply_hlc(high, low, close)
	}
}

pub struct DxBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<DxParams>,
	pub rows: usize,
	pub cols: usize,
}

impl DxBatchOutput {
	pub fn row_for_params(&self, p: &DxParams) -> Option<usize> {
		self.combos
			.iter()
			.position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
	}
	pub fn values_for(&self, p: &DxParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &DxBatchRange) -> Vec<DxParams> {
	let (start, end, step) = r.period;
	if step == 0 || start == end {
		return vec![DxParams { period: Some(start) }];
	}
	(start..=end)
		.step_by(step)
		.map(|p| DxParams { period: Some(p) })
		.collect()
}

pub fn dx_batch_with_kernel(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	sweep: &DxBatchRange,
	k: Kernel,
) -> Result<DxBatchOutput, DxError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(DxError::InvalidPeriod { period: 0, data_len: 0 }),
	};

	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	dx_batch_par_slice(high, low, close, sweep, simd)
}

#[inline(always)]
pub fn dx_batch_slice(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	sweep: &DxBatchRange,
	kern: Kernel,
) -> Result<DxBatchOutput, DxError> {
	dx_batch_inner(high, low, close, sweep, kern, false)
}

#[inline(always)]
pub fn dx_batch_par_slice(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	sweep: &DxBatchRange,
	kern: Kernel,
) -> Result<DxBatchOutput, DxError> {
	dx_batch_inner(high, low, close, sweep, kern, true)
}

fn dx_batch_inner(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	sweep: &DxBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<DxBatchOutput, DxError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(DxError::InvalidPeriod { period: 0, data_len: 0 });
	}
	let len = high.len().min(low.len()).min(close.len());
	let first = (0..len)
		.find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
		.ok_or(DxError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if len - first < max_p {
		return Err(DxError::NotEnoughValidData {
			needed: max_p,
			valid: len - first,
		});
	}
	let rows = combos.len();
	let cols = len;
	
	// Use uninitialized memory with proper NaN prefixes
	let mut buf_mu = make_uninit_matrix(rows, cols);
	let warmup_periods: Vec<usize> = combos.iter()
		.map(|c| first + c.period.unwrap() - 1)
		.collect();
	init_matrix_prefixes(&mut buf_mu, cols, &warmup_periods);
	
	// Convert to mutable slice for computation
	let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
	let values: &mut [f64] = unsafe { 
		core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len()) 
	};
	
	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let period = combos[row].period.unwrap();
		match kern {
			Kernel::Scalar => dx_row_scalar(high, low, close, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => dx_row_avx2(high, low, close, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => dx_row_avx512(high, low, close, first, period, out_row),
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
	
	// Convert back to Vec from ManuallyDrop
	let values = unsafe {
		Vec::from_raw_parts(
			buf_guard.as_mut_ptr() as *mut f64,
			buf_guard.len(),
			buf_guard.capacity(),
		)
	};
	
	Ok(DxBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
unsafe fn dx_row_scalar(high: &[f64], low: &[f64], close: &[f64], first: usize, period: usize, out: &mut [f64]) {
	dx_scalar(high, low, close, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn dx_row_avx2(high: &[f64], low: &[f64], close: &[f64], first: usize, period: usize, out: &mut [f64]) {
	dx_scalar(high, low, close, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn dx_row_avx512(high: &[f64], low: &[f64], close: &[f64], first: usize, period: usize, out: &mut [f64]) {
	if period <= 32 {
		dx_row_avx512_short(high, low, close, first, period, out);
	} else {
		dx_row_avx512_long(high, low, close, first, period, out);
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn dx_row_avx512_short(high: &[f64], low: &[f64], close: &[f64], first: usize, period: usize, out: &mut [f64]) {
	dx_scalar(high, low, close, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn dx_row_avx512_long(high: &[f64], low: &[f64], close: &[f64], first: usize, period: usize, out: &mut [f64]) {
	dx_scalar(high, low, close, period, first, out)
}

#[inline(always)]
pub fn expand_grid_dx(r: &DxBatchRange) -> Vec<DxParams> {
	expand_grid(r)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_dx_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let default_params = DxParams { period: None };
		let input = DxInput::from_candles(&candles, default_params);
		let output = dx_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_dx_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let input = DxInput::from_candles(&candles, DxParams::default());
		let result = dx_with_kernel(&input, kernel)?;
		let expected_last_five = [
			43.72121533411883,
			41.47251493226443,
			43.43041386436222,
			43.22673458811955,
			51.65514026197179,
		];
		let start = result.values.len().saturating_sub(5);
		for (i, &val) in result.values[start..].iter().enumerate() {
			let diff = (val - expected_last_five[i]).abs();
			assert!(
				diff < 1e-4,
				"[{}] DX {:?} mismatch at idx {}: got {}, expected {}",
				test_name,
				kernel,
				i,
				val,
				expected_last_five[i]
			);
		}
		Ok(())
	}

	fn check_dx_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let input = DxInput::with_default_candles(&candles);
		match input.data {
			DxData::Candles { .. } => {}
			_ => panic!("Expected DxData::Candles"),
		}
		let output = dx_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_dx_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [2.0, 2.5, 3.0];
		let low = [1.0, 1.2, 2.1];
		let close = [1.5, 2.3, 2.2];
		let params = DxParams { period: Some(0) };
		let input = DxInput::from_hlc_slices(&high, &low, &close, params);
		let res = dx_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] DX should fail with zero period", test_name);
		Ok(())
	}

	fn check_dx_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [3.0, 4.0];
		let low = [2.0, 3.0];
		let close = [2.5, 3.5];
		let params = DxParams { period: Some(14) };
		let input = DxInput::from_hlc_slices(&high, &low, &close, params);
		let res = dx_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] DX should fail with period exceeding length",
			test_name
		);
		Ok(())
	}

	fn check_dx_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [3.0];
		let low = [2.0];
		let close = [2.5];
		let params = DxParams { period: Some(14) };
		let input = DxInput::from_hlc_slices(&high, &low, &close, params);
		let res = dx_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] DX should fail with insufficient data", test_name);
		Ok(())
	}

	fn check_dx_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let first_params = DxParams { period: Some(14) };
		let first_input = DxInput::from_candles(&candles, first_params);
		let first_result = dx_with_kernel(&first_input, kernel)?;

		let second_params = DxParams { period: Some(14) };
		let second_input = DxInput::from_hlc_slices(
			&first_result.values,
			&first_result.values,
			&first_result.values,
			second_params,
		);
		let second_result = dx_with_kernel(&second_input, kernel)?;
		assert_eq!(second_result.values.len(), first_result.values.len());
		for i in 28..second_result.values.len() {
			assert!(
				!second_result.values[i].is_nan(),
				"[{}] Expected no NaN after index 28, found NaN at idx {}",
				test_name,
				i
			);
		}
		Ok(())
	}

	fn check_dx_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = DxInput::from_candles(&candles, DxParams { period: Some(14) });
		let res = dx_with_kernel(&input, kernel)?;
		assert_eq!(res.values.len(), candles.close.len());
		if res.values.len() > 50 {
			for (i, &val) in res.values[50..].iter().enumerate() {
				assert!(
					!val.is_nan(),
					"[{}] Found unexpected NaN at out-index {}",
					test_name,
					50 + i
				);
			}
		}
		Ok(())
	}

	fn check_dx_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let high = source_type(&candles, "high");
		let low = source_type(&candles, "low");
		let close = source_type(&candles, "close");
		let period = 14;

		let input = DxInput::from_candles(&candles, DxParams { period: Some(period) });
		let batch_output = dx_with_kernel(&input, kernel)?.values;

		let mut stream = DxStream::try_new(DxParams { period: Some(period) })?;
		let mut stream_values = Vec::with_capacity(candles.close.len());
		for ((&h, &l), &c) in high.iter().zip(low).zip(close) {
			match stream.update(h, l, c) {
				Some(dx_val) => stream_values.push(dx_val),
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
				"[{}] DX streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
				test_name,
				i,
				b,
				s,
				diff
			);
		}
		Ok(())
	}

	macro_rules! generate_all_dx_tests {
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
	#[cfg(debug_assertions)]
	fn check_dx_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		// Define comprehensive parameter combinations
		let test_params = vec![
			// Default parameters
			DxParams::default(),
			// Minimum period
			DxParams { period: Some(2) },
			// Small periods
			DxParams { period: Some(5) },
			DxParams { period: Some(7) },
			// Medium periods
			DxParams { period: Some(10) },
			DxParams { period: Some(14) },  // Default value
			DxParams { period: Some(20) },
			// Large periods
			DxParams { period: Some(30) },
			DxParams { period: Some(50) },
			// Very large periods
			DxParams { period: Some(100) },
			DxParams { period: Some(200) },
		];

		for (param_idx, params) in test_params.iter().enumerate() {
			let input = DxInput::from_candles(&candles, params.clone());
			let output = dx_with_kernel(&input, kernel)?;

			for (i, &val) in output.values.iter().enumerate() {
				if val.is_nan() {
					continue; // NaN values are expected during warmup
				}

				let bits = val.to_bits();

				// Check all three poison patterns
				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
						test_name,
						val,
						bits,
						i,
						params.period.unwrap_or(14),
						param_idx
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
						test_name,
						val,
						bits,
						i,
						params.period.unwrap_or(14),
						param_idx
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
						test_name,
						val,
						bits,
						i,
						params.period.unwrap_or(14),
						param_idx
					);
				}
			}
		}

		Ok(())
	}

	#[cfg(not(debug_assertions))]
	fn check_dx_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(()) // No-op in release builds
	}

	#[cfg(test)]
	#[allow(clippy::float_cmp)]
	fn check_dx_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		use proptest::prelude::*;
		skip_if_unsupported!(kernel, test_name);

		// Strategy generates realistic OHLC data
		let strat = (2usize..=50)  // DX requires minimum period of 2
			.prop_flat_map(|period| {
				(
					100.0f64..5000.0f64,  // Base price
					(period + 20)..400,    // Data length
					0.001f64..0.05f64,     // Volatility factor
					-0.01f64..0.01f64,     // Trend factor
					Just(period),
				)
			})
			.prop_map(|(base_price, data_len, volatility, trend, period)| {
				// Generate synthetic OHLC data
				let mut high = Vec::with_capacity(data_len);
				let mut low = Vec::with_capacity(data_len);
				let mut close = Vec::with_capacity(data_len);
				
				let mut price = base_price;
				for i in 0..data_len {
					// Add trend and random volatility
					let trend_component = trend * i as f64;
					let random_component = ((i * 7 + 13) % 17) as f64 / 17.0 - 0.5;
					price = base_price + trend_component + random_component * volatility * base_price;
					
					// Generate OHLC with realistic constraints
					let daily_volatility = volatility * price;
					let h = price + daily_volatility * (0.5 + ((i * 3) % 7) as f64 / 14.0);
					let l = price - daily_volatility * (0.5 + ((i * 5) % 7) as f64 / 14.0);
					let c = l + (h - l) * (0.3 + ((i * 11) % 7) as f64 / 10.0);
					
					high.push(h);
					low.push(l);
					close.push(c);
				}
				
				(high, low, close, period)
			});

		proptest::test_runner::TestRunner::default()
			.run(&strat, |(high, low, close, period)| {
				let params = DxParams { period: Some(period) };
				let input = DxInput::from_hlc_slices(&high, &low, &close, params.clone());

				let DxOutput { values: out } = dx_with_kernel(&input, kernel).unwrap();
				let DxOutput { values: ref_out } = dx_with_kernel(&input, Kernel::Scalar).unwrap();

				// Property 1: DX values should be between 0 and 100
				for (i, &val) in out.iter().enumerate() {
					if !val.is_nan() {
						prop_assert!(
							val >= -1e-9 && val <= 100.0 + 1e-9,
							"[{}] DX value {} at index {} is outside [0, 100] range",
							test_name, val, i
						);
					}
				}

				// Property 2: Warmup period should be respected
				// Note: Since synthetic data has no NaN values, first_valid_idx = 0
				// Therefore warmup = 0 + period - 1 = period - 1
				let warmup = period - 1;
				for i in 0..warmup {
					prop_assert!(
						out[i].is_nan(),
						"[{}] Expected NaN during warmup at index {}, got {}",
						test_name, i, out[i]
					);
				}

				// Property 3: After warmup, values should not be NaN (unless input has NaN)
				if out.len() > warmup + 10 {
					for i in (warmup + 10)..out.len() {
						prop_assert!(
							!out[i].is_nan(),
							"[{}] Unexpected NaN after warmup at index {}",
							test_name, i
						);
					}
				}

				// Property 4: Kernel consistency - all kernels should produce same results
				for (i, (&val, &ref_val)) in out.iter().zip(ref_out.iter()).enumerate() {
					if val.is_nan() && ref_val.is_nan() {
						continue;
					}
					
					let diff = (val - ref_val).abs();
					prop_assert!(
						diff < 1e-9,
						"[{}] Kernel mismatch at index {}: {} vs {} (diff: {})",
						test_name, i, val, ref_val, diff
					);
				}

				// Property 5: Special case - constant prices
				let all_same_high = high.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10);
				let all_same_low = low.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10);
				let all_same_close = close.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10);
				
				if all_same_high && all_same_low && all_same_close {
					// With no directional movement, DX should be near 0
					if out.len() > warmup + 10 {
						let stable_vals = &out[warmup + 10..];
						for (i, &val) in stable_vals.iter().enumerate() {
							if !val.is_nan() {
								prop_assert!(
									val < 1.0,  // Tightened threshold - no movement means DX near 0
									"[{}] With constant prices, expected DX < 1.0, got {} at index {}",
									test_name, val, warmup + 10 + i
								);
							}
						}
					}
				}

				// Property 6: Improved trend detection (for longer series)
				if period <= 20 && out.len() > 100 {
					// Calculate actual trend in the data
					let mid = out.len() / 2;
					let first_half_avg_price = close[..mid].iter().sum::<f64>() / mid as f64;
					let second_half_avg_price = close[mid..].iter().sum::<f64>() / (out.len() - mid) as f64;
					let price_change = ((second_half_avg_price - first_half_avg_price) / first_half_avg_price).abs();
					
					// For significant price changes, DX should reflect trend strength
					if price_change > 0.05 {
						// Compare average DX in second half vs first half
						let first_half_dx = &out[warmup..mid];
						let second_half_dx = &out[mid..];
						
						let first_avg = first_half_dx.iter()
							.filter(|v| !v.is_nan())
							.sum::<f64>() / first_half_dx.len() as f64;
						let second_avg = second_half_dx.iter()
							.filter(|v| !v.is_nan())
							.sum::<f64>() / second_half_dx.len() as f64;
						
						// In trending markets, average DX should be meaningful (> 20)
						prop_assert!(
							second_avg > 20.0 || first_avg > 20.0,
							"[{}] Expected higher average DX in trending market. First half avg: {}, Second half avg: {}",
							test_name, first_avg, second_avg
						);
					}
				}

				// Property 7: Perfect trend test
				// Generate a perfect uptrend or downtrend and verify high DX
				if period <= 14 && out.len() > 50 {
					// Use the first close price as base for perfect trend
					let trend_base = close[0];
					let perfect_trend = (0..50)
						.map(|i| {
							let price = trend_base + (i as f64 * trend_base * 0.01); // 1% per period
							let h = price * 1.005;  // Small high-low range
							let l = price * 0.995;
							let c = price;
							(h, l, c)
						})
						.collect::<Vec<_>>();
					
					let perfect_high: Vec<f64> = perfect_trend.iter().map(|&(h, _, _)| h).collect();
					let perfect_low: Vec<f64> = perfect_trend.iter().map(|&(_, l, _)| l).collect();
					let perfect_close: Vec<f64> = perfect_trend.iter().map(|&(_, _, c)| c).collect();
					
					let perfect_input = DxInput::from_hlc_slices(&perfect_high, &perfect_low, &perfect_close, params.clone());
					let DxOutput { values: perfect_out } = dx_with_kernel(&perfect_input, kernel).unwrap();
					
					// In a perfect trend, DX should be high after stabilization
					if perfect_out.len() > warmup + 10 {
						let stable_dx = &perfect_out[warmup + 10..];
						let avg_dx = stable_dx.iter()
							.filter(|v| !v.is_nan())
							.sum::<f64>() / stable_dx.len() as f64;
						
						prop_assert!(
							avg_dx > 50.0,  // Strong trend should show DX > 50
							"[{}] Expected high DX (>50) in perfect trend, got avg {}",
							test_name, avg_dx
						);
					}
				}

				// Property 8: Ranging market test
				// Generate oscillating prices and verify low DX
				if period <= 14 && out.len() > 50 {
					// Use the first close price as base for ranging market
					let range_base = close[0];
					let ranging_data = (0..50)
						.map(|i| {
							// Oscillate between two price levels
							let price = if i % 4 < 2 {
								range_base * 1.01
							} else {
								range_base * 0.99
							};
							let h = price * 1.002;
							let l = price * 0.998;
							let c = price;
							(h, l, c)
						})
						.collect::<Vec<_>>();
					
					let ranging_high: Vec<f64> = ranging_data.iter().map(|&(h, _, _)| h).collect();
					let ranging_low: Vec<f64> = ranging_data.iter().map(|&(_, l, _)| l).collect();
					let ranging_close: Vec<f64> = ranging_data.iter().map(|&(_, _, c)| c).collect();
					
					let ranging_input = DxInput::from_hlc_slices(&ranging_high, &ranging_low, &ranging_close, params.clone());
					let DxOutput { values: ranging_out } = dx_with_kernel(&ranging_input, kernel).unwrap();
					
					// In a ranging market, DX should be low after stabilization
					if ranging_out.len() > warmup + 10 {
						let stable_dx = &ranging_out[warmup + 10..];
						let avg_dx = stable_dx.iter()
							.filter(|v| !v.is_nan())
							.sum::<f64>() / stable_dx.len() as f64;
						
						// Note: Even small oscillations can produce moderate DX values
						// since DX measures absolute directional movement
						prop_assert!(
							avg_dx < 65.0,  // More realistic threshold for oscillating markets
							"[{}] Expected moderate DX (<65) in ranging market, got avg {}",
							test_name, avg_dx
						);
					}
				}

				Ok(())
			})
			.unwrap();

		Ok(())
	}

	generate_all_dx_tests!(
		check_dx_partial_params,
		check_dx_accuracy,
		check_dx_default_candles,
		check_dx_zero_period,
		check_dx_period_exceeds_length,
		check_dx_very_small_dataset,
		check_dx_reinput,
		check_dx_nan_handling,
		check_dx_streaming,
		check_dx_no_poison
	);
	
	#[cfg(test)]
	generate_all_dx_tests!(check_dx_property);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let output = DxBatchBuilder::new().kernel(kernel).apply_candles(&c)?;

		let def = DxParams::default();
		let row = output.values_for(&def).expect("default row missing");

		assert_eq!(row.len(), c.close.len());

		let expected = [
			43.72121533411883,
			41.47251493226443,
			43.43041386436222,
			43.22673458811955,
			51.65514026197179,
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
	fn check_batch_sweep(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let output = DxBatchBuilder::new()
			.kernel(kernel)
			.period_range(10, 30, 5)
			.apply_candles(&c)?;

		let expected_combos = 5; // periods: 10, 15, 20, 25, 30
		assert_eq!(output.combos.len(), expected_combos);
		assert_eq!(output.rows, expected_combos);
		assert_eq!(output.cols, c.close.len());

		Ok(())
	}

	#[cfg(debug_assertions)]
	fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		// Test various parameter sweep configurations
		let test_configs = vec![
			// (period_start, period_end, period_step)
			(2, 10, 2),      // Small periods
			(5, 25, 5),      // Medium periods
			(30, 60, 15),    // Large periods
			(2, 5, 1),       // Dense small range
			(10, 20, 2),     // Medium range with small steps
			(14, 14, 0),     // Single period (default)
			(5, 50, 15),     // Wide range
			(100, 200, 50),  // Very large periods
		];

		for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
			let output = DxBatchBuilder::new()
				.kernel(kernel)
				.period_range(p_start, p_end, p_step)
				.apply_candles(&c)?;

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
		}

		Ok(())
	}

	#[cfg(not(debug_assertions))]
	fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(())
	}

	gen_batch_tests!(check_batch_default_row);
	gen_batch_tests!(check_batch_sweep);
	gen_batch_tests!(check_batch_no_poison);
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "dx", signature = (high, low, close, period, kernel=None))]
pub fn dx_py<'py>(
	py: Python<'py>,
	high: PyReadonlyArray1<f64>,
	low: PyReadonlyArray1<f64>,
	close: PyReadonlyArray1<f64>,
	period: usize,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
	let high = high.as_slice()?;
	let low = low.as_slice()?;
	let close = close.as_slice()?;
	
	// Validate kernel parameter if provided
	let kernel_enum = validate_kernel(kernel, false)?;
	
	py.allow_threads(|| {
		let params = DxParams { period: Some(period) };
		let input = DxInput::from_hlc_slices(high, low, close, params);
		let result = dx_with_kernel(&input, kernel_enum)
			.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
		Ok(result.values)
	})
	.and_then(|values| Ok(values.into_pyarray(py)))
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "dx_batch", signature = (high, low, close, period_range))]
pub fn dx_batch_py<'py>(
	py: Python<'py>,
	high: PyReadonlyArray1<f64>,
	low: PyReadonlyArray1<f64>,
	close: PyReadonlyArray1<f64>,
	period_range: (usize, usize, usize),
) -> PyResult<Bound<'py, PyDict>> {
	let high_slice = high.as_slice()?;
	let low_slice = low.as_slice()?;
	let close_slice = close.as_slice()?;
	
	// Calculate parameter combinations
	let (start, end, step) = period_range;
	let params = expand_grid(&DxBatchRange { period: (start, end, step) });
	let rows = params.len();
	let cols = high_slice.len();
	
	// Pre-allocate output array for zero-copy
	let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let slice_out = unsafe { out_arr.as_slice_mut()? };
	
	let combos = py.allow_threads(|| {
		// Call the optimized inner function
		dx_batch_inner_into(high_slice, low_slice, close_slice, &params, slice_out)
			.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
		
		Ok::<Vec<DxParams>, pyo3::PyErr>(params)
	})?;
	
	let dict = PyDict::new(py);
	dict.set_item("values", out_arr.reshape((rows, cols))?)?;
	dict.set_item("rows", rows)?;
	dict.set_item("cols", cols)?;
	
	// Create parameter list
	let py_params = PyList::new(py, combos.iter().map(|p| {
		let param_dict = PyDict::new(py);
		param_dict.set_item("period", p.period.unwrap_or(14)).unwrap();
		param_dict
	}))?;
	dict.set_item("params", py_params)?;
	
	Ok(dict.into())
}

// Helper function for batch processing that writes directly to pre-allocated buffer
#[cfg(feature = "python")]
fn dx_batch_inner_into(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	params: &[DxParams],
	out: &mut [f64],
) -> Result<(), DxError> {
	let len = high.len().min(low.len()).min(close.len());
	let kernel = detect_best_kernel();
	
	// Find first valid index
	let first_valid_idx = (0..len)
		.find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
		.ok_or(DxError::AllValuesNaN)?;
	
	// Process each parameter combination
	for (row, p) in params.iter().enumerate() {
		let period = p.period.unwrap_or(14);
		
		// Validate period
		if period == 0 || period > len {
			return Err(DxError::InvalidPeriod { period, data_len: len });
		}
		
		if (len - first_valid_idx) < period {
			return Err(DxError::NotEnoughValidData {
				needed: period,
				valid: len - first_valid_idx,
			});
		}
		
		// Get the output slice for this row
		let start_idx = row * len;
		let out_row = &mut out[start_idx..start_idx + len];
		
		// Call the appropriate kernel
		unsafe {
			match kernel {
				Kernel::Scalar | Kernel::ScalarBatch => {
					dx_scalar(high, low, close, period, first_valid_idx, out_row);
				}
				#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
				Kernel::Avx2 | Kernel::Avx2Batch => {
					dx_avx2(high, low, close, period, first_valid_idx, out_row);
				}
				#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
				Kernel::Avx512 | Kernel::Avx512Batch => {
					dx_avx512(high, low, close, period, first_valid_idx, out_row);
				}
				_ => unreachable!(),
			}
		}
	}
	
	Ok(())
}

#[cfg(feature = "python")]
#[pyclass(name = "DxStream")]
pub struct DxStreamPy {
	inner: DxStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl DxStreamPy {
	#[new]
	pub fn new(period: usize) -> PyResult<Self> {
		let params = DxParams { period: Some(period) };
		let inner = DxStream::try_new(params)
			.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
		Ok(Self { inner })
	}
	
	pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
		self.inner.update(high, low, close)
	}
}

// WASM bindings

/// Write directly to output slice - no allocations
#[inline]
pub fn dx_into_slice(dst: &mut [f64], input: &DxInput, kern: Kernel) -> Result<(), DxError> {
	let (high, low, close) = match &input.data {
		DxData::Candles { candles } => {
			let high = candles
				.select_candle_field("high")
				.map_err(|e| DxError::SelectCandleFieldError(e.to_string()))?;
			let low = candles
				.select_candle_field("low")
				.map_err(|e| DxError::SelectCandleFieldError(e.to_string()))?;
			let close = candles
				.select_candle_field("close")
				.map_err(|e| DxError::SelectCandleFieldError(e.to_string()))?;
			(high, low, close)
		}
		DxData::HlcSlices { high, low, close } => (*high, *low, *close),
	};
	
	let period = input.params.period.unwrap_or(14);
	let len = high.len();
	
	// Validate inputs
	if len == 0 || low.len() == 0 || close.len() == 0 {
		return Err(DxError::EmptyData);
	}
	
	if high.len() != low.len() || high.len() != close.len() {
		return Err(DxError::EmptyData);
	}
	
	if dst.len() != len {
		return Err(DxError::InvalidPeriod {
			period: dst.len(),
			data_len: len,
		});
	}
	
	if period == 0 || period > len {
		return Err(DxError::InvalidPeriod { period, data_len: len });
	}
	
	let first_valid_idx = high
		.iter()
		.zip(low)
		.zip(close)
		.position(|((&h, &l), &c)| !h.is_nan() && !l.is_nan() && !c.is_nan())
		.ok_or(DxError::AllValuesNaN)?;
		
	let total_required = first_valid_idx + period;
	if total_required > len {
		return Err(DxError::NotEnoughValidData {
			needed: total_required,
			valid: len - first_valid_idx,
		});
	}
	
	let warmup_period = first_valid_idx + period - 1;
	
	// Select kernel based on whether batch operations are requested
	let chosen = match kern {
		Kernel::Auto => detect_best_kernel(),
		_ => kern,
	};
	
	// Compute DX values
	match chosen {
		#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
		Kernel::Avx512 => dx_avx512(high, low, close, period, first_valid_idx, dst),
		#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
		Kernel::Avx2 => dx_avx2(high, low, close, period, first_valid_idx, dst),
		_ => dx_scalar(high, low, close, period, first_valid_idx, dst),
	}
	
	// Fill warmup with NaN
	for v in &mut dst[..warmup_period] {
		*v = f64::NAN;
	}
	
	Ok(())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn dx_js(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
	let params = DxParams { period: Some(period) };
	let input = DxInput::from_hlc_slices(high, low, close, params);
	
	let mut output = vec![0.0; high.len()];  // Single allocation
	dx_into_slice(&mut output, &input, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
		
	Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn dx_into(
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
		let params = DxParams { period: Some(period) };
		let input = DxInput::from_hlc_slices(high, low, close, params);
		
		// CRITICAL: Check aliasing for ALL input pointers
		if high_ptr == out_ptr || low_ptr == out_ptr || close_ptr == out_ptr {
			// Aliasing detected - use temporary buffer
			let mut temp = vec![0.0; len];
			dx_into_slice(&mut temp, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			out.copy_from_slice(&temp);
		} else {
			// No aliasing - write directly
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			dx_into_slice(out, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn dx_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn dx_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe { let _ = Vec::from_raw_parts(ptr, len, len); }
	}
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct DxBatchConfig {
	pub period_range: (usize, usize, usize), // (start, end, step)
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct DxBatchJsOutput {
	pub values: Vec<f64>,
	pub combos: Vec<DxParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = dx_batch)]
pub fn dx_batch_js(high: &[f64], low: &[f64], close: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	let config: DxBatchConfig =
		serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
		
	let batch_range = DxBatchRange::from_tuple(config.period_range);
	let result = dx_batch_with_kernel(high, low, close, &batch_range, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	// Generate combos for output
	let combos = DxParams::generate_batch_params(config.period_range);
		
	let output = DxBatchJsOutput {
		values: result.values,
		combos,
		rows: result.rows,
		cols: result.cols,
	};
	
	serde_wasm_bindgen::to_value(&output)
		.map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn dx_batch_into(
	high_ptr: *const f64,
	low_ptr: *const f64,
	close_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period_start: usize,
	period_end: usize,
	period_step: usize,
) -> Result<(), JsValue> {
	if high_ptr.is_null() || low_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}
	
	unsafe {
		let high = std::slice::from_raw_parts(high_ptr, len);
		let low = std::slice::from_raw_parts(low_ptr, len);
		let close = std::slice::from_raw_parts(close_ptr, len);
		let batch_range = DxBatchRange::from_tuple((period_start, period_end, period_step));
		let combos = DxParams::generate_batch_params((period_start, period_end, period_step));
		let n_combos = combos.len();
		
		if high_ptr == out_ptr || low_ptr == out_ptr || close_ptr == out_ptr {
			// Aliasing detected - use temporary buffer
			let result = dx_batch_with_kernel(high, low, close, &batch_range, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			let out = std::slice::from_raw_parts_mut(out_ptr, len * n_combos);
			out.copy_from_slice(&result.values);
		} else {
			// No aliasing - write directly
			// Use the Python-style batch_inner_into function for direct writing
			let params = combos;
			let out = std::slice::from_raw_parts_mut(out_ptr, len * n_combos);
			
			// Write directly into the output buffer using the pattern from Python bindings
			let first = high.iter().zip(low).zip(close)
				.position(|((&h, &l), &c)| !h.is_nan() && !l.is_nan() && !c.is_nan())
				.ok_or_else(|| JsValue::from_str("All values are NaN"))?;
			
			let mut buf_uninit = make_uninit_matrix(params.len(), len);
			let warmup_periods: Vec<usize> = params.iter()
				.map(|p| first + p.period.unwrap() - 1)
				.collect();
			init_matrix_prefixes(&mut buf_uninit, len, &warmup_periods);
			
			// Convert to initialized slice
			let buf_ptr = buf_uninit.as_mut_ptr() as *mut f64;
			std::mem::forget(buf_uninit);
			let slice_out = std::slice::from_raw_parts_mut(buf_ptr, params.len() * len);
			
			// Process each parameter combination
			for (i, param) in params.iter().enumerate() {
				let row_offset = i * len;
				let row = &mut slice_out[row_offset..row_offset + len];
				
				let warmup = first + param.period.unwrap() - 1;
				dx_scalar(high, low, close, param.period.unwrap(), first, row);
				
				// NaN prefix already set by init_matrix_prefixes
			}
			
			// Copy to output
			out.copy_from_slice(slice_out);
		}
		Ok(())
	}
}
