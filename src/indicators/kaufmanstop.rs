//! # Kaufmanstop Indicator
//!
//! Adaptive price stop based on the average true range of price (high-low).
//! Supports batch parameter sweeping and optional AVX acceleration.
//!
//! ## Parameters
//! - **period**: Window size for range average (default: 22)
//! - **mult**: Multiplier for averaged range (default: 2.0)
//! - **direction**: "long" (stop below price) or "short" (above) (default: "long")
//! - **ma_type**: Type of moving average for range ("sma", "ema", etc.; default: "sma")
//!
//! ## Errors
//! - **EmptyData**: All relevant slices empty
//! - **InvalidPeriod**: Zero/too large period
//! - **NotEnoughValidData**: Not enough non-NaN data
//! - **AllValuesNaN**: All inputs NaN
//!
//! ## Returns
//! - **Ok(KaufmanstopOutput)**: Output vector length matches input, leading NaNs where window not filled
//! - **Err(KaufmanstopError)**: Error on failure
use crate::indicators::moving_averages::ma::{ma, MaData};
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

#[derive(Debug, Clone)]
pub enum KaufmanstopData<'a> {
	Candles { candles: &'a Candles },
	Slices { high: &'a [f64], low: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct KaufmanstopOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct KaufmanstopParams {
	pub period: Option<usize>,
	pub mult: Option<f64>,
	pub direction: Option<String>,
	pub ma_type: Option<String>,
}

impl Default for KaufmanstopParams {
	fn default() -> Self {
		Self {
			period: Some(22),
			mult: Some(2.0),
			direction: Some("long".to_string()),
			ma_type: Some("sma".to_string()),
		}
	}
}

#[derive(Debug, Clone)]
pub struct KaufmanstopInput<'a> {
	pub data: KaufmanstopData<'a>,
	pub params: KaufmanstopParams,
}

impl<'a> KaufmanstopInput<'a> {
	#[inline]
	pub fn from_candles(candles: &'a Candles, params: KaufmanstopParams) -> Self {
		Self {
			data: KaufmanstopData::Candles { candles },
			params,
		}
	}
	#[inline]
	pub fn from_slices(high: &'a [f64], low: &'a [f64], params: KaufmanstopParams) -> Self {
		Self {
			data: KaufmanstopData::Slices { high, low },
			params,
		}
	}
	#[inline]
	pub fn with_default_candles(candles: &'a Candles) -> Self {
		Self::from_candles(candles, KaufmanstopParams::default())
	}
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(22)
	}
	#[inline]
	pub fn get_mult(&self) -> f64 {
		self.params.mult.unwrap_or(2.0)
	}
	#[inline]
	pub fn get_direction(&self) -> &str {
		self.params.direction.as_deref().unwrap_or("long")
	}
	#[inline]
	pub fn get_ma_type(&self) -> &str {
		self.params.ma_type.as_deref().unwrap_or("sma")
	}
}

impl<'a> AsRef<[f64]> for KaufmanstopInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		// Since kaufmanstop uses two data series (high and low), 
		// we'll return the high series as the primary reference
		match &self.data {
			KaufmanstopData::Candles { candles } => {
				candles.select_candle_field("high").unwrap_or(&[])
			}
			KaufmanstopData::Slices { high, .. } => high,
		}
	}
}

#[derive(Clone, Debug)]
pub struct KaufmanstopBuilder {
	period: Option<usize>,
	mult: Option<f64>,
	direction: Option<String>,
	ma_type: Option<String>,
	kernel: Kernel,
}

impl Default for KaufmanstopBuilder {
	fn default() -> Self {
		Self {
			period: None,
			mult: None,
			direction: None,
			ma_type: None,
			kernel: Kernel::Auto,
		}
	}
}

impl KaufmanstopBuilder {
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
	pub fn ma_type<S: Into<String>>(mut self, m: S) -> Self {
		self.ma_type = Some(m.into());
		self
	}
	#[inline(always)]
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}

	#[inline(always)]
	pub fn apply(self, c: &Candles) -> Result<KaufmanstopOutput, KaufmanstopError> {
		let p = KaufmanstopParams {
			period: self.period,
			mult: self.mult,
			direction: self.direction,
			ma_type: self.ma_type,
		};
		let i = KaufmanstopInput::from_candles(c, p);
		kaufmanstop_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<KaufmanstopOutput, KaufmanstopError> {
		let p = KaufmanstopParams {
			period: self.period,
			mult: self.mult,
			direction: self.direction,
			ma_type: self.ma_type,
		};
		let i = KaufmanstopInput::from_slices(high, low, p);
		kaufmanstop_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<KaufmanstopStream, KaufmanstopError> {
		let p = KaufmanstopParams {
			period: self.period,
			mult: self.mult,
			direction: self.direction,
			ma_type: self.ma_type,
		};
		KaufmanstopStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum KaufmanstopError {
	#[error("kaufmanstop: Empty data provided.")]
	EmptyData,
	#[error("kaufmanstop: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("kaufmanstop: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("kaufmanstop: All values are NaN.")]
	AllValuesNaN,
}

#[cfg(feature = "wasm")]
impl From<KaufmanstopError> for JsValue {
	fn from(err: KaufmanstopError) -> Self {
		JsValue::from_str(&err.to_string())
	}
}

#[inline]
pub fn kaufmanstop(input: &KaufmanstopInput) -> Result<KaufmanstopOutput, KaufmanstopError> {
	kaufmanstop_with_kernel(input, Kernel::Auto)
}

pub fn kaufmanstop_with_kernel(
	input: &KaufmanstopInput,
	kernel: Kernel,
) -> Result<KaufmanstopOutput, KaufmanstopError> {
	let (high, low) = match &input.data {
		KaufmanstopData::Candles { candles } => {
			let high = candles
				.select_candle_field("high")
				.map_err(|_| KaufmanstopError::EmptyData)?;
			let low = candles
				.select_candle_field("low")
				.map_err(|_| KaufmanstopError::EmptyData)?;
			(high, low)
		}
		KaufmanstopData::Slices { high, low } => {
			if high.is_empty() || low.is_empty() {
				return Err(KaufmanstopError::EmptyData);
			}
			(*high, *low)
		}
	};

	if high.is_empty() || low.is_empty() {
		return Err(KaufmanstopError::EmptyData);
	}

	let period = input.get_period();
	let mult = input.get_mult();
	let direction = input.get_direction();
	let ma_type = input.get_ma_type();

	if period == 0 || period > high.len() || period > low.len() {
		return Err(KaufmanstopError::InvalidPeriod {
			period,
			data_len: high.len().min(low.len()),
		});
	}

	let first_valid_idx = high
		.iter()
		.zip(low.iter())
		.position(|(&h, &l)| !h.is_nan() && !l.is_nan())
		.ok_or(KaufmanstopError::AllValuesNaN)?;

	if (high.len() - first_valid_idx) < period {
		return Err(KaufmanstopError::NotEnoughValidData {
			needed: period,
			valid: high.len() - first_valid_idx,
		});
	}

	// Use helper function to allocate with NaN prefix
	let mut hl_diff = alloc_with_nan_prefix(high.len(), first_valid_idx);
	for i in first_valid_idx..high.len() {
		if high[i].is_nan() || low[i].is_nan() {
			hl_diff[i] = f64::NAN;
		} else {
			hl_diff[i] = high[i] - low[i];
		}
	}

	let ma_input = MaData::Slice(&hl_diff[first_valid_idx..]);
	let hl_diff_ma = ma(ma_type, ma_input, period).map_err(|_| KaufmanstopError::AllValuesNaN)?;

	// Use alloc_with_nan_prefix instead of vec![f64::NAN; ...]
	let mut out = alloc_with_nan_prefix(high.len(), first_valid_idx + period - 1);

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => kaufmanstop_scalar(
				high,
				low,
				&hl_diff_ma,
				period,
				first_valid_idx,
				mult,
				direction,
				&mut out,
			),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => kaufmanstop_avx2(
				high,
				low,
				&hl_diff_ma,
				period,
				first_valid_idx,
				mult,
				direction,
				&mut out,
			),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => kaufmanstop_avx512(
				high,
				low,
				&hl_diff_ma,
				period,
				first_valid_idx,
				mult,
				direction,
				&mut out,
			),
			_ => unreachable!(),
		}
	}
	Ok(KaufmanstopOutput { values: out })
}

#[inline]
pub fn kaufmanstop_scalar(
	high: &[f64],
	low: &[f64],
	range_ma: &[f64],
	period: usize,
	first: usize,
	mult: f64,
	direction: &str,
	out: &mut [f64],
) {
	for (i, &val) in range_ma.iter().enumerate() {
		let idx = first + i;
		if idx < high.len() {
			if direction.eq_ignore_ascii_case("long") {
				out[idx] = low[idx] - val * mult;
			} else {
				out[idx] = high[idx] + val * mult;
			}
		}
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn kaufmanstop_avx2(
	high: &[f64],
	low: &[f64],
	range_ma: &[f64],
	period: usize,
	first: usize,
	mult: f64,
	direction: &str,
	out: &mut [f64],
) {
	kaufmanstop_scalar(high, low, range_ma, period, first, mult, direction, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn kaufmanstop_avx512(
	high: &[f64],
	low: &[f64],
	range_ma: &[f64],
	period: usize,
	first: usize,
	mult: f64,
	direction: &str,
	out: &mut [f64],
) {
	if period <= 32 {
		unsafe { kaufmanstop_avx512_short(high, low, range_ma, period, first, mult, direction, out) }
	} else {
		unsafe { kaufmanstop_avx512_long(high, low, range_ma, period, first, mult, direction, out) }
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn kaufmanstop_avx512_short(
	high: &[f64],
	low: &[f64],
	range_ma: &[f64],
	_period: usize,
	first: usize,
	mult: f64,
	direction: &str,
	out: &mut [f64],
) {
	kaufmanstop_scalar(high, low, range_ma, _period, first, mult, direction, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn kaufmanstop_avx512_long(
	high: &[f64],
	low: &[f64],
	range_ma: &[f64],
	_period: usize,
	first: usize,
	mult: f64,
	direction: &str,
	out: &mut [f64],
) {
	kaufmanstop_scalar(high, low, range_ma, _period, first, mult, direction, out)
}

#[derive(Debug, Clone)]
pub struct KaufmanstopStream {
	period: usize,
	mult: f64,
	direction: String,
	ma_type: String,
	range_buffer: Vec<f64>,
	buffer_head: usize,
	filled: bool,
	ma_vals: Vec<f64>,
}

impl KaufmanstopStream {
	pub fn try_new(params: KaufmanstopParams) -> Result<Self, KaufmanstopError> {
		let period = params.period.unwrap_or(22);
		let mult = params.mult.unwrap_or(2.0);
		let direction = params.direction.unwrap_or_else(|| "long".to_string());
		let ma_type = params.ma_type.unwrap_or_else(|| "sma".to_string());
		if period == 0 {
			return Err(KaufmanstopError::InvalidPeriod { period, data_len: 0 });
		}
		Ok(Self {
			period,
			mult,
			direction,
			ma_type,
			range_buffer: vec![f64::NAN; period],
			buffer_head: 0,
			filled: false,
			ma_vals: vec![f64::NAN; period],
		})
	}
	#[inline(always)]
	pub fn update(&mut self, high: f64, low: f64) -> Option<f64> {
		let diff = if high.is_nan() || low.is_nan() {
			f64::NAN
		} else {
			high - low
		};
		self.range_buffer[self.buffer_head] = diff;
		self.buffer_head = (self.buffer_head + 1) % self.period;

		if !self.filled && self.buffer_head == 0 {
			self.filled = true;
		}
		if !self.filled {
			return None;
		}

		let ma_val = ma(&self.ma_type, MaData::Slice(&self.range_buffer), self.period)
			.ok()
			.and_then(|v| v.last().copied());
		ma_val.map(|val| {
			if self.direction.eq_ignore_ascii_case("long") {
				low - val * self.mult
			} else {
				high + val * self.mult
			}
		})
	}
}

#[derive(Clone, Debug)]
pub struct KaufmanstopBatchRange {
	pub period: (usize, usize, usize),
	pub mult: (f64, f64, f64),
	// String parameters use (start, end, step) tuple for consistency,
	// but step is always 0.0 and ignored since strings don't have numeric steps
	pub direction: (String, String, f64),
	pub ma_type: (String, String, f64),
}

impl Default for KaufmanstopBatchRange {
	fn default() -> Self {
		Self {
			period: (22, 22, 0),
			mult: (2.0, 2.0, 0.0),
			direction: ("long".to_string(), "long".to_string(), 0.0),
			ma_type: ("sma".to_string(), "sma".to_string(), 0.0),
		}
	}
}

#[derive(Clone, Debug, Default)]
pub struct KaufmanstopBatchBuilder {
	range: KaufmanstopBatchRange,
	kernel: Kernel,
}

impl KaufmanstopBatchBuilder {
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
	#[inline]
	pub fn mult_range(mut self, start: f64, end: f64, step: f64) -> Self {
		self.range.mult = (start, end, step);
		self
	}
	#[inline]
	pub fn mult_static(mut self, x: f64) -> Self {
		self.range.mult = (x, x, 0.0);
		self
	}
	#[inline]
	pub fn direction_static<S: Into<String>>(mut self, dir: S) -> Self {
		let s = dir.into();
		self.range.direction = (s.clone(), s, 0.0);
		self
	}
	#[inline]
	pub fn ma_type_static<S: Into<String>>(mut self, t: S) -> Self {
		let s = t.into();
		self.range.ma_type = (s.clone(), s, 0.0);
		self
	}
	pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<KaufmanstopBatchOutput, KaufmanstopError> {
		kaufmanstop_batch_with_kernel(high, low, &self.range, self.kernel)
	}
	pub fn with_default_slices(
		high: &[f64],
		low: &[f64],
		k: Kernel,
	) -> Result<KaufmanstopBatchOutput, KaufmanstopError> {
		KaufmanstopBatchBuilder::new().kernel(k).apply_slices(high, low)
	}
	pub fn apply_candles(self, c: &Candles) -> Result<KaufmanstopBatchOutput, KaufmanstopError> {
		let high = c.select_candle_field("high").map_err(|_| KaufmanstopError::EmptyData)?;
		let low = c.select_candle_field("low").map_err(|_| KaufmanstopError::EmptyData)?;
		self.apply_slices(high, low)
	}
	pub fn with_default_candles(c: &Candles) -> Result<KaufmanstopBatchOutput, KaufmanstopError> {
		KaufmanstopBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c)
	}
}

#[derive(Clone, Debug)]
pub struct KaufmanstopBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<KaufmanstopParams>,
	pub rows: usize,
	pub cols: usize,
}
impl KaufmanstopBatchOutput {
	pub fn row_for_params(&self, p: &KaufmanstopParams) -> Option<usize> {
		self.combos.iter().position(|c| {
			c.period.unwrap_or(22) == p.period.unwrap_or(22)
				&& (c.mult.unwrap_or(2.0) - p.mult.unwrap_or(2.0)).abs() < 1e-12
				&& c.direction.as_ref().unwrap_or(&"long".to_string())
					== p.direction.as_ref().unwrap_or(&"long".to_string())
				&& c.ma_type.as_ref().unwrap_or(&"sma".to_string()) == p.ma_type.as_ref().unwrap_or(&"sma".to_string())
		})
	}
	pub fn values_for(&self, p: &KaufmanstopParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &KaufmanstopBatchRange) -> Vec<KaufmanstopParams> {
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
	fn axis_string((start, end, _step): (String, String, f64)) -> Vec<String> {
		// For string parameters, we only support single values or start/end pairs
		// The step parameter is ignored for strings
		if start == end {
			return vec![start.clone()];
		}
		vec![start.clone(), end.clone()]
	}
	let periods = axis_usize(r.period);
	let mults = axis_f64(r.mult);
	let directions = axis_string(r.direction.clone());
	let ma_types = axis_string(r.ma_type.clone());
	let mut out = Vec::with_capacity(periods.len() * mults.len() * directions.len() * ma_types.len());
	for &p in &periods {
		for &m in &mults {
			for d in &directions {
				for t in &ma_types {
					out.push(KaufmanstopParams {
						period: Some(p),
						mult: Some(m),
						direction: Some(d.clone()),
						ma_type: Some(t.clone()),
					});
				}
			}
		}
	}
	out
}

#[inline(always)]
pub fn kaufmanstop_batch_with_kernel(
	high: &[f64],
	low: &[f64],
	sweep: &KaufmanstopBatchRange,
	k: Kernel,
) -> Result<KaufmanstopBatchOutput, KaufmanstopError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(KaufmanstopError::InvalidPeriod { period: 0, data_len: 0 }),
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	kaufmanstop_batch_par_slice(high, low, sweep, simd)
}

#[inline(always)]
pub fn kaufmanstop_batch_slice(
	high: &[f64],
	low: &[f64],
	sweep: &KaufmanstopBatchRange,
	kern: Kernel,
) -> Result<KaufmanstopBatchOutput, KaufmanstopError> {
	kaufmanstop_batch_inner(high, low, sweep, kern, false)
}

#[inline(always)]
pub fn kaufmanstop_batch_par_slice(
	high: &[f64],
	low: &[f64],
	sweep: &KaufmanstopBatchRange,
	kern: Kernel,
) -> Result<KaufmanstopBatchOutput, KaufmanstopError> {
	kaufmanstop_batch_inner(high, low, sweep, kern, true)
}

#[inline(always)]
fn kaufmanstop_batch_inner(
	high: &[f64],
	low: &[f64],
	sweep: &KaufmanstopBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<KaufmanstopBatchOutput, KaufmanstopError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(KaufmanstopError::InvalidPeriod { period: 0, data_len: 0 });
	}
	let first = high
		.iter()
		.zip(low.iter())
		.position(|(&h, &l)| !h.is_nan() && !l.is_nan())
		.ok_or(KaufmanstopError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if high.len() - first < max_p {
		return Err(KaufmanstopError::NotEnoughValidData {
			needed: max_p,
			valid: high.len() - first,
		});
	}
	let rows = combos.len();
	let cols = high.len();
	// Use helper function to allocate with NaN prefix
	let mut range_buf = alloc_with_nan_prefix(high.len(), first);
	for i in first..high.len() {
		if high[i].is_nan() || low[i].is_nan() {
			range_buf[i] = f64::NAN;
		} else {
			range_buf[i] = high[i] - low[i];
		}
	}
	
	// Use make_uninit_matrix and init_matrix_prefixes instead of vec![f64::NAN; ...]
	let mut buf_mu = make_uninit_matrix(rows, cols);
	
	// Calculate warmup periods for each parameter combination
	let warmup_periods: Vec<usize> = combos.iter()
		.map(|c| first + c.period.unwrap() - 1)
		.collect();
	
	init_matrix_prefixes(&mut buf_mu, cols, &warmup_periods);
	
	// Convert to mutable slice for computation
	let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
	let out: &mut [f64] = unsafe { 
		core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len()) 
	};
	
	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let prm = &combos[row];
		let period = prm.period.unwrap();
		let mult = prm.mult.unwrap();
		let direction = prm.direction.as_ref().unwrap();
		let ma_type = prm.ma_type.as_ref().unwrap();
		let ma_input = MaData::Slice(&range_buf[first..]);
		let hl_diff_ma = match ma(ma_type, ma_input, period) {
			Ok(v) => v,
			Err(_) => {
				for x in out_row.iter_mut() {
					*x = f64::NAN;
				}
				return;
			}
		};
		match kern {
			Kernel::Scalar => kaufmanstop_row_scalar(high, low, &hl_diff_ma, period, first, mult, direction, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => kaufmanstop_row_avx2(high, low, &hl_diff_ma, period, first, mult, direction, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => kaufmanstop_row_avx512(high, low, &hl_diff_ma, period, first, mult, direction, out_row),
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
	
	// Convert back to Vec from ManuallyDrop
	let values = unsafe {
		Vec::from_raw_parts(
			buf_guard.as_mut_ptr() as *mut f64,
			buf_guard.len(),
			buf_guard.capacity() * core::mem::size_of::<std::mem::MaybeUninit<f64>>() / core::mem::size_of::<f64>(),
		)
	};
	
	Ok(KaufmanstopBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
pub unsafe fn kaufmanstop_row_scalar(
	high: &[f64],
	low: &[f64],
	range_ma: &[f64],
	period: usize,
	first: usize,
	mult: f64,
	direction: &str,
	out: &mut [f64],
) {
	kaufmanstop_scalar(high, low, range_ma, period, first, mult, direction, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn kaufmanstop_row_avx2(
	high: &[f64],
	low: &[f64],
	range_ma: &[f64],
	period: usize,
	first: usize,
	mult: f64,
	direction: &str,
	out: &mut [f64],
) {
	kaufmanstop_scalar(high, low, range_ma, period, first, mult, direction, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn kaufmanstop_row_avx512(
	high: &[f64],
	low: &[f64],
	range_ma: &[f64],
	period: usize,
	first: usize,
	mult: f64,
	direction: &str,
	out: &mut [f64],
) {
	if period <= 32 {
		kaufmanstop_row_avx512_short(high, low, range_ma, period, first, mult, direction, out);
	} else {
		kaufmanstop_row_avx512_long(high, low, range_ma, period, first, mult, direction, out);
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn kaufmanstop_row_avx512_short(
	high: &[f64],
	low: &[f64],
	range_ma: &[f64],
	period: usize,
	first: usize,
	mult: f64,
	direction: &str,
	out: &mut [f64],
) {
	kaufmanstop_scalar(high, low, range_ma, period, first, mult, direction, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn kaufmanstop_row_avx512_long(
	high: &[f64],
	low: &[f64],
	range_ma: &[f64],
	period: usize,
	first: usize,
	mult: f64,
	direction: &str,
	out: &mut [f64],
) {
	kaufmanstop_scalar(high, low, range_ma, period, first, mult, direction, out)
}

#[inline(always)]
pub fn expand_grid_wrapper(r: &KaufmanstopBatchRange) -> Vec<KaufmanstopParams> {
	expand_grid(r)
}

#[cfg(feature = "python")]
#[pyfunction(name = "kaufmanstop")]
#[pyo3(signature = (high, low, period=22, mult=2.0, direction="long", ma_type="sma", kernel=None))]
pub fn kaufmanstop_py<'py>(
	py: Python<'py>,
	high: numpy::PyReadonlyArray1<'py, f64>,
	low: numpy::PyReadonlyArray1<'py, f64>,
	period: usize,
	mult: f64,
	direction: &str,
	ma_type: &str,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};

	let high_slice = high.as_slice()?;
	let low_slice = low.as_slice()?;
	
	if high_slice.len() != low_slice.len() {
		return Err(PyValueError::new_err("High and low arrays must have the same length"));
	}
	
	let kern = validate_kernel(kernel, false)?;
	let params = KaufmanstopParams {
		period: Some(period),
		mult: Some(mult),
		direction: Some(direction.to_string()),
		ma_type: Some(ma_type.to_string()),
	};
	let input = KaufmanstopInput::from_slices(high_slice, low_slice, params);

	// Perform computation within allow_threads for better performance
	let result = py.allow_threads(|| kaufmanstop_with_kernel(&input, kern));

	match result {
		Ok(output) => Ok(output.values.into_pyarray(py)),
		Err(e) => Err(PyValueError::new_err(e.to_string())),
	}
}

#[cfg(feature = "python")]
#[pyclass(name = "KaufmanstopStream")]
pub struct KaufmanstopStreamPy {
	stream: KaufmanstopStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl KaufmanstopStreamPy {
	#[new]
	fn new(period: usize, mult: f64, direction: &str, ma_type: &str) -> PyResult<Self> {
		let params = KaufmanstopParams {
			period: Some(period),
			mult: Some(mult),
			direction: Some(direction.to_string()),
			ma_type: Some(ma_type.to_string()),
		};
		let stream = KaufmanstopStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(KaufmanstopStreamPy { stream })
	}

	fn update(&mut self, high: f64, low: f64) -> Option<f64> {
		self.stream.update(high, low)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "kaufmanstop_batch")]
#[pyo3(signature = (high, low, period_range, mult_range=(2.0, 2.0, 0.0), direction="long", ma_type="sma", kernel=None))]
pub fn kaufmanstop_batch_py<'py>(
	py: Python<'py>,
	high: numpy::PyReadonlyArray1<'py, f64>,
	low: numpy::PyReadonlyArray1<'py, f64>,
	period_range: (usize, usize, usize),
	mult_range: (f64, f64, f64),
	direction: &str,
	ma_type: &str,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;

	let high_slice = high.as_slice()?;
	let low_slice = low.as_slice()?;
	
	if high_slice.len() != low_slice.len() {
		return Err(PyValueError::new_err("High and low arrays must have the same length"));
	}

	let sweep = KaufmanstopBatchRange {
		period: period_range,
		mult: mult_range,
		direction: (direction.to_string(), direction.to_string(), 0.0),
		ma_type: (ma_type.to_string(), ma_type.to_string(), 0.0),
	};

	let kern = validate_kernel(kernel, true)?;
	
	// Perform computation within allow_threads for better performance
	let result = py.allow_threads(|| kaufmanstop_batch_with_kernel(high_slice, low_slice, &sweep, kern));

	match result {
		Ok(output) => {
			let dict = PyDict::new(py);
			
			// Convert values to numpy array
			dict.set_item("values", output.values.into_pyarray(py))?;
			
			// Convert combos to list of dicts
			let combos_list = PyList::empty(py);
			for combo in output.combos {
				let combo_dict = PyDict::new(py);
				combo_dict.set_item("period", combo.period.unwrap_or(22))?;
				combo_dict.set_item("mult", combo.mult.unwrap_or(2.0))?;
				combo_dict.set_item("direction", combo.direction.unwrap_or_else(|| "long".to_string()))?;
				combo_dict.set_item("ma_type", combo.ma_type.unwrap_or_else(|| "sma".to_string()))?;
				combos_list.append(combo_dict)?;
			}
			dict.set_item("combos", combos_list)?;
			dict.set_item("rows", output.rows)?;
			dict.set_item("cols", output.cols)?;
			
			Ok(dict)
		}
		Err(e) => Err(PyValueError::new_err(e.to_string())),
	}
}

// ================== WASM Bindings ==================
#[cfg(feature = "wasm")]
/// Core helper that writes directly to an output slice with no allocations.
/// This is the foundation for all WASM APIs.
pub fn kaufmanstop_into_slice(
	dst: &mut [f64],
	input: &KaufmanstopInput,
	kern: Kernel,
) -> Result<(), KaufmanstopError> {
	// Get data slices from input
	let (high, low) = match &input.data {
		KaufmanstopData::Candles { candles } => {
			let high = candles.select_candle_field("high")
				.map_err(|_| KaufmanstopError::EmptyData)?;
			let low = candles.select_candle_field("low")
				.map_err(|_| KaufmanstopError::EmptyData)?;
			(high, low)
		}
		KaufmanstopData::Slices { high, low } => (*high, *low),
	};

	// Validate inputs
	if high.is_empty() || low.is_empty() {
		return Err(KaufmanstopError::EmptyData);
	}
	if high.len() != low.len() {
		return Err(KaufmanstopError::InvalidPeriod {
			period: 0,
			data_len: high.len(),
		});
	}
	if dst.len() != high.len() {
		return Err(KaufmanstopError::InvalidPeriod {
			period: 0,
			data_len: high.len(),
		});
	}

	// Compute the indicator
	let result = kaufmanstop_with_kernel(input, kern)?;
	
	// Copy result to destination slice
	dst.copy_from_slice(&result.values);
	Ok(())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
/// Safe API that allocates a new vector and returns it.
/// `high` and `low` are JavaScript Float64Arrays
pub fn kaufmanstop_js(
	high: &[f64],
	low: &[f64],
	period: usize,
	mult: f64,
	direction: &str,
	ma_type: &str,
) -> Result<Vec<f64>, JsError> {
	let params = KaufmanstopParams {
		period: Some(period),
		mult: Some(mult),
		direction: Some(direction.to_string()),
		ma_type: Some(ma_type.to_string()),
	};
	let input = KaufmanstopInput::from_slices(high, low, params);
	
	match kaufmanstop(&input) {
		Ok(output) => Ok(output.values),
		Err(e) => Err(JsError::new(&e.to_string())),
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
/// Fast API that writes directly to pre-allocated memory.
/// Performs aliasing checks between input and output pointers.
pub fn kaufmanstop_into(
	high_ptr: *const f64,
	low_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period: usize,
	mult: f64,
	direction: &str,
	ma_type: &str,
) -> Result<(), JsError> {
	if high_ptr.is_null() || low_ptr.is_null() || out_ptr.is_null() {
		return Err(JsError::new("Null pointer passed to kaufmanstop_into"));
	}

	// SAFETY: We verify pointers are non-null and trust the caller for:
	// 1. Valid memory allocation
	// 2. Proper length
	// 3. No data races
	unsafe {
		let high = std::slice::from_raw_parts(high_ptr, len);
		let low = std::slice::from_raw_parts(low_ptr, len);
		let out = std::slice::from_raw_parts_mut(out_ptr, len);

		// Aliasing detection: Check if output overlaps with either input
		let high_start = high_ptr as usize;
		let high_end = high_start + len * std::mem::size_of::<f64>();
		let low_start = low_ptr as usize;
		let low_end = low_start + len * std::mem::size_of::<f64>();
		let out_start = out_ptr as usize;
		let out_end = out_start + len * std::mem::size_of::<f64>();

		// Check if output overlaps with high
		let overlaps_high = (out_start < high_end) && (high_start < out_end);
		// Check if output overlaps with low
		let overlaps_low = (out_start < low_end) && (low_start < out_end);

		if overlaps_high || overlaps_low {
			// Overlap detected, use safe path with allocation
			let params = KaufmanstopParams {
				period: Some(period),
				mult: Some(mult),
				direction: Some(direction.to_string()),
				ma_type: Some(ma_type.to_string()),
			};
			let input = KaufmanstopInput::from_slices(high, low, params);
			let result = kaufmanstop(&input)
				.map_err(|e| JsError::new(&e.to_string()))?;
			out.copy_from_slice(&result.values);
		} else {
			// No overlap, use direct write
			let params = KaufmanstopParams {
				period: Some(period),
				mult: Some(mult),
				direction: Some(direction.to_string()),
				ma_type: Some(ma_type.to_string()),
			};
			let input = KaufmanstopInput::from_slices(high, low, params);
			kaufmanstop_into_slice(out, &input, Kernel::Auto)
				.map_err(|e| JsError::new(&e.to_string()))?;
		}
	}
	Ok(())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
/// Allocates memory for a f64 array of the given length.
/// Returns a pointer that must be freed with kaufmanstop_free.
pub fn kaufmanstop_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
/// Frees memory allocated by kaufmanstop_alloc.
/// SAFETY: ptr must have been allocated by kaufmanstop_alloc with the same length.
pub unsafe fn kaufmanstop_free(ptr: *mut f64, len: usize) {
	if ptr.is_null() {
		return;
	}
	Vec::from_raw_parts(ptr, len, len);
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct KaufmanstopBatchMeta {
	pub combos: Vec<KaufmanstopParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
/// Batch computation with parameter sweep.
/// Returns JavaScript object with { values: Float64Array, combos: Array, rows: number, cols: number }
pub fn kaufmanstop_batch_js(
	high: &[f64],
	low: &[f64],
	period_start: usize,
	period_end: usize,
	period_step: usize,
	mult_start: f64,
	mult_end: f64,
	mult_step: f64,
	direction: &str,
	ma_type: &str,
) -> Result<JsValue, JsError> {
	let sweep = KaufmanstopBatchRange {
		period: (period_start, period_end, period_step),
		mult: (mult_start, mult_end, mult_step),
		direction: (direction.to_string(), direction.to_string(), 0.0),
		ma_type: (ma_type.to_string(), ma_type.to_string(), 0.0),
	};
	
	match kaufmanstop_batch_slice(high, low, &sweep, Kernel::Auto) {
		Ok(output) => {
			let meta = KaufmanstopBatchMeta {
				combos: output.combos,
				rows: output.rows,
				cols: output.cols,
			};
			
			// Create JS object manually
			let js_object = js_sys::Object::new();
			
			// Add values as Float64Array
			let values_array = js_sys::Float64Array::from(&output.values[..]);
			js_sys::Reflect::set(&js_object, &"values".into(), &values_array.into())
				.map_err(|e| JsError::new(&format!("Failed to set values: {:?}", e)))?;
			
			// Add metadata
			let meta_value = serde_wasm_bindgen::to_value(&meta)?;
			let combos = js_sys::Reflect::get(&meta_value, &"combos".into())
				.map_err(|e| JsError::new(&format!("Failed to get combos: {:?}", e)))?;
			js_sys::Reflect::set(&js_object, &"combos".into(), &combos)
				.map_err(|e| JsError::new(&format!("Failed to set combos: {:?}", e)))?;
			
			let rows = js_sys::Reflect::get(&meta_value, &"rows".into())
				.map_err(|e| JsError::new(&format!("Failed to get rows: {:?}", e)))?;
			js_sys::Reflect::set(&js_object, &"rows".into(), &rows)
				.map_err(|e| JsError::new(&format!("Failed to set rows: {:?}", e)))?;
			
			let cols = js_sys::Reflect::get(&meta_value, &"cols".into())
				.map_err(|e| JsError::new(&format!("Failed to get cols: {:?}", e)))?;
			js_sys::Reflect::set(&js_object, &"cols".into(), &cols)
				.map_err(|e| JsError::new(&format!("Failed to set cols: {:?}", e)))?;
			
			Ok(js_object.into())
		}
		Err(e) => Err(JsError::new(&e.to_string())),
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
/// Unified batch API that accepts a config object with ranges.
pub fn kaufmanstop_batch_unified_js(
	high: &[f64],
	low: &[f64],
	config: JsValue,
) -> Result<JsValue, JsError> {
	#[derive(Deserialize)]
	struct BatchConfig {
		period_range: Option<(usize, usize, usize)>,
		mult_range: Option<(f64, f64, f64)>,
		direction: Option<String>,
		ma_type: Option<String>,
	}
	
	let config: BatchConfig = serde_wasm_bindgen::from_value(config)?;
	
	let sweep = KaufmanstopBatchRange {
		period: config.period_range.unwrap_or((22, 22, 0)),
		mult: config.mult_range.unwrap_or((2.0, 2.0, 0.0)),
		direction: config.direction
			.map(|d| (d.clone(), d, 0.0))
			.unwrap_or(("long".to_string(), "long".to_string(), 0.0)),
		ma_type: config.ma_type
			.map(|t| (t.clone(), t, 0.0))
			.unwrap_or(("sma".to_string(), "sma".to_string(), 0.0)),
	};
	
	match kaufmanstop_batch_slice(high, low, &sweep, Kernel::Auto) {
		Ok(output) => {
			let meta = KaufmanstopBatchMeta {
				combos: output.combos,
				rows: output.rows,
				cols: output.cols,
			};
			
			// Create JS object
			let js_object = js_sys::Object::new();
			
			// Add values as Float64Array
			let values_array = js_sys::Float64Array::from(&output.values[..]);
			js_sys::Reflect::set(&js_object, &"values".into(), &values_array.into())
				.map_err(|e| JsError::new(&format!("Failed to set values: {:?}", e)))?;
			
			// Add metadata
			let meta_value = serde_wasm_bindgen::to_value(&meta)?;
			let combos = js_sys::Reflect::get(&meta_value, &"combos".into())
				.map_err(|e| JsError::new(&format!("Failed to get combos: {:?}", e)))?;
			js_sys::Reflect::set(&js_object, &"combos".into(), &combos)
				.map_err(|e| JsError::new(&format!("Failed to set combos: {:?}", e)))?;
			
			let rows = js_sys::Reflect::get(&meta_value, &"rows".into())
				.map_err(|e| JsError::new(&format!("Failed to get rows: {:?}", e)))?;
			js_sys::Reflect::set(&js_object, &"rows".into(), &rows)
				.map_err(|e| JsError::new(&format!("Failed to set rows: {:?}", e)))?;
			
			let cols = js_sys::Reflect::get(&meta_value, &"cols".into())
				.map_err(|e| JsError::new(&format!("Failed to get cols: {:?}", e)))?;
			js_sys::Reflect::set(&js_object, &"cols".into(), &cols)
				.map_err(|e| JsError::new(&format!("Failed to set cols: {:?}", e)))?;
			
			Ok(js_object.into())
		}
		Err(e) => Err(JsError::new(&e.to_string())),
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
/// Fast batch API that writes to pre-allocated memory.
pub fn kaufmanstop_batch_into(
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
	direction: &str,
	ma_type: &str,
) -> Result<JsValue, JsError> {
	if high_ptr.is_null() || low_ptr.is_null() || out_ptr.is_null() {
		return Err(JsError::new("Null pointer passed to kaufmanstop_batch_into"));
	}

	unsafe {
		let high = std::slice::from_raw_parts(high_ptr, len);
		let low = std::slice::from_raw_parts(low_ptr, len);
		
		let sweep = KaufmanstopBatchRange {
			period: (period_start, period_end, period_step),
			mult: (mult_start, mult_end, mult_step),
			direction: (direction.to_string(), direction.to_string(), 0.0),
			ma_type: (ma_type.to_string(), ma_type.to_string(), 0.0),
		};
		
		let output = kaufmanstop_batch_slice(high, low, &sweep, Kernel::Auto)
			.map_err(|e| JsError::new(&e.to_string()))?;
		
		// Calculate expected output size
		let expected_size = output.rows * output.cols;
		let out = std::slice::from_raw_parts_mut(out_ptr, expected_size);
		
		// Copy results to output buffer
		out.copy_from_slice(&output.values);
		
		// Return metadata
		let meta = KaufmanstopBatchMeta {
			combos: output.combos,
			rows: output.rows,
			cols: output.cols,
		};
		
		serde_wasm_bindgen::to_value(&meta).map_err(Into::into)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_kaufmanstop_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = KaufmanstopInput::with_default_candles(&candles);
		let output = kaufmanstop_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}
	fn check_kaufmanstop_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = KaufmanstopInput::with_default_candles(&candles);
		let result = kaufmanstop_with_kernel(&input, kernel)?;
		let expected_last_five = [
			56711.545454545456,
			57132.72727272727,
			57015.72727272727,
			57137.18181818182,
			56516.09090909091,
		];
		let start = result.values.len().saturating_sub(5);
		for (i, &val) in result.values[start..].iter().enumerate() {
			let diff = (val - expected_last_five[i]).abs();
			assert!(
				diff < 1e-1,
				"[{}] Kaufmanstop {:?} mismatch at idx {}: got {}, expected {}",
				test_name,
				kernel,
				i,
				val,
				expected_last_five[i]
			);
		}
		Ok(())
	}
	fn check_kaufmanstop_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [10.0, 20.0, 30.0];
		let low = [5.0, 15.0, 25.0];
		let params = KaufmanstopParams {
			period: Some(0),
			mult: None,
			direction: None,
			ma_type: None,
		};
		let input = KaufmanstopInput::from_slices(&high, &low, params);
		let res = kaufmanstop_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] Kaufmanstop should fail with zero period", test_name);
		Ok(())
	}
	fn check_kaufmanstop_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [10.0, 20.0, 30.0];
		let low = [5.0, 15.0, 25.0];
		let params = KaufmanstopParams {
			period: Some(10),
			mult: None,
			direction: None,
			ma_type: None,
		};
		let input = KaufmanstopInput::from_slices(&high, &low, params);
		let res = kaufmanstop_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] Kaufmanstop should fail with period exceeding length",
			test_name
		);
		Ok(())
	}
	fn check_kaufmanstop_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [42.0];
		let low = [41.0];
		let params = KaufmanstopParams {
			period: Some(22),
			mult: None,
			direction: None,
			ma_type: None,
		};
		let input = KaufmanstopInput::from_slices(&high, &low, params);
		let res = kaufmanstop_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] Kaufmanstop should fail with insufficient data",
			test_name
		);
		Ok(())
	}
	fn check_kaufmanstop_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = KaufmanstopInput::with_default_candles(&candles);
		let res = kaufmanstop_with_kernel(&input, kernel)?;
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
	fn check_kaufmanstop_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let high = candles.select_candle_field("high").unwrap();
		let low = candles.select_candle_field("low").unwrap();
		
		// Create stream
		let params = KaufmanstopParams {
			period: Some(22),
			mult: Some(2.0),
			direction: Some("long".to_string()),
			ma_type: Some("sma".to_string()),
		};
		let mut stream = KaufmanstopStream::try_new(params)?;
		
		// Feed data to stream
		let mut stream_results = Vec::new();
		for i in 0..high.len() {
			if let Some(val) = stream.update(high[i], low[i]) {
				stream_results.push(val);
			} else {
				stream_results.push(f64::NAN);
			}
		}
		
		// Compare with batch computation
		let input = KaufmanstopInput::with_default_candles(&candles);
		let batch_result = kaufmanstop_with_kernel(&input, kernel)?;
		
		// Stream should match batch after warmup period
		let warmup = 22 + 21; // period + (period - 1)
		for i in warmup..high.len() {
			let diff = (stream_results[i] - batch_result.values[i]).abs();
			assert!(
				diff < 1e-10 || (stream_results[i].is_nan() && batch_result.values[i].is_nan()),
				"[{}] Stream vs batch mismatch at index {}: {} vs {}",
				test_name,
				i,
				stream_results[i],
				batch_result.values[i]
			);
		}
		Ok(())
	}

	macro_rules! generate_all_kaufmanstop_tests {
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

	generate_all_kaufmanstop_tests!(
		check_kaufmanstop_partial_params,
		check_kaufmanstop_accuracy,
		check_kaufmanstop_zero_period,
		check_kaufmanstop_period_exceeds_length,
		check_kaufmanstop_very_small_dataset,
		check_kaufmanstop_nan_handling,
		check_kaufmanstop_streaming
	);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let output = KaufmanstopBatchBuilder::new().kernel(kernel).apply_candles(&c)?;
		let def = KaufmanstopParams::default();
		let row = output.values_for(&def).expect("default row missing");
		assert_eq!(row.len(), c.close.len());
		let expected = [
			56711.545454545456,
			57132.72727272727,
			57015.72727272727,
			57137.18181818182,
			56516.09090909091,
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
