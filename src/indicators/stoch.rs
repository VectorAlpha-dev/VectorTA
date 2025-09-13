//! # Stochastic Oscillator (Stoch)
//!
//! A momentum indicator comparing a particular closing price to a range of prices over a certain period.
//! Calculates %K (fast stochastic) and %D (slow stochastic) lines using high/low range normalization.
//!
//! ## Parameters
//! - **fastk_period**: Window for highest high/lowest low calculation (default: 14)
//! - **slowk_period**: MA period for %K smoothing (default: 3)
//! - **slowk_ma_type**: MA type for %K smoothing (default: "sma")
//! - **slowd_period**: MA period for %D calculation (default: 3)
//! - **slowd_ma_type**: MA type for %D calculation (default: "sma")
//!
//! ## Inputs
//! - High, low, and close price series (or candles)
//! - All series must have the same length
//!
//! ## Returns
//! - **k**: %K line as `Vec<f64>` (length matches input, range 0-100)
//! - **d**: %D line as `Vec<f64>` (length matches input, range 0-100)
//!
//! ## Developer Notes
//! - **AVX2/AVX512 kernels**: Currently stubs that call scalar implementation
//! - **Streaming update**: O(n) performance due to recalculating min/max over full buffers
//! - **Memory optimization**: Properly uses zero-copy helper functions (alloc_with_nan_prefix, make_uninit_matrix, init_matrix_prefixes)
//! - **TODO**: Implement actual SIMD kernels for AVX2/AVX512
//! - **TODO**: Optimize streaming to maintain rolling min/max for O(1) updates

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::{exceptions::PyValueError, prelude::*};
#[cfg(feature = "python")]
use pyo3::types::PyDict;

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

use crate::indicators::moving_averages::ma::{ma, MaData};
use crate::indicators::utility_functions::{max_rolling, min_rolling};
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

// === Input/Output Structs ===

#[derive(Debug, Clone)]
pub enum StochData<'a> {
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
pub struct StochOutput {
	pub k: Vec<f64>,
	pub d: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct StochParams {
	pub fastk_period: Option<usize>,
	pub slowk_period: Option<usize>,
	pub slowk_ma_type: Option<String>,
	pub slowd_period: Option<usize>,
	pub slowd_ma_type: Option<String>,
}

impl Default for StochParams {
	fn default() -> Self {
		Self {
			fastk_period: Some(14),
			slowk_period: Some(3),
			slowk_ma_type: Some("sma".to_string()),
			slowd_period: Some(3),
			slowd_ma_type: Some("sma".to_string()),
		}
	}
}

#[derive(Debug, Clone)]
pub struct StochInput<'a> {
	pub data: StochData<'a>,
	pub params: StochParams,
}

impl<'a> StochInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, p: StochParams) -> Self {
		Self {
			data: StochData::Candles { candles: c },
			params: p,
		}
	}
	#[inline]
	pub fn from_slices(high: &'a [f64], low: &'a [f64], close: &'a [f64], p: StochParams) -> Self {
		Self {
			data: StochData::Slices { high, low, close },
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, StochParams::default())
	}
	#[inline]
	pub fn get_fastk_period(&self) -> usize {
		self.params.fastk_period.unwrap_or(14)
	}
	#[inline]
	pub fn get_slowk_period(&self) -> usize {
		self.params.slowk_period.unwrap_or(3)
	}
	#[inline]
	pub fn get_slowk_ma_type(&self) -> String {
		self.params.slowk_ma_type.clone().unwrap_or_else(|| "sma".to_string())
	}
	#[inline]
	pub fn get_slowd_period(&self) -> usize {
		self.params.slowd_period.unwrap_or(3)
	}
	#[inline]
	pub fn get_slowd_ma_type(&self) -> String {
		self.params.slowd_ma_type.clone().unwrap_or_else(|| "sma".to_string())
	}
}

// === Builder ===

#[derive(Copy, Clone, Debug)]
pub struct StochBuilder {
	fastk_period: Option<usize>,
	slowk_period: Option<usize>,
	slowk_ma_type: Option<&'static str>,
	slowd_period: Option<usize>,
	slowd_ma_type: Option<&'static str>,
	kernel: Kernel,
}

impl Default for StochBuilder {
	fn default() -> Self {
		Self {
			fastk_period: None,
			slowk_period: None,
			slowk_ma_type: None,
			slowd_period: None,
			slowd_ma_type: None,
			kernel: Kernel::Auto,
		}
	}
}

impl StochBuilder {
	#[inline(always)]
	pub fn new() -> Self {
		Self::default()
	}
	#[inline(always)]
	pub fn fastk_period(mut self, n: usize) -> Self {
		self.fastk_period = Some(n);
		self
	}
	#[inline(always)]
	pub fn slowk_period(mut self, n: usize) -> Self {
		self.slowk_period = Some(n);
		self
	}
	#[inline(always)]
	pub fn slowk_ma_type(mut self, t: &'static str) -> Self {
		self.slowk_ma_type = Some(t);
		self
	}
	#[inline(always)]
	pub fn slowd_period(mut self, n: usize) -> Self {
		self.slowd_period = Some(n);
		self
	}
	#[inline(always)]
	pub fn slowd_ma_type(mut self, t: &'static str) -> Self {
		self.slowd_ma_type = Some(t);
		self
	}
	#[inline(always)]
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}

	#[inline(always)]
	pub fn apply(self, c: &Candles) -> Result<StochOutput, StochError> {
		let p = StochParams {
			fastk_period: self.fastk_period,
			slowk_period: self.slowk_period,
			slowk_ma_type: self.slowk_ma_type.map(|s| s.to_string()),
			slowd_period: self.slowd_period,
			slowd_ma_type: self.slowd_ma_type.map(|s| s.to_string()),
		};
		let i = StochInput::from_candles(c, p);
		stoch_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slices(self, high: &[f64], low: &[f64], close: &[f64]) -> Result<StochOutput, StochError> {
		let p = StochParams {
			fastk_period: self.fastk_period,
			slowk_period: self.slowk_period,
			slowk_ma_type: self.slowk_ma_type.map(|s| s.to_string()),
			slowd_period: self.slowd_period,
			slowd_ma_type: self.slowd_ma_type.map(|s| s.to_string()),
		};
		let i = StochInput::from_slices(high, low, close, p);
		stoch_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<StochStream, StochError> {
		let p = StochParams {
			fastk_period: self.fastk_period,
			slowk_period: self.slowk_period,
			slowk_ma_type: self.slowk_ma_type.map(|s| s.to_string()),
			slowd_period: self.slowd_period,
			slowd_ma_type: self.slowd_ma_type.map(|s| s.to_string()),
		};
		StochStream::try_new(p)
	}
}

// === Errors ===

#[derive(Debug, Error)]
pub enum StochError {
	#[error("stoch: Empty data provided.")]
	EmptyData,
	#[error("stoch: Mismatched length of input data (high, low, close).")]
	MismatchedLength,
	#[error("stoch: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("stoch: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("stoch: All values are NaN.")]
	AllValuesNaN,
	#[error("stoch: {0}")]
	Other(String),
}

// === Indicator Functions (API Parity) ===

#[inline]
pub fn stoch(input: &StochInput) -> Result<StochOutput, StochError> {
	stoch_with_kernel(input, Kernel::Auto)
}

pub fn stoch_with_kernel(input: &StochInput, kernel: Kernel) -> Result<StochOutput, StochError> {
	let (high, low, close) = match &input.data {
		StochData::Candles { candles } => {
			let high = candles
				.select_candle_field("high")
				.map_err(|e| StochError::Other(e.to_string()))?;
			let low = candles
				.select_candle_field("low")
				.map_err(|e| StochError::Other(e.to_string()))?;
			let close = candles
				.select_candle_field("close")
				.map_err(|e| StochError::Other(e.to_string()))?;
			(high, low, close)
		}
		StochData::Slices { high, low, close } => (*high, *low, *close),
	};

	let data_len = high.len();
	if data_len == 0 || low.is_empty() || close.is_empty() {
		return Err(StochError::EmptyData);
	}
	if data_len != low.len() || data_len != close.len() {
		return Err(StochError::MismatchedLength);
	}

	let fastk_period = input.get_fastk_period();
	let slowk_period = input.get_slowk_period();
	let slowd_period = input.get_slowd_period();

	if fastk_period == 0 || fastk_period > data_len {
		return Err(StochError::InvalidPeriod {
			period: fastk_period,
			data_len,
		});
	}
	if slowk_period == 0 || slowk_period > data_len {
		return Err(StochError::InvalidPeriod {
			period: slowk_period,
			data_len,
		});
	}
	if slowd_period == 0 || slowd_period > data_len {
		return Err(StochError::InvalidPeriod {
			period: slowd_period,
			data_len,
		});
	}

	let first_valid_idx = high
		.iter()
		.zip(low.iter())
		.zip(close.iter())
		.position(|((h, l), c)| !h.is_nan() && !l.is_nan() && !c.is_nan())
		.ok_or(StochError::AllValuesNaN)?;

	if (data_len - first_valid_idx) < fastk_period {
		return Err(StochError::NotEnoughValidData {
			needed: fastk_period,
			valid: data_len - first_valid_idx,
		});
	}

	// Use alloc_with_nan_prefix for zero-copy allocation
	let mut hh = alloc_with_nan_prefix(data_len, first_valid_idx + fastk_period - 1);
	let mut ll = alloc_with_nan_prefix(data_len, first_valid_idx + fastk_period - 1);

	let max_vals = max_rolling(&high[first_valid_idx..], fastk_period).map_err(|e| StochError::Other(e.to_string()))?;
	let min_vals = min_rolling(&low[first_valid_idx..], fastk_period).map_err(|e| StochError::Other(e.to_string()))?;

	for (i, &val) in max_vals.iter().enumerate() {
		hh[i + first_valid_idx] = val;
	}
	for (i, &val) in min_vals.iter().enumerate() {
		ll[i + first_valid_idx] = val;
	}

	// Use alloc_with_nan_prefix for zero-copy allocation
	let mut k_raw = alloc_with_nan_prefix(data_len, first_valid_idx + fastk_period - 1);

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};
	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => {
				stoch_scalar(high, low, close, &hh, &ll, fastk_period, first_valid_idx, &mut k_raw)
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => {
				stoch_avx2(high, low, close, &hh, &ll, fastk_period, first_valid_idx, &mut k_raw)
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => {
				stoch_avx512(high, low, close, &hh, &ll, fastk_period, first_valid_idx, &mut k_raw)
			}
			_ => unreachable!(),
		}
	}

	let slowk_ma_type = input.get_slowk_ma_type();
	let slowd_ma_type = input.get_slowd_ma_type();
	let k_vec =
		ma(&slowk_ma_type, MaData::Slice(&k_raw), slowk_period).map_err(|e| StochError::Other(e.to_string()))?;
	let d_vec =
		ma(&slowd_ma_type, MaData::Slice(&k_vec), slowd_period).map_err(|e| StochError::Other(e.to_string()))?;
	Ok(StochOutput { k: k_vec, d: d_vec })
}

// New: write into user buffers, ALMA-style
pub fn stoch_into_slices(
	out_k: &mut [f64],
	out_d: &mut [f64],
	input: &StochInput,
	kernel: Kernel,
) -> Result<(), StochError> {
	let StochOutput { k, d } = stoch_with_kernel(input, kernel)?;
	if out_k.len() != k.len() || out_d.len() != d.len() {
		return Err(StochError::MismatchedLength);
	}
	out_k.copy_from_slice(&k);
	out_d.copy_from_slice(&d);
	Ok(())
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn stoch_avx512(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	hh: &[f64],
	ll: &[f64],
	fastk_period: usize,
	first_valid: usize,
	out: &mut [f64],
) {
	if fastk_period <= 32 {
		unsafe { stoch_avx512_short(high, low, close, hh, ll, fastk_period, first_valid, out) }
	} else {
		unsafe { stoch_avx512_long(high, low, close, hh, ll, fastk_period, first_valid, out) }
	}
}

#[inline]
pub fn stoch_scalar(
	_high: &[f64],
	_low: &[f64],
	close: &[f64],
	hh: &[f64],
	ll: &[f64],
	fastk_period: usize,
	first_val: usize,
	out: &mut [f64],
) {
	for i in (first_val + fastk_period - 1)..close.len() {
		let denom = hh[i] - ll[i];
		if denom.abs() < f64::EPSILON {
			out[i] = 50.0;
		} else {
			out[i] = 100.0 * (close[i] - ll[i]) / denom;
		}
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn stoch_avx2(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	hh: &[f64],
	ll: &[f64],
	fastk_period: usize,
	first_valid: usize,
	out: &mut [f64],
) {
	stoch_scalar(high, low, close, hh, ll, fastk_period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn stoch_avx512_short(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	hh: &[f64],
	ll: &[f64],
	fastk_period: usize,
	first_valid: usize,
	out: &mut [f64],
) {
	stoch_scalar(high, low, close, hh, ll, fastk_period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn stoch_avx512_long(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	hh: &[f64],
	ll: &[f64],
	fastk_period: usize,
	first_valid: usize,
	out: &mut [f64],
) {
	stoch_scalar(high, low, close, hh, ll, fastk_period, first_valid, out)
}

// === Batch API ===

#[derive(Clone, Debug)]
pub struct StochBatchRange {
	pub fastk_period: (usize, usize, usize),
	pub slowk_period: (usize, usize, usize),
	pub slowk_ma_type: (String, String, f64), // Step as dummy, static only
	pub slowd_period: (usize, usize, usize),
	pub slowd_ma_type: (String, String, f64), // Step as dummy, static only
}

impl Default for StochBatchRange {
	fn default() -> Self {
		Self {
			fastk_period: (14, 14, 0),
			slowk_period: (3, 3, 0),
			slowk_ma_type: ("sma".to_string(), "sma".to_string(), 0.0),
			slowd_period: (3, 3, 0),
			slowd_ma_type: ("sma".to_string(), "sma".to_string(), 0.0),
		}
	}
}

#[derive(Clone, Debug, Default)]
pub struct StochBatchBuilder {
	range: StochBatchRange,
	kernel: Kernel,
}

impl StochBatchBuilder {
	pub fn new() -> Self {
		Self::default()
	}
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	pub fn fastk_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.fastk_period = (start, end, step);
		self
	}
	pub fn fastk_period_static(mut self, p: usize) -> Self {
		self.range.fastk_period = (p, p, 0);
		self
	}
	pub fn slowk_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.slowk_period = (start, end, step);
		self
	}
	pub fn slowk_period_static(mut self, p: usize) -> Self {
		self.range.slowk_period = (p, p, 0);
		self
	}
	pub fn slowk_ma_type_static(mut self, t: &str) -> Self {
		self.range.slowk_ma_type = (t.to_string(), t.to_string(), 0.0);
		self
	}
	pub fn slowd_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.slowd_period = (start, end, step);
		self
	}
	pub fn slowd_period_static(mut self, p: usize) -> Self {
		self.range.slowd_period = (p, p, 0);
		self
	}
	pub fn slowd_ma_type_static(mut self, t: &str) -> Self {
		self.range.slowd_ma_type = (t.to_string(), t.to_string(), 0.0);
		self
	}

	pub fn apply_slices(self, high: &[f64], low: &[f64], close: &[f64]) -> Result<StochBatchOutput, StochError> {
		stoch_batch_with_kernel(high, low, close, &self.range, self.kernel)
	}
	pub fn apply_candles(self, c: &Candles) -> Result<StochBatchOutput, StochError> {
		let high = source_type(c, "high");
		let low = source_type(c, "low");
		let close = source_type(c, "close");
		self.apply_slices(high, low, close)
	}
}

pub fn stoch_batch_with_kernel(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	sweep: &StochBatchRange,
	k: Kernel,
) -> Result<StochBatchOutput, StochError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(StochError::InvalidPeriod { period: 0, data_len: 0 }),
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	stoch_batch_par_slice(high, low, close, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct StochBatchOutput {
	pub k: Vec<f64>,
	pub d: Vec<f64>,
	pub combos: Vec<StochParams>,
	pub rows: usize,
	pub cols: usize,
}
impl StochBatchOutput {
	pub fn row_for_params(&self, p: &StochParams) -> Option<usize> {
		self.combos.iter().position(|c| {
			c.fastk_period == p.fastk_period
				&& c.slowk_period == p.slowk_period
				&& c.slowk_ma_type == p.slowk_ma_type
				&& c.slowd_period == p.slowd_period
				&& c.slowd_ma_type == p.slowd_ma_type
		})
	}
	pub fn values_for(&self, p: &StochParams) -> Option<(&[f64], &[f64])> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			(&self.k[start..start + self.cols], &self.d[start..start + self.cols])
		})
	}
}

#[inline(always)]
fn expand_grid(r: &StochBatchRange) -> Vec<StochParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	fn axis_str((start, end, _): (String, String, f64)) -> Vec<String> {
		if start == end {
			vec![start]
		} else {
			vec![start, end]
		}
	}
	let fastk_periods = axis_usize(r.fastk_period);
	let slowk_periods = axis_usize(r.slowk_period);
	let slowk_types = axis_str(r.slowk_ma_type.clone());
	let slowd_periods = axis_usize(r.slowd_period);
	let slowd_types = axis_str(r.slowd_ma_type.clone());
	let mut out = Vec::with_capacity(
		fastk_periods.len() * slowk_periods.len() * slowk_types.len() * slowd_periods.len() * slowd_types.len(),
	);
	for &fkp in &fastk_periods {
		for &skp in &slowk_periods {
			for skt in &slowk_types {
				for &sdp in &slowd_periods {
					for sdt in &slowd_types {
						out.push(StochParams {
							fastk_period: Some(fkp),
							slowk_period: Some(skp),
							slowk_ma_type: Some(skt.clone()),
							slowd_period: Some(sdp),
							slowd_ma_type: Some(sdt.clone()),
						});
					}
				}
			}
		}
	}
	out
}

#[inline(always)]
pub fn stoch_batch_slice(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	sweep: &StochBatchRange,
	kern: Kernel,
) -> Result<StochBatchOutput, StochError> {
	stoch_batch_inner(high, low, close, sweep, kern, false)
}

#[inline(always)]
pub fn stoch_batch_par_slice(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	sweep: &StochBatchRange,
	kern: Kernel,
) -> Result<StochBatchOutput, StochError> {
	stoch_batch_inner(high, low, close, sweep, kern, true)
}

#[inline(always)]
fn stoch_batch_inner(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	sweep: &StochBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<StochBatchOutput, StochError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(StochError::InvalidPeriod { period: 0, data_len: 0 });
	}
	
	let n = high.len();
	if n == 0 || low.len() != n || close.len() != n {
		return Err(StochError::MismatchedLength);
	}
	
	let first = high
		.iter()
		.zip(low.iter())
		.zip(close.iter())
		.position(|((h, l), c)| !h.is_nan() && !l.is_nan() && !c.is_nan())
		.ok_or(StochError::AllValuesNaN)?;
	let max_fkp = combos.iter().map(|c| c.fastk_period.unwrap()).max().unwrap();
	if n - first < max_fkp {
		return Err(StochError::NotEnoughValidData {
			needed: max_fkp,
			valid: n - first,
		});
	}

	// Allocate K and D matrices uninitialized
	let rows = combos.len();
	let cols = n;

	let mut k_mu = make_uninit_matrix(rows, cols);
	let mut d_mu = make_uninit_matrix(rows, cols);

	// Warmup for raw %K. Further MA warmups are handled by ma.rs and copied over.
	let warm_k: Vec<usize> = combos.iter().map(|c| first + c.fastk_period.unwrap() - 1).collect();
	init_matrix_prefixes(&mut k_mu, cols, &warm_k);
	init_matrix_prefixes(&mut d_mu, cols, &warm_k); // at least as conservative for %D

	// Convert to &mut [f64] to write final values
	let mut k_guard = core::mem::ManuallyDrop::new(k_mu);
	let mut d_guard = core::mem::ManuallyDrop::new(d_mu);
	let k_mat: &mut [f64] = unsafe { core::slice::from_raw_parts_mut(k_guard.as_mut_ptr() as *mut f64, k_guard.len()) };
	let d_mat: &mut [f64] = unsafe { core::slice::from_raw_parts_mut(d_guard.as_mut_ptr() as *mut f64, d_guard.len()) };
	let do_row = |row: usize, dst_k: &mut [f64], dst_d: &mut [f64]| {
		let prm = &combos[row];

		// Build hh/ll with a single NaN prefix alloc per row
		let mut hh = alloc_with_nan_prefix(cols, first + prm.fastk_period.unwrap() - 1);
		let mut ll = alloc_with_nan_prefix(cols, first + prm.fastk_period.unwrap() - 1);
		let highs = max_rolling(&high[first..], prm.fastk_period.unwrap()).unwrap();
		let lows = min_rolling(&low[first..], prm.fastk_period.unwrap()).unwrap();
		for (i, &v) in highs.iter().enumerate() {
			hh[first + i] = v;
		}
		for (i, &v) in lows.iter().enumerate() {
			ll[first + i] = v;
		}

		let mut k_raw = alloc_with_nan_prefix(cols, first + prm.fastk_period.unwrap() - 1);
		unsafe {
			match kern {
				Kernel::Scalar => stoch_row_scalar(high, low, close, &hh, &ll, prm.fastk_period.unwrap(), first, &mut k_raw),
				#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
				Kernel::Avx2 => stoch_row_avx2(high, low, close, &hh, &ll, prm.fastk_period.unwrap(), first, &mut k_raw),
				#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
				Kernel::Avx512 => stoch_row_avx512(high, low, close, &hh, &ll, prm.fastk_period.unwrap(), first, &mut k_raw),
				_ => unreachable!(),
			}
		}

		let k_vec = ma(
			prm.slowk_ma_type.as_ref().unwrap(),
			MaData::Slice(&k_raw),
			prm.slowk_period.unwrap(),
		).unwrap();
		let d_vec = ma(
			prm.slowd_ma_type.as_ref().unwrap(),
			MaData::Slice(&k_vec),
			prm.slowd_period.unwrap(),
		).unwrap();

		dst_k.copy_from_slice(&k_vec);
		dst_d.copy_from_slice(&d_vec);
	};
	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		k_mat.par_chunks_mut(cols).zip(d_mat.par_chunks_mut(cols)).enumerate()
			.for_each(|(row, (krow, drow))| do_row(row, krow, drow));
		#[cfg(target_arch = "wasm32")]
		for (row, (krow, drow)) in k_mat.chunks_mut(cols).zip(d_mat.chunks_mut(cols)).enumerate() {
			do_row(row, krow, drow);
		}
	} else {
		for (row, (krow, drow)) in k_mat.chunks_mut(cols).zip(d_mat.chunks_mut(cols)).enumerate() {
			do_row(row, krow, drow);
		}
	}

	let k = unsafe { Vec::from_raw_parts(k_guard.as_mut_ptr() as *mut f64, k_guard.len(), k_guard.capacity()) };
	let d = unsafe { Vec::from_raw_parts(d_guard.as_mut_ptr() as *mut f64, d_guard.len(), d_guard.capacity()) };
	core::mem::forget(k_guard);
	core::mem::forget(d_guard);

	Ok(StochBatchOutput { k, d, combos, rows, cols })
}

#[inline(always)]
unsafe fn stoch_row_scalar(
	_high: &[f64],
	_low: &[f64],
	close: &[f64],
	hh: &[f64],
	ll: &[f64],
	fastk_period: usize,
	first: usize,
	out: &mut [f64],
) {
	for i in (first + fastk_period - 1)..close.len() {
		let denom = hh[i] - ll[i];
		if denom.abs() < f64::EPSILON {
			out[i] = 50.0;
		} else {
			out[i] = 100.0 * (close[i] - ll[i]) / denom;
		}
	}
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn stoch_row_avx2(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	hh: &[f64],
	ll: &[f64],
	fastk_period: usize,
	first: usize,
	out: &mut [f64],
) {
	stoch_row_scalar(high, low, close, hh, ll, fastk_period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn stoch_row_avx512(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	hh: &[f64],
	ll: &[f64],
	fastk_period: usize,
	first: usize,
	out: &mut [f64],
) {
	if fastk_period <= 32 {
		stoch_row_avx512_short(high, low, close, hh, ll, fastk_period, first, out)
	} else {
		stoch_row_avx512_long(high, low, close, hh, ll, fastk_period, first, out)
	}
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn stoch_row_avx512_short(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	hh: &[f64],
	ll: &[f64],
	fastk_period: usize,
	first: usize,
	out: &mut [f64],
) {
	stoch_row_scalar(high, low, close, hh, ll, fastk_period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn stoch_row_avx512_long(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	hh: &[f64],
	ll: &[f64],
	fastk_period: usize,
	first: usize,
	out: &mut [f64],
) {
	stoch_row_scalar(high, low, close, hh, ll, fastk_period, first, out)
}

// === Streaming ===

#[derive(Debug, Clone)]
pub struct StochStream {
	fastk_period: usize,
	slowk_period: usize,
	slowk_ma_type: String,
	slowd_period: usize,
	slowd_ma_type: String,
	high_buf: Vec<f64>,
	low_buf: Vec<f64>,
	close_buf: Vec<f64>,
	k_stream: Option<Vec<f64>>,
	d_stream: Option<Vec<f64>>,
	head: usize,
	filled: bool,
}

impl StochStream {
	pub fn try_new(params: StochParams) -> Result<Self, StochError> {
		let fastk_period = params.fastk_period.unwrap_or(14);
		let slowk_period = params.slowk_period.unwrap_or(3);
		let slowd_period = params.slowd_period.unwrap_or(3);
		if fastk_period == 0 || slowk_period == 0 || slowd_period == 0 {
			return Err(StochError::InvalidPeriod { period: 0, data_len: 0 });
		}
		Ok(Self {
			fastk_period,
			slowk_period,
			slowk_ma_type: params.slowk_ma_type.unwrap_or_else(|| "sma".to_string()),
			slowd_period,
			slowd_ma_type: params.slowd_ma_type.unwrap_or_else(|| "sma".to_string()),
			high_buf: vec![f64::NAN; fastk_period],
			low_buf: vec![f64::NAN; fastk_period],
			close_buf: vec![f64::NAN; fastk_period],
			k_stream: None,
			d_stream: None,
			head: 0,
			filled: false,
		})
	}
	pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64)> {
		self.high_buf[self.head] = high;
		self.low_buf[self.head] = low;
		self.close_buf[self.head] = close;
		self.head = (self.head + 1) % self.fastk_period;
		if !self.filled && self.head == 0 {
			self.filled = true;
		}
		if !self.filled {
			return None;
		}
		let start = if self.head == 0 { 0 } else { self.head };
		let mut highs = vec![];
		let mut lows = vec![];
		let mut closes = vec![];
		for i in 0..self.fastk_period {
			let idx = (start + i) % self.fastk_period;
			highs.push(self.high_buf[idx]);
			lows.push(self.low_buf[idx]);
			closes.push(self.close_buf[idx]);
		}
		let max_h = highs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
		let min_l = lows.iter().cloned().fold(f64::INFINITY, f64::min);
		let last_close = closes[self.fastk_period - 1];
		let k_val = if (max_h - min_l).abs() < f64::EPSILON {
			50.0
		} else {
			100.0 * (last_close - min_l) / (max_h - min_l)
		};
		let mut k_vec = self
			.k_stream
			.take()
			.unwrap_or_else(|| vec![f64::NAN; self.slowk_period]);
		k_vec.remove(0);
		k_vec.push(k_val);
		self.k_stream = Some(k_vec.clone());
		
		// Try to calculate smoothed K, if fails use raw value
		let k_last = match ma(&self.slowk_ma_type, MaData::Slice(&k_vec), self.slowk_period) {
			Ok(slowk) => *slowk.last().unwrap_or(&f64::NAN),
			Err(_) => k_val,  // If MA fails (e.g., not enough valid data), use raw value
		};
		
		let mut d_vec = self
			.d_stream
			.take()
			.unwrap_or_else(|| vec![f64::NAN; self.slowd_period]);
		d_vec.remove(0);
		d_vec.push(k_last);
		self.d_stream = Some(d_vec.clone());
		
		// Try to calculate smoothed D, if fails use K value
		let d_last = match ma(&self.slowd_ma_type, MaData::Slice(&d_vec), self.slowd_period) {
			Ok(slowd) => *slowd.last().unwrap_or(&f64::NAN),
			Err(_) => k_last,  // If MA fails (e.g., not enough valid data), use K value
		};
		
		Some((k_last, d_last))
	}
}

// === Python Bindings ===

#[cfg(feature = "python")]
#[pyfunction(name="stoch")]
#[pyo3(signature = (high, low, close, fastk_period=14, slowk_period=3, slowk_ma_type="sma", slowd_period=3, slowd_ma_type="sma", kernel=None))]
pub fn stoch_py<'py>(
	py: Python<'py>,
	high: PyReadonlyArray1<'py, f64>,
	low:  PyReadonlyArray1<'py, f64>,
	close:PyReadonlyArray1<'py, f64>,
	fastk_period: usize,
	slowk_period: usize,
	slowk_ma_type: &str,
	slowd_period: usize,
	slowd_ma_type: &str,
	kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
	let hi = high.as_slice()?;
	let lo = low.as_slice()?;
	let cl = close.as_slice()?;
	let params = StochParams {
		fastk_period: Some(fastk_period),
		slowk_period: Some(slowk_period),
		slowk_ma_type: Some(slowk_ma_type.to_string()),
		slowd_period: Some(slowd_period),
		slowd_ma_type: Some(slowd_ma_type.to_string()),
	};
	let kern = validate_kernel(kernel, false)?;
	let input = StochInput::from_slices(hi, lo, cl, params);
	let out = py.allow_threads(|| stoch_with_kernel(&input, kern))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;
	Ok((out.k.into_pyarray(py), out.d.into_pyarray(py)))
}

#[cfg(feature = "python")]
#[pyfunction(name="stoch_batch")]
#[pyo3(signature = (high, low, close, fastk_range, slowk_range, slowk_ma_type, slowd_range, slowd_ma_type, kernel=None))]
pub fn stoch_batch_py<'py>(
	py: Python<'py>,
	high: PyReadonlyArray1<'py, f64>,
	low:  PyReadonlyArray1<'py, f64>,
	close:PyReadonlyArray1<'py, f64>,
	fastk_range: (usize, usize, usize),
	slowk_range: (usize, usize, usize),
	slowk_ma_type: &str, // static in sweep
	slowd_range: (usize, usize, usize),
	slowd_ma_type: &str, // static in sweep
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
	let hi = high.as_slice()?;
	let lo = low.as_slice()?;
	let cl = close.as_slice()?;

	let sweep = StochBatchRange {
		fastk_period: fastk_range,
		slowk_period: slowk_range,
		slowk_ma_type: (slowk_ma_type.to_string(), slowk_ma_type.to_string(), 0.0),
		slowd_period: slowd_range,
		slowd_ma_type: (slowd_ma_type.to_string(), slowd_ma_type.to_string(), 0.0),
	};

	let kern = validate_kernel(kernel, true)?;
	let out = py.allow_threads(|| stoch_batch_with_kernel(hi, lo, cl, &sweep, kern))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	let rows = out.rows;
	let cols = out.cols;

	let dict = PyDict::new(py);

	// 2D shape views for convenience; still contiguous
	let k_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let d_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	unsafe { k_arr.as_slice_mut()? }.copy_from_slice(&out.k);
	unsafe { d_arr.as_slice_mut()? }.copy_from_slice(&out.d);

	dict.set_item("k", k_arr.reshape((rows, cols))?)?;
	dict.set_item("d", d_arr.reshape((rows, cols))?)?;
	dict.set_item("fastk_periods", out.combos.iter().map(|p| p.fastk_period.unwrap() as u64).collect::<Vec<_>>().into_pyarray(py))?;
	dict.set_item("slowk_periods", out.combos.iter().map(|p| p.slowk_period.unwrap() as u64).collect::<Vec<_>>().into_pyarray(py))?;
	dict.set_item("slowk_types",  out.combos.iter().map(|p| p.slowk_ma_type.as_deref().unwrap_or("sma")).collect::<Vec<_>>())?;
	dict.set_item("slowd_periods", out.combos.iter().map(|p| p.slowd_period.unwrap() as u64).collect::<Vec<_>>().into_pyarray(py))?;
	dict.set_item("slowd_types",  out.combos.iter().map(|p| p.slowd_ma_type.as_deref().unwrap_or("sma")).collect::<Vec<_>>())?;

	Ok(dict)
}

// Optional: streaming wrapper parity with ALMA
#[cfg(feature = "python")]
#[pyclass(name="StochStream")]
pub struct StochStreamPy { 
	stream: StochStream 
}

#[cfg(feature = "python")]
#[pymethods]
impl StochStreamPy {
	#[new]
	fn new(fastk_period: usize, slowk_period: usize, slowk_ma_type: &str, slowd_period: usize, slowd_ma_type: &str) -> PyResult<Self> {
		let params = StochParams {
			fastk_period: Some(fastk_period),
			slowk_period: Some(slowk_period),
			slowk_ma_type: Some(slowk_ma_type.to_string()),
			slowd_period: Some(slowd_period),
			slowd_ma_type: Some(slowd_ma_type.to_string()),
		};
		Ok(Self { stream: StochStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))? })
	}
	fn update(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64)> {
		self.stream.update(high, low, close)
	}
}

// === WASM Bindings ===

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct StochResult {
	pub values: Vec<f64>, // [k..., d...]
	pub rows: usize,      // 2
	pub cols: usize,      // data length
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = stoch)]
pub fn stoch_js(high: &[f64], low: &[f64], close: &[f64],
				fastk_period: usize, slowk_period: usize, slowk_ma_type: &str,
				slowd_period: usize, slowd_ma_type: &str) -> Result<JsValue, JsValue> {
	let params = StochParams {
		fastk_period: Some(fastk_period),
		slowk_period: Some(slowk_period),
		slowk_ma_type: Some(slowk_ma_type.to_string()),
		slowd_period: Some(slowd_period),
		slowd_ma_type: Some(slowd_ma_type.to_string()),
	};
	let input = StochInput::from_slices(high, low, close, params);
	let out = stoch_with_kernel(&input, detect_best_kernel()).map_err(|e| JsValue::from_str(&e.to_string()))?;
	let mut values = out.k;
	values.extend_from_slice(&out.d);
	serde_wasm_bindgen::to_value(&StochResult { values, rows: 2, cols: high.len() })
		.map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct StochBatchJsOutput {
	pub values: Vec<f64>,        // [all K rows..., then all D rows...]
	pub combos: Vec<StochParams>,
	pub rows_per_combo: usize,   // 2
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = stoch_batch)]
pub fn stoch_batch_unified_js(
	high: &[f64], low: &[f64], close: &[f64],
	fastk_start: usize, fastk_end: usize, fastk_step: usize,
	slowk_start: usize, slowk_end: usize, slowk_step: usize,
	slowk_ma_type: &str,
	slowd_start: usize, slowd_end: usize, slowd_step: usize,
	slowd_ma_type: &str,
) -> Result<JsValue, JsValue> {
	let sweep = StochBatchRange {
		fastk_period: (fastk_start, fastk_end, fastk_step),
		slowk_period: (slowk_start, slowk_end, slowk_step),
		slowk_ma_type: (slowk_ma_type.to_string(), slowk_ma_type.to_string(), 0.0),
		slowd_period: (slowd_start, slowd_end, slowd_step),
		slowd_ma_type: (slowd_ma_type.to_string(), slowd_ma_type.to_string(), 0.0),
	};
	let out = stoch_batch_inner(high, low, close, &sweep, detect_best_kernel(), false)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	let mut values = out.k.clone();
	values.extend_from_slice(&out.d);
	let js = StochBatchJsOutput {
		values,
		combos: out.combos,
		rows_per_combo: 2,
		cols: out.cols,
	};
	serde_wasm_bindgen::to_value(&js).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

// Optional: raw pointers API to avoid extra allocations in tight loops
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn stoch_alloc(len: usize) -> *mut f64 {
	let mut v = Vec::<f64>::with_capacity(len);
	let ptr = v.as_mut_ptr();
	core::mem::forget(v);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn stoch_free(ptr: *mut f64, len: usize) { 
	unsafe { let _ = Vec::from_raw_parts(ptr, len, len); } 
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = stoch_into)]
pub fn stoch_into_js(
	high_ptr: *const f64, low_ptr: *const f64, close_ptr: *const f64, len: usize,
	fastk_period: usize, slowk_period: usize, slowk_ma_type: &str,
	slowd_period: usize, slowd_ma_type: &str,
	out_k_ptr: *mut f64, out_d_ptr: *mut f64
) -> Result<(), JsValue> {
	if [high_ptr, low_ptr, close_ptr, out_k_ptr, out_d_ptr].iter().any(|p| p.is_null()) {
		return Err(JsValue::from_str("null pointer"));
	}
	unsafe {
		let hi = core::slice::from_raw_parts(high_ptr, len);
		let lo = core::slice::from_raw_parts(low_ptr, len);
		let cl = core::slice::from_raw_parts(close_ptr, len);
		let mut ok = core::slice::from_raw_parts_mut(out_k_ptr, len);
		let mut od = core::slice::from_raw_parts_mut(out_d_ptr, len);
		let params = StochParams {
			fastk_period: Some(fastk_period),
			slowk_period: Some(slowk_period),
			slowk_ma_type: Some(slowk_ma_type.to_string()),
			slowd_period: Some(slowd_period),
			slowd_ma_type: Some(slowd_ma_type.to_string()),
		};
		let input = StochInput::from_slices(hi, lo, cl, params);
		stoch_into_slices(&mut ok, &mut od, &input, detect_best_kernel())
			.map_err(|e| JsValue::from_str(&e.to_string()))
	}
}

// === Tests ===

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;
	use crate::utilities::enums::Kernel;

	fn check_stoch_partial_params(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let default_params = StochParams::default();
		let input = StochInput::from_candles(&candles, default_params);
		let output = stoch_with_kernel(&input, kernel)?;
		assert_eq!(output.k.len(), candles.close.len());
		assert_eq!(output.d.len(), candles.close.len());
		Ok(())
	}
	fn check_stoch_accuracy(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = StochInput::from_candles(&candles, StochParams::default());
		let result = stoch_with_kernel(&input, kernel)?;
		assert_eq!(result.k.len(), candles.close.len());
		assert_eq!(result.d.len(), candles.close.len());
		let last_five_k = [
			42.51122827572717,
			40.13864479593807,
			37.853934778363374,
			37.337021714266086,
			36.26053890551548,
		];
		let last_five_d = [
			41.36561869426493,
			41.7691857059163,
			40.16793595000925,
			38.44320042952222,
			37.15049846604803,
		];
		let k_slice = &result.k[result.k.len() - 5..];
		let d_slice = &result.d[result.d.len() - 5..];
		for i in 0..5 {
			assert!(
				(k_slice[i] - last_five_k[i]).abs() < 1e-6,
				"Mismatch in K at {}: got {}, expected {}",
				i,
				k_slice[i],
				last_five_k[i]
			);
			assert!(
				(d_slice[i] - last_five_d[i]).abs() < 1e-6,
				"Mismatch in D at {}: got {}, expected {}",
				i,
				d_slice[i],
				last_five_d[i]
			);
		}
		Ok(())
	}
	fn check_stoch_default_candles(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = StochInput::with_default_candles(&candles);
		let output = stoch_with_kernel(&input, kernel)?;
		assert_eq!(output.k.len(), candles.close.len());
		assert_eq!(output.d.len(), candles.close.len());
		Ok(())
	}
	fn check_stoch_zero_period(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let high = [10.0, 11.0, 12.0];
		let low = [9.0, 9.5, 10.5];
		let close = [9.5, 10.6, 11.5];
		let params = StochParams {
			fastk_period: Some(0),
			..Default::default()
		};
		let input = StochInput::from_slices(&high, &low, &close, params);
		let result = stoch_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}
	fn check_stoch_period_exceeds_length(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let high = [10.0, 11.0, 12.0];
		let low = [9.0, 9.5, 10.5];
		let close = [9.5, 10.6, 11.5];
		let params = StochParams {
			fastk_period: Some(10),
			..Default::default()
		};
		let input = StochInput::from_slices(&high, &low, &close, params);
		let result = stoch_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}
	fn check_stoch_all_nan(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let nan_data = [f64::NAN, f64::NAN, f64::NAN];
		let params = StochParams::default();
		let input = StochInput::from_slices(&nan_data, &nan_data, &nan_data, params);
		let result = stoch_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}

	#[cfg(debug_assertions)]
	fn check_stoch_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		
		// Define comprehensive parameter combinations
		let test_params = vec![
			// Default parameters
			StochParams::default(),
			// Minimum periods
			StochParams {
				fastk_period: Some(2),
				slowk_period: Some(1),
				slowd_period: Some(1),
				slowk_ma_type: Some("sma".to_string()),
				slowd_ma_type: Some("sma".to_string()),
			},
			// Small periods
			StochParams {
				fastk_period: Some(5),
				slowk_period: Some(2),
				slowd_period: Some(2),
				slowk_ma_type: Some("sma".to_string()),
				slowd_ma_type: Some("sma".to_string()),
			},
			// Medium periods with EMA
			StochParams {
				fastk_period: Some(10),
				slowk_period: Some(5),
				slowd_period: Some(3),
				slowk_ma_type: Some("ema".to_string()),
				slowd_ma_type: Some("ema".to_string()),
			},
			// Default fastk with different smoothing
			StochParams {
				fastk_period: Some(14),
				slowk_period: Some(5),
				slowd_period: Some(5),
				slowk_ma_type: Some("sma".to_string()),
				slowd_ma_type: Some("ema".to_string()),
			},
			// Large fastk period
			StochParams {
				fastk_period: Some(20),
				slowk_period: Some(3),
				slowd_period: Some(3),
				slowk_ma_type: Some("sma".to_string()),
				slowd_ma_type: Some("sma".to_string()),
			},
			// Very large periods
			StochParams {
				fastk_period: Some(50),
				slowk_period: Some(10),
				slowd_period: Some(10),
				slowk_ma_type: Some("ema".to_string()),
				slowd_ma_type: Some("sma".to_string()),
			},
			// Maximum practical periods
			StochParams {
				fastk_period: Some(100),
				slowk_period: Some(20),
				slowd_period: Some(15),
				slowk_ma_type: Some("sma".to_string()),
				slowd_ma_type: Some("sma".to_string()),
			},
			// Asymmetric smoothing periods
			StochParams {
				fastk_period: Some(7),
				slowk_period: Some(1),
				slowd_period: Some(7),
				slowk_ma_type: Some("sma".to_string()),
				slowd_ma_type: Some("ema".to_string()),
			},
			// Another edge case
			StochParams {
				fastk_period: Some(3),
				slowk_period: Some(3),
				slowd_period: Some(1),
				slowk_ma_type: Some("ema".to_string()),
				slowd_ma_type: Some("sma".to_string()),
			},
		];
		
		for (param_idx, params) in test_params.iter().enumerate() {
			let input = StochInput::from_candles(&candles, params.clone());
			let output = stoch_with_kernel(&input, kernel)?;
			
			// Check K values
			for (i, &val) in output.k.iter().enumerate() {
				if val.is_nan() {
					continue; // NaN values are expected during warmup
				}
				
				let bits = val.to_bits();
				
				// Check all three poison patterns
				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in K values \
						 with params: fastk_period={}, slowk_period={}, slowd_period={}, \
						 slowk_ma_type={}, slowd_ma_type={} (param set {})",
						test_name, val, bits, i,
						params.fastk_period.unwrap_or(14),
						params.slowk_period.unwrap_or(3),
						params.slowd_period.unwrap_or(3),
						params.slowk_ma_type.as_deref().unwrap_or("sma"),
						params.slowd_ma_type.as_deref().unwrap_or("sma"),
						param_idx
					);
				}
				
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in K values \
						 with params: fastk_period={}, slowk_period={}, slowd_period={}, \
						 slowk_ma_type={}, slowd_ma_type={} (param set {})",
						test_name, val, bits, i,
						params.fastk_period.unwrap_or(14),
						params.slowk_period.unwrap_or(3),
						params.slowd_period.unwrap_or(3),
						params.slowk_ma_type.as_deref().unwrap_or("sma"),
						params.slowd_ma_type.as_deref().unwrap_or("sma"),
						param_idx
					);
				}
				
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in K values \
						 with params: fastk_period={}, slowk_period={}, slowd_period={}, \
						 slowk_ma_type={}, slowd_ma_type={} (param set {})",
						test_name, val, bits, i,
						params.fastk_period.unwrap_or(14),
						params.slowk_period.unwrap_or(3),
						params.slowd_period.unwrap_or(3),
						params.slowk_ma_type.as_deref().unwrap_or("sma"),
						params.slowd_ma_type.as_deref().unwrap_or("sma"),
						param_idx
					);
				}
			}
			
			// Check D values
			for (i, &val) in output.d.iter().enumerate() {
				if val.is_nan() {
					continue; // NaN values are expected during warmup
				}
				
				let bits = val.to_bits();
				
				// Check all three poison patterns
				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in D values \
						 with params: fastk_period={}, slowk_period={}, slowd_period={}, \
						 slowk_ma_type={}, slowd_ma_type={} (param set {})",
						test_name, val, bits, i,
						params.fastk_period.unwrap_or(14),
						params.slowk_period.unwrap_or(3),
						params.slowd_period.unwrap_or(3),
						params.slowk_ma_type.as_deref().unwrap_or("sma"),
						params.slowd_ma_type.as_deref().unwrap_or("sma"),
						param_idx
					);
				}
				
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in D values \
						 with params: fastk_period={}, slowk_period={}, slowd_period={}, \
						 slowk_ma_type={}, slowd_ma_type={} (param set {})",
						test_name, val, bits, i,
						params.fastk_period.unwrap_or(14),
						params.slowk_period.unwrap_or(3),
						params.slowd_period.unwrap_or(3),
						params.slowk_ma_type.as_deref().unwrap_or("sma"),
						params.slowd_ma_type.as_deref().unwrap_or("sma"),
						param_idx
					);
				}
				
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in D values \
						 with params: fastk_period={}, slowk_period={}, slowd_period={}, \
						 slowk_ma_type={}, slowd_ma_type={} (param set {})",
						test_name, val, bits, i,
						params.fastk_period.unwrap_or(14),
						params.slowk_period.unwrap_or(3),
						params.slowd_period.unwrap_or(3),
						params.slowk_ma_type.as_deref().unwrap_or("sma"),
						params.slowd_ma_type.as_deref().unwrap_or("sma"),
						param_idx
					);
				}
			}
		}
		
		Ok(())
	}

	#[cfg(not(debug_assertions))]
	fn check_stoch_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(()) // No-op in release builds
	}

	#[cfg(feature = "proptest")]
	#[allow(clippy::float_cmp)]
	fn check_stoch_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		use proptest::prelude::*;
		skip_if_unsupported!(kernel, test_name);

		// Generate realistic OHLC data with proper price relationships and trends
		let strat = (2usize..=50)
			.prop_flat_map(|fastk_period| {
				(
					// Generate price series with trend component
					prop::collection::vec(
						(1.0f64..1000.0f64, 0.001f64..0.1f64), // (price, volatility)
						fastk_period.max(10)..400,
					),
					Just(fastk_period),
					1usize..=10,  // slowk_period
					1usize..=10,  // slowd_period
					prop::bool::ANY,  // use_ema (false=sma, true=ema)
					-0.01f64..0.01f64,  // trend factor
					prop::bool::ANY,  // test flat market
				)
			})
			.prop_flat_map(|(price_vol_pairs, fastk_period, slowk_period, slowd_period, use_ema, trend, is_flat)| {
				// Generate close position within high-low range for each bar
				let len = price_vol_pairs.len();
				(
					Just((price_vol_pairs, fastk_period, slowk_period, slowd_period, use_ema, trend, is_flat)),
					prop::collection::vec(-1.0f64..1.0f64, len), // Close position factor
					prop::collection::vec(0.0f64..1.0f64, len),  // Beta distribution parameters for realistic close
				)
			})
			.prop_map(|((price_vol_pairs, fastk_period, slowk_period, slowd_period, use_ema, trend, is_flat), close_factors, beta_params)| {
				// Generate OHLC data maintaining high >= close >= low relationship
				let mut high = Vec::with_capacity(price_vol_pairs.len());
				let mut low = Vec::with_capacity(price_vol_pairs.len());
				let mut close = Vec::with_capacity(price_vol_pairs.len());
				
				let mut cumulative_trend = 1.0;
				
				for (i, ((base_price, volatility), (close_factor, beta))) in 
					price_vol_pairs.into_iter().zip(close_factors.into_iter().zip(beta_params)).enumerate() 
				{
					// Apply trend
					cumulative_trend *= 1.0 + trend;
					let trended_price = base_price * cumulative_trend;
					
					if is_flat {
						// Test flat market case
						let flat_price = if i == 0 { base_price } else { high[0] };
						high.push(flat_price);
						low.push(flat_price);
						close.push(flat_price);
					} else {
						let spread = trended_price * volatility;
						let h = trended_price + spread;
						let l = (trended_price - spread).max(0.01);
						
						// Use beta-like distribution for more realistic close positions
						// Most closes near middle, fewer at extremes
						let beta_factor = if beta < 0.5 {
							2.0 * beta * beta  // Bias toward low
						} else {
							1.0 - 2.0 * (1.0 - beta) * (1.0 - beta)  // Bias toward high
						};
						
						let close_position = close_factor * 0.5 + beta_factor * 0.5;
						let c = l + (h - l) * ((close_position + 1.0) / 2.0);
						
						high.push(h);
						low.push(l);
						close.push(c.clamp(l, h));
					}
				}
				
				let ma_type = if use_ema { "ema" } else { "sma" };
				
				(high, low, close, fastk_period, slowk_period, slowd_period, ma_type.to_string(), is_flat)
			});

		proptest::test_runner::TestRunner::default()
			.run(&strat, |(high, low, close, fastk_period, slowk_period, slowd_period, ma_type, is_flat)| {
				let params = StochParams {
					fastk_period: Some(fastk_period),
					slowk_period: Some(slowk_period),
					slowk_ma_type: Some(ma_type.clone()),
					slowd_period: Some(slowd_period),
					slowd_ma_type: Some(ma_type.clone()),
				};
				
				let input = StochInput::from_slices(&high, &low, &close, params.clone());
				
				// Test with specified kernel
				let result = stoch_with_kernel(&input, kernel)?;
				
				// Test kernel consistency with scalar
				let ref_result = stoch_with_kernel(&input, Kernel::Scalar)?;
				
				// Validate output lengths
				prop_assert_eq!(result.k.len(), high.len());
				prop_assert_eq!(result.d.len(), high.len());
				
				// Calculate proper warmup period accounting for cascading MAs
				// fastk needs fastk_period-1, then slowk smoothing adds more, then slowd adds more
				let warmup_k = fastk_period - 1;  // Raw stoch warmup
				let warmup_slowk = if ma_type == "ema" { 0 } else { slowk_period - 1 };  // Additional for K smoothing
				let warmup_slowd = if ma_type == "ema" { 0 } else { slowd_period - 1 };  // Additional for D smoothing
				let expected_warmup = warmup_k.max(warmup_k + warmup_slowk).max(warmup_k + warmup_slowk + warmup_slowd);
				
				// Validate warmup period - initial values should be NaN
				for i in 0..warmup_k.min(high.len()) {
					prop_assert!(
						result.k[i].is_nan(),
						"K[{}] should be NaN during initial warmup but was {}", i, result.k[i]
					);
					prop_assert!(
						result.d[i].is_nan(),
						"D[{}] should be NaN during initial warmup but was {}", i, result.d[i]
					);
				}
				
				// Validate mathematical properties after warmup
				for i in expected_warmup..high.len() {
					let k_val = result.k[i];
					let d_val = result.d[i];
					let ref_k = ref_result.k[i];
					let ref_d = ref_result.d[i];
					
					// Property 1: K and D must be in [0, 100] range (or NaN during extended warmup)
					if !k_val.is_nan() {
						prop_assert!(
							k_val >= -1e-9 && k_val <= 100.0 + 1e-9,
							"K[{}] = {} is outside [0, 100] range", i, k_val
						);
					}
					
					if !d_val.is_nan() {
						prop_assert!(
							d_val >= -1e-9 && d_val <= 100.0 + 1e-9,
							"D[{}] = {} is outside [0, 100] range", i, d_val
						);
					}
					
					// Property 2: Test kernel consistency (different SIMD kernels should produce same results)
					if k_val.is_finite() && ref_k.is_finite() {
						let k_diff = (k_val - ref_k).abs();
						let k_ulp_diff = k_val.to_bits().abs_diff(ref_k.to_bits());
						prop_assert!(
							k_diff <= 1e-9 || k_ulp_diff <= 4,
							"K mismatch at [{}]: {} vs {} (diff={}, ULP={})",
							i, k_val, ref_k, k_diff, k_ulp_diff
						);
					}
					
					if d_val.is_finite() && ref_d.is_finite() {
						let d_diff = (d_val - ref_d).abs();
						let d_ulp_diff = d_val.to_bits().abs_diff(ref_d.to_bits());
						prop_assert!(
							d_diff <= 1e-9 || d_ulp_diff <= 4,
							"D mismatch at [{}]: {} vs {} (diff={}, ULP={})",
							i, d_val, ref_d, d_diff, d_ulp_diff
						);
					}
					
					// Property 3: Special cases validation (relaxed for smoothed values)
					if i >= fastk_period - 1 && !k_val.is_nan() {
						let window_start = i + 1 - fastk_period;
						let window_high = &high[window_start..=i];
						let window_low = &low[window_start..=i];
						
						let max_h = window_high.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
						let min_l = window_low.iter().cloned().fold(f64::INFINITY, f64::min);
						
						// Special case: flat market (high == low)
						if is_flat || (max_h - min_l).abs() < f64::EPSILON {
							// K should be 50 when there's no price range
							prop_assert!(
								(k_val - 50.0).abs() < 1e-6,
								"K[{}] = {} should be 50 in flat market", i, k_val
							);
						} else {
							// For non-flat markets, check extremes with relaxed bounds for smoothing
							// Note: Raw K would be 100/0 at extremes, but smoothing dampens this
							
							// When close equals highest high in the window
							if (close[i] - max_h).abs() < 1e-10 {
								// With slowk_period=1, K should be very close to 100
								// With larger periods, it's smoothed so we relax the bound
								let expected_min = if slowk_period == 1 { 99.0 } else { 85.0 };
								prop_assert!(
									k_val >= expected_min,
									"K[{}] = {} should be >= {} when close equals highest high (slowk_period={})", 
									i, k_val, expected_min, slowk_period
								);
							}
							
							// When close equals lowest low in the window
							if (close[i] - min_l).abs() < 1e-10 {
								// With slowk_period=1, K should be very close to 0
								// With larger periods, it's smoothed so we relax the bound
								let expected_max = if slowk_period == 1 { 1.0 } else { 15.0 };
								prop_assert!(
									k_val <= expected_max,
									"K[{}] = {} should be <= {} when close equals lowest low (slowk_period={})", 
									i, k_val, expected_max, slowk_period
								);
							}
						}
					}
				}
				
				// Property 4: D should be a smoothed version of K
				// After sufficient data points, D should be less volatile than K
				let k_valid: Vec<f64> = result.k.iter().filter(|x| x.is_finite()).copied().collect();
				let d_valid: Vec<f64> = result.d.iter().filter(|x| x.is_finite()).copied().collect();
				
				if k_valid.len() > 10 && d_valid.len() > 10 && !is_flat {
					// Calculate simple variance as a volatility measure
					let k_mean = k_valid.iter().sum::<f64>() / k_valid.len() as f64;
					let d_mean = d_valid.iter().sum::<f64>() / d_valid.len() as f64;
					
					let k_var = k_valid.iter().map(|x| (x - k_mean).powi(2)).sum::<f64>() / k_valid.len() as f64;
					let d_var = d_valid.iter().map(|x| (x - d_mean).powi(2)).sum::<f64>() / d_valid.len() as f64;
					
					// D should generally have lower or equal variance than K (it's smoothed)
					// Tightened tolerance for better sensitivity
					if slowd_period > 1 && k_var > 1e-6 {  // Only test if there's meaningful variance
						prop_assert!(
							d_var <= k_var * 1.01,  // Tightened from 1.1 to 1.01
							"D variance {} should be <= K variance {} (smoothing effect with slowd_period={})",
							d_var, k_var, slowd_period
						);
					}
					
					// Special case: when slowd_period = 1, D should equal K (no additional smoothing)
					if slowd_period == 1 {
						for i in expected_warmup..result.k.len() {
							if result.k[i].is_finite() && result.d[i].is_finite() {
								prop_assert!(
									(result.k[i] - result.d[i]).abs() < 1e-9,
									"When slowd_period=1, D[{}]={} should equal K[{}]={}",
									i, result.d[i], i, result.k[i]
								);
							}
						}
					}
				}
				
				// Property 5: Test that SMA and EMA produce different results
				// Run the same data with opposite MA type to verify difference
				if !is_flat && high.len() > fastk_period + 10 {
					let opposite_ma_type = if ma_type == "sma" { "ema" } else { "sma" };
					let opposite_params = StochParams {
						fastk_period: Some(fastk_period),
						slowk_period: Some(slowk_period),
						slowk_ma_type: Some(opposite_ma_type.to_string()),
						slowd_period: Some(slowd_period),
						slowd_ma_type: Some(opposite_ma_type.to_string()),
					};
					
					let opposite_input = StochInput::from_slices(&high, &low, &close, opposite_params);
					let opposite_result = stoch_with_kernel(&opposite_input, kernel)?;
					
					// Count how many values differ between SMA and EMA
					let mut diff_count = 0;
					let mut total_valid = 0;
					for i in expected_warmup..result.k.len() {
						if result.k[i].is_finite() && opposite_result.k[i].is_finite() {
							total_valid += 1;
							if (result.k[i] - opposite_result.k[i]).abs() > 1e-6 {
								diff_count += 1;
							}
						}
					}
					
					// At least 80% of values should differ between SMA and EMA
					// (allows for some similar values during flat periods)
					if total_valid > 10 && slowk_period > 1 {
						let diff_ratio = diff_count as f64 / total_valid as f64;
						prop_assert!(
							diff_ratio >= 0.8,
							"SMA and EMA should produce different results: only {}/{} values differ ({}%)",
							diff_count, total_valid, (diff_ratio * 100.0) as i32
						);
					}
				}
				
				Ok(())
			})?;

		Ok(())
	}

	macro_rules! generate_all_stoch_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $( #[test] fn [<$test_fn _scalar_f64>]() { let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar); } )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $( #[test] fn [<$test_fn _avx2_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2); } )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $( #[test] fn [<$test_fn _avx512_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512); } )*
            }
        }
    }
	generate_all_stoch_tests!(
		check_stoch_partial_params,
		check_stoch_accuracy,
		check_stoch_default_candles,
		check_stoch_zero_period,
		check_stoch_period_exceeds_length,
		check_stoch_all_nan,
		check_stoch_no_poison
	);
	
	#[cfg(feature = "proptest")]
	generate_all_stoch_tests!(check_stoch_property);
	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let output = StochBatchBuilder::new().kernel(kernel).apply_candles(&c)?;

		let def = StochParams::default();
		let (row_k, row_d) = output.values_for(&def).expect("default row missing");

		assert_eq!(row_k.len(), c.close.len());
		assert_eq!(row_d.len(), c.close.len());

		let expected_k = [
			42.51122827572717,
			40.13864479593807,
			37.853934778363374,
			37.337021714266086,
			36.26053890551548,
		];
		let expected_d = [
			41.36561869426493,
			41.7691857059163,
			40.16793595000925,
			38.44320042952222,
			37.15049846604803,
		];
		let start = row_k.len() - 5;
		for (i, &v) in row_k[start..].iter().enumerate() {
			assert!(
				(v - expected_k[i]).abs() < 1e-6,
				"[{test}] default-row K mismatch at idx {i}: {v} vs {expected_k:?}"
			);
		}
		for (i, &v) in row_d[start..].iter().enumerate() {
			assert!(
				(v - expected_d[i]).abs() < 1e-6,
				"[{test}] default-row D mismatch at idx {i}: {v} vs {expected_d:?}"
			);
		}
		Ok(())
	}

	#[cfg(debug_assertions)]
	fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		
		// Test various parameter sweep configurations
		let test_configs = vec![
			// (fastk_start, fastk_end, fastk_step, slowk_start, slowk_end, slowk_step, slowd_start, slowd_end, slowd_step)
			(2, 10, 2, 1, 5, 1, 1, 5, 1),      // Small periods with dense steps
			(5, 25, 5, 2, 10, 2, 2, 10, 2),    // Medium periods
			(10, 30, 10, 3, 9, 3, 3, 9, 3),    // Large steps
			(14, 14, 0, 1, 5, 1, 1, 5, 1),     // Static fastk, sweep smoothing
			(2, 5, 1, 3, 3, 0, 3, 3, 0),       // Dense small range, static smoothing
			(20, 50, 15, 5, 15, 5, 5, 15, 5),  // Large periods
			(7, 21, 7, 2, 6, 2, 2, 6, 2),      // Weekly periods
			(3, 12, 3, 1, 3, 1, 1, 3, 1),      // Small range all params
		];
		
		for (cfg_idx, &(fk_start, fk_end, fk_step, sk_start, sk_end, sk_step, sd_start, sd_end, sd_step)) in 
			test_configs.iter().enumerate() 
		{
			let output = StochBatchBuilder::new()
				.kernel(kernel)
				.fastk_period_range(fk_start, fk_end, fk_step)
				.slowk_period_range(sk_start, sk_end, sk_step)
				.slowd_period_range(sd_start, sd_end, sd_step)
				.slowk_ma_type_static("sma")
				.slowd_ma_type_static("sma")
				.apply_candles(&c)?;
			
			// Check K values
			for (idx, &val) in output.k.iter().enumerate() {
				if val.is_nan() {
					continue;
				}
				
				let bits = val.to_bits();
				let row = idx / output.cols;
				let col = idx % output.cols;
				let combo = &output.combos[row];
				
				// Check all three poison patterns with detailed context
				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in K values with params: \
						 fastk_period={}, slowk_period={}, slowd_period={}, \
						 slowk_ma_type={}, slowd_ma_type={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.fastk_period.unwrap_or(14),
						combo.slowk_period.unwrap_or(3),
						combo.slowd_period.unwrap_or(3),
						combo.slowk_ma_type.as_deref().unwrap_or("sma"),
						combo.slowd_ma_type.as_deref().unwrap_or("sma")
					);
				}
				
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in K values with params: \
						 fastk_period={}, slowk_period={}, slowd_period={}, \
						 slowk_ma_type={}, slowd_ma_type={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.fastk_period.unwrap_or(14),
						combo.slowk_period.unwrap_or(3),
						combo.slowd_period.unwrap_or(3),
						combo.slowk_ma_type.as_deref().unwrap_or("sma"),
						combo.slowd_ma_type.as_deref().unwrap_or("sma")
					);
				}
				
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in K values with params: \
						 fastk_period={}, slowk_period={}, slowd_period={}, \
						 slowk_ma_type={}, slowd_ma_type={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.fastk_period.unwrap_or(14),
						combo.slowk_period.unwrap_or(3),
						combo.slowd_period.unwrap_or(3),
						combo.slowk_ma_type.as_deref().unwrap_or("sma"),
						combo.slowd_ma_type.as_deref().unwrap_or("sma")
					);
				}
			}
			
			// Check D values
			for (idx, &val) in output.d.iter().enumerate() {
				if val.is_nan() {
					continue;
				}
				
				let bits = val.to_bits();
				let row = idx / output.cols;
				let col = idx % output.cols;
				let combo = &output.combos[row];
				
				// Check all three poison patterns with detailed context
				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in D values with params: \
						 fastk_period={}, slowk_period={}, slowd_period={}, \
						 slowk_ma_type={}, slowd_ma_type={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.fastk_period.unwrap_or(14),
						combo.slowk_period.unwrap_or(3),
						combo.slowd_period.unwrap_or(3),
						combo.slowk_ma_type.as_deref().unwrap_or("sma"),
						combo.slowd_ma_type.as_deref().unwrap_or("sma")
					);
				}
				
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in D values with params: \
						 fastk_period={}, slowk_period={}, slowd_period={}, \
						 slowk_ma_type={}, slowd_ma_type={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.fastk_period.unwrap_or(14),
						combo.slowk_period.unwrap_or(3),
						combo.slowd_period.unwrap_or(3),
						combo.slowk_ma_type.as_deref().unwrap_or("sma"),
						combo.slowd_ma_type.as_deref().unwrap_or("sma")
					);
				}
				
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in D values with params: \
						 fastk_period={}, slowk_period={}, slowd_period={}, \
						 slowk_ma_type={}, slowd_ma_type={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.fastk_period.unwrap_or(14),
						combo.slowk_period.unwrap_or(3),
						combo.slowd_period.unwrap_or(3),
						combo.slowk_ma_type.as_deref().unwrap_or("sma"),
						combo.slowd_ma_type.as_deref().unwrap_or("sma")
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
