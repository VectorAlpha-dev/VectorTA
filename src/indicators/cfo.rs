//! # Chande Forecast Oscillator (CFO)
//!
//! Calculates `scalar * ((source - LinReg(source, period)) / source)` over a moving window.
//!
//! ## Parameters
//! - **period**: Window size for internal linear regression (default 14)
//! - **scalar**: Multiplier for output (default 100.0)
//!
//! ## Errors
//! - **AllValuesNaN**: All input data values are NaN
//! - **InvalidPeriod**: Period is zero or exceeds data length
//! - **NoData**: Input data is empty
//!
//! ## Returns
//! - `Ok(CfoOutput)` on success (with output vector)
//! - `Err(CfoError)` on failure
//!
//! ## Example
//! ```
//! use ta_indicators::{cfo, CfoInput, CfoParams};
//! let prices = vec![1.0; 20];
//! let params = CfoParams { period: Some(14), scalar: Some(100.0) };
//! let input = CfoInput::from_slice(&prices, params);
//! let output = cfo(&input).expect("Failed to calculate CFO");
//! assert_eq!(output.values.len(), prices.len());
//! ```
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyUntypedArrayMethods};
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

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
	alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
// AVec import removed - using standard Vec and helpers instead
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

// --- Input, Params, Output, Builder, Stream, Batch Structs ---

#[derive(Debug, Clone)]
pub enum CfoData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

impl<'a> AsRef<[f64]> for CfoInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			CfoData::Slice(slice) => slice,
			CfoData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

#[derive(Debug, Clone)]
pub struct CfoOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct CfoParams {
	pub period: Option<usize>,
	pub scalar: Option<f64>,
}

impl Default for CfoParams {
	fn default() -> Self {
		Self {
			period: Some(14),
			scalar: Some(100.0),
		}
	}
}

#[derive(Debug, Clone)]
pub struct CfoInput<'a> {
	pub data: CfoData<'a>,
	pub params: CfoParams,
}

impl<'a> CfoInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: CfoParams) -> Self {
		Self {
			data: CfoData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slice(sl: &'a [f64], p: CfoParams) -> Self {
		Self {
			data: CfoData::Slice(sl),
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", CfoParams::default())
	}
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(14)
	}
	#[inline]
	pub fn get_scalar(&self) -> f64 {
		self.params.scalar.unwrap_or(100.0)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct CfoBuilder {
	period: Option<usize>,
	scalar: Option<f64>,
	kernel: Kernel,
}

impl Default for CfoBuilder {
	fn default() -> Self {
		Self {
			period: None,
			scalar: None,
			kernel: Kernel::Auto,
		}
	}
}

impl CfoBuilder {
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
	pub fn scalar(mut self, x: f64) -> Self {
		self.scalar = Some(x);
		self
	}
	#[inline(always)]
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline(always)]
	pub fn apply(self, c: &Candles) -> Result<CfoOutput, CfoError> {
		let p = CfoParams {
			period: self.period,
			scalar: self.scalar,
		};
		let i = CfoInput::from_candles(c, "close", p);
		cfo_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slice(self, d: &[f64]) -> Result<CfoOutput, CfoError> {
		let p = CfoParams {
			period: self.period,
			scalar: self.scalar,
		};
		let i = CfoInput::from_slice(d, p);
		cfo_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<CfoStream, CfoError> {
		let p = CfoParams {
			period: self.period,
			scalar: self.scalar,
		};
		CfoStream::try_new(p)
	}
}

// --- Error ---

#[derive(Debug, Error)]
pub enum CfoError {
	#[error("cfo: All values are NaN.")]
	AllValuesNaN,
	#[error("cfo: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("cfo: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("cfo: No data provided.")]
	NoData,
}

#[cfg(feature = "wasm")]
impl From<CfoError> for JsValue {
	fn from(err: CfoError) -> Self {
		JsValue::from_str(&err.to_string())
	}
}

// --- Core API ---

#[inline]
pub fn cfo(input: &CfoInput) -> Result<CfoOutput, CfoError> {
	cfo_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn cfo_into_slice(dst: &mut [f64], input: &CfoInput, kernel: Kernel) -> Result<(), CfoError> {
	let data: &[f64] = input.as_ref();
	let len = data.len();
	let period = input.get_period();
	let scalar = input.get_scalar();

	if len == 0 {
		return Err(CfoError::NoData);
	}

	if dst.len() != data.len() {
		return Err(CfoError::InvalidPeriod {
			period: dst.len(),
			data_len: data.len(),
		});
	}

	let first = data.iter().position(|x| !x.is_nan()).ok_or(CfoError::AllValuesNaN)?;

	if period == 0 || period > len {
		return Err(CfoError::InvalidPeriod { period, data_len: len });
	}
	if (len - first) < period {
		return Err(CfoError::NotEnoughValidData {
			needed: period,
			valid: len - first,
		});
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	// Fill warmup period with NaN
	let warmup_end = first + period - 1;
	for v in &mut dst[..warmup_end] {
		*v = f64::NAN;
	}

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => cfo_scalar(data, period, scalar, first, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => cfo_avx2(data, period, scalar, first, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => cfo_avx512(data, period, scalar, first, dst),
			#[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
			Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
				cfo_scalar(data, period, scalar, first, dst)
			}
			_ => unreachable!(),
		}
	}

	Ok(())
}

pub fn cfo_with_kernel(input: &CfoInput, kernel: Kernel) -> Result<CfoOutput, CfoError> {
	let data: &[f64] = input.as_ref();
	let len = data.len();
	let period = input.get_period();
	let scalar = input.get_scalar();

	if len == 0 {
		return Err(CfoError::NoData);
	}

	let first = data.iter().position(|x| !x.is_nan()).ok_or(CfoError::AllValuesNaN)?;

	if period == 0 || period > len {
		return Err(CfoError::InvalidPeriod { period, data_len: len });
	}
	if (len - first) < period {
		return Err(CfoError::NotEnoughValidData {
			needed: period,
			valid: len - first,
		});
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	// Use the same helper as ALMA for NaN prefix allocation
	let mut out = alloc_with_nan_prefix(len, first + period - 1);

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => cfo_scalar(data, period, scalar, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => cfo_avx2(data, period, scalar, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => cfo_avx512(data, period, scalar, first, &mut out),
			#[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
			Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
				// Fallback to scalar when AVX is not available
				cfo_scalar(data, period, scalar, first, &mut out)
			}
			_ => unreachable!(),
		}
	}

	Ok(CfoOutput { values: out })
}

// --- Scalar Logic ---

#[inline]
pub fn cfo_scalar(data: &[f64], period: usize, scalar: f64, first_valid: usize, out: &mut [f64]) {
	let size = data.len();
	let x = (period * (period + 1)) / 2;
	let x2 = (period * (period + 1) * (2 * period + 1)) / 6;
	let x_f = x as f64;
	let x2_f = x2 as f64;
	let period_f = period as f64;
	let bd = 1.0 / (period_f * x2_f - x_f * x_f);

	let mut sum_y = 0.0;
	let mut sum_xy = 0.0;
	for i in 0..(period - 1) {
		let x_i = (i + 1) as f64;
		let val = data[i + first_valid];
		sum_y += val;
		sum_xy += val * x_i;
	}
	for i in (first_valid + period - 1)..size {
		let val = data[i];
		sum_xy += val * period_f;
		sum_y += val;

		let b = (period_f * sum_xy - x_f * sum_y) * bd;
		let a = (sum_y - b * x_f) / period_f;
		let forecast = a + b * period_f;

		if !val.is_nan() {
			out[i] = scalar * (val - forecast) / val;
		}
		sum_xy -= sum_y;
		let oldest_idx = i - (period - 1);
		let oldest_val = data[oldest_idx];
		sum_y -= oldest_val;
	}
}

// --- SIMD Stubs ---

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn cfo_avx512(data: &[f64], period: usize, scalar: f64, first_valid: usize, out: &mut [f64]) {
	if period <= 32 {
		unsafe { cfo_avx512_short(data, period, scalar, first_valid, out) }
	} else {
		unsafe { cfo_avx512_long(data, period, scalar, first_valid, out) }
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn cfo_avx2(data: &[f64], period: usize, scalar: f64, first_valid: usize, out: &mut [f64]) {
	cfo_scalar(data, period, scalar, first_valid, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn cfo_avx512_short(data: &[f64], period: usize, scalar: f64, first_valid: usize, out: &mut [f64]) {
	cfo_scalar(data, period, scalar, first_valid, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn cfo_avx512_long(data: &[f64], period: usize, scalar: f64, first_valid: usize, out: &mut [f64]) {
	cfo_scalar(data, period, scalar, first_valid, out);
}

// --- Row, Batch, Stream, Grid Expansion (Alma-parity) ---

#[inline(always)]
pub fn cfo_row_scalar(
	data: &[f64],
	first: usize,
	period: usize,
	_stride: usize, // unused, kept for API compatibility
	scalar: f64,
	out: &mut [f64],
) {
	cfo_scalar(data, period, scalar, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn cfo_row_avx2(
	data: &[f64],
	first: usize,
	period: usize,
	_stride: usize, // unused, kept for API compatibility
	scalar: f64,
	out: &mut [f64],
) {
	cfo_scalar(data, period, scalar, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn cfo_row_avx512(data: &[f64], first: usize, period: usize, stride: usize, scalar: f64, out: &mut [f64]) {
	if period <= 32 {
		cfo_row_avx512_short(data, first, period, stride, scalar, out)
	} else {
		cfo_row_avx512_long(data, first, period, stride, scalar, out)
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn cfo_row_avx512_short(
	data: &[f64],
	first: usize,
	period: usize,
	_stride: usize, // unused, kept for API compatibility
	scalar: f64,
	out: &mut [f64],
) {
	cfo_scalar(data, period, scalar, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn cfo_row_avx512_long(
	data: &[f64],
	first: usize,
	period: usize,
	_stride: usize, // unused, kept for API compatibility
	scalar: f64,
	out: &mut [f64],
) {
	cfo_scalar(data, period, scalar, first, out)
}

// --- Stream API ---

#[derive(Debug, Clone)]
pub struct CfoStream {
	period: usize,
	scalar: f64,
	buf: Vec<f64>,
	idx: usize,
	filled: bool,
}

impl CfoStream {
	pub fn try_new(params: CfoParams) -> Result<Self, CfoError> {
		let period = params.period.unwrap_or(14);
		if period == 0 {
			return Err(CfoError::InvalidPeriod { period, data_len: 0 });
		}
		let scalar = params.scalar.unwrap_or(100.0);
		Ok(Self {
			period,
			scalar,
			buf: vec![f64::NAN; period],
			idx: 0,
			filled: false,
		})
	}
	#[inline(always)]
	pub fn update(&mut self, value: f64) -> Option<f64> {
		self.buf[self.idx] = value;
		self.idx = (self.idx + 1) % self.period;
		if !self.filled && self.idx == 0 {
			self.filled = true;
		}
		if !self.filled {
			return None;
		}
		Some(self.calc())
	}
	#[inline(always)]
	fn calc(&self) -> f64 {
		let n = self.period;
		let x = (n * (n + 1)) / 2;
		let x2 = (n * (n + 1) * (2 * n + 1)) / 6;
		let x_f = x as f64;
		let x2_f = x2 as f64;
		let period_f = n as f64;
		let bd = 1.0 / (period_f * x2_f - x_f * x_f);

		let mut sum_y = 0.0;
		let mut sum_xy = 0.0;
		let mut idx = self.idx;
		for i in 0..n {
			let v = self.buf[idx];
			sum_y += v;
			sum_xy += v * (i as f64 + 1.0);
			idx = (idx + 1) % n;
		}
		let b = (period_f * sum_xy - x_f * sum_y) * bd;
		let a = (sum_y - b * x_f) / period_f;
		let forecast = a + b * period_f;
		let cur = self.buf[(self.idx + n - 1) % n];
		self.scalar * (cur - forecast) / cur
	}
}

// --- Batch Sweep/Builder API ---

#[derive(Clone, Debug)]
pub struct CfoBatchRange {
	pub period: (usize, usize, usize),
	pub scalar: (f64, f64, f64),
}

impl Default for CfoBatchRange {
	fn default() -> Self {
		Self {
			period: (14, 60, 1),
			scalar: (100.0, 100.0, 0.0), // Keep default at 100.0 for static sweeps
		}
	}
}

#[derive(Clone, Debug, Default)]
pub struct CfoBatchBuilder {
	range: CfoBatchRange,
	kernel: Kernel,
}

impl CfoBatchBuilder {
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
	pub fn scalar_range(mut self, start: f64, end: f64, step: f64) -> Self {
		self.range.scalar = (start, end, step);
		self
	}
	#[inline]
	pub fn scalar_static(mut self, s: f64) -> Self {
		self.range.scalar = (s, s, 0.0);
		self
	}
	pub fn apply_slice(self, data: &[f64]) -> Result<CfoBatchOutput, CfoError> {
		cfo_batch_with_kernel(data, &self.range, self.kernel)
	}
	pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<CfoBatchOutput, CfoError> {
		CfoBatchBuilder::new().kernel(k).apply_slice(data)
	}
	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<CfoBatchOutput, CfoError> {
		let slice = source_type(c, src);
		self.apply_slice(slice)
	}
	pub fn with_default_candles(c: &Candles) -> Result<CfoBatchOutput, CfoError> {
		CfoBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
	}
}

pub fn cfo_batch_with_kernel(data: &[f64], sweep: &CfoBatchRange, k: Kernel) -> Result<CfoBatchOutput, CfoError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(CfoError::InvalidPeriod { period: 0, data_len: 0 }),
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	cfo_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct CfoBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<CfoParams>,
	pub rows: usize,
	pub cols: usize,
}

impl CfoBatchOutput {
	pub fn row_for_params(&self, p: &CfoParams) -> Option<usize> {
		self.combos.iter().position(|c| {
			c.period.unwrap_or(14) == p.period.unwrap_or(14)
				&& (c.scalar.unwrap_or(100.0) - p.scalar.unwrap_or(100.0)).abs() < 1e-12
		})
	}
	pub fn values_for(&self, p: &CfoParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &CfoBatchRange) -> Vec<CfoParams> {
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
	let scalars = axis_f64(r.scalar);
	let mut out = Vec::with_capacity(periods.len() * scalars.len());
	for &p in &periods {
		for &s in &scalars {
			out.push(CfoParams {
				period: Some(p),
				scalar: Some(s),
			});
		}
	}
	out
}

#[inline(always)]
pub fn cfo_batch_slice(data: &[f64], sweep: &CfoBatchRange, kern: Kernel) -> Result<CfoBatchOutput, CfoError> {
	cfo_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn cfo_batch_par_slice(data: &[f64], sweep: &CfoBatchRange, kern: Kernel) -> Result<CfoBatchOutput, CfoError> {
	cfo_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn cfo_batch_inner(
	data: &[f64],
	sweep: &CfoBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<CfoBatchOutput, CfoError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(CfoError::InvalidPeriod { period: 0, data_len: 0 });
	}
	if data.is_empty() {
		return Err(CfoError::NoData);
	}
	let first = data.iter().position(|x| !x.is_nan()).ok_or(CfoError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if data.len() - first < max_p {
		return Err(CfoError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}
	let rows = combos.len();
	let cols = data.len();

	// 1. Allocate *uninitialized* matrix using ALMA helper
	let mut buf_mu = make_uninit_matrix(rows, cols);

	// 2. Fill the NaN warm-up prefix row-wise using ALMA helper
	let warm: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap() - 1).collect();
	init_matrix_prefixes(&mut buf_mu, cols, &warm);

	// 3. Convert &[MaybeUninit<f64>] → &mut [f64]
	let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
	let values: &mut [f64] =
		unsafe { core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len()) };

	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let period = combos[row].period.unwrap();
		let scalar = combos[row].scalar.unwrap();
		match kern {
			Kernel::Scalar => cfo_row_scalar(data, first, period, max_p, scalar, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => cfo_row_avx2(data, first, period, max_p, scalar, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => cfo_row_avx512(data, first, period, max_p, scalar, out_row),
			#[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
			Kernel::Avx2 | Kernel::Avx512 => cfo_row_scalar(data, first, period, max_p, scalar, out_row),
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
	// 4. Reclaim the buffer as a normal Vec<f64> for the return value
	let values = unsafe {
		Vec::from_raw_parts(
			buf_guard.as_mut_ptr() as *mut f64,
			buf_guard.len(),
			buf_guard.capacity(),
		)
	};

	Ok(CfoBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

// New function for Python bindings - writes directly to output buffer
#[inline(always)]
pub fn cfo_batch_inner_into(
	data: &[f64],
	sweep: &CfoBatchRange,
	kern: Kernel,
	parallel: bool,
	output: &mut [f64],
) -> Result<Vec<CfoParams>, CfoError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(CfoError::InvalidPeriod { period: 0, data_len: 0 });
	}
	if data.is_empty() {
		return Err(CfoError::NoData);
	}
	let first = data.iter().position(|x| !x.is_nan()).ok_or(CfoError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if data.len() - first < max_p {
		return Err(CfoError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}
	let rows = combos.len();
	let cols = data.len();

	// Initialize NaN prefixes row-wise using helper
	let warm: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap() - 1).collect();
	for (row, &warmup) in warm.iter().enumerate() {
		let row_start = row * cols;
		let warmup = warmup.min(cols);
		output[row_start..row_start + warmup].fill(f64::NAN);
	}

	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let period = combos[row].period.unwrap();
		let scalar = combos[row].scalar.unwrap();
		match kern {
			Kernel::Scalar => cfo_row_scalar(data, first, period, max_p, scalar, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => cfo_row_avx2(data, first, period, max_p, scalar, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => cfo_row_avx512(data, first, period, max_p, scalar, out_row),
			#[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
			Kernel::Avx2 | Kernel::Avx512 => cfo_row_scalar(data, first, period, max_p, scalar, out_row),
			_ => unreachable!(),
		}
	};

	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			output
				.par_chunks_mut(cols)
				.enumerate()
				.for_each(|(row, slice)| do_row(row, slice));
		}

		#[cfg(target_arch = "wasm32")]
		{
			for (row, slice) in output.chunks_mut(cols).enumerate() {
				do_row(row, slice);
			}
		}
	} else {
		for (row, slice) in output.chunks_mut(cols).enumerate() {
			do_row(row, slice);
		}
	}

	Ok(combos)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_cfo_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let default_params = CfoParams {
			period: None,
			scalar: None,
		};
		let input = CfoInput::from_candles(&candles, "close", default_params);
		let output = cfo_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_cfo_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = CfoInput::from_candles(&candles, "close", CfoParams::default());
		let result = cfo_with_kernel(&input, kernel)?;
		let expected_last_five = [
			0.5998626489475746,
			0.47578011282578453,
			0.20349744599816233,
			0.0919617952835795,
			-0.5676291145560617,
		];
		let start = result.values.len().saturating_sub(5);
		for (i, &val) in result.values[start..].iter().enumerate() {
			let diff = (val - expected_last_five[i]).abs();
			assert!(
				diff < 1e-6,
				"[{}] CFO {:?} mismatch at idx {}: got {}, expected {}",
				test_name,
				kernel,
				i,
				val,
				expected_last_five[i]
			);
		}
		Ok(())
	}

	fn check_cfo_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = CfoInput::with_default_candles(&candles);
		match input.data {
			CfoData::Candles { source, .. } => assert_eq!(source, "close"),
			_ => panic!("Expected CfoData::Candles"),
		}
		let output = cfo_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_cfo_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let input_data = [10.0, 20.0, 30.0];
		let params = CfoParams {
			period: Some(0),
			scalar: Some(100.0),
		};
		let input = CfoInput::from_slice(&input_data, params);
		let res = cfo_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] CFO should fail with zero period", test_name);
		Ok(())
	}

	fn check_cfo_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data_small = [10.0, 20.0, 30.0];
		let params = CfoParams {
			period: Some(10),
			scalar: Some(100.0),
		};
		let input = CfoInput::from_slice(&data_small, params);
		let res = cfo_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] CFO should fail with period exceeding length",
			test_name
		);
		Ok(())
	}

	fn check_cfo_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let single_point = [42.0];
		let params = CfoParams {
			period: Some(14),
			scalar: Some(100.0),
		};
		let input = CfoInput::from_slice(&single_point, params);
		let res = cfo_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] CFO should fail with insufficient data", test_name);
		Ok(())
	}

	fn check_cfo_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let first_params = CfoParams {
			period: Some(14),
			scalar: Some(100.0),
		};
		let first_input = CfoInput::from_candles(&candles, "close", first_params);
		let first_result = cfo_with_kernel(&first_input, kernel)?;

		let second_params = CfoParams {
			period: Some(14),
			scalar: Some(100.0),
		};
		let second_input = CfoInput::from_slice(&first_result.values, second_params);
		let second_result = cfo_with_kernel(&second_input, kernel)?;

		assert_eq!(second_result.values.len(), first_result.values.len());
		for i in 240..second_result.values.len() {
			assert!(
				!second_result.values[i].is_nan(),
				"[{}] Expected no NaN after idx 240, found NaN at {}",
				test_name,
				i
			);
		}
		Ok(())
	}

	fn check_cfo_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let input = CfoInput::from_candles(
			&candles,
			"close",
			CfoParams {
				period: Some(14),
				scalar: Some(100.0),
			},
		);
		let res = cfo_with_kernel(&input, kernel)?;
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

	fn check_cfo_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let period = 14;
		let scalar = 100.0;

		let input = CfoInput::from_candles(
			&candles,
			"close",
			CfoParams {
				period: Some(period),
				scalar: Some(scalar),
			},
		);
		let batch_output = cfo_with_kernel(&input, kernel)?.values;

		let mut stream = CfoStream::try_new(CfoParams {
			period: Some(period),
			scalar: Some(scalar),
		})?;

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
				diff < 1e-9,
				"[{}] CFO streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
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
	fn check_cfo_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		// Test with different parameter combinations to better catch uninitialized reads
		let param_sets = vec![
			CfoParams {
				period: Some(14),
				scalar: Some(100.0),
			},
			CfoParams {
				period: Some(7),
				scalar: Some(50.0),
			},
			CfoParams {
				period: Some(21),
				scalar: Some(200.0),
			},
			CfoParams {
				period: Some(30),
				scalar: Some(100.0),
			},
		];

		for params in param_sets {
			let input = CfoInput::from_candles(&candles, "close", params.clone());
			let output = cfo_with_kernel(&input, kernel)?;

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
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} with params {:?}",
						test_name, val, bits, i, params
					);
				}

				// Check for init_matrix_prefixes poison (0x22222222_22222222)
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} with params {:?}",
						test_name, val, bits, i, params
					);
				}

				// Check for make_uninit_matrix poison (0x33333333_33333333)
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} with params {:?}",
						test_name, val, bits, i, params
					);
				}
			}
		}

		Ok(())
	}

	// Release mode stub - does nothing
	#[cfg(not(debug_assertions))]
	fn check_cfo_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(())
	}

	macro_rules! generate_all_cfo_tests {
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

	generate_all_cfo_tests!(
		check_cfo_partial_params,
		check_cfo_accuracy,
		check_cfo_default_candles,
		check_cfo_zero_period,
		check_cfo_period_exceeds_length,
		check_cfo_very_small_dataset,
		check_cfo_reinput,
		check_cfo_nan_handling,
		check_cfo_streaming,
		check_cfo_no_poison
	);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let output = CfoBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;

		let def = CfoParams::default();
		let row = output.values_for(&def).expect("default row missing");

		assert_eq!(row.len(), c.close.len());

		let expected = [
			0.5998626489475746,
			0.47578011282578453,
			0.20349744599816233,
			0.0919617952835795,
			-0.5676291145560617,
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
					let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto);
				}
			}
		};
	}

	// Check for poison values in batch output - only runs in debug mode
	#[cfg(debug_assertions)]
	fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		// Test batch with multiple parameter combinations
		let output = CfoBatchBuilder::new()
			.kernel(kernel)
			.period_range(5, 30, 5) // 5, 10, 15, 20, 25, 30
			.scalar_range(50.0, 200.0, 50.0) // 50, 100, 150, 200
			.apply_candles(&c, "close")?;

		// Check every value in the entire batch matrix for poison patterns
		for (idx, &val) in output.values.iter().enumerate() {
			// Skip NaN values as they're expected in warmup periods
			if val.is_nan() {
				continue;
			}

			let bits = val.to_bits();
			let row = idx / output.cols;
			let col = idx % output.cols;

			// Check for alloc_with_nan_prefix poison (0x11111111_11111111)
			if bits == 0x11111111_11111111 {
				panic!(
					"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {})",
					test, val, bits, row, col, idx
				);
			}

			// Check for init_matrix_prefixes poison (0x22222222_22222222)
			if bits == 0x22222222_22222222 {
				panic!(
					"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {})",
					test, val, bits, row, col, idx
				);
			}

			// Check for make_uninit_matrix poison (0x33333333_33333333)
			if bits == 0x33333333_33333333 {
				panic!(
					"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {})",
					test, val, bits, row, col, idx
				);
			}
		}

		Ok(())
	}

	// Release mode stub - does nothing
	#[cfg(not(debug_assertions))]
	fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(())
	}

	gen_batch_tests!(check_batch_default_row);
	gen_batch_tests!(check_batch_no_poison);
}

#[cfg(feature = "python")]
#[pyfunction(name = "cfo")]
#[pyo3(signature = (data, period=14, scalar=100.0, kernel=None))]
pub fn cfo_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	period: usize,
	scalar: f64,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
	use numpy::{IntoPyArray, PyArrayMethods};

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, false)?;

	let params = CfoParams {
		period: Some(period),
		scalar: Some(scalar),
	};
	let cfo_in = CfoInput::from_slice(slice_in, params);

	// GOOD: Get Vec<f64> from Rust function
	let result_vec: Vec<f64> = py
		.allow_threads(|| cfo_with_kernel(&cfo_in, kern).map(|o| o.values))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	// GOOD: Zero-copy transfer to NumPy
	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "CfoStream")]
pub struct CfoStreamPy {
	stream: CfoStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl CfoStreamPy {
	#[new]
	fn new(period: usize, scalar: f64) -> PyResult<Self> {
		let params = CfoParams {
			period: Some(period),
			scalar: Some(scalar),
		};
		let stream = CfoStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(CfoStreamPy { stream })
	}

	/// Updates the stream with a new value and returns the calculated CFO value.
	/// Returns `None` if the buffer is not yet full.
	fn update(&mut self, value: f64) -> Option<f64> {
		self.stream.update(value)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "cfo_batch")]
#[pyo3(signature = (data, period_range, scalar_range, kernel=None))]
pub fn cfo_batch_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	period_range: (usize, usize, usize),
	scalar_range: (f64, f64, f64),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, true)?;

	let sweep = CfoBatchRange {
		period: period_range,
		scalar: scalar_range,
	};

	let combos = expand_grid(&sweep);
	let rows = combos.len();
	let cols = slice_in.len();

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
			cfo_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
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
		"scalars",
		combos
			.iter()
			.map(|p| p.scalar.unwrap())
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;

	Ok(dict)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cfo_js(data: &[f64], period: usize, scalar: f64) -> Result<Vec<f64>, JsValue> {
	let params = CfoParams {
		period: Some(period),
		scalar: Some(scalar),
	};
	let input = CfoInput::from_slice(data, params);

	let mut output = vec![0.0; data.len()];  // Single allocation
	cfo_into_slice(&mut output, &input, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;

	Ok(output)
}

// Deprecated - use cfo_batch instead
#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[deprecated(note = "Use cfo_batch instead")]
pub fn cfo_batch_js(
	data: &[f64],
	period_start: usize,
	period_end: usize,
	period_step: usize,
	scalar_start: f64,
	scalar_end: f64,
	scalar_step: f64,
) -> Result<Vec<f64>, JsValue> {
	// Deprecated - use cfo_batch instead
	let sweep = CfoBatchRange {
		period: (period_start, period_end, period_step),
		scalar: (scalar_start, scalar_end, scalar_step),
	};

	// Use the existing batch function with parallel=false for WASM
	cfo_batch_inner(data, &sweep, Kernel::Scalar, false)
		.map(|output| output.values)
		.map_err(|e| JsValue::from_str(&e.to_string()))
}

// Deprecated - use cfo_batch instead
#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[deprecated(note = "Use cfo_batch instead")]
pub fn cfo_batch_metadata_js(
	period_start: usize,
	period_end: usize,
	period_step: usize,
	scalar_start: f64,
	scalar_end: f64,
	scalar_step: f64,
) -> Result<Vec<f64>, JsValue> {
	// Deprecated - use cfo_batch instead
	let sweep = CfoBatchRange {
		period: (period_start, period_end, period_step),
		scalar: (scalar_start, scalar_end, scalar_step),
	};

	let combos = expand_grid(&sweep);
	let mut metadata = Vec::with_capacity(combos.len() * 2);

	for combo in combos {
		metadata.push(combo.period.unwrap() as f64);
		metadata.push(combo.scalar.unwrap());
	}

	Ok(metadata)
}

// New ergonomic WASM API
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct CfoBatchConfig {
	pub period_range: (usize, usize, usize),
	pub scalar_range: (f64, f64, f64),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct CfoBatchJsOutput {
	pub values: Vec<f64>,
	pub combos: Vec<CfoParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cfo_batch(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	// 1. Deserialize the configuration object from JavaScript
	let config: CfoBatchConfig =
		serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

	let sweep = CfoBatchRange {
		period: config.period_range,
		scalar: config.scalar_range,
	};

	// 2. Run the existing core logic
	let output = cfo_batch_inner(data, &sweep, Kernel::Auto, false).map_err(|e| JsValue::from_str(&e.to_string()))?;

	// 3. Create the structured output
	let js_output = CfoBatchJsOutput {
		values: output.values,
		combos: output.combos,
		rows: output.rows,
		cols: output.cols,
	};

	// 4. Serialize the output struct into a JavaScript object
	serde_wasm_bindgen::to_value(&js_output).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

// WASM Memory Management Functions
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cfo_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cfo_free(ptr: *mut f64, len: usize) {
	unsafe {
		let _ = Vec::from_raw_parts(ptr, len, len);
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cfo_into(
	in_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period: usize,
	scalar: f64,
) -> Result<(), JsValue> {
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("null pointer passed to cfo_into"));
	}

	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);

		if period == 0 || period > len {
			return Err(JsValue::from_str("Invalid period"));
		}

		let params = CfoParams {
			period: Some(period),
			scalar: Some(scalar),
		};
		let input = CfoInput::from_slice(data, params);

		if in_ptr == out_ptr {  // CRITICAL: Aliasing check
			let mut temp = vec![0.0; len];
			cfo_into_slice(&mut temp, &input, Kernel::Auto)?;
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			out.copy_from_slice(&temp);
		} else {
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			cfo_into_slice(out, &input, Kernel::Auto)?;
		}
		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cfo_batch_into(
	in_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period_start: usize,
	period_end: usize,
	period_step: usize,
	scalar_start: f64,
	scalar_end: f64,
	scalar_step: f64,
) -> Result<usize, JsValue> {
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("null pointer passed to cfo_batch_into"));
	}

	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);

		let sweep = CfoBatchRange {
			period: (period_start, period_end, period_step),
			scalar: (scalar_start, scalar_end, scalar_step),
		};

		let combos = expand_grid(&sweep);
		let rows = combos.len();
		let cols = len;

		let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);

		cfo_batch_inner_into(data, &sweep, Kernel::Auto, false, out).map_err(|e| JsValue::from_str(&e.to_string()))?;

		Ok(rows)
	}
}
