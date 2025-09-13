//! # Tilson T3 Moving Average (T3)
//!
//! A specialized moving average that applies multiple iterations of an
//! exponential smoothing algorithm, enhanced by a volume factor (`v_factor`)
//! parameter. API matches alma.rs. SIMD/AVX variants forward to scalar logic by default.
//!
//! ## Parameters
//! - **period**: The look-back period for smoothing (defaults to 5).
//! - **volume_factor**: Controls the depth of the T3 smoothing. Higher values = more smoothing (default 0.0).
//!   The implementation validates that volume_factor is not NaN or infinite.
//!
//! ## Errors
//! - **AllValuesNaN**: tilson: All input data values are `NaN`.
//! - **InvalidPeriod**: tilson: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: tilson: Not enough valid data points for the requested `period`.
//! - **InvalidVolumeFactor**: tilson: `volume_factor` is `NaN` or infinite.
//!
//! ## Returns
//! - **`Ok(TilsonOutput)`** on success, containing a `Vec<f64>` matching the input.
//! - **`Err(TilsonError)`** otherwise.
//!
//! ## Developer Notes
//! - **AVX2/AVX512 kernels**: Currently stubs calling scalar implementation
//! - **Streaming update**: O(1) - maintains 6 EMA cascade states with simple update calculations
//! - **Memory optimization**: Uses `alloc_with_nan_prefix` for zero-copy allocation
//! - **Current status**: Main scalar implementation complete with 6-level EMA cascade
//! - **Optimization opportunities**:
//!   - Implement vectorized AVX2/AVX512 kernels for 6-level EMA cascade
//!   - Consider SIMD for parallel processing of multiple EMA levels
//!   - Optimize coefficient calculations (c1, c2, c3, c4) with vector operations
//!   - Potential for FMA instructions in the weighted sum calculation

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
	alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
// AVec not needed for Tilson since we don't store weights like ALMA
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
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
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

impl<'a> AsRef<[f64]> for TilsonInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			TilsonData::Slice(slice) => slice,
			TilsonData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

#[derive(Debug, Clone)]
pub enum TilsonData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct TilsonOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct TilsonParams {
	pub period: Option<usize>,
	pub volume_factor: Option<f64>,
}

impl Default for TilsonParams {
	fn default() -> Self {
		Self {
			period: Some(5),
			volume_factor: Some(0.0),
		}
	}
}

#[derive(Debug, Clone)]
pub struct TilsonInput<'a> {
	pub data: TilsonData<'a>,
	pub params: TilsonParams,
}

impl<'a> TilsonInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: TilsonParams) -> Self {
		Self {
			data: TilsonData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slice(sl: &'a [f64], p: TilsonParams) -> Self {
		Self {
			data: TilsonData::Slice(sl),
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", TilsonParams::default())
	}
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(5)
	}
	#[inline]
	pub fn get_volume_factor(&self) -> f64 {
		self.params.volume_factor.unwrap_or(0.0)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct TilsonBuilder {
	period: Option<usize>,
	volume_factor: Option<f64>,
	kernel: Kernel,
}

impl Default for TilsonBuilder {
	fn default() -> Self {
		Self {
			period: None,
			volume_factor: None,
			kernel: Kernel::Auto,
		}
	}
}

impl TilsonBuilder {
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
	pub fn volume_factor(mut self, v: f64) -> Self {
		self.volume_factor = Some(v);
		self
	}
	#[inline(always)]
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}

	#[inline(always)]
	pub fn apply(self, c: &Candles) -> Result<TilsonOutput, TilsonError> {
		let p = TilsonParams {
			period: self.period,
			volume_factor: self.volume_factor,
		};
		let i = TilsonInput::from_candles(c, "close", p);
		tilson_with_kernel(&i, self.kernel)
	}

	#[inline(always)]
	pub fn apply_slice(self, d: &[f64]) -> Result<TilsonOutput, TilsonError> {
		let p = TilsonParams {
			period: self.period,
			volume_factor: self.volume_factor,
		};
		let i = TilsonInput::from_slice(d, p);
		tilson_with_kernel(&i, self.kernel)
	}

	#[inline(always)]
	pub fn into_stream(self) -> Result<TilsonStream, TilsonError> {
		let p = TilsonParams {
			period: self.period,
			volume_factor: self.volume_factor,
		};
		TilsonStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum TilsonError {
	#[error("tilson: Input data slice is empty.")]
	EmptyInputData,

	#[error("tilson: All values are NaN.")]
	AllValuesNaN,

	#[error("tilson: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },

	#[error("tilson: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },

	#[error("tilson: Invalid volume factor: {v_factor}")]
	InvalidVolumeFactor { v_factor: f64 },
}

#[inline]
pub fn tilson(input: &TilsonInput) -> Result<TilsonOutput, TilsonError> {
	tilson_with_kernel(input, Kernel::Auto)
}

pub fn tilson_with_kernel(input: &TilsonInput, kernel: Kernel) -> Result<TilsonOutput, TilsonError> {
	let (data, period, v_factor, first, len, chosen) = tilson_prepare(input, kernel)?;
	let lookback_total = 6 * (period - 1);
	let warm = first + lookback_total;

	let mut out = alloc_with_nan_prefix(len, warm);
	tilson_compute_into(data, period, v_factor, first, chosen, &mut out)?;
	Ok(TilsonOutput { values: out })
}

#[inline]
pub fn tilson_into_slice(dst: &mut [f64], input: &TilsonInput, kern: Kernel) -> Result<(), TilsonError> {
	let (data, period, v_factor, first, _len, chosen) = tilson_prepare(input, kern)?;
	if dst.len() != data.len() {
		return Err(TilsonError::InvalidPeriod { period: dst.len(), data_len: data.len() });
	}
	tilson_compute_into(data, period, v_factor, first, chosen, dst)?;
	let warm = first + 6 * (period - 1);
	let warm_end = warm.min(dst.len());
	for v in &mut dst[..warm_end] { *v = f64::NAN; }
	Ok(())
}

#[inline]
pub fn tilson_scalar(
	data: &[f64],
	period: usize,
	v_factor: f64,
	first_valid: usize,
	out: &mut [f64],
) -> Result<(), TilsonError> {
	let len = data.len();
	let lookback_total = 6 * (period - 1);
	debug_assert_eq!(len, out.len());

	if len == 0 { return Err(TilsonError::EmptyInputData); }
	if period == 0 || len - first_valid < period { return Err(TilsonError::InvalidPeriod { period, data_len: len }); }
	if v_factor.is_nan() || v_factor.is_infinite() { return Err(TilsonError::InvalidVolumeFactor { v_factor }); }
	if lookback_total + first_valid >= len {
		return Err(TilsonError::NotEnoughValidData {
			needed: lookback_total + 1,
			valid: len - first_valid,
		});
	}

	let k = 2.0 / (period as f64 + 1.0);
	let one_minus_k = 1.0 - k;

	let temp = v_factor * v_factor;
	let c1 = -(temp * v_factor);
	let c2 = 3.0 * (temp - c1);
	let c3 = -6.0 * temp - 3.0 * (v_factor - c1);
	let c4 = 1.0 + 3.0 * v_factor - c1 + 3.0 * temp;

	let mut today = 0_usize;
	let mut temp_real;
	let mut e1;
	let mut e2;
	let mut e3;
	let mut e4;
	let mut e5;
	let mut e6;

	temp_real = 0.0;
	for i in 0..period {
		temp_real += data[first_valid + today + i];
	}
	e1 = temp_real / (period as f64);
	today += period;

	temp_real = e1;
	for _ in 1..period {
		e1 = k * data[first_valid + today] + one_minus_k * e1;
		temp_real += e1;
		today += 1;
	}
	e2 = temp_real / (period as f64);

	temp_real = e2;
	for _ in 1..period {
		e1 = k * data[first_valid + today] + one_minus_k * e1;
		e2 = k * e1 + one_minus_k * e2;
		temp_real += e2;
		today += 1;
	}
	e3 = temp_real / (period as f64);

	temp_real = e3;
	for _ in 1..period {
		e1 = k * data[first_valid + today] + one_minus_k * e1;
		e2 = k * e1 + one_minus_k * e2;
		e3 = k * e2 + one_minus_k * e3;
		temp_real += e3;
		today += 1;
	}
	e4 = temp_real / (period as f64);

	temp_real = e4;
	for _ in 1..period {
		e1 = k * data[first_valid + today] + one_minus_k * e1;
		e2 = k * e1 + one_minus_k * e2;
		e3 = k * e2 + one_minus_k * e3;
		e4 = k * e3 + one_minus_k * e4;
		temp_real += e4;
		today += 1;
	}
	e5 = temp_real / (period as f64);

	temp_real = e5;
	for _ in 1..period {
		e1 = k * data[first_valid + today] + one_minus_k * e1;
		e2 = k * e1 + one_minus_k * e2;
		e3 = k * e2 + one_minus_k * e3;
		e4 = k * e3 + one_minus_k * e4;
		e5 = k * e4 + one_minus_k * e5;
		temp_real += e5;
		today += 1;
	}
	e6 = temp_real / (period as f64);

	let start_idx = first_valid + lookback_total;
	let end_idx = len - 1;

	let mut idx = start_idx;
	if idx < len {
		out[idx] = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3;
	}
	idx += 1;

	while (first_valid + today) <= end_idx {
		e1 = k * data[first_valid + today] + one_minus_k * e1;
		e2 = k * e1 + one_minus_k * e2;
		e3 = k * e2 + one_minus_k * e3;
		e4 = k * e3 + one_minus_k * e4;
		e5 = k * e4 + one_minus_k * e5;
		e6 = k * e5 + one_minus_k * e6;

		if idx < len {
			out[idx] = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3;
		}

		today += 1;
		idx += 1;
	}

	Ok(())
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn tilson_simd128(
	data: &[f64],
	period: usize,
	v_factor: f64,
	first_valid: usize,
	out: &mut [f64],
) -> Result<(), TilsonError> {
	use core::arch::wasm32::*;
	
	// For now, fall back to scalar implementation
	// TODO: Implement optimized SIMD128 version
	tilson_scalar(data, period, v_factor, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn tilson_avx512(
	data: &[f64],
	period: usize,
	v_factor: f64,
	first_valid: usize,
	out: &mut [f64],
) -> Result<(), TilsonError> {
	tilson_scalar(data, period, v_factor, first_valid, out)
}

#[inline]
pub fn tilson_avx2(
	data: &[f64],
	period: usize,
	v_factor: f64,
	first_valid: usize,
	out: &mut [f64],
) -> Result<(), TilsonError> {
	tilson_scalar(data, period, v_factor, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn tilson_avx512_short(
	data: &[f64],
	period: usize,
	v_factor: f64,
	first_valid: usize,
	out: &mut [f64],
) -> Result<(), TilsonError> {
	tilson_scalar(data, period, v_factor, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn tilson_avx512_long(
	data: &[f64],
	period: usize,
	v_factor: f64,
	first_valid: usize,
	out: &mut [f64],
) -> Result<(), TilsonError> {
	tilson_scalar(data, period, v_factor, first_valid, out)
}

#[inline]
pub fn tilson_batch_with_kernel(
	data: &[f64],
	sweep: &TilsonBatchRange,
	k: Kernel,
) -> Result<TilsonBatchOutput, TilsonError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => {
			return Err(TilsonError::InvalidPeriod { period: 0, data_len: 0 });
		}
	};

	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	tilson_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct TilsonBatchRange {
	pub period: (usize, usize, usize),
	pub volume_factor: (f64, f64, f64),
}

impl Default for TilsonBatchRange {
	fn default() -> Self {
		Self {
			period: (5, 40, 1),
			volume_factor: (0.0, 1.0, 0.1),
		}
	}
}

#[derive(Clone, Debug, Default)]
pub struct TilsonBatchBuilder {
	range: TilsonBatchRange,
	kernel: Kernel,
}

impl TilsonBatchBuilder {
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
	pub fn volume_factor_range(mut self, start: f64, end: f64, step: f64) -> Self {
		self.range.volume_factor = (start, end, step);
		self
	}
	#[inline]
	pub fn volume_factor_static(mut self, v: f64) -> Self {
		self.range.volume_factor = (v, v, 0.0);
		self
	}

	pub fn apply_slice(self, data: &[f64]) -> Result<TilsonBatchOutput, TilsonError> {
		tilson_batch_with_kernel(data, &self.range, self.kernel)
	}
	pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<TilsonBatchOutput, TilsonError> {
		TilsonBatchBuilder::new().kernel(k).apply_slice(data)
	}

	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<TilsonBatchOutput, TilsonError> {
		let slice = source_type(c, src);
		self.apply_slice(slice)
	}

	pub fn with_default_candles(c: &Candles) -> Result<TilsonBatchOutput, TilsonError> {
		TilsonBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
	}
}

#[derive(Clone, Debug)]
pub struct TilsonBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<TilsonParams>,
	pub rows: usize,
	pub cols: usize,
}
impl TilsonBatchOutput {
	pub fn row_for_params(&self, p: &TilsonParams) -> Option<usize> {
		self.combos.iter().position(|c| {
			c.period.unwrap_or(5) == p.period.unwrap_or(5)
				&& (c.volume_factor.unwrap_or(0.0) - p.volume_factor.unwrap_or(0.0)).abs() < 1e-12
		})
	}
	pub fn values_for(&self, p: &TilsonParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &TilsonBatchRange) -> Vec<TilsonParams> {
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
	let v_factors = axis_f64(r.volume_factor);

	let mut out = Vec::with_capacity(periods.len() * v_factors.len());
	for &p in &periods {
		for &v in &v_factors {
			out.push(TilsonParams {
				period: Some(p),
				volume_factor: Some(v),
			});
		}
	}
	out
}

#[inline(always)]
pub fn tilson_batch_slice(
	data: &[f64],
	sweep: &TilsonBatchRange,
	kern: Kernel,
) -> Result<TilsonBatchOutput, TilsonError> {
	tilson_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn tilson_batch_par_slice(
	data: &[f64],
	sweep: &TilsonBatchRange,
	kern: Kernel,
) -> Result<TilsonBatchOutput, TilsonError> {
	tilson_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn tilson_batch_inner(
	data: &[f64],
	sweep: &TilsonBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<TilsonBatchOutput, TilsonError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(TilsonError::InvalidPeriod { period: 0, data_len: 0 });
	}

	if data.is_empty() {
		return Err(TilsonError::EmptyInputData);
	}

	let first = data.iter().position(|x| !x.is_nan()).ok_or(TilsonError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if data.len() - first < 6 * (max_p - 1) + 1 {
		return Err(TilsonError::NotEnoughValidData {
			needed: 6 * (max_p - 1) + 1,
			valid: data.len() - first,
		});
	}
	let rows = combos.len();
	let cols = data.len();
	let warm: Vec<usize> = combos.iter().map(|c| first + 6 * (c.period.unwrap() - 1)).collect();

	// ------------- 1. allocate uninitialised & stamp NaN prefixes
	let mut raw = make_uninit_matrix(rows, cols);
	unsafe { init_matrix_prefixes(&mut raw, cols, &warm) };

	// ------------- 2. worker that fills ONE row -------------
	let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
		let period = combos[row].period.unwrap();
		let v_factor = combos[row].volume_factor.unwrap();

		// cast this row to &mut [f64]
		let out_row = core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());
		tilson_row_scalar(data, first, period, v_factor, out_row);
	};

	// ------------- 3. run every row in (parallel) iterator ---
	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			raw.par_chunks_mut(cols)
				.enumerate()
				.for_each(|(row, slice)| do_row(row, slice));
		}

		#[cfg(target_arch = "wasm32")]
		{
			for (row, slice) in raw.chunks_mut(cols).enumerate() {
				do_row(row, slice);
			}
		}
	} else {
		for (row, slice) in raw.chunks_mut(cols).enumerate() {
			do_row(row, slice);
		}
	}

	// ------------- 4. convert to Vec<f64> like alma.rs ----------
	let mut guard = core::mem::ManuallyDrop::new(raw);
	let values = unsafe {
		Vec::from_raw_parts(
			guard.as_mut_ptr() as *mut f64,
			guard.len(),
			guard.capacity(),
		)
	};

	Ok(TilsonBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
pub unsafe fn tilson_row_scalar(data: &[f64], first: usize, period: usize, v_factor: f64, out: &mut [f64]) {
	let _ = tilson_scalar(data, period, v_factor, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn tilson_row_avx2(data: &[f64], first: usize, period: usize, v_factor: f64, out: &mut [f64]) {
	tilson_row_scalar(data, first, period, v_factor, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn tilson_row_avx512(data: &[f64], first: usize, period: usize, v_factor: f64, out: &mut [f64]) {
	tilson_row_scalar(data, first, period, v_factor, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn tilson_row_avx512_short(data: &[f64], first: usize, period: usize, v_factor: f64, out: &mut [f64]) {
	tilson_row_scalar(data, first, period, v_factor, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn tilson_row_avx512_long(data: &[f64], first: usize, period: usize, v_factor: f64, out: &mut [f64]) {
	tilson_row_scalar(data, first, period, v_factor, out);
}

#[derive(Debug, Clone)]
pub struct TilsonStream {
	period: usize,
	v_factor: f64,
	e: [f64; 6],
	k: f64,
	one_minus_k: f64,
	c1: f64,
	c2: f64,
	c3: f64,
	c4: f64,
	lookback_total: usize,
	values_seen: usize,
	initialized: bool,
	warmup_buffer: Vec<f64>,
	warmup_index: usize,
}

impl TilsonStream {
	pub fn try_new(params: TilsonParams) -> Result<Self, TilsonError> {
		let period = params.period.unwrap_or(5);
		let v_factor = params.volume_factor.unwrap_or(0.0);
		if period == 0 {
			return Err(TilsonError::InvalidPeriod { period, data_len: 0 });
		}
		if v_factor.is_nan() || v_factor.is_infinite() {
			return Err(TilsonError::InvalidVolumeFactor { v_factor });
		}
		let lookback_total = 6 * (period - 1);
		let k = 2.0 / (period as f64 + 1.0);
		
		// Pre-calculate T3 coefficients
		let t = v_factor * v_factor;
		let c1 = -(t * v_factor);
		let c2 = 3.0 * (t - c1);
		let c3 = -6.0 * t - 3.0 * (v_factor - c1);
		let c4 = 1.0 + 3.0 * v_factor - c1 + 3.0 * t;
		
		Ok(Self {
			period,
			v_factor,
			e: [0.0; 6],
			k,
			one_minus_k: 1.0 - k,
			c1,
			c2,
			c3,
			c4,
			lookback_total,
			values_seen: 0,
			initialized: false,
			warmup_buffer: Vec::with_capacity(lookback_total + 1),
			warmup_index: 0,
		})
	}

	#[inline(always)]
	pub fn update(&mut self, value: f64) -> Option<f64> {
		// During warmup, collect values
		if self.values_seen < self.lookback_total {
			self.warmup_buffer.push(value);
			self.values_seen += 1;
			return None;
		}
		
		// Initialize after collecting exactly lookback_total values
		if !self.initialized {
			self.initialize_from_warmup();
			self.initialized = true;
			// Now process the current value (the lookback_total+1 th value)
		}
		
		self.values_seen += 1;
		
		// Update the EMA cascade with the new value
		self.e[0] = self.k * value + self.one_minus_k * self.e[0];
		for i in 1..6 {
			self.e[i] = self.k * self.e[i - 1] + self.one_minus_k * self.e[i];
		}
		
		// Calculate and return T3 value
		Some(self.c1 * self.e[5] + self.c2 * self.e[4] + self.c3 * self.e[3] + self.c4 * self.e[2])
	}
	
	fn initialize_from_warmup(&mut self) {
		// Match the scalar kernel's warmup initialization
		// The scalar implementation consumes exactly lookback_total values for warmup
		let period = self.period;
		let k = self.k;
		let one_minus_k = self.one_minus_k;
		
		// We need exactly lookback_total values, which is 6 * (period - 1)
		// This is distributed as: period + 5*(period-1) values
		
		// Initialize e1 as average of first period values
		let mut temp_real = 0.0;
		for i in 0..period {
			temp_real += self.warmup_buffer[self.warmup_index + i];
		}
		self.e[0] = temp_real / (period as f64);
		self.warmup_index += period;
		
		// Initialize e2
		temp_real = self.e[0];
		for j in 1..period {
			if self.warmup_index < self.warmup_buffer.len() {
				self.e[0] = k * self.warmup_buffer[self.warmup_index] + one_minus_k * self.e[0];
				temp_real += self.e[0];
				self.warmup_index += 1;
			}
		}
		self.e[1] = temp_real / (period as f64);
		
		// Initialize e3
		temp_real = self.e[1];
		for j in 1..period {
			if self.warmup_index < self.warmup_buffer.len() {
				self.e[0] = k * self.warmup_buffer[self.warmup_index] + one_minus_k * self.e[0];
				self.e[1] = k * self.e[0] + one_minus_k * self.e[1];
				temp_real += self.e[1];
				self.warmup_index += 1;
			}
		}
		self.e[2] = temp_real / (period as f64);
		
		// Initialize e4
		temp_real = self.e[2];
		for j in 1..period {
			if self.warmup_index < self.warmup_buffer.len() {
				self.e[0] = k * self.warmup_buffer[self.warmup_index] + one_minus_k * self.e[0];
				self.e[1] = k * self.e[0] + one_minus_k * self.e[1];
				self.e[2] = k * self.e[1] + one_minus_k * self.e[2];
				temp_real += self.e[2];
				self.warmup_index += 1;
			}
		}
		self.e[3] = temp_real / (period as f64);
		
		// Initialize e5
		temp_real = self.e[3];
		for j in 1..period {
			if self.warmup_index < self.warmup_buffer.len() {
				self.e[0] = k * self.warmup_buffer[self.warmup_index] + one_minus_k * self.e[0];
				self.e[1] = k * self.e[0] + one_minus_k * self.e[1];
				self.e[2] = k * self.e[1] + one_minus_k * self.e[2];
				self.e[3] = k * self.e[2] + one_minus_k * self.e[3];
				temp_real += self.e[3];
				self.warmup_index += 1;
			}
		}
		self.e[4] = temp_real / (period as f64);
		
		// Initialize e6 
		// At this point we should have used period + 4*(period-1) values
		// The last loop uses the remaining period-1 values to reach lookback_total
		temp_real = self.e[4];
		for j in 1..period {
			if self.warmup_index < self.warmup_buffer.len() {
				self.e[0] = k * self.warmup_buffer[self.warmup_index] + one_minus_k * self.e[0];
				self.e[1] = k * self.e[0] + one_minus_k * self.e[1];
				self.e[2] = k * self.e[1] + one_minus_k * self.e[2];
				self.e[3] = k * self.e[2] + one_minus_k * self.e[3];
				self.e[4] = k * self.e[3] + one_minus_k * self.e[4];
				temp_real += self.e[4];
				self.warmup_index += 1;
			}
		}
		self.e[5] = temp_real / (period as f64);
		
		self.initialized = true;
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_tilson_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let default_params = TilsonParams {
			period: None,
			volume_factor: None,
		};
		let input = TilsonInput::from_candles(&candles, "close", default_params);
		let output = tilson_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_tilson_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let input = TilsonInput::from_candles(&candles, "close", TilsonParams::default());
		let result = tilson_with_kernel(&input, kernel)?;
		let expected_last_five = [
			59304.716332473254,
			59283.56868015526,
			59261.16173577631,
			59240.25895948583,
			59203.544843167765,
		];
		let start = result.values.len().saturating_sub(5);
		for (i, &val) in result.values[start..].iter().enumerate() {
			let diff = (val - expected_last_five[i]).abs();
			assert!(
				diff < 1e-8,
				"[{}] TILSON {:?} mismatch at idx {}: got {}, expected {}",
				test_name,
				kernel,
				i,
				val,
				expected_last_five[i]
			);
		}
		Ok(())
	}

	fn check_tilson_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let input = TilsonInput::with_default_candles(&candles);
		match input.data {
			TilsonData::Candles { source, .. } => assert_eq!(source, "close"),
			_ => panic!("Expected TilsonData::Candles"),
		}
		let output = tilson_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_tilson_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let input_data = [10.0, 20.0, 30.0];
		let params = TilsonParams {
			period: Some(0),
			volume_factor: None,
		};
		let input = TilsonInput::from_slice(&input_data, params);
		let res = tilson_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] TILSON should fail with zero period", test_name);
		Ok(())
	}

	fn check_tilson_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let input_data: [f64; 0] = [];
		let params = TilsonParams {
			period: Some(5),
			volume_factor: Some(0.0),
		};
		let input = TilsonInput::from_slice(&input_data, params);
		let res = tilson_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] TILSON should fail with empty input", test_name);
		if let Err(e) = res {
			assert!(
				matches!(e, TilsonError::EmptyInputData),
				"[{}] Expected EmptyInputData error",
				test_name
			);
		}
		Ok(())
	}

	fn check_tilson_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data_small = [10.0, 20.0, 30.0];
		let params = TilsonParams {
			period: Some(10),
			volume_factor: None,
		};
		let input = TilsonInput::from_slice(&data_small, params);
		let res = tilson_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] TILSON should fail with period exceeding length",
			test_name
		);
		Ok(())
	}

	fn check_tilson_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let single_point = [42.0];
		let params = TilsonParams {
			period: Some(9),
			volume_factor: None,
		};
		let input = TilsonInput::from_slice(&single_point, params);
		let res = tilson_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] TILSON should fail with insufficient data",
			test_name
		);
		Ok(())
	}

	fn check_tilson_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let first_params = TilsonParams {
			period: Some(5),
			volume_factor: None,
		};
		let first_input = TilsonInput::from_candles(&candles, "close", first_params);
		let first_result = tilson_with_kernel(&first_input, kernel)?;

		let second_params = TilsonParams {
			period: Some(3),
			volume_factor: Some(0.7),
		};
		let second_input = TilsonInput::from_slice(&first_result.values, second_params);
		let second_result = tilson_with_kernel(&second_input, kernel)?;

		assert_eq!(second_result.values.len(), first_result.values.len());
		for i in 240..second_result.values.len() {
			assert!(second_result.values[i].is_finite());
		}
		Ok(())
	}

	fn check_tilson_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let input = TilsonInput::from_candles(
			&candles,
			"close",
			TilsonParams {
				period: Some(5),
				volume_factor: Some(0.0),
			},
		);
		let res = tilson_with_kernel(&input, kernel)?;
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

	fn check_tilson_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let period = 5;
		let v_factor = 0.0;

		let input = TilsonInput::from_candles(
			&candles,
			"close",
			TilsonParams {
				period: Some(period),
				volume_factor: Some(v_factor),
			},
		);
		let batch_output = tilson_with_kernel(&input, kernel)?.values;

		let mut stream = TilsonStream::try_new(TilsonParams {
			period: Some(period),
			volume_factor: Some(v_factor),
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
				"[{}] TILSON streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
				test_name,
				i,
				b,
				s,
				diff
			);
		}
		Ok(())
	}

	macro_rules! generate_all_tilson_tests {
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

	// Check for poison values in single output - only runs in debug mode
	#[cfg(debug_assertions)]
	fn check_tilson_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		// Test with multiple parameter combinations to better catch uninitialized memory bugs
		let test_periods = vec![3, 5, 8, 10, 15, 20, 30, 50];
		let test_v_factors = vec![0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0];

		for &period in &test_periods {
			for &v_factor in &test_v_factors {
				let params = TilsonParams {
					period: Some(period),
					volume_factor: Some(v_factor),
				};
				let input = TilsonInput::from_candles(&candles, "close", params);

				// Skip if we don't have enough data for this period
				if candles.close.len() < 6 * (period - 1) + 1 {
					continue;
				}

				let output = match tilson_with_kernel(&input, kernel) {
					Ok(o) => o,
					Err(_) => continue, // Skip if this combination causes an error
				};

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
                            "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} with period {} and v_factor {}",
                            test_name, val, bits, i, period, v_factor
                        );
					}

					// Check for init_matrix_prefixes poison (0x22222222_22222222)
					if bits == 0x22222222_22222222 {
						panic!(
                            "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} with period {} and v_factor {}",
                            test_name, val, bits, i, period, v_factor
                        );
					}

					// Check for make_uninit_matrix poison (0x33333333_33333333)
					if bits == 0x33333333_33333333 {
						panic!(
                            "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} with period {} and v_factor {}",
                            test_name, val, bits, i, period, v_factor
                        );
					}
				}
			}
		}

		Ok(())
	}

	// Release mode stub - does nothing
	#[cfg(not(debug_assertions))]
	fn check_tilson_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(())
	}

	#[cfg(feature = "proptest")]
	#[allow(clippy::float_cmp)]
	fn check_tilson_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		use proptest::prelude::*;
		skip_if_unsupported!(kernel, test_name);

		// Generate test data with appropriate ranges for Tilson
		// Period: 1-30 (reasonable range for T3)
		// Volume factor: 0.0-1.0 (valid range)
		// Data length: Must be at least 6*(period-1)+1 for valid output
		let strat = (1usize..=30)
			.prop_flat_map(|period| {
				// Ensure data is long enough for the warmup period
				let min_len = (6 * period.saturating_sub(1) + 1).max(period);
				(
					prop::collection::vec(
						(-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
						min_len..400,
					),
					Just(period),
					0.0f64..=1.0f64,  // Volume factor range [0, 1]
				)
			});

		proptest::test_runner::TestRunner::default()
			.run(&strat, |(data, period, volume_factor)| {
				let params = TilsonParams {
					period: Some(period),
					volume_factor: Some(volume_factor),
				};
				let input = TilsonInput::from_slice(&data, params);

				// Compute with the specified kernel
				let TilsonOutput { values: out } = tilson_with_kernel(&input, kernel).unwrap();
				// Compute reference with scalar kernel for comparison
				let TilsonOutput { values: ref_out } = tilson_with_kernel(&input, Kernel::Scalar).unwrap();

				// Property 1: Output length must match input
				prop_assert_eq!(out.len(), data.len(), "Output length mismatch");

				// Calculate warmup period (assuming clean input data, no NaN)
				let warmup_end = 6 * (period - 1);

				// Property 2: Warmup period values must be NaN
				for i in 0..warmup_end.min(out.len()) {
					prop_assert!(
						out[i].is_nan(),
						"Expected NaN during warmup at index {}, got {}",
						i,
						out[i]
					);
				}

				// Check if the entire data array is constant (for Property 5)
				let is_constant_data = data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-12);

				// Property 3: Values after warmup should be finite and SIMD-consistent
				for i in warmup_end..data.len() {
					let y = out[i];
					let r = ref_out[i];

					// Property 4: SIMD consistency check
					if y.is_finite() && r.is_finite() {
						let y_bits = y.to_bits();
						let r_bits = r.to_bits();
						let ulp_diff = y_bits.abs_diff(r_bits);

						prop_assert!(
							(y - r).abs() <= 1e-9 || ulp_diff <= 8,  // Allow slightly more ULP for T3's complexity
							"SIMD mismatch at idx {}: {} vs {} (ULP={})",
							i,
							y,
							r,
							ulp_diff
						);
					} else {
						// For non-finite values, require exact bit match
						prop_assert_eq!(
							y.to_bits(),
							r.to_bits(),
							"Non-finite value mismatch at index {}",
							i
						);
					}

					// Property 5: For constant data, output should converge to that constant
					if is_constant_data && i >= warmup_end + period {
						let const_val = data[0];
						prop_assert!(
							(y - const_val).abs() <= 1e-9,
							"Constant data property failed at idx {}: expected {}, got {}",
							i,
							const_val,
							y
						);
					}
				}

				// Property 6: Special case for period=1
				if period == 1 {
					// With period=1, after warmup (which is 0), output should approximately equal input
					// Note: Even with period=1, Tilson applies 6 cascaded EMAs which can accumulate floating-point errors
					for i in 0..data.len() {
						if out[i].is_finite() && data[i].is_finite() {
							// Use relative tolerance for large values
							let tol = (data[i].abs() * 1e-10).max(1e-9);
							prop_assert!(
								(out[i] - data[i]).abs() <= tol,
								"Period=1 property failed at idx {}: expected {}, got {}, diff={}",
								i,
								data[i],
								out[i],
								(out[i] - data[i]).abs()
							);
						}
					}
				}

				// Property 7: Volume factor edge cases
				if volume_factor == 0.0 && warmup_end < data.len() {
					// With volume_factor=0, c1=0, c2=0, c3=0, c4=1
					// This means output = e3 (third EMA)
					// Verify output is finite and reasonable
					for i in warmup_end..data.len() {
						prop_assert!(
							out[i].is_finite(),
							"With volume_factor=0, output should be finite at idx {}",
							i
						);
					}
				}

				// Note: Removed monotonicity check as it's too strict for a 6-cascaded EMA smoothing indicator

				Ok(())
			})
			.unwrap();

		Ok(())
	}

	generate_all_tilson_tests!(
		check_tilson_partial_params,
		check_tilson_accuracy,
		check_tilson_default_candles,
		check_tilson_zero_period,
		check_tilson_empty_input,
		check_tilson_period_exceeds_length,
		check_tilson_very_small_dataset,
		check_tilson_reinput,
		check_tilson_nan_handling,
		check_tilson_streaming,
		check_tilson_no_poison,
		check_tilson_property
	);

	#[test]
	fn test_volume_factor_validation() {
		// Need at least 6*(period-1)+1 = 13 points for period=3
		let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
		
		// Test with volume_factor = 1.5 (outside typical [0,1] range) - should work
		let params1 = TilsonParams {
			period: Some(3),
			volume_factor: Some(1.5),
		};
		let input1 = TilsonInput::from_slice(&data, params1);
		assert!(tilson(&input1).is_ok(), "volume_factor=1.5 should be accepted");
		
		// Test with volume_factor = -0.5 (negative) - should work
		let params2 = TilsonParams {
			period: Some(3),
			volume_factor: Some(-0.5),
		};
		let input2 = TilsonInput::from_slice(&data, params2);
		assert!(tilson(&input2).is_ok(), "volume_factor=-0.5 should be accepted");
		
		// Test with NaN - should be rejected
		let params3 = TilsonParams {
			period: Some(3),
			volume_factor: Some(f64::NAN),
		};
		let input3 = TilsonInput::from_slice(&data, params3);
		assert!(tilson(&input3).is_err(), "volume_factor=NaN should be rejected");
		
		// Test with infinite - should be rejected
		let params4 = TilsonParams {
			period: Some(3),
			volume_factor: Some(f64::INFINITY),
		};
		let input4 = TilsonInput::from_slice(&data, params4);
		assert!(tilson(&input4).is_err(), "volume_factor=INFINITY should be rejected");
	}

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let output = TilsonBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;

		let def = TilsonParams::default();
		let row = output.values_for(&def).expect("default row missing");

		assert_eq!(row.len(), c.close.len());

		let expected = [
			59304.716332473254,
			59283.56868015526,
			59261.16173577631,
			59240.25895948583,
			59203.544843167765,
		];
		let start = row.len() - 5;
		for (i, &v) in row[start..].iter().enumerate() {
			assert!(
				(v - expected[i]).abs() < 1e-8,
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

		// Test multiple batch configurations with different parameter ranges
		let test_configs = vec![
			// (period_start, period_end, period_step, v_factor_start, v_factor_end, v_factor_step)
			(3, 10, 2, 0.0, 0.5, 0.25),  // Small periods, low v_factors
			(5, 20, 5, 0.0, 1.0, 0.2),   // Medium periods, full v_factor range
			(10, 50, 10, 0.3, 0.7, 0.2), // Large periods, mid v_factors
			(20, 40, 10, 0.0, 1.0, 0.5), // Large periods, few v_factors
			(5, 5, 1, 0.0, 1.0, 0.1),    // Single period, many v_factors
			(15, 15, 1, 0.5, 0.5, 0.1),  // Single period, single v_factor
		];

		for (p_start, p_end, p_step, v_start, v_end, v_step) in test_configs {
			let output = TilsonBatchBuilder::new()
				.kernel(kernel)
				.period_range(p_start, p_end, p_step)
				.volume_factor_range(v_start, v_end, v_step)
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
				let params = output.combos.get(row);
				let period = params.map(|p| p.period.unwrap_or(0)).unwrap_or(0);
				let v_factor = params.map(|p| p.volume_factor.unwrap_or(0.0)).unwrap_or(0.0);

				// Check for alloc_with_nan_prefix poison (0x11111111_11111111)
				if bits == 0x11111111_11111111 {
					panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (period {}, v_factor {}, flat index {})",
                        test, val, bits, row, col, period, v_factor, idx
                    );
				}

				// Check for init_matrix_prefixes poison (0x22222222_22222222)
				if bits == 0x22222222_22222222 {
					panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (period {}, v_factor {}, flat index {})",
                        test, val, bits, row, col, period, v_factor, idx
                    );
				}

				// Check for make_uninit_matrix poison (0x33333333_33333333)
				if bits == 0x33333333_33333333 {
					panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (period {}, v_factor {}, flat index {})",
                        test, val, bits, row, col, period, v_factor, idx
                    );
				}
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

// ========== Helper Functions for Bindings ==========

/// Centralized validation and preparation for Tilson calculation
#[inline]
fn tilson_prepare<'a>(
	input: &'a TilsonInput,
	kernel: Kernel,
) -> Result<(&'a [f64], usize, f64, usize, usize, Kernel), TilsonError> {
	let data: &[f64] = input.as_ref();

	if data.is_empty() {
		return Err(TilsonError::EmptyInputData);
	}

	let first = data.iter().position(|x| !x.is_nan()).ok_or(TilsonError::AllValuesNaN)?;
	let len = data.len();
	let period = input.get_period();
	let v_factor = input.get_volume_factor();

	if period == 0 || period > len {
		return Err(TilsonError::InvalidPeriod { period, data_len: len });
	}
	if (len - first) < period {
		return Err(TilsonError::NotEnoughValidData {
			needed: period,
			valid: len - first,
		});
	}
	if v_factor.is_nan() || v_factor.is_infinite() {
		return Err(TilsonError::InvalidVolumeFactor { v_factor });
	}

	let lookback_total = 6 * (period - 1);
	if (len - first) < lookback_total + 1 {
		return Err(TilsonError::NotEnoughValidData {
			needed: lookback_total + 1,
			valid: len - first,
		});
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	Ok((data, period, v_factor, first, len, chosen))
}

/// Compute Tilson directly into pre-allocated output buffer
#[inline]
fn tilson_compute_into(
	data: &[f64],
	period: usize,
	v_factor: f64,
	first: usize,
	chosen: Kernel,
	out: &mut [f64],
) -> Result<(), TilsonError> {
	unsafe {
		#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
		{
			if matches!(chosen, Kernel::Scalar | Kernel::ScalarBatch) {
				tilson_simd128(data, period, v_factor, first, out)?;
				return Ok(());
			}
		}
		
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => tilson_scalar(data, period, v_factor, first, out)?,
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => tilson_avx2(data, period, v_factor, first, out)?,
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => tilson_avx512(data, period, v_factor, first, out)?,
			#[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
			Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
				// Fallback to scalar when AVX is not available
				tilson_scalar(data, period, v_factor, first, out)?
			}
			Kernel::Auto => unreachable!(),
		}
	}
	Ok(())
}

/// Optimized batch calculation that writes directly to pre-allocated buffer
#[inline(always)]
fn tilson_batch_inner_into(
	data: &[f64],
	sweep: &TilsonBatchRange,
	kern: Kernel,
	parallel: bool,
	out: &mut [f64],
) -> Result<Vec<TilsonParams>, TilsonError> {
	// ---------- 0. parameter checks ----------
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(TilsonError::InvalidPeriod { period: 0, data_len: 0 });
	}

	if data.is_empty() {
		return Err(TilsonError::EmptyInputData);
	}

	let first = data.iter().position(|x| !x.is_nan()).ok_or(TilsonError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();

	if data.len() - first < 6 * (max_p - 1) + 1 {
		return Err(TilsonError::NotEnoughValidData {
			needed: 6 * (max_p - 1) + 1,
			valid: data.len() - first,
		});
	}

	// ---------- 1. matrix dimensions ----------
	let rows = combos.len();
	let cols = data.len();

	// ---------- 2. build per-row warm-up lengths ----------
	let warm: Vec<usize> = combos
		.iter()
		.map(|c| (6 * (c.period.unwrap() - 1) + first).min(cols))
		.collect();

	// ---------- 3. reinterpret output slice as MaybeUninit for efficient initialization ----------
	let out_uninit = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len()) };

	unsafe { init_matrix_prefixes(out_uninit, cols, &warm) };

	// ---------- 4. closure that fills ONE row ----------
	let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
		let period = combos[row].period.unwrap();
		let v_factor = combos[row].volume_factor.unwrap();

		// cast this row to &mut [f64] so the row-kernel can write normally
		let out_row = core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

		match kern {
			Kernel::Scalar | Kernel::ScalarBatch => tilson_row_scalar(data, first, period, v_factor, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => tilson_row_avx2(data, first, period, v_factor, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => tilson_row_avx512(data, first, period, v_factor, out_row),
			#[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
			Kernel::Avx2 | Kernel::Avx512 | Kernel::Avx2Batch | Kernel::Avx512Batch => {
				tilson_row_scalar(data, first, period, v_factor, out_row)
			}
			_ => unreachable!(),
		}
	};

	// ---------- 5. run all rows (optionally in parallel) ----------
	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			out_uninit
				.par_chunks_mut(cols)
				.enumerate()
				.for_each(|(row, slice)| do_row(row, slice));
		}
		#[cfg(target_arch = "wasm32")]
		{
			for (row, slice) in out_uninit.chunks_mut(cols).enumerate() {
				do_row(row, slice);
			}
		}
	} else {
		for (row, slice) in out_uninit.chunks_mut(cols).enumerate() {
			do_row(row, slice);
		}
	}

	Ok(combos)
}

// ========== Python Bindings ==========

#[cfg(feature = "python")]
#[pyfunction(name = "tilson")]
#[pyo3(signature = (data, period, volume_factor=None, kernel=None))]
/// Compute the Tilson T3 Moving Average of the input data.
///
/// The Tilson T3 is a moving average with reduced lag achieved through multiple
/// iterations of exponential smoothing.
///
/// Parameters:
/// -----------
/// data : np.ndarray
///     Input data array (float64).
/// period : int
///     Number of data points in the moving average window (must be >= 1).
/// volume_factor : float, optional
///     Controls the depth of T3 smoothing. Range [0.0, 1.0].
///     Default is 0.0. Higher values = more smoothing.
/// kernel : str, optional
///     Computation kernel to use: 'auto', 'scalar', 'avx2', 'avx512'.
///     Default is 'auto' which auto-detects the best available.
///
/// Returns:
/// --------
/// np.ndarray
///     Array of Tilson T3 values, same length as input.
///
/// Raises:
/// -------
/// ValueError
///     If inputs are invalid (period is zero, exceeds data length, etc).
pub fn tilson_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	period: usize,
	volume_factor: Option<f64>,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
	use numpy::{IntoPyArray, PyArrayMethods};

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, false)?;

	let params = TilsonParams {
		period: Some(period),
		volume_factor: volume_factor.or(Some(0.0)),
	};
	let tilson_in = TilsonInput::from_slice(slice_in, params);

	let result_vec: Vec<f64> = py
		.allow_threads(|| tilson_with_kernel(&tilson_in, kern).map(|o| o.values))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "TilsonStream")]
pub struct TilsonStreamPy {
	stream: TilsonStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl TilsonStreamPy {
	#[new]
	fn new(period: usize, volume_factor: Option<f64>) -> PyResult<Self> {
		let params = TilsonParams {
			period: Some(period),
			volume_factor: volume_factor.or(Some(0.0)),
		};
		let stream = TilsonStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(TilsonStreamPy { stream })
	}

	/// Updates the stream with a new value and returns the calculated Tilson value.
	/// Returns `None` if the buffer is not yet full.
	fn update(&mut self, value: f64) -> Option<f64> {
		self.stream.update(value)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "tilson_batch")]
#[pyo3(signature = (data, period_range, volume_factor_range=None, kernel=None))]
/// Compute Tilson T3 for multiple parameter combinations in a single pass.
///
/// Parameters:
/// -----------
/// data : np.ndarray
///     Input data array (float64).
/// period_range : tuple
///     (start, end, step) for period values to compute.
/// volume_factor_range : tuple, optional
///     (start, end, step) for volume_factor values. Default is (0.0, 0.0, 0.0).
/// kernel : str, optional
///     Computation kernel: 'auto', 'scalar', 'avx2', 'avx512'.
///     Default is 'auto' which auto-detects the best available.
///
/// Returns:
/// --------
/// dict
///     Dictionary with 'values' (2D array), 'periods', and 'volume_factors' arrays.
pub fn tilson_batch_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	period_range: (usize, usize, usize),
	volume_factor_range: Option<(f64, f64, f64)>,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, true)?;  // true for batch operations

	let sweep = TilsonBatchRange {
		period: period_range,
		volume_factor: volume_factor_range.unwrap_or((0.0, 0.0, 0.0)),
	};

	// Calculate dimensions
	let combos = expand_grid(&sweep);
	let rows = combos.len();
	let cols = slice_in.len();

	// Pre-allocate output array (OK for batch operations)
	let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let slice_out = unsafe { out_arr.as_slice_mut()? };

	// Compute without GIL
	let combos = py
		.allow_threads(|| {
			// Handle kernel selection for batch operations
			let kernel = match kern {
				Kernel::Auto => detect_best_batch_kernel(),
				k => k,
			};

			// Map batch kernels to regular kernels
			let simd = match kernel {
				Kernel::Avx512Batch => Kernel::Avx512,
				Kernel::Avx2Batch => Kernel::Avx2,
				Kernel::ScalarBatch => Kernel::Scalar,
				_ => unreachable!(),
			};

			tilson_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
		})
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	// Build result dictionary
	let dict = PyDict::new(py);
	dict.set_item("values", out_arr.reshape((rows, cols))?)?;
	dict.set_item(
		"periods",
		combos.iter().map(|p| p.period.unwrap() as u64).collect::<Vec<_>>().into_pyarray(py),
	)?;
	dict.set_item(
		"volume_factors",
		combos.iter().map(|p| p.volume_factor.unwrap()).collect::<Vec<_>>().into_pyarray(py),
	)?;
	Ok(dict)
}

#[cfg(feature = "python")]
#[pyfunction(name = "tilson_into")]
#[pyo3(signature = (data, period, volume_factor=None, kernel=None))]
pub fn tilson_into_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	period: usize,
	volume_factor: Option<f64>,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
	use numpy::{PyArray1, PyArrayMethods};
	let slice_in = data.as_slice()?;
	let out = unsafe { PyArray1::<f64>::new(py, [slice_in.len()], false) };
	let slice_out = unsafe { out.as_slice_mut()? };

	let kern = validate_kernel(kernel, false)?;
	let params = TilsonParams { period: Some(period), volume_factor: Some(volume_factor.unwrap_or(0.0)) };
	let input = TilsonInput::from_slice(slice_in, params);

	py.allow_threads(|| tilson_into_slice(slice_out, &input, kern))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;
	Ok(out)
}

// ========== WASM Bindings ==========

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct TilsonBatchConfig {
	pub period_range: (usize, usize, usize),
	pub volume_factor_range: (f64, f64, f64),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct TilsonBatchJsOutput {
	pub values: Vec<f64>,
	pub combos: Vec<TilsonParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = tilson_js)]
/// Compute the Tilson T3 Moving Average.
///
/// # Arguments
/// * `data` - Input data array
/// * `period` - Period (must be >= 1)
/// * `volume_factor` - Volume factor (0.0 to 1.0), defaults to 0.0
///
/// # Returns
/// Array of Tilson values, same length as input
pub fn tilson_js(data: &[f64], period: usize, volume_factor: f64) -> Result<Vec<f64>, JsValue> {
	let params = TilsonParams {
		period: Some(period),
		volume_factor: Some(volume_factor),
	};
	let input = TilsonInput::from_slice(data, params);
	let mut out = vec![0.0; data.len()];
	tilson_into_slice(&mut out, &input, detect_best_kernel()).map_err(|e| JsValue::from_str(&e.to_string()))?;
	Ok(out)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = tilson_batch_js)]
/// Compute Tilson for multiple parameter combinations in a single pass.
///
/// # Arguments
/// * `data` - Input data array
/// * `period_start`, `period_end`, `period_step` - Period range parameters
/// * `v_factor_start`, `v_factor_end`, `v_factor_step` - Volume factor range parameters
///
/// # Returns
/// Flattened array of values (row-major order)
pub fn tilson_batch_js(
	data: &[f64],
	period_start: usize,
	period_end: usize,
	period_step: usize,
	v_factor_start: f64,
	v_factor_end: f64,
	v_factor_step: f64,
) -> Result<Vec<f64>, JsValue> {
	let sweep = TilsonBatchRange {
		period: (period_start, period_end, period_step),
		volume_factor: (v_factor_start, v_factor_end, v_factor_step),
	};

	// Use ScalarBatch kernel for WASM batch operations
	let output = tilson_batch_with_kernel(data, &sweep, Kernel::ScalarBatch).map_err(|e| JsValue::from_str(&e.to_string()))?;

	Ok(output.values)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = tilson_batch_metadata_js)]
/// Get metadata about batch computation.
///
/// # Arguments
/// * Period and volume factor range parameters (same as tilson_batch_js)
///
/// # Returns
/// Array containing [periods array, volume_factors array] flattened
pub fn tilson_batch_metadata_js(
	period_start: usize,
	period_end: usize,
	period_step: usize,
	v_factor_start: f64,
	v_factor_end: f64,
	v_factor_step: f64,
) -> Vec<f64> {
	let sweep = TilsonBatchRange {
		period: (period_start, period_end, period_step),
		volume_factor: (v_factor_start, v_factor_end, v_factor_step),
	};

	let combos = expand_grid(&sweep);
	let mut result = Vec::with_capacity(combos.len() * 2);

	// First, all periods
	for combo in &combos {
		result.push(combo.period.unwrap() as f64);
	}

	// Then, all volume factors
	for combo in &combos {
		result.push(combo.volume_factor.unwrap());
	}

	result
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = tilson_batch)]
pub fn tilson_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	let config: TilsonBatchConfig =
		serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

	let sweep = TilsonBatchRange {
		period: config.period_range,
		volume_factor: config.volume_factor_range,
	};

	// Use ScalarBatch kernel for WASM batch operations
	let output = tilson_batch_with_kernel(data, &sweep, Kernel::ScalarBatch).map_err(|e| JsValue::from_str(&e.to_string()))?;

	let js_output = TilsonBatchJsOutput {
		values: output.values,
		combos: output.combos,
		rows: output.rows,
		cols: output.cols,
	};

	serde_wasm_bindgen::to_value(&js_output).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn tilson_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn tilson_free(ptr: *mut f64, len: usize) {
	unsafe {
		let _ = Vec::from_raw_parts(ptr, len, len);
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn tilson_into(
	in_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period: usize,
	volume_factor: f64,
) -> Result<(), JsValue> {
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("null pointer passed to tilson_into"));
	}

	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);

		if period == 0 || period > len {
			return Err(JsValue::from_str("Invalid period"));
		}

		let params = TilsonParams {
			period: Some(period),
			volume_factor: Some(volume_factor),
		};
		let input = TilsonInput::from_slice(data, params);

		// Find first non-NaN value
		let first = data.iter().position(|&x| !x.is_nan()).ok_or_else(|| JsValue::from_str("All values are NaN"))?;
		
		// Calculate warmup period
		let warmup = first + 6 * (period - 1);
		
		if in_ptr == out_ptr {
			let mut temp = vec![f64::NAN; len];
			tilson_compute_into(data, period, volume_factor, first, Kernel::Scalar, &mut temp).map_err(|e| JsValue::from_str(&e.to_string()))?;
			std::ptr::copy_nonoverlapping(temp.as_ptr(), out_ptr, len);
		} else {
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			// Initialize warmup period with NaN
			for i in 0..warmup.min(len) {
				out[i] = f64::NAN;
			}
			tilson_compute_into(data, period, volume_factor, first, Kernel::Scalar, out).map_err(|e| JsValue::from_str(&e.to_string()))?;
		}

		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn tilson_batch_into(
	in_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period_start: usize,
	period_end: usize,
	period_step: usize,
	v_factor_start: f64,
	v_factor_end: f64,
	v_factor_step: f64,
) -> Result<usize, JsValue> {
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("null pointer passed to tilson_batch_into"));
	}

	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);

		let sweep = TilsonBatchRange {
			period: (period_start, period_end, period_step),
			volume_factor: (v_factor_start, v_factor_end, v_factor_step),
		};

		let combos = expand_grid(&sweep);
		let rows = combos.len();
		let cols = len;
		let total = rows * cols;

		if total == 0 {
			return Err(JsValue::from_str("Invalid batch configuration"));
		}

		let out = std::slice::from_raw_parts_mut(out_ptr, total);

		tilson_batch_inner_into(data, &sweep, Kernel::Scalar, false, out)
			.map_err(|e| JsValue::from_str(&e.to_string()))?;

		Ok(rows)
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[deprecated(
	since = "1.0.0",
	note = "For weight reuse patterns, use the fast/unsafe API with persistent buffers"
)]
pub struct TilsonContext {
	period: usize,
	c1: f64,
	c2: f64,
	c3: f64,
	c4: f64,
	kernel: Kernel,
	// State for the 6 EMAs
	ema1: f64,
	ema2: f64,
	ema3: f64,
	ema4: f64,
	ema5: f64,
	ema6: f64,
	initialized: bool,
	warmup_count: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[allow(deprecated)]
impl TilsonContext {
	#[wasm_bindgen(constructor)]
	#[deprecated(
		since = "1.0.0",
		note = "For weight reuse patterns, use the fast/unsafe API with persistent buffers"
	)]
	pub fn new(period: usize, volume_factor: f64) -> Result<TilsonContext, JsValue> {
		if period == 0 {
			return Err(JsValue::from_str("Invalid period: 0"));
		}
		if volume_factor.is_nan() || volume_factor.is_infinite() {
			return Err(JsValue::from_str(&format!("Invalid volume factor: {}", volume_factor)));
		}

		let c1 = -volume_factor.powi(3);
		let c2 = 3.0 * volume_factor.powi(2) + 3.0 * volume_factor.powi(3);
		let c3 = -6.0 * volume_factor.powi(2) - 3.0 * volume_factor - 3.0 * volume_factor.powi(3);
		let c4 = 1.0 + 3.0 * volume_factor + volume_factor.powi(3) + 3.0 * volume_factor.powi(2);

		Ok(TilsonContext {
			period,
			c1,
			c2,
			c3,
			c4,
			kernel: Kernel::Scalar,  // Use Scalar for WASM
			ema1: 0.0,
			ema2: 0.0,
			ema3: 0.0,
			ema4: 0.0,
			ema5: 0.0,
			ema6: 0.0,
			initialized: false,
			warmup_count: 0,
		})
	}

	#[wasm_bindgen]
	pub fn update(&mut self, value: f64) -> Option<f64> {
		if value.is_nan() {
			return None;
		}

		let alpha = 2.0 / (self.period as f64 + 1.0);

		if !self.initialized {
			self.ema1 = value;
			self.ema2 = value;
			self.ema3 = value;
			self.ema4 = value;
			self.ema5 = value;
			self.ema6 = value;
			self.initialized = true;
		} else {
			self.ema1 = alpha * value + (1.0 - alpha) * self.ema1;
			self.ema2 = alpha * self.ema1 + (1.0 - alpha) * self.ema2;
			self.ema3 = alpha * self.ema2 + (1.0 - alpha) * self.ema3;
			self.ema4 = alpha * self.ema3 + (1.0 - alpha) * self.ema4;
			self.ema5 = alpha * self.ema4 + (1.0 - alpha) * self.ema5;
			self.ema6 = alpha * self.ema5 + (1.0 - alpha) * self.ema6;
		}

		self.warmup_count += 1;

		// Tilson warmup is 6 * (period - 1)
		if self.warmup_count <= 6 * (self.period - 1) {
			None
		} else {
			Some(self.c1 * self.ema6 + self.c2 * self.ema5 + self.c3 * self.ema4 + self.c4 * self.ema3)
		}
	}

	#[wasm_bindgen]
	pub fn reset(&mut self) {
		self.ema1 = 0.0;
		self.ema2 = 0.0;
		self.ema3 = 0.0;
		self.ema4 = 0.0;
		self.ema5 = 0.0;
		self.ema6 = 0.0;
		self.initialized = false;
		self.warmup_count = 0;
	}

	#[wasm_bindgen]
	pub fn get_warmup_period(&self) -> usize {
		6 * (self.period - 1)
	}
}
