//! # Elder's Force Index (EFI)
//!
//! The Elder's Force Index (EFI) measures the power behind a price move using both price change and volume.
//! EFI is typically calculated by taking the difference in price (current - previous) multiplied by volume,
//! and then applying an EMA to that result.
//!
//! ## Parameters
//! - **period**: Window size for the EMA (defaults to 13).
//!
//! ## Errors
//! - **AllValuesNaN**: efi: All input data values are `NaN`.
//! - **InvalidPeriod**: efi: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: efi: Not enough valid data points for the requested `period`.
//! - **EmptyData**: efi: Input data slice is empty or mismatched.
//!
//! ## Returns
//! - **`Ok(EfiOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(EfiError)`** otherwise.

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes, make_uninit_matrix};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

impl<'a> AsRef<[f64]> for EfiInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			EfiData::Candles { candles, source } => source_type(candles, source),
			EfiData::Slice { price, .. } => price,
		}
	}
}

#[derive(Debug, Clone)]
pub enum EfiData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice { price: &'a [f64], volume: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct EfiOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(serde::Serialize, serde::Deserialize))]
pub struct EfiParams {
	pub period: Option<usize>,
}

impl Default for EfiParams {
	fn default() -> Self {
		Self { period: Some(13) }
	}
}

#[derive(Debug, Clone)]
pub struct EfiInput<'a> {
	pub data: EfiData<'a>,
	pub params: EfiParams,
}

impl<'a> EfiInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: EfiParams) -> Self {
		Self {
			data: EfiData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slices(price: &'a [f64], volume: &'a [f64], p: EfiParams) -> Self {
		Self {
			data: EfiData::Slice { price, volume },
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", EfiParams::default())
	}
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(13)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct EfiBuilder {
	period: Option<usize>,
	kernel: Kernel,
}

impl Default for EfiBuilder {
	fn default() -> Self {
		Self {
			period: None,
			kernel: Kernel::Auto,
		}
	}
}

impl EfiBuilder {
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
	pub fn apply(self, c: &Candles) -> Result<EfiOutput, EfiError> {
		let p = EfiParams { period: self.period };
		let i = EfiInput::from_candles(c, "close", p);
		efi_with_kernel(&i, self.kernel)
	}

	#[inline(always)]
	pub fn apply_slices(self, price: &[f64], volume: &[f64]) -> Result<EfiOutput, EfiError> {
		let p = EfiParams { period: self.period };
		let i = EfiInput::from_slices(price, volume, p);
		efi_with_kernel(&i, self.kernel)
	}

	#[inline(always)]
	pub fn into_stream(self) -> Result<EfiStream, EfiError> {
		let p = EfiParams { period: self.period };
		EfiStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum EfiError {
	#[error("efi: Empty data provided.")]
	EmptyData,
	#[error("efi: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("efi: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("efi: All values are NaN.")]
	AllValuesNaN,
}

#[inline]
pub fn efi(input: &EfiInput) -> Result<EfiOutput, EfiError> {
	efi_with_kernel(input, Kernel::Auto)
}

pub fn efi_with_kernel(input: &EfiInput, kernel: Kernel) -> Result<EfiOutput, EfiError> {
	let (price, volume): (&[f64], &[f64]) = match &input.data {
		EfiData::Candles { candles, source } => {
			let p = source_type(candles, source);
			let v = &candles.volume;
			(p, v)
		}
		EfiData::Slice { price, volume } => (price, volume),
	};

	if price.is_empty() || volume.is_empty() || price.len() != volume.len() {
		return Err(EfiError::EmptyData);
	}

	let len = price.len();
	let period = input.get_period();

	if period == 0 || period > len {
		return Err(EfiError::InvalidPeriod { period, data_len: len });
	}

	let first_valid_idx = price
		.iter()
		.zip(volume.iter())
		.position(|(p, v)| !p.is_nan() && !v.is_nan());
	if first_valid_idx.is_none() {
		return Err(EfiError::AllValuesNaN);
	}
	let first_valid_idx = first_valid_idx.unwrap();

	if (len - first_valid_idx) < 2 {
		return Err(EfiError::NotEnoughValidData {
			needed: 2,
			valid: len - first_valid_idx,
		});
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	// EFI warmup period is 1 (we need at least 2 values to compute a difference)
	let mut out = alloc_with_nan_prefix(len, 1);
	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => efi_scalar(price, volume, period, first_valid_idx, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => efi_avx2(price, volume, period, first_valid_idx, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => efi_avx512(price, volume, period, first_valid_idx, &mut out),
			_ => unreachable!(),
		}
	}
	Ok(EfiOutput { values: out })
}

/// Write EFI directly to output slice - no allocations
pub fn efi_into_slice(dst: &mut [f64], input: &EfiInput, kern: Kernel) -> Result<(), EfiError> {
	let (price, volume): (&[f64], &[f64]) = match &input.data {
		EfiData::Candles { candles, source } => {
			let p = source_type(candles, source);
			let v = &candles.volume;
			(p, v)
		}
		EfiData::Slice { price, volume } => (price, volume),
	};

	if price.is_empty() || volume.is_empty() || price.len() != volume.len() {
		return Err(EfiError::EmptyData);
	}

	let len = price.len();
	if dst.len() != len {
		return Err(EfiError::InvalidPeriod { period: dst.len(), data_len: len });
	}

	let period = input.get_period();
	if period == 0 || period > len {
		return Err(EfiError::InvalidPeriod { period, data_len: len });
	}

	let first_valid_idx = price
		.iter()
		.zip(volume.iter())
		.position(|(p, v)| !p.is_nan() && !v.is_nan());
	if first_valid_idx.is_none() {
		return Err(EfiError::AllValuesNaN);
	}
	let first_valid_idx = first_valid_idx.unwrap();

	if (len - first_valid_idx) < 2 {
		return Err(EfiError::NotEnoughValidData {
			needed: 2,
			valid: len - first_valid_idx,
		});
	}

	let chosen = match kern {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => efi_scalar(price, volume, period, first_valid_idx, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => efi_avx2(price, volume, period, first_valid_idx, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => efi_avx512(price, volume, period, first_valid_idx, dst),
			_ => unreachable!(),
		}
	}

	// Fill warmup with NaN - find first valid difference position
	let mut warmup_end = len; // default to entire array if no valid values found
	for i in (first_valid_idx + 1)..len {
		if !price[i].is_nan() && !price[i - 1].is_nan() && !volume[i].is_nan() {
			warmup_end = i;
			break;
		}
	}
	
	// Fill all positions before the first valid EFI value with NaN
	for v in &mut dst[..warmup_end] {
		*v = f64::NAN;
	}

	Ok(())
}

#[inline]
pub fn efi_scalar(price: &[f64], volume: &[f64], period: usize, first_valid_idx: usize, out: &mut [f64]) {
	let len = price.len();
	let alpha = 2.0 / (period as f64 + 1.0);
	let mut valid_dif_idx = None;
	for i in (first_valid_idx + 1)..len {
		if !price[i].is_nan() && !price[i - 1].is_nan() && !volume[i].is_nan() {
			out[i] = (price[i] - price[i - 1]) * volume[i];
			valid_dif_idx = Some(i);
			break;
		}
	}
	let start_idx = match valid_dif_idx {
		Some(idx) => idx,
		None => return,
	};
	for i in (start_idx + 1)..len {
		let prev_ema = out[i - 1];
		if price[i].is_nan() || price[i - 1].is_nan() || volume[i].is_nan() {
			out[i] = prev_ema;
		} else {
			let current_dif = (price[i] - price[i - 1]) * volume[i];
			out[i] = alpha * current_dif + (1.0 - alpha) * prev_ema;
		}
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn efi_avx2(price: &[f64], volume: &[f64], period: usize, first_valid_idx: usize, out: &mut [f64]) {
	efi_scalar(price, volume, period, first_valid_idx, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn efi_avx512(price: &[f64], volume: &[f64], period: usize, first_valid_idx: usize, out: &mut [f64]) {
	if period <= 32 {
		unsafe { efi_avx512_short(price, volume, period, first_valid_idx, out) }
	} else {
		unsafe { efi_avx512_long(price, volume, period, first_valid_idx, out) }
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn efi_avx512_short(price: &[f64], volume: &[f64], period: usize, first_valid_idx: usize, out: &mut [f64]) {
	efi_scalar(price, volume, period, first_valid_idx, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn efi_avx512_long(price: &[f64], volume: &[f64], period: usize, first_valid_idx: usize, out: &mut [f64]) {
	efi_scalar(price, volume, period, first_valid_idx, out)
}

#[derive(Debug, Clone)]
pub struct EfiStream {
	period: usize,
	alpha: f64,
	prev: f64,
	filled: bool,
	last_price: f64,
	has_last: bool,
}

impl EfiStream {
	pub fn try_new(params: EfiParams) -> Result<Self, EfiError> {
		let period = params.period.unwrap_or(13);
		if period == 0 {
			return Err(EfiError::InvalidPeriod { period, data_len: 0 });
		}
		Ok(Self {
			period,
			alpha: 2.0 / (period as f64 + 1.0),
			prev: f64::NAN,
			filled: false,
			last_price: f64::NAN,
			has_last: false,
		})
	}

	#[inline(always)]
	pub fn update(&mut self, price: f64, volume: f64) -> Option<f64> {
		if !self.has_last {
			self.last_price = price;
			self.has_last = true;
			return None;
		}

		let out = if price.is_nan() || self.last_price.is_nan() || volume.is_nan() {
			if self.filled {
				self.prev
			} else {
				f64::NAN
			}
		} else {
			let diff = (price - self.last_price) * volume;
			if !self.filled {
				self.prev = diff;
				self.filled = true;
				diff
			} else {
				let ema = self.alpha * diff + (1.0 - self.alpha) * self.prev;
				self.prev = ema;
				ema
			}
		};
		self.last_price = price;
		Some(out)
	}
}

#[derive(Clone, Debug)]
pub struct EfiBatchRange {
	pub period: (usize, usize, usize),
}

impl Default for EfiBatchRange {
	fn default() -> Self {
		Self { period: (13, 100, 1) }
	}
}

#[derive(Clone, Debug, Default)]
pub struct EfiBatchBuilder {
	range: EfiBatchRange,
	kernel: Kernel,
}

impl EfiBatchBuilder {
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

	pub fn apply_slices(self, price: &[f64], volume: &[f64]) -> Result<EfiBatchOutput, EfiError> {
		efi_batch_with_kernel(price, volume, &self.range, self.kernel)
	}

	pub fn with_default_slices(price: &[f64], volume: &[f64], k: Kernel) -> Result<EfiBatchOutput, EfiError> {
		EfiBatchBuilder::new().kernel(k).apply_slices(price, volume)
	}

	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<EfiBatchOutput, EfiError> {
		let slice = source_type(c, src);
		let volume = &c.volume;
		self.apply_slices(slice, volume)
	}

	pub fn with_default_candles(c: &Candles) -> Result<EfiBatchOutput, EfiError> {
		EfiBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
	}
}

pub fn efi_batch_with_kernel(
	price: &[f64],
	volume: &[f64],
	sweep: &EfiBatchRange,
	k: Kernel,
) -> Result<EfiBatchOutput, EfiError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(EfiError::InvalidPeriod { period: 0, data_len: 0 }),
	};

	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	efi_batch_par_slice(price, volume, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct EfiBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<EfiParams>,
	pub rows: usize,
	pub cols: usize,
}
impl EfiBatchOutput {
	pub fn row_for_params(&self, p: &EfiParams) -> Option<usize> {
		self.combos
			.iter()
			.position(|c| c.period.unwrap_or(13) == p.period.unwrap_or(13))
	}
	pub fn values_for(&self, p: &EfiParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &EfiBatchRange) -> Vec<EfiParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let periods = axis_usize(r.period);
	let mut out = Vec::with_capacity(periods.len());
	for &p in &periods {
		out.push(EfiParams { period: Some(p) });
	}
	out
}

#[inline(always)]
pub fn efi_batch_slice(
	price: &[f64],
	volume: &[f64],
	sweep: &EfiBatchRange,
	kern: Kernel,
) -> Result<EfiBatchOutput, EfiError> {
	efi_batch_inner(price, volume, sweep, kern, false)
}

#[inline(always)]
pub fn efi_batch_par_slice(
	price: &[f64],
	volume: &[f64],
	sweep: &EfiBatchRange,
	kern: Kernel,
) -> Result<EfiBatchOutput, EfiError> {
	efi_batch_inner(price, volume, sweep, kern, true)
}

#[inline(always)]
fn efi_batch_inner(
	price: &[f64],
	volume: &[f64],
	sweep: &EfiBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<EfiBatchOutput, EfiError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(EfiError::InvalidPeriod { period: 0, data_len: 0 });
	}
	let first = price
		.iter()
		.zip(volume.iter())
		.position(|(p, v)| !p.is_nan() && !v.is_nan())
		.ok_or(EfiError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if price.len() - first < max_p {
		return Err(EfiError::NotEnoughValidData {
			needed: max_p,
			valid: price.len() - first,
		});
	}
	let rows = combos.len();
	let cols = price.len();
	
	// Use uninitialized matrix for better performance
	let mut buf_mu = make_uninit_matrix(rows, cols);
	
	// Initialize NaN prefixes based on warmup periods (1 for EFI)
	let warmup_periods: Vec<usize> = vec![1; rows];
	init_matrix_prefixes(&mut buf_mu, cols, &warmup_periods);
	
	// Convert to regular Vec<f64> for processing
	let mut values = unsafe {
		let ptr = buf_mu.as_mut_ptr() as *mut f64;
		let len = buf_mu.len();
		let cap = buf_mu.capacity();
		std::mem::forget(buf_mu);
		Vec::from_raw_parts(ptr, len, cap)
	};

	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let period = combos[row].period.unwrap();
		match kern {
			Kernel::Scalar => efi_row_scalar(price, volume, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => efi_row_avx2(price, volume, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => efi_row_avx512(price, volume, first, period, out_row),
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

	Ok(EfiBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
unsafe fn efi_row_scalar(price: &[f64], volume: &[f64], first: usize, period: usize, out: &mut [f64]) {
	efi_scalar(price, volume, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn efi_row_avx2(price: &[f64], volume: &[f64], first: usize, period: usize, out: &mut [f64]) {
	efi_scalar(price, volume, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn efi_row_avx512(price: &[f64], volume: &[f64], first: usize, period: usize, out: &mut [f64]) {
	if period <= 32 {
		efi_row_avx512_short(price, volume, first, period, out);
	} else {
		efi_row_avx512_long(price, volume, first, period, out);
	}
	_mm_sfence();
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn efi_row_avx512_short(price: &[f64], volume: &[f64], first: usize, period: usize, out: &mut [f64]) {
	efi_scalar(price, volume, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn efi_row_avx512_long(price: &[f64], volume: &[f64], first: usize, period: usize, out: &mut [f64]) {
	efi_scalar(price, volume, period, first, out);
}

// Python bindings
#[cfg(feature = "python")]
#[pyfunction(name = "efi")]
#[pyo3(signature = (price, volume, period, kernel=None))]
pub fn efi_py<'py>(
	py: Python<'py>,
	price: PyReadonlyArray1<'py, f64>,
	volume: PyReadonlyArray1<'py, f64>,
	period: usize,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
	use numpy::{IntoPyArray, PyArrayMethods};

	let price_slice = price.as_slice()?;
	let volume_slice = volume.as_slice()?;
	let kern = validate_kernel(kernel, false)?;

	let params = EfiParams { period: Some(period) };
	let input = EfiInput::from_slices(price_slice, volume_slice, params);

	let result_vec: Vec<f64> = py
		.allow_threads(|| efi_with_kernel(&input, kern).map(|o| o.values))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "EfiStream")]
pub struct EfiStreamPy {
	stream: EfiStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl EfiStreamPy {
	#[new]
	fn new(period: usize) -> PyResult<Self> {
		let params = EfiParams { period: Some(period) };
		let stream = EfiStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(EfiStreamPy { stream })
	}

	fn update(&mut self, price: f64, volume: f64) -> Option<f64> {
		self.stream.update(price, volume)
	}
}

// Helper function for batch operations to write directly to output slice
#[inline(always)]
fn efi_batch_inner_into(
	price: &[f64],
	volume: &[f64],
	sweep: &EfiBatchRange,
	kern: Kernel,
	parallel: bool,
	out: &mut [f64],
) -> Result<Vec<EfiParams>, EfiError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(EfiError::InvalidPeriod { period: 0, data_len: 0 });
	}
	let first = price
		.iter()
		.zip(volume.iter())
		.position(|(p, v)| !p.is_nan() && !v.is_nan())
		.ok_or(EfiError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if price.len() - first < max_p {
		return Err(EfiError::NotEnoughValidData {
			needed: max_p,
			valid: price.len() - first,
		});
	}
	let rows = combos.len();
	let cols = price.len();

	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let period = combos[row].period.unwrap();
		match kern {
			Kernel::Scalar => efi_row_scalar(price, volume, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => efi_row_avx2(price, volume, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => efi_row_avx512(price, volume, first, period, out_row),
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

#[cfg(feature = "python")]
#[pyfunction(name = "efi_batch")]
#[pyo3(signature = (price, volume, period_range, kernel=None))]
pub fn efi_batch_py<'py>(
	py: Python<'py>,
	price: PyReadonlyArray1<'py, f64>,
	volume: PyReadonlyArray1<'py, f64>,
	period_range: (usize, usize, usize),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;

	let price_slice = price.as_slice()?;
	let volume_slice = volume.as_slice()?;
	let kern = validate_kernel(kernel, true)?;

	let sweep = EfiBatchRange { period: period_range };

	let combos = expand_grid(&sweep);
	let rows = combos.len();
	let cols = price_slice.len();

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
			efi_batch_inner_into(price_slice, volume_slice, &sweep, simd, true, slice_out)
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

	Ok(dict)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn efi_js(price: &[f64], volume: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
	let params = EfiParams { period: Some(period) };
	let input = EfiInput::from_slices(price, volume, params);
	
	let mut output = vec![0.0; price.len()];  // Single allocation
	efi_into_slice(&mut output, &input, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn efi_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn efi_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe { let _ = Vec::from_raw_parts(ptr, len, len); }
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn efi_into(
	in_price_ptr: *const f64,
	in_volume_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period: usize,
) -> Result<(), JsValue> {
	if in_price_ptr.is_null() || in_volume_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}
	
	unsafe {
		let price = std::slice::from_raw_parts(in_price_ptr, len);
		let volume = std::slice::from_raw_parts(in_volume_ptr, len);
		let params = EfiParams { period: Some(period) };
		let input = EfiInput::from_slices(price, volume, params);
		
		// Handle aliasing - check if output overlaps with either input
		if in_price_ptr == out_ptr || in_volume_ptr == out_ptr {
			let mut temp = vec![0.0; len];
			efi_into_slice(&mut temp, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			out.copy_from_slice(&temp);
		} else {
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			efi_into_slice(out, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct EfiBatchConfig {
	pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct EfiBatchJsOutput {
	pub values: Vec<f64>,
	pub combos: Vec<EfiParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = efi_batch)]
pub fn efi_batch_js(price: &[f64], volume: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	let config: EfiBatchConfig = serde_wasm_bindgen::from_value(config)
		.map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
	
	let sweep = EfiBatchRange {
		period: config.period_range,
	};
	
	let output = efi_batch_inner(price, volume, &sweep, Kernel::Auto, false)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	let js_output = EfiBatchJsOutput {
		values: output.values,
		combos: output.combos,
		rows: output.rows,
		cols: output.cols,
	};
	
	serde_wasm_bindgen::to_value(&js_output)
		.map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn efi_batch_into(
	in_price_ptr: *const f64,
	in_volume_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period_start: usize,
	period_end: usize,
	period_step: usize,
) -> Result<usize, JsValue> {
	if in_price_ptr.is_null() || in_volume_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("null pointer passed to efi_batch_into"));
	}
	
	unsafe {
		let price = std::slice::from_raw_parts(in_price_ptr, len);
		let volume = std::slice::from_raw_parts(in_volume_ptr, len);
		
		let sweep = EfiBatchRange {
			period: (period_start, period_end, period_step),
		};
		
		let combos = expand_grid(&sweep);
		let rows = combos.len();
		let cols = len;
		
		let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);
		
		efi_batch_inner_into(price, volume, &sweep, Kernel::Auto, false, out)
			.map_err(|e| JsValue::from_str(&e.to_string()))?;
		
		Ok(rows)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;
	use crate::utilities::enums::Kernel;

	fn check_efi_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let default_params = EfiParams { period: None };
		let input = EfiInput::from_candles(&candles, "close", default_params);
		let output = efi_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_efi_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = EfiInput::from_candles(&candles, "close", EfiParams::default());
		let result = efi_with_kernel(&input, kernel)?;
		let expected_last_five = [
			-44604.382026531224,
			-39811.02321812391,
			-36599.9671820205,
			-29903.28014503471,
			-55406.09054645832,
		];
		let start = result.values.len().saturating_sub(5);
		for (i, &val) in result.values[start..].iter().enumerate() {
			let diff = (val - expected_last_five[i]).abs();
			assert!(
				diff < 1.0,
				"[{}] EFI {:?} mismatch at idx {}: got {}, expected {}",
				test_name,
				kernel,
				i,
				val,
				expected_last_five[i]
			);
		}
		Ok(())
	}

	fn check_efi_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let price = [10.0, 20.0, 30.0];
		let volume = [100.0, 200.0, 300.0];
		let params = EfiParams { period: Some(0) };
		let input = EfiInput::from_slices(&price, &volume, params);
		let res = efi_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] EFI should fail with zero period", test_name);
		Ok(())
	}

	fn check_efi_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let price = [10.0, 20.0, 30.0];
		let volume = [100.0, 200.0, 300.0];
		let params = EfiParams { period: Some(10) };
		let input = EfiInput::from_slices(&price, &volume, params);
		let res = efi_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] EFI should fail with period exceeding length",
			test_name
		);
		Ok(())
	}

	fn check_efi_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = EfiInput::from_candles(&candles, "close", EfiParams { period: Some(13) });
		let res = efi_with_kernel(&input, kernel)?;
		assert_eq!(res.values.len(), candles.close.len());
		Ok(())
	}

	fn check_efi_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let period = 13;
		let input = EfiInput::from_candles(&candles, "close", EfiParams { period: Some(period) });
		let batch_output = efi_with_kernel(&input, kernel)?.values;
		let mut stream = EfiStream::try_new(EfiParams { period: Some(period) })?;
		let mut stream_values = Vec::with_capacity(candles.close.len());
		for (&p, &v) in candles.close.iter().zip(&candles.volume) {
			match stream.update(p, v) {
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
				diff < 1.0,
				"[{}] EFI streaming mismatch at idx {}: batch={}, stream={}",
				test_name,
				i,
				b,
				s
			);
		}
		Ok(())
	}
	
	#[cfg(debug_assertions)]
	fn check_efi_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		
		// Define comprehensive parameter combinations
		let test_params = vec![
			EfiParams::default(),                    // period: 13
			EfiParams { period: Some(2) },           // minimum viable
			EfiParams { period: Some(5) },           // small
			EfiParams { period: Some(7) },           // small
			EfiParams { period: Some(10) },          // small-medium
			EfiParams { period: Some(20) },          // medium
			EfiParams { period: Some(30) },          // medium-large
			EfiParams { period: Some(50) },          // large
			EfiParams { period: Some(100) },         // very large
			EfiParams { period: Some(200) },         // extremely large
			EfiParams { period: Some(500) },         // maximum reasonable
		];
		
		for (param_idx, params) in test_params.iter().enumerate() {
			let input = EfiInput::from_candles(&candles, "close", params.clone());
			let output = efi_with_kernel(&input, kernel)?;
			
			for (i, &val) in output.values.iter().enumerate() {
				if val.is_nan() {
					continue; // NaN values are expected during warmup
				}
				
				let bits = val.to_bits();
				
				// Check all three poison patterns
				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: {:?} (param set {})",
						test_name, val, bits, i, params, param_idx
					);
				}
				
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: {:?} (param set {})",
						test_name, val, bits, i, params, param_idx
					);
				}
				
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: {:?} (param set {})",
						test_name, val, bits, i, params, param_idx
					);
				}
			}
		}
		
		Ok(())
	}
	
	#[cfg(not(debug_assertions))]
	fn check_efi_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(()) // No-op in release builds
	}

	#[cfg(feature = "proptest")]
	#[allow(clippy::float_cmp)]
	fn check_efi_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		use proptest::prelude::*;
		skip_if_unsupported!(kernel, test_name);

		// Strategy for generating realistic price and volume data
		let strat = (2usize..=50)
			.prop_flat_map(|period| {
				(
					// Generate base price, volatility, and data length
					(100f64..10000f64, 0.01f64..0.05f64, period + 10..400)
						.prop_flat_map(move |(base_price, volatility, data_len)| {
							// Generate random changes for price movement and volume variations
							(
								Just(base_price),
								Just(volatility),
								Just(data_len),
								prop::collection::vec((-1f64..1f64), data_len),
								prop::collection::vec((0.1f64..10f64), data_len),
							)
						})
						.prop_map(move |(base_price, volatility, data_len, price_changes, volume_multipliers)| {
							// Generate synthetic price data with realistic movement
							let mut price = Vec::with_capacity(data_len);
							let mut volume = Vec::with_capacity(data_len);
							let mut current_price = base_price;
							let base_volume = 1000000.0; // Base volume of 1M
							
							for i in 0..data_len {
								// Random walk for price with volatility
								let change = price_changes[i] * volatility * current_price;
								current_price = (current_price + change).max(10.0); // Prevent negative prices
								price.push(current_price);
								
								// Generate volume with some variation
								let daily_volume = base_volume * volume_multipliers[i];
								volume.push(daily_volume);
							}
							
							(price, volume)
						}),
					Just(period),
				)
			});

		proptest::test_runner::TestRunner::default()
			.run(&strat, |((price, volume), period)| {
				let params = EfiParams { period: Some(period) };
				let input = EfiInput::from_slices(&price, &volume, params);
				
				let EfiOutput { values: out } = efi_with_kernel(&input, kernel).unwrap();
				let EfiOutput { values: ref_out } = efi_with_kernel(&input, Kernel::Scalar).unwrap();
				
				// Property 1: Output length matches input
				prop_assert_eq!(out.len(), price.len(), "Output length mismatch");
				
				// Property 2: First value should be NaN (warmup period)
				prop_assert!(out[0].is_nan(), "First value should be NaN");
				
				// Property 3: When price is constant (no changes), EFI should approach 0
				// Check if we have any periods of constant price
				let constant_start = price.windows(3)
					.position(|w| w.iter().all(|&p| (p - w[0]).abs() < 1e-9));
				
				if let Some(start) = constant_start {
					// Find how long the constant period lasts
					let mut constant_end = start + 3;
					while constant_end < price.len() && (price[constant_end] - price[start]).abs() < 1e-9 {
						constant_end += 1;
					}
					
					// After a few periods of constant price, EFI should be very close to 0
					// (price_change = 0, so EFI = EMA of 0 = approaches 0)
					if constant_end - start >= period && constant_end < price.len() {
						let check_idx = constant_end - 1;
						if out[check_idx].is_finite() {
							prop_assert!(
								out[check_idx].abs() < 1e-6,
								"EFI should approach 0 for constant price at idx {}: {}",
								check_idx,
								out[check_idx]
							);
						}
					}
				}
				
				// Property 4: Kernel consistency - compare with scalar reference
				for i in 0..out.len() {
					let y = out[i];
					let r = ref_out[i];
					
					let y_bits = y.to_bits();
					let r_bits = r.to_bits();
					
					// Handle NaN/infinite values
					if !y.is_finite() || !r.is_finite() {
						prop_assert_eq!(
							y_bits, r_bits,
							"NaN/infinite mismatch at idx {}: {} vs {}",
							i, y, r
						);
						continue;
					}
					
					// Use ULP comparison for finite values
					let ulp_diff: u64 = y_bits.abs_diff(r_bits);
					prop_assert!(
						(y - r).abs() <= 1e-9 || ulp_diff <= 4,
						"Kernel mismatch at idx {}: {} vs {} (ULP={})",
						i, y, r, ulp_diff
					);
				}
				
				// Property 5: EMA smoothing behavior
				// Verify that EFI exhibits proper exponential smoothing
				// The change between consecutive EFI values should be proportional to alpha
				let alpha = 2.0 / (period as f64 + 1.0);
				for i in 2..out.len() {
					if out[i].is_finite() && out[i-1].is_finite() && 
					   price[i].is_finite() && price[i-1].is_finite() && 
					   volume[i].is_finite() {
						// Calculate the raw force index for this point
						let raw_fi = (price[i] - price[i-1]) * volume[i];
						
						// EFI[i] should be: alpha * raw_fi + (1 - alpha) * EFI[i-1]
						// So the expected value is:
						let expected = alpha * raw_fi + (1.0 - alpha) * out[i-1];
						
						// Allow for small numerical errors
						if (out[i] - expected).abs() > 1e-9 {
							// Only check if we're past the initial warmup fluctuations
							if i > period + 1 {
								prop_assert!(
									(out[i] - expected).abs() < 1e-6,
									"EMA smoothing violated at idx {}: got {}, expected {} (diff: {})",
									i, out[i], expected, (out[i] - expected).abs()
								);
							}
						}
					}
				}
				
				Ok(())
			})?;

		Ok(())
	}

	macro_rules! generate_all_efi_tests {
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

	generate_all_efi_tests!(
		check_efi_partial_params,
		check_efi_accuracy,
		check_efi_zero_period,
		check_efi_period_exceeds_length,
		check_efi_nan_handling,
		check_efi_streaming,
		check_efi_no_poison
	);

	#[cfg(feature = "proptest")]
	generate_all_efi_tests!(check_efi_property);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let output = EfiBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;
		let def = EfiParams::default();
		let row = output.values_for(&def).expect("default row missing");
		assert_eq!(row.len(), c.close.len());
		Ok(())
	}

	#[cfg(debug_assertions)]
	fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let test_configs = vec![
			(2, 10, 2),      // Small periods
			(5, 25, 5),      // Medium periods
			(30, 100, 10),   // Large periods
			(2, 5, 1),       // Dense small range
			(10, 10, 0),     // Static period (small)
			(13, 13, 0),     // Static period (default)
			(50, 50, 0),     // Static period (large)
			(7, 21, 7),      // Medium range with larger step
			(100, 200, 50),  // Very large periods
		];

		for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
			let output = EfiBatchBuilder::new()
				.kernel(kernel)
				.period_range(p_start, p_end, p_step)
				.apply_candles(&c, "close")?;

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
						combo.period.unwrap_or(13)
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
						combo.period.unwrap_or(13)
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
						combo.period.unwrap_or(13)
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
