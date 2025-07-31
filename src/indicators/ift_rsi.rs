//! # Inverse Fisher Transform RSI (IFT RSI)
//!
//! Applies Inverse Fisher Transform to a WMA-smoothed RSI series.
//! API closely matches alma.rs for interface, kernels, builders, batch/grid support, and error handling.

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

use crate::indicators::rsi::{rsi, RsiError, RsiInput, RsiParams};
use crate::indicators::wma::{wma, WmaError, WmaInput, WmaParams};
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
use std::error::Error;
use thiserror::Error;

impl<'a> AsRef<[f64]> for IftRsiInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			IftRsiData::Slice(slice) => slice,
			IftRsiData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

#[derive(Debug, Clone)]
pub enum IftRsiData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct IftRsiOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct IftRsiParams {
	pub rsi_period: Option<usize>,
	pub wma_period: Option<usize>,
}

impl Default for IftRsiParams {
	fn default() -> Self {
		Self {
			rsi_period: Some(5),
			wma_period: Some(9),
		}
	}
}

#[derive(Debug, Clone)]
pub struct IftRsiInput<'a> {
	pub data: IftRsiData<'a>,
	pub params: IftRsiParams,
}

impl<'a> IftRsiInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: IftRsiParams) -> Self {
		Self {
			data: IftRsiData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slice(sl: &'a [f64], p: IftRsiParams) -> Self {
		Self {
			data: IftRsiData::Slice(sl),
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", IftRsiParams::default())
	}
	#[inline]
	pub fn get_rsi_period(&self) -> usize {
		self.params.rsi_period.unwrap_or(5)
	}
	#[inline]
	pub fn get_wma_period(&self) -> usize {
		self.params.wma_period.unwrap_or(9)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct IftRsiBuilder {
	rsi_period: Option<usize>,
	wma_period: Option<usize>,
	kernel: Kernel,
}

impl Default for IftRsiBuilder {
	fn default() -> Self {
		Self {
			rsi_period: None,
			wma_period: None,
			kernel: Kernel::Auto,
		}
	}
}

impl IftRsiBuilder {
	#[inline(always)]
	pub fn new() -> Self {
		Self::default()
	}
	#[inline(always)]
	pub fn rsi_period(mut self, n: usize) -> Self {
		self.rsi_period = Some(n);
		self
	}
	#[inline(always)]
	pub fn wma_period(mut self, n: usize) -> Self {
		self.wma_period = Some(n);
		self
	}
	#[inline(always)]
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}

	#[inline(always)]
	pub fn apply(self, c: &Candles) -> Result<IftRsiOutput, IftRsiError> {
		let p = IftRsiParams {
			rsi_period: self.rsi_period,
			wma_period: self.wma_period,
		};
		let i = IftRsiInput::from_candles(c, "close", p);
		ift_rsi_with_kernel(&i, self.kernel)
	}

	#[inline(always)]
	pub fn apply_slice(self, d: &[f64]) -> Result<IftRsiOutput, IftRsiError> {
		let p = IftRsiParams {
			rsi_period: self.rsi_period,
			wma_period: self.wma_period,
		};
		let i = IftRsiInput::from_slice(d, p);
		ift_rsi_with_kernel(&i, self.kernel)
	}

	#[inline(always)]
	pub fn into_stream(self) -> Result<IftRsiStream, IftRsiError> {
		let p = IftRsiParams {
			rsi_period: self.rsi_period,
			wma_period: self.wma_period,
		};
		IftRsiStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum IftRsiError {
	#[error("ift_rsi: No data provided.")]
	EmptyData,
	#[error("ift_rsi: All values are NaN.")]
	AllValuesNaN,
	#[error("ift_rsi: Invalid RSI period {rsi_period} or WMA period {wma_period}, data length = {data_len}.")]
	InvalidPeriod {
		rsi_period: usize,
		wma_period: usize,
		data_len: usize,
	},
	#[error("ift_rsi: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("ift_rsi: RSI calculation error: {0}")]
	RsiCalculationError(String),
	#[error("ift_rsi: WMA calculation error: {0}")]
	WmaCalculationError(String),
}

#[inline]
pub fn ift_rsi(input: &IftRsiInput) -> Result<IftRsiOutput, IftRsiError> {
	ift_rsi_with_kernel(input, Kernel::Auto)
}

pub fn ift_rsi_with_kernel(input: &IftRsiInput, kernel: Kernel) -> Result<IftRsiOutput, IftRsiError> {
	let data: &[f64] = match &input.data {
		IftRsiData::Candles { candles, source } => source_type(candles, source),
		IftRsiData::Slice(sl) => sl,
	};

	if data.is_empty() {
		return Err(IftRsiError::EmptyData);
	}
	let first = data.iter().position(|x| !x.is_nan()).ok_or(IftRsiError::AllValuesNaN)?;
	let len = data.len();
	let rsi_period = input.get_rsi_period();
	let wma_period = input.get_wma_period();
	if rsi_period == 0 || wma_period == 0 || rsi_period > len || wma_period > len {
		return Err(IftRsiError::InvalidPeriod {
			rsi_period,
			wma_period,
			data_len: len,
		});
	}
	let needed = rsi_period.max(wma_period);
	if (len - first) < needed {
		return Err(IftRsiError::NotEnoughValidData {
			needed,
			valid: len - first,
		});
	}
	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	// Calculate warmup period for IFT RSI (RSI warmup + WMA warmup - 1)
	let warmup_period = first + rsi_period + wma_period - 2;
	let mut out = alloc_with_nan_prefix(data.len(), warmup_period);

	// Calculate warmup period: rsi_period + wma_period - 1
	let warmup_period = rsi_period + wma_period - 1;
	let mut out = alloc_with_nan_prefix(len, warmup_period);

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => {
				ift_rsi_scalar(data, rsi_period, wma_period, first, &mut out)?;
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => {
				ift_rsi_avx2(data, rsi_period, wma_period, first, &mut out)?;
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => {
				ift_rsi_avx512(data, rsi_period, wma_period, first, &mut out)?;
			}
			_ => unreachable!(),
		}
	}

	Ok(IftRsiOutput { values: out })
}

#[inline(always)]
fn ift_rsi_compute_into(
	data: &[f64],
	rsi_period: usize,
	wma_period: usize,
	first_valid: usize,
	out: &mut [f64],
) -> Result<(), IftRsiError> {
	let sliced = &data[first_valid..];
	let mut rsi_values = rsi(&RsiInput::from_slice(
		sliced,
		RsiParams {
			period: Some(rsi_period),
		},
	))
	.map_err(|e| IftRsiError::RsiCalculationError(e.to_string()))?
	.values;

	for val in rsi_values.iter_mut() {
		if !val.is_nan() {
			*val = 0.1 * (*val - 50.0);
		}
	}

	let wma_values = wma(&WmaInput::from_slice(
		&rsi_values,
		WmaParams {
			period: Some(wma_period),
		},
	))
	.map_err(|e| IftRsiError::WmaCalculationError(e.to_string()))?
	.values;

	for (i, &w) in wma_values.iter().enumerate() {
		if !w.is_nan() {
			let two_w = 2.0 * w;
			let numerator = two_w * two_w - 1.0;
			let denominator = two_w * two_w + 1.0;
			out[first_valid + i] = numerator / denominator;
		}
	}
	Ok(())
}

#[inline]
pub fn ift_rsi_scalar(
	data: &[f64],
	rsi_period: usize,
	wma_period: usize,
	first_valid: usize,
	out: &mut [f64],
) -> Result<IftRsiOutput, IftRsiError> {
	ift_rsi_compute_into(data, rsi_period, wma_period, first_valid, out)?;
	Ok(IftRsiOutput { values: out.to_vec() })
}

/// Write directly to output slice - no allocations
pub fn ift_rsi_into_slice(
	dst: &mut [f64],
	input: &IftRsiInput,
	kern: Kernel,
) -> Result<(), IftRsiError> {
	let data: &[f64] = match &input.data {
		IftRsiData::Candles { candles, source } => source_type(candles, source),
		IftRsiData::Slice(sl) => sl,
	};

	if data.is_empty() {
		return Err(IftRsiError::EmptyData);
	}

	if dst.len() != data.len() {
		return Err(IftRsiError::InvalidPeriod {
			rsi_period: input.get_rsi_period(),
			wma_period: input.get_wma_period(),
			data_len: data.len(),
		});
	}

	let first = data.iter().position(|x| !x.is_nan()).ok_or(IftRsiError::AllValuesNaN)?;
	let rsi_period = input.get_rsi_period();
	let wma_period = input.get_wma_period();
	
	if rsi_period == 0 || wma_period == 0 || rsi_period > data.len() || wma_period > data.len() {
		return Err(IftRsiError::InvalidPeriod {
			rsi_period,
			wma_period,
			data_len: data.len(),
		});
	}

	// Compute into dst
	match kern {
		Kernel::Scalar | Kernel::ScalarBatch => {
			ift_rsi_compute_into(data, rsi_period, wma_period, first, dst)?;
		}
		#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
		Kernel::Avx2 | Kernel::Avx2Batch => {
			// AVX2 is a stub, use scalar
			ift_rsi_compute_into(data, rsi_period, wma_period, first, dst)?;
		}
		#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
		Kernel::Avx512 | Kernel::Avx512Batch => {
			// AVX512 is a stub, use scalar
			ift_rsi_compute_into(data, rsi_period, wma_period, first, dst)?;
		}
		_ => unreachable!(),
	}
	
	Ok(())
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn ift_rsi_avx512(
	data: &[f64],
	rsi_period: usize,
	wma_period: usize,
	first_valid: usize,
	out: &mut [f64],
) -> Result<IftRsiOutput, IftRsiError> {
	ift_rsi_scalar(data, rsi_period, wma_period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn ift_rsi_avx2(
	data: &[f64],
	rsi_period: usize,
	wma_period: usize,
	first_valid: usize,
	out: &mut [f64],
) -> Result<IftRsiOutput, IftRsiError> {
	ift_rsi_scalar(data, rsi_period, wma_period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn ift_rsi_avx512_short(
	data: &[f64],
	rsi_period: usize,
	wma_period: usize,
	first_valid: usize,
	out: &mut [f64],
) -> Result<IftRsiOutput, IftRsiError> {
	ift_rsi_avx512(data, rsi_period, wma_period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn ift_rsi_avx512_long(
	data: &[f64],
	rsi_period: usize,
	wma_period: usize,
	first_valid: usize,
	out: &mut [f64],
) -> Result<IftRsiOutput, IftRsiError> {
	ift_rsi_avx512(data, rsi_period, wma_period, first_valid, out)
}

#[inline]
pub fn ift_rsi_batch_with_kernel(
	data: &[f64],
	sweep: &IftRsiBatchRange,
	k: Kernel,
) -> Result<IftRsiBatchOutput, IftRsiError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => {
			return Err(IftRsiError::InvalidPeriod {
				rsi_period: 0,
				wma_period: 0,
				data_len: 0,
			})
		}
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	ift_rsi_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct IftRsiBatchRange {
	pub rsi_period: (usize, usize, usize),
	pub wma_period: (usize, usize, usize),
}

impl Default for IftRsiBatchRange {
	fn default() -> Self {
		Self {
			rsi_period: (5, 21, 1),
			wma_period: (9, 21, 1),
		}
	}
}

#[derive(Clone, Debug, Default)]
pub struct IftRsiBatchBuilder {
	range: IftRsiBatchRange,
	kernel: Kernel,
}

impl IftRsiBatchBuilder {
	pub fn new() -> Self {
		Self::default()
	}
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline]
	pub fn rsi_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.rsi_period = (start, end, step);
		self
	}
	#[inline]
	pub fn rsi_period_static(mut self, p: usize) -> Self {
		self.range.rsi_period = (p, p, 0);
		self
	}
	#[inline]
	pub fn wma_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.wma_period = (start, end, step);
		self
	}
	#[inline]
	pub fn wma_period_static(mut self, n: usize) -> Self {
		self.range.wma_period = (n, n, 0);
		self
	}
	pub fn apply_slice(self, data: &[f64]) -> Result<IftRsiBatchOutput, IftRsiError> {
		ift_rsi_batch_with_kernel(data, &self.range, self.kernel)
	}
	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<IftRsiBatchOutput, IftRsiError> {
		let slice = source_type(c, src);
		self.apply_slice(slice)
	}
	pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<IftRsiBatchOutput, IftRsiError> {
		IftRsiBatchBuilder::new().kernel(k).apply_slice(data)
	}
	pub fn with_default_candles(c: &Candles) -> Result<IftRsiBatchOutput, IftRsiError> {
		IftRsiBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
	}
}

#[derive(Clone, Debug)]
pub struct IftRsiBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<IftRsiParams>,
	pub rows: usize,
	pub cols: usize,
}

impl IftRsiBatchOutput {
	pub fn row_for_params(&self, p: &IftRsiParams) -> Option<usize> {
		self.combos.iter().position(|c| {
			c.rsi_period.unwrap_or(5) == p.rsi_period.unwrap_or(5)
				&& c.wma_period.unwrap_or(9) == p.wma_period.unwrap_or(9)
		})
	}

	pub fn values_for(&self, p: &IftRsiParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &IftRsiBatchRange) -> Vec<IftRsiParams> {
	fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let rsi_periods = axis(r.rsi_period);
	let wma_periods = axis(r.wma_period);
	let mut out = Vec::with_capacity(rsi_periods.len() * wma_periods.len());
	for &rsi_p in &rsi_periods {
		for &wma_p in &wma_periods {
			out.push(IftRsiParams {
				rsi_period: Some(rsi_p),
				wma_period: Some(wma_p),
			});
		}
	}
	out
}

#[inline(always)]
pub fn ift_rsi_batch_slice(
	data: &[f64],
	sweep: &IftRsiBatchRange,
	kern: Kernel,
) -> Result<IftRsiBatchOutput, IftRsiError> {
	ift_rsi_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn ift_rsi_batch_par_slice(
	data: &[f64],
	sweep: &IftRsiBatchRange,
	kern: Kernel,
) -> Result<IftRsiBatchOutput, IftRsiError> {
	ift_rsi_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn ift_rsi_batch_inner(
	data: &[f64],
	sweep: &IftRsiBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<IftRsiBatchOutput, IftRsiError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(IftRsiError::InvalidPeriod {
			rsi_period: 0,
			wma_period: 0,
			data_len: 0,
		});
	}
	let first = data.iter().position(|x| !x.is_nan()).ok_or(IftRsiError::AllValuesNaN)?;
	let max_rsi = combos.iter().map(|c| c.rsi_period.unwrap()).max().unwrap();
	let max_wma = combos.iter().map(|c| c.wma_period.unwrap()).max().unwrap();
	let max_p = max_rsi.max(max_wma);
	if data.len() - first < max_p {
		return Err(IftRsiError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}

	let rows = combos.len();
	let cols = data.len();
	
	// Calculate warmup periods for each parameter combination
	let warmup_periods: Vec<usize> = combos
		.iter()
		.map(|c| first + c.rsi_period.unwrap() + c.wma_period.unwrap() - 2)
		.collect();
	
	// Use uninitialized memory with proper NaN prefixes
	let mut buf_mu = make_uninit_matrix(rows, cols);
	init_matrix_prefixes(&mut buf_mu, cols, &warmup_periods);
	
	// Convert to mutable slice for computation
	let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
	let values: &mut [f64] = unsafe { 
		core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len()) 
	};

	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let rsi_p = combos[row].rsi_period.unwrap();
		let wma_p = combos[row].wma_period.unwrap();
		match kern {
			Kernel::Scalar => ift_rsi_row_scalar(data, first, rsi_p, wma_p, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => ift_rsi_row_avx2(data, first, rsi_p, wma_p, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => ift_rsi_row_avx512(data, first, rsi_p, wma_p, out_row),
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

	// Convert back to Vec for output
	let values = unsafe {
		Vec::from_raw_parts(
			buf_guard.as_mut_ptr() as *mut f64,
			buf_guard.len(),
			buf_guard.capacity(),
		)
	};
	
	Ok(IftRsiBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
fn ift_rsi_batch_inner_into(
	data: &[f64],
	sweep: &IftRsiBatchRange,
	kern: Kernel,
	parallel: bool,
	out: &mut [f64],
) -> Result<Vec<IftRsiParams>, IftRsiError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(IftRsiError::InvalidPeriod {
			rsi_period: 0,
			wma_period: 0,
			data_len: 0,
		});
	}
	let first = data.iter().position(|x| !x.is_nan()).ok_or(IftRsiError::AllValuesNaN)?;
	let max_rsi = combos.iter().map(|c| c.rsi_period.unwrap()).max().unwrap();
	let max_wma = combos.iter().map(|c| c.wma_period.unwrap()).max().unwrap();
	let max_p = max_rsi.max(max_wma);
	if data.len() - first < max_p {
		return Err(IftRsiError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}

	let rows = combos.len();
	let cols = data.len();

	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let rsi_p = combos[row].rsi_period.unwrap();
		let wma_p = combos[row].wma_period.unwrap();
		match kern {
			Kernel::Scalar => ift_rsi_row_scalar(data, first, rsi_p, wma_p, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => ift_rsi_row_avx2(data, first, rsi_p, wma_p, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => ift_rsi_row_avx512(data, first, rsi_p, wma_p, out_row),
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
unsafe fn ift_rsi_row_scalar(data: &[f64], first: usize, rsi_period: usize, wma_period: usize, out: &mut [f64]) {
	let sliced = &data[first..];
	let mut rsi_values = match rsi(&RsiInput::from_slice(
		sliced,
		RsiParams {
			period: Some(rsi_period),
		},
	)) {
		Ok(r) => r.values,
		Err(_) => {
			return;
		}
	};
	for val in rsi_values.iter_mut() {
		if !val.is_nan() {
			*val = 0.1 * (*val - 50.0);
		}
	}
	let wma_values = match wma(&WmaInput::from_slice(
		&rsi_values,
		WmaParams {
			period: Some(wma_period),
		},
	)) {
		Ok(w) => w.values,
		Err(_) => {
			return;
		}
	};
	for (i, &w) in wma_values.iter().enumerate() {
		if !w.is_nan() {
			let two_w = 2.0 * w;
			let numerator = two_w * two_w - 1.0;
			let denominator = two_w * two_w + 1.0;
			out[first + i] = numerator / denominator;
		}
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn ift_rsi_row_avx2(data: &[f64], first: usize, rsi_period: usize, wma_period: usize, out: &mut [f64]) {
	ift_rsi_row_scalar(data, first, rsi_period, wma_period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn ift_rsi_row_avx512(data: &[f64], first: usize, rsi_period: usize, wma_period: usize, out: &mut [f64]) {
	if rsi_period.max(wma_period) <= 32 {
		ift_rsi_row_avx512_short(data, first, rsi_period, wma_period, out);
	} else {
		ift_rsi_row_avx512_long(data, first, rsi_period, wma_period, out);
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn ift_rsi_row_avx512_short(data: &[f64], first: usize, rsi_period: usize, wma_period: usize, out: &mut [f64]) {
	ift_rsi_row_scalar(data, first, rsi_period, wma_period, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn ift_rsi_row_avx512_long(data: &[f64], first: usize, rsi_period: usize, wma_period: usize, out: &mut [f64]) {
	ift_rsi_row_scalar(data, first, rsi_period, wma_period, out);
}

#[derive(Debug, Clone)]
pub struct IftRsiStream {
	rsi_period: usize,
	wma_period: usize,
	rsi_buf: Vec<f64>,
	wma_buf: Vec<f64>,
	head: usize,
	filled: bool,
}

impl IftRsiStream {
	pub fn try_new(params: IftRsiParams) -> Result<Self, IftRsiError> {
		let rsi_period = params.rsi_period.unwrap_or(5);
		let wma_period = params.wma_period.unwrap_or(9);
		if rsi_period == 0 || wma_period == 0 {
			return Err(IftRsiError::InvalidPeriod {
				rsi_period,
				wma_period,
				data_len: 0,
			});
		}
		Ok(Self {
			rsi_period,
			wma_period,
			rsi_buf: vec![f64::NAN; rsi_period],
			wma_buf: vec![f64::NAN; wma_period],
			head: 0,
			filled: false,
		})
	}
	pub fn update(&mut self, value: f64) -> Option<f64> {
		self.rsi_buf[self.head % self.rsi_period] = value;
		self.head += 1;
		if self.head < self.rsi_period {
			return None;
		}
		let rsi_slice = &self.rsi_buf;
		let rsi_val = match rsi(&RsiInput::from_slice(
			rsi_slice,
			RsiParams {
				period: Some(self.rsi_period),
			},
		)) {
			Ok(res) => res.values.last().cloned().unwrap_or(f64::NAN),
			Err(_) => return None,
		};
		let v1 = 0.1 * (rsi_val - 50.0);
		self.wma_buf[(self.head - self.rsi_period) % self.wma_period] = v1;
		if self.head < self.rsi_period + self.wma_period - 1 {
			return None;
		}
		let wma_slice = &self.wma_buf;
		let wma_val = match wma(&WmaInput::from_slice(
			wma_slice,
			WmaParams {
				period: Some(self.wma_period),
			},
		)) {
			Ok(res) => res.values.last().cloned().unwrap_or(f64::NAN),
			Err(_) => return None,
		};
		if wma_val.is_nan() {
			None
		} else {
			let two_w = 2.0 * wma_val;
			let numerator = two_w * two_w - 1.0;
			let denominator = two_w * two_w + 1.0;
			Some(numerator / denominator)
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_ift_rsi_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let default_params = IftRsiParams {
			rsi_period: None,
			wma_period: None,
		};
		let input = IftRsiInput::from_candles(&candles, "close", default_params);
		let output = ift_rsi_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_ift_rsi_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = IftRsiInput::from_candles(&candles, "close", IftRsiParams::default());
		let result = ift_rsi_with_kernel(&input, kernel)?;
		let expected_last_five = [
			-0.27763026899967286,
			-0.367418234207824,
			-0.1650156844504996,
			-0.26631220621545837,
			0.28324385010826775,
		];
		let start = result.values.len().saturating_sub(5);
		for (i, &val) in result.values[start..].iter().enumerate() {
			let diff = (val - expected_last_five[i]).abs();
			assert!(
				diff < 1e-8,
				"[{}] IFT_RSI {:?} mismatch at idx {}: got {}, expected {}",
				test_name,
				kernel,
				i,
				val,
				expected_last_five[i]
			);
		}
		Ok(())
	}

	fn check_ift_rsi_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = IftRsiInput::with_default_candles(&candles);
		let output = ift_rsi_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_ift_rsi_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let input_data = [10.0, 20.0, 30.0];
		let params = IftRsiParams {
			rsi_period: Some(0),
			wma_period: Some(9),
		};
		let input = IftRsiInput::from_slice(&input_data, params);
		let res = ift_rsi_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] IFT_RSI should fail with zero period", test_name);
		Ok(())
	}

	fn check_ift_rsi_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data_small = [10.0, 20.0, 30.0];
		let params = IftRsiParams {
			rsi_period: Some(10),
			wma_period: Some(9),
		};
		let input = IftRsiInput::from_slice(&data_small, params);
		let res = ift_rsi_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] IFT_RSI should fail with period exceeding length",
			test_name
		);
		Ok(())
	}

	fn check_ift_rsi_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let single_point = [42.0];
		let params = IftRsiParams {
			rsi_period: Some(5),
			wma_period: Some(9),
		};
		let input = IftRsiInput::from_slice(&single_point, params);
		let res = ift_rsi_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] IFT_RSI should fail with insufficient data",
			test_name
		);
		Ok(())
	}

	fn check_ift_rsi_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let first_params = IftRsiParams {
			rsi_period: Some(5),
			wma_period: Some(9),
		};
		let first_input = IftRsiInput::from_candles(&candles, "close", first_params);
		let first_result = ift_rsi_with_kernel(&first_input, kernel)?;
		let second_params = IftRsiParams {
			rsi_period: Some(5),
			wma_period: Some(9),
		};
		let second_input = IftRsiInput::from_slice(&first_result.values, second_params);
		let second_result = ift_rsi_with_kernel(&second_input, kernel)?;
		assert_eq!(second_result.values.len(), first_result.values.len());
		Ok(())
	}

	fn check_ift_rsi_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = IftRsiInput::from_candles(
			&candles,
			"close",
			IftRsiParams {
				rsi_period: Some(5),
				wma_period: Some(9),
			},
		);
		let res = ift_rsi_with_kernel(&input, kernel)?;
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
	fn check_ift_rsi_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let test_params = vec![
			IftRsiParams::default(),  // rsi_period: 5, wma_period: 9
			IftRsiParams {
				rsi_period: Some(2),
				wma_period: Some(2),
			},
			IftRsiParams {
				rsi_period: Some(3),
				wma_period: Some(5),
			},
			IftRsiParams {
				rsi_period: Some(7),
				wma_period: Some(14),
			},
			IftRsiParams {
				rsi_period: Some(14),
				wma_period: Some(21),
			},
			IftRsiParams {
				rsi_period: Some(21),
				wma_period: Some(9),
			},
			IftRsiParams {
				rsi_period: Some(50),
				wma_period: Some(50),
			},
			IftRsiParams {
				rsi_period: Some(100),
				wma_period: Some(100),
			},
			IftRsiParams {
				rsi_period: Some(2),
				wma_period: Some(50),
			},
			IftRsiParams {
				rsi_period: Some(50),
				wma_period: Some(2),
			},
			IftRsiParams {
				rsi_period: Some(9),
				wma_period: Some(21),
			},
			IftRsiParams {
				rsi_period: Some(25),
				wma_period: Some(10),
			},
		];

		for (param_idx, params) in test_params.iter().enumerate() {
			let input = IftRsiInput::from_candles(&candles, "close", params.clone());
			let output = ift_rsi_with_kernel(&input, kernel)?;

			for (i, &val) in output.values.iter().enumerate() {
				if val.is_nan() {
					continue; // NaN values are expected during warmup
				}

				let bits = val.to_bits();

				// Check all three poison patterns
				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: rsi_period={}, wma_period={} (param set {})",
						test_name, val, bits, i, 
						params.rsi_period.unwrap_or(5),
						params.wma_period.unwrap_or(9),
						param_idx
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: rsi_period={}, wma_period={} (param set {})",
						test_name, val, bits, i,
						params.rsi_period.unwrap_or(5),
						params.wma_period.unwrap_or(9),
						param_idx
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: rsi_period={}, wma_period={} (param set {})",
						test_name, val, bits, i,
						params.rsi_period.unwrap_or(5),
						params.wma_period.unwrap_or(9),
						param_idx
					);
				}
			}
		}

		Ok(())
	}

	#[cfg(not(debug_assertions))]
	fn check_ift_rsi_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(()) // No-op in release builds
	}

	macro_rules! generate_all_ift_rsi_tests {
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

	generate_all_ift_rsi_tests!(
		check_ift_rsi_partial_params,
		check_ift_rsi_accuracy,
		check_ift_rsi_default_candles,
		check_ift_rsi_zero_period,
		check_ift_rsi_period_exceeds_length,
		check_ift_rsi_very_small_dataset,
		check_ift_rsi_reinput,
		check_ift_rsi_nan_handling,
		check_ift_rsi_no_poison
	);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let output = IftRsiBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;
		let def = IftRsiParams::default();
		let row = output.values_for(&def).expect("default row missing");
		assert_eq!(row.len(), c.close.len());
		Ok(())
	}

	#[cfg(debug_assertions)]
	fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		// Test various parameter sweep configurations
		let test_configs = vec![
			// (rsi_start, rsi_end, rsi_step, wma_start, wma_end, wma_step)
			(2, 10, 2, 2, 10, 2),      // Small periods
			(5, 25, 5, 5, 25, 5),      // Medium periods
			(30, 60, 15, 30, 60, 15),  // Large periods
			(2, 5, 1, 2, 5, 1),        // Dense small range
			(9, 15, 3, 9, 15, 3),      // Mid-range dense
			(2, 2, 0, 2, 20, 2),       // Static RSI, varying WMA
			(2, 20, 2, 9, 9, 0),       // Varying RSI, static WMA
		];

		for (cfg_idx, &(rsi_start, rsi_end, rsi_step, wma_start, wma_end, wma_step)) in test_configs.iter().enumerate() {
			let output = IftRsiBatchBuilder::new()
				.kernel(kernel)
				.rsi_period_range(rsi_start, rsi_end, rsi_step)
				.wma_period_range(wma_start, wma_end, wma_step)
				.apply_candles(&c, "close")?;

			for (idx, &val) in output.values.iter().enumerate() {
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
						 at row {} col {} (flat index {}) with params: rsi_period={}, wma_period={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.rsi_period.unwrap_or(5),
						combo.wma_period.unwrap_or(9)
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: rsi_period={}, wma_period={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.rsi_period.unwrap_or(5),
						combo.wma_period.unwrap_or(9)
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: rsi_period={}, wma_period={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.rsi_period.unwrap_or(5),
						combo.wma_period.unwrap_or(9)
					);
				}
			}
		}

		Ok(())
	}

	#[cfg(not(debug_assertions))]
	fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(()) // No-op in release builds
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
#[pyfunction(name = "ift_rsi")]
#[pyo3(signature = (data, rsi_period, wma_period, kernel=None))]
pub fn ift_rsi_py<'py>(
	py: Python<'py>,
	data: PyReadonlyArray1<'py, f64>,
	rsi_period: usize,
	wma_period: usize,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
	use numpy::{IntoPyArray, PyArrayMethods};

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, false)?;
	
	let params = IftRsiParams {
		rsi_period: Some(rsi_period),
		wma_period: Some(wma_period),
	};
	let input = IftRsiInput::from_slice(slice_in, params);

	let result_vec: Vec<f64> = py
		.allow_threads(|| ift_rsi_with_kernel(&input, kern).map(|o| o.values))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "IftRsiStream")]
pub struct IftRsiStreamPy {
	stream: IftRsiStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl IftRsiStreamPy {
	#[new]
	fn new(rsi_period: usize, wma_period: usize) -> PyResult<Self> {
		let params = IftRsiParams {
			rsi_period: Some(rsi_period),
			wma_period: Some(wma_period),
		};
		let stream = IftRsiStream::try_new(params)
			.map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(IftRsiStreamPy { stream })
	}

	fn update(&mut self, value: f64) -> Option<f64> {
		self.stream.update(value)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "ift_rsi_batch")]
#[pyo3(signature = (data, rsi_period_range, wma_period_range, kernel=None))]
pub fn ift_rsi_batch_py<'py>(
	py: Python<'py>,
	data: PyReadonlyArray1<'py, f64>,
	rsi_period_range: (usize, usize, usize),
	wma_period_range: (usize, usize, usize),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, true)?;

	let sweep = IftRsiBatchRange {
		rsi_period: rsi_period_range,
		wma_period: wma_period_range,
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
			ift_rsi_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
		})
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	let dict = PyDict::new(py);
	dict.set_item("values", out_arr.reshape((rows, cols))?)?;
	dict.set_item(
		"rsi_periods",
		combos
			.iter()
			.map(|p| p.rsi_period.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;
	dict.set_item(
		"wma_periods",
		combos
			.iter()
			.map(|p| p.wma_period.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;

	Ok(dict)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ift_rsi_js(data: &[f64], rsi_period: usize, wma_period: usize) -> Result<Vec<f64>, JsValue> {
	let params = IftRsiParams {
		rsi_period: Some(rsi_period),
		wma_period: Some(wma_period),
	};
	let input = IftRsiInput::from_slice(data, params);
	
	let mut output = vec![0.0; data.len()];  // Single allocation
	
	#[cfg(target_arch = "wasm32")]
	let kernel = detect_best_kernel();
	#[cfg(not(target_arch = "wasm32"))]
	let kernel = Kernel::Scalar;
	
	ift_rsi_into_slice(&mut output, &input, kernel)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ift_rsi_into(
	in_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	rsi_period: usize,
	wma_period: usize,
) -> Result<(), JsValue> {
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}
	
	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);
		let params = IftRsiParams {
			rsi_period: Some(rsi_period),
			wma_period: Some(wma_period),
		};
		let input = IftRsiInput::from_slice(data, params);
		
		#[cfg(target_arch = "wasm32")]
		let kernel = detect_best_kernel();
		#[cfg(not(target_arch = "wasm32"))]
		let kernel = Kernel::Scalar;
		
		if in_ptr == out_ptr as *const f64 {  // CRITICAL: Aliasing check
			let mut temp = vec![0.0; len];
			ift_rsi_into_slice(&mut temp, &input, kernel)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			out.copy_from_slice(&temp);
		} else {
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			ift_rsi_into_slice(out, &input, kernel)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ift_rsi_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ift_rsi_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe { let _ = Vec::from_raw_parts(ptr, len, len); }
	}
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct IftRsiBatchConfig {
	pub rsi_period_range: (usize, usize, usize),
	pub wma_period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct IftRsiBatchJsOutput {
	pub values: Vec<f64>,
	pub combos: Vec<IftRsiParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = ift_rsi_batch)]
pub fn ift_rsi_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	let config: IftRsiBatchConfig =
		serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
	
	let sweep = IftRsiBatchRange {
		rsi_period: config.rsi_period_range,
		wma_period: config.wma_period_range,
	};
	
	#[cfg(target_arch = "wasm32")]
	let kernel = detect_best_kernel();
	#[cfg(not(target_arch = "wasm32"))]
	let kernel = Kernel::Scalar;
	
	let output = ift_rsi_batch_inner(data, &sweep, kernel, false)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	let js_output = IftRsiBatchJsOutput {
		values: output.values,
		combos: output.combos,
		rows: output.rows,
		cols: output.cols,
	};
	
	serde_wasm_bindgen::to_value(&js_output)
		.map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}
