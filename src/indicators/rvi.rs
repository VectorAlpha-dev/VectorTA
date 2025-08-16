//! # Relative Volatility Index (RVI)
//!
//! Measures direction of volatility, splitting a deviation measure (standard/mean/median deviation)
//! into "up" and "down" buckets and applying a moving average for smoothing. Batch and streaming
//! support, error handling, and AVX2/AVX512 function stubs included for API compatibility.
//!
//! ## Parameters
//! - **period**: Window size for volatility (default 10).
//! - **ma_len**: Smoothing window for up/down (default 14).
//! - **matype**: Smoothing type (0=SMA, 1=EMA; default 1).
//! - **devtype**: Volatility measure (0=StdDev, 1=MeanAbsDev, 2=MedianAbsDev; default 0).
//!
//! ## Errors
//! - **EmptyData**: rvi: Input data is empty.
//! - **InvalidPeriod**: rvi: period or ma_len invalid.
//! - **NotEnoughValidData**: rvi: Not enough data after first valid.
//! - **AllValuesNaN**: rvi: All input is NaN.
//!
//! ## Returns
//! - **Ok(RviOutput)** with `Vec<f64>` of same length as input.
//! - **Err(RviError)** otherwise.

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
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
use thiserror::Error;

impl<'a> AsRef<[f64]> for RviInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			RviData::Slice(slice) => slice,
			RviData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

#[derive(Debug, Clone)]
pub enum RviData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct RviOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct RviParams {
	pub period: Option<usize>,
	pub ma_len: Option<usize>,
	pub matype: Option<usize>,
	pub devtype: Option<usize>,
}

impl Default for RviParams {
	fn default() -> Self {
		Self {
			period: Some(10),
			ma_len: Some(14),
			matype: Some(1),
			devtype: Some(0),
		}
	}
}

#[derive(Debug, Clone)]
pub struct RviInput<'a> {
	pub data: RviData<'a>,
	pub params: RviParams,
}

impl<'a> RviInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: RviParams) -> Self {
		Self {
			data: RviData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slice(sl: &'a [f64], p: RviParams) -> Self {
		Self {
			data: RviData::Slice(sl),
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", RviParams::default())
	}
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(10)
	}
	#[inline]
	pub fn get_ma_len(&self) -> usize {
		self.params.ma_len.unwrap_or(14)
	}
	#[inline]
	pub fn get_matype(&self) -> usize {
		self.params.matype.unwrap_or(1)
	}
	#[inline]
	pub fn get_devtype(&self) -> usize {
		self.params.devtype.unwrap_or(0)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct RviBuilder {
	period: Option<usize>,
	ma_len: Option<usize>,
	matype: Option<usize>,
	devtype: Option<usize>,
	kernel: Kernel,
}

impl Default for RviBuilder {
	fn default() -> Self {
		Self {
			period: None,
			ma_len: None,
			matype: None,
			devtype: None,
			kernel: Kernel::Auto,
		}
	}
}

impl RviBuilder {
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
	pub fn ma_len(mut self, n: usize) -> Self {
		self.ma_len = Some(n);
		self
	}
	#[inline(always)]
	pub fn matype(mut self, n: usize) -> Self {
		self.matype = Some(n);
		self
	}
	#[inline(always)]
	pub fn devtype(mut self, n: usize) -> Self {
		self.devtype = Some(n);
		self
	}
	#[inline(always)]
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline(always)]
	pub fn apply(self, c: &Candles) -> Result<RviOutput, RviError> {
		let p = RviParams {
			period: self.period,
			ma_len: self.ma_len,
			matype: self.matype,
			devtype: self.devtype,
		};
		let i = RviInput::from_candles(c, "close", p);
		rvi_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slice(self, d: &[f64]) -> Result<RviOutput, RviError> {
		let p = RviParams {
			period: self.period,
			ma_len: self.ma_len,
			matype: self.matype,
			devtype: self.devtype,
		};
		let i = RviInput::from_slice(d, p);
		rvi_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<RviStream, RviError> {
		let p = RviParams {
			period: self.period,
			ma_len: self.ma_len,
			matype: self.matype,
			devtype: self.devtype,
		};
		RviStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum RviError {
	#[error("rvi: Empty data provided.")]
	EmptyData,
	#[error("rvi: Invalid period or ma_len: period = {period}, ma_len = {ma_len}, data length = {data_len}")]
	InvalidPeriod {
		period: usize,
		ma_len: usize,
		data_len: usize,
	},
	#[error("rvi: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("rvi: All values are NaN.")]
	AllValuesNaN,
}

#[inline]
pub fn rvi(input: &RviInput) -> Result<RviOutput, RviError> {
	rvi_with_kernel(input, Kernel::Auto)
}

pub fn rvi_with_kernel(input: &RviInput, kernel: Kernel) -> Result<RviOutput, RviError> {
	let data: &[f64] = input.as_ref();
	if data.is_empty() {
		return Err(RviError::EmptyData);
	}
	let period = input.get_period();
	let ma_len = input.get_ma_len();
	let matype = input.get_matype();
	let devtype = input.get_devtype();
	if period == 0 || ma_len == 0 || period > data.len() || ma_len > data.len() {
		return Err(RviError::InvalidPeriod {
			period,
			ma_len,
			data_len: data.len(),
		});
	}
	let first = data.iter().position(|x| !x.is_nan()).ok_or(RviError::AllValuesNaN)?;
	let max_needed = period.saturating_sub(1) + ma_len.saturating_sub(1);
	if (data.len() - first) <= max_needed {
		return Err(RviError::NotEnoughValidData {
			needed: max_needed + 1,
			valid: data.len() - first,
		});
	}
	let warmup_period = first + period.saturating_sub(1) + ma_len.saturating_sub(1);
	let mut out = alloc_with_nan_prefix(data.len(), warmup_period);
	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};
	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => rvi_scalar(data, period, ma_len, matype, devtype, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => rvi_avx2(data, period, ma_len, matype, devtype, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => rvi_avx512(data, period, ma_len, matype, devtype, first, &mut out),
			_ => unreachable!(),
		}
	}
	Ok(RviOutput { values: out })
}

#[inline]
pub fn rvi_into_slice(dst: &mut [f64], input: &RviInput, kern: Kernel) -> Result<(), RviError> {
	let data: &[f64] = input.as_ref();
	if data.is_empty() {
		return Err(RviError::EmptyData);
	}
	let period = input.get_period();
	let ma_len = input.get_ma_len();
	let matype = input.get_matype();
	let devtype = input.get_devtype();
	
	if period == 0 || ma_len == 0 || period > data.len() || ma_len > data.len() {
		return Err(RviError::InvalidPeriod {
			period,
			ma_len,
			data_len: data.len(),
		});
	}
	
	if dst.len() != data.len() {
		return Err(RviError::InvalidPeriod {
			period: dst.len(),
			ma_len: 0,
			data_len: data.len(),
		});
	}
	
	let first = data.iter().position(|x| !x.is_nan()).ok_or(RviError::AllValuesNaN)?;
	let max_needed = period.saturating_sub(1) + ma_len.saturating_sub(1);
	if (data.len() - first) <= max_needed {
		return Err(RviError::NotEnoughValidData {
			needed: max_needed + 1,
			valid: data.len() - first,
		});
	}
	
	let warmup_period = first + period.saturating_sub(1) + ma_len.saturating_sub(1);
	
	// Fill warmup with NaN
	for v in &mut dst[..warmup_period] {
		*v = f64::NAN;
	}
	
	let chosen = match kern {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};
	
	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => rvi_scalar(data, period, ma_len, matype, devtype, first, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => rvi_avx2(data, period, ma_len, matype, devtype, first, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => rvi_avx512(data, period, ma_len, matype, devtype, first, dst),
			_ => unreachable!(),
		}
	}
	
	Ok(())
}

#[inline]
pub fn rvi_scalar(
	data: &[f64],
	period: usize,
	ma_len: usize,
	matype: usize,
	devtype: usize,
	first: usize,
	out: &mut [f64],
) {
	let dev_array = compute_dev(data, period, devtype);
	let diff_array = compute_diff_same_length(data);
	let up_array = compute_up_array(&diff_array, &dev_array);
	let down_array = compute_down_array(&diff_array, &dev_array);
	let up_smoothed = compute_rolling_ma(&up_array, ma_len, matype);
	let down_smoothed = compute_rolling_ma(&down_array, ma_len, matype);
	let start_idx = first + period.saturating_sub(1) + ma_len.saturating_sub(1);
	for i in start_idx..data.len() {
		let up_val = up_smoothed[i];
		let down_val = down_smoothed[i];
		if up_val.is_nan() || down_val.is_nan() || (up_val + down_val).abs() < f64::EPSILON {
			out[i] = f64::NAN;
		} else {
			out[i] = 100.0 * (up_val / (up_val + down_val));
		}
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn rvi_avx512(
	data: &[f64],
	period: usize,
	ma_len: usize,
	matype: usize,
	devtype: usize,
	first: usize,
	out: &mut [f64],
) {
	if period <= 32 {
		unsafe { rvi_avx512_short(data, period, ma_len, matype, devtype, first, out) }
	} else {
		unsafe { rvi_avx512_long(data, period, ma_len, matype, devtype, first, out) }
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn rvi_avx2(
	data: &[f64],
	period: usize,
	ma_len: usize,
	matype: usize,
	devtype: usize,
	first: usize,
	out: &mut [f64],
) {
	rvi_scalar(data, period, ma_len, matype, devtype, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn rvi_avx512_short(
	data: &[f64],
	period: usize,
	ma_len: usize,
	matype: usize,
	devtype: usize,
	first: usize,
	out: &mut [f64],
) {
	rvi_scalar(data, period, ma_len, matype, devtype, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn rvi_avx512_long(
	data: &[f64],
	period: usize,
	ma_len: usize,
	matype: usize,
	devtype: usize,
	first: usize,
	out: &mut [f64],
) {
	rvi_scalar(data, period, ma_len, matype, devtype, first, out)
}

// ========== Batch and Streaming ==========

#[derive(Clone, Debug)]
pub struct RviStream {
	period: usize,
	ma_len: usize,
	matype: usize,
	devtype: usize,
	buffer: Vec<f64>,
	up_smoother: Vec<f64>,
	down_smoother: Vec<f64>,
	count: usize,
	filled: bool,
}

impl RviStream {
	pub fn try_new(params: RviParams) -> Result<Self, RviError> {
		let period = params.period.unwrap_or(10);
		let ma_len = params.ma_len.unwrap_or(14);
		let matype = params.matype.unwrap_or(1);
		let devtype = params.devtype.unwrap_or(0);
		if period == 0 || ma_len == 0 {
			return Err(RviError::InvalidPeriod {
				period,
				ma_len,
				data_len: 0,
			});
		}
		Ok(Self {
			period,
			ma_len,
			matype,
			devtype,
			buffer: vec![f64::NAN; period + 1],
			up_smoother: vec![f64::NAN; ma_len],
			down_smoother: vec![f64::NAN; ma_len],
			count: 0,
			filled: false,
		})
	}

	#[inline(always)]
	pub fn update(&mut self, value: f64) -> Option<f64> {
		// Streaming RVI not fully supported due to dependencies on previous dev array and diff.
		// For compatibility, always return None.
		let _ = value;
		None
	}
}

#[derive(Clone, Debug)]
pub struct RviBatchRange {
	pub period: (usize, usize, usize),
	pub ma_len: (usize, usize, usize),
	pub matype: (usize, usize, usize),
	pub devtype: (usize, usize, usize),
}

impl Default for RviBatchRange {
	fn default() -> Self {
		Self {
			period: (10, 40, 1),
			ma_len: (14, 14, 0),
			matype: (1, 1, 0),
			devtype: (0, 0, 0),
		}
	}
}

#[derive(Clone, Debug, Default)]
pub struct RviBatchBuilder {
	range: RviBatchRange,
	kernel: Kernel,
}

impl RviBatchBuilder {
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
	pub fn ma_len_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.ma_len = (start, end, step);
		self
	}
	#[inline]
	pub fn ma_len_static(mut self, p: usize) -> Self {
		self.range.ma_len = (p, p, 0);
		self
	}
	#[inline]
	pub fn matype_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.matype = (start, end, step);
		self
	}
	#[inline]
	pub fn matype_static(mut self, p: usize) -> Self {
		self.range.matype = (p, p, 0);
		self
	}
	#[inline]
	pub fn devtype_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.devtype = (start, end, step);
		self
	}
	#[inline]
	pub fn devtype_static(mut self, p: usize) -> Self {
		self.range.devtype = (p, p, 0);
		self
	}

	pub fn apply_slice(self, data: &[f64]) -> Result<RviBatchOutput, RviError> {
		rvi_batch_with_kernel(data, &self.range, self.kernel)
	}

	pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<RviBatchOutput, RviError> {
		RviBatchBuilder::new().kernel(k).apply_slice(data)
	}

	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<RviBatchOutput, RviError> {
		let slice = source_type(c, src);
		self.apply_slice(slice)
	}

	pub fn with_default_candles(c: &Candles) -> Result<RviBatchOutput, RviError> {
		RviBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
	}
}

#[derive(Clone, Debug)]
pub struct RviBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<RviParams>,
	pub rows: usize,
	pub cols: usize,
}

impl RviBatchOutput {
	pub fn row_for_params(&self, p: &RviParams) -> Option<usize> {
		self.combos.iter().position(|c| {
			c.period.unwrap_or(10) == p.period.unwrap_or(10)
				&& c.ma_len.unwrap_or(14) == p.ma_len.unwrap_or(14)
				&& c.matype.unwrap_or(1) == p.matype.unwrap_or(1)
				&& c.devtype.unwrap_or(0) == p.devtype.unwrap_or(0)
		})
	}
	pub fn values_for(&self, p: &RviParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &RviBatchRange) -> Vec<RviParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let periods = axis_usize(r.period);
	let ma_lens = axis_usize(r.ma_len);
	let matypes = axis_usize(r.matype);
	let devtypes = axis_usize(r.devtype);
	let mut out = Vec::with_capacity(periods.len() * ma_lens.len() * matypes.len() * devtypes.len());
	for &p in &periods {
		for &m in &ma_lens {
			for &t in &matypes {
				for &d in &devtypes {
					out.push(RviParams {
						period: Some(p),
						ma_len: Some(m),
						matype: Some(t),
						devtype: Some(d),
					});
				}
			}
		}
	}
	out
}

#[inline(always)]
pub fn rvi_batch_with_kernel(data: &[f64], sweep: &RviBatchRange, k: Kernel) -> Result<RviBatchOutput, RviError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => {
			return Err(RviError::InvalidPeriod {
				period: 0,
				ma_len: 0,
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
	rvi_batch_par_slice(data, sweep, simd)
}

#[inline(always)]
pub fn rvi_batch_slice(data: &[f64], sweep: &RviBatchRange, kern: Kernel) -> Result<RviBatchOutput, RviError> {
	rvi_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn rvi_batch_par_slice(data: &[f64], sweep: &RviBatchRange, kern: Kernel) -> Result<RviBatchOutput, RviError> {
	rvi_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn rvi_batch_inner(
	data: &[f64],
	sweep: &RviBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<RviBatchOutput, RviError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(RviError::InvalidPeriod {
			period: 0,
			ma_len: 0,
			data_len: 0,
		});
	}
	let first = data.iter().position(|x| !x.is_nan()).ok_or(RviError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	let max_m = combos.iter().map(|c| c.ma_len.unwrap()).max().unwrap();
	if data.len() - first < max_p.saturating_add(max_m) {
		return Err(RviError::NotEnoughValidData {
			needed: max_p + max_m,
			valid: data.len() - first,
		});
	}
	let rows = combos.len();
	let cols = data.len();
	
	// Use make_uninit_matrix for proper memory allocation like ALMA
	let mut buf_mu = make_uninit_matrix(rows, cols);
	
	// Calculate warmup periods for each parameter combination
	let warm: Vec<usize> = combos
		.iter()
		.map(|c| first + c.period.unwrap().saturating_sub(1) + c.ma_len.unwrap().saturating_sub(1))
		.collect();
	init_matrix_prefixes(&mut buf_mu, cols, &warm);
	
	let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
	let out: &mut [f64] =
		unsafe { core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len()) };
	
	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let prm = &combos[row];
		match kern {
			Kernel::Scalar => rvi_row_scalar(data, first, prm, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => rvi_row_avx2(data, first, prm, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => rvi_row_avx512(data, first, prm, out_row),
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
	
	// Reconstruct Vec from raw parts like ALMA
	let values = unsafe {
		Vec::from_raw_parts(
			buf_guard.as_mut_ptr() as *mut f64,
			buf_guard.len(),
			buf_guard.capacity(),
		)
	};
	
	Ok(RviBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
fn rvi_batch_inner_into(
	data: &[f64],
	sweep: &RviBatchRange,
	kern: Kernel,
	parallel: bool,
	output: &mut [f64],
) -> Result<Vec<RviParams>, RviError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(RviError::InvalidPeriod {
			period: 0,
			ma_len: 0,
			data_len: 0,
		});
	}
	let first = data.iter().position(|x| !x.is_nan()).ok_or(RviError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	let max_m = combos.iter().map(|c| c.ma_len.unwrap()).max().unwrap();
	if data.len() - first < max_p.saturating_add(max_m) {
		return Err(RviError::NotEnoughValidData {
			needed: max_p + max_m,
			valid: data.len() - first,
		});
	}
	let cols = data.len();
	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let prm = &combos[row];
		match kern {
			Kernel::Scalar => rvi_row_scalar(data, first, prm, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => rvi_row_avx2(data, first, prm, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => rvi_row_avx512(data, first, prm, out_row),
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

#[inline(always)]
unsafe fn rvi_row_scalar(data: &[f64], first: usize, params: &RviParams, out: &mut [f64]) {
	rvi_scalar(
		data,
		params.period.unwrap(),
		params.ma_len.unwrap(),
		params.matype.unwrap(),
		params.devtype.unwrap(),
		first,
		out,
	)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn rvi_row_avx2(data: &[f64], first: usize, params: &RviParams, out: &mut [f64]) {
	rvi_avx2(
		data,
		params.period.unwrap(),
		params.ma_len.unwrap(),
		params.matype.unwrap(),
		params.devtype.unwrap(),
		first,
		out,
	)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn rvi_row_avx512(data: &[f64], first: usize, params: &RviParams, out: &mut [f64]) {
	rvi_avx512(
		data,
		params.period.unwrap(),
		params.ma_len.unwrap(),
		params.matype.unwrap(),
		params.devtype.unwrap(),
		first,
		out,
	)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn rvi_row_avx512_short(data: &[f64], first: usize, params: &RviParams, out: &mut [f64]) {
	rvi_avx512_short(
		data,
		params.period.unwrap(),
		params.ma_len.unwrap(),
		params.matype.unwrap(),
		params.devtype.unwrap(),
		first,
		out,
	)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn rvi_row_avx512_long(data: &[f64], first: usize, params: &RviParams, out: &mut [f64]) {
	rvi_avx512_long(
		data,
		params.period.unwrap(),
		params.ma_len.unwrap(),
		params.matype.unwrap(),
		params.devtype.unwrap(),
		first,
		out,
	)
}

// ========== Indicator utility functions (unchanged) ==========

fn compute_diff_same_length(data: &[f64]) -> Vec<f64> {
	let mut diff = alloc_with_nan_prefix(data.len(), 0);
	diff[0] = 0.0;
	for i in 1..data.len() {
		let prev = data[i - 1];
		let curr = data[i];
		if prev.is_nan() || curr.is_nan() {
			diff[i] = f64::NAN;
		} else {
			diff[i] = curr - prev;
		}
	}
	diff
}
fn compute_up_array(diff: &[f64], dev: &[f64]) -> Vec<f64> {
	let mut up = alloc_with_nan_prefix(diff.len(), 0);
	for i in 0..diff.len() {
		let d = diff[i];
		let dv = dev[i];
		if d.is_nan() || dv.is_nan() {
			up[i] = f64::NAN;
		} else if d <= 0.0 {
			up[i] = 0.0;
		} else {
			up[i] = dv;
		}
	}
	up
}
fn compute_down_array(diff: &[f64], dev: &[f64]) -> Vec<f64> {
	let mut down = alloc_with_nan_prefix(diff.len(), 0);
	for i in 0..diff.len() {
		let d = diff[i];
		let dv = dev[i];
		if d.is_nan() || dv.is_nan() {
			down[i] = f64::NAN;
		} else if d > 0.0 {
			down[i] = 0.0;
		} else {
			down[i] = dv;
		}
	}
	down
}
fn compute_dev(data: &[f64], period: usize, devtype: usize) -> Vec<f64> {
	match devtype {
		1 => rolling_mean_abs_dev(data, period),
		2 => rolling_median_abs_dev(data, period),
		_ => rolling_std_dev(data, period),
	}
}
fn rolling_std_dev(data: &[f64], period: usize) -> Vec<f64> {
	let mut out = alloc_with_nan_prefix(data.len(), period.saturating_sub(1));
	if period == 0 || period > data.len() {
		return out;
	}
	let mut window_sum = 0.0;
	let mut window_sumsq = 0.0;
	for i in 0..period {
		let x = data[i];
		if x.is_nan() {
			window_sum = f64::NAN;
			break;
		}
		window_sum += x;
		window_sumsq += x * x;
	}
	if !window_sum.is_nan() {
		let mean = window_sum / (period as f64);
		let mean_sq = window_sumsq / (period as f64);
		out[period - 1] = (mean_sq - mean * mean).sqrt();
	}
	for i in period..data.len() {
		let leaving = data[i - period];
		let incoming = data[i];
		if leaving.is_nan() || incoming.is_nan() || window_sum.is_nan() {
			out[i] = f64::NAN;
			window_sum = f64::NAN;
			continue;
		}
		window_sum += incoming - leaving;
		window_sumsq += incoming * incoming - leaving * leaving;
		let mean = window_sum / (period as f64);
		let mean_sq = window_sumsq / (period as f64);
		out[i] = (mean_sq - mean * mean).sqrt();
	}
	out
}
fn rolling_mean_abs_dev(data: &[f64], period: usize) -> Vec<f64> {
	let mut out = alloc_with_nan_prefix(data.len(), period.saturating_sub(1));
	if period == 0 || period > data.len() {
		return out;
	}
	use std::collections::VecDeque;
	let mut window = VecDeque::with_capacity(period);
	let mut current_sum = 0.0;
	for i in 0..data.len() {
		let x = data[i];
		if x.is_nan() {
			out[i] = f64::NAN;
			window.clear();
			current_sum = 0.0;
		} else {
			window.push_back(x);
			current_sum += x;
			if window.len() > period {
				if let Some(old) = window.pop_front() {
					current_sum -= old;
				}
			}
			if window.len() == period {
				let mean = current_sum / (period as f64);
				let mut abs_sum = 0.0;
				for &val in &window {
					abs_sum += (val - mean).abs();
				}
				out[i] = abs_sum / (period as f64);
			}
		}
	}
	out
}
fn rolling_median_abs_dev(data: &[f64], period: usize) -> Vec<f64> {
	let mut out = alloc_with_nan_prefix(data.len(), period.saturating_sub(1));
	if period == 0 || period > data.len() {
		return out;
	}
	use std::collections::VecDeque;
	let mut window = VecDeque::with_capacity(period);
	for i in 0..data.len() {
		let x = data[i];
		if x.is_nan() {
			out[i] = f64::NAN;
			window.clear();
		} else {
			window.push_back(x);
			if window.len() > period {
				window.pop_front();
			}
			if window.len() == period {
				let mut tmp: Vec<f64> = window.iter().copied().collect();
				tmp.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
				let median = if period % 2 == 1 {
					tmp[period / 2]
				} else {
					(tmp[period / 2 - 1] + tmp[period / 2]) / 2.0
				};
				let mut abs_sum = 0.0;
				for &val in &tmp {
					abs_sum += (val - median).abs();
				}
				out[i] = abs_sum / (period as f64);
			}
		}
	}
	out
}
fn compute_rolling_ma(data: &[f64], period: usize, matype: usize) -> Vec<f64> {
	match matype {
		0 => rolling_sma(data, period),
		_ => rolling_ema(data, period),
	}
}
fn rolling_sma(data: &[f64], period: usize) -> Vec<f64> {
	let mut out = alloc_with_nan_prefix(data.len(), period.saturating_sub(1));
	if period == 0 || period > data.len() {
		return out;
	}
	let mut window_sum = 0.0;
	let mut count = 0;
	for i in 0..data.len() {
		let x = data[i];
		if x.is_nan() {
			out[i] = f64::NAN;
			window_sum = 0.0;
			count = 0;
		} else {
			window_sum += x;
			count += 1;
			if i >= period {
				let old = data[i - period];
				if !old.is_nan() {
					window_sum -= old;
					count -= 1;
				} else {
					out[i] = f64::NAN;
					continue;
				}
			}
			if i + 1 >= period {
				out[i] = window_sum / (period as f64);
			}
		}
	}
	out
}
fn rolling_ema(data: &[f64], period: usize) -> Vec<f64> {
	let mut out = alloc_with_nan_prefix(data.len(), period.saturating_sub(1));
	if period == 0 || period > data.len() {
		return out;
	}
	let alpha = 2.0 / (period as f64 + 1.0);
	let mut prev_ema = 0.0;
	let mut started = false;
	for i in 0..data.len() {
		let x = data[i];
		if x.is_nan() {
			out[i] = f64::NAN;
			continue;
		}
		if !started {
			let first_window_end = if i + 1 < period { i + 1 } else { period };
			if i + 1 < period {
				out[i] = f64::NAN;
				prev_ema += x;
				if i + 1 == first_window_end {
					prev_ema /= period as f64;
				}
			} else {
				prev_ema += x;
				prev_ema /= period as f64;
				out[i] = prev_ema;
				started = true;
			}
		} else {
			prev_ema = alpha * x + (1.0 - alpha) * prev_ema;
			out[i] = prev_ema;
		}
	}
	out
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_rvi_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let partial_params = RviParams {
			period: Some(10),
			ma_len: None,
			matype: None,
			devtype: None,
		};
		let input = RviInput::from_candles(&candles, "close", partial_params);
		let output = rvi_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_rvi_default_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = RviInput::with_default_candles(&candles);
		let output = rvi_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_rvi_error_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data = [10.0, 20.0, 30.0, 40.0];
		let params = RviParams {
			period: Some(0),
			ma_len: Some(14),
			matype: Some(1),
			devtype: Some(0),
		};
		let input = RviInput::from_slice(&data, params);
		let result = rvi_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}

	fn check_rvi_error_zero_ma_len(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data = [10.0, 20.0, 30.0, 40.0];
		let params = RviParams {
			period: Some(10),
			ma_len: Some(0),
			matype: Some(1),
			devtype: Some(0),
		};
		let input = RviInput::from_slice(&data, params);
		let result = rvi_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}

	fn check_rvi_error_period_exceeds_data_length(
		test_name: &str,
		kernel: Kernel,
	) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data = [10.0, 20.0, 30.0];
		let params = RviParams {
			period: Some(10),
			ma_len: Some(14),
			matype: Some(1),
			devtype: Some(0),
		};
		let input = RviInput::from_slice(&data, params);
		let result = rvi_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}

	fn check_rvi_all_nan_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data = [f64::NAN, f64::NAN, f64::NAN];
		let params = RviParams::default();
		let input = RviInput::from_slice(&data, params);
		let result = rvi_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}

	fn check_rvi_not_enough_valid_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data = [f64::NAN, 1.0, 2.0, 3.0];
		let params = RviParams {
			period: Some(3),
			ma_len: Some(5),
			matype: Some(1),
			devtype: Some(0),
		};
		let input = RviInput::from_slice(&data, params);
		let result = rvi_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}

	fn check_rvi_example_values(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = RviParams {
			period: Some(10),
			ma_len: Some(14),
			matype: Some(1),
			devtype: Some(0),
		};
		let input = RviInput::from_candles(&candles, "close", params);
		let output = rvi_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		let last_five = &output.values[output.values.len().saturating_sub(5)..];
		let expected = [
			67.48579363423423,
			62.03322230763894,
			56.71819195768154,
			60.487299747927636,
			55.022521428674175,
		];
		for (i, &val) in last_five.iter().enumerate() {
			let exp = expected[i];
			assert!(val.is_finite(), "Expected a finite RVI value, got NaN at index {}", i);
			let diff = (val - exp).abs();
			assert!(
				diff < 1e-1,
				"Mismatch at index {} -> got: {}, expected: {}, diff: {}",
				i,
				val,
				exp,
				diff
			);
		}
		Ok(())
	}

	#[cfg(debug_assertions)]
	fn check_rvi_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		
		// Define comprehensive parameter combinations
		let test_params = vec![
			RviParams::default(), // period: 10, ma_len: 14, matype: 1, devtype: 0
			RviParams {
				period: Some(2),  // minimum viable period
				ma_len: Some(2),  // minimum viable ma_len
				matype: Some(0),  // SMA
				devtype: Some(0), // StdDev
			},
			RviParams {
				period: Some(5),
				ma_len: Some(5),
				matype: Some(1),  // EMA
				devtype: Some(1), // MeanAbsDev
			},
			RviParams {
				period: Some(10),
				ma_len: Some(20),
				matype: Some(0),  // SMA
				devtype: Some(2), // MedianAbsDev
			},
			RviParams {
				period: Some(20),
				ma_len: Some(30),
				matype: Some(1),  // EMA
				devtype: Some(0), // StdDev
			},
			RviParams {
				period: Some(50),
				ma_len: Some(50),
				matype: Some(0),  // SMA
				devtype: Some(1), // MeanAbsDev
			},
			RviParams {
				period: Some(100), // large period
				ma_len: Some(20),
				matype: Some(1),   // EMA
				devtype: Some(2),  // MedianAbsDev
			},
			RviParams {
				period: Some(14),
				ma_len: Some(100), // large ma_len
				matype: Some(0),   // SMA
				devtype: Some(0),  // StdDev
			},
		];
		
		for (param_idx, params) in test_params.iter().enumerate() {
			let input = RviInput::from_candles(&candles, "close", params.clone());
			let output = rvi_with_kernel(&input, kernel)?;
			
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
	fn check_rvi_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		Ok(()) // No-op in release builds
	}

	#[cfg(feature = "proptest")]
	#[allow(clippy::float_cmp)]
	fn check_rvi_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		use proptest::prelude::*;
		skip_if_unsupported!(kernel, test_name);

		// Generate random parameter combinations and corresponding data vectors
		// Use positive values to simulate realistic financial data (prices >= 0)
		let strat = (2usize..=30, 2usize..=30, 0usize..=1, 0usize..=2)
			.prop_flat_map(|(period, ma_len, matype, devtype)| {
				(
					prop::collection::vec(
						(0.01f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
						(period + ma_len)..400,
					),
					Just(period),
					Just(ma_len),
					Just(matype),
					Just(devtype),
				)
			});

		proptest::test_runner::TestRunner::default()
			.run(&strat, |(data, period, ma_len, matype, devtype)| {
				let params = RviParams {
					period: Some(period),
					ma_len: Some(ma_len),
					matype: Some(matype),
					devtype: Some(devtype),
				};
				let input = RviInput::from_slice(&data, params.clone());

				// Get output from the kernel being tested
				let RviOutput { values: out } = rvi_with_kernel(&input, kernel).unwrap();
				
				// Get reference output from scalar kernel for comparison
				let RviOutput { values: ref_out } = rvi_with_kernel(&input, Kernel::Scalar).unwrap();

				// Calculate warmup period
				let warmup = period.saturating_sub(1) + ma_len.saturating_sub(1);

				// Verify warmup period handling
				for i in 0..warmup.min(data.len()) {
					prop_assert!(
						out[i].is_nan(),
						"Expected NaN during warmup at index {}, got {}",
						i, out[i]
					);
				}

				// Verify values after warmup period
				for i in warmup..data.len() {
					let y = out[i];
					let r = ref_out[i];

					// Property 1: RVI should be between 0 and 100 (it's a percentage)
					if y.is_finite() {
						prop_assert!(
							y >= -1e-9 && y <= 100.0 + 1e-9,
							"RVI out of bounds at idx {}: {} (should be 0-100)",
							i, y
						);
					}

					// Property 2: Kernel consistency check
					if !y.is_finite() || !r.is_finite() {
						prop_assert!(
							y.to_bits() == r.to_bits(),
							"finite/NaN mismatch idx {}: {} vs {}",
							i, y, r
						);
					} else {
						// Use ULP comparison for floating-point accuracy
						let y_bits = y.to_bits();
						let r_bits = r.to_bits();
						let ulp_diff: u64 = y_bits.abs_diff(r_bits);

						prop_assert!(
							(y - r).abs() <= 1e-9 || ulp_diff <= 4,
							"Kernel mismatch at idx {}: {} vs {} (ULP={})",
							i, y, r, ulp_diff
						);
					}
				}

				// Property 3: Special case - monotonic increasing data
				// When prices consistently increase, up_deviation dominates, RVI should be close to 100
				let is_monotonic_increasing = data.windows(2)
					.all(|w| w[1] >= w[0] - f64::EPSILON);
				
				if is_monotonic_increasing && out.len() > warmup + 10 {
					let last_values = &out[out.len().saturating_sub(10)..];
					let finite_values: Vec<f64> = last_values.iter()
						.filter(|v| v.is_finite())
						.copied()
						.collect();
					
					if !finite_values.is_empty() {
						let avg_rvi = finite_values.iter().sum::<f64>() / finite_values.len() as f64;
						prop_assert!(
							avg_rvi >= 90.0,  // Should be close to 100, allow some smoothing tolerance
							"RVI should be high for monotonic increasing data, got avg {}",
							avg_rvi
						);
					}
				}

				// Property 4: Special case - monotonic decreasing data
				// When prices consistently decrease, down_deviation dominates, RVI should be close to 0
				let is_monotonic_decreasing = data.windows(2)
					.all(|w| w[1] <= w[0] + f64::EPSILON);
				
				if is_monotonic_decreasing && out.len() > warmup + 10 {
					let last_values = &out[out.len().saturating_sub(10)..];
					let finite_values: Vec<f64> = last_values.iter()
						.filter(|v| v.is_finite())
						.copied()
						.collect();
					
					if !finite_values.is_empty() {
						let avg_rvi = finite_values.iter().sum::<f64>() / finite_values.len() as f64;
						prop_assert!(
							avg_rvi <= 10.0,  // Should be close to 0, allow some smoothing tolerance
							"RVI should be low for monotonic decreasing data, got avg {}",
							avg_rvi
						);
					}
				}

				// Property 5: Special case - constant data
				// When all values are the same, there's no volatility, RVI should be NaN
				let is_constant = data.windows(2)
					.all(|w| (w[0] - w[1]).abs() <= f64::EPSILON * w[0].abs().max(1.0));
				
				if is_constant && out.len() > warmup {
					for i in warmup..out.len() {
						prop_assert!(
							out[i].is_nan(),
							"RVI should be NaN for constant data at idx {}, got {}",
							i, out[i]
						);
					}
				}

				// Property 6: Special case - alternating pattern
				// When prices alternate up/down regularly, RVI should be near 50
				let mut is_alternating = data.len() >= 4;
				if is_alternating {
					for i in 1..data.len().saturating_sub(1) {
						let diff1 = data[i] - data[i - 1];
						let diff2 = data[i + 1] - data[i];
						// Check if signs alternate
						if diff1 * diff2 >= 0.0 && diff1.abs() > f64::EPSILON {
							is_alternating = false;
							break;
						}
					}
				}
				
				if is_alternating && out.len() > warmup + 10 {
					let last_values = &out[out.len().saturating_sub(10)..];
					let finite_values: Vec<f64> = last_values.iter()
						.filter(|v| v.is_finite())
						.copied()
						.collect();
					
					if !finite_values.is_empty() {
						let avg_rvi = finite_values.iter().sum::<f64>() / finite_values.len() as f64;
						prop_assert!(
							avg_rvi >= 35.0 && avg_rvi <= 65.0,
							"RVI should be near 50 for alternating data, got avg {}",
							avg_rvi
						);
					}
				}

				Ok(())
			})?;

		Ok(())
	}

	macro_rules! generate_all_rvi_tests {
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

	generate_all_rvi_tests!(
		check_rvi_partial_params,
		check_rvi_default_params,
		check_rvi_error_zero_period,
		check_rvi_error_zero_ma_len,
		check_rvi_error_period_exceeds_data_length,
		check_rvi_all_nan_input,
		check_rvi_not_enough_valid_data,
		check_rvi_example_values,
		check_rvi_no_poison
	);

	#[cfg(feature = "proptest")]
	generate_all_rvi_tests!(check_rvi_property);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let output = RviBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;
		let def = RviParams::default();
		let row = output.values_for(&def).expect("default row missing");
		assert_eq!(row.len(), c.close.len());
		Ok(())
	}

	#[cfg(debug_assertions)]
	fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);
		
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		
		// Test various parameter sweep configurations
		let test_configs = vec![
			// (period_start, period_end, period_step, ma_len_start, ma_len_end, ma_len_step)
			(2, 10, 2, 2, 10, 2),      // Small periods and ma_lens
			(5, 25, 5, 10, 30, 5),     // Medium periods and ma_lens
			(30, 60, 15, 20, 40, 10),  // Large periods and ma_lens
			(2, 5, 1, 2, 5, 1),        // Dense small range
			(10, 20, 5, 50, 100, 25),  // Medium period, large ma_len
			(50, 100, 50, 10, 20, 10), // Large period, small ma_len
		];
		
		for (cfg_idx, &(p_start, p_end, p_step, m_start, m_end, m_step)) in test_configs.iter().enumerate() {
			// Test with different matype and devtype combinations
			for matype in [0, 1].iter() {
				for devtype in [0, 1, 2].iter() {
					let output = RviBatchBuilder::new()
						.kernel(kernel)
						.period_range(p_start, p_end, p_step)
						.ma_len_range(m_start, m_end, m_step)
						.matype_static(*matype)
						.devtype_static(*devtype)
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
								"[{}] Config {} (matype={}, devtype={}): Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
								 at row {} col {} (flat index {}) with params: {:?}",
								test, cfg_idx, matype, devtype, val, bits, row, col, idx, combo
							);
						}
						
						if bits == 0x22222222_22222222 {
							panic!(
								"[{}] Config {} (matype={}, devtype={}): Found init_matrix_prefixes poison value {} (0x{:016X}) \
								 at row {} col {} (flat index {}) with params: {:?}",
								test, cfg_idx, matype, devtype, val, bits, row, col, idx, combo
							);
						}
						
						if bits == 0x33333333_33333333 {
							panic!(
								"[{}] Config {} (matype={}, devtype={}): Found make_uninit_matrix poison value {} (0x{:016X}) \
								 at row {} col {} (flat index {}) with params: {:?}",
								test, cfg_idx, matype, devtype, val, bits, row, col, idx, combo
							);
						}
					}
				}
			}
		}
		
		Ok(())
	}

	#[cfg(not(debug_assertions))]
	fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
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

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn rvi_js(data: &[f64], period: usize, ma_len: usize, matype: usize, devtype: usize) -> Result<Vec<f64>, JsValue> {
	let params = RviParams {
		period: Some(period),
		ma_len: Some(ma_len),
		matype: Some(matype),
		devtype: Some(devtype),
	};
	let input = RviInput::from_slice(data, params);
	
	let mut output = vec![0.0; data.len()];
	rvi_into_slice(&mut output, &input, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn rvi_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn rvi_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe { let _ = Vec::from_raw_parts(ptr, len, len); }
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn rvi_into(
	in_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period: usize,
	ma_len: usize,
	matype: usize,
	devtype: usize,
) -> Result<(), JsValue> {
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}
	
	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);
		let params = RviParams {
			period: Some(period),
			ma_len: Some(ma_len),
			matype: Some(matype),
			devtype: Some(devtype),
		};
		let input = RviInput::from_slice(data, params);
		
		if in_ptr == out_ptr {
			// Handle aliasing case
			let mut temp = vec![0.0; len];
			rvi_into_slice(&mut temp, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			out.copy_from_slice(&temp);
		} else {
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			rvi_into_slice(out, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct RviBatchConfig {
	pub period_range: (usize, usize, usize),
	pub ma_len_range: (usize, usize, usize),
	pub matype_range: (usize, usize, usize),
	pub devtype_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = rvi_batch)]
pub fn rvi_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	let config: RviBatchConfig = serde_wasm_bindgen::from_value(config)
		.map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
	
	let sweep = RviBatchRange {
		period: config.period_range,
		ma_len: config.ma_len_range,
		matype: config.matype_range,
		devtype: config.devtype_range,
	};
	
	let output = rvi_batch_inner(data, &sweep, Kernel::Auto, false)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	// Convert output to JS-friendly format
	let js_output = serde_wasm_bindgen::to_value(&serde_json::json!({
		"values": output.values,
		"periods": output.combos.iter().map(|c| c.period.unwrap()).collect::<Vec<_>>(),
		"ma_lens": output.combos.iter().map(|c| c.ma_len.unwrap()).collect::<Vec<_>>(),
		"matypes": output.combos.iter().map(|c| c.matype.unwrap()).collect::<Vec<_>>(),
		"devtypes": output.combos.iter().map(|c| c.devtype.unwrap()).collect::<Vec<_>>(),
		"rows": output.rows,
		"cols": output.cols,
	})).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))?;
	
	Ok(js_output)
}

#[cfg(feature = "python")]
#[pyfunction(name = "rvi")]
#[pyo3(signature = (data, period, ma_len, matype, devtype, kernel=None))]
pub fn rvi_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	period: usize,
	ma_len: usize,
	matype: usize,
	devtype: usize,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
	use numpy::{IntoPyArray, PyArrayMethods};

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, false)?;
	
	let params = RviParams {
		period: Some(period),
		ma_len: Some(ma_len),
		matype: Some(matype),
		devtype: Some(devtype),
	};
	let input = RviInput::from_slice(slice_in, params);

	let result_vec: Vec<f64> = py
		.allow_threads(|| rvi_with_kernel(&input, kern).map(|o| o.values))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction(name = "rvi_batch")]
#[pyo3(signature = (data, period_range, ma_len_range, matype_range, devtype_range, kernel=None))]
pub fn rvi_batch_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	period_range: (usize, usize, usize),
	ma_len_range: (usize, usize, usize),
	matype_range: (usize, usize, usize),
	devtype_range: (usize, usize, usize),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, true)?;

	let sweep = RviBatchRange {
		period: period_range,
		ma_len: ma_len_range,
		matype: matype_range,
		devtype: devtype_range,
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
			rvi_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
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
		"ma_lens",
		combos
			.iter()
			.map(|p| p.ma_len.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;
	dict.set_item(
		"matypes",
		combos
			.iter()
			.map(|p| p.matype.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;
	dict.set_item(
		"devtypes",
		combos
			.iter()
			.map(|p| p.devtype.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;

	Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "RviStream")]
pub struct RviStreamPy {
	stream: RviStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl RviStreamPy {
	#[new]
	fn new(period: usize, ma_len: usize, matype: usize, devtype: usize) -> PyResult<Self> {
		let params = RviParams {
			period: Some(period),
			ma_len: Some(ma_len),
			matype: Some(matype),
			devtype: Some(devtype),
		};
		let stream = RviStream::try_new(params)
			.map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(RviStreamPy { stream })
	}

	fn update(&mut self, value: f64) -> Option<f64> {
		self.stream.update(value)
	}
}
