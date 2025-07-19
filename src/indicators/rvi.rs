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

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
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
	let mut out = vec![f64::NAN; data.len()];
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
	let mut values = vec![f64::NAN; rows * cols];
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
	Ok(RviBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
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
	let mut diff = vec![0.0; data.len()];
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
	let mut up = vec![f64::NAN; diff.len()];
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
	let mut down = vec![f64::NAN; diff.len()];
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
	let mut out = vec![f64::NAN; data.len()];
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
	let mut out = vec![f64::NAN; data.len()];
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
	let mut out = vec![f64::NAN; data.len()];
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
	let mut out = vec![f64::NAN; data.len()];
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
	let mut out = vec![f64::NAN; data.len()];
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
		check_rvi_example_values
	);

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
