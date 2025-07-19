//! # Stochastic Fast (StochF)
//!
//! A momentum indicator comparing a security’s closing price to its price range (high-low) over
//! a specified lookback (`fastk_period`). Then applies a moving average (`fastd_period`) on
//! the %K values to obtain %D. "Fast" variant uses shorter averaging and is more sensitive.
//!
//! ## Parameters
//! - **fastk_period**: Lookback period for highest high/lowest low. Defaults to 5.
//! - **fastd_period**: Period for moving average of %K. Defaults to 3.
//! - **fastd_matype**: MA type (only SMA=0 supported). Defaults to 0.
//!
//! ## Errors
//! - **EmptyData**: stochf: Input slice(s) are empty.
//! - **InvalidPeriod**: stochf: Zero or out-of-bounds period.
//! - **AllValuesNaN**: stochf: All values are NaN.
//! - **NotEnoughValidData**: stochf: Not enough valid data after first valid index.
//!
//! ## Returns
//! - **`Ok(StochfOutput)`** with `.k` and `.d` series matching input length.
//! - **`Err(StochfError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum StochfData<'a> {
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
pub struct StochfOutput {
	pub k: Vec<f64>,
	pub d: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct StochfParams {
	pub fastk_period: Option<usize>,
	pub fastd_period: Option<usize>,
	pub fastd_matype: Option<usize>,
}

impl Default for StochfParams {
	fn default() -> Self {
		Self {
			fastk_period: Some(5),
			fastd_period: Some(3),
			fastd_matype: Some(0),
		}
	}
}

#[derive(Debug, Clone)]
pub struct StochfInput<'a> {
	pub data: StochfData<'a>,
	pub params: StochfParams,
}

impl<'a> StochfInput<'a> {
	#[inline]
	pub fn from_candles(candles: &'a Candles, params: StochfParams) -> Self {
		Self {
			data: StochfData::Candles { candles },
			params,
		}
	}
	#[inline]
	pub fn from_slices(high: &'a [f64], low: &'a [f64], close: &'a [f64], params: StochfParams) -> Self {
		Self {
			data: StochfData::Slices { high, low, close },
			params,
		}
	}
	#[inline]
	pub fn with_default_candles(candles: &'a Candles) -> Self {
		Self::from_candles(candles, StochfParams::default())
	}
	#[inline]
	pub fn get_fastk_period(&self) -> usize {
		self.params.fastk_period.unwrap_or(5)
	}
	#[inline]
	pub fn get_fastd_period(&self) -> usize {
		self.params.fastd_period.unwrap_or(3)
	}
	#[inline]
	pub fn get_fastd_matype(&self) -> usize {
		self.params.fastd_matype.unwrap_or(0)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct StochfBuilder {
	fastk_period: Option<usize>,
	fastd_period: Option<usize>,
	fastd_matype: Option<usize>,
	kernel: Kernel,
}

impl Default for StochfBuilder {
	fn default() -> Self {
		Self {
			fastk_period: None,
			fastd_period: None,
			fastd_matype: None,
			kernel: Kernel::Auto,
		}
	}
}

impl StochfBuilder {
	#[inline]
	pub fn new() -> Self {
		Self::default()
	}
	#[inline]
	pub fn fastk_period(mut self, n: usize) -> Self {
		self.fastk_period = Some(n);
		self
	}
	#[inline]
	pub fn fastd_period(mut self, n: usize) -> Self {
		self.fastd_period = Some(n);
		self
	}
	#[inline]
	pub fn fastd_matype(mut self, t: usize) -> Self {
		self.fastd_matype = Some(t);
		self
	}
	#[inline]
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline]
	pub fn apply(self, candles: &Candles) -> Result<StochfOutput, StochfError> {
		let p = StochfParams {
			fastk_period: self.fastk_period,
			fastd_period: self.fastd_period,
			fastd_matype: self.fastd_matype,
		};
		let i = StochfInput::from_candles(candles, p);
		stochf_with_kernel(&i, self.kernel)
	}
	#[inline]
	pub fn apply_slices(self, high: &[f64], low: &[f64], close: &[f64]) -> Result<StochfOutput, StochfError> {
		let p = StochfParams {
			fastk_period: self.fastk_period,
			fastd_period: self.fastd_period,
			fastd_matype: self.fastd_matype,
		};
		let i = StochfInput::from_slices(high, low, close, p);
		stochf_with_kernel(&i, self.kernel)
	}
	#[inline]
	pub fn into_stream(self) -> Result<StochfStream, StochfError> {
		let p = StochfParams {
			fastk_period: self.fastk_period,
			fastd_period: self.fastd_period,
			fastd_matype: self.fastd_matype,
		};
		StochfStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum StochfError {
	#[error("stochf: Empty data provided.")]
	EmptyData,
	#[error("stochf: Invalid period (fastk={fastk}, fastd={fastd}), data length={data_len}.")]
	InvalidPeriod {
		fastk: usize,
		fastd: usize,
		data_len: usize,
	},
	#[error("stochf: All values are NaN.")]
	AllValuesNaN,
	#[error("stochf: Not enough valid data after first valid index (needed={needed}, valid={valid}).")]
	NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn stochf(input: &StochfInput) -> Result<StochfOutput, StochfError> {
	stochf_with_kernel(input, Kernel::Auto)
}

pub fn stochf_with_kernel(input: &StochfInput, kernel: Kernel) -> Result<StochfOutput, StochfError> {
	let (high, low, close) = match &input.data {
		StochfData::Candles { candles } => {
			let high = candles
				.select_candle_field("high")
				.map_err(|_| StochfError::EmptyData)?;
			let low = candles.select_candle_field("low").map_err(|_| StochfError::EmptyData)?;
			let close = candles
				.select_candle_field("close")
				.map_err(|_| StochfError::EmptyData)?;
			(high, low, close)
		}
		StochfData::Slices { high, low, close } => (*high, *low, *close),
	};

	if high.is_empty() || low.is_empty() || close.is_empty() {
		return Err(StochfError::EmptyData);
	}
	let len = high.len();
	if low.len() != len || close.len() != len {
		return Err(StochfError::EmptyData);
	}

	let fastk_period = input.get_fastk_period();
	let fastd_period = input.get_fastd_period();
	let matype = input.get_fastd_matype();

	if fastk_period == 0 || fastd_period == 0 || fastk_period > len || fastd_period > len {
		return Err(StochfError::InvalidPeriod {
			fastk: fastk_period,
			fastd: fastd_period,
			data_len: len,
		});
	}
	let first_valid_idx = (0..len)
		.find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
		.ok_or(StochfError::AllValuesNaN)?;
	if (len - first_valid_idx) < fastk_period {
		return Err(StochfError::NotEnoughValidData {
			needed: fastk_period,
			valid: len - first_valid_idx,
		});
	}

	let mut k_vals = vec![f64::NAN; len];
	let mut d_vals = vec![f64::NAN; len];

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => stochf_scalar(
				high,
				low,
				close,
				fastk_period,
				fastd_period,
				matype,
				first_valid_idx,
				&mut k_vals,
				&mut d_vals,
			),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => stochf_avx2(
				high,
				low,
				close,
				fastk_period,
				fastd_period,
				matype,
				first_valid_idx,
				&mut k_vals,
				&mut d_vals,
			),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => stochf_avx512(
				high,
				low,
				close,
				fastk_period,
				fastd_period,
				matype,
				first_valid_idx,
				&mut k_vals,
				&mut d_vals,
			),
			_ => unreachable!(),
		}
	}

	Ok(StochfOutput { k: k_vals, d: d_vals })
}

#[inline]
pub unsafe fn stochf_scalar(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	fastk_period: usize,
	fastd_period: usize,
	matype: usize,
	first_valid_idx: usize,
	k_vals: &mut [f64],
	d_vals: &mut [f64],
) {
	let len = high.len();
	for i in first_valid_idx..len {
		if i < first_valid_idx + fastk_period - 1 {
			continue;
		}
		let start = i + 1 - fastk_period;
		let (mut hh, mut ll) = (f64::NEG_INFINITY, f64::INFINITY);
		for j in start..=i {
			let h_j = high[j];
			let l_j = low[j];
			if h_j > hh {
				hh = h_j;
			}
			if l_j < ll {
				ll = l_j;
			}
		}
		if hh == ll {
			k_vals[i] = if close[i] == hh { 100.0 } else { 0.0 };
		} else {
			k_vals[i] = 100.0 * (close[i] - ll) / (hh - ll);
		}
	}
	if matype != 0 {
		d_vals.fill(f64::NAN);
	} else {
		let mut sma_sum = 0.0;
		let mut count = 0;
		for i in 0..len {
			let v = k_vals[i];
			if v.is_nan() {
				d_vals[i] = f64::NAN;
				continue;
			}
			if count < fastd_period {
				sma_sum += v;
				count += 1;
				if count == fastd_period {
					d_vals[i] = sma_sum / (fastd_period as f64);
				} else {
					d_vals[i] = f64::NAN;
				}
			} else {
				sma_sum += v - k_vals[i - fastd_period];
				d_vals[i] = sma_sum / (fastd_period as f64);
			}
		}
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn stochf_avx2(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	fastk_period: usize,
	fastd_period: usize,
	matype: usize,
	first_valid_idx: usize,
	k_vals: &mut [f64],
	d_vals: &mut [f64],
) {
	stochf_scalar(
		high,
		low,
		close,
		fastk_period,
		fastd_period,
		matype,
		first_valid_idx,
		k_vals,
		d_vals,
	);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn stochf_avx512(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	fastk_period: usize,
	fastd_period: usize,
	matype: usize,
	first_valid_idx: usize,
	k_vals: &mut [f64],
	d_vals: &mut [f64],
) {
	if fastk_period <= 32 {
		stochf_avx512_short(
			high,
			low,
			close,
			fastk_period,
			fastd_period,
			matype,
			first_valid_idx,
			k_vals,
			d_vals,
		);
	} else {
		stochf_avx512_long(
			high,
			low,
			close,
			fastk_period,
			fastd_period,
			matype,
			first_valid_idx,
			k_vals,
			d_vals,
		);
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn stochf_avx512_short(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	fastk_period: usize,
	fastd_period: usize,
	matype: usize,
	first_valid_idx: usize,
	k_vals: &mut [f64],
	d_vals: &mut [f64],
) {
	stochf_scalar(
		high,
		low,
		close,
		fastk_period,
		fastd_period,
		matype,
		first_valid_idx,
		k_vals,
		d_vals,
	);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn stochf_avx512_long(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	fastk_period: usize,
	fastd_period: usize,
	matype: usize,
	first_valid_idx: usize,
	k_vals: &mut [f64],
	d_vals: &mut [f64],
) {
	stochf_scalar(
		high,
		low,
		close,
		fastk_period,
		fastd_period,
		matype,
		first_valid_idx,
		k_vals,
		d_vals,
	);
}

#[derive(Debug, Clone)]
pub struct StochfStream {
	fastk_period: usize,
	fastd_period: usize,
	fastd_matype: usize,
	high_buffer: Vec<f64>,
	low_buffer: Vec<f64>,
	close_buffer: Vec<f64>,
	k_buffer: Vec<f64>,
	d_sma_sum: f64,
	head: usize,
	count: usize,
	k_head: usize,
	k_count: usize,
	filled: bool,
}

impl StochfStream {
	pub fn try_new(params: StochfParams) -> Result<Self, StochfError> {
		let fastk_period = params.fastk_period.unwrap_or(5);
		let fastd_period = params.fastd_period.unwrap_or(3);
		let fastd_matype = params.fastd_matype.unwrap_or(0);

		if fastk_period == 0 || fastd_period == 0 {
			return Err(StochfError::InvalidPeriod {
				fastk: fastk_period,
				fastd: fastd_period,
				data_len: 0,
			});
		}

		Ok(Self {
			fastk_period,
			fastd_period,
			fastd_matype,
			high_buffer: vec![f64::NAN; fastk_period],
			low_buffer: vec![f64::NAN; fastk_period],
			close_buffer: vec![f64::NAN; fastk_period],
			k_buffer: vec![f64::NAN; fastd_period],
			d_sma_sum: 0.0,
			head: 0,
			count: 0,
			k_head: 0,
			k_count: 0,
			filled: false,
		})
	}

	pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64)> {
		self.high_buffer[self.head] = high;
		self.low_buffer[self.head] = low;
		self.close_buffer[self.head] = close;
		self.head = (self.head + 1) % self.fastk_period;
		if self.count < self.fastk_period {
			self.count += 1;
		}
		if self.count < self.fastk_period {
			return None;
		}

		let mut hh = f64::NEG_INFINITY;
		let mut ll = f64::INFINITY;
		for i in 0..self.fastk_period {
			let idx = (self.head + i) % self.fastk_period;
			let h = self.high_buffer[idx];
			let l = self.low_buffer[idx];
			if h > hh {
				hh = h;
			}
			if l < ll {
				ll = l;
			}
		}
		let k = if hh == ll {
			if self.close_buffer[(self.head + self.fastk_period - 1) % self.fastk_period] == hh {
				100.0
			} else {
				0.0
			}
		} else {
			100.0 * (self.close_buffer[(self.head + self.fastk_period - 1) % self.fastk_period] - ll) / (hh - ll)
		};

		self.k_buffer[self.k_head] = k;
		self.k_head = (self.k_head + 1) % self.fastd_period;
		if self.k_count < self.fastd_period {
			self.k_count += 1;
		}

		let d = if self.fastd_matype != 0 {
			f64::NAN
		} else if self.k_count < self.fastd_period {
			f64::NAN
		} else {
			let mut sum = 0.0;
			for i in 0..self.fastd_period {
				sum += self.k_buffer[i];
			}
			sum / (self.fastd_period as f64)
		};
		Some((k, d))
	}
}

#[derive(Clone, Debug)]
pub struct StochfBatchRange {
	pub fastk_period: (usize, usize, usize),
	pub fastd_period: (usize, usize, usize),
}

impl Default for StochfBatchRange {
	fn default() -> Self {
		Self {
			fastk_period: (5, 14, 1),
			fastd_period: (3, 3, 0),
		}
	}
}

#[derive(Clone, Debug, Default)]
pub struct StochfBatchBuilder {
	range: StochfBatchRange,
	kernel: Kernel,
}

impl StochfBatchBuilder {
	pub fn new() -> Self {
		Self::default()
	}
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	pub fn fastk_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.fastk_period = (start, end, step);
		self
	}
	pub fn fastk_static(mut self, p: usize) -> Self {
		self.range.fastk_period = (p, p, 0);
		self
	}
	pub fn fastd_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.fastd_period = (start, end, step);
		self
	}
	pub fn fastd_static(mut self, p: usize) -> Self {
		self.range.fastd_period = (p, p, 0);
		self
	}
	pub fn apply_slices(self, high: &[f64], low: &[f64], close: &[f64]) -> Result<StochfBatchOutput, StochfError> {
		stochf_batch_with_kernel(high, low, close, &self.range, self.kernel)
	}
	pub fn with_default_slices(
		high: &[f64],
		low: &[f64],
		close: &[f64],
		k: Kernel,
	) -> Result<StochfBatchOutput, StochfError> {
		StochfBatchBuilder::new().kernel(k).apply_slices(high, low, close)
	}
	pub fn apply_candles(self, c: &Candles) -> Result<StochfBatchOutput, StochfError> {
		let high = source_type(c, "high");
		let low = source_type(c, "low");
		let close = source_type(c, "close");
		self.apply_slices(high, low, close)
	}
	pub fn with_default_candles(c: &Candles) -> Result<StochfBatchOutput, StochfError> {
		StochfBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c)
	}
}

pub fn stochf_batch_with_kernel(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	sweep: &StochfBatchRange,
	k: Kernel,
) -> Result<StochfBatchOutput, StochfError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => {
			return Err(StochfError::InvalidPeriod {
				fastk: 0,
				fastd: 0,
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
	stochf_batch_par_slice(high, low, close, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct StochfBatchOutput {
	pub k: Vec<f64>,
	pub d: Vec<f64>,
	pub combos: Vec<StochfParams>,
	pub rows: usize,
	pub cols: usize,
}
impl StochfBatchOutput {
	pub fn row_for_params(&self, p: &StochfParams) -> Option<usize> {
		self.combos.iter().position(|c| {
			c.fastk_period.unwrap_or(5) == p.fastk_period.unwrap_or(5)
				&& c.fastd_period.unwrap_or(3) == p.fastd_period.unwrap_or(3)
				&& c.fastd_matype.unwrap_or(0) == p.fastd_matype.unwrap_or(0)
		})
	}
	pub fn k_for(&self, p: &StochfParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.k[start..start + self.cols]
		})
	}
	pub fn d_for(&self, p: &StochfParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.d[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &StochfBatchRange) -> Vec<StochfParams> {
	fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let fastk = axis(r.fastk_period);
	let fastd = axis(r.fastd_period);
	let mut out = Vec::with_capacity(fastk.len() * fastd.len());
	for &k in &fastk {
		for &d in &fastd {
			out.push(StochfParams {
				fastk_period: Some(k),
				fastd_period: Some(d),
				fastd_matype: Some(0),
			});
		}
	}
	out
}

#[inline(always)]
pub fn stochf_batch_slice(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	sweep: &StochfBatchRange,
	kern: Kernel,
) -> Result<StochfBatchOutput, StochfError> {
	stochf_batch_inner(high, low, close, sweep, kern, false)
}

#[inline(always)]
pub fn stochf_batch_par_slice(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	sweep: &StochfBatchRange,
	kern: Kernel,
) -> Result<StochfBatchOutput, StochfError> {
	stochf_batch_inner(high, low, close, sweep, kern, true)
}

#[inline(always)]
fn stochf_batch_inner(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	sweep: &StochfBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<StochfBatchOutput, StochfError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(StochfError::InvalidPeriod {
			fastk: 0,
			fastd: 0,
			data_len: 0,
		});
	}
	let first = (0..high.len())
		.find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
		.ok_or(StochfError::AllValuesNaN)?;
	let max_k = combos.iter().map(|c| c.fastk_period.unwrap()).max().unwrap();
	if high.len() - first < max_k {
		return Err(StochfError::NotEnoughValidData {
			needed: max_k,
			valid: high.len() - first,
		});
	}
	let rows = combos.len();
	let cols = high.len();
	let mut k_out = vec![f64::NAN; rows * cols];
	let mut d_out = vec![f64::NAN; rows * cols];
	let do_row = |row: usize, kout: &mut [f64], dout: &mut [f64]| unsafe {
		let fastk_period = combos[row].fastk_period.unwrap();
		let fastd_period = combos[row].fastd_period.unwrap();
		let matype = combos[row].fastd_matype.unwrap();
		match kern {
			Kernel::Scalar => {
				stochf_row_scalar(high, low, close, first, fastk_period, fastd_period, matype, kout, dout)
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => stochf_row_avx2(high, low, close, first, fastk_period, fastd_period, matype, kout, dout),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => stochf_row_avx512(high, low, close, first, fastk_period, fastd_period, matype, kout, dout),
			_ => unreachable!(),
		}
	};
	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			k_out
				.par_chunks_mut(cols)
				.zip(d_out.par_chunks_mut(cols))
				.enumerate()
				.for_each(|(row, (k, d))| do_row(row, k, d));
		}

		#[cfg(target_arch = "wasm32")]
		{
			for (row, (k, d)) in k_out.chunks_mut(cols).zip(d_out.chunks_mut(cols)).enumerate() {
				do_row(row, k, d);
			}
		}
	} else {
		for (row, (k, d)) in k_out.chunks_mut(cols).zip(d_out.chunks_mut(cols)).enumerate() {
			do_row(row, k, d);
		}
	}
	Ok(StochfBatchOutput {
		k: k_out,
		d: d_out,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
unsafe fn stochf_row_scalar(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	first: usize,
	fastk_period: usize,
	fastd_period: usize,
	matype: usize,
	k_out: &mut [f64],
	d_out: &mut [f64],
) {
	let len = high.len();
	for i in first..len {
		if i < first + fastk_period - 1 {
			continue;
		}
		let start = i + 1 - fastk_period;
		let (mut hh, mut ll) = (f64::NEG_INFINITY, f64::INFINITY);
		for j in start..=i {
			let h = high[j];
			let l = low[j];
			if h > hh {
				hh = h;
			}
			if l < ll {
				ll = l;
			}
		}
		if hh == ll {
			k_out[i] = if close[i] == hh { 100.0 } else { 0.0 };
		} else {
			k_out[i] = 100.0 * (close[i] - ll) / (hh - ll);
		}
	}
	if matype != 0 {
		d_out.fill(f64::NAN);
	} else {
		let mut sma_sum = 0.0;
		let mut count = 0;
		for i in 0..len {
			let v = k_out[i];
			if v.is_nan() {
				d_out[i] = f64::NAN;
				continue;
			}
			if count < fastd_period {
				sma_sum += v;
				count += 1;
				if count == fastd_period {
					d_out[i] = sma_sum / (fastd_period as f64);
				} else {
					d_out[i] = f64::NAN;
				}
			} else {
				sma_sum += v - k_out[i - fastd_period];
				d_out[i] = sma_sum / (fastd_period as f64);
			}
		}
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
unsafe fn stochf_row_avx2(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	first: usize,
	fastk_period: usize,
	fastd_period: usize,
	matype: usize,
	k_out: &mut [f64],
	d_out: &mut [f64],
) {
	stochf_row_scalar(
		high,
		low,
		close,
		first,
		fastk_period,
		fastd_period,
		matype,
		k_out,
		d_out,
	);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn stochf_row_avx512(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	first: usize,
	fastk_period: usize,
	fastd_period: usize,
	matype: usize,
	k_out: &mut [f64],
	d_out: &mut [f64],
) {
	if fastk_period <= 32 {
		stochf_row_avx512_short(
			high,
			low,
			close,
			first,
			fastk_period,
			fastd_period,
			matype,
			k_out,
			d_out,
		);
	} else {
		stochf_row_avx512_long(
			high,
			low,
			close,
			first,
			fastk_period,
			fastd_period,
			matype,
			k_out,
			d_out,
		);
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn stochf_row_avx512_short(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	first: usize,
	fastk_period: usize,
	fastd_period: usize,
	matype: usize,
	k_out: &mut [f64],
	d_out: &mut [f64],
) {
	stochf_row_scalar(
		high,
		low,
		close,
		first,
		fastk_period,
		fastd_period,
		matype,
		k_out,
		d_out,
	);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn stochf_row_avx512_long(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	first: usize,
	fastk_period: usize,
	fastd_period: usize,
	matype: usize,
	k_out: &mut [f64],
	d_out: &mut [f64],
) {
	stochf_row_scalar(
		high,
		low,
		close,
		first,
		fastk_period,
		fastd_period,
		matype,
		k_out,
		d_out,
	);
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_stochf_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = StochfParams {
			fastk_period: None,
			fastd_period: None,
			fastd_matype: None,
		};
		let input = StochfInput::from_candles(&candles, params);
		let output = stochf_with_kernel(&input, kernel)?;
		assert_eq!(output.k.len(), candles.close.len());
		assert_eq!(output.d.len(), candles.close.len());
		Ok(())
	}

	fn check_stochf_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = StochfParams {
			fastk_period: Some(5),
			fastd_period: Some(3),
			fastd_matype: Some(0),
		};
		let input = StochfInput::from_candles(&candles, params);
		let output = stochf_with_kernel(&input, kernel)?;
		let expected_k = [
			80.6987399770905,
			40.88471849865952,
			15.507246376811594,
			36.920529801324506,
			32.1880650994575,
		];
		let expected_d = [
			70.99960994145033,
			61.44725644908976,
			45.696901617520815,
			31.104164892265487,
			28.205280425864817,
		];
		let k_slice = &output.k[output.k.len() - 5..];
		let d_slice = &output.d[output.d.len() - 5..];
		for i in 0..5 {
			assert!((k_slice[i] - expected_k[i]).abs() < 1e-4, "K mismatch at idx {}", i);
			assert!((d_slice[i] - expected_d[i]).abs() < 1e-4, "D mismatch at idx {}", i);
		}
		Ok(())
	}

	fn check_stochf_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = StochfInput::with_default_candles(&candles);
		let output = stochf_with_kernel(&input, kernel)?;
		assert_eq!(output.k.len(), candles.close.len());
		assert_eq!(output.d.len(), candles.close.len());
		Ok(())
	}

	fn check_stochf_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data = [10.0, 20.0, 30.0, 40.0, 50.0];
		let params = StochfParams {
			fastk_period: Some(0),
			fastd_period: Some(3),
			fastd_matype: Some(0),
		};
		let input = StochfInput::from_slices(&data, &data, &data, params);
		let res = stochf_with_kernel(&input, kernel);
		assert!(res.is_err());
		Ok(())
	}

	fn check_stochf_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data = [10.0, 20.0, 30.0];
		let params = StochfParams {
			fastk_period: Some(10),
			fastd_period: Some(3),
			fastd_matype: Some(0),
		};
		let input = StochfInput::from_slices(&data, &data, &data, params);
		let res = stochf_with_kernel(&input, kernel);
		assert!(res.is_err());
		Ok(())
	}

	fn check_stochf_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data = [42.0];
		let params = StochfParams {
			fastk_period: Some(9),
			fastd_period: Some(3),
			fastd_matype: Some(0),
		};
		let input = StochfInput::from_slices(&data, &data, &data, params);
		let res = stochf_with_kernel(&input, kernel);
		assert!(res.is_err());
		Ok(())
	}

	fn check_stochf_slice_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = StochfParams {
			fastk_period: Some(5),
			fastd_period: Some(3),
			fastd_matype: Some(0),
		};
		let input1 = StochfInput::from_candles(&candles, params.clone());
		let res1 = stochf_with_kernel(&input1, kernel)?;
		let input2 = StochfInput::from_slices(&res1.k, &res1.k, &res1.k, params);
		let res2 = stochf_with_kernel(&input2, kernel)?;
		assert_eq!(res2.k.len(), res1.k.len());
		assert_eq!(res2.d.len(), res1.d.len());
		Ok(())
	}

	macro_rules! generate_all_stochf_tests {
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

	generate_all_stochf_tests!(
		check_stochf_partial_params,
		check_stochf_accuracy,
		check_stochf_default_candles,
		check_stochf_zero_period,
		check_stochf_period_exceeds_length,
		check_stochf_very_small_dataset,
		check_stochf_slice_reinput
	);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let output = StochfBatchBuilder::new().kernel(kernel).apply_candles(&c)?;
		let def = StochfParams::default();
		let krow = output.k_for(&def).expect("default row missing");
		let drow = output.d_for(&def).expect("default row missing");
		assert_eq!(krow.len(), c.close.len());
		assert_eq!(drow.len(), c.close.len());
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
