//! # Coppock Curve (CC)
//!
//! The Coppock Curve is a momentum indicator that sums two different ROC values
//! (long and short), and then smooths the sum with a chosen MA (e.g. WMA, SMA, etc.).
//!
//! Classic defaults:
//! - Short ROC = 11
//! - Long ROC = 14
//! - MA period = 10
//! - MA type = "wma"
//!
//! ## Parameters
//! - **short_roc_period**: Period for short ROC (defaults to 11).
//! - **long_roc_period**: Period for long ROC (defaults to 14).
//! - **ma_period**: Period for smoothing (defaults to 10).
//! - **ma_type**: Type of MA (e.g., `"wma"`, `"ema"`, `"sma"`). Defaults to `"wma"`.
//! - **source**: Candle field (e.g. `"close"`, `"hlc3"`). Defaults to `"close"`.
//!
//! ## Errors
//! - **EmptyData**: Input data slice is empty.
//! - **AllValuesNaN**: All data values are `NaN`.
//! - **NotEnoughValidData**: Not enough valid data to compute at least one output.
//! - **InvalidPeriod**: Zero or out-of-bounds short/long/MA periods.
//! - **MaError**: Underlying error from the `ma(...)` function.
//!
//! ## Returns
//! - `Ok(CoppockOutput)` on success, containing a vector matching the input length,
//!   with leading `NaN`s until the earliest valid index.
//! - `Err(CoppockError)` otherwise.

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
use std::convert::AsRef;
use std::error::Error;
use std::mem::ManuallyDrop;
use thiserror::Error;

impl<'a> AsRef<[f64]> for CoppockInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			CoppockData::Slice(slice) => slice,
			CoppockData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

#[derive(Debug, Clone)]
pub enum CoppockData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct CoppockOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct CoppockParams {
	pub short_roc_period: Option<usize>,
	pub long_roc_period: Option<usize>,
	pub ma_period: Option<usize>,
	pub ma_type: Option<String>,
}

impl Default for CoppockParams {
	fn default() -> Self {
		Self {
			short_roc_period: Some(11),
			long_roc_period: Some(14),
			ma_period: Some(10),
			ma_type: Some("wma".to_string()),
		}
	}
}

#[derive(Debug, Clone)]
pub struct CoppockInput<'a> {
	pub data: CoppockData<'a>,
	pub params: CoppockParams,
}

impl<'a> CoppockInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: CoppockParams) -> Self {
		Self {
			data: CoppockData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slice(sl: &'a [f64], p: CoppockParams) -> Self {
		Self {
			data: CoppockData::Slice(sl),
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", CoppockParams::default())
	}
	#[inline]
	pub fn get_short_roc_period(&self) -> usize {
		self.params.short_roc_period.unwrap_or(11)
	}
	#[inline]
	pub fn get_long_roc_period(&self) -> usize {
		self.params.long_roc_period.unwrap_or(14)
	}
	#[inline]
	pub fn get_ma_period(&self) -> usize {
		self.params.ma_period.unwrap_or(10)
	}
	#[inline]
	pub fn get_ma_type(&self) -> String {
		self.params.ma_type.clone().unwrap_or_else(|| "wma".to_string())
	}
}

#[derive(Clone, Debug)]
pub struct CoppockBuilder {
	short: Option<usize>,
	long: Option<usize>,
	ma: Option<usize>,
	ma_type: Option<String>,
	kernel: Kernel,
}

impl Default for CoppockBuilder {
	fn default() -> Self {
		Self {
			short: None,
			long: None,
			ma: None,
			ma_type: None,
			kernel: Kernel::Auto,
		}
	}
}

impl CoppockBuilder {
	#[inline(always)]
	pub fn new() -> Self {
		Self::default()
	}
	#[inline(always)]
	pub fn short_roc_period(mut self, n: usize) -> Self {
		self.short = Some(n);
		self
	}
	#[inline(always)]
	pub fn long_roc_period(mut self, n: usize) -> Self {
		self.long = Some(n);
		self
	}
	#[inline(always)]
	pub fn ma_period(mut self, n: usize) -> Self {
		self.ma = Some(n);
		self
	}
	#[inline(always)]
	pub fn ma_type<T: Into<String>>(mut self, t: T) -> Self {
		self.ma_type = Some(t.into());
		self
	}
	#[inline(always)]
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline(always)]
	pub fn apply(self, c: &Candles) -> Result<CoppockOutput, CoppockError> {
		let p = CoppockParams {
			short_roc_period: self.short,
			long_roc_period: self.long,
			ma_period: self.ma,
			ma_type: self.ma_type,
		};
		let i = CoppockInput::from_candles(c, "close", p);
		coppock_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slice(self, d: &[f64]) -> Result<CoppockOutput, CoppockError> {
		let p = CoppockParams {
			short_roc_period: self.short,
			long_roc_period: self.long,
			ma_period: self.ma,
			ma_type: self.ma_type,
		};
		let i = CoppockInput::from_slice(d, p);
		coppock_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<CoppockStream, CoppockError> {
		let p = CoppockParams {
			short_roc_period: self.short,
			long_roc_period: self.long,
			ma_period: self.ma,
			ma_type: self.ma_type,
		};
		CoppockStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum CoppockError {
	#[error("coppock: Empty data provided.")]
	EmptyData,
	#[error("coppock: All values are NaN.")]
	AllValuesNaN,
	#[error("coppock: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("coppock: Invalid period usage => short={short}, long={long}, ma={ma}, data_len={data_len}")]
	InvalidPeriod {
		short: usize,
		long: usize,
		ma: usize,
		data_len: usize,
	},
	#[error("coppock: Underlying MA error: {0}")]
	MaError(#[from] Box<dyn Error>),
}

#[inline]
pub fn coppock(input: &CoppockInput) -> Result<CoppockOutput, CoppockError> {
	coppock_with_kernel(input, Kernel::Auto)
}

pub fn coppock_with_kernel(input: &CoppockInput, kernel: Kernel) -> Result<CoppockOutput, CoppockError> {
	let data: &[f64] = input.as_ref();
	if data.is_empty() {
		return Err(CoppockError::EmptyData);
	}

	let short = input.get_short_roc_period();
	let long = input.get_long_roc_period();
	let ma_p = input.get_ma_period();
	let data_len = data.len();

	if short == 0 || long == 0 || ma_p == 0 || short > data_len || long > data_len || ma_p > data_len {
		return Err(CoppockError::InvalidPeriod {
			short,
			long,
			ma: ma_p,
			data_len,
		});
	}

	let first = data
		.iter()
		.position(|&x| !x.is_nan())
		.ok_or(CoppockError::AllValuesNaN)?;
	let largest_roc = short.max(long);
	if (data_len - first) < largest_roc {
		return Err(CoppockError::NotEnoughValidData {
			needed: largest_roc,
			valid: data_len - first,
		});
	}

	// Calculate warmup period for sum_roc
	let warmup_period = first + largest_roc;

	// REPLACE: let mut sum_roc = AVec::<f64>::with_capacity(CACHELINE_ALIGN, data_len);
	// REPLACE: sum_roc.resize(data_len, f64::NAN);
	// WITH:
	let mut sum_roc = alloc_with_nan_prefix(data_len, warmup_period);

	unsafe {
		match match kernel {
			Kernel::Auto => detect_best_kernel(),
			other => other,
		} {
			Kernel::Scalar | Kernel::ScalarBatch => coppock_scalar(data, short, long, first, &mut sum_roc),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => coppock_avx2(data, short, long, first, &mut sum_roc),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => coppock_avx512(data, short, long, first, &mut sum_roc),
			_ => unreachable!(),
		}
	}

	let ma_type = input.get_ma_type();
	let smoothed = ma(&ma_type, MaData::Slice(&sum_roc), ma_p).map_err(CoppockError::MaError)?;

	Ok(CoppockOutput { values: smoothed })
}

#[inline]
pub fn coppock_scalar(data: &[f64], short: usize, long: usize, first: usize, out: &mut [f64]) {
	let largest = short.max(long);
	let start_idx = first + largest;
	for i in start_idx..data.len() {
		let current = data[i];
		let prev_short = data[i - short];
		let short_val = ((current / prev_short) - 1.0) * 100.0;
		let prev_long = data[i - long];
		let long_val = ((current / prev_long) - 1.0) * 100.0;
		out[i] = short_val + long_val;
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
pub unsafe fn coppock_avx2(data: &[f64], short: usize, long: usize, first: usize, out: &mut [f64]) {
	coppock_scalar(data, short, long, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
pub unsafe fn coppock_avx512(data: &[f64], short: usize, long: usize, first: usize, out: &mut [f64]) {
	if short.max(long) <= 32 {
		coppock_avx512_short(data, short, long, first, out)
	} else {
		coppock_avx512_long(data, short, long, first, out)
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
pub unsafe fn coppock_avx512_short(data: &[f64], short: usize, long: usize, first: usize, out: &mut [f64]) {
	coppock_scalar(data, short, long, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
pub unsafe fn coppock_avx512_long(data: &[f64], short: usize, long: usize, first: usize, out: &mut [f64]) {
	coppock_scalar(data, short, long, first, out)
}

#[derive(Debug, Clone)]
pub struct CoppockStream {
	short: usize,
	long: usize,
	ma_period: usize,
	ma_type: String,
	buffer: Vec<f64>,
	head: usize,
	filled: bool,
	ma_buf: Vec<f64>,
	ma_head: usize,
	ma_filled: bool,
}

impl CoppockStream {
	pub fn try_new(params: CoppockParams) -> Result<Self, CoppockError> {
		let short = params.short_roc_period.unwrap_or(11);
		let long = params.long_roc_period.unwrap_or(14);
		let ma_period = params.ma_period.unwrap_or(10);
		let ma_type = params.ma_type.unwrap_or_else(|| "wma".to_string());
		if short == 0 || long == 0 || ma_period == 0 {
			return Err(CoppockError::InvalidPeriod {
				short,
				long,
				ma: ma_period,
				data_len: 0,
			});
		}
		// buffer needs to hold the entire window for “short” and “long”
		Ok(Self {
			short,
			long,
			ma_period,
			ma_type,
			buffer: vec![f64::NAN; long.max(short) + 1],
			head: 0,
			filled: false,
			ma_buf: vec![f64::NAN; ma_period],
			ma_head: 0,
			ma_filled: false,
		})
	}

	#[inline(always)]
	pub fn update(&mut self, value: f64) -> Option<f64> {
		let n = self.buffer.len();
		// 1) Write new price into “write_idx”
		let write_idx = self.head;
		self.buffer[write_idx] = value;
		// 2) Advance head to next slot
		self.head = (write_idx + 1) % n;
		// 3) Once this ring buffer has wrapped once, “filled” = true
		if !self.filled && self.head == 0 {
			self.filled = true;
		}
		if !self.filled {
			// We haven’t yet seen “long.max(short)+1” prices => cannot compute ROC
			return None;
		}

		// 4) “idx” is the location we just wrote into
		let idx = write_idx;
		let cur = self.buffer[idx];
		let prev_short = self.buffer[(idx + n - self.short) % n];
		let prev_long = self.buffer[(idx + n - self.long) % n];
		if prev_short.is_nan() || prev_long.is_nan() || cur.is_nan() {
			return None;
		}
		// 5) Compute the two ROCs (short and long)
		let short_val = ((cur / prev_short) - 1.0) * 100.0;
		let long_val = ((cur / prev_long) - 1.0) * 100.0;
		let sum_roc = short_val + long_val;

		// 6) Push into the MA circular buffer
		let ma_n = self.ma_buf.len();
		let write_ma = self.ma_head;
		self.ma_buf[write_ma] = sum_roc;
		self.ma_head = (write_ma + 1) % ma_n;
		if !self.ma_filled && self.ma_head == 0 {
			self.ma_filled = true;
		}
		if !self.ma_filled {
			// haven’t yet seen “ma_period” values => return None
			return None;
		}

		// 7) Perform WMA or SMA over the last “ma_period” entries
		let mut smoothed = 0.0;
		if self.ma_type == "wma" {
			// denom = 1 + 2 + ⋯ + ma_n = ma_n * (ma_n + 1) / 2
			let denom = (ma_n * (ma_n + 1) / 2) as f64;
			for i in 0..ma_n {
				// “idx_in_window” = (ma_head + i) % ma_n
				// When i = ma_n - 1 => (ma_head + ma_n - 1) % ma_n is the “newest” sum_roc
				let idx2 = (self.ma_head + i) % ma_n;
				smoothed += self.ma_buf[idx2] * (i + 1) as f64;
			}
			smoothed /= denom;
		} else if self.ma_type == "sma" {
			let mut count = 0;
			for i in 0..ma_n {
				let idx2 = (self.ma_head + i) % ma_n;
				let v = self.ma_buf[idx2];
				if !v.is_nan() {
					smoothed += v;
					count += 1;
				}
			}
			if count > 0 {
				smoothed /= count as f64;
			}
		} else {
			// only "wma" and "sma" supported in streaming
			return None;
		}

		Some(smoothed)
	}
}
#[derive(Clone, Debug)]
pub struct CoppockBatchRange {
	pub short: (usize, usize, usize),
	pub long: (usize, usize, usize),
	pub ma: (usize, usize, usize),
}

impl Default for CoppockBatchRange {
	fn default() -> Self {
		Self {
			short: (11, 11, 0),
			long: (14, 14, 0),
			ma: (10, 10, 0),
		}
	}
}

#[derive(Clone, Debug, Default)]
pub struct CoppockBatchBuilder {
	range: CoppockBatchRange,
	kernel: Kernel,
}

impl CoppockBatchBuilder {
	pub fn new() -> Self {
		Self::default()
	}
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline]
	pub fn short_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.short = (start, end, step);
		self
	}
	#[inline]
	pub fn short_static(mut self, n: usize) -> Self {
		self.range.short = (n, n, 0);
		self
	}
	#[inline]
	pub fn long_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.long = (start, end, step);
		self
	}
	#[inline]
	pub fn long_static(mut self, n: usize) -> Self {
		self.range.long = (n, n, 0);
		self
	}
	#[inline]
	pub fn ma_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.ma = (start, end, step);
		self
	}
	#[inline]
	pub fn ma_static(mut self, n: usize) -> Self {
		self.range.ma = (n, n, 0);
		self
	}
	pub fn apply_slice(self, data: &[f64]) -> Result<CoppockBatchOutput, CoppockError> {
		coppock_batch_with_kernel(data, &self.range, self.kernel)
	}
	pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<CoppockBatchOutput, CoppockError> {
		CoppockBatchBuilder::new().kernel(k).apply_slice(data)
	}
	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<CoppockBatchOutput, CoppockError> {
		let slice = source_type(c, src);
		self.apply_slice(slice)
	}
	pub fn with_default_candles(c: &Candles) -> Result<CoppockBatchOutput, CoppockError> {
		CoppockBatchBuilder::new()
			.kernel(Kernel::Auto)
			.apply_candles(c, "close")
	}
}

pub fn coppock_batch_with_kernel(
	data: &[f64],
	sweep: &CoppockBatchRange,
	k: Kernel,
) -> Result<CoppockBatchOutput, CoppockError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => {
			return Err(CoppockError::InvalidPeriod {
				short: 0,
				long: 0,
				ma: 0,
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
	coppock_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct CoppockBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<CoppockParams>,
	pub rows: usize,
	pub cols: usize,
}

impl CoppockBatchOutput {
	pub fn row_for_params(&self, p: &CoppockParams) -> Option<usize> {
		self.combos.iter().position(|c| {
			c.short_roc_period.unwrap_or(11) == p.short_roc_period.unwrap_or(11)
				&& c.long_roc_period.unwrap_or(14) == p.long_roc_period.unwrap_or(14)
				&& c.ma_period.unwrap_or(10) == p.ma_period.unwrap_or(10)
		})
	}
	pub fn values_for(&self, p: &CoppockParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &CoppockBatchRange) -> Vec<CoppockParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let shorts = axis_usize(r.short);
	let longs = axis_usize(r.long);
	let mas = axis_usize(r.ma);
	let mut out = Vec::with_capacity(shorts.len() * longs.len() * mas.len());
	for &s in &shorts {
		for &l in &longs {
			for &m in &mas {
				out.push(CoppockParams {
					short_roc_period: Some(s),
					long_roc_period: Some(l),
					ma_period: Some(m),
					ma_type: Some("wma".to_string()),
				});
			}
		}
	}
	out
}

#[inline(always)]
pub fn coppock_batch_slice(
	data: &[f64],
	sweep: &CoppockBatchRange,
	kern: Kernel,
) -> Result<CoppockBatchOutput, CoppockError> {
	coppock_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn coppock_batch_par_slice(
	data: &[f64],
	sweep: &CoppockBatchRange,
	kern: Kernel,
) -> Result<CoppockBatchOutput, CoppockError> {
	coppock_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn coppock_batch_inner(
	data: &[f64],
	sweep: &CoppockBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<CoppockBatchOutput, CoppockError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(CoppockError::InvalidPeriod {
			short: 0,
			long: 0,
			ma: 0,
			data_len: 0,
		});
	}
	let first = data
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(CoppockError::AllValuesNaN)?;
	let max_roc = combos
		.iter()
		.map(|c| c.short_roc_period.unwrap().max(c.long_roc_period.unwrap()))
		.max()
		.unwrap();
	if data.len() - first < max_roc {
		return Err(CoppockError::NotEnoughValidData {
			needed: max_roc,
			valid: data.len() - first,
		});
	}

	let rows = combos.len();
	let cols = data.len();

	// Step 1: Allocate uninitialized matrix
	// REPLACE: let mut values = vec![f64::NAN; rows * cols];
	// WITH:
	let mut buf_mu = make_uninit_matrix(rows, cols);

	// Step 2: Calculate warmup periods for each row
	// For Coppock, warmup = first_valid + largest_roc + (ma_period - 1)
	let warm: Vec<usize> = combos
		.iter()
		.map(|c| {
			let short = c.short_roc_period.unwrap();
			let long = c.long_roc_period.unwrap();
			let ma_p = c.ma_period.unwrap();
			let largest = short.max(long);
			// The sum_roc starts producing values at first + largest
			// Then MA adds its own warmup of (ma_p - 1)
			first + largest + (ma_p - 1)
		})
		.collect();

	// Step 3: Initialize NaN prefixes for each row
	init_matrix_prefixes(&mut buf_mu, cols, &warm);

	// Step 4: Convert to mutable slice for computation
	let mut buf_guard = ManuallyDrop::new(buf_mu);
	let values: &mut [f64] =
		unsafe { core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len()) };

	// A single closure that (for a given row) computes:
	// 1) raw "sum_roc" into a temporary Vec<f64>
	// 2) applies ma(...) to that temp
	// 3) writes the smoothed result into out_row
	let do_row = |row: usize, out_row: &mut [f64]| {
		let c = &combos[row];
		let short = c.short_roc_period.unwrap();
		let long = c.long_roc_period.unwrap();
		let ma_p = c.ma_period.unwrap();
		let ma_type = c.ma_type.clone().unwrap_or_else(|| "wma".to_string());
		let largest = short.max(long);

		// Calculate warmup for sum_roc
		let sum_roc_warmup = first + largest;

		// Prepare a "sum_roc" buffer
		// REPLACE: let mut sum_roc = vec![f64::NAN; cols];
		// WITH:
		let mut sum_roc = alloc_with_nan_prefix(cols, sum_roc_warmup);

		// Fill sum_roc[i] = ROC_short + ROC_long for i >= first + largest
		for i in (first + largest)..cols {
			let current = data[i];
			let prev_short = data[i - short];
			let short_val = ((current / prev_short) - 1.0) * 100.0;
			let prev_long = data[i - long];
			let long_val = ((current / prev_long) - 1.0) * 100.0;
			sum_roc[i] = short_val + long_val;
		}

		// Now smooth “sum_roc” with MA
		// (We unwrap because these parameters should always be valid here.)
		let smoothed = ma(&ma_type, MaData::Slice(&sum_roc), ma_p).expect("MA error inside batch");

		// Copy the smoothed vector into this row’s slice
		out_row.copy_from_slice(&smoothed);
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

	// Step 6: Reclaim as Vec<f64>
	let values = unsafe {
		Vec::from_raw_parts(
			buf_guard.as_mut_ptr() as *mut f64,
			buf_guard.len(),
			buf_guard.capacity(),
		)
	};

	Ok(CoppockBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
pub fn coppock_row_scalar(
	data: &[f64],
	first: usize,
	short: usize,
	long: usize,
	_stride: usize,
	_w_ptr: *const f64,
	_inv_n: f64,
	out: &mut [f64],
) {
	let largest = short.max(long);
	for i in (first + largest)..data.len() {
		let current = data[i];
		let prev_short = data[i - short];
		let short_val = ((current / prev_short) - 1.0) * 100.0;
		let prev_long = data[i - long];
		let long_val = ((current / prev_long) - 1.0) * 100.0;
		out[i] = short_val + long_val;
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
pub unsafe fn coppock_row_avx2(
	data: &[f64],
	first: usize,
	short: usize,
	long: usize,
	stride: usize,
	w_ptr: *const f64,
	inv_n: f64,
	out: &mut [f64],
) {
	coppock_row_scalar(data, first, short, long, stride, w_ptr, inv_n, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
pub unsafe fn coppock_row_avx512(
	data: &[f64],
	first: usize,
	short: usize,
	long: usize,
	stride: usize,
	w_ptr: *const f64,
	inv_n: f64,
	out: &mut [f64],
) {
	if short.max(long) <= 32 {
		coppock_row_avx512_short(data, first, short, long, stride, w_ptr, inv_n, out)
	} else {
		coppock_row_avx512_long(data, first, short, long, stride, w_ptr, inv_n, out)
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
pub unsafe fn coppock_row_avx512_short(
	data: &[f64],
	first: usize,
	short: usize,
	long: usize,
	stride: usize,
	w_ptr: *const f64,
	inv_n: f64,
	out: &mut [f64],
) {
	coppock_row_scalar(data, first, short, long, stride, w_ptr, inv_n, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
pub unsafe fn coppock_row_avx512_long(
	data: &[f64],
	first: usize,
	short: usize,
	long: usize,
	stride: usize,
	w_ptr: *const f64,
	inv_n: f64,
	out: &mut [f64],
) {
	coppock_row_scalar(data, first, short, long, stride, w_ptr, inv_n, out)
}

#[inline(always)]
fn expand_grid_coppock(_r: &CoppockBatchRange) -> Vec<CoppockParams> {
	vec![CoppockParams::default()]
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_coppock_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let default_params = CoppockParams::default();
		let input = CoppockInput::from_candles(&candles, "close", default_params);
		let output = coppock_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}
	fn check_coppock_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = CoppockInput::with_default_candles(&candles);
		let result = coppock_with_kernel(&input, kernel)?;
		let expected_last_five = [
			-1.4542764618985533,
			-1.3795224034983653,
			-1.614331648987457,
			-1.9179048338714915,
			-2.1096548435774625,
		];
		let start = result.values.len().saturating_sub(5);
		for (i, &val) in result.values[start..].iter().enumerate() {
			let diff = (val - expected_last_five[i]).abs();
			assert!(
				diff < 1e-7,
				"[{}] Coppock {:?} mismatch at idx {}: got {}, expected {}",
				test_name,
				kernel,
				i,
				val,
				expected_last_five[i]
			);
		}
		Ok(())
	}
	fn check_coppock_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = CoppockInput::with_default_candles(&candles);
		match input.data {
			CoppockData::Candles { source, .. } => assert_eq!(source, "close"),
			_ => panic!("Expected CoppockData::Candles"),
		}
		let output = coppock_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}
	fn check_coppock_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let input_data = [10.0, 20.0, 30.0];
		let params = CoppockParams {
			short_roc_period: Some(0),
			long_roc_period: Some(14),
			ma_period: Some(10),
			ma_type: Some("wma".to_string()),
		};
		let input = CoppockInput::from_slice(&input_data, params);
		let res = coppock_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] Coppock should fail with zero short period",
			test_name
		);
		Ok(())
	}
	fn check_coppock_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data_small = [10.0, 20.0, 30.0];
		let params = CoppockParams {
			short_roc_period: Some(14),
			long_roc_period: Some(20),
			ma_period: Some(10),
			ma_type: Some("wma".to_string()),
		};
		let input = CoppockInput::from_slice(&data_small, params);
		let res = coppock_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] Coppock should fail with short/long>data.len()",
			test_name
		);
		Ok(())
	}
	fn check_coppock_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let single_point = [42.0];
		let params = CoppockParams {
			short_roc_period: Some(11),
			long_roc_period: Some(14),
			ma_period: Some(10),
			ma_type: Some("wma".to_string()),
		};
		let input = CoppockInput::from_slice(&single_point, params);
		let res = coppock_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] Coppock should fail with insufficient data",
			test_name
		);
		Ok(())
	}
	fn check_coppock_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let default_params = CoppockParams::default(); // short=11, long=14, ma=10, ma_type="wma"
		let first_input = CoppockInput::from_candles(&candles, "close", default_params.clone());
		let first_result = coppock_with_kernel(&first_input, kernel)?;

		let second_params = CoppockParams {
			short_roc_period: Some(5),
			long_roc_period: Some(8),
			ma_period: Some(3),
			ma_type: Some("sma".to_string()),
		};
		let second_input = CoppockInput::from_slice(&first_result.values, second_params.clone());
		let second_result = coppock_with_kernel(&second_input, kernel)?;

		assert_eq!(second_result.values.len(), first_result.values.len());

		let short1 = default_params.short_roc_period.unwrap();
		let long1 = default_params.long_roc_period.unwrap();
		let ma1 = default_params.ma_period.unwrap();
		let largest1 = short1.max(long1);
		let first_valid1 = largest1 + (ma1 - 1);

		let short2 = second_params.short_roc_period.unwrap();
		let long2 = second_params.long_roc_period.unwrap();
		let ma2 = second_params.ma_period.unwrap();
		let largest2 = short2.max(long2);
		let first_valid2 = first_valid1 + largest2 + (ma2 - 1);

		for i in first_valid2..second_result.values.len() {
			assert!(
				!second_result.values[i].is_nan(),
				"[{}] Expected no NaN after index {}, found NaN at {}",
				test_name,
				first_valid2,
				i
			);
		}

		Ok(())
	}
	fn check_coppock_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = CoppockInput::from_candles(
			&candles,
			"close",
			CoppockParams {
				short_roc_period: Some(11),
				long_roc_period: Some(14),
				ma_period: Some(10),
				ma_type: Some("wma".to_string()),
			},
		);
		let res = coppock_with_kernel(&input, kernel)?;
		assert_eq!(res.values.len(), candles.close.len());
		if res.values.len() > 30 {
			for (i, &val) in res.values[30..].iter().enumerate() {
				assert!(
					!val.is_nan(),
					"[{}] Found unexpected NaN at out-index {}",
					test_name,
					30 + i
				);
			}
		}
		Ok(())
	}
	fn check_coppock_streaming(test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let short = 11;
		let long = 14;
		let ma_period = 10;
		let ma_type = "wma".to_string();
		let input = CoppockInput::from_candles(
			&candles,
			"close",
			CoppockParams {
				short_roc_period: Some(short),
				long_roc_period: Some(long),
				ma_period: Some(ma_period),
				ma_type: Some(ma_type.clone()),
			},
		);
		let batch_output = coppock_with_kernel(&input, Kernel::Scalar)?.values;
		let mut stream = CoppockStream::try_new(CoppockParams {
			short_roc_period: Some(short),
			long_roc_period: Some(long),
			ma_period: Some(ma_period),
			ma_type: Some(ma_type),
		})?;
		let mut stream_values = Vec::with_capacity(candles.close.len());
		for &price in &candles.close {
			match stream.update(price) {
				Some(v) => stream_values.push(v),
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
				diff < 1e-8,
				"[{}] Coppock streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
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
	fn check_coppock_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		// Test with multiple parameter combinations to increase chance of catching bugs
		let param_combos = vec![
			CoppockParams {
				short_roc_period: Some(11),
				long_roc_period: Some(14),
				ma_period: Some(10),
				ma_type: Some("wma".to_string()),
			},
			CoppockParams {
				short_roc_period: Some(5),
				long_roc_period: Some(8),
				ma_period: Some(3),
				ma_type: Some("sma".to_string()),
			},
			CoppockParams {
				short_roc_period: Some(20),
				long_roc_period: Some(25),
				ma_period: Some(15),
				ma_type: Some("ema".to_string()),
			},
		];

		for params in param_combos {
			let input = CoppockInput::from_candles(&candles, "close", params);
			let output = coppock_with_kernel(&input, kernel)?;

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
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {}",
						test_name, val, bits, i
					);
				}

				// Check for init_matrix_prefixes poison (0x22222222_22222222)
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {}",
						test_name, val, bits, i
					);
				}

				// Check for make_uninit_matrix poison (0x33333333_33333333)
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {}",
						test_name, val, bits, i
					);
				}
			}
		}

		Ok(())
	}

	// Release mode stub - does nothing
	#[cfg(not(debug_assertions))]
	fn check_coppock_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(())
	}

	macro_rules! generate_all_coppock_tests {
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
	generate_all_coppock_tests!(
		check_coppock_partial_params,
		check_coppock_accuracy,
		check_coppock_default_candles,
		check_coppock_zero_period,
		check_coppock_period_exceeds_length,
		check_coppock_very_small_dataset,
		check_coppock_reinput,
		check_coppock_nan_handling,
		check_coppock_streaming,
		check_coppock_no_poison
	);
	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let output = CoppockBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;
		let def = CoppockParams::default();
		let row = output.values_for(&def).expect("default row missing");
		assert_eq!(row.len(), c.close.len());

		let expected = [
			-1.4542764618985533,
			-1.3795224034983653,
			-1.614331648987457,
			-1.9179048338714915,
			-2.1096548435774625,
		];
		let start = row.len() - 5;
		for (i, &v) in row[start..].iter().enumerate() {
			assert!(
				(v - expected[i]).abs() < 1e-7,
				"[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
			);
		}
		Ok(())
	}

	// Check for poison values in batch output - only runs in debug mode
	#[cfg(debug_assertions)]
	fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		// Test batch with multiple parameter combinations
		let output = CoppockBatchBuilder::new()
			.kernel(kernel)
			.short_range(5, 15, 5) // 5, 10, 15
			.long_range(10, 20, 5) // 10, 15, 20
			.ma_range(3, 9, 3) // 3, 6, 9
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
