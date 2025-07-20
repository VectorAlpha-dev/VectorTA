//! # Chaikin Accumulation/Distribution Oscillator (ADOSC)
//!
//! A momentum indicator that subtracts a long-term EMA of the Accumulation/Distribution Line (ADL)
//! from a short-term EMA of the ADL to identify shifts in market buying/selling pressure.
//!
//! ## Parameters
//! - **short_period**: The shorter EMA period (default: 3)
//! - **long_period**: The longer EMA period (default: 10)
//!
//! ## Errors
//! - **AllValuesNaN**: All values are NaN.
//! - **InvalidPeriod**: short or long period is zero, or longer period exceeds input length.
//! - **ShortPeriodGreaterThanLong**: short >= long.
//! - **NotEnoughValidData**: Not enough valid rows for the requested period.
//! - **EmptySlices**: At least one slice is empty.
//!
//! ## Returns
//! - `Ok(AdoscOutput)` on success, containing Vec<f64> for each row
//! - `Err(AdoscError)` otherwise.

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

use crate::utilities::data_loader::Candles;
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
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum AdoscData<'a> {
	Candles {
		candles: &'a Candles,
	},
	Slices {
		high: &'a [f64],
		low: &'a [f64],
		close: &'a [f64],
		volume: &'a [f64],
	},
}

#[derive(Debug, Clone)]
pub struct AdoscOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct AdoscParams {
	pub short_period: Option<usize>,
	pub long_period: Option<usize>,
}

impl Default for AdoscParams {
	fn default() -> Self {
		Self {
			short_period: Some(3),
			long_period: Some(10),
		}
	}
}

#[derive(Debug, Clone)]
pub struct AdoscInput<'a> {
	pub data: AdoscData<'a>,
	pub params: AdoscParams,
}

impl<'a> AdoscInput<'a> {
	#[inline]
	pub fn from_candles(candles: &'a Candles, params: AdoscParams) -> Self {
		Self {
			data: AdoscData::Candles { candles },
			params,
		}
	}
	#[inline]
	pub fn from_slices(
		high: &'a [f64],
		low: &'a [f64],
		close: &'a [f64],
		volume: &'a [f64],
		params: AdoscParams,
	) -> Self {
		Self {
			data: AdoscData::Slices {
				high,
				low,
				close,
				volume,
			},
			params,
		}
	}
	#[inline]
	pub fn with_default_candles(candles: &'a Candles) -> Self {
		Self {
			data: AdoscData::Candles { candles },
			params: AdoscParams::default(),
		}
	}
	#[inline]
	pub fn get_short_period(&self) -> usize {
		self.params.short_period.unwrap_or(3)
	}
	#[inline]
	pub fn get_long_period(&self) -> usize {
		self.params.long_period.unwrap_or(10)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct AdoscBuilder {
	short_period: Option<usize>,
	long_period: Option<usize>,
	kernel: Kernel,
}

impl Default for AdoscBuilder {
	fn default() -> Self {
		Self {
			short_period: None,
			long_period: None,
			kernel: Kernel::Auto,
		}
	}
}

impl AdoscBuilder {
	#[inline(always)]
	pub fn new() -> Self {
		Self::default()
	}
	#[inline(always)]
	pub fn short_period(mut self, n: usize) -> Self {
		self.short_period = Some(n);
		self
	}
	#[inline(always)]
	pub fn long_period(mut self, n: usize) -> Self {
		self.long_period = Some(n);
		self
	}
	#[inline(always)]
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline(always)]
	pub fn apply(self, c: &Candles) -> Result<AdoscOutput, AdoscError> {
		let p = AdoscParams {
			short_period: self.short_period,
			long_period: self.long_period,
		};
		let i = AdoscInput::from_candles(c, p);
		adosc_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slices(
		self,
		high: &[f64],
		low: &[f64],
		close: &[f64],
		volume: &[f64],
	) -> Result<AdoscOutput, AdoscError> {
		let p = AdoscParams {
			short_period: self.short_period,
			long_period: self.long_period,
		};
		let i = AdoscInput::from_slices(high, low, close, volume, p);
		adosc_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<AdoscStream, AdoscError> {
		let p = AdoscParams {
			short_period: self.short_period,
			long_period: self.long_period,
		};
		AdoscStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum AdoscError {
	#[error("adosc: All values are NaN.")]
	AllValuesNaN,
	#[error("adosc: Invalid period: short={short}, long={long}, data length={data_len}")]
	InvalidPeriod { short: usize, long: usize, data_len: usize },
	#[error("adosc: short_period must be less than long_period: short={short}, long={long}")]
	ShortPeriodGreaterThanLong { short: usize, long: usize },
	#[error("adosc: Not enough valid data: needed={needed}, valid={valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("adosc: At least one slice is empty: high={high}, low={low}, close={close}, volume={volume}")]
	EmptySlices {
		high: usize,
		low: usize,
		close: usize,
		volume: usize,
	},
	#[error("adosc: Invalid output length: expected={expected}, actual={actual}")]
	InvalidLength { expected: usize, actual: usize },
}

#[inline]
pub fn adosc(input: &AdoscInput) -> Result<AdoscOutput, AdoscError> {
	adosc_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn adosc_prepare<'a>(
	input: &'a AdoscInput,
	kernel: Kernel,
) -> Result<
	(
		// high
		&'a [f64],
		// low
		&'a [f64],
		// close
		&'a [f64],
		// volume
		&'a [f64],
		// short
		usize,
		// long
		usize,
		// first
		usize,
		// len
		usize,
		// chosen
		Kernel,
	),
	AdoscError,
> {
	let (high, low, close, volume) = match &input.data {
		AdoscData::Candles { candles } => {
			let n = candles.close.len();
			if n == 0 {
				return Err(AdoscError::EmptySlices {
					high: 0,
					low: 0,
					close: 0,
					volume: 0,
				});
			}
			(
				candles.high.as_slice(),
				candles.low.as_slice(),
				candles.close.as_slice(),
				candles.volume.as_slice(),
			)
		}
		AdoscData::Slices {
			high,
			low,
			close,
			volume,
		} => {
			if high.is_empty() || low.is_empty() || close.is_empty() || volume.is_empty() {
				return Err(AdoscError::EmptySlices {
					high: high.len(),
					low: low.len(),
					close: close.len(),
					volume: volume.len(),
				});
			}
			(*high, *low, *close, *volume)
		}
	};

	let len = close.len();
	let short = input.get_short_period();
	let long = input.get_long_period();

	// Validation checks
	if short == 0 || long == 0 || long > len {
		return Err(AdoscError::InvalidPeriod {
			short,
			long,
			data_len: len,
		});
	}
	if short >= long {
		return Err(AdoscError::ShortPeriodGreaterThanLong { short, long });
	}

	let first = 0;
	if len < long {
		return Err(AdoscError::NotEnoughValidData {
			needed: long,
			valid: len,
		});
	}

	// Kernel auto-detection only once
	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	Ok((high, low, close, volume, short, long, first, len, chosen))
}

pub fn adosc_with_kernel(input: &AdoscInput, kernel: Kernel) -> Result<AdoscOutput, AdoscError> {
	let (high, low, close, volume, short, long, first, len, chosen) = adosc_prepare(input, kernel)?;

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => adosc_scalar(high, low, close, volume, short, long, first, len),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => adosc_avx2(high, low, close, volume, short, long, first, len),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => adosc_avx512(high, low, close, volume, short, long, first, len),
			#[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
			Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
				adosc_scalar(high, low, close, volume, short, long, first, len)
			}
			_ => unreachable!(),
		}
	}
}

#[inline(always)]
pub unsafe fn adosc_scalar(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	short: usize,
	long: usize,
	_first: usize,
	len: usize,
) -> Result<AdoscOutput, AdoscError> {
	let alpha_short = 2.0 / (short as f64 + 1.0);
	let alpha_long = 2.0 / (long as f64 + 1.0);

	// ADOSC starts computing from index 0, no warmup period
	let mut adosc_values = alloc_with_nan_prefix(len, 0);
	let mut sum_ad = 0.0;
	let h = high[0];
	let l = low[0];
	let c = close[0];
	let v = volume[0];
	let hl = h - l;
	let mfm = if hl != 0.0 { ((c - l) - (h - c)) / hl } else { 0.0 };
	let mfv = mfm * v;
	sum_ad += mfv;
	let mut short_ema = sum_ad;
	let mut long_ema = sum_ad;
	adosc_values[0] = short_ema - long_ema;

	for i in 1..len {
		let h = high[i];
		let l = low[i];
		let c = close[i];
		let v = volume[i];
		let hl = h - l;
		let mfm = if hl != 0.0 { ((c - l) - (h - c)) / hl } else { 0.0 };
		let mfv = mfm * v;
		sum_ad += mfv;
		short_ema = alpha_short * sum_ad + (1.0 - alpha_short) * short_ema;
		long_ema = alpha_long * sum_ad + (1.0 - alpha_long) * long_ema;
		adosc_values[i] = short_ema - long_ema;
	}
	Ok(AdoscOutput { values: adosc_values })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn adosc_avx2(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	short: usize,
	long: usize,
	first: usize,
	len: usize,
) -> Result<AdoscOutput, AdoscError> {
	adosc_scalar(high, low, close, volume, short, long, first, len)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn adosc_avx512(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	short: usize,
	long: usize,
	first: usize,
	len: usize,
) -> Result<AdoscOutput, AdoscError> {
	adosc_scalar(high, low, close, volume, short, long, first, len)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn adosc_avx512_short(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	short: usize,
	long: usize,
	first: usize,
	len: usize,
) -> Result<AdoscOutput, AdoscError> {
	adosc_scalar(high, low, close, volume, short, long, first, len)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn adosc_avx512_long(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	short: usize,
	long: usize,
	first: usize,
	len: usize,
) -> Result<AdoscOutput, AdoscError> {
	adosc_scalar(high, low, close, volume, short, long, first, len)
}

#[derive(Clone, Debug)]
pub struct AdoscBatchRange {
	pub short_period: (usize, usize, usize),
	pub long_period: (usize, usize, usize),
}

impl Default for AdoscBatchRange {
	fn default() -> Self {
		Self {
			short_period: (3, 10, 1),
			long_period: (10, 30, 1),
		}
	}
}

#[derive(Clone, Debug, Default)]
pub struct AdoscBatchBuilder {
	range: AdoscBatchRange,
	kernel: Kernel,
}

impl AdoscBatchBuilder {
	pub fn new() -> Self {
		Self::default()
	}
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline]
	pub fn short_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.short_period = (start, end, step);
		self
	}
	#[inline]
	pub fn long_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.long_period = (start, end, step);
		self
	}
	pub fn apply_slices(
		self,
		high: &[f64],
		low: &[f64],
		close: &[f64],
		volume: &[f64],
	) -> Result<AdoscBatchOutput, AdoscError> {
		adosc_batch_with_kernel(high, low, close, volume, &self.range, self.kernel)
	}
	pub fn apply_candles(self, candles: &Candles) -> Result<AdoscBatchOutput, AdoscError> {
		self.apply_slices(
			candles.high.as_slice(),
			candles.low.as_slice(),
			candles.close.as_slice(),
			candles.volume.as_slice(),
		)
	}
}

#[derive(Clone, Debug)]
pub struct AdoscBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<AdoscParams>,
	pub rows: usize,
	pub cols: usize,
}
impl AdoscBatchOutput {
	pub fn row_for_params(&self, p: &AdoscParams) -> Option<usize> {
		self.combos.iter().position(|c| {
			c.short_period.unwrap_or(3) == p.short_period.unwrap_or(3)
				&& c.long_period.unwrap_or(10) == p.long_period.unwrap_or(10)
		})
	}
	pub fn values_for(&self, p: &AdoscParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

fn expand_grid(r: &AdoscBatchRange) -> Vec<AdoscParams> {
	fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let shorts = axis(r.short_period);
	let longs = axis(r.long_period);

	let mut out = Vec::new();
	for &short in &shorts {
		for &long in &longs {
			if short == 0 || long == 0 || short >= long {
				continue;
			}
			out.push(AdoscParams {
				short_period: Some(short),
				long_period: Some(long),
			});
		}
	}
	out
}

pub fn adosc_batch_with_kernel(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	sweep: &AdoscBatchRange,
	k: Kernel,
) -> Result<AdoscBatchOutput, AdoscError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => {
			return Err(AdoscError::InvalidPeriod {
				short: 0,
				long: 0,
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
	adosc_batch_par_slice(high, low, close, volume, sweep, simd)
}

pub fn adosc_batch_slice(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	sweep: &AdoscBatchRange,
	kern: Kernel,
) -> Result<AdoscBatchOutput, AdoscError> {
	adosc_batch_inner(high, low, close, volume, sweep, kern, false)
}

pub fn adosc_batch_par_slice(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	sweep: &AdoscBatchRange,
	kern: Kernel,
) -> Result<AdoscBatchOutput, AdoscError> {
	adosc_batch_inner(high, low, close, volume, sweep, kern, true)
}

fn adosc_batch_inner(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	sweep: &AdoscBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<AdoscBatchOutput, AdoscError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(AdoscError::InvalidPeriod {
			short: 0,
			long: 0,
			data_len: 0,
		});
	}
	let first = 0;
	let len = close.len();
	let rows = combos.len();
	let cols = len;

	// Use zero-copy memory allocation
	let mut buf_mu = make_uninit_matrix(rows, cols);

	// ADOSC computes from index 0, so warmup period is 0 for all rows
	let warm: Vec<usize> = vec![0; rows];
	init_matrix_prefixes(&mut buf_mu, cols, &warm);

	// Convert to mutable slice for computation
	let mut buf_guard = std::mem::ManuallyDrop::new(buf_mu);
	let values: &mut [f64] =
		unsafe { std::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len()) };
	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let prm = &combos[row];
		let short = prm.short_period.unwrap();
		let long = prm.long_period.unwrap();
		match kern {
			Kernel::Scalar => {
				let out = adosc_row_scalar(high, low, close, volume, short, long, first, out_row);
				if out.is_err() {
					out_row.fill(f64::NAN);
				}
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => {
				let out = adosc_row_avx2(high, low, close, volume, short, long, first, out_row);
				if out.is_err() {
					out_row.fill(f64::NAN);
				}
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => {
				let out = adosc_row_avx512(high, low, close, volume, short, long, first, out_row);
				if out.is_err() {
					out_row.fill(f64::NAN);
				}
			}
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

	// Reclaim as Vec<f64>
	let values = unsafe {
		Vec::from_raw_parts(
			buf_guard.as_mut_ptr() as *mut f64,
			buf_guard.len(),
			buf_guard.capacity(),
		)
	};

	Ok(AdoscBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
pub fn adosc_batch_inner_into(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	sweep: &AdoscBatchRange,
	kern: Kernel,
	parallel: bool,
	out: &mut [f64],
) -> Result<Vec<AdoscParams>, AdoscError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(AdoscError::InvalidPeriod {
			short: 0,
			long: 0,
			data_len: 0,
		});
	}
	let first = 0;
	let len = close.len();
	let cols = len;

	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let prm = &combos[row];
		let short = prm.short_period.unwrap();
		let long = prm.long_period.unwrap();
		match kern {
			Kernel::Scalar => {
				let out = adosc_row_scalar(high, low, close, volume, short, long, first, out_row);
				if out.is_err() {
					out_row.fill(f64::NAN);
				}
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => {
				let out = adosc_row_avx2(high, low, close, volume, short, long, first, out_row);
				if out.is_err() {
					out_row.fill(f64::NAN);
				}
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => {
				let out = adosc_row_avx512(high, low, close, volume, short, long, first, out_row);
				if out.is_err() {
					out_row.fill(f64::NAN);
				}
			}
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
pub unsafe fn adosc_row_scalar(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	short: usize,
	long: usize,
	_first: usize,
	out: &mut [f64],
) -> Result<(), AdoscError> {
	let len = out.len();
	let alpha_short = 2.0 / (short as f64 + 1.0);
	let alpha_long = 2.0 / (long as f64 + 1.0);
	let mut sum_ad = 0.0;
	let h = high[0];
	let l = low[0];
	let c = close[0];
	let v = volume[0];
	let hl = h - l;
	let mfm = if hl != 0.0 { ((c - l) - (h - c)) / hl } else { 0.0 };
	let mfv = mfm * v;
	sum_ad += mfv;
	let mut short_ema = sum_ad;
	let mut long_ema = sum_ad;
	out[0] = short_ema - long_ema;
	for i in 1..len {
		let h = high[i];
		let l = low[i];
		let c = close[i];
		let v = volume[i];
		let hl = h - l;
		let mfm = if hl != 0.0 { ((c - l) - (h - c)) / hl } else { 0.0 };
		let mfv = mfm * v;
		sum_ad += mfv;
		short_ema = alpha_short * sum_ad + (1.0 - alpha_short) * short_ema;
		long_ema = alpha_long * sum_ad + (1.0 - alpha_long) * long_ema;
		out[i] = short_ema - long_ema;
	}
	Ok(())
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn adosc_row_avx2(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	short: usize,
	long: usize,
	first: usize,
	out: &mut [f64],
) -> Result<(), AdoscError> {
	adosc_row_scalar(high, low, close, volume, short, long, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn adosc_row_avx512(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	short: usize,
	long: usize,
	first: usize,
	out: &mut [f64],
) -> Result<(), AdoscError> {
	adosc_row_scalar(high, low, close, volume, short, long, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn adosc_row_avx512_short(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	short: usize,
	long: usize,
	first: usize,
	out: &mut [f64],
) -> Result<(), AdoscError> {
	adosc_row_scalar(high, low, close, volume, short, long, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn adosc_row_avx512_long(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	short: usize,
	long: usize,
	first: usize,
	out: &mut [f64],
) -> Result<(), AdoscError> {
	adosc_row_scalar(high, low, close, volume, short, long, first, out)
}

pub struct AdoscStream {
	short_period: usize,
	long_period: usize,
	alpha_short: f64,
	alpha_long: f64,
	sum_ad: f64,
	short_ema: f64,
	long_ema: f64,
	initialized: bool,
}

impl AdoscStream {
	pub fn try_new(params: AdoscParams) -> Result<Self, AdoscError> {
		let short = params.short_period.unwrap_or(3);
		let long = params.long_period.unwrap_or(10);
		if short == 0 || long == 0 {
			return Err(AdoscError::InvalidPeriod {
				short,
				long,
				data_len: 0,
			});
		}
		if short >= long {
			return Err(AdoscError::ShortPeriodGreaterThanLong { short, long });
		}
		Ok(Self {
			short_period: short,
			long_period: long,
			alpha_short: 2.0 / (short as f64 + 1.0),
			alpha_long: 2.0 / (long as f64 + 1.0),
			sum_ad: 0.0,
			short_ema: 0.0,
			long_ema: 0.0,
			initialized: false,
		})
	}
	#[inline(always)]
	pub fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> f64 {
		let hl = high - low;
		let mfm = if hl != 0.0 {
			((close - low) - (high - close)) / hl
		} else {
			0.0
		};
		let mfv = mfm * volume;
		self.sum_ad += mfv;
		if !self.initialized {
			self.short_ema = self.sum_ad;
			self.long_ema = self.sum_ad;
			self.initialized = true;
		} else {
			self.short_ema = self.alpha_short * self.sum_ad + (1.0 - self.alpha_short) * self.short_ema;
			self.long_ema = self.alpha_long * self.sum_ad + (1.0 - self.alpha_long) * self.long_ema;
		}
		self.short_ema - self.long_ema
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_adosc_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = AdoscInput::with_default_candles(&candles);
		let result = adosc_with_kernel(&input, kernel)?;
		assert_eq!(result.values.len(), candles.close.len());
		let expected_last_five = [-166.2175, -148.9983, -144.9052, -128.5921, -142.0772];
		let start_index = result.values.len().saturating_sub(5);
		let result_last_five = &result.values[start_index..];
		for (i, &actual) in result_last_five.iter().enumerate() {
			let expected = expected_last_five[i];
			assert!(
				(actual - expected).abs() < 1e-1,
				"ADOSC value mismatch at index {}: expected {}, got {}",
				i,
				expected,
				actual
			);
		}
		for (i, &val) in result.values.iter().enumerate() {
			assert!(
				val.is_finite(),
				"ADOSC output at index {} should be finite, got {}",
				i,
				val
			);
		}
		Ok(())
	}

	fn check_adosc_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let partial_params = AdoscParams {
			short_period: Some(2),
			long_period: None,
		};
		let input = AdoscInput::from_candles(&candles, partial_params);
		let result = adosc_with_kernel(&input, kernel)?;
		assert_eq!(result.values.len(), candles.close.len());
		let missing_short = AdoscParams {
			short_period: None,
			long_period: Some(12),
		};
		let input_missing = AdoscInput::from_candles(&candles, missing_short);
		let result_missing = adosc_with_kernel(&input_missing, kernel)?;
		assert_eq!(result_missing.values.len(), candles.close.len());
		Ok(())
	}

	fn check_adosc_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = AdoscInput::with_default_candles(&candles);
		match input.data {
			AdoscData::Candles { .. } => {}
			_ => panic!("Expected AdoscData::Candles variant"),
		}
		let result = adosc_with_kernel(&input, kernel)?;
		assert_eq!(result.values.len(), candles.close.len());
		Ok(())
	}

	fn check_adosc_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [10.0, 10.0, 10.0];
		let low = [5.0, 5.0, 5.0];
		let close = [7.0, 7.0, 7.0];
		let volume = [1000.0, 1000.0, 1000.0];
		let zero_short = AdoscParams {
			short_period: Some(0),
			long_period: Some(10),
		};
		let input = AdoscInput::from_slices(&high, &low, &close, &volume, zero_short);
		let result = adosc_with_kernel(&input, kernel);
		assert!(result.is_err());
		let zero_long = AdoscParams {
			short_period: Some(3),
			long_period: Some(0),
		};
		let input2 = AdoscInput::from_slices(&high, &low, &close, &volume, zero_long);
		let result2 = adosc_with_kernel(&input2, kernel);
		assert!(result2.is_err());
		Ok(())
	}

	fn check_adosc_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [10.0, 11.0, 12.0];
		let low = [5.0, 5.5, 6.0];
		let close = [7.0, 8.0, 9.0];
		let volume = [1000.0, 1000.0, 1000.0];
		let params = AdoscParams {
			short_period: Some(3),
			long_period: Some(10),
		};
		let input = AdoscInput::from_slices(&high, &low, &close, &volume, params);
		let result = adosc_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}

	fn check_adosc_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [10.0];
		let low = [5.0];
		let close = [7.0];
		let volume = [1000.0];
		let params = AdoscParams {
			short_period: Some(3),
			long_period: Some(10),
		};
		let input = AdoscInput::from_slices(&high, &low, &close, &volume, params);
		let result = adosc_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}

	fn check_adosc_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let first_params = AdoscParams {
			short_period: Some(3),
			long_period: Some(10),
		};
		let first_input = AdoscInput::from_candles(&candles, first_params);
		let first_result = adosc_with_kernel(&first_input, kernel)?;
		assert_eq!(first_result.values.len(), candles.close.len());
		let second_params = AdoscParams {
			short_period: Some(2),
			long_period: Some(6),
		};
		let second_input = AdoscInput::from_slices(
			&first_result.values,
			&first_result.values,
			&first_result.values,
			&first_result.values,
			second_params,
		);
		let second_result = adosc_with_kernel(&second_input, kernel)?;
		assert_eq!(second_result.values.len(), first_result.values.len());
		Ok(())
	}

	fn check_adosc_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = AdoscInput::from_candles(&candles, AdoscParams::default());
		let result = adosc_with_kernel(&input, kernel)?;
		assert_eq!(result.values.len(), candles.close.len());
		if result.values.len() > 240 {
			for (i, &val) in result.values[240..].iter().enumerate() {
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

	fn check_adosc_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = AdoscParams {
			short_period: Some(3),
			long_period: Some(10),
		};
		let input = AdoscInput::from_candles(&candles, params.clone());
		let batch_output = adosc_with_kernel(&input, kernel)?.values;
		let mut stream = AdoscStream::try_new(params)?;
		let mut stream_values = Vec::with_capacity(candles.close.len());
		for ((&h, &l), (&c, &v)) in candles
			.high
			.iter()
			.zip(candles.low.iter())
			.zip(candles.close.iter().zip(candles.volume.iter()))
		{
			stream_values.push(stream.update(h, l, c, v));
		}
		assert_eq!(batch_output.len(), stream_values.len());
		for (i, (&b, &s)) in batch_output.iter().zip(stream_values.iter()).enumerate() {
			let diff = (b - s).abs();
			assert!(
				diff < 1e-9,
				"[{}] ADOSC streaming mismatch at idx {}: batch={}, stream={}, diff={}",
				test_name,
				i,
				b,
				s,
				diff
			);
		}
		Ok(())
	}

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let batch = AdoscBatchBuilder::new().kernel(kernel).apply_candles(&candles)?;
		let def = AdoscParams::default();
		let row = batch.values_for(&def).expect("default row missing");
		assert_eq!(row.len(), candles.close.len());
		Ok(())
	}

	macro_rules! generate_all_adosc_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(
                    #[test]
                    fn [<$test_fn _scalar_f64>]() { let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar); }
                )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(
                    #[test]
                    fn [<$test_fn _avx2_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2); }
                    #[test]
                    fn [<$test_fn _avx512_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512); }
                )*
            }
        }
    }

	generate_all_adosc_tests!(
		check_adosc_accuracy,
		check_adosc_partial_params,
		check_adosc_default_candles,
		check_adosc_zero_period,
		check_adosc_period_exceeds_length,
		check_adosc_very_small_dataset,
		check_adosc_reinput,
		check_adosc_nan_handling,
		check_adosc_streaming
	);

	macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste::paste! {
                #[test] fn [<$fn_name _scalar>]()      { let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch); }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx2>]()        { let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch); }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx512>]()      { let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch); }
                #[test] fn [<$fn_name _auto_detect>]() { let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto); }
            }
        };
    }

	gen_batch_tests!(check_batch_default_row);
}

/// Write ADOSC directly to output slice - no allocations
#[inline]
pub fn adosc_into_slice(dst: &mut [f64], input: &AdoscInput, kern: Kernel) -> Result<(), AdoscError> {
	let (high, low, close, volume, short, long, first, len, chosen) = adosc_prepare(input, kern)?;
	
	if dst.len() != len {
		return Err(AdoscError::InvalidLength {
			expected: len,
			actual: dst.len(),
		});
	}
	
	// Compute directly into the output slice
	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => {
				let alpha_short = 2.0 / (short as f64 + 1.0);
				let alpha_long = 2.0 / (long as f64 + 1.0);
				
				// ADOSC starts computing from index 0, no warmup period
				let mut sum_ad = 0.0;
				let h = high[0];
				let l = low[0];
				let c = close[0];
				let v = volume[0];
				let hl = h - l;
				let mfm = if hl != 0.0 { ((c - l) - (h - c)) / hl } else { 0.0 };
				let mfv = mfm * v;
				sum_ad += mfv;
				let mut short_ema = sum_ad;
				let mut long_ema = sum_ad;
				dst[0] = short_ema - long_ema;
				
				for i in 1..len {
					let h = high[i];
					let l = low[i];
					let c = close[i];
					let v = volume[i];
					let hl = h - l;
					let mfm = if hl != 0.0 { ((c - l) - (h - c)) / hl } else { 0.0 };
					let mfv = mfm * v;
					sum_ad += mfv;
					short_ema = alpha_short * sum_ad + (1.0 - alpha_short) * short_ema;
					long_ema = alpha_long * sum_ad + (1.0 - alpha_long) * long_ema;
					dst[i] = short_ema - long_ema;
				}
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
				// For now, AVX2/AVX512 fall back to scalar
				let alpha_short = 2.0 / (short as f64 + 1.0);
				let alpha_long = 2.0 / (long as f64 + 1.0);
				
				let mut sum_ad = 0.0;
				let h = high[0];
				let l = low[0];
				let c = close[0];
				let v = volume[0];
				let hl = h - l;
				let mfm = if hl != 0.0 { ((c - l) - (h - c)) / hl } else { 0.0 };
				let mfv = mfm * v;
				sum_ad += mfv;
				let mut short_ema = sum_ad;
				let mut long_ema = sum_ad;
				dst[0] = short_ema - long_ema;
				
				for i in 1..len {
					let h = high[i];
					let l = low[i];
					let c = close[i];
					let v = volume[i];
					let hl = h - l;
					let mfm = if hl != 0.0 { ((c - l) - (h - c)) / hl } else { 0.0 };
					let mfv = mfm * v;
					sum_ad += mfv;
					short_ema = alpha_short * sum_ad + (1.0 - alpha_short) * short_ema;
					long_ema = alpha_long * sum_ad + (1.0 - alpha_long) * long_ema;
					dst[i] = short_ema - long_ema;
				}
			}
			#[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
			Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
				// AVX not supported, use scalar
				let alpha_short = 2.0 / (short as f64 + 1.0);
				let alpha_long = 2.0 / (long as f64 + 1.0);
				
				let mut sum_ad = 0.0;
				let h = high[0];
				let l = low[0];
				let c = close[0];
				let v = volume[0];
				let hl = h - l;
				let mfm = if hl != 0.0 { ((c - l) - (h - c)) / hl } else { 0.0 };
				let mfv = mfm * v;
				sum_ad += mfv;
				let mut short_ema = sum_ad;
				let mut long_ema = sum_ad;
				dst[0] = short_ema - long_ema;
				
				for i in 1..len {
					let h = high[i];
					let l = low[i];
					let c = close[i];
					let v = volume[i];
					let hl = h - l;
					let mfm = if hl != 0.0 { ((c - l) - (h - c)) / hl } else { 0.0 };
					let mfv = mfm * v;
					sum_ad += mfv;
					short_ema = alpha_short * sum_ad + (1.0 - alpha_short) * short_ema;
					long_ema = alpha_long * sum_ad + (1.0 - alpha_long) * long_ema;
					dst[i] = short_ema - long_ema;
				}
			}
			_ => unreachable!(),
		}
	}
	
	// ADOSC has no warmup period - all values are valid
	Ok(())
}

#[cfg(feature = "python")]
#[pyfunction(name = "adosc")]
#[pyo3(signature = (high, low, close, volume, short_period, long_period, kernel=None))]
pub fn adosc_py<'py>(
	py: Python<'py>,
	high: numpy::PyReadonlyArray1<'py, f64>,
	low: numpy::PyReadonlyArray1<'py, f64>,
	close: numpy::PyReadonlyArray1<'py, f64>,
	volume: numpy::PyReadonlyArray1<'py, f64>,
	short_period: usize,
	long_period: usize,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
	use numpy::{IntoPyArray, PyArrayMethods};

	let high_slice = high.as_slice()?;
	let low_slice = low.as_slice()?;
	let close_slice = close.as_slice()?;
	let volume_slice = volume.as_slice()?;

	// Validate all slices have the same length
	let len = close_slice.len();
	if high_slice.len() != len || low_slice.len() != len || volume_slice.len() != len {
		return Err(PyValueError::new_err(format!(
			"All input arrays must have the same length. Got high={}, low={}, close={}, volume={}",
			high_slice.len(),
			low_slice.len(),
			close_slice.len(),
			volume_slice.len()
		)));
	}

	// Use kernel validation for safety
	let kern = validate_kernel(kernel, false)?;

	// Build input struct
	let params = AdoscParams {
		short_period: Some(short_period),
		long_period: Some(long_period),
	};
	let adosc_in = AdoscInput::from_slices(high_slice, low_slice, close_slice, volume_slice, params);

	// GOOD: Get Vec<f64> from Rust function
	let result_vec: Vec<f64> = py
		.allow_threads(|| adosc_with_kernel(&adosc_in, kern).map(|o| o.values))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	// GOOD: Zero-copy transfer to NumPy
	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "AdoscStream")]
pub struct AdoscStreamPy {
	stream: AdoscStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl AdoscStreamPy {
	#[new]
	fn new(short_period: usize, long_period: usize) -> PyResult<Self> {
		let params = AdoscParams {
			short_period: Some(short_period),
			long_period: Some(long_period),
		};
		let stream = AdoscStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(AdoscStreamPy { stream })
	}

	/// Updates the stream with new values and returns the calculated ADOSC value.
	fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> f64 {
		self.stream.update(high, low, close, volume)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "adosc_batch")]
#[pyo3(signature = (high, low, close, volume, short_period_range, long_period_range, kernel=None))]
pub fn adosc_batch_py<'py>(
	py: Python<'py>,
	high: numpy::PyReadonlyArray1<'py, f64>,
	low: numpy::PyReadonlyArray1<'py, f64>,
	close: numpy::PyReadonlyArray1<'py, f64>,
	volume: numpy::PyReadonlyArray1<'py, f64>,
	short_period_range: (usize, usize, usize),
	long_period_range: (usize, usize, usize),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;

	let high_slice = high.as_slice()?;
	let low_slice = low.as_slice()?;
	let close_slice = close.as_slice()?;
	let volume_slice = volume.as_slice()?;

	// Validate all slices have the same length
	let len = close_slice.len();
	if high_slice.len() != len || low_slice.len() != len || volume_slice.len() != len {
		return Err(PyValueError::new_err(format!(
			"All input arrays must have the same length. Got high={}, low={}, close={}, volume={}",
			high_slice.len(),
			low_slice.len(),
			close_slice.len(),
			volume_slice.len()
		)));
	}

	let sweep = AdoscBatchRange {
		short_period: short_period_range,
		long_period: long_period_range,
	};

	// Expand grid to know rows*cols
	let combos = expand_grid(&sweep);
	let rows = combos.len();
	let cols = len;

	// Pre-allocate uninitialized NumPy array (1-D, will reshape later)
	// NOTE: PyArray1::new() creates uninitialized memory, not zero-initialized
	let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let slice_out = unsafe { out_arr.as_slice_mut()? };

	// Use kernel validation for safety
	let kern = validate_kernel(kernel, true)?;

	// Heavy work without the GIL
	let combos = py
		.allow_threads(|| -> Result<Vec<AdoscParams>, AdoscError> {
			// Resolve Kernel::Auto to a specific kernel
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

			// Use the optimized batch function that writes directly to output
			adosc_batch_inner_into(
				high_slice,
				low_slice,
				close_slice,
				volume_slice,
				&sweep,
				simd,
				true,  // parallel
				slice_out,
			)
		})
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	// Build dict with the GIL
	let dict = PyDict::new(py);
	dict.set_item("values", out_arr.reshape((rows, cols))?)?;
	dict.set_item(
		"short_periods",
		combos
			.iter()
			.map(|p| p.short_period.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;
	dict.set_item(
		"long_periods",
		combos
			.iter()
			.map(|p| p.long_period.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;

	Ok(dict)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn adosc_js(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	short_period: usize,
	long_period: usize,
) -> Result<Vec<f64>, JsValue> {
	let params = AdoscParams {
		short_period: Some(short_period),
		long_period: Some(long_period),
	};
	let input = AdoscInput::from_slices(high, low, close, volume, params);

	// Single allocation with zero-copy pattern
	let mut output = vec![0.0; high.len()];
	adosc_into_slice(&mut output, &input, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn adosc_batch_js(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	short_period_start: usize,
	short_period_end: usize,
	short_period_step: usize,
	long_period_start: usize,
	long_period_end: usize,
	long_period_step: usize,
) -> Result<Vec<f64>, JsValue> {
	let sweep = AdoscBatchRange {
		short_period: (short_period_start, short_period_end, short_period_step),
		long_period: (long_period_start, long_period_end, long_period_step),
	};

	// Use the existing batch function with parallel=false for WASM
	adosc_batch_inner(high, low, close, volume, &sweep, Kernel::Scalar, false)
		.map(|output| output.values)
		.map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn adosc_batch_metadata_js(
	short_period_start: usize,
	short_period_end: usize,
	short_period_step: usize,
	long_period_start: usize,
	long_period_end: usize,
	long_period_step: usize,
) -> Result<Vec<f64>, JsValue> {
	let sweep = AdoscBatchRange {
		short_period: (short_period_start, short_period_end, short_period_step),
		long_period: (long_period_start, long_period_end, long_period_step),
	};

	let combos = expand_grid(&sweep);
	let mut metadata = Vec::with_capacity(combos.len() * 2);

	for combo in combos {
		metadata.push(combo.short_period.unwrap() as f64);
		metadata.push(combo.long_period.unwrap() as f64);
	}

	Ok(metadata)
}

// New ergonomic WASM API
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct AdoscBatchConfig {
	pub short_period_range: (usize, usize, usize),
	pub long_period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct AdoscBatchJsOutput {
	pub values: Vec<f64>,
	pub combos: Vec<AdoscParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = adosc_batch)]
pub fn adosc_batch_unified_js(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	config: JsValue,
) -> Result<JsValue, JsValue> {
	// 1. Deserialize the configuration object from JavaScript
	let config: AdoscBatchConfig =
		serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

	let sweep = AdoscBatchRange {
		short_period: config.short_period_range,
		long_period: config.long_period_range,
	};

	// 2. Run the existing core logic
	let output = adosc_batch_inner(high, low, close, volume, &sweep, Kernel::Scalar, false)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;

	// 3. Create the structured output
	let js_output = AdoscBatchJsOutput {
		values: output.values,
		combos: output.combos,
		rows: output.rows,
		cols: output.cols,
	};

	// 4. Serialize the output struct into a JavaScript object
	serde_wasm_bindgen::to_value(&js_output).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn adosc_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn adosc_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe {
			let _ = Vec::from_raw_parts(ptr, len, len);
		}
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn adosc_into(
	high_ptr: *const f64,
	low_ptr: *const f64,
	close_ptr: *const f64,
	volume_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	short_period: usize,
	long_period: usize,
) -> Result<(), JsValue> {
	if high_ptr.is_null() || low_ptr.is_null() || close_ptr.is_null() || volume_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}
	
	unsafe {
		let high = std::slice::from_raw_parts(high_ptr, len);
		let low = std::slice::from_raw_parts(low_ptr, len);
		let close = std::slice::from_raw_parts(close_ptr, len);
		let volume = std::slice::from_raw_parts(volume_ptr, len);
		
		let params = AdoscParams {
			short_period: Some(short_period),
			long_period: Some(long_period),
		};
		let input = AdoscInput::from_slices(high, low, close, volume, params);
		
		// Check for aliasing - if output pointer matches ANY input pointer
		if out_ptr as *const f64 == high_ptr || out_ptr as *const f64 == low_ptr || out_ptr as *const f64 == close_ptr || out_ptr as *const f64 == volume_ptr {
			// Handle aliasing by using temporary buffer
			let mut temp = vec![0.0; len];
			adosc_into_slice(&mut temp, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			out.copy_from_slice(&temp);
		} else {
			// No aliasing, compute directly into output
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			adosc_into_slice(out, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		
		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn adosc_batch_into(
	high_ptr: *const f64,
	low_ptr: *const f64,
	close_ptr: *const f64,
	volume_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	short_period_start: usize,
	short_period_end: usize,
	short_period_step: usize,
	long_period_start: usize,
	long_period_end: usize,
	long_period_step: usize,
) -> Result<usize, JsValue> {
	if high_ptr.is_null() || low_ptr.is_null() || close_ptr.is_null() || volume_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("null pointer passed to adosc_batch_into"));
	}

	unsafe {
		let high = std::slice::from_raw_parts(high_ptr, len);
		let low = std::slice::from_raw_parts(low_ptr, len);
		let close = std::slice::from_raw_parts(close_ptr, len);
		let volume = std::slice::from_raw_parts(volume_ptr, len);

		let sweep = AdoscBatchRange {
			short_period: (short_period_start, short_period_end, short_period_step),
			long_period: (long_period_start, long_period_end, long_period_step),
		};

		let combos = expand_grid(&sweep);
		let rows = combos.len();
		let cols = len;

		let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);

		adosc_batch_inner_into(high, low, close, volume, &sweep, Kernel::Auto, false, out)
			.map_err(|e| JsValue::from_str(&e.to_string()))?;

		Ok(rows)
	}
}
