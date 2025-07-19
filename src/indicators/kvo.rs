//! # Klinger Volume Oscillator (KVO)
//!
//! The Klinger Volume Oscillator (KVO) is designed to capture long-term
//! money flow trends, while remaining sensitive enough to short-term
//! fluctuations. It uses high, low, close prices and volume to measure
//! volume force (VF), then applies two separate EMAs (short and long)
//! to VF and calculates the difference.
//!
//! ## Parameters
//! - **short_period**: The short EMA period. Defaults to 2.
//! - **long_period**: The long EMA period. Defaults to 5.
//!
//! ## Errors
//! - **AllValuesNaN**: kvo: All input data values are `NaN`.
//! - **InvalidPeriod**: kvo: `short_period` < 1 or `long_period` < `short_period`.
//! - **NotEnoughValidData**: kvo: Not enough valid data points for calculation.
//! - **EmptyData**: kvo: Input data slice is empty or not found.
//!
//! ## Returns
//! - **`Ok(KvoOutput)`** on success, containing a `Vec<f64>` matching input length.
//! - **`Err(KvoError)`** otherwise.

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
pub enum KvoData<'a> {
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
pub struct KvoOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct KvoParams {
	pub short_period: Option<usize>,
	pub long_period: Option<usize>,
}

impl Default for KvoParams {
	fn default() -> Self {
		Self {
			short_period: Some(2),
			long_period: Some(5),
		}
	}
}

#[derive(Debug, Clone)]
pub struct KvoInput<'a> {
	pub data: KvoData<'a>,
	pub params: KvoParams,
}

impl<'a> KvoInput<'a> {
	#[inline]
	pub fn from_candles(candles: &'a Candles, params: KvoParams) -> Self {
		Self {
			data: KvoData::Candles { candles },
			params,
		}
	}

	#[inline]
	pub fn from_slices(
		high: &'a [f64],
		low: &'a [f64],
		close: &'a [f64],
		volume: &'a [f64],
		params: KvoParams,
	) -> Self {
		Self {
			data: KvoData::Slices {
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
		Self::from_candles(candles, KvoParams::default())
	}

	#[inline]
	pub fn get_short_period(&self) -> usize {
		self.params.short_period.unwrap_or(2)
	}

	#[inline]
	pub fn get_long_period(&self) -> usize {
		self.params.long_period.unwrap_or(5)
	}
}

#[derive(Debug, Clone, Copy)]
pub struct KvoBuilder {
	short_period: Option<usize>,
	long_period: Option<usize>,
	kernel: Kernel,
}

impl Default for KvoBuilder {
	fn default() -> Self {
		Self {
			short_period: None,
			long_period: None,
			kernel: Kernel::Auto,
		}
	}
}

impl KvoBuilder {
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
	pub fn apply(self, c: &Candles) -> Result<KvoOutput, KvoError> {
		let params = KvoParams {
			short_period: self.short_period,
			long_period: self.long_period,
		};
		let input = KvoInput::from_candles(c, params);
		kvo_with_kernel(&input, self.kernel)
	}

	#[inline(always)]
	pub fn apply_slices(self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Result<KvoOutput, KvoError> {
		let params = KvoParams {
			short_period: self.short_period,
			long_period: self.long_period,
		};
		let input = KvoInput::from_slices(high, low, close, volume, params);
		kvo_with_kernel(&input, self.kernel)
	}

	#[inline(always)]
	pub fn into_stream(self) -> Result<KvoStream, KvoError> {
		let params = KvoParams {
			short_period: self.short_period,
			long_period: self.long_period,
		};
		KvoStream::try_new(params)
	}
}

#[derive(Debug, Error)]
pub enum KvoError {
	#[error("kvo: Empty data provided.")]
	EmptyData,
	#[error("kvo: Invalid period settings: short={short}, long={long}")]
	InvalidPeriod { short: usize, long: usize },
	#[error("kvo: Not enough valid data: found {valid} valid points after the first valid index.")]
	NotEnoughValidData { valid: usize },
	#[error("kvo: All values are NaN.")]
	AllValuesNaN,
}

#[inline]
pub fn kvo(input: &KvoInput) -> Result<KvoOutput, KvoError> {
	kvo_with_kernel(input, Kernel::Auto)
}

pub fn kvo_with_kernel(input: &KvoInput, kernel: Kernel) -> Result<KvoOutput, KvoError> {
	let (high, low, close, volume): (&[f64], &[f64], &[f64], &[f64]) = match &input.data {
		KvoData::Candles { candles } => (
			source_type(candles, "high"),
			source_type(candles, "low"),
			source_type(candles, "close"),
			source_type(candles, "volume"),
		),
		KvoData::Slices {
			high,
			low,
			close,
			volume,
		} => (*high, *low, *close, *volume),
	};

	if high.is_empty() || low.is_empty() || close.is_empty() || volume.is_empty() {
		return Err(KvoError::EmptyData);
	}

	let short_period = input.get_short_period();
	let long_period = input.get_long_period();
	if short_period < 1 || long_period < short_period {
		return Err(KvoError::InvalidPeriod {
			short: short_period,
			long: long_period,
		});
	}

	let first_valid_idx = high
		.iter()
		.zip(low.iter())
		.zip(close.iter())
		.zip(volume.iter())
		.position(|(((h, l), c), v)| !h.is_nan() && !l.is_nan() && !c.is_nan() && !v.is_nan());
	let first_valid_idx = match first_valid_idx {
		Some(idx) => idx,
		None => return Err(KvoError::AllValuesNaN),
	};

	if (high.len() - first_valid_idx) < 2 {
		return Err(KvoError::NotEnoughValidData {
			valid: high.len() - first_valid_idx,
		});
	}

	let mut out = vec![f64::NAN; high.len()];
	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => kvo_scalar(
				high,
				low,
				close,
				volume,
				short_period,
				long_period,
				first_valid_idx,
				&mut out,
			),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => kvo_avx2(
				high,
				low,
				close,
				volume,
				short_period,
				long_period,
				first_valid_idx,
				&mut out,
			),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => kvo_avx512(
				high,
				low,
				close,
				volume,
				short_period,
				long_period,
				first_valid_idx,
				&mut out,
			),
			_ => unreachable!(),
		}
	}
	Ok(KvoOutput { values: out })
}

#[inline]
pub unsafe fn kvo_scalar(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	short_period: usize,
	long_period: usize,
	first_valid_idx: usize,
	out: &mut [f64],
) {
	let short_per = 2.0 / (short_period as f64 + 1.0);
	let long_per = 2.0 / (long_period as f64 + 1.0);

	let mut trend = -1;
	let mut cm = 0.0;
	let mut prev_hlc = high[first_valid_idx] + low[first_valid_idx] + close[first_valid_idx];
	let mut short_ema = 0.0;
	let mut long_ema = 0.0;

	for i in (first_valid_idx + 1)..high.len() {
		let hlc = high[i] + low[i] + close[i];
		let dm = high[i] - low[i];

		if hlc > prev_hlc && trend != 1 {
			trend = 1;
			cm = high[i - 1] - low[i - 1];
		} else if hlc < prev_hlc && trend != 0 {
			trend = 0;
			cm = high[i - 1] - low[i - 1];
		}
		cm += dm;
		let vf = volume[i] * (dm / cm * 2.0 - 1.0).abs() * 100.0 * if trend == 1 { 1.0 } else { -1.0 };

		if i == first_valid_idx + 1 {
			short_ema = vf;
			long_ema = vf;
		} else {
			short_ema = (vf - short_ema) * short_per + short_ema;
			long_ema = (vf - long_ema) * long_per + long_ema;
		}
		out[i] = short_ema - long_ema;
		prev_hlc = hlc;
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn kvo_avx2(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	short_period: usize,
	long_period: usize,
	first_valid_idx: usize,
	out: &mut [f64],
) {
	kvo_scalar(
		high,
		low,
		close,
		volume,
		short_period,
		long_period,
		first_valid_idx,
		out,
	)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn kvo_avx512(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	short_period: usize,
	long_period: usize,
	first_valid_idx: usize,
	out: &mut [f64],
) {
	if short_period <= 32 && long_period <= 32 {
		kvo_avx512_short(
			high,
			low,
			close,
			volume,
			short_period,
			long_period,
			first_valid_idx,
			out,
		)
	} else {
		kvo_avx512_long(
			high,
			low,
			close,
			volume,
			short_period,
			long_period,
			first_valid_idx,
			out,
		)
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn kvo_avx512_short(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	short_period: usize,
	long_period: usize,
	first_valid_idx: usize,
	out: &mut [f64],
) {
	kvo_scalar(
		high,
		low,
		close,
		volume,
		short_period,
		long_period,
		first_valid_idx,
		out,
	)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn kvo_avx512_long(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	short_period: usize,
	long_period: usize,
	first_valid_idx: usize,
	out: &mut [f64],
) {
	kvo_scalar(
		high,
		low,
		close,
		volume,
		short_period,
		long_period,
		first_valid_idx,
		out,
	)
}

#[derive(Clone, Debug)]
pub struct KvoBatchRange {
	pub short_period: (usize, usize, usize),
	pub long_period: (usize, usize, usize),
}

impl Default for KvoBatchRange {
	fn default() -> Self {
		Self {
			short_period: (2, 10, 1),
			long_period: (5, 20, 1),
		}
	}
}

#[derive(Clone, Debug, Default)]
pub struct KvoBatchBuilder {
	range: KvoBatchRange,
	kernel: Kernel,
}

impl KvoBatchBuilder {
	pub fn new() -> Self {
		Self::default()
	}
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}

	#[inline]
	pub fn short_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.short_period = (start, end, step);
		self
	}
	#[inline]
	pub fn short_static(mut self, v: usize) -> Self {
		self.range.short_period = (v, v, 0);
		self
	}
	#[inline]
	pub fn long_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.long_period = (start, end, step);
		self
	}
	#[inline]
	pub fn long_static(mut self, v: usize) -> Self {
		self.range.long_period = (v, v, 0);
		self
	}

	pub fn apply_slices(
		self,
		high: &[f64],
		low: &[f64],
		close: &[f64],
		volume: &[f64],
	) -> Result<KvoBatchOutput, KvoError> {
		kvo_batch_with_kernel(high, low, close, volume, &self.range, self.kernel)
	}
	pub fn with_default_slices(
		high: &[f64],
		low: &[f64],
		close: &[f64],
		volume: &[f64],
		k: Kernel,
	) -> Result<KvoBatchOutput, KvoError> {
		KvoBatchBuilder::new().kernel(k).apply_slices(high, low, close, volume)
	}

	pub fn apply_candles(self, c: &Candles) -> Result<KvoBatchOutput, KvoError> {
		let high = source_type(c, "high");
		let low = source_type(c, "low");
		let close = source_type(c, "close");
		let volume = source_type(c, "volume");
		self.apply_slices(high, low, close, volume)
	}

	pub fn with_default_candles(c: &Candles, k: Kernel) -> Result<KvoBatchOutput, KvoError> {
		KvoBatchBuilder::new().kernel(k).apply_candles(c)
	}
}

pub fn kvo_batch_with_kernel(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	sweep: &KvoBatchRange,
	k: Kernel,
) -> Result<KvoBatchOutput, KvoError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(KvoError::InvalidPeriod { short: 0, long: 0 }),
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	kvo_batch_par_slice(high, low, close, volume, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct KvoBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<KvoParams>,
	pub rows: usize,
	pub cols: usize,
}
impl KvoBatchOutput {
	pub fn row_for_params(&self, p: &KvoParams) -> Option<usize> {
		self.combos.iter().position(|c| {
			c.short_period.unwrap_or(2) == p.short_period.unwrap_or(2)
				&& c.long_period.unwrap_or(5) == p.long_period.unwrap_or(5)
		})
	}
	pub fn values_for(&self, p: &KvoParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &KvoBatchRange) -> Vec<KvoParams> {
	fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let shorts = axis(r.short_period);
	let longs = axis(r.long_period);
	let mut out = Vec::with_capacity(shorts.len() * longs.len());
	for &s in &shorts {
		for &l in &longs {
			if s >= 1 && l >= s {
				out.push(KvoParams {
					short_period: Some(s),
					long_period: Some(l),
				});
			}
		}
	}
	out
}

#[inline(always)]
pub fn kvo_batch_slice(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	sweep: &KvoBatchRange,
	kern: Kernel,
) -> Result<KvoBatchOutput, KvoError> {
	kvo_batch_inner(high, low, close, volume, sweep, kern, false)
}
#[inline(always)]
pub fn kvo_batch_par_slice(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	sweep: &KvoBatchRange,
	kern: Kernel,
) -> Result<KvoBatchOutput, KvoError> {
	kvo_batch_inner(high, low, close, volume, sweep, kern, true)
}

#[inline(always)]
fn kvo_batch_inner(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	sweep: &KvoBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<KvoBatchOutput, KvoError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(KvoError::InvalidPeriod { short: 0, long: 0 });
	}
	let first = high
		.iter()
		.zip(low)
		.zip(close)
		.zip(volume)
		.position(|(((h, l), c), v)| !h.is_nan() && !l.is_nan() && !c.is_nan() && !v.is_nan())
		.ok_or(KvoError::AllValuesNaN)?;
	let max_short = combos.iter().map(|c| c.short_period.unwrap()).max().unwrap();
	let max_long = combos.iter().map(|c| c.long_period.unwrap()).max().unwrap();
	if high.len() - first < 2 {
		return Err(KvoError::NotEnoughValidData {
			valid: high.len() - first,
		});
	}
	let rows = combos.len();
	let cols = high.len();
	let mut values = vec![f64::NAN; rows * cols];
	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let short = combos[row].short_period.unwrap();
		let long = combos[row].long_period.unwrap();
		match kern {
			Kernel::Scalar => kvo_row_scalar(high, low, close, volume, first, short, long, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => kvo_row_avx2(high, low, close, volume, first, short, long, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => kvo_row_avx512(high, low, close, volume, first, short, long, out_row),
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
	Ok(KvoBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
unsafe fn kvo_row_scalar(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	first: usize,
	short_period: usize,
	long_period: usize,
	out: &mut [f64],
) {
	kvo_scalar(high, low, close, volume, short_period, long_period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn kvo_row_avx2(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	first: usize,
	short_period: usize,
	long_period: usize,
	out: &mut [f64],
) {
	kvo_scalar(high, low, close, volume, short_period, long_period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn kvo_row_avx512(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	first: usize,
	short_period: usize,
	long_period: usize,
	out: &mut [f64],
) {
	if short_period <= 32 && long_period <= 32 {
		kvo_row_avx512_short(high, low, close, volume, first, short_period, long_period, out)
	} else {
		kvo_row_avx512_long(high, low, close, volume, first, short_period, long_period, out)
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn kvo_row_avx512_short(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	first: usize,
	short_period: usize,
	long_period: usize,
	out: &mut [f64],
) {
	kvo_scalar(high, low, close, volume, short_period, long_period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn kvo_row_avx512_long(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	first: usize,
	short_period: usize,
	long_period: usize,
	out: &mut [f64],
) {
	kvo_scalar(high, low, close, volume, short_period, long_period, first, out)
}

#[derive(Debug, Clone)]
pub struct KvoStream {
	short_period: usize,
	long_period: usize,
	short_alpha: f64,
	long_alpha: f64,
	prev_hlc: f64,
	prev_dm: f64,
	trend: i32,
	cm: f64,
	short_ema: f64,
	long_ema: f64,
	filled: bool,
	first: bool,
}

impl KvoStream {
	pub fn try_new(params: KvoParams) -> Result<Self, KvoError> {
		let short_period = params.short_period.unwrap_or(2);
		let long_period = params.long_period.unwrap_or(5);
		if short_period < 1 || long_period < short_period {
			return Err(KvoError::InvalidPeriod {
				short: short_period,
				long: long_period,
			});
		}
		Ok(Self {
			short_period,
			long_period,
			short_alpha: 2.0 / (short_period as f64 + 1.0),
			long_alpha: 2.0 / (long_period as f64 + 1.0),
			prev_hlc: 0.0,
			prev_dm: 0.0,
			trend: -1,
			cm: 0.0,
			short_ema: 0.0,
			long_ema: 0.0,
			filled: false,
			first: true,
		})
	}

	#[inline(always)]
	pub fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> Option<f64> {
		if self.first {
			self.prev_hlc = high + low + close;
			self.prev_dm = high - low;
			self.first = false;
			return None;
		}
		let hlc = high + low + close;
		let dm = high - low;
		if hlc > self.prev_hlc && self.trend != 1 {
			self.trend = 1;
			self.cm = self.prev_dm;
		} else if hlc < self.prev_hlc && self.trend != 0 {
			self.trend = 0;
			self.cm = self.prev_dm;
		}
		self.cm += dm;
		let vf = volume * (dm / self.cm * 2.0 - 1.0).abs() * 100.0 * if self.trend == 1 { 1.0 } else { -1.0 };
		if !self.filled {
			self.short_ema = vf;
			self.long_ema = vf;
			self.filled = true;
		} else {
			self.short_ema = (vf - self.short_ema) * self.short_alpha + self.short_ema;
			self.long_ema = (vf - self.long_ema) * self.long_alpha + self.long_ema;
		}
		self.prev_hlc = hlc;
		self.prev_dm = dm;
		Some(self.short_ema - self.long_ema)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_kvo_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let default_params = KvoParams {
			short_period: None,
			long_period: None,
		};
		let input = KvoInput::from_candles(&candles, default_params);
		let output = kvo_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());

		Ok(())
	}

	fn check_kvo_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = KvoInput::from_candles(&candles, KvoParams::default());
		let result = kvo_with_kernel(&input, kernel)?;
		let expected_last_five = [
			-246.42698280402647,
			530.8651474164992,
			237.2148311016648,
			608.8044103976362,
			-6339.615516805162,
		];
		let start = result.values.len().saturating_sub(5);
		for (i, &val) in result.values[start..].iter().enumerate() {
			let diff = (val - expected_last_five[i]).abs();
			assert!(
				diff < 1e-1,
				"[{}] KVO {:?} mismatch at idx {}: got {}, expected {}",
				test_name,
				kernel,
				i,
				val,
				expected_last_five[i]
			);
		}
		Ok(())
	}

	fn check_kvo_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = KvoInput::with_default_candles(&candles);
		match input.data {
			KvoData::Candles { .. } => {}
			_ => panic!("Expected KvoData::Candles"),
		}
		let output = kvo_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_kvo_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = KvoParams {
			short_period: Some(0),
			long_period: Some(5),
		};
		let input = KvoInput::from_candles(&candles, params);
		let res = kvo_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] KVO should fail with zero short period", test_name);
		Ok(())
	}

	fn check_kvo_period_invalid(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = KvoParams {
			short_period: Some(5),
			long_period: Some(2),
		};
		let input = KvoInput::from_candles(&candles, params);
		let res = kvo_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] KVO should fail with long_period < short_period",
			test_name
		);
		Ok(())
	}

	fn check_kvo_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let mut candles = read_candles_from_csv(file_path)?;
		candles.high.truncate(1);
		candles.low.truncate(1);
		candles.close.truncate(1);
		candles.volume.truncate(1);
		let input = KvoInput::from_candles(&candles, KvoParams::default());
		let res = kvo_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] KVO should fail with insufficient data", test_name);
		Ok(())
	}

	fn check_kvo_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let first_params = KvoParams {
			short_period: Some(2),
			long_period: Some(5),
		};
		let first_input = KvoInput::from_candles(&candles, first_params);
		let first_result = kvo_with_kernel(&first_input, kernel)?;
		let second_params = KvoParams {
			short_period: Some(2),
			long_period: Some(5),
		};
		let second_input = KvoInput::from_slices(
			&candles.high,
			&candles.low,
			&candles.close,
			&first_result.values,
			second_params,
		);
		let _ = kvo_with_kernel(&second_input, kernel);
		Ok(())
	}

	fn check_kvo_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = KvoInput::from_candles(&candles, KvoParams::default());
		let res = kvo_with_kernel(&input, kernel)?;
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

	fn check_kvo_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let short = 2;
		let long = 5;

		let input = KvoInput::from_candles(
			&candles,
			KvoParams {
				short_period: Some(short),
				long_period: Some(long),
			},
		);
		let batch_output = kvo_with_kernel(&input, kernel)?.values;

		let mut stream = KvoStream::try_new(KvoParams {
			short_period: Some(short),
			long_period: Some(long),
		})?;
		let mut stream_values = Vec::with_capacity(candles.close.len());
		for ((&h, &l), (&c, &v)) in candles
			.high
			.iter()
			.zip(&candles.low)
			.zip(candles.close.iter().zip(&candles.volume))
		{
			match stream.update(h, l, c, v) {
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
				"[{}] KVO streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
				test_name,
				i,
				b,
				s,
				diff
			);
		}
		Ok(())
	}

	macro_rules! generate_all_kvo_tests {
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

	generate_all_kvo_tests!(
		check_kvo_partial_params,
		check_kvo_accuracy,
		check_kvo_default_candles,
		check_kvo_zero_period,
		check_kvo_period_invalid,
		check_kvo_very_small_dataset,
		check_kvo_reinput,
		check_kvo_nan_handling,
		check_kvo_streaming
	);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let output = KvoBatchBuilder::new().kernel(kernel).apply_candles(&c)?;
		let def = KvoParams::default();
		let row = output.values_for(&def).expect("default row missing");
		assert_eq!(row.len(), c.close.len());
		let expected = [
			-246.42698280402647,
			530.8651474164992,
			237.2148311016648,
			608.8044103976362,
			-6339.615516805162,
		];
		let start = row.len() - 5;
		for (i, &v) in row[start..].iter().enumerate() {
			assert!(
				(v - expected[i]).abs() < 1e-1,
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
	gen_batch_tests!(check_batch_default_row);
}
