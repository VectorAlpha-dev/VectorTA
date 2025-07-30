//! # Decycler Oscillator (DEC_OSC)
//!
//! An oscillator that applies two sequential high-pass filters (2-pole) to remove
//! cyclical components from the data. The residual is then scaled by `k` and expressed
//! as a percentage of the original input series.
//!
//! ## Parameters
//! - **hp_period**: The period used for the primary high-pass filter. Defaults to 125.
//! - **k**: Multiplier for the final oscillator values. Defaults to 1.0.
//!
//! ## Errors
//! - **AllValuesNaN**: dec_osc: All input data values are `NaN`.
//! - **InvalidPeriod**: dec_osc: `hp_period` < 2 or exceeds the data length.
//! - **NotEnoughValidData**: dec_osc: Fewer than 2 valid (non-`NaN`) data points remain.
//! - **InvalidK**: dec_osc: `k` is `NaN` or non-positive.
//!
//! ## Returns
//! - **`Ok(DecOscOutput)`** on success, containing a `Vec<f64>` matching the input length.
//! - **`Err(DecOscError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::f64::consts::PI;
use thiserror::Error;

impl<'a> AsRef<[f64]> for DecOscInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			DecOscData::Slice(slice) => slice,
			DecOscData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

#[derive(Debug, Clone)]
pub enum DecOscData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct DecOscOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct DecOscParams {
	pub hp_period: Option<usize>,
	pub k: Option<f64>,
}

impl Default for DecOscParams {
	fn default() -> Self {
		Self {
			hp_period: Some(125),
			k: Some(1.0),
		}
	}
}

#[derive(Debug, Clone)]
pub struct DecOscInput<'a> {
	pub data: DecOscData<'a>,
	pub params: DecOscParams,
}

impl<'a> DecOscInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: DecOscParams) -> Self {
		Self {
			data: DecOscData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slice(sl: &'a [f64], p: DecOscParams) -> Self {
		Self {
			data: DecOscData::Slice(sl),
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", DecOscParams::default())
	}
	#[inline]
	pub fn get_hp_period(&self) -> usize {
		self.params.hp_period.unwrap_or(125)
	}
	#[inline]
	pub fn get_k(&self) -> f64 {
		self.params.k.unwrap_or(1.0)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct DecOscBuilder {
	hp_period: Option<usize>,
	k: Option<f64>,
	kernel: Kernel,
}

impl Default for DecOscBuilder {
	fn default() -> Self {
		Self {
			hp_period: None,
			k: None,
			kernel: Kernel::Auto,
		}
	}
}

impl DecOscBuilder {
	#[inline(always)]
	pub fn new() -> Self {
		Self::default()
	}
	#[inline(always)]
	pub fn hp_period(mut self, n: usize) -> Self {
		self.hp_period = Some(n);
		self
	}
	#[inline(always)]
	pub fn k(mut self, v: f64) -> Self {
		self.k = Some(v);
		self
	}
	#[inline(always)]
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}

	#[inline(always)]
	pub fn apply(self, c: &Candles) -> Result<DecOscOutput, DecOscError> {
		let p = DecOscParams {
			hp_period: self.hp_period,
			k: self.k,
		};
		let i = DecOscInput::from_candles(c, "close", p);
		dec_osc_with_kernel(&i, self.kernel)
	}

	#[inline(always)]
	pub fn apply_slice(self, d: &[f64]) -> Result<DecOscOutput, DecOscError> {
		let p = DecOscParams {
			hp_period: self.hp_period,
			k: self.k,
		};
		let i = DecOscInput::from_slice(d, p);
		dec_osc_with_kernel(&i, self.kernel)
	}

	#[inline(always)]
	pub fn into_stream(self) -> Result<DecOscStream, DecOscError> {
		let p = DecOscParams {
			hp_period: self.hp_period,
			k: self.k,
		};
		DecOscStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum DecOscError {
	#[error("dec_osc: All values are NaN.")]
	AllValuesNaN,

	#[error("dec_osc: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },

	#[error("dec_osc: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },

	#[error("dec_osc: Invalid K: k = {k}")]
	InvalidK { k: f64 },
}

#[inline]
pub fn dec_osc(input: &DecOscInput) -> Result<DecOscOutput, DecOscError> {
	dec_osc_with_kernel(input, Kernel::Auto)
}

pub fn dec_osc_with_kernel(input: &DecOscInput, kernel: Kernel) -> Result<DecOscOutput, DecOscError> {
	let data: &[f64] = match &input.data {
		DecOscData::Candles { candles, source } => source_type(candles, source),
		DecOscData::Slice(sl) => sl,
	};

	let len = data.len();
	let period = input.get_hp_period();
	let k_val = input.get_k();

	let first = data.iter().position(|x| !x.is_nan()).ok_or(DecOscError::AllValuesNaN)?;

	if period < 2 || period > len {
		return Err(DecOscError::InvalidPeriod { period, data_len: len });
	}
	if (len - first) < 2 {
		return Err(DecOscError::NotEnoughValidData {
			needed: 2,
			valid: len - first,
		});
	}
	if k_val <= 0.0 || k_val.is_nan() {
		return Err(DecOscError::InvalidK { k: k_val });
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	let mut out = vec![f64::NAN; len];

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => dec_osc_scalar(data, period, k_val, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => dec_osc_avx2(data, period, k_val, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => dec_osc_avx512(data, period, k_val, first, &mut out),
			_ => unreachable!(),
		}
	}

	Ok(DecOscOutput { values: out })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn dec_osc_avx512(data: &[f64], period: usize, k_val: f64, first: usize, out: &mut [f64]) {
	if period <= 32 {
		unsafe { dec_osc_avx512_short(data, period, k_val, first, out) }
	} else {
		unsafe { dec_osc_avx512_long(data, period, k_val, first, out) }
	}
}

#[inline]
pub fn dec_osc_scalar(data: &[f64], period: usize, k_val: f64, first: usize, out: &mut [f64]) {
	assert!(out.len() >= data.len(), "`out` must be at least as long as `data`");

	let len = data.len();
	let half_period = (period as f64) * 0.5;

	let angle1 = 2.0 * PI * 0.707 / (period as f64);
	let sin1 = angle1.sin();
	let cos1 = angle1.cos();
	let alpha1 = 1.0 + ((sin1 - 1.0) / cos1);
	let c1 = (1.0 - alpha1 / 2.0) * (1.0 - alpha1 / 2.0);
	let one_minus_alpha1 = 1.0 - alpha1;
	let one_minus_alpha1_sq = one_minus_alpha1 * one_minus_alpha1;

	let angle2 = 2.0 * PI * 0.707 / half_period;
	let sin2 = angle2.sin();
	let cos2 = angle2.cos();
	let alpha2 = 1.0 + ((sin2 - 1.0) / cos2);
	let c2 = (1.0 - alpha2 / 2.0) * (1.0 - alpha2 / 2.0);
	let one_minus_alpha2 = 1.0 - alpha2;
	let one_minus_alpha2_sq = one_minus_alpha2 * one_minus_alpha2;

	let mut hp_prev_2;
	let mut hp_prev_1;
	let mut decosc_prev_2;
	let mut decosc_prev_1;

	{
		let val0 = data[first];
		out[first] = f64::NAN;
		hp_prev_2 = val0;
		hp_prev_1 = val0;
		decosc_prev_2 = 0.0;
		decosc_prev_1 = 0.0;
	}

	if first + 1 < len {
		let val1 = data[first + 1];
		out[first + 1] = f64::NAN;
		hp_prev_2 = hp_prev_1;
		hp_prev_1 = val1;

		let dec = val1 - hp_prev_1;
		decosc_prev_2 = decosc_prev_1;
		decosc_prev_1 = dec;
	}
	for i in (first + 2)..len {
		let d0 = data[i];
		let d1 = data[i - 1];
		let d2 = data[i - 2];

		let hp0 =
			c1 * d0 - 2.0 * c1 * d1 + c1 * d2 + 2.0 * one_minus_alpha1 * hp_prev_1 - one_minus_alpha1_sq * hp_prev_2;

		let dec = d0 - hp0;
		let d_dec1 = d1 - hp_prev_1;
		let d_dec2 = d2 - hp_prev_2;

		let decosc0 = c2 * dec - 2.0 * c2 * d_dec1 + c2 * d_dec2 + 2.0 * one_minus_alpha2 * decosc_prev_1
			- one_minus_alpha2_sq * decosc_prev_2;

		out[i] = 100.0 * k_val * decosc0 / d0;

		hp_prev_2 = hp_prev_1;
		hp_prev_1 = hp0;
		decosc_prev_2 = decosc_prev_1;
		decosc_prev_1 = decosc0;
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn dec_osc_avx2(data: &[f64], period: usize, k_val: f64, first: usize, out: &mut [f64]) {
	// AVX2 stub - call scalar.
	dec_osc_scalar(data, period, k_val, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn dec_osc_avx512_short(data: &[f64], period: usize, k_val: f64, first: usize, out: &mut [f64]) {
	// AVX512 short stub - call scalar.
	dec_osc_scalar(data, period, k_val, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn dec_osc_avx512_long(data: &[f64], period: usize, k_val: f64, first: usize, out: &mut [f64]) {
	// AVX512 long stub - call scalar.
	dec_osc_scalar(data, period, k_val, first, out)
}

#[inline(always)]
pub fn dec_osc_batch_with_kernel(
	data: &[f64],
	sweep: &DecOscBatchRange,
	k: Kernel,
) -> Result<DecOscBatchOutput, DecOscError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(DecOscError::InvalidPeriod { period: 0, data_len: 0 }),
	};

	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	dec_osc_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct DecOscBatchRange {
	pub hp_period: (usize, usize, usize),
	pub k: (f64, f64, f64),
}

impl Default for DecOscBatchRange {
	fn default() -> Self {
		Self {
			hp_period: (125, 125, 0),
			k: (1.0, 1.0, 0.0),
		}
	}
}

#[derive(Clone, Debug, Default)]
pub struct DecOscBatchBuilder {
	range: DecOscBatchRange,
	kernel: Kernel,
}

impl DecOscBatchBuilder {
	pub fn new() -> Self {
		Self::default()
	}
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline]
	pub fn hp_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.hp_period = (start, end, step);
		self
	}
	#[inline]
	pub fn hp_period_static(mut self, p: usize) -> Self {
		self.range.hp_period = (p, p, 0);
		self
	}
	#[inline]
	pub fn k_range(mut self, start: f64, end: f64, step: f64) -> Self {
		self.range.k = (start, end, step);
		self
	}
	#[inline]
	pub fn k_static(mut self, x: f64) -> Self {
		self.range.k = (x, x, 0.0);
		self
	}

	pub fn apply_slice(self, data: &[f64]) -> Result<DecOscBatchOutput, DecOscError> {
		dec_osc_batch_with_kernel(data, &self.range, self.kernel)
	}

	pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<DecOscBatchOutput, DecOscError> {
		DecOscBatchBuilder::new().kernel(k).apply_slice(data)
	}

	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<DecOscBatchOutput, DecOscError> {
		let slice = source_type(c, src);
		self.apply_slice(slice)
	}

	pub fn with_default_candles(c: &Candles) -> Result<DecOscBatchOutput, DecOscError> {
		DecOscBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
	}
}

#[derive(Clone, Debug)]
pub struct DecOscBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<DecOscParams>,
	pub rows: usize,
	pub cols: usize,
}
impl DecOscBatchOutput {
	pub fn row_for_params(&self, p: &DecOscParams) -> Option<usize> {
		self.combos.iter().position(|c| {
			c.hp_period.unwrap_or(125) == p.hp_period.unwrap_or(125)
				&& (c.k.unwrap_or(1.0) - p.k.unwrap_or(1.0)).abs() < 1e-12
		})
	}

	pub fn values_for(&self, p: &DecOscParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &DecOscBatchRange) -> Vec<DecOscParams> {
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

	let periods = axis_usize(r.hp_period);
	let ks = axis_f64(r.k);

	let mut out = Vec::with_capacity(periods.len() * ks.len());
	for &p in &periods {
		for &k in &ks {
			out.push(DecOscParams {
				hp_period: Some(p),
				k: Some(k),
			});
		}
	}
	out
}

#[inline(always)]
pub fn dec_osc_batch_slice(
	data: &[f64],
	sweep: &DecOscBatchRange,
	kern: Kernel,
) -> Result<DecOscBatchOutput, DecOscError> {
	dec_osc_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn dec_osc_batch_par_slice(
	data: &[f64],
	sweep: &DecOscBatchRange,
	kern: Kernel,
) -> Result<DecOscBatchOutput, DecOscError> {
	dec_osc_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn dec_osc_batch_inner(
	data: &[f64],
	sweep: &DecOscBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<DecOscBatchOutput, DecOscError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(DecOscError::InvalidPeriod { period: 0, data_len: 0 });
	}

	let first = data.iter().position(|x| !x.is_nan()).ok_or(DecOscError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.hp_period.unwrap()).max().unwrap();
	if data.len() - first < 2 {
		return Err(DecOscError::NotEnoughValidData {
			needed: 2,
			valid: data.len() - first,
		});
	}

	let rows = combos.len();
	let cols = data.len();

	let mut values = vec![f64::NAN; rows * cols];
	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let period = combos[row].hp_period.unwrap();
		let k_val = combos[row].k.unwrap();
		match kern {
			Kernel::Scalar => dec_osc_row_scalar(data, first, period, k_val, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => dec_osc_row_avx2(data, first, period, k_val, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => dec_osc_row_avx512(data, first, period, k_val, out_row),
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

	Ok(DecOscBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
unsafe fn dec_osc_row_scalar(data: &[f64], first: usize, period: usize, k_val: f64, out: &mut [f64]) {
	dec_osc_scalar(data, period, k_val, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn dec_osc_row_avx2(data: &[f64], first: usize, period: usize, k_val: f64, out: &mut [f64]) {
	dec_osc_scalar(data, period, k_val, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn dec_osc_row_avx512(data: &[f64], first: usize, period: usize, k_val: f64, out: &mut [f64]) {
	if period <= 32 {
		dec_osc_row_avx512_short(data, first, period, k_val, out)
	} else {
		dec_osc_row_avx512_long(data, first, period, k_val, out)
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn dec_osc_row_avx512_short(data: &[f64], first: usize, period: usize, k_val: f64, out: &mut [f64]) {
	dec_osc_scalar(data, period, k_val, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn dec_osc_row_avx512_long(data: &[f64], first: usize, period: usize, k_val: f64, out: &mut [f64]) {
	dec_osc_scalar(data, period, k_val, first, out)
}

#[derive(Debug, Clone)]
pub struct DecOscStream {
	period: usize,
	k: f64,
	hp_prev_2: f64,
	hp_prev_1: f64,
	decosc_prev_2: f64,
	decosc_prev_1: f64,
	index: usize,
	filled: bool,
}

impl DecOscStream {
	pub fn try_new(params: DecOscParams) -> Result<Self, DecOscError> {
		let period = params.hp_period.unwrap_or(125);
		if period < 2 {
			return Err(DecOscError::InvalidPeriod { period, data_len: 0 });
		}
		let k = params.k.unwrap_or(1.0);
		if k <= 0.0 || k.is_nan() {
			return Err(DecOscError::InvalidK { k });
		}
		Ok(Self {
			period,
			k,
			hp_prev_2: f64::NAN,
			hp_prev_1: f64::NAN,
			decosc_prev_2: 0.0,
			decosc_prev_1: 0.0,
			index: 0,
			filled: false,
		})
	}

	#[inline(always)]
	pub fn update(&mut self, value: f64) -> Option<f64> {
		self.index += 1;
		if self.index == 1 {
			self.hp_prev_2 = value;
			self.hp_prev_1 = value;
			return None;
		}
		if self.index == 2 {
			self.hp_prev_2 = self.hp_prev_1;
			self.hp_prev_1 = value;
			let dec = value - self.hp_prev_1;
			self.decosc_prev_2 = self.decosc_prev_1;
			self.decosc_prev_1 = dec;
			return None;
		}

		let period = self.period;
		let k = self.k;
		let half_period = (period as f64) * 0.5;

		let angle1 = 2.0 * PI * 0.707 / (period as f64);
		let sin1 = angle1.sin();
		let cos1 = angle1.cos();
		let alpha1 = 1.0 + ((sin1 - 1.0) / cos1);
		let c1 = (1.0 - alpha1 / 2.0) * (1.0 - alpha1 / 2.0);
		let one_minus_alpha1 = 1.0 - alpha1;
		let one_minus_alpha1_sq = one_minus_alpha1 * one_minus_alpha1;

		let angle2 = 2.0 * PI * 0.707 / half_period;
		let sin2 = angle2.sin();
		let cos2 = angle2.cos();
		let alpha2 = 1.0 + ((sin2 - 1.0) / cos2);
		let c2 = (1.0 - alpha2 / 2.0) * (1.0 - alpha2 / 2.0);
		let one_minus_alpha2 = 1.0 - alpha2;
		let one_minus_alpha2_sq = one_minus_alpha2 * one_minus_alpha2;

		let hp0 =
			c1 * value - 2.0 * c1 * self.hp_prev_1 + c1 * self.hp_prev_2 + 2.0 * one_minus_alpha1 * self.hp_prev_1
				- one_minus_alpha1_sq * self.hp_prev_2;

		let dec = value - hp0;
		let d_dec1 = self.hp_prev_1 - self.hp_prev_1;
		let d_dec2 = self.hp_prev_2 - self.hp_prev_2;

		let decosc0 = c2 * dec - 2.0 * c2 * d_dec1 + c2 * d_dec2 + 2.0 * one_minus_alpha2 * self.decosc_prev_1
			- one_minus_alpha2_sq * self.decosc_prev_2;

		self.hp_prev_2 = self.hp_prev_1;
		self.hp_prev_1 = hp0;
		self.decosc_prev_2 = self.decosc_prev_1;
		self.decosc_prev_1 = decosc0;
		Some(100.0 * k * decosc0 / value)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_dec_osc_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let default_params = DecOscParams {
			hp_period: None,
			k: None,
		};
		let input = DecOscInput::from_candles(&candles, "close", default_params);
		let output = dec_osc_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());

		Ok(())
	}

	fn check_dec_osc_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = DecOscInput::from_candles(&candles, "close", DecOscParams::default());
		let result = dec_osc_with_kernel(&input, kernel)?;

		if result.values.len() > 5 {
			let expected_last_five = [
				-1.5036367540303395,
				-1.4037875172207006,
				-1.3174199471429475,
				-1.2245874070642693,
				-1.1638422627265639,
			];
			let start = result.values.len().saturating_sub(5);
			for (i, &val) in result.values[start..].iter().enumerate() {
				let diff = (val - expected_last_five[i]).abs();
				assert!(
					diff < 1e-7,
					"[{}] DEC_OSC {:?} mismatch at idx {}: got {}, expected {}",
					test_name,
					kernel,
					i,
					val,
					expected_last_five[i]
				);
			}
		}
		Ok(())
	}

	fn check_dec_osc_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = DecOscInput::with_default_candles(&candles);
		match input.data {
			DecOscData::Candles { source, .. } => assert_eq!(source, "close"),
			_ => panic!("Expected DecOscData::Candles"),
		}
		let output = dec_osc_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_dec_osc_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let input_data = [10.0, 20.0, 30.0];
		let params = DecOscParams {
			hp_period: Some(0),
			k: Some(1.0),
		};
		let input = DecOscInput::from_slice(&input_data, params);
		let res = dec_osc_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] DEC_OSC should fail with zero period", test_name);
		Ok(())
	}

	fn check_dec_osc_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data_small = [10.0, 20.0, 30.0];
		let params = DecOscParams {
			hp_period: Some(10),
			k: Some(1.0),
		};
		let input = DecOscInput::from_slice(&data_small, params);
		let res = dec_osc_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] DEC_OSC should fail with period exceeding length",
			test_name
		);
		Ok(())
	}

	fn check_dec_osc_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let single_point = [42.0];
		let params = DecOscParams {
			hp_period: Some(125),
			k: Some(1.0),
		};
		let input = DecOscInput::from_slice(&single_point, params);
		let res = dec_osc_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] DEC_OSC should fail with insufficient data",
			test_name
		);
		Ok(())
	}

	fn check_dec_osc_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let first_params = DecOscParams {
			hp_period: Some(50),
			k: Some(1.0),
		};
		let first_input = DecOscInput::from_candles(&candles, "close", first_params);
		let first_result = dec_osc_with_kernel(&first_input, kernel)?;
		let second_params = DecOscParams {
			hp_period: Some(50),
			k: Some(1.0),
		};
		let second_input = DecOscInput::from_slice(&first_result.values, second_params);
		let second_result = dec_osc_with_kernel(&second_input, kernel)?;
		assert_eq!(second_result.values.len(), first_result.values.len());
		Ok(())
	}

	#[cfg(debug_assertions)]
	fn check_dec_osc_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		
		// Define comprehensive parameter combinations for DEC_OSC
		let test_params = vec![
			DecOscParams::default(),                                      // hp_period: 125, k: 1.0
			DecOscParams { hp_period: Some(2), k: Some(1.0) },          // minimum viable period
			DecOscParams { hp_period: Some(10), k: Some(1.0) },         // small period
			DecOscParams { hp_period: Some(50), k: Some(1.0) },         // medium period
			DecOscParams { hp_period: Some(125), k: Some(1.0) },        // default period
			DecOscParams { hp_period: Some(200), k: Some(1.0) },        // large period
			DecOscParams { hp_period: Some(500), k: Some(1.0) },        // very large period
			DecOscParams { hp_period: Some(50), k: Some(0.5) },         // small k
			DecOscParams { hp_period: Some(50), k: Some(2.0) },         // large k
			DecOscParams { hp_period: Some(125), k: Some(0.1) },        // very small k
			DecOscParams { hp_period: Some(125), k: Some(10.0) },       // very large k
			DecOscParams { hp_period: Some(20), k: Some(1.5) },         // mixed params
			DecOscParams { hp_period: Some(100), k: Some(0.75) },       // mixed params
		];
		
		for (param_idx, params) in test_params.iter().enumerate() {
			let input = DecOscInput::from_candles(&candles, "close", params.clone());
			let output = dec_osc_with_kernel(&input, kernel)?;
			
			for (i, &val) in output.values.iter().enumerate() {
				if val.is_nan() {
					continue; // NaN values are expected during warmup
				}
				
				let bits = val.to_bits();
				
				// Check all three poison patterns
				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: hp_period={}, k={} (param set {})",
						test_name, val, bits, i, 
						params.hp_period.unwrap_or(125), 
						params.k.unwrap_or(1.0), 
						param_idx
					);
				}
				
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: hp_period={}, k={} (param set {})",
						test_name, val, bits, i,
						params.hp_period.unwrap_or(125),
						params.k.unwrap_or(1.0),
						param_idx
					);
				}
				
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: hp_period={}, k={} (param set {})",
						test_name, val, bits, i,
						params.hp_period.unwrap_or(125),
						params.k.unwrap_or(1.0),
						param_idx
					);
				}
			}
		}
		
		Ok(())
	}

	#[cfg(not(debug_assertions))]
	fn check_dec_osc_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		Ok(()) // No-op in release builds
	}

	macro_rules! generate_all_dec_osc_tests {
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

	generate_all_dec_osc_tests!(
		check_dec_osc_partial_params,
		check_dec_osc_accuracy,
		check_dec_osc_default_candles,
		check_dec_osc_zero_period,
		check_dec_osc_period_exceeds_length,
		check_dec_osc_very_small_dataset,
		check_dec_osc_reinput,
		check_dec_osc_no_poison
	);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let output = DecOscBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;
		let def = DecOscParams::default();
		let row = output.values_for(&def).expect("default row missing");
		assert_eq!(row.len(), c.close.len());
		let expected = [
			-1.5036367540303395,
			-1.4037875172207006,
			-1.3174199471429475,
			-1.2245874070642693,
			-1.1638422627265639,
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

	#[cfg(debug_assertions)]
	fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);
		
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		
		// Test various parameter sweep configurations for DEC_OSC
		let test_configs = vec![
			// (hp_period_start, hp_period_end, hp_period_step, k_start, k_end, k_step)
			(2, 10, 2, 1.0, 1.0, 0.0),          // Small periods, fixed k
			(10, 50, 10, 1.0, 1.0, 0.0),        // Medium periods, fixed k
			(50, 150, 25, 1.0, 1.0, 0.0),       // Large periods, fixed k
			(125, 125, 0, 0.5, 2.0, 0.5),       // Fixed period, varying k
			(20, 40, 5, 0.5, 1.5, 0.25),        // Mixed sweep
			(100, 200, 50, 1.0, 1.0, 0.0),      // Very large periods
			(2, 5, 1, 0.1, 10.0, 4.95),         // Dense small range with extreme k
		];
		
		for (cfg_idx, &(hp_start, hp_end, hp_step, k_start, k_end, k_step)) in test_configs.iter().enumerate() {
			let mut builder = DecOscBatchBuilder::new().kernel(kernel);
			
			// Configure period range
			if hp_step > 0 {
				builder = builder.hp_period_range(hp_start, hp_end, hp_step);
			} else {
				builder = builder.hp_period_range(hp_start, hp_start, 1);
			}
			
			// Configure k range
			if k_step > 0.0 {
				builder = builder.k_range(k_start, k_end, k_step);
			}
			
			let output = builder.apply_candles(&c, "close")?;
			
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
						 at row {} col {} (flat index {}) with params: hp_period={}, k={}",
						test, cfg_idx, val, bits, row, col, idx, 
						combo.hp_period.unwrap_or(125),
						combo.k.unwrap_or(1.0)
					);
				}
				
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: hp_period={}, k={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.hp_period.unwrap_or(125),
						combo.k.unwrap_or(1.0)
					);
				}
				
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: hp_period={}, k={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.hp_period.unwrap_or(125),
						combo.k.unwrap_or(1.0)
					);
				}
			}
		}
		
		Ok(())
	}

	#[cfg(not(debug_assertions))]
	fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		Ok(()) // No-op in release builds
	}
}
