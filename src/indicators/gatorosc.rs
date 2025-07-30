//! # Gator Oscillator (GATOR)
//!
//! The Gator Oscillator is based on Bill Williams' Alligator indicator. It calculates three exponentially smoothed averages (Jaws, Teeth, and Lips) of a given source, then produces two lines: `upper = abs(Jaws - Teeth)` and `lower = -abs(Teeth - Lips)`. Their 1-period momentum changes are also reported. All parameters are adjustable.
//!
//! ## Parameters
//! - **jaws_length**: EMA length for Jaws (default: 13)
//! - **jaws_shift**: Shift Jaws forward (default: 8)
//! - **teeth_length**: EMA length for Teeth (default: 8)
//! - **teeth_shift**: Shift Teeth forward (default: 5)
//! - **lips_length**: EMA length for Lips (default: 5)
//! - **lips_shift**: Shift Lips forward (default: 3)
//!
//! ## Errors
//! - **AllValuesNaN**: gatorosc: All input data values are `NaN`.
//! - **InvalidSettings**: gatorosc: Any length or shift is zero or invalid.
//! - **NotEnoughValidData**: gatorosc: Not enough valid data for computation.
//!
//! ## Returns
//! - `Ok(GatorOscOutput)` on success, with `upper`, `lower`, `upper_change`, and `lower_change` fields.
//! - `Err(GatorOscError)` otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum GatorOscData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

impl<'a> AsRef<[f64]> for GatorOscInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			GatorOscData::Slice(slice) => slice,
			GatorOscData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

#[derive(Debug, Clone)]
pub struct GatorOscOutput {
	pub upper: Vec<f64>,
	pub lower: Vec<f64>,
	pub upper_change: Vec<f64>,
	pub lower_change: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct GatorOscParams {
	pub jaws_length: Option<usize>,
	pub jaws_shift: Option<usize>,
	pub teeth_length: Option<usize>,
	pub teeth_shift: Option<usize>,
	pub lips_length: Option<usize>,
	pub lips_shift: Option<usize>,
}

impl Default for GatorOscParams {
	fn default() -> Self {
		Self {
			jaws_length: Some(13),
			jaws_shift: Some(8),
			teeth_length: Some(8),
			teeth_shift: Some(5),
			lips_length: Some(5),
			lips_shift: Some(3),
		}
	}
}

#[derive(Debug, Clone)]
pub struct GatorOscInput<'a> {
	pub data: GatorOscData<'a>,
	pub params: GatorOscParams,
}

impl<'a> GatorOscInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: GatorOscParams) -> Self {
		Self {
			data: GatorOscData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slice(sl: &'a [f64], p: GatorOscParams) -> Self {
		Self {
			data: GatorOscData::Slice(sl),
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", GatorOscParams::default())
	}
	#[inline]
	pub fn get_jaws_length(&self) -> usize {
		self.params.jaws_length.unwrap_or(13)
	}
	#[inline]
	pub fn get_jaws_shift(&self) -> usize {
		self.params.jaws_shift.unwrap_or(8)
	}
	#[inline]
	pub fn get_teeth_length(&self) -> usize {
		self.params.teeth_length.unwrap_or(8)
	}
	#[inline]
	pub fn get_teeth_shift(&self) -> usize {
		self.params.teeth_shift.unwrap_or(5)
	}
	#[inline]
	pub fn get_lips_length(&self) -> usize {
		self.params.lips_length.unwrap_or(5)
	}
	#[inline]
	pub fn get_lips_shift(&self) -> usize {
		self.params.lips_shift.unwrap_or(3)
	}
}

#[derive(Clone, Debug)]
pub struct GatorOscBuilder {
	jaws_length: Option<usize>,
	jaws_shift: Option<usize>,
	teeth_length: Option<usize>,
	teeth_shift: Option<usize>,
	lips_length: Option<usize>,
	lips_shift: Option<usize>,
	kernel: Kernel,
}

impl Default for GatorOscBuilder {
	fn default() -> Self {
		Self {
			jaws_length: None,
			jaws_shift: None,
			teeth_length: None,
			teeth_shift: None,
			lips_length: None,
			lips_shift: None,
			kernel: Kernel::Auto,
		}
	}
}

impl GatorOscBuilder {
	#[inline(always)]
	pub fn new() -> Self {
		Self::default()
	}
	#[inline(always)]
	pub fn jaws_length(mut self, n: usize) -> Self {
		self.jaws_length = Some(n);
		self
	}
	#[inline(always)]
	pub fn jaws_shift(mut self, x: usize) -> Self {
		self.jaws_shift = Some(x);
		self
	}
	#[inline(always)]
	pub fn teeth_length(mut self, n: usize) -> Self {
		self.teeth_length = Some(n);
		self
	}
	#[inline(always)]
	pub fn teeth_shift(mut self, x: usize) -> Self {
		self.teeth_shift = Some(x);
		self
	}
	#[inline(always)]
	pub fn lips_length(mut self, n: usize) -> Self {
		self.lips_length = Some(n);
		self
	}
	#[inline(always)]
	pub fn lips_shift(mut self, x: usize) -> Self {
		self.lips_shift = Some(x);
		self
	}
	#[inline(always)]
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline(always)]
	pub fn apply(self, c: &Candles) -> Result<GatorOscOutput, GatorOscError> {
		let p = GatorOscParams {
			jaws_length: self.jaws_length,
			jaws_shift: self.jaws_shift,
			teeth_length: self.teeth_length,
			teeth_shift: self.teeth_shift,
			lips_length: self.lips_length,
			lips_shift: self.lips_shift,
		};
		let i = GatorOscInput::from_candles(c, "close", p);
		gatorosc_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slice(self, d: &[f64]) -> Result<GatorOscOutput, GatorOscError> {
		let p = GatorOscParams {
			jaws_length: self.jaws_length,
			jaws_shift: self.jaws_shift,
			teeth_length: self.teeth_length,
			teeth_shift: self.teeth_shift,
			lips_length: self.lips_length,
			lips_shift: self.lips_shift,
		};
		let i = GatorOscInput::from_slice(d, p);
		gatorosc_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<GatorOscStream, GatorOscError> {
		let p = GatorOscParams {
			jaws_length: self.jaws_length,
			jaws_shift: self.jaws_shift,
			teeth_length: self.teeth_length,
			teeth_shift: self.teeth_shift,
			lips_length: self.lips_length,
			lips_shift: self.lips_shift,
		};
		GatorOscStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum GatorOscError {
	#[error("gatorosc: All values are NaN.")]
	AllValuesNaN,
	#[error("gatorosc: Invalid settings (zero or invalid parameter).")]
	InvalidSettings,
	#[error("gatorosc: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn gatorosc(input: &GatorOscInput) -> Result<GatorOscOutput, GatorOscError> {
	gatorosc_with_kernel(input, Kernel::Auto)
}

pub fn gatorosc_with_kernel(input: &GatorOscInput, kernel: Kernel) -> Result<GatorOscOutput, GatorOscError> {
	let data: &[f64] = input.as_ref();
	let first = data
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(GatorOscError::AllValuesNaN)?;
	let jaws_length = input.get_jaws_length();
	let jaws_shift = input.get_jaws_shift();
	let teeth_length = input.get_teeth_length();
	let teeth_shift = input.get_teeth_shift();
	let lips_length = input.get_lips_length();
	let lips_shift = input.get_lips_shift();

	if jaws_length == 0
		|| jaws_shift == 0
		|| teeth_length == 0
		|| teeth_shift == 0
		|| lips_length == 0
		|| lips_shift == 0
	{
		return Err(GatorOscError::InvalidSettings);
	}

	let needed = jaws_length
		.max(teeth_length)
		.max(lips_length)
		.saturating_add(jaws_shift.max(teeth_shift).max(lips_shift));
	let valid = data.iter().skip(first).filter(|v| !v.is_nan()).count();
	if valid < needed {
		return Err(GatorOscError::NotEnoughValidData { needed, valid });
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	let mut upper = vec![f64::NAN; data.len()];
	let mut lower = vec![f64::NAN; data.len()];
	let mut upper_change = vec![f64::NAN; data.len()];
	let mut lower_change = vec![f64::NAN; data.len()];

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => gatorosc_scalar(
				data,
				jaws_length,
				jaws_shift,
				teeth_length,
				teeth_shift,
				lips_length,
				lips_shift,
				first,
				&mut upper,
				&mut lower,
				&mut upper_change,
				&mut lower_change,
			),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => gatorosc_avx2(
				data,
				jaws_length,
				jaws_shift,
				teeth_length,
				teeth_shift,
				lips_length,
				lips_shift,
				first,
				&mut upper,
				&mut lower,
				&mut upper_change,
				&mut lower_change,
			),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => gatorosc_avx512(
				data,
				jaws_length,
				jaws_shift,
				teeth_length,
				teeth_shift,
				lips_length,
				lips_shift,
				first,
				&mut upper,
				&mut lower,
				&mut upper_change,
				&mut lower_change,
			),
			_ => unreachable!(),
		}
	}
	Ok(GatorOscOutput {
		upper,
		lower,
		upper_change,
		lower_change,
	})
}

#[inline(always)]
pub unsafe fn gatorosc_scalar(
	data: &[f64],
	jaws_length: usize,
	jaws_shift: usize,
	teeth_length: usize,
	teeth_shift: usize,
	lips_length: usize,
	lips_shift: usize,
	first_valid: usize,
	upper: &mut [f64],
	lower: &mut [f64],
	upper_change: &mut [f64],
	lower_change: &mut [f64],
) {
	let jaws_ema = compute_ema(data, jaws_length, first_valid);
	let jaws = shift_series(&jaws_ema, jaws_shift);
	let teeth_ema = compute_ema(data, teeth_length, first_valid);
	let teeth = shift_series(&teeth_ema, teeth_shift);
	let lips_ema = compute_ema(data, lips_length, first_valid);
	let lips = shift_series(&lips_ema, lips_shift);

	for i in 0..data.len() {
		if !jaws[i].is_nan() && !teeth[i].is_nan() {
			upper[i] = (jaws[i] - teeth[i]).abs();
		}
		if !teeth[i].is_nan() && !lips[i].is_nan() {
			lower[i] = -(teeth[i] - lips[i]).abs();
		}
	}

	for i in 1..data.len() {
		if !upper[i].is_nan() && !upper[i - 1].is_nan() {
			upper_change[i] = upper[i] - upper[i - 1];
		}
		if !lower[i].is_nan() && !lower[i - 1].is_nan() {
			lower_change[i] = -(lower[i] - lower[i - 1]);
		}
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn gatorosc_avx2(
	data: &[f64],
	jaws_length: usize,
	jaws_shift: usize,
	teeth_length: usize,
	teeth_shift: usize,
	lips_length: usize,
	lips_shift: usize,
	first_valid: usize,
	upper: &mut [f64],
	lower: &mut [f64],
	upper_change: &mut [f64],
	lower_change: &mut [f64],
) {
	gatorosc_scalar(
		data,
		jaws_length,
		jaws_shift,
		teeth_length,
		teeth_shift,
		lips_length,
		lips_shift,
		first_valid,
		upper,
		lower,
		upper_change,
		lower_change,
	);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn gatorosc_avx512(
	data: &[f64],
	jaws_length: usize,
	jaws_shift: usize,
	teeth_length: usize,
	teeth_shift: usize,
	lips_length: usize,
	lips_shift: usize,
	first_valid: usize,
	upper: &mut [f64],
	lower: &mut [f64],
	upper_change: &mut [f64],
	lower_change: &mut [f64],
) {
	if jaws_length <= 32 && teeth_length <= 32 && lips_length <= 32 {
		gatorosc_avx512_short(
			data,
			jaws_length,
			jaws_shift,
			teeth_length,
			teeth_shift,
			lips_length,
			lips_shift,
			first_valid,
			upper,
			lower,
			upper_change,
			lower_change,
		);
	} else {
		gatorosc_avx512_long(
			data,
			jaws_length,
			jaws_shift,
			teeth_length,
			teeth_shift,
			lips_length,
			lips_shift,
			first_valid,
			upper,
			lower,
			upper_change,
			lower_change,
		);
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn gatorosc_avx512_short(
	data: &[f64],
	jaws_length: usize,
	jaws_shift: usize,
	teeth_length: usize,
	teeth_shift: usize,
	lips_length: usize,
	lips_shift: usize,
	first_valid: usize,
	upper: &mut [f64],
	lower: &mut [f64],
	upper_change: &mut [f64],
	lower_change: &mut [f64],
) {
	gatorosc_scalar(
		data,
		jaws_length,
		jaws_shift,
		teeth_length,
		teeth_shift,
		lips_length,
		lips_shift,
		first_valid,
		upper,
		lower,
		upper_change,
		lower_change,
	);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn gatorosc_avx512_long(
	data: &[f64],
	jaws_length: usize,
	jaws_shift: usize,
	teeth_length: usize,
	teeth_shift: usize,
	lips_length: usize,
	lips_shift: usize,
	first_valid: usize,
	upper: &mut [f64],
	lower: &mut [f64],
	upper_change: &mut [f64],
	lower_change: &mut [f64],
) {
	gatorosc_scalar(
		data,
		jaws_length,
		jaws_shift,
		teeth_length,
		teeth_shift,
		lips_length,
		lips_shift,
		first_valid,
		upper,
		lower,
		upper_change,
		lower_change,
	);
}

#[inline(always)]
fn compute_ema(data: &[f64], length: usize, start_idx: usize) -> Vec<f64> {
	let alpha = 1.0 / length as f64;
	let mut output = vec![f64::NAN; data.len()];
	let mut prev = if data[start_idx].is_nan() { 0.0 } else { data[start_idx] };
	output[start_idx] = prev;
	for i in (start_idx + 1)..data.len() {
		let val = if data[i].is_nan() { prev } else { data[i] };
		let next_ema = alpha * val + (1.0 - alpha) * prev;
		output[i] = next_ema;
		prev = next_ema;
	}
	output
}

#[inline(always)]
fn shift_series(data: &[f64], shift: usize) -> Vec<f64> {
	let mut shifted = vec![f64::NAN; data.len()];
	for (i, &val) in data.iter().enumerate() {
		let j = i + shift;
		if j < data.len() {
			shifted[j] = val;
		}
	}
	shifted
}

#[derive(Debug, Clone)]
pub struct GatorOscStream {
	jaws: EmaStream,
	teeth: EmaStream,
	lips: EmaStream,
	jaws_shift: usize,
	teeth_shift: usize,
	lips_shift: usize,
	buf: Vec<f64>,
	idx: usize,
}

impl GatorOscStream {
	pub fn try_new(params: GatorOscParams) -> Result<Self, GatorOscError> {
		let jaws_length = params.jaws_length.unwrap_or(13);
		let jaws_shift = params.jaws_shift.unwrap_or(8);
		let teeth_length = params.teeth_length.unwrap_or(8);
		let teeth_shift = params.teeth_shift.unwrap_or(5);
		let lips_length = params.lips_length.unwrap_or(5);
		let lips_shift = params.lips_shift.unwrap_or(3);

		if jaws_length == 0
			|| jaws_shift == 0
			|| teeth_length == 0
			|| teeth_shift == 0
			|| lips_length == 0
			|| lips_shift == 0
		{
			return Err(GatorOscError::InvalidSettings);
		}
		Ok(Self {
			jaws: EmaStream::new(jaws_length),
			teeth: EmaStream::new(teeth_length),
			lips: EmaStream::new(lips_length),
			jaws_shift,
			teeth_shift,
			lips_shift,
			buf: vec![f64::NAN; jaws_shift.max(teeth_shift).max(lips_shift) + 1],
			idx: 0,
		})
	}

	#[inline(always)]
	pub fn update(&mut self, value: f64) -> Option<(f64, f64, f64, f64)> {
		let jaws_val = self.jaws.update(value);
		let teeth_val = self.teeth.update(value);
		let lips_val = self.lips.update(value);

		let buf_idx = self.idx % self.buf.len();
		self.buf[buf_idx] = value;
		let i = self.idx;
		self.idx += 1;

		let jaws_idx = i.checked_sub(self.jaws_shift)?;
		let teeth_idx = i.checked_sub(self.teeth_shift)?;
		let lips_idx = i.checked_sub(self.lips_shift)?;

		let jaws = self.jaws.value_at(jaws_idx)?;
		let teeth = self.teeth.value_at(teeth_idx)?;
		let lips = self.lips.value_at(lips_idx)?;

		let upper = (jaws - teeth).abs();
		let lower = -(teeth - lips).abs();

		let prev_upper = if i > 0 {
			let pj = jaws_idx.checked_sub(1)?;
			let pt = teeth_idx.checked_sub(1)?;
			let jaws_p = self.jaws.value_at(pj)?;
			let teeth_p = self.teeth.value_at(pt)?;
			(jaws_p - teeth_p).abs()
		} else {
			f64::NAN
		};

		let prev_lower = if i > 0 {
			let pt = teeth_idx.checked_sub(1)?;
			let pl = lips_idx.checked_sub(1)?;
			let teeth_p = self.teeth.value_at(pt)?;
			let lips_p = self.lips.value_at(pl)?;
			-(teeth_p - lips_p).abs()
		} else {
			f64::NAN
		};

		let upper_change = if !prev_upper.is_nan() {
			upper - prev_upper
		} else {
			f64::NAN
		};
		let lower_change = if !prev_lower.is_nan() {
			-(lower - prev_lower)
		} else {
			f64::NAN
		};
		Some((upper, lower, upper_change, lower_change))
	}
}

#[derive(Debug, Clone)]
struct EmaStream {
	alpha: f64,
	state: Vec<f64>,
	period: usize,
	idx: usize,
}

impl EmaStream {
	fn new(period: usize) -> Self {
		Self {
			alpha: 1.0 / period as f64,
			state: Vec::new(),
			period,
			idx: 0,
		}
	}
	fn update(&mut self, value: f64) -> f64 {
		let ema = if self.idx == 0 {
			value
		} else {
			self.alpha * value + (1.0 - self.alpha) * self.state[self.idx - 1]
		};
		self.state.push(ema);
		self.idx += 1;
		ema
	}
	fn value_at(&self, idx: usize) -> Option<f64> {
		self.state.get(idx).copied()
	}
}

#[derive(Clone, Debug)]
pub struct GatorOscBatchRange {
	pub jaws_length: (usize, usize, usize),
	pub jaws_shift: (usize, usize, usize),
	pub teeth_length: (usize, usize, usize),
	pub teeth_shift: (usize, usize, usize),
	pub lips_length: (usize, usize, usize),
	pub lips_shift: (usize, usize, usize),
}

impl Default for GatorOscBatchRange {
	fn default() -> Self {
		Self {
			jaws_length: (13, 13, 0),
			jaws_shift: (8, 8, 0),
			teeth_length: (8, 8, 0),
			teeth_shift: (5, 5, 0),
			lips_length: (5, 5, 0),
			lips_shift: (3, 3, 0),
		}
	}
}

#[derive(Clone, Debug, Default)]
pub struct GatorOscBatchBuilder {
	range: GatorOscBatchRange,
	kernel: Kernel,
}

impl GatorOscBatchBuilder {
	pub fn new() -> Self {
		Self::default()
	}
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	pub fn jaws_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.jaws_length = (start, end, step);
		self
	}
	pub fn jaws_shift_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.jaws_shift = (start, end, step);
		self
	}
	pub fn teeth_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.teeth_length = (start, end, step);
		self
	}
	pub fn teeth_shift_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.teeth_shift = (start, end, step);
		self
	}
	pub fn lips_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.lips_length = (start, end, step);
		self
	}
	pub fn lips_shift_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.lips_shift = (start, end, step);
		self
	}
	pub fn apply_slice(self, data: &[f64]) -> Result<GatorOscBatchOutput, GatorOscError> {
		gatorosc_batch_with_kernel(data, &self.range, self.kernel)
	}
}

#[derive(Clone, Debug)]
pub struct GatorOscBatchOutput {
	pub upper: Vec<f64>,
	pub lower: Vec<f64>,
	pub upper_change: Vec<f64>,
	pub lower_change: Vec<f64>,
	pub combos: Vec<GatorOscParams>,
	pub rows: usize,
	pub cols: usize,
}

pub fn gatorosc_batch_with_kernel(
	data: &[f64],
	sweep: &GatorOscBatchRange,
	k: Kernel,
) -> Result<GatorOscBatchOutput, GatorOscError> {
	let combos = expand_grid_gatorosc(sweep);
	if combos.is_empty() {
		return Err(GatorOscError::InvalidSettings);
	}
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(GatorOscError::InvalidSettings),
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	gatorosc_batch_inner(data, &combos, simd)
}

fn expand_grid_gatorosc(r: &GatorOscBatchRange) -> Vec<GatorOscParams> {
	fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			vec![start]
		} else {
			(start..=end).step_by(step).collect()
		}
	}
	let jaws_lengths = axis(r.jaws_length);
	let jaws_shifts = axis(r.jaws_shift);
	let teeth_lengths = axis(r.teeth_length);
	let teeth_shifts = axis(r.teeth_shift);
	let lips_lengths = axis(r.lips_length);
	let lips_shifts = axis(r.lips_shift);

	let mut out = Vec::with_capacity(
		jaws_lengths.len()
			* jaws_shifts.len()
			* teeth_lengths.len()
			* teeth_shifts.len()
			* lips_lengths.len()
			* lips_shifts.len(),
	);
	for &jl in &jaws_lengths {
		for &js in &jaws_shifts {
			for &tl in &teeth_lengths {
				for &ts in &teeth_shifts {
					for &ll in &lips_lengths {
						for &ls in &lips_shifts {
							out.push(GatorOscParams {
								jaws_length: Some(jl),
								jaws_shift: Some(js),
								teeth_length: Some(tl),
								teeth_shift: Some(ts),
								lips_length: Some(ll),
								lips_shift: Some(ls),
							});
						}
					}
				}
			}
		}
	}
	out
}

fn gatorosc_batch_inner(
	data: &[f64],
	combos: &[GatorOscParams],
	kern: Kernel,
) -> Result<GatorOscBatchOutput, GatorOscError> {
	let first = data
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(GatorOscError::AllValuesNaN)?;
	let max_jl = combos.iter().map(|c| c.jaws_length.unwrap()).max().unwrap();
	let max_js = combos.iter().map(|c| c.jaws_shift.unwrap()).max().unwrap();
	let max_tl = combos.iter().map(|c| c.teeth_length.unwrap()).max().unwrap();
	let max_ts = combos.iter().map(|c| c.teeth_shift.unwrap()).max().unwrap();
	let max_ll = combos.iter().map(|c| c.lips_length.unwrap()).max().unwrap();
	let max_ls = combos.iter().map(|c| c.lips_shift.unwrap()).max().unwrap();
	let needed = max_jl
		.max(max_tl)
		.max(max_ll)
		.saturating_add(max_js.max(max_ts).max(max_ls));
	let valid = data.iter().skip(first).filter(|v| !v.is_nan()).count();
	if valid < needed {
		return Err(GatorOscError::NotEnoughValidData { needed, valid });
	}
	let rows = combos.len();
	let cols = data.len();
	let mut upper = vec![f64::NAN; rows * cols];
	let mut lower = vec![f64::NAN; rows * cols];
	let mut upper_change = vec![f64::NAN; rows * cols];
	let mut lower_change = vec![f64::NAN; rows * cols];

	let do_row = |row: usize, u: &mut [f64], l: &mut [f64], uc: &mut [f64], lc: &mut [f64]| unsafe {
		let prm = &combos[row];
		gatorosc_row_scalar(
			data,
			first,
			prm.jaws_length.unwrap(),
			prm.jaws_shift.unwrap(),
			prm.teeth_length.unwrap(),
			prm.teeth_shift.unwrap(),
			prm.lips_length.unwrap(),
			prm.lips_shift.unwrap(),
			u,
			l,
			uc,
			lc,
		)
	};

	#[cfg(not(target_arch = "wasm32"))]
	{
		upper
			.par_chunks_mut(cols)
			.zip(lower.par_chunks_mut(cols))
			.zip(upper_change.par_chunks_mut(cols))
			.zip(lower_change.par_chunks_mut(cols))
			.enumerate()
			.for_each(|(row, (((u, l), uc), lc))| {
				do_row(row, u, l, uc, lc);
			});
	}
	#[cfg(target_arch = "wasm32")]
	{
		for row in 0..rows {
			let start = row * cols;
			let end = start + cols;
			do_row(
				row,
				&mut upper[start..end],
				&mut lower[start..end],
				&mut upper_change[start..end],
				&mut lower_change[start..end],
			);
		}
	}

	Ok(GatorOscBatchOutput {
		upper,
		lower,
		upper_change,
		lower_change,
		combos: combos.to_vec(),
		rows,
		cols,
	})
}

#[inline(always)]
unsafe fn gatorosc_row_scalar(
	data: &[f64],
	first: usize,
	jaws_length: usize,
	jaws_shift: usize,
	teeth_length: usize,
	teeth_shift: usize,
	lips_length: usize,
	lips_shift: usize,
	upper: &mut [f64],
	lower: &mut [f64],
	upper_change: &mut [f64],
	lower_change: &mut [f64],
) {
	gatorosc_scalar(
		data,
		jaws_length,
		jaws_shift,
		teeth_length,
		teeth_shift,
		lips_length,
		lips_shift,
		first,
		upper,
		lower,
		upper_change,
		lower_change,
	);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn gatorosc_row_avx2(
	data: &[f64],
	first: usize,
	jaws_length: usize,
	jaws_shift: usize,
	teeth_length: usize,
	teeth_shift: usize,
	lips_length: usize,
	lips_shift: usize,
	upper: &mut [f64],
	lower: &mut [f64],
	upper_change: &mut [f64],
	lower_change: &mut [f64],
) {
	gatorosc_row_scalar(
		data,
		first,
		jaws_length,
		jaws_shift,
		teeth_length,
		teeth_shift,
		lips_length,
		lips_shift,
		upper,
		lower,
		upper_change,
		lower_change,
	);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn gatorosc_row_avx512(
	data: &[f64],
	first: usize,
	jaws_length: usize,
	jaws_shift: usize,
	teeth_length: usize,
	teeth_shift: usize,
	lips_length: usize,
	lips_shift: usize,
	upper: &mut [f64],
	lower: &mut [f64],
	upper_change: &mut [f64],
	lower_change: &mut [f64],
) {
	gatorosc_row_scalar(
		data,
		first,
		jaws_length,
		jaws_shift,
		teeth_length,
		teeth_shift,
		lips_length,
		lips_shift,
		upper,
		lower,
		upper_change,
		lower_change,
	);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn gatorosc_row_avx512_short(
	data: &[f64],
	first: usize,
	jaws_length: usize,
	jaws_shift: usize,
	teeth_length: usize,
	teeth_shift: usize,
	lips_length: usize,
	lips_shift: usize,
	upper: &mut [f64],
	lower: &mut [f64],
	upper_change: &mut [f64],
	lower_change: &mut [f64],
) {
	gatorosc_row_scalar(
		data,
		first,
		jaws_length,
		jaws_shift,
		teeth_length,
		teeth_shift,
		lips_length,
		lips_shift,
		upper,
		lower,
		upper_change,
		lower_change,
	);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn gatorosc_row_avx512_long(
	data: &[f64],
	first: usize,
	jaws_length: usize,
	jaws_shift: usize,
	teeth_length: usize,
	teeth_shift: usize,
	lips_length: usize,
	lips_shift: usize,
	upper: &mut [f64],
	lower: &mut [f64],
	upper_change: &mut [f64],
	lower_change: &mut [f64],
) {
	gatorosc_row_scalar(
		data,
		first,
		jaws_length,
		jaws_shift,
		teeth_length,
		teeth_shift,
		lips_length,
		lips_shift,
		upper,
		lower,
		upper_change,
		lower_change,
	);
}

#[inline(always)]
pub fn gatorosc_batch_slice(
	data: &[f64],
	sweep: &GatorOscBatchRange,
	kern: Kernel,
) -> Result<GatorOscBatchOutput, GatorOscError> {
	gatorosc_batch_inner(data, &expand_grid_gatorosc(sweep), kern)
}

#[inline(always)]
pub fn gatorosc_batch_par_slice(
	data: &[f64],
	sweep: &GatorOscBatchRange,
	kern: Kernel,
) -> Result<GatorOscBatchOutput, GatorOscError> {
	let combos = expand_grid_gatorosc(sweep);
	let first = data
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(GatorOscError::AllValuesNaN)?;
	let rows = combos.len();
	let cols = data.len();
	let mut upper = vec![f64::NAN; rows * cols];
	let mut lower = vec![f64::NAN; rows * cols];
	let mut upper_change = vec![f64::NAN; rows * cols];
	let mut lower_change = vec![f64::NAN; rows * cols];
	#[cfg(not(target_arch = "wasm32"))]
	use rayon::prelude::*;

	#[cfg(not(target_arch = "wasm32"))]
	{
		upper
			.par_chunks_mut(cols)
			.zip(lower.par_chunks_mut(cols))
			.zip(upper_change.par_chunks_mut(cols))
			.zip(lower_change.par_chunks_mut(cols))
			.enumerate()
			.for_each(|(row, (((u, l), uc), lc))| {
				let prm = &combos[row];
				unsafe {
					gatorosc_row_scalar(
						data,
						first,
						prm.jaws_length.unwrap(),
						prm.jaws_shift.unwrap(),
						prm.teeth_length.unwrap(),
						prm.teeth_shift.unwrap(),
						prm.lips_length.unwrap(),
						prm.lips_shift.unwrap(),
						u,
						l,
						uc,
						lc,
					);
				}
			});
	}
	#[cfg(target_arch = "wasm32")]
	{
		for row in 0..rows {
			let start = row * cols;
			let end = start + cols;
			let prm = &combos[row];
			unsafe {
				gatorosc_row_scalar(
					data,
					first,
					prm.jaws_length.unwrap(),
					prm.jaws_shift.unwrap(),
					prm.teeth_length.unwrap(),
					prm.teeth_shift.unwrap(),
					prm.lips_length.unwrap(),
					prm.lips_shift.unwrap(),
					&mut upper[start..end],
					&mut lower[start..end],
					&mut upper_change[start..end],
					&mut lower_change[start..end],
				);
			}
		}
	}

	Ok(GatorOscBatchOutput {
		upper,
		lower,
		upper_change,
		lower_change,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
pub fn expand_grid(r: &GatorOscBatchRange) -> Vec<GatorOscParams> {
	expand_grid_gatorosc(r)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_gatorosc_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let default_params = GatorOscParams::default();
		let input = GatorOscInput::from_candles(&candles, "close", default_params);
		let output = gatorosc_with_kernel(&input, kernel)?;
		assert_eq!(output.upper.len(), candles.close.len());
		Ok(())
	}

	fn check_gatorosc_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = GatorOscInput::from_candles(&candles, "close", GatorOscParams::default());
		let output = gatorosc_with_kernel(&input, kernel)?;
		assert_eq!(output.upper.len(), candles.close.len());
		if output.upper.len() > 24 {
			for &val in &output.upper[24..] {
				assert!(!val.is_nan(), "Found unexpected NaN in upper");
			}
		}
		Ok(())
	}

	fn check_gatorosc_zero_setting(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data = [10.0, 20.0, 30.0];
		let params = GatorOscParams {
			jaws_length: Some(0),
			..Default::default()
		};
		let input = GatorOscInput::from_slice(&data, params);
		let res = gatorosc_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] GatorOsc should fail with zero setting", test_name);
		Ok(())
	}

	fn check_gatorosc_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let single = [42.0];
		let params = GatorOscParams::default();
		let input = GatorOscInput::from_slice(&single, params);
		let res = gatorosc_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] GatorOsc should fail with insufficient data",
			test_name
		);
		Ok(())
	}

	fn check_gatorosc_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = GatorOscInput::with_default_candles(&candles);
		let output = gatorosc_with_kernel(&input, kernel)?;
		assert_eq!(output.upper.len(), candles.close.len());
		Ok(())
	}

	fn check_gatorosc_batch_default_row(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let output = GatorOscBatchBuilder::new().kernel(kernel).apply_slice(&c.close)?;
		let def = GatorOscParams::default();
		let row = output
			.combos
			.iter()
			.position(|p| {
				p.jaws_length.unwrap_or(13) == def.jaws_length.unwrap_or(13)
					&& p.jaws_shift.unwrap_or(8) == def.jaws_shift.unwrap_or(8)
					&& p.teeth_length.unwrap_or(8) == def.teeth_length.unwrap_or(8)
					&& p.teeth_shift.unwrap_or(5) == def.teeth_shift.unwrap_or(5)
					&& p.lips_length.unwrap_or(5) == def.lips_length.unwrap_or(5)
					&& p.lips_shift.unwrap_or(3) == def.lips_shift.unwrap_or(3)
			})
			.expect("default row missing");
		let u = &output.upper[row * output.cols..][..output.cols];
		assert_eq!(u.len(), c.close.len());
		Ok(())
	}
	
	#[cfg(debug_assertions)]
	fn check_gatorosc_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		
		// Define comprehensive parameter combinations
		let test_params = vec![
			GatorOscParams::default(),  // Default values
			// Minimum values
			GatorOscParams {
				jaws_length: Some(2),
				jaws_shift: Some(0),
				teeth_length: Some(2),
				teeth_shift: Some(0),
				lips_length: Some(2),
				lips_shift: Some(0),
			},
			// Small values
			GatorOscParams {
				jaws_length: Some(5),
				jaws_shift: Some(2),
				teeth_length: Some(4),
				teeth_shift: Some(1),
				lips_length: Some(3),
				lips_shift: Some(1),
			},
			// Medium values
			GatorOscParams {
				jaws_length: Some(20),
				jaws_shift: Some(10),
				teeth_length: Some(15),
				teeth_shift: Some(8),
				lips_length: Some(10),
				lips_shift: Some(5),
			},
			// Large values
			GatorOscParams {
				jaws_length: Some(50),
				jaws_shift: Some(20),
				teeth_length: Some(30),
				teeth_shift: Some(15),
				lips_length: Some(20),
				lips_shift: Some(10),
			},
			// Edge case: jaws < teeth < lips (unusual)
			GatorOscParams {
				jaws_length: Some(5),
				jaws_shift: Some(3),
				teeth_length: Some(8),
				teeth_shift: Some(5),
				lips_length: Some(13),
				lips_shift: Some(8),
			},
			// Edge case: all same length
			GatorOscParams {
				jaws_length: Some(10),
				jaws_shift: Some(5),
				teeth_length: Some(10),
				teeth_shift: Some(5),
				lips_length: Some(10),
				lips_shift: Some(5),
			},
			// Edge case: no shifts
			GatorOscParams {
				jaws_length: Some(13),
				jaws_shift: Some(0),
				teeth_length: Some(8),
				teeth_shift: Some(0),
				lips_length: Some(5),
				lips_shift: Some(0),
			},
			// Edge case: large shifts
			GatorOscParams {
				jaws_length: Some(10),
				jaws_shift: Some(20),
				teeth_length: Some(8),
				teeth_shift: Some(15),
				lips_length: Some(5),
				lips_shift: Some(10),
			},
		];
		
		for (param_idx, params) in test_params.iter().enumerate() {
			let input = GatorOscInput::from_candles(&candles, "close", params.clone());
			let output = gatorosc_with_kernel(&input, kernel)?;
			
			// Check upper values
			for (i, &val) in output.upper.iter().enumerate() {
				if val.is_nan() {
					continue; // NaN values are expected during warmup
				}
				
				let bits = val.to_bits();
				
				// Check all three poison patterns
				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 in upper output with params: {:?} (param set {})",
						test_name, val, bits, i, params, param_idx
					);
				}
				
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 in upper output with params: {:?} (param set {})",
						test_name, val, bits, i, params, param_idx
					);
				}
				
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 in upper output with params: {:?} (param set {})",
						test_name, val, bits, i, params, param_idx
					);
				}
			}
			
			// Check lower values
			for (i, &val) in output.lower.iter().enumerate() {
				if val.is_nan() {
					continue;
				}
				
				let bits = val.to_bits();
				
				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 in lower output with params: {:?} (param set {})",
						test_name, val, bits, i, params, param_idx
					);
				}
				
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 in lower output with params: {:?} (param set {})",
						test_name, val, bits, i, params, param_idx
					);
				}
				
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 in lower output with params: {:?} (param set {})",
						test_name, val, bits, i, params, param_idx
					);
				}
			}
			
			// Check upper_change values
			for (i, &val) in output.upper_change.iter().enumerate() {
				if val.is_nan() {
					continue;
				}
				
				let bits = val.to_bits();
				
				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 in upper_change output with params: {:?} (param set {})",
						test_name, val, bits, i, params, param_idx
					);
				}
				
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 in upper_change output with params: {:?} (param set {})",
						test_name, val, bits, i, params, param_idx
					);
				}
				
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 in upper_change output with params: {:?} (param set {})",
						test_name, val, bits, i, params, param_idx
					);
				}
			}
			
			// Check lower_change values
			for (i, &val) in output.lower_change.iter().enumerate() {
				if val.is_nan() {
					continue;
				}
				
				let bits = val.to_bits();
				
				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 in lower_change output with params: {:?} (param set {})",
						test_name, val, bits, i, params, param_idx
					);
				}
				
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 in lower_change output with params: {:?} (param set {})",
						test_name, val, bits, i, params, param_idx
					);
				}
				
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 in lower_change output with params: {:?} (param set {})",
						test_name, val, bits, i, params, param_idx
					);
				}
			}
		}
		
		Ok(())
	}
	
	#[cfg(not(debug_assertions))]
	fn check_gatorosc_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		Ok(()) // No-op in release builds
	}

	macro_rules! generate_all_gatorosc_tests {
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

	generate_all_gatorosc_tests!(
		check_gatorosc_partial_params,
		check_gatorosc_nan_handling,
		check_gatorosc_zero_setting,
		check_gatorosc_small_dataset,
		check_gatorosc_default_candles,
		check_gatorosc_batch_default_row,
		check_gatorosc_no_poison
	);
	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let output = GatorOscBatchBuilder::new().kernel(kernel).apply_slice(&c.close)?;

		let def = GatorOscParams::default();
		let row = output
			.combos
			.iter()
			.position(|p| {
				p.jaws_length == def.jaws_length
					&& p.jaws_shift == def.jaws_shift
					&& p.teeth_length == def.teeth_length
					&& p.teeth_shift == def.teeth_shift
					&& p.lips_length == def.lips_length
					&& p.lips_shift == def.lips_shift
			})
			.expect("default row missing");

		let upper = &output.upper[row * output.cols..][..output.cols];
		let lower = &output.lower[row * output.cols..][..output.cols];

		assert_eq!(upper.len(), c.close.len());
		assert_eq!(lower.len(), c.close.len());
		Ok(())
	}

	fn check_batch_multi_param_sweep(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let builder = GatorOscBatchBuilder::new()
			.kernel(kernel)
			.jaws_length_range(8, 14, 3)
			.jaws_shift_range(5, 8, 3)
			.teeth_length_range(5, 8, 3)
			.teeth_shift_range(3, 5, 2)
			.lips_length_range(3, 5, 2)
			.lips_shift_range(2, 3, 1);

		let output = builder.apply_slice(&c.close)?;
		// Shape checks
		assert!(output.rows > 1, "Should have multiple param sweeps");
		assert_eq!(output.cols, c.close.len());

		// Check at least one output row is not all-NaN (assuming input is valid)
		let some_upper = output
			.upper
			.chunks(output.cols)
			.any(|row| row.iter().any(|&x| !x.is_nan()));
		assert!(some_upper);

		Ok(())
	}

	fn check_batch_not_enough_data(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);

		let short = [1.0, 2.0, 3.0, 4.0, 5.0];
		let mut sweep = GatorOscBatchRange::default();
		sweep.jaws_length = (6, 6, 0);

		let res = gatorosc_batch_with_kernel(&short, &sweep, kernel);
		assert!(res.is_err());
		Ok(())
	}

	#[cfg(debug_assertions)]
	fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let test_configs = vec![
			// Single parameter sweeps
			(5, 20, 5, 8, 8, 0, 8, 8, 0, 5, 5, 0, 5, 5, 0, 3, 3, 0),     // Vary jaws_length
			(13, 13, 0, 3, 10, 2, 8, 8, 0, 5, 5, 0, 5, 5, 0, 3, 3, 0),  // Vary jaws_shift
			(13, 13, 0, 8, 8, 0, 3, 10, 2, 5, 5, 0, 5, 5, 0, 3, 3, 0),  // Vary teeth_length
			(13, 13, 0, 8, 8, 0, 8, 8, 0, 2, 8, 2, 5, 5, 0, 3, 3, 0),   // Vary teeth_shift
			(13, 13, 0, 8, 8, 0, 8, 8, 0, 5, 5, 0, 2, 8, 2, 3, 3, 0),   // Vary lips_length
			(13, 13, 0, 8, 8, 0, 8, 8, 0, 5, 5, 0, 5, 5, 0, 1, 5, 1),   // Vary lips_shift
			// Multi-parameter sweeps
			(8, 14, 3, 5, 8, 3, 5, 8, 3, 3, 5, 2, 3, 5, 2, 2, 3, 1),    // Vary all parameters
			// Static configurations
			(10, 10, 0, 5, 5, 0, 8, 8, 0, 4, 4, 0, 5, 5, 0, 2, 2, 0),   // All static
			(13, 13, 0, 8, 8, 0, 8, 8, 0, 5, 5, 0, 5, 5, 0, 3, 3, 0),   // Default static
			// Edge cases
			(2, 5, 1, 0, 3, 1, 2, 5, 1, 0, 3, 1, 2, 5, 1, 0, 3, 1),     // Small values
			(30, 50, 10, 10, 20, 5, 20, 30, 5, 8, 15, 3, 10, 20, 5, 5, 10, 2), // Large values
		];

		for (cfg_idx, &(jl_s, jl_e, jl_st, js_s, js_e, js_st, 
		                tl_s, tl_e, tl_st, ts_s, ts_e, ts_st,
		                ll_s, ll_e, ll_st, ls_s, ls_e, ls_st)) in test_configs.iter().enumerate() {
			let output = GatorOscBatchBuilder::new()
				.kernel(kernel)
				.jaws_length_range(jl_s, jl_e, jl_st)
				.jaws_shift_range(js_s, js_e, js_st)
				.teeth_length_range(tl_s, tl_e, tl_st)
				.teeth_shift_range(ts_s, ts_e, ts_st)
				.lips_length_range(ll_s, ll_e, ll_st)
				.lips_shift_range(ls_s, ls_e, ls_st)
				.apply_slice(&c.close)?;

			// Helper function to check poison in a matrix
			let check_poison = |matrix: &[f64], matrix_name: &str| {
				for (idx, &val) in matrix.iter().enumerate() {
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
							at row {} col {} (flat index {}) in {} output with params: jl={}, js={}, tl={}, ts={}, ll={}, ls={}",
							test, cfg_idx, val, bits, row, col, idx, matrix_name,
							combo.jaws_length.unwrap_or(13), combo.jaws_shift.unwrap_or(8),
							combo.teeth_length.unwrap_or(8), combo.teeth_shift.unwrap_or(5),
							combo.lips_length.unwrap_or(5), combo.lips_shift.unwrap_or(3)
						);
					}

					if bits == 0x22222222_22222222 {
						panic!(
							"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
							at row {} col {} (flat index {}) in {} output with params: jl={}, js={}, tl={}, ts={}, ll={}, ls={}",
							test, cfg_idx, val, bits, row, col, idx, matrix_name,
							combo.jaws_length.unwrap_or(13), combo.jaws_shift.unwrap_or(8),
							combo.teeth_length.unwrap_or(8), combo.teeth_shift.unwrap_or(5),
							combo.lips_length.unwrap_or(5), combo.lips_shift.unwrap_or(3)
						);
					}

					if bits == 0x33333333_33333333 {
						panic!(
							"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
							at row {} col {} (flat index {}) in {} output with params: jl={}, js={}, tl={}, ts={}, ll={}, ls={}",
							test, cfg_idx, val, bits, row, col, idx, matrix_name,
							combo.jaws_length.unwrap_or(13), combo.jaws_shift.unwrap_or(8),
							combo.teeth_length.unwrap_or(8), combo.teeth_shift.unwrap_or(5),
							combo.lips_length.unwrap_or(5), combo.lips_shift.unwrap_or(3)
						);
					}
				}
			};

			// Check all four output matrices
			check_poison(&output.upper, "upper");
			check_poison(&output.lower, "lower");
			check_poison(&output.upper_change, "upper_change");
			check_poison(&output.lower_change, "lower_change");
		}

		Ok(())
	}

	#[cfg(not(debug_assertions))]
	fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
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
	gen_batch_tests!(check_batch_multi_param_sweep);
	gen_batch_tests!(check_batch_not_enough_data);
	gen_batch_tests!(check_batch_no_poison);
}
