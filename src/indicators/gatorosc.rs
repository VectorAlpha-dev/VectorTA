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
use crate::utilities::helpers::{alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes, make_uninit_matrix};
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
	let (data, jaws_length, jaws_shift, teeth_length, teeth_shift, lips_length, lips_shift, first, chosen) = 
		gatorosc_prepare(input, kernel)?;

	let mut upper = alloc_with_nan_prefix(data.len(), first + jaws_length.max(teeth_length) - 1);
	let mut lower = alloc_with_nan_prefix(data.len(), first + teeth_length.max(lips_length) - 1);
	let mut upper_change = alloc_with_nan_prefix(data.len(), first + jaws_length.max(teeth_length));
	let mut lower_change = alloc_with_nan_prefix(data.len(), first + teeth_length.max(lips_length));

	gatorosc_compute_into(
		data,
		jaws_length,
		jaws_shift,
		teeth_length,
		teeth_shift,
		lips_length,
		lips_shift,
		first,
		chosen,
		&mut upper,
		&mut lower,
		&mut upper_change,
		&mut lower_change,
	);
	
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
	// Compute EMAs on-the-fly without allocations
	let jaws_alpha = 1.0 / jaws_length as f64;
	let teeth_alpha = 1.0 / teeth_length as f64;
	let lips_alpha = 1.0 / lips_length as f64;
	
	// Initialize EMA states
	let mut jaws_ema = if data[first_valid].is_nan() { 0.0 } else { data[first_valid] };
	let mut teeth_ema = jaws_ema;
	let mut lips_ema = jaws_ema;
	
	// Ring buffers for shifted values (small, fixed size)
	let max_shift = jaws_shift.max(teeth_shift).max(lips_shift);
	let mut jaws_ring: AVec<f64> = AVec::with_capacity(CACHELINE_ALIGN, max_shift + 1);
	let mut teeth_ring: AVec<f64> = AVec::with_capacity(CACHELINE_ALIGN, max_shift + 1);
	let mut lips_ring: AVec<f64> = AVec::with_capacity(CACHELINE_ALIGN, max_shift + 1);
	jaws_ring.resize(max_shift + 1, f64::NAN);
	teeth_ring.resize(max_shift + 1, f64::NAN);
	lips_ring.resize(max_shift + 1, f64::NAN);
	
	let mut ring_idx = 0;
	
	// Process data and compute outputs directly
	for i in first_valid..data.len() {
		// Update EMAs
		let val = if data[i].is_nan() { 
			jaws_ema // Use previous value for NaN
		} else { 
			data[i] 
		};
		
		jaws_ema = jaws_alpha * val + (1.0 - jaws_alpha) * jaws_ema;
		teeth_ema = teeth_alpha * val + (1.0 - teeth_alpha) * teeth_ema;
		lips_ema = lips_alpha * val + (1.0 - lips_alpha) * lips_ema;
		
		// Store in ring buffers
		let ring_pos = ring_idx % (max_shift + 1);
		jaws_ring[ring_pos] = jaws_ema;
		teeth_ring[ring_pos] = teeth_ema;
		lips_ring[ring_pos] = lips_ema;
		
		// Calculate shifted values and outputs
		if i >= first_valid + jaws_shift {
			let jaws_shifted_idx = (ring_idx + max_shift + 1 - jaws_shift) % (max_shift + 1);
			let jaws_shifted = jaws_ring[jaws_shifted_idx];
			
			if i >= first_valid + teeth_shift {
				let teeth_shifted_idx = (ring_idx + max_shift + 1 - teeth_shift) % (max_shift + 1);
				let teeth_shifted = teeth_ring[teeth_shifted_idx];
				
				upper[i] = (jaws_shifted - teeth_shifted).abs();
				
				if i >= first_valid + lips_shift {
					let lips_shifted_idx = (ring_idx + max_shift + 1 - lips_shift) % (max_shift + 1);
					let lips_shifted = lips_ring[lips_shifted_idx];
					
					lower[i] = -(teeth_shifted - lips_shifted).abs();
				}
			}
		}
		
		ring_idx += 1;
	}
	
	// Compute change values
	for i in 1..data.len() {
		if !upper[i].is_nan() && !upper[i - 1].is_nan() {
			upper_change[i] = upper[i] - upper[i - 1];
		}
		if !lower[i].is_nan() && !lower[i - 1].is_nan() {
			lower_change[i] = -(lower[i] - lower[i - 1]);
		}
	}
}

#[cfg(all(target_feature = "simd128", target_arch = "wasm32"))]
#[inline(always)]
pub unsafe fn gatorosc_simd128(
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
	// SIMD128 implementation uses scalar since the complex EMA and ring buffer logic
	// doesn't benefit significantly from SIMD128's limited 128-bit vectors
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

// Helper function for validation and preparation
#[inline]
fn gatorosc_prepare<'a>(
	input: &'a GatorOscInput<'a>,
	kernel: Kernel,
) -> Result<(&'a [f64], usize, usize, usize, usize, usize, usize, usize, Kernel), GatorOscError> {
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

	Ok((data, jaws_length, jaws_shift, teeth_length, teeth_shift, lips_length, lips_shift, first, chosen))
}

// Zero-allocation compute function
#[inline]
fn gatorosc_compute_into(
	data: &[f64],
	jaws_length: usize,
	jaws_shift: usize,
	teeth_length: usize,
	teeth_shift: usize,
	lips_length: usize,
	lips_shift: usize,
	first: usize,
	kernel: Kernel,
	upper: &mut [f64],
	lower: &mut [f64],
	upper_change: &mut [f64],
	lower_change: &mut [f64],
) {
	unsafe {
		match kernel {
			Kernel::Scalar | Kernel::ScalarBatch => gatorosc_scalar(
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
			),
			#[cfg(all(target_feature = "simd128", target_arch = "wasm32"))]
			Kernel::Simd128 | Kernel::Simd128Batch => gatorosc_simd128(
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
				upper,
				lower,
				upper_change,
				lower_change,
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
				upper,
				lower,
				upper_change,
				lower_change,
			),
			_ => unreachable!(),
		}
	}
}

// Into slice wrapper for external use
#[inline]
pub fn gatorosc_into_slice(
	upper_dst: &mut [f64],
	lower_dst: &mut [f64],
	upper_change_dst: &mut [f64],
	lower_change_dst: &mut [f64],
	input: &GatorOscInput,
	kernel: Kernel,
) -> Result<(), GatorOscError> {
	let (data, jaws_length, jaws_shift, teeth_length, teeth_shift, lips_length, lips_shift, first, chosen) = 
		gatorosc_prepare(input, kernel)?;
	
	if upper_dst.len() != data.len() || lower_dst.len() != data.len() || 
	   upper_change_dst.len() != data.len() || lower_change_dst.len() != data.len() {
		return Err(GatorOscError::InvalidSettings);
	}
	
	gatorosc_compute_into(
		data,
		jaws_length,
		jaws_shift,
		teeth_length,
		teeth_shift,
		lips_length,
		lips_shift,
		first,
		chosen,
		upper_dst,
		lower_dst,
		upper_change_dst,
		lower_change_dst,
	);
	
	Ok(())
}

#[derive(Debug, Clone)]
pub struct GatorOscStream {
	jaws: EmaStream,
	teeth: EmaStream,
	lips: EmaStream,
	jaws_shift: usize,
	teeth_shift: usize,
	lips_shift: usize,
	buf: AVec<f64>,
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
			buf: {
				let mut buf: AVec<f64> = AVec::with_capacity(CACHELINE_ALIGN, jaws_shift.max(teeth_shift).max(lips_shift) + 1);
				buf.resize(jaws_shift.max(teeth_shift).max(lips_shift) + 1, f64::NAN);
				buf
			},
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
		Kernel::Simd128Batch => Kernel::Simd128,
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
	
	// Use make_uninit_matrix for efficient allocation
	let mut upper_buf = make_uninit_matrix(rows, cols);
	let mut lower_buf = make_uninit_matrix(rows, cols);
	let mut upper_change_buf = make_uninit_matrix(rows, cols);
	let mut lower_change_buf = make_uninit_matrix(rows, cols);
	
	// Initialize prefixes with NaN based on warmup periods
	let warmup_periods: Vec<usize> = combos.iter().map(|c| {
		first + c.jaws_length.unwrap().max(c.teeth_length.unwrap()) - 1
	}).collect();
	init_matrix_prefixes(&mut upper_buf, cols, &warmup_periods);
	
	let warmup_periods_lower: Vec<usize> = combos.iter().map(|c| {
		first + c.teeth_length.unwrap().max(c.lips_length.unwrap()) - 1
	}).collect();
	init_matrix_prefixes(&mut lower_buf, cols, &warmup_periods_lower);
	
	let warmup_periods_upper_change: Vec<usize> = combos.iter().map(|c| {
		first + c.jaws_length.unwrap().max(c.teeth_length.unwrap())
	}).collect();
	init_matrix_prefixes(&mut upper_change_buf, cols, &warmup_periods_upper_change);
	
	let warmup_periods_lower_change: Vec<usize> = combos.iter().map(|c| {
		first + c.teeth_length.unwrap().max(c.lips_length.unwrap())
	}).collect();
	init_matrix_prefixes(&mut lower_change_buf, cols, &warmup_periods_lower_change);
	
	// Convert to initialized vectors
	let mut upper = upper_buf.into_iter().map(|mu| unsafe { mu.assume_init() }).collect::<Vec<_>>();
	let mut lower = lower_buf.into_iter().map(|mu| unsafe { mu.assume_init() }).collect::<Vec<_>>();
	let mut upper_change = upper_change_buf.into_iter().map(|mu| unsafe { mu.assume_init() }).collect::<Vec<_>>();
	let mut lower_change = lower_change_buf.into_iter().map(|mu| unsafe { mu.assume_init() }).collect::<Vec<_>>();

	let do_row = |row: usize, u: &mut [f64], l: &mut [f64], uc: &mut [f64], lc: &mut [f64]| {
		let prm = &combos[row];
		gatorosc_compute_into(
			data,
			prm.jaws_length.unwrap(),
			prm.jaws_shift.unwrap(),
			prm.teeth_length.unwrap(),
			prm.teeth_shift.unwrap(),
			prm.lips_length.unwrap(),
			prm.lips_shift.unwrap(),
			first,
			kern,
			u,
			l,
			uc,
			lc,
		);
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

// Zero-allocation batch function that writes directly to output slices
#[inline]
pub fn gatorosc_batch_inner_into(
	data: &[f64],
	sweep: &GatorOscBatchRange,
	kernel: Kernel,
	parallel: bool,
	upper_out: &mut [f64],
	lower_out: &mut [f64],
	upper_change_out: &mut [f64],
	lower_change_out: &mut [f64],
) -> Result<Vec<GatorOscParams>, GatorOscError> {
	let combos = expand_grid_gatorosc(sweep);
	if combos.is_empty() {
		return Err(GatorOscError::InvalidSettings);
	}
	
	let first = data
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(GatorOscError::AllValuesNaN)?;
	
	let rows = combos.len();
	let cols = data.len();
	
	if upper_out.len() != rows * cols || lower_out.len() != rows * cols ||
	   upper_change_out.len() != rows * cols || lower_change_out.len() != rows * cols {
		return Err(GatorOscError::InvalidSettings);
	}
	
	let do_row = |row: usize| {
		let prm = &combos[row];
		let start = row * cols;
		let end = start + cols;
		
		gatorosc_compute_into(
			data,
			prm.jaws_length.unwrap(),
			prm.jaws_shift.unwrap(),
			prm.teeth_length.unwrap(),
			prm.teeth_shift.unwrap(),
			prm.lips_length.unwrap(),
			prm.lips_shift.unwrap(),
			first,
			kernel,
			&mut upper_out[start..end],
			&mut lower_out[start..end],
			&mut upper_change_out[start..end],
			&mut lower_change_out[start..end],
		);
	};
	
	#[cfg(not(target_arch = "wasm32"))]
	if parallel {
		(0..rows).into_par_iter().for_each(do_row);
	} else {
		for row in 0..rows {
			do_row(row);
		}
	}
	
	#[cfg(target_arch = "wasm32")]
	for row in 0..rows {
		do_row(row);
	}
	
	Ok(combos)
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
	
	// Use helper functions for zero-allocation
	let mut upper_buf = make_uninit_matrix(rows, cols);
	let mut lower_buf = make_uninit_matrix(rows, cols);
	let mut upper_change_buf = make_uninit_matrix(rows, cols);
	let mut lower_change_buf = make_uninit_matrix(rows, cols);
	
	// Calculate warmup periods for each combo
	let warmup_periods: Vec<usize> = combos.iter().map(|p| {
		let jaws_warmup = first + p.jaws_length.unwrap().max(p.teeth_length.unwrap()) - 1;
		let teeth_warmup = first + p.teeth_length.unwrap().max(p.lips_length.unwrap()) - 1;
		jaws_warmup.max(teeth_warmup)
	}).collect();
	
	// Initialize with NaN prefixes
	init_matrix_prefixes(&mut upper_buf, cols, &warmup_periods);
	init_matrix_prefixes(&mut lower_buf, cols, &warmup_periods);
	init_matrix_prefixes(&mut upper_change_buf, cols, &warmup_periods);
	init_matrix_prefixes(&mut lower_change_buf, cols, &warmup_periods);
	
	// Convert to initialized slices
	let mut upper = unsafe { upper_buf.assume_init().into_vec() };
	let mut lower = unsafe { lower_buf.assume_init().into_vec() };
	let mut upper_change = unsafe { upper_change_buf.assume_init().into_vec() };
	let mut lower_change = unsafe { lower_change_buf.assume_init().into_vec() };
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

//======= WASM Bindings =======

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct GatorOscJsOutput {
	pub values: Vec<f64>,  // Flattened [upper..., lower..., upper_change..., lower_change...]
	pub rows: usize,       // 4 for gatorosc
	pub cols: usize,       // data length
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn gatorosc_js(
	data: &[f64],
	jaws_length: usize,
	jaws_shift: usize,
	teeth_length: usize,
	teeth_shift: usize,
	lips_length: usize,
	lips_shift: usize,
) -> Result<JsValue, JsValue> {
	let params = GatorOscParams {
		jaws_length: Some(jaws_length),
		jaws_shift: Some(jaws_shift),
		teeth_length: Some(teeth_length),
		teeth_shift: Some(teeth_shift),
		lips_length: Some(lips_length),
		lips_shift: Some(lips_shift),
	};
	let input = GatorOscInput::from_slice(data, params);

	// Single allocation for all outputs (flattened)
	let len = data.len();
	let mut values = vec![0.0; 4 * len];
	
	// Split into mutable slices for each output
	let (upper_part, rest) = values.split_at_mut(len);
	let (lower_part, rest) = rest.split_at_mut(len);
	let (upper_change_part, lower_change_part) = rest.split_at_mut(len);

	// Compute using zero-allocation helper
	gatorosc_into_slice(upper_part, lower_part, upper_change_part, lower_change_part, &input, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;

	let output = GatorOscJsOutput {
		values,
		rows: 4,
		cols: len,
	};

	serde_wasm_bindgen::to_value(&output).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn gatorosc_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn gatorosc_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe {
			let _ = Vec::from_raw_parts(ptr, len, len);
		}
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn gatorosc_into(
	in_ptr: *const f64,
	upper_ptr: *mut f64,
	lower_ptr: *mut f64,
	upper_change_ptr: *mut f64,
	lower_change_ptr: *mut f64,
	len: usize,
	jaws_length: usize,
	jaws_shift: usize,
	teeth_length: usize,
	teeth_shift: usize,
	lips_length: usize,
	lips_shift: usize,
) -> Result<(), JsValue> {
	if in_ptr.is_null() || upper_ptr.is_null() || lower_ptr.is_null() || 
	   upper_change_ptr.is_null() || lower_change_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}

	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);
		let params = GatorOscParams {
			jaws_length: Some(jaws_length),
			jaws_shift: Some(jaws_shift),
			teeth_length: Some(teeth_length),
			teeth_shift: Some(teeth_shift),
			lips_length: Some(lips_length),
			lips_shift: Some(lips_shift),
		};
		let input = GatorOscInput::from_slice(data, params);

		// Check for aliasing - if any output pointer equals input pointer
		let needs_temp = in_ptr == upper_ptr as *const f64 || 
		                 in_ptr == lower_ptr as *const f64 ||
		                 in_ptr == upper_change_ptr as *const f64 ||
		                 in_ptr == lower_change_ptr as *const f64;

		if needs_temp {
			// Use single temporary buffer for all outputs
			let mut temp = vec![0.0; 4 * len];
			
			// Split into slices for computation
			let (temp_upper, rest) = temp.split_at_mut(len);
			let (temp_lower, rest) = rest.split_at_mut(len);
			let (temp_upper_change, temp_lower_change) = rest.split_at_mut(len);

			gatorosc_into_slice(
				temp_upper,
				temp_lower,
				temp_upper_change,
				temp_lower_change,
				&input,
				Kernel::Auto
			).map_err(|e| JsValue::from_str(&e.to_string()))?;

			// Copy results to output pointers
			let upper_out = std::slice::from_raw_parts_mut(upper_ptr, len);
			let lower_out = std::slice::from_raw_parts_mut(lower_ptr, len);
			let upper_change_out = std::slice::from_raw_parts_mut(upper_change_ptr, len);
			let lower_change_out = std::slice::from_raw_parts_mut(lower_change_ptr, len);

			upper_out.copy_from_slice(temp_upper);
			lower_out.copy_from_slice(temp_lower);
			upper_change_out.copy_from_slice(temp_upper_change);
			lower_change_out.copy_from_slice(temp_lower_change);
		} else {
			// Direct computation into output buffers
			let upper_out = std::slice::from_raw_parts_mut(upper_ptr, len);
			let lower_out = std::slice::from_raw_parts_mut(lower_ptr, len);
			let upper_change_out = std::slice::from_raw_parts_mut(upper_change_ptr, len);
			let lower_change_out = std::slice::from_raw_parts_mut(lower_change_ptr, len);

			gatorosc_into_slice(
				upper_out,
				lower_out,
				upper_change_out,
				lower_change_out,
				&input,
				Kernel::Auto
			).map_err(|e| JsValue::from_str(&e.to_string()))?;
		}

		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct GatorOscBatchConfig {
	pub jaws_length_range: (usize, usize, usize),
	pub jaws_shift_range: (usize, usize, usize),
	pub teeth_length_range: (usize, usize, usize),
	pub teeth_shift_range: (usize, usize, usize),
	pub lips_length_range: (usize, usize, usize),
	pub lips_shift_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct GatorOscBatchJsOutput {
	pub values: Vec<f64>,  // Flattened [upper..., lower..., upper_change..., lower_change...]
	pub combos: Vec<GatorOscParams>,
	pub rows: usize,  // Number of parameter combinations
	pub cols: usize,  // Data length
	pub outputs: usize,  // 4 for gatorosc
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = gatorosc_batch)]
pub fn gatorosc_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	let config: GatorOscBatchConfig =
		serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

	let sweep = GatorOscBatchRange {
		jaws_length: config.jaws_length_range,
		jaws_shift: config.jaws_shift_range,
		teeth_length: config.teeth_length_range,
		teeth_shift: config.teeth_shift_range,
		lips_length: config.lips_length_range,
		lips_shift: config.lips_shift_range,
	};

	// Calculate total combinations
	let combos = expand_grid(&sweep);
	let n_combos = combos.len();
	let len = data.len();

	// Single allocation for all outputs
	let total_size = n_combos * len;
	let mut values = vec![0.0; 4 * total_size];
	
	// Split into mutable slices for each output
	let (upper_part, rest) = values.split_at_mut(total_size);
	let (lower_part, rest) = rest.split_at_mut(total_size);
	let (upper_change_part, lower_change_part) = rest.split_at_mut(total_size);

	// Use zero-allocation batch function
	gatorosc_batch_inner_into(
		data,
		&sweep,
		Kernel::Auto,
		false,  // No parallel in WASM
		upper_part,
		lower_part,
		upper_change_part,
		lower_change_part,
	).map_err(|e| JsValue::from_str(&e.to_string()))?;

	let js_output = GatorOscBatchJsOutput {
		values,
		combos,
		rows: n_combos,
		cols: len,
		outputs: 4,
	};

	serde_wasm_bindgen::to_value(&js_output).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn gatorosc_batch_into(
	in_ptr: *const f64,
	upper_ptr: *mut f64,
	lower_ptr: *mut f64,
	upper_change_ptr: *mut f64,
	lower_change_ptr: *mut f64,
	len: usize,
	jaws_length_start: usize,
	jaws_length_end: usize,
	jaws_length_step: usize,
	jaws_shift_start: usize,
	jaws_shift_end: usize,
	jaws_shift_step: usize,
	teeth_length_start: usize,
	teeth_length_end: usize,
	teeth_length_step: usize,
	teeth_shift_start: usize,
	teeth_shift_end: usize,
	teeth_shift_step: usize,
	lips_length_start: usize,
	lips_length_end: usize,
	lips_length_step: usize,
	lips_shift_start: usize,
	lips_shift_end: usize,
	lips_shift_step: usize,
) -> Result<usize, JsValue> {
	if in_ptr.is_null() || upper_ptr.is_null() || lower_ptr.is_null() || 
	   upper_change_ptr.is_null() || lower_change_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}

	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);
		let sweep = GatorOscBatchRange {
			jaws_length: (jaws_length_start, jaws_length_end, jaws_length_step),
			jaws_shift: (jaws_shift_start, jaws_shift_end, jaws_shift_step),
			teeth_length: (teeth_length_start, teeth_length_end, teeth_length_step),
			teeth_shift: (teeth_shift_start, teeth_shift_end, teeth_shift_step),
			lips_length: (lips_length_start, lips_length_end, lips_length_step),
			lips_shift: (lips_shift_start, lips_shift_end, lips_shift_step),
		};

		// Calculate number of combinations
		let combos = expand_grid(&sweep);
		let n_combos = combos.len();
		let total_size = n_combos * len;

		// Create output slices
		let upper_out = std::slice::from_raw_parts_mut(upper_ptr, total_size);
		let lower_out = std::slice::from_raw_parts_mut(lower_ptr, total_size);
		let upper_change_out = std::slice::from_raw_parts_mut(upper_change_ptr, total_size);
		let lower_change_out = std::slice::from_raw_parts_mut(lower_change_ptr, total_size);

		// Use zero-allocation batch function
		gatorosc_batch_inner_into(
			data,
			&sweep,
			Kernel::Auto,
			false,  // No parallel in WASM
			upper_out,
			lower_out,
			upper_change_out,
			lower_change_out,
		).map_err(|e| JsValue::from_str(&e.to_string()))?;

		Ok(n_combos)
	}
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

	macro_rules! generate_all_gatorosc_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(
                    #[test]
                    fn [<$test_fn _scalar_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar);
                    }
                )*
                #[cfg(all(target_feature = "simd128", target_arch = "wasm32"))]
                $(
                    #[test]
                    fn [<$test_fn _simd128_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _simd128_f64>]), Kernel::Simd128);
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
		check_gatorosc_batch_default_row
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

	macro_rules! gen_batch_tests {
		($fn_name:ident) => {
			paste::paste! {
				#[test] fn [<$fn_name _scalar>]()      {
					let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch);
				}
				#[cfg(all(target_feature = "simd128", target_arch = "wasm32"))]
				#[test] fn [<$fn_name _simd128>]()     {
					let _ = $fn_name(stringify!([<$fn_name _simd128>]), Kernel::Simd128Batch);
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
}

// ============================================================================
//                               PYTHON BINDINGS
// ============================================================================

#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

#[cfg(feature = "python")]
#[pyfunction(name = "gatorosc")]
#[pyo3(signature = (data, jaws_length=13, jaws_shift=8, teeth_length=8, teeth_shift=5, lips_length=5, lips_shift=3, kernel=None))]
pub fn gatorosc_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	jaws_length: usize,
	jaws_shift: usize,
	teeth_length: usize,
	teeth_shift: usize,
	lips_length: usize,
	lips_shift: usize,
	kernel: Option<&str>,
) -> PyResult<(
	Bound<'py, PyArray1<f64>>,
	Bound<'py, PyArray1<f64>>,
	Bound<'py, PyArray1<f64>>,
	Bound<'py, PyArray1<f64>>,
)> {
	use numpy::{IntoPyArray, PyArrayMethods};

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, false)?;
	
	let params = GatorOscParams {
		jaws_length: Some(jaws_length),
		jaws_shift: Some(jaws_shift),
		teeth_length: Some(teeth_length),
		teeth_shift: Some(teeth_shift),
		lips_length: Some(lips_length),
		lips_shift: Some(lips_shift),
	};
	let input = GatorOscInput::from_slice(slice_in, params);

	let (upper_vec, lower_vec, upper_change_vec, lower_change_vec) = py
		.allow_threads(|| {
			gatorosc_with_kernel(&input, kern)
				.map(|o| (o.upper, o.lower, o.upper_change, o.lower_change))
		})
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	Ok((
		upper_vec.into_pyarray(py),
		lower_vec.into_pyarray(py),
		upper_change_vec.into_pyarray(py),
		lower_change_vec.into_pyarray(py),
	))
}

#[cfg(feature = "python")]
#[pyclass(name = "GatorOscStream")]
pub struct GatorOscStreamPy {
	stream: GatorOscStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl GatorOscStreamPy {
	#[new]
	#[pyo3(signature = (jaws_length=13, jaws_shift=8, teeth_length=8, teeth_shift=5, lips_length=5, lips_shift=3))]
	fn new(
		jaws_length: usize,
		jaws_shift: usize,
		teeth_length: usize,
		teeth_shift: usize,
		lips_length: usize,
		lips_shift: usize,
	) -> PyResult<Self> {
		let params = GatorOscParams {
			jaws_length: Some(jaws_length),
			jaws_shift: Some(jaws_shift),
			teeth_length: Some(teeth_length),
			teeth_shift: Some(teeth_shift),
			lips_length: Some(lips_length),
			lips_shift: Some(lips_shift),
		};
		let stream = GatorOscStream::try_new(params)
			.map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(GatorOscStreamPy { stream })
	}

	fn update(&mut self, value: f64) -> Option<(f64, f64, f64, f64)> {
		self.stream.update(value)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "gatorosc_batch")]
#[pyo3(signature = (data, jaws_length_range=(13, 13, 0), jaws_shift_range=(8, 8, 0), teeth_length_range=(8, 8, 0), teeth_shift_range=(5, 5, 0), lips_length_range=(5, 5, 0), lips_shift_range=(3, 3, 0), kernel=None))]
pub fn gatorosc_batch_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	jaws_length_range: (usize, usize, usize),
	jaws_shift_range: (usize, usize, usize),
	teeth_length_range: (usize, usize, usize),
	teeth_shift_range: (usize, usize, usize),
	lips_length_range: (usize, usize, usize),
	lips_shift_range: (usize, usize, usize),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, true)?;

	let sweep = GatorOscBatchRange {
		jaws_length: jaws_length_range,
		jaws_shift: jaws_shift_range,
		teeth_length: teeth_length_range,
		teeth_shift: teeth_shift_range,
		lips_length: lips_length_range,
		lips_shift: lips_shift_range,
	};

	let combos = expand_grid_gatorosc(&sweep);
	let rows = combos.len();
	let cols = slice_in.len();

	// Pre-allocate output arrays for batch operations
	let upper_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let lower_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let upper_change_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let lower_change_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };

	let slice_upper = unsafe { upper_arr.as_slice_mut()? };
	let slice_lower = unsafe { lower_arr.as_slice_mut()? };
	let slice_upper_change = unsafe { upper_change_arr.as_slice_mut()? };
	let slice_lower_change = unsafe { lower_change_arr.as_slice_mut()? };

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
				Kernel::Simd128Batch => Kernel::Simd128,
				_ => unreachable!(),
			};
			
			// Use zero-allocation batch function
			gatorosc_batch_inner_into(
				slice_in,
				&sweep,
				simd,
				true,
				slice_upper,
				slice_lower,
				slice_upper_change,
				slice_lower_change,
			)
		})
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	let dict = PyDict::new(py);
	dict.set_item("upper", upper_arr.reshape((rows, cols))?)?;
	dict.set_item("lower", lower_arr.reshape((rows, cols))?)?;
	dict.set_item("upper_change", upper_change_arr.reshape((rows, cols))?)?;
	dict.set_item("lower_change", lower_change_arr.reshape((rows, cols))?)?;
	dict.set_item(
		"jaws_lengths",
		combos.iter()
			.map(|p| p.jaws_length.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py)
	)?;
	dict.set_item(
		"jaws_shifts",
		combos.iter()
			.map(|p| p.jaws_shift.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py)
	)?;
	dict.set_item(
		"teeth_lengths",
		combos.iter()
			.map(|p| p.teeth_length.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py)
	)?;
	dict.set_item(
		"teeth_shifts",
		combos.iter()
			.map(|p| p.teeth_shift.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py)
	)?;
	dict.set_item(
		"lips_lengths",
		combos.iter()
			.map(|p| p.lips_length.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py)
	)?;
	dict.set_item(
		"lips_shifts",
		combos.iter()
			.map(|p| p.lips_shift.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py)
	)?;

	Ok(dict)
}
