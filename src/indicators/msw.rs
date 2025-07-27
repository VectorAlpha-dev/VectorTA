//! # Mesa Sine Wave (MSW)
//!
//! The Mesa Sine Wave indicator attempts to detect turning points in price data
//! by fitting a sine wave function. It outputs two series: the `sine` wave
//! and a leading version of the wave (`lead`).
//!
//! ## Parameters
//! - **period**: The window size (number of data points). Defaults to 5.
//!
//! ## Errors
//! - **EmptyData**: msw: Input data slice is empty.
//! - **InvalidPeriod**: msw: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: msw: Fewer than `period` valid (non-`NaN`) data points remain
//!   after the first valid index.
//! - **AllValuesNaN**: msw: All input data values are `NaN`.
//!
//! ## Returns
//! - **`Ok(MswOutput)`** on success, containing two `Vec<f64>` of equal length:
//!   `sine` and `lead`, both matching the input length, with leading `NaN`s until
//!   the Mesa Sine Wave window is filled.
//! - **`Err(MswError)`** otherwise.

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
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

#[allow(clippy::approx_constant)]
const TULIP_PI: f64 = 3.1415926;
const TULIP_TPI: f64 = 2.0 * TULIP_PI;

impl<'a> AsRef<[f64]> for MswInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			MswData::Slice(slice) => slice,
			MswData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

#[derive(Debug, Clone)]
pub enum MswData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct MswOutput {
	pub sine: Vec<f64>,
	pub lead: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct MswParams {
	pub period: Option<usize>,
}

impl Default for MswParams {
	fn default() -> Self {
		Self { period: Some(5) }
	}
}

#[derive(Debug, Clone)]
pub struct MswInput<'a> {
	pub data: MswData<'a>,
	pub params: MswParams,
}

impl<'a> MswInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: MswParams) -> Self {
		Self {
			data: MswData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slice(sl: &'a [f64], p: MswParams) -> Self {
		Self {
			data: MswData::Slice(sl),
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", MswParams::default())
	}
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(5)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct MswBuilder {
	period: Option<usize>,
	kernel: Kernel,
}

impl Default for MswBuilder {
	fn default() -> Self {
		Self {
			period: None,
			kernel: Kernel::Auto,
		}
	}
}

impl MswBuilder {
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
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline(always)]
	pub fn apply(self, c: &Candles) -> Result<MswOutput, MswError> {
		let p = MswParams { period: self.period };
		let i = MswInput::from_candles(c, "close", p);
		msw_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slice(self, d: &[f64]) -> Result<MswOutput, MswError> {
		let p = MswParams { period: self.period };
		let i = MswInput::from_slice(d, p);
		msw_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<MswStream, MswError> {
		let p = MswParams { period: self.period };
		MswStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum MswError {
	#[error("msw: Empty data provided for MSW.")]
	EmptyData,
	#[error("msw: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("msw: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("msw: All values are NaN.")]
	AllValuesNaN,
}

#[inline]
pub fn msw(input: &MswInput) -> Result<MswOutput, MswError> {
	msw_with_kernel(input, Kernel::Auto)
}

pub fn msw_with_kernel(input: &MswInput, kernel: Kernel) -> Result<MswOutput, MswError> {
	let data: &[f64] = match &input.data {
		MswData::Candles { candles, source } => source_type(candles, source),
		MswData::Slice(sl) => sl,
	};
	if data.is_empty() {
		return Err(MswError::EmptyData);
	}
	let period = input.get_period();
	let first = data.iter().position(|x| !x.is_nan()).ok_or(MswError::AllValuesNaN)?;
	let len = data.len();
	if period == 0 || period > len {
		return Err(MswError::InvalidPeriod { period, data_len: len });
	}
	if (len - first) < period {
		return Err(MswError::NotEnoughValidData {
			needed: period,
			valid: len - first,
		});
	}
	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};
	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => msw_scalar(data, period, first, len),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => msw_avx2(data, period, first, len),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => msw_avx512(data, period, first, len),
			_ => unreachable!(),
		}
	}
}

#[inline]
pub unsafe fn msw_scalar(data: &[f64], period: usize, first: usize, len: usize) -> Result<MswOutput, MswError> {
	let mut sine = alloc_with_nan_prefix(len, first + period - 1);
	let mut lead = alloc_with_nan_prefix(len, first + period - 1);

	let mut cos_table: AVec<f64, aligned_vec::ConstAlign<CACHELINE_ALIGN>> =
		AVec::with_capacity(CACHELINE_ALIGN, period);
	let mut sin_table: AVec<f64, aligned_vec::ConstAlign<CACHELINE_ALIGN>> =
		AVec::with_capacity(CACHELINE_ALIGN, period);
	cos_table.resize(period, 0.0);
	sin_table.resize(period, 0.0);

	for j in 0..period {
		let angle = TULIP_TPI * j as f64 / period as f64;
		cos_table[j] = angle.cos();
		sin_table[j] = angle.sin();
	}
	for i in (first + period - 1)..len {
		let mut rp = 0.0;
		let mut ip = 0.0;
		for j in 0..period {
			let weight = data[i - j];
			rp += cos_table[j] * weight;
			ip += sin_table[j] * weight;
		}
		let mut phase = if rp.abs() > 0.001 {
			atan(ip / rp)
		} else {
			TULIP_PI * if ip < 0.0 { -1.0 } else { 1.0 }
		};
		if rp < 0.0 {
			phase += TULIP_PI;
		}
		phase += TULIP_PI / 2.0;
		if phase < 0.0 {
			phase += TULIP_TPI;
		}
		if phase > TULIP_TPI {
			phase -= TULIP_TPI;
		}
		sine[i] = phase.sin();
		lead[i] = (phase + TULIP_PI / 4.0).sin();
	}
	Ok(MswOutput { sine, lead })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn msw_avx2(data: &[f64], period: usize, first: usize, len: usize) -> Result<MswOutput, MswError> {
	msw_scalar(data, period, first, len)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn msw_avx512(data: &[f64], period: usize, first: usize, len: usize) -> Result<MswOutput, MswError> {
	msw_scalar(data, period, first, len)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn msw_avx512_short(data: &[f64], period: usize, first: usize, len: usize) -> Result<MswOutput, MswError> {
	msw_scalar(data, period, first, len)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn msw_avx512_long(data: &[f64], period: usize, first: usize, len: usize) -> Result<MswOutput, MswError> {
	msw_scalar(data, period, first, len)
}

pub fn atan(x: f64) -> f64 {
	x.atan()
}

#[derive(Debug, Clone)]
pub struct MswStream {
	period: usize,
	buffer: Vec<f64>,
	cos_table: Vec<f64>,
	sin_table: Vec<f64>,
	head: usize,
	filled: bool,
}

impl MswStream {
	pub fn try_new(params: MswParams) -> Result<Self, MswError> {
		let period = params.period.unwrap_or(5);
		if period == 0 {
			return Err(MswError::InvalidPeriod { period, data_len: 0 });
		}
		let mut cos_table = Vec::with_capacity(period);
		let mut sin_table = Vec::with_capacity(period);
		for j in 0..period {
			let angle = TULIP_TPI * j as f64 / period as f64;
			cos_table.push(angle.cos());
			sin_table.push(angle.sin());
		}
		Ok(Self {
			period,
			buffer: vec![f64::NAN; period],
			cos_table,
			sin_table,
			head: 0,
			filled: false,
		})
	}
	#[inline(always)]
	pub fn update(&mut self, value: f64) -> Option<(f64, f64)> {
		self.buffer[self.head] = value;
		self.head = (self.head + 1) % self.period;
		if !self.filled && self.head == 0 {
			self.filled = true;
		}
		if !self.filled {
			return None;
		}
		Some(self.dot_ring())
	}
	#[inline(always)]
	fn dot_ring(&self) -> (f64, f64) {
		let mut rp = 0.0;
		let mut ip = 0.0;
		// `self.head` always points to the next insertion position, which is
		// the oldest sample in the ring buffer. The most recent value is the
		// element just before `head`. The batch implementation processes data
		// from newest to oldest, so mirror that ordering here.
		let mut idx = (self.head + self.period - 1) % self.period;
		for j in 0..self.period {
			rp += self.cos_table[j] * self.buffer[idx];
			ip += self.sin_table[j] * self.buffer[idx];
			idx = if idx == 0 { self.period - 1 } else { idx - 1 };
		}
		let mut phase = if rp.abs() > 0.001 {
			atan(ip / rp)
		} else {
			TULIP_PI * if ip < 0.0 { -1.0 } else { 1.0 }
		};
		if rp < 0.0 {
			phase += TULIP_PI;
		}
		phase += TULIP_PI / 2.0;
		if phase < 0.0 {
			phase += TULIP_TPI;
		}
		if phase > TULIP_TPI {
			phase -= TULIP_TPI;
		}
		(phase.sin(), (phase + TULIP_PI / 4.0).sin())
	}
}

#[derive(Clone, Debug)]
pub struct MswBatchRange {
	pub period: (usize, usize, usize),
}

impl Default for MswBatchRange {
	fn default() -> Self {
		Self { period: (5, 30, 1) }
	}
}

#[derive(Clone, Debug, Default)]
pub struct MswBatchBuilder {
	range: MswBatchRange,
	kernel: Kernel,
}

impl MswBatchBuilder {
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
	pub fn apply_slice(self, data: &[f64]) -> Result<MswBatchOutput, MswError> {
		msw_batch_with_kernel(data, &self.range, self.kernel)
	}
	pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<MswBatchOutput, MswError> {
		MswBatchBuilder::new().kernel(k).apply_slice(data)
	}
	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<MswBatchOutput, MswError> {
		let slice = source_type(c, src);
		self.apply_slice(slice)
	}
	pub fn with_default_candles(c: &Candles) -> Result<MswBatchOutput, MswError> {
		MswBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
	}
}

pub fn msw_batch_with_kernel(data: &[f64], sweep: &MswBatchRange, k: Kernel) -> Result<MswBatchOutput, MswError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(MswError::InvalidPeriod { period: 0, data_len: 0 }),
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	msw_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct MswBatchOutput {
	pub sine: Vec<f64>,
	pub lead: Vec<f64>,
	pub combos: Vec<MswParams>,
	pub rows: usize,
	pub cols: usize,
}

impl MswBatchOutput {
	pub fn row_for_params(&self, p: &MswParams) -> Option<usize> {
		self.combos
			.iter()
			.position(|c| c.period.unwrap_or(5) == p.period.unwrap_or(5))
	}
	pub fn sine_for(&self, p: &MswParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.sine[start..start + self.cols]
		})
	}
	pub fn lead_for(&self, p: &MswParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.lead[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &MswBatchRange) -> Vec<MswParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let periods = axis_usize(r.period);
	periods.into_iter().map(|p| MswParams { period: Some(p) }).collect()
}

#[inline(always)]
pub fn msw_batch_slice(data: &[f64], sweep: &MswBatchRange, kern: Kernel) -> Result<MswBatchOutput, MswError> {
	msw_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn msw_batch_par_slice(data: &[f64], sweep: &MswBatchRange, kern: Kernel) -> Result<MswBatchOutput, MswError> {
	msw_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn msw_batch_inner(
	data: &[f64],
	sweep: &MswBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<MswBatchOutput, MswError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(MswError::InvalidPeriod { period: 0, data_len: 0 });
	}
	let first = data.iter().position(|x| !x.is_nan()).ok_or(MswError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if data.len() - first < max_p {
		return Err(MswError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}
	let rows = combos.len();
	let cols = data.len();
	
	// Use uninitialized memory for better performance
	let mut sine_buf = make_uninit_matrix(rows, cols);
	let mut lead_buf = make_uninit_matrix(rows, cols);
	
	// Initialize NaN prefixes for each row based on warmup periods
	let warmup_periods: Vec<usize> = combos.iter().map(|c| {
		let period = c.period.unwrap();
		first + period - 1
	}).collect();
	init_matrix_prefixes(&mut sine_buf, cols, &warmup_periods);
	init_matrix_prefixes(&mut lead_buf, cols, &warmup_periods);
	
	// Convert to mutable slices for computation
	let mut sine_guard = core::mem::ManuallyDrop::new(sine_buf);
	let mut lead_guard = core::mem::ManuallyDrop::new(lead_buf);
	let sine: &mut [f64] = unsafe { 
		core::slice::from_raw_parts_mut(sine_guard.as_mut_ptr() as *mut f64, sine_guard.len()) 
	};
	let lead: &mut [f64] = unsafe { 
		core::slice::from_raw_parts_mut(lead_guard.as_mut_ptr() as *mut f64, lead_guard.len()) 
	};
	let do_row = |row: usize, sine_row: &mut [f64], lead_row: &mut [f64]| unsafe {
		let period = combos[row].period.unwrap();
		match kern {
			Kernel::Scalar => msw_row_scalar(data, first, period, sine_row, lead_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => msw_row_avx2(data, first, period, sine_row, lead_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => msw_row_avx512(data, first, period, sine_row, lead_row),
			_ => unreachable!(),
		}
	};
	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			sine.par_chunks_mut(cols)
				.zip(lead.par_chunks_mut(cols))
				.enumerate()
				.for_each(|(row, (sine_row, lead_row))| do_row(row, sine_row, lead_row));
		}

		#[cfg(target_arch = "wasm32")]
		{
			for (row, (sine_row, lead_row)) in sine.chunks_mut(cols).zip(lead.chunks_mut(cols)).enumerate() {
				do_row(row, sine_row, lead_row);
			}
		}
	} else {
		for (row, (sine_row, lead_row)) in sine.chunks_mut(cols).zip(lead.chunks_mut(cols)).enumerate() {
			do_row(row, sine_row, lead_row);
		}
	}
	
	// Convert back to owned vectors
	let sine_vec = unsafe {
		Vec::from_raw_parts(
			sine_guard.as_mut_ptr() as *mut f64,
			sine_guard.len(),
			sine_guard.capacity(),
		)
	};
	let lead_vec = unsafe {
		Vec::from_raw_parts(
			lead_guard.as_mut_ptr() as *mut f64,
			lead_guard.len(),
			lead_guard.capacity(),
		)
	};
	
	Ok(MswBatchOutput {
		sine: sine_vec,
		lead: lead_vec,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
fn msw_batch_inner_into(
	data: &[f64],
	sweep: &MswBatchRange,
	kern: Kernel,
	parallel: bool,
	sine_out: &mut [f64],
	lead_out: &mut [f64],
) -> Result<Vec<MswParams>, MswError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(MswError::InvalidPeriod { period: 0, data_len: 0 });
	}
	
	let first = data.iter().position(|x| !x.is_nan()).ok_or(MswError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if data.len() - first < max_p {
		return Err(MswError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}
	
	let rows = combos.len();
	let cols = data.len();
	
	if sine_out.len() != rows * cols || lead_out.len() != rows * cols {
		return Err(MswError::InvalidPeriod { period: 0, data_len: 0 });
	}
	
	// Initialize NaN prefixes for each row
	for (row, combo) in combos.iter().enumerate() {
		let period = combo.period.unwrap();
		let warmup = first + period - 1;
		let start = row * cols;
		for i in 0..warmup.min(cols) {
			sine_out[start + i] = f64::NAN;
			lead_out[start + i] = f64::NAN;
		}
	}
	
	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			use rayon::prelude::*;
			sine_out
				.par_chunks_mut(cols)
				.zip(lead_out.par_chunks_mut(cols))
				.enumerate()
				.for_each(|(row, (sine_slice, lead_slice))| unsafe {
					let period = combos[row].period.unwrap();
					match kern {
						Kernel::Scalar => msw_row_scalar(data, first, period, sine_slice, lead_slice),
						#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
						Kernel::Avx2 => msw_row_avx2(data, first, period, sine_slice, lead_slice),
						#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
						Kernel::Avx512 => msw_row_avx512(data, first, period, sine_slice, lead_slice),
						_ => unreachable!(),
					}
				});
		}
		
		#[cfg(target_arch = "wasm32")]
		{
			for row in 0..rows {
				let period = combos[row].period.unwrap();
				let start = row * cols;
				let end = start + cols;
				unsafe {
					match kern {
						Kernel::Scalar => msw_row_scalar(data, first, period, &mut sine_out[start..end], &mut lead_out[start..end]),
						#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
						Kernel::Avx2 => msw_row_avx2(data, first, period, &mut sine_out[start..end], &mut lead_out[start..end]),
						#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
						Kernel::Avx512 => msw_row_avx512(data, first, period, &mut sine_out[start..end], &mut lead_out[start..end]),
						_ => unreachable!(),
					}
				}
			}
		}
	} else {
		for row in 0..rows {
			let period = combos[row].period.unwrap();
			let start = row * cols;
			let end = start + cols;
			unsafe {
				match kern {
					Kernel::Scalar => msw_row_scalar(data, first, period, &mut sine_out[start..end], &mut lead_out[start..end]),
					#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
					Kernel::Avx2 => msw_row_avx2(data, first, period, &mut sine_out[start..end], &mut lead_out[start..end]),
					#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
					Kernel::Avx512 => msw_row_avx512(data, first, period, &mut sine_out[start..end], &mut lead_out[start..end]),
					_ => unreachable!(),
				}
			}
		}
	}
	
	Ok(combos)
}

#[inline(always)]
unsafe fn msw_row_scalar(data: &[f64], first: usize, period: usize, sine: &mut [f64], lead: &mut [f64]) {
	let mut cos_table: AVec<f64, aligned_vec::ConstAlign<CACHELINE_ALIGN>> =
		AVec::with_capacity(CACHELINE_ALIGN, period);
	let mut sin_table: AVec<f64, aligned_vec::ConstAlign<CACHELINE_ALIGN>> =
		AVec::with_capacity(CACHELINE_ALIGN, period);
	cos_table.resize(period, 0.0);
	sin_table.resize(period, 0.0);
	for j in 0..period {
		let angle = TULIP_TPI * j as f64 / period as f64;
		cos_table[j] = angle.cos();
		sin_table[j] = angle.sin();
	}
	for i in (first + period - 1)..data.len() {
		let mut rp = 0.0;
		let mut ip = 0.0;
		for j in 0..period {
			let weight = data[i - j];
			rp += cos_table[j] * weight;
			ip += sin_table[j] * weight;
		}
		let mut phase = if rp.abs() > 0.001 {
			atan(ip / rp)
		} else {
			TULIP_PI * if ip < 0.0 { -1.0 } else { 1.0 }
		};
		if rp < 0.0 {
			phase += TULIP_PI;
		}
		phase += TULIP_PI / 2.0;
		if phase < 0.0 {
			phase += TULIP_TPI;
		}
		if phase > TULIP_TPI {
			phase -= TULIP_TPI;
		}
		sine[i] = phase.sin();
		lead[i] = (phase + TULIP_PI / 4.0).sin();
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn msw_row_avx2(data: &[f64], first: usize, period: usize, sine: &mut [f64], lead: &mut [f64]) {
	msw_row_scalar(data, first, period, sine, lead)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn msw_row_avx512(data: &[f64], first: usize, period: usize, sine: &mut [f64], lead: &mut [f64]) {
	if period <= 32 {
		msw_row_avx512_short(data, first, period, sine, lead)
	} else {
		msw_row_avx512_long(data, first, period, sine, lead)
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn msw_row_avx512_short(data: &[f64], first: usize, period: usize, sine: &mut [f64], lead: &mut [f64]) {
	msw_row_scalar(data, first, period, sine, lead)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn msw_row_avx512_long(data: &[f64], first: usize, period: usize, sine: &mut [f64], lead: &mut [f64]) {
	msw_row_scalar(data, first, period, sine, lead)
}

#[cfg(feature = "python")]
#[pyfunction(name = "msw")]
#[pyo3(signature = (data, period, kernel=None))]
pub fn msw_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	period: usize,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
	use numpy::{IntoPyArray, PyArrayMethods};

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, false)?;
	
	let params = MswParams {
		period: Some(period),
	};
	let msw_in = MswInput::from_slice(slice_in, params);
	
	// Get MswOutput struct containing Vec<f64> from Rust function
	let output = py
		.allow_threads(|| msw_with_kernel(&msw_in, kern))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;
	
	// Build output dictionary with zero-copy transfer
	let dict = PyDict::new(py);
	dict.set_item("sine", output.sine.into_pyarray(py))?;
	dict.set_item("lead", output.lead.into_pyarray(py))?;
	
	Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "MswStream")]
pub struct MswStreamPy {
	stream: MswStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl MswStreamPy {
	#[new]
	fn new(period: usize) -> PyResult<Self> {
		let params = MswParams {
			period: Some(period),
		};
		let stream = MswStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(MswStreamPy { stream })
	}
	
	/// Updates the stream with a new value and returns the calculated MSW values.
	/// Returns `None` if the buffer is not yet full, otherwise returns a tuple of (sine, lead).
	fn update(&mut self, value: f64) -> Option<(f64, f64)> {
		self.stream.update(value)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "msw_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
pub fn msw_batch_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	period_range: (usize, usize, usize),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, true)?;
	
	let sweep = MswBatchRange {
		period: period_range,
	};
	
	// Calculate dimensions
	let combos = expand_grid(&sweep);
	let rows = combos.len();
	let cols = slice_in.len();
	
	// Pre-allocate output arrays (OK for batch operations)
	let out_sine = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let out_lead = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let slice_out_sine = unsafe { out_sine.as_slice_mut()? };
	let slice_out_lead = unsafe { out_lead.as_slice_mut()? };
	
	// Compute without GIL
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
			
			// Write directly to pre-allocated arrays
			msw_batch_inner_into(slice_in, &sweep, simd, true, slice_out_sine, slice_out_lead)
		})
		.map_err(|e| PyValueError::new_err(e.to_string()))?;
	
	// Build dict with zero-copy transfers
	let dict = PyDict::new(py);
	dict.set_item("sine", out_sine.reshape((rows, cols))?)?;
	dict.set_item("lead", out_lead.reshape((rows, cols))?)?;
	dict.set_item(
		"periods",
		combos
			.iter()
			.map(|p| p.period.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;
	
	Ok(dict)
}

/// Write directly to output slices - no allocations
#[inline]
pub fn msw_into_slice(
	sine_dst: &mut [f64],
	lead_dst: &mut [f64],
	input: &MswInput,
	kern: Kernel,
) -> Result<(), MswError> {
	let data: &[f64] = match &input.data {
		MswData::Candles { candles, source } => source_type(candles, source),
		MswData::Slice(sl) => sl,
	};
	
	if data.is_empty() {
		return Err(MswError::EmptyData);
	}
	
	let period = input.get_period();
	let first = data.iter().position(|x| !x.is_nan()).ok_or(MswError::AllValuesNaN)?;
	let len = data.len();
	
	if period == 0 || period > len {
		return Err(MswError::InvalidPeriod { period, data_len: len });
	}
	
	if (len - first) < period {
		return Err(MswError::NotEnoughValidData {
			needed: period,
			valid: len - first,
		});
	}
	
	if sine_dst.len() != data.len() || lead_dst.len() != data.len() {
		return Err(MswError::InvalidPeriod {
			period: sine_dst.len(),
			data_len: data.len(),
		});
	}
	
	let chosen = match kern {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};
	
	// Compute directly into destination slices
	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => {
				msw_scalar_into(data, period, first, len, sine_dst, lead_dst)
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => {
				// AVX2 is currently a stub, use scalar
				msw_scalar_into(data, period, first, len, sine_dst, lead_dst)
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => {
				// AVX512 is currently a stub, use scalar
				msw_scalar_into(data, period, first, len, sine_dst, lead_dst)
			}
			_ => unreachable!(),
		}
	}?;
	
	// Fill warmup period with NaN
	let warmup = first + period - 1;
	for v in &mut sine_dst[..warmup] {
		*v = f64::NAN;
	}
	for v in &mut lead_dst[..warmup] {
		*v = f64::NAN;
	}
	
	Ok(())
}

/// Scalar implementation that writes directly to output slices
#[inline]
unsafe fn msw_scalar_into(
	data: &[f64],
	period: usize,
	first: usize,
	len: usize,
	sine: &mut [f64],
	lead: &mut [f64],
) -> Result<(), MswError> {
	let mut cos_table: AVec<f64, aligned_vec::ConstAlign<CACHELINE_ALIGN>> =
		AVec::with_capacity(CACHELINE_ALIGN, period);
	let mut sin_table: AVec<f64, aligned_vec::ConstAlign<CACHELINE_ALIGN>> =
		AVec::with_capacity(CACHELINE_ALIGN, period);
	cos_table.resize(period, 0.0);
	sin_table.resize(period, 0.0);

	for j in 0..period {
		let angle = TULIP_TPI * j as f64 / period as f64;
		cos_table[j] = angle.cos();
		sin_table[j] = angle.sin();
	}
	
	for i in (first + period - 1)..len {
		let mut rp = 0.0;
		let mut ip = 0.0;
		for j in 0..period {
			let weight = data[i - j];
			rp += cos_table[j] * weight;
			ip += sin_table[j] * weight;
		}
		let mut phase = if rp.abs() > 0.001 {
			atan(ip / rp)
		} else {
			TULIP_PI * if ip < 0.0 { -1.0 } else { 1.0 }
		};
		if rp < 0.0 {
			phase += TULIP_PI;
		}
		phase += TULIP_PI / 2.0;
		if phase < 0.0 {
			phase += TULIP_TPI;
		}
		if phase > TULIP_TPI {
			phase -= TULIP_TPI;
		}
		sine[i] = phase.sin();
		lead[i] = (phase + TULIP_PI / 4.0).sin();
	}
	
	Ok(())
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct MswJsOutput {
	pub sine: Vec<f64>,
	pub lead: Vec<f64>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn msw_js(data: &[f64], period: usize) -> Result<JsValue, JsValue> {
	let params = MswParams { period: Some(period) };
	let input = MswInput::from_slice(data, params);
	
	let mut sine_output = vec![0.0; data.len()];
	let mut lead_output = vec![0.0; data.len()];
	
	msw_into_slice(&mut sine_output, &mut lead_output, &input, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	// Create the structured output
	let js_output = MswJsOutput {
		sine: sine_output,
		lead: lead_output,
	};
	
	// Serialize the output struct into a JavaScript object
	serde_wasm_bindgen::to_value(&js_output)
		.map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

// Keep msw_wasm as deprecated alias for backward compatibility
#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[deprecated(since = "1.0.0", note = "Use msw_js instead")]
pub fn msw_wasm(data: &[f64], period: usize) -> Result<JsValue, JsValue> {
	msw_js(data, period)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn msw_into(
	in_ptr: *const f64,
	sine_ptr: *mut f64,
	lead_ptr: *mut f64,
	len: usize,
	period: usize,
) -> Result<(), JsValue> {
	if in_ptr.is_null() || sine_ptr.is_null() || lead_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}
	
	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);
		
		if period == 0 || period > len {
			return Err(JsValue::from_str("Invalid period"));
		}
		
		let params = MswParams { period: Some(period) };
		let input = MswInput::from_slice(data, params);
		
		// Check for aliasing - any output pointer matches input or each other
		let aliasing = in_ptr as *const _ == sine_ptr as *const _ 
			|| in_ptr as *const _ == lead_ptr as *const _
			|| sine_ptr == lead_ptr;
		
		if aliasing {
			// Use temporary buffers when aliasing detected
			let mut temp_sine = vec![0.0; len];
			let mut temp_lead = vec![0.0; len];
			msw_into_slice(&mut temp_sine, &mut temp_lead, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			
			// Copy to output pointers
			let sine_out = std::slice::from_raw_parts_mut(sine_ptr, len);
			let lead_out = std::slice::from_raw_parts_mut(lead_ptr, len);
			sine_out.copy_from_slice(&temp_sine);
			lead_out.copy_from_slice(&temp_lead);
		} else {
			// Direct computation into output slices
			let sine_out = std::slice::from_raw_parts_mut(sine_ptr, len);
			let lead_out = std::slice::from_raw_parts_mut(lead_ptr, len);
			msw_into_slice(sine_out, lead_out, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		
		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn msw_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn msw_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe {
			let _ = Vec::from_raw_parts(ptr, len, len);
		}
	}
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct MswBatchConfig {
	pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct MswBatchJsOutput {
	pub sine: Vec<f64>,
	pub lead: Vec<f64>,
	pub combos: Vec<MswParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = msw_batch)]
pub fn msw_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	let config: MswBatchConfig =
		serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
	
	let sweep = MswBatchRange {
		period: config.period_range,
	};
	
	let output = msw_batch_inner(data, &sweep, Kernel::Auto, false)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	let js_output = MswBatchJsOutput {
		sine: output.sine,
		lead: output.lead,
		combos: output.combos,
		rows: output.rows,
		cols: output.cols,
	};
	
	serde_wasm_bindgen::to_value(&js_output)
		.map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn msw_batch_js(
	data: &[f64],
	period_start: usize,
	period_end: usize,
	period_step: usize,
) -> Result<JsValue, JsValue> {
	let sweep = MswBatchRange {
		period: (period_start, period_end, period_step),
	};
	
	// Use the existing batch function with parallel=false for WASM
	let output = msw_batch_inner(data, &sweep, Kernel::Auto, false)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	// Create the structured output
	let js_output = MswBatchJsOutput {
		sine: output.sine,
		lead: output.lead,
		combos: output.combos,
		rows: output.rows,
		cols: output.cols,
	};
	
	// Serialize the output struct into a JavaScript object
	serde_wasm_bindgen::to_value(&js_output)
		.map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn msw_batch_metadata_js(
	period_start: usize,
	period_end: usize,
	period_step: usize,
) -> Result<Vec<f64>, JsValue> {
	let sweep = MswBatchRange {
		period: (period_start, period_end, period_step),
	};
	
	let combos = expand_grid(&sweep);
	let metadata = combos.iter().map(|combo| combo.period.unwrap() as f64).collect();
	
	Ok(metadata)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn msw_batch_into(
	in_ptr: *const f64,
	sine_ptr: *mut f64,
	lead_ptr: *mut f64,
	len: usize,
	period_start: usize,
	period_end: usize,
	period_step: usize,
) -> Result<usize, JsValue> {
	if in_ptr.is_null() || sine_ptr.is_null() || lead_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}
	
	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);
		
		let sweep = MswBatchRange {
			period: (period_start, period_end, period_step),
		};
		
		let combos = expand_grid(&sweep);
		if combos.is_empty() {
			return Err(JsValue::from_str("No valid parameter combinations"));
		}
		
		let rows = combos.len();
		let cols = len;
		let total_len = rows * cols;
		
		let sine_out = std::slice::from_raw_parts_mut(sine_ptr, total_len);
		let lead_out = std::slice::from_raw_parts_mut(lead_ptr, total_len);
		
		// Process each parameter combination
		for (idx, params) in combos.iter().enumerate() {
			let row_start = idx * cols;
			let row_end = row_start + cols;
			
			let input = MswInput::from_slice(data, params.clone());
			
			msw_into_slice(
				&mut sine_out[row_start..row_end],
				&mut lead_out[row_start..row_end],
				&input,
				Kernel::Auto,
			)
			.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		
		Ok(rows)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_msw_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let default_params = MswParams { period: None };
		let input_default = MswInput::from_candles(&candles, "close", default_params);
		let output_default = msw_with_kernel(&input_default, kernel)?;
		assert_eq!(output_default.sine.len(), candles.close.len());
		assert_eq!(output_default.lead.len(), candles.close.len());
		Ok(())
	}

	fn check_msw_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = MswParams { period: Some(5) };
		let input = MswInput::from_candles(&candles, "close", params);
		let msw_result = msw_with_kernel(&input, kernel)?;
		let expected_last_five_sine = [
			-0.49733966449848194,
			-0.8909425976991894,
			-0.709353328514554,
			-0.40483478076837887,
			-0.8817006719953886,
		];
		let expected_last_five_lead = [
			-0.9651269132969991,
			-0.30888310410390457,
			-0.003182174183612666,
			0.36030983330963545,
			-0.28983704937461496,
		];
		let start = msw_result.sine.len().saturating_sub(5);
		for (i, &val) in msw_result.sine[start..].iter().enumerate() {
			let diff = (val - expected_last_five_sine[i]).abs();
			assert!(
				diff < 1e-1,
				"[{}] MSW sine mismatch at idx {}: got {}, expected {}",
				test_name,
				i,
				val,
				expected_last_five_sine[i]
			);
		}
		for (i, &val) in msw_result.lead[start..].iter().enumerate() {
			let diff = (val - expected_last_five_lead[i]).abs();
			assert!(
				diff < 1e-1,
				"[{}] MSW lead mismatch at idx {}: got {}, expected {}",
				test_name,
				i,
				val,
				expected_last_five_lead[i]
			);
		}
		Ok(())
	}

	fn check_msw_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = MswInput::with_default_candles(&candles);
		let output = msw_with_kernel(&input, kernel)?;
		assert_eq!(output.sine.len(), candles.close.len());
		assert_eq!(output.lead.len(), candles.close.len());
		Ok(())
	}

	fn check_msw_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let input_data = [10.0, 20.0, 30.0];
		let params = MswParams { period: Some(0) };
		let input = MswInput::from_slice(&input_data, params);
		let res = msw_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] MSW should fail with zero period", test_name);
		Ok(())
	}

	fn check_msw_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data_small = [10.0, 20.0, 30.0];
		let params = MswParams { period: Some(10) };
		let input = MswInput::from_slice(&data_small, params);
		let res = msw_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] MSW should fail with period exceeding length",
			test_name
		);
		Ok(())
	}

	fn check_msw_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let single_point = [42.0];
		let params = MswParams { period: Some(5) };
		let input = MswInput::from_slice(&single_point, params);
		let res = msw_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] MSW should fail with insufficient data", test_name);
		Ok(())
	}

	fn check_msw_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = MswParams { period: Some(5) };
		let input = MswInput::from_candles(&candles, "close", params);
		let res = msw_with_kernel(&input, kernel)?;
		assert_eq!(res.sine.len(), candles.close.len());
		assert_eq!(res.lead.len(), candles.close.len());
		Ok(())
	}

	fn check_msw_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let period = 5;
		let input = MswInput::from_candles(&candles, "close", MswParams { period: Some(period) });
		let batch_output = msw_with_kernel(&input, kernel)?;
		let mut stream = MswStream::try_new(MswParams { period: Some(period) })?;
		let mut sine_stream = Vec::with_capacity(candles.close.len());
		let mut lead_stream = Vec::with_capacity(candles.close.len());
		for &price in &candles.close {
			match stream.update(price) {
				Some((s, l)) => {
					sine_stream.push(s);
					lead_stream.push(l);
				}
				None => {
					sine_stream.push(f64::NAN);
					lead_stream.push(f64::NAN);
				}
			}
		}
		assert_eq!(batch_output.sine.len(), sine_stream.len());
		assert_eq!(batch_output.lead.len(), lead_stream.len());
		for (i, (&b, &s)) in batch_output.sine.iter().zip(sine_stream.iter()).enumerate() {
			if b.is_nan() && s.is_nan() {
				continue;
			}
			let diff = (b - s).abs();
			assert!(
				diff < 1e-9,
				"[{}] MSW streaming sine mismatch at idx {}: batch={}, stream={}, diff={}",
				test_name,
				i,
				b,
				s,
				diff
			);
		}
		for (i, (&b, &l)) in batch_output.lead.iter().zip(lead_stream.iter()).enumerate() {
			if b.is_nan() && l.is_nan() {
				continue;
			}
			let diff = (b - l).abs();
			assert!(
				diff < 1e-9,
				"[{}] MSW streaming lead mismatch at idx {}: batch={}, stream={}, diff={}",
				test_name,
				i,
				b,
				l,
				diff
			);
		}
		Ok(())
	}

	macro_rules! generate_all_msw_tests {
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
	generate_all_msw_tests!(
		check_msw_partial_params,
		check_msw_accuracy,
		check_msw_default_candles,
		check_msw_zero_period,
		check_msw_period_exceeds_length,
		check_msw_very_small_dataset,
		check_msw_nan_handling,
		check_msw_streaming
	);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let output = MswBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;
		let def = MswParams::default();
		let row = output.sine_for(&def).expect("default row missing");
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
