//! # Super Smoother Filter
//!
//! A double-pole smoothing filter that reduces high-frequency noise and preserves significant trend information.
//! Parameters allow flexible window sizing. SIMD acceleration is stubbed to scalar for API parity.
//!
//! ## Parameters
//! - **period**: Main lookback length (defaults to 14). Must be ≥ 1 and ≤ the data length.
//!
//! ## Errors
//! - **AllValuesNaN**: supersmoother: All input data values are NaN.
//! - **InvalidPeriod**: supersmoother: period is 0 or > data length.
//! - **NotEnoughValidData**: supersmoother: Not enough valid data points for the requested period.
//! - **EmptyData**: supersmoother: No input data.
//!
//! ## Returns
//! - `Ok(SuperSmootherOutput)` on success (`values: Vec<f64>` matching input).
//! - `Err(SuperSmootherError)` on validation or computation errors.

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
use std::f64::consts::PI;
use std::mem::MaybeUninit;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum SuperSmootherData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

impl<'a> AsRef<[f64]> for SuperSmootherInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			SuperSmootherData::Slice(sl) => sl,
			SuperSmootherData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

#[derive(Debug, Clone)]
pub struct SuperSmootherOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct SuperSmootherParams {
	pub period: Option<usize>,
}

impl Default for SuperSmootherParams {
	fn default() -> Self {
		Self { period: Some(14) }
	}
}

#[derive(Debug, Clone)]
pub struct SuperSmootherInput<'a> {
	pub data: SuperSmootherData<'a>,
	pub params: SuperSmootherParams,
}

impl<'a> SuperSmootherInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: SuperSmootherParams) -> Self {
		Self {
			data: SuperSmootherData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slice(sl: &'a [f64], p: SuperSmootherParams) -> Self {
		Self {
			data: SuperSmootherData::Slice(sl),
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", SuperSmootherParams::default())
	}
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(14)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct SuperSmootherBuilder {
	period: Option<usize>,
	kernel: Kernel,
}

impl Default for SuperSmootherBuilder {
	fn default() -> Self {
		Self {
			period: None,
			kernel: Kernel::Auto,
		}
	}
}

impl SuperSmootherBuilder {
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
	pub fn apply(self, c: &Candles) -> Result<SuperSmootherOutput, SuperSmootherError> {
		let p = SuperSmootherParams { period: self.period };
		let i = SuperSmootherInput::from_candles(c, "close", p);
		supersmoother_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slice(self, d: &[f64]) -> Result<SuperSmootherOutput, SuperSmootherError> {
		let p = SuperSmootherParams { period: self.period };
		let i = SuperSmootherInput::from_slice(d, p);
		supersmoother_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<SuperSmootherStream, SuperSmootherError> {
		let p = SuperSmootherParams { period: self.period };
		SuperSmootherStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum SuperSmootherError {
	#[error("supersmoother: All values are NaN.")]
	AllValuesNaN,
	#[error("supersmoother: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("supersmoother: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("supersmoother: Empty data provided.")]
	EmptyData,
}

#[inline]
pub fn supersmoother(input: &SuperSmootherInput) -> Result<SuperSmootherOutput, SuperSmootherError> {
	supersmoother_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
pub fn supersmoother_with_kernel(
	input: &SuperSmootherInput,
	kernel: Kernel,
) -> Result<SuperSmootherOutput, SuperSmootherError> {
	// ---------- 0. validation ----------
	let data: &[f64] = input.as_ref();
	if data.is_empty() {
		return Err(SuperSmootherError::EmptyData);
	}

	let first = data
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(SuperSmootherError::AllValuesNaN)?;

	let len = data.len();
	let period = input.get_period();

	if period == 0 || period > len {
		return Err(SuperSmootherError::InvalidPeriod { period, data_len: len });
	}
	if (len - first) < period {
		return Err(SuperSmootherError::NotEnoughValidData {
			needed: period,
			valid: len - first,
		});
	}

	// ---------- 1. prepare the output buffer ----------
	//   All indices 0‥warm-1 are guaranteed NaN so the stream version lines up.
	let warm = first + period - 1;
	let mut out = alloc_with_nan_prefix(len, warm);

	// ---------- 2. choose kernel ----------
	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	// ---------- 3. do the work ----------
	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => {
				supersmoother_row_scalar(data, first, period, &mut out);
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => {
				supersmoother_row_avx2(data, first, period, &mut out);
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => {
				supersmoother_row_avx512(data, first, period, &mut out);
			}
			_ => unreachable!(),
		}
	}

	// ---------- 4. package and return ----------
	Ok(SuperSmootherOutput { values: out })
}

/// Write SuperSmoother values directly to output slice - no allocations
pub fn supersmoother_into_slice(
	dst: &mut [f64],
	input: &SuperSmootherInput,
	kernel: Kernel,
) -> Result<(), SuperSmootherError> {
	// ---------- 0. validation ----------
	let data: &[f64] = input.as_ref();
	if data.is_empty() {
		return Err(SuperSmootherError::EmptyData);
	}

	let first = data
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(SuperSmootherError::AllValuesNaN)?;

	let len = data.len();
	let period = input.get_period();

	if period == 0 || period > len {
		return Err(SuperSmootherError::InvalidPeriod { period, data_len: len });
	}
	if (len - first) < period {
		return Err(SuperSmootherError::NotEnoughValidData {
			needed: period,
			valid: len - first,
		});
	}

	// Verify output buffer size matches input
	if dst.len() != data.len() {
		return Err(SuperSmootherError::InvalidPeriod {
			period: dst.len(),
			data_len: data.len(),
		});
	}

	// ---------- 1. choose kernel ----------
	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	// ---------- 2. compute directly into dst ----------
	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => {
				supersmoother_row_scalar(data, first, period, dst);
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => {
				supersmoother_row_avx2(data, first, period, dst);
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => {
				supersmoother_row_avx512(data, first, period, dst);
			}
			_ => unreachable!("Unsupported kernel"),
		}
	}

	// Note: supersmoother_row_* functions already handle warmup by filling with NaN
	Ok(())
}

#[inline]
pub unsafe fn supersmoother_scalar(
	data: &[f64],
	period: usize,
	first: usize,
) -> Result<SuperSmootherOutput, SuperSmootherError> {
	let len = data.len();
	if len == 0 {
		return Err(SuperSmootherError::EmptyData);
	}
	if period == 0 {
		return Err(SuperSmootherError::InvalidPeriod { period, data_len: len });
	}

	let warm = first + period - 1;
	if warm >= len {
		return Err(SuperSmootherError::NotEnoughValidData {
			needed: period,
			valid: len - first,
		});
	}

	let mut out = alloc_with_nan_prefix(len, warm);
	let a = (-1.414_f64 * PI / (period as f64)).exp();
	let a_sq = a * a;
	let b = 2.0 * a * (1.414_f64 * PI / (period as f64)).cos();
	let c = (1.0 + a_sq - b) * 0.5;

	// Initial conditions
	if len > warm {
		out[warm] = data[warm];
	}
	if len > warm + 1 {
		out[warm + 1] = data[warm + 1];
	}
	// Main calculation
	for i in (warm + 2)..len {
		let prev_1 = out[i - 1];
		let prev_2 = out[i - 2];
		let d_i = data[i];
		let d_im1 = data[i - 1];
		out[i] = c * (d_i + d_im1) + b * prev_1 - a_sq * prev_2;
	}
	Ok(SuperSmootherOutput { values: out })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn supersmoother_avx2(
	data: &[f64],
	period: usize,
	first: usize,
) -> Result<SuperSmootherOutput, SuperSmootherError> {
	supersmoother_scalar(data, period, first)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn supersmoother_avx512(
	data: &[f64],
	period: usize,
	first: usize,
) -> Result<SuperSmootherOutput, SuperSmootherError> {
	if period <= 32 {
		supersmoother_avx512_short(data, period, first)
	} else {
		supersmoother_avx512_long(data, period, first)
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn supersmoother_avx512_short(
	data: &[f64],
	period: usize,
	first: usize,
) -> Result<SuperSmootherOutput, SuperSmootherError> {
	supersmoother_scalar(data, period, first)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn supersmoother_avx512_long(
	data: &[f64],
	period: usize,
	first: usize,
) -> Result<SuperSmootherOutput, SuperSmootherError> {
	supersmoother_scalar(data, period, first)
}

#[derive(Debug, Clone)]
pub struct SuperSmootherStream {
	period: usize,
	buffer: Vec<f64>,
	head: usize,
	filled: bool,
	a: f64,
	a_sq: f64,
	b: f64,
	c: f64,
}

impl SuperSmootherStream {
	pub fn try_new(params: SuperSmootherParams) -> Result<Self, SuperSmootherError> {
		let period = params.period.unwrap_or(14);
		if period == 0 {
			return Err(SuperSmootherError::InvalidPeriod { period, data_len: 0 });
		}
		let a = (-1.414_f64 * PI / (period as f64)).exp();
		let a_sq = a * a;
		let b = 2.0 * a * (1.414_f64 * PI / (period as f64)).cos();
		let c = (1.0 + a_sq - b) * 0.5;
		Ok(Self {
			period,
			buffer: vec![f64::NAN; 2],
			head: 0,
			filled: false,
			a,
			a_sq,
			b,
			c,
		})
	}
	#[inline(always)]
	pub fn update(&mut self, value: f64, prev: Option<(f64, f64)>) -> Option<f64> {
		self.buffer[self.head] = value;
		self.head = (self.head + 1) % 2;
		if !self.filled && self.head == 0 {
			self.filled = true;
		}
		if !self.filled {
			return None;
		}
		let prev_1 = if let Some((p1, _)) = prev {
			p1
		} else {
			self.buffer[(self.head + 1) % 2]
		};
		let prev_2 = if let Some((_, p2)) = prev {
			p2
		} else {
			self.buffer[self.head]
		};
		Some(self.c * (value + prev_1) + self.b * prev_1 - self.a_sq * prev_2)
	}
}

#[derive(Clone, Debug)]
pub struct SuperSmootherBatchRange {
	pub period: (usize, usize, usize),
}

impl Default for SuperSmootherBatchRange {
	fn default() -> Self {
		Self { period: (14, 100, 1) }
	}
}

#[derive(Clone, Debug, Default)]
pub struct SuperSmootherBatchBuilder {
	range: SuperSmootherBatchRange,
	kernel: Kernel,
}

impl SuperSmootherBatchBuilder {
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
	pub fn apply_slice(self, data: &[f64]) -> Result<SuperSmootherBatchOutput, SuperSmootherError> {
		supersmoother_batch_with_kernel(data, &self.range, self.kernel)
	}
	pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<SuperSmootherBatchOutput, SuperSmootherError> {
		SuperSmootherBatchBuilder::new().kernel(k).apply_slice(data)
	}
	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<SuperSmootherBatchOutput, SuperSmootherError> {
		let slice = source_type(c, src);
		self.apply_slice(slice)
	}
	pub fn with_default_candles(c: &Candles) -> Result<SuperSmootherBatchOutput, SuperSmootherError> {
		SuperSmootherBatchBuilder::new()
			.kernel(Kernel::Auto)
			.apply_candles(c, "close")
	}
}

pub fn supersmoother_batch_with_kernel(
	data: &[f64],
	sweep: &SuperSmootherBatchRange,
	k: Kernel,
) -> Result<SuperSmootherBatchOutput, SuperSmootherError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(SuperSmootherError::InvalidPeriod { period: 0, data_len: 0 }),
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	supersmoother_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct SuperSmootherBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<SuperSmootherParams>,
	pub rows: usize,
	pub cols: usize,
}
impl SuperSmootherBatchOutput {
	pub fn row_for_params(&self, p: &SuperSmootherParams) -> Option<usize> {
		self.combos
			.iter()
			.position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
	}
	pub fn values_for(&self, p: &SuperSmootherParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &SuperSmootherBatchRange) -> Vec<SuperSmootherParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let periods = axis_usize(r.period);
	let mut out = Vec::with_capacity(periods.len());
	for &p in &periods {
		out.push(SuperSmootherParams { period: Some(p) });
	}
	out
}

#[inline(always)]
pub fn supersmoother_batch_slice(
	data: &[f64],
	sweep: &SuperSmootherBatchRange,
	kern: Kernel,
) -> Result<SuperSmootherBatchOutput, SuperSmootherError> {
	supersmoother_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn supersmoother_batch_par_slice(
	data: &[f64],
	sweep: &SuperSmootherBatchRange,
	kern: Kernel,
) -> Result<SuperSmootherBatchOutput, SuperSmootherError> {
	supersmoother_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn supersmoother_batch_inner(
	data: &[f64],
	sweep: &SuperSmootherBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<SuperSmootherBatchOutput, SuperSmootherError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(SuperSmootherError::InvalidPeriod { period: 0, data_len: 0 });
	}
	let first = data
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(SuperSmootherError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if data.len() - first < max_p {
		return Err(SuperSmootherError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}
	let rows = combos.len();
	let cols = data.len();
	let mut raw = make_uninit_matrix(rows, cols);
	let warm: Vec<usize> = combos
		.iter()
		.map(|c| first + c.period.unwrap() - 1) // NaN prefix length for each row
		.collect();
	unsafe { init_matrix_prefixes(&mut raw, cols, &warm) };

	// ---------- 2. closure that fills one row ----------
	let do_row = |row: usize, dst_mu: &mut [std::mem::MaybeUninit<f64>]| unsafe {
		let period = combos[row].period.unwrap();

		// Cast just this row to &mut [f64]
		let out_row = core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

		match kern {
			Kernel::Scalar => supersmoother_row_scalar(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => supersmoother_row_avx2(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => supersmoother_row_avx512(data, first, period, out_row),
			_ => unreachable!(),
		}
	};

	// ---------- 3. run every row directly into `raw` ----------
	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			raw.par_chunks_mut(cols)
				.enumerate()
				.for_each(|(row, slice)| do_row(row, slice));
		}

		#[cfg(target_arch = "wasm32")]
		{
			for (row, slice) in raw.chunks_mut(cols).enumerate() {
				do_row(row, slice);
			}
		}
	} else {
		for (row, slice) in raw.chunks_mut(cols).enumerate() {
			do_row(row, slice);
		}
	}

	// ---------- 4. all elements are now initialised – transmute ----------
	let values: Vec<f64> = unsafe { std::mem::transmute(raw) };
	Ok(SuperSmootherBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
pub fn supersmoother_batch_inner_into(
	data: &[f64],
	sweep: &SuperSmootherBatchRange,
	kern: Kernel,
	parallel: bool,
	out: &mut [f64],
) -> Result<Vec<SuperSmootherParams>, SuperSmootherError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(SuperSmootherError::InvalidPeriod { period: 0, data_len: 0 });
	}

	let first = data
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(SuperSmootherError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if data.len() - first < max_p {
		return Err(SuperSmootherError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}

	let rows = combos.len();
	let cols = data.len();

	// Initialize NaN prefixes directly in the output buffer
	let warm: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap() - 1).collect();

	for (row, &warmup) in warm.iter().enumerate() {
		let row_start = row * cols;
		for i in 0..warmup {
			out[row_start + i] = f64::NAN;
		}
	}

	// Process each row directly into the output buffer
	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			use rayon::prelude::*;
			out.par_chunks_mut(cols).enumerate().for_each(|(row, out_row)| unsafe {
				let period = combos[row].period.unwrap();
				match kern {
					Kernel::Scalar => supersmoother_row_scalar(data, first, period, out_row),
					#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
					Kernel::Avx2 => supersmoother_row_avx2(data, first, period, out_row),
					#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
					Kernel::Avx512 => supersmoother_row_avx512(data, first, period, out_row),
					_ => unreachable!(),
				}
			});
		}
		#[cfg(target_arch = "wasm32")]
		{
			for (row, out_row) in out.chunks_mut(cols).enumerate() {
				unsafe {
					let period = combos[row].period.unwrap();
					match kern {
						Kernel::Scalar => supersmoother_row_scalar(data, first, period, out_row),
						_ => unreachable!(),
					}
				}
			}
		}
	} else {
		for (row, out_row) in out.chunks_mut(cols).enumerate() {
			unsafe {
				let period = combos[row].period.unwrap();
				match kern {
					Kernel::Scalar => supersmoother_row_scalar(data, first, period, out_row),
					#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
					Kernel::Avx2 => supersmoother_row_avx2(data, first, period, out_row),
					#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
					Kernel::Avx512 => supersmoother_row_avx512(data, first, period, out_row),
					_ => unreachable!(),
				}
			}
		}
	}

	Ok(combos)
}

#[inline(always)]
pub unsafe fn supersmoother_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	let len = data.len();
	let a = (-1.414_f64 * PI / (period as f64)).exp();
	let a_sq = a * a;
	let b = 2.0 * a * (1.414_f64 * PI / (period as f64)).cos();
	let c = (1.0 + a_sq - b) * 0.5;
	for i in 0..first + period - 1 {
		out[i] = f64::NAN;
	}
	if len > first + period - 1 {
		out[first + period - 1] = data[first + period - 1];
	}
	if len > first + period {
		out[first + period] = data[first + period];
	}
	for i in (first + period + 1)..len {
		let prev_1 = out[i - 1];
		let prev_2 = out[i - 2];
		let d_i = data[i];
		let d_im1 = data[i - 1];
		out[i] = c * (d_i + d_im1) + b * prev_1 - a_sq * prev_2;
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn supersmoother_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	supersmoother_row_scalar(data, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn supersmoother_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	if period <= 32 {
		supersmoother_row_avx512_short(data, first, period, out)
	} else {
		supersmoother_row_avx512_long(data, first, period, out)
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn supersmoother_row_avx512_short(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	supersmoother_row_scalar(data, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn supersmoother_row_avx512_long(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	supersmoother_row_scalar(data, first, period, out)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_supersmoother_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let default_params = SuperSmootherParams { period: None };
		let input = SuperSmootherInput::from_candles(&candles, "close", default_params);
		let output = supersmoother_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}
	fn check_supersmoother_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = SuperSmootherParams { period: Some(14) };
		let input = SuperSmootherInput::from_candles(&candles, "close", params);
		let result = supersmoother_with_kernel(&input, kernel)?;
		let out_vals = &result.values;
		let expected_last_five = [
			59140.98229179739,
			59172.03593376982,
			59179.40342783722,
			59171.22758152845,
			59127.859841077094,
		];
		let start_idx = out_vals.len() - 5;
		for (i, &val) in out_vals[start_idx..].iter().enumerate() {
			let diff = (val - expected_last_five[i]).abs();
			assert!(
				diff < 1e-8,
				"[{}] mismatch at idx {}: got {}, expected {}",
				test_name,
				i,
				val,
				expected_last_five[i]
			);
		}
		Ok(())
	}
	fn check_supersmoother_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = SuperSmootherInput::with_default_candles(&candles);
		match input.data {
			SuperSmootherData::Candles { source, .. } => assert_eq!(source, "close"),
			_ => panic!("Expected SuperSmootherData::Candles"),
		}
		let output = supersmoother_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}
	fn check_supersmoother_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let input_data = [10.0, 20.0, 30.0];
		let params = SuperSmootherParams { period: Some(0) };
		let input = SuperSmootherInput::from_slice(&input_data, params);
		let res = supersmoother_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] should fail with zero period", test_name);
		Ok(())
	}
	fn check_supersmoother_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data_small = [10.0, 20.0, 30.0];
		let params = SuperSmootherParams { period: Some(10) };
		let input = SuperSmootherInput::from_slice(&data_small, params);
		let res = supersmoother_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] should fail with period exceeding length", test_name);
		Ok(())
	}
	fn check_supersmoother_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let single_point = [42.0];
		let params = SuperSmootherParams { period: Some(14) };
		let input = SuperSmootherInput::from_slice(&single_point, params);
		let res = supersmoother_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] should fail with insufficient data", test_name);
		Ok(())
	}
	fn check_supersmoother_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let first_params = SuperSmootherParams { period: Some(14) };
		let first_input = SuperSmootherInput::from_candles(&candles, "close", first_params);
		let first_result = supersmoother_with_kernel(&first_input, kernel)?;
		let second_params = SuperSmootherParams { period: Some(10) };
		let second_input = SuperSmootherInput::from_slice(&first_result.values, second_params);
		let second_result = supersmoother_with_kernel(&second_input, kernel)?;
		assert_eq!(second_result.values.len(), first_result.values.len());
		Ok(())
	}
	fn check_supersmoother_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = SuperSmootherInput::from_candles(&candles, "close", SuperSmootherParams { period: Some(14) });
		let res = supersmoother_with_kernel(&input, kernel)?;
		assert_eq!(res.values.len(), candles.close.len());
		if res.values.len() > 240 {
			for (i, &val) in res.values[240..].iter().enumerate() {
				assert!(
					val.is_finite(),
					"[{}] Found unexpected NaN at out-index {}",
					test_name,
					240 + i
				);
			}
		}
		Ok(())
	}
	macro_rules! generate_all_supersmoother_tests {
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
	// Check for poison values in single output - only runs in debug mode
	#[cfg(debug_assertions)]
	fn check_supersmoother_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		// Test multiple parameter combinations to catch uninitialized memory reads
		let test_periods = vec![3, 7, 10, 14, 20, 30, 50, 100, 200];

		for period in test_periods {
			let params = SuperSmootherParams { period: Some(period) };
			let input = SuperSmootherInput::from_candles(&candles, "close", params);

			// Skip if period is too large for the data
			if period > candles.close.len() {
				continue;
			}

			let output = supersmoother_with_kernel(&input, kernel)?;

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
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} with period {}",
						test_name, val, bits, i, period
					);
				}

				// Check for init_matrix_prefixes poison (0x22222222_22222222)
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} with period {}",
						test_name, val, bits, i, period
					);
				}

				// Check for make_uninit_matrix poison (0x33333333_33333333)
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} with period {}",
						test_name, val, bits, i, period
					);
				}
			}
		}

		Ok(())
	}

	// Release mode stub - does nothing
	#[cfg(not(debug_assertions))]
	fn check_supersmoother_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(())
	}

	generate_all_supersmoother_tests!(
		check_supersmoother_partial_params,
		check_supersmoother_accuracy,
		check_supersmoother_default_candles,
		check_supersmoother_zero_period,
		check_supersmoother_period_exceeds_length,
		check_supersmoother_very_small_dataset,
		check_supersmoother_reinput,
		check_supersmoother_nan_handling,
		check_supersmoother_no_poison
	);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let output = SuperSmootherBatchBuilder::new()
			.kernel(kernel)
			.apply_candles(&c, "close")?;
		let def = SuperSmootherParams::default();
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

	// Check for poison values in batch output - only runs in debug mode
	#[cfg(debug_assertions)]
	fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		// Test multiple different batch configurations to catch edge cases
		let batch_configs = vec![
			(3, 10, 2),    // Small periods with small step
			(10, 30, 10),  // Medium periods
			(20, 100, 20), // Large periods
			(5, 5, 1),     // Single period (edge case)
			(2, 50, 1),    // Many periods starting from minimum
		];

		for (start, end, step) in batch_configs {
			// Skip if the largest period exceeds data length
			if end > c.close.len() {
				continue;
			}

			let output = SuperSmootherBatchBuilder::new()
				.kernel(kernel)
				.period_range(start, end, step)
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
				let period = if row < output.combos.len() {
					output.combos[row].period.unwrap_or(0)
				} else {
					0
				};

				// Check for alloc_with_nan_prefix poison (0x11111111_11111111)
				if bits == 0x11111111_11111111 {
					panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with period {} in batch ({}, {}, {})",
                        test, val, bits, row, col, idx, period, start, end, step
                    );
				}

				// Check for init_matrix_prefixes poison (0x22222222_22222222)
				if bits == 0x22222222_22222222 {
					panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {}) with period {} in batch ({}, {}, {})",
                        test, val, bits, row, col, idx, period, start, end, step
                    );
				}

				// Check for make_uninit_matrix poison (0x33333333_33333333)
				if bits == 0x33333333_33333333 {
					panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with period {} in batch ({}, {}, {})",
                        test, val, bits, row, col, idx, period, start, end, step
                    );
				}
			}
		}

		Ok(())
	}

	// Release mode stub - does nothing
	#[cfg(not(debug_assertions))]
	fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(())
	}

	gen_batch_tests!(check_batch_default_row);
	gen_batch_tests!(check_batch_no_poison);
}

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

#[cfg(feature = "python")]
#[pyfunction(name = "supersmoother")]
#[pyo3(signature = (data, period, kernel=None))]
/// Compute the SuperSmoother filter (2-pole) of the input data.
///
/// SuperSmoother is a double-pole smoothing filter that reduces high-frequency noise
/// while preserving trend information. It provides better smoothing than EMA with
/// similar lag characteristics.
///
/// Parameters:
/// -----------
/// data : np.ndarray
///     Input data array (float64).
/// period : int
///     Smoothing period, must be >= 1 and <= data length.
/// kernel : str, optional
///     Computation kernel to use: 'auto', 'scalar', 'avx2', 'avx512'.
///     Default is 'auto' which auto-detects the best available.
///
/// Returns:
/// --------
/// np.ndarray
///     Array of SuperSmoother values, same length as input.
///
/// Raises:
/// -------
/// ValueError
///     If inputs are invalid (period = 0, exceeds data length, all NaN, etc).
pub fn supersmoother_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	period: usize,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
	use numpy::{IntoPyArray, PyArrayMethods};

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, false)?; // false for single operations

	let params = SuperSmootherParams { period: Some(period) };
	let ss_in = SuperSmootherInput::from_slice(slice_in, params);

	// Get Vec<f64> from Rust function and zero-copy transfer to NumPy
	let result_vec: Vec<f64> = py
		.allow_threads(|| supersmoother_with_kernel(&ss_in, kern).map(|o| o.values))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "SuperSmootherStream")]
pub struct SuperSmootherStreamPy {
	stream: SuperSmootherStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl SuperSmootherStreamPy {
	#[new]
	fn new(period: usize) -> PyResult<Self> {
		let params = SuperSmootherParams { period: Some(period) };
		let stream = SuperSmootherStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(SuperSmootherStreamPy { stream })
	}

	/// Updates the stream with a new value and returns the calculated SuperSmoother value.
	/// Returns `None` if the buffer is not yet full.
	fn update(&mut self, value: f64) -> Option<f64> {
		self.stream.update(value, None)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "supersmoother_batch")]
#[pyo3(signature = (data, period_start, period_end, period_step, kernel=None))]
/// Compute SuperSmoother for multiple periods in a single pass.
///
/// Parameters:
/// -----------
/// data : np.ndarray
///     Input data array (float64).
/// period_start : int
///     Starting period value.
/// period_end : int
///     Ending period value (inclusive).
/// period_step : int
///     Step size between periods.
/// kernel : str, optional
///     Computation kernel: 'auto', 'scalar', 'avx2', 'avx512'.
///     Default is 'auto' which auto-detects the best available.
///
/// Returns:
/// --------
/// dict
///     Dictionary with 'values' (2D array, rows=periods, cols=data length)
///     and 'periods' array.
pub fn supersmoother_batch_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	period_start: usize,
	period_end: usize,
	period_step: usize,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;

	let slice_in = data.as_slice()?;

	let sweep = SuperSmootherBatchRange {
		period: (period_start, period_end, period_step),
	};

	let combos = expand_grid(&sweep);
	let rows = combos.len();
	let cols = slice_in.len();

	// Pre-allocate output array (correct for batch operations)
	let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let slice_out = unsafe { out_arr.as_slice_mut()? };

	let kern = validate_kernel(kernel, true)?;

	// Compute without GIL, writing directly to the NumPy array
	let combos = py
		.allow_threads(|| {
			// Handle kernel selection for batch operations
			let kernel = match kern {
				Kernel::Auto => detect_best_batch_kernel(),
				k => k,
			};

			// Map batch kernels to regular kernels
			let simd = match kernel {
				Kernel::Avx512Batch => Kernel::Avx512,
				Kernel::Avx2Batch => Kernel::Avx2,
				Kernel::ScalarBatch => Kernel::Scalar,
				_ => kernel,
			};

			supersmoother_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
		})
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	// Build result dictionary
	let dict = PyDict::new(py);
	dict.set_item("values", out_arr.reshape((rows, cols))?)?;

	// For single-parameter indicators like SuperSmoother
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

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct SuperSmootherBatchConfig {
	pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct SuperSmootherBatchJsOutput {
	pub values: Vec<f64>,
	pub combos: Vec<SuperSmootherParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn supersmoother_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
	let params = SuperSmootherParams { period: Some(period) };
	let input = SuperSmootherInput::from_slice(data, params);

	// Allocate output buffer once
	let mut output = vec![0.0; data.len()];

	// Compute directly into output buffer
	supersmoother_into_slice(&mut output, &input, Kernel::Auto).map_err(|e| JsValue::from_str(&e.to_string()))?;

	Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = supersmoother_batch)]
pub fn supersmoother_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	let config: SuperSmootherBatchConfig =
		serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

	let sweep = SuperSmootherBatchRange {
		period: config.period_range,
	};

	let kernel = detect_best_batch_kernel();
	let output =
		supersmoother_batch_inner(data, &sweep, kernel, false).map_err(|e| JsValue::from_str(&e.to_string()))?;

	let js_output = SuperSmootherBatchJsOutput {
		values: output.values,
		combos: output.combos,
		rows: output.rows,
		cols: output.cols,
	};

	serde_wasm_bindgen::to_value(&js_output).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

// Keep the old function for backward compatibility but mark as deprecated
#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[deprecated(since = "1.0.0", note = "Use supersmoother_batch instead")]
pub fn supersmoother_batch_js(
	data: &[f64],
	period_start: usize,
	period_end: usize,
	period_step: usize,
) -> Result<Vec<f64>, JsValue> {
	let sweep = SuperSmootherBatchRange {
		period: (period_start, period_end, period_step),
	};

	// Use the existing batch function with parallel=false for WASM
	supersmoother_batch_inner(data, &sweep, Kernel::Scalar, false)
		.map(|output| output.values)
		.map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn supersmoother_batch_metadata_js(
	period_start: usize,
	period_end: usize,
	period_step: usize,
) -> Result<Vec<f64>, JsValue> {
	let sweep = SuperSmootherBatchRange {
		period: (period_start, period_end, period_step),
	};

	let combos = expand_grid(&sweep);
	let metadata: Vec<f64> = combos.iter().map(|combo| combo.period.unwrap() as f64).collect();

	Ok(metadata)
}

// ================== Zero-Copy WASM Functions ==================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn supersmoother_alloc(len: usize) -> *mut f64 {
	// Allocate memory for input/output buffer
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec); // Prevent deallocation
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn supersmoother_free(ptr: *mut f64, len: usize) {
	// Free allocated memory
	if !ptr.is_null() {
		unsafe {
			let _ = Vec::from_raw_parts(ptr, len, len);
		}
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn supersmoother_into(in_ptr: *const f64, out_ptr: *mut f64, len: usize, period: usize) -> Result<(), JsValue> {
	// Check for null pointers
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}

	unsafe {
		// Create slice from pointer
		let data = std::slice::from_raw_parts(in_ptr, len);

		// Validate inputs
		if period == 0 || period > len {
			return Err(JsValue::from_str("Invalid period"));
		}

		// Create input
		let params = SuperSmootherParams { period: Some(period) };
		let input = SuperSmootherInput::from_slice(data, params);

		if in_ptr == out_ptr as *const f64 {
			// CRITICAL: Aliasing check - in-place operation
			let mut temp = vec![0.0; len];
			supersmoother_into_slice(&mut temp, &input, Kernel::Auto).map_err(|e| JsValue::from_str(&e.to_string()))?;
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			out.copy_from_slice(&temp);
		} else {
			// Direct write to output buffer
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			supersmoother_into_slice(out, &input, Kernel::Auto).map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn supersmoother_batch_into(
	in_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period_start: usize,
	period_end: usize,
	period_step: usize,
) -> Result<usize, JsValue> {
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("null pointer passed to supersmoother_batch_into"));
	}

	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);

		let sweep = SuperSmootherBatchRange {
			period: (period_start, period_end, period_step),
		};

		let combos = expand_grid(&sweep);
		let rows = combos.len();

		// Create mutable output slice
		let out_slice = std::slice::from_raw_parts_mut(out_ptr, rows * len);

		// Use batch_inner_into for direct writes
		let kernel = detect_best_batch_kernel();
		supersmoother_batch_inner_into(data, &sweep, kernel, false, out_slice)
			.map_err(|e| JsValue::from_str(&e.to_string()))?;

		Ok(rows)
	}
}
