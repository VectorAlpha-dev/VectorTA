//! # Vortex Indicator (VI)
//!
//! Computes the positive (VI+) and negative (VI-) vortex indicators based on a specified period.
//! Supports batch computation, builder pattern, parameter sweeps, AVX2/AVX512 feature stubs, and streaming API.
//!
//! ## Parameters
//! - **period**: Lookback window size (default: 14).
//!
//! ## Errors
//! - **EmptyData**: vi: Input slices are empty.
//! - **InvalidPeriod**: vi: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: vi: Not enough valid data points for the requested `period`.
//! - **AllValuesNaN**: vi: All input data values are `NaN`.
//!
//! ## Returns
//! - **`Ok(ViOutput)`** on success, with `.plus` and `.minus` of length matching the input.
//! - **`Err(ViError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum ViData<'a> {
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
pub struct ViOutput {
	pub plus: Vec<f64>,
	pub minus: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct ViParams {
	pub period: Option<usize>,
}

impl Default for ViParams {
	fn default() -> Self {
		Self { period: Some(14) }
	}
}

#[derive(Debug, Clone)]
pub struct ViInput<'a> {
	pub data: ViData<'a>,
	pub params: ViParams,
}

impl<'a> ViInput<'a> {
	#[inline]
	pub fn from_candles(candles: &'a Candles, params: ViParams) -> Self {
		Self {
			data: ViData::Candles { candles },
			params,
		}
	}
	#[inline]
	pub fn from_slices(high: &'a [f64], low: &'a [f64], close: &'a [f64], params: ViParams) -> Self {
		Self {
			data: ViData::Slices { high, low, close },
			params,
		}
	}
	#[inline]
	pub fn with_default_candles(candles: &'a Candles) -> Self {
		Self {
			data: ViData::Candles { candles },
			params: ViParams::default(),
		}
	}
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(14)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct ViBuilder {
	period: Option<usize>,
	kernel: Kernel,
}

impl Default for ViBuilder {
	fn default() -> Self {
		Self {
			period: None,
			kernel: Kernel::Auto,
		}
	}
}

impl ViBuilder {
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
	pub fn apply(self, c: &Candles) -> Result<ViOutput, ViError> {
		let p = ViParams { period: self.period };
		let i = ViInput::from_candles(c, p);
		vi_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slices(self, high: &[f64], low: &[f64], close: &[f64]) -> Result<ViOutput, ViError> {
		let p = ViParams { period: self.period };
		let i = ViInput::from_slices(high, low, close, p);
		vi_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<ViStream, ViError> {
		let p = ViParams { period: self.period };
		ViStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum ViError {
	#[error("vi: Empty data provided.")]
	EmptyData,
	#[error("vi: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("vi: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("vi: All values are NaN.")]
	AllValuesNaN,
}

#[inline]
pub fn vi(input: &ViInput) -> Result<ViOutput, ViError> {
	vi_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn vi_prepare<'a>(
	input: &'a ViInput,
	kernel: Kernel,
) -> Result<(&'a [f64], &'a [f64], &'a [f64], usize, usize, Kernel), ViError> {
	let (high, low, close) = match &input.data {
		ViData::Candles { candles } => (
			source_type(candles, "high"),
			source_type(candles, "low"),
			source_type(candles, "close"),
		),
		ViData::Slices { high, low, close } => (*high, *low, *close),
	};

	if high.is_empty() || low.is_empty() || close.is_empty() {
		return Err(ViError::EmptyData);
	}
	let len = high.len();
	if len != low.len() || len != close.len() {
		return Err(ViError::EmptyData);
	}

	let period = input.get_period();
	if period == 0 || period > len {
		return Err(ViError::InvalidPeriod { period, data_len: len });
	}

	let first = (0..len)
		.find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
		.ok_or(ViError::AllValuesNaN)?;

	if len - first < period {
		return Err(ViError::NotEnoughValidData { needed: period, valid: len - first });
	}

	let chosen = match kernel { Kernel::Auto => detect_best_kernel(), k => k };
	Ok((high, low, close, period, first, chosen))
}

#[inline(always)]
fn vi_compute_into(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	period: usize,
	first: usize,
	kernel: Kernel,
	plus: &mut [f64],
	minus: &mut [f64],
) {
	unsafe {
		match kernel {
			Kernel::Scalar | Kernel::ScalarBatch => vi_scalar(high, low, close, period, first, plus, minus),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => vi_avx2(high, low, close, period, first, plus, minus),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => vi_avx512(high, low, close, period, first, plus, minus),
			_ => unreachable!(),
		}
	}
}

pub fn vi_with_kernel(input: &ViInput, kernel: Kernel) -> Result<ViOutput, ViError> {
	let (h, l, c, period, first, chosen) = vi_prepare(input, kernel)?;
	let mut plus  = alloc_with_nan_prefix(h.len(), first + period - 1);
	let mut minus = alloc_with_nan_prefix(h.len(), first + period - 1);
	vi_compute_into(h, l, c, period, first, chosen, &mut plus, &mut minus);
	Ok(ViOutput { plus, minus })
}

// General, non-wasm into-slice function matching ALMA pattern
pub fn vi_into_slice(
	dst_plus: &mut [f64],
	dst_minus: &mut [f64],
	input: &ViInput,
	kernel: Kernel,
) -> Result<(), ViError> {
	let (h, l, c, period, first, chosen) = vi_prepare(input, kernel)?;
	if dst_plus.len() != h.len() || dst_minus.len() != h.len() {
		return Err(ViError::InvalidPeriod { period: dst_plus.len(), data_len: h.len() });
	}
	vi_compute_into(h, l, c, period, first, chosen, dst_plus, dst_minus);
	let warm = first + period - 1;
	for i in 0..warm {
		dst_plus[i] = f64::NAN;
		dst_minus[i] = f64::NAN;
	}
	Ok(())
}

#[inline]
pub unsafe fn vi_scalar(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	period: usize,
	first: usize,
	plus: &mut [f64],
	minus: &mut [f64],
) {
	let length = high.len();
	let mut tr = vec![0.0; length];
	let mut vp = vec![0.0; length];
	let mut vm = vec![0.0; length];
	tr[first] = high[first] - low[first];
	for i in (first + 1)..length {
		tr[i] = (high[i] - low[i])
			.max((high[i] - close[i - 1]).abs())
			.max((low[i] - close[i - 1]).abs());
		vp[i] = (high[i] - low[i - 1]).abs();
		vm[i] = (low[i] - high[i - 1]).abs();
	}
	let mut sum_tr = 0.0;
	let mut sum_vp = 0.0;
	let mut sum_vm = 0.0;
	for i in first..(first + period) {
		sum_tr += tr[i];
		sum_vp += vp[i];
		sum_vm += vm[i];
	}
	plus[first + period - 1] = sum_vp / sum_tr;
	minus[first + period - 1] = sum_vm / sum_tr;
	for i in (first + period)..length {
		sum_tr += tr[i] - tr[i - period];
		sum_vp += vp[i] - vp[i - period];
		sum_vm += vm[i] - vm[i - period];
		plus[i] = sum_vp / sum_tr;
		minus[i] = sum_vm / sum_tr;
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vi_avx2(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	period: usize,
	first: usize,
	plus: &mut [f64],
	minus: &mut [f64],
) {
	vi_scalar(high, low, close, period, first, plus, minus);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vi_avx512(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	period: usize,
	first: usize,
	plus: &mut [f64],
	minus: &mut [f64],
) {
	if period <= 32 {
		vi_avx512_short(high, low, close, period, first, plus, minus);
	} else {
		vi_avx512_long(high, low, close, period, first, plus, minus);
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vi_avx512_short(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	period: usize,
	first: usize,
	plus: &mut [f64],
	minus: &mut [f64],
) {
	vi_scalar(high, low, close, period, first, plus, minus);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vi_avx512_long(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	period: usize,
	first: usize,
	plus: &mut [f64],
	minus: &mut [f64],
) {
	vi_scalar(high, low, close, period, first, plus, minus);
}

#[derive(Debug, Clone)]
pub struct ViStream {
	period: usize,
	tr: Vec<f64>,
	vp: Vec<f64>,
	vm: Vec<f64>,
	idx: usize,
	filled: bool,
	sum_tr: f64,
	sum_vp: f64,
	sum_vm: f64,
}

impl ViStream {
	pub fn try_new(params: ViParams) -> Result<Self, ViError> {
		let period = params.period.unwrap_or(14);
		if period == 0 {
			return Err(ViError::InvalidPeriod { period, data_len: 0 });
		}
		Ok(Self {
			period,
			tr: vec![0.0; period],
			vp: vec![0.0; period],
			vm: vec![0.0; period],
			idx: 0,
			filled: false,
			sum_tr: 0.0,
			sum_vp: 0.0,
			sum_vm: 0.0,
		})
	}
	pub fn update(
		&mut self,
		high: f64,
		low: f64,
		close: f64,
		prev_low: f64,
		prev_high: f64,
		prev_close: f64,
	) -> Option<(f64, f64)> {
		let i = self.idx % self.period;
		let tr = (high - low)
			.max((high - prev_close).abs())
			.max((low - prev_close).abs());
		let vp = (high - prev_low).abs();
		let vm = (low - prev_high).abs();
		self.sum_tr += tr - self.tr[i];
		self.sum_vp += vp - self.vp[i];
		self.sum_vm += vm - self.vm[i];
		self.tr[i] = tr;
		self.vp[i] = vp;
		self.vm[i] = vm;
		self.idx += 1;
		if !self.filled && self.idx >= self.period {
			self.filled = true;
		}
		if self.filled {
			Some((self.sum_vp / self.sum_tr, self.sum_vm / self.sum_tr))
		} else {
			None
		}
	}
}

#[derive(Clone, Debug)]
pub struct ViBatchRange {
	pub period: (usize, usize, usize),
}

impl Default for ViBatchRange {
	fn default() -> Self {
		Self { period: (14, 14, 0) }
	}
}

#[derive(Clone, Debug, Default)]
pub struct ViBatchBuilder {
	range: ViBatchRange,
	kernel: Kernel,
}

impl ViBatchBuilder {
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
	pub fn apply_slices(self, high: &[f64], low: &[f64], close: &[f64]) -> Result<ViBatchOutput, ViError> {
		vi_batch_with_kernel(high, low, close, &self.range, self.kernel)
	}
	pub fn apply_candles(self, c: &Candles) -> Result<ViBatchOutput, ViError> {
		let high = source_type(c, "high");
		let low = source_type(c, "low");
		let close = source_type(c, "close");
		self.apply_slices(high, low, close)
	}
}

#[derive(Clone, Debug)]
pub struct ViBatchOutput {
	pub plus: Vec<f64>,
	pub minus: Vec<f64>,
	pub combos: Vec<ViParams>,
	pub rows: usize,
	pub cols: usize,
}
impl ViBatchOutput {
	pub fn row_for_params(&self, p: &ViParams) -> Option<usize> {
		self.combos
			.iter()
			.position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
	}
	pub fn plus_for(&self, p: &ViParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.plus[start..start + self.cols]
		})
	}
	pub fn minus_for(&self, p: &ViParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.minus[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &ViBatchRange) -> Vec<ViParams> {
	let (start, end, step) = r.period;
	if step == 0 || start == end { 
		return vec![ViParams { period: Some(start) }]; 
	}
	(start..=end).step_by(step).map(|p| ViParams { period: Some(p) }).collect()
}

pub fn vi_batch_with_kernel(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	sweep: &ViBatchRange,
	k: Kernel,
) -> Result<ViBatchOutput, ViError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => {
			return Err(ViError::InvalidPeriod { period: 0, data_len: 0 });
		}
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	vi_batch_par_slice(high, low, close, sweep, simd)
}

pub fn vi_batch_slice(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	sweep: &ViBatchRange,
	kern: Kernel,
) -> Result<ViBatchOutput, ViError> {
	vi_batch_inner(high, low, close, sweep, kern, false)
}
pub fn vi_batch_par_slice(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	sweep: &ViBatchRange,
	kern: Kernel,
) -> Result<ViBatchOutput, ViError> {
	vi_batch_inner(high, low, close, sweep, kern, true)
}
#[inline(always)]
fn vi_batch_inner(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	sweep: &ViBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<ViBatchOutput, ViError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(ViError::InvalidPeriod { period: 0, data_len: 0 });
	}
	if high.is_empty() || low.is_empty() || close.is_empty() {
		return Err(ViError::EmptyData);
	}
	let first = (0..high.len())
		.find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
		.ok_or(ViError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if high.len() - first < max_p {
		return Err(ViError::NotEnoughValidData {
			needed: max_p,
			valid: high.len() - first,
		});
	}
	let rows = combos.len();
	let cols = high.len();
	
	// Use uninitialized memory for better performance
	let mut plus_mu = make_uninit_matrix(rows, cols);
	let mut minus_mu = make_uninit_matrix(rows, cols);
	
	// Calculate warmup periods for each parameter combination
	let warm: Vec<usize> = combos
		.iter()
		.map(|c| first + c.period.unwrap() - 1)
		.collect();
	
	// Initialize the prefix with NaN
	init_matrix_prefixes(&mut plus_mu, cols, &warm);
	init_matrix_prefixes(&mut minus_mu, cols, &warm);
	
	// Convert to mutable slices safely
	let mut plus_guard = core::mem::ManuallyDrop::new(plus_mu);
	let mut minus_guard = core::mem::ManuallyDrop::new(minus_mu);
	let plus: &mut [f64] = unsafe {
		core::slice::from_raw_parts_mut(plus_guard.as_mut_ptr() as *mut f64, plus_guard.len())
	};
	let minus: &mut [f64] = unsafe {
		core::slice::from_raw_parts_mut(minus_guard.as_mut_ptr() as *mut f64, minus_guard.len())
	};
	let do_row = |row: usize, plus_row: &mut [f64], minus_row: &mut [f64]| unsafe {
		let period = combos[row].period.unwrap();
		match kern {
			Kernel::Scalar | Kernel::ScalarBatch => vi_row_scalar(high, low, close, first, period, plus_row, minus_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => vi_row_avx2(high, low, close, first, period, plus_row, minus_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => vi_row_avx512(high, low, close, first, period, plus_row, minus_row),
			#[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
			Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
				vi_row_scalar(high, low, close, first, period, plus_row, minus_row)
			}
			_ => unreachable!("Unexpected kernel type: {:?}", kern),
		}
	};
	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			plus.par_chunks_mut(cols)
				.zip(minus.par_chunks_mut(cols))
				.enumerate()
				.for_each(|(row, (p, m))| do_row(row, p, m));
		}

		#[cfg(target_arch = "wasm32")]
		{
			for ((row, p), m) in plus.chunks_mut(cols).enumerate().zip(minus.chunks_mut(cols)) {
				do_row(row, p, m);
			}
		}
	} else {
		for ((row, p), m) in plus.chunks_mut(cols).enumerate().zip(minus.chunks_mut(cols)) {
			do_row(row, p, m);
		}
	}
	// Convert ManuallyDrop back to Vec<f64> for the output
	let plus_vec = unsafe {
		Vec::from_raw_parts(
			plus_guard.as_mut_ptr() as *mut f64,
			plus_guard.len(),
			plus_guard.capacity(),   // Use capacity, not len
		)
	};
	let minus_vec = unsafe {
		Vec::from_raw_parts(
			minus_guard.as_mut_ptr() as *mut f64,
			minus_guard.len(),
			minus_guard.capacity(),  // Use capacity, not len
		)
	};
	// DO NOT call core::mem::forget - ManuallyDrop already prevents drop
	
	Ok(ViBatchOutput {
		plus: plus_vec,
		minus: minus_vec,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
unsafe fn vi_row_scalar(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	first: usize,
	period: usize,
	plus: &mut [f64],
	minus: &mut [f64],
) {
	vi_scalar(high, low, close, period, first, plus, minus);
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn vi_row_avx2(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	first: usize,
	period: usize,
	plus: &mut [f64],
	minus: &mut [f64],
) {
	vi_scalar(high, low, close, period, first, plus, minus);
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn vi_row_avx512(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	first: usize,
	period: usize,
	plus: &mut [f64],
	minus: &mut [f64],
) {
	if period <= 32 {
		vi_row_avx512_short(high, low, close, first, period, plus, minus);
	} else {
		vi_row_avx512_long(high, low, close, first, period, plus, minus);
	}
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn vi_row_avx512_short(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	first: usize,
	period: usize,
	plus: &mut [f64],
	minus: &mut [f64],
) {
	vi_scalar(high, low, close, period, first, plus, minus);
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn vi_row_avx512_long(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	first: usize,
	period: usize,
	plus: &mut [f64],
	minus: &mut [f64],
) {
	vi_scalar(high, low, close, period, first, plus, minus);
}
// Helper function for batch processing that writes directly to output slices
// Now available for both WASM and Python
fn vi_batch_inner_into(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	sweep: &ViBatchRange,
	kernel: Kernel,
	parallel: bool,
	out_plus: &mut [f64],
	out_minus: &mut [f64],
) -> Result<Vec<ViParams>, ViError> {
	let combos = expand_grid(&sweep);
	let rows = combos.len();
	let cols = close.len();
	if rows == 0 { return Err(ViError::InvalidPeriod { period: 0, data_len: 0 }); }
	if out_plus.len() != rows * cols || out_minus.len() != rows * cols { return Err(ViError::EmptyData); }

	// find first valid once
	if high.is_empty() || low.is_empty() || close.is_empty() { return Err(ViError::EmptyData); }
	let first = (0..high.len())
		.find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
		.ok_or(ViError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if high.len() - first < max_p {
		return Err(ViError::NotEnoughValidData { needed: max_p, valid: high.len() - first });
	}

	let do_row = |row: usize, p_row: &mut [f64], m_row: &mut [f64]| {
		let period = combos[row].period.unwrap();
		unsafe { 
			match kernel {
				Kernel::Scalar | Kernel::ScalarBatch => vi_scalar(high, low, close, period, first, p_row, m_row),
				#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
				Kernel::Avx2 | Kernel::Avx2Batch => vi_avx2(high, low, close, period, first, p_row, m_row),
				#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
				Kernel::Avx512 | Kernel::Avx512Batch => vi_avx512(high, low, close, period, first, p_row, m_row),
				// For WASM or non-x86_64, fallback to scalar
				_ => vi_scalar(high, low, close, period, first, p_row, m_row),
			}
		}
		for i in 0..(first + period - 1) {
			p_row[i] = f64::NAN;
			m_row[i] = f64::NAN;
		}
	};

	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		out_plus.par_chunks_mut(cols)
			.zip(out_minus.par_chunks_mut(cols))
			.enumerate()
			.for_each(|(row, (p, m))| do_row(row, p, m));
		#[cfg(target_arch = "wasm32")]
		for ((row, p), m) in out_plus.chunks_mut(cols).enumerate().zip(out_minus.chunks_mut(cols)) {
			do_row(row, p, m);
		}
	} else {
		for ((row, p), m) in out_plus.chunks_mut(cols).enumerate().zip(out_minus.chunks_mut(cols)) {
			do_row(row, p, m);
		}
	}
	Ok(combos)
}

// ==========================
// WASM Bindings
// ==========================

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct ViJsResult {
	pub plus: Vec<f64>,
	pub minus: Vec<f64>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vi_js(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	period: usize,
) -> Result<JsValue, JsValue> {
	let mut plus = vec![0.0; high.len()];
	let mut minus = vec![0.0; high.len()];
	
	vi_into_slice_wasm(&mut plus, &mut minus, high, low, close, period, detect_best_kernel())
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	let result = ViJsResult { plus, minus };
	
	serde_wasm_bindgen::to_value(&result)
		.map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vi_unified_js(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	period: usize,
) -> Result<Vec<f64>, JsValue> {
	// Preallocate flattened result vector [plus..., minus...]
	let mut result = vec![0.0; high.len() * 2];
	
	// Split into plus and minus slices
	let (plus_slice, minus_slice) = result.split_at_mut(high.len());
	
	// Compute directly into slices
	vi_into_slice_wasm(plus_slice, minus_slice, high, low, close, period, detect_best_kernel())
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	Ok(result)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vi_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len * 2); // Allocate for both plus and minus
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vi_free(ptr: *mut f64, len: usize) {
	unsafe {
		let _ = Vec::from_raw_parts(ptr, len * 2, len * 2);
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vi_into(
	high_ptr: *const f64,
	low_ptr: *const f64,
	close_ptr: *const f64,
	plus_ptr: *mut f64,
	minus_ptr: *mut f64,
	len: usize,
	period: usize,
) -> Result<(), JsValue> {
	if high_ptr.is_null() || low_ptr.is_null() || close_ptr.is_null() || plus_ptr.is_null() || minus_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}
	
	unsafe {
		let high = std::slice::from_raw_parts(high_ptr, len);
		let low = std::slice::from_raw_parts(low_ptr, len);
		let close = std::slice::from_raw_parts(close_ptr, len);
		
		// Direct computation into output slices
		let plus_out = std::slice::from_raw_parts_mut(plus_ptr, len);
		let minus_out = std::slice::from_raw_parts_mut(minus_ptr, len);
		
		vi_into_slice_wasm(plus_out, minus_out, high, low, close, period, detect_best_kernel())
			.map_err(|e| JsValue::from_str(&e.to_string()))?;
		
		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct ViBatchConfig {
	pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct ViBatchJsOutput {
	pub plus: Vec<f64>,
	pub minus: Vec<f64>,
	pub periods: Vec<usize>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = vi_batch)]
pub fn vi_batch_js(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	config: JsValue,
) -> Result<JsValue, JsValue> {
	let config: ViBatchConfig = serde_wasm_bindgen::from_value(config)
		.map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
	
	// Execute batch - use vi_batch_with_kernel to properly resolve the kernel
	let sweep = ViBatchRange { period: config.period_range };
	let output = vi_batch_with_kernel(high, low, close, &sweep, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	// Extract periods from combos
	let periods: Vec<usize> = output.combos.iter()
		.map(|p| p.period.unwrap_or(14))
		.collect();
	
	let js_output = ViBatchJsOutput {
		plus: output.plus,
		minus: output.minus,
		periods,
		rows: output.rows,
		cols: output.cols,
	};
	
	serde_wasm_bindgen::to_value(&js_output)
		.map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vi_batch_into(
	high_ptr: *const f64,
	low_ptr: *const f64,
	close_ptr: *const f64,
	plus_ptr: *mut f64,
	minus_ptr: *mut f64,
	len: usize,
	period_start: usize,
	period_end: usize,
	period_step: usize,
) -> Result<usize, JsValue> {
	if high_ptr.is_null() || low_ptr.is_null() || close_ptr.is_null() || plus_ptr.is_null() || minus_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}
	
	unsafe {
		let high = std::slice::from_raw_parts(high_ptr, len);
		let low = std::slice::from_raw_parts(low_ptr, len);
		let close = std::slice::from_raw_parts(close_ptr, len);
		
		let sweep = ViBatchRange {
			period: (period_start, period_end, period_step),
		};
		
		// Calculate number of combinations
		let combos = expand_grid(&sweep);
		let rows = combos.len();
		let cols = len;
		
		let plus_out = std::slice::from_raw_parts_mut(plus_ptr, rows * cols);
		let minus_out = std::slice::from_raw_parts_mut(minus_ptr, rows * cols);
		
		// Use the vi_batch_inner_into function
		let _ = vi_batch_inner_into(high, low, close, &sweep, Kernel::Auto, false, plus_out, minus_out)
			.map_err(|e| JsValue::from_str(&e.to_string()))?;
		
		Ok(rows)
	}
}

// WASM-specific vi_into_slice wrapper for compatibility
#[cfg(feature = "wasm")]
pub fn vi_into_slice_wasm(
	dst_plus: &mut [f64],
	dst_minus: &mut [f64],
	high: &[f64],
	low: &[f64],
	close: &[f64],
	period: usize,
	kern: Kernel,
) -> Result<(), ViError> {
	let params = ViParams { period: Some(period) };
	let input = ViInput::from_slices(high, low, close, params);
	vi_into_slice(dst_plus, dst_minus, &input, kern)
}


#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_vi_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let default_params = ViParams { period: None };
		let input = ViInput::from_candles(&candles, default_params);
		let output = vi_with_kernel(&input, kernel)?;
		assert_eq!(output.plus.len(), candles.close.len());
		assert_eq!(output.minus.len(), candles.close.len());
		Ok(())
	}
	fn check_vi_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = ViInput::from_candles(&candles, ViParams::default());
		let result = vi_with_kernel(&input, kernel)?;
		let expected_last_five_plus = [
			0.9970238095238095,
			0.9871071716357775,
			0.9464453759945247,
			0.890897412369242,
			0.9206478557604156,
		];
		let expected_last_five_minus = [
			1.0097117794486214,
			1.04174053182917,
			1.1152365471811105,
			1.181684712791338,
			1.1894672506875827,
		];
		let n = result.plus.len();
		let plus_slice = &result.plus[n - 5..];
		let minus_slice = &result.minus[n - 5..];
		for (i, &val) in plus_slice.iter().enumerate() {
			let expected = expected_last_five_plus[i];
			assert!(
				(val - expected).abs() < 1e-8,
				"[{}] VI+ mismatch at idx {}: got {}, expected {}",
				test_name,
				i,
				val,
				expected
			);
		}
		for (i, &val) in minus_slice.iter().enumerate() {
			let expected = expected_last_five_minus[i];
			assert!(
				(val - expected).abs() < 1e-8,
				"[{}] VI- mismatch at idx {}: got {}, expected {}",
				test_name,
				i,
				val,
				expected
			);
		}
		Ok(())
	}
	fn check_vi_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = ViInput::with_default_candles(&candles);
		let output = vi_with_kernel(&input, kernel)?;
		assert_eq!(output.plus.len(), candles.close.len());
		assert_eq!(output.minus.len(), candles.close.len());
		Ok(())
	}
	fn check_vi_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let input_data = [10.0, 20.0, 30.0];
		let params = ViParams { period: Some(0) };
		let input = ViInput::from_slices(&input_data, &input_data, &input_data, params);
		let res = vi_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] VI should fail with zero period", test_name);
		Ok(())
	}
	fn check_vi_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data_small = [10.0, 20.0, 30.0];
		let params = ViParams { period: Some(10) };
		let input = ViInput::from_slices(&data_small, &data_small, &data_small, params);
		let res = vi_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] VI should fail with period exceeding length",
			test_name
		);
		Ok(())
	}
	fn check_vi_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let single_point = [42.0];
		let params = ViParams { period: Some(14) };
		let input = ViInput::from_slices(&single_point, &single_point, &single_point, params);
		let res = vi_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] VI should fail with insufficient data", test_name);
		Ok(())
	}
	fn check_vi_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = ViInput::from_candles(&candles, ViParams::default());
		let res = vi_with_kernel(&input, kernel)?;
		assert_eq!(res.plus.len(), candles.close.len());
		if res.plus.len() > 20 {
			for (i, &val) in res.plus[20..].iter().enumerate() {
				assert!(
					!val.is_nan(),
					"[{}] Found unexpected NaN at out-index {}",
					test_name,
					20 + i
				);
			}
		}
		Ok(())
	}

	#[cfg(debug_assertions)]
	fn check_vi_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		// Define comprehensive parameter combinations
		let test_params = vec![
			ViParams::default(), // period: 14
			ViParams { period: Some(1) },
			ViParams { period: Some(2) },
			ViParams { period: Some(5) },
			ViParams { period: Some(7) },
			ViParams { period: Some(10) },
			ViParams { period: Some(20) },
			ViParams { period: Some(30) },
			ViParams { period: Some(50) },
			ViParams { period: Some(100) },
			ViParams { period: Some(200) },
		];

		for (param_idx, params) in test_params.iter().enumerate() {
			let input = ViInput::from_candles(&candles, params.clone());
			let output = vi_with_kernel(&input, kernel)?;

			// Check plus array for poison values
			for (i, &val) in output.plus.iter().enumerate() {
				if val.is_nan() {
					continue; // NaN values are expected during warmup
				}

				let bits = val.to_bits();

				// Check all three poison patterns
				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in plus array \
						 with params: period={} (param set {})",
						test_name,
						val,
						bits,
						i,
						params.period.unwrap_or(14),
						param_idx
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in plus array \
						 with params: period={} (param set {})",
						test_name,
						val,
						bits,
						i,
						params.period.unwrap_or(14),
						param_idx
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in plus array \
						 with params: period={} (param set {})",
						test_name,
						val,
						bits,
						i,
						params.period.unwrap_or(14),
						param_idx
					);
				}
			}

			// Check minus array for poison values
			for (i, &val) in output.minus.iter().enumerate() {
				if val.is_nan() {
					continue; // NaN values are expected during warmup
				}

				let bits = val.to_bits();

				// Check all three poison patterns
				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in minus array \
						 with params: period={} (param set {})",
						test_name,
						val,
						bits,
						i,
						params.period.unwrap_or(14),
						param_idx
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in minus array \
						 with params: period={} (param set {})",
						test_name,
						val,
						bits,
						i,
						params.period.unwrap_or(14),
						param_idx
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in minus array \
						 with params: period={} (param set {})",
						test_name,
						val,
						bits,
						i,
						params.period.unwrap_or(14),
						param_idx
					);
				}
			}
		}

		Ok(())
	}

	#[cfg(not(debug_assertions))]
	fn check_vi_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(()) // No-op in release builds
	}
	
	#[cfg(feature = "proptest")]
	#[allow(clippy::float_cmp)]
	fn check_vi_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		use proptest::prelude::*;
		skip_if_unsupported!(kernel, test_name);
		
		// Strategy to generate valid OHLC data with more variability
		let strat = (2usize..=100)
			.prop_flat_map(|period| {
				// First generate a length for all vectors
				(period + 50..400).prop_flat_map(move |len| {
					(
						// Generate base prices and volatility factors with same length
						prop::collection::vec(
							(50.0f64..500.0f64).prop_filter("finite", |x| x.is_finite()),
							len,
						),
						prop::collection::vec(
							(0.001f64..0.05f64), // Variable volatility from 0.1% to 5%
							len,
						),
						prop::collection::vec(
							(0.0f64..1.0f64), // Variable close position within range
							len,
						),
						Just(period),
					)
				})
			});
		
		proptest::test_runner::TestRunner::default()
			.run(&strat, |(base_prices, volatilities, close_positions, period)| {
				// Generate more realistic OHLC data with variability
				let mut high = Vec::with_capacity(base_prices.len());
				let mut low = Vec::with_capacity(base_prices.len());
				let mut close = Vec::with_capacity(base_prices.len());
				
				// Ensure all vectors have the same length
				assert_eq!(base_prices.len(), volatilities.len());
				assert_eq!(base_prices.len(), close_positions.len());
				
				for i in 0..base_prices.len() {
					let price = base_prices[i];
					let vol = volatilities[i];
					let close_pos = close_positions[i];
					
					// Variable range based on volatility
					let range = price * vol;
					let h = price + range * (0.3 + vol * 2.0); // High varies with volatility
					let l = price - range * (0.3 + vol * 2.0); // Low varies with volatility
					let c = l + (h - l) * close_pos; // Close at variable position
					
					high.push(h);
					low.push(l);
					close.push(c);
				}
				
				let params = ViParams { period: Some(period) };
				let input = ViInput::from_slices(&high, &low, &close, params.clone());
				
				// Get outputs from the kernel being tested and scalar reference
				let ViOutput { plus: out_plus, minus: out_minus } = 
					vi_with_kernel(&input, kernel).unwrap();
				let ViOutput { plus: ref_plus, minus: ref_minus } = 
					vi_with_kernel(&input, Kernel::Scalar).unwrap();
				
				// Property 1: VI+ and VI- must be non-negative (they're ratios of absolute values)
				for i in 0..out_plus.len() {
					if !out_plus[i].is_nan() {
						prop_assert!(
							out_plus[i] >= -1e-9,
							"[{}] VI+ negative at idx {}: {}",
							test_name, i, out_plus[i]
						);
					}
					if !out_minus[i].is_nan() {
						prop_assert!(
							out_minus[i] >= -1e-9,
							"[{}] VI- negative at idx {}: {}",
							test_name, i, out_minus[i]
						);
					}
				}
				
				// Property 2: Warmup period handling - first (period - 1) values should be NaN
				let first_valid = (0..high.len())
					.find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
					.unwrap_or(0);
				let warmup_end = first_valid + period - 1;
				
				for i in 0..warmup_end.min(out_plus.len()) {
					prop_assert!(
						out_plus[i].is_nan(),
						"[{}] Expected NaN during warmup at idx {}, got {}",
						test_name, i, out_plus[i]
					);
					prop_assert!(
						out_minus[i].is_nan(),
						"[{}] Expected NaN during warmup at idx {}, got {}",
						test_name, i, out_minus[i]
					);
				}
				
				// Property 3: Kernel consistency - different kernels should produce same results
				for i in warmup_end..out_plus.len() {
					let plus_bits = out_plus[i].to_bits();
					let ref_plus_bits = ref_plus[i].to_bits();
					let minus_bits = out_minus[i].to_bits();
					let ref_minus_bits = ref_minus[i].to_bits();
					
					if !out_plus[i].is_finite() || !ref_plus[i].is_finite() {
						prop_assert!(
							plus_bits == ref_plus_bits,
							"[{}] VI+ finite/NaN mismatch at idx {}: {} vs {}",
							test_name, i, out_plus[i], ref_plus[i]
						);
					} else {
						let ulp_diff = plus_bits.abs_diff(ref_plus_bits);
						prop_assert!(
							(out_plus[i] - ref_plus[i]).abs() <= 1e-9 || ulp_diff <= 4,
							"[{}] VI+ mismatch at idx {}: {} vs {} (ULP={})",
							test_name, i, out_plus[i], ref_plus[i], ulp_diff
						);
					}
					
					if !out_minus[i].is_finite() || !ref_minus[i].is_finite() {
						prop_assert!(
							minus_bits == ref_minus_bits,
							"[{}] VI- finite/NaN mismatch at idx {}: {} vs {}",
							test_name, i, out_minus[i], ref_minus[i]
						);
					} else {
						let ulp_diff = minus_bits.abs_diff(ref_minus_bits);
						prop_assert!(
							(out_minus[i] - ref_minus[i]).abs() <= 1e-9 || ulp_diff <= 4,
							"[{}] VI- mismatch at idx {}: {} vs {} (ULP={})",
							test_name, i, out_minus[i], ref_minus[i], ulp_diff
						);
					}
				}
				
				// Property 4: Period=1 special case
				if period == 1 {
					// With period=1, VI calculations are based on single bar
					// The values should be well-defined after warmup
					if warmup_end < out_plus.len() {
						prop_assert!(
							out_plus[warmup_end].is_finite(),
							"[{}] VI+ should be finite for period=1 at idx {}",
							test_name, warmup_end
						);
						prop_assert!(
							out_minus[warmup_end].is_finite(),
							"[{}] VI- should be finite for period=1 at idx {}",
							test_name, warmup_end
						);
					}
				}
				
				// Property 5: Extreme low volatility case
				// When volatility approaches zero (all values very similar)
				let very_low_vol = high.windows(2).all(|w| (w[0] - w[1]).abs() < 0.01) &&
								   low.windows(2).all(|w| (w[0] - w[1]).abs() < 0.01) &&
								   close.windows(2).all(|w| (w[0] - w[1]).abs() < 0.01);
				
				if very_low_vol && warmup_end + 5 < out_plus.len() {
					// In extremely low volatility, VI+ and VI- should be close to 1.0
					// because vortex movements are balanced
					for i in (warmup_end + 5)..out_plus.len().min(warmup_end + 15) {
						if out_plus[i].is_finite() && out_minus[i].is_finite() {
							// Both should be near 1.0 in balanced markets
							prop_assert!(
								out_plus[i] >= 0.5 && out_plus[i] <= 1.5,
								"[{}] VI+ in low volatility should be near 1.0, got {} at idx {}",
								test_name, out_plus[i], i
							);
							prop_assert!(
								out_minus[i] >= 0.5 && out_minus[i] <= 1.5,
								"[{}] VI- in low volatility should be near 1.0, got {} at idx {}",
								test_name, out_minus[i], i
							);
						}
					}
				}
				
				// Property 6: Trending behavior - check across multiple windows
				// Check if general trend direction affects VI relationship
				if warmup_end + period * 2 < out_plus.len() {
					// Check multiple windows for trend consistency
					let mut up_windows = 0;
					let mut down_windows = 0;
					
					let step_size = (period / 2).max(1); // Ensure step is at least 1
					for window_start in (warmup_end..out_plus.len() - period).step_by(step_size) {
						let window_end = (window_start + period).min(out_plus.len());
						
						// Check trend in this window
						let window_close = &close[window_start..window_end];
						if window_close.len() > 2 {
							let trend_up = window_close.windows(2)
								.filter(|w| w[1] > w[0])
								.count() > window_close.len() * 2 / 3;
							let trend_down = window_close.windows(2)
								.filter(|w| w[1] < w[0])
								.count() > window_close.len() * 2 / 3;
							
							if trend_up || trend_down {
								// Calculate average VI values for this window
								let avg_plus: f64 = out_plus[window_start..window_end]
									.iter()
									.filter(|x| x.is_finite())
									.sum::<f64>() / (window_end - window_start) as f64;
								let avg_minus: f64 = out_minus[window_start..window_end]
									.iter()
									.filter(|x| x.is_finite())
									.sum::<f64>() / (window_end - window_start) as f64;
								
								if avg_plus.is_finite() && avg_minus.is_finite() {
									if trend_up && avg_plus > avg_minus {
										up_windows += 1;
									} else if trend_down && avg_minus > avg_plus {
										down_windows += 1;
									}
								}
							}
						}
					}
					
					// VI should correctly identify trend direction in most windows
					let total_trend_windows = up_windows + down_windows;
					if total_trend_windows > 0 {
						let accuracy = (up_windows + down_windows) as f64 / total_trend_windows as f64;
						prop_assert!(
							accuracy >= 0.6, // Should be right at least 60% of the time
							"[{}] VI trend detection accuracy too low: {}",
							test_name, accuracy
						);
					}
				}
				
				// Property 7: Output bounds - VI values typically between 0 and 2.5
				for i in warmup_end..out_plus.len() {
					if out_plus[i].is_finite() {
						prop_assert!(
							out_plus[i] >= 0.0 && out_plus[i] <= 2.5,
							"[{}] VI+ outside typical bounds at idx {}: {}",
							test_name, i, out_plus[i]
						);
					}
					if out_minus[i].is_finite() {
						prop_assert!(
							out_minus[i] >= 0.0 && out_minus[i] <= 2.5,
							"[{}] VI- outside typical bounds at idx {}: {}",
							test_name, i, out_minus[i]
						);
					}
				}
				
				// Property 8: Formula verification for small window
				// Manually calculate VI for a specific point to verify formula
				if period <= 5 && warmup_end + 5 < high.len() && warmup_end >= period {
					let idx = warmup_end;
					
					// Calculate True Range, VM+ and VM- manually
					let mut tr_sum = 0.0;
					let mut vp_sum = 0.0;
					let mut vm_sum = 0.0;
					
					// First bar in period (special case)
					let first_idx = idx + 1 - period;
					tr_sum += high[first_idx] - low[first_idx];
					
					// Remaining bars
					for j in (first_idx + 1)..=idx {
						let tr = (high[j] - low[j])
							.max((high[j] - close[j - 1]).abs())
							.max((low[j] - close[j - 1]).abs());
						let vp = (high[j] - low[j - 1]).abs();
						let vm = (low[j] - high[j - 1]).abs();
						
						tr_sum += tr;
						vp_sum += vp;
						vm_sum += vm;
					}
					
					if tr_sum > 1e-10 {
						let expected_plus = vp_sum / tr_sum;
						let expected_minus = vm_sum / tr_sum;
						
						prop_assert!(
							(out_plus[idx] - expected_plus).abs() < 1e-6,
							"[{}] VI+ formula verification failed at idx {}: {} vs {}",
							test_name, idx, out_plus[idx], expected_plus
						);
						prop_assert!(
							(out_minus[idx] - expected_minus).abs() < 1e-6,
							"[{}] VI- formula verification failed at idx {}: {} vs {}",
							test_name, idx, out_minus[idx], expected_minus
						);
					}
				}
				
				// Property 9: VI+ and VI- relationship
				// The sum of VI+ and VI- relates to overall market activity
				// When one is high, the other tends to be lower (but not always)
				for i in warmup_end..out_plus.len() {
					if out_plus[i].is_finite() && out_minus[i].is_finite() {
						// Sum typically ranges from 1.5 to 2.5 in normal markets
						let sum = out_plus[i] + out_minus[i];
						prop_assert!(
							sum >= 0.8 && sum <= 3.0,
							"[{}] VI+ + VI- sum outside normal range at idx {}: {} ({} + {})",
							test_name, i, sum, out_plus[i], out_minus[i]
						);
						
						// When one VI is very high, the other shouldn't also be very high
						if out_plus[i] > 1.5 {
							prop_assert!(
								out_minus[i] <= 1.5,
								"[{}] Both VI+ and VI- are high at idx {}: {} and {}",
								test_name, i, out_plus[i], out_minus[i]
							);
						}
						if out_minus[i] > 1.5 {
							prop_assert!(
								out_plus[i] <= 1.5,
								"[{}] Both VI- and VI+ are high at idx {}: {} and {}",
								test_name, i, out_minus[i], out_plus[i]
							);
						}
					}
				}
				
				// Property 10: No poison values in debug mode
				#[cfg(debug_assertions)]
				{
					for i in 0..out_plus.len() {
						if !out_plus[i].is_nan() {
							let bits = out_plus[i].to_bits();
							prop_assert!(
								bits != 0x11111111_11111111 && 
								bits != 0x22222222_22222222 && 
								bits != 0x33333333_33333333,
								"[{}] Found poison value in VI+ at idx {}: 0x{:016X}",
								test_name, i, bits
							);
						}
						if !out_minus[i].is_nan() {
							let bits = out_minus[i].to_bits();
							prop_assert!(
								bits != 0x11111111_11111111 && 
								bits != 0x22222222_22222222 && 
								bits != 0x33333333_33333333,
								"[{}] Found poison value in VI- at idx {}: 0x{:016X}",
								test_name, i, bits
							);
						}
					}
				}
				
				Ok(())
			})
			.unwrap();
		
		Ok(())
	}
	
	macro_rules! generate_all_vi_tests {
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
	generate_all_vi_tests!(
		check_vi_partial_params,
		check_vi_accuracy,
		check_vi_default_candles,
		check_vi_zero_period,
		check_vi_period_exceeds_length,
		check_vi_very_small_dataset,
		check_vi_nan_handling,
		check_vi_no_poison
	);
	
	#[cfg(feature = "proptest")]
	generate_all_vi_tests!(check_vi_property);
	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let output = ViBatchBuilder::new().kernel(kernel).apply_candles(&c)?;
		let def = ViParams::default();
		let row = output.plus_for(&def).expect("default row missing");
		assert_eq!(row.len(), c.close.len());
		Ok(())
	}

	#[cfg(debug_assertions)]
	fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		// Test various parameter sweep configurations
		let test_configs = vec![
			// (period_start, period_end, period_step)
			(2, 10, 2),       // Small periods
			(5, 25, 5),       // Medium periods
			(20, 50, 10),     // Large periods
			(2, 5, 1),        // Dense small range
			(14, 14, 0),      // Single value (default period)
			(30, 60, 15),     // Mixed range
			(50, 100, 25),    // Very large periods
			(100, 200, 50),   // Extra large periods
		];

		for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
			let output = ViBatchBuilder::new()
				.kernel(kernel)
				.period_range(p_start, p_end, p_step)
				.apply_candles(&c)?;

			// Check plus array for poison values
			for (idx, &val) in output.plus.iter().enumerate() {
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
						 at row {} col {} (flat index {}) in plus array with params: period={}",
						test,
						cfg_idx,
						val,
						bits,
						row,
						col,
						idx,
						combo.period.unwrap_or(14)
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in plus array with params: period={}",
						test,
						cfg_idx,
						val,
						bits,
						row,
						col,
						idx,
						combo.period.unwrap_or(14)
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in plus array with params: period={}",
						test,
						cfg_idx,
						val,
						bits,
						row,
						col,
						idx,
						combo.period.unwrap_or(14)
					);
				}
			}

			// Check minus array for poison values
			for (idx, &val) in output.minus.iter().enumerate() {
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
						 at row {} col {} (flat index {}) in minus array with params: period={}",
						test,
						cfg_idx,
						val,
						bits,
						row,
						col,
						idx,
						combo.period.unwrap_or(14)
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in minus array with params: period={}",
						test,
						cfg_idx,
						val,
						bits,
						row,
						col,
						idx,
						combo.period.unwrap_or(14)
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in minus array with params: period={}",
						test,
						cfg_idx,
						val,
						bits,
						row,
						col,
						idx,
						combo.period.unwrap_or(14)
					);
				}
			}
		}

		Ok(())
	}

	#[cfg(not(debug_assertions))]
	fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(()) // No-op in release builds
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

// =============================================================================
// Python Bindings  
// =============================================================================
// Python bindings have been moved to src/bindings/python.rs for consistency.
// The following functions are exported and registered there:
// - vi_py: Single VI calculation
// - vi_batch_py: Batch VI calculation 
// - ViStreamPy: Streaming VI interface

#[cfg(feature = "python")]
#[pyfunction(name = "vi")]
#[pyo3(signature = (high, low, close, period, kernel=None))]
pub fn vi_py<'py>(
	py: Python<'py>,
	high: PyReadonlyArray1<'py, f64>,
	low: PyReadonlyArray1<'py, f64>,
	close: PyReadonlyArray1<'py, f64>,
	period: usize,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
	let h = high.as_slice()?;
	let l = low.as_slice()?;
	let c = close.as_slice()?;
	
	// Check for mismatched lengths
	if h.len() != l.len() || h.len() != c.len() {
		return Err(PyValueError::new_err(format!(
			"Input data length mismatch: high={}, low={}, close={}",
			h.len(), l.len(), c.len()
		)));
	}
	
	let params = ViParams { period: Some(period) };
	let input = ViInput::from_slices(h, l, c, params);
	let kern = validate_kernel(kernel, false)?;

	let (plus, minus) = py
		.allow_threads(|| {
			let mut plus = vec![0.0; h.len()];
			let mut minus = vec![0.0; h.len()];
			vi_into_slice(&mut plus, &mut minus, &input, kern)
				.map(|_| (plus, minus))
		})
		.map_err(|e: ViError| PyValueError::new_err(e.to_string()))?;

	let d = PyDict::new(py);
	d.set_item("plus",  plus.into_pyarray(py))?;
	d.set_item("minus", minus.into_pyarray(py))?;
	Ok(d)
}

#[cfg(feature = "python")]
#[pyfunction(name = "vi_batch")]
#[pyo3(signature = (high, low, close, period_range, kernel=None))]
pub fn vi_batch_py<'py>(
	py: Python<'py>,
	high: PyReadonlyArray1<'py, f64>,
	low: PyReadonlyArray1<'py, f64>,
	close: PyReadonlyArray1<'py, f64>,
	period_range: (usize, usize, usize),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
	use numpy::PyArray2;

	let h = high.as_slice()?;
	let l = low.as_slice()?;
	let c = close.as_slice()?;

	let sweep = ViBatchRange { period: period_range };
	let kern = validate_kernel(kernel, true)?;

	// expand grid to get rows/cols
	let combos = expand_grid(&sweep);
	let rows = combos.len();
	let cols = h.len();

	let out_plus = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let out_minus = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let slice_plus  = unsafe { out_plus.as_slice_mut()? };
	let slice_minus = unsafe { out_minus.as_slice_mut()? };

	py.allow_threads(|| {
		let simd = match kern {
			Kernel::Avx512Batch => Kernel::Avx512,
			Kernel::Avx2Batch   => Kernel::Avx2,
			Kernel::ScalarBatch => Kernel::Scalar,
			Kernel::Auto        => detect_best_batch_kernel().to_scalar_equivalent(),
			_ => Kernel::Scalar,
		};
		// helper to coerce Batch→Scalar like ALMA does
		vi_batch_inner_into(h, l, c, &sweep, simd, true, slice_plus, slice_minus)
	})
	.map_err(|e| PyValueError::new_err(e.to_string()))?;

	let d = PyDict::new(py);
	d.set_item("plus",  out_plus.reshape((rows, cols))?)?;
	d.set_item("minus", out_minus.reshape((rows, cols))?)?;
	d.set_item("periods",
		combos.iter().map(|p| p.period.unwrap_or(14) as u64).collect::<Vec<_>>().into_pyarray(py)
	)?;
	Ok(d)
}

#[cfg(feature = "python")]
#[pyclass(name = "ViStream")]
pub struct ViStreamPy { 
	stream: ViStream,
	prev_high: Option<f64>,
	prev_low: Option<f64>,
	prev_close: Option<f64>,
}

#[cfg(feature = "python")]
#[pymethods]
impl ViStreamPy {
	#[new]
	fn new(period: usize) -> PyResult<Self> {
		let s = ViStream::try_new(ViParams { period: Some(period) })
			.map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(Self { 
			stream: s,
			prev_high: None,
			prev_low: None,
			prev_close: None,
		})
	}
	
	// Returns (vi_plus, vi_minus) or None
	fn update(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64)> {
		// VI Stream requires previous values
		match (self.prev_high, self.prev_low, self.prev_close) {
			(Some(ph), Some(pl), Some(pc)) => {
				let result = self.stream.update(high, low, close, pl, ph, pc);
				self.prev_high = Some(high);
				self.prev_low = Some(low);
				self.prev_close = Some(close);
				result
			}
			_ => {
				// First value, just store it
				self.prev_high = Some(high);
				self.prev_low = Some(low);
				self.prev_close = Some(close);
				None
			}
		}
	}
}

// ADD helper on Kernel to match ALMA batch mapping
#[cfg(feature = "python")]
trait BatchToScalar {
	fn to_scalar_equivalent(self) -> Kernel;
}
#[cfg(feature = "python")]
impl BatchToScalar for Kernel {
	fn to_scalar_equivalent(self) -> Kernel {
		match self {
			Kernel::Avx512Batch => Kernel::Avx512,
			Kernel::Avx2Batch   => Kernel::Avx2,
			Kernel::ScalarBatch => Kernel::Scalar,
			Kernel::Auto        => Kernel::Scalar,
			k => k,
		}
	}
}
