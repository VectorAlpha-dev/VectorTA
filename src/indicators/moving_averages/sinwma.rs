//! # Sine Weighted Moving Average (SINWMA)
//!
//! A specialized weighted moving average that applies sine coefficients to create
//! a symmetric bell-shaped weighting curve. The weights follow the pattern
//! `sin((k+1)π/(period+1))` for k=0 to period-1, forming a sine window that
//! peaks at the center of the data window. This creates a smooth, balanced
//! weighting that reduces edge effects while maintaining responsiveness.
//!
//! ## Parameters
//! - **period**: Number of data points to include in each weighted sum (defaults to 14).
//!
//! ## Errors
//! - **AllValuesNaN**: sinwma: All input data values are `NaN`.
//! - **InvalidPeriod**: sinwma: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: sinwma: Not enough valid data points for the requested `period`.
//! - **ZeroSumSines**: sinwma: Sum of sines is zero or too close to zero.
//!
//! ## Returns
//! - **`Ok(SinWmaOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(SinWmaError)`** otherwise.
//!
//! ## Developer Notes
//! - **AVX2 kernel**: ❌ Stub - calls scalar implementation
//! - **AVX512 kernel**: ❌ Stub - both short and long variants call scalar
//! - **Streaming update**: ⚠️ O(n) - `dot_ring()` iterates through all period weights
//! - **Memory optimization**: ✅ Uses `alloc_with_nan_prefix` for zero-copy output allocation
//! - **Current status**: Functional with pre-computed weights but missing SIMD optimizations
//! - **Optimization opportunities**:
//!   - Implement AVX2 kernel for vectorized weighted sum operations
//!   - Implement AVX512 short/long kernels for wider vector processing
//!   - Streaming update could potentially maintain incremental weighted sum for O(1)
//!   - Weighted sum calculations are ideal for SIMD parallelization

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
use std::f64::consts::PI;
use std::mem::MaybeUninit;
use thiserror::Error;

#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
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

impl<'a> AsRef<[f64]> for SinWmaInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			SinWmaData::Slice(slice) => slice,
			SinWmaData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

#[derive(Debug, Clone)]
pub enum SinWmaData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct SinWmaOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct SinWmaParams {
	pub period: Option<usize>,
}

impl Default for SinWmaParams {
	fn default() -> Self {
		Self { period: Some(14) }
	}
}

#[derive(Debug, Clone)]
pub struct SinWmaInput<'a> {
	pub data: SinWmaData<'a>,
	pub params: SinWmaParams,
}

impl<'a> SinWmaInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: SinWmaParams) -> Self {
		Self {
			data: SinWmaData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slice(sl: &'a [f64], p: SinWmaParams) -> Self {
		Self {
			data: SinWmaData::Slice(sl),
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", SinWmaParams::default())
	}
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(14)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct SinWmaBuilder {
	period: Option<usize>,
	kernel: Kernel,
}

impl Default for SinWmaBuilder {
	fn default() -> Self {
		Self {
			period: None,
			kernel: Kernel::Auto,
		}
	}
}

impl SinWmaBuilder {
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
	pub fn apply(self, c: &Candles) -> Result<SinWmaOutput, SinWmaError> {
		let p = SinWmaParams { period: self.period };
		let i = SinWmaInput::from_candles(c, "close", p);
		sinwma_with_kernel(&i, self.kernel)
	}

	#[inline(always)]
	pub fn apply_slice(self, d: &[f64]) -> Result<SinWmaOutput, SinWmaError> {
		let p = SinWmaParams { period: self.period };
		let i = SinWmaInput::from_slice(d, p);
		sinwma_with_kernel(&i, self.kernel)
	}

	#[inline(always)]
	pub fn into_stream(self) -> Result<SinWmaStream, SinWmaError> {
		let p = SinWmaParams { period: self.period };
		SinWmaStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum SinWmaError {
	#[error("sinwma: No data provided (empty slice).")]
	EmptyInputData,
	#[error("sinwma: All values are NaN.")]
	AllValuesNaN,
	#[error("sinwma: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("sinwma: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("sinwma: Sum of sines is zero or too close to zero. sum_sines = {sum_sines}")]
	ZeroSumSines { sum_sines: f64 },
}

#[inline]
pub fn sinwma(input: &SinWmaInput) -> Result<SinWmaOutput, SinWmaError> {
	sinwma_with_kernel(input, Kernel::Auto)
}

pub fn sinwma_with_kernel(input: &SinWmaInput, kernel: Kernel) -> Result<SinWmaOutput, SinWmaError> {
	let (data, weights, period, first, chosen) = sinwma_prepare(input, kernel)?;

	let mut out = alloc_with_nan_prefix(data.len(), first + period - 1);

	sinwma_compute_into(data, &weights, period, first, chosen, &mut out);

	Ok(SinWmaOutput { values: out })
}

/// Write SINWMA values directly to output slice - no allocations.
/// The output slice must be the same length as the input data.
#[inline]
pub fn sinwma_into_slice(dst: &mut [f64], input: &SinWmaInput, kern: Kernel) -> Result<(), SinWmaError> {
	let (data, weights, period, first, chosen) = sinwma_prepare(input, kern)?;

	// Verify output buffer size matches input
	if dst.len() != data.len() {
		return Err(SinWmaError::InvalidPeriod {
			period: dst.len(),
			data_len: data.len(),
		});
	}

	// Compute SINWMA values directly into dst
	sinwma_compute_into(data, &weights, period, first, chosen, dst);

	// Fill warmup period with NaN
	let warmup_end = first + period - 1;
	for v in &mut dst[..warmup_end] {
		*v = f64::NAN;
	}

	Ok(())
}

#[inline(always)]
fn sinwma_prepare<'a>(
	input: &'a SinWmaInput,
	kernel: Kernel,
) -> Result<
	(
		// data
		&'a [f64],
		// weights
		AVec<f64>,
		// period
		usize,
		// first
		usize,
		// chosen
		Kernel,
	),
	SinWmaError,
> {
	let data: &[f64] = input.as_ref();
	let len = data.len();

	if len == 0 {
		return Err(SinWmaError::EmptyInputData);
	}

	let first = data.iter().position(|x| !x.is_nan()).ok_or(SinWmaError::AllValuesNaN)?;

	let period = input.get_period();

	// Validation checks
	if period == 0 || period > len {
		return Err(SinWmaError::InvalidPeriod { period, data_len: len });
	}
	if (len - first) < period {
		return Err(SinWmaError::NotEnoughValidData {
			needed: period,
			valid: len - first,
		});
	}

	// Build weights once
	let mut weights: AVec<f64> = AVec::with_capacity(CACHELINE_ALIGN, period);
	weights.resize(period, 0.0);
	let mut sum_sines = 0.0;
	for k in 0..period {
		let angle = (k as f64 + 1.0) * PI / (period as f64 + 1.0);
		let val = angle.sin();
		weights[k] = val;
		sum_sines += val;
	}

	if sum_sines.abs() < f64::EPSILON {
		return Err(SinWmaError::ZeroSumSines { sum_sines });
	}
	let inv_sum = 1.0 / sum_sines;
	for w in &mut weights[..] {
		*w *= inv_sum;
	}

	// Kernel auto-detection only once
	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		k => k,
	};

	Ok((data, weights, period, first, chosen))
}

#[inline(always)]
fn sinwma_compute_into(data: &[f64], weights: &[f64], period: usize, first: usize, kernel: Kernel, out: &mut [f64]) {
	unsafe {
		match kernel {
			Kernel::Scalar | Kernel::ScalarBatch => sinwma_scalar(data, weights, period, first, out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => sinwma_avx2(data, weights, period, first, out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => sinwma_avx512(data, weights, period, first, out),
			#[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
			Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
				sinwma_scalar(data, weights, period, first, out)
			}
			_ => unreachable!(),
		}
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn sinwma_avx512(data: &[f64], weights: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	if period <= 32 {
		unsafe { sinwma_avx512_short(data, weights, period, first_valid, out) }
	} else {
		unsafe { sinwma_avx512_long(data, weights, period, first_valid, out) }
	}
}

#[inline]
pub fn sinwma_scalar(data: &[f64], weights: &[f64], period: usize, first_val: usize, out: &mut [f64]) {
	assert_eq!(weights.len(), period, "weights.len() must equal `period`");
	assert!(out.len() >= data.len(), "`out` must be at least as long as `data`");

	let p4 = period & !3;

	for i in (first_val + period - 1)..data.len() {
		let start = i + 1 - period;
		let window = &data[start..start + period];

		let mut sum = 0.0;
		for (d4, w4) in window[..p4].chunks_exact(4).zip(weights[..p4].chunks_exact(4)) {
			sum += d4[0] * w4[0] + d4[1] * w4[1] + d4[2] * w4[2] + d4[3] * w4[3];
		}

		for (d, w) in window[p4..].iter().zip(&weights[p4..]) {
			sum += d * w;
		}

		out[i] = sum;
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn sinwma_avx2(data: &[f64], weights: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	// AVX2 stub - forwards to scalar for now
	sinwma_scalar(data, weights, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f,fma")]
pub unsafe fn sinwma_avx512_short(data: &[f64], weights: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	// AVX512 short stub - forwards to scalar for now
	sinwma_scalar(data, weights, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f,fma,avx512dq")]
pub unsafe fn sinwma_avx512_long(data: &[f64], weights: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	// AVX512 long stub - forwards to scalar for now
	sinwma_scalar(data, weights, period, first_valid, out)
}

#[derive(Debug, Clone)]
pub struct SinWmaStream {
	period: usize,
	weights: Vec<f64>,
	buffer: Vec<f64>,
	head: usize,
	filled: bool,
}

impl SinWmaStream {
	pub fn try_new(params: SinWmaParams) -> Result<Self, SinWmaError> {
		let period = params.period.unwrap_or(14);
		if period == 0 {
			return Err(SinWmaError::InvalidPeriod { period, data_len: 0 });
		}

		let mut weights = Vec::with_capacity(period);
		let mut sum_sines = 0.0;
		for k in 0..period {
			let angle = (k as f64 + 1.0) * PI / (period as f64 + 1.0);
			let val = angle.sin();
			weights.push(val);
			sum_sines += val;
		}
		if sum_sines.abs() < f64::EPSILON {
			return Err(SinWmaError::ZeroSumSines { sum_sines });
		}
		for w in &mut weights {
			*w /= sum_sines;
		}

		Ok(Self {
			period,
			weights,
			buffer: vec![f64::NAN; period],
			head: 0,
			filled: false,
		})
	}

	#[inline(always)]
	pub fn update(&mut self, value: f64) -> Option<f64> {
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
	fn dot_ring(&self) -> f64 {
		let mut sum = 0.0;
		let mut idx = self.head;
		for &w in &self.weights {
			sum += w * self.buffer[idx];
			idx = (idx + 1) % self.period;
		}
		sum
	}
}

#[derive(Clone, Debug)]
pub struct SinWmaBatchRange {
	pub period: (usize, usize, usize),
}

impl Default for SinWmaBatchRange {
	fn default() -> Self {
		Self { period: (14, 50, 1) }
	}
}

#[derive(Clone, Debug, Default)]
pub struct SinWmaBatchBuilder {
	range: SinWmaBatchRange,
	kernel: Kernel,
}

impl SinWmaBatchBuilder {
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

	pub fn apply_slice(self, data: &[f64]) -> Result<SinWmaBatchOutput, SinWmaError> {
		sinwma_batch_with_kernel(data, &self.range, self.kernel)
	}

	pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<SinWmaBatchOutput, SinWmaError> {
		SinWmaBatchBuilder::new().kernel(k).apply_slice(data)
	}

	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<SinWmaBatchOutput, SinWmaError> {
		let slice = source_type(c, src);
		self.apply_slice(slice)
	}

	pub fn with_default_candles(c: &Candles) -> Result<SinWmaBatchOutput, SinWmaError> {
		SinWmaBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
	}
}

pub fn sinwma_batch_with_kernel(
	data: &[f64],
	sweep: &SinWmaBatchRange,
	k: Kernel,
) -> Result<SinWmaBatchOutput, SinWmaError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(SinWmaError::InvalidPeriod { period: 0, data_len: 0 }),
	};

	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	sinwma_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct SinWmaBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<SinWmaParams>,
	pub rows: usize,
	pub cols: usize,
}
impl SinWmaBatchOutput {
	pub fn row_for_params(&self, p: &SinWmaParams) -> Option<usize> {
		self.combos
			.iter()
			.position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
	}

	pub fn values_for(&self, p: &SinWmaParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &SinWmaBatchRange) -> Vec<SinWmaParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}

	let periods = axis_usize(r.period);

	let mut out = Vec::with_capacity(periods.len());
	for &p in &periods {
		out.push(SinWmaParams { period: Some(p) });
	}
	out
}

#[inline]
fn round_up8(x: usize) -> usize {
	(x + 7) & !7
}

#[inline(always)]
pub unsafe fn sinwma_row_dispatch(
	data: &[f64], first: usize, period: usize, w_ptr: *const f64, out: &mut [f64], kern: Kernel
) {
	match kern {
		Kernel::Scalar | Kernel::ScalarBatch => sinwma_row_scalar(data, first, period, w_ptr, out),
		#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
		Kernel::Avx2 | Kernel::Avx2Batch => sinwma_row_avx2(data, first, period, w_ptr, out),
		#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
		Kernel::Avx512 | Kernel::Avx512Batch => sinwma_row_avx512(data, first, period, w_ptr, out),
		#[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
		Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
			sinwma_row_scalar(data, first, period, w_ptr, out)
		}
		_ => unreachable!(),
	}
}

#[inline(always)]
pub fn sinwma_batch_slice(
	data: &[f64],
	sweep: &SinWmaBatchRange,
	kern: Kernel,
) -> Result<SinWmaBatchOutput, SinWmaError> {
	sinwma_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn sinwma_batch_par_slice(
	data: &[f64],
	sweep: &SinWmaBatchRange,
	kern: Kernel,
) -> Result<SinWmaBatchOutput, SinWmaError> {
	sinwma_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn sinwma_batch_inner(
	data: &[f64],
	sweep: &SinWmaBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<SinWmaBatchOutput, SinWmaError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(SinWmaError::InvalidPeriod { period: 0, data_len: 0 });
	}

	if data.is_empty() {
		return Err(SinWmaError::EmptyInputData);
	}

	let first = data.iter().position(|x| !x.is_nan()).ok_or(SinWmaError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if data.len() - first < max_p {
		return Err(SinWmaError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}

	let rows = combos.len();
	let cols = data.len();
	let warm: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap() - 1).collect();

	let mut raw = make_uninit_matrix(rows, cols);
	init_matrix_prefixes(&mut raw, cols, &warm);

	// Precompute normalized sine weights into flat buffer with padded stride
	let stride = round_up8(max_p);
	let cap = rows * stride;
	let mut flat_w = AVec::<f64>::with_capacity(CACHELINE_ALIGN, cap);
	flat_w.resize(cap, 0.0);

	for (row, prm) in combos.iter().enumerate() {
		let p = prm.period.unwrap();
		let base = row * stride;
		let mut sum = 0.0;
		for k in 0..p {
			let a = (k as f64 + 1.0) * PI / (p as f64 + 1.0);
			let v = a.sin();
			flat_w[base + k] = v;
			sum += v;
		}
		let inv = 1.0 / sum;
		for k in 0..p {
			flat_w[base + k] *= inv;
		}
	}

	// Closure that fills one row
	let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
		let p = combos[row].period.unwrap();
		let w_ptr = flat_w.as_ptr().add(row * stride);
		let dst = core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());
		sinwma_row_dispatch(data, first, p, w_ptr, dst, kern);
	};

	// Run every row directly into `raw`
	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			raw.par_chunks_mut(cols).enumerate().for_each(|(r, sl)| do_row(r, sl));
		}

		#[cfg(target_arch = "wasm32")]
		{
			for (r, sl) in raw.chunks_mut(cols).enumerate() {
				do_row(r, sl);
			}
		}
	} else {
		for (r, sl) in raw.chunks_mut(cols).enumerate() {
			do_row(r, sl);
		}
	}

	// Finalize matrix like ALMA (no transmute)
	let mut guard = core::mem::ManuallyDrop::new(raw);
	let values = unsafe {
		Vec::from_raw_parts(guard.as_mut_ptr() as *mut f64, guard.len(), guard.capacity())
	};

	Ok(SinWmaBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
fn sinwma_batch_inner_into(
	data: &[f64],
	sweep: &SinWmaBatchRange,
	kern: Kernel,
	parallel: bool,
	out: &mut [f64],
) -> Result<Vec<SinWmaParams>, SinWmaError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(SinWmaError::InvalidPeriod { period: 0, data_len: 0 });
	}

	if data.is_empty() {
		return Err(SinWmaError::EmptyInputData);
	}

	let first = data.iter().position(|x| !x.is_nan()).ok_or(SinWmaError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if data.len() - first < max_p {
		return Err(SinWmaError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}

	let rows = combos.len();
	let cols = data.len();
	let warm: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap() - 1).collect();

	// Cast output to MaybeUninit and prefix NaNs
	let out_mu = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len()) };
	init_matrix_prefixes(out_mu, cols, &warm);

	// Precompute normalized sine weights into flat buffer with padded stride
	let stride = round_up8(max_p);
	let cap = rows * stride;
	let mut flat_w = AVec::<f64>::with_capacity(CACHELINE_ALIGN, cap);
	flat_w.resize(cap, 0.0);

	for (row, prm) in combos.iter().enumerate() {
		let p = prm.period.unwrap();
		let base = row * stride;
		let mut sum = 0.0;
		for k in 0..p {
			let a = (k as f64 + 1.0) * PI / (p as f64 + 1.0);
			let v = a.sin();
			flat_w[base + k] = v;
			sum += v;
		}
		let inv = 1.0 / sum;
		for k in 0..p {
			flat_w[base + k] *= inv;
		}
	}

	// Row closure now reuses the table
	let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
		let p = combos[row].period.unwrap();
		let w_ptr = flat_w.as_ptr().add(row * stride);
		let dst = core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());
		sinwma_row_dispatch(data, first, p, w_ptr, dst, kern);
	};

	// Run every row directly into the output buffer
	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			out_mu
				.par_chunks_mut(cols)
				.enumerate()
				.for_each(|(r, sl)| do_row(r, sl));
		}
		#[cfg(target_arch = "wasm32")]
		{
			for (r, sl) in out_mu.chunks_mut(cols).enumerate() {
				do_row(r, sl);
			}
		}
	} else {
		for (r, sl) in out_mu.chunks_mut(cols).enumerate() {
			do_row(r, sl);
		}
	}

	Ok(combos)
}

#[inline(always)]
pub unsafe fn sinwma_row_scalar(data: &[f64], first: usize, period: usize, w_ptr: *const f64, out: &mut [f64]) {
	let p4 = period & !3;
	for i in (first + period - 1)..data.len() {
		let start = i + 1 - period;
		let mut sum = 0.0;
		for k in (0..p4).step_by(4) {
			let w = std::slice::from_raw_parts(w_ptr.add(k), 4);
			let d = &data[start + k..start + k + 4];
			sum += d[0] * w[0] + d[1] * w[1] + d[2] * w[2] + d[3] * w[3];
		}
		for k in p4..period {
			sum += *data.get_unchecked(start + k) * *w_ptr.add(k);
		}
		out[i] = sum;
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn sinwma_row_avx2(data: &[f64], first: usize, period: usize, w_ptr: *const f64, out: &mut [f64]) {
	sinwma_row_scalar(data, first, period, w_ptr, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f,fma,avx512dq")]
pub unsafe fn sinwma_row_avx512(data: &[f64], first: usize, period: usize, w_ptr: *const f64, out: &mut [f64]) {
	if period <= 32 {
		sinwma_row_avx512_short(data, first, period, w_ptr, out);
	} else {
		sinwma_row_avx512_long(data, first, period, w_ptr, out);
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f,fma")]
pub unsafe fn sinwma_row_avx512_short(data: &[f64], first: usize, period: usize, w_ptr: *const f64, out: &mut [f64]) {
	sinwma_row_scalar(data, first, period, w_ptr, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f,fma,avx512dq")]
pub unsafe fn sinwma_row_avx512_long(data: &[f64], first: usize, period: usize, w_ptr: *const f64, out: &mut [f64]) {
	sinwma_row_scalar(data, first, period, w_ptr, out)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;
	#[cfg(feature = "proptest")]
	use proptest::prelude::*;

	fn check_sinwma_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let default_params = SinWmaParams { period: None };
		let input = SinWmaInput::from_candles(&candles, "close", default_params);
		let output = sinwma_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_sinwma_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = SinWmaInput::from_candles(&candles, "close", SinWmaParams { period: Some(14) });
		let result = sinwma_with_kernel(&input, kernel)?;
		let expected_last_five = [
			59376.72903536103,
			59300.76862770367,
			59229.27622157621,
			59178.48781774477,
			59154.66580703081,
		];
		let start = result.values.len().saturating_sub(5);
		for (i, &val) in result.values[start..].iter().enumerate() {
			let diff = (val - expected_last_five[i]).abs();
			assert!(
				diff < 1e-6,
				"[{}] SINWMA {:?} mismatch at idx {}: got {}, expected {}",
				test_name,
				kernel,
				i,
				val,
				expected_last_five[i]
			);
		}
		Ok(())
	}

	fn check_sinwma_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = SinWmaInput::with_default_candles(&candles);
		match input.data {
			SinWmaData::Candles { source, .. } => assert_eq!(source, "close"),
			_ => panic!("Expected SinWmaData::Candles"),
		}
		let output = sinwma_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_sinwma_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let input_data = [10.0, 20.0, 30.0];
		let params = SinWmaParams { period: Some(0) };
		let input = SinWmaInput::from_slice(&input_data, params);
		let res = sinwma_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] SINWMA should fail with zero period", test_name);
		Ok(())
	}

	fn check_sinwma_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data_small = [10.0, 20.0, 30.0];
		let params = SinWmaParams { period: Some(10) };
		let input = SinWmaInput::from_slice(&data_small, params);
		let res = sinwma_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] SINWMA should fail with period exceeding length",
			test_name
		);
		Ok(())
	}

	fn check_sinwma_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let single_point = [42.0];
		let params = SinWmaParams { period: Some(14) };
		let input = SinWmaInput::from_slice(&single_point, params);
		let res = sinwma_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] SINWMA should fail with insufficient data",
			test_name
		);
		Ok(())
	}

	fn check_sinwma_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let first_params = SinWmaParams { period: Some(14) };
		let first_input = SinWmaInput::from_candles(&candles, "close", first_params);
		let first_result = sinwma_with_kernel(&first_input, kernel)?;

		let second_params = SinWmaParams { period: Some(5) };
		let second_input = SinWmaInput::from_slice(&first_result.values, second_params);
		let second_result = sinwma_with_kernel(&second_input, kernel)?;

		assert_eq!(second_result.values.len(), first_result.values.len());
		for &val in second_result.values.iter().skip(240) {
			assert!(val.is_finite());
		}
		Ok(())
	}

	fn check_sinwma_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = SinWmaInput::from_candles(&candles, "close", SinWmaParams { period: Some(14) });
		let res = sinwma_with_kernel(&input, kernel)?;
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

	fn check_sinwma_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let period = 14;

		let input = SinWmaInput::from_candles(&candles, "close", SinWmaParams { period: Some(period) });
		let batch_output = sinwma_with_kernel(&input, kernel)?.values;

		let mut stream = SinWmaStream::try_new(SinWmaParams { period: Some(period) })?;

		let mut stream_values = Vec::with_capacity(candles.close.len());
		for &price in &candles.close {
			match stream.update(price) {
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
				"[{}] SINWMA streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
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
	fn check_sinwma_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		// Test multiple parameter combinations to increase coverage
		let test_periods = vec![5, 10, 14, 20, 30, 50];

		for period in test_periods {
			let params = SinWmaParams { period: Some(period) };
			let input = SinWmaInput::from_candles(&candles, "close", params);
			let output = sinwma_with_kernel(&input, kernel)?;

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
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} (period={})",
						test_name, val, bits, i, period
					);
				}

				// Check for init_matrix_prefixes poison (0x22222222_22222222)
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} (period={})",
						test_name, val, bits, i, period
					);
				}

				// Check for make_uninit_matrix poison (0x33333333_33333333)
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} (period={})",
						test_name, val, bits, i, period
					);
				}
			}
		}

		Ok(())
	}

	// Release mode stub - does nothing
	#[cfg(not(debug_assertions))]
	fn check_sinwma_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(())
	}

	#[cfg(feature = "proptest")]
	#[allow(clippy::float_cmp)]
	fn check_sinwma_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		use proptest::prelude::*;
		skip_if_unsupported!(kernel, test_name);

		// Strategy: generate period first, then data of appropriate length
		let strat = (1usize..=100)
			.prop_flat_map(|period| {
				(
					// Data must be at least as long as period
					prop::collection::vec(
						(-1e5f64..1e5f64).prop_filter("finite", |x| x.is_finite()),
						period..=500,
					),
					Just(period),
				)
			});

		proptest::test_runner::TestRunner::default()
			.run(&strat, |(data, period)| {
				let params = SinWmaParams { period: Some(period) };
				let input = SinWmaInput::from_slice(&data, params);

				let SinWmaOutput { values: out } = sinwma_with_kernel(&input, kernel).unwrap();
				let SinWmaOutput { values: ref_out } = sinwma_with_kernel(&input, Kernel::Scalar).unwrap();

				// Property 1: Output length matches input length
				prop_assert_eq!(out.len(), data.len(), "Output length should match input length");

				// Property 2: NaN values in warmup period
				// SINWMA has warmup of period - 1 samples
				let warmup_end = period - 1;
				for i in 0..warmup_end.min(data.len()) {
					prop_assert!(
						out[i].is_nan(),
						"[{}] Expected NaN at index {} during warmup (period={})",
						test_name,
						i,
						period
					);
				}

				// Property 3: Values after warmup are within window bounds
				for i in warmup_end..data.len() {
					let y = out[i];
					
					// Skip if NaN (shouldn't happen after warmup but be safe)
					if y.is_nan() {
						continue;
					}

					// Get the window that contributed to this output
					let window_start = i + 1 - period;
					let window = &data[window_start..=i];
					
					let lo = window.iter().cloned().fold(f64::INFINITY, f64::min);
					let hi = window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
					
					// Allow small tolerance for floating point
					let tolerance = 1e-9 + (hi - lo).abs() * 1e-12;
					prop_assert!(
						y >= lo - tolerance && y <= hi + tolerance,
						"[{}] idx {}: value {} not in window bounds [{}, {}] (period={})",
						test_name,
						i,
						y,
						lo,
						hi,
						period
					);
				}

				// Property 4: Constant input produces constant output
				if data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-12) && !data.is_empty() {
					for i in warmup_end..data.len() {
						if !out[i].is_nan() {
							prop_assert!(
								(out[i] - data[0]).abs() <= 1e-9,
								"[{}] Constant input should produce constant output: expected {}, got {} at index {}",
								test_name,
								data[0],
								out[i],
								i
							);
						}
					}
				}

				// Property 5: Period=1 special case
				if period == 1 {
					// With period=1, sine weight is sin(π/2) = 1.0, normalized to 1.0
					// So output should equal input after warmup (which is 0 for period=1)
					for i in 0..data.len() {
						if !out[i].is_nan() && !data[i].is_nan() {
							prop_assert!(
								(out[i] - data[i]).abs() <= 1e-12,
								"[{}] Period=1 should pass through input: expected {}, got {} at index {}",
								test_name,
								data[i],
								out[i],
								i
							);
						}
					}
				}

				// Property 6: Kernel consistency
				// Since AVX2/AVX512 currently forward to scalar, all should match exactly
				for i in 0..data.len() {
					if out[i].is_nan() && ref_out[i].is_nan() {
						continue;
					}
					
					// Check for exact bit-level equality since they use same implementation
					let y_bits = out[i].to_bits();
					let r_bits = ref_out[i].to_bits();
					
					prop_assert_eq!(
						y_bits,
						r_bits,
						"[{}] Kernel consistency failed at index {}: {:?} gives {}, Scalar gives {}",
						test_name,
						i,
						kernel,
						out[i],
						ref_out[i]
					);
				}

				// Property 7: Verify sine weights properties
				// The sine weights should sum to 1.0 after normalization
				// We can indirectly verify this by checking that the output is a weighted average
				// For a window of all positive values, output should be positive
				// For a window of all negative values, output should be negative
				for i in warmup_end..data.len() {
					if out[i].is_nan() {
						continue;
					}
					
					let window_start = i + 1 - period;
					let window = &data[window_start..=i];
					
					let all_positive = window.iter().all(|&x| x > 0.0);
					let all_negative = window.iter().all(|&x| x < 0.0);
					
					if all_positive {
						prop_assert!(
							out[i] > 0.0,
							"[{}] All positive window should produce positive output, got {} at index {}",
							test_name,
							out[i],
							i
						);
					}
					
					if all_negative {
						prop_assert!(
							out[i] < 0.0,
							"[{}] All negative window should produce negative output, got {} at index {}",
							test_name,
							out[i],
							i
						);
					}
				}

				// Property 8: Verify center-weighted behavior
				// SINWMA uses sine weights that form a bell curve peaking in the middle
				// The weights follow sin((k+1)π/(period+1)) for k=0 to period-1
				// This means the output should be biased toward middle values of the window
				if data.len() >= period * 2 {
					for i in (warmup_end + period)..data.len().min(warmup_end + period * 3) {
						if out[i].is_nan() {
							continue;
						}
						
						let window_start = i + 1 - period;
						let window = &data[window_start..=i];
						
						// Only test if window has significant variation
						let window_min = window.iter().cloned().fold(f64::INFINITY, f64::min);
						let window_max = window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
						let range = window_max - window_min;
						
						if range > 1.0 {
							// Get values from different parts of the window
							let middle_idx = period / 2;
							let middle_value = window[middle_idx];
							let first_value = window[0];
							let last_value = window[period - 1];
							
							// For a clearly ascending or descending pattern through the window,
							// the output should be closer to the middle than to the extremes
							let clearly_ascending = first_value < middle_value && middle_value < last_value 
								&& (last_value - first_value) > range * 0.8;
							let clearly_descending = first_value > middle_value && middle_value > last_value
								&& (first_value - last_value) > range * 0.8;
							
							if clearly_ascending || clearly_descending {
								// Output should be closer to middle than to either extreme
								let dist_to_middle = (out[i] - middle_value).abs();
								let dist_to_first = (out[i] - first_value).abs();
								let dist_to_last = (out[i] - last_value).abs();
								
								// The output should be closer to the middle value than to the extremes
								// Allow some tolerance since weights don't completely ignore edges
								prop_assert!(
									dist_to_middle < dist_to_first.min(dist_to_last) * 1.2,
									"[{}] idx {}: SINWMA output {} should be closer to middle {} than to extremes [{}, {}] (period={})",
									test_name,
									i,
									out[i],
									middle_value,
									first_value,
									last_value,
									period
								);
							}
						}
					}
				}

				Ok(())
			})?;

		Ok(())
	}

	macro_rules! generate_all_sinwma_tests {
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

	generate_all_sinwma_tests!(
		check_sinwma_partial_params,
		check_sinwma_accuracy,
		check_sinwma_default_candles,
		check_sinwma_zero_period,
		check_sinwma_period_exceeds_length,
		check_sinwma_very_small_dataset,
		check_sinwma_reinput,
		check_sinwma_nan_handling,
		check_sinwma_streaming,
		check_sinwma_no_poison
	);

	#[cfg(feature = "proptest")]
	generate_all_sinwma_tests!(check_sinwma_property);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let output = SinWmaBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;

		let def = SinWmaParams::default();
		let row = output.values_for(&def).expect("default row missing");

		assert_eq!(row.len(), c.close.len());

		let expected = [
			59376.72903536103,
			59300.76862770367,
			59229.27622157621,
			59178.48781774477,
			59154.66580703081,
		];
		let start = row.len() - 5;
		for (i, &v) in row[start..].iter().enumerate() {
			assert!(
				(v - expected[i]).abs() < 1e-6,
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

		// Test multiple batch configurations to increase coverage
		let test_configs = vec![
			(5, 15, 5),   // Small periods
			(10, 30, 10), // Medium periods
			(20, 50, 15), // Large periods
			(2, 10, 2),   // Edge case: very small periods
		];

		for (start, end, step) in test_configs {
			let output = SinWmaBatchBuilder::new()
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
				let period = output.combos[row].period.unwrap();

				// Check for alloc_with_nan_prefix poison (0x11111111_11111111)
				if bits == 0x11111111_11111111 {
					panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {}, period={})",
                        test, val, bits, row, col, idx, period
                    );
				}

				// Check for init_matrix_prefixes poison (0x22222222_22222222)
				if bits == 0x22222222_22222222 {
					panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {}, period={})",
                        test, val, bits, row, col, idx, period
                    );
				}

				// Check for make_uninit_matrix poison (0x33333333_33333333)
				if bits == 0x33333333_33333333 {
					panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {}, period={})",
                        test, val, bits, row, col, idx, period
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

#[cfg(feature = "python")]
#[pyfunction(name = "sinwma")]
#[pyo3(signature = (data, period, kernel=None))]
/// Compute the Sine Weighted Moving Average (SINWMA) of the input data.
///
/// Parameters
/// ----------
/// data : numpy.ndarray
///     Input data array
/// period : int
///     The period for the SINWMA calculation (must be >= 1)
/// kernel : str, optional
///     Kernel to use: 'auto' (default), 'scalar', 'avx2', 'avx512'
///
/// Returns
/// -------
/// numpy.ndarray
///     SINWMA values
///
/// Raises
/// ------
/// ValueError
///     If period is invalid or data is insufficient
pub fn sinwma_py<'py>(
	py: Python<'py>,
	data: PyReadonlyArray1<'py, f64>,
	period: usize,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, false)?;

	let params = SinWmaParams { period: Some(period) };
	let input = SinWmaInput::from_slice(slice_in, params);

	let result_vec: Vec<f64> = py
		.allow_threads(|| sinwma_with_kernel(&input, kern).map(|o| o.values))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "SinWmaStream")]
/// Streaming Sine Weighted Moving Average calculator.
///
/// This class maintains internal state to calculate SINWMA values
/// incrementally as new data points arrive.
pub struct SinWmaStreamPy {
	stream: SinWmaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl SinWmaStreamPy {
	#[new]
	/// Create a new SINWMA stream calculator.
	///
	/// Parameters
	/// ----------
	/// period : int
	///     The period for the SINWMA calculation
	///
	/// Raises
	/// ------
	/// ValueError
	///     If period is invalid
	fn new(period: usize) -> PyResult<Self> {
		let params = SinWmaParams { period: Some(period) };
		let stream = SinWmaStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(SinWmaStreamPy { stream })
	}

	/// Updates the stream with a new value and returns the calculated SINWMA value.
	/// Returns `None` if the buffer is not yet full.
	fn update(&mut self, value: f64) -> Option<f64> {
		self.stream.update(value)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "sinwma_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
/// Compute SINWMA for multiple periods efficiently in a single pass.
///
/// Parameters
/// ----------
/// data : numpy.ndarray
///     Input data array
/// period_range : tuple[int, int, int]
///     Range of periods as (start, end, step)
/// kernel : str, optional
///     Kernel to use: 'auto' (default), 'scalar', 'avx2', 'avx512'
///
/// Returns
/// -------
/// dict
///     Dictionary with 'values' (2D array) and 'periods' (1D array)
///
/// Raises
/// ------
/// ValueError
///     If parameters are invalid or data is insufficient
pub fn sinwma_batch_py<'py>(
	py: Python<'py>,
	data: PyReadonlyArray1<'py, f64>,
	period_range: (usize, usize, usize),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;

	let slice_in = data.as_slice()?;

	let sweep = SinWmaBatchRange { period: period_range };

	let combos = expand_grid(&sweep);
	let rows = combos.len();
	let cols = slice_in.len();

	let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let slice_out = unsafe { out_arr.as_slice_mut()? };

	let kern = validate_kernel(kernel, true)?;

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
			sinwma_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
		})
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	let dict = PyDict::new(py);
	dict.set_item("values", out_arr.reshape((rows, cols))?)?;
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
#[wasm_bindgen]
pub fn sinwma_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
	let params = SinWmaParams { period: Some(period) };
	let input = SinWmaInput::from_slice(data, params);

	// Allocate output buffer once
	let mut output = vec![0.0; data.len()];

	// Compute directly into output buffer
	sinwma_into_slice(&mut output, &input, Kernel::Auto).map_err(|e| JsValue::from_str(&e.to_string()))?;

	Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn sinwma_alloc(len: usize) -> *mut f64 {
	// Allocate memory for input/output buffer
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec); // Prevent deallocation
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn sinwma_free(ptr: *mut f64, len: usize) {
	// Free allocated memory
	if !ptr.is_null() {
		unsafe {
			let _ = Vec::from_raw_parts(ptr, len, len);
		}
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn sinwma_into(in_ptr: *const f64, out_ptr: *mut f64, len: usize, period: usize) -> Result<(), JsValue> {
	// Check for null pointers
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("null pointer passed to sinwma_into"));
	}

	unsafe {
		// Create slice from pointer
		let data = std::slice::from_raw_parts(in_ptr, len);

		// Validate inputs
		if period == 0 || period > len {
			return Err(JsValue::from_str("Invalid period"));
		}

		let params = SinWmaParams { period: Some(period) };
		let input = SinWmaInput::from_slice(data, params);

		// Check for aliasing (same memory location)
		if in_ptr == out_ptr {
			// Need temporary buffer for in-place operation
			let mut temp = vec![0.0; len];
			sinwma_into_slice(&mut temp, &input, Kernel::Auto).map_err(|e| JsValue::from_str(&e.to_string()))?;

			// Copy result back to output
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			out.copy_from_slice(&temp);
		} else {
			// Direct computation into output buffer
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			sinwma_into_slice(out, &input, Kernel::Auto).map_err(|e| JsValue::from_str(&e.to_string()))?;
		}

		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct SinWmaBatchConfig {
	pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct SinWmaBatchJsOutput {
	pub values: Vec<f64>,
	pub periods: Vec<usize>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = sinwma_batch)]
pub fn sinwma_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	let config: SinWmaBatchConfig =
		serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

	let sweep = SinWmaBatchRange {
		period: config.period_range,
	};

	let output = sinwma_batch_with_kernel(data, &sweep, Kernel::Auto).map_err(|e| JsValue::from_str(&e.to_string()))?;

	let periods: Vec<usize> = output.combos.iter().map(|c| c.period.unwrap()).collect();

	let js_output = SinWmaBatchJsOutput {
		values: output.values,
		periods,
		rows: output.rows,
		cols: output.cols,
	};

	serde_wasm_bindgen::to_value(&js_output).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

// Keep the old API for backward compatibility
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn sinwma_batch_js(
	data: &[f64],
	period_start: usize,
	period_end: usize,
	period_step: usize,
) -> Result<Vec<f64>, JsValue> {
	let sweep = SinWmaBatchRange {
		period: (period_start, period_end, period_step),
	};

	// Use the existing batch function with parallel=false for WASM
	sinwma_batch_with_kernel(data, &sweep, Kernel::Auto)
		.map(|output| output.values)
		.map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn sinwma_batch_metadata_js(period_start: usize, period_end: usize, period_step: usize) -> Vec<u32> {
	let sweep = SinWmaBatchRange {
		period: (period_start, period_end, period_step),
	};

	let combos = expand_grid(&sweep);
	combos.iter().map(|p| p.period.unwrap() as u32).collect()
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn sinwma_batch_rows_cols_js(
	period_start: usize,
	period_end: usize,
	period_step: usize,
	data_len: usize,
) -> Vec<u32> {
	let sweep = SinWmaBatchRange {
		period: (period_start, period_end, period_step),
	};

	let combos = expand_grid(&sweep);
	vec![combos.len() as u32, data_len as u32]
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn sinwma_batch_into(
	in_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period_start: usize,
	period_end: usize,
	period_step: usize,
) -> Result<usize, JsValue> {
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("null pointer passed to sinwma_batch_into"));
	}
	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);
		let sweep = SinWmaBatchRange { period: (period_start, period_end, period_step) };
		let combos = expand_grid(&sweep);
		let rows = combos.len();
		let cols = len;

		let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);
		// Map batch kernel -> base kernel and compute directly into `out`
		let simd = match detect_best_batch_kernel() {
			Kernel::Avx512Batch => Kernel::Avx512,
			Kernel::Avx2Batch   => Kernel::Avx2,
			_                   => Kernel::Scalar,
		};
		sinwma_batch_inner_into(data, &sweep, simd, false, out)
			.map_err(|e| JsValue::from_str(&e.to_string()))?;
		Ok(rows)
	}
}
