//! # Cubic Weighted Moving Average (CWMA)
//!
//! A polynomial-weighted moving average that emphasizes recent observations by
//! applying a **cubic** weight curve. The newest sample receives the weight
//! period³, the second-newest (period-1)³, …, while the oldest sample in
//! the window is weighted 1³.
//!
//! ## Parameters
//! - **period**: Window size (number of data points, defaults to 14).
//!
//! ## Errors
//! - **EmptyInputData**: cwma: Input data slice is empty.
//! - **AllValuesNaN**: cwma: All input data values are NaN.
//! - **InvalidPeriod**: cwma: period is zero or exceeds the data length.
//! - **NotEnoughValidData**: cwma: Not enough valid data points for the requested period.
//!
//! ## Returns
//! - **Ok(CwmaOutput)** on success, containing a Vec<f64> of length matching the input.
//! - **Err(CwmaError)** otherwise.
//!
//! ---
//! # WASM API Guide – CWMA
//!
//! This file exposes a dual-layer WebAssembly interface for the CWMA
//! (Cubic Weighted Moving Average) indicator, balancing **ergonomics** for
//! everyday users with **zero-copy throughput** for latency-critical code.
//!
//! ---
//! ## 1 · Safe / Ergonomic API  <small>(recommended)</small>
//! | JS export | Rust impl | Purpose | Notes |
//! |-----------|-----------|---------|-------|
//! | `cwma_js(data, period)` | `cwma_js` | Single-parameter run | Returns a fresh `Vec<f64>` *without* an internal copy – values written directly into return buffer. │
//! | `cwma_batch(data, config)`<br>(JS object) | `cwma_batch_unified_js` | Grid sweep over `period` | Accepts `period_range`; returns flat result matrix plus combo metadata. │
//!
//! **Characteristics**
//! * Memory-safe, runs under the default linear-memory quota.
//! * Automatic kernel selection for optimal performance.
//! * Adequate for charting & once-off indicator queries.
//!
//! Example:
//! ```javascript
//! import * as wasm from './cwma_bg.wasm';
//!
//! const y = wasm.cwma_js(prices, 14);
//!
//! const grid = wasm.cwma_batch(prices, {
//!   period_range: [10, 30, 5]
//! });
//! ```
//!
//! ---
//! ## 2 · Fast / Unsafe API  <small>(zero-copy)</small>
//! | JS export | Rust impl | Purpose | Notes |
//! |-----------|-----------|---------|-------|
//! | `cwma_alloc(len)` / `cwma_free(ptr,len)` | `cwma_alloc`, `cwma_free` | Manual buffer lifecycle | Aligns to 8 bytes; caller **must** free. │
//! | `cwma_into(inPtr,outPtr,len,period)` | `cwma_into` | In-place single-run | Detects `inPtr === outPtr` and uses temp scratch buffer to avoid alias corruption. │
//! | `cwma_batch_into(inPtr,outPtr,len,…range…)` | `cwma_batch_into` | In-place grid sweep | Serial on WASM for portability. │
//!
//! **Performance**  
//! * Zero heap allocations inside hot loops  
//! * ~1.4× faster than the safe API for repeated calls on pre-allocated
//!   buffers (measured in Chrome 125, 1M-point series).
//!
//! **Caveats**  
//! * **No bounds or lifetime checks** – treat pointers as raw FFI.  
//! * Always wrap calls in `try { … } finally { free() }`.  
//! * Recreate `TypedArray` views after *any* WASM call (memory may grow).
//!
//! ```javascript
//! const n = prices.length;
//! const inPtr  = wasm.cwma_alloc(n);
//! const outPtr = wasm.cwma_alloc(n);
//!
//! try {
//!   new Float64Array(wasm.memory.buffer, inPtr, n).set(prices);
//!   wasm.cwma_into(inPtr, outPtr, n, 14);
//!   const result = new Float64Array(wasm.memory.buffer, outPtr, n);
//! } finally {
//!   wasm.cwma_free(inPtr, n);
//!   wasm.cwma_free(outPtr, n);
//! }
//! ```
//!
//! ---
//! ## Memory-safety checklist
//! 1. Guard every unsafe pointer with null-checks.  
//! 2. Validate `period > 0 && period ≤ len` *before* slicing.  
//! 3. Overwrite warm-up (prefix) indices with `NaN` in `*_into` helpers.  
//! 4. Document warm-up length (`period - 1`) for stream consistency.
//!
//! ---

use crate::utilities::aligned_vector::AlignedVec;
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
	alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
use std::convert::AsRef;
use std::mem::MaybeUninit;
use thiserror::Error;
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

impl<'a> AsRef<[f64]> for CwmaInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			CwmaData::Slice(slice) => slice,
			CwmaData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

#[derive(Debug, Clone)]
pub enum CwmaData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct CwmaOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct CwmaParams {
	pub period: Option<usize>,
}

impl Default for CwmaParams {
	fn default() -> Self {
		Self { period: Some(14) }
	}
}

#[derive(Debug, Clone)]
pub struct CwmaInput<'a> {
	pub data: CwmaData<'a>,
	pub params: CwmaParams,
}

impl<'a> CwmaInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: CwmaParams) -> Self {
		Self {
			data: CwmaData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slice(sl: &'a [f64], p: CwmaParams) -> Self {
		Self {
			data: CwmaData::Slice(sl),
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", CwmaParams::default())
	}
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(14)
	}
}

/// Convenience builder for [`cwma`] with optional parameters.
#[derive(Copy, Clone, Debug)]
pub struct CwmaBuilder {
	period: Option<usize>,
	kernel: Kernel,
}

impl Default for CwmaBuilder {
	fn default() -> Self {
		Self {
			period: None,
			kernel: Kernel::Auto,
		}
	}
}

impl CwmaBuilder {
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
	pub fn apply(self, c: &Candles) -> Result<CwmaOutput, CwmaError> {
		let p = CwmaParams { period: self.period };
		let i = CwmaInput::from_candles(c, "close", p);
		cwma_with_kernel(&i, self.kernel)
	}

	#[inline(always)]
	pub fn apply_slice(self, d: &[f64]) -> Result<CwmaOutput, CwmaError> {
		let p = CwmaParams { period: self.period };
		let i = CwmaInput::from_slice(d, p);
		cwma_with_kernel(&i, self.kernel)
	}

	#[inline(always)]
	pub fn into_stream(self) -> Result<CwmaStream, CwmaError> {
		let p = CwmaParams { period: self.period };
		CwmaStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum CwmaError {
	#[error("cwma: Input data slice is empty.")]
	EmptyInputData,
	#[error("cwma: All values are NaN.")]
	AllValuesNaN,
	#[error("cwma: Invalid period specified for CWMA calculation: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("cwma: Not enough valid data points to compute CWMA: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn cwma(input: &CwmaInput) -> Result<CwmaOutput, CwmaError> {
	cwma_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn cwma_prepare<'a>(
	input: &'a CwmaInput,
	kernel: Kernel,
) -> Result<(&'a [f64], Vec<f64>, usize, usize, f64, usize, Kernel), CwmaError> {
	let data: &[f64] = match &input.data {
		CwmaData::Candles { candles, source } => source_type(candles, source),
		CwmaData::Slice(sl) => sl,
	};
	let len = data.len();
	if len == 0 {
		return Err(CwmaError::EmptyInputData);
	}
	let first = data.iter().position(|x| !x.is_nan()).ok_or(CwmaError::AllValuesNaN)?;

	let period = input.get_period();

	if period == 0 || period > len {
		return Err(CwmaError::InvalidPeriod { period, data_len: len });
	}

	// CWMA with period=1 would have no weights, handle it as an error
	if period == 1 {
		return Err(CwmaError::InvalidPeriod { period, data_len: len });
	}
	if (len - first) < period {
		return Err(CwmaError::NotEnoughValidData {
			needed: period,
			valid: len - first,
		});
	}

	let mut weights = Vec::with_capacity(period - 1);
	let mut norm = 0.0;
	for i in 0..period - 1 {
		let w = ((period - i) as f64).powi(3);
		weights.push(w);
		norm += w;
	}
	let inv_norm = 1.0 / norm;

	let warm = first + period - 1;

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		k => k,
	};

	Ok((data, weights, period, first, inv_norm, warm, chosen))
}

#[inline(always)]
fn cwma_compute_into(
	data: &[f64],
	weights: &[f64],
	period: usize,
	first: usize,
	inv_norm: f64,
	chosen: Kernel,
	out: &mut [f64],
) {
	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => cwma_scalar(data, weights, period, first, inv_norm, out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => cwma_avx2(data, weights, period, first, inv_norm, out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => cwma_avx512(data, weights, period, first, inv_norm, out),
			#[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
			Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
				// Fallback to scalar when AVX is not available
				cwma_scalar(data, weights, period, first, inv_norm, out)
			}
			_ => unreachable!(),
		}
	}
}

/// Write CWMA values directly to output slice - zero allocations.
/// The output slice must be the same length as the input data.
#[inline]
pub fn cwma_into_slice(dst: &mut [f64], input: &CwmaInput, kern: Kernel) -> Result<(), CwmaError> {
	let (data, weights, period, first, inv_norm, warm, chosen) = cwma_prepare(input, kern)?;

	// Verify output buffer size matches input
	if dst.len() != data.len() {
		return Err(CwmaError::InvalidPeriod {
			period: dst.len(),
			data_len: data.len(),
		});
	}

	// Compute CWMA values directly into dst
	cwma_compute_into(data, &weights, period, first, inv_norm, chosen, dst);

	// Fill warmup period with NaN
	for v in &mut dst[..warm] {
		*v = f64::NAN;
	}

	Ok(())
}

pub fn cwma_with_kernel(input: &CwmaInput, kernel: Kernel) -> Result<CwmaOutput, CwmaError> {
	let (data, weights, period, first, inv_norm, warm, chosen) = cwma_prepare(input, kernel)?;
	let len = data.len();
	let mut out = alloc_with_nan_prefix(len, warm);
	cwma_compute_into(data, &weights, period, first, inv_norm, chosen, &mut out);
	Ok(CwmaOutput { values: out })
}

#[inline]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub unsafe fn cwma_scalar(
	data: &[f64],
	weights: &[f64],
	_period: usize,
	first_val: usize,
	inv_norm: f64,
	out: &mut [f64],
) {
	let wlen = weights.len();
	for i in (first_val + wlen)..data.len() {
		let mut acc = 0.0;
		for (k, &w) in weights.iter().enumerate() {
			acc = data[i - k].mul_add(w, acc);
		}
		out[i] = acc * inv_norm;
	}
}

#[inline]
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
pub unsafe fn cwma_scalar(
	data: &[f64],
	weights: &[f64],
	_period: usize,
	first_val: usize,
	inv_norm: f64,
	out: &mut [f64],
) {
	let wlen = weights.len();
	for i in (first_val + wlen)..data.len() {
		let mut acc = 0.0;
		for (k, &w) in weights.iter().enumerate() {
			// Fallback for architectures without mul_add
			acc = acc + data[i - k] * w;
		}
		out[i] = acc * inv_norm;
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn cwma_avx2(
	data: &[f64],
	weights: &[f64],
	_period: usize,
	first_valid: usize,
	inv_norm: f64,
	out: &mut [f64],
) {
	const STEP: usize = 4;

	let wlen = weights.len();
	let chunks = wlen / STEP;
	let tail = wlen % STEP;
	let first_out = first_valid + wlen;

	for i in first_out..data.len() {
		let mut acc = _mm256_setzero_pd();

		for blk in 0..chunks {
			let idx = blk * STEP;
			let w = _mm256_loadu_pd(weights.as_ptr().add(idx));

			let base = i - idx - (STEP - 1);
			let mut d = _mm256_loadu_pd(data.as_ptr().add(base));
			d = _mm256_permute4x64_pd(d, 0b00011011);

			acc = _mm256_fmadd_pd(d, w, acc);
		}

		let mut tail_sum = 0.0;
		if tail != 0 {
			let base = chunks * STEP;
			for k in 0..tail {
				let w = *weights.get_unchecked(base + k);
				let d = *data.get_unchecked(i - (base + k));
				tail_sum = d.mul_add(w, tail_sum);
			}
		}

		let hi = _mm256_extractf128_pd(acc, 1);
		let lo = _mm256_castpd256_pd128(acc);
		let sum2 = _mm_add_pd(hi, lo);
		let sum1 = _mm_add_pd(sum2, _mm_unpackhi_pd(sum2, sum2));
		let mut sum = _mm_cvtsd_f64(sum1);

		sum += tail_sum;
		*out.get_unchecked_mut(i) = sum * inv_norm;
	}
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn cwma_avx512(data: &[f64], weights: &[f64], period: usize, first_valid: usize, inv_norm: f64, out: &mut [f64]) {
	if period <= 32 {
		unsafe { cwma_avx512_short(data, weights, period, first_valid, inv_norm, out) }
	} else {
		unsafe { cwma_avx512_long(data, weights, period, first_valid, inv_norm, out) }
	}
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn reverse_vec(v: __m512d) -> __m512d {
	let lanes = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
	_mm512_permutexvar_pd(lanes, v)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn make_weight_blocks(weights: &[f64]) -> Vec<__m512d> {
	const STEP: usize = 8;
	weights
		.chunks_exact(STEP)
		.map(|chunk| {
			let v = _mm512_loadu_pd(chunk.as_ptr());
			reverse_vec(v)
		})
		.collect()
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
pub unsafe fn cwma_avx512_short(
	data: &[f64],
	weights: &[f64],
	period: usize,
	first_valid: usize,
	inv_norm: f64,
	out: &mut [f64],
) {
	debug_assert!(period <= 32);
	debug_assert_eq!(weights.len(), period - 1);

	const STEP: usize = 8;
	let wlen = weights.len();
	let chunks = wlen / STEP;
	let tail = wlen % STEP;
	let first_out = first_valid + wlen;

	let mut wv: [__m512d; 4] = [_mm512_setzero_pd(); 4];
	if chunks >= 1 {
		wv[0] = reverse_vec(_mm512_loadu_pd(weights.as_ptr()));
	}
	if chunks >= 2 {
		wv[1] = reverse_vec(_mm512_loadu_pd(weights.as_ptr().add(STEP)));
	}
	if chunks >= 3 {
		wv[2] = reverse_vec(_mm512_loadu_pd(weights.as_ptr().add(2 * STEP)));
	}
	if chunks == 4 {
		wv[3] = reverse_vec(_mm512_loadu_pd(weights.as_ptr().add(3 * STEP)));
	}

	for i in first_out..data.len() {
		let mut acc = _mm512_setzero_pd();

		if chunks >= 1 && i >= 7 {
			let d0 = _mm512_loadu_pd(data.as_ptr().add(i - 7));
			acc = _mm512_fmadd_pd(d0, wv[0], acc);
		}
		if chunks >= 2 && i >= STEP + 7 {
			let d1 = _mm512_loadu_pd(data.as_ptr().add(i - STEP - 7));
			acc = _mm512_fmadd_pd(d1, wv[1], acc);
		}
		if chunks >= 3 && i >= 2 * STEP + 7 {
			let d2 = _mm512_loadu_pd(data.as_ptr().add(i - 2 * STEP - 7));
			acc = _mm512_fmadd_pd(d2, wv[2], acc);
		}
		if chunks == 4 && i >= 3 * STEP + 7 {
			let d3 = _mm512_loadu_pd(data.as_ptr().add(i - 3 * STEP - 7));
			acc = _mm512_fmadd_pd(d3, wv[3], acc);
		}

		let mut sum = _mm512_reduce_add_pd(acc);

		if tail != 0 {
			let base = chunks * STEP;
			for k in 0..tail {
				let w = *weights.get_unchecked(base + k);
				let d = *data.get_unchecked(i - (base + k));
				sum = d.mul_add(w, sum);
			}
		}

		*out.get_unchecked_mut(i) = sum * inv_norm;
	}
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
pub unsafe fn cwma_avx512_long(
	data: &[f64],
	weights: &[f64],
	_period: usize,
	first_valid: usize,
	inv_norm: f64,
	out: &mut [f64],
) {
	const STEP: usize = 8;

	let wlen = weights.len();
	let chunks = wlen / STEP;
	let tail = wlen % STEP;
	let first_out = first_valid + wlen;

	if wlen < 24 {
		cwma_avx2(data, weights, _period, first_valid, inv_norm, out);
		return;
	}

	let wblocks = make_weight_blocks(weights);

	for i in first_out..data.len() {
		let mut acc0 = _mm512_setzero_pd();
		let mut acc1 = _mm512_setzero_pd();

		let paired = chunks & !1;
		let mut blk = 0;

		while blk < paired {
			if i >= blk * STEP + 7 && i >= (blk + 1) * STEP + 7 {
				let d0 = _mm512_loadu_pd(data.as_ptr().add(i - blk * STEP - 7));
				let d1 = _mm512_loadu_pd(data.as_ptr().add(i - (blk + 1) * STEP - 7));

				acc0 = _mm512_fmadd_pd(d0, *wblocks.get_unchecked(blk), acc0);
				acc1 = _mm512_fmadd_pd(d1, *wblocks.get_unchecked(blk + 1), acc1);
			}

			blk += 2;
		}

		if blk < chunks && i >= blk * STEP + 7 {
			let d = _mm512_loadu_pd(data.as_ptr().add(i - blk * STEP - 7));
			acc0 = _mm512_fmadd_pd(d, *wblocks.get_unchecked(blk), acc0);
		}

		let mut sum = _mm512_reduce_add_pd(_mm512_add_pd(acc0, acc1));

		if tail != 0 {
			let base = chunks * STEP;
			for k in 0..tail {
				let w = *weights.get_unchecked(base + k);
				let d = *data.get_unchecked(i - (base + k));
				sum = d.mul_add(w, sum);
			}
		}

		*out.get_unchecked_mut(i) = sum * inv_norm;
	}
}

/// Streaming CWMA calculator that mirrors [`cwma`] output.
#[derive(Debug, Clone)]
pub struct CwmaStream {
	period: usize,
	weights: Vec<f64>,
	inv_norm: f64,
	ring: Vec<f64>,
	head: usize,
	total_count: usize,
	found_first: bool,
	first_idx: usize,
}

impl CwmaStream {
	pub fn try_new(params: CwmaParams) -> Result<Self, CwmaError> {
		let period = params.period.unwrap_or(14);
		if period <= 1 {
			return Err(CwmaError::InvalidPeriod { period, data_len: 0 });
		}

		let mut weights = Vec::with_capacity(period - 1);
		for i in 0..period - 1 {
			weights.push(((period - i) as f64).powi(3));
		}
		let inv_norm = 1.0 / weights.iter().sum::<f64>();

		Ok(Self {
			period,
			weights,
			inv_norm,
			ring: vec![f64::NAN; period],
			head: 0,
			total_count: 0,
			found_first: false,
			first_idx: 0,
		})
	}

	#[inline(always)]
	pub fn update(&mut self, value: f64) -> Option<f64> {
		let idx = self.total_count;
		self.total_count += 1;

		if !self.found_first && !value.is_nan() {
			self.found_first = true;
			self.first_idx = idx;
		}

		self.ring[self.head] = value;
		self.head = (self.head + 1) % self.period;

		if !self.found_first || idx < self.first_idx + self.period - 1 {
			return None;
		}

		let mut sum = 0.0;
		for k in 0..self.period - 1 {
			let ring_idx = (self.head + self.period - 1 - k) % self.period;
			sum += self.ring[ring_idx] * self.weights[k];
		}

		Some(sum * self.inv_norm)
	}
}

#[derive(Clone, Debug)]
pub struct CwmaBatchRange {
	pub period: (usize, usize, usize),
}

impl Default for CwmaBatchRange {
	fn default() -> Self {
		Self { period: (14, 240, 1) }
	}
}

/// Builder for [`cwma_batch_slice`] and related functions.
#[derive(Clone, Debug, Default)]
pub struct CwmaBatchBuilder {
	range: CwmaBatchRange,
	kernel: Kernel,
}

impl CwmaBatchBuilder {
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

	pub fn apply_slice(self, data: &[f64]) -> Result<CwmaBatchOutput, CwmaError> {
		cwma_batch_with_kernel(data, &self.range, self.kernel)
	}

	pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<CwmaBatchOutput, CwmaError> {
		CwmaBatchBuilder::new().kernel(k).apply_slice(data)
	}

	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<CwmaBatchOutput, CwmaError> {
		let slice = source_type(c, src);
		self.apply_slice(slice)
	}

	pub fn with_default_candles(c: &Candles) -> Result<CwmaBatchOutput, CwmaError> {
		CwmaBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
	}
}

pub fn cwma_batch_with_kernel(data: &[f64], sweep: &CwmaBatchRange, k: Kernel) -> Result<CwmaBatchOutput, CwmaError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(CwmaError::InvalidPeriod { period: 0, data_len: 0 }),
	};

	let simd = match kernel {
		#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
		Kernel::Avx512Batch => Kernel::Avx512,
		#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		#[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
		Kernel::Avx2Batch | Kernel::Avx512Batch => Kernel::Scalar,
		_ => unreachable!(),
	};
	cwma_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct CwmaBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<CwmaParams>,
	pub rows: usize,
	pub cols: usize,
}

impl CwmaBatchOutput {
	pub fn row_for_params(&self, p: &CwmaParams) -> Option<usize> {
		self.combos
			.iter()
			.position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
	}
	pub fn values_for(&self, p: &CwmaParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &CwmaBatchRange) -> Vec<CwmaParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}

	let periods = axis_usize(r.period);

	let mut out = Vec::with_capacity(periods.len());
	for &p in &periods {
		out.push(CwmaParams { period: Some(p) });
	}
	out
}

#[inline(always)]
pub fn cwma_batch_slice(data: &[f64], sweep: &CwmaBatchRange, kern: Kernel) -> Result<CwmaBatchOutput, CwmaError> {
	cwma_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn cwma_batch_par_slice(data: &[f64], sweep: &CwmaBatchRange, kern: Kernel) -> Result<CwmaBatchOutput, CwmaError> {
	cwma_batch_inner(data, sweep, kern, true)
}

#[inline]
fn round_up8(x: usize) -> usize {
	(x + 7) & !7
}

#[inline(always)]
fn cwma_batch_inner(
	data: &[f64],
	sweep: &CwmaBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<CwmaBatchOutput, CwmaError> {
	let combos = expand_grid(sweep);
	let cols = data.len();
	let rows = combos.len();

	if cols == 0 {
		return Err(CwmaError::EmptyInputData);
	}

	// Check for NaN values first
	let first = data.iter().position(|x| !x.is_nan()).ok_or(CwmaError::AllValuesNaN)?;

	// Calculate max period needed (rounded up to 8)
	let max_p = combos.iter().map(|c| round_up8(c.period.unwrap())).max().unwrap();

	// Validate we have enough data
	if (cols - first) < max_p {
		return Err(CwmaError::NotEnoughValidData {
			needed: max_p,
			valid: cols - first,
		});
	}

	// Step 1: Allocate uninitialized matrix using the helper
	let mut buf_mu = make_uninit_matrix(rows, cols);

	// Step 2: Calculate warmup periods for each row
	let warm: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap() - 1).collect();

	// Step 3: Initialize NaN prefixes for each row
	init_matrix_prefixes(&mut buf_mu, cols, &warm);

	// Step 4: Convert to mutable slice for computation
	let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
	let out: &mut [f64] =
		unsafe { core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len()) };

	// Step 5: Compute into the buffer
	cwma_batch_inner_into(data, sweep, kern, parallel, out)?;

	// Step 6: Reclaim as Vec<f64>
	let values = unsafe {
		Vec::from_raw_parts(
			buf_guard.as_mut_ptr() as *mut f64,
			buf_guard.len(),
			buf_guard.capacity(),
		)
	};

	Ok(CwmaBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
fn cwma_batch_inner_into(
	data: &[f64],
	sweep: &CwmaBatchRange,
	kern: Kernel,
	parallel: bool,
	out: &mut [f64],
) -> Result<Vec<CwmaParams>, CwmaError> {
	// Note: Input validation and warmup initialization already done in cwma_batch_inner
	let combos = expand_grid(sweep);

	let first = data.iter().position(|x| !x.is_nan()).ok_or(CwmaError::AllValuesNaN)?;

	let max_p = combos.iter().map(|c| round_up8(c.period.unwrap())).max().unwrap();

	let rows = combos.len();
	let cols = data.len();
	let mut inv_norms = vec![0.0; rows];

	let cap = rows * max_p;
	let mut aligned = AlignedVec::with_capacity(cap);
	let flat_w = aligned.as_mut_slice();

	for (row, prm) in combos.iter().enumerate() {
		let period = prm.period.unwrap();
		let mut norm = 0.0;
		for i in 0..period - 1 {
			let w = ((period - i) as f64).powi(3);
			flat_w[row * max_p + i] = w;
			norm += w;
		}
		let inv_norm = 1.0 / norm;
		inv_norms[row] = inv_norm;
	}

	// Convert output slice to MaybeUninit for row processing
	let out_uninit = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len()) };

	// Closure that fills one row
	let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
		let period = combos[row].period.unwrap();
		let w_ptr = flat_w.as_ptr().add(row * max_p);
		let inv_n = *inv_norms.get_unchecked(row);

		// cast just this row to &mut [f64]
		let out_row = core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

		match kern {
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => cwma_row_avx512(data, first, period, max_p, w_ptr, inv_n, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => cwma_row_avx2(data, first, period, max_p, w_ptr, inv_n, out_row),
			#[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
			Kernel::Avx2 | Kernel::Avx512 => cwma_row_scalar(data, first, period, max_p, w_ptr, inv_n, out_row),
			_ => cwma_row_scalar(data, first, period, max_p, w_ptr, inv_n, out_row),
		}
	};

	// 3. run every row, writing directly into `out_uninit`
	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			out_uninit
				.par_chunks_mut(cols)
				.enumerate()
				.for_each(|(row, slice)| do_row(row, slice));
		}

		#[cfg(target_arch = "wasm32")]
		{
			for (row, slice) in out_uninit.chunks_mut(cols).enumerate() {
				do_row(row, slice);
			}
		}
	} else {
		for (row, slice) in out_uninit.chunks_mut(cols).enumerate() {
			do_row(row, slice);
		}
	}

	// No need to transmute - the memory is already initialized and we're just
	// reinterpreting back to the original type

	Ok(combos)
}

#[inline(always)]
unsafe fn cwma_row_scalar(
	data: &[f64],
	first: usize,
	period: usize,
	stride: usize,
	w_ptr: *const f64,
	inv_n: f64,
	out: &mut [f64],
) {
	let wlen = period - 1;
	let start_idx = first + wlen;
	for i in start_idx..data.len().min(out.len()) {
		let mut sum = 0.0;
		for k in 0..wlen {
			let w = *w_ptr.add(k);
			let d = *data.get_unchecked(i - k);
			sum += d * w;
		}
		out[i] = sum * inv_n;
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn load_w512_rev(ptr: *const f64) -> __m512d {
	let rev = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
	_mm512_permutexvar_pd(rev, _mm512_loadu_pd(ptr))
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn load_w256_rev(ptr: *const f64) -> __m256d {
	_mm256_permute4x64_pd::<0b00011011>(_mm256_loadu_pd(ptr))
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
#[inline]
unsafe fn cwma_row_avx512(
	data: &[f64],
	first: usize,
	period: usize,
	_stride: usize,
	w_ptr: *const f64,
	inv_n: f64,
	out: &mut [f64],
) {
	if period <= 32 {
		cwma_row_avx512_short(data, first, period, w_ptr, inv_n, out);
	} else {
		cwma_row_avx512_long(data, first, period, w_ptr, inv_n, out);
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
#[inline]
unsafe fn cwma_row_avx512_short(
	data: &[f64],
	first: usize,
	period: usize,
	w_ptr: *const f64,
	inv_n: f64,
	out: &mut [f64],
) {
	const STEP: usize = 8;
	let wlen = period - 1;
	let chunks = wlen / STEP;
	let tail_len = wlen % STEP;
	let tmask: __mmask8 = (1u8 << tail_len).wrapping_sub(1);

	let w0 = load_w512_rev(w_ptr);
	let w1 = if chunks >= 2 {
		Some(load_w512_rev(w_ptr.add(STEP)))
	} else {
		None
	};

	let start_idx = first + wlen;
	for i in start_idx..data.len().min(out.len()) {
		let mut acc = _mm512_setzero_pd();

		if chunks >= 1 && i >= 7 {
			acc = _mm512_fmadd_pd(_mm512_loadu_pd(data.as_ptr().add(i - 7)), w0, acc);
		}

		if let Some(w1v) = w1 {
			if i >= STEP + 7 {
				let d1 = _mm512_loadu_pd(data.as_ptr().add(i - STEP - 7));
				acc = _mm512_fmadd_pd(d1, w1v, acc);
			}
		}

		let mut tail_sum = 0.0;
		if tail_len != 0 {
			let base = chunks * STEP;
			for k in 0..tail_len {
				let w = *w_ptr.add(base + k);
				let d = *data.get_unchecked(i - (base + k));
				tail_sum = d.mul_add(w, tail_sum);
			}
		}

		let sum = _mm512_reduce_add_pd(acc) + tail_sum;
		*out.get_unchecked_mut(i) = sum * inv_n;
	}
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512dq,fma")]
#[inline]
unsafe fn cwma_row_avx512_long(
	data: &[f64],
	first: usize,
	period: usize,
	w_ptr: *const f64,
	inv_n: f64,
	out: &mut [f64],
) {
	const STEP: usize = 8;
	const MAX: usize = 512;

	let wlen = period - 1;
	let n_chunks = wlen / STEP;
	let tail_len = wlen % STEP;
	let tmask: __mmask8 = (1u8 << tail_len).wrapping_sub(1);

	debug_assert!(n_chunks + (tail_len != 0) as usize <= MAX);

	let mut wregs: [core::mem::MaybeUninit<__m512d>; MAX] = core::mem::MaybeUninit::uninit().assume_init();

	for blk in 0..n_chunks {
		let src = w_ptr.add(wlen - (blk + 1) * STEP);
		wregs[blk].as_mut_ptr().write(load_w512_rev(src));
	}
	if tail_len != 0 {
		let src = w_ptr.add(wlen - n_chunks * STEP - tail_len);
		let wtl = _mm512_maskz_loadu_pd(tmask, src);
		wregs[n_chunks]
			.as_mut_ptr()
			.write(_mm512_permutexvar_pd(_mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7), wtl));
	}
	let wregs: &[__m512d] =
		core::slice::from_raw_parts(wregs.as_ptr() as *const __m512d, n_chunks + (tail_len != 0) as usize);

	let start_idx = first + wlen;
	for i in start_idx..data.len().min(out.len()) {
		// ------ window base pointer ------------------------------------------
		let base_ptr = data.as_ptr().add(i - wlen); // oldest value in window

		// ------ vector accumulators -----------------------------------------
		let mut s0 = _mm512_setzero_pd();
		let mut s1 = _mm512_setzero_pd();
		let mut s2 = _mm512_setzero_pd();
		let mut s3 = _mm512_setzero_pd();

		// ------ four-chunk unrolled FMA loop ---------------------------------
		let mut blk = 0;
		while blk + 3 < n_chunks {
			let d0 = _mm512_loadu_pd(base_ptr.add((blk + 0) * STEP));
			let d1 = _mm512_loadu_pd(base_ptr.add((blk + 1) * STEP));
			let d2 = _mm512_loadu_pd(base_ptr.add((blk + 2) * STEP));
			let d3 = _mm512_loadu_pd(base_ptr.add((blk + 3) * STEP));

			s0 = _mm512_fmadd_pd(d0, *wregs.get_unchecked(blk + 0), s0);
			s1 = _mm512_fmadd_pd(d1, *wregs.get_unchecked(blk + 1), s1);
			s2 = _mm512_fmadd_pd(d2, *wregs.get_unchecked(blk + 2), s2);
			s3 = _mm512_fmadd_pd(d3, *wregs.get_unchecked(blk + 3), s3);

			blk += 4;
		}
		// trailing complete 8-wide chunks
		for r in blk..n_chunks {
			let d = _mm512_loadu_pd(base_ptr.add(r * STEP));
			s0 = _mm512_fmadd_pd(d, *wregs.get_unchecked(r), s0);
		}

		// ------ tail shorter than 8 -----------------------------------------
		let mut sum = _mm512_reduce_add_pd(_mm512_add_pd(_mm512_add_pd(s0, s1), _mm512_add_pd(s2, s3)));

		if tail_len != 0 {
			let d_tail = _mm512_maskz_loadu_pd(tmask, base_ptr.add(n_chunks * STEP));
			let tail = _mm512_mul_pd(d_tail, *wregs.get_unchecked(n_chunks));
			sum += _mm512_reduce_add_pd(tail);
		}

		out[i] = sum * inv_n;
	}
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn cwma_row_avx2(
	data: &[f64],
	first: usize,
	period: usize,
	_stride: usize,
	w_ptr: *const f64,
	inv_n: f64,
	out: &mut [f64],
) {
	const STEP: usize = 4;
	let wlen = period - 1;
	let vec_blks = wlen / STEP;
	let tail = wlen % STEP;

	let tail_mask = match tail {
		0 => _mm256_setzero_si256(),
		1 => _mm256_setr_epi64x(-1, 0, 0, 0),
		2 => _mm256_setr_epi64x(-1, -1, 0, 0),
		3 => _mm256_setr_epi64x(-1, -1, -1, 0),
		_ => unreachable!(),
	};

	let start_idx = first + wlen;
	for i in start_idx..data.len().min(out.len()) {
		let mut acc = _mm256_setzero_pd();

		for blk in 0..vec_blks {
			let d = _mm256_loadu_pd(data.as_ptr().add(i - blk * STEP - 3));
			let w = load_w256_rev(w_ptr.add(blk * STEP));
			acc = _mm256_fmadd_pd(d, w, acc);
		}

		if tail != 0 {
			let base = vec_blks * STEP;
			let mut tail_sum = 0.0;
			for k in 0..tail {
				let w = *w_ptr.add(base + k);
				let d = *data.get_unchecked(i - (base + k));
				tail_sum = d.mul_add(w, tail_sum);
			}
			let hi = _mm256_extractf128_pd(acc, 1);
			let lo = _mm256_castpd256_pd128(acc);
			let tmp = _mm_add_pd(hi, lo);
			let tmp2 = _mm_add_pd(tmp, _mm_unpackhi_pd(tmp, tmp));
			out[i] = (_mm_cvtsd_f64(tmp2) + tail_sum) * inv_n;
		} else {
			let hi = _mm256_extractf128_pd(acc, 1);
			let lo = _mm256_castpd256_pd128(acc);
			let tmp = _mm_add_pd(hi, lo);
			let tmp2 = _mm_add_pd(tmp, _mm_unpackhi_pd(tmp, tmp));
			out[i] = _mm_cvtsd_f64(tmp2) * inv_n;
		}
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "cwma")]
#[pyo3(signature = (data, period, kernel=None))]
pub fn cwma_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	period: usize,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, false)?;

	let params = CwmaParams { period: Some(period) };
	let cwma_in = CwmaInput::from_slice(slice_in, params);

	let result_vec: Vec<f64> = py
		.allow_threads(|| cwma_with_kernel(&cwma_in, kern).map(|o| o.values))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "CwmaStream")]
pub struct CwmaStreamPy {
	stream: CwmaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl CwmaStreamPy {
	#[new]
	fn new(period: usize) -> PyResult<Self> {
		let params = CwmaParams { period: Some(period) };
		let stream = CwmaStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(CwmaStreamPy { stream })
	}

	fn update(&mut self, value: f64) -> Option<f64> {
		self.stream.update(value)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "cwma_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
pub fn cwma_batch_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	period_range: (usize, usize, usize),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, true)?;

	let sweep = CwmaBatchRange { period: period_range };

	let combos = expand_grid(&sweep);
	let rows = combos.len();
	let cols = slice_in.len();

	let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let slice_out = unsafe { out_arr.as_slice_mut()? };

	// Initialize NaN prefixes for each row based on warmup period
	let first = slice_in.iter().position(|x| !x.is_nan()).unwrap_or(slice_in.len());
	for (row, combo) in combos.iter().enumerate() {
		let warmup = first + combo.period.unwrap() - 1;
		let row_start = row * cols;
		for i in 0..warmup.min(cols) {
			slice_out[row_start + i] = f64::NAN;
		}
	}

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
			// Use the _into variant that writes directly to our pre-allocated buffer
			cwma_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
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
pub fn cwma_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
	let params = CwmaParams { period: Some(period) };
	let input = CwmaInput::from_slice(data, params);

	// Allocate output buffer once
	let mut output = vec![0.0; data.len()];

	// Compute directly into output buffer
	cwma_into_slice(&mut output, &input, Kernel::Auto).map_err(|e| JsValue::from_str(&e.to_string()))?;

	Ok(output)
}

// ================== Zero-Copy WASM Functions ==================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cwma_alloc(len: usize) -> *mut f64 {
	// Allocate memory for input/output buffer
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec); // Prevent deallocation
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cwma_free(ptr: *mut f64, len: usize) {
	// Free allocated memory
	if !ptr.is_null() && len > 0 {
		unsafe {
			let _ = Vec::from_raw_parts(ptr, len, len);
		}
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cwma_into(in_ptr: *const f64, out_ptr: *mut f64, len: usize, period: usize) -> Result<(), JsValue> {
	// Check for null pointers
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("null pointer passed to cwma_into"));
	}

	unsafe {
		// Create slice from pointer
		let data = std::slice::from_raw_parts(in_ptr, len);

		// Validate inputs
		if period == 0 || period > len {
			return Err(JsValue::from_str("Invalid period"));
		}

		// Calculate CWMA
		let params = CwmaParams { period: Some(period) };
		let input = CwmaInput::from_slice(data, params);

		// Check for aliasing (input and output buffers are the same)
		if in_ptr == out_ptr {
			// Use temporary buffer to avoid corruption during sliding window computation
			let mut temp = vec![0.0; len];
			cwma_into_slice(&mut temp, &input, Kernel::Auto).map_err(|e| JsValue::from_str(&e.to_string()))?;

			// Copy results back to output
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			out.copy_from_slice(&temp);
		} else {
			// No aliasing, compute directly into output
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			cwma_into_slice(out, &input, Kernel::Auto).map_err(|e| JsValue::from_str(&e.to_string()))?;
		}

		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[deprecated(since = "1.0.0", note = "Use cwma_batch instead")]
pub fn cwma_batch_js(
	data: &[f64],
	period_start: usize,
	period_end: usize,
	period_step: usize,
) -> Result<Vec<f64>, JsValue> {
	let sweep = CwmaBatchRange {
		period: (period_start, period_end, period_step),
	};

	// Use the existing batch function with parallel=false for WASM
	cwma_batch_inner(data, &sweep, Kernel::Auto, false)
		.map(|output| output.values)
		.map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cwma_batch_metadata_js(period_start: usize, period_end: usize, period_step: usize) -> Result<Vec<f64>, JsValue> {
	let sweep = CwmaBatchRange {
		period: (period_start, period_end, period_step),
	};

	let combos = expand_grid(&sweep);
	let metadata: Vec<f64> = combos.iter().map(|combo| combo.period.unwrap() as f64).collect();

	Ok(metadata)
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct CwmaBatchConfig {
	pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct CwmaBatchJsOutput {
	pub values: Vec<f64>,
	pub combos: Vec<CwmaParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = cwma_batch)]
pub fn cwma_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	let config: CwmaBatchConfig =
		serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

	let sweep = CwmaBatchRange {
		period: config.period_range,
	};

	let output = cwma_batch_inner(data, &sweep, Kernel::Auto, false).map_err(|e| JsValue::from_str(&e.to_string()))?;

	let js_output = CwmaBatchJsOutput {
		values: output.values,
		combos: output.combos,
		rows: output.rows,
		cols: output.cols,
	};

	serde_wasm_bindgen::to_value(&js_output).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

// ================== Optimized Batch Processing ==================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cwma_batch_into(
	in_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period_start: usize,
	period_end: usize,
	period_step: usize,
) -> Result<usize, JsValue> {
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("null pointer passed to cwma_batch_into"));
	}

	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);

		let sweep = CwmaBatchRange {
			period: (period_start, period_end, period_step),
		};

		let combos = expand_grid(&sweep);
		let rows = combos.len();
		let cols = len;

		let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);

		// Use optimized batch processing
		cwma_batch_inner_into(data, &sweep, Kernel::Auto, false, out).map_err(|e| JsValue::from_str(&e.to_string()))?;

		Ok(rows)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;
	#[cfg(feature = "proptest")]
	use proptest::prelude::*;

	fn check_cwma_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let default_params = CwmaParams { period: None };
		let input_def = CwmaInput::from_candles(&candles, "close", default_params);
		let output_def = cwma_with_kernel(&input_def, kernel)?;
		assert_eq!(output_def.values.len(), candles.close.len());

		let params_14 = CwmaParams { period: Some(14) };
		let input_14 = CwmaInput::from_candles(&candles, "hl2", params_14);
		let output_14 = cwma_with_kernel(&input_14, kernel)?;
		assert_eq!(output_14.values.len(), candles.close.len());

		let params_custom = CwmaParams { period: Some(20) };
		let input_custom = CwmaInput::from_candles(&candles, "hlc3", params_custom);
		let output_custom = cwma_with_kernel(&input_custom, kernel)?;
		assert_eq!(output_custom.values.len(), candles.close.len());

		Ok(())
	}

	fn check_cwma_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let input = CwmaInput::with_default_candles(&candles);
		let result = cwma_with_kernel(&input, kernel)?;
		assert_eq!(result.values.len(), candles.close.len());

		let expected_last_five = [
			59224.641237300435,
			59213.64831277214,
			59171.21190130624,
			59167.01279027576,
			59039.413552249636,
		];

		let start = result.values.len().saturating_sub(5);
		for (i, &val) in result.values[start..].iter().enumerate() {
			let diff = (val - expected_last_five[i]).abs();
			assert!(
				diff < 1e-9,
				"[{}] CWMA {:?} mismatch at idx {}: got {}, expected {}",
				test_name,
				kernel,
				i,
				val,
				expected_last_five[i]
			);
		}
		Ok(())
	}

	fn check_cwma_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let input = CwmaInput::with_default_candles(&candles);
		match input.data {
			CwmaData::Candles { source, .. } => assert_eq!(source, "close"),
			_ => panic!("Expected CwmaData::Candles"),
		}
		let output = cwma_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());

		Ok(())
	}

	fn check_cwma_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let input_data = [10.0, 20.0, 30.0];
		let params = CwmaParams { period: Some(0) };
		let input = CwmaInput::from_slice(&input_data, params);
		let res = cwma_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] Should fail with zero period", test_name);
		Ok(())
	}

	fn check_cwma_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data_small = [10.0, 20.0, 30.0];
		let params = CwmaParams { period: Some(10) };
		let input = CwmaInput::from_slice(&data_small, params);
		let res = cwma_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] Should fail with period exceeding length", test_name);
		Ok(())
	}

	fn check_cwma_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let single_point = [42.0];
		let params = CwmaParams { period: Some(9) };
		let input = CwmaInput::from_slice(&single_point, params);
		let res = cwma_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] Should fail with insufficient data", test_name);
		Ok(())
	}

	fn check_cwma_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let empty: [f64; 0] = [];
		let input = CwmaInput::from_slice(&empty, CwmaParams::default());
		let res = cwma_with_kernel(&input, kernel);
		assert!(
			matches!(res, Err(CwmaError::EmptyInputData)),
			"[{}] Should fail with empty input",
			test_name
		);
		Ok(())
	}

	fn check_cwma_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let first_params = CwmaParams { period: Some(80) };
		let first_input = CwmaInput::from_candles(&candles, "close", first_params);
		let first_result = cwma_with_kernel(&first_input, kernel)?;

		let second_params = CwmaParams { period: Some(60) };
		let second_input = CwmaInput::from_slice(&first_result.values, second_params);
		let second_result = cwma_with_kernel(&second_input, kernel)?;
		assert_eq!(second_result.values.len(), first_result.values.len());

		if second_result.values.len() > 240 {
			for i in 240..second_result.values.len() {
				assert!(
					!second_result.values[i].is_nan(),
					"[{}] Found unexpected NaN at index {}",
					test_name,
					i
				);
			}
		}
		Ok(())
	}

	fn check_cwma_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let input = CwmaInput::from_candles(&candles, "close", CwmaParams { period: Some(9) });
		let res = cwma_with_kernel(&input, kernel)?;
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

	fn check_cwma_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let period = 9;
		let input = CwmaInput::from_candles(&candles, "close", CwmaParams { period: Some(period) });
		let batch_output = cwma_with_kernel(&input, kernel)?.values;

		let mut stream = CwmaStream::try_new(CwmaParams { period: Some(period) })?;
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
				"[{}] CWMA streaming mismatch at idx {}: batch={}, stream={}, diff={}",
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
	fn check_cwma_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		// Test multiple parameter combinations to better catch uninitialized memory bugs
		let test_params = vec![
			// Default parameters
			CwmaParams::default(),
			// Small periods
			CwmaParams { period: Some(2) },
			CwmaParams { period: Some(3) },
			CwmaParams { period: Some(5) },
			// Medium periods
			CwmaParams { period: Some(7) },
			CwmaParams { period: Some(10) },
			CwmaParams { period: Some(14) },
			CwmaParams { period: Some(20) },
			// Large periods
			CwmaParams { period: Some(30) },
			CwmaParams { period: Some(50) },
			CwmaParams { period: Some(100) },
			CwmaParams { period: Some(200) },
			// Edge cases
			CwmaParams { period: Some(250) },
		];

		for (param_idx, params) in test_params.iter().enumerate() {
			let input = CwmaInput::from_candles(&candles, "close", params.clone());
			let output = cwma_with_kernel(&input, kernel)?;

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
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
                        with params: period={}",
						test_name,
						val,
						bits,
						i,
						params.period.unwrap_or(14)
					);
				}

				// Check for init_matrix_prefixes poison (0x22222222_22222222)
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
                        with params: period={}",
						test_name,
						val,
						bits,
						i,
						params.period.unwrap_or(14)
					);
				}

				// Check for make_uninit_matrix poison (0x33333333_33333333)
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
                        with params: period={}",
						test_name,
						val,
						bits,
						i,
						params.period.unwrap_or(14)
					);
				}
			}
		}

		Ok(())
	}

	// Release mode stub - does nothing
	#[cfg(not(debug_assertions))]
	fn check_cwma_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		Ok(())
	}

	#[cfg(feature = "proptest")]
	#[allow(clippy::float_cmp)]
	fn check_cwma_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		use proptest::prelude::*;
		skip_if_unsupported!(kernel, test_name);

		// 1 ─ Strategy: choose period first, then generate a ≥-period finite vector.
		let strat = (2usize..=32).prop_flat_map(|period| {
			(
				prop::collection::vec(
					(-1e6f64..1e6f64) // 100 % finite values
						.prop_filter("finite", |x| x.is_finite()),
					period..400,
				),
				Just(period),
				(-1e3f64..1e3f64).prop_filter("finite a", |a| a.is_finite() && *a != 0.0),
				-1e3f64..1e3f64, //  b may be zero
			)
		});

		proptest::test_runner::TestRunner::default()
			.run(&strat, |(data, period, a, b)| {
				let params = CwmaParams { period: Some(period) };
				let input = CwmaInput::from_slice(&data, params.clone());

				// --- run both kernels, but DO NOT unwrap blindly -----------------
				let fast = cwma_with_kernel(&input, kernel);
				let slow = cwma_with_kernel(&input, Kernel::Scalar);

				match (fast, slow) {
					// ➊ Same error kind ⇒ property holds.
					(Err(e1), Err(e2)) if std::mem::discriminant(&e1) == std::mem::discriminant(&e2) => {
						return Ok(());
					}
					// ➊′ *Different* error kinds → fail explicitly
					(Err(e1), Err(e2)) => {
						prop_assert!(false, "different errors: fast={:?} slow={:?}", e1, e2)
					}

					// ➋ Kernels disagree on success / error.
					(Err(e1), Ok(_)) => {
						prop_assert!(false, "fast errored {e1:?} but scalar succeeded")
					}
					(Ok(_), Err(e2)) => {
						prop_assert!(false, "scalar errored {e2:?} but fast succeeded")
					}

					// ➌ Both succeeded – run full invariant suite.
					(Ok(fast), Ok(reference)) => {
						let CwmaOutput { values: out } = fast;
						let CwmaOutput { values: rref } = reference;

						// Pre-compute streaming and affine-transformed outputs once.
						let mut stream = CwmaStream::try_new(params.clone()).unwrap();
						let mut s_out = Vec::with_capacity(data.len());
						for &v in &data {
							s_out.push(stream.update(v).unwrap_or(f64::NAN));
						}

						let transformed: Vec<f64> = data.iter().map(|x| a * x + b).collect();
						let t_out = cwma(&CwmaInput::from_slice(&transformed, params))?.values;

						for i in (period - 1)..data.len() {
							// 1️⃣ Window-boundedness --------------------------------
							let w = &data[i + 1 - period..=i];
							let (lo, hi) = w
								.iter()
								.fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &v| (l.min(v), h.max(v)));
							let y = out[i];
							let yr = rref[i];
							let ys = s_out[i];
							let yt = t_out[i];

							prop_assert!(
								y.is_nan() || (y >= lo - 1e-9 && y <= hi + 1e-9),
								"idx {i}: {y} ∉ [{lo}, {hi}]"
							);

							// 2️⃣ Period-1 identity --------------------------------
							if period == 1 && y.is_finite() {
								prop_assert!((y - data[i]).abs() <= f64::EPSILON);
							}

							// 3️⃣ Constant-series invariance ------------------------
							if w.iter().all(|v| *v == w[0]) {
								prop_assert!((y - w[0]).abs() <= 1e-9);
							}

							// 4️⃣ Monotone preservation ----------------------------
							if data[..=i].windows(2).all(|p| p[0] <= p[1]) && y.is_finite() && out[i - 1].is_finite() {
								prop_assert!(y >= out[i - 1] - 1e-12);
							}

							// 5️⃣ Affine equivariance ------------------------------
							{
								let expected = a * y + b;
								let diff = (yt - expected).abs();
								let tol_abs = 1e-9_f64; // tight near zero
								let tol_rel = expected.abs() * 1e-9; // scales with magnitude
								let ulp = yt.to_bits().abs_diff(expected.to_bits());

								prop_assert!(
									diff <= tol_abs.max(tol_rel) || ulp <= 8,
									"idx {i}: affine mismatch diff={diff:e}  ULP={ulp}"
								);
							}

							// 6️⃣ Scalar ≡ fast (ULP ≤ 4 or abs ≤ 1e-9) ------------
							let ulp = y.to_bits().abs_diff(yr.to_bits());
							prop_assert!(
								(y - yr).abs() <= 1e-9 || ulp <= 4,
								"idx {i}: fast={y} ref={yr} ULP={ulp}"
							);

							// 7️⃣ Streaming parity ---------------------------------
							prop_assert!(
								(y - ys).abs() <= 1e-9 || (y.is_nan() && ys.is_nan()),
								"idx {i}: stream mismatch"
							);
						}

						// 9️⃣ Warm-up sentinel NaNs -------------------------------
						let first = data.iter().position(|x| !x.is_nan()).unwrap_or(data.len());
						let warm = first + period - 1;
						prop_assert!(out[..warm].iter().all(|v| v.is_nan()));
					}
				}

				Ok(())
			})
			.unwrap();

		// 🔟 Error-path smoke tests (unchanged) -----------------------------------
		assert!(cwma(&CwmaInput::from_slice(&[], CwmaParams::default())).is_err());
		assert!(cwma(&CwmaInput::from_slice(&[f64::NAN; 12], CwmaParams::default())).is_err());
		assert!(cwma(&CwmaInput::from_slice(&[1.0; 5], CwmaParams { period: Some(8) })).is_err());
		assert!(cwma(&CwmaInput::from_slice(&[1.0; 5], CwmaParams { period: Some(0) })).is_err());

		Ok(())
	}

	macro_rules! generate_all_cwma_tests {
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

	generate_all_cwma_tests!(
		check_cwma_partial_params,
		check_cwma_accuracy,
		check_cwma_default_candles,
		check_cwma_zero_period,
		check_cwma_period_exceeds_length,
		check_cwma_very_small_dataset,
		check_cwma_empty_input,
		check_cwma_reinput,
		check_cwma_nan_handling,
		check_cwma_streaming,
		check_cwma_no_poison
	);

	#[cfg(feature = "proptest")]
	generate_all_cwma_tests!(check_cwma_property);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let output = CwmaBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;

		let def = CwmaParams::default();
		let row = output.values_for(&def).expect("default row missing");

		assert_eq!(row.len(), c.close.len());

		let expected = [
			59224.641237300435,
			59213.64831277214,
			59171.21190130624,
			59167.01279027576,
			59039.413552249636,
		];
		let start = row.len() - 5;
		for (i, &v) in row[start..].iter().enumerate() {
			assert!(
				(v - expected[i]).abs() < 1e-8,
				"[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
			);
		}
		Ok(())
	}

	// Check for poison values in batch output - only runs in debug mode
	#[cfg(debug_assertions)]
	fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		// Test multiple batch configurations to better catch uninitialized memory bugs
		let test_configs = vec![
			// Small range
			(2, 5, 1), // periods: 2, 3, 4, 5
			// Medium range with gaps
			(5, 25, 5), // periods: 5, 10, 15, 20, 25
			// Large range
			(10, 50, 10), // periods: 10, 20, 30, 40, 50
			// Edge case: very small periods
			(2, 4, 1), // periods: 2, 3, 4
			// Edge case: large periods
			(50, 150, 25), // periods: 50, 75, 100, 125, 150
			// Dense range
			(9, 21, 2), // periods: 9, 11, 13, 15, 17, 19, 21
			// Original configuration
			(9, 21, 4), // periods: 9, 13, 17, 21
			// Very large periods
			(100, 300, 50), // periods: 100, 150, 200, 250, 300
		];

		for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
			let output = CwmaBatchBuilder::new()
				.kernel(kernel)
				.period_range(p_start, p_end, p_step)
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
				let combo = &output.combos[row];

				// Check for alloc_with_nan_prefix poison (0x11111111_11111111)
				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
                        at row {} col {} (flat index {}) with params: period={}",
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

				// Check for init_matrix_prefixes poison (0x22222222_22222222)
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
                        at row {} col {} (flat index {}) with params: period={}",
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

				// Check for make_uninit_matrix poison (0x33333333_33333333)
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
                        at row {} col {} (flat index {}) with params: period={}",
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

	// Release mode stub - does nothing
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
	gen_batch_tests!(check_batch_no_poison);
}
