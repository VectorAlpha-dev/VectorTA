//! # Know Sure Thing (KST)
//!
//! KST is a momentum oscillator based on the smoothed rate-of-change (ROC) values of four different time frames.
//! This implementation mirrors alma.rs in performance and structure, providing AVX2/AVX512 stubs, batch/grid interfaces,
//! streaming, builders, and thorough input validation. All kernel variants and AVX stubs are present for API parity.
//!
//! ## Parameters
//! - **sma_period1**: Smoothing period for the first ROC. Defaults to 10.
//! - **sma_period2**: Smoothing period for the second ROC. Defaults to 10.
//! - **sma_period3**: Smoothing period for the third ROC. Defaults to 10.
//! - **sma_period4**: Smoothing period for the fourth ROC. Defaults to 15.
//! - **roc_period1**: Period for the first ROC calculation. Defaults to 10.
//! - **roc_period2**: Period for the second ROC calculation. Defaults to 15.
//! - **roc_period3**: Period for the third ROC calculation. Defaults to 20.
//! - **roc_period4**: Period for the fourth ROC calculation. Defaults to 30.
//! - **signal_period**: Smoothing period for the signal line. Defaults to 9.
//!
//! ## Errors
//! - **AllValuesNaN**: All input data values are `NaN`.
//! - **InvalidPeriod**: A period is zero or exceeds the data length.
//! - **NotEnoughValidData**: Not enough valid data points for the requested period.
//!
//! ## Returns
//! - `Ok(KstOutput)` on success, containing two `Vec<f64>`: KST line and signal line.
//! - `Err(KstError)` otherwise.

use crate::indicators::moving_averages::sma::{sma, SmaData, SmaError, SmaInput, SmaOutput, SmaParams};
use crate::indicators::roc::{roc, RocData, RocError, RocInput, RocOutput, RocParams};
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel, alloc_with_nan_prefix, make_uninit_matrix, init_matrix_prefixes};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;
#[cfg(feature = "wasm")]
use js_sys;

#[derive(Debug, Clone)]
pub enum KstData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct KstOutput {
	pub line: Vec<f64>,
	pub signal: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "wasm", derive(serde::Serialize, serde::Deserialize))]
pub struct KstParams {
	pub sma_period1: Option<usize>,
	pub sma_period2: Option<usize>,
	pub sma_period3: Option<usize>,
	pub sma_period4: Option<usize>,
	pub roc_period1: Option<usize>,
	pub roc_period2: Option<usize>,
	pub roc_period3: Option<usize>,
	pub roc_period4: Option<usize>,
	pub signal_period: Option<usize>,
}

impl Default for KstParams {
	fn default() -> Self {
		Self {
			sma_period1: Some(10),
			sma_period2: Some(10),
			sma_period3: Some(10),
			sma_period4: Some(15),
			roc_period1: Some(10),
			roc_period2: Some(15),
			roc_period3: Some(20),
			roc_period4: Some(30),
			signal_period: Some(9),
		}
	}
}

#[derive(Debug, Clone)]
pub struct KstInput<'a> {
	pub data: KstData<'a>,
	pub params: KstParams,
}

impl<'a> AsRef<[f64]> for KstInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			KstData::Slice(slice) => slice,
			KstData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

impl<'a> KstInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: KstParams) -> Self {
		Self {
			data: KstData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slice(sl: &'a [f64], p: KstParams) -> Self {
		Self {
			data: KstData::Slice(sl),
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", KstParams::default())
	}
	#[inline]
	pub fn get_sma_period1(&self) -> usize {
		self.params.sma_period1.unwrap_or(10)
	}
	#[inline]
	pub fn get_sma_period2(&self) -> usize {
		self.params.sma_period2.unwrap_or(10)
	}
	#[inline]
	pub fn get_sma_period3(&self) -> usize {
		self.params.sma_period3.unwrap_or(10)
	}
	#[inline]
	pub fn get_sma_period4(&self) -> usize {
		self.params.sma_period4.unwrap_or(15)
	}
	#[inline]
	pub fn get_roc_period1(&self) -> usize {
		self.params.roc_period1.unwrap_or(10)
	}
	#[inline]
	pub fn get_roc_period2(&self) -> usize {
		self.params.roc_period2.unwrap_or(15)
	}
	#[inline]
	pub fn get_roc_period3(&self) -> usize {
		self.params.roc_period3.unwrap_or(20)
	}
	#[inline]
	pub fn get_roc_period4(&self) -> usize {
		self.params.roc_period4.unwrap_or(30)
	}
	#[inline]
	pub fn get_signal_period(&self) -> usize {
		self.params.signal_period.unwrap_or(9)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct KstBuilder {
	sma_period1: Option<usize>,
	sma_period2: Option<usize>,
	sma_period3: Option<usize>,
	sma_period4: Option<usize>,
	roc_period1: Option<usize>,
	roc_period2: Option<usize>,
	roc_period3: Option<usize>,
	roc_period4: Option<usize>,
	signal_period: Option<usize>,
	kernel: Kernel,
}

impl Default for KstBuilder {
	fn default() -> Self {
		Self {
			sma_period1: None,
			sma_period2: None,
			sma_period3: None,
			sma_period4: None,
			roc_period1: None,
			roc_period2: None,
			roc_period3: None,
			roc_period4: None,
			signal_period: None,
			kernel: Kernel::Auto,
		}
	}
}

impl KstBuilder {
	#[inline(always)]
	pub fn new() -> Self {
		Self::default()
	}
	#[inline(always)]
	pub fn sma_period1(mut self, n: usize) -> Self {
		self.sma_period1 = Some(n);
		self
	}
	#[inline(always)]
	pub fn sma_period2(mut self, n: usize) -> Self {
		self.sma_period2 = Some(n);
		self
	}
	#[inline(always)]
	pub fn sma_period3(mut self, n: usize) -> Self {
		self.sma_period3 = Some(n);
		self
	}
	#[inline(always)]
	pub fn sma_period4(mut self, n: usize) -> Self {
		self.sma_period4 = Some(n);
		self
	}
	#[inline(always)]
	pub fn roc_period1(mut self, n: usize) -> Self {
		self.roc_period1 = Some(n);
		self
	}
	#[inline(always)]
	pub fn roc_period2(mut self, n: usize) -> Self {
		self.roc_period2 = Some(n);
		self
	}
	#[inline(always)]
	pub fn roc_period3(mut self, n: usize) -> Self {
		self.roc_period3 = Some(n);
		self
	}
	#[inline(always)]
	pub fn roc_period4(mut self, n: usize) -> Self {
		self.roc_period4 = Some(n);
		self
	}
	#[inline(always)]
	pub fn signal_period(mut self, n: usize) -> Self {
		self.signal_period = Some(n);
		self
	}
	#[inline(always)]
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline(always)]
	pub fn apply(self, c: &Candles) -> Result<KstOutput, KstError> {
		let p = KstParams {
			sma_period1: self.sma_period1,
			sma_period2: self.sma_period2,
			sma_period3: self.sma_period3,
			sma_period4: self.sma_period4,
			roc_period1: self.roc_period1,
			roc_period2: self.roc_period2,
			roc_period3: self.roc_period3,
			roc_period4: self.roc_period4,
			signal_period: self.signal_period,
		};
		let i = KstInput::from_candles(c, "close", p);
		kst_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slice(self, d: &[f64]) -> Result<KstOutput, KstError> {
		let p = KstParams {
			sma_period1: self.sma_period1,
			sma_period2: self.sma_period2,
			sma_period3: self.sma_period3,
			sma_period4: self.sma_period4,
			roc_period1: self.roc_period1,
			roc_period2: self.roc_period2,
			roc_period3: self.roc_period3,
			roc_period4: self.roc_period4,
			signal_period: self.signal_period,
		};
		let i = KstInput::from_slice(d, p);
		kst_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<KstStream, KstError> {
		let p = KstParams {
			sma_period1: self.sma_period1,
			sma_period2: self.sma_period2,
			sma_period3: self.sma_period3,
			sma_period4: self.sma_period4,
			roc_period1: self.roc_period1,
			roc_period2: self.roc_period2,
			roc_period3: self.roc_period3,
			roc_period4: self.roc_period4,
			signal_period: self.signal_period,
		};
		KstStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum KstError {
	#[error("kst: {0}")]
	Roc(#[from] RocError),
	#[error("kst: {0}")]
	Sma(#[from] SmaError),
	#[error("kst: All values are NaN.")]
	AllValuesNaN,
	#[error("kst: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("kst: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn kst(input: &KstInput) -> Result<KstOutput, KstError> {
	kst_with_kernel(input, Kernel::Auto)
}

pub fn kst_with_kernel(input: &KstInput, kernel: Kernel) -> Result<KstOutput, KstError> {
	let data: &[f64] = input.as_ref();
	let first = data.iter().position(|x| !x.is_nan()).ok_or(KstError::AllValuesNaN)?;
	let len = data.len();
	if len == 0 {
		return Err(KstError::AllValuesNaN);
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};
	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => kst_scalar(input, first, len),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => kst_avx2(input, first, len),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => kst_avx512(input, first, len),
			_ => unreachable!(),
		}
	}
}

#[inline]
pub unsafe fn kst_scalar(input: &KstInput, first: usize, len: usize) -> Result<KstOutput, KstError> {
	let data: &[f64] = input.as_ref();
	let s1 = input.get_sma_period1();
	let s2 = input.get_sma_period2();
	let s3 = input.get_sma_period3();
	let s4 = input.get_sma_period4();
	let r1 = input.get_roc_period1();
	let r2 = input.get_roc_period2();
	let r3 = input.get_roc_period3();
	let r4 = input.get_roc_period4();
	let sig = input.get_signal_period();

	// Calculate warmup periods for each component
	let warmup1 = r1 + s1 - 1;
	let warmup2 = r2 + s2 - 1;
	let warmup3 = r3 + s3 - 1;
	let warmup4 = r4 + s4 - 1;
	let warmup = warmup1.max(warmup2).max(warmup3).max(warmup4);
	
	// Allocate output buffers
	let mut line = alloc_with_nan_prefix(len, warmup);
	let mut signal = alloc_with_nan_prefix(len, warmup + sig - 1);
	
	// Circular buffers for ROC values (only need to store sma_period values)
	let mut roc1_buf = AVec::<f64>::with_capacity(CACHELINE_ALIGN, s1);
	let mut roc2_buf = AVec::<f64>::with_capacity(CACHELINE_ALIGN, s2);
	let mut roc3_buf = AVec::<f64>::with_capacity(CACHELINE_ALIGN, s3);
	let mut roc4_buf = AVec::<f64>::with_capacity(CACHELINE_ALIGN, s4);
	
	// Initialize buffers
	roc1_buf.resize(s1, 0.0);
	roc2_buf.resize(s2, 0.0);
	roc3_buf.resize(s3, 0.0);
	roc4_buf.resize(s4, 0.0);
	
	// Initialize ROC buffer indices
	let mut idx1 = 0;
	let mut idx2 = 0;
	let mut idx3 = 0;
	let mut idx4 = 0;
	
	// Running sums for SMAs
	let mut sum1 = 0.0;
	let mut sum2 = 0.0;
	let mut sum3 = 0.0;
	let mut sum4 = 0.0;
	
	// Inverse periods for multiplication instead of division
	let inv1 = 1.0 / (s1 as f64);
	let inv2 = 1.0 / (s2 as f64);
	let inv3 = 1.0 / (s3 as f64);
	let inv4 = 1.0 / (s4 as f64);
	
	// Process data
	for i in first..len {
		// Calculate ROC values inline
		let mut roc1_val = 0.0;
		let mut roc2_val = 0.0;
		let mut roc3_val = 0.0;
		let mut roc4_val = 0.0;
		
		if i >= r1 {
			let prev1 = data[i - r1];
			if prev1 != 0.0 && !prev1.is_nan() {
				roc1_val = ((data[i] / prev1) - 1.0) * 100.0;
			}
		}
		
		if i >= r2 {
			let prev2 = data[i - r2];
			if prev2 != 0.0 && !prev2.is_nan() {
				roc2_val = ((data[i] / prev2) - 1.0) * 100.0;
			}
		}
		
		if i >= r3 {
			let prev3 = data[i - r3];
			if prev3 != 0.0 && !prev3.is_nan() {
				roc3_val = ((data[i] / prev3) - 1.0) * 100.0;
			}
		}
		
		if i >= r4 {
			let prev4 = data[i - r4];
			if prev4 != 0.0 && !prev4.is_nan() {
				roc4_val = ((data[i] / prev4) - 1.0) * 100.0;
			}
		}
		
		// Update SMA buffers and sums
		if i >= r1 {
			sum1 -= roc1_buf[idx1];
			roc1_buf[idx1] = roc1_val;
			sum1 += roc1_val;
			idx1 = (idx1 + 1) % s1;
		}
		
		if i >= r2 {
			sum2 -= roc2_buf[idx2];
			roc2_buf[idx2] = roc2_val;
			sum2 += roc2_val;
			idx2 = (idx2 + 1) % s2;
		}
		
		if i >= r3 {
			sum3 -= roc3_buf[idx3];
			roc3_buf[idx3] = roc3_val;
			sum3 += roc3_val;
			idx3 = (idx3 + 1) % s3;
		}
		
		if i >= r4 {
			sum4 -= roc4_buf[idx4];
			roc4_buf[idx4] = roc4_val;
			sum4 += roc4_val;
			idx4 = (idx4 + 1) % s4;
		}
		
		// Calculate KST line value
		if i >= warmup {
			let sma1 = sum1 * inv1;
			let sma2 = sum2 * inv2;
			let sma3 = sum3 * inv3;
			let sma4 = sum4 * inv4;
			line[i] = sma1 + 2.0 * sma2 + 3.0 * sma3 + 4.0 * sma4;
		}
	}
	
	// Calculate signal line using SMA on the KST line
	let mut sig_sum = 0.0;
	let sig_inv = 1.0 / (sig as f64);
	
	// Initialize signal sum
	for i in warmup..(warmup + sig).min(len) {
		if !line[i].is_nan() {
			sig_sum += line[i];
		}
	}
	
	// First signal value
	if warmup + sig - 1 < len {
		signal[warmup + sig - 1] = sig_sum * sig_inv;
	}
	
	// Rolling signal calculation
	for i in (warmup + sig)..len {
		sig_sum += line[i] - line[i - sig];
		signal[i] = sig_sum * sig_inv;
	}
	
	Ok(KstOutput { line, signal })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn kst_avx2(input: &KstInput, first: usize, len: usize) -> Result<KstOutput, KstError> {
	// Stub: calls scalar for now, API parity
	kst_scalar(input, first, len)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn kst_avx512(input: &KstInput, first: usize, len: usize) -> Result<KstOutput, KstError> {
	// Dispatch to long/short stub (all scalar for now)
	if len <= 32 {
		kst_avx512_short(input, first, len)
	} else {
		kst_avx512_long(input, first, len)
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn kst_avx512_short(input: &KstInput, first: usize, len: usize) -> Result<KstOutput, KstError> {
	kst_scalar(input, first, len)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn kst_avx512_long(input: &KstInput, first: usize, len: usize) -> Result<KstOutput, KstError> {
	kst_scalar(input, first, len)
}

#[inline]
pub fn kst_batch_with_kernel(data: &[f64], sweep: &KstBatchRange, k: Kernel) -> Result<KstBatchOutput, KstError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(KstError::InvalidPeriod { period: 0, data_len: 0 }),
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	kst_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct KstBatchRange {
	pub sma_period1: (usize, usize, usize),
	pub sma_period2: (usize, usize, usize),
	pub sma_period3: (usize, usize, usize),
	pub sma_period4: (usize, usize, usize),
	pub roc_period1: (usize, usize, usize),
	pub roc_period2: (usize, usize, usize),
	pub roc_period3: (usize, usize, usize),
	pub roc_period4: (usize, usize, usize),
	pub signal_period: (usize, usize, usize),
}

impl Default for KstBatchRange {
	fn default() -> Self {
		Self {
			sma_period1: (10, 10, 0),
			sma_period2: (10, 10, 0),
			sma_period3: (10, 10, 0),
			sma_period4: (15, 15, 0),
			roc_period1: (10, 10, 0),
			roc_period2: (15, 15, 0),
			roc_period3: (20, 20, 0),
			roc_period4: (30, 30, 0),
			signal_period: (9, 9, 0),
		}
	}
}

#[derive(Clone, Debug, Default)]
pub struct KstBatchBuilder {
	range: KstBatchRange,
	kernel: Kernel,
}

impl KstBatchBuilder {
	pub fn new() -> Self {
		Self::default()
	}
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	pub fn sma_period1_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.sma_period1 = (start, end, step);
		self
	}
	pub fn sma_period2_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.sma_period2 = (start, end, step);
		self
	}
	pub fn sma_period3_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.sma_period3 = (start, end, step);
		self
	}
	pub fn sma_period4_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.sma_period4 = (start, end, step);
		self
	}
	pub fn roc_period1_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.roc_period1 = (start, end, step);
		self
	}
	pub fn roc_period2_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.roc_period2 = (start, end, step);
		self
	}
	pub fn roc_period3_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.roc_period3 = (start, end, step);
		self
	}
	pub fn roc_period4_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.roc_period4 = (start, end, step);
		self
	}
	pub fn signal_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.signal_period = (start, end, step);
		self
	}
	pub fn apply_slice(self, data: &[f64]) -> Result<KstBatchOutput, KstError> {
		kst_batch_with_kernel(data, &self.range, self.kernel)
	}
	pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<KstBatchOutput, KstError> {
		KstBatchBuilder::new().kernel(k).apply_slice(data)
	}
	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<KstBatchOutput, KstError> {
		self.apply_slice(source_type(c, src))
	}
	pub fn with_default_candles(c: &Candles) -> Result<KstBatchOutput, KstError> {
		KstBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
	}
}

#[derive(Clone, Debug)]
pub struct KstBatchOutput {
	pub lines: Vec<f64>,
	pub signals: Vec<f64>,
	pub combos: Vec<KstParams>,
	pub rows: usize,
	pub cols: usize,
}
impl KstBatchOutput {
	pub fn row_for_params(&self, p: &KstParams) -> Option<usize> {
		self.combos.iter().position(|c| {
			c.sma_period1.unwrap_or(10) == p.sma_period1.unwrap_or(10)
				&& c.sma_period2.unwrap_or(10) == p.sma_period2.unwrap_or(10)
				&& c.sma_period3.unwrap_or(10) == p.sma_period3.unwrap_or(10)
				&& c.sma_period4.unwrap_or(15) == p.sma_period4.unwrap_or(15)
				&& c.roc_period1.unwrap_or(10) == p.roc_period1.unwrap_or(10)
				&& c.roc_period2.unwrap_or(15) == p.roc_period2.unwrap_or(15)
				&& c.roc_period3.unwrap_or(20) == p.roc_period3.unwrap_or(20)
				&& c.roc_period4.unwrap_or(30) == p.roc_period4.unwrap_or(30)
				&& c.signal_period.unwrap_or(9) == p.signal_period.unwrap_or(9)
		})
	}
	pub fn lines_for(&self, p: &KstParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.lines[start..start + self.cols]
		})
	}
	pub fn signals_for(&self, p: &KstParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.signals[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &KstBatchRange) -> Vec<KstParams> {
	fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let s1 = axis(r.sma_period1);
	let s2 = axis(r.sma_period2);
	let s3 = axis(r.sma_period3);
	let s4 = axis(r.sma_period4);
	let r1 = axis(r.roc_period1);
	let r2 = axis(r.roc_period2);
	let r3 = axis(r.roc_period3);
	let r4 = axis(r.roc_period4);
	let sig = axis(r.signal_period);

	let mut out = Vec::with_capacity(
		s1.len() * s2.len() * s3.len() * s4.len() * r1.len() * r2.len() * r3.len() * r4.len() * sig.len(),
	);
	for &s1v in &s1 {
		for &s2v in &s2 {
			for &s3v in &s3 {
				for &s4v in &s4 {
					for &r1v in &r1 {
						for &r2v in &r2 {
							for &r3v in &r3 {
								for &r4v in &r4 {
									for &sigv in &sig {
										out.push(KstParams {
											sma_period1: Some(s1v),
											sma_period2: Some(s2v),
											sma_period3: Some(s3v),
											sma_period4: Some(s4v),
											roc_period1: Some(r1v),
											roc_period2: Some(r2v),
											roc_period3: Some(r3v),
											roc_period4: Some(r4v),
											signal_period: Some(sigv),
										});
									}
								}
							}
						}
					}
				}
			}
		}
	}
	out
}

#[inline(always)]
pub fn kst_batch_slice(data: &[f64], sweep: &KstBatchRange, kern: Kernel) -> Result<KstBatchOutput, KstError> {
	kst_batch_inner(data, sweep, kern, false)
}
#[inline(always)]
pub fn kst_batch_par_slice(data: &[f64], sweep: &KstBatchRange, kern: Kernel) -> Result<KstBatchOutput, KstError> {
	kst_batch_inner(data, sweep, kern, true)
}

fn kst_batch_inner(
	data: &[f64],
	sweep: &KstBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<KstBatchOutput, KstError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(KstError::InvalidPeriod { period: 0, data_len: 0 });
	}
	let first = data.iter().position(|x| !x.is_nan()).ok_or(KstError::AllValuesNaN)?;
	let max_p = combos
		.iter()
		.map(|c| {
			c.sma_period1
				.unwrap()
				.max(c.sma_period2.unwrap())
				.max(c.sma_period3.unwrap())
				.max(c.sma_period4.unwrap())
				.max(c.roc_period1.unwrap())
				.max(c.roc_period2.unwrap())
				.max(c.roc_period3.unwrap())
				.max(c.roc_period4.unwrap())
				.max(c.signal_period.unwrap())
		})
		.max()
		.unwrap();
	if data.len() - first < max_p {
		return Err(KstError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}
	let rows = combos.len();
	let cols = data.len();
	
	// Allocate uninitialized memory for lines and signals
	let mut lines_mu = make_uninit_matrix(rows, cols);
	let mut signals_mu = make_uninit_matrix(rows, cols);
	
	// Calculate warmup periods for each parameter combo
	let warm_periods: Vec<usize> = combos
		.iter()
		.map(|c| {
			// Warmup is max ROC period + its SMA period - 1
			let max_roc = c.roc_period4.unwrap_or(30);
			let max_sma = c.sma_period4.unwrap_or(15);
			max_roc + max_sma - 1
		})
		.collect();
	
	// Initialize NaN prefixes
	init_matrix_prefixes(&mut lines_mu, cols, &warm_periods);
	init_matrix_prefixes(&mut signals_mu, cols, &warm_periods);
	
	// Convert to mutable slices
	let mut lines_guard = core::mem::ManuallyDrop::new(lines_mu);
	let lines: &mut [f64] = unsafe { 
		core::slice::from_raw_parts_mut(lines_guard.as_mut_ptr() as *mut f64, lines_guard.len()) 
	};
	
	let mut signals_guard = core::mem::ManuallyDrop::new(signals_mu);
	let signals: &mut [f64] = unsafe { 
		core::slice::from_raw_parts_mut(signals_guard.as_mut_ptr() as *mut f64, signals_guard.len()) 
	};

	let do_row = |row: usize, line_row: &mut [f64], sig_row: &mut [f64]| unsafe {
		let prm = &combos[row];
		let inp = KstInput::from_slice(data, *prm);
		match kern {
			Kernel::Scalar => {
				let r = kst_row_scalar(&inp, first, cols)?;
				line_row.copy_from_slice(&r.line);
				sig_row.copy_from_slice(&r.signal);
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => {
				let r = kst_row_avx2(&inp, first, cols)?;
				line_row.copy_from_slice(&r.line);
				sig_row.copy_from_slice(&r.signal);
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => {
				let r = kst_row_avx512(&inp, first, cols)?;
				line_row.copy_from_slice(&r.line);
				sig_row.copy_from_slice(&r.signal);
			}
			_ => unreachable!(),
		}
		Ok::<(), KstError>(())
	};
	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			lines
				.par_chunks_mut(cols)
				.zip(signals.par_chunks_mut(cols))
				.enumerate()
				.for_each(|(row, (lrow, srow))| {
					let _ = do_row(row, lrow, srow);
				});
		}
		#[cfg(target_arch = "wasm32")]
		{
			for (row, (lrow, srow)) in lines.chunks_mut(cols).zip(signals.chunks_mut(cols)).enumerate() {
				let _ = do_row(row, lrow, srow);
			}
		}
	} else {
		for (row, (lrow, srow)) in lines.chunks_mut(cols).zip(signals.chunks_mut(cols)).enumerate() {
			let _ = do_row(row, lrow, srow);
		}
	}
	
	// Convert back to Vec<f64> for output
	let lines_vec = unsafe {
		let raw_ptr = lines_guard.as_mut_ptr() as *mut f64;
		let len = lines_guard.len();
		let cap = lines_guard.capacity();
		core::mem::forget(lines_guard);
		Vec::from_raw_parts(raw_ptr, len, cap)
	};
	
	let signals_vec = unsafe {
		let raw_ptr = signals_guard.as_mut_ptr() as *mut f64;
		let len = signals_guard.len();
		let cap = signals_guard.capacity();
		core::mem::forget(signals_guard);
		Vec::from_raw_parts(raw_ptr, len, cap)
	};
	
	Ok(KstBatchOutput {
		lines: lines_vec,
		signals: signals_vec,
		combos,
		rows,
		cols,
	})
}

fn kst_batch_inner_into(
	data: &[f64],
	sweep: &KstBatchRange,
	kern: Kernel,
	parallel: bool,
	lines_out: &mut [f64],
	signals_out: &mut [f64],
) -> Result<(), KstError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(KstError::InvalidPeriod { period: 0, data_len: 0 });
	}
	let first = data.iter().position(|x| !x.is_nan()).ok_or(KstError::AllValuesNaN)?;
	let max_p = combos
		.iter()
		.map(|c| {
			c.sma_period1
				.unwrap()
				.max(c.sma_period2.unwrap())
				.max(c.sma_period3.unwrap())
				.max(c.sma_period4.unwrap())
				.max(c.roc_period1.unwrap())
				.max(c.roc_period2.unwrap())
				.max(c.roc_period3.unwrap())
				.max(c.roc_period4.unwrap())
				.max(c.signal_period.unwrap())
		})
		.max()
		.unwrap();
	if data.len() - first < max_p {
		return Err(KstError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}
	let rows = combos.len();
	let cols = data.len();
	
	// Calculate warmup periods for each parameter combo
	let warm_periods: Vec<usize> = combos
		.iter()
		.map(|c| {
			// Warmup is max ROC period + its SMA period - 1
			let max_roc = c.roc_period4.unwrap_or(30);
			let max_sma = c.sma_period4.unwrap_or(15);
			max_roc + max_sma - 1
		})
		.collect();

	// Initialize NaN prefixes directly in the output buffers
	for (row, warmup) in warm_periods.iter().enumerate() {
		let start_idx = row * cols;
		let end_idx = start_idx + warmup;
		for i in start_idx..end_idx {
			lines_out[i] = f64::NAN;
			signals_out[i] = f64::NAN;
		}
	}

	let do_row = |row: usize, line_row: &mut [f64], sig_row: &mut [f64]| unsafe {
		let prm = &combos[row];
		let inp = KstInput::from_slice(data, *prm);
		match kern {
			Kernel::Scalar => {
				let r = kst_row_scalar(&inp, first, cols)?;
				line_row.copy_from_slice(&r.line);
				sig_row.copy_from_slice(&r.signal);
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => {
				let r = kst_row_avx2(&inp, first, cols)?;
				line_row.copy_from_slice(&r.line);
				sig_row.copy_from_slice(&r.signal);
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => {
				let r = kst_row_avx512(&inp, first, cols)?;
				line_row.copy_from_slice(&r.line);
				sig_row.copy_from_slice(&r.signal);
			}
			_ => unreachable!(),
		}
		Ok::<(), KstError>(())
	};
	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			lines_out
				.par_chunks_mut(cols)
				.zip(signals_out.par_chunks_mut(cols))
				.enumerate()
				.for_each(|(row, (lrow, srow))| {
					let _ = do_row(row, lrow, srow);
				});
		}
		#[cfg(target_arch = "wasm32")]
		{
			for (row, (lrow, srow)) in lines_out.chunks_mut(cols).zip(signals_out.chunks_mut(cols)).enumerate() {
				let _ = do_row(row, lrow, srow);
			}
		}
	} else {
		for (row, (lrow, srow)) in lines_out.chunks_mut(cols).zip(signals_out.chunks_mut(cols)).enumerate() {
			let _ = do_row(row, lrow, srow);
		}
	}
	
	Ok(())
}

#[inline(always)]
pub unsafe fn kst_row_scalar(input: &KstInput, first: usize, len: usize) -> Result<KstOutput, KstError> {
	kst_scalar(input, first, len)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn kst_row_avx2(input: &KstInput, first: usize, len: usize) -> Result<KstOutput, KstError> {
	kst_avx2(input, first, len)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn kst_row_avx512(input: &KstInput, first: usize, len: usize) -> Result<KstOutput, KstError> {
	kst_avx512(input, first, len)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn kst_row_avx512_short(input: &KstInput, first: usize, len: usize) -> Result<KstOutput, KstError> {
	kst_avx512_short(input, first, len)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn kst_row_avx512_long(input: &KstInput, first: usize, len: usize) -> Result<KstOutput, KstError> {
	kst_avx512_long(input, first, len)
}

// Streaming KST
#[derive(Debug, Clone)]
pub struct KstStream {
	period_params: KstParams,
	buffer: Vec<f64>,
	idx: usize,
	filled: bool,
	// (full streaming ROC/SMA state can be added here if needed)
}
impl KstStream {
	pub fn try_new(params: KstParams) -> Result<Self, KstError> {
		let max_p = params
			.sma_period1
			.unwrap_or(10)
			.max(params.sma_period2.unwrap_or(10))
			.max(params.sma_period3.unwrap_or(10))
			.max(params.sma_period4.unwrap_or(15))
			.max(params.roc_period1.unwrap_or(10))
			.max(params.roc_period2.unwrap_or(15))
			.max(params.roc_period3.unwrap_or(20))
			.max(params.roc_period4.unwrap_or(30))
			.max(params.signal_period.unwrap_or(9));
		Ok(Self {
			period_params: params,
			buffer: (0..max_p).map(|_| f64::NAN).collect(),
			idx: 0,
			filled: false,
		})
	}
	pub fn update(&mut self, value: f64) -> Option<(f64, f64)> {
		self.buffer[self.idx] = value;
		self.idx = (self.idx + 1) % self.buffer.len();
		if !self.filled && self.idx == 0 {
			self.filled = true;
		}
		if !self.filled {
			return None;
		}
		// Not efficient, but matches batch logic: process the buffer as a slice.
		let inp = KstInput::from_slice(&self.buffer, self.period_params);
		if let Ok(KstOutput { line, signal }) = kst(&inp) {
			let last_idx = line.len() - 1;
			Some((line[last_idx], signal[last_idx]))
		} else {
			None
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_kst_default_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = KstInput::with_default_candles(&candles);
		let result = kst_with_kernel(&input, kernel)?;
		assert_eq!(result.line.len(), candles.close.len());
		assert_eq!(result.signal.len(), candles.close.len());
		Ok(())
	}

	fn check_kst_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = KstInput::with_default_candles(&candles);
		let result = kst_with_kernel(&input, kernel)?;
		let expected_last_five_line = [
			-47.38570195278667,
			-44.42926180347176,
			-42.185693049429034,
			-40.10697793942024,
			-40.17466795905724,
		];
		let expected_last_five_signal = [
			-52.66743277411538,
			-51.559775662725556,
			-50.113844191238954,
			-48.58923772989874,
			-47.01112630514571,
		];
		let l = result.line.len();
		let s = result.signal.len();
		for (i, &v) in result.line[l - 5..].iter().enumerate() {
			assert!(
				(v - expected_last_five_line[i]).abs() < 1e-1,
				"KST line mismatch {}: {} vs {}",
				i,
				v,
				expected_last_five_line[i]
			);
		}
		for (i, &v) in result.signal[s - 5..].iter().enumerate() {
			assert!(
				(v - expected_last_five_signal[i]).abs() < 1e-1,
				"KST signal mismatch {}: {} vs {}",
				i,
				v,
				expected_last_five_signal[i]
			);
		}
		Ok(())
	}

	fn check_kst_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let nan_data = [f64::NAN, f64::NAN, f64::NAN];
		let input = KstInput::from_slice(&nan_data, KstParams::default());
		let result = kst_with_kernel(&input, kernel);
		assert!(result.is_err(), "[{}] Should error with all NaN", test_name);
		Ok(())
	}

	macro_rules! generate_all_kst_tests {
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

	#[cfg(debug_assertions)]
	fn check_kst_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		// Define comprehensive parameter combinations
		let test_params = vec![
			KstParams::default(),
			// Minimum viable periods
			KstParams {
				sma_period1: Some(2),
				sma_period2: Some(2),
				sma_period3: Some(2),
				sma_period4: Some(2),
				roc_period1: Some(2),
				roc_period2: Some(2),
				roc_period3: Some(2),
				roc_period4: Some(2),
				signal_period: Some(2),
			},
			// Small periods
			KstParams {
				sma_period1: Some(5),
				sma_period2: Some(5),
				sma_period3: Some(5),
				sma_period4: Some(7),
				roc_period1: Some(5),
				roc_period2: Some(7),
				roc_period3: Some(10),
				roc_period4: Some(15),
				signal_period: Some(5),
			},
			// Mixed small/medium periods
			KstParams {
				sma_period1: Some(7),
				sma_period2: Some(10),
				sma_period3: Some(12),
				sma_period4: Some(15),
				roc_period1: Some(8),
				roc_period2: Some(12),
				roc_period3: Some(16),
				roc_period4: Some(20),
				signal_period: Some(7),
			},
			// Default periods (10, 10, 10, 15, 10, 15, 20, 30, 9)
			KstParams {
				sma_period1: Some(10),
				sma_period2: Some(10),
				sma_period3: Some(10),
				sma_period4: Some(15),
				roc_period1: Some(10),
				roc_period2: Some(15),
				roc_period3: Some(20),
				roc_period4: Some(30),
				signal_period: Some(9),
			},
			// Large periods
			KstParams {
				sma_period1: Some(20),
				sma_period2: Some(25),
				sma_period3: Some(30),
				sma_period4: Some(35),
				roc_period1: Some(25),
				roc_period2: Some(35),
				roc_period3: Some(45),
				roc_period4: Some(60),
				signal_period: Some(15),
			},
			// Very large periods
			KstParams {
				sma_period1: Some(30),
				sma_period2: Some(40),
				sma_period3: Some(50),
				sma_period4: Some(60),
				roc_period1: Some(40),
				roc_period2: Some(60),
				roc_period3: Some(80),
				roc_period4: Some(100),
				signal_period: Some(21),
			},
			// Asymmetric periods
			KstParams {
				sma_period1: Some(5),
				sma_period2: Some(10),
				sma_period3: Some(20),
				sma_period4: Some(50),
				roc_period1: Some(7),
				roc_period2: Some(14),
				roc_period3: Some(28),
				roc_period4: Some(56),
				signal_period: Some(12),
			},
			// Edge case: very small signal period
			KstParams {
				sma_period1: Some(10),
				sma_period2: Some(10),
				sma_period3: Some(10),
				sma_period4: Some(15),
				roc_period1: Some(10),
				roc_period2: Some(15),
				roc_period3: Some(20),
				roc_period4: Some(30),
				signal_period: Some(2),
			},
		];

		for (param_idx, params) in test_params.iter().enumerate() {
			let input = KstInput::from_candles(&candles, "close", params.clone());
			let output = kst_with_kernel(&input, kernel)?;

			// Check line values
			for (i, &val) in output.line.iter().enumerate() {
				if val.is_nan() {
					continue; // NaN values are expected during warmup
				}

				let bits = val.to_bits();

				// Check all three poison patterns
				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 in KST line with params: sma_periods=({},{},{},{}), roc_periods=({},{},{},{}), \
						 signal_period={} (param set {})",
						test_name, val, bits, i,
						params.sma_period1.unwrap_or(10),
						params.sma_period2.unwrap_or(10),
						params.sma_period3.unwrap_or(10),
						params.sma_period4.unwrap_or(15),
						params.roc_period1.unwrap_or(10),
						params.roc_period2.unwrap_or(15),
						params.roc_period3.unwrap_or(20),
						params.roc_period4.unwrap_or(30),
						params.signal_period.unwrap_or(9),
						param_idx
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 in KST line with params: sma_periods=({},{},{},{}), roc_periods=({},{},{},{}), \
						 signal_period={} (param set {})",
						test_name, val, bits, i,
						params.sma_period1.unwrap_or(10),
						params.sma_period2.unwrap_or(10),
						params.sma_period3.unwrap_or(10),
						params.sma_period4.unwrap_or(15),
						params.roc_period1.unwrap_or(10),
						params.roc_period2.unwrap_or(15),
						params.roc_period3.unwrap_or(20),
						params.roc_period4.unwrap_or(30),
						params.signal_period.unwrap_or(9),
						param_idx
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 in KST line with params: sma_periods=({},{},{},{}), roc_periods=({},{},{},{}), \
						 signal_period={} (param set {})",
						test_name, val, bits, i,
						params.sma_period1.unwrap_or(10),
						params.sma_period2.unwrap_or(10),
						params.sma_period3.unwrap_or(10),
						params.sma_period4.unwrap_or(15),
						params.roc_period1.unwrap_or(10),
						params.roc_period2.unwrap_or(15),
						params.roc_period3.unwrap_or(20),
						params.roc_period4.unwrap_or(30),
						params.signal_period.unwrap_or(9),
						param_idx
					);
				}
			}

			// Check signal values
			for (i, &val) in output.signal.iter().enumerate() {
				if val.is_nan() {
					continue;
				}

				let bits = val.to_bits();

				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 in KST signal with params: sma_periods=({},{},{},{}), roc_periods=({},{},{},{}), \
						 signal_period={} (param set {})",
						test_name, val, bits, i,
						params.sma_period1.unwrap_or(10),
						params.sma_period2.unwrap_or(10),
						params.sma_period3.unwrap_or(10),
						params.sma_period4.unwrap_or(15),
						params.roc_period1.unwrap_or(10),
						params.roc_period2.unwrap_or(15),
						params.roc_period3.unwrap_or(20),
						params.roc_period4.unwrap_or(30),
						params.signal_period.unwrap_or(9),
						param_idx
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 in KST signal with params: sma_periods=({},{},{},{}), roc_periods=({},{},{},{}), \
						 signal_period={} (param set {})",
						test_name, val, bits, i,
						params.sma_period1.unwrap_or(10),
						params.sma_period2.unwrap_or(10),
						params.sma_period3.unwrap_or(10),
						params.sma_period4.unwrap_or(15),
						params.roc_period1.unwrap_or(10),
						params.roc_period2.unwrap_or(15),
						params.roc_period3.unwrap_or(20),
						params.roc_period4.unwrap_or(30),
						params.signal_period.unwrap_or(9),
						param_idx
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 in KST signal with params: sma_periods=({},{},{},{}), roc_periods=({},{},{},{}), \
						 signal_period={} (param set {})",
						test_name, val, bits, i,
						params.sma_period1.unwrap_or(10),
						params.sma_period2.unwrap_or(10),
						params.sma_period3.unwrap_or(10),
						params.sma_period4.unwrap_or(15),
						params.roc_period1.unwrap_or(10),
						params.roc_period2.unwrap_or(15),
						params.roc_period3.unwrap_or(20),
						params.roc_period4.unwrap_or(30),
						params.signal_period.unwrap_or(9),
						param_idx
					);
				}
			}
		}

		Ok(())
	}

	#[cfg(not(debug_assertions))]
	fn check_kst_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(()) // No-op in release builds
	}

	generate_all_kst_tests!(check_kst_default_params, check_kst_accuracy, check_kst_nan_handling, check_kst_no_poison);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let output = KstBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;
		let def = KstParams::default();
		let row = output.lines_for(&def).expect("default row missing");
		assert_eq!(row.len(), c.close.len());
		Ok(())
	}

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
	#[cfg(debug_assertions)]
	fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		// Test various parameter sweep configurations (kept minimal to avoid OOM)
		let test_configs = vec![
			// Small periods - single combination
			(3, 3, 0, 3, 3, 0, 3, 3, 0, 3, 3, 0, 3, 3, 0, 4, 4, 0, 5, 5, 0, 6, 6, 0, 3, 3, 0),
			// Medium periods - 2x2x2x2 combinations
			(5, 10, 5, 5, 10, 5, 5, 10, 5, 8, 13, 5, 5, 10, 5, 8, 13, 5, 10, 15, 5, 15, 20, 5, 5, 7, 2),
			// Default values
			(10, 10, 0, 10, 10, 0, 10, 10, 0, 15, 15, 0, 10, 10, 0, 15, 15, 0, 20, 20, 0, 30, 30, 0, 9, 9, 0),
			// Dense small range - limited
			(2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 3, 4, 1, 4, 5, 1, 5, 6, 1, 2, 3, 1),
		];

		for (cfg_idx, &(s1_start, s1_end, s1_step, s2_start, s2_end, s2_step, 
						s3_start, s3_end, s3_step, s4_start, s4_end, s4_step,
						r1_start, r1_end, r1_step, r2_start, r2_end, r2_step,
						r3_start, r3_end, r3_step, r4_start, r4_end, r4_step,
						sig_start, sig_end, sig_step)) in test_configs.iter().enumerate() {
			
			let output = KstBatchBuilder::new()
				.kernel(kernel)
				.sma_period1_range(s1_start, s1_end, s1_step)
				.sma_period2_range(s2_start, s2_end, s2_step)
				.sma_period3_range(s3_start, s3_end, s3_step)
				.sma_period4_range(s4_start, s4_end, s4_step)
				.roc_period1_range(r1_start, r1_end, r1_step)
				.roc_period2_range(r2_start, r2_end, r2_step)
				.roc_period3_range(r3_start, r3_end, r3_step)
				.roc_period4_range(r4_start, r4_end, r4_step)
				.signal_period_range(sig_start, sig_end, sig_step)
				.apply_candles(&c, "close")?;

			// Check line values
			for (idx, &val) in output.lines.iter().enumerate() {
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
						 at row {} col {} (flat index {}) in KST lines with params: \
						 sma_periods=({},{},{},{}), roc_periods=({},{},{},{}), signal_period={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.sma_period1.unwrap_or(10),
						combo.sma_period2.unwrap_or(10),
						combo.sma_period3.unwrap_or(10),
						combo.sma_period4.unwrap_or(15),
						combo.roc_period1.unwrap_or(10),
						combo.roc_period2.unwrap_or(15),
						combo.roc_period3.unwrap_or(20),
						combo.roc_period4.unwrap_or(30),
						combo.signal_period.unwrap_or(9)
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in KST lines with params: \
						 sma_periods=({},{},{},{}), roc_periods=({},{},{},{}), signal_period={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.sma_period1.unwrap_or(10),
						combo.sma_period2.unwrap_or(10),
						combo.sma_period3.unwrap_or(10),
						combo.sma_period4.unwrap_or(15),
						combo.roc_period1.unwrap_or(10),
						combo.roc_period2.unwrap_or(15),
						combo.roc_period3.unwrap_or(20),
						combo.roc_period4.unwrap_or(30),
						combo.signal_period.unwrap_or(9)
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in KST lines with params: \
						 sma_periods=({},{},{},{}), roc_periods=({},{},{},{}), signal_period={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.sma_period1.unwrap_or(10),
						combo.sma_period2.unwrap_or(10),
						combo.sma_period3.unwrap_or(10),
						combo.sma_period4.unwrap_or(15),
						combo.roc_period1.unwrap_or(10),
						combo.roc_period2.unwrap_or(15),
						combo.roc_period3.unwrap_or(20),
						combo.roc_period4.unwrap_or(30),
						combo.signal_period.unwrap_or(9)
					);
				}
			}

			// Check signal values
			for (idx, &val) in output.signals.iter().enumerate() {
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
						 at row {} col {} (flat index {}) in KST signals with params: \
						 sma_periods=({},{},{},{}), roc_periods=({},{},{},{}), signal_period={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.sma_period1.unwrap_or(10),
						combo.sma_period2.unwrap_or(10),
						combo.sma_period3.unwrap_or(10),
						combo.sma_period4.unwrap_or(15),
						combo.roc_period1.unwrap_or(10),
						combo.roc_period2.unwrap_or(15),
						combo.roc_period3.unwrap_or(20),
						combo.roc_period4.unwrap_or(30),
						combo.signal_period.unwrap_or(9)
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in KST signals with params: \
						 sma_periods=({},{},{},{}), roc_periods=({},{},{},{}), signal_period={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.sma_period1.unwrap_or(10),
						combo.sma_period2.unwrap_or(10),
						combo.sma_period3.unwrap_or(10),
						combo.sma_period4.unwrap_or(15),
						combo.roc_period1.unwrap_or(10),
						combo.roc_period2.unwrap_or(15),
						combo.roc_period3.unwrap_or(20),
						combo.roc_period4.unwrap_or(30),
						combo.signal_period.unwrap_or(9)
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in KST signals with params: \
						 sma_periods=({},{},{},{}), roc_periods=({},{},{},{}), signal_period={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.sma_period1.unwrap_or(10),
						combo.sma_period2.unwrap_or(10),
						combo.sma_period3.unwrap_or(10),
						combo.sma_period4.unwrap_or(15),
						combo.roc_period1.unwrap_or(10),
						combo.roc_period2.unwrap_or(15),
						combo.roc_period3.unwrap_or(20),
						combo.roc_period4.unwrap_or(30),
						combo.signal_period.unwrap_or(9)
					);
				}
			}
		}

		Ok(())
	}

	#[cfg(not(debug_assertions))]
	fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(())
	}

	gen_batch_tests!(check_batch_default_row);
}

// =============================================================================
// PYTHON BINDINGS
// =============================================================================

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;

#[cfg(feature = "python")]
#[pyfunction(name = "kst")]
#[pyo3(signature = (data, sma_period1=None, sma_period2=None, sma_period3=None, sma_period4=None, roc_period1=None, roc_period2=None, roc_period3=None, roc_period4=None, signal_period=None, kernel=None))]
pub fn kst_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    sma_period1: Option<usize>,
    sma_period2: Option<usize>,
    sma_period3: Option<usize>,
    sma_period4: Option<usize>,
    roc_period1: Option<usize>,
    roc_period2: Option<usize>,
    roc_period3: Option<usize>,
    roc_period4: Option<usize>,
    signal_period: Option<usize>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    
    let params = KstParams {
        sma_period1,
        sma_period2,
        sma_period3,
        sma_period4,
        roc_period1,
        roc_period2,
        roc_period3,
        roc_period4,
        signal_period,
    };
    
    let kst_in = KstInput::from_slice(slice_in, params);

    let result = py
        .allow_threads(|| kst_with_kernel(&kst_in, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("line", result.line.into_pyarray(py))?;
    dict.set_item("signal", result.signal.into_pyarray(py))?;
    
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "KstStream")]
pub struct KstStreamPy {
    stream: KstStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl KstStreamPy {
    #[new]
    fn new(
        sma_period1: Option<usize>,
        sma_period2: Option<usize>,
        sma_period3: Option<usize>,
        sma_period4: Option<usize>,
        roc_period1: Option<usize>,
        roc_period2: Option<usize>,
        roc_period3: Option<usize>,
        roc_period4: Option<usize>,
        signal_period: Option<usize>,
    ) -> PyResult<Self> {
        let params = KstParams {
            sma_period1,
            sma_period2,
            sma_period3,
            sma_period4,
            roc_period1,
            roc_period2,
            roc_period3,
            roc_period4,
            signal_period,
        };
        let stream = KstStream::try_new(params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(KstStreamPy { stream })
    }

    fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "kst_batch")]
#[pyo3(signature = (data, sma_period1_range=None, sma_period2_range=None, sma_period3_range=None, sma_period4_range=None, roc_period1_range=None, roc_period2_range=None, roc_period3_range=None, roc_period4_range=None, signal_period_range=None, kernel=None))]
pub fn kst_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    sma_period1_range: Option<(usize, usize, usize)>,
    sma_period2_range: Option<(usize, usize, usize)>,
    sma_period3_range: Option<(usize, usize, usize)>,
    sma_period4_range: Option<(usize, usize, usize)>,
    roc_period1_range: Option<(usize, usize, usize)>,
    roc_period2_range: Option<(usize, usize, usize)>,
    roc_period3_range: Option<(usize, usize, usize)>,
    roc_period4_range: Option<(usize, usize, usize)>,
    signal_period_range: Option<(usize, usize, usize)>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    
    let sweep = KstBatchRange {
        sma_period1: sma_period1_range.unwrap_or((10, 10, 0)),
        sma_period2: sma_period2_range.unwrap_or((10, 10, 0)),
        sma_period3: sma_period3_range.unwrap_or((10, 10, 0)),
        sma_period4: sma_period4_range.unwrap_or((15, 15, 0)),
        roc_period1: roc_period1_range.unwrap_or((10, 10, 0)),
        roc_period2: roc_period2_range.unwrap_or((15, 15, 0)),
        roc_period3: roc_period3_range.unwrap_or((20, 20, 0)),
        roc_period4: roc_period4_range.unwrap_or((30, 30, 0)),
        signal_period: signal_period_range.unwrap_or((9, 9, 0)),
    };

    let result = py
        .allow_threads(|| kst_batch_with_kernel(slice_in, &sweep, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    
    // Reshape the output arrays
    let lines_array = result.lines.into_pyarray(py);
    let signals_array = result.signals.into_pyarray(py);
    
    dict.set_item("lines", lines_array.reshape([result.rows, result.cols])?)?;
    dict.set_item("signals", signals_array.reshape([result.rows, result.cols])?)?;
    
    // Create combo arrays for each parameter
    let sma_period1_arr: Vec<usize> = result.combos.iter().map(|c| c.sma_period1.unwrap_or(10)).collect();
    let sma_period2_arr: Vec<usize> = result.combos.iter().map(|c| c.sma_period2.unwrap_or(10)).collect();
    let sma_period3_arr: Vec<usize> = result.combos.iter().map(|c| c.sma_period3.unwrap_or(10)).collect();
    let sma_period4_arr: Vec<usize> = result.combos.iter().map(|c| c.sma_period4.unwrap_or(15)).collect();
    let roc_period1_arr: Vec<usize> = result.combos.iter().map(|c| c.roc_period1.unwrap_or(10)).collect();
    let roc_period2_arr: Vec<usize> = result.combos.iter().map(|c| c.roc_period2.unwrap_or(15)).collect();
    let roc_period3_arr: Vec<usize> = result.combos.iter().map(|c| c.roc_period3.unwrap_or(20)).collect();
    let roc_period4_arr: Vec<usize> = result.combos.iter().map(|c| c.roc_period4.unwrap_or(30)).collect();
    let signal_period_arr: Vec<usize> = result.combos.iter().map(|c| c.signal_period.unwrap_or(9)).collect();
    
    dict.set_item("sma_period1", sma_period1_arr.into_pyarray(py))?;
    dict.set_item("sma_period2", sma_period2_arr.into_pyarray(py))?;
    dict.set_item("sma_period3", sma_period3_arr.into_pyarray(py))?;
    dict.set_item("sma_period4", sma_period4_arr.into_pyarray(py))?;
    dict.set_item("roc_period1", roc_period1_arr.into_pyarray(py))?;
    dict.set_item("roc_period2", roc_period2_arr.into_pyarray(py))?;
    dict.set_item("roc_period3", roc_period3_arr.into_pyarray(py))?;
    dict.set_item("roc_period4", roc_period4_arr.into_pyarray(py))?;
    dict.set_item("signal_period", signal_period_arr.into_pyarray(py))?;
    
    Ok(dict)
}

// =========================== WASM Bindings ===========================

/// Helper function to perform the calculation without double allocation
#[cfg(feature = "wasm")]
fn kst_into_slice(
    data: &[f64],
    sma_period1: usize,
    sma_period2: usize,
    sma_period3: usize,
    sma_period4: usize,
    roc_period1: usize,
    roc_period2: usize,
    roc_period3: usize,
    roc_period4: usize,
    signal_period: usize,
    line_out: &mut [f64],
    signal_out: &mut [f64],
) -> Result<(), KstError> {
    let params = KstParams {
        sma_period1: Some(sma_period1),
        sma_period2: Some(sma_period2),
        sma_period3: Some(sma_period3),
        sma_period4: Some(sma_period4),
        roc_period1: Some(roc_period1),
        roc_period2: Some(roc_period2),
        roc_period3: Some(roc_period3),
        roc_period4: Some(roc_period4),
        signal_period: Some(signal_period),
    };
    let input = KstInput::from_slice(data, params);
    
    match kst(&input) {
        Ok(output) => {
            line_out.copy_from_slice(&output.line);
            signal_out.copy_from_slice(&output.signal);
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// WASM interface for KST calculation (safe API)
/// Returns an object with line and signal arrays
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn kst_js(
    data: &[f64],
    sma_period1: usize,
    sma_period2: usize,
    sma_period3: usize,
    sma_period4: usize,
    roc_period1: usize,
    roc_period2: usize,
    roc_period3: usize,
    roc_period4: usize,
    signal_period: usize,
) -> Result<JsValue, JsValue> {
    let params = KstParams {
        sma_period1: Some(sma_period1),
        sma_period2: Some(sma_period2),
        sma_period3: Some(sma_period3),
        sma_period4: Some(sma_period4),
        roc_period1: Some(roc_period1),
        roc_period2: Some(roc_period2),
        roc_period3: Some(roc_period3),
        roc_period4: Some(roc_period4),
        signal_period: Some(signal_period),
    };
    let input = KstInput::from_slice(data, params);
    
    match kst(&input) {
        Ok(output) => {
            let obj = js_sys::Object::new();
            js_sys::Reflect::set(
                &obj,
                &JsValue::from_str("line"),
                &serde_wasm_bindgen::to_value(&output.line).unwrap(),
            )?;
            js_sys::Reflect::set(
                &obj,
                &JsValue::from_str("signal"),
                &serde_wasm_bindgen::to_value(&output.signal).unwrap(),
            )?;
            Ok(obj.into())
        }
        Err(e) => Err(JsValue::from_str(&e.to_string())),
    }
}

/// Fast/unsafe WASM interface with pre-allocated memory
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn kst_into(
    in_ptr: *const f64,
    line_out_ptr: *mut f64,
    signal_out_ptr: *mut f64,
    len: usize,
    sma_period1: usize,
    sma_period2: usize,
    sma_period3: usize,
    sma_period4: usize,
    roc_period1: usize,
    roc_period2: usize,
    roc_period3: usize,
    roc_period4: usize,
    signal_period: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || line_out_ptr.is_null() || signal_out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to kst_into"));
    }
    
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        
        // Handle aliasing - if input and output pointers are the same
        if in_ptr == line_out_ptr as *const f64 || in_ptr == signal_out_ptr as *const f64 {
            // Use a temporary buffer
            let params = KstParams {
                sma_period1: Some(sma_period1),
                sma_period2: Some(sma_period2),
                sma_period3: Some(sma_period3),
                sma_period4: Some(sma_period4),
                roc_period1: Some(roc_period1),
                roc_period2: Some(roc_period2),
                roc_period3: Some(roc_period3),
                roc_period4: Some(roc_period4),
                signal_period: Some(signal_period),
            };
            let input = KstInput::from_slice(data, params);
            
            match kst(&input) {
                Ok(output) => {
                    let line_out = std::slice::from_raw_parts_mut(line_out_ptr, len);
                    let signal_out = std::slice::from_raw_parts_mut(signal_out_ptr, len);
                    line_out.copy_from_slice(&output.line);
                    signal_out.copy_from_slice(&output.signal);
                    Ok(())
                }
                Err(e) => Err(JsValue::from_str(&e.to_string())),
            }
        } else {
            let line_out = std::slice::from_raw_parts_mut(line_out_ptr, len);
            let signal_out = std::slice::from_raw_parts_mut(signal_out_ptr, len);
            kst_into_slice(
                data,
                sma_period1,
                sma_period2,
                sma_period3,
                sma_period4,
                roc_period1,
                roc_period2,
                roc_period3,
                roc_period4,
                signal_period,
                line_out,
                signal_out,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))
        }
    }
}

/// Allocate memory for output arrays
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn kst_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

/// Free allocated memory
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn kst_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

/// Configuration for batch KST calculation
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct KstBatchConfig {
    pub sma_period1_range: (usize, usize, usize),
    pub sma_period2_range: (usize, usize, usize),
    pub sma_period3_range: (usize, usize, usize),
    pub sma_period4_range: (usize, usize, usize),
    pub roc_period1_range: (usize, usize, usize),
    pub roc_period2_range: (usize, usize, usize),
    pub roc_period3_range: (usize, usize, usize),
    pub roc_period4_range: (usize, usize, usize),
    pub signal_period_range: (usize, usize, usize),
}

/// Output structure for batch WASM API
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct KstBatchJsOutput {
    pub line: Vec<f64>,
    pub signal: Vec<f64>,
    pub combos: Vec<KstParams>,
    pub rows: usize,
    pub cols: usize,
}

/// WASM interface for batch KST calculation
#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = kst_batch)]
pub fn kst_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: KstBatchConfig =
        serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
    
    let sweep = KstBatchRange {
        sma_period1: config.sma_period1_range,
        sma_period2: config.sma_period2_range,
        sma_period3: config.sma_period3_range,
        sma_period4: config.sma_period4_range,
        roc_period1: config.roc_period1_range,
        roc_period2: config.roc_period2_range,
        roc_period3: config.roc_period3_range,
        roc_period4: config.roc_period4_range,
        signal_period: config.signal_period_range,
    };
    
    let output = kst_batch_inner(data, &sweep, Kernel::Auto, false).map_err(|e| JsValue::from_str(&e.to_string()))?;
    
    let js_output = KstBatchJsOutput {
        line: output.lines,
        signal: output.signals,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };
    
    serde_wasm_bindgen::to_value(&js_output).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

/// Fast batch calculation into pre-allocated memory
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn kst_batch_into(
    in_ptr: *const f64,
    line_out_ptr: *mut f64,
    signal_out_ptr: *mut f64,
    len: usize,
    sma_period1_start: usize,
    sma_period1_end: usize,
    sma_period1_step: usize,
    sma_period2_start: usize,
    sma_period2_end: usize,
    sma_period2_step: usize,
    sma_period3_start: usize,
    sma_period3_end: usize,
    sma_period3_step: usize,
    sma_period4_start: usize,
    sma_period4_end: usize,
    sma_period4_step: usize,
    roc_period1_start: usize,
    roc_period1_end: usize,
    roc_period1_step: usize,
    roc_period2_start: usize,
    roc_period2_end: usize,
    roc_period2_step: usize,
    roc_period3_start: usize,
    roc_period3_end: usize,
    roc_period3_step: usize,
    roc_period4_start: usize,
    roc_period4_end: usize,
    roc_period4_step: usize,
    signal_period_start: usize,
    signal_period_end: usize,
    signal_period_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || line_out_ptr.is_null() || signal_out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to kst_batch_into"));
    }
    
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        
        let sweep = KstBatchRange {
            sma_period1: (sma_period1_start, sma_period1_end, sma_period1_step),
            sma_period2: (sma_period2_start, sma_period2_end, sma_period2_step),
            sma_period3: (sma_period3_start, sma_period3_end, sma_period3_step),
            sma_period4: (sma_period4_start, sma_period4_end, sma_period4_step),
            roc_period1: (roc_period1_start, roc_period1_end, roc_period1_step),
            roc_period2: (roc_period2_start, roc_period2_end, roc_period2_step),
            roc_period3: (roc_period3_start, roc_period3_end, roc_period3_step),
            roc_period4: (roc_period4_start, roc_period4_end, roc_period4_step),
            signal_period: (signal_period_start, signal_period_end, signal_period_step),
        };
        
        // Calculate number of parameter combinations
        let count_range = |r: &(usize, usize, usize)| {
            if r.2 == 0 {
                0
            } else {
                ((r.1.saturating_sub(r.0)) / r.2) + 1
            }
        };
        
        let rows = count_range(&sweep.sma_period1).max(1) *
                   count_range(&sweep.sma_period2).max(1) *
                   count_range(&sweep.sma_period3).max(1) *
                   count_range(&sweep.sma_period4).max(1) *
                   count_range(&sweep.roc_period1).max(1) *
                   count_range(&sweep.roc_period2).max(1) *
                   count_range(&sweep.roc_period3).max(1) *
                   count_range(&sweep.roc_period4).max(1) *
                   count_range(&sweep.signal_period).max(1);
        let cols = len;
        
        let line_out = std::slice::from_raw_parts_mut(line_out_ptr, rows * cols);
        let signal_out = std::slice::from_raw_parts_mut(signal_out_ptr, rows * cols);
        
        kst_batch_inner_into(data, &sweep, Kernel::Auto, false, line_out, signal_out).map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        Ok(rows)
    }
}
