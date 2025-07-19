//! # Jurik Moving Average (JMA)
//!
//! A minimal-lag smoothing methodology developed by Mark Jurik. JMA adapts quickly
//! to market moves while reducing noise. Parameters (`period`, `phase`, `power`)
//! control window size, phase shift, and smoothing aggressiveness.
//!
//! ## Parameters
//! - **period**: Window size (number of data points).
//! - **phase**: Shift in [-100.0, 100.0], curve displacement (default: 50.0).
//! - **power**: Exponent for smoothing ratio (default: 2).
//!
//! ## Errors
//! - **AllValuesNaN**: jma: All input data values are `NaN`.
//! - **InvalidPeriod**: jma: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: jma: Not enough valid data points for the requested `period`.
//! - **InvalidPhase**: jma: `phase` is `NaN` or infinite.
//!
//! ## Returns
//! - **`Ok(JmaOutput)`** on success, containing a `Vec<f64>`.
//! - **`Err(JmaError)`** otherwise.

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
use std::mem::MaybeUninit;
use thiserror::Error;

impl<'a> AsRef<[f64]> for JmaInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			JmaData::Slice(slice) => slice,
			JmaData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

#[derive(Debug, Clone)]
pub enum JmaData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct JmaOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
	all(target_arch = "wasm32", feature = "wasm"),
	derive(serde::Serialize, serde::Deserialize)
)]
pub struct JmaParams {
	pub period: Option<usize>,
	pub phase: Option<f64>,
	pub power: Option<u32>,
}

impl Default for JmaParams {
	fn default() -> Self {
		Self {
			period: Some(7),
			phase: Some(50.0),
			power: Some(2),
		}
	}
}

#[derive(Debug, Clone)]
pub struct JmaInput<'a> {
	pub data: JmaData<'a>,
	pub params: JmaParams,
}

impl<'a> JmaInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: JmaParams) -> Self {
		Self {
			data: JmaData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slice(sl: &'a [f64], p: JmaParams) -> Self {
		Self {
			data: JmaData::Slice(sl),
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", JmaParams::default())
	}
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(7)
	}
	#[inline]
	pub fn get_phase(&self) -> f64 {
		self.params.phase.unwrap_or(50.0)
	}
	#[inline]
	pub fn get_power(&self) -> u32 {
		self.params.power.unwrap_or(2)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct JmaBuilder {
	period: Option<usize>,
	phase: Option<f64>,
	power: Option<u32>,
	kernel: Kernel,
}

impl Default for JmaBuilder {
	fn default() -> Self {
		Self {
			period: None,
			phase: None,
			power: None,
			kernel: Kernel::Auto,
		}
	}
}

impl JmaBuilder {
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
	pub fn phase(mut self, x: f64) -> Self {
		self.phase = Some(x);
		self
	}
	#[inline(always)]
	pub fn power(mut self, p: u32) -> Self {
		self.power = Some(p);
		self
	}
	#[inline(always)]
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}

	#[inline(always)]
	pub fn apply(self, c: &Candles) -> Result<JmaOutput, JmaError> {
		let p = JmaParams {
			period: self.period,
			phase: self.phase,
			power: self.power,
		};
		let i = JmaInput::from_candles(c, "close", p);
		jma_with_kernel(&i, self.kernel)
	}

	#[inline(always)]
	pub fn apply_slice(self, d: &[f64]) -> Result<JmaOutput, JmaError> {
		let p = JmaParams {
			period: self.period,
			phase: self.phase,
			power: self.power,
		};
		let i = JmaInput::from_slice(d, p);
		jma_with_kernel(&i, self.kernel)
	}

	#[inline(always)]
	pub fn into_stream(self) -> Result<JmaStream, JmaError> {
		let p = JmaParams {
			period: self.period,
			phase: self.phase,
			power: self.power,
		};
		JmaStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum JmaError {
	#[error("jma: All values are NaN.")]
	AllValuesNaN,
	#[error("jma: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("jma: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("jma: Invalid phase: {phase}")]
	InvalidPhase { phase: f64 },
	#[error("jma: Invalid output buffer size: expected = {expected}, actual = {actual}")]
	InvalidOutputBuffer { expected: usize, actual: usize },
}

#[inline]
pub fn jma(input: &JmaInput) -> Result<JmaOutput, JmaError> {
	jma_with_kernel(input, Kernel::Auto)
}

pub fn jma_with_kernel(input: &JmaInput, kernel: Kernel) -> Result<JmaOutput, JmaError> {
	let data: &[f64] = match &input.data {
		JmaData::Candles { candles, source } => source_type(candles, source),
		JmaData::Slice(sl) => sl,
	};
	let first = data.iter().position(|x| !x.is_nan()).ok_or(JmaError::AllValuesNaN)?;
	let len = data.len();
	let period = input.get_period();
	let phase = input.get_phase();
	let power = input.get_power();

	if period == 0 || period > len {
		return Err(JmaError::InvalidPeriod { period, data_len: len });
	}
	if (len - first) < period {
		return Err(JmaError::NotEnoughValidData {
			needed: period,
			valid: len - first,
		});
	}
	if phase.is_nan() || phase.is_infinite() {
		return Err(JmaError::InvalidPhase { phase });
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	let warm = first + period; // first valid + look-back window
	let mut out = alloc_with_nan_prefix(len, warm);
	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => jma_scalar(data, period, phase, power, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => jma_avx2(data, period, phase, power, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => jma_avx512(data, period, phase, power, first, &mut out),
			_ => unreachable!(),
		}
	}
	Ok(JmaOutput { values: out })
}

pub fn jma_with_kernel_into(input: &JmaInput, kernel: Kernel, out: &mut [f64]) -> Result<(), JmaError> {
	let data: &[f64] = match &input.data {
		JmaData::Candles { candles, source } => source_type(candles, source),
		JmaData::Slice(sl) => sl,
	};
	let len = data.len();

	// Ensure output buffer is the correct size
	if out.len() != len {
		return Err(JmaError::InvalidOutputBuffer {
			expected: len,
			actual: out.len(),
		});
	}

	let first = data.iter().position(|x| !x.is_nan()).ok_or(JmaError::AllValuesNaN)?;
	let period = input.get_period();
	let phase = input.get_phase();
	let power = input.get_power();

	if period == 0 || period > len {
		return Err(JmaError::InvalidPeriod { period, data_len: len });
	}
	if (len - first) < period {
		return Err(JmaError::NotEnoughValidData {
			needed: period,
			valid: len - first,
		});
	}
	if phase.is_nan() || phase.is_infinite() {
		return Err(JmaError::InvalidPhase { phase });
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	let warm = first + period;
	// Initialize NaN prefix
	out[..warm].fill(f64::NAN);

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => jma_scalar(data, period, phase, power, first, out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => jma_avx2(data, period, phase, power, first, out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => jma_avx512(data, period, phase, power, first, out),
			_ => unreachable!(),
		}
	}
	Ok(())
}

#[inline]
pub fn jma_scalar(data: &[f64], period: usize, phase: f64, power: u32, first_valid: usize, output: &mut [f64]) {
	assert_eq!(data.len(), output.len());
	assert!(first_valid < data.len());

	let pr = if phase < -100.0 {
		0.5
	} else if phase > 100.0 {
		2.5
	} else {
		phase / 100.0 + 1.5
	};

	let beta = {
		let num = 0.45 * (period as f64 - 1.0);
		num / (num + 2.0)
	};
	let one_minus_beta = 1.0 - beta;

	let alpha = beta.powi(power as i32);
	let one_minus_alpha = 1.0 - alpha;
	let alpha_sq = alpha * alpha;
	let oma_sq = one_minus_alpha * one_minus_alpha;

	let mut e0 = data[first_valid];
	let mut e1 = 0.0;
	let mut e2 = 0.0;
	let mut j_prev = data[first_valid];

	output[first_valid] = j_prev;

	unsafe {
		for i in (first_valid + 1)..data.len() {
			let price = *data.get_unchecked(i);

			e0 = one_minus_alpha * price + alpha * e0;

			e1 = (price - e0) * one_minus_beta + beta * e1;
			let diff = e0 + pr * e1 - j_prev;

			e2 = diff * oma_sq + alpha_sq * e2;
			let j = j_prev + e2;

			*output.get_unchecked_mut(i) = j;

			j_prev = j;
		}
	}
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn jma_avx2(data: &[f64], period: usize, phase: f64, power: u32, first_valid: usize, output: &mut [f64]) {
	assert_eq!(data.len(), output.len());
	assert!(first_valid < data.len());

	let pr = if phase < -100.0 {
		0.5
	} else if phase > 100.0 {
		2.5
	} else {
		phase / 100.0 + 1.5
	};

	let beta = {
		let num = 0.45 * (period as f64 - 1.0);
		num / (num + 2.0)
	};
	let one_minus_beta = 1.0 - beta;

	let alpha = beta.powi(power as i32);
	let one_minus_alpha = 1.0 - alpha;
	let alpha_sq = alpha * alpha;
	let oma_sq = one_minus_alpha * one_minus_alpha;

	let mut e0 = data[first_valid];
	let mut e1 = 0.0;
	let mut e2 = 0.0;
	let mut j_prev = e0;

	output[first_valid] = j_prev;

	unsafe {
		for i in (first_valid + 1)..data.len() {
			let price = *data.get_unchecked(i);

			e0 = one_minus_alpha.mul_add(price, alpha * e0);
			e1 = (price - e0).mul_add(one_minus_beta, beta * e1);

			let diff = e0 + pr * e1 - j_prev;
			e2 = diff.mul_add(oma_sq, alpha_sq * e2);

			j_prev += e2;
			*output.get_unchecked_mut(i) = j_prev;
		}
	}
}

#[inline(always)]
fn jma_consts(period: usize, phase: f64, power: u32) -> (f64, f64, f64, f64, f64, f64, f64) {
	let pr = if phase < -100.0 {
		0.5
	} else if phase > 100.0 {
		2.5
	} else {
		phase / 100.0 + 1.5
	};

	let beta = {
		let num = 0.45 * (period as f64 - 1.0);
		num / (num + 2.0)
	};
	let alpha = beta.powi(power as i32);
	(
		pr,
		beta,
		alpha,
		alpha * alpha,
		(1.0 - alpha) * (1.0 - alpha),
		1.0 - alpha,
		1.0 - beta,
	)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512dq,avx512vl,fma")]
#[inline]
pub unsafe fn jma_avx512(data: &[f64], period: usize, phase: f64, power: u32, first_valid: usize, out: &mut [f64]) {
	debug_assert!(data.len() == out.len() && first_valid < data.len());

	let (pr, beta, alpha, alpha_sq, oma_sq, one_minus_alpha, one_minus_beta) = jma_consts(period, phase, power);

	let pr_v = _mm512_set1_pd(pr);
	let oma_sq_v = _mm512_set1_pd(oma_sq);
	let alpha_sq_v = _mm512_set1_pd(alpha_sq);
	let one_minus_alpha_v = _mm512_set1_pd(one_minus_alpha);
	let alpha_v = _mm512_set1_pd(alpha);
	let one_minus_beta_v = _mm512_set1_pd(one_minus_beta);
	let beta_v = _mm512_set1_pd(beta);

	let mut e0 = data[first_valid];
	let mut e1 = 0.0;
	let mut e2 = 0.0;
	let mut j_prev = e0;

	out[first_valid] = j_prev;

	let mut i = first_valid + 1;
	let n = data.len();

	// Unroll loop by 4 (as an example to leverage AVX512 register space for ILP)
	while i + 3 < n {
		for k in 0..4 {
			let price = *data.get_unchecked(i + k);

			e0 = one_minus_alpha.mul_add(price, alpha * e0);
			e1 = (price - e0).mul_add(one_minus_beta, beta * e1);
			let diff = e0 + pr * e1 - j_prev;
			e2 = diff.mul_add(oma_sq, alpha_sq * e2);
			j_prev += e2;

			*out.get_unchecked_mut(i + k) = j_prev;
		}
		i += 4;
	}

	// Scalar tail for remaining elements
	while i < n {
		let price = *data.get_unchecked(i);
		e0 = one_minus_alpha.mul_add(price, alpha * e0);
		e1 = (price - e0).mul_add(one_minus_beta, beta * e1);
		let diff = e0 + pr * e1 - j_prev;
		e2 = diff.mul_add(oma_sq, alpha_sq * e2);
		j_prev += e2;

		*out.get_unchecked_mut(i) = j_prev;
		i += 1;
	}
}

// ===== BATCH & STREAMING API =====

#[derive(Debug, Clone)]
pub struct JmaStream {
	period: usize,
	phase: f64,
	power: u32,
	alpha: f64,
	beta: f64,
	phase_ratio: f64,
	initialized: bool,
	e0: f64,
	e1: f64,
	e2: f64,
	jma_prev: f64,
}

impl JmaStream {
	pub fn try_new(params: JmaParams) -> Result<Self, JmaError> {
		let period = params.period.unwrap_or(7);
		if period == 0 {
			return Err(JmaError::InvalidPeriod { period, data_len: 0 });
		}
		let phase = params.phase.unwrap_or(50.0);
		if phase.is_nan() || phase.is_infinite() {
			return Err(JmaError::InvalidPhase { phase });
		}
		let power = params.power.unwrap_or(2);
		let phase_ratio = if phase < -100.0 {
			0.5
		} else if phase > 100.0 {
			2.5
		} else {
			(phase / 100.0) + 1.5
		};
		let beta = {
			let numerator = 0.45 * (period as f64 - 1.0);
			let denominator = numerator + 2.0;
			if denominator.abs() < f64::EPSILON {
				0.0
			} else {
				numerator / denominator
			}
		};
		let alpha = beta.powi(power as i32);
		Ok(Self {
			period,
			phase,
			power,
			alpha,
			beta,
			phase_ratio,
			initialized: false,
			e0: f64::NAN,
			e1: 0.0,
			e2: 0.0,
			jma_prev: f64::NAN,
		})
	}

	#[inline(always)]
	pub fn update(&mut self, value: f64) -> Option<f64> {
		if !self.initialized {
			if value.is_nan() {
				return None;
			}
			self.initialized = true;
			self.e0 = value;
			self.e1 = 0.0;
			self.e2 = 0.0;
			self.jma_prev = value;
			return Some(value);
		}
		let src = value;
		self.e0 = (1.0 - self.alpha) * src + self.alpha * self.e0;
		self.e1 = (src - self.e0) * (1.0 - self.beta) + self.beta * self.e1;
		let diff = self.e0 + self.phase_ratio * self.e1 - self.jma_prev;
		self.e2 = diff * (1.0 - self.alpha).powi(2) + self.alpha.powi(2) * self.e2;
		self.jma_prev = self.e2 + self.jma_prev;
		Some(self.jma_prev)
	}
}

#[derive(Clone, Debug)]
pub struct JmaBatchRange {
	pub period: (usize, usize, usize),
	pub phase: (f64, f64, f64),
	pub power: (u32, u32, u32),
}

impl Default for JmaBatchRange {
	fn default() -> Self {
		Self {
			period: (7, 240, 1),
			phase: (50.0, 50.0, 0.0),
			power: (2, 2, 0),
		}
	}
}

#[derive(Clone, Debug, Default)]
pub struct JmaBatchBuilder {
	range: JmaBatchRange,
	kernel: Kernel,
}

impl JmaBatchBuilder {
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
	#[inline]
	pub fn phase_range(mut self, start: f64, end: f64, step: f64) -> Self {
		self.range.phase = (start, end, step);
		self
	}
	#[inline]
	pub fn phase_static(mut self, x: f64) -> Self {
		self.range.phase = (x, x, 0.0);
		self
	}
	#[inline]
	pub fn power_range(mut self, start: u32, end: u32, step: u32) -> Self {
		self.range.power = (start, end, step);
		self
	}
	#[inline]
	pub fn power_static(mut self, p: u32) -> Self {
		self.range.power = (p, p, 0);
		self
	}
	pub fn apply_slice(self, data: &[f64]) -> Result<JmaBatchOutput, JmaError> {
		jma_batch_with_kernel(data, &self.range, self.kernel)
	}
	pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<JmaBatchOutput, JmaError> {
		JmaBatchBuilder::new().kernel(k).apply_slice(data)
	}
	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<JmaBatchOutput, JmaError> {
		let slice = source_type(c, src);
		self.apply_slice(slice)
	}
	pub fn with_default_candles(c: &Candles) -> Result<JmaBatchOutput, JmaError> {
		JmaBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
	}
}

pub fn jma_batch_with_kernel(data: &[f64], sweep: &JmaBatchRange, k: Kernel) -> Result<JmaBatchOutput, JmaError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(JmaError::InvalidPeriod { period: 0, data_len: 0 }),
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	jma_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct JmaBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<JmaParams>,
	pub rows: usize,
	pub cols: usize,
}

impl JmaBatchOutput {
	pub fn row_for_params(&self, p: &JmaParams) -> Option<usize> {
		self.combos.iter().position(|c| {
			c.period.unwrap_or(7) == p.period.unwrap_or(7)
				&& (c.phase.unwrap_or(50.0) - p.phase.unwrap_or(50.0)).abs() < 1e-12
				&& c.power.unwrap_or(2) == p.power.unwrap_or(2)
		})
	}
	pub fn values_for(&self, p: &JmaParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &JmaBatchRange) -> Vec<JmaParams> {
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
	fn axis_u32((start, end, step): (u32, u32, u32)) -> Vec<u32> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step as usize).collect()
	}
	let periods = axis_usize(r.period);
	let phases = axis_f64(r.phase);
	let powers = axis_u32(r.power);
	let mut out = Vec::with_capacity(periods.len() * phases.len() * powers.len());
	for &p in &periods {
		for &ph in &phases {
			for &po in &powers {
				out.push(JmaParams {
					period: Some(p),
					phase: Some(ph),
					power: Some(po),
				});
			}
		}
	}
	out
}

#[inline(always)]
pub fn jma_batch_slice(data: &[f64], sweep: &JmaBatchRange, kern: Kernel) -> Result<JmaBatchOutput, JmaError> {
	jma_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn jma_batch_par_slice(data: &[f64], sweep: &JmaBatchRange, kern: Kernel) -> Result<JmaBatchOutput, JmaError> {
	jma_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn jma_batch_inner(
	data: &[f64],
	sweep: &JmaBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<JmaBatchOutput, JmaError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(JmaError::InvalidPeriod { period: 0, data_len: 0 });
	}
	let first = data.iter().position(|x| !x.is_nan()).ok_or(JmaError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if data.len() - first < max_p {
		return Err(JmaError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}
	let rows = combos.len();
	let cols = data.len();
	let warm: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap()).collect();

	let mut raw = make_uninit_matrix(rows, cols);
	unsafe { init_matrix_prefixes(&mut raw, cols, &warm) };

	// ---------- 2. closure that fills ONE row ---------------------------
	let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
		let prm = &combos[row];
		let period = prm.period.unwrap();
		let phase = prm.phase.unwrap();
		let power = prm.power.unwrap();

		// Cast the uninit slice to &mut [f64] for the row writers
		let out_row = core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

		match kern {
			Kernel::Scalar => jma_row_scalar(data, first, period, phase, power, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => jma_row_avx2(data, first, period, phase, power, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => jma_row_avx512(data, first, period, phase, power, out_row),
			_ => unreachable!(),
		}
	};

	// ---------- 3. run every row ----------------------------------------
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

	// ---------- 4. transmute into fully-initialised Vec<f64> -------------
	let values: Vec<f64> = unsafe { std::mem::transmute(raw) };

	Ok(JmaBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
fn jma_batch_inner_into(
	data: &[f64],
	sweep: &JmaBatchRange,
	kern: Kernel,
	parallel: bool,
	out: &mut [f64],
) -> Result<(Vec<JmaParams>, usize, usize), JmaError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(JmaError::InvalidPeriod { period: 0, data_len: 0 });
	}
	let first = data.iter().position(|x| !x.is_nan()).ok_or(JmaError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if data.len() - first < max_p {
		return Err(JmaError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}
	let rows = combos.len();
	let cols = data.len();

	// Ensure output buffer is the correct size
	if out.len() != rows * cols {
		return Err(JmaError::InvalidOutputBuffer {
			expected: rows * cols,
			actual: out.len(),
		});
	}

	let warm: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap()).collect();

	// Cast output to MaybeUninit for initialization
	let out_uninit = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len()) };
	unsafe { init_matrix_prefixes(out_uninit, cols, &warm) };

	// ---------- closure that fills ONE row ---------------------------
	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let prm = &combos[row];
		let period = prm.period.unwrap();
		let phase = prm.phase.unwrap();
		let power = prm.power.unwrap();

		match kern {
			Kernel::Scalar => jma_row_scalar(data, first, period, phase, power, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => jma_row_avx2(data, first, period, phase, power, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => jma_row_avx512(data, first, period, phase, power, out_row),
			_ => unreachable!(),
		}
	};

	// ---------- run every row ----------------------------------------
	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			out.par_chunks_mut(cols)
				.enumerate()
				.for_each(|(row, slice)| do_row(row, slice));
		}
		#[cfg(target_arch = "wasm32")]
		{
			for (row, slice) in out.chunks_mut(cols).enumerate() {
				do_row(row, slice);
			}
		}
	} else {
		for (row, slice) in out.chunks_mut(cols).enumerate() {
			do_row(row, slice);
		}
	}

	Ok((combos, rows, cols))
}

#[inline(always)]
unsafe fn jma_row_scalar(data: &[f64], first: usize, period: usize, phase: f64, power: u32, out: &mut [f64]) {
	jma_scalar(data, period, phase, power, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn jma_row_avx2(data: &[f64], first: usize, period: usize, phase: f64, power: u32, out: &mut [f64]) {
	jma_avx2(data, period, phase, power, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn jma_row_avx512(data: &[f64], first: usize, period: usize, phase: f64, power: u32, out: &mut [f64]) {
	jma_avx512(data, period, phase, power, first, out);
}

#[inline(always)]
pub fn expand_grid_jma(r: &JmaBatchRange) -> Vec<JmaParams> {
	expand_grid(r)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_jma_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let default_params = JmaParams {
			period: None,
			phase: None,
			power: None,
		};
		let input = JmaInput::from_candles(&candles, "close", default_params);
		let output = jma_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());

		Ok(())
	}

	fn check_jma_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = JmaInput::from_candles(&candles, "close", JmaParams::default());
		let result = jma_with_kernel(&input, kernel)?;
		let expected_last_five = [
			59305.04794668568,
			59261.270455005455,
			59156.791263606865,
			59128.30656791065,
			58918.89223153998,
		];
		let start = result.values.len().saturating_sub(5);
		for (i, &val) in result.values[start..].iter().enumerate() {
			let diff = (val - expected_last_five[i]).abs();
			assert!(
				diff < 1e-6,
				"[{}] JMA {:?} mismatch at idx {}: got {}, expected {}",
				test_name,
				kernel,
				i,
				val,
				expected_last_five[i]
			);
		}
		Ok(())
	}

	fn check_jma_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = JmaInput::with_default_candles(&candles);
		match input.data {
			JmaData::Candles { source, .. } => assert_eq!(source, "close"),
			_ => panic!("Expected JmaData::Candles"),
		}
		let output = jma_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_jma_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let input_data = [10.0, 20.0, 30.0];
		let params = JmaParams {
			period: Some(0),
			phase: None,
			power: None,
		};
		let input = JmaInput::from_slice(&input_data, params);
		let res = jma_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] JMA should fail with zero period", test_name);
		Ok(())
	}

	fn check_jma_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data_small = [10.0, 20.0, 30.0];
		let params = JmaParams {
			period: Some(10),
			phase: None,
			power: None,
		};
		let input = JmaInput::from_slice(&data_small, params);
		let res = jma_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] JMA should fail with period exceeding length",
			test_name
		);
		Ok(())
	}

	fn check_jma_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let single_point = [42.0];
		let params = JmaParams {
			period: Some(7),
			phase: None,
			power: None,
		};
		let input = JmaInput::from_slice(&single_point, params);
		let res = jma_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] JMA should fail with insufficient data", test_name);
		Ok(())
	}

	fn check_jma_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let first_params = JmaParams {
			period: Some(7),
			phase: None,
			power: None,
		};
		let first_input = JmaInput::from_candles(&candles, "close", first_params);
		let first_result = jma_with_kernel(&first_input, kernel)?;
		let second_params = JmaParams {
			period: Some(7),
			phase: None,
			power: None,
		};
		let second_input = JmaInput::from_slice(&first_result.values, second_params);
		let second_result = jma_with_kernel(&second_input, kernel)?;
		assert_eq!(second_result.values.len(), first_result.values.len());
		Ok(())
	}

	fn check_jma_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = JmaInput::from_candles(
			&candles,
			"close",
			JmaParams {
				period: Some(7),
				phase: None,
				power: None,
			},
		);
		let res = jma_with_kernel(&input, kernel)?;
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

	fn check_jma_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let period = 7;
		let phase = 50.0;
		let power = 2;
		let input = JmaInput::from_candles(
			&candles,
			"close",
			JmaParams {
				period: Some(period),
				phase: Some(phase),
				power: Some(power),
			},
		);
		let batch_output = jma_with_kernel(&input, kernel)?.values;
		let mut stream = JmaStream::try_new(JmaParams {
			period: Some(period),
			phase: Some(phase),
			power: Some(power),
		})?;
		let mut stream_values = Vec::with_capacity(candles.close.len());
		for &price in &candles.close {
			match stream.update(price) {
				Some(jma_val) => stream_values.push(jma_val),
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
				diff < 1e-8,
				"[{}] JMA streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
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
	fn check_jma_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		// Test multiple parameter combinations to better catch uninitialized memory bugs
		let test_params = vec![
			// Default parameters
			JmaParams::default(),
			// Small periods with various phases and powers
			JmaParams {
				period: Some(3),
				phase: Some(0.0),
				power: Some(1),
			},
			JmaParams {
				period: Some(3),
				phase: Some(50.0),
				power: Some(2),
			},
			JmaParams {
				period: Some(3),
				phase: Some(100.0),
				power: Some(3),
			},
			// Medium periods
			JmaParams {
				period: Some(7),
				phase: Some(25.0),
				power: Some(1),
			},
			JmaParams {
				period: Some(7),
				phase: Some(50.0),
				power: Some(2),
			},
			JmaParams {
				period: Some(7),
				phase: Some(75.0),
				power: Some(3),
			},
			JmaParams {
				period: Some(10),
				phase: Some(0.0),
				power: Some(2),
			},
			JmaParams {
				period: Some(14),
				phase: Some(100.0),
				power: Some(2),
			},
			// Large periods
			JmaParams {
				period: Some(20),
				phase: Some(50.0),
				power: Some(1),
			},
			JmaParams {
				period: Some(30),
				phase: Some(50.0),
				power: Some(2),
			},
			JmaParams {
				period: Some(50),
				phase: Some(50.0),
				power: Some(3),
			},
			// Edge cases
			JmaParams {
				period: Some(1),
				phase: Some(0.0),
				power: Some(1),
			},
			JmaParams {
				period: Some(100),
				phase: Some(100.0),
				power: Some(5),
			},
			// Extreme phases
			JmaParams {
				period: Some(10),
				phase: Some(-100.0),
				power: Some(2),
			},
			JmaParams {
				period: Some(10),
				phase: Some(200.0),
				power: Some(2),
			},
		];

		for (param_idx, params) in test_params.iter().enumerate() {
			let input = JmaInput::from_candles(&candles, "close", params.clone());
			let output = jma_with_kernel(&input, kernel)?;

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
                        with params: period={}, phase={}, power={}",
						test_name,
						val,
						bits,
						i,
						params.period.unwrap_or(7),
						params.phase.unwrap_or(50.0),
						params.power.unwrap_or(2)
					);
				}

				// Check for init_matrix_prefixes poison (0x22222222_22222222)
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
                        with params: period={}, phase={}, power={}",
						test_name,
						val,
						bits,
						i,
						params.period.unwrap_or(7),
						params.phase.unwrap_or(50.0),
						params.power.unwrap_or(2)
					);
				}

				// Check for make_uninit_matrix poison (0x33333333_33333333)
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
                        with params: period={}, phase={}, power={}",
						test_name,
						val,
						bits,
						i,
						params.period.unwrap_or(7),
						params.phase.unwrap_or(50.0),
						params.power.unwrap_or(2)
					);
				}
			}
		}

		Ok(())
	}

	// Release mode stub - does nothing
	#[cfg(not(debug_assertions))]
	fn check_jma_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(())
	}

	macro_rules! generate_all_jma_tests {
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

	generate_all_jma_tests!(
		check_jma_partial_params,
		check_jma_accuracy,
		check_jma_default_candles,
		check_jma_zero_period,
		check_jma_period_exceeds_length,
		check_jma_very_small_dataset,
		check_jma_reinput,
		check_jma_nan_handling,
		check_jma_streaming,
		check_jma_no_poison
	);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let output = JmaBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;

		let def = JmaParams::default();
		let row = output.values_for(&def).expect("default row missing");

		assert_eq!(row.len(), c.close.len());

		let expected = [
			59305.04794668568,
			59261.270455005455,
			59156.791263606865,
			59128.30656791065,
			58918.89223153998,
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

		// Test multiple batch configurations to better catch uninitialized memory bugs
		let batch_configs = vec![
			// Default test case
			(5, 20, 5, 0.0, 100.0, 50.0, 1, 3, 1),
			// Small periods with extreme phases
			(1, 10, 3, -100.0, 200.0, 100.0, 1, 5, 2),
			// Large periods with small phase increments
			(50, 100, 25, -50.0, 50.0, 25.0, 2, 4, 1),
			// Single period, varying phase and power
			(14, 14, 1, -100.0, 200.0, 50.0, 1, 5, 1),
			// Edge cases with minimum values
			(1, 1, 1, 0.0, 0.0, 1.0, 1, 1, 1),
			// Testing with power variations
			(10, 30, 10, 50.0, 50.0, 1.0, 1, 5, 2),
			// Large batch with many combinations
			(5, 50, 5, -50.0, 150.0, 25.0, 1, 3, 1),
			// Testing negative to positive phase transitions
			(7, 21, 7, -100.0, 100.0, 40.0, 2, 2, 1),
			// Maximum practical values
			(80, 100, 20, 100.0, 200.0, 50.0, 4, 5, 1),
		];

		for (idx, config) in batch_configs.iter().enumerate() {
			let output = JmaBatchBuilder::new()
				.kernel(kernel)
				.period_range(config.0, config.1, config.2)
				.phase_range(config.3, config.4, config.5)
				.power_range(config.6, config.7, config.8)
				.apply_candles(&c, "close")?;

			// Check every value in the entire batch matrix for poison patterns
			for (val_idx, &val) in output.values.iter().enumerate() {
				// Skip NaN values as they're expected in warmup periods
				if val.is_nan() {
					continue;
				}

				let bits = val.to_bits();
				let row = val_idx / output.cols;
				let col = val_idx % output.cols;
				let combo = &output.combos[row];

				// Check for alloc_with_nan_prefix poison (0x11111111_11111111)
				if bits == 0x11111111_11111111 {
					panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {}) \
                        in batch config #{}: period_range=({},{},{}), phase_range=({},{},{}), power_range=({},{},{}) \
                        combo params: period={}, phase={}, power={}",
                        test, val, bits, row, col, val_idx, idx,
                        config.0, config.1, config.2, config.3, config.4, config.5, config.6, config.7, config.8,
                        combo.period.unwrap_or(7), combo.phase.unwrap_or(50.0), combo.power.unwrap_or(2)
                    );
				}

				// Check for init_matrix_prefixes poison (0x22222222_22222222)
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {}) \
                        in batch config #{}: period_range=({},{},{}), phase_range=({},{},{}), power_range=({},{},{}) \
                        combo params: period={}, phase={}, power={}",
						test,
						val,
						bits,
						row,
						col,
						val_idx,
						idx,
						config.0,
						config.1,
						config.2,
						config.3,
						config.4,
						config.5,
						config.6,
						config.7,
						config.8,
						combo.period.unwrap_or(7),
						combo.phase.unwrap_or(50.0),
						combo.power.unwrap_or(2)
					);
				}

				// Check for make_uninit_matrix poison (0x33333333_33333333)
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {}) \
                        in batch config #{}: period_range=({},{},{}), phase_range=({},{},{}), power_range=({},{},{}) \
                        combo params: period={}, phase={}, power={}",
						test,
						val,
						bits,
						row,
						col,
						val_idx,
						idx,
						config.0,
						config.1,
						config.2,
						config.3,
						config.4,
						config.5,
						config.6,
						config.7,
						config.8,
						combo.period.unwrap_or(7),
						combo.phase.unwrap_or(50.0),
						combo.power.unwrap_or(2)
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

// Python bindings
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

#[cfg(feature = "python")]
#[pyfunction(name = "jma")]
#[pyo3(signature = (data, period, phase=50.0, power=2, kernel=None))]
pub fn jma_py<'py>(
	py: Python<'py>,
	data: PyReadonlyArray1<'py, f64>,
	period: usize,
	phase: f64,
	power: u32,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, false)?;

	let params = JmaParams {
		period: Some(period),
		phase: Some(phase),
		power: Some(power),
	};
	let jma_in = JmaInput::from_slice(slice_in, params);

	// Get Vec<f64> from Rust function - zero-copy pattern
	let result_vec: Vec<f64> = py
		.allow_threads(|| jma_with_kernel(&jma_in, kern).map(|o| o.values))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	// Zero-copy transfer to NumPy
	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction(name = "jma_batch")]
#[pyo3(signature = (data, period_range, phase_range=(50.0, 50.0, 0.0), power_range=(2, 2, 0), kernel=None))]
pub fn jma_batch_py<'py>(
	py: Python<'py>,
	data: PyReadonlyArray1<'py, f64>,
	period_range: (usize, usize, usize),
	phase_range: (f64, f64, f64),
	power_range: (u32, u32, u32),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, true)?;

	let sweep = JmaBatchRange {
		period: period_range,
		phase: phase_range,
		power: power_range,
	};

	// Expand grid to get all combinations
	let combos = expand_grid(&sweep);
	if combos.is_empty() {
		return Err(PyValueError::new_err("Invalid parameter ranges"));
	}

	let rows = combos.len();
	let cols = slice_in.len();

	// Pre-allocate output array (OK for batch operations)
	let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let slice_out = unsafe { out_arr.as_slice_mut()? };

	// Compute without GIL
	let (combos_result, _, _) = py
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

			jma_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
		})
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	// Build result dictionary with zero-copy parameter arrays
	let dict = PyDict::new(py);
	dict.set_item("values", out_arr.reshape((rows, cols))?)?;

	// Zero-copy transfer for parameter arrays
	dict.set_item(
		"periods",
		combos_result
			.iter()
			.map(|p| p.period.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;
	dict.set_item(
		"phases",
		combos_result
			.iter()
			.map(|p| p.phase.unwrap())
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;
	dict.set_item(
		"powers",
		combos_result
			.iter()
			.map(|p| p.power.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;

	Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "JmaStream")]
pub struct JmaStreamPy {
	inner: JmaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl JmaStreamPy {
	#[new]
	#[pyo3(signature = (period, phase=50.0, power=2))]
	fn new(period: usize, phase: f64, power: u32) -> PyResult<Self> {
		let params = JmaParams {
			period: Some(period),
			phase: Some(phase),
			power: Some(power),
		};

		let stream =
			JmaStream::try_new(params).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

		Ok(Self { inner: stream })
	}

	fn update(&mut self, value: f64) -> Option<f64> {
		self.inner.update(value)
	}
}

// ================== Zero-Copy Helper for WASM ==================
/// Write JMA values directly to output slice - no allocations
/// This helper is the core optimization that enables zero-allocation writes.
/// The output slice must be the same length as the input data.
#[inline]
pub fn jma_into_slice(dst: &mut [f64], input: &JmaInput, kern: Kernel) -> Result<(), JmaError> {
	let data: &[f64] = match &input.data {
		JmaData::Candles { candles, source } => source_type(candles, source),
		JmaData::Slice(sl) => sl,
	};

	// Verify output buffer size matches input
	if dst.len() != data.len() {
		return Err(JmaError::InvalidOutputBuffer {
			expected: data.len(),
			actual: dst.len(),
		});
	}

	// Use existing jma_with_kernel_into which already writes directly to dst
	jma_with_kernel_into(input, kern, dst)
}

// WASM bindings
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use serde::{Deserialize, Serialize};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn jma_js(data: &[f64], period: usize, phase: f64, power: u32) -> Result<Vec<f64>, JsValue> {
	let params = JmaParams {
		period: Some(period),
		phase: Some(phase),
		power: Some(power),
	};
	let input = JmaInput::from_slice(data, params);

	// Allocate output buffer once
	let mut output = vec![0.0; data.len()];

	// Compute directly into output buffer
	jma_into_slice(&mut output, &input, Kernel::Auto).map_err(|e| JsValue::from_str(&e.to_string()))?;

	Ok(output)
}

// ================== Zero-Copy WASM Functions ==================

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn jma_alloc(len: usize) -> *mut f64 {
	// Allocate memory for input/output buffer
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec); // Prevent deallocation
	ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn jma_free(ptr: *mut f64, len: usize) {
	// Free allocated memory
	if !ptr.is_null() {
		unsafe {
			let _ = Vec::from_raw_parts(ptr, len, len);
		}
	}
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn jma_into(
	in_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period: usize,
	phase: f64,
	power: u32,
) -> Result<(), JsValue> {
	// Check for null pointers
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("null pointer passed to jma_into"));
	}

	unsafe {
		// Create slice from pointer
		let data = std::slice::from_raw_parts(in_ptr, len);

		// Validate inputs
		if period == 0 || period > len {
			return Err(JsValue::from_str("Invalid period"));
		}

		// Calculate JMA
		let params = JmaParams {
			period: Some(period),
			phase: Some(phase),
			power: Some(power),
		};
		let input = JmaInput::from_slice(data, params);

		// Handle aliasing (in_ptr == out_ptr)
		if in_ptr == out_ptr {
			// Use temporary buffer to avoid corruption during sliding window computation
			let mut temp = vec![0.0; len];
			jma_into_slice(&mut temp, &input, Kernel::Auto).map_err(|e| JsValue::from_str(&e.to_string()))?;

			// Copy results back to output
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			out.copy_from_slice(&temp);
		} else {
			// No aliasing, compute directly into output
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			jma_into_slice(out, &input, Kernel::Auto).map_err(|e| JsValue::from_str(&e.to_string()))?;
		}

		Ok(())
	}
}

// ================== Batch Processing Structures ==================

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct JmaBatchConfig {
	pub period_range: (usize, usize, usize),
	pub phase_range: (f64, f64, f64),
	pub power_range: (u32, u32, u32),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct JmaBatchJsOutput {
	pub values: Vec<f64>,
	pub combos: Vec<JmaParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = jma_batch)]
pub fn jma_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	let config: JmaBatchConfig =
		serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

	let sweep = JmaBatchRange {
		period: config.period_range,
		phase: config.phase_range,
		power: config.power_range,
	};

	let output = jma_batch_inner(data, &sweep, Kernel::Scalar, false).map_err(|e| JsValue::from_str(&e.to_string()))?;

	let js_output = JmaBatchJsOutput {
		values: output.values,
		combos: output.combos,
		rows: output.rows,
		cols: output.cols,
	};

	serde_wasm_bindgen::to_value(&js_output).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
#[deprecated(since = "1.0.0", note = "Use jma_batch instead")]
pub fn jma_batch_js(
	data: &[f64],
	period_start: usize,
	period_end: usize,
	period_step: usize,
	phase_start: f64,
	phase_end: f64,
	phase_step: f64,
	power_start: u32,
	power_end: u32,
	power_step: u32,
) -> Result<Vec<f64>, JsValue> {
	let sweep = JmaBatchRange {
		period: (period_start, period_end, period_step),
		phase: (phase_start, phase_end, phase_step),
		power: (power_start, power_end, power_step),
	};

	// Use the existing batch function with parallel=false for WASM
	jma_batch_inner(data, &sweep, Kernel::Scalar, false)
		.map(|output| output.values)
		.map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn jma_batch_metadata_js(
	period_start: usize,
	period_end: usize,
	period_step: usize,
	phase_start: f64,
	phase_end: f64,
	phase_step: f64,
	power_start: u32,
	power_end: u32,
	power_step: u32,
) -> Vec<f64> {
	let mut metadata = Vec::new();

	let mut current_period = period_start;
	while current_period <= period_end {
		let mut current_phase = phase_start;
		while current_phase <= phase_end || (phase_step == 0.0 && current_phase == phase_start) {
			let mut current_power = power_start;
			while current_power <= power_end || (power_step == 0 && current_power == power_start) {
				metadata.push(current_period as f64);
				metadata.push(current_phase);
				metadata.push(current_power as f64);

				if power_step == 0 {
					break;
				}
				current_power += power_step;
			}
			if phase_step == 0.0 {
				break;
			}
			current_phase += phase_step;
		}
		if period_step == 0 {
			break;
		}
		current_period += period_step;
	}

	metadata
}

// ================== Optimized Batch Processing ==================

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn jma_batch_into(
	in_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period_start: usize,
	period_end: usize,
	period_step: usize,
	phase_start: f64,
	phase_end: f64,
	phase_step: f64,
	power_start: u32,
	power_end: u32,
	power_step: u32,
) -> Result<usize, JsValue> {
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("null pointer passed to jma_batch_into"));
	}

	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);

		let sweep = JmaBatchRange {
			period: (period_start, period_end, period_step),
			phase: (phase_start, phase_end, phase_step),
			power: (power_start, power_end, power_step),
		};

		let combos = expand_grid_jma(&sweep);
		let rows = combos.len();
		let cols = len;

		let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);

		// Use optimized batch processing
		jma_batch_inner_into(data, &sweep, Kernel::Scalar, false, out)
			.map_err(|e| JsValue::from_str(&e.to_string()))?;

		Ok(rows)
	}
}
