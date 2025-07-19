//! # Gaussian Filter
//!
//! A parametric smoothing technique that approximates a Gaussian response using
//! a cascade of discrete poles. Its parameters (`period`, `poles`) control the
//! filter's length and the number of cascaded stages that shape the overall
//! filter response.
//!
//! ## Parameters
//! - **period**: Window size (number of data points).
//! - **poles**: The number of poles (1..4) to use for the filter.
//!
//! ## Errors
//! - **NoData**: gaussian: No data provided.
//! - **InvalidPoles**: gaussian: `poles` is out of range (expected 1..4).
//! - **InvalidPeriod**: gaussian: `period` is zero or exceeds the data length.
//! - **PeriodLongerThanData**: gaussian: The `period` is longer than the data length.
//! - **AllValuesNaN**: gaussian: All input data values are `NaN`.
//! - **NotEnoughValidData**: gaussian: Not enough valid data for the requested `period`.
//!
//! ## Returns
//! - **`Ok(GaussianOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(GaussianError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
	alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use std::arch::x86_64::*;
use std::f64::consts::PI;
use std::mem::MaybeUninit;
use thiserror::Error;

const LANES_AVX512: usize = 8;
const LANES_AVX2: usize = 4;

impl<'a> AsRef<[f64]> for GaussianInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			GaussianData::Slice(slice) => slice,
			GaussianData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

#[derive(Debug, Clone)]
pub enum GaussianData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct GaussianOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct GaussianParams {
	pub period: Option<usize>,
	pub poles: Option<usize>,
}

impl Default for GaussianParams {
	fn default() -> Self {
		Self {
			period: Some(14),
			poles: Some(4),
		}
	}
}

#[derive(Debug, Clone)]
pub struct GaussianInput<'a> {
	pub data: GaussianData<'a>,
	pub params: GaussianParams,
}

impl<'a> GaussianInput<'a> {
	pub fn from_candles(c: &'a Candles, s: &'a str, p: GaussianParams) -> Self {
		Self {
			data: GaussianData::Candles { candles: c, source: s },
			params: p,
		}
	}
	pub fn from_slice(sl: &'a [f64], p: GaussianParams) -> Self {
		Self {
			data: GaussianData::Slice(sl),
			params: p,
		}
	}
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", GaussianParams::default())
	}
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(14)
	}
	pub fn get_poles(&self) -> usize {
		self.params.poles.unwrap_or(4)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct GaussianBuilder {
	period: Option<usize>,
	poles: Option<usize>,
	kernel: Kernel,
}

impl Default for GaussianBuilder {
	fn default() -> Self {
		Self {
			period: None,
			poles: None,
			kernel: Kernel::Auto,
		}
	}
}

impl GaussianBuilder {
	pub fn new() -> Self {
		Self::default()
	}
	pub fn period(mut self, n: usize) -> Self {
		self.period = Some(n);
		self
	}
	pub fn poles(mut self, k: usize) -> Self {
		self.poles = Some(k);
		self
	}
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	pub fn apply(self, c: &Candles) -> Result<GaussianOutput, GaussianError> {
		let p = GaussianParams {
			period: self.period,
			poles: self.poles,
		};
		let i = GaussianInput::from_candles(c, "close", p);
		gaussian_with_kernel(&i, self.kernel)
	}
	pub fn apply_slice(self, d: &[f64]) -> Result<GaussianOutput, GaussianError> {
		let p = GaussianParams {
			period: self.period,
			poles: self.poles,
		};
		let i = GaussianInput::from_slice(d, p);
		gaussian_with_kernel(&i, self.kernel)
	}
	pub fn into_stream(self) -> Result<GaussianStream, GaussianError> {
		let p = GaussianParams {
			period: self.period,
			poles: self.poles,
		};
		GaussianStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum GaussianError {
	#[error("gaussian: No data provided to Gaussian filter.")]
	NoData,
	#[error("gaussian: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("gaussian: Invalid number of poles: expected 1..4, got {poles}")]
	InvalidPoles { poles: usize },
	#[error("Gaussian filter period is longer than the data. period={period}, data_len={data_len}")]
	PeriodLongerThanData { period: usize, data_len: usize },
	#[error("gaussian: All values are NaN.")]
	AllValuesNaN,
	#[error("gaussian: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn gaussian(input: &GaussianInput) -> Result<GaussianOutput, GaussianError> {
	gaussian_with_kernel(input, Kernel::Auto)
}

// ---------------------------------------------------------------------------
// CPU-feature probe helpers
// ---------------------------------------------------------------------------
#[inline(always)]
fn has_avx2() -> bool {
	// x86/x86_64: cheap CPUID test provided by std.
	#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
	{
		std::is_x86_feature_detected!("avx2")
	}
	// All other architectures: no AVX2
	#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
	{
		false
	}
}

// ---------------------------------------------------------------------------
// Public scalar façade
// ---------------------------------------------------------------------------
#[inline(always)]
pub fn gaussian_scalar(data: &[f64], period: usize, poles: usize, out: &mut [f64]) {
	debug_assert_eq!(data.len(), out.len());

	// Choose at run time.  The FMA path is guarded so it is **never** entered
	// on hardware that would raise an illegal-instruction fault.
	if has_avx2() {
		// Fast, register-only implementation that relies on fused multiply-add.
		unsafe { gaussian_scalar_fma(data, period, poles, out) }
	} else {
		// Always-works baseline (no FMA required, still allocation-free).
		gaussian_scalar_fallback(data, period, poles, out)
	}
}

// ---------------------------------------------------------------------------
// Fast path – uses mul_add (FMA on AVX2/FMA3 or AArch64, plain MUL+ADD
// otherwise; the #[target_feature] attribute guarantees legality on x86).
// ---------------------------------------------------------------------------
#[cfg_attr(any(target_arch = "x86", target_arch = "x86_64"), target_feature(enable = "fma"))] // no-op on non-x86 targets
unsafe fn gaussian_scalar_fma(data: &[f64], period: usize, poles: usize, out: &mut [f64]) {
	use core::f64::consts::PI;

	let beta = {
		let num = 1.0 - (2.0 * PI / period as f64).cos();
		let den = (2.0f64).powf(1.0 / poles as f64) - 1.0;
		num / den
	};
	let alpha = {
		let tmp = beta * beta + 2.0 * beta;
		-beta + tmp.sqrt()
	};

	match poles {
		1 => gaussian_poles1_fma(data, alpha, out),
		2 => gaussian_poles2_fma(data, alpha, out),
		3 => gaussian_poles3_fma(data, alpha, out),
		4 => gaussian_poles4_fma(data, alpha, out),
		_ => core::hint::unreachable_unchecked(),
	}
}

// ---------------------------------------------------------------------------
// Fallback path – identical math, no mul_add requirement.
// (This is the allocation-free version suggested earlier; keep the original
// heap-allocating variant if you prefer.)
// ---------------------------------------------------------------------------
fn gaussian_scalar_fallback(data: &[f64], period: usize, poles: usize, out: &mut [f64]) {
	use core::f64::consts::PI;

	let beta = {
		let num = 1.0 - (2.0 * PI / period as f64).cos();
		let den = (2.0f64).powf(1.0 / poles as f64) - 1.0;
		num / den
	};
	let alpha = {
		let tmp = beta * beta + 2.0 * beta;
		-beta + tmp.sqrt()
	};

	// ─── same non-allocating kernels;  mul_add becomes MUL+ADD if FMA absent ──
	unsafe {
		match poles {
			1 => gaussian_poles1_fma(data, alpha, out),
			2 => gaussian_poles2_fma(data, alpha, out),
			3 => gaussian_poles3_fma(data, alpha, out),
			4 => gaussian_poles4_fma(data, alpha, out),
			_ => core::hint::unreachable_unchecked(),
		}
	}
}

// ---------------------------------------------------------------------------
// with-kernel dispatcher – now totally safe on every architecture.
// ---------------------------------------------------------------------------
pub fn gaussian_with_kernel(input: &GaussianInput, kernel: Kernel) -> Result<GaussianOutput, GaussianError> {
	let data: &[f64] = match &input.data {
		GaussianData::Candles { candles, source } => source_type(candles, source),
		GaussianData::Slice(sl) => sl,
	};

	let len = data.len();
	let period = input.get_period();
	let poles = input.get_poles();

	if len == 0 {
		return Err(GaussianError::NoData);
	}
	if period == 0 || period > len {
		return Err(GaussianError::InvalidPeriod { period, data_len: len });
	}
	if !(1..=4).contains(&poles) {
		return Err(GaussianError::InvalidPoles { poles });
	}
	if len < period {
		return Err(GaussianError::PeriodLongerThanData { period, data_len: len });
	}

	let first_valid = data
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(GaussianError::AllValuesNaN)?;
	if len - first_valid < period {
		return Err(GaussianError::NotEnoughValidData {
			needed: period,
			valid: len - first_valid,
		});
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};
	let warm = first_valid + period;
	let mut out = alloc_with_nan_prefix(len, warm);

	unsafe {
		match chosen {
			// scalar paths – now self-dispatch inside `gaussian_scalar`
			Kernel::Scalar | Kernel::ScalarBatch => gaussian_scalar(data, period, poles, &mut out),

			// avx paths – compiled only when nightly-avx is available
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => gaussian_avx2(data, period, poles, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => gaussian_avx512(data, period, poles, &mut out),
			#[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
			Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
				gaussian_scalar(data, period, poles, &mut out)
			}

			Kernel::Auto => unreachable!(),
		}
	}

	Ok(GaussianOutput { values: out })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
#[allow(unused_variables)]
pub unsafe fn gaussian_avx2(data: &[f64], period: usize, poles: usize, out: &mut [f64]) {
	gaussian_scalar(data, period, poles, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
#[allow(unused_variables)]
pub unsafe fn gaussian_avx512(data: &[f64], period: usize, poles: usize, out: &mut [f64]) {
	gaussian_scalar(data, period, poles, out);
}

#[inline(always)]
unsafe fn gaussian_poles1_fma(inp: &[f64], alpha: f64, out: &mut [f64]) {
	let c0 = alpha;
	let c1 = 1.0 - alpha;

	let mut prev = 0.0;
	for i in 0..inp.len() {
		let x = *inp.get_unchecked(i);
		prev = c1.mul_add(prev, c0 * x); // FMA: prev = c0*x + c1*prev
		*out.get_unchecked_mut(i) = prev;
	}
}

#[inline(always)]
unsafe fn gaussian_poles2_fma(inp: &[f64], alpha: f64, out: &mut [f64]) {
	let a2 = alpha * alpha;
	let one = 1.0 - alpha;
	let c0 = a2;
	let c1 = 2.0 * one;
	let c2 = -(one * one);

	let mut prev1 = 0.0; // y[n‑1]
	let mut prev0 = 0.0; // y[n‑2]

	for i in 0..inp.len() {
		let x = *inp.get_unchecked(i);
		let y = c2.mul_add(prev0, c1.mul_add(prev1, c0 * x)); // 2 × FMA chain
		prev0 = prev1;
		prev1 = y;
		*out.get_unchecked_mut(i) = y;
	}
}

#[inline(always)]
unsafe fn gaussian_poles3_fma(inp: &[f64], alpha: f64, out: &mut [f64]) {
	let a3 = alpha * alpha * alpha;
	let one = 1.0 - alpha;
	let one2 = one * one;

	let c0 = a3;
	let c1 = 3.0 * one;
	let c2 = -3.0 * one2;
	let c3 = one2 * one;

	let mut p2 = 0.0; // y[n‑1]
	let mut p1 = 0.0; // y[n‑2]
	let mut p0 = 0.0; // y[n‑3]

	for i in 0..inp.len() {
		let x = *inp.get_unchecked(i);
		let y = c3.mul_add(
			p0,
			c2.mul_add(
				p1,
				c1.mul_add(p2, c0 * x), // 3‑deep FMA chain
			),
		);
		p0 = p1;
		p1 = p2;
		p2 = y;
		*out.get_unchecked_mut(i) = y;
	}
}

#[inline(always)]
unsafe fn gaussian_poles4_fma(inp: &[f64], alpha: f64, out: &mut [f64]) {
	let a4 = alpha * alpha * alpha * alpha;
	let one = 1.0 - alpha;
	let one2 = one * one;
	let one3 = one2 * one;

	let c0 = a4;
	let c1 = 4.0 * one;
	let c2 = -6.0 * one2;
	let c3 = 4.0 * one3;
	let c4 = -(one3 * one);

	let mut p3 = 0.0; // y[n‑1]
	let mut p2 = 0.0; // y[n‑2]
	let mut p1 = 0.0; // y[n‑3]
	let mut p0 = 0.0; // y[n‑4]

	for i in 0..inp.len() {
		let x = *inp.get_unchecked(i);
		let y = c4.mul_add(
			p0,
			c3.mul_add(
				p1,
				c2.mul_add(
					p2,
					c1.mul_add(p3, c0 * x), // 4‑deep FMA chain
				),
			),
		);
		p0 = p1;
		p1 = p2;
		p2 = p3;
		p3 = y;
		*out.get_unchecked_mut(i) = y;
	}
}

#[inline(always)]
fn gaussian_poles1(data: &[f64], n: usize, alpha: f64) -> Vec<f64> {
	let c0 = alpha;
	let c1 = 1.0 - alpha;
	let mut fil = vec![0.0; 1 + n];
	for i in 0..n {
		fil[i + 1] = c0 * data[i] + c1 * fil[i];
	}
	fil[1..1 + n].to_vec()
}
#[inline(always)]
fn gaussian_poles2(data: &[f64], n: usize, alpha: f64) -> Vec<f64> {
	let a2 = alpha * alpha;
	let one_a = 1.0 - alpha;
	let c0 = a2;
	let c1 = 2.0 * one_a;
	let c2 = -(one_a * one_a);
	let mut fil = vec![0.0; 2 + n];
	for i in 0..n {
		fil[i + 2] = c0 * data[i] + c1 * fil[i + 1] + c2 * fil[i];
	}
	fil[2..2 + n].to_vec()
}
#[inline(always)]
fn gaussian_poles3(data: &[f64], n: usize, alpha: f64) -> Vec<f64> {
	let a3 = alpha * alpha * alpha;
	let one_a = 1.0 - alpha;
	let one_a2 = one_a * one_a;
	let c0 = a3;
	let c1 = 3.0 * one_a;
	let c2 = -3.0 * one_a2;
	let c3 = one_a2 * one_a;
	let mut fil = vec![0.0; 3 + n];
	for i in 0..n {
		fil[i + 3] = c0 * data[i] + c1 * fil[i + 2] + c2 * fil[i + 1] + c3 * fil[i];
	}
	fil[3..3 + n].to_vec()
}
#[inline(always)]
fn gaussian_poles4(data: &[f64], n: usize, alpha: f64) -> Vec<f64> {
	let a4 = alpha * alpha * alpha * alpha;
	let one_a = 1.0 - alpha;
	let one_a2 = one_a * one_a;
	let one_a3 = one_a2 * one_a;
	let c0 = a4;
	let c1 = 4.0 * one_a;
	let c2 = -6.0 * one_a2;
	let c3 = 4.0 * one_a3;
	let c4 = -(one_a3 * one_a);
	let mut fil = vec![0.0; 4 + n];
	for i in 0..n {
		fil[i + 4] = c0 * data[i] + c1 * fil[i + 3] + c2 * fil[i + 2] + c3 * fil[i + 1] + c4 * fil[i];
	}
	fil[4..4 + n].to_vec()
}

#[derive(Debug, Clone)]
pub struct GaussianStream {
	period: usize,
	poles: usize,
	alpha: f64,
	state: Vec<f64>,
	idx: usize,
	init: bool,
}

impl GaussianStream {
	pub fn try_new(params: GaussianParams) -> Result<Self, GaussianError> {
		let period = params.period.unwrap_or(14);
		let poles = params.poles.unwrap_or(4);
		if period == 0 {
			return Err(GaussianError::InvalidPeriod { period, data_len: 0 });
		}
		if !(1..=4).contains(&poles) {
			return Err(GaussianError::InvalidPoles { poles });
		}
		let beta = {
			let numerator = 1.0 - (2.0 * PI / period as f64).cos();
			let denominator = (2.0_f64).powf(1.0 / poles as f64) - 1.0;
			numerator / denominator
		};
		let alpha = {
			let tmp = beta * beta + 2.0 * beta;
			-beta + tmp.sqrt()
		};
		Ok(Self {
			period,
			poles,
			alpha,
			state: vec![0.0; poles],
			idx: 0,
			init: false,
		})
	}
	pub fn update(&mut self, value: f64) -> f64 {
		let p = self.poles;
		let a = self.alpha;
		let mut prev = value;
		for s in 0..p {
			let last = self.state[s];
			let next = a * prev + (1.0 - a) * last;
			self.state[s] = next;
			prev = next;
		}
		prev
	}
}

// Batch

#[derive(Clone, Debug)]
pub struct GaussianBatchRange {
	pub period: (usize, usize, usize),
	pub poles: (usize, usize, usize),
}

impl Default for GaussianBatchRange {
	fn default() -> Self {
		Self {
			period: (14, 120, 1),
			poles: (1, 4, 1),
		}
	}
}

#[derive(Clone, Debug, Default)]
pub struct GaussianBatchBuilder {
	range: GaussianBatchRange,
	kernel: Kernel,
}

impl GaussianBatchBuilder {
	pub fn new() -> Self {
		Self::default()
	}
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.period = (start, end, step);
		self
	}
	pub fn period_static(mut self, p: usize) -> Self {
		self.range.period = (p, p, 0);
		self
	}
	pub fn poles_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.poles = (start, end, step);
		self
	}
	pub fn poles_static(mut self, p: usize) -> Self {
		self.range.poles = (p, p, 0);
		self
	}
	pub fn apply_slice(self, data: &[f64]) -> Result<GaussianBatchOutput, GaussianError> {
		gaussian_batch_with_kernel(data, &self.range, self.kernel)
	}
	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<GaussianBatchOutput, GaussianError> {
		let slice = source_type(c, src);
		self.apply_slice(slice)
	}
	pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<GaussianBatchOutput, GaussianError> {
		GaussianBatchBuilder::new().kernel(k).apply_slice(data)
	}
	pub fn with_default_candles(c: &Candles) -> Result<GaussianBatchOutput, GaussianError> {
		GaussianBatchBuilder::new()
			.kernel(Kernel::Auto)
			.apply_candles(c, "close")
	}
}

#[derive(Clone, Debug)]
pub struct GaussianBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<GaussianParams>,
	pub rows: usize,
	pub cols: usize,
}

impl GaussianBatchOutput {
	pub fn row_for_params(&self, p: &GaussianParams) -> Option<usize> {
		self.combos.iter().position(|c| {
			c.period.unwrap_or(14) == p.period.unwrap_or(14) && c.poles.unwrap_or(4) == p.poles.unwrap_or(4)
		})
	}
	pub fn values_for(&self, p: &GaussianParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &GaussianBatchRange) -> Vec<GaussianParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let periods = axis_usize(r.period);
	let poles = axis_usize(r.poles);
	let mut out = Vec::with_capacity(periods.len() * poles.len());
	for &p in &periods {
		for &k in &poles {
			out.push(GaussianParams {
				period: Some(p),
				poles: Some(k),
			});
		}
	}
	out
}

pub fn gaussian_batch_with_kernel(
	data: &[f64],
	sweep: &GaussianBatchRange,
	k: Kernel,
) -> Result<GaussianBatchOutput, GaussianError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(GaussianError::NoData),
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	gaussian_batch_par_slice(data, sweep, simd)
}

#[inline(always)]
pub fn gaussian_batch_slice(
	data: &[f64],
	sweep: &GaussianBatchRange,
	kern: Kernel,
) -> Result<GaussianBatchOutput, GaussianError> {
	gaussian_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn gaussian_batch_par_slice(
	data: &[f64],
	sweep: &GaussianBatchRange,
	kern: Kernel,
) -> Result<GaussianBatchOutput, GaussianError> {
	gaussian_batch_inner(data, sweep, kern, true)
}

/// Computes Gaussian directly into a provided output slice, avoiding allocation.
/// The output slice must be the same length as the input data.
#[inline]
pub fn gaussian_into_slice(dst: &mut [f64], input: &GaussianInput, kern: Kernel) -> Result<(), GaussianError> {
	let data = input.as_ref();

	// Verify output buffer size matches input
	if dst.len() != data.len() {
		return Err(GaussianError::InvalidPeriod {
			period: dst.len(),
			data_len: data.len(),
		});
	}

	gaussian_with_kernel_into(input, kern, dst)
}

#[inline(always)]
fn gaussian_with_kernel_into(input: &GaussianInput, kernel: Kernel, out: &mut [f64]) -> Result<(), GaussianError> {
	let (data, period, poles, first, chosen) = gaussian_prepare(input, kernel)?;

	// Initialize NaN prefix
	out[..first + period - 1].fill(f64::NAN);

	// Compute directly into output buffer
	gaussian_compute_into(data, period, poles, first, chosen, out);

	Ok(())
}

#[inline(always)]
fn gaussian_prepare<'a>(
	input: &'a GaussianInput,
	kernel: Kernel,
) -> Result<
	(
		// data
		&'a [f64],
		// period
		usize,
		// poles
		usize,
		// first
		usize,
		// chosen
		Kernel,
	),
	GaussianError,
> {
	let data: &[f64] = input.as_ref();
	let len = data.len();

	if len == 0 {
		return Err(GaussianError::NoData);
	}

	let first = data
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(GaussianError::AllValuesNaN)?;

	let period = input.params.period.unwrap_or(14);
	let poles = input.params.poles.unwrap_or(4);

	if period == 0 || period > len {
		return Err(GaussianError::InvalidPeriod { period, data_len: len });
	}

	if !(1..=4).contains(&poles) {
		return Err(GaussianError::InvalidPoles { poles });
	}

	if len - first < period {
		return Err(GaussianError::NotEnoughValidData {
			needed: period,
			valid: len - first,
		});
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	Ok((data, period, poles, first, chosen))
}

#[inline(always)]
fn gaussian_compute_into(data: &[f64], period: usize, poles: usize, first: usize, kernel: Kernel, out: &mut [f64]) {
	unsafe {
		match kernel {
			Kernel::Scalar | Kernel::ScalarBatch => gaussian_scalar(data, period, poles, out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => gaussian_avx2(data, period, poles, out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => gaussian_avx512(data, period, poles, out),
			_ => unreachable!(),
		}
	}
}

#[inline(always)]
fn alpha_from(period: usize, poles: usize) -> f64 {
	let beta = {
		let numerator = 1.0 - (2.0 * PI / period as f64).cos();
		let denominator = (2.0_f64).powf(1.0 / poles as f64) - 1.0;
		numerator / denominator
	};
	let tmp = beta * beta + 2.0 * beta;
	-beta + tmp.sqrt()
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
#[inline]
unsafe fn gaussian_rows8_avx512(data: &[f64], params: &[GaussianParams], out_rows: &mut [f64], cols: usize) {
	debug_assert_eq!(params.len(), LANES_AVX512);
	debug_assert_eq!(out_rows.len(), LANES_AVX512 * cols);

	// ---- per-lane α and pole-count ----------------------------------------
	let mut alpha_arr = [0.0f64; LANES_AVX512];
	let mut pole_arr = [0u32; LANES_AVX512];
	for (lane, prm) in params.iter().enumerate() {
		let p = prm.period.unwrap_or(14);
		let k = prm.poles.unwrap_or(4);
		alpha_arr[lane] = alpha_from(p, k);
		pole_arr[lane] = k as u32;
	}

	let alpha_v = _mm512_loadu_pd(alpha_arr.as_ptr());
	let one_minus_v = _mm512_sub_pd(_mm512_set1_pd(1.0), alpha_v);

	// stage-masks: lane bit = 1 when that stage is **present**
	let mask_for = |stage: u32| -> __mmask8 {
		let mut m: u8 = 0;
		for lane in 0..LANES_AVX512 {
			if pole_arr[lane] > stage {
				m |= 1 << lane;
			}
		}
		m as __mmask8
	};
	let m0 = mask_for(0);
	let m1 = mask_for(1);
	let m2 = mask_for(2);
	let m3 = mask_for(3);

	// state per pole
	let mut st0 = _mm512_setzero_pd();
	let mut st1 = _mm512_setzero_pd();
	let mut st2 = _mm512_setzero_pd();
	let mut st3 = _mm512_setzero_pd();

	// ---- main time loop ----------------------------------------------------
	for (t, &x_n) in data.iter().enumerate() {
		let x_vec = _mm512_set1_pd(x_n);

		// stage 0  : y = α x + (1-α) st
		let y0 = _mm512_fmadd_pd(alpha_v, x_vec, _mm512_mul_pd(one_minus_v, st0));
		st0 = _mm512_mask_mov_pd(st0, m0, y0);

		// stage 1
		let y1 = _mm512_fmadd_pd(alpha_v, st0, _mm512_mul_pd(one_minus_v, st1));
		st1 = _mm512_mask_mov_pd(st1, m1, y1);

		// stage 2
		let y2 = _mm512_fmadd_pd(alpha_v, st1, _mm512_mul_pd(one_minus_v, st2));
		st2 = _mm512_mask_mov_pd(st2, m2, y2);

		// stage 3
		let y3 = _mm512_fmadd_pd(alpha_v, st2, _mm512_mul_pd(one_minus_v, st3));
		st3 = _mm512_mask_mov_pd(st3, m3, y3);

		let mut y = st0;
		y = _mm512_mask_mov_pd(y, m1, st1);
		y = _mm512_mask_mov_pd(y, m2, st2);
		y = _mm512_mask_mov_pd(y, m3, st3);

		// scatter → row-major
		let mut tmp = [0.0f64; LANES_AVX512];
		_mm512_storeu_pd(tmp.as_mut_ptr(), y);
		for lane in 0..LANES_AVX512 {
			out_rows[lane * cols + t] = tmp[lane];
		}
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn gaussian_batch_tile_avx2(
	data: &[f64],
	combos: &[GaussianParams],
	out_mu: &mut [core::mem::MaybeUninit<f64>],
	cols: usize,
) {
	// view as &[f64] for the SIMD helpers
	let out = core::slice::from_raw_parts_mut(out_mu.as_mut_ptr() as *mut f64, out_mu.len());

	let mut row = 0;
	while row + LANES_AVX2 <= combos.len() {
		gaussian_rows4_avx2(
			data,
			&combos[row..row + LANES_AVX2],
			&mut out[row * cols..(row + LANES_AVX2) * cols],
			cols,
		);
		row += LANES_AVX2;
	}
	for r in row..combos.len() {
		gaussian_row_scalar(data, &combos[r], &mut out[r * cols..(r + 1) * cols]);
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn gaussian_rows4_avx2(data: &[f64], params: &[GaussianParams], out_rows: &mut [f64], cols: usize) {
	debug_assert_eq!(params.len(), LANES_AVX2);
	debug_assert_eq!(out_rows.len(), LANES_AVX2 * cols);

	// lane-specific α and pole-count
	let mut alpha_arr = [0.0f64; LANES_AVX2];
	let mut pole_arr = [0u32; LANES_AVX2];
	for (l, prm) in params.iter().enumerate() {
		alpha_arr[l] = alpha_from(prm.period.unwrap_or(14), prm.poles.unwrap_or(4));
		pole_arr[l] = prm.poles.unwrap_or(4) as u32;
	}
	let alpha_v = _mm256_loadu_pd(alpha_arr.as_ptr());
	let one_minus_v = _mm256_sub_pd(_mm256_set1_pd(1.0), alpha_v);

	// state per pole
	let mut st0 = _mm256_setzero_pd();
	let mut st1 = _mm256_setzero_pd();
	let mut st2 = _mm256_setzero_pd();
	let mut st3 = _mm256_setzero_pd();

	// scratch arrays to pick lane-wise final value
	let mut y0a = [0.0; LANES_AVX2];
	let mut y1a = [0.0; LANES_AVX2];
	let mut y2a = [0.0; LANES_AVX2];
	let mut y3a = [0.0; LANES_AVX2];

	for (t, &x_n) in data.iter().enumerate() {
		let x_v = _mm256_set1_pd(x_n);

		// stage-0 … stage-3
		let y0_v = _mm256_fmadd_pd(alpha_v, x_v, _mm256_mul_pd(one_minus_v, st0));
		st0 = y0_v;

		let y1_v = _mm256_fmadd_pd(alpha_v, st0, _mm256_mul_pd(one_minus_v, st1));
		st1 = y1_v;

		let y2_v = _mm256_fmadd_pd(alpha_v, st1, _mm256_mul_pd(one_minus_v, st2));
		st2 = y2_v;

		let y3_v = _mm256_fmadd_pd(alpha_v, st2, _mm256_mul_pd(one_minus_v, st3));
		st3 = y3_v;

		// store to scratch ­- (compilers fold these back-to-back stores nicely)
		_mm256_storeu_pd(y0a.as_mut_ptr(), y0_v);
		_mm256_storeu_pd(y1a.as_mut_ptr(), y1_v);
		_mm256_storeu_pd(y2a.as_mut_ptr(), y2_v);
		_mm256_storeu_pd(y3a.as_mut_ptr(), y3_v);

		// choose per-lane final output and scatter
		for lane in 0..LANES_AVX2 {
			let final_y = match pole_arr[lane] {
				1 => y0a[lane],
				2 => y1a[lane],
				3 => y2a[lane],
				_ => y3a[lane], // 4
			};
			out_rows[lane * cols + t] = final_y;
		}
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
#[inline]
unsafe fn gaussian_batch_tile_avx512(
	data: &[f64],
	combos: &[GaussianParams],
	out_mu: &mut [core::mem::MaybeUninit<f64>], // <-- changed
	cols: usize,
) {
	// temporary view as &mut [f64]
	let out = core::slice::from_raw_parts_mut(out_mu.as_mut_ptr() as *mut f64, out_mu.len());

	let mut row = 0;
	while row + LANES_AVX512 <= combos.len() {
		gaussian_rows8_avx512(
			data,
			&combos[row..row + LANES_AVX512],
			&mut out[row * cols..(row + LANES_AVX512) * cols],
			cols,
		);
		row += LANES_AVX512;
	}
	// tail rows (≤7) – scalar
	for r in row..combos.len() {
		gaussian_row_scalar(data, &combos[r], &mut out[r * cols..(r + 1) * cols]);
	}
}

#[inline(always)]
fn gaussian_batch_inner(
	data: &[f64],
	sweep: &GaussianBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<GaussianBatchOutput, GaussianError> {
	#[cfg(not(target_arch = "wasm32"))]
	use rayon::prelude::*;
	use std::{arch::is_x86_feature_detected, mem::MaybeUninit};

	// ------------------------------------------------------------
	// 0.  Expand grid  +  warm-up lengths
	// ----------------------------------------------------------
	let combos = expand_grid(sweep);
	if combos.is_empty() || data.is_empty() {
		return Err(GaussianError::NoData);
	}
	let cols = data.len();
	let rows = combos.len();

	let first_valid = data
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(GaussianError::AllValuesNaN)?;

	for c in &combos {
		let period = c.period.unwrap_or(14);
		let poles = c.poles.unwrap_or(4);
		if period == 0 || period > cols {
			return Err(GaussianError::InvalidPeriod { period, data_len: cols });
		}
		if !(1..=4).contains(&poles) {
			return Err(GaussianError::InvalidPoles { poles });
		}
		if cols - first_valid < period {
			return Err(GaussianError::NotEnoughValidData {
				needed: period,
				valid: cols - first_valid,
			});
		}
	}

	let warm: Vec<usize> = combos
		.iter()
		.map(|c| {
			let w = first_valid + c.period.unwrap_or(14);
			w.min(cols)
		})
		.collect();

	// ------------------------------------------------------------
	// 1.  Allocate matrix & write NaN prefixes
	// ----------------------------------------------------------
	let mut raw = make_uninit_matrix(rows, cols);

	#[cfg(not(target_arch = "wasm32"))]
	{
		raw.par_chunks_mut(cols).zip(warm.par_iter()).for_each(|(row, &w)| {
			for cell in &mut row[..w] {
				cell.write(f64::NAN);
			}
		});
	}
	#[cfg(target_arch = "wasm32")]
	{
		for (row, &w) in raw.chunks_mut(cols).zip(warm.iter()) {
			for cell in &mut row[..w] {
				cell.write(f64::NAN);
			}
		}
	}

	// ------------------------------------------------------------
	// 2.  CPU-feature check  +  row-runner selection
	// ----------------------------------------------------------
	#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
	let have_avx512 = cfg!(target_feature = "avx512f") && is_x86_feature_detected!("avx512f");
	#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
	let have_avx512 = false;

	#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
	let have_avx2 = cfg!(target_feature = "avx2") && is_x86_feature_detected!("avx2");
	#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
	let have_avx2 = false;

	let chosen = match kern {
		Kernel::Avx512 if have_avx512 => Kernel::Avx512,
		Kernel::Avx2 if have_avx2 => Kernel::Avx2,
		_ => Kernel::Scalar,
	};

	type RowRunner = unsafe fn(&[f64], &GaussianParams, &mut [f64]);

	let row_runner: RowRunner = match chosen {
		#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
		Kernel::Avx512 => gaussian_row_avx512,
		#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
		Kernel::Avx2 => gaussian_row_avx2,
		_ => gaussian_row_scalar, // already `unsafe fn`
	};

	// ------------------------------------------------------------
	// 3.  Helper to run one row
	// ----------------------------------------------------------
	#[inline(always)]
	unsafe fn compute_row(
		row_idx: usize,
		dst_mu: &mut [MaybeUninit<f64>],
		combos: &[GaussianParams],
		data: &[f64],
		cols: usize,
		runner: RowRunner,
	) {
		let out = std::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, cols);
		runner(data, &combos[row_idx], out);
	}

	// ------------------------------------------------------------
	// 4.  Main computation (parallel / serial)
	// ----------------------------------------------------------
	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			match chosen {
				// ---- AVX-512 (8-row tiles) ----------------------------------
				#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
				Kernel::Avx512 => {
					let tiles = rows / LANES_AVX512;

					raw.par_chunks_exact_mut(cols * LANES_AVX512)
						.zip(combos.par_chunks_exact(LANES_AVX512))
						.for_each(|(dst_blk, prm_blk)| unsafe {
							gaussian_batch_tile_avx512(data, prm_blk, dst_blk, cols);
						});

					raw[tiles * cols * LANES_AVX512..]
						.par_chunks_mut(cols)
						.enumerate()
						.for_each(|(i, dst)| unsafe {
							compute_row(tiles * LANES_AVX512 + i, dst, &combos, data, cols, row_runner);
						});
				}

				// ---- AVX-2 (4-row tiles) ------------------------------------
				#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
				Kernel::Avx2 => {
					let tiles = rows / LANES_AVX2;

					raw.par_chunks_exact_mut(cols * LANES_AVX2)
						.zip(combos.par_chunks_exact(LANES_AVX2))
						.for_each(|(dst_blk, prm_blk)| unsafe {
							gaussian_batch_tile_avx2(data, prm_blk, dst_blk, cols);
						});

					raw[tiles * cols * LANES_AVX2..]
						.par_chunks_mut(cols)
						.enumerate()
						.for_each(|(i, dst)| unsafe {
							compute_row(tiles * LANES_AVX2 + i, dst, &combos, data, cols, row_runner);
						});
				}

				// ---- Scalar fallback (parallel) -----------------------------
				_ => {
					raw.par_chunks_mut(cols).enumerate().for_each(|(row, dst)| unsafe {
						compute_row(row, dst, &combos, data, cols, row_runner);
					});
				}
			}
		}
		#[cfg(target_arch = "wasm32")]
		{
			// For WASM, always use sequential processing
			for (row, dst) in raw.chunks_mut(cols).enumerate() {
				unsafe {
					compute_row(row, dst, &combos, data, cols, row_runner);
				}
			}
		}
	} else {
		match chosen {
			// ---- AVX-512 serial -----------------------------------------
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => {
				let tiles = rows / LANES_AVX512;
				unsafe {
					gaussian_batch_tile_avx512(
						data,
						&combos[..tiles * LANES_AVX512],
						&mut raw[..tiles * cols * LANES_AVX512],
						cols,
					);
				}
				for (i, dst) in raw[tiles * cols * LANES_AVX512..].chunks_mut(cols).enumerate() {
					unsafe {
						compute_row(tiles * LANES_AVX512 + i, dst, &combos, data, cols, row_runner);
					}
				}
			}

			// ---- AVX-2 serial -------------------------------------------
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => {
				let tiles = rows / LANES_AVX2;
				unsafe {
					gaussian_batch_tile_avx2(
						data,
						&combos[..tiles * LANES_AVX2],
						&mut raw[..tiles * cols * LANES_AVX2],
						cols,
					);
				}
				for (i, dst) in raw[tiles * cols * LANES_AVX2..].chunks_mut(cols).enumerate() {
					unsafe {
						compute_row(tiles * LANES_AVX2 + i, dst, &combos, data, cols, row_runner);
					}
				}
			}

			// ---- Scalar serial ------------------------------------------
			_ => {
				for (row, dst) in raw.chunks_mut(cols).enumerate() {
					unsafe {
						compute_row(row, dst, &combos, data, cols, row_runner);
					}
				}
			}
		}
	}

	// ------------------------------------------------------------
	// 5.  Finalise
	// ----------------------------------------------------------
	let values: Vec<f64> = unsafe { std::mem::transmute(raw) };
	Ok(GaussianBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
pub unsafe fn gaussian_row_scalar(data: &[f64], prm: &GaussianParams, out: &mut [f64]) {
	gaussian_scalar(data, prm.period.unwrap_or(14), prm.poles.unwrap_or(4), out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn gaussian_row_avx2(data: &[f64], prm: &GaussianParams, out: &mut [f64]) {
	let mut combos = [prm.clone(); LANES_AVX2];
	let mut buf = vec![0.0f64; LANES_AVX2 * data.len()];
	gaussian_rows4_avx2(data, &combos, &mut buf, data.len());
	out.copy_from_slice(&buf[..data.len()]);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
#[inline]
pub unsafe fn gaussian_row_avx512(data: &[f64], prm: &GaussianParams, out: &mut [f64]) {
	// fast path: 1-row “batch” using the SIMD core
	let mut combos = [prm.clone(); LANES_AVX512];
	let mut buf = vec![0.0f64; LANES_AVX512 * data.len()];
	gaussian_rows8_avx512(data, &combos, &mut buf, data.len());
	out.copy_from_slice(&buf[..data.len()]);
}

// =================== Macro-Based Unit Tests ====================

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;
	use proptest::prelude::*;

	fn check_gaussian_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let default_params = GaussianParams {
			period: None,
			poles: None,
		};
		let input = GaussianInput::from_candles(&candles, "close", default_params);
		let output = gaussian_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_gaussian_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = GaussianParams {
			period: Some(14),
			poles: Some(4),
		};
		let input = GaussianInput::from_candles(&candles, "close", params);
		let result = gaussian_with_kernel(&input, kernel)?;
		let expected_last_five = [
			59221.90637814869,
			59236.15215167245,
			59207.10087088464,
			59178.48276885589,
			59085.36983209433,
		];
		let start = result.values.len().saturating_sub(5);
		for (i, &val) in result.values[start..].iter().enumerate() {
			let diff = (val - expected_last_five[i]).abs();
			assert!(
				diff < 1e-4,
				"[{}] Gaussian {:?} mismatch at idx {}: got {}, expected {}",
				test_name,
				kernel,
				i,
				val,
				expected_last_five[i]
			);
		}
		Ok(())
	}

	fn check_gaussian_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = GaussianInput::with_default_candles(&candles);
		match input.data {
			GaussianData::Candles { source, .. } => assert_eq!(source, "close"),
			_ => panic!("Expected GaussianData::Candles"),
		}
		let output = gaussian_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_gaussian_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data = [1.0, 2.0, 3.0];
		let params = GaussianParams {
			period: Some(0),
			poles: Some(2),
		};
		let input = GaussianInput::from_slice(&data, params);
		let res = gaussian_with_kernel(&input, kernel);
		assert!(
			matches!(res, Err(GaussianError::InvalidPeriod { .. })),
			"[{test_name}] expected InvalidPeriod error"
		);
		Ok(())
	}

	fn check_gaussian_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let empty: [f64; 0] = [];
		let input = GaussianInput::from_slice(&empty, GaussianParams::default());
		let res = gaussian_with_kernel(&input, kernel);
		assert!(
			matches!(res, Err(GaussianError::NoData)),
			"[{test_name}] expected NoData error"
		);
		Ok(())
	}

	fn check_gaussian_invalid_poles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data = [1.0, 2.0, 3.0, 4.0];
		let params = GaussianParams {
			period: Some(2),
			poles: Some(5),
		};
		let input = GaussianInput::from_slice(&data, params);
		let res = gaussian_with_kernel(&input, kernel);
		assert!(
			matches!(res, Err(GaussianError::InvalidPoles { .. })),
			"[{test_name}] expected InvalidPoles error"
		);
		Ok(())
	}

	fn check_gaussian_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data = [f64::NAN; 5];
		let params = GaussianParams {
			period: Some(3),
			poles: None,
		};
		let input = GaussianInput::from_slice(&data, params);
		let res = gaussian_with_kernel(&input, kernel);
		assert!(
			matches!(res, Err(GaussianError::AllValuesNaN)),
			"[{test_name}] expected AllValuesNaN error"
		);
		Ok(())
	}

	fn check_gaussian_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data_small = [10.0, 20.0, 30.0];
		let params = GaussianParams {
			period: Some(10),
			poles: None,
		};
		let input = GaussianInput::from_slice(&data_small, params);
		let res = gaussian_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] Gaussian should fail with period exceeding length",
			test_name
		);
		Ok(())
	}

	fn check_gaussian_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let single_point = [42.0];
		let params = GaussianParams {
			period: Some(14),
			poles: None,
		};
		let input = GaussianInput::from_slice(&single_point, params);
		let res = gaussian_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] Gaussian should fail with insufficient data",
			test_name
		);
		Ok(())
	}

	fn check_gaussian_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let first_params = GaussianParams {
			period: Some(14),
			poles: Some(4),
		};
		let first_input = GaussianInput::from_candles(&candles, "close", first_params);
		let first_result = gaussian_with_kernel(&first_input, kernel)?;
		assert_eq!(first_result.values.len(), candles.close.len());
		let second_params = GaussianParams {
			period: Some(7),
			poles: Some(2),
		};
		let second_input = GaussianInput::from_slice(&first_result.values, second_params);
		let second_result = gaussian_with_kernel(&second_input, kernel)?;
		assert_eq!(second_result.values.len(), first_result.values.len());
		for i in 10..second_result.values.len() {
			assert!(!second_result.values[i].is_nan(), "NaN found at index {}", i);
		}
		Ok(())
	}

	fn check_gaussian_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = GaussianInput::from_candles(&candles, "close", GaussianParams::default());
		let res = gaussian_with_kernel(&input, kernel)?;
		assert_eq!(res.values.len(), candles.close.len());
		let skip = input.params.poles.unwrap_or(4);
		for val in res.values.iter().skip(skip) {
			assert!(
				val.is_finite(),
				"[{}] Gaussian output should be finite once settled.",
				test_name
			);
		}
		Ok(())
	}

	fn check_gaussian_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let period = 14;
		let poles = 4;
		let input = GaussianInput::from_candles(
			&candles,
			"close",
			GaussianParams {
				period: Some(period),
				poles: Some(poles),
			},
		);
		let batch_output = gaussian_with_kernel(&input, kernel)?.values;
		let mut stream = GaussianStream::try_new(GaussianParams {
			period: Some(period),
			poles: Some(poles),
		})?;
		let mut stream_values = Vec::with_capacity(candles.close.len());
		for &price in &candles.close {
			stream_values.push(stream.update(price));
		}
		assert_eq!(batch_output.len(), stream_values.len());
		let skip = poles;
		for (i, (&b, &s)) in batch_output.iter().zip(stream_values.iter()).enumerate().skip(skip) {
			if b.is_nan() && s.is_nan() {
				continue;
			}
			let diff = (b - s).abs();
			assert!(
				diff < 1e-9,
				"[{}] Gaussian streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
				test_name,
				i,
				b,
				s,
				diff
			);
		}
		Ok(())
	}

	fn check_gaussian_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);

		let strat = (
			proptest::collection::vec((-1e6f64..1e6).prop_filter("finite", |x| x.is_finite()), 10..200),
			1usize..30,
		)
			.prop_filter("period <= len", |(d, p)| *p <= d.len());

		proptest::test_runner::TestRunner::default()
			.run(&strat, |(data, period)| {
				let params = GaussianParams {
					period: Some(period),
					poles: Some(2),
				};
				let input = GaussianInput::from_slice(&data, params);
				let GaussianOutput { values: out } = gaussian_with_kernel(&input, kernel).unwrap();
				prop_assert_eq!(out.len(), data.len());
				Ok(())
			})
			.unwrap();

		Ok(())
	}

	macro_rules! generate_all_gaussian_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(
                    #[test]
                    fn [<$test_fn _scalar_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar);
                    }
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
	fn check_gaussian_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		// Test with multiple parameter combinations
		let test_cases = vec![
			GaussianParams {
				period: Some(14),
				poles: Some(4),
			}, // default
			GaussianParams {
				period: Some(10),
				poles: Some(1),
			}, // minimum poles
			GaussianParams {
				period: Some(30),
				poles: Some(2),
			}, // medium values
			GaussianParams {
				period: Some(20),
				poles: Some(3),
			}, // different combination
			GaussianParams {
				period: Some(50),
				poles: Some(4),
			}, // larger period
			GaussianParams {
				period: Some(5),
				poles: Some(1),
			}, // small period, min poles
			GaussianParams {
				period: Some(100),
				poles: Some(4),
			}, // large period
			GaussianParams {
				period: None,
				poles: None,
			}, // None values (use defaults)
		];

		for params in test_cases {
			let input = GaussianInput::from_candles(&candles, "close", params);
			let output = gaussian_with_kernel(&input, kernel)?;

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
                         with params period={:?}, poles={:?}",
						test_name, val, bits, i, params.period, params.poles
					);
				}

				// Check for init_matrix_prefixes poison (0x22222222_22222222)
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
                         with params period={:?}, poles={:?}",
						test_name, val, bits, i, params.period, params.poles
					);
				}

				// Check for make_uninit_matrix poison (0x33333333_33333333)
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
                         with params period={:?}, poles={:?}",
						test_name, val, bits, i, params.period, params.poles
					);
				}
			}
		}

		Ok(())
	}

	// Release mode stub - does nothing
	#[cfg(not(debug_assertions))]
	fn check_gaussian_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		Ok(())
	}

	generate_all_gaussian_tests!(
		check_gaussian_partial_params,
		check_gaussian_accuracy,
		check_gaussian_default_candles,
		check_gaussian_zero_period,
		check_gaussian_period_exceeds_length,
		check_gaussian_very_small_dataset,
		check_gaussian_empty_input,
		check_gaussian_invalid_poles,
		check_gaussian_all_nan,
		check_gaussian_reinput,
		check_gaussian_nan_handling,
		check_gaussian_streaming,
		check_gaussian_property,
		check_gaussian_no_poison
	);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let output = GaussianBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;
		let def = GaussianParams::default();
		let row = output.values_for(&def).expect("default row missing");
		assert_eq!(row.len(), c.close.len());
		let expected = [
			59221.90637814869,
			59236.15215167245,
			59207.10087088464,
			59178.48276885589,
			59085.36983209433,
		];
		let start = row.len() - 5;
		for (i, &v) in row[start..].iter().enumerate() {
			assert!(
				(v - expected[i]).abs() < 1e-4,
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
				#[test] fn [<$fn_name _avx2>]()        {
					let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch);
				}
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
	fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		// Test multiple batch configurations with different parameter ranges
		let batch_configs = vec![
			// Original test case
			((10, 30, 10), (1, 4, 1)),
			// Edge cases
			((5, 5, 0), (1, 1, 0)),      // Single parameter combo
			((100, 120, 20), (2, 4, 2)), // Larger periods
			((7, 21, 7), (1, 3, 2)),     // Different steps
			((15, 45, 15), (1, 4, 3)),   // Multiple poles
			((3, 12, 3), (1, 2, 1)),     // Small periods
		];

		for ((p_start, p_end, p_step), (poles_start, poles_end, poles_step)) in batch_configs {
			let output = GaussianBatchBuilder::new()
				.kernel(kernel)
				.period_range(p_start, p_end, p_step)
				.poles_range(poles_start, poles_end, poles_step)
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
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} \
                         (flat index {}) with params period={:?}, poles={:?}",
						test, val, bits, row, col, idx, combo.period, combo.poles
					);
				}

				// Check for init_matrix_prefixes poison (0x22222222_22222222)
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} \
                         (flat index {}) with params period={:?}, poles={:?}",
						test, val, bits, row, col, idx, combo.period, combo.poles
					);
				}

				// Check for make_uninit_matrix poison (0x33333333_33333333)
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} \
                         (flat index {}) with params period={:?}, poles={:?}",
						test, val, bits, row, col, idx, combo.period, combo.poles
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

// WASM bindings
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

// Helper function for batch operations with output buffer
#[inline(always)]
fn gaussian_batch_inner_into(
	data: &[f64],
	sweep: &GaussianBatchRange,
	kern: Kernel,
	parallel: bool,
	out: &mut [f64],
) -> Result<Vec<GaussianParams>, GaussianError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() || data.is_empty() {
		return Err(GaussianError::NoData);
	}
	let cols = data.len();
	let rows = combos.len();

	if out.len() != rows * cols {
		return Err(GaussianError::InvalidPeriod {
			period: 0,
			data_len: out.len(),
		});
	}

	let first_valid = data
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(GaussianError::AllValuesNaN)?;

	for c in &combos {
		let period = c.period.unwrap_or(14);
		let poles = c.poles.unwrap_or(4);
		if period == 0 || period > cols {
			return Err(GaussianError::InvalidPeriod { period, data_len: cols });
		}
		if !(1..=4).contains(&poles) {
			return Err(GaussianError::InvalidPoles { poles });
		}
		if cols - first_valid < period {
			return Err(GaussianError::NotEnoughValidData {
				needed: period,
				valid: cols - first_valid,
			});
		}
	}

	let warm: Vec<usize> = combos
		.iter()
		.map(|c| {
			let w = first_valid + c.period.unwrap_or(14);
			w.min(cols)
		})
		.collect();

	// Reinterpret output slice as MaybeUninit
	let raw = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len()) };

	// Initialize NaN prefixes
	unsafe { init_matrix_prefixes(raw, cols, &warm) };

	// Determine kernel
	let chosen = match kern {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	type RowRunner = unsafe fn(&[f64], &GaussianParams, &mut [f64]);
	let row_runner: RowRunner = match chosen {
		#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
		Kernel::Avx512 => gaussian_row_avx512,
		#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
		Kernel::Avx2 => gaussian_row_avx2,
		_ => gaussian_row_scalar,
	};

	// Process rows
	let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
		let dst = std::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());
		row_runner(data, &combos[row], dst);
	};

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

	Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "gaussian")]
#[pyo3(signature = (data, period, poles, kernel=None))]
pub fn gaussian_py<'py>(
	py: Python<'py>,
	data: PyReadonlyArray1<'py, f64>,
	period: usize,
	poles: usize,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, false)?;

	let params = GaussianParams {
		period: Some(period),
		poles: Some(poles),
	};
	let gaussian_in = GaussianInput::from_slice(slice_in, params);

	let result_vec: Vec<f64> = py
		.allow_threads(|| gaussian_with_kernel(&gaussian_in, kern).map(|o| o.values))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "GaussianStream")]
pub struct GaussianStreamPy {
	stream: GaussianStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl GaussianStreamPy {
	#[new]
	fn new(period: usize, poles: usize) -> PyResult<Self> {
		let params = GaussianParams {
			period: Some(period),
			poles: Some(poles),
		};
		let stream = GaussianStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(GaussianStreamPy { stream })
	}

	fn update(&mut self, value: f64) -> Option<f64> {
		Some(self.stream.update(value))
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "gaussian_batch")]
#[pyo3(signature = (data, period_range, poles_range, kernel=None))]
pub fn gaussian_batch_py<'py>(
	py: Python<'py>,
	data: PyReadonlyArray1<'py, f64>,
	period_range: (usize, usize, usize),
	poles_range: (usize, usize, usize),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, true)?;

	let sweep = GaussianBatchRange {
		period: period_range,
		poles: poles_range,
	};

	let combos = expand_grid(&sweep);
	let rows = combos.len();
	let cols = slice_in.len();

	// Pre-allocate output array
	let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let slice_out = unsafe { out_arr.as_slice_mut()? };

	// Heavy work without the GIL
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
			gaussian_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
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
	dict.set_item(
		"poles",
		combos
			.iter()
			.map(|p| p.poles.unwrap_or(4) as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;

	Ok(dict)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn gaussian_js(data: &[f64], period: usize, poles: usize) -> Result<Vec<f64>, JsValue> {
	let params = GaussianParams {
		period: Some(period),
		poles: Some(poles),
	};
	let input = GaussianInput::from_slice(data, params);

	// Allocate output buffer once
	let mut output = vec![0.0; data.len()];

	// Compute directly into output buffer
	gaussian_into_slice(&mut output, &input, Kernel::Auto).map_err(|e| JsValue::from_str(&e.to_string()))?;

	Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn gaussian_into(
	in_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period: usize,
	poles: usize,
) -> Result<(), JsValue> {
	// Check for null pointers
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("null pointer passed to gaussian_into"));
	}

	unsafe {
		// Create slice from pointer
		let data = std::slice::from_raw_parts(in_ptr, len);

		// Validate inputs
		if period == 0 || period > len {
			return Err(JsValue::from_str("Invalid period"));
		}

		if !(1..=4).contains(&poles) {
			return Err(JsValue::from_str("Invalid poles (must be 1-4)"));
		}

		// Calculate Gaussian
		let params = GaussianParams {
			period: Some(period),
			poles: Some(poles),
		};
		let input = GaussianInput::from_slice(data, params);

		// Check for aliasing (input and output buffers are the same)
		if in_ptr == out_ptr {
			// Use temporary buffer to avoid corruption during sliding window computation
			let mut temp = vec![0.0; len];
			gaussian_into_slice(&mut temp, &input, Kernel::Auto).map_err(|e| JsValue::from_str(&e.to_string()))?;

			// Copy results back to output
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			out.copy_from_slice(&temp);
		} else {
			// No aliasing, compute directly into output
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			gaussian_into_slice(out, &input, Kernel::Auto).map_err(|e| JsValue::from_str(&e.to_string()))?;
		}

		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn gaussian_alloc(len: usize) -> *mut f64 {
	// Allocate memory for input/output buffer
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec); // Prevent deallocation
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn gaussian_free(ptr: *mut f64, len: usize) {
	// Free allocated memory
	if !ptr.is_null() {
		unsafe {
			let _ = Vec::from_raw_parts(ptr, len, len);
		}
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn gaussian_batch_js(
	data: &[f64],
	period_start: usize,
	period_end: usize,
	period_step: usize,
	poles_start: usize,
	poles_end: usize,
	poles_step: usize,
) -> Result<Vec<f64>, JsValue> {
	let sweep = GaussianBatchRange {
		period: (period_start, period_end, period_step),
		poles: (poles_start, poles_end, poles_step),
	};

	gaussian_batch_inner(data, &sweep, Kernel::Auto, false)
		.map(|output| output.values)
		.map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn gaussian_batch_metadata_js(
	period_start: usize,
	period_end: usize,
	period_step: usize,
	poles_start: usize,
	poles_end: usize,
	poles_step: usize,
) -> Result<Vec<f64>, JsValue> {
	let sweep = GaussianBatchRange {
		period: (period_start, period_end, period_step),
		poles: (poles_start, poles_end, poles_step),
	};

	let combos = expand_grid(&sweep);
	let metadata: Vec<f64> = combos
		.iter()
		.flat_map(|combo| vec![combo.period.unwrap_or(14) as f64, combo.poles.unwrap_or(4) as f64])
		.collect();

	Ok(metadata)
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct GaussianBatchConfig {
	pub period_range: (usize, usize, usize),
	pub poles_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct GaussianBatchJsOutput {
	pub values: Vec<f64>,
	pub combos: Vec<GaussianParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = gaussian_batch)]
pub fn gaussian_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	let config: GaussianBatchConfig =
		serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

	let sweep = GaussianBatchRange {
		period: config.period_range,
		poles: config.poles_range,
	};

	let output =
		gaussian_batch_inner(data, &sweep, Kernel::Auto, false).map_err(|e| JsValue::from_str(&e.to_string()))?;

	let js_output = GaussianBatchJsOutput {
		values: output.values,
		combos: output.combos,
		rows: output.rows,
		cols: output.cols,
	};

	serde_wasm_bindgen::to_value(&js_output).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn gaussian_batch_into(
	in_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period_start: usize,
	period_end: usize,
	period_step: usize,
	poles_start: usize,
	poles_end: usize,
	poles_step: usize,
) -> Result<usize, JsValue> {
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("null pointer passed to gaussian_batch_into"));
	}

	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);

		let sweep = GaussianBatchRange {
			period: (period_start, period_end, period_step),
			poles: (poles_start, poles_end, poles_step),
		};

		let combos = expand_grid(&sweep);
		let rows = combos.len();
		let cols = len;

		let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);

		// Use optimized batch processing
		gaussian_batch_inner_into(data, &sweep, Kernel::Auto, false, out)
			.map_err(|e| JsValue::from_str(&e.to_string()))?;

		Ok(rows)
	}
}
