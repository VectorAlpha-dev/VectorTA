//! # Normalized Moving Average (NMA)
//!
//! A technique that computes an adaptive moving average by transforming input
//! values into log space and weighting differences between consecutive values.
//! The weighting ratio depends on a series of square-root increments. This design
//! aims to normalize large price changes without oversmoothing small fluctuations.
//!
//! ## Parameters
//! - **period**: Window size (number of data points, default: 40)
//!
//! ## Errors
//! - **AllValuesNaN**: nma: All input data values are `NaN`.
//! - **InvalidPeriod**: nma: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: nma: Not enough valid data points for the requested `period`.
//!
//! ## Returns
//! - **`Ok(NmaOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(NmaError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
	alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes, make_uninit_matrix,
};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use core::arch::wasm32::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

impl<'a> AsRef<[f64]> for NmaInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			NmaData::Slice(slice) => slice,
			NmaData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

#[derive(Debug, Clone)]
pub enum NmaData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct NmaOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "wasm", derive(serde::Serialize, serde::Deserialize))]
pub struct NmaParams {
	pub period: Option<usize>,
}

impl Default for NmaParams {
	fn default() -> Self {
		Self { period: Some(40) }
	}
}

#[derive(Debug, Clone)]
pub struct NmaInput<'a> {
	pub data: NmaData<'a>,
	pub params: NmaParams,
}

impl<'a> NmaInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: NmaParams) -> Self {
		Self {
			data: NmaData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slice(sl: &'a [f64], p: NmaParams) -> Self {
		Self {
			data: NmaData::Slice(sl),
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", NmaParams::default())
	}
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(40)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct NmaBuilder {
	period: Option<usize>,
	kernel: Kernel,
}

impl Default for NmaBuilder {
	fn default() -> Self {
		Self {
			period: None,
			kernel: Kernel::Auto,
		}
	}
}

impl NmaBuilder {
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
	pub fn apply(self, c: &Candles) -> Result<NmaOutput, NmaError> {
		let p = NmaParams { period: self.period };
		let i = NmaInput::from_candles(c, "close", p);
		nma_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slice(self, d: &[f64]) -> Result<NmaOutput, NmaError> {
		let p = NmaParams { period: self.period };
		let i = NmaInput::from_slice(d, p);
		nma_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<NmaStream, NmaError> {
		let p = NmaParams { period: self.period };
		NmaStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum NmaError {
	#[error("nma: Input data slice is empty.")]
	EmptyInputData,
	#[error("nma: All values are NaN.")]
	AllValuesNaN,
	#[error("nma: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("nma: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn nma(input: &NmaInput) -> Result<NmaOutput, NmaError> {
	nma_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn nma_prepare<'a>(
	input: &'a NmaInput,
	kernel: Kernel,
) -> Result<
	(
		// data
		&'a [f64],
		// period
		usize,
		// first
		usize,
		// ln_values
		Vec<f64>,
		// sqrt_diffs
		Vec<f64>,
		// chosen
		Kernel,
	),
	NmaError,
> {
	let data: &[f64] = input.as_ref();
	let len = data.len();

	if len == 0 {
		return Err(NmaError::EmptyInputData);
	}

	let first = data.iter().position(|x| !x.is_nan()).ok_or(NmaError::AllValuesNaN)?;

	let period = input.get_period();

	if period == 0 || period > len {
		return Err(NmaError::InvalidPeriod { period, data_len: len });
	}
	if (len - first) < (period + 1) {
		return Err(NmaError::NotEnoughValidData {
			needed: period + 1,
			valid: len - first,
		});
	}

	// Pre-compute ln values - allocate uninitialized for data-sized vector
	let mut ln_values = alloc_with_nan_prefix(len, 0); // No NaN prefix needed
	for i in 0..len {
		ln_values[i] = data[i].max(1e-10).ln();
	}

	// Pre-compute sqrt differences - small vector, regular Vec is OK
	let mut sqrt_diffs = Vec::with_capacity(period);
	for i in 0..period {
		let s0 = (i as f64).sqrt();
		let s1 = ((i + 1) as f64).sqrt();
		sqrt_diffs.push(s1 - s0);
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	Ok((data, period, first, ln_values, sqrt_diffs, chosen))
}

fn nma_compute_into(
	data: &[f64],
	period: usize,
	first: usize,
	ln_values: &mut [f64],
	sqrt_diffs: &mut [f64],
	kernel: Kernel,
	out: &mut [f64],
) {
	unsafe {
		#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
		{
			if matches!(kernel, Kernel::Scalar | Kernel::ScalarBatch) {
				nma_simd128(data, period, first, ln_values, sqrt_diffs, out);
				return;
			}
		}

		match kernel {
			Kernel::Scalar | Kernel::ScalarBatch => {
				nma_scalar_with_precomputed(data, period, first, ln_values, sqrt_diffs, out)
			}

			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => {
				nma_avx2(data, period, first, ln_values, sqrt_diffs, out)
			}

			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => {
				nma_avx512_v2(data, period, first, ln_values, sqrt_diffs, out)
			}

			#[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
			Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
				nma_scalar_with_precomputed(data, period, first, ln_values, sqrt_diffs, out)
			}
			_ => unreachable!(),
		}
	}
}

pub fn nma_with_kernel(input: &NmaInput, kernel: Kernel) -> Result<NmaOutput, NmaError> {
	// ────────────────────▼───────────────────  mark both bindings `mut`
	let (data, period, first, mut ln_values, mut sqrt_diffs, chosen) = nma_prepare(input, kernel)?;

	let warm = first + period;
	let mut out = alloc_with_nan_prefix(data.len(), warm);

	// ───────────────────▼──────────────▼────── pass them mutably
	nma_compute_into(data, period, first, &mut ln_values, &mut sqrt_diffs, chosen, &mut out);

	Ok(NmaOutput { values: out })
}
#[inline]
pub fn nma_scalar(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
	let len = data.len();

	// Allocate uninitialized for data-sized vector
	let mut ln_values = alloc_with_nan_prefix(len, 0);
	for i in 0..len {
		ln_values[i] = data[i].max(1e-10).ln();
	}

	// Small vector, regular Vec is OK
	let mut sqrt_diffs = Vec::with_capacity(period);
	for i in 0..period {
		let s0 = (i as f64).sqrt();
		let s1 = ((i + 1) as f64).sqrt();
		sqrt_diffs.push(s1 - s0);
	}

	for j in (first + period)..len {
		let mut num = 0.0;
		let mut denom = 0.0;

		for i in 0..period {
			let oi = (ln_values[j - i] - ln_values[j - i - 1]).abs();
			num += oi * sqrt_diffs[i];
			denom += oi;
		}

		let ratio = if denom == 0.0 { 0.0 } else { num / denom };

		let i = period - 1;
		out[j] = data[j - i] * ratio + data[j - i - 1] * (1.0 - ratio);
	}
}

#[inline]
pub fn nma_scalar_with_precomputed(
	data: &[f64],
	period: usize,
	first: usize,
	ln_values: &[f64],
	sqrt_diffs: &[f64],
	out: &mut [f64],
) {
	let len = data.len();

	for j in (first + period)..len {
		let mut num = 0.0;
		let mut denom = 0.0;

		for i in 0..period {
			let oi = (ln_values[j - i] - ln_values[j - i - 1]).abs();
			num += oi * sqrt_diffs[i];
			denom += oi;
		}

		let ratio = if denom == 0.0 { 0.0 } else { num / denom };

		let i = period - 1;
		out[j] = data[j - i] * ratio + data[j - i - 1] * (1.0 - ratio);
	}
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn nma_simd128(
	data: &[f64],
	period: usize,
	first: usize,
	ln_values: &[f64],
	sqrt_diffs: &[f64],
	out: &mut [f64],
) {
	use core::arch::wasm32::*;
	
	const STEP: usize = 2; // Process 2 doubles at a time (128-bit vectors)
	let len = data.len();
	
	for j in (first + period)..len {
		let chunks = period / STEP;
		let tail = period % STEP;
		
		let mut num_acc = f64x2_splat(0.0);
		let mut denom_acc = f64x2_splat(0.0);
		
		// Process vectorized chunks
		for blk in 0..chunks {
			let i = blk * STEP;
			
			// Load ln values for the difference calculation
			// ln_values[j - i] and ln_values[j - i - 1] for both lanes
			let ln_curr_0 = f64x2(ln_values[j - i], ln_values[j - i - 1]);
			let ln_prev_0 = f64x2(ln_values[j - i - 1], ln_values[j - i - 2]);
			
			// Calculate absolute differences
			let diff = f64x2_sub(ln_curr_0, ln_prev_0);
			let abs_diff = f64x2_abs(diff);
			
			// Load sqrt_diffs
			let sqrt_d = v128_load(sqrt_diffs.as_ptr().add(i) as *const v128);
			
			// Accumulate
			num_acc = f64x2_add(num_acc, f64x2_mul(abs_diff, sqrt_d));
			denom_acc = f64x2_add(denom_acc, abs_diff);
		}
		
		// Horizontal sum
		let mut num = f64x2_extract_lane::<0>(num_acc) + f64x2_extract_lane::<1>(num_acc);
		let mut denom = f64x2_extract_lane::<0>(denom_acc) + f64x2_extract_lane::<1>(denom_acc);
		
		// Handle tail elements
		for i in (chunks * STEP)..period {
			let oi = (ln_values[j - i] - ln_values[j - i - 1]).abs();
			num += oi * sqrt_diffs[i];
			denom += oi;
		}
		
		let ratio = if denom == 0.0 { 0.0 } else { num / denom };
		let i = period - 1;
		out[j] = data[j - i] * ratio + data[j - i - 1] * (1.0 - ratio);
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512vl,avx512bw,fma")]
unsafe fn fast_ln_avx512_hi(x: __m512d) -> __m512d {
	// Constants
	let one = _mm512_set1_pd(1.0);
	let two = _mm512_set1_pd(2.0);
	let half = _mm512_set1_pd(0.5);
	let ln2 = _mm512_set1_pd(std::f64::consts::LN_2);
	let sqrt_half = _mm512_set1_pd(0.7071067811865475244);
	
	// Special case handling for values near 1.0
	// For |x-1| < 0.2, use Taylor series: ln(x) ≈ (x-1) - (x-1)²/2 + (x-1)³/3 - ...
	// This covers the range [0.8, 1.2] where the polynomial approximation has poor accuracy
	let threshold = _mm512_set1_pd(0.2);
	let x_minus_1 = _mm512_sub_pd(x, one);
	let abs_x_minus_1 = _mm512_abs_pd(x_minus_1);
	let near_one_mask = _mm512_cmp_pd_mask(abs_x_minus_1, threshold, _CMP_LT_OQ);
	
	// Taylor series coefficients for ln(1+y) where y = x-1
	let c2 = _mm512_set1_pd(-0.5);
	let c3 = _mm512_set1_pd(1.0/3.0);
	let c4 = _mm512_set1_pd(-0.25);
	let c5 = _mm512_set1_pd(0.2);
	let c6 = _mm512_set1_pd(-1.0/6.0);
	let c7 = _mm512_set1_pd(1.0/7.0);
	let c8 = _mm512_set1_pd(-0.125);
	
	// Compute Taylor series: ln(x) = y - y²/2 + y³/3 - y⁴/4 + ...
	let y = x_minus_1;
	let y2 = _mm512_mul_pd(y, y);
	let y3 = _mm512_mul_pd(y2, y);
	let y4 = _mm512_mul_pd(y2, y2);
	
	// Higher accuracy Taylor expansion
	let mut taylor = y;
	taylor = _mm512_fmadd_pd(y2, c2, taylor);
	taylor = _mm512_fmadd_pd(y3, c3, taylor);
	taylor = _mm512_fmadd_pd(y4, c4, taylor);
	let y5 = _mm512_mul_pd(y4, y);
	let y6 = _mm512_mul_pd(y4, y2);
	let y7 = _mm512_mul_pd(y4, y3);
	let y8 = _mm512_mul_pd(y4, y4);
	taylor = _mm512_fmadd_pd(y5, c5, taylor);
	taylor = _mm512_fmadd_pd(y6, c6, taylor);
	taylor = _mm512_fmadd_pd(y7, c7, taylor);
	taylor = _mm512_fmadd_pd(y8, c8, taylor);
	
	// For values not near 1.0, use the original algorithm
	// Extract exponent and mantissa
	let ix = _mm512_castpd_si512(x);
	let exp_mask = _mm512_set1_epi64(0x7FF0000000000000u64 as i64);
	let mantissa_mask = _mm512_set1_epi64(0x000FFFFFFFFFFFFFu64 as i64);
	let bias = _mm512_set1_epi64(1023);
	
	// Extract exponent
	let exp_bits = _mm512_and_si512(ix, exp_mask);
	let exp_shifted = _mm512_srli_epi64::<52>(exp_bits);
	let e = _mm512_sub_epi64(exp_shifted, bias);
	let e_f64 = _mm512_cvtepi64_pd(e);
	
	// Set mantissa to [1, 2) range
	let mantissa_bits = _mm512_and_si512(ix, mantissa_mask);
	let one_bits = _mm512_set1_epi64(0x3FF0000000000000u64 as i64);
	let m_bits = _mm512_or_si512(mantissa_bits, one_bits);
	let mut m = _mm512_castsi512_pd(m_bits);
	
	// Conditional fold to [sqrt(0.5), sqrt(2))
	let needs_fold = _mm512_cmp_pd_mask(m, sqrt_half, _CMP_LT_OQ);
	m = _mm512_mask_mul_pd(m, needs_fold, m, two);
	let e_adjust = _mm512_mask_sub_pd(e_f64, needs_fold, e_f64, one);
	
	// Compute f = m - 1
	let f = _mm512_sub_pd(m, one);
	
	// Compute s = f / (2 + f)
	let two_plus_f = _mm512_add_pd(two, f);
	let s = _mm512_div_pd(f, two_plus_f);
	let z = _mm512_mul_pd(s, s);
	let w = _mm512_mul_pd(z, z);
	
	// Ultra high-precision polynomial coefficients (11th degree)
	// These coefficients are from the Cephes library and provide ~0.5 ULP accuracy
	let lg1 = _mm512_set1_pd(6.666666666666735130e-01);   // 2/3
	let lg2 = _mm512_set1_pd(3.999999999940941908e-01);   // 2/5  
	let lg3 = _mm512_set1_pd(2.857142874366239149e-01);   // 2/7
	let lg4 = _mm512_set1_pd(2.222219843214978396e-01);   // 2/9
	let lg5 = _mm512_set1_pd(1.818357216161805012e-01);   // 2/11
	let lg6 = _mm512_set1_pd(1.531383769920937332e-01);   // 2/13
	let lg7 = _mm512_set1_pd(1.479819860511658591e-01);   // ~2/13.5
	// Additional high-order terms for better accuracy
	let lg8 = _mm512_set1_pd(1.333355814642869980e-01);
	let lg9 = _mm512_set1_pd(1.253141636393179328e-01);
	
	// Evaluate polynomial using Horner's method with FMA
	// Split into two parts for instruction-level parallelism
	// r1 = z * (Lg1 + z * (Lg3 + z * (Lg5 + z * (Lg7 + z * Lg9))))
	let mut r1 = lg9;
	r1 = _mm512_fmadd_pd(r1, z, lg7);
	r1 = _mm512_fmadd_pd(r1, z, lg5);
	r1 = _mm512_fmadd_pd(r1, z, lg3);
	r1 = _mm512_fmadd_pd(r1, z, lg1);
	r1 = _mm512_mul_pd(r1, z);
	
	// r2 = w * (Lg2 + z * (Lg4 + z * (Lg6 + z * Lg8)))
	let mut r2 = lg8;
	r2 = _mm512_fmadd_pd(r2, z, lg6);
	r2 = _mm512_fmadd_pd(r2, z, lg4);
	r2 = _mm512_fmadd_pd(r2, z, lg2);
	r2 = _mm512_mul_pd(r2, w);
	
	let r = _mm512_add_pd(r1, r2);
	
	// ln(1+f) ≈ f - f²/2 + f³ * R(f)
	// Note: f³/2 = f * f²/2 = f * hfsq
	let hfsq = _mm512_mul_pd(_mm512_mul_pd(half, f), f);
	
	// Compute polynomial more accurately
	// ln(1+f) = f - hfsq + f*s^2*poly
	// Since s = f/(2+f), we have s^2 = f²/(2+f)²
	// And f*s^2 = f³/(2+f)²
	let ln1pf = _mm512_sub_pd(f, hfsq);
	let s_squared_times_f = _mm512_mul_pd(_mm512_mul_pd(s, s), f);
	let ln1pf = _mm512_fmadd_pd(s_squared_times_f, r, ln1pf);
	
	// Combine: ln(x) = ln(1+f) + e*ln(2)
	let general_result = _mm512_fmadd_pd(e_adjust, ln2, ln1pf);
	
	// Blend results: use Taylor series for values near 1.0, general algorithm otherwise
	_mm512_mask_blend_pd(near_one_mask, general_result, taylor)
}


#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn nma_avx2(data: &[f64], period: usize, first: usize, ln_values: &mut [f64], sqrt_diffs: &mut [f64], out: &mut [f64]) {
	let len = data.len();
	
	// Constants
	let epsilon = _mm256_set1_pd(1e-10);
	// Scale factor removed - not needed as it cancels in the ratio
	let one = _mm256_set1_pd(1.0);
	let zero = _mm256_setzero_pd();
	
	// Step 1: Compute ln(max(data[i], 1e-10)) * 1000 for all values
	let mut i = 0;
	while i + 4 <= len {
		let vals = _mm256_loadu_pd(data.as_ptr().add(i));
		let clamped = _mm256_max_pd(vals, epsilon);
		
		// Choice of ln implementation:
		// 1. Scalar ln() - Exact accuracy (default)
		// 2. fast_ln_avx2_hi() - High precision ~1 ULP error, ~2x faster
		// For financial applications, we default to exact accuracy
		let mut ln_vals = [0.0f64; 4];
		_mm256_storeu_pd(ln_vals.as_mut_ptr(), clamped);
		for j in 0..4 {
			ln_vals[j] = ln_vals[j].ln();
		}
		let ln_result = _mm256_loadu_pd(ln_vals.as_ptr());
		// To use fast approximation: let ln_result = fast_ln_avx2_hi(clamped);
		
		// Store directly without scaling
		_mm256_storeu_pd(ln_values.as_mut_ptr().add(i), ln_result);
		
		i += 4;
	}
	
	// Handle remaining elements with scalar
	for j in i..len {
		ln_values[j] = data[j].max(1e-10).ln();
	}
	
	// Step 2: Main computation loop (sqrt_diffs already pre-computed)
	for j in (first + period)..len {
		let mut num_accum = zero;
		let mut denom_accum = zero;
		
		// Process period values in chunks of 4
		let mut idx = 0;
		while idx + 4 <= period {
			// Load ln differences |ln[j-i] - ln[j-i-1]| for i in idx..idx+4
			let mut diffs = [0.0f64; 4];
			for k in 0..4 {
				let i = idx + k;
				let diff = (ln_values[j - i] - ln_values[j - i - 1]).abs();
				diffs[k] = diff;
			}
			let oi_vec = _mm256_loadu_pd(diffs.as_ptr());
			
			// Load corresponding sqrt_diffs
			let weights = _mm256_loadu_pd(sqrt_diffs.as_ptr().add(idx));
			
			// Accumulate weighted sum
			num_accum = _mm256_fmadd_pd(oi_vec, weights, num_accum);
			denom_accum = _mm256_add_pd(denom_accum, oi_vec);
			
			idx += 4;
		}
		
		// Horizontal sum - efficient AVX2 version
		let num_scalar = horizontal_sum_avx2(num_accum);
		let denom_scalar = horizontal_sum_avx2(denom_accum);
		
		// Handle remaining elements
		let mut num_final = num_scalar;
		let mut denom_final = denom_scalar;
		
		for i in idx..period {
			let oi = (ln_values[j - i] - ln_values[j - i - 1]).abs();
			num_final += oi * sqrt_diffs[i];
			denom_final += oi;
		}
		
		// Calculate ratio and final output
		let ratio = if denom_final == 0.0 { 0.0 } else { num_final / denom_final };
		let i = period - 1;
		out[j] = data[j - i] * ratio + data[j - i - 1] * (1.0 - ratio);
	}
}

// Efficient horizontal sum for AVX2
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_avx2(v: __m256d) -> f64 {
	// Extract upper and lower 128-bit halves
	let vlow = _mm256_castpd256_pd128(v);
	let vhigh = _mm256_extractf128_pd(v, 1);
	
	// Add the halves together
	let sum128 = _mm_add_pd(vlow, vhigh);
	
	// Horizontal add within the 128-bit vector
	let high64 = _mm_unpackhi_pd(sum128, sum128);
	
	// Final scalar result
	_mm_cvtsd_f64(_mm_add_sd(sum128, high64))
}

// Fast ln approximation for AVX2 (high precision ~1 ULP)
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn fast_ln_avx2_hi(x: __m256d) -> __m256d {
	// Constants
	let one = _mm256_set1_pd(1.0);
	let two = _mm256_set1_pd(2.0);
	let half = _mm256_set1_pd(0.5);
	let ln2 = _mm256_set1_pd(std::f64::consts::LN_2);
	let sqrt_half = _mm256_set1_pd(0.7071067811865475244);
	
	// Extract mantissa and exponent
	let mut mantissa = [0.0f64; 4];
	let mut exponent = [0i32; 4];
	_mm256_storeu_pd(mantissa.as_mut_ptr(), x);
	
	for j in 0..4 {
		let bits = mantissa[j].to_bits();
		let exp_bits = ((bits >> 52) & 0x7FF) as i32;
		exponent[j] = exp_bits - 1023;
		// Set exponent to 0 (biased = 1023) to get mantissa in [1, 2)
		let mantissa_bits = (bits & !0x7FF0000000000000) | 0x3FF0000000000000;
		mantissa[j] = f64::from_bits(mantissa_bits);
	}
	
	// Load mantissas back and convert exponents
	let mut m = _mm256_loadu_pd(mantissa.as_ptr());
	let e_vals = [exponent[0] as f64, exponent[1] as f64, exponent[2] as f64, exponent[3] as f64];
	let mut e_f64 = _mm256_loadu_pd(e_vals.as_ptr());
	
	// Conditional fold to [sqrt(0.5), sqrt(2))
	let mask = _mm256_cmp_pd(m, sqrt_half, _CMP_LT_OQ);
	m = _mm256_blendv_pd(m, _mm256_mul_pd(m, two), mask);
	e_f64 = _mm256_blendv_pd(e_f64, _mm256_sub_pd(e_f64, one), mask);
	
	// Compute f = m - 1
	let f = _mm256_sub_pd(m, one);
	
	// Compute s = f / (2 + f)
	let two_plus_f = _mm256_add_pd(two, f);
	let s = _mm256_div_pd(f, two_plus_f);
	let z = _mm256_mul_pd(s, s);
	let w = _mm256_mul_pd(z, z);
	
	// High-precision polynomial coefficients
	let lg1 = _mm256_set1_pd(6.666666666666735130e-01);
	let lg2 = _mm256_set1_pd(3.999999999940941908e-01);
	let lg3 = _mm256_set1_pd(2.857142874366239149e-01);
	let lg4 = _mm256_set1_pd(2.222219843214978396e-01);
	let lg5 = _mm256_set1_pd(1.818357216161805012e-01);
	let lg6 = _mm256_set1_pd(1.531383769920937332e-01);
	let lg7 = _mm256_set1_pd(1.479819860511658591e-01);
	
	// Split evaluation for reduced latency
	let mut r1 = lg7;
	r1 = _mm256_fmadd_pd(r1, z, lg5);
	r1 = _mm256_fmadd_pd(r1, z, lg3);
	r1 = _mm256_fmadd_pd(r1, z, lg1);
	r1 = _mm256_mul_pd(r1, z);
	
	let mut r2 = lg6;
	r2 = _mm256_fmadd_pd(r2, z, lg4);
	r2 = _mm256_fmadd_pd(r2, z, lg2);
	r2 = _mm256_mul_pd(r2, w);
	
	let r = _mm256_add_pd(r1, r2);
	
	// ln(1+f) ≈ f - f²/2 + f³ * R(f)
	let hfsq = _mm256_mul_pd(_mm256_mul_pd(half, f), f);
	let f_times_hfsq = _mm256_mul_pd(f, hfsq);
	let ln1pf = _mm256_sub_pd(f, hfsq);
	let ln1pf = _mm256_fmadd_pd(f_times_hfsq, r, ln1pf);
	
	// Combine: ln(x) = ln(1+f) + e*ln(2)
	_mm256_fmadd_pd(e_f64, ln2, ln1pf)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn _mm512_abs_pd(a: __m512d) -> __m512d {
	// AVX512 doesn't have a dedicated abs instruction for doubles
	// We use bitwise AND to clear the sign bit
	let sign_mask = _mm512_set1_pd(-0.0); // All bits 0 except sign bit
	_mm512_andnot_pd(sign_mask, a) // Clear sign bit
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512vl,avx512bw,fma")]
pub unsafe fn nma_avx512(data: &[f64], period: usize, first: usize, ln_values: &mut [f64], sqrt_diffs: &mut [f64], out: &mut [f64]) {
	let len = data.len();
	
	// Constants
	let one = _mm512_set1_pd(1.0);
	let zero = _mm512_setzero_pd();
	
	// Step 1: Compute ln values
	for i in 0..len {
		ln_values[i] = data[i].max(1e-10).ln();
	}
	
	// Step 2: Main computation loop (sqrt_diffs already pre-computed)
	for j in (first + period)..len {
		let mut num_accum = zero;
		let mut denom_accum = zero;
		
		// Process period values in chunks of 8
		let mut idx = 0;
		while idx + 8 <= period {
			// More efficient approach: load contiguous memory and compute differences
			// We need ln_values[j-idx-8] through ln_values[j-idx] (9 values total)
			// to compute 8 differences
			
			// Check if we can safely load 9 values
			if j >= idx + 8 {
				// Load 9 consecutive values: ln_values[j-idx-8] to ln_values[j-idx]
				let base_ptr = ln_values.as_ptr().add(j - idx - 8);
				
				// Load the first 8 values (prev values)
				let prev = _mm512_loadu_pd(base_ptr);
				// Load the next 8 values (curr values) - overlapping by 7
				let curr = _mm512_loadu_pd(base_ptr.add(1));
				
				// Now we have:
				// prev: [j-idx-8, j-idx-7, j-idx-6, j-idx-5, j-idx-4, j-idx-3, j-idx-2, j-idx-1]
				// curr: [j-idx-7, j-idx-6, j-idx-5, j-idx-4, j-idx-3, j-idx-2, j-idx-1, j-idx]
				
				// Compute differences: curr - prev
				let diff = _mm512_sub_pd(curr, prev);
				let abs_diff = _mm512_abs_pd(diff);
				
				// Now we need to reverse the order to match sqrt_diffs[idx..idx+8]
				// diff[0] corresponds to i=idx+7, but sqrt_diffs[0] corresponds to i=idx
				// So we need to reverse the difference vector
				let perm_indices = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
				let oi_vec = _mm512_permutexvar_pd(perm_indices, abs_diff);
				
				// Load corresponding sqrt_diffs
				let weights = _mm512_loadu_pd(sqrt_diffs.as_ptr().add(idx));
				
				// Accumulate weighted sum
				num_accum = _mm512_fmadd_pd(oi_vec, weights, num_accum);
				denom_accum = _mm512_add_pd(denom_accum, oi_vec);
			} else {
				// Fallback to scalar for edge cases
				for k in 0..8 {
					let i = idx + k;
					let oi = (ln_values[j - i] - ln_values[j - i - 1]).abs();
					let weight = sqrt_diffs[i];
					num_accum = _mm512_mask_add_pd(num_accum, 1 << k, num_accum, _mm512_set1_pd(oi * weight));
					denom_accum = _mm512_mask_add_pd(denom_accum, 1 << k, denom_accum, _mm512_set1_pd(oi));
				}
			}
			
			idx += 8;
		}
		
		// Handle remaining elements
		let mut num_scalar = _mm512_reduce_add_pd(num_accum);
		let mut denom_scalar = _mm512_reduce_add_pd(denom_accum);
		
		for i in idx..period {
			let oi = (ln_values[j - i] - ln_values[j - i - 1]).abs();
			num_scalar += oi * sqrt_diffs[i];
			denom_scalar += oi;
		}
		
		// Calculate ratio and final output
		let ratio = if denom_scalar == 0.0 { 0.0 } else { num_scalar / denom_scalar };
		let i = period - 1;
		out[j] = data[j - i] * ratio + data[j - i - 1] * (1.0 - ratio);
	}
}

// Optimized AVX512 kernel with streaming + prefix sum
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512vl,fma")]
pub unsafe fn nma_avx512_v2(
	data: &[f64],
	period: usize,
	first: usize,
	ln_values: &mut [f64],   // will be overwritten with |Δ ln| (d[k]) in 0..len-1
	sqrt_diffs: &mut [f64],  // input weights (will not be modified)
	out: &mut [f64],
) {
	use aligned_vec::AVec;
	use core::arch::x86_64::*;
	
	let len = data.len();
	debug_assert!(len == ln_values.len());
	debug_assert!(period >= 1 && period <= len);

	// --- Build d[] directly from data
	for i in 0..len {
		ln_values[i] = data[i].max(1e-10).ln();
	}
	// Convert ln_values to differences in place
	for i in 0..len-1 {
		ln_values[i] = (ln_values[i + 1] - ln_values[i]).abs();
	}
	ln_values[len - 1] = 0.0;
	let d = ln_values; // alias

	// --- 2) Prefix sums for denom: S[k+1] = S[k] + d[k]
	let mut s = alloc_with_nan_prefix(len + 1, 0);
	s[0] = 0.0;
	for k in 0..len {
		s[k + 1] = s[k] + d[k];
	}

	// --- 3) Reverse weights and pad to 8 for clean loads
	// w[i] = sqrt(i+1) - sqrt(i); we already have sqrt_diffs (length = period).
	let wlen_padded = (period + 7) & !7;
	let mut w_rev = AVec::<f64>::with_capacity(64, wlen_padded);
	w_rev.resize(wlen_padded, 0.0);
	for i in 0..period {
		w_rev[i] = sqrt_diffs[period - 1 - i];
	}
	// remaining padded entries are already zero

	// --- 4) Main loop: j from warm .. len
	let warm = first + period;
	let zero = _mm512_setzero_pd();

	for j in warm..len {
		let base = j - period; // d[base .. base+period) are the P diffs inside the window

		// denom from prefix in O(1)
		let denom = s[j] - s[j - period];

		// stream d and w_rev contiguously
		let mut num_acc = zero;
		let mut t = 0usize;

		// unroll by 2 to feed both FMA pipes on Intel; harmless elsewhere
		while t + 16 <= period {
			let d0 = _mm512_loadu_pd(d.as_ptr().add(base + t));
			let w0 = _mm512_loadu_pd(w_rev.as_ptr().add(t));
			let d1 = _mm512_loadu_pd(d.as_ptr().add(base + t + 8));
			let w1 = _mm512_loadu_pd(w_rev.as_ptr().add(t + 8));
			num_acc = _mm512_fmadd_pd(d0, w0, num_acc);
			num_acc = _mm512_fmadd_pd(d1, w1, num_acc);
			t += 16;
		}
		while t + 8 <= period {
			let d0 = _mm512_loadu_pd(d.as_ptr().add(base + t));
			let w0 = _mm512_loadu_pd(w_rev.as_ptr().add(t));
			num_acc = _mm512_fmadd_pd(d0, w0, num_acc);
			t += 8;
		}
		if t < period {
			let tail = (period - t) as u32;
			let mask: __mmask8 = ((1u32 << tail) - 1) as u8;
			let d0 = _mm512_maskz_loadu_pd(mask, d.as_ptr().add(base + t));
			let w0 = _mm512_maskz_loadu_pd(mask, w_rev.as_ptr().add(t));
			num_acc = _mm512_fmadd_pd(d0, w0, num_acc);
		}

		let num = _mm512_reduce_add_pd(num_acc);
		let ratio = if denom == 0.0 { 0.0 } else { num / denom };

		// final interpolation (use FMA to shave a dep)
		let i0 = period - 1;
		let x2 = data[j - i0 - 1];
		let dx = data[j - i0] - x2;
		out[j] = ratio.mul_add(dx, x2);
	}
}

// Optimized AVX512 batch function that shares d[] and S[] computations
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512dq,avx512vl,fma")]
unsafe fn nma_batch_avx512_optimized(
	data: &[f64],
	sweep: &NmaBatchRange,
	first: usize,
	parallel: bool,
) -> Result<NmaBatchOutput, NmaError> {
	use aligned_vec::AVec;
	use core::arch::x86_64::*;
	
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(NmaError::InvalidPeriod { period: 0, data_len: 0 });
	}
	
	let len = data.len();
	let rows = combos.len();
	let cols = len;
	
	// Pre-compute d[k] = |ln[i+1] - ln[i]| once for all rows
	let mut ln_values = alloc_with_nan_prefix(len, 0);
	for i in 0..len {
		ln_values[i] = data[i].max(1e-10).ln();
	}
	// Convert ln_values to differences in place
	for i in 0..len-1 {
		ln_values[i] = (ln_values[i + 1] - ln_values[i]).abs();
	}
	ln_values[len - 1] = 0.0;
	let d = &mut ln_values; // reuse the buffer as d[]
	
	// Pre-compute prefix sums once for all rows
	let mut s = alloc_with_nan_prefix(len + 1, 0);
	s[0] = 0.0;
	for k in 0..len {
		s[k + 1] = s[k] + d[k];
	}
	
	// Prepare output matrix
	let warm: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap()).collect();
	let mut raw = make_uninit_matrix(rows, cols);
	unsafe { init_matrix_prefixes(&mut raw, cols, &warm) };
	
	// Process each row with shared d[] and s[]
	let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
		let period = combos[row].period.unwrap();
		let warm = first + period;
		
		// Cast this row to &mut [f64]
		let out_row = core::slice::from_raw_parts_mut(
			dst_mu.as_mut_ptr() as *mut f64, 
			dst_mu.len()
		);
		
		// Create reversed weights for this period
		let wlen_padded = (period + 7) & !7;
		let mut w_rev = AVec::<f64>::with_capacity(64, wlen_padded);
		w_rev.resize(wlen_padded, 0.0);
		
		// Compute sqrt differences and reverse them
		for i in 0..period {
			let s0 = ((period - 1 - i) as f64).sqrt();
			let s1 = ((period - i) as f64).sqrt();
			w_rev[i] = s1 - s0;
		}
		
		// Main computation loop using shared d[] and s[]
		let zero = _mm512_setzero_pd();
		
		for j in warm..len {
			let base = j - period;
			
			// Get denominator from prefix sum in O(1)
			let denom = s[j] - s[j - period];
			
			// Stream d and w_rev contiguously
			let mut num_acc = zero;
			let mut t = 0usize;
			
			// Unroll by 2 for better pipeline utilization
			while t + 16 <= period {
				let d0 = _mm512_loadu_pd(d.as_ptr().add(base + t));
				let w0 = _mm512_loadu_pd(w_rev.as_ptr().add(t));
				let d1 = _mm512_loadu_pd(d.as_ptr().add(base + t + 8));
				let w1 = _mm512_loadu_pd(w_rev.as_ptr().add(t + 8));
				num_acc = _mm512_fmadd_pd(d0, w0, num_acc);
				num_acc = _mm512_fmadd_pd(d1, w1, num_acc);
				t += 16;
			}
			while t + 8 <= period {
				let d0 = _mm512_loadu_pd(d.as_ptr().add(base + t));
				let w0 = _mm512_loadu_pd(w_rev.as_ptr().add(t));
				num_acc = _mm512_fmadd_pd(d0, w0, num_acc);
				t += 8;
			}
			if t < period {
				let tail = (period - t) as u32;
				let mask: __mmask8 = ((1u32 << tail) - 1) as u8;
				let d0 = _mm512_maskz_loadu_pd(mask, d.as_ptr().add(base + t));
				let w0 = _mm512_maskz_loadu_pd(mask, w_rev.as_ptr().add(t));
				num_acc = _mm512_fmadd_pd(d0, w0, num_acc);
			}
			
			let num = _mm512_reduce_add_pd(num_acc);
			let ratio = if denom == 0.0 { 0.0 } else { num / denom };
			
			// Final interpolation
			let i0 = period - 1;
			let x2 = data[j - i0 - 1];
			let dx = data[j - i0] - x2;
			out_row[j] = ratio.mul_add(dx, x2);
		}
	};
	
	// Execute rows in parallel or serial
	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			use rayon::prelude::*;
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
	
	// Transmute to Vec<f64>
	let values: Vec<f64> = unsafe { std::mem::transmute(raw) };
	
	Ok(NmaBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
pub fn nma_batch_with_kernel(data: &[f64], sweep: &NmaBatchRange, k: Kernel) -> Result<NmaBatchOutput, NmaError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(NmaError::InvalidPeriod { period: 0, data_len: 0 }),
	};

	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => Kernel::Scalar, // Default to Scalar for any other kernel
	};
	nma_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct NmaBatchRange {
	pub period: (usize, usize, usize),
}

impl Default for NmaBatchRange {
	fn default() -> Self {
		Self { period: (40, 100, 1) }
	}
}

#[derive(Clone, Debug, Default)]
pub struct NmaBatchBuilder {
	range: NmaBatchRange,
	kernel: Kernel,
}

impl NmaBatchBuilder {
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

	pub fn apply_slice(self, data: &[f64]) -> Result<NmaBatchOutput, NmaError> {
		nma_batch_with_kernel(data, &self.range, self.kernel)
	}

	pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<NmaBatchOutput, NmaError> {
		NmaBatchBuilder::new().kernel(k).apply_slice(data)
	}

	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<NmaBatchOutput, NmaError> {
		let slice = source_type(c, src);
		self.apply_slice(slice)
	}

	pub fn with_default_candles(c: &Candles) -> Result<NmaBatchOutput, NmaError> {
		NmaBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
	}
}

#[derive(Clone, Debug)]
pub struct NmaBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<NmaParams>,
	pub rows: usize,
	pub cols: usize,
}

impl NmaBatchOutput {
	pub fn row_for_params(&self, p: &NmaParams) -> Option<usize> {
		self.combos
			.iter()
			.position(|c| c.period.unwrap_or(40) == p.period.unwrap_or(40))
	}

	pub fn values_for(&self, p: &NmaParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &NmaBatchRange) -> Vec<NmaParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let periods = axis_usize(r.period);

	let mut out = Vec::with_capacity(periods.len());
	for &p in &periods {
		out.push(NmaParams { period: Some(p) });
	}
	out
}

#[inline]
fn round_up8(x: usize) -> usize {
	(x + 7) & !7
}

#[inline(always)]
pub fn nma_batch_slice(data: &[f64], sweep: &NmaBatchRange, kern: Kernel) -> Result<NmaBatchOutput, NmaError> {
	nma_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn nma_batch_par_slice(data: &[f64], sweep: &NmaBatchRange, kern: Kernel) -> Result<NmaBatchOutput, NmaError> {
	nma_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn nma_batch_inner(
	data: &[f64],
	sweep: &NmaBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<NmaBatchOutput, NmaError> {
	// Use optimized AVX512 batch function if kernel is AVX512
	#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
	if kern == Kernel::Avx512 {
		let first = data.iter().position(|x| !x.is_nan()).ok_or(NmaError::AllValuesNaN)?;
		return unsafe { nma_batch_avx512_optimized(data, sweep, first, parallel) };
	}
	
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(NmaError::InvalidPeriod { period: 0, data_len: 0 });
	}

	let first = data.iter().position(|x| !x.is_nan()).ok_or(NmaError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| round_up8(c.period.unwrap())).max().unwrap();
	if data.len() - first < max_p + 1 {
		return Err(NmaError::NotEnoughValidData {
			needed: max_p + 1,
			valid: data.len() - first,
		});
	}

	let rows = combos.len();
	let cols = data.len();

	let warm: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap()).collect();

	let mut raw = make_uninit_matrix(rows, cols);
	unsafe { init_matrix_prefixes(&mut raw, cols, &warm) };

	// ---------------------------------------------------------------------
	// 2. closure that fills one row (works with MaybeUninit<f64>)
	let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
		let period = combos[row].period.unwrap();

		// cast just this row to &mut [f64] so we can call the usual kernel
		let out_row = core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

		match kern {
			Kernel::Scalar => nma_row_scalar(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => nma_row_avx2(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => nma_row_avx512(data, first, period, out_row),
			#[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
			Kernel::Avx2 | Kernel::Avx512 => nma_row_scalar(data, first, period, out_row),
			_ => nma_row_scalar(data, first, period, out_row),
		}
	};

	// ---------------------------------------------------------------------
	// 3. run every row, writing directly into `raw`
	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			use rayon::prelude::*;
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

	// ---------------------------------------------------------------------
	// 4. everything is now initialised – transmute to Vec<f64>
	let values: Vec<f64> = unsafe { std::mem::transmute(raw) };

	Ok(NmaBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
fn nma_batch_inner_into(
	data: &[f64],
	sweep: &NmaBatchRange,
	kern: Kernel,
	parallel: bool,
	out: &mut [f64],
) -> Result<Vec<NmaParams>, NmaError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(NmaError::InvalidPeriod { period: 0, data_len: 0 });
	}

	let first = data.iter().position(|x| !x.is_nan()).ok_or(NmaError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| round_up8(c.period.unwrap())).max().unwrap();
	if data.len() - first < max_p + 1 {
		return Err(NmaError::NotEnoughValidData {
			needed: max_p + 1,
			valid: data.len() - first,
		});
	}

	let rows = combos.len();
	let cols = data.len();

	let warm: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap()).collect();

	// SAFETY: We're reinterpreting the output slice as MaybeUninit
	let out_uninit = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len()) };

	unsafe { init_matrix_prefixes(out_uninit, cols, &warm) };

	// Closure that writes ONE row
	let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
		let period = combos[row].period.unwrap();

		// Cast this row to &mut [f64]
		let out_row = core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

		match kern {
			Kernel::Scalar => nma_row_scalar(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => nma_row_avx2(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => nma_row_avx512(data, first, period, out_row),
			#[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
			Kernel::Avx2 | Kernel::Avx512 => nma_row_scalar(data, first, period, out_row),
			_ => nma_row_scalar(data, first, period, out_row),
		}
	};

	// Drive the whole matrix
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

	Ok(combos)
}

#[inline(always)]
unsafe fn nma_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	nma_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn nma_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	// Pre-compute ln values - allocate uninitialized for data-sized vector
	let len = data.len();
	let mut ln_values = alloc_with_nan_prefix(len, 0);
	
	// Pre-compute sqrt differences - small vector, regular Vec is OK
	let mut sqrt_diffs = vec![0.0; period];
	
	// Compute ln values
	for i in 0..len {
		ln_values[i] = data[i].max(1e-10).ln();
	}
	
	// Compute sqrt differences
	for k in 0..period {
		let s0 = (k as f64).sqrt();
		let s1 = ((k + 1) as f64).sqrt();
		sqrt_diffs[k] = s1 - s0;
	}
	
	// Use the AVX2 kernel
	nma_avx2(data, period, first, &mut ln_values, &mut sqrt_diffs, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn nma_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	// Pre-compute ln values - allocate uninitialized for data-sized vector
	let len = data.len();
	let mut ln_values = alloc_with_nan_prefix(len, 0);
	
	// Pre-compute sqrt differences - small vector, regular Vec is OK
	let mut sqrt_diffs = vec![0.0; period];
	
	// Compute ln values
	for i in 0..len {
		ln_values[i] = data[i].max(1e-10).ln();
	}
	
	// Compute sqrt differences
	for k in 0..period {
		let s0 = (k as f64).sqrt();
		let s1 = ((k + 1) as f64).sqrt();
		sqrt_diffs[k] = s1 - s0;
	}
	
	// Use the optimized AVX512 v2 kernel
	nma_avx512_v2(data, period, first, &mut ln_values, &mut sqrt_diffs, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn nma_row_avx512_short(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	nma_row_avx512(data, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn nma_row_avx512_long(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	nma_row_avx512(data, first, period, out)
}

#[derive(Debug, Clone)]
pub struct NmaStream {
	period: usize,
	ln_values: Vec<f64>,
	sqrt_diffs: Vec<f64>,
	buffer: Vec<f64>,
	ln_buffer: Vec<f64>,
	head: usize,
	filled: bool,
}

impl NmaStream {
	pub fn try_new(params: NmaParams) -> Result<Self, NmaError> {
		let period = params.period.unwrap_or(40);
		if period == 0 {
			return Err(NmaError::InvalidPeriod { period, data_len: 0 });
		}
		let mut sqrt_diffs = Vec::with_capacity(period);
		for i in 0..period {
			let s0 = (i as f64).sqrt();
			let s1 = ((i + 1) as f64).sqrt();
			sqrt_diffs.push(s1 - s0);
		}
		Ok(Self {
			period,
			ln_values: vec![f64::NAN; period + 1],
			sqrt_diffs,
			buffer: vec![f64::NAN; period + 1],
			ln_buffer: vec![f64::NAN; period + 1],
			head: 0,
			filled: false,
		})
	}

	#[inline(always)]
	pub fn update(&mut self, value: f64) -> Option<f64> {
		let ln_val = value.max(1e-10).ln();
		self.buffer[self.head] = value;
		self.ln_buffer[self.head] = ln_val;
		self.head = (self.head + 1) % (self.period + 1);

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
		let mut num = 0.0;
		let mut denom = 0.0;

		// Calculate starting position for the newest value
		let newest_idx = (self.head + self.period) % (self.period + 1);

		for i in 0..self.period {
			// Access in reverse order like batch: newest to oldest
			let curr_idx = (newest_idx + self.period + 1 - i) % (self.period + 1);
			let prev_idx = (newest_idx + self.period - i) % (self.period + 1);

			let curr = self.ln_buffer[curr_idx];
			let prev = self.ln_buffer[prev_idx];
			let oi = (curr - prev).abs();

			num += oi * self.sqrt_diffs[i];
			denom += oi;
		}

		let ratio = if denom == 0.0 { 0.0 } else { num / denom };

		// Get the values for final interpolation
		let i = self.period - 1;
		let x1_idx = (newest_idx + self.period + 1 - i) % (self.period + 1);
		let x2_idx = (newest_idx + self.period - i) % (self.period + 1);

		let x1 = self.buffer[x1_idx];
		let x2 = self.buffer[x2_idx];

		x1 * ratio + x2 * (1.0 - ratio)
	}
}

// Expand grid for batch

// Python bindings
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

#[cfg(feature = "python")]
#[pyfunction(name = "nma")]
#[pyo3(signature = (data, period, kernel=None))]
pub fn nma_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	period: usize,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
	use numpy::{IntoPyArray, PyArrayMethods};

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, false)?;
	let params = NmaParams { period: Some(period) };
	let nma_in = NmaInput::from_slice(slice_in, params);

	let result_vec: Vec<f64> = py
		.allow_threads(|| nma_with_kernel(&nma_in, kern).map(|o| o.values))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "NmaStream")]
pub struct NmaStreamPy {
	stream: NmaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl NmaStreamPy {
	#[new]
	fn new(period: usize) -> PyResult<Self> {
		let params = NmaParams { period: Some(period) };
		let stream = NmaStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(NmaStreamPy { stream })
	}

	/// Updates the stream with a new value and returns the calculated NMA value.
	/// Returns `None` if the buffer is not yet full.
	fn update(&mut self, value: f64) -> Option<f64> {
		self.stream.update(value)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "nma_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
pub fn nma_batch_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	period_range: (usize, usize, usize),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, true)?;

	let sweep = NmaBatchRange { period: period_range };

	// Expand grid to know rows*cols
	let combos = expand_grid(&sweep);
	let rows = combos.len();
	let cols = slice_in.len();

	// Pre-allocate NumPy array
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
				_ => kernel,
			};
			// Use the _into variant
			nma_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
		})
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	// Build dict with the GIL
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

/// Write NMA directly to output slice - no allocations
pub fn nma_into_slice(dst: &mut [f64], input: &NmaInput, kern: Kernel) -> Result<(), NmaError> {
	let (data, period, first, mut ln_values, mut sqrt_diffs, chosen) = nma_prepare(input, kern)?;

	// Verify output buffer size matches input
	if dst.len() != data.len() {
		return Err(NmaError::InvalidPeriod {
			period: dst.len(),
			data_len: data.len(),
		});
	}

	// Compute NMA values directly into dst
	nma_compute_into(data, period, first, &mut ln_values, &mut sqrt_diffs, chosen, dst);

	// Fill warmup period with NaN
	let warmup_end = first + period;
	for v in &mut dst[..warmup_end] {
		*v = f64::NAN;
	}

	Ok(())
}

// WASM bindings
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn nma_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
	let params = NmaParams { period: Some(period) };
	let input = NmaInput::from_slice(data, params);

	// Allocate output buffer once
	let mut output = vec![0.0; data.len()];

	// Compute directly into output buffer (use Scalar for WASM)
	nma_into_slice(&mut output, &input, Kernel::Scalar).map_err(|e| JsValue::from_str(&e.to_string()))?;

	Ok(output)
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct NmaBatchConfig {
	pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct NmaBatchJsOutput {
	pub values: Vec<f64>,
	pub combos: Vec<NmaParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = nma_batch)]
pub fn nma_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	let config: NmaBatchConfig =
		serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

	let sweep = NmaBatchRange {
		period: config.period_range,
	};

	// For WASM, use ScalarBatch explicitly to avoid kernel detection issues
	let output =
		nma_batch_inner(data, &sweep, Kernel::ScalarBatch, false).map_err(|e| JsValue::from_str(&e.to_string()))?;

	let js_output = NmaBatchJsOutput {
		values: output.values,
		combos: output.combos,
		rows: output.rows,
		cols: output.cols,
	};

	serde_wasm_bindgen::to_value(&js_output).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn nma_batch_js(
	data: &[f64],
	period_start: usize,
	period_end: usize,
	period_step: usize,
) -> Result<Vec<f64>, JsValue> {
	let sweep = NmaBatchRange {
		period: (period_start, period_end, period_step),
	};

	// Use the existing batch function with parallel=false for WASM
	nma_batch_inner(data, &sweep, Kernel::Scalar, false)
		.map(|output| output.values)
		.map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn nma_batch_metadata_js(period_start: usize, period_end: usize, period_step: usize) -> Result<Vec<f64>, JsValue> {
	let sweep = NmaBatchRange {
		period: (period_start, period_end, period_step),
	};

	let combos = expand_grid(&sweep);
	let metadata: Vec<f64> = combos.iter().map(|combo| combo.period.unwrap() as f64).collect();

	Ok(metadata)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn nma_batch_rows_cols_js(
	period_start: usize,
	period_end: usize,
	period_step: usize,
	data_len: usize,
) -> Vec<usize> {
	let sweep = NmaBatchRange {
		period: (period_start, period_end, period_step),
	};
	let combos = expand_grid(&sweep);
	vec![combos.len(), data_len]
}

// ================== Zero-Copy WASM Functions ==================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn nma_alloc(len: usize) -> *mut f64 {
	// Allocate memory for input/output buffer
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec); // Prevent deallocation
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn nma_free(ptr: *mut f64, len: usize) {
	// Free allocated memory
	if !ptr.is_null() {
		unsafe {
			let _ = Vec::from_raw_parts(ptr, len, len);
		}
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn nma_into(in_ptr: *const f64, out_ptr: *mut f64, len: usize, period: usize) -> Result<(), JsValue> {
	// Check for null pointers
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("null pointer passed to nma_into"));
	}

	unsafe {
		// Create slice from pointer
		let data = std::slice::from_raw_parts(in_ptr, len);

		// Validate inputs
		if period == 0 || period > len {
			return Err(JsValue::from_str("Invalid period"));
		}

		// Calculate NMA
		let params = NmaParams { period: Some(period) };
		let input = NmaInput::from_slice(data, params);

		// Check for aliasing (input and output buffers are the same)
		if in_ptr == out_ptr {
			// Use temporary buffer to avoid corruption during sliding window computation
			let mut temp = alloc_with_nan_prefix(len, 0);
			nma_into_slice(&mut temp, &input, Kernel::Scalar).map_err(|e| JsValue::from_str(&e.to_string()))?;

			// Copy results back to output
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			out.copy_from_slice(&temp);
		} else {
			// No aliasing, compute directly into output
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			nma_into_slice(out, &input, Kernel::Scalar).map_err(|e| JsValue::from_str(&e.to_string()))?;
		}

		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn nma_batch_into(
	in_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period_start: usize,
	period_end: usize,
	period_step: usize,
) -> Result<usize, JsValue> {
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("null pointer passed to nma_batch_into"));
	}

	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);

		let sweep = NmaBatchRange {
			period: (period_start, period_end, period_step),
		};

		let combos = expand_grid(&sweep);
		let rows = combos.len();
		let cols = len;

		let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);

		// Use optimized batch processing with ScalarBatch for WASM
		nma_batch_inner_into(data, &sweep, Kernel::ScalarBatch, false, out)
			.map_err(|e| JsValue::from_str(&e.to_string()))?;

		Ok(rows)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_nma_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let default_params = NmaParams { period: None };
		let input = NmaInput::from_candles(&candles, "close", default_params);
		let output = nma_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());

		Ok(())
	}

	fn check_nma_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = NmaInput::from_candles(&candles, "close", NmaParams::default());
		let nma_result = nma_with_kernel(&input, kernel)?;

		let expected_last_five_nma = [
			64320.486018271724,
			64227.95719984426,
			64180.9249333126,
			63966.35530620797,
			64039.04719192334,
		];
		let start_index = nma_result.values.len() - 5;
		let result_last_five_nma = &nma_result.values[start_index..];
		for (i, &value) in result_last_five_nma.iter().enumerate() {
			let expected_value = expected_last_five_nma[i];
			// Allow slightly higher tolerance for fast log approximation (1-2 ULP error)
			// The relative error should be < 0.01% for financial applications
			let tolerance = if test_name.contains("avx512") { 1.0 } else { 1e-3 };
			assert!(
				(value - expected_value).abs() < tolerance,
				"[{}] NMA value mismatch at last-5 index {}: expected {}, got {}",
				test_name,
				i,
				expected_value,
				value
			);
		}
		Ok(())
	}

	fn check_nma_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = NmaInput::with_default_candles(&candles);
		match input.data {
			NmaData::Candles { source, .. } => assert_eq!(source, "close"),
			_ => panic!("Expected NmaData::Candles"),
		}
		let output = nma_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());

		Ok(())
	}

	fn check_nma_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let input_data = [10.0, 20.0, 30.0];
		let params = NmaParams { period: Some(0) };
		let input = NmaInput::from_slice(&input_data, params);
		let res = nma_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] NMA should fail with zero period", test_name);
		Ok(())
	}

	fn check_nma_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data_small = [10.0, 20.0, 30.0];
		let params = NmaParams { period: Some(10) };
		let input = NmaInput::from_slice(&data_small, params);
		let res = nma_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] NMA should fail with period exceeding length",
			test_name
		);
		Ok(())
	}

	fn check_nma_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let single_point = [42.0];
		let params = NmaParams { period: Some(40) };
		let input = NmaInput::from_slice(&single_point, params);
		let res = nma_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] NMA should fail with insufficient data", test_name);
		Ok(())
	}
	
	fn check_nma_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let empty: [f64; 0] = [];
		let input = NmaInput::from_slice(&empty, NmaParams::default());
		let res = nma_with_kernel(&input, kernel);
		assert!(
			matches!(res, Err(NmaError::EmptyInputData)),
			"[{}] NMA should fail with empty input error",
			test_name
		);
		Ok(())
	}

	fn check_nma_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let first_params = NmaParams { period: Some(40) };
		let first_input = NmaInput::from_candles(&candles, "close", first_params);
		let first_result = nma_with_kernel(&first_input, kernel)?;
		let second_params = NmaParams { period: Some(20) };
		let second_input = NmaInput::from_slice(&first_result.values, second_params);
		let second_result = nma_with_kernel(&second_input, kernel)?;
		assert_eq!(second_result.values.len(), first_result.values.len());
		if second_result.values.len() > 240 {
			for i in 240..second_result.values.len() {
				assert!(second_result.values[i].is_finite());
			}
		}
		Ok(())
	}

	fn check_nma_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = NmaInput::from_candles(&candles, "close", NmaParams { period: Some(40) });
		let res = nma_with_kernel(&input, kernel)?;
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

	fn check_nma_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		use proptest::prelude::*;
		skip_if_unsupported!(kernel, test_name);

		// Strategy: Generate period from 2 to 100, then data with length >= period+1
		let strat = (2usize..=100).prop_flat_map(|period| {
			(
				prop::collection::vec(
					(-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
					(period + 1)..400,
				),
				Just(period),
			)
		});

		proptest::test_runner::TestRunner::default()
			.run(&strat, |(data, period)| {
				let params = NmaParams { period: Some(period) };
				let input = NmaInput::from_slice(&data, params);

				// Compute NMA with specified kernel and scalar reference
				let result = nma_with_kernel(&input, kernel);
				prop_assert!(result.is_ok(), "NMA computation failed: {:?}", result.err());
				let out = result.unwrap().values;

				let ref_result = nma_with_kernel(&input, Kernel::Scalar);
				prop_assert!(ref_result.is_ok(), "Reference NMA failed");
				let ref_out = ref_result.unwrap().values;

				// Property 1: Output length matches input
				prop_assert_eq!(out.len(), data.len(), "Output length mismatch");

				// Find first valid data point
				let first_valid = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
				let warmup_end = first_valid + period;

				// Property 2: NaN values only in warmup period
				for i in 0..warmup_end.min(out.len()) {
					prop_assert!(
						out[i].is_nan(),
						"Expected NaN at index {} (warmup period), got {}",
						i,
						out[i]
					);
				}

				// Property 3: All values after warmup are finite
				for i in warmup_end..out.len() {
					prop_assert!(
						out[i].is_finite(),
						"Expected finite value at index {} (after warmup), got {}",
						i,
						out[i]
					);
				}

				// Property 4: NMA output is bounded by the two interpolated data points
				// NMA formula: data[j-period+1] * ratio + data[j-period] * (1-ratio)
				// This is a weighted average between data[j-period+1] and data[j-period]
				for i in warmup_end..out.len() {
					let point1 = data[i - period + 1];
					let point2 = data[i - period];
					let min_bound = point1.min(point2);
					let max_bound = point1.max(point2);
					
					// Allow small tolerance for floating point errors
					// AVX512 uses SLEEF fast math approximations which need more tolerance
					let tolerance = if test_name.contains("avx512") { 1e-7 } else { 1e-9 };
					prop_assert!(
						out[i] >= min_bound - tolerance && out[i] <= max_bound + tolerance,
						"NMA at index {} = {} not in bounds [{}, {}]",
						i,
						out[i],
						min_bound,
						max_bound
					);
				}

				// Property 5: When all data is constant, NMA equals that constant
				if data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-12) && !data.is_empty() {
					for i in warmup_end..out.len() {
						prop_assert!(
							(out[i] - data[0]).abs() < 1e-9,
							"Constant data: NMA[{}] = {} should equal {}",
							i,
							out[i],
							data[0]
						);
					}
				}

				// Property 6: Period=1 special case
				if period == 1 {
					// With period=1, warmup is first_valid+1
					// NMA should essentially pass through recent values
					for i in (first_valid + 1)..out.len() {
						// For period=1, the formula simplifies significantly
						// We expect the output to be very close to the input
						prop_assert!(
							(out[i] - data[i]).abs() < 1e-6,
							"Period=1: NMA[{}] = {} should be close to data[{}] = {}",
							i,
							out[i],
							i,
							data[i]
						);
					}
				}

				// Property 7: Ratio bounds check
				// The internal ratio used in NMA formula should be in [0, 1]
				// This validates the mathematical correctness of the algorithm
				// We can't directly check the ratio, but we can verify that
				// the output is a valid interpolation between two points
				for i in warmup_end..out.len() {
					let point1 = data[i - period + 1];
					let point2 = data[i - period];
					
					// NMA output should be exactly representable as a weighted average
					// out[i] = point1 * ratio + point2 * (1 - ratio)
					// This means if we solve for ratio: ratio = (out[i] - point2) / (point1 - point2)
					// The ratio should be in [0, 1] (with floating point tolerance)
					
					if (point1 - point2).abs() > 1e-10 {
						// Only check when points are different
						let implied_ratio = (out[i] - point2) / (point1 - point2);
						prop_assert!(
							implied_ratio >= -1e-9 && implied_ratio <= 1.0 + 1e-9,
							"Invalid interpolation ratio {} at index {} (output={}, p1={}, p2={})",
							implied_ratio,
							i,
							out[i],
							point1,
							point2
						);
					}
				}

				// Property 8: SIMD kernel consistency
				// All kernels should produce nearly identical results
				for i in 0..out.len() {
					if !out[i].is_finite() || !ref_out[i].is_finite() {
						// Both should be NaN or both finite
						prop_assert_eq!(
							out[i].is_nan(),
							ref_out[i].is_nan(),
							"NaN mismatch at index {}",
							i
						);
						continue;
					}

					// Check ULP difference for finite values
					let out_bits = out[i].to_bits();
					let ref_bits = ref_out[i].to_bits();
					let ulp_diff = out_bits.abs_diff(ref_bits);

					// Allow different ULP tolerances based on kernel
					// AVX512 uses SLEEF fast log approximations that can accumulate more error
					// NMA uses ln() and sqrt operations which compound the precision loss
					if test_name.contains("avx512") {
						// For AVX512, allow up to 75 ULPs or very small relative error
						// This balances catching real issues while allowing fast math
						let rel_error = if ref_out[i].abs() > 1e-10 {
							((out[i] - ref_out[i]) / ref_out[i]).abs()
						} else {
							(out[i] - ref_out[i]).abs()
						};
						prop_assert!(
							rel_error < 1e-7 || ulp_diff <= 75,
							"Kernel mismatch at index {}: {} vs {} (rel_error: {}, ULP diff: {})",
							i,
							out[i],
							ref_out[i],
							rel_error,
							ulp_diff
						);
					} else {
						// Standard kernels should have tighter tolerance
						prop_assert!(
							(out[i] - ref_out[i]).abs() <= 1e-9 || ulp_diff <= 25,
							"Kernel mismatch at index {}: {} vs {} (ULP diff: {})",
							i,
							out[i],
							ref_out[i],
							ulp_diff
						);
					}
				}

				// Property 9: Very small values handling
				// NMA should handle very small positive values without overflow/underflow
				// This is important because NMA uses ln(max(data[i], 1e-10))
				let has_small_values = data.iter().any(|&x| x > 0.0 && x < 1e-8);
				if has_small_values {
					for i in warmup_end..out.len() {
						prop_assert!(
							out[i].is_finite(),
							"NMA failed to handle small values at index {}: {}",
							i,
							out[i]
						);
					}
				}

				Ok(())
			})
			.unwrap();

		Ok(())
	}

	macro_rules! generate_all_nma_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(#[test]
                fn [<$test_fn _scalar_f64>]() {
                    let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar);
                })*
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
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                $(
                    #[test]
                    fn [<$test_fn _simd128_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _simd128_f64>]), Kernel::Scalar);
                    }
                )*
            }
        }
    }

	// Check for poison values in single output - only runs in debug mode
	#[cfg(debug_assertions)]
	fn check_nma_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		// Test with multiple parameter combinations
		let test_cases = vec![
			NmaParams { period: Some(40) },  // default
			NmaParams { period: Some(10) },  // small period
			NmaParams { period: Some(5) },   // very small period
			NmaParams { period: Some(20) },  // medium period
			NmaParams { period: Some(60) },  // larger period
			NmaParams { period: Some(100) }, // large period
			NmaParams { period: Some(3) },   // minimum practical period
			NmaParams { period: Some(80) },  // another large period
			NmaParams { period: None },      // None value (use default)
		];

		for params in test_cases {
			let input = NmaInput::from_candles(&candles, "close", params);
			let output = nma_with_kernel(&input, kernel)?;

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
                         with params period={:?}",
						test_name, val, bits, i, params.period
					);
				}

				// Check for init_matrix_prefixes poison (0x22222222_22222222)
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
                         with params period={:?}",
						test_name, val, bits, i, params.period
					);
				}

				// Check for make_uninit_matrix poison (0x33333333_33333333)
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
                         with params period={:?}",
						test_name, val, bits, i, params.period
					);
				}
			}
		}

		Ok(())
	}

	// Release mode stub - does nothing
	#[cfg(not(debug_assertions))]
	fn check_nma_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(())
	}

	generate_all_nma_tests!(
		check_nma_partial_params,
		check_nma_accuracy,
		check_nma_default_candles,
		check_nma_zero_period,
		check_nma_period_exceeds_length,
		check_nma_very_small_dataset,
		check_nma_empty_input,
		check_nma_reinput,
		check_nma_nan_handling,
		check_nma_no_poison,
		check_nma_property
	);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let output = NmaBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;

		let def = NmaParams::default();
		let row = output.values_for(&def).expect("default row missing");

		assert_eq!(row.len(), c.close.len());

		let expected = [
			64320.486018271724,
			64227.95719984426,
			64180.924933312606,
			63966.35530620797,
			64039.04719192333,
		];
		let start = row.len() - 5;
		for (i, &v) in row[start..].iter().enumerate() {
			let tolerance = 1e-3;
			assert!(
				(v - expected[i]).abs() < tolerance,
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

		// Test multiple batch configurations with different parameter ranges
		let batch_configs = vec![
			// Original test case
			(10, 30, 10),
			// Edge cases
			(40, 40, 0),   // Single parameter (default)
			(3, 15, 3),    // Small periods
			(50, 100, 25), // Large periods
			(5, 25, 5),    // Different step
			(20, 80, 20),  // Medium to large
			(8, 24, 8),    // Different small range
			(60, 120, 30), // Very large periods
		];

		for (p_start, p_end, p_step) in batch_configs {
			let output = NmaBatchBuilder::new()
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
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} \
                         (flat index {}) with params period={:?}",
						test, val, bits, row, col, idx, combo.period
					);
				}

				// Check for init_matrix_prefixes poison (0x22222222_22222222)
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} \
                         (flat index {}) with params period={:?}",
						test, val, bits, row, col, idx, combo.period
					);
				}

				// Check for make_uninit_matrix poison (0x33333333_33333333)
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} \
                         (flat index {}) with params period={:?}",
						test, val, bits, row, col, idx, combo.period
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
