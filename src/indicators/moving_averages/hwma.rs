//! # Holt-Winters Moving Average (HWMA)
//!
//! Triple-smoothed adaptive moving average with three parameters (`na`, `nb`, `nc`).
//! Each parameter adjusts a component of smoothing: level (`na`), trend (`nb`), and acceleration (`nc`).
//! API and implementation structure matches alma.rs, including AVX stubs, kernel support, batch/grid/stream, and unit tests.
//!
//! ## Parameters
//! - **na**: Smoothing for level (0,1)
//! - **nb**: Smoothing for trend (0,1)
//! - **nc**: Smoothing for acceleration (0,1)
//!
//! ## Errors
//! - **EmptyData**: hwma: The provided data array is empty.
//! - **AllValuesNaN**: hwma: All input data values are `NaN`.
//! - **InvalidParams**: hwma: One or more of `(na, nb, nc)` are out of (0,1).
//!
//! ## Returns
//! - **`Ok(HwmaOutput)`** with results, or **`Err(HwmaError)`** on failure.

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
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

impl<'a> AsRef<[f64]> for HwmaInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			HwmaData::Slice(slice) => slice,
			HwmaData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

#[derive(Debug, Clone)]
pub enum HwmaData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct HwmaOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct HwmaParams {
	pub na: Option<f64>,
	pub nb: Option<f64>,
	pub nc: Option<f64>,
}

impl Default for HwmaParams {
	fn default() -> Self {
		Self {
			na: Some(0.2),
			nb: Some(0.1),
			nc: Some(0.1),
		}
	}
}

#[derive(Debug, Clone)]
pub struct HwmaInput<'a> {
	pub data: HwmaData<'a>,
	pub params: HwmaParams,
}

impl<'a> HwmaInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: HwmaParams) -> Self {
		Self {
			data: HwmaData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slice(sl: &'a [f64], p: HwmaParams) -> Self {
		Self {
			data: HwmaData::Slice(sl),
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", HwmaParams::default())
	}
	#[inline]
	pub fn get_na(&self) -> f64 {
		self.params.na.unwrap_or(0.2)
	}
	#[inline]
	pub fn get_nb(&self) -> f64 {
		self.params.nb.unwrap_or(0.1)
	}
	#[inline]
	pub fn get_nc(&self) -> f64 {
		self.params.nc.unwrap_or(0.1)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct HwmaBuilder {
	na: Option<f64>,
	nb: Option<f64>,
	nc: Option<f64>,
	kernel: Kernel,
}

impl Default for HwmaBuilder {
	fn default() -> Self {
		Self {
			na: None,
			nb: None,
			nc: None,
			kernel: Kernel::Auto,
		}
	}
}

impl HwmaBuilder {
	#[inline(always)]
	pub fn new() -> Self {
		Self::default()
	}
	#[inline(always)]
	pub fn na(mut self, x: f64) -> Self {
		self.na = Some(x);
		self
	}
	#[inline(always)]
	pub fn nb(mut self, x: f64) -> Self {
		self.nb = Some(x);
		self
	}
	#[inline(always)]
	pub fn nc(mut self, x: f64) -> Self {
		self.nc = Some(x);
		self
	}
	#[inline(always)]
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}

	#[inline(always)]
	pub fn apply(self, c: &Candles) -> Result<HwmaOutput, HwmaError> {
		let p = HwmaParams {
			na: self.na,
			nb: self.nb,
			nc: self.nc,
		};
		let i = HwmaInput::from_candles(c, "close", p);
		hwma_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slice(self, d: &[f64]) -> Result<HwmaOutput, HwmaError> {
		let p = HwmaParams {
			na: self.na,
			nb: self.nb,
			nc: self.nc,
		};
		let i = HwmaInput::from_slice(d, p);
		hwma_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<HwmaStream, HwmaError> {
		let p = HwmaParams {
			na: self.na,
			nb: self.nb,
			nc: self.nc,
		};
		HwmaStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum HwmaError {
	#[error("hwma: calculation received empty data array.")]
	EmptyData,
	#[error("hwma: All values in input data are NaN.")]
	AllValuesNaN,
	#[error("hwma: Parameters (na, nb, nc) must be in (0,1). Received: na={na}, nb={nb}, nc={nc}")]
	InvalidParams { na: f64, nb: f64, nc: f64 },
	#[error("hwma: Invalid output buffer size: expected = {expected}, actual = {actual}")]
	InvalidOutputBuffer { expected: usize, actual: usize },
}

#[inline]
pub fn hwma(input: &HwmaInput) -> Result<HwmaOutput, HwmaError> {
	hwma_with_kernel(input, Kernel::Auto)
}

pub fn hwma_with_kernel(input: &HwmaInput, kernel: Kernel) -> Result<HwmaOutput, HwmaError> {
	let data: &[f64] = input.as_ref();
	let len = data.len();
	if len == 0 {
		return Err(HwmaError::EmptyData);
	}
	let first = data.iter().position(|x| !x.is_nan()).ok_or(HwmaError::AllValuesNaN)?;
	let na = input.get_na();
	let nb = input.get_nb();
	let nc = input.get_nc();

	if !na.is_finite() || !nb.is_finite() || !nc.is_finite() {
		return Err(HwmaError::InvalidParams { na, nb, nc });
	}
	if !(na > 0.0 && na < 1.0 && nb > 0.0 && nb < 1.0 && nc > 0.0 && nc < 1.0) {
		return Err(HwmaError::InvalidParams { na, nb, nc });
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	let mut out = alloc_with_nan_prefix(len, first);
	unsafe {
		// For WASM, use SIMD128 when available instead of scalar
		#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
		{
			if matches!(chosen, Kernel::Scalar | Kernel::ScalarBatch) {
				hwma_simd128(data, na, nb, nc, first, &mut out);
				return Ok(HwmaOutput { values: out });
			}
		}

		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => hwma_scalar(data, na, nb, nc, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => hwma_avx2(data, na, nb, nc, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => hwma_avx512(data, na, nb, nc, first, &mut out),
			_ => hwma_scalar(data, na, nb, nc, first, &mut out),
		}
	}

	Ok(HwmaOutput { values: out })
}

/// Computes HWMA into a pre-allocated output buffer.
///
/// # Zero-Copy Design
/// This function follows a zero-copy philosophy for optimal performance.
/// The output buffer must be pre-allocated by the caller and should already
/// contain appropriate NaN values in the warmup period if needed.
///
/// # Arguments
/// * `input` - The HWMA input containing data and parameters
/// * `out` - Pre-allocated output buffer (must be same length as input data)
///
/// # Returns
/// * `Ok(())` if successful
/// * `Err(HwmaError)` if the computation fails
#[inline]
pub fn hwma_into(input: &HwmaInput, out: &mut [f64]) -> Result<(), HwmaError> {
	hwma_with_kernel_into(input, Kernel::Auto, out)
}

/// Computes HWMA into a pre-allocated output buffer with specified kernel.
///
/// # Zero-Copy Design
/// This function is designed for zero-copy operation and maximum performance.
/// Unlike the allocating version, this function does NOT initialize the warmup
/// period with NaN values. The caller is responsible for any necessary
/// initialization of the output buffer.
///
/// This design choice eliminates redundant memory operations when the buffer
/// is already properly initialized (e.g., from Python/WASM bindings).
///
/// # Arguments
/// * `input` - The HWMA input containing data and parameters
/// * `kernel` - SIMD kernel to use for computation
/// * `out` - Pre-allocated output buffer (must be same length as input data)
///
/// # Returns
/// * `Ok(())` if successful
/// * `Err(HwmaError)` if the computation fails
pub fn hwma_with_kernel_into(input: &HwmaInput, kernel: Kernel, out: &mut [f64]) -> Result<(), HwmaError> {
	let data: &[f64] = input.as_ref();
	let len = data.len();
	if len == 0 {
		return Err(HwmaError::EmptyData);
	}

	// Ensure output buffer is the correct size
	if out.len() != len {
		return Err(HwmaError::InvalidOutputBuffer {
			expected: len,
			actual: out.len(),
		});
	}

	let first = data.iter().position(|x| !x.is_nan()).ok_or(HwmaError::AllValuesNaN)?;
	let na = input.get_na();
	let nb = input.get_nb();
	let nc = input.get_nc();

	if !na.is_finite() || !nb.is_finite() || !nc.is_finite() {
		return Err(HwmaError::InvalidParams { na, nb, nc });
	}
	if !(na > 0.0 && na < 1.0 && nb > 0.0 && nb < 1.0 && nc > 0.0 && nc < 1.0) {
		return Err(HwmaError::InvalidParams { na, nb, nc });
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	// NOTE: This function expects the output buffer to be pre-initialized.
	// The caller is responsible for handling NaN prefix initialization.
	// This follows the zero-copy philosophy to avoid redundant memory operations.

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => hwma_scalar(data, na, nb, nc, first, out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => hwma_avx2(data, na, nb, nc, first, out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => hwma_avx512(data, na, nb, nc, first, out),
			_ => {
				// For WASM, use SIMD128 when available
				#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
				{
					hwma_simd128(data, na, nb, nc, first, out);
					return Ok(());
				}
				hwma_scalar(data, na, nb, nc, first, out)
			}
		}
	}

	Ok(())
}

#[inline(always)]
pub fn hwma_scalar(data: &[f64], na: f64, nb: f64, nc: f64, first_valid: usize, out: &mut [f64]) {
	debug_assert_eq!(data.len(), out.len());
	if first_valid >= data.len() {
		return;
	}

	// Pre-compute complements once.
	let one_m_na = 1.0 - na;
	let one_m_nb = 1.0 - nb;
	let one_m_nc = 1.0 - nc;

	// State registers
	let mut f = data[first_valid]; // level
	let mut v = 0.0; // velocity / trend
	let mut a = 0.0; // acceleration

	// Main pass
	for i in first_valid..data.len() {
		// SAFETY: bounds checked by loop guard.
		let price = unsafe { *data.get_unchecked(i) };

		// f = na·price + (1-na)·(f + v + 0.5 a)
		let fv_sum = f + v + 0.5 * a;
		let f_new = na.mul_add(price, one_m_na * fv_sum);

		// v = nb·(f_new - f) + (1-nb)·(v + a)
		let v_new = nb.mul_add(f_new - f, one_m_nb * (v + a));

		// a = nc·(v_new - v) + (1-nc)·a
		let a_new = nc.mul_add(v_new - v, one_m_nc * a);

		// Output HWMA = f + v + 0.5 a
		out[i] = f_new + v_new + 0.5 * a_new;

		// Roll state
		f = f_new;
		v = v_new;
		a = a_new;
	}
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn hwma_simd128(data: &[f64], na: f64, nb: f64, nc: f64, first: usize, out: &mut [f64]) {
	use core::arch::wasm32::*;

	debug_assert_eq!(data.len(), out.len());
	if first >= data.len() {
		return;
	}

	// Pre-compute complements once.
	let one_m_na = 1.0 - na;
	let one_m_nb = 1.0 - nb;
	let one_m_nc = 1.0 - nc;
	const HALF: f64 = 0.5;

	// State registers - we process 2 elements at a time with SIMD128
	let mut f = f64x2_splat(data[first]); // level
	let mut v = f64x2_splat(0.0); // velocity / trend
	let mut a = f64x2_splat(0.0); // acceleration

	// Broadcast coefficients
	let na_vec = f64x2_splat(na);
	let one_m_na_vec = f64x2_splat(one_m_na);
	let nb_vec = f64x2_splat(nb);
	let one_m_nb_vec = f64x2_splat(one_m_nb);
	let nc_vec = f64x2_splat(nc);
	let one_m_nc_vec = f64x2_splat(one_m_nc);
	let half_vec = f64x2_splat(HALF);

	// Initialize first element
	out[first] = data[first];

	// Process two elements at a time
	let mut i = first + 1;
	while i + 1 < data.len() {
		// Load two prices
		let price = v128_load(data.as_ptr().add(i) as *const v128);

		// f = na·price + (1-na)·(f + v + 0.5 a)
		let fv_sum = f64x2_add(f, f64x2_add(v, f64x2_mul(half_vec, a)));
		let f_new = f64x2_add(f64x2_mul(na_vec, price), f64x2_mul(one_m_na_vec, fv_sum));

		// v = nb·(f_new - f) + (1-nb)·(v + a)
		let v_new = f64x2_add(
			f64x2_mul(nb_vec, f64x2_sub(f_new, f)),
			f64x2_mul(one_m_nb_vec, f64x2_add(v, a)),
		);

		// a = nc·(v_new - v) + (1-nc)·a
		let a_new = f64x2_add(f64x2_mul(nc_vec, f64x2_sub(v_new, v)), f64x2_mul(one_m_nc_vec, a));

		// Output HWMA = f + v + 0.5 a
		let result = f64x2_add(f_new, f64x2_add(v_new, f64x2_mul(half_vec, a_new)));
		v128_store(out.as_mut_ptr().add(i) as *mut v128, result);

		// Roll state
		f = f_new;
		v = v_new;
		a = a_new;

		i += 2;
	}

	// Handle remaining element if any
	if i < data.len() {
		// Extract last computed state
		let f_last = f64x2_extract_lane::<1>(f);
		let v_last = f64x2_extract_lane::<1>(v);
		let a_last = f64x2_extract_lane::<1>(a);

		// Compute final element
		let price = data[i];
		let fv_sum = f_last + v_last + 0.5 * a_last;
		let f_new = na * price + one_m_na * fv_sum;
		let v_new = nb * (f_new - f_last) + one_m_nb * (v_last + a_last);
		let a_new = nc * (v_new - v_last) + one_m_nc * a_last;
		out[i] = f_new + v_new + 0.5 * a_new;
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn hwma_avx2(data: &[f64], na: f64, nb: f64, nc: f64, first: usize, out: &mut [f64]) {
	debug_assert_eq!(data.len(), out.len());
	if first >= data.len() {
		return;
	}

	// -------- coefficients (kept once) ----------------------------
	let one_m_na = 1.0 - na;
	let one_m_nb = 1.0 - nb;
	let one_m_nc = 1.0 - nc;
	const HALF: f64 = 0.5;

	// -------- state registers -------------------------------------
	let mut f = data[first]; // level
	let mut v = 0.0; // trend
	let mut a = 0.0; // acceleration

	// -------- main loop -------------------------------------------
	for i in first..data.len() {
		// SAFETY: `i` checked by loop guard.
		let price = unsafe { *data.get_unchecked(i) };

		// ---- level (f') ----
		// s = f + v + 0.5·a   computed with one FMA
		let s = HALF.mul_add(a, f + v);
		// f' = na·price + (1-na)·s
		let f_new = na.mul_add(price, one_m_na * s);

		// ---- trend (v') ----
		let diff_f = f_new - f;
		// v' = nb·(f'-f) + (1-nb)·(v + a)
		let v_new = nb.mul_add(diff_f, one_m_nb * (v + a));

		// ---- acceleration (a') ----
		let diff_v = v_new - v;
		// a' = nc·(v'-v) + (1-nc)·a
		let a_new = nc.mul_add(diff_v, one_m_nc * a);

		// ---- output HWMA = f' + v' + 0.5·a'  (one more FMA) ----
		out[i] = HALF.mul_add(a_new, f_new + v_new);

		// ---- roll state ----
		f = f_new;
		v = v_new;
		a = a_new;
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
#[inline]
pub unsafe fn hwma_avx512(data: &[f64], na: f64, nb: f64, nc: f64, first: usize, out: &mut [f64]) {
	hwma_avx2(data, na, nb, nc, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn hwma_avx512_short(data: &[f64], na: f64, nb: f64, nc: f64, first_valid: usize, out: &mut [f64]) {
	hwma_scalar(data, na, nb, nc, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn hwma_avx512_long(data: &[f64], na: f64, nb: f64, nc: f64, first_valid: usize, out: &mut [f64]) {
	hwma_scalar(data, na, nb, nc, first_valid, out)
}

#[derive(Debug, Clone)]
pub struct HwmaStream {
	na: f64,
	nb: f64,
	nc: f64,
	last_f: f64,
	last_v: f64,
	last_a: f64,
	filled: bool,
}

impl HwmaStream {
	pub fn try_new(params: HwmaParams) -> Result<Self, HwmaError> {
		let na = params.na.unwrap_or(0.2);
		let nb = params.nb.unwrap_or(0.1);
		let nc = params.nc.unwrap_or(0.1);

		if !na.is_finite() || !nb.is_finite() || !nc.is_finite() {
			return Err(HwmaError::InvalidParams { na, nb, nc });
		}
		if !(na > 0.0 && na < 1.0 && nb > 0.0 && nb < 1.0 && nc > 0.0 && nc < 1.0) {
			return Err(HwmaError::InvalidParams { na, nb, nc });
		}
		Ok(Self {
			na,
			nb,
			nc,
			last_f: f64::NAN,
			last_v: 0.0,
			last_a: 0.0,
			filled: false,
		})
	}

	#[inline(always)]
	pub fn update(&mut self, value: f64) -> Option<f64> {
		if !self.filled {
			self.last_f = value;
			self.last_v = 0.0;
			self.last_a = 0.0;
			self.filled = true;
			return Some(self.last_f + self.last_v + 0.5 * self.last_a);
		}
		let f = (1.0 - self.na) * (self.last_f + self.last_v + 0.5 * self.last_a) + self.na * value;
		let v = (1.0 - self.nb) * (self.last_v + self.last_a) + self.nb * (f - self.last_f);
		let a = (1.0 - self.nc) * self.last_a + self.nc * (v - self.last_v);
		let hwma_val = f + v + 0.5 * a;
		self.last_f = f;
		self.last_v = v;
		self.last_a = a;
		Some(hwma_val)
	}
}

#[derive(Clone, Debug)]
pub struct HwmaBatchRange {
	pub na: (f64, f64, f64),
	pub nb: (f64, f64, f64),
	pub nc: (f64, f64, f64),
}

impl Default for HwmaBatchRange {
	fn default() -> Self {
		Self {
			na: (0.2, 6.0, 0.0),
			nb: (0.1, 4.0, 0.0),
			nc: (0.1, 4.0, 0.0),
		}
	}
}

#[derive(Clone, Debug, Default)]
pub struct HwmaBatchBuilder {
	range: HwmaBatchRange,
	kernel: Kernel,
}

impl HwmaBatchBuilder {
	pub fn new() -> Self {
		Self::default()
	}
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline]
	pub fn na_range(mut self, start: f64, end: f64, step: f64) -> Self {
		self.range.na = (start, end, step);
		self
	}
	#[inline]
	pub fn nb_range(mut self, start: f64, end: f64, step: f64) -> Self {
		self.range.nb = (start, end, step);
		self
	}
	#[inline]
	pub fn nc_range(mut self, start: f64, end: f64, step: f64) -> Self {
		self.range.nc = (start, end, step);
		self
	}
	#[inline]
	pub fn na_static(mut self, v: f64) -> Self {
		self.range.na = (v, v, 0.0);
		self
	}
	#[inline]
	pub fn nb_static(mut self, v: f64) -> Self {
		self.range.nb = (v, v, 0.0);
		self
	}
	#[inline]
	pub fn nc_static(mut self, v: f64) -> Self {
		self.range.nc = (v, v, 0.0);
		self
	}

	pub fn apply_slice(self, data: &[f64]) -> Result<HwmaBatchOutput, HwmaError> {
		hwma_batch_with_kernel(data, &self.range, self.kernel)
	}
	pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<HwmaBatchOutput, HwmaError> {
		HwmaBatchBuilder::new().kernel(k).apply_slice(data)
	}
	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<HwmaBatchOutput, HwmaError> {
		let slice = source_type(c, src);
		self.apply_slice(slice)
	}
	pub fn with_default_candles(c: &Candles) -> Result<HwmaBatchOutput, HwmaError> {
		HwmaBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
	}
}

pub fn hwma_batch_with_kernel(data: &[f64], sweep: &HwmaBatchRange, k: Kernel) -> Result<HwmaBatchOutput, HwmaError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(HwmaError::EmptyData),
	};

	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	hwma_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct HwmaBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<HwmaParams>,
	pub rows: usize,
	pub cols: usize,
}
impl HwmaBatchOutput {
	pub fn row_for_params(&self, p: &HwmaParams) -> Option<usize> {
		self.combos.iter().position(|c| {
			(c.na.unwrap_or(0.2) - p.na.unwrap_or(0.2)).abs() < 1e-12
				&& (c.nb.unwrap_or(0.1) - p.nb.unwrap_or(0.1)).abs() < 1e-12
				&& (c.nc.unwrap_or(0.1) - p.nc.unwrap_or(0.1)).abs() < 1e-12
		})
	}
	pub fn values_for(&self, p: &HwmaParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &HwmaBatchRange) -> Vec<HwmaParams> {
	fn axis((start, end, step): (f64, f64, f64)) -> Vec<f64> {
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
	let nas = axis(r.na);
	let nbs = axis(r.nb);
	let ncs = axis(r.nc);

	let mut out = Vec::with_capacity(nas.len() * nbs.len() * ncs.len());
	for &a in &nas {
		for &b in &nbs {
			for &c in &ncs {
				out.push(HwmaParams {
					na: Some(a),
					nb: Some(b),
					nc: Some(c),
				});
			}
		}
	}
	out
}

#[inline(always)]
pub fn hwma_batch_slice(data: &[f64], sweep: &HwmaBatchRange, kern: Kernel) -> Result<HwmaBatchOutput, HwmaError> {
	hwma_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn hwma_batch_par_slice(data: &[f64], sweep: &HwmaBatchRange, kern: Kernel) -> Result<HwmaBatchOutput, HwmaError> {
	hwma_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn hwma_batch_inner(
	data: &[f64],
	sweep: &HwmaBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<HwmaBatchOutput, HwmaError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(HwmaError::EmptyData);
	}
	let len = data.len();
	if len == 0 {
		return Err(HwmaError::EmptyData);
	}
	let first = data.iter().position(|x| !x.is_nan()).ok_or(HwmaError::AllValuesNaN)?;
	for prm in &combos {
		let na = prm.na.unwrap();
		let nb = prm.nb.unwrap();
		let nc = prm.nc.unwrap();
		if !na.is_finite() || !nb.is_finite() || !nc.is_finite() {
			return Err(HwmaError::InvalidParams { na, nb, nc });
		}
		if !(na > 0.0 && na < 1.0 && nb > 0.0 && nb < 1.0 && nc > 0.0 && nc < 1.0) {
			return Err(HwmaError::InvalidParams { na, nb, nc });
		}
	}
	let rows = combos.len();
	let cols = len;
	let warm: Vec<usize> = std::iter::repeat(first).take(rows).collect();

	// ----- 2. allocate rows×cols as MaybeUninit and write the NaN prefixes --------
	let mut buf_mu = make_uninit_matrix(rows, cols);
	unsafe { init_matrix_prefixes(&mut buf_mu, cols, &warm) };

	// Use ManuallyDrop pattern to maintain capacity (following alma.rs pattern)
	let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
	let out: &mut [f64] =
		unsafe { core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len()) };

	// ----- 3. closure that fills one row ------------------------------------------
	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let prm = &combos[row];
		let na = prm.na.unwrap();
		let nb = prm.nb.unwrap();
		let nc = prm.nc.unwrap();

		match kern {
			Kernel::Scalar | Kernel::ScalarBatch => hwma_row_scalar(data, first, na, nb, nc, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => hwma_row_avx2(data, first, na, nb, nc, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => hwma_row_avx512(data, first, na, nb, nc, out_row),
			_ => hwma_row_scalar(data, first, na, nb, nc, out_row),
		}
	};

	// ----- 4. run every row, writing directly into `out` ---------------------------
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

	// ----- 5. reclaim as Vec<f64> once everything is initialized -----------------
	let values = unsafe {
		Vec::from_raw_parts(
			buf_guard.as_mut_ptr() as *mut f64,
			buf_guard.len(),
			buf_guard.capacity(),
		)
	};

	Ok(HwmaBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
fn hwma_batch_inner_into(
	data: &[f64],
	sweep: &HwmaBatchRange,
	kern: Kernel,
	parallel: bool,
	out: &mut [f64],
) -> Result<(Vec<HwmaParams>, usize, usize), HwmaError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(HwmaError::EmptyData);
	}
	let len = data.len();
	if len == 0 {
		return Err(HwmaError::EmptyData);
	}
	let first = data.iter().position(|x| !x.is_nan()).ok_or(HwmaError::AllValuesNaN)?;
	for prm in &combos {
		let na = prm.na.unwrap();
		let nb = prm.nb.unwrap();
		let nc = prm.nc.unwrap();
		if !na.is_finite() || !nb.is_finite() || !nc.is_finite() {
			return Err(HwmaError::InvalidParams { na, nb, nc });
		}
		if !(na > 0.0 && na < 1.0 && nb > 0.0 && nb < 1.0 && nc > 0.0 && nc < 1.0) {
			return Err(HwmaError::InvalidParams { na, nb, nc });
		}
	}
	let rows = combos.len();
	let cols = len;

	// Ensure output buffer is the correct size
	if out.len() != rows * cols {
		return Err(HwmaError::InvalidOutputBuffer {
			expected: rows * cols,
			actual: out.len(),
		});
	}

	let warm: Vec<usize> = std::iter::repeat(first).take(rows).collect();

	// Cast output to MaybeUninit for initialization
	let out_uninit = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len()) };
	unsafe { init_matrix_prefixes(out_uninit, cols, &warm) };

	// ----- closure that fills one row ------------------------------------------
	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let prm = &combos[row];
		let na = prm.na.unwrap();
		let nb = prm.nb.unwrap();
		let nc = prm.nc.unwrap();

		match kern {
			Kernel::Scalar | Kernel::ScalarBatch => hwma_row_scalar(data, first, na, nb, nc, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => hwma_row_avx2(data, first, na, nb, nc, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => hwma_row_avx512(data, first, na, nb, nc, out_row),
			_ => hwma_row_scalar(data, first, na, nb, nc, out_row),
		}
	};

	// ----- run every row -----------
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
unsafe fn hwma_row_scalar(data: &[f64], first: usize, na: f64, nb: f64, nc: f64, out: &mut [f64]) {
	hwma_scalar(data, na, nb, nc, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn hwma_row_avx2(data: &[f64], first: usize, na: f64, nb: f64, nc: f64, out: &mut [f64]) {
	hwma_avx2(data, na, nb, nc, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn hwma_row_avx512(data: &[f64], first: usize, na: f64, nb: f64, nc: f64, out: &mut [f64]) {
	hwma_avx2(data, na, nb, nc, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn hwma_row_avx512_short(data: &[f64], first: usize, na: f64, nb: f64, nc: f64, out: &mut [f64]) {
	hwma_scalar(data, na, nb, nc, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn hwma_row_avx512_long(data: &[f64], first: usize, na: f64, nb: f64, nc: f64, out: &mut [f64]) {
	hwma_scalar(data, na, nb, nc, first, out);
}

#[inline(always)]
pub fn expand_grid_hwma(r: &HwmaBatchRange) -> Vec<HwmaParams> {
	expand_grid(r)
}

// Python bindings
#[cfg(feature = "python")]
use numpy::ndarray::{Array1, Array2};
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

#[cfg(feature = "python")]
#[pyfunction(name = "hwma")]
#[pyo3(signature = (data, na, nb, nc, kernel=None))]
pub fn hwma_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	na: f64,
	nb: f64,
	nc: f64,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
	use numpy::{IntoPyArray, PyArrayMethods};

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, false)?; // Validate before allow_threads

	let params = HwmaParams {
		na: Some(na),
		nb: Some(nb),
		nc: Some(nc),
	};
	let input = HwmaInput::from_slice(slice_in, params);

	// GOOD: Get Vec<f64> from Rust function
	let result_vec: Vec<f64> = py
		.allow_threads(|| hwma_with_kernel(&input, kern).map(|o| o.values))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	// GOOD: Zero-copy transfer to NumPy
	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction(name = "hwma_batch")]
#[pyo3(signature = (data, na_range, nb_range, nc_range, kernel=None))]
pub fn hwma_batch_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	na_range: (f64, f64, f64),
	nb_range: (f64, f64, f64),
	nc_range: (f64, f64, f64),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, true)?; // true for batch operations

	let sweep = HwmaBatchRange {
		na: na_range,
		nb: nb_range,
		nc: nc_range,
	};

	// Calculate dimensions
	let combos = expand_grid(&sweep);
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

			hwma_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
		})
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	// Build result dictionary
	let dict = PyDict::new(py);
	dict.set_item("values", out_arr.reshape((rows, cols))?)?;

	// Use zero-copy for parameter arrays
	dict.set_item(
		"na_values",
		combos_result
			.iter()
			.map(|p| p.na.unwrap())
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;
	dict.set_item(
		"nb_values",
		combos_result
			.iter()
			.map(|p| p.nb.unwrap())
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;
	dict.set_item(
		"nc_values",
		combos_result
			.iter()
			.map(|p| p.nc.unwrap())
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;

	Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "HwmaStream")]
pub struct HwmaStreamPy {
	inner: HwmaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl HwmaStreamPy {
	#[new]
	pub fn new(na: f64, nb: f64, nc: f64) -> PyResult<Self> {
		let params = HwmaParams {
			na: Some(na),
			nb: Some(nb),
			nc: Some(nc),
		};
		match HwmaStream::try_new(params) {
			Ok(stream) => Ok(Self { inner: stream }),
			Err(e) => Err(PyValueError::new_err(format!("HwmaStream error: {}", e))),
		}
	}

	pub fn update(&mut self, value: f64) -> Option<f64> {
		self.inner.update(value)
	}
}

// ================== WASM Bindings ==================

// Helper function alias for consistency with ALMA pattern
#[inline(always)]
pub fn hwma_into_slice(dst: &mut [f64], input: &HwmaInput, kern: Kernel) -> Result<(), HwmaError> {
	hwma_with_kernel_into(input, kern, dst)
}

// ================== Safe API ==================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn hwma_js(data: &[f64], na: f64, nb: f64, nc: f64) -> Result<Vec<f64>, JsValue> {
	let params = HwmaParams {
		na: Some(na),
		nb: Some(nb),
		nc: Some(nc),
	};
	let input = HwmaInput::from_slice(data, params);

	// Allocate output buffer once
	let mut output = vec![0.0; data.len()];

	// Compute directly into output buffer
	hwma_into_slice(&mut output, &input, Kernel::Auto).map_err(|e| JsValue::from_str(&e.to_string()))?;

	Ok(output)
}

// ================== Batch API ==================

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct HwmaBatchConfig {
	pub na_range: (f64, f64, f64),
	pub nb_range: (f64, f64, f64),
	pub nc_range: (f64, f64, f64),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct HwmaBatchJsOutput {
	pub values: Vec<f64>,
	pub combos: Vec<HwmaParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = hwma_batch)]
pub fn hwma_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	let config: HwmaBatchConfig =
		serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

	let sweep = HwmaBatchRange {
		na: config.na_range,
		nb: config.nb_range,
		nc: config.nc_range,
	};

	let output = hwma_batch_inner(data, &sweep, Kernel::Auto, false).map_err(|e| JsValue::from_str(&e.to_string()))?;

	let js_output = HwmaBatchJsOutput {
		values: output.values,
		combos: output.combos,
		rows: output.rows,
		cols: output.cols,
	};

	serde_wasm_bindgen::to_value(&js_output).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

// ================== Zero-Copy WASM Functions ==================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn hwma_alloc(len: usize) -> *mut f64 {
	// Allocate memory for input/output buffer
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec); // Prevent deallocation
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn hwma_free(ptr: *mut f64, len: usize) {
	// Free allocated memory
	if !ptr.is_null() {
		unsafe {
			let _ = Vec::from_raw_parts(ptr, len, len);
		}
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = hwma_into)]
pub fn hwma_ptr_into(
	in_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	na: f64,
	nb: f64,
	nc: f64,
) -> Result<(), JsValue> {
	// Check for null pointers
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("null pointer passed to hwma_into"));
	}

	unsafe {
		// Create slice from pointer
		let data = std::slice::from_raw_parts(in_ptr, len);

		// Validate inputs
		if len == 0 {
			return Err(JsValue::from_str("Empty data"));
		}

		// Calculate HWMA
		let params = HwmaParams {
			na: Some(na),
			nb: Some(nb),
			nc: Some(nc),
		};
		let input = HwmaInput::from_slice(data, params);

		// Check for aliasing (input and output buffers are the same)
		if in_ptr == out_ptr {
			// Use temporary buffer to avoid corruption during sliding window computation
			let mut temp = vec![0.0; len];
			hwma_into_slice(&mut temp, &input, Kernel::Auto).map_err(|e| JsValue::from_str(&e.to_string()))?;

			// Copy results back to output
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			out.copy_from_slice(&temp);
		} else {
			// No aliasing, compute directly into output
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			hwma_into_slice(out, &input, Kernel::Auto).map_err(|e| JsValue::from_str(&e.to_string()))?;
		}

		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn hwma_batch_into(
	in_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	na_start: f64,
	na_end: f64,
	na_step: f64,
	nb_start: f64,
	nb_end: f64,
	nb_step: f64,
	nc_start: f64,
	nc_end: f64,
	nc_step: f64,
) -> Result<usize, JsValue> {
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("null pointer passed to hwma_batch_into"));
	}

	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);

		let sweep = HwmaBatchRange {
			na: (na_start, na_end, na_step),
			nb: (nb_start, nb_end, nb_step),
			nc: (nc_start, nc_end, nc_step),
		};

		let combos = expand_grid(&sweep);
		let rows = combos.len();
		let cols = len;

		let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);

		// Use optimized batch processing
		hwma_batch_inner_into(data, &sweep, Kernel::Auto, false, out).map_err(|e| JsValue::from_str(&e.to_string()))?;

		Ok(rows)
	}
}

// ================== Legacy WASM Functions for Test Compatibility ==================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn hwma_batch_js(
	data: &[f64],
	na_start: f64,
	na_end: f64,
	na_step: f64,
	nb_start: f64,
	nb_end: f64,
	nb_step: f64,
	nc_start: f64,
	nc_end: f64,
	nc_step: f64,
) -> Result<Vec<f64>, JsValue> {
	let sweep = HwmaBatchRange {
		na: (na_start, na_end, na_step),
		nb: (nb_start, nb_end, nb_step),
		nc: (nc_start, nc_end, nc_step),
	};

	hwma_batch_inner(data, &sweep, Kernel::Auto, false)
		.map(|output| output.values)
		.map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn hwma_batch_metadata_js(
	na_start: f64,
	na_end: f64,
	na_step: f64,
	nb_start: f64,
	nb_end: f64,
	nb_step: f64,
	nc_start: f64,
	nc_end: f64,
	nc_step: f64,
) -> Vec<f64> {
	let sweep = HwmaBatchRange {
		na: (na_start, na_end, na_step),
		nb: (nb_start, nb_end, nb_step),
		nc: (nc_start, nc_end, nc_step),
	};

	let combos = expand_grid(&sweep);
	let mut result = Vec::new();

	for combo in combos {
		result.push(combo.na.unwrap());
		result.push(combo.nb.unwrap());
		result.push(combo.nc.unwrap());
	}

	result
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;
	use proptest::prelude::*;

	fn check_hwma_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let default_params = HwmaParams {
			na: None,
			nb: None,
			nc: None,
		};
		let input = HwmaInput::from_candles(&candles, "close", default_params);
		let output = hwma_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_hwma_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = HwmaInput::from_candles(&candles, "close", HwmaParams::default());
		let result = hwma_with_kernel(&input, kernel)?;
		let expected_last_five = [
			57941.04005793378,
			58106.90324194954,
			58250.474156632234,
			58428.90005831887,
			58499.37021151028,
		];
		let start = result.values.len().saturating_sub(5);
		for (i, &val) in result.values[start..].iter().enumerate() {
			let diff = (val - expected_last_five[i]).abs();
			assert!(
				diff < 1e-3,
				"[{}] HWMA {:?} mismatch at idx {}: got {}, expected {}",
				test_name,
				kernel,
				i,
				val,
				expected_last_five[i]
			);
		}
		Ok(())
	}

	fn check_hwma_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = HwmaInput::with_default_candles(&candles);
		match input.data {
			HwmaData::Candles { source, .. } => assert_eq!(source, "close"),
			_ => panic!("Expected HwmaData::Candles"),
		}
		let output = hwma_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_hwma_invalid_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data = [10.0, 20.0, 30.0];
		let params = HwmaParams {
			na: Some(-0.2),
			nb: Some(1.1),
			nc: Some(0.1),
		};
		let input = HwmaInput::from_slice(&data, params);
		let result = hwma_with_kernel(&input, kernel);
		assert!(matches!(result, Err(HwmaError::InvalidParams { .. })));
		Ok(())
	}

	fn check_hwma_invalid_nan_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data = [1.0, 2.0];
		let params = HwmaParams {
			na: Some(f64::NAN),
			nb: Some(0.5),
			nc: Some(0.5),
		};
		let input = HwmaInput::from_slice(&data, params);
		let res = hwma_with_kernel(&input, kernel);
		assert!(matches!(res, Err(HwmaError::InvalidParams { .. })));
		Ok(())
	}

	fn check_hwma_empty_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data: [f64; 0] = [];
		let params = HwmaParams {
			na: Some(0.2),
			nb: Some(0.1),
			nc: Some(0.1),
		};
		let input = HwmaInput::from_slice(&data, params);
		let result = hwma_with_kernel(&input, kernel);
		assert!(matches!(result, Err(HwmaError::EmptyData)));
		Ok(())
	}

	fn check_hwma_small_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data = [42.0];
		let params = HwmaParams {
			na: Some(0.2),
			nb: Some(0.1),
			nc: Some(0.1),
		};
		let input = HwmaInput::from_slice(&data, params);
		let result = hwma_with_kernel(&input, kernel)?;
		assert_eq!(result.values.len(), data.len());
		assert!((result.values[0] - data[0]).abs() < 1e-12);
		Ok(())
	}

	fn check_hwma_slice_data_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params_1 = HwmaParams::default();
		let input_1 = HwmaInput::from_candles(&candles, "close", params_1);
		let result_1 = hwma_with_kernel(&input_1, kernel)?;
		assert_eq!(result_1.values.len(), candles.close.len());
		let params_2 = HwmaParams {
			na: Some(0.3),
			nb: Some(0.15),
			nc: Some(0.05),
		};
		let input_2 = HwmaInput::from_slice(&result_1.values, params_2);
		let result_2 = hwma_with_kernel(&input_2, kernel)?;
		assert_eq!(result_2.values.len(), result_1.values.len());
		for i in 240..result_2.values.len() {
			assert!(!result_2.values[i].is_nan());
		}
		Ok(())
	}

	fn check_hwma_nan_check(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = HwmaParams::default();
		let input = HwmaInput::from_candles(&candles, "close", params);
		let result = hwma_with_kernel(&input, kernel)?;
		assert_eq!(result.values.len(), candles.close.len());
		if result.values.len() > 240 {
			for i in 240..result.values.len() {
				assert!(!result.values[i].is_nan());
			}
		}
		Ok(())
	}

	fn check_hwma_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let params = HwmaParams::default();
		let input = HwmaInput::from_candles(&candles, "close", params.clone());
		let batch = hwma_with_kernel(&input, kernel)?.values;

		let mut stream = HwmaStream::try_new(params)?;
		let mut streaming = Vec::with_capacity(candles.close.len());
		for &v in &candles.close {
			match stream.update(v) {
				Some(x) => streaming.push(x),
				None => streaming.push(f64::NAN),
			}
		}

		assert_eq!(batch.len(), streaming.len());
		for (i, (&b, &s)) in batch.iter().zip(streaming.iter()).enumerate() {
			if b.is_nan() && s.is_nan() {
				continue;
			}
			assert!((b - s).abs() < 1e-9, "[{test_name}] mismatch at {i}");
		}
		Ok(())
	}

	fn check_hwma_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let strat = (
			proptest::collection::vec((-1e6f64..1e6).prop_filter("finite", |x| x.is_finite()), 20..100),
			0.01f64..0.99,
			0.01f64..0.99,
			0.01f64..0.99,
		);

		proptest::test_runner::TestRunner::default().run(&strat, |(data, na, nb, nc)| {
			let params = HwmaParams {
				na: Some(na),
				nb: Some(nb),
				nc: Some(nc),
			};
			let input = HwmaInput::from_slice(&data, params);
			let HwmaOutput { values } = hwma_with_kernel(&input, kernel).unwrap();

			let mut min = f64::INFINITY;
			let mut max = f64::NEG_INFINITY;
			for (idx, &x) in data.iter().enumerate() {
				if !x.is_nan() {
					if x < min {
						min = x;
					}
					if x > max {
						max = x;
					}
				}
				let y = values[idx];
				prop_assert!(
					y.is_nan() || (y >= min - 1e-9 && y <= max + 1e-9),
					"idx {idx}: {y} not in [{min}, {max}]"
				);
			}
			Ok(())
		})?;

		Ok(())
	}

	// Check for poison values in single output - only runs in debug mode
	#[cfg(debug_assertions)]
	fn check_hwma_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		// Test multiple parameter combinations to better catch uninitialized memory bugs
		let test_params = vec![
			// Default parameters
			HwmaParams::default(),
			// Small na, nb, nc values
			HwmaParams {
				na: Some(0.05),
				nb: Some(0.05),
				nc: Some(0.05),
			},
			HwmaParams {
				na: Some(0.1),
				nb: Some(0.1),
				nc: Some(0.1),
			},
			HwmaParams {
				na: Some(0.15),
				nb: Some(0.1),
				nc: Some(0.05),
			},
			// Medium values
			HwmaParams {
				na: Some(0.2),
				nb: Some(0.2),
				nc: Some(0.2),
			},
			HwmaParams {
				na: Some(0.3),
				nb: Some(0.3),
				nc: Some(0.3),
			},
			HwmaParams {
				na: Some(0.4),
				nb: Some(0.4),
				nc: Some(0.4),
			},
			HwmaParams {
				na: Some(0.5),
				nb: Some(0.5),
				nc: Some(0.5),
			},
			// Large values
			HwmaParams {
				na: Some(0.6),
				nb: Some(0.6),
				nc: Some(0.6),
			},
			HwmaParams {
				na: Some(0.7),
				nb: Some(0.7),
				nc: Some(0.7),
			},
			HwmaParams {
				na: Some(0.8),
				nb: Some(0.8),
				nc: Some(0.8),
			},
			HwmaParams {
				na: Some(0.9),
				nb: Some(0.9),
				nc: Some(0.9),
			},
			// Mixed values
			HwmaParams {
				na: Some(0.1),
				nb: Some(0.5),
				nc: Some(0.9),
			},
			HwmaParams {
				na: Some(0.9),
				nb: Some(0.5),
				nc: Some(0.1),
			},
			HwmaParams {
				na: Some(0.5),
				nb: Some(0.3),
				nc: Some(0.2),
			},
			// Edge cases
			HwmaParams {
				na: Some(0.01),
				nb: Some(0.01),
				nc: Some(0.01),
			},
			HwmaParams {
				na: Some(0.99),
				nb: Some(0.99),
				nc: Some(0.99),
			},
		];

		for (param_idx, params) in test_params.iter().enumerate() {
			let input = HwmaInput::from_candles(&candles, "close", params.clone());
			let output = hwma_with_kernel(&input, kernel)?;

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
                        with params: na={}, nb={}, nc={}",
						test_name,
						val,
						bits,
						i,
						params.na.unwrap_or(0.2),
						params.nb.unwrap_or(0.1),
						params.nc.unwrap_or(0.1)
					);
				}

				// Check for init_matrix_prefixes poison (0x22222222_22222222)
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
                        with params: na={}, nb={}, nc={}",
						test_name,
						val,
						bits,
						i,
						params.na.unwrap_or(0.2),
						params.nb.unwrap_or(0.1),
						params.nc.unwrap_or(0.1)
					);
				}

				// Check for make_uninit_matrix poison (0x33333333_33333333)
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
                        with params: na={}, nb={}, nc={}",
						test_name,
						val,
						bits,
						i,
						params.na.unwrap_or(0.2),
						params.nb.unwrap_or(0.1),
						params.nc.unwrap_or(0.1)
					);
				}
			}
		}

		Ok(())
	}

	// Release mode stub - does nothing
	#[cfg(not(debug_assertions))]
	fn check_hwma_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(())
	}

	macro_rules! generate_all_hwma_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $( #[test] fn [<$test_fn _scalar_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar);
                })*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $( #[test] fn [<$test_fn _avx2_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2);
                })*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $( #[test] fn [<$test_fn _avx512_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512);
                })*
                // Test WASM SIMD128 implementation
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                $( #[test] fn [<$test_fn _simd128_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _simd128_f64>]), Kernel::Scalar);
                })*
            }
        }
    }
	generate_all_hwma_tests!(
		check_hwma_partial_params,
		check_hwma_accuracy,
		check_hwma_default_candles,
		check_hwma_invalid_params,
		check_hwma_invalid_nan_params,
		check_hwma_empty_data,
		check_hwma_small_data,
		check_hwma_slice_data_reinput,
		check_hwma_nan_check,
		check_hwma_streaming,
		check_hwma_property,
		check_hwma_no_poison
	);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let output = HwmaBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;

		let def = HwmaParams::default();
		let row = output.values_for(&def).expect("default row missing");

		assert_eq!(row.len(), c.close.len());

		let expected = [
			57941.04005793378,
			58106.90324194954,
			58250.474156632234,
			58428.90005831887,
			58499.37021151028,
		];
		let start = row.len() - 5;
		for (i, &v) in row[start..].iter().enumerate() {
			assert!(
				(v - expected[i]).abs() < 1e-3,
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
		let test_configs = vec![
			// Small ranges with all params
			(0.05, 0.15, 0.05, 0.05, 0.15, 0.05, 0.05, 0.15, 0.05),
			// Medium ranges
			(0.1, 0.5, 0.1, 0.1, 0.5, 0.1, 0.1, 0.5, 0.1),
			// Large ranges
			(0.5, 0.9, 0.1, 0.5, 0.9, 0.1, 0.5, 0.9, 0.1),
			// Different param ranges
			(0.1, 0.9, 0.4, 0.1, 0.5, 0.2, 0.1, 0.3, 0.1),
			// Edge case: very small values
			(0.01, 0.05, 0.01, 0.01, 0.05, 0.01, 0.01, 0.05, 0.01),
			// Edge case: near 1.0 values
			(0.8, 0.99, 0.05, 0.8, 0.99, 0.05, 0.8, 0.99, 0.05),
			// Mixed ranges
			(0.1, 0.9, 0.2, 0.2, 0.8, 0.3, 0.3, 0.7, 0.2),
			// Original configuration
			(0.1, 0.9, 0.4, 0.1, 0.9, 0.4, 0.1, 0.9, 0.4),
			// Dense grid
			(0.1, 0.3, 0.05, 0.1, 0.3, 0.05, 0.1, 0.3, 0.05),
		];

		for (cfg_idx, &(na_start, na_end, na_step, nb_start, nb_end, nb_step, nc_start, nc_end, nc_step)) in
			test_configs.iter().enumerate()
		{
			let output = HwmaBatchBuilder::new()
				.kernel(kernel)
				.na_range(na_start, na_end, na_step)
				.nb_range(nb_start, nb_end, nb_step)
				.nc_range(nc_start, nc_end, nc_step)
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
                        at row {} col {} (flat index {}) with params: na={}, nb={}, nc={}",
						test,
						cfg_idx,
						val,
						bits,
						row,
						col,
						idx,
						combo.na.unwrap_or(0.2),
						combo.nb.unwrap_or(0.1),
						combo.nc.unwrap_or(0.1)
					);
				}

				// Check for init_matrix_prefixes poison (0x22222222_22222222)
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
                        at row {} col {} (flat index {}) with params: na={}, nb={}, nc={}",
						test,
						cfg_idx,
						val,
						bits,
						row,
						col,
						idx,
						combo.na.unwrap_or(0.2),
						combo.nb.unwrap_or(0.1),
						combo.nc.unwrap_or(0.1)
					);
				}

				// Check for make_uninit_matrix poison (0x33333333_33333333)
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
                        at row {} col {} (flat index {}) with params: na={}, nb={}, nc={}",
						test,
						cfg_idx,
						val,
						bits,
						row,
						col,
						idx,
						combo.na.unwrap_or(0.2),
						combo.nb.unwrap_or(0.1),
						combo.nc.unwrap_or(0.1)
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
