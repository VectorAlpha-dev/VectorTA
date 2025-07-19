//! # Accelerator Oscillator (ACOSC)
//!
//! Bill Williams’ AC Oscillator: measures median price acceleration via SMA5, SMA34, and further SMA5 smoothing.
//!
//! ## Parameters
//! - None (fixed: periods are 5 and 34)
//!
//! ## Errors
//! - **CandleFieldError**: Failed to get high/low from candles
//! - **LengthMismatch**: Slices have different lengths
//! - **NotEnoughData**: Less than 39 data points
//!
//! ## Returns
//! - `Ok(AcoscOutput)` with vectors of `osc` and `change`

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
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum AcoscData<'a> {
	Candles { candles: &'a Candles },
	Slices { high: &'a [f64], low: &'a [f64] },
}

#[derive(Debug, Clone, Default)]
pub struct AcoscParams {} // ACOSC is fixed (no param grid), but kept for parity

#[derive(Debug, Clone)]
pub struct AcoscInput<'a> {
	pub data: AcoscData<'a>,
	pub params: AcoscParams,
}
impl<'a> AcoscInput<'a> {
	#[inline]
	pub fn from_candles(candles: &'a Candles, params: AcoscParams) -> Self {
		Self {
			data: AcoscData::Candles { candles },
			params,
		}
	}
	#[inline]
	pub fn from_slices(high: &'a [f64], low: &'a [f64], params: AcoscParams) -> Self {
		Self {
			data: AcoscData::Slices { high, low },
			params,
		}
	}
	#[inline]
	pub fn with_default_candles(candles: &'a Candles) -> Self {
		Self {
			data: AcoscData::Candles { candles },
			params: AcoscParams::default(),
		}
	}
}

#[derive(Debug, Clone)]
pub struct AcoscOutput {
	pub osc: Vec<f64>,
	pub change: Vec<f64>,
}

#[derive(Debug, Error)]
pub enum AcoscError {
	#[error("acosc: Failed to get high/low fields from candles: {msg}")]
	CandleFieldError { msg: String },
	#[error("acosc: Mismatch in high/low candle data lengths: high_len={high_len}, low_len={low_len}")]
	LengthMismatch { high_len: usize, low_len: usize },
	#[error("acosc: Not enough data points: required={required}, actual={actual}")]
	NotEnoughData { required: usize, actual: usize },
}

#[inline]
pub fn acosc(input: &AcoscInput) -> Result<AcoscOutput, AcoscError> {
	acosc_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn acosc_prepare<'a>(input: &'a AcoscInput, kernel: Kernel) -> Result<(&'a [f64], &'a [f64], Kernel), AcoscError> {
	let (high, low) = match &input.data {
		AcoscData::Candles { candles } => {
			let h = candles
				.select_candle_field("high")
				.map_err(|e| AcoscError::CandleFieldError { msg: e.to_string() })?;
			let l = candles
				.select_candle_field("low")
				.map_err(|e| AcoscError::CandleFieldError { msg: e.to_string() })?;
			(h, l)
		}
		AcoscData::Slices { high, low } => (*high, *low),
	};

	if high.len() != low.len() {
		return Err(AcoscError::LengthMismatch {
			high_len: high.len(),
			low_len: low.len(),
		});
	}

	let len = low.len();
	const REQUIRED_LENGTH: usize = 39; // PERIOD_SMA34 + PERIOD_SMA5
	if len < REQUIRED_LENGTH {
		return Err(AcoscError::NotEnoughData {
			required: REQUIRED_LENGTH,
			actual: len,
		});
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	Ok((high, low, chosen))
}
pub fn acosc_with_kernel(input: &AcoscInput, kernel: Kernel) -> Result<AcoscOutput, AcoscError> {
	let (high, low, chosen) = acosc_prepare(input, kernel)?;

	let len = low.len();
	const REQUIRED_LENGTH: usize = 39; // PERIOD_SMA34 + PERIOD_SMA5

	// Calculate warmup period (ACOSC needs 39 data points before producing values)
	let warmup_period = REQUIRED_LENGTH - 1; // 38 (since we start producing at index 38)

	// Use zero-copy allocation
	let mut osc = alloc_with_nan_prefix(len, warmup_period);
	let mut change = alloc_with_nan_prefix(len, warmup_period);

	// Use the compute_into pattern for consistency with ALMA
	acosc_compute_into(high, low, chosen, &mut osc, &mut change);
	Ok(AcoscOutput { osc, change })
}

#[inline(always)]
fn acosc_compute_into(high: &[f64], low: &[f64], kernel: Kernel, osc_out: &mut [f64], change_out: &mut [f64]) {
	unsafe {
		match kernel {
			Kernel::Scalar | Kernel::ScalarBatch => acosc_scalar(high, low, osc_out, change_out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => acosc_avx2(high, low, osc_out, change_out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => acosc_avx512(high, low, osc_out, change_out),
			#[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
			Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
				acosc_scalar(high, low, osc_out, change_out)
			}
			Kernel::Auto => {
				unreachable!("Kernel::Auto should be resolved before calling compute_into")
			}
		}
	}
}

#[inline]
pub fn acosc_scalar(high: &[f64], low: &[f64], osc: &mut [f64], change: &mut [f64]) {
	// SCALAR LOGIC UNCHANGED
	const PERIOD_SMA5: usize = 5;
	const PERIOD_SMA34: usize = 34;
	const INV5: f64 = 1.0 / 5.0;
	const INV34: f64 = 1.0 / 34.0;
	let len = high.len();
	let mut queue5 = [0.0; PERIOD_SMA5];
	let mut queue34 = [0.0; PERIOD_SMA34];
	let mut queue5_ao = [0.0; PERIOD_SMA5];
	let mut sum5 = 0.0;
	let mut sum34 = 0.0;
	let mut sum5_ao = 0.0;
	let mut idx5 = 0;
	let mut idx34 = 0;
	let mut idx5_ao = 0;
	for i in 0..PERIOD_SMA34 {
		let med = (high[i] + low[i]) * 0.5;
		sum34 += med;
		queue34[i] = med;
		if i < PERIOD_SMA5 {
			sum5 += med;
			queue5[i] = med;
		}
	}
	for i in PERIOD_SMA34..(PERIOD_SMA34 + PERIOD_SMA5 - 1) {
		let med = (high[i] + low[i]) * 0.5;
		sum34 += med - queue34[idx34];
		queue34[idx34] = med;
		idx34 = (idx34 + 1) % PERIOD_SMA34;
		let sma34 = sum34 * INV34;
		sum5 += med - queue5[idx5];
		queue5[idx5] = med;
		idx5 = (idx5 + 1) % PERIOD_SMA5;
		let sma5 = sum5 * INV5;
		let ao = sma5 - sma34;
		sum5_ao += ao;
		queue5_ao[idx5_ao] = ao;
		idx5_ao += 1;
	}
	if idx5_ao == PERIOD_SMA5 {
		idx5_ao = 0;
	}
	let mut prev_res = 0.0;
	for i in (PERIOD_SMA34 + PERIOD_SMA5 - 1)..len {
		let med = (high[i] + low[i]) * 0.5;
		sum34 += med - queue34[idx34];
		queue34[idx34] = med;
		idx34 = (idx34 + 1) % PERIOD_SMA34;
		let sma34 = sum34 * INV34;
		sum5 += med - queue5[idx5];
		queue5[idx5] = med;
		idx5 = (idx5 + 1) % PERIOD_SMA5;
		let sma5 = sum5 * INV5;
		let ao = sma5 - sma34;
		let old_ao = queue5_ao[idx5_ao];
		sum5_ao += ao - old_ao;
		queue5_ao[idx5_ao] = ao;
		idx5_ao = (idx5_ao + 1) % PERIOD_SMA5;
		let sma5_ao = sum5_ao * INV5;
		let res = ao - sma5_ao;
		let mom = res - prev_res;
		prev_res = res;
		osc[i] = res;
		change[i] = mom;
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn acosc_avx512(high: &[f64], low: &[f64], osc: &mut [f64], change: &mut [f64]) {
	acosc_scalar(high, low, osc, change)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn acosc_avx2(high: &[f64], low: &[f64], osc: &mut [f64], change: &mut [f64]) {
	acosc_scalar(high, low, osc, change)
}
#[inline]
pub fn acosc_avx512_short(high: &[f64], low: &[f64], osc: &mut [f64], change: &mut [f64]) {
	acosc_scalar(high, low, osc, change)
}
#[inline]
pub fn acosc_avx512_long(high: &[f64], low: &[f64], osc: &mut [f64], change: &mut [f64]) {
	acosc_scalar(high, low, osc, change)
}

// Stream API (stateful tick-by-tick)
#[derive(Debug, Clone)]
pub struct AcoscStream {
	queue5: [f64; 5],
	queue34: [f64; 34],
	queue5_ao: [f64; 5],
	sum5: f64,
	sum34: f64,
	sum5_ao: f64,
	idx5: usize,
	idx34: usize,
	idx5_ao: usize,
	filled: usize,
	prev_res: f64,
}
impl AcoscStream {
	pub fn try_new(_params: AcoscParams) -> Result<Self, AcoscError> {
		Ok(Self {
			queue5: [0.0; 5],
			queue34: [0.0; 34],
			queue5_ao: [0.0; 5],
			sum5: 0.0,
			sum34: 0.0,
			sum5_ao: 0.0,
			idx5: 0,
			idx34: 0,
			idx5_ao: 0,
			filled: 0,
			prev_res: 0.0,
		})
	}
	#[inline(always)]
	pub fn update(&mut self, high: f64, low: f64) -> Option<(f64, f64)> {
		let med = (high + low) * 0.5;
		self.filled += 1;
		if self.filled <= 34 {
			self.sum34 += med;
			self.queue34[self.filled - 1] = med;
			if self.filled <= 5 {
				self.sum5 += med;
				self.queue5[self.filled - 1] = med;
			}
			return None;
		}
		if self.filled < 39 {
			self.sum34 += med - self.queue34[self.idx34];
			self.queue34[self.idx34] = med;
			self.idx34 = (self.idx34 + 1) % 34;
			let sma34 = self.sum34 / 34.0;
			self.sum5 += med - self.queue5[self.idx5];
			self.queue5[self.idx5] = med;
			self.idx5 = (self.idx5 + 1) % 5;
			let sma5 = self.sum5 / 5.0;
			let ao = sma5 - sma34;
			self.sum5_ao += ao;
			self.queue5_ao[self.idx5_ao] = ao;
			self.idx5_ao = (self.idx5_ao + 1) % 5;
			return None;
		}
		self.sum34 += med - self.queue34[self.idx34];
		self.queue34[self.idx34] = med;
		self.idx34 = (self.idx34 + 1) % 34;
		let sma34 = self.sum34 / 34.0;
		self.sum5 += med - self.queue5[self.idx5];
		self.queue5[self.idx5] = med;
		self.idx5 = (self.idx5 + 1) % 5;
		let sma5 = self.sum5 / 5.0;
		let ao = sma5 - sma34;
		let old_ao = self.queue5_ao[self.idx5_ao];
		self.sum5_ao += ao - old_ao;
		self.queue5_ao[self.idx5_ao] = ao;
		self.idx5_ao = (self.idx5_ao + 1) % 5;
		let sma5_ao = self.sum5_ao / 5.0;
		let res = ao - sma5_ao;
		let mom = res - self.prev_res;
		self.prev_res = res;
		Some((res, mom))
	}
}

// --- Batch/Builder API ---

#[derive(Clone, Debug)]
pub struct AcoscBatchRange {} // For parity only

impl Default for AcoscBatchRange {
	fn default() -> Self {
		Self {}
	}
}

#[derive(Clone, Debug, Default)]
pub struct AcoscBatchBuilder {
	kernel: Kernel,
}
impl AcoscBatchBuilder {
	pub fn new() -> Self {
		Self::default()
	}
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	pub fn apply_slice(self, high: &[f64], low: &[f64]) -> Result<AcoscBatchOutput, AcoscError> {
		acosc_batch_with_kernel(high, low, self.kernel)
	}
	pub fn with_default_slice(high: &[f64], low: &[f64], k: Kernel) -> Result<AcoscBatchOutput, AcoscError> {
		AcoscBatchBuilder::new().kernel(k).apply_slice(high, low)
	}
	pub fn apply_candles(self, c: &Candles) -> Result<AcoscBatchOutput, AcoscError> {
		let high = c
			.select_candle_field("high")
			.map_err(|e| AcoscError::CandleFieldError { msg: e.to_string() })?;
		let low = c
			.select_candle_field("low")
			.map_err(|e| AcoscError::CandleFieldError { msg: e.to_string() })?;
		self.apply_slice(high, low)
	}
	pub fn with_default_candles(c: &Candles) -> Result<AcoscBatchOutput, AcoscError> {
		AcoscBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c)
	}
}
#[derive(Clone, Debug)]
pub struct AcoscBatchOutput {
	pub osc: Vec<f64>,
	pub change: Vec<f64>,
	pub rows: usize,
	pub cols: usize,
}
pub fn acosc_batch_with_kernel(high: &[f64], low: &[f64], k: Kernel) -> Result<AcoscBatchOutput, AcoscError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => {
			return Err(AcoscError::NotEnoughData {
				required: 39,
				actual: 0,
			})
		}
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	acosc_batch_par_slice(high, low, simd)
}
#[inline(always)]
pub fn acosc_batch_slice(high: &[f64], low: &[f64], kern: Kernel) -> Result<AcoscBatchOutput, AcoscError> {
	acosc_batch_inner(high, low, kern, false)
}
#[inline(always)]
pub fn acosc_batch_par_slice(high: &[f64], low: &[f64], kern: Kernel) -> Result<AcoscBatchOutput, AcoscError> {
	acosc_batch_inner(high, low, kern, true)
}
#[inline(always)]
fn acosc_batch_inner(high: &[f64], low: &[f64], kern: Kernel, _parallel: bool) -> Result<AcoscBatchOutput, AcoscError> {
	let cols = high.len();
	let rows = 1; // ACOSC has no parameters, so always 1 row

	// Check for minimum data length
	const REQUIRED_LENGTH: usize = 39; // PERIOD_SMA34 + PERIOD_SMA5
	if cols < REQUIRED_LENGTH {
		return Err(AcoscError::NotEnoughData {
			required: REQUIRED_LENGTH,
			actual: cols,
		});
	}

	// Step 1: Allocate uninitialized matrices for both outputs
	let mut buf_osc_mu = make_uninit_matrix(rows, cols);
	let mut buf_change_mu = make_uninit_matrix(rows, cols);

	// Step 2: Calculate warmup periods (constant for ACOSC)
	const WARMUP_PERIOD: usize = 38; // PERIOD_SMA34 + PERIOD_SMA5 - 1
	let warmup_periods = vec![WARMUP_PERIOD]; // Single row

	// Step 3: Initialize NaN prefixes for each matrix
	init_matrix_prefixes(&mut buf_osc_mu, cols, &warmup_periods);
	init_matrix_prefixes(&mut buf_change_mu, cols, &warmup_periods);

	// Step 4: Convert to mutable slices for computation
	let mut buf_osc_guard = ManuallyDrop::new(buf_osc_mu);
	let mut buf_change_guard = ManuallyDrop::new(buf_change_mu);

	let osc_slice: &mut [f64] =
		unsafe { core::slice::from_raw_parts_mut(buf_osc_guard.as_mut_ptr() as *mut f64, buf_osc_guard.len()) };

	let change_slice: &mut [f64] =
		unsafe { core::slice::from_raw_parts_mut(buf_change_guard.as_mut_ptr() as *mut f64, buf_change_guard.len()) };

	// Step 5: Compute into the buffers
	acosc_compute_into(high, low, kern, osc_slice, change_slice);

	// Step 6: Reclaim as Vec<f64>
	let osc = unsafe {
		Vec::from_raw_parts(
			buf_osc_guard.as_mut_ptr() as *mut f64,
			buf_osc_guard.len(),
			buf_osc_guard.capacity(),
		)
	};

	let change = unsafe {
		Vec::from_raw_parts(
			buf_change_guard.as_mut_ptr() as *mut f64,
			buf_change_guard.len(),
			buf_change_guard.capacity(),
		)
	};

	Ok(AcoscBatchOutput {
		osc,
		change,
		rows,
		cols,
	})
}
#[inline(always)]
pub fn expand_grid(_r: &AcoscBatchRange) -> Vec<AcoscParams> {
	vec![AcoscParams::default()]
}

// --- Row kernel API (batch) ---
#[inline(always)]
pub unsafe fn acosc_row_scalar(high: &[f64], low: &[f64], out_osc: &mut [f64], out_change: &mut [f64]) {
	acosc_scalar(high, low, out_osc, out_change)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn acosc_row_avx2(high: &[f64], low: &[f64], out_osc: &mut [f64], out_change: &mut [f64]) {
	acosc_avx2(high, low, out_osc, out_change)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn acosc_row_avx512(high: &[f64], low: &[f64], out_osc: &mut [f64], out_change: &mut [f64]) {
	acosc_avx512(high, low, out_osc, out_change)
}
#[inline(always)]
pub fn acosc_row_avx512_short(high: &[f64], low: &[f64], out_osc: &mut [f64], out_change: &mut [f64]) {
	acosc_scalar(high, low, out_osc, out_change)
}
#[inline(always)]
pub fn acosc_row_avx512_long(high: &[f64], low: &[f64], out_osc: &mut [f64], out_change: &mut [f64]) {
	acosc_scalar(high, low, out_osc, out_change)
}

// --- Optional: AcoscBuilder for strict parity ---
#[derive(Copy, Clone, Debug, Default)]
pub struct AcoscBuilder {
	kernel: Kernel,
}
impl AcoscBuilder {
	#[inline(always)]
	pub fn new() -> Self {
		Self::default()
	}
	#[inline(always)]
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline(always)]
	pub fn apply_candles(self, candles: &Candles) -> Result<AcoscOutput, AcoscError> {
		let input = AcoscInput::with_default_candles(candles);
		acosc_with_kernel(&input, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<AcoscOutput, AcoscError> {
		let input = AcoscInput::from_slices(high, low, AcoscParams::default());
		acosc_with_kernel(&input, self.kernel)
	}
}

#[cfg(feature = "python")]
use numpy;
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "python")]
#[pyfunction(name = "acosc")]
#[pyo3(signature = (high, low, kernel=None))]
/// Compute the Accelerator Oscillator (ACOSC) of the input data.
///
/// Bill Williams' AC Oscillator measures median price acceleration via SMA5, SMA34,
/// and further SMA5 smoothing.
///
/// Parameters:
/// -----------
/// high : np.ndarray
///     Array of high prices (float64).
/// low : np.ndarray
///     Array of low prices (float64).
/// kernel : str, optional
///     Computation kernel to use: 'auto', 'scalar', 'avx2', 'avx512'.
///     Default is 'auto' which auto-detects the best available.
///
/// Returns:
/// --------
/// tuple[np.ndarray, np.ndarray]
///     Tuple of (osc, change) arrays, same length as input.
///
/// Raises:
/// -------
/// ValueError
///     If high/low lengths mismatch or insufficient data.
pub fn acosc_py<'py>(
	py: Python<'py>,
	high: numpy::PyReadonlyArray1<'py, f64>,
	low: numpy::PyReadonlyArray1<'py, f64>,
	kernel: Option<&str>,
) -> PyResult<(Bound<'py, numpy::PyArray1<f64>>, Bound<'py, numpy::PyArray1<f64>>)> {
	use numpy::{PyArray1, PyArrayMethods};

	let high_slice = high.as_slice()?; // zero-copy, read-only view
	let low_slice = low.as_slice()?; // zero-copy, read-only view

	// Parse and validate kernel
	let kern = crate::utilities::kernel_validation::validate_kernel(kernel, false)?;

	// Build input struct
	let params = AcoscParams::default();
	let acosc_in = AcoscInput::from_slices(high_slice, low_slice, params);

	// Allocate NumPy output buffers
	let out_osc = unsafe { PyArray1::<f64>::new(py, [high_slice.len()], false) };
	let out_change = unsafe { PyArray1::<f64>::new(py, [high_slice.len()], false) };
	let slice_osc = unsafe { out_osc.as_slice_mut()? };
	let slice_change = unsafe { out_change.as_slice_mut()? };

	// Heavy lifting without the GIL
	py.allow_threads(|| -> Result<(), AcoscError> {
		// Prepare and resolve kernel
		let (_, _, chosen) = acosc_prepare(&acosc_in, kern)?;

		// SAFETY: We must write to ALL elements before returning to Python
		// 1. Write NaN prefix to the first 38 elements (ACOSC warmup period)
		const WARMUP_PERIOD: usize = 38; // PERIOD_SMA34 + PERIOD_SMA5 - 1
		if high_slice.len() >= WARMUP_PERIOD {
			slice_osc[..WARMUP_PERIOD].fill(f64::NAN);
			slice_change[..WARMUP_PERIOD].fill(f64::NAN);
		} else {
			// If data is shorter than warmup, fill everything with NaN
			slice_osc.fill(f64::NAN);
			slice_change.fill(f64::NAN);
		}

		// 2. acosc_compute_into MUST write to all elements from index 38 onwards
		// This is guaranteed by the ACOSC algorithm implementation
		acosc_compute_into(high_slice, low_slice, chosen, slice_osc, slice_change);

		Ok(())
	})
	.map_err(|e| PyValueError::new_err(e.to_string()))?;

	Ok((out_osc.into(), out_change.into()))
}

#[cfg(feature = "python")]
#[pyclass(name = "AcoscStream")]
pub struct AcoscStreamPy {
	stream: AcoscStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl AcoscStreamPy {
	#[new]
	fn new() -> PyResult<Self> {
		let params = AcoscParams::default();
		let stream = AcoscStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(AcoscStreamPy { stream })
	}

	/// Updates the stream with new high/low values and returns the calculated (osc, change) tuple.
	/// Returns `None` if the buffer is not yet full.
	fn update(&mut self, high: f64, low: f64) -> Option<(f64, f64)> {
		self.stream.update(high, low)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "acosc_batch")]
#[pyo3(signature = (high, low, kernel=None))]
/// Compute ACOSC in batch mode (since ACOSC has no parameters, this is equivalent to single mode).
///
/// Parameters:
/// -----------
/// high : np.ndarray
///     Array of high prices (float64).
/// low : np.ndarray
///     Array of low prices (float64).
/// kernel : str, optional
///     Computation kernel: 'auto', 'scalar', 'avx2', 'avx512'.
///     Default is 'auto' which auto-detects the best available.
///
/// Returns:
/// --------
/// dict
///     Dictionary with 'osc' and 'change' arrays (2D with 1 row).
pub fn acosc_batch_py<'py>(
	py: Python<'py>,
	high: numpy::PyReadonlyArray1<'py, f64>,
	low: numpy::PyReadonlyArray1<'py, f64>,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
	use numpy::{PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;

	let high_slice = high.as_slice()?;
	let low_slice = low.as_slice()?;

	// Parse and validate kernel
	let kern = crate::utilities::kernel_validation::validate_kernel(kernel, true)?;

	// Since ACOSC has no parameters, we always have 1 row
	let rows = 1;
	let cols = high_slice.len();

	// Pre-allocate NumPy arrays
	let out_osc = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let out_change = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let slice_osc = unsafe { out_osc.as_slice_mut()? };
	let slice_change = unsafe { out_change.as_slice_mut()? };

	// Heavy work without the GIL
	py.allow_threads(|| -> Result<(), AcoscError> {
		// Resolve Kernel::Auto to a specific kernel
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

		// SAFETY: We must write to ALL elements before returning to Python
		// 1. Write NaN prefix to the first 38 elements (ACOSC warmup period)
		const WARMUP_PERIOD: usize = 38; // PERIOD_SMA34 + PERIOD_SMA5 - 1
		if high_slice.len() >= WARMUP_PERIOD {
			slice_osc[..WARMUP_PERIOD].fill(f64::NAN);
			slice_change[..WARMUP_PERIOD].fill(f64::NAN);
		} else {
			// If data is shorter than warmup, fill everything with NaN
			slice_osc.fill(f64::NAN);
			slice_change.fill(f64::NAN);
		}

		// 2. acosc_compute_into MUST write to all elements from index 38 onwards
		// This is guaranteed by the ACOSC algorithm implementation
		acosc_compute_into(high_slice, low_slice, simd, slice_osc, slice_change);
		Ok(())
	})
	.map_err(|e| PyValueError::new_err(e.to_string()))?;

	// Build dict with the GIL
	let dict = PyDict::new(py);
	dict.set_item("osc", out_osc.reshape((rows, cols))?)?;
	dict.set_item("change", out_change.reshape((rows, cols))?)?;

	Ok(dict)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn acosc_js(high: &[f64], low: &[f64]) -> Result<Vec<f64>, JsValue> {
	let params = AcoscParams::default();
	let input = AcoscInput::from_slices(high, low, params);

	acosc_with_kernel(&input, Kernel::Scalar)
		.map(|o| {
			// Flatten osc and change into a single vector
			let mut result = Vec::with_capacity(o.osc.len() * 2);
			result.extend_from_slice(&o.osc);
			result.extend_from_slice(&o.change);
			result
		})
		.map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn acosc_batch_js(high: &[f64], low: &[f64]) -> Result<Vec<f64>, JsValue> {
	// Since ACOSC has no parameters, batch is the same as single
	acosc_js(high, low)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn acosc_batch_metadata_js() -> Result<Vec<f64>, JsValue> {
	// Since ACOSC has no parameters, return empty metadata
	Ok(vec![])
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_acosc_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let default_params = AcoscParams::default();
		let input = AcoscInput::from_candles(&candles, default_params);
		let output = acosc_with_kernel(&input, kernel)?;
		assert_eq!(output.osc.len(), candles.close.len());
		assert_eq!(output.change.len(), candles.close.len());
		Ok(())
	}

	fn check_acosc_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = AcoscInput::with_default_candles(&candles);
		let result = acosc_with_kernel(&input, kernel)?;
		assert_eq!(result.osc.len(), candles.close.len());
		assert_eq!(result.change.len(), candles.close.len());
		let expected_last_five_acosc_osc = [273.30, 383.72, 357.7, 291.25, 176.84];
		let expected_last_five_acosc_change = [49.6, 110.4, -26.0, -66.5, -114.4];
		let start = result.osc.len().saturating_sub(5);
		for (i, &val) in result.osc[start..].iter().enumerate() {
			assert!(
				(val - expected_last_five_acosc_osc[i]).abs() < 1e-1,
				"[{}] ACOSC {:?} osc mismatch idx {}: got {}, expected {}",
				test_name,
				kernel,
				i,
				val,
				expected_last_five_acosc_osc[i]
			);
		}
		for (i, &val) in result.change[start..].iter().enumerate() {
			assert!(
				(val - expected_last_five_acosc_change[i]).abs() < 1e-1,
				"[{}] ACOSC {:?} change mismatch idx {}: got {}, expected {}",
				test_name,
				kernel,
				i,
				val,
				expected_last_five_acosc_change[i]
			);
		}
		Ok(())
	}

	fn check_acosc_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = AcoscInput::with_default_candles(&candles);
		match input.data {
			AcoscData::Candles { .. } => {}
			_ => panic!("Expected AcoscData::Candles variant"),
		}
		let output = acosc_with_kernel(&input, kernel)?;
		assert_eq!(output.osc.len(), candles.close.len());
		Ok(())
	}

	fn check_acosc_too_short(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [100.0, 101.0];
		let low = [99.0, 98.0];
		let params = AcoscParams::default();
		let input = AcoscInput::from_slices(&high, &low, params);
		let result = acosc_with_kernel(&input, kernel);
		assert!(result.is_err(), "[{}] Should fail with not enough data", test_name);
		Ok(())
	}

	fn check_acosc_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = AcoscInput::with_default_candles(&candles);
		let first_result = acosc_with_kernel(&input, kernel)?;
		assert_eq!(first_result.osc.len(), candles.close.len());
		assert_eq!(first_result.change.len(), candles.close.len());
		let input2 = AcoscInput::from_slices(&candles.high, &candles.low, AcoscParams::default());
		let second_result = acosc_with_kernel(&input2, kernel)?;
		assert_eq!(second_result.osc.len(), candles.close.len());
		for (a, b) in second_result.osc.iter().zip(first_result.osc.iter()) {
			if a.is_nan() && b.is_nan() {
				continue;
			}
			assert!((a - b).abs() < 1e-8, "Reinput values mismatch: {} vs {}", a, b);
		}
		Ok(())
	}

	fn check_acosc_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = AcoscInput::with_default_candles(&candles);
		let result = acosc_with_kernel(&input, kernel)?;
		if result.osc.len() > 240 {
			for i in 240..result.osc.len() {
				assert!(!result.osc[i].is_nan(), "Found NaN in osc at {}", i);
				assert!(!result.change[i].is_nan(), "Found NaN in change at {}", i);
			}
		}
		Ok(())
	}

	fn check_acosc_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = AcoscInput::with_default_candles(&candles);
		let batch = acosc_with_kernel(&input, kernel)?;
		let mut stream = AcoscStream::try_new(AcoscParams::default())?;
		let mut osc_stream = Vec::with_capacity(candles.close.len());
		let mut change_stream = Vec::with_capacity(candles.close.len());
		for (&h, &l) in candles.high.iter().zip(candles.low.iter()) {
			match stream.update(h, l) {
				Some((o, c)) => {
					osc_stream.push(o);
					change_stream.push(c);
				}
				None => {
					osc_stream.push(f64::NAN);
					change_stream.push(f64::NAN);
				}
			}
		}
		assert_eq!(batch.osc.len(), osc_stream.len());
		assert_eq!(batch.change.len(), change_stream.len());
		for (i, (&a, &b)) in batch.osc.iter().zip(osc_stream.iter()).enumerate() {
			if a.is_nan() && b.is_nan() {
				continue;
			}
			assert!(
				(a - b).abs() < 1e-9,
				"Streaming osc mismatch at idx {}: {} vs {}",
				i,
				a,
				b
			);
		}
		for (i, (&a, &b)) in batch.change.iter().zip(change_stream.iter()).enumerate() {
			if a.is_nan() && b.is_nan() {
				continue;
			}
			assert!(
				(a - b).abs() < 1e-9,
				"Streaming change mismatch at idx {}: {} vs {}",
				i,
				a,
				b
			);
		}
		Ok(())
	}

	macro_rules! generate_all_acosc_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(#[test]
                  fn [<$test_fn _scalar_f64>]() {
                      let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar);
                  })*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(#[test]
                  fn [<$test_fn _avx2_f64>]() {
                      let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2);
                  }
                  #[test]
                  fn [<$test_fn _avx512_f64>]() {
                      let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512);
                  })*
            }
        }
    }
	generate_all_acosc_tests!(
		check_acosc_partial_params,
		check_acosc_accuracy,
		check_acosc_default_candles,
		check_acosc_too_short,
		check_acosc_reinput,
		check_acosc_nan_handling,
		check_acosc_streaming,
		check_acosc_no_poison
	);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let output = AcoscBatchBuilder::new().kernel(kernel).apply_candles(&c)?;
		assert_eq!(output.osc.len(), c.close.len());
		Ok(())
	}

	// Debug mode test to check for poison values
	#[cfg(debug_assertions)]
	fn check_acosc_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = AcoscInput::with_default_candles(&candles);
		let output = acosc_with_kernel(&input, kernel)?;

		// Check osc values for poison patterns
		for (i, &val) in output.osc.iter().enumerate() {
			if val.is_nan() {
				continue;
			}

			let bits = val.to_bits();

			// Check for alloc_with_nan_prefix poison (0x11111111_11111111)
			if bits == 0x11111111_11111111 {
				panic!(
					"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in osc",
					test_name, val, bits, i
				);
			}
		}

		// Check change values for poison patterns
		for (i, &val) in output.change.iter().enumerate() {
			if val.is_nan() {
				continue;
			}

			let bits = val.to_bits();

			// Check for alloc_with_nan_prefix poison (0x11111111_11111111)
			if bits == 0x11111111_11111111 {
				panic!(
					"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in change",
					test_name, val, bits, i
				);
			}
		}

		Ok(())
	}

	// Debug mode test for batch operations
	#[cfg(debug_assertions)]
	fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let output = AcoscBatchBuilder::new().kernel(kernel).apply_candles(&c)?;

		// Check osc values for poison patterns
		for (idx, &val) in output.osc.iter().enumerate() {
			if val.is_nan() {
				continue;
			}

			let bits = val.to_bits();

			// Check for alloc_with_nan_prefix poison (0x11111111_11111111)
			if bits == 0x11111111_11111111 {
				panic!(
					"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in osc",
					test, val, bits, idx
				);
			}

			// Check for init_matrix_prefixes poison (0x22222222_22222222)
			if bits == 0x22222222_22222222 {
				panic!(
					"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in osc",
					test, val, bits, idx
				);
			}

			// Check for make_uninit_matrix poison (0x33333333_33333333)
			if bits == 0x33333333_33333333 {
				panic!(
					"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in osc",
					test, val, bits, idx
				);
			}
		}

		// Check change values for poison patterns
		for (idx, &val) in output.change.iter().enumerate() {
			if val.is_nan() {
				continue;
			}

			let bits = val.to_bits();

			// Check for alloc_with_nan_prefix poison (0x11111111_11111111)
			if bits == 0x11111111_11111111 {
				panic!(
					"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in change",
					test, val, bits, idx
				);
			}

			// Check for init_matrix_prefixes poison (0x22222222_22222222)
			if bits == 0x22222222_22222222 {
				panic!(
					"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in change",
					test, val, bits, idx
				);
			}

			// Check for make_uninit_matrix poison (0x33333333_33333333)
			if bits == 0x33333333_33333333 {
				panic!(
					"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in change",
					test, val, bits, idx
				);
			}
		}

		Ok(())
	}

	// Release mode stubs
	#[cfg(not(debug_assertions))]
	fn check_acosc_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(())
	}

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
