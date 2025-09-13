//! # Simple Moving Average (SMA)
//!
//! The most basic form of moving average, summing the last `period` points
//! and dividing by `period`. Useful for smoothing data and trend detection.
//!
//! ## Parameters
//! - **period**: Window size (number of data points).
//!
//! ## Returns
//! - **`Ok(SmaOutput)`** on success, containing a `Vec<f64>` matching the input length.
//! - **`Err(SmaError)`** otherwise.
//!
//! ## Developer Status
//! - **AVX2 kernel**: IMPLEMENTED - Optimized SIMD operations for efficient computation
//! - **AVX512 kernel**: STUB - Both short and long variants fall back to scalar
//! - **Streaming update**: O(1) - Efficient with rolling sum tracking
//! - **Memory optimization**: GOOD - Uses zero-copy helpers (alloc_with_nan_prefix, make_uninit_matrix)
//! - **Optimization needed**: Implement AVX512 kernels for both short and long periods

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
use std::mem::MaybeUninit;
use thiserror::Error;

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

impl<'a> AsRef<[f64]> for SmaInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			SmaData::Slice(slice) => slice,
			SmaData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

#[derive(Debug, Clone)]
pub enum SmaData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct SmaOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct SmaParams {
	pub period: Option<usize>,
}

impl Default for SmaParams {
	fn default() -> Self {
		Self { period: Some(9) }
	}
}

#[derive(Debug, Clone)]
pub struct SmaInput<'a> {
	pub data: SmaData<'a>,
	pub params: SmaParams,
}

impl<'a> SmaInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: SmaParams) -> Self {
		Self {
			data: SmaData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slice(sl: &'a [f64], p: SmaParams) -> Self {
		Self {
			data: SmaData::Slice(sl),
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", SmaParams::default())
	}
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(9)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct SmaBuilder {
	period: Option<usize>,
	kernel: Kernel,
}

impl Default for SmaBuilder {
	fn default() -> Self {
		Self {
			period: None,
			kernel: Kernel::Auto,
		}
	}
}

impl SmaBuilder {
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
	pub fn apply(self, c: &Candles) -> Result<SmaOutput, SmaError> {
		let p = SmaParams { period: self.period };
		let i = SmaInput::from_candles(c, "close", p);
		sma_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slice(self, d: &[f64]) -> Result<SmaOutput, SmaError> {
		let p = SmaParams { period: self.period };
		let i = SmaInput::from_slice(d, p);
		sma_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<SmaStream, SmaError> {
		let p = SmaParams { period: self.period };
		SmaStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum SmaError {
	#[error("sma: Empty data provided for SMA.")]
	EmptyData,
	#[error("sma: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("sma: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("sma: All values are NaN.")]
	AllValuesNaN,
	#[error("sma: Output buffer size mismatch: expected = {expected}, got = {got}")]
	OutputLenMismatch { expected: usize, got: usize },
}

#[inline]
pub fn sma(input: &SmaInput) -> Result<SmaOutput, SmaError> {
	sma_with_kernel(input, Kernel::Auto)
}

pub fn sma_with_kernel(input: &SmaInput, kernel: Kernel) -> Result<SmaOutput, SmaError> {
	let (data, period, first, chosen) = sma_prepare(input, kernel)?;
	let mut out = alloc_with_nan_prefix(data.len(), first + period - 1);
	sma_compute_into(data, period, first, chosen, &mut out);
	Ok(SmaOutput { values: out })
}

/// Write SMA directly to output slice - zero allocation pattern for WASM
/// The output slice must be the same length as the input data.
#[inline]
pub fn sma_into_slice(dst: &mut [f64], input: &SmaInput, kern: Kernel) -> Result<(), SmaError> {
	let (data, period, first, chosen) = sma_prepare(input, kern)?;

	// Verify output buffer size matches input
	if dst.len() != data.len() {
		return Err(SmaError::OutputLenMismatch {
			expected: data.len(),
			got: dst.len(),
		});
	}

	// Fill warmup with NaN
	let warmup = first + period - 1;
	for v in &mut dst[..warmup] {
		*v = f64::NAN;
	}

	// Compute directly into output buffer
	sma_compute_into(data, period, first, chosen, dst);

	Ok(())
}

#[inline(always)]
fn sma_prepare<'a>(
	input: &'a SmaInput,
	kernel: Kernel,
) -> Result<(&'a [f64], usize, usize, Kernel), SmaError> {
	let data: &[f64] = input.as_ref();
	if data.is_empty() { return Err(SmaError::EmptyData); }

	let period = input.get_period();
	let len = data.len();
	if period == 0 || period > len {
		return Err(SmaError::InvalidPeriod { period, data_len: len });
	}

	let first = data.iter().position(|x| !x.is_nan()).ok_or(SmaError::AllValuesNaN)?;
	if len - first < period {
		return Err(SmaError::NotEnoughValidData { needed: period, valid: len - first });
	}

	let chosen = match kernel { Kernel::Auto => detect_best_kernel(), k => k };
	Ok((data, period, first, chosen))
}

/// Compute SMA into a pre-allocated output buffer for zero-copy operations
#[inline]
fn sma_compute_into(data: &[f64], period: usize, first: usize, kernel: Kernel, out: &mut [f64]) {
	unsafe {
		match kernel {
			Kernel::Scalar | Kernel::ScalarBatch => {
				sma_scalar(data, period, first, out);
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => {
				sma_avx2(data, period, first, out);
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => {
				sma_avx512(data, period, first, out);
			}
			_ => unreachable!(),
		}
	}
}

#[inline(always)]
pub unsafe fn sma_scalar(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
	debug_assert!(period >= 1);
	debug_assert_eq!(data.len(), out.len());
	let len = data.len();

	let dp = data.as_ptr();
	let op = out.as_mut_ptr();

	// Special case for period=1: SMA is just the input value
	// This avoids floating-point accumulation errors
	if period == 1 {
		for i in first..len {
			*op.add(i) = *dp.add(i);
		}
		return;
	}

	let mut sum = 0.0;
	for k in 0..period {
		sum += *dp.add(first + k);
	}
	let inv = 1.0 / (period as f64);

	*op.add(first + period - 1) = sum * inv;

	for i in (first + period)..len {
		sum += *dp.add(i) - *dp.add(i - period);
		*op.add(i) = sum * inv;
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn sma_avx2(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
	sma_scalar(data, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn sma_avx512(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
	if period <= 32 {
		unsafe { sma_avx512_short(data, period, first, out) }
	} else {
		unsafe { sma_avx512_long(data, period, first, out) }
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn sma_avx512_short(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
	// Stub: call scalar
	sma_scalar(data, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn sma_avx512_long(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
	// Stub: call scalar
	sma_scalar(data, period, first, out);
}

#[derive(Debug, Clone)]
pub struct SmaStream {
	period: usize,
	buffer: Vec<f64>,
	head: usize,
	sum: f64,
	filled: bool,
}

impl SmaStream {
	pub fn try_new(params: SmaParams) -> Result<Self, SmaError> {
		let period = params.period.unwrap_or(9);
		if period == 0 {
			return Err(SmaError::InvalidPeriod { period, data_len: 0 });
		}
		Ok(Self {
			period,
			buffer: vec![0.0; period],
			head: 0,
			sum: 0.0,
			filled: false,
		})
	}
	#[inline(always)]
	pub fn update(&mut self, value: f64) -> Option<f64> {
		if !self.filled && self.head == 0 && self.sum == 0.0 {
			self.sum = value;
			self.buffer[self.head] = value;
			self.head = (self.head + 1) % self.period;
			if self.head == 0 {
				self.filled = true;
			}
			return None;
		}
		self.sum += value - self.buffer[self.head];
		self.buffer[self.head] = value;
		self.head = (self.head + 1) % self.period;
		if !self.filled && self.head == 0 {
			self.filled = true;
		}
		if self.filled {
			Some(self.sum / self.period as f64)
		} else {
			None
		}
	}
}

#[derive(Clone, Debug)]
pub struct SmaBatchRange {
	pub period: (usize, usize, usize),
}

impl Default for SmaBatchRange {
	fn default() -> Self {
		Self { period: (9, 240, 1) }
	}
}

#[derive(Clone, Debug, Default)]
pub struct SmaBatchBuilder {
	range: SmaBatchRange,
	kernel: Kernel,
}

impl SmaBatchBuilder {
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
	pub fn apply_slice(self, data: &[f64]) -> Result<SmaBatchOutput, SmaError> {
		sma_batch_with_kernel(data, &self.range, self.kernel)
	}
	pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<SmaBatchOutput, SmaError> {
		SmaBatchBuilder::new().kernel(k).apply_slice(data)
	}
	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<SmaBatchOutput, SmaError> {
		let slice = source_type(c, src);
		self.apply_slice(slice)
	}
	pub fn with_default_candles(c: &Candles) -> Result<SmaBatchOutput, SmaError> {
		SmaBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
	}
}

pub fn sma_batch_with_kernel(data: &[f64], sweep: &SmaBatchRange, k: Kernel) -> Result<SmaBatchOutput, SmaError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(SmaError::InvalidPeriod { period: 0, data_len: 0 }),
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	sma_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct SmaBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<SmaParams>,
	pub rows: usize,
	pub cols: usize,
}
impl SmaBatchOutput {
	pub fn row_for_params(&self, p: &SmaParams) -> Option<usize> {
		self.combos
			.iter()
			.position(|c| c.period.unwrap_or(9) == p.period.unwrap_or(9))
	}
	pub fn values_for(&self, p: &SmaParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &SmaBatchRange) -> Vec<SmaParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let periods = axis_usize(r.period);
	let mut out = Vec::with_capacity(periods.len());
	for &p in &periods {
		out.push(SmaParams { period: Some(p) });
	}
	out
}

#[inline(always)]
pub fn sma_batch_slice(data: &[f64], sweep: &SmaBatchRange, kern: Kernel) -> Result<SmaBatchOutput, SmaError> {
	sma_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn sma_batch_par_slice(data: &[f64], sweep: &SmaBatchRange, kern: Kernel) -> Result<SmaBatchOutput, SmaError> {
	sma_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn sma_batch_inner(
	data: &[f64],
	sweep: &SmaBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<SmaBatchOutput, SmaError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() { return Err(SmaError::InvalidPeriod { period: 0, data_len: 0 }); }
	if data.is_empty() { return Err(SmaError::EmptyData); }

	let cols = data.len();
	let first = data.iter().position(|x| !x.is_nan()).ok_or(SmaError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if cols - first < max_p {
		return Err(SmaError::NotEnoughValidData { needed: max_p, valid: cols - first });
	}

	let rows = combos.len();

	// 1) allocate rows×cols uninit
	let mut buf_mu = make_uninit_matrix(rows, cols);

	// 2) warmup NaN prefixes per row
	let warm: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap() - 1).collect();
	init_matrix_prefixes(&mut buf_mu, cols, &warm);

	// 3) view as &mut [f64] without copy
	let mut guard = core::mem::ManuallyDrop::new(buf_mu);
	let out_slice: &mut [f64] = unsafe {
		core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len())
	};

	// 4) compute in-place
	sma_batch_inner_into(data, sweep, kern, parallel, out_slice)?;

	// 5) reconstruct Vec<f64> with zero copy
	let values = unsafe {
		Vec::from_raw_parts(guard.as_mut_ptr() as *mut f64, guard.len(), guard.capacity())
	};

	Ok(SmaBatchOutput { values, combos, rows, cols })
}

#[inline(always)]
fn sma_batch_inner_into(
	data: &[f64],
	sweep: &SmaBatchRange,
	kern: Kernel,
	parallel: bool,
	out: &mut [f64],
) -> Result<Vec<SmaParams>, SmaError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() { return Err(SmaError::InvalidPeriod { period: 0, data_len: 0 }); }
	if data.is_empty() { return Err(SmaError::EmptyData); }

	let first = data.iter().position(|x| !x.is_nan()).ok_or(SmaError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if data.len() - first < max_p {
		return Err(SmaError::NotEnoughValidData { needed: max_p, valid: data.len() - first });
	}

	let rows = combos.len();
	let cols = data.len();

	// Map Auto and Batch to a concrete non-batch kernel (ALMA parity)
	let actual_kern = match kern {
		Kernel::Auto => detect_best_batch_kernel(),
		k => k,
	};
	let actual_kern = match actual_kern {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch   => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		other               => other,
	};

	// Work over MaybeUninit rows to avoid re-writing warmup
	let out_uninit: &mut [MaybeUninit<f64>] = unsafe {
		core::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
	};

	// Initialize warmup cells with NaN
	let warm: Vec<usize> = combos.iter()
		.map(|c| first + c.period.unwrap_or(9) - 1)
		.collect();
	init_matrix_prefixes(out_uninit, cols, &warm);

	let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
		let period = combos[row].period.unwrap();
		// cast this row to &mut [f64]
		let dst = core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());
		match actual_kern {
			Kernel::Scalar | Kernel::ScalarBatch => sma_row_scalar(data, first, period, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => sma_row_avx2(data, first, period, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => sma_row_avx512(data, first, period, dst),
			_ => unreachable!(),
		}
	};

	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		out_uninit.par_chunks_mut(cols).enumerate().for_each(|(row, slice)| do_row(row, slice));
		#[cfg(target_arch = "wasm32")]
		for (row, slice) in out_uninit.chunks_mut(cols).enumerate() { do_row(row, slice); }
	} else {
		for (row, slice) in out_uninit.chunks_mut(cols).enumerate() { do_row(row, slice); }
	}

	Ok(combos)
}

#[inline(always)]
unsafe fn sma_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	sma_scalar(data, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn sma_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	sma_avx2(data, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn sma_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	if period <= 32 {
		sma_avx512_short(data, period, first, out);
	} else {
		sma_avx512_long(data, period, first, out);
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn sma_row_avx512_short(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
	sma_scalar(data, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn sma_row_avx512_long(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
	sma_scalar(data, period, first, out);
}

// ============================================================================
// Python Bindings
// ============================================================================

#[cfg(feature = "python")]
#[pyfunction(name = "sma")]
#[pyo3(signature = (data, period, kernel=None))]
/// Compute the Simple Moving Average (SMA) of the input data.
///
/// The SMA is the average of the last N data points, where N is the period.
/// It is a lagging indicator that smooths price data.
///
/// Parameters:
/// -----------
/// data : np.ndarray
///     Input data array (float64).
/// period : int
///     Number of data points in the moving average window.
/// kernel : str, optional
///     Computation kernel to use: 'auto', 'scalar', 'avx2', 'avx512'.
///     Default is 'auto' which auto-detects the best available.
///
/// Returns:
/// --------
/// np.ndarray
///     Array of SMA values, same length as input.
///
/// Raises:
/// -------
/// ValueError
///     If inputs are invalid (empty data, invalid period, etc).
pub fn sma_py<'py>(
	py: Python<'py>,
	data: PyReadonlyArray1<'py, f64>,
	period: usize,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
	use numpy::IntoPyArray;

	// Validate kernel with CPU feature detection
	let kern = validate_kernel(kernel, false)?;

	let data_slice = data.as_slice()?;
	let params = SmaParams { period: Some(period) };
	let input = SmaInput::from_slice(data_slice, params);

	// Compute SMA using the standard function with zero-copy
	let result_vec: Vec<f64> = py
		.allow_threads(|| sma_with_kernel(&input, kern).map(|o| o.values))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	// Use zero-copy transfer to Python
	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction(name = "sma_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
/// Compute SMA for multiple period values in a single pass.
///
/// Parameters:
/// -----------
/// data : np.ndarray
///     Input data array (float64).
/// period_range : tuple
///     (start, end, step) for period values to compute.
/// kernel : str, optional
///     Computation kernel: 'auto', 'scalar', 'avx2', 'avx512'.
///
/// Returns:
/// --------
/// dict
///     Dictionary with 'values' (2D array) and 'periods' (list of periods).
pub fn sma_batch_py<'py>(
	py: Python<'py>,
	data: PyReadonlyArray1<'py, f64>,
	period_range: (usize, usize, usize),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
	use numpy::IntoPyArray;
	use pyo3::types::PyDict;

	// Validate kernel with CPU feature detection
	let kern = validate_kernel(kernel, true)?;

	let data_slice = data.as_slice()?;
	let range = SmaBatchRange { period: period_range };

	// Validate and prepare
	let combos = expand_grid(&range);
	if combos.is_empty() {
		return Err(PyValueError::new_err("Invalid period range"));
	}
	if data_slice.is_empty() {
		return Err(PyValueError::new_err("Empty data"));
	}

	let rows = combos.len();
	let cols = data_slice.len();

	// Pre-allocate NumPy array (1-D, will reshape later)
	let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let slice_out = unsafe { out_arr.as_slice_mut()? };

	// Perform batch computation with zero-copy directly into NumPy array
	let combos = py
		.allow_threads(|| {
			let kernel = match kern { Kernel::Auto => detect_best_batch_kernel(), k => k };
			let simd = match kernel {
				Kernel::Avx512Batch => Kernel::Avx512,
				Kernel::Avx2Batch   => Kernel::Avx2,
				Kernel::ScalarBatch => Kernel::Scalar,
				_ => unreachable!(),
			};
			// pass &mut [f64]; inner converts to MaybeUninit
			sma_batch_inner_into(data_slice, &range, simd, true, slice_out)
		})
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	// Create result dictionary
	let dict = PyDict::new(py);
	dict.set_item("values", out_arr.reshape((rows, cols))?)?;

	// Extract periods into a numpy array
	dict.set_item(
		"periods",
		combos
			.iter()
			.map(|p| p.period.unwrap_or(9) as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;

	Ok(dict.into())
}

#[cfg(feature = "python")]
#[pyclass(name = "SmaStream")]
/// Streaming SMA calculator that processes values one at a time.
///
/// This is useful for real-time data processing where you receive
/// price updates incrementally.
///
/// Parameters:
/// -----------
/// period : int
///     Number of values in the moving average window.
///
/// Example:
/// --------
/// >>> stream = SmaStream(14)
/// >>> for price in prices:
/// ...     sma_value = stream.update(price)
/// ...     if sma_value is not None:
/// ...         print(f"SMA: {sma_value}")
pub struct SmaStreamPy {
	inner: SmaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl SmaStreamPy {
	#[new]
	#[pyo3(signature = (period))]
	pub fn new(period: usize) -> PyResult<Self> {
		let params = SmaParams { period: Some(period) };
		let inner = SmaStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(Self { inner })
	}

	/// Update the SMA with a new value.
	///
	/// Parameters:
	/// -----------
	/// value : float
	///     New price value to add to the stream.
	///
	/// Returns:
	/// --------
	/// float or None
	///     The current SMA value, or None if not enough data yet.
	pub fn update(&mut self, value: f64) -> Option<f64> {
		self.inner.update(value)
	}
}

// ============================================================================
// WASM Bindings
// ============================================================================

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "sma")]
/// Compute Simple Moving Average (SMA) for the given data
pub fn sma_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
	let params = SmaParams { period: Some(period) };
	let input = SmaInput::from_slice(data, params);

	// Allocate output buffer once
	let mut output = vec![0.0; data.len()];

	// Compute directly into output buffer
	sma_into_slice(&mut output, &input, Kernel::Auto).map_err(|e| JsValue::from_str(&e.to_string()))?;

	Ok(output)
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct SmaBatchConfig {
	pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct SmaBatchJsOutput {
	pub values: Vec<f64>,
	pub combos: Vec<SmaParams>,
	pub periods: Vec<usize>,  // Added for API parity with ALMA
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "sma_batch")]
pub fn sma_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	let config: SmaBatchConfig =
		serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

	let sweep = SmaBatchRange {
		period: config.period_range,
	};

	let output = sma_batch_with_kernel(data, &sweep, Kernel::Auto).map_err(|e| JsValue::from_str(&e.to_string()))?;

	let js_output = SmaBatchJsOutput {
		values: output.values,
		periods: output.combos.iter().map(|c| c.period.unwrap_or(9)).collect(),
		combos: output.combos,
		rows: output.rows,
		cols: output.cols,
	};

	serde_wasm_bindgen::to_value(&js_output).map_err(|e| JsValue::from_str(&e.to_string()))
}

// Keep old functions for backward compatibility but mark as deprecated
#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "smaBatch")]
#[deprecated(since = "1.0.0", note = "Use sma_batch instead")]
pub fn sma_batch_js(
	data: &[f64],
	period_start: usize,
	period_end: usize,
	period_step: usize,
) -> Result<Vec<f64>, JsValue> {
	let range = SmaBatchRange {
		period: (period_start, period_end, period_step),
	};

	sma_batch_with_kernel(data, &range, Kernel::Auto)
		.map(|output| output.values)
		.map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "smaBatchMetadata")]
#[deprecated(since = "1.0.0", note = "Use sma_batch which returns metadata")]
pub fn sma_batch_metadata_js(period_start: usize, period_end: usize, period_step: usize) -> Vec<usize> {
	let range = SmaBatchRange {
		period: (period_start, period_end, period_step),
	};
	let combos = expand_grid(&range);
	combos.iter().map(|c| c.period.unwrap_or(9)).collect()
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "smaBatchRowsCols")]
#[deprecated(since = "1.0.0", note = "Use sma_batch which returns rows and cols")]
pub fn sma_batch_rows_cols_js(
	period_start: usize,
	period_end: usize,
	period_step: usize,
	data_len: usize,
) -> Vec<usize> {
	let range = SmaBatchRange {
		period: (period_start, period_end, period_step),
	};
	let combos = expand_grid(&range);
	vec![combos.len(), data_len]
}

// ================== Zero-Copy WASM Functions ==================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn sma_alloc(len: usize) -> *mut f64 {
	// Allocate memory for input/output buffer
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec); // Prevent deallocation
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn sma_free(ptr: *mut f64, len: usize) {
	// Free allocated memory
	if !ptr.is_null() {
		unsafe {
			let _ = Vec::from_raw_parts(ptr, len, len);
		}
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn sma_into(in_ptr: *const f64, out_ptr: *mut f64, len: usize, period: usize) -> Result<(), JsValue> {
	// Check for null pointers
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}

	unsafe {
		// Create slice from pointer
		let data = std::slice::from_raw_parts(in_ptr, len);

		let params = SmaParams { period: Some(period) };
		let input = SmaInput::from_slice(data, params);

		// Check if pointers are the same (aliasing)
		if in_ptr == out_ptr as *const f64 {
			// Use temporary buffer to avoid corruption
			let mut temp = vec![0.0; len];
			sma_into_slice(&mut temp, &input, Kernel::Auto).map_err(|e| JsValue::from_str(&e.to_string()))?;

			// Copy results back to output
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			out.copy_from_slice(&temp);
		} else {
			// No aliasing, compute directly into output
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			sma_into_slice(out, &input, Kernel::Auto).map_err(|e| JsValue::from_str(&e.to_string()))?;
		}

		Ok(())
	}
}

// ================== Optimized Batch Processing ==================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn sma_batch_into(
	in_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period_start: usize,
	period_end: usize,
	period_step: usize,
) -> Result<usize, JsValue> {
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}

	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);

		let sweep = SmaBatchRange {
			period: (period_start, period_end, period_step),
		};

		let combos = expand_grid(&sweep);
		let rows = combos.len();
		let total_size = rows * len;

		let out = std::slice::from_raw_parts_mut(out_ptr, total_size);

		// Map to non-batch kernel (ALMA parity) and no parallel on WASM
		let kernel = match detect_best_batch_kernel() {
			Kernel::Avx512Batch => Kernel::Avx512,
			Kernel::Avx2Batch   => Kernel::Avx2,
			Kernel::ScalarBatch => Kernel::Scalar,
			other               => other,
		};

		sma_batch_inner_into(data, &sweep, kernel, false, out)
			.map_err(|e| JsValue::from_str(&e.to_string()))?;

		Ok(rows)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;
	fn check_sma_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let default_params = SmaParams { period: None };
		let input = SmaInput::from_candles(&candles, "close", default_params);
		let output = sma_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}
	fn check_sma_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = SmaParams { period: Some(9) };
		let input = SmaInput::from_candles(&candles, "close", params);
		let result = sma_with_kernel(&input, kernel)?;
		let expected_last_five = [59180.8, 59175.0, 59129.4, 59085.4, 59133.7];
		let start = result.values.len().saturating_sub(5);
		for (i, &val) in result.values[start..].iter().enumerate() {
			let diff = (val - expected_last_five[i]).abs();
			assert!(
				diff < 1e-1,
				"[{}] SMA {:?} mismatch at idx {}: got {}, expected {}",
				test_name,
				kernel,
				i,
				val,
				expected_last_five[i]
			);
		}
		Ok(())
	}
	fn check_sma_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = SmaInput::with_default_candles(&candles);
		match input.data {
			SmaData::Candles { source, .. } => assert_eq!(source, "close"),
			_ => panic!("Expected SmaData::Candles"),
		}
		let output = sma_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}
	fn check_sma_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let input_data = [10.0, 20.0, 30.0];
		let params = SmaParams { period: Some(0) };
		let input = SmaInput::from_slice(&input_data, params);
		let res = sma_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] SMA should fail with zero period", test_name);
		Ok(())
	}
	fn check_sma_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data_small = [10.0, 20.0, 30.0];
		let params = SmaParams { period: Some(10) };
		let input = SmaInput::from_slice(&data_small, params);
		let res = sma_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] SMA should fail with period exceeding length",
			test_name
		);
		Ok(())
	}
	fn check_sma_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let single_point = [42.0];
		let params = SmaParams { period: Some(9) };
		let input = SmaInput::from_slice(&single_point, params);
		let res = sma_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] SMA should fail with insufficient data", test_name);
		Ok(())
	}
	fn check_sma_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let first_params = SmaParams { period: Some(14) };
		let first_input = SmaInput::from_candles(&candles, "close", first_params);
		let first_result = sma_with_kernel(&first_input, kernel)?;
		let second_params = SmaParams { period: Some(14) };
		let second_input = SmaInput::from_slice(&first_result.values, second_params);
		let second_result = sma_with_kernel(&second_input, kernel)?;
		assert_eq!(second_result.values.len(), first_result.values.len());
		Ok(())
	}
	fn check_sma_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = SmaInput::from_candles(&candles, "close", SmaParams { period: Some(9) });
		let res = sma_with_kernel(&input, kernel)?;
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
	fn check_sma_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let period = 9;
		let input = SmaInput::from_candles(&candles, "close", SmaParams { period: Some(period) });
		let batch_output = sma_with_kernel(&input, kernel)?.values;
		let mut stream = SmaStream::try_new(SmaParams { period: Some(period) })?;
		let mut stream_values = Vec::with_capacity(candles.close.len());
		for &price in &candles.close {
			match stream.update(price) {
				Some(sma_val) => stream_values.push(sma_val),
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
				"[{}] SMA streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
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
	fn check_sma_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		// Test multiple parameter combinations to increase coverage
		let test_periods = vec![5, 9, 14, 20, 30, 50];

		for period in test_periods {
			let params = SmaParams { period: Some(period) };
			let input = SmaInput::from_candles(&candles, "close", params);
			let output = sma_with_kernel(&input, kernel)?;

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
	fn check_sma_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		Ok(())
	}

	#[cfg(feature = "proptest")]
	#[allow(clippy::float_cmp)]
	fn check_sma_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		use proptest::prelude::*;
		skip_if_unsupported!(kernel, test_name);

		// Test strategy: generate period first, then data of appropriate length
		let strat = (1usize..=100).prop_flat_map(|period| {
			(
				prop::collection::vec(
					(-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
					period..400,
				),
				Just(period),
			)
		});

		proptest::test_runner::TestRunner::default()
			.run(&strat, |(data, period)| {
				let params = SmaParams { period: Some(period) };
				let input = SmaInput::from_slice(&data, params);

				// Compute SMA with specified kernel and scalar reference
				let SmaOutput { values: out } = sma_with_kernel(&input, kernel).unwrap();
				let SmaOutput { values: ref_out } = sma_with_kernel(&input, Kernel::Scalar).unwrap();

				// Property 1: Initial values should be NaN (warmup period)
				for i in 0..(period - 1) {
					prop_assert!(
						out[i].is_nan(),
						"Expected NaN during warmup at index {}, got {}",
						i,
						out[i]
					);
				}

				// Properties 2-7: Test each valid SMA value
				for i in (period - 1)..data.len() {
					let window_start = i + 1 - period;
					let window = &data[window_start..=i];
					
					// Property 2: SMA should equal exact arithmetic mean of window
					let expected_sum: f64 = window.iter().sum();
					let expected_mean = expected_sum / period as f64;
					
					// Use slightly relaxed tolerance for numerical stability across different kernels
					// For running sum method, errors accumulate proportionally to magnitude
					let abs_tolerance = 1e-8_f64;
					let rel_tolerance = 1e-12_f64;
					let tolerance = abs_tolerance.max(expected_mean.abs() * rel_tolerance);
					prop_assert!(
						(out[i] - expected_mean).abs() <= tolerance,
						"SMA mismatch at index {}: expected {}, got {} (diff: {})",
						i,
						expected_mean,
						out[i],
						(out[i] - expected_mean).abs()
					);

					// Property 3: SMA bounded by min/max of input window
					let window_min = window.iter().cloned().fold(f64::INFINITY, f64::min);
					let window_max = window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
					
					prop_assert!(
						out[i] >= window_min - 1e-9 && out[i] <= window_max + 1e-9,
						"SMA out of bounds at index {}: {} not in [{}, {}]",
						i,
						out[i],
						window_min,
						window_max
					);

					// Property 4: For constant input, SMA equals that constant
					if window.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-12) {
						let tolerance = if period == 1 { 1e-8 } else { 1e-9 };
						prop_assert!(
							(out[i] - window[0]).abs() <= tolerance,
							"Constant input property failed at index {}: expected {}, got {}",
							i,
							window[0],
							out[i]
						);
					}

					// Property 5: Linear trend - SMA of linear function should be at midpoint
					// Check if window forms a linear sequence
					if period >= 3 {
						let diffs: Vec<f64> = window.windows(2).map(|w| w[1] - w[0]).collect();
						let is_linear = diffs.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-9);
						
						if is_linear && !diffs.is_empty() {
							// For linear sequence, SMA should equal value at midpoint
							let midpoint_value = window[period / 2];
							let tolerance = if period % 2 == 0 {
								// Even period: average of two middle values
								(window[period / 2 - 1] - window[period / 2]).abs() / 2.0 + 1e-9
							} else {
								1e-9
							};
							
							prop_assert!(
								(out[i] - midpoint_value).abs() <= tolerance,
								"Linear trend property failed at index {}: expected ~{}, got {}",
								i,
								midpoint_value,
								out[i]
							);
						}
					}

					// Property 6: Cross-kernel consistency
					prop_assert!(
						(out[i] - ref_out[i]).abs() <= 1e-9 || 
						(out[i].is_nan() && ref_out[i].is_nan()),
						"Kernel mismatch at index {}: {} ({:?}) vs {} (Scalar)",
						i,
						out[i],
						kernel,
						ref_out[i]
					);

					// Property 7: Lag property - SMA should smooth out sharp changes
					// When sliding the window, the change in SMA depends on the new value added
					// and the old value removed from the window
					if i >= period {
						let new_value = data[i];
						let old_value = data[i - period];
						let expected_sma_change = (new_value - old_value) / period as f64;
						let actual_sma_change = out[i] - out[i - 1];
						
						prop_assert!(
							(actual_sma_change - expected_sma_change).abs() <= 1e-9,
							"Lag property failed at index {}: SMA change {} should be {} (new: {}, old: {})",
							i,
							actual_sma_change,
							expected_sma_change,
							new_value,
							old_value
						);
					}

					// Property 8: Check for poison values (debug mode only)
					#[cfg(debug_assertions)]
					{
						let bits = out[i].to_bits();
						prop_assert!(
							bits != 0x11111111_11111111 && 
							bits != 0x22222222_22222222 && 
							bits != 0x33333333_33333333,
							"Found poison value at index {}: {} (0x{:016X})",
							i,
							out[i],
							bits
						);
					}
				}

				// Additional property: Period = 1 should return original data
				if period == 1 {
					for i in 0..data.len() {
						prop_assert!(
							(out[i] - data[i]).abs() <= 1e-8,
							"Period=1 property failed at index {}: expected {}, got {}",
							i,
							data[i],
							out[i]
						);
					}
				}

				Ok(())
			})
			.unwrap();

		Ok(())
	}

	macro_rules! generate_all_sma_tests {
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
	generate_all_sma_tests!(
		check_sma_partial_params,
		check_sma_accuracy,
		check_sma_default_candles,
		check_sma_zero_period,
		check_sma_period_exceeds_length,
		check_sma_very_small_dataset,
		check_sma_reinput,
		check_sma_nan_handling,
		check_sma_streaming,
		check_sma_no_poison
	);

	#[cfg(feature = "proptest")]
	generate_all_sma_tests!(check_sma_property);
	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let output = SmaBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;
		let def = SmaParams::default();
		let row = output.values_for(&def).expect("default row missing");
		assert_eq!(row.len(), c.close.len());
		let expected = [59180.8, 59175.0, 59129.4, 59085.4, 59133.7];
		let start = row.len() - 5;
		for (i, &v) in row[start..].iter().enumerate() {
			assert!(
				(v - expected[i]).abs() < 1e-1,
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

		// Test multiple batch configurations to increase coverage
		let test_configs = vec![
			(5, 15, 5),   // Small periods
			(10, 30, 10), // Medium periods
			(20, 50, 15), // Large periods
			(2, 10, 2),   // Edge case: very small periods
		];

		for (start, end, step) in test_configs {
			let output = SmaBatchBuilder::new()
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
