//! # Weighted Moving Average (WMA)
//!
//! A moving average where each data point in the window is assigned a linearly increasing weight. The most recent values carry the highest weights, making the WMA more responsive to new data than a simple moving average.
//!
//! ## Parameters
//! - **period**: Window size (must be >= 2).
//!
//! ## Returns
//! - **`Ok(WmaOutput)`** on success, containing a `Vec<f64>` matching the input length.
//! - **`Err(WmaError)`** otherwise.
//!
//! ## Developer Status
//! - **AVX2 kernel**: STUB - Falls back to scalar implementation
//! - **AVX512 kernel**: STUB - Falls back to scalar implementation
//! - **Streaming update**: O(1) - Efficient weight accumulation approach
//! - **Memory optimization**: Uses zero-copy helpers (alloc_with_nan_prefix, make_uninit_matrix) ✓
//! - **Optimization needed**: Implement SIMD kernels for vectorized weight calculations
//! - **Note**: Streaming uses rolling window with precomputed weights

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
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

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

impl<'a> AsRef<[f64]> for WmaInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			WmaData::Slice(slice) => slice,
			WmaData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

#[derive(Debug, Clone)]
pub enum WmaData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct WmaOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct WmaParams {
	pub period: Option<usize>,
}

impl Default for WmaParams {
	fn default() -> Self {
		Self { period: Some(30) }
	}
}

#[derive(Debug, Clone)]
pub struct WmaInput<'a> {
	pub data: WmaData<'a>,
	pub params: WmaParams,
}

impl<'a> WmaInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: WmaParams) -> Self {
		Self {
			data: WmaData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slice(sl: &'a [f64], p: WmaParams) -> Self {
		Self {
			data: WmaData::Slice(sl),
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", WmaParams::default())
	}
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(30)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct WmaBuilder {
	period: Option<usize>,
	kernel: Kernel,
}

impl Default for WmaBuilder {
	fn default() -> Self {
		Self {
			period: None,
			kernel: Kernel::Auto,
		}
	}
}

impl WmaBuilder {
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
	pub fn apply(self, c: &Candles) -> Result<WmaOutput, WmaError> {
		let p = WmaParams { period: self.period };
		let i = WmaInput::from_candles(c, "close", p);
		wma_with_kernel(&i, self.kernel)
	}

	#[inline(always)]
	pub fn apply_slice(self, d: &[f64]) -> Result<WmaOutput, WmaError> {
		let p = WmaParams { period: self.period };
		let i = WmaInput::from_slice(d, p);
		wma_with_kernel(&i, self.kernel)
	}

	#[inline(always)]
	pub fn into_stream(self) -> Result<WmaStream, WmaError> {
		let p = WmaParams { period: self.period };
		WmaStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum WmaError {
	#[error("wma: Input data slice is empty.")]
	EmptyInputData,

	#[error("wma: All values are NaN.")]
	AllValuesNaN,

	#[error("wma: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },

	#[error("wma: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },

	#[error("wma: Invalid kernel - batch kernel required.")]
	InvalidKernel,
}

#[inline]
pub fn wma(input: &WmaInput) -> Result<WmaOutput, WmaError> {
	wma_with_kernel(input, Kernel::Auto)
}

pub fn wma_with_kernel(input: &WmaInput, kernel: Kernel) -> Result<WmaOutput, WmaError> {
	let (data, period, first, chosen) = wma_prepare(input, kernel)?;
	let len = data.len();
	let warm = first + period - 1;
	let mut out = alloc_with_nan_prefix(len, warm);

	wma_compute_into(data, period, first, chosen, &mut out);

	Ok(WmaOutput { values: out })
}

#[inline]
pub fn wma_into_slice(dst: &mut [f64], input: &WmaInput, kern: Kernel) -> Result<(), WmaError> {
	let (data, period, first, chosen) = wma_prepare(input, kern)?;

	if dst.len() != data.len() {
		return Err(WmaError::InvalidPeriod {
			period: dst.len(),
			data_len: data.len(),
		});
	}

	wma_compute_into(data, period, first, chosen, dst);

	let warmup_end = first + period - 1;
	for v in &mut dst[..warmup_end] {
		*v = f64::NAN;
	}

	Ok(())
}

fn wma_prepare<'a>(
	input: &'a WmaInput,
	kernel: Kernel,
) -> Result<
	(
		// data
		&'a [f64],
		// period
		usize,
		// first
		usize,
		// chosen
		Kernel,
	),
	WmaError,
> {
	let data: &[f64] = input.as_ref();
	let len = data.len();
	if len == 0 {
		return Err(WmaError::EmptyInputData);
	}

	let first = data.iter().position(|x| !x.is_nan()).ok_or(WmaError::AllValuesNaN)?;
	let period = input.get_period();

	if period < 2 || period > len {
		return Err(WmaError::InvalidPeriod { period, data_len: len });
	}
	if len - first < period {
		return Err(WmaError::NotEnoughValidData {
			needed: period,
			valid: len - first,
		});
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		k => k,
	};

	Ok((data, period, first, chosen))
}

#[inline(always)]
fn wma_compute_into(data: &[f64], period: usize, first: usize, kernel: Kernel, out: &mut [f64]) {
	unsafe {
		match kernel {
			Kernel::Scalar | Kernel::ScalarBatch => wma_scalar(data, period, first, out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => wma_avx2(data, period, first, out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => wma_avx512(data, period, first, out),
			_ => unreachable!(),
		}
	}
}

#[inline]
pub fn wma_scalar(data: &[f64], period: usize, first_val: usize, out: &mut [f64]) {
	let lookback = period - 1;
	let sum_of_weights = (period * (period + 1)) >> 1;
	let divider = sum_of_weights as f64;

	let mut weighted_sum = 0.0;
	let mut plain_sum = 0.0;

	for i in 0..lookback {
		let val = data[first_val + i];
		weighted_sum += (i as f64 + 1.0) * val;
		plain_sum += val;
	}

	let first_wma_idx = first_val + lookback;
	let val = data[first_wma_idx];
	weighted_sum += (period as f64) * val;
	plain_sum += val;

	out[first_wma_idx] = weighted_sum / divider;

	weighted_sum -= plain_sum;
	plain_sum -= data[first_val];

	for i in (first_wma_idx + 1)..data.len() {
		let val = data[i];
		weighted_sum += (period as f64) * val;
		plain_sum += val;

		out[i] = weighted_sum / divider;

		weighted_sum -= plain_sum;
		let old_val = data[i - lookback];
		plain_sum -= old_val;
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn wma_avx512(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	wma_scalar(data, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn wma_avx2(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	wma_scalar(data, period, first_valid, out)
}

#[inline]
pub fn wma_avx512_short(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	wma_scalar(data, period, first_valid, out)
}

#[inline]
pub fn wma_avx512_long(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	wma_scalar(data, period, first_valid, out)
}

#[inline(always)]
pub fn wma_with_kernel_batch(data: &[f64], sweep: &WmaBatchRange, k: Kernel) -> Result<WmaBatchOutput, WmaError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(WmaError::InvalidKernel),
	};

	let simd = match kernel {
		#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
		Kernel::Avx512Batch => Kernel::Avx512,
		#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => Kernel::Scalar,
	};
	wma_batch_par_slice(data, sweep, simd)
}

#[derive(Debug, Clone)]
pub struct WmaStream {
	period: usize,
	buffer: Vec<f64>,
	head: usize,
	filled: bool,
}

impl WmaStream {
	pub fn try_new(params: WmaParams) -> Result<Self, WmaError> {
		let period = params.period.unwrap_or(30);
		if period < 2 {
			return Err(WmaError::InvalidPeriod { period, data_len: 0 });
		}
		Ok(Self {
			period,
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
		Some(self.weighted_average())
	}

	#[inline(always)]
	fn weighted_average(&self) -> f64 {
		let mut sum = 0.0;
		let mut weight_sum = 0.0;
		let mut idx = self.head;
		for i in 0..self.period {
			let weight = (i + 1) as f64;
			let val = self.buffer[idx];
			sum += val * weight;
			weight_sum += weight;
			idx = (idx + 1) % self.period;
		}
		sum / weight_sum
	}
}

#[derive(Clone, Debug)]
pub struct WmaBatchRange {
	pub period: (usize, usize, usize),
}

impl Default for WmaBatchRange {
	fn default() -> Self {
		Self { period: (2, 240, 1) }
	}
}

#[derive(Clone, Debug, Default)]
pub struct WmaBatchBuilder {
	range: WmaBatchRange,
	kernel: Kernel,
}

impl WmaBatchBuilder {
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

	pub fn apply_slice(self, data: &[f64]) -> Result<WmaBatchOutput, WmaError> {
		wma_with_kernel_batch(data, &self.range, self.kernel)
	}

	pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<WmaBatchOutput, WmaError> {
		WmaBatchBuilder::new().kernel(k).apply_slice(data)
	}

	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<WmaBatchOutput, WmaError> {
		let slice = source_type(c, src);
		self.apply_slice(slice)
	}

	pub fn with_default_candles(c: &Candles) -> Result<WmaBatchOutput, WmaError> {
		WmaBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
	}
}

#[derive(Clone, Debug)]
pub struct WmaBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<WmaParams>,
	pub rows: usize,
	pub cols: usize,
}
impl WmaBatchOutput {
	pub fn row_for_params(&self, p: &WmaParams) -> Option<usize> {
		self.combos
			.iter()
			.position(|c| c.period.unwrap_or(30) == p.period.unwrap_or(30))
	}

	pub fn values_for(&self, p: &WmaParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &WmaBatchRange) -> Vec<WmaParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}

	let periods = axis_usize(r.period);
	let mut out = Vec::with_capacity(periods.len());
	for &p in &periods {
		out.push(WmaParams { period: Some(p) });
	}
	out
}

#[inline(always)]
pub fn wma_batch_slice(data: &[f64], sweep: &WmaBatchRange, kern: Kernel) -> Result<WmaBatchOutput, WmaError> {
	wma_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn wma_batch_par_slice(data: &[f64], sweep: &WmaBatchRange, kern: Kernel) -> Result<WmaBatchOutput, WmaError> {
	wma_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn wma_batch_inner(
	data: &[f64],
	sweep: &WmaBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<WmaBatchOutput, WmaError> {
	let combos = expand_grid(sweep);
	let rows = combos.len();
	let cols = data.len();
	if cols == 0 {
		return Err(WmaError::EmptyInputData);
	}

	// Allocate uninitialized rows×cols
	let mut buf_mu = make_uninit_matrix(rows, cols);

	// Warmup prefixes per row
	let first = data.iter().position(|x| !x.is_nan()).ok_or(WmaError::AllValuesNaN)?;
	let warm: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap() - 1).collect();
	init_matrix_prefixes(&mut buf_mu, cols, &warm);

	// Reinterpret as &mut [f64] without extra writes
	let mut guard = core::mem::ManuallyDrop::new(buf_mu);
	let out: &mut [f64] = unsafe {
		core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len())
	};

	// Fill in-place
	wma_batch_inner_into(data, sweep, kern, parallel, out)?;

	// Materialize Vec<f64> without copying
	let values = unsafe {
		Vec::from_raw_parts(guard.as_mut_ptr() as *mut f64, guard.len(), guard.capacity())
	};

	Ok(WmaBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
fn wma_batch_inner_into(
	data: &[f64],
	sweep: &WmaBatchRange,
	kern: Kernel,
	parallel: bool,
	out: &mut [f64],
) -> Result<Vec<WmaParams>, WmaError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(WmaError::InvalidPeriod { period: 0, data_len: 0 });
	}

	let first = data.iter().position(|x| !x.is_nan()).ok_or(WmaError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if data.len() - first < max_p {
		return Err(WmaError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}

	let rows = combos.len();
	let cols = data.len();

	// ---------- 1.  How many leading NaNs each row needs ----------
	let warm: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap() - 1).collect();

	// ---------- 2.  Reinterpret output slice as MaybeUninit for efficient initialization ----------
	// SAFETY: We're reinterpreting the output slice as MaybeUninit to use the efficient
	// init_matrix_prefixes function. This is safe because:
	// 1. MaybeUninit<T> has the same layout as T
	// 2. We ensure all values are written before the slice is used again
	let out_uninit = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len()) };

	unsafe { init_matrix_prefixes(out_uninit, cols, &warm) };

	// ---------- 3.  Closure that fills one row ----------
	let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
		let period = combos[row].period.unwrap();

		// Re-interpret this row as &mut [f64] so the kernel can write directly.
		let out_row = core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

		match kern {
			Kernel::Scalar | Kernel::ScalarBatch => wma_row_scalar(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => wma_row_avx2(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => wma_row_avx512(data, first, period, out_row),
			Kernel::Auto | _ => wma_row_scalar(data, first, period, out_row),
		}
	};

	// ---------- 4.  Run every row ----------
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
unsafe fn wma_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	wma_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn wma_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	wma_row_scalar(data, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn wma_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	if period <= 32 {
		wma_row_avx512_short(data, first, period, out);
	} else {
		wma_row_avx512_long(data, first, period, out);
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn wma_row_avx512_short(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	wma_row_scalar(data, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn wma_row_avx512_long(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	wma_row_scalar(data, first, period, out)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;
	use paste::paste;

	fn check_wma_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let default_params = WmaParams { period: None };
		let input = WmaInput::from_candles(&candles, "close", default_params);
		let output = wma_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());

		let params_period_14 = WmaParams { period: Some(14) };
		let input2 = WmaInput::from_candles(&candles, "hl2", params_period_14);
		let output2 = wma_with_kernel(&input2, kernel)?;
		assert_eq!(output2.values.len(), candles.close.len());

		let params_custom = WmaParams { period: Some(20) };
		let input3 = WmaInput::from_candles(&candles, "hlc3", params_custom);
		let output3 = wma_with_kernel(&input3, kernel)?;
		assert_eq!(output3.values.len(), candles.close.len());
		Ok(())
	}

	fn check_wma_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let data = &candles.close;
		let default_params = WmaParams::default();
		let input = WmaInput::from_candles(&candles, "close", default_params);
		let result = wma_with_kernel(&input, kernel)?;

		let expected_last_five = [
			59638.52903225806,
			59563.7376344086,
			59489.4064516129,
			59432.02580645162,
			59350.58279569892,
		];
		assert!(result.values.len() >= 5, "Not enough WMA values");
		assert_eq!(
			result.values.len(),
			data.len(),
			"WMA output length should match input length"
		);
		let start_index = result.values.len().saturating_sub(5);
		let last_five = &result.values[start_index..];
		for (i, &value) in last_five.iter().enumerate() {
			assert!(
				(value - expected_last_five[i]).abs() < 1e-6,
				"WMA value mismatch at index {}: expected {}, got {}",
				i,
				expected_last_five[i],
				value
			);
		}
		let period = input.params.period.unwrap_or(30);
		for val in result.values.iter().skip(period - 1) {
			if !val.is_nan() {
				assert!(val.is_finite(), "WMA output should be finite");
			}
		}
		Ok(())
	}

	fn check_wma_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = WmaInput::with_default_candles(&candles);
		match input.data {
			WmaData::Candles { source, .. } => assert_eq!(source, "close"),
			_ => panic!("Expected WmaData::Candles"),
		}
		let output = wma_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_wma_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let input_data = [10.0, 20.0, 30.0];
		let params = WmaParams { period: Some(0) };
		let input = WmaInput::from_slice(&input_data, params);
		let res = wma_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] WMA should fail with zero period", test_name);
		Ok(())
	}

	fn check_wma_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data_small = [10.0, 20.0, 30.0];
		let params = WmaParams { period: Some(10) };
		let input = WmaInput::from_slice(&data_small, params);
		let res = wma_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] WMA should fail with period exceeding length",
			test_name
		);
		Ok(())
	}

	fn check_wma_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let single_point = [42.0];
		let params = WmaParams { period: Some(9) };
		let input = WmaInput::from_slice(&single_point, params);
		let res = wma_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] WMA should fail with insufficient data", test_name);
		Ok(())
	}

	fn check_wma_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let first_params = WmaParams { period: Some(14) };
		let first_input = WmaInput::from_candles(&candles, "close", first_params);
		let first_result = wma_with_kernel(&first_input, kernel)?;
		let second_params = WmaParams { period: Some(5) };
		let second_input = WmaInput::from_slice(&first_result.values, second_params);
		let second_result = wma_with_kernel(&second_input, kernel)?;
		assert_eq!(second_result.values.len(), first_result.values.len());
		for val in &second_result.values[50..] {
			assert!(!val.is_nan());
		}
		Ok(())
	}

	fn check_wma_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = WmaParams { period: Some(14) };
		let input = WmaInput::from_candles(&candles, "close", params);
		let result = wma_with_kernel(&input, kernel)?;
		assert_eq!(result.values.len(), candles.close.len());
		if result.values.len() > 50 {
			for i in 50..result.values.len() {
				assert!(!result.values[i].is_nan());
			}
		}
		Ok(())
	}

	fn check_wma_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let period = 30;
		let input = WmaInput::from_candles(&candles, "close", WmaParams { period: Some(period) });
		let batch_output = wma_with_kernel(&input, kernel)?.values;

		let mut stream = WmaStream::try_new(WmaParams { period: Some(period) })?;
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
				diff < 1e-8,
				"[{}] WMA streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
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
	fn check_wma_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		// Test with multiple parameter combinations to increase chance of catching bugs
		let test_periods = vec![
			2,   // Minimum valid period
			5,   // Small period
			10,  // Small period
			14,  // Common period
			20,  // Medium period
			30,  // Default period
			50,  // Large period
			100, // Very large period
			200, // Extra large period
		];

		for &period in &test_periods {
			// Skip if period would be too large for the data
			if period > candles.close.len() {
				continue;
			}

			let input = WmaInput::from_candles(&candles, "close", WmaParams { period: Some(period) });
			let output = wma_with_kernel(&input, kernel)?;

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
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} with period {}",
						test_name, val, bits, i, period
					);
				}

				// Check for init_matrix_prefixes poison (0x22222222_22222222)
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} with period {}",
						test_name, val, bits, i, period
					);
				}

				// Check for make_uninit_matrix poison (0x33333333_33333333)
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} with period {}",
						test_name, val, bits, i, period
					);
				}
			}
		}

		Ok(())
	}

	// Release mode stub - does nothing
	#[cfg(not(debug_assertions))]
	fn check_wma_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(())
	}

	macro_rules! generate_all_wma_tests {
        ($($test_fn:ident),*) => {
            paste! {
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

	#[cfg(feature = "proptest")]
	#[allow(clippy::float_cmp)]
	fn check_wma_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		use proptest::prelude::*;
		skip_if_unsupported!(kernel, test_name);

		// Test strategy: generate period first (2 is minimum valid), then data of appropriate length
		let strat = (2usize..=100).prop_flat_map(|period| {
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
				let params = WmaParams { period: Some(period) };
				let input = WmaInput::from_slice(&data, params.clone());

				// Get output from the kernel being tested
				let WmaOutput { values: out } = wma_with_kernel(&input, kernel).unwrap();
				
				// Get reference output from scalar kernel for cross-kernel verification
				let WmaOutput { values: ref_out } = wma_with_kernel(&input, Kernel::Scalar).unwrap();

				// Find first non-NaN value in input
				let first = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
				let warmup_end = first + period - 1;

				// Property 1: Warmup period - first (first + period - 1) values should be NaN
				for i in 0..warmup_end.min(out.len()) {
					prop_assert!(
						out[i].is_nan(),
						"Expected NaN during warmup at index {}, got {}",
						i,
						out[i]
					);
				}

				// Property 2: Valid values after warmup - all should be finite
				for i in warmup_end..out.len() {
					prop_assert!(
						out[i].is_finite(),
						"Expected finite value after warmup at index {}, got {}",
						i,
						out[i]
					);
				}

				// Properties 3-11 for valid WMA values
				for i in warmup_end..data.len() {
					let window_start = i + 1 - period;
					let window = &data[window_start..=i];
					
					// Property 3: Bounds checking - WMA should be within [min, max] of window
					let lo = window.iter().cloned().fold(f64::INFINITY, f64::min);
					let hi = window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
					let y = out[i];
					
					prop_assert!(
						y >= lo - 1e-9 && y <= hi + 1e-9,
						"WMA at index {} = {} is outside window bounds [{}, {}]",
						i, y, lo, hi
					);

					// Property 4: Constant input - if all values same, WMA = that value
					if window.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-12) {
						prop_assert!(
							(y - window[0]).abs() <= 1e-9,
							"Constant input should produce constant output: {} vs {}",
							y, window[0]
						);
					}

					// Property 5: Cross-kernel consistency
					let r = ref_out[i];
					if y.is_finite() && r.is_finite() {
						let y_bits = y.to_bits();
						let r_bits = r.to_bits();
						let ulp_diff = y_bits.abs_diff(r_bits);
						
						prop_assert!(
							(y - r).abs() <= 1e-9 || ulp_diff <= 4,
							"Kernel mismatch at index {}: {} vs {} (ULP={})",
							i, y, r, ulp_diff
						);
					}
				}

				// Property 6: Period=2 edge case (minimum valid period)
				if period == 2 && out.len() >= 2 {
					let idx = warmup_end;
					if idx < out.len() {
						// WMA with period 2: weights are [1, 2], sum = 3
						// WMA = (1*data[i-1] + 2*data[i]) / 3
						let expected = (data[idx - 1] + 2.0 * data[idx]) / 3.0;
						prop_assert!(
							(out[idx] - expected).abs() <= 1e-9,
							"Period=2 calculation mismatch: {} vs expected {}",
							out[idx], expected
						);
					}
				}

				// Property 7: Weight formula verification for small periods
				if period <= 5 && warmup_end < out.len() {
					let idx = warmup_end;
					let window_start = idx + 1 - period;
					
					// Calculate WMA manually using the exact formula
					let mut weighted_sum = 0.0;
					let mut weight_sum = 0.0;
					for (j, &val) in data[window_start..=idx].iter().enumerate() {
						let weight = (j + 1) as f64;
						weighted_sum += weight * val;
						weight_sum += weight;
					}
					let expected = weighted_sum / weight_sum;
					
					prop_assert!(
						(out[idx] - expected).abs() <= 1e-9,
						"Weight formula verification failed at index {}: {} vs expected {}",
						idx, out[idx], expected
					);
				}

				// Property 8: Responsiveness - WMA responds more to recent values
				// Test with a step function: if data suddenly changes, WMA should move toward new value
				if data.len() >= period * 2 {
					let mid = data.len() / 2;
					if mid > warmup_end {
						// Create synthetic step: first half low, second half high
						let mut step_data = vec![10.0; data.len()];
						for i in mid..step_data.len() {
							step_data[i] = 100.0;
						}
						
						let step_input = WmaInput::from_slice(&step_data, params.clone());
						let WmaOutput { values: step_out } = wma_with_kernel(&step_input, kernel).unwrap();
						
						// After the step, WMA should be closer to 100 than 10 (due to higher weights on recent)
						if mid + period < step_out.len() {
							let wma_after_step = step_out[mid + period - 1];
							let distance_to_new = (wma_after_step - 100.0).abs();
							let distance_to_old = (wma_after_step - 10.0).abs();
							prop_assert!(
								distance_to_new < distance_to_old,
								"WMA should respond more to recent values: {} should be closer to 100 than 10",
								wma_after_step
							);
						}
					}
				}

				// Property 9: Exact period length data - single valid output
				if data.len() == period {
					let valid_count = out.iter().filter(|x| x.is_finite()).count();
					prop_assert!(
						valid_count == 1,
						"With data.len() == period, should have exactly 1 valid output, got {}",
						valid_count
					);
					
					// The single valid value should be at the last index
					prop_assert!(
						out[data.len() - 1].is_finite(),
						"Last value should be valid when data.len() == period"
					);
				}

				// Property 10: Monotonicity preservation
				// If input is strictly increasing, WMA should be increasing (allowing for numerical tolerance)
				let is_monotonic_increasing = data.windows(2).all(|w| w[1] >= w[0] - 1e-12);
				if is_monotonic_increasing && out.len() > warmup_end + 1 {
					for i in (warmup_end + 1)..out.len() {
						// Allow small tolerance for numerical errors
						prop_assert!(
							out[i] >= out[i - 1] - 1e-9,
							"Monotonic input should produce monotonic WMA: {} < {} at index {}",
							out[i], out[i - 1], i
						);
					}
				}

				// Property 11: Poison value detection (in debug mode)
				#[cfg(debug_assertions)]
				{
					for (i, &val) in out.iter().enumerate() {
						if !val.is_nan() {
							let bits = val.to_bits();
							prop_assert!(
								bits != 0x11111111_11111111 && 
								bits != 0x22222222_22222222 && 
								bits != 0x33333333_33333333,
								"Found poison value at index {}: {} (0x{:016X})",
								i, val, bits
							);
						}
					}
				}

				Ok(())
			})
			.unwrap();

		Ok(())
	}

	generate_all_wma_tests!(
		check_wma_partial_params,
		check_wma_accuracy,
		check_wma_default_candles,
		check_wma_zero_period,
		check_wma_period_exceeds_length,
		check_wma_very_small_dataset,
		check_wma_reinput,
		check_wma_nan_handling,
		check_wma_streaming,
		check_wma_no_poison
	);

	#[cfg(feature = "proptest")]
	generate_all_wma_tests!(check_wma_property);

	fn check_invalid_kernel_error(test: &str) -> Result<(), Box<dyn Error>> {
		let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
		let sweep = WmaBatchRange { period: (2, 5, 1) };
		
		// Test with non-batch kernels - should return InvalidKernel error
		let non_batch_kernels = vec![Kernel::Scalar, Kernel::Avx2, Kernel::Avx512];
		for kernel in non_batch_kernels {
			let result = wma_with_kernel_batch(&data, &sweep, kernel);
			assert!(
				matches!(result, Err(WmaError::InvalidKernel)),
				"[{}] Expected InvalidKernel error for {:?}, got {:?}",
				test, kernel, result
			);
		}
		
		// Test with batch kernels - should succeed
		let batch_kernels = vec![Kernel::Auto, Kernel::ScalarBatch];
		#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
		let batch_kernels = vec![Kernel::Auto, Kernel::ScalarBatch, Kernel::Avx2Batch, Kernel::Avx512Batch];
		
		for kernel in batch_kernels {
			let result = wma_with_kernel_batch(&data, &sweep, kernel);
			assert!(
				result.is_ok(),
				"[{}] Expected success for batch kernel {:?}, got error: {:?}",
				test, kernel, result.err()
			);
		}
		
		Ok(())
	}

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let output = WmaBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;

		let def = WmaParams::default();
		let row = output.values_for(&def).expect("default row missing");

		assert_eq!(row.len(), c.close.len());

		let expected = [
			59638.52903225806,
			59563.7376344086,
			59489.4064516129,
			59432.02580645162,
			59350.58279569892,
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

		// Test multiple batch configurations to increase detection coverage
		let batch_configs = vec![
			(2, 10, 1),    // Edge case: starting from minimum period
			(5, 25, 5),    // Small range with step 5
			(10, 30, 10),  // Medium range with larger step
			(20, 100, 10), // Large range with step 10
			(30, 150, 30), // Default start with large step
			(50, 200, 50), // Very large periods
			(2, 5, 1),     // Very small range to test edge cases
		];

		for (start, end, step) in batch_configs {
			// Skip configurations that would exceed data length
			if start > c.close.len() {
				continue;
			}

			let output = WmaBatchBuilder::new()
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
				let period = output.combos[row].period.unwrap_or(0);

				// Check for alloc_with_nan_prefix poison (0x11111111_11111111)
				if bits == 0x11111111_11111111 {
					panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {}) for period {} in range ({}, {}, {})",
                        test, val, bits, row, col, idx, period, start, end, step
                    );
				}

				// Check for init_matrix_prefixes poison (0x22222222_22222222)
				if bits == 0x22222222_22222222 {
					panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {}) for period {} in range ({}, {}, {})",
                        test, val, bits, row, col, idx, period, start, end, step
                    );
				}

				// Check for make_uninit_matrix poison (0x33333333_33333333)
				if bits == 0x33333333_33333333 {
					panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {}) for period {} in range ({}, {}, {})",
                        test, val, bits, row, col, idx, period, start, end, step
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
			paste! {
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
	
	#[test]
	fn test_invalid_kernel_error() {
		let _ = check_invalid_kernel_error("test_invalid_kernel_error");
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "wma")]
#[pyo3(signature = (data, period, kernel=None))]
/// Compute the Weighted Moving Average (WMA) of the input data.
///
/// WMA assigns linearly increasing weights to each data point in the window,
/// with the most recent values carrying the highest weights.
///
/// Parameters:
/// -----------
/// data : np.ndarray
///     Input data array (float64).
/// period : int
///     Window size (must be >= 2).
/// kernel : str, optional
///     Computation kernel to use: 'auto', 'scalar', 'avx2', 'avx512'.
///     Default is 'auto' which auto-detects the best available.
///
/// Returns:
/// --------
/// np.ndarray
///     Array of WMA values, same length as input.
///
/// Raises:
/// -------
/// ValueError
///     If inputs are invalid (period < 2, exceeds data length, etc).
pub fn wma_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	period: usize,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
	use numpy::{IntoPyArray, PyArrayMethods};

	let slice_in = data.as_slice()?; // zero-copy, read-only view

	// Parse and validate kernel before allow_threads
	let kern = validate_kernel(kernel, false)?;

	// Build input struct
	let params = WmaParams { period: Some(period) };
	let wma_in = WmaInput::from_slice(slice_in, params);

	// Get Vec<f64> from Rust function
	let result_vec: Vec<f64> = py
		.allow_threads(|| wma_with_kernel(&wma_in, kern).map(|o| o.values))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	// Zero-copy transfer to NumPy
	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "WmaStream")]
pub struct WmaStreamPy {
	stream: WmaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl WmaStreamPy {
	#[new]
	fn new(period: usize) -> PyResult<Self> {
		let params = WmaParams { period: Some(period) };
		let stream = WmaStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(WmaStreamPy { stream })
	}

	/// Updates the stream with a new value and returns the calculated WMA value.
	/// Returns `None` if the buffer is not yet full.
	fn update(&mut self, value: f64) -> Option<f64> {
		self.stream.update(value)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "wma_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
/// Compute WMA for multiple period values in a single pass.
///
/// Parameters:
/// -----------
/// data : np.ndarray
///     Input data array (float64).
/// period_range : tuple
///     (start, end, step) for period values to compute.
/// kernel : str, optional
///     Computation kernel: 'auto', 'scalar', 'avx2', 'avx512'.
///     Default is 'auto' which auto-detects the best available.
///
/// Returns:
/// --------
/// dict
///     Dictionary with 'values' (2D array) and 'periods' arrays.
pub fn wma_batch_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	period_range: (usize, usize, usize),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;

	let slice_in = data.as_slice()?;

	let sweep = WmaBatchRange { period: period_range };

	// Parse and validate kernel
	let kern = validate_kernel(kernel, true)?;

	// 1. Expand grid once to know rows*cols
	let combos = expand_grid(&sweep);
	let rows = combos.len();
	let cols = slice_in.len();

	// 2. Pre-allocate NumPy array (1-D, will reshape later)
	let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let slice_out = unsafe { out_arr.as_slice_mut()? };

	// 3. Heavy work without the GIL
	let combos = py
		.allow_threads(|| {
			// Resolve Kernel::Auto to a specific kernel
			let kernel = match kern {
				Kernel::Auto => detect_best_batch_kernel(),
				k => k,
			};
			let simd = match kernel {
				#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
				Kernel::Avx512Batch => Kernel::Avx512,
				#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
				Kernel::Avx2Batch => Kernel::Avx2,
				Kernel::ScalarBatch => Kernel::Scalar,
				_ => Kernel::Scalar,
			};
			// Use the _into variant that writes directly to our pre-allocated buffer
			wma_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
		})
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	// 4. Build dict with the GIL
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
	dict.set_item("rows", rows)?;
	dict.set_item("cols", cols)?;

	Ok(dict)
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct WmaBatchConfig {
	pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct WmaBatchJsOutput {
	pub values: Vec<f64>,
	pub combos: Vec<WmaParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = wma_batch)]
pub fn wma_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	let cfg: WmaBatchConfig =
		serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
	let sweep = WmaBatchRange { period: cfg.period_range };

	// detect_best_kernel() is correct here (inner expects scalar/avx, not *Batch)
	let out = wma_batch_inner(data, &sweep, detect_best_kernel(), false)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;

	let js = WmaBatchJsOutput { values: out.values, combos: out.combos, rows: out.rows, cols: out.cols };
	serde_wasm_bindgen::to_value(&js).map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn wma_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
	let params = WmaParams { period: Some(period) };
	let input = WmaInput::from_slice(data, params);

	let mut output = vec![0.0; data.len()];
	wma_into_slice(&mut output, &input, Kernel::Auto).map_err(|e| JsValue::from_str(&e.to_string()))?;

	Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn wma_batch_js(
	data: &[f64],
	period_start: usize,
	period_end: usize,
	period_step: usize,
) -> Result<Vec<f64>, JsValue> {
	let sweep = WmaBatchRange {
		period: (period_start, period_end, period_step),
	};

	// Use the existing batch function with parallel=false for WASM
	wma_batch_inner(data, &sweep, Kernel::Auto, false)
		.map(|output| output.values)
		.map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn wma_batch_metadata_js(period_start: usize, period_end: usize, period_step: usize) -> Result<Vec<f64>, JsValue> {
	let sweep = WmaBatchRange {
		period: (period_start, period_end, period_step),
	};

	let combos = expand_grid(&sweep);
	let mut metadata = Vec::with_capacity(combos.len());

	for combo in combos {
		metadata.push(combo.period.unwrap() as f64);
	}

	Ok(metadata)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn wma_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn wma_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe {
			let _ = Vec::from_raw_parts(ptr, len, len);
		}
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn wma_into(
	in_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period: usize,
) -> Result<(), JsValue> {
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}

	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);
		let params = WmaParams { period: Some(period) };
		let input = WmaInput::from_slice(data, params);

		if in_ptr == out_ptr {
			// Handle aliasing - allocate temporary buffer
			let mut temp = vec![0.0; len];
			wma_into_slice(&mut temp, &input, Kernel::Auto).map_err(|e| JsValue::from_str(&e.to_string()))?;
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			out.copy_from_slice(&temp);
		} else {
			// No aliasing - write directly to output
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			wma_into_slice(out, &input, Kernel::Auto).map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn wma_batch_into(
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

		let sweep = WmaBatchRange {
			period: (period_start, period_end, period_step),
		};

		let combos = expand_grid(&sweep);
		let rows = combos.len();
		let total_size = rows * len;

		let out = std::slice::from_raw_parts_mut(out_ptr, total_size);

		// Use the existing batch function that writes directly to the output
		wma_batch_inner_into(data, &sweep, Kernel::Auto, false, out)
			.map_err(|e| JsValue::from_str(&e.to_string()))?;

		Ok(rows)
	}
}
