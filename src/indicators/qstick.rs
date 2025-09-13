//! # Qstick
//!
//! Qstick measures the average difference between the Close and Open over a specified period.
//! A positive Qstick indicates that, on average, the market closes above its open, while a
//! negative Qstick indicates the opposite.
//!
//! ## Parameters
//! - **period**: The window size (number of data points). Defaults to 5.
//!
//! ## Errors
//! - **AllValuesNaN**: qstick: All input data values are `NaN`.
//! - **InvalidPeriod**: qstick: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: qstick: Not enough valid data points for the requested `period`.
//!
//! ## Returns
//! - **`Ok(QstickOutput)`** on success, containing a `Vec<f64>` matching the input length.
//! - **`Err(QstickError)`** otherwise.

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
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[derive(Debug, Clone)]
pub enum QstickData<'a> {
	Candles {
		candles: &'a Candles,
		open_source: &'a str,
		close_source: &'a str,
	},
	Slices {
		open: &'a [f64],
		close: &'a [f64],
	},
}

#[derive(Debug, Clone)]
pub struct QstickOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct QstickParams {
	pub period: Option<usize>,
}

impl Default for QstickParams {
	fn default() -> Self {
		Self { period: Some(5) }
	}
}

#[derive(Debug, Clone)]
pub struct QstickInput<'a> {
	pub data: QstickData<'a>,
	pub params: QstickParams,
}

impl<'a> QstickInput<'a> {
	#[inline]
	pub fn from_candles(
		candles: &'a Candles,
		open_source: &'a str,
		close_source: &'a str,
		params: QstickParams,
	) -> Self {
		Self {
			data: QstickData::Candles {
				candles,
				open_source,
				close_source,
			},
			params,
		}
	}

	#[inline]
	pub fn from_slices(open: &'a [f64], close: &'a [f64], params: QstickParams) -> Self {
		Self {
			data: QstickData::Slices { open, close },
			params,
		}
	}

	#[inline]
	pub fn with_default_candles(candles: &'a Candles) -> Self {
		Self {
			data: QstickData::Candles {
				candles,
				open_source: "open",
				close_source: "close",
			},
			params: QstickParams::default(),
		}
	}

	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(5)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct QstickBuilder {
	period: Option<usize>,
	kernel: Kernel,
}

impl Default for QstickBuilder {
	fn default() -> Self {
		Self {
			period: None,
			kernel: Kernel::Auto,
		}
	}
}

impl QstickBuilder {
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
	pub fn apply(self, candles: &Candles) -> Result<QstickOutput, QstickError> {
		let params = QstickParams { period: self.period };
		let input = QstickInput::from_candles(candles, "open", "close", params);
		qstick_with_kernel(&input, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slices(self, open: &[f64], close: &[f64]) -> Result<QstickOutput, QstickError> {
		let params = QstickParams { period: self.period };
		let input = QstickInput::from_slices(open, close, params);
		qstick_with_kernel(&input, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<QstickStream, QstickError> {
		let params = QstickParams { period: self.period };
		QstickStream::try_new(params)
	}
}

#[derive(Debug, Error)]
pub enum QstickError {
	#[error("qstick: All values are NaN.")]
	AllValuesNaN,
	#[error("qstick: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("qstick: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn qstick(input: &QstickInput) -> Result<QstickOutput, QstickError> {
	qstick_with_kernel(input, Kernel::Auto)
}

pub fn qstick_with_kernel(input: &QstickInput, kernel: Kernel) -> Result<QstickOutput, QstickError> {
	let (open, close) = match &input.data {
		QstickData::Candles {
			candles,
			open_source,
			close_source,
		} => {
			let open = source_type(candles, open_source);
			let close = source_type(candles, close_source);
			(open, close)
		}
		QstickData::Slices { open, close } => (*open, *close),
	};

	let len = open.len().min(close.len());
	let period = input.get_period();

	if period == 0 || period > len {
		return Err(QstickError::InvalidPeriod { period, data_len: len });
	}

	// Find first valid index by checking both open and close
	let mut first = 0;
	for i in 0..len {
		if !open[i].is_nan() && !close[i].is_nan() {
			first = i;
			break;
		}
		if i == len - 1 {
			return Err(QstickError::AllValuesNaN);
		}
	}

	if (len - first) < period {
		return Err(QstickError::NotEnoughValidData {
			needed: period,
			valid: len - first,
		});
	}

	let mut out = alloc_with_nan_prefix(len, first + period - 1);

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => qstick_scalar(open, close, period, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => qstick_avx2(open, close, period, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => qstick_avx512(open, close, period, first, &mut out),
			_ => unreachable!(),
		}
	}

	Ok(QstickOutput { values: out })
}

#[inline]
pub fn qstick_scalar(open: &[f64], close: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	let mut sum = 0.0;
	// Compute initial sum of differences
	for i in first_valid..first_valid + period {
		sum += close[i] - open[i];
	}
	let inv_period = 1.0 / (period as f64);
	out[first_valid + period - 1] = sum * inv_period;
	
	// Rolling window computation
	for i in (first_valid + period)..close.len() {
		sum += (close[i] - open[i]) - (close[i - period] - open[i - period]);
		out[i] = sum * inv_period;
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn qstick_avx512(open: &[f64], close: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	qstick_scalar(open, close, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn qstick_avx2(open: &[f64], close: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	qstick_scalar(open, close, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn qstick_avx512_short(open: &[f64], close: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	qstick_avx512(open, close, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn qstick_avx512_long(open: &[f64], close: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	qstick_avx512(open, close, period, first_valid, out)
}

#[inline]
pub fn qstick_batch_with_kernel(
	open: &[f64],
	close: &[f64],
	sweep: &QstickBatchRange,
	kernel: Kernel,
) -> Result<QstickBatchOutput, QstickError> {
	let kern = match kernel {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(QstickError::InvalidPeriod { period: 0, data_len: 0 }),
	};
	let simd = match kern {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	qstick_batch_par_slice(open, close, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct QstickBatchRange {
	pub period: (usize, usize, usize),
}

impl Default for QstickBatchRange {
	fn default() -> Self {
		Self { period: (5, 240, 1) }
	}
}

#[derive(Clone, Debug, Default)]
pub struct QstickBatchBuilder {
	range: QstickBatchRange,
	kernel: Kernel,
}

impl QstickBatchBuilder {
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
	pub fn apply_slices(self, open: &[f64], close: &[f64]) -> Result<QstickBatchOutput, QstickError> {
		qstick_batch_with_kernel(open, close, &self.range, self.kernel)
	}
	pub fn apply_candles(self, c: &Candles, open_src: &str, close_src: &str) -> Result<QstickBatchOutput, QstickError> {
		let open = source_type(c, open_src);
		let close = source_type(c, close_src);
		self.apply_slices(open, close)
	}
	pub fn with_default_candles(c: &Candles) -> Result<QstickBatchOutput, QstickError> {
		QstickBatchBuilder::new()
			.kernel(Kernel::Auto)
			.apply_candles(c, "open", "close")
	}
}

#[derive(Clone, Debug)]
pub struct QstickBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<QstickParams>,
	pub rows: usize,
	pub cols: usize,
}

impl QstickBatchOutput {
	pub fn row_for_params(&self, p: &QstickParams) -> Option<usize> {
		self.combos
			.iter()
			.position(|c| c.period.unwrap_or(5) == p.period.unwrap_or(5))
	}
	pub fn values_for(&self, p: &QstickParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &QstickBatchRange) -> Vec<QstickParams> {
	let (start, end, step) = r.period;
	if step == 0 || start == end {
		return vec![QstickParams { period: Some(start) }];
	}
	
	// Pre-calculate capacity to avoid reallocations
	let count = ((end - start) / step) + 1;
	let mut out = Vec::with_capacity(count);
	
	let mut p = start;
	while p <= end {
		out.push(QstickParams { period: Some(p) });
		p += step;
	}
	out
}

#[inline(always)]
pub fn qstick_batch_slice(
	open: &[f64],
	close: &[f64],
	sweep: &QstickBatchRange,
	kern: Kernel,
) -> Result<QstickBatchOutput, QstickError> {
	qstick_batch_inner(open, close, sweep, kern, false)
}

#[inline(always)]
pub fn qstick_batch_par_slice(
	open: &[f64],
	close: &[f64],
	sweep: &QstickBatchRange,
	kern: Kernel,
) -> Result<QstickBatchOutput, QstickError> {
	qstick_batch_inner(open, close, sweep, kern, true)
}

#[inline(always)]
fn qstick_batch_inner(
	open: &[f64],
	close: &[f64],
	sweep: &QstickBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<QstickBatchOutput, QstickError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(QstickError::InvalidPeriod { period: 0, data_len: 0 });
	}
	let len = open.len().min(close.len());
	
	// Find first valid index
	let mut first = 0;
	for i in 0..len {
		if !open[i].is_nan() && !close[i].is_nan() {
			first = i;
			break;
		}
		if i == len - 1 {
			return Err(QstickError::AllValuesNaN);
		}
	}
	
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if len - first < max_p {
		return Err(QstickError::NotEnoughValidData {
			needed: max_p,
			valid: len - first,
		});
	}
	let rows = combos.len();
	let cols = len;
	
	// Use proper uninitialized memory allocation like ALMA
	let mut buf_mu = make_uninit_matrix(rows, cols);
	
	// Calculate warmup periods for each parameter combination
	let warm: Vec<usize> = combos
		.iter()
		.map(|c| first + c.period.unwrap() - 1)
		.collect();
	init_matrix_prefixes(&mut buf_mu, cols, &warm);
	
	// Convert to mutable slice
	let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
	let out: &mut [f64] =
		unsafe { core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len()) };
	
	qstick_batch_inner_into(open, close, sweep, kern, parallel, out)?;
	
	// Take ownership of the buffer
	let values = unsafe {
		Vec::from_raw_parts(
			buf_guard.as_mut_ptr() as *mut f64,
			buf_guard.len(),
			buf_guard.capacity()
		)
	};
	
	Ok(QstickBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
fn qstick_batch_inner_into(
	open: &[f64],
	close: &[f64],
	sweep: &QstickBatchRange,
	kern: Kernel,
	parallel: bool,
	out: &mut [f64],
) -> Result<Vec<QstickParams>, QstickError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(QstickError::InvalidPeriod { period: 0, data_len: 0 });
	}

	let len = open.len().min(close.len());
	let cols = len;

	// first valid across both inputs
	let first = (0..len)
		.find(|&i| !open[i].is_nan() && !close[i].is_nan())
		.ok_or(QstickError::AllValuesNaN)?;

	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if len - first < max_p {
		return Err(QstickError::NotEnoughValidData {
			needed: max_p,
			valid: len - first,
		});
	}

	// Initialize NaN prefixes for each row based on warmup period
	for (row, combo) in combos.iter().enumerate() {
		let warmup = first + combo.period.unwrap() - 1;
		let row_start = row * cols;
		for i in 0..warmup.min(cols) {
			out[row_start + i] = f64::NAN;
		}
	}

	// Treat output as uninitialized, like alma_batch_inner_into
	let out_mu: &mut [MaybeUninit<f64>] = unsafe {
		std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
	};

	let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
		let period = combos[row].period.unwrap();
		// cast current row to f64 slice
		let dst: &mut [f64] =
			std::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

		match kern {
			Kernel::Scalar | Kernel::ScalarBatch | Kernel::Auto =>
				qstick_scalar(open, close, period, first, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch =>
				qstick_avx2(open, close, period, first, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch =>
				qstick_avx512(open, close, period, first, dst),
			#[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
			_ => qstick_scalar(open, close, period, first, dst),
		}
	};

	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			use rayon::prelude::*;
			out_mu.par_chunks_mut(cols).enumerate().for_each(|(row, slice)| do_row(row, slice));
		}
		#[cfg(target_arch = "wasm32")]
		{
			for (row, slice) in out_mu.chunks_mut(cols).enumerate() { do_row(row, slice); }
		}
	} else {
		for (row, slice) in out_mu.chunks_mut(cols).enumerate() { do_row(row, slice); }
	}

	Ok(combos)
}

#[inline(always)]
unsafe fn qstick_row_scalar(open: &[f64], close: &[f64], first: usize, period: usize, out: &mut [f64]) {
	qstick_scalar(open, close, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn qstick_row_avx2(open: &[f64], close: &[f64], first: usize, period: usize, out: &mut [f64]) {
	qstick_avx2(open, close, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn qstick_row_avx512(open: &[f64], close: &[f64], first: usize, period: usize, out: &mut [f64]) {
	qstick_avx512(open, close, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn qstick_row_avx512_short(open: &[f64], close: &[f64], first: usize, period: usize, out: &mut [f64]) {
	qstick_avx512_short(open, close, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn qstick_row_avx512_long(open: &[f64], close: &[f64], first: usize, period: usize, out: &mut [f64]) {
	qstick_avx512_long(open, close, period, first, out)
}

#[derive(Debug, Clone)]
pub struct QstickStream {
	period: usize,
	buffer: Vec<f64>,
	head: usize,
	filled: bool,
	sum: f64,
}

impl QstickStream {
	pub fn try_new(params: QstickParams) -> Result<Self, QstickError> {
		let period = params.period.unwrap_or(5);
		if period == 0 {
			return Err(QstickError::InvalidPeriod { period, data_len: 0 });
		}
		Ok(Self {
			period,
			buffer: vec![f64::NAN; period],
			head: 0,
			filled: false,
			sum: 0.0,
		})
	}

	#[inline(always)]
	pub fn update(&mut self, open: f64, close: f64) -> Option<f64> {
		let diff = close - open;
		if self.buffer[self.head].is_nan() {
			self.sum += diff;
		} else {
			self.sum += diff - self.buffer[self.head];
		}
		self.buffer[self.head] = diff;
		self.head = (self.head + 1) % self.period;
		if !self.filled && self.head == 0 {
			self.filled = true;
		}
		if self.filled {
			Some(self.sum / (self.period as f64))
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
	#[cfg(feature = "proptest")]
	use proptest::prelude::*;
	fn check_qstick_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let default_params = QstickParams { period: None };
		let input_default = QstickInput::from_candles(&candles, "open", "close", default_params);
		let output_default = qstick_with_kernel(&input_default, kernel)?;
		assert_eq!(output_default.values.len(), candles.close.len());
		let params_period_7 = QstickParams { period: Some(7) };
		let input_period_7 = QstickInput::from_candles(&candles, "open", "close", params_period_7);
		let output_period_7 = qstick_with_kernel(&input_period_7, kernel)?;
		assert_eq!(output_period_7.values.len(), candles.close.len());
		Ok(())
	}
	fn check_qstick_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = QstickParams { period: Some(5) };
		let input = QstickInput::from_candles(&candles, "open", "close", params);
		let result = qstick_with_kernel(&input, kernel)?;
		let expected_last_five_qstick = [219.4, 61.6, -51.8, -53.4, -123.2];
		let start_index = result.values.len() - 5;
		let result_last_five = &result.values[start_index..];
		for (i, &value) in result_last_five.iter().enumerate() {
			let expected_value = expected_last_five_qstick[i];
			assert!(
				(value - expected_value).abs() < 1e-1,
				"[{}] Qstick mismatch at idx {}: got {}, expected {}",
				test_name,
				i,
				value,
				expected_value
			);
		}
		Ok(())
	}
	fn check_qstick_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let open_data = [10.0, 20.0, 30.0];
		let close_data = [15.0, 25.0, 35.0];
		let params = QstickParams { period: Some(0) };
		let input = QstickInput::from_slices(&open_data, &close_data, params);
		let res = qstick_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] Qstick should fail with zero period", test_name);
		Ok(())
	}
	fn check_qstick_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let open_data = [10.0, 20.0, 30.0];
		let close_data = [15.0, 25.0, 35.0];
		let params = QstickParams { period: Some(10) };
		let input = QstickInput::from_slices(&open_data, &close_data, params);
		let res = qstick_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] Qstick should fail with period exceeding length",
			test_name
		);
		Ok(())
	}
	fn check_qstick_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let open_data = [50.0];
		let close_data = [55.0];
		let params = QstickParams { period: Some(5) };
		let input = QstickInput::from_slices(&open_data, &close_data, params);
		let res = qstick_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] Qstick should fail with insufficient data",
			test_name
		);
		Ok(())
	}
	fn check_qstick_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let first_params = QstickParams { period: Some(5) };
		let first_input = QstickInput::from_candles(&candles, "open", "close", first_params);
		let first_result = qstick_with_kernel(&first_input, kernel)?;
		let second_params = QstickParams { period: Some(5) };
		let second_input = QstickInput::from_slices(&first_result.values, &first_result.values, second_params);
		let second_result = qstick_with_kernel(&second_input, kernel)?;
		assert_eq!(second_result.values.len(), first_result.values.len());
		for i in 10..second_result.values.len() {
			assert!(
				!second_result.values[i].is_nan(),
				"[{}] Qstick Slice Reinput: Expected no NaN after idx 10, found NaN at idx {}",
				test_name,
				i
			);
		}
		Ok(())
	}
	fn check_qstick_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = QstickParams { period: Some(5) };
		let input = QstickInput::from_candles(&candles, "open", "close", params);
		let qstick_result = qstick_with_kernel(&input, kernel)?;
		if qstick_result.values.len() > 50 {
			for i in 50..qstick_result.values.len() {
				assert!(
					!qstick_result.values[i].is_nan(),
					"[{}] Expected no NaN after index 50, found NaN at index {}",
					test_name,
					i
				);
			}
		}
		Ok(())
	}

	#[cfg(debug_assertions)]
	fn check_qstick_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		// Define comprehensive parameter combinations
		let test_params = vec![
			QstickParams::default(),                    // period: 5
			QstickParams { period: Some(2) },          // minimum viable
			QstickParams { period: Some(3) },          // small
			QstickParams { period: Some(7) },          // small-medium
			QstickParams { period: Some(10) },         // medium
			QstickParams { period: Some(20) },         // medium-large
			QstickParams { period: Some(30) },         // large
			QstickParams { period: Some(50) },         // very large
			QstickParams { period: Some(100) },        // extra large
		];

		for (param_idx, params) in test_params.iter().enumerate() {
			let input = QstickInput::from_candles(&candles, "open", "close", params.clone());
			let output = qstick_with_kernel(&input, kernel)?;

			for (i, &val) in output.values.iter().enumerate() {
				if val.is_nan() {
					continue; // NaN values are expected during warmup
				}

				let bits = val.to_bits();

				// Check all three poison patterns
				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
						test_name, val, bits, i, params.period.unwrap_or(5), param_idx
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
						test_name, val, bits, i, params.period.unwrap_or(5), param_idx
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
						test_name, val, bits, i, params.period.unwrap_or(5), param_idx
					);
				}
			}
		}

		Ok(())
	}

	#[cfg(not(debug_assertions))]
	fn check_qstick_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(()) // No-op in release builds
	}
	
	#[cfg(feature = "proptest")]
	#[allow(clippy::float_cmp)]
	fn check_qstick_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		use proptest::prelude::*;
		skip_if_unsupported!(kernel, test_name);
		
		// Strategy for generating test data
		let strat = (1usize..=64)
			.prop_flat_map(|period| {
				(period..=400usize).prop_flat_map(move |len| {
					(
						// Generate open prices
						prop::collection::vec(
							(1.0f64..10000.0f64).prop_filter("finite", |x| x.is_finite()),
							len,
						),
						// Generate close prices as deltas from open
						prop::collection::vec(
							(-100.0f64..100.0f64).prop_filter("finite", |x| x.is_finite()),
							len,
						),
						Just(period),
					)
				})
			});
		
		proptest::test_runner::TestRunner::default()
			.run(&strat, |(open_prices, close_deltas, period)| {
				// Create close prices from open + delta
				let close_prices: Vec<f64> = open_prices
					.iter()
					.zip(close_deltas.iter())
					.map(|(o, d)| o + d)
					.collect();
				
				let params = QstickParams {
					period: Some(period),
				};
				let input = QstickInput::from_slices(&open_prices, &close_prices, params);
				
				let QstickOutput { values: out } = qstick_with_kernel(&input, kernel).unwrap();
				let QstickOutput { values: ref_out } = qstick_with_kernel(&input, Kernel::Scalar).unwrap();
				
				// Test 1: Warmup period - first (period - 1) values should be NaN
				for i in 0..(period - 1) {
					prop_assert!(
						out[i].is_nan(),
						"Expected NaN during warmup at index {}, got {}",
						i,
						out[i]
					);
				}
				
				// Test valid output values
				for i in (period - 1)..open_prices.len() {
					let window_start = i + 1 - period;
					let window_end = i + 1;
					
					// Calculate the differences in the window
					let diffs: Vec<f64> = (window_start..window_end)
						.map(|j| close_prices[j] - open_prices[j])
						.collect();
					
					// Test 2: Bounds property - QStick should be within min/max of differences
					let min_diff = diffs.iter().cloned().fold(f64::INFINITY, f64::min);
					let max_diff = diffs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
					let y = out[i];
					
					prop_assert!(
						y.is_nan() || (y >= min_diff - 1e-9 && y <= max_diff + 1e-9),
						"idx {}: QStick {} not in bounds [{}, {}]",
						i, y, min_diff, max_diff
					);
					
					// Test 3: Period=1 property - should equal close - open
					if period == 1 {
						let expected = close_prices[i] - open_prices[i];
						prop_assert!(
							(y - expected).abs() <= 1e-10,
							"Period=1: expected {}, got {} at index {}",
							expected, y, i
						);
					}
					
					// Test 4: Constant difference property
					if diffs.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10) {
						let expected = diffs[0];
						prop_assert!(
							(y - expected).abs() <= 1e-9,
							"Constant diff: expected {}, got {} at index {}",
							expected, y, i
						);
					}
					
					// Test 5: Zero difference property - if open == close everywhere
					if diffs.iter().all(|&d| d.abs() < 1e-10) {
						prop_assert!(
							y.abs() <= 1e-9,
							"Zero diff: expected 0, got {} at index {}",
							y, i
						);
					}
					
					// Test 6: Manual calculation verification
					let expected_qstick = diffs.iter().sum::<f64>() / (period as f64);
					prop_assert!(
						(y - expected_qstick).abs() <= 1e-9,
						"Manual calc: expected {}, got {} at index {}",
						expected_qstick, y, i
					);
					
					// Test 7: Kernel consistency - compare with scalar reference
					let r = ref_out[i];
					let y_bits = y.to_bits();
					let r_bits = r.to_bits();
					
					if !y.is_finite() || !r.is_finite() {
						prop_assert!(
							y.to_bits() == r.to_bits(),
							"finite/NaN mismatch idx {}: {} vs {}",
							i, y, r
						);
						continue;
					}
					
					// ULP tolerance for floating point comparison
					let ulp_diff: u64 = y_bits.abs_diff(r_bits);
					prop_assert!(
						(y - r).abs() <= 1e-9 || ulp_diff <= 4,
						"Kernel mismatch idx {}: {} vs {} (ULP={})",
						i, y, r, ulp_diff
					);
				}
				
				Ok(())
			})
			.unwrap();
		
		Ok(())
	}
	
	macro_rules! generate_all_qstick_tests {
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
	generate_all_qstick_tests!(
		check_qstick_partial_params,
		check_qstick_accuracy,
		check_qstick_zero_period,
		check_qstick_period_exceeds_length,
		check_qstick_very_small_dataset,
		check_qstick_reinput,
		check_qstick_nan_handling,
		check_qstick_no_poison
	);
	
	#[cfg(feature = "proptest")]
	generate_all_qstick_tests!(check_qstick_property);
	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let output = QstickBatchBuilder::new()
			.kernel(kernel)
			.apply_candles(&c, "open", "close")?;
		let def = QstickParams::default();
		let row = output.values_for(&def).expect("default row missing");
		assert_eq!(row.len(), c.close.len());
		let expected = [219.4, 61.6, -51.8, -53.4, -123.2];
		let start = row.len() - 5;
		for (i, &v) in row[start..].iter().enumerate() {
			assert!(
				(v - expected[i]).abs() < 1e-1,
				"[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
			);
		}
		Ok(())
	}

	#[cfg(debug_assertions)]
	fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		// Test various parameter sweep configurations
		let test_configs = vec![
			(2, 10, 2),      // Small periods
			(5, 25, 5),      // Medium periods  
			(30, 60, 15),    // Large periods
			(2, 5, 1),       // Dense small range
			(10, 50, 10),    // Wide range medium step
			(15, 30, 5),     // Medium range small step
			(2, 100, 20),    // Full range large step
		];

		for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
			let output = QstickBatchBuilder::new()
				.kernel(kernel)
				.period_range(p_start, p_end, p_step)
				.apply_candles(&c, "open", "close")?;

			for (idx, &val) in output.values.iter().enumerate() {
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
						 at row {} col {} (flat index {}) with params: period={}",
						test, cfg_idx, val, bits, row, col, idx, combo.period.unwrap_or(5)
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}",
						test, cfg_idx, val, bits, row, col, idx, combo.period.unwrap_or(5)
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}",
						test, cfg_idx, val, bits, row, col, idx, combo.period.unwrap_or(5)
					);
				}
			}
		}

		Ok(())
	}

	#[cfg(not(debug_assertions))]
	fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(()) // No-op in release builds
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
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;

#[cfg(feature = "python")]
#[pyfunction(name = "qstick")]
#[pyo3(signature = (open, close, period, kernel=None))]
pub fn qstick_py<'py>(
	py: Python<'py>,
	open: PyReadonlyArray1<'py, f64>,
	close: PyReadonlyArray1<'py, f64>,
	period: usize,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
	let open_slice = open.as_slice()?;
	let close_slice = close.as_slice()?;
	let kern = validate_kernel(kernel, false)?;
	
	let params = QstickParams {
		period: Some(period),
	};
	let input = QstickInput::from_slices(open_slice, close_slice, params);
	
	let result_vec: Vec<f64> = py
		.allow_threads(|| qstick_with_kernel(&input, kern).map(|o| o.values))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;
	
	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "QstickStream")]
pub struct QstickStreamPy {
	stream: QstickStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl QstickStreamPy {
	#[new]
	pub fn new(period: usize) -> PyResult<Self> {
		let params = QstickParams {
			period: Some(period),
		};
		let stream = QstickStream::try_new(params)
			.map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(QstickStreamPy { stream })
	}
	
	pub fn update(&mut self, open: f64, close: f64) -> Option<f64> {
		self.stream.update(open, close)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "qstick_batch")]
#[pyo3(signature = (open, close, period_range, kernel=None))]
pub fn qstick_batch_py<'py>(
	py: Python<'py>,
	open: PyReadonlyArray1<'py, f64>,
	close: PyReadonlyArray1<'py, f64>,
	period_range: (usize, usize, usize),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
	let open_slice = open.as_slice()?;
	let close_slice = close.as_slice()?;
	let kern = validate_kernel(kernel, true)?;
	
	let sweep = QstickBatchRange {
		period: period_range,
	};
	
	let combos = expand_grid(&sweep);
	let rows = combos.len();
	let cols = open_slice.len();
	
	let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let slice_out = unsafe { out_arr.as_slice_mut()? };
	
	let combos = py.allow_threads(|| {
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
		
		// Use the optimized batch_inner_into function
		qstick_batch_inner_into(open_slice, close_slice, &sweep, simd, true, slice_out)
	})
	.map_err(|e| PyValueError::new_err(e.to_string()))?;
	
	let dict = PyDict::new(py);
	dict.set_item("values", out_arr.reshape((rows, cols))?)?;
	dict.set_item(
		"periods",
		combos.iter()
			.map(|p| p.period.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py)
	)?;
	
	Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_qstick_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
	m.add_function(wrap_pyfunction!(qstick_py, m)?)?;
	m.add_function(wrap_pyfunction!(qstick_batch_py, m)?)?;
	m.add_class::<QstickStreamPy>()?;
	Ok(())
}

/// Write qstick directly to output slice - no allocations
pub fn qstick_into_slice(
	dst: &mut [f64],
	open: &[f64],
	close: &[f64],
	period: usize,
	kern: Kernel,
) -> Result<(), QstickError> {
	// Validate inputs
	let len = open.len().min(close.len());
	if len == 0 {
		return Err(QstickError::InvalidPeriod { period, data_len: len });
	}
	if dst.len() != len {
		return Err(QstickError::InvalidPeriod { period, data_len: len });
	}
	if period == 0 || period > len {
		return Err(QstickError::InvalidPeriod { period, data_len: len });
	}
	
	// Find first valid index
	let mut first_valid = 0;
	for i in 0..len {
		if !open[i].is_nan() && !close[i].is_nan() {
			first_valid = i;
			break;
		}
		if i == len - 1 {
			return Err(QstickError::AllValuesNaN);
		}
	}
	
	// Check if we have enough valid data
	if len - first_valid < period {
		return Err(QstickError::NotEnoughValidData {
			needed: period,
			valid: len - first_valid,
		});
	}
	
	// Compute directly into output
	let kernel = match kern {
		Kernel::Auto => detect_best_kernel(),
		k => k,
	};
	
	match kernel {
		Kernel::Scalar | Kernel::ScalarBatch => qstick_scalar(open, close, period, first_valid, dst),
		#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
		Kernel::Avx2 | Kernel::Avx2Batch => qstick_avx2(open, close, period, first_valid, dst),
		#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
		Kernel::Avx512 | Kernel::Avx512Batch => qstick_avx512(open, close, period, first_valid, dst),
		_ => unreachable!(),
	}
	
	// match alma.rs: apply warmup NaNs after compute
	let warmup_end = first_valid + period - 1;
	for v in &mut dst[..warmup_end] { *v = f64::NAN; }
	
	Ok(())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn qstick_js(open: &[f64], close: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
	// Validate minimum inputs
	let len = open.len();
	if len != close.len() {
		return Err(JsValue::from_str("Open and close arrays must have the same length"));
	}
	
	// Single allocation
	let mut output = vec![0.0; len];
	
	qstick_into_slice(&mut output, open, close, period, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn qstick_into(
	open_ptr: *const f64,
	close_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period: usize,
) -> Result<(), JsValue> {
	if open_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}
	
	unsafe {
		let open = std::slice::from_raw_parts(open_ptr, len);
		let close = std::slice::from_raw_parts(close_ptr, len);
		
		// CRITICAL: Check for aliasing
		// If either input overlaps with output, we need a temporary buffer
		if open_ptr == out_ptr || close_ptr == out_ptr {
			let mut temp = vec![0.0; len];
			qstick_into_slice(&mut temp, open, close, period, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			out.copy_from_slice(&temp);
		} else {
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			qstick_into_slice(out, open, close, period, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn qstick_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn qstick_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe {
			let _ = Vec::from_raw_parts(ptr, len, len);
		}
	}
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct QstickBatchConfig {
	pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct QstickBatchJsOutput {
	pub values: Vec<f64>,
	pub combos: Vec<QstickParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = qstick_batch)]
pub fn qstick_batch_unified_js(open: &[f64], close: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	let config: QstickBatchConfig =
		serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
	
	let len = open.len();
	if len != close.len() {
		return Err(JsValue::from_str("Open and close arrays must have the same length"));
	}
	
	let sweep = QstickBatchRange {
		period: config.period_range,
	};
	
	let output = qstick_batch_inner(open, close, &sweep, Kernel::Auto, false)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	let js_output = QstickBatchJsOutput {
		values: output.values,
		combos: output.combos,
		rows: output.rows,
		cols: output.cols,
	};
	
	serde_wasm_bindgen::to_value(&js_output).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn qstick_batch_into(
	open_ptr: *const f64,
	close_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period_start: usize,
	period_end: usize,
	period_step: usize,
) -> Result<usize, JsValue> {
	if open_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}
	
	unsafe {
		let open = std::slice::from_raw_parts(open_ptr, len);
		let close = std::slice::from_raw_parts(close_ptr, len);
		
		let sweep = QstickBatchRange {
			period: (period_start, period_end, period_step),
		};
		
		let combos = expand_grid(&sweep);
		let rows = combos.len();
		let total_size = rows * len;
		
		// For batch, we write directly to the output - no aliasing check needed
		// as batch output is different size than input
		let out = std::slice::from_raw_parts_mut(out_ptr, total_size);
		
		qstick_batch_inner_into(open, close, &sweep, Kernel::Auto, false, out)
			.map_err(|e| JsValue::from_str(&e.to_string()))?;
		
		Ok(rows)
	}
}
