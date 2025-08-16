//! # Volume Oscillator (VOSC)
//!
//! Measures changes in volume trends using two moving averages (short and long).
//!
//! ## Formula
//! ```ignore
//! vosc = 100 * ((short_avg - long_avg) / long_avg)
//! ```
//!
//! ## Parameters
//! - **short_period**: The short window size. Defaults to 2.
//! - **long_period**: The long window size. Defaults to 5.
//!
//! ## Errors
//! - **EmptyData**: vosc: Input data slice is empty.
//! - **InvalidShortPeriod**: vosc: `short_period` is zero or exceeds the data length.
//! - **InvalidLongPeriod**: vosc: `long_period` is zero or exceeds the data length.
//! - **ShortPeriodGreaterThanLongPeriod**: vosc: `short_period` is greater than `long_period`.
//! - **NotEnoughValidData**: vosc: Fewer than `long_period` valid data points after the first valid index.
//! - **AllValuesNaN**: vosc: All input data values are `NaN`.
//!
//! ## Returns
//! - **`Ok(VoscOutput)`** on success, containing a `Vec<f64>` matching input length.
//! - **`Err(VoscError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes, make_uninit_matrix};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use thiserror::Error;

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

impl<'a> AsRef<[f64]> for VoscInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			VoscData::Slice(slice) => slice,
			VoscData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

#[derive(Debug, Clone)]
pub enum VoscData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct VoscOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct VoscParams {
	pub short_period: Option<usize>,
	pub long_period: Option<usize>,
}

impl Default for VoscParams {
	fn default() -> Self {
		Self {
			short_period: Some(2),
			long_period: Some(5),
		}
	}
}

#[derive(Debug, Clone)]
pub struct VoscInput<'a> {
	pub data: VoscData<'a>,
	pub params: VoscParams,
}

impl<'a> VoscInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: VoscParams) -> Self {
		Self {
			data: VoscData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slice(sl: &'a [f64], p: VoscParams) -> Self {
		Self {
			data: VoscData::Slice(sl),
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "volume", VoscParams::default())
	}
	#[inline]
	pub fn get_short_period(&self) -> usize {
		self.params.short_period.unwrap_or(2)
	}
	#[inline]
	pub fn get_long_period(&self) -> usize {
		self.params.long_period.unwrap_or(5)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct VoscBuilder {
	short_period: Option<usize>,
	long_period: Option<usize>,
	kernel: Kernel,
}

impl Default for VoscBuilder {
	fn default() -> Self {
		Self {
			short_period: None,
			long_period: None,
			kernel: Kernel::Auto,
		}
	}
}

impl VoscBuilder {
	#[inline(always)]
	pub fn new() -> Self {
		Self::default()
	}
	#[inline(always)]
	pub fn short_period(mut self, n: usize) -> Self {
		self.short_period = Some(n);
		self
	}
	#[inline(always)]
	pub fn long_period(mut self, n: usize) -> Self {
		self.long_period = Some(n);
		self
	}
	#[inline(always)]
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}

	#[inline(always)]
	pub fn apply(self, c: &Candles) -> Result<VoscOutput, VoscError> {
		let p = VoscParams {
			short_period: self.short_period,
			long_period: self.long_period,
		};
		let i = VoscInput::from_candles(c, "volume", p);
		vosc_with_kernel(&i, self.kernel)
	}

	#[inline(always)]
	pub fn apply_slice(self, d: &[f64]) -> Result<VoscOutput, VoscError> {
		let p = VoscParams {
			short_period: self.short_period,
			long_period: self.long_period,
		};
		let i = VoscInput::from_slice(d, p);
		vosc_with_kernel(&i, self.kernel)
	}

	#[inline(always)]
	pub fn into_stream(self) -> Result<VoscStream, VoscError> {
		let p = VoscParams {
			short_period: self.short_period,
			long_period: self.long_period,
		};
		VoscStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum VoscError {
	#[error("vosc: Empty data provided for VOSC.")]
	EmptyData,
	#[error("vosc: Invalid short period: short_period = {period}, data length = {data_len}")]
	InvalidShortPeriod { period: usize, data_len: usize },
	#[error("vosc: Invalid long period: long_period = {period}, data length = {data_len}")]
	InvalidLongPeriod { period: usize, data_len: usize },
	#[error("vosc: short_period > long_period")]
	ShortPeriodGreaterThanLongPeriod,
	#[error("vosc: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("vosc: All values are NaN.")]
	AllValuesNaN,
}

#[inline]
pub fn vosc(input: &VoscInput) -> Result<VoscOutput, VoscError> {
	vosc_with_kernel(input, Kernel::Auto)
}

pub fn vosc_with_kernel(input: &VoscInput, kernel: Kernel) -> Result<VoscOutput, VoscError> {
	let data: &[f64] = match &input.data {
		VoscData::Candles { candles, source } => source_type(candles, source),
		VoscData::Slice(sl) => sl,
	};

	if data.is_empty() {
		return Err(VoscError::EmptyData);
	}

	let short_period = input.get_short_period();
	let long_period = input.get_long_period();

	if short_period == 0 || short_period > data.len() {
		return Err(VoscError::InvalidShortPeriod {
			period: short_period,
			data_len: data.len(),
		});
	}
	if long_period == 0 || long_period > data.len() {
		return Err(VoscError::InvalidLongPeriod {
			period: long_period,
			data_len: data.len(),
		});
	}
	if short_period > long_period {
		return Err(VoscError::ShortPeriodGreaterThanLongPeriod);
	}

	let first = match data.iter().position(|&x| !x.is_nan()) {
		Some(idx) => idx,
		None => return Err(VoscError::AllValuesNaN),
	};
	if (data.len() - first) < long_period {
		return Err(VoscError::NotEnoughValidData {
			needed: long_period,
			valid: data.len() - first,
		});
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	let warmup_period = first + long_period - 1;
	let mut out = alloc_with_nan_prefix(data.len(), warmup_period);

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => vosc_scalar(data, short_period, long_period, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => vosc_avx2(data, short_period, long_period, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => vosc_avx512(data, short_period, long_period, first, &mut out),
			_ => unreachable!(),
		}
	}

	Ok(VoscOutput { values: out })
}

#[inline]
pub fn vosc_scalar(data: &[f64], short_period: usize, long_period: usize, first_valid: usize, out: &mut [f64]) {
	let mut short_sum = 0.0;
	let mut long_sum = 0.0;
	for i in first_valid..(first_valid + long_period) {
		let v = data[i];
		if i >= (first_valid + long_period - short_period) {
			short_sum += v;
		}
		long_sum += v;
	}

	let short_div = 1.0 / (short_period as f64);
	let long_div = 1.0 / (long_period as f64);
	let init_idx = first_valid + long_period - 1;
	let mut savg = short_sum * short_div;
	let mut lavg = long_sum * long_div;
	out[init_idx] = 100.0 * (savg - lavg) / lavg;

	for i in (first_valid + long_period)..data.len() {
		short_sum += data[i];
		short_sum -= data[i - short_period];
		long_sum += data[i];
		long_sum -= data[i - long_period];

		savg = short_sum * short_div;
		lavg = long_sum * long_div;
		out[i] = 100.0 * (savg - lavg) / lavg;
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn vosc_avx512(data: &[f64], short_period: usize, long_period: usize, first_valid: usize, out: &mut [f64]) {
	vosc_scalar(data, short_period, long_period, first_valid, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn vosc_avx2(data: &[f64], short_period: usize, long_period: usize, first_valid: usize, out: &mut [f64]) {
	vosc_scalar(data, short_period, long_period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vosc_avx512_short(
	data: &[f64],
	short_period: usize,
	long_period: usize,
	first_valid: usize,
	out: &mut [f64],
) {
	vosc_scalar(data, short_period, long_period, first_valid, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vosc_avx512_long(
	data: &[f64],
	short_period: usize,
	long_period: usize,
	first_valid: usize,
	out: &mut [f64],
) {
	vosc_scalar(data, short_period, long_period, first_valid, out)
}

#[inline]
pub fn vosc_row_scalar(data: &[f64], first: usize, short_period: usize, long_period: usize, out: &mut [f64]) {
	vosc_scalar(data, short_period, long_period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vosc_row_avx2(data: &[f64], first: usize, short_period: usize, long_period: usize, out: &mut [f64]) {
	vosc_scalar(data, short_period, long_period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vosc_row_avx512(data: &[f64], first: usize, short_period: usize, long_period: usize, out: &mut [f64]) {
	vosc_scalar(data, short_period, long_period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vosc_row_avx512_short(
	data: &[f64],
	first: usize,
	short_period: usize,
	long_period: usize,
	out: &mut [f64],
) {
	vosc_scalar(data, short_period, long_period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vosc_row_avx512_long(
	data: &[f64],
	first: usize,
	short_period: usize,
	long_period: usize,
	out: &mut [f64],
) {
	vosc_scalar(data, short_period, long_period, first, out)
}

#[derive(Debug, Clone)]
pub struct VoscStream {
	short_period: usize,
	long_period: usize,
	short_buf: Vec<f64>,
	long_buf: Vec<f64>,
	short_head: usize,
	long_head: usize,
	short_filled: bool,
	long_filled: bool,
}

impl VoscStream {
	pub fn try_new(params: VoscParams) -> Result<Self, VoscError> {
		let short_period = params.short_period.unwrap_or(2);
		let long_period = params.long_period.unwrap_or(5);
		if short_period == 0 {
			return Err(VoscError::InvalidShortPeriod {
				period: short_period,
				data_len: 0,
			});
		}
		if long_period == 0 {
			return Err(VoscError::InvalidLongPeriod {
				period: long_period,
				data_len: 0,
			});
		}
		if short_period > long_period {
			return Err(VoscError::ShortPeriodGreaterThanLongPeriod);
		}
		Ok(Self {
			short_period,
			long_period,
			short_buf: vec![f64::NAN; short_period],
			long_buf: vec![f64::NAN; long_period],
			short_head: 0,
			long_head: 0,
			short_filled: false,
			long_filled: false,
		})
	}

	#[inline(always)]
	pub fn update(&mut self, value: f64) -> Option<f64> {
		self.short_buf[self.short_head] = value;
		self.long_buf[self.long_head] = value;

		self.short_head = (self.short_head + 1) % self.short_period;
		self.long_head = (self.long_head + 1) % self.long_period;

		if !self.short_filled && self.short_head == 0 {
			self.short_filled = true;
		}
		if !self.long_filled && self.long_head == 0 {
			self.long_filled = true;
		}
		if !self.short_filled || !self.long_filled {
			return None;
		}
		let short_avg = self.short_buf.iter().copied().sum::<f64>() / self.short_period as f64;
		let long_avg = self.long_buf.iter().copied().sum::<f64>() / self.long_period as f64;
		Some(100.0 * (short_avg - long_avg) / long_avg)
	}
}

#[derive(Clone, Debug)]
pub struct VoscBatchRange {
	pub short_period: (usize, usize, usize),
	pub long_period: (usize, usize, usize),
}

impl Default for VoscBatchRange {
	fn default() -> Self {
		Self {
			short_period: (2, 10, 1),
			long_period: (5, 20, 1),
		}
	}
}

#[derive(Clone, Debug, Default)]
pub struct VoscBatchBuilder {
	range: VoscBatchRange,
	kernel: Kernel,
}

impl VoscBatchBuilder {
	pub fn new() -> Self {
		Self::default()
	}
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline]
	pub fn short_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.short_period = (start, end, step);
		self
	}
	#[inline]
	pub fn short_period_static(mut self, n: usize) -> Self {
		self.range.short_period = (n, n, 0);
		self
	}
	#[inline]
	pub fn long_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.long_period = (start, end, step);
		self
	}
	#[inline]
	pub fn long_period_static(mut self, n: usize) -> Self {
		self.range.long_period = (n, n, 0);
		self
	}
	pub fn apply_slice(self, data: &[f64]) -> Result<VoscBatchOutput, VoscError> {
		vosc_batch_with_kernel(data, &self.range, self.kernel)
	}
	pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<VoscBatchOutput, VoscError> {
		VoscBatchBuilder::new().kernel(k).apply_slice(data)
	}
	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<VoscBatchOutput, VoscError> {
		let slice = source_type(c, src);
		self.apply_slice(slice)
	}
	pub fn with_default_candles(c: &Candles) -> Result<VoscBatchOutput, VoscError> {
		VoscBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "volume")
	}
}

pub fn vosc_batch_with_kernel(data: &[f64], sweep: &VoscBatchRange, k: Kernel) -> Result<VoscBatchOutput, VoscError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(VoscError::InvalidLongPeriod { period: 0, data_len: 0 }),
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	vosc_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct VoscBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<VoscParams>,
	pub rows: usize,
	pub cols: usize,
}
impl VoscBatchOutput {
	pub fn row_for_params(&self, p: &VoscParams) -> Option<usize> {
		self.combos.iter().position(|c| {
			c.short_period.unwrap_or(2) == p.short_period.unwrap_or(2)
				&& c.long_period.unwrap_or(5) == p.long_period.unwrap_or(5)
		})
	}

	pub fn values_for(&self, p: &VoscParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &VoscBatchRange) -> Vec<VoscParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let shorts = axis_usize(r.short_period);
	let longs = axis_usize(r.long_period);
	let mut out = Vec::with_capacity(shorts.len() * longs.len());
	for &s in &shorts {
		for &l in &longs {
			if s <= l {
				out.push(VoscParams {
					short_period: Some(s),
					long_period: Some(l),
				});
			}
		}
	}
	out
}

#[inline(always)]
pub fn vosc_batch_slice(data: &[f64], sweep: &VoscBatchRange, kern: Kernel) -> Result<VoscBatchOutput, VoscError> {
	vosc_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn vosc_batch_par_slice(data: &[f64], sweep: &VoscBatchRange, kern: Kernel) -> Result<VoscBatchOutput, VoscError> {
	vosc_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn vosc_batch_inner(
	data: &[f64],
	sweep: &VoscBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<VoscBatchOutput, VoscError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(VoscError::InvalidLongPeriod { period: 0, data_len: 0 });
	}

	let first = data.iter().position(|x| !x.is_nan()).ok_or(VoscError::AllValuesNaN)?;
	let max_long = combos.iter().map(|c| c.long_period.unwrap()).max().unwrap();
	if data.len() - first < max_long {
		return Err(VoscError::NotEnoughValidData {
			needed: max_long,
			valid: data.len() - first,
		});
	}

	let rows = combos.len();
	let cols = data.len();
	
	// Use uninitialized memory with proper prefixes, matching ALMA pattern
	let mut buf_mu = make_uninit_matrix(rows, cols);
	
	// Calculate warmup periods for each parameter combination
	let warmup_periods: Vec<usize> = combos
		.iter()
		.map(|c| first + c.long_period.unwrap() - 1)
		.collect();
	
	init_matrix_prefixes(&mut buf_mu, cols, &warmup_periods);
	
	// Convert to Vec<f64> from MaybeUninit
	let values = unsafe {
		let ptr = buf_mu.as_mut_ptr() as *mut f64;
		let len = buf_mu.len();
		let cap = buf_mu.capacity();
		std::mem::forget(buf_mu);
		Vec::from_raw_parts(ptr, len, cap)
	};
	
	// Convert back to mutable slices for processing
	let mut values = values;
	
	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let short = combos[row].short_period.unwrap();
		let long = combos[row].long_period.unwrap();
		match kern {
			Kernel::Scalar => vosc_row_scalar(data, first, short, long, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => vosc_row_avx2(data, first, short, long, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => vosc_row_avx512(data, first, short, long, out_row),
			_ => unreachable!(),
		}
	};
	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			values
				.par_chunks_mut(cols)
				.enumerate()
				.for_each(|(row, slice)| do_row(row, slice));
		}

		#[cfg(target_arch = "wasm32")]
		{
			for (row, slice) in values.chunks_mut(cols).enumerate() {
				do_row(row, slice);
			}
		}
	} else {
		for (row, slice) in values.chunks_mut(cols).enumerate() {
			do_row(row, slice);
		}
	}

	Ok(VoscBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_vosc_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let volume = candles
			.select_candle_field("volume")
			.expect("Failed to extract volume data");
		let params = VoscParams {
			short_period: Some(2),
			long_period: Some(5),
		};
		let input = VoscInput::from_candles(&candles, "volume", params);
		let vosc_result = vosc_with_kernel(&input, kernel)?;

		assert_eq!(vosc_result.values.len(), volume.len(), "VOSC length mismatch");
		let expected_last_five_vosc = [
			-39.478510754298895,
			-25.886077312645188,
			-21.155087549723756,
			-12.36093768813373,
			48.70809369473075,
		];
		let start_index = vosc_result.values.len() - 5;
		let result_last_five_vosc = &vosc_result.values[start_index..];
		for (i, &value) in result_last_five_vosc.iter().enumerate() {
			let expected_value = expected_last_five_vosc[i];
			assert!(
				(value - expected_value).abs() < 1e-1,
				"VOSC mismatch at index {}: expected {}, got {}",
				i,
				expected_value,
				value
			);
		}
		for i in 0..(5 - 1) {
			assert!(vosc_result.values[i].is_nan());
		}

		let default_input = VoscInput::with_default_candles(&candles);
		let default_vosc_result = vosc_with_kernel(&default_input, kernel)?;
		assert_eq!(default_vosc_result.values.len(), volume.len());
		Ok(())
	}

	fn check_vosc_zero_period(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);
		let input_data = [10.0, 20.0, 30.0];
		let params = VoscParams {
			short_period: Some(0),
			long_period: Some(5),
		};
		let input = VoscInput::from_slice(&input_data, params);
		let res = vosc_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] VOSC should fail with zero short_period", test);
		Ok(())
	}

	fn check_vosc_short_gt_long(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);
		let input_data = [10.0, 20.0, 30.0, 40.0, 50.0];
		let params = VoscParams {
			short_period: Some(5),
			long_period: Some(2),
		};
		let input = VoscInput::from_slice(&input_data, params);
		let res = vosc_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] VOSC should fail when short_period > long_period",
			test
		);
		Ok(())
	}

	fn check_vosc_not_enough_valid(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);
		let data = [f64::NAN, f64::NAN, 1.0, 2.0, 3.0];
		let params = VoscParams {
			short_period: Some(2),
			long_period: Some(5),
		};
		let input = VoscInput::from_slice(&data, params);
		let res = vosc_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] VOSC should fail with not enough valid data", test);
		Ok(())
	}

	fn check_vosc_all_nan(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);
		let data = [f64::NAN, f64::NAN, f64::NAN];
		let params = VoscParams {
			short_period: Some(2),
			long_period: Some(3),
		};
		let input = VoscInput::from_slice(&data, params);
		let res = vosc_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] VOSC should fail with all NaN", test);
		Ok(())
	}

	fn check_vosc_streaming(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let volume = candles
			.select_candle_field("volume")
			.expect("Failed to extract volume data");
		let short_period = 2;
		let long_period = 5;
		let input = VoscInput::from_candles(
			&candles,
			"volume",
			VoscParams {
				short_period: Some(short_period),
				long_period: Some(long_period),
			},
		);
		let batch_output = vosc_with_kernel(&input, kernel)?.values;

		let mut stream = VoscStream::try_new(VoscParams {
			short_period: Some(short_period),
			long_period: Some(long_period),
		})?;
		let mut stream_values = Vec::with_capacity(volume.len());
		for &v in volume {
			match stream.update(v) {
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
				"[{}] VOSC streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
				test,
				i,
				b,
				s,
				diff
			);
		}
		Ok(())
	}

	#[cfg(debug_assertions)]
	fn check_vosc_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		
		// Define comprehensive parameter combinations
		let test_params = vec![
			VoscParams::default(),  // short: 2, long: 5
			VoscParams {
				short_period: Some(1),  // minimum viable
				long_period: Some(2),
			},
			VoscParams {
				short_period: Some(1),  // minimum short with larger long
				long_period: Some(5),
			},
			VoscParams {
				short_period: Some(2),  // small
				long_period: Some(10),
			},
			VoscParams {
				short_period: Some(5),  // medium
				long_period: Some(20),
			},
			VoscParams {
				short_period: Some(10),  // large
				long_period: Some(50),
			},
			VoscParams {
				short_period: Some(20),  // very large
				long_period: Some(100),
			},
			VoscParams {
				short_period: Some(3),  // edge case: close periods
				long_period: Some(5),
			},
			VoscParams {
				short_period: Some(10),  // edge case: equal ratio
				long_period: Some(10),
			},
			VoscParams {
				short_period: Some(4),  // specific combinations
				long_period: Some(12),
			},
			VoscParams {
				short_period: Some(7),
				long_period: Some(21),
			},
			VoscParams {
				short_period: Some(14),
				long_period: Some(28),
			},
		];
		
		for (param_idx, params) in test_params.iter().enumerate() {
			let input = VoscInput::from_candles(&candles, "volume", params.clone());
			let output = vosc_with_kernel(&input, kernel)?;
			
			for (i, &val) in output.values.iter().enumerate() {
				if val.is_nan() {
					continue; // NaN values are expected during warmup
				}
				
				let bits = val.to_bits();
				
				// Check all three poison patterns
				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: short_period={}, long_period={} (param set {})",
						test_name, val, bits, i, 
						params.short_period.unwrap_or(2),
						params.long_period.unwrap_or(5),
						param_idx
					);
				}
				
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: short_period={}, long_period={} (param set {})",
						test_name, val, bits, i,
						params.short_period.unwrap_or(2),
						params.long_period.unwrap_or(5),
						param_idx
					);
				}
				
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: short_period={}, long_period={} (param set {})",
						test_name, val, bits, i,
						params.short_period.unwrap_or(2),
						params.long_period.unwrap_or(5),
						param_idx
					);
				}
			}
		}
		
		Ok(())
	}

	#[cfg(not(debug_assertions))]
	fn check_vosc_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		Ok(()) // No-op in release builds
	}

	macro_rules! generate_all_vosc_tests {
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

	#[cfg(feature = "proptest")]
	#[allow(clippy::float_cmp)]
	fn check_vosc_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		use proptest::prelude::*;
		skip_if_unsupported!(kernel, test_name);

		let strat = (1usize..=50, 1usize..=50)
			.prop_flat_map(|(short, long)| {
				let max_period = short.max(long);
				(
					prop::collection::vec(
						(0.1f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
						max_period..400,
					),
					Just((short, long)),
				)
			});

		proptest::test_runner::TestRunner::default()
			.run(&strat, |(data, (short_period, long_period))| {
				// Skip invalid combinations
				if short_period > long_period {
					return Ok(());
				}

				let params = VoscParams {
					short_period: Some(short_period),
					long_period: Some(long_period),
				};
				let input = VoscInput::from_slice(&data, params);

				let VoscOutput { values: out } = vosc_with_kernel(&input, kernel).unwrap();
				let VoscOutput { values: ref_out } = vosc_with_kernel(&input, Kernel::Scalar).unwrap();

				// Property 1: Warmup period validation
				// First (long_period - 1) values should be NaN
				for i in 0..(long_period - 1) {
					prop_assert!(
						out[i].is_nan(),
						"Expected NaN during warmup at index {}, got {}",
						i,
						out[i]
					);
				}

				// Property 2: Kernel consistency
				// All kernels should produce identical results
				for i in (long_period - 1)..data.len() {
					let y = out[i];
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

					let ulp_diff: u64 = y_bits.abs_diff(r_bits);

					prop_assert!(
						(y - r).abs() <= 1e-9 || ulp_diff <= 4,
						"Kernel mismatch idx {}: {} vs {} (ULP={})",
						i, y, r, ulp_diff
					);
				}

				// Property 3: Mathematical formula verification
				// VOSC = 100 * ((short_avg - long_avg) / long_avg)
				// Only verify for indices where we have full windows
				for i in long_period..data.len() {
					// For sliding window, we look at the most recent 'period' values
					let short_start = i + 1 - short_period;
					let long_start = i + 1 - long_period;
					
					let short_sum: f64 = data[short_start..=i].iter().sum();
					let long_sum: f64 = data[long_start..=i].iter().sum();
					
					let short_avg = short_sum / short_period as f64;
					let long_avg = long_sum / long_period as f64;
					
					let expected = 100.0 * (short_avg - long_avg) / long_avg;
					let actual = out[i];
					
					prop_assert!(
						(actual - expected).abs() <= 1e-9,
						"Formula mismatch at idx {}: expected {}, got {}",
						i, expected, actual
					);
				}

				// Property 4: Zero oscillation for equal periods
				// When short_period == long_period, VOSC should be 0
				if short_period == long_period {
					for i in (long_period - 1)..data.len() {
						prop_assert!(
							out[i].abs() <= 1e-9,
							"Expected 0 when periods equal at idx {}: got {}",
							i, out[i]
						);
					}
				}

				// Property 5: Constant volume stability
				// If all volumes are the same, VOSC should be 0
				if data.windows(2).all(|w| (w[0] - w[1]).abs() <= f64::EPSILON) {
					for i in (long_period - 1)..data.len() {
						prop_assert!(
							out[i].abs() <= 1e-9,
							"Expected 0 for constant volume at idx {}: got {}",
							i, out[i]
						);
					}
				}

				Ok(())
			})
			.unwrap();

		Ok(())
	}

	generate_all_vosc_tests!(
		check_vosc_accuracy,
		check_vosc_zero_period,
		check_vosc_short_gt_long,
		check_vosc_not_enough_valid,
		check_vosc_all_nan,
		check_vosc_streaming,
		check_vosc_no_poison
	);

	#[cfg(feature = "proptest")]
	generate_all_vosc_tests!(check_vosc_property);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let output = VoscBatchBuilder::new().kernel(kernel).apply_candles(&c, "volume")?;
		let def = VoscParams::default();
		let row = output.values_for(&def).expect("default row missing");
		assert_eq!(row.len(), c.volume.len());

		let expected = [
			-39.478510754298895,
			-25.886077312645188,
			-21.155087549723756,
			-12.36093768813373,
			48.70809369473075,
		];
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
	fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);
		
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		
		// Test various parameter sweep configurations
		let test_configs = vec![
			// (short_start, short_end, short_step, long_start, long_end, long_step)
			(1, 5, 1, 2, 10, 2),      // Small periods
			(2, 10, 2, 5, 20, 5),     // Medium periods
			(10, 20, 5, 20, 50, 10),  // Large periods
			(1, 3, 1, 3, 6, 1),       // Dense small range
			(5, 15, 2, 10, 30, 5),    // Medium range with overlap
			(2, 2, 0, 5, 25, 5),      // Static short, varying long
			(1, 10, 3, 10, 10, 0),    // Varying short, static long
			(3, 9, 3, 9, 27, 9),      // Specific ratio patterns
			(1, 5, 1, 5, 5, 0),       // Edge case: converging to equal
		];
		
		for (cfg_idx, &(s_start, s_end, s_step, l_start, l_end, l_step)) in test_configs.iter().enumerate() {
			let output = VoscBatchBuilder::new()
				.kernel(kernel)
				.short_period_range(s_start, s_end, s_step)
				.long_period_range(l_start, l_end, l_step)
				.apply_candles(&c, "volume")?;
			
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
						 at row {} col {} (flat index {}) with params: short_period={}, long_period={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.short_period.unwrap_or(2),
						combo.long_period.unwrap_or(5)
					);
				}
				
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: short_period={}, long_period={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.short_period.unwrap_or(2),
						combo.long_period.unwrap_or(5)
					);
				}
				
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: short_period={}, long_period={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.short_period.unwrap_or(2),
						combo.long_period.unwrap_or(5)
					);
				}
			}
		}
		
		Ok(())
	}

	#[cfg(not(debug_assertions))]
	fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
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

// ================================================================================================
// Python Bindings
// ================================================================================================

#[cfg(feature = "python")]
#[pyfunction(name = "vosc")]
#[pyo3(signature = (data, short_period=2, long_period=5, kernel=None))]
pub fn vosc_py<'py>(
	py: Python<'py>,
	data: PyReadonlyArray1<'py, f64>,
	short_period: usize,
	long_period: usize,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
	use numpy::{IntoPyArray, PyArrayMethods};

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, false)?;

	let params = VoscParams {
		short_period: Some(short_period),
		long_period: Some(long_period),
	};
	let input = VoscInput::from_slice(slice_in, params);

	// Get Vec<f64> from Rust function and convert to NumPy with zero-copy
	let result_vec: Vec<f64> = py
		.allow_threads(|| vosc_with_kernel(&input, kern).map(|o| o.values))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "VoscStream")]
pub struct VoscStreamPy {
	stream: VoscStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl VoscStreamPy {
	#[new]
	fn new(short_period: usize, long_period: usize) -> PyResult<Self> {
		let params = VoscParams {
			short_period: Some(short_period),
			long_period: Some(long_period),
		};
		let stream = VoscStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(VoscStreamPy { stream })
	}

	/// Updates the stream with a new value and returns the calculated VOSC value.
	/// Returns `None` if the buffer is not yet full.
	fn update(&mut self, value: f64) -> Option<f64> {
		self.stream.update(value)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "vosc_batch")]
#[pyo3(signature = (data, short_period_range, long_period_range, kernel=None))]
pub fn vosc_batch_py<'py>(
	py: Python<'py>,
	data: PyReadonlyArray1<'py, f64>,
	short_period_range: (usize, usize, usize),
	long_period_range: (usize, usize, usize),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, true)?;

	let sweep = VoscBatchRange {
		short_period: short_period_range,
		long_period: long_period_range,
	};

	// Expand grid to know dimensions
	let combos = expand_grid(&sweep);
	if combos.is_empty() {
		return Err(PyValueError::new_err("No valid parameter combinations"));
	}
	let rows = combos.len();
	let cols = slice_in.len();

	// Pre-allocate uninitialized NumPy array (acceptable for batch operations)
	let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let slice_out = unsafe { out_arr.as_slice_mut()? };

	// Heavy work without the GIL
	let combos = py
		.allow_threads(|| -> Result<Vec<VoscParams>, VoscError> {
			// Resolve Kernel::Auto to a specific kernel
			let kernel = match kern {
				Kernel::Auto => detect_best_batch_kernel(),
				k => k,
			};

			// Map batch kernel to regular kernel
			let simd = match kernel {
				Kernel::Avx512Batch => Kernel::Avx512,
				Kernel::Avx2Batch => Kernel::Avx2,
				Kernel::ScalarBatch => Kernel::Scalar,
				_ => kernel,
			};

			let result = vosc_batch_inner(slice_in, &sweep, simd, true)?;

			// Copy results to the pre-allocated buffer
			slice_out.copy_from_slice(&result.values);

			Ok(result.combos)
		})
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	// Build result dictionary
	let dict = PyDict::new(py);
	dict.set_item("values", out_arr.reshape((rows, cols))?)?;
	dict.set_item(
		"short_periods",
		combos
			.iter()
			.map(|p| p.short_period.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;
	dict.set_item(
		"long_periods",
		combos
			.iter()
			.map(|p| p.long_period.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;

	Ok(dict)
}

// ============================================================================
// WASM API
// ============================================================================

#[cfg(feature = "wasm")]
/// Write VOSC values directly to output slice - no allocations
pub fn vosc_into_slice(
	dst: &mut [f64], 
	input: &VoscInput, 
	kern: Kernel
) -> Result<(), VoscError> {
	let data: &[f64] = match &input.data {
		VoscData::Candles { candles, source } => source_type(candles, source),
		VoscData::Slice(sl) => sl,
	};

	if data.is_empty() {
		return Err(VoscError::EmptyData);
	}

	if dst.len() != data.len() {
		return Err(VoscError::NotEnoughValidData {
			needed: data.len(),
			valid: dst.len(),
		});
	}

	let short_period = input.get_short_period();
	let long_period = input.get_long_period();

	if short_period == 0 || short_period > data.len() {
		return Err(VoscError::InvalidShortPeriod {
			period: short_period,
			data_len: data.len(),
		});
	}
	if long_period == 0 || long_period > data.len() {
		return Err(VoscError::InvalidLongPeriod {
			period: long_period,
			data_len: data.len(),
		});
	}
	if short_period > long_period {
		return Err(VoscError::ShortPeriodGreaterThanLongPeriod);
	}

	let first = match data.iter().position(|&x| !x.is_nan()) {
		Some(idx) => idx,
		None => return Err(VoscError::AllValuesNaN),
	};
	
	if (data.len() - first) < long_period {
		return Err(VoscError::NotEnoughValidData {
			needed: long_period,
			valid: data.len() - first,
		});
	}

	let chosen = match kern {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	let warmup_period = first + long_period - 1;

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => vosc_scalar(data, short_period, long_period, first, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => vosc_avx2(data, short_period, long_period, first, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => vosc_avx512(data, short_period, long_period, first, dst),
			_ => unreachable!(),
		}
	}
	
	// Fill warmup with NaN
	for v in &mut dst[..warmup_period] {
		*v = f64::NAN;
	}

	Ok(())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vosc_js(data: &[f64], short_period: usize, long_period: usize) -> Result<Vec<f64>, JsValue> {
	let params = VoscParams {
		short_period: Some(short_period),
		long_period: Some(long_period),
	};
	let input = VoscInput::from_slice(data, params);
	
	let mut output = vec![0.0; data.len()];  // Single allocation
	vosc_into_slice(&mut output, &input, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vosc_into(
	in_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	short_period: usize,
	long_period: usize,
) -> Result<(), JsValue> {
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}
	
	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);
		let params = VoscParams {
			short_period: Some(short_period),
			long_period: Some(long_period),
		};
		let input = VoscInput::from_slice(data, params);
		
		if in_ptr == out_ptr {  // CRITICAL: Aliasing check
			let mut temp = vec![0.0; len];
			vosc_into_slice(&mut temp, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			out.copy_from_slice(&temp);
		} else {
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			vosc_into_slice(out, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vosc_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vosc_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe { let _ = Vec::from_raw_parts(ptr, len, len); }
	}
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct VoscBatchConfig {
	pub short_period_range: (usize, usize, usize),
	pub long_period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct VoscBatchJsOutput {
	pub values: Vec<f64>,
	pub combos: Vec<VoscParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = vosc_batch)]
pub fn vosc_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	let config: VoscBatchConfig =
		serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

	let sweep = VoscBatchRange {
		short_period: config.short_period_range,
		long_period: config.long_period_range,
	};

	let output = vosc_batch_inner(data, &sweep, Kernel::Auto, false)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;

	let js_output = VoscBatchJsOutput {
		values: output.values,
		combos: output.combos,
		rows: output.rows,
		cols: output.cols,
	};

	serde_wasm_bindgen::to_value(&js_output).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}
