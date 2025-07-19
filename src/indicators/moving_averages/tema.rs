//! # Triple Exponential Moving Average (TEMA)
//!
//! Applies three exponential moving averages in succession to reduce lag and noise.
//! TEMA is calculated as: `TEMA = 3*EMA1 - 3*EMA2 + EMA3`, with all EMAs using the same period.
//!
//! ## Parameters
//! - **period**: Window size (number of data points, must be ≥ 1).
//!
//! ## Errors
//! - **AllValuesNaN**: tema: All input data values are `NaN`.
//! - **InvalidPeriod**: tema: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: tema: Not enough valid data points for the requested `period`.
//!
//! ## Returns
//! - **`Ok(TemaOutput)`** on success, containing a `Vec<f64>` matching the input length.
//! - **`Err(TemaError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
	alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes, make_uninit_matrix,
};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

// ========== Input Data Types ==========

#[derive(Debug, Clone)]
pub enum TemaData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct TemaInput<'a> {
	pub data: TemaData<'a>,
	pub params: TemaParams,
}

impl<'a> TemaInput<'a> {
	#[inline]
	pub fn from_candles(candles: &'a Candles, source: &'a str, params: TemaParams) -> Self {
		Self {
			data: TemaData::Candles { candles, source },
			params,
		}
	}
	#[inline]
	pub fn from_slice(slice: &'a [f64], params: TemaParams) -> Self {
		Self {
			data: TemaData::Slice(slice),
			params,
		}
	}
	#[inline]
	pub fn with_default_candles(candles: &'a Candles) -> Self {
		Self::from_candles(candles, "close", TemaParams::default())
	}
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(9)
	}
}

impl<'a> AsRef<[f64]> for TemaInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			TemaData::Slice(slice) => slice,
			TemaData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

// ========== Parameter Structs ==========

#[derive(Debug, Clone, Copy)]
pub struct TemaParams {
	pub period: Option<usize>,
}

impl Default for TemaParams {
	fn default() -> Self {
		Self { period: Some(9) }
	}
}

// ========== Output ==========

#[derive(Debug, Clone)]
pub struct TemaOutput {
	pub values: Vec<f64>,
}

// ========== Error Types ==========

#[derive(Debug, Error)]
pub enum TemaError {
	#[error("tema: Input data slice is empty.")]
	EmptyInputData,
	#[error("tema: All values are NaN.")]
	AllValuesNaN,
	#[error("tema: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("tema: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
}

// ========== Builder ==========

#[derive(Copy, Clone, Debug)]
pub struct TemaBuilder {
	period: Option<usize>,
	kernel: Kernel,
}

impl Default for TemaBuilder {
	fn default() -> Self {
		Self {
			period: None,
			kernel: Kernel::Auto,
		}
	}
}

impl TemaBuilder {
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
	pub fn apply(self, c: &Candles) -> Result<TemaOutput, TemaError> {
		let p = TemaParams { period: self.period };
		let i = TemaInput::from_candles(c, "close", p);
		tema_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slice(self, d: &[f64]) -> Result<TemaOutput, TemaError> {
		let p = TemaParams { period: self.period };
		let i = TemaInput::from_slice(d, p);
		tema_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<TemaStream, TemaError> {
		let p = TemaParams { period: self.period };
		TemaStream::try_new(p)
	}
}

// ========== Indicator API ==========

#[inline]
pub fn tema(input: &TemaInput) -> Result<TemaOutput, TemaError> {
	tema_with_kernel(input, Kernel::Auto)
}

pub fn tema_with_kernel(input: &TemaInput, kernel: Kernel) -> Result<TemaOutput, TemaError> {
	let data: &[f64] = match &input.data {
		TemaData::Candles { candles, source } => source_type(candles, source),
		TemaData::Slice(sl) => sl,
	};

	if data.is_empty() {
		return Err(TemaError::EmptyInputData);
	}

	let first = data.iter().position(|x| !x.is_nan()).ok_or(TemaError::AllValuesNaN)?;
	let len = data.len();
	let period = input.get_period();

	if period == 0 || period > len {
		return Err(TemaError::InvalidPeriod { period, data_len: len });
	}
	if (len - first) < period {
		return Err(TemaError::NotEnoughValidData {
			needed: period,
			valid: len - first,
		});
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	let lookback = (period - 1) * 3;
	let warm = first + lookback;

	let mut out = alloc_with_nan_prefix(len, warm);
	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => tema_scalar(data, period, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => tema_avx2(data, period, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => tema_avx512(data, period, first, &mut out),
			_ => unreachable!(),
		}
	}

	Ok(TemaOutput { values: out })
}

// ========== Scalar Implementation ==========

#[inline]
pub fn tema_scalar(data: &[f64], period: usize, first_val: usize, out: &mut [f64]) {
	let n = data.len();
	let lookback = (period - 1) * 3;
	let per = 2.0 / (period as f64 + 1.0);
	let per1 = 1.0 - per;

	let mut ema1 = data[first_val];
	let mut ema2 = 0.0;
	let mut ema3 = 0.0;

	for i in first_val..n {
		let price = data[i];

		ema1 = ema1 * per1 + price * per;
		if i == first_val + (period - 1) {
			ema2 = ema1;
		}
		if i >= first_val + (period - 1) {
			ema2 = ema2 * per1 + ema1 * per;
		}
		if i == first_val + 2 * (period - 1) {
			ema3 = ema2;
		}
		if i >= first_val + 2 * (period - 1) {
			ema3 = ema3 * per1 + ema2 * per;
		}
		if i >= first_val + lookback {
			out[i] = 3.0 * ema1 - 3.0 * ema2 + ema3;
		}
	}
}

// ========== AVX2/AVX512 Stubs ==========

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn tema_avx512(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	unsafe { tema_avx512_long(data, period, first_valid, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn tema_avx2(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	tema_scalar(data, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn tema_avx512_short(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	tema_scalar(data, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn tema_avx512_long(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	tema_scalar(data, period, first_valid, out)
}

// ========== Streaming ==========

#[derive(Debug, Clone)]
pub struct TemaStream {
	period: usize,
	buf: Vec<f64>,
	ema1: f64,
	ema2: f64,
	ema3: f64,
	pos: usize,
	filled: bool,
	step: usize,
	per: f64,
	per1: f64,
	valid: usize,
}

impl TemaStream {
	pub fn try_new(params: TemaParams) -> Result<Self, TemaError> {
		let period = params.period.unwrap_or(9);
		if period == 0 {
			return Err(TemaError::InvalidPeriod { period, data_len: 0 });
		}
		Ok(Self {
			period,
			buf: vec![f64::NAN; period],
			ema1: f64::NAN,
			ema2: 0.0,
			ema3: 0.0,
			pos: 0,
			filled: false,
			step: 0,
			per: 2.0 / (period as f64 + 1.0),
			per1: 1.0 - (2.0 / (period as f64 + 1.0)),
			valid: 0,
		})
	}
	#[inline(always)]
	pub fn update(&mut self, value: f64) -> Option<f64> {
		if self.filled {
			self.ema1 = self.ema1 * self.per1 + value * self.per;
			self.ema2 = self.ema2 * self.per1 + self.ema1 * self.per;
			self.ema3 = self.ema3 * self.per1 + self.ema2 * self.per;
			let tema_val = 3.0 * self.ema1 - 3.0 * self.ema2 + self.ema3;
			return Some(tema_val);
		}

		if self.valid == 0 {
			self.ema1 = value;
			self.valid += 1;
			self.buf[self.pos] = value;
			self.pos = (self.pos + 1) % self.period;
			return None;
		}

		self.ema1 = self.ema1 * self.per1 + value * self.per;
		self.buf[self.pos] = value;
		self.pos = (self.pos + 1) % self.period;
		self.valid += 1;

		if self.valid == self.period {
			self.ema2 = self.ema1;
		} else if self.valid > self.period {
			self.ema2 = self.ema2 * self.per1 + self.ema1 * self.per;
		}

		if self.valid == 2 * self.period - 1 {
			self.ema3 = self.ema2;
		} else if self.valid > 2 * self.period - 1 {
			self.ema3 = self.ema3 * self.per1 + self.ema2 * self.per;
		}

		if self.valid > (self.period - 1) * 3 {
			self.filled = true;
			let tema_val = 3.0 * self.ema1 - 3.0 * self.ema2 + self.ema3;
			Some(tema_val)
		} else {
			None
		}
	}
}

// ========== Batch Processing ==========

#[derive(Clone, Debug)]
pub struct TemaBatchRange {
	pub period: (usize, usize, usize),
}

impl Default for TemaBatchRange {
	fn default() -> Self {
		Self { period: (9, 240, 1) }
	}
}

#[derive(Clone, Debug, Default)]
pub struct TemaBatchBuilder {
	range: TemaBatchRange,
	kernel: Kernel,
}

impl TemaBatchBuilder {
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
	pub fn apply_slice(self, data: &[f64]) -> Result<TemaBatchOutput, TemaError> {
		tema_batch_with_kernel(data, &self.range, self.kernel)
	}
	pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<TemaBatchOutput, TemaError> {
		TemaBatchBuilder::new().kernel(k).apply_slice(data)
	}
	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<TemaBatchOutput, TemaError> {
		let slice = source_type(c, src);
		self.apply_slice(slice)
	}
	pub fn with_default_candles(c: &Candles) -> Result<TemaBatchOutput, TemaError> {
		TemaBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
	}
}

pub fn tema_batch_with_kernel(data: &[f64], sweep: &TemaBatchRange, k: Kernel) -> Result<TemaBatchOutput, TemaError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => {
			return Err(TemaError::InvalidPeriod { period: 0, data_len: 0 });
		}
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	tema_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct TemaBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<TemaParams>,
	pub rows: usize,
	pub cols: usize,
}
impl TemaBatchOutput {
	pub fn row_for_params(&self, p: &TemaParams) -> Option<usize> {
		self.combos
			.iter()
			.position(|c| c.period.unwrap_or(9) == p.period.unwrap_or(9))
	}
	pub fn values_for(&self, p: &TemaParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &TemaBatchRange) -> Vec<TemaParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let periods = axis_usize(r.period);
	let mut out = Vec::with_capacity(periods.len());
	for &p in &periods {
		out.push(TemaParams { period: Some(p) });
	}
	out
}

#[inline(always)]
pub fn tema_batch_slice(data: &[f64], sweep: &TemaBatchRange, kern: Kernel) -> Result<TemaBatchOutput, TemaError> {
	tema_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn tema_batch_par_slice(data: &[f64], sweep: &TemaBatchRange, kern: Kernel) -> Result<TemaBatchOutput, TemaError> {
	tema_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn tema_batch_inner(
	data: &[f64],
	sweep: &TemaBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<TemaBatchOutput, TemaError> {
	// ---------- 0. parameter checks ----------
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(TemaError::InvalidPeriod { period: 0, data_len: 0 });
	}

	let first = data.iter().position(|x| !x.is_nan()).ok_or(TemaError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if data.len() - first < max_p {
		return Err(TemaError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}

	// ---------- 1. matrix dimensions ----------
	let rows = combos.len();
	let cols = data.len();

	// ---------- 2. build per-row warm-up lengths ----------
	let warm: Vec<usize> = combos
		.iter()
		.map(|c| (first + (c.period.unwrap() - 1) * 3).min(cols)) // (period-1)*3 matches tema_scalar, clamped to cols
		.collect();

	// ---------- 3. allocate rows×cols uninitialised, fill NaN prefixes ----------
	let mut raw = make_uninit_matrix(rows, cols);
	unsafe { init_matrix_prefixes(&mut raw, cols, &warm) };

	// ---------- 4. closure that fills ONE row ----------
	let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
		let period = combos[row].period.unwrap();

		// cast this row to &mut [f64] so the row-kernel can write normally
		let out_row = core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

		match kern {
			Kernel::Scalar => tema_row_scalar(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => tema_row_avx2(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => tema_row_avx512(data, first, period, out_row),
			_ => unreachable!(),
		}
	};

	// ---------- 5. run all rows (optionally in parallel) ----------
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

	// ---------- 6. transmute to fully-initialised Vec<f64> ----------
	let values: Vec<f64> = unsafe { std::mem::transmute(raw) };

	// ---------- 7. package ----------
	Ok(TemaBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
unsafe fn tema_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	tema_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
unsafe fn tema_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	tema_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
unsafe fn tema_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	if period <= 32 {
		tema_row_avx512_short(data, first, period, out)
	} else {
		tema_row_avx512_long(data, first, period, out)
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
unsafe fn tema_row_avx512_short(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	tema_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
unsafe fn tema_row_avx512_long(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	tema_scalar(data, period, first, out)
}

#[inline(always)]
fn expand_grid_tema(r: &TemaBatchRange) -> Vec<TemaParams> {
	let mut out = Vec::new();
	let (start, end, step) = r.period;
	if step == 0 || start == end {
		out.push(TemaParams { period: Some(start) });
	} else {
		let mut p = start;
		while p <= end {
			out.push(TemaParams { period: Some(p) });
			p += step;
		}
	}
	out
}

// ========== Unit Tests ==========

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_tema_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let default_params = TemaParams { period: None };
		let input = TemaInput::from_candles(&candles, "close", default_params);
		let output = tema_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}
	fn check_tema_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = TemaInput::from_candles(&candles, "close", TemaParams::default());
		let result = tema_with_kernel(&input, kernel)?;
		let expected_last_five = [
			59281.895570662884,
			59257.25021607971,
			59172.23342859784,
			59175.218345941066,
			58934.24395798363,
		];
		let start = result.values.len().saturating_sub(5);
		for (i, &val) in result.values[start..].iter().enumerate() {
			let diff = (val - expected_last_five[i]).abs();
			assert!(
				diff < 1e-8,
				"[{}] TEMA {:?} mismatch at idx {}: got {}, expected {}",
				test_name,
				kernel,
				i,
				val,
				expected_last_five[i]
			);
		}
		Ok(())
	}
	fn check_tema_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = TemaInput::with_default_candles(&candles);
		match input.data {
			TemaData::Candles { source, .. } => assert_eq!(source, "close"),
			_ => panic!("Expected TemaData::Candles"),
		}
		let output = tema_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}
	fn check_tema_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let input_data = [10.0, 20.0, 30.0];
		let params = TemaParams { period: Some(0) };
		let input = TemaInput::from_slice(&input_data, params);
		let res = tema_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] TEMA should fail with zero period", test_name);
		Ok(())
	}
	fn check_tema_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let input_data: [f64; 0] = [];
		let params = TemaParams { period: Some(9) };
		let input = TemaInput::from_slice(&input_data, params);
		let res = tema_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] TEMA should fail with empty input", test_name);
		if let Err(e) = res {
			assert!(
				matches!(e, TemaError::EmptyInputData),
				"[{}] Expected EmptyInputData error",
				test_name
			);
		}
		Ok(())
	}
	fn check_tema_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data_small = [10.0, 20.0, 30.0];
		let params = TemaParams { period: Some(10) };
		let input = TemaInput::from_slice(&data_small, params);
		let res = tema_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] TEMA should fail with period exceeding length",
			test_name
		);
		Ok(())
	}
	fn check_tema_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let single_point = [42.0];
		let params = TemaParams { period: Some(9) };
		let input = TemaInput::from_slice(&single_point, params);
		let res = tema_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] TEMA should fail with insufficient data", test_name);
		Ok(())
	}
	fn check_tema_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let first_params = TemaParams { period: Some(9) };
		let first_input = TemaInput::from_candles(&candles, "close", first_params);
		let first_result = tema_with_kernel(&first_input, kernel)?;
		let second_params = TemaParams { period: Some(9) };
		let second_input = TemaInput::from_slice(&first_result.values, second_params);
		let second_result = tema_with_kernel(&second_input, kernel)?;
		assert_eq!(second_result.values.len(), first_result.values.len());
		Ok(())
	}
	fn check_tema_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = TemaInput::from_candles(&candles, "close", TemaParams { period: Some(9) });
		let res = tema_with_kernel(&input, kernel)?;
		assert_eq!(res.values.len(), candles.close.len());
		if res.values.len() > 50 {
			for (i, &val) in res.values[50..].iter().enumerate() {
				assert!(
					!val.is_nan(),
					"[{}] Found unexpected NaN at out-index {}",
					test_name,
					50 + i
				);
			}
		}
		Ok(())
	}
	fn check_tema_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let period = 9;
		let input = TemaInput::from_candles(&candles, "close", TemaParams { period: Some(period) });
		let batch_output = tema_with_kernel(&input, kernel)?.values;
		let mut stream = TemaStream::try_new(TemaParams { period: Some(period) })?;
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
				diff < 1e-9,
				"[{}] TEMA streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
				test_name,
				i,
				b,
				s,
				diff
			);
		}
		Ok(())
	}

	macro_rules! generate_all_tema_tests {
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

	// Check for poison values in single output - only runs in debug mode
	#[cfg(debug_assertions)]
	fn check_tema_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		// Test with multiple parameter combinations to better catch uninitialized memory bugs
		let test_periods = vec![5, 9, 14, 20, 50, 100, 200];

		for &period in &test_periods {
			let params = TemaParams { period: Some(period) };
			let input = TemaInput::from_candles(&candles, "close", params);

			// Skip if we don't have enough data for this period
			if candles.close.len() < period {
				continue;
			}

			let output = match tema_with_kernel(&input, kernel) {
				Ok(o) => o,
				Err(_) => continue, // Skip if this period causes an error
			};

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
	fn check_tema_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(())
	}

	generate_all_tema_tests!(
		check_tema_partial_params,
		check_tema_accuracy,
		check_tema_default_candles,
		check_tema_zero_period,
		check_tema_empty_input,
		check_tema_period_exceeds_length,
		check_tema_very_small_dataset,
		check_tema_reinput,
		check_tema_nan_handling,
		check_tema_streaming,
		check_tema_no_poison
	);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let output = TemaBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;
		let def = TemaParams::default();
		let row = output.values_for(&def).expect("default row missing");
		assert_eq!(row.len(), c.close.len());
		let expected = [
			59281.895570662884,
			59257.25021607971,
			59172.23342859784,
			59175.218345941066,
			58934.24395798363,
		];
		let start = row.len() - 5;
		for (i, &v) in row[start..].iter().enumerate() {
			assert!(
				(v - expected[i]).abs() < 1e-8,
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

		// Test multiple batch configurations with different period ranges
		let test_configs = vec![
			(5, 15, 2),    // Small periods with fine steps
			(10, 50, 5),   // Medium periods
			(20, 100, 10), // Large periods
			(50, 200, 25), // Very large periods
			(3, 3, 1),     // Single small period
			(150, 150, 1), // Single large period
		];

		for (start, end, step) in test_configs {
			let output = TemaBatchBuilder::new()
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
				let period = output.combos.get(row).map(|p| p.period.unwrap_or(0)).unwrap_or(0);

				// Check for alloc_with_nan_prefix poison (0x11111111_11111111)
				if bits == 0x11111111_11111111 {
					panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (period {}, flat index {})",
                        test, val, bits, row, col, period, idx
                    );
				}

				// Check for init_matrix_prefixes poison (0x22222222_22222222)
				if bits == 0x22222222_22222222 {
					panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (period {}, flat index {})",
                        test, val, bits, row, col, period, idx
                    );
				}

				// Check for make_uninit_matrix poison (0x33333333_33333333)
				if bits == 0x33333333_33333333 {
					panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (period {}, flat index {})",
                        test, val, bits, row, col, period, idx
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

// ========== Helper Functions for Bindings ==========

/// Centralized validation and preparation for TEMA calculation
#[inline]
fn tema_prepare<'a>(
	input: &'a TemaInput,
	kernel: Kernel,
) -> Result<(&'a [f64], usize, usize, usize, Kernel), TemaError> {
	let data: &[f64] = match &input.data {
		TemaData::Candles { candles, source } => source_type(candles, source),
		TemaData::Slice(sl) => sl,
	};

	if data.is_empty() {
		return Err(TemaError::EmptyInputData);
	}

	let first = data.iter().position(|x| !x.is_nan()).ok_or(TemaError::AllValuesNaN)?;
	let len = data.len();
	let period = input.get_period();

	if period == 0 || period > len {
		return Err(TemaError::InvalidPeriod { period, data_len: len });
	}
	if (len - first) < period {
		return Err(TemaError::NotEnoughValidData {
			needed: period,
			valid: len - first,
		});
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	Ok((data, period, first, len, chosen))
}

/// Compute TEMA directly into pre-allocated output buffer
#[inline]
fn tema_compute_into(data: &[f64], period: usize, first: usize, chosen: Kernel, out: &mut [f64]) {
	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => tema_scalar(data, period, first, out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => tema_avx2(data, period, first, out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => tema_avx512(data, period, first, out),
			#[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
			Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
				// Fallback to scalar when AVX is not available
				tema_scalar(data, period, first, out)
			}
			Kernel::Auto => unreachable!(),
		}
	}
}

/// Optimized batch calculation that writes directly to pre-allocated buffer
#[inline(always)]
fn tema_batch_inner_into(
	data: &[f64],
	sweep: &TemaBatchRange,
	kern: Kernel,
	parallel: bool,
	out: &mut [f64],
) -> Result<Vec<TemaParams>, TemaError> {
	// ---------- 0. parameter checks ----------
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(TemaError::InvalidPeriod { period: 0, data_len: 0 });
	}

	if data.is_empty() {
		return Err(TemaError::EmptyInputData);
	}

	let first = data.iter().position(|x| !x.is_nan()).ok_or(TemaError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if data.len() - first < max_p {
		return Err(TemaError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}

	// ---------- 1. matrix dimensions ----------
	let rows = combos.len();
	let cols = data.len();

	// ---------- 2. build per-row warm-up lengths ----------
	let warm: Vec<usize> = combos
		.iter()
		.map(|c| (first + (c.period.unwrap() - 1) * 3).min(cols)) // (period-1)*3 matches tema_scalar, clamped to cols
		.collect();

	// ---------- 3. reinterpret output slice as MaybeUninit for efficient initialization ----------
	let out_uninit = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len()) };

	unsafe { init_matrix_prefixes(out_uninit, cols, &warm) };

	// ---------- 4. closure that fills ONE row ----------
	let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
		let period = combos[row].period.unwrap();

		// cast this row to &mut [f64] so the row-kernel can write normally
		let out_row = core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

		match kern {
			Kernel::Scalar => tema_row_scalar(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => tema_row_avx2(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => tema_row_avx512(data, first, period, out_row),
			#[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
			Kernel::Avx2 | Kernel::Avx512 => tema_row_scalar(data, first, period, out_row),
			_ => unreachable!(),
		}
	};

	// ---------- 5. run all rows (optionally in parallel) ----------
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

// ========== Python Bindings ==========

#[cfg(feature = "python")]
#[pyfunction(name = "tema")]
#[pyo3(signature = (data, period, kernel=None))]
/// Compute the Triple Exponential Moving Average (TEMA) of the input data.
///
/// TEMA applies three exponential moving averages in succession to reduce lag and noise.
/// It is calculated as: TEMA = 3*EMA1 - 3*EMA2 + EMA3.
///
/// Parameters:
/// -----------
/// data : np.ndarray
///     Input data array (float64).
/// period : int
///     Number of data points in the moving average window (must be >= 1).
/// kernel : str, optional
///     Computation kernel to use: 'auto', 'scalar', 'avx2', 'avx512'.
///     Default is 'auto' which auto-detects the best available.
///
/// Returns:
/// --------
/// np.ndarray
///     Array of TEMA values, same length as input.
///
/// Raises:
/// -------
/// ValueError
///     If inputs are invalid (period is zero, exceeds data length, etc).
pub fn tema_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	period: usize,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
	use numpy::{PyArray1, PyArrayMethods};

	let slice_in = data.as_slice()?; // zero-copy, read-only view

	// Parse kernel string to enum
	let kern = match kernel {
		None | Some("auto") => Kernel::Auto,
		Some("scalar") => Kernel::Scalar,
		Some("avx2") => Kernel::Avx2,
		Some("avx512") => Kernel::Avx512,
		Some(k) => return Err(PyValueError::new_err(format!("Unknown kernel: {}", k))),
	};

	// ---------- build input struct -------------------------------------------------
	let params = TemaParams { period: Some(period) };
	let tema_in = TemaInput::from_slice(slice_in, params);

	// ---------- allocate NumPy output buffer ---------------------------------------
	let out_arr = unsafe { PyArray1::<f64>::new(py, [slice_in.len()], false) };
	let slice_out = unsafe { out_arr.as_slice_mut()? }; // safe: we own the array

	// ---------- heavy lifting without the GIL --------------------------------------
	py.allow_threads(|| -> Result<(), TemaError> {
		let (data, period, first, _len, chosen) = tema_prepare(&tema_in, kern)?;
		// Initialize NaN prefix
		let lookback = (period - 1) * 3;
		let warm = (first + lookback).min(slice_out.len());
		slice_out[..warm].fill(f64::NAN);
		// Compute TEMA
		tema_compute_into(data, period, first, chosen, slice_out);
		Ok(())
	})
	.map_err(|e| PyValueError::new_err(e.to_string()))?; // unify error type

	Ok(out_arr.into())
}

#[cfg(feature = "python")]
#[pyclass(name = "TemaStream")]
pub struct TemaStreamPy {
	stream: TemaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl TemaStreamPy {
	#[new]
	fn new(period: usize) -> PyResult<Self> {
		let params = TemaParams { period: Some(period) };
		let stream = TemaStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(TemaStreamPy { stream })
	}

	/// Updates the stream with a new value and returns the calculated TEMA value.
	/// Returns `None` if the buffer is not yet full.
	fn update(&mut self, value: f64) -> Option<f64> {
		self.stream.update(value)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "tema_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
/// Compute TEMA for multiple period values in a single pass.
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
pub fn tema_batch_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	period_range: (usize, usize, usize),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;

	let slice_in = data.as_slice()?;

	let sweep = TemaBatchRange { period: period_range };

	// 1. Expand grid once to know rows*cols
	let combos = expand_grid(&sweep);
	let rows = combos.len();
	let cols = slice_in.len();

	// 2. Pre-allocate NumPy array (1-D, will reshape later)
	let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let slice_out = unsafe { out_arr.as_slice_mut()? };

	// Parse kernel string to enum
	let kern = match kernel {
		None | Some("auto") => Kernel::Auto,
		Some("scalar") => Kernel::ScalarBatch,
		Some("avx2") => Kernel::Avx2Batch,
		Some("avx512") => Kernel::Avx512Batch,
		Some(k) => return Err(PyValueError::new_err(format!("Unknown kernel: {}", k))),
	};

	// 3. Heavy work without the GIL
	let combos = py
		.allow_threads(|| {
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
			// Use the _into variant that writes directly to our pre-allocated buffer
			tema_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
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

	Ok(dict)
}

// ========== WASM Bindings ==========

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn tema_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
	let params = TemaParams { period: Some(period) };
	let input = TemaInput::from_slice(data, params);

	tema_with_kernel(&input, Kernel::Scalar)
		.map(|o| o.values)
		.map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn tema_batch_js(
	data: &[f64],
	period_start: usize,
	period_end: usize,
	period_step: usize,
) -> Result<Vec<f64>, JsValue> {
	let sweep = TemaBatchRange {
		period: (period_start, period_end, period_step),
	};

	// Use the existing batch function with parallel=false for WASM
	tema_batch_inner(data, &sweep, Kernel::Scalar, false)
		.map(|output| output.values)
		.map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn tema_batch_metadata_js(period_start: usize, period_end: usize, period_step: usize) -> Result<Vec<f64>, JsValue> {
	let sweep = TemaBatchRange {
		period: (period_start, period_end, period_step),
	};

	let combos = expand_grid(&sweep);
	let metadata: Vec<f64> = combos.iter().map(|c| c.period.unwrap() as f64).collect();

	Ok(metadata)
}
