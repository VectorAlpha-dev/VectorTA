//! # Relative Strength Xtra (RSX)
//!
//! A smoothed oscillator similar to RSI that attempts to reduce lag and noise. The calculation uses an IIR filter approach for smoothing while retaining responsiveness.
//!
//! ## Parameters
//! - **period**: The lookback window for RSX calculations. Defaults to 14.
//!
//! ## Errors
//! - **AllValuesNaN**: rsx: All input data values are `NaN`.
//! - **InvalidPeriod**: rsx: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: rsx: Not enough valid data points for the requested `period`.
//!
//! ## Returns
//! - **`Ok(RsxOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(RsxError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use thiserror::Error;

impl<'a> AsRef<[f64]> for RsxInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			RsxData::Slice(slice) => slice,
			RsxData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

#[derive(Debug, Clone)]
pub enum RsxData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct RsxOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct RsxParams {
	pub period: Option<usize>,
}

impl Default for RsxParams {
	fn default() -> Self {
		Self { period: Some(14) }
	}
}

#[derive(Debug, Clone)]
pub struct RsxInput<'a> {
	pub data: RsxData<'a>,
	pub params: RsxParams,
}

impl<'a> RsxInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: RsxParams) -> Self {
		Self {
			data: RsxData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slice(sl: &'a [f64], p: RsxParams) -> Self {
		Self {
			data: RsxData::Slice(sl),
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", RsxParams::default())
	}
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(14)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct RsxBuilder {
	period: Option<usize>,
	kernel: Kernel,
}

impl Default for RsxBuilder {
	fn default() -> Self {
		Self {
			period: None,
			kernel: Kernel::Auto,
		}
	}
}

impl RsxBuilder {
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
	pub fn apply(self, c: &Candles) -> Result<RsxOutput, RsxError> {
		let p = RsxParams { period: self.period };
		let i = RsxInput::from_candles(c, "close", p);
		rsx_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slice(self, d: &[f64]) -> Result<RsxOutput, RsxError> {
		let p = RsxParams { period: self.period };
		let i = RsxInput::from_slice(d, p);
		rsx_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<RsxStream, RsxError> {
		let p = RsxParams { period: self.period };
		RsxStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum RsxError {
	#[error("rsx: All values are NaN.")]
	AllValuesNaN,

	#[error("rsx: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },

	#[error("rsx: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn rsx(input: &RsxInput) -> Result<RsxOutput, RsxError> {
	rsx_with_kernel(input, Kernel::Auto)
}

pub fn rsx_with_kernel(input: &RsxInput, kernel: Kernel) -> Result<RsxOutput, RsxError> {
	let data: &[f64] = match &input.data {
		RsxData::Candles { candles, source } => source_type(candles, source),
		RsxData::Slice(sl) => sl,
	};

	let first = data.iter().position(|x| !x.is_nan()).ok_or(RsxError::AllValuesNaN)?;

	let len = data.len();
	let period = input.get_period();

	if period == 0 || period > len {
		return Err(RsxError::InvalidPeriod { period, data_len: len });
	}
	if (len - first) < period {
		return Err(RsxError::NotEnoughValidData {
			needed: period,
			valid: len - first,
		});
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	let mut out = vec![f64::NAN; len];
	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => rsx_scalar(data, period, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => rsx_avx2(data, period, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => rsx_avx512(data, period, first, &mut out),
			_ => unreachable!(),
		}
	}

	Ok(RsxOutput { values: out })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn rsx_avx512(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	if period <= 32 {
		unsafe { rsx_avx512_short(data, period, first_valid, out) }
	} else {
		unsafe { rsx_avx512_long(data, period, first_valid, out) }
	}
}

#[inline]
pub fn rsx_scalar(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
	let mut f0 = 0.0;
	let mut f8 = 0.0;
	let mut f18 = 0.0;
	let mut f20 = 0.0;
	let mut f28 = 0.0;
	let mut f30 = 0.0;
	let mut f38 = 0.0;
	let mut f40 = 0.0;
	let mut f48 = 0.0;
	let mut f50 = 0.0;
	let mut f58 = 0.0;
	let mut f60 = 0.0;
	let mut f68 = 0.0;
	let mut f70 = 0.0;
	let mut f78 = 0.0;
	let mut f80 = 0.0;
	let mut f88 = 0.0;
	let mut f90 = 0.0;
	let mut is_initialized = false;

	let start_calc_idx = first + period - 1;
	for i in start_calc_idx..data.len() {
		let val = data[i];
		if !is_initialized {
			f90 = 1.0;
			f0 = 0.0;
			f88 = if period >= 6 { (period - 1) as f64 } else { 5.0 };
			f8 = 100.0 * val;
			f18 = 3.0 / (period as f64 + 2.0);
			f20 = 1.0 - f18;
			out[i] = f64::NAN;
			is_initialized = true;
		} else {
			f90 = if f88 <= f90 { f88 + 1.0 } else { f90 + 1.0 };
			let f10 = f8;
			f8 = 100.0 * val;
			let v8 = f8 - f10;
			f28 = f20 * f28 + f18 * v8;
			f30 = f18 * f28 + f20 * f30;
			let v_c = f28 * 1.5 - f30 * 0.5;
			f38 = f20 * f38 + f18 * v_c;
			f40 = f18 * f38 + f20 * f40;
			let v10 = f38 * 1.5 - f40 * 0.5;
			f48 = f20 * f48 + f18 * v10;
			f50 = f18 * f48 + f20 * f50;
			let v14 = f48 * 1.5 - f50 * 0.5;
			f58 = f20 * f58 + f18 * v8.abs();
			f60 = f18 * f58 + f20 * f60;
			let v18 = f58 * 1.5 - f60 * 0.5;
			f68 = f20 * f68 + f18 * v18;
			f70 = f18 * f68 + f20 * f70;
			let v1c = f68 * 1.5 - f70 * 0.5;
			f78 = f20 * f78 + f18 * v1c;
			f80 = f18 * f78 + f20 * f80;
			let v20_ = f78 * 1.5 - f80 * 0.5;

			if f88 >= f90 && f8 != f10 {
				f0 = 1.0;
			}
			if (f88 - f90).abs() < f64::EPSILON && f0 == 0.0 {
				f90 = 0.0;
			}

			if f88 < f90 && v20_ > 1e-10 {
				let mut v4 = (v14 / v20_ + 1.0) * 50.0;
				if v4 > 100.0 {
					v4 = 100.0;
				}
				if v4 < 0.0 {
					v4 = 0.0;
				}
				out[i] = v4;
			} else {
				out[i] = 50.0;
			}
		}
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn rsx_avx2(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
	rsx_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn rsx_avx512_short(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
	rsx_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn rsx_avx512_long(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
	rsx_scalar(data, period, first, out)
}

#[derive(Debug, Clone)]
pub struct RsxStream {
	period: usize,
	buf: Vec<f64>,
	head: usize,
	filled: bool,
	state: Option<[f64; 18]>,
}

impl RsxStream {
	pub fn try_new(params: RsxParams) -> Result<Self, RsxError> {
		let period = params.period.unwrap_or(14);
		if period == 0 {
			return Err(RsxError::InvalidPeriod { period, data_len: 0 });
		}
		Ok(Self {
			period,
			buf: vec![f64::NAN; period],
			head: 0,
			filled: false,
			state: None,
		})
	}

	#[inline(always)]
	pub fn update(&mut self, value: f64) -> Option<f64> {
		self.buf[self.head] = value;
		self.head = (self.head + 1) % self.period;

		if !self.filled && self.head == 0 {
			self.filled = true;
			return Some(self.ring_calc(value));
		}

		if !self.filled {
			return None;
		}

		Some(self.ring_calc(value))
	}

	#[inline(always)]
	fn ring_calc(&mut self, val: f64) -> f64 {
		let period = self.period;
		let mut s = self.state.unwrap_or([0.0; 18]);
		let mut out_val = f64::NAN;
		let mut is_initialized = self.state.is_some();

		if !is_initialized {
			s[16] = if period >= 6 { (period - 1) as f64 } else { 5.0 };
			s[17] = 1.0;
			s[1] = 100.0 * val;
			s[2] = 3.0 / (period as f64 + 2.0);
			s[3] = 1.0 - s[2];
			is_initialized = true;
			out_val = f64::NAN;
		} else {
			s[17] = if s[16] <= s[17] { s[16] + 1.0 } else { s[17] + 1.0 };
			let prev = s[1];
			s[1] = 100.0 * val;
			let v8 = s[1] - prev;
			s[4] = s[3] * s[4] + s[2] * v8;
			s[5] = s[2] * s[4] + s[3] * s[5];
			let v_c = s[4] * 1.5 - s[5] * 0.5;
			s[6] = s[3] * s[6] + s[2] * v_c;
			s[7] = s[2] * s[6] + s[3] * s[7];
			let v10 = s[6] * 1.5 - s[7] * 0.5;
			s[8] = s[3] * s[8] + s[2] * v10;
			s[9] = s[2] * s[8] + s[3] * s[9];
			let v14 = s[8] * 1.5 - s[9] * 0.5;
			s[10] = s[3] * s[10] + s[2] * v8.abs();
			s[11] = s[2] * s[10] + s[3] * s[11];
			let v18 = s[10] * 1.5 - s[11] * 0.5;
			s[12] = s[3] * s[12] + s[2] * v18;
			s[13] = s[2] * s[12] + s[3] * s[13];
			let v1c = s[12] * 1.5 - s[13] * 0.5;
			s[14] = s[3] * s[14] + s[2] * v1c;
			s[15] = s[2] * s[14] + s[3] * s[15];
			let v20_ = s[14] * 1.5 - s[15] * 0.5;
			if s[16] >= s[17] && s[1] != prev {
				s[0] = 1.0;
			}
			if (s[16] - s[17]).abs() < f64::EPSILON && s[0] == 0.0 {
				s[17] = 0.0;
			}
			if s[16] < s[17] && v20_ > 1e-10 {
				let mut v4 = (v14 / v20_ + 1.0) * 50.0;
				if v4 > 100.0 {
					v4 = 100.0;
				}
				if v4 < 0.0 {
					v4 = 0.0;
				}
				out_val = v4;
			} else {
				out_val = 50.0;
			}
		}
		self.state = Some(s);
		out_val
	}
}

#[derive(Clone, Debug)]
pub struct RsxBatchRange {
	pub period: (usize, usize, usize),
}

impl Default for RsxBatchRange {
	fn default() -> Self {
		Self { period: (14, 100, 1) }
	}
}

#[derive(Clone, Debug, Default)]
pub struct RsxBatchBuilder {
	range: RsxBatchRange,
	kernel: Kernel,
}

impl RsxBatchBuilder {
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

	pub fn apply_slice(self, data: &[f64]) -> Result<RsxBatchOutput, RsxError> {
		rsx_batch_with_kernel(data, &self.range, self.kernel)
	}

	pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<RsxBatchOutput, RsxError> {
		RsxBatchBuilder::new().kernel(k).apply_slice(data)
	}

	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<RsxBatchOutput, RsxError> {
		let slice = source_type(c, src);
		self.apply_slice(slice)
	}

	pub fn with_default_candles(c: &Candles) -> Result<RsxBatchOutput, RsxError> {
		RsxBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
	}
}

pub fn rsx_batch_with_kernel(data: &[f64], sweep: &RsxBatchRange, k: Kernel) -> Result<RsxBatchOutput, RsxError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(RsxError::InvalidPeriod { period: 0, data_len: 0 }),
	};

	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	rsx_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct RsxBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<RsxParams>,
	pub rows: usize,
	pub cols: usize,
}
impl RsxBatchOutput {
	pub fn row_for_params(&self, p: &RsxParams) -> Option<usize> {
		self.combos
			.iter()
			.position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
	}

	pub fn values_for(&self, p: &RsxParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &RsxBatchRange) -> Vec<RsxParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let periods = axis_usize(r.period);
	let mut out = Vec::with_capacity(periods.len());
	for &p in &periods {
		out.push(RsxParams { period: Some(p) });
	}
	out
}

#[inline(always)]
pub fn rsx_batch_slice(data: &[f64], sweep: &RsxBatchRange, kern: Kernel) -> Result<RsxBatchOutput, RsxError> {
	rsx_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn rsx_batch_par_slice(data: &[f64], sweep: &RsxBatchRange, kern: Kernel) -> Result<RsxBatchOutput, RsxError> {
	rsx_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn rsx_batch_inner(
	data: &[f64],
	sweep: &RsxBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<RsxBatchOutput, RsxError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(RsxError::InvalidPeriod { period: 0, data_len: 0 });
	}

	let first = data.iter().position(|x| !x.is_nan()).ok_or(RsxError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if data.len() - first < max_p {
		return Err(RsxError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}

	let rows = combos.len();
	let cols = data.len();
	let mut values = vec![f64::NAN; rows * cols];

	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let period = combos[row].period.unwrap();
		match kern {
			Kernel::Scalar => rsx_row_scalar(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => rsx_row_avx2(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => rsx_row_avx512(data, first, period, out_row),
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

	Ok(RsxBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
unsafe fn rsx_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	rsx_scalar(data, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn rsx_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	rsx_avx2(data, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn rsx_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	rsx_avx512(data, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn rsx_row_avx512_short(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	rsx_avx512_short(data, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn rsx_row_avx512_long(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	rsx_avx512_long(data, period, first, out);
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_rsx_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let default_params = RsxParams { period: None };
		let input_default = RsxInput::from_candles(&candles, "close", default_params);
		let output_default = rsx_with_kernel(&input_default, kernel)?;
		assert_eq!(output_default.values.len(), candles.close.len());

		let params_period_10 = RsxParams { period: Some(10) };
		let input_period_10 = RsxInput::from_candles(&candles, "hl2", params_period_10);
		let output_period_10 = rsx_with_kernel(&input_period_10, kernel)?;
		assert_eq!(output_period_10.values.len(), candles.close.len());

		let params_custom = RsxParams { period: Some(20) };
		let input_custom = RsxInput::from_candles(&candles, "hlc3", params_custom);
		let output_custom = rsx_with_kernel(&input_custom, kernel)?;
		assert_eq!(output_custom.values.len(), candles.close.len());

		Ok(())
	}

	fn check_rsx_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = RsxParams { period: Some(14) };
		let input = RsxInput::from_candles(&candles, "close", params);
		let rsx_result = rsx_with_kernel(&input, kernel)?;

		let expected_last_five_rsx = [
			46.11486311289701,
			46.88048640321688,
			47.174443049619995,
			47.48751360654475,
			46.552886446171684,
		];
		let start_index = rsx_result.values.len() - 5;
		let result_last_five_rsx = &rsx_result.values[start_index..];
		for (i, &value) in result_last_five_rsx.iter().enumerate() {
			let expected_value = expected_last_five_rsx[i];
			assert!(
				(value - expected_value).abs() < 1e-1,
				"RSX mismatch at index {}: expected {}, got {}",
				i,
				expected_value,
				value
			);
		}
		Ok(())
	}

	fn check_rsx_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = RsxInput::with_default_candles(&candles);
		match input.data {
			RsxData::Candles { source, .. } => assert_eq!(source, "close"),
			_ => panic!("Expected RsxData::Candles"),
		}
		let output = rsx_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_rsx_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let input_data = [10.0, 20.0, 30.0];
		let params = RsxParams { period: Some(0) };
		let input = RsxInput::from_slice(&input_data, params);
		let res = rsx_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] RSX should fail with zero period", test_name);
		Ok(())
	}

	fn check_rsx_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data_small = [10.0, 20.0, 30.0];
		let params = RsxParams { period: Some(10) };
		let input = RsxInput::from_slice(&data_small, params);
		let res = rsx_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] RSX should fail with period exceeding length",
			test_name
		);
		Ok(())
	}

	fn check_rsx_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let single_point = [42.0];
		let params = RsxParams { period: Some(14) };
		let input = RsxInput::from_slice(&single_point, params);
		let res = rsx_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] RSX should fail with insufficient data", test_name);
		Ok(())
	}

	fn check_rsx_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let first_params = RsxParams { period: Some(14) };
		let first_input = RsxInput::from_candles(&candles, "close", first_params);
		let first_result = rsx_with_kernel(&first_input, kernel)?;

		let second_params = RsxParams { period: Some(14) };
		let second_input = RsxInput::from_slice(&first_result.values, second_params);
		let second_result = rsx_with_kernel(&second_input, kernel)?;

		assert_eq!(second_result.values.len(), first_result.values.len());
		Ok(())
	}

	fn check_rsx_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let input = RsxInput::from_candles(&candles, "close", RsxParams { period: Some(14) });
		let res = rsx_with_kernel(&input, kernel)?;
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

	fn check_rsx_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let period = 14;

		let input = RsxInput::from_candles(&candles, "close", RsxParams { period: Some(period) });
		let batch_output = rsx_with_kernel(&input, kernel)?.values;

		let mut stream = RsxStream::try_new(RsxParams { period: Some(period) })?;

		let mut stream_values = Vec::with_capacity(candles.close.len());
		for &price in &candles.close {
			match stream.update(price) {
				Some(rsx_val) => stream_values.push(rsx_val),
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
				"[{}] RSX streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
				test_name,
				i,
				b,
				s,
				diff
			);
		}
		Ok(())
	}

	macro_rules! generate_all_rsx_tests {
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

	generate_all_rsx_tests!(
		check_rsx_partial_params,
		check_rsx_accuracy,
		check_rsx_default_candles,
		check_rsx_zero_period,
		check_rsx_period_exceeds_length,
		check_rsx_very_small_dataset,
		check_rsx_reinput,
		check_rsx_nan_handling,
		check_rsx_streaming
	);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let output = RsxBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;

		let def = RsxParams::default();
		let row = output.values_for(&def).expect("default row missing");

		assert_eq!(row.len(), c.close.len());

		let expected = [
			46.11486311289701,
			46.88048640321688,
			47.174443049619995,
			47.48751360654475,
			46.552886446171684,
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
}
