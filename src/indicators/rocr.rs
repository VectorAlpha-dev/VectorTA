//! # Rate of Change Ratio (ROCR)
//!
//! The Rate of Change Ratio (ROCR) measures the ratio between the current price
//! and the price `period` bars ago. Centered around 1.0; >1 means increase, <1 decrease.
//!
//! ## Parameters
//! - **period**: Lookback window (number of data points), default 9.
//!
//! ## Errors
//! - **EmptyData**: rocr: Input data slice is empty.
//! - **InvalidPeriod**: rocr: `period` is zero or exceeds data length.
//! - **NotEnoughValidData**: rocr: Fewer than `period` valid (non-NaN) data points after first valid index.
//! - **AllValuesNaN**: rocr: All input data values are `NaN`.
//!
//! ## Returns
//! - **`Ok(RocrOutput)`** on success, containing a `Vec<f64>` matching input length.
//! - **`Err(RocrError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

impl<'a> AsRef<[f64]> for RocrInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			RocrData::Slice(slice) => slice,
			RocrData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

#[derive(Debug, Clone)]
pub enum RocrData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct RocrOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct RocrParams {
	pub period: Option<usize>,
}

impl Default for RocrParams {
	fn default() -> Self {
		Self { period: Some(10) }
	}
}

#[derive(Debug, Clone)]
pub struct RocrInput<'a> {
	pub data: RocrData<'a>,
	pub params: RocrParams,
}

impl<'a> RocrInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: RocrParams) -> Self {
		Self {
			data: RocrData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slice(sl: &'a [f64], p: RocrParams) -> Self {
		Self {
			data: RocrData::Slice(sl),
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", RocrParams::default())
	}
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(9)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct RocrBuilder {
	period: Option<usize>,
	kernel: Kernel,
}

impl Default for RocrBuilder {
	fn default() -> Self {
		Self {
			period: None,
			kernel: Kernel::Auto,
		}
	}
}

impl RocrBuilder {
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
	pub fn apply(self, c: &Candles) -> Result<RocrOutput, RocrError> {
		let p = RocrParams { period: self.period };
		let i = RocrInput::from_candles(c, "close", p);
		rocr_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slice(self, d: &[f64]) -> Result<RocrOutput, RocrError> {
		let p = RocrParams { period: self.period };
		let i = RocrInput::from_slice(d, p);
		rocr_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<RocrStream, RocrError> {
		let p = RocrParams { period: self.period };
		RocrStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum RocrError {
	#[error("rocr: Empty data provided.")]
	EmptyData,
	#[error("rocr: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("rocr: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("rocr: All values are NaN.")]
	AllValuesNaN,
}

#[inline]
pub fn rocr(input: &RocrInput) -> Result<RocrOutput, RocrError> {
	rocr_with_kernel(input, Kernel::Auto)
}

pub fn rocr_with_kernel(input: &RocrInput, kernel: Kernel) -> Result<RocrOutput, RocrError> {
	let data: &[f64] = match &input.data {
		RocrData::Candles { candles, source } => source_type(candles, source),
		RocrData::Slice(sl) => sl,
	};

	if data.is_empty() {
		return Err(RocrError::EmptyData);
	}

	let first = data.iter().position(|x| !x.is_nan()).ok_or(RocrError::AllValuesNaN)?;

	let len = data.len();
	let period = input.get_period();

	if period == 0 || period > len {
		return Err(RocrError::InvalidPeriod { period, data_len: len });
	}
	if (len - first) < period {
		return Err(RocrError::NotEnoughValidData {
			needed: period,
			valid: len - first,
		});
	}

	let mut out = vec![f64::NAN; len];

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => rocr_scalar(data, period, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => rocr_avx2(data, period, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => rocr_avx512(data, period, first, &mut out),
			_ => unreachable!(),
		}
	}

	Ok(RocrOutput { values: out })
}

#[inline]
pub fn rocr_scalar(data: &[f64], period: usize, first_val: usize, out: &mut [f64]) {
	for i in (first_val + period)..data.len() {
		let current = data[i];
		let past = data[i - period];
		out[i] = if past == 0.0 || past.is_nan() {
			0.0
		} else {
			current / past
		};
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn rocr_avx512(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	rocr_scalar(data, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn rocr_avx2(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	rocr_scalar(data, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn rocr_avx512_short(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	rocr_scalar(data, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn rocr_avx512_long(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	rocr_scalar(data, period, first_valid, out)
}

pub struct RocrStream {
	period: usize,
	buffer: Vec<f64>,
	head: usize,
	filled: bool,
}

impl RocrStream {
	pub fn try_new(params: RocrParams) -> Result<Self, RocrError> {
		let period = params.period.unwrap_or(9);
		if period == 0 {
			return Err(RocrError::InvalidPeriod { period, data_len: 0 });
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
		let out = if self.filled {
			let prev = self.buffer[self.head];
			if prev == 0.0 || prev.is_nan() {
				0.0
			} else {
				value / prev
			}
		} else {
			f64::NAN
		};
		self.buffer[self.head] = value;
		self.head = (self.head + 1) % self.period;
		if !self.filled && self.head == 0 {
			self.filled = true;
		}
		Some(out)
	}
}

#[derive(Clone, Debug)]
pub struct RocrBatchRange {
	pub period: (usize, usize, usize),
}

impl Default for RocrBatchRange {
	fn default() -> Self {
		Self { period: (9, 240, 1) }
	}
}

#[derive(Clone, Debug, Default)]
pub struct RocrBatchBuilder {
	range: RocrBatchRange,
	kernel: Kernel,
}

impl RocrBatchBuilder {
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

	pub fn apply_slice(self, data: &[f64]) -> Result<RocrBatchOutput, RocrError> {
		rocr_batch_with_kernel(data, &self.range, self.kernel)
	}

	pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<RocrBatchOutput, RocrError> {
		RocrBatchBuilder::new().kernel(k).apply_slice(data)
	}

	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<RocrBatchOutput, RocrError> {
		let slice = source_type(c, src);
		self.apply_slice(slice)
	}

	pub fn with_default_candles(c: &Candles) -> Result<RocrBatchOutput, RocrError> {
		RocrBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
	}
}

pub fn rocr_batch_with_kernel(data: &[f64], sweep: &RocrBatchRange, k: Kernel) -> Result<RocrBatchOutput, RocrError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(RocrError::InvalidPeriod { period: 0, data_len: 0 }),
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	rocr_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct RocrBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<RocrParams>,
	pub rows: usize,
	pub cols: usize,
}
impl RocrBatchOutput {
	pub fn row_for_params(&self, p: &RocrParams) -> Option<usize> {
		self.combos
			.iter()
			.position(|c| c.period.unwrap_or(9) == p.period.unwrap_or(9))
	}

	pub fn values_for(&self, p: &RocrParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &RocrBatchRange) -> Vec<RocrParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let periods = axis_usize(r.period);

	let mut out = Vec::with_capacity(periods.len());
	for &p in &periods {
		out.push(RocrParams { period: Some(p) });
	}
	out
}

#[inline(always)]
pub fn rocr_batch_slice(data: &[f64], sweep: &RocrBatchRange, kern: Kernel) -> Result<RocrBatchOutput, RocrError> {
	rocr_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn rocr_batch_par_slice(data: &[f64], sweep: &RocrBatchRange, kern: Kernel) -> Result<RocrBatchOutput, RocrError> {
	rocr_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn rocr_batch_inner(
	data: &[f64],
	sweep: &RocrBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<RocrBatchOutput, RocrError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(RocrError::InvalidPeriod { period: 0, data_len: 0 });
	}

	let first = data.iter().position(|x| !x.is_nan()).ok_or(RocrError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if data.len() - first < max_p {
		return Err(RocrError::NotEnoughValidData {
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
			Kernel::Scalar => rocr_row_scalar(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => rocr_row_avx2(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => rocr_row_avx512(data, first, period, out_row),
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

	Ok(RocrBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
unsafe fn rocr_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	for i in (first + period)..data.len() {
		let current = data[i];
		let past = data[i - period];
		out[i] = if past == 0.0 || past.is_nan() {
			0.0
		} else {
			current / past
		};
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn rocr_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	rocr_row_scalar(data, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn rocr_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	rocr_row_scalar(data, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn rocr_row_avx512_short(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	rocr_row_scalar(data, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn rocr_row_avx512_long(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	rocr_row_scalar(data, first, period, out)
}

#[inline(always)]
pub fn expand_grid_rocr(r: &RocrBatchRange) -> Vec<RocrParams> {
	expand_grid(r)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_rocr_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let default_params = RocrParams { period: None };
		let input = RocrInput::from_candles(&candles, "close", default_params);
		let output = rocr_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());

		Ok(())
	}

	fn check_rocr_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let input = RocrInput::from_candles(&candles, "close", RocrParams { period: Some(10) });
		let result = rocr_with_kernel(&input, kernel)?;
		let expected_last_five = [
			0.9977448290950706,
			0.9944380965183492,
			0.9967247986764135,
			0.9950545846019277,
			0.984954072979463,
		];
		let start = result.values.len().saturating_sub(5);
		for (i, &val) in result.values[start..].iter().enumerate() {
			let diff = (val - expected_last_five[i]).abs();
			assert!(
				diff < 1e-8,
				"[{}] ROCR {:?} mismatch at idx {}: got {}, expected {}",
				test_name,
				kernel,
				i,
				val,
				expected_last_five[i]
			);
		}
		Ok(())
	}

	fn check_rocr_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let input = RocrInput::with_default_candles(&candles);
		match input.data {
			RocrData::Candles { source, .. } => assert_eq!(source, "close"),
			_ => panic!("Expected RocrData::Candles"),
		}
		let output = rocr_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());

		Ok(())
	}

	fn check_rocr_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let input_data = [10.0, 20.0, 30.0];
		let params = RocrParams { period: Some(0) };
		let input = RocrInput::from_slice(&input_data, params);
		let res = rocr_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] ROCR should fail with zero period", test_name);
		Ok(())
	}

	fn check_rocr_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data_small = [10.0, 20.0, 30.0];
		let params = RocrParams { period: Some(10) };
		let input = RocrInput::from_slice(&data_small, params);
		let res = rocr_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] ROCR should fail with period exceeding length",
			test_name
		);
		Ok(())
	}

	fn check_rocr_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let single_point = [42.0];
		let params = RocrParams { period: Some(9) };
		let input = RocrInput::from_slice(&single_point, params);
		let res = rocr_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] ROCR should fail with insufficient data", test_name);
		Ok(())
	}

	fn check_rocr_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let first_params = RocrParams { period: Some(14) };
		let first_input = RocrInput::from_candles(&candles, "close", first_params);
		let first_result = rocr_with_kernel(&first_input, kernel)?;

		let second_params = RocrParams { period: Some(14) };
		let second_input = RocrInput::from_slice(&first_result.values, second_params);
		let second_result = rocr_with_kernel(&second_input, kernel)?;

		assert_eq!(second_result.values.len(), first_result.values.len());
		for i in 28..second_result.values.len() {
			assert!(
				!second_result.values[i].is_nan(),
				"[{}] ROCR Slice Reinput {:?} unexpected NaN at idx {}",
				test_name,
				kernel,
				i,
			);
		}
		Ok(())
	}

	fn check_rocr_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let input = RocrInput::from_candles(&candles, "close", RocrParams { period: Some(9) });
		let res = rocr_with_kernel(&input, kernel)?;
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

	fn check_rocr_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let period = 9;

		let input = RocrInput::from_candles(&candles, "close", RocrParams { period: Some(period) });
		let batch_output = rocr_with_kernel(&input, kernel)?.values;

		let mut stream = RocrStream::try_new(RocrParams { period: Some(period) })?;

		let mut stream_values = Vec::with_capacity(candles.close.len());
		for &price in &candles.close {
			match stream.update(price) {
				Some(rocr_val) => stream_values.push(rocr_val),
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
				"[{}] ROCR streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
				test_name,
				i,
				b,
				s,
				diff
			);
		}
		Ok(())
	}

	macro_rules! generate_all_rocr_tests {
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

	generate_all_rocr_tests!(
		check_rocr_partial_params,
		check_rocr_accuracy,
		check_rocr_default_candles,
		check_rocr_zero_period,
		check_rocr_period_exceeds_length,
		check_rocr_very_small_dataset,
		check_rocr_reinput,
		check_rocr_nan_handling,
		check_rocr_streaming
	);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let output = RocrBatchBuilder::new()
			.period_static(10)
			.kernel(kernel)
			.apply_candles(&c, "close")?;

		let def = RocrParams::default();
		let row = output.values_for(&def).expect("default row missing");

		assert_eq!(row.len(), c.close.len());

		let expected = [
			0.9977448290950706,
			0.9944380965183492,
			0.9967247986764135,
			0.9950545846019277,
			0.984954072979463,
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
					let _ = $fn_name(stringify!([<$fn_name _auto_detect>]),
									 Kernel::Auto);
				}
			}
		};
	}
	gen_batch_tests!(check_batch_default_row);
}
