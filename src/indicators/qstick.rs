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
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::error::Error;
use thiserror::Error;

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
	let diff: Vec<f64> = open.iter().zip(close.iter()).take(len).map(|(&o, &c)| c - o).collect();

	let first = diff
		.iter()
		.position(|&x| !x.is_nan())
		.ok_or(QstickError::AllValuesNaN)?;

	if (len - first) < period {
		return Err(QstickError::NotEnoughValidData {
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
			Kernel::Scalar | Kernel::ScalarBatch => qstick_scalar(&diff, period, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => qstick_avx2(&diff, period, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => qstick_avx512(&diff, period, first, &mut out),
			_ => unreachable!(),
		}
	}

	Ok(QstickOutput { values: out })
}

#[inline]
pub fn qstick_scalar(diff: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	let mut sum = 0.0;
	for &value in diff[first_valid..first_valid + period].iter() {
		sum += value;
	}
	let inv_period = 1.0 / (period as f64);
	out[first_valid + period - 1] = sum * inv_period;
	for i in (first_valid + period)..diff.len() {
		sum += diff[i] - diff[i - period];
		out[i] = sum * inv_period;
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn qstick_avx512(diff: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	qstick_scalar(diff, period, first_valid, out)
}

#[inline]
pub fn qstick_avx2(diff: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	qstick_scalar(diff, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn qstick_avx512_short(diff: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	qstick_avx512(diff, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn qstick_avx512_long(diff: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	qstick_avx512(diff, period, first_valid, out)
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
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let periods = axis_usize(r.period);
	let mut out = Vec::with_capacity(periods.len());
	for &p in &periods {
		out.push(QstickParams { period: Some(p) });
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
	let mut diffs: Vec<f64> = open.iter().zip(close.iter()).take(len).map(|(&o, &c)| c - o).collect();
	let first = diffs
		.iter()
		.position(|&x| !x.is_nan())
		.ok_or(QstickError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if len - first < max_p {
		return Err(QstickError::NotEnoughValidData {
			needed: max_p,
			valid: len - first,
		});
	}
	let rows = combos.len();
	let cols = len;
	let mut values = vec![f64::NAN; rows * cols];
	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let period = combos[row].period.unwrap();
		match kern {
			Kernel::Scalar => qstick_row_scalar(&diffs, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => qstick_row_avx2(&diffs, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => qstick_row_avx512(&diffs, first, period, out_row),
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
	Ok(QstickBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
unsafe fn qstick_row_scalar(diff: &[f64], first: usize, period: usize, out: &mut [f64]) {
	qstick_scalar(diff, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn qstick_row_avx2(diff: &[f64], first: usize, period: usize, out: &mut [f64]) {
	qstick_avx2(diff, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn qstick_row_avx512(diff: &[f64], first: usize, period: usize, out: &mut [f64]) {
	qstick_avx512(diff, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn qstick_row_avx512_short(diff: &[f64], first: usize, period: usize, out: &mut [f64]) {
	qstick_avx512_short(diff, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn qstick_row_avx512_long(diff: &[f64], first: usize, period: usize, out: &mut [f64]) {
	qstick_avx512_long(diff, period, first, out)
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
		check_qstick_nan_handling
	);
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
