//! # Momentum (MOM)
//!
//! MOM measures the amount that a security’s price has changed over a given time span.
//! It is calculated by subtracting the previous price (from the chosen period) from the
//! current price, i.e., `momentum[i] = data[i] - data[i - period]`.
//!
//! ## Parameters
//! - **period**: The lookback window size (number of data points). Defaults to 10.
//!
//! ## Errors
//! - **AllValuesNaN**: mom: All input data values are `NaN`.
//! - **InvalidPeriod**: mom: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: mom: Not enough valid data points for the requested `period`.
//!
//! ## Returns
//! - **`Ok(MomOutput)`** on success, containing a `Vec<f64>` matching the input length.
//! - **`Err(MomError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use thiserror::Error;

impl<'a> AsRef<[f64]> for MomInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			MomData::Slice(slice) => slice,
			MomData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

#[derive(Debug, Clone)]
pub enum MomData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct MomOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct MomParams {
	pub period: Option<usize>,
}

impl Default for MomParams {
	fn default() -> Self {
		Self { period: Some(10) }
	}
}

#[derive(Debug, Clone)]
pub struct MomInput<'a> {
	pub data: MomData<'a>,
	pub params: MomParams,
}

impl<'a> MomInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: MomParams) -> Self {
		Self {
			data: MomData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slice(sl: &'a [f64], p: MomParams) -> Self {
		Self {
			data: MomData::Slice(sl),
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", MomParams::default())
	}
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(10)
	}
}

#[derive(Clone, Debug)]
pub struct MomBuilder {
	period: Option<usize>,
	kernel: Kernel,
}

impl Default for MomBuilder {
	fn default() -> Self {
		Self {
			period: None,
			kernel: Kernel::Auto,
		}
	}
}

impl MomBuilder {
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
	pub fn apply(self, c: &Candles) -> Result<MomOutput, MomError> {
		let p = MomParams { period: self.period };
		let i = MomInput::from_candles(c, "close", p);
		mom_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slice(self, d: &[f64]) -> Result<MomOutput, MomError> {
		let p = MomParams { period: self.period };
		let i = MomInput::from_slice(d, p);
		mom_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<MomStream, MomError> {
		let p = MomParams { period: self.period };
		MomStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum MomError {
	#[error("mom: All values are NaN.")]
	AllValuesNaN,

	#[error("mom: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },

	#[error("mom: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn mom(input: &MomInput) -> Result<MomOutput, MomError> {
	mom_with_kernel(input, Kernel::Auto)
}

pub fn mom_with_kernel(input: &MomInput, kernel: Kernel) -> Result<MomOutput, MomError> {
	let data: &[f64] = match &input.data {
		MomData::Candles { candles, source } => source_type(candles, source),
		MomData::Slice(sl) => sl,
	};

	let first = data.iter().position(|x| !x.is_nan()).ok_or(MomError::AllValuesNaN)?;

	let len = data.len();
	let period = input.get_period();

	if period == 0 || period > len {
		return Err(MomError::InvalidPeriod { period, data_len: len });
	}
	if (len - first) < period {
		return Err(MomError::NotEnoughValidData {
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
			Kernel::Scalar | Kernel::ScalarBatch => mom_scalar(data, period, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => mom_avx2(data, period, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => mom_avx512(data, period, first, &mut out),
			_ => unreachable!(),
		}
	}

	Ok(MomOutput { values: out })
}

#[inline(always)]
pub fn mom_scalar(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	for i in (first_valid + period)..data.len() {
		out[i] = data[i] - data[i - period];
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn mom_avx512(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	unsafe {
		if period <= 32 {
			mom_avx512_short(data, period, first_valid, out);
		} else {
			mom_avx512_long(data, period, first_valid, out);
		}
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn mom_avx512_short(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	// For API parity; fallback to scalar logic
	mom_scalar(data, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn mom_avx512_long(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	// For API parity; fallback to scalar logic
	mom_scalar(data, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn mom_avx2(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
	// For API parity; fallback to scalar logic
	mom_scalar(data, period, first_valid, out)
}

#[derive(Debug, Clone)]
pub struct MomStream {
	period: usize,
	buffer: Vec<f64>,
	head: usize,
	filled: bool,
}

impl MomStream {
	pub fn try_new(params: MomParams) -> Result<Self, MomError> {
		let period = params.period.unwrap_or(10);
		if period == 0 {
			return Err(MomError::InvalidPeriod { period, data_len: 0 });
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
		let prev = self.buffer[self.head];
		self.buffer[self.head] = value;
		self.head = (self.head + 1) % self.period;

		if !self.filled && self.head == 0 {
			self.filled = true;
		}
		if !self.filled {
			return None;
		}
		Some(value - prev)
	}
}

#[derive(Clone, Debug)]
pub struct MomBatchRange {
	pub period: (usize, usize, usize),
}

impl Default for MomBatchRange {
	fn default() -> Self {
		Self { period: (10, 10, 0) }
	}
}

#[derive(Clone, Debug, Default)]
pub struct MomBatchBuilder {
	range: MomBatchRange,
	kernel: Kernel,
}

impl MomBatchBuilder {
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
	pub fn apply_slice(self, data: &[f64]) -> Result<MomBatchOutput, MomError> {
		mom_batch_with_kernel(data, &self.range, self.kernel)
	}
	pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<MomBatchOutput, MomError> {
		MomBatchBuilder::new().kernel(k).apply_slice(data)
	}
	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<MomBatchOutput, MomError> {
		let slice = source_type(c, src);
		self.apply_slice(slice)
	}
	pub fn with_default_candles(c: &Candles) -> Result<MomBatchOutput, MomError> {
		MomBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
	}
}

pub fn mom_batch_with_kernel(data: &[f64], sweep: &MomBatchRange, k: Kernel) -> Result<MomBatchOutput, MomError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(MomError::InvalidPeriod { period: 0, data_len: 0 }),
	};

	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	mom_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct MomBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<MomParams>,
	pub rows: usize,
	pub cols: usize,
}
impl MomBatchOutput {
	pub fn row_for_params(&self, p: &MomParams) -> Option<usize> {
		self.combos
			.iter()
			.position(|c| c.period.unwrap_or(10) == p.period.unwrap_or(10))
	}

	pub fn values_for(&self, p: &MomParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &MomBatchRange) -> Vec<MomParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}

	let periods = axis_usize(r.period);
	let mut out = Vec::with_capacity(periods.len());
	for &p in &periods {
		out.push(MomParams { period: Some(p) });
	}
	out
}

#[inline(always)]
pub fn mom_batch_slice(data: &[f64], sweep: &MomBatchRange, kern: Kernel) -> Result<MomBatchOutput, MomError> {
	mom_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn mom_batch_par_slice(data: &[f64], sweep: &MomBatchRange, kern: Kernel) -> Result<MomBatchOutput, MomError> {
	mom_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn mom_batch_inner(
	data: &[f64],
	sweep: &MomBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<MomBatchOutput, MomError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(MomError::InvalidPeriod { period: 0, data_len: 0 });
	}
	let first = data.iter().position(|x| !x.is_nan()).ok_or(MomError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if data.len() - first < max_p {
		return Err(MomError::NotEnoughValidData {
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
			Kernel::Scalar => mom_row_scalar(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => mom_row_avx2(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => mom_row_avx512(data, first, period, out_row),
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

	Ok(MomBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
unsafe fn mom_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	for i in (first + period)..data.len() {
		out[i] = data[i] - data[i - period];
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn mom_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	mom_row_scalar(data, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn mom_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	if period <= 32 {
		mom_row_avx512_short(data, first, period, out);
	} else {
		mom_row_avx512_long(data, first, period, out);
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn mom_row_avx512_short(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	mom_row_scalar(data, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn mom_row_avx512_long(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	mom_row_scalar(data, first, period, out)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_mom_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let default_params = MomParams { period: None };
		let input = MomInput::from_candles(&candles, "close", default_params);
		let output = mom_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());

		Ok(())
	}

	fn check_mom_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let input = MomInput::from_candles(&candles, "close", MomParams::default());
		let result = mom_with_kernel(&input, kernel)?;
		let expected_last_five = [-134.0, -331.0, -194.0, -294.0, -896.0];
		let start = result.values.len().saturating_sub(5);
		for (i, &val) in result.values[start..].iter().enumerate() {
			let diff = (val - expected_last_five[i]).abs();
			assert!(
				diff < 1e-1,
				"[{}] MOM {:?} mismatch at idx {}: got {}, expected {}",
				test_name,
				kernel,
				i,
				val,
				expected_last_five[i]
			);
		}
		Ok(())
	}

	fn check_mom_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let input = MomInput::with_default_candles(&candles);
		match input.data {
			MomData::Candles { source, .. } => assert_eq!(source, "close"),
			_ => panic!("Expected MomData::Candles"),
		}
		let output = mom_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());

		Ok(())
	}

	fn check_mom_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let input_data = [10.0, 20.0, 30.0];
		let params = MomParams { period: Some(0) };
		let input = MomInput::from_slice(&input_data, params);
		let res = mom_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] MOM should fail with zero period", test_name);
		Ok(())
	}

	fn check_mom_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data_small = [10.0, 20.0, 30.0];
		let params = MomParams { period: Some(10) };
		let input = MomInput::from_slice(&data_small, params);
		let res = mom_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] MOM should fail with period exceeding length",
			test_name
		);
		Ok(())
	}

	fn check_mom_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let single_point = [42.0];
		let params = MomParams { period: Some(9) };
		let input = MomInput::from_slice(&single_point, params);
		let res = mom_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] MOM should fail with insufficient data", test_name);
		Ok(())
	}

	fn check_mom_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let first_params = MomParams { period: Some(14) };
		let first_input = MomInput::from_candles(&candles, "close", first_params);
		let first_result = mom_with_kernel(&first_input, kernel)?;

		let second_params = MomParams { period: Some(14) };
		let second_input = MomInput::from_slice(&first_result.values, second_params);
		let second_result = mom_with_kernel(&second_input, kernel)?;

		assert_eq!(second_result.values.len(), first_result.values.len());
		for i in 28..second_result.values.len() {
			assert!(
				!second_result.values[i].is_nan(),
				"[{}] MOM Slice Reinput {:?} unexpected NaN at idx {}",
				test_name,
				kernel,
				i
			);
		}
		Ok(())
	}

	fn check_mom_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let input = MomInput::from_candles(&candles, "close", MomParams { period: Some(10) });
		let res = mom_with_kernel(&input, kernel)?;
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

	fn check_mom_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let period = 10;
		let input = MomInput::from_candles(&candles, "close", MomParams { period: Some(period) });
		let batch_output = mom_with_kernel(&input, kernel)?.values;

		let mut stream = MomStream::try_new(MomParams { period: Some(period) })?;
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
				"[{}] MOM streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
				test_name,
				i,
				b,
				s,
				diff
			);
		}
		Ok(())
	}

	macro_rules! generate_all_mom_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $( #[test]
                fn [<$test_fn _scalar_f64>]() {
                    let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar);
                })*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $( #[test]
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

	generate_all_mom_tests!(
		check_mom_partial_params,
		check_mom_accuracy,
		check_mom_default_candles,
		check_mom_zero_period,
		check_mom_period_exceeds_length,
		check_mom_very_small_dataset,
		check_mom_reinput,
		check_mom_nan_handling,
		check_mom_streaming
	);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let output = MomBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;

		let def = MomParams::default();
		let row = output.values_for(&def).expect("default row missing");

		assert_eq!(row.len(), c.close.len());

		let expected = [-134.0, -331.0, -194.0, -294.0, -896.0];
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
