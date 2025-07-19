//! # Kurtosis
//!
//! Kurtosis is a measure of the "tailedness" of a distribution, computed over a sliding window.  
//! This indicator returns the excess kurtosis, using the uncorrected moment-based formula.
//!
//! ## Parameters
//! - **period**: Window size (number of data points, default: 5)
//!
//! ## Errors
//! - **AllValuesNaN**: kurtosis: All input data values are `NaN`.
//! - **InvalidPeriod**: kurtosis: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: kurtosis: Not enough valid data points for the requested `period`.
//!
//! ## Returns
//! - **`Ok(KurtosisOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(KurtosisError)`** otherwise.

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

impl<'a> AsRef<[f64]> for KurtosisInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			KurtosisData::Slice(slice) => slice,
			KurtosisData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

#[derive(Debug, Clone)]
pub enum KurtosisData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct KurtosisOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct KurtosisParams {
	pub period: Option<usize>,
}

impl Default for KurtosisParams {
	fn default() -> Self {
		Self { period: Some(5) }
	}
}

#[derive(Debug, Clone)]
pub struct KurtosisInput<'a> {
	pub data: KurtosisData<'a>,
	pub params: KurtosisParams,
}

impl<'a> KurtosisInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: KurtosisParams) -> Self {
		Self {
			data: KurtosisData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slice(sl: &'a [f64], p: KurtosisParams) -> Self {
		Self {
			data: KurtosisData::Slice(sl),
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "hl2", KurtosisParams::default())
	}
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(5)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct KurtosisBuilder {
	period: Option<usize>,
	kernel: Kernel,
}

impl Default for KurtosisBuilder {
	fn default() -> Self {
		Self {
			period: None,
			kernel: Kernel::Auto,
		}
	}
}

impl KurtosisBuilder {
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
	pub fn apply(self, c: &Candles) -> Result<KurtosisOutput, KurtosisError> {
		let p = KurtosisParams { period: self.period };
		let i = KurtosisInput::from_candles(c, "hl2", p);
		kurtosis_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slice(self, d: &[f64]) -> Result<KurtosisOutput, KurtosisError> {
		let p = KurtosisParams { period: self.period };
		let i = KurtosisInput::from_slice(d, p);
		kurtosis_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<KurtosisStream, KurtosisError> {
		let p = KurtosisParams { period: self.period };
		KurtosisStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum KurtosisError {
	#[error("kurtosis: All values are NaN.")]
	AllValuesNaN,
	#[error("kurtosis: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("kurtosis: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("kurtosis: Invalid period (zero or missing).")]
	ZeroOrMissingPeriod,
}

#[inline]
pub fn kurtosis(input: &KurtosisInput) -> Result<KurtosisOutput, KurtosisError> {
	kurtosis_with_kernel(input, Kernel::Auto)
}

pub fn kurtosis_with_kernel(input: &KurtosisInput, kernel: Kernel) -> Result<KurtosisOutput, KurtosisError> {
	let data: &[f64] = input.as_ref();

	let first = data
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(KurtosisError::AllValuesNaN)?;

	let len = data.len();
	let period = input.get_period();

	if period == 0 || period > len {
		return Err(KurtosisError::InvalidPeriod { period, data_len: len });
	}
	if (len - first) < period {
		return Err(KurtosisError::NotEnoughValidData {
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
			Kernel::Scalar | Kernel::ScalarBatch => kurtosis_scalar(data, period, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => kurtosis_avx2(data, period, first, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => kurtosis_avx512(data, period, first, &mut out),
			_ => unreachable!(),
		}
	}
	Ok(KurtosisOutput { values: out })
}

#[inline]
pub fn kurtosis_scalar(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
	for i in (first + period - 1)..data.len() {
		let start = i + 1 - period;
		let window = &data[start..start + period];

		if window.iter().any(|x| x.is_nan()) {
			out[i] = f64::NAN;
			continue;
		}
		let n = period as f64;
		let mean = window.iter().sum::<f64>() / n;
		let mut m2 = 0.0;
		let mut m4 = 0.0;
		for &val in window {
			let diff = val - mean;
			m2 += diff * diff;
			m4 += diff.powi(4);
		}
		m2 /= n;
		m4 /= n;

		if m2.abs() < f64::EPSILON {
			out[i] = f64::NAN;
		} else {
			out[i] = (m4 / (m2 * m2)) - 3.0;
		}
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn kurtosis_avx512(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
	if period <= 32 {
		unsafe { kurtosis_avx512_short(data, period, first, out) }
	} else {
		unsafe { kurtosis_avx512_long(data, period, first, out) }
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn kurtosis_avx2(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
	unsafe { kurtosis_scalar(data, period, first, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn kurtosis_avx512_short(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
	kurtosis_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn kurtosis_avx512_long(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
	kurtosis_scalar(data, period, first, out)
}

#[inline(always)]
pub fn kurtosis_batch_with_kernel(
	data: &[f64],
	sweep: &KurtosisBatchRange,
	k: Kernel,
) -> Result<KurtosisBatchOutput, KurtosisError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(KurtosisError::InvalidPeriod { period: 0, data_len: 0 }),
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	kurtosis_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct KurtosisBatchRange {
	pub period: (usize, usize, usize),
}

impl Default for KurtosisBatchRange {
	fn default() -> Self {
		Self { period: (5, 50, 1) }
	}
}

#[derive(Clone, Debug, Default)]
pub struct KurtosisBatchBuilder {
	range: KurtosisBatchRange,
	kernel: Kernel,
}

impl KurtosisBatchBuilder {
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
	pub fn apply_slice(self, data: &[f64]) -> Result<KurtosisBatchOutput, KurtosisError> {
		kurtosis_batch_with_kernel(data, &self.range, self.kernel)
	}
	pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<KurtosisBatchOutput, KurtosisError> {
		KurtosisBatchBuilder::new().kernel(k).apply_slice(data)
	}
	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<KurtosisBatchOutput, KurtosisError> {
		let slice = source_type(c, src);
		self.apply_slice(slice)
	}
	pub fn with_default_candles(c: &Candles) -> Result<KurtosisBatchOutput, KurtosisError> {
		KurtosisBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "hl2")
	}
}

#[derive(Clone, Debug)]
pub struct KurtosisBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<KurtosisParams>,
	pub rows: usize,
	pub cols: usize,
}

impl KurtosisBatchOutput {
	pub fn row_for_params(&self, p: &KurtosisParams) -> Option<usize> {
		self.combos
			.iter()
			.position(|c| c.period.unwrap_or(5) == p.period.unwrap_or(5))
	}
	pub fn values_for(&self, p: &KurtosisParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &KurtosisBatchRange) -> Vec<KurtosisParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let periods = axis_usize(r.period);
	let mut out = Vec::with_capacity(periods.len());
	for &p in &periods {
		out.push(KurtosisParams { period: Some(p) });
	}
	out
}

#[inline(always)]
pub fn kurtosis_batch_slice(
	data: &[f64],
	sweep: &KurtosisBatchRange,
	kern: Kernel,
) -> Result<KurtosisBatchOutput, KurtosisError> {
	kurtosis_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn kurtosis_batch_par_slice(
	data: &[f64],
	sweep: &KurtosisBatchRange,
	kern: Kernel,
) -> Result<KurtosisBatchOutput, KurtosisError> {
	kurtosis_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn kurtosis_batch_inner(
	data: &[f64],
	sweep: &KurtosisBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<KurtosisBatchOutput, KurtosisError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(KurtosisError::InvalidPeriod { period: 0, data_len: 0 });
	}

	let first = data
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(KurtosisError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if data.len() - first < max_p {
		return Err(KurtosisError::NotEnoughValidData {
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
			Kernel::Scalar => kurtosis_row_scalar(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => kurtosis_row_avx2(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => kurtosis_row_avx512(data, first, period, out_row),
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

	Ok(KurtosisBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
unsafe fn kurtosis_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	kurtosis_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn kurtosis_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	kurtosis_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn kurtosis_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	kurtosis_avx512(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn kurtosis_row_avx512_short(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	kurtosis_avx512_short(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn kurtosis_row_avx512_long(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
	kurtosis_avx512_long(data, period, first, out)
}

#[derive(Debug, Clone)]
pub struct KurtosisStream {
	period: usize,
	buffer: Vec<f64>,
	head: usize,
	filled: bool,
}

impl KurtosisStream {
	pub fn try_new(params: KurtosisParams) -> Result<Self, KurtosisError> {
		let period = params.period.unwrap_or(5);
		if period == 0 {
			return Err(KurtosisError::InvalidPeriod { period, data_len: 0 });
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
		Some(self.kurtosis_ring())
	}

	#[inline(always)]
	fn kurtosis_ring(&self) -> f64 {
		let n = self.period as f64;
		if self.buffer.iter().any(|x| x.is_nan()) {
			return f64::NAN;
		}
		let mean = self.buffer.iter().sum::<f64>() / n;
		let mut m2 = 0.0;
		let mut m4 = 0.0;
		for &val in &self.buffer {
			let diff = val - mean;
			m2 += diff * diff;
			m4 += diff.powi(4);
		}
		m2 /= n;
		m4 /= n;
		if m2.abs() < f64::EPSILON {
			f64::NAN
		} else {
			(m4 / (m2 * m2)) - 3.0
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_kurtosis_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let default_params = KurtosisParams { period: None };
		let input = KurtosisInput::from_candles(&candles, "close", default_params);
		let output = kurtosis_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());

		Ok(())
	}

	fn check_kurtosis_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let input = KurtosisInput::from_candles(&candles, "hl2", KurtosisParams::default());
		let result = kurtosis_with_kernel(&input, kernel)?;
		let expected_last_five = [
			-0.5438903789933454,
			-1.6848139264816433,
			-1.6331336745945797,
			-0.6130805596586351,
			-0.027802601135927585,
		];
		let start = result.values.len().saturating_sub(5);
		for (i, &val) in result.values[start..].iter().enumerate() {
			let diff = (val - expected_last_five[i]).abs();
			assert!(
				diff < 1e-6,
				"[{}] KURTOSIS {:?} mismatch at idx {}: got {}, expected {}",
				test_name,
				kernel,
				i,
				val,
				expected_last_five[i]
			);
		}
		Ok(())
	}

	fn check_kurtosis_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let input = KurtosisInput::with_default_candles(&candles);
		match input.data {
			KurtosisData::Candles { source, .. } => assert_eq!(source, "hl2"),
			_ => panic!("Expected KurtosisData::Candles"),
		}
		let output = kurtosis_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());

		Ok(())
	}

	fn check_kurtosis_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let input_data = [10.0, 20.0, 30.0];
		let params = KurtosisParams { period: Some(0) };
		let input = KurtosisInput::from_slice(&input_data, params);
		let res = kurtosis_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] KURTOSIS should fail with zero period", test_name);
		Ok(())
	}

	fn check_kurtosis_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data_small = [10.0, 20.0, 30.0];
		let params = KurtosisParams { period: Some(10) };
		let input = KurtosisInput::from_slice(&data_small, params);
		let res = kurtosis_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] KURTOSIS should fail with period exceeding length",
			test_name
		);
		Ok(())
	}

	fn check_kurtosis_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let single_point = [42.0];
		let params = KurtosisParams { period: Some(5) };
		let input = KurtosisInput::from_slice(&single_point, params);
		let res = kurtosis_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] KURTOSIS should fail with insufficient data",
			test_name
		);
		Ok(())
	}

	fn check_kurtosis_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let first_params = KurtosisParams { period: Some(5) };
		let first_input = KurtosisInput::from_candles(&candles, "close", first_params);
		let first_result = kurtosis_with_kernel(&first_input, kernel)?;

		let second_params = KurtosisParams { period: Some(5) };
		let second_input = KurtosisInput::from_slice(&first_result.values, second_params);
		let second_result = kurtosis_with_kernel(&second_input, kernel)?;

		assert_eq!(second_result.values.len(), first_result.values.len());
		Ok(())
	}

	fn check_kurtosis_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let input = KurtosisInput::from_candles(&candles, "close", KurtosisParams { period: Some(5) });
		let res = kurtosis_with_kernel(&input, kernel)?;
		assert_eq!(res.values.len(), candles.close.len());
		if res.values.len() > 20 {
			for (i, &val) in res.values[20..].iter().enumerate() {
				assert!(
					!val.is_nan(),
					"[{}] Found unexpected NaN at out-index {}",
					test_name,
					20 + i
				);
			}
		}
		Ok(())
	}

	fn check_kurtosis_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let period = 5;

		let input = KurtosisInput::from_candles(&candles, "close", KurtosisParams { period: Some(period) });
		let batch_output = kurtosis_with_kernel(&input, kernel)?.values;

		let mut stream = KurtosisStream::try_new(KurtosisParams { period: Some(period) })?;
		let mut stream_values = Vec::with_capacity(candles.close.len());
		for &price in &candles.close {
			match stream.update(price) {
				Some(kurtosis_val) => stream_values.push(kurtosis_val),
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
				"[{}] KURTOSIS streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
				test_name,
				i,
				b,
				s,
				diff
			);
		}
		Ok(())
	}

	macro_rules! generate_all_kurtosis_tests {
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

	generate_all_kurtosis_tests!(
		check_kurtosis_partial_params,
		check_kurtosis_accuracy,
		check_kurtosis_default_candles,
		check_kurtosis_zero_period,
		check_kurtosis_period_exceeds_length,
		check_kurtosis_very_small_dataset,
		check_kurtosis_reinput,
		check_kurtosis_nan_handling,
		check_kurtosis_streaming
	);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let output = KurtosisBatchBuilder::new().kernel(kernel).apply_candles(&c, "hl2")?;

		let def = KurtosisParams::default();
		let row = output.values_for(&def).expect("default row missing");

		assert_eq!(row.len(), c.close.len());

		let expected = [
			-0.5438903789933454,
			-1.6848139264816433,
			-1.6331336745945797,
			-0.6130805596586351,
			-0.027802601135927585,
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
