//! # Elder's Force Index (EFI)
//!
//! The Elder's Force Index (EFI) measures the power behind a price move using both price change and volume.
//! EFI is typically calculated by taking the difference in price (current - previous) multiplied by volume,
//! and then applying an EMA to that result.
//!
//! ## Parameters
//! - **period**: Window size for the EMA (defaults to 13).
//!
//! ## Errors
//! - **AllValuesNaN**: efi: All input data values are `NaN`.
//! - **InvalidPeriod**: efi: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: efi: Not enough valid data points for the requested `period`.
//! - **EmptyData**: efi: Input data slice is empty or mismatched.
//!
//! ## Returns
//! - **`Ok(EfiOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(EfiError)`** otherwise.

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

impl<'a> AsRef<[f64]> for EfiInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			EfiData::Candles { candles, source } => source_type(candles, source),
			EfiData::Slice { price, .. } => price,
		}
	}
}

#[derive(Debug, Clone)]
pub enum EfiData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice { price: &'a [f64], volume: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct EfiOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct EfiParams {
	pub period: Option<usize>,
}

impl Default for EfiParams {
	fn default() -> Self {
		Self { period: Some(13) }
	}
}

#[derive(Debug, Clone)]
pub struct EfiInput<'a> {
	pub data: EfiData<'a>,
	pub params: EfiParams,
}

impl<'a> EfiInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: EfiParams) -> Self {
		Self {
			data: EfiData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slices(price: &'a [f64], volume: &'a [f64], p: EfiParams) -> Self {
		Self {
			data: EfiData::Slice { price, volume },
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", EfiParams::default())
	}
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(13)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct EfiBuilder {
	period: Option<usize>,
	kernel: Kernel,
}

impl Default for EfiBuilder {
	fn default() -> Self {
		Self {
			period: None,
			kernel: Kernel::Auto,
		}
	}
}

impl EfiBuilder {
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
	pub fn apply(self, c: &Candles) -> Result<EfiOutput, EfiError> {
		let p = EfiParams { period: self.period };
		let i = EfiInput::from_candles(c, "close", p);
		efi_with_kernel(&i, self.kernel)
	}

	#[inline(always)]
	pub fn apply_slices(self, price: &[f64], volume: &[f64]) -> Result<EfiOutput, EfiError> {
		let p = EfiParams { period: self.period };
		let i = EfiInput::from_slices(price, volume, p);
		efi_with_kernel(&i, self.kernel)
	}

	#[inline(always)]
	pub fn into_stream(self) -> Result<EfiStream, EfiError> {
		let p = EfiParams { period: self.period };
		EfiStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum EfiError {
	#[error("efi: Empty data provided.")]
	EmptyData,
	#[error("efi: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("efi: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("efi: All values are NaN.")]
	AllValuesNaN,
}

#[inline]
pub fn efi(input: &EfiInput) -> Result<EfiOutput, EfiError> {
	efi_with_kernel(input, Kernel::Auto)
}

pub fn efi_with_kernel(input: &EfiInput, kernel: Kernel) -> Result<EfiOutput, EfiError> {
	let (price, volume): (&[f64], &[f64]) = match &input.data {
		EfiData::Candles { candles, source } => {
			let p = source_type(candles, source);
			let v = &candles.volume;
			(p, v)
		}
		EfiData::Slice { price, volume } => (price, volume),
	};

	if price.is_empty() || volume.is_empty() || price.len() != volume.len() {
		return Err(EfiError::EmptyData);
	}

	let len = price.len();
	let period = input.get_period();

	if period == 0 || period > len {
		return Err(EfiError::InvalidPeriod { period, data_len: len });
	}

	let first_valid_idx = price
		.iter()
		.zip(volume.iter())
		.position(|(p, v)| !p.is_nan() && !v.is_nan());
	if first_valid_idx.is_none() {
		return Err(EfiError::AllValuesNaN);
	}
	let first_valid_idx = first_valid_idx.unwrap();

	if (len - first_valid_idx) < 2 {
		return Err(EfiError::NotEnoughValidData {
			needed: 2,
			valid: len - first_valid_idx,
		});
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	let mut out = vec![f64::NAN; len];
	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => efi_scalar(price, volume, period, first_valid_idx, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => efi_avx2(price, volume, period, first_valid_idx, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => efi_avx512(price, volume, period, first_valid_idx, &mut out),
			_ => unreachable!(),
		}
	}
	Ok(EfiOutput { values: out })
}

#[inline]
pub fn efi_scalar(price: &[f64], volume: &[f64], period: usize, first_valid_idx: usize, out: &mut [f64]) {
	let len = price.len();
	let alpha = 2.0 / (period as f64 + 1.0);
	let mut valid_dif_idx = None;
	for i in (first_valid_idx + 1)..len {
		if !price[i].is_nan() && !price[i - 1].is_nan() && !volume[i].is_nan() {
			out[i] = (price[i] - price[i - 1]) * volume[i];
			valid_dif_idx = Some(i);
			break;
		}
	}
	let start_idx = match valid_dif_idx {
		Some(idx) => idx,
		None => return,
	};
	for i in (start_idx + 1)..len {
		let prev_ema = out[i - 1];
		if price[i].is_nan() || price[i - 1].is_nan() || volume[i].is_nan() {
			out[i] = prev_ema;
		} else {
			let current_dif = (price[i] - price[i - 1]) * volume[i];
			out[i] = alpha * current_dif + (1.0 - alpha) * prev_ema;
		}
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn efi_avx2(price: &[f64], volume: &[f64], period: usize, first_valid_idx: usize, out: &mut [f64]) {
	efi_scalar(price, volume, period, first_valid_idx, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn efi_avx512(price: &[f64], volume: &[f64], period: usize, first_valid_idx: usize, out: &mut [f64]) {
	if period <= 32 {
		unsafe { efi_avx512_short(price, volume, period, first_valid_idx, out) }
	} else {
		unsafe { efi_avx512_long(price, volume, period, first_valid_idx, out) }
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn efi_avx512_short(price: &[f64], volume: &[f64], period: usize, first_valid_idx: usize, out: &mut [f64]) {
	efi_scalar(price, volume, period, first_valid_idx, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn efi_avx512_long(price: &[f64], volume: &[f64], period: usize, first_valid_idx: usize, out: &mut [f64]) {
	efi_scalar(price, volume, period, first_valid_idx, out)
}

#[derive(Debug, Clone)]
pub struct EfiStream {
	period: usize,
	alpha: f64,
	prev: f64,
	filled: bool,
	last_price: f64,
	has_last: bool,
}

impl EfiStream {
	pub fn try_new(params: EfiParams) -> Result<Self, EfiError> {
		let period = params.period.unwrap_or(13);
		if period == 0 {
			return Err(EfiError::InvalidPeriod { period, data_len: 0 });
		}
		Ok(Self {
			period,
			alpha: 2.0 / (period as f64 + 1.0),
			prev: f64::NAN,
			filled: false,
			last_price: f64::NAN,
			has_last: false,
		})
	}

	#[inline(always)]
	pub fn update(&mut self, price: f64, volume: f64) -> Option<f64> {
		if !self.has_last {
			self.last_price = price;
			self.has_last = true;
			return None;
		}

		let out = if price.is_nan() || self.last_price.is_nan() || volume.is_nan() {
			if self.filled {
				self.prev
			} else {
				f64::NAN
			}
		} else {
			let diff = (price - self.last_price) * volume;
			if !self.filled {
				self.prev = diff;
				self.filled = true;
				diff
			} else {
				let ema = self.alpha * diff + (1.0 - self.alpha) * self.prev;
				self.prev = ema;
				ema
			}
		};
		self.last_price = price;
		Some(out)
	}
}

#[derive(Clone, Debug)]
pub struct EfiBatchRange {
	pub period: (usize, usize, usize),
}

impl Default for EfiBatchRange {
	fn default() -> Self {
		Self { period: (13, 100, 1) }
	}
}

#[derive(Clone, Debug, Default)]
pub struct EfiBatchBuilder {
	range: EfiBatchRange,
	kernel: Kernel,
}

impl EfiBatchBuilder {
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

	pub fn apply_slices(self, price: &[f64], volume: &[f64]) -> Result<EfiBatchOutput, EfiError> {
		efi_batch_with_kernel(price, volume, &self.range, self.kernel)
	}

	pub fn with_default_slices(price: &[f64], volume: &[f64], k: Kernel) -> Result<EfiBatchOutput, EfiError> {
		EfiBatchBuilder::new().kernel(k).apply_slices(price, volume)
	}

	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<EfiBatchOutput, EfiError> {
		let slice = source_type(c, src);
		let volume = &c.volume;
		self.apply_slices(slice, volume)
	}

	pub fn with_default_candles(c: &Candles) -> Result<EfiBatchOutput, EfiError> {
		EfiBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
	}
}

pub fn efi_batch_with_kernel(
	price: &[f64],
	volume: &[f64],
	sweep: &EfiBatchRange,
	k: Kernel,
) -> Result<EfiBatchOutput, EfiError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(EfiError::InvalidPeriod { period: 0, data_len: 0 }),
	};

	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	efi_batch_par_slice(price, volume, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct EfiBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<EfiParams>,
	pub rows: usize,
	pub cols: usize,
}
impl EfiBatchOutput {
	pub fn row_for_params(&self, p: &EfiParams) -> Option<usize> {
		self.combos
			.iter()
			.position(|c| c.period.unwrap_or(13) == p.period.unwrap_or(13))
	}
	pub fn values_for(&self, p: &EfiParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &EfiBatchRange) -> Vec<EfiParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let periods = axis_usize(r.period);
	let mut out = Vec::with_capacity(periods.len());
	for &p in &periods {
		out.push(EfiParams { period: Some(p) });
	}
	out
}

#[inline(always)]
pub fn efi_batch_slice(
	price: &[f64],
	volume: &[f64],
	sweep: &EfiBatchRange,
	kern: Kernel,
) -> Result<EfiBatchOutput, EfiError> {
	efi_batch_inner(price, volume, sweep, kern, false)
}

#[inline(always)]
pub fn efi_batch_par_slice(
	price: &[f64],
	volume: &[f64],
	sweep: &EfiBatchRange,
	kern: Kernel,
) -> Result<EfiBatchOutput, EfiError> {
	efi_batch_inner(price, volume, sweep, kern, true)
}

#[inline(always)]
fn efi_batch_inner(
	price: &[f64],
	volume: &[f64],
	sweep: &EfiBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<EfiBatchOutput, EfiError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(EfiError::InvalidPeriod { period: 0, data_len: 0 });
	}
	let first = price
		.iter()
		.zip(volume.iter())
		.position(|(p, v)| !p.is_nan() && !v.is_nan())
		.ok_or(EfiError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if price.len() - first < max_p {
		return Err(EfiError::NotEnoughValidData {
			needed: max_p,
			valid: price.len() - first,
		});
	}
	let rows = combos.len();
	let cols = price.len();
	let mut values = vec![f64::NAN; rows * cols];

	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let period = combos[row].period.unwrap();
		match kern {
			Kernel::Scalar => efi_row_scalar(price, volume, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => efi_row_avx2(price, volume, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => efi_row_avx512(price, volume, first, period, out_row),
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

	Ok(EfiBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
unsafe fn efi_row_scalar(price: &[f64], volume: &[f64], first: usize, period: usize, out: &mut [f64]) {
	efi_scalar(price, volume, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn efi_row_avx2(price: &[f64], volume: &[f64], first: usize, period: usize, out: &mut [f64]) {
	efi_scalar(price, volume, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn efi_row_avx512(price: &[f64], volume: &[f64], first: usize, period: usize, out: &mut [f64]) {
	if period <= 32 {
		efi_row_avx512_short(price, volume, first, period, out);
	} else {
		efi_row_avx512_long(price, volume, first, period, out);
	}
	_mm_sfence();
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn efi_row_avx512_short(price: &[f64], volume: &[f64], first: usize, period: usize, out: &mut [f64]) {
	efi_scalar(price, volume, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn efi_row_avx512_long(price: &[f64], volume: &[f64], first: usize, period: usize, out: &mut [f64]) {
	efi_scalar(price, volume, period, first, out);
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;
	use crate::utilities::enums::Kernel;

	fn check_efi_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let default_params = EfiParams { period: None };
		let input = EfiInput::from_candles(&candles, "close", default_params);
		let output = efi_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_efi_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = EfiInput::from_candles(&candles, "close", EfiParams::default());
		let result = efi_with_kernel(&input, kernel)?;
		let expected_last_five = [
			-44604.382026531224,
			-39811.02321812391,
			-36599.9671820205,
			-29903.28014503471,
			-55406.09054645832,
		];
		let start = result.values.len().saturating_sub(5);
		for (i, &val) in result.values[start..].iter().enumerate() {
			let diff = (val - expected_last_five[i]).abs();
			assert!(
				diff < 1.0,
				"[{}] EFI {:?} mismatch at idx {}: got {}, expected {}",
				test_name,
				kernel,
				i,
				val,
				expected_last_five[i]
			);
		}
		Ok(())
	}

	fn check_efi_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let price = [10.0, 20.0, 30.0];
		let volume = [100.0, 200.0, 300.0];
		let params = EfiParams { period: Some(0) };
		let input = EfiInput::from_slices(&price, &volume, params);
		let res = efi_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] EFI should fail with zero period", test_name);
		Ok(())
	}

	fn check_efi_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let price = [10.0, 20.0, 30.0];
		let volume = [100.0, 200.0, 300.0];
		let params = EfiParams { period: Some(10) };
		let input = EfiInput::from_slices(&price, &volume, params);
		let res = efi_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] EFI should fail with period exceeding length",
			test_name
		);
		Ok(())
	}

	fn check_efi_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = EfiInput::from_candles(&candles, "close", EfiParams { period: Some(13) });
		let res = efi_with_kernel(&input, kernel)?;
		assert_eq!(res.values.len(), candles.close.len());
		Ok(())
	}

	fn check_efi_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let period = 13;
		let input = EfiInput::from_candles(&candles, "close", EfiParams { period: Some(period) });
		let batch_output = efi_with_kernel(&input, kernel)?.values;
		let mut stream = EfiStream::try_new(EfiParams { period: Some(period) })?;
		let mut stream_values = Vec::with_capacity(candles.close.len());
		for (&p, &v) in candles.close.iter().zip(&candles.volume) {
			match stream.update(p, v) {
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
				diff < 1.0,
				"[{}] EFI streaming mismatch at idx {}: batch={}, stream={}",
				test_name,
				i,
				b,
				s
			);
		}
		Ok(())
	}

	macro_rules! generate_all_efi_tests {
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

	generate_all_efi_tests!(
		check_efi_partial_params,
		check_efi_accuracy,
		check_efi_zero_period,
		check_efi_period_exceeds_length,
		check_efi_nan_handling,
		check_efi_streaming
	);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let output = EfiBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;
		let def = EfiParams::default();
		let row = output.values_for(&def).expect("default row missing");
		assert_eq!(row.len(), c.close.len());
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
