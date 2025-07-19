//! # True Strength Index (TSI)
//!
//! A momentum oscillator with double EMA smoothing on momentum and its absolute value.
//! Oscillates between positive/negative, indicating trend strength/direction.
//!
//! ## Parameters
//! - **long_period**: Default = 25
//! - **short_period**: Default = 13
//!
//! ## Errors
//! - **AllValuesNaN**: tsi: All input data values are `NaN`.
//! - **InvalidPeriod**: tsi: One or both periods are zero or exceed data length.
//! - **NotEnoughValidData**: tsi: Not enough valid data after first valid index.
//!
//! ## Returns
//! - **`Ok(TsiOutput)`** on success with `Vec<f64>` matching input.
//! - **`Err(TsiError)`** otherwise.
use crate::indicators::ema::{ema, EmaError, EmaInput, EmaOutput, EmaParams};
use crate::indicators::mom::{mom, MomError, MomInput, MomOutput, MomParams};
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

impl<'a> AsRef<[f64]> for TsiInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			TsiData::Slice(slice) => slice,
			TsiData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

#[derive(Debug, Clone)]
pub enum TsiData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct TsiOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct TsiParams {
	pub long_period: Option<usize>,
	pub short_period: Option<usize>,
}

impl Default for TsiParams {
	fn default() -> Self {
		Self {
			long_period: Some(25),
			short_period: Some(13),
		}
	}
}

#[derive(Debug, Clone)]
pub struct TsiInput<'a> {
	pub data: TsiData<'a>,
	pub params: TsiParams,
}

impl<'a> TsiInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: TsiParams) -> Self {
		Self {
			data: TsiData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slice(sl: &'a [f64], p: TsiParams) -> Self {
		Self {
			data: TsiData::Slice(sl),
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", TsiParams::default())
	}
	#[inline]
	pub fn get_long_period(&self) -> usize {
		self.params.long_period.unwrap_or(25)
	}
	#[inline]
	pub fn get_short_period(&self) -> usize {
		self.params.short_period.unwrap_or(13)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct TsiBuilder {
	long_period: Option<usize>,
	short_period: Option<usize>,
	kernel: Kernel,
}

impl Default for TsiBuilder {
	fn default() -> Self {
		Self {
			long_period: None,
			short_period: None,
			kernel: Kernel::Auto,
		}
	}
}

impl TsiBuilder {
	#[inline(always)]
	pub fn new() -> Self {
		Self::default()
	}
	#[inline(always)]
	pub fn long_period(mut self, n: usize) -> Self {
		self.long_period = Some(n);
		self
	}
	#[inline(always)]
	pub fn short_period(mut self, n: usize) -> Self {
		self.short_period = Some(n);
		self
	}
	#[inline(always)]
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}

	#[inline(always)]
	pub fn apply(self, c: &Candles) -> Result<TsiOutput, TsiError> {
		let p = TsiParams {
			long_period: self.long_period,
			short_period: self.short_period,
		};
		let i = TsiInput::from_candles(c, "close", p);
		tsi_with_kernel(&i, self.kernel)
	}

	#[inline(always)]
	pub fn apply_slice(self, d: &[f64]) -> Result<TsiOutput, TsiError> {
		let p = TsiParams {
			long_period: self.long_period,
			short_period: self.short_period,
		};
		let i = TsiInput::from_slice(d, p);
		tsi_with_kernel(&i, self.kernel)
	}

	#[inline(always)]
	pub fn into_stream(self) -> Result<TsiStream, TsiError> {
		let p = TsiParams {
			long_period: self.long_period,
			short_period: self.short_period,
		};
		TsiStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum TsiError {
	#[error("tsi: All values are NaN.")]
	AllValuesNaN,
	#[error("tsi: Invalid period: long = {long_period}, short = {short_period}, data length = {data_len}")]
	InvalidPeriod {
		long_period: usize,
		short_period: usize,
		data_len: usize,
	},
	#[error("tsi: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("tsi: EMA sub-error: {0}")]
	EmaSubError(#[from] EmaError),
	#[error("tsi: MOM sub-error: {0}")]
	MomSubError(#[from] MomError),
}

#[inline]
pub fn tsi(input: &TsiInput) -> Result<TsiOutput, TsiError> {
	tsi_with_kernel(input, Kernel::Auto)
}

pub fn tsi_with_kernel(input: &TsiInput, kernel: Kernel) -> Result<TsiOutput, TsiError> {
	let data: &[f64] = match &input.data {
		TsiData::Candles { candles, source } => source_type(candles, source),
		TsiData::Slice(sl) => sl,
	};

	let first = data.iter().position(|x| !x.is_nan()).ok_or(TsiError::AllValuesNaN)?;
	let len = data.len();
	let long = input.get_long_period();
	let short = input.get_short_period();

	if long == 0 || short == 0 || long > len || short > len {
		return Err(TsiError::InvalidPeriod {
			long_period: long,
			short_period: short,
			data_len: len,
		});
	}
	let needed = 1 + long + short;
	if (len - first) < needed {
		return Err(TsiError::NotEnoughValidData {
			needed,
			valid: len - first,
		});
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => tsi_scalar(data, long, short, first),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => tsi_avx2(data, long, short, first),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => tsi_avx512(data, long, short, first),
			_ => unreachable!(),
		}
	}
}

#[inline]
pub unsafe fn tsi_scalar(data: &[f64], long: usize, short: usize, first: usize) -> Result<TsiOutput, TsiError> {
	let mut out = vec![f64::NAN; data.len()];

	let mom_input = MomInput::from_slice(&data[first..], MomParams { period: Some(1) });
	let mom_output: MomOutput = mom(&mom_input)?;
	let abs_mom: Vec<f64> = mom_output
		.values
		.iter()
		.map(|&v| if v.is_nan() { f64::NAN } else { v.abs() })
		.collect();

	let ema_long_numer = EmaInput::from_slice(&mom_output.values, EmaParams { period: Some(long) });
	let ema_long_numer: EmaOutput = ema(&ema_long_numer)?;

	let ema_short_numer = EmaInput::from_slice(&ema_long_numer.values, EmaParams { period: Some(short) });
	let ema_short_numer: EmaOutput = ema(&ema_short_numer)?;

	let ema_long_denom = EmaInput::from_slice(&abs_mom, EmaParams { period: Some(long) });
	let ema_long_denom: EmaOutput = ema(&ema_long_denom)?;

	let ema_short_denom = EmaInput::from_slice(&ema_long_denom.values, EmaParams { period: Some(short) });
	let ema_short_denom: EmaOutput = ema(&ema_short_denom)?;

	for i in 0..data.len() {
		if i < first {
			out[i] = f64::NAN;
		} else {
			let idx = i - first;
			let numer = ema_short_numer.values[idx];
			let denom = ema_short_denom.values[idx];
			if numer.is_nan() || denom.is_nan() || denom == 0.0 {
				out[i] = f64::NAN;
			} else {
				out[i] = 100.0 * (numer / denom);
			}
		}
	}
	Ok(TsiOutput { values: out })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn tsi_avx2(data: &[f64], long: usize, short: usize, first: usize) -> Result<TsiOutput, TsiError> {
	// Stub, use scalar
	tsi_scalar(data, long, short, first)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn tsi_avx512(data: &[f64], long: usize, short: usize, first: usize) -> Result<TsiOutput, TsiError> {
	// Stub, dispatch short/long
	if long <= 32 && short <= 32 {
		tsi_avx512_short(data, long, short, first)
	} else {
		tsi_avx512_long(data, long, short, first)
	}
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn tsi_avx512_short(data: &[f64], long: usize, short: usize, first: usize) -> Result<TsiOutput, TsiError> {
	// Stub, use scalar
	tsi_scalar(data, long, short, first)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn tsi_avx512_long(data: &[f64], long: usize, short: usize, first: usize) -> Result<TsiOutput, TsiError> {
	// Stub, use scalar
	tsi_scalar(data, long, short, first)
}

// Streaming variant
#[derive(Debug, Clone)]
pub struct TsiStream {
	long: usize,
	short: usize,
	buffer: Vec<f64>,
	idx: usize,
	filled: bool,
	ema_long_num: crate::indicators::ema::EmaStream,
	ema_short_num: crate::indicators::ema::EmaStream,
	ema_long_den: crate::indicators::ema::EmaStream,
	ema_short_den: crate::indicators::ema::EmaStream,
}
impl TsiStream {
	pub fn try_new(params: TsiParams) -> Result<Self, TsiError> {
		let long = params.long_period.unwrap_or(25);
		let short = params.short_period.unwrap_or(13);
		Ok(Self {
			long,
			short,
			buffer: vec![f64::NAN; 1],
			idx: 0,
			filled: false,
			ema_long_num: crate::indicators::ema::EmaStream::try_new(EmaParams { period: Some(long) })?,
			ema_short_num: crate::indicators::ema::EmaStream::try_new(EmaParams { period: Some(short) })?,
			ema_long_den: crate::indicators::ema::EmaStream::try_new(EmaParams { period: Some(long) })?,
			ema_short_den: crate::indicators::ema::EmaStream::try_new(EmaParams { period: Some(short) })?,
		})
	}
	#[inline(always)]
	pub fn update(&mut self, value: f64) -> Option<f64> {
		let mom = if self.idx == 0 {
			f64::NAN
		} else {
			value - self.buffer[0]
		};
		self.buffer[0] = value;
		self.idx += 1;
		if mom.is_nan() {
			return None;
		}
		let ema_long_num = self.ema_long_num.update(mom)?;
		let ema_short_num = self.ema_short_num.update(ema_long_num)?;
		let ema_long_den = self.ema_long_den.update(mom.abs())?;
		let ema_short_den = self.ema_short_den.update(ema_long_den)?;
		if ema_short_den == 0.0 {
			Some(f64::NAN)
		} else {
			Some(100.0 * (ema_short_num / ema_short_den))
		}
	}
}

// Batch/grid API
#[derive(Clone, Debug)]
pub struct TsiBatchRange {
	pub long_period: (usize, usize, usize),
	pub short_period: (usize, usize, usize),
}
impl Default for TsiBatchRange {
	fn default() -> Self {
		Self {
			long_period: (25, 25, 1),
			short_period: (13, 13, 1),
		}
	}
}

#[derive(Clone, Debug, Default)]
pub struct TsiBatchBuilder {
	range: TsiBatchRange,
	kernel: Kernel,
}
impl TsiBatchBuilder {
	pub fn new() -> Self {
		Self::default()
	}
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	pub fn long_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.long_period = (start, end, step);
		self
	}
	pub fn long_static(mut self, n: usize) -> Self {
		self.range.long_period = (n, n, 1);
		self
	}
	pub fn short_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.short_period = (start, end, step);
		self
	}
	pub fn short_static(mut self, n: usize) -> Self {
		self.range.short_period = (n, n, 1);
		self
	}
	pub fn apply_slice(self, data: &[f64]) -> Result<TsiBatchOutput, TsiError> {
		tsi_batch_with_kernel(data, &self.range, self.kernel)
	}
	pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<TsiBatchOutput, TsiError> {
		TsiBatchBuilder::new().kernel(k).apply_slice(data)
	}
	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<TsiBatchOutput, TsiError> {
		let slice = source_type(c, src);
		self.apply_slice(slice)
	}
	pub fn with_default_candles(c: &Candles) -> Result<TsiBatchOutput, TsiError> {
		TsiBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
	}
}

pub fn tsi_batch_with_kernel(data: &[f64], sweep: &TsiBatchRange, k: Kernel) -> Result<TsiBatchOutput, TsiError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => {
			return Err(TsiError::InvalidPeriod {
				long_period: 0,
				short_period: 0,
				data_len: 0,
			})
		}
	};

	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	tsi_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct TsiBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<TsiParams>,
	pub rows: usize,
	pub cols: usize,
}
impl TsiBatchOutput {
	pub fn row_for_params(&self, p: &TsiParams) -> Option<usize> {
		self.combos.iter().position(|c| {
			c.long_period.unwrap_or(25) == p.long_period.unwrap_or(25)
				&& c.short_period.unwrap_or(13) == p.short_period.unwrap_or(13)
		})
	}
	pub fn values_for(&self, p: &TsiParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &TsiBatchRange) -> Vec<TsiParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let longs = axis_usize(r.long_period);
	let shorts = axis_usize(r.short_period);

	let mut out = Vec::with_capacity(longs.len() * shorts.len());
	for &l in &longs {
		for &s in &shorts {
			out.push(TsiParams {
				long_period: Some(l),
				short_period: Some(s),
			});
		}
	}
	out
}

#[inline(always)]
pub fn tsi_batch_slice(data: &[f64], sweep: &TsiBatchRange, kern: Kernel) -> Result<TsiBatchOutput, TsiError> {
	tsi_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn tsi_batch_par_slice(data: &[f64], sweep: &TsiBatchRange, kern: Kernel) -> Result<TsiBatchOutput, TsiError> {
	tsi_batch_inner(data, sweep, kern, true)
}

fn tsi_batch_inner(
	data: &[f64],
	sweep: &TsiBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<TsiBatchOutput, TsiError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(TsiError::InvalidPeriod {
			long_period: 0,
			short_period: 0,
			data_len: 0,
		});
	}

	let first = data.iter().position(|x| !x.is_nan()).ok_or(TsiError::AllValuesNaN)?;
	let max_long = combos.iter().map(|c| c.long_period.unwrap()).max().unwrap();
	let max_short = combos.iter().map(|c| c.short_period.unwrap()).max().unwrap();
	let max_needed = 1 + max_long + max_short;
	if data.len() - first < max_needed {
		return Err(TsiError::NotEnoughValidData {
			needed: max_needed,
			valid: data.len() - first,
		});
	}

	let rows = combos.len();
	let cols = data.len();
	let mut values = vec![f64::NAN; rows * cols];

	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let p = &combos[row];
		match kern {
			Kernel::Scalar => tsi_row_scalar(data, p.long_period.unwrap(), p.short_period.unwrap(), first, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => tsi_row_avx2(data, p.long_period.unwrap(), p.short_period.unwrap(), first, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => tsi_row_avx512(data, p.long_period.unwrap(), p.short_period.unwrap(), first, out_row),
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

	Ok(TsiBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
pub unsafe fn tsi_row_scalar(data: &[f64], long: usize, short: usize, first: usize, out: &mut [f64]) {
	let r = tsi_scalar(data, long, short, first).unwrap();
	out.copy_from_slice(&r.values);
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn tsi_row_avx2(data: &[f64], long: usize, short: usize, first: usize, out: &mut [f64]) {
	let r = tsi_avx2(data, long, short, first).unwrap();
	out.copy_from_slice(&r.values);
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn tsi_row_avx512(data: &[f64], long: usize, short: usize, first: usize, out: &mut [f64]) {
	if long <= 32 && short <= 32 {
		tsi_row_avx512_short(data, long, short, first, out);
	} else {
		tsi_row_avx512_long(data, long, short, first, out);
	}
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn tsi_row_avx512_short(data: &[f64], long: usize, short: usize, first: usize, out: &mut [f64]) {
	let r = tsi_avx512_short(data, long, short, first).unwrap();
	out.copy_from_slice(&r.values);
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn tsi_row_avx512_long(data: &[f64], long: usize, short: usize, first: usize, out: &mut [f64]) {
	let r = tsi_avx512_long(data, long, short, first).unwrap();
	out.copy_from_slice(&r.values);
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;
	use paste::paste;

	fn check_tsi_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let default_params = TsiParams {
			long_period: None,
			short_period: None,
		};
		let input = TsiInput::from_candles(&candles, "close", default_params);
		let output = tsi_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_tsi_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let params = TsiParams {
			long_period: Some(25),
			short_period: Some(13),
		};
		let input = TsiInput::from_candles(&candles, "close", params);
		let tsi_result = tsi_with_kernel(&input, kernel)?;

		let expected_last_five = [
			-17.757654061849838,
			-17.367527062626184,
			-17.305577681249513,
			-16.937565646991143,
			-17.61825617316731,
		];
		let start = tsi_result.values.len().saturating_sub(5);
		for (i, &val) in tsi_result.values[start..].iter().enumerate() {
			let diff = (val - expected_last_five[i]).abs();
			assert!(
				diff < 1e-7,
				"[{}] TSI {:?} mismatch at idx {}: got {}, expected {}",
				test_name,
				kernel,
				i,
				val,
				expected_last_five[i]
			);
		}
		Ok(())
	}

	fn check_tsi_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let input_data = [10.0, 20.0, 30.0];
		let params = TsiParams {
			long_period: Some(0),
			short_period: Some(13),
		};
		let input = TsiInput::from_slice(&input_data, params);
		let res = tsi_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] TSI should fail with zero period", test_name);
		Ok(())
	}

	fn check_tsi_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data_small = [10.0, 20.0, 30.0];
		let params = TsiParams {
			long_period: Some(25),
			short_period: Some(13),
		};
		let input = TsiInput::from_slice(&data_small, params);
		let res = tsi_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] TSI should fail with period exceeding length",
			test_name
		);
		Ok(())
	}

	fn check_tsi_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let single_point = [42.0];
		let params = TsiParams {
			long_period: Some(25),
			short_period: Some(13),
		};
		let input = TsiInput::from_slice(&single_point, params);
		let res = tsi_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] TSI should fail with insufficient data", test_name);
		Ok(())
	}

	fn check_tsi_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let input = TsiInput::with_default_candles(&candles);
		match input.data {
			TsiData::Candles { source, .. } => assert_eq!(source, "close"),
			_ => panic!("Expected TsiData::Candles"),
		}
		let output = tsi_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_tsi_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let first_params = TsiParams {
			long_period: Some(25),
			short_period: Some(13),
		};
		let first_input = TsiInput::from_candles(&candles, "close", first_params);
		let first_result = tsi_with_kernel(&first_input, kernel)?;

		let second_params = TsiParams {
			long_period: Some(25),
			short_period: Some(13),
		};
		let second_input = TsiInput::from_slice(&first_result.values, second_params);
		let second_result = tsi_with_kernel(&second_input, kernel)?;

		assert_eq!(second_result.values.len(), first_result.values.len());
		Ok(())
	}

	fn check_tsi_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let input = TsiInput::from_candles(
			&candles,
			"close",
			TsiParams {
				long_period: Some(25),
				short_period: Some(13),
			},
		);
		let res = tsi_with_kernel(&input, kernel)?;
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

	macro_rules! generate_all_tsi_tests {
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
	generate_all_tsi_tests!(
		check_tsi_partial_params,
		check_tsi_accuracy,
		check_tsi_zero_period,
		check_tsi_period_exceeds_length,
		check_tsi_very_small_dataset,
		check_tsi_default_candles,
		check_tsi_reinput,
		check_tsi_nan_handling
	);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let output = TsiBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;
		let def = TsiParams::default();
		let row = output.values_for(&def).expect("default row missing");
		assert_eq!(row.len(), c.close.len());
		let expected = [
			-17.757654061849838,
			-17.367527062626184,
			-17.305577681249513,
			-16.937565646991143,
			-17.61825617316731,
		];
		let start = row.len() - 5;
		for (i, &v) in row[start..].iter().enumerate() {
			assert!(
				(v - expected[i]).abs() < 1e-7,
				"[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
			);
		}
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
}
