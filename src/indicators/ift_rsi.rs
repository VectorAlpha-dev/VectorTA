//! # Inverse Fisher Transform RSI (IFT RSI)
//!
//! Applies Inverse Fisher Transform to a WMA-smoothed RSI series.
//! API closely matches alma.rs for interface, kernels, builders, batch/grid support, and error handling.

use crate::indicators::rsi::{rsi, RsiError, RsiInput, RsiParams};
use crate::indicators::wma::{wma, WmaError, WmaInput, WmaParams};
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

impl<'a> AsRef<[f64]> for IftRsiInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			IftRsiData::Slice(slice) => slice,
			IftRsiData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

#[derive(Debug, Clone)]
pub enum IftRsiData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct IftRsiOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct IftRsiParams {
	pub rsi_period: Option<usize>,
	pub wma_period: Option<usize>,
}

impl Default for IftRsiParams {
	fn default() -> Self {
		Self {
			rsi_period: Some(5),
			wma_period: Some(9),
		}
	}
}

#[derive(Debug, Clone)]
pub struct IftRsiInput<'a> {
	pub data: IftRsiData<'a>,
	pub params: IftRsiParams,
}

impl<'a> IftRsiInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: IftRsiParams) -> Self {
		Self {
			data: IftRsiData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slice(sl: &'a [f64], p: IftRsiParams) -> Self {
		Self {
			data: IftRsiData::Slice(sl),
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", IftRsiParams::default())
	}
	#[inline]
	pub fn get_rsi_period(&self) -> usize {
		self.params.rsi_period.unwrap_or(5)
	}
	#[inline]
	pub fn get_wma_period(&self) -> usize {
		self.params.wma_period.unwrap_or(9)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct IftRsiBuilder {
	rsi_period: Option<usize>,
	wma_period: Option<usize>,
	kernel: Kernel,
}

impl Default for IftRsiBuilder {
	fn default() -> Self {
		Self {
			rsi_period: None,
			wma_period: None,
			kernel: Kernel::Auto,
		}
	}
}

impl IftRsiBuilder {
	#[inline(always)]
	pub fn new() -> Self {
		Self::default()
	}
	#[inline(always)]
	pub fn rsi_period(mut self, n: usize) -> Self {
		self.rsi_period = Some(n);
		self
	}
	#[inline(always)]
	pub fn wma_period(mut self, n: usize) -> Self {
		self.wma_period = Some(n);
		self
	}
	#[inline(always)]
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}

	#[inline(always)]
	pub fn apply(self, c: &Candles) -> Result<IftRsiOutput, IftRsiError> {
		let p = IftRsiParams {
			rsi_period: self.rsi_period,
			wma_period: self.wma_period,
		};
		let i = IftRsiInput::from_candles(c, "close", p);
		ift_rsi_with_kernel(&i, self.kernel)
	}

	#[inline(always)]
	pub fn apply_slice(self, d: &[f64]) -> Result<IftRsiOutput, IftRsiError> {
		let p = IftRsiParams {
			rsi_period: self.rsi_period,
			wma_period: self.wma_period,
		};
		let i = IftRsiInput::from_slice(d, p);
		ift_rsi_with_kernel(&i, self.kernel)
	}

	#[inline(always)]
	pub fn into_stream(self) -> Result<IftRsiStream, IftRsiError> {
		let p = IftRsiParams {
			rsi_period: self.rsi_period,
			wma_period: self.wma_period,
		};
		IftRsiStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum IftRsiError {
	#[error("ift_rsi: No data provided.")]
	EmptyData,
	#[error("ift_rsi: All values are NaN.")]
	AllValuesNaN,
	#[error("ift_rsi: Invalid RSI period {rsi_period} or WMA period {wma_period}, data length = {data_len}.")]
	InvalidPeriod {
		rsi_period: usize,
		wma_period: usize,
		data_len: usize,
	},
	#[error("ift_rsi: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("ift_rsi: RSI calculation error: {0}")]
	RsiCalculationError(String),
	#[error("ift_rsi: WMA calculation error: {0}")]
	WmaCalculationError(String),
}

#[inline]
pub fn ift_rsi(input: &IftRsiInput) -> Result<IftRsiOutput, IftRsiError> {
	ift_rsi_with_kernel(input, Kernel::Auto)
}

pub fn ift_rsi_with_kernel(input: &IftRsiInput, kernel: Kernel) -> Result<IftRsiOutput, IftRsiError> {
	let data: &[f64] = match &input.data {
		IftRsiData::Candles { candles, source } => source_type(candles, source),
		IftRsiData::Slice(sl) => sl,
	};

	if data.is_empty() {
		return Err(IftRsiError::EmptyData);
	}
	let first = data.iter().position(|x| !x.is_nan()).ok_or(IftRsiError::AllValuesNaN)?;
	let len = data.len();
	let rsi_period = input.get_rsi_period();
	let wma_period = input.get_wma_period();
	if rsi_period == 0 || wma_period == 0 || rsi_period > len || wma_period > len {
		return Err(IftRsiError::InvalidPeriod {
			rsi_period,
			wma_period,
			data_len: len,
		});
	}
	let needed = rsi_period.max(wma_period);
	if (len - first) < needed {
		return Err(IftRsiError::NotEnoughValidData {
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
			Kernel::Scalar | Kernel::ScalarBatch => {
				ift_rsi_scalar(data, rsi_period, wma_period, first, &mut vec![f64::NAN; len])
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => ift_rsi_avx2(data, rsi_period, wma_period, first, &mut vec![f64::NAN; len]),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => {
				ift_rsi_avx512(data, rsi_period, wma_period, first, &mut vec![f64::NAN; len])
			}
			_ => unreachable!(),
		}
	}
}

#[inline]
pub fn ift_rsi_scalar(
	data: &[f64],
	rsi_period: usize,
	wma_period: usize,
	first_valid: usize,
	out: &mut [f64],
) -> Result<IftRsiOutput, IftRsiError> {
	let sliced = &data[first_valid..];
	let mut rsi_values = rsi(&RsiInput::from_slice(
		sliced,
		RsiParams {
			period: Some(rsi_period),
		},
	))
	.map_err(|e| IftRsiError::RsiCalculationError(e.to_string()))?
	.values;

	for val in rsi_values.iter_mut() {
		if !val.is_nan() {
			*val = 0.1 * (*val - 50.0);
		}
	}

	let wma_values = wma(&WmaInput::from_slice(
		&rsi_values,
		WmaParams {
			period: Some(wma_period),
		},
	))
	.map_err(|e| IftRsiError::WmaCalculationError(e.to_string()))?
	.values;

	for (i, &w) in wma_values.iter().enumerate() {
		if !w.is_nan() {
			let two_w = 2.0 * w;
			let numerator = two_w * two_w - 1.0;
			let denominator = two_w * two_w + 1.0;
			out[first_valid + i] = numerator / denominator;
		}
	}
	Ok(IftRsiOutput { values: out.to_vec() })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn ift_rsi_avx512(
	data: &[f64],
	rsi_period: usize,
	wma_period: usize,
	first_valid: usize,
	out: &mut [f64],
) -> Result<IftRsiOutput, IftRsiError> {
	ift_rsi_scalar(data, rsi_period, wma_period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn ift_rsi_avx2(
	data: &[f64],
	rsi_period: usize,
	wma_period: usize,
	first_valid: usize,
	out: &mut [f64],
) -> Result<IftRsiOutput, IftRsiError> {
	ift_rsi_scalar(data, rsi_period, wma_period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn ift_rsi_avx512_short(
	data: &[f64],
	rsi_period: usize,
	wma_period: usize,
	first_valid: usize,
	out: &mut [f64],
) -> Result<IftRsiOutput, IftRsiError> {
	ift_rsi_avx512(data, rsi_period, wma_period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn ift_rsi_avx512_long(
	data: &[f64],
	rsi_period: usize,
	wma_period: usize,
	first_valid: usize,
	out: &mut [f64],
) -> Result<IftRsiOutput, IftRsiError> {
	ift_rsi_avx512(data, rsi_period, wma_period, first_valid, out)
}

#[inline]
pub fn ift_rsi_batch_with_kernel(
	data: &[f64],
	sweep: &IftRsiBatchRange,
	k: Kernel,
) -> Result<IftRsiBatchOutput, IftRsiError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => {
			return Err(IftRsiError::InvalidPeriod {
				rsi_period: 0,
				wma_period: 0,
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
	ift_rsi_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct IftRsiBatchRange {
	pub rsi_period: (usize, usize, usize),
	pub wma_period: (usize, usize, usize),
}

impl Default for IftRsiBatchRange {
	fn default() -> Self {
		Self {
			rsi_period: (5, 21, 1),
			wma_period: (9, 21, 1),
		}
	}
}

#[derive(Clone, Debug, Default)]
pub struct IftRsiBatchBuilder {
	range: IftRsiBatchRange,
	kernel: Kernel,
}

impl IftRsiBatchBuilder {
	pub fn new() -> Self {
		Self::default()
	}
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline]
	pub fn rsi_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.rsi_period = (start, end, step);
		self
	}
	#[inline]
	pub fn rsi_period_static(mut self, p: usize) -> Self {
		self.range.rsi_period = (p, p, 0);
		self
	}
	#[inline]
	pub fn wma_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.wma_period = (start, end, step);
		self
	}
	#[inline]
	pub fn wma_period_static(mut self, n: usize) -> Self {
		self.range.wma_period = (n, n, 0);
		self
	}
	pub fn apply_slice(self, data: &[f64]) -> Result<IftRsiBatchOutput, IftRsiError> {
		ift_rsi_batch_with_kernel(data, &self.range, self.kernel)
	}
	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<IftRsiBatchOutput, IftRsiError> {
		let slice = source_type(c, src);
		self.apply_slice(slice)
	}
	pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<IftRsiBatchOutput, IftRsiError> {
		IftRsiBatchBuilder::new().kernel(k).apply_slice(data)
	}
	pub fn with_default_candles(c: &Candles) -> Result<IftRsiBatchOutput, IftRsiError> {
		IftRsiBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
	}
}

#[derive(Clone, Debug)]
pub struct IftRsiBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<IftRsiParams>,
	pub rows: usize,
	pub cols: usize,
}

impl IftRsiBatchOutput {
	pub fn row_for_params(&self, p: &IftRsiParams) -> Option<usize> {
		self.combos.iter().position(|c| {
			c.rsi_period.unwrap_or(5) == p.rsi_period.unwrap_or(5)
				&& c.wma_period.unwrap_or(9) == p.wma_period.unwrap_or(9)
		})
	}

	pub fn values_for(&self, p: &IftRsiParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &IftRsiBatchRange) -> Vec<IftRsiParams> {
	fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let rsi_periods = axis(r.rsi_period);
	let wma_periods = axis(r.wma_period);
	let mut out = Vec::with_capacity(rsi_periods.len() * wma_periods.len());
	for &rsi_p in &rsi_periods {
		for &wma_p in &wma_periods {
			out.push(IftRsiParams {
				rsi_period: Some(rsi_p),
				wma_period: Some(wma_p),
			});
		}
	}
	out
}

#[inline(always)]
pub fn ift_rsi_batch_slice(
	data: &[f64],
	sweep: &IftRsiBatchRange,
	kern: Kernel,
) -> Result<IftRsiBatchOutput, IftRsiError> {
	ift_rsi_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn ift_rsi_batch_par_slice(
	data: &[f64],
	sweep: &IftRsiBatchRange,
	kern: Kernel,
) -> Result<IftRsiBatchOutput, IftRsiError> {
	ift_rsi_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn ift_rsi_batch_inner(
	data: &[f64],
	sweep: &IftRsiBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<IftRsiBatchOutput, IftRsiError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(IftRsiError::InvalidPeriod {
			rsi_period: 0,
			wma_period: 0,
			data_len: 0,
		});
	}
	let first = data.iter().position(|x| !x.is_nan()).ok_or(IftRsiError::AllValuesNaN)?;
	let max_rsi = combos.iter().map(|c| c.rsi_period.unwrap()).max().unwrap();
	let max_wma = combos.iter().map(|c| c.wma_period.unwrap()).max().unwrap();
	let max_p = max_rsi.max(max_wma);
	if data.len() - first < max_p {
		return Err(IftRsiError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}

	let rows = combos.len();
	let cols = data.len();
	let mut values = vec![f64::NAN; rows * cols];

	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let rsi_p = combos[row].rsi_period.unwrap();
		let wma_p = combos[row].wma_period.unwrap();
		match kern {
			Kernel::Scalar => ift_rsi_row_scalar(data, first, rsi_p, wma_p, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => ift_rsi_row_avx2(data, first, rsi_p, wma_p, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => ift_rsi_row_avx512(data, first, rsi_p, wma_p, out_row),
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

	Ok(IftRsiBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
unsafe fn ift_rsi_row_scalar(data: &[f64], first: usize, rsi_period: usize, wma_period: usize, out: &mut [f64]) {
	let sliced = &data[first..];
	let mut rsi_values = match rsi(&RsiInput::from_slice(
		sliced,
		RsiParams {
			period: Some(rsi_period),
		},
	)) {
		Ok(r) => r.values,
		Err(_) => {
			return;
		}
	};
	for val in rsi_values.iter_mut() {
		if !val.is_nan() {
			*val = 0.1 * (*val - 50.0);
		}
	}
	let wma_values = match wma(&WmaInput::from_slice(
		&rsi_values,
		WmaParams {
			period: Some(wma_period),
		},
	)) {
		Ok(w) => w.values,
		Err(_) => {
			return;
		}
	};
	for (i, &w) in wma_values.iter().enumerate() {
		if !w.is_nan() {
			let two_w = 2.0 * w;
			let numerator = two_w * two_w - 1.0;
			let denominator = two_w * two_w + 1.0;
			out[first + i] = numerator / denominator;
		}
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn ift_rsi_row_avx2(data: &[f64], first: usize, rsi_period: usize, wma_period: usize, out: &mut [f64]) {
	ift_rsi_row_scalar(data, first, rsi_period, wma_period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn ift_rsi_row_avx512(data: &[f64], first: usize, rsi_period: usize, wma_period: usize, out: &mut [f64]) {
	if rsi_period.max(wma_period) <= 32 {
		ift_rsi_row_avx512_short(data, first, rsi_period, wma_period, out);
	} else {
		ift_rsi_row_avx512_long(data, first, rsi_period, wma_period, out);
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn ift_rsi_row_avx512_short(data: &[f64], first: usize, rsi_period: usize, wma_period: usize, out: &mut [f64]) {
	ift_rsi_row_scalar(data, first, rsi_period, wma_period, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn ift_rsi_row_avx512_long(data: &[f64], first: usize, rsi_period: usize, wma_period: usize, out: &mut [f64]) {
	ift_rsi_row_scalar(data, first, rsi_period, wma_period, out);
}

#[derive(Debug, Clone)]
pub struct IftRsiStream {
	rsi_period: usize,
	wma_period: usize,
	rsi_buf: Vec<f64>,
	wma_buf: Vec<f64>,
	head: usize,
	filled: bool,
}

impl IftRsiStream {
	pub fn try_new(params: IftRsiParams) -> Result<Self, IftRsiError> {
		let rsi_period = params.rsi_period.unwrap_or(5);
		let wma_period = params.wma_period.unwrap_or(9);
		if rsi_period == 0 || wma_period == 0 {
			return Err(IftRsiError::InvalidPeriod {
				rsi_period,
				wma_period,
				data_len: 0,
			});
		}
		Ok(Self {
			rsi_period,
			wma_period,
			rsi_buf: vec![f64::NAN; rsi_period],
			wma_buf: vec![f64::NAN; wma_period],
			head: 0,
			filled: false,
		})
	}
	pub fn update(&mut self, value: f64) -> Option<f64> {
		self.rsi_buf[self.head % self.rsi_period] = value;
		self.head += 1;
		if self.head < self.rsi_period {
			return None;
		}
		let rsi_slice = &self.rsi_buf;
		let rsi_val = match rsi(&RsiInput::from_slice(
			rsi_slice,
			RsiParams {
				period: Some(self.rsi_period),
			},
		)) {
			Ok(res) => res.values.last().cloned().unwrap_or(f64::NAN),
			Err(_) => return None,
		};
		let v1 = 0.1 * (rsi_val - 50.0);
		self.wma_buf[(self.head - self.rsi_period) % self.wma_period] = v1;
		if self.head < self.rsi_period + self.wma_period - 1 {
			return None;
		}
		let wma_slice = &self.wma_buf;
		let wma_val = match wma(&WmaInput::from_slice(
			wma_slice,
			WmaParams {
				period: Some(self.wma_period),
			},
		)) {
			Ok(res) => res.values.last().cloned().unwrap_or(f64::NAN),
			Err(_) => return None,
		};
		if wma_val.is_nan() {
			None
		} else {
			let two_w = 2.0 * wma_val;
			let numerator = two_w * two_w - 1.0;
			let denominator = two_w * two_w + 1.0;
			Some(numerator / denominator)
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_ift_rsi_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let default_params = IftRsiParams {
			rsi_period: None,
			wma_period: None,
		};
		let input = IftRsiInput::from_candles(&candles, "close", default_params);
		let output = ift_rsi_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_ift_rsi_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = IftRsiInput::from_candles(&candles, "close", IftRsiParams::default());
		let result = ift_rsi_with_kernel(&input, kernel)?;
		let expected_last_five = [
			-0.27763026899967286,
			-0.367418234207824,
			-0.1650156844504996,
			-0.26631220621545837,
			0.28324385010826775,
		];
		let start = result.values.len().saturating_sub(5);
		for (i, &val) in result.values[start..].iter().enumerate() {
			let diff = (val - expected_last_five[i]).abs();
			assert!(
				diff < 1e-8,
				"[{}] IFT_RSI {:?} mismatch at idx {}: got {}, expected {}",
				test_name,
				kernel,
				i,
				val,
				expected_last_five[i]
			);
		}
		Ok(())
	}

	fn check_ift_rsi_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = IftRsiInput::with_default_candles(&candles);
		let output = ift_rsi_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_ift_rsi_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let input_data = [10.0, 20.0, 30.0];
		let params = IftRsiParams {
			rsi_period: Some(0),
			wma_period: Some(9),
		};
		let input = IftRsiInput::from_slice(&input_data, params);
		let res = ift_rsi_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] IFT_RSI should fail with zero period", test_name);
		Ok(())
	}

	fn check_ift_rsi_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data_small = [10.0, 20.0, 30.0];
		let params = IftRsiParams {
			rsi_period: Some(10),
			wma_period: Some(9),
		};
		let input = IftRsiInput::from_slice(&data_small, params);
		let res = ift_rsi_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] IFT_RSI should fail with period exceeding length",
			test_name
		);
		Ok(())
	}

	fn check_ift_rsi_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let single_point = [42.0];
		let params = IftRsiParams {
			rsi_period: Some(5),
			wma_period: Some(9),
		};
		let input = IftRsiInput::from_slice(&single_point, params);
		let res = ift_rsi_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] IFT_RSI should fail with insufficient data",
			test_name
		);
		Ok(())
	}

	fn check_ift_rsi_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let first_params = IftRsiParams {
			rsi_period: Some(5),
			wma_period: Some(9),
		};
		let first_input = IftRsiInput::from_candles(&candles, "close", first_params);
		let first_result = ift_rsi_with_kernel(&first_input, kernel)?;
		let second_params = IftRsiParams {
			rsi_period: Some(5),
			wma_period: Some(9),
		};
		let second_input = IftRsiInput::from_slice(&first_result.values, second_params);
		let second_result = ift_rsi_with_kernel(&second_input, kernel)?;
		assert_eq!(second_result.values.len(), first_result.values.len());
		Ok(())
	}

	fn check_ift_rsi_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = IftRsiInput::from_candles(
			&candles,
			"close",
			IftRsiParams {
				rsi_period: Some(5),
				wma_period: Some(9),
			},
		);
		let res = ift_rsi_with_kernel(&input, kernel)?;
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

	macro_rules! generate_all_ift_rsi_tests {
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

	generate_all_ift_rsi_tests!(
		check_ift_rsi_partial_params,
		check_ift_rsi_accuracy,
		check_ift_rsi_default_candles,
		check_ift_rsi_zero_period,
		check_ift_rsi_period_exceeds_length,
		check_ift_rsi_very_small_dataset,
		check_ift_rsi_reinput,
		check_ift_rsi_nan_handling
	);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let output = IftRsiBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;
		let def = IftRsiParams::default();
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
