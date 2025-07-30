//! # Laguerre RSI (LRSI)
//!
//! A momentum oscillator using a Laguerre filter, similar to RSI, but with different
//! responsiveness and smoothness characteristics. This implementation matches the
//! structure and feature parity of alma.rs, including AVX stubs, batch/grid support,
//! builder and streaming API, and full input validation.
//!
//! ## Parameters
//! - **alpha**: Smoothing factor (0 < alpha < 1). Default: 0.2
//!
//! ## Errors
//! - **AllValuesNaN**: lrsi: All input data values are `NaN`.
//! - **InvalidAlpha**: lrsi: `alpha` not in (0, 1).
//! - **EmptyData**: lrsi: Empty input.
//! - **NotEnoughValidData**: lrsi: Not enough valid data.
//!
//! ## Returns
//! - **Ok(LrsiOutput)** with `Vec<f64>` matching input
//! - **Err(LrsiError)** otherwise

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes, make_uninit_matrix};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum LrsiData<'a> {
	Candles { candles: &'a Candles },
	Slices { high: &'a [f64], low: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct LrsiOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct LrsiParams {
	pub alpha: Option<f64>,
}

impl Default for LrsiParams {
	fn default() -> Self {
		Self { alpha: Some(0.2) }
	}
}

#[derive(Debug, Clone)]
pub struct LrsiInput<'a> {
	pub data: LrsiData<'a>,
	pub params: LrsiParams,
}

impl<'a> LrsiInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, p: LrsiParams) -> Self {
		Self {
			data: LrsiData::Candles { candles: c },
			params: p,
		}
	}
	#[inline]
	pub fn from_slices(high: &'a [f64], low: &'a [f64], p: LrsiParams) -> Self {
		Self {
			data: LrsiData::Slices { high, low },
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, LrsiParams::default())
	}
	#[inline]
	pub fn get_alpha(&self) -> f64 {
		self.params.alpha.unwrap_or(0.2)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct LrsiBuilder {
	alpha: Option<f64>,
	kernel: Kernel,
}

impl Default for LrsiBuilder {
	fn default() -> Self {
		Self {
			alpha: None,
			kernel: Kernel::Auto,
		}
	}
}

impl LrsiBuilder {
	#[inline(always)]
	pub fn new() -> Self {
		Self::default()
	}
	#[inline(always)]
	pub fn alpha(mut self, x: f64) -> Self {
		self.alpha = Some(x);
		self
	}
	#[inline(always)]
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}

	#[inline(always)]
	pub fn apply(self, c: &Candles) -> Result<LrsiOutput, LrsiError> {
		let p = LrsiParams { alpha: self.alpha };
		let i = LrsiInput::from_candles(c, p);
		lrsi_with_kernel(&i, self.kernel)
	}

	#[inline(always)]
	pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<LrsiOutput, LrsiError> {
		let p = LrsiParams { alpha: self.alpha };
		let i = LrsiInput::from_slices(high, low, p);
		lrsi_with_kernel(&i, self.kernel)
	}

	#[inline(always)]
	pub fn into_stream(self) -> Result<LrsiStream, LrsiError> {
		let p = LrsiParams { alpha: self.alpha };
		LrsiStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum LrsiError {
	#[error("lrsi: Empty data provided.")]
	EmptyData,
	#[error("lrsi: Invalid alpha: alpha = {alpha}. Must be between 0 and 1.")]
	InvalidAlpha { alpha: f64 },
	#[error("lrsi: All values are NaN.")]
	AllValuesNaN,
	#[error("lrsi: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn lrsi(input: &LrsiInput) -> Result<LrsiOutput, LrsiError> {
	lrsi_with_kernel(input, Kernel::Auto)
}

pub fn lrsi_with_kernel(input: &LrsiInput, kernel: Kernel) -> Result<LrsiOutput, LrsiError> {
	let (high, low) = match &input.data {
		LrsiData::Candles { candles } => {
			let high = candles.select_candle_field("high").unwrap();
			let low = candles.select_candle_field("low").unwrap();
			(high, low)
		}
		LrsiData::Slices { high, low } => (*high, *low),
	};

	if high.is_empty() || low.is_empty() {
		return Err(LrsiError::EmptyData);
	}

	let alpha = input.get_alpha();
	if !(0.0 < alpha && alpha < 1.0) {
		return Err(LrsiError::InvalidAlpha { alpha });
	}

	let mut price = Vec::with_capacity(high.len());
	for i in 0..high.len() {
		price.push((high[i] + low[i]) / 2.0);
	}

	let first_valid_idx = match price.iter().position(|&x| !x.is_nan()) {
		Some(idx) => idx,
		None => return Err(LrsiError::AllValuesNaN),
	};
	let n = high.len();
	if n - first_valid_idx < 4 {
		return Err(LrsiError::NotEnoughValidData {
			needed: 4,
			valid: n - first_valid_idx,
		});
	}

	let mut out = vec![f64::NAN; n];

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => lrsi_scalar(&price, alpha, first_valid_idx, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => lrsi_avx2(&price, alpha, first_valid_idx, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => lrsi_avx512(&price, alpha, first_valid_idx, &mut out),
			_ => unreachable!(),
		}
	}

	Ok(LrsiOutput { values: out })
}

#[inline]
pub fn lrsi_scalar(price: &[f64], alpha: f64, first: usize, out: &mut [f64]) {
	let n = price.len();
	let mut l0 = vec![f64::NAN; n];
	let mut l1 = vec![f64::NAN; n];
	let mut l2 = vec![f64::NAN; n];
	let mut l3 = vec![f64::NAN; n];
	let gamma = 1.0 - alpha;

	l0[first] = price[first];
	l1[first] = price[first];
	l2[first] = price[first];
	l3[first] = price[first];
	out[first] = 0.0;

	for i in (first + 1)..n {
		let p = price[i];
		if p.is_nan() {
			continue;
		}
		let l0_prev = l0[i - 1];
		let l1_prev = l1[i - 1];
		let l2_prev = l2[i - 1];
		let l3_prev = l3[i - 1];

		l0[i] = alpha * p + gamma * l0_prev;
		l1[i] = -gamma * l0[i] + l0_prev + gamma * l1_prev;
		l2[i] = -gamma * l1[i] + l1_prev + gamma * l2_prev;
		l3[i] = -gamma * l2[i] + l2_prev + gamma * l3_prev;

		let mut cu = 0.0;
		let mut cd = 0.0;
		if l0[i] >= l1[i] {
			cu += l0[i] - l1[i];
		} else {
			cd += l1[i] - l0[i];
		}
		if l1[i] >= l2[i] {
			cu += l1[i] - l2[i];
		} else {
			cd += l2[i] - l1[i];
		}
		if l2[i] >= l3[i] {
			cu += l2[i] - l3[i];
		} else {
			cd += l3[i] - l2[i];
		}

		out[i] = if (cu + cd).abs() < f64::EPSILON {
			0.0
		} else {
			cu / (cu + cd)
		};
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn lrsi_avx2(price: &[f64], alpha: f64, first: usize, out: &mut [f64]) {
	lrsi_scalar(price, alpha, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn lrsi_avx512(price: &[f64], alpha: f64, first: usize, out: &mut [f64]) {
	lrsi_scalar(price, alpha, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn lrsi_avx512_short(price: &[f64], alpha: f64, first: usize, out: &mut [f64]) {
	lrsi_scalar(price, alpha, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn lrsi_avx512_long(price: &[f64], alpha: f64, first: usize, out: &mut [f64]) {
	lrsi_scalar(price, alpha, first, out)
}

// Streaming API
#[derive(Debug, Clone)]
pub struct LrsiStream {
	alpha: f64,
	gamma: f64,
	l0: f64,
	l1: f64,
	l2: f64,
	l3: f64,
	initialized: bool,
}

impl LrsiStream {
	pub fn try_new(params: LrsiParams) -> Result<Self, LrsiError> {
		let alpha = params.alpha.unwrap_or(0.2);
		if !(0.0 < alpha && alpha < 1.0) {
			return Err(LrsiError::InvalidAlpha { alpha });
		}
		Ok(Self {
			alpha,
			gamma: 1.0 - alpha,
			l0: f64::NAN,
			l1: f64::NAN,
			l2: f64::NAN,
			l3: f64::NAN,
			initialized: false,
		})
	}
	#[inline(always)]
	pub fn update(&mut self, price: f64) -> Option<f64> {
		if price.is_nan() {
			return None;
		}
		if !self.initialized {
			self.l0 = price;
			self.l1 = price;
			self.l2 = price;
			self.l3 = price;
			self.initialized = true;
			return Some(0.0);
		}
		let alpha = self.alpha;
		let gamma = self.gamma;

		let l0 = alpha * price + gamma * self.l0;
		let l1 = -gamma * l0 + self.l0 + gamma * self.l1;
		let l2 = -gamma * l1 + self.l1 + gamma * self.l2;
		let l3 = -gamma * l2 + self.l2 + gamma * self.l3;

		self.l0 = l0;
		self.l1 = l1;
		self.l2 = l2;
		self.l3 = l3;

		let mut cu = 0.0;
		let mut cd = 0.0;
		if l0 >= l1 {
			cu += l0 - l1;
		} else {
			cd += l1 - l0;
		}
		if l1 >= l2 {
			cu += l1 - l2;
		} else {
			cd += l2 - l1;
		}
		if l2 >= l3 {
			cu += l2 - l3;
		} else {
			cd += l3 - l2;
		}
		Some(if (cu + cd).abs() < f64::EPSILON {
			0.0
		} else {
			cu / (cu + cd)
		})
	}
}

#[derive(Clone, Debug)]
pub struct LrsiBatchRange {
	pub alpha: (f64, f64, f64),
}

impl Default for LrsiBatchRange {
	fn default() -> Self {
		Self { alpha: (0.2, 0.2, 0.0) }
	}
}

#[derive(Clone, Debug, Default)]
pub struct LrsiBatchBuilder {
	range: LrsiBatchRange,
	kernel: Kernel,
}

impl LrsiBatchBuilder {
	pub fn new() -> Self {
		Self::default()
	}
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}

	#[inline]
	pub fn alpha_range(mut self, start: f64, end: f64, step: f64) -> Self {
		self.range.alpha = (start, end, step);
		self
	}
	#[inline]
	pub fn alpha_static(mut self, x: f64) -> Self {
		self.range.alpha = (x, x, 0.0);
		self
	}

	pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<LrsiBatchOutput, LrsiError> {
		lrsi_batch_with_kernel(high, low, &self.range, self.kernel)
	}

	pub fn with_default_slices(high: &[f64], low: &[f64], k: Kernel) -> Result<LrsiBatchOutput, LrsiError> {
		LrsiBatchBuilder::new().kernel(k).apply_slices(high, low)
	}

	pub fn apply_candles(self, c: &Candles) -> Result<LrsiBatchOutput, LrsiError> {
		let high = c.select_candle_field("high").unwrap();
		let low = c.select_candle_field("low").unwrap();
		self.apply_slices(high, low)
	}

	pub fn with_default_candles(c: &Candles) -> Result<LrsiBatchOutput, LrsiError> {
		LrsiBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c)
	}
}

pub fn lrsi_batch_with_kernel(
	high: &[f64],
	low: &[f64],
	sweep: &LrsiBatchRange,
	k: Kernel,
) -> Result<LrsiBatchOutput, LrsiError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(LrsiError::EmptyData),
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	lrsi_batch_par_slice(high, low, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct LrsiBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<LrsiParams>,
	pub rows: usize,
	pub cols: usize,
}
impl LrsiBatchOutput {
	pub fn row_for_params(&self, p: &LrsiParams) -> Option<usize> {
		self.combos
			.iter()
			.position(|c| (c.alpha.unwrap_or(0.2) - p.alpha.unwrap_or(0.2)).abs() < 1e-12)
	}

	pub fn values_for(&self, p: &LrsiParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &LrsiBatchRange) -> Vec<LrsiParams> {
	fn axis_f64((start, end, step): (f64, f64, f64)) -> Vec<f64> {
		if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
			return vec![start];
		}
		let mut v = Vec::new();
		let mut x = start;
		while x <= end + 1e-12 {
			v.push(x);
			x += step;
		}
		v
	}

	let alphas = axis_f64(r.alpha);

	let mut out = Vec::with_capacity(alphas.len());
	for &a in &alphas {
		out.push(LrsiParams { alpha: Some(a) });
	}
	out
}

#[inline(always)]
pub fn lrsi_batch_slice(
	high: &[f64],
	low: &[f64],
	sweep: &LrsiBatchRange,
	kern: Kernel,
) -> Result<LrsiBatchOutput, LrsiError> {
	lrsi_batch_inner(high, low, sweep, kern, false)
}

#[inline(always)]
pub fn lrsi_batch_par_slice(
	high: &[f64],
	low: &[f64],
	sweep: &LrsiBatchRange,
	kern: Kernel,
) -> Result<LrsiBatchOutput, LrsiError> {
	lrsi_batch_inner(high, low, sweep, kern, true)
}

#[inline(always)]
fn lrsi_batch_inner(
	high: &[f64],
	low: &[f64],
	sweep: &LrsiBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<LrsiBatchOutput, LrsiError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(LrsiError::EmptyData);
	}
	if high.len() == 0 || low.len() == 0 {
		return Err(LrsiError::EmptyData);
	}
	let mut price = Vec::with_capacity(high.len());
	for i in 0..high.len() {
		price.push((high[i] + low[i]) / 2.0);
	}

	let first = price.iter().position(|&x| !x.is_nan()).ok_or(LrsiError::AllValuesNaN)?;
	let rows = combos.len();
	let cols = high.len();
	if cols - first < 4 {
		return Err(LrsiError::NotEnoughValidData {
			needed: 4,
			valid: cols - first,
		});
	}
	let mut values = vec![f64::NAN; rows * cols];
	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let alpha = combos[row].alpha.unwrap();
		match kern {
			Kernel::Scalar => lrsi_row_scalar(&price, first, alpha, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => lrsi_row_avx2(&price, first, alpha, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => lrsi_row_avx512(&price, first, alpha, out_row),
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

	Ok(LrsiBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
unsafe fn lrsi_row_scalar(price: &[f64], first: usize, alpha: f64, out: &mut [f64]) {
	lrsi_scalar(price, alpha, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn lrsi_row_avx2(price: &[f64], first: usize, alpha: f64, out: &mut [f64]) {
	lrsi_scalar(price, alpha, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn lrsi_row_avx512(price: &[f64], first: usize, alpha: f64, out: &mut [f64]) {
	lrsi_scalar(price, alpha, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn lrsi_row_avx512_short(price: &[f64], first: usize, alpha: f64, out: &mut [f64]) {
	lrsi_scalar(price, alpha, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn lrsi_row_avx512_long(price: &[f64], first: usize, alpha: f64, out: &mut [f64]) {
	lrsi_scalar(price, alpha, first, out)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_lrsi_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let default_params = LrsiParams { alpha: None };
		let input = LrsiInput::from_candles(&candles, default_params);
		let output = lrsi_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_lrsi_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = LrsiInput::from_candles(&candles, LrsiParams::default());
		let lrsi_result = lrsi_with_kernel(&input, kernel)?;
		assert_eq!(lrsi_result.values.len(), candles.close.len());
		let expected_last_five_lrsi = [0.0, 0.0, 0.0, 0.0, 0.0];
		let start_index = lrsi_result.values.len() - 5;
		let result_last_five_lrsi = &lrsi_result.values[start_index..];
		for (i, &value) in result_last_five_lrsi.iter().enumerate() {
			let expected_value = expected_last_five_lrsi[i];
			assert!(
				(value - expected_value).abs() < 1e-9,
				"LRSI mismatch at index {}: expected {}, got {}",
				i,
				expected_value,
				value
			);
		}
		Ok(())
	}

	fn check_lrsi_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = LrsiInput::with_default_candles(&candles);
		let output = lrsi_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_lrsi_invalid_alpha(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [1.0, 2.0];
		let low = [1.0, 2.0];
		let params = LrsiParams { alpha: Some(1.2) };
		let input = LrsiInput::from_slices(&high, &low, params);
		let result = lrsi_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}

	fn check_lrsi_empty_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high: [f64; 0] = [];
		let low: [f64; 0] = [];
		let params = LrsiParams::default();
		let input = LrsiInput::from_slices(&high, &low, params);
		let result = lrsi_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}

	fn check_lrsi_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [f64::NAN, f64::NAN, f64::NAN];
		let low = [f64::NAN, f64::NAN, f64::NAN];
		let params = LrsiParams::default();
		let input = LrsiInput::from_slices(&high, &low, params);
		let result = lrsi_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}

	fn check_lrsi_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [1.0, 1.0];
		let low = [1.0, 1.0];
		let params = LrsiParams::default();
		let input = LrsiInput::from_slices(&high, &low, params);
		let result = lrsi_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}

	fn check_lrsi_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let high = candles.select_candle_field("high").unwrap();
		let low = candles.select_candle_field("low").unwrap();

		let input = LrsiInput::from_slices(high, low, LrsiParams::default());
		let batch_output = lrsi_with_kernel(&input, kernel)?.values;

		let mut stream = LrsiStream::try_new(LrsiParams::default())?;
		let mut stream_values = Vec::with_capacity(high.len());
		for i in 0..high.len() {
			let price = (high[i] + low[i]) / 2.0;
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
				"[{}] LRSI streaming mismatch at idx {}: batch={}, stream={}, diff={}",
				test_name,
				i,
				b,
				s,
				diff
			);
		}
		Ok(())
	}

	macro_rules! generate_all_lrsi_tests {
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

	generate_all_lrsi_tests!(
		check_lrsi_partial_params,
		check_lrsi_accuracy,
		check_lrsi_default_candles,
		check_lrsi_invalid_alpha,
		check_lrsi_empty_data,
		check_lrsi_all_nan,
		check_lrsi_very_small_dataset,
		check_lrsi_streaming
	);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let output = LrsiBatchBuilder::new().kernel(kernel).apply_candles(&c)?;

		let def = LrsiParams::default();
		let row = output.values_for(&def).expect("default row missing");
		assert_eq!(row.len(), c.close.len());

		let expected = [0.0, 0.0, 0.0, 0.0, 0.0];
		let start = row.len() - 5;
		for (i, &v) in row[start..].iter().enumerate() {
			assert!(
				(v - expected[i]).abs() < 1e-9,
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
