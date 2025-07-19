//! # Bollinger Bands Width (BBW)
//!
//! Bollinger Bands Width (sometimes called Bandwidth) shows the relative distance between
//! the upper and lower Bollinger Bands compared to the middle band.
//! It is typically calculated as: `(upper_band - lower_band) / middle_band`
//!
//! ## Parameters
//! - **period**: Underlying MA window (default: 20)
//! - **devup**: Upward multiplier (default: 2.0)
//! - **devdn**: Downward multiplier (default: 2.0)
//! - **matype**: MA type as string (default: "sma")
//! - **devtype**: 0 = stddev, 1 = mean_ad, 2 = median_ad (default: 0)
//!
//! ## Returns
//! - **`Ok(BollingerBandsWidthOutput)`**: Vec<f64> of same length as input
//! - **`Err(BollingerBandsWidthError)`** otherwise

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
	alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::mem::MaybeUninit;
use thiserror::Error;

impl<'a> AsRef<[f64]> for BollingerBandsWidthInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			BollingerBandsWidthData::Slice(s) => s,
			BollingerBandsWidthData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

#[derive(Debug, Clone)]
pub enum BollingerBandsWidthData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct BollingerBandsWidthOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct BollingerBandsWidthParams {
	pub period: Option<usize>,
	pub devup: Option<f64>,
	pub devdn: Option<f64>,
	pub matype: Option<String>,
	pub devtype: Option<usize>,
}

impl Default for BollingerBandsWidthParams {
	fn default() -> Self {
		Self {
			period: Some(20),
			devup: Some(2.0),
			devdn: Some(2.0),
			matype: Some("sma".to_string()),
			devtype: Some(0),
		}
	}
}

#[derive(Debug, Clone)]
pub struct BollingerBandsWidthInput<'a> {
	pub data: BollingerBandsWidthData<'a>,
	pub params: BollingerBandsWidthParams,
}

impl<'a> BollingerBandsWidthInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: BollingerBandsWidthParams) -> Self {
		Self {
			data: BollingerBandsWidthData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slice(sl: &'a [f64], p: BollingerBandsWidthParams) -> Self {
		Self {
			data: BollingerBandsWidthData::Slice(sl),
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", BollingerBandsWidthParams::default())
	}
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(20)
	}
	#[inline]
	pub fn get_devup(&self) -> f64 {
		self.params.devup.unwrap_or(2.0)
	}
	#[inline]
	pub fn get_devdn(&self) -> f64 {
		self.params.devdn.unwrap_or(2.0)
	}
	#[inline]
	pub fn get_matype(&self) -> String {
		self.params.matype.clone().unwrap_or_else(|| "sma".to_string())
	}
	#[inline]
	pub fn get_devtype(&self) -> usize {
		self.params.devtype.unwrap_or(0)
	}
}

#[derive(Debug, Error)]
pub enum BollingerBandsWidthError {
	#[error("bbw: Empty data provided.")]
	EmptyData,
	#[error("bbw: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("bbw: All values are NaN.")]
	AllValuesNaN,
	#[error("bbw: Underlying MA or Deviation function failed: {0}")]
	UnderlyingFunctionFailed(String),
	#[error("bbw: MA calculation error: {0}")]
	MaError(String),
	#[error("bbw: Deviation calculation error: {0}")]
	DeviationError(String),
	#[error("bbw: Not enough valid data for period: needed={needed}, valid={valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
}

#[derive(Clone, Debug)]
pub struct BollingerBandsWidthBuilder {
	period: Option<usize>,
	devup: Option<f64>,
	devdn: Option<f64>,
	matype: Option<String>,
	devtype: Option<usize>,
	kernel: Kernel,
}

impl Default for BollingerBandsWidthBuilder {
	fn default() -> Self {
		Self {
			period: None,
			devup: None,
			devdn: None,
			matype: None,
			devtype: None,
			kernel: Kernel::Auto,
		}
	}
}

impl BollingerBandsWidthBuilder {
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
	pub fn devup(mut self, x: f64) -> Self {
		self.devup = Some(x);
		self
	}
	#[inline(always)]
	pub fn devdn(mut self, x: f64) -> Self {
		self.devdn = Some(x);
		self
	}
	#[inline(always)]
	pub fn matype(mut self, x: &str) -> Self {
		self.matype = Some(x.to_string());
		self
	}
	#[inline(always)]
	pub fn devtype(mut self, x: usize) -> Self {
		self.devtype = Some(x);
		self
	}
	#[inline(always)]
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline(always)]
	pub fn apply(self, c: &Candles) -> Result<BollingerBandsWidthOutput, BollingerBandsWidthError> {
		let p = BollingerBandsWidthParams {
			period: self.period,
			devup: self.devup,
			devdn: self.devdn,
			matype: self.matype,
			devtype: self.devtype,
		};
		let i = BollingerBandsWidthInput::from_candles(c, "close", p);
		bollinger_bands_width_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slice(self, d: &[f64]) -> Result<BollingerBandsWidthOutput, BollingerBandsWidthError> {
		let p = BollingerBandsWidthParams {
			period: self.period,
			devup: self.devup,
			devdn: self.devdn,
			matype: self.matype,
			devtype: self.devtype,
		};
		let i = BollingerBandsWidthInput::from_slice(d, p);
		bollinger_bands_width_with_kernel(&i, self.kernel)
	}
}

#[inline]
pub fn bollinger_bands_width(
	input: &BollingerBandsWidthInput,
) -> Result<BollingerBandsWidthOutput, BollingerBandsWidthError> {
	bollinger_bands_width_with_kernel(input, Kernel::Auto)
}

pub fn bollinger_bands_width_with_kernel(
	input: &BollingerBandsWidthInput,
	kernel: Kernel,
) -> Result<BollingerBandsWidthOutput, BollingerBandsWidthError> {
	let data: &[f64] = input.as_ref();
	if data.is_empty() {
		return Err(BollingerBandsWidthError::EmptyData);
	}

	let period = input.get_period();
	if period == 0 || period > data.len() {
		return Err(BollingerBandsWidthError::InvalidPeriod {
			period,
			data_len: data.len(),
		});
	}

	let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
		Some(idx) => idx,
		None => return Err(BollingerBandsWidthError::AllValuesNaN),
	};

	if (data.len() - first_valid_idx) < period {
		return Err(BollingerBandsWidthError::NotEnoughValidData {
			needed: period,
			valid: data.len() - first_valid_idx,
		});
	}

	// Calculate warmup period for zero-copy allocation
	let warmup_period = first_valid_idx + period - 1;
	let mut out = alloc_with_nan_prefix(data.len(), warmup_period);

	bollinger_bands_width_into(data, input, &mut out, kernel)?;
	Ok(BollingerBandsWidthOutput { values: out })
}

/// Compute Bollinger Bands Width directly into a pre-allocated buffer.
/// This is the zero-copy variant used by Python/WASM bindings.
pub fn bollinger_bands_width_into(
	data: &[f64],
	input: &BollingerBandsWidthInput,
	out: &mut [f64],
	kernel: Kernel,
) -> Result<(), BollingerBandsWidthError> {
	if data.is_empty() {
		return Err(BollingerBandsWidthError::EmptyData);
	}
	if data.len() != out.len() {
		return Err(BollingerBandsWidthError::InvalidPeriod {
			period: 0,
			data_len: data.len(),
		});
	}
	let period = input.get_period();
	if period == 0 || period > data.len() {
		return Err(BollingerBandsWidthError::InvalidPeriod {
			period,
			data_len: data.len(),
		});
	}
	let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
		Some(idx) => idx,
		None => return Err(BollingerBandsWidthError::AllValuesNaN),
	};
	if (data.len() - first_valid_idx) < period {
		return Err(BollingerBandsWidthError::NotEnoughValidData {
			needed: period,
			valid: data.len() - first_valid_idx,
		});
	}

	// Note: The caller is responsible for pre-filling NaN values in the warmup period
	// using alloc_with_nan_prefix or similar methods

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => {
				bollinger_bands_width_scalar_into(data, input, first_valid_idx, out)
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => bollinger_bands_width_avx2_into(data, input, first_valid_idx, out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => bollinger_bands_width_avx512_into(data, input, first_valid_idx, out),
			_ => unreachable!(),
		}
	}
}

#[inline]
pub unsafe fn bollinger_bands_width_scalar(
	data: &[f64],
	input: &BollingerBandsWidthInput,
	first_valid_idx: usize,
) -> Result<BollingerBandsWidthOutput, BollingerBandsWidthError> {
	let period = input.get_period();
	let warmup_period = first_valid_idx + period - 1;
	let mut out = alloc_with_nan_prefix(data.len(), warmup_period);
	bollinger_bands_width_scalar_into(data, input, first_valid_idx, &mut out)?;
	Ok(BollingerBandsWidthOutput { values: out })
}

#[inline]
pub unsafe fn bollinger_bands_width_scalar_into(
	data: &[f64],
	input: &BollingerBandsWidthInput,
	first_valid_idx: usize,
	out: &mut [f64],
) -> Result<(), BollingerBandsWidthError> {
	let period = input.get_period();
	let devup = input.get_devup();
	let devdn = input.get_devdn();
	let matype = input.get_matype();
	let devtype = input.get_devtype();
	let ma_data = match &input.data {
		BollingerBandsWidthData::Candles { candles, source } => {
			crate::indicators::moving_averages::ma::MaData::Candles { candles, source }
		}
		BollingerBandsWidthData::Slice(slice) => crate::indicators::moving_averages::ma::MaData::Slice(slice),
	};
	let middle = crate::indicators::moving_averages::ma::ma(&matype, ma_data, period)
		.map_err(|e| BollingerBandsWidthError::UnderlyingFunctionFailed(e.to_string()))?;
	let dev_input = crate::indicators::deviation::DevInput::from_slice(
		data,
		crate::indicators::deviation::DevParams {
			period: Some(period),
			devtype: Some(devtype),
		},
	);
	let dev_values = crate::indicators::deviation::deviation(&dev_input)
		.map_err(|e| BollingerBandsWidthError::UnderlyingFunctionFailed(e.to_string()))?;

	for i in (first_valid_idx + period - 1)..data.len() {
		let middle_band = middle[i];
		let upper_band = middle[i] + devup * dev_values[i];
		let lower_band = middle[i] - devdn * dev_values[i];
		out[i] = (upper_band - lower_band) / middle_band;
	}
	Ok(())
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn bollinger_bands_width_avx512(
	data: &[f64],
	input: &BollingerBandsWidthInput,
	first_valid_idx: usize,
) -> Result<BollingerBandsWidthOutput, BollingerBandsWidthError> {
	bollinger_bands_width_scalar(data, input, first_valid_idx)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn bollinger_bands_width_avx512_into(
	data: &[f64],
	input: &BollingerBandsWidthInput,
	first_valid_idx: usize,
	out: &mut [f64],
) -> Result<(), BollingerBandsWidthError> {
	bollinger_bands_width_scalar_into(data, input, first_valid_idx, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn bollinger_bands_width_avx2(
	data: &[f64],
	input: &BollingerBandsWidthInput,
	first_valid_idx: usize,
) -> Result<BollingerBandsWidthOutput, BollingerBandsWidthError> {
	bollinger_bands_width_scalar(data, input, first_valid_idx)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn bollinger_bands_width_avx2_into(
	data: &[f64],
	input: &BollingerBandsWidthInput,
	first_valid_idx: usize,
	out: &mut [f64],
) -> Result<(), BollingerBandsWidthError> {
	bollinger_bands_width_scalar_into(data, input, first_valid_idx, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn bollinger_bands_width_avx512_short(
	data: &[f64],
	input: &BollingerBandsWidthInput,
	first_valid_idx: usize,
) -> Result<BollingerBandsWidthOutput, BollingerBandsWidthError> {
	bollinger_bands_width_avx512(data, input, first_valid_idx)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn bollinger_bands_width_avx512_long(
	data: &[f64],
	input: &BollingerBandsWidthInput,
	first_valid_idx: usize,
) -> Result<BollingerBandsWidthOutput, BollingerBandsWidthError> {
	bollinger_bands_width_avx512(data, input, first_valid_idx)
}

#[inline(always)]
pub fn bollinger_bands_width_row_scalar(
	data: &[f64],
	input: &BollingerBandsWidthInput,
	first_valid_idx: usize,
	out: &mut [f64],
) {
	let period = input.get_period();
	let devup = input.get_devup();
	let devdn = input.get_devdn();
	let matype = input.get_matype();
	let devtype = input.get_devtype();
	let ma_data = match &input.data {
		BollingerBandsWidthData::Candles { candles, source } => {
			crate::indicators::moving_averages::ma::MaData::Candles { candles, source }
		}
		BollingerBandsWidthData::Slice(slice) => crate::indicators::moving_averages::ma::MaData::Slice(slice),
	};
	let middle = crate::indicators::moving_averages::ma::ma(&matype, ma_data, period).unwrap();
	let dev_input = crate::indicators::deviation::DevInput::from_slice(
		data,
		crate::indicators::deviation::DevParams {
			period: Some(period),
			devtype: Some(devtype),
		},
	);
	let dev_values = crate::indicators::deviation::deviation(&dev_input).unwrap();
	for i in (first_valid_idx + period - 1)..data.len() {
		let middle_band = middle[i];
		let upper_band = middle[i] + devup * dev_values[i];
		let lower_band = middle[i] - devdn * dev_values[i];
		out[i] = (upper_band - lower_band) / middle_band;
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn bollinger_bands_width_row_avx2(
	data: &[f64],
	input: &BollingerBandsWidthInput,
	first_valid_idx: usize,
	out: &mut [f64],
) {
	bollinger_bands_width_row_scalar(data, input, first_valid_idx, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn bollinger_bands_width_row_avx512(
	data: &[f64],
	input: &BollingerBandsWidthInput,
	first_valid_idx: usize,
	out: &mut [f64],
) {
	bollinger_bands_width_row_scalar(data, input, first_valid_idx, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn bollinger_bands_width_row_avx512_short(
	data: &[f64],
	input: &BollingerBandsWidthInput,
	first_valid_idx: usize,
	out: &mut [f64],
) {
	bollinger_bands_width_row_avx512(data, input, first_valid_idx, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn bollinger_bands_width_row_avx512_long(
	data: &[f64],
	input: &BollingerBandsWidthInput,
	first_valid_idx: usize,
	out: &mut [f64],
) {
	bollinger_bands_width_row_avx512(data, input, first_valid_idx, out)
}

#[derive(Clone, Debug)]
pub struct BollingerBandsWidthBatchRange {
	pub period: (usize, usize, usize),
	pub devup: (f64, f64, f64),
	pub devdn: (f64, f64, f64),
}

impl Default for BollingerBandsWidthBatchRange {
	fn default() -> Self {
		Self {
			period: (20, 60, 1),
			devup: (2.0, 2.0, 0.0),
			devdn: (2.0, 2.0, 0.0),
		}
	}
}

#[derive(Clone, Debug, Default)]
pub struct BollingerBandsWidthBatchBuilder {
	range: BollingerBandsWidthBatchRange,
	kernel: Kernel,
}

impl BollingerBandsWidthBatchBuilder {
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
	pub fn devup_range(mut self, start: f64, end: f64, step: f64) -> Self {
		self.range.devup = (start, end, step);
		self
	}
	#[inline]
	pub fn devdn_range(mut self, start: f64, end: f64, step: f64) -> Self {
		self.range.devdn = (start, end, step);
		self
	}
	pub fn apply_slice(self, data: &[f64]) -> Result<BollingerBandsWidthBatchOutput, BollingerBandsWidthError> {
		bollinger_bands_width_batch_with_kernel(data, &self.range, self.kernel)
	}
	pub fn apply_candles(
		self,
		c: &Candles,
		src: &str,
	) -> Result<BollingerBandsWidthBatchOutput, BollingerBandsWidthError> {
		let slice = source_type(c, src);
		self.apply_slice(slice)
	}
	pub fn with_default_candles(c: &Candles) -> Result<BollingerBandsWidthBatchOutput, BollingerBandsWidthError> {
		BollingerBandsWidthBatchBuilder::new()
			.kernel(Kernel::Auto)
			.apply_candles(c, "close")
	}
}

#[derive(Clone, Debug)]
pub struct BollingerBandsWidthBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<BollingerBandsWidthParams>,
	pub rows: usize,
	pub cols: usize,
}
impl BollingerBandsWidthBatchOutput {
	pub fn row_for_params(&self, p: &BollingerBandsWidthParams) -> Option<usize> {
		self.combos.iter().position(|c| {
			c.period.unwrap_or(20) == p.period.unwrap_or(20)
				&& (c.devup.unwrap_or(2.0) - p.devup.unwrap_or(2.0)).abs() < 1e-12
				&& (c.devdn.unwrap_or(2.0) - p.devdn.unwrap_or(2.0)).abs() < 1e-12
		})
	}
	pub fn values_for(&self, p: &BollingerBandsWidthParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &BollingerBandsWidthBatchRange) -> Vec<BollingerBandsWidthParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
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

	let periods = axis_usize(r.period);
	let devups = axis_f64(r.devup);
	let devdns = axis_f64(r.devdn);

	let mut out = Vec::with_capacity(periods.len() * devups.len() * devdns.len());
	for &p in &periods {
		for &u in &devups {
			for &d in &devdns {
				out.push(BollingerBandsWidthParams {
					period: Some(p),
					devup: Some(u),
					devdn: Some(d),
					matype: Some("sma".to_string()),
					devtype: Some(0),
				});
			}
		}
	}
	out
}

#[inline(always)]
pub fn bollinger_bands_width_batch_with_kernel(
	data: &[f64],
	sweep: &BollingerBandsWidthBatchRange,
	k: Kernel,
) -> Result<BollingerBandsWidthBatchOutput, BollingerBandsWidthError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(BollingerBandsWidthError::InvalidPeriod { period: 0, data_len: 0 }),
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	bollinger_bands_width_batch_par_slice(data, sweep, simd)
}

#[inline(always)]
pub fn bollinger_bands_width_batch_slice(
	data: &[f64],
	sweep: &BollingerBandsWidthBatchRange,
	kern: Kernel,
) -> Result<BollingerBandsWidthBatchOutput, BollingerBandsWidthError> {
	bollinger_bands_width_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn bollinger_bands_width_batch_par_slice(
	data: &[f64],
	sweep: &BollingerBandsWidthBatchRange,
	kern: Kernel,
) -> Result<BollingerBandsWidthBatchOutput, BollingerBandsWidthError> {
	bollinger_bands_width_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn bollinger_bands_width_batch_inner(
	data: &[f64],
	sweep: &BollingerBandsWidthBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<BollingerBandsWidthBatchOutput, BollingerBandsWidthError> {
	let combos = expand_grid(sweep);
	let rows = combos.len();
	let cols = data.len();

	// Check for empty data and return AllValuesNaN for consistency
	if cols == 0 {
		return Err(BollingerBandsWidthError::AllValuesNaN);
	}

	// Step 1: Allocate uninitialized matrix
	let mut buf_mu = make_uninit_matrix(rows, cols);

	// Step 2: Calculate warmup periods for each row
	let warm: Vec<usize> = combos
		.iter()
		.map(|c| data.iter().position(|x| !x.is_nan()).unwrap_or(0) + c.period.unwrap() - 1)
		.collect();

	// Step 3: Initialize NaN prefixes for each row
	init_matrix_prefixes(&mut buf_mu, cols, &warm);

	// Step 4: Convert to mutable slice for computation
	let mut buf_guard = std::mem::ManuallyDrop::new(buf_mu);
	let values_slice: &mut [f64] =
		unsafe { core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len()) };

	// Step 5: Compute into the buffer
	bollinger_bands_width_batch_inner_into(data, sweep, kern, parallel, values_slice)?;

	// Step 6: Reclaim as Vec<f64>
	let values = unsafe {
		Vec::from_raw_parts(
			buf_guard.as_mut_ptr() as *mut f64,
			buf_guard.len(),
			buf_guard.capacity(),
		)
	};

	Ok(BollingerBandsWidthBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

/// Compute batch Bollinger Bands Width directly into a pre-allocated buffer.
/// This is the zero-copy variant used by Python bindings for batch operations.
#[inline(always)]
pub fn bollinger_bands_width_batch_inner_into(
	data: &[f64],
	sweep: &BollingerBandsWidthBatchRange,
	kern: Kernel,
	parallel: bool,
	out: &mut [f64],
) -> Result<Vec<BollingerBandsWidthParams>, BollingerBandsWidthError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(BollingerBandsWidthError::InvalidPeriod { period: 0, data_len: 0 });
	}
	let first = data
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(BollingerBandsWidthError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if data.len() - first < max_p {
		return Err(BollingerBandsWidthError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}

	let cols = data.len();

	// Group combinations by (period, matype, devtype) to avoid redundant calculations
	use std::collections::HashMap;
	let mut groups: HashMap<(usize, String, usize), Vec<(usize, f64, f64)>> = HashMap::new();

	for (idx, combo) in combos.iter().enumerate() {
		let key = (
			combo.period.unwrap(),
			combo.matype.as_ref().unwrap_or(&"sma".to_string()).clone(),
			combo.devtype.unwrap_or(0),
		);
		groups
			.entry(key)
			.or_insert_with(Vec::new)
			.push((idx, combo.devup.unwrap(), combo.devdn.unwrap()));
	}

	// Process each unique (period, matype, devtype) group
	for ((period, matype, devtype), indices) in groups {
		// Compute MA and deviation once for this group
		let ma_data = crate::indicators::moving_averages::ma::MaData::Slice(data);
		let middle = crate::indicators::moving_averages::ma::ma(&matype, ma_data, period)
			.map_err(|e| BollingerBandsWidthError::MaError(e.to_string()))?;

		let dev_input = crate::indicators::deviation::DevInput::from_slice(
			data,
			crate::indicators::deviation::DevParams {
				period: Some(period),
				devtype: Some(devtype),
			},
		);
		let dev_values = crate::indicators::deviation::deviation(&dev_input)
			.map_err(|e| BollingerBandsWidthError::DeviationError(e.to_string()))?;

		// Now compute BBW for each (devup, devdn) combination in this group
		if parallel {
			#[cfg(not(target_arch = "wasm32"))]
			{
				use rayon::prelude::*;
				// Create a list of (start_idx, end_idx) for parallel processing
				let ranges: Vec<_> = indices
					.iter()
					.map(|&(idx, devup, devdn)| (idx * cols, (idx + 1) * cols, devup, devdn))
					.collect();

				// Clone necessary data for parallel processing
				let middle_ref = &middle;
				let dev_values_ref = &dev_values.values;

				// Use parallel chunks_mut to safely partition the output slice
				out.par_chunks_mut(cols).enumerate().for_each(|(row_idx, out_row)| {
					// Find the corresponding parameters for this row
					if let Some(&(_, devup, devdn)) = indices.iter().find(|&&(idx, _, _)| idx == row_idx) {
						for i in (first + period - 1)..cols {
							let middle_band = middle_ref[i];
							let upper_band = middle_band + devup * dev_values_ref[i];
							let lower_band = middle_band - devdn * dev_values_ref[i];
							out_row[i] = (upper_band - lower_band) / middle_band;
						}
					}
				});
			}

			#[cfg(target_arch = "wasm32")]
			{
				for &(idx, devup, devdn) in &indices {
					let start = idx * cols;
					let end = start + cols;
					let out_row = &mut out[start..end];
					for i in (first + period - 1)..cols {
						let middle_band = middle[i];
						let upper_band = middle_band + devup * dev_values.values[i];
						let lower_band = middle_band - devdn * dev_values.values[i];
						out_row[i] = (upper_band - lower_band) / middle_band;
					}
				}
			}
		} else {
			for &(idx, devup, devdn) in &indices {
				let start = idx * cols;
				let end = start + cols;
				let out_row = &mut out[start..end];
				for i in (first + period - 1)..cols {
					let middle_band = middle[i];
					let upper_band = middle_band + devup * dev_values.values[i];
					let lower_band = middle_band - devdn * dev_values.values[i];
					out_row[i] = (upper_band - lower_band) / middle_band;
				}
			}
		}
	}

	Ok(combos)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;
	use crate::utilities::enums::Kernel;
	use paste::paste;

	fn check_bbw_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let partial_params = BollingerBandsWidthParams {
			period: Some(22),
			devup: Some(2.2),
			devdn: None,
			matype: Some("ema".to_string()),
			devtype: None,
		};
		let input = BollingerBandsWidthInput::from_candles(&candles, "hl2", partial_params);
		let output = bollinger_bands_width_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_bbw_default(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = BollingerBandsWidthInput::with_default_candles(&candles);
		let output = bollinger_bands_width_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_bbw_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data = [10.0, 20.0, 30.0];
		let params = BollingerBandsWidthParams {
			period: Some(0),
			..Default::default()
		};
		let input = BollingerBandsWidthInput::from_slice(&data, params);
		let result = bollinger_bands_width_with_kernel(&input, kernel);
		assert!(result.is_err(), "[{}] Expected error for zero period", test_name);
		Ok(())
	}

	fn check_bbw_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data = [10.0, 20.0, 30.0];
		let params = BollingerBandsWidthParams {
			period: Some(10),
			..Default::default()
		};
		let input = BollingerBandsWidthInput::from_slice(&data, params);
		let result = bollinger_bands_width_with_kernel(&input, kernel);
		assert!(
			result.is_err(),
			"[{}] Expected error for period > data.len()",
			test_name
		);
		Ok(())
	}

	fn check_bbw_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data = [42.0];
		let input = BollingerBandsWidthInput::from_slice(&data, BollingerBandsWidthParams::default());
		let result = bollinger_bands_width_with_kernel(&input, kernel);
		assert!(result.is_err(), "[{}] Expected error for small data", test_name);
		Ok(())
	}

	fn check_bbw_nan_check(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = BollingerBandsWidthInput::with_default_candles(&candles);
		let result = bollinger_bands_width_with_kernel(&input, kernel)?;
		let check_index = 240;
		if result.values.len() > check_index {
			for i in check_index..result.values.len() {
				// at least some values after check_index should not be NaN
				if !result.values[i].is_nan() {
					return Ok(());
				}
			}
			panic!("All BBWidth values from index {} onward are NaN.", check_index);
		}
		Ok(())
	}

	// Batch grid test: only parity, doesn't check numerical values
	fn check_batch_default_row(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let output = BollingerBandsWidthBatchBuilder::new()
			.kernel(kernel)
			.apply_candles(&c, "close")?;
		let def = BollingerBandsWidthParams::default();
		let row = output.values_for(&def).expect("default row missing");
		assert_eq!(row.len(), c.close.len());
		Ok(())
	}

	// Check for poison values in single output - only runs in debug mode
	#[cfg(debug_assertions)]
	fn check_bollinger_bands_width_no_poison(
		test_name: &str,
		kernel: Kernel,
	) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		// Test multiple parameter combinations to increase coverage
		let test_params = vec![
			// Default parameters
			BollingerBandsWidthParams::default(),
			// Small period
			BollingerBandsWidthParams {
				period: Some(5),
				devup: Some(1.0),
				devdn: Some(1.0),
				matype: Some("sma".to_string()),
				devtype: Some(0),
			},
			// Large period
			BollingerBandsWidthParams {
				period: Some(50),
				devup: Some(3.0),
				devdn: Some(3.0),
				matype: Some("ema".to_string()),
				devtype: Some(1),
			},
			// Asymmetric deviations
			BollingerBandsWidthParams {
				period: Some(15),
				devup: Some(2.5),
				devdn: Some(1.5),
				matype: Some("wma".to_string()),
				devtype: Some(2),
			},
			// Edge case parameters
			BollingerBandsWidthParams {
				period: Some(2),
				devup: Some(0.5),
				devdn: Some(0.5),
				matype: Some("sma".to_string()),
				devtype: Some(0),
			},
		];

		// Test with different sources too
		let sources = vec!["close", "hl2", "hlc3", "ohlc4"];

		for params in test_params {
			for &source in &sources {
				let input = BollingerBandsWidthInput::from_candles(&candles, source, params.clone());
				let output = bollinger_bands_width_with_kernel(&input, kernel)?;

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
                            "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} with params: period={}, devup={}, devdn={}, source={}",
                            test_name, val, bits, i, 
                            params.period.unwrap_or(20),
                            params.devup.unwrap_or(2.0),
                            params.devdn.unwrap_or(2.0),
                            source
                        );
					}

					// Check for init_matrix_prefixes poison (0x22222222_22222222)
					if bits == 0x22222222_22222222 {
						panic!(
                            "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} with params: period={}, devup={}, devdn={}, source={}",
                            test_name, val, bits, i,
                            params.period.unwrap_or(20),
                            params.devup.unwrap_or(2.0),
                            params.devdn.unwrap_or(2.0),
                            source
                        );
					}

					// Check for make_uninit_matrix poison (0x33333333_33333333)
					if bits == 0x33333333_33333333 {
						panic!(
                            "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} with params: period={}, devup={}, devdn={}, source={}",
                            test_name, val, bits, i,
                            params.period.unwrap_or(20),
                            params.devup.unwrap_or(2.0),
                            params.devdn.unwrap_or(2.0),
                            source
                        );
					}
				}
			}
		}

		Ok(())
	}

	// Release mode stub - does nothing
	#[cfg(not(debug_assertions))]
	fn check_bollinger_bands_width_no_poison(
		_test_name: &str,
		_kernel: Kernel,
	) -> Result<(), Box<dyn std::error::Error>> {
		Ok(())
	}

	// Check for poison values in batch output - only runs in debug mode
	#[cfg(debug_assertions)]
	fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		// Test multiple batch configurations with diverse parameter ranges
		let batch_configs = vec![
			// Wide range of periods with standard deviations
			(2, 50, 5, 1.0, 3.0, 0.5, 1.0, 3.0, 0.5),
			// Small periods with varying deviations
			(5, 15, 2, 0.5, 3.5, 0.25, 0.5, 3.5, 0.25),
			// Large periods with small deviation range
			(40, 100, 10, 1.5, 2.5, 0.1, 1.5, 2.5, 0.1),
			// Asymmetric deviations
			(10, 30, 5, 1.0, 4.0, 1.0, 0.5, 2.0, 0.5),
			// Edge case: very small periods
			(2, 5, 1, 0.1, 5.0, 0.5, 0.1, 5.0, 0.5),
		];

		let sources = vec!["close", "hl2", "ohlc4"];

		for (
			period_start,
			period_end,
			period_step,
			devup_start,
			devup_end,
			devup_step,
			devdn_start,
			devdn_end,
			devdn_step,
		) in batch_configs
		{
			for &source in &sources {
				let output = BollingerBandsWidthBatchBuilder::new()
					.kernel(kernel)
					.period_range(period_start, period_end, period_step)
					.devup_range(devup_start, devup_end, devup_step)
					.devdn_range(devdn_start, devdn_end, devdn_step)
					.apply_candles(&c, source)?;

				// Check every value in the entire batch matrix for poison patterns
				for (idx, &val) in output.values.iter().enumerate() {
					// Skip NaN values as they're expected in warmup periods
					if val.is_nan() {
						continue;
					}

					let bits = val.to_bits();
					let row = idx / output.cols;
					let col = idx % output.cols;

					// Get the parameters for this row
					let params = &output.combos[row];

					// Check for alloc_with_nan_prefix poison (0x11111111_11111111)
					if bits == 0x11111111_11111111 {
						panic!(
                            "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with params: period={}, devup={}, devdn={}, source={}",
                            test, val, bits, row, col, idx,
                            params.period.unwrap_or(20),
                            params.devup.unwrap_or(2.0),
                            params.devdn.unwrap_or(2.0),
                            source
                        );
					}

					// Check for init_matrix_prefixes poison (0x22222222_22222222)
					if bits == 0x22222222_22222222 {
						panic!(
                            "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {}) with params: period={}, devup={}, devdn={}, source={}",
                            test, val, bits, row, col, idx,
                            params.period.unwrap_or(20),
                            params.devup.unwrap_or(2.0),
                            params.devdn.unwrap_or(2.0),
                            source
                        );
					}

					// Check for make_uninit_matrix poison (0x33333333_33333333)
					if bits == 0x33333333_33333333 {
						panic!(
                            "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with params: period={}, devup={}, devdn={}, source={}",
                            test, val, bits, row, col, idx,
                            params.period.unwrap_or(20),
                            params.devup.unwrap_or(2.0),
                            params.devdn.unwrap_or(2.0),
                            source
                        );
					}
				}
			}
		}

		Ok(())
	}

	// Release mode stub - does nothing
	#[cfg(not(debug_assertions))]
	fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		Ok(())
	}

	macro_rules! generate_all_bbw_tests {
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

	generate_all_bbw_tests!(
		check_bbw_partial_params,
		check_bbw_default,
		check_bbw_zero_period,
		check_bbw_period_exceeds_length,
		check_bbw_very_small_dataset,
		check_bbw_nan_check,
		check_bollinger_bands_width_no_poison
	);

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
	gen_batch_tests!(check_batch_no_poison);
}

#[cfg(feature = "python")]
#[pyfunction(name = "bollinger_bands_width")]
#[pyo3(signature = (data, period, devup, devdn, matype=None, devtype=None, kernel=None))]
pub fn bollinger_bands_width_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	period: usize,
	devup: f64,
	devdn: f64,
	matype: Option<&str>,
	devtype: Option<usize>,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
	use crate::utilities::kernel_validation::validate_kernel;
	use numpy::{PyArray1, PyArrayMethods};
	use pyo3::exceptions::PyValueError;

	let slice_in = data.as_slice()?; // zero-copy, read-only view

	// Use kernel validation for safety
	let kern = validate_kernel(kernel, false)?;

	// Build input struct
	let params = BollingerBandsWidthParams {
		period: Some(period),
		devup: Some(devup),
		devdn: Some(devdn),
		matype: matype.map(|s| s.to_string()),
		devtype: devtype,
	};
	let bbw_in = BollingerBandsWidthInput::from_slice(slice_in, params);

	// SAFETY: PyArray1::new() creates uninitialized memory, not zero-initialized.
	// We MUST write to ALL elements before returning these arrays to Python.
	// Python/NumPy's memory model requires that all array elements are initialized.
	// Returning uninitialized memory to Python is undefined behavior and can cause
	// crashes or expose sensitive data from previous memory contents.
	let out_arr = unsafe { PyArray1::<f64>::new(py, [slice_in.len()], false) };
	let slice_out = unsafe { out_arr.as_slice_mut()? };

	// Heavy lifting without the GIL
	py.allow_threads(|| -> Result<(), BollingerBandsWidthError> {
		// First, we need to calculate the warmup period and initialize with NaN
		let first_valid_idx = slice_in.iter().position(|&x| !x.is_nan()).unwrap_or(0);
		let warmup_period = first_valid_idx + period - 1;

		// Initialize warmup period with NaN
		for i in 0..warmup_period.min(slice_out.len()) {
			slice_out[i] = f64::NAN;
		}

		// SAFETY: Compute directly into the pre-allocated NumPy buffer.
		// This avoids an extra allocation and copy.
		bollinger_bands_width_into(slice_in, &bbw_in, slice_out, kern)?;
		Ok(())
	})
	.map_err(|e| PyValueError::new_err(e.to_string()))?;

	Ok(out_arr)
}

#[cfg(feature = "python")]
#[pyclass(name = "BollingerBandsWidthStream")]
pub struct BollingerBandsWidthStreamPy {
	period: usize,
	devup: f64,
	devdn: f64,
	matype: String,
	devtype: usize,
	buffer: Vec<f64>,
}

#[cfg(feature = "python")]
#[pymethods]
impl BollingerBandsWidthStreamPy {
	#[new]
	fn new(period: usize, devup: f64, devdn: f64, matype: Option<&str>, devtype: Option<usize>) -> PyResult<Self> {
		use pyo3::exceptions::PyValueError;

		if period == 0 {
			return Err(PyValueError::new_err("Period must be greater than 0"));
		}

		Ok(BollingerBandsWidthStreamPy {
			period,
			devup,
			devdn,
			matype: matype.unwrap_or("sma").to_string(),
			devtype: devtype.unwrap_or(0),
			buffer: Vec::with_capacity(period),
		})
	}

	/// Updates the stream with a new value and returns the calculated BBW value.
	/// Returns `None` if the buffer is not yet full.
	fn update(&mut self, value: f64) -> Option<f64> {
		use pyo3::exceptions::PyValueError;

		self.buffer.push(value);

		if self.buffer.len() > self.period {
			self.buffer.remove(0);
		}

		if self.buffer.len() < self.period {
			return None;
		}

		// Calculate BBW using the current buffer
		let params = BollingerBandsWidthParams {
			period: Some(self.period),
			devup: Some(self.devup),
			devdn: Some(self.devdn),
			matype: Some(self.matype.clone()),
			devtype: Some(self.devtype),
		};
		let input = BollingerBandsWidthInput::from_slice(&self.buffer, params);

		match bollinger_bands_width(&input) {
			Ok(result) => result.values.last().copied(),
			Err(_) => None,
		}
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "bollinger_bands_width_batch")]
#[pyo3(signature = (data, period_range, devup_range, devdn_range, matype=None, devtype=None, kernel=None))]
pub fn bollinger_bands_width_batch_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	period_range: (usize, usize, usize),
	devup_range: (f64, f64, f64),
	devdn_range: (f64, f64, f64),
	matype: Option<&str>,
	devtype: Option<usize>,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
	use crate::utilities::kernel_validation::validate_kernel;
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::exceptions::PyValueError;
	use pyo3::types::PyDict;

	let slice_in = data.as_slice()?;

	let sweep = BollingerBandsWidthBatchRange {
		period: period_range,
		devup: devup_range,
		devdn: devdn_range,
	};

	// Use kernel validation for safety
	let kern = validate_kernel(kernel, true)?;

	// Expand grid to know dimensions
	let combos = expand_grid(&sweep);
	let rows = combos.len();
	let cols = slice_in.len();

	// SAFETY: Pre-allocate NumPy array. The zero-copy functions will handle NaN initialization
	// for the warmup periods.
	let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let slice_out = unsafe { out_arr.as_slice_mut()? };

	// Heavy work without the GIL
	let (combos, matype_used, devtype_used) = py
		.allow_threads(
			|| -> Result<(Vec<BollingerBandsWidthParams>, String, usize), BollingerBandsWidthError> {
				let matype_str = matype.unwrap_or("sma").to_string();
				let devtype_val = devtype.unwrap_or(0);

				// Resolve kernel
				let kernel = match kern {
					Kernel::Auto => detect_best_batch_kernel(),
					k => k,
				};

				// Calculate warmup periods for each row and initialize NaN prefixes
				let first_valid = slice_in.iter().position(|&x| !x.is_nan()).unwrap_or(0);
				let warmup_periods: Vec<usize> = (0..rows)
					.map(|i| {
						let period = if period_range.2 == 0 {
							period_range.0
						} else {
							period_range.0 + i * period_range.2
						};
						first_valid + period - 1
					})
					.collect();

				// Initialize NaN values for warmup periods in each row
				for (row_idx, &warmup) in warmup_periods.iter().enumerate() {
					let row_start = row_idx * cols;
					for col_idx in 0..warmup.min(cols) {
						slice_out[row_start + col_idx] = f64::NAN;
					}
				}

				// SAFETY: Compute directly into the pre-allocated NumPy buffer.
				// This avoids an extra allocation and copy.
				let mut combos = bollinger_bands_width_batch_inner_into(slice_in, &sweep, kernel, true, slice_out)?;

				// Update combos with matype and devtype
				for combo in &mut combos {
					combo.matype = Some(matype_str.clone());
					combo.devtype = Some(devtype_val);
				}

				Ok((combos, matype_str, devtype_val))
			},
		)
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	// Build result dict
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
	dict.set_item(
		"devups",
		combos
			.iter()
			.map(|p| p.devup.unwrap())
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;
	dict.set_item(
		"devdns",
		combos
			.iter()
			.map(|p| p.devdn.unwrap())
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;
	dict.set_item("matype", matype_used)?;
	dict.set_item("devtype", devtype_used)?;

	Ok(dict)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn bollinger_bands_width_js(
	data: &[f64],
	period: usize,
	devup: f64,
	devdn: f64,
	matype: Option<String>,
	devtype: Option<usize>,
) -> Result<Vec<f64>, JsValue> {
	let params = BollingerBandsWidthParams {
		period: Some(period),
		devup: Some(devup),
		devdn: Some(devdn),
		matype: matype.or_else(|| Some("sma".to_string())),
		devtype: devtype.or(Some(0)),
	};
	let input = BollingerBandsWidthInput::from_slice(data, params);

	bollinger_bands_width_with_kernel(&input, Kernel::Scalar)
		.map(|o| o.values)
		.map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn bollinger_bands_width_batch_js(
	data: &[f64],
	period_start: usize,
	period_end: usize,
	period_step: usize,
	devup_start: f64,
	devup_end: f64,
	devup_step: f64,
	devdn_start: f64,
	devdn_end: f64,
	devdn_step: f64,
) -> Result<Vec<f64>, JsValue> {
	let sweep = BollingerBandsWidthBatchRange {
		period: (period_start, period_end, period_step),
		devup: (devup_start, devup_end, devup_step),
		devdn: (devdn_start, devdn_end, devdn_step),
	};

	bollinger_bands_width_batch_inner(data, &sweep, Kernel::Scalar, false)
		.map(|output| output.values)
		.map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn bollinger_bands_width_batch_metadata_js(
	period_start: usize,
	period_end: usize,
	period_step: usize,
	devup_start: f64,
	devup_end: f64,
	devup_step: f64,
	devdn_start: f64,
	devdn_end: f64,
	devdn_step: f64,
) -> Result<Vec<f64>, JsValue> {
	let sweep = BollingerBandsWidthBatchRange {
		period: (period_start, period_end, period_step),
		devup: (devup_start, devup_end, devup_step),
		devdn: (devdn_start, devdn_end, devdn_step),
	};

	let combos = expand_grid(&sweep);
	let mut metadata = Vec::with_capacity(combos.len() * 3);

	for combo in combos {
		metadata.push(combo.period.unwrap() as f64);
		metadata.push(combo.devup.unwrap());
		metadata.push(combo.devdn.unwrap());
	}

	Ok(metadata)
}

// New ergonomic WASM API
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct BollingerBandsWidthBatchConfig {
	pub period_range: (usize, usize, usize),
	pub devup_range: (f64, f64, f64),
	pub devdn_range: (f64, f64, f64),
	pub matype: Option<String>,
	pub devtype: Option<usize>,
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct BollingerBandsWidthBatchJsOutput {
	pub values: Vec<f64>,
	pub combos: Vec<BollingerBandsWidthParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = bollinger_bands_width_batch)]
pub fn bollinger_bands_width_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	// Deserialize the configuration object from JavaScript
	let config: BollingerBandsWidthBatchConfig =
		serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

	let sweep = BollingerBandsWidthBatchRange {
		period: config.period_range,
		devup: config.devup_range,
		devdn: config.devdn_range,
	};

	// Run the existing core logic
	let mut output = bollinger_bands_width_batch_inner(data, &sweep, Kernel::Scalar, false)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;

	// Update combos with matype and devtype
	for combo in &mut output.combos {
		combo.matype = config.matype.clone().or_else(|| Some("sma".to_string()));
		combo.devtype = config.devtype.or(Some(0));
	}

	// Create the structured output
	let js_output = BollingerBandsWidthBatchJsOutput {
		values: output.values,
		combos: output.combos,
		rows: output.rows,
		cols: output.cols,
	};

	// Serialize the output struct into a JavaScript object
	serde_wasm_bindgen::to_value(&js_output).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}
