//! # Variable Length Moving Average (VLMA)
//!
//! Adaptive moving average whose period varies between `min_period` and `max_period` depending on deviation from a long-term average.
//! Supports batch/grid parameter sweeps, kernel selection, streaming updates, and AVX2/AVX512 stubs for future SIMD support.
//!
//! ## Parameters
//! - **min_period**: Minimum period (default: 5)
//! - **max_period**: Maximum period (default: 50)
//! - **matype**: Moving average type (default: "sma")
//! - **devtype**: Deviation type (0=std, 1=mad, 2=median, default: 0)
//!
//! ## Returns
//! - **Ok(VlmaOutput)** on success, containing a Vec<f64> of indicator values
//! - **Err(VlmaError)** on failure
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

use crate::indicators::deviation::{deviation, DevInput, DevParams};
use crate::indicators::moving_averages::ma::{ma, MaData};
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
	alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

impl<'a> AsRef<[f64]> for VlmaInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			VlmaData::Slice(sl) => sl,
			VlmaData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

#[derive(Debug, Clone)]
pub enum VlmaData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct VlmaOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct VlmaParams {
	pub min_period: Option<usize>,
	pub max_period: Option<usize>,
	pub matype: Option<String>,
	pub devtype: Option<usize>,
}

impl Default for VlmaParams {
	fn default() -> Self {
		Self {
			min_period: Some(5),
			max_period: Some(50),
			matype: Some("sma".to_string()),
			devtype: Some(0),
		}
	}
}

#[derive(Debug, Clone)]
pub struct VlmaInput<'a> {
	pub data: VlmaData<'a>,
	pub params: VlmaParams,
}

impl<'a> VlmaInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: VlmaParams) -> Self {
		Self {
			data: VlmaData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slice(sl: &'a [f64], p: VlmaParams) -> Self {
		Self {
			data: VlmaData::Slice(sl),
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", VlmaParams::default())
	}
	#[inline]
	pub fn get_min_period(&self) -> usize {
		self.params.min_period.unwrap_or(5)
	}
	#[inline]
	pub fn get_max_period(&self) -> usize {
		self.params.max_period.unwrap_or(50)
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

#[derive(Clone, Debug)]
pub struct VlmaBuilder {
	min_period: Option<usize>,
	max_period: Option<usize>,
	matype: Option<String>,
	devtype: Option<usize>,
	kernel: Kernel,
}

impl Default for VlmaBuilder {
	fn default() -> Self {
		Self {
			min_period: None,
			max_period: None,
			matype: None,
			devtype: None,
			kernel: Kernel::Auto,
		}
	}
}

impl VlmaBuilder {
	#[inline(always)]
	pub fn new() -> Self {
		Self::default()
	}
	#[inline(always)]
	pub fn min_period(mut self, n: usize) -> Self {
		self.min_period = Some(n);
		self
	}
	#[inline(always)]
	pub fn max_period(mut self, n: usize) -> Self {
		self.max_period = Some(n);
		self
	}
	#[inline(always)]
	pub fn matype<S: Into<String>>(mut self, t: S) -> Self {
		self.matype = Some(t.into());
		self
	}
	#[inline(always)]
	pub fn devtype(mut self, d: usize) -> Self {
		self.devtype = Some(d);
		self
	}
	#[inline(always)]
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline(always)]
	pub fn apply(self, c: &Candles) -> Result<VlmaOutput, VlmaError> {
		let p = VlmaParams {
			min_period: self.min_period,
			max_period: self.max_period,
			matype: self.matype,
			devtype: self.devtype,
		};
		let i = VlmaInput::from_candles(c, "close", p);
		vlma_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slice(self, d: &[f64]) -> Result<VlmaOutput, VlmaError> {
		let p = VlmaParams {
			min_period: self.min_period,
			max_period: self.max_period,
			matype: self.matype,
			devtype: self.devtype,
		};
		let i = VlmaInput::from_slice(d, p);
		vlma_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<VlmaStream, VlmaError> {
		let p = VlmaParams {
			min_period: self.min_period,
			max_period: self.max_period,
			matype: self.matype,
			devtype: self.devtype,
		};
		VlmaStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum VlmaError {
	#[error("vlma: Empty data provided.")]
	EmptyData,
	#[error("vlma: min_period={min_period} is greater than max_period={max_period}.")]
	InvalidPeriodRange { min_period: usize, max_period: usize },
	#[error("vlma: Invalid period: min_period={min_period}, max_period={max_period}, data length={data_len}.")]
	InvalidPeriod {
		min_period: usize,
		max_period: usize,
		data_len: usize,
	},
	#[error("vlma: All values are NaN.")]
	AllValuesNaN,
	#[error("vlma: Not enough valid data: needed={needed}, valid={valid}.")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("vlma: Error in MA calculation: {0}")]
	MaError(String),
	#[error("vlma: Error in Deviation calculation: {0}")]
	DevError(String),
}

#[inline]
pub fn vlma(input: &VlmaInput) -> Result<VlmaOutput, VlmaError> {
	vlma_with_kernel(input, Kernel::Auto)
}

pub fn vlma_with_kernel(input: &VlmaInput, kernel: Kernel) -> Result<VlmaOutput, VlmaError> {
	let data: &[f64] = input.as_ref();

	if data.is_empty() {
		return Err(VlmaError::EmptyData);
	}

	let min_period = input.get_min_period();
	let max_period = input.get_max_period();
	if min_period > max_period {
		return Err(VlmaError::InvalidPeriodRange { min_period, max_period });
	}

	if max_period == 0 || max_period > data.len() {
		return Err(VlmaError::InvalidPeriod {
			min_period,
			max_period,
			data_len: data.len(),
		});
	}

	let first = data.iter().position(|&x| !x.is_nan()).ok_or(VlmaError::AllValuesNaN)?;

	if (data.len() - first) < max_period {
		return Err(VlmaError::NotEnoughValidData {
			needed: max_period,
			valid: data.len() - first,
		});
	}

	let matype = input.get_matype();
	let devtype = input.get_devtype();

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		k => k,
	};

	let mut out = alloc_with_nan_prefix(data.len(), first + max_period - 1);
	
	vlma_compute_into(data, min_period, max_period, &matype, devtype, first, chosen, &mut out)?;
	
	Ok(VlmaOutput { values: out })
}

#[inline(always)]
fn vlma_prepare<'a>(
	input: &'a VlmaInput,
	kernel: Kernel,
) -> Result<(&'a [f64], usize, usize, String, usize, usize, Kernel), VlmaError> {
	let data: &[f64] = input.as_ref();

	if data.is_empty() {
		return Err(VlmaError::EmptyData);
	}

	let min_period = input.get_min_period();
	let max_period = input.get_max_period();
	if min_period > max_period {
		return Err(VlmaError::InvalidPeriodRange { min_period, max_period });
	}

	if max_period == 0 || max_period > data.len() {
		return Err(VlmaError::InvalidPeriod {
			min_period,
			max_period,
			data_len: data.len(),
		});
	}

	let first = data.iter().position(|&x| !x.is_nan()).ok_or(VlmaError::AllValuesNaN)?;

	if (data.len() - first) < max_period {
		return Err(VlmaError::NotEnoughValidData {
			needed: max_period,
			valid: data.len() - first,
		});
	}

	let matype = input.get_matype();
	let devtype = input.get_devtype();

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		k => k,
	};

	Ok((data, min_period, max_period, matype, devtype, first, chosen))
}

#[inline(always)]
fn vlma_compute_into(
	data: &[f64],
	min_period: usize,
	max_period: usize,
	matype: &str,
	devtype: usize,
	first: usize,
	kernel: Kernel,
	out: &mut [f64],
) -> Result<(), VlmaError> {
	unsafe {
		match kernel {
			Kernel::Scalar | Kernel::ScalarBatch => {
				vlma_scalar_into(data, min_period, max_period, matype, devtype, first, out)?;
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => {
				vlma_avx2_into(data, min_period, max_period, matype, devtype, first, out)?;
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => {
				vlma_avx512_into(data, min_period, max_period, matype, devtype, first, out)?;
			}
			_ => unreachable!(),
		}
	}
	Ok(())
}

#[inline(always)]
unsafe fn vlma_scalar(
	data: &[f64],
	min_period: usize,
	max_period: usize,
	matype: String,
	devtype: usize,
	first_valid: usize,
) -> Result<VlmaOutput, VlmaError> {
	let warmup_period = first_valid + max_period - 1;
	let mut out = alloc_with_nan_prefix(data.len(), warmup_period);
	vlma_scalar_into(data, min_period, max_period, &matype, devtype, first_valid, &mut out)?;
	Ok(VlmaOutput { values: out })
}

#[inline(always)]
unsafe fn vlma_scalar_into(
	data: &[f64],
	min_period: usize,
	max_period: usize,
	matype: &str,
	devtype: usize,
	first_valid: usize,
	out: &mut [f64],
) -> Result<(), VlmaError> {
	let mean = ma(matype, MaData::Slice(data), max_period).map_err(|e| VlmaError::MaError(e.to_string()))?;

	let dev_params = DevParams {
		period: Some(max_period),
		devtype: Some(devtype),
	};
	let dev_input = DevInput::from_slice(data, dev_params);
	let dev = deviation(&dev_input).map_err(|e| VlmaError::DevError(e.to_string()))?;

	let warmup_period = first_valid + max_period - 1;
	let mut a = alloc_with_nan_prefix(data.len(), warmup_period);
	let mut b = alloc_with_nan_prefix(data.len(), warmup_period);
	let mut c = alloc_with_nan_prefix(data.len(), warmup_period);
	let mut d = alloc_with_nan_prefix(data.len(), warmup_period);

	for i in 0..data.len() {
		if !mean[i].is_nan() && !dev[i].is_nan() {
			a[i] = mean[i] - 1.75 * dev[i];
			b[i] = mean[i] - 0.25 * dev[i];
			c[i] = mean[i] + 0.25 * dev[i];
			d[i] = mean[i] + 1.75 * dev[i];
		}
	}

	let mut periods = vec![0.0; data.len()];  // OK: This tracks the adaptive period and is not output

	let mut last_val = data[first_valid];
	out[first_valid] = last_val;
	periods[first_valid] = max_period as f64;

	for i in (first_valid + 1)..data.len() {
		if data[i].is_nan() {
			out[i] = f64::NAN;
			continue;
		}
		let prev_period = if periods[i - 1] == 0.0 {
			max_period as f64
		} else {
			periods[i - 1]
		};

		let mut new_period = if !a[i].is_nan() && !b[i].is_nan() && !c[i].is_nan() && !d[i].is_nan() {
			if data[i] < a[i] || data[i] > d[i] {
				prev_period - 1.0
			} else if data[i] >= b[i] && data[i] <= c[i] {
				prev_period + 1.0
			} else {
				prev_period
			}
		} else {
			prev_period
		};

		if new_period < min_period as f64 {
			new_period = min_period as f64;
		} else if new_period > max_period as f64 {
			new_period = max_period as f64;
		}

		let sc = 2.0 / (new_period + 1.0);
		let new_val = data[i] * sc + (1.0 - sc) * last_val;
		periods[i] = new_period;
		last_val = new_val;

		if i >= first_valid + max_period - 1 {
			out[i] = new_val;
		}
	}

	Ok(())
}

#[inline(always)]
unsafe fn vlma_row_scalar(
	data: &[f64],
	min_period: usize,
	max_period: usize,
	matype: &str,
	devtype: usize,
	first_valid: usize,
	out: &mut [f64],
) -> Result<(), VlmaError> {
	vlma_scalar_into(data, min_period, max_period, matype, devtype, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn vlma_avx2(
	data: &[f64],
	min_period: usize,
	max_period: usize,
	matype: String,
	devtype: usize,
	first_valid: usize,
) -> Result<VlmaOutput, VlmaError> {
	let warmup_period = first_valid + max_period - 1;
	let mut out = alloc_with_nan_prefix(data.len(), warmup_period);
	vlma_avx2_into(data, min_period, max_period, &matype, devtype, first_valid, &mut out)?;
	Ok(VlmaOutput { values: out })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn vlma_avx2_into(
	data: &[f64],
	min_period: usize,
	max_period: usize,
	matype: &str,
	devtype: usize,
	first_valid: usize,
	out: &mut [f64],
) -> Result<(), VlmaError> {
	// For now, delegate to scalar implementation
	vlma_scalar_into(data, min_period, max_period, matype, devtype, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn vlma_row_avx2(
	data: &[f64],
	min_period: usize,
	max_period: usize,
	matype: &str,
	devtype: usize,
	first_valid: usize,
	out: &mut [f64],
) -> Result<(), VlmaError> {
	vlma_avx2_into(data, min_period, max_period, matype, devtype, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn vlma_avx512(
	data: &[f64],
	min_period: usize,
	max_period: usize,
	matype: String,
	devtype: usize,
	first_valid: usize,
) -> Result<VlmaOutput, VlmaError> {
	let warmup_period = first_valid + max_period - 1;
	let mut out = alloc_with_nan_prefix(data.len(), warmup_period);
	vlma_avx512_into(data, min_period, max_period, &matype, devtype, first_valid, &mut out)?;
	Ok(VlmaOutput { values: out })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn vlma_avx512_into(
	data: &[f64],
	min_period: usize,
	max_period: usize,
	matype: &str,
	devtype: usize,
	first_valid: usize,
	out: &mut [f64],
) -> Result<(), VlmaError> {
	if max_period <= 32 {
		vlma_avx512_short_into(data, min_period, max_period, matype, devtype, first_valid, out)
	} else {
		vlma_avx512_long_into(data, min_period, max_period, matype, devtype, first_valid, out)
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn vlma_row_avx512(
	data: &[f64],
	min_period: usize,
	max_period: usize,
	matype: &str,
	devtype: usize,
	first_valid: usize,
	out: &mut [f64],
) -> Result<(), VlmaError> {
	vlma_avx512_into(data, min_period, max_period, matype, devtype, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn vlma_avx512_short(
	data: &[f64],
	min_period: usize,
	max_period: usize,
	matype: String,
	devtype: usize,
	first_valid: usize,
) -> Result<VlmaOutput, VlmaError> {
	let warmup_period = first_valid + max_period - 1;
	let mut out = alloc_with_nan_prefix(data.len(), warmup_period);
	vlma_avx512_short_into(data, min_period, max_period, &matype, devtype, first_valid, &mut out)?;
	Ok(VlmaOutput { values: out })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn vlma_avx512_short_into(
	data: &[f64],
	min_period: usize,
	max_period: usize,
	matype: &str,
	devtype: usize,
	first_valid: usize,
	out: &mut [f64],
) -> Result<(), VlmaError> {
	// For now, delegate to scalar implementation
	vlma_scalar_into(data, min_period, max_period, matype, devtype, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn vlma_avx512_long(
	data: &[f64],
	min_period: usize,
	max_period: usize,
	matype: String,
	devtype: usize,
	first_valid: usize,
) -> Result<VlmaOutput, VlmaError> {
	let warmup_period = first_valid + max_period - 1;
	let mut out = alloc_with_nan_prefix(data.len(), warmup_period);
	vlma_avx512_long_into(data, min_period, max_period, &matype, devtype, first_valid, &mut out)?;
	Ok(VlmaOutput { values: out })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn vlma_avx512_long_into(
	data: &[f64],
	min_period: usize,
	max_period: usize,
	matype: &str,
	devtype: usize,
	first_valid: usize,
	out: &mut [f64],
) -> Result<(), VlmaError> {
	// For now, delegate to scalar implementation
	vlma_scalar_into(data, min_period, max_period, matype, devtype, first_valid, out)
}

#[derive(Debug, Clone)]
pub struct VlmaStream {
	min_period: usize,
	max_period: usize,
	matype: String,
	devtype: usize,
	buffer: Vec<f64>,
	head: usize,
	filled: bool,
	period: f64,
	last_val: f64,
}

impl VlmaStream {
	pub fn try_new(params: VlmaParams) -> Result<Self, VlmaError> {
		let min_period = params.min_period.unwrap_or(5);
		let max_period = params.max_period.unwrap_or(50);
		let matype = params.matype.unwrap_or_else(|| "sma".to_string());
		let devtype = params.devtype.unwrap_or(0);

		if min_period > max_period {
			return Err(VlmaError::InvalidPeriodRange { min_period, max_period });
		}
		if max_period == 0 {
			return Err(VlmaError::InvalidPeriod {
				min_period,
				max_period,
				data_len: 0,
			});
		}

		Ok(Self {
			min_period,
			max_period,
			matype,
			devtype,
			buffer: vec![f64::NAN; max_period],
			head: 0,
			filled: false,
			period: max_period as f64,
			last_val: f64::NAN,
		})
	}

	#[inline(always)]
	pub fn update(&mut self, value: f64) -> Option<f64> {
		self.buffer[self.head] = value;
		self.head = (self.head + 1) % self.max_period;
		if !self.filled && self.head == 0 {
			self.filled = true;
		}
		if !self.filled {
			if self.last_val.is_nan() {
				self.last_val = value;
				return Some(value);
			}
			let sc = 2.0 / (self.period + 1.0);
			let new_val = value * sc + (1.0 - sc) * self.last_val;
			self.last_val = new_val;
			return None;
		}

		let mut window: Vec<f64> = Vec::with_capacity(self.max_period);
		for i in 0..self.max_period {
			let idx = (self.head + i) % self.max_period;
			let v = self.buffer[idx];
			if !v.is_nan() {
				window.push(v);
			}
		}
		if window.len() < self.max_period {
			return None;
		}

		let mean = ma(&self.matype, MaData::Slice(&window), self.max_period)
			.ok()?
			.last()?
			.clone();
		let dev_params = DevParams {
			period: Some(self.max_period),
			devtype: Some(self.devtype),
		};
		let dev = deviation(&DevInput::from_slice(&window, dev_params))
			.ok()?
			.last()?
			.clone();

		let a = mean - 1.75 * dev;
		let b = mean - 0.25 * dev;
		let c = mean + 0.25 * dev;
		let d = mean + 1.75 * dev;

		let prev_period = if self.period == 0.0 {
			self.max_period as f64
		} else {
			self.period
		};
		let mut new_period = if value < a || value > d {
			prev_period - 1.0
		} else if value >= b && value <= c {
			prev_period + 1.0
		} else {
			prev_period
		};

		if new_period < self.min_period as f64 {
			new_period = self.min_period as f64;
		} else if new_period > self.max_period as f64 {
			new_period = self.max_period as f64;
		}
		let sc = 2.0 / (new_period + 1.0);
		let new_val = if self.last_val.is_nan() {
			value
		} else {
			value * sc + (1.0 - sc) * self.last_val
		};
		self.period = new_period;
		self.last_val = new_val;
		Some(new_val)
	}
}

#[derive(Clone, Debug)]
pub struct VlmaBatchRange {
	pub min_period: (usize, usize, usize),
	pub max_period: (usize, usize, usize),
	pub matype: (String, String, String),
	pub devtype: (usize, usize, usize),
}

impl Default for VlmaBatchRange {
	fn default() -> Self {
		Self {
			min_period: (5, 5, 0),
			max_period: (50, 50, 0),
			matype: ("sma".to_string(), "sma".to_string(), "".to_string()),
			devtype: (0, 0, 0),
		}
	}
}

#[derive(Clone, Debug, Default)]
pub struct VlmaBatchBuilder {
	range: VlmaBatchRange,
	kernel: Kernel,
}

impl VlmaBatchBuilder {
	pub fn new() -> Self {
		Self::default()
	}
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline]
	pub fn min_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.min_period = (start, end, step);
		self
	}
	#[inline]
	pub fn max_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.max_period = (start, end, step);
		self
	}
	pub fn matype_static<S: Into<String>>(mut self, v: S) -> Self {
		let s = v.into();
		self.range.matype = (s.clone(), s, "".to_string());
		self
	}
	#[inline]
	pub fn devtype_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.devtype = (start, end, step);
		self
	}
	pub fn apply_slice(self, data: &[f64]) -> Result<VlmaBatchOutput, VlmaError> {
		vlma_batch_with_kernel(data, &self.range, self.kernel)
	}
	pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<VlmaBatchOutput, VlmaError> {
		VlmaBatchBuilder::new().kernel(k).apply_slice(data)
	}
	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<VlmaBatchOutput, VlmaError> {
		let slice = source_type(c, src);
		self.apply_slice(slice)
	}
	pub fn with_default_candles(c: &Candles) -> Result<VlmaBatchOutput, VlmaError> {
		VlmaBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
	}
}

#[derive(Clone, Debug)]
pub struct VlmaBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<VlmaParams>,
	pub rows: usize,
	pub cols: usize,
}

impl VlmaBatchOutput {
	pub fn row_for_params(&self, p: &VlmaParams) -> Option<usize> {
		self.combos.iter().position(|c| {
			c.min_period.unwrap_or(5) == p.min_period.unwrap_or(5)
				&& c.max_period.unwrap_or(50) == p.max_period.unwrap_or(50)
				&& c.matype.as_ref().unwrap_or(&"sma".to_string()) == p.matype.as_ref().unwrap_or(&"sma".to_string())
				&& c.devtype.unwrap_or(0) == p.devtype.unwrap_or(0)
		})
	}
	pub fn values_for(&self, p: &VlmaParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
	if step == 0 || start == end {
		return vec![start];
	}
	(start..=end).step_by(step).collect()
}
fn axis_string((start, end, _): (String, String, String)) -> Vec<String> {
	if start == end {
		vec![start]
	} else {
		vec![start, end]
	}
}
fn axis_usize_step((start, end, step): (usize, usize, usize)) -> Vec<usize> {
	if step == 0 || start == end {
		vec![start]
	} else {
		(start..=end).step_by(step).collect()
	}
}
fn axis_devtype((start, end, step): (usize, usize, usize)) -> Vec<usize> {
	if step == 0 || start == end {
		vec![start]
	} else {
		(start..=end).step_by(step).collect()
	}
}

fn expand_grid(r: &VlmaBatchRange) -> Vec<VlmaParams> {
	let min_periods = axis_usize(r.min_period);
	let max_periods = axis_usize(r.max_period);
	let matypes = axis_string(r.matype.clone());
	let devtypes = axis_devtype(r.devtype);
	let mut out = Vec::with_capacity(min_periods.len() * max_periods.len() * matypes.len() * devtypes.len());
	for &mn in &min_periods {
		for &mx in &max_periods {
			for mt in &matypes {
				for &dt in &devtypes {
					out.push(VlmaParams {
						min_period: Some(mn),
						max_period: Some(mx),
						matype: Some(mt.clone()),
						devtype: Some(dt),
					});
				}
			}
		}
	}
	out
}

#[inline(always)]
pub fn vlma_batch_with_kernel(data: &[f64], sweep: &VlmaBatchRange, k: Kernel) -> Result<VlmaBatchOutput, VlmaError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => {
			return Err(VlmaError::InvalidPeriod {
				min_period: 0,
				max_period: 0,
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
	vlma_batch_par_slice(data, sweep, simd)
}

#[inline(always)]
pub fn vlma_batch_slice(data: &[f64], sweep: &VlmaBatchRange, kern: Kernel) -> Result<VlmaBatchOutput, VlmaError> {
	vlma_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn vlma_batch_par_slice(data: &[f64], sweep: &VlmaBatchRange, kern: Kernel) -> Result<VlmaBatchOutput, VlmaError> {
	vlma_batch_inner(data, sweep, kern, true)
}

fn vlma_batch_inner(
	data: &[f64],
	sweep: &VlmaBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<VlmaBatchOutput, VlmaError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(VlmaError::InvalidPeriod {
			min_period: 0,
			max_period: 0,
			data_len: 0,
		});
	}
	let first = data.iter().position(|x| !x.is_nan()).ok_or(VlmaError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.max_period.unwrap()).max().unwrap();
	if data.len() - first < max_p {
		return Err(VlmaError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}
	let rows = combos.len();
	let cols = data.len();
	let mut uninit_values = make_uninit_matrix(rows, cols);
	
	// Initialize warmup periods for each row
	let warmup_periods: Vec<usize> = combos
		.iter()
		.map(|p| first + p.max_period.unwrap() - 1)
		.collect();
	init_matrix_prefixes(&mut uninit_values, cols, &warmup_periods);
	
	// Convert to initialized slice for computation
	let values_ptr = uninit_values.as_mut_ptr() as *mut f64;
	let values = unsafe { std::slice::from_raw_parts_mut(values_ptr, rows * cols) };
	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let min_period = combos[row].min_period.unwrap();
		let max_period = combos[row].max_period.unwrap();
		let matype = combos[row].matype.as_ref().unwrap();
		let devtype = combos[row].devtype.unwrap();
		match kern {
			Kernel::Scalar => {
				vlma_row_scalar(data, min_period, max_period, matype, devtype, first, out_row).unwrap();
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => {
				vlma_row_avx2(data, min_period, max_period, matype, devtype, first, out_row).unwrap();
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => {
				vlma_row_avx512(data, min_period, max_period, matype, devtype, first, out_row).unwrap();
			}
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
	
	// Convert initialized MaybeUninit<f64> back to Vec<f64>
	let values_vec = unsafe {
		let ptr = uninit_values.as_ptr() as *const f64;
		Vec::from_raw_parts(ptr as *mut f64, rows * cols, rows * cols)
	};
	std::mem::forget(uninit_values);
	
	Ok(VlmaBatchOutput {
		values: values_vec,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
pub fn vlma_batch_inner_into(
	data: &[f64],
	sweep: &VlmaBatchRange,
	kern: Kernel,
	parallel: bool,
	out: &mut [f64],
) -> Result<Vec<VlmaParams>, VlmaError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(VlmaError::InvalidPeriod {
			min_period: 0,
			max_period: 0,
			data_len: 0,
		});
	}
	let first = data.iter().position(|x| !x.is_nan()).ok_or(VlmaError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.max_period.unwrap()).max().unwrap();
	if data.len() - first < max_p {
		return Err(VlmaError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}
	let rows = combos.len();
	let cols = data.len();
	
	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let min_period = combos[row].min_period.unwrap();
		let max_period = combos[row].max_period.unwrap();
		let matype = combos[row].matype.as_ref().unwrap();
		let devtype = combos[row].devtype.unwrap();
		match kern {
			Kernel::Scalar => {
				vlma_row_scalar(data, min_period, max_period, matype, devtype, first, out_row).unwrap();
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => {
				vlma_row_avx2(data, min_period, max_period, matype, devtype, first, out_row).unwrap();
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => {
				vlma_row_avx512(data, min_period, max_period, matype, devtype, first, out_row).unwrap();
			}
			_ => unreachable!(),
		}
	};
	
	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			out.par_chunks_mut(cols)
				.enumerate()
				.for_each(|(row, slice)| do_row(row, slice));
		}

		#[cfg(target_arch = "wasm32")]
		{
			for (row, slice) in out.chunks_mut(cols).enumerate() {
				do_row(row, slice);
			}
		}
	} else {
		for (row, slice) in out.chunks_mut(cols).enumerate() {
			do_row(row, slice);
		}
	}
	
	Ok(combos)
}

#[inline(always)]
pub fn expand_grid_vlma(r: &VlmaBatchRange) -> Vec<VlmaParams> {
	expand_grid(r)
}


#[cfg(feature = "python")]
#[pyfunction(name = "vlma")]
#[pyo3(signature = (data, min_period=5, max_period=50, matype="sma", devtype=0, kernel=None))]
pub fn vlma_py<'py>(
	py: Python<'py>,
	data: PyReadonlyArray1<'py, f64>,
	min_period: usize,
	max_period: usize,
	matype: &str,
	devtype: usize,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
	use numpy::{IntoPyArray, PyArrayMethods};
	
	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, false)?;
	
	let params = VlmaParams {
		min_period: Some(min_period),
		max_period: Some(max_period),
		matype: Some(matype.to_string()),
		devtype: Some(devtype),
	};
	let input = VlmaInput::from_slice(slice_in, params);
	
	let result_vec: Vec<f64> = py
		.allow_threads(|| vlma_with_kernel(&input, kern).map(|o| o.values))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;
	
	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "VlmaStream")]
pub struct VlmaStreamPy {
	stream: VlmaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl VlmaStreamPy {
	#[new]
	fn new(min_period: usize, max_period: usize, matype: &str, devtype: usize) -> PyResult<Self> {
		let params = VlmaParams {
			min_period: Some(min_period),
			max_period: Some(max_period),
			matype: Some(matype.to_string()),
			devtype: Some(devtype),
		};
		let stream = VlmaStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(VlmaStreamPy { stream })
	}
	
	fn update(&mut self, value: f64) -> Option<f64> {
		self.stream.update(value)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "vlma_batch")]
#[pyo3(signature = (data, min_period_range=(5, 5, 0), max_period_range=(50, 50, 0), devtype_range=(0, 0, 0), matype="sma", kernel=None))]
pub fn vlma_batch_py<'py>(
	py: Python<'py>,
	data: PyReadonlyArray1<'py, f64>,
	min_period_range: (usize, usize, usize),
	max_period_range: (usize, usize, usize),
	devtype_range: (usize, usize, usize),
	matype: &str,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
	use numpy::{IntoPyArray, PyArrayMethods};
	
	let slice_in = data.as_slice()?;
	
	let sweep = VlmaBatchRange {
		min_period: min_period_range,
		max_period: max_period_range,
		matype: (matype.to_string(), matype.to_string(), "".to_string()),
		devtype: devtype_range,
	};
	
	let combos = expand_grid(&sweep);
	let rows = combos.len();
	let cols = slice_in.len();
	
	let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let slice_out = unsafe { out_arr.as_slice_mut()? };
	
	let kern = validate_kernel(kernel, true)?;
	
	let combos = py
		.allow_threads(|| {
			let kernel = match kern {
				Kernel::Auto => detect_best_batch_kernel(),
				k => k,
			};
			let simd = match kernel {
				Kernel::Avx512Batch => Kernel::Avx512,
				Kernel::Avx2Batch => Kernel::Avx2,
				Kernel::ScalarBatch => Kernel::Scalar,
				_ => unreachable!(),
			};
			vlma_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
		})
		.map_err(|e| PyValueError::new_err(e.to_string()))?;
	
	let dict = PyDict::new(py);
	dict.set_item("values", out_arr.reshape((rows, cols))?)?;
	dict.set_item(
		"min_periods",
		combos
			.iter()
			.map(|p| p.min_period.unwrap())
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;
	dict.set_item(
		"max_periods",
		combos
			.iter()
			.map(|p| p.max_period.unwrap())
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;
	dict.set_item(
		"devtypes",
		combos
			.iter()
			.map(|p| p.devtype.unwrap())
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;
	
	Ok(dict)
}

// WASM bindings following ALMA pattern

/// Helper function to write directly to output slice - no allocations
#[inline]
pub fn vlma_into_slice(
	dst: &mut [f64],
	input: &VlmaInput,
	kern: Kernel,
) -> Result<(), VlmaError> {
	// This is a placeholder implementation that still allocates internally
	// A proper zero-allocation implementation would require refactoring vlma_scalar
	// to accept an output slice parameter
	let data: &[f64] = input.as_ref();
	
	if dst.len() != data.len() {
		return Err(VlmaError::InvalidPeriod {
			min_period: 0,
			max_period: 0,
			data_len: dst.len(),
		});
	}
	
	let output = vlma_with_kernel(input, kern)?;
	dst.copy_from_slice(&output.values);
	Ok(())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vlma_js(
	data: &[f64],
	min_period: usize,
	max_period: usize,
	matype: &str,
	devtype: usize,
) -> Result<Vec<f64>, JsValue> {
	let params = VlmaParams {
		min_period: Some(min_period),
		max_period: Some(max_period),
		matype: Some(matype.to_string()),
		devtype: Some(devtype),
	};
	let input = VlmaInput::from_slice(data, params);
	
	// Safe API is allowed one allocation - directly call the main function
	vlma_with_kernel(&input, Kernel::Auto)
		.map(|o| o.values)
		.map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vlma_into(
	in_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	min_period: usize,
	max_period: usize,
	matype: &str,
	devtype: usize,
) -> Result<(), JsValue> {
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}
	
	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);
		let params = VlmaParams {
			min_period: Some(min_period),
			max_period: Some(max_period),
			matype: Some(matype.to_string()),
			devtype: Some(devtype),
		};
		let input = VlmaInput::from_slice(data, params);
		
		// Check for aliasing
		if in_ptr == out_ptr as *const f64 {
			// Use temporary buffer for in-place operation
			let mut temp = vec![0.0; len];
			vlma_into_slice(&mut temp, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			out.copy_from_slice(&temp);
		} else {
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			vlma_into_slice(out, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vlma_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vlma_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe {
			let _ = Vec::from_raw_parts(ptr, len, len);
		}
	}
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct VlmaBatchConfig {
	pub min_period_range: (usize, usize, usize),
	pub max_period_range: (usize, usize, usize),
	pub devtype_range: (usize, usize, usize),
	pub matype: String,
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct VlmaBatchJsOutput {
	pub values: Vec<f64>,
	pub combos: Vec<VlmaParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = vlma_batch)]
pub fn vlma_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	let config: VlmaBatchConfig = serde_wasm_bindgen::from_value(config)
		.map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
	
	let sweep = VlmaBatchRange {
		min_period: config.min_period_range,
		max_period: config.max_period_range,
		matype: (config.matype.clone(), config.matype.clone(), "".to_string()),
		devtype: config.devtype_range,
	};
	
	let output = vlma_batch_inner(data, &sweep, Kernel::Auto, false)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	let js_output = VlmaBatchJsOutput {
		values: output.values,
		combos: output.combos,
		rows: output.rows,
		cols: output.cols,
	};
	
	serde_wasm_bindgen::to_value(&js_output)
		.map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vlma_batch_into(
	in_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	min_period_start: usize,
	min_period_end: usize,
	min_period_step: usize,
	max_period_start: usize,
	max_period_end: usize,
	max_period_step: usize,
	devtype_start: usize,
	devtype_end: usize,
	devtype_step: usize,
	matype: &str,
) -> Result<usize, JsValue> {
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("null pointer passed to vlma_batch_into"));
	}
	
	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);
		
		let sweep = VlmaBatchRange {
			min_period: (min_period_start, min_period_end, min_period_step),
			max_period: (max_period_start, max_period_end, max_period_step),
			matype: (matype.to_string(), matype.to_string(), "".to_string()),
			devtype: (devtype_start, devtype_end, devtype_step),
		};
		
		let combos = expand_grid(&sweep);
		let total_len = combos.len() * len;
		let out_slice = std::slice::from_raw_parts_mut(out_ptr, total_len);
		
		let _ = vlma_batch_inner_into(data, &sweep, Kernel::Auto, false, out_slice)
			.map_err(|e| JsValue::from_str(&e.to_string()))?;
		
		Ok(combos.len())
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;
	fn check_vlma_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let default_params = VlmaParams {
			min_period: None,
			max_period: None,
			matype: None,
			devtype: None,
		};
		let input_default = VlmaInput::from_candles(&candles, "close", default_params);
		let output_default = vlma_with_kernel(&input_default, kernel)?;
		assert_eq!(output_default.values.len(), candles.close.len());
		Ok(())
	}
	fn check_vlma_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let close_prices = candles.select_candle_field("close")?;
		let params = VlmaParams {
			min_period: Some(5),
			max_period: Some(50),
			matype: Some("sma".to_string()),
			devtype: Some(0),
		};
		let input = VlmaInput::from_candles(&candles, "close", params);
		let vlma_result = vlma_with_kernel(&input, kernel)?;
		assert_eq!(vlma_result.values.len(), close_prices.len());
		let required_len = 5;
		assert!(vlma_result.values.len() >= required_len, "VLMA length is too short");
		let test_vals = [
			59376.252799490234,
			59343.71066624187,
			59292.92555520155,
			59269.93796266796,
			59167.4483022233,
		];
		let start_idx = vlma_result.values.len() - test_vals.len();
		let actual_slice = &vlma_result.values[start_idx..];
		for (i, &val) in actual_slice.iter().enumerate() {
			let expected = test_vals[i];
			if !val.is_nan() {
				assert!(
					(val - expected).abs() < 1e-1,
					"Mismatch at index {}: expected {}, got {}",
					i,
					expected,
					val
				);
			}
		}
		Ok(())
	}
	fn check_vlma_zero_or_inverted_periods(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let input_data = [10.0, 20.0, 30.0, 40.0];
		let params_min_greater = VlmaParams {
			min_period: Some(10),
			max_period: Some(5),
			matype: Some("sma".to_string()),
			devtype: Some(0),
		};
		let input_min_greater = VlmaInput::from_slice(&input_data, params_min_greater);
		let result = vlma_with_kernel(&input_min_greater, kernel);
		assert!(result.is_err());
		let params_zero_max = VlmaParams {
			min_period: Some(5),
			max_period: Some(0),
			matype: Some("sma".to_string()),
			devtype: Some(0),
		};
		let input_zero_max = VlmaInput::from_slice(&input_data, params_zero_max);
		let result2 = vlma_with_kernel(&input_zero_max, kernel);
		assert!(result2.is_err());
		Ok(())
	}
	fn check_vlma_not_enough_data(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let input_data = [10.0, 20.0, 30.0];
		let params = VlmaParams {
			min_period: Some(5),
			max_period: Some(10),
			matype: Some("sma".to_string()),
			devtype: Some(0),
		};
		let input = VlmaInput::from_slice(&input_data, params);
		let result = vlma_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}
	fn check_vlma_all_nan(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let input_data = [f64::NAN, f64::NAN, f64::NAN];
		let params = VlmaParams {
			min_period: Some(2),
			max_period: Some(3),
			matype: Some("sma".to_string()),
			devtype: Some(0),
		};
		let input = VlmaInput::from_slice(&input_data, params);
		let result = vlma_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}
	fn check_vlma_slice_reinput(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let first_params = VlmaParams {
			min_period: Some(5),
			max_period: Some(20),
			matype: Some("ema".to_string()),
			devtype: Some(1),
		};
		let first_input = VlmaInput::from_candles(&candles, "close", first_params);
		let first_result = vlma_with_kernel(&first_input, kernel)?;
		let second_params = VlmaParams {
			min_period: Some(5),
			max_period: Some(20),
			matype: Some("ema".to_string()),
			devtype: Some(1),
		};
		let second_input = VlmaInput::from_slice(&first_result.values, second_params);
		let second_result = vlma_with_kernel(&second_input, kernel)?;
		assert_eq!(second_result.values.len(), first_result.values.len());
		Ok(())
	}
	fn check_vlma_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = VlmaParams {
			min_period: Some(5),
			max_period: Some(50),
			matype: Some("sma".to_string()),
			devtype: Some(0),
		};
		let input = VlmaInput::from_candles(&candles, "close", params.clone());
		let batch_output = vlma_with_kernel(&input, kernel)?.values;
		let mut stream = VlmaStream::try_new(params)?;
		let mut stream_values = Vec::with_capacity(candles.close.len());
		for &price in &candles.close {
			match stream.update(price) {
				Some(v) => stream_values.push(v),
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
				"[{}] VLMA streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
				test_name,
				i,
				b,
				s,
				diff
			);
		}
		Ok(())
	}
	macro_rules! generate_all_vlma_tests {
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
	generate_all_vlma_tests!(
		check_vlma_partial_params,
		check_vlma_accuracy,
		check_vlma_zero_or_inverted_periods,
		check_vlma_not_enough_data,
		check_vlma_all_nan,
		check_vlma_slice_reinput,
		check_vlma_streaming
	);
	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let output = VlmaBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;
		let def = VlmaParams::default();
		let row = output.values_for(&def).expect("default row missing");
		assert_eq!(row.len(), c.close.len());
		let expected = [
			59376.252799490234,
			59343.71066624187,
			59292.92555520155,
			59269.93796266796,
			59167.4483022233,
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
