//! # Z-Score (Zscore)
//!
//! A statistical measurement that describes a value's relationship to the mean of a group of values,
//! measured in terms of standard deviations. A Z-Score of 0 indicates the value is identical to the mean,
//! while positive/negative Z-Scores indicate how many standard deviations above/below the mean the value is.
//!
//! ## Parameters
//! - **period**: Window size (number of data points). Defaults to 14.
//! - **ma_type**: Type of moving average for the mean. Defaults to `"sma"`.
//! - **nbdev**: Multiplier for deviation. Defaults to 1.0.
//! - **devtype**: 0 = stddev, 1 = mean abs dev, 2 = median abs dev. Defaults to 0.
//!
//! ## Errors
//! - **AllValuesNaN**: zscore: All input data values are `NaN`.
//! - **InvalidPeriod**: zscore: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: zscore: Not enough valid data points for the requested `period`.
//! - **DevError**: zscore: Underlying error from the deviation function.
//! - **MaError**: zscore: Underlying error from the moving average function.
//!
//! ## Returns
//! - **`Ok(ZscoreOutput)`** on success, containing a `Vec<f64>` matching the input.
//! - **`Err(ZscoreError)`** otherwise.

use crate::indicators::deviation::{deviation, DevError, DevInput, DevParams, DeviationData, DeviationOutput};
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
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
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
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::error::Error;
use thiserror::Error;

impl<'a> AsRef<[f64]> for ZscoreInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			ZscoreData::Slice(slice) => slice,
			ZscoreData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

#[derive(Debug, Clone)]
pub enum ZscoreData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct ZscoreOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct ZscoreParams {
	pub period: Option<usize>,
	pub ma_type: Option<String>,
	pub nbdev: Option<f64>,
	pub devtype: Option<usize>,
}

impl Default for ZscoreParams {
	fn default() -> Self {
		Self {
			period: Some(14),
			ma_type: Some("sma".to_string()),
			nbdev: Some(1.0),
			devtype: Some(0),
		}
	}
}

#[derive(Debug, Clone)]
pub struct ZscoreInput<'a> {
	pub data: ZscoreData<'a>,
	pub params: ZscoreParams,
}

impl<'a> ZscoreInput<'a> {
	#[inline]
	pub fn from_candles(candles: &'a Candles, source: &'a str, params: ZscoreParams) -> Self {
		Self {
			data: ZscoreData::Candles { candles, source },
			params,
		}
	}
	#[inline]
	pub fn from_slice(slice: &'a [f64], params: ZscoreParams) -> Self {
		Self {
			data: ZscoreData::Slice(slice),
			params,
		}
	}
	#[inline]
	pub fn with_default_candles(candles: &'a Candles) -> Self {
		Self::from_candles(candles, "close", ZscoreParams::default())
	}
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(14)
	}
	#[inline]
	pub fn get_ma_type(&self) -> String {
		self.params.ma_type.clone().unwrap_or_else(|| "sma".to_string())
	}
	#[inline]
	pub fn get_nbdev(&self) -> f64 {
		self.params.nbdev.unwrap_or(1.0)
	}
	#[inline]
	pub fn get_devtype(&self) -> usize {
		self.params.devtype.unwrap_or(0)
	}
}

#[derive(Clone, Debug)]
pub struct ZscoreBuilder {
	period: Option<usize>,
	ma_type: Option<String>,
	nbdev: Option<f64>,
	devtype: Option<usize>,
	kernel: Kernel,
}

impl Default for ZscoreBuilder {
	fn default() -> Self {
		Self {
			period: None,
			ma_type: None,
			nbdev: None,
			devtype: None,
			kernel: Kernel::Auto,
		}
	}
}

impl ZscoreBuilder {
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
	pub fn ma_type<T: Into<String>>(mut self, t: T) -> Self {
		self.ma_type = Some(t.into());
		self
	}
	#[inline(always)]
	pub fn nbdev(mut self, x: f64) -> Self {
		self.nbdev = Some(x);
		self
	}
	#[inline(always)]
	pub fn devtype(mut self, dt: usize) -> Self {
		self.devtype = Some(dt);
		self
	}
	#[inline(always)]
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline(always)]
	pub fn apply(self, candles: &Candles) -> Result<ZscoreOutput, ZscoreError> {
		let p = ZscoreParams {
			period: self.period,
			ma_type: self.ma_type,
			nbdev: self.nbdev,
			devtype: self.devtype,
		};
		let i = ZscoreInput::from_candles(candles, "close", p);
		zscore_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slice(self, data: &[f64]) -> Result<ZscoreOutput, ZscoreError> {
		let p = ZscoreParams {
			period: self.period,
			ma_type: self.ma_type,
			nbdev: self.nbdev,
			devtype: self.devtype,
		};
		let i = ZscoreInput::from_slice(data, p);
		zscore_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<ZscoreStream, ZscoreError> {
		let p = ZscoreParams {
			period: self.period,
			ma_type: self.ma_type,
			nbdev: self.nbdev,
			devtype: self.devtype,
		};
		ZscoreStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum ZscoreError {
	#[error("zscore: All values are NaN.")]
	AllValuesNaN,
	#[error("zscore: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("zscore: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("zscore: DevError {0}")]
	DevError(#[from] DevError),
	#[error("zscore: MaError {0}")]
	MaError(String),
}

#[inline]
pub fn zscore(input: &ZscoreInput) -> Result<ZscoreOutput, ZscoreError> {
	zscore_with_kernel(input, Kernel::Auto)
}

pub fn zscore_with_kernel(input: &ZscoreInput, kernel: Kernel) -> Result<ZscoreOutput, ZscoreError> {
	let data: &[f64] = input.as_ref();

	let first = data.iter().position(|x| !x.is_nan()).ok_or(ZscoreError::AllValuesNaN)?;
	let len = data.len();
	let period = input.get_period();
	if period == 0 || period > len {
		return Err(ZscoreError::InvalidPeriod { period, data_len: len });
	}
	if (len - first) < period {
		return Err(ZscoreError::NotEnoughValidData {
			needed: period,
			valid: len - first,
		});
	}

	let ma_type = input.get_ma_type();
	let nbdev = input.get_nbdev();
	let devtype = input.get_devtype();

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => zscore_scalar(data, period, first, &ma_type, nbdev, devtype),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => zscore_avx2(data, period, first, &ma_type, nbdev, devtype),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => zscore_avx512(data, period, first, &ma_type, nbdev, devtype),
			_ => unreachable!(),
		}
	}
}

#[inline]
pub unsafe fn zscore_scalar(
	data: &[f64],
	period: usize,
	first: usize,
	ma_type: &str,
	nbdev: f64,
	devtype: usize,
) -> Result<ZscoreOutput, ZscoreError> {
	let means = ma(ma_type, MaData::Slice(data), period)
		.map_err(|e| ZscoreError::MaError(e.to_string()))?;
	let dev_input = DevInput {
		data: DeviationData::Slice(data),
		params: DevParams {
			period: Some(period),
			devtype: Some(devtype),
		},
	};
	let mut sigmas = deviation(&dev_input)?.values;
	for v in &mut sigmas {
		*v *= nbdev;
	}
	let mut out = alloc_with_nan_prefix(data.len(), first + period - 1);
	// Fill remaining values with NaN for binding compatibility
	for i in (first + period - 1)..out.len() {
		out[i] = f64::NAN;
	}
	for i in (first + period - 1)..data.len() {
		let mean = means[i - first];
		let sigma = sigmas[i - first];
		let value = data[i];
		out[i] = if sigma == 0.0 || sigma.is_nan() {
			f64::NAN
		} else {
			(value - mean) / sigma
		};
	}
	Ok(ZscoreOutput { values: out })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn zscore_avx2(
	data: &[f64],
	period: usize,
	first: usize,
	ma_type: &str,
	nbdev: f64,
	devtype: usize,
) -> Result<ZscoreOutput, ZscoreError> {
	zscore_scalar(data, period, first, ma_type, nbdev, devtype)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn zscore_avx512(
	data: &[f64],
	period: usize,
	first: usize,
	ma_type: &str,
	nbdev: f64,
	devtype: usize,
) -> Result<ZscoreOutput, ZscoreError> {
	if period <= 32 {
		zscore_avx512_short(data, period, first, ma_type, nbdev, devtype)
	} else {
		zscore_avx512_long(data, period, first, ma_type, nbdev, devtype)
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn zscore_avx512_short(
	data: &[f64],
	period: usize,
	first: usize,
	ma_type: &str,
	nbdev: f64,
	devtype: usize,
) -> Result<ZscoreOutput, ZscoreError> {
	zscore_scalar(data, period, first, ma_type, nbdev, devtype)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn zscore_avx512_long(
	data: &[f64],
	period: usize,
	first: usize,
	ma_type: &str,
	nbdev: f64,
	devtype: usize,
) -> Result<ZscoreOutput, ZscoreError> {
	zscore_scalar(data, period, first, ma_type, nbdev, devtype)
}

#[derive(Debug, Clone)]
pub struct ZscoreStream {
	period: usize,
	ma_type: String,
	nbdev: f64,
	devtype: usize,
	buffer: Vec<f64>,
	head: usize,
	filled: bool,
}

impl ZscoreStream {
	pub fn try_new(params: ZscoreParams) -> Result<Self, ZscoreError> {
		let period = params.period.unwrap_or(14);
		if period == 0 {
			return Err(ZscoreError::InvalidPeriod { period, data_len: 0 });
		}
		let ma_type = params.ma_type.unwrap_or_else(|| "sma".to_string());
		let nbdev = params.nbdev.unwrap_or(1.0);
		let devtype = params.devtype.unwrap_or(0);
		Ok(Self {
			period,
			ma_type,
			nbdev,
			devtype,
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
		Some(self.compute_zscore())
	}

	#[inline(always)]
	fn compute_zscore(&self) -> f64 {
		let mut ordered = vec![0.0; self.period];
		let mut idx = self.head;
		for i in 0..self.period {
			ordered[i] = self.buffer[idx];
			idx = (idx + 1) % self.period;
		}
		let means = match ma(&self.ma_type, MaData::Slice(&ordered), self.period) {
			Ok(m) => m,
			Err(_) => return f64::NAN,
		};
		let dev_input = DevInput {
			data: DeviationData::Slice(&ordered),
			params: DevParams {
				period: Some(self.period),
				devtype: Some(self.devtype),
			},
		};
		let mut sigmas = match deviation(&dev_input) {
			Ok(d) => d.values,
			Err(_) => return f64::NAN,
		};
		for s in &mut sigmas {
			*s *= self.nbdev;
		}
		let mean = means[self.period - 1];
		let sigma = sigmas[self.period - 1];
		let value = ordered[self.period - 1];
		if sigma == 0.0 || sigma.is_nan() {
			f64::NAN
		} else {
			(value - mean) / sigma
		}
	}
}

#[derive(Clone, Debug)]
pub struct ZscoreBatchRange {
	pub period: (usize, usize, usize),
	pub ma_type: (String, String, String),
	pub nbdev: (f64, f64, f64),
	pub devtype: (usize, usize, usize),
}

impl Default for ZscoreBatchRange {
	fn default() -> Self {
		Self {
			period: (14, 50, 1),
			ma_type: ("sma".to_string(), "sma".to_string(), "".to_string()),
			nbdev: (1.0, 1.0, 0.0),
			devtype: (0, 0, 0),
		}
	}
}

#[derive(Clone, Debug, Default)]
pub struct ZscoreBatchBuilder {
	range: ZscoreBatchRange,
	kernel: Kernel,
}

impl ZscoreBatchBuilder {
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
	#[inline]
	pub fn ma_type_static<T: Into<String>>(mut self, s: T) -> Self {
		let val = s.into();
		self.range.ma_type = (val.clone(), val.clone(), "".to_string());
		self
	}
	#[inline]
	pub fn nbdev_range(mut self, start: f64, end: f64, step: f64) -> Self {
		self.range.nbdev = (start, end, step);
		self
	}
	#[inline]
	pub fn nbdev_static(mut self, x: f64) -> Self {
		self.range.nbdev = (x, x, 0.0);
		self
	}
	#[inline]
	pub fn devtype_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.devtype = (start, end, step);
		self
	}
	#[inline]
	pub fn devtype_static(mut self, x: usize) -> Self {
		self.range.devtype = (x, x, 0);
		self
	}
	pub fn apply_slice(self, data: &[f64]) -> Result<ZscoreBatchOutput, ZscoreError> {
		zscore_batch_with_kernel(data, &self.range, self.kernel)
	}
	pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<ZscoreBatchOutput, ZscoreError> {
		ZscoreBatchBuilder::new().kernel(k).apply_slice(data)
	}
	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<ZscoreBatchOutput, ZscoreError> {
		let slice = source_type(c, src);
		self.apply_slice(slice)
	}
	pub fn with_default_candles(c: &Candles) -> Result<ZscoreBatchOutput, ZscoreError> {
		ZscoreBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
	}
}

pub fn zscore_batch_with_kernel(
	data: &[f64],
	sweep: &ZscoreBatchRange,
	k: Kernel,
) -> Result<ZscoreBatchOutput, ZscoreError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => {
			return Err(ZscoreError::InvalidPeriod { period: 0, data_len: 0 });
		}
	};

	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	zscore_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct ZscoreBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<ZscoreParams>,
	pub rows: usize,
	pub cols: usize,
}

impl ZscoreBatchOutput {
	pub fn row_for_params(&self, p: &ZscoreParams) -> Option<usize> {
		self.combos.iter().position(|c| {
			c.period.unwrap_or(14) == p.period.unwrap_or(14)
				&& c.ma_type.as_ref().unwrap_or(&"sma".to_string()) == p.ma_type.as_ref().unwrap_or(&"sma".to_string())
				&& (c.nbdev.unwrap_or(1.0) - p.nbdev.unwrap_or(1.0)).abs() < 1e-12
				&& c.devtype.unwrap_or(0) == p.devtype.unwrap_or(0)
		})
	}
	pub fn values_for(&self, p: &ZscoreParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &ZscoreBatchRange) -> Vec<ZscoreParams> {
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
	fn axis_string((start, end, step): (String, String, String)) -> Vec<String> {
		if start == end {
			return vec![start];
		}
		vec![start]
	}

	let periods = axis_usize(r.period);
	let ma_types = axis_string(r.ma_type.clone());
	let nbdevs = axis_f64(r.nbdev);
	let devtypes = axis_usize(r.devtype);

	let mut out = Vec::with_capacity(periods.len() * ma_types.len() * nbdevs.len() * devtypes.len());
	for &p in &periods {
		for mt in &ma_types {
			for &n in &nbdevs {
				for &dt in &devtypes {
					out.push(ZscoreParams {
						period: Some(p),
						ma_type: Some(mt.clone()),
						nbdev: Some(n),
						devtype: Some(dt),
					});
				}
			}
		}
	}
	out
}

#[inline(always)]
pub fn zscore_batch_slice(
	data: &[f64],
	sweep: &ZscoreBatchRange,
	kern: Kernel,
) -> Result<ZscoreBatchOutput, ZscoreError> {
	zscore_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn zscore_batch_par_slice(
	data: &[f64],
	sweep: &ZscoreBatchRange,
	kern: Kernel,
) -> Result<ZscoreBatchOutput, ZscoreError> {
	zscore_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn zscore_batch_inner(
	data: &[f64],
	sweep: &ZscoreBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<ZscoreBatchOutput, ZscoreError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(ZscoreError::InvalidPeriod { period: 0, data_len: 0 });
	}

	let first = data.iter().position(|x| !x.is_nan()).ok_or(ZscoreError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if data.len() - first < max_p {
		return Err(ZscoreError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}

	let rows = combos.len();
	let cols = data.len();
	
	// Use uninitialized memory like ALMA
	let mut buf_mu = make_uninit_matrix(rows, cols);
	
	// Calculate warmup periods for each combination
	let warmup_periods: Vec<usize> = combos
		.iter()
		.map(|c| first + c.period.unwrap() - 1)
		.collect();
	
	// Initialize NaN prefixes for each row
	init_matrix_prefixes(&mut buf_mu, cols, &warmup_periods);
	
	// Create a guard to manage the buffer
	let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
	let out: &mut [f64] = 
		unsafe { core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len()) };
	
	// Fill remaining values with NaN for binding compatibility
	for (row_idx, warmup) in warmup_periods.iter().enumerate() {
		let row_start = row_idx * cols;
		for col in *warmup..cols {
			out[row_start + col] = f64::NAN;
		}
	}
	
	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let prm = &combos[row];
		let period = prm.period.unwrap();
		let ma_type = prm.ma_type.as_ref().unwrap();
		let nbdev = prm.nbdev.unwrap();
		let devtype = prm.devtype.unwrap();
		match kern {
			Kernel::Scalar => zscore_row_scalar(data, first, period, ma_type, nbdev, devtype, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => zscore_row_avx2(data, first, period, ma_type, nbdev, devtype, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => zscore_row_avx512(data, first, period, ma_type, nbdev, devtype, out_row),
			_ => unreachable!(),
		}
	};
	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			out
				.par_chunks_mut(cols)
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
	
	// Convert the uninitialized buffer to a proper Vec
	let values = unsafe {
		Vec::from_raw_parts(
			buf_guard.as_mut_ptr() as *mut f64,
			buf_guard.len(),
			buf_guard.capacity(),
		)
	};
	
	Ok(ZscoreBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
unsafe fn zscore_row_scalar(
	data: &[f64],
	first: usize,
	period: usize,
	ma_type: &str,
	nbdev: f64,
	devtype: usize,
	out: &mut [f64],
) {
	let means = match ma(ma_type, MaData::Slice(data), period) {
		Ok(m) => m,
		Err(_) => {
			// If MA fails, fill output with NaN
			out.fill(f64::NAN);
			return;
		}
	};
	let dev_input = DevInput {
		data: DeviationData::Slice(data),
		params: DevParams {
			period: Some(period),
			devtype: Some(devtype),
		},
	};
	let mut sigmas = match deviation(&dev_input) {
		Ok(d) => d.values,
		Err(_) => {
			// If deviation fails, fill output with NaN
			out.fill(f64::NAN);
			return;
		}
	};
	for v in &mut sigmas {
		*v *= nbdev;
	}
	for i in (first + period - 1)..data.len() {
		let mean = means[i - first];
		let sigma = sigmas[i - first];
		let value = data[i];
		out[i] = if sigma == 0.0 || sigma.is_nan() {
			f64::NAN
		} else {
			(value - mean) / sigma
		};
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn zscore_row_avx2(
	data: &[f64],
	first: usize,
	period: usize,
	ma_type: &str,
	nbdev: f64,
	devtype: usize,
	out: &mut [f64],
) {
	zscore_row_scalar(data, first, period, ma_type, nbdev, devtype, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn zscore_row_avx512(
	data: &[f64],
	first: usize,
	period: usize,
	ma_type: &str,
	nbdev: f64,
	devtype: usize,
	out: &mut [f64],
) {
	if period <= 32 {
		zscore_row_avx512_short(data, first, period, ma_type, nbdev, devtype, out)
	} else {
		zscore_row_avx512_long(data, first, period, ma_type, nbdev, devtype, out)
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn zscore_row_avx512_short(
	data: &[f64],
	first: usize,
	period: usize,
	ma_type: &str,
	nbdev: f64,
	devtype: usize,
	out: &mut [f64],
) {
	zscore_row_scalar(data, first, period, ma_type, nbdev, devtype, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn zscore_row_avx512_long(
	data: &[f64],
	first: usize,
	period: usize,
	ma_type: &str,
	nbdev: f64,
	devtype: usize,
	out: &mut [f64],
) {
	zscore_row_scalar(data, first, period, ma_type, nbdev, devtype, out)
}

#[inline(always)]
fn expand_grid_zscore(r: &ZscoreBatchRange) -> Vec<ZscoreParams> {
	expand_grid(r)
}

#[inline(always)]
pub fn zscore_batch_inner_into(
	data: &[f64],
	sweep: &ZscoreBatchRange,
	kern: Kernel,
	parallel: bool,
	out: &mut [f64],
) -> Result<Vec<ZscoreParams>, ZscoreError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(ZscoreError::InvalidPeriod { period: 0, data_len: 0 });
	}

	let first = data.iter().position(|x| !x.is_nan()).ok_or(ZscoreError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if data.len() - first < max_p {
		return Err(ZscoreError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}

	let cols = data.len();
	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let prm = &combos[row];
		let period = prm.period.unwrap();
		let ma_type = prm.ma_type.as_ref().unwrap();
		let nbdev = prm.nbdev.unwrap();
		let devtype = prm.devtype.unwrap();
		match kern {
			Kernel::Scalar => zscore_row_scalar(data, first, period, ma_type, nbdev, devtype, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => zscore_row_avx2(data, first, period, ma_type, nbdev, devtype, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => zscore_row_avx512(data, first, period, ma_type, nbdev, devtype, out_row),
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

#[cfg(feature = "python")]
#[pyfunction(name = "zscore")]
#[pyo3(signature = (data, period=14, ma_type="sma", nbdev=1.0, devtype=0, kernel=None))]
pub fn zscore_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	period: usize,
	ma_type: &str,
	nbdev: f64,
	devtype: usize,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
	use numpy::{IntoPyArray, PyArrayMethods};

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, false)?;
	let params = ZscoreParams {
		period: Some(period),
		ma_type: Some(ma_type.to_string()),
		nbdev: Some(nbdev),
		devtype: Some(devtype),
	};
	let input = ZscoreInput::from_slice(slice_in, params);

	let result_vec: Vec<f64> = py
		.allow_threads(|| zscore_with_kernel(&input, kern).map(|o| o.values))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "ZscoreStream")]
pub struct ZscoreStreamPy {
	stream: ZscoreStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl ZscoreStreamPy {
	#[new]
	fn new(period: usize, ma_type: &str, nbdev: f64, devtype: usize) -> PyResult<Self> {
		let params = ZscoreParams {
			period: Some(period),
			ma_type: Some(ma_type.to_string()),
			nbdev: Some(nbdev),
			devtype: Some(devtype),
		};
		let stream = ZscoreStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(ZscoreStreamPy { stream })
	}

	fn update(&mut self, value: f64) -> Option<f64> {
		self.stream.update(value)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "zscore_batch")]
#[pyo3(signature = (data, period_range, ma_type="sma", nbdev_range=(1.0, 1.0, 0.0), devtype_range=(0, 0, 0), kernel=None))]
pub fn zscore_batch_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	period_range: (usize, usize, usize),
	ma_type: &str,
	nbdev_range: (f64, f64, f64),
	devtype_range: (usize, usize, usize),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;

	let slice_in = data.as_slice()?;

	let sweep = ZscoreBatchRange {
		period: period_range,
		ma_type: (ma_type.to_string(), ma_type.to_string(), "".to_string()),
		nbdev: nbdev_range,
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
			zscore_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
		})
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

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
		"nbdevs",
		combos
			.iter()
			.map(|p| p.nbdev.unwrap())
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;
	dict.set_item(
		"devtypes",
		combos
			.iter()
			.map(|p| p.devtype.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;

	Ok(dict)
}

// WASM helper functions

/// Write zscore directly to output slice - no allocations
pub fn zscore_into_slice(
	dst: &mut [f64],
	input: &ZscoreInput,
	kern: Kernel,
) -> Result<(), ZscoreError> {
	let data: &[f64] = input.as_ref();
	
	if dst.len() != data.len() {
		return Err(ZscoreError::InvalidPeriod { 
			period: 0, 
			data_len: dst.len() 
		});
	}
	
	let first = data.iter().position(|x| !x.is_nan()).ok_or(ZscoreError::AllValuesNaN)?;
	let len = data.len();
	let period = input.get_period();
	if period == 0 || period > len {
		return Err(ZscoreError::InvalidPeriod { period, data_len: len });
	}
	if (len - first) < period {
		return Err(ZscoreError::NotEnoughValidData {
			needed: period,
			valid: len - first,
		});
	}

	let ma_type = input.get_ma_type();
	let nbdev = input.get_nbdev();
	let devtype = input.get_devtype();

	let chosen = match kern {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => zscore_compute_into_scalar(data, period, first, &ma_type, nbdev, devtype, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => zscore_compute_into_avx2(data, period, first, &ma_type, nbdev, devtype, dst),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => zscore_compute_into_avx512(data, period, first, &ma_type, nbdev, devtype, dst),
			_ => return Err(ZscoreError::InvalidPeriod { period: 0, data_len: 0 }),
		}
	}?;
	
	Ok(())
}

#[inline]
unsafe fn zscore_compute_into_scalar(
	data: &[f64],
	period: usize,
	first: usize,
	ma_type: &str,
	nbdev: f64,
	devtype: usize,
	out: &mut [f64],
) -> Result<(), ZscoreError> {
	// Fill warmup with NaN
	for v in &mut out[..(first + period - 1)] {
		*v = f64::NAN;
	}

	let means = ma(ma_type, MaData::Slice(data), period)
		.map_err(|e| ZscoreError::MaError(e.to_string()))?;
	let dev_input = DevInput {
		data,
		params: DevParams {
			period: Some(period),
			devtype: Some(devtype),
		},
	};
	let mut sigmas = deviation(&dev_input)?.values;
	for v in &mut sigmas {
		*v *= nbdev;
	}
	
	for i in (first + period - 1)..data.len() {
		let mean = means[i - first];
		let sigma = sigmas[i - first];
		let value = data[i];
		out[i] = if sigma == 0.0 || sigma.is_nan() {
			f64::NAN
		} else {
			(value - mean) / sigma
		};
	}
	Ok(())
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
unsafe fn zscore_compute_into_avx2(
	data: &[f64],
	period: usize,
	first: usize,
	ma_type: &str,
	nbdev: f64,
	devtype: usize,
	out: &mut [f64],
) -> Result<(), ZscoreError> {
	zscore_compute_into_scalar(data, period, first, ma_type, nbdev, devtype, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
unsafe fn zscore_compute_into_avx512(
	data: &[f64],
	period: usize,
	first: usize,
	ma_type: &str,
	nbdev: f64,
	devtype: usize,
	out: &mut [f64],
) -> Result<(), ZscoreError> {
	zscore_compute_into_scalar(data, period, first, ma_type, nbdev, devtype, out)
}

// WASM bindings

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn zscore_js(data: &[f64], period: usize, ma_type: &str, nbdev: f64, devtype: usize) -> Result<Vec<f64>, JsValue> {
	let params = ZscoreParams {
		period: Some(period),
		ma_type: Some(ma_type.to_string()),
		nbdev: Some(nbdev),
		devtype: Some(devtype),
	};
	let input = ZscoreInput::from_slice(data, params);
	
	let mut output = vec![0.0; data.len()];
	
	zscore_into_slice(&mut output, &input, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn zscore_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn zscore_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe {
			let _ = Vec::from_raw_parts(ptr, len, len);
		}
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn zscore_into(
	in_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period: usize,
	ma_type: &str,
	nbdev: f64,
	devtype: usize,
) -> Result<(), JsValue> {
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}
	
	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);
		let params = ZscoreParams {
			period: Some(period),
			ma_type: Some(ma_type.to_string()),
			nbdev: Some(nbdev),
			devtype: Some(devtype),
		};
		let input = ZscoreInput::from_slice(data, params);
		
		if in_ptr == out_ptr {  // Aliasing check
			let mut temp = vec![0.0; len];
			zscore_into_slice(&mut temp, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			out.copy_from_slice(&temp);
		} else {
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			zscore_into_slice(out, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
	}
	
	Ok(())
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct ZscoreBatchConfig {
	pub period_range: (usize, usize, usize),
	pub ma_type: String,
	pub nbdev_range: (f64, f64, f64),
	pub devtype_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct ZscoreBatchJsOutput {
	pub values: Vec<f64>,
	pub combos: Vec<ZscoreParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = zscore_batch)]
pub fn zscore_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	let config: ZscoreBatchConfig = 
		serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
	
	let sweep = ZscoreBatchRange {
		period: config.period_range,
		ma_type: (config.ma_type.clone(), config.ma_type.clone(), "".to_string()),
		nbdev: config.nbdev_range,
		devtype: config.devtype_range,
	};
	
	let output = zscore_batch_inner(data, &sweep, Kernel::Auto, false)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	let js_output = ZscoreBatchJsOutput {
		values: output.values,
		combos: output.combos,
		rows: output.rows,
		cols: output.cols,
	};
	
	serde_wasm_bindgen::to_value(&js_output).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn zscore_batch_into(
	in_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period_start: usize,
	period_end: usize,
	period_step: usize,
	ma_type: &str,
	nbdev_start: f64,
	nbdev_end: f64,
	nbdev_step: f64,
	devtype_start: usize,
	devtype_end: usize,
	devtype_step: usize,
) -> Result<usize, JsValue> {
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}
	
	let sweep = ZscoreBatchRange {
		period: (period_start, period_end, period_step),
		ma_type: (ma_type.to_string(), ma_type.to_string(), "".to_string()),
		nbdev: (nbdev_start, nbdev_end, nbdev_step),
		devtype: (devtype_start, devtype_end, devtype_step),
	};
	
	let combos = expand_grid(&sweep);
	let n_combos = combos.len();
	
	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);
		let out = std::slice::from_raw_parts_mut(out_ptr, n_combos * len);
		
		let simd = detect_best_kernel();
		zscore_batch_inner_into(data, &sweep, simd, false, out)
			.map_err(|e| JsValue::from_str(&e.to_string()))?;
	}
	
	Ok(n_combos)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;
	fn check_zscore_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let default_params = ZscoreParams {
			period: None,
			ma_type: None,
			nbdev: None,
			devtype: None,
		};
		let input = ZscoreInput::from_candles(&candles, "close", default_params);
		let output = zscore_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}
	fn check_zscore_with_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let input_data = [10.0, 20.0, 30.0];
		let params = ZscoreParams {
			period: Some(0),
			ma_type: None,
			nbdev: None,
			devtype: None,
		};
		let input = ZscoreInput::from_slice(&input_data, params);
		let res = zscore_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] Zscore should fail with zero period", test_name);
		Ok(())
	}
	fn check_zscore_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data_small = [10.0, 20.0, 30.0];
		let params = ZscoreParams {
			period: Some(10),
			ma_type: None,
			nbdev: None,
			devtype: None,
		};
		let input = ZscoreInput::from_slice(&data_small, params);
		let res = zscore_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] Zscore should fail with period exceeding length",
			test_name
		);
		Ok(())
	}
	fn check_zscore_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let single_point = [42.0];
		let params = ZscoreParams {
			period: Some(14),
			ma_type: None,
			nbdev: None,
			devtype: None,
		};
		let input = ZscoreInput::from_slice(&single_point, params);
		let res = zscore_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] Zscore should fail with insufficient data",
			test_name
		);
		Ok(())
	}
	fn check_zscore_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let input_data = [f64::NAN, f64::NAN, f64::NAN];
		let params = ZscoreParams::default();
		let input = ZscoreInput::from_slice(&input_data, params);
		let res = zscore_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] Zscore should fail when all values are NaN",
			test_name
		);
		Ok(())
	}
	fn check_zscore_input_with_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = ZscoreInput::with_default_candles(&candles);
		match input.data {
			ZscoreData::Candles { source, .. } => assert_eq!(source, "close"),
			_ => panic!("Expected ZscoreData::Candles"),
		}
		Ok(())
	}
	macro_rules! generate_all_zscore_tests {
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
	#[cfg(debug_assertions)]
	fn check_zscore_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		
		// Define comprehensive parameter combinations
		let test_params = vec![
			// Default parameters
			ZscoreParams::default(),
			// Minimum viable period
			ZscoreParams {
				period: Some(2),
				ma_type: Some("sma".to_string()),
				nbdev: Some(1.0),
				devtype: Some(0),
			},
			// Small periods with different MA types
			ZscoreParams {
				period: Some(5),
				ma_type: Some("ema".to_string()),
				nbdev: Some(1.0),
				devtype: Some(0),
			},
			ZscoreParams {
				period: Some(10),
				ma_type: Some("wma".to_string()),
				nbdev: Some(2.0),
				devtype: Some(0),
			},
			// Medium periods with different deviation types
			ZscoreParams {
				period: Some(20),
				ma_type: Some("sma".to_string()),
				nbdev: Some(1.5),
				devtype: Some(1), // mean abs dev
			},
			ZscoreParams {
				period: Some(30),
				ma_type: Some("ema".to_string()),
				nbdev: Some(2.5),
				devtype: Some(2), // median abs dev
			},
			// Large periods
			ZscoreParams {
				period: Some(50),
				ma_type: Some("wma".to_string()),
				nbdev: Some(3.0),
				devtype: Some(0),
			},
			ZscoreParams {
				period: Some(100),
				ma_type: Some("sma".to_string()),
				nbdev: Some(1.0),
				devtype: Some(1),
			},
			// Edge cases with different nbdev values
			ZscoreParams {
				period: Some(14),
				ma_type: Some("ema".to_string()),
				nbdev: Some(0.5),
				devtype: Some(0),
			},
			ZscoreParams {
				period: Some(14),
				ma_type: Some("sma".to_string()),
				nbdev: Some(0.1),
				devtype: Some(2),
			},
			ZscoreParams {
				period: Some(25),
				ma_type: Some("wma".to_string()),
				nbdev: Some(4.0),
				devtype: Some(1),
			},
			// Mixed parameter combinations
			ZscoreParams {
				period: Some(7),
				ma_type: Some("ema".to_string()),
				nbdev: Some(1.618), // golden ratio
				devtype: Some(0),
			},
			ZscoreParams {
				period: Some(21),
				ma_type: Some("sma".to_string()),
				nbdev: Some(2.718), // e
				devtype: Some(1),
			},
			ZscoreParams {
				period: Some(42),
				ma_type: Some("wma".to_string()),
				nbdev: Some(3.14159), // pi
				devtype: Some(2),
			},
		];
		
		for (param_idx, params) in test_params.iter().enumerate() {
			let input = ZscoreInput::from_candles(&candles, "close", params.clone());
			let output = zscore_with_kernel(&input, kernel)?;
			
			for (i, &val) in output.values.iter().enumerate() {
				if val.is_nan() {
					continue; // NaN values are expected during warmup
				}
				
				let bits = val.to_bits();
				
				// Check all three poison patterns
				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: period={}, ma_type={}, nbdev={}, devtype={} (param set {})",
						test_name, val, bits, i, 
						params.period.unwrap_or(14),
						params.ma_type.as_deref().unwrap_or("sma"),
						params.nbdev.unwrap_or(1.0),
						params.devtype.unwrap_or(0),
						param_idx
					);
				}
				
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: period={}, ma_type={}, nbdev={}, devtype={} (param set {})",
						test_name, val, bits, i,
						params.period.unwrap_or(14),
						params.ma_type.as_deref().unwrap_or("sma"),
						params.nbdev.unwrap_or(1.0),
						params.devtype.unwrap_or(0),
						param_idx
					);
				}
				
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: period={}, ma_type={}, nbdev={}, devtype={} (param set {})",
						test_name, val, bits, i,
						params.period.unwrap_or(14),
						params.ma_type.as_deref().unwrap_or("sma"),
						params.nbdev.unwrap_or(1.0),
						params.devtype.unwrap_or(0),
						param_idx
					);
				}
			}
		}
		
		Ok(())
	}
	
	#[cfg(not(debug_assertions))]
	fn check_zscore_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(()) // No-op in release builds
	}
	
	generate_all_zscore_tests!(
		check_zscore_partial_params,
		check_zscore_with_zero_period,
		check_zscore_period_exceeds_length,
		check_zscore_very_small_dataset,
		check_zscore_all_nan,
		check_zscore_input_with_default_candles,
		check_zscore_no_poison
	);
	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let output = ZscoreBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;
		let def = ZscoreParams::default();
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
	#[cfg(debug_assertions)]
	fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		
		// Test various parameter sweep configurations
		let test_configs = vec![
			// (period_start, period_end, period_step, nbdev_start, nbdev_end, nbdev_step, devtype)
			(2, 10, 2, 0.5, 2.0, 0.5, 0),       // Small periods, standard deviation
			(5, 25, 5, 1.0, 3.0, 1.0, 1),       // Medium periods, mean abs dev
			(10, 50, 10, 1.5, 3.5, 1.0, 2),     // Large periods, median abs dev
			(2, 5, 1, 0.1, 1.0, 0.3, 0),        // Dense small range
			(14, 14, 0, 1.0, 4.0, 0.5, 0),      // Single period, multiple nbdev
			(20, 40, 10, 2.0, 2.0, 0.0, 1),     // Multiple periods, single nbdev
		];
		
		// Test with different MA types
		let ma_types = vec!["sma", "ema", "wma"];
		
		for (cfg_idx, &(period_start, period_end, period_step, nbdev_start, nbdev_end, nbdev_step, devtype)) in test_configs.iter().enumerate() {
			for ma_type in &ma_types {
				let mut builder = ZscoreBatchBuilder::new().kernel(kernel);
				
				// Configure period range
				if period_step > 0 {
					builder = builder.period_range(period_start, period_end, period_step);
				} else {
					builder = builder.period_static(period_start);
				}
				
				// Configure nbdev range
				if nbdev_step > 0.0 {
					builder = builder.nbdev_range(nbdev_start, nbdev_end, nbdev_step);
				} else {
					builder = builder.nbdev_static(nbdev_start);
				}
				
				// Configure MA type
				builder = builder.ma_type_static(ma_type.to_string());
				
				// Configure devtype
				builder = builder.devtype_static(devtype);
				
				let output = builder.apply_candles(&c, "close")?;
				
				for (idx, &val) in output.values.iter().enumerate() {
					if val.is_nan() {
						continue;
					}
					
					let bits = val.to_bits();
					let row = idx / output.cols;
					let col = idx % output.cols;
					let combo = &output.combos[row];
					
					// Check all three poison patterns with detailed context
					if bits == 0x11111111_11111111 {
						panic!(
							"[{}] Config {} (MA: {}): Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
							 at row {} col {} (flat index {}) with params: period={}, ma_type={}, nbdev={}, devtype={}",
							test, cfg_idx, ma_type, val, bits, row, col, idx,
							combo.period.unwrap_or(14),
							combo.ma_type.as_deref().unwrap_or("sma"),
							combo.nbdev.unwrap_or(1.0),
							combo.devtype.unwrap_or(0)
						);
					}
					
					if bits == 0x22222222_22222222 {
						panic!(
							"[{}] Config {} (MA: {}): Found init_matrix_prefixes poison value {} (0x{:016X}) \
							 at row {} col {} (flat index {}) with params: period={}, ma_type={}, nbdev={}, devtype={}",
							test, cfg_idx, ma_type, val, bits, row, col, idx,
							combo.period.unwrap_or(14),
							combo.ma_type.as_deref().unwrap_or("sma"),
							combo.nbdev.unwrap_or(1.0),
							combo.devtype.unwrap_or(0)
						);
					}
					
					if bits == 0x33333333_33333333 {
						panic!(
							"[{}] Config {} (MA: {}): Found make_uninit_matrix poison value {} (0x{:016X}) \
							 at row {} col {} (flat index {}) with params: period={}, ma_type={}, nbdev={}, devtype={}",
							test, cfg_idx, ma_type, val, bits, row, col, idx,
							combo.period.unwrap_or(14),
							combo.ma_type.as_deref().unwrap_or("sma"),
							combo.nbdev.unwrap_or(1.0),
							combo.devtype.unwrap_or(0)
						);
					}
				}
			}
		}
		
		// Additional test with all devtypes
		let devtype_test = ZscoreBatchBuilder::new()
			.kernel(kernel)
			.period_range(10, 30, 10)
			.nbdev_static(2.0)
			.ma_type_static("ema")
			.devtype_range(0, 2, 1)
			.apply_candles(&c, "close")?;
		
		for (idx, &val) in devtype_test.values.iter().enumerate() {
			if val.is_nan() {
				continue;
			}
			
			let bits = val.to_bits();
			let row = idx / devtype_test.cols;
			let col = idx % devtype_test.cols;
			let combo = &devtype_test.combos[row];
			
			if bits == 0x11111111_11111111 || bits == 0x22222222_22222222 || bits == 0x33333333_33333333 {
				let poison_type = if bits == 0x11111111_11111111 {
					"alloc_with_nan_prefix"
				} else if bits == 0x22222222_22222222 {
					"init_matrix_prefixes"
				} else {
					"make_uninit_matrix"
				};
				
				panic!(
					"[{}] Devtype test: Found {} poison value {} (0x{:016X}) \
					 at row {} col {} (flat index {}) with params: period={}, ma_type={}, nbdev={}, devtype={}",
					test, poison_type, val, bits, row, col, idx,
					combo.period.unwrap_or(14),
					combo.ma_type.as_deref().unwrap_or("sma"),
					combo.nbdev.unwrap_or(1.0),
					combo.devtype.unwrap_or(0)
				);
			}
		}
		
		Ok(())
	}
	
	#[cfg(not(debug_assertions))]
	fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(()) // No-op in release builds
	}
	
	gen_batch_tests!(check_batch_default_row);
	gen_batch_tests!(check_batch_no_poison);
}
