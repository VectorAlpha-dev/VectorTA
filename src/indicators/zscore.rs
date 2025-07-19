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

use crate::indicators::deviation::{deviation, DevError, DevInput, DevParams, DeviationOutput};
use crate::indicators::moving_averages::ma::{ma, MaData};
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
	let means = ma(ma_type, MaData::Slice(data), period).unwrap_or_else(|_| vec![f64::NAN; data.len() - first]);
	let dev_input = DevInput {
		data,
		params: DevParams {
			period: Some(period),
			devtype: Some(devtype),
		},
	};
	let mut sigmas = deviation(&dev_input)
		.unwrap_or_else(|_| DeviationOutput {
			values: vec![f64::NAN; data.len() - first],
		})
		.values;
	for v in &mut sigmas {
		*v *= nbdev;
	}
	let mut out = vec![f64::NAN; data.len()];
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
		let means =
			ma(&self.ma_type, MaData::Slice(&ordered), self.period).unwrap_or_else(|_| vec![f64::NAN; self.period]);
		let dev_input = DevInput {
			data: &ordered,
			params: DevParams {
				period: Some(self.period),
				devtype: Some(self.devtype),
			},
		};
		let mut sigmas = deviation(&dev_input)
			.unwrap_or_else(|_| DeviationOutput {
				values: vec![f64::NAN; self.period],
			})
			.values;
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
	let mut values = vec![f64::NAN; rows * cols];
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
	let means = ma(ma_type, MaData::Slice(data), period).unwrap_or_else(|_| vec![f64::NAN; data.len() - first]);
	let dev_input = DevInput {
		data,
		params: DevParams {
			period: Some(period),
			devtype: Some(devtype),
		},
	};
	let mut sigmas = deviation(&dev_input)
		.unwrap_or_else(|_| DeviationOutput {
			values: vec![f64::NAN; data.len() - first],
		})
		.values;
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
	generate_all_zscore_tests!(
		check_zscore_partial_params,
		check_zscore_with_zero_period,
		check_zscore_period_exceeds_length,
		check_zscore_very_small_dataset,
		check_zscore_all_nan,
		check_zscore_input_with_default_candles
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
	gen_batch_tests!(check_batch_default_row);
}
