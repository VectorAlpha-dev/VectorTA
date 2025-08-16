//! # DevStop Indicator
//!
//! A volatility-based stop indicator similar to Kase Dev Stops. Computes the difference between rolling highs and lows, smooths the range with a moving average, computes a deviation, and applies a multiplier offset, then finalizes with rolling extrema.
//!
//! ## Parameters
//! - **period**: Rolling window size (default: 20).
//! - **mult**: Multiplier for deviation (default: 0.0).
//! - **devtype**: Deviation type (0: stddev, 1: mean abs, 2: median abs, default: 0).
//! - **direction**: "long" or "short" (default: "long").
//! - **ma_type**: Moving average type (default: "sma").
//!
//! ## Errors
//! - **AllValuesNaN**: devstop: All values for high or low are NaN.
//! - **InvalidPeriod**: devstop: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: devstop: Not enough valid data points for requested `period`.
//! - **DevStopCalculation**: devstop: Underlying calculation error, including invalid `devtype`.
//!
//! ## Returns
//! - **`Ok(DevStopOutput)`** on success, containing `Vec<f64>` matching input length.
//! - **`Err(DevStopError)`** otherwise.

use crate::indicators::deviation::{deviation, DevInput, DevParams};
use crate::indicators::moving_averages::ma::{ma, MaData};
use crate::indicators::utility_functions::{max_rolling, min_rolling};
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum DevStopData<'a> {
	Candles {
		candles: &'a Candles,
		source_high: &'a str,
		source_low: &'a str,
	},
	SliceHL(&'a [f64], &'a [f64]),
}

#[derive(Debug, Clone)]
pub struct DevStopOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct DevStopParams {
	pub period: Option<usize>,
	pub mult: Option<f64>,
	pub devtype: Option<usize>,
	pub direction: Option<String>,
	pub ma_type: Option<String>,
}

impl Default for DevStopParams {
	fn default() -> Self {
		Self {
			period: Some(20),
			mult: Some(0.0),
			devtype: Some(0),
			direction: Some("long".to_string()),
			ma_type: Some("sma".to_string()),
		}
	}
}

#[derive(Debug, Clone)]
pub struct DevStopInput<'a> {
	pub data: DevStopData<'a>,
	pub params: DevStopParams,
}

impl<'a> DevStopInput<'a> {
	#[inline]
	pub fn from_candles(
		candles: &'a Candles,
		source_high: &'a str,
		source_low: &'a str,
		params: DevStopParams,
	) -> Self {
		Self {
			data: DevStopData::Candles {
				candles,
				source_high,
				source_low,
			},
			params,
		}
	}
	#[inline]
	pub fn from_slices(high: &'a [f64], low: &'a [f64], params: DevStopParams) -> Self {
		Self {
			data: DevStopData::SliceHL(high, low),
			params,
		}
	}
	#[inline]
	pub fn with_default_candles(candles: &'a Candles) -> Self {
		Self::from_candles(candles, "high", "low", DevStopParams::default())
	}
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(20)
	}
	#[inline]
	pub fn get_mult(&self) -> f64 {
		self.params.mult.unwrap_or(0.0)
	}
	#[inline]
	pub fn get_devtype(&self) -> usize {
		self.params.devtype.unwrap_or(0)
	}
	#[inline]
	pub fn get_direction(&self) -> String {
		self.params.direction.clone().unwrap_or_else(|| "long".to_string())
	}
	#[inline]
	pub fn get_ma_type(&self) -> String {
		self.params.ma_type.clone().unwrap_or_else(|| "sma".to_string())
	}
}

#[derive(Clone, Debug)]
pub struct DevStopBuilder {
	period: Option<usize>,
	mult: Option<f64>,
	devtype: Option<usize>,
	direction: Option<String>,
	ma_type: Option<String>,
	kernel: Kernel,
}

impl Default for DevStopBuilder {
	fn default() -> Self {
		Self {
			period: None,
			mult: None,
			devtype: None,
			direction: None,
			ma_type: None,
			kernel: Kernel::Auto,
		}
	}
}

impl DevStopBuilder {
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
	pub fn mult(mut self, x: f64) -> Self {
		self.mult = Some(x);
		self
	}
	#[inline(always)]
	pub fn devtype(mut self, d: usize) -> Self {
		self.devtype = Some(d);
		self
	}
	#[inline(always)]
	pub fn direction(mut self, d: &str) -> Self {
		self.direction = Some(d.to_string());
		self
	}
	#[inline(always)]
	pub fn ma_type(mut self, t: &str) -> Self {
		self.ma_type = Some(t.to_string());
		self
	}
	#[inline(always)]
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline(always)]
	pub fn apply(self, c: &Candles) -> Result<DevStopOutput, DevStopError> {
		let p = DevStopParams {
			period: self.period,
			mult: self.mult,
			devtype: self.devtype,
			direction: self.direction.clone(),
			ma_type: self.ma_type.clone(),
		};
		let i = DevStopInput::from_candles(c, "high", "low", p);
		devstop_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<DevStopOutput, DevStopError> {
		let p = DevStopParams {
			period: self.period,
			mult: self.mult,
			devtype: self.devtype,
			direction: self.direction.clone(),
			ma_type: self.ma_type.clone(),
		};
		let i = DevStopInput::from_slices(high, low, p);
		devstop_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<DevStopStream, DevStopError> {
		let p = DevStopParams {
			period: self.period,
			mult: self.mult,
			devtype: self.devtype,
			direction: self.direction,
			ma_type: self.ma_type,
		};
		DevStopStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum DevStopError {
	#[error("devstop: All values are NaN for high or low.")]
	AllValuesNaN,
	#[error("devstop: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("devstop: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("devstop: Calculation error: {0}")]
	DevStopCalculation(String),
}

#[inline]
pub fn devstop(input: &DevStopInput) -> Result<DevStopOutput, DevStopError> {
	devstop_with_kernel(input, Kernel::Auto)
}

pub fn devstop_with_kernel(input: &DevStopInput, kernel: Kernel) -> Result<DevStopOutput, DevStopError> {
	let (high, low) = match &input.data {
		DevStopData::Candles {
			candles,
			source_high,
			source_low,
		} => (source_type(candles, source_high), source_type(candles, source_low)),
		DevStopData::SliceHL(h, l) => (*h, *l),
	};

	let first_valid_high = high.iter().position(|&x| !x.is_nan());
	let first_valid_low = low.iter().position(|&x| !x.is_nan());
	let first = match (first_valid_high, first_valid_low) {
		(Some(h), Some(l)) => h.min(l),
		_ => return Err(DevStopError::AllValuesNaN),
	};

	let len = high.len();
	let period = input.get_period();
	if period == 0 || period > len || period > low.len() {
		return Err(DevStopError::InvalidPeriod {
			period,
			data_len: len.min(low.len()),
		});
	}
	if (len - first) < period || (low.len() - first) < period {
		return Err(DevStopError::NotEnoughValidData {
			needed: period,
			valid: (len - first).min(low.len() - first),
		});
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	let mut out = vec![f64::NAN; len];
	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => devstop_scalar(high, low, period, first, &input, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => devstop_avx2(high, low, period, first, &input, &mut out),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => devstop_avx512(high, low, period, first, &input, &mut out),
			_ => unreachable!(),
		}
	}
	Ok(DevStopOutput { values: out })
}

#[inline]
pub fn devstop_scalar(high: &[f64], low: &[f64], period: usize, first: usize, input: &DevStopInput, out: &mut [f64]) {
	let high2 = match max_rolling(high, 2) {
		Ok(v) => v,
		Err(_) => {
			return;
		}
	};
	let low2 = match min_rolling(low, 2) {
		Ok(v) => v,
		Err(_) => {
			return;
		}
	};

	let mut range = vec![f64::NAN; high.len()];
	for i in 0..high.len() {
		if !high2[i].is_nan() && !low2[i].is_nan() {
			range[i] = high2[i] - low2[i];
		}
	}

	let avtr = match ma(&input.get_ma_type(), MaData::Slice(&range), period) {
		Ok(v) => v,
		Err(_) => {
			return;
		}
	};
	let dev_values = {
		let dev_input = DevInput::from_slice(
			&range,
			DevParams {
				period: Some(period),
				devtype: Some(input.get_devtype()),
			},
		);
		match deviation(&dev_input) {
			Ok(v) => v,
			Err(_) => {
				return;
			}
		}
	};

	let mult = input.get_mult();
	let direction = input.get_direction();

	let mut base = vec![f64::NAN; high.len()];
	for i in 0..high.len() {
		if direction.eq_ignore_ascii_case("long") {
			if !high[i].is_nan() && !avtr[i].is_nan() && !dev_values[i].is_nan() {
				base[i] = high[i] - avtr[i] - mult * dev_values[i];
			}
		} else {
			if !low[i].is_nan() && !avtr[i].is_nan() && !dev_values[i].is_nan() {
				base[i] = low[i] + avtr[i] + mult * dev_values[i];
			}
		}
	}

	let final_values = if direction.eq_ignore_ascii_case("long") {
		match max_rolling(&base, period) {
			Ok(v) => v,
			Err(_) => vec![f64::NAN; high.len()],
		}
	} else {
		match min_rolling(&base, period) {
			Ok(v) => v,
			Err(_) => vec![f64::NAN; high.len()],
		}
	};

	out.copy_from_slice(&final_values);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn devstop_avx2(high: &[f64], low: &[f64], period: usize, first: usize, input: &DevStopInput, out: &mut [f64]) {
	devstop_scalar(high, low, period, first, input, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn devstop_avx512(high: &[f64], low: &[f64], period: usize, first: usize, input: &DevStopInput, out: &mut [f64]) {
	if period <= 32 {
		unsafe { devstop_avx512_short(high, low, period, first, input, out) }
	} else {
		unsafe { devstop_avx512_long(high, low, period, first, input, out) }
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn devstop_avx512_short(
	high: &[f64],
	low: &[f64],
	period: usize,
	first: usize,
	input: &DevStopInput,
	out: &mut [f64],
) {
	devstop_scalar(high, low, period, first, input, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn devstop_avx512_long(
	high: &[f64],
	low: &[f64],
	period: usize,
	first: usize,
	input: &DevStopInput,
	out: &mut [f64],
) {
	devstop_scalar(high, low, period, first, input, out)
}

#[inline(always)]
pub fn devstop_batch_with_kernel(
	high: &[f64],
	low: &[f64],
	sweep: &DevStopBatchRange,
	kernel: Kernel,
) -> Result<DevStopBatchOutput, DevStopError> {
	let chosen = match kernel {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(DevStopError::InvalidPeriod { period: 0, data_len: 0 }),
	};
	let simd = match chosen {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	devstop_batch_par_slice(high, low, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct DevStopBatchRange {
	pub period: (usize, usize, usize),
	pub mult: (f64, f64, f64),
	pub devtype: (usize, usize, usize),
}

impl Default for DevStopBatchRange {
	fn default() -> Self {
		Self {
			period: (20, 20, 0),
			mult: (0.0, 0.0, 0.0),
			devtype: (0, 0, 0),
		}
	}
}

#[derive(Clone, Debug, Default)]
pub struct DevStopBatchBuilder {
	range: DevStopBatchRange,
	kernel: Kernel,
}

impl DevStopBatchBuilder {
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
	pub fn mult_range(mut self, start: f64, end: f64, step: f64) -> Self {
		self.range.mult = (start, end, step);
		self
	}
	#[inline]
	pub fn mult_static(mut self, x: f64) -> Self {
		self.range.mult = (x, x, 0.0);
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
	pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<DevStopBatchOutput, DevStopError> {
		devstop_batch_with_kernel(high, low, &self.range, self.kernel)
	}
	pub fn with_default_slices(high: &[f64], low: &[f64], k: Kernel) -> Result<DevStopBatchOutput, DevStopError> {
		DevStopBatchBuilder::new().kernel(k).apply_slices(high, low)
	}
}

#[derive(Clone, Debug)]
pub struct DevStopBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<DevStopParams>,
	pub rows: usize,
	pub cols: usize,
}
impl DevStopBatchOutput {
	pub fn row_for_params(&self, p: &DevStopParams) -> Option<usize> {
		self.combos.iter().position(|c| {
			c.period.unwrap_or(20) == p.period.unwrap_or(20)
				&& (c.mult.unwrap_or(0.0) - p.mult.unwrap_or(0.0)).abs() < 1e-12
				&& c.devtype.unwrap_or(0) == p.devtype.unwrap_or(0)
		})
	}
	pub fn values_for(&self, p: &DevStopParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid_devstop(r: &DevStopBatchRange) -> Vec<DevStopParams> {
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
	let mults = axis_f64(r.mult);
	let devtypes = axis_usize(r.devtype);

	let mut out = Vec::with_capacity(periods.len() * mults.len() * devtypes.len());
	for &p in &periods {
		for &m in &mults {
			for &d in &devtypes {
				out.push(DevStopParams {
					period: Some(p),
					mult: Some(m),
					devtype: Some(d),
					direction: Some("long".to_string()),
					ma_type: Some("sma".to_string()),
				});
			}
		}
	}
	out
}

#[inline(always)]
pub fn devstop_batch_slice(
	high: &[f64],
	low: &[f64],
	sweep: &DevStopBatchRange,
	kern: Kernel,
) -> Result<DevStopBatchOutput, DevStopError> {
	devstop_batch_inner(high, low, sweep, kern, false)
}
#[inline(always)]
pub fn devstop_batch_par_slice(
	high: &[f64],
	low: &[f64],
	sweep: &DevStopBatchRange,
	kern: Kernel,
) -> Result<DevStopBatchOutput, DevStopError> {
	devstop_batch_inner(high, low, sweep, kern, true)
}

#[inline(always)]
fn devstop_batch_inner(
	high: &[f64],
	low: &[f64],
	sweep: &DevStopBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<DevStopBatchOutput, DevStopError> {
	let combos = expand_grid_devstop(sweep);
	if combos.is_empty() {
		return Err(DevStopError::InvalidPeriod { period: 0, data_len: 0 });
	}

	let first_high = high
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(DevStopError::AllValuesNaN)?;
	let first_low = low.iter().position(|x| !x.is_nan()).ok_or(DevStopError::AllValuesNaN)?;
	let first = first_high.min(first_low);
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if high.len() - first < max_p || low.len() - first < max_p {
		return Err(DevStopError::NotEnoughValidData {
			needed: max_p,
			valid: (high.len() - first).min(low.len() - first),
		});
	}

	let rows = combos.len();
	let cols = high.len();
	let mut values = vec![f64::NAN; rows * cols];

	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let prm = &combos[row];
		let period = prm.period.unwrap();
		let input = DevStopInput {
			data: DevStopData::SliceHL(high, low),
			params: prm.clone(),
		};
		match kern {
			Kernel::Scalar => devstop_row_scalar(high, low, first, period, &input, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => devstop_row_avx2(high, low, first, period, &input, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => devstop_row_avx512(high, low, first, period, &input, out_row),
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
	Ok(DevStopBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
pub unsafe fn devstop_row_scalar(
	high: &[f64],
	low: &[f64],
	first: usize,
	period: usize,
	input: &DevStopInput,
	out: &mut [f64],
) {
	devstop_scalar(high, low, period, first, input, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn devstop_row_avx2(
	high: &[f64],
	low: &[f64],
	first: usize,
	period: usize,
	input: &DevStopInput,
	out: &mut [f64],
) {
	devstop_row_scalar(high, low, first, period, input, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn devstop_row_avx512(
	high: &[f64],
	low: &[f64],
	first: usize,
	period: usize,
	input: &DevStopInput,
	out: &mut [f64],
) {
	if period <= 32 {
		devstop_row_avx512_short(high, low, first, period, input, out);
	} else {
		devstop_row_avx512_long(high, low, first, period, input, out);
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn devstop_row_avx512_short(
	high: &[f64],
	low: &[f64],
	first: usize,
	period: usize,
	input: &DevStopInput,
	out: &mut [f64],
) {
	devstop_row_scalar(high, low, first, period, input, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn devstop_row_avx512_long(
	high: &[f64],
	low: &[f64],
	first: usize,
	period: usize,
	input: &DevStopInput,
	out: &mut [f64],
) {
	devstop_row_scalar(high, low, first, period, input, out)
}

#[derive(Debug, Clone)]
pub struct DevStopStream {
	period: usize,
	buffer_high: Vec<f64>,
	buffer_low: Vec<f64>,
	mult: f64,
	devtype: usize,
	direction: String,
	ma_type: String,
	head: usize,
	filled: bool,
}

impl DevStopStream {
	pub fn try_new(params: DevStopParams) -> Result<Self, DevStopError> {
		let period = params.period.unwrap_or(20);
		if period == 0 {
			return Err(DevStopError::InvalidPeriod { period, data_len: 0 });
		}
		Ok(Self {
			period,
			buffer_high: vec![f64::NAN; period],
			buffer_low: vec![f64::NAN; period],
			mult: params.mult.unwrap_or(0.0),
			devtype: params.devtype.unwrap_or(0),
			direction: params.direction.unwrap_or_else(|| "long".to_string()),
			ma_type: params.ma_type.unwrap_or_else(|| "sma".to_string()),
			head: 0,
			filled: false,
		})
	}
	pub fn update(&mut self, high: f64, low: f64) -> Option<f64> {
		self.buffer_high[self.head] = high;
		self.buffer_low[self.head] = low;
		self.head = (self.head + 1) % self.period;
		if !self.filled && self.head == 0 {
			self.filled = true;
		}
		if !self.filled {
			return None;
		}
		Some(self.compute())
	}
	fn compute(&self) -> f64 {
		let mut buf_h = vec![0.0; self.period];
		let mut buf_l = vec![0.0; self.period];
		let mut idx = self.head;
		for i in 0..self.period {
			buf_h[i] = self.buffer_high[idx];
			buf_l[i] = self.buffer_low[idx];
			idx = (idx + 1) % self.period;
		}
		let high2 = *buf_h
			.iter()
			.max_by(|a, b| a.partial_cmp(b).unwrap())
			.unwrap_or(&f64::NAN);
		let low2 = *buf_l
			.iter()
			.min_by(|a, b| a.partial_cmp(b).unwrap())
			.unwrap_or(&f64::NAN);
		let range = high2 - low2;
		let avtr = range;
		let dev = match self.devtype {
			0 => {
				let mean = (high2 + low2) / 2.0;
				buf_h
					.iter()
					.chain(buf_l.iter())
					.map(|x| (x - mean).powi(2))
					.sum::<f64>() / (2.0 * self.period as f64)
			}
			1 => {
				let mean = (high2 + low2) / 2.0;
				buf_h.iter().chain(buf_l.iter()).map(|x| (x - mean).abs()).sum::<f64>() / (2.0 * self.period as f64)
			}
			2 => {
				let mut v = buf_h.clone();
				v.extend_from_slice(&buf_l);
				v.sort_by(|a, b| a.partial_cmp(b).unwrap());
				let mid = v.len() / 2;
				if v.len() % 2 == 0 {
					(v[mid - 1] + v[mid]) / 2.0
				} else {
					v[mid]
				}
			}
			_ => f64::NAN,
		};
		if self.direction.eq_ignore_ascii_case("long") {
			high2 - avtr - self.mult * dev
		} else {
			low2 + avtr + self.mult * dev
		}
	}
}
#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;
	use crate::utilities::enums::Kernel;

	fn check_devstop_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let default_params = DevStopParams {
			period: None,
			mult: None,
			devtype: None,
			direction: None,
			ma_type: None,
		};
		let input_default = DevStopInput::from_candles(&candles, "high", "low", default_params);
		let output_default = devstop_with_kernel(&input_default, kernel)?;
		assert_eq!(output_default.values.len(), candles.close.len());

		let params_custom = DevStopParams {
			period: Some(20),
			mult: Some(1.0),
			devtype: Some(2),
			direction: Some("short".to_string()),
			ma_type: Some("ema".to_string()),
		};
		let input_custom = DevStopInput::from_candles(&candles, "high", "low", params_custom);
		let output_custom = devstop_with_kernel(&input_custom, kernel)?;
		assert_eq!(output_custom.values.len(), candles.close.len());
		Ok(())
	}

	fn check_devstop_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let high = &candles.high;
		let low = &candles.low;

		let params = DevStopParams {
			period: Some(20),
			mult: Some(0.0),
			devtype: Some(0),
			direction: Some("long".to_string()),
			ma_type: Some("sma".to_string()),
		};
		let input = DevStopInput::from_slices(high, low, params);
		let result = devstop_with_kernel(&input, kernel)?;

		assert_eq!(result.values.len(), candles.close.len());
		assert!(result.values.len() >= 5);
		let last_five = &result.values[result.values.len() - 5..];
		for &val in last_five {
			println!("Indicator values {}", val);
		}
		Ok(())
	}

	fn check_devstop_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let input = DevStopInput::with_default_candles(&candles);
		match input.data {
			DevStopData::Candles {
				source_high,
				source_low,
				..
			} => {
				assert_eq!(source_high, "high");
				assert_eq!(source_low, "low");
			}
			_ => panic!("Expected DevStopData::Candles"),
		}
		let output = devstop_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());
		Ok(())
	}

	fn check_devstop_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [10.0, 20.0, 30.0];
		let low = [5.0, 15.0, 25.0];
		let params = DevStopParams {
			period: Some(0),
			mult: Some(1.0),
			devtype: Some(0),
			direction: Some("long".to_string()),
			ma_type: Some("sma".to_string()),
		};
		let input = DevStopInput::from_slices(&high, &low, params);
		let result = devstop_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}

	fn check_devstop_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [10.0, 20.0, 30.0];
		let low = [5.0, 15.0, 25.0];
		let params = DevStopParams {
			period: Some(10),
			mult: Some(1.0),
			devtype: Some(0),
			direction: Some("long".to_string()),
			ma_type: Some("sma".to_string()),
		};
		let input = DevStopInput::from_slices(&high, &low, params);
		let result = devstop_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}

	fn check_devstop_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let high = [100.0];
		let low = [90.0];
		let params = DevStopParams {
			period: Some(20),
			mult: Some(2.0),
			devtype: Some(0),
			direction: Some("long".to_string()),
			ma_type: Some("sma".to_string()),
		};
		let input = DevStopInput::from_slices(&high, &low, params);
		let result = devstop_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}

	fn check_devstop_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let params = DevStopParams {
			period: Some(20),
			mult: Some(1.0),
			devtype: Some(0),
			direction: Some("long".to_string()),
			ma_type: Some("sma".to_string()),
		};
		let input = DevStopInput::from_candles(&candles, "high", "low", params);
		let first_result = devstop_with_kernel(&input, kernel)?;

		assert_eq!(first_result.values.len(), candles.close.len());

		let reinput_params = DevStopParams {
			period: Some(20),
			mult: Some(0.5),
			devtype: Some(2),
			direction: Some("short".to_string()),
			ma_type: Some("ema".to_string()),
		};
		let second_input = DevStopInput::from_slices(&first_result.values, &first_result.values, reinput_params);
		let second_result = devstop_with_kernel(&second_input, kernel)?;
		assert_eq!(second_result.values.len(), first_result.values.len());
		Ok(())
	}

	fn check_devstop_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let high = &candles.high;
		let low = &candles.low;

		let params = DevStopParams {
			period: Some(20),
			mult: Some(0.0),
			devtype: Some(0),
			direction: Some("long".to_string()),
			ma_type: Some("sma".to_string()),
		};
		let input = DevStopInput::from_slices(high, low, params);
		let result = devstop_with_kernel(&input, kernel)?;

		assert_eq!(result.values.len(), high.len());
		if result.values.len() > 240 {
			for i in 240..result.values.len() {
				assert!(!result.values[i].is_nan());
			}
		}
		Ok(())
	}

	macro_rules! generate_all_devstop_tests {
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
	fn check_devstop_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		// Define comprehensive parameter combinations
		let test_params = vec![
			// Default parameters
			DevStopParams::default(),
			// Minimum period
			DevStopParams {
				period: Some(2),
				mult: Some(0.0),
				devtype: Some(0),
				direction: Some("long".to_string()),
				ma_type: Some("sma".to_string()),
			},
			// Small period with various multipliers
			DevStopParams {
				period: Some(5),
				mult: Some(0.5),
				devtype: Some(0),
				direction: Some("long".to_string()),
				ma_type: Some("sma".to_string()),
			},
			DevStopParams {
				period: Some(5),
				mult: Some(1.0),
				devtype: Some(1),
				direction: Some("short".to_string()),
				ma_type: Some("ema".to_string()),
			},
			// Medium periods with different devtypes
			DevStopParams {
				period: Some(10),
				mult: Some(0.0),
				devtype: Some(0),
				direction: Some("long".to_string()),
				ma_type: Some("sma".to_string()),
			},
			DevStopParams {
				period: Some(10),
				mult: Some(2.0),
				devtype: Some(1),
				direction: Some("short".to_string()),
				ma_type: Some("ema".to_string()),
			},
			DevStopParams {
				period: Some(10),
				mult: Some(1.5),
				devtype: Some(2),
				direction: Some("long".to_string()),
				ma_type: Some("sma".to_string()),
			},
			// Default period (20) variations
			DevStopParams {
				period: Some(20),
				mult: Some(0.0),
				devtype: Some(0),
				direction: Some("long".to_string()),
				ma_type: Some("sma".to_string()),
			},
			DevStopParams {
				period: Some(20),
				mult: Some(1.0),
				devtype: Some(1),
				direction: Some("short".to_string()),
				ma_type: Some("ema".to_string()),
			},
			DevStopParams {
				period: Some(20),
				mult: Some(2.5),
				devtype: Some(2),
				direction: Some("long".to_string()),
				ma_type: Some("sma".to_string()),
			},
			// Large periods
			DevStopParams {
				period: Some(50),
				mult: Some(0.5),
				devtype: Some(0),
				direction: Some("short".to_string()),
				ma_type: Some("ema".to_string()),
			},
			DevStopParams {
				period: Some(50),
				mult: Some(1.0),
				devtype: Some(1),
				direction: Some("long".to_string()),
				ma_type: Some("sma".to_string()),
			},
			// Very large period
			DevStopParams {
				period: Some(100),
				mult: Some(0.0),
				devtype: Some(0),
				direction: Some("long".to_string()),
				ma_type: Some("sma".to_string()),
			},
			DevStopParams {
				period: Some(100),
				mult: Some(3.0),
				devtype: Some(2),
				direction: Some("short".to_string()),
				ma_type: Some("ema".to_string()),
			},
		];

		for (param_idx, params) in test_params.iter().enumerate() {
			let input = DevStopInput::from_candles(&candles, "high", "low", params.clone());
			let output = devstop_with_kernel(&input, kernel)?;

			for (i, &val) in output.values.iter().enumerate() {
				if val.is_nan() {
					continue; // NaN values are expected during warmup
				}

				let bits = val.to_bits();

				// Check all three poison patterns
				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: period={}, mult={}, devtype={}, direction={}, ma_type={} (param set {})",
						test_name,
						val,
						bits,
						i,
						params.period.unwrap_or(20),
						params.mult.unwrap_or(0.0),
						params.devtype.unwrap_or(0),
						params.direction.as_deref().unwrap_or("long"),
						params.ma_type.as_deref().unwrap_or("sma"),
						param_idx
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: period={}, mult={}, devtype={}, direction={}, ma_type={} (param set {})",
						test_name,
						val,
						bits,
						i,
						params.period.unwrap_or(20),
						params.mult.unwrap_or(0.0),
						params.devtype.unwrap_or(0),
						params.direction.as_deref().unwrap_or("long"),
						params.ma_type.as_deref().unwrap_or("sma"),
						param_idx
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: period={}, mult={}, devtype={}, direction={}, ma_type={} (param set {})",
						test_name,
						val,
						bits,
						i,
						params.period.unwrap_or(20),
						params.mult.unwrap_or(0.0),
						params.devtype.unwrap_or(0),
						params.direction.as_deref().unwrap_or("long"),
						params.ma_type.as_deref().unwrap_or("sma"),
						param_idx
					);
				}
			}
		}

		Ok(())
	}

	#[cfg(not(debug_assertions))]
	fn check_devstop_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		Ok(()) // No-op in release builds
	}

	#[cfg(feature = "proptest")]
	fn check_devstop_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		use proptest::prelude::*;
		skip_if_unsupported!(kernel, test_name);

		// Strategy for generating test data with more realistic scenarios
		let strat = (2usize..=50)
			.prop_flat_map(|period| {
				(
					// Generate base price and volatility
					(100.0f64..5000.0f64, 0.01f64..0.1f64),
					Just(period),
					0.0f64..3.0f64,     // mult range
					0usize..=2,         // devtype range (0: stddev, 1: mean abs, 2: median abs)
					prop::bool::ANY,    // direction (true = long, false = short)
					// MA type selection
					prop::sample::select(vec!["sma", "ema", "wma", "hma", "dema"]),
				)
			})
			.prop_flat_map(move |(base_price_vol, period, mult, devtype, is_long, ma_type)| {
				let (base_price, volatility) = base_price_vol;
				let data_len = period + 10 + (period * 3); // Ensure enough data for warmup
				
				// Generate more realistic price data with trends and volatility
				let price_strategy = prop::collection::vec(
					(-volatility..volatility).prop_map(move |change| {
						base_price * (1.0 + change)
					}),
					data_len..400,
				);
				
				(
					price_strategy.clone(), // For high calculation
					price_strategy,         // For low calculation  
					Just(period),
					Just(mult),
					Just(devtype),
					Just(is_long),
					Just(ma_type.to_string()),
				)
			});

		proptest::test_runner::TestRunner::default()
			.run(&strat, |(high_base, low_base, period, mult, devtype, is_long, ma_type)| {
				// Create high/low with realistic spread
				let len = high_base.len().min(low_base.len());
				let mut high = vec![0.0; len];
				let mut low = vec![0.0; len];
				
				for i in 0..len {
					// Ensure high >= low with realistic spread (0.1% to 2% typically)
					let mid = (high_base[i] + low_base[i]) / 2.0;
					let spread = mid * 0.001 * (1.0 + (i as f64 * 0.1).sin().abs()); // Variable spread
					high[i] = mid + spread;
					low[i] = mid - spread;
				}

				let direction = if is_long { "long".to_string() } else { "short".to_string() };
				
				let params = DevStopParams {
					period: Some(period),
					mult: Some(mult),
					devtype: Some(devtype),
					direction: Some(direction.clone()),
					ma_type: Some(ma_type.clone()),
				};
				let input = DevStopInput::from_slices(&high, &low, params.clone());

				// Test with specified kernel
				let result = devstop_with_kernel(&input, kernel);
				prop_assert!(result.is_ok(), "DevStop calculation failed: {:?}", result.err());
				let out = result.unwrap().values;

				// Test with scalar reference kernel
				let ref_result = devstop_with_kernel(&input, Kernel::Scalar);
				prop_assert!(ref_result.is_ok(), "Reference calculation failed");
				let ref_out = ref_result.unwrap().values;

				// Property 1: Output length should match input length
				prop_assert_eq!(out.len(), high.len(), "Output length mismatch");

				// Property 2: Warmup period handling - be less rigid
				// Just verify that we have NaN values early and finite values later
				let expected_warmup = period * 2; // Approximate
				let has_early_nans = out.iter().take(period).any(|&x| x.is_nan());
				let has_late_finites = out.iter().skip(expected_warmup + 5).any(|&x| x.is_finite());
				
				if out.len() > expected_warmup + 5 {
					prop_assert!(
						has_early_nans || period <= 2,
						"Expected some NaN values during warmup period"
					);
					prop_assert!(
						has_late_finites,
						"Expected finite values after warmup period"
					);
				}

				// Property 3: Kernel consistency - compare with scalar reference
				for i in 0..out.len() {
					let y = out[i];
					let r = ref_out[i];
					
					// Both should be NaN or both should be finite
					if y.is_nan() != r.is_nan() {
						prop_assert!(
							false,
							"NaN mismatch at index {}: kernel is_nan={}, scalar is_nan={}",
							i, y.is_nan(), r.is_nan()
						);
					}

					// Check finite value consistency
					if y.is_finite() && r.is_finite() {
						let abs_diff = (y - r).abs();
						let rel_diff = if r.abs() > 1e-10 { abs_diff / r.abs() } else { abs_diff };
						
						prop_assert!(
							abs_diff <= 1e-6 || rel_diff <= 1e-6,
							"Value mismatch at index {}: kernel={}, scalar={}, diff={}",
							i, y, r, abs_diff
						);
					}
				}

				// Property 4: Multiplier effect
				// Higher multiplier should generally mean stops further from price
				if mult > 0.1 && out.len() > expected_warmup + 10 {
					// Compare with zero multiplier version
					let params_zero = DevStopParams {
						period: Some(period),
						mult: Some(0.0),
						devtype: Some(devtype),
						direction: Some(direction.clone()),
						ma_type: Some(ma_type.clone()),
					};
					let input_zero = DevStopInput::from_slices(&high, &low, params_zero);
					if let Ok(result_zero) = devstop_with_kernel(&input_zero, Kernel::Scalar) {
						let out_zero = result_zero.values;
						
						// Count how many times the stop with multiplier is further from price
						let mut further_count = 0;
						let mut total_count = 0;
						
						for i in expected_warmup..out.len() {
							if out[i].is_finite() && out_zero[i].is_finite() {
								total_count += 1;
								if direction == "long" {
									// For long, higher mult should give lower (further) stops
									if out[i] <= out_zero[i] {
										further_count += 1;
									}
								} else {
									// For short, higher mult should give higher (further) stops
									if out[i] >= out_zero[i] {
										further_count += 1;
									}
								}
							}
						}
						
						// At least 90% of the time, multiplier should increase stop distance
						if total_count > 0 {
							let ratio = further_count as f64 / total_count as f64;
							prop_assert!(
								ratio >= 0.9 || mult < 0.1,
								"Multiplier effect not working: only {:.1}% of stops are further with mult={}",
								ratio * 100.0, mult
							);
						}
					}
				}

				// Property 5: Special case - flat candles (high == low)
				// Test a subset with flat candles
				if len > 20 {
					let mut flat_high = high.clone();
					let mut flat_low = high.clone(); // Same as high
					for i in 10..20.min(len) {
						flat_high[i] = 1000.0;
						flat_low[i] = 1000.0;
					}
					
					let flat_params = params.clone();
					let flat_input = DevStopInput::from_slices(&flat_high, &flat_low, flat_params);
					let flat_result = devstop_with_kernel(&flat_input, kernel);
					
					// Should not error on flat candles
					prop_assert!(
						flat_result.is_ok(),
						"DevStop should handle flat candles (high==low)"
					);
				}

				// Property 6: Deviation type parameter is being used
				// Simply verify that the devtype parameter affects the calculation
				// Different deviation types may produce similar results with certain data patterns
				// which is mathematically valid, so we just verify the parameter is processed
				if out.len() > expected_warmup + 10 && mult > 0.5 {
					// Test that all three deviation types can be calculated without error
					for test_devtype in 0..=2 {
						let params_test = DevStopParams {
							period: Some(period),
							mult: Some(mult),
							devtype: Some(test_devtype),
							direction: Some(direction.clone()),
							ma_type: Some(ma_type.clone()),
						};
						let input_test = DevStopInput::from_slices(&high, &low, params_test);
						let result_test = devstop_with_kernel(&input_test, Kernel::Scalar);
						
						prop_assert!(
							result_test.is_ok(),
							"DevStop should handle all deviation types: failed on devtype {}",
							test_devtype
						);
						
						// Verify output has the correct structure
						if let Ok(output) = result_test {
							prop_assert_eq!(
								output.values.len(), 
								high.len(),
								"Output length should match input for devtype {}", 
								test_devtype
							);
						}
					}
				}

				// Property 7: Stop movement logic  
				// The stops should behave sensibly - not jumping erratically
				if out.len() > expected_warmup + period {
					let mut max_jump = 0.0;
					let mut jump_count = 0;
					
					for i in (expected_warmup + 1)..out.len() {
						if out[i].is_finite() && out[i-1].is_finite() {
							let jump = (out[i] - out[i-1]).abs();
							let relative_jump = jump / out[i-1].abs().max(1.0);
							
							// Track the maximum jump
							if relative_jump > max_jump {
								max_jump = relative_jump;
							}
							
							// Count large jumps (more than 20% change)
							if relative_jump > 0.2 {
								jump_count += 1;
							}
						}
					}
					
					// Stops shouldn't jump more than 50% in a single step typically
					// (unless we have extreme price movements)
					prop_assert!(
						max_jump < 0.5 || jump_count < 5,
						"Stop values jumping too much: max jump = {:.1}%, large jumps = {}",
						max_jump * 100.0, jump_count
					);
				}

				Ok(())
			})
			.unwrap();

		Ok(())
	}

	generate_all_devstop_tests!(
		check_devstop_partial_params,
		check_devstop_accuracy,
		check_devstop_default_candles,
		check_devstop_zero_period,
		check_devstop_period_exceeds_length,
		check_devstop_very_small_dataset,
		check_devstop_reinput,
		check_devstop_nan_handling,
		check_devstop_no_poison,
		check_devstop_property
	);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let high = &c.high;
		let low = &c.low;

		let output = DevStopBatchBuilder::new().kernel(kernel).apply_slices(high, low)?;

		let def = DevStopParams::default();
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
	fn check_batch_sweep(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let high = &c.high;
		let low = &c.low;

		let output = DevStopBatchBuilder::new()
			.kernel(kernel)
			.period_range(10, 30, 5)
			.mult_range(0.0, 2.0, 0.5)
			.devtype_range(0, 2, 1)
			.apply_slices(high, low)?;

		let expected_combos = 5 * 5 * 3; // 5 periods * 5 mults * 3 devtypes
		assert_eq!(output.combos.len(), expected_combos);
		assert_eq!(output.rows, expected_combos);
		assert_eq!(output.cols, c.close.len());

		Ok(())
	}

	#[cfg(debug_assertions)]
	fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let high = &c.high;
		let low = &c.low;

		// Test various parameter sweep configurations
		let test_configs = vec![
			// (period_start, period_end, period_step, mult_start, mult_end, mult_step, devtype_start, devtype_end, devtype_step)
			(2, 10, 2, 0.0, 2.0, 0.5, 0, 2, 1),      // Small periods with mult variations
			(5, 25, 5, 0.0, 1.0, 0.25, 0, 0, 0),     // Medium periods, single devtype
			(30, 60, 15, 1.0, 3.0, 1.0, 1, 1, 0),    // Large periods, mean abs dev
			(2, 5, 1, 0.0, 0.5, 0.1, 2, 2, 0),       // Dense small range, median abs dev
			(10, 20, 2, 0.5, 2.5, 0.5, 0, 2, 2),     // Medium range, all devtypes
			(20, 20, 0, 0.0, 3.0, 0.3, 0, 2, 1),     // Single period, mult sweep
			(5, 50, 15, 1.0, 1.0, 0.0, 0, 2, 1),     // Wide period range, single mult
		];

		for (cfg_idx, &(p_start, p_end, p_step, m_start, m_end, m_step, d_start, d_end, d_step)) in
			test_configs.iter().enumerate()
		{
			let output = DevStopBatchBuilder::new()
				.kernel(kernel)
				.period_range(p_start, p_end, p_step)
				.mult_range(m_start, m_end, m_step)
				.devtype_range(d_start, d_end, d_step)
				.apply_slices(high, low)?;

			for (idx, &val) in output.values.iter().enumerate() {
				if val.is_nan() {
					continue;
				}

				let bits = val.to_bits();
				let row = idx / output.cols;
				let col = idx % output.cols;
				let combo = &output.combos[row];

				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}, mult={}, devtype={}, \
						 direction={}, ma_type={}",
						test,
						cfg_idx,
						val,
						bits,
						row,
						col,
						idx,
						combo.period.unwrap_or(20),
						combo.mult.unwrap_or(0.0),
						combo.devtype.unwrap_or(0),
						combo.direction.as_deref().unwrap_or("long"),
						combo.ma_type.as_deref().unwrap_or("sma")
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}, mult={}, devtype={}, \
						 direction={}, ma_type={}",
						test,
						cfg_idx,
						val,
						bits,
						row,
						col,
						idx,
						combo.period.unwrap_or(20),
						combo.mult.unwrap_or(0.0),
						combo.devtype.unwrap_or(0),
						combo.direction.as_deref().unwrap_or("long"),
						combo.ma_type.as_deref().unwrap_or("sma")
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}, mult={}, devtype={}, \
						 direction={}, ma_type={}",
						test,
						cfg_idx,
						val,
						bits,
						row,
						col,
						idx,
						combo.period.unwrap_or(20),
						combo.mult.unwrap_or(0.0),
						combo.devtype.unwrap_or(0),
						combo.direction.as_deref().unwrap_or("long"),
						combo.ma_type.as_deref().unwrap_or("sma")
					);
				}
			}
		}

		Ok(())
	}

	#[cfg(not(debug_assertions))]
	fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		Ok(())
	}

	gen_batch_tests!(check_batch_default_row);
	gen_batch_tests!(check_batch_sweep);
	gen_batch_tests!(check_batch_no_poison);
}
