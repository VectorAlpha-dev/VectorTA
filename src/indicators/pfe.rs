//! # Polarized Fractal Efficiency (PFE)
//!
//! Measures the efficiency of price movement over a period, producing signed values
//! (positive = upward efficiency, negative = downward), then smooths with EMA.
//!
//! ## Parameters
//! - **period**: Lookback window (default: 10)
//! - **smoothing**: EMA smoothing window (default: 5)
//!
//! ## Errors
//! - **AllValuesNaN**: pfe: All input data values are `NaN`.
//! - **InvalidPeriod**: pfe: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: pfe: Not enough valid data points for the requested `period`.
//!
//! ## Returns
//! - **`Ok(PfeOutput)`**: `Vec<f64>` matching input length (leading NaN for non-computable values)
//! - **`Err(PfeError)`** otherwise
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

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

impl<'a> AsRef<[f64]> for PfeInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			PfeData::Slice(slice) => slice,
			PfeData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

#[derive(Debug, Clone)]
pub enum PfeData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct PfeOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(serde::Serialize, serde::Deserialize))]
pub struct PfeParams {
	pub period: Option<usize>,
	pub smoothing: Option<usize>,
}

impl Default for PfeParams {
	fn default() -> Self {
		Self {
			period: Some(10),
			smoothing: Some(5),
		}
	}
}

#[derive(Debug, Clone)]
pub struct PfeInput<'a> {
	pub data: PfeData<'a>,
	pub params: PfeParams,
}

impl<'a> PfeInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: PfeParams) -> Self {
		Self {
			data: PfeData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slice(sl: &'a [f64], p: PfeParams) -> Self {
		Self {
			data: PfeData::Slice(sl),
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", PfeParams::default())
	}
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(10)
	}
	#[inline]
	pub fn get_smoothing(&self) -> usize {
		self.params.smoothing.unwrap_or(5)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct PfeBuilder {
	period: Option<usize>,
	smoothing: Option<usize>,
	kernel: Kernel,
}

impl Default for PfeBuilder {
	fn default() -> Self {
		Self {
			period: None,
			smoothing: None,
			kernel: Kernel::Auto,
		}
	}
}

impl PfeBuilder {
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
	pub fn smoothing(mut self, s: usize) -> Self {
		self.smoothing = Some(s);
		self
	}
	#[inline(always)]
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline(always)]
	pub fn apply(self, c: &Candles) -> Result<PfeOutput, PfeError> {
		let p = PfeParams {
			period: self.period,
			smoothing: self.smoothing,
		};
		let i = PfeInput::from_candles(c, "close", p);
		pfe_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slice(self, d: &[f64]) -> Result<PfeOutput, PfeError> {
		let p = PfeParams {
			period: self.period,
			smoothing: self.smoothing,
		};
		let i = PfeInput::from_slice(d, p);
		pfe_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<PfeStream, PfeError> {
		let p = PfeParams {
			period: self.period,
			smoothing: self.smoothing,
		};
		PfeStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum PfeError {
	#[error("pfe: Input data slice is empty.")]
	EmptyInputData,
	#[error("pfe: All values are NaN.")]
	AllValuesNaN,
	#[error("pfe: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },
	#[error("pfe: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("pfe: Invalid smoothing: {smoothing}")]
	InvalidSmoothing { smoothing: usize },
}

#[inline(always)]
fn pfe_prepare<'a>(
	input: &'a PfeInput,
	k: Kernel,
) -> Result<(&'a [f64], usize, usize, usize, Kernel), PfeError> {
	let data: &[f64] = input.as_ref();
	let len = data.len();
	if len == 0 {
		return Err(PfeError::EmptyInputData);
	}
	let first = data.iter().position(|x| !x.is_nan()).ok_or(PfeError::AllValuesNaN)?;
	let period = input.get_period();
	let smoothing = input.get_smoothing();

	if period == 0 || period > len {
		return Err(PfeError::InvalidPeriod { period, data_len: len });
	}
	// Earliest valid output index for PFE is first + period (note: not -1).
	if len - first < period + 1 {
		return Err(PfeError::NotEnoughValidData { needed: period + 1, valid: len - first });
	}
	if smoothing == 0 {
		return Err(PfeError::InvalidSmoothing { smoothing });
	}

	let chosen = match k { Kernel::Auto => detect_best_kernel(), other => other };
	Ok((data, period, smoothing, first, chosen))
}

#[inline(always)]
fn pfe_compute_into(
	data: &[f64],
	period: usize,
	smoothing: usize,
	first: usize,
	_kernel: Kernel, // SIMD ignored per instructions
	out: &mut [f64],
) {
	let len = data.len();
	let start = first + period;        // earliest computable t
	let alpha = 2.0 / (smoothing as f64 + 1.0);

	let mut ema_started = false;
	let mut ema_val = 0.0;

	for t in start..len {
		let diff = data[t] - data[t - period];
		let long_leg = (diff.mul_add(diff, (period as f64).powi(2))).sqrt();

		// sum_{k=t-period+1..t} sqrt(1 + (ΔP)^2)
		let mut short_leg = 0.0;
		for k in (t - period + 1)..=t {
			let d = data[k] - data[k - 1];
			short_leg += (1.0 + d * d).sqrt();
		}

		let raw = if short_leg.abs() < f64::EPSILON { 0.0 } else { 100.0 * long_leg / short_leg };
		let signed = if diff.is_nan() { f64::NAN } else if diff > 0.0 { raw } else { -raw };

		let val = if signed.is_nan() {
			f64::NAN
		} else if !ema_started {
			ema_started = true;
			ema_val = signed;
			signed
		} else {
			ema_val = alpha * signed + (1.0 - alpha) * ema_val;
			ema_val
		};

		out[t] = val;
	}
}

#[inline]
pub fn pfe(input: &PfeInput) -> Result<PfeOutput, PfeError> {
	pfe_with_kernel(input, Kernel::Auto)
}

pub fn pfe_with_kernel(input: &PfeInput, kernel: Kernel) -> Result<PfeOutput, PfeError> {
	let (data, period, smoothing, first, chosen) = pfe_prepare(input, kernel)?;
	// Warmup count for PFE is first + period
	let mut out = alloc_with_nan_prefix(data.len(), first + period);
	pfe_compute_into(data, period, smoothing, first, chosen, &mut out);
	Ok(PfeOutput { values: out })
}

#[inline]
pub fn pfe_into_slice(dst: &mut [f64], input: &PfeInput, k: Kernel) -> Result<(), PfeError> {
	let (data, period, smoothing, first, chosen) = pfe_prepare(input, k)?;
	if dst.len() != data.len() {
		return Err(PfeError::InvalidPeriod { period: dst.len(), data_len: data.len() });
	}
	pfe_compute_into(data, period, smoothing, first, chosen, dst);
	// ensure warmup prefix is NaN without bulk fills
	for v in &mut dst[..(first + period)] { *v = f64::NAN; }
	Ok(())
}

// Removed old pfe_scalar and AVX stubs - now using unified pfe_compute_into

#[inline]
pub fn pfe_batch_with_kernel(data: &[f64], sweep: &PfeBatchRange, k: Kernel) -> Result<PfeBatchOutput, PfeError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(PfeError::InvalidPeriod { period: 0, data_len: 0 }),
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	pfe_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct PfeBatchRange {
	pub period: (usize, usize, usize),
	pub smoothing: (usize, usize, usize),
}

impl Default for PfeBatchRange {
	fn default() -> Self {
		Self {
			period: (10, 40, 1),
			smoothing: (5, 10, 1),
		}
	}
}

#[derive(Clone, Debug, Default)]
pub struct PfeBatchBuilder {
	range: PfeBatchRange,
	kernel: Kernel,
}

impl PfeBatchBuilder {
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
	pub fn smoothing_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.smoothing = (start, end, step);
		self
	}
	#[inline]
	pub fn smoothing_static(mut self, s: usize) -> Self {
		self.range.smoothing = (s, s, 0);
		self
	}
	pub fn apply_slice(self, data: &[f64]) -> Result<PfeBatchOutput, PfeError> {
		pfe_batch_with_kernel(data, &self.range, self.kernel)
	}
	pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<PfeBatchOutput, PfeError> {
		PfeBatchBuilder::new().kernel(k).apply_slice(data)
	}
	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<PfeBatchOutput, PfeError> {
		let slice = source_type(c, src);
		self.apply_slice(slice)
	}
	pub fn with_default_candles(c: &Candles) -> Result<PfeBatchOutput, PfeError> {
		PfeBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
	}
}

#[derive(Clone, Debug)]
pub struct PfeBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<PfeParams>,
	pub rows: usize,
	pub cols: usize,
}
impl PfeBatchOutput {
	pub fn row_for_params(&self, p: &PfeParams) -> Option<usize> {
		self.combos.iter().position(|c| {
			c.period.unwrap_or(10) == p.period.unwrap_or(10) && c.smoothing.unwrap_or(5) == p.smoothing.unwrap_or(5)
		})
	}
	pub fn values_for(&self, p: &PfeParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &PfeBatchRange) -> Vec<PfeParams> {
	fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let periods = axis(r.period);
	let smoothings = axis(r.smoothing);

	let mut out = Vec::with_capacity(periods.len() * smoothings.len());
	for &p in &periods {
		for &s in &smoothings {
			out.push(PfeParams {
				period: Some(p),
				smoothing: Some(s),
			});
		}
	}
	out
}

#[inline(always)]
pub fn pfe_batch_slice(data: &[f64], sweep: &PfeBatchRange, kern: Kernel) -> Result<PfeBatchOutput, PfeError> {
	pfe_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn pfe_batch_par_slice(data: &[f64], sweep: &PfeBatchRange, kern: Kernel) -> Result<PfeBatchOutput, PfeError> {
	pfe_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
pub fn pfe_batch_inner_into(
	data: &[f64],
	sweep: &PfeBatchRange,
	kern: Kernel,
	parallel: bool,
	out: &mut [f64],
) -> Result<Vec<PfeParams>, PfeError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(PfeError::InvalidPeriod { period: 0, data_len: 0 });
	}

	let first = data.iter().position(|x| !x.is_nan()).ok_or(PfeError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if data.len() - first < max_p {
		return Err(PfeError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}

	let rows = combos.len();
	let cols = data.len();
	
	// Initialize warmup NaNs for all rows efficiently
	for (row, combo) in combos.iter().enumerate() {
		let warmup = first + combo.period.unwrap();
		let row_start = row * cols;
		for i in 0..warmup {
			out[row_start + i] = f64::NAN;
		}
	}
	
	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let period = combos[row].period.unwrap();
		let smoothing = combos[row].smoothing.unwrap();
		match kern {
			Kernel::Scalar => {
				let out = pfe_row_scalar(data, first, period, smoothing, out_row);
				out;
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => {
				let out = pfe_row_avx2(data, first, period, smoothing, out_row);
				out;
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => {
				let out = pfe_row_avx512(data, first, period, smoothing, out_row);
				out;
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
fn pfe_batch_inner(
	data: &[f64],
	sweep: &PfeBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<PfeBatchOutput, PfeError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(PfeError::InvalidPeriod { period: 0, data_len: 0 });
	}

	let first = data.iter().position(|x| !x.is_nan()).ok_or(PfeError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if data.len() - first < max_p {
		return Err(PfeError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}

	let rows = combos.len();
	let cols = data.len();
	
	let mut buf_mu = make_uninit_matrix(rows, cols);
	let warm: Vec<usize> = combos
		.iter()
		.map(|c| first + c.period.unwrap())
		.collect();
	init_matrix_prefixes(&mut buf_mu, cols, &warm);
	
	let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
	let values_slice: &mut [f64] =
		unsafe { core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, rows * cols) };
	
	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let period = combos[row].period.unwrap();
		let smoothing = combos[row].smoothing.unwrap();
		match kern {
			Kernel::Scalar => {
				let out = pfe_row_scalar(data, first, period, smoothing, out_row);
				out;
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => {
				let out = pfe_row_avx2(data, first, period, smoothing, out_row);
				out;
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => {
				let out = pfe_row_avx512(data, first, period, smoothing, out_row);
				out;
			}
			_ => unreachable!(),
		}
	};

	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			values_slice
				.par_chunks_mut(cols)
				.enumerate()
				.for_each(|(row, slice)| do_row(row, slice));
		}

		#[cfg(target_arch = "wasm32")]
		{
			for (row, slice) in values_slice.chunks_mut(cols).enumerate() {
				do_row(row, slice);
			}
		}
	} else {
		for (row, slice) in values_slice.chunks_mut(cols).enumerate() {
			do_row(row, slice);
		}
	}
	
	let values = unsafe {
		Vec::from_raw_parts(
			buf_guard.as_mut_ptr() as *mut f64,
			buf_guard.len(),
			buf_guard.capacity(),
		)
	};
	
	Ok(PfeBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
unsafe fn pfe_row_scalar(
	data: &[f64],
	first: usize,
	period: usize,
	smoothing: usize,
	out_row: &mut [f64],
) {
	let len = data.len();
	let start = first + period;
	let alpha = 2.0 / (smoothing as f64 + 1.0);

	let mut ema_started = false;
	let mut ema_val = 0.0;

	for t in start..len {
		let diff = data[t] - data[t - period];
		let long_leg = (diff.mul_add(diff, (period as f64).powi(2))).sqrt();

		let mut short_leg = 0.0;
		for k in (t - period + 1)..=t {
			let d = data[k] - data[k - 1];
			short_leg += (1.0 + d * d).sqrt();
		}

		let raw = if short_leg.abs() < f64::EPSILON { 0.0 } else { 100.0 * long_leg / short_leg };
		let signed = if diff.is_nan() { f64::NAN } else if diff > 0.0 { raw } else { -raw };

		let val = if signed.is_nan() {
			f64::NAN
		} else if !ema_started {
			ema_started = true;
			ema_val = signed;
			signed
		} else {
			ema_val = alpha * signed + (1.0 - alpha) * ema_val;
			ema_val
		};

		out_row[t] = val;
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn pfe_row_avx2(data: &[f64], first: usize, period: usize, smoothing: usize, out: &mut [f64]) {
	pfe_row_scalar(data, first, period, smoothing, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn pfe_row_avx512(data: &[f64], first: usize, period: usize, smoothing: usize, out: &mut [f64]) {
	pfe_row_scalar(data, first, period, smoothing, out)
}

use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct PfeStream {
	period: usize,
	smoothing: usize,
	// We keep period+1 elements so that we can always refer to [0] = P_{t-period}
	// and [period] = P_t.  When length < period+1, return None.
	buffer: VecDeque<f64>,

	// EMA state
	ema_val: f64,
	started: bool,
	// Running sum of short leg components
	short_leg_sum: f64,
}

impl PfeStream {
	pub fn try_new(params: PfeParams) -> Result<Self, PfeError> {
		let period = params.period.unwrap_or(10);
		if period == 0 {
			return Err(PfeError::InvalidPeriod { period, data_len: 0 });
		}
		let smoothing = params.smoothing.unwrap_or(5);
		if smoothing == 0 {
			return Err(PfeError::InvalidSmoothing { smoothing });
		}

		Ok(Self {
			period,
			smoothing,
			buffer: VecDeque::with_capacity(period + 1),
			ema_val: 0.0,
			started: false,
			short_leg_sum: 0.0,
		})
	}

	/// Pushes one new price into the stream.  Returns `None` until we have
	/// collected (period+1) values.  Once we have exactly period+1 values,
	/// we compute:
	///   diff = P_t - P_{t-period},
	///   numerator = sqrt(diff² + period²),
	///   denominator = sum_{i=0..period-1} sqrt(1 + (ΔP)^2) over the sliding window,
	///   raw_pfe = 100 * (numerator / denominator) with correct sign,
	///   then EMA-smooth that raw value.
	pub fn update(&mut self, price: f64) -> Option<f64> {
		// 1) Push new price
		let is_full = self.buffer.len() == self.period + 1;
		
		if is_full {
			// Update running sum before modifying buffer
			// Remove the contribution of the oldest segment (buffer[0] to buffer[1])
			let old_diff = self.buffer[1] - self.buffer[0];
			self.short_leg_sum -= (1.0 + old_diff.powi(2)).sqrt();
			
			// Add the contribution of the new segment (last value to new price)
			let last_val = self.buffer[self.period];
			let new_diff = price - last_val;
			self.short_leg_sum += (1.0 + new_diff.powi(2)).sqrt();
			
			// Remove oldest price
			self.buffer.pop_front();
		}
		
		self.buffer.push_back(price);

		// 2) If we don't yet have (period+1) points, return None
		if self.buffer.len() < self.period + 1 {
			return None;
		}
		
		// First time we have enough data - calculate initial sum
		if !is_full {
			self.short_leg_sum = 0.0;
			for i in 0..self.period {
				let p_i = self.buffer[i];
				let p_next = self.buffer[i + 1];
				let step_diff = p_next - p_i;
				self.short_leg_sum += (1.0 + step_diff.powi(2)).sqrt();
			}
		}

		// 3) Now buffer.len() == period+1.  Let:
		//      front = P_{t-period},   // buffer[0]
		//      back  = P_t,            // buffer[period]
		let front = self.buffer[0];
		let back = *self.buffer.get(self.period).unwrap();

		// 4) Compute diff = P_t - P_{t-period}
		let diff = back - front;

		// 5) Long leg = sqrt(diff² + period²)
		let long_leg = (diff.powi(2) + (self.period as f64).powi(2)).sqrt();

		// 6) Short leg is already maintained in self.short_leg_sum
		let short_leg = self.short_leg_sum;

		// 7) raw PFE = 100 * (long_leg / short_leg), or 0 if denominator ≈ 0
		let raw_pfe = if short_leg.abs() < f64::EPSILON {
			0.0
		} else {
			100.0 * long_leg / short_leg
		};

		// 8) Apply sign based on diff
		let signed = if diff > 0.0 { raw_pfe } else { -raw_pfe };

		// 9) EMA‐smooth using alpha = 2/(smoothing+1)
		let alpha = 2.0 / (self.smoothing as f64 + 1.0);
		let out_val = if !self.started {
			// seed the EMA on the first available raw value
			self.ema_val = signed;
			self.started = true;
			signed
		} else {
			self.ema_val = alpha * signed + (1.0 - alpha) * self.ema_val;
			self.ema_val
		};

		Some(out_val)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "pfe")]
#[pyo3(signature = (data, period, smoothing, kernel=None))]
pub fn pfe_py<'py>(
	py: Python<'py>,
	data: PyReadonlyArray1<'py, f64>,
	period: usize,
	smoothing: usize,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
	use numpy::{IntoPyArray, PyArrayMethods};

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, false)?;
	let params = PfeParams {
		period: Some(period),
		smoothing: Some(smoothing),
	};
	let pfe_in = PfeInput::from_slice(slice_in, params);

	let result_vec: Vec<f64> = py
		.allow_threads(|| pfe_with_kernel(&pfe_in, kern).map(|o| o.values))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction(name = "pfe_batch")]
#[pyo3(signature = (data, period_range, smoothing_range, kernel=None))]
pub fn pfe_batch_py<'py>(
	py: Python<'py>,
	data: PyReadonlyArray1<'py, f64>,
	period_range: (usize, usize, usize),
	smoothing_range: (usize, usize, usize),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;

	let slice_in = data.as_slice()?;

	let sweep = PfeBatchRange {
		period: period_range,
		smoothing: smoothing_range,
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
			pfe_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
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
		"smoothings",
		combos
			.iter()
			.map(|p| p.smoothing.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;

	Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "PfeStream")]
pub struct PfeStreamPy {
	stream: PfeStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl PfeStreamPy {
	#[new]
	fn new(period: usize, smoothing: usize) -> PyResult<Self> {
		let params = PfeParams {
			period: Some(period),
			smoothing: Some(smoothing),
		};
		let stream = PfeStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(PfeStreamPy { stream })
	}

	fn update(&mut self, value: f64) -> Option<f64> {
		self.stream.update(value)
	}
}

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn pfe_js(data: &[f64], period: usize, smoothing: usize) -> Result<Vec<f64>, JsValue> {
	let params = PfeParams { period: Some(period), smoothing: Some(smoothing) };
	let input = PfeInput::from_slice(data, params);
	let mut out = vec![0.0; data.len()];
	pfe_into_slice(&mut out, &input, detect_best_kernel()).map_err(|e| JsValue::from_str(&e.to_string()))?;
	Ok(out)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn pfe_alloc(len: usize) -> *mut f64 {
	let mut v = Vec::<f64>::with_capacity(len);
	let p = v.as_mut_ptr();
	std::mem::forget(v);
	p
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn pfe_free(ptr: *mut f64, len: usize) {
	unsafe { let _ = Vec::from_raw_parts(ptr, len, len); }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn pfe_into(in_ptr: *const f64, out_ptr: *mut f64, len: usize, period: usize, smoothing: usize) -> Result<(), JsValue> {
	if in_ptr.is_null() || out_ptr.is_null() { return Err(JsValue::from_str("null pointer")); }
	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);
		let params = PfeParams { period: Some(period), smoothing: Some(smoothing) };
		let input = PfeInput::from_slice(data, params);
		if in_ptr == out_ptr {
			let mut tmp = vec![0.0; len];
			pfe_into_slice(&mut tmp, &input, detect_best_kernel()).map_err(|e| JsValue::from_str(&e.to_string()))?;
			std::slice::from_raw_parts_mut(out_ptr, len).copy_from_slice(&tmp);
		} else {
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			pfe_into_slice(out, &input, detect_best_kernel()).map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct PfeBatchConfig { pub period_range: (usize, usize, usize), pub smoothing_range: (usize, usize, usize) }

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct PfeBatchJsOutput { pub values: Vec<f64>, pub combos: Vec<PfeParams>, pub rows: usize, pub cols: usize }

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = pfe_batch)]
pub fn pfe_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	let cfg: PfeBatchConfig = serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
	let sweep = PfeBatchRange { period: cfg.period_range, smoothing: cfg.smoothing_range };
	let out = pfe_batch_inner(data, &sweep, detect_best_kernel(), false).map_err(|e| JsValue::from_str(&e.to_string()))?;
	let js = PfeBatchJsOutput { values: out.values, combos: out.combos, rows: out.rows, cols: out.cols };
	serde_wasm_bindgen::to_value(&js).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn pfe_batch_into(
	in_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	p_start: usize, p_end: usize, p_step: usize,
	s_start: usize, s_end: usize, s_step: usize,
) -> Result<usize, JsValue> {
	if in_ptr.is_null() || out_ptr.is_null() { return Err(JsValue::from_str("null pointer")); }
	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);
		let sweep = PfeBatchRange { period: (p_start, p_end, p_step), smoothing: (s_start, s_end, s_step) };
		let combos = pfe_batch_inner_into(data, &sweep, detect_best_kernel(), false, std::slice::from_raw_parts_mut(out_ptr, len * expand_grid(&sweep).len()))
			.map_err(|e| JsValue::from_str(&e.to_string()))?;
		Ok(combos.len())
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_pfe_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let default_params = PfeParams {
			period: None,
			smoothing: None,
		};
		let input = PfeInput::from_candles(&candles, "close", default_params);
		let output = pfe_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());

		Ok(())
	}

	fn check_pfe_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let close_prices = &candles.close;

		let params = PfeParams {
			period: Some(10),
			smoothing: Some(5),
		};
		let input = PfeInput::from_candles(&candles, "close", params);
		let pfe_result = pfe_with_kernel(&input, kernel)?;

		assert_eq!(pfe_result.values.len(), close_prices.len());

		let expected_last_five_pfe = [-13.03562252, -11.93979855, -9.94609862, -9.73372410, -14.88374798];
		let start_index = pfe_result.values.len() - 5;
		let result_last_five_pfe = &pfe_result.values[start_index..];
		for (i, &value) in result_last_five_pfe.iter().enumerate() {
			let expected_value = expected_last_five_pfe[i];
			assert!(
				(value - expected_value).abs() < 1e-8,
				"[{}] PFE mismatch at idx {}: got {}, expected {}",
				test_name,
				i,
				value,
				expected_value
			);
		}

		for i in 0..(10 - 1) {
			assert!(pfe_result.values[i].is_nan());
		}

		Ok(())
	}

	fn check_pfe_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let input = PfeInput::with_default_candles(&candles);
		match input.data {
			PfeData::Candles { source, .. } => assert_eq!(source, "close"),
			_ => panic!("Expected PfeData::Candles"),
		}
		let output = pfe_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());

		Ok(())
	}

	fn check_pfe_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let input_data = [10.0, 20.0, 30.0];
		let params = PfeParams {
			period: Some(0),
			smoothing: Some(5),
		};
		let input = PfeInput::from_slice(&input_data, params);
		let res = pfe_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] PFE should fail with zero period", test_name);
		Ok(())
	}

	fn check_pfe_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data_small = [10.0, 20.0, 30.0];
		let params = PfeParams {
			period: Some(10),
			smoothing: Some(2),
		};
		let input = PfeInput::from_slice(&data_small, params);
		let res = pfe_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] PFE should fail with period exceeding length",
			test_name
		);
		Ok(())
	}

	fn check_pfe_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let single_point = [42.0];
		let params = PfeParams {
			period: Some(10),
			smoothing: Some(2),
		};
		let input = PfeInput::from_slice(&single_point, params);
		let res = pfe_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] PFE should fail with insufficient data", test_name);
		Ok(())
	}

	fn check_pfe_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let first_params = PfeParams {
			period: Some(10),
			smoothing: Some(5),
		};
		let first_input = PfeInput::from_candles(&candles, "close", first_params);
		let first_result = pfe_with_kernel(&first_input, kernel)?;

		let second_params = PfeParams {
			period: Some(10),
			smoothing: Some(5),
		};
		let second_input = PfeInput::from_slice(&first_result.values, second_params);
		let second_result = pfe_with_kernel(&second_input, kernel)?;

		assert_eq!(second_result.values.len(), first_result.values.len());
		for i in 20..second_result.values.len() {
			assert!(
				!second_result.values[i].is_nan(),
				"[{}] Expected no NaN after index 20, but found NaN at index {}",
				test_name,
				i
			);
		}
		Ok(())
	}

	fn check_pfe_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let input = PfeInput::from_candles(
			&candles,
			"close",
			PfeParams {
				period: Some(10),
				smoothing: Some(5),
			},
		);
		let res = pfe_with_kernel(&input, kernel)?;
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

	fn check_pfe_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let period = 10;
		let smoothing = 5;

		let input = PfeInput::from_candles(
			&candles,
			"close",
			PfeParams {
				period: Some(period),
				smoothing: Some(smoothing),
			},
		);
		let batch_output = pfe_with_kernel(&input, kernel)?.values;

		let mut stream = PfeStream::try_new(PfeParams {
			period: Some(period),
			smoothing: Some(smoothing),
		})?;

		let mut stream_values = Vec::with_capacity(candles.close.len());
		for &price in &candles.close {
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
				"[{}] PFE streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
				test_name,
				i,
				b,
				s,
				diff
			);
		}
		Ok(())
	}

	#[cfg(debug_assertions)]
	fn check_pfe_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		// Define comprehensive parameter combinations
		let test_params = vec![
			PfeParams::default(),  // period: 10, smoothing: 5
			PfeParams {
				period: Some(2),   // minimum viable period
				smoothing: Some(1), // minimum smoothing
			},
			PfeParams {
				period: Some(5),
				smoothing: Some(2),
			},
			PfeParams {
				period: Some(10),
				smoothing: Some(3),
			},
			PfeParams {
				period: Some(14),
				smoothing: Some(5),
			},
			PfeParams {
				period: Some(20),
				smoothing: Some(5),
			},
			PfeParams {
				period: Some(20),
				smoothing: Some(10),
			},
			PfeParams {
				period: Some(50),  // large period
				smoothing: Some(15),
			},
			PfeParams {
				period: Some(100), // very large period
				smoothing: Some(20),
			},
			PfeParams {
				period: Some(3),
				smoothing: Some(30), // smoothing > period case
			},
		];

		for (param_idx, params) in test_params.iter().enumerate() {
			let input = PfeInput::from_candles(&candles, "close", params.clone());
			let output = pfe_with_kernel(&input, kernel)?;

			for (i, &val) in output.values.iter().enumerate() {
				if val.is_nan() {
					continue; // NaN values are expected during warmup
				}

				let bits = val.to_bits();

				// Check all three poison patterns
				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: period={}, smoothing={} (param set {})",
						test_name, val, bits, i, 
						params.period.unwrap_or(10), 
						params.smoothing.unwrap_or(5), 
						param_idx
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: period={}, smoothing={} (param set {})",
						test_name, val, bits, i, 
						params.period.unwrap_or(10), 
						params.smoothing.unwrap_or(5), 
						param_idx
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: period={}, smoothing={} (param set {})",
						test_name, val, bits, i, 
						params.period.unwrap_or(10), 
						params.smoothing.unwrap_or(5), 
						param_idx
					);
				}
			}
		}

		Ok(())
	}

	#[cfg(not(debug_assertions))]
	fn check_pfe_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(()) // No-op in release builds
	}

	#[cfg(feature = "proptest")]
	#[allow(clippy::float_cmp)]
	fn check_pfe_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		use proptest::prelude::*;
		skip_if_unsupported!(kernel, test_name);

		// More comprehensive data generation strategy
		let strat = (2usize..=64)
			.prop_flat_map(|period| {
				(
					// Generate varied price data - from very small to very large values
					prop::collection::vec(
						prop::strategy::Union::new(vec![
							// Small values
							(0.01f64..10.0f64).boxed(),
							// Medium values  
							(10.0f64..1000.0f64).boxed(),
							// Large values
							(1000.0f64..100000.0f64).boxed(),
						]).prop_filter("finite", |x| x.is_finite()),
						period + 50..400,
					),
					Just(period),
					1usize..=20, // smoothing range
				)
			});

		proptest::test_runner::TestRunner::default()
			.run(&strat, |(data, period, smoothing)| {
				let params = PfeParams {
					period: Some(period),
					smoothing: Some(smoothing),
				};
				let input = PfeInput::from_slice(&data, params);

				let PfeOutput { values: out } = pfe_with_kernel(&input, kernel).unwrap();
				let PfeOutput { values: ref_out } = pfe_with_kernel(&input, Kernel::Scalar).unwrap();

				// Basic structure checks
				prop_assert_eq!(out.len(), data.len(), "[{}] Output length mismatch", test_name);

				// Warmup period validation
				for i in 0..period {
					prop_assert!(
						out[i].is_nan(),
						"[{}] Expected NaN at index {} during warmup (period={})",
						test_name,
						i,
						period
					);
				}

				// Main validation loop with proper floating point tolerance
				for i in period..data.len() {
					let y = out[i];
					let r = ref_out[i];

					// Both NaN or both finite
					if y.is_nan() != r.is_nan() {
						prop_assert!(
							false,
							"[{}] NaN mismatch at index {}: {} vs {}",
							test_name,
							i,
							y,
							r
						);
					}

					if y.is_finite() {
						// Theoretical bounds check
						prop_assert!(
							y >= -100.0 && y <= 100.0,
							"[{}] PFE value {} at index {} out of bounds [-100, 100]",
							test_name,
							y,
							i
						);

						// Kernel consistency with ULP tolerance
						let y_bits = y.to_bits();
						let r_bits = r.to_bits();
						let ulp_diff = y_bits.abs_diff(r_bits);
						
						// Allow small ULP difference for accumulated floating point errors
						prop_assert!(
							(y - r).abs() <= 1e-9 || ulp_diff <= 4,
							"[{}] Kernel mismatch at index {}: {} vs {} (ULP diff: {})",
							test_name,
							i,
							y,
							r,
							ulp_diff
						);
					}
				}

				// Verify PFE calculation logic for specific patterns
				// 1. Perfect straight line up should have high positive efficiency
				let straight_up: Vec<f64> = (0..100).map(|i| 100.0 + i as f64).collect();
				let straight_params = PfeParams {
					period: Some(10),
					smoothing: Some(1), // Minimal smoothing to see raw PFE
				};
				let straight_input = PfeInput::from_slice(&straight_up, straight_params);
				if let Ok(straight_out) = pfe_with_kernel(&straight_input, kernel) {
					// After warmup, straight line should have efficiency near 100
					for i in 15..straight_out.values.len() {
						if straight_out.values[i].is_finite() {
							prop_assert!(
								straight_out.values[i] > 50.0,
								"[{}] Straight line up should have high positive efficiency, got {} at index {}",
								test_name,
								straight_out.values[i],
								i
							);
						}
					}
				}

				// 2. Zigzag pattern should have lower absolute efficiency
				let zigzag: Vec<f64> = (0..100).map(|i| {
					if i % 2 == 0 { 100.0 + (i as f64) } else { 100.0 + (i as f64) - 5.0 }
				}).collect();
				let zigzag_params = PfeParams {
					period: Some(10),
					smoothing: Some(1),
				};
				let zigzag_input = PfeInput::from_slice(&zigzag, zigzag_params.clone());
				let straight_input2 = PfeInput::from_slice(&straight_up, zigzag_params);
				
				if let (Ok(zigzag_out), Ok(straight_out2)) = 
					(pfe_with_kernel(&zigzag_input, kernel), pfe_with_kernel(&straight_input2, kernel)) {
					// Compare average absolute efficiency
					let zigzag_avg: f64 = zigzag_out.values[20..50]
						.iter()
						.filter(|x| x.is_finite())
						.map(|x| x.abs())
						.sum::<f64>() / 30.0;
					let straight_avg: f64 = straight_out2.values[20..50]
						.iter()
						.filter(|x| x.is_finite())
						.map(|x| x.abs())
						.sum::<f64>() / 30.0;
					
					prop_assert!(
						zigzag_avg < straight_avg,
						"[{}] Zigzag pattern should have lower efficiency ({}) than straight line ({})",
						test_name,
						zigzag_avg,
						straight_avg
					);
				}

				// 3. Test EMA smoothing effect
				if smoothing > 1 && data.len() > period + 30 {
					// Calculate unsmoothed version
					let unsmoothed_params = PfeParams {
						period: Some(period),
						smoothing: Some(1),
					};
					let unsmoothed_input = PfeInput::from_slice(&data, unsmoothed_params);
					
					if let Ok(unsmoothed_out) = pfe_with_kernel(&unsmoothed_input, kernel) {
						// Use a larger window for more stable variance calculation
						let window_start = period + 10;
						let window_end = (period + 30).min(out.len());
						
						if window_end > window_start {
							// Calculate variance to compare smoothness
							let smoothed_variance = calculate_variance(&out[window_start..window_end]);
							let unsmoothed_variance = calculate_variance(&unsmoothed_out.values[window_start..window_end]);
							
							// Check for extreme price jumps in the data
							let mut has_extreme_jumps = false;
							for i in 1..data.len() {
								let ratio = if data[i-1] != 0.0 {
									(data[i] / data[i-1]).abs()
								} else {
									f64::INFINITY
								};
								// If any price changes by more than 100x, consider it extreme
								if ratio > 100.0 || ratio < 0.01 {
									has_extreme_jumps = true;
									break;
								}
							}
							
							// Only apply smoothness check if:
							// 1. Both variances are finite
							// 2. There's meaningful variance (not near-constant)
							// 3. Data doesn't have extreme jumps
							// 4. Smoothing parameter is reasonable (not too high)
							if smoothed_variance.is_finite() && unsmoothed_variance.is_finite() 
								&& unsmoothed_variance > 1e-6
								&& !has_extreme_jumps
								&& smoothing <= 10 {
								// For normal data, smoothing should reduce variance
								// Allow up to 50% higher variance as EMA can amplify certain patterns
								prop_assert!(
									smoothed_variance <= unsmoothed_variance * 1.5,
									"[{}] Smoothed variance ({}) should be <= 1.5x unsmoothed variance ({})",
									test_name,
									smoothed_variance,
									unsmoothed_variance
								);
							}
						}
					}
				}

				// 4. Edge case: period = 2 (minimum viable)
				if period == 2 {
					// With period=2, we're only looking at 2 points
					// The calculation should still produce valid results
					for i in 2..out.len() {
						if out[i].is_finite() {
							prop_assert!(
								out[i] >= -100.0 && out[i] <= 100.0,
								"[{}] Period=2 should still produce valid bounded values, got {} at index {}",
								test_name,
								out[i],
								i
							);
						}
					}
				}

				// 5. Constant prices verification - mathematically should be -100
				let constant: Vec<f64> = vec![500.0; 50];
				let const_params = PfeParams {
					period: Some(10),
					smoothing: Some(1), // No smoothing to see raw value
				};
				let const_input = PfeInput::from_slice(&constant, const_params);
				if let Ok(const_out) = pfe_with_kernel(&const_input, kernel) {
					// For constant prices:
					// diff = 0, long_leg = period, short_leg = period * 1 = period
					// raw_pfe = 100 * (period/period) = 100
					// signed_pfe = -100 (because diff <= 0)
					for i in 15..const_out.values.len() {
						if const_out.values[i].is_finite() {
							prop_assert!(
								(const_out.values[i] - (-100.0)).abs() < 1e-6,
								"[{}] Constant prices should produce exactly -100, got {} at index {}",
								test_name,
								const_out.values[i],
								i
							);
						}
					}
				}

				Ok(())
			})
			.unwrap();

		// Helper function for variance calculation
		fn calculate_variance(values: &[f64]) -> f64 {
			let finite_values: Vec<f64> = values.iter()
				.filter(|x| x.is_finite())
				.copied()
				.collect();
			
			if finite_values.is_empty() {
				return f64::NAN;
			}
			
			let mean = finite_values.iter().sum::<f64>() / finite_values.len() as f64;
			let variance = finite_values.iter()
				.map(|x| (x - mean).powi(2))
				.sum::<f64>() / finite_values.len() as f64;
			variance
		}

		Ok(())
	}

	macro_rules! generate_all_pfe_tests {
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

	generate_all_pfe_tests!(
		check_pfe_partial_params,
		check_pfe_accuracy,
		check_pfe_default_candles,
		check_pfe_zero_period,
		check_pfe_period_exceeds_length,
		check_pfe_very_small_dataset,
		check_pfe_reinput,
		check_pfe_nan_handling,
		check_pfe_streaming,
		check_pfe_no_poison
	);

	#[cfg(feature = "proptest")]
	generate_all_pfe_tests!(check_pfe_property);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let output = PfeBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;

		let def = PfeParams::default();
		let row = output.values_for(&def).expect("default row missing");

		assert_eq!(row.len(), c.close.len());

		let expected = [-13.03562252, -11.93979855, -9.94609862, -9.73372410, -14.88374798];
		let start = row.len() - 5;
		for (i, &v) in row[start..].iter().enumerate() {
			assert!(
				(v - expected[i]).abs() < 1e-8,
				"[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
			);
		}
		Ok(())
	}

	#[cfg(debug_assertions)]
	fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		// Test various parameter sweep configurations
		let test_configs = vec![
			// (period_start, period_end, period_step, smoothing_start, smoothing_end, smoothing_step)
			(2, 10, 2, 1, 5, 1),       // Small periods and smoothing
			(5, 25, 5, 2, 10, 2),      // Medium periods and smoothing  
			(30, 60, 15, 5, 20, 5),    // Large periods and smoothing
			(2, 5, 1, 1, 3, 1),        // Dense small range
			(10, 10, 0, 1, 20, 1),     // Static period, varying smoothing
			(2, 100, 10, 5, 5, 0),     // Varying period, static smoothing
			(14, 21, 7, 3, 9, 3),      // Medium focused range
			(50, 100, 25, 10, 30, 10), // Large focused range
		];

		for (cfg_idx, &(p_start, p_end, p_step, s_start, s_end, s_step)) in test_configs.iter().enumerate() {
			let output = PfeBatchBuilder::new()
				.kernel(kernel)
				.period_range(p_start, p_end, p_step)
				.smoothing_range(s_start, s_end, s_step)
				.apply_candles(&c, "close")?;

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
						"[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}, smoothing={}",
						test, cfg_idx, val, bits, row, col, idx, 
						combo.period.unwrap_or(10), 
						combo.smoothing.unwrap_or(5)
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}, smoothing={}",
						test, cfg_idx, val, bits, row, col, idx, 
						combo.period.unwrap_or(10), 
						combo.smoothing.unwrap_or(5)
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}, smoothing={}",
						test, cfg_idx, val, bits, row, col, idx, 
						combo.period.unwrap_or(10), 
						combo.smoothing.unwrap_or(5)
					);
				}
			}
		}

		Ok(())
	}

	#[cfg(not(debug_assertions))]
	fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(()) // No-op in release builds
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
	gen_batch_tests!(check_batch_no_poison);
}
