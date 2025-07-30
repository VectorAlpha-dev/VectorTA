//! # Empirical Mode Decomposition (EMD)
//!
//! Implements the Empirical Mode Decomposition indicator with band-pass filtering and moving averages,
//! yielding upperband, middleband, and lowerband outputs.
//!
//! ## Parameters
//! - **period**: Window for the band-pass filter (default: 20)
//! - **delta**: Band-pass phase parameter (default: 0.5)
//! - **fraction**: Peak/valley scaling factor (default: 0.1)
//!
//! ## Errors
//! - **AllValuesNaN**: emd: All input data values are `NaN`.
//! - **InvalidPeriod**: emd: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: emd: Not enough valid data points for requested `period`.
//! - **InvalidDelta**: emd: `delta` is `NaN` or infinite.
//! - **InvalidFraction**: emd: `fraction` is `NaN` or infinite.
//!
//! ## Returns
//! - **`Ok(EmdOutput)`** on success, containing upperband/middleband/lowerband as `Vec<f64>`.
//! - **`Err(EmdError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes, make_uninit_matrix};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

impl<'a> AsRef<[f64]> for EmdInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			EmdData::Candles { candles } => source_type(candles, "close"),
			EmdData::Slices { close, .. } => close,
		}
	}
}

#[derive(Debug, Clone)]
pub enum EmdData<'a> {
	Candles {
		candles: &'a Candles,
	},
	Slices {
		high: &'a [f64],
		low: &'a [f64],
		close: &'a [f64],
		volume: &'a [f64],
	},
}

#[derive(Debug, Clone)]
pub struct EmdOutput {
	pub upperband: Vec<f64>,
	pub middleband: Vec<f64>,
	pub lowerband: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct EmdParams {
	pub period: Option<usize>,
	pub delta: Option<f64>,
	pub fraction: Option<f64>,
}

impl Default for EmdParams {
	fn default() -> Self {
		Self {
			period: Some(20),
			delta: Some(0.5),
			fraction: Some(0.1),
		}
	}
}

#[derive(Debug, Clone)]
pub struct EmdInput<'a> {
	pub data: EmdData<'a>,
	pub params: EmdParams,
}

impl<'a> EmdInput<'a> {
	#[inline]
	pub fn from_candles(candles: &'a Candles, params: EmdParams) -> Self {
		Self {
			data: EmdData::Candles { candles },
			params,
		}
	}

	#[inline]
	pub fn from_slices(
		high: &'a [f64],
		low: &'a [f64],
		close: &'a [f64],
		volume: &'a [f64],
		params: EmdParams,
	) -> Self {
		Self {
			data: EmdData::Slices {
				high,
				low,
				close,
				volume,
			},
			params,
		}
	}

	#[inline]
	pub fn with_default_candles(candles: &'a Candles) -> Self {
		Self::from_candles(candles, EmdParams::default())
	}

	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(20)
	}
	#[inline]
	pub fn get_delta(&self) -> f64 {
		self.params.delta.unwrap_or(0.5)
	}
	#[inline]
	pub fn get_fraction(&self) -> f64 {
		self.params.fraction.unwrap_or(0.1)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct EmdBuilder {
	period: Option<usize>,
	delta: Option<f64>,
	fraction: Option<f64>,
	kernel: Kernel,
}

impl Default for EmdBuilder {
	fn default() -> Self {
		Self {
			period: None,
			delta: None,
			fraction: None,
			kernel: Kernel::Auto,
		}
	}
}

impl EmdBuilder {
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
	pub fn delta(mut self, d: f64) -> Self {
		self.delta = Some(d);
		self
	}
	#[inline(always)]
	pub fn fraction(mut self, f: f64) -> Self {
		self.fraction = Some(f);
		self
	}
	#[inline(always)]
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}

	#[inline(always)]
	pub fn apply(self, c: &Candles) -> Result<EmdOutput, EmdError> {
		let p = EmdParams {
			period: self.period,
			delta: self.delta,
			fraction: self.fraction,
		};
		let i = EmdInput::from_candles(c, p);
		emd_with_kernel(&i, self.kernel)
	}

	#[inline(always)]
	pub fn apply_slices(self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Result<EmdOutput, EmdError> {
		let p = EmdParams {
			period: self.period,
			delta: self.delta,
			fraction: self.fraction,
		};
		let i = EmdInput::from_slices(high, low, close, volume, p);
		emd_with_kernel(&i, self.kernel)
	}

	#[inline(always)]
	pub fn into_stream(self) -> Result<EmdStream, EmdError> {
		let p = EmdParams {
			period: self.period,
			delta: self.delta,
			fraction: self.fraction,
		};
		EmdStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum EmdError {
	#[error("emd: All values are NaN.")]
	AllValuesNaN,

	#[error("emd: Invalid period: period = {period}, data length = {data_len}")]
	InvalidPeriod { period: usize, data_len: usize },

	#[error("emd: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },

	#[error("emd: Invalid delta: {delta}")]
	InvalidDelta { delta: f64 },

	#[error("emd: Invalid fraction: {fraction}")]
	InvalidFraction { fraction: f64 },

	#[error("emd: Invalid input length: expected = {expected}, actual = {actual}")]
	InvalidInputLength { expected: usize, actual: usize },
}

#[inline]
pub fn emd(input: &EmdInput) -> Result<EmdOutput, EmdError> {
	emd_with_kernel(input, Kernel::Auto)
}

fn emd_prepare<'a>(input: &'a EmdInput<'a>, kernel: Kernel) -> Result<(&'a [f64], &'a [f64], &'a [f64], &'a [f64], usize, f64, f64, usize, Kernel), EmdError> {
	let (high, low, close, volume) = match &input.data {
		EmdData::Candles { candles } => {
			let high = source_type(candles, "high");
			let low = source_type(candles, "low");
			let close = source_type(candles, "close");
			let volume = source_type(candles, "volume");
			(high, low, close, volume)
		}
		EmdData::Slices { high, low, close, volume } => {
			(*high, *low, *close, *volume)
		}
	};

	let len = high.len();
	let period = input.get_period();
	let delta = input.get_delta();
	let fraction = input.get_fraction();

	let first = (0..len)
		.find(|&i| !high[i].is_nan() && !low[i].is_nan())
		.ok_or(EmdError::AllValuesNaN)?;

	if period == 0 || period > len {
		return Err(EmdError::InvalidPeriod { period, data_len: len });
	}
	let needed = (2 * period).max(50);
	if (len - first) < needed {
		return Err(EmdError::NotEnoughValidData {
			needed,
			valid: len - first,
		});
	}
	if delta.is_nan() || delta.is_infinite() {
		return Err(EmdError::InvalidDelta { delta });
	}
	if fraction.is_nan() || fraction.is_infinite() {
		return Err(EmdError::InvalidFraction { fraction });
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	let warmup_period = first + 2 * period - 1;

	Ok((high, low, close, volume, period, delta, fraction, warmup_period, chosen))
}

fn emd_calc(
	high: &[f64],
	low: &[f64],
	period: usize,
	delta: f64,
	fraction: f64,
	first: usize,
	len: usize,
	kernel: Kernel,
) -> Result<EmdOutput, EmdError> {
	unsafe {
		match kernel {
			Kernel::Scalar | Kernel::ScalarBatch => emd_scalar(high, low, period, delta, fraction, first, len),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => emd_avx2(high, low, period, delta, fraction, first, len),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => emd_avx512(high, low, period, delta, fraction, first, len),
			_ => unreachable!(),
		}
	}
}

pub fn emd_with_kernel(input: &EmdInput, kernel: Kernel) -> Result<EmdOutput, EmdError> {
	let (high, low) = match &input.data {
		EmdData::Candles { candles } => {
			let high = source_type(candles, "high");
			let low = source_type(candles, "low");
			(high, low)
		}
		EmdData::Slices { high, low, .. } => (*high, *low),
	};

	let len = high.len();
	let period = input.get_period();
	let delta = input.get_delta();
	let fraction = input.get_fraction();

	let first = (0..len)
		.find(|&i| !high[i].is_nan() && !low[i].is_nan())
		.ok_or(EmdError::AllValuesNaN)?;

	if period == 0 || period > len {
		return Err(EmdError::InvalidPeriod { period, data_len: len });
	}
	let needed = (2 * period).max(50);
	if (len - first) < needed {
		return Err(EmdError::NotEnoughValidData {
			needed,
			valid: len - first,
		});
	}
	if delta.is_nan() || delta.is_infinite() {
		return Err(EmdError::InvalidDelta { delta });
	}
	if fraction.is_nan() || fraction.is_infinite() {
		return Err(EmdError::InvalidFraction { fraction });
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => emd_scalar(high, low, period, delta, fraction, first, len),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => emd_avx2(high, low, period, delta, fraction, first, len),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => emd_avx512(high, low, period, delta, fraction, first, len),
			_ => unreachable!(),
		}
	}
}

#[inline]
pub unsafe fn emd_scalar(
	high: &[f64],
	low: &[f64],
	period: usize,
	delta: f64,
	fraction: f64,
	first: usize,
	len: usize,
) -> Result<EmdOutput, EmdError> {
	// Warmup periods for each band
	let per_up_low = 50;
	let per_mid = 2 * period;
	let upperband_warmup = first + per_up_low - 1;
	let middleband_warmup = first + per_mid - 1;
	
	let mut upperband = alloc_with_nan_prefix(len, upperband_warmup);
	let mut middleband = alloc_with_nan_prefix(len, middleband_warmup);
	let mut lowerband = alloc_with_nan_prefix(len, upperband_warmup);

	let beta = (2.0 * std::f64::consts::PI / period as f64).cos();
	let gamma = 1.0 / ((4.0 * std::f64::consts::PI * delta / period as f64).cos());
	let alpha = gamma - (gamma * gamma - 1.0).sqrt();
	let half_one_minus_alpha = 0.5 * (1.0 - alpha);

	let mut sum_up = 0.0;
	let mut sum_mb = 0.0;
	let mut sum_low = 0.0;
	let mut sp_ring = vec![0.0; per_up_low];
	let mut sv_ring = vec![0.0; per_up_low];
	let mut bp_ring = vec![0.0; per_mid];
	let mut idx_up_low = 0_usize;
	let mut idx_mid = 0_usize;

	let mut bp_prev1 = 0.0;
	let mut bp_prev2 = 0.0;
	let mut peak_prev = 0.0;
	let mut valley_prev = 0.0;
	let mut initialized = false;
	let up_low_sub = per_up_low - 1;
	let mid_sub = per_mid - 1;

	for i in 0..len {
		if i < first {
			continue;
		}
		let price = (high[i] + low[i]) * 0.5;
		if !initialized {
			bp_prev1 = price;
			bp_prev2 = price;
			peak_prev = price;
			valley_prev = price;
			initialized = true;
		}
		let bp_curr = if i >= first + 2 {
			let price_i2 = (high[i - 2] + low[i - 2]) * 0.5;
			half_one_minus_alpha * (price - price_i2) + beta * (1.0 + alpha) * bp_prev1 - alpha * bp_prev2
		} else {
			price
		};
		let mut peak_curr = peak_prev;
		let mut valley_curr = valley_prev;
		if i >= first + 2 {
			if bp_prev1 > bp_curr && bp_prev1 > bp_prev2 {
				peak_curr = bp_prev1;
			}
			if bp_prev1 < bp_curr && bp_prev1 < bp_prev2 {
				valley_curr = bp_prev1;
			}
		}
		let sp = peak_curr * fraction;
		let sv = valley_curr * fraction;
		sum_up += sp;
		sum_low += sv;
		sum_mb += bp_curr;
		let old_sp = sp_ring[idx_up_low];
		let old_sv = sv_ring[idx_up_low];
		let old_bp = bp_ring[idx_mid];
		sp_ring[idx_up_low] = sp;
		sv_ring[idx_up_low] = sv;
		bp_ring[idx_mid] = bp_curr;
		if i >= first + per_up_low {
			sum_up -= old_sp;
			sum_low -= old_sv;
		}
		if i >= first + per_mid {
			sum_mb -= old_bp;
		}
		idx_up_low = (idx_up_low + 1) % per_up_low;
		idx_mid = (idx_mid + 1) % per_mid;
		if i >= first + up_low_sub {
			upperband[i] = sum_up / per_up_low as f64;
			lowerband[i] = sum_low / per_up_low as f64;
		}
		if i >= first + mid_sub {
			middleband[i] = sum_mb / per_mid as f64;
		}
		bp_prev2 = bp_prev1;
		bp_prev1 = bp_curr;
		peak_prev = peak_curr;
		valley_prev = valley_curr;
	}

	Ok(EmdOutput {
		upperband,
		middleband,
		lowerband,
	})
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn emd_avx2(
	high: &[f64],
	low: &[f64],
	period: usize,
	delta: f64,
	fraction: f64,
	first: usize,
	len: usize,
) -> Result<EmdOutput, EmdError> {
	emd_scalar(high, low, period, delta, fraction, first, len)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn emd_avx512(
	high: &[f64],
	low: &[f64],
	period: usize,
	delta: f64,
	fraction: f64,
	first: usize,
	len: usize,
) -> Result<EmdOutput, EmdError> {
	if period <= 32 {
		emd_avx512_short(high, low, period, delta, fraction, first, len)
	} else {
		emd_avx512_long(high, low, period, delta, fraction, first, len)
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn emd_avx512_short(
	high: &[f64],
	low: &[f64],
	period: usize,
	delta: f64,
	fraction: f64,
	first: usize,
	len: usize,
) -> Result<EmdOutput, EmdError> {
	emd_scalar(high, low, period, delta, fraction, first, len)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn emd_avx512_long(
	high: &[f64],
	low: &[f64],
	period: usize,
	delta: f64,
	fraction: f64,
	first: usize,
	len: usize,
) -> Result<EmdOutput, EmdError> {
	emd_scalar(high, low, period, delta, fraction, first, len)
}

#[inline(always)]
pub fn emd_row_scalar(
	high: &[f64],
	low: &[f64],
	period: usize,
	delta: f64,
	fraction: f64,
	first: usize,
	len: usize,
) -> Result<EmdOutput, EmdError> {
	unsafe { emd_scalar(high, low, period, delta, fraction, first, len) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn emd_row_avx2(
	high: &[f64],
	low: &[f64],
	period: usize,
	delta: f64,
	fraction: f64,
	first: usize,
	len: usize,
) -> Result<EmdOutput, EmdError> {
	unsafe { emd_avx2(high, low, period, delta, fraction, first, len) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn emd_row_avx512(
	high: &[f64],
	low: &[f64],
	period: usize,
	delta: f64,
	fraction: f64,
	first: usize,
	len: usize,
) -> Result<EmdOutput, EmdError> {
	unsafe { emd_avx512(high, low, period, delta, fraction, first, len) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn emd_row_avx512_short(
	high: &[f64],
	low: &[f64],
	period: usize,
	delta: f64,
	fraction: f64,
	first: usize,
	len: usize,
) -> Result<EmdOutput, EmdError> {
	unsafe { emd_avx512_short(high, low, period, delta, fraction, first, len) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn emd_row_avx512_long(
	high: &[f64],
	low: &[f64],
	period: usize,
	delta: f64,
	fraction: f64,
	first: usize,
	len: usize,
) -> Result<EmdOutput, EmdError> {
	unsafe { emd_avx512_long(high, low, period, delta, fraction, first, len) }
}

#[derive(Debug, Clone)]
pub struct EmdStream {
	period: usize,
	delta: f64,
	fraction: f64,
	per_up_low: usize,
	per_mid: usize,
	sum_up: f64,
	sum_low: f64,
	sum_mb: f64,
	sp_ring: Vec<f64>,
	sv_ring: Vec<f64>,
	bp_ring: Vec<f64>,
	idx_up_low: usize,
	idx_mid: usize,
	bp_prev1: f64,
	bp_prev2: f64,
	peak_prev: f64,
	valley_prev: f64,
	initialized: bool,
	count: usize,
}

impl EmdStream {
	pub fn try_new(params: EmdParams) -> Result<Self, EmdError> {
		let period = params.period.unwrap_or(20);
		let delta = params.delta.unwrap_or(0.5);
		let fraction = params.fraction.unwrap_or(0.1);

		if period == 0 {
			return Err(EmdError::InvalidPeriod { period, data_len: 0 });
		}
		if delta.is_nan() || delta.is_infinite() {
			return Err(EmdError::InvalidDelta { delta });
		}
		if fraction.is_nan() || fraction.is_infinite() {
			return Err(EmdError::InvalidFraction { fraction });
		}

		Ok(Self {
			period,
			delta,
			fraction,
			per_up_low: 50,
			per_mid: 2 * period,
			sum_up: 0.0,
			sum_low: 0.0,
			sum_mb: 0.0,
			sp_ring: vec![0.0; 50],
			sv_ring: vec![0.0; 50],
			bp_ring: vec![0.0; 2 * period],
			idx_up_low: 0,
			idx_mid: 0,
			bp_prev1: 0.0,
			bp_prev2: 0.0,
			peak_prev: 0.0,
			valley_prev: 0.0,
			initialized: false,
			count: 0,
		})
	}

	#[inline(always)]
	pub fn update(&mut self, high: f64, low: f64) -> (Option<f64>, Option<f64>, Option<f64>) {
		let price = (high + low) * 0.5;
		let beta = (2.0 * std::f64::consts::PI / self.period as f64).cos();
		let gamma = 1.0 / ((4.0 * std::f64::consts::PI * self.delta / self.period as f64).cos());
		let alpha = gamma - (gamma * gamma - 1.0).sqrt();
		let half_one_minus_alpha = 0.5 * (1.0 - alpha);

		if !self.initialized {
			self.bp_prev1 = price;
			self.bp_prev2 = price;
			self.peak_prev = price;
			self.valley_prev = price;
			self.initialized = true;
		}
		let bp_curr = if self.count >= 2 {
			let price_i2 = price;
			half_one_minus_alpha * (price - price_i2) + beta * (1.0 + alpha) * self.bp_prev1 - alpha * self.bp_prev2
		} else {
			price
		};
		let mut peak_curr = self.peak_prev;
		let mut valley_curr = self.valley_prev;
		if self.count >= 2 {
			if self.bp_prev1 > bp_curr && self.bp_prev1 > self.bp_prev2 {
				peak_curr = self.bp_prev1;
			}
			if self.bp_prev1 < bp_curr && self.bp_prev1 < self.bp_prev2 {
				valley_curr = self.bp_prev1;
			}
		}
		let sp = peak_curr * self.fraction;
		let sv = valley_curr * self.fraction;
		self.sum_up += sp;
		self.sum_low += sv;
		self.sum_mb += bp_curr;
		let old_sp = self.sp_ring[self.idx_up_low];
		let old_sv = self.sv_ring[self.idx_up_low];
		let old_bp = self.bp_ring[self.idx_mid];
		self.sp_ring[self.idx_up_low] = sp;
		self.sv_ring[self.idx_up_low] = sv;
		self.bp_ring[self.idx_mid] = bp_curr;
		if self.count >= self.per_up_low {
			self.sum_up -= old_sp;
			self.sum_low -= old_sv;
		}
		if self.count >= self.per_mid {
			self.sum_mb -= old_bp;
		}
		self.idx_up_low = (self.idx_up_low + 1) % self.per_up_low;
		self.idx_mid = (self.idx_mid + 1) % self.per_mid;
		let mut ub = None;
		let mut lb = None;
		let mut mb = None;
		if self.count + 1 >= self.per_up_low {
			ub = Some(self.sum_up / self.per_up_low as f64);
			lb = Some(self.sum_low / self.per_up_low as f64);
		}
		if self.count + 1 >= self.per_mid {
			mb = Some(self.sum_mb / self.per_mid as f64);
		}
		self.bp_prev2 = self.bp_prev1;
		self.bp_prev1 = bp_curr;
		self.peak_prev = peak_curr;
		self.valley_prev = valley_curr;
		self.count += 1;
		(ub, mb, lb)
	}
}

// Batch struct/logic
#[derive(Clone, Debug)]
pub struct EmdBatchRange {
	pub period: (usize, usize, usize),
	pub delta: (f64, f64, f64),
	pub fraction: (f64, f64, f64),
}

impl Default for EmdBatchRange {
	fn default() -> Self {
		Self {
			period: (20, 20, 0),
			delta: (0.5, 0.5, 0.0),
			fraction: (0.1, 0.1, 0.0),
		}
	}
}

#[derive(Clone, Debug, Default)]
pub struct EmdBatchBuilder {
	range: EmdBatchRange,
	kernel: Kernel,
}

impl EmdBatchBuilder {
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
	pub fn delta_range(mut self, start: f64, end: f64, step: f64) -> Self {
		self.range.delta = (start, end, step);
		self
	}
	#[inline]
	pub fn delta_static(mut self, x: f64) -> Self {
		self.range.delta = (x, x, 0.0);
		self
	}
	#[inline]
	pub fn fraction_range(mut self, start: f64, end: f64, step: f64) -> Self {
		self.range.fraction = (start, end, step);
		self
	}
	#[inline]
	pub fn fraction_static(mut self, x: f64) -> Self {
		self.range.fraction = (x, x, 0.0);
		self
	}
	pub fn apply_slices(
		self,
		high: &[f64],
		low: &[f64],
		close: &[f64],
		volume: &[f64],
	) -> Result<EmdBatchOutput, EmdError> {
		emd_batch_with_kernel(high, low, &self.range, self.kernel)
	}
	pub fn with_default_slices(
		high: &[f64],
		low: &[f64],
		close: &[f64],
		volume: &[f64],
		k: Kernel,
	) -> Result<EmdBatchOutput, EmdError> {
		EmdBatchBuilder::new().kernel(k).apply_slices(high, low, close, volume)
	}
	pub fn apply_candles(self, c: &Candles) -> Result<EmdBatchOutput, EmdError> {
		let high = source_type(c, "high");
		let low = source_type(c, "low");
		let close = source_type(c, "close");
		let volume = source_type(c, "volume");
		self.apply_slices(high, low, close, volume)
	}
}

pub fn emd_batch_with_kernel(
	high: &[f64],
	low: &[f64],
	sweep: &EmdBatchRange,
	k: Kernel,
) -> Result<EmdBatchOutput, EmdError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => return Err(EmdError::InvalidPeriod { period: 0, data_len: 0 }),
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	emd_batch_par_slice(high, low, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct EmdBatchOutput {
	pub upperband: Vec<f64>,
	pub middleband: Vec<f64>,
	pub lowerband: Vec<f64>,
	pub combos: Vec<EmdParams>,
	pub rows: usize,
	pub cols: usize,
}

#[inline(always)]
fn expand_grid(r: &EmdBatchRange) -> Vec<EmdParams> {
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
	let deltas = axis_f64(r.delta);
	let fractions = axis_f64(r.fraction);

	let mut out = Vec::with_capacity(periods.len() * deltas.len() * fractions.len());
	for &p in &periods {
		for &d in &deltas {
			for &f in &fractions {
				out.push(EmdParams {
					period: Some(p),
					delta: Some(d),
					fraction: Some(f),
				});
			}
		}
	}
	out
}

#[inline(always)]
pub fn emd_batch_slice(
	high: &[f64],
	low: &[f64],
	sweep: &EmdBatchRange,
	kern: Kernel,
) -> Result<EmdBatchOutput, EmdError> {
	emd_batch_inner(high, low, sweep, kern, false)
}

#[inline(always)]
pub fn emd_batch_par_slice(
	high: &[f64],
	low: &[f64],
	sweep: &EmdBatchRange,
	kern: Kernel,
) -> Result<EmdBatchOutput, EmdError> {
	emd_batch_inner(high, low, sweep, kern, true)
}

#[inline(always)]
fn emd_batch_inner(
	high: &[f64],
	low: &[f64],
	sweep: &EmdBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<EmdBatchOutput, EmdError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(EmdError::InvalidPeriod { period: 0, data_len: 0 });
	}
	let len = high.len();
	let first = (0..len)
		.find(|&i| !high[i].is_nan() && !low[i].is_nan())
		.ok_or(EmdError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	let needed = (2 * max_p).max(50);
	if len - first < needed {
		return Err(EmdError::NotEnoughValidData {
			needed,
			valid: len - first,
		});
	}

	let rows = combos.len();
	let cols = len;

	// Calculate warmup periods for each row
	let warmup_periods_upper: Vec<usize> = combos.iter()
		.map(|_| first + 49) // upperband/lowerband warmup is always first + 49
		.collect();
	let warmup_periods_middle: Vec<usize> = combos.iter()
		.map(|c| first + 2 * c.period.unwrap() - 1)
		.collect();

	// Use uninitialized matrix allocation with proper NaN prefixes
	let mut upperband_mu = make_uninit_matrix(rows, cols);
	init_matrix_prefixes(&mut upperband_mu, cols, &warmup_periods_upper);
	let mut middleband_mu = make_uninit_matrix(rows, cols);
	init_matrix_prefixes(&mut middleband_mu, cols, &warmup_periods_middle);
	let mut lowerband_mu = make_uninit_matrix(rows, cols);
	init_matrix_prefixes(&mut lowerband_mu, cols, &warmup_periods_upper);

	// Convert to mutable slices
	let upperband_slice = unsafe {
		std::slice::from_raw_parts_mut(upperband_mu.as_mut_ptr() as *mut f64, rows * cols)
	};
	let middleband_slice = unsafe {
		std::slice::from_raw_parts_mut(middleband_mu.as_mut_ptr() as *mut f64, rows * cols)
	};
	let lowerband_slice = unsafe {
		std::slice::from_raw_parts_mut(lowerband_mu.as_mut_ptr() as *mut f64, rows * cols)
	};

	let do_row = |row: usize, ub: &mut [f64], mb: &mut [f64], lb: &mut [f64]| {
		let prm = &combos[row];
		let period = prm.period.unwrap();
		let delta = prm.delta.unwrap();
		let fraction = prm.fraction.unwrap();
		let out = unsafe { emd_row_scalar(high, low, period, delta, fraction, first, cols) }
			.expect("emd row computation failed");
		ub.copy_from_slice(&out.upperband);
		mb.copy_from_slice(&out.middleband);
		lb.copy_from_slice(&out.lowerband);
	};
	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			upperband_slice
				.par_chunks_mut(cols)
				.zip(middleband_slice.par_chunks_mut(cols))
				.zip(lowerband_slice.par_chunks_mut(cols))
				.enumerate()
				.for_each(|(row, ((ub, mb), lb))| {
					do_row(row, ub, mb, lb);
				});
		}
		#[cfg(target_arch = "wasm32")]
		{
			for row in 0..rows {
				let ub = &mut upperband_slice[row * cols..(row + 1) * cols];
				let mb = &mut middleband_slice[row * cols..(row + 1) * cols];
				let lb = &mut lowerband_slice[row * cols..(row + 1) * cols];
				do_row(row, ub, mb, lb);
			}
		}
	} else {
		for row in 0..rows {
			let ub = &mut upperband_slice[row * cols..(row + 1) * cols];
			let mb = &mut middleband_slice[row * cols..(row + 1) * cols];
			let lb = &mut lowerband_slice[row * cols..(row + 1) * cols];
			do_row(row, ub, mb, lb);
		}
	}

	// Convert back to owned Vecs
	let upperband = unsafe {
		Vec::from_raw_parts(upperband_mu.as_mut_ptr() as *mut f64, rows * cols, rows * cols)
	};
	let middleband = unsafe {
		Vec::from_raw_parts(middleband_mu.as_mut_ptr() as *mut f64, rows * cols, rows * cols)
	};
	let lowerband = unsafe {
		Vec::from_raw_parts(lowerband_mu.as_mut_ptr() as *mut f64, rows * cols, rows * cols)
	};
	
	// Forget the original uninitialized vectors to prevent double-free
	std::mem::forget(upperband_mu);
	std::mem::forget(middleband_mu);
	std::mem::forget(lowerband_mu);

	Ok(EmdBatchOutput {
		upperband,
		middleband,
		lowerband,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
fn expand_grid_for_emdbatch(r: &EmdBatchRange) -> Vec<EmdParams> {
	expand_grid(r)
}

// API parity: this is required for batch indicator discovery/row mapping
impl EmdBatchOutput {
	pub fn row_for_params(&self, p: &EmdParams) -> Option<usize> {
		self.combos.iter().position(|c| {
			c.period.unwrap_or(20) == p.period.unwrap_or(20)
				&& (c.delta.unwrap_or(0.5) - p.delta.unwrap_or(0.5)).abs() < 1e-12
				&& (c.fraction.unwrap_or(0.1) - p.fraction.unwrap_or(0.1)).abs() < 1e-12
		})
	}
	pub fn bands_for(&self, p: &EmdParams) -> Option<(&[f64], &[f64], &[f64])> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			(
				&self.upperband[start..start + self.cols],
				&self.middleband[start..start + self.cols],
				&self.lowerband[start..start + self.cols],
			)
		})
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_emd_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let params = EmdParams::default();
		let input = EmdInput::from_candles(&candles, params);
		let emd_result = emd_with_kernel(&input, kernel)?;

		let expected_last_five_upper = [
			50.33760237677157,
			50.28850695686447,
			50.23941153695737,
			50.19031611705027,
			48.709744457737344,
		];
		let expected_last_five_middle = [
			-368.71064280396706,
			-399.11033986231377,
			-421.9368852621732,
			-437.879217150269,
			-447.3257167904511,
		];
		let expected_last_five_lower = [
			-60.67834136221248,
			-60.93110347122829,
			-61.68154077026321,
			-62.43197806929814,
			-63.18241536833306,
		];

		let len = candles.close.len();
		let start_idx = len - 5;
		let actual_ub = &emd_result.upperband[start_idx..];
		let actual_mb = &emd_result.middleband[start_idx..];
		let actual_lb = &emd_result.lowerband[start_idx..];
		for i in 0..5 {
			assert!(
				(actual_ub[i] - expected_last_five_upper[i]).abs() < 1e-6,
				"Upperband mismatch at index {}: expected {}, got {}",
				i,
				expected_last_five_upper[i],
				actual_ub[i]
			);
			assert!(
				(actual_mb[i] - expected_last_five_middle[i]).abs() < 1e-6,
				"Middleband mismatch at index {}: expected {}, got {}",
				i,
				expected_last_five_middle[i],
				actual_mb[i]
			);
			assert!(
				(actual_lb[i] - expected_last_five_lower[i]).abs() < 1e-6,
				"Lowerband mismatch at index {}: expected {}, got {}",
				i,
				expected_last_five_lower[i],
				actual_lb[i]
			);
		}
		Ok(())
	}

	fn check_emd_empty_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let empty_data: [f64; 0] = [];
		let params = EmdParams::default();
		let input = EmdInput::from_slices(&empty_data, &empty_data, &empty_data, &empty_data, params);
		let result = emd_with_kernel(&input, kernel);
		assert!(result.is_err(), "Expected error on empty data");
		Ok(())
	}

	fn check_emd_all_nans(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data = [f64::NAN, f64::NAN, f64::NAN];
		let params = EmdParams::default();
		let input = EmdInput::from_slices(&data, &data, &data, &data, params);
		let result = emd_with_kernel(&input, kernel);
		assert!(result.is_err(), "Expected error for all-NaN data");
		Ok(())
	}

	fn check_emd_invalid_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data = [1.0, 2.0, 3.0];
		let params = EmdParams {
			period: Some(0),
			..Default::default()
		};
		let input = EmdInput::from_slices(&data, &data, &data, &data, params);
		let result = emd_with_kernel(&input, kernel);
		assert!(result.is_err(), "Expected error for zero period");
		Ok(())
	}

	fn check_emd_not_enough_valid_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data = vec![10.0; 10];
		let params = EmdParams {
			period: Some(20),
			..Default::default()
		};
		let input = EmdInput::from_slices(&data, &data, &data, &data, params);
		let result = emd_with_kernel(&input, kernel);
		assert!(result.is_err(), "Expected error for not enough valid data");
		Ok(())
	}

	fn check_emd_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = EmdInput::with_default_candles(&candles);
		let result = emd_with_kernel(&input, kernel);
		assert!(result.is_ok(), "Expected EMD to succeed with default params");
		Ok(())
	}

	macro_rules! generate_all_emd_tests {
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
	fn check_emd_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		// Define comprehensive parameter combinations
		let test_params = vec![
			// Default parameters
			EmdParams::default(),
			// Minimum viable parameters
			EmdParams {
				period: Some(2),
				delta: Some(0.1),
				fraction: Some(0.05),
			},
			// Small period variations
			EmdParams {
				period: Some(5),
				delta: Some(0.3),
				fraction: Some(0.1),
			},
			EmdParams {
				period: Some(10),
				delta: Some(0.5),
				fraction: Some(0.15),
			},
			// Medium periods
			EmdParams {
				period: Some(20),
				delta: Some(0.4),
				fraction: Some(0.1),
			},
			EmdParams {
				period: Some(30),
				delta: Some(0.6),
				fraction: Some(0.2),
			},
			// Large periods
			EmdParams {
				period: Some(50),
				delta: Some(0.7),
				fraction: Some(0.25),
			},
			EmdParams {
				period: Some(100),
				delta: Some(0.8),
				fraction: Some(0.3),
			},
			// Edge cases with extreme delta
			EmdParams {
				period: Some(15),
				delta: Some(0.1),
				fraction: Some(0.1),
			},
			EmdParams {
				period: Some(15),
				delta: Some(0.9),
				fraction: Some(0.1),
			},
			// Edge cases with extreme fraction
			EmdParams {
				period: Some(25),
				delta: Some(0.5),
				fraction: Some(0.01),
			},
			EmdParams {
				period: Some(25),
				delta: Some(0.5),
				fraction: Some(0.5),
			},
			// Additional combinations
			EmdParams {
				period: Some(40),
				delta: Some(0.65),
				fraction: Some(0.12),
			},
			EmdParams {
				period: Some(7),
				delta: Some(0.25),
				fraction: Some(0.08),
			},
		];

		for (param_idx, params) in test_params.iter().enumerate() {
			let input = EmdInput::from_candles(&candles, params.clone());
			let output = emd_with_kernel(&input, kernel)?;

			// Check upperband array
			for (i, &val) in output.upperband.iter().enumerate() {
				if val.is_nan() {
					continue; // NaN values are expected during warmup
				}

				let bits = val.to_bits();

				// Check all three poison patterns
				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in upperband \
						 with params: period={}, delta={}, fraction={} (param set {})",
						test_name, val, bits, i,
						params.period.unwrap_or(20),
						params.delta.unwrap_or(0.5),
						params.fraction.unwrap_or(0.1),
						param_idx
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in upperband \
						 with params: period={}, delta={}, fraction={} (param set {})",
						test_name, val, bits, i,
						params.period.unwrap_or(20),
						params.delta.unwrap_or(0.5),
						params.fraction.unwrap_or(0.1),
						param_idx
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in upperband \
						 with params: period={}, delta={}, fraction={} (param set {})",
						test_name, val, bits, i,
						params.period.unwrap_or(20),
						params.delta.unwrap_or(0.5),
						params.fraction.unwrap_or(0.1),
						param_idx
					);
				}
			}

			// Check middleband array
			for (i, &val) in output.middleband.iter().enumerate() {
				if val.is_nan() {
					continue; // NaN values are expected during warmup
				}

				let bits = val.to_bits();

				// Check all three poison patterns
				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in middleband \
						 with params: period={}, delta={}, fraction={} (param set {})",
						test_name, val, bits, i,
						params.period.unwrap_or(20),
						params.delta.unwrap_or(0.5),
						params.fraction.unwrap_or(0.1),
						param_idx
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in middleband \
						 with params: period={}, delta={}, fraction={} (param set {})",
						test_name, val, bits, i,
						params.period.unwrap_or(20),
						params.delta.unwrap_or(0.5),
						params.fraction.unwrap_or(0.1),
						param_idx
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in middleband \
						 with params: period={}, delta={}, fraction={} (param set {})",
						test_name, val, bits, i,
						params.period.unwrap_or(20),
						params.delta.unwrap_or(0.5),
						params.fraction.unwrap_or(0.1),
						param_idx
					);
				}
			}

			// Check lowerband array
			for (i, &val) in output.lowerband.iter().enumerate() {
				if val.is_nan() {
					continue; // NaN values are expected during warmup
				}

				let bits = val.to_bits();

				// Check all three poison patterns
				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in lowerband \
						 with params: period={}, delta={}, fraction={} (param set {})",
						test_name, val, bits, i,
						params.period.unwrap_or(20),
						params.delta.unwrap_or(0.5),
						params.fraction.unwrap_or(0.1),
						param_idx
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in lowerband \
						 with params: period={}, delta={}, fraction={} (param set {})",
						test_name, val, bits, i,
						params.period.unwrap_or(20),
						params.delta.unwrap_or(0.5),
						params.fraction.unwrap_or(0.1),
						param_idx
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in lowerband \
						 with params: period={}, delta={}, fraction={} (param set {})",
						test_name, val, bits, i,
						params.period.unwrap_or(20),
						params.delta.unwrap_or(0.5),
						params.fraction.unwrap_or(0.1),
						param_idx
					);
				}
			}
		}

		Ok(())
	}

	#[cfg(not(debug_assertions))]
	fn check_emd_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(()) // No-op in release builds
	}

	generate_all_emd_tests!(
		check_emd_accuracy,
		check_emd_empty_data,
		check_emd_all_nans,
		check_emd_invalid_period,
		check_emd_not_enough_valid_data,
		check_emd_default_candles,
		check_emd_no_poison
	);
	#[cfg(test)]
	mod batch_tests {
		use super::*;
		use crate::skip_if_unsupported;
		use crate::utilities::data_loader::read_candles_from_csv;

		fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
			skip_if_unsupported!(kernel, test);

			let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
			let c = read_candles_from_csv(file)?;

			let output = EmdBatchBuilder::new().kernel(kernel).apply_candles(&c)?;

			let def = EmdParams::default();
			let (ub, mb, lb) = output.bands_for(&def).expect("default row missing");

			assert_eq!(ub.len(), c.close.len(), "Upperband length mismatch");
			assert_eq!(mb.len(), c.close.len(), "Middleband length mismatch");
			assert_eq!(lb.len(), c.close.len(), "Lowerband length mismatch");

			// Spot check last values vs. single-batch computation (if desired, could hardcode here)
			let expected_last_five_upper = [
				50.33760237677157,
				50.28850695686447,
				50.23941153695737,
				50.19031611705027,
				48.709744457737344,
			];
			let expected_last_five_middle = [
				-368.71064280396706,
				-399.11033986231377,
				-421.9368852621732,
				-437.879217150269,
				-447.3257167904511,
			];
			let expected_last_five_lower = [
				-60.67834136221248,
				-60.93110347122829,
				-61.68154077026321,
				-62.43197806929814,
				-63.18241536833306,
			];
			let len = ub.len();
			for i in 0..5 {
				assert!(
					(ub[len - 5 + i] - expected_last_five_upper[i]).abs() < 1e-6,
					"[{test}] upperband mismatch idx {i}: {} vs {}",
					ub[len - 5 + i],
					expected_last_five_upper[i]
				);
				assert!(
					(mb[len - 5 + i] - expected_last_five_middle[i]).abs() < 1e-6,
					"[{test}] middleband mismatch idx {i}: {} vs {}",
					mb[len - 5 + i],
					expected_last_five_middle[i]
				);
				assert!(
					(lb[len - 5 + i] - expected_last_five_lower[i]).abs() < 1e-6,
					"[{test}] lowerband mismatch idx {i}: {} vs {}",
					lb[len - 5 + i],
					expected_last_five_lower[i]
				);
			}

			Ok(())
		}

		fn check_batch_param_sweep(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
			skip_if_unsupported!(kernel, test);

			let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
			let c = read_candles_from_csv(file)?;

			// Sweep period over 20 and 22, delta over 0.5 and 0.6, fraction over 0.1 and 0.2
			let output = EmdBatchBuilder::new()
				.kernel(kernel)
				.period_range(20, 22, 2)
				.delta_range(0.5, 0.6, 0.1)
				.fraction_range(0.1, 0.2, 0.1)
				.apply_candles(&c)?;

			assert!(output.rows == 8, "Expected 8 rows (2*2*2 grid), got {}", output.rows);
			assert_eq!(output.cols, c.close.len());

			// Verify that bands_for returns correct shapes for all combos
			for params in &output.combos {
				let (ub, mb, lb) = output.bands_for(params).expect("row for params missing in sweep");
				assert_eq!(ub.len(), output.cols);
				assert_eq!(mb.len(), output.cols);
				assert_eq!(lb.len(), output.cols);
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

		#[cfg(debug_assertions)]
		fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
			skip_if_unsupported!(kernel, test);

			let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
			let c = read_candles_from_csv(file)?;

			// Test various parameter sweep configurations
			let test_configs = vec![
				// (period range, delta range, fraction range)
				// Small periods with varying delta and fraction
				(5, 15, 5, 0.1, 0.5, 0.2, 0.05, 0.15, 0.05),
				// Medium periods
				(10, 30, 10, 0.3, 0.7, 0.2, 0.1, 0.2, 0.05),
				// Large periods
				(20, 50, 15, 0.5, 0.8, 0.15, 0.15, 0.3, 0.075),
				// Dense small range
				(8, 12, 1, 0.4, 0.6, 0.1, 0.08, 0.12, 0.02),
				// Single values (no sweep)
				(20, 20, 0, 0.5, 0.5, 0.0, 0.1, 0.1, 0.0),
				// Wide period range with fine delta/fraction
				(5, 40, 5, 0.2, 0.8, 0.1, 0.05, 0.25, 0.05),
				// Edge case: minimum periods with varying other params
				(2, 6, 2, 0.1, 0.9, 0.4, 0.01, 0.5, 0.245),
			];

			for (cfg_idx, &(p_start, p_end, p_step, d_start, d_end, d_step, f_start, f_end, f_step)) in
				test_configs.iter().enumerate()
			{
				let output = EmdBatchBuilder::new()
					.kernel(kernel)
					.period_range(p_start, p_end, p_step)
					.delta_range(d_start, d_end, d_step)
					.fraction_range(f_start, f_end, f_step)
					.apply_candles(&c)?;

				// Check upperband matrix
				for (idx, &val) in output.upperband.iter().enumerate() {
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
							"[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) in upperband \
							 at row {} col {} (flat index {}) with params: period={}, delta={}, fraction={}",
							test, cfg_idx, val, bits, row, col, idx,
							combo.period.unwrap_or(20),
							combo.delta.unwrap_or(0.5),
							combo.fraction.unwrap_or(0.1)
						);
					}

					if bits == 0x22222222_22222222 {
						panic!(
							"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) in upperband \
							 at row {} col {} (flat index {}) with params: period={}, delta={}, fraction={}",
							test, cfg_idx, val, bits, row, col, idx,
							combo.period.unwrap_or(20),
							combo.delta.unwrap_or(0.5),
							combo.fraction.unwrap_or(0.1)
						);
					}

					if bits == 0x33333333_33333333 {
						panic!(
							"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) in upperband \
							 at row {} col {} (flat index {}) with params: period={}, delta={}, fraction={}",
							test, cfg_idx, val, bits, row, col, idx,
							combo.period.unwrap_or(20),
							combo.delta.unwrap_or(0.5),
							combo.fraction.unwrap_or(0.1)
						);
					}
				}

				// Check middleband matrix
				for (idx, &val) in output.middleband.iter().enumerate() {
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
							"[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) in middleband \
							 at row {} col {} (flat index {}) with params: period={}, delta={}, fraction={}",
							test, cfg_idx, val, bits, row, col, idx,
							combo.period.unwrap_or(20),
							combo.delta.unwrap_or(0.5),
							combo.fraction.unwrap_or(0.1)
						);
					}

					if bits == 0x22222222_22222222 {
						panic!(
							"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) in middleband \
							 at row {} col {} (flat index {}) with params: period={}, delta={}, fraction={}",
							test, cfg_idx, val, bits, row, col, idx,
							combo.period.unwrap_or(20),
							combo.delta.unwrap_or(0.5),
							combo.fraction.unwrap_or(0.1)
						);
					}

					if bits == 0x33333333_33333333 {
						panic!(
							"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) in middleband \
							 at row {} col {} (flat index {}) with params: period={}, delta={}, fraction={}",
							test, cfg_idx, val, bits, row, col, idx,
							combo.period.unwrap_or(20),
							combo.delta.unwrap_or(0.5),
							combo.fraction.unwrap_or(0.1)
						);
					}
				}

				// Check lowerband matrix
				for (idx, &val) in output.lowerband.iter().enumerate() {
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
							"[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) in lowerband \
							 at row {} col {} (flat index {}) with params: period={}, delta={}, fraction={}",
							test, cfg_idx, val, bits, row, col, idx,
							combo.period.unwrap_or(20),
							combo.delta.unwrap_or(0.5),
							combo.fraction.unwrap_or(0.1)
						);
					}

					if bits == 0x22222222_22222222 {
						panic!(
							"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) in lowerband \
							 at row {} col {} (flat index {}) with params: period={}, delta={}, fraction={}",
							test, cfg_idx, val, bits, row, col, idx,
							combo.period.unwrap_or(20),
							combo.delta.unwrap_or(0.5),
							combo.fraction.unwrap_or(0.1)
						);
					}

					if bits == 0x33333333_33333333 {
						panic!(
							"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) in lowerband \
							 at row {} col {} (flat index {}) with params: period={}, delta={}, fraction={}",
							test, cfg_idx, val, bits, row, col, idx,
							combo.period.unwrap_or(20),
							combo.delta.unwrap_or(0.5),
							combo.fraction.unwrap_or(0.1)
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

		gen_batch_tests!(check_batch_default_row);
		gen_batch_tests!(check_batch_param_sweep);
		gen_batch_tests!(check_batch_no_poison);
	}
}

// Python bindings
#[cfg(feature = "python")]
#[pyfunction(name = "emd")]
#[pyo3(signature = (high, low, close, volume, period, delta, fraction, kernel=None))]
pub fn emd_py<'py>(
	py: Python<'py>,
	high: PyReadonlyArray1<'py, f64>,
	low: PyReadonlyArray1<'py, f64>,
	close: PyReadonlyArray1<'py, f64>,
	volume: PyReadonlyArray1<'py, f64>,
	period: usize,
	delta: f64,
	fraction: f64,
	kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
	let high_slice = high.as_slice()?;
	let low_slice = low.as_slice()?;
	let close_slice = close.as_slice()?;
	let volume_slice = volume.as_slice()?;
	let kern = validate_kernel(kernel, false)?;

	let params = EmdParams {
		period: Some(period),
		delta: Some(delta),
		fraction: Some(fraction),
	};
	let input = EmdInput::from_slices(high_slice, low_slice, close_slice, volume_slice, params);

	let (upperband_vec, middleband_vec, lowerband_vec) = py
		.allow_threads(|| {
			emd_with_kernel(&input, kern)
				.map(|o| (o.upperband, o.middleband, o.lowerband))
		})
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	Ok((
		upperband_vec.into_pyarray(py),
		middleband_vec.into_pyarray(py),
		lowerband_vec.into_pyarray(py),
	))
}

#[cfg(feature = "python")]
#[pyclass(name = "EmdStream")]
pub struct EmdStreamPy {
	stream: EmdStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl EmdStreamPy {
	#[new]
	fn new(period: usize, delta: f64, fraction: f64) -> PyResult<Self> {
		let params = EmdParams {
			period: Some(period),
			delta: Some(delta),
			fraction: Some(fraction),
		};
		let stream = EmdStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(EmdStreamPy { stream })
	}

	fn update(&mut self, high: f64, low: f64) -> (Option<f64>, Option<f64>, Option<f64>) {
		self.stream.update(high, low)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "emd_batch")]
#[pyo3(signature = (high, low, close, volume, period_range, delta_range, fraction_range, kernel=None))]
pub fn emd_batch_py<'py>(
	py: Python<'py>,
	high: PyReadonlyArray1<'py, f64>,
	low: PyReadonlyArray1<'py, f64>,
	close: PyReadonlyArray1<'py, f64>,
	volume: PyReadonlyArray1<'py, f64>,
	period_range: (usize, usize, usize),
	delta_range: (f64, f64, f64),
	fraction_range: (f64, f64, f64),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
	let high_slice = high.as_slice()?;
	let low_slice = low.as_slice()?;
	let kern = validate_kernel(kernel, true)?;

	let sweep = EmdBatchRange {
		period: period_range,
		delta: delta_range,
		fraction: fraction_range,
	};

	let output = py
		.allow_threads(|| emd_batch_with_kernel(high_slice, low_slice, &sweep, kern))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	let dict = PyDict::new(py);
	
	// Reshape the flattened arrays into 2D matrices
	let rows = output.rows;
	let cols = output.cols;
	
	// Create reshaped arrays
	let upperband_arr = output.upperband.into_pyarray(py);
	let middleband_arr = output.middleband.into_pyarray(py);
	let lowerband_arr = output.lowerband.into_pyarray(py);
	
	dict.set_item("upperband", upperband_arr.reshape((rows, cols))?)?;
	dict.set_item("middleband", middleband_arr.reshape((rows, cols))?)?;
	dict.set_item("lowerband", lowerband_arr.reshape((rows, cols))?)?;
	
	// Add parameter arrays
	dict.set_item(
		"periods",
		output.combos
			.iter()
			.map(|p| p.period.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;
	dict.set_item(
		"deltas",
		output.combos
			.iter()
			.map(|p| p.delta.unwrap())
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;
	dict.set_item(
		"fractions",
		output.combos
			.iter()
			.map(|p| p.fraction.unwrap())
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;

	Ok(dict)
}

// ############################################
// WASM Bindings
// ############################################

/// Write EMD directly to output slices - no allocations
pub fn emd_into_slice(
	upperband_dst: &mut [f64],
	middleband_dst: &mut [f64],
	lowerband_dst: &mut [f64],
	input: &EmdInput,
	kern: Kernel,
) -> Result<(), EmdError> {
	let (high, low, close, volume, period, delta, fraction, warmup_period, chosen) = emd_prepare(input, kern)?;

	let len = high.len();
	if upperband_dst.len() != len || middleband_dst.len() != len || lowerband_dst.len() != len {
		return Err(EmdError::InvalidInputLength {
			expected: len,
			actual: upperband_dst.len(),
		});
	}

	// Compute EMD directly into the output slices
	let result = emd_calc(&high, &low, period, delta, fraction, 0, len, chosen)?;

	// Copy results to output slices
	upperband_dst.copy_from_slice(&result.upperband);
	middleband_dst.copy_from_slice(&result.middleband);
	lowerband_dst.copy_from_slice(&result.lowerband);

	Ok(())
}

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct EmdResult {
	values: Vec<f64>, // [upperband..., middleband..., lowerband...]
	rows: usize,      // 3 for EMD
	cols: usize,      // data length
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl EmdResult {
	#[wasm_bindgen(getter)]
	pub fn values(&self) -> Vec<f64> {
		self.values.clone()
	}

	#[wasm_bindgen(getter)]
	pub fn rows(&self) -> usize {
		self.rows
	}

	#[wasm_bindgen(getter)]
	pub fn cols(&self) -> usize {
		self.cols
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn emd_js(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	period: usize,
	delta: f64,
	fraction: f64,
) -> Result<EmdResult, JsValue> {
	let params = EmdParams {
		period: Some(period),
		delta: Some(delta),
		fraction: Some(fraction),
	};

	// Create candles from the input data
	let len = high.len();
	if low.len() != len || close.len() != len || volume.len() != len {
		return Err(JsValue::from_str("All input arrays must have the same length"));
	}

	let mut hl2 = Vec::with_capacity(len);
	let mut hlc3 = Vec::with_capacity(len);
	let mut ohlc4 = Vec::with_capacity(len);
	let mut hlcc4 = Vec::with_capacity(len);
	
	for i in 0..len {
		let h = high[i];
		let l = low[i];
		let c = close[i];
		hl2.push((h + l) / 2.0);
		hlc3.push((h + l + c) / 3.0);
		ohlc4.push((0.0 + h + l + c) / 4.0);  // open is 0
		hlcc4.push((h + l + c + c) / 4.0);
	}
	
	let candles = Candles {
		timestamp: vec![0; len],
		open: vec![0.0; len], // EMD doesn't use open prices
		high: high.to_vec(),
		low: low.to_vec(),
		close: close.to_vec(),
		volume: volume.to_vec(),
		hl2,
		hlc3,
		ohlc4,
		hlcc4,
	};

	let input = EmdInput::from_candles(&candles, params);

	// Single allocation for all three outputs
	let mut values = vec![0.0; len * 3];
	let (upper_slice, rest) = values.split_at_mut(len);
	let (middle_slice, lower_slice) = rest.split_at_mut(len);

	emd_into_slice(upper_slice, middle_slice, lower_slice, &input, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;

	Ok(EmdResult { values, rows: 3, cols: len })
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn emd_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn emd_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe {
			let _ = Vec::from_raw_parts(ptr, len, len);
		}
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn emd_into(
	high_ptr: *const f64,
	low_ptr: *const f64,
	close_ptr: *const f64,
	volume_ptr: *const f64,
	upper_ptr: *mut f64,
	middle_ptr: *mut f64,
	lower_ptr: *mut f64,
	len: usize,
	period: usize,
	delta: f64,
	fraction: f64,
) -> Result<(), JsValue> {
	if high_ptr.is_null()
		|| low_ptr.is_null()
		|| close_ptr.is_null()
		|| volume_ptr.is_null()
		|| upper_ptr.is_null()
		|| middle_ptr.is_null()
		|| lower_ptr.is_null()
	{
		return Err(JsValue::from_str("Null pointer provided"));
	}

	unsafe {
		let high = std::slice::from_raw_parts(high_ptr, len);
		let low = std::slice::from_raw_parts(low_ptr, len);
		let close = std::slice::from_raw_parts(close_ptr, len);
		let volume = std::slice::from_raw_parts(volume_ptr, len);

		let params = EmdParams {
			period: Some(period),
			delta: Some(delta),
			fraction: Some(fraction),
		};

		let mut hl2 = Vec::with_capacity(len);
		let mut hlc3 = Vec::with_capacity(len);
		let mut ohlc4 = Vec::with_capacity(len);
		let mut hlcc4 = Vec::with_capacity(len);
		
		for i in 0..len {
			let h = high[i];
			let l = low[i];
			let c = close[i];
			hl2.push((h + l) / 2.0);
			hlc3.push((h + l + c) / 3.0);
			ohlc4.push((0.0 + h + l + c) / 4.0);  // open is 0
			hlcc4.push((h + l + c + c) / 4.0);
		}
		
		let candles = Candles {
			timestamp: vec![0; len],
			open: vec![0.0; len],
			high: high.to_vec(),
			low: low.to_vec(),
			close: close.to_vec(),
			volume: volume.to_vec(),
			hl2,
			hlc3,
			ohlc4,
			hlcc4,
		};

		let input = EmdInput::from_candles(&candles, params);

		// Check for aliasing - any input pointer matching any output pointer
		let input_ptrs = [high_ptr as *const u8, low_ptr as *const u8, close_ptr as *const u8, volume_ptr as *const u8];
		let output_ptrs = [upper_ptr as *const u8, middle_ptr as *const u8, lower_ptr as *const u8];

		let has_aliasing = input_ptrs.iter().any(|&inp| output_ptrs.iter().any(|&out| inp == out));

		if has_aliasing {
			// Use temporary buffers for aliased operation
			let mut temp_upper = vec![0.0; len];
			let mut temp_middle = vec![0.0; len];
			let mut temp_lower = vec![0.0; len];

			emd_into_slice(&mut temp_upper, &mut temp_middle, &mut temp_lower, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;

			let upper_out = std::slice::from_raw_parts_mut(upper_ptr, len);
			let middle_out = std::slice::from_raw_parts_mut(middle_ptr, len);
			let lower_out = std::slice::from_raw_parts_mut(lower_ptr, len);

			upper_out.copy_from_slice(&temp_upper);
			middle_out.copy_from_slice(&temp_middle);
			lower_out.copy_from_slice(&temp_lower);
		} else {
			// Direct computation into output buffers
			let upper_out = std::slice::from_raw_parts_mut(upper_ptr, len);
			let middle_out = std::slice::from_raw_parts_mut(middle_ptr, len);
			let lower_out = std::slice::from_raw_parts_mut(lower_ptr, len);

			emd_into_slice(upper_out, middle_out, lower_out, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}

		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct EmdBatchConfig {
	pub period_range: (usize, usize, usize),
	pub delta_range: (f64, f64, f64),
	pub fraction_range: (f64, f64, f64),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct EmdBatchJsOutput {
	pub upperband: Vec<f64>,
	pub middleband: Vec<f64>,
	pub lowerband: Vec<f64>,
	pub combos: Vec<EmdParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = emd_batch)]
pub fn emd_batch_unified_js(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	volume: &[f64],
	config: JsValue,
) -> Result<JsValue, JsValue> {
	let config: EmdBatchConfig =
		serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

	let sweep = EmdBatchRange {
		period: config.period_range,
		delta: config.delta_range,
		fraction: config.fraction_range,
	};

	let len = high.len();
	let mut hl2 = Vec::with_capacity(len);
	let mut hlc3 = Vec::with_capacity(len);
	let mut ohlc4 = Vec::with_capacity(len);
	let mut hlcc4 = Vec::with_capacity(len);
	
	for i in 0..len {
		let h = high[i];
		let l = low[i];
		let c = close[i];
		hl2.push((h + l) / 2.0);
		hlc3.push((h + l + c) / 3.0);
		ohlc4.push((0.0 + h + l + c) / 4.0);  // open is 0
		hlcc4.push((h + l + c + c) / 4.0);
	}
	
	let candles = Candles {
		timestamp: vec![0; len],
		open: vec![0.0; len],
		high: high.to_vec(),
		low: low.to_vec(),
		close: close.to_vec(),
		volume: volume.to_vec(),
		hl2,
		hlc3,
		ohlc4,
		hlcc4,
	};

	let output = emd_batch_slice(high, low, &sweep, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;

	let js_output = EmdBatchJsOutput {
		upperband: output.upperband,
		middleband: output.middleband,
		lowerband: output.lowerband,
		combos: output.combos,
		rows: output.rows,
		cols: output.cols,
	};

	serde_wasm_bindgen::to_value(&js_output).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn emd_batch_into(
	high_ptr: *const f64,
	low_ptr: *const f64,
	close_ptr: *const f64,
	volume_ptr: *const f64,
	upper_ptr: *mut f64,
	middle_ptr: *mut f64,
	lower_ptr: *mut f64,
	len: usize,
	period_start: usize,
	period_end: usize,
	period_step: usize,
	delta_start: f64,
	delta_end: f64,
	delta_step: f64,
	fraction_start: f64,
	fraction_end: f64,
	fraction_step: f64,
) -> Result<usize, JsValue> {
	if high_ptr.is_null()
		|| low_ptr.is_null()
		|| close_ptr.is_null()
		|| volume_ptr.is_null()
		|| upper_ptr.is_null()
		|| middle_ptr.is_null()
		|| lower_ptr.is_null()
	{
		return Err(JsValue::from_str("null pointer passed to emd_batch_into"));
	}

	unsafe {
		let high = std::slice::from_raw_parts(high_ptr, len);
		let low = std::slice::from_raw_parts(low_ptr, len);
		let close = std::slice::from_raw_parts(close_ptr, len);
		let volume = std::slice::from_raw_parts(volume_ptr, len);

		let sweep = EmdBatchRange {
			period: (period_start, period_end, period_step),
			delta: (delta_start, delta_end, delta_step),
			fraction: (fraction_start, fraction_end, fraction_step),
		};

		let mut hl2 = Vec::with_capacity(len);
		let mut hlc3 = Vec::with_capacity(len);
		let mut ohlc4 = Vec::with_capacity(len);
		let mut hlcc4 = Vec::with_capacity(len);
		
		for i in 0..len {
			let h = high[i];
			let l = low[i];
			let c = close[i];
			hl2.push((h + l) / 2.0);
			hlc3.push((h + l + c) / 3.0);
			ohlc4.push((0.0 + h + l + c) / 4.0);  // open is 0
			hlcc4.push((h + l + c + c) / 4.0);
		}
		
		let candles = Candles {
			timestamp: vec![0; len],
			open: vec![0.0; len],
			high: high.to_vec(),
			low: low.to_vec(),
			close: close.to_vec(),
			volume: volume.to_vec(),
			hl2,
			hlc3,
			ohlc4,
			hlcc4,
		};

		let output = emd_batch_slice(high, low, &sweep, Kernel::Auto)
			.map_err(|e| JsValue::from_str(&e.to_string()))?;

		let rows = output.rows;
		let cols = output.cols;
		let total_len = rows * cols;

		// Check output buffer sizes
		let upper_slice = std::slice::from_raw_parts_mut(upper_ptr, total_len);
		let middle_slice = std::slice::from_raw_parts_mut(middle_ptr, total_len);
		let lower_slice = std::slice::from_raw_parts_mut(lower_ptr, total_len);

		upper_slice.copy_from_slice(&output.upperband);
		middle_slice.copy_from_slice(&output.middleband);
		lower_slice.copy_from_slice(&output.lowerband);

		Ok(rows)
	}
}
