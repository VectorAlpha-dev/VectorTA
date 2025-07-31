//! # Squeeze Momentum Indicator (SMI)
//!
//! Detects market volatility "squeeze" and provides a smoothed momentum signal. Mirrors alma.rs in structure, features, and performance.
//!
//! ## Parameters
//! - **length_bb**: Lookback window for Bollinger Bands (default: 20)
//! - **mult_bb**: BB stddev multiplier (default: 2.0)
//! - **length_kc**: Lookback window for Keltner Channels (default: 20)
//! - **mult_kc**: KC multiplier (default: 1.5)
//!
//! ## Errors
//! - **AllValuesNaN**: All input values are NaN
//! - **InvalidLength**: A lookback parameter is zero or exceeds data length
//! - **InconsistentDataLength**: High/low/close have different lengths
//! - **NotEnoughValidData**: Not enough valid data for requested lookback
//!
//! ## Returns
//! - **`Ok(SqueezeMomentumOutput)`** on success
//! - **`Err(SqueezeMomentumError)`** otherwise

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
	alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes, make_uninit_matrix,
};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::mem::MaybeUninit;
use thiserror::Error;

// --- Core Data Types ---

#[derive(Debug, Clone)]
pub enum SqueezeMomentumData<'a> {
	Candles {
		candles: &'a Candles,
	},
	Slices {
		high: &'a [f64],
		low: &'a [f64],
		close: &'a [f64],
	},
}

#[derive(Debug, Clone)]
pub struct SqueezeMomentumParams {
	pub length_bb: Option<usize>,
	pub mult_bb: Option<f64>,
	pub length_kc: Option<usize>,
	pub mult_kc: Option<f64>,
}

impl Default for SqueezeMomentumParams {
	fn default() -> Self {
		Self {
			length_bb: Some(20),
			mult_bb: Some(2.0),
			length_kc: Some(20),
			mult_kc: Some(1.5),
		}
	}
}

#[derive(Debug, Clone)]
pub struct SqueezeMomentumInput<'a> {
	pub data: SqueezeMomentumData<'a>,
	pub params: SqueezeMomentumParams,
}

impl<'a> SqueezeMomentumInput<'a> {
	#[inline(always)]
	pub fn from_candles(c: &'a Candles, params: SqueezeMomentumParams) -> Self {
		Self {
			data: SqueezeMomentumData::Candles { candles: c },
			params,
		}
	}
	#[inline(always)]
	pub fn from_slices(high: &'a [f64], low: &'a [f64], close: &'a [f64], params: SqueezeMomentumParams) -> Self {
		Self {
			data: SqueezeMomentumData::Slices { high, low, close },
			params,
		}
	}
	#[inline(always)]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, SqueezeMomentumParams::default())
	}
}

#[derive(Debug, Clone)]
pub struct SqueezeMomentumOutput {
	pub squeeze: Vec<f64>,
	pub momentum: Vec<f64>,
	pub momentum_signal: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct SqueezeMomentumBuilder {
	length_bb: Option<usize>,
	mult_bb: Option<f64>,
	length_kc: Option<usize>,
	mult_kc: Option<f64>,
	kernel: Kernel,
}

impl Default for SqueezeMomentumBuilder {
	fn default() -> Self {
		Self {
			length_bb: None,
			mult_bb: None,
			length_kc: None,
			mult_kc: None,
			kernel: Kernel::Auto,
		}
	}
}

impl SqueezeMomentumBuilder {
	#[inline(always)]
	pub fn new() -> Self {
		Self::default()
	}
	#[inline(always)]
	pub fn length_bb(mut self, n: usize) -> Self {
		self.length_bb = Some(n);
		self
	}
	#[inline(always)]
	pub fn mult_bb(mut self, x: f64) -> Self {
		self.mult_bb = Some(x);
		self
	}
	#[inline(always)]
	pub fn length_kc(mut self, n: usize) -> Self {
		self.length_kc = Some(n);
		self
	}
	#[inline(always)]
	pub fn mult_kc(mut self, x: f64) -> Self {
		self.mult_kc = Some(x);
		self
	}
	#[inline(always)]
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}

	#[inline(always)]
	pub fn apply(self, c: &Candles) -> Result<SqueezeMomentumOutput, SqueezeMomentumError> {
		let p = SqueezeMomentumParams {
			length_bb: self.length_bb,
			mult_bb: self.mult_bb,
			length_kc: self.length_kc,
			mult_kc: self.mult_kc,
		};
		let i = SqueezeMomentumInput::from_candles(c, p);
		squeeze_momentum_with_kernel(&i, self.kernel)
	}

	#[inline(always)]
	pub fn apply_slices(
		self,
		high: &[f64],
		low: &[f64],
		close: &[f64],
	) -> Result<SqueezeMomentumOutput, SqueezeMomentumError> {
		let p = SqueezeMomentumParams {
			length_bb: self.length_bb,
			mult_bb: self.mult_bb,
			length_kc: self.length_kc,
			mult_kc: self.mult_kc,
		};
		let i = SqueezeMomentumInput::from_slices(high, low, close, p);
		squeeze_momentum_with_kernel(&i, self.kernel)
	}
}

#[derive(Debug, Error)]
pub enum SqueezeMomentumError {
	#[error("smi: Empty data provided for Squeeze Momentum.")]
	EmptyData,
	#[error("smi: Invalid length parameter: length = {length}, data length = {data_len}")]
	InvalidLength { length: usize, data_len: usize },
	#[error("smi: High/low/close arrays have inconsistent lengths.")]
	InconsistentDataLength,
	#[error("smi: All values are NaN.")]
	AllValuesNaN,
	#[error("smi: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
}

// --- Main Kernel Selection ---

#[inline]
pub fn squeeze_momentum(input: &SqueezeMomentumInput) -> Result<SqueezeMomentumOutput, SqueezeMomentumError> {
	squeeze_momentum_with_kernel(input, Kernel::Auto)
}

/// Write squeeze momentum results directly into pre-allocated slices.
/// This is more efficient than allocating new vectors when the caller already has buffers.
/// All slices must have the same length as the input data.
#[inline]
pub fn squeeze_momentum_into_slices(
	squeeze_dst: &mut [f64],
	momentum_dst: &mut [f64], 
	momentum_signal_dst: &mut [f64],
	input: &SqueezeMomentumInput,
	kern: Kernel,
) -> Result<(), SqueezeMomentumError> {
	let output = squeeze_momentum_with_kernel(input, kern)?;
	
	// Verify slice lengths match
	let expected_len = match &input.data {
		SqueezeMomentumData::Candles { candles } => candles.close.len(),
		SqueezeMomentumData::Slices { close, .. } => close.len(),
	};
	
	if squeeze_dst.len() != expected_len || momentum_dst.len() != expected_len || momentum_signal_dst.len() != expected_len {
		return Err(SqueezeMomentumError::InconsistentDataLength);
	}
	
	// Copy results into destination slices
	squeeze_dst.copy_from_slice(&output.squeeze);
	momentum_dst.copy_from_slice(&output.momentum);
	momentum_signal_dst.copy_from_slice(&output.momentum_signal);
	
	Ok(())
}

pub fn squeeze_momentum_with_kernel(
	input: &SqueezeMomentumInput,
	kernel: Kernel,
) -> Result<SqueezeMomentumOutput, SqueezeMomentumError> {
	let (high, low, close): (&[f64], &[f64], &[f64]) = match &input.data {
		SqueezeMomentumData::Candles { candles } => (
			source_type(candles, "high"),
			source_type(candles, "low"),
			source_type(candles, "close"),
		),
		SqueezeMomentumData::Slices { high, low, close } => (*high, *low, *close),
	};
	if high.is_empty() || low.is_empty() || close.is_empty() {
		return Err(SqueezeMomentumError::EmptyData);
	}
	if high.len() != low.len() || low.len() != close.len() {
		return Err(SqueezeMomentumError::InconsistentDataLength);
	}
	let length_bb = input.params.length_bb.unwrap_or(20);
	let mult_bb = input.params.mult_bb.unwrap_or(2.0);
	let length_kc = input.params.length_kc.unwrap_or(20);
	let mult_kc = input.params.mult_kc.unwrap_or(1.5);
	if length_bb == 0 || length_bb > close.len() {
		return Err(SqueezeMomentumError::InvalidLength {
			length: length_bb,
			data_len: close.len(),
		});
	}
	if length_kc == 0 || length_kc > close.len() {
		return Err(SqueezeMomentumError::InvalidLength {
			length: length_kc,
			data_len: close.len(),
		});
	}
	let first_valid = (0..close.len())
		.find(|&i| !(high[i].is_nan() || low[i].is_nan() || close[i].is_nan()))
		.ok_or(SqueezeMomentumError::AllValuesNaN)?;
	let needed = length_bb.max(length_kc);
	if (high.len() - first_valid) < needed {
		return Err(SqueezeMomentumError::NotEnoughValidData {
			needed,
			valid: high.len() - first_valid,
		});
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};
	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => {
				squeeze_momentum_scalar(high, low, close, length_bb, mult_bb, length_kc, mult_kc, first_valid)
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => {
				squeeze_momentum_avx2(high, low, close, length_bb, mult_bb, length_kc, mult_kc, first_valid)
			}
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => {
				squeeze_momentum_avx512(high, low, close, length_bb, mult_bb, length_kc, mult_kc, first_valid)
			}
			_ => unreachable!(),
		}
	}
}

#[inline]
pub unsafe fn squeeze_momentum_scalar(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	length_bb: usize,
	mult_bb: f64,
	length_kc: usize,
	mult_kc: f64,
	first_valid: usize,
) -> Result<SqueezeMomentumOutput, SqueezeMomentumError> {
	Ok(crate::indicators::squeeze_momentum::squeeze_momentum_scalar_impl(
		high,
		low,
		close,
		length_bb,
		mult_bb,
		length_kc,
		mult_kc,
		first_valid,
	))
}

// These are stubs, they point back to the scalar implementation as required for API parity.

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn squeeze_momentum_avx2(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	length_bb: usize,
	mult_bb: f64,
	length_kc: usize,
	mult_kc: f64,
	first_valid: usize,
) -> Result<SqueezeMomentumOutput, SqueezeMomentumError> {
	squeeze_momentum_scalar(high, low, close, length_bb, mult_bb, length_kc, mult_kc, first_valid)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn squeeze_momentum_avx512(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	length_bb: usize,
	mult_bb: f64,
	length_kc: usize,
	mult_kc: f64,
	first_valid: usize,
) -> Result<SqueezeMomentumOutput, SqueezeMomentumError> {
	if length_kc <= 32 {
		squeeze_momentum_avx512_short(high, low, close, length_bb, mult_bb, length_kc, mult_kc, first_valid)
	} else {
		squeeze_momentum_avx512_long(high, low, close, length_bb, mult_bb, length_kc, mult_kc, first_valid)
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn squeeze_momentum_avx512_short(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	length_bb: usize,
	mult_bb: f64,
	length_kc: usize,
	mult_kc: f64,
	first_valid: usize,
) -> Result<SqueezeMomentumOutput, SqueezeMomentumError> {
	squeeze_momentum_scalar(high, low, close, length_bb, mult_bb, length_kc, mult_kc, first_valid)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn squeeze_momentum_avx512_long(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	length_bb: usize,
	mult_bb: f64,
	length_kc: usize,
	mult_kc: f64,
	first_valid: usize,
) -> Result<SqueezeMomentumOutput, SqueezeMomentumError> {
	squeeze_momentum_scalar(high, low, close, length_bb, mult_bb, length_kc, mult_kc, first_valid)
}

// --- Scalar Computation Implementation ---

pub fn squeeze_momentum_scalar_impl(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	length_bb: usize,
	mult_bb: f64,
	length_kc: usize,
	mult_kc: f64,
	first_valid: usize,
) -> SqueezeMomentumOutput {
	use crate::indicators::sma::{sma, SmaInput, SmaParams};

	let n = close.len();

	let bb_sma_params = SmaParams {
		period: Some(length_bb),
	};
	let bb_sma_input = SmaInput::from_slice(close, bb_sma_params);
	let bb_sma_output = sma(&bb_sma_input).unwrap();
	let basis = &bb_sma_output.values;
	let dev = stddev_slice(close, length_bb);
	let warmup_bb = length_bb.saturating_sub(1);
	let mut upper_bb = alloc_with_nan_prefix(n, warmup_bb);
	let mut lower_bb = alloc_with_nan_prefix(n, warmup_bb);
	for i in first_valid..n {
		if i + 1 >= length_bb && !basis[i].is_nan() && !dev[i].is_nan() {
			upper_bb[i] = basis[i] + mult_bb * dev[i];
			lower_bb[i] = basis[i] - mult_bb * dev[i];
		}
	}
	let kc_sma_params = SmaParams {
		period: Some(length_kc),
	};
	let kc_sma_input = SmaInput::from_slice(close, kc_sma_params.clone());
	let kc_sma_output = sma(&kc_sma_input).unwrap();
	let kc_ma = &kc_sma_output.values;
	let true_range = true_range_slice(high, low, close);
	let tr_sma_input = SmaInput::from_slice(&true_range, kc_sma_params.clone());
	let tr_sma_output = sma(&tr_sma_input).unwrap();
	let tr_ma = &tr_sma_output.values;
	let warmup_kc = length_kc.saturating_sub(1);
	let mut upper_kc = alloc_with_nan_prefix(n, warmup_kc);
	let mut lower_kc = alloc_with_nan_prefix(n, warmup_kc);
	for i in first_valid..n {
		if i + 1 >= length_kc && !kc_ma[i].is_nan() && !tr_ma[i].is_nan() {
			upper_kc[i] = kc_ma[i] + tr_ma[i] * mult_kc;
			lower_kc[i] = kc_ma[i] - tr_ma[i] * mult_kc;
		}
	}
	let warmup_squeeze = length_bb.max(length_kc).saturating_sub(1);
	let mut squeeze = alloc_with_nan_prefix(n, warmup_squeeze);
	for i in first_valid..n {
		if !lower_bb[i].is_nan() && !upper_bb[i].is_nan() && !lower_kc[i].is_nan() && !upper_kc[i].is_nan() {
			let sqz_on = lower_bb[i] > lower_kc[i] && upper_bb[i] < upper_kc[i];
			let sqz_off = lower_bb[i] < lower_kc[i] && upper_bb[i] > upper_kc[i];
			let no_sqz = !sqz_on && !sqz_off;
			squeeze[i] = if no_sqz {
				0.0
			} else if sqz_on {
				-1.0
			} else {
				1.0
			};
		}
	}
	let highest_vals = rolling_high_slice(high, length_kc);
	let lowest_vals = rolling_low_slice(low, length_kc);
	let sma_kc_input = SmaInput::from_slice(close, kc_sma_params);
	let sma_kc_output = sma(&sma_kc_input).unwrap();
	let ma_kc = &sma_kc_output.values;
	let mut momentum_raw = alloc_with_nan_prefix(n, warmup_kc);
	for i in first_valid..n {
		if i + 1 >= length_kc
			&& !close[i].is_nan()
			&& !highest_vals[i].is_nan()
			&& !lowest_vals[i].is_nan()
			&& !ma_kc[i].is_nan()
		{
			let mid = (highest_vals[i] + lowest_vals[i]) / 2.0;
			momentum_raw[i] = close[i] - (mid + ma_kc[i]) / 2.0;
		}
	}
	let momentum = linearreg_slice(&momentum_raw, length_kc);
	let warmup_signal = length_kc.saturating_sub(1) + 1; // +1 for the lag
	let mut momentum_signal = alloc_with_nan_prefix(n, warmup_signal);
	for i in first_valid..(n.saturating_sub(1)) {
		if !momentum[i].is_nan() && !momentum[i + 1].is_nan() {
			let next = momentum[i + 1];
			let curr = momentum[i];
			if next > 0.0 {
				momentum_signal[i + 1] = if next > curr { 1.0 } else { 2.0 };
			} else {
				momentum_signal[i + 1] = if next < curr { -1.0 } else { -2.0 };
			}
		}
	}
	SqueezeMomentumOutput {
		squeeze,
		momentum,
		momentum_signal,
	}
}

// --- Batch Parameter Sweep Support ---

#[derive(Clone, Debug)]
pub struct SqueezeMomentumBatchRange {
	pub length_bb: (usize, usize, usize),
	pub mult_bb: (f64, f64, f64),
	pub length_kc: (usize, usize, usize),
	pub mult_kc: (f64, f64, f64),
}

impl Default for SqueezeMomentumBatchRange {
	fn default() -> Self {
		Self {
			length_bb: (20, 20, 0),
			mult_bb: (2.0, 2.0, 0.0),
			length_kc: (20, 20, 0),
			mult_kc: (1.5, 1.5, 0.0),
		}
	}
}

#[derive(Clone, Debug, Default)]
pub struct SqueezeMomentumBatchBuilder {
	range: SqueezeMomentumBatchRange,
	kernel: Kernel,
}

impl SqueezeMomentumBatchBuilder {
	pub fn new() -> Self {
		Self::default()
	}
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	pub fn length_bb_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.length_bb = (start, end, step);
		self
	}
	pub fn length_bb_static(mut self, p: usize) -> Self {
		self.range.length_bb = (p, p, 0);
		self
	}
	pub fn mult_bb_range(mut self, start: f64, end: f64, step: f64) -> Self {
		self.range.mult_bb = (start, end, step);
		self
	}
	pub fn mult_bb_static(mut self, x: f64) -> Self {
		self.range.mult_bb = (x, x, 0.0);
		self
	}
	pub fn length_kc_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.length_kc = (start, end, step);
		self
	}
	pub fn length_kc_static(mut self, p: usize) -> Self {
		self.range.length_kc = (p, p, 0);
		self
	}
	pub fn mult_kc_range(mut self, start: f64, end: f64, step: f64) -> Self {
		self.range.mult_kc = (start, end, step);
		self
	}
	pub fn mult_kc_static(mut self, x: f64) -> Self {
		self.range.mult_kc = (x, x, 0.0);
		self
	}

	pub fn apply_slices(
		self,
		high: &[f64],
		low: &[f64],
		close: &[f64],
	) -> Result<SqueezeMomentumBatchOutput, SqueezeMomentumError> {
		squeeze_momentum_batch_with_kernel(high, low, close, &self.range, self.kernel)
	}

	pub fn apply_candles(self, c: &Candles) -> Result<SqueezeMomentumBatchOutput, SqueezeMomentumError> {
		let high = source_type(c, "high");
		let low = source_type(c, "low");
		let close = source_type(c, "close");
		self.apply_slices(high, low, close)
	}
}

#[derive(Clone, Debug)]
pub struct SqueezeMomentumBatchParams {
	pub length_bb: usize,
	pub mult_bb: f64,
	pub length_kc: usize,
	pub mult_kc: f64,
}

#[derive(Clone, Debug)]
pub struct SqueezeMomentumBatchOutput {
	pub momentum: Vec<f64>,
	pub combos: Vec<SqueezeMomentumBatchParams>,
	pub rows: usize,
	pub cols: usize,
}

impl SqueezeMomentumBatchOutput {
	pub fn row_for_params(&self, p: &SqueezeMomentumBatchParams) -> Option<usize> {
		self.combos.iter().position(|c| {
			c.length_bb == p.length_bb
				&& (c.mult_bb - p.mult_bb).abs() < 1e-12
				&& c.length_kc == p.length_kc
				&& (c.mult_kc - p.mult_kc).abs() < 1e-12
		})
	}
	pub fn values_for(&self, p: &SqueezeMomentumBatchParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.momentum[start..start + self.cols]
		})
	}
}

fn expand_grid_sm(range: &SqueezeMomentumBatchRange) -> Vec<SqueezeMomentumBatchParams> {
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
	let length_bbs = axis_usize(range.length_bb);
	let mult_bbs = axis_f64(range.mult_bb);
	let length_kcs = axis_usize(range.length_kc);
	let mult_kcs = axis_f64(range.mult_kc);
	let mut out = Vec::with_capacity(length_bbs.len() * mult_bbs.len() * length_kcs.len() * mult_kcs.len());
	for &lbb in &length_bbs {
		for &mbb in &mult_bbs {
			for &lkc in &length_kcs {
				for &mkc in &mult_kcs {
					out.push(SqueezeMomentumBatchParams {
						length_bb: lbb,
						mult_bb: mbb,
						length_kc: lkc,
						mult_kc: mkc,
					});
				}
			}
		}
	}
	out
}

pub fn squeeze_momentum_batch_with_kernel(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	sweep: &SqueezeMomentumBatchRange,
	kernel: Kernel,
) -> Result<SqueezeMomentumBatchOutput, SqueezeMomentumError> {
	let combos = expand_grid_sm(sweep);
	if combos.is_empty() {
		return Err(SqueezeMomentumError::InvalidLength { length: 0, data_len: 0 });
	}
	let first_valid = (0..close.len())
		.find(|&i| !(high[i].is_nan() || low[i].is_nan() || close[i].is_nan()))
		.ok_or(SqueezeMomentumError::AllValuesNaN)?;
	let max_l = combos.iter().map(|c| c.length_bb.max(c.length_kc)).max().unwrap();
	if close.len() - first_valid < max_l {
		return Err(SqueezeMomentumError::NotEnoughValidData {
			needed: max_l,
			valid: close.len() - first_valid,
		});
	}
	let rows = combos.len();
	let cols = close.len();
	// Create uninitialized matrix
	let mut buf_momentum = make_uninit_matrix(rows, cols);
	// Calculate warmup periods for each row (based on length_kc since momentum depends on it)
	let warmup_periods: Vec<usize> = combos.iter().map(|p| p.length_kc.saturating_sub(1)).collect();
	// Initialize NaN prefixes
	init_matrix_prefixes(&mut buf_momentum, cols, &warmup_periods);
	// Get mutable slice for computation
	let momentum = unsafe {
		std::slice::from_raw_parts_mut(buf_momentum.as_mut_ptr() as *mut f64, rows * cols)
	};

	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let p = &combos[row];
		let result = squeeze_momentum_row_scalar(
			high,
			low,
			close,
			first_valid,
			p.length_bb,
			p.mult_bb,
			p.length_kc,
			p.mult_kc,
			out_row,
		);
		result
	};

	#[cfg(not(target_arch = "wasm32"))]
	{
		momentum.par_chunks_mut(cols).enumerate().for_each(|(row, slice)| {
			do_row(row, slice);
		});
	}
	#[cfg(target_arch = "wasm32")]
	{
		for (row, slice) in momentum.chunks_mut(cols).enumerate() {
			do_row(row, slice);
		}
	}

	// Convert MaybeUninit<f64> to Vec<f64>
	let momentum_vec = unsafe {
		let ptr = buf_momentum.as_mut_ptr() as *mut f64;
		let len = buf_momentum.len();
		let cap = buf_momentum.capacity();
		std::mem::forget(buf_momentum);
		Vec::from_raw_parts(ptr, len, cap)
	};
	
	Ok(SqueezeMomentumBatchOutput {
		momentum: momentum_vec,
		combos,
		rows,
		cols,
	})
}

// Per-row kernel, just outputting the momentum vector for the parameter set
pub unsafe fn squeeze_momentum_row_scalar(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	first_valid: usize,
	length_bb: usize,
	mult_bb: f64,
	length_kc: usize,
	mult_kc: f64,
	out: &mut [f64],
) {
	let result = squeeze_momentum_scalar_impl(high, low, close, length_bb, mult_bb, length_kc, mult_kc, first_valid);
	out.copy_from_slice(&result.momentum);
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
pub unsafe fn squeeze_momentum_row_avx2(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	first_valid: usize,
	length_bb: usize,
	mult_bb: f64,
	length_kc: usize,
	mult_kc: f64,
	out: &mut [f64],
) {
	squeeze_momentum_row_scalar(
		high,
		low,
		close,
		first_valid,
		length_bb,
		mult_bb,
		length_kc,
		mult_kc,
		out,
	)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
pub unsafe fn squeeze_momentum_row_avx512(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	first_valid: usize,
	length_bb: usize,
	mult_bb: f64,
	length_kc: usize,
	mult_kc: f64,
	out: &mut [f64],
) {
	squeeze_momentum_row_scalar(
		high,
		low,
		close,
		first_valid,
		length_bb,
		mult_bb,
		length_kc,
		mult_kc,
		out,
	)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
pub unsafe fn squeeze_momentum_row_avx512_short(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	first_valid: usize,
	length_bb: usize,
	mult_bb: f64,
	length_kc: usize,
	mult_kc: f64,
	out: &mut [f64],
) {
	squeeze_momentum_row_scalar(
		high,
		low,
		close,
		first_valid,
		length_bb,
		mult_bb,
		length_kc,
		mult_kc,
		out,
	)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
pub unsafe fn squeeze_momentum_row_avx512_long(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	first_valid: usize,
	length_bb: usize,
	mult_bb: f64,
	length_kc: usize,
	mult_kc: f64,
	out: &mut [f64],
) {
	squeeze_momentum_row_scalar(
		high,
		low,
		close,
		first_valid,
		length_bb,
		mult_bb,
		length_kc,
		mult_kc,
		out,
	)
}

// --- Utilities (Unchanged from scalar, as in original) ---

fn stddev_slice(data: &[f64], period: usize) -> Vec<f64> {
	let warmup = period.saturating_sub(1);
	let mut output = alloc_with_nan_prefix(data.len(), warmup);
	if period == 0 || period > data.len() {
		return output;
	}
	let mut window_sum = 0.0;
	let mut window_sumsq = 0.0;
	for i in 0..period {
		let v = data[i];
		if v.is_finite() {
			window_sum += v;
			window_sumsq += v * v;
		}
	}
	let mut count = period;
	if count > 0 {
		output[period - 1] = variance_to_stddev(window_sum, window_sumsq, count);
	}
	for i in period..data.len() {
		let old_v = data[i - period];
		let new_v = data[i];
		if old_v.is_finite() {
			window_sum -= old_v;
			window_sumsq -= old_v * old_v;
		}
		if new_v.is_finite() {
			window_sum += new_v;
			window_sumsq += new_v * new_v;
		}
		output[i] = variance_to_stddev(window_sum, window_sumsq, count);
	}
	output
}
fn variance_to_stddev(sum: f64, sumsq: f64, count: usize) -> f64 {
	if count < 2 {
		return f64::NAN;
	}
	let mean = sum / (count as f64);
	let var = (sumsq / (count as f64)) - (mean * mean);
	if var.is_sign_negative() {
		f64::NAN
	} else {
		var.sqrt()
	}
}
fn true_range_slice(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
	if high.len() != low.len() || low.len() != close.len() {
		return vec![];
	}
	let mut output = alloc_with_nan_prefix(high.len(), 0); // TR has no warmup
	let mut prev_close = close[0];
	output[0] = high[0].max(low[0]) - low[0].min(high[0]);
	for i in 1..high.len() {
		if !high[i].is_nan() && !low[i].is_nan() && !prev_close.is_nan() {
			let tr1 = high[i] - low[i];
			let tr2 = (high[i] - prev_close).abs();
			let tr3 = (low[i] - prev_close).abs();
			output[i] = tr1.max(tr2).max(tr3);
		}
		prev_close = close[i];
	}
	output
}
fn rolling_high_slice(data: &[f64], period: usize) -> Vec<f64> {
	let warmup = period.saturating_sub(1);
	let mut output = alloc_with_nan_prefix(data.len(), warmup);
	if period == 0 || period > data.len() {
		return output;
	}
	let mut deque = Vec::new();
	for i in 0..data.len() {
		if !data[i].is_nan() {
			deque.push(data[i]);
		} else {
			deque.push(f64::NAN);
		}
		if i + 1 >= period {
			if i + 1 > period {
				deque.remove(0);
			}
			output[i] = deque.iter().copied().fold(f64::NAN, |a, b| a.max(b));
		}
	}
	output
}
fn rolling_low_slice(data: &[f64], period: usize) -> Vec<f64> {
	let warmup = period.saturating_sub(1);
	let mut output = alloc_with_nan_prefix(data.len(), warmup);
	if period == 0 || period > data.len() {
		return output;
	}
	let mut deque = Vec::new();
	for i in 0..data.len() {
		if !data[i].is_nan() {
			deque.push(data[i]);
		} else {
			deque.push(f64::NAN);
		}
		if i + 1 >= period {
			if i + 1 > period {
				deque.remove(0);
			}
			let mut mn = f64::NAN;
			for &v in &deque {
				if !v.is_nan() && (mn.is_nan() || v < mn) {
					mn = v;
				}
			}
			output[i] = mn;
		}
	}
	output
}
fn linearreg_slice(data: &[f64], period: usize) -> Vec<f64> {
	let warmup = period.saturating_sub(1);
	let mut output = alloc_with_nan_prefix(data.len(), warmup);
	if period == 0 || period > data.len() {
		return output;
	}
	for i in (period - 1)..data.len() {
		let subset = &data[i + 1 - period..=i];
		if subset.iter().all(|x| x.is_finite()) {
			output[i] = linear_regression_last_point(subset);
		}
	}
	output
}
fn linear_regression_last_point(window: &[f64]) -> f64 {
	let n = window.len();
	if n < 2 {
		return f64::NAN;
	}
	let mut sum_x = 0.0;
	let mut sum_y = 0.0;
	let mut sum_xy = 0.0;
	let mut sum_x2 = 0.0;
	for (i, &val) in window.iter().enumerate() {
		let x = (i + 1) as f64;
		sum_x += x;
		sum_y += val;
		sum_xy += x * val;
		sum_x2 += x * x;
	}
	let n_f = n as f64;
	let denom = (n_f * sum_x2) - (sum_x * sum_x);
	if denom.abs() < f64::EPSILON {
		return f64::NAN;
	}
	let slope = (n_f * sum_xy - sum_x * sum_y) / denom;
	let intercept = (sum_y - slope * sum_x) / n_f;
	let x_last = n_f;
	intercept + slope * x_last
}

// --- Tests: Parity with alma.rs, for all kernels, errors, accuracy, etc. ---

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_smi_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = SqueezeMomentumParams {
			length_bb: None,
			mult_bb: None,
			length_kc: None,
			mult_kc: None,
		};
		let input = SqueezeMomentumInput::from_candles(&candles, params);
		let output = squeeze_momentum_with_kernel(&input, kernel)?;
		assert_eq!(output.squeeze.len(), candles.close.len());
		Ok(())
	}

	fn check_smi_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = SqueezeMomentumInput::with_default_candles(&candles);
		let output = squeeze_momentum_with_kernel(&input, kernel)?;
		let expected_last_five = [-170.9, -155.4, -65.3, -61.1, -178.1];
		let n = output.momentum.len();
		let start = n.saturating_sub(5);
		for (i, &val) in output.momentum[start..].iter().enumerate() {
			let diff = (val - expected_last_five[i]).abs();
			assert!(
				diff < 1e-1,
				"[{}] SMI {:?} mismatch at idx {}: got {}, expected {}",
				test_name,
				kernel,
				i,
				val,
				expected_last_five[i]
			);
		}
		Ok(())
	}

	fn check_smi_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = SqueezeMomentumInput::with_default_candles(&candles);
		let output = squeeze_momentum_with_kernel(&input, kernel)?;
		assert_eq!(output.squeeze.len(), candles.close.len());
		Ok(())
	}

	fn check_smi_zero_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let h = [10.0, 20.0, 30.0];
		let l = [10.0, 20.0, 30.0];
		let c = [10.0, 20.0, 30.0];
		let params = SqueezeMomentumParams {
			length_bb: Some(0),
			mult_bb: None,
			length_kc: Some(0),
			mult_kc: None,
		};
		let input = SqueezeMomentumInput::from_slices(&h, &l, &c, params);
		let res = squeeze_momentum_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] SMI should fail with zero length", test_name);
		Ok(())
	}

	fn check_smi_length_exceeds(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let h = [10.0, 20.0, 30.0];
		let l = [10.0, 20.0, 30.0];
		let c = [10.0, 20.0, 30.0];
		let params = SqueezeMomentumParams {
			length_bb: Some(10),
			mult_bb: None,
			length_kc: Some(10),
			mult_kc: None,
		};
		let input = SqueezeMomentumInput::from_slices(&h, &l, &c, params);
		let res = squeeze_momentum_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] SMI should fail with length exceeding", test_name);
		Ok(())
	}

	fn check_smi_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let h = [f64::NAN, f64::NAN, f64::NAN];
		let l = [f64::NAN, f64::NAN, f64::NAN];
		let c = [f64::NAN, f64::NAN, f64::NAN];
		let params = SqueezeMomentumParams::default();
		let input = SqueezeMomentumInput::from_slices(&h, &l, &c, params);
		let res = squeeze_momentum_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] SMI should fail with all NaN", test_name);
		Ok(())
	}

	fn check_smi_inconsistent_lengths(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let h = [1.0, 2.0, 3.0];
		let l = [1.0, 2.0];
		let c = [1.0, 2.0, 3.0];
		let params = SqueezeMomentumParams::default();
		let input = SqueezeMomentumInput::from_slices(&h, &l, &c, params);
		let res = squeeze_momentum_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] SMI should fail with inconsistent data lengths",
			test_name
		);
		Ok(())
	}

	fn check_smi_minimum_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let h = [10.0, 12.0, 14.0];
		let l = [5.0, 6.0, 7.0];
		let c = [7.0, 11.0, 10.0];
		let params = SqueezeMomentumParams {
			length_bb: Some(5),
			mult_bb: Some(2.0),
			length_kc: Some(5),
			mult_kc: Some(1.5),
		};
		let input = SqueezeMomentumInput::from_slices(&h, &l, &c, params);
		let result = squeeze_momentum_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}

	#[cfg(debug_assertions)]
	fn check_smi_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		// Define comprehensive parameter combinations
		let test_params = vec![
			// Default parameters
			SqueezeMomentumParams::default(),
			// Minimum periods
			SqueezeMomentumParams {
				length_bb: Some(2),
				mult_bb: Some(1.0),
				length_kc: Some(2),
				mult_kc: Some(1.0),
			},
			// Small periods with default multipliers
			SqueezeMomentumParams {
				length_bb: Some(5),
				mult_bb: Some(2.0),
				length_kc: Some(5),
				mult_kc: Some(1.5),
			},
			// Medium periods with various multipliers
			SqueezeMomentumParams {
				length_bb: Some(10),
				mult_bb: Some(1.5),
				length_kc: Some(10),
				mult_kc: Some(2.0),
			},
			SqueezeMomentumParams {
				length_bb: Some(14),
				mult_bb: Some(2.5),
				length_kc: Some(14),
				mult_kc: Some(1.0),
			},
			// Standard periods with edge multipliers
			SqueezeMomentumParams {
				length_bb: Some(20),
				mult_bb: Some(0.5),
				length_kc: Some(20),
				mult_kc: Some(0.5),
			},
			SqueezeMomentumParams {
				length_bb: Some(20),
				mult_bb: Some(3.0),
				length_kc: Some(20),
				mult_kc: Some(3.0),
			},
			// Large periods
			SqueezeMomentumParams {
				length_bb: Some(50),
				mult_bb: Some(2.0),
				length_kc: Some(50),
				mult_kc: Some(1.5),
			},
			// Very large periods
			SqueezeMomentumParams {
				length_bb: Some(100),
				mult_bb: Some(1.5),
				length_kc: Some(100),
				mult_kc: Some(2.0),
			},
			// Asymmetric periods
			SqueezeMomentumParams {
				length_bb: Some(10),
				mult_bb: Some(2.0),
				length_kc: Some(20),
				mult_kc: Some(1.5),
			},
			SqueezeMomentumParams {
				length_bb: Some(30),
				mult_bb: Some(1.5),
				length_kc: Some(15),
				mult_kc: Some(2.5),
			},
		];

		for (param_idx, params) in test_params.iter().enumerate() {
			let input = SqueezeMomentumInput::from_candles(&candles, params.clone());
			let output = squeeze_momentum_with_kernel(&input, kernel)?;

			// Check squeeze values
			for (i, &val) in output.squeeze.iter().enumerate() {
				if val.is_nan() {
					continue; // NaN values are expected during warmup
				}

				let bits = val.to_bits();

				// Check all three poison patterns
				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in squeeze \
						 with params: length_bb={}, mult_bb={}, length_kc={}, mult_kc={} (param set {})",
						test_name, val, bits, i,
						params.length_bb.unwrap_or(20),
						params.mult_bb.unwrap_or(2.0),
						params.length_kc.unwrap_or(20),
						params.mult_kc.unwrap_or(1.5),
						param_idx
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in squeeze \
						 with params: length_bb={}, mult_bb={}, length_kc={}, mult_kc={} (param set {})",
						test_name, val, bits, i,
						params.length_bb.unwrap_or(20),
						params.mult_bb.unwrap_or(2.0),
						params.length_kc.unwrap_or(20),
						params.mult_kc.unwrap_or(1.5),
						param_idx
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in squeeze \
						 with params: length_bb={}, mult_bb={}, length_kc={}, mult_kc={} (param set {})",
						test_name, val, bits, i,
						params.length_bb.unwrap_or(20),
						params.mult_bb.unwrap_or(2.0),
						params.length_kc.unwrap_or(20),
						params.mult_kc.unwrap_or(1.5),
						param_idx
					);
				}
			}

			// Check momentum values
			for (i, &val) in output.momentum.iter().enumerate() {
				if val.is_nan() {
					continue;
				}

				let bits = val.to_bits();

				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in momentum \
						 with params: length_bb={}, mult_bb={}, length_kc={}, mult_kc={} (param set {})",
						test_name, val, bits, i,
						params.length_bb.unwrap_or(20),
						params.mult_bb.unwrap_or(2.0),
						params.length_kc.unwrap_or(20),
						params.mult_kc.unwrap_or(1.5),
						param_idx
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in momentum \
						 with params: length_bb={}, mult_bb={}, length_kc={}, mult_kc={} (param set {})",
						test_name, val, bits, i,
						params.length_bb.unwrap_or(20),
						params.mult_bb.unwrap_or(2.0),
						params.length_kc.unwrap_or(20),
						params.mult_kc.unwrap_or(1.5),
						param_idx
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in momentum \
						 with params: length_bb={}, mult_bb={}, length_kc={}, mult_kc={} (param set {})",
						test_name, val, bits, i,
						params.length_bb.unwrap_or(20),
						params.mult_bb.unwrap_or(2.0),
						params.length_kc.unwrap_or(20),
						params.mult_kc.unwrap_or(1.5),
						param_idx
					);
				}
			}

			// Check momentum_signal values
			for (i, &val) in output.momentum_signal.iter().enumerate() {
				if val.is_nan() {
					continue;
				}

				let bits = val.to_bits();

				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in momentum_signal \
						 with params: length_bb={}, mult_bb={}, length_kc={}, mult_kc={} (param set {})",
						test_name, val, bits, i,
						params.length_bb.unwrap_or(20),
						params.mult_bb.unwrap_or(2.0),
						params.length_kc.unwrap_or(20),
						params.mult_kc.unwrap_or(1.5),
						param_idx
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in momentum_signal \
						 with params: length_bb={}, mult_bb={}, length_kc={}, mult_kc={} (param set {})",
						test_name, val, bits, i,
						params.length_bb.unwrap_or(20),
						params.mult_bb.unwrap_or(2.0),
						params.length_kc.unwrap_or(20),
						params.mult_kc.unwrap_or(1.5),
						param_idx
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in momentum_signal \
						 with params: length_bb={}, mult_bb={}, length_kc={}, mult_kc={} (param set {})",
						test_name, val, bits, i,
						params.length_bb.unwrap_or(20),
						params.mult_bb.unwrap_or(2.0),
						params.length_kc.unwrap_or(20),
						params.mult_kc.unwrap_or(1.5),
						param_idx
					);
				}
			}
		}

		Ok(())
	}

	#[cfg(not(debug_assertions))]
	fn check_smi_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		Ok(()) // No-op in release builds
	}

	macro_rules! generate_all_smi_tests {
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

	generate_all_smi_tests!(
		check_smi_partial_params,
		check_smi_accuracy,
		check_smi_default_candles,
		check_smi_zero_length,
		check_smi_length_exceeds,
		check_smi_all_nan,
		check_smi_inconsistent_lengths,
		check_smi_minimum_data,
		check_smi_no_poison
	);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let output = SqueezeMomentumBatchBuilder::new().kernel(kernel).apply_candles(&c)?;
		let def = SqueezeMomentumBatchParams {
			length_bb: 20,
			mult_bb: 2.0,
			length_kc: 20,
			mult_kc: 1.5,
		};
		let row = output.values_for(&def).expect("default row missing");
		assert_eq!(row.len(), c.close.len());
		Ok(())
	}

	#[cfg(debug_assertions)]
	fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		// Test various parameter sweep configurations
		let test_configs = vec![
			// (length_bb_start, length_bb_end, length_bb_step, mult_bb_start, mult_bb_end, mult_bb_step,
			//  length_kc_start, length_kc_end, length_kc_step, mult_kc_start, mult_kc_end, mult_kc_step)
			(2, 10, 2, 1.0, 2.0, 0.5, 2, 10, 2, 1.0, 2.0, 0.5),      // Small periods
			(5, 25, 5, 2.0, 2.0, 0.0, 5, 25, 5, 1.5, 1.5, 0.0),      // Medium periods, static multipliers
			(10, 10, 0, 1.0, 3.0, 0.5, 10, 10, 0, 1.0, 3.0, 0.5),    // Static periods, varying multipliers
			(2, 5, 1, 1.5, 1.5, 0.0, 2, 5, 1, 2.0, 2.0, 0.0),        // Dense small range
			(30, 60, 15, 2.0, 2.0, 0.0, 30, 60, 15, 1.5, 1.5, 0.0),  // Large periods
			(20, 30, 5, 1.0, 2.5, 0.5, 15, 25, 5, 1.0, 2.0, 0.5),    // Mixed ranges
			(8, 12, 1, 0.5, 3.0, 0.5, 8, 12, 1, 0.5, 2.5, 0.5),      // Dense medium range with wide multipliers
		];

		for (cfg_idx, &(lbb_start, lbb_end, lbb_step, mbb_start, mbb_end, mbb_step,
		                lkc_start, lkc_end, lkc_step, mkc_start, mkc_end, mkc_step)) in
			test_configs.iter().enumerate()
		{
			let output = SqueezeMomentumBatchBuilder::new()
				.kernel(kernel)
				.length_bb_range(lbb_start, lbb_end, lbb_step)
				.mult_bb_range(mbb_start, mbb_end, mbb_step)
				.length_kc_range(lkc_start, lkc_end, lkc_step)
				.mult_kc_range(mkc_start, mkc_end, mkc_step)
				.apply_candles(&c)?;

			// Only check momentum values (as per SqueezeMomentumBatchOutput structure)
			for (idx, &val) in output.momentum.iter().enumerate() {
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
						at row {} col {} (flat index {}) with params: length_bb={}, mult_bb={}, length_kc={}, mult_kc={}",
						test,
						cfg_idx,
						val,
						bits,
						row,
						col,
						idx,
						combo.length_bb,
						combo.mult_bb,
						combo.length_kc,
						combo.mult_kc
					);
				}

				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						at row {} col {} (flat index {}) with params: length_bb={}, mult_bb={}, length_kc={}, mult_kc={}",
						test,
						cfg_idx,
						val,
						bits,
						row,
						col,
						idx,
						combo.length_bb,
						combo.mult_bb,
						combo.length_kc,
						combo.mult_kc
					);
				}

				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						at row {} col {} (flat index {}) with params: length_bb={}, mult_bb={}, length_kc={}, mult_kc={}",
						test,
						cfg_idx,
						val,
						bits,
						row,
						col,
						idx,
						combo.length_bb,
						combo.mult_bb,
						combo.length_kc,
						combo.mult_kc
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

// --- Python Bindings ---

#[cfg(feature = "python")]
#[pyfunction(name = "squeeze_momentum")]
#[pyo3(signature = (high, low, close, length_bb=None, mult_bb=None, length_kc=None, mult_kc=None, kernel=None))]
pub fn squeeze_momentum_py<'py>(
	py: Python<'py>,
	high: PyReadonlyArray1<'py, f64>,
	low: PyReadonlyArray1<'py, f64>,
	close: PyReadonlyArray1<'py, f64>,
	length_bb: Option<usize>,
	mult_bb: Option<f64>,
	length_kc: Option<usize>,
	mult_kc: Option<f64>,
	kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
	let high_slice = high.as_slice()?;
	let low_slice = low.as_slice()?;
	let close_slice = close.as_slice()?;
	let kern = validate_kernel(kernel, false)?;

	let params = SqueezeMomentumParams {
		length_bb,
		mult_bb,
		length_kc,
		mult_kc,
	};
	let input = SqueezeMomentumInput::from_slices(high_slice, low_slice, close_slice, params);

	let (squeeze_vec, momentum_vec, momentum_signal_vec) = py
		.allow_threads(|| squeeze_momentum_with_kernel(&input, kern).map(|o| (o.squeeze, o.momentum, o.momentum_signal)))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	Ok((
		squeeze_vec.into_pyarray(py),
		momentum_vec.into_pyarray(py),
		momentum_signal_vec.into_pyarray(py),
	))
}

#[cfg(feature = "python")]
#[pyclass(name = "SqueezeMomentumStream")]
pub struct SqueezeMomentumStreamPy {
	// Note: Since squeeze_momentum is a complex indicator that requires multiple lookback periods,
	// streaming is not straightforward and would require implementing a complex state management.
	// This implementation maintains a bounded buffer and recomputes on each update.
	// Unlike ALMA which has O(1) streaming updates, this has O(n) complexity where n is the lookback period.
	// TODO: Consider implementing true streaming with state management for better performance.
	length_bb: usize,
	mult_bb: f64,
	length_kc: usize,
	mult_kc: f64,
	data_buffer: Vec<(f64, f64, f64)>, // (high, low, close)
}

#[cfg(feature = "python")]
#[pymethods]
impl SqueezeMomentumStreamPy {
	#[new]
	#[pyo3(signature = (length_bb=None, mult_bb=None, length_kc=None, mult_kc=None))]
	fn new(length_bb: Option<usize>, mult_bb: Option<f64>, length_kc: Option<usize>, mult_kc: Option<f64>) -> Self {
		Self {
			length_bb: length_bb.unwrap_or(20),
			mult_bb: mult_bb.unwrap_or(2.0),
			length_kc: length_kc.unwrap_or(20),
			mult_kc: mult_kc.unwrap_or(1.5),
			data_buffer: Vec::new(),
		}
	}

	fn update(&mut self, high: f64, low: f64, close: f64) -> (Option<f64>, Option<f64>, Option<f64>) {
		self.data_buffer.push((high, low, close));
		
		let needed = self.length_bb.max(self.length_kc);
		if self.data_buffer.len() < needed {
			return (None, None, None);
		}

		// Keep buffer at maximum required size to avoid unbounded growth
		while self.data_buffer.len() > needed * 2 {
			self.data_buffer.remove(0);
		}

		// Extract slices from buffer
		let highs: Vec<f64> = self.data_buffer.iter().map(|(h, _, _)| *h).collect();
		let lows: Vec<f64> = self.data_buffer.iter().map(|(_, l, _)| *l).collect();
		let closes: Vec<f64> = self.data_buffer.iter().map(|(_, _, c)| *c).collect();

		let params = SqueezeMomentumParams {
			length_bb: Some(self.length_bb),
			mult_bb: Some(self.mult_bb),
			length_kc: Some(self.length_kc),
			mult_kc: Some(self.mult_kc),
		};
		let input = SqueezeMomentumInput::from_slices(&highs, &lows, &closes, params);

		match squeeze_momentum(&input) {
			Ok(output) => {
				let last_idx = output.squeeze.len() - 1;
				(
					Some(output.squeeze[last_idx]),
					Some(output.momentum[last_idx]),
					Some(output.momentum_signal[last_idx]),
				)
			}
			Err(_) => (None, None, None),
		}
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "squeeze_momentum_batch")]
#[pyo3(signature = (high, low, close, length_bb_range=None, mult_bb_range=None, length_kc_range=None, mult_kc_range=None, kernel=None))]
pub fn squeeze_momentum_batch_py<'py>(
	py: Python<'py>,
	high: PyReadonlyArray1<'py, f64>,
	low: PyReadonlyArray1<'py, f64>,
	close: PyReadonlyArray1<'py, f64>,
	length_bb_range: Option<(usize, usize, usize)>,
	mult_bb_range: Option<(f64, f64, f64)>,
	length_kc_range: Option<(usize, usize, usize)>,
	mult_kc_range: Option<(f64, f64, f64)>,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
	let high_slice = high.as_slice()?;
	let low_slice = low.as_slice()?;
	let close_slice = close.as_slice()?;
	let kern = validate_kernel(kernel, true)?;

	let sweep = SqueezeMomentumBatchRange {
		length_bb: length_bb_range.unwrap_or((20, 20, 0)),
		mult_bb: mult_bb_range.unwrap_or((2.0, 2.0, 0.0)),
		length_kc: length_kc_range.unwrap_or((20, 20, 0)),
		mult_kc: mult_kc_range.unwrap_or((1.5, 1.5, 0.0)),
	};

	let combos = expand_grid_sm(&sweep);
	let rows = combos.len();
	let cols = close_slice.len();

	// Pre-allocate output array for momentum
	let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let slice_out = unsafe { out_arr.as_slice_mut()? };

	let combos = py
		.allow_threads(|| {
			let kernel = match kern {
				Kernel::Auto => detect_best_batch_kernel(),
				k => k,
			};

			// Map batch kernels to regular kernels if needed
			let simd = match kernel {
				#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
				Kernel::Avx512Batch => Kernel::Avx512,
				#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
				Kernel::Avx2Batch => Kernel::Avx2,
				Kernel::ScalarBatch => Kernel::Scalar,
				_ => kernel,
			};

			// Run batch computation
			let result = squeeze_momentum_batch_with_kernel(high_slice, low_slice, close_slice, &sweep, simd)?;
			
			// Copy momentum values to output
			slice_out.copy_from_slice(&result.momentum);
			
			Ok::<Vec<SqueezeMomentumBatchParams>, SqueezeMomentumError>(result.combos)
		})
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	// Build result dictionary
	let dict = PyDict::new(py);
	dict.set_item("values", out_arr.reshape((rows, cols))?)?;
	
	// Add parameter arrays
	dict.set_item(
		"length_bb",
		combos.iter().map(|p| p.length_bb as u64).collect::<Vec<_>>().into_pyarray(py),
	)?;
	dict.set_item(
		"mult_bb",
		combos.iter().map(|p| p.mult_bb).collect::<Vec<_>>().into_pyarray(py),
	)?;
	dict.set_item(
		"length_kc",
		combos.iter().map(|p| p.length_kc as u64).collect::<Vec<_>>().into_pyarray(py),
	)?;
	dict.set_item(
		"mult_kc",
		combos.iter().map(|p| p.mult_kc).collect::<Vec<_>>().into_pyarray(py),
	)?;

	Ok(dict)
}

// --- WASM Bindings ---

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct SqueezeMomentumResult {
	values: Vec<f64>,
	rows: usize,
	cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl SqueezeMomentumResult {
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
pub fn squeeze_momentum_js(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	length_bb: usize,
	mult_bb: f64,
	length_kc: usize,
	mult_kc: f64,
) -> Result<Vec<f64>, JsValue> {
	let params = SqueezeMomentumParams {
		length_bb: Some(length_bb),
		mult_bb: Some(mult_bb),
		length_kc: Some(length_kc),
		mult_kc: Some(mult_kc),
	};
	
	let input = SqueezeMomentumInput::from_slices(high, low, close, params);
	
	// Single allocation for all three outputs
	let data_len = high.len();
	let mut output = alloc_with_nan_prefix(data_len * 3, 0); // squeeze, momentum, momentum_signal
	
	// Split the output into three slices
	let (squeeze_slice, rest) = output.split_at_mut(data_len);
	let (momentum_slice, momentum_signal_slice) = rest.split_at_mut(data_len);
	
	squeeze_momentum_into_slices(
		squeeze_slice,
		momentum_slice,
		momentum_signal_slice,
		&input,
		detect_best_kernel(),
	)
	.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn squeeze_momentum_into(
	in_ptr: *const f64,
	squeeze_ptr: *mut f64,
	momentum_ptr: *mut f64,
	momentum_signal_ptr: *mut f64,
	len: usize,
	length_bb: usize,
	mult_bb: f64,
	length_kc: usize,
	mult_kc: f64,
) -> Result<(), JsValue> {
	if in_ptr.is_null() || squeeze_ptr.is_null() || momentum_ptr.is_null() || momentum_signal_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}
	
	unsafe {
		// Create slices from input pointers (high, low, close are consecutive)
		let data = std::slice::from_raw_parts(in_ptr, len * 3);
		let high = &data[..len];
		let low = &data[len..len * 2];
		let close = &data[len * 2..];
		
		let params = SqueezeMomentumParams {
			length_bb: Some(length_bb),
			mult_bb: Some(mult_bb),
			length_kc: Some(length_kc),
			mult_kc: Some(mult_kc),
		};
		
		let input = SqueezeMomentumInput::from_slices(high, low, close, params);
		
		// Check for aliasing between input and any output
		let in_ptr_cast = in_ptr as *const u8;
		let squeeze_ptr_cast = squeeze_ptr as *const u8;
		let momentum_ptr_cast = momentum_ptr as *const u8;
		let momentum_signal_ptr_cast = momentum_signal_ptr as *const u8;
		
		// Check if any output aliases with input (considering the full input range)
		let input_start = in_ptr_cast;
		let input_end = in_ptr_cast.add(len * 3 * std::mem::size_of::<f64>());
		
		let needs_temp = (squeeze_ptr_cast >= input_start && squeeze_ptr_cast < input_end) ||
		                 (momentum_ptr_cast >= input_start && momentum_ptr_cast < input_end) ||
		                 (momentum_signal_ptr_cast >= input_start && momentum_signal_ptr_cast < input_end);
		
		if needs_temp {
			// Use temporary buffers if any output aliases with input
			let mut temp_squeeze = alloc_with_nan_prefix(len, 0);
			let mut temp_momentum = alloc_with_nan_prefix(len, 0);
			let mut temp_momentum_signal = alloc_with_nan_prefix(len, 0);
			
			squeeze_momentum_into_slices(
				&mut temp_squeeze,
				&mut temp_momentum,
				&mut temp_momentum_signal,
				&input,
				detect_best_kernel(),
			)
			.map_err(|e| JsValue::from_str(&e.to_string()))?;
			
			let squeeze_out = std::slice::from_raw_parts_mut(squeeze_ptr, len);
			let momentum_out = std::slice::from_raw_parts_mut(momentum_ptr, len);
			let momentum_signal_out = std::slice::from_raw_parts_mut(momentum_signal_ptr, len);
			
			squeeze_out.copy_from_slice(&temp_squeeze);
			momentum_out.copy_from_slice(&temp_momentum);
			momentum_signal_out.copy_from_slice(&temp_momentum_signal);
		} else {
			// Direct write if no aliasing
			let squeeze_out = std::slice::from_raw_parts_mut(squeeze_ptr, len);
			let momentum_out = std::slice::from_raw_parts_mut(momentum_ptr, len);
			let momentum_signal_out = std::slice::from_raw_parts_mut(momentum_signal_ptr, len);
			
			squeeze_momentum_into_slices(
				squeeze_out,
				momentum_out,
				momentum_signal_out,
				&input,
				detect_best_kernel(),
			)
			.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		
		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn squeeze_momentum_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn squeeze_momentum_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe {
			let _ = Vec::from_raw_parts(ptr, len, len);
		}
	}
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct SqueezeMomentumBatchConfig {
	pub length_bb_range: (usize, usize, usize),
	pub mult_bb_range: (f64, f64, f64),
	pub length_kc_range: (usize, usize, usize),
	pub mult_kc_range: (f64, f64, f64),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct SqueezeMomentumBatchJsOutput {
	pub values: Vec<f64>,         // Only momentum values for consistency with Python
	pub rows: usize,               // Number of parameter combinations
	pub cols: usize,               // Data length
	pub length_bb: Vec<usize>,
	pub mult_bb: Vec<f64>,
	pub length_kc: Vec<usize>,
	pub mult_kc: Vec<f64>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = squeeze_momentum_batch)]
pub fn squeeze_momentum_batch_js(
	high: &[f64],
	low: &[f64], 
	close: &[f64],
	config: JsValue,
) -> Result<JsValue, JsValue> {
	let config: SqueezeMomentumBatchConfig = 
		serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	let range = SqueezeMomentumBatchRange {
		length_bb: config.length_bb_range,
		mult_bb: config.mult_bb_range,
		length_kc: config.length_kc_range,
		mult_kc: config.mult_kc_range,
	};
	
	let result = squeeze_momentum_batch_with_kernel(high, low, close, &range, detect_best_batch_kernel())
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	let js_output = SqueezeMomentumBatchJsOutput {
		values: result.momentum,  // Only momentum values, matching Python
		rows: result.combos.len(),
		cols: high.len(),
		length_bb: result.combos.iter().map(|c| c.length_bb).collect(),
		mult_bb: result.combos.iter().map(|c| c.mult_bb).collect(),
		length_kc: result.combos.iter().map(|c| c.length_kc).collect(),
		mult_kc: result.combos.iter().map(|c| c.mult_kc).collect(),
	};
	
	serde_wasm_bindgen::to_value(&js_output).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}
