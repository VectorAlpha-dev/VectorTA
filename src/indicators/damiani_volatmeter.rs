//! # Damiani Volatmeter
//!
//! A volatility indicator using ATR and standard deviation bands to measure trend activity and a threshold for "anti-trend." Supports batch evaluation, parameter sweeps, and SIMD kernels. This indicator follows the API conventions and modular layout of alma.rs, supporting AVX2/AVX512 (with stubs), and is fully unit tested.
//!
//! ## Parameters
//! - **vis_atr**: ATR period for volatility (default 13)
//! - **vis_std**: Std window for volatility (default 20)
//! - **sed_atr**: ATR period for "sedation" (default 40)
//! - **sed_std**: Std window for "sedation" (default 100)
//! - **threshold**: Offset constant (default 1.4)
//!
//! ## Errors
//! - **AllValuesNaN**: all input data values are NaN
//! - **InvalidPeriod**: one or more periods are zero or exceed the data length
//! - **NotEnoughValidData**: not enough valid data points for requested lookback
//!
//! ## Returns
//! - **`Ok(DamianiVolatmeterOutput)`** with `vol`, `anti` arrays, else `Err(DamianiVolatmeterError)`

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
use thiserror::Error;

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

impl<'a> AsRef<[f64]> for DamianiVolatmeterInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			DamianiVolatmeterData::Slice(slice) => slice,
			DamianiVolatmeterData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

#[derive(Debug, Clone)]
pub enum DamianiVolatmeterData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct DamianiVolatmeterOutput {
	pub vol: Vec<f64>,
	pub anti: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct DamianiVolatmeterParams {
	pub vis_atr: Option<usize>,
	pub vis_std: Option<usize>,
	pub sed_atr: Option<usize>,
	pub sed_std: Option<usize>,
	pub threshold: Option<f64>,
}

impl Default for DamianiVolatmeterParams {
	fn default() -> Self {
		Self {
			vis_atr: Some(13),
			vis_std: Some(20),
			sed_atr: Some(40),
			sed_std: Some(100),
			threshold: Some(1.4),
		}
	}
}

#[derive(Debug, Clone)]
pub struct DamianiVolatmeterInput<'a> {
	pub data: DamianiVolatmeterData<'a>,
	pub params: DamianiVolatmeterParams,
}

impl<'a> DamianiVolatmeterInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: DamianiVolatmeterParams) -> Self {
		Self {
			data: DamianiVolatmeterData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slice(sl: &'a [f64], p: DamianiVolatmeterParams) -> Self {
		Self {
			data: DamianiVolatmeterData::Slice(sl),
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", DamianiVolatmeterParams::default())
	}
	#[inline]
	pub fn get_vis_atr(&self) -> usize {
		self.params.vis_atr.unwrap_or(13)
	}
	#[inline]
	pub fn get_vis_std(&self) -> usize {
		self.params.vis_std.unwrap_or(20)
	}
	#[inline]
	pub fn get_sed_atr(&self) -> usize {
		self.params.sed_atr.unwrap_or(40)
	}
	#[inline]
	pub fn get_sed_std(&self) -> usize {
		self.params.sed_std.unwrap_or(100)
	}
	#[inline]
	pub fn get_threshold(&self) -> f64 {
		self.params.threshold.unwrap_or(1.4)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct DamianiVolatmeterBuilder {
	vis_atr: Option<usize>,
	vis_std: Option<usize>,
	sed_atr: Option<usize>,
	sed_std: Option<usize>,
	threshold: Option<f64>,
	kernel: Kernel,
}

impl Default for DamianiVolatmeterBuilder {
	fn default() -> Self {
		Self {
			vis_atr: None,
			vis_std: None,
			sed_atr: None,
			sed_std: None,
			threshold: None,
			kernel: Kernel::Auto,
		}
	}
}

impl DamianiVolatmeterBuilder {
	#[inline(always)]
	pub fn new() -> Self {
		Self::default()
	}
	#[inline(always)]
	pub fn vis_atr(mut self, n: usize) -> Self {
		self.vis_atr = Some(n);
		self
	}
	#[inline(always)]
	pub fn vis_std(mut self, n: usize) -> Self {
		self.vis_std = Some(n);
		self
	}
	#[inline(always)]
	pub fn sed_atr(mut self, n: usize) -> Self {
		self.sed_atr = Some(n);
		self
	}
	#[inline(always)]
	pub fn sed_std(mut self, n: usize) -> Self {
		self.sed_std = Some(n);
		self
	}
	#[inline(always)]
	pub fn threshold(mut self, x: f64) -> Self {
		self.threshold = Some(x);
		self
	}
	#[inline(always)]
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}

	#[inline(always)]
	pub fn apply(self, c: &Candles) -> Result<DamianiVolatmeterOutput, DamianiVolatmeterError> {
		let p = DamianiVolatmeterParams {
			vis_atr: self.vis_atr,
			vis_std: self.vis_std,
			sed_atr: self.sed_atr,
			sed_std: self.sed_std,
			threshold: self.threshold,
		};
		let i = DamianiVolatmeterInput::from_candles(c, "close", p);
		damiani_volatmeter_with_kernel(&i, self.kernel)
	}

	#[inline(always)]
	pub fn apply_slice(self, d: &[f64]) -> Result<DamianiVolatmeterOutput, DamianiVolatmeterError> {
		let p = DamianiVolatmeterParams {
			vis_atr: self.vis_atr,
			vis_std: self.vis_std,
			sed_atr: self.sed_atr,
			sed_std: self.sed_std,
			threshold: self.threshold,
		};
		let i = DamianiVolatmeterInput::from_slice(d, p);
		damiani_volatmeter_with_kernel(&i, self.kernel)
	}

	#[inline(always)]
	pub fn into_stream<'a>(
		self,
		candles: &'a Candles,
		src: &'a str,
	) -> Result<DamianiVolatmeterStream<'a>, DamianiVolatmeterError> {
		let p = DamianiVolatmeterParams {
			vis_atr: self.vis_atr,
			vis_std: self.vis_std,
			sed_atr: self.sed_atr,
			sed_std: self.sed_std,
			threshold: self.threshold,
		};
		DamianiVolatmeterStream::new_from_candles(candles, src, p)
	}
}

#[derive(Debug, Error)]
pub enum DamianiVolatmeterError {
	#[error("damiani_volatmeter: All values are NaN.")]
	AllValuesNaN,
	#[error("damiani_volatmeter: Invalid period: data length = {data_len}, vis_atr = {vis_atr}, vis_std = {vis_std}, sed_atr = {sed_atr}, sed_std = {sed_std}")]
	InvalidPeriod {
		data_len: usize,
		vis_atr: usize,
		vis_std: usize,
		sed_atr: usize,
		sed_std: usize,
	},
	#[error("damiani_volatmeter: Not enough valid data after first non-NaN index. needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("damiani_volatmeter: Empty data provided.")]
	EmptyData,
}

#[inline]
pub fn damiani_volatmeter(input: &DamianiVolatmeterInput) -> Result<DamianiVolatmeterOutput, DamianiVolatmeterError> {
	damiani_volatmeter_with_kernel(input, Kernel::Auto)
}

fn damiani_volatmeter_prepare<'a>(
	input: &'a DamianiVolatmeterInput,
	kernel: Kernel,
) -> Result<
	(
		&'a [f64],  // high
		&'a [f64],  // low
		&'a [f64],  // close
		usize,      // vis_atr
		usize,      // vis_std
		usize,      // sed_atr
		usize,      // sed_std
		f64,        // threshold
		usize,      // first
		usize,      // warmup_period
		Kernel,     // chosen kernel
	),
	DamianiVolatmeterError,
> {
	// Extract three parallel slices: high, low, and close
	let (high, low, close): (&[f64], &[f64], &[f64]) = match &input.data {
		DamianiVolatmeterData::Candles { candles, source: _ } => {
			// For Candles, use "high", "low", and "close"
			let h = source_type(candles, "high");
			let l = source_type(candles, "low");
			let c = source_type(candles, "close");
			(h, l, c)
		}
		DamianiVolatmeterData::Slice(slice) => {
			// If the user only provided a bare &[f64], treat it as "close",
			// and set "high"="low"="close" so that true range = 0 on all bars.
			(slice, slice, slice)
		}
	};

	let len = close.len();
	if len == 0 {
		return Err(DamianiVolatmeterError::EmptyData);
	}

	let vis_atr = input.get_vis_atr();
	let vis_std = input.get_vis_std();
	let sed_atr = input.get_sed_atr();
	let sed_std = input.get_sed_std();
	let threshold = input.get_threshold();

	// Validate zero or out‐of‐bounds periods
	if vis_atr == 0
		|| vis_std == 0
		|| sed_atr == 0
		|| sed_std == 0
		|| vis_atr > len
		|| vis_std > len
		|| sed_atr > len
		|| sed_std > len
	{
		return Err(DamianiVolatmeterError::InvalidPeriod {
			data_len: len,
			vis_atr,
			vis_std,
			sed_atr,
			sed_std,
		});
	}

	// First non‐NaN index in "close"
	let first = close
		.iter()
		.position(|&x| !x.is_nan())
		.ok_or(DamianiVolatmeterError::AllValuesNaN)?;
	let needed = *[vis_atr, vis_std, sed_atr, sed_std, 3].iter().max().unwrap();
	if (len - first) < needed {
		return Err(DamianiVolatmeterError::NotEnoughValidData {
			needed,
			valid: len - first,
		});
	}

	// Choose kernel
	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	// Calculate warmup period
	let warmup_period = *[vis_atr, vis_std, sed_atr, sed_std, 3].iter().max().unwrap();

	Ok((
		high,
		low,
		close,
		vis_atr,
		vis_std,
		sed_atr,
		sed_std,
		threshold,
		first,
		warmup_period,
		chosen,
	))
}

pub fn damiani_volatmeter_with_kernel(
	input: &DamianiVolatmeterInput,
	kernel: Kernel,
) -> Result<DamianiVolatmeterOutput, DamianiVolatmeterError> {
	let (high, low, close, vis_atr, vis_std, sed_atr, sed_std, threshold, first, warmup_period, chosen) = 
		damiani_volatmeter_prepare(input, kernel)?;
	
	let len = close.len();
	let mut vol = alloc_with_nan_prefix(len, warmup_period);
	let mut anti = alloc_with_nan_prefix(len, warmup_period);

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => damiani_volatmeter_scalar(
				high, low, close, vis_atr, vis_std, sed_atr, sed_std, threshold, first, &mut vol, &mut anti,
			),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => damiani_volatmeter_avx2(
				high, low, close, vis_atr, vis_std, sed_atr, sed_std, threshold, first, &mut vol, &mut anti,
			),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => damiani_volatmeter_avx512(
				high, low, close, vis_atr, vis_std, sed_atr, sed_std, threshold, first, &mut vol, &mut anti,
			),
			_ => unreachable!(),
		}
	}

	Ok(DamianiVolatmeterOutput { vol, anti })
}

#[inline]
pub fn damiani_volatmeter_into_slice(
	vol_dst: &mut [f64],
	anti_dst: &mut [f64],
	input: &DamianiVolatmeterInput,
	kernel: Kernel,
) -> Result<(), DamianiVolatmeterError> {
	let (high, low, close, vis_atr, vis_std, sed_atr, sed_std, threshold, first, warmup_period, chosen) = 
		damiani_volatmeter_prepare(input, kernel)?;

	let len = close.len();

	// Validate destination slices have correct length
	if vol_dst.len() != len || anti_dst.len() != len {
		return Err(DamianiVolatmeterError::InvalidPeriod {
			data_len: len,
			vis_atr: vol_dst.len(),
			vis_std: anti_dst.len(),
			sed_atr: 0,
			sed_std: 0,
		});
	}

	// Fill warmup period with NaN
	for i in 0..warmup_period {
		vol_dst[i] = f64::NAN;
		anti_dst[i] = f64::NAN;
	}

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => damiani_volatmeter_scalar(
				high, low, close, vis_atr, vis_std, sed_atr, sed_std, threshold, first, vol_dst, anti_dst,
			),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => damiani_volatmeter_avx2(
				high, low, close, vis_atr, vis_std, sed_atr, sed_std, threshold, first, vol_dst, anti_dst,
			),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => damiani_volatmeter_avx512(
				high, low, close, vis_atr, vis_std, sed_atr, sed_std, threshold, first, vol_dst, anti_dst,
			),
			_ => unreachable!(),
		}
	}

	Ok(())
}

#[inline]
pub unsafe fn damiani_volatmeter_scalar(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	vis_atr: usize,
	vis_std: usize,
	sed_atr: usize,
	sed_std: usize,
	threshold: f64,
	first: usize,
	vol: &mut [f64],
	anti: &mut [f64],
) {
	let len = close.len();
	let mut atr_vis_val = f64::NAN;
	let mut atr_sed_val = f64::NAN;
	let mut sum_vis = 0.0;
	let mut sum_sed = 0.0;

	// “prev_close” starts at close[first], exactly as in the old implementation
	let mut prev_close = close[first];

	// Ring buffers and running sums for the two StdDev windows
	let mut ring_vis = vec![0.0; vis_std];
	let mut ring_sed = vec![0.0; sed_std];
	let mut sum_vis_std = 0.0;
	let mut sum_sq_vis_std = 0.0;
	let mut sum_sed_std = 0.0;
	let mut sum_sq_sed_std = 0.0;
	let mut idx_vis = 0;
	let mut idx_sed = 0;
	let mut filled_vis = 0;
	let mut filled_sed = 0;

	// Lag constant (half lag)
	let lag_s = 0.5_f64;

	for i in 0..len {
		// Compute “True Range” exactly as in the original (high, low, prev_close)
		let tr = if i > 0 {
			let tr1 = high[i] - low[i];
			let tr2 = (high[i] - prev_close).abs();
			let tr3 = (low[i] - prev_close).abs();
			tr1.max(tr2).max(tr3)
		} else {
			0.0
		};
		// Update prev_close to today’s close
		prev_close = close[i];

		// ----- ATR for “vis” line -----
		if i < vis_atr {
			sum_vis += tr;
			if i == vis_atr - 1 {
				atr_vis_val = sum_vis / (vis_atr as f64);
			}
		} else if atr_vis_val.is_finite() {
			atr_vis_val = ((vis_atr as f64 - 1.0) * atr_vis_val + tr) / (vis_atr as f64);
		}

		// ----- ATR for “sed” line -----
		if i < sed_atr {
			sum_sed += tr;
			if i == sed_atr - 1 {
				atr_sed_val = sum_sed / (sed_atr as f64);
			}
		} else if atr_sed_val.is_finite() {
			atr_sed_val = ((sed_atr as f64 - 1.0) * atr_sed_val + tr) / (sed_atr as f64);
		}

		// Insert current “price‐for‐StdDev” (= close, with NaNs treated as 0) into both rings
		let val = if close[i].is_nan() { 0.0 } else { close[i] };
		// —— update “vis” ring buffer
		let old_v = ring_vis[idx_vis];
		ring_vis[idx_vis] = val;
		idx_vis = (idx_vis + 1) % vis_std;
		if filled_vis < vis_std {
			filled_vis += 1;
			sum_vis_std += val;
			sum_sq_vis_std += val * val;
		} else {
			sum_vis_std = sum_vis_std - old_v + val;
			sum_sq_vis_std = sum_sq_vis_std - (old_v * old_v) + (val * val);
		}
		// —— update “sed” ring buffer
		let old_s = ring_sed[idx_sed];
		ring_sed[idx_sed] = val;
		idx_sed = (idx_sed + 1) % sed_std;
		if filled_sed < sed_std {
			filled_sed += 1;
			sum_sed_std += val;
			sum_sq_sed_std += val * val;
		} else {
			sum_sed_std = sum_sed_std - old_s + val;
			sum_sq_sed_std = sum_sq_sed_std - (old_s * old_s) + (val * val);
		}

		// Only start computing “vol” and “anti” once EVERY lookback is satisfied:
		if i >= *[vis_atr, vis_std, sed_atr, sed_std, 3].iter().max().unwrap() {
			// Previous two “vol” values needed for the lag component
			let p1 = if i >= 1 && !vol[i - 1].is_nan() {
				vol[i - 1]
			} else {
				0.0
			};
			let p3 = if i >= 3 && !vol[i - 3].is_nan() {
				vol[i - 3]
			} else {
				0.0
			};

			// Safely handle ATR Sed = 0 or NaN
			let sed_safe = if atr_sed_val.is_finite() && atr_sed_val != 0.0 {
				atr_sed_val
			} else {
				atr_sed_val + f64::EPSILON
			};

			// Compute “vol[i]” exactly as the old version did
			vol[i] = (atr_vis_val / sed_safe) + lag_s * (p1 - p3);

			// Only compute “anti[i]” once both StdDev windows are completely filled
			if filled_vis == vis_std && filled_sed == sed_std {
				// Population‐stddev for “vis” window
				let mean_vis = sum_vis_std / (vis_std as f64);
				let mean_sq_vis = sum_sq_vis_std / (vis_std as f64);
				let var_vis = (mean_sq_vis - mean_vis * mean_vis).max(0.0);
				let std_vis = var_vis.sqrt();

				// Population‐stddev for “sed” window
				let mean_sed = sum_sed_std / (sed_std as f64);
				let mean_sq_sed = sum_sq_sed_std / (sed_std as f64);
				let var_sed = (mean_sq_sed - mean_sed * mean_sed).max(0.0);
				let std_sed = var_sed.sqrt();

				let ratio = if std_sed != 0.0 {
					std_vis / std_sed
				} else {
					std_vis / (std_sed + f64::EPSILON)
				};
				anti[i] = threshold - ratio;
			}
		}
	}
}

#[inline]
fn stddev(sum: f64, sum_sq: f64, n: usize) -> f64 {
	if n == 0 {
		return 0.0;
	}
	let mean = sum / n as f64;
	let mean_sq = sum_sq / n as f64;
	let var = mean_sq - mean * mean;
	if var <= 0.0 {
		0.0
	} else {
		var.sqrt()
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn damiani_volatmeter_avx512(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	vis_atr: usize,
	vis_std: usize,
	sed_atr: usize,
	sed_std: usize,
	threshold: f64,
	first: usize,
	vol: &mut [f64],
	anti: &mut [f64],
) {
	damiani_volatmeter_scalar(
		high, low, close, vis_atr, vis_std, sed_atr, sed_std, threshold, first, vol, anti,
	)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn damiani_volatmeter_avx2(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	vis_atr: usize,
	vis_std: usize,
	sed_atr: usize,
	sed_std: usize,
	threshold: f64,
	first: usize,
	vol: &mut [f64],
	anti: &mut [f64],
) {
	damiani_volatmeter_scalar(
		high, low, close, vis_atr, vis_std, sed_atr, sed_std, threshold, first, vol, anti,
	)
}

pub fn damiani_volatmeter_batch_with_kernel(
	data: &[f64],
	sweep: &DamianiVolatmeterBatchRange,
	k: Kernel,
) -> Result<DamianiVolatmeterBatchOutput, DamianiVolatmeterError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => {
			return Err(DamianiVolatmeterError::InvalidPeriod {
				data_len: 0,
				vis_atr: 0,
				vis_std: 0,
				sed_atr: 0,
				sed_std: 0,
			})
		}
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	damiani_volatmeter_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct DamianiVolatmeterBatchRange {
	pub vis_atr: (usize, usize, usize),
	pub vis_std: (usize, usize, usize),
	pub sed_atr: (usize, usize, usize),
	pub sed_std: (usize, usize, usize),
	pub threshold: (f64, f64, f64),
}
impl Default for DamianiVolatmeterBatchRange {
	fn default() -> Self {
		Self {
			vis_atr: (13, 40, 1),
			vis_std: (20, 40, 1),
			sed_atr: (40, 40, 0),
			sed_std: (100, 100, 0),
			threshold: (1.4, 1.4, 0.0),
		}
	}
}
#[derive(Clone, Debug, Default)]
pub struct DamianiVolatmeterBatchBuilder {
	range: DamianiVolatmeterBatchRange,
	kernel: Kernel,
}
impl DamianiVolatmeterBatchBuilder {
	pub fn new() -> Self {
		Self::default()
	}
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	pub fn vis_atr_range(mut self, s: usize, e: usize, step: usize) -> Self {
		self.range.vis_atr = (s, e, step);
		self
	}
	pub fn vis_std_range(mut self, s: usize, e: usize, step: usize) -> Self {
		self.range.vis_std = (s, e, step);
		self
	}
	pub fn sed_atr_range(mut self, s: usize, e: usize, step: usize) -> Self {
		self.range.sed_atr = (s, e, step);
		self
	}
	pub fn sed_std_range(mut self, s: usize, e: usize, step: usize) -> Self {
		self.range.sed_std = (s, e, step);
		self
	}
	pub fn threshold_range(mut self, s: f64, e: f64, step: f64) -> Self {
		self.range.threshold = (s, e, step);
		self
	}

	pub fn apply_slice(self, data: &[f64]) -> Result<DamianiVolatmeterBatchOutput, DamianiVolatmeterError> {
		damiani_volatmeter_batch_with_kernel(data, &self.range, self.kernel)
	}
	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<DamianiVolatmeterBatchOutput, DamianiVolatmeterError> {
		let slice = source_type(c, src);
		self.apply_slice(slice)
	}
}

#[derive(Clone, Debug)]
pub struct DamianiVolatmeterBatchOutput {
	pub vol: Vec<f64>,
	pub anti: Vec<f64>,
	pub combos: Vec<DamianiVolatmeterParams>,
	pub rows: usize,
	pub cols: usize,
}
impl DamianiVolatmeterBatchOutput {
	pub fn row_for_params(&self, p: &DamianiVolatmeterParams) -> Option<usize> {
		self.combos.iter().position(|c| {
			c.vis_atr == p.vis_atr
				&& c.vis_std == p.vis_std
				&& c.sed_atr == p.sed_atr
				&& c.sed_std == p.sed_std
				&& (c.threshold.unwrap_or(1.4) - p.threshold.unwrap_or(1.4)).abs() < 1e-12
		})
	}
	pub fn vol_for(&self, p: &DamianiVolatmeterParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.vol[start..start + self.cols]
		})
	}
	pub fn anti_for(&self, p: &DamianiVolatmeterParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.anti[start..start + self.cols]
		})
	}
}
#[inline(always)]
pub fn damiani_volatmeter_batch_slice(
	data: &[f64],
	sweep: &DamianiVolatmeterBatchRange,
	kern: Kernel,
) -> Result<DamianiVolatmeterBatchOutput, DamianiVolatmeterError> {
	damiani_volatmeter_batch_inner(data, sweep, kern, false)
}
#[inline(always)]
pub fn damiani_volatmeter_batch_par_slice(
	data: &[f64],
	sweep: &DamianiVolatmeterBatchRange,
	kern: Kernel,
) -> Result<DamianiVolatmeterBatchOutput, DamianiVolatmeterError> {
	damiani_volatmeter_batch_inner(data, sweep, kern, true)
}
fn expand_grid(r: &DamianiVolatmeterBatchRange) -> Vec<DamianiVolatmeterParams> {
	fn axis_usize((s, e, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || s == e {
			return vec![s];
		}
		(s..=e).step_by(step).collect()
	}
	fn axis_f64((s, e, step): (f64, f64, f64)) -> Vec<f64> {
		if step.abs() < 1e-12 || (s - e).abs() < 1e-12 {
			return vec![s];
		}
		let mut v = Vec::new();
		let mut x = s;
		while x <= e + 1e-12 {
			v.push(x);
			x += step;
		}
		v
	}
	let vis_atrs = axis_usize(r.vis_atr);
	let vis_stds = axis_usize(r.vis_std);
	let sed_atrs = axis_usize(r.sed_atr);
	let sed_stds = axis_usize(r.sed_std);
	let thresholds = axis_f64(r.threshold);
	let mut out =
		Vec::with_capacity(vis_atrs.len() * vis_stds.len() * sed_atrs.len() * sed_stds.len() * thresholds.len());
	for &va in &vis_atrs {
		for &vs in &vis_stds {
			for &sa in &sed_atrs {
				for &ss in &sed_stds {
					for &th in &thresholds {
						out.push(DamianiVolatmeterParams {
							vis_atr: Some(va),
							vis_std: Some(vs),
							sed_atr: Some(sa),
							sed_std: Some(ss),
							threshold: Some(th),
						});
					}
				}
			}
		}
	}
	out
}
#[inline(always)]
fn damiani_volatmeter_batch_inner(
	data: &[f64],
	sweep: &DamianiVolatmeterBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<DamianiVolatmeterBatchOutput, DamianiVolatmeterError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(DamianiVolatmeterError::InvalidPeriod {
			data_len: 0,
			vis_atr: 0,
			vis_std: 0,
			sed_atr: 0,
			sed_std: 0,
		});
	}
	let first = data
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(DamianiVolatmeterError::AllValuesNaN)?;
	let max_p = combos
		.iter()
		.map(|c| {
			*[
				c.vis_atr.unwrap(),
				c.vis_std.unwrap(),
				c.sed_atr.unwrap(),
				c.sed_std.unwrap(),
			]
			.iter()
			.max()
			.unwrap()
		})
		.max()
		.unwrap();
	if data.len() - first < max_p {
		return Err(DamianiVolatmeterError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}
	let rows = combos.len();
	let cols = data.len();
	
	// Allocate uninitialized matrices
	let mut vol_mu = make_uninit_matrix(rows, cols);
	let mut anti_mu = make_uninit_matrix(rows, cols);
	
	// Calculate warmup periods for each combination
	let warmup_periods: Vec<usize> = combos
		.iter()
		.map(|p| {
			*[
				p.vis_atr.unwrap(),
				p.vis_std.unwrap(),
				p.sed_atr.unwrap(),
				p.sed_std.unwrap(),
				3
			].iter().max().unwrap()
		})
		.collect();
	
	// Initialize NaN prefixes
	init_matrix_prefixes(&mut vol_mu, cols, &warmup_periods);
	init_matrix_prefixes(&mut anti_mu, cols, &warmup_periods);
	
	// Convert to mutable slices
	let mut vol_guard = core::mem::ManuallyDrop::new(vol_mu);
	let mut anti_guard = core::mem::ManuallyDrop::new(anti_mu);
	let vol: &mut [f64] = unsafe { core::slice::from_raw_parts_mut(vol_guard.as_mut_ptr() as *mut f64, vol_guard.len()) };
	let anti: &mut [f64] = unsafe { core::slice::from_raw_parts_mut(anti_guard.as_mut_ptr() as *mut f64, anti_guard.len()) };

	let do_row = |row: usize, out_vol: &mut [f64], out_anti: &mut [f64]| unsafe {
		let prm = &combos[row];
		match kern {
			Kernel::Scalar => damiani_volatmeter_row_scalar(
				data,
				first,
				prm.vis_atr.unwrap(),
				prm.vis_std.unwrap(),
				prm.sed_atr.unwrap(),
				prm.sed_std.unwrap(),
				prm.threshold.unwrap(),
				out_vol,
				out_anti,
			),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => damiani_volatmeter_row_avx2(
				data,
				first,
				prm.vis_atr.unwrap(),
				prm.vis_std.unwrap(),
				prm.sed_atr.unwrap(),
				prm.sed_std.unwrap(),
				prm.threshold.unwrap(),
				out_vol,
				out_anti,
			),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => damiani_volatmeter_row_avx512(
				data,
				first,
				prm.vis_atr.unwrap(),
				prm.vis_std.unwrap(),
				prm.sed_atr.unwrap(),
				prm.sed_std.unwrap(),
				prm.threshold.unwrap(),
				out_vol,
				out_anti,
			),
			_ => unreachable!(),
		}
	};
	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			vol.par_chunks_mut(cols)
				.zip(anti.par_chunks_mut(cols))
				.enumerate()
				.for_each(|(row, (outv, outa))| do_row(row, outv, outa));
		}

		#[cfg(target_arch = "wasm32")]
		{
			for (row, (outv, outa)) in vol.chunks_mut(cols).zip(anti.chunks_mut(cols)).enumerate() {
				do_row(row, outv, outa);
			}
		}
	} else {
		for (row, (outv, outa)) in vol.chunks_mut(cols).zip(anti.chunks_mut(cols)).enumerate() {
			do_row(row, outv, outa);
		}
	}
	
	// Convert back to owned vectors
	let vol = unsafe {
		Vec::from_raw_parts(
			vol_guard.as_mut_ptr() as *mut f64,
			vol_guard.len(),
			vol_guard.capacity(),
		)
	};
	
	let anti = unsafe {
		Vec::from_raw_parts(
			anti_guard.as_mut_ptr() as *mut f64,
			anti_guard.len(),
			anti_guard.capacity(),
		)
	};
	
	Ok(DamianiVolatmeterBatchOutput {
		vol,
		anti,
		combos,
		rows,
		cols,
	})
}

/// Version of batch computation that writes directly to pre-allocated slices for Python bindings
#[inline(always)]
fn damiani_volatmeter_batch_inner_into(
	data: &[f64],
	sweep: &DamianiVolatmeterBatchRange,
	kern: Kernel,
	parallel: bool,
	vol_out: &mut [f64],
	anti_out: &mut [f64],
) -> Result<Vec<DamianiVolatmeterParams>, DamianiVolatmeterError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(DamianiVolatmeterError::InvalidPeriod {
			data_len: 0,
			vis_atr: 0,
			vis_std: 0,
			sed_atr: 0,
			sed_std: 0,
		});
	}
	let first = data
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(DamianiVolatmeterError::AllValuesNaN)?;
	let max_p = combos
		.iter()
		.flat_map(|p| {
			[
				p.vis_atr.unwrap_or(13),
				p.vis_std.unwrap_or(20),
				p.sed_atr.unwrap_or(40),
				p.sed_std.unwrap_or(100),
				3,
			]
		})
		.max()
		.unwrap_or(100);
	if (data.len() - first) < max_p {
		return Err(DamianiVolatmeterError::NotEnoughValidData {
			needed: max_p,
			valid: data.len() - first,
		});
	}
	let rows = combos.len();
	let cols = data.len();
	let total_size = rows * cols;
	
	// Ensure output slices have the correct size
	if vol_out.len() != total_size || anti_out.len() != total_size {
		return Err(DamianiVolatmeterError::InvalidPeriod {
			data_len: data.len(),
			vis_atr: 0,
			vis_std: 0,
			sed_atr: 0,
			sed_std: 0,
		});
	}

	let do_row = |row: usize, out_vol: &mut [f64], out_anti: &mut [f64]| {
		let p = &combos[row];
		let input = DamianiVolatmeterInput::from_slice(data, p.clone());
		match kern {
			Kernel::Scalar | Kernel::ScalarBatch => scalar::damiani_volatmeter_scalar(
				&input,
				out_vol,
				out_anti,
			),
			Kernel::Avx2 | Kernel::Avx2Batch => avx2::damiani_volatmeter_avx2(
				&input,
				out_vol,
				out_anti,
			),
			Kernel::Avx512 | Kernel::Avx512Batch => avx512::damiani_volatmeter_avx512(
				&input,
				out_vol,
				out_anti,
			),
			_ => unreachable!(),
		}
	};
	
	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			vol_out.par_chunks_mut(cols)
				.zip(anti_out.par_chunks_mut(cols))
				.enumerate()
				.for_each(|(row, (outv, outa))| do_row(row, outv, outa));
		}

		#[cfg(target_arch = "wasm32")]
		{
			for (row, (outv, outa)) in vol_out.chunks_mut(cols).zip(anti_out.chunks_mut(cols)).enumerate() {
				do_row(row, outv, outa);
			}
		}
	} else {
		for (row, (outv, outa)) in vol_out.chunks_mut(cols).zip(anti_out.chunks_mut(cols)).enumerate() {
			do_row(row, outv, outa);
		}
	}
	
	Ok(combos)
}

#[inline(always)]
pub unsafe fn damiani_volatmeter_row_scalar(
	data: &[f64],
	first: usize,
	vis_atr: usize,
	vis_std: usize,
	sed_atr: usize,
	sed_std: usize,
	threshold: f64,
	vol: &mut [f64],
	anti: &mut [f64],
) {
	// Call the new scalar signature by passing `data` for high, low, and close:
	damiani_volatmeter_scalar(
		data, // high
		data, // low
		data, // close
		vis_atr, vis_std, sed_atr, sed_std, threshold, first, vol, anti,
	)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn damiani_volatmeter_row_avx2(
	data: &[f64],
	first: usize,
	vis_atr: usize,
	vis_std: usize,
	sed_atr: usize,
	sed_std: usize,
	threshold: f64,
	vol: &mut [f64],
	anti: &mut [f64],
) {
	damiani_volatmeter_scalar(
		data, // high
		data, // low
		data, // close
		vis_atr, vis_std, sed_atr, sed_std, threshold, first, vol, anti,
	)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn damiani_volatmeter_row_avx512(
	data: &[f64],
	first: usize,
	vis_atr: usize,
	vis_std: usize,
	sed_atr: usize,
	sed_std: usize,
	threshold: f64,
	vol: &mut [f64],
	anti: &mut [f64],
) {
	damiani_volatmeter_scalar(
		data, // high
		data, // low
		data, // close
		vis_atr, vis_std, sed_atr, sed_std, threshold, first, vol, anti,
	)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn damiani_volatmeter_row_avx512_short(
	data: &[f64],
	first: usize,
	vis_atr: usize,
	vis_std: usize,
	sed_atr: usize,
	sed_std: usize,
	threshold: f64,
	vol: &mut [f64],
	anti: &mut [f64],
) {
	damiani_volatmeter_scalar(
		data, // high
		data, // low
		data, // close
		vis_atr, vis_std, sed_atr, sed_std, threshold, first, vol, anti,
	)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn damiani_volatmeter_row_avx512_long(
	data: &[f64],
	first: usize,
	vis_atr: usize,
	vis_std: usize,
	sed_atr: usize,
	sed_std: usize,
	threshold: f64,
	vol: &mut [f64],
	anti: &mut [f64],
) {
	damiani_volatmeter_scalar(
		data, // high
		data, // low
		data, // close
		vis_atr, vis_std, sed_atr, sed_std, threshold, first, vol, anti,
	)
}

#[derive(Debug, Clone)]
pub struct DamianiVolatmeterStream<'a> {
	// The three parallel slices:
	high: &'a [f64],
	low: &'a [f64],
	close: &'a [f64],

	vis_atr: usize,
	vis_std: usize,
	sed_atr: usize,
	sed_std: usize,
	threshold: f64,

	// "index" = which bar we are on, 0..len-1
	index: usize,

	// Running state:
	atr_vis_val: f64,
	atr_sed_val: f64,
	sum_vis: f64,
	sum_sed: f64,
	prev_close: f64,

	ring_vis: Vec<f64>,
	ring_sed: Vec<f64>,
	sum_vis_std: f64,
	sum_sq_vis_std: f64,
	sum_sed_std: f64,
	sum_sq_sed_std: f64,
	idx_vis: usize,
	idx_sed: usize,
	filled_vis: usize,
	filled_sed: usize,

	// Lag values for vol: we need to maintain vol[i-1], vol[i-2], vol[i-3]
	vol_history: [f64; 3], // Changed from p1, p3 to a proper buffer
	lag_s: f64,
}

impl<'a> DamianiVolatmeterStream<'a> {
	pub fn new_from_candles(
		candles: &'a Candles,
		src: &'a str,
		params: DamianiVolatmeterParams,
	) -> Result<Self, DamianiVolatmeterError> {
		let high = source_type(candles, "high");
		let low = source_type(candles, "low");
		let close = source_type(candles, src);

		let len = close.len();
		if len == 0 {
			return Err(DamianiVolatmeterError::EmptyData);
		}

		let vis_atr = params.vis_atr.unwrap_or(13);
		let vis_std = params.vis_std.unwrap_or(20);
		let sed_atr = params.sed_atr.unwrap_or(40);
		let sed_std = params.sed_std.unwrap_or(100);
		let threshold = params.threshold.unwrap_or(1.4);

		if vis_atr == 0
			|| vis_std == 0
			|| sed_atr == 0
			|| sed_std == 0
			|| vis_atr > len
			|| vis_std > len
			|| sed_atr > len
			|| sed_std > len
		{
			return Err(DamianiVolatmeterError::InvalidPeriod {
				data_len: len,
				vis_atr,
				vis_std,
				sed_atr,
				sed_std,
			});
		}

		let first = close
			.iter()
			.position(|&x| !x.is_nan())
			.ok_or(DamianiVolatmeterError::AllValuesNaN)?;
		let needed = *[vis_atr, vis_std, sed_atr, sed_std, 3].iter().max().unwrap();
		if (len - first) < needed {
			return Err(DamianiVolatmeterError::NotEnoughValidData {
				needed,
				valid: len - first,
			});
		}

		// Initialize prev_close to the first non-NaN close value, matching scalar
		let initial_prev_close = close[first];

		Ok(Self {
			high,
			low,
			close,
			vis_atr,
			vis_std,
			sed_atr,
			sed_std,
			threshold,

			index: 0,
			atr_vis_val: f64::NAN,
			atr_sed_val: f64::NAN,
			sum_vis: 0.0,
			sum_sed: 0.0,
			prev_close: initial_prev_close,

			ring_vis: vec![0.0; vis_std],
			ring_sed: vec![0.0; sed_std],
			sum_vis_std: 0.0,
			sum_sq_vis_std: 0.0,
			sum_sed_std: 0.0,
			sum_sq_sed_std: 0.0,
			idx_vis: 0,
			idx_sed: 0,
			filled_vis: 0,
			filled_sed: 0,

			vol_history: [f64::NAN; 3], // Initialize with NaN
			lag_s: 0.5,
		})
	}

	pub fn update(&mut self) -> Option<(f64, f64)> {
		let i = self.index;
		let len = self.close.len();
		if i >= len {
			return None;
		}

		// Compute "True Range" exactly like scalar - for i=0, tr=0
		let tr = if i > 0 {
			let hi = self.high[i];
			let lo = self.low[i];
			let pc = self.prev_close;

			let tr1 = hi - lo;
			let tr2 = (hi - pc).abs();
			let tr3 = (lo - pc).abs();
			tr1.max(tr2).max(tr3)
		} else {
			0.0
		};

		// Update prev_close after computing tr
		self.prev_close = self.close[i];

		// ----- ATR for "vis" -----
		if i < self.vis_atr {
			self.sum_vis += tr;
			if i == self.vis_atr - 1 {
				self.atr_vis_val = self.sum_vis / (self.vis_atr as f64);
			}
		} else if self.atr_vis_val.is_finite() {
			self.atr_vis_val = ((self.vis_atr as f64 - 1.0) * self.atr_vis_val + tr) / (self.vis_atr as f64);
		}

		// ----- ATR for "sed" -----
		if i < self.sed_atr {
			self.sum_sed += tr;
			if i == self.sed_atr - 1 {
				self.atr_sed_val = self.sum_sed / (self.sed_atr as f64);
			}
		} else if self.atr_sed_val.is_finite() {
			self.atr_sed_val = ((self.sed_atr as f64 - 1.0) * self.atr_sed_val + tr) / (self.sed_atr as f64);
		}

		// ---- Insert price‐for‐StdDev (use close, treat NaN as 0) ----
		let val = if self.close[i].is_nan() { 0.0 } else { self.close[i] };

		// Update "vis" ring buffer:
		let old_v = self.ring_vis[self.idx_vis];
		self.ring_vis[self.idx_vis] = val;
		self.idx_vis = (self.idx_vis + 1) % self.vis_std;
		if self.filled_vis < self.vis_std {
			self.filled_vis += 1;
			self.sum_vis_std += val;
			self.sum_sq_vis_std += val * val;
		} else {
			self.sum_vis_std = self.sum_vis_std - old_v + val;
			self.sum_sq_vis_std = self.sum_sq_vis_std - (old_v * old_v) + (val * val);
		}

		// Update "sed" ring buffer:
		let old_s = self.ring_sed[self.idx_sed];
		self.ring_sed[self.idx_sed] = val;
		self.idx_sed = (self.idx_sed + 1) % self.sed_std;
		if self.filled_sed < self.sed_std {
			self.filled_sed += 1;
			self.sum_sed_std += val;
			self.sum_sq_sed_std += val * val;
		} else {
			self.sum_sed_std = self.sum_sed_std - old_s + val;
			self.sum_sq_sed_std = self.sum_sq_sed_std - (old_s * old_s) + (val * val);
		}

		// Increment index for next call
		self.index += 1;

		// Only start computing vol/anti once we have _all_ lookbacks:
		let needed = *[self.vis_atr, self.vis_std, self.sed_atr, self.sed_std, 3]
			.iter()
			.max()
			.unwrap();
		if i < needed {
			return None;
		}

		// Get previous vol values from history
		let p1 = if !self.vol_history[0].is_nan() {
			self.vol_history[0]
		} else {
			0.0
		};
		let p3 = if !self.vol_history[2].is_nan() {
			self.vol_history[2]
		} else {
			0.0
		};

		// Avoid divide‐by‐zero on sed:
		let sed_safe = if self.atr_sed_val.is_finite() && self.atr_sed_val != 0.0 {
			self.atr_sed_val
		} else {
			self.atr_sed_val + f64::EPSILON
		};

		// Compute vol[i]:
		let vol_val = (self.atr_vis_val / sed_safe) + self.lag_s * (p1 - p3);

		// Shift the vol history buffer
		self.vol_history[2] = self.vol_history[1];
		self.vol_history[1] = self.vol_history[0];
		self.vol_history[0] = vol_val;

		// Compute anti[i] only if both StdDev windows are full:
		let anti_val = if self.filled_vis == self.vis_std && self.filled_sed == self.sed_std {
			let std_vis = stddev(self.sum_vis_std, self.sum_sq_vis_std, self.vis_std);
			let std_sed = stddev(self.sum_sed_std, self.sum_sq_sed_std, self.sed_std);
			let ratio = if std_sed != 0.0 {
				std_vis / std_sed
			} else {
				std_vis / (std_sed + f64::EPSILON)
			};
			self.threshold - ratio
		} else {
			f64::NAN
		};

		Some((vol_val, anti_val))
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;
	#[cfg(feature = "proptest")]
	use proptest::prelude::*;
	fn check_damiani_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = DamianiVolatmeterParams::default();
		let input = DamianiVolatmeterInput::from_candles(&candles, "close", params);
		let output = damiani_volatmeter_with_kernel(&input, kernel)?;
		assert_eq!(output.vol.len(), candles.close.len());
		assert_eq!(output.anti.len(), candles.close.len());
		Ok(())
	}
	fn check_damiani_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = DamianiVolatmeterInput::from_candles(&candles, "close", DamianiVolatmeterParams::default());
		let output = damiani_volatmeter_with_kernel(&input, kernel)?;
		let n = output.vol.len();
		let expected_vol = [
			0.9009485470514558,
			0.8333604467044887,
			0.815318380178986,
			0.8276892636184923,
			0.879447954127426,
		];
		let expected_anti = [
			1.1227721577887388,
			1.1250333024152703,
			1.1325501989919875,
			1.1403866079746106,
			1.1392919184055932,
		];
		let start = n - 5;
		for i in 0..5 {
			let diff_vol = (output.vol[start + i] - expected_vol[i]).abs();
			let diff_anti = (output.anti[start + i] - expected_anti[i]).abs();
			assert!(
				diff_vol < 1e-2,
				"vol mismatch at index {}: expected {}, got {}",
				start + i,
				expected_vol[i],
				output.vol[start + i]
			);
			assert!(
				diff_anti < 1e-2,
				"anti mismatch at index {}: expected {}, got {}",
				start + i,
				expected_anti[i],
				output.anti[start + i]
			);
		}
		Ok(())
	}
	fn check_damiani_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let mut params = DamianiVolatmeterParams::default();
		params.vis_atr = Some(0);
		let input = DamianiVolatmeterInput::from_candles(&candles, "close", params);
		let res = damiani_volatmeter_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] should fail with zero period", test_name);
		Ok(())
	}
	fn check_damiani_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let mut params = DamianiVolatmeterParams::default();
		params.vis_atr = Some(99999);
		let input = DamianiVolatmeterInput::from_candles(&candles, "close", params);
		let res = damiani_volatmeter_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] should fail if period exceeds length", test_name);
		Ok(())
	}
	fn check_damiani_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data = [42.0];
		let params = DamianiVolatmeterParams {
			vis_atr: Some(9),
			vis_std: Some(9),
			sed_atr: Some(9),
			sed_std: Some(9),
			threshold: Some(1.4),
		};
		let input = DamianiVolatmeterInput::from_slice(&data, params);
		let res = damiani_volatmeter_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] should fail with insufficient data", test_name);
		Ok(())
	}
	fn check_damiani_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = DamianiVolatmeterInput::from_candles(&candles, "close", DamianiVolatmeterParams::default());
		let batch = damiani_volatmeter_with_kernel(&input, kernel)?;

		let mut stream =
			DamianiVolatmeterStream::new_from_candles(&candles, "close", DamianiVolatmeterParams::default())?;

		let mut stream_vol = Vec::with_capacity(candles.close.len());
		let mut stream_anti = Vec::with_capacity(candles.close.len());

		for _ in 0..candles.close.len() {
			if let Some((v, a)) = stream.update() {
				stream_vol.push(v);
				stream_anti.push(a);
			} else {
				stream_vol.push(f64::NAN);
				stream_anti.push(f64::NAN);
			}
		}

		for (i, (&bv, &sv)) in batch.vol.iter().zip(stream_vol.iter()).enumerate() {
			if bv.is_nan() && sv.is_nan() {
				continue;
			}
			let diff = (bv - sv).abs();
			assert!(
				diff < 1e-8,
				"[{}] streaming vol mismatch at idx {}: batch={}, stream={}",
				test_name,
				i,
				bv,
				sv
			);
		}

		for (i, (&ba, &sa)) in batch.anti.iter().zip(stream_anti.iter()).enumerate() {
			if ba.is_nan() && sa.is_nan() {
				continue;
			}
			let diff = (ba - sa).abs();
			assert!(
				diff < 1e-8,
				"[{}] streaming anti mismatch at idx {}: batch={}, stream={}",
				test_name,
				i,
				ba,
				sa
			);
		}

		Ok(())
	}

	fn check_damiani_input_with_default_candles(
		_test_name: &str,
		_kernel: Kernel,
	) -> Result<(), Box<dyn std::error::Error>> {
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = DamianiVolatmeterInput::with_default_candles(&candles);
		match input.data {
			DamianiVolatmeterData::Candles { source, .. } => {
				assert_eq!(source, "close");
			}
			_ => panic!("Expected DamianiVolatmeterData::Candles"),
		}
		Ok(())
	}
	fn check_damiani_params_with_defaults(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		let default_params = DamianiVolatmeterParams::default();
		assert_eq!(default_params.vis_atr, Some(13));
		assert_eq!(default_params.vis_std, Some(20));
		assert_eq!(default_params.sed_atr, Some(40));
		assert_eq!(default_params.sed_std, Some(100));
		assert_eq!(default_params.threshold, Some(1.4));
		Ok(())
	}
	fn check_batch_default_row(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let output = DamianiVolatmeterBatchBuilder::new()
			.kernel(kernel)
			.apply_candles(&c, "close")?;
		let def = DamianiVolatmeterParams::default();
		let vol_row = output.vol_for(&def).expect("default vol row missing");
		let anti_row = output.anti_for(&def).expect("default anti row missing");
		assert_eq!(vol_row.len(), c.close.len());
		assert_eq!(anti_row.len(), c.close.len());

		// Compute expected values using the same "close-only" approach
		let close_slice = source_type(&c, "close");
		let input = DamianiVolatmeterInput::from_slice(close_slice, def.clone());
		let expected_output = damiani_volatmeter(&input)?;

		// Compare the last 5 values
		let start = vol_row.len() - 5;
		for i in 0..5 {
			let idx = start + i;
			assert!(
				(vol_row[idx] - expected_output.vol[idx]).abs() < 1e-10,
				"[{test_name}] default-vol-row mismatch at idx {i}: batch={} vs expected={}",
				vol_row[idx],
				expected_output.vol[idx]
			);
			assert!(
				(anti_row[idx] - expected_output.anti[idx]).abs() < 1e-10,
				"[{test_name}] default-anti-row mismatch at idx {i}: batch={} vs expected={}",
				anti_row[idx],
				expected_output.anti[idx]
			);
		}
		Ok(())
	}
	
	fn check_damiani_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let empty: [f64; 0] = [];
		let params = DamianiVolatmeterParams::default();
		let input = DamianiVolatmeterInput::from_slice(&empty, params);
		let res = damiani_volatmeter_with_kernel(&input, kernel);
		assert!(
			matches!(res, Err(DamianiVolatmeterError::EmptyData)),
			"[{}] should fail with empty input",
			test_name
		);
		Ok(())
	}
	
	fn check_damiani_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data = vec![f64::NAN; 200];
		let params = DamianiVolatmeterParams::default();
		let input = DamianiVolatmeterInput::from_slice(&data, params);
		let res = damiani_volatmeter_with_kernel(&input, kernel);
		assert!(
			matches!(res, Err(DamianiVolatmeterError::AllValuesNaN)),
			"[{}] should fail with all NaN values",
			test_name
		);
		Ok(())
	}
	
	fn check_damiani_invalid_threshold(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		
		// Test with NaN threshold
		let mut params = DamianiVolatmeterParams::default();
		params.threshold = Some(f64::NAN);
		let input = DamianiVolatmeterInput::from_candles(&candles, "close", params);
		let res = damiani_volatmeter_with_kernel(&input, kernel);
		// Unlike alma which has explicit threshold validation, damiani will compute but produce NaN results
		// This is acceptable behavior - the indicator doesn't fail but produces NaN outputs
		assert!(res.is_ok(), "[{}] should not fail with NaN threshold", test_name);
		
		// Test with negative threshold (which is mathematically valid for this indicator)
		params.threshold = Some(-1.0);
		let input2 = DamianiVolatmeterInput::from_candles(&candles, "close", params);
		let res2 = damiani_volatmeter_with_kernel(&input2, kernel);
		assert!(res2.is_ok(), "[{}] should work with negative threshold", test_name);
		Ok(())
	}
	
	fn check_damiani_invalid_periods(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
		
		// Test with zero vis_atr
		let mut params = DamianiVolatmeterParams::default();
		params.vis_atr = Some(0);
		let input = DamianiVolatmeterInput::from_slice(&data, params);
		let res = damiani_volatmeter_with_kernel(&input, kernel);
		assert!(
			matches!(res, Err(DamianiVolatmeterError::InvalidPeriod { .. })),
			"[{}] should fail with zero vis_atr",
			test_name
		);
		
		// Test with zero vis_std
		params = DamianiVolatmeterParams::default();
		params.vis_std = Some(0);
		let input2 = DamianiVolatmeterInput::from_slice(&data, params);
		let res2 = damiani_volatmeter_with_kernel(&input2, kernel);
		assert!(
			matches!(res2, Err(DamianiVolatmeterError::InvalidPeriod { .. })),
			"[{}] should fail with zero vis_std",
			test_name
		);
		
		// Test with period exceeding data length
		params = DamianiVolatmeterParams::default();
		params.sed_std = Some(1000);
		let input3 = DamianiVolatmeterInput::from_slice(&data, params);
		let res3 = damiani_volatmeter_with_kernel(&input3, kernel);
		assert!(
			matches!(res3, Err(DamianiVolatmeterError::InvalidPeriod { .. })),
			"[{}] should fail with period exceeding length",
			test_name
		);
		Ok(())
	}
	
	fn check_damiani_into_existing_slice(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = DamianiVolatmeterParams::default();
		let input = DamianiVolatmeterInput::from_candles(&candles, "close", params);
		
		// Compute using regular function
		let output1 = damiani_volatmeter_with_kernel(&input, kernel)?;
		
		// Compute into existing slices
		let mut vol2 = vec![0.0; candles.close.len()];
		let mut anti2 = vec![0.0; candles.close.len()];
		damiani_volatmeter_into_slice(&mut vol2, &mut anti2, &input, kernel)?;
		
		// Compare results
		assert_eq!(output1.vol.len(), vol2.len());
		assert_eq!(output1.anti.len(), anti2.len());
		
		// Check values match
		for i in 0..output1.vol.len() {
			if output1.vol[i].is_nan() && vol2[i].is_nan() {
				continue;
			}
			assert!(
				(output1.vol[i] - vol2[i]).abs() < 1e-10,
				"[{}] vol mismatch at index {}: {} vs {}",
				test_name, i, output1.vol[i], vol2[i]
			);
		}
		
		for i in 0..output1.anti.len() {
			if output1.anti[i].is_nan() && anti2[i].is_nan() {
				continue;
			}
			assert!(
				(output1.anti[i] - anti2[i]).abs() < 1e-10,
				"[{}] anti mismatch at index {}: {} vs {}",
				test_name, i, output1.anti[i], anti2[i]
			);
		}
		Ok(())
	}
	
	#[cfg(feature = "proptest")]
	fn check_damiani_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		
		proptest!(|(
			data in prop::collection::vec(any::<f64>().prop_filter("Not NaN", |x| !x.is_nan()), 200..=1000),
			vis_atr in 5usize..20,
			vis_std in 10usize..30,
			sed_atr in 20usize..50,
			sed_std in 50usize..150,
			threshold in 0.5f64..3.0,
		)| {
			let params = DamianiVolatmeterParams {
				vis_atr: Some(vis_atr),
				vis_std: Some(vis_std),
				sed_atr: Some(sed_atr),
				sed_std: Some(sed_std),
				threshold: Some(threshold),
			};
			let input = DamianiVolatmeterInput::from_slice(&data, params);
			
			// Test that the function doesn't panic and produces output of correct length
			match damiani_volatmeter_with_kernel(&input, kernel) {
				Ok(output) => {
					prop_assert_eq!(output.vol.len(), data.len());
					prop_assert_eq!(output.anti.len(), data.len());
					
					// Check that warmup period is respected
					let warmup = *[vis_atr, vis_std, sed_atr, sed_std, 3].iter().max().unwrap();
					
					// Vol values should start appearing after warmup
					let first_non_nan_vol = output.vol.iter().position(|&x| !x.is_nan());
					if let Some(idx) = first_non_nan_vol {
						prop_assert!(idx >= warmup - 1);
					}
					
					// Anti values need both stddev windows filled, so may appear later
					let first_non_nan_anti = output.anti.iter().position(|&x| !x.is_nan());
					if let Some(idx) = first_non_nan_anti {
						prop_assert!(idx >= warmup - 1);
					}
				}
				Err(DamianiVolatmeterError::InvalidPeriod { .. }) => {
					// This can happen if generated periods exceed data length
					prop_assert!(sed_std > data.len() || sed_atr > data.len() || 
					            vis_std > data.len() || vis_atr > data.len());
				}
				Err(e) => {
					panic!("Unexpected error: {:?}", e);
				}
			}
		});
		
		Ok(())
	}
	
	macro_rules! generate_all_damiani_tests {
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
	generate_all_damiani_tests!(
		check_damiani_partial_params,
		check_damiani_accuracy,
		check_damiani_zero_period,
		check_damiani_period_exceeds_length,
		check_damiani_very_small_dataset,
		check_damiani_streaming,
		check_damiani_input_with_default_candles,
		check_damiani_params_with_defaults,
		check_damiani_empty_input,
		check_damiani_all_nan,
		check_damiani_invalid_threshold,
		check_damiani_invalid_periods,
		check_damiani_into_existing_slice
	);
	
	#[cfg(feature = "proptest")]
	generate_all_damiani_tests!(check_damiani_property);
	
	gen_batch_tests!(check_batch_default_row);
}

#[cfg(feature = "python")]
#[pyfunction(name = "damiani_volatmeter")]
#[pyo3(signature = (data, vis_atr, vis_std, sed_atr, sed_std, threshold, kernel=None))]
pub fn damiani_volatmeter_py<'py>(
	py: Python<'py>,
	data: PyReadonlyArray1<'py, f64>,
	vis_atr: usize,
	vis_std: usize,
	sed_atr: usize,
	sed_std: usize,
	threshold: f64,
	kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
	use numpy::{IntoPyArray, PyArrayMethods};

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, false)?;
	
	let params = DamianiVolatmeterParams {
		vis_atr: Some(vis_atr),
		vis_std: Some(vis_std),
		sed_atr: Some(sed_atr),
		sed_std: Some(sed_std),
		threshold: Some(threshold),
	};
	let input = DamianiVolatmeterInput::from_slice(slice_in, params);

	// Get both vectors from Rust function
	let (vol_vec, anti_vec): (Vec<f64>, Vec<f64>) = py.allow_threads(|| {
		damiani_volatmeter_with_kernel(&input, kern)
			.map(|o| (o.vol, o.anti))
	})
	.map_err(|e| PyValueError::new_err(e.to_string()))?;

	// Zero-copy transfer for both outputs
	Ok((
		vol_vec.into_pyarray(py),
		anti_vec.into_pyarray(py)
	))
}

#[cfg(feature = "python")]
#[pyclass(name = "DamianiVolatmeterStream")]
pub struct DamianiVolatmeterStreamPy {
	stream: DamianiVolatmeterStream<'static>,
	high: Vec<f64>,
	low: Vec<f64>,
	close: Vec<f64>,
}

#[cfg(feature = "python")]
#[pymethods]
impl DamianiVolatmeterStreamPy {
	#[new]
	fn new(
		high: Vec<f64>,
		low: Vec<f64>,
		close: Vec<f64>,
		vis_atr: usize,
		vis_std: usize,
		sed_atr: usize,
		sed_std: usize,
		threshold: f64,
	) -> PyResult<Self> {
		// Store the data in the struct
		let mut instance = Self { stream: unsafe { std::mem::zeroed() }, high, low, close };
		
		// Create candles from the stored data
		let candles = Candles {
			open: vec![0.0; instance.close.len()],  // Not used
			high: instance.high.clone(),
			low: instance.low.clone(),
			close: instance.close.clone(),
			volume: vec![0.0; instance.close.len()],  // Not used
		};
		
		let params = DamianiVolatmeterParams {
			vis_atr: Some(vis_atr),
			vis_std: Some(vis_std),
			sed_atr: Some(sed_atr),
			sed_std: Some(sed_std),
			threshold: Some(threshold),
		};
		
		// Use transmute to convert the lifetime
		let stream: DamianiVolatmeterStream<'static> = unsafe {
			std::mem::transmute(
				DamianiVolatmeterStream::new_from_candles(&candles, "close", params)
					.map_err(|e| PyValueError::new_err(e.to_string()))?
			)
		};
		
		instance.stream = stream;
		Ok(instance)
	}

	fn update(&mut self) -> Option<(f64, f64)> {
		self.stream.update()
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "damiani_volatmeter_batch")]
#[pyo3(signature = (data, vis_atr_range, vis_std_range, sed_atr_range, sed_std_range, threshold_range, kernel=None))]
pub fn damiani_volatmeter_batch_py<'py>(
	py: Python<'py>,
	data: PyReadonlyArray1<'py, f64>,
	vis_atr_range: (usize, usize, usize),
	vis_std_range: (usize, usize, usize),
	sed_atr_range: (usize, usize, usize),
	sed_std_range: (usize, usize, usize),
	threshold_range: (f64, f64, f64),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, true)?; // Validate before allow_threads

	let sweep = DamianiVolatmeterBatchRange {
		vis_atr: vis_atr_range,
		vis_std: vis_std_range,
		sed_atr: sed_atr_range,
		sed_std: sed_std_range,
		threshold: threshold_range,
	};

	// Calculate dimensions
	let combos = expand_grid(&sweep);
	let rows = combos.len();
	let cols = slice_in.len();

	// Pre-allocate output arrays (OK for batch operations)
	let vol_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let anti_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let vol_slice = unsafe { vol_arr.as_slice_mut()? };
	let anti_slice = unsafe { anti_arr.as_slice_mut()? };

	// Compute without GIL
	let combos = py.allow_threads(|| {
		// Handle kernel selection for batch operations
		let kernel = match kern {
			Kernel::Auto => detect_best_batch_kernel(),
			k => k,
		};
		let simd = match kernel {
			Kernel::Avx512Batch => Kernel::Avx512,
			Kernel::Avx2Batch => Kernel::Avx2,
			Kernel::ScalarBatch => Kernel::Scalar,
			_ => kernel,
		};
		
		// Use the new _into function that writes directly to pre-allocated slices
		damiani_volatmeter_batch_inner_into(slice_in, &sweep, simd, true, vol_slice, anti_slice)
	})
	.map_err(|e| PyValueError::new_err(e.to_string()))?;

	// Build result dictionary
	let dict = PyDict::new(py);
	dict.set_item("vol", vol_arr.reshape((rows, cols))?)?;
	dict.set_item("anti", anti_arr.reshape((rows, cols))?)?;
	
	// Add parameter arrays
	dict.set_item(
		"vis_atr",
		combos.iter()
			.map(|p| p.vis_atr.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py)
	)?;
	dict.set_item(
		"vis_std",
		combos.iter()
			.map(|p| p.vis_std.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py)
	)?;
	dict.set_item(
		"sed_atr",
		combos.iter()
			.map(|p| p.sed_atr.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py)
	)?;
	dict.set_item(
		"sed_std",
		combos.iter()
			.map(|p| p.sed_std.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py)
	)?;
	dict.set_item(
		"thresholds",
		combos.iter()
			.map(|p| p.threshold.unwrap())
			.collect::<Vec<_>>()
			.into_pyarray(py)
	)?;

	Ok(dict)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn damiani_volatmeter_js(
	data: &[f64],
	vis_atr: usize,
	vis_std: usize,
	sed_atr: usize,
	sed_std: usize,
	threshold: f64,
) -> Result<Vec<f64>, JsValue> {
	let params = DamianiVolatmeterParams {
		vis_atr: Some(vis_atr),
		vis_std: Some(vis_std),
		sed_atr: Some(sed_atr),
		sed_std: Some(sed_std),
		threshold: Some(threshold),
	};
	let input = DamianiVolatmeterInput::from_slice(data, params);

	// Single allocation for both outputs
	let mut result = vec![0.0; data.len() * 2];
	let (vol_part, anti_part) = result.split_at_mut(data.len());

	damiani_volatmeter_into_slice(vol_part, anti_part, &input, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;

	Ok(result)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn damiani_volatmeter_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn damiani_volatmeter_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe {
			let _ = Vec::from_raw_parts(ptr, len, len);
		}
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn damiani_volatmeter_into(
	in_ptr: *const f64,
	vol_ptr: *mut f64,
	anti_ptr: *mut f64,
	len: usize,
	vis_atr: usize,
	vis_std: usize,
	sed_atr: usize,
	sed_std: usize,
	threshold: f64,
) -> Result<(), JsValue> {
	if in_ptr.is_null() || vol_ptr.is_null() || anti_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}

	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);
		let params = DamianiVolatmeterParams {
			vis_atr: Some(vis_atr),
			vis_std: Some(vis_std),
			sed_atr: Some(sed_atr),
			sed_std: Some(sed_std),
			threshold: Some(threshold),
		};
		let input = DamianiVolatmeterInput::from_slice(data, params);

		// Check for aliasing - need to handle all possible combinations
		let vol_aliased = in_ptr == vol_ptr;
		let anti_aliased = in_ptr == anti_ptr;
		let vol_anti_aliased = vol_ptr == anti_ptr;

		if vol_anti_aliased {
			return Err(JsValue::from_str("vol_ptr and anti_ptr cannot be the same"));
		}

		if vol_aliased || anti_aliased {
			// Need temporary buffer(s) for aliased outputs
			let mut temp_vol = if vol_aliased { vec![0.0; len] } else { vec![] };
			let mut temp_anti = if anti_aliased { vec![0.0; len] } else { vec![] };

			let vol_out = if vol_aliased {
				&mut temp_vol
			} else {
				std::slice::from_raw_parts_mut(vol_ptr, len)
			};

			let anti_out = if anti_aliased {
				&mut temp_anti
			} else {
				std::slice::from_raw_parts_mut(anti_ptr, len)
			};

			damiani_volatmeter_into_slice(vol_out, anti_out, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;

			// Copy back if aliased
			if vol_aliased {
				let vol = std::slice::from_raw_parts_mut(vol_ptr, len);
				vol.copy_from_slice(&temp_vol);
			}
			if anti_aliased {
				let anti = std::slice::from_raw_parts_mut(anti_ptr, len);
				anti.copy_from_slice(&temp_anti);
			}
		} else {
			// No aliasing, direct computation
			let vol = std::slice::from_raw_parts_mut(vol_ptr, len);
			let anti = std::slice::from_raw_parts_mut(anti_ptr, len);
			damiani_volatmeter_into_slice(vol, anti, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct DamianiVolatmeterBatchConfig {
	pub vis_atr_range: (usize, usize, usize),
	pub vis_std_range: (usize, usize, usize),
	pub sed_atr_range: (usize, usize, usize),
	pub sed_std_range: (usize, usize, usize),
	pub threshold_range: (f64, f64, f64),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct DamianiVolatmeterBatchJsOutput {
	pub vol: Vec<f64>,
	pub anti: Vec<f64>,
	pub combos: Vec<DamianiVolatmeterParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = damiani_volatmeter_batch)]
pub fn damiani_volatmeter_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	let config: DamianiVolatmeterBatchConfig =
		serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

	let sweep = DamianiVolatmeterBatchRange {
		vis_atr: config.vis_atr_range,
		vis_std: config.vis_std_range,
		sed_atr: config.sed_atr_range,
		sed_std: config.sed_std_range,
		threshold: config.threshold_range,
	};

	let output = damiani_volatmeter_batch_inner(data, &sweep, Kernel::Auto, false)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;

	let js_output = DamianiVolatmeterBatchJsOutput {
		vol: output.vol,
		anti: output.anti,
		combos: output.combos,
		rows: output.rows,
		cols: output.cols,
	};

	serde_wasm_bindgen::to_value(&js_output).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn damiani_volatmeter_batch_into(
	in_ptr: *const f64,
	vol_ptr: *mut f64,
	anti_ptr: *mut f64,
	len: usize,
	vis_atr_start: usize,
	vis_atr_end: usize,
	vis_atr_step: usize,
	vis_std_start: usize,
	vis_std_end: usize,
	vis_std_step: usize,
	sed_atr_start: usize,
	sed_atr_end: usize,
	sed_atr_step: usize,
	sed_std_start: usize,
	sed_std_end: usize,
	sed_std_step: usize,
	threshold_start: f64,
	threshold_end: f64,
	threshold_step: f64,
) -> Result<usize, JsValue> {
	if in_ptr.is_null() || vol_ptr.is_null() || anti_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}

	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);

		let sweep = DamianiVolatmeterBatchRange {
			vis_atr: (vis_atr_start, vis_atr_end, vis_atr_step),
			vis_std: (vis_std_start, vis_std_end, vis_std_step),
			sed_atr: (sed_atr_start, sed_atr_end, sed_atr_step),
			sed_std: (sed_std_start, sed_std_end, sed_std_step),
			threshold: (threshold_start, threshold_end, threshold_step),
		};

		let combos = expand_grid(&sweep);
		let rows = combos.len();
		let cols = len;

		// Check for aliasing
		if vol_ptr == anti_ptr {
			return Err(JsValue::from_str("vol_ptr and anti_ptr cannot be the same"));
		}
		
		if in_ptr == vol_ptr || in_ptr == anti_ptr {
			// Need temporary buffers for aliasing
			let result = damiani_volatmeter_batch_inner(data, &sweep, Kernel::Auto, false)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			
			let vol_out = std::slice::from_raw_parts_mut(vol_ptr, rows * cols);
			let anti_out = std::slice::from_raw_parts_mut(anti_ptr, rows * cols);
			
			vol_out.copy_from_slice(&result.vol);
			anti_out.copy_from_slice(&result.anti);
		} else {
			// No aliasing, direct computation
			let vol_out = std::slice::from_raw_parts_mut(vol_ptr, rows * cols);
			let anti_out = std::slice::from_raw_parts_mut(anti_ptr, rows * cols);
			
			damiani_volatmeter_batch_inner_into(data, &sweep, Kernel::Auto, false, vol_out, anti_out)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}

		Ok(rows)
	}
}
