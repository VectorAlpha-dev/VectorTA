//! # Keltner Channels
//!
//! A volatility-based envelope indicator. The middle band is a moving average (MA) of a user-specified source,
//! and the upper and lower bands are derived by adding or subtracting a multiple of an internally computed Average True Range (ATR).
//!
//! ## Parameters
//! - **period**: Lookback length for both the moving average and ATR (default: 20).
//! - **multiplier**: ATR multiplier for upper/lower bands (default: 2.0).
//! - **ma_type**: MA type ("ema", "sma", etc.; default: "ema").
//!
//! ## Errors
//! - **KeltnerEmptyData**: keltner: Input data is empty.
//! - **KeltnerInvalidPeriod**: keltner: `period` is zero or exceeds data length.
//! - **KeltnerNotEnoughValidData**: keltner: Not enough valid data after first valid index.
//! - **KeltnerAllValuesNaN**: keltner: All values are NaN.
//! - **KeltnerMaError**: keltner: MA error.
//!
//! ## Returns
//! - **Ok(KeltnerOutput)**: Contains upper_band, middle_band, lower_band.
//! - **Err(KeltnerError)**

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

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

// Input & data types

#[derive(Debug, Clone)]
pub enum KeltnerData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64], &'a [f64], &'a [f64], &'a [f64]), // high, low, close, source
}

#[derive(Debug, Clone)]
pub struct KeltnerOutput {
	pub upper_band: Vec<f64>,
	pub middle_band: Vec<f64>,
	pub lower_band: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct KeltnerParams {
	pub period: Option<usize>,
	pub multiplier: Option<f64>,
	pub ma_type: Option<String>,
}

impl Default for KeltnerParams {
	fn default() -> Self {
		Self {
			period: Some(20),
			multiplier: Some(2.0),
			ma_type: Some("ema".to_string()),
		}
	}
}

#[derive(Debug, Clone)]
pub struct KeltnerInput<'a> {
	pub data: KeltnerData<'a>,
	pub params: KeltnerParams,
}

impl<'a> KeltnerInput<'a> {
	#[inline]
	pub fn from_candles(candles: &'a Candles, source: &'a str, params: KeltnerParams) -> Self {
		Self {
			data: KeltnerData::Candles { candles, source },
			params,
		}
	}
	#[inline]
	pub fn from_slice(
		high: &'a [f64],
		low: &'a [f64],
		close: &'a [f64],
		source: &'a [f64],
		params: KeltnerParams,
	) -> Self {
		Self {
			data: KeltnerData::Slice(high, low, close, source),
			params,
		}
	}
	#[inline]
	pub fn with_default_candles(candles: &'a Candles) -> Self {
		Self::from_candles(candles, "close", KeltnerParams::default())
	}
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(20)
	}
	#[inline]
	pub fn get_multiplier(&self) -> f64 {
		self.params.multiplier.unwrap_or(2.0)
	}
	#[inline]
	pub fn get_ma_type(&self) -> &str {
		self.params.ma_type.as_deref().unwrap_or("ema")
	}
}

impl<'a> AsRef<[f64]> for KeltnerInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			KeltnerData::Slice(_, _, _, source) => source,
			KeltnerData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

#[derive(Clone, Debug)]
pub struct KeltnerBuilder {
	period: Option<usize>,
	multiplier: Option<f64>,
	ma_type: Option<String>,
	kernel: Kernel,
}

impl Default for KeltnerBuilder {
	fn default() -> Self {
		Self {
			period: None,
			multiplier: None,
			ma_type: None,
			kernel: Kernel::Auto,
		}
	}
}

impl KeltnerBuilder {
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
	pub fn multiplier(mut self, x: f64) -> Self {
		self.multiplier = Some(x);
		self
	}
	#[inline(always)]
	pub fn ma_type(mut self, mt: &str) -> Self {
		self.ma_type = Some(mt.to_lowercase());
		self
	}
	#[inline(always)]
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}

	#[inline(always)]
	pub fn apply(self, c: &Candles) -> Result<KeltnerOutput, KeltnerError> {
		let p = KeltnerParams {
			period: self.period,
			multiplier: self.multiplier,
			ma_type: self.ma_type,
		};
		let i = KeltnerInput::from_candles(c, "close", p);
		keltner_with_kernel(&i, self.kernel)
	}
	
	#[inline(always)]
	pub fn apply_slice(self, high: &[f64], low: &[f64], close: &[f64], source: &[f64]) -> Result<KeltnerOutput, KeltnerError> {
		let p = KeltnerParams {
			period: self.period,
			multiplier: self.multiplier,
			ma_type: self.ma_type,
		};
		let i = KeltnerInput::from_slice(high, low, close, source, p);
		keltner_with_kernel(&i, self.kernel)
	}
	
	#[inline(always)]
	pub fn into_stream(self) -> Result<KeltnerStream, KeltnerError> {
		let p = KeltnerParams {
			period: self.period,
			multiplier: self.multiplier,
			ma_type: self.ma_type,
		};
		KeltnerStream::try_new(p)
	}
}

// Error handling

#[derive(Debug, Error)]
pub enum KeltnerError {
	#[error("keltner: empty data provided.")]
	KeltnerEmptyData,
	#[error("keltner: invalid period: period = {period}, data length = {data_len}")]
	KeltnerInvalidPeriod { period: usize, data_len: usize },
	#[error("keltner: not enough valid data: needed = {needed}, valid = {valid}")]
	KeltnerNotEnoughValidData { needed: usize, valid: usize },
	#[error("keltner: all values are NaN.")]
	KeltnerAllValuesNaN,
	#[error("keltner: MA error: {0}")]
	KeltnerMaError(String),
}

// Core indicator API

#[inline]
pub fn keltner(input: &KeltnerInput) -> Result<KeltnerOutput, KeltnerError> {
	keltner_with_kernel(input, Kernel::Auto)
}

pub fn keltner_with_kernel(input: &KeltnerInput, kernel: Kernel) -> Result<KeltnerOutput, KeltnerError> {
	let (high, low, close, source_slice): (&[f64], &[f64], &[f64], &[f64]) = match &input.data {
		KeltnerData::Candles { candles, source } => (
			candles
				.select_candle_field("high")
				.map_err(|e| KeltnerError::KeltnerMaError(e.to_string()))?,
			candles
				.select_candle_field("low")
				.map_err(|e| KeltnerError::KeltnerMaError(e.to_string()))?,
			candles
				.select_candle_field("close")
				.map_err(|e| KeltnerError::KeltnerMaError(e.to_string()))?,
			source_type(candles, source),
		),
		KeltnerData::Slice(h, l, c, s) => (*h, *l, *c, *s),
	};
	let period = input.get_period();
	let len = close.len();
	if len == 0 {
		return Err(KeltnerError::KeltnerEmptyData);
	}
	if period == 0 || period > len {
		return Err(KeltnerError::KeltnerInvalidPeriod { period, data_len: len });
	}
	let first = close
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(KeltnerError::KeltnerAllValuesNaN)?;

	if (len - first) < period {
		return Err(KeltnerError::KeltnerNotEnoughValidData {
			needed: period,
			valid: len - first,
		});
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	let warmup_period = period - 1;
	let mut upper_band = alloc_with_nan_prefix(len, warmup_period);
	let mut middle_band = alloc_with_nan_prefix(len, warmup_period);
	let mut lower_band = alloc_with_nan_prefix(len, warmup_period);

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => keltner_scalar(
				high,
				low,
				close,
				source_slice,
				period,
				input.get_multiplier(),
				input.get_ma_type(),
				first,
				&mut upper_band,
				&mut middle_band,
				&mut lower_band,
			),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => keltner_avx2(
				high,
				low,
				close,
				source_slice,
				period,
				input.get_multiplier(),
				input.get_ma_type(),
				first,
				&mut upper_band,
				&mut middle_band,
				&mut lower_band,
			),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => keltner_avx512(
				high,
				low,
				close,
				source_slice,
				period,
				input.get_multiplier(),
				input.get_ma_type(),
				first,
				&mut upper_band,
				&mut middle_band,
				&mut lower_band,
			),
			_ => unreachable!(),
		}
	}
	Ok(KeltnerOutput {
		upper_band,
		middle_band,
		lower_band,
	})
}

/// Write keltner output directly to pre-allocated slices - zero allocations
#[inline(always)]
pub fn keltner_into_slice(
	upper_dst: &mut [f64],
	middle_dst: &mut [f64],
	lower_dst: &mut [f64],
	input: &KeltnerInput,
	kernel: Kernel,
) -> Result<(), KeltnerError> {
	let (high, low, close, source_slice): (&[f64], &[f64], &[f64], &[f64]) = match &input.data {
		KeltnerData::Candles { candles, source } => (
			candles
				.select_candle_field("high")
				.map_err(|e| KeltnerError::KeltnerMaError(e.to_string()))?,
			candles
				.select_candle_field("low")
				.map_err(|e| KeltnerError::KeltnerMaError(e.to_string()))?,
			candles
				.select_candle_field("close")
				.map_err(|e| KeltnerError::KeltnerMaError(e.to_string()))?,
			source_type(candles, source),
		),
		KeltnerData::Slice(h, l, c, s) => (*h, *l, *c, *s),
	};
	
	let period = input.get_period();
	let len = close.len();
	
	if len == 0 {
		return Err(KeltnerError::KeltnerEmptyData);
	}
	
	if upper_dst.len() != len || middle_dst.len() != len || lower_dst.len() != len {
		return Err(KeltnerError::KeltnerInvalidPeriod { period: 0, data_len: len });
	}
	
	if period == 0 || period > len {
		return Err(KeltnerError::KeltnerInvalidPeriod { period, data_len: len });
	}
	
	let first = close
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(KeltnerError::KeltnerAllValuesNaN)?;

	if (len - first) < period {
		return Err(KeltnerError::KeltnerNotEnoughValidData {
			needed: period,
			valid: len - first,
		});
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => keltner_scalar(
				high,
				low,
				close,
				source_slice,
				period,
				input.get_multiplier(),
				input.get_ma_type(),
				first,
				upper_dst,
				middle_dst,
				lower_dst,
			),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => keltner_avx2(
				high,
				low,
				close,
				source_slice,
				period,
				input.get_multiplier(),
				input.get_ma_type(),
				first,
				upper_dst,
				middle_dst,
				lower_dst,
			),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => keltner_avx512(
				high,
				low,
				close,
				source_slice,
				period,
				input.get_multiplier(),
				input.get_ma_type(),
				first,
				upper_dst,
				middle_dst,
				lower_dst,
			),
			#[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
			Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => keltner_scalar(
				high,
				low,
				close,
				source_slice,
				period,
				input.get_multiplier(),
				input.get_ma_type(),
				first,
				upper_dst,
				middle_dst,
				lower_dst,
			),
			_ => unreachable!(),
		}
	}
	
	// Fill warmup with NaN
	let warmup_period = period - 1;
	for i in 0..warmup_period {
		upper_dst[i] = f64::NAN;
		middle_dst[i] = f64::NAN;
		lower_dst[i] = f64::NAN;
	}
	
	Ok(())
}

// Scalar calculation (core logic)

#[inline]
pub fn keltner_scalar(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	source: &[f64],
	period: usize,
	multiplier: f64,
	ma_type: &str,
	first: usize,
	upper: &mut [f64],
	middle: &mut [f64],
	lower: &mut [f64],
) {
	let len = close.len();
	let mut atr = alloc_with_nan_prefix(len, period - 1);
	let alpha = 1.0 / (period as f64);
	let mut sum_tr = 0.0;
	let mut rma = f64::NAN;
	for i in 0..len {
		let tr = if i == 0 {
			high[0] - low[0]
		} else {
			let hl = high[i] - low[i];
			let hc = (high[i] - close[i - 1]).abs();
			let lc = (low[i] - close[i - 1]).abs();
			hl.max(hc).max(lc)
		};
		if i < period {
			sum_tr += tr;
			if i == period - 1 {
				rma = sum_tr / (period as f64);
				atr[i] = rma;
			}
		} else {
			rma += alpha * (tr - rma);
			atr[i] = rma;
		}
	}
	// Pre-allocate MA values buffer to avoid allocation
	let mut ma_values = alloc_with_nan_prefix(len, period - 1);
	
	// Use into_slice functions for common MA types to avoid allocation
	match ma_type {
		"ema" => {
			use crate::indicators::moving_averages::ema::{ema_into_slice, EmaInput, EmaData, EmaParams};
			let ema_input = EmaInput {
				data: EmaData::Slice(source),
				params: EmaParams { period: Some(period) },
			};
			let _ = ema_into_slice(&mut ma_values, &ema_input, Kernel::Auto);
		},
		"sma" => {
			use crate::indicators::moving_averages::sma::{sma_into_slice, SmaInput, SmaData, SmaParams};
			let sma_input = SmaInput {
				data: SmaData::Slice(source),
				params: SmaParams { period: Some(period) },
			};
			let _ = sma_into_slice(&mut ma_values, &sma_input, Kernel::Auto);
		},
		_ => {
			// Fallback for other MA types - this allocates but is unavoidable
			if let Ok(result) = crate::indicators::moving_averages::ma::ma(
				ma_type,
				crate::indicators::moving_averages::ma::MaData::Slice(source),
				period,
			) {
				// Copy the result into our pre-allocated buffer
				ma_values.copy_from_slice(&result);
			}
		}
	}
	
	for i in (first + period - 1)..len {
		let ma_v = ma_values[i];
		let atr_v = atr[i];
		if ma_v.is_nan() || atr_v.is_nan() {
			continue;
		}
		middle[i] = ma_v;
		upper[i] = ma_v + multiplier * atr_v;
		lower[i] = ma_v - multiplier * atr_v;
	}
}

// AVX2 stub

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn keltner_avx2(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	source: &[f64],
	period: usize,
	multiplier: f64,
	ma_type: &str,
	first: usize,
	upper: &mut [f64],
	middle: &mut [f64],
	lower: &mut [f64],
) {
	keltner_scalar(
		high, low, close, source, period, multiplier, ma_type, first, upper, middle, lower,
	)
}

// AVX512 stub

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn keltner_avx512(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	source: &[f64],
	period: usize,
	multiplier: f64,
	ma_type: &str,
	first: usize,
	upper: &mut [f64],
	middle: &mut [f64],
	lower: &mut [f64],
) {
	if period <= 32 {
		unsafe {
			keltner_avx512_short(
				high, low, close, source, period, multiplier, ma_type, first, upper, middle, lower,
			)
		}
	} else {
		unsafe {
			keltner_avx512_long(
				high, low, close, source, period, multiplier, ma_type, first, upper, middle, lower,
			)
		}
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn keltner_avx512_short(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	source: &[f64],
	period: usize,
	multiplier: f64,
	ma_type: &str,
	first: usize,
	upper: &mut [f64],
	middle: &mut [f64],
	lower: &mut [f64],
) {
	keltner_scalar(
		high, low, close, source, period, multiplier, ma_type, first, upper, middle, lower,
	)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn keltner_avx512_long(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	source: &[f64],
	period: usize,
	multiplier: f64,
	ma_type: &str,
	first: usize,
	upper: &mut [f64],
	middle: &mut [f64],
	lower: &mut [f64],
) {
	keltner_scalar(
		high, low, close, source, period, multiplier, ma_type, first, upper, middle, lower,
	)
}

// Row/batch/parallel support

#[inline(always)]
pub fn keltner_row_scalar(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	source: &[f64],
	period: usize,
	multiplier: f64,
	ma_type: &str,
	first: usize,
	upper: &mut [f64],
	middle: &mut [f64],
	lower: &mut [f64],
) {
	keltner_scalar(
		high, low, close, source, period, multiplier, ma_type, first, upper, middle, lower,
	)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn keltner_row_avx2(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	source: &[f64],
	period: usize,
	multiplier: f64,
	ma_type: &str,
	first: usize,
	upper: &mut [f64],
	middle: &mut [f64],
	lower: &mut [f64],
) {
	keltner_avx2(
		high, low, close, source, period, multiplier, ma_type, first, upper, middle, lower,
	)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn keltner_row_avx512(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	source: &[f64],
	period: usize,
	multiplier: f64,
	ma_type: &str,
	first: usize,
	upper: &mut [f64],
	middle: &mut [f64],
	lower: &mut [f64],
) {
	keltner_avx512(
		high, low, close, source, period, multiplier, ma_type, first, upper, middle, lower,
	)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn keltner_row_avx512_short(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	source: &[f64],
	period: usize,
	multiplier: f64,
	ma_type: &str,
	first: usize,
	upper: &mut [f64],
	middle: &mut [f64],
	lower: &mut [f64],
) {
	keltner_avx512_short(
		high, low, close, source, period, multiplier, ma_type, first, upper, middle, lower,
	)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn keltner_row_avx512_long(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	source: &[f64],
	period: usize,
	multiplier: f64,
	ma_type: &str,
	first: usize,
	upper: &mut [f64],
	middle: &mut [f64],
	lower: &mut [f64],
) {
	keltner_avx512_long(
		high, low, close, source, period, multiplier, ma_type, first, upper, middle, lower,
	)
}

// Batch support

#[derive(Clone, Debug)]
pub struct KeltnerBatchRange {
	pub period: (usize, usize, usize),
	pub multiplier: (f64, f64, f64),
}

impl Default for KeltnerBatchRange {
	fn default() -> Self {
		Self {
			period: (20, 60, 10),
			multiplier: (2.0, 2.0, 0.0),
		}
	}
}

#[derive(Clone, Debug)]
pub struct KeltnerBatchBuilder {
	range: KeltnerBatchRange,
	kernel: Kernel,
}

impl Default for KeltnerBatchBuilder {
	fn default() -> Self {
		Self {
			range: KeltnerBatchRange::default(),
			kernel: Kernel::Auto,
		}
	}
}
impl KeltnerBatchBuilder {
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
	pub fn multiplier_range(mut self, start: f64, end: f64, step: f64) -> Self {
		self.range.multiplier = (start, end, step);
		self
	}
	#[inline]
	pub fn multiplier_static(mut self, m: f64) -> Self {
		self.range.multiplier = (m, m, 0.0);
		self
	}
	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<KeltnerBatchOutput, KeltnerError> {
		let h = c
			.select_candle_field("high")
			.map_err(|e| KeltnerError::KeltnerMaError(e.to_string()))?;
		let l = c
			.select_candle_field("low")
			.map_err(|e| KeltnerError::KeltnerMaError(e.to_string()))?;
		let cl = c
			.select_candle_field("close")
			.map_err(|e| KeltnerError::KeltnerMaError(e.to_string()))?;
		let src_v = source_type(c, src);
		self.apply_slice(&h, &l, &cl, src_v)
	}
	pub fn apply_slice(
		self,
		high: &[f64],
		low: &[f64],
		close: &[f64],
		source: &[f64],
	) -> Result<KeltnerBatchOutput, KeltnerError> {
		keltner_batch_with_kernel(high, low, close, source, &self.range, self.kernel)
	}
	
	pub fn with_default_slice(high: &[f64], low: &[f64], close: &[f64], source: &[f64], k: Kernel) -> Result<KeltnerBatchOutput, KeltnerError> {
		KeltnerBatchBuilder::new().kernel(k).apply_slice(high, low, close, source)
	}
}

#[derive(Clone, Debug)]
pub struct KeltnerBatchOutput {
	pub upper_band: Vec<f64>,
	pub middle_band: Vec<f64>,
	pub lower_band: Vec<f64>,
	pub combos: Vec<KeltnerParams>,
	pub rows: usize,
	pub cols: usize,
}
impl KeltnerBatchOutput {
	pub fn row_for_params(&self, p: &KeltnerParams) -> Option<usize> {
		self.combos.iter().position(|c| {
			c.period.unwrap_or(20) == p.period.unwrap_or(20)
				&& (c.multiplier.unwrap_or(2.0) - p.multiplier.unwrap_or(2.0)).abs() < 1e-12
		})
	}
	pub fn values_for(&self, p: &KeltnerParams) -> Option<(&[f64], &[f64], &[f64])> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			(
				&self.upper_band[start..start + self.cols],
				&self.middle_band[start..start + self.cols],
				&self.lower_band[start..start + self.cols],
			)
		})
	}
}

fn expand_grid(r: &KeltnerBatchRange) -> Vec<KeltnerParams> {
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
	let mults = axis_f64(r.multiplier);
	let mut out = Vec::with_capacity(periods.len() * mults.len());
	for &p in &periods {
		for &m in &mults {
			out.push(KeltnerParams {
				period: Some(p),
				multiplier: Some(m),
				ma_type: None,
			});
		}
	}
	out
}

pub fn keltner_batch_with_kernel(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	source: &[f64],
	sweep: &KeltnerBatchRange,
	k: Kernel,
) -> Result<KeltnerBatchOutput, KeltnerError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => {
			return Err(KeltnerError::KeltnerInvalidPeriod { period: 0, data_len: 0 });
		}
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	keltner_batch_par_slice(high, low, close, source, sweep, simd)
}

pub fn keltner_batch_slice(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	source: &[f64],
	sweep: &KeltnerBatchRange,
	kern: Kernel,
) -> Result<KeltnerBatchOutput, KeltnerError> {
	keltner_batch_inner(high, low, close, source, sweep, kern, false)
}
pub fn keltner_batch_par_slice(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	source: &[f64],
	sweep: &KeltnerBatchRange,
	kern: Kernel,
) -> Result<KeltnerBatchOutput, KeltnerError> {
	keltner_batch_inner(high, low, close, source, sweep, kern, true)
}
fn keltner_batch_inner(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	source: &[f64],
	sweep: &KeltnerBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<KeltnerBatchOutput, KeltnerError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(KeltnerError::KeltnerInvalidPeriod { period: 0, data_len: 0 });
	}
	let len = close.len();
	let first = close
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(KeltnerError::KeltnerAllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if len - first < max_p {
		return Err(KeltnerError::KeltnerNotEnoughValidData {
			needed: max_p,
			valid: len - first,
		});
	}
	let rows = combos.len();
	let cols = len;
	
	// Calculate warmup periods for each row
	let warmup_periods: Vec<usize> = combos
		.iter()
		.map(|c| c.period.unwrap() - 1)
		.collect();
	
	// Allocate uninitialized matrices and initialize NaN prefixes
	let mut upper_mu = make_uninit_matrix(rows, cols);
	let mut middle_mu = make_uninit_matrix(rows, cols);
	let mut lower_mu = make_uninit_matrix(rows, cols);
	
	init_matrix_prefixes(&mut upper_mu, cols, &warmup_periods);
	init_matrix_prefixes(&mut middle_mu, cols, &warmup_periods);
	init_matrix_prefixes(&mut lower_mu, cols, &warmup_periods);
	
	// Convert to mutable slices
	let mut upper_guard = core::mem::ManuallyDrop::new(upper_mu);
	let mut middle_guard = core::mem::ManuallyDrop::new(middle_mu);
	let mut lower_guard = core::mem::ManuallyDrop::new(lower_mu);
	
	let upper: &mut [f64] = unsafe { core::slice::from_raw_parts_mut(upper_guard.as_mut_ptr() as *mut f64, upper_guard.len()) };
	let middle: &mut [f64] = unsafe { core::slice::from_raw_parts_mut(middle_guard.as_mut_ptr() as *mut f64, middle_guard.len()) };
	let lower: &mut [f64] = unsafe { core::slice::from_raw_parts_mut(lower_guard.as_mut_ptr() as *mut f64, lower_guard.len()) };
	
	let do_row = |row: usize, up: &mut [f64], mid: &mut [f64], low_out: &mut [f64]| unsafe {
		let period = combos[row].period.unwrap();
		let mult = combos[row].multiplier.unwrap();
		keltner_row_scalar(high, low, close, source, period, mult, "ema", first, up, mid, low_out)
	};
	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			upper
				.par_chunks_mut(cols)
				.zip(middle.par_chunks_mut(cols))
				.zip(lower.par_chunks_mut(cols))
				.enumerate()
				.for_each(|(row, ((u, m), l))| do_row(row, u, m, l));
		}

		#[cfg(target_arch = "wasm32")]
		{
			for ((row, u), (m, l)) in upper
				.chunks_mut(cols)
				.enumerate()
				.zip(middle.chunks_mut(cols).zip(lower.chunks_mut(cols)))
			{
				do_row(row, u, m, l);
			}
		}
	} else {
		for ((row, u), (m, l)) in upper
			.chunks_mut(cols)
			.enumerate()
			.zip(middle.chunks_mut(cols).zip(lower.chunks_mut(cols)))
		{
			do_row(row, u, m, l);
		}
	}
	// Convert back to Vec
	let upper = unsafe {
		let ptr = upper_guard.as_mut_ptr() as *mut f64;
		let len = upper_guard.len();
		let cap = upper_guard.capacity();
		core::mem::forget(upper_guard);
		Vec::from_raw_parts(ptr, len, cap)
	};
	
	let middle = unsafe {
		let ptr = middle_guard.as_mut_ptr() as *mut f64;
		let len = middle_guard.len();
		let cap = middle_guard.capacity();
		core::mem::forget(middle_guard);
		Vec::from_raw_parts(ptr, len, cap)
	};
	
	let lower = unsafe {
		let ptr = lower_guard.as_mut_ptr() as *mut f64;
		let len = lower_guard.len();
		let cap = lower_guard.capacity();
		core::mem::forget(lower_guard);
		Vec::from_raw_parts(ptr, len, cap)
	};
	
	Ok(KeltnerBatchOutput {
		upper_band: upper,
		middle_band: middle,
		lower_band: lower,
		combos,
		rows,
		cols,
	})
}

// Streaming mode

#[derive(Debug, Clone)]
pub struct KeltnerStream {
	period: usize,
	multiplier: f64,
	ma_type: String,
	ma_impl: MaImpl,
	atr: f64,
	count: usize,
	prev_close: f64,
}

#[derive(Debug, Clone)]
enum MaImpl {
	Ema {
		alpha: f64,
		value: f64,
	},
	Sma {
		buffer: Vec<f64>,
		sum: f64,
		idx: usize,
		filled: bool,
	},
}

impl KeltnerStream {
	pub fn try_new(params: KeltnerParams) -> Result<Self, KeltnerError> {
		let period = params.period.unwrap_or(20);
		let multiplier = params.multiplier.unwrap_or(2.0);
		let ma_type = params.ma_type.unwrap_or("ema".to_string());
		if period == 0 {
			return Err(KeltnerError::KeltnerInvalidPeriod { period, data_len: 0 });
		}
		let ma_impl = match ma_type.as_str() {
			"sma" => MaImpl::Sma {
				buffer: vec![0.0; period],
				sum: 0.0,
				idx: 0,
				filled: false,
			},
			_ => MaImpl::Ema {
				alpha: 2.0 / (period as f64 + 1.0),
				value: 0.0,
			},
		};
		Ok(Self {
			period,
			multiplier,
			ma_type,
			ma_impl,
			atr: 0.0,
			count: 0,
			prev_close: f64::NAN,
		})
	}
	#[inline(always)]
	pub fn update(&mut self, high: f64, low: f64, close: f64, source: f64) -> Option<(f64, f64, f64)> {
		let tr = if self.count == 0 {
			high - low
		} else {
			let hl = high - low;
			let hc = (high - self.prev_close).abs();
			let lc = (low - self.prev_close).abs();
			hl.max(hc).max(lc)
		};
		self.prev_close = close;
		self.count += 1;

		if self.count < self.period {
			match &mut self.ma_impl {
				MaImpl::Ema { alpha, value } => {
					if self.count == 1 {
						*value = source;
					} else {
						*value += (source - *value) * *alpha;
					}
				}
				MaImpl::Sma {
					buffer,
					sum,
					idx,
					filled,
				} => {
					if *filled {
						*sum -= buffer[*idx];
					}
					buffer[*idx] = source;
					*sum += source;
					*idx = (*idx + 1) % self.period;
					if !*filled && *idx == 0 {
						*filled = true;
					}
				}
			}
			self.atr += tr;
			return None;
		}

		if self.count == self.period {
			self.atr = (self.atr + tr) / self.period as f64;
		} else {
			self.atr += (tr - self.atr) / self.period as f64;
		}

		let ma_val = match &mut self.ma_impl {
			MaImpl::Ema { alpha, value } => {
				if self.count == 1 {
					*value = source;
				} else {
					*value += (source - *value) * *alpha;
				}
				*value
			}
			MaImpl::Sma {
				buffer,
				sum,
				idx,
				filled,
			} => {
				if *filled {
					*sum -= buffer[*idx];
				}
				buffer[*idx] = source;
				*sum += source;
				*idx = (*idx + 1) % self.period;
				if !*filled {
					*filled = *idx == 0;
				}
				if *filled {
					*sum / self.period as f64
				} else {
					f64::NAN
				}
			}
		};

		let upper = ma_val + self.multiplier * self.atr;
		let lower = ma_val - self.multiplier * self.atr;
		Some((upper, ma_val, lower))
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;
	use crate::utilities::enums::Kernel;

	fn check_keltner_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let params = KeltnerParams {
			period: Some(20),
			multiplier: Some(2.0),
			ma_type: Some("ema".to_string()),
		};
		let input = KeltnerInput::from_candles(&candles, "close", params);
		let result = keltner_with_kernel(&input, kernel)?;

		assert_eq!(result.upper_band.len(), candles.close.len());
		assert_eq!(result.middle_band.len(), candles.close.len());
		assert_eq!(result.lower_band.len(), candles.close.len());

		let last_five_index = candles.close.len().saturating_sub(5);
		let expected_upper = [
			61619.504155205745,
			61503.56119134791,
			61387.47897150178,
			61286.61078267451,
			61206.25688331261,
		];
		let expected_middle = [
			59758.339871629956,
			59703.35512195091,
			59640.083205574636,
			59593.884805043715,
			59504.46720456336,
		];
		let expected_lower = [
			57897.17558805417,
			57903.14905255391,
			57892.68743964749,
			57901.158827412924,
			57802.67752581411,
		];
		let last_five_upper = &result.upper_band[last_five_index..];
		let last_five_middle = &result.middle_band[last_five_index..];
		let last_five_lower = &result.lower_band[last_five_index..];
		for i in 0..5 {
			let diff_u = (last_five_upper[i] - expected_upper[i]).abs();
			let diff_m = (last_five_middle[i] - expected_middle[i]).abs();
			let diff_l = (last_five_lower[i] - expected_lower[i]).abs();
			assert!(
				diff_u < 1e-1,
				"Upper band mismatch at index {}: expected {}, got {}",
				i,
				expected_upper[i],
				last_five_upper[i]
			);
			assert!(
				diff_m < 1e-1,
				"Middle band mismatch at index {}: expected {}, got {}",
				i,
				expected_middle[i],
				last_five_middle[i]
			);
			assert!(
				diff_l < 1e-1,
				"Lower band mismatch at index {}: expected {}, got {}",
				i,
				expected_lower[i],
				last_five_lower[i]
			);
		}
		Ok(())
	}

	fn check_keltner_default_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let default_params = KeltnerParams::default();
		let input = KeltnerInput::from_candles(&candles, "close", default_params);
		let result = keltner_with_kernel(&input, kernel)?;
		assert_eq!(result.upper_band.len(), candles.close.len());
		assert_eq!(result.middle_band.len(), candles.close.len());
		assert_eq!(result.lower_band.len(), candles.close.len());
		Ok(())
	}

	fn check_keltner_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = KeltnerParams {
			period: Some(0),
			multiplier: Some(2.0),
			ma_type: Some("ema".to_string()),
		};
		let input = KeltnerInput::from_candles(&candles, "close", params);
		let result = keltner_with_kernel(&input, kernel);
		assert!(result.is_err());
		if let Err(e) = result {
			assert!(e.to_string().contains("invalid period"));
		}
		Ok(())
	}

	fn check_keltner_large_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = KeltnerParams {
			period: Some(999999),
			multiplier: Some(2.0),
			ma_type: Some("ema".to_string()),
		};
		let input = KeltnerInput::from_candles(&candles, "close", params);
		let result = keltner_with_kernel(&input, kernel);
		assert!(result.is_err());
		if let Err(e) = result {
			assert!(e.to_string().contains("invalid period"));
		}
		Ok(())
	}

	fn check_keltner_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = KeltnerParams::default();
		let input = KeltnerInput::from_candles(&candles, "close", params);
		let result = keltner_with_kernel(&input, kernel)?;
		assert_eq!(result.middle_band.len(), candles.close.len());
		if result.middle_band.len() > 240 {
			for (i, &val) in result.middle_band[240..].iter().enumerate() {
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

	fn check_keltner_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let period = 20;
		let multiplier = 2.0;

		let params = KeltnerParams {
			period: Some(period),
			multiplier: Some(multiplier),
			ma_type: Some("ema".to_string()),
		};
		let input = KeltnerInput::from_candles(&candles, "close", params.clone());
		let batch_output = keltner_with_kernel(&input, kernel)?;

		let mut stream = KeltnerStream::try_new(params)?;
		let mut upper_stream = Vec::with_capacity(candles.close.len());
		let mut middle_stream = Vec::with_capacity(candles.close.len());
		let mut lower_stream = Vec::with_capacity(candles.close.len());

		for i in 0..candles.close.len() {
			let hi = candles.high[i];
			let lo = candles.low[i];
			let cl = candles.close[i];
			let src = candles.close[i]; // using "close" as the MA source for streaming
			match stream.update(hi, lo, cl, src) {
				Some((up, mid, low)) => {
					upper_stream.push(up);
					middle_stream.push(mid);
					lower_stream.push(low);
				}
				None => {
					upper_stream.push(f64::NAN);
					middle_stream.push(f64::NAN);
					lower_stream.push(f64::NAN);
				}
			}
		}
		assert_eq!(batch_output.upper_band.len(), upper_stream.len());
		assert_eq!(batch_output.middle_band.len(), middle_stream.len());
		assert_eq!(batch_output.lower_band.len(), lower_stream.len());
		for (i, (&b, &s)) in batch_output.middle_band.iter().zip(middle_stream.iter()).enumerate() {
			if b.is_nan() && s.is_nan() {
				continue;
			}
			let diff = (b - s).abs();
			assert!(
				diff < 1e-8,
				"[{}] Keltner streaming mismatch at idx {}: batch={}, stream={}, diff={}",
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
	fn check_keltner_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		
		// Define comprehensive parameter combinations
		let test_params = vec![
			KeltnerParams::default(),  // period: 20, multiplier: 2.0, ma_type: "ema"
			KeltnerParams { period: Some(2), multiplier: Some(1.0), ma_type: Some("ema".to_string()) },    // minimum period
			KeltnerParams { period: Some(5), multiplier: Some(0.5), ma_type: Some("ema".to_string()) },    // small period, small multiplier
			KeltnerParams { period: Some(10), multiplier: Some(1.5), ma_type: Some("sma".to_string()) },   // medium period with SMA
			KeltnerParams { period: Some(20), multiplier: Some(3.0), ma_type: Some("ema".to_string()) },   // default period, large multiplier
			KeltnerParams { period: Some(50), multiplier: Some(2.5), ma_type: Some("sma".to_string()) },   // large period
			KeltnerParams { period: Some(100), multiplier: Some(1.0), ma_type: Some("ema".to_string()) },  // very large period
			KeltnerParams { period: Some(14), multiplier: Some(2.0), ma_type: Some("sma".to_string()) },   // common period with SMA
			KeltnerParams { period: Some(7), multiplier: Some(1.0), ma_type: Some("ema".to_string()) },    // week period
			KeltnerParams { period: Some(21), multiplier: Some(1.5), ma_type: Some("ema".to_string()) },   // 3-week period
			KeltnerParams { period: Some(30), multiplier: Some(2.0), ma_type: Some("sma".to_string()) },   // month period
			KeltnerParams { period: Some(3), multiplier: Some(0.75), ma_type: Some("ema".to_string()) },   // very small period
		];
		
		for (param_idx, params) in test_params.iter().enumerate() {
			let input = KeltnerInput::from_candles(&candles, "close", params.clone());
			let output = keltner_with_kernel(&input, kernel)?;
			
			// Check all three output bands
			for (band_name, band_values) in [
				("upper", &output.upper_band),
				("middle", &output.middle_band),
				("lower", &output.lower_band),
			] {
				for (i, &val) in band_values.iter().enumerate() {
					if val.is_nan() {
						continue; // NaN values are expected during warmup
					}
					
					let bits = val.to_bits();
					
					// Check all three poison patterns
					if bits == 0x11111111_11111111 {
						panic!(
							"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
							 in {} band with params: period={}, multiplier={}, ma_type={} (param set {})",
							test_name, val, bits, i, band_name,
							params.period.unwrap_or(20),
							params.multiplier.unwrap_or(2.0),
							params.ma_type.as_deref().unwrap_or("ema"),
							param_idx
						);
					}
					
					if bits == 0x22222222_22222222 {
						panic!(
							"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
							 in {} band with params: period={}, multiplier={}, ma_type={} (param set {})",
							test_name, val, bits, i, band_name,
							params.period.unwrap_or(20),
							params.multiplier.unwrap_or(2.0),
							params.ma_type.as_deref().unwrap_or("ema"),
							param_idx
						);
					}
					
					if bits == 0x33333333_33333333 {
						panic!(
							"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
							 in {} band with params: period={}, multiplier={}, ma_type={} (param set {})",
							test_name, val, bits, i, band_name,
							params.period.unwrap_or(20),
							params.multiplier.unwrap_or(2.0),
							params.ma_type.as_deref().unwrap_or("ema"),
							param_idx
						);
					}
				}
			}
		}
		
		Ok(())
	}
	
	#[cfg(not(debug_assertions))]
	fn check_keltner_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(()) // No-op in release builds
	}

	fn check_batch_default_row(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let output = KeltnerBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;

		let def = KeltnerParams::default();
		let (upper, middle, lower) = output.values_for(&def).expect("default row missing");

		assert_eq!(upper.len(), c.close.len());
		assert_eq!(middle.len(), c.close.len());
		assert_eq!(lower.len(), c.close.len());

		Ok(())
	}

	#[cfg(debug_assertions)]
	fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		
		// Test various parameter sweep configurations
		let test_configs = vec![
			// (period_start, period_end, period_step, multiplier_start, multiplier_end, multiplier_step)
			(2, 10, 2, 0.5, 2.5, 0.5),      // Small periods with various multipliers
			(5, 25, 5, 1.0, 3.0, 1.0),      // Medium periods
			(30, 60, 15, 2.0, 2.0, 0.0),    // Large periods, static multiplier
			(2, 5, 1, 1.5, 2.5, 0.25),      // Dense small range
			(10, 30, 10, 0.75, 2.25, 0.75), // Common trading periods
			(14, 21, 7, 1.0, 2.0, 0.5),     // Week-based periods
			(20, 20, 0, 0.5, 3.0, 0.5),     // Static period, varying multipliers
		];
		
		for (cfg_idx, &(p_start, p_end, p_step, m_start, m_end, m_step)) in test_configs.iter().enumerate() {
			let output = KeltnerBatchBuilder::new()
				.kernel(kernel)
				.period_range(p_start, p_end, p_step)
				.multiplier_range(m_start, m_end, m_step)
				.apply_candles(&c, "close")?;
			
			// Check all three output bands
			for (band_name, band_values) in [
				("upper", &output.upper_band),
				("middle", &output.middle_band),
				("lower", &output.lower_band),
			] {
				for (idx, &val) in band_values.iter().enumerate() {
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
							 at row {} col {} (flat index {}) in {} band with params: period={}, multiplier={}",
							test, cfg_idx, val, bits, row, col, idx, band_name,
							combo.period.unwrap_or(20),
							combo.multiplier.unwrap_or(2.0)
						);
					}
					
					if bits == 0x22222222_22222222 {
						panic!(
							"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
							 at row {} col {} (flat index {}) in {} band with params: period={}, multiplier={}",
							test, cfg_idx, val, bits, row, col, idx, band_name,
							combo.period.unwrap_or(20),
							combo.multiplier.unwrap_or(2.0)
						);
					}
					
					if bits == 0x33333333_33333333 {
						panic!(
							"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
							 at row {} col {} (flat index {}) in {} band with params: period={}, multiplier={}",
							test, cfg_idx, val, bits, row, col, idx, band_name,
							combo.period.unwrap_or(20),
							combo.multiplier.unwrap_or(2.0)
						);
					}
				}
			}
		}
		
		Ok(())
	}
	
	#[cfg(not(debug_assertions))]
	fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(())
	}

	macro_rules! generate_all_keltner_tests {
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

	generate_all_keltner_tests!(
		check_keltner_accuracy,
		check_keltner_default_params,
		check_keltner_zero_period,
		check_keltner_large_period,
		check_keltner_nan_handling,
		check_keltner_streaming,
		check_keltner_no_poison
	);

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

// Python bindings

#[cfg(feature = "python")]
#[pyfunction(name = "keltner")]
#[pyo3(signature = (high, low, close, source, period, multiplier, ma_type, kernel=None))]
pub fn keltner_py<'py>(
	py: Python<'py>,
	high: numpy::PyReadonlyArray1<'py, f64>,
	low: numpy::PyReadonlyArray1<'py, f64>,
	close: numpy::PyReadonlyArray1<'py, f64>,
	source: numpy::PyReadonlyArray1<'py, f64>,
	period: usize,
	multiplier: f64,
	ma_type: &str,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;

	let high_slice = high.as_slice()?;
	let low_slice = low.as_slice()?;
	let close_slice = close.as_slice()?;
	let source_slice = source.as_slice()?;
	
	let kern = validate_kernel(kernel, false)?;
	let params = KeltnerParams {
		period: Some(period),
		multiplier: Some(multiplier),
		ma_type: Some(ma_type.to_string()),
	};
	let input = KeltnerInput::from_slice(high_slice, low_slice, close_slice, source_slice, params);

	let result = py
		.allow_threads(|| keltner_with_kernel(&input, kern))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	let dict = PyDict::new(py);
	dict.set_item("upper_band", result.upper_band.into_pyarray(py))?;
	dict.set_item("middle_band", result.middle_band.into_pyarray(py))?;
	dict.set_item("lower_band", result.lower_band.into_pyarray(py))?;
	
	Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "KeltnerStream")]
pub struct KeltnerStreamPy {
	stream: KeltnerStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl KeltnerStreamPy {
	#[new]
	fn new(period: usize, multiplier: f64, ma_type: &str) -> PyResult<Self> {
		let params = KeltnerParams {
			period: Some(period),
			multiplier: Some(multiplier),
			ma_type: Some(ma_type.to_string()),
		};
		let stream = KeltnerStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(KeltnerStreamPy { stream })
	}

	fn update(&mut self, high: f64, low: f64, close: f64, source: f64) -> Option<(f64, f64, f64)> {
		self.stream.update(high, low, close, source)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "keltner_batch")]
#[pyo3(signature = (high, low, close, source, period_range, multiplier_range, ma_type="ema", kernel=None))]
pub fn keltner_batch_py<'py>(
	py: Python<'py>,
	high: numpy::PyReadonlyArray1<'py, f64>,
	low: numpy::PyReadonlyArray1<'py, f64>,
	close: numpy::PyReadonlyArray1<'py, f64>,
	source: numpy::PyReadonlyArray1<'py, f64>,
	period_range: (usize, usize, usize),
	multiplier_range: (f64, f64, f64),
	ma_type: &str,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;

	let high_slice = high.as_slice()?;
	let low_slice = low.as_slice()?;
	let close_slice = close.as_slice()?;
	let source_slice = source.as_slice()?;

	let sweep = KeltnerBatchRange {
		period: period_range,
		multiplier: multiplier_range,
	};

	let kern = validate_kernel(kernel, true)?;

	let output = py
		.allow_threads(|| keltner_batch_with_kernel(high_slice, low_slice, close_slice, source_slice, &sweep, kern))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	let rows = output.rows;
	let cols = output.cols;

	let dict = PyDict::new(py);
	dict.set_item("upper_band", output.upper_band.into_pyarray(py).reshape((rows, cols))?)?;
	dict.set_item("middle_band", output.middle_band.into_pyarray(py).reshape((rows, cols))?)?;
	dict.set_item("lower_band", output.lower_band.into_pyarray(py).reshape((rows, cols))?)?;
	dict.set_item(
		"periods",
		output.combos
			.iter()
			.map(|p| p.period.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;
	dict.set_item(
		"multipliers",
		output.combos
			.iter()
			.map(|p| p.multiplier.unwrap())
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;

	Ok(dict)
}

// WASM bindings

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct KeltnerMultiResult {
	values: Vec<f64>, // [upper..., middle..., lower...]
	rows: usize,      // 3 for keltner (upper, middle, lower)
	cols: usize,      // data length
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl KeltnerMultiResult {
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

/// Safe API - returns flattened array [upper..., middle..., lower...]
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn keltner_js(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	source: &[f64],
	period: usize,
	multiplier: f64,
	ma_type: &str,
) -> Result<KeltnerMultiResult, JsValue> {
	let params = KeltnerParams {
		period: Some(period),
		multiplier: Some(multiplier),
		ma_type: Some(ma_type.to_string()),
	};
	let input = KeltnerInput::from_slice(high, low, close, source, params);
	
	let len = close.len();
	let mut output = vec![0.0; len * 3]; // 3 outputs
	
	// Split into three slices
	let (upper_part, rest) = output.split_at_mut(len);
	let (middle_part, lower_part) = rest.split_at_mut(len);
	
	keltner_into_slice(upper_part, middle_part, lower_part, &input, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	Ok(KeltnerMultiResult {
		values: output,
		rows: 3,
		cols: len,
	})
}

/// Fast API with aliasing detection - separate pointers for each output
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn keltner_into(
	high_ptr: *const f64,
	low_ptr: *const f64,
	close_ptr: *const f64,
	source_ptr: *const f64,
	upper_ptr: *mut f64,
	middle_ptr: *mut f64,
	lower_ptr: *mut f64,
	len: usize,
	period: usize,
	multiplier: f64,
	ma_type: &str,
) -> Result<(), JsValue> {
	if high_ptr.is_null() || low_ptr.is_null() || close_ptr.is_null() || source_ptr.is_null() ||
	   upper_ptr.is_null() || middle_ptr.is_null() || lower_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}
	
	unsafe {
		let high = std::slice::from_raw_parts(high_ptr, len);
		let low = std::slice::from_raw_parts(low_ptr, len);
		let close = std::slice::from_raw_parts(close_ptr, len);
		let source = std::slice::from_raw_parts(source_ptr, len);
		
		let params = KeltnerParams {
			period: Some(period),
			multiplier: Some(multiplier),
			ma_type: Some(ma_type.to_string()),
		};
		let input = KeltnerInput::from_slice(high, low, close, source, params);
		
		// Check for aliasing between input and output pointers
		let input_ptrs = [high_ptr as *const f64, low_ptr as *const f64, 
		                  close_ptr as *const f64, source_ptr as *const f64];
		let output_ptrs = [upper_ptr as *const f64, middle_ptr as *const f64, 
		                   lower_ptr as *const f64];
		
		let has_aliasing = input_ptrs.iter().any(|&in_ptr| {
			output_ptrs.iter().any(|&out_ptr| in_ptr == out_ptr)
		});
		
		if has_aliasing {
			// Handle aliasing by using temporary buffers
			let mut temp_upper = vec![0.0; len];
			let mut temp_middle = vec![0.0; len];
			let mut temp_lower = vec![0.0; len];
			
			keltner_into_slice(&mut temp_upper, &mut temp_middle, &mut temp_lower, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			
			let upper_out = std::slice::from_raw_parts_mut(upper_ptr, len);
			let middle_out = std::slice::from_raw_parts_mut(middle_ptr, len);
			let lower_out = std::slice::from_raw_parts_mut(lower_ptr, len);
			
			upper_out.copy_from_slice(&temp_upper);
			middle_out.copy_from_slice(&temp_middle);
			lower_out.copy_from_slice(&temp_lower);
		} else {
			// No aliasing, write directly
			let upper_out = std::slice::from_raw_parts_mut(upper_ptr, len);
			let middle_out = std::slice::from_raw_parts_mut(middle_ptr, len);
			let lower_out = std::slice::from_raw_parts_mut(lower_ptr, len);
			
			keltner_into_slice(upper_out, middle_out, lower_out, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		
		Ok(())
	}
}

/// Memory allocation for WASM
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn keltner_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

/// Memory deallocation for WASM
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn keltner_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe {
			let _ = Vec::from_raw_parts(ptr, len, len);
		}
	}
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct KeltnerBatchConfig {
	pub period_range: (usize, usize, usize),
	pub multiplier_range: (f64, f64, f64),
	pub ma_type: String,
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct KeltnerBatchJsOutput {
	pub upper_band: Vec<f64>,
	pub middle_band: Vec<f64>,
	pub lower_band: Vec<f64>,
	pub combos: Vec<KeltnerParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = keltner_batch)]
pub fn keltner_batch_js(
	high: &[f64],
	low: &[f64],
	close: &[f64],
	source: &[f64],
	config: JsValue,
) -> Result<JsValue, JsValue> {
	let config: KeltnerBatchConfig = serde_wasm_bindgen::from_value(config)
		.map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
	
	let sweep = KeltnerBatchRange {
		period: config.period_range,
		multiplier: config.multiplier_range,
	};
	
	let output = keltner_batch_with_kernel(high, low, close, source, &sweep, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	let combos = output.combos.clone();
	
	let js_output = KeltnerBatchJsOutput {
		upper_band: output.upper_band,
		middle_band: output.middle_band,
		lower_band: output.lower_band,
		combos,
		rows: output.rows,
		cols: output.cols,
	};
	
	serde_wasm_bindgen::to_value(&js_output)
		.map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}
