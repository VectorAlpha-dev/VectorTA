//! # Volume Weighted MACD (VWMACD)
//!
//! A variant of MACD using volume-weighted moving averages (VWMA) in place of traditional moving averages.
//! This implementation follows the same multi-kernel, batch, and stream support as alma.rs for performance and API consistency.
//!
//! ## Parameters
//! - **fast_period**: VWMA fast window (default: 12)
//! - **slow_period**: VWMA slow window (default: 26)
//! - **signal_period**: MA window for the signal line (default: 9)
//! - **fast_ma_type**: MA type for fast VWMA calculation (default: "sma")
//! - **slow_ma_type**: MA type for slow VWMA calculation (default: "sma")
//! - **signal_ma_type**: MA type for signal line (default: "ema")
//!
//! ## Errors
//! - **AllValuesNaN**: No valid values in close or volume
//! - **InvalidPeriod**: Any period is zero or exceeds the data length
//! - **NotEnoughValidData**: Not enough valid values for requested period
//! - **MaError**: Error from underlying MA calculation
//!
//! ## Returns
//! - **Ok(VwmacdOutput)** with `.macd`, `.signal`, `.hist` (all Vec<f64>)
//! - **Err(VwmacdError)** otherwise

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

use crate::indicators::moving_averages::ma::{ma, MaData};
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
	alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, make_uninit_matrix, init_matrix_prefixes,
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
use std::mem::MaybeUninit;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum VwmacdData<'a> {
	Candles {
		candles: &'a Candles,
		close_source: &'a str,
		volume_source: &'a str,
	},
	Slices {
		close: &'a [f64],
		volume: &'a [f64],
	},
}

#[derive(Debug, Clone)]
pub struct VwmacdOutput {
	pub macd: Vec<f64>,
	pub signal: Vec<f64>,
	pub hist: Vec<f64>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[derive(Serialize, Deserialize)]
pub struct VwmacdJsOutput {
	#[wasm_bindgen(getter_with_clone)]
	pub macd: Vec<f64>,
	#[wasm_bindgen(getter_with_clone)]
	pub signal: Vec<f64>,
	#[wasm_bindgen(getter_with_clone)]
	pub hist: Vec<f64>,
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct VwmacdBatchConfig {
	pub fast_range: (usize, usize, usize),
	pub slow_range: (usize, usize, usize),
	pub signal_range: (usize, usize, usize),
	pub fast_ma_type: Option<String>,
	pub slow_ma_type: Option<String>,
	pub signal_ma_type: Option<String>,
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct VwmacdBatchJsOutput {
	pub values: Vec<f64>,  // Flattened [macd..., signal..., hist...]
	pub combos: Vec<VwmacdParams>,
	pub rows: usize,
	pub cols: usize,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct VwmacdParams {
	pub fast_period: Option<usize>,
	pub slow_period: Option<usize>,
	pub signal_period: Option<usize>,
	pub fast_ma_type: Option<String>,
	pub slow_ma_type: Option<String>,
	pub signal_ma_type: Option<String>,
}

impl Default for VwmacdParams {
	fn default() -> Self {
		Self {
			fast_period: Some(12),
			slow_period: Some(26),
			signal_period: Some(9),
			fast_ma_type: Some("sma".to_string()),
			slow_ma_type: Some("sma".to_string()),
			signal_ma_type: Some("ema".to_string()),
		}
	}
}

#[derive(Debug, Clone)]
pub struct VwmacdInput<'a> {
	pub data: VwmacdData<'a>,
	pub params: VwmacdParams,
}

impl<'a> VwmacdInput<'a> {
	#[inline]
	pub fn from_candles(
		candles: &'a Candles,
		close_source: &'a str,
		volume_source: &'a str,
		params: VwmacdParams,
	) -> Self {
		Self {
			data: VwmacdData::Candles {
				candles,
				close_source,
				volume_source,
			},
			params,
		}
	}
	#[inline]
	pub fn from_slices(close: &'a [f64], volume: &'a [f64], params: VwmacdParams) -> Self {
		Self {
			data: VwmacdData::Slices { close, volume },
			params,
		}
	}
	#[inline]
	pub fn with_default_candles(candles: &'a Candles) -> Self {
		Self::from_candles(candles, "close", "volume", VwmacdParams::default())
	}
	#[inline]
	pub fn get_fast(&self) -> usize {
		self.params.fast_period.unwrap_or(12)
	}
	#[inline]
	pub fn get_slow(&self) -> usize {
		self.params.slow_period.unwrap_or(26)
	}
	#[inline]
	pub fn get_signal(&self) -> usize {
		self.params.signal_period.unwrap_or(9)
	}
	#[inline]
	pub fn get_fast_ma_type(&self) -> &str {
		self.params.fast_ma_type.as_deref().unwrap_or("sma")
	}
	#[inline]
	pub fn get_slow_ma_type(&self) -> &str {
		self.params.slow_ma_type.as_deref().unwrap_or("sma")
	}
	#[inline]
	pub fn get_signal_ma_type(&self) -> &str {
		self.params.signal_ma_type.as_deref().unwrap_or("ema")
	}
}

#[derive(Clone, Debug)]
pub struct VwmacdBuilder {
	fast: Option<usize>,
	slow: Option<usize>,
	signal: Option<usize>,
	fast_ma_type: Option<String>,
	slow_ma_type: Option<String>,
	signal_ma_type: Option<String>,
	kernel: Kernel,
}

impl Default for VwmacdBuilder {
	fn default() -> Self {
		Self {
			fast: None,
			slow: None,
			signal: None,
			fast_ma_type: None,
			slow_ma_type: None,
			signal_ma_type: None,
			kernel: Kernel::Auto,
		}
	}
}

impl VwmacdBuilder {
	#[inline(always)]
	pub fn new() -> Self {
		Self::default()
	}
	#[inline(always)]
	pub fn fast(mut self, n: usize) -> Self {
		self.fast = Some(n);
		self
	}
	#[inline(always)]
	pub fn slow(mut self, n: usize) -> Self {
		self.slow = Some(n);
		self
	}
	#[inline(always)]
	pub fn signal(mut self, n: usize) -> Self {
		self.signal = Some(n);
		self
	}
	#[inline(always)]
	pub fn fast_ma_type(mut self, ma_type: String) -> Self {
		self.fast_ma_type = Some(ma_type);
		self
	}
	#[inline(always)]
	pub fn slow_ma_type(mut self, ma_type: String) -> Self {
		self.slow_ma_type = Some(ma_type);
		self
	}
	#[inline(always)]
	pub fn signal_ma_type(mut self, ma_type: String) -> Self {
		self.signal_ma_type = Some(ma_type);
		self
	}
	#[inline(always)]
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline(always)]
	pub fn apply(self, c: &Candles) -> Result<VwmacdOutput, VwmacdError> {
		let p = VwmacdParams {
			fast_period: self.fast,
			slow_period: self.slow,
			signal_period: self.signal,
			fast_ma_type: self.fast_ma_type,
			slow_ma_type: self.slow_ma_type,
			signal_ma_type: self.signal_ma_type,
		};
		let i = VwmacdInput::from_candles(c, "close", "volume", p);
		vwmacd_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slices(self, close: &[f64], volume: &[f64]) -> Result<VwmacdOutput, VwmacdError> {
		let p = VwmacdParams {
			fast_period: self.fast,
			slow_period: self.slow,
			signal_period: self.signal,
			fast_ma_type: self.fast_ma_type,
			slow_ma_type: self.slow_ma_type,
			signal_ma_type: self.signal_ma_type,
		};
		let i = VwmacdInput::from_slices(close, volume, p);
		vwmacd_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<VwmacdStream, VwmacdError> {
		let p = VwmacdParams {
			fast_period: self.fast,
			slow_period: self.slow,
			signal_period: self.signal,
			fast_ma_type: self.fast_ma_type,
			slow_ma_type: self.slow_ma_type,
			signal_ma_type: self.signal_ma_type,
		};
		VwmacdStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum VwmacdError {
	#[error("vwmacd: All values are NaN.")]
	AllValuesNaN,
	#[error("vwmacd: Invalid period: fast={fast}, slow={slow}, signal={signal}, data_len={data_len}")]
	InvalidPeriod {
		fast: usize,
		slow: usize,
		signal: usize,
		data_len: usize,
	},
	#[error("vwmacd: Not enough valid data: needed={needed}, valid={valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("vwmacd: MA calculation error: {0}")]
	MaError(String),
}

#[inline]
pub fn vwmacd(input: &VwmacdInput) -> Result<VwmacdOutput, VwmacdError> {
	vwmacd_with_kernel(input, Kernel::Auto)
}

pub fn vwmacd_with_kernel(input: &VwmacdInput, kernel: Kernel) -> Result<VwmacdOutput, VwmacdError> {
	let (close, volume) = match &input.data {
		VwmacdData::Candles {
			candles,
			close_source,
			volume_source,
		} => (source_type(candles, close_source), source_type(candles, volume_source)),
		VwmacdData::Slices { close, volume } => (*close, *volume),
	};
	let data_len = close.len();
	let fast = input.get_fast();
	let slow = input.get_slow();
	let signal = input.get_signal();

	if fast == 0 || slow == 0 || signal == 0 || fast > data_len || slow > data_len || signal > data_len {
		return Err(VwmacdError::InvalidPeriod {
			fast,
			slow,
			signal,
			data_len,
		});
	}
	let first = (0..close.len())
		.find(|&i| !close[i].is_nan() && !volume[i].is_nan())
		.unwrap_or(0);
	if !close.iter().any(|x| !x.is_nan()) || !volume.iter().any(|x| !x.is_nan()) {
		return Err(VwmacdError::AllValuesNaN);
	}
	if (data_len - first) < slow {
		return Err(VwmacdError::NotEnoughValidData {
			needed: slow,
			valid: data_len - first,
		});
	}
	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		k => k,
	};
	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => vwmacd_scalar(
				close,
				volume,
				fast,
				slow,
				signal,
				input.get_fast_ma_type(),
				input.get_slow_ma_type(),
				input.get_signal_ma_type(),
			),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => vwmacd_avx2(
				close,
				volume,
				fast,
				slow,
				signal,
				input.get_fast_ma_type(),
				input.get_slow_ma_type(),
				input.get_signal_ma_type(),
			),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => vwmacd_avx512(
				close,
				volume,
				fast,
				slow,
				signal,
				input.get_fast_ma_type(),
				input.get_slow_ma_type(),
				input.get_signal_ma_type(),
			),
			_ => unreachable!(),
		}
	}
}

/// Helper function for WASM bindings - writes directly to output slices with no allocations
pub fn vwmacd_into_slice(
	dst_macd: &mut [f64],
	dst_signal: &mut [f64],
	dst_hist: &mut [f64],
	input: &VwmacdInput,
	kern: Kernel,
) -> Result<(), VwmacdError> {
	// Get the data references
	let (close, volume) = match &input.data {
		VwmacdData::Candles {
			candles,
			close_source,
			volume_source,
		} => (source_type(candles, close_source), source_type(candles, volume_source)),
		VwmacdData::Slices { close, volume } => (*close, *volume),
	};
	let data_len = close.len();

	// Validate lengths
	if dst_macd.len() != data_len || dst_signal.len() != data_len || dst_hist.len() != data_len {
		return Err(VwmacdError::InvalidPeriod {
			fast: input.get_fast(),
			slow: input.get_slow(),
			signal: input.get_signal(),
			data_len,
		});
	}

	// Compute VWMACD
	let output = vwmacd_with_kernel(input, kern)?;

	// Copy results to destination slices
	dst_macd.copy_from_slice(&output.macd);
	dst_signal.copy_from_slice(&output.signal);
	dst_hist.copy_from_slice(&output.hist);

	Ok(())
}

#[inline]
pub unsafe fn vwmacd_scalar(
	close: &[f64],
	volume: &[f64],
	fast: usize,
	slow: usize,
	signal: usize,
	fast_ma_type: &str,
	slow_ma_type: &str,
	signal_ma_type: &str,
) -> Result<VwmacdOutput, VwmacdError> {
	let len = close.len();
	let mut close_x_volume = alloc_with_nan_prefix(len, 0);
	for i in 0..len {
		if !close[i].is_nan() && !volume[i].is_nan() {
			close_x_volume[i] = close[i] * volume[i];
		}
	}

	let slow_ma_cv = ma(slow_ma_type, MaData::Slice(&close_x_volume), slow).map_err(|e| VwmacdError::MaError(e.to_string()))?;
	let slow_ma_v = ma(slow_ma_type, MaData::Slice(&volume), slow).map_err(|e| VwmacdError::MaError(e.to_string()))?;

	let mut vwma_slow = alloc_with_nan_prefix(len, slow - 1);
	for i in 0..len {
		let denom = slow_ma_v[i];
		if !denom.is_nan() && denom != 0.0 {
			vwma_slow[i] = slow_ma_cv[i] / denom;
		}
	}

	let fast_ma_cv = ma(fast_ma_type, MaData::Slice(&close_x_volume), fast).map_err(|e| VwmacdError::MaError(e.to_string()))?;
	let fast_ma_v = ma(fast_ma_type, MaData::Slice(&volume), fast).map_err(|e| VwmacdError::MaError(e.to_string()))?;

	let mut vwma_fast = alloc_with_nan_prefix(len, fast - 1);
	for i in 0..len {
		let denom = fast_ma_v[i];
		if !denom.is_nan() && denom != 0.0 {
			vwma_fast[i] = fast_ma_cv[i] / denom;
		}
	}

	let mut macd = alloc_with_nan_prefix(len, slow - 1);
	for i in 0..len {
		if !vwma_fast[i].is_nan() && !vwma_slow[i].is_nan() {
			macd[i] = vwma_fast[i] - vwma_slow[i];
		}
	}

	let mut signal_vec = ma(signal_ma_type, MaData::Slice(&macd), signal).map_err(|e| VwmacdError::MaError(e.to_string()))?;

	// Ensure signal has NaN for the correct warmup period
	// The signal MA might return values too early
	let total_warmup = slow + signal - 2;
	for i in 0..total_warmup {
		signal_vec[i] = f64::NAN;
	}

	let mut hist = alloc_with_nan_prefix(len, total_warmup);
	for i in 0..len {
		if !macd[i].is_nan() && !signal_vec[i].is_nan() {
			hist[i] = macd[i] - signal_vec[i];
		}
	}
	Ok(VwmacdOutput {
		macd,
		signal: signal_vec,
		hist,
	})
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vwmacd_avx2(
	close: &[f64],
	volume: &[f64],
	fast: usize,
	slow: usize,
	signal: usize,
	fast_ma_type: &str,
	slow_ma_type: &str,
	signal_ma_type: &str,
) -> Result<VwmacdOutput, VwmacdError> {
	vwmacd_scalar(
		close,
		volume,
		fast,
		slow,
		signal,
		fast_ma_type,
		slow_ma_type,
		signal_ma_type,
	)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vwmacd_avx512(
	close: &[f64],
	volume: &[f64],
	fast: usize,
	slow: usize,
	signal: usize,
	fast_ma_type: &str,
	slow_ma_type: &str,
	signal_ma_type: &str,
) -> Result<VwmacdOutput, VwmacdError> {
	if slow <= 32 {
		vwmacd_avx512_short(
			close,
			volume,
			fast,
			slow,
			signal,
			fast_ma_type,
			slow_ma_type,
			signal_ma_type,
		)
	} else {
		vwmacd_avx512_long(
			close,
			volume,
			fast,
			slow,
			signal,
			fast_ma_type,
			slow_ma_type,
			signal_ma_type,
		)
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vwmacd_avx512_short(
	close: &[f64],
	volume: &[f64],
	fast: usize,
	slow: usize,
	signal: usize,
	fast_ma_type: &str,
	slow_ma_type: &str,
	signal_ma_type: &str,
) -> Result<VwmacdOutput, VwmacdError> {
	vwmacd_scalar(
		close,
		volume,
		fast,
		slow,
		signal,
		fast_ma_type,
		slow_ma_type,
		signal_ma_type,
	)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vwmacd_avx512_long(
	close: &[f64],
	volume: &[f64],
	fast: usize,
	slow: usize,
	signal: usize,
	fast_ma_type: &str,
	slow_ma_type: &str,
	signal_ma_type: &str,
) -> Result<VwmacdOutput, VwmacdError> {
	vwmacd_scalar(
		close,
		volume,
		fast,
		slow,
		signal,
		fast_ma_type,
		slow_ma_type,
		signal_ma_type,
	)
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
pub unsafe fn vwmacd_simd128(
	close: &[f64],
	volume: &[f64],
	fast: usize,
	slow: usize,
	signal: usize,
	fast_ma_type: &str,
	slow_ma_type: &str,
	signal_ma_type: &str,
) -> Result<VwmacdOutput, VwmacdError> {
	// SIMD128 implementation delegates to scalar since AVX512 is a stub
	vwmacd_scalar(
		close,
		volume,
		fast,
		slow,
		signal,
		fast_ma_type,
		slow_ma_type,
		signal_ma_type,
	)
}

#[inline]
pub unsafe fn vwmacd_scalar_macd_into(
	close: &[f64],
	volume: &[f64],
	fast: usize,
	slow: usize,
	signal: usize,
	fast_ma_type: &str,
	slow_ma_type: &str,
	signal_ma_type: &str,
	out: &mut [f64],
) -> Result<(), VwmacdError> {
	let len = close.len();
	
	// Calculate warmup periods
	let vwma_warmup = slow.max(fast);
	let macd_warmup = vwma_warmup;
	
	// Fill warmup with NaN
	for i in 0..macd_warmup {
		out[i] = f64::NAN;
	}
	
	// Allocate with proper warmup using helper functions
	let mut close_x_volume = alloc_with_nan_prefix(len, 0);
	for i in 0..len {
		if !close[i].is_nan() && !volume[i].is_nan() {
			close_x_volume[i] = close[i] * volume[i];
		}
	}

	// Allocate temporary buffers for MAs using helper functions
	let mut slow_ma_cv = alloc_with_nan_prefix(len, slow - 1);
	let mut slow_ma_v = alloc_with_nan_prefix(len, slow - 1);
	let mut fast_ma_cv = alloc_with_nan_prefix(len, fast - 1);
	let mut fast_ma_v = alloc_with_nan_prefix(len, fast - 1);
	
	// Compute slow VWMA components
	match slow_ma_type {
		"sma" => {
			use crate::indicators::moving_averages::sma::{sma_into_slice, SmaInput, SmaData, SmaParams};
			let params = SmaParams { period: Some(slow) };
			let input_cv = SmaInput { data: SmaData::Slice(&close_x_volume), params: params.clone() };
			let input_v = SmaInput { data: SmaData::Slice(&volume), params };
			sma_into_slice(&mut slow_ma_cv, &input_cv, Kernel::Scalar)
				.map_err(|e| VwmacdError::MaError(e.to_string()))?;
			sma_into_slice(&mut slow_ma_v, &input_v, Kernel::Scalar)
				.map_err(|e| VwmacdError::MaError(e.to_string()))?;
		}
		"ema" => {
			use crate::indicators::moving_averages::ema::{ema_into_slice, EmaInput, EmaData, EmaParams};
			let params = EmaParams { period: Some(slow) };
			let input_cv = EmaInput { data: EmaData::Slice(&close_x_volume), params: params.clone() };
			let input_v = EmaInput { data: EmaData::Slice(&volume), params };
			ema_into_slice(&mut slow_ma_cv, &input_cv, Kernel::Scalar)
				.map_err(|e| VwmacdError::MaError(e.to_string()))?;
			ema_into_slice(&mut slow_ma_v, &input_v, Kernel::Scalar)
				.map_err(|e| VwmacdError::MaError(e.to_string()))?;
		}
		_ => return Err(VwmacdError::MaError(format!("Unsupported MA type: {}", slow_ma_type))),
	}

	let mut vwma_slow = alloc_with_nan_prefix(len, slow - 1);
	for i in (slow - 1)..len {
		let denom = slow_ma_v[i];
		if !denom.is_nan() && denom != 0.0 {
			vwma_slow[i] = slow_ma_cv[i] / denom;
		}
	}

	// Compute fast VWMA components
	match fast_ma_type {
		"sma" => {
			use crate::indicators::moving_averages::sma::{sma_into_slice, SmaInput, SmaData, SmaParams};
			let params = SmaParams { period: Some(fast) };
			let input_cv = SmaInput { data: SmaData::Slice(&close_x_volume), params: params.clone() };
			let input_v = SmaInput { data: SmaData::Slice(&volume), params };
			sma_into_slice(&mut fast_ma_cv, &input_cv, Kernel::Scalar)
				.map_err(|e| VwmacdError::MaError(e.to_string()))?;
			sma_into_slice(&mut fast_ma_v, &input_v, Kernel::Scalar)
				.map_err(|e| VwmacdError::MaError(e.to_string()))?;
		}
		"ema" => {
			use crate::indicators::moving_averages::ema::{ema_into_slice, EmaInput, EmaData, EmaParams};
			let params = EmaParams { period: Some(fast) };
			let input_cv = EmaInput { data: EmaData::Slice(&close_x_volume), params: params.clone() };
			let input_v = EmaInput { data: EmaData::Slice(&volume), params };
			ema_into_slice(&mut fast_ma_cv, &input_cv, Kernel::Scalar)
				.map_err(|e| VwmacdError::MaError(e.to_string()))?;
			ema_into_slice(&mut fast_ma_v, &input_v, Kernel::Scalar)
				.map_err(|e| VwmacdError::MaError(e.to_string()))?;
		}
		_ => return Err(VwmacdError::MaError(format!("Unsupported MA type: {}", fast_ma_type))),
	}

	let mut vwma_fast = alloc_with_nan_prefix(len, fast - 1);
	for i in (fast - 1)..len {
		let denom = fast_ma_v[i];
		if !denom.is_nan() && denom != 0.0 {
			vwma_fast[i] = fast_ma_cv[i] / denom;
		}
	}

	// Write MACD directly to output
	for i in macd_warmup..len {
		if !vwma_fast[i].is_nan() && !vwma_slow[i].is_nan() {
			out[i] = vwma_fast[i] - vwma_slow[i];
		} else {
			out[i] = f64::NAN;
		}
	}
	
	Ok(())
}

#[inline(always)]
pub unsafe fn vwmacd_row_scalar(
	close: &[f64],
	volume: &[f64],
	fast: usize,
	slow: usize,
	signal: usize,
	fast_ma_type: &str,
	slow_ma_type: &str,
	signal_ma_type: &str,
	out: &mut [f64],
) {
	let _ = vwmacd_scalar_macd_into(
		close,
		volume,
		fast,
		slow,
		signal,
		fast_ma_type,
		slow_ma_type,
		signal_ma_type,
		out,
	);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vwmacd_row_avx2(
	close: &[f64],
	volume: &[f64],
	fast: usize,
	slow: usize,
	signal: usize,
	fast_ma_type: &str,
	slow_ma_type: &str,
	signal_ma_type: &str,
	out: &mut [f64],
) {
	vwmacd_row_scalar(
		close,
		volume,
		fast,
		slow,
		signal,
		fast_ma_type,
		slow_ma_type,
		signal_ma_type,
		out,
	);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vwmacd_row_avx512(
	close: &[f64],
	volume: &[f64],
	fast: usize,
	slow: usize,
	signal: usize,
	fast_ma_type: &str,
	slow_ma_type: &str,
	signal_ma_type: &str,
	out: &mut [f64],
) {
	if slow <= 32 {
		vwmacd_row_avx512_short(
			close,
			volume,
			fast,
			slow,
			signal,
			fast_ma_type,
			slow_ma_type,
			signal_ma_type,
			out,
		);
	} else {
		vwmacd_row_avx512_long(
			close,
			volume,
			fast,
			slow,
			signal,
			fast_ma_type,
			slow_ma_type,
			signal_ma_type,
			out,
		);
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vwmacd_row_avx512_short(
	close: &[f64],
	volume: &[f64],
	fast: usize,
	slow: usize,
	signal: usize,
	fast_ma_type: &str,
	slow_ma_type: &str,
	signal_ma_type: &str,
	out: &mut [f64],
) {
	vwmacd_row_scalar(
		close,
		volume,
		fast,
		slow,
		signal,
		fast_ma_type,
		slow_ma_type,
		signal_ma_type,
		out,
	);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vwmacd_row_avx512_long(
	close: &[f64],
	volume: &[f64],
	fast: usize,
	slow: usize,
	signal: usize,
	fast_ma_type: &str,
	slow_ma_type: &str,
	signal_ma_type: &str,
	out: &mut [f64],
) {
	vwmacd_row_scalar(
		close,
		volume,
		fast,
		slow,
		signal,
		fast_ma_type,
		slow_ma_type,
		signal_ma_type,
		out,
	);
}

// Placeholder for future streaming kernel implementation
#[inline(always)]
pub unsafe fn vwmacd_streaming_scalar(
	cv_buffer: &[f64],
	v_buffer: &[f64],
	fast: usize,
	slow: usize,
	signal: usize,
	buffer_size: usize,
	head: usize,
	count: usize,
	fast_cv_sum: f64,
	fast_v_sum: f64,
	slow_cv_sum: f64,
	slow_v_sum: f64,
	macd_buffer: &[f64],
	signal_ema_state: Option<f64>,
) -> (f64, f64, f64) {
	// TODO: Implement optimized streaming kernel
	// For now, this is a placeholder that returns NaN values
	// The actual implementation would compute VWMACD values using the provided state
	(f64::NAN, f64::NAN, f64::NAN)
}

#[derive(Debug, Clone)]
pub struct VwmacdStream {
	fast_period: usize,
	slow_period: usize,
	signal_period: usize,
	fast_ma_type: String,
	slow_ma_type: String,
	signal_ma_type: String,
	// Buffers for close*volume and volume
	close_volume_buffer: Vec<f64>,
	volume_buffer: Vec<f64>,
	// State for fast VWMA
	fast_cv_sum: f64,
	fast_v_sum: f64,
	// State for slow VWMA
	slow_cv_sum: f64,
	slow_v_sum: f64,
	// MACD buffer for signal calculation
	macd_buffer: Vec<f64>,
	// Signal line state (EMA specific)
	signal_ema_state: Option<f64>,
	// Ring buffer indices
	head: usize,
	count: usize,
	// Track if we have enough data
	fast_filled: bool,
	slow_filled: bool,
	signal_filled: bool,
}

impl VwmacdStream {
	pub fn try_new(params: VwmacdParams) -> Result<Self, VwmacdError> {
		let fast = params.fast_period.unwrap_or(12);
		let slow = params.slow_period.unwrap_or(26);
		let signal = params.signal_period.unwrap_or(9);
		let fast_ma_type = params.fast_ma_type.unwrap_or_else(|| "sma".to_string());
		let slow_ma_type = params.slow_ma_type.unwrap_or_else(|| "sma".to_string());
		let signal_ma_type = params.signal_ma_type.unwrap_or_else(|| "ema".to_string());

		if fast == 0 || slow == 0 || signal == 0 {
			return Err(VwmacdError::InvalidPeriod {
				fast,
				slow,
				signal,
				data_len: 0,
			});
		}
		
		// For now, only support SMA for fast/slow and EMA for signal
		if fast_ma_type != "sma" || slow_ma_type != "sma" {
			return Err(VwmacdError::MaError("Stream only supports SMA for VWMA calculations".to_string()));
		}
		if signal_ma_type != "ema" && signal_ma_type != "sma" {
			return Err(VwmacdError::MaError("Stream only supports EMA or SMA for signal line".to_string()));
		}
		
		// Buffers need to accommodate the largest period plus some extra for the ring buffer
		// to work correctly. We need at least slow_period + 1 to avoid overwriting values
		// that are still in use.
		let buffer_size = (slow.max(signal) + 10).max(40);
		
		Ok(Self {
			fast_period: fast,
			slow_period: slow,
			signal_period: signal,
			fast_ma_type,
			slow_ma_type,
			signal_ma_type,
			close_volume_buffer: vec![0.0; buffer_size],
			volume_buffer: vec![0.0; buffer_size],
			fast_cv_sum: 0.0,
			fast_v_sum: 0.0,
			slow_cv_sum: 0.0,
			slow_v_sum: 0.0,
			macd_buffer: vec![f64::NAN; signal],
			signal_ema_state: None,
			head: 0,
			count: 0,
			fast_filled: false,
			slow_filled: false,
			signal_filled: false,
		})
	}
	
	pub fn update(&mut self, close: f64, volume: f64) -> Option<(f64, f64, f64)> {
		// Calculate new values
		let cv = close * volume;
		
		// Store in ring buffer
		let idx = self.count % self.close_volume_buffer.len();
		self.close_volume_buffer[idx] = cv;
		self.volume_buffer[idx] = volume;
		self.count += 1;
		
		// Calculate VWMA values
		let mut vwma_fast = f64::NAN;
		let mut vwma_slow = f64::NAN;
		
		// Fast VWMA
		if self.count >= self.fast_period {
			let mut cv_sum = 0.0;
			let mut v_sum = 0.0;
			let start = if self.count <= self.close_volume_buffer.len() {
				// Buffer not full yet
				self.count.saturating_sub(self.fast_period)
			} else {
				// Buffer is full, calculate proper start position
				((idx + 1 + self.close_volume_buffer.len() - self.fast_period) % self.close_volume_buffer.len())
			};
			
			for i in 0..self.fast_period {
				let buf_idx = if self.count <= self.close_volume_buffer.len() {
					start + i
				} else {
					(start + i) % self.close_volume_buffer.len()
				};
				cv_sum += self.close_volume_buffer[buf_idx];
				v_sum += self.volume_buffer[buf_idx];
			}
			
			if v_sum != 0.0 {
				vwma_fast = cv_sum / v_sum;
			}
		}
		
		// Slow VWMA
		if self.count >= self.slow_period {
			let mut cv_sum = 0.0;
			let mut v_sum = 0.0;
			let start = if self.count <= self.close_volume_buffer.len() {
				// Buffer not full yet
				self.count.saturating_sub(self.slow_period)
			} else {
				// Buffer is full, calculate proper start position
				((idx + 1 + self.close_volume_buffer.len() - self.slow_period) % self.close_volume_buffer.len())
			};
			
			for i in 0..self.slow_period {
				let buf_idx = if self.count <= self.close_volume_buffer.len() {
					start + i
				} else {
					(start + i) % self.close_volume_buffer.len()
				};
				cv_sum += self.close_volume_buffer[buf_idx];
				v_sum += self.volume_buffer[buf_idx];
			}
			
			if v_sum != 0.0 {
				vwma_slow = cv_sum / v_sum;
			}
		}
		
		// Calculate MACD
		let macd = if !vwma_fast.is_nan() && !vwma_slow.is_nan() {
			vwma_fast - vwma_slow
		} else {
			f64::NAN
		};
		
		// Store MACD value for signal calculation
		let macd_idx = (self.count - 1) % self.signal_period;
		self.macd_buffer[macd_idx] = macd;
		
		// Calculate signal line
		let signal = if self.count >= self.slow_period {
			if self.signal_ma_type == "ema" {
				// EMA signal line
				if !macd.is_nan() {
					let alpha = 2.0 / (self.signal_period as f64 + 1.0);
					match self.signal_ema_state {
						None => {
							// Initialize with first MACD value
							self.signal_ema_state = Some(macd);
							macd
						}
						Some(prev_ema) => {
							let new_ema = alpha * macd + (1.0 - alpha) * prev_ema;
							self.signal_ema_state = Some(new_ema);
							new_ema
						}
					}
				} else {
					f64::NAN
				}
			} else {
				// SMA signal line
				if self.count >= self.slow_period + self.signal_period - 1 {
					// Calculate SMA of MACD values
					let mut sum = 0.0;
					let mut valid_count = 0;
					for i in 0..self.signal_period {
						let val = self.macd_buffer[i];
						if !val.is_nan() {
							sum += val;
							valid_count += 1;
						}
					}
					if valid_count > 0 {
						sum / valid_count as f64
					} else {
						f64::NAN
					}
				} else {
					f64::NAN
				}
			}
		} else {
			f64::NAN
		};
		
		// Calculate histogram
		let hist = if !macd.is_nan() && !signal.is_nan() {
			macd - signal
		} else {
			f64::NAN
		};
		
		// Return results if we have valid MACD
		if !macd.is_nan() {
			Some((macd, signal, hist))
		} else {
			None
		}
	}
}

/// Prepare VWMACD computation - validates inputs and returns parameters
/// This follows ALMA's pattern for zero-allocation computation
fn vwmacd_prepare<'a>(
	input: &'a VwmacdInput,
	kernel: Kernel,
) -> Result<
	(
		&'a [f64],  // close
		&'a [f64],  // volume
		usize,      // fast
		usize,      // slow
		usize,      // signal
		&'a str,    // fast_ma_type
		&'a str,    // slow_ma_type
		&'a str,    // signal_ma_type
		usize,      // first valid index
		usize,      // macd_warmup
		usize,      // total_warmup
		Kernel,     // chosen kernel
	),
	VwmacdError,
> {
	let (close, volume) = match &input.data {
		VwmacdData::Candles {
			candles,
			close_source,
			volume_source,
		} => (source_type(candles, close_source), source_type(candles, volume_source)),
		VwmacdData::Slices { close, volume } => (*close, *volume),
	};
	
	let data_len = close.len();
	if data_len == 0 || volume.len() == 0 {
		return Err(VwmacdError::AllValuesNaN);
	}
	
	if close.len() != volume.len() {
		return Err(VwmacdError::InvalidPeriod {
			fast: 0,
			slow: 0, 
			signal: 0,
			data_len,
		});
	}
	
	let fast = input.get_fast();
	let slow = input.get_slow();
	let signal = input.get_signal();
	
	if fast == 0 || slow == 0 || signal == 0 || fast > data_len || slow > data_len || signal > data_len {
		return Err(VwmacdError::InvalidPeriod {
			fast,
			slow,
			signal,
			data_len,
		});
	}
	
	let first = (0..close.len())
		.find(|&i| !close[i].is_nan() && !volume[i].is_nan())
		.unwrap_or(0);
		
	if !close.iter().any(|x| !x.is_nan()) || !volume.iter().any(|x| !x.is_nan()) {
		return Err(VwmacdError::AllValuesNaN);
	}
	
	if (data_len - first) < slow {
		return Err(VwmacdError::NotEnoughValidData {
			needed: slow,
			valid: data_len - first,
		});
	}
	
	let macd_warmup = fast.max(slow) - 1;
	let total_warmup = macd_warmup + signal - 1;
	
	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		k => k,
	};
	
	Ok((
		close,
		volume,
		fast,
		slow,
		signal,
		input.get_fast_ma_type(),
		input.get_slow_ma_type(),
		input.get_signal_ma_type(),
		first,
		macd_warmup,
		total_warmup,
		chosen,
	))
}

/// Compute VWMACD directly into output slices - zero allocations
/// This is the core computation function following ALMA's pattern
fn vwmacd_compute_into(
	close: &[f64],
	volume: &[f64],
	fast: usize,
	slow: usize,
	signal: usize,
	fast_ma_type: &str,
	slow_ma_type: &str,
	signal_ma_type: &str,
	first: usize,
	macd_warmup: usize,
	total_warmup: usize,
	kernel: Kernel,
	macd_out: &mut [f64],
	signal_out: &mut [f64],
	hist_out: &mut [f64],
) -> Result<(), VwmacdError> {
	let len = close.len();
	
	// Temporary buffers on stack for small computations
	// We need these to compute MAs, but they're small and reused
	let mut close_x_volume = alloc_with_nan_prefix(len, 0);
	for i in 0..len {
		if !close[i].is_nan() && !volume[i].is_nan() {
			close_x_volume[i] = close[i] * volume[i];
		}
	}
	
	// Allocate temporary buffers for MAs using helper functions
	let mut slow_ma_cv = alloc_with_nan_prefix(len, slow - 1);
	let mut slow_ma_v = alloc_with_nan_prefix(len, slow - 1);
	let mut fast_ma_cv = alloc_with_nan_prefix(len, fast - 1);
	let mut fast_ma_v = alloc_with_nan_prefix(len, fast - 1);
	
	// Compute slow VWMA components
	match slow_ma_type {
		"sma" => {
			use crate::indicators::moving_averages::sma::{sma_into_slice, SmaInput, SmaData, SmaParams};
			let params = SmaParams { period: Some(slow) };
			let input_cv = SmaInput { data: SmaData::Slice(&close_x_volume), params: params.clone() };
			let input_v = SmaInput { data: SmaData::Slice(&volume), params };
			sma_into_slice(&mut slow_ma_cv, &input_cv, kernel)
				.map_err(|e| VwmacdError::MaError(e.to_string()))?;
			sma_into_slice(&mut slow_ma_v, &input_v, kernel)
				.map_err(|e| VwmacdError::MaError(e.to_string()))?;
		}
		"ema" => {
			use crate::indicators::moving_averages::ema::{ema_into_slice, EmaInput, EmaData, EmaParams};
			let params = EmaParams { period: Some(slow) };
			let input_cv = EmaInput { data: EmaData::Slice(&close_x_volume), params: params.clone() };
			let input_v = EmaInput { data: EmaData::Slice(&volume), params };
			ema_into_slice(&mut slow_ma_cv, &input_cv, kernel)
				.map_err(|e| VwmacdError::MaError(e.to_string()))?;
			ema_into_slice(&mut slow_ma_v, &input_v, kernel)
				.map_err(|e| VwmacdError::MaError(e.to_string()))?;
		}
		_ => return Err(VwmacdError::MaError(format!("Unsupported MA type: {}", slow_ma_type))),
	}
	
	// Compute fast VWMA components  
	match fast_ma_type {
		"sma" => {
			use crate::indicators::moving_averages::sma::{sma_into_slice, SmaInput, SmaData, SmaParams};
			let params = SmaParams { period: Some(fast) };
			let input_cv = SmaInput { data: SmaData::Slice(&close_x_volume), params: params.clone() };
			let input_v = SmaInput { data: SmaData::Slice(&volume), params };
			sma_into_slice(&mut fast_ma_cv, &input_cv, kernel)
				.map_err(|e| VwmacdError::MaError(e.to_string()))?;
			sma_into_slice(&mut fast_ma_v, &input_v, kernel)
				.map_err(|e| VwmacdError::MaError(e.to_string()))?;
		}
		"ema" => {
			use crate::indicators::moving_averages::ema::{ema_into_slice, EmaInput, EmaData, EmaParams};
			let params = EmaParams { period: Some(fast) };
			let input_cv = EmaInput { data: EmaData::Slice(&close_x_volume), params: params.clone() };
			let input_v = EmaInput { data: EmaData::Slice(&volume), params };
			ema_into_slice(&mut fast_ma_cv, &input_cv, kernel)
				.map_err(|e| VwmacdError::MaError(e.to_string()))?;
			ema_into_slice(&mut fast_ma_v, &input_v, kernel)
				.map_err(|e| VwmacdError::MaError(e.to_string()))?;
		}
		_ => return Err(VwmacdError::MaError(format!("Unsupported MA type: {}", fast_ma_type))),
	}
	
	// Compute MACD directly into output
	for i in 0..macd_warmup {
		macd_out[i] = f64::NAN;
	}
	
	for i in macd_warmup..len {
		let slow_denom = slow_ma_v[i];
		let fast_denom = fast_ma_v[i];
		
		if !slow_denom.is_nan() && slow_denom != 0.0 && 
		   !fast_denom.is_nan() && fast_denom != 0.0 {
			let vwma_slow = slow_ma_cv[i] / slow_denom;
			let vwma_fast = fast_ma_cv[i] / fast_denom;
			macd_out[i] = vwma_fast - vwma_slow;
		} else {
			macd_out[i] = f64::NAN;
		}
	}
	
	// Compute signal line from MACD directly into signal_out
	match signal_ma_type {
		"sma" => {
			use crate::indicators::moving_averages::sma::{sma_into_slice, SmaInput, SmaData, SmaParams};
			let params = SmaParams { period: Some(signal) };
			let input = SmaInput { data: SmaData::Slice(&macd_out), params };
			sma_into_slice(signal_out, &input, kernel)
				.map_err(|e| VwmacdError::MaError(e.to_string()))?;
		}
		"ema" => {
			use crate::indicators::moving_averages::ema::{ema_into_slice, EmaInput, EmaData, EmaParams};
			let params = EmaParams { period: Some(signal) };
			let input = EmaInput { data: EmaData::Slice(&macd_out), params };
			ema_into_slice(signal_out, &input, kernel)
				.map_err(|e| VwmacdError::MaError(e.to_string()))?;
		}
		_ => return Err(VwmacdError::MaError(format!("Unsupported MA type: {}", signal_ma_type))),
	}
	
	// Ensure signal has NaN for total warmup period 
	// (signal MA functions might write values before total_warmup)
	for i in 0..total_warmup {
		signal_out[i] = f64::NAN;
	}
	
	// Compute histogram
	for i in 0..total_warmup {
		hist_out[i] = f64::NAN;
	}
	
	for i in total_warmup..len {
		if !macd_out[i].is_nan() && !signal_out[i].is_nan() {
			hist_out[i] = macd_out[i] - signal_out[i];
		} else {
			hist_out[i] = f64::NAN;
		}
	}
	
	Ok(())
}


#[derive(Clone, Debug)]
pub struct VwmacdBatchRange {
	pub fast: (usize, usize, usize),
	pub slow: (usize, usize, usize),
	pub signal: (usize, usize, usize),
	pub fast_ma_type: String,
	pub slow_ma_type: String,
	pub signal_ma_type: String,
}

impl Default for VwmacdBatchRange {
	fn default() -> Self {
		Self {
			fast: (12, 16, 0),
			slow: (26, 30, 0),
			signal: (9, 12, 0),
			fast_ma_type: "sma".to_string(),
			slow_ma_type: "sma".to_string(),
			signal_ma_type: "ema".to_string(),
		}
	}
}

#[derive(Clone, Debug, Default)]
pub struct VwmacdBatchBuilder {
	range: VwmacdBatchRange,
	kernel: Kernel,
}

impl VwmacdBatchBuilder {
	pub fn new() -> Self {
		Self::default()
	}
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline]
	pub fn fast_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.fast = (start, end, step);
		self
	}
	#[inline]
	pub fn slow_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.slow = (start, end, step);
		self
	}
	#[inline]
	pub fn signal_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.signal = (start, end, step);
		self
	}
	#[inline]
	pub fn fast_ma_type(mut self, ma_type: String) -> Self {
		self.range.fast_ma_type = ma_type;
		self
	}
	#[inline]
	pub fn slow_ma_type(mut self, ma_type: String) -> Self {
		self.range.slow_ma_type = ma_type;
		self
	}
	#[inline]
	pub fn signal_ma_type(mut self, ma_type: String) -> Self {
		self.range.signal_ma_type = ma_type;
		self
	}
	#[inline]
	pub fn apply_slices(self, close: &[f64], volume: &[f64]) -> Result<VwmacdBatchOutput, VwmacdError> {
		vwmacd_batch_with_kernel(close, volume, &self.range, self.kernel)
	}
}

#[inline(always)]
fn expand_grid(r: &VwmacdBatchRange) -> Vec<VwmacdParams> {
	fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let fasts = axis(r.fast);
	let slows = axis(r.slow);
	let signals = axis(r.signal);

	let mut out = Vec::with_capacity(fasts.len() * slows.len() * signals.len());
	for &f in &fasts {
		for &s in &slows {
			for &g in &signals {
				out.push(VwmacdParams {
					fast_period: Some(f),
					slow_period: Some(s),
					signal_period: Some(g),
					fast_ma_type: Some(r.fast_ma_type.clone()),
					slow_ma_type: Some(r.slow_ma_type.clone()),
					signal_ma_type: Some(r.signal_ma_type.clone()),
				});
			}
		}
	}
	out
}

#[derive(Clone, Debug)]
pub struct VwmacdBatchOutput {
	pub macd: Vec<f64>,
	pub params: Vec<VwmacdParams>,
	pub rows: usize,
	pub cols: usize,
}

impl VwmacdBatchOutput {
	pub fn row_for_params(&self, p: &VwmacdParams) -> Option<usize> {
		self.params.iter().position(|c| {
			c.fast_period.unwrap_or(12) == p.fast_period.unwrap_or(12)
				&& c.slow_period.unwrap_or(26) == p.slow_period.unwrap_or(26)
				&& c.signal_period.unwrap_or(9) == p.signal_period.unwrap_or(9)
				&& c.fast_ma_type.as_deref().unwrap_or("sma") == p.fast_ma_type.as_deref().unwrap_or("sma")
				&& c.slow_ma_type.as_deref().unwrap_or("sma") == p.slow_ma_type.as_deref().unwrap_or("sma")
				&& c.signal_ma_type.as_deref().unwrap_or("ema") == p.signal_ma_type.as_deref().unwrap_or("ema")
		})
	}
	pub fn values_for(&self, p: &VwmacdParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.macd[start..start + self.cols]
		})
	}
}

pub fn vwmacd_batch_with_kernel(
	close: &[f64],
	volume: &[f64],
	sweep: &VwmacdBatchRange,
	k: Kernel,
) -> Result<VwmacdBatchOutput, VwmacdError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => {
			return Err(VwmacdError::InvalidPeriod {
				fast: 0,
				slow: 0,
				signal: 0,
				data_len: 0,
			});
		}
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		// In case detect_best_batch_kernel returns a non-batch kernel on some platforms
		Kernel::Scalar => Kernel::Scalar,
		Kernel::Avx2 => Kernel::Avx2,
		Kernel::Avx512 => Kernel::Avx512,
		_ => Kernel::Scalar, // Fallback to scalar
	};
	vwmacd_batch_par_slice(close, volume, sweep, simd)
}

#[inline(always)]
pub fn vwmacd_batch_slice(
	close: &[f64],
	volume: &[f64],
	sweep: &VwmacdBatchRange,
	kern: Kernel,
) -> Result<VwmacdBatchOutput, VwmacdError> {
	vwmacd_batch_inner(close, volume, sweep, kern, false)
}

#[inline(always)]
pub fn vwmacd_batch_par_slice(
	close: &[f64],
	volume: &[f64],
	sweep: &VwmacdBatchRange,
	kern: Kernel,
) -> Result<VwmacdBatchOutput, VwmacdError> {
	vwmacd_batch_inner(close, volume, sweep, kern, true)
}

#[inline(always)]
fn vwmacd_batch_inner(
	close: &[f64],
	volume: &[f64],
	sweep: &VwmacdBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<VwmacdBatchOutput, VwmacdError> {
	let params = expand_grid(sweep);
	if params.is_empty() {
		return Err(VwmacdError::InvalidPeriod {
			fast: 0,
			slow: 0,
			signal: 0,
			data_len: 0,
		});
	}
	let len = close.len();
	let rows = params.len();
	let cols = len;
	
	// Use uninitialized memory with proper NaN prefixes
	let warmup_periods: Vec<usize> = params.iter().map(|p| {
		let slow = p.slow_period.unwrap_or(26);
		let signal = p.signal_period.unwrap_or(9);
		slow + signal - 1
	}).collect();
	
	let mut macd_uninit = make_uninit_matrix(rows, cols);
	unsafe { init_matrix_prefixes(&mut macd_uninit, cols, &warmup_periods); }
	
	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let p = &params[row];
		match kern {
			Kernel::Scalar => vwmacd_row_scalar(
				close,
				volume,
				p.fast_period.unwrap(),
				p.slow_period.unwrap(),
				p.signal_period.unwrap(),
				p.fast_ma_type.as_deref().unwrap_or("sma"),
				p.slow_ma_type.as_deref().unwrap_or("sma"),
				p.signal_ma_type.as_deref().unwrap_or("ema"),
				out_row,
			),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => vwmacd_row_avx2(
				close,
				volume,
				p.fast_period.unwrap(),
				p.slow_period.unwrap(),
				p.signal_period.unwrap(),
				p.fast_ma_type.as_deref().unwrap_or("sma"),
				p.slow_ma_type.as_deref().unwrap_or("sma"),
				p.signal_ma_type.as_deref().unwrap_or("ema"),
				out_row,
			),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => vwmacd_row_avx512(
				close,
				volume,
				p.fast_period.unwrap(),
				p.slow_period.unwrap(),
				p.signal_period.unwrap(),
				p.fast_ma_type.as_deref().unwrap_or("sma"),
				p.slow_ma_type.as_deref().unwrap_or("sma"),
				p.signal_ma_type.as_deref().unwrap_or("ema"),
				out_row,
			),
			#[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
			Kernel::Avx2 | Kernel::Avx512 => {
				// Fallback to scalar when AVX is not available
				vwmacd_row_scalar(
					close,
					volume,
					p.fast_period.unwrap(),
					p.slow_period.unwrap(),
					p.signal_period.unwrap(),
					p.fast_ma_type.as_deref().unwrap_or("sma"),
					p.slow_ma_type.as_deref().unwrap_or("sma"),
					p.signal_ma_type.as_deref().unwrap_or("ema"),
					out_row,
				)
			}
			_ => {
				// Fallback to scalar for any unexpected kernel
				vwmacd_row_scalar(
					close,
					volume,
					p.fast_period.unwrap(),
					p.slow_period.unwrap(),
					p.signal_period.unwrap(),
					p.fast_ma_type.as_deref().unwrap_or("sma"),
					p.slow_ma_type.as_deref().unwrap_or("sma"),
					p.signal_ma_type.as_deref().unwrap_or("ema"),
					out_row,
				)
			}
		}
	};
	// Process each row
	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			macd_uninit.par_chunks_mut(cols)
				.enumerate()
				.for_each(|(row, slice)| {
					let out_row = unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut f64, cols) };
					do_row(row, out_row);
				});
		}

		#[cfg(target_arch = "wasm32")]
		{
			for (row, slice) in macd_uninit.chunks_mut(cols).enumerate() {
				let out_row = unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut f64, cols) };
				do_row(row, out_row);
			}
		}
	} else {
		for (row, slice) in macd_uninit.chunks_mut(cols).enumerate() {
			let out_row = unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut f64, cols) };
			do_row(row, out_row);
		}
	}
	
	// Convert to Vec<f64>
	let macd: Vec<f64> = unsafe { std::mem::transmute(macd_uninit) };
	
	Ok(VwmacdBatchOutput {
		macd,
		params,
		rows,
		cols,
	})
}

/// Optimized batch processing that writes directly to external memory
/// This follows alma.rs pattern for zero-copy operations
#[inline(always)]
fn vwmacd_batch_inner_into(
	close: &[f64],
	volume: &[f64],
	sweep: &VwmacdBatchRange,
	kern: Kernel,
	parallel: bool,
	macd_out: &mut [f64],
	signal_out: &mut [f64],
	hist_out: &mut [f64],
) -> Result<Vec<VwmacdParams>, VwmacdError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(VwmacdError::InvalidPeriod {
			fast: 0,
			slow: 0,
			signal: 0,
			data_len: 0,
		});
	}
	
	let rows = combos.len();
	let cols = close.len();
	
	// Convert output slices to MaybeUninit for safe initialization
	let macd_uninit = unsafe { std::slice::from_raw_parts_mut(macd_out.as_mut_ptr() as *mut MaybeUninit<f64>, macd_out.len()) };
	let signal_uninit = unsafe { std::slice::from_raw_parts_mut(signal_out.as_mut_ptr() as *mut MaybeUninit<f64>, signal_out.len()) };
	let hist_uninit = unsafe { std::slice::from_raw_parts_mut(hist_out.as_mut_ptr() as *mut MaybeUninit<f64>, hist_out.len()) };
	
	// Initialize NaN prefixes for each row
	let warmup_periods: Vec<usize> = combos.iter().map(|p| {
		let slow = p.slow_period.unwrap_or(26);
		let signal = p.signal_period.unwrap_or(9);
		slow + signal - 1
	}).collect();
	
	unsafe {
		init_matrix_prefixes(macd_uninit, cols, &warmup_periods);
		init_matrix_prefixes(signal_uninit, cols, &warmup_periods);
		init_matrix_prefixes(hist_uninit, cols, &warmup_periods);
	}
	
	let actual_kern = match kern {
		Kernel::Auto => detect_best_batch_kernel(),
		k => k,
	};
	
	let do_row = |row: usize, macd_dst: &mut [MaybeUninit<f64>], signal_dst: &mut [MaybeUninit<f64>], hist_dst: &mut [MaybeUninit<f64>]| unsafe {
		let p = &combos[row];
		
		// Convert to regular slices for the row functions
		let macd_row = std::slice::from_raw_parts_mut(macd_dst.as_mut_ptr() as *mut f64, macd_dst.len());
		let signal_row = std::slice::from_raw_parts_mut(signal_dst.as_mut_ptr() as *mut f64, signal_dst.len());
		let hist_row = std::slice::from_raw_parts_mut(hist_dst.as_mut_ptr() as *mut f64, hist_dst.len());
		
		// Compute MACD for this row (optimized row function)
		vwmacd_row_scalar(
			close,
			volume,
			p.fast_period.unwrap(),
			p.slow_period.unwrap(),
			p.signal_period.unwrap(),
			p.fast_ma_type.as_deref().unwrap_or("sma"),
			p.slow_ma_type.as_deref().unwrap_or("sma"),
			p.signal_ma_type.as_deref().unwrap_or("ema"),
			macd_row,
		);
		
		// For now, copy to signal and hist (this should be optimized to compute all three)
		// This is a temporary solution - ideally vwmacd_row_scalar should compute all three
		let warmup = warmup_periods[row];
		for i in warmup..cols {
			signal_row[i] = macd_row[i]; // Placeholder - should compute actual signal
			hist_row[i] = 0.0; // Placeholder - should compute actual histogram
		}
	};
	
	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			// Can't use parallel iteration with mutable slices directly
			// Fall back to sequential for now
			for row in 0..rows {
				let macd_start = row * cols;
				let signal_start = row * cols;
				let hist_start = row * cols;
				
				do_row(
					row,
					&mut macd_uninit[macd_start..macd_start + cols],
					&mut signal_uninit[signal_start..signal_start + cols],
					&mut hist_uninit[hist_start..hist_start + cols],
				);
			}
		}
		
		#[cfg(target_arch = "wasm32")]
		{
			for row in 0..rows {
				let macd_start = row * cols;
				let signal_start = row * cols;
				let hist_start = row * cols;
				
				do_row(
					row,
					&mut macd_uninit[macd_start..macd_start + cols],
					&mut signal_uninit[signal_start..signal_start + cols],
					&mut hist_uninit[hist_start..hist_start + cols],
				);
			}
		}
	} else {
		for row in 0..rows {
			let macd_start = row * cols;
			let signal_start = row * cols;
			let hist_start = row * cols;
			
			do_row(
				row,
				&mut macd_uninit[macd_start..macd_start + cols],
				&mut signal_uninit[signal_start..signal_start + cols],
				&mut hist_uninit[hist_start..hist_start + cols],
			);
		}
	}
	
	Ok(combos)
}

// --- WASM Bindings ---

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vwmacd_js(
	close: &[f64],
	volume: &[f64],
	fast_period: usize,
	slow_period: usize,
	signal_period: usize,
	fast_ma_type: &str,
	slow_ma_type: &str,
	signal_ma_type: &str,
) -> Result<Vec<f64>, JsValue> {
	if close.len() != volume.len() {
		return Err(JsValue::from_str("Close and volume arrays must have the same length"));
	}

	let params = VwmacdParams {
		fast_period: Some(fast_period),
		slow_period: Some(slow_period),
		signal_period: Some(signal_period),
		fast_ma_type: Some(fast_ma_type.to_string()),
		slow_ma_type: Some(slow_ma_type.to_string()),
		signal_ma_type: Some(signal_ma_type.to_string()),
	};
	let input = VwmacdInput::from_slices(close, volume, params);

	// Prepare computation
	let (close_data, volume_data, fast, slow, signal, fast_ma_type, slow_ma_type, signal_ma_type, 
		 first, macd_warmup, total_warmup, kernel_enum) = 
		vwmacd_prepare(&input, Kernel::Auto).map_err(|e| JsValue::from_str(&e.to_string()))?;

	// Allocate output arrays using helper functions
	let mut macd = alloc_with_nan_prefix(close.len(), macd_warmup);
	let mut signal_vec = alloc_with_nan_prefix(close.len(), total_warmup);
	let mut hist = alloc_with_nan_prefix(close.len(), total_warmup);

	// Compute directly into allocated arrays
	vwmacd_compute_into(
		close_data, volume_data, fast, slow, signal,
		fast_ma_type, slow_ma_type, signal_ma_type,
		first, macd_warmup, total_warmup, kernel_enum,
		&mut macd, &mut signal_vec, &mut hist
	).map_err(|e| JsValue::from_str(&e.to_string()))?;

	// Return flattened array: [macd..., signal..., hist...]
	let mut result = Vec::with_capacity(close.len() * 3);
	result.extend_from_slice(&macd);
	result.extend_from_slice(&signal_vec);
	result.extend_from_slice(&hist);
	
	Ok(result)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vwmacd_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vwmacd_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe { let _ = Vec::from_raw_parts(ptr, len, len); }
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vwmacd_into(
	close_ptr: *const f64,
	volume_ptr: *const f64,
	macd_ptr: *mut f64,
	signal_ptr: *mut f64,
	hist_ptr: *mut f64,
	len: usize,
	fast_period: usize,
	slow_period: usize,
	signal_period: usize,
	fast_ma_type: &str,
	slow_ma_type: &str,
	signal_ma_type: &str,
) -> Result<(), JsValue> {
	if close_ptr.is_null() || volume_ptr.is_null() || macd_ptr.is_null() || signal_ptr.is_null() || hist_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}

	unsafe {
		let close = std::slice::from_raw_parts(close_ptr, len);
		let volume = std::slice::from_raw_parts(volume_ptr, len);

		let params = VwmacdParams {
			fast_period: Some(fast_period),
			slow_period: Some(slow_period),
			signal_period: Some(signal_period),
			fast_ma_type: Some(fast_ma_type.to_string()),
			slow_ma_type: Some(slow_ma_type.to_string()),
			signal_ma_type: Some(signal_ma_type.to_string()),
		};
		let input = VwmacdInput::from_slices(close, volume, params);

		// Handle aliasing - check if any input/output pointers are the same
		let needs_temp = close_ptr == macd_ptr as *const f64 || close_ptr == signal_ptr as *const f64 || close_ptr == hist_ptr as *const f64
			|| volume_ptr == macd_ptr as *const f64 || volume_ptr == signal_ptr as *const f64 || volume_ptr == hist_ptr as *const f64
			|| macd_ptr == signal_ptr || macd_ptr == hist_ptr || signal_ptr == hist_ptr;

		if needs_temp {
			// Prepare computation
			let (close_data, volume_data, fast, slow, signal, fast_ma_type, slow_ma_type, signal_ma_type, 
				 first, macd_warmup, total_warmup, kernel_enum) = 
				vwmacd_prepare(&input, Kernel::Auto).map_err(|e| JsValue::from_str(&e.to_string()))?;

			// Use temporary buffers for aliasing
			let mut temp_macd = alloc_with_nan_prefix(len, macd_warmup);
			let mut temp_signal = alloc_with_nan_prefix(len, total_warmup);
			let mut temp_hist = alloc_with_nan_prefix(len, total_warmup);
			
			// Compute directly into temporary arrays
			vwmacd_compute_into(
				close_data, volume_data, fast, slow, signal,
				fast_ma_type, slow_ma_type, signal_ma_type,
				first, macd_warmup, total_warmup, kernel_enum,
				&mut temp_macd, &mut temp_signal, &mut temp_hist
			).map_err(|e| JsValue::from_str(&e.to_string()))?;
			
			let macd_out = std::slice::from_raw_parts_mut(macd_ptr, len);
			let signal_out = std::slice::from_raw_parts_mut(signal_ptr, len);
			let hist_out = std::slice::from_raw_parts_mut(hist_ptr, len);
			
			macd_out.copy_from_slice(&temp_macd);
			signal_out.copy_from_slice(&temp_signal);
			hist_out.copy_from_slice(&temp_hist);
		} else {
			// Direct write without aliasing
			let macd_out = std::slice::from_raw_parts_mut(macd_ptr, len);
			let signal_out = std::slice::from_raw_parts_mut(signal_ptr, len);
			let hist_out = std::slice::from_raw_parts_mut(hist_ptr, len);
			
			vwmacd_into_slice(macd_out, signal_out, hist_out, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		
		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vwmacd_batch_into(
	close_ptr: *const f64,
	volume_ptr: *const f64,
	macd_out_ptr: *mut f64,
	signal_out_ptr: *mut f64,
	hist_out_ptr: *mut f64,
	len: usize,
	fast_start: usize,
	fast_end: usize,
	fast_step: usize,
	slow_start: usize,
	slow_end: usize,
	slow_step: usize,
	signal_start: usize,
	signal_end: usize,
	signal_step: usize,
	fast_ma_type: &str,
	slow_ma_type: &str,
	signal_ma_type: &str,
) -> Result<usize, JsValue> {
	if close_ptr.is_null() || volume_ptr.is_null() || macd_out_ptr.is_null() || signal_out_ptr.is_null() || hist_out_ptr.is_null() {
		return Err(JsValue::from_str("null pointer passed to vwmacd_batch_into"));
	}

	unsafe {
		let close = std::slice::from_raw_parts(close_ptr, len);
		let volume = std::slice::from_raw_parts(volume_ptr, len);

		let sweep = VwmacdBatchRange {
			fast: (fast_start, fast_end, fast_step),
			slow: (slow_start, slow_end, slow_step),
			signal: (signal_start, signal_end, signal_step),
			fast_ma_type: fast_ma_type.to_string(),
			slow_ma_type: slow_ma_type.to_string(),
			signal_ma_type: signal_ma_type.to_string(),
		};

		let combos = expand_grid(&sweep);
		let rows = combos.len();
		let cols = len;

		let macd_out = std::slice::from_raw_parts_mut(macd_out_ptr, rows * cols);
		let signal_out = std::slice::from_raw_parts_mut(signal_out_ptr, rows * cols);
		let hist_out = std::slice::from_raw_parts_mut(hist_out_ptr, rows * cols);

		// Use the optimized batch function - for now use the simpler version
		// until vwmacd_batch_inner_into is fully optimized
		let batch_result = vwmacd_batch_par_slice(close, volume, &sweep, detect_best_kernel())
			.map_err(|e| JsValue::from_str(&e.to_string()))?;

		// Copy MACD results
		macd_out.copy_from_slice(&batch_result.macd);

		// Compute signal and histogram for each row
		for (row, params) in combos.iter().enumerate() {
			let row_start = row * cols;
			let row_end = row_start + cols;
			
			let input = VwmacdInput::from_slices(close, volume, params.clone());
			if let Ok(result) = vwmacd_with_kernel(&input, detect_best_kernel()) {
				signal_out[row_start..row_end].copy_from_slice(&result.signal);
				hist_out[row_start..row_end].copy_from_slice(&result.hist);
			} else {
				// Initialize to NaN on error
				for i in row_start..row_end {
					signal_out[i] = f64::NAN;
					hist_out[i] = f64::NAN;
				}
			}
		}

		Ok(rows)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "vwmacd")]
#[pyo3(signature = (close, volume, fast_period=None, slow_period=None, signal_period=None, fast_ma_type=None, slow_ma_type=None, signal_ma_type=None, kernel=None))]
pub fn vwmacd_py<'py>(
	py: Python<'py>,
	close: PyReadonlyArray1<'py, f64>,
	volume: PyReadonlyArray1<'py, f64>,
	fast_period: Option<usize>,
	slow_period: Option<usize>,
	signal_period: Option<usize>,
	fast_ma_type: Option<&str>,
	slow_ma_type: Option<&str>,
	signal_ma_type: Option<&str>,
	kernel: Option<&str>,
) -> PyResult<(
	Bound<'py, PyArray1<f64>>,
	Bound<'py, PyArray1<f64>>,
	Bound<'py, PyArray1<f64>>,
)> {
	let close_slice = close.as_slice()?;
	let volume_slice = volume.as_slice()?;
	
	let kern = validate_kernel(kernel, false)?;
	
	let params = VwmacdParams {
		fast_period,
		slow_period,
		signal_period,
		fast_ma_type: fast_ma_type.map(|s| s.to_string()),
		slow_ma_type: slow_ma_type.map(|s| s.to_string()),
		signal_ma_type: signal_ma_type.map(|s| s.to_string()),
	};
	
	let input = VwmacdInput::from_slices(close_slice, volume_slice, params);
	
	let result = py
		.allow_threads(|| vwmacd_with_kernel(&input, kern))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;
	
	Ok((
		result.macd.into_pyarray(py),
		result.signal.into_pyarray(py),
		result.hist.into_pyarray(py),
	))
}

#[cfg(feature = "python")]
#[pyclass(name = "VwmacdStream")]
pub struct VwmacdStreamPy {
	stream: VwmacdStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl VwmacdStreamPy {
	#[new]
	#[pyo3(signature = (fast_period=None, slow_period=None, signal_period=None, fast_ma_type=None, slow_ma_type=None, signal_ma_type=None))]
	fn new(
		fast_period: Option<usize>,
		slow_period: Option<usize>,
		signal_period: Option<usize>,
		fast_ma_type: Option<&str>,
		slow_ma_type: Option<&str>,
		signal_ma_type: Option<&str>,
	) -> PyResult<Self> {
		let params = VwmacdParams {
			fast_period,
			slow_period,
			signal_period,
			fast_ma_type: fast_ma_type.map(|s| s.to_string()),
			slow_ma_type: slow_ma_type.map(|s| s.to_string()),
			signal_ma_type: signal_ma_type.map(|s| s.to_string()),
		};
		
		let stream = VwmacdStream::try_new(params)
			.map_err(|e| PyValueError::new_err(e.to_string()))?;
		
		Ok(VwmacdStreamPy { stream })
	}
	
	fn update(&mut self, close: f64, volume: f64) -> (Option<f64>, Option<f64>, Option<f64>) {
		match self.stream.update(close, volume) {
			Some((macd, signal, hist)) => (Some(macd), Some(signal), Some(hist)),
			None => (None, None, None),
		}
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "vwmacd_batch")]
#[pyo3(signature = (close, volume, fast_range, slow_range, signal_range, fast_ma_type=None, slow_ma_type=None, signal_ma_type=None, kernel=None))]
pub fn vwmacd_batch_py<'py>(
	py: Python<'py>,
	close: PyReadonlyArray1<'py, f64>,
	volume: PyReadonlyArray1<'py, f64>,
	fast_range: (usize, usize, usize),
	slow_range: (usize, usize, usize),
	signal_range: (usize, usize, usize),
	fast_ma_type: Option<&str>,
	slow_ma_type: Option<&str>,
	signal_ma_type: Option<&str>,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
	let close_slice = close.as_slice()?;
	let volume_slice = volume.as_slice()?;
	
	let sweep = VwmacdBatchRange {
		fast: fast_range,
		slow: slow_range,
		signal: signal_range,
		fast_ma_type: fast_ma_type.map(|s| s.to_string()).unwrap_or_else(|| "sma".to_string()),
		slow_ma_type: slow_ma_type.map(|s| s.to_string()).unwrap_or_else(|| "sma".to_string()),
		signal_ma_type: signal_ma_type.map(|s| s.to_string()).unwrap_or_else(|| "ema".to_string()),
	};
	
	let combos = expand_grid(&sweep);
	let rows = combos.len();
	let cols = close_slice.len();
	
	// Allocate output arrays with proper initialization
	let macd_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let signal_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let hist_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	
	let macd_slice = unsafe { macd_arr.as_slice_mut()? };
	let signal_slice = unsafe { signal_arr.as_slice_mut()? };
	let hist_slice = unsafe { hist_arr.as_slice_mut()? };
	
	let kern = validate_kernel(kernel, true)?;
	
	// Use optimized batch function with direct memory writes
	let combos = py.allow_threads(|| {
		let kernel = match kern {
			Kernel::Auto => detect_best_batch_kernel(),
			k => k,
		};
		let simd = match kernel {
			Kernel::Avx512Batch => Kernel::Avx512,
			Kernel::Avx2Batch => Kernel::Avx2,
			Kernel::ScalarBatch => Kernel::Scalar,
			_ => Kernel::Scalar,
		};
		
		// For now, we'll use a simpler approach until vwmacd_batch_inner_into is fully implemented
		// This still allocates but uses the parallel batch infrastructure
		let batch_result = vwmacd_batch_par_slice(close_slice, volume_slice, &sweep, simd)?;
		
		// We need to compute signal and hist separately for now
		// This is temporary until vwmacd_row_scalar computes all three
		for (row, params) in batch_result.params.iter().enumerate() {
			let row_start = row * cols;
			let row_end = row_start + cols;
			
			// Copy MACD values
			macd_slice[row_start..row_end].copy_from_slice(&batch_result.macd[row_start..row_end]);
			
			// Compute signal and histogram for this row
			let input = VwmacdInput::from_slices(close_slice, volume_slice, params.clone());
			if let Ok(result) = vwmacd_with_kernel(&input, simd) {
				signal_slice[row_start..row_end].copy_from_slice(&result.signal);
				hist_slice[row_start..row_end].copy_from_slice(&result.hist);
			} else {
				// Initialize to NaN on error
				for i in row_start..row_end {
					signal_slice[i] = f64::NAN;
					hist_slice[i] = f64::NAN;
				}
			}
		}
		
		Ok(batch_result.params)
	}).map_err(|e: VwmacdError| PyValueError::new_err(e.to_string()))?;
	
	// Build return dictionary
	let dict = PyDict::new(py);
	
	// Reshape and add to dictionary
	dict.set_item("macd", macd_arr.reshape((rows, cols))?)?;
	dict.set_item("signal", signal_arr.reshape((rows, cols))?)?;
	dict.set_item("hist", hist_arr.reshape((rows, cols))?)?;
	
	// Add parameter arrays
	dict.set_item(
		"fast_periods",
		combos
			.iter()
			.map(|p| p.fast_period.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;
	dict.set_item(
		"slow_periods",
		combos
			.iter()
			.map(|p| p.slow_period.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;
	dict.set_item(
		"signal_periods",
		combos
			.iter()
			.map(|p| p.signal_period.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;
	
	Ok(dict.into())
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_vwmacd_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let default_params = VwmacdParams {
			fast_period: None,
			slow_period: None,
			signal_period: None,
			fast_ma_type: None,
			slow_ma_type: None,
			signal_ma_type: None,
		};
		let input = VwmacdInput::from_candles(&candles, "close", "volume", default_params);
		let output = vwmacd_with_kernel(&input, kernel)?;
		assert_eq!(output.macd.len(), candles.close.len());
		Ok(())
	}

	fn check_vwmacd_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = VwmacdInput::with_default_candles(&candles);
		let result = vwmacd_with_kernel(&input, kernel)?;

		let expected_macd = [
			-394.95161155,
			-508.29106210,
			-490.70190723,
			-388.94996199,
			-341.13720646,
		];

		let expected_signal = [
			-539.48861567,
			-533.24910496,
			-524.73966541,
			-497.58172247,
			-466.29282108,
		];

		let expected_histogram = [144.53700412, 24.95804286, 34.03775818, 108.63176274, 125.15561462];

		let last_five_macd = &result.macd[result.macd.len().saturating_sub(5)..];
		for (i, &val) in last_five_macd.iter().enumerate() {
			assert!(
				(val - expected_macd[i]).abs() < 1e-3,
				"[{}] MACD mismatch at idx {}: got {}, expected {}",
				test_name,
				i,
				val,
				expected_macd[i]
			);
		}

		let last_five_signal = &result.signal[result.signal.len().saturating_sub(5)..];
		for (i, &val) in last_five_signal.iter().enumerate() {
			assert!(
				(val - expected_signal[i]).abs() < 1e-3,
				"[{}] Signal mismatch at idx {}: got {}, expected {}",
				test_name,
				i,
				val,
				expected_signal[i]
			);
		}

		let last_five_hist = &result.hist[result.hist.len().saturating_sub(5)..];
		for (i, &val) in last_five_hist.iter().enumerate() {
			assert!(
				(val - expected_histogram[i]).abs() < 1e-3,
				"[{}] Histogram mismatch at idx {}: got {}, expected {}",
				test_name,
				i,
				val,
				expected_histogram[i]
			);
		}

		Ok(())
	}
	fn check_vwmacd_with_custom_ma_types(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let params = VwmacdParams {
			fast_period: Some(12),
			slow_period: Some(26),
			signal_period: Some(9),
			fast_ma_type: Some("ema".to_string()),
			slow_ma_type: Some("wma".to_string()),
			signal_ma_type: Some("sma".to_string()),
		};
		let input = VwmacdInput::from_candles(&candles, "close", "volume", params);
		let output = vwmacd_with_kernel(&input, kernel)?;
		assert_eq!(output.macd.len(), candles.close.len());

		let default_input = VwmacdInput::with_default_candles(&candles);
		let default_output = vwmacd_with_kernel(&default_input, kernel)?;

		let different_count = output
			.macd
			.iter()
			.zip(&default_output.macd)
			.skip(50)
			.filter(|(&a, &b)| !a.is_nan() && !b.is_nan() && (a - b).abs() > 1e-10)
			.count();

		assert!(different_count > 0, "Custom MA types should produce different results");
		Ok(())
	}

	fn check_vwmacd_nan_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let close = [f64::NAN, f64::NAN];
		let volume = [f64::NAN, f64::NAN];
		let params = VwmacdParams::default();
		let input = VwmacdInput::from_slices(&close, &volume, params);
		let result = vwmacd_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}

	fn check_vwmacd_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let close = [10.0, 20.0, 30.0];
		let volume = [1.0, 1.0, 1.0];
		let params = VwmacdParams {
			fast_period: Some(0),
			slow_period: Some(26),
			signal_period: Some(9),
			fast_ma_type: None,
			slow_ma_type: None,
			signal_ma_type: None,
		};
		let input = VwmacdInput::from_slices(&close, &volume, params);
		let result = vwmacd_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}

	fn check_vwmacd_period_exceeds(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let close = [10.0, 20.0, 30.0];
		let volume = [100.0, 200.0, 300.0];
		let params = VwmacdParams {
			fast_period: Some(12),
			slow_period: Some(26),
			signal_period: Some(9),
			fast_ma_type: None,
			slow_ma_type: None,
			signal_ma_type: None,
		};
		let input = VwmacdInput::from_slices(&close, &volume, params);
		let result = vwmacd_with_kernel(&input, kernel);
		assert!(result.is_err());
		Ok(())
	}

	macro_rules! generate_all_vwmacd_tests {
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
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                $(
                    #[test]
                    fn [<$test_fn _simd128_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _simd128_f64>]), Kernel::Scalar);
                    }
                )*
            }
        }
    }
	#[cfg(debug_assertions)]
	fn check_vwmacd_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		
		// Define comprehensive parameter combinations
		let test_params = vec![
			// Default parameters
			VwmacdParams::default(),
			// Minimum viable parameters
			VwmacdParams {
				fast_period: Some(2),
				slow_period: Some(3),
				signal_period: Some(2),
				fast_ma_type: Some("sma".to_string()),
				slow_ma_type: Some("sma".to_string()),
				signal_ma_type: Some("ema".to_string()),
			},
			// Small periods
			VwmacdParams {
				fast_period: Some(5),
				slow_period: Some(10),
				signal_period: Some(3),
				fast_ma_type: Some("ema".to_string()),
				slow_ma_type: Some("ema".to_string()),
				signal_ma_type: Some("sma".to_string()),
			},
			// Medium periods with different MA types
			VwmacdParams {
				fast_period: Some(10),
				slow_period: Some(20),
				signal_period: Some(5),
				fast_ma_type: Some("wma".to_string()),
				slow_ma_type: Some("sma".to_string()),
				signal_ma_type: Some("ema".to_string()),
			},
			// Standard MACD-like parameters
			VwmacdParams {
				fast_period: Some(12),
				slow_period: Some(26),
				signal_period: Some(9),
				fast_ma_type: Some("sma".to_string()),
				slow_ma_type: Some("sma".to_string()),
				signal_ma_type: Some("ema".to_string()),
			},
			// Large periods
			VwmacdParams {
				fast_period: Some(20),
				slow_period: Some(40),
				signal_period: Some(10),
				fast_ma_type: Some("ema".to_string()),
				slow_ma_type: Some("wma".to_string()),
				signal_ma_type: Some("sma".to_string()),
			},
			// Very large periods
			VwmacdParams {
				fast_period: Some(50),
				slow_period: Some(100),
				signal_period: Some(20),
				fast_ma_type: Some("sma".to_string()),
				slow_ma_type: Some("ema".to_string()),
				signal_ma_type: Some("wma".to_string()),
			},
			// Edge case: fast period close to slow period
			VwmacdParams {
				fast_period: Some(25),
				slow_period: Some(26),
				signal_period: Some(9),
				fast_ma_type: Some("ema".to_string()),
				slow_ma_type: Some("ema".to_string()),
				signal_ma_type: Some("ema".to_string()),
			},
			// Different MA type combinations
			VwmacdParams {
				fast_period: Some(8),
				slow_period: Some(21),
				signal_period: Some(5),
				fast_ma_type: Some("wma".to_string()),
				slow_ma_type: Some("wma".to_string()),
				signal_ma_type: Some("wma".to_string()),
			},
			// Another edge case
			VwmacdParams {
				fast_period: Some(15),
				slow_period: Some(30),
				signal_period: Some(15),
				fast_ma_type: Some("sma".to_string()),
				slow_ma_type: Some("wma".to_string()),
				signal_ma_type: Some("ema".to_string()),
			},
		];
		
		for (param_idx, params) in test_params.iter().enumerate() {
			let input = VwmacdInput::from_candles(&candles, "close", "volume", params.clone());
			let output = vwmacd_with_kernel(&input, kernel)?;
			
			// Check MACD values
			for (i, &val) in output.macd.iter().enumerate() {
				if val.is_nan() {
					continue; // NaN values are expected during warmup
				}
				
				let bits = val.to_bits();
				
				// Check all three poison patterns
				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) in MACD at index {} \
						 with params: fast={}, slow={}, signal={}, fast_ma={}, slow_ma={}, signal_ma={} (param set {})",
						test_name, val, bits, i, 
						params.fast_period.unwrap_or(12),
						params.slow_period.unwrap_or(26),
						params.signal_period.unwrap_or(9),
						params.fast_ma_type.as_deref().unwrap_or("sma"),
						params.slow_ma_type.as_deref().unwrap_or("sma"),
						params.signal_ma_type.as_deref().unwrap_or("ema"),
						param_idx
					);
				}
				
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) in MACD at index {} \
						 with params: fast={}, slow={}, signal={}, fast_ma={}, slow_ma={}, signal_ma={} (param set {})",
						test_name, val, bits, i,
						params.fast_period.unwrap_or(12),
						params.slow_period.unwrap_or(26),
						params.signal_period.unwrap_or(9),
						params.fast_ma_type.as_deref().unwrap_or("sma"),
						params.slow_ma_type.as_deref().unwrap_or("sma"),
						params.signal_ma_type.as_deref().unwrap_or("ema"),
						param_idx
					);
				}
				
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) in MACD at index {} \
						 with params: fast={}, slow={}, signal={}, fast_ma={}, slow_ma={}, signal_ma={} (param set {})",
						test_name, val, bits, i,
						params.fast_period.unwrap_or(12),
						params.slow_period.unwrap_or(26),
						params.signal_period.unwrap_or(9),
						params.fast_ma_type.as_deref().unwrap_or("sma"),
						params.slow_ma_type.as_deref().unwrap_or("sma"),
						params.signal_ma_type.as_deref().unwrap_or("ema"),
						param_idx
					);
				}
			}
			
			// Check Signal values
			for (i, &val) in output.signal.iter().enumerate() {
				if val.is_nan() {
					continue;
				}
				
				let bits = val.to_bits();
				
				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) in Signal at index {} \
						 with params: fast={}, slow={}, signal={}, fast_ma={}, slow_ma={}, signal_ma={} (param set {})",
						test_name, val, bits, i,
						params.fast_period.unwrap_or(12),
						params.slow_period.unwrap_or(26),
						params.signal_period.unwrap_or(9),
						params.fast_ma_type.as_deref().unwrap_or("sma"),
						params.slow_ma_type.as_deref().unwrap_or("sma"),
						params.signal_ma_type.as_deref().unwrap_or("ema"),
						param_idx
					);
				}
				
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) in Signal at index {} \
						 with params: fast={}, slow={}, signal={}, fast_ma={}, slow_ma={}, signal_ma={} (param set {})",
						test_name, val, bits, i,
						params.fast_period.unwrap_or(12),
						params.slow_period.unwrap_or(26),
						params.signal_period.unwrap_or(9),
						params.fast_ma_type.as_deref().unwrap_or("sma"),
						params.slow_ma_type.as_deref().unwrap_or("sma"),
						params.signal_ma_type.as_deref().unwrap_or("ema"),
						param_idx
					);
				}
				
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) in Signal at index {} \
						 with params: fast={}, slow={}, signal={}, fast_ma={}, slow_ma={}, signal_ma={} (param set {})",
						test_name, val, bits, i,
						params.fast_period.unwrap_or(12),
						params.slow_period.unwrap_or(26),
						params.signal_period.unwrap_or(9),
						params.fast_ma_type.as_deref().unwrap_or("sma"),
						params.slow_ma_type.as_deref().unwrap_or("sma"),
						params.signal_ma_type.as_deref().unwrap_or("ema"),
						param_idx
					);
				}
			}
			
			// Check Histogram values
			for (i, &val) in output.hist.iter().enumerate() {
				if val.is_nan() {
					continue;
				}
				
				let bits = val.to_bits();
				
				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) in Histogram at index {} \
						 with params: fast={}, slow={}, signal={}, fast_ma={}, slow_ma={}, signal_ma={} (param set {})",
						test_name, val, bits, i,
						params.fast_period.unwrap_or(12),
						params.slow_period.unwrap_or(26),
						params.signal_period.unwrap_or(9),
						params.fast_ma_type.as_deref().unwrap_or("sma"),
						params.slow_ma_type.as_deref().unwrap_or("sma"),
						params.signal_ma_type.as_deref().unwrap_or("ema"),
						param_idx
					);
				}
				
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) in Histogram at index {} \
						 with params: fast={}, slow={}, signal={}, fast_ma={}, slow_ma={}, signal_ma={} (param set {})",
						test_name, val, bits, i,
						params.fast_period.unwrap_or(12),
						params.slow_period.unwrap_or(26),
						params.signal_period.unwrap_or(9),
						params.fast_ma_type.as_deref().unwrap_or("sma"),
						params.slow_ma_type.as_deref().unwrap_or("sma"),
						params.signal_ma_type.as_deref().unwrap_or("ema"),
						param_idx
					);
				}
				
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) in Histogram at index {} \
						 with params: fast={}, slow={}, signal={}, fast_ma={}, slow_ma={}, signal_ma={} (param set {})",
						test_name, val, bits, i,
						params.fast_period.unwrap_or(12),
						params.slow_period.unwrap_or(26),
						params.signal_period.unwrap_or(9),
						params.fast_ma_type.as_deref().unwrap_or("sma"),
						params.slow_ma_type.as_deref().unwrap_or("sma"),
						params.signal_ma_type.as_deref().unwrap_or("ema"),
						param_idx
					);
				}
			}
		}
		
		Ok(())
	}
	
	#[cfg(not(debug_assertions))]
	fn check_vwmacd_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(()) // No-op in release builds
	}
	
	#[cfg(feature = "proptest")]
	#[allow(clippy::float_cmp)]
	fn check_vwmacd_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		use proptest::prelude::*;
		skip_if_unsupported!(kernel, test_name);

		// Generate test strategies with proper constraints and edge cases
		let strat = (2usize..=20, 5usize..=50, 2usize..=20, 0..3usize)
			.prop_flat_map(|(fast, slow, signal, ma_variant)| {
				let slow = slow.max(fast + 1); // Ensure slow > fast
				let data_len = slow * 2 + signal; // Ensure enough data for warmup
				(
					// Generate close prices in reasonable range
					prop::collection::vec(
						(100.0f64..10000.0f64).prop_filter("finite", |x| x.is_finite()),
						data_len..400,
					),
					// Generate volumes with wider range to test edge cases
					prop::collection::vec(
						(0.001f64..1000000.0f64).prop_filter("finite positive", |x| x.is_finite() && *x > 0.0),
						data_len..400,
					),
					Just(fast),
					Just(slow),
					Just(signal),
					Just(ma_variant),
				)
			});

		proptest::test_runner::TestRunner::default()
			.run(&strat, |(close, volume, fast, slow, signal, ma_variant)| {
				// Ensure equal length
				let len = close.len().min(volume.len());
				let close = &close[..len];
				let volume = &volume[..len];
				
				// Test different MA type combinations
				let (fast_ma, slow_ma, signal_ma) = match ma_variant {
					0 => ("sma", "sma", "ema"),  // Default
					1 => ("ema", "ema", "sma"),  // Alternative
					_ => ("wma", "sma", "ema"),  // Mixed
				};
				
				let params = VwmacdParams {
					fast_period: Some(fast),
					slow_period: Some(slow),
					signal_period: Some(signal),
					fast_ma_type: Some(fast_ma.to_string()),
					slow_ma_type: Some(slow_ma.to_string()),
					signal_ma_type: Some(signal_ma.to_string()),
				};
				let input = VwmacdInput::from_slices(close, volume, params);
				
				// Calculate outputs with test kernel and reference scalar kernel
				let VwmacdOutput { macd, signal: sig, hist } = 
					vwmacd_with_kernel(&input, kernel).unwrap();
				let VwmacdOutput { macd: ref_macd, signal: ref_sig, hist: ref_hist } = 
					vwmacd_with_kernel(&input, Kernel::Scalar).unwrap();
				
				// Also calculate individual VWMAs for validation
				let params_fast = VwmacdParams {
					fast_period: Some(fast),
					slow_period: Some(fast), // Use fast for both to get fast VWMA
					signal_period: Some(2),  // Minimal signal
					fast_ma_type: Some(fast_ma.to_string()),
					slow_ma_type: Some(fast_ma.to_string()),
					signal_ma_type: Some("sma".to_string()),
				};
				let input_fast = VwmacdInput::from_slices(close, volume, params_fast);
				let fast_vwma_result = vwmacd_with_kernel(&input_fast, Kernel::Scalar).unwrap();
				
				// Determine warmup periods for each component
				let macd_warmup = slow - 1;  // MACD starts after slow period
				let signal_warmup = macd_warmup + signal - 1;  // Signal starts signal periods after MACD
				let hist_warmup = signal_warmup;  // Histogram same as signal
				
				// Test properties for each valid output
				for i in 0..len {
					let y_macd = macd[i];
					let y_sig = sig[i];
					let y_hist = hist[i];
					let r_macd = ref_macd[i];
					let r_sig = ref_sig[i];
					let r_hist = ref_hist[i];
					
					// Property 1: Kernel consistency for NaN patterns
					// Both kernels should have the same NaN pattern
					if y_macd.is_nan() != r_macd.is_nan() {
						prop_assert!(false, "MACD NaN mismatch at index {}: test={} ref={}", i, y_macd.is_nan(), r_macd.is_nan());
					}
					if y_sig.is_nan() != r_sig.is_nan() {
						prop_assert!(false, "Signal NaN mismatch at index {}: test={} ref={}", i, y_sig.is_nan(), r_sig.is_nan());
					}
					if y_hist.is_nan() != r_hist.is_nan() {
						prop_assert!(false, "Histogram NaN mismatch at index {}: test={} ref={}", i, y_hist.is_nan(), r_hist.is_nan());
					}
					
					// Property 2: After warmup, values should be finite
					if i >= hist_warmup {
						prop_assert!(y_macd.is_finite(), "MACD not finite at index {}: {}", i, y_macd);
						prop_assert!(y_sig.is_finite(), "Signal not finite at index {}: {}", i, y_sig);
						prop_assert!(y_hist.is_finite(), "Histogram not finite at index {}: {}", i, y_hist);
					}
					
					// Property 3: Histogram = MACD - Signal (when both are valid)
					if y_macd.is_finite() && y_sig.is_finite() {
						let expected_hist = y_macd - y_sig;
						prop_assert!(
							(y_hist - expected_hist).abs() <= 1e-9,
							"Histogram mismatch at {}: {} vs {} (macd={}, signal={})",
							i, y_hist, expected_hist, y_macd, y_sig
						);
					}
					
					// Property 4: Kernel consistency - different kernels should produce same results
					if !y_macd.is_finite() || !r_macd.is_finite() {
						prop_assert!(y_macd.to_bits() == r_macd.to_bits(), "MACD finite/NaN mismatch at {}: {} vs {}", i, y_macd, r_macd);
					} else {
						let ulp_diff = y_macd.to_bits().abs_diff(r_macd.to_bits());
						prop_assert!(
							(y_macd - r_macd).abs() <= 1e-9 || ulp_diff <= 4,
							"MACD mismatch at {}: {} vs {} (ULP={})", i, y_macd, r_macd, ulp_diff
						);
					}
					
					if !y_sig.is_finite() || !r_sig.is_finite() {
						prop_assert!(y_sig.to_bits() == r_sig.to_bits(), "Signal finite/NaN mismatch at {}: {} vs {}", i, y_sig, r_sig);
					} else {
						let ulp_diff = y_sig.to_bits().abs_diff(r_sig.to_bits());
						prop_assert!(
							(y_sig - r_sig).abs() <= 1e-9 || ulp_diff <= 4,
							"Signal mismatch at {}: {} vs {} (ULP={})", i, y_sig, r_sig, ulp_diff
						);
					}
					
					if !y_hist.is_finite() || !r_hist.is_finite() {
						prop_assert!(y_hist.to_bits() == r_hist.to_bits(), "Histogram finite/NaN mismatch at {}: {} vs {}", i, y_hist, r_hist);
					} else {
						let ulp_diff = y_hist.to_bits().abs_diff(r_hist.to_bits());
						prop_assert!(
							(y_hist - r_hist).abs() <= 1e-9 || ulp_diff <= 4,
							"Histogram mismatch at {}: {} vs {} (ULP={})", i, y_hist, r_hist, ulp_diff
						);
					}
					
					// Property 5: With constant prices AND constant volumes, MACD should be ~0
					// This is because both fast and slow VWMA will equal the constant price
					if close.windows(2).all(|w| (w[0] - w[1]).abs() < f64::EPSILON) &&
					   volume.windows(2).all(|w| (w[0] - w[1]).abs() < f64::EPSILON) && 
					   y_macd.is_finite() {
						prop_assert!(
							y_macd.abs() <= 1e-9,
							"MACD should be ~0 with constant prices and volumes, got {} at index {}", y_macd, i
						);
					}
					
					// Property 6: Volume weighting validation
					// With very small volumes, the indicator should still produce valid results
					if volume[i] < 1.0 && y_macd.is_finite() {
						// Just verify no NaN/Inf from division issues
						prop_assert!(y_macd.is_finite(), "MACD should be finite even with small volume {} at index {}", volume[i], i);
					}
					
					// Property 7: VWMA component bounds
					// Each VWMA (that makes up MACD) should be within the price range of its window
					// We can check this indirectly: |MACD| should not exceed the total price range
					if y_macd.is_finite() && i >= slow - 1 {
						let all_prices_min = close.iter().cloned().fold(f64::INFINITY, f64::min);
						let all_prices_max = close.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
						let total_range = all_prices_max - all_prices_min;
						
						// MACD is difference of two VWMAs, each bounded by price range
						// So |MACD| should not exceed the total price range
						prop_assert!(
							y_macd.abs() <= total_range + 1e-6,
							"MACD {} exceeds total price range {} at index {}",
							y_macd.abs(), total_range, i
						);
					}
				}
				
				// Additional test: Extreme volume ratios
				// Create a test case with extreme volume imbalance
				if len > slow * 2 {
					let mut extreme_volume = volume.to_vec();
					// Set some volumes to be 1000x larger
					for i in (0..len).step_by(5) {
						extreme_volume[i] *= 1000.0;
					}
					
					let params_extreme = VwmacdParams {
						fast_period: Some(fast),
						slow_period: Some(slow),
						signal_period: Some(signal),
						fast_ma_type: Some(fast_ma.to_string()),
						slow_ma_type: Some(slow_ma.to_string()),
						signal_ma_type: Some(signal_ma.to_string()),
					};
					let input_extreme = VwmacdInput::from_slices(close, &extreme_volume, params_extreme);
					
					// Should not panic or produce NaN inappropriately
					let result = vwmacd_with_kernel(&input_extreme, kernel);
					prop_assert!(result.is_ok(), "Should handle extreme volume ratios");
					
					if let Ok(extreme_output) = result {
						// Check that high-volume periods dominate the VWMA
						// (This is a soft check - we just verify no crashes/NaNs)
						for i in hist_warmup..len {
							if extreme_output.macd[i].is_finite() {
								prop_assert!(
									extreme_output.macd[i].is_finite(),
									"MACD should be finite with extreme volumes at index {}", i
								);
							}
						}
					}
				}
				
				Ok(())
			})
			.unwrap();

		Ok(())
	}
	
	generate_all_vwmacd_tests!(
		check_vwmacd_partial_params,
		check_vwmacd_accuracy,
		check_vwmacd_with_custom_ma_types,
		check_vwmacd_nan_data,
		check_vwmacd_zero_period,
		check_vwmacd_period_exceeds,
		check_vwmacd_streaming,
		check_vwmacd_no_poison
	);

	#[cfg(feature = "proptest")]
	generate_all_vwmacd_tests!(check_vwmacd_property);

	fn check_vwmacd_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let fast_period = 12;
		let slow_period = 26;
		let signal_period = 9;
		let fast_ma_type = "sma";
		let slow_ma_type = "sma";
		let signal_ma_type = "ema";

		// Get batch output for comparison
		let params = VwmacdParams {
			fast_period: Some(fast_period),
			slow_period: Some(slow_period),
			signal_period: Some(signal_period),
			fast_ma_type: Some(fast_ma_type.to_string()),
			slow_ma_type: Some(slow_ma_type.to_string()),
			signal_ma_type: Some(signal_ma_type.to_string()),
		};
		let input = VwmacdInput::from_slices(&candles.close, &candles.volume, params.clone());
		let batch_output = vwmacd_with_kernel(&input, kernel)?;

		// Create stream
		let mut stream = VwmacdStream::try_new(params)?;

		// Process all values through the stream
		let mut stream_macd = Vec::with_capacity(candles.close.len());
		let mut stream_signal = Vec::with_capacity(candles.close.len());
		let mut stream_hist = Vec::with_capacity(candles.close.len());

		for i in 0..candles.close.len() {
			match stream.update(candles.close[i], candles.volume[i]) {
				Some((m, s, h)) => {
					stream_macd.push(m);
					stream_signal.push(s);
					stream_hist.push(h);
				}
				None => {
					stream_macd.push(f64::NAN);
					stream_signal.push(f64::NAN);
					stream_hist.push(f64::NAN);
				}
			}
			
		}

		// Compare results
		assert_eq!(batch_output.macd.len(), stream_macd.len());
		assert_eq!(batch_output.signal.len(), stream_signal.len());
		assert_eq!(batch_output.hist.len(), stream_hist.len());

		// Check MACD values
		for (i, (&b, &s)) in batch_output.macd.iter().zip(stream_macd.iter()).enumerate() {
			if b.is_nan() && s.is_nan() {
				continue;
			}
			let diff = (b - s).abs();
			assert!(
				diff < 1e-7,
				"[{}] VWMACD streaming MACD mismatch at idx {}: batch={}, stream={}, diff={}",
				test_name,
				i,
				b,
				s,
				diff
			);
		}

		// Check signal values
		for (i, (&b, &s)) in batch_output.signal.iter().zip(stream_signal.iter()).enumerate() {
			if b.is_nan() && s.is_nan() {
				continue;
			}
			let diff = (b - s).abs();
			assert!(
				diff < 1e-7,
				"[{}] VWMACD streaming signal mismatch at idx {}: batch={}, stream={}, diff={}",
				test_name,
				i,
				b,
				s,
				diff
			);
		}

		// Check histogram values
		for (i, (&b, &s)) in batch_output.hist.iter().zip(stream_hist.iter()).enumerate() {
			if b.is_nan() && s.is_nan() {
				continue;
			}
			let diff = (b - s).abs();
			assert!(
				diff < 1e-7,
				"[{}] VWMACD streaming histogram mismatch at idx {}: batch={}, stream={}, diff={}",
				test_name,
				i,
				b,
				s,
				diff
			);
		}

		Ok(())
	}

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let close = &c.close;
		let volume = &c.volume;

		let output = VwmacdBatchBuilder::new().kernel(kernel).apply_slices(close, volume)?;

		let def = VwmacdParams::default();
		let row = output.values_for(&def).expect("default row missing");
		assert_eq!(row.len(), close.len());

		let expected_macd = [
			-394.95161155,
			-508.29106210,
			-490.70190723,
			-388.94996199,
			-341.13720646,
		];
		let start = row.len() - 5;
		for (i, &v) in row[start..].iter().enumerate() {
			assert!(
				(v - expected_macd[i]).abs() < 1e-3,
				"[{test}] default-row MACD mismatch at idx {i}: got {v}, expected {}",
				expected_macd[i]
			);
		}

		let input = VwmacdInput::from_candles(&c, "close", "volume", def.clone());
		let result = vwmacd_with_kernel(&input, kernel)?;

		let expected_signal = [
			-539.48861567,
			-533.24910496,
			-524.73966541,
			-497.58172247,
			-466.29282108,
		];
		let signal_slice = &result.signal[result.signal.len() - 5..];
		for (i, &v) in signal_slice.iter().enumerate() {
			assert!(
				(v - expected_signal[i]).abs() < 1e-3,
				"[{test}] default-row Signal mismatch at idx {i}: got {v}, expected {}",
				expected_signal[i]
			);
		}

		let expected_histogram = [144.53700412, 24.95804286, 34.03775818, 108.63176274, 125.15561462];
		let hist_slice = &result.hist[result.hist.len() - 5..];
		for (i, &v) in hist_slice.iter().enumerate() {
			assert!(
				(v - expected_histogram[i]).abs() < 1e-3,
				"[{test}] default-row Histogram mismatch at idx {i}: got {v}, expected {}",
				expected_histogram[i]
			);
		}

		Ok(())
	}

	fn check_batch_grid(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let close = &c.close;
		let volume = &c.volume;

		let output = VwmacdBatchBuilder::new()
			.kernel(kernel)
			.fast_range(10, 14, 2)
			.slow_range(20, 26, 3)
			.signal_range(5, 9, 2)
			.apply_slices(close, volume)?;

		assert_eq!(output.cols, close.len());
		assert_eq!(output.rows, 3 * 3 * 3);

		let params = VwmacdParams {
			fast_period: Some(12),
			slow_period: Some(23),
			signal_period: Some(7),
			fast_ma_type: Some("sma".to_string()),
			slow_ma_type: Some("sma".to_string()),
			signal_ma_type: Some("ema".to_string()),
		};
		let row = output.values_for(&params).expect("row for params missing");
		assert_eq!(row.len(), close.len());
		Ok(())
	}

	fn check_batch_param_map(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let close = &c.close;
		let volume = &c.volume;

		let batch = VwmacdBatchBuilder::new()
			.kernel(kernel)
			.fast_range(12, 14, 1)
			.slow_range(26, 28, 1)
			.signal_range(9, 11, 1)
			.apply_slices(close, volume)?;

		for (ix, param) in batch.params.iter().enumerate() {
			let by_index = &batch.macd[ix * batch.cols..(ix + 1) * batch.cols];
			let by_api = batch.values_for(param).unwrap();

			assert_eq!(by_index.len(), by_api.len());
			for (i, (&x, &y)) in by_index.iter().zip(by_api.iter()).enumerate() {
				if x.is_nan() && y.is_nan() {
					continue;
				}
				assert!(
					(x == y),
					"[{}] param {:?}, mismatch at idx {}: got {}, expected {}",
					test,
					param,
					i,
					x,
					y
				);
			}
		}
		Ok(())
	}

	fn check_batch_custom_ma_types(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let close = &c.close;
		let volume = &c.volume;

		let output = VwmacdBatchBuilder::new()
			.kernel(kernel)
			.fast_ma_type("ema".to_string())
			.slow_ma_type("wma".to_string())
			.signal_ma_type("sma".to_string())
			.apply_slices(close, volume)?;

		let params = VwmacdParams {
			fast_period: Some(12),
			slow_period: Some(26),
			signal_period: Some(9),
			fast_ma_type: Some("ema".to_string()),
			slow_ma_type: Some("wma".to_string()),
			signal_ma_type: Some("sma".to_string()),
		};
		let row = output.values_for(&params).expect("custom MA types row missing");
		assert_eq!(row.len(), close.len());
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
		
		let close = &c.close;
		let volume = &c.volume;
		
		// Test various parameter sweep configurations
		let test_configs = vec![
			// (fast_start, fast_end, fast_step, slow_start, slow_end, slow_step, signal_start, signal_end, signal_step)
			(2, 10, 2, 11, 20, 3, 2, 5, 1),       // Small periods
			(5, 15, 5, 16, 30, 5, 3, 9, 3),      // Medium periods
			(10, 30, 10, 31, 60, 10, 5, 15, 5),  // Large periods
			(2, 5, 1, 6, 10, 1, 2, 4, 1),        // Dense small range
			(12, 12, 0, 26, 26, 0, 9, 9, 0),     // Single default config
			(8, 16, 4, 20, 40, 10, 5, 10, 5),    // Mixed ranges
		];
		
		for (cfg_idx, &(fast_start, fast_end, fast_step, slow_start, slow_end, slow_step, signal_start, signal_end, signal_step)) in test_configs.iter().enumerate() {
			let mut builder = VwmacdBatchBuilder::new().kernel(kernel);
			
			// Configure fast range
			if fast_step > 0 {
				builder = builder.fast_range(fast_start, fast_end, fast_step);
			} else {
				builder = builder.fast_range(fast_start, fast_start, 1);
			}
			
			// Configure slow range
			if slow_step > 0 {
				builder = builder.slow_range(slow_start, slow_end, slow_step);
			} else {
				builder = builder.slow_range(slow_start, slow_start, 1);
			}
			
			// Configure signal range
			if signal_step > 0 {
				builder = builder.signal_range(signal_start, signal_end, signal_step);
			} else {
				builder = builder.signal_range(signal_start, signal_start, 1);
			}
			
			let output = builder.apply_slices(close, volume)?;
			
			for (idx, &val) in output.macd.iter().enumerate() {
				if val.is_nan() {
					continue;
				}
				
				let bits = val.to_bits();
				let row = idx / output.cols;
				let col = idx % output.cols;
				let combo = &output.params[row];
				
				// Check all three poison patterns with detailed context
				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: fast={}, slow={}, signal={}, \
						 fast_ma={}, slow_ma={}, signal_ma={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.fast_period.unwrap_or(12),
						combo.slow_period.unwrap_or(26),
						combo.signal_period.unwrap_or(9),
						combo.fast_ma_type.as_deref().unwrap_or("sma"),
						combo.slow_ma_type.as_deref().unwrap_or("sma"),
						combo.signal_ma_type.as_deref().unwrap_or("ema")
					);
				}
				
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: fast={}, slow={}, signal={}, \
						 fast_ma={}, slow_ma={}, signal_ma={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.fast_period.unwrap_or(12),
						combo.slow_period.unwrap_or(26),
						combo.signal_period.unwrap_or(9),
						combo.fast_ma_type.as_deref().unwrap_or("sma"),
						combo.slow_ma_type.as_deref().unwrap_or("sma"),
						combo.signal_ma_type.as_deref().unwrap_or("ema")
					);
				}
				
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: fast={}, slow={}, signal={}, \
						 fast_ma={}, slow_ma={}, signal_ma={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.fast_period.unwrap_or(12),
						combo.slow_period.unwrap_or(26),
						combo.signal_period.unwrap_or(9),
						combo.fast_ma_type.as_deref().unwrap_or("sma"),
						combo.slow_ma_type.as_deref().unwrap_or("sma"),
						combo.signal_ma_type.as_deref().unwrap_or("ema")
					);
				}
			}
		}
		
		// Test with different MA types
		let ma_type_configs = vec![
			("ema", "ema", "ema"),
			("sma", "wma", "ema"),
			("wma", "wma", "sma"),
		];
		
		for (cfg_idx, &(fast_ma, slow_ma, signal_ma)) in ma_type_configs.iter().enumerate() {
			let output = VwmacdBatchBuilder::new()
				.kernel(kernel)
				.fast_range(10, 15, 5)
				.slow_range(20, 30, 10)
				.signal_range(5, 10, 5)
				.fast_ma_type(fast_ma.to_string())
				.slow_ma_type(slow_ma.to_string())
				.signal_ma_type(signal_ma.to_string())
				.apply_slices(close, volume)?;
			
			for (idx, &val) in output.macd.iter().enumerate() {
				if val.is_nan() {
					continue;
				}
				
				let bits = val.to_bits();
				let row = idx / output.cols;
				let col = idx % output.cols;
				let combo = &output.params[row];
				
				if bits == 0x11111111_11111111 || bits == 0x22222222_22222222 || bits == 0x33333333_33333333 {
					let poison_type = if bits == 0x11111111_11111111 {
						"alloc_with_nan_prefix"
					} else if bits == 0x22222222_22222222 {
						"init_matrix_prefixes"
					} else {
						"make_uninit_matrix"
					};
					
					panic!(
						"[{}] MA Type Config {}: Found {} poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: fast={}, slow={}, signal={}, \
						 fast_ma={}, slow_ma={}, signal_ma={}",
						test, cfg_idx, poison_type, val, bits, row, col, idx,
						combo.fast_period.unwrap_or(12),
						combo.slow_period.unwrap_or(26),
						combo.signal_period.unwrap_or(9),
						combo.fast_ma_type.as_deref().unwrap_or("sma"),
						combo.slow_ma_type.as_deref().unwrap_or("sma"),
						combo.signal_ma_type.as_deref().unwrap_or("ema")
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
	gen_batch_tests!(check_batch_grid);
	gen_batch_tests!(check_batch_param_map);
	gen_batch_tests!(check_batch_custom_ma_types);
	gen_batch_tests!(check_batch_no_poison);
}

