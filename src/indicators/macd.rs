//! # Moving Average Convergence/Divergence (MACD)
//!
//! A trend-following momentum indicator that shows the relationship between two moving averages of a data series.
//! Calculates the MACD line, signal line, and histogram.
//!
//! ## Parameters
//! - **fast_period**: Shorter moving average period (default: 12)
//! - **slow_period**: Longer moving average period (default: 26)
//! - **signal_period**: Signal line moving average period (default: 9)
//! - **ma_type**: Moving average type for all components (default: "ema")
//!
//! ## Errors
//! - **AllValuesNaN**: All input values are NaN.
//! - **InvalidPeriod**: One or more periods are zero or exceed data length.
//! - **NotEnoughValidData**: Insufficient valid data points for requested period(s).
//!
//! ## Returns
//! - `Ok(MacdOutput)` on success, containing MACD, signal, and histogram vectors
//! - `Err(MacdError)` otherwise

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

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
	alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(feature = "wasm")]
use crate::utilities::helpers::detect_wasm_kernel;
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::error::Error;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum MacdData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct MacdOutput {
	pub macd: Vec<f64>,
	pub signal: Vec<f64>,
	pub hist: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct MacdParams {
	pub fast_period: Option<usize>,
	pub slow_period: Option<usize>,
	pub signal_period: Option<usize>,
	pub ma_type: Option<String>,
}

impl Default for MacdParams {
	fn default() -> Self {
		Self {
			fast_period: Some(12),
			slow_period: Some(26),
			signal_period: Some(9),
			ma_type: Some("ema".to_string()),
		}
	}
}

#[derive(Debug, Clone)]
pub struct MacdInput<'a> {
	pub data: MacdData<'a>,
	pub params: MacdParams,
}

impl<'a> MacdInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: MacdParams) -> Self {
		Self {
			data: MacdData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slice(sl: &'a [f64], p: MacdParams) -> Self {
		Self {
			data: MacdData::Slice(sl),
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", MacdParams::default())
	}
	#[inline]
	pub fn get_fast_period(&self) -> usize {
		self.params.fast_period.unwrap_or(12)
	}
	#[inline]
	pub fn get_slow_period(&self) -> usize {
		self.params.slow_period.unwrap_or(26)
	}
	#[inline]
	pub fn get_signal_period(&self) -> usize {
		self.params.signal_period.unwrap_or(9)
	}
	#[inline]
	pub fn get_ma_type(&self) -> String {
		self.params.ma_type.clone().unwrap_or_else(|| "ema".to_string())
	}
}

#[derive(Clone, Debug)]
pub struct MacdBuilder {
	fast_period: Option<usize>,
	slow_period: Option<usize>,
	signal_period: Option<usize>,
	ma_type: Option<String>,
	kernel: Kernel,
}

impl Default for MacdBuilder {
	fn default() -> Self {
		Self {
			fast_period: None,
			slow_period: None,
			signal_period: None,
			ma_type: None,
			kernel: Kernel::Auto,
		}
	}
}

impl MacdBuilder {
	#[inline]
	pub fn new() -> Self {
		Self::default()
	}
	#[inline]
	pub fn get_fast_period(mut self, n: usize) -> Self {
		self.fast_period = Some(n);
		self
	}
	#[inline]
	pub fn get_slow_period(mut self, n: usize) -> Self {
		self.slow_period = Some(n);
		self
	}
	#[inline]
	pub fn get_signal_period(mut self, n: usize) -> Self {
		self.signal_period = Some(n);
		self
	}
	#[inline]
	pub fn ma_type<S: Into<String>>(mut self, s: S) -> Self {
		self.ma_type = Some(s.into());
		self
	}
	#[inline]
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}

	#[inline]
	pub fn apply(self, c: &Candles) -> Result<MacdOutput, MacdError> {
		let p = MacdParams {
			fast_period: self.fast_period,
			slow_period: self.slow_period,
			signal_period: self.signal_period,
			ma_type: self.ma_type,
		};
		let i = MacdInput::from_candles(c, "close", p);
		macd_with_kernel(&i, self.kernel)
	}

	#[inline]
	pub fn apply_slice(self, d: &[f64]) -> Result<MacdOutput, MacdError> {
		let p = MacdParams {
			fast_period: self.fast_period,
			slow_period: self.slow_period,
			signal_period: self.signal_period,
			ma_type: self.ma_type,
		};
		let i = MacdInput::from_slice(d, p);
		macd_with_kernel(&i, self.kernel)
	}
}

#[derive(Debug, Error)]
pub enum MacdError {
	#[error("macd: All values are NaN.")]
	AllValuesNaN,
	#[error("macd: Invalid period: fast = {fast}, slow = {slow}, signal = {signal}, data length = {data_len}")]
	InvalidPeriod {
		fast: usize,
		slow: usize,
		signal: usize,
		data_len: usize,
	},
	#[error("macd: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },
	#[error("macd: Unknown MA type: {0}")]
	UnknownMA(String),
}

#[inline]
pub fn macd(input: &MacdInput) -> Result<MacdOutput, MacdError> {
	macd_with_kernel(input, Kernel::Auto)
}

pub fn macd_with_kernel(input: &MacdInput, kernel: Kernel) -> Result<MacdOutput, MacdError> {
	let data: &[f64] = match &input.data {
		MacdData::Candles { candles, source } => source_type(candles, source),
		MacdData::Slice(sl) => sl,
	};

	let len = data.len();
	let fast = input.get_fast_period();
	let slow = input.get_slow_period();
	let signal = input.get_signal_period();
	let ma_type = input.get_ma_type();

	let first = data.iter().position(|x| !x.is_nan()).ok_or(MacdError::AllValuesNaN)?;
	if fast == 0 || slow == 0 || signal == 0 || fast > len || slow > len || signal > len {
		return Err(MacdError::InvalidPeriod {
			fast,
			slow,
			signal,
			data_len: len,
		});
	}
	if (len - first) < slow {
		return Err(MacdError::NotEnoughValidData {
			needed: slow,
			valid: len - first,
		});
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};
	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => macd_scalar(data, fast, slow, signal, &ma_type, first),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => macd_avx2(data, fast, slow, signal, &ma_type, first),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => macd_avx512(data, fast, slow, signal, &ma_type, first),
			_ => unreachable!(),
		}
	}
}

#[inline(always)]
pub unsafe fn macd_scalar(
	data: &[f64],
	fast: usize,
	slow: usize,
	signal: usize,
	ma_type: &str,
	first: usize,
) -> Result<MacdOutput, MacdError> {
	use crate::indicators::moving_averages::ma::{ma, MaData};
	let len = data.len();
	let fast_ma = ma(ma_type, MaData::Slice(data), fast).map_err(|_| MacdError::AllValuesNaN)?;
	let slow_ma = ma(ma_type, MaData::Slice(data), slow).map_err(|_| MacdError::AllValuesNaN)?;

	// MACD warmup is when we have enough data for the slow MA
	let warmup = first + slow - 1;
	let mut macd = alloc_with_nan_prefix(len, warmup);
	for i in warmup..len {
		if fast_ma[i].is_nan() || slow_ma[i].is_nan() {
			continue;
		}
		macd[i] = fast_ma[i] - slow_ma[i];
	}
	let signal_ma = ma(ma_type, MaData::Slice(&macd), signal).map_err(|_| MacdError::AllValuesNaN)?;
	// Signal warmup is MACD warmup + signal period - 1
	let signal_warmup = warmup + signal - 1;
	let mut signal_vec = alloc_with_nan_prefix(len, signal_warmup);
	let mut hist = alloc_with_nan_prefix(len, signal_warmup);
	for i in first..len {
		if macd[i].is_nan() || signal_ma[i].is_nan() {
			continue;
		}
		signal_vec[i] = signal_ma[i];
		hist[i] = macd[i] - signal_ma[i];
	}
	Ok(MacdOutput {
		macd,
		signal: signal_vec,
		hist,
	})
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn macd_avx2(
	data: &[f64],
	fast: usize,
	slow: usize,
	signal: usize,
	ma_type: &str,
	first: usize,
) -> Result<MacdOutput, MacdError> {
	macd_scalar(data, fast, slow, signal, ma_type, first)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn macd_avx512(
	data: &[f64],
	fast: usize,
	slow: usize,
	signal: usize,
	ma_type: &str,
	first: usize,
) -> Result<MacdOutput, MacdError> {
	macd_scalar(data, fast, slow, signal, ma_type, first)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn macd_avx512_short(
	data: &[f64],
	fast: usize,
	slow: usize,
	signal: usize,
	ma_type: &str,
	first: usize,
) -> Result<MacdOutput, MacdError> {
	macd_avx512(data, fast, slow, signal, ma_type, first)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn macd_avx512_long(
	data: &[f64],
	fast: usize,
	slow: usize,
	signal: usize,
	ma_type: &str,
	first: usize,
) -> Result<MacdOutput, MacdError> {
	macd_avx512(data, fast, slow, signal, ma_type, first)
}

#[inline(always)]
pub fn macd_row_scalar(
	data: &[f64],
	fast: usize,
	slow: usize,
	signal: usize,
	ma_type: &str,
	first: usize,
) -> Result<MacdOutput, MacdError> {
	unsafe { macd_scalar(data, fast, slow, signal, ma_type, first) }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn macd_row_avx2(
	data: &[f64],
	fast: usize,
	slow: usize,
	signal: usize,
	ma_type: &str,
	first: usize,
) -> Result<MacdOutput, MacdError> {
	unsafe { macd_avx2(data, fast, slow, signal, ma_type, first) }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn macd_row_avx512(
	data: &[f64],
	fast: usize,
	slow: usize,
	signal: usize,
	ma_type: &str,
	first: usize,
) -> Result<MacdOutput, MacdError> {
	unsafe { macd_avx512(data, fast, slow, signal, ma_type, first) }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn macd_row_avx512_short(
	data: &[f64],
	fast: usize,
	slow: usize,
	signal: usize,
	ma_type: &str,
	first: usize,
) -> Result<MacdOutput, MacdError> {
	unsafe { macd_avx512_short(data, fast, slow, signal, ma_type, first) }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn macd_row_avx512_long(
	data: &[f64],
	fast: usize,
	slow: usize,
	signal: usize,
	ma_type: &str,
	first: usize,
) -> Result<MacdOutput, MacdError> {
	unsafe { macd_avx512_long(data, fast, slow, signal, ma_type, first) }
}

#[derive(Clone, Debug)]
pub struct MacdBatchRange {
	pub fast_period: (usize, usize, usize),
	pub slow_period: (usize, usize, usize),
	pub signal_period: (usize, usize, usize),
	pub ma_type: (String, String, String),
}

impl Default for MacdBatchRange {
	fn default() -> Self {
		Self {
			fast_period: (12, 12, 0),
			slow_period: (26, 26, 0),
			signal_period: (9, 9, 0),
			ma_type: ("ema".to_string(), "ema".to_string(), "".to_string()),
		}
	}
}

#[derive(Clone, Debug, Default)]
pub struct MacdBatchBuilder {
	range: MacdBatchRange,
	kernel: Kernel,
}

impl MacdBatchBuilder {
	pub fn new() -> Self {
		Self::default()
	}
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline]
	pub fn fast_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.fast_period = (start, end, step);
		self
	}
	#[inline]
	pub fn slow_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.slow_period = (start, end, step);
		self
	}
	#[inline]
	pub fn signal_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.signal_period = (start, end, step);
		self
	}
	#[inline]
	pub fn ma_type_static(mut self, s: &str) -> Self {
		self.range.ma_type = (s.to_string(), s.to_string(), "".to_string());
		self
	}

	pub fn apply_slice(self, data: &[f64]) -> Result<MacdBatchOutput, MacdError> {
		macd_batch_with_kernel(data, &self.range, self.kernel)
	}
	pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<MacdBatchOutput, MacdError> {
		MacdBatchBuilder::new().kernel(k).apply_slice(data)
	}
	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<MacdBatchOutput, MacdError> {
		let slice = source_type(c, src);
		self.apply_slice(slice)
	}
	pub fn with_default_candles(c: &Candles) -> Result<MacdBatchOutput, MacdError> {
		MacdBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
	}
}

pub fn macd_batch_with_kernel(data: &[f64], sweep: &MacdBatchRange, k: Kernel) -> Result<MacdBatchOutput, MacdError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => {
			return Err(MacdError::InvalidPeriod {
				fast: 0,
				slow: 0,
				signal: 0,
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
	macd_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct MacdBatchOutput {
	pub macd: Vec<f64>,
	pub signal: Vec<f64>,
	pub hist: Vec<f64>,
	pub combos: Vec<MacdParams>,
	pub rows: usize,
	pub cols: usize,
}

#[inline(always)]
pub fn expand_grid(r: &MacdBatchRange) -> Vec<MacdParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let fasts = axis_usize(r.fast_period);
	let slows = axis_usize(r.slow_period);
	let signals = axis_usize(r.signal_period);
	let ma_types = vec![r.ma_type.0.clone()]; // For now, static MA type

	let mut combos = vec![];
	for &f in &fasts {
		for &s in &slows {
			for &g in &signals {
				for t in &ma_types {
					combos.push(MacdParams {
						fast_period: Some(f),
						slow_period: Some(s),
						signal_period: Some(g),
						ma_type: Some(t.clone()),
					});
				}
			}
		}
	}
	combos
}

pub fn macd_batch_par_slice(data: &[f64], sweep: &MacdBatchRange, simd: Kernel) -> Result<MacdBatchOutput, MacdError> {
	let combos = expand_grid(sweep);
	let rows = combos.len();
	let cols = data.len();
	
	if cols == 0 {
		return Err(MacdError::AllValuesNaN);
	}

	// Use uninitialized memory for optimal performance
	let mut macd_buf = make_uninit_matrix(rows, cols);
	let mut signal_buf = make_uninit_matrix(rows, cols);
	let mut hist_buf = make_uninit_matrix(rows, cols);

	// Calculate warmup periods for each combination
	let first = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
	let warmup_periods: Vec<usize> = combos
		.iter()
		.map(|p| {
			let slow = p.slow_period.unwrap_or(26);
			let signal = p.signal_period.unwrap_or(9);
			// Signal warmup is MACD warmup + signal period - 1
			first + slow + signal - 2
		})
		.collect();

	// Initialize NaN prefixes for warmup periods
	init_matrix_prefixes(&mut macd_buf, cols, &warmup_periods);
	init_matrix_prefixes(&mut signal_buf, cols, &warmup_periods);
	init_matrix_prefixes(&mut hist_buf, cols, &warmup_periods);

	// Convert to mutable slices for computation
	let mut macd_guard = core::mem::ManuallyDrop::new(macd_buf);
	let mut signal_guard = core::mem::ManuallyDrop::new(signal_buf);
	let mut hist_guard = core::mem::ManuallyDrop::new(hist_buf);
	
	let macd_out: &mut [f64] = unsafe {
		core::slice::from_raw_parts_mut(macd_guard.as_mut_ptr() as *mut f64, macd_guard.len())
	};
	let signal_out: &mut [f64] = unsafe {
		core::slice::from_raw_parts_mut(signal_guard.as_mut_ptr() as *mut f64, signal_guard.len())
	};
	let hist_out: &mut [f64] = unsafe {
		core::slice::from_raw_parts_mut(hist_guard.as_mut_ptr() as *mut f64, hist_guard.len())
	};

	// Process each parameter combination
	for (idx, p) in combos.iter().enumerate() {
		let fast = p.fast_period.unwrap_or(12);
		let slow = p.slow_period.unwrap_or(26);
		let sig = p.signal_period.unwrap_or(9);
		let ma_type = p.ma_type.clone().unwrap_or_else(|| "ema".to_string());
		
		let row_start = idx * cols;
		let row_end = row_start + cols;
		
		let out = match unsafe {
			match simd {
				Kernel::Scalar => macd_scalar(data, fast, slow, sig, &ma_type, first),
				#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
				Kernel::Avx2 => macd_avx2(data, fast, slow, sig, &ma_type, first),
				#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
				Kernel::Avx512 => macd_avx512(data, fast, slow, sig, &ma_type, first),
				_ => unreachable!(),
			}
		} {
			Ok(out) => {
				macd_out[row_start..row_end].copy_from_slice(&out.macd);
				signal_out[row_start..row_end].copy_from_slice(&out.signal);
				hist_out[row_start..row_end].copy_from_slice(&out.hist);
			}
			Err(_) => {
				// Already filled with NaN by init_matrix_prefixes
			}
		};
	}

	// Convert back to Vec for output
	let macd = unsafe {
		Vec::from_raw_parts(
			macd_guard.as_mut_ptr() as *mut f64,
			macd_guard.len(),
			macd_guard.capacity(),
		)
	};
	let signal = unsafe {
		Vec::from_raw_parts(
			signal_guard.as_mut_ptr() as *mut f64,
			signal_guard.len(),
			signal_guard.capacity(),
		)
	};
	let hist = unsafe {
		Vec::from_raw_parts(
			hist_guard.as_mut_ptr() as *mut f64,
			hist_guard.len(),
			hist_guard.capacity(),
		)
	};

	Ok(MacdBatchOutput {
		macd,
		signal,
		hist,
		combos,
		rows,
		cols,
	})
}

#[cfg(any(feature = "python", feature = "wasm"))]
pub fn macd_batch_inner_into(
	data: &[f64],
	sweep: &MacdBatchRange,
	simd: Kernel,
	_fill_invalid: bool,  // Not needed since we pre-fill with NaN
	macd_out: &mut [f64],
	signal_out: &mut [f64],
	hist_out: &mut [f64],
) -> Result<Vec<MacdParams>, MacdError> {
	let combos = expand_grid(sweep);
	let rows = combos.len();
	let cols = data.len();
	let first = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
	
	// Ensure output buffers are correct size
	if macd_out.len() != rows * cols || signal_out.len() != rows * cols || hist_out.len() != rows * cols {
		return Err(MacdError::InvalidPeriod {
			fast: 0,
			slow: 0,
			signal: 0,
			data_len: data.len(),
		});
	}
	
	// Process each parameter combination
	for (idx, p) in combos.iter().enumerate() {
		let fast = p.fast_period.unwrap_or(12);
		let slow = p.slow_period.unwrap_or(26);
		let sig = p.signal_period.unwrap_or(9);
		let ma_type = p.ma_type.clone().unwrap_or_else(|| "ema".to_string());
		
		let row_start = idx * cols;
		let row_end = row_start + cols;
		
		match unsafe {
			match simd {
				Kernel::Scalar => macd_scalar(data, fast, slow, sig, &ma_type, first),
				#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
				Kernel::Avx2 => macd_avx2(data, fast, slow, sig, &ma_type, first),
				#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
				Kernel::Avx512 => macd_avx512(data, fast, slow, sig, &ma_type, first),
				_ => unreachable!(),
			}
		} {
			Ok(out) => {
				macd_out[row_start..row_end].copy_from_slice(&out.macd);
				signal_out[row_start..row_end].copy_from_slice(&out.signal);
				hist_out[row_start..row_end].copy_from_slice(&out.hist);
			}
			Err(_) => {
				// Output buffers should already be pre-filled with NaN by the caller
				// for the appropriate warmup periods, so we don't need to do anything here
			}
		};
	}
	
	Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "macd")]
#[pyo3(signature = (data, fast_period, slow_period, signal_period, ma_type, kernel=None))]
pub fn macd_py<'py>(
	py: Python<'py>,
	data: PyReadonlyArray1<'py, f64>,
	fast_period: usize,
	slow_period: usize,
	signal_period: usize,
	ma_type: &str,
	kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
	use numpy::{IntoPyArray, PyArrayMethods};

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, false)?;

	let params = MacdParams {
		fast_period: Some(fast_period),
		slow_period: Some(slow_period),
		signal_period: Some(signal_period),
		ma_type: Some(ma_type.to_string()),
	};
	let input = MacdInput::from_slice(slice_in, params);

	let (macd_vec, signal_vec, hist_vec) = py
		.allow_threads(|| {
			macd_with_kernel(&input, kern)
				.map(|o| (o.macd, o.signal, o.hist))
		})
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	Ok((
		macd_vec.into_pyarray(py),
		signal_vec.into_pyarray(py),
		hist_vec.into_pyarray(py),
	))
}

#[cfg(feature = "python")]
#[pyclass(name = "MacdStream")]
pub struct MacdStreamPy {
	fast_period: usize,
	slow_period: usize,
	signal_period: usize,
	ma_type: String,
	data_buffer: Vec<f64>,
}

#[cfg(feature = "python")]
#[pymethods]
impl MacdStreamPy {
	#[new]
	fn new(fast_period: usize, slow_period: usize, signal_period: usize, ma_type: &str) -> PyResult<Self> {
		Ok(MacdStreamPy {
			fast_period,
			slow_period,
			signal_period,
			ma_type: ma_type.to_string(),
			data_buffer: Vec::new(),
		})
	}

	fn update(&mut self, value: f64) -> Option<(f64, f64, f64)> {
		self.data_buffer.push(value);
		
		// Need at least slow_period + signal_period - 1 values
		let min_needed = self.slow_period + self.signal_period - 1;
		if self.data_buffer.len() < min_needed {
			return None;
		}
		
		// Calculate MACD on the buffer
		let params = MacdParams {
			fast_period: Some(self.fast_period),
			slow_period: Some(self.slow_period),
			signal_period: Some(self.signal_period),
			ma_type: Some(self.ma_type.clone()),
		};
		let input = MacdInput::from_slice(&self.data_buffer, params);
		
		match macd(&input) {
			Ok(output) => {
				let last_idx = output.macd.len() - 1;
				Some((output.macd[last_idx], output.signal[last_idx], output.hist[last_idx]))
			}
			Err(_) => None,
		}
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "macd_batch")]
#[pyo3(signature = (data, fast_period_range, slow_period_range, signal_period_range, ma_type, kernel=None))]
pub fn macd_batch_py<'py>(
	py: Python<'py>,
	data: PyReadonlyArray1<'py, f64>,
	fast_period_range: (usize, usize, usize),
	slow_period_range: (usize, usize, usize),
	signal_period_range: (usize, usize, usize),
	ma_type: &str,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, true)?;

	let sweep = MacdBatchRange {
		fast_period: fast_period_range,
		slow_period: slow_period_range,
		signal_period: signal_period_range,
		ma_type: (ma_type.to_string(), ma_type.to_string(), String::new()),
	};

	let combos = expand_grid(&sweep);
	let rows = combos.len();
	let cols = slice_in.len();

	// Pre-allocate output arrays
	let macd_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let signal_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let hist_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	
	let macd_slice = unsafe { macd_arr.as_slice_mut()? };
	let signal_slice = unsafe { signal_arr.as_slice_mut()? };
	let hist_slice = unsafe { hist_arr.as_slice_mut()? };

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
			macd_batch_inner_into(slice_in, &sweep, simd, true, macd_slice, signal_slice, hist_slice)
		})
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	let dict = PyDict::new(py);
	dict.set_item("macd", macd_arr.reshape((rows, cols))?)?;
	dict.set_item("signal", signal_arr.reshape((rows, cols))?)?;
	dict.set_item("hist", hist_arr.reshape((rows, cols))?)?;
	dict.set_item(
		"fast_periods",
		combos.iter().map(|p| p.fast_period.unwrap() as u64).collect::<Vec<_>>().into_pyarray(py),
	)?;
	dict.set_item(
		"slow_periods",
		combos.iter().map(|p| p.slow_period.unwrap() as u64).collect::<Vec<_>>().into_pyarray(py),
	)?;
	dict.set_item(
		"signal_periods",
		combos.iter().map(|p| p.signal_period.unwrap() as u64).collect::<Vec<_>>().into_pyarray(py),
	)?;

	Ok(dict)
}

// =============================================================================
// WASM BINDINGS
// =============================================================================

#[cfg(feature = "wasm")]
/// Write MACD directly to output slices - no allocations
pub fn macd_into_slices(
	macd_dst: &mut [f64],
	signal_dst: &mut [f64], 
	hist_dst: &mut [f64],
	input: &MacdInput,
	kern: Kernel,
) -> Result<(), MacdError> {
	// Get data from input
	let data: &[f64] = match &input.data {
		MacdData::Candles { candles, source } => source_type(candles, source),
		MacdData::Slice(sl) => sl,
	};
	
	let len = data.len();
	
	// Validate output slice lengths
	if macd_dst.len() != len || signal_dst.len() != len || hist_dst.len() != len {
		return Err(MacdError::InvalidPeriod {
			fast: 0,
			slow: 0, 
			signal: 0,
			data_len: len,
		});
	}
	
	// Compute MACD values
	let result = macd_with_kernel(input, kern)?;
	
	// Copy results to output slices
	macd_dst.copy_from_slice(&result.macd);
	signal_dst.copy_from_slice(&result.signal);
	hist_dst.copy_from_slice(&result.hist);
	
	Ok(())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[derive(Serialize, Deserialize)]
pub struct MacdResult {
	values: Vec<f64>, // [macd..., signal..., hist...]
	rows: usize,      // 3 for MACD
	cols: usize,      // data length
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl MacdResult {
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
pub fn macd_js(
	data: &[f64],
	fast_period: usize,
	slow_period: usize,
	signal_period: usize,
	ma_type: &str,
) -> Result<MacdResult, JsValue> {
	let params = MacdParams {
		fast_period: Some(fast_period),
		slow_period: Some(slow_period),
		signal_period: Some(signal_period),
		ma_type: Some(ma_type.to_string()),
	};
	let input = MacdInput::from_slice(data, params);
	
	// Single allocation for all outputs
	let mut values = vec![0.0; data.len() * 3];
	let (macd_slice, rest) = values.split_at_mut(data.len());
	let (signal_slice, hist_slice) = rest.split_at_mut(data.len());
	
	macd_into_slices(macd_slice, signal_slice, hist_slice, &input, detect_wasm_kernel())
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	Ok(MacdResult {
		values,
		rows: 3,
		cols: data.len(),
	})
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn macd_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn macd_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe { let _ = Vec::from_raw_parts(ptr, len, len); }
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn macd_into(
	in_ptr: *const f64,
	macd_ptr: *mut f64,
	signal_ptr: *mut f64,
	hist_ptr: *mut f64,
	len: usize,
	fast_period: usize,
	slow_period: usize,
	signal_period: usize,
	ma_type: &str,
) -> Result<(), JsValue> {
	if in_ptr.is_null() || macd_ptr.is_null() || signal_ptr.is_null() || hist_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}
	
	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);
		let params = MacdParams {
			fast_period: Some(fast_period),
			slow_period: Some(slow_period),
			signal_period: Some(signal_period),
			ma_type: Some(ma_type.to_string()),
		};
		let input = MacdInput::from_slice(data, params);
		
		// Check for aliasing with ALL output pointers
		let needs_temp = in_ptr == macd_ptr as *const f64 || 
						 in_ptr == signal_ptr as *const f64 || 
						 in_ptr == hist_ptr as *const f64;
		
		if needs_temp {
			// Use temporary buffers for all outputs
			let mut temp_macd = vec![0.0; len];
			let mut temp_signal = vec![0.0; len];
			let mut temp_hist = vec![0.0; len];
			
			macd_into_slices(&mut temp_macd, &mut temp_signal, &mut temp_hist, &input, detect_wasm_kernel())
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			
			// Copy results to output pointers
			let macd_out = std::slice::from_raw_parts_mut(macd_ptr, len);
			let signal_out = std::slice::from_raw_parts_mut(signal_ptr, len);
			let hist_out = std::slice::from_raw_parts_mut(hist_ptr, len);
			
			macd_out.copy_from_slice(&temp_macd);
			signal_out.copy_from_slice(&temp_signal);
			hist_out.copy_from_slice(&temp_hist);
		} else {
			// Direct write to output slices
			let macd_out = std::slice::from_raw_parts_mut(macd_ptr, len);
			let signal_out = std::slice::from_raw_parts_mut(signal_ptr, len);
			let hist_out = std::slice::from_raw_parts_mut(hist_ptr, len);
			
			macd_into_slices(macd_out, signal_out, hist_out, &input, detect_wasm_kernel())
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		
		Ok(())
	}
}

// Batch processing structures
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct MacdBatchConfig {
	pub fast_period_range: (usize, usize, usize),
	pub slow_period_range: (usize, usize, usize),
	pub signal_period_range: (usize, usize, usize),
	pub ma_type: String,
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct MacdBatchJsResult {
	pub macd: Vec<f64>,
	pub signal: Vec<f64>,
	pub hist: Vec<f64>,
	pub rows: usize,
	pub cols: usize,
	pub fast_periods: Vec<usize>,
	pub slow_periods: Vec<usize>,
	pub signal_periods: Vec<usize>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = macd_batch)]
pub fn macd_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	let config: MacdBatchConfig = serde_wasm_bindgen::from_value(config)
		.map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
	
	let sweep = MacdBatchRange {
		fast_period: config.fast_period_range,
		slow_period: config.slow_period_range,
		signal_period: config.signal_period_range,
		ma_type: (config.ma_type.clone(), config.ma_type.clone(), String::new()),
	};
	
	let combos = expand_grid(&sweep);
	let rows = combos.len();
	let cols = data.len();
	
	// Allocate output arrays
	let mut macd = vec![0.0; rows * cols];
	let mut signal = vec![0.0; rows * cols];
	let mut hist = vec![0.0; rows * cols];
	
	// Use the shared batch function
	let kernel = detect_best_batch_kernel();
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => Kernel::Scalar,
	};
	
	let result_combos = macd_batch_inner_into(data, &sweep, simd, true, &mut macd, &mut signal, &mut hist)
		.map_err(|e| JsValue::from_str(&format!("Batch computation error: {}", e)))?;
	
	let js_output = MacdBatchJsResult {
		macd,
		signal,
		hist,
		rows,
		cols,
		fast_periods: result_combos.iter().map(|p| p.fast_period.unwrap()).collect(),
		slow_periods: result_combos.iter().map(|p| p.slow_period.unwrap()).collect(),
		signal_periods: result_combos.iter().map(|p| p.signal_period.unwrap()).collect(),
	};
	
	serde_wasm_bindgen::to_value(&js_output)
		.map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;
	use crate::utilities::enums::Kernel;

	fn check_macd_partial_params(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file)?;

		let default_params = MacdParams {
			fast_period: None,
			slow_period: None,
			signal_period: None,
			ma_type: None,
		};
		let input = MacdInput::from_candles(&candles, "close", default_params);
		let output = macd_with_kernel(&input, kernel)?;
		assert_eq!(output.macd.len(), candles.close.len());
		assert_eq!(output.signal.len(), candles.close.len());
		assert_eq!(output.hist.len(), candles.close.len());
		Ok(())
	}

	fn check_macd_accuracy(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file)?;

		let params = MacdParams::default();
		let input = MacdInput::from_candles(&candles, "close", params);
		let result = macd_with_kernel(&input, kernel)?;

		let expected_macd = [
			-629.8674025082801,
			-600.2986584356258,
			-581.6188884820076,
			-551.1020443476082,
			-560.798510688488,
		];
		let expected_signal = [
			-721.9744591891067,
			-697.6392990384105,
			-674.4352169271299,
			-649.7685824112256,
			-631.9745680666781,
		];
		let expected_hist = [
			92.10705668082664,
			97.34064060278467,
			92.81632844512228,
			98.6665380636174,
			71.17605737819008,
		];
		let len = result.macd.len();
		let start = len - 5;
		for i in 0..5 {
			assert!((result.macd[start + i] - expected_macd[i]).abs() < 1e-1);
			assert!((result.signal[start + i] - expected_signal[i]).abs() < 1e-1);
			assert!((result.hist[start + i] - expected_hist[i]).abs() < 1e-1);
		}
		Ok(())
	}

	fn check_macd_zero_period(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let input_data = [10.0, 20.0, 30.0];
		let params = MacdParams {
			fast_period: Some(0),
			slow_period: Some(26),
			signal_period: Some(9),
			ma_type: Some("ema".to_string()),
		};
		let input = MacdInput::from_slice(&input_data, params);
		let res = macd_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] MACD should fail with zero fast period", test);
		Ok(())
	}

	fn check_macd_period_exceeds_length(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let data = [10.0, 20.0, 30.0];
		let params = MacdParams {
			fast_period: Some(12),
			slow_period: Some(26),
			signal_period: Some(9),
			ma_type: Some("ema".to_string()),
		};
		let input = MacdInput::from_slice(&data, params);
		let res = macd_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] MACD should fail with period exceeding length", test);
		Ok(())
	}

	fn check_macd_very_small_dataset(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let data = [42.0];
		let params = MacdParams {
			fast_period: Some(12),
			slow_period: Some(26),
			signal_period: Some(9),
			ma_type: Some("ema".to_string()),
		};
		let input = MacdInput::from_slice(&data, params);
		let res = macd_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] MACD should fail with insufficient data", test);
		Ok(())
	}

	fn check_macd_reinput(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file)?;

		let params = MacdParams::default();
		let input = MacdInput::from_candles(&candles, "close", params.clone());
		let first_result = macd_with_kernel(&input, kernel)?;

		let reinput = MacdInput::from_slice(&first_result.macd, params);
		let re_result = macd_with_kernel(&reinput, kernel)?;

		assert_eq!(re_result.macd.len(), first_result.macd.len());
		for i in 52..re_result.macd.len() {
			assert!(!re_result.macd[i].is_nan());
		}
		Ok(())
	}

	fn check_macd_nan_handling(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file)?;

		let params = MacdParams::default();
		let input = MacdInput::from_candles(&candles, "close", params);
		let res = macd_with_kernel(&input, kernel)?;
		let n = res.macd.len();
		if n > 240 {
			for i in 240..n {
				assert!(!res.macd[i].is_nan());
				assert!(!res.signal[i].is_nan());
				assert!(!res.hist[i].is_nan());
			}
		}
		Ok(())
	}

	macro_rules! generate_all_macd_tests {
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
	generate_all_macd_tests!(
		check_macd_partial_params,
		check_macd_accuracy,
		check_macd_zero_period,
		check_macd_period_exceeds_length,
		check_macd_very_small_dataset,
		check_macd_reinput,
		check_macd_nan_handling
	);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let output = MacdBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;
		let def = MacdParams::default();
		let row = output
			.combos
			.iter()
			.position(|prm| {
				prm.fast_period == def.fast_period
					&& prm.slow_period == def.slow_period
					&& prm.signal_period == def.signal_period
					&& prm.ma_type == def.ma_type
			})
			.expect("default row missing");
		let start = row * output.cols;
		let macd = &output.macd[start..start + output.cols];
		let signal = &output.signal[start..start + output.cols];
		let hist = &output.hist[start..start + output.cols];
		let expected_macd = [
			-629.8674025082801,
			-600.2986584356258,
			-581.6188884820076,
			-551.1020443476082,
			-560.798510688488,
		];
		let len = macd.len();
		let s = len - 5;
		for i in 0..5 {
			assert!((macd[s + i] - expected_macd[i]).abs() < 1e-1);
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
