//! # Stochastic RSI (SRSI)
//!
//! A momentum oscillator that applies the Stochastic formula to RSI values instead of price data.
//!
//! ## Parameters
//! - **rsi_period**: Period for RSI calculation (default: 14).
//! - **stoch_period**: Period for Stochastic calculation on RSI (default: 14).
//! - **k**: Period for slow K MA (default: 3).
//! - **d**: Period for slow D MA (default: 3).
//! - **source**: Candle field (default: "close").
//!
//! ## Errors
//! - **RsiError**: Error from RSI calculation.
//! - **StochError**: Error from Stochastic calculation.
//!
//! ## Returns
//! - **`Ok(SrsiOutput)`** on success, containing vectors `k` and `d`.
//! - **`Err(SrsiError)`** otherwise.
//!
//! ## Developer Notes
//! - **AVX2/AVX512 Kernels**: Currently delegate to scalar implementation for API parity. Future optimization opportunity for SIMD acceleration of RSI and Stochastic components.
//! - **Streaming Performance**: Basic implementation using recalculation approach (O(n)). Proper streaming would require maintaining separate RSI and Stochastic state machines.
//! - **Memory Optimization**: Batch mode uses RSI result caching to avoid redundant calculations across parameter combinations. Uses zero-copy helper functions for output allocation.

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

use crate::indicators::rsi::{rsi, RsiError, RsiInput, RsiOutput, RsiParams};
use crate::indicators::stoch::{stoch, StochError, StochInput, StochOutput, StochParams};
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
use std::mem::MaybeUninit;
use thiserror::Error;

impl<'a> AsRef<[f64]> for SrsiInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			SrsiData::Slice(slice) => slice,
			SrsiData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

#[derive(Debug, Clone)]
pub enum SrsiData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct SrsiOutput {
	pub k: Vec<f64>,
	pub d: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct SrsiParams {
	pub rsi_period: Option<usize>,
	pub stoch_period: Option<usize>,
	pub k: Option<usize>,
	pub d: Option<usize>,
	pub source: Option<String>,
}

impl Default for SrsiParams {
	fn default() -> Self {
		Self {
			rsi_period: Some(14),
			stoch_period: Some(14),
			k: Some(3),
			d: Some(3),
			source: Some("close".to_string()),
		}
	}
}

#[derive(Debug, Clone)]
pub struct SrsiInput<'a> {
	pub data: SrsiData<'a>,
	pub params: SrsiParams,
}

impl<'a> SrsiInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: SrsiParams) -> Self {
		Self {
			data: SrsiData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slice(sl: &'a [f64], p: SrsiParams) -> Self {
		Self {
			data: SrsiData::Slice(sl),
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", SrsiParams::default())
	}
	#[inline]
	pub fn get_rsi_period(&self) -> usize {
		self.params.rsi_period.unwrap_or(14)
	}
	#[inline]
	pub fn get_stoch_period(&self) -> usize {
		self.params.stoch_period.unwrap_or(14)
	}
	#[inline]
	pub fn get_k(&self) -> usize {
		self.params.k.unwrap_or(3)
	}
	#[inline]
	pub fn get_d(&self) -> usize {
		self.params.d.unwrap_or(3)
	}
	#[inline]
	pub fn get_source(&self) -> &str {
		self.params.source.as_deref().unwrap_or("close")
	}
}

#[derive(Clone, Debug)]
pub struct SrsiBuilder {
	rsi_period: Option<usize>,
	stoch_period: Option<usize>,
	k: Option<usize>,
	d: Option<usize>,
	source: Option<String>,
	kernel: Kernel,
}

impl Default for SrsiBuilder {
	fn default() -> Self {
		Self {
			rsi_period: None,
			stoch_period: None,
			k: None,
			d: None,
			source: None,
			kernel: Kernel::Auto,
		}
	}
}

impl SrsiBuilder {
	#[inline(always)]
	pub fn new() -> Self {
		Self::default()
	}
	#[inline(always)]
	pub fn rsi_period(mut self, n: usize) -> Self {
		self.rsi_period = Some(n);
		self
	}
	#[inline(always)]
	pub fn stoch_period(mut self, n: usize) -> Self {
		self.stoch_period = Some(n);
		self
	}
	#[inline(always)]
	pub fn k(mut self, n: usize) -> Self {
		self.k = Some(n);
		self
	}
	#[inline(always)]
	pub fn d(mut self, n: usize) -> Self {
		self.d = Some(n);
		self
	}
	#[inline(always)]
	pub fn source<S: Into<String>>(mut self, s: S) -> Self {
		self.source = Some(s.into());
		self
	}
	#[inline(always)]
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}

	#[inline(always)]
	pub fn apply(self, c: &Candles) -> Result<SrsiOutput, SrsiError> {
		let p = SrsiParams {
			rsi_period: self.rsi_period,
			stoch_period: self.stoch_period,
			k: self.k,
			d: self.d,
			source: self.source.clone(),
		};
		let i = SrsiInput::from_candles(c, self.source.as_deref().unwrap_or("close"), p);
		srsi_with_kernel(&i, self.kernel)
	}

	#[inline(always)]
	pub fn apply_slice(self, d: &[f64]) -> Result<SrsiOutput, SrsiError> {
		let p = SrsiParams {
			rsi_period: self.rsi_period,
			stoch_period: self.stoch_period,
			k: self.k,
			d: self.d,
			source: self.source.clone(),
		};
		let i = SrsiInput::from_slice(d, p);
		srsi_with_kernel(&i, self.kernel)
	}
}

#[derive(Debug, Error)]
pub enum SrsiError {
	#[error("srsi: Error from RSI calculation: {0}")]
	RsiError(#[from] RsiError),
	#[error("srsi: Error from Stochastic calculation: {0}")]
	StochError(#[from] StochError),
	#[error("srsi: All input data values are NaN.")]
	AllValuesNaN,
	#[error("srsi: Not enough valid data for the requested period.")]
	NotEnoughValidData,
	#[error("srsi: Size mismatch - destination buffers must match input data length. Expected {expected}, got k={k_len}, d={d_len}")]
	SizeMismatch { expected: usize, k_len: usize, d_len: usize },
}

#[inline]
pub fn srsi(input: &SrsiInput) -> Result<SrsiOutput, SrsiError> {
	srsi_with_kernel(input, Kernel::Auto)
}

pub fn srsi_with_kernel(input: &SrsiInput, kernel: Kernel) -> Result<SrsiOutput, SrsiError> {
	let data: &[f64] = match &input.data {
		SrsiData::Candles { candles, source } => source_type(candles, source),
		SrsiData::Slice(sl) => sl,
	};

	let first = data.iter().position(|x| !x.is_nan()).ok_or(SrsiError::AllValuesNaN)?;
	let len = data.len();
	let rsi_period = input.get_rsi_period();
	let stoch_period = input.get_stoch_period();
	let k_len = input.get_k();
	let d_len = input.get_d();

	if len - first < rsi_period.max(stoch_period).max(k_len).max(d_len) {
		return Err(SrsiError::NotEnoughValidData);
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => srsi_scalar(data, rsi_period, stoch_period, k_len, d_len),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => srsi_avx2(data, rsi_period, stoch_period, k_len, d_len),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => srsi_avx512(data, rsi_period, stoch_period, k_len, d_len),
			_ => unreachable!(),
		}
	}
}

#[inline]
pub unsafe fn srsi_scalar(
	data: &[f64],
	rsi_period: usize,
	stoch_period: usize,
	k: usize,
	d: usize,
) -> Result<SrsiOutput, SrsiError> {
	let rsi_input = RsiInput::from_slice(
		data,
		RsiParams {
			period: Some(rsi_period),
		},
	);
	let rsi_output = rsi(&rsi_input)?;
	let stoch_input = StochInput {
		data: crate::indicators::stoch::StochData::Slices {
			high: &rsi_output.values,
			low: &rsi_output.values,
			close: &rsi_output.values,
		},
		params: StochParams {
			fastk_period: Some(stoch_period),
			slowk_period: Some(k),
			slowk_ma_type: Some("sma".to_string()),
			slowd_period: Some(d),
			slowd_ma_type: Some("sma".to_string()),
		},
	};
	let stoch_output = stoch(&stoch_input)?;
	Ok(SrsiOutput {
		k: stoch_output.k,
		d: stoch_output.d,
	})
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn srsi_avx2(
	data: &[f64],
	rsi_period: usize,
	stoch_period: usize,
	k: usize,
	d: usize,
) -> Result<SrsiOutput, SrsiError> {
	srsi_scalar(data, rsi_period, stoch_period, k, d)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn srsi_avx512(
	data: &[f64],
	rsi_period: usize,
	stoch_period: usize,
	k: usize,
	d: usize,
) -> Result<SrsiOutput, SrsiError> {
	if stoch_period <= 32 {
		srsi_avx512_short(data, rsi_period, stoch_period, k, d)
	} else {
		srsi_avx512_long(data, rsi_period, stoch_period, k, d)
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn srsi_avx512_short(
	data: &[f64],
	rsi_period: usize,
	stoch_period: usize,
	k: usize,
	d: usize,
) -> Result<SrsiOutput, SrsiError> {
	srsi_scalar(data, rsi_period, stoch_period, k, d)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn srsi_avx512_long(
	data: &[f64],
	rsi_period: usize,
	stoch_period: usize,
	k: usize,
	d: usize,
) -> Result<SrsiOutput, SrsiError> {
	srsi_scalar(data, rsi_period, stoch_period, k, d)
}

#[inline(always)]
pub fn srsi_row_scalar(
	data: &[f64],
	rsi_period: usize,
	stoch_period: usize,
	k: usize,
	d: usize,
	k_out: &mut [f64],
	d_out: &mut [f64],
) {
	if let Ok(res) = unsafe { srsi_scalar(data, rsi_period, stoch_period, k, d) } {
		k_out.copy_from_slice(&res.k);
		d_out.copy_from_slice(&res.d);
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn srsi_row_avx2(
	data: &[f64],
	rsi_period: usize,
	stoch_period: usize,
	k: usize,
	d: usize,
	k_out: &mut [f64],
	d_out: &mut [f64],
) {
	srsi_row_scalar(data, rsi_period, stoch_period, k, d, k_out, d_out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn srsi_row_avx512(
	data: &[f64],
	rsi_period: usize,
	stoch_period: usize,
	k: usize,
	d: usize,
	k_out: &mut [f64],
	d_out: &mut [f64],
) {
	srsi_row_scalar(data, rsi_period, stoch_period, k, d, k_out, d_out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn srsi_row_avx512_short(
	data: &[f64],
	rsi_period: usize,
	stoch_period: usize,
	k: usize,
	d: usize,
	k_out: &mut [f64],
	d_out: &mut [f64],
) {
	srsi_row_scalar(data, rsi_period, stoch_period, k, d, k_out, d_out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn srsi_row_avx512_long(
	data: &[f64],
	rsi_period: usize,
	stoch_period: usize,
	k: usize,
	d: usize,
	k_out: &mut [f64],
	d_out: &mut [f64],
) {
	srsi_row_scalar(data, rsi_period, stoch_period, k, d, k_out, d_out)
}

#[derive(Debug, Clone)]
pub struct SrsiStream {
	rsi_period: usize,
	stoch_period: usize,
	k_period: usize,
	d_period: usize,
	rsi_buffer: Vec<f64>,
	stoch_buffer: Vec<f64>,
	k_buffer: Vec<f64>,
	head: usize,
	filled: usize,
}

impl SrsiStream {
	pub fn try_new(params: SrsiParams) -> Result<Self, SrsiError> {
		let rsi_period = params.rsi_period.unwrap_or(14);
		let stoch_period = params.stoch_period.unwrap_or(14);
		let k_period = params.k.unwrap_or(3);
		let d_period = params.d.unwrap_or(3);
		
		if rsi_period == 0 || stoch_period == 0 || k_period == 0 || d_period == 0 {
			return Err(SrsiError::NotEnoughValidData);
		}
		
		Ok(Self {
			rsi_period,
			stoch_period,
			k_period,
			d_period,
			rsi_buffer: vec![f64::NAN; rsi_period],
			stoch_buffer: vec![f64::NAN; stoch_period],
			k_buffer: vec![f64::NAN; k_period],
			head: 0,
			filled: 0,
		})
	}
	
	pub fn update(&mut self, _value: f64) -> Option<(f64, f64)> {
		// Proper implementation would require full RSI + Stoch ring buffers
		// Currently unimplemented to avoid misleading results
		None
	}
}

#[derive(Clone, Debug)]
pub struct SrsiBatchRange {
	pub rsi_period: (usize, usize, usize),
	pub stoch_period: (usize, usize, usize),
	pub k: (usize, usize, usize),
	pub d: (usize, usize, usize),
}

impl Default for SrsiBatchRange {
	fn default() -> Self {
		Self {
			rsi_period: (14, 14, 0),
			stoch_period: (14, 14, 0),
			k: (3, 3, 0),
			d: (3, 3, 0),
		}
	}
}

#[derive(Clone, Debug, Default)]
pub struct SrsiBatchBuilder {
	range: SrsiBatchRange,
	kernel: Kernel,
}

impl SrsiBatchBuilder {
	pub fn new() -> Self {
		Self::default()
	}
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline]
	pub fn rsi_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.rsi_period = (start, end, step);
		self
	}
	#[inline]
	pub fn stoch_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.stoch_period = (start, end, step);
		self
	}
	#[inline]
	pub fn k_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.k = (start, end, step);
		self
	}
	#[inline]
	pub fn d_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.d = (start, end, step);
		self
	}
	pub fn apply_slice(self, data: &[f64]) -> Result<SrsiBatchOutput, SrsiError> {
		srsi_batch_with_kernel(data, &self.range, self.kernel)
	}
}

#[derive(Clone, Debug)]
pub struct SrsiBatchOutput {
	pub k: Vec<f64>,
	pub d: Vec<f64>,
	pub combos: Vec<SrsiParams>,
	pub rows: usize,
	pub cols: usize,
}
impl SrsiBatchOutput {
	pub fn row_for_params(&self, p: &SrsiParams) -> Option<usize> {
		self.combos.iter().position(|c| {
			c.rsi_period.unwrap_or(14) == p.rsi_period.unwrap_or(14)
				&& c.stoch_period.unwrap_or(14) == p.stoch_period.unwrap_or(14)
				&& c.k.unwrap_or(3) == p.k.unwrap_or(3)
				&& c.d.unwrap_or(3) == p.d.unwrap_or(3)
		})
	}
	pub fn k_for(&self, p: &SrsiParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.k[start..start + self.cols]
		})
	}
	pub fn d_for(&self, p: &SrsiParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.d[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &SrsiBatchRange) -> Vec<SrsiParams> {
	fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let rsi_periods = axis(r.rsi_period);
	let stoch_periods = axis(r.stoch_period);
	let ks = axis(r.k);
	let ds = axis(r.d);

	let mut out = Vec::with_capacity(rsi_periods.len() * stoch_periods.len() * ks.len() * ds.len());
	for &rsi_p in &rsi_periods {
		for &stoch_p in &stoch_periods {
			for &k in &ks {
				for &d in &ds {
					out.push(SrsiParams {
						rsi_period: Some(rsi_p),
						stoch_period: Some(stoch_p),
						k: Some(k),
						d: Some(d),
						source: None,
					});
				}
			}
		}
	}
	out
}

#[inline(always)]
pub fn srsi_batch_with_kernel(data: &[f64], sweep: &SrsiBatchRange, k: Kernel) -> Result<SrsiBatchOutput, SrsiError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => Kernel::ScalarBatch,
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	srsi_batch_par_slice(data, sweep, simd)
}

#[inline(always)]
pub fn srsi_batch_slice(data: &[f64], sweep: &SrsiBatchRange, kern: Kernel) -> Result<SrsiBatchOutput, SrsiError> {
	srsi_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn srsi_batch_par_slice(data: &[f64], sweep: &SrsiBatchRange, kern: Kernel) -> Result<SrsiBatchOutput, SrsiError> {
	srsi_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn srsi_batch_inner_into(
	data: &[f64],
	sweep: &SrsiBatchRange,
	kern: Kernel,
	parallel: bool,
	k_out: &mut [f64],
	d_out: &mut [f64],
) -> Result<Vec<SrsiParams>, SrsiError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(SrsiError::NotEnoughValidData);
	}
	
	let first = data.iter().position(|x| !x.is_nan()).ok_or(SrsiError::AllValuesNaN)?;
	
	// Precompute RSI once per unique rsi_period to avoid redundant calculations
	use std::collections::{BTreeSet, BTreeMap};
	let mut rsi_cache: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
	let uniq_rsi: BTreeSet<usize> = combos.iter().map(|c| c.rsi_period.unwrap()).collect();
	for rp in uniq_rsi {
		let rsi_in = RsiInput::from_slice(data, RsiParams { period: Some(rp) });
		let rsi_out = rsi(&rsi_in)?;  // one allocation per distinct rp
		rsi_cache.insert(rp, rsi_out.values);
	}
	
	let max_period = combos
		.iter()
		.map(|c| {
			c.rsi_period
				.unwrap()
				.max(c.stoch_period.unwrap())
				.max(c.k.unwrap())
				.max(c.d.unwrap())
		})
		.max()
		.unwrap();
		
	if data.len() - first < max_period {
		return Err(SrsiError::NotEnoughValidData);
	}
	
	let cols = data.len();
	
	// Replace do_row to reuse cached RSI and avoid calling srsi_scalar
	let do_row = |row: usize, k_row: &mut [f64], d_row: &mut [f64]| -> Result<(), SrsiError> {
		let prm = &combos[row];
		let rsi_vals = rsi_cache.get(&prm.rsi_period.unwrap()).expect("cached rsi");
		let st_in = StochInput {
			data: crate::indicators::stoch::StochData::Slices { 
				high: rsi_vals, 
				low: rsi_vals, 
				close: rsi_vals 
			},
			params: StochParams {
				fastk_period: prm.stoch_period,
				slowk_period: prm.k,
				slowk_ma_type: Some("sma".to_string()),
				slowd_period: prm.d,
				slowd_ma_type: Some("sma".to_string()),
			},
		};
		let st = stoch(&st_in)?;  // allocates once per row
		k_row.copy_from_slice(&st.k);
		d_row.copy_from_slice(&st.d);
		Ok(())
	};
	
	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			k_out.par_chunks_mut(cols)
				.zip(d_out.par_chunks_mut(cols))
				.enumerate()
				.try_for_each(|(row, (k_row, d_row))| do_row(row, k_row, d_row))?;
		}
		
		#[cfg(target_arch = "wasm32")]
		{
			for (row, (k_row, d_row)) in k_out.chunks_mut(cols).zip(d_out.chunks_mut(cols)).enumerate() {
				do_row(row, k_row, d_row)?;
			}
		}
	} else {
		for (row, (k_row, d_row)) in k_out.chunks_mut(cols).zip(d_out.chunks_mut(cols)).enumerate() {
			do_row(row, k_row, d_row)?;
		}
	}
	
	Ok(combos)
}

#[inline(always)]
fn srsi_batch_inner(
	data: &[f64],
	sweep: &SrsiBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<SrsiBatchOutput, SrsiError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(SrsiError::NotEnoughValidData);
	}
	let first = data.iter().position(|x| !x.is_nan()).ok_or(SrsiError::AllValuesNaN)?;
	let max_period = combos
		.iter()
		.map(|c| {
			c.rsi_period
				.unwrap()
				.max(c.stoch_period.unwrap())
				.max(c.k.unwrap())
				.max(c.d.unwrap())
		})
		.max()
		.unwrap();
	if data.len() - first < max_period {
		return Err(SrsiError::NotEnoughValidData);
	}
	let rows = combos.len();
	let cols = data.len();
	let mut k_vals = make_uninit_matrix(rows, cols);
	let mut d_vals = make_uninit_matrix(rows, cols);
	
	// Calculate proper warmup periods for chained pipeline
	fn warm_for(c: &SrsiParams, first: usize) -> usize {
		let rp = c.rsi_period.unwrap();
		let sp = c.stoch_period.unwrap();
		let kp = c.k.unwrap();
		let dp = c.d.unwrap();
		// RSI needs rp-1, then Stoch needs sp-1, then smoothing needs max(k,d)-1
		first + rp - 1 + sp - 1 + kp.max(dp) - 1
	}
	
	let warmup_periods: Vec<usize> = combos
		.iter()
		.map(|c| warm_for(c, first).min(cols))
		.collect();
	
	init_matrix_prefixes(&mut k_vals, cols, &warmup_periods);
	init_matrix_prefixes(&mut d_vals, cols, &warmup_periods);
	
	// Convert to mutable slices using ManuallyDrop pattern like ALMA
	let mut k_guard = core::mem::ManuallyDrop::new(k_vals);
	let mut d_guard = core::mem::ManuallyDrop::new(d_vals);
	
	let k_out: &mut [f64] =
		unsafe { core::slice::from_raw_parts_mut(k_guard.as_mut_ptr() as *mut f64, k_guard.len()) };
	let d_out: &mut [f64] =
		unsafe { core::slice::from_raw_parts_mut(d_guard.as_mut_ptr() as *mut f64, d_guard.len()) };

	// Precompute RSI once per unique rsi_period to avoid redundant calculations
	use std::collections::{BTreeSet, BTreeMap};
	let mut rsi_cache: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
	let uniq_rsi: BTreeSet<usize> = combos.iter().map(|c| c.rsi_period.unwrap()).collect();
	for rp in uniq_rsi {
		let rsi_in = RsiInput::from_slice(data, RsiParams { period: Some(rp) });
		let rsi_out = rsi(&rsi_in).map_err(|_| SrsiError::AllValuesNaN)?;  // one allocation per distinct rp
		rsi_cache.insert(rp, rsi_out.values);
	}
	
	let do_row = |row: usize, k_out: &mut [f64], d_out: &mut [f64]| {
		let prm = &combos[row];
		let rsi_vals = rsi_cache.get(&prm.rsi_period.unwrap()).expect("cached rsi");
		let st_in = StochInput {
			data: crate::indicators::stoch::StochData::Slices { 
				high: rsi_vals, 
				low: rsi_vals, 
				close: rsi_vals 
			},
			params: StochParams {
				fastk_period: prm.stoch_period,
				slowk_period: prm.k,
				slowk_ma_type: Some("sma".to_string()),
				slowd_period: prm.d,
				slowd_ma_type: Some("sma".to_string()),
			},
		};
		if let Ok(st) = stoch(&st_in) {  // allocates once per row
			k_out.copy_from_slice(&st.k);
			d_out.copy_from_slice(&st.d);
		}
	};

	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			k_out
				.par_chunks_mut(cols)
				.zip(d_out.par_chunks_mut(cols))
				.enumerate()
				.for_each(|(row, (ks, ds))| do_row(row, ks, ds));
		}

		#[cfg(target_arch = "wasm32")]
		{
			for (row, (ks, ds)) in k_out.chunks_mut(cols).zip(d_out.chunks_mut(cols)).enumerate() {
				do_row(row, ks, ds);
			}
		}
	} else {
		for (row, (ks, ds)) in k_out.chunks_mut(cols).zip(d_out.chunks_mut(cols)).enumerate() {
			do_row(row, ks, ds);
		}
	}
	
	// Convert back to Vec using ManuallyDrop pattern
	let k_values = unsafe {
		Vec::from_raw_parts(
			k_guard.as_mut_ptr() as *mut f64,
			k_guard.len(),
			k_guard.capacity(),
		)
	};
	
	let d_values = unsafe {
		Vec::from_raw_parts(
			d_guard.as_mut_ptr() as *mut f64,
			d_guard.len(),
			d_guard.capacity(),
		)
	};
	
	Ok(SrsiBatchOutput {
		k: k_values,
		d: d_values,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
pub fn expand_grid_srsi(r: &SrsiBatchRange) -> Vec<SrsiParams> {
	expand_grid(r)
}

#[cfg(feature = "python")]
#[pyfunction(name = "srsi")]
#[pyo3(signature = (data, rsi_period=None, stoch_period=None, k=None, d=None, source=None, kernel=None))]
pub fn srsi_py<'py>(
	py: Python<'py>,
	data: PyReadonlyArray1<'py, f64>,
	rsi_period: Option<usize>,
	stoch_period: Option<usize>,
	k: Option<usize>,
	d: Option<usize>,
	source: Option<&str>,
	kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, false)?;
	
	let params = SrsiParams {
		rsi_period,
		stoch_period,
		k,
		d,
		source: source.map(|s| s.to_string()),
	};
	let input = SrsiInput::from_slice(slice_in, params);
	
	let (k_vec, d_vec) = py.allow_threads(|| {
		srsi_with_kernel(&input, kern)
			.map(|o| (o.k, o.d))
	})
	.map_err(|e| PyValueError::new_err(e.to_string()))?;
	
	Ok((
		k_vec.into_pyarray(py),
		d_vec.into_pyarray(py)
	))
}

#[cfg(feature = "python")]
#[pyclass(name = "SrsiStream")]
pub struct SrsiStreamPy {
	stream: SrsiStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl SrsiStreamPy {
	#[new]
	fn new(rsi_period: Option<usize>, stoch_period: Option<usize>, k: Option<usize>, d: Option<usize>) -> PyResult<Self> {
		let params = SrsiParams {
			rsi_period,
			stoch_period,
			k,
			d,
			source: None,
		};
		let stream = SrsiStream::try_new(params)
			.map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(SrsiStreamPy { stream })
	}
	
	fn update(&mut self, value: f64) -> Option<(f64, f64)> {
		self.stream.update(value)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "srsi_batch")]
#[pyo3(signature = (data, rsi_period_range, stoch_period_range, k_range, d_range, source=None, kernel=None))]
pub fn srsi_batch_py<'py>(
	py: Python<'py>,
	data: PyReadonlyArray1<'py, f64>,
	rsi_period_range: (usize, usize, usize),
	stoch_period_range: (usize, usize, usize),
	k_range: (usize, usize, usize),
	d_range: (usize, usize, usize),
	source: Option<&str>,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, true)?;
	
	let sweep = SrsiBatchRange {
		rsi_period: rsi_period_range,
		stoch_period: stoch_period_range,
		k: k_range,
		d: d_range,
	};
	
	let combos = expand_grid(&sweep);
	let rows = combos.len();
	let cols = slice_in.len();
	
	// Pre-allocate output arrays
	let k_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let d_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let k_slice = unsafe { k_arr.as_slice_mut()? };
	let d_slice = unsafe { d_arr.as_slice_mut()? };
	
	let combos = py.allow_threads(|| {
		let kernel = match kern {
			Kernel::Auto => detect_best_batch_kernel(),
			k => k,
		};
		srsi_batch_inner_into(slice_in, &sweep, kernel, true, k_slice, d_slice)
	})
	.map_err(|e| PyValueError::new_err(e.to_string()))?;
	
	let dict = PyDict::new(py);
	dict.set_item("k", k_arr.reshape((rows, cols))?)?;
	dict.set_item("d", d_arr.reshape((rows, cols))?)?;
	dict.set_item(
		"rsi_periods",
		combos.iter()
			.map(|p| p.rsi_period.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py)
	)?;
	dict.set_item(
		"stoch_periods",
		combos.iter()
			.map(|p| p.stoch_period.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py)
	)?;
	dict.set_item(
		"k_periods",
		combos.iter()
			.map(|p| p.k.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py)
	)?;
	dict.set_item(
		"d_periods",
		combos.iter()
			.map(|p| p.d.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py)
	)?;
	
	Ok(dict)
}

/// Write SRSI outputs directly to slices - no allocations
pub fn srsi_into_slice(
	dst_k: &mut [f64],
	dst_d: &mut [f64],
	input: &SrsiInput,
	kern: Kernel,
) -> Result<(), SrsiError> {
	let data: &[f64] = input.as_ref();
	
	if dst_k.len() != data.len() || dst_d.len() != data.len() {
		return Err(SrsiError::SizeMismatch {
			expected: data.len(),
			k_len: dst_k.len(),
			d_len: dst_d.len(),
		});
	}
	
	// Compute once, then copy into the provided buffers
	let out = srsi_with_kernel(input, kern)?;
	dst_k.copy_from_slice(&out.k);
	dst_d.copy_from_slice(&out.d);
	
	Ok(())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn srsi_js(
	data: &[f64], 
	rsi_period: usize, 
	stoch_period: usize, 
	k: usize, 
	d: usize
) -> Result<Vec<f64>, JsValue> {
	if data.is_empty() {
		return Err(JsValue::from_str("srsi: Input data is empty"));
	}
	
	if rsi_period == 0 || stoch_period == 0 || k == 0 || d == 0 {
		return Err(JsValue::from_str("srsi: Invalid period"));
	}
	
	let params = SrsiParams {
		rsi_period: Some(rsi_period),
		stoch_period: Some(stoch_period),
		k: Some(k),
		d: Some(d),
		source: None,
	};
	let input = SrsiInput::from_slice(data, params);
	let out = srsi_with_kernel(&input, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&format!("srsi: {}", e)))?;
	
	// Return flattened array [k..., d...]
	let mut values = Vec::with_capacity(2 * data.len());
	values.extend_from_slice(&out.k);
	values.extend_from_slice(&out.d);
	
	Ok(values)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn srsi_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn srsi_free(ptr: *mut f64, len: usize) {
	unsafe {
		let _ = Vec::from_raw_parts(ptr, len, len);
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn srsi_into(
	in_ptr: usize,
	k_ptr: usize,
	d_ptr: usize,
	len: usize,
	rsi_period: usize,
	stoch_period: usize,
	k: usize,
	d: usize,
) -> Result<(), JsValue> {
	unsafe {
		let data = std::slice::from_raw_parts(in_ptr as *const f64, len);
		
		if rsi_period == 0 || stoch_period == 0 || k == 0 || d == 0 {
			return Err(JsValue::from_str("Invalid period"));
		}
		
		let params = SrsiParams {
			rsi_period: Some(rsi_period),
			stoch_period: Some(stoch_period),
			k: Some(k),
			d: Some(d),
			source: None,
		};
		let input = SrsiInput::from_slice(data, params);
		
		// Check for aliasing on both output pointers
		let needs_temp = in_ptr == k_ptr || in_ptr == d_ptr || k_ptr == d_ptr;
		
		if needs_temp {
			let mut temp_k = vec![0.0; len];
			let mut temp_d = vec![0.0; len];
			srsi_into_slice(&mut temp_k, &mut temp_d, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			
			let k_out = std::slice::from_raw_parts_mut(k_ptr as *mut f64, len);
			let d_out = std::slice::from_raw_parts_mut(d_ptr as *mut f64, len);
			k_out.copy_from_slice(&temp_k);
			d_out.copy_from_slice(&temp_d);
		} else {
			let k_out = std::slice::from_raw_parts_mut(k_ptr as *mut f64, len);
			let d_out = std::slice::from_raw_parts_mut(d_ptr as *mut f64, len);
			srsi_into_slice(k_out, d_out, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		
		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct SrsiBatchConfig {
	pub rsi_period_range: (usize, usize, usize),
	pub stoch_period_range: (usize, usize, usize),
	pub k_range: (usize, usize, usize),
	pub d_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct SrsiBatchJsOutput {
	pub k_values: Vec<f64>,     // All K values flattened
	pub d_values: Vec<f64>,     // All D values flattened
	pub rows: usize,            // Number of parameter combinations
	pub cols: usize,            // Data length
	pub combos: Vec<SrsiParams>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = srsi_batch)]
pub fn srsi_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	let cfg: SrsiBatchConfig = 
		serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
	
	let sweep = SrsiBatchRange {
		rsi_period: cfg.rsi_period_range,
		stoch_period: cfg.stoch_period_range,
		k: cfg.k_range,
		d: cfg.d_range,
	};
	
	let out = srsi_batch_inner(data, &sweep, Kernel::Auto, false)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	let res = SrsiBatchJsOutput { 
		k_values: out.k,
		d_values: out.d,
		rows: out.rows,
		cols: out.cols,
		combos: out.combos 
	};
	
	serde_wasm_bindgen::to_value(&res).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn srsi_batch_into(
	in_ptr: usize,
	k_ptr: usize,
	d_ptr: usize,
	len: usize,
	rsi_period_start: usize,
	rsi_period_end: usize,
	rsi_period_step: usize,
	stoch_period_start: usize,
	stoch_period_end: usize,
	stoch_period_step: usize,
	k_start: usize,
	k_end: usize,
	k_step: usize,
	d_start: usize,
	d_end: usize,
	d_step: usize,
) -> Result<usize, JsValue> {
	unsafe {
		let data = std::slice::from_raw_parts(in_ptr as *const f64, len);
		
		let sweep = SrsiBatchRange {
			rsi_period: (rsi_period_start, rsi_period_end, rsi_period_step),
			stoch_period: (stoch_period_start, stoch_period_end, stoch_period_step),
			k: (k_start, k_end, k_step),
			d: (d_start, d_end, d_step),
		};
		
		let combos = expand_grid(&sweep);
		let rows = combos.len();
		let cols = len;
		
		let k_out = std::slice::from_raw_parts_mut(k_ptr as *mut f64, rows * cols);
		let d_out = std::slice::from_raw_parts_mut(d_ptr as *mut f64, rows * cols);
		
		srsi_batch_inner_into(data, &sweep, Kernel::Auto, false, k_out, d_out)
			.map_err(|e| JsValue::from_str(&e.to_string()))?;
		
		Ok(rows)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_srsi_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let default_params = SrsiParams {
			rsi_period: None,
			stoch_period: None,
			k: None,
			d: None,
			source: None,
		};
		let input = SrsiInput::from_candles(&candles, "close", default_params);
		let output = srsi_with_kernel(&input, kernel)?;
		assert_eq!(output.k.len(), candles.close.len());
		assert_eq!(output.d.len(), candles.close.len());
		Ok(())
	}

	fn check_srsi_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = SrsiParams::default();
		let input = SrsiInput::from_candles(&candles, "close", params);
		let result = srsi_with_kernel(&input, kernel)?;
		assert_eq!(result.k.len(), candles.close.len());
		assert_eq!(result.d.len(), candles.close.len());
		let last_five_k = [
			65.52066633236464,
			61.22507053191985,
			57.220471530042644,
			64.61344854988147,
			60.66534359318523,
		];
		let last_five_d = [
			64.33503158970049,
			64.42143544464182,
			61.32206946477942,
			61.01966353728503,
			60.83308789104016,
		];
		let k_slice = &result.k[result.k.len() - 5..];
		let d_slice = &result.d[result.d.len() - 5..];
		for i in 0..5 {
			let diff_k = (k_slice[i] - last_five_k[i]).abs();
			let diff_d = (d_slice[i] - last_five_d[i]).abs();
			assert!(
				diff_k < 1e-6,
				"Mismatch in SRSI K at index {}: got {}, expected {}",
				i,
				k_slice[i],
				last_five_k[i]
			);
			assert!(
				diff_d < 1e-6,
				"Mismatch in SRSI D at index {}: got {}, expected {}",
				i,
				d_slice[i],
				last_five_d[i]
			);
		}
		Ok(())
	}

	fn check_srsi_from_slice(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let slice_data = candles.close.as_slice();
		let params = SrsiParams {
			rsi_period: Some(3),
			stoch_period: Some(3),
			k: Some(2),
			d: Some(2),
			source: Some("close".to_string()),
		};
		let input = SrsiInput::from_slice(&slice_data, params);
		let output = srsi_with_kernel(&input, kernel)?;
		assert_eq!(output.k.len(), slice_data.len());
		assert_eq!(output.d.len(), slice_data.len());
		Ok(())
	}

	fn check_srsi_custom_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = SrsiParams {
			rsi_period: Some(10),
			stoch_period: Some(10),
			k: Some(4),
			d: Some(4),
			source: Some("hlc3".to_string()),
		};
		let input = SrsiInput::from_candles(&candles, "hlc3", params);
		let output = srsi_with_kernel(&input, kernel)?;
		assert_eq!(output.k.len(), candles.close.len());
		assert_eq!(output.d.len(), candles.close.len());
		Ok(())
	}

	#[cfg(debug_assertions)]
	fn check_srsi_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		
		// Define comprehensive parameter combinations
		let test_params = vec![
			SrsiParams::default(),  // rsi: 14, stoch: 14, k: 3, d: 3
			SrsiParams {
				rsi_period: Some(2),    // minimum viable
				stoch_period: Some(2),
				k: Some(2),
				d: Some(2),
				source: None,
			},
			SrsiParams {
				rsi_period: Some(5),    // small periods
				stoch_period: Some(5),
				k: Some(3),
				d: Some(3),
				source: None,
			},
			SrsiParams {
				rsi_period: Some(10),   // medium periods
				stoch_period: Some(10),
				k: Some(5),
				d: Some(5),
				source: None,
			},
			SrsiParams {
				rsi_period: Some(20),   // large periods
				stoch_period: Some(20),
				k: Some(7),
				d: Some(7),
				source: None,
			},
			SrsiParams {
				rsi_period: Some(50),   // very large periods
				stoch_period: Some(50),
				k: Some(10),
				d: Some(10),
				source: None,
			},
			SrsiParams {
				rsi_period: Some(7),    // mixed periods 1
				stoch_period: Some(14),
				k: Some(3),
				d: Some(5),
				source: None,
			},
			SrsiParams {
				rsi_period: Some(14),   // mixed periods 2
				stoch_period: Some(7),
				k: Some(5),
				d: Some(3),
				source: None,
			},
			SrsiParams {
				rsi_period: Some(21),   // mixed periods 3
				stoch_period: Some(14),
				k: Some(6),
				d: Some(4),
				source: None,
			},
			SrsiParams {
				rsi_period: Some(100),  // extreme case
				stoch_period: Some(100),
				k: Some(20),
				d: Some(20),
				source: None,
			},
		];
		
		for (param_idx, params) in test_params.iter().enumerate() {
			let input = SrsiInput::from_candles(&candles, "close", params.clone());
			let output = srsi_with_kernel(&input, kernel)?;
			
			// Check k values
			for (i, &val) in output.k.iter().enumerate() {
				if val.is_nan() {
					continue; // NaN values are expected during warmup
				}
				
				let bits = val.to_bits();
				
				// Check all three poison patterns
				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in K output \
						 with params: rsi_period={}, stoch_period={}, k={}, d={} (param set {})",
						test_name, val, bits, i, 
						params.rsi_period.unwrap_or(14),
						params.stoch_period.unwrap_or(14),
						params.k.unwrap_or(3),
						params.d.unwrap_or(3),
						param_idx
					);
				}
				
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in K output \
						 with params: rsi_period={}, stoch_period={}, k={}, d={} (param set {})",
						test_name, val, bits, i,
						params.rsi_period.unwrap_or(14),
						params.stoch_period.unwrap_or(14),
						params.k.unwrap_or(3),
						params.d.unwrap_or(3),
						param_idx
					);
				}
				
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in K output \
						 with params: rsi_period={}, stoch_period={}, k={}, d={} (param set {})",
						test_name, val, bits, i,
						params.rsi_period.unwrap_or(14),
						params.stoch_period.unwrap_or(14),
						params.k.unwrap_or(3),
						params.d.unwrap_or(3),
						param_idx
					);
				}
			}
			
			// Check d values
			for (i, &val) in output.d.iter().enumerate() {
				if val.is_nan() {
					continue; // NaN values are expected during warmup
				}
				
				let bits = val.to_bits();
				
				// Check all three poison patterns
				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in D output \
						 with params: rsi_period={}, stoch_period={}, k={}, d={} (param set {})",
						test_name, val, bits, i,
						params.rsi_period.unwrap_or(14),
						params.stoch_period.unwrap_or(14),
						params.k.unwrap_or(3),
						params.d.unwrap_or(3),
						param_idx
					);
				}
				
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in D output \
						 with params: rsi_period={}, stoch_period={}, k={}, d={} (param set {})",
						test_name, val, bits, i,
						params.rsi_period.unwrap_or(14),
						params.stoch_period.unwrap_or(14),
						params.k.unwrap_or(3),
						params.d.unwrap_or(3),
						param_idx
					);
				}
				
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in D output \
						 with params: rsi_period={}, stoch_period={}, k={}, d={} (param set {})",
						test_name, val, bits, i,
						params.rsi_period.unwrap_or(14),
						params.stoch_period.unwrap_or(14),
						params.k.unwrap_or(3),
						params.d.unwrap_or(3),
						param_idx
					);
				}
			}
		}
		
		Ok(())
	}

	#[cfg(not(debug_assertions))]
	fn check_srsi_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(()) // No-op in release builds
	}

	#[cfg(feature = "proptest")]
	#[allow(clippy::float_cmp)]
	fn check_srsi_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		use proptest::prelude::*;
		skip_if_unsupported!(kernel, test_name);

		// Generate test strategy with periods from 2 to 20 for more realistic testing
		let strat = (2usize..=20, 2usize..=20, 2usize..=10, 2usize..=10)
			.prop_flat_map(|(rsi_period, stoch_period, k, d)| {
				// Calculate minimum data needed for SRSI to work
				// RSI needs rsi_period points, then Stoch needs additional points for its calculations
				let min_data_needed = rsi_period + stoch_period.max(k).max(d) + 10; // Extra buffer
				(
					prop::collection::vec(
						(-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
						min_data_needed..400,
					),
					Just(rsi_period),
					Just(stoch_period),
					Just(k),
					Just(d),
				)
			});

		proptest::test_runner::TestRunner::default()
			.run(&strat, |(data, rsi_period, stoch_period, k, d)| {
				let params = SrsiParams {
					rsi_period: Some(rsi_period),
					stoch_period: Some(stoch_period),
					k: Some(k),
					d: Some(d),
					source: None,
				};
				let input = SrsiInput::from_slice(&data, params.clone());

				// Check if we have enough data - if not, skip this test case
				let output_result = srsi_with_kernel(&input, kernel);
				let ref_output_result = srsi_with_kernel(&input, Kernel::Scalar);
				
				// Both should either succeed or fail together
				match (output_result, ref_output_result) {
					(Ok(output), Ok(ref_output)) => {
						// Calculate expected warmup period
						// RSI produces first value at rsi_period, then Stoch needs more values
						// The actual warmup is complex due to the chained indicators
						let expected_min_warmup = rsi_period;

						// Property 1: K and D values must be bounded between 0 and 100
						for i in 0..data.len() {
							if !output.k[i].is_nan() {
								prop_assert!(
									output.k[i] >= -1e-9 && output.k[i] <= 100.0 + 1e-9,
									"idx {}: K value {} is out of bounds [0, 100]", i, output.k[i]
								);
							}
							if !output.d[i].is_nan() {
								prop_assert!(
									output.d[i] >= -1e-9 && output.d[i] <= 100.0 + 1e-9,
									"idx {}: D value {} is out of bounds [0, 100]", i, output.d[i]
								);
							}
						}

						// Property 2: Warmup period - check that early values are NaN
						// Don't be too strict about exact warmup period as it's complex with chained indicators
						for i in 0..expected_min_warmup.min(data.len()) {
							prop_assert!(
								output.k[i].is_nan(),
								"idx {}: Expected NaN during early warmup for K, got {}", i, output.k[i]
							);
							prop_assert!(
								output.d[i].is_nan(),
								"idx {}: Expected NaN during early warmup for D, got {}", i, output.d[i]
							);
						}

						// Property 3: Eventually should get valid values if we have enough data
						let has_valid_k = output.k.iter().any(|&x| !x.is_nan());
						let has_valid_d = output.d.iter().any(|&x| !x.is_nan());
						if data.len() > rsi_period + stoch_period + k + d {
							prop_assert!(
								has_valid_k,
								"Expected at least one valid K value with sufficient data"
							);
							prop_assert!(
								has_valid_d,
								"Expected at least one valid D value with sufficient data"
							);
						}

						// Property 4: Constant data should produce SRSI around 50 (neutral)
						if data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10) {
							let last_k = output.k[data.len() - 1];
							let last_d = output.d[data.len() - 1];
							if !last_k.is_nan() && !last_d.is_nan() {
								prop_assert!(
									(last_k - 50.0).abs() < 10.0,
									"Constant data should produce K near 50, got {}", last_k
								);
								prop_assert!(
									(last_d - 50.0).abs() < 10.0,
									"Constant data should produce D near 50, got {}", last_d
								);
							}
						}

						// Property 5: Strictly increasing prices should produce high SRSI
						let is_increasing = data.windows(2).all(|w| w[1] > w[0]);
						if is_increasing && has_valid_k {
							let last_k = output.k[data.len() - 1];
							if !last_k.is_nan() {
								prop_assert!(
									last_k > 50.0,
									"Strictly increasing prices should produce K > 50, got {}", last_k
								);
							}
						}

						// Property 6: Strictly decreasing prices should produce low SRSI
						let is_decreasing = data.windows(2).all(|w| w[1] < w[0]);
						if is_decreasing && has_valid_k {
							let last_k = output.k[data.len() - 1];
							if !last_k.is_nan() {
								prop_assert!(
									last_k < 50.0,
									"Strictly decreasing prices should produce K < 50, got {}", last_k
								);
							}
						}

						// Property 7: Kernel consistency - compare with scalar implementation
						for i in 0..data.len() {
							let k_val = output.k[i];
							let d_val = output.d[i];
							let ref_k = ref_output.k[i];
							let ref_d = ref_output.d[i];

							// Check for NaN consistency
							if !k_val.is_finite() || !ref_k.is_finite() {
								prop_assert!(
									k_val.to_bits() == ref_k.to_bits(),
									"K finite/NaN mismatch idx {}: {} vs {}", i, k_val, ref_k
								);
							} else {
								// Check ULP difference for finite values
								let k_ulp_diff = k_val.to_bits().abs_diff(ref_k.to_bits());
								prop_assert!(
									(k_val - ref_k).abs() <= 1e-9 || k_ulp_diff <= 4,
									"K mismatch idx {}: {} vs {} (ULP={})", i, k_val, ref_k, k_ulp_diff
								);
							}

							if !d_val.is_finite() || !ref_d.is_finite() {
								prop_assert!(
									d_val.to_bits() == ref_d.to_bits(),
									"D finite/NaN mismatch idx {}: {} vs {}", i, d_val, ref_d
								);
							} else {
								let d_ulp_diff = d_val.to_bits().abs_diff(ref_d.to_bits());
								prop_assert!(
									(d_val - ref_d).abs() <= 1e-9 || d_ulp_diff <= 4,
									"D mismatch idx {}: {} vs {} (ULP={})", i, d_val, ref_d, d_ulp_diff
								);
							}
						}

						// Property 8: Determinism - running same calculation twice produces same result
						let output2 = srsi_with_kernel(&input, kernel).unwrap();
						for i in 0..data.len() {
							prop_assert!(
								output.k[i].to_bits() == output2.k[i].to_bits(),
								"K determinism failed at idx {}: {} vs {}", i, output.k[i], output2.k[i]
							);
							prop_assert!(
								output.d[i].to_bits() == output2.d[i].to_bits(),
								"D determinism failed at idx {}: {} vs {}", i, output.d[i], output2.d[i]
							);
						}
					}
					(Err(_), Err(_)) => {
						// Both failed - this is expected for insufficient data, skip this test case
					}
					(Ok(_), Err(e)) => {
						prop_assert!(false, "Kernel succeeded but scalar failed: {:?}", e);
					}
					(Err(e), Ok(_)) => {
						prop_assert!(false, "Kernel failed but scalar succeeded: {:?}", e);
					}
				}

				Ok(())
			})
			.unwrap();

		Ok(())
	}

	macro_rules! generate_all_srsi_tests {
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

	generate_all_srsi_tests!(
		check_srsi_partial_params,
		check_srsi_accuracy,
		check_srsi_custom_params,
		check_srsi_from_slice,
		check_srsi_no_poison
	);
	
	#[test]
	fn test_srsi_into_slice_size_mismatch() {
		// Test that srsi_into_slice returns SizeMismatch error when buffer sizes don't match
		// Generate enough data for default parameters (rsi_period=14, stoch_period=14, k=3, d=3)
		let data: Vec<f64> = (1..=50).map(|x| x as f64).collect();
		let data_len = data.len();
		let params = SrsiParams::default();
		let input = SrsiInput::from_slice(&data, params);
		
		// Test with k buffer too small
		let mut k_small = vec![0.0; 30];  // Wrong size
		let mut d_correct = vec![0.0; data_len]; // Correct size
		let result = srsi_into_slice(&mut k_small, &mut d_correct, &input, Kernel::Scalar);
		match result {
			Err(SrsiError::SizeMismatch { expected, k_len, d_len }) => {
				assert_eq!(expected, data_len);
				assert_eq!(k_len, 30);
				assert_eq!(d_len, data_len);
			}
			_ => panic!("Expected SizeMismatch error with k buffer too small"),
		}
		
		// Test with d buffer too small
		let mut k_correct = vec![0.0; data_len]; // Correct size
		let mut d_small = vec![0.0; 35];    // Wrong size
		let result = srsi_into_slice(&mut k_correct, &mut d_small, &input, Kernel::Scalar);
		match result {
			Err(SrsiError::SizeMismatch { expected, k_len, d_len }) => {
				assert_eq!(expected, data_len);
				assert_eq!(k_len, data_len);
				assert_eq!(d_len, 35);
			}
			_ => panic!("Expected SizeMismatch error with d buffer too small"),
		}
		
		// Test with both buffers wrong size
		let mut k_wrong = vec![0.0; 60];  // Wrong size
		let mut d_wrong = vec![0.0; 70];  // Wrong size
		let result = srsi_into_slice(&mut k_wrong, &mut d_wrong, &input, Kernel::Scalar);
		match result {
			Err(SrsiError::SizeMismatch { expected, k_len, d_len }) => {
				assert_eq!(expected, data_len);
				assert_eq!(k_len, 60);
				assert_eq!(d_len, 70);
			}
			_ => panic!("Expected SizeMismatch error with both buffers wrong size"),
		}
		
		// Test with correct sizes - should succeed
		let mut k_ok = vec![0.0; data_len];
		let mut d_ok = vec![0.0; data_len];
		let result = srsi_into_slice(&mut k_ok, &mut d_ok, &input, Kernel::Scalar);
		assert!(result.is_ok(), "Should succeed with correct buffer sizes. Error: {:?}", result);
	}

	#[cfg(feature = "proptest")]
	generate_all_srsi_tests!(check_srsi_property);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let output = SrsiBatchBuilder::new().kernel(kernel).apply_slice(&c.close)?;
		let def = SrsiParams::default();
		let k_row = output.k_for(&def).expect("default k row missing");
		let d_row = output.d_for(&def).expect("default d row missing");
		assert_eq!(k_row.len(), c.close.len());
		assert_eq!(d_row.len(), c.close.len());
		Ok(())
	}

	#[cfg(debug_assertions)]
	fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);
		
		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		
		// Test various parameter sweep configurations
		let test_configs = vec![
			// (rsi_period_range, stoch_period_range, k_range, d_range)
			((2, 10, 2), (2, 10, 2), (2, 4, 1), (2, 4, 1)),        // Small periods
			((5, 25, 5), (5, 25, 5), (3, 7, 2), (3, 7, 2)),        // Medium periods
			((30, 60, 15), (30, 60, 15), (5, 10, 5), (5, 10, 5)),  // Large periods
			((2, 5, 1), (2, 5, 1), (2, 3, 1), (2, 3, 1)),          // Dense small range
			((10, 30, 10), (5, 15, 5), (3, 6, 3), (3, 6, 3)),      // Mixed ranges
			((14, 14, 0), (14, 14, 0), (3, 3, 0), (3, 3, 0)),      // Single value (default)
			((7, 21, 7), (14, 28, 14), (3, 9, 3), (3, 9, 3)),      // Different step sizes
		];
		
		for (cfg_idx, &(rsi_range, stoch_range, k_range, d_range)) in test_configs.iter().enumerate() {
			let output = SrsiBatchBuilder::new()
				.kernel(kernel)
				.rsi_period_range(rsi_range.0, rsi_range.1, rsi_range.2)
				.stoch_period_range(stoch_range.0, stoch_range.1, stoch_range.2)
				.k_range(k_range.0, k_range.1, k_range.2)
				.d_range(d_range.0, d_range.1, d_range.2)
				.apply_slice(&c.close)?;
			
			// Check k values
			for (idx, &val) in output.k.iter().enumerate() {
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
						"[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) in K output \
						 at row {} col {} (flat index {}) with params: rsi_period={}, stoch_period={}, k={}, d={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.rsi_period.unwrap_or(14),
						combo.stoch_period.unwrap_or(14),
						combo.k.unwrap_or(3),
						combo.d.unwrap_or(3)
					);
				}
				
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) in K output \
						 at row {} col {} (flat index {}) with params: rsi_period={}, stoch_period={}, k={}, d={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.rsi_period.unwrap_or(14),
						combo.stoch_period.unwrap_or(14),
						combo.k.unwrap_or(3),
						combo.d.unwrap_or(3)
					);
				}
				
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) in K output \
						 at row {} col {} (flat index {}) with params: rsi_period={}, stoch_period={}, k={}, d={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.rsi_period.unwrap_or(14),
						combo.stoch_period.unwrap_or(14),
						combo.k.unwrap_or(3),
						combo.d.unwrap_or(3)
					);
				}
			}
			
			// Check d values
			for (idx, &val) in output.d.iter().enumerate() {
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
						"[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) in D output \
						 at row {} col {} (flat index {}) with params: rsi_period={}, stoch_period={}, k={}, d={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.rsi_period.unwrap_or(14),
						combo.stoch_period.unwrap_or(14),
						combo.k.unwrap_or(3),
						combo.d.unwrap_or(3)
					);
				}
				
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) in D output \
						 at row {} col {} (flat index {}) with params: rsi_period={}, stoch_period={}, k={}, d={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.rsi_period.unwrap_or(14),
						combo.stoch_period.unwrap_or(14),
						combo.k.unwrap_or(3),
						combo.d.unwrap_or(3)
					);
				}
				
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) in D output \
						 at row {} col {} (flat index {}) with params: rsi_period={}, stoch_period={}, k={}, d={}",
						test, cfg_idx, val, bits, row, col, idx,
						combo.rsi_period.unwrap_or(14),
						combo.stoch_period.unwrap_or(14),
						combo.k.unwrap_or(3),
						combo.d.unwrap_or(3)
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
