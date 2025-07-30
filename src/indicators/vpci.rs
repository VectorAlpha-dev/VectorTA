//! # Volume Price Confirmation Index (VPCI)
//!
//! VPCI confirms price movements using volume-weighted moving averages (VWMAs), comparing
//! price and volume trends to detect confluence/divergence. It supports SIMD kernels and
//! batch grid evaluation for hyperparameter sweeps.
//!
//! ## Parameters
//! - **short_range**: Window size for short-term averages (default: 5).
//! - **long_range**: Window size for long-term averages (default: 25).
//!
//! ## Errors
//! - **VpciError::AllValuesNaN**: All close or volume values are NaN.
//! - **VpciError::InvalidRange**: A range (period) is zero or exceeds data length.
//! - **VpciError::NotEnoughValidData**: Not enough valid data for a range.
//! - **VpciError::SmaError**: Underlying SMA error.
//!
//! ## Returns
//! - **Ok(VpciOutput)** on success (`vpci`, `vpcis` of same length as input).
//! - **Err(VpciError)** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
	alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel,
	init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::mem::{ManuallyDrop, MaybeUninit};
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
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
use thiserror::Error;

use crate::indicators::sma::{sma, SmaData, SmaError, SmaInput, SmaParams};

#[derive(Debug, Clone)]
pub enum VpciData<'a> {
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
pub struct VpciOutput {
	pub vpci: Vec<f64>,
	pub vpcis: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct VpciParams {
	pub short_range: Option<usize>,
	pub long_range: Option<usize>,
}

impl Default for VpciParams {
	fn default() -> Self {
		Self {
			short_range: Some(5),
			long_range: Some(25),
		}
	}
}

#[derive(Debug, Clone)]
pub struct VpciInput<'a> {
	pub data: VpciData<'a>,
	pub params: VpciParams,
}

impl<'a> VpciInput<'a> {
	#[inline]
	pub fn from_candles(
		candles: &'a Candles,
		close_source: &'a str,
		volume_source: &'a str,
		params: VpciParams,
	) -> Self {
		Self {
			data: VpciData::Candles {
				candles,
				close_source,
				volume_source,
			},
			params,
		}
	}

	#[inline]
	pub fn from_slices(close: &'a [f64], volume: &'a [f64], params: VpciParams) -> Self {
		Self {
			data: VpciData::Slices { close, volume },
			params,
		}
	}

	#[inline]
	pub fn with_default_candles(candles: &'a Candles) -> Self {
		Self {
			data: VpciData::Candles {
				candles,
				close_source: "close",
				volume_source: "volume",
			},
			params: VpciParams::default(),
		}
	}

	#[inline]
	pub fn get_short_range(&self) -> usize {
		self.params.short_range.unwrap_or(5)
	}
	#[inline]
	pub fn get_long_range(&self) -> usize {
		self.params.long_range.unwrap_or(25)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct VpciBuilder {
	short_range: Option<usize>,
	long_range: Option<usize>,
	kernel: Kernel,
}

impl Default for VpciBuilder {
	fn default() -> Self {
		Self {
			short_range: None,
			long_range: None,
			kernel: Kernel::Auto,
		}
	}
}

impl VpciBuilder {
	#[inline(always)]
	pub fn new() -> Self {
		Self::default()
	}
	#[inline(always)]
	pub fn short_range(mut self, n: usize) -> Self {
		self.short_range = Some(n);
		self
	}
	#[inline(always)]
	pub fn long_range(mut self, n: usize) -> Self {
		self.long_range = Some(n);
		self
	}
	#[inline(always)]
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline(always)]
	pub fn apply(self, c: &Candles) -> Result<VpciOutput, VpciError> {
		let p = VpciParams {
			short_range: self.short_range,
			long_range: self.long_range,
		};
		let i = VpciInput::from_candles(c, "close", "volume", p);
		vpci_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slices(self, close: &[f64], volume: &[f64]) -> Result<VpciOutput, VpciError> {
		let p = VpciParams {
			short_range: self.short_range,
			long_range: self.long_range,
		};
		let i = VpciInput::from_slices(close, volume, p);
		vpci_with_kernel(&i, self.kernel)
	}
}

#[derive(Debug, Error)]
pub enum VpciError {
	#[error("vpci: All close or volume values are NaN.")]
	AllValuesNaN,

	#[error("vpci: Invalid range: period = {period}, data length = {data_len}")]
	InvalidRange { period: usize, data_len: usize },

	#[error("vpci: Not enough valid data: needed = {needed}, valid = {valid}")]
	NotEnoughValidData { needed: usize, valid: usize },

	#[error("vpci: SMA error: {0}")]
	SmaError(#[from] SmaError),
	
	#[error("vpci: Mismatched input lengths: close = {close_len}, volume = {volume_len}")]
	MismatchedInputLengths { close_len: usize, volume_len: usize },
	
	#[error("vpci: Kernel not available")]
	KernelNotAvailable,
}

#[inline]
pub fn vpci(input: &VpciInput) -> Result<VpciOutput, VpciError> {
	vpci_with_kernel(input, Kernel::Auto)
}

pub fn vpci_with_kernel(input: &VpciInput, kernel: Kernel) -> Result<VpciOutput, VpciError> {
	let (close, volume): (&[f64], &[f64]) = match &input.data {
		VpciData::Candles {
			candles,
			close_source,
			volume_source,
		} => (source_type(candles, close_source), source_type(candles, volume_source)),
		VpciData::Slices { close, volume } => (*close, *volume),
	};

	let len = close.len();
	let first = close
		.iter()
		.zip(volume.iter())
		.position(|(c, v)| !c.is_nan() && !v.is_nan())
		.ok_or(VpciError::AllValuesNaN)?;

	let short_range = input.get_short_range();
	let long_range = input.get_long_range();

	if short_range == 0 || long_range == 0 || short_range > len || long_range > len {
		return Err(VpciError::InvalidRange {
			period: short_range.max(long_range),
			data_len: len,
		});
	}
	if (len - first) < long_range {
		return Err(VpciError::NotEnoughValidData {
			needed: long_range,
			valid: len - first,
		});
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		k => k,
	};

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => vpci_scalar(close, volume, short_range, long_range),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => vpci_avx2(close, volume, short_range, long_range),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => vpci_avx512(close, volume, short_range, long_range),
			_ => unreachable!(),
		}
	}
}

#[inline]
pub unsafe fn vpci_scalar(close: &[f64], volume: &[f64], short: usize, long: usize) -> Result<VpciOutput, VpciError> {
	let len = close.len();
	
	// Find first valid index
	let first = close
		.iter()
		.zip(volume.iter())
		.position(|(c, v)| !c.is_nan() && !v.is_nan())
		.unwrap_or(0);
	
	// Allocate output arrays with proper NaN prefix
	let warmup = first + long - 1;
	let mut vpci = alloc_with_nan_prefix(len, warmup);
	let mut vpcis = alloc_with_nan_prefix(len, warmup);
	
	// Allocate arrays for VWMAs (volume-weighted moving averages)
	let mut vwma_long = alloc_with_nan_prefix(len, warmup);
	let mut vwma_short = alloc_with_nan_prefix(len, warmup);
	
	// Calculate VWMAs manually to avoid intermediate allocation
	// First, calculate VWMA long
	if warmup < len {
		let mut sum_cv = 0.0;
		let mut sum_v = 0.0;
		
		// Initialize window for long VWMA
		for i in 0..long {
			let idx = first + i;
			if idx < len {
				sum_cv += close[idx] * volume[idx];
				sum_v += volume[idx];
			}
		}
		
		let first_idx = first + long - 1;
		if sum_v != 0.0 {
			vwma_long[first_idx] = sum_cv / sum_v;
		}
		
		// Sliding window for long VWMA
		for i in (first_idx + 1)..len {
			sum_cv += close[i] * volume[i];
			sum_v += volume[i];
			let old_idx = i - long;
			sum_cv -= close[old_idx] * volume[old_idx];
			sum_v -= volume[old_idx];
			if sum_v != 0.0 {
				vwma_long[i] = sum_cv / sum_v;
			}
		}
	}
	
	// Calculate VWMA short
	if first + short - 1 < len {
		let mut sum_cv = 0.0;
		let mut sum_v = 0.0;
		
		// Initialize window for short VWMA
		for i in 0..short {
			let idx = first + i;
			if idx < len {
				sum_cv += close[idx] * volume[idx];
				sum_v += volume[idx];
			}
		}
		
		let first_idx = first + short - 1;
		if sum_v != 0.0 {
			vwma_short[first_idx] = sum_cv / sum_v;
		}
		
		// Sliding window for short VWMA
		for i in (first_idx + 1)..len {
			sum_cv += close[i] * volume[i];
			sum_v += volume[i];
			let old_idx = i - short;
			sum_cv -= close[old_idx] * volume[old_idx];
			sum_v -= volume[old_idx];
			if sum_v != 0.0 {
				vwma_short[i] = sum_cv / sum_v;
			}
		}
	}

	// Calculate all SMAs (these already use uninitialized memory internally)
	let sma_close_long = sma(&SmaInput {
		data: SmaData::Slice(close),
		params: SmaParams { period: Some(long) },
	})?
	.values;
	let sma_close_short = sma(&SmaInput {
		data: SmaData::Slice(close),
		params: SmaParams { period: Some(short) },
	})?
	.values;
	let sma_volume_long = sma(&SmaInput {
		data: SmaData::Slice(volume),
		params: SmaParams { period: Some(long) },
	})?
	.values;
	let sma_volume_short = sma(&SmaInput {
		data: SmaData::Slice(volume),
		params: SmaParams { period: Some(short) },
	})?
	.values;

	// Calculate VPCI values only after warmup period
	for i in warmup..len {
		// Calculate VPCI components
		let vpc = vwma_long[i] - sma_close_long[i];
		let vpr = if !sma_close_short[i].is_nan() && sma_close_short[i] != 0.0 {
			vwma_short[i] / sma_close_short[i]
		} else {
			f64::NAN
		};
		let vm = if !sma_volume_long[i].is_nan() && sma_volume_long[i] != 0.0 {
			sma_volume_short[i] / sma_volume_long[i]
		} else {
			f64::NAN
		};
		
		vpci[i] = vpc * vpr * vm;
	}
	
	// Calculate VPCIS (smoothed VPCI) using sliding window to avoid allocation
	// VPCIS = SMA(VPCI * Volume, short) / SMA(Volume, short)
	if warmup < len {
		let mut sum_vpci_vol = 0.0;
		
		// Initialize window for VPCIS calculation
		for i in 0..short {
			let idx = warmup + i - short + 1;
			if idx >= warmup && idx < len {
				if !vpci[idx].is_nan() && !volume[idx].is_nan() {
					sum_vpci_vol += vpci[idx] * volume[idx];
				}
			}
		}
		
		// Calculate first VPCIS value
		if !sma_volume_short[warmup].is_nan() && sma_volume_short[warmup] != 0.0 {
			vpcis[warmup] = sum_vpci_vol / short as f64 / sma_volume_short[warmup];
		}
		
		// Sliding window for VPCIS
		for i in (warmup + 1)..len {
			// Add new value
			if !vpci[i].is_nan() && !volume[i].is_nan() {
				sum_vpci_vol += vpci[i] * volume[i];
			}
			
			// Remove old value
			let old_idx = i - short;
			if old_idx >= warmup && !vpci[old_idx].is_nan() && !volume[old_idx].is_nan() {
				sum_vpci_vol -= vpci[old_idx] * volume[old_idx];
			}
			
			// Calculate VPCIS
			if !sma_volume_short[i].is_nan() && sma_volume_short[i] != 0.0 {
				vpcis[i] = sum_vpci_vol / short as f64 / sma_volume_short[i];
			}
		}
	}
	
	Ok(VpciOutput { vpci, vpcis })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vpci_avx2(close: &[f64], volume: &[f64], short: usize, long: usize) -> Result<VpciOutput, VpciError> {
	vpci_scalar(close, volume, short, long)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vpci_avx512(close: &[f64], volume: &[f64], short: usize, long: usize) -> Result<VpciOutput, VpciError> {
	vpci_scalar(close, volume, short, long)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vpci_avx512_short(
	close: &[f64],
	volume: &[f64],
	short: usize,
	long: usize,
) -> Result<VpciOutput, VpciError> {
	vpci_avx512(close, volume, short, long)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vpci_avx512_long(
	close: &[f64],
	volume: &[f64],
	short: usize,
	long: usize,
) -> Result<VpciOutput, VpciError> {
	vpci_avx512(close, volume, short, long)
}

#[inline]
pub fn vpci_batch_with_kernel(
	close: &[f64],
	volume: &[f64],
	sweep: &VpciBatchRange,
	kernel: Kernel,
) -> Result<VpciBatchOutput, VpciError> {
	let k = match kernel {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => {
			return Err(VpciError::InvalidRange {
				period: 0,
				data_len: close.len(),
			})
		}
	};
	let simd = match k {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	vpci_batch_par_slice(close, volume, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct VpciBatchRange {
	pub short_range: (usize, usize, usize),
	pub long_range: (usize, usize, usize),
}

impl Default for VpciBatchRange {
	fn default() -> Self {
		Self {
			short_range: (5, 20, 1),
			long_range: (25, 60, 5),
		}
	}
}

#[derive(Clone, Debug, Default)]
pub struct VpciBatchBuilder {
	range: VpciBatchRange,
	kernel: Kernel,
}

impl VpciBatchBuilder {
	pub fn new() -> Self {
		Self::default()
	}
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	pub fn short_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.short_range = (start, end, step);
		self
	}
	pub fn long_range(mut self, start: usize, end: usize, step: usize) -> Self {
		self.range.long_range = (start, end, step);
		self
	}
	pub fn apply_slices(self, close: &[f64], volume: &[f64]) -> Result<VpciBatchOutput, VpciError> {
		vpci_batch_with_kernel(close, volume, &self.range, self.kernel)
	}
}

#[derive(Clone, Debug)]
pub struct VpciBatchOutput {
	pub vpci: Vec<f64>,
	pub vpcis: Vec<f64>,
	pub combos: Vec<VpciParams>,
	pub rows: usize,
	pub cols: usize,
}
impl VpciBatchOutput {
	pub fn row_for_params(&self, p: &VpciParams) -> Option<usize> {
		self.combos.iter().position(|c| {
			c.short_range.unwrap_or(5) == p.short_range.unwrap_or(5)
				&& c.long_range.unwrap_or(25) == p.long_range.unwrap_or(25)
		})
	}
	pub fn vpci_for(&self, p: &VpciParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.vpci[start..start + self.cols]
		})
	}
	pub fn vpcis_for(&self, p: &VpciParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.vpcis[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &VpciBatchRange) -> Vec<VpciParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}
	let shorts = axis_usize(r.short_range);
	let longs = axis_usize(r.long_range);

	let mut out = Vec::with_capacity(shorts.len() * longs.len());
	for &s in &shorts {
		for &l in &longs {
			out.push(VpciParams {
				short_range: Some(s),
				long_range: Some(l),
			});
		}
	}
	out
}

#[inline(always)]
pub fn vpci_batch_slice(
	close: &[f64],
	volume: &[f64],
	sweep: &VpciBatchRange,
	kernel: Kernel,
) -> Result<VpciBatchOutput, VpciError> {
	vpci_batch_inner(close, volume, sweep, kernel, false)
}

#[inline(always)]
pub fn vpci_batch_par_slice(
	close: &[f64],
	volume: &[f64],
	sweep: &VpciBatchRange,
	kernel: Kernel,
) -> Result<VpciBatchOutput, VpciError> {
	vpci_batch_inner(close, volume, sweep, kernel, true)
}

#[inline(always)]
fn vpci_batch_inner(
	close: &[f64],
	volume: &[f64],
	sweep: &VpciBatchRange,
	kernel: Kernel,
	parallel: bool,
) -> Result<VpciBatchOutput, VpciError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(VpciError::InvalidRange {
			period: 0,
			data_len: close.len(),
		});
	}

	let len = close.len();
	let first = close
		.iter()
		.zip(volume.iter())
		.position(|(c, v)| !c.is_nan() && !v.is_nan())
		.ok_or(VpciError::AllValuesNaN)?;
	let max_long = combos.iter().map(|c| c.long_range.unwrap()).max().unwrap();
	if len - first < max_long {
		return Err(VpciError::NotEnoughValidData {
			needed: max_long,
			valid: len - first,
		});
	}

	let rows = combos.len();
	let cols = len;
	
	// Create uninitialized matrices for output
	let mut vpci_mu = make_uninit_matrix(rows, cols);
	let mut vpcis_mu = make_uninit_matrix(rows, cols);
	
	// Calculate warmup periods for each row
	let warmup_periods: Vec<usize> = combos
		.iter()
		.map(|c| first + c.long_range.unwrap() - 1)
		.collect();
	
	// Initialize NaN prefixes
	init_matrix_prefixes(&mut vpci_mu, cols, &warmup_periods);
	init_matrix_prefixes(&mut vpcis_mu, cols, &warmup_periods);
	
	// Convert to mutable slices for computation
	let mut vpci_guard = ManuallyDrop::new(vpci_mu);
	let mut vpcis_guard = ManuallyDrop::new(vpcis_mu);
	let vpci: &mut [f64] = unsafe { 
		std::slice::from_raw_parts_mut(vpci_guard.as_mut_ptr() as *mut f64, rows * cols) 
	};
	let vpcis: &mut [f64] = unsafe { 
		std::slice::from_raw_parts_mut(vpcis_guard.as_mut_ptr() as *mut f64, rows * cols) 
	};

	let do_row = |row: usize, vpci_out: &mut [f64], vpcis_out: &mut [f64]| unsafe {
		let prm = &combos[row];
		let VpciOutput { vpci, vpcis } = match kernel {
			Kernel::Scalar => vpci_row_scalar(close, volume, prm.short_range.unwrap(), prm.long_range.unwrap()),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => vpci_row_avx2(close, volume, prm.short_range.unwrap(), prm.long_range.unwrap()),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => vpci_row_avx512(close, volume, prm.short_range.unwrap(), prm.long_range.unwrap()),
			_ => unreachable!(),
		}?;
		vpci_out.copy_from_slice(&vpci);
		vpcis_out.copy_from_slice(&vpcis);
		Ok::<(), VpciError>(())
	};

	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			vpci.par_chunks_mut(cols)
				.zip(vpcis.par_chunks_mut(cols))
				.enumerate()
				.for_each(|(row, (v, vs))| {
					let _ = do_row(row, v, vs);
				});
		}
		#[cfg(target_arch = "wasm32")]
		{
			for (row, (v, vs)) in vpci.chunks_mut(cols).zip(vpcis.chunks_mut(cols)).enumerate() {
				let _ = do_row(row, v, vs);
			}
		}
	} else {
		for (row, (v, vs)) in vpci.chunks_mut(cols).zip(vpcis.chunks_mut(cols)).enumerate() {
			let _ = do_row(row, v, vs);
		}
	}

	// Convert back to Vec
	let vpci_vec = unsafe {
		let ptr = vpci_guard.as_mut_ptr() as *mut f64;
		let len = rows * cols;
		let cap = vpci_guard.capacity();
		std::mem::forget(vpci_guard);
		Vec::from_raw_parts(ptr, len, cap)
	};
	
	let vpcis_vec = unsafe {
		let ptr = vpcis_guard.as_mut_ptr() as *mut f64;
		let len = rows * cols;
		let cap = vpcis_guard.capacity();
		std::mem::forget(vpcis_guard);
		Vec::from_raw_parts(ptr, len, cap)
	};
	
	Ok(VpciBatchOutput {
		vpci: vpci_vec,
		vpcis: vpcis_vec,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
pub unsafe fn vpci_row_scalar(
	close: &[f64],
	volume: &[f64],
	short: usize,
	long: usize,
) -> Result<VpciOutput, VpciError> {
	vpci_scalar(close, volume, short, long)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vpci_row_avx2(close: &[f64], volume: &[f64], short: usize, long: usize) -> Result<VpciOutput, VpciError> {
	vpci_avx2(close, volume, short, long)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vpci_row_avx512(
	close: &[f64],
	volume: &[f64],
	short: usize,
	long: usize,
) -> Result<VpciOutput, VpciError> {
	if long <= 32 {
		vpci_row_avx512_short(close, volume, short, long)
	} else {
		vpci_row_avx512_long(close, volume, short, long)
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vpci_row_avx512_short(
	close: &[f64],
	volume: &[f64],
	short: usize,
	long: usize,
) -> Result<VpciOutput, VpciError> {
	vpci_avx512(close, volume, short, long)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vpci_row_avx512_long(
	close: &[f64],
	volume: &[f64],
	short: usize,
	long: usize,
) -> Result<VpciOutput, VpciError> {
	vpci_avx512(close, volume, short, long)
}

#[inline(always)]
pub fn expand_grid_vpci(r: &VpciBatchRange) -> Vec<VpciParams> {
	expand_grid(r)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_vpci_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = VpciParams {
			short_range: Some(3),
			long_range: None,
		};
		let input = VpciInput::from_candles(&candles, "close", "volume", params);
		let output = vpci_with_kernel(&input, kernel)?;
		assert_eq!(output.vpci.len(), candles.close.len());
		assert_eq!(output.vpcis.len(), candles.close.len());
		Ok(())
	}

	fn check_vpci_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let params = VpciParams {
			short_range: Some(5),
			long_range: Some(25),
		};
		let input = VpciInput::from_candles(&candles, "close", "volume", params);
		let output = vpci_with_kernel(&input, kernel)?;

		let vpci_len = output.vpci.len();
		let vpcis_len = output.vpcis.len();
		assert_eq!(vpci_len, candles.close.len());
		assert_eq!(vpcis_len, candles.close.len());

		let vpci_last_five = &output.vpci[vpci_len.saturating_sub(5)..];
		let vpcis_last_five = &output.vpcis[vpcis_len.saturating_sub(5)..];
		let expected_vpci = [
			-319.65148214323426,
			-133.61700649928346,
			-144.76194155503174,
			-83.55576212490328,
			-169.53504207700533,
		];
		let expected_vpcis = [
			-1049.2826640115732,
			-694.1067814399748,
			-519.6960416662324,
			-330.9401404636258,
			-173.004986803695,
		];
		for (i, &val) in vpci_last_five.iter().enumerate() {
			let diff = (val - expected_vpci[i]).abs();
			assert!(
				diff < 1e-1,
				"[{}] VPCI mismatch at idx {}: got {}, expected {}",
				test_name,
				i,
				val,
				expected_vpci[i]
			);
		}
		for (i, &val) in vpcis_last_five.iter().enumerate() {
			let diff = (val - expected_vpcis[i]).abs();
			assert!(
				diff < 1e-1,
				"[{}] VPCIS mismatch at idx {}: got {}, expected {}",
				test_name,
				i,
				val,
				expected_vpcis[i]
			);
		}
		Ok(())
	}

	fn check_vpci_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = VpciInput::with_default_candles(&candles);
		let output = vpci_with_kernel(&input, kernel)?;
		assert_eq!(output.vpci.len(), candles.close.len());
		assert_eq!(output.vpcis.len(), candles.close.len());
		Ok(())
	}

	fn check_vpci_slice_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let close_data = [10.0, 12.0, 14.0, 13.0, 15.0];
		let volume_data = [100.0, 200.0, 300.0, 250.0, 400.0];
		let params = VpciParams {
			short_range: Some(2),
			long_range: Some(3),
		};
		let input = VpciInput::from_slices(&close_data, &volume_data, params);
		let output = vpci_with_kernel(&input, kernel)?;
		assert_eq!(output.vpci.len(), close_data.len());
		assert_eq!(output.vpcis.len(), close_data.len());
		Ok(())
	}

	macro_rules! generate_all_vpci_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $( #[test] fn [<$test_fn _scalar_f64>]() { let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar); } )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(
                    #[test] fn [<$test_fn _avx2_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2); }
                    #[test] fn [<$test_fn _avx512_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512); }
                )*
            }
        }
    }

	generate_all_vpci_tests!(
		check_vpci_partial_params,
		check_vpci_accuracy,
		check_vpci_default_candles,
		check_vpci_slice_input
	);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;
		let close = &c.close;
		let volume = &c.volume;

		let output = VpciBatchBuilder::new().kernel(kernel).apply_slices(close, volume)?;

		let def = VpciParams::default();
		let row = output.vpci_for(&def).expect("default row missing");

		assert_eq!(row.len(), close.len());

		let expected = [
			-319.65148214323426,
			-133.61700649928346,
			-144.76194155503174,
			-83.55576212490328,
			-169.53504207700533,
		];
		let start = row.len() - 5;
		for (i, &v) in row[start..].iter().enumerate() {
			assert!(
				(v - expected[i]).abs() < 1e-1,
				"[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
			);
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

#[cfg(feature = "python")]
#[pyfunction(name = "vpci")]
#[pyo3(signature = (close, volume, short_range, long_range, kernel=None))]
pub fn vpci_py<'py>(
	py: Python<'py>,
	close: numpy::PyReadonlyArray1<'py, f64>,
	volume: numpy::PyReadonlyArray1<'py, f64>,
	short_range: usize,
	long_range: usize,
	kernel: Option<&str>,
) -> PyResult<(Bound<'py, numpy::PyArray1<f64>>, Bound<'py, numpy::PyArray1<f64>>)> {
	use numpy::{IntoPyArray, PyArrayMethods};

	let close_slice = close.as_slice()?;
	let volume_slice = volume.as_slice()?;
	
	if close_slice.len() != volume_slice.len() {
		return Err(PyValueError::new_err("Close and volume arrays must have the same length"));
	}
	
	let kern = validate_kernel(kernel, false)?;
	let params = VpciParams {
		short_range: Some(short_range),
		long_range: Some(long_range),
	};
	let input = VpciInput::from_slices(close_slice, volume_slice, params);

	let (vpci_vec, vpcis_vec) = py
		.allow_threads(|| vpci_with_kernel(&input, kern).map(|o| (o.vpci, o.vpcis)))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	Ok((vpci_vec.into_pyarray(py), vpcis_vec.into_pyarray(py)))
}

#[cfg(feature = "python")]
#[pyfunction(name = "vpci_batch")]
#[pyo3(signature = (close, volume, short_range_tuple, long_range_tuple, kernel=None))]
pub fn vpci_batch_py<'py>(
	py: Python<'py>,
	close: numpy::PyReadonlyArray1<'py, f64>,
	volume: numpy::PyReadonlyArray1<'py, f64>,
	short_range_tuple: (usize, usize, usize),
	long_range_tuple: (usize, usize, usize),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;

	let close_slice = close.as_slice()?;
	let volume_slice = volume.as_slice()?;
	
	if close_slice.len() != volume_slice.len() {
		return Err(PyValueError::new_err("Close and volume arrays must have the same length"));
	}

	let sweep = VpciBatchRange {
		short_range: short_range_tuple,
		long_range: long_range_tuple,
	};

	let combos = expand_grid(&sweep);
	let rows = combos.len();
	let cols = close_slice.len();

	// Pre-allocate output arrays for batch operations
	let vpci_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let vpcis_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let vpci_slice = unsafe { vpci_arr.as_slice_mut()? };
	let vpcis_slice = unsafe { vpcis_arr.as_slice_mut()? };

	let kern = validate_kernel(kernel, true)?;

	let combos = py
		.allow_threads(|| {
			let kernel = match kern {
				Kernel::Auto => detect_best_batch_kernel(),
				k => k,
			};
			
			// Map batch kernels to regular kernels for computation
			let simd = match kernel {
				Kernel::Avx512Batch => Kernel::Avx512,
				Kernel::Avx2Batch => Kernel::Avx2,
				Kernel::ScalarBatch => Kernel::Scalar,
				_ => kernel,
			};
			
			// Compute batch results
			let result = vpci_batch_inner(close_slice, volume_slice, &sweep, simd, true)?;
			
			// Copy results to pre-allocated arrays
			vpci_slice.copy_from_slice(&result.vpci);
			vpcis_slice.copy_from_slice(&result.vpcis);
			
			Ok::<Vec<VpciParams>, VpciError>(result.combos)
		})
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	let dict = PyDict::new(py);
	dict.set_item("vpci", vpci_arr.reshape((rows, cols))?)?;
	dict.set_item("vpcis", vpcis_arr.reshape((rows, cols))?)?;
	dict.set_item(
		"short_ranges",
		combos
			.iter()
			.map(|p| p.short_range.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;
	dict.set_item(
		"long_ranges",
		combos
			.iter()
			.map(|p| p.long_range.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;

	Ok(dict)
}

/// WASM helper: Write directly to output slices - no allocations
#[cfg(feature = "wasm")]
pub fn vpci_into_slice(
	vpci_dst: &mut [f64],
	vpcis_dst: &mut [f64],
	input: &VpciInput,
	kern: Kernel,
) -> Result<(), VpciError> {
	let kernel = match kern {
		Kernel::Auto => detect_best_kernel(),
		k => k,
	};
	
	let (close, volume) = match &input.data {
		VpciData::Candles { candles, close_source, volume_source } => {
			(source_type(candles, close_source), source_type(candles, volume_source))
		}
		VpciData::Slices { close, volume } => (*close, *volume),
	};
	
	let len = close.len();
	if len != volume.len() {
		return Err(VpciError::MismatchedInputLengths {
			close_len: len,
			volume_len: volume.len(),
		});
	}
	
	if vpci_dst.len() != len || vpcis_dst.len() != len {
		return Err(VpciError::InvalidRange {
			period: len,
			data_len: vpci_dst.len(),
		});
	}
	
	let short_range = input.get_short_range();
	let long_range = input.get_long_range();
	
	// Find first valid index and calculate warmup
	let first = close.iter()
		.zip(volume.iter())
		.position(|(c, v)| !c.is_nan() && !v.is_nan())
		.ok_or(VpciError::AllValuesNaN)?;
	
	let warmup = first + long_range - 1;
	
	// Compute VPCI values
	let output = unsafe {
		match kernel {
			Kernel::Scalar => vpci_scalar(close, volume, short_range, long_range)?,
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 => vpci_avx2(close, volume, short_range, long_range)?,
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 => vpci_avx512(close, volume, short_range, long_range)?,
			_ => return Err(VpciError::KernelNotAvailable),
		}
	};
	
	// Copy results to output slices
	vpci_dst.copy_from_slice(&output.vpci);
	vpcis_dst.copy_from_slice(&output.vpcis);
	
	// Fill warmup with NaN
	for v in &mut vpci_dst[..warmup] {
		*v = f64::NAN;
	}
	for v in &mut vpcis_dst[..warmup] {
		*v = f64::NAN;
	}
	
	Ok(())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vpci_js(
	close: &[f64], 
	volume: &[f64], 
	short_range: usize, 
	long_range: usize
) -> Result<Vec<f64>, JsValue> {
	if close.len() != volume.len() {
		return Err(JsValue::from_str(&format!(
			"Mismatched input lengths: close = {}, volume = {}", 
			close.len(), 
			volume.len()
		)));
	}
	
	let params = VpciParams {
		short_range: Some(short_range),
		long_range: Some(long_range),
	};
	let input = VpciInput::from_slices(close, volume, params);
	
	// Pre-allocate single output vector for flattened results
	let len = close.len();
	let mut output = vec![0.0; len * 2];
	
	// Split the output into vpci and vpcis slices
	let (vpci_slice, vpcis_slice) = output.split_at_mut(len);
	
	// Use zero-copy helper
	vpci_into_slice(vpci_slice, vpcis_slice, &input, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vpci_into(
	close_ptr: *const f64,
	volume_ptr: *const f64,
	vpci_ptr: *mut f64,
	vpcis_ptr: *mut f64,
	len: usize,
	short_range: usize,
	long_range: usize,
) -> Result<(), JsValue> {
	if close_ptr.is_null() || volume_ptr.is_null() || vpci_ptr.is_null() || vpcis_ptr.is_null() {
		return Err(JsValue::from_str("null pointer passed to vpci_into"));
	}
	
	unsafe {
		let close = std::slice::from_raw_parts(close_ptr, len);
		let volume = std::slice::from_raw_parts(volume_ptr, len);
		
		let params = VpciParams {
			short_range: Some(short_range),
			long_range: Some(long_range),
		};
		let input = VpciInput::from_slices(close, volume, params);
		
		// Check aliasing for all pointer combinations
		let need_temp = close_ptr == vpci_ptr as *const f64 || 
		                close_ptr == vpcis_ptr as *const f64 ||
		                volume_ptr == vpci_ptr as *const f64 || 
		                volume_ptr == vpcis_ptr as *const f64 ||
		                vpci_ptr == vpcis_ptr;
		
		if need_temp {
			// Use temporary buffers if any aliasing detected
			let mut temp_vpci = vec![0.0; len];
			let mut temp_vpcis = vec![0.0; len];
			
			vpci_into_slice(&mut temp_vpci, &mut temp_vpcis, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			
			let vpci_out = std::slice::from_raw_parts_mut(vpci_ptr, len);
			let vpcis_out = std::slice::from_raw_parts_mut(vpcis_ptr, len);
			vpci_out.copy_from_slice(&temp_vpci);
			vpcis_out.copy_from_slice(&temp_vpcis);
		} else {
			// Direct computation when no aliasing
			let vpci_out = std::slice::from_raw_parts_mut(vpci_ptr, len);
			let vpcis_out = std::slice::from_raw_parts_mut(vpcis_ptr, len);
			
			vpci_into_slice(vpci_out, vpcis_out, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		
		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vpci_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vpci_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe { let _ = Vec::from_raw_parts(ptr, len, len); }
	}
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct VpciBatchConfig {
	pub short_range: (usize, usize, usize),
	pub long_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct VpciBatchJsOutput {
	pub vpci: Vec<f64>,
	pub vpcis: Vec<f64>,
	pub combos: Vec<VpciParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = vpci_batch)]
pub fn vpci_batch_js(close: &[f64], volume: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	if close.len() != volume.len() {
		return Err(JsValue::from_str(&format!(
			"Mismatched input lengths: close = {}, volume = {}", 
			close.len(), 
			volume.len()
		)));
	}
	
	let config: VpciBatchConfig =
		serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
	
	let sweep = VpciBatchRange {
		short_range: config.short_range,
		long_range: config.long_range,
	};
	
	let output = vpci_batch_inner(close, volume, &sweep, detect_best_kernel(), false)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	let js_output = VpciBatchJsOutput {
		vpci: output.vpci,
		vpcis: output.vpcis,
		combos: output.combos,
		rows: output.rows,
		cols: output.cols,
	};
	
	serde_wasm_bindgen::to_value(&js_output).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vpci_batch_into(
	close_ptr: *const f64,
	volume_ptr: *const f64,
	vpci_ptr: *mut f64,
	vpcis_ptr: *mut f64,
	len: usize,
	short_start: usize,
	short_end: usize,
	short_step: usize,
	long_start: usize,
	long_end: usize,
	long_step: usize,
) -> Result<usize, JsValue> {
	if close_ptr.is_null() || volume_ptr.is_null() || vpci_ptr.is_null() || vpcis_ptr.is_null() {
		return Err(JsValue::from_str("null pointer passed to vpci_batch_into"));
	}
	
	unsafe {
		let close = std::slice::from_raw_parts(close_ptr, len);
		let volume = std::slice::from_raw_parts(volume_ptr, len);
		
		let sweep = VpciBatchRange {
			short_range: (short_start, short_end, short_step),
			long_range: (long_start, long_end, long_step),
		};
		
		let combos = expand_grid_vpci(&sweep);
		let rows = combos.len();
		let total_len = rows * len;
		
		// Need to handle aliasing only between outputs and inputs
		let need_temp = close_ptr == vpci_ptr as *const f64 || 
		                close_ptr == vpcis_ptr as *const f64 ||
		                volume_ptr == vpci_ptr as *const f64 || 
		                volume_ptr == vpcis_ptr as *const f64;
		
		if need_temp {
			// Run batch into temporary buffers
			let output = vpci_batch_inner(close, volume, &sweep, detect_best_kernel(), false)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			
			// Copy to output pointers
			let vpci_out = std::slice::from_raw_parts_mut(vpci_ptr, total_len);
			let vpcis_out = std::slice::from_raw_parts_mut(vpcis_ptr, total_len);
			vpci_out.copy_from_slice(&output.vpci);
			vpcis_out.copy_from_slice(&output.vpcis);
		} else {
			// Direct computation
			let vpci_out = std::slice::from_raw_parts_mut(vpci_ptr, total_len);
			let vpcis_out = std::slice::from_raw_parts_mut(vpcis_ptr, total_len);
			
			// Call batch_inner_into if it exists, otherwise compute and copy
			let output = vpci_batch_inner(close, volume, &sweep, detect_best_kernel(), false)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			
			vpci_out.copy_from_slice(&output.vpci);
			vpcis_out.copy_from_slice(&output.vpcis);
		}
		
		Ok(rows)
	}
}
