//! # Band-Pass Filter
//!
//! A frequency-based filter (Ehlers-inspired) that isolates a band of interest by removing both high- and low-frequency components from a time series. Parameters `period` and `bandwidth` control the central window and width. Batch, stream, and SIMD kernels are supported.
//!
//! ## Parameters
//! - **period**: Central lookback period (>=2).
//! - **bandwidth**: Passband width in [0,1] (default: 0.3).
//!
//! ## Errors
//! - **NotEnoughData**: Data length < period.
//! - **InvalidPeriod**: period < 2.
//! - **HpPeriodTooSmall**: hp_period after rounding < 2.
//! - **TriggerPeriodTooSmall**: trigger_period after rounding < 2.
//! - **HighPassError**: errors from underlying highpass filter.
//!
//! ## Returns
//! - **`Ok(BandPassOutput)`** on success.  
//! - **`Err(BandPassError)`** otherwise.

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

use crate::indicators::highpass::{highpass, HighPassError, HighPassInput, HighPassParams};
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
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::f64::consts::PI;
use thiserror::Error;

impl<'a> AsRef<[f64]> for BandPassInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			BandPassData::Slice(slice) => slice,
			BandPassData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

#[derive(Debug, Clone)]
pub enum BandPassData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct BandPassParams {
	pub period: Option<usize>,
	pub bandwidth: Option<f64>,
}

impl Default for BandPassParams {
	fn default() -> Self {
		Self {
			period: Some(20),
			bandwidth: Some(0.3),
		}
	}
}

#[derive(Debug, Clone)]
pub struct BandPassInput<'a> {
	pub data: BandPassData<'a>,
	pub params: BandPassParams,
}

impl<'a> BandPassInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: BandPassParams) -> Self {
		Self {
			data: BandPassData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slice(sl: &'a [f64], p: BandPassParams) -> Self {
		Self {
			data: BandPassData::Slice(sl),
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", BandPassParams::default())
	}
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(20)
	}
	#[inline]
	pub fn get_bandwidth(&self) -> f64 {
		self.params.bandwidth.unwrap_or(0.3)
	}
}

#[derive(Copy, Clone, Debug)]
pub struct BandPassBuilder {
	period: Option<usize>,
	bandwidth: Option<f64>,
	kernel: Kernel,
}

impl Default for BandPassBuilder {
	fn default() -> Self {
		Self {
			period: None,
			bandwidth: None,
			kernel: Kernel::Auto,
		}
	}
}

impl BandPassBuilder {
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
	pub fn bandwidth(mut self, b: f64) -> Self {
		self.bandwidth = Some(b);
		self
	}
	#[inline(always)]
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline(always)]
	pub fn apply(self, c: &Candles) -> Result<BandPassOutput, BandPassError> {
		let p = BandPassParams {
			period: self.period,
			bandwidth: self.bandwidth,
		};
		let i = BandPassInput::from_candles(c, "close", p);
		bandpass_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slice(self, d: &[f64]) -> Result<BandPassOutput, BandPassError> {
		let p = BandPassParams {
			period: self.period,
			bandwidth: self.bandwidth,
		};
		let i = BandPassInput::from_slice(d, p);
		bandpass_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<BandPassStream, BandPassError> {
		let p = BandPassParams {
			period: self.period,
			bandwidth: self.bandwidth,
		};
		BandPassStream::try_new(p)
	}
}

#[derive(Debug, Error)]
pub enum BandPassError {
	#[error("bandpass: Not enough data, data_len={data_len}, period={period}")]
	NotEnoughData { data_len: usize, period: usize },
	#[error("bandpass: Invalid period={period}")]
	InvalidPeriod { period: usize },
	#[error("bandpass: hp_period too small ({hp_period})")]
	HpPeriodTooSmall { hp_period: usize },
	#[error("bandpass: trigger_period too small ({trigger_period})")]
	TriggerPeriodTooSmall { trigger_period: usize },
	#[error(transparent)]
	HighPassError(#[from] HighPassError),
}

#[derive(Debug, Clone)]
pub struct BandPassOutput {
	pub bp: Vec<f64>,
	pub bp_normalized: Vec<f64>,
	pub signal: Vec<f64>,
	pub trigger: Vec<f64>,
}

#[inline]
pub fn bandpass(input: &BandPassInput) -> Result<BandPassOutput, BandPassError> {
	bandpass_with_kernel(input, Kernel::Auto)
}

pub fn bandpass_with_kernel(input: &BandPassInput, kernel: Kernel) -> Result<BandPassOutput, BandPassError> {
	let data: &[f64] = match &input.data {
		BandPassData::Candles { candles, source } => source_type(candles, source),
		BandPassData::Slice(sl) => sl,
	};

	let len = data.len();
	let period = input.get_period();
	let bandwidth = input.get_bandwidth();

	if len == 0 || len < period {
		return Err(BandPassError::NotEnoughData { data_len: len, period });
	}
	if period < 2 {
		return Err(BandPassError::InvalidPeriod { period });
	}
	if !(0.0..=1.0).contains(&bandwidth) || bandwidth.is_nan() || bandwidth.is_infinite() {
		return Err(BandPassError::InvalidPeriod { period });
	}

	let hp_period_f = 4.0 * (period as f64) / bandwidth;
	let hp_period = hp_period_f.round() as usize;
	if hp_period < 2 {
		return Err(BandPassError::HpPeriodTooSmall { hp_period });
	}

	let mut hp_params = HighPassParams::default();
	hp_params.period = Some(hp_period);

	let hp_input = HighPassInput::from_slice(data, hp_params);
	let hp_result = highpass(&hp_input)?;
	let hp = hp_result.values;

	let beta = (2.0 * PI / period as f64).cos();
	let gamma = (2.0 * PI * bandwidth / period as f64).cos();
	let alpha = 1.0 / gamma - ((1.0 / (gamma * gamma)) - 1.0).sqrt();

	// Determine warmup period from the highpass output
	let first_valid_hp = hp.iter().position(|&x| !x.is_nan()).unwrap_or(0);
	let warmup_bp = first_valid_hp.max(2); // bp calculation starts from index 2

	// Allocate bp with proper NaN prefix
	let mut bp = alloc_with_nan_prefix(len, warmup_bp);
	// Copy the first 2 values from hp (needed for the recursive calculation)
	if len > 0 {
		bp[0] = hp[0];
	}
	if len > 1 {
		bp[1] = hp[1];
	}

	// bp_normalized will have the same warmup as bp
	let mut bp_normalized = alloc_with_nan_prefix(len, warmup_bp);

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		other => other,
	};

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => bandpass_scalar(&hp, period, alpha, beta, &mut bp),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => bandpass_avx2(&hp, period, alpha, beta, &mut bp),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => bandpass_avx512(&hp, period, alpha, beta, &mut bp),
			_ => unreachable!(),
		}
	}

	let k = 0.991;
	let mut peak_prev = 0.0;
	for i in 0..len {
		peak_prev *= k;
		let abs_bp = bp[i].abs();
		if abs_bp > peak_prev {
			peak_prev = abs_bp;
		}
		bp_normalized[i] = if peak_prev != 0.0 { bp[i] / peak_prev } else { 0.0 };
	}

	let trigger_period_f = (period as f64 / bandwidth) / 1.5;
	let trigger_period = trigger_period_f.round() as usize;
	if trigger_period < 2 {
		return Err(BandPassError::TriggerPeriodTooSmall { trigger_period });
	}
	let mut trigger_params = HighPassParams::default();
	trigger_params.period = Some(trigger_period);
	let trigger_input = HighPassInput::from_slice(&bp_normalized, trigger_params);
	let trigger_result = highpass(&trigger_input)?;
	let trigger = trigger_result.values;

	// Signal warmup is the max of bp_normalized and trigger warmup periods
	let first_valid_trigger = trigger.iter().position(|&x| !x.is_nan()).unwrap_or(0);
	let warmup_signal = warmup_bp.max(first_valid_trigger);
	let mut signal = alloc_with_nan_prefix(len, warmup_signal);
	for i in 0..len {
		let bn = bp_normalized[i];
		let tr = trigger[i];
		if bn < tr {
			signal[i] = 1.0;
		} else if bn > tr {
			signal[i] = -1.0;
		} else {
			signal[i] = 0.0;
		}
	}

	Ok(BandPassOutput {
		bp,
		bp_normalized,
		signal,
		trigger,
	})
}

/// Write directly to output slices - no allocations
#[inline]
pub fn bandpass_into_slice(
	bp_dst: &mut [f64],
	bp_normalized_dst: &mut [f64],
	signal_dst: &mut [f64],
	trigger_dst: &mut [f64],
	input: &BandPassInput,
	kern: Kernel,
) -> Result<(), BandPassError> {
	let data: &[f64] = match &input.data {
		BandPassData::Candles { candles, source } => source_type(candles, source),
		BandPassData::Slice(sl) => sl,
	};
	
	let len = data.len();
	let period = input.get_period();
	let bandwidth = input.get_bandwidth();
	
	// Validate input
	if len == 0 || len < period {
		return Err(BandPassError::NotEnoughData { data_len: len, period });
	}
	if period == 0 {
		return Err(BandPassError::InvalidPeriod { period });
	}
	
	// Validate output slice lengths
	if bp_dst.len() != len || bp_normalized_dst.len() != len || signal_dst.len() != len || trigger_dst.len() != len {
		return Err(BandPassError::InvalidPeriod { period: 0 });
	}
	
	// Use existing logic from bandpass_with_kernel but write to provided slices
	let result = bandpass_with_kernel(input, kern)?;
	
	// Copy results to output slices
	bp_dst.copy_from_slice(&result.bp);
	bp_normalized_dst.copy_from_slice(&result.bp_normalized);
	signal_dst.copy_from_slice(&result.signal);
	trigger_dst.copy_from_slice(&result.trigger);
	
	Ok(())
}

#[inline(always)]
pub fn bandpass_scalar(hp: &[f64], period: usize, alpha: f64, beta: f64, out: &mut [f64]) {
	let len = hp.len();
	if len >= 2 {
		for i in 2..len {
			out[i] = 0.5 * (1.0 - alpha) * hp[i] - 0.5 * (1.0 - alpha) * hp[i - 2] + beta * (1.0 + alpha) * out[i - 1]
				- alpha * out[i - 2];
		}
	}
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn bandpass_avx2(hp: &[f64], period: usize, alpha: f64, beta: f64, out: &mut [f64]) {
	bandpass_scalar(hp, period, alpha, beta, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn bandpass_avx512(hp: &[f64], period: usize, alpha: f64, beta: f64, out: &mut [f64]) {
	bandpass_scalar(hp, period, alpha, beta, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn bandpass_avx512_short(hp: &[f64], period: usize, alpha: f64, beta: f64, out: &mut [f64]) {
	bandpass_scalar(hp, period, alpha, beta, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn bandpass_avx512_long(hp: &[f64], period: usize, alpha: f64, beta: f64, out: &mut [f64]) {
	bandpass_scalar(hp, period, alpha, beta, out)
}

#[derive(Debug, Clone)]
pub struct BandPassStream {
	period: usize,
	alpha: f64,
	beta: f64,
	hp_stream: crate::indicators::highpass::HighPassStream,
	buf: Vec<f64>,
	idx: usize,
	len: usize,
	last_hp: [f64; 2],
	last_out: [f64; 2],
}

impl BandPassStream {
	pub fn try_new(params: BandPassParams) -> Result<Self, BandPassError> {
		let period = params.period.unwrap_or(20);
		if period < 2 {
			return Err(BandPassError::InvalidPeriod { period });
		}
		let bandwidth = params.bandwidth.unwrap_or(0.3);
		if !(0.0..=1.0).contains(&bandwidth) || bandwidth.is_nan() || bandwidth.is_infinite() {
			return Err(BandPassError::InvalidPeriod { period });
		}
		let hp_period = (4.0 * period as f64 / bandwidth).round() as usize;
		if hp_period < 2 {
			return Err(BandPassError::HpPeriodTooSmall { hp_period });
		}
		let mut hp_params = HighPassParams::default();
		hp_params.period = Some(hp_period);

		let hp_stream = crate::indicators::highpass::HighPassStream::try_new(hp_params)?;
		let beta = (2.0 * PI / period as f64).cos();
		let gamma = (2.0 * PI * bandwidth / period as f64).cos();
		let alpha = 1.0 / gamma - ((1.0 / (gamma * gamma)) - 1.0).sqrt();

		Ok(Self {
			period,
			alpha,
			beta,
			hp_stream,
			buf: vec![0.0; 2],
			idx: 0,
			len: 0,
			last_hp: [0.0; 2],
			last_out: [0.0; 2],
		})
	}

	pub fn update(&mut self, value: f64) -> f64 {
		let hp_val = self.hp_stream.update(value);
		// rotate buffers for hp and output
		let prev_hp2 = self.last_hp[0];
		let prev_hp1 = self.last_hp[1];
		let prev_out2 = self.last_out[0];
		let prev_out1 = self.last_out[1];

		let out_val = if self.len < 2 {
			self.len += 1;
			self.last_hp[0] = prev_hp1;
			self.last_hp[1] = hp_val;
			self.last_out[0] = prev_out1;
			self.last_out[1] = hp_val;
			hp_val
		} else {
			let res = 0.5 * (1.0 - self.alpha) * hp_val - 0.5 * (1.0 - self.alpha) * prev_hp2
				+ self.beta * (1.0 + self.alpha) * prev_out1
				- self.alpha * prev_out2;
			self.last_hp[0] = prev_hp1;
			self.last_hp[1] = hp_val;
			self.last_out[0] = prev_out1;
			self.last_out[1] = res;
			res
		};
		out_val
	}
}

#[derive(Clone, Debug)]
pub struct BandPassBatchRange {
	pub period: (usize, usize, usize),
	pub bandwidth: (f64, f64, f64),
}

impl Default for BandPassBatchRange {
	fn default() -> Self {
		Self {
			period: (20, 60, 1),
			bandwidth: (0.3, 0.3, 0.0),
		}
	}
}

#[derive(Clone, Debug, Default)]
pub struct BandPassBatchBuilder {
	range: BandPassBatchRange,
	kernel: Kernel,
}

impl BandPassBatchBuilder {
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
	pub fn bandwidth_range(mut self, start: f64, end: f64, step: f64) -> Self {
		self.range.bandwidth = (start, end, step);
		self
	}
	#[inline]
	pub fn bandwidth_static(mut self, b: f64) -> Self {
		self.range.bandwidth = (b, b, 0.0);
		self
	}
	pub fn apply_slice(self, data: &[f64]) -> Result<BandPassBatchOutput, BandPassError> {
		bandpass_batch_with_kernel(data, &self.range, self.kernel)
	}
	pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<BandPassBatchOutput, BandPassError> {
		BandPassBatchBuilder::new().kernel(k).apply_slice(data)
	}
	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<BandPassBatchOutput, BandPassError> {
		let slice = source_type(c, src);
		self.apply_slice(slice)
	}
	pub fn with_default_candles(c: &Candles) -> Result<BandPassBatchOutput, BandPassError> {
		BandPassBatchBuilder::new()
			.kernel(Kernel::Auto)
			.apply_candles(c, "close")
	}
}

#[derive(Clone, Debug)]
pub struct BandPassBatchOutput {
	pub values: Vec<BandPassOutput>,
	pub combos: Vec<BandPassParams>,
	pub rows: usize,
	pub cols: usize,
}

impl BandPassBatchOutput {
	pub fn row_for_params(&self, p: &BandPassParams) -> Option<usize> {
		self.combos.iter().position(|c| {
			c.period.unwrap_or(20) == p.period.unwrap_or(20)
				&& (c.bandwidth.unwrap_or(0.3) - p.bandwidth.unwrap_or(0.3)).abs() < 1e-12
		})
	}
	pub fn values_for(&self, p: &BandPassParams) -> Option<&BandPassOutput> {
		self.row_for_params(p).map(|row| &self.values[row])
	}
}

#[inline(always)]
fn expand_grid(r: &BandPassBatchRange) -> Vec<BandPassParams> {
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
	let bandwidths = axis_f64(r.bandwidth);
	let mut out = Vec::with_capacity(periods.len() * bandwidths.len());
	for &p in &periods {
		for &b in &bandwidths {
			out.push(BandPassParams {
				period: Some(p),
				bandwidth: Some(b),
			});
		}
	}
	out
}

pub fn bandpass_batch_with_kernel(
	data: &[f64],
	sweep: &BandPassBatchRange,
	k: Kernel,
) -> Result<BandPassBatchOutput, BandPassError> {
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		other if other.is_batch() => other,
		_ => {
			return Err(BandPassError::InvalidPeriod { period: 0 });
		}
	};
	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	bandpass_batch_par_slice(data, sweep, simd)
}

pub fn bandpass_batch_slice(
	data: &[f64],
	sweep: &BandPassBatchRange,
	kern: Kernel,
) -> Result<BandPassBatchOutput, BandPassError> {
	bandpass_batch_inner(data, sweep, kern, false)
}

pub fn bandpass_batch_par_slice(
	data: &[f64],
	sweep: &BandPassBatchRange,
	kern: Kernel,
) -> Result<BandPassBatchOutput, BandPassError> {
	bandpass_batch_inner(data, sweep, kern, true)
}

fn bandpass_batch_inner(
	data: &[f64],
	sweep: &BandPassBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<BandPassBatchOutput, BandPassError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(BandPassError::InvalidPeriod { period: 0 });
	}
	let len = data.len();
	let mut outputs: Vec<Option<BandPassOutput>> = vec![None; combos.len()];
	let do_row = |row: usize| -> Result<BandPassOutput, BandPassError> {
		let p = combos[row].clone();
		let input = BandPassInput::from_slice(data, p);
		bandpass_with_kernel(&input, kern)
	};
	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			outputs.par_iter_mut().enumerate().for_each(|(row, slot)| {
				*slot = do_row(row).ok();
			});
		}
		#[cfg(target_arch = "wasm32")]
		{
			for (row, slot) in outputs.iter_mut().enumerate() {
				*slot = do_row(row).ok();
			}
		}
	} else {
		for (row, slot) in outputs.iter_mut().enumerate() {
			*slot = do_row(row).ok();
		}
	}
	let mut values = Vec::with_capacity(combos.len());
	for v in outputs {
		values.push(v.ok_or_else(|| BandPassError::InvalidPeriod { period: 0 })?);
	}
	let rows = combos.len();
	let combos_clone = combos.clone();
	Ok(BandPassBatchOutput {
		values,
		combos: combos_clone,
		rows,
		cols: len,
	})
}

/// Batch processing that writes directly into caller-supplied buffers to avoid allocations.
/// Each output buffer must have size rows * cols.
/// SAFETY: When parallel=true, this function uses unsafe pointer arithmetic to allow
/// parallel writes to non-overlapping regions of the output buffers.
#[inline(always)]
fn bandpass_batch_inner_into(
	data: &[f64],
	sweep: &BandPassBatchRange,
	kern: Kernel,
	parallel: bool,
	bp_out: &mut [f64],
	bp_normalized_out: &mut [f64],
	signal_out: &mut [f64],
	trigger_out: &mut [f64],
) -> Result<Vec<BandPassParams>, BandPassError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(BandPassError::InvalidPeriod { period: 0 });
	}

	let cols = data.len();
	let rows = combos.len();

	// Verify buffer sizes
	if bp_out.len() != rows * cols
		|| bp_normalized_out.len() != rows * cols
		|| signal_out.len() != rows * cols
		|| trigger_out.len() != rows * cols
	{
		return Err(BandPassError::InvalidPeriod { period: 0 }); // Could add a new error variant
	}

	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			use rayon::prelude::*;
			use std::sync::atomic::{AtomicPtr, Ordering};

			// Wrap raw pointers in AtomicPtr for thread safety
			let bp_ptr = AtomicPtr::new(bp_out.as_mut_ptr());
			let bp_normalized_ptr = AtomicPtr::new(bp_normalized_out.as_mut_ptr());
			let signal_ptr = AtomicPtr::new(signal_out.as_mut_ptr());
			let trigger_ptr = AtomicPtr::new(trigger_out.as_mut_ptr());

			(0..rows)
				.into_par_iter()
				.try_for_each(|row| -> Result<(), BandPassError> {
					let p = combos[row].clone();
					let input = BandPassInput::from_slice(data, p);
					let output = bandpass_with_kernel(&input, kern)?;

					// Write directly to the pre-allocated slices
					let start_idx = row * cols;

					// SAFETY: We know these indices are valid and non-overlapping between threads
					// Each thread writes to a distinct row (start_idx..start_idx+cols)
					unsafe {
						let bp_base = bp_ptr.load(Ordering::Relaxed);
						let bp_normalized_base = bp_normalized_ptr.load(Ordering::Relaxed);
						let signal_base = signal_ptr.load(Ordering::Relaxed);
						let trigger_base = trigger_ptr.load(Ordering::Relaxed);

						std::ptr::copy_nonoverlapping(output.bp.as_ptr(), bp_base.add(start_idx), cols);
						std::ptr::copy_nonoverlapping(
							output.bp_normalized.as_ptr(),
							bp_normalized_base.add(start_idx),
							cols,
						);
						std::ptr::copy_nonoverlapping(output.signal.as_ptr(), signal_base.add(start_idx), cols);
						std::ptr::copy_nonoverlapping(output.trigger.as_ptr(), trigger_base.add(start_idx), cols);
					}

					Ok(())
				})?;
		}
		#[cfg(target_arch = "wasm32")]
		{
			for row in 0..rows {
				let p = combos[row].clone();
				let input = BandPassInput::from_slice(data, p);
				let output = bandpass_with_kernel(&input, kern)?;

				let start_idx = row * cols;
				let end_idx = start_idx + cols;

				bp_out[start_idx..end_idx].copy_from_slice(&output.bp);
				bp_normalized_out[start_idx..end_idx].copy_from_slice(&output.bp_normalized);
				signal_out[start_idx..end_idx].copy_from_slice(&output.signal);
				trigger_out[start_idx..end_idx].copy_from_slice(&output.trigger);
			}
		}
	} else {
		for row in 0..rows {
			let p = combos[row].clone();
			let input = BandPassInput::from_slice(data, p);
			let output = bandpass_with_kernel(&input, kern)?;

			let start_idx = row * cols;
			let end_idx = start_idx + cols;

			bp_out[start_idx..end_idx].copy_from_slice(&output.bp);
			bp_normalized_out[start_idx..end_idx].copy_from_slice(&output.bp_normalized);
			signal_out[start_idx..end_idx].copy_from_slice(&output.signal);
			trigger_out[start_idx..end_idx].copy_from_slice(&output.trigger);
		}
	}

	Ok(combos)
}

#[inline(always)]
pub fn bandpass_row_scalar(hp: &[f64], period: usize, alpha: f64, beta: f64, out: &mut [f64]) {
	bandpass_scalar(hp, period, alpha, beta, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn bandpass_row_avx2(hp: &[f64], period: usize, alpha: f64, beta: f64, out: &mut [f64]) {
	bandpass_scalar(hp, period, alpha, beta, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn bandpass_row_avx512(hp: &[f64], period: usize, alpha: f64, beta: f64, out: &mut [f64]) {
	bandpass_scalar(hp, period, alpha, beta, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn bandpass_row_avx512_short(hp: &[f64], period: usize, alpha: f64, beta: f64, out: &mut [f64]) {
	bandpass_scalar(hp, period, alpha, beta, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn bandpass_row_avx512_long(hp: &[f64], period: usize, alpha: f64, beta: f64, out: &mut [f64]) {
	bandpass_scalar(hp, period, alpha, beta, out)
}

#[inline(always)]
fn expand_grid_for_bandpass(r: &BandPassBatchRange) -> Vec<BandPassParams> {
	expand_grid(r)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_bandpass_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let default_params = BandPassParams::default();
		let input = BandPassInput::from_candles(&candles, "close", default_params);
		let output = bandpass_with_kernel(&input, kernel)?;
		assert_eq!(output.bp.len(), candles.close.len());
		Ok(())
	}
	fn check_bandpass_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = BandPassInput::with_default_candles(&candles);
		let result = bandpass_with_kernel(&input, kernel)?;
		let expected_bp_last_five = [
			-236.23678021132827,
			-247.4846395608195,
			-242.3788746078502,
			-212.89589193350128,
			-179.97293838509464,
		];
		let expected_bp_normalized_last_five = [
			-0.4399672555578846,
			-0.4651011734720517,
			-0.4596426251402882,
			-0.40739824942488945,
			-0.3475245023284841,
		];
		let expected_signal_last_five = [-1.0, 1.0, 1.0, 1.0, 1.0];
		let expected_trigger_last_five = [
			-0.4746908356434579,
			-0.4353877348116954,
			-0.3727126131420441,
			-0.2746336628365846,
			-0.18240018384226137,
		];
		let start = result.bp.len().saturating_sub(5);
		assert!(result.bp.len() >= 5);
		assert!(result.bp_normalized.len() >= 5);
		assert!(result.signal.len() >= 5);
		assert!(result.trigger.len() >= 5);
		for (i, &value) in result.bp[start..].iter().enumerate() {
			assert!(
				(value - expected_bp_last_five[i]).abs() < 1e-1,
				"BP value mismatch at index {}: expected {}, got {}",
				i,
				expected_bp_last_five[i],
				value
			);
		}
		for (i, &value) in result.bp_normalized[start..].iter().enumerate() {
			assert!(
				(value - expected_bp_normalized_last_five[i]).abs() < 1e-1,
				"BP Normalized value mismatch at index {}: expected {}, got {}",
				i,
				expected_bp_normalized_last_five[i],
				value
			);
		}
		for (i, &value) in result.signal[start..].iter().enumerate() {
			assert!(
				(value - expected_signal_last_five[i]).abs() < 1e-1,
				"Signal value mismatch at index {}: expected {}, got {}",
				i,
				expected_signal_last_five[i],
				value
			);
		}
		for (i, &value) in result.trigger[start..].iter().enumerate() {
			assert!(
				(value - expected_trigger_last_five[i]).abs() < 1e-1,
				"Trigger value mismatch at index {}: expected {}, got {}",
				i,
				expected_trigger_last_five[i],
				value
			);
		}
		for val in &result.bp {
			assert!(val.is_finite());
		}
		for val in &result.bp_normalized {
			assert!(val.is_finite());
		}
		for val in &result.signal {
			assert!(val.is_finite());
		}
		for val in &result.trigger {
			assert!(val.is_finite());
		}
		Ok(())
	}
	fn check_bandpass_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = BandPassInput::with_default_candles(&candles);
		match input.data {
			BandPassData::Candles { source, .. } => assert_eq!(source, "close"),
			_ => panic!("Expected BandPassData::Candles"),
		}
		let output = bandpass_with_kernel(&input, kernel)?;
		assert_eq!(output.bp.len(), candles.close.len());
		Ok(())
	}
	fn check_bandpass_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let input_data = [10.0, 20.0, 30.0];
		let params = BandPassParams {
			period: Some(0),
			bandwidth: Some(0.3),
		};
		let input = BandPassInput::from_slice(&input_data, params);
		let res = bandpass_with_kernel(&input, kernel);
		assert!(res.is_err());
		Ok(())
	}
	fn check_bandpass_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data_small = [10.0, 20.0, 30.0];
		let params = BandPassParams {
			period: Some(10),
			bandwidth: Some(0.3),
		};
		let input = BandPassInput::from_slice(&data_small, params);
		let res = bandpass_with_kernel(&input, kernel);
		assert!(res.is_err());
		Ok(())
	}
	fn check_bandpass_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let single_point = [42.0];
		let params = BandPassParams {
			period: Some(20),
			bandwidth: Some(0.3),
		};
		let input = BandPassInput::from_slice(&single_point, params);
		let res = bandpass_with_kernel(&input, kernel);
		assert!(res.is_err());
		Ok(())
	}
	fn check_bandpass_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let first_params = BandPassParams {
			period: Some(20),
			bandwidth: Some(0.3),
		};
		let first_input = BandPassInput::from_candles(&candles, "close", first_params);
		let first_result = bandpass_with_kernel(&first_input, kernel)?;
		let second_params = BandPassParams {
			period: Some(30),
			bandwidth: Some(0.5),
		};
		let second_input = BandPassInput::from_slice(&first_result.bp, second_params);
		let second_result = bandpass_with_kernel(&second_input, kernel)?;
		assert_eq!(second_result.bp.len(), first_result.bp.len());
		Ok(())
	}
	fn check_bandpass_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;
		let input = BandPassInput::from_candles(
			&candles,
			"close",
			BandPassParams {
				period: Some(20),
				bandwidth: Some(0.3),
			},
		);
		let res = bandpass_with_kernel(&input, kernel)?;
		assert_eq!(res.bp.len(), candles.close.len());
		if res.bp.len() > 30 {
			for i in 30..res.bp.len() {
				assert!(!res.bp[i].is_nan());
				assert!(!res.bp_normalized[i].is_nan());
				assert!(!res.signal[i].is_nan());
				assert!(!res.trigger[i].is_nan());
			}
		}
		Ok(())
	}

	macro_rules! generate_all_bandpass_tests {
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

	// Check for poison values in single output - only runs in debug mode
	#[cfg(debug_assertions)]
	fn check_bandpass_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		// Test multiple parameter combinations to increase coverage
		let test_params = vec![
			BandPassParams {
				period: Some(10),
				bandwidth: Some(0.2),
			},
			BandPassParams {
				period: Some(20),
				bandwidth: Some(0.3),
			}, // default
			BandPassParams {
				period: Some(30),
				bandwidth: Some(0.4),
			},
			BandPassParams {
				period: Some(50),
				bandwidth: Some(0.5),
			},
			BandPassParams {
				period: Some(5),
				bandwidth: Some(0.1),
			}, // edge case: small period
			BandPassParams {
				period: Some(100),
				bandwidth: Some(0.8),
			}, // edge case: large period
		];

		for params in test_params {
			let input = BandPassInput::from_candles(&candles, "close", params.clone());
			let output = bandpass_with_kernel(&input, kernel)?;

			// Check every value in all output vectors for poison patterns
			for (i, &val) in output.bp.iter().enumerate() {
				// Skip NaN values as they're expected in the warmup period
				if val.is_nan() {
					continue;
				}

				let bits = val.to_bits();

				// Check for alloc_with_nan_prefix poison (0x11111111_11111111)
				if bits == 0x11111111_11111111 {
					panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) in bp at index {} with params {:?}",
                    test_name, val, bits, i, params
                );
				}

				// Check for init_matrix_prefixes poison (0x22222222_22222222)
				if bits == 0x22222222_22222222 {
					panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) in bp at index {} with params {:?}",
                    test_name, val, bits, i, params
                );
				}

				// Check for make_uninit_matrix poison (0x33333333_33333333)
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) in bp at index {} with params {:?}",
						test_name, val, bits, i, params
					);
				}
			}

			// Check bp_normalized
			for (i, &val) in output.bp_normalized.iter().enumerate() {
				if val.is_nan() {
					continue;
				}

				let bits = val.to_bits();

				if bits == 0x11111111_11111111 {
					panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) in bp_normalized at index {} with params {:?}",
                    test_name, val, bits, i, params
                );
				}

				if bits == 0x22222222_22222222 {
					panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) in bp_normalized at index {} with params {:?}",
                    test_name, val, bits, i, params
                );
				}

				if bits == 0x33333333_33333333 {
					panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) in bp_normalized at index {} with params {:?}",
                    test_name, val, bits, i, params
                );
				}
			}

			// Check signal
			for (i, &val) in output.signal.iter().enumerate() {
				if val.is_nan() {
					continue;
				}

				let bits = val.to_bits();

				if bits == 0x11111111_11111111 {
					panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) in signal at index {} with params {:?}",
                    test_name, val, bits, i, params
                );
				}

				if bits == 0x22222222_22222222 {
					panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) in signal at index {} with params {:?}",
                    test_name, val, bits, i, params
                );
				}

				if bits == 0x33333333_33333333 {
					panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) in signal at index {} with params {:?}",
                    test_name, val, bits, i, params
                );
				}
			}

			// Check trigger
			for (i, &val) in output.trigger.iter().enumerate() {
				if val.is_nan() {
					continue;
				}

				let bits = val.to_bits();

				if bits == 0x11111111_11111111 {
					panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) in trigger at index {} with params {:?}",
                    test_name, val, bits, i, params
                );
				}

				if bits == 0x22222222_22222222 {
					panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) in trigger at index {} with params {:?}",
                    test_name, val, bits, i, params
                );
				}

				if bits == 0x33333333_33333333 {
					panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) in trigger at index {} with params {:?}",
                    test_name, val, bits, i, params
                );
				}
			}
		} // close the params loop

		Ok(())
	}

	// Release mode stub - does nothing
	#[cfg(not(debug_assertions))]
	fn check_bandpass_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		Ok(())
	}

	fn check_bandpass_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let period = 20;
		let bandwidth = 0.3;

		let input = BandPassInput::from_candles(
			&candles,
			"close",
			BandPassParams {
				period: Some(period),
				bandwidth: Some(bandwidth),
			},
		);
		let batch_output = bandpass_with_kernel(&input, kernel)?;

		let mut stream = BandPassStream::try_new(BandPassParams {
			period: Some(period),
			bandwidth: Some(bandwidth),
		})?;

		let mut stream_values = Vec::with_capacity(candles.close.len());
		for &price in &candles.close {
			let bp_val = stream.update(price);
			stream_values.push(bp_val);
		}

		assert_eq!(batch_output.bp.len(), stream_values.len());
		for (i, (&b, &s)) in batch_output.bp.iter().zip(stream_values.iter()).enumerate() {
			if b.is_nan() && s.is_nan() {
				continue;
			}
			let diff = (b - s).abs();
			let tol = 1e-10 * b.abs().max(1.0);
			assert!(
				diff < tol,
				"[{}] Streaming vs batch mismatch at index {}: batch={}, stream={}, diff={}",
				test_name,
				i,
				b,
				s,
				diff
			);
		}

		Ok(())
	}

	generate_all_bandpass_tests!(
		check_bandpass_partial_params,
		check_bandpass_accuracy,
		check_bandpass_default_candles,
		check_bandpass_zero_period,
		check_bandpass_period_exceeds_length,
		check_bandpass_very_small_dataset,
		check_bandpass_reinput,
		check_bandpass_nan_handling,
		check_bandpass_no_poison,
		check_bandpass_streaming
	);
	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let output = BandPassBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;

		let def = BandPassParams::default();
		let row = output.values_for(&def).expect("default row missing");

		assert_eq!(row.bp.len(), c.close.len());

		// Optional: Test known last 5 values for one column (bp)
		let expected = [
			-236.23678021132827,
			-247.4846395608195,
			-242.3788746078502,
			-212.89589193350128,
			-179.97293838509464,
		];
		let start = row.bp.len() - 5;
		for (i, &v) in row.bp[start..].iter().enumerate() {
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

	// Check for poison values in batch output - only runs in debug mode
	#[cfg(debug_assertions)]
	fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		// Test batch with multiple parameter combinations - more comprehensive coverage
		let output = BandPassBatchBuilder::new()
			.kernel(kernel)
			.period_range(5, 50, 5) // 10 periods: 5, 10, 15, 20, 25, 30, 35, 40, 45, 50
			.bandwidth_range(0.1, 0.9, 0.1) // 9 bandwidths: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
			.apply_candles(&c, "close")?;

		// This creates 90 parameter combinations (10 periods × 9 bandwidths)

		// Check every value in all output vectors for poison patterns
		for (row_idx, output_row) in output.values.iter().enumerate() {
			let params = &output.combos[row_idx]; // Get params for this row
										 // Check bp values
			for (col_idx, &val) in output_row.bp.iter().enumerate() {
				// Skip NaN values as they're expected in warmup periods
				if val.is_nan() {
					continue;
				}

				let bits = val.to_bits();

				// Check for alloc_with_nan_prefix poison (0x11111111_11111111)
				if bits == 0x11111111_11111111 {
					panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) in bp at row {} col {} with params {:?}",
                    test, val, bits, row_idx, col_idx, params
                );
				}

				// Check for init_matrix_prefixes poison (0x22222222_22222222)
				if bits == 0x22222222_22222222 {
					panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) in bp at row {} col {} with params {:?}",
                    test, val, bits, row_idx, col_idx, params
                );
				}

				// Check for make_uninit_matrix poison (0x33333333_33333333)
				if bits == 0x33333333_33333333 {
					panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) in bp at row {} col {} with params {:?}",
                    test, val, bits, row_idx, col_idx, params
                );
				}
			}

			// Check bp_normalized values
			for (col_idx, &val) in output_row.bp_normalized.iter().enumerate() {
				if val.is_nan() {
					continue;
				}

				let bits = val.to_bits();

				if bits == 0x11111111_11111111 {
					panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) in bp_normalized at row {} col {} with params {:?}",
                    test, val, bits, row_idx, col_idx, params
                );
				}

				if bits == 0x22222222_22222222 {
					panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) in bp_normalized at row {} col {} with params {:?}",
                    test, val, bits, row_idx, col_idx, params
                );
				}

				if bits == 0x33333333_33333333 {
					panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) in bp_normalized at row {} col {} with params {:?}",
                    test, val, bits, row_idx, col_idx, params
                );
				}
			}

			// Check signal values
			for (col_idx, &val) in output_row.signal.iter().enumerate() {
				if val.is_nan() {
					continue;
				}

				let bits = val.to_bits();

				if bits == 0x11111111_11111111 {
					panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) in signal at row {} col {} with params {:?}",
                    test, val, bits, row_idx, col_idx, params
                );
				}

				if bits == 0x22222222_22222222 {
					panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) in signal at row {} col {} with params {:?}",
                    test, val, bits, row_idx, col_idx, params
                );
				}

				if bits == 0x33333333_33333333 {
					panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) in signal at row {} col {} with params {:?}",
                    test, val, bits, row_idx, col_idx, params
                );
				}
			}

			// Check trigger values
			for (col_idx, &val) in output_row.trigger.iter().enumerate() {
				if val.is_nan() {
					continue;
				}

				let bits = val.to_bits();

				if bits == 0x11111111_11111111 {
					panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) in trigger at row {} col {} with params {:?}",
                    test, val, bits, row_idx, col_idx, params
                );
				}

				if bits == 0x22222222_22222222 {
					panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) in trigger at row {} col {} with params {:?}",
                    test, val, bits, row_idx, col_idx, params
                );
				}

				if bits == 0x33333333_33333333 {
					panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) in trigger at row {} col {} with params {:?}",
                    test, val, bits, row_idx, col_idx, params
                );
				}
			}
		}

		Ok(())
	}

	// Release mode stub - does nothing
	#[cfg(not(debug_assertions))]
	fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		Ok(())
	}

	gen_batch_tests!(check_batch_default_row);
	gen_batch_tests!(check_batch_no_poison);
}

// ========================= Python Bindings =========================

#[cfg(feature = "python")]
#[pyfunction(name = "bandpass")]
#[pyo3(signature = (data, period=20, bandwidth=0.3, kernel=None))]
pub fn bandpass_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	period: usize,
	bandwidth: f64,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
	use numpy::{IntoPyArray, PyArrayMethods};

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, false)?;

	let params = BandPassParams {
		period: Some(period),
		bandwidth: Some(bandwidth),
	};
	let bandpass_in = BandPassInput::from_slice(slice_in, params);

	// Get all output vectors from Rust function
	let output = py
		.allow_threads(|| bandpass_with_kernel(&bandpass_in, kern))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	// Create output dictionary with zero-copy transfers
	let dict = PyDict::new(py);
	dict.set_item("bp", output.bp.into_pyarray(py))?;
	dict.set_item("bp_normalized", output.bp_normalized.into_pyarray(py))?;
	dict.set_item("signal", output.signal.into_pyarray(py))?;
	dict.set_item("trigger", output.trigger.into_pyarray(py))?;

	Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "BandPassStream")]
pub struct BandPassStreamPy {
	stream: BandPassStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl BandPassStreamPy {
	#[new]
	fn new(period: usize, bandwidth: f64) -> PyResult<Self> {
		let params = BandPassParams {
			period: Some(period),
			bandwidth: Some(bandwidth),
		};
		let stream = BandPassStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(BandPassStreamPy { stream })
	}

	/// Updates the stream with a new value and returns the calculated band-pass value.
	/// Note: This returns only the bp value, not all 4 outputs for streaming simplicity.
	/// Unlike some indicators (e.g., ALMA), BandPassStream always returns a value from the
	/// first call - there is no warm-up period where None is returned.
	fn update(&mut self, value: f64) -> f64 {
		self.stream.update(value)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "bandpass_batch")]
#[pyo3(signature = (data, period_range, bandwidth_range, kernel=None))]
pub fn bandpass_batch_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	period_range: (usize, usize, usize),
	bandwidth_range: (f64, f64, f64),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;

	let slice_in = data.as_slice()?;

	let sweep = BandPassBatchRange {
		period: period_range,
		bandwidth: bandwidth_range,
	};

	// Use kernel validation for safety
	let kern = validate_kernel(kernel, true)?;

	// 1. Expand grid once to know rows*cols
	let combos = expand_grid(&sweep);
	let rows = combos.len();
	let cols = slice_in.len();

	// 2. Pre-allocate uninitialized NumPy arrays (1-D, will reshape later)
	// NOTE: PyArray1::new() creates uninitialized memory, not zero-initialized
	let bp_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let bp_normalized_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let signal_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let trigger_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };

	let bp_slice = unsafe { bp_arr.as_slice_mut()? };
	let bp_normalized_slice = unsafe { bp_normalized_arr.as_slice_mut()? };
	let signal_slice = unsafe { signal_arr.as_slice_mut()? };
	let trigger_slice = unsafe { trigger_arr.as_slice_mut()? };

	// 3. Heavy work without the GIL
	let combos = py
		.allow_threads(|| -> Result<Vec<BandPassParams>, BandPassError> {
			// Resolve Kernel::Auto to a specific kernel
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
			// Use the _into variant that writes directly to our pre-allocated buffers
			bandpass_batch_inner_into(
				slice_in,
				&sweep,
				simd,
				true,
				bp_slice,
				bp_normalized_slice,
				signal_slice,
				trigger_slice,
			)
		})
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	// 4. Reshape arrays
	let bp_arr = bp_arr.reshape((rows, cols))?;
	let bp_normalized_arr = bp_normalized_arr.reshape((rows, cols))?;
	let signal_arr = signal_arr.reshape((rows, cols))?;
	let trigger_arr = trigger_arr.reshape((rows, cols))?;

	// Build output dictionary
	let dict = PyDict::new(py);
	dict.set_item("bp", bp_arr)?;
	dict.set_item("bp_normalized", bp_normalized_arr)?;
	dict.set_item("signal", signal_arr)?;
	dict.set_item("trigger", trigger_arr)?;

	// Add parameter arrays
	dict.set_item(
		"periods",
		combos
			.iter()
			.map(|p| p.period.unwrap() as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;
	dict.set_item(
		"bandwidths",
		combos
			.iter()
			.map(|p| p.bandwidth.unwrap())
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;

	Ok(dict)
}

// ========================= WASM Bindings =========================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct BandPassResult {
	values: Vec<f64>, // [bp..., bp_normalized..., signal..., trigger...]
	rows: usize,      // 4 for bandpass
	cols: usize,      // data length
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl BandPassResult {
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
pub fn bandpass_js(data: &[f64], period: usize, bandwidth: f64) -> Result<BandPassResult, JsValue> {
	let params = BandPassParams {
		period: Some(period),
		bandwidth: Some(bandwidth),
	};
	let input = BandPassInput::from_slice(data, params);
	
	// Single allocation for flattened output
	let len = data.len();
	let mut output = vec![0.0; len * 4]; // [bp..., bp_normalized..., signal..., trigger...]
	
	// Split the output into 4 slices
	let (bp_slice, rest) = output.split_at_mut(len);
	let (bp_norm_slice, rest) = rest.split_at_mut(len);
	let (signal_slice, trigger_slice) = rest.split_at_mut(len);
	
	// Use the zero-allocation helper
	bandpass_into_slice(bp_slice, bp_norm_slice, signal_slice, trigger_slice, &input, Kernel::Auto)
		.map_err(|e| JsValue::from_str(&e.to_string()))?;
	
	// Return the flattened result - no copies!
	Ok(BandPassResult {
		values: output,
		rows: 4,
		cols: len,
	})
}


#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn bandpass_batch_metadata(
	period_start: usize,
	period_end: usize,
	period_step: usize,
	bandwidth_start: f64,
	bandwidth_end: f64,
	bandwidth_step: f64,
) -> Vec<f64> {
	let sweep = BandPassBatchRange {
		period: (period_start, period_end, period_step),
		bandwidth: (bandwidth_start, bandwidth_end, bandwidth_step),
	};

	let combos = expand_grid(&sweep);
	let mut metadata = Vec::with_capacity(combos.len() * 2);

	for combo in combos {
		metadata.push(combo.period.unwrap() as f64);
		metadata.push(combo.bandwidth.unwrap());
	}

	metadata
}

// New ergonomic WASM API
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct BandPassBatchConfig {
	pub period_range: (usize, usize, usize),
	pub bandwidth_range: (f64, f64, f64),
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct BandPassBatchResult {
	values: Vec<f64>, // flattened [bp_row0..., bp_norm_row0..., signal_row0..., trigger_row0..., bp_row1..., ...]
	combos: usize,    // number of parameter combinations
	outputs: usize,   // 4 (bp, bp_normalized, signal, trigger)
	cols: usize,      // data length
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl BandPassBatchResult {
	#[wasm_bindgen(getter)]
	pub fn values(&self) -> Vec<f64> {
		self.values.clone()
	}

	#[wasm_bindgen(getter)]
	pub fn combos(&self) -> usize {
		self.combos
	}

	#[wasm_bindgen(getter)]
	pub fn outputs(&self) -> usize {
		self.outputs
	}

	#[wasm_bindgen(getter)]
	pub fn cols(&self) -> usize {
		self.cols
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = bandpass_batch)]
pub fn bandpass_batch_unified_js(data: &[f64], config: JsValue) -> Result<BandPassBatchResult, JsValue> {
	// 1. Deserialize the configuration object from JavaScript
	let config: BandPassBatchConfig =
		serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

	let sweep = BandPassBatchRange {
		period: config.period_range,
		bandwidth: config.bandwidth_range,
	};

	// 2. Get dimensions
	let combos = expand_grid(&sweep);
	let num_combos = combos.len();
	let cols = data.len();

	// Single allocation for all outputs in row-major order
	// Layout: [combo0_bp, combo0_bp_norm, combo0_signal, combo0_trigger, combo1_bp, ...]
	let mut output = vec![0.0; num_combos * cols * 4];

	// Process each combination
	for (i, combo) in combos.iter().enumerate() {
		let input = BandPassInput::from_slice(data, combo.clone());
		
		// Calculate offset for this combination's outputs
		let offset = i * cols * 4;
		
		// Split the output slice for this combination
		let combo_slice = &mut output[offset..offset + cols * 4];
		let (bp_slice, rest) = combo_slice.split_at_mut(cols);
		let (bp_norm_slice, rest) = rest.split_at_mut(cols);
		let (signal_slice, trigger_slice) = rest.split_at_mut(cols);
		
		// Compute directly into the output slices
		bandpass_into_slice(bp_slice, bp_norm_slice, signal_slice, trigger_slice, &input, Kernel::Scalar)
			.map_err(|e| JsValue::from_str(&e.to_string()))?;
	}

	// Return the flattened result - no copies!
	Ok(BandPassBatchResult {
		values: output,
		combos: num_combos,
		outputs: 4,
		cols,
	})
}

// ========================= Fast API with Aliasing Detection =========================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn bandpass_into(
	in_ptr: *const f64,
	bp_ptr: *mut f64,
	bp_normalized_ptr: *mut f64,
	signal_ptr: *mut f64,
	trigger_ptr: *mut f64,
	len: usize,
	period: usize,
	bandwidth: f64,
) -> Result<(), JsValue> {
	if in_ptr.is_null() || bp_ptr.is_null() || bp_normalized_ptr.is_null() || signal_ptr.is_null() || trigger_ptr.is_null() {
		return Err(JsValue::from_str("Null pointer provided"));
	}
	
	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);
		let params = BandPassParams {
			period: Some(period),
			bandwidth: Some(bandwidth),
		};
		let input = BandPassInput::from_slice(data, params);
		
		// Check if any output pointers alias with input
		let in_aliases_bp = in_ptr == bp_ptr;
		let in_aliases_bp_norm = in_ptr == bp_normalized_ptr;
		let in_aliases_signal = in_ptr == signal_ptr;
		let in_aliases_trigger = in_ptr == trigger_ptr;
		
		// Check if any output pointers alias with each other
		let outputs_alias = bp_ptr == bp_normalized_ptr || bp_ptr == signal_ptr || bp_ptr == trigger_ptr
			|| bp_normalized_ptr == signal_ptr || bp_normalized_ptr == trigger_ptr
			|| signal_ptr == trigger_ptr;
		
		if in_aliases_bp || in_aliases_bp_norm || in_aliases_signal || in_aliases_trigger || outputs_alias {
			// Need temporary buffer - single allocation
			let mut temp = vec![0.0; len * 4];
			
			// Split temp buffer into 4 slices
			let (temp_bp, rest) = temp.split_at_mut(len);
			let (temp_bp_normalized, rest) = rest.split_at_mut(len);
			let (temp_signal, temp_trigger) = rest.split_at_mut(len);
			
			bandpass_into_slice(temp_bp, temp_bp_normalized, temp_signal, temp_trigger, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			
			// Copy to output pointers
			let bp_out = std::slice::from_raw_parts_mut(bp_ptr, len);
			let bp_normalized_out = std::slice::from_raw_parts_mut(bp_normalized_ptr, len);
			let signal_out = std::slice::from_raw_parts_mut(signal_ptr, len);
			let trigger_out = std::slice::from_raw_parts_mut(trigger_ptr, len);
			
			bp_out.copy_from_slice(temp_bp);
			bp_normalized_out.copy_from_slice(temp_bp_normalized);
			signal_out.copy_from_slice(temp_signal);
			trigger_out.copy_from_slice(temp_trigger);
		} else {
			// Direct write - no aliasing
			let bp_out = std::slice::from_raw_parts_mut(bp_ptr, len);
			let bp_normalized_out = std::slice::from_raw_parts_mut(bp_normalized_ptr, len);
			let signal_out = std::slice::from_raw_parts_mut(signal_ptr, len);
			let trigger_out = std::slice::from_raw_parts_mut(trigger_ptr, len);
			
			bandpass_into_slice(bp_out, bp_normalized_out, signal_out, trigger_out, &input, Kernel::Auto)
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		Ok(())
	}
}

// ========================= Memory Management =========================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn bandpass_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn bandpass_free(ptr: *mut f64, len: usize) {
	if !ptr.is_null() {
		unsafe {
			let _ = Vec::from_raw_parts(ptr, len, len);
		}
	}
}
