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

use crate::indicators::rsi::{rsi, RsiError, RsiInput, RsiOutput, RsiParams};
use crate::indicators::stoch::{stoch, StochError, StochInput, StochOutput, StochParams};
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
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
	let mut k_vals = vec![f64::NAN; rows * cols];
	let mut d_vals = vec![f64::NAN; rows * cols];

	let do_row = |row: usize, k_out: &mut [f64], d_out: &mut [f64]| unsafe {
		let prm = &combos[row];
		srsi_row_scalar(
			data,
			prm.rsi_period.unwrap(),
			prm.stoch_period.unwrap(),
			prm.k.unwrap(),
			prm.d.unwrap(),
			k_out,
			d_out,
		);
	};

	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			k_vals
				.par_chunks_mut(cols)
				.zip(d_vals.par_chunks_mut(cols))
				.enumerate()
				.for_each(|(row, (ks, ds))| do_row(row, ks, ds));
		}

		#[cfg(target_arch = "wasm32")]
		{
			for (row, (ks, ds)) in k_vals.chunks_mut(cols).zip(d_vals.chunks_mut(cols)).enumerate() {
				do_row(row, ks, ds);
			}
		}
	} else {
		for (row, (ks, ds)) in k_vals.chunks_mut(cols).zip(d_vals.chunks_mut(cols)).enumerate() {
			do_row(row, ks, ds);
		}
	}
	Ok(SrsiBatchOutput {
		k: k_vals,
		d: d_vals,
		combos,
		rows,
		cols,
	})
}

#[inline(always)]
pub fn expand_grid_srsi(r: &SrsiBatchRange) -> Vec<SrsiParams> {
	expand_grid(r)
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
		check_srsi_from_slice
	);

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
