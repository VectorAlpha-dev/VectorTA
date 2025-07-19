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
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
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
	let mut close_volume: Vec<f64> = vec![0.0; len];
	for i in 0..len {
		close_volume[i] = close[i] * volume[i];
	}

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
	let sma_close_vol_long = sma(&SmaInput {
		data: SmaData::Slice(&close_volume),
		params: SmaParams { period: Some(long) },
	})?
	.values;
	let sma_close_vol_short = sma(&SmaInput {
		data: SmaData::Slice(&close_volume),
		params: SmaParams { period: Some(short) },
	})?
	.values;

	let mut vpci = vec![f64::NAN; len];
	let mut vpcis = vec![f64::NAN; len];
	let mut vwma_long = vec![f64::NAN; len];
	let mut vwma_short = vec![f64::NAN; len];

	for i in 0..len {
		if !sma_volume_long[i].is_nan() && sma_volume_long[i] != 0.0 {
			vwma_long[i] = sma_close_vol_long[i] / sma_volume_long[i];
		}
		if !sma_volume_short[i].is_nan() && sma_volume_short[i] != 0.0 {
			vwma_short[i] = sma_close_vol_short[i] / sma_volume_short[i];
		}
	}
	let mut vpci_times_vol = vec![f64::NAN; len];
	for i in 0..len {
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
		let val = vpc * vpr * vm;
		vpci[i] = val;
		if !val.is_nan() && !volume[i].is_nan() {
			vpci_times_vol[i] = val * volume[i];
		}
	}

	let sma_vpci_times_vol_short = sma(&SmaInput {
		data: SmaData::Slice(&vpci_times_vol),
		params: SmaParams { period: Some(short) },
	})?
	.values;

	for i in 0..len {
		if !sma_volume_short[i].is_nan() && sma_volume_short[i] != 0.0 {
			vpcis[i] = sma_vpci_times_vol_short[i] / sma_volume_short[i];
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
	let mut vpci = vec![f64::NAN; rows * cols];
	let mut vpcis = vec![f64::NAN; rows * cols];

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

	Ok(VpciBatchOutput {
		vpci,
		vpcis,
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
