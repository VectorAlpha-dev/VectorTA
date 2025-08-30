//! # Trend Flex Filter (TrendFlex)
//!
//! Highlights momentum shifts using a super smoother and volatility measurement around it.
//! Adapts to market volatility, amplifying or dampening its reaction accordingly.
//!
//! ## Parameters
//! - **period**: Primary lookback period (defaults to 20).
//!
//! ## Errors
//! - **NoDataProvided**: No input data provided.
//! - **AllValuesNaN**: All input data are NaN.
//! - **ZeroTrendFlexPeriod**: period is zero.
//! - **TrendFlexPeriodExceedsData**: period > data length.
//! - **SmootherPeriodExceedsData**: supersmoother period > data length.
//!
//! ## Returns
//! - **Ok(TrendFlexOutput)**: Vec<f64> matching input length.
//! - **Err(TrendFlexError)**: otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
	alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
use aligned_vec::{AVec, ConstAlign, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

// Input handling (AsRef)
impl<'a> AsRef<[f64]> for TrendFlexInput<'a> {
	#[inline(always)]
	fn as_ref(&self) -> &[f64] {
		match &self.data {
			TrendFlexData::Slice(slice) => slice,
			TrendFlexData::Candles { candles, source } => source_type(candles, source),
		}
	}
}

// Input/Output/Param types
#[derive(Debug, Clone)]
pub enum TrendFlexData<'a> {
	Candles { candles: &'a Candles, source: &'a str },
	Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct TrendFlexOutput {
	pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct TrendFlexParams {
	pub period: Option<usize>,
}

impl Default for TrendFlexParams {
	fn default() -> Self {
		Self { period: Some(20) }
	}
}

#[derive(Debug, Clone)]
pub struct TrendFlexInput<'a> {
	pub data: TrendFlexData<'a>,
	pub params: TrendFlexParams,
}

impl<'a> TrendFlexInput<'a> {
	#[inline]
	pub fn from_candles(c: &'a Candles, s: &'a str, p: TrendFlexParams) -> Self {
		Self {
			data: TrendFlexData::Candles { candles: c, source: s },
			params: p,
		}
	}
	#[inline]
	pub fn from_slice(sl: &'a [f64], p: TrendFlexParams) -> Self {
		Self {
			data: TrendFlexData::Slice(sl),
			params: p,
		}
	}
	#[inline]
	pub fn with_default_candles(c: &'a Candles) -> Self {
		Self::from_candles(c, "close", TrendFlexParams::default())
	}
	#[inline]
	pub fn get_period(&self) -> usize {
		self.params.period.unwrap_or(20)
	}
}

// Builder
#[derive(Copy, Clone, Debug)]
pub struct TrendFlexBuilder {
	period: Option<usize>,
	kernel: Kernel,
}

impl Default for TrendFlexBuilder {
	fn default() -> Self {
		Self {
			period: None,
			kernel: Kernel::Auto,
		}
	}
}

impl TrendFlexBuilder {
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
	pub fn kernel(mut self, k: Kernel) -> Self {
		self.kernel = k;
		self
	}
	#[inline(always)]
	pub fn apply(self, c: &Candles) -> Result<TrendFlexOutput, TrendFlexError> {
		let p = TrendFlexParams { period: self.period };
		let i = TrendFlexInput::from_candles(c, "close", p);
		trendflex_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn apply_slice(self, d: &[f64]) -> Result<TrendFlexOutput, TrendFlexError> {
		let p = TrendFlexParams { period: self.period };
		let i = TrendFlexInput::from_slice(d, p);
		trendflex_with_kernel(&i, self.kernel)
	}
	#[inline(always)]
	pub fn into_stream(self) -> Result<TrendFlexStream, TrendFlexError> {
		let p = TrendFlexParams { period: self.period };
		TrendFlexStream::try_new(p)
	}
}

// Error
#[derive(Debug, Error)]
pub enum TrendFlexError {
	#[error("trendflex: No data provided.")]
	NoDataProvided,
	#[error("trendflex: All values are NaN.")]
	AllValuesNaN,
	#[error("trendflex: period = 0")]
	ZeroTrendFlexPeriod { period: usize },
	#[error("trendflex: period > data len: period = {period}, data_len = {data_len}")]
	TrendFlexPeriodExceedsData { period: usize, data_len: usize },
	#[error("trendflex: smoother period > data len: ss_period = {ss_period}, data_len = {data_len}")]
	SmootherPeriodExceedsData { ss_period: usize, data_len: usize },
}

// Main entrypoint
#[inline]
pub fn trendflex(input: &TrendFlexInput) -> Result<TrendFlexOutput, TrendFlexError> {
	trendflex_with_kernel(input, Kernel::Auto)
}

pub fn trendflex_with_kernel(input: &TrendFlexInput, kernel: Kernel) -> Result<TrendFlexOutput, TrendFlexError> {
	let data: &[f64] = input.as_ref();
	let len = data.len();
	if len == 0 {
		return Err(TrendFlexError::NoDataProvided);
	}

	let period = input.get_period();
	if period == 0 {
		return Err(TrendFlexError::ZeroTrendFlexPeriod { period });
	}
	if period >= len {
		return Err(TrendFlexError::TrendFlexPeriodExceedsData { period, data_len: len });
	}

	let first = data
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(TrendFlexError::AllValuesNaN)?;
	let ss_period = ((period as f64) / 2.0).round() as usize;
	if ss_period > len {
		return Err(TrendFlexError::SmootherPeriodExceedsData {
			ss_period,
			data_len: len,
		});
	}

	let warm = first + period;
	let mut out = alloc_with_nan_prefix(len, warm);

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		k => k,
	};

	// Compute directly into `out` past the warmup prefix.
	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => trendflex_scalar_into(data, period, ss_period, first, &mut out)?,
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => trendflex_avx2_into(data, period, ss_period, first, &mut out)?,
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => trendflex_avx512_into(data, period, ss_period, first, &mut out)?,
			#[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
			Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
				trendflex_scalar_into(data, period, ss_period, first, &mut out)?
			}
			Kernel::Auto => unreachable!(),
		}
	}

	Ok(TrendFlexOutput { values: out })
}

pub fn trendflex_into_slice(
	dst: &mut [f64],
	input: &TrendFlexInput,
	kernel: Kernel,
) -> Result<(), TrendFlexError> {
	let data: &[f64] = input.as_ref();
	let len = data.len();
	if dst.len() != len {
		return Err(TrendFlexError::TrendFlexPeriodExceedsData {
			period: dst.len(),
			data_len: len,
		});
	}
	if len == 0 {
		return Err(TrendFlexError::NoDataProvided);
	}
	let period = input.get_period();
	if period == 0 {
		return Err(TrendFlexError::ZeroTrendFlexPeriod { period });
	}
	if period >= len {
		return Err(TrendFlexError::TrendFlexPeriodExceedsData { period, data_len: len });
	}
	let first = data.iter().position(|x| !x.is_nan()).ok_or(TrendFlexError::AllValuesNaN)?;
	let ss_period = ((period as f64) / 2.0).round() as usize;
	if ss_period > data.len() {
		return Err(TrendFlexError::SmootherPeriodExceedsData {
			ss_period,
			data_len: data.len(),
		});
	}

	let chosen = match kernel {
		Kernel::Auto => detect_best_kernel(),
		k => k,
	};

	unsafe {
		match chosen {
			Kernel::Scalar | Kernel::ScalarBatch => trendflex_scalar_into(data, period, ss_period, first, dst)?,
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => trendflex_avx2_into(data, period, ss_period, first, dst)?,
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => trendflex_avx512_into(data, period, ss_period, first, dst)?,
			#[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
			Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
				trendflex_scalar_into(data, period, ss_period, first, dst)?
			}
			Kernel::Auto => unreachable!(),
		}
	}

	let warmup_end = first + period;
	for v in &mut dst[..warmup_end] {
		*v = f64::NAN;
	}
	Ok(())
}

// In-place scalar kernel that writes directly into output slice
#[inline]
unsafe fn trendflex_scalar_into(
	data: &[f64],
	period: usize,
	ss_period: usize,
	first_valid: usize,
	out: &mut [f64],
) -> Result<(), TrendFlexError> {
	use std::f64::consts::PI;

	let len = data.len();
	let warm = first_valid + period;
	
	// Ensure NaN prefix is set for warmup period
	for i in 0..warm {
		out[i] = f64::NAN;
	}

	if first_valid == 0 {
		// super smoother over full series
		let mut ssf = vec![0.0; len];
		ssf[0] = data[0];
		if len > 1 {
			ssf[1] = data[1];
		}

		let a = (-1.414_f64 * PI / ss_period as f64).exp();
		let a_sq = a * a;
		let b = 2.0 * a * (1.414_f64 * PI / ss_period as f64).cos();
		let c = (1.0 + a_sq - b) * 0.5;

		for i in 2..len {
			ssf[i] = c * (data[i] + data[i - 1]) + b * ssf[i - 1] - a_sq * ssf[i - 2];
		}

		let tp_f = period as f64;
		let inv_tp = 1.0 / tp_f;
		let mut ms_prev = 0.0;
		let mut rolling_sum = ssf[..period].iter().sum::<f64>();

		for i in period..len {
			let my_sum = (tp_f * ssf[i] - rolling_sum) * inv_tp;
			let ms_current = 0.04 * my_sum * my_sum + 0.96 * ms_prev;
			ms_prev = ms_current;

			out[i] = if ms_current != 0.0 {
				my_sum / ms_current.sqrt()
			} else {
				0.0
			};
			rolling_sum += ssf[i] - ssf[i - period];
		}
		// prefix [..warm) already NaN via alloc_with_nan_prefix
		return Ok(());
	}

	// first_valid > 0: operate on tail, write back in-place to `out[first_valid..]`
	let m = len - first_valid;
	if m < period {
		// nothing to compute; leave out as-is (NaN prefix covers entire tail)
		return Ok(());
	}
	if m < ss_period {
		return Err(TrendFlexError::SmootherPeriodExceedsData { ss_period, data_len: m });
	}

	let tail = &data[first_valid..];
	let mut ssf = vec![0.0; m];
	ssf[0] = tail[0];
	if m > 1 {
		ssf[1] = tail[1];
	}

	let a = (-1.414_f64 * PI / ss_period as f64).exp();
	let a_sq = a * a;
	let b = 2.0 * a * (1.414_f64 * PI / ss_period as f64).cos();
	let c = (1.0 + a_sq - b) * 0.5;

	for i in 2..m {
		ssf[i] = c * (tail[i] + tail[i - 1]) + b * ssf[i - 1] - a_sq * ssf[i - 2];
	}

	let tp_f = period as f64;
	let inv_tp = 1.0 / tp_f;
	let mut ms_prev = 0.0;
	let mut rolling_sum = ssf[..period].iter().sum::<f64>();

	for i in period..m {
		let my_sum = (tp_f * ssf[i] - rolling_sum) * inv_tp;
		let ms_current = 0.04 * my_sum * my_sum + 0.96 * ms_prev;
		ms_prev = ms_current;

		out[first_valid + i] = if ms_current != 0.0 {
			my_sum / ms_current.sqrt()
		} else {
			0.0
		};
		rolling_sum += ssf[i] - ssf[i - period];
	}

	Ok(())
}

// AVX2 implementation (micro-SIMD)
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn trendflex_avx2_into(
	data: &[f64],
	period: usize,
	ss_period: usize,
	first_valid: usize,
	out: &mut [f64],
) -> Result<(), TrendFlexError> {
    use std::f64::consts::PI;

    let len = data.len();
    let warm = first_valid + period;
    for i in 0..warm.min(out.len()) {
        *out.get_unchecked_mut(i) = f64::NAN;
    }

    if first_valid >= len { return Ok(()); }

    // Super smoother coefficients
    let a = (-1.414_f64 * PI / ss_period as f64).exp();
    let a_sq = a * a;
    let b = 2.0 * a * (1.414_f64 * PI / ss_period as f64).cos();
    let c = (1.0 + a_sq - b) * 0.5;

    // Helper to compute a single series in a streaming fashion with a ring buffer
    #[inline(always)]
    unsafe fn run_series_avx2(
        x: &[f64], period: usize, a_sq: f64, b: f64, c: f64,
        out: &mut [f64], out_off: usize
    ) {
        let n = x.len();
        if n == 0 { return; }
        let mut prev2 = x[0];
        let mut prev1 = if n > 1 { x[1] } else { x[0] };
        // initialize ring buffer and rolling sum
        let mut ring = vec![0.0f64; period];
        let mut sum = 0.0f64;
        let mut head = 0usize;
        // seed first entries
        ring[head] = prev2; sum += prev2; head = (head + 1) % period;
        if n > 1 { ring[head] = prev1; sum += prev1; head = (head + 1) % period; }

        let tp_f = period as f64;
        let inv_tp = 1.0 / tp_f;
        let mut ms_prev = 0.0f64;

        // fill until we have period samples
        let mut i = 2usize;
        while i < n && i < period {
            let cur = c * (x[i] + x[i-1]) + b * prev1 - a_sq * prev2;
            prev2 = prev1; prev1 = cur;
            sum += cur;
            ring[head] = cur; head = (head + 1) % period;
            i += 1;
        }
        // main loop: produce outputs
        while i < n {
            _mm_prefetch(x.as_ptr().add(i + 16).cast(), _MM_HINT_T0);
            let cur = c * (x[i] + x[i-1]) + b * prev1 - a_sq * prev2;
            prev2 = prev1; prev1 = cur;

            let my_sum = (tp_f * cur - sum) * inv_tp;
            // ms_current = 0.04*my_sum*my_sum + 0.96*ms_prev
            let v = _mm_set_sd(my_sum);
            let sq = _mm_mul_sd(v, v);
            let s04 = _mm_mul_sd(_mm_set_sd(0.04), sq);
            let s96 = _mm_mul_sd(_mm_set_sd(0.96), _mm_set_sd(ms_prev));
            let ms_cur = _mm_add_sd(s04, s96);
            let ms_current = _mm_cvtsd_f64(ms_cur);
            ms_prev = ms_current;

            let out_val = if ms_current != 0.0 {
                let denom = _mm_sqrt_sd(_mm_setzero_pd(), _mm_set_sd(ms_current));
                let denom_s = _mm_cvtsd_f64(denom);
                my_sum / denom_s
            } else { 0.0 };
            // non-temporal store
            _mm_stream_sd(out.get_unchecked_mut(out_off + i) as *mut f64, _mm_set_sd(out_val));

            // update rolling sum and ring
            let old = ring[head];
            sum += cur - old;
            ring[head] = cur;
            head = (head + 1) % period;

            i += 1;
        }
    }

    if first_valid == 0 {
        run_series_avx2(data, period, a_sq, b, c, out, 0);
        return Ok(());
    }

    // Tail-only case
    let m = len - first_valid;
    if m < period { return Ok(()); }
    if m < ss_period { return Err(TrendFlexError::SmootherPeriodExceedsData { ss_period, data_len: m }); }
    let tail = &data[first_valid..];
    run_series_avx2(tail, period, a_sq, b, c, out, first_valid);
    Ok(())
}

// AVX512 stub
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f,avx512dq,fma")]
unsafe fn trendflex_avx512_into(
	data: &[f64],
	period: usize,
	ss_period: usize,
	first_valid: usize,
	out: &mut [f64],
) -> Result<(), TrendFlexError> {
    use std::f64::consts::PI;

    let len = data.len();
    let warm = first_valid + period;
    for i in 0..warm.min(out.len()) {
        *out.get_unchecked_mut(i) = f64::NAN;
    }

    if first_valid >= len { return Ok(()); }

    let a = (-1.414_f64 * PI / ss_period as f64).exp();
    let a_sq = a * a;
    let b = 2.0 * a * (1.414_f64 * PI / ss_period as f64).cos();
    let c = (1.0 + a_sq - b) * 0.5;

    // Streaming series with ring buffer and non-temporal stores
    #[inline(always)]
    unsafe fn run_series_avx512(
        x: &[f64], period: usize, a_sq: f64, b: f64, c: f64,
        out: &mut [f64], out_off: usize
    ) {
        let n = x.len();
        if n == 0 { return; }
        let mut prev2 = x[0];
        let mut prev1 = if n > 1 { x[1] } else { x[0] };
        let mut ring = vec![0.0f64; period];
        let mut sum = 0.0f64;
        let mut head = 0usize;
        ring[head] = prev2; sum += prev2; head = (head + 1) % period;
        if n > 1 { ring[head] = prev1; sum += prev1; head = (head + 1) % period; }

        let tp_f = period as f64;
        let inv_tp = 1.0 / tp_f;
        let mut ms_prev = 0.0f64;

        let mut i = 2usize;
        while i < n && i < period {
            let cur = c * (x[i] + x[i-1]) + b * prev1 - a_sq * prev2;
            prev2 = prev1; prev1 = cur;
            sum += cur;
            ring[head] = cur; head = (head + 1) % period;
            i += 1;
        }
        while i < n {
            _mm_prefetch(x.as_ptr().add(i + 32).cast(), _MM_HINT_T0);
            let cur = c * (x[i] + x[i-1]) + b * prev1 - a_sq * prev2;
            prev2 = prev1; prev1 = cur;

            let my_sum = (tp_f * cur - sum) * inv_tp;
            let v = _mm_set_sd(my_sum);
            let sq = _mm_mul_sd(v, v);
            let s04 = _mm_mul_sd(_mm_set_sd(0.04), sq);
            let s96 = _mm_mul_sd(_mm_set_sd(0.96), _mm_set_sd(ms_prev));
            let ms_cur = _mm_add_sd(s04, s96);
            let ms_current = _mm_cvtsd_f64(ms_cur);
            ms_prev = ms_current;

            let out_val = if ms_current != 0.0 {
                let denom = _mm_sqrt_sd(_mm_setzero_pd(), _mm_set_sd(ms_current));
                my_sum / _mm_cvtsd_f64(denom)
            } else { 0.0 };
            _mm_stream_sd(out.get_unchecked_mut(out_off + i) as *mut f64, _mm_set_sd(out_val));

            let old = ring[head];
            sum += cur - old;
            ring[head] = cur;
            head = (head + 1) % period;

            i += 1;
        }
    }

    if first_valid == 0 {
        run_series_avx512(data, period, a_sq, b, c, out, 0);
        return Ok(());
    }

    let m = len - first_valid;
    if m < period { return Ok(()); }
    if m < ss_period { return Err(TrendFlexError::SmootherPeriodExceedsData { ss_period, data_len: m }); }
    let tail = &data[first_valid..];
    run_series_avx512(tail, period, a_sq, b, c, out, first_valid);
    Ok(())
}

// AVX512 short stub
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
unsafe fn trendflex_avx512_short_into(
	data: &[f64],
	period: usize,
	ss_period: usize,
	first_valid: usize,
	out: &mut [f64],
) -> Result<(), TrendFlexError> {
	trendflex_scalar_into(data, period, ss_period, first_valid, out)
}

// AVX512 long stub
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
unsafe fn trendflex_avx512_long_into(
	data: &[f64],
	period: usize,
	ss_period: usize,
	first_valid: usize,
	out: &mut [f64],
) -> Result<(), TrendFlexError> {
	trendflex_scalar_into(data, period, ss_period, first_valid, out)
}

// Streaming implementation
#[derive(Debug, Clone)]
pub struct TrendFlexStream {
	period: usize,
	ss_period: usize,
	ssf: Vec<f64>,
	ms_prev: f64,
	buffer: Vec<f64>,
	sum: f64,
	idx: usize,
	filled: bool,
	last_raw: Option<f64>,
	a: f64,
	a_sq: f64,
	b: f64,
	c: f64,
}

impl TrendFlexStream {
	pub fn try_new(params: TrendFlexParams) -> Result<Self, TrendFlexError> {
		let period = params.period.unwrap_or(20);
		if period == 0 {
			return Err(TrendFlexError::ZeroTrendFlexPeriod { period });
		}
		let ss_period = ((period as f64) / 2.0).round() as usize;
		if ss_period == 0 {
			return Err(TrendFlexError::SmootherPeriodExceedsData { ss_period, data_len: 0 });
		}

		use std::f64::consts::PI;
		let a = (-1.414_f64 * PI / (ss_period as f64)).exp();
		let a_sq = a * a;
		let b = 2.0 * a * (1.414_f64 * PI / (ss_period as f64)).cos();
		let c = (1.0 + a_sq - b) * 0.5;

		Ok(Self {
			period,
			ss_period,
			ssf: Vec::with_capacity(ss_period.max(3)),
			ms_prev: 0.0,
			buffer: vec![0.0; period],
			sum: 0.0,
			idx: 0,
			filled: false,
			last_raw: None,
			a,
			a_sq,
			b,
			c,
		})
	}

	#[inline(always)]
	pub fn update(&mut self, value: f64) -> Option<f64> {
		let n = self.ssf.len();
		if n == 0 {
			self.ssf.push(value);
			self.buffer[self.idx] = value;
			self.sum += value;
			self.idx = (self.idx + 1) % self.period;
			self.last_raw = Some(value);
			return None;
		}
		if n == 1 {
			self.ssf.push(value);
			self.buffer[self.idx] = value;
			self.sum += value;
			self.idx = (self.idx + 1) % self.period;
			self.last_raw = Some(value);
			return None;
		}
		let prev_raw = self.last_raw.unwrap();
		let prev1 = self.ssf[n - 1];
		let prev2 = self.ssf[n - 2];
		let new_ssf = self.c * (value + prev_raw) + self.b * prev1 - self.a_sq * prev2;
		self.ssf.push(new_ssf);

		self.last_raw = Some(value);
		let p = self.period;
		let old = self.buffer[self.idx];
		let rolling_sum = self.sum;
		let my_sum = (p as f64 * new_ssf - rolling_sum) / (p as f64);
		self.sum = rolling_sum + new_ssf - old;
		self.buffer[self.idx] = new_ssf;
		self.idx = (self.idx + 1) % p;

		if !self.filled && self.ssf.len() > p {
			self.filled = true;
		}
		if !self.filled {
			return None;
		}

		let tp_f = p as f64;
		let inv_tp = 1.0 / tp_f;

		let ms_current = 0.04 * my_sum * my_sum + 0.96 * self.ms_prev;
		self.ms_prev = ms_current;

		Some(if ms_current != 0.0 {
			my_sum / ms_current.sqrt()
		} else {
			0.0
		})
	}
}

// Batch functions for direct buffer writing
#[inline(always)]
pub fn trendflex_batch_inner_into(
	data: &[f64],
	sweep: &TrendFlexBatchRange,
	kern: Kernel,
	parallel: bool,
	out: &mut [f64],
) -> Result<Vec<TrendFlexParams>, TrendFlexError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(TrendFlexError::NoDataProvided);
	}
	
	let first = data
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(TrendFlexError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if data.len() - first < max_p {
		return Err(TrendFlexError::TrendFlexPeriodExceedsData {
			period: max_p,
			data_len: data.len() - first,
		});
	}
	
	let rows = combos.len();
	let cols = data.len();
	
	// Ensure output buffer is the right size
	assert_eq!(out.len(), rows * cols, "Output buffer size mismatch");
	
	let warm: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap()).collect();
	
	// Initialize output buffer with NaN prefixes
	for (row, &warmup) in warm.iter().enumerate() {
		let start = row * cols;
		let end = start + warmup;
		out[start..end].fill(f64::NAN);
	}
	
	// Resolve Auto kernel to actual kernel
	let actual_kern = match kern {
		Kernel::Auto => detect_best_batch_kernel(),
		k => k,
	};
	
	// Helper that fills one row
	let do_row = |row: usize, out_row: &mut [f64]| unsafe {
		let period = combos[row].period.unwrap();
		
		match actual_kern {
			Kernel::Scalar | Kernel::ScalarBatch => trendflex_row_scalar(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => trendflex_row_avx2(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => trendflex_row_avx512(data, first, period, out_row),
			#[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
			Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => trendflex_row_scalar(data, first, period, out_row),
			Kernel::Auto => unreachable!("Auto kernel should have been resolved"),
		}
	};
	
	// Run every row, writing directly into output buffer
	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			use rayon::prelude::*;
			out.par_chunks_mut(cols)
				.enumerate()
				.for_each(|(row, slice)| do_row(row, slice));
		}
		
		#[cfg(target_arch = "wasm32")]
		{
			for (row, slice) in out.chunks_mut(cols).enumerate() {
				do_row(row, slice);
			}
		}
	} else {
		for (row, slice) in out.chunks_mut(cols).enumerate() {
			do_row(row, slice);
		}
	}
	
	Ok(combos)
}

// Batch grid
#[derive(Clone, Debug)]
pub struct TrendFlexBatchRange {
	pub period: (usize, usize, usize),
}

impl Default for TrendFlexBatchRange {
	fn default() -> Self {
		Self { period: (20, 80, 1) }
	}
}

#[derive(Clone, Debug, Default)]
pub struct TrendFlexBatchBuilder {
	range: TrendFlexBatchRange,
	kernel: Kernel,
}

impl TrendFlexBatchBuilder {
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

	pub fn apply_slice(self, data: &[f64]) -> Result<TrendFlexBatchOutput, TrendFlexError> {
		trendflex_batch_with_kernel(data, &self.range, self.kernel)
	}
	pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<TrendFlexBatchOutput, TrendFlexError> {
		TrendFlexBatchBuilder::new().kernel(k).apply_slice(data)
	}
	pub fn apply_candles(self, c: &Candles, src: &str) -> Result<TrendFlexBatchOutput, TrendFlexError> {
		let slice = source_type(c, src);
		self.apply_slice(slice)
	}
	pub fn with_default_candles(c: &Candles) -> Result<TrendFlexBatchOutput, TrendFlexError> {
		TrendFlexBatchBuilder::new()
			.kernel(Kernel::Auto)
			.apply_candles(c, "close")
	}
}

pub fn trendflex_batch_with_kernel(
	data: &[f64],
	sweep: &TrendFlexBatchRange,
	k: Kernel,
) -> Result<TrendFlexBatchOutput, TrendFlexError> {
	// Coerce non-batch kernels to their batch equivalents
	let kernel = match k {
		Kernel::Auto => detect_best_batch_kernel(),
		Kernel::Scalar => Kernel::ScalarBatch,
		Kernel::Avx2 => Kernel::Avx2Batch,
		Kernel::Avx512 => Kernel::Avx512Batch,
		other if other.is_batch() => other,
		_ => Kernel::ScalarBatch, // Fallback for any unexpected kernel
	};

	let simd = match kernel {
		Kernel::Avx512Batch => Kernel::Avx512,
		Kernel::Avx2Batch => Kernel::Avx2,
		Kernel::ScalarBatch => Kernel::Scalar,
		_ => unreachable!(),
	};
	trendflex_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct TrendFlexBatchOutput {
	pub values: Vec<f64>,
	pub combos: Vec<TrendFlexParams>,
	pub rows: usize,
	pub cols: usize,
}

impl TrendFlexBatchOutput {
	pub fn row_for_params(&self, p: &TrendFlexParams) -> Option<usize> {
		self.combos
			.iter()
			.position(|c| c.period.unwrap_or(20) == p.period.unwrap_or(20))
	}
	pub fn values_for(&self, p: &TrendFlexParams) -> Option<&[f64]> {
		self.row_for_params(p).map(|row| {
			let start = row * self.cols;
			&self.values[start..start + self.cols]
		})
	}
}

#[inline(always)]
fn expand_grid(r: &TrendFlexBatchRange) -> Vec<TrendFlexParams> {
	fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
		if step == 0 || start == end {
			return vec![start];
		}
		(start..=end).step_by(step).collect()
	}

	let periods = axis_usize(r.period);

	let mut out = Vec::with_capacity(periods.len());
	for &p in &periods {
		out.push(TrendFlexParams { period: Some(p) });
	}
	out
}

#[inline(always)]
pub fn trendflex_batch_slice(
	data: &[f64],
	sweep: &TrendFlexBatchRange,
	kern: Kernel,
) -> Result<TrendFlexBatchOutput, TrendFlexError> {
	trendflex_batch_inner(data, sweep, kern, false)
}
#[inline(always)]
pub fn trendflex_batch_par_slice(
	data: &[f64],
	sweep: &TrendFlexBatchRange,
	kern: Kernel,
) -> Result<TrendFlexBatchOutput, TrendFlexError> {
	trendflex_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn trendflex_batch_inner(
	data: &[f64],
	sweep: &TrendFlexBatchRange,
	kern: Kernel,
	parallel: bool,
) -> Result<TrendFlexBatchOutput, TrendFlexError> {
	let combos = expand_grid(sweep);
	if combos.is_empty() {
		return Err(TrendFlexError::NoDataProvided);
	}
	let first = data
		.iter()
		.position(|x| !x.is_nan())
		.ok_or(TrendFlexError::AllValuesNaN)?;
	let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
	if data.len() - first < max_p {
		return Err(TrendFlexError::TrendFlexPeriodExceedsData {
			period: max_p,
			data_len: data.len() - first,
		});
	}
	let rows = combos.len();
	let cols = data.len();

	let warm: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap()).collect();
	let mut raw = make_uninit_matrix(rows, cols);

	// 2. write NaN prefixes for each row *before* any heavy work starts
	unsafe {
		init_matrix_prefixes(&mut raw, cols, &warm);
	}

	// Resolve Auto kernel to actual kernel
	let actual_kern = match kern {
		Kernel::Auto => detect_best_batch_kernel(),
		k => k,
	};

	// 3. helper that fills **one row**; receives &mut [MaybeUninit<f64>]
	let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
		let period = combos[row].period.unwrap();

		// Cast this single row to &mut [f64] so the existing row helpers work
		let out_row = core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

		match actual_kern {
			Kernel::Scalar | Kernel::ScalarBatch => trendflex_row_scalar(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx2 | Kernel::Avx2Batch => trendflex_row_avx2(data, first, period, out_row),
			#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
			Kernel::Avx512 | Kernel::Avx512Batch => trendflex_row_avx512(data, first, period, out_row),
			#[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
			Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => trendflex_row_scalar(data, first, period, out_row),
			Kernel::Auto => unreachable!("Auto kernel should have been resolved"),
		}
	};

	// 4. run every row, writing directly into `raw`
	if parallel {
		#[cfg(not(target_arch = "wasm32"))]
		{
			raw.par_chunks_mut(cols)
				.enumerate()
				.for_each(|(row, slice)| do_row(row, slice));
		}

		#[cfg(target_arch = "wasm32")]
		{
			for (row, slice) in raw.chunks_mut(cols).enumerate() {
				do_row(row, slice);
			}
		}
	} else {
		for (row, slice) in raw.chunks_mut(cols).enumerate() {
			do_row(row, slice);
		}
	}

	// 5. all elements are now initialised → convert to Vec<f64> using ManuallyDrop + from_raw_parts
	use core::mem::ManuallyDrop;
	let mut guard = ManuallyDrop::new(raw);
	let values: Vec<f64> = unsafe {
		Vec::from_raw_parts(guard.as_mut_ptr() as *mut f64, guard.len(), guard.capacity())
	};

	Ok(TrendFlexBatchOutput {
		values,
		combos,
		rows,
		cols,
	})
}

// Row functions -- AVX variants are just stubs to scalar
#[inline(always)]
unsafe fn trendflex_row_scalar(data: &[f64], first: usize, period: usize, out_row: &mut [f64]) {
	let ss_period = ((period as f64) / 2.0).round() as usize;
	let _ = trendflex_scalar_into(data, period, ss_period, first, out_row);
	// NaN prefixes per-row are already set by init_matrix_prefixes before this call.
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn trendflex_row_avx2(data: &[f64], first: usize, period: usize, out_row: &mut [f64]) {
	let ss_period = ((period as f64) / 2.0).round() as usize;
	let _ = trendflex_avx2_into(data, period, ss_period, first, out_row);
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn trendflex_row_avx512(data: &[f64], first: usize, period: usize, out_row: &mut [f64]) {
	let ss_period = ((period as f64) / 2.0).round() as usize;
	let _ = trendflex_avx512_into(data, period, ss_period, first, out_row);
}

// Test coverage -- use alma.rs style macros and patterns

#[cfg(test)]
mod tests {
	use super::*;
	use crate::skip_if_unsupported;
	use crate::utilities::data_loader::read_candles_from_csv;

	fn check_trendflex_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let default_params = TrendFlexParams { period: None };
		let input = TrendFlexInput::from_candles(&candles, "close", default_params);
		let output = trendflex_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());

		Ok(())
	}

	fn check_trendflex_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let params = TrendFlexParams { period: Some(20) };
		let input = TrendFlexInput::from_candles(&candles, "close", params);
		let result = trendflex_with_kernel(&input, kernel)?;
		let expected_last_five = [
			-0.19724678008015128,
			-0.1238001236481444,
			-0.10515389737087717,
			-0.1149541079904878,
			-0.16006869484450567,
		];
		let start = result.values.len().saturating_sub(5);
		for (i, &val) in result.values[start..].iter().enumerate() {
			let diff = (val - expected_last_five[i]).abs();
			assert!(
				diff < 1e-8,
				"[{}] TrendFlex {:?} mismatch at idx {}: got {}, expected {}",
				test_name,
				kernel,
				i,
				val,
				expected_last_five[i]
			);
		}
		Ok(())
	}

	fn check_trendflex_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let input = TrendFlexInput::with_default_candles(&candles);
		match input.data {
			TrendFlexData::Candles { source, .. } => assert_eq!(source, "close"),
			_ => panic!("Expected TrendFlexData::Candles"),
		}
		let output = trendflex_with_kernel(&input, kernel)?;
		assert_eq!(output.values.len(), candles.close.len());

		Ok(())
	}

	fn check_trendflex_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let input_data = [10.0, 20.0, 30.0];
		let params = TrendFlexParams { period: Some(0) };
		let input = TrendFlexInput::from_slice(&input_data, params);
		let res = trendflex_with_kernel(&input, kernel);
		assert!(res.is_err(), "[{}] TrendFlex should fail with zero period", test_name);
		Ok(())
	}

	fn check_trendflex_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let data_small = [10.0, 20.0, 30.0];
		let params = TrendFlexParams { period: Some(10) };
		let input = TrendFlexInput::from_slice(&data_small, params);
		let res = trendflex_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] TrendFlex should fail with period exceeding length",
			test_name
		);
		Ok(())
	}

	fn check_trendflex_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let single_point = [42.0];
		let params = TrendFlexParams { period: Some(9) };
		let input = TrendFlexInput::from_slice(&single_point, params);
		let res = trendflex_with_kernel(&input, kernel);
		assert!(
			res.is_err(),
			"[{}] TrendFlex should fail with insufficient data",
			test_name
		);
		Ok(())
	}

	fn check_trendflex_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let first_params = TrendFlexParams { period: Some(20) };
		let first_input = TrendFlexInput::from_candles(&candles, "close", first_params);
		let first_result = trendflex_with_kernel(&first_input, kernel)?;

		let second_params = TrendFlexParams { period: Some(10) };
		let second_input = TrendFlexInput::from_slice(&first_result.values, second_params);
		let second_result = trendflex_with_kernel(&second_input, kernel)?;

		assert_eq!(second_result.values.len(), first_result.values.len());
		if second_result.values.len() > 240 {
			for (i, &val) in second_result.values[240..].iter().enumerate() {
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

	fn check_trendflex_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);
		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let input = TrendFlexInput::from_candles(&candles, "close", TrendFlexParams { period: Some(20) });
		let res = trendflex_with_kernel(&input, kernel)?;
		assert_eq!(res.values.len(), candles.close.len());
		if res.values.len() > 240 {
			for (i, &val) in res.values[240..].iter().enumerate() {
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

	fn check_trendflex_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		let period = 20;

		let input = TrendFlexInput::from_candles(&candles, "close", TrendFlexParams { period: Some(period) });
		let batch_output = trendflex_with_kernel(&input, kernel)?.values;

		let mut stream = TrendFlexStream::try_new(TrendFlexParams { period: Some(period) })?;

		let mut stream_values = Vec::with_capacity(candles.close.len());
		for &price in &candles.close {
			match stream.update(price) {
				Some(tf_val) => stream_values.push(tf_val),
				None => stream_values.push(f64::NAN),
			}
		}

		assert_eq!(batch_output.len(), stream_values.len());
		for (i, (&b, &s)) in batch_output.iter().zip(stream_values.iter()).enumerate() {
			if b.is_nan() && s.is_nan() {
				continue;
			}
			let diff = (b - s).abs();
			assert!(
				diff < 1e-9,
				"[{}] TrendFlex streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
				test_name,
				i,
				b,
				s,
				diff
			);
		}
		Ok(())
	}

	macro_rules! generate_all_trendflex_tests {
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
	fn check_trendflex_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test_name);

		let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let candles = read_candles_from_csv(file_path)?;

		// Test with multiple parameter combinations to better catch uninitialized memory bugs
		let test_periods = vec![5, 10, 20, 30, 50, 80, 100, 150];

		for &period in &test_periods {
			let params = TrendFlexParams { period: Some(period) };
			let input = TrendFlexInput::from_candles(&candles, "close", params);

			// Skip if we don't have enough data for this period
			if candles.close.len() < period {
				continue;
			}

			let output = match trendflex_with_kernel(&input, kernel) {
				Ok(o) => o,
				Err(_) => continue, // Skip if this period causes an error
			};

			// Check every value for poison patterns
			for (i, &val) in output.values.iter().enumerate() {
				// Skip NaN values as they're expected in the warmup period
				if val.is_nan() {
					continue;
				}

				let bits = val.to_bits();

				// Check for alloc_with_nan_prefix poison (0x11111111_11111111)
				if bits == 0x11111111_11111111 {
					panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} with period {}",
						test_name, val, bits, i, period
					);
				}

				// Check for init_matrix_prefixes poison (0x22222222_22222222)
				if bits == 0x22222222_22222222 {
					panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} with period {}",
						test_name, val, bits, i, period
					);
				}

				// Check for make_uninit_matrix poison (0x33333333_33333333)
				if bits == 0x33333333_33333333 {
					panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} with period {}",
						test_name, val, bits, i, period
					);
				}
			}
		}

		Ok(())
	}

	// Release mode stub - does nothing
	#[cfg(not(debug_assertions))]
	fn check_trendflex_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(())
	}

	#[cfg(feature = "proptest")]
	#[allow(clippy::float_cmp)]
	fn check_trendflex_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
		use proptest::prelude::*;
		skip_if_unsupported!(kernel, test_name);

		let strat = (1usize..=64)
			.prop_flat_map(|period| {
				(
					prop::collection::vec(
						(-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
						period..400,
					),
					Just(period),
				)
			});

		proptest::test_runner::TestRunner::default()
			.run(&strat, |(data, period)| {
				let input = TrendFlexInput::from_slice(&data, TrendFlexParams { period: Some(period) });
				let output = trendflex_with_kernel(&input, kernel)?;

				// Property 1: Output length matches input
				prop_assert_eq!(
					output.values.len(),
					data.len(),
					"Output length mismatch"
				);

				// Find first non-NaN value in data
				let first = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
				let warmup = first + period;

				// Property 2: Warmup period - values before warmup should be NaN
				for i in 0..warmup.min(data.len()) {
					prop_assert!(
						output.values[i].is_nan(),
						"Expected NaN in warmup period at index {}, got {}",
						i,
						output.values[i]
					);
				}

				// Property 3: Finite values after warmup
				for i in warmup..output.values.len() {
					prop_assert!(
						output.values[i].is_finite(),
						"Output at index {} is not finite: {}",
						i,
						output.values[i]
					);
				}

				// Property 4: Scale invariance - due to normalization, scaling input shouldn't dramatically change output
				// Test with a scale factor of 10
				if data.len() > warmup + 10 {
					let scale_factor = 10.0;
					let scaled_data: Vec<f64> = data.iter().map(|&x| x * scale_factor).collect();
					let scaled_input = TrendFlexInput::from_slice(&scaled_data, TrendFlexParams { period: Some(period) });
					let scaled_output = trendflex_with_kernel(&scaled_input, kernel)?;
					
					// Compare normalized outputs - they should be very similar
					let mut similarity_count = 0;
					let mut total_compared = 0;
					for i in warmup..output.values.len() {
						if output.values[i].is_finite() && scaled_output.values[i].is_finite() {
							// Allow for some numerical differences, but shapes should be similar
							let diff = (output.values[i] - scaled_output.values[i]).abs();
							// TrendFlex is normalized, so outputs should be very close
							if diff < 0.5 {
								similarity_count += 1;
							}
							total_compared += 1;
						}
					}
					
					if total_compared > 0 {
						let similarity_ratio = similarity_count as f64 / total_compared as f64;
						prop_assert!(
							similarity_ratio > 0.9,
							"Scale invariance failed: only {:.1}% of values are similar after scaling",
							similarity_ratio * 100.0
						);
					}
				}

				// Property 5: Trend response - monotonic sequences should produce appropriate signed values
				if data.len() > warmup + 20 {
					// Check if we have a monotonic increasing sequence
					let mut is_increasing = true;
					let mut is_decreasing = true;
					for i in (warmup + 1)..data.len().min(warmup + 50) {
						if data[i] <= data[i - 1] {
							is_increasing = false;
						}
						if data[i] >= data[i - 1] {
							is_decreasing = false;
						}
					}
					
					// For strong trends, TrendFlex should respond appropriately
					if is_increasing {
						// Count positive values in the output after warmup
						let positive_count = output.values[warmup..]
							.iter()
							.filter(|&&v| v > 0.0)
							.count();
						let total = output.values.len() - warmup;
						let positive_ratio = positive_count as f64 / total as f64;
						prop_assert!(
							positive_ratio > 0.7,
							"Increasing trend should produce mostly positive values, got {:.1}% positive",
							positive_ratio * 100.0
						);
					} else if is_decreasing {
						// Count negative values in the output after warmup
						let negative_count = output.values[warmup..]
							.iter()
							.filter(|&&v| v < 0.0)
							.count();
						let total = output.values.len() - warmup;
						let negative_ratio = negative_count as f64 / total as f64;
						prop_assert!(
							negative_ratio > 0.7,
							"Decreasing trend should produce mostly negative values, got {:.1}% negative",
							negative_ratio * 100.0
						);
					}
				}

				// Property 6: Constant input produces values near zero
				// Since there's no trend in constant data, TrendFlex should converge near 0
				let all_same = data[first..].windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10);
				if all_same && data.len() > warmup + 10 {
					let last_values = &output.values[(data.len() - 5)..];
					for val in last_values {
						prop_assert!(
							val.abs() < 0.1,
							"Constant input should produce values near 0, got {}",
							val
						);
					}
				}

				// Property 7: Period = 1 special case
				if period == 1 {
					// With period=1, super smoother period = round(1/2) = 1
					// Values should still be finite after warmup
					for i in (first + 1)..output.values.len() {
						prop_assert!(
							output.values[i].is_finite(),
							"Period=1 should still produce finite values at index {}",
							i
						);
					}
				}

				// Property 8: Large period behavior
				// When period is close to data length, should still produce valid output
				if data.len() > 5 && period >= data.len().saturating_sub(5) && data.len() > period {
					// Should have at least some non-NaN values at the end
					let last_idx = data.len() - 1;
					if last_idx >= warmup {
						prop_assert!(
							output.values[last_idx].is_finite(),
							"Large period should still produce finite values at the end"
						);
					}
				}

				// Property 9: Kernel consistency
				if cfg!(all(feature = "nightly-avx", target_arch = "x86_64")) {
					// Test that different kernels produce identical results
					let scalar_output = trendflex_with_kernel(&input, Kernel::Scalar)?;
					
					for i in 0..output.values.len() {
						if output.values[i].is_finite() && scalar_output.values[i].is_finite() {
							prop_assert!(
								(output.values[i] - scalar_output.values[i]).abs() < 1e-9,
								"Kernel consistency failed at index {}: {} vs {}",
								i,
								output.values[i],
								scalar_output.values[i]
							);
						} else {
							prop_assert_eq!(
								output.values[i].is_nan(),
								scalar_output.values[i].is_nan(),
								"NaN mismatch between kernels at index {}",
								i
							);
						}
					}
				}

				Ok(())
			})
			.map_err(|e| e.into())
	}

	#[cfg(feature = "proptest")]
	generate_all_trendflex_tests!(check_trendflex_property);

	// Test for trendflex_into_slice validation (prevents panic when period > data.len())
	#[test]
	fn test_trendflex_into_slice_validation() {
		// Test case 1: period exceeds data length
		let data = vec![1.0, 2.0, 3.0];
		let params = TrendFlexParams { period: Some(10) };
		let input = TrendFlexInput::from_slice(&data, params);
		let mut out = vec![0.0; data.len()];
		
		let result = trendflex_into_slice(&mut out, &input, Kernel::Scalar);
		assert!(result.is_err());
		match result {
			Err(TrendFlexError::TrendFlexPeriodExceedsData { period, data_len }) => {
				assert_eq!(period, 10);
				assert_eq!(data_len, 3);
			}
			_ => panic!("Expected TrendFlexPeriodExceedsData error"),
		}
		
		// Test case 2: empty data
		let empty_data: Vec<f64> = vec![];
		let params = TrendFlexParams { period: Some(5) };
		let input = TrendFlexInput::from_slice(&empty_data, params);
		let mut out = vec![];
		
		let result = trendflex_into_slice(&mut out, &input, Kernel::Scalar);
		assert!(result.is_err());
		match result {
			Err(TrendFlexError::NoDataProvided) => {},
			_ => panic!("Expected NoDataProvided error"),
		}
		
		// Test case 3: zero period
		let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
		let params = TrendFlexParams { period: Some(0) };
		let input = TrendFlexInput::from_slice(&data, params);
		let mut out = vec![0.0; data.len()];
		
		let result = trendflex_into_slice(&mut out, &input, Kernel::Scalar);
		assert!(result.is_err());
		match result {
			Err(TrendFlexError::ZeroTrendFlexPeriod { period }) => {
				assert_eq!(period, 0);
			}
			_ => panic!("Expected ZeroTrendFlexPeriod error"),
		}
		
		// Test case 4: valid input should work
		let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
		let params = TrendFlexParams { period: Some(3) };
		let input = TrendFlexInput::from_slice(&data, params);
		let mut out = vec![0.0; data.len()];
		
		let result = trendflex_into_slice(&mut out, &input, Kernel::Scalar);
		assert!(result.is_ok());
	}

	// Test for batch kernel coercion
	#[test]
	fn test_trendflex_batch_kernel_coercion() {
		let data = vec![1.0; 50];
		let sweep = TrendFlexBatchRange { period: (5, 10, 1) };
		
		// Test that non-batch kernels are coerced to batch kernels
		let result_scalar = trendflex_batch_with_kernel(&data, &sweep, Kernel::Scalar);
		assert!(result_scalar.is_ok());
		
		#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
		{
			let result_avx2 = trendflex_batch_with_kernel(&data, &sweep, Kernel::Avx2);
			assert!(result_avx2.is_ok());
			
			let result_avx512 = trendflex_batch_with_kernel(&data, &sweep, Kernel::Avx512);
			assert!(result_avx512.is_ok());
		}
		
		// Test that batch kernels still work
		let result_scalar_batch = trendflex_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch);
		assert!(result_scalar_batch.is_ok());
	}

	generate_all_trendflex_tests!(
		check_trendflex_partial_params,
		check_trendflex_accuracy,
		check_trendflex_default_candles,
		check_trendflex_zero_period,
		check_trendflex_period_exceeds_length,
		check_trendflex_very_small_dataset,
		check_trendflex_reinput,
		check_trendflex_nan_handling,
		check_trendflex_streaming,
		check_trendflex_no_poison
	);

	fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		let output = TrendFlexBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;

		let def = TrendFlexParams::default();
		let row = output.values_for(&def).expect("default row missing");

		assert_eq!(row.len(), c.close.len());

		let expected = [
			-0.19724678008015128,
			-0.1238001236481444,
			-0.10515389737087717,
			-0.1149541079904878,
			-0.16006869484450567,
		];
		let start = row.len() - 5;
		for (i, &v) in row[start..].iter().enumerate() {
			assert!(
				(v - expected[i]).abs() < 1e-8,
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
	fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
		skip_if_unsupported!(kernel, test);

		let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
		let c = read_candles_from_csv(file)?;

		// Test multiple batch configurations with different period ranges
		let test_configs = vec![
			(5, 20, 3),    // Small periods with fine steps
			(10, 50, 5),   // Medium periods
			(20, 100, 10), // Large periods
			(30, 120, 15), // Very large periods
			(7, 7, 1),     // Single small period
			(80, 80, 1),   // Single large period
			(15, 45, 5),   // Medium range
		];

		for (start, end, step) in test_configs {
			let output = TrendFlexBatchBuilder::new()
				.kernel(kernel)
				.period_range(start, end, step)
				.apply_candles(&c, "close")?;

			// Check every value in the entire batch matrix for poison patterns
			for (idx, &val) in output.values.iter().enumerate() {
				// Skip NaN values as they're expected in warmup periods
				if val.is_nan() {
					continue;
				}

				let bits = val.to_bits();
				let row = idx / output.cols;
				let col = idx % output.cols;
				let period = output.combos.get(row).map(|p| p.period.unwrap_or(0)).unwrap_or(0);

				// Check for alloc_with_nan_prefix poison (0x11111111_11111111)
				if bits == 0x11111111_11111111 {
					panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (period {}, flat index {})",
                        test, val, bits, row, col, period, idx
                    );
				}

				// Check for init_matrix_prefixes poison (0x22222222_22222222)
				if bits == 0x22222222_22222222 {
					panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (period {}, flat index {})",
                        test, val, bits, row, col, period, idx
                    );
				}

				// Check for make_uninit_matrix poison (0x33333333_33333333)
				if bits == 0x33333333_33333333 {
					panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (period {}, flat index {})",
                        test, val, bits, row, col, period, idx
                    );
				}
			}
		}

		Ok(())
	}

	// Release mode stub - does nothing
	#[cfg(not(debug_assertions))]
	fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
		Ok(())
	}

	gen_batch_tests!(check_batch_default_row);
	gen_batch_tests!(check_batch_no_poison);
}

#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pyfunction(name = "trendflex")]
#[pyo3(signature = (data, period=None, kernel=None))]
/// Compute the Trend Flex Filter (TrendFlex) of the input data.
///
/// Highlights momentum shifts using a super smoother and volatility measurement.
/// Adapts to market volatility, amplifying or dampening its reaction accordingly.
///
/// Parameters:
/// -----------
/// data : np.ndarray
///     Input data array (float64).
/// period : int, optional
///     Primary lookback period (default: 20).
/// kernel : str, optional
///     Computation kernel to use: 'auto', 'scalar', 'avx2', 'avx512'.
///     Default is 'auto' which auto-detects the best available.
///
/// Returns:
/// --------
/// np.ndarray
///     Array of TrendFlex values, same length as input.
///
/// Raises:
/// -------
/// ValueError
///     If inputs are invalid (period = 0, period > data length, etc).
pub fn trendflex_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	period: Option<usize>,
	kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
	use numpy::{IntoPyArray, PyArrayMethods};

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, false)?;

	// Build input struct
	let params = TrendFlexParams { period };
	let trendflex_in = TrendFlexInput::from_slice(slice_in, params);

	// Get Vec<f64> from Rust function
	let result_vec: Vec<f64> = py
		.allow_threads(|| trendflex_with_kernel(&trendflex_in, kern).map(|o| o.values))
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	// Zero-copy transfer to NumPy
	Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "TrendFlexStream")]
pub struct TrendFlexStreamPy {
	stream: TrendFlexStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl TrendFlexStreamPy {
	#[new]
	fn new(period: Option<usize>) -> PyResult<Self> {
		let params = TrendFlexParams { period };
		let stream = TrendFlexStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
		Ok(TrendFlexStreamPy { stream })
	}

	/// Updates the stream with a new value and returns the calculated TrendFlex value.
	/// Returns `None` if the buffer is not yet full.
	fn update(&mut self, value: f64) -> Option<f64> {
		self.stream.update(value)
	}
}

#[cfg(feature = "python")]
#[pyfunction(name = "trendflex_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
/// Compute TrendFlex for multiple period values in a single pass.
///
/// Parameters:
/// -----------
/// data : np.ndarray
///     Input data array (float64).
/// period_range : tuple
///     (start, end, step) for period values to compute.
/// kernel : str, optional
///     Computation kernel: 'auto', 'scalar', 'avx2', 'avx512'.
///     Default is 'auto' which auto-detects the best available.
///
/// Returns:
/// --------
/// dict
///     Dictionary with 'values' (2D array) and 'periods' array.
pub fn trendflex_batch_py<'py>(
	py: Python<'py>,
	data: numpy::PyReadonlyArray1<'py, f64>,
	period_range: (usize, usize, usize),
	kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
	use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
	use pyo3::types::PyDict;

	let slice_in = data.as_slice()?;
	let kern = validate_kernel(kernel, true)?;  // true for batch operations

	let sweep = TrendFlexBatchRange { period: period_range };
	
	// Calculate dimensions
	let combos = expand_grid(&sweep);
	let rows = combos.len();
	let cols = slice_in.len();
	
	// Pre-allocate NumPy array (like ALMA does)
	let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
	let slice_out = unsafe { out_arr.as_slice_mut()? };

	// Heavy work without the GIL
	let combos = py
		.allow_threads(|| -> Result<Vec<TrendFlexParams>, TrendFlexError> {
			// Handle kernel selection for batch operations
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
			
			// Use the new _batch_inner_into function
			trendflex_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
		})
		.map_err(|e| PyValueError::new_err(e.to_string()))?;

	// Build output dict
	let dict = PyDict::new(py);
	dict.set_item("values", out_arr.reshape((rows, cols))?)?;

	// Extract periods from combos using into_pyarray() for zero-copy
	dict.set_item(
		"periods",
		combos
			.iter()
			.map(|p| p.period.unwrap_or(20) as u64)
			.collect::<Vec<_>>()
			.into_pyarray(py),
	)?;

	Ok(dict)
}

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct TrendFlexBatchConfig {
	pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct TrendFlexBatchJsOutput {
	pub values: Vec<f64>,
	pub combos: Vec<TrendFlexParams>,
	pub rows: usize,
	pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
/// Compute the Trend Flex Filter (TrendFlex) of the input data.
///
/// # Arguments
/// * `data` - Input data array
/// * `period` - Primary lookback period
///
/// # Returns
/// Array of TrendFlex values, same length as input
pub fn trendflex_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
	let params = TrendFlexParams { period: Some(period) };
	let input = TrendFlexInput::from_slice(data, params);

	trendflex_with_kernel(&input, Kernel::Auto)
		.map(|o| o.values)
		.map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
/// Compute TrendFlex for multiple period values in a single pass.
///
/// # Arguments
/// * `data` - Input data array
/// * `period_start`, `period_end`, `period_step` - Period range parameters
///
/// # Returns
/// Flattened array of values (row-major order)
pub fn trendflex_batch_js(
	data: &[f64],
	period_start: usize,
	period_end: usize,
	period_step: usize,
) -> Result<Vec<f64>, JsValue> {
	let sweep = TrendFlexBatchRange {
		period: (period_start, period_end, period_step),
	};

	// Use the existing batch function with parallel=false for WASM
	trendflex_batch_inner(data, &sweep, Kernel::Auto, false)
		.map(|output| output.values)
		.map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
/// Get metadata about batch computation.
///
/// # Arguments
/// * Period range parameters (same as trendflex_batch_js)
///
/// # Returns
/// Array containing period values
pub fn trendflex_batch_metadata_js(
	period_start: usize,
	period_end: usize,
	period_step: usize,
) -> Result<Vec<f64>, JsValue> {
	let sweep = TrendFlexBatchRange {
		period: (period_start, period_end, period_step),
	};

	let combos = expand_grid(&sweep);
	let metadata: Vec<f64> = combos.iter().map(|combo| combo.period.unwrap_or(20) as f64).collect();

	Ok(metadata)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = trendflex_batch)]
pub fn trendflex_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
	let config: TrendFlexBatchConfig =
		serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

	let sweep = TrendFlexBatchRange {
		period: config.period_range,
	};

	let output = trendflex_batch_inner(data, &sweep, Kernel::Auto, false).map_err(|e| JsValue::from_str(&e.to_string()))?;

	let js_output = TrendFlexBatchJsOutput {
		values: output.values,
		combos: output.combos,
		rows: output.rows,
		cols: output.cols,
	};

	serde_wasm_bindgen::to_value(&js_output).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn trendflex_alloc(len: usize) -> *mut f64 {
	let mut vec = Vec::<f64>::with_capacity(len);
	let ptr = vec.as_mut_ptr();
	std::mem::forget(vec);
	ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn trendflex_free(ptr: *mut f64, len: usize) {
	unsafe {
		let _ = Vec::from_raw_parts(ptr, len, len);
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn trendflex_into(
	in_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period: usize,
) -> Result<(), JsValue> {
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("null pointer passed to trendflex_into"));
	}
	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);
		if period == 0 || period >= len {
			return Err(JsValue::from_str("Invalid period"));
		}
		let input = TrendFlexInput::from_slice(data, TrendFlexParams { period: Some(period) });
		if in_ptr == out_ptr {
			let mut tmp = vec![0.0; len];
			trendflex_into_slice(&mut tmp, &input, detect_best_kernel())
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
			std::slice::from_raw_parts_mut(out_ptr, len).copy_from_slice(&tmp);
		} else {
			let out = std::slice::from_raw_parts_mut(out_ptr, len);
			trendflex_into_slice(out, &input, detect_best_kernel())
				.map_err(|e| JsValue::from_str(&e.to_string()))?;
		}
		Ok(())
	}
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn trendflex_batch_into(
	in_ptr: *const f64,
	out_ptr: *mut f64,
	len: usize,
	period_start: usize,
	period_end: usize,
	period_step: usize,
) -> Result<usize, JsValue> {
	if in_ptr.is_null() || out_ptr.is_null() {
		return Err(JsValue::from_str("null pointer passed to trendflex_batch_into"));
	}

	unsafe {
		let data = std::slice::from_raw_parts(in_ptr, len);

		// Build the batch sweep
		let sweep = TrendFlexBatchRange {
			period: (period_start, period_end, period_step),
		};

		// Calculate the number of combinations
		let combos = expand_grid(&sweep);
		let n_combos = combos.len();
		let total_size = n_combos * len;

		// Get output slice
		let out_slice = std::slice::from_raw_parts_mut(out_ptr, total_size);

		// Use trendflex_batch_inner_into to write directly
		trendflex_batch_inner_into(data, &sweep, Kernel::Auto, false, out_slice)
			.map_err(|e| JsValue::from_str(&e.to_string()))?;

		Ok(n_combos)
	}
}
