//! # Mesa Sine Wave (MSW)
//!
//! The Mesa Sine Wave indicator attempts to detect turning points in price data
//! by fitting a sine wave function. It outputs two series: the `sine` wave
//! and a leading version of the wave (`lead`).
//!
//! ## Parameters
//! - **period**: The window size (number of data points). Defaults to 5.
//!
//! ## Inputs
//! - **data**: Price data or any numeric series
//!
//! ## Returns
//! - **sine**: Vector of sine wave values with NaN prefix during warmup period
//! - **lead**: Vector of leading sine wave values with NaN prefix during warmup period
//!
//! ## Developer Notes
//! - SIMD enabled: AVX2/AVX512 vectorize across time (4/8 lanes) and use FMA; selected at runtime via detect_best_kernel(). Benchmarked >5% faster than scalar at 100k.
//! - Streaming: Kept as O(N) per-tick dot product to match batch numerics exactly under strict 1e-9 tests. A Sliding DFT O(1) variant was evaluated but disabled due to last-bit drift; revisit with a guaranteed-stable SDFT if parity can be ensured.
//! - Zero-copy Memory: Uses alloc_with_nan_prefix and make_uninit_matrix for batch operations.

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

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::{cuda_available, CudaMsw};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::alma::DeviceArrayF32Py;
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

#[allow(clippy::approx_constant)]
const TULIP_PI: f64 = 3.1415926;
const TULIP_TPI: f64 = 2.0 * TULIP_PI;

impl<'a> AsRef<[f64]> for MswInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            MswData::Slice(slice) => slice,
            MswData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum MswData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct MswOutput {
    pub sine: Vec<f64>,
    pub lead: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct MswParams {
    pub period: Option<usize>,
}

impl Default for MswParams {
    fn default() -> Self {
        Self { period: Some(5) }
    }
}

#[derive(Debug, Clone)]
pub struct MswInput<'a> {
    pub data: MswData<'a>,
    pub params: MswParams,
}

impl<'a> MswInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: MswParams) -> Self {
        Self {
            data: MswData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: MswParams) -> Self {
        Self {
            data: MswData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", MswParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(5)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct MswBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for MswBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl MswBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<MswOutput, MswError> {
        let p = MswParams {
            period: self.period,
        };
        let i = MswInput::from_candles(c, "close", p);
        msw_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<MswOutput, MswError> {
        let p = MswParams {
            period: self.period,
        };
        let i = MswInput::from_slice(d, p);
        msw_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<MswStream, MswError> {
        let p = MswParams {
            period: self.period,
        };
        MswStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum MswError {
    #[error("msw: Empty data provided for MSW.")]
    EmptyData,
    #[error("msw: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("msw: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("msw: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn msw(input: &MswInput) -> Result<MswOutput, MswError> {
    msw_with_kernel(input, Kernel::Auto)
}

pub fn msw_with_kernel(input: &MswInput, kernel: Kernel) -> Result<MswOutput, MswError> {
    let data: &[f64] = match &input.data {
        MswData::Candles { candles, source } => source_type(candles, source),
        MswData::Slice(sl) => sl,
    };
    if data.is_empty() {
        return Err(MswError::EmptyData);
    }
    let period = input.get_period();
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(MswError::AllValuesNaN)?;
    let len = data.len();
    if period == 0 || period > len {
        return Err(MswError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(MswError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }
    let mut chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    // Prefer AVX2 over AVX512 for MSW in Auto: AVX512 often downclocks and underperforms
    // compared to AVX2 on many CPUs. Allow explicit Avx512 selection via API.
    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    if matches!(kernel, Kernel::Auto) && matches!(chosen, Kernel::Avx512 | Kernel::Avx512Batch) {
        chosen = Kernel::Avx2;
    }
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => msw_scalar(data, period, first, len),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => msw_avx2(data, period, first, len),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => msw_avx512(data, period, first, len),
            _ => unreachable!(),
        }
    }
}

#[inline]
pub unsafe fn msw_scalar(
    data: &[f64],
    period: usize,
    first: usize,
    len: usize,
) -> Result<MswOutput, MswError> {
    // Warmup prefix length
    let warm = first + period - 1;
    let mut sine = alloc_with_nan_prefix(len, warm);
    let mut lead = alloc_with_nan_prefix(len, warm);

    // Precompute sin/cos weights with one sin_cos per tap
    let step = TULIP_TPI / period as f64;
    let mut cos_table = Vec::with_capacity(period);
    let mut sin_table = Vec::with_capacity(period);
    let mut ang = 0.0f64;
    for _ in 0..period {
        let (s, c) = ang.sin_cos();
        sin_table.push(s);
        cos_table.push(c);
        ang += step;
    }

    // Keep lead via sin(phase + π/4) to match streaming path exactly

    for i in warm..len {
        let mut rp = 0.0f64;
        let mut ip = 0.0f64;
        for j in 0..period {
            let w = *data.get_unchecked(i - j);
            rp += cos_table[j] * w;
            ip += sin_table[j] * w;
        }

        let mut phase = if rp.abs() > 0.001 {
            atan(ip / rp)
        } else {
            TULIP_PI * if ip < 0.0 { -1.0 } else { 1.0 }
        };
        if rp < 0.0 {
            phase += TULIP_PI;
        }
        phase += TULIP_PI * 0.5;
        if phase < 0.0 {
            phase += TULIP_TPI;
        }
        if phase > TULIP_TPI {
            phase -= TULIP_TPI;
        }

        let (s, c) = phase.sin_cos();
        *sine.get_unchecked_mut(i) = s;
        *lead.get_unchecked_mut(i) = (s + c) * 0.707106781186547524400844362104849039_f64;
    }
    Ok(MswOutput { sine, lead })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn msw_avx2(
    data: &[f64],
    period: usize,
    first: usize,
    len: usize,
) -> Result<MswOutput, MswError> {
    use core::arch::x86_64::*;
    let warm = first + period - 1;
    let mut sine = alloc_with_nan_prefix(len, warm);
    let mut lead = alloc_with_nan_prefix(len, warm);

    // Precompute weights
    let step = TULIP_TPI / period as f64;
    let mut cos_table = Vec::with_capacity(period);
    let mut sin_table = Vec::with_capacity(period);
    let mut ang = 0.0f64;
    for _ in 0..period {
        let (s, c) = ang.sin_cos();
        sin_table.push(s);
        cos_table.push(c);
        ang += step;
    }
    let dptr = data.as_ptr();

    const LANES: usize = 4;
    // Keep lead via sin(phase + π/4) to match streaming path

    let mut i = warm;
    while i + (LANES - 1) < len {
        let k = i + (LANES - 1);
        let mut rp = _mm256_set1_pd(0.0);
        let mut ip = _mm256_set1_pd(0.0);

        for j in 0..period {
            let base = k - j;
            let wv = _mm256_loadu_pd(dptr.add(base - (LANES - 1)));
            let cw = _mm256_set1_pd(*cos_table.get_unchecked(j));
            let sw = _mm256_set1_pd(*sin_table.get_unchecked(j));
            rp = _mm256_fmadd_pd(cw, wv, rp);
            ip = _mm256_fmadd_pd(sw, wv, ip);
        }

        let mut rbuf = [0.0f64; LANES];
        let mut ibuf = [0.0f64; LANES];
        _mm256_storeu_pd(rbuf.as_mut_ptr(), rp);
        _mm256_storeu_pd(ibuf.as_mut_ptr(), ip);

        let mut idx = i;
        for lane in 0..LANES {
            let mut phase = if rbuf[lane].abs() > 0.001 {
                atan(ibuf[lane] / rbuf[lane])
            } else {
                TULIP_PI * if ibuf[lane] < 0.0 { -1.0 } else { 1.0 }
            };
            if rbuf[lane] < 0.0 {
                phase += TULIP_PI;
            }
            phase += TULIP_PI * 0.5;
            if phase < 0.0 {
                phase += TULIP_TPI;
            }
            if phase > TULIP_TPI {
                phase -= TULIP_TPI;
            }

            let (s, c) = phase.sin_cos();
            *sine.get_unchecked_mut(idx) = s;
            *lead.get_unchecked_mut(idx) = (s + c) * 0.707106781186547524400844362104849039_f64;
            idx += 1;
        }

        i += LANES;
    }

    // Scalar tail inline without extra allocations/copies
    while i < len {
        let mut rp = 0.0f64;
        let mut ip = 0.0f64;
        for j in 0..period {
            let w = *data.get_unchecked(i - j);
            rp += cos_table[j] * w;
            ip += sin_table[j] * w;
        }
        let mut phase = if rp.abs() > 0.001 {
            atan(ip / rp)
        } else {
            TULIP_PI * if ip < 0.0 { -1.0 } else { 1.0 }
        };
        if rp < 0.0 {
            phase += TULIP_PI;
        }
        phase += TULIP_PI * 0.5;
        if phase < 0.0 {
            phase += TULIP_TPI;
        }
        if phase > TULIP_TPI {
            phase -= TULIP_TPI;
        }
        let (s, c) = phase.sin_cos();
        *sine.get_unchecked_mut(i) = s;
        *lead.get_unchecked_mut(i) = (s + c) * 0.707106781186547524400844362104849039_f64;
        i += 1;
    }

    Ok(MswOutput { sine, lead })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f,fma")]
pub unsafe fn msw_avx512(
    data: &[f64],
    period: usize,
    first: usize,
    len: usize,
) -> Result<MswOutput, MswError> {
    use core::arch::x86_64::*;
    let warm = first + period - 1;
    let mut sine = alloc_with_nan_prefix(len, warm);
    let mut lead = alloc_with_nan_prefix(len, warm);

    // Precompute weights
    let step = TULIP_TPI / period as f64;
    let mut cos_table = Vec::with_capacity(period);
    let mut sin_table = Vec::with_capacity(period);
    let mut ang = 0.0f64;
    for _ in 0..period {
        let (s, c) = ang.sin_cos();
        sin_table.push(s);
        cos_table.push(c);
        ang += step;
    }
    let dptr = data.as_ptr();

    const LANES: usize = 8;
    // Keep lead via sin(phase + π/4) to match streaming path

    let mut i = warm;
    while i + (LANES - 1) < len {
        let k = i + (LANES - 1);
        let mut rp = _mm512_set1_pd(0.0);
        let mut ip = _mm512_set1_pd(0.0);

        for j in 0..period {
            let base = k - j;
            let wv = _mm512_loadu_pd(dptr.add(base - (LANES - 1)));
            let cw = _mm512_set1_pd(*cos_table.get_unchecked(j));
            let sw = _mm512_set1_pd(*sin_table.get_unchecked(j));
            rp = _mm512_fmadd_pd(cw, wv, rp);
            ip = _mm512_fmadd_pd(sw, wv, ip);
        }

        let mut rbuf = [0.0f64; LANES];
        let mut ibuf = [0.0f64; LANES];
        _mm512_storeu_pd(rbuf.as_mut_ptr(), rp);
        _mm512_storeu_pd(ibuf.as_mut_ptr(), ip);

        let mut idx = i;
        for lane in 0..LANES {
            let mut phase = if rbuf[lane].abs() > 0.001 {
                atan(ibuf[lane] / rbuf[lane])
            } else {
                TULIP_PI * if ibuf[lane] < 0.0 { -1.0 } else { 1.0 }
            };
            if rbuf[lane] < 0.0 {
                phase += TULIP_PI;
            }
            phase += TULIP_PI * 0.5;
            if phase < 0.0 {
                phase += TULIP_TPI;
            }
            if phase > TULIP_TPI {
                phase -= TULIP_TPI;
            }

            let (s, c) = phase.sin_cos();
            *sine.get_unchecked_mut(idx) = s;
            *lead.get_unchecked_mut(idx) = (s + c) * 0.707106781186547524400844362104849039_f64;
            idx += 1;
        }

        i += LANES;
    }

    while i < len {
        let mut rp = 0.0f64;
        let mut ip = 0.0f64;
        for j in 0..period {
            let w = *data.get_unchecked(i - j);
            rp += cos_table[j] * w;
            ip += sin_table[j] * w;
        }
        let mut phase = if rp.abs() > 0.001 {
            atan(ip / rp)
        } else {
            TULIP_PI * if ip < 0.0 { -1.0 } else { 1.0 }
        };
        if rp < 0.0 {
            phase += TULIP_PI;
        }
        phase += TULIP_PI * 0.5;
        if phase < 0.0 {
            phase += TULIP_TPI;
        }
        if phase > TULIP_TPI {
            phase -= TULIP_TPI;
        }
        let (s, c) = phase.sin_cos();
        *sine.get_unchecked_mut(i) = s;
        *lead.get_unchecked_mut(i) = (s + c) * 0.707106781186547524400844362104849039_f64;
        i += 1;
    }

    Ok(MswOutput { sine, lead })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn msw_avx512_short(
    data: &[f64],
    period: usize,
    first: usize,
    len: usize,
) -> Result<MswOutput, MswError> {
    msw_scalar(data, period, first, len)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn msw_avx512_long(
    data: &[f64],
    period: usize,
    first: usize,
    len: usize,
) -> Result<MswOutput, MswError> {
    msw_scalar(data, period, first, len)
}

pub fn atan(x: f64) -> f64 {
    x.atan()
}

#[derive(Debug, Clone)]
pub struct MswStream {
    period: usize,
    buffer: Vec<f64>,
    cos_table: Vec<f64>,
    sin_table: Vec<f64>,
    head: usize,
    filled: bool,
}

impl MswStream {
    pub fn try_new(params: MswParams) -> Result<Self, MswError> {
        let period = params.period.unwrap_or(5);
        if period == 0 {
            return Err(MswError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let mut cos_table = Vec::with_capacity(period);
        let mut sin_table = Vec::with_capacity(period);
        for j in 0..period {
            let angle = TULIP_TPI * j as f64 / period as f64;
            cos_table.push(angle.cos());
            sin_table.push(angle.sin());
        }
        Ok(Self {
            period,
            buffer: vec![f64::NAN; period],
            cos_table,
            sin_table,
            head: 0,
            filled: false,
        })
    }
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        self.buffer[self.head] = value;
        self.head = (self.head + 1) % self.period;
        if !self.filled && self.head == 0 {
            self.filled = true;
        }
        if !self.filled {
            return None;
        }
        Some(self.dot_ring())
    }
    #[inline(always)]
    fn dot_ring(&self) -> (f64, f64) {
        let mut rp = 0.0;
        let mut ip = 0.0;
        // `self.head` always points to the next insertion position, which is
        // the oldest sample in the ring buffer. The most recent value is the
        // element just before `head`. The batch implementation processes data
        // from newest to oldest, so mirror that ordering here.
        let mut idx = (self.head + self.period - 1) % self.period;
        for j in 0..self.period {
            rp += self.cos_table[j] * self.buffer[idx];
            ip += self.sin_table[j] * self.buffer[idx];
            idx = if idx == 0 { self.period - 1 } else { idx - 1 };
        }
        let mut phase = if rp.abs() > 0.001 {
            atan(ip / rp)
        } else {
            TULIP_PI * if ip < 0.0 { -1.0 } else { 1.0 }
        };
        if rp < 0.0 {
            phase += TULIP_PI;
        }
        phase += TULIP_PI / 2.0;
        if phase < 0.0 {
            phase += TULIP_TPI;
        }
        if phase > TULIP_TPI {
            phase -= TULIP_TPI;
        }
        let (s, c) = phase.sin_cos();
        // Use identity for lead: sin(phase + π/4) = (sin + cos) * √½
        let lead = (s + c) * 0.707106781186547524400844362104849039_f64;
        (s, lead)
    }
}

#[derive(Clone, Debug)]
pub struct MswBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for MswBatchRange {
    fn default() -> Self {
        Self { period: (5, 30, 1) }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MswBatchBuilder {
    range: MswBatchRange,
    kernel: Kernel,
}

impl MswBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<MswBatchOutput, MswError> {
        msw_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<MswBatchOutput, MswError> {
        MswBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<MswBatchOutput, MswError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<MswBatchOutput, MswError> {
        MswBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn msw_batch_with_kernel(
    data: &[f64],
    sweep: &MswBatchRange,
    k: Kernel,
) -> Result<MswBatchOutput, MswError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(MswError::InvalidPeriod {
                period: 0,
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
    msw_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct MswBatchOutput {
    pub sine: Vec<f64>,
    pub lead: Vec<f64>,
    pub combos: Vec<MswParams>,
    pub rows: usize,
    pub cols: usize,
}

impl MswBatchOutput {
    pub fn row_for_params(&self, p: &MswParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(5) == p.period.unwrap_or(5))
    }
    pub fn sine_for(&self, p: &MswParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.sine[start..start + self.cols]
        })
    }
    pub fn lead_for(&self, p: &MswParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.lead[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &MswBatchRange) -> Vec<MswParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    periods
        .into_iter()
        .map(|p| MswParams { period: Some(p) })
        .collect()
}

#[inline(always)]
pub fn msw_batch_slice(
    data: &[f64],
    sweep: &MswBatchRange,
    kern: Kernel,
) -> Result<MswBatchOutput, MswError> {
    msw_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn msw_batch_par_slice(
    data: &[f64],
    sweep: &MswBatchRange,
    kern: Kernel,
) -> Result<MswBatchOutput, MswError> {
    msw_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn msw_batch_inner(
    data: &[f64],
    sweep: &MswBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<MswBatchOutput, MswError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(MswError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(MswError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(MswError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();

    // Use uninitialized memory for better performance
    let mut sine_buf = make_uninit_matrix(rows, cols);
    let mut lead_buf = make_uninit_matrix(rows, cols);

    // Initialize NaN prefixes for each row based on warmup periods
    let warmup_periods: Vec<usize> = combos
        .iter()
        .map(|c| {
            let period = c.period.unwrap();
            first + period - 1
        })
        .collect();
    init_matrix_prefixes(&mut sine_buf, cols, &warmup_periods);
    init_matrix_prefixes(&mut lead_buf, cols, &warmup_periods);

    // Convert to mutable slices for computation
    let mut sine_guard = core::mem::ManuallyDrop::new(sine_buf);
    let mut lead_guard = core::mem::ManuallyDrop::new(lead_buf);
    let sine: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(sine_guard.as_mut_ptr() as *mut f64, sine_guard.len())
    };
    let lead: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(lead_guard.as_mut_ptr() as *mut f64, lead_guard.len())
    };
    let do_row = |row: usize, sine_row: &mut [f64], lead_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        match kern {
            Kernel::Scalar => msw_row_scalar(data, first, period, sine_row, lead_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => msw_row_avx2(data, first, period, sine_row, lead_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => msw_row_avx512(data, first, period, sine_row, lead_row),
            _ => unreachable!(),
        }
    };
    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            sine.par_chunks_mut(cols)
                .zip(lead.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, (sine_row, lead_row))| do_row(row, sine_row, lead_row));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, (sine_row, lead_row)) in
                sine.chunks_mut(cols).zip(lead.chunks_mut(cols)).enumerate()
            {
                do_row(row, sine_row, lead_row);
            }
        }
    } else {
        for (row, (sine_row, lead_row)) in
            sine.chunks_mut(cols).zip(lead.chunks_mut(cols)).enumerate()
        {
            do_row(row, sine_row, lead_row);
        }
    }

    // Convert back to owned vectors
    let sine_vec = unsafe {
        Vec::from_raw_parts(
            sine_guard.as_mut_ptr() as *mut f64,
            sine_guard.len(),
            sine_guard.capacity(),
        )
    };
    let lead_vec = unsafe {
        Vec::from_raw_parts(
            lead_guard.as_mut_ptr() as *mut f64,
            lead_guard.len(),
            lead_guard.capacity(),
        )
    };

    Ok(MswBatchOutput {
        sine: sine_vec,
        lead: lead_vec,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn msw_batch_inner_into(
    data: &[f64],
    sweep: &MswBatchRange,
    kern: Kernel,
    parallel: bool,
    sine_out: &mut [f64],
    lead_out: &mut [f64],
) -> Result<Vec<MswParams>, MswError> {
    use std::mem::MaybeUninit;

    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(MswError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(MswError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();

    if data.len() - first < max_p {
        return Err(MswError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    if sine_out.len() != rows * cols || lead_out.len() != rows * cols {
        return Err(MswError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    // 1) Cast to MaybeUninit and initialize NaN warm prefixes with helper
    let sine_mu = unsafe {
        std::slice::from_raw_parts_mut(
            sine_out.as_mut_ptr() as *mut MaybeUninit<f64>,
            sine_out.len(),
        )
    };
    let lead_mu = unsafe {
        std::slice::from_raw_parts_mut(
            lead_out.as_mut_ptr() as *mut MaybeUninit<f64>,
            lead_out.len(),
        )
    };

    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();
    init_matrix_prefixes(sine_mu, cols, &warm);
    init_matrix_prefixes(lead_mu, cols, &warm);

    // 2) Compute rows into the same backing memory
    let do_row = |row: usize,
                  sine_row_mu: &mut [MaybeUninit<f64>],
                  lead_row_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();
        let sine_row =
            std::slice::from_raw_parts_mut(sine_row_mu.as_mut_ptr() as *mut f64, sine_row_mu.len());
        let lead_row =
            std::slice::from_raw_parts_mut(lead_row_mu.as_mut_ptr() as *mut f64, lead_row_mu.len());

        match kern {
            Kernel::Scalar => msw_row_scalar(data, first, period, sine_row, lead_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => msw_row_avx2(data, first, period, sine_row, lead_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => msw_row_avx512(data, first, period, sine_row, lead_row),
            _ => unreachable!(),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            sine_mu
                .par_chunks_mut(cols)
                .zip(lead_mu.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, (sine_row_mu, lead_row_mu))| {
                    do_row(row, sine_row_mu, lead_row_mu)
                });
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, (sine_row_mu, lead_row_mu)) in sine_mu
                .chunks_mut(cols)
                .zip(lead_mu.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, sine_row_mu, lead_row_mu);
            }
        }
    } else {
        for (row, (sine_row_mu, lead_row_mu)) in sine_mu
            .chunks_mut(cols)
            .zip(lead_mu.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, sine_row_mu, lead_row_mu);
        }
    }

    Ok(combos)
}

#[inline(always)]
unsafe fn msw_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    sine: &mut [f64],
    lead: &mut [f64],
) {
    let step = TULIP_TPI / period as f64;
    let mut cos_table = Vec::with_capacity(period);
    let mut sin_table = Vec::with_capacity(period);
    let mut ang = 0.0f64;
    for _ in 0..period {
        let (s, c) = ang.sin_cos();
        sin_table.push(s);
        cos_table.push(c);
        ang += step;
    }

    // Keep lead via sin(phase + π/4) to match streaming path

    let warm = first + period - 1;
    for i in warm..data.len() {
        let mut rp = 0.0f64;
        let mut ip = 0.0f64;
        for j in 0..period {
            let w = *data.get_unchecked(i - j);
            rp += cos_table[j] * w;
            ip += sin_table[j] * w;
        }

        let mut phase = if rp.abs() > 0.001 {
            atan(ip / rp)
        } else {
            TULIP_PI * if ip < 0.0 { -1.0 } else { 1.0 }
        };
        if rp < 0.0 {
            phase += TULIP_PI;
        }
        phase += TULIP_PI * 0.5;
        if phase < 0.0 {
            phase += TULIP_TPI;
        }
        if phase > TULIP_TPI {
            phase -= TULIP_TPI;
        }
        let (s, c) = phase.sin_cos();
        sine[i] = s;
        lead[i] = (s + c) * 0.707106781186547524400844362104849039_f64;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn msw_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    sine: &mut [f64],
    lead: &mut [f64],
) {
    use core::arch::x86_64::*;
    let warm = first + period - 1;

    let step = TULIP_TPI / period as f64;
    let mut cos_table = Vec::with_capacity(period);
    let mut sin_table = Vec::with_capacity(period);
    let mut ang = 0.0f64;
    for _ in 0..period {
        let (s, c) = ang.sin_cos();
        sin_table.push(s);
        cos_table.push(c);
        ang += step;
    }
    let dptr = data.as_ptr();

    const LANES: usize = 4;
    // Keep lead via sin(phase + π/4) to match streaming path

    let mut i = warm;
    while i + (LANES - 1) < data.len() {
        let k = i + (LANES - 1);
        let mut rp = _mm256_set1_pd(0.0);
        let mut ip = _mm256_set1_pd(0.0);

        for j in 0..period {
            let base = k - j;
            let wv = _mm256_loadu_pd(dptr.add(base - (LANES - 1)));
            let cw = _mm256_set1_pd(*cos_table.get_unchecked(j));
            let sw = _mm256_set1_pd(*sin_table.get_unchecked(j));
            rp = _mm256_fmadd_pd(cw, wv, rp);
            ip = _mm256_fmadd_pd(sw, wv, ip);
        }

        let mut rbuf = [0.0f64; LANES];
        let mut ibuf = [0.0f64; LANES];
        _mm256_storeu_pd(rbuf.as_mut_ptr(), rp);
        _mm256_storeu_pd(ibuf.as_mut_ptr(), ip);

        let mut idx = i;
        for lane in 0..LANES {
            let mut phase = if rbuf[lane].abs() > 0.001 {
                atan(ibuf[lane] / rbuf[lane])
            } else {
                TULIP_PI * if ibuf[lane] < 0.0 { -1.0 } else { 1.0 }
            };
            if rbuf[lane] < 0.0 {
                phase += TULIP_PI;
            }
            phase += TULIP_PI * 0.5;
            if phase < 0.0 {
                phase += TULIP_TPI;
            }
            if phase > TULIP_TPI {
                phase -= TULIP_TPI;
            }
            let (s, c) = phase.sin_cos();
            sine[idx] = s;
            lead[idx] = (s + c) * 0.707106781186547524400844362104849039_f64;
            idx += 1;
        }

        i += LANES;
    }

    while i < data.len() {
        let mut rp = 0.0;
        let mut ip = 0.0;
        for j in 0..period {
            let w = *dptr.add(i - j);
            rp = (*cos_table.get_unchecked(j)).mul_add(w, rp);
            ip = (*sin_table.get_unchecked(j)).mul_add(w, ip);
        }
        let mut phase = if rp.abs() > 0.001 {
            atan(ip / rp)
        } else {
            TULIP_PI * if ip < 0.0 { -1.0 } else { 1.0 }
        };
        if rp < 0.0 {
            phase += TULIP_PI;
        }
        phase += TULIP_PI * 0.5;
        if phase < 0.0 {
            phase += TULIP_TPI;
        }
        if phase > TULIP_TPI {
            phase -= TULIP_TPI;
        }
        let (s, c) = phase.sin_cos();
        sine[i] = s;
        lead[i] = (s + c) * 0.707106781186547524400844362104849039_f64;
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn msw_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    sine: &mut [f64],
    lead: &mut [f64],
) {
    if period <= 32 {
        msw_row_avx512_short(data, first, period, sine, lead)
    } else {
        msw_row_avx512_long(data, first, period, sine, lead)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
unsafe fn msw_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    sine: &mut [f64],
    lead: &mut [f64],
) {
    use core::arch::x86_64::*;
    let warm = first + period - 1;

    let step = TULIP_TPI / period as f64;
    let mut cos_table = Vec::with_capacity(period);
    let mut sin_table = Vec::with_capacity(period);
    let mut ang = 0.0f64;
    for _ in 0..period {
        let (s, c) = ang.sin_cos();
        sin_table.push(s);
        cos_table.push(c);
        ang += step;
    }
    let dptr = data.as_ptr();

    const LANES: usize = 8;
    // Keep lead via sin(phase + π/4) to match streaming path

    let mut i = warm;
    while i + (LANES - 1) < data.len() {
        let k = i + (LANES - 1);
        let mut rp = _mm512_set1_pd(0.0);
        let mut ip = _mm512_set1_pd(0.0);

        for j in 0..period {
            let base = k - j;
            let wv = _mm512_loadu_pd(dptr.add(base - (LANES - 1)));
            let cw = _mm512_set1_pd(*cos_table.get_unchecked(j));
            let sw = _mm512_set1_pd(*sin_table.get_unchecked(j));
            rp = _mm512_fmadd_pd(cw, wv, rp);
            ip = _mm512_fmadd_pd(sw, wv, ip);
        }

        let mut rbuf = [0.0f64; LANES];
        let mut ibuf = [0.0f64; LANES];
        _mm512_storeu_pd(rbuf.as_mut_ptr(), rp);
        _mm512_storeu_pd(ibuf.as_mut_ptr(), ip);

        let mut idx = i;
        for lane in 0..LANES {
            let mut phase = if rbuf[lane].abs() > 0.001 {
                atan(ibuf[lane] / rbuf[lane])
            } else {
                TULIP_PI * if ibuf[lane] < 0.0 { -1.0 } else { 1.0 }
            };
            if rbuf[lane] < 0.0 {
                phase += TULIP_PI;
            }
            phase += TULIP_PI * 0.5;
            if phase < 0.0 {
                phase += TULIP_TPI;
            }
            if phase > TULIP_TPI {
                phase -= TULIP_TPI;
            }
            let (s, c) = phase.sin_cos();
            sine[idx] = s;
            lead[idx] = (s + c) * 0.707106781186547524400844362104849039_f64;
            idx += 1;
        }

        i += LANES;
    }

    while i < data.len() {
        let mut rp = 0.0;
        let mut ip = 0.0;
        for j in 0..period {
            let w = *dptr.add(i - j);
            rp = (*cos_table.get_unchecked(j)).mul_add(w, rp);
            ip = (*sin_table.get_unchecked(j)).mul_add(w, ip);
        }
        let mut phase = if rp.abs() > 0.001 {
            atan(ip / rp)
        } else {
            TULIP_PI * if ip < 0.0 { -1.0 } else { 1.0 }
        };
        if rp < 0.0 {
            phase += TULIP_PI;
        }
        phase += TULIP_PI * 0.5;
        if phase < 0.0 {
            phase += TULIP_TPI;
        }
        if phase > TULIP_TPI {
            phase -= TULIP_TPI;
        }
        let (s, c) = phase.sin_cos();
        sine[i] = s;
        lead[i] = (s + c) * 0.707106781186547524400844362104849039_f64;
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
unsafe fn msw_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    sine: &mut [f64],
    lead: &mut [f64],
) {
    // For now identical to the short variant; keep as a separate symbol
    // to allow future period-specific tuning.
    msw_row_avx512_short(data, first, period, sine, lead)
}

#[cfg(feature = "python")]
#[pyfunction(name = "msw")]
#[pyo3(signature = (data, period, kernel=None))]
pub fn msw_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    use numpy::PyArrayMethods;
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = MswParams {
        period: Some(period),
    };
    let msw_in = MswInput::from_slice(slice_in, params);

    let out = py
        .allow_threads(|| msw_with_kernel(&msw_in, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok((out.sine.into_pyarray(py), out.lead.into_pyarray(py)))
}

#[cfg(feature = "python")]
#[pyclass(name = "MswStream")]
pub struct MswStreamPy {
    stream: MswStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl MswStreamPy {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        let params = MswParams {
            period: Some(period),
        };
        let stream =
            MswStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(MswStreamPy { stream })
    }

    /// Updates the stream with a new value and returns the calculated MSW values.
    /// Returns `None` if the buffer is not yet full, otherwise returns a tuple of (sine, lead).
    fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "msw_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
pub fn msw_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    let sweep = MswBatchRange {
        period: period_range,
    };

    // Calculate dimensions
    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();

    // Pre-allocate output arrays (OK for batch operations)
    let out_sine = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let out_lead = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out_sine = unsafe { out_sine.as_slice_mut()? };
    let slice_out_lead = unsafe { out_lead.as_slice_mut()? };

    // Compute without GIL
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

            // Write directly to pre-allocated arrays
            msw_batch_inner_into(slice_in, &sweep, simd, true, slice_out_sine, slice_out_lead)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Build dict with zero-copy transfers
    let dict = PyDict::new(py);
    dict.set_item("sine", out_sine.reshape((rows, cols))?)?;
    dict.set_item("lead", out_lead.reshape((rows, cols))?)?;
    dict.set_item(
        "periods",
        combos
            .iter()
            .map(|p| p.period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    Ok(dict)
}

/// Write directly to output slices - no allocations
#[inline]
pub fn msw_into_slice(
    sine_dst: &mut [f64],
    lead_dst: &mut [f64],
    input: &MswInput,
    kern: Kernel,
) -> Result<(), MswError> {
    let data: &[f64] = match &input.data {
        MswData::Candles { candles, source } => source_type(candles, source),
        MswData::Slice(sl) => sl,
    };

    if data.is_empty() {
        return Err(MswError::EmptyData);
    }

    let period = input.get_period();
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(MswError::AllValuesNaN)?;
    let len = data.len();

    if period == 0 || period > len {
        return Err(MswError::InvalidPeriod {
            period,
            data_len: len,
        });
    }

    if (len - first) < period {
        return Err(MswError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    if sine_dst.len() != data.len() || lead_dst.len() != data.len() {
        return Err(MswError::InvalidPeriod {
            period: sine_dst.len(),
            data_len: data.len(),
        });
    }

    let chosen = match kern {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    // Compute directly into destination slices
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                msw_scalar_into(data, period, first, len, sine_dst, lead_dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                // AVX2 is currently a stub, use scalar
                msw_scalar_into(data, period, first, len, sine_dst, lead_dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                // AVX512 is currently a stub, use scalar
                msw_scalar_into(data, period, first, len, sine_dst, lead_dst)
            }
            _ => unreachable!(),
        }
    }?;

    // Fill warmup period with NaN
    let warmup = first + period - 1;
    for v in &mut sine_dst[..warmup] {
        *v = f64::NAN;
    }
    for v in &mut lead_dst[..warmup] {
        *v = f64::NAN;
    }

    Ok(())
}

/// Scalar implementation that writes directly to output slices
#[inline]
unsafe fn msw_scalar_into(
    data: &[f64],
    period: usize,
    first: usize,
    len: usize,
    sine: &mut [f64],
    lead: &mut [f64],
) -> Result<(), MswError> {
    // Precompute weights with one sin_cos per tap
    let step = TULIP_TPI / period as f64;
    let mut cos_table = Vec::with_capacity(period);
    let mut sin_table = Vec::with_capacity(period);
    let mut ang = 0.0f64;
    for _ in 0..period {
        let (s, c) = ang.sin_cos();
        sin_table.push(s);
        cos_table.push(c);
        ang += step;
    }

    let warm = first + period - 1;

    for i in warm..len {
        let mut rp = 0.0f64;
        let mut ip = 0.0f64;
        for j in 0..period {
            let w = *data.get_unchecked(i - j);
            rp += cos_table[j] * w;
            ip += sin_table[j] * w;
        }

        let mut phase = if rp.abs() > 0.001 {
            atan(ip / rp)
        } else {
            TULIP_PI * if ip < 0.0 { -1.0 } else { 1.0 }
        };
        if rp < 0.0 {
            phase += TULIP_PI;
        }
        phase += TULIP_PI * 0.5;
        if phase < 0.0 {
            phase += TULIP_TPI;
        }
        if phase > TULIP_TPI {
            phase -= TULIP_TPI;
        }

        let (s, c) = phase.sin_cos();
        *sine.get_unchecked_mut(i) = s;
        *lead.get_unchecked_mut(i) = (s + c) * 0.707106781186547524400844362104849039_f64;
    }
    Ok(())
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct MswJsOutput {
    pub sine: Vec<f64>,
    pub lead: Vec<f64>,
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct MswResult {
    pub values: Vec<f64>, // [sine rows first … then lead rows]
    pub rows: usize,      // 2
    pub cols: usize,      // data length
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn msw_js(data: &[f64], period: usize) -> Result<JsValue, JsValue> {
    let params = MswParams {
        period: Some(period),
    };
    let input = MswInput::from_slice(data, params);

    let len = data.len();
    let mut values = vec![f64::NAN; 2 * len];
    let (sine, lead) = values.split_at_mut(len);

    msw_into_slice(sine, lead, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let res = MswResult {
        values,
        rows: 2,
        cols: len,
    };
    serde_wasm_bindgen::to_value(&res)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

// Keep msw_wasm as deprecated alias for backward compatibility
#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[deprecated(since = "1.0.0", note = "Use msw_js instead")]
pub fn msw_wasm(data: &[f64], period: usize) -> Result<JsValue, JsValue> {
    msw_js(data, period)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn msw_into_flat(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer"));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let (sine, lead) = (
            std::slice::from_raw_parts_mut(out_ptr, len),
            std::slice::from_raw_parts_mut(out_ptr.add(len), len),
        );
        let input = MswInput::from_slice(
            data,
            MswParams {
                period: Some(period),
            },
        );
        msw_into_slice(sine, lead, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn msw_into(
    in_ptr: *const f64,
    sine_ptr: *mut f64,
    lead_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || sine_ptr.is_null() || lead_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        if period == 0 || period > len {
            return Err(JsValue::from_str("Invalid period"));
        }

        let params = MswParams {
            period: Some(period),
        };
        let input = MswInput::from_slice(data, params);

        // Check for aliasing - any output pointer matches input or each other
        let aliasing = in_ptr as *const _ == sine_ptr as *const _
            || in_ptr as *const _ == lead_ptr as *const _
            || sine_ptr == lead_ptr;

        if aliasing {
            // Use temporary buffers when aliasing detected
            let mut temp_sine = vec![0.0; len];
            let mut temp_lead = vec![0.0; len];
            msw_into_slice(&mut temp_sine, &mut temp_lead, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            // Copy to output pointers
            let sine_out = std::slice::from_raw_parts_mut(sine_ptr, len);
            let lead_out = std::slice::from_raw_parts_mut(lead_ptr, len);
            sine_out.copy_from_slice(&temp_sine);
            lead_out.copy_from_slice(&temp_lead);
        } else {
            // Direct computation into output slices
            let sine_out = std::slice::from_raw_parts_mut(sine_ptr, len);
            let lead_out = std::slice::from_raw_parts_mut(lead_ptr, len);
            msw_into_slice(sine_out, lead_out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn msw_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn msw_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct MswBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct MswBatchJsOutput {
    pub sine: Vec<f64>,
    pub lead: Vec<f64>,
    pub combos: Vec<MswParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct MswBatchFlatJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<MswParams>,
    pub rows: usize, // 2 * combos.len()
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = msw_batch)]
pub fn msw_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: MswBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
    let sweep = MswBatchRange {
        period: config.period_range,
    };

    // Plan buffers: first half for sine rows, second half for lead rows
    let combos = expand_grid(&sweep);
    if combos.is_empty() {
        return Err(JsValue::from_str("No parameter combinations"));
    }
    let rows = combos.len();
    let cols = data.len();

    let mut values = vec![f64::NAN; 2 * rows * cols];
    let (sine_out, lead_out) = values.split_at_mut(rows * cols);

    msw_batch_inner_into(data, &sweep, Kernel::Auto, false, sine_out, lead_out)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let out = MswBatchFlatJsOutput {
        values,
        combos,
        rows: 2 * rows,
        cols,
    };
    serde_wasm_bindgen::to_value(&out)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn msw_batch_js(
    data: &[f64],
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<JsValue, JsValue> {
    let sweep = MswBatchRange {
        period: (period_start, period_end, period_step),
    };

    // Use the existing batch function with parallel=false for WASM
    let output = msw_batch_inner(data, &sweep, Kernel::Auto, false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Create the structured output
    let js_output = MswBatchJsOutput {
        sine: output.sine,
        lead: output.lead,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };

    // Serialize the output struct into a JavaScript object
    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn msw_batch_metadata_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = MswBatchRange {
        period: (period_start, period_end, period_step),
    };

    let combos = expand_grid(&sweep);
    let metadata = combos
        .iter()
        .map(|combo| combo.period.unwrap() as f64)
        .collect();

    Ok(metadata)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn msw_batch_into_flat(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer"));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let sweep = MswBatchRange {
            period: (period_start, period_end, period_step),
        };
        let combos = expand_grid(&sweep);
        if combos.is_empty() {
            return Err(JsValue::from_str("No parameter combinations"));
        }
        let rows = combos.len();
        let cols = len;

        let sine_out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);
        let lead_out = std::slice::from_raw_parts_mut(out_ptr.add(rows * cols), rows * cols);

        msw_batch_inner_into(data, &sweep, Kernel::Auto, false, sine_out, lead_out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(2 * rows) // number of rows written
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn msw_batch_into(
    in_ptr: *const f64,
    sine_ptr: *mut f64,
    lead_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || sine_ptr.is_null() || lead_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        let sweep = MswBatchRange {
            period: (period_start, period_end, period_step),
        };

        let combos = expand_grid(&sweep);
        if combos.is_empty() {
            return Err(JsValue::from_str("No valid parameter combinations"));
        }

        let rows = combos.len();
        let cols = len;
        let total_len = rows * cols;

        let sine_out = std::slice::from_raw_parts_mut(sine_ptr, total_len);
        let lead_out = std::slice::from_raw_parts_mut(lead_ptr, total_len);

        // Process each parameter combination
        for (idx, params) in combos.iter().enumerate() {
            let row_start = idx * cols;
            let row_end = row_start + cols;

            let input = MswInput::from_slice(data, params.clone());

            msw_into_slice(
                &mut sine_out[row_start..row_end],
                &mut lead_out[row_start..row_end],
                &input,
                Kernel::Auto,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_msw_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = MswParams { period: None };
        let input_default = MswInput::from_candles(&candles, "close", default_params);
        let output_default = msw_with_kernel(&input_default, kernel)?;
        assert_eq!(output_default.sine.len(), candles.close.len());
        assert_eq!(output_default.lead.len(), candles.close.len());
        Ok(())
    }

    fn check_msw_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = MswParams { period: Some(5) };
        let input = MswInput::from_candles(&candles, "close", params);
        let msw_result = msw_with_kernel(&input, kernel)?;
        let expected_last_five_sine = [
            -0.49733966449848194,
            -0.8909425976991894,
            -0.709353328514554,
            -0.40483478076837887,
            -0.8817006719953886,
        ];
        let expected_last_five_lead = [
            -0.9651269132969991,
            -0.30888310410390457,
            -0.003182174183612666,
            0.36030983330963545,
            -0.28983704937461496,
        ];
        let start = msw_result.sine.len().saturating_sub(5);
        for (i, &val) in msw_result.sine[start..].iter().enumerate() {
            let diff = (val - expected_last_five_sine[i]).abs();
            assert!(
                diff < 1e-1,
                "[{}] MSW sine mismatch at idx {}: got {}, expected {}",
                test_name,
                i,
                val,
                expected_last_five_sine[i]
            );
        }
        for (i, &val) in msw_result.lead[start..].iter().enumerate() {
            let diff = (val - expected_last_five_lead[i]).abs();
            assert!(
                diff < 1e-1,
                "[{}] MSW lead mismatch at idx {}: got {}, expected {}",
                test_name,
                i,
                val,
                expected_last_five_lead[i]
            );
        }
        Ok(())
    }

    fn check_msw_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = MswInput::with_default_candles(&candles);
        let output = msw_with_kernel(&input, kernel)?;
        assert_eq!(output.sine.len(), candles.close.len());
        assert_eq!(output.lead.len(), candles.close.len());
        Ok(())
    }

    fn check_msw_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = MswParams { period: Some(0) };
        let input = MswInput::from_slice(&input_data, params);
        let res = msw_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] MSW should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_msw_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = MswParams { period: Some(10) };
        let input = MswInput::from_slice(&data_small, params);
        let res = msw_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] MSW should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_msw_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = MswParams { period: Some(5) };
        let input = MswInput::from_slice(&single_point, params);
        let res = msw_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] MSW should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_msw_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = MswParams { period: Some(5) };
        let input = MswInput::from_candles(&candles, "close", params);
        let res = msw_with_kernel(&input, kernel)?;
        assert_eq!(res.sine.len(), candles.close.len());
        assert_eq!(res.lead.len(), candles.close.len());
        Ok(())
    }

    fn check_msw_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 5;
        let input = MswInput::from_candles(
            &candles,
            "close",
            MswParams {
                period: Some(period),
            },
        );
        let batch_output = msw_with_kernel(&input, kernel)?;
        let mut stream = MswStream::try_new(MswParams {
            period: Some(period),
        })?;
        let mut sine_stream = Vec::with_capacity(candles.close.len());
        let mut lead_stream = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some((s, l)) => {
                    sine_stream.push(s);
                    lead_stream.push(l);
                }
                None => {
                    sine_stream.push(f64::NAN);
                    lead_stream.push(f64::NAN);
                }
            }
        }
        assert_eq!(batch_output.sine.len(), sine_stream.len());
        assert_eq!(batch_output.lead.len(), lead_stream.len());
        for (i, (&b, &s)) in batch_output.sine.iter().zip(sine_stream.iter()).enumerate() {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] MSW streaming sine mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        for (i, (&b, &l)) in batch_output.lead.iter().zip(lead_stream.iter()).enumerate() {
            if b.is_nan() && l.is_nan() {
                continue;
            }
            let diff = (b - l).abs();
            assert!(
                diff < 1e-9,
                "[{}] MSW streaming lead mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                l,
                diff
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_msw_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Define comprehensive parameter combinations
        let test_params = vec![
            MswParams::default(),            // period: 5 (default)
            MswParams { period: Some(2) },   // minimum viable period
            MswParams { period: Some(3) },   // small period
            MswParams { period: Some(7) },   // medium-small period
            MswParams { period: Some(10) },  // medium period
            MswParams { period: Some(20) },  // large period
            MswParams { period: Some(50) },  // very large period
            MswParams { period: Some(100) }, // maximum reasonable period
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = MswInput::from_candles(&candles, "close", params.clone());
            let output = msw_with_kernel(&input, kernel)?;

            // Check sine values
            for (i, &val) in output.sine.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 in sine output with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(5),
                        param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 in sine output with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(5),
                        param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 in sine output with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(5),
                        param_idx
                    );
                }
            }

            // Check lead values
            for (i, &val) in output.lead.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 in lead output with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(5),
                        param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 in lead output with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(5),
                        param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 in lead output with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(5),
                        param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_msw_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(()) // No-op in release builds
    }

    macro_rules! generate_all_msw_tests {
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
    generate_all_msw_tests!(
        check_msw_partial_params,
        check_msw_accuracy,
        check_msw_default_candles,
        check_msw_zero_period,
        check_msw_period_exceeds_length,
        check_msw_very_small_dataset,
        check_msw_nan_handling,
        check_msw_streaming,
        check_msw_no_poison
    );

    #[cfg(feature = "proptest")]
    fn check_msw_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Strategy for generating realistic test data
        let strat = (2usize..=50) // Period range for MSW
            .prop_flat_map(|period| {
                (period..=400).prop_flat_map(move |data_len| {
                    prop_oneof![
                        // 60% normal trading data
                        6 => prop::collection::vec(
                            (10.0f64..10000.0f64).prop_filter("finite", |x| x.is_finite()),
                            data_len
                        ).prop_map(move |v| (v, period)),
                        // 15% constant prices
                        3 => prop::collection::vec(
                            Just(100.0f64),
                            data_len
                        ).prop_map(move |v| (v, period)),
                        // 15% trending data (monotonic)
                        3 => (0.0f64..100.0f64, 0.01f64..1.0f64).prop_map(move |(start, step)| {
                            let data: Vec<f64> = (0..data_len)
                                .map(|i| start + (i as f64) * step)
                                .collect();
                            (data, period)
                        }),
                        // 10% edge cases (zeros, small values)
                        2 => prop_oneof![
                            prop::collection::vec(Just(0.0f64), data_len),
                            prop::collection::vec((0.0001f64..0.01f64), data_len),
                        ].prop_map(move |v| (v, period))
                    ]
                })
            });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, period)| {
                let params = MswParams {
                    period: Some(period),
                };
                let input = MswInput::from_slice(&data, params.clone());

                // Get outputs from the specified kernel and scalar reference
                let output = msw_with_kernel(&input, kernel).unwrap();
                let ref_output = msw_with_kernel(&input, Kernel::Scalar).unwrap();

                // Property 1: Output length matches input
                prop_assert_eq!(output.sine.len(), data.len(), "Sine output length mismatch");
                prop_assert_eq!(output.lead.len(), data.len(), "Lead output length mismatch");

                // Find first valid index
                let first_valid = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
                let warmup_end = first_valid + period - 1;

                // Property 2: Warmup period verification
                for i in 0..warmup_end.min(data.len()) {
                    prop_assert!(
                        output.sine[i].is_nan(),
                        "Sine[{}] should be NaN during warmup (first_valid={}, period={})",
                        i,
                        first_valid,
                        period
                    );
                    prop_assert!(
                        output.lead[i].is_nan(),
                        "Lead[{}] should be NaN during warmup (first_valid={}, period={})",
                        i,
                        first_valid,
                        period
                    );
                }

                // Property 3: First valid output at correct index
                if warmup_end < data.len() {
                    prop_assert!(
                        !output.sine[warmup_end].is_nan(),
                        "Sine[{}] should be valid after warmup",
                        warmup_end
                    );
                    prop_assert!(
                        !output.lead[warmup_end].is_nan(),
                        "Lead[{}] should be valid after warmup",
                        warmup_end
                    );
                }

                // Property 4: Value bounds - sine and lead are bounded [-1, 1]
                for i in warmup_end..data.len() {
                    let sine_val = output.sine[i];
                    let lead_val = output.lead[i];

                    if !sine_val.is_nan() {
                        prop_assert!(
                            sine_val >= -1.0 - 1e-9 && sine_val <= 1.0 + 1e-9,
                            "Sine[{}] = {} is outside [-1, 1] bounds",
                            i,
                            sine_val
                        );
                    }

                    if !lead_val.is_nan() {
                        prop_assert!(
                            lead_val >= -1.0 - 1e-9 && lead_val <= 1.0 + 1e-9,
                            "Lead[{}] = {} is outside [-1, 1] bounds",
                            i,
                            lead_val
                        );
                    }
                }

                // Property 5: Constant data produces consistent phase values
                if data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10)
                    && warmup_end + 5 < data.len()
                {
                    // For constant data, all outputs should have the same phase
                    // since the weighted sum is constant * sum_of_weights
                    let first_sine = output.sine[warmup_end];
                    let first_lead = output.lead[warmup_end];

                    for i in (warmup_end + 1)..(warmup_end + 5).min(data.len()) {
                        let sine_val = output.sine[i];
                        let lead_val = output.lead[i];

                        if !sine_val.is_nan() && !first_sine.is_nan() {
                            prop_assert!(
                                (sine_val - first_sine).abs() < 1e-9,
                                "Constant data: Sine[{}] = {} differs from first = {}",
                                i,
                                sine_val,
                                first_sine
                            );
                        }
                        if !lead_val.is_nan() && !first_lead.is_nan() {
                            prop_assert!(
                                (lead_val - first_lead).abs() < 1e-9,
                                "Constant data: Lead[{}] = {} differs from first = {}",
                                i,
                                lead_val,
                                first_lead
                            );
                        }
                    }
                }

                // Property 6: Mathematical phase relationship verification
                // Lead = sin(phase + π/4) where Sine = sin(phase)
                // We can verify: lead ≈ sine * cos(π/4) + cos(asin(sine)) * sin(π/4)
                // Simplified: lead ≈ sine * 0.707 + sqrt(1 - sine²) * 0.707
                if warmup_end + 10 < data.len() {
                    const COS_PI4: f64 = 0.7071067811865476; // cos(π/4) = sin(π/4)

                    for i in (warmup_end + 5)..(warmup_end + 10).min(data.len()) {
                        let sine_val = output.sine[i];
                        let lead_val = output.lead[i];

                        if !sine_val.is_nan() && !lead_val.is_nan() && sine_val.abs() < 0.999 {
                            // Calculate expected lead from sine using trigonometric identity
                            // sin(θ + π/4) = sin(θ)cos(π/4) + cos(θ)sin(π/4)
                            // cos(θ) = ±sqrt(1 - sin²(θ))
                            let cos_phase = (1.0 - sine_val * sine_val).sqrt();
                            // Try both positive and negative cosine (we don't know the quadrant)
                            let expected_lead_pos = sine_val * COS_PI4 + cos_phase * COS_PI4;
                            let expected_lead_neg = sine_val * COS_PI4 - cos_phase * COS_PI4;

                            let diff_pos = (lead_val - expected_lead_pos).abs();
                            let diff_neg = (lead_val - expected_lead_neg).abs();
                            let min_diff = diff_pos.min(diff_neg);

                            prop_assert!(
								min_diff < 0.01,
								"Phase relationship incorrect at [{}]: sine={}, lead={}, expected ≈ {} or {}",
								i, sine_val, lead_val, expected_lead_pos, expected_lead_neg
							);
                        }
                    }
                }

                // Property 7: Zero data produces specific deterministic values
                if data.iter().all(|&x| x.abs() < 1e-10) && warmup_end < data.len() {
                    // For all-zero data:
                    // rp = sum(cos_table[j] * 0) = 0
                    // ip = sum(sin_table[j] * 0) = 0
                    // Since rp.abs() <= 0.001 and ip == 0 (not < 0):
                    // phase = π * 1.0 = π
                    // Since rp == 0 (not < 0), no additional π added
                    // phase += π/2 → phase = π + π/2 = 3π/2
                    // sine = sin(3π/2) = -1
                    // lead = sin(3π/2 + π/4) = sin(7π/4) = -0.7071...

                    const EXPECTED_SINE: f64 = -1.0;
                    const EXPECTED_LEAD: f64 = -0.7071067811865476; // sin(7π/4)

                    for i in warmup_end..(warmup_end + 3).min(data.len()) {
                        let sine_val = output.sine[i];
                        let lead_val = output.lead[i];

                        prop_assert!(
                            (sine_val - EXPECTED_SINE).abs() < 1e-7,
                            "Zero data: Sine[{}] = {}, expected {}",
                            i,
                            sine_val,
                            EXPECTED_SINE
                        );
                        prop_assert!(
                            (lead_val - EXPECTED_LEAD).abs() < 1e-7,
                            "Zero data: Lead[{}] = {}, expected {}",
                            i,
                            lead_val,
                            EXPECTED_LEAD
                        );
                    }
                }

                // Property 8: Period=2 special case (minimum valid period)
                // With period=2, we only look at current and previous value
                // This tests the edge case of minimum lookback
                if period == 2 && warmup_end < data.len() {
                    // For period=2:
                    // cos_table = [cos(0), cos(π)] = [1, -1]
                    // sin_table = [sin(0), sin(π)] = [0, 0]
                    // So ip = 0 always, and rp = data[i] * 1 + data[i-1] * (-1) = data[i] - data[i-1]

                    for i in warmup_end..(warmup_end + 3).min(data.len()) {
                        let sine_val = output.sine[i];
                        let lead_val = output.lead[i];

                        // Verify outputs are valid
                        prop_assert!(
                            !sine_val.is_nan(),
                            "Period=2: Sine[{}] should not be NaN",
                            i
                        );
                        prop_assert!(
                            !lead_val.is_nan(),
                            "Period=2: Lead[{}] should not be NaN",
                            i
                        );

                        // Verify bounds
                        prop_assert!(
                            sine_val >= -1.0 - 1e-9 && sine_val <= 1.0 + 1e-9,
                            "Period=2: Sine[{}] = {} out of bounds",
                            i,
                            sine_val
                        );
                        prop_assert!(
                            lead_val >= -1.0 - 1e-9 && lead_val <= 1.0 + 1e-9,
                            "Period=2: Lead[{}] = {} out of bounds",
                            i,
                            lead_val
                        );

                        // For alternating data [a, b, a, b, ...], should produce consistent phase
                        if i >= 3
                            && i >= warmup_end + 2
                            && (data[i] - data[i - 2]).abs() < 1e-10
                            && (data[i - 1] - data[i - 3]).abs() < 1e-10
                        {
                            let prev_sine = output.sine[i - 2];
                            prop_assert!(
								(sine_val - prev_sine).abs() < 1e-6,
								"Period=2 with alternating data: Sine should repeat every 2 samples"
							);
                        }
                    }
                }

                // Property 9: Kernel consistency - all kernels produce identical results
                for i in 0..data.len() {
                    let sine_val = output.sine[i];
                    let ref_sine = ref_output.sine[i];
                    let lead_val = output.lead[i];
                    let ref_lead = ref_output.lead[i];

                    // Check sine consistency
                    if sine_val.is_nan() && ref_sine.is_nan() {
                        continue; // Both NaN is OK
                    }

                    if sine_val.is_finite() && ref_sine.is_finite() {
                        let sine_bits = sine_val.to_bits();
                        let ref_sine_bits = ref_sine.to_bits();
                        let ulp_diff = sine_bits.abs_diff(ref_sine_bits);

                        prop_assert!(
                            (sine_val - ref_sine).abs() <= 1e-9 || ulp_diff <= 5,
                            "Kernel mismatch for sine at [{}]: {} vs {} (ULP={})",
                            i,
                            sine_val,
                            ref_sine,
                            ulp_diff
                        );
                    } else {
                        prop_assert_eq!(
                            sine_val.is_nan(),
                            ref_sine.is_nan(),
                            "Kernel NaN mismatch for sine at [{}]",
                            i
                        );
                    }

                    // Check lead consistency
                    if lead_val.is_nan() && ref_lead.is_nan() {
                        continue; // Both NaN is OK
                    }

                    if lead_val.is_finite() && ref_lead.is_finite() {
                        let lead_bits = lead_val.to_bits();
                        let ref_lead_bits = ref_lead.to_bits();
                        let ulp_diff = lead_bits.abs_diff(ref_lead_bits);

                        prop_assert!(
                            (lead_val - ref_lead).abs() <= 1e-9 || ulp_diff <= 5,
                            "Kernel mismatch for lead at [{}]: {} vs {} (ULP={})",
                            i,
                            lead_val,
                            ref_lead,
                            ulp_diff
                        );
                    } else {
                        prop_assert_eq!(
                            lead_val.is_nan(),
                            ref_lead.is_nan(),
                            "Kernel NaN mismatch for lead at [{}]",
                            i
                        );
                    }
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    #[cfg(feature = "proptest")]
    generate_all_msw_tests!(check_msw_property);

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = MswBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = MswParams::default();
        let row = output.sine_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test various parameter sweep configurations
        let test_configs = vec![
            (2, 10, 2),    // Small periods
            (5, 25, 5),    // Medium periods
            (30, 60, 15),  // Large periods
            (2, 5, 1),     // Dense small range
            (10, 20, 2),   // Dense medium range
            (15, 30, 3),   // Medium-large range
            (50, 100, 10), // Very large periods
        ];

        for (cfg_idx, &(period_start, period_end, period_step)) in test_configs.iter().enumerate() {
            let output = MswBatchBuilder::new()
                .kernel(kernel)
                .period_range(period_start, period_end, period_step)
                .apply_candles(&c, "close")?;

            // Check sine values
            for (idx, &val) in output.sine.iter().enumerate() {
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
						 at row {} col {} (flat index {}) in sine output with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(5)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in sine output with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(5)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in sine output with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(5)
                    );
                }
            }

            // Check lead values
            for (idx, &val) in output.lead.iter().enumerate() {
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
						 at row {} col {} (flat index {}) in lead output with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(5)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in lead output with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(5)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in lead output with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(5)
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
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
    gen_batch_tests!(check_batch_no_poison);
}

// ==================== Python CUDA bindings ====================
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "msw_cuda_batch_dev")]
#[pyo3(signature = (close_f32, period_range, device_id=0))]
pub fn msw_cuda_batch_dev_py<'py>(
    py: Python<'py>,
    close_f32: numpy::PyReadonlyArray1<'py, f32>,
    period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<(DeviceArrayF32Py, Bound<'py, pyo3::types::PyDict>)> {
    use numpy::IntoPyArray;
    if !cuda_available() { return Err(PyValueError::new_err("CUDA not available")); }
    let slice = close_f32.as_slice()?;
    let sweep = MswBatchRange { period: period_range };
    let (inner, combos) = py
        .allow_threads(|| {
            let cuda = CudaMsw::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
            cuda.msw_batch_dev(slice, &sweep).map_err(|e| PyValueError::new_err(e.to_string()))
        })?;
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item(
        "periods",
        combos
            .iter()
            .map(|p| p.period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", 2 * combos.len())?;
    dict.set_item("cols", slice.len())?;
    Ok((DeviceArrayF32Py { inner }, dict))
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "msw_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, period, device_id=0))]
pub fn msw_cuda_many_series_one_param_dev_py<'py>(
    py: Python<'py>,
    data_tm_f32: numpy::PyReadonlyArray2<'py, f32>,
    period: usize,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    use numpy::PyUntypedArrayMethods;
    if !cuda_available() { return Err(PyValueError::new_err("CUDA not available")); }
    let shape = data_tm_f32.shape();
    if shape.len() != 2 { return Err(PyValueError::new_err("expected 2D array (rows x cols)")); }
    let rows = shape[0];
    let cols = shape[1];
    let flat = data_tm_f32.as_slice()?;
    let params = MswParams { period: Some(period) };
    let inner = py
        .allow_threads(|| {
            let cuda = CudaMsw::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
            cuda
                .msw_many_series_one_param_time_major_dev(flat, cols, rows, &params)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })?;
    Ok(DeviceArrayF32Py { inner })
}
