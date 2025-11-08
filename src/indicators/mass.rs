//! # Mass Index (MASS)
//!
//! The Mass Index is an indicator that uses the ratio of two exponential moving averages
//! (both using period=9) of the range (high - low) and sums these ratios over `period` bars.
//! This implementation follows the Tulip Indicators reference for MASS, with a default period of 5.
//!
//! ## Parameters
//! - **period**: The summation window size. Defaults to 5.
//!
//! ## Inputs
//! - **high**: High price data
//! - **low**: Low price data
//!
//! ## Returns
//! - **values**: Vector of Mass Index values with NaN prefix during warmup period
//!
//! ## Developer Notes
//! - **SIMD status**: AVX2/AVX512 mirror scalar math and add light prefetching. Due to loop-carried EMA
//!   dependencies, lane-wise SIMD is not viable; expected gains are modest and workload-dependent.
//! - **Batch status**: Row-specific batch kernels not attempted here. A future pass can precompute the ratio
//!   and a prefix sum once, then fill rows via differences for multi-x speedups without changing outputs.
//! - **Streaming**: O(1) updates with a mask-based ring buffer; exact division preserved. Matches scalar
//!   path byte-for-byte. FMA-friendly updates; no unsafe in scalar logic.
//! - **Zero-copy Memory**: Uses alloc_with_nan_prefix and make_uninit_matrix for batch operations

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::cuda_available;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::mass_wrapper::CudaMass;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::alma::DeviceArrayF32Py;
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
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum MassData<'a> {
    Candles {
        candles: &'a Candles,
        high_source: &'a str,
        low_source: &'a str,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct MassOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct MassParams {
    pub period: Option<usize>,
}

impl Default for MassParams {
    fn default() -> Self {
        Self { period: Some(5) }
    }
}

#[derive(Debug, Clone)]
pub struct MassInput<'a> {
    pub data: MassData<'a>,
    pub params: MassParams,
}

impl<'a> MassInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        high_source: &'a str,
        low_source: &'a str,
        params: MassParams,
    ) -> Self {
        Self {
            data: MassData::Candles {
                candles,
                high_source,
                low_source,
            },
            params,
        }
    }

    #[inline]
    pub fn from_slices(high: &'a [f64], low: &'a [f64], params: MassParams) -> Self {
        Self {
            data: MassData::Slices { high, low },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: MassData::Candles {
                candles,
                high_source: "high",
                low_source: "low",
            },
            params: MassParams::default(),
        }
    }

    #[inline]
    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| MassParams::default().period.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum MassError {
    #[error("mass: Empty data provided.")]
    EmptyData,
    #[error("mass: High and low slices must have the same length.")]
    DifferentLengthHL,
    #[error("mass: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("mass: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("mass: All values are NaN.")]
    AllValuesNaN,
}

#[derive(Clone, Debug)]
pub struct MassBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for MassBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl MassBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<MassOutput, MassError> {
        let p = MassParams {
            period: self.period,
        };
        let i = MassInput::from_candles(c, "high", "low", p);
        mass_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<MassOutput, MassError> {
        let p = MassParams {
            period: self.period,
        };
        let i = MassInput::from_slices(high, low, p);
        mass_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<MassStream, MassError> {
        let p = MassParams {
            period: self.period,
        };
        MassStream::try_new(p)
    }
}

#[inline(always)]
fn mass_prepare<'a>(
    input: &'a MassInput,
    kernel: Kernel,
) -> Result<(&'a [f64], &'a [f64], usize, usize, Kernel), MassError> {
    let (high, low) = match &input.data {
        MassData::Candles {
            candles,
            high_source,
            low_source,
        } => (
            source_type(candles, high_source),
            source_type(candles, low_source),
        ),
        MassData::Slices { high, low } => (*high, *low),
    };

    if high.is_empty() || low.is_empty() {
        return Err(MassError::EmptyData);
    }
    if high.len() != low.len() {
        return Err(MassError::DifferentLengthHL);
    }

    let period = input.get_period();
    if period == 0 || period > high.len() {
        return Err(MassError::InvalidPeriod {
            period,
            data_len: high.len(),
        });
    }

    let first = (0..high.len())
        .find(|&i| !high[i].is_nan() && !low[i].is_nan())
        .ok_or(MassError::AllValuesNaN)?;

    // Two nested EMA(9)s -> need 16 warm bars before summing, then (period-1) for the sum window.
    let needed_bars = 16 + period - 1;
    if high.len() - first < needed_bars {
        return Err(MassError::NotEnoughValidData {
            needed: needed_bars,
            valid: high.len() - first,
        });
    }

    // SIMD prefetch variant shows mixed results; default Auto short-circuits to Scalar for stability.
    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        k => k,
    };
    Ok((high, low, period, first, chosen))
}

#[inline(always)]
fn mass_compute_into(
    high: &[f64],
    low: &[f64],
    period: usize,
    first: usize,
    kern: Kernel,
    out: &mut [f64],
) {
    unsafe {
        match kern {
            Kernel::Scalar | Kernel::ScalarBatch => mass_scalar(high, low, period, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => mass_avx2(high, low, period, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => mass_avx512(high, low, period, first, out),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                mass_scalar(high, low, period, first, out)
            }
            _ => unreachable!(),
        }
    }
}

#[inline]
pub fn mass(input: &MassInput) -> Result<MassOutput, MassError> {
    mass_with_kernel(input, Kernel::Auto)
}

pub fn mass_with_kernel(input: &MassInput, kernel: Kernel) -> Result<MassOutput, MassError> {
    let (high, low, period, first, chosen) = mass_prepare(input, kernel)?;
    let warmup_end = first + 16 + period - 1;
    let mut out = alloc_with_nan_prefix(high.len(), warmup_end);
    mass_compute_into(high, low, period, first, chosen, &mut out);
    Ok(MassOutput { values: out })
}

#[inline]
pub fn mass_into_slice(
    dst: &mut [f64],
    input: &MassInput,
    kernel: Kernel,
) -> Result<(), MassError> {
    let (high, low, period, first, chosen) = mass_prepare(input, kernel)?;
    if dst.len() != high.len() {
        return Err(MassError::InvalidPeriod {
            period: dst.len(),
            data_len: high.len(),
        });
    }
    mass_compute_into(high, low, period, first, chosen, dst);
    let warmup_end = first + 16 + period - 1;
    // Match alloc_with_nan_prefix quiet-NaN pattern for warmup prefix
    let qnan = f64::from_bits(0x7ff8_0000_0000_0000);
    for v in &mut dst[..warmup_end] {
        *v = qnan;
    }
    Ok(())
}

/// Writes Mass Index values into a caller-provided buffer without allocating.
///
/// - Preserves the NaN warmup prefix exactly as the Vec-returning API.
/// - `out.len()` must equal the input length; returns an error on mismatch.
/// - Uses `Kernel::Auto` dispatch consistent with this module’s runtime selection.
#[cfg(not(feature = "wasm"))]
#[inline]
pub fn mass_into(input: &MassInput, out: &mut [f64]) -> Result<(), MassError> {
    mass_into_slice(out, input, Kernel::Auto)
}

#[inline]
pub fn mass_scalar(
    high: &[f64],
    low: &[f64],
    period: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    // Constants for EMA(9)
    const ALPHA: f64 = 2.0 / 10.0; // 0.2
    const INV_ALPHA: f64 = 1.0 - ALPHA; // 0.8

    let n = high.len();
    if n == 0 {
        return;
    }

    // Warmup boundaries
    let start_ema2 = first_valid_idx + 8;
    let start_ratio = first_valid_idx + 16;
    let start_out = start_ratio + (period - 1);

    // Initial EMA seeds
    let mut ema1 = high[first_valid_idx] - low[first_valid_idx];
    let mut ema2 = ema1;

    // Cacheline-aligned ring buffer
    let mut ring: AVec<f64> = AVec::with_capacity(CACHELINE_ALIGN, period);
    ring.resize(period, 0.0);

    let mut ring_index: usize = 0;
    let mut sum_ratio: f64 = 0.0;

    unsafe {
        let hp = high.as_ptr();
        let lp = low.as_ptr();
        let outp = out.as_mut_ptr();
        let rp = ring.as_mut_ptr();

        let mut i = first_valid_idx;
        while i < n {
            let hl = *hp.add(i) - *lp.add(i);
            // ema1 = ema1 * 0.8 + hl * 0.2
            ema1 = ema1.mul_add(INV_ALPHA, hl * ALPHA);

            if i == start_ema2 {
                ema2 = ema1;
            }
            if i >= start_ema2 {
                // ema2 = ema2 * 0.8 + ema1 * 0.2
                ema2 = ema2.mul_add(INV_ALPHA, ema1 * ALPHA);

                if i >= start_ratio {
                    let ratio = ema1 / ema2;

                    // Sliding sum via ring buffer
                    sum_ratio -= *rp.add(ring_index);
                    *rp.add(ring_index) = ratio;
                    sum_ratio += ratio;

                    // Advance ring index without modulo
                    ring_index += 1;
                    if ring_index == period {
                        ring_index = 0;
                    }

                    if i >= start_out {
                        *outp.add(i) = sum_ratio;
                    }
                }
            }

            i += 1;
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn mass_avx512(
    high: &[f64],
    low: &[f64],
    period: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        unsafe { mass_avx512_short(high, low, period, first_valid_idx, out) }
    } else {
        unsafe { mass_avx512_long(high, low, period, first_valid_idx, out) }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn mass_avx2(
    high: &[f64],
    low: &[f64],
    period: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};

    const ALPHA: f64 = 2.0 / 10.0;
    const INV_ALPHA: f64 = 1.0 - ALPHA;

    let n = high.len();
    if n == 0 {
        return;
    }

    let start_ema2 = first_valid_idx + 8;
    let start_ratio = first_valid_idx + 16;
    let start_out = start_ratio + (period - 1);

    let mut ema1 = high[first_valid_idx] - low[first_valid_idx];
    let mut ema2 = ema1;

    let mut ring: AVec<f64> = AVec::with_capacity(CACHELINE_ALIGN, period);
    ring.resize(period, 0.0);

    let mut ring_index: usize = 0;
    let mut sum_ratio: f64 = 0.0;

    unsafe {
        let hp = high.as_ptr();
        let lp = low.as_ptr();
        let outp = out.as_mut_ptr();
        let rp = ring.as_mut_ptr();

        const PF_DIST: usize = 64;

        let mut i = first_valid_idx;
        while i < n {
            let pf = i + PF_DIST;
            if pf < n {
                _mm_prefetch(hp.add(pf) as *const i8, _MM_HINT_T0);
                _mm_prefetch(lp.add(pf) as *const i8, _MM_HINT_T0);
                _mm_prefetch(outp.add(pf) as *const i8, _MM_HINT_T0);
            }

            let hl = *hp.add(i) - *lp.add(i);
            ema1 = ema1.mul_add(INV_ALPHA, hl * ALPHA);

            if i == start_ema2 {
                ema2 = ema1;
            }
            if i >= start_ema2 {
                ema2 = ema2.mul_add(INV_ALPHA, ema1 * ALPHA);

                if i >= start_ratio {
                    let ratio = ema1 / ema2;

                    sum_ratio -= *rp.add(ring_index);
                    *rp.add(ring_index) = ratio;
                    sum_ratio += ratio;

                    ring_index += 1;
                    if ring_index == period {
                        ring_index = 0;
                    }

                    if i >= start_out {
                        *outp.add(i) = sum_ratio;
                    }
                }
            }

            i += 1;
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn mass_avx512_short(
    high: &[f64],
    low: &[f64],
    period: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    // For single recursive EMA chain, heavy SIMD gives no lane-parallelism.
    // Use the AVX2 prefetch variant for potential small wins.
    mass_avx2(high, low, period, first_valid_idx, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn mass_avx512_long(
    high: &[f64],
    low: &[f64],
    period: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    mass_avx2(high, low, period, first_valid_idx, out);
}

#[derive(Debug, Clone)]
pub struct MassStream {
    // config
    period: usize,

    // ring buffer for O(1) rolling-sum
    ring: Box<[f64]>,
    idx: usize,
    mask: usize, // power-of-two mask or usize::MAX when not power-of-two
    sum_ratio: f64,

    // EMA state
    alpha: f64,
    inv_alpha: f64,
    ema1: f64,
    ema2: f64,

    // progress counters / warmup thresholds
    t: usize,          // bars seen since first valid
    warm_ema2: usize,  // == 8
    warm_ratio: usize, // == 16
    warm_out: usize,   // == 16 + period - 1
}

impl MassStream {
    #[inline]
    pub fn try_new(params: MassParams) -> Result<Self, MassError> {
        let period = params.period.unwrap_or(5);
        if period == 0 {
            return Err(MassError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }

        let ring = vec![0.0; period].into_boxed_slice();
        // If period is a power of two, (i+1)&mask wraps branchlessly; else fall back to compare/reset.
        let mask = if period.is_power_of_two() {
            period - 1
        } else {
            usize::MAX
        };

        Ok(Self {
            period,
            ring,
            idx: 0,
            mask,
            sum_ratio: 0.0,

            alpha: 2.0 / 10.0, // EMA(9) => α = 2/(9+1)
            inv_alpha: 1.0 - (2.0 / 10.0),

            ema1: f64::NAN,
            ema2: f64::NAN,

            t: 0,
            warm_ema2: 8,
            warm_ratio: 16,
            warm_out: 16 + (period - 1),
        })
    }

    /// Returns None during warmup; Some(value) once we have at least 16 + period - 1 bars.
    /// O(1) per-bar: one subtraction, a couple FMAs, one division, and constant-time ring maintenance.
    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64) -> Option<f64> {
        let hl = high - low;

        // seed both EMAs on the first bar
        if self.t == 0 {
            self.ema1 = hl;
            self.ema2 = hl;
            self.t = 1;
            return None;
        }

        // ema1 = ema1*(1-α) + hl*α   (mul_add encourages FMA on CPUs that support it)
        self.ema1 = self.ema1.mul_add(self.inv_alpha, hl * self.alpha);

        // seed ema2 on the 9th bar (t == 8), then keep updating thereafter
        if self.t == self.warm_ema2 {
            self.ema2 = self.ema1;
        }
        if self.t >= self.warm_ema2 {
            self.ema2 = self.ema2.mul_add(self.inv_alpha, self.ema1 * self.alpha);
        }

        let mut out = None;

        if self.t >= self.warm_ratio {
            // exact division (matches offline kernels)
            let ratio = self.ema1 / self.ema2;

            // slide O(1) rolling sum via ring buffer (safe indexing)
            let old = self.ring[self.idx];
            self.sum_ratio = (self.sum_ratio - old) + ratio;
            self.ring[self.idx] = ratio;

            // advance ring index (branchless for power-of-two periods)
            if self.mask != usize::MAX {
                self.idx = (self.idx + 1) & self.mask;
            } else {
                self.idx += 1;
                if self.idx == self.period {
                    self.idx = 0;
                }
            }

            if self.t >= self.warm_out {
                out = Some(self.sum_ratio);
            }
        }

        self.t += 1;
        out
    }
}

#[derive(Clone, Debug)]
pub struct MassBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for MassBatchRange {
    fn default() -> Self {
        Self { period: (5, 20, 1) }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MassBatchBuilder {
    range: MassBatchRange,
    kernel: Kernel,
}

impl MassBatchBuilder {
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
    pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<MassBatchOutput, MassError> {
        mass_batch_with_kernel(high, low, &self.range, self.kernel)
    }
    pub fn with_default_slices(
        high: &[f64],
        low: &[f64],
        k: Kernel,
    ) -> Result<MassBatchOutput, MassError> {
        MassBatchBuilder::new().kernel(k).apply_slices(high, low)
    }
    pub fn apply_candles(self, c: &Candles) -> Result<MassBatchOutput, MassError> {
        let high = source_type(c, "high");
        let low = source_type(c, "low");
        self.apply_slices(high, low)
    }
    pub fn with_default_candles(c: &Candles) -> Result<MassBatchOutput, MassError> {
        MassBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c)
    }
}

pub fn mass_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    sweep: &MassBatchRange,
    k: Kernel,
) -> Result<MassBatchOutput, MassError> {
    // Batch Auto defaults to ScalarBatch to avoid mixed SIMD results; callers may request SIMD explicitly.
    let kernel = match k {
        Kernel::Auto => Kernel::ScalarBatch,
        other if other.is_batch() => other,
        _ => {
            return Err(MassError::InvalidPeriod {
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
    mass_batch_par_slice(high, low, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct MassBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<MassParams>,
    pub rows: usize,
    pub cols: usize,
}

impl MassBatchOutput {
    pub fn row_for_params(&self, p: &MassParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(5) == p.period.unwrap_or(5))
    }
    pub fn values_for(&self, p: &MassParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct MassBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct MassBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<MassParams>,
    pub rows: usize,
    pub cols: usize,
}

#[inline(always)]
fn expand_grid_mass(r: &MassBatchRange) -> Vec<MassParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(MassParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn mass_batch_slice(
    high: &[f64],
    low: &[f64],
    sweep: &MassBatchRange,
    kern: Kernel,
) -> Result<MassBatchOutput, MassError> {
    mass_batch_inner(high, low, sweep, kern, false)
}

#[inline(always)]
pub fn mass_batch_par_slice(
    high: &[f64],
    low: &[f64],
    sweep: &MassBatchRange,
    kern: Kernel,
) -> Result<MassBatchOutput, MassError> {
    mass_batch_inner(high, low, sweep, kern, true)
}

#[inline(always)]
fn mass_batch_inner(
    high: &[f64],
    low: &[f64],
    sweep: &MassBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<MassBatchOutput, MassError> {
    let combos = expand_grid_mass(sweep);
    if combos.is_empty() {
        return Err(MassError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    if high.is_empty() || low.is_empty() || high.len() != low.len() {
        return Err(MassError::DifferentLengthHL);
    }

    let first = (0..high.len())
        .find(|&i| !high[i].is_nan() && !low[i].is_nan())
        .ok_or(MassError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    let needed_bars = 16 + max_p - 1;
    if high.len() - first < needed_bars {
        return Err(MassError::NotEnoughValidData {
            needed: needed_bars,
            valid: high.len() - first,
        });
    }

    let rows = combos.len();
    let cols = high.len();

    // Use uninitialized memory with proper warmup calculation
    let mut buf_mu = make_uninit_matrix(rows, cols);

    // Calculate warmup periods for each parameter combination
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + 16 + c.period.unwrap() - 1)
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    // Convert to mutable slice for computation
    let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
    let values_slice: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    // Resolve Auto: default to Scalar for stability (explicit SIMD is still available)
    let actual_kern = match kern {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        match actual_kern {
            Kernel::Scalar => mass_row_scalar(high, low, period, first, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => mass_row_avx2(high, low, period, first, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => mass_row_avx512(high, low, period, first, out_row),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx512 => mass_row_scalar(high, low, period, first, out_row),
            _ => mass_row_scalar(high, low, period, first, out_row), // Fallback to scalar
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            values_slice
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in values_slice.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in values_slice.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    // Convert ManuallyDrop buffer to Vec
    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(MassBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn mass_row_scalar(high: &[f64], low: &[f64], period: usize, first: usize, out: &mut [f64]) {
    mass_scalar(high, low, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn mass_row_avx2(high: &[f64], low: &[f64], period: usize, first: usize, out: &mut [f64]) {
    mass_avx2(high, low, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn mass_row_avx512(high: &[f64], low: &[f64], period: usize, first: usize, out: &mut [f64]) {
    if period <= 32 {
        mass_row_avx512_short(high, low, period, first, out);
    } else {
        mass_row_avx512_long(high, low, period, first, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn mass_row_avx512_short(
    high: &[f64],
    low: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    mass_avx2(high, low, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn mass_row_avx512_long(
    high: &[f64],
    low: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    mass_avx2(high, low, period, first, out);
}

// Tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_mass_into_matches_api() {
        // Construct a small but non-trivial OHLC set (uses only high/low)
        let len = 256usize;
        let mut ts = Vec::with_capacity(len);
        let mut open = Vec::with_capacity(len);
        let mut high = Vec::with_capacity(len);
        let mut low = Vec::with_capacity(len);
        let mut close = Vec::with_capacity(len);
        let mut volume = Vec::with_capacity(len);

        for i in 0..len {
            let x = i as f64;
            // Base price oscillates mildly
            let base = (x * 0.01).mul_add(100.0, (x * 0.07).sin() * 2.0);
            // Range varies to exercise EMA of range
            let range = 1.0 + (x * 0.005).sin().abs() * 3.0;
            let h = base + range * 0.5;
            let l = base - range * 0.5;

            ts.push(i as i64);
            open.push(base);
            high.push(h);
            low.push(l);
            close.push(base * 0.999 + 0.001 * h);
            volume.push(1000.0 + (i % 10) as f64);
        }

        let candles = crate::utilities::data_loader::Candles::new(
            ts, open, high.clone(), low.clone(), close, volume,
        );

        let input = MassInput::from_candles(&candles, "high", "low", MassParams::default());

        // Baseline via Vec-returning API
        let base = mass(&input).expect("mass() should succeed");

        // Into-API into preallocated buffer
        let mut out = vec![0.0f64; len];
        mass_into(&input, &mut out).expect("mass_into() should succeed");

        assert_eq!(base.values.len(), out.len());

        // Helper: treat NaN == NaN as equal
        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b)
        }

        for i in 0..len {
            assert!(
                eq_or_both_nan(base.values[i], out[i]),
                "Mismatch at index {}: got {}, expected {}",
                i,
                out[i],
                base.values[i]
            );
        }
    }

    fn check_mass_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = MassParams { period: None };
        let input_default = MassInput::from_candles(&candles, "high", "low", default_params);
        let output_default = mass_with_kernel(&input_default, kernel)?;
        assert_eq!(output_default.values.len(), candles.high.len());
        Ok(())
    }

    fn check_mass_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = MassParams { period: Some(5) };
        let input = MassInput::from_candles(&candles, "high", "low", params);
        let mass_result = mass_with_kernel(&input, kernel)?;
        assert_eq!(
            mass_result.values.len(),
            candles.high.len(),
            "MASS length mismatch"
        );
        let expected_last_five = [
            4.512263952194651,
            4.126178935431121,
            3.838738456245828,
            3.6450956734739375,
            3.6748009093527125,
        ];
        let result_len = mass_result.values.len();
        assert!(
            result_len >= 5,
            "MASS output length is too short for comparison"
        );
        let start_idx = result_len - 5;
        let result_slice = &mass_result.values[start_idx..];
        for (i, &value) in result_slice.iter().enumerate() {
            let expected = expected_last_five[i];
            assert!(
                (value - expected).abs() < 1e-7,
                "MASS mismatch at index {}: expected {}, got {}",
                start_idx + i,
                expected,
                value
            );
        }
        Ok(())
    }

    fn check_mass_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = MassInput::with_default_candles(&candles);
        match input.data {
            MassData::Candles {
                high_source,
                low_source,
                ..
            } => {
                assert_eq!(high_source, "high");
                assert_eq!(low_source, "low");
            }
            _ => panic!("Expected MassData::Candles variant"),
        }
        let output = mass_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.high.len());
        Ok(())
    }

    fn check_mass_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high_data = [10.0, 15.0, 20.0];
        let low_data = [5.0, 10.0, 12.0];
        let params = MassParams { period: Some(0) };
        let input = MassInput::from_slices(&high_data, &low_data, params);
        let result = mass_with_kernel(&input, kernel);
        assert!(result.is_err(), "Expected an error for zero period");
        Ok(())
    }

    fn check_mass_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high_data = [10.0, 15.0, 20.0];
        let low_data = [5.0, 10.0, 12.0];
        let params = MassParams { period: Some(10) };
        let input = MassInput::from_slices(&high_data, &low_data, params);
        let result = mass_with_kernel(&input, kernel);
        assert!(result.is_err(), "Expected an error for period > data.len()");
        Ok(())
    }

    fn check_mass_very_small_data_set(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high_data = [10.0];
        let low_data = [5.0];
        let params = MassParams { period: Some(5) };
        let input = MassInput::from_slices(&high_data, &low_data, params);
        let result = mass_with_kernel(&input, kernel);
        assert!(
            result.is_err(),
            "Expected error for data smaller than needed bars"
        );
        Ok(())
    }

    fn check_mass_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = MassParams { period: Some(5) };
        let first_input = MassInput::from_candles(&candles, "high", "low", first_params);
        let first_result = mass_with_kernel(&first_input, kernel)?;
        let second_params = MassParams { period: Some(5) };
        let second_input =
            MassInput::from_slices(&first_result.values, &first_result.values, second_params);
        let second_result = mass_with_kernel(&second_input, kernel)?;
        assert_eq!(
            second_result.values.len(),
            first_result.values.len(),
            "Second MASS output length mismatch"
        );
        Ok(())
    }

    fn check_mass_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 5;
        let params = MassParams {
            period: Some(period),
        };
        let input = MassInput::from_candles(&candles, "high", "low", params);
        let mass_result = mass_with_kernel(&input, kernel)?;
        assert_eq!(
            mass_result.values.len(),
            candles.high.len(),
            "MASS length mismatch"
        );
        if mass_result.values.len() > 240 {
            for i in 240..mass_result.values.len() {
                assert!(
                    !mass_result.values[i].is_nan(),
                    "Expected no NaN after index 240, but found NaN at index {}",
                    i
                );
            }
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_mass_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Comprehensive parameter combinations for mass
        let test_params = vec![
            MassParams::default(),            // period: 5
            MassParams { period: Some(2) },   // minimum period
            MassParams { period: Some(3) },   // small
            MassParams { period: Some(4) },   // small
            MassParams { period: Some(5) },   // default
            MassParams { period: Some(10) },  // medium
            MassParams { period: Some(20) },  // medium-large
            MassParams { period: Some(30) },  // large
            MassParams { period: Some(50) },  // large
            MassParams { period: Some(100) }, // very large
            MassParams { period: Some(200) }, // very large
            MassParams { period: Some(255) }, // edge case
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = MassInput::from_candles(&candles, "high", "low", params.clone());
            let output = mass_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
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
						 with params: period={} (param set {})",
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
						 with params: period={} (param set {})",
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
    fn check_mass_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(()) // No-op in release builds
    }

    #[cfg(test)]
    fn check_mass_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Generate realistic test parameters for Mass Index
        let strat = (2usize..=100)
            .prop_flat_map(|period| {
                (
                    // Generate high/low data with proper relationship
                    prop::collection::vec(
                        (0f64..1000f64).prop_filter("finite", |x| x.is_finite()),
                        (16 + period)..=500, // Need at least 16 + period for warmup
                    ),
                    Just(period),
                    // Add a scenario selector for different data patterns
                    0usize..=6,
                )
            })
            .prop_map(|(mut base_data, period, scenario)| {
                // Generate high and low arrays based on scenario
                let mut high = Vec::with_capacity(base_data.len());
                let mut low = Vec::with_capacity(base_data.len());

                match scenario {
                    0 => {
                        // Random data with realistic high/low relationship
                        for val in base_data {
                            let range = val * 0.1; // 10% range
                            high.push(val + range / 2.0);
                            low.push(val - range / 2.0);
                        }
                    }
                    1 => {
                        // Zero range (high == low)
                        for val in base_data {
                            high.push(val);
                            low.push(val);
                        }
                    }
                    2 => {
                        // Constant range
                        let constant_range = 10.0;
                        for val in base_data {
                            high.push(val + constant_range / 2.0);
                            low.push(val - constant_range / 2.0);
                        }
                    }
                    3 => {
                        // Increasing volatility (but keep it reasonable)
                        for (i, val) in base_data.iter().enumerate() {
                            let range = 1.0 + (i as f64 * 0.1).min(20.0); // Cap at 20
                            high.push(val + range);
                            low.push(val - range);
                        }
                    }
                    4 => {
                        // Decreasing volatility
                        for (i, val) in base_data.iter().enumerate() {
                            let range = (20.0 - (i as f64 * 0.1)).max(0.5); // Min at 0.5
                            high.push(val + range);
                            low.push(val - range);
                        }
                    }
                    5 => {
                        // Data with realistic volatility spikes (10x normal volatility)
                        for (i, val) in base_data.iter().enumerate() {
                            let range = if i % 20 == 0 { 50.0 } else { 5.0 }; // 10x spike, not 100x
                            high.push(val + range);
                            low.push(val - range);
                        }
                    }
                    6 => {
                        // Gradually approaching zero range (edge case)
                        for (i, val) in base_data.iter().enumerate() {
                            // Range decreases exponentially toward zero
                            let range = 10.0 * (0.95_f64).powi(i as i32);
                            high.push(val + range);
                            low.push(val - range);
                        }
                    }
                    _ => unreachable!(),
                }

                (high, low, period)
            });

        proptest::test_runner::TestRunner::default()
			.run(&strat, |(high, low, period)| {
				let params = MassParams { period: Some(period) };
				let input = MassInput::from_slices(&high, &low, params);

				// Calculate output with the test kernel
				let MassOutput { values: out } =
					mass_with_kernel(&input, kernel).unwrap();

				// Calculate reference output with scalar kernel
				let MassOutput { values: ref_out } =
					mass_with_kernel(&input, Kernel::Scalar).unwrap();

				// Property 1: Warmup period validation
				// First 16 + period - 1 values should be NaN
				let warmup_end = 16 + period - 1;
				for i in 0..warmup_end.min(high.len()) {
					prop_assert!(
						out[i].is_nan(),
						"Expected NaN during warmup at index {}, got {}", i, out[i]
					);
				}

				// Test properties for each valid output value
				for i in warmup_end..high.len() {
					let y = out[i];
					let r = ref_out[i];

					// Property 2: Kernel consistency
					// Different kernels should produce identical or near-identical results
					if y.is_finite() && r.is_finite() {
						let y_bits = y.to_bits();
						let r_bits = r.to_bits();
						let ulp_diff: u64 = y_bits.abs_diff(r_bits);

						prop_assert!(
							(y - r).abs() <= 1e-9 || ulp_diff <= 8,
							"Kernel mismatch at idx {}: {} vs {} (ULP={})",
							i, y, r, ulp_diff
						);
					} else {
						// Both should be NaN or both should be finite
						prop_assert_eq!(
							y.is_nan(), r.is_nan(),
							"NaN mismatch at idx {}: {} vs {}", i, y, r
						);
					}

					// Property 3: Bounded output
					// Mass Index should be positive and typically between 0 and 2.5*period
					if y.is_finite() {
						prop_assert!(
							y > 0.0,
							"Mass Index should be positive at idx {}, got {}", i, y
						);

						// Mass Index rarely exceeds 2.5*period in realistic conditions
						prop_assert!(
							y <= (period as f64) * 2.5,
							"Mass Index unusually high at idx {}: {} (period={})", i, y, period
						);
					}

					// Property 4: Stable range property (combines zero and constant range)
					// When range is stable (constant or zero), Mass Index approaches period
					let window_start = i.saturating_sub(period - 1);
					let window_end = i + 1;
					let ranges: Vec<f64> = (window_start..window_end)
						.map(|j| high[j] - low[j])
						.collect();

					// Check if range is constant across the window
					let is_constant_range = ranges.windows(2)
						.all(|w| (w[0] - w[1]).abs() < 1e-9);

					// Only check convergence if we're well past the warmup and range has been stable
					// Need at least 2*period bars after range becomes stable for convergence
					if is_constant_range && y.is_finite() && i >= warmup_end + 2 * period {
						let avg_range = ranges.iter().sum::<f64>() / ranges.len() as f64;

						// For zero or near-zero range
						if avg_range < f64::EPSILON {
							prop_assert!(
								(y - period as f64).abs() <= 1e-6,
								"Zero range Mass Index should be ~{} at idx {}, got {}", period, i, y
							);
						}
						// For constant non-zero range with reasonable size
						else if avg_range > 0.01 && avg_range < 100.0 {
							// Only check when range is in a reasonable band
							// Realistic tolerance: Mass Index converges gradually to period
							let tolerance = (period as f64) * 0.2 + 2.0;
							prop_assert!(
								(y - period as f64).abs() <= tolerance,
								"Constant range Mass Index should be close to {} at idx {}, got {} (tolerance: {})",
								period, i, y, tolerance
							);
						}
					}

					// Property 5: High >= Low constraint
					// Verify the indicator respects high >= low relationship
					for j in window_start..window_end {
						prop_assert!(
							high[j] >= low[j] - f64::EPSILON,
							"High should be >= Low at index {}: high={}, low={}", j, high[j], low[j]
						);
					}

					// Property 6: No infinite values
					// All non-warmup outputs should be finite (not infinite)
					prop_assert!(
						!y.is_infinite(),
						"Found infinite value at idx {}: {}", i, y
					);

					// Property 7: Volatility response pattern
					// Mass Index responds to volatility changes with a predictable pattern
					if i >= warmup_end + period && y.is_finite() {
						// Calculate average range in current window
						let avg_range = ranges.iter().sum::<f64>() / ranges.len() as f64;

						// For very low volatility (near zero range), Mass Index converges toward period
						// Note: convergence is gradual and depends on the EMA smoothing
						if avg_range < 0.001 {
							// More realistic tolerance based on how far from zero the range is
							let tolerance = if avg_range < 1e-10 {
								1.0
							} else {
								// Scale tolerance with period for very small but non-zero ranges
								(period as f64) * 0.25 + 2.0
							};
							prop_assert!(
								(y - period as f64).abs() <= tolerance,
								"Low volatility Mass Index should be near {} at idx {}, got {} (avg_range: {}, tolerance: {})",
								period, i, y, avg_range, tolerance
							);
						}

						// For significant volatility changes, verify response
						if i > warmup_end + period + 5 {
							// Compare current and previous window volatility
							let prev_window_start = (i - 5).saturating_sub(period - 1);
							let prev_window_end = i - 4;
							let prev_ranges: Vec<f64> = (prev_window_start..prev_window_end)
								.map(|j| high[j] - low[j])
								.collect();
							let prev_avg_range = prev_ranges.iter().sum::<f64>() / prev_ranges.len() as f64;

							// If volatility doubled, Mass Index should show some increase
							if avg_range > prev_avg_range * 2.0 && prev_avg_range > 0.1 {
								let prev_mass = out[i - 5];
								if prev_mass.is_finite() {
									prop_assert!(
										y >= prev_mass - 0.5, // Allow some lag but expect general increase
										"Mass Index should respond to doubling volatility: {} at idx {} vs {} at idx {}",
										y, i, prev_mass, i - 5
									);
								}
							}
						}

						// General bounds check
						prop_assert!(
							y >= (period as f64) * 0.3 && y <= (period as f64) * 2.5,
							"Mass Index out of reasonable bounds at idx {}: {} (period={})",
							i, y, period
						);
					}
				}

				Ok(())
			})
			.unwrap();

        Ok(())
    }

    macro_rules! generate_all_mass_tests {
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

    generate_all_mass_tests!(
        check_mass_partial_params,
        check_mass_accuracy,
        check_mass_default_candles,
        check_mass_zero_period,
        check_mass_period_exceeds_length,
        check_mass_very_small_data_set,
        check_mass_reinput,
        check_mass_nan_handling,
        check_mass_no_poison
    );

    #[cfg(test)]
    generate_all_mass_tests!(check_mass_property);
    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file)?;
        let output = MassBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&candles)?;
        let def = MassParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), candles.high.len());

        let expected = [
            4.512263952194651,
            4.126178935431121,
            3.838738456245828,
            3.6450956734739375,
            3.6748009093527125,
        ];
        let start = row.len().saturating_sub(5);
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-7,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test various parameter sweep configurations
        let test_configs = vec![
            (2, 10, 2),    // Small periods with step 2
            (5, 25, 5),    // Medium periods with step 5
            (30, 60, 15),  // Large periods with step 15
            (2, 5, 1),     // Dense small range
            (10, 10, 0),   // Static period
            (50, 100, 25), // Very large periods
            (3, 15, 3),    // Another small/medium range
            (20, 40, 10),  // Medium range
        ];

        for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
            let output = MassBatchBuilder::new()
                .kernel(kernel)
                .period_range(p_start, p_end, p_step)
                .apply_candles(&c)?;

            for (idx, &val) in output.values.iter().enumerate() {
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
						 at row {} col {} (flat index {}) with params: period={}",
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
						 at row {} col {} (flat index {}) with params: period={}",
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
						 at row {} col {} (flat index {}) with params: period={}",
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
    fn check_batch_no_poison(
        _test: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    // Macro for batch testing like alma.rs
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
#[pyfunction(name = "mass")]
#[pyo3(signature = (high, low, period, kernel=None))]
pub fn mass_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = MassParams {
        period: Some(period),
    };
    let input = MassInput::from_slices(high_slice, low_slice, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| mass_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "MassStream")]
pub struct MassStreamPy {
    stream: MassStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl MassStreamPy {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        let params = MassParams {
            period: Some(period),
        };
        let stream =
            MassStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(MassStreamPy { stream })
    }

    fn update(&mut self, high: f64, low: f64) -> Option<f64> {
        self.stream.update(high, low)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "mass_batch")]
#[pyo3(signature = (high, low, period_range, kernel=None))]
pub fn mass_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;

    let sweep = MassBatchRange {
        period: period_range,
    };

    let combos = expand_grid_mass(&sweep);
    let rows = combos.len();
    let cols = high_slice.len();

    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    let kern = validate_kernel(kernel, true)?;

    let combos = py
        .allow_threads(|| {
            let kernel = match kern {
                Kernel::Auto => Kernel::ScalarBatch,
                k => k,
            };
            let simd = match kernel {
                Kernel::Avx512Batch => Kernel::Avx512,
                Kernel::Avx2Batch => Kernel::Avx2,
                Kernel::ScalarBatch => Kernel::Scalar,
                _ => unreachable!(),
            };
            mass_batch_inner_into(high_slice, low_slice, &sweep, simd, true, slice_out)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
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

// ---------------- CUDA Python bindings ----------------
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "mass_cuda_batch_dev")]
#[pyo3(signature = (high_f32, low_f32, period_range, device_id=0))]
pub fn mass_cuda_batch_dev_py<'py>(
    py: Python<'py>,
    high_f32: numpy::PyReadonlyArray1<'py, f32>,
    low_f32: numpy::PyReadonlyArray1<'py, f32>,
    period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<(DeviceArrayF32Py, Bound<'py, pyo3::types::PyDict>)> {
    use numpy::{IntoPyArray, PyArrayMethods};
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let high = high_f32.as_slice()?;
    let low = low_f32.as_slice()?;
    let sweep = MassBatchRange {
        period: period_range,
    };

    let (inner, combos) = py.allow_threads(|| {
        let mut cuda =
            CudaMass::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.mass_batch_dev(high, low, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    let dict = pyo3::types::PyDict::new(py);
    let periods: Vec<u64> = combos
        .iter()
        .map(|c| c.period.unwrap_or(0) as u64)
        .collect();
    dict.set_item("periods", periods.into_pyarray(py))?;

    Ok((DeviceArrayF32Py { inner }, dict))
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "mass_cuda_many_series_one_param_dev")]
#[pyo3(signature = (high_tm_f32, low_tm_f32, period, device_id=0))]
pub fn mass_cuda_many_series_one_param_dev_py<'py>(
    py: Python<'py>,
    high_tm_f32: numpy::PyReadonlyArray2<'py, f32>,
    low_tm_f32: numpy::PyReadonlyArray2<'py, f32>,
    period: usize,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    use numpy::PyUntypedArrayMethods;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let hs = high_tm_f32.shape();
    let ls = low_tm_f32.shape();
    if hs != ls || hs.len() != 2 {
        return Err(PyValueError::new_err("expected matching 2D arrays"));
    }
    let rows = hs[0];
    let cols = hs[1];
    let high = high_tm_f32.as_slice()?;
    let low = low_tm_f32.as_slice()?;
    let params = MassParams {
        period: Some(period),
    };

    let inner = py.allow_threads(|| {
        let mut cuda =
            CudaMass::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.mass_many_series_one_param_time_major_dev(high, low, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(DeviceArrayF32Py { inner })
}

#[cfg(feature = "python")]
pub fn register_mass_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mass_py, m)?)?;
    m.add_function(wrap_pyfunction!(mass_batch_py, m)?)?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(mass_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(mass_cuda_many_series_one_param_dev_py, m)?)?;
    }
    Ok(())
}

#[cfg(any(feature = "python", feature = "wasm"))]
#[inline(always)]
fn mass_batch_inner_into(
    high: &[f64],
    low: &[f64],
    sweep: &MassBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<MassParams>, MassError> {
    let combos = expand_grid_mass(sweep);
    if combos.is_empty() {
        return Err(MassError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    if high.is_empty() || low.is_empty() || high.len() != low.len() {
        return Err(MassError::DifferentLengthHL);
    }

    let first = (0..high.len())
        .find(|&i| !high[i].is_nan() && !low[i].is_nan())
        .ok_or(MassError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    let needed_bars = 16 + max_p - 1;
    if high.len() - first < needed_bars {
        return Err(MassError::NotEnoughValidData {
            needed: needed_bars,
            valid: high.len() - first,
        });
    }

    let cols = high.len();

    // Initialize NaN prefixes for each row based on warmup period
    // This is necessary because the buffer comes from Python/WASM and wasn't created by our helpers
    for (row, combo) in combos.iter().enumerate() {
        let period = combo.period.unwrap();
        let warmup_end = first + 16 + period - 1;
        let row_start = row * cols;
        for i in 0..warmup_end.min(cols) {
            out[row_start + i] = f64::NAN;
        }
    }

    // Resolve Auto to Scalar for stability (explicit SIMD still allowed)
    let actual_kern = match kern {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        match actual_kern {
            Kernel::Scalar => mass_row_scalar(high, low, period, first, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => mass_row_avx2(high, low, period, first, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => mass_row_avx512(high, low, period, first, out_row),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx512 => mass_row_scalar(high, low, period, first, out_row),
            _ => mass_row_scalar(high, low, period, first, out_row), // Fallback to scalar
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
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

// WASM bindings
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn mass_js(high: &[f64], low: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = MassParams {
        period: Some(period),
    };
    let input = MassInput::from_slices(high, low, params);

    let mut output = vec![0.0; high.len()];

    mass_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn mass_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to mass_into"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);

        if period == 0 || period > len {
            return Err(JsValue::from_str("Invalid period"));
        }

        let params = MassParams {
            period: Some(period),
        };
        let input = MassInput::from_slices(high, low, params);

        // Check for aliasing with either input
        if high_ptr == out_ptr || low_ptr == out_ptr {
            let mut temp = vec![0.0; len];
            mass_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            mass_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn mass_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn mass_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = mass_batch)]
pub fn mass_batch_unified_js(
    high: &[f64],
    low: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: MassBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = MassBatchRange {
        period: config.period_range,
    };

    let output = mass_batch_inner(high, low, &sweep, Kernel::Auto, false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = MassBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };

    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn mass_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to mass_batch_into"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);

        let sweep = MassBatchRange {
            period: (period_start, period_end, period_step),
        };

        let combos = expand_grid_mass(&sweep);
        let rows = combos.len();
        let cols = len;

        let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);

        mass_batch_inner_into(high, low, &sweep, Kernel::Auto, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct MassStreamWasm {
    stream: MassStream,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl MassStreamWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(period: usize) -> Result<MassStreamWasm, JsValue> {
        let params = MassParams {
            period: Some(period),
        };
        let stream = MassStream::try_new(params).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(MassStreamWasm { stream })
    }

    pub fn update(&mut self, high: f64, low: f64) -> Option<f64> {
        self.stream.update(high, low)
    }
}
