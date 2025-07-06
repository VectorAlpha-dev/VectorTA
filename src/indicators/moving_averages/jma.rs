//! # Jurik Moving Average (JMA)
//!
//! A minimal-lag smoothing methodology developed by Mark Jurik. JMA adapts quickly
//! to market moves while reducing noise. Parameters (`period`, `phase`, `power`)
//! control window size, phase shift, and smoothing aggressiveness.
//!
//! ## Parameters
//! - **period**: Window size (number of data points).
//! - **phase**: Shift in [-100.0, 100.0], curve displacement (default: 50.0).
//! - **power**: Exponent for smoothing ratio (default: 2).
//!
//! ## Errors
//! - **AllValuesNaN**: jma: All input data values are `NaN`.
//! - **InvalidPeriod**: jma: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: jma: Not enough valid data points for the requested `period`.
//! - **InvalidPhase**: jma: `phase` is `NaN` or infinite.
//!
//! ## Returns
//! - **`Ok(JmaOutput)`** on success, containing a `Vec<f64>`.
//! - **`Err(JmaError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel, alloc_with_nan_prefix, make_uninit_matrix, init_matrix_prefixes};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;
use std::mem::MaybeUninit;  

impl<'a> AsRef<[f64]> for JmaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            JmaData::Slice(slice) => slice,
            JmaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum JmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct JmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct JmaParams {
    pub period: Option<usize>,
    pub phase: Option<f64>,
    pub power: Option<u32>,
}

impl Default for JmaParams {
    fn default() -> Self {
        Self {
            period: Some(7),
            phase: Some(50.0),
            power: Some(2),
        }
    }
}

#[derive(Debug, Clone)]
pub struct JmaInput<'a> {
    pub data: JmaData<'a>,
    pub params: JmaParams,
}

impl<'a> JmaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: JmaParams) -> Self {
        Self {
            data: JmaData::Candles { candles: c, source: s },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: JmaParams) -> Self {
        Self { data: JmaData::Slice(sl), params: p }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", JmaParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize { self.params.period.unwrap_or(7) }
    #[inline]
    pub fn get_phase(&self) -> f64 { self.params.phase.unwrap_or(50.0) }
    #[inline]
    pub fn get_power(&self) -> u32 { self.params.power.unwrap_or(2) }
}

#[derive(Copy, Clone, Debug)]
pub struct JmaBuilder {
    period: Option<usize>,
    phase: Option<f64>,
    power: Option<u32>,
    kernel: Kernel,
}

impl Default for JmaBuilder {
    fn default() -> Self {
        Self { period: None, phase: None, power: None, kernel: Kernel::Auto }
    }
}

impl JmaBuilder {
    #[inline(always)]
    pub fn new() -> Self { Self::default() }
    #[inline(always)]
    pub fn period(mut self, n: usize) -> Self { self.period = Some(n); self }
    #[inline(always)]
    pub fn phase(mut self, x: f64) -> Self { self.phase = Some(x); self }
    #[inline(always)]
    pub fn power(mut self, p: u32) -> Self { self.power = Some(p); self }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self { self.kernel = k; self }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<JmaOutput, JmaError> {
        let p = JmaParams { period: self.period, phase: self.phase, power: self.power };
        let i = JmaInput::from_candles(c, "close", p);
        jma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<JmaOutput, JmaError> {
        let p = JmaParams { period: self.period, phase: self.phase, power: self.power };
        let i = JmaInput::from_slice(d, p);
        jma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<JmaStream, JmaError> {
        let p = JmaParams { period: self.period, phase: self.phase, power: self.power };
        JmaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum JmaError {
    #[error("jma: All values are NaN.")]
    AllValuesNaN,
    #[error("jma: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("jma: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("jma: Invalid phase: {phase}")]
    InvalidPhase { phase: f64 },
    #[error("jma: Invalid output buffer size: expected = {expected}, actual = {actual}")]
    InvalidOutputBuffer { expected: usize, actual: usize },
}

#[inline]
pub fn jma(input: &JmaInput) -> Result<JmaOutput, JmaError> {
    jma_with_kernel(input, Kernel::Auto)
}

pub fn jma_with_kernel(input: &JmaInput, kernel: Kernel) -> Result<JmaOutput, JmaError> {
    let data: &[f64] = match &input.data {
        JmaData::Candles { candles, source } => source_type(candles, source),
        JmaData::Slice(sl) => sl,
    };
    let first = data.iter().position(|x| !x.is_nan()).ok_or(JmaError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();
    let phase = input.get_phase();
    let power = input.get_power();

    if period == 0 || period > len {
        return Err(JmaError::InvalidPeriod { period, data_len: len });
    }
    if (len - first) < period {
        return Err(JmaError::NotEnoughValidData { needed: period, valid: len - first });
    }
    if phase.is_nan() || phase.is_infinite() {
        return Err(JmaError::InvalidPhase { phase });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let warm = first + period;                 // first valid + look-back window
    let mut out = alloc_with_nan_prefix(len, warm);
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                jma_scalar(data, period, phase, power, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                jma_avx2(data, period, phase, power, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                jma_avx512(data, period, phase, power, first, &mut out)
            }
            _ => unreachable!(),
        }
    }
    Ok(JmaOutput { values: out })
}

#[inline]
pub fn jma_into(input: &JmaInput, out: &mut [f64]) -> Result<(), JmaError> {
    jma_with_kernel_into(input, Kernel::Auto, out)
}

pub fn jma_with_kernel_into(
    input: &JmaInput,
    kernel: Kernel,
    out: &mut [f64],
) -> Result<(), JmaError> {
    let data: &[f64] = match &input.data {
        JmaData::Candles { candles, source } => source_type(candles, source),
        JmaData::Slice(sl) => sl,
    };
    let len = data.len();
    
    // Ensure output buffer is the correct size
    if out.len() != len {
        return Err(JmaError::InvalidOutputBuffer {
            expected: len,
            actual: out.len(),
        });
    }
    
    let first = data.iter().position(|x| !x.is_nan()).ok_or(JmaError::AllValuesNaN)?;
    let period = input.get_period();
    let phase = input.get_phase();
    let power = input.get_power();

    if period == 0 || period > len {
        return Err(JmaError::InvalidPeriod { period, data_len: len });
    }
    if (len - first) < period {
        return Err(JmaError::NotEnoughValidData { needed: period, valid: len - first });
    }
    if phase.is_nan() || phase.is_infinite() {
        return Err(JmaError::InvalidPhase { phase });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let warm = first + period;
    // Initialize NaN prefix
    out[..warm].fill(f64::NAN);

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                jma_scalar(data, period, phase, power, first, out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                jma_avx2(data, period, phase, power, first, out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                jma_avx512(data, period, phase, power, first, out)
            }
            _ => unreachable!(),
        }
    }
    Ok(())
}

#[inline]
pub fn jma_scalar(
    data: &[f64],
    period: usize,
    phase: f64,
    power: u32,
    first_valid: usize,
    output: &mut [f64],
) {
    assert_eq!(data.len(), output.len());
    assert!(first_valid < data.len());

    let pr = if phase < -100.0 {
        0.5
    } else if phase > 100.0 {
        2.5
    
        } else {
        phase / 100.0 + 1.5
    };

    let beta = {
        let num = 0.45 * (period as f64 - 1.0);
        num / (num + 2.0)
    };
    let one_minus_beta = 1.0 - beta;

    let alpha = beta.powi(power as i32);
    let one_minus_alpha = 1.0 - alpha;
    let alpha_sq = alpha * alpha;
    let oma_sq = one_minus_alpha * one_minus_alpha;

    let mut e0 = data[first_valid];
    let mut e1 = 0.0;
    let mut e2 = 0.0;
    let mut j_prev = data[first_valid];

    output[first_valid] = j_prev;

    unsafe {
        for i in (first_valid + 1)..data.len() {
            let price = *data.get_unchecked(i);

            e0 = one_minus_alpha * price + alpha * e0;

            e1 = (price - e0) * one_minus_beta + beta * e1;
            let diff = e0 + pr * e1 - j_prev;

            e2 = diff * oma_sq + alpha_sq * e2;
            let j = j_prev + e2;

            *output.get_unchecked_mut(i) = j;

            j_prev = j;
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn jma_avx2(
    data:        &[f64],
    period:      usize,
    phase:       f64,
    power:       u32,
    first_valid: usize,
    output:         &mut [f64],
) {
    assert_eq!(data.len(), output.len());
    assert!(first_valid < data.len());

    let pr = if phase < -100.0 {
        0.5
    } else if phase > 100.0 {
        2.5
    
        } else {
        phase / 100.0 + 1.5
    };

    let beta = {
        let num = 0.45 * (period as f64 - 1.0);
        num / (num + 2.0)
    };
    let one_minus_beta = 1.0 - beta;

    let alpha = beta.powi(power as i32);
    let one_minus_alpha = 1.0 - alpha;
    let alpha_sq = alpha * alpha;
    let oma_sq = one_minus_alpha * one_minus_alpha;

    let mut e0 = data[first_valid];
    let mut e1 = 0.0;
    let mut e2 = 0.0;
    let mut j_prev = e0;

    output[first_valid] = j_prev;

    unsafe {
        for i in (first_valid + 1)..data.len() {
            let price = *data.get_unchecked(i);

            e0 = one_minus_alpha.mul_add(price, alpha * e0);
            e1 = (price - e0).mul_add(one_minus_beta, beta * e1);

            let diff = e0 + pr * e1 - j_prev;
            e2 = diff.mul_add(oma_sq, alpha_sq * e2);

            j_prev += e2;
            *output.get_unchecked_mut(i) = j_prev;
        }
    }
}

#[inline(always)]
fn jma_consts(period: usize, phase: f64, power: u32) -> (f64, f64, f64, f64, f64, f64, f64) {
    let pr = if phase < -100.0 {
        0.5
    } else if phase > 100.0 {
        2.5
    
        } else {
        phase / 100.0 + 1.5
    };

    let beta = {
        let num = 0.45 * (period as f64 - 1.0);
        num / (num + 2.0)
    };
    let alpha = beta.powi(power as i32);
    (
        pr,
        beta,
        alpha,
        alpha * alpha,
        (1.0 - alpha) * (1.0 - alpha),
        1.0 - alpha,
        1.0 - beta,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512dq,avx512vl,fma")]
#[inline]
pub unsafe fn jma_avx512(
    data: &[f64],
    period: usize,
    phase: f64,
    power: u32,
    first_valid: usize,
    out: &mut [f64],
) {
    debug_assert!(data.len() == out.len() && first_valid < data.len());

    let (pr, beta, alpha, alpha_sq, oma_sq, one_minus_alpha, one_minus_beta) =
        jma_consts(period, phase, power);

    let pr_v = _mm512_set1_pd(pr);
    let oma_sq_v = _mm512_set1_pd(oma_sq);
    let alpha_sq_v = _mm512_set1_pd(alpha_sq);
    let one_minus_alpha_v = _mm512_set1_pd(one_minus_alpha);
    let alpha_v = _mm512_set1_pd(alpha);
    let one_minus_beta_v = _mm512_set1_pd(one_minus_beta);
    let beta_v = _mm512_set1_pd(beta);

    let mut e0 = data[first_valid];
    let mut e1 = 0.0;
    let mut e2 = 0.0;
    let mut j_prev = e0;

    out[first_valid] = j_prev;

    let mut i = first_valid + 1;
    let n = data.len();

    // Unroll loop by 4 (as an example to leverage AVX512 register space for ILP)
    while i + 3 < n {
        for k in 0..4 {
            let price = *data.get_unchecked(i + k);

            e0 = one_minus_alpha.mul_add(price, alpha * e0);
            e1 = (price - e0).mul_add(one_minus_beta, beta * e1);
            let diff = e0 + pr * e1 - j_prev;
            e2 = diff.mul_add(oma_sq, alpha_sq * e2);
            j_prev += e2;

            *out.get_unchecked_mut(i + k) = j_prev;
        }
        i += 4;
    }

    // Scalar tail for remaining elements
    while i < n {
        let price = *data.get_unchecked(i);
        e0 = one_minus_alpha.mul_add(price, alpha * e0);
        e1 = (price - e0).mul_add(one_minus_beta, beta * e1);
        let diff = e0 + pr * e1 - j_prev;
        e2 = diff.mul_add(oma_sq, alpha_sq * e2);
        j_prev += e2;

        *out.get_unchecked_mut(i) = j_prev;
        i += 1;
    }
}

// ===== BATCH & STREAMING API =====

#[derive(Debug, Clone)]
pub struct JmaStream {
    period: usize,
    phase: f64,
    power: u32,
    alpha: f64,
    beta: f64,
    phase_ratio: f64,
    initialized: bool,
    e0: f64,
    e1: f64,
    e2: f64,
    jma_prev: f64,
}

impl JmaStream {
    pub fn try_new(params: JmaParams) -> Result<Self, JmaError> {
        let period = params.period.unwrap_or(7);
        if period == 0 {
            return Err(JmaError::InvalidPeriod { period, data_len: 0 });
        }
        let phase = params.phase.unwrap_or(50.0);
        if phase.is_nan() || phase.is_infinite() {
            return Err(JmaError::InvalidPhase { phase });
        }
        let power = params.power.unwrap_or(2);
        let phase_ratio = if phase < -100.0 {
            0.5
        } else if phase > 100.0 {
            2.5
        
            } else {
            (phase / 100.0) + 1.5
        };
        let beta = {
            let numerator = 0.45 * (period as f64 - 1.0);
            let denominator = numerator + 2.0;
            if denominator.abs() < f64::EPSILON { 0.0 } else { numerator / denominator }
        };
        let alpha = beta.powi(power as i32);
        Ok(Self {
            period,
            phase,
            power,
            alpha,
            beta,
            phase_ratio,
            initialized: false,
            e0: f64::NAN,
            e1: 0.0,
            e2: 0.0,
            jma_prev: f64::NAN,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        if !self.initialized {
            if value.is_nan() {
                return None;
            }
            self.initialized = true;
            self.e0 = value;
            self.e1 = 0.0;
            self.e2 = 0.0;
            self.jma_prev = value;
            return Some(value);
        }
        let src = value;
        self.e0 = (1.0 - self.alpha) * src + self.alpha * self.e0;
        self.e1 = (src - self.e0) * (1.0 - self.beta) + self.beta * self.e1;
        let diff = self.e0 + self.phase_ratio * self.e1 - self.jma_prev;
        self.e2 = diff * (1.0 - self.alpha).powi(2) + self.alpha.powi(2) * self.e2;
        self.jma_prev = self.e2 + self.jma_prev;
        Some(self.jma_prev)
    }
}

#[derive(Clone, Debug)]
pub struct JmaBatchRange {
    pub period: (usize, usize, usize),
    pub phase: (f64, f64, f64),
    pub power: (u32, u32, u32),
}

impl Default for JmaBatchRange {
    fn default() -> Self {
        Self {
            period: (7, 240, 1),
            phase: (50.0, 50.0, 0.0),
            power: (2, 2, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct JmaBatchBuilder {
    range: JmaBatchRange,
    kernel: Kernel,
}

impl JmaBatchBuilder {
    pub fn new() -> Self { Self::default() }
    pub fn kernel(mut self, k: Kernel) -> Self { self.kernel = k; self }
    #[inline]
    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self { self.range.period = (start, end, step); self }
    #[inline]
    pub fn period_static(mut self, p: usize) -> Self { self.range.period = (p, p, 0); self }
    #[inline]
    pub fn phase_range(mut self, start: f64, end: f64, step: f64) -> Self { self.range.phase = (start, end, step); self }
    #[inline]
    pub fn phase_static(mut self, x: f64) -> Self { self.range.phase = (x, x, 0.0); self }
    #[inline]
    pub fn power_range(mut self, start: u32, end: u32, step: u32) -> Self { self.range.power = (start, end, step); self }
    #[inline]
    pub fn power_static(mut self, p: u32) -> Self { self.range.power = (p, p, 0); self }
    pub fn apply_slice(self, data: &[f64]) -> Result<JmaBatchOutput, JmaError> {
        jma_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<JmaBatchOutput, JmaError> {
        JmaBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<JmaBatchOutput, JmaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<JmaBatchOutput, JmaError> {
        JmaBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
    }
}

pub fn jma_batch_with_kernel(
    data: &[f64],
    sweep: &JmaBatchRange,
    k: Kernel,
) -> Result<JmaBatchOutput, JmaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(JmaError::InvalidPeriod { period: 0, data_len: 0 }),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    jma_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct JmaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<JmaParams>,
    pub rows: usize,
    pub cols: usize,
}

impl JmaBatchOutput {
    pub fn row_for_params(&self, p: &JmaParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(7) == p.period.unwrap_or(7)
                && (c.phase.unwrap_or(50.0) - p.phase.unwrap_or(50.0)).abs() < 1e-12
                && c.power.unwrap_or(2) == p.power.unwrap_or(2)
        })
    }
    pub fn values_for(&self, p: &JmaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &JmaBatchRange) -> Vec<JmaParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end { return vec![start]; }
        (start..=end).step_by(step).collect()
    }
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Vec<f64> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 { return vec![start]; }
        let mut v = Vec::new();
        let mut x = start;
        while x <= end + 1e-12 {
            v.push(x);
            x += step;
        }
        v
    }
    fn axis_u32((start, end, step): (u32, u32, u32)) -> Vec<u32> {
        if step == 0 || start == end { return vec![start]; }
        (start..=end).step_by(step as usize).collect()
    }
    let periods = axis_usize(r.period);
    let phases = axis_f64(r.phase);
    let powers = axis_u32(r.power);
    let mut out = Vec::with_capacity(periods.len() * phases.len() * powers.len());
    for &p in &periods {
        for &ph in &phases {
            for &po in &powers {
                out.push(JmaParams {
                    period: Some(p),
                    phase: Some(ph),
                    power: Some(po),
                });
            }
        }
    }
    out
}

#[inline(always)]
pub fn jma_batch_slice(
    data: &[f64],
    sweep: &JmaBatchRange,
    kern: Kernel,
) -> Result<JmaBatchOutput, JmaError> {
    jma_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn jma_batch_par_slice(
    data: &[f64],
    sweep: &JmaBatchRange,
    kern: Kernel,
) -> Result<JmaBatchOutput, JmaError> {
    jma_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn jma_batch_inner(
    data: &[f64],
    sweep: &JmaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<JmaBatchOutput, JmaError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(JmaError::InvalidPeriod { period: 0, data_len: 0 });
    }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(JmaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(JmaError::NotEnoughValidData { needed: max_p, valid: data.len() - first });
    }
    let rows = combos.len();
    let cols = data.len();
    let warm: Vec<usize> =
        combos.iter().map(|c| first + c.period.unwrap()).collect();

    let mut raw = make_uninit_matrix(rows, cols);
    unsafe { init_matrix_prefixes(&mut raw, cols, &warm) };

    // ---------- 2. closure that fills ONE row ---------------------------
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let prm     = &combos[row];
        let period  = prm.period.unwrap();
        let phase   = prm.phase.unwrap();
        let power   = prm.power.unwrap();

        // Cast the uninit slice to &mut [f64] for the row writers
        let out_row = core::slice::from_raw_parts_mut(
            dst_mu.as_mut_ptr() as *mut f64,
            dst_mu.len(),
        );

        match kern {
            Kernel::Scalar => jma_row_scalar (data, first, period, phase, power, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2   => jma_row_avx2   (data, first, period, phase, power, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => jma_row_avx512 (data, first, period, phase, power, out_row),
            _ => unreachable!(),
        }
    };

    // ---------- 3. run every row ----------------------------------------
    if parallel {

        #[cfg(not(target_arch = "wasm32"))] {

        raw.par_chunks_mut(cols)

                .enumerate()

                .for_each(|(row, slice)| do_row(row, slice));

        }

        #[cfg(target_arch = "wasm32")] {

        for (row, slice) in raw.chunks_mut(cols).enumerate() {

                    do_row(row, slice);

        }
        }
    } else {
        for (row, slice) in raw.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    // ---------- 4. transmute into fully-initialised Vec<f64> -------------
    let values: Vec<f64> = unsafe { std::mem::transmute(raw) };

    Ok(JmaBatchOutput { values, combos, rows, cols })
}

#[inline(always)]
fn jma_batch_inner_into(
    data: &[f64],
    sweep: &JmaBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<(Vec<JmaParams>, usize, usize), JmaError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(JmaError::InvalidPeriod { period: 0, data_len: 0 });
    }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(JmaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(JmaError::NotEnoughValidData { needed: max_p, valid: data.len() - first });
    }
    let rows = combos.len();
    let cols = data.len();
    
    // Ensure output buffer is the correct size
    if out.len() != rows * cols {
        return Err(JmaError::InvalidOutputBuffer {
            expected: rows * cols,
            actual: out.len(),
        });
    }
    
    let warm: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap()).collect();

    // Cast output to MaybeUninit for initialization
    let out_uninit = unsafe {
        std::slice::from_raw_parts_mut(
            out.as_mut_ptr() as *mut MaybeUninit<f64>,
            out.len()
        )
    };
    unsafe { init_matrix_prefixes(out_uninit, cols, &warm) };

    // ---------- closure that fills ONE row ---------------------------
    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let prm = &combos[row];
        let period = prm.period.unwrap();
        let phase = prm.phase.unwrap();
        let power = prm.power.unwrap();

        match kern {
            Kernel::Scalar => jma_row_scalar(data, first, period, phase, power, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => jma_row_avx2(data, first, period, phase, power, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => jma_row_avx512(data, first, period, phase, power, out_row),
            _ => unreachable!(),
        }
    };

    // ---------- run every row ----------------------------------------
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

    Ok((combos, rows, cols))
}

#[inline(always)]
unsafe fn jma_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    phase: f64,
    power: u32,
    out: &mut [f64],
) {
    jma_scalar(data, period, phase, power, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn jma_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    phase: f64,
    power: u32,
    out: &mut [f64],
) {
    jma_avx2(data, period, phase, power, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn jma_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    phase: f64,
    power: u32,
    out: &mut [f64],
) {
    jma_avx512(data, period, phase, power, first, out);
}

#[inline(always)]
pub fn expand_grid_jma(r: &JmaBatchRange) -> Vec<JmaParams> {
    expand_grid(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_jma_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = JmaParams { period: None, phase: None, power: None };
        let input = JmaInput::from_candles(&candles, "close", default_params);
        let output = jma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_jma_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = JmaInput::from_candles(&candles, "close", JmaParams::default());
        let result = jma_with_kernel(&input, kernel)?;
        let expected_last_five = [
            59305.04794668568,
            59261.270455005455,
            59156.791263606865,
            59128.30656791065,
            58918.89223153998,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(diff < 1e-6, "[{}] JMA {:?} mismatch at idx {}: got {}, expected {}", test_name, kernel, i, val, expected_last_five[i]);
        }
        Ok(())
    }

    fn check_jma_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = JmaInput::with_default_candles(&candles);
        match input.data {
            JmaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected JmaData::Candles"),
        }
        let output = jma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_jma_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = JmaParams { period: Some(0), phase: None, power: None };
        let input = JmaInput::from_slice(&input_data, params);
        let res = jma_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] JMA should fail with zero period", test_name);
        Ok(())
    }

    fn check_jma_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = JmaParams { period: Some(10), phase: None, power: None };
        let input = JmaInput::from_slice(&data_small, params);
        let res = jma_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] JMA should fail with period exceeding length", test_name);
        Ok(())
    }

    fn check_jma_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = JmaParams { period: Some(7), phase: None, power: None };
        let input = JmaInput::from_slice(&single_point, params);
        let res = jma_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] JMA should fail with insufficient data", test_name);
        Ok(())
    }

    fn check_jma_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = JmaParams { period: Some(7), phase: None, power: None };
        let first_input = JmaInput::from_candles(&candles, "close", first_params);
        let first_result = jma_with_kernel(&first_input, kernel)?;
        let second_params = JmaParams { period: Some(7), phase: None, power: None };
        let second_input = JmaInput::from_slice(&first_result.values, second_params);
        let second_result = jma_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    fn check_jma_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = JmaInput::from_candles(&candles, "close", JmaParams { period: Some(7), phase: None, power: None });
        let res = jma_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 240 {
            for (i, &val) in res.values[240..].iter().enumerate() {
                assert!(!val.is_nan(), "[{}] Found unexpected NaN at out-index {}", test_name, 240 + i);
            }
        }
        Ok(())
    }

    fn check_jma_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 7;
        let phase = 50.0;
        let power = 2;
        let input = JmaInput::from_candles(&candles, "close", JmaParams { period: Some(period), phase: Some(phase), power: Some(power) });
        let batch_output = jma_with_kernel(&input, kernel)?.values;
        let mut stream = JmaStream::try_new(JmaParams { period: Some(period), phase: Some(phase), power: Some(power) })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(jma_val) => stream_values.push(jma_val),
                None => stream_values.push(f64::NAN),
            }
        }
        assert_eq!(batch_output.len(), stream_values.len());
        for (i, (&b, &s)) in batch_output.iter().zip(stream_values.iter()).enumerate() {
            if b.is_nan() && s.is_nan() { continue; }
            let diff = (b - s).abs();
            assert!(diff < 1e-8, "[{}] JMA streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}", test_name, i, b, s, diff);
        }
        Ok(())
    }

    macro_rules! generate_all_jma_tests {
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

    generate_all_jma_tests!(
        check_jma_partial_params,
        check_jma_accuracy,
        check_jma_default_candles,
        check_jma_zero_period,
        check_jma_period_exceeds_length,
        check_jma_very_small_dataset,
        check_jma_reinput,
        check_jma_nan_handling,
        check_jma_streaming
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = JmaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = JmaParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            59305.04794668568,
            59261.270455005455,
            59156.791263606865,
            59128.30656791065,
            58918.89223153998,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-6,
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

// Python bindings
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(feature = "python")]
use numpy::{PyArray1, PyReadonlyArray1, IntoPyArray};

#[cfg(feature = "python")]
#[pyfunction(name = "jma")]
#[pyo3(signature = (arr_in, period, phase=50.0, power=2))]
pub fn jma_py<'py>(
    py: Python<'py>,
    arr_in: PyReadonlyArray1<'py, f64>,
    period: usize,
    phase: f64,
    power: u32,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    use numpy::PyArrayMethods;
    
    let slice_in = arr_in.as_slice()?; // zero-copy, read-only view
    
    // Pre-allocate NumPy output buffer
    let out_arr = unsafe { PyArray1::<f64>::new(py, [slice_in.len()], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? }; // safe: we own the array
    
    // Prepare JMA input
    let jma_in = JmaInput::from_slice(slice_in, JmaParams {
        period: Some(period),
        phase: Some(phase),
        power: Some(power),
    });
    
    // Heavy lifting without the GIL
    py.allow_threads(|| -> Result<(), JmaError> {
        jma_into(&jma_in, slice_out)
    })
    .map_err(|e| PyValueError::new_err(format!("JMA error: {}", e)))?;
    
    Ok(out_arr)
}

#[cfg(feature = "python")]
use ndarray::{Array2, Array1};

#[cfg(feature = "python")]
#[pyfunction(name = "jma_batch")]
#[pyo3(signature = (arr_in, period_range, phase_range=(50.0, 50.0, 0.0), power_range=(2, 2, 0)))]
pub fn jma_batch_py<'py>(
    py: Python<'py>,
    arr_in: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    phase_range: (f64, f64, f64),
    power_range: (u32, u32, u32),
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{PyArray2, PyArrayMethods};
    
    let slice_in = arr_in.as_slice()?; // zero-copy, read-only view
    let sweep = JmaBatchRange {
        period: period_range,
        phase: phase_range,
        power: power_range,
    };
    
    // Expand grid to get all combinations
    let combos = expand_grid(&sweep);
    if combos.is_empty() {
        return Err(PyValueError::new_err("Invalid parameter ranges"));
    }
    
    let rows = combos.len();
    let cols = slice_in.len();
    
    // Pre-allocate NumPy array (1-D, will reshape later)
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? }; // safe: we own the array
    
    // Heavy work without the GIL
    let (_, final_rows, final_cols) = py.allow_threads(|| -> Result<(Vec<JmaParams>, usize, usize), JmaError> {
        // Detect best kernel
        let kernel = match detect_best_batch_kernel() {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            _ => Kernel::Scalar,
        };
        
        // Use the new _into function with parallel=true
        jma_batch_inner_into(slice_in, &sweep, kernel, true, slice_out)
    })
    .map_err(|e| PyValueError::new_err(format!("JMA batch error: {}", e)))?;
    
    // Extract metadata and convert to NumPy arrays for zero-copy
    let periods = combos
        .iter()
        .map(|c| c.period.unwrap())
        .collect::<Vec<_>>()
        .into_pyarray(py);
    let phases = combos
        .iter()
        .map(|c| c.phase.unwrap())
        .collect::<Vec<_>>()
        .into_pyarray(py);
    let powers = combos
        .iter()
        .map(|c| c.power.unwrap())
        .collect::<Vec<_>>()
        .into_pyarray(py);
    
    // Reshape to 2D
    let out_2d = out_arr.reshape((final_rows, final_cols))?;
    
    // Create dictionary output
    let dict = PyDict::new(py);
    dict.set_item("values", out_2d)?;
    dict.set_item("periods", periods)?;
    dict.set_item("phases", phases)?;
    dict.set_item("powers", powers)?;
    
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "JmaStream")]
pub struct JmaStreamPy {
    inner: JmaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl JmaStreamPy {
    #[new]
    #[pyo3(signature = (period, phase=50.0, power=2))]
    fn new(period: usize, phase: f64, power: u32) -> PyResult<Self> {
        let params = JmaParams {
            period: Some(period),
            phase: Some(phase),
            power: Some(power),
        };
        
        let stream = JmaStream::try_new(params)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        
        Ok(Self { inner: stream })
    }
    
    fn update(&mut self, value: f64) -> Option<f64> {
        self.inner.update(value)
    }
}

// WASM bindings
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn jma_js(data: &[f64], period: usize, phase: f64, power: u32) -> Result<Vec<f64>, JsValue> {
    let params = JmaParams {
        period: Some(period),
        phase: Some(phase),
        power: Some(power),
    };
    
    let input = JmaInput::from_slice(data, params);
    
    match jma_with_kernel(&input, Kernel::Scalar) {
        Ok(output) => Ok(output.values),
        Err(e) => Err(JsValue::from_str(&e.to_string())),
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn jma_batch_js(
    data: &[f64],
    period_start: usize,
    period_end: usize,
    period_step: usize,
    phase_start: f64,
    phase_end: f64,
    phase_step: f64,
    power_start: u32,
    power_end: u32,
    power_step: u32,
) -> Result<Vec<f64>, JsValue> {
    let sweep = JmaBatchRange {
        period: (period_start, period_end, period_step),
        phase: (phase_start, phase_end, phase_step),
        power: (power_start, power_end, power_step),
    };

    // Use the existing batch function with parallel=false for WASM
    jma_batch_inner(data, &sweep, Kernel::Scalar, false)
        .map(|output| output.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn jma_batch_metadata_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
    phase_start: f64,
    phase_end: f64,
    phase_step: f64,
    power_start: u32,
    power_end: u32,
    power_step: u32,
) -> Vec<f64> {
    let mut metadata = Vec::new();
    
    let mut current_period = period_start;
    while current_period <= period_end {
        let mut current_phase = phase_start;
        while current_phase <= phase_end || (phase_step == 0.0 && current_phase == phase_start) {
            let mut current_power = power_start;
            while current_power <= power_end || (power_step == 0 && current_power == power_start) {
                metadata.push(current_period as f64);
                metadata.push(current_phase);
                metadata.push(current_power as f64);
                
                if power_step == 0 { break; }
                current_power += power_step;
            }
            if phase_step == 0.0 { break; }
            current_phase += phase_step;
        }
        if period_step == 0 { break; }
        current_period += period_step;
    }
    
    metadata
}
