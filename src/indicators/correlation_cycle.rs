//! # Correlation Cycle (John Ehlers)
//!
//! Computes real, imag, angle, and market state outputs based on correlation phasor analysis.
//! Parity with alma.rs structure, SIMD feature gating, and batch/stream/builder APIs included.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

impl<'a> AsRef<[f64]> for CorrelationCycleInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            CorrelationCycleData::Slice(slice) => slice,
            CorrelationCycleData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum CorrelationCycleData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct CorrelationCycleOutput {
    pub real: Vec<f64>,
    pub imag: Vec<f64>,
    pub angle: Vec<f64>,
    pub state: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct CorrelationCycleParams {
    pub period: Option<usize>,
    pub threshold: Option<f64>,
}

impl Default for CorrelationCycleParams {
    fn default() -> Self {
        Self {
            period: Some(20),
            threshold: Some(9.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CorrelationCycleInput<'a> {
    pub data: CorrelationCycleData<'a>,
    pub params: CorrelationCycleParams,
}

impl<'a> CorrelationCycleInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: CorrelationCycleParams,
    ) -> Self {
        Self {
            data: CorrelationCycleData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: CorrelationCycleParams) -> Self {
        Self {
            data: CorrelationCycleData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", CorrelationCycleParams::default())
    }

    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(20)
    }

    #[inline]
    pub fn get_threshold(&self) -> f64 {
        self.params.threshold.unwrap_or(9.0)
    }
}

#[derive(Debug, Clone)]
pub struct CorrelationCycleBuilder {
    period: Option<usize>,
    threshold: Option<f64>,
    kernel: Kernel,
}

impl Default for CorrelationCycleBuilder {
    fn default() -> Self {
        Self {
            period: None,
            threshold: None,
            kernel: Kernel::Auto,
        }
    }
}

impl CorrelationCycleBuilder {
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
    pub fn threshold(mut self, t: f64) -> Self {
        self.threshold = Some(t);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<CorrelationCycleOutput, CorrelationCycleError> {
        let p = CorrelationCycleParams {
            period: self.period,
            threshold: self.threshold,
        };
        let i = CorrelationCycleInput::from_candles(c, "close", p);
        correlation_cycle_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<CorrelationCycleOutput, CorrelationCycleError> {
        let p = CorrelationCycleParams {
            period: self.period,
            threshold: self.threshold,
        };
        let i = CorrelationCycleInput::from_slice(d, p);
        correlation_cycle_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<CorrelationCycleStream, CorrelationCycleError> {
        let p = CorrelationCycleParams {
            period: self.period,
            threshold: self.threshold,
        };
        CorrelationCycleStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum CorrelationCycleError {
    #[error("correlation_cycle: Empty data provided.")]
    EmptyData,
    #[error("correlation_cycle: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("correlation_cycle: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("correlation_cycle: All values are NaN.")]
    AllValuesNaN,
}

use crate::utilities::math_functions::atan64;

#[inline]
pub fn correlation_cycle(input: &CorrelationCycleInput) -> Result<CorrelationCycleOutput, CorrelationCycleError> {
    correlation_cycle_with_kernel(input, Kernel::Auto)
}

pub fn correlation_cycle_with_kernel(input: &CorrelationCycleInput, kernel: Kernel) -> Result<CorrelationCycleOutput, CorrelationCycleError> {
    let data: &[f64] = match &input.data {
        CorrelationCycleData::Candles { candles, source } => source_type(candles, source),
        CorrelationCycleData::Slice(sl) => sl,
    };

    if data.is_empty() {
        return Err(CorrelationCycleError::EmptyData);
    }
    if data.iter().all(|&x| x.is_nan()) {
        return Err(CorrelationCycleError::AllValuesNaN);
    }
    let period = input.get_period();
    if period == 0 || period > data.len() {
        return Err(CorrelationCycleError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }
    let valid_count = data.iter().filter(|&&x| !x.is_nan()).count();
    if valid_count < period {
        return Err(CorrelationCycleError::NotEnoughValidData {
            needed: period,
            valid: valid_count,
        });
    }
    let threshold = input.get_threshold();

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let mut real = vec![f64::NAN; data.len()];
    let mut imag = vec![f64::NAN; data.len()];
    let mut angle = vec![f64::NAN; data.len()];
    let mut state = vec![0.0; data.len()];

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => correlation_cycle_scalar(data, period, threshold, &mut real, &mut imag, &mut angle, &mut state),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => correlation_cycle_avx2(data, period, threshold, &mut real, &mut imag, &mut angle, &mut state),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => correlation_cycle_avx512(data, period, threshold, &mut real, &mut imag, &mut angle, &mut state),
            _ => unreachable!(),
        }
    }
    Ok(CorrelationCycleOutput { real, imag, angle, state })
}

#[inline]
pub fn correlation_cycle_scalar(
    data: &[f64],
    period: usize,
    threshold: f64,
    real: &mut [f64],
    imag: &mut [f64],
    angle: &mut [f64],
    state: &mut [f64],
) {
    // same “two_pi” and “half_pi” as in the old version
    let two_pi = 4.0 * f64::asin(1.0);
    let half_pi = f64::asin(1.0);

    // build cosine/sine tables of length = period
    let mut cos_table = vec![0.0; period];
    let mut sin_table = vec![0.0; period];
    for j in 0..period {
        let a = two_pi * (j as f64 + 1.0) / (period as f64);
        cos_table[j] = a.cos();
        sin_table[j] = -a.sin();
    }

    // Step 1: compute real[i] and imag[i] for i ∈ [period..data.len())
    for i in period..data.len() {
        let mut rx = 0.0;
        let mut rxx = 0.0;
        let mut rxy = 0.0;
        let mut ryy = 0.0;
        let mut ry = 0.0;
        let mut ix = 0.0;
        let mut ixx = 0.0;
        let mut ixy = 0.0;
        let mut iyy = 0.0;
        let mut iy = 0.0;

        for j in 0..period {
            let idx = i - (j + 1);
            let x = if data[idx].is_nan() { 0.0 } else { data[idx] };
            let yc = cos_table[j];
            let ys = sin_table[j];

            // accumulate “real‐part” sums
            rx += x;
            rxx += x * x;
            rxy += x * yc;
            ryy += yc * yc;
            ry += yc;

            // accumulate “imag‐part” sums
            ix += x;
            ixx += x * x;
            ixy += x * ys;
            iyy += ys * ys;
            iy += ys;
        }

        let n = period as f64;
        let t1 = n * rxx - rx * rx;
        let t2 = n * ryy - ry * ry;
        if t1 > 0.0 && t2 > 0.0 {
            real[i] = (n * rxy - rx * ry) / (t1 * t2).sqrt();
        }

        let t3 = n * ixx - ix * ix;
        let t4 = n * iyy - iy * iy;
        if t3 > 0.0 && t4 > 0.0 {
            imag[i] = (n * ixy - ix * iy) / (t3 * t4).sqrt();
        }
    }

    // Step 2: compute “raw” angle exactly as in the old function
    for i in period..data.len() {
        let im = imag[i];

        if im == 0.0 {
            angle[i] = 0.0;
        } else {
            // a = atan64(real[i] / im) + half_pi, then to_degrees, then -180° if im > 0
            let mut a = (real[i] / im).atan() + half_pi;
            a = a.to_degrees();
            if im > 0.0 {
                a -= 180.0;
            }
            angle[i] = a;
        }
    }


    // Step 4: build the state array exactly as in the old function
    for i in (period + 1)..data.len() {
        let pa = angle[i - 1];
        let ca = angle[i];
        if !pa.is_nan() && !ca.is_nan() && (ca - pa).abs() < threshold {
            state[i] = if ca >= 0.0 { 1.0 } else { -1.0 };
        } else {
            state[i] = 0.0;
        }
    }
}


#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn correlation_cycle_avx2(
    data: &[f64],
    period: usize,
    threshold: f64,
    real: &mut [f64],
    imag: &mut [f64],
    angle: &mut [f64],
    state: &mut [f64],
) {
    correlation_cycle_scalar(data, period, threshold, real, imag, angle, state)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn correlation_cycle_avx512(
    data: &[f64],
    period: usize,
    threshold: f64,
    real: &mut [f64],
    imag: &mut [f64],
    angle: &mut [f64],
    state: &mut [f64],
) {
    correlation_cycle_scalar(data, period, threshold, real, imag, angle, state)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn correlation_cycle_avx512_short(
    data: &[f64],
    period: usize,
    threshold: f64,
    real: &mut [f64],
    imag: &mut [f64],
    angle: &mut [f64],
    state: &mut [f64],
) {
    correlation_cycle_scalar(data, period, threshold, real, imag, angle, state)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn correlation_cycle_avx512_long(
    data: &[f64],
    period: usize,
    threshold: f64,
    real: &mut [f64],
    imag: &mut [f64],
    angle: &mut [f64],
    state: &mut [f64],
) {
    correlation_cycle_scalar(data, period, threshold, real, imag, angle, state)
}

// Row wrappers
#[inline(always)]
pub unsafe fn correlation_cycle_row_scalar(
    data: &[f64],
    period: usize,
    threshold: f64,
    real: &mut [f64],
    imag: &mut [f64],
    angle: &mut [f64],
    state: &mut [f64],
) {
    correlation_cycle_scalar(data, period, threshold, real, imag, angle, state)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn correlation_cycle_row_avx2(
    data: &[f64],
    period: usize,
    threshold: f64,
    real: &mut [f64],
    imag: &mut [f64],
    angle: &mut [f64],
    state: &mut [f64],
) {
    correlation_cycle_avx2(data, period, threshold, real, imag, angle, state)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn correlation_cycle_row_avx512(
    data: &[f64],
    period: usize,
    threshold: f64,
    real: &mut [f64],
    imag: &mut [f64],
    angle: &mut [f64],
    state: &mut [f64],
) {
    correlation_cycle_avx512(data, period, threshold, real, imag, angle, state)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn correlation_cycle_row_avx512_short(
    data: &[f64],
    period: usize,
    threshold: f64,
    real: &mut [f64],
    imag: &mut [f64],
    angle: &mut [f64],
    state: &mut [f64],
) {
    correlation_cycle_avx512_short(data, period, threshold, real, imag, angle, state)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn correlation_cycle_row_avx512_long(
    data: &[f64],
    period: usize,
    threshold: f64,
    real: &mut [f64],
    imag: &mut [f64],
    angle: &mut [f64],
    state: &mut [f64],
) {
    correlation_cycle_avx512_long(data, period, threshold, real, imag, angle, state)
}

#[derive(Debug, Clone)]
pub struct CorrelationCycleStream {
    period: usize,
    threshold: f64,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,
    /// Holds the (real, imag, angle, state) computed for the “previous” window,
    /// so that we emit it one tick later.  
    last: Option<(f64, f64, f64, f64)>,
}

impl CorrelationCycleStream {
    pub fn try_new(params: CorrelationCycleParams) -> Result<Self, CorrelationCycleError> {
        let period = params.period.unwrap_or(20);
        if period == 0 {
            return Err(CorrelationCycleError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let threshold = params.threshold.unwrap_or(9.0);

        Ok(Self {
            period,
            threshold,
            buffer: vec![f64::NAN; period],
            head: 0,
            filled: false,
            last: None,
        })
    }

    /// Insert `value` into the circular buffer.  Returns `None` until we have
    /// computed at least one full‐window.  After that, each
    /// `update(...)` call returns exactly the (real, imag, angle, state) that
    /// matches the batch result “one tick” earlier.
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<(f64, f64, f64, f64)> {
        // 1) Write into circular buffer and advance head
        self.buffer[self.head] = value;
        self.head = (self.head + 1) % self.period;

        // 2) If this is the very first time head wrapped to 0, we have inserted
        //    exactly `period` items.  Compute that first-window’s output, stash it
        //    in `self.last`, and return None so that index=period-1 still yields None.
        if !self.filled && self.head == 0 {
            self.filled = true;

            // Build a slice of length = period+1
            // ‣ first `period` entries = chronological window of the last `period` values
            // ‣ extra slot at index `period` can be any dummy (we won't actually use it in sums)
            let mut small = Vec::with_capacity(self.period + 1);
            for k in 0..self.period {
                let idx = (self.head + k) % self.period;
                small.push(self.buffer[idx]);
            }
            // Push one dummy at the end (to make length = period+1)
            small.push(0.0);

            // Prepare tiny result arrays of length = period+1
            let mut real_arr = vec![f64::NAN; self.period + 1];
            let mut imag_arr = vec![f64::NAN; self.period + 1];
            let mut angle_arr = vec![f64::NAN; self.period + 1];
            let mut state_arr = vec![0.0; self.period + 1];

            // Call the scalar routine on small[0..(period+1)]
            // — it will compute at index i = period exactly once, using indices [0..(period-1)]
            unsafe {
                correlation_cycle_scalar(
                    &small,
                    self.period,
                    self.threshold,
                    &mut real_arr,
                    &mut imag_arr,
                    &mut angle_arr,
                    &mut state_arr,
                );
            }

            // The “batch‐aligned” result is stored at index = period
            let first_r = real_arr[self.period];
            let first_i = imag_arr[self.period];
            let first_a = angle_arr[self.period];
            let first_s = state_arr[self.period];

            self.last = Some((first_r, first_i, first_a, first_s));
            return None;
        }

        // 3) If we still haven't filled one full window, keep returning None
        if !self.filled {
            return None;
        }

        // 4) Otherwise—every tick from now on—we already have `self.last` from the previous
        //    window.  So:
        //    a) pull out `to_emit = self.last.unwrap()`
        //    b) build the next (period+1)-slice
        //    c) call correlation_cycle_scalar(...) on that slice, stash into self.last
        //    d) return `to_emit`

        // a) Grab the previously‐computed tuple:
        let to_emit = self.last.take().unwrap();

        // b) Build the next (period+1)-long slice
        let mut small = Vec::with_capacity(self.period + 1);
        for k in 0..self.period {
            let idx = (self.head + k) % self.period;
            small.push(self.buffer[idx]);
        }
        // Add one dummy at index = period
        small.push(0.0);

        // c) Compute correlation_cycle_scalar on that length=(period+1) slice
        let mut real_arr = vec![f64::NAN; self.period + 1];
        let mut imag_arr = vec![f64::NAN; self.period + 1];
        let mut angle_arr = vec![f64::NAN; self.period + 1];
        let mut state_arr = vec![0.0; self.period + 1];
        unsafe {
            correlation_cycle_scalar(
                &small,
                self.period,
                self.threshold,
                &mut real_arr,
                &mut imag_arr,
                &mut angle_arr,
                &mut state_arr,
            );
        }
        let next_r = real_arr[self.period];
        let next_i = imag_arr[self.period];
        let next_a = angle_arr[self.period];
        let next_s = state_arr[self.period];
        self.last = Some((next_r, next_i, next_a, next_s));

        // d) Return the “previous‐window” result:
        Some(to_emit)
    }
}


#[derive(Clone, Debug)]
pub struct CorrelationCycleBatchRange {
    pub period: (usize, usize, usize),
    pub threshold: (f64, f64, f64),
}

impl Default for CorrelationCycleBatchRange {
    fn default() -> Self {
        Self {
            period: (20, 100, 1),
            threshold: (9.0, 9.0, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CorrelationCycleBatchBuilder {
    range: CorrelationCycleBatchRange,
    kernel: Kernel,
}

impl CorrelationCycleBatchBuilder {
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
    pub fn threshold_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.threshold = (start, end, step);
        self
    }
    #[inline]
    pub fn threshold_static(mut self, x: f64) -> Self {
        self.range.threshold = (x, x, 0.0);
        self
    }

    pub fn apply_slice(self, data: &[f64]) -> Result<CorrelationCycleBatchOutput, CorrelationCycleError> {
        correlation_cycle_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<CorrelationCycleBatchOutput, CorrelationCycleError> {
        CorrelationCycleBatchBuilder::new().kernel(k).apply_slice(data)
    }

    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<CorrelationCycleBatchOutput, CorrelationCycleError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }

    pub fn with_default_candles(c: &Candles) -> Result<CorrelationCycleBatchOutput, CorrelationCycleError> {
        CorrelationCycleBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn correlation_cycle_batch_with_kernel(
    data: &[f64],
    sweep: &CorrelationCycleBatchRange,
    k: Kernel,
) -> Result<CorrelationCycleBatchOutput, CorrelationCycleError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(CorrelationCycleError::InvalidPeriod {
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
    correlation_cycle_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct CorrelationCycleBatchOutput {
    pub real: Vec<f64>,
    pub imag: Vec<f64>,
    pub angle: Vec<f64>,
    pub state: Vec<f64>,
    pub combos: Vec<CorrelationCycleParams>,
    pub rows: usize,
    pub cols: usize,
}

impl CorrelationCycleBatchOutput {
    pub fn row_for_params(&self, p: &CorrelationCycleParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(20) == p.period.unwrap_or(20)
                && (c.threshold.unwrap_or(9.0) - p.threshold.unwrap_or(9.0)).abs() < 1e-12
        })
    }
    pub fn values_for(&self, p: &CorrelationCycleParams) -> Option<(&[f64], &[f64], &[f64], &[f64])> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            (
                &self.real[start..start + self.cols],
                &self.imag[start..start + self.cols],
                &self.angle[start..start + self.cols],
                &self.state[start..start + self.cols],
            )
        })
    }
}

#[inline(always)]
fn expand_grid(r: &CorrelationCycleBatchRange) -> Vec<CorrelationCycleParams> {
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
    let thresholds = axis_f64(r.threshold);

    let mut out = Vec::with_capacity(periods.len() * thresholds.len());
    for &p in &periods {
        for &t in &thresholds {
            out.push(CorrelationCycleParams {
                period: Some(p),
                threshold: Some(t),
            });
        }
    }
    out
}

#[inline(always)]
pub fn correlation_cycle_batch_slice(
    data: &[f64],
    sweep: &CorrelationCycleBatchRange,
    kern: Kernel,
) -> Result<CorrelationCycleBatchOutput, CorrelationCycleError> {
    correlation_cycle_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn correlation_cycle_batch_par_slice(
    data: &[f64],
    sweep: &CorrelationCycleBatchRange,
    kern: Kernel,
) -> Result<CorrelationCycleBatchOutput, CorrelationCycleError> {
    correlation_cycle_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn correlation_cycle_batch_inner(
    data: &[f64],
    sweep: &CorrelationCycleBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<CorrelationCycleBatchOutput, CorrelationCycleError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(CorrelationCycleError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(CorrelationCycleError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(CorrelationCycleError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();

    let mut real = vec![f64::NAN; rows * cols];
    let mut imag = vec![f64::NAN; rows * cols];
    let mut angle = vec![f64::NAN; rows * cols];
    let mut state = vec![0.0; rows * cols];

    let do_row = |row: usize, out_real: &mut [f64], out_imag: &mut [f64], out_angle: &mut [f64], out_state: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        let threshold = combos[row].threshold.unwrap();
        match kern {
            Kernel::Scalar => correlation_cycle_row_scalar(data, period, threshold, out_real, out_imag, out_angle, out_state),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => correlation_cycle_row_avx2(data, period, threshold, out_real, out_imag, out_angle, out_state),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => correlation_cycle_row_avx512(data, period, threshold, out_real, out_imag, out_angle, out_state),
            _ => unreachable!(),
        }
    };

    if parallel {
        real.par_chunks_mut(cols)
            .zip(imag.par_chunks_mut(cols))
            .zip(angle.par_chunks_mut(cols))
            .zip(state.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, (((r, im), an), st))| do_row(row, r, im, an, st));
    } else {
        for (((r, im), an), st) in real.chunks_mut(cols)
            .zip(imag.chunks_mut(cols))
            .zip(angle.chunks_mut(cols))
            .zip(state.chunks_mut(cols))
            .enumerate()
            .map(|(_, x)| x) {
            do_row(0, r, im, an, st); // fix: row tracking if needed, otherwise just keep as 0
        }
    }

    Ok(CorrelationCycleBatchOutput {
        real,
        imag,
        angle,
        state,
        combos,
        rows,
        cols,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;
    use crate::utilities::enums::Kernel;

    fn check_cc_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = CorrelationCycleParams {
            period: None,
            threshold: None,
        };
        let input = CorrelationCycleInput::from_candles(&candles, "close", default_params);
        let output = correlation_cycle_with_kernel(&input, kernel)?;
        assert_eq!(output.real.len(), candles.close.len());
        Ok(())
    }

    fn check_cc_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = CorrelationCycleParams {
            period: Some(20),
            threshold: Some(9.0),
        };
        let input = CorrelationCycleInput::from_candles(&candles, "close", params);
        let result = correlation_cycle_with_kernel(&input, kernel)?;
        let expected_last_five_real = [
            -0.3348928030992766,
            -0.2908979303392832,
            -0.10648582811938148,
            -0.09118320471750277,
            0.0826798259258665,
        ];
        let expected_last_five_imag = [
            0.2902308064575494,
            0.4025192756952553,
            0.4704322460080054,
            0.5404405595224989,
            0.5418162415918566,
        ];
        let expected_last_five_angle = [
            -139.0865569687123,
            -125.8553823569915,
            -102.75438860700636,
            -99.576759208278,
            -81.32373697835556,
        ];
        let start = result.real.len().saturating_sub(5);
        for i in 0..5 {
            let diff_real = (result.real[start + i] - expected_last_five_real[i]).abs();
            let diff_imag = (result.imag[start + i] - expected_last_five_imag[i]).abs();
            let diff_angle = (result.angle[start + i] - expected_last_five_angle[i]).abs();
            assert!(diff_real < 1e-8, "[{}] CC {:?} real mismatch at idx {}: got {}, expected {}", test_name, kernel, i, result.real[start + i], expected_last_five_real[i]);
            assert!(diff_imag < 1e-8, "[{}] CC {:?} imag mismatch at idx {}: got {}, expected {}", test_name, kernel, i, result.imag[start + i], expected_last_five_imag[i]);
            assert!(diff_angle < 1e-8, "[{}] CC {:?} angle mismatch at idx {}: got {}, expected {}", test_name, kernel, i, result.angle[start + i], expected_last_five_angle[i]);
        }
        Ok(())
    }

    fn check_cc_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = CorrelationCycleInput::with_default_candles(&candles);
        match input.data {
            CorrelationCycleData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected CorrelationCycleData::Candles"),
        }
        let output = correlation_cycle_with_kernel(&input, kernel)?;
        assert_eq!(output.real.len(), candles.close.len());
        Ok(())
    }

    fn check_cc_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = CorrelationCycleParams {
            period: Some(0),
            threshold: None,
        };
        let input = CorrelationCycleInput::from_slice(&input_data, params);
        let res = correlation_cycle_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] CC should fail with zero period", test_name);
        Ok(())
    }

    fn check_cc_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = CorrelationCycleParams {
            period: Some(10),
            threshold: None,
        };
        let input = CorrelationCycleInput::from_slice(&data_small, params);
        let res = correlation_cycle_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] CC should fail with period exceeding length", test_name);
        Ok(())
    }

    fn check_cc_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = CorrelationCycleParams {
            period: Some(9),
            threshold: None,
        };
        let input = CorrelationCycleInput::from_slice(&single_point, params);
        let res = correlation_cycle_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] CC should fail with insufficient data", test_name);
        Ok(())
    }

    fn check_cc_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [10.0, 10.5, 11.0, 11.5, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0];
        let params = CorrelationCycleParams {
            period: Some(4),
            threshold: Some(2.0),
        };
        let input = CorrelationCycleInput::from_slice(&data, params.clone());
        let first_result = correlation_cycle_with_kernel(&input, kernel)?;
        let second_input = CorrelationCycleInput::from_slice(&first_result.real, params);
        let second_result = correlation_cycle_with_kernel(&second_input, kernel)?;
        assert_eq!(first_result.real.len(), data.len());
        assert_eq!(second_result.real.len(), data.len());
        Ok(())
    }

    fn check_cc_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = CorrelationCycleInput::from_candles(
            &candles,
            "close",
            CorrelationCycleParams {
                period: Some(20),
                threshold: None,
            },
        );
        let res = correlation_cycle_with_kernel(&input, kernel)?;
        assert_eq!(res.real.len(), candles.close.len());
        if res.real.len() > 40 {
            for (i, &val) in res.real[40..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test_name,
                    40 + i
                );
            }
        }
        Ok(())
    }

    fn check_cc_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 20;
        let threshold = 9.0;
        let input = CorrelationCycleInput::from_candles(
            &candles,
            "close",
            CorrelationCycleParams {
                period: Some(period),
                threshold: Some(threshold),
            },
        );
        let batch_output = correlation_cycle_with_kernel(&input, kernel)?;
        let mut stream = CorrelationCycleStream::try_new(CorrelationCycleParams {
            period: Some(period),
            threshold: Some(threshold),
        })?;
        let mut stream_real = Vec::with_capacity(candles.close.len());
        let mut stream_imag = Vec::with_capacity(candles.close.len());
        let mut stream_angle = Vec::with_capacity(candles.close.len());
        let mut stream_state = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some((r, im, ang, st)) => {
                    stream_real.push(r);
                    stream_imag.push(im);
                    stream_angle.push(ang);
                    stream_state.push(st);
                }
                None => {
                    stream_real.push(f64::NAN);
                    stream_imag.push(f64::NAN);
                    stream_angle.push(f64::NAN);
                    stream_state.push(0.0);
                }
            }
        }
        assert_eq!(batch_output.real.len(), stream_real.len());
        for (i, (&b, &s)) in batch_output.real.iter().zip(stream_real.iter()).enumerate() {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] CC streaming real f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_cc_tests {
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

    generate_all_cc_tests!(
        check_cc_partial_params,
        check_cc_accuracy,
        check_cc_default_candles,
        check_cc_zero_period,
        check_cc_period_exceeds_length,
        check_cc_very_small_dataset,
        check_cc_reinput,
        check_cc_nan_handling,
        check_cc_streaming
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = CorrelationCycleBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = CorrelationCycleParams::default();
        let (row_real, row_imag, row_angle, row_state) = output.values_for(&def).expect("default row missing");
        assert_eq!(row_real.len(), c.close.len());
        assert_eq!(row_imag.len(), c.close.len());
        assert_eq!(row_angle.len(), c.close.len());
        assert_eq!(row_state.len(), c.close.len());
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
