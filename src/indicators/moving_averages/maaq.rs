//! # Moving Average Adaptive Q (MAAQ)
//!
//! An adaptive moving average that adjusts smoothing based on the ratio of short-term noise
//! to long-term signal, with period, fast, and slow smoothing coefficients. Matches alma.rs API.
//!
//! ## Parameters
//! - **period**: Window size.
//! - **fast_period**: Smoothing coefficient (fast).
//! - **slow_period**: Smoothing coefficient (slow).
//!
//! ## Errors
//! - **AllValuesNaN**: All input values are `NaN`.
//! - **InvalidPeriod**: Any window is zero or period exceeds data length.
//! - **NotEnoughValidData**: Not enough data for the requested period.
//!
//! ## Returns
//! - **Ok(MaaqOutput)** on success, with output values.
//! - **Err(MaaqError)** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel, alloc_with_nan_prefix, init_matrix_prefixes, make_uninit_matrix};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;
use std::mem::MaybeUninit;

impl<'a> AsRef<[f64]> for MaaqInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            MaaqData::Slice(slice) => slice,
            MaaqData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum MaaqData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct MaaqOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct MaaqParams {
    pub period: Option<usize>,
    pub fast_period: Option<usize>,
    pub slow_period: Option<usize>,
}

impl Default for MaaqParams {
    fn default() -> Self {
        Self {
            period: Some(11),
            fast_period: Some(2),
            slow_period: Some(30),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MaaqInput<'a> {
    pub data: MaaqData<'a>,
    pub params: MaaqParams,
}

impl<'a> MaaqInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: MaaqParams) -> Self {
        Self {
            data: MaaqData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: MaaqParams) -> Self {
        Self {
            data: MaaqData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", MaaqParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(11)
    }
    #[inline]
    pub fn get_fast_period(&self) -> usize {
        self.params.fast_period.unwrap_or(2)
    }
    #[inline]
    pub fn get_slow_period(&self) -> usize {
        self.params.slow_period.unwrap_or(30)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct MaaqBuilder {
    period: Option<usize>,
    fast_period: Option<usize>,
    slow_period: Option<usize>,
    kernel: Kernel,
}

impl Default for MaaqBuilder {
    fn default() -> Self {
        Self {
            period: None,
            fast_period: None,
            slow_period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl MaaqBuilder {
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
    pub fn fast_period(mut self, n: usize) -> Self {
        self.fast_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn slow_period(mut self, n: usize) -> Self {
        self.slow_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<MaaqOutput, MaaqError> {
        let p = MaaqParams {
            period: self.period,
            fast_period: self.fast_period,
            slow_period: self.slow_period,
        };
        let i = MaaqInput::from_candles(c, "close", p);
        maaq_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<MaaqOutput, MaaqError> {
        let p = MaaqParams {
            period: self.period,
            fast_period: self.fast_period,
            slow_period: self.slow_period,
        };
        let i = MaaqInput::from_slice(d, p);
        maaq_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<MaaqStream, MaaqError> {
        let p = MaaqParams {
            period: self.period,
            fast_period: self.fast_period,
            slow_period: self.slow_period,
        };
        MaaqStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum MaaqError {
    #[error("maaq: All values are NaN.")]
    AllValuesNaN,
    #[error("maaq: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("maaq: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("maaq: periods cannot be zero: period = {period}, fast = {fast_p}, slow = {slow_p}")]
    ZeroPeriods {
        period: usize,
        fast_p: usize,
        slow_p: usize,
    },
}

#[inline]
pub fn maaq(input: &MaaqInput) -> Result<MaaqOutput, MaaqError> {
    maaq_with_kernel(input, Kernel::Auto)
}

pub fn maaq_with_kernel(input: &MaaqInput, kernel: Kernel) -> Result<MaaqOutput, MaaqError> {
    let data: &[f64] = match &input.data {
        MaaqData::Candles { candles, source } => source_type(candles, source),
        MaaqData::Slice(sl) => sl,
    };

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(MaaqError::AllValuesNaN)?;

    let len = data.len();
    let period = input.get_period();
    let fast_p = input.get_fast_period();
    let slow_p = input.get_slow_period();

    if period == 0 || fast_p == 0 || slow_p == 0 {
        return Err(MaaqError::ZeroPeriods {
            period,
            fast_p,
            slow_p,
        });
    }
    if period > len {
        return Err(MaaqError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(MaaqError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let warm  = first + period;
    let mut out = alloc_with_nan_prefix(len, warm);

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other        => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                // `out` is written in-place
                maaq_scalar(data, period, fast_p, slow_p, first, &mut out)?;
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                maaq_avx2(data, period, fast_p, slow_p, first, &mut out)?;
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                maaq_avx512(data, period, fast_p, slow_p, first, &mut out)?;
            }
            _ => unreachable!(),
        }
    }

    Ok(MaaqOutput { values: out })
}

#[inline]
pub fn maaq_scalar(
    data: &[f64],
    period: usize,
    fast_p: usize,
    slow_p: usize,
    first: usize,          // kept for API compatibility; still unused
    out: &mut [f64],
) -> Result<(), MaaqError> {
    let len      = data.len();
    let fast_sc  = 2.0 / (fast_p  as f64 + 1.0);
    let slow_sc  = 2.0 / (slow_p  as f64 + 1.0);

    // pre-compute absolute price differences
    let mut diff = vec![0.0; len];
    for i in 1..len {
        diff[i] = (data[i] - data[i - 1]).abs();
    }

    // warm-up: the first `period` outputs equal the raw prices
    out[..period].copy_from_slice(&data[..period]);

    // rolling sum of |Δprice|
    let mut rolling_sum = diff[..period].iter().sum::<f64>();

    for i in period..len {
        // slide the window
        rolling_sum += diff[i];
        rolling_sum -= diff[i - period];

        // efficiency ratio ER = |price[i] − price[i-period]| / Σ|Δprice|
        let noise   = rolling_sum;
        let signal  = (data[i] - data[i - period]).abs();
        let ratio   = if noise.abs() < f64::EPSILON { 0.0 } else { signal / noise };

        // smoothing constant SC  = (ratio * fast_sc + slow_sc)²   ← no mul_add
        let sc   = ratio * fast_sc + slow_sc;
        let temp = sc * sc;

        // adaptive EMA update
        let prev_val  = out[i - 1];
        out[i] = prev_val + temp * (data[i] - prev_val);
    }
    Ok(())
}


#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn maaq_avx2(
    data:  &[f64],
    period: usize,
    fast_p: usize,
    slow_p: usize,
    _first: usize,          // kept for API compatibility; still unused
    out:   &mut [f64],
) -> Result<(), MaaqError> {
    // ------ safety & basic checks ------------------------------------------------------------
    let len = data.len();
    assert_eq!(len, out.len(), "output slice length must match input");

    // ------ 1 · pre-compute constants --------------------------------------------------------
    let fast_sc = 2.0 / (fast_p as f64 + 1.0);
    let slow_sc = 2.0 / (slow_p as f64 + 1.0);

    // ------ 2 · rolling-window buffers -------------------------------------------------------
    // diff[0] starts as 0.0 so scalar & SIMD paths have identical initial sums
    let mut diffs    = vec![0.0f64; period];
    let mut vol_sum  = 0.0;

    // fill slots 1‥period-1 (|Δprice| between successive bars)
    for i in 1..period {
        let d = (data[i] - data[i - 1]).abs();
        diffs[i] = d;
        vol_sum += d;
    }

    // seed output with raw prices for the warm-up area
    out[..period].copy_from_slice(&data[..period]);
    let mut prev_val = data[period - 1];

    // ------ 3 · first computable point (index = period) -------------------------------------
    // ❶ insert newest |Δ| BEFORE computing ER₀ so window now covers period bars
    let new_diff = (data[period] - data[period - 1]).abs();
    diffs[0] = new_diff;
    vol_sum += new_diff;

    let er0 = if vol_sum > f64::EPSILON {
        (data[period] - data[0]).abs() / vol_sum
    
        } else {
        0.0
    };
    let mut sc = fast_sc.mul_add(er0, slow_sc); // (fast_sc * ER) + slow_sc
    sc *= sc;                                   // square once
    prev_val  = sc.mul_add(data[period] - prev_val, prev_val);
    out[period] = prev_val;

    let mut idx = 1;            // ring-buffer head: oldest diff is now at slot 1

    // ------ 4 · main streaming loop ----------------------------------------------------------
    for i in (period + 1)..len {
        // roll window: drop oldest |Δ|, add newest |Δ|
        vol_sum -= diffs[idx];
        let nd = (data[i] - data[i - 1]).abs();
        diffs[idx] = nd;
        vol_sum += nd;
        idx += 1;
        if idx == period { idx = 0; }

        // efficiency ratio
        let er = if vol_sum > f64::EPSILON {
            (data[i] - data[i - period]).abs() / vol_sum
        
            } else {
            0.0
        };

        // adaptive smoothing constant (squared)
        let mut sc = fast_sc.mul_add(er, slow_sc);
        sc *= sc;

        // EMA-style update using fused multiply-add
        prev_val = sc.mul_add(data[i] - prev_val, prev_val);
        out[i]   = prev_val;
    }

    Ok(())
}


#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn maaq_avx512(
    data: &[f64],
    period: usize,
    fast_p: usize,
    slow_p: usize,
    first: usize,
    out: &mut [f64],
) -> Result<(), MaaqError> {
    maaq_avx2(data, period, fast_p, slow_p, first, out)?;
    Ok(())
}

// Streaming/Stateful MaaqStream
#[derive(Debug, Clone)]
pub struct MaaqStream {
    period: usize,
    fast_period: usize,
    slow_period: usize,
    buffer: Vec<f64>,
    diff: Vec<f64>,
    head: usize,
    filled: bool,
    last: f64,
    count: usize,
}

impl MaaqStream {
    pub fn try_new(params: MaaqParams) -> Result<Self, MaaqError> {
        let period = params.period.unwrap_or(11);
        let fast_p = params.fast_period.unwrap_or(2);
        let slow_p = params.slow_period.unwrap_or(30);
        if period == 0 || fast_p == 0 || slow_p == 0 {
            return Err(MaaqError::ZeroPeriods {
                period,
                fast_p,
                slow_p,
            });
        }
        Ok(Self {
            period,
            fast_period: fast_p,
            slow_period: slow_p,
            buffer: vec![0.0; period],
            diff: vec![0.0; period],
            head: 0,
            filled: false,
            last: f64::NAN,
            count: 0,
        })
    }
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        let prev = if self.count > 0 {
            self.buffer[(self.head + self.period - 1) % self.period]
        
            } else {
            value
        };
        let old_value = self.buffer[self.head];
        let d = (value - prev).abs();
        self.buffer[self.head] = value;
        self.diff[self.head] = d;
        self.head = (self.head + 1) % self.period;
        self.count += 1;
        if !self.filled {
            self.last = value;
            if self.head == 0 {
                self.filled = true;
            }
            return Some(value);
        }
        let sum: f64 = self.diff.iter().sum();
        let noise = sum;
        let signal = (value - old_value).abs();
        let fast_sc = 2.0 / (self.fast_period as f64 + 1.0);
        let slow_sc = 2.0 / (self.slow_period as f64 + 1.0);
        let ratio = if noise.abs() < f64::EPSILON {
            0.0
        
            } else {
            signal / noise
        };
        let sc = ratio.mul_add(fast_sc, slow_sc);
        let temp = sc * sc;
        let out = self.last + temp * (value - self.last);
        self.last = out;
        Some(out)
    }
}

#[derive(Clone, Debug)]
pub struct MaaqBatchRange {
    pub period: (usize, usize, usize),
    pub fast_period: (usize, usize, usize),
    pub slow_period: (usize, usize, usize),
}

impl Default for MaaqBatchRange {
    fn default() -> Self {
        Self {
            period: (11, 50, 1),
            fast_period: (2, 2, 0),
            slow_period: (30, 30, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MaaqBatchBuilder {
    range: MaaqBatchRange,
    kernel: Kernel,
}

impl MaaqBatchBuilder {
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
    pub fn fast_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.fast_period = (start, end, step);
        self
    }
    #[inline]
    pub fn fast_period_static(mut self, x: usize) -> Self {
        self.range.fast_period = (x, x, 0);
        self
    }
    #[inline]
    pub fn slow_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.slow_period = (start, end, step);
        self
    }
    #[inline]
    pub fn slow_period_static(mut self, s: usize) -> Self {
        self.range.slow_period = (s, s, 0);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<MaaqBatchOutput, MaaqError> {
        maaq_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<MaaqBatchOutput, MaaqError> {
        MaaqBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<MaaqBatchOutput, MaaqError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<MaaqBatchOutput, MaaqError> {
        MaaqBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn maaq_batch_with_kernel(
    data: &[f64],
    sweep: &MaaqBatchRange,
    k: Kernel,
) -> Result<MaaqBatchOutput, MaaqError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(MaaqError::InvalidPeriod {
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
    maaq_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct MaaqBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<MaaqParams>,
    pub rows: usize,
    pub cols: usize,
}

impl MaaqBatchOutput {
    pub fn row_for_params(&self, p: &MaaqParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(11) == p.period.unwrap_or(11)
                && c.fast_period.unwrap_or(2) == p.fast_period.unwrap_or(2)
                && c.slow_period.unwrap_or(30) == p.slow_period.unwrap_or(30)
        })
    }
    pub fn values_for(&self, p: &MaaqParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &MaaqBatchRange) -> Vec<MaaqParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let fasts = axis_usize(r.fast_period);
    let slows = axis_usize(r.slow_period);
    let mut out = Vec::with_capacity(periods.len() * fasts.len() * slows.len());
    for &p in &periods {
        for &f in &fasts {
            for &s in &slows {
                out.push(MaaqParams {
                    period: Some(p),
                    fast_period: Some(f),
                    slow_period: Some(s),
                });
            }
        }
    }
    out
}

#[inline(always)]
pub fn maaq_batch_slice(
    data: &[f64],
    sweep: &MaaqBatchRange,
    kern: Kernel,
) -> Result<MaaqBatchOutput, MaaqError> {
    maaq_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn maaq_batch_par_slice(
    data: &[f64],
    sweep: &MaaqBatchRange,
    kern: Kernel,
) -> Result<MaaqBatchOutput, MaaqError> {
    maaq_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn maaq_batch_inner(
    data: &[f64],
    sweep: &MaaqBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<MaaqBatchOutput, MaaqError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(MaaqError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(MaaqError::AllValuesNaN)?;
    let max_p = combos
        .iter()
        .map(|c| c.period.unwrap())
        .max()
        .unwrap();
    if data.len() - first < max_p {
        return Err(MaaqError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();

    // Per-row warm prefix: first non-NaN + that row’s period
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap())
        .collect();

    // 1. allocate the matrix as MaybeUninit and write the NaN prefixes
    let mut raw = make_uninit_matrix(rows, cols);
    unsafe { init_matrix_prefixes(&mut raw, cols, &warm) };

    // 2. closure that fills one row; gets &mut [MaybeUninit<f64>]
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period  = combos[row].period.unwrap();
        let fast_p  = combos[row].fast_period.unwrap();
        let slow_p  = combos[row].slow_period.unwrap();

        // cast this row to &mut [f64]
        let out_row = core::slice::from_raw_parts_mut(
            dst_mu.as_mut_ptr() as *mut f64,
            dst_mu.len(),
        );

        match kern {
            Kernel::Scalar => maaq_row_scalar(data, first, period, fast_p, slow_p, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2   => maaq_row_avx2  (data, first, period, fast_p, slow_p, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => maaq_row_avx512(data, first, period, fast_p, slow_p, out_row),
            _ => unreachable!(),
        }
    };

    // 3. run every row, writing directly into `raw`
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

    // 4. all elements are now initialised – transmute to Vec<f64>
    let values: Vec<f64> = unsafe { std::mem::transmute(raw) };

    // ---------- 5. package result ----------
    Ok(MaaqBatchOutput { values, combos, rows, cols })
}

#[inline(always)]
unsafe fn maaq_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    fast_p: usize,
    slow_p: usize,
    out: &mut [f64],
) {
    maaq_scalar(data, period, fast_p, slow_p, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn maaq_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    fast_p: usize,
    slow_p: usize,
    out: &mut [f64],
) {
   maaq_avx2(data, period, fast_p, slow_p, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn maaq_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    fast_p: usize,
    slow_p: usize,
    out: &mut [f64],
) {
    maaq_avx2(data, period, fast_p, slow_p, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn maaq_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    fast_p: usize,
    slow_p: usize,
    out: &mut [f64],
) {
    maaq_row_scalar(data, first, period, fast_p, slow_p, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn maaq_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    fast_p: usize,
    slow_p: usize,
    out: &mut [f64],
) {
    maaq_row_scalar(data, first, period, fast_p, slow_p, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_maaq_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = MaaqParams {
            period: None,
            fast_period: None,
            slow_period: None,
        };
        let input = MaaqInput::from_candles(&candles, "close", default_params);
        let output = maaq_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_maaq_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = MaaqInput::from_candles(&candles, "close", MaaqParams::default());
        let result = maaq_with_kernel(&input, kernel)?;
        let expected_last_five = [
            59747.657115949725,
            59740.803138018055,
            59724.24153333905,
            59720.60576365108,
            59673.9954445178,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-2,
                "[{}] MAAQ {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_maaq_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = MaaqInput::with_default_candles(&candles);
        match input.data {
            MaaqData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected MaaqData::Candles"),
        }
        let output = maaq_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_maaq_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = MaaqParams {
            period: Some(0),
            fast_period: Some(0),
            slow_period: Some(0),
        };
        let input = MaaqInput::from_slice(&input_data, params);
        let res = maaq_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] MAAQ should fail with zero periods",
            test_name
        );
        Ok(())
    }

    fn check_maaq_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = MaaqParams {
            period: Some(10),
            fast_period: Some(2),
            slow_period: Some(10),
        };
        let input = MaaqInput::from_slice(&data_small, params);
        let res = maaq_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] MAAQ should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_maaq_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = MaaqParams {
            period: Some(9),
            fast_period: Some(2),
            slow_period: Some(10),
        };
        let input = MaaqInput::from_slice(&single_point, params);
        let res = maaq_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] MAAQ should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_maaq_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = MaaqParams {
            period: Some(11),
            fast_period: Some(2),
            slow_period: Some(30),
        };
        let first_input = MaaqInput::from_candles(&candles, "close", first_params);
        let first_result = maaq_with_kernel(&first_input, kernel)?;
        let second_params = MaaqParams {
            period: Some(5),
            fast_period: Some(2),
            slow_period: Some(10),
        };
        let second_input = MaaqInput::from_slice(&first_result.values, second_params);
        let second_result = maaq_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    fn check_maaq_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = MaaqInput::from_candles(
            &candles,
            "close",
            MaaqParams {
                period: Some(11),
                fast_period: Some(2),
                slow_period: Some(30),
            },
        );
        let res = maaq_with_kernel(&input, kernel)?;
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

    fn check_maaq_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 11;
        let fast_p = 2;
        let slow_p = 30;
        let input = MaaqInput::from_candles(
            &candles,
            "close",
            MaaqParams {
                period: Some(period),
                fast_period: Some(fast_p),
                slow_period: Some(slow_p),
            },
        );
        let batch_output = maaq_with_kernel(&input, kernel)?.values;
        let mut stream = MaaqStream::try_new(MaaqParams {
            period: Some(period),
            fast_period: Some(fast_p),
            slow_period: Some(slow_p),
        })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(maaq_val) => stream_values.push(maaq_val),
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
                "[{}] MAAQ streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_maaq_tests {
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

    generate_all_maaq_tests!(
        check_maaq_partial_params,
        check_maaq_accuracy,
        check_maaq_default_candles,
        check_maaq_zero_period,
        check_maaq_period_exceeds_length,
        check_maaq_very_small_dataset,
        check_maaq_reinput,
        check_maaq_nan_handling,
        check_maaq_streaming
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = MaaqBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = MaaqParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
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
