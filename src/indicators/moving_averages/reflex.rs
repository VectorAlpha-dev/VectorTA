//! # Reflex
//!
//! An indicator (attributed to John Ehlers) designed to detect turning points in a time
//! series by comparing a 2-pole filtered version of the data to a projected slope over
//! a specified window (`period`). It then adjusts its output (`Reflex`) based on the
//! difference between predicted and past values, normalized by a rolling measure of
//! variance. Includes batch/grid operation, builder APIs, and supports AVX2/AVX512 (stubbed).
//!
//! ## Parameters
//! - **period**: The window size used for measuring and predicting the slope (must be ≥ 2).
//!
//! ## Errors
//! - **NoData**: reflex: No data provided (empty slice).
//! - **InvalidPeriod**: reflex: `period` < 2.
//! - **NotEnoughData**: reflex: The available data is shorter than `period`.
//! - **AllValuesNaN**: reflex: All input data values are `NaN`.
//!
//! ## Returns
//! - **`Ok(ReflexOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(ReflexError)`** otherwise.

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

impl<'a> AsRef<[f64]> for ReflexInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            ReflexData::Slice(slice) => slice,
            ReflexData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ReflexData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct ReflexOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct ReflexParams {
    pub period: Option<usize>,
}

impl Default for ReflexParams {
    fn default() -> Self {
        Self { period: Some(20) }
    }
}

#[derive(Debug, Clone)]
pub struct ReflexInput<'a> {
    pub data: ReflexData<'a>,
    pub params: ReflexParams,
}

impl<'a> ReflexInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: ReflexParams) -> Self {
        Self {
            data: ReflexData::Candles { candles: c, source: s },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: ReflexParams) -> Self {
        Self {
            data: ReflexData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", ReflexParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(20)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ReflexBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for ReflexBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl ReflexBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<ReflexOutput, ReflexError> {
        let p = ReflexParams { period: self.period };
        let i = ReflexInput::from_candles(c, "close", p);
        reflex_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<ReflexOutput, ReflexError> {
        let p = ReflexParams { period: self.period };
        let i = ReflexInput::from_slice(d, p);
        reflex_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<ReflexStream, ReflexError> {
        let p = ReflexParams { period: self.period };
        ReflexStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum ReflexError {
    #[error("reflex: No data available for Reflex.")]
    NoData,
    #[error("reflex: Reflex period must be >=2. Provided period was {period}")]
    InvalidPeriod { period: usize },
    #[error("reflex: Not enough data: needed {needed}, found {found}")]
    NotEnoughData { needed: usize, found: usize },
    #[error("reflex: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn reflex(input: &ReflexInput) -> Result<ReflexOutput, ReflexError> {
    reflex_with_kernel(input, Kernel::Auto)
}

pub fn reflex_with_kernel(input: &ReflexInput, kernel: Kernel) -> Result<ReflexOutput, ReflexError> {
    let data: &[f64] = match &input.data {
        ReflexData::Candles { candles, source } => source_type(candles, source),
        ReflexData::Slice(sl) => sl,
    };
    let first = data.iter().position(|x| !x.is_nan()).ok_or(ReflexError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();

    if len == 0 {
        return Err(ReflexError::NoData);
    }
    if period < 2 {
        return Err(ReflexError::InvalidPeriod { period });
    }
    if period > len {
        return Err(ReflexError::NotEnoughData {
            needed: period,
            found: len,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let warm = first + period;
    let mut out = alloc_with_nan_prefix(len, period);
    for x in &mut out[..period.min(len)] {
        *x = 0.0;
    }

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                reflex_scalar(data, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                reflex_avx2(data, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                reflex_avx512(data, period, first, &mut out)
            }
            _ => unreachable!(),
        }
    }

    Ok(ReflexOutput { values: out })
}

#[inline]
pub fn reflex_scalar(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    let len = data.len();
    if len == 0 || period < 2 {
        return;
    }

    // ------------------------------------------------------------------------
    // 2-pole smoothing filter coefficients (identical to the original version)
    // ------------------------------------------------------------------------
    let half_period = (period / 2).max(1);
    let a     = (-1.414_f64 * std::f64::consts::PI / half_period as f64).exp();
    let a_sq  = a * a;
    let b     = 2.0 * a * (1.414_f64 * std::f64::consts::PI / half_period as f64).cos();
    let c     = (1.0 + a_sq - b) * 0.5;

    // ------------------------------------------------------------------------
    // Working buffers
    // ------------------------------------------------------------------------
    let mut ssf  = vec![0.0; len];   // 2-pole smoothed series
    let mut ms   = vec![0.0; len];   // rolling mean-square of “my_sum”
    let mut sums = vec![0.0; len];   // raw “my_sum” values (for debugging)

    // ------------------------------------------------------------------------
    // Seed the first two ssf values (per the original algorithm)
    // ------------------------------------------------------------------------
    ssf[0] = data[0];
    if len > 1 {
        ssf[1] = data[1];
    }

    let period_f = period as f64;

    // ------------------------------------------------------------------------
    // Main loop
    // ------------------------------------------------------------------------
    for i in 2..len {
        // ---- 1. update the 2-pole smoothed price (ssf[i]) -------------------
        let d_i     = data[i];
        let d_im1   = data[i - 1];
        let ssf_im1 = ssf[i - 1];
        let ssf_im2 = ssf[i - 2];

        let ssf_i = c * (d_i + d_im1) + b * ssf_im1 - a_sq * ssf_im2;
        ssf[i] = ssf_i;

        // ---- 2. once we have at least `period` values, compute Reflex -------
        if i >= period {
            // slope of the line connecting ssf[i-period] … ssf[i]
            let slope = (ssf[i - period] - ssf_i) / period_f;

            // ∑_{t = 1..period} ( predicted – past )
            let mut my_sum = 0.0;
            for t in 1..=period {
                let pred = ssf_i + slope * (t as f64);
                let past = ssf[i - t];
                my_sum += pred - past;
            }
            my_sum /= period_f;
            sums[i] = my_sum;

            // exponentially-weighted rolling variance proxy (ms[i])
            let ms_im1 = ms[i - 1];
            let ms_i   = 0.04 * my_sum * my_sum + 0.96 * ms_im1;
            ms[i] = ms_i;

            // ---- 3. write output *only* after the warm-up prefix ------------
            if i >= period && ms_i > 0.0 {
                out[i] = my_sum / ms_i.sqrt();
            }
            // else: leave the NaN written by `alloc_with_nan_prefix`
        }
    }
}


#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn reflex_avx2(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    reflex_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn reflex_avx512(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    if period <= 32 {
        unsafe { reflex_avx512_short(data, period, first, out) }
    } else {
        unsafe { reflex_avx512_long(data, period, first, out) }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn reflex_avx512_short(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    reflex_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn reflex_avx512_long(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    reflex_scalar(data, period, first, out)
}

// --- Streaming API ---

#[derive(Debug, Clone)]
pub struct ReflexStream {
    period: usize,

    // coefficients (all constant after construction)
    a:    f64,
    a_sq: f64,
    b:    f64,
    c:    f64,

    // we keep a circular buffer of length (period + 1) for all past ssf[]
    ssf_buf: Vec<f64>,

    // running sum of “last period” ssf values:
    //   at time t (just before computing output if t >= period),
    //   `ssf_sum` = Σ_{k = t - period .. t - 1} ssf[k].
    ssf_sum:  f64,

    // we need the raw price from one step ago, so we can compute
    //   ssf[t] = c*(data[t] + data[t-1]) + b*ssf[t-1] - a_sq*ssf[t-2]
    last_data: Option<f64>,

    // keep a single “ms[t-1]” so that ms[t] = 0.04·my_sum² + 0.96·ms[t-1]
    last_ms: f64,

    // how many values have been fed in so far (this is “t” in the batch code)
    count:   usize,
}

impl ReflexStream {
    pub fn try_new(params: ReflexParams) -> Result<Self, ReflexError> {
        let period = params.period.unwrap_or(20);
        if period < 2 {
            return Err(ReflexError::InvalidPeriod { period });
        }

        // exactly the same coefficients that `reflex_scalar` uses:
        //
        //     let half_period = (period / 2).max(1);
        //     let a      = exp(-1.414 * π / half_period);
        //     let a_sq   = a * a;
        //     let b      = 2.0 * a * cos(1.414 * π / half_period);
        //     let c      = (1.0 + a_sq - b) * 0.5;
        //
        // we compute `half_period` as f64 because that’s how the scalar version does it.
        let half_period = (period / 2).max(1) as f64;
        let a          = (-1.414_f64 * std::f64::consts::PI / half_period).exp();
        let a_sq       = a * a;
        let b          = 2.0 * a * (1.414_f64 * std::f64::consts::PI / half_period).cos();
        let c          = (1.0 + a_sq - b) * 0.5;

        Ok(Self {
            period,

            a,
            a_sq,
            b,
            c,

            // buffer for ssf[ t mod (period+1) ], so we can index ssf[t-1], ssf[t-2], ssf[t-period]
            ssf_buf: vec![0.0; period + 1],

            // at the very start, we have no ssf history => sum = 0
            ssf_sum: 0.0,

            last_data: None,
            last_ms:   0.0,
            count:     0,
        })
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        let t = self.count;
        let period = self.period;

        // 1) compute ssf[t] exactly as in `reflex_scalar`:
        let ssf_t: f64 = if t == 0 {
            // at t = 0: ssf[0] = data[0]
            value
        } else if t == 1 {
            // at t = 1: ssf[1] = data[1]
            value
        
            } else {
            // for t >= 2: ssf[t] = c*(data[t] + data[t-1]) + b*ssf[t-1] - a_sq*ssf[t-2]
            let prev_data = self.last_data.unwrap();
            let idx1 = (t - 1) % (period + 1);
            let idx2 = (t - 2) % (period + 1);
            let ssf_t1 = self.ssf_buf[idx1];
            let ssf_t2 = self.ssf_buf[idx2];
            self.c * (value + prev_data) + self.b * ssf_t1 - self.a_sq * ssf_t2
        };

        // 2) if t >= period, compute the normalized “Reflex” exactly as in batch:
        let mut out_val = 0.0;
        if t >= period {
            // ssf[t - period]:
            let idx_period = (t - period) % (period + 1);
            let ssf_t_period = self.ssf_buf[idx_period];

            let period_f = period as f64;
            let my_sum = ssf_t
                + ((ssf_t_period - ssf_t) * (period_f + 1.0) / (2.0 * period_f))
                - (self.ssf_sum / period_f);

            let my_sum_sq = my_sum * my_sum;
            let ms_t = 0.04 * my_sum_sq + 0.96 * self.last_ms;
            self.last_ms = ms_t;

            if ms_t > 0.0 {
                out_val = my_sum / ms_t.sqrt();
} else {
                out_val = 0.0;
            }
        }

        // 3) update the rolling sum of ssf for the “next” step:
        //
        //    If t < period, we haven’t reached a full window yet, so we simply
        //    add this ssf[t] to `ssf_sum`.  At the moment t == period, that
        //    means `ssf_sum = Σ_{i=0..period-1} ssf[i]`, which is exactly what
        //    the batch code wants before computing “my_sum” at i == period.
        //
        //    Once t >= period, we must subtract off ssf[t - period] and add
        //    ssf[t], so that `ssf_sum = Σ_{i = (t - period + 1) .. t}` for the
        //    next iteration.
        if t < period {
            self.ssf_sum += ssf_t;
        
            } else {
            let idx_remove = (t - period) % (period + 1);
            let remove_ssf = self.ssf_buf[idx_remove];
            self.ssf_sum = self.ssf_sum - remove_ssf + ssf_t;
        }

        // 4) store the new ssf[t] into our circular buffer:
        self.ssf_buf[t % (period + 1)] = ssf_t;

        // 5) remember this raw price so the *next* call can use data[t-1]:
        self.last_data = Some(value);

        // 6) advance the counter:
        self.count += 1;

        // 7) return `Some(out_val)` only once t >= period; otherwise return None
        if t >= period {
            Some(out_val)
        
            } else {
            None
        }
    }
}


// --- Batch/grid API ---

#[derive(Clone, Debug)]
pub struct ReflexBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for ReflexBatchRange {
    fn default() -> Self {
        Self { period: (20, 20, 0) }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ReflexBatchBuilder {
    range: ReflexBatchRange,
    kernel: Kernel,
}

impl ReflexBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<ReflexBatchOutput, ReflexError> {
        reflex_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<ReflexBatchOutput, ReflexError> {
        ReflexBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<ReflexBatchOutput, ReflexError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<ReflexBatchOutput, ReflexError> {
        ReflexBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn reflex_batch_with_kernel(
    data: &[f64],
    sweep: &ReflexBatchRange,
    k: Kernel,
) -> Result<ReflexBatchOutput, ReflexError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(ReflexError::InvalidPeriod { period: 0 }),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    reflex_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct ReflexBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<ReflexParams>,
    pub rows: usize,
    pub cols: usize,
}

impl ReflexBatchOutput {
    pub fn row_for_params(&self, p: &ReflexParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(20) == p.period.unwrap_or(20)
        })
    }
    pub fn values_for(&self, p: &ReflexParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &ReflexBatchRange) -> Vec<ReflexParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(ReflexParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn reflex_batch_slice(
    data: &[f64],
    sweep: &ReflexBatchRange,
    kern: Kernel,
) -> Result<ReflexBatchOutput, ReflexError> {
    reflex_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn reflex_batch_par_slice(
    data: &[f64],
    sweep: &ReflexBatchRange,
    kern: Kernel,
) -> Result<ReflexBatchOutput, ReflexError> {
    reflex_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn reflex_batch_inner(
    data: &[f64],
    sweep: &ReflexBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<ReflexBatchOutput, ReflexError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(ReflexError::InvalidPeriod { period: 0 });
    }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(ReflexError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(ReflexError::NotEnoughData { needed: max_p, found: data.len() - first });
    }

    let rows = combos.len();
    let cols = data.len();
    let warm: Vec<usize> = combos.iter()
                                .map(|c| c.period.unwrap())
                                .collect();
    let mut raw: Vec<MaybeUninit<f64>> = make_uninit_matrix(rows, cols);
    unsafe { init_matrix_prefixes(&mut raw, cols, &warm) };

    // ---- row-filler closure ----------------------------------------------------
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();

        // cast this row to &mut [f64]
        let out_row = core::slice::from_raw_parts_mut(
            dst_mu.as_mut_ptr() as *mut f64,
            dst_mu.len(),
        );

        match kern {
            Kernel::Scalar => reflex_row_scalar(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2   => reflex_row_avx2  (data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => reflex_row_avx512(data, first, period, out_row),
            _ => unreachable!(),
        }
    };

    // ---- fill every row directly into `raw` ------------------------------------
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

    } else {
        for (row, slice) in raw.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    // ---- transmute after all rows are written ----------------------------------
    let mut values: Vec<f64> = unsafe { std::mem::transmute(raw) };

    for (row, prm) in combos.iter().enumerate() {
        let p = prm.period.unwrap();
        let start = row * cols;
        for cell in &mut values[start .. start + p.min(cols)] {
            *cell = 0.0;
        }
    }
    return Ok(ReflexBatchOutput { values, combos, rows, cols });
}

#[inline(always)]
unsafe fn reflex_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    reflex_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn reflex_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    reflex_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn reflex_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    if period <= 32 {
        reflex_row_avx512_short(data, first, period, out);
    
        } else {
        reflex_row_avx512_long(data, first, period, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn reflex_row_avx512_short(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    reflex_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn reflex_row_avx512_long(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    reflex_scalar(data, period, first, out)
}

// -- Test coverage macros: ALMA parity --

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_reflex_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = ReflexParams { period: None };
        let input = ReflexInput::from_candles(&candles, "close", default_params);
        let output = reflex_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        let params_period_14 = ReflexParams { period: Some(14) };
        let input2 = ReflexInput::from_candles(&candles, "hl2", params_period_14);
        let output2 = reflex_with_kernel(&input2, kernel)?;
        assert_eq!(output2.values.len(), candles.close.len());
        let params_custom = ReflexParams { period: Some(30) };
        let input3 = ReflexInput::from_candles(&candles, "hlc3", params_custom);
        let output3 = reflex_with_kernel(&input3, kernel)?;
        assert_eq!(output3.values.len(), candles.close.len());
        Ok(())
    }

    fn check_reflex_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = ReflexParams::default();
        let input = ReflexInput::from_candles(&candles, "close", default_params);
        let result = reflex_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        let len = result.values.len();
        let expected_last_five = [
            0.8085220962465361,
            0.445264715886137,
            0.13861699036615063,
            -0.03598639652007061,
            -0.224906760543743,
        ];
        let start_idx = len - 5;
        let last_five = &result.values[start_idx..];
        for (i, &val) in last_five.iter().enumerate() {
            let exp = expected_last_five[i];
            assert!(
                (val - exp).abs() < 1e-7,
                "[{}] Reflex mismatch at idx {}: got {}, expected {}",
                test_name, i, val, exp
            );
        }
        Ok(())
    }

    fn check_reflex_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = ReflexInput::with_default_candles(&candles);
        match input.data {
            ReflexData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected ReflexData::Candles"),
        }
        let output = reflex_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_reflex_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = ReflexParams { period: Some(0) };
        let input = ReflexInput::from_slice(&input_data, params);
        let res = reflex_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] Reflex should fail with zero period", test_name);
        Ok(())
    }

    fn check_reflex_period_less_than_two(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = ReflexParams { period: Some(1) };
        let input = ReflexInput::from_slice(&input_data, params);
        let res = reflex_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] Reflex should fail with period<2", test_name);
        Ok(())
    }

    fn check_reflex_very_small_data_set(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [42.0];
        let params = ReflexParams { period: Some(2) };
        let input = ReflexInput::from_slice(&input_data, params);
        let res = reflex_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] Reflex should fail with insufficient data", test_name);
        Ok(())
    }

    fn check_reflex_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = ReflexParams { period: Some(14) };
        let first_input = ReflexInput::from_candles(&candles, "close", first_params);
        let first_result = reflex_with_kernel(&first_input, kernel)?;
        assert_eq!(first_result.values.len(), candles.close.len());
        let second_params = ReflexParams { period: Some(10) };
        let second_input = ReflexInput::from_slice(&first_result.values, second_params);
        let second_result = reflex_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        for i in 14..second_result.values.len() {
            assert!(second_result.values[i].is_finite());
        }
        Ok(())
    }

    fn check_reflex_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 14;
        let params = ReflexParams { period: Some(period) };
        let input = ReflexInput::from_candles(&candles, "close", params);
        let result = reflex_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        if result.values.len() > period {
            for i in period..result.values.len() {
                assert!(
                    result.values[i].is_finite(),
                    "[{}] Unexpected NaN at index {}",
                    test_name, i
                );
            }
        }
        Ok(())
    }

    fn check_reflex_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 14;
        let params = ReflexParams { period: Some(period) };
        let input = ReflexInput::from_candles(&candles, "close", params.clone());
        let batch_output = reflex_with_kernel(&input, kernel)?.values;
        let mut stream = ReflexStream::try_new(params)?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(v) => stream_values.push(v),
                None => stream_values.push(0.0),
            }
        }
        assert_eq!(batch_output.len(), stream_values.len());
        for (i, (&b, &s)) in batch_output.iter().zip(stream_values.iter()).enumerate() {
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] Reflex streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name, i, b, s, diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_reflex_tests {
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

    generate_all_reflex_tests!(
        check_reflex_partial_params,
        check_reflex_accuracy,
        check_reflex_default_candles,
        check_reflex_zero_period,
        check_reflex_period_less_than_two,
        check_reflex_very_small_data_set,
        check_reflex_reinput,
        check_reflex_nan_handling,
        check_reflex_streaming
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = ReflexBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = ReflexParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());

        let expected = [
            0.8085220962465361,
            0.445264715886137,
            0.13861699036615063,
            -0.03598639652007061,
            -0.224906760543743,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-7,
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
