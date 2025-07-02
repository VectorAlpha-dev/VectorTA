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
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel, alloc_with_nan_prefix, init_matrix_prefixes, make_uninit_matrix};
use aligned_vec::{AVec, CACHELINE_ALIGN, ConstAlign};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;
use std::mem::MaybeUninit;

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
        Self { data: TrendFlexData::Candles { candles: c, source: s }, params: p }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: TrendFlexParams) -> Self {
        Self { data: TrendFlexData::Slice(sl), params: p }
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
        Self { period: None, kernel: Kernel::Auto }
    }
}

impl TrendFlexBuilder {
    #[inline(always)]
    pub fn new() -> Self { Self::default() }
    #[inline(always)]
    pub fn period(mut self, n: usize) -> Self {
        self.period = Some(n); self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k; self
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
    let data: &[f64] = match &input.data {
        TrendFlexData::Candles { candles, source } => source_type(candles, source),
        TrendFlexData::Slice(sl) => sl,
    };

    let len = data.len();
    if len == 0 { return Err(TrendFlexError::NoDataProvided); }

    let period = input.get_period();
    if period == 0 { return Err(TrendFlexError::ZeroTrendFlexPeriod { period }); }
    if period > len { return Err(TrendFlexError::TrendFlexPeriodExceedsData { period, data_len: len }); }

    let ss_period = ((period as f64) / 2.0).round() as usize;
    if ss_period > len { return Err(TrendFlexError::SmootherPeriodExceedsData { ss_period, data_len: len }); }

    let first = data.iter().position(|x| !x.is_nan()).ok_or(TrendFlexError::AllValuesNaN)?;

    let warm = first + period;                    // identical to streaming impl
    let mut out = alloc_with_nan_prefix(len, warm);

    // --- choose kernel & run ---------------------------------------------------
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k            => k,
    };

    // all kernel stubs still call `trendflex_scalar / _avx*`, but
    // we copy their **computed part** into our pre-allocated buffer
    unsafe {
        let calc = match chosen {
            Kernel::Scalar | Kernel::ScalarBatch   =>
                trendflex_scalar(data, period, first, &mut out)?,
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2   | Kernel::Avx2Batch     =>
                trendflex_avx2  (data, period, first, &mut out)?,
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch   =>
                trendflex_avx512(data, period, first, &mut out)?,
            _ => unreachable!(),
        };
        // preserve the NaN prefix we just allocated
        out[warm..].copy_from_slice(&calc.values[warm..]);
    }

    Ok(TrendFlexOutput { values: out })
}

// Scalar solution, called by all AVX stubs too
#[inline]
pub unsafe fn trendflex_scalar(
    data: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) -> Result<TrendFlexOutput, TrendFlexError> {
    use std::f64::consts::PI;

    let len = data.len();
    let ss_period = ((period as f64) / 2.0).round() as usize;
    let mut tf = vec![f64::NAN; len];

    if first_valid == 0 {
        let mut ssf: AVec<f64, ConstAlign<CACHELINE_ALIGN>> = AVec::with_capacity(CACHELINE_ALIGN, len);
        ssf.resize(len, 0.0);
        ssf[0] = data[0];
        if len > 1 { ssf[1] = data[1]; }

        let a = (-1.414_f64 * PI / (ss_period as f64)).exp();
        let a_sq = a * a;
        let b = 2.0 * a * (1.414_f64 * PI / (ss_period as f64)).cos();
        let c = (1.0 + a_sq - b) * 0.5;

        for i in 2..len {
            ssf[i] = c * (data[i] + data[i - 1]) + b * ssf[i - 1] - a_sq * ssf[i - 2];
        }

        let mut ms_prev = 0.0;
        let tp_f = period as f64;
        let inv_tp = 1.0 / tp_f;
        let mut rolling_sum = ssf[..period].iter().sum::<f64>();

        for i in period..len {
            let my_sum = (tp_f * ssf[i] - rolling_sum) * inv_tp;
            let ms_current = 0.04 * my_sum * my_sum + 0.96 * ms_prev;
            ms_prev = ms_current;

            tf[i] = if ms_current != 0.0 { my_sum / ms_current.sqrt() } else { 0.0 };
            rolling_sum += ssf[i] - ssf[i - period];
        }
    } else {
        let m = len - first_valid;
        if m < period {
            return Ok(TrendFlexOutput { values: vec![f64::NAN; len] });
        }
        if m < ss_period {
            return Err(TrendFlexError::SmootherPeriodExceedsData { ss_period, data_len: m });
        }
        let mut tmp_data: AVec<f64, ConstAlign<CACHELINE_ALIGN>> = AVec::with_capacity(CACHELINE_ALIGN, m);
        tmp_data.resize(m, 0.0);
        tmp_data.copy_from_slice(&data[first_valid..]);

        let mut tmp_ssf: AVec<f64, ConstAlign<CACHELINE_ALIGN>> = AVec::with_capacity(CACHELINE_ALIGN, m);
        tmp_ssf.resize(m, 0.0);
        tmp_ssf[0] = tmp_data[0];
        if m > 1 { tmp_ssf[1] = tmp_data[1]; }

        let a = (-1.414_f64 * PI / (ss_period as f64)).exp();
        let a_sq = a * a;
        let b = 2.0 * a * (1.414_f64 * PI / (ss_period as f64)).cos();
        let c = (1.0 + a_sq - b) * 0.5;

        for i in 2..m {
            tmp_ssf[i] = c * (tmp_data[i] + tmp_data[i - 1]) + b * tmp_ssf[i - 1] - a_sq * tmp_ssf[i - 2];
        }

        let mut ms_prev = 0.0;
        let tp_f = period as f64;
        let inv_tp = 1.0 / tp_f;
        let mut rolling_sum = tmp_ssf[..period].iter().sum::<f64>();

        let mut tmp_tf: AVec<f64, ConstAlign<CACHELINE_ALIGN>> = AVec::with_capacity(CACHELINE_ALIGN, m);
        tmp_tf.resize(m, f64::NAN);

        for i in period..m {
            let my_sum = (tp_f * tmp_ssf[i] - rolling_sum) * inv_tp;
            let ms_current = 0.04 * (my_sum * my_sum) + 0.96 * ms_prev;
            ms_prev = ms_current;
            tmp_tf[i] = if ms_current != 0.0 { my_sum / ms_current.sqrt() } else { 0.0 };
            rolling_sum += tmp_ssf[i] - tmp_ssf[i - period];
        }
        for i in 0..m {
            tf[first_valid + i] = tmp_tf[i];
        }
    }

    Ok(TrendFlexOutput { values: tf })
}

// AVX2 stub
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn trendflex_avx2(
    data: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) -> Result<TrendFlexOutput, TrendFlexError> {
    // Calls scalar solution, maintains API
    trendflex_scalar(data, period, first_valid, out)
}

// AVX512 stub
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn trendflex_avx512(
    data: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) -> Result<TrendFlexOutput, TrendFlexError> {
    if period <= 32 {
        trendflex_avx512_short(data, period, first_valid, out)
    
        } else {
        trendflex_avx512_long(data, period, first_valid, out)
    }
}

// AVX512 short stub
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn trendflex_avx512_short(
    data: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) -> Result<TrendFlexOutput, TrendFlexError> {
    trendflex_scalar(data, period, first_valid, out)
}

// AVX512 long stub
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn trendflex_avx512_long(
    data: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) -> Result<TrendFlexOutput, TrendFlexError> {
    trendflex_scalar(data, period, first_valid, out)
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
            a, a_sq, b, c,
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
        if !self.filled { return None; }

        let tp_f = p as f64;
        let inv_tp = 1.0 / tp_f;

        let ms_current = 0.04 * my_sum * my_sum + 0.96 * self.ms_prev;
        self.ms_prev = ms_current;

        Some(if ms_current != 0.0 { my_sum / ms_current.sqrt() } else { 0.0 })
    }
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
    pub fn new() -> Self { Self::default() }
    pub fn kernel(mut self, k: Kernel) -> Self { self.kernel = k; self }
    #[inline]
    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.period = (start, end, step); self
    }
    #[inline]
    pub fn period_static(mut self, p: usize) -> Self {
        self.range.period = (p, p, 0); self
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
        TrendFlexBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
    }
}

pub fn trendflex_batch_with_kernel(
    data: &[f64],
    sweep: &TrendFlexBatchRange,
    k: Kernel,
) -> Result<TrendFlexBatchOutput, TrendFlexError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(TrendFlexError::NoDataProvided),
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
        self.combos.iter().position(|c| c.period.unwrap_or(20) == p.period.unwrap_or(20))
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
        if step == 0 || start == end { return vec![start]; }
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
    let first = data.iter().position(|x| !x.is_nan()).ok_or(TrendFlexError::AllValuesNaN)?;
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
    unsafe { init_matrix_prefixes(&mut raw, cols, &warm); }

    // 3. helper that fills **one row**; receives &mut [MaybeUninit<f64>]
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period  = combos[row].period.unwrap();

        // Cast this single row to &mut [f64] so the existing row helpers work
        let out_row = core::slice::from_raw_parts_mut(
            dst_mu.as_mut_ptr() as *mut f64,
            dst_mu.len(),
        );

        match kern {
            Kernel::Scalar => trendflex_row_scalar(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2   => trendflex_row_avx2  (data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => trendflex_row_avx512(data, first, period, out_row),
            _ => unreachable!(),
        }
    };

    // 4. run every row, writing directly into `raw`
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

    // 5. all elements are now initialised → transmute to Vec<f64>
    let values: Vec<f64> = unsafe { std::mem::transmute(raw) };

    Ok(TrendFlexBatchOutput { values, combos, rows, cols })
}

// Row functions -- AVX variants are just stubs to scalar
#[inline(always)]
unsafe fn trendflex_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    let res = trendflex_scalar(data, period, first, out);
    if let Ok(v) = res {
        out.copy_from_slice(&v.values);
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn trendflex_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    trendflex_row_scalar(data, first, period, out);
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn trendflex_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        trendflex_row_avx512_short(data, first, period, out);
    
        } else {
        trendflex_row_avx512_long(data, first, period, out);
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn trendflex_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    trendflex_row_scalar(data, first, period, out);
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn trendflex_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    trendflex_row_scalar(data, first, period, out);
}

// Test coverage -- use alma.rs style macros and patterns

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

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
        assert!(
            res.is_err(),
            "[{}] TrendFlex should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_trendflex_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
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

    fn check_trendflex_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
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

        let input = TrendFlexInput::from_candles(
            &candles,
            "close",
            TrendFlexParams { period: Some(20) },
        );
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

        let input = TrendFlexInput::from_candles(
            &candles,
            "close",
            TrendFlexParams { period: Some(period) },
        );
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
            if b.is_nan() && s.is_nan() { continue; }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] TrendFlex streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name, i, b, s, diff
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

    generate_all_trendflex_tests!(
        check_trendflex_partial_params,
        check_trendflex_accuracy,
        check_trendflex_default_candles,
        check_trendflex_zero_period,
        check_trendflex_period_exceeds_length,
        check_trendflex_very_small_dataset,
        check_trendflex_reinput,
        check_trendflex_nan_handling,
        check_trendflex_streaming
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = TrendFlexBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

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
    gen_batch_tests!(check_batch_default_row);
}
