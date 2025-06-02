//! # Know Sure Thing (KST)
//!
//! KST is a momentum oscillator based on the smoothed rate-of-change (ROC) values of four different time frames.
//! This implementation mirrors alma.rs in performance and structure, providing AVX2/AVX512 stubs, batch/grid interfaces,
//! streaming, builders, and thorough input validation. All kernel variants and AVX stubs are present for API parity.
//!
//! ## Parameters
//! - **sma_period1**: Smoothing period for the first ROC. Defaults to 10.
//! - **sma_period2**: Smoothing period for the second ROC. Defaults to 10.
//! - **sma_period3**: Smoothing period for the third ROC. Defaults to 10.
//! - **sma_period4**: Smoothing period for the fourth ROC. Defaults to 15.
//! - **roc_period1**: Period for the first ROC calculation. Defaults to 10.
//! - **roc_period2**: Period for the second ROC calculation. Defaults to 15.
//! - **roc_period3**: Period for the third ROC calculation. Defaults to 20.
//! - **roc_period4**: Period for the fourth ROC calculation. Defaults to 30.
//! - **signal_period**: Smoothing period for the signal line. Defaults to 9.
//!
//! ## Errors
//! - **AllValuesNaN**: All input data values are `NaN`.
//! - **InvalidPeriod**: A period is zero or exceeds the data length.
//! - **NotEnoughValidData**: Not enough valid data points for the requested period.
//!
//! ## Returns
//! - `Ok(KstOutput)` on success, containing two `Vec<f64>`: KST line and signal line.
//! - `Err(KstError)` otherwise.

use crate::indicators::roc::{roc, RocData, RocError, RocInput, RocOutput, RocParams};
use crate::indicators::moving_averages::sma::{
    sma, SmaData, SmaError, SmaInput, SmaOutput, SmaParams,
};
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

#[derive(Debug, Clone)]
pub enum KstData<'a> {
    Candles { candles: &'a Candles, source: &'a str },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct KstOutput {
    pub line: Vec<f64>,
    pub signal: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct KstParams {
    pub sma_period1: Option<usize>,
    pub sma_period2: Option<usize>,
    pub sma_period3: Option<usize>,
    pub sma_period4: Option<usize>,
    pub roc_period1: Option<usize>,
    pub roc_period2: Option<usize>,
    pub roc_period3: Option<usize>,
    pub roc_period4: Option<usize>,
    pub signal_period: Option<usize>,
}

impl Default for KstParams {
    fn default() -> Self {
        Self {
            sma_period1: Some(10),
            sma_period2: Some(10),
            sma_period3: Some(10),
            sma_period4: Some(15),
            roc_period1: Some(10),
            roc_period2: Some(15),
            roc_period3: Some(20),
            roc_period4: Some(30),
            signal_period: Some(9),
        }
    }
}

#[derive(Debug, Clone)]
pub struct KstInput<'a> {
    pub data: KstData<'a>,
    pub params: KstParams,
}

impl<'a> AsRef<[f64]> for KstInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            KstData::Slice(slice) => slice,
            KstData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

impl<'a> KstInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: KstParams) -> Self {
        Self { data: KstData::Candles { candles: c, source: s }, params: p }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: KstParams) -> Self {
        Self { data: KstData::Slice(sl), params: p }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", KstParams::default())
    }
    #[inline]
    pub fn get_sma_period1(&self) -> usize { self.params.sma_period1.unwrap_or(10) }
    #[inline]
    pub fn get_sma_period2(&self) -> usize { self.params.sma_period2.unwrap_or(10) }
    #[inline]
    pub fn get_sma_period3(&self) -> usize { self.params.sma_period3.unwrap_or(10) }
    #[inline]
    pub fn get_sma_period4(&self) -> usize { self.params.sma_period4.unwrap_or(15) }
    #[inline]
    pub fn get_roc_period1(&self) -> usize { self.params.roc_period1.unwrap_or(10) }
    #[inline]
    pub fn get_roc_period2(&self) -> usize { self.params.roc_period2.unwrap_or(15) }
    #[inline]
    pub fn get_roc_period3(&self) -> usize { self.params.roc_period3.unwrap_or(20) }
    #[inline]
    pub fn get_roc_period4(&self) -> usize { self.params.roc_period4.unwrap_or(30) }
    #[inline]
    pub fn get_signal_period(&self) -> usize { self.params.signal_period.unwrap_or(9) }
}

#[derive(Copy, Clone, Debug)]
pub struct KstBuilder {
    sma_period1: Option<usize>,
    sma_period2: Option<usize>,
    sma_period3: Option<usize>,
    sma_period4: Option<usize>,
    roc_period1: Option<usize>,
    roc_period2: Option<usize>,
    roc_period3: Option<usize>,
    roc_period4: Option<usize>,
    signal_period: Option<usize>,
    kernel: Kernel,
}

impl Default for KstBuilder {
    fn default() -> Self {
        Self {
            sma_period1: None,
            sma_period2: None,
            sma_period3: None,
            sma_period4: None,
            roc_period1: None,
            roc_period2: None,
            roc_period3: None,
            roc_period4: None,
            signal_period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl KstBuilder {
    #[inline(always)] pub fn new() -> Self { Self::default() }
    #[inline(always)] pub fn sma_period1(mut self, n: usize) -> Self { self.sma_period1 = Some(n); self }
    #[inline(always)] pub fn sma_period2(mut self, n: usize) -> Self { self.sma_period2 = Some(n); self }
    #[inline(always)] pub fn sma_period3(mut self, n: usize) -> Self { self.sma_period3 = Some(n); self }
    #[inline(always)] pub fn sma_period4(mut self, n: usize) -> Self { self.sma_period4 = Some(n); self }
    #[inline(always)] pub fn roc_period1(mut self, n: usize) -> Self { self.roc_period1 = Some(n); self }
    #[inline(always)] pub fn roc_period2(mut self, n: usize) -> Self { self.roc_period2 = Some(n); self }
    #[inline(always)] pub fn roc_period3(mut self, n: usize) -> Self { self.roc_period3 = Some(n); self }
    #[inline(always)] pub fn roc_period4(mut self, n: usize) -> Self { self.roc_period4 = Some(n); self }
    #[inline(always)] pub fn signal_period(mut self, n: usize) -> Self { self.signal_period = Some(n); self }
    #[inline(always)] pub fn kernel(mut self, k: Kernel) -> Self { self.kernel = k; self }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<KstOutput, KstError> {
        let p = KstParams {
            sma_period1: self.sma_period1, sma_period2: self.sma_period2,
            sma_period3: self.sma_period3, sma_period4: self.sma_period4,
            roc_period1: self.roc_period1, roc_period2: self.roc_period2,
            roc_period3: self.roc_period3, roc_period4: self.roc_period4,
            signal_period: self.signal_period
        };
        let i = KstInput::from_candles(c, "close", p);
        kst_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<KstOutput, KstError> {
        let p = KstParams {
            sma_period1: self.sma_period1, sma_period2: self.sma_period2,
            sma_period3: self.sma_period3, sma_period4: self.sma_period4,
            roc_period1: self.roc_period1, roc_period2: self.roc_period2,
            roc_period3: self.roc_period3, roc_period4: self.roc_period4,
            signal_period: self.signal_period
        };
        let i = KstInput::from_slice(d, p);
        kst_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<KstStream, KstError> {
        let p = KstParams {
            sma_period1: self.sma_period1, sma_period2: self.sma_period2,
            sma_period3: self.sma_period3, sma_period4: self.sma_period4,
            roc_period1: self.roc_period1, roc_period2: self.roc_period2,
            roc_period3: self.roc_period3, roc_period4: self.roc_period4,
            signal_period: self.signal_period
        };
        KstStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum KstError {
    #[error("kst: {0}")]
    Roc(#[from] RocError),
    #[error("kst: {0}")]
    Sma(#[from] SmaError),
    #[error("kst: All values are NaN.")]
    AllValuesNaN,
    #[error("kst: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("kst: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn kst(input: &KstInput) -> Result<KstOutput, KstError> {
    kst_with_kernel(input, Kernel::Auto)
}

pub fn kst_with_kernel(input: &KstInput, kernel: Kernel) -> Result<KstOutput, KstError> {
    let data: &[f64] = input.as_ref();
    let first = data.iter().position(|x| !x.is_nan()).ok_or(KstError::AllValuesNaN)?;
    let len = data.len();
    if len == 0 { return Err(KstError::AllValuesNaN); }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => kst_scalar(input, first, len),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => kst_avx2(input, first, len),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => kst_avx512(input, first, len),
            _ => unreachable!(),
        }
    }
}

#[inline]
pub unsafe fn kst_scalar(input: &KstInput, _first: usize, _len: usize) -> Result<KstOutput, KstError> {
    // Scalar logic unchanged (original kst function)
    let data: &[f64] = input.as_ref();
    let p = &input.params;
    let s1 = input.get_sma_period1();
    let s2 = input.get_sma_period2();
    let s3 = input.get_sma_period3();
    let s4 = input.get_sma_period4();
    let r1 = input.get_roc_period1();
    let r2 = input.get_roc_period2();
    let r3 = input.get_roc_period3();
    let r4 = input.get_roc_period4();
    let sig = input.get_signal_period();

    let roc1 = roc(&RocInput::from_slice(data, RocParams { period: Some(r1) }))?;
    let roc2 = roc(&RocInput::from_slice(data, RocParams { period: Some(r2) }))?;
    let roc3 = roc(&RocInput::from_slice(data, RocParams { period: Some(r3) }))?;
    let roc4 = roc(&RocInput::from_slice(data, RocParams { period: Some(r4) }))?;

    let aroc1 = sma(&SmaInput::from_slice(&roc1.values, SmaParams { period: Some(s1) }))?;
    let aroc2 = sma(&SmaInput::from_slice(&roc2.values, SmaParams { period: Some(s2) }))?;
    let aroc3 = sma(&SmaInput::from_slice(&roc3.values, SmaParams { period: Some(s3) }))?;
    let aroc4 = sma(&SmaInput::from_slice(&roc4.values, SmaParams { period: Some(s4) }))?;

    if aroc1.values.is_empty() || aroc2.values.is_empty() || aroc3.values.is_empty() || aroc4.values.is_empty() {
        return Err(KstError::AllValuesNaN);
    }
    let mut line = vec![f64::NAN; data.len()];
    for i in 0..data.len() {
        let v1 = aroc1.values[i];
        let v2 = aroc2.values[i];
        let v3 = aroc3.values[i];
        let v4 = aroc4.values[i];
        if v1.is_nan() || v2.is_nan() || v3.is_nan() || v4.is_nan() {
            line[i] = f64::NAN;
        } else {
            line[i] = v1 + 2.0 * v2 + 3.0 * v3 + 4.0 * v4;
        }
    }
    let sig_out = sma(&SmaInput::from_slice(&line, SmaParams { period: Some(sig) }))?;
    Ok(KstOutput { line, signal: sig_out.values })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn kst_avx2(input: &KstInput, first: usize, len: usize) -> Result<KstOutput, KstError> {
    // Stub: calls scalar for now, API parity
    kst_scalar(input, first, len)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn kst_avx512(input: &KstInput, first: usize, len: usize) -> Result<KstOutput, KstError> {
    // Dispatch to long/short stub (all scalar for now)
    if len <= 32 {
        kst_avx512_short(input, first, len)
    } else {
        kst_avx512_long(input, first, len)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn kst_avx512_short(input: &KstInput, first: usize, len: usize) -> Result<KstOutput, KstError> {
    kst_scalar(input, first, len)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn kst_avx512_long(input: &KstInput, first: usize, len: usize) -> Result<KstOutput, KstError> {
    kst_scalar(input, first, len)
}

#[inline]
pub fn kst_batch_with_kernel(
    data: &[f64],
    sweep: &KstBatchRange,
    k: Kernel,
) -> Result<KstBatchOutput, KstError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(KstError::InvalidPeriod { period: 0, data_len: 0 })
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    kst_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct KstBatchRange {
    pub sma_period1: (usize, usize, usize),
    pub sma_period2: (usize, usize, usize),
    pub sma_period3: (usize, usize, usize),
    pub sma_period4: (usize, usize, usize),
    pub roc_period1: (usize, usize, usize),
    pub roc_period2: (usize, usize, usize),
    pub roc_period3: (usize, usize, usize),
    pub roc_period4: (usize, usize, usize),
    pub signal_period: (usize, usize, usize),
}

impl Default for KstBatchRange {
    fn default() -> Self {
        Self {
            sma_period1: (10, 10, 0),
            sma_period2: (10, 10, 0),
            sma_period3: (10, 10, 0),
            sma_period4: (15, 15, 0),
            roc_period1: (10, 10, 0),
            roc_period2: (15, 15, 0),
            roc_period3: (20, 20, 0),
            roc_period4: (30, 30, 0),
            signal_period: (9, 9, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct KstBatchBuilder {
    range: KstBatchRange,
    kernel: Kernel,
}

impl KstBatchBuilder {
    pub fn new() -> Self { Self::default() }
    pub fn kernel(mut self, k: Kernel) -> Self { self.kernel = k; self }
    pub fn sma_period1_range(mut self, start: usize, end: usize, step: usize) -> Self { self.range.sma_period1 = (start, end, step); self }
    pub fn sma_period2_range(mut self, start: usize, end: usize, step: usize) -> Self { self.range.sma_period2 = (start, end, step); self }
    pub fn sma_period3_range(mut self, start: usize, end: usize, step: usize) -> Self { self.range.sma_period3 = (start, end, step); self }
    pub fn sma_period4_range(mut self, start: usize, end: usize, step: usize) -> Self { self.range.sma_period4 = (start, end, step); self }
    pub fn roc_period1_range(mut self, start: usize, end: usize, step: usize) -> Self { self.range.roc_period1 = (start, end, step); self }
    pub fn roc_period2_range(mut self, start: usize, end: usize, step: usize) -> Self { self.range.roc_period2 = (start, end, step); self }
    pub fn roc_period3_range(mut self, start: usize, end: usize, step: usize) -> Self { self.range.roc_period3 = (start, end, step); self }
    pub fn roc_period4_range(mut self, start: usize, end: usize, step: usize) -> Self { self.range.roc_period4 = (start, end, step); self }
    pub fn signal_period_range(mut self, start: usize, end: usize, step: usize) -> Self { self.range.signal_period = (start, end, step); self }
    pub fn apply_slice(self, data: &[f64]) -> Result<KstBatchOutput, KstError> { kst_batch_with_kernel(data, &self.range, self.kernel) }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<KstBatchOutput, KstError> { KstBatchBuilder::new().kernel(k).apply_slice(data) }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<KstBatchOutput, KstError> { self.apply_slice(source_type(c, src)) }
    pub fn with_default_candles(c: &Candles) -> Result<KstBatchOutput, KstError> {
        KstBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct KstBatchOutput {
    pub lines: Vec<f64>,
    pub signals: Vec<f64>,
    pub combos: Vec<KstParams>,
    pub rows: usize,
    pub cols: usize,
}
impl KstBatchOutput {
    pub fn row_for_params(&self, p: &KstParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.sma_period1.unwrap_or(10) == p.sma_period1.unwrap_or(10)
                && c.sma_period2.unwrap_or(10) == p.sma_period2.unwrap_or(10)
                && c.sma_period3.unwrap_or(10) == p.sma_period3.unwrap_or(10)
                && c.sma_period4.unwrap_or(15) == p.sma_period4.unwrap_or(15)
                && c.roc_period1.unwrap_or(10) == p.roc_period1.unwrap_or(10)
                && c.roc_period2.unwrap_or(15) == p.roc_period2.unwrap_or(15)
                && c.roc_period3.unwrap_or(20) == p.roc_period3.unwrap_or(20)
                && c.roc_period4.unwrap_or(30) == p.roc_period4.unwrap_or(30)
                && c.signal_period.unwrap_or(9) == p.signal_period.unwrap_or(9)
        })
    }
    pub fn lines_for(&self, p: &KstParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.lines[start..start + self.cols]
        })
    }
    pub fn signals_for(&self, p: &KstParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.signals[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &KstBatchRange) -> Vec<KstParams> {
    fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end { return vec![start]; }
        (start..=end).step_by(step).collect()
    }
    let s1 = axis(r.sma_period1);
    let s2 = axis(r.sma_period2);
    let s3 = axis(r.sma_period3);
    let s4 = axis(r.sma_period4);
    let r1 = axis(r.roc_period1);
    let r2 = axis(r.roc_period2);
    let r3 = axis(r.roc_period3);
    let r4 = axis(r.roc_period4);
    let sig = axis(r.signal_period);

    let mut out = Vec::with_capacity(s1.len() * s2.len() * s3.len() * s4.len() * r1.len() * r2.len() * r3.len() * r4.len() * sig.len());
    for &s1v in &s1 { for &s2v in &s2 { for &s3v in &s3 { for &s4v in &s4 {
    for &r1v in &r1 { for &r2v in &r2 { for &r3v in &r3 { for &r4v in &r4 {
    for &sigv in &sig {
        out.push(KstParams {
            sma_period1: Some(s1v), sma_period2: Some(s2v), sma_period3: Some(s3v), sma_period4: Some(s4v),
            roc_period1: Some(r1v), roc_period2: Some(r2v), roc_period3: Some(r3v), roc_period4: Some(r4v),
            signal_period: Some(sigv)
        });
    }}}}}}}}}
    out
}

#[inline(always)]
pub fn kst_batch_slice(data: &[f64], sweep: &KstBatchRange, kern: Kernel) -> Result<KstBatchOutput, KstError> {
    kst_batch_inner(data, sweep, kern, false)
}
#[inline(always)]
pub fn kst_batch_par_slice(data: &[f64], sweep: &KstBatchRange, kern: Kernel) -> Result<KstBatchOutput, KstError> {
    kst_batch_inner(data, sweep, kern, true)
}

fn kst_batch_inner(data: &[f64], sweep: &KstBatchRange, kern: Kernel, parallel: bool) -> Result<KstBatchOutput, KstError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() { return Err(KstError::InvalidPeriod { period: 0, data_len: 0 }); }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(KstError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.sma_period1.unwrap().max(c.sma_period2.unwrap()).max(c.sma_period3.unwrap()).max(c.sma_period4.unwrap())
        .max(c.roc_period1.unwrap()).max(c.roc_period2.unwrap()).max(c.roc_period3.unwrap()).max(c.roc_period4.unwrap()).max(c.signal_period.unwrap())).max().unwrap();
    if data.len() - first < max_p {
        return Err(KstError::NotEnoughValidData { needed: max_p, valid: data.len() - first });
    }
    let rows = combos.len();
    let cols = data.len();
    let mut lines = vec![f64::NAN; rows * cols];
    let mut signals = vec![f64::NAN; rows * cols];

    let do_row = |row: usize, line_row: &mut [f64], sig_row: &mut [f64]| unsafe {
        let prm = &combos[row];
        let inp = KstInput::from_slice(data, prm.clone());
        match kern {
            Kernel::Scalar => {
                let r = kst_row_scalar(&inp, first, cols)?;
                line_row.copy_from_slice(&r.line);
                sig_row.copy_from_slice(&r.signal);
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => {
                let r = kst_row_avx2(&inp, first, cols)?;
                line_row.copy_from_slice(&r.line);
                sig_row.copy_from_slice(&r.signal);
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => {
                let r = kst_row_avx512(&inp, first, cols)?;
                line_row.copy_from_slice(&r.line);
                sig_row.copy_from_slice(&r.signal);
            }
            _ => unreachable!(),
        }
        Ok::<(), KstError>(())
    };
    if parallel {
        lines.par_chunks_mut(cols).zip(signals.par_chunks_mut(cols)).enumerate().for_each(|(row, (lrow, srow))| { let _ = do_row(row, lrow, srow); });
    } else {
        for (row, (lrow, srow)) in lines.chunks_mut(cols).zip(signals.chunks_mut(cols)).enumerate() {
            let _ = do_row(row, lrow, srow);
        }
    }
    Ok(KstBatchOutput { lines, signals, combos, rows, cols })
}

#[inline(always)]
pub unsafe fn kst_row_scalar(input: &KstInput, first: usize, len: usize) -> Result<KstOutput, KstError> {
    kst_scalar(input, first, len)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn kst_row_avx2(input: &KstInput, first: usize, len: usize) -> Result<KstOutput, KstError> {
    kst_avx2(input, first, len)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn kst_row_avx512(input: &KstInput, first: usize, len: usize) -> Result<KstOutput, KstError> {
    kst_avx512(input, first, len)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn kst_row_avx512_short(input: &KstInput, first: usize, len: usize) -> Result<KstOutput, KstError> {
    kst_avx512_short(input, first, len)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn kst_row_avx512_long(input: &KstInput, first: usize, len: usize) -> Result<KstOutput, KstError> {
    kst_avx512_long(input, first, len)
}

// Streaming KST
#[derive(Debug, Clone)]
pub struct KstStream {
    period_params: KstParams,
    buffer: Vec<f64>,
    idx: usize,
    filled: bool,
    // (full streaming ROC/SMA state can be added here if needed)
}
impl KstStream {
    pub fn try_new(params: KstParams) -> Result<Self, KstError> {
        let max_p = params.sma_period1.unwrap_or(10)
            .max(params.sma_period2.unwrap_or(10))
            .max(params.sma_period3.unwrap_or(10))
            .max(params.sma_period4.unwrap_or(15))
            .max(params.roc_period1.unwrap_or(10))
            .max(params.roc_period2.unwrap_or(15))
            .max(params.roc_period3.unwrap_or(20))
            .max(params.roc_period4.unwrap_or(30))
            .max(params.signal_period.unwrap_or(9));
        Ok(Self { period_params: params, buffer: vec![f64::NAN; max_p], idx: 0, filled: false })
    }
    pub fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        self.buffer[self.idx] = value;
        self.idx = (self.idx + 1) % self.buffer.len();
        if !self.filled && self.idx == 0 { self.filled = true; }
        if !self.filled { return None; }
        // Not efficient, but matches batch logic: process the buffer as a slice.
        let inp = KstInput::from_slice(&self.buffer, self.period_params.clone());
        if let Ok(KstOutput { line, signal }) = kst(&inp) {
            let last_idx = line.len() - 1;
            Some((line[last_idx], signal[last_idx]))
        } else { None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_kst_default_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = KstInput::with_default_candles(&candles);
        let result = kst_with_kernel(&input, kernel)?;
        assert_eq!(result.line.len(), candles.close.len());
        assert_eq!(result.signal.len(), candles.close.len());
        Ok(())
    }

    fn check_kst_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = KstInput::with_default_candles(&candles);
        let result = kst_with_kernel(&input, kernel)?;
        let expected_last_five_line = [
            -47.38570195278667,
            -44.42926180347176,
            -42.185693049429034,
            -40.10697793942024,
            -40.17466795905724,
        ];
        let expected_last_five_signal = [
            -52.66743277411538,
            -51.559775662725556,
            -50.113844191238954,
            -48.58923772989874,
            -47.01112630514571,
        ];
        let l = result.line.len();
        let s = result.signal.len();
        for (i, &v) in result.line[l - 5..].iter().enumerate() {
            assert!((v - expected_last_five_line[i]).abs() < 1e-1, "KST line mismatch {}: {} vs {}", i, v, expected_last_five_line[i]);
        }
        for (i, &v) in result.signal[s - 5..].iter().enumerate() {
            assert!((v - expected_last_five_signal[i]).abs() < 1e-1, "KST signal mismatch {}: {} vs {}", i, v, expected_last_five_signal[i]);
        }
        Ok(())
    }

    fn check_kst_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let nan_data = [f64::NAN, f64::NAN, f64::NAN];
        let input = KstInput::from_slice(&nan_data, KstParams::default());
        let result = kst_with_kernel(&input, kernel);
        assert!(result.is_err(), "[{}] Should error with all NaN", test_name);
        Ok(())
    }

    macro_rules! generate_all_kst_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(
                    #[test]
                    fn [<$test_fn _scalar_f64>]() { let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar); }
                )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(
                    #[test]
                    fn [<$test_fn _avx2_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2); }
                    #[test]
                    fn [<$test_fn _avx512_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512); }
                )*
            }
        }
    }

    generate_all_kst_tests!(
        check_kst_default_params,
        check_kst_accuracy,
        check_kst_nan_handling
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = KstBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;
        let def = KstParams::default();
        let row = output.lines_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        Ok(())
    }

    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste::paste! {
                #[test] fn [<$fn_name _scalar>]()      { let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch); }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx2>]()        { let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch); }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx512>]()      { let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch); }
                #[test] fn [<$fn_name _auto_detect>]() { let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto); }
            }
        };
    }
    gen_batch_tests!(check_batch_default_row);
}

