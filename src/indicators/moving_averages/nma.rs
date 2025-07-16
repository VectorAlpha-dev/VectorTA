//! # Normalized Moving Average (NMA)
//!
//! A technique that computes an adaptive moving average by transforming input
//! values into log space and weighting differences between consecutive values.
//! The weighting ratio depends on a series of square-root increments. This design
//! aims to normalize large price changes without oversmoothing small fluctuations.
//!
//! ## Parameters
//! - **period**: Window size (number of data points, default: 40)
//!
//! ## Errors
//! - **AllValuesNaN**: nma: All input data values are `NaN`.
//! - **InvalidPeriod**: nma: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: nma: Not enough valid data points for the requested `period`.
//!
//! ## Returns
//! - **`Ok(NmaOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(NmaError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

impl<'a> AsRef<[f64]> for NmaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            NmaData::Slice(slice) => slice,
            NmaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum NmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct NmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
pub struct NmaParams {
    pub period: Option<usize>,
}

impl Default for NmaParams {
    fn default() -> Self {
        Self { period: Some(40) }
    }
}

#[derive(Debug, Clone)]
pub struct NmaInput<'a> {
    pub data: NmaData<'a>,
    pub params: NmaParams,
}

impl<'a> NmaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: NmaParams) -> Self {
        Self {
            data: NmaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: NmaParams) -> Self {
        Self {
            data: NmaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", NmaParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(40)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct NmaBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for NmaBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl NmaBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<NmaOutput, NmaError> {
        let p = NmaParams {
            period: self.period,
        };
        let i = NmaInput::from_candles(c, "close", p);
        nma_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<NmaOutput, NmaError> {
        let p = NmaParams {
            period: self.period,
        };
        let i = NmaInput::from_slice(d, p);
        nma_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<NmaStream, NmaError> {
        let p = NmaParams {
            period: self.period,
        };
        NmaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum NmaError {
    #[error("nma: All values are NaN.")]
    AllValuesNaN,
    #[error("nma: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("nma: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn nma(input: &NmaInput) -> Result<NmaOutput, NmaError> {
    nma_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn nma_prepare<'a>(
    input: &'a NmaInput,
    kernel: Kernel,
) -> Result<
    (
        /*data*/ &'a [f64],
        /*period*/ usize,
        /*first*/ usize,
        /*ln_values*/ Vec<f64>,
        /*sqrt_diffs*/ Vec<f64>,
        /*chosen*/ Kernel,
    ),
    NmaError,
> {
    let data: &[f64] = input.as_ref();
    let len = data.len();

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(NmaError::AllValuesNaN)?;

    let period = input.get_period();

    if period == 0 || period > len {
        return Err(NmaError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < (period + 1) {
        return Err(NmaError::NotEnoughValidData {
            needed: period + 1,
            valid: len - first,
        });
    }

    // Pre-compute ln values
    let mut ln_values = Vec::with_capacity(len);
    ln_values.extend(data.iter().map(|&val| {
        let clamped = val.max(1e-10);
        clamped.ln() * 1000.0
    }));

    // Pre-compute sqrt differences
    let mut sqrt_diffs = Vec::with_capacity(period);
    for i in 0..period {
        let s0 = (i as f64).sqrt();
        let s1 = ((i + 1) as f64).sqrt();
        sqrt_diffs.push(s1 - s0);
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    Ok((data, period, first, ln_values, sqrt_diffs, chosen))
}

fn nma_compute_into(
    data: &[f64],
    period: usize,
    first: usize,
    ln_values: &mut [f64],
    sqrt_diffs: &mut [f64],
    kernel: Kernel,
    out: &mut [f64],
) {
    match kernel {
        Kernel::Scalar | Kernel::ScalarBatch => {
            nma_scalar_with_precomputed(data, period, first, ln_values, sqrt_diffs, out)
        }

        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 | Kernel::Avx2Batch => {
            nma_scalar_with_precomputed(data, period, first, ln_values, sqrt_diffs, out)
        }

        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 | Kernel::Avx512Batch => {
            /* ────────────────────────────────
             *  AVX-512 needs |Δ ln| for every row,
             *  so allocate a full-length scratch.
             * ────────────────────────────────*/
            let mut diff = vec![0.0f64; data.len()];

            // note the two *mutable* slices
            unsafe { nma_avx512(data, period, first, ln_values, &mut diff, out) }
        }

        _ => unreachable!(),
    }
}

pub fn nma_with_kernel(input: &NmaInput, kernel: Kernel) -> Result<NmaOutput, NmaError> {
    // ────────────────────▼───────────────────  mark both bindings `mut`
    let (data, period, first, mut ln_values, mut sqrt_diffs, chosen) = nma_prepare(input, kernel)?;

    let warm = first + period;
    let mut out = alloc_with_nan_prefix(data.len(), warm);

    // ───────────────────▼──────────────▼────── pass them mutably
    nma_compute_into(
        data,
        period,
        first,
        &mut ln_values,
        &mut sqrt_diffs,
        chosen,
        &mut out,
    );

    Ok(NmaOutput { values: out })
}
#[inline]
pub fn nma_scalar(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    let len = data.len();

    let mut ln_values = Vec::with_capacity(len);
    ln_values.extend(data.iter().map(|&val| {
        let clamped = val.max(1e-10);
        clamped.ln() * 1000.0
    }));

    let mut sqrt_diffs = Vec::with_capacity(period);
    for i in 0..period {
        let s0 = (i as f64).sqrt();
        let s1 = ((i + 1) as f64).sqrt();
        sqrt_diffs.push(s1 - s0);
    }

    for j in (first + period)..len {
        let mut num = 0.0;
        let mut denom = 0.0;

        for i in 0..period {
            let oi = (ln_values[j - i] - ln_values[j - i - 1]).abs();
            num += oi * sqrt_diffs[i];
            denom += oi;
        }

        let ratio = if denom == 0.0 { 0.0 } else { num / denom };

        let i = period - 1;
        out[j] = data[j - i] * ratio + data[j - i - 1] * (1.0 - ratio);
    }
}

#[inline]
pub fn nma_scalar_with_precomputed(
    data: &[f64],
    period: usize,
    first: usize,
    ln_values: &[f64],
    sqrt_diffs: &[f64],
    out: &mut [f64],
) {
    let len = data.len();

    for j in (first + period)..len {
        let mut num = 0.0;
        let mut denom = 0.0;

        for i in 0..period {
            let oi = (ln_values[j - i] - ln_values[j - i - 1]).abs();
            num += oi * sqrt_diffs[i];
            denom += oi;
        }

        let ratio = if denom == 0.0 { 0.0 } else { num / denom };

        let i = period - 1;
        out[j] = data[j - i] * ratio + data[j - i - 1] * (1.0 - ratio);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512vl,avx512bw,fma")]
pub unsafe fn nma_avx512(
    data: &[f64],
    period: usize,
    first: usize,
    ln: &mut [f64],
    diff: &mut [f64],
    out: &mut [f64],
) {
    let n = data.len();
    debug_assert!(ln.len() == n && diff.len() == n && out.len() == n);
    debug_assert!(first + period <= n); // validated by caller

    /* --------------------------------------------------------
     * 1. ln(x) * 1 000  (x clamped to 1 e-10)
     * ------------------------------------------------------*/
    let eps = _mm512_set1_pd(1.0e-10);
    let scale = _mm512_set1_pd(1_000.0);

    let mut i = 0usize;
    while i + 8 <= n {
        let mut v = _mm512_loadu_pd(data.as_ptr().add(i));
        v = _mm512_max_pd(v, eps); // clamp negatives / zeros
        #[cfg(feature = "sleef")]
        {
            v = Sleef_logd8_u10(v); // vectorised ln
        }
        #[cfg(not(feature = "sleef"))]
        {
            let mut tmp = [0.0f64; 8];
            _mm512_storeu_pd(tmp.as_mut_ptr(), v);
            for x in &mut tmp {
                *x = x.ln();
            }
            v = _mm512_loadu_pd(tmp.as_ptr());
        }
        v = _mm512_mul_pd(v, scale);
        _mm512_storeu_pd(ln.as_mut_ptr().add(i), v);
        i += 8;
    }
    for k in i..n {
        ln[k] = data[k].max(1.0e-10).ln() * 1_000.0;
    }

    /* --------------------------------------------------------
     * 2. |Δ ln|
     * ------------------------------------------------------*/
    diff[0] = 0.0;
    for k in 1..n {
        diff[k] = (ln[k] - ln[k - 1]).abs();
    }

    /* --------------------------------------------------------
     * 3. Pre-compute √-weights, 8-packed and **reversed**
     *    (lane 0 corresponds to diff[j-0], lane 7 to diff[j-7])
     * ------------------------------------------------------*/
    let blocks = (period + 7) / 8;
    let mut wv: Vec<__m512d> = Vec::with_capacity(blocks);

    for base in (0..period).step_by(8) {
        let mut buf = [0.0_f64; 8];
        for j in 0..8 {
            let k = base + j;
            if k < period {
                buf[7 - j] = ((k as f64 + 1.0).sqrt() - (k as f64).sqrt());
            }
        }
        wv.push(_mm512_loadu_pd(buf.as_ptr()));
    }

    /* --------------------------------------------------------
     * 4. Initial denominator (= Σ|Δln| over the first window)
     * ------------------------------------------------------*/
    let mut denom: f64 = diff[first..first + period].iter().sum();

    /* --------------------------------------------------------
     * 5. Main loop – dot-product, blend, slide window
     * ------------------------------------------------------*/
    for j in (first + period)..n {
        // 5.1 Dot-product of the last `period` diffs with weights
        let mut acc = _mm512_setzero_pd();
        for (blk, &w) in wv.iter().enumerate() {
            // Load diff[j-0 … j-7], then diff[j-8 … j-15], …
            let d = _mm512_loadu_pd(diff.as_ptr().add(j - blk * 8));
            acc = _mm512_fmadd_pd(d, w, acc); // fused multiply-add
        }

        // 5.2 Horizontal reduce to scalar
        let num = _mm512_reduce_add_pd(acc);

        // 5.3 Blend the two price points using num/denom
        let ratio = if denom == 0.0 { 0.0 } else { num / denom };
        let tail = period - 1; // newest value’s look-back
        out[j] = data[j - tail] * ratio + data[j - tail - 1] * (1.0 - ratio);

        // 5.4 Slide denominator window
        denom += diff[j] - diff[j - period];
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[cfg(target_feature = "avx2")]
pub fn nma_avx2(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    unsafe { nma_scalar(data, period, first, out) }
}

#[inline(always)]
pub fn nma_batch_with_kernel(
    data: &[f64],
    sweep: &NmaBatchRange,
    k: Kernel,
) -> Result<NmaBatchOutput, NmaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(NmaError::InvalidPeriod {
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
    nma_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct NmaBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for NmaBatchRange {
    fn default() -> Self {
        Self {
            period: (40, 100, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct NmaBatchBuilder {
    range: NmaBatchRange,
    kernel: Kernel,
}

impl NmaBatchBuilder {
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

    pub fn apply_slice(self, data: &[f64]) -> Result<NmaBatchOutput, NmaError> {
        nma_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<NmaBatchOutput, NmaError> {
        NmaBatchBuilder::new().kernel(k).apply_slice(data)
    }

    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<NmaBatchOutput, NmaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }

    pub fn with_default_candles(c: &Candles) -> Result<NmaBatchOutput, NmaError> {
        NmaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct NmaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<NmaParams>,
    pub rows: usize,
    pub cols: usize,
}

impl NmaBatchOutput {
    pub fn row_for_params(&self, p: &NmaParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(40) == p.period.unwrap_or(40))
    }

    pub fn values_for(&self, p: &NmaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &NmaBatchRange) -> Vec<NmaParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);

    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(NmaParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn nma_batch_slice(
    data: &[f64],
    sweep: &NmaBatchRange,
    kern: Kernel,
) -> Result<NmaBatchOutput, NmaError> {
    nma_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn nma_batch_par_slice(
    data: &[f64],
    sweep: &NmaBatchRange,
    kern: Kernel,
) -> Result<NmaBatchOutput, NmaError> {
    nma_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn nma_batch_inner(
    data: &[f64],
    sweep: &NmaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<NmaBatchOutput, NmaError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(NmaError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(NmaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p + 1 {
        return Err(NmaError::NotEnoughValidData {
            needed: max_p + 1,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();

    let warm: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap()).collect();

    let mut raw = make_uninit_matrix(rows, cols);
    unsafe { init_matrix_prefixes(&mut raw, cols, &warm) };

    // ---------------------------------------------------------------------
    // 2. closure that fills one row (works with MaybeUninit<f64>)
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();

        // cast just this row to &mut [f64] so we can call the usual kernel
        let out_row =
            core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        match kern {
            Kernel::Scalar => nma_row_scalar(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => nma_row_avx2(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => nma_row_avx512(data, first, period, out_row),
            _ => unreachable!(),
        }
    };

    // ---------------------------------------------------------------------
    // 3. run every row, writing directly into `raw`
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

    // ---------------------------------------------------------------------
    // 4. everything is now initialised – transmute to Vec<f64>
    let values: Vec<f64> = unsafe { std::mem::transmute(raw) };

    Ok(NmaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn nma_batch_inner_into(
    data: &[f64],
    sweep: &NmaBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<NmaParams>, NmaError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(NmaError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(NmaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p + 1 {
        return Err(NmaError::NotEnoughValidData {
            needed: max_p + 1,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();

    let warm: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap()).collect();

    // SAFETY: We're reinterpreting the output slice as MaybeUninit
    let out_uninit = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };

    unsafe { init_matrix_prefixes(out_uninit, cols, &warm) };

    // Closure that writes ONE row
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();

        // Cast this row to &mut [f64]
        let out_row =
            core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        match kern {
            Kernel::Scalar => nma_row_scalar(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => nma_row_avx2(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => nma_row_avx512(data, first, period, out_row),
            _ => unreachable!(),
        }
    };

    // Drive the whole matrix
    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_uninit
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in out_uninit.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in out_uninit.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    Ok(combos)
}

#[inline(always)]
unsafe fn nma_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    nma_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn nma_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    nma_row_scalar(data, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn nma_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    if period <= 32 {
        nma_row_avx512_short(data, first, period, out);
    } else {
        nma_row_avx512_long(data, first, period, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn nma_row_avx512_short(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    nma_row_scalar(data, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn nma_row_avx512_long(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    nma_row_scalar(data, first, period, out)
}

#[derive(Debug, Clone)]
pub struct NmaStream {
    period: usize,
    ln_values: Vec<f64>,
    sqrt_diffs: Vec<f64>,
    buffer: Vec<f64>,
    ln_buffer: Vec<f64>,
    head: usize,
    filled: bool,
}

impl NmaStream {
    pub fn try_new(params: NmaParams) -> Result<Self, NmaError> {
        let period = params.period.unwrap_or(40);
        if period == 0 {
            return Err(NmaError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let mut sqrt_diffs = Vec::with_capacity(period);
        for i in 0..period {
            let s0 = (i as f64).sqrt();
            let s1 = ((i + 1) as f64).sqrt();
            sqrt_diffs.push(s1 - s0);
        }
        Ok(Self {
            period,
            ln_values: vec![f64::NAN; period + 1],
            sqrt_diffs,
            buffer: vec![f64::NAN; period + 1],
            ln_buffer: vec![f64::NAN; period + 1],
            head: 0,
            filled: false,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        let ln_val = value.max(1e-10).ln() * 1000.0;
        self.buffer[self.head] = value;
        self.ln_buffer[self.head] = ln_val;
        self.head = (self.head + 1) % (self.period + 1);

        if !self.filled && self.head == 0 {
            self.filled = true;
        }
        if !self.filled {
            return None;
        }
        Some(self.dot_ring())
    }

    #[inline(always)]
    fn dot_ring(&self) -> f64 {
        let mut num = 0.0;
        let mut denom = 0.0;

        // Calculate starting position for the newest value
        let newest_idx = (self.head + self.period) % (self.period + 1);

        for i in 0..self.period {
            // Access in reverse order like batch: newest to oldest
            let curr_idx = (newest_idx + self.period + 1 - i) % (self.period + 1);
            let prev_idx = (newest_idx + self.period - i) % (self.period + 1);

            let curr = self.ln_buffer[curr_idx];
            let prev = self.ln_buffer[prev_idx];
            let oi = (curr - prev).abs();

            num += oi * self.sqrt_diffs[i];
            denom += oi;
        }

        let ratio = if denom == 0.0 { 0.0 } else { num / denom };

        // Get the values for final interpolation
        let i = self.period - 1;
        let x1_idx = (newest_idx + self.period + 1 - i) % (self.period + 1);
        let x2_idx = (newest_idx + self.period - i) % (self.period + 1);

        let x1 = self.buffer[x1_idx];
        let x2 = self.buffer[x2_idx];

        x1 * ratio + x2 * (1.0 - ratio)
    }
}

// Expand grid for batch

// Python bindings
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

#[cfg(feature = "python")]
#[pyfunction(name = "nma")]
pub fn nma_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let slice_in = data.as_slice()?;
    let params = NmaParams {
        period: Some(period),
    };
    let nma_in = NmaInput::from_slice(slice_in, params);

    // Allocate NumPy output buffer
    let out_arr = unsafe { PyArray1::<f64>::new(py, [slice_in.len()], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    // Heavy lifting without the GIL
    py.allow_threads(|| -> Result<(), NmaError> {
        let (data, period, first, mut ln_values, mut sqrt_diffs, chosen) =
            nma_prepare(&nma_in, Kernel::Auto)?;
        // Initialize prefix with NaN
        let warm = first + period;
        slice_out[..warm].fill(f64::NAN);
        nma_compute_into(
            data,
            period,
            first,
            &mut ln_values,
            &mut sqrt_diffs,
            chosen,
            slice_out,
        );
        Ok(())
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(out_arr)
}

#[cfg(feature = "python")]
#[pyclass(name = "NmaStream")]
pub struct NmaStreamPy {
    stream: NmaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl NmaStreamPy {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        let params = NmaParams {
            period: Some(period),
        };
        let stream =
            NmaStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NmaStreamPy { stream })
    }

    /// Updates the stream with a new value and returns the calculated NMA value.
    /// Returns `None` if the buffer is not yet full.
    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "nma_batch")]
pub fn nma_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
) -> PyResult<Bound<'py, PyDict>> {
    let slice_in = data.as_slice()?;
    let sweep = NmaBatchRange {
        period: period_range,
    };

    // Expand grid to know rows*cols
    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();

    // Pre-allocate NumPy array
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    // Heavy work without the GIL
    let combos = py
        .allow_threads(|| {
            let kernel = match Kernel::Auto {
                Kernel::Auto => detect_best_batch_kernel(),
                k => k,
            };
            let simd = match kernel {
                Kernel::Avx512Batch => Kernel::Avx512,
                Kernel::Avx2Batch => Kernel::Avx2,
                Kernel::ScalarBatch => Kernel::Scalar,
                _ => unreachable!(),
            };
            // Use the _into variant
            nma_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Build dict with the GIL
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

// WASM bindings
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn nma_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = NmaParams {
        period: Some(period),
    };
    let input = NmaInput::from_slice(data, params);

    nma_with_kernel(&input, Kernel::Scalar)
        .map(|o| o.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn nma_batch_js(
    data: &[f64],
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = NmaBatchRange {
        period: (period_start, period_end, period_step),
    };

    // Use the existing batch function with parallel=false for WASM
    nma_batch_inner(data, &sweep, Kernel::Scalar, false)
        .map(|output| output.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn nma_batch_metadata_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = NmaBatchRange {
        period: (period_start, period_end, period_step),
    };

    let combos = expand_grid(&sweep);
    let metadata: Vec<f64> = combos
        .iter()
        .map(|combo| combo.period.unwrap() as f64)
        .collect();

    Ok(metadata)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn nma_batch_rows_cols_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
    data_len: usize,
) -> Vec<usize> {
    let sweep = NmaBatchRange {
        period: (period_start, period_end, period_step),
    };
    let combos = expand_grid(&sweep);
    vec![combos.len(), data_len]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_nma_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = NmaParams { period: None };
        let input = NmaInput::from_candles(&candles, "close", default_params);
        let output = nma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_nma_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = NmaInput::from_candles(&candles, "close", NmaParams::default());
        let nma_result = nma_with_kernel(&input, kernel)?;

        let expected_last_five_nma = [
            64320.486018271724,
            64227.95719984426,
            64180.9249333126,
            63966.35530620797,
            64039.04719192334,
        ];
        let start_index = nma_result.values.len() - 5;
        let result_last_five_nma = &nma_result.values[start_index..];
        for (i, &value) in result_last_five_nma.iter().enumerate() {
            let expected_value = expected_last_five_nma[i];
            assert!(
                (value - expected_value).abs() < 1e-3,
                "[{}] NMA value mismatch at last-5 index {}: expected {}, got {}",
                test_name,
                i,
                expected_value,
                value
            );
        }
        Ok(())
    }

    fn check_nma_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = NmaInput::with_default_candles(&candles);
        match input.data {
            NmaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected NmaData::Candles"),
        }
        let output = nma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_nma_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = NmaParams { period: Some(0) };
        let input = NmaInput::from_slice(&input_data, params);
        let res = nma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] NMA should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_nma_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = NmaParams { period: Some(10) };
        let input = NmaInput::from_slice(&data_small, params);
        let res = nma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] NMA should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_nma_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = NmaParams { period: Some(40) };
        let input = NmaInput::from_slice(&single_point, params);
        let res = nma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] NMA should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_nma_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = NmaParams { period: Some(40) };
        let first_input = NmaInput::from_candles(&candles, "close", first_params);
        let first_result = nma_with_kernel(&first_input, kernel)?;
        let second_params = NmaParams { period: Some(20) };
        let second_input = NmaInput::from_slice(&first_result.values, second_params);
        let second_result = nma_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        if second_result.values.len() > 240 {
            for i in 240..second_result.values.len() {
                assert!(second_result.values[i].is_finite());
            }
        }
        Ok(())
    }

    fn check_nma_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = NmaInput::from_candles(&candles, "close", NmaParams { period: Some(40) });
        let res = nma_with_kernel(&input, kernel)?;
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

    macro_rules! generate_all_nma_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(#[test]
                fn [<$test_fn _scalar_f64>]() {
                    let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar);
                })*
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
    fn check_nma_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test with multiple parameter combinations
        let test_cases = vec![
            NmaParams { period: Some(40) },  // default
            NmaParams { period: Some(10) },  // small period
            NmaParams { period: Some(5) },   // very small period
            NmaParams { period: Some(20) },  // medium period
            NmaParams { period: Some(60) },  // larger period
            NmaParams { period: Some(100) }, // large period
            NmaParams { period: Some(3) },   // minimum practical period
            NmaParams { period: Some(80) },  // another large period
            NmaParams { period: None },      // None value (use default)
        ];

        for params in test_cases {
            let input = NmaInput::from_candles(&candles, "close", params);
            let output = nma_with_kernel(&input, kernel)?;

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
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
                         with params period={:?}",
                        test_name, val, bits, i, params.period
                    );
                }

                // Check for init_matrix_prefixes poison (0x22222222_22222222)
                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
                         with params period={:?}",
                        test_name, val, bits, i, params.period
                    );
                }

                // Check for make_uninit_matrix poison (0x33333333_33333333)
                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
                         with params period={:?}",
                        test_name, val, bits, i, params.period
                    );
                }
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_nma_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    generate_all_nma_tests!(
        check_nma_partial_params,
        check_nma_accuracy,
        check_nma_default_candles,
        check_nma_zero_period,
        check_nma_period_exceeds_length,
        check_nma_very_small_dataset,
        check_nma_reinput,
        check_nma_nan_handling,
        check_nma_no_poison
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = NmaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = NmaParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            64320.486018271724,
            64227.95719984426,
            64180.9249333126,
            63966.35530620797,
            64039.04719192334,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-3,
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

        // Test multiple batch configurations with different parameter ranges
        let batch_configs = vec![
            // Original test case
            (10, 30, 10),
            // Edge cases
            (40, 40, 0),   // Single parameter (default)
            (3, 15, 3),    // Small periods
            (50, 100, 25), // Large periods
            (5, 25, 5),    // Different step
            (20, 80, 20),  // Medium to large
            (8, 24, 8),    // Different small range
            (60, 120, 30), // Very large periods
        ];

        for (p_start, p_end, p_step) in batch_configs {
            let output = NmaBatchBuilder::new()
                .kernel(kernel)
                .period_range(p_start, p_end, p_step)
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
                let combo = &output.combos[row];

                // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} \
                         (flat index {}) with params period={:?}",
                        test, val, bits, row, col, idx, combo.period
                    );
                }

                // Check for init_matrix_prefixes poison (0x22222222_22222222)
                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} \
                         (flat index {}) with params period={:?}",
                        test, val, bits, row, col, idx, combo.period
                    );
                }

                // Check for make_uninit_matrix poison (0x33333333_33333333)
                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} \
                         (flat index {}) with params period={:?}",
                        test, val, bits, row, col, idx, combo.period
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
