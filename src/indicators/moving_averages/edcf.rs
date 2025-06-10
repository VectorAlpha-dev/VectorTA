//! # Ehlers Distance Coefficient Filter (EDCF)
//!
//! John Ehlers' Distance Coefficient Filter (EDCF) uses squared distances between successive points to build a non-linear, volatility-sensitive weighted average. Higher weights are assigned to prices following larger recent price changes, smoothing out trendless noise. Re-applying EDCF to its own output can provide multi-stage smoothing.
//!
//! ## Parameters
//! - **period**: Window size (number of data points). (defaults to 15)
//!
//! ## Errors
//! - **NoData**: edcf: No data provided to EDCF filter.
//! - **AllValuesNaN**: edcf: All input data values are `NaN`.
//! - **InvalidPeriod**: edcf: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: edcf: Not enough valid data points for the requested `period`.
//! - **NaNFound**: edcf: A `NaN` was encountered after the first valid index.
//!
//! ## Returns
//! - **`Ok(EdcfOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(EdcfError)`** otherwise.

use crate::utilities::aligned_vector::AlignedVec;
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel, make_uninit_matrix, init_matrix_prefixes, alloc_with_nan_prefix};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use rayon::prelude::*;
use std::convert::AsRef;
use thiserror::Error;
use std::mem::MaybeUninit;

#[derive(Debug, Clone)]
pub enum EdcfData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

impl<'a> AsRef<[f64]> for EdcfInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            EdcfData::Slice(slice) => slice,
            EdcfData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EdcfParams {
    pub period: Option<usize>,
}

impl Default for EdcfParams {
    fn default() -> Self {
        Self { period: Some(15) }
    }
}

#[derive(Debug, Clone)]
pub struct EdcfOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct EdcfInput<'a> {
    pub data: EdcfData<'a>,
    pub params: EdcfParams,
}

impl<'a> EdcfInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: EdcfParams) -> Self {
        Self {
            data: EdcfData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: EdcfParams) -> Self {
        Self {
            data: EdcfData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", EdcfParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(15)
    }
}

#[derive(Debug, Error)]
pub enum EdcfError {
    #[error("edcf: No data provided to EDCF filter.")]
    NoData,
    #[error("edcf: All values are NaN.")]
    AllValuesNaN,
    #[error("edcf: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("edcf: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("edcf: NaN found in data after the first valid index.")]
    NaNFound,
}

#[derive(Copy, Clone, Debug)]
pub struct EdcfBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for EdcfBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl EdcfBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<EdcfOutput, EdcfError> {
        let p = EdcfParams {
            period: self.period,
        };
        let i = EdcfInput::from_candles(c, "close", p);
        edcf_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<EdcfOutput, EdcfError> {
        let p = EdcfParams {
            period: self.period,
        };
        let i = EdcfInput::from_slice(d, p);
        edcf_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<EdcfStream, EdcfError> {
        let p = EdcfParams {
            period: self.period,
        };
        EdcfStream::try_new(p)
    }
}

#[inline]
pub fn edcf(input: &EdcfInput) -> Result<EdcfOutput, EdcfError> {
    edcf_with_kernel(input, Kernel::Auto)
}

pub fn edcf_with_kernel(input: &EdcfInput, kernel: Kernel) -> Result<EdcfOutput, EdcfError> {
    let data: &[f64] = match &input.data {
        EdcfData::Candles { candles, source } => source_type(candles, source),
        EdcfData::Slice(sl) => sl,
    };
    let len = data.len();
    if len == 0 {
        return Err(EdcfError::NoData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(EdcfError::AllValuesNaN)?;
    let period = input.get_period();
    if period == 0 || period > len {
        return Err(EdcfError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    let needed = 2 * period;
    if (len - first) < needed {
        return Err(EdcfError::NotEnoughValidData {
            needed,
            valid: len - first,
        });
    }
    if data[first..].iter().any(|&v| v.is_nan()) {
        return Err(EdcfError::NaNFound);
    }

    let warm = first + 2 * period;
    let mut out = alloc_with_nan_prefix(len, warm);
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => edcf_scalar(data, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => edcf_avx2(data, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => edcf_avx512(data, period, first, &mut out),
            _ => unreachable!(),
        }
    }
    Ok(EdcfOutput { values: out })
}

#[inline(always)]
pub fn edcf_scalar(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    let len = data.len();
    let mut dist: Vec<f64> = vec![0.0; len];

    unsafe {
        let dp = data.as_ptr();
        let wp = dist.as_mut_ptr();

        let dist_start = first_valid + period;
        for k in dist_start..len {
            let xk = *dp.add(k);
            let mut sum_sq = 0.0;
            for lb in 1..period {
                let diff = xk - *dp.add(k - lb);
                sum_sq = diff.mul_add(diff, sum_sq);
            }
            *wp.add(k) = sum_sq;
        }

        let start_j = first_valid + 2 * period;
        for j in start_j..len {
            let mut num = 0.0;
            let mut coef_sum = 0.0;
            for i in 0..period {
                let k = j - i;
                let w = *wp.add(k);
                let v = *dp.add(k);

                num = w.mul_add(v, num);
                coef_sum += w;
            }
            if coef_sum != 0.0 {
                *out.get_unchecked_mut(j) = num / coef_sum;
            }
        }
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn hsum_m256d(v: __m256d) -> f64 {
    let hi = _mm256_extractf128_pd(v, 1);
    let lo = _mm256_castpd256_pd128(v);
    let sum2 = _mm_add_pd(hi, lo);
    let hi64 = _mm_unpackhi_pd(sum2, sum2);
    _mm_cvtsd_f64(_mm_add_sd(sum2, hi64))
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn edcf_avx2(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    const STEP: usize = 4;
    let len = data.len();
    let chunks = period / STEP;
    let tail_len = period % STEP;

    let mut dist: Vec<f64> = vec![0.0; len];
    let dp = data.as_ptr();
    let wp = dist.as_mut_ptr();

    for k in (first_valid + period)..len {
        let xk_vec = _mm256_broadcast_sd(&*dp.add(k));
        let mut acc = _mm256_setzero_pd();

        for blk in 0..chunks {
            let ptr = dp.add(k - (blk + 1) * STEP);
            let d = _mm256_loadu_pd(ptr);
            let diff = _mm256_sub_pd(xk_vec, d);
            acc = _mm256_fmadd_pd(diff, diff, acc);
        }

        let mut sum_tail = 0.0;
        for lb in (chunks * STEP + 1)..period {
            let diff = *dp.add(k) - *dp.add(k - lb);
            sum_tail += diff * diff;
        }

        *wp.add(k) = hsum_m256d(acc) + sum_tail;
    }

    for j in (first_valid + 2 * period)..len {
        let start_k = j + 1 - period;

        let mut num_vec = _mm256_setzero_pd();
        let mut coef_vec = _mm256_setzero_pd();
        for blk in 0..chunks {
            let idx = start_k + blk * STEP;
            let d = _mm256_loadu_pd(dp.add(idx));
            let w = _mm256_loadu_pd(wp.add(idx));
            num_vec = _mm256_fmadd_pd(w, d, num_vec);
            coef_vec = _mm256_add_pd(coef_vec, w);
        }

        let mut num = hsum_m256d(num_vec);
        let mut coef = hsum_m256d(coef_vec);

        for i in (chunks * STEP)..period {
            let k = start_k + i;
            let w = *wp.add(k);
            num += w * *dp.add(k);
            coef += w;
        }

        if coef != 0.0 {
            *out.get_unchecked_mut(j) = num / coef;
        }
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512dq,fma")]
pub unsafe fn edcf_avx512(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    const STEP: usize = 8;

    let len = data.len();
    let p_minus1 = period - 1;
    let chunks = p_minus1 / STEP;
    let tail_len = p_minus1 % STEP;
    let tail_mask: __mmask8 = (1u8 << tail_len).wrapping_sub(1);

    let mut dist: Vec<f64> = vec![0.0; len];
    let dp = data.as_ptr();
    let wp = dist.as_mut_ptr();

    for k in (first_valid + period)..len {
        let xk_vec = _mm512_set1_pd(*dp.add(k));
        let mut acc = _mm512_setzero_pd();

        let start = k - p_minus1;
        for blk in 0..chunks {
            let d = _mm512_loadu_pd(dp.add(start + blk * STEP));
            let diff = _mm512_sub_pd(xk_vec, d);
            acc = _mm512_fmadd_pd(diff, diff, acc);
        }

        if tail_len != 0 {
            let base = dp.add(start + chunks * STEP);
            let d = _mm512_maskz_loadu_pd(tail_mask, base);
            let diff = _mm512_mask_sub_pd(_mm512_setzero_pd(), tail_mask, xk_vec, d);
            let sq = _mm512_mul_pd(diff, diff);
            acc = _mm512_add_pd(acc, sq);
        }

        *wp.add(k) = _mm512_reduce_add_pd(acc);
    }

    for j in (first_valid + 2 * period)..len {
        let start_k = j - p_minus1;

        let mut num_vec = _mm512_setzero_pd();
        let mut coef_vec = _mm512_setzero_pd();

        for blk in 0..chunks {
            let idx = start_k + blk * STEP;
            let d = _mm512_loadu_pd(dp.add(idx));
            let w = _mm512_loadu_pd(wp.add(idx));
            num_vec = _mm512_fmadd_pd(w, d, num_vec);
            coef_vec = _mm512_add_pd(coef_vec, w);
        }

        if tail_len != 0 {
            let idx = start_k + chunks * STEP;
            let d = _mm512_maskz_loadu_pd(tail_mask, dp.add(idx));
            let w = _mm512_maskz_loadu_pd(tail_mask, wp.add(idx));
            num_vec = _mm512_fmadd_pd(w, d, num_vec);
            coef_vec = _mm512_add_pd(coef_vec, w);
        }

        let w0 = *wp.add(j);
        let v0 = *dp.add(j);
        let num = _mm512_reduce_add_pd(num_vec) + w0 * v0;
        let coef = _mm512_reduce_add_pd(coef_vec) + w0;

        if coef != 0.0 {
            *out.get_unchecked_mut(j) = num / coef;
        }
    }
}

#[derive(Debug, Clone)]
pub struct EdcfStream {
    period: usize,
    buffer: Vec<f64>,
    dist: Vec<f64>,
    head: usize,
    filled: bool,
}

impl EdcfStream {
    pub fn try_new(params: EdcfParams) -> Result<Self, EdcfError> {
        let period = params.period.unwrap_or(15);
        if period == 0 {
            return Err(EdcfError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        Ok(Self {
            period,
            buffer: vec![f64::NAN; period],
            dist: vec![0.0; period],
            head: 0,
            filled: false,
        })
    }
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.buffer[self.head] = value;
        self.head = (self.head + 1) % self.period;

        if !self.filled && self.head == 0 {
            self.filled = true;
        }
        if !self.filled {
            return None;
        }

        let mut sum_sq = 0.0;
        let mut xk = value;
        let mut pos = self.head;
        for lb in 1..self.period {
            pos = (pos + self.period - 1) % self.period;
            let diff = xk - self.buffer[pos];
            sum_sq += diff * diff;
        }
        self.dist[self.head] = sum_sq;

        let mut num = 0.0;
        let mut coef_sum = 0.0;
        pos = self.head;
        for _ in 0..self.period {
            let w = self.dist[pos];
            let v = self.buffer[pos];
            num += w * v;
            coef_sum += w;
            pos = (pos + self.period - 1) % self.period;
        }
        if coef_sum != 0.0 {
            Some(num / coef_sum)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub struct EdcfBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for EdcfBatchRange {
    fn default() -> Self {
        Self {
            period: (15, 50, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct EdcfBatchBuilder {
    range: EdcfBatchRange,
    kernel: Kernel,
}

impl EdcfBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<EdcfBatchOutput, EdcfError> {
        edcf_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<EdcfBatchOutput, EdcfError> {
        EdcfBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<EdcfBatchOutput, EdcfError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<EdcfBatchOutput, EdcfError> {
        EdcfBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn edcf_batch_with_kernel(
    data: &[f64],
    sweep: &EdcfBatchRange,
    k: Kernel,
) -> Result<EdcfBatchOutput, EdcfError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(EdcfError::InvalidPeriod {
                period: 0,
                data_len: 0,
            })
        }
    };
    let simd = match kernel {
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512Batch => Kernel::Avx512,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    edcf_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct EdcfBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<EdcfParams>,
    pub rows: usize,
    pub cols: usize,
}
impl EdcfBatchOutput {
    pub fn row_for_params(&self, p: &EdcfParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(15) == p.period.unwrap_or(15))
    }
    pub fn values_for(&self, p: &EdcfParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &EdcfBatchRange) -> Vec<EdcfParams> {
    let (start, end, step) = r.period;

    // Build the list of periods
    let periods: Vec<usize> = if step == 0 || start == end {
        vec![start]                          // static single value
    } else {
        (start..=end).step_by(step).collect()
    };

    // Map periods → EdcfParams
    periods
        .into_iter()
        .map(|p| EdcfParams { period: Some(p) })
        .collect()
}

#[inline(always)]
pub fn edcf_batch_slice(
    data: &[f64],
    sweep: &EdcfBatchRange,
    kern: Kernel,
) -> Result<EdcfBatchOutput, EdcfError> {
    edcf_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn edcf_batch_par_slice(
    data: &[f64],
    sweep: &EdcfBatchRange,
    kern: Kernel,
) -> Result<EdcfBatchOutput, EdcfError> {
    edcf_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn edcf_batch_inner(
    data: &[f64],
    sweep: &EdcfBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<EdcfBatchOutput, EdcfError> {
    // ─────────────────── guards unchanged ───────────────────
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(EdcfError::InvalidPeriod { period: 0, data_len: 0 });
    }

    let first = data.iter().position(|x| !x.is_nan())
                    .ok_or(EdcfError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < 2 * max_p {
        return Err(EdcfError::NotEnoughValidData {
            needed: 2 * max_p,
            valid:  data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();

    // ─────────────────── 1️⃣ allocate as MaybeUninit ───────────────────
    let mut raw = make_uninit_matrix(rows, cols);

    // ─────────────────── 2️⃣ seed NaN prefixes ────────────────────────
    let warm: Vec<usize> = combos.iter()
        .map(|c| first + 2 * c.period.unwrap())
        .collect();
    unsafe { init_matrix_prefixes(&mut raw, cols, &warm) };

    // ─────────────────── 3️⃣ per-row kernel writes into MaybeUninit ───
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();

        // Cast this single row to &mut [f64] for the kernel call
        let dst = core::slice::from_raw_parts_mut(
            dst_mu.as_mut_ptr() as *mut f64,
            dst_mu.len(),
        );

        match kern {
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => edcf_row_avx512(data, first, period, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2   => edcf_row_avx2  (data, first, period, dst),
            _              => edcf_row_scalar(data, first, period, dst),
        }
    };

    if parallel {
        raw.par_chunks_mut(cols)
           .enumerate()
           .for_each(|(row, slice)| do_row(row, slice));
    } else {
        for (row, slice) in raw.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    // ─────────────────── 4️⃣ now every element is initialised ──────────
    let values: Vec<f64> = unsafe { std::mem::transmute(raw) };

    Ok(EdcfBatchOutput { values, combos, rows, cols })
}

#[inline(always)]
unsafe fn edcf_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    edcf_scalar(data, period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn edcf_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    edcf_avx2(data, period, first, out);
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn edcf_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    edcf_avx512(data, period, first, out);
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_edcf_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = EdcfInput::from_candles(&candles, "close", EdcfParams { period: None });
        let result = edcf_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        Ok(())
    }

    fn check_edcf_accuracy_last_five(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = EdcfInput::from_candles(&candles, "hl2", EdcfParams { period: Some(15) });
        let result = edcf_with_kernel(&input, kernel)?;
        let expected = [
            59593.332275678375,
            59731.70263288801,
            59766.41512339413,
            59655.66162110993,
            59332.492883847,
        ];
        let len = result.values.len();
        let start = len - expected.len();
        for (i, &v) in result.values[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-8,
                "[{}] EDCF mismatch at {}: got {}, expected {}",
                test_name,
                start + i,
                v,
                expected[i]
            );
        }
        Ok(())
    }

    fn check_edcf_with_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = EdcfInput::with_default_candles(&candles);
        match input.data {
            EdcfData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected EdcfData::Candles"),
        }
        let period = input.get_period();
        assert_eq!(period, 15);
        Ok(())
    }

    fn check_edcf_with_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [10.0, 20.0, 30.0];
        let input = EdcfInput::from_slice(&data, EdcfParams { period: Some(0) });
        let result = edcf_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_edcf_with_no_data(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data: [f64; 0] = [];
        let input = EdcfInput::from_slice(&data, EdcfParams { period: Some(15) });
        let result = edcf_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_edcf_with_period_exceeding_data_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [10.0, 20.0, 30.0];
        let input = EdcfInput::from_slice(&data, EdcfParams { period: Some(10) });
        let result = edcf_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_edcf_very_small_data_set(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [42.0];
        let input = EdcfInput::from_slice(&data, EdcfParams { period: Some(15) });
        let result = edcf_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_edcf_with_slice_data_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_input =
            EdcfInput::from_candles(&candles, "close", EdcfParams { period: Some(15) });
        let first_result = edcf_with_kernel(&first_input, kernel)?;
        let first_values = first_result.values;
        let second_input = EdcfInput::from_slice(&first_values, EdcfParams { period: Some(5) });
        let second_result = edcf_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_values.len());
        if second_result.values.len() > 240 {
            for (i, &val) in second_result.values.iter().enumerate().skip(240) {
                assert!(
                    !val.is_nan(),
                    "Found NaN in second EDCF output at index {}",
                    i
                );
            }
        }
        Ok(())
    }

    fn check_edcf_accuracy_nan_check(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 15;
        let input = EdcfInput::from_candles(
            &candles,
            "close",
            EdcfParams {
                period: Some(period),
            },
        );
        let result = edcf_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        let start_index = 2 * period;
        if result.values.len() > start_index {
            for (i, &val) in result.values.iter().enumerate().skip(start_index) {
                assert!(!val.is_nan(), "Found NaN in EDCF output at index {}", i);
            }
        }
        Ok(())
    }

    macro_rules! generate_all_edcf_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(
                    #[test]
                    fn [<$test_fn _scalar_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar);
                    }
                    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                    #[test]
                    fn [<$test_fn _avx2_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2);
                    }
                    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                    #[test]
                    fn [<$test_fn _avx512_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512);
                    }
                )*
            }
        }
    }

    generate_all_edcf_tests!(
        check_edcf_partial_params,
        check_edcf_accuracy_last_five,
        check_edcf_with_default_candles,
        check_edcf_with_zero_period,
        check_edcf_with_no_data,
        check_edcf_with_period_exceeding_data_length,
        check_edcf_very_small_data_set,
        check_edcf_with_slice_data_reinput,
        check_edcf_accuracy_nan_check
    );

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = EdcfBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = EdcfParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        Ok(())
    }

    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste::paste! {
                #[test]
                fn [<$fn_name _scalar>]() {
                    let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test]
                fn [<$fn_name _avx2>]() {
                    let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test]
                fn [<$fn_name _avx512>]() {
                    let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch);
                }
                #[test]
                fn [<$fn_name _auto_detect>]() {
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto);
                }
            }
        };
    }

    gen_batch_tests!(check_batch_default_row);
}
