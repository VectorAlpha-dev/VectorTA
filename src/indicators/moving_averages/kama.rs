//! # Kaufman Adaptive Moving Average (KAMA)
//!
//! An adaptive moving average that dynamically adjusts its smoothing factor
//! based on price noise or volatility. When price movements are relatively stable,
//! KAMA becomes smoother, filtering out minor fluctuations. Conversely, in more
//! volatile or trending periods, KAMA becomes more reactive, aiming to catch the
//! prevailing trend sooner.
//!
//! ## Parameters
//! - **period**: Core lookback length for the KAMA calculation (defaults to 30).
//!
//! ## Errors
//! - **NoData**: kama: No input data was provided.
//! - **AllValuesNaN**: kama: All input data is `NaN`.
//! - **InvalidPeriod**: kama: `period` is zero or exceeds the data length.
//! - **NotEnoughData**: kama: Not enough data to calculate KAMA for the requested `period`.
//!
//! ## Returns
//! - **`Ok(KamaOutput)`** on success, containing a `Vec<f64>` with length matching the input.
//! - **`Err(KamaError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_kernel, detect_best_batch_kernel, make_uninit_matrix, init_matrix_prefixes, alloc_with_nan_prefix};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;
use std::mem::MaybeUninit;

#[derive(Debug, Clone)]
pub enum KamaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct KamaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct KamaParams {
    pub period: Option<usize>,
}

impl Default for KamaParams {
    fn default() -> Self {
        KamaParams { period: Some(30) }
    }
}

#[derive(Debug, Clone)]
pub struct KamaInput<'a> {
    pub data: KamaData<'a>,
    pub params: KamaParams,
}

impl<'a> AsRef<[f64]> for KamaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            KamaData::Slice(slice) => slice,
            KamaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

impl<'a> KamaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: KamaParams) -> Self {
        Self {
            data: KamaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: KamaParams) -> Self {
        Self {
            data: KamaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", KamaParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(30)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct KamaBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for KamaBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl KamaBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<KamaOutput, KamaError> {
        let p = KamaParams { period: self.period };
        let i = KamaInput::from_candles(c, "close", p);
        kama_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<KamaOutput, KamaError> {
        let p = KamaParams { period: self.period };
        let i = KamaInput::from_slice(d, p);
        kama_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<KamaStream, KamaError> {
        let p = KamaParams { period: self.period };
        KamaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum KamaError {
    #[error("kama: No data provided for KAMA.")]
    NoData,
    #[error("kama: All values are NaN.")]
    AllValuesNaN,
    #[error("kama: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("kama: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughData { needed: usize, valid: usize },
}

#[inline]
pub fn kama(input: &KamaInput) -> Result<KamaOutput, KamaError> {
    kama_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn kama_prepare<'a>(
    input: &'a KamaInput,
    kernel: Kernel,
) -> Result<(
    /*data*/ &'a [f64],
    /*period*/ usize,
    /*first*/ usize,
    /*chosen*/ Kernel,
), KamaError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(KamaError::NoData);
    }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(KamaError::AllValuesNaN)?;
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(KamaError::InvalidPeriod { period, data_len: len });
    }
    if (len - first) <= period {
        return Err(KamaError::NotEnoughData { needed: period + 1, valid: len - first });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    Ok((data, period, first, chosen))
}

#[inline(always)]
fn kama_compute_into(
    data: &[f64],
    period: usize,
    first: usize,
    kernel: Kernel,
    out: &mut [f64],
) {
    unsafe {
        match kernel {
            Kernel::Scalar | Kernel::ScalarBatch => kama_scalar(data, period, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => kama_avx2(data, period, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => kama_avx512(data, period, first, out),
            _ => unreachable!(),
        }
    }
}

pub fn kama_with_kernel(input: &KamaInput, kernel: Kernel) -> Result<KamaOutput, KamaError> {
    let (data, period, first, chosen) = kama_prepare(input, kernel)?;
    
    let warm = first + period;
    let mut out = alloc_with_nan_prefix(data.len(), warm);
    
    kama_compute_into(data, period, first, chosen, &mut out);
    
    Ok(KamaOutput { values: out })
}

#[inline]
pub fn kama_scalar(
    data: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    assert!(
        out.len() >= data.len(),
        "`out` must be at least as long as `data`"
    );

    let len = data.len();
    let lookback = period.saturating_sub(1);

    let const_max = 2.0 / (30.0 + 1.0);
    let const_diff = (2.0 / (2.0 + 1.0)) - const_max;

    let mut sum_roc1 = 0.0;
    let mut today = first_valid;
    for i in 0..=lookback {
        sum_roc1 += (data[today + i + 1] - data[today + i]).abs();
    }

    let initial_idx = today + lookback + 1;
    let mut prev_kama = data[initial_idx];
    out[initial_idx] = prev_kama;

    let mut trailing_idx = today;
    let mut trailing_value = data[trailing_idx];

    for i in (initial_idx + 1)..len {
        let price = data[i];

        sum_roc1 -= (data[trailing_idx + 1] - trailing_value).abs();

        sum_roc1 += (price - data[i - 1]).abs();

        trailing_value = data[trailing_idx + 1];
        trailing_idx += 1;

        let direction = (price - data[trailing_idx]).abs();
        let er = if sum_roc1 == 0.0 { 0.0 } else { direction / sum_roc1 };
        let sc = (er * const_diff + const_max).powi(2);
        prev_kama += (price - prev_kama) * sc;
        out[i] = prev_kama;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn kama_avx2(
    data: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    use core::arch::x86_64::*;

    const ABS_MASK: i64 = 0x7FFF_FFFF_FFFF_FFFFu64 as i64;
    debug_assert!(period >= 2 && period <= data.len());
    debug_assert_eq!(data.len(), out.len());

    /* ----------------------------------------------------------- *
     * 1.  Σ|Δprice| for the first window                           *
     * ----------------------------------------------------------- */
    let lookback = period - 1;
    let mut sum_roc1: f64 = 0.0;
    let base = data.as_ptr().add(first_valid);

    if lookback >= 15 {
        let mask_pd = _mm256_castsi256_pd(_mm256_set1_epi64x(ABS_MASK));
        let mut acc0 = _mm256_setzero_pd();
        let mut acc1 = _mm256_setzero_pd();
        let mut idx = 0usize;

        // helper: |ptr[idx+1] – ptr[idx]|
        #[inline(always)]
        unsafe fn abs_diff(ptr: *const f64, ofs: usize, m: __m256d) -> __m256d {
            let a = _mm256_loadu_pd(ptr.add(ofs));
            let b = _mm256_loadu_pd(ptr.add(ofs + 1));
            _mm256_and_pd(_mm256_sub_pd(b, a), m)
        }

        while idx + 15 <= lookback {
            acc0 = _mm256_add_pd(acc0, abs_diff(base, idx,      mask_pd));
            acc1 = _mm256_add_pd(acc1, abs_diff(base, idx + 4,  mask_pd));
            acc0 = _mm256_add_pd(acc0, abs_diff(base, idx + 8,  mask_pd));
            acc1 = _mm256_add_pd(acc1, abs_diff(base, idx + 12, mask_pd));
            idx += 16;
        }

        // horizontal reduction (AVX2 – no native reduce)
        let sumv = _mm256_add_pd(acc0, acc1);                         // 4-lanes
        let hi   = _mm256_extractf128_pd::<1>(sumv);                  // upper 2
        let lo   = _mm256_castpd256_pd128(sumv);                      // lower 2
        let pair = _mm_add_pd(lo, hi);                                // 2-lanes
        sum_roc1 = _mm_cvtsd_f64(pair) + _mm_cvtsd_f64(_mm_unpackhi_pd(pair, pair));   // scalar :contentReference[oaicite:0]{index=0}

        // scalar tail (<16)
        while idx <= lookback {
            sum_roc1 += (*base.add(idx + 1) - *base.add(idx)).abs();
            idx += 1;
        }
    } else {
        for k in 0..=lookback {
            sum_roc1 += (*base.add(k + 1) - *base.add(k)).abs();
        }
    }

    /* ----------------------------------------------------------- *
     * 2.  Seed first KAMA                                         *
     * ----------------------------------------------------------- */
    let init_idx = first_valid + lookback + 1;
    let mut kama = *data.get_unchecked(init_idx);
    *out.get_unchecked_mut(init_idx) = kama;

    /* ----------------------------------------------------------- *
     * 3.  Rolling update                                          *
     * ----------------------------------------------------------- */
    let const_max  = 2.0 / 31.0;
    let const_diff = (2.0 / 3.0) - const_max;

    let mut tail_idx = first_valid;
    let mut tail_val = *data.get_unchecked(tail_idx);

    for i in (init_idx + 1)..data.len() {
        // Σ|Δp| update
        let price     = *data.get_unchecked(i);
        let new_diff  = (price - *data.get_unchecked(i - 1)).abs();

        let next_tail = *data.get_unchecked(tail_idx + 1);
        let old_diff  = (next_tail - tail_val).abs();
        sum_roc1 += new_diff - old_diff;

        tail_val  = next_tail;
        tail_idx += 1;

        // smoothing constant (square cheaper than powi) :contentReference[oaicite:1]{index=1}
        let direction = (price - *data.get_unchecked(tail_idx)).abs();
        let er = if sum_roc1 == 0.0 { 0.0 } else { direction / sum_roc1 };
        let t  = er.mul_add(const_diff, const_max);                   // one FMA, one round :contentReference[oaicite:2]{index=2}
        let sc = t * t;

        // KAMA recurrence – compiler emits vfmadd132sd on AVX2 targets :contentReference[oaicite:3]{index=3}
        kama = (price - kama).mul_add(sc, kama);

        *out.get_unchecked_mut(i) = kama;                             // scalar store

        // Prefetch 128 B ahead into L2 (T1 hint) :contentReference[oaicite:4]{index=4}
        _mm_prefetch(data.as_ptr().add(i + 128) as *const i8, _MM_HINT_T1);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512vl,fma")]
#[inline]
pub unsafe fn kama_avx512(
    data:        &[f64],
    period:      usize,
    first_valid: usize,
    out:         &mut [f64],
) {
    use core::arch::x86_64::*;

    const ABS_MASK: i64 = 0x7FFF_FFFF_FFFF_FFFFu64 as i64;

    debug_assert!(period >= 2 && period <= data.len());
    debug_assert_eq!(data.len(), out.len());

    /* ------------------------------------------------------------------ *
     * 1.  Σ|Δprice| for the first window                                  *
     * ------------------------------------------------------------------ */
    let lookback = period - 1;
    let mut sum_roc1: f64 = 0.0;
    let base = data.as_ptr().add(first_valid);

    if lookback >= 31 {
        let mask_pd = _mm512_castsi512_pd(_mm512_set1_epi64(ABS_MASK));
        let mut acc0 = _mm512_setzero_pd();
        let mut acc1 = _mm512_setzero_pd();
        let mut acc2 = _mm512_setzero_pd();
        let mut acc3 = _mm512_setzero_pd();

        #[inline(always)]
        unsafe fn abs_diff(ptr: *const f64, idx: usize, mask: __m512d) -> __m512d {
            let a = _mm512_loadu_pd(ptr.add(idx));
            let b = _mm512_loadu_pd(ptr.add(idx + 1));
            _mm512_and_pd(_mm512_sub_pd(b, a), mask)
        }

        let mut j = 0usize;
        while j + 31 <= lookback {
            acc0 = _mm512_add_pd(acc0, abs_diff(base, j,      mask_pd));
            acc1 = _mm512_add_pd(acc1, abs_diff(base, j + 8,  mask_pd));
            acc2 = _mm512_add_pd(acc2, abs_diff(base, j + 16, mask_pd));
            acc3 = _mm512_add_pd(acc3, abs_diff(base, j + 24, mask_pd));
            j += 32;
        }
        let acc_all = _mm512_add_pd(_mm512_add_pd(acc0, acc1), _mm512_add_pd(acc2, acc3));
        sum_roc1 = _mm512_reduce_add_pd(acc_all);          // horizontal sum :contentReference[oaicite:0]{index=0}

        while j <= lookback {
            sum_roc1 += (*base.add(j + 1) - *base.add(j)).abs();
            j += 1;
        }
    } else {
        for k in 0..=lookback {
            sum_roc1 += (*base.add(k + 1) - *base.add(k)).abs();
        }
    }

    /* ------------------------------------------------------------------ *
     * 2.  Seed first output                                              *
     * ------------------------------------------------------------------ */
    let init_idx = first_valid + lookback + 1;
    let mut kama = *data.get_unchecked(init_idx);
    *out.get_unchecked_mut(init_idx) = kama;

    /* ------------------------------------------------------------------ *
     * 3.  Rolling KAMA update (scalar recurrence)                        *
     * ------------------------------------------------------------------ */
    let const_max  = 2.0 / 31.0;
    let const_diff = (2.0 / 3.0) - const_max;

    let mut tail_idx = first_valid;
    let mut tail_val = *data.get_unchecked(tail_idx);

    for i in (init_idx + 1)..data.len() {
        /* ---- update Σ|Δp| ---- */
        let price     = *data.get_unchecked(i);
        let new_diff  = (price - *data.get_unchecked(i - 1)).abs();

        let next_tail = *data.get_unchecked(tail_idx + 1);
        let old_diff  = (next_tail - tail_val).abs();
        sum_roc1 += new_diff - old_diff;

        tail_val = next_tail;
        tail_idx += 1;

        /* ---- efficiency ratio & smoothing constant ---- */
        let direction = (price - *data.get_unchecked(tail_idx)).abs();
        let er = if sum_roc1 == 0.0 { 0.0 } else { direction / sum_roc1 };

        // fused multiply-add → one FMA µ-op; square via multiplication (faster than powi) :contentReference[oaicite:1]{index=1}
        let t  = er.mul_add(const_diff, const_max);
        let sc = t * t;

        /* ---- KAMA recurrence ---- */
        // compiler lowers mul_add to `vfmadd132sd` on AVX-512 targets :contentReference[oaicite:2]{index=2}
        kama = (price - kama).mul_add(sc, kama);

        *out.get_unchecked_mut(i) = kama;  // regular scalar store (no NT store) :contentReference[oaicite:3]{index=3}

        // Prefetch two cache lines ahead; _MM_HINT_T1 prefers L2 :contentReference[oaicite:4]{index=4}
        _mm_prefetch(data.as_ptr().add(i + 128) as *const i8, _MM_HINT_T1);
    }
}

use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct KamaStream {
    period: usize,
    buffer: VecDeque<f64>,
    prev_kama: f64,
    sum_roc1: f64,
    const_max: f64,
    const_diff: f64,
}

impl KamaStream {
    pub fn try_new(params: KamaParams) -> Result<Self, KamaError> {
        let period = params.period.unwrap_or(30);
        if period == 0 {
            return Err(KamaError::InvalidPeriod { period, data_len: 0 });
        }
        Ok(Self {
            period,
            buffer: VecDeque::with_capacity(period + 1),
            prev_kama: 0.0,
            sum_roc1: 0.0,
            const_max: 2.0 / (30.0 + 1.0),
            const_diff: (2.0 / (2.0 + 1.0)) - (2.0 / (30.0 + 1.0)),
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        if self.buffer.len() < self.period {
            self.buffer.push_back(value);
            return None;
        }

        if self.buffer.len() == self.period {
            self.sum_roc1 = 0.0;
            for i in 0..(self.period - 1) {
                let a = self.buffer[i];
                let b = self.buffer[i + 1];
                self.sum_roc1 += (b - a).abs();
            }
            if let Some(&last) = self.buffer.back() {
                self.sum_roc1 += (value - last).abs();
            }

            self.prev_kama = value;
            self.buffer.push_back(value);
            return Some(self.prev_kama);
        }

        let old_front = self.buffer.pop_front().unwrap();
        let new_front = *self.buffer.front().unwrap();

        self.sum_roc1 -= (new_front - old_front).abs();

        let last = *self.buffer.back().unwrap();
        self.sum_roc1 += (value - last).abs();

        let direction = (value - new_front).abs();

        let er = if self.sum_roc1 == 0.0 {
            0.0
        
            } else {
            direction / self.sum_roc1
        };
        let sc = (er * self.const_diff + self.const_max).powi(2);
        self.prev_kama += (value - self.prev_kama) * sc;

        self.buffer.push_back(value);
        Some(self.prev_kama)
    }
}


#[derive(Clone, Debug)]
pub struct KamaBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for KamaBatchRange {
    fn default() -> Self {
        Self { period: (30, 240, 1) }
    }
}

#[derive(Clone, Debug, Default)]
pub struct KamaBatchBuilder {
    range: KamaBatchRange,
    kernel: Kernel,
}

impl KamaBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<KamaBatchOutput, KamaError> {
        kama_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<KamaBatchOutput, KamaError> {
        KamaBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<KamaBatchOutput, KamaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<KamaBatchOutput, KamaError> {
        KamaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn kama_batch_with_kernel(
    data: &[f64],
    sweep: &KamaBatchRange,
    k: Kernel,
) -> Result<KamaBatchOutput, KamaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(KamaError::InvalidPeriod { period: 0, data_len: 0 }),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    kama_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct KamaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<KamaParams>,
    pub rows: usize,
    pub cols: usize,
}

impl KamaBatchOutput {
    pub fn row_for_params(&self, p: &KamaParams) -> Option<usize> {
        self.combos.iter().position(|c| c.period.unwrap_or(30) == p.period.unwrap_or(30))
    }
    pub fn values_for(&self, p: &KamaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &KamaBatchRange) -> Vec<KamaParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    periods.into_iter().map(|p| KamaParams { period: Some(p) }).collect()
}

#[inline(always)]
pub fn kama_batch_slice(
    data: &[f64],
    sweep: &KamaBatchRange,
    kern: Kernel,
) -> Result<KamaBatchOutput, KamaError> {
    kama_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn kama_batch_par_slice(
    data: &[f64],
    sweep: &KamaBatchRange,
    kern: Kernel,
) -> Result<KamaBatchOutput, KamaError> {
    kama_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn kama_batch_inner(
    data: &[f64],
    sweep: &KamaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<KamaBatchOutput, KamaError> {
    let combos = expand_grid(sweep);
    let cols = data.len();
    let rows = combos.len();
    let mut values = vec![0.0; rows * cols];
    
    kama_batch_inner_into(data, sweep, kern, parallel, &mut values)?;
    
    Ok(KamaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn kama_batch_inner_into(
    data: &[f64],
    sweep: &KamaBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<KamaParams>, KamaError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(KamaError::InvalidPeriod { period: 0, data_len: 0 });
    }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(KamaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first <= max_p {
        return Err(KamaError::NotEnoughData { needed: max_p + 1, valid: data.len() - first });
    }
    let rows = combos.len();
    let cols = data.len();

    /* ---------------------------------------------------------------
    * 1.  prepare warmup periods and initialize NaN prefixes
    * ------------------------------------------------------------- */
    let warm: Vec<usize> = combos.iter()
                                .map(|c| first + c.period.unwrap())
                                .collect();

    // SAFETY: We're reinterpreting the output slice as MaybeUninit to use the efficient
    // init_matrix_prefixes function. This is safe because:
    // 1. MaybeUninit<T> has the same layout as T
    // 2. We ensure all values are written before the slice is used again
    let out_uninit = unsafe {
        std::slice::from_raw_parts_mut(
            out.as_mut_ptr() as *mut MaybeUninit<f64>,
            out.len()
        )
    };

    unsafe { init_matrix_prefixes(out_uninit, cols, &warm) };

    /* ---------------------------------------------------------------
    * 2.  helper that fills a single row
    * ------------------------------------------------------------- */
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period  = combos[row].period.unwrap();

        // Cast the row slice (which is definitely ours to mutate) to f64
        let dst = core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        match kern {
            Kernel::Scalar | Kernel::ScalarBatch => kama_row_scalar(data, first, period, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => kama_row_avx2(data, first, period, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => kama_row_avx512(data, first, period, dst),
            _ => unreachable!(),
        }
    };

    /* ---------------------------------------------------------------
    * 3.  run every row kernel; no element is read before it is written
    * ------------------------------------------------------------- */
    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_uninit.par_chunks_mut(cols)
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
unsafe fn kama_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    kama_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn kama_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    kama_avx2(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn kama_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    kama_avx512(data, period, first, out)
}


#[inline(always)]
pub fn expand_grid_kama(r: &KamaBatchRange) -> Vec<KamaParams> {
    expand_grid(r)
}

// Python bindings
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "kama")]
pub fn kama_py<'py>(
    py: Python<'py>,
    arr_in: PyReadonlyArray1<'py, f64>,
    period: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    use numpy::{PyArray1, PyArrayMethods};
    
    let slice_in = arr_in.as_slice()?; // zero-copy, read-only view
    
    // Build input struct
    let params = KamaParams {
        period: Some(period),
    };
    let kama_in = KamaInput::from_slice(slice_in, params);
    
    // Allocate NumPy output buffer
    let out_arr = unsafe { PyArray1::<f64>::new(py, [slice_in.len()], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? }; // safe: we own the array
    
    // Heavy lifting without the GIL
    py.allow_threads(|| -> Result<(), KamaError> {
        let (data, per, first, chosen) = kama_prepare(&kama_in, Kernel::Auto)?;
        // Prefix initialize exactly once
        slice_out[..first + per].fill(f64::NAN);
        kama_compute_into(data, per, first, chosen, slice_out);
        Ok(())
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?; // unify error type
    
    Ok(out_arr)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "kama_batch")]
pub fn kama_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;
    
    let slice_in = data.as_slice()?;
    
    let sweep = KamaBatchRange {
        period: period_range,
    };
    
    // 1. Expand grid once to know rows*cols
    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();
    
    // 2. Pre-allocate NumPy array (1-D, will reshape later)
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };
    
    // 3. Heavy work without the GIL
    let combos = py.allow_threads(|| {
        // Resolve Kernel::Auto to a specific kernel
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
        // Use the _into variant that writes directly to our pre-allocated buffer
        kama_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    
    // 4. Build dict with the GIL
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


#[cfg(feature = "python")]
#[pyclass(name = "KamaStream")]
pub struct KamaStreamPy {
    inner: KamaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl KamaStreamPy {
    #[new]
    pub fn new(period: usize) -> PyResult<Self> {
        let params = KamaParams {
            period: Some(period),
        };
        match KamaStream::try_new(params) {
            Ok(stream) => Ok(Self { inner: stream }),
            Err(e) => Err(PyValueError::new_err(format!("KamaStream error: {}", e))),
        }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.inner.update(value)
    }
}

// WASM bindings
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn kama_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = KamaParams {
        period: Some(period),
    };
    let input = KamaInput::from_slice(data, params);
    match kama_with_kernel(&input, Kernel::Scalar) {
        Ok(output) => Ok(output.values),
        Err(e) => Err(JsValue::from_str(&format!("KAMA error: {}", e))),
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn kama_batch_js(
    data: &[f64],
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = KamaBatchRange {
        period: (period_start, period_end, period_step),
    };
    match kama_batch_slice(data, &sweep, Kernel::Scalar) {
        Ok(output) => Ok(output.values),
        Err(e) => Err(JsValue::from_str(&format!("KAMA batch error: {}", e))),
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn kama_batch_metadata_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Vec<f64> {
    let periods: Vec<usize> = if period_step == 0 || period_start == period_end {
        vec![period_start]
    } else {
        (period_start..=period_end).step_by(period_step).collect()
    };
    
    let mut result = Vec::new();
    for &period in &periods {
        result.push(period as f64);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_kama_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = KamaParams { period: None };
        let input = KamaInput::from_candles(&candles, "close", default_params);
        let output = kama_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_kama_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = KamaInput::with_default_candles(&candles);
        let result = kama_with_kernel(&input, kernel)?;
        let expected_last_five = [
            60234.925553804125,
            60176.838757545665,
            60115.177367962766,
            60071.37070833558,
            59992.79386218023,
        ];
        assert!(
            result.values.len() >= 5,
            "Expected at least 5 values to compare"
        );
        assert_eq!(
            result.values.len(),
            candles.close.len(),
            "KAMA output length does not match input length"
        );
        let start_index = result.values.len().saturating_sub(5);
        let last_five = &result.values[start_index..];
        for (i, &val) in last_five.iter().enumerate() {
            let exp = expected_last_five[i];
            assert!(
                (val - exp).abs() < 1e-6,
                "KAMA mismatch at last-five index {}: expected {}, got {}",
                i,
                exp,
                val
            );
        }
        Ok(())
    }

    fn check_kama_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = KamaInput::with_default_candles(&candles);
        match input.data {
            KamaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected KamaData::Candles"),
        }
        let output = kama_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_kama_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = KamaParams { period: Some(0) };
        let input = KamaInput::from_slice(&input_data, params);
        let res = kama_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] KAMA should fail with zero period", test_name);
        Ok(())
    }

    fn check_kama_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = KamaParams { period: Some(10) };
        let input = KamaInput::from_slice(&data_small, params);
        let res = kama_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] KAMA should fail with period exceeding length", test_name);
        Ok(())
    }

    fn check_kama_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = KamaParams { period: Some(30) };
        let input = KamaInput::from_slice(&single_point, params);
        let res = kama_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] KAMA should fail with insufficient data", test_name);
        Ok(())
    }

    fn check_kama_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = KamaParams { period: Some(30) };
        let first_input = KamaInput::from_candles(&candles, "close", first_params);
        let first_result = kama_with_kernel(&first_input, kernel)?;
        let second_params = KamaParams { period: Some(10) };
        let second_input = KamaInput::from_slice(&first_result.values, second_params);
        let second_result = kama_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    fn check_kama_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = KamaInput::from_candles(
            &candles,
            "close",
            KamaParams { period: Some(30) },
        );
        let res = kama_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        for val in res.values.iter().skip(30) {
            assert!(val.is_finite());
        }
        Ok(())
    }

    fn check_kama_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 30;
        let input = KamaInput::from_candles(
            &candles,
            "close",
            KamaParams { period: Some(period) },
        );
        let batch_output = kama_with_kernel(&input, kernel)?.values;
        let mut stream = KamaStream::try_new(KamaParams { period: Some(period) })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(val) => stream_values.push(val),
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
                "[{}] KAMA streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_kama_tests {
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
    // Check for poison values in single output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_kama_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test with multiple parameter combinations to better catch uninitialized memory bugs
        let test_periods = vec![2, 5, 10, 14, 20, 30, 50, 100, 200];
        let test_sources = vec!["open", "high", "low", "close", "hl2", "hlc3", "ohlc4"];

        for period in &test_periods {
            for source in &test_sources {
                let input = KamaInput::from_candles(
                    &candles,
                    source,
                    KamaParams {
                        period: Some(*period),
                    },
                );
                let output = kama_with_kernel(&input, kernel)?;

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
                            "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} with period={}, source={}",
                            test_name, val, bits, i, period, source
                        );
                    }

                    // Check for init_matrix_prefixes poison (0x22222222_22222222)
                    if bits == 0x22222222_22222222 {
                        panic!(
                            "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} with period={}, source={}",
                            test_name, val, bits, i, period, source
                        );
                    }

                    // Check for make_uninit_matrix poison (0x33333333_33333333)
                    if bits == 0x33333333_33333333 {
                        panic!(
                            "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} with period={}, source={}",
                            test_name, val, bits, i, period, source
                        );
                    }
                }
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_kama_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    generate_all_kama_tests!(
        check_kama_partial_params,
        check_kama_accuracy,
        check_kama_default_candles,
        check_kama_zero_period,
        check_kama_period_exceeds_length,
        check_kama_very_small_dataset,
        check_kama_reinput,
        check_kama_nan_handling,
        check_kama_streaming,
        check_kama_no_poison
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = KamaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = KamaParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [
            60234.925553804125,
            60176.838757545665,
            60115.177367962766,
            60071.37070833558,
            59992.79386218023,
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

    // Check for poison values in batch output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test batch with multiple parameter combinations to better catch uninitialized memory bugs
        let test_sources = vec!["open", "high", "low", "close", "hl2", "hlc3", "ohlc4"];
        
        for source in &test_sources {
            // Test with comprehensive period ranges
            let output = KamaBatchBuilder::new()
                .kernel(kernel)
                .period_range(2, 200, 3)  // Wide range: 2 to 200 with step 3
                .apply_candles(&c, source)?;

            // Check every value in the entire batch matrix for poison patterns
            for (idx, &val) in output.values.iter().enumerate() {
                // Skip NaN values as they're expected in warmup periods
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;

                // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with source={}",
                        test, val, bits, row, col, idx, source
                    );
                }

                // Check for init_matrix_prefixes poison (0x22222222_22222222)
                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {}) with source={}",
                        test, val, bits, row, col, idx, source
                    );
                }

                // Check for make_uninit_matrix poison (0x33333333_33333333)
                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with source={}",
                        test, val, bits, row, col, idx, source
                    );
                }
            }
        }

        // Also test edge cases with very small and very large periods
        let edge_case_ranges = vec![(2, 5, 1), (190, 200, 2), (50, 100, 10)];
        for (start, end, step) in edge_case_ranges {
            let output = KamaBatchBuilder::new()
                .kernel(kernel)
                .period_range(start, end, step)
                .apply_candles(&c, "close")?;

            for (idx, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;

                if bits == 0x11111111_11111111 || bits == 0x22222222_22222222 || bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found poison value {} (0x{:016X}) at row {} col {} with range ({},{},{})",
                        test, val, bits, row, col, start, end, step
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
