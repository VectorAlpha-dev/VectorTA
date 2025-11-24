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
//! - **EmptyInputData**: kama: Input data slice is empty.
//! - **AllValuesNaN**: kama: All input data is `NaN`.
//! - **InvalidPeriod**: kama: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: kama: Not enough valid data to calculate KAMA for the requested `period`.
//!
//! ## Returns
//! - **`Ok(KamaOutput)`** on success, containing a `Vec<f64>` with length matching the input.
//! - **`Err(KamaError)`** otherwise.
//!
//! ## Developer Notes
//! - **Scalar path**: ✅ Safe, loop-jammed hot loop; uses `mul_add` and squares via multiply (no `powi`), reuses trailing load.
//! - **AVX2/AVX512**: ✅ Vectorized initial Σ|Δp|; FMA; reuses `next_tail` for direction and clamps prefetch.
//! - **Streaming update**: ✅ O(1) ring-update using fixed-size rings (no VecDeque); identical warmup behavior.
//! - **Batch**: ✅ Shared prefix-sum precompute for Σ|Δp| seeding across rows (reduces per-row setup).
//! - **Memory**: ✅ `alloc_with_nan_prefix` and matrix helpers; no O(N) temporaries for outputs.
//! - **Status**: Stable; AVX2/AVX512 are modestly faster than scalar at 100k; keep enabled.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::cuda_available;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::CudaKama;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::kama_wrapper::DeviceArrayF32Kama;
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
#[cfg_attr(feature = "wasm", derive(serde::Serialize, serde::Deserialize))]
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
        let p = KamaParams {
            period: self.period,
        };
        let i = KamaInput::from_candles(c, "close", p);
        kama_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<KamaOutput, KamaError> {
        let p = KamaParams {
            period: self.period,
        };
        let i = KamaInput::from_slice(d, p);
        kama_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<KamaStream, KamaError> {
        let p = KamaParams {
            period: self.period,
        };
        KamaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum KamaError {
    #[error("kama: Input data slice is empty.")]
    EmptyInputData,
    #[error("kama: All values are NaN.")]
    AllValuesNaN,
    #[error("kama: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("kama: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("kama: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("kama: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange { start: usize, end: usize, step: usize },
    #[error("kama: Invalid kernel for batch operation: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("kama: invalid input: {0}")]
    InvalidInput(&'static str),
}

#[inline]
pub fn kama(input: &KamaInput) -> Result<KamaOutput, KamaError> {
    kama_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn kama_prepare<'a>(
    input: &'a KamaInput,
    kernel: Kernel,
) -> Result<
    (
        // data
        &'a [f64],
        // period
        usize,
        // first
        usize,
        // chosen
        Kernel,
    ),
    KamaError,
> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(KamaError::EmptyInputData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(KamaError::AllValuesNaN)?;
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(KamaError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) <= period {
        return Err(KamaError::NotEnoughValidData {
            needed: period + 1,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    Ok((data, period, first, chosen))
}

#[inline(always)]
fn kama_compute_into(data: &[f64], period: usize, first: usize, kernel: Kernel, out: &mut [f64]) {
    unsafe {
        match kernel {
            Kernel::Scalar | Kernel::ScalarBatch => kama_scalar(data, period, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => kama_avx2(data, period, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => kama_avx512(data, period, first, out),
            _ => kama_scalar(data, period, first, out), // Fallback to scalar
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

/// Compute KAMA into a caller-provided output buffer (no allocations).
///
/// - Preserves NaN warmups identical to the Vec-returning API.
/// - The output slice length must equal the input length.
#[cfg(not(feature = "wasm"))]
pub fn kama_into(input: &KamaInput, out: &mut [f64]) -> Result<(), KamaError> {
    let (data, period, first, chosen) = kama_prepare(input, Kernel::Auto)?;

    if out.len() != data.len() {
        return Err(KamaError::OutputLengthMismatch {
            expected: data.len(),
            got: out.len(),
        });
    }

    // Prefill NaN warmup prefix using the same quiet-NaN pattern
    let warm = first + period;
    let qnan = f64::from_bits(0x7ff8_0000_0000_0000);
    let pref = warm.min(out.len());
    for v in &mut out[..pref] {
        *v = qnan;
    }

    // Compute values into the provided buffer
    kama_compute_into(data, period, first, chosen, out);

    Ok(())
}

/// Compute KAMA directly into the provided output slice.
/// The output slice must be the same length as the input data.
#[inline]
pub fn kama_into_slice(dst: &mut [f64], input: &KamaInput, kern: Kernel) -> Result<(), KamaError> {
    let (data, period, first, chosen) = kama_prepare(input, kern)?;

    // Verify output buffer size matches input
    if dst.len() != data.len() {
        return Err(KamaError::OutputLengthMismatch {
            expected: data.len(),
            got: dst.len(),
        });
    }

    // Compute KAMA values directly into dst
    kama_compute_into(data, period, first, chosen, dst);

    // Fill warmup period with NaN
    let warmup_end = first + period;
    for v in &mut dst[..warmup_end] {
        *v = f64::NAN;
    }

    Ok(())
}

#[inline(always)]
pub fn kama_scalar(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    assert_eq!(
        out.len(),
        data.len(),
        "`out` must be the same length as `data`"
    );

    let len = data.len();
    let lookback = period.saturating_sub(1);

    let const_max = 2.0 / (30.0 + 1.0);
    let const_diff = (2.0 / (2.0 + 1.0)) - const_max;

    // 1) Initial Σ|Δp| over the first window [first_valid .. first_valid+period]
    let mut sum_roc1 = 0.0;
    let today = first_valid;
    for i in 0..=lookback {
        let a = data[today + i];
        let b = data[today + i + 1];
        sum_roc1 += (b - a).abs();
    }

    // 2) Seed at index = first_valid + lookback + 1
    let initial_idx = today + lookback + 1;
    let mut kama = data[initial_idx];
    out[initial_idx] = kama;

    // Maintain a trailing window pointer/value to drop the oldest |Δp|
    let mut trailing_idx = today;
    let mut trailing_value = data[trailing_idx];

    // 3) Rolling update
    for i in (initial_idx + 1)..len {
        let price_prev = data[i - 1];
        let price = data[i];

        // update Σ|Δp|: drop oldest diff, add newest diff
        let next_tail = data[trailing_idx + 1];
        let old_diff = (next_tail - trailing_value).abs();
        let new_diff = (price - price_prev).abs();
        sum_roc1 += new_diff - old_diff;

        // advance trailing window
        trailing_value = next_tail;
        trailing_idx += 1;

        // Efficiency ratio + smoothing constant
        let direction = (price - trailing_value).abs();
        let er = if sum_roc1 == 0.0 {
            0.0
        } else {
            direction / sum_roc1
        };
        let t = er.mul_add(const_diff, const_max);
        let sc = t * t; // cheaper than powi(2)

        // KAMA recurrence; mul_add allows FMA on capable targets
        kama = (price - kama).mul_add(sc, kama);
        out[i] = kama;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn kama_avx2(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    use core::arch::x86_64::*;

    const ABS_MASK: i64 = 0x7FFF_FFFF_FFFF_FFFFu64 as i64;
    debug_assert!(period >= 2 && period <= data.len());
    debug_assert_eq!(data.len(), out.len());

    // ----------------------------------------------------------- *
    // 1.  Σ|Δprice| for the first window                           *
    // -----------------------------------------------------------
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
            acc0 = _mm256_add_pd(acc0, abs_diff(base, idx, mask_pd));
            acc1 = _mm256_add_pd(acc1, abs_diff(base, idx + 4, mask_pd));
            acc0 = _mm256_add_pd(acc0, abs_diff(base, idx + 8, mask_pd));
            acc1 = _mm256_add_pd(acc1, abs_diff(base, idx + 12, mask_pd));
            idx += 16;
        }

        // horizontal reduction (AVX2 – no native reduce)
        let sumv = _mm256_add_pd(acc0, acc1); // 4-lanes
        let hi = _mm256_extractf128_pd::<1>(sumv); // upper 2
        let lo = _mm256_castpd256_pd128(sumv); // lower 2
        let pair = _mm_add_pd(lo, hi); // 2-lanes
        sum_roc1 = _mm_cvtsd_f64(pair) + _mm_cvtsd_f64(_mm_unpackhi_pd(pair, pair)); // scalar

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

    // ----------------------------------------------------------- *
    // 2.  Seed first KAMA                                         *
    // -----------------------------------------------------------
    let init_idx = first_valid + lookback + 1;
    let mut kama = *data.get_unchecked(init_idx);
    *out.get_unchecked_mut(init_idx) = kama;

    // ----------------------------------------------------------- *
    // 3.  Rolling update                                          *
    // -----------------------------------------------------------
    let const_max = 2.0 / 31.0;
    let const_diff = (2.0 / 3.0) - const_max;

    let mut tail_idx = first_valid;
    let mut tail_val = *data.get_unchecked(tail_idx);

    for i in (init_idx + 1)..data.len() {
        // Σ|Δp| update
        let price = *data.get_unchecked(i);
        let new_diff = (price - *data.get_unchecked(i - 1)).abs();

        let next_tail = *data.get_unchecked(tail_idx + 1);
        let old_diff = (next_tail - tail_val).abs();
        sum_roc1 += new_diff - old_diff;

        tail_val = next_tail;
        tail_idx += 1;

        // smoothing constant (square cheaper than powi)
        // Reuse next_tail to avoid an extra load
        let direction = (price - next_tail).abs();
        let er = if sum_roc1 == 0.0 {
            0.0
        } else {
            direction / sum_roc1
        };
        let t = er.mul_add(const_diff, const_max); // one FMA, one round
        let sc = t * t;

        // KAMA recurrence – compiler emits vfmadd132sd on AVX2 targets
        kama = (price - kama).mul_add(sc, kama);

        *out.get_unchecked_mut(i) = kama; // scalar store

        // Prefetch with in-bounds address (clamped)
        let pf = core::cmp::min(i + 128, data.len() - 1);
        _mm_prefetch(data.as_ptr().add(pf) as *const i8, _MM_HINT_T1);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512vl,fma")]
#[inline]
pub unsafe fn kama_avx512(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    use core::arch::x86_64::*;

    const ABS_MASK: i64 = 0x7FFF_FFFF_FFFF_FFFFu64 as i64;

    debug_assert!(period >= 2 && period <= data.len());
    debug_assert_eq!(data.len(), out.len());

    // ------------------------------------------------------------------ *
    // 1.  Σ|Δprice| for the first window                                  *
    // ------------------------------------------------------------------
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
            acc0 = _mm512_add_pd(acc0, abs_diff(base, j, mask_pd));
            acc1 = _mm512_add_pd(acc1, abs_diff(base, j + 8, mask_pd));
            acc2 = _mm512_add_pd(acc2, abs_diff(base, j + 16, mask_pd));
            acc3 = _mm512_add_pd(acc3, abs_diff(base, j + 24, mask_pd));
            j += 32;
        }
        let acc_all = _mm512_add_pd(_mm512_add_pd(acc0, acc1), _mm512_add_pd(acc2, acc3));
        sum_roc1 = _mm512_reduce_add_pd(acc_all); // horizontal sum

        while j <= lookback {
            sum_roc1 += (*base.add(j + 1) - *base.add(j)).abs();
            j += 1;
        }
    } else {
        for k in 0..=lookback {
            sum_roc1 += (*base.add(k + 1) - *base.add(k)).abs();
        }
    }

    // ------------------------------------------------------------------ *
    // 2.  Seed first output                                              *
    // ------------------------------------------------------------------
    let init_idx = first_valid + lookback + 1;
    let mut kama = *data.get_unchecked(init_idx);
    *out.get_unchecked_mut(init_idx) = kama;

    // ------------------------------------------------------------------ *
    // 3.  Rolling KAMA update (scalar recurrence)                        *
    // ------------------------------------------------------------------
    let const_max = 2.0 / 31.0;
    let const_diff = (2.0 / 3.0) - const_max;

    let mut tail_idx = first_valid;
    let mut tail_val = *data.get_unchecked(tail_idx);

    for i in (init_idx + 1)..data.len() {
        // ---- update Σ|Δp| ----
        let price = *data.get_unchecked(i);
        let new_diff = (price - *data.get_unchecked(i - 1)).abs();

        let next_tail = *data.get_unchecked(tail_idx + 1);
        let old_diff = (next_tail - tail_val).abs();
        sum_roc1 += new_diff - old_diff;

        tail_val = next_tail;
        tail_idx += 1;

        // ---- efficiency ratio & smoothing constant ----
        // Reuse next_tail to avoid an extra load
        let direction = (price - next_tail).abs();
        let er = if sum_roc1 == 0.0 {
            0.0
        } else {
            direction / sum_roc1
        };

        // fused multiply-add → one FMA µ-op; square via multiplication (faster than powi)
        let t = er.mul_add(const_diff, const_max);
        let sc = t * t;

        // ---- KAMA recurrence ----
        // compiler lowers mul_add to `vfmadd132sd` on AVX-512 targets
        kama = (price - kama).mul_add(sc, kama);

        *out.get_unchecked_mut(i) = kama; // regular scalar store (no NT store)

        // Prefetch with in-bounds address (clamped)
        let pf = core::cmp::min(i + 128, data.len() - 1);
        _mm_prefetch(data.as_ptr().add(pf) as *const i8, _MM_HINT_T1);
    }
}

// Decision: streaming uses fixed-size ring buffers for prices and |Δp| diffs.
// Rationale: removes growable deque overhead and branchiness; identical outputs.

#[derive(Debug, Clone)]
pub struct KamaStream {
    period: usize,

    // Fixed-size rings
    prices: Vec<f64>, // last `period` prices, oldest at head_p
    diffs: Vec<f64>,  // last `period` |Δp| diffs, oldest at head_d

    head_p: usize, // index of oldest price
    head_d: usize, // index of oldest diff
    count: usize,  // number of ingested samples
    seeded: bool,  // whether first KAMA was emitted

    prev_price: f64, // p_{t-1}
    prev_kama: f64,  // KAMA_{t-1}
    sum_roc1: f64,   // rolling Σ|Δp| over window

    // Smoothing constant parameters (Kaufman: slow=30, fast=2)
    const_max: f64,  // 2/(30+1)
    const_diff: f64, // (2/(2+1)) - const_max
}

impl KamaStream {
    #[inline]
    pub fn try_new(params: KamaParams) -> Result<Self, KamaError> {
        let period = params.period.unwrap_or(30);
        if period == 0 {
            return Err(KamaError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        Ok(Self {
            period,
            prices: vec![0.0; period],
            diffs: vec![0.0; period],
            head_p: 0,
            head_d: 0,
            count: 0,
            seeded: false,
            prev_price: 0.0,
            prev_kama: 0.0,
            sum_roc1: 0.0,
            const_max: 2.0 / (30.0 + 1.0),
            const_diff: (2.0 / (2.0 + 1.0)) - (2.0 / (30.0 + 1.0)),
        })
    }

    /// Push one price; returns `Some(kama)` after warmup, else `None`.
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        // First sample
        if self.count == 0 {
            self.prices[0] = value;
            self.prev_price = value;
            self.count = 1;
            return None;
        }

        // New absolute diff vs previous price
        let new_diff = (value - self.prev_price).abs();

        // Accumulate until we have `period` historical prices
        if self.count < self.period {
            // Store sequentially; keep rolling sum of |Δp|
            self.diffs[self.count - 1] = new_diff;
            self.sum_roc1 += new_diff;

            self.prices[self.count] = value;
            self.prev_price = value;
            self.count += 1;
            return None;
        }

        // Seed: emit first KAMA (equals current price), completing Σ|Δp| with edge diff
        if !self.seeded {
            self.sum_roc1 += new_diff;

            self.prev_kama = value;

            // finalize rings for steady-state sliding
            self.diffs[self.period - 1] = new_diff;

            // replace oldest price with current value and advance head
            self.prices[self.head_p] = value;
            self.head_p = if self.period == 1 { 0 } else { 1 };
            self.head_d = 0;

            self.prev_price = value;
            self.seeded = true;
            return Some(self.prev_kama);
        }

        // Steady-state updates
        let old_diff = self.diffs[self.head_d];
        self.sum_roc1 += new_diff - old_diff;

        let tail_price = self.prices[self.head_p];
        let direction = (value - tail_price).abs();
        let er = if self.sum_roc1 == 0.0 {
            0.0
        } else {
            // Use multiply-by-reciprocal form (one div + one mul)
            direction * (1.0 / self.sum_roc1)
        };
        let t = er.mul_add(self.const_diff, self.const_max);
        let sc = t * t;

        self.prev_kama = (value - self.prev_kama).mul_add(sc, self.prev_kama);

        // Slide rings
        self.diffs[self.head_d] = new_diff;
        self.head_d = if self.head_d + 1 == self.period {
            0
        } else {
            self.head_d + 1
        };

        self.prices[self.head_p] = value;
        self.head_p = if self.head_p + 1 == self.period {
            0
        } else {
            self.head_p + 1
        };

        self.prev_price = value;
        Some(self.prev_kama)
    }
}

#[derive(Clone, Debug)]
pub struct KamaBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for KamaBatchRange {
    fn default() -> Self {
        Self {
            period: (30, 240, 1),
        }
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
        _ => return Err(KamaError::InvalidKernelForBatch(k)),
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
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(30) == p.period.unwrap_or(30))
    }
    pub fn values_for(&self, p: &KamaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &KamaBatchRange) -> Result<Vec<KamaParams>, KamaError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, KamaError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            return Ok((start..=end).step_by(step).collect());
        }
        // reversed bounds supported by stepping down
        let mut v = Vec::new();
        let mut cur = start;
        loop {
            v.push(cur);
            match cur.checked_sub(step) {
                Some(next) if next >= end => {
                    cur = next;
                }
                _ => break,
            }
        }
        if v.is_empty() {
            Err(KamaError::InvalidRange { start, end, step })
        } else {
            Ok(v)
        }
    }
    let periods = axis_usize(r.period)?;
    let combos: Vec<KamaParams> = periods
        .into_iter()
        .map(|p| KamaParams { period: Some(p) })
        .collect();
    if combos.is_empty() {
        return Err(KamaError::InvalidRange {
            start: r.period.0,
            end: r.period.1,
            step: r.period.2,
        });
    }
    Ok(combos)
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
    let combos = expand_grid(sweep)?;
    let cols = data.len();
    let rows = combos.len();

    if cols == 0 {
        return Err(KamaError::EmptyInputData);
    }

    // Guard rows * cols multiplication before allocation
    let total_cells = rows
        .checked_mul(cols)
        .ok_or(KamaError::InvalidInput("rows*cols overflow"))?;

    // Step 1: Allocate uninitialized matrix
    let mut buf_mu = make_uninit_matrix(rows, cols);

    // Step 2: Calculate warmup periods for each row
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(KamaError::AllValuesNaN)?;

    // Validate that no period exceeds data length
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first <= max_p {
        return Err(KamaError::NotEnoughValidData {
            needed: max_p + 1,
            valid: data.len() - first,
        });
    }

    let warm: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap()).collect();

    // Step 3: Initialize NaN prefixes for each row
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    // Step 4: Convert to mutable slice for computation
    let mut buf_guard = ManuallyDrop::new(buf_mu);
    let out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    // Step 5: Compute into the buffer
    kama_batch_inner_into(data, sweep, kern, parallel, out)?;

    // Step 6: Reclaim as Vec<f64>
    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            total_cells,
            buf_guard.capacity(),
        )
    };

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
    let combos = expand_grid(sweep)?;
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(KamaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first <= max_p {
        return Err(KamaError::NotEnoughValidData {
            needed: max_p + 1,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();

    // Verify output slice length
    let expected = rows
        .checked_mul(cols)
        .ok_or(KamaError::InvalidInput("rows*cols overflow"))?;
    if out.len() != expected {
        return Err(KamaError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    // ---------------------------------------------------------------
    // 1.  Shared precompute for all rows: abs_delta and prefix sums
    //     abs_delta[t] = |p[t] - p[t-1]| with zeros before `first`
    //     prefix[t]    = Σ abs_delta[0..=t]
    // -------------------------------------------------------------
    let mut abs_delta = vec![0.0f64; cols];
    if cols > 1 {
        // No contribution before `first`
        for i in (first + 1)..cols {
            let a = data[i];
            let b = data[i - 1];
            abs_delta[i] = (a - b).abs();
        }
    }
    let mut prefix = vec![0.0f64; cols];
    let mut run = 0.0f64;
    for i in 0..cols {
        run += abs_delta[i];
        prefix[i] = run;
    }

    // ---------------------------------------------------------------
    // 2.  helper that fills a single row
    // -------------------------------------------------------------
    // We need to reinterpret the f64 slice as MaybeUninit for the row processing
    let out_uninit = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };

    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();

        // Cast the row slice (which is definitely ours to mutate) to f64
        let dst = core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        match kern {
            // Use row-specific scalar that leverages shared prefix sums for the initial Σ|Δp|
            Kernel::Scalar | Kernel::ScalarBatch => {
                kama_row_scalar_prefixed(data, &prefix, first, period, dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => kama_row_avx2(data, first, period, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => kama_row_avx512(data, first, period, dst),
            _ => kama_row_scalar(data, first, period, dst), // Fallback to scalar
        }
    };

    // ---------------------------------------------------------------
    // 3.  run every row kernel; no element is read before it is written
    // -------------------------------------------------------------
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
unsafe fn kama_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    kama_scalar(data, period, first, out)
}

#[inline(always)]
unsafe fn kama_row_scalar_prefixed(
    data: &[f64],
    prefix: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    let len = data.len();
    let lookback = period - 1;

    // Initial Σ|Δp| via prefix sums: Σ_{t=first..first+period-1} |p[t+1]-p[t]|
    let sum0 = prefix[first + period] - prefix[first];

    // Seed index and initial KAMA value
    let init_idx = first + lookback + 1; // == first + period
    let mut kama = data[init_idx];
    out[init_idx] = kama;

    // Rolling state
    let mut sum_roc1 = sum0;
    let const_max = 2.0 / 31.0;
    let const_diff = (2.0 / 3.0) - const_max;
    let mut trailing_idx = first;
    let mut trailing_value = data[trailing_idx];

    for i in (init_idx + 1)..len {
        let price_prev = data[i - 1];
        let price = data[i];

        // Update Σ|Δp| via ring update
        let next_tail = data[trailing_idx + 1];
        let old_diff = (next_tail - trailing_value).abs();
        let new_diff = (price - price_prev).abs();
        sum_roc1 += new_diff - old_diff;

        trailing_value = next_tail;
        trailing_idx += 1;

        // Efficiency ratio and smoothing constant
        let direction = (price - trailing_value).abs();
        let er = if sum_roc1 == 0.0 {
            0.0
        } else {
            direction / sum_roc1
        };
        let t = er.mul_add(const_diff, const_max);
        let sc = t * t;

        // KAMA recurrence
        kama = (price - kama).mul_add(sc, kama);
        out[i] = kama;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn kama_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    kama_avx2(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn kama_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    kama_avx512(data, period, first, out)
}

// Python bindings
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(all(feature = "python", feature = "cuda"))]
use numpy::PyReadonlyArray2;
#[cfg(feature = "python")]
use numpy::PyUntypedArrayMethods;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(all(feature = "python", feature = "cuda"))]
use pyo3::types::PyDict;

#[cfg(feature = "python")]
#[pyfunction(name = "kama")]
#[pyo3(signature = (data, period, kernel=None))]
pub fn kama_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = KamaParams {
        period: Some(period),
    };
    let kama_in = KamaInput::from_slice(slice_in, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| kama_with_kernel(&kama_in, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction(name = "kama_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
pub fn kama_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    // Check for all-NaN early before allocating output
    let first = slice_in
        .iter()
        .position(|x| !x.is_nan())
        .ok_or_else(|| PyValueError::new_err("kama: All values are NaN."))?;

    let sweep = KamaBatchRange {
        period: period_range,
    };

    let combos = expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = slice_in.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("kama_batch: rows*cols overflow"))?;

    let out_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    // Initialize NaN prefixes before computation
    let warm: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap()).collect();

    // Initialize NaN values for warmup periods
    for (row, &warmup) in warm.iter().enumerate() {
        let start = row * cols;
        let end = start + warmup.min(cols);
        for v in &mut slice_out[start..end] {
            *v = f64::NAN;
        }
    }

    let combos = py
        .allow_threads(|| {
            let kernel = match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                k => k,
            };
            let simd = match kernel {
                Kernel::Avx512Batch => Kernel::Avx512,
                Kernel::Avx2Batch => Kernel::Avx2,
                Kernel::ScalarBatch => Kernel::Scalar,
                // Handle non-batch kernels that might be returned
                Kernel::Scalar => Kernel::Scalar,
                Kernel::Avx2 => Kernel::Avx2,
                Kernel::Avx512 => Kernel::Avx512,
                _ => Kernel::Scalar, // Fallback
            };

            kama_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
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

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "kama_cuda_batch_dev")]
#[pyo3(signature = (data_f32, period_range, device_id=0))]
pub fn kama_cuda_batch_dev_py(
    py: Python<'_>,
    data_f32: PyReadonlyArray1<'_, f32>,
    period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<KamaDeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let slice_in = data_f32.as_slice()?;
    let sweep = KamaBatchRange {
        period: period_range,
    };

    let inner = py.allow_threads(|| {
        let cuda = CudaKama::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.kama_batch_dev(slice_in, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(KamaDeviceArrayF32Py { inner })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "kama_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, period, device_id=0))]
pub fn kama_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    data_tm_f32: PyReadonlyArray2<'_, f32>,
    period: usize,
    device_id: usize,
) -> PyResult<KamaDeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let flat_in = data_tm_f32.as_slice()?;
    let rows = data_tm_f32.shape()[0];
    let cols = data_tm_f32.shape()[1];
    let params = KamaParams {
        period: Some(period),
    };

    let inner = py.allow_threads(|| {
        let cuda = CudaKama::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.kama_many_series_one_param_time_major_dev(flat_in, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(KamaDeviceArrayF32Py { inner })
}

// KAMA-specific CUDA Array Interface v3 + DLPack (context-guarded handle)
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", unsendable)]
pub struct KamaDeviceArrayF32Py {
    pub(crate) inner: DeviceArrayF32Kama,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl KamaDeviceArrayF32Py {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("shape", (self.inner.rows, self.inner.cols))?;
        d.set_item("typestr", "<f4")?;
        d.set_item(
            "strides",
            (
                self.inner.cols * std::mem::size_of::<f32>(),
                std::mem::size_of::<f32>(),
            ),
        )?;
        d.set_item("data", (self.inner.device_ptr() as usize, false))?;
        // Kernels synchronize before returning; omit 'stream' per CAI v3.
        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        // 2 == kDLCUDA per DLPack
        (2, self.inner.device_id as i32)
    }

    #[pyo3(signature = (stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &self,
        py: Python<'py>,
        stream: Option<pyo3::PyObject>,
        max_version: Option<(u32, u32)>,
        _dl_device: Option<(i32, i32)>,
        _copy: Option<bool>,
    ) -> PyResult<PyObject> {
        use std::ffi::c_void;

        // We synchronize producer work before returning; we only accept None or allowed integers.
        if let Some(obj) = &stream {
            // Accept None implicitly; if provided and int==0 (disallowed), error.
            if let Ok(i) = obj.extract::<i64>(py) {
                if i == 0 {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "__dlpack__: stream 0 is disallowed for CUDA",
                    ));
                }
            }
        }

        #[repr(C)]
        struct DLDevice { device_type: i32, device_id: i32 }
        #[repr(C)]
        struct DLDataType { code: u8, bits: u8, lanes: u16 }
        #[repr(C)]
        struct DLTensor {
            data: *mut c_void,
            device: DLDevice,
            ndim: i32,
            dtype: DLDataType,
            shape: *mut i64,
            strides: *mut i64,
            byte_offset: u64,
        }
        #[repr(C)]
        struct DLManagedTensor {
            dl_tensor: DLTensor,
            manager_ctx: *mut c_void,
            deleter: Option<extern "C" fn(*mut DLManagedTensor)>,
        }
        #[repr(C)]
        struct DLPackVersion { major: u32, minor: u32 }
        #[repr(C)]
        struct DLManagedTensorVersioned {
            version: DLPackVersion,
            manager_ctx: *mut c_void,
            deleter: Option<extern "C" fn(*mut DLManagedTensorVersioned)>,
            flags: u64,
            dl_tensor: DLTensor,
        }
        struct DlpGuard {
            _shape: Box<[i64; 2]>,
            _strides: Box<[i64; 2]>,
            _ctx: std::sync::Arc<cust::context::Context>,
        }

        extern "C" fn legacy_managed_deleter(p: *mut DLManagedTensor) {
            unsafe {
                if p.is_null() { return; }
                let guard_ptr = (*p).manager_ctx as *mut DlpGuard;
                if !guard_ptr.is_null() {
                    drop(Box::from_raw(guard_ptr));
                }
                drop(Box::from_raw(p));
            }
        }
        extern "C" fn versioned_managed_deleter(p: *mut DLManagedTensorVersioned) {
            unsafe {
                if p.is_null() { return; }
                let guard_ptr = (*p).manager_ctx as *mut DlpGuard;
                if !guard_ptr.is_null() {
                    drop(Box::from_raw(guard_ptr));
                }
                drop(Box::from_raw(p));
            }
        }

        // Capsule destructor honoring rename-to-"used_*" rule per DLPack Python spec.
        unsafe extern "C" fn legacy_capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
            let name = b"dltensor\0";
            let ptr = pyo3::ffi::PyCapsule_GetPointer(
                capsule,
                name.as_ptr() as *const _,
            ) as *mut DLManagedTensor;
            if !ptr.is_null() {
                if let Some(del) = (*ptr).deleter { del(ptr); }
                let used = b"used_dltensor\0";
                pyo3::ffi::PyCapsule_SetName(capsule, used.as_ptr() as *const _);
            }
        }
        unsafe extern "C" fn versioned_capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
            let name = b"dltensor_versioned\0";
            let ptr = pyo3::ffi::PyCapsule_GetPointer(
                capsule,
                name.as_ptr() as *const _,
            ) as *mut DLManagedTensorVersioned;
            if !ptr.is_null() {
                if let Some(del) = (*ptr).deleter { del(ptr); }
                let used = b"used_dltensor_versioned\0";
                pyo3::ffi::PyCapsule_SetName(capsule, used.as_ptr() as *const _);
            }
        }

        let rows = self.inner.rows as i64;
        let cols = self.inner.cols as i64;
        let len = (rows as i128) * (cols as i128);
        let data_ptr: *mut c_void = if len == 0 { std::ptr::null_mut() } else { (self.inner.device_ptr() as usize) as *mut c_void };

        let shape = Box::new([rows, cols]);
        let strides = Box::new([cols, 1i64]); // strides are in elements for DLPack
        let guard = Box::new(DlpGuard { _shape: shape, _strides: strides, _ctx: self.inner.ctx.clone() });
        let guard_ptr = Box::into_raw(guard);
        let guard_ref = unsafe { &*guard_ptr };

        let want_versioned = matches!(max_version, Some((maj, _)) if maj >= 1);
        if want_versioned {
            let mt = Box::new(DLManagedTensorVersioned {
                version: DLPackVersion { major: 1, minor: 0 },
                manager_ctx: guard_ptr as *mut c_void,
                deleter: Some(versioned_managed_deleter),
                flags: 0,
                dl_tensor: DLTensor {
                    data: data_ptr,
                    device: DLDevice { device_type: 2, device_id: self.inner.device_id as i32 },
                    ndim: 2,
                    dtype: DLDataType { code: 2, bits: 32, lanes: 1 },
                    shape: guard_ref._shape.as_ptr() as *mut i64,
                    strides: guard_ref._strides.as_ptr() as *mut i64,
                    byte_offset: 0,
                },
            });
            let raw = Box::into_raw(mt) as *mut c_void;
            let name = b"dltensor_versioned\0";
            let cap = unsafe { pyo3::ffi::PyCapsule_New(raw, name.as_ptr() as *const _, Some(versioned_capsule_destructor)) };
            if cap.is_null() {
                unsafe { versioned_managed_deleter(raw as *mut DLManagedTensorVersioned); }
                return Err(pyo3::exceptions::PyRuntimeError::new_err("failed to create DLPack v1.x capsule"));
            }
            Ok(unsafe { PyObject::from_owned_ptr(py, cap) })
        } else {
            let mt = Box::new(DLManagedTensor {
                dl_tensor: DLTensor {
                    data: data_ptr,
                    device: DLDevice { device_type: 2, device_id: self.inner.device_id as i32 },
                    ndim: 2,
                    dtype: DLDataType { code: 2, bits: 32, lanes: 1 },
                    shape: guard_ref._shape.as_ptr() as *mut i64,
                    strides: guard_ref._strides.as_ptr() as *mut i64,
                    byte_offset: 0,
                },
                manager_ctx: guard_ptr as *mut c_void,
                deleter: Some(legacy_managed_deleter),
            });
            let raw = Box::into_raw(mt) as *mut c_void;
            let name = b"dltensor\0";
            let cap = unsafe { pyo3::ffi::PyCapsule_New(raw, name.as_ptr() as *const _, Some(legacy_capsule_destructor)) };
            if cap.is_null() {
                unsafe { legacy_managed_deleter(raw as *mut DLManagedTensor); }
                return Err(pyo3::exceptions::PyRuntimeError::new_err("failed to create legacy DLPack capsule"));
            }
            Ok(unsafe { PyObject::from_owned_ptr(py, cap) })
        }
    }
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

// ================== WASM Bindings ==================
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn kama_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = KamaParams {
        period: Some(period),
    };
    let input = KamaInput::from_slice(data, params);

    // Allocate output buffer once
    let mut output = vec![0.0; data.len()];

    // Compute directly into output buffer
    kama_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

// ================== Zero-Copy WASM Functions ==================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn kama_alloc(len: usize) -> *mut f64 {
    // Allocate memory for input/output buffer
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec); // Prevent deallocation
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn kama_free(ptr: *mut f64, len: usize) {
    // Free allocated memory
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn kama_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    // Check for null pointers
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to kama_into"));
    }

    unsafe {
        // Create slice from pointer
        let data = std::slice::from_raw_parts(in_ptr, len);

        // Validate inputs
        if period == 0 || period > len {
            return Err(JsValue::from_str("Invalid period"));
        }

        // Create KAMA input
        let params = KamaParams {
            period: Some(period),
        };
        let input = KamaInput::from_slice(data, params);

        // Check for aliasing (input and output buffers are the same)
        if in_ptr == out_ptr {
            // Use temporary buffer to avoid corruption during sliding window computation
            let mut temp = vec![0.0; len];
            kama_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            // Copy results back to output
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            // No aliasing, compute directly into output
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            kama_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

// ================== Batch Processing ==================

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct KamaBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct KamaBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<KamaParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = kama_batch)]
pub fn kama_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: KamaBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = KamaBatchRange {
        period: config.period_range,
    };

    let output = kama_batch_inner(data, &sweep, Kernel::Auto, false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = KamaBatchJsOutput {
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
pub fn kama_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to kama_batch_into"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        let sweep = KamaBatchRange {
            period: (period_start, period_end, period_step),
        };

        let combos = expand_grid(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let cols = len;

        let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);

        // Use optimized batch processing
        kama_batch_inner_into(data, &sweep, Kernel::Auto, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}

// Keep legacy batch function for backwards compatibility
#[cfg(feature = "wasm")]
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
    match kama_batch_slice(data, &sweep, Kernel::Auto) {
        Ok(output) => Ok(output.values),
        Err(e) => Err(JsValue::from_str(&format!("KAMA batch error: {}", e))),
    }
}

#[cfg(feature = "wasm")]
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
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

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
        assert!(
            res.is_err(),
            "[{}] KAMA should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_kama_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = KamaParams { period: Some(10) };
        let input = KamaInput::from_slice(&data_small, params);
        let res = kama_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] KAMA should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_kama_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = KamaParams { period: Some(30) };
        let input = KamaInput::from_slice(&single_point, params);
        let res = kama_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] KAMA should fail with insufficient data",
            test_name
        );
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
        let input = KamaInput::from_candles(&candles, "close", KamaParams { period: Some(30) });
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
            KamaParams {
                period: Some(period),
            },
        );
        let batch_output = kama_with_kernel(&input, kernel)?.values;
        let mut stream = KamaStream::try_new(KamaParams {
            period: Some(period),
        })?;
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

    fn check_kama_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Strategy: Generate period from 2 to 100, then data with length >= period+1
        let strat = (2usize..=100).prop_flat_map(|period| {
            (
                prop::collection::vec(
                    (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                    (period + 1)..400,
                ),
                Just(period),
            )
        });

        proptest::test_runner::TestRunner::default().run(&strat, |(data, period)| {
            let params = KamaParams {
                period: Some(period),
            };
            let input = KamaInput::from_slice(&data, params);

            // Compute KAMA with specified kernel and scalar reference
            let result = kama_with_kernel(&input, kernel);
            prop_assert!(
                result.is_ok(),
                "KAMA computation failed: {:?}",
                result.err()
            );
            let out = result.unwrap().values;

            let ref_result = kama_with_kernel(&input, Kernel::Scalar);
            prop_assert!(ref_result.is_ok(), "Reference KAMA failed");
            let ref_out = ref_result.unwrap().values;

            // Property 1: Output length matches input
            prop_assert_eq!(out.len(), data.len(), "Output length mismatch");

            // Find first valid data point
            let first_valid = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
            let warmup_end = first_valid + period;

            // Property 2: NaN values only in warmup period
            for i in 0..warmup_end.min(out.len()) {
                prop_assert!(
                    out[i].is_nan(),
                    "Expected NaN at index {} (warmup period), got {}",
                    i,
                    out[i]
                );
            }

            // Property 3: All values after warmup are finite
            for i in warmup_end..out.len() {
                prop_assert!(
                    out[i].is_finite(),
                    "Expected finite value at index {} (after warmup), got {}",
                    i,
                    out[i]
                );
            }

            // Property 4: KAMA values bounded by min/max of the trailing window
            for i in warmup_end..out.len() {
                // Get the relevant window for this KAMA value
                let window_start = i.saturating_sub(period);
                let window = &data[window_start..=i];
                let min_val = window.iter().cloned().fold(f64::INFINITY, f64::min);
                let max_val = window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                prop_assert!(
                    out[i] >= min_val - 1e-6 && out[i] <= max_val + 1e-6,
                    "KAMA at index {} = {} is outside window bounds [{}, {}]",
                    i,
                    out[i],
                    min_val,
                    max_val
                );
            }

            // Property 5: For constant data, KAMA should converge to that value
            if data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10) && !data.is_empty() {
                let const_val = data[first_valid];
                // After sufficient iterations, KAMA should be very close to the constant
                if out.len() > warmup_end + period * 2 {
                    let last_val = out[out.len() - 1];
                    prop_assert!(
                        (last_val - const_val).abs() < 1e-6,
                        "For constant data {}, KAMA should converge but got {}",
                        const_val,
                        last_val
                    );
                }
            }

            // Property 6: Kernel consistency (compare with scalar reference)
            for i in warmup_end..out.len() {
                let diff = (out[i] - ref_out[i]).abs();
                let ulps = {
                    if out[i] == ref_out[i] {
                        0
                    } else {
                        let a_bits = out[i].to_bits() as i64;
                        let b_bits = ref_out[i].to_bits() as i64;
                        (a_bits.wrapping_sub(b_bits)).unsigned_abs()
                    }
                };

                // Allow up to 10 ULPs difference for floating-point precision
                prop_assert!(
                    ulps <= 10 || diff < 1e-10,
                    "Kernel mismatch at index {}: {} vs {} (diff={}, ulps={})",
                    i,
                    out[i],
                    ref_out[i],
                    diff,
                    ulps
                );
            }

            // Property 7: Smoothness - KAMA changes bounded by maximum smoothing constant
            // KAMA formula: kama += (price - kama) * sc
            // Maximum sc = (2/3)^2 ≈ 0.445
            for i in (warmup_end + 1)..out.len() {
                let change = (out[i] - out[i - 1]).abs();
                let price = data[i];
                let prev_kama = out[i - 1];
                // Maximum possible change is when efficiency ratio = 1 (perfect trend)
                let max_possible_change = (price - prev_kama).abs() * 0.445;
                prop_assert!(
                    change <= max_possible_change + 1e-6,
                    "KAMA change {} exceeds maximum possible {} at index {}",
                    change,
                    max_possible_change,
                    i
                );
            }

            // Property 8: Monotonicity for monotonic data
            // If data is strictly increasing/decreasing after warmup, KAMA should follow
            let post_warmup_data = &data[warmup_end..];
            if post_warmup_data.len() > period {
                let is_increasing = post_warmup_data.windows(2).all(|w| w[1] >= w[0] - 1e-10);
                let is_decreasing = post_warmup_data.windows(2).all(|w| w[1] <= w[0] + 1e-10);

                if is_increasing {
                    for i in (warmup_end + period)..out.len() {
                        prop_assert!(
                            out[i] >= out[i - 1] - 1e-6,
                            "KAMA should be non-decreasing for increasing data at index {}",
                            i
                        );
                    }
                }
                if is_decreasing {
                    for i in (warmup_end + period)..out.len() {
                        prop_assert!(
                            out[i] <= out[i - 1] + 1e-6,
                            "KAMA should be non-increasing for decreasing data at index {}",
                            i
                        );
                    }
                }
            }

            // Property 9: Zero volatility case
            // When a window has identical values (sum_roc1 = 0), KAMA should not change
            // This tests the edge case where efficiency ratio = 0
            for i in (warmup_end + period)..out.len() {
                // Check if the last 'period' values are all the same
                let window_start = i - period + 1;
                let window = &data[window_start..=i];
                let all_same = window.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10);

                if all_same && i > warmup_end + period {
                    // When all values in window are the same, sum_roc1 = 0, so ER = 0
                    // This means smoothing constant is at minimum: (0 * const_diff + const_max)^2
                    // KAMA should barely change (only by minimum smoothing)
                    let change = (out[i] - out[i - 1]).abs();
                    let min_sc = (2.0 / 31.0_f64).powi(2); // ≈ 0.00416
                    let max_change = (data[i] - out[i - 1]).abs() * min_sc;
                    prop_assert!(
							change <= max_change + 1e-9,
							"With zero volatility at index {}, KAMA change {} exceeds minimum expected {}",
							i,
							change,
							max_change
						);
                }
            }

            Ok(())
        })?;

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
        check_kama_no_poison,
        check_kama_property
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
                .period_range(2, 200, 3) // Wide range: 2 to 200 with step 3
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

                if bits == 0x11111111_11111111
                    || bits == 0x22222222_22222222
                    || bits == 0x33333333_33333333
                {
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

    #[test]
    fn test_kama_into_matches_api() -> Result<(), Box<dyn Error>> {
        // Prepare a realistic input from the same dataset used elsewhere in this module
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = KamaParams::default();
        let input = KamaInput::from_candles(&candles, "close", params);

        // Baseline via Vec-returning API
        let base = kama(&input)?;

        // Preallocate output and compute via no-allocation API
        let mut out = vec![0.0f64; candles.close.len()];
        kama_into(&input, &mut out)?;

        // Length and element-wise equality (NaN == NaN permitted)
        assert_eq!(base.values.len(), out.len());
        for (a, b) in base.values.iter().zip(out.iter()) {
            let equal = (a.is_nan() && b.is_nan()) || (*a - *b).abs() <= 1e-12;
            assert!(equal, "Mismatch: base={} vs into={}", a, b);
        }
        Ok(())
    }
}
