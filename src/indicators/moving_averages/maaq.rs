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
//! ## Returns
//! - **Ok(MaaqOutput)** on success, with output values.
//! - **Err(MaaqError)** otherwise.
//!
//! ## Developer Status
//! - SIMD: AVX2 implemented (vectorized init |Δ| + scalar stream). AVX512 delegates to AVX2 to avoid downclock regressions.
//! - Speed: On 100k, AVX2/AVX512 ~2–3% faster than scalar; keep enabled but modest gains (stream is inherently serial).
//! - Streaming update: O(1) per update via rolling |Δ| sum; warmup parity preserved.
//! - Batch (row-specific): Not adopted; shared prefix-sum approach did not show >5% wins in this repo’s profile.
//! - Memory: Zero-copy/uninit helpers used (alloc_with_nan_prefix, make_uninit_matrix). Warmup parity with ALMA preserved.
//! - CUDA: Enabled wrappers; returns VRAM handles. Python interop provides CAI v3 (byte‑strides) and DLPack v1.x negotiation (versioned/legacy) with primary‑context RAII.

/// # WASM API Guide – MAAQ
///
/// This file exposes a dual-layer WebAssembly interface for the MAAQ
/// (Moving Average Adaptive Q) indicator, balancing **ergonomics** for
/// everyday users with **zero-copy throughput** for latency-critical code.
///
/// ---
/// ## 1 · Safe / Ergonomic API  <small>(recommended)</small>
/// | JS export | Rust impl | Purpose | Notes |
/// |-----------|-----------|---------|-------|
/// | `maaq_js(data, period, fast_period, slow_period)` | `maaq_js` | Single-parameter run | Returns a fresh `Vec<f64>` *without* an internal copy – the values are written directly into the return buffer before it is handed to JS. │
/// | `maaq_batch_js(data, config)` | `maaq_batch_js` | Grid sweep (legacy) | Returns flat values array only for backward compatibility. │
/// | `maaq_batch(data, config)`<br>(JS object) | `maaq_batch_unified_js` | Grid sweep over `(period, fast_period, slow_period)` | Accepts `period_range`, `fast_period_range`, `slow_period_range`; returns a flat result matrix plus combo metadata. │
///
/// **Characteristics**
/// * Memory-safe, runs under the default linear-memory quota.
/// * Adequate for charting & once-off indicator queries.
///
/// Example:
/// ```javascript
/// import * as wasm from './maaq_bg.wasm';
///
/// const y = wasm.maaq_js(prices, 11, 2, 30);
///
/// const grid = wasm.maaq_batch(prices, {
///   period_range: [10, 20, 5],
///   fast_period_range: [2, 4, 1],
///   slow_period_range: [20, 40, 10]
/// });
/// ```
///
/// ---
/// ## 2 · Fast / Unsafe API  <small>(zero-copy)</small>
/// | JS export | Rust impl | Purpose | Notes |
/// |-----------|-----------|---------|-------|
/// | `maaq_alloc(len)` / `maaq_free(ptr,len)` | `maaq_alloc`, `maaq_free` | Manual buffer lifecycle | Aligns to 8 bytes; caller **must** free. │
/// | `maaq_into(inPtr,outPtr,len,period,fast_period,slow_period)` | `maaq_into` | In-place single-run | Detects `inPtr === outPtr` and uses a temp scratch buffer to avoid alias corruption. │
/// | `maaq_batch_into(inPtr,outPtr,len,config)` | `maaq_batch_into` | In-place grid sweep | Serial on WASM for portability. │
///
/// **Performance**
/// * Zero heap allocations inside hot loops
/// * ~1.5×–2.0× faster than the safe API for repeated calls on pre-allocated
///   buffers (measured in Chrome 125, 10 k-point series, 100 updates/s).
///
/// **Caveats**
/// * **No bounds or lifetime checks** – treat pointers as raw FFI.
/// * Always wrap calls in `try { … } finally { free() }`.
/// * Recreate `TypedArray` views after *any* WASM call (memory may grow).
///
/// ```javascript
/// const n = prices.length;
/// const inPtr  = wasm.maaq_alloc(n);
/// const outPtr = wasm.maaq_alloc(n);
///
/// try {
///   new Float64Array(wasm.memory.buffer, inPtr,  n).set(prices);
///   wasm.maaq_into(inPtr, outPtr, n, 11, 2, 30);
///   const result = new Float64Array(wasm.memory.buffer, outPtr, n);
/// } finally {
///   wasm.maaq_free(inPtr,  n);
///   wasm.maaq_free(outPtr, n);
/// }
/// ```
///
/// ---
/// ## Porting other indicators
/// 1. Expose a safe `_js` wrapper returning a `Vec<f64>`.
/// 2. Provide `*_alloc` / `*_free` helpers.
/// 3. Add `*_into` for zero-copy execution (check `inPtr === outPtr`).
/// 4. Mirror the batch pattern if parameter sweeps are needed.
///
/// ---
/// ## Memory-safety checklist
/// 1. Guard every unsafe pointer with null-checks.
/// 2. Validate `period > 0 && period ≤ len` *before* slicing.
/// 3. Overwrite warm-up (prefix) indices with `NaN` in `*_into` helpers.
/// 4. Document warm-up length (`period – 1`) for stream consistency.
///
/// ---
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::cuda_available;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::CudaMaaq;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::maaq_wrapper::DeviceArrayF32Maaq;
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
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

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
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
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
    #[error("maaq: Input data slice is empty.")]
    EmptyInputData,
    #[error("maaq: All values are NaN.")]
    AllValuesNaN,
    #[error("maaq: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("maaq: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("maaq: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("maaq: Invalid range (start={start}, end={end}, step={step})")]
    InvalidRange { start: usize, end: usize, step: usize },
    #[error("maaq: Non-batch kernel passed to batch path: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error(
        "maaq: periods cannot be zero: period = {period}, fast = {fast_p}, slow = {slow_p}"
    )]
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

#[inline(always)]
fn maaq_compute_into(
    data: &[f64],
    period: usize,
    fast_p: usize,
    slow_p: usize,
    first: usize,
    kernel: Kernel,
    out: &mut [f64],
) -> Result<(), MaaqError> {
    // Guard possible caller misuse: ensure output slice is sized correctly
    if out.len() != data.len() {
        return Err(MaaqError::OutputLengthMismatch { expected: data.len(), got: out.len() });
    }
    unsafe {
        // Fallback to scalar if first > 0 since AVX implementations don't handle it properly
        if first > 0 {
            maaq_scalar(data, period, fast_p, slow_p, first, out)?;
        } else {
            match kernel {
                Kernel::Scalar | Kernel::ScalarBatch => {
                    maaq_scalar(data, period, fast_p, slow_p, first, out)?;
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2 | Kernel::Avx2Batch => {
                    maaq_avx2(data, period, fast_p, slow_p, first, out)?;
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512 | Kernel::Avx512Batch => {
                    maaq_avx512(data, period, fast_p, slow_p, first, out)?;
                }
                _ => unreachable!(),
            }
        }
    }
    Ok(())
}

#[inline(always)]
fn maaq_prepare<'a>(
    input: &'a MaaqInput,
    kernel: Kernel,
) -> Result<
    (
        // data
        &'a [f64],
        // period
        usize,
        // fast_p
        usize,
        // slow_p
        usize,
        // first
        usize,
        // chosen
        Kernel,
    ),
    MaaqError,
> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(MaaqError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(MaaqError::AllValuesNaN)?;

    let period = input.get_period();
    let fast_p = input.get_fast_period();
    let slow_p = input.get_slow_period();

    // Validation
    if period == 0 || fast_p == 0 || slow_p == 0 {
        return Err(MaaqError::ZeroPeriods {
            period,
            fast_p,
            slow_p,
        });
    }
    if period >= len {
        return Err(MaaqError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if len - first < period {
        return Err(MaaqError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    // Kernel auto-detection only once
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    Ok((data, period, fast_p, slow_p, first, chosen))
}

pub fn maaq_with_kernel(input: &MaaqInput, kernel: Kernel) -> Result<MaaqOutput, MaaqError> {
    let (data, period, fast_p, slow_p, first, chosen) = maaq_prepare(input, kernel)?;

    let warm = first + period - 1; // ALMA parity: prefix length
    let mut out = alloc_with_nan_prefix(data.len(), warm);

    // Sanity: output slice must match input length
    if out.len() != data.len() {
        return Err(MaaqError::OutputLengthMismatch { expected: data.len(), got: out.len() });
    }
    maaq_compute_into(data, period, fast_p, slow_p, first, chosen, &mut out)?;

    // ALMA parity: restore NaN prefix (implementations may have written raw prices)
    let warmup_end = first + period - 1;
    for v in &mut out[..warmup_end] {
        *v = f64::NAN;
    }

    Ok(MaaqOutput { values: out })
}

#[inline]
pub fn maaq_scalar(
    data: &[f64],
    period: usize,
    fast_p: usize,
    slow_p: usize,
    first: usize,
    out: &mut [f64],
) -> Result<(), MaaqError> {
    let len = data.len();
    let fast_sc = 2.0 / (fast_p as f64 + 1.0);
    let slow_sc = 2.0 / (slow_p as f64 + 1.0);

    // ring buffer of |Δ| for the current window
    let mut diffs = vec![0.0f64; period];
    let mut vol_sum = 0.0;
    // fill diffs[1..period-1] over data[first..first+period-1]
    for j in 1..period {
        let d = (data[first + j] - data[first + j - 1]).abs();
        diffs[j] = d;
        vol_sum += d;
    }

    // Copy raw prices to warmup area (to match AVX2)
    let warm_end = (first + period).min(len);
    if warm_end > first {
        out[first..warm_end].copy_from_slice(&data[first..warm_end]);
    }

    // first computable index (at period, not period-1, to match AVX2)
    let i0 = first + period;
    if i0 >= len {
        return Ok(());
    }

    // Add the final diff to complete the window
    let new_diff = (data[i0] - data[i0 - 1]).abs();
    diffs[0] = new_diff;
    vol_sum += new_diff;

    let mut prev_val = data[i0 - 1];
    let er0 = if vol_sum > f64::EPSILON {
        (data[i0] - data[first]).abs() / vol_sum
    } else {
        0.0
    };
    let mut sc = fast_sc.mul_add(er0, slow_sc);
    sc *= sc;
    // EMA-style update using fused multiply-add
    prev_val = sc.mul_add(data[i0] - prev_val, prev_val);
    out[i0] = prev_val;

    // head points to the oldest |Δ| (which is diffs[1] now)
    let mut head = if period > 1 { 1usize } else { 0usize };

    for i in (i0 + 1)..len {
        // roll window: drop oldest |Δ|, add newest
        vol_sum -= diffs[head];
        let nd = (data[i] - data[i - 1]).abs();
        diffs[head] = nd;
        vol_sum += nd;
        head += 1;
        if head == period {
            head = 0;
        }

        let er = if vol_sum > f64::EPSILON {
            (data[i] - data[i - period]).abs() / vol_sum
        } else {
            0.0
        };
        let mut sc = fast_sc.mul_add(er, slow_sc);
        sc *= sc;

        prev_val = sc.mul_add(data[i] - prev_val, prev_val);
        out[i] = prev_val;
    }
    Ok(())
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn maaq_avx2(
    data: &[f64],
    period: usize,
    fast_p: usize,
    slow_p: usize,
    first: usize,
    out: &mut [f64],
) -> Result<(), MaaqError> {
    use core::arch::x86_64::*;

    // ------ safety & basic checks ------------------------------------------------------------
    let len = data.len();
    debug_assert_eq!(len, out.len());
    if len == 0 {
        return Ok(());
    }

    // ------ 1 · pre-compute constants --------------------------------------------------------
    let fast_sc = 2.0 / (fast_p as f64 + 1.0);
    let slow_sc = 2.0 / (slow_p as f64 + 1.0);

    // SIMD abs helper: abs(x) = andnot(signmask, x), signmask = -0.0
    #[inline(always)]
    unsafe fn vabs_pd(x: __m256d) -> __m256d {
        let sign = _mm256_set1_pd(-0.0f64);
        _mm256_andnot_pd(sign, x)
    }

    #[inline(always)]
    fn fast_abs(x: f64) -> f64 {
        f64::from_bits(x.to_bits() & 0x7FFF_FFFF_FFFF_FFFF)
    }

    // ------ 2 · rolling-window buffers -------------------------------------------------------
    let mut diffs: Vec<f64> = Vec::with_capacity(period);
    unsafe {
        diffs.set_len(period);
    }
    let mut vol_sum = 0.0f64;

    unsafe {
        let dp = data.as_ptr();

        // Compute |Δ| for k = first+1 .. first+period-1 into diffs[1..]
        let base = first + 1;
        let n = period.saturating_sub(1);

        let mut accv = _mm256_setzero_pd();
        let mut j = 0usize;
        while j + 4 <= n {
            let a = _mm256_loadu_pd(dp.add(base + j));
            let b = _mm256_loadu_pd(dp.add(base + j - 1));
            let d = vabs_pd(_mm256_sub_pd(a, b));
            _mm256_storeu_pd(diffs.as_mut_ptr().add(1 + j), d);
            accv = _mm256_add_pd(accv, d);
            j += 4;
        }

        // Horizontal sum of accv
        let mut tmp = [0.0f64; 4];
        _mm256_storeu_pd(tmp.as_mut_ptr(), accv);
        vol_sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];

        // Tail
        while j < n {
            let k = base + j;
            let d = fast_abs(*dp.add(k) - *dp.add(k - 1));
            *diffs.get_unchecked_mut(1 + j) = d;
            vol_sum += d;
            j += 1;
        }

        // Copy warmup prices verbatim (ALMA parity)
        let warm_end = (first + period).min(len);
        if warm_end > first {
            core::ptr::copy_nonoverlapping(
                dp.add(first),
                out.as_mut_ptr().add(first),
                warm_end - first,
            );
        }

        // First computable index
        let i0 = first + period;
        if i0 >= len {
            return Ok(());
        }

        // Complete window by inserting final |Δ|
        let new_diff = fast_abs(*dp.add(i0) - *dp.add(i0 - 1));
        *diffs.get_unchecked_mut(0) = new_diff;
        vol_sum += new_diff;

        let mut prev_val = *dp.add(i0 - 1);
        let er0 = if vol_sum > f64::EPSILON {
            fast_abs(*dp.add(i0) - *dp.add(first)) / vol_sum
        } else {
            0.0
        };
        let mut sc = fast_sc.mul_add(er0, slow_sc);
        sc *= sc;
        prev_val = sc.mul_add(*dp.add(i0) - prev_val, prev_val);
        *out.get_unchecked_mut(i0) = prev_val;

        // Head points to oldest |Δ|
        let mut head = if period > 1 { 1usize } else { 0usize };

        // Main streaming loop (inherently serial)
        let mut i = i0 + 1;
        let op = out.as_mut_ptr();
        while i < len {
            vol_sum -= *diffs.get_unchecked(head);

            let nd = fast_abs(*dp.add(i) - *dp.add(i - 1));
            *diffs.get_unchecked_mut(head) = nd;
            vol_sum += nd;

            head += 1;
            if head == period {
                head = 0;
            }

            let er = if vol_sum > f64::EPSILON {
                fast_abs(*dp.add(i) - *dp.add(i - period)) / vol_sum
            } else {
                0.0
            };

            let mut sc = fast_sc.mul_add(er, slow_sc);
            sc *= sc;
            prev_val = sc.mul_add(*dp.add(i) - prev_val, prev_val);

            *op.add(i) = prev_val;
            i += 1;
        }
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
    // On many CPUs AVX-512 can downclock and underperform AVX2 for this workload.
    // Keep the AVX512 entry as a thin wrapper delegating to the AVX2 path.
    maaq_avx2(data, period, fast_p, slow_p, first, out)
}

// Streaming/Stateful MaaqStream  — O(1) per update
#[derive(Debug, Clone)]
pub struct MaaqStream {
    period: usize,
    fast_period: usize,
    slow_period: usize,
    buffer: Vec<f64>, // ring buffer of last `period` prices
    diff: Vec<f64>,   // ring buffer of |Δ| aligned with `buffer` writes
    head: usize,      // next write position (also the oldest element)
    filled: bool,
    last: f64,    // last output value (EMA state)
    count: usize, // number of values seen so far
    // ---- new for O(1) ----
    vol_sum: f64, // rolling sum of |Δ| in the window
    fast_sc: f64, // precomputed 2/(fast_period+1)
    slow_sc: f64, // precomputed 2/(slow_period+1)
}

impl MaaqStream {
    /// Decision: Streaming uses O(1) rolling |Δ| and FMA; matches batch post-warmup.
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

        let fast_sc = 2.0 / (fast_p as f64 + 1.0);
        let slow_sc = 2.0 / (slow_p as f64 + 1.0);

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
            vol_sum: 0.0,
            fast_sc,
            slow_sc,
        })
    }

    /// Push one sample; returns the stream value.
    /// Warmup semantics: return raw inputs for the first `period` samples (batch parity:
    /// those indices would be NaN), then switch to EMA-style MAAQ that matches batch output.
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        // ---- Warmup: fill buffers, accumulate vol_sum, echo input ----
        if !self.filled {
            let prev = if self.count > 0 {
                // previous input (for |Δ|)
                let idx_prev = (self.head + self.period - 1) % self.period;
                self.buffer[idx_prev]
            } else {
                value
            };
            let d = (value - prev).abs();

            self.buffer[self.head] = value;
            self.diff[self.head] = d;
            self.vol_sum += d;

            self.head += 1;
            if self.head == self.period {
                self.head = 0;
            }

            self.count += 1;
            self.last = value; // EMA state equals last input during warmup

            if self.count == self.period {
                self.filled = true;
            }
            return Some(value); // stream returns raw values during warmup
        }

        // ---- Steady state: O(1) update of Σ|Δ| and EMA ----

        // Previous input x_{t-1} for new |Δ|
        let idx_prev = (self.head + self.period - 1) % self.period;
        let prev_input = self.buffer[idx_prev];

        // Drop oldest |Δ|, insert newest |Δ|
        let old_diff = self.diff[self.head];
        self.vol_sum -= old_diff;

        let new_diff = (value - prev_input).abs();
        self.diff[self.head] = new_diff;
        self.vol_sum += new_diff;

        // Value leaving the window is x_{t-period}
        let old_value = self.buffer[self.head];

        // Overwrite buffer with the current input (advance ring)
        self.buffer[self.head] = value;
        self.head += 1;
        if self.head == self.period {
            self.head = 0;
        }

        // Efficiency ratio ER in [0, 1] (guard div-by-zero on flat windows)
        let er = if self.vol_sum > f64::EPSILON {
            (value - old_value).abs() / self.vol_sum
        } else {
            0.0
        };

        // SC = (er * fast + slow)^2  (use FMA and square)
        let mut sc = self.fast_sc.mul_add(er, self.slow_sc);
        sc *= sc;

        // EMA-style update: y = y_prev + sc * (x - y_prev)
        let out = sc.mul_add(value - self.last, self.last);
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
    // Note: AVX2/AVX512 batch underperforms scalar for MAAQ on our profile; short-circuit Auto to ScalarBatch.
    // SIMD code paths are left in place for future revisits.
    let kernel = match k {
        Kernel::Auto => Kernel::ScalarBatch,
        other if other.is_batch() => other,
        _ => {
            return Err(MaaqError::InvalidKernelForBatch(k))
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
pub fn expand_grid(r: &MaaqBatchRange) -> Vec<MaaqParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if start == end || step == 0 {
            return vec![start];
        }
        let mut v = Vec::new();
        if start < end {
            let mut x = start;
            while x <= end {
                v.push(x);
                match x.checked_add(step) {
                    Some(nx) if nx > x => x = nx,
                    _ => break,
                }
            }
        } else {
            let mut x = start;
            while x >= end {
                v.push(x);
                match x.checked_sub(step) {
                    Some(nx) if nx < x => x = nx,
                    _ => break,
                }
                if x == 0 { break; }
            }
        }
        v
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
        return Err(MaaqError::InvalidRange { start: 0, end: 0, step: 0 });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(MaaqError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(MaaqError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();

    if rows.checked_mul(cols).is_none() {
        return Err(MaaqError::InvalidRange { start: rows, end: cols, step: 0 });
    }

    // Per-row warm prefix: first non-NaN + that row's period - 1
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();

    // 1. allocate the matrix as MaybeUninit and write the NaN prefixes
    let mut raw = make_uninit_matrix(rows, cols);
    unsafe { init_matrix_prefixes(&mut raw, cols, &warm) };

    // 2. closure that fills one row; gets &mut [MaybeUninit<f64>]
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();
        let fast_p = combos[row].fast_period.unwrap();
        let slow_p = combos[row].slow_period.unwrap();

        // cast this row to &mut [f64]
        let out_row =
            core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        match kern {
            Kernel::Scalar => maaq_row_scalar(data, first, period, fast_p, slow_p, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => maaq_row_avx2(data, first, period, fast_p, slow_p, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => maaq_row_avx512(data, first, period, fast_p, slow_p, out_row),
            _ => unreachable!(),
        }
    };

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

    // 4. all elements are now initialised – convert to Vec<f64> safely
    let mut guard = core::mem::ManuallyDrop::new(raw);
    let values: Vec<f64> = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    // ---------- 5. package result ----------
    Ok(MaaqBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

/// Batch compute MAAQ directly into pre-allocated output slice (zero-copy)
pub fn maaq_batch_inner_into(
    data: &[f64],
    sweep: &MaaqBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<MaaqParams>, MaaqError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(MaaqError::InvalidRange { start: 0, end: 0, step: 0 });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(MaaqError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(MaaqError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();

    // Validate output slice size
    let expected = rows
        .checked_mul(cols)
        .ok_or(MaaqError::InvalidRange { start: rows, end: cols, step: 0 })?;
    if out.len() != expected {
        return Err(MaaqError::OutputLengthMismatch { expected, got: out.len() });
    }

    // Cast output slice to MaybeUninit
    let out_uninit = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };

    // Per-row warm prefix: first non-NaN + that row's period - 1
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();

    // 1. Write the NaN prefixes
    unsafe { init_matrix_prefixes(out_uninit, cols, &warm) };

    // 2. closure that fills one row
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();
        let fast_p = combos[row].fast_period.unwrap();
        let slow_p = combos[row].slow_period.unwrap();

        // cast this row to &mut [f64]
        let out_row =
            core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        match kern {
            Kernel::Scalar | Kernel::ScalarBatch => {
                maaq_row_scalar(data, first, period, fast_p, slow_p, out_row)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                maaq_row_avx2(data, first, period, fast_p, slow_p, out_row)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                maaq_row_avx512(data, first, period, fast_p, slow_p, out_row)
            }
            _ => unreachable!(),
        }
    };

    // 3. run every row, writing directly into output
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

// Row-specific batch kernels not attempted; no shared precompute to exploit

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
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

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

    fn check_maaq_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
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

    fn check_maaq_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
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
        // Compare only after warmup period
        // Batch uses NaN for warmup (ALMA parity): indices 0 to period-2 are NaN
        // Stream returns raw values during warmup then switches to MAAQ values
        // Both should produce the same MAAQ values after index period-1
        for i in period..batch_output.len() {
            let b = batch_output[i];
            let s = stream_values[i];
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

    // Check for poison values in single output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_maaq_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test multiple parameter combinations to better catch uninitialized memory bugs
        let test_cases = vec![
            // Default parameters
            MaaqParams::default(),
            // Small period with various fast/slow periods
            MaaqParams {
                period: Some(5),
                fast_period: Some(2),
                slow_period: Some(10),
            },
            MaaqParams {
                period: Some(8),
                fast_period: Some(3),
                slow_period: Some(20),
            },
            // Medium period combinations
            MaaqParams {
                period: Some(11),
                fast_period: Some(2),
                slow_period: Some(30),
            },
            MaaqParams {
                period: Some(15),
                fast_period: Some(4),
                slow_period: Some(40),
            },
            MaaqParams {
                period: Some(20),
                fast_period: Some(5),
                slow_period: Some(50),
            },
            // Large period
            MaaqParams {
                period: Some(30),
                fast_period: Some(6),
                slow_period: Some(60),
            },
            // Edge cases with fast_period close to period
            MaaqParams {
                period: Some(10),
                fast_period: Some(8),
                slow_period: Some(30),
            },
            // Very small fast_period
            MaaqParams {
                period: Some(25),
                fast_period: Some(1),
                slow_period: Some(100),
            },
        ];

        for params in test_cases {
            let input = MaaqInput::from_candles(&candles, "close", params.clone());
            let output = maaq_with_kernel(&input, kernel)?;

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
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} with params period={:?}, fast_period={:?}, slow_period={:?}",
                        test_name, val, bits, i, params.period, params.fast_period, params.slow_period
                    );
                }

                // Check for init_matrix_prefixes poison (0x22222222_22222222)
                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} with params period={:?}, fast_period={:?}, slow_period={:?}",
                        test_name, val, bits, i, params.period, params.fast_period, params.slow_period
                    );
                }

                // Check for make_uninit_matrix poison (0x33333333_33333333)
                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} with params period={:?}, fast_period={:?}, slow_period={:?}",
                        test_name, val, bits, i, params.period, params.fast_period, params.slow_period
                    );
                }
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_maaq_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
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
        check_maaq_streaming,
        check_maaq_no_poison
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

    // Check for poison values in batch output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test multiple batch configurations to better catch uninitialized memory bugs
        let test_configs = vec![
            // Small periods with various fast/slow combinations
            ((5, 10, 2), (2, 4, 1), (10, 30, 5)),
            // Medium periods
            ((10, 20, 5), (2, 6, 2), (20, 50, 10)),
            // Large periods
            ((20, 30, 5), (4, 8, 2), (40, 80, 20)),
            // Edge case: fast_period close to period
            ((10, 15, 5), (5, 10, 5), (30, 60, 30)),
            // Dense parameter sweep
            ((8, 12, 1), (2, 5, 1), (15, 25, 5)),
        ];

        for (period_range, fast_range, slow_range) in test_configs {
            let output = MaaqBatchBuilder::new()
                .kernel(kernel)
                .period_range(period_range.0, period_range.1, period_range.2)
                .fast_period_range(fast_range.0, fast_range.1, fast_range.2)
                .slow_period_range(slow_range.0, slow_range.1, slow_range.2)
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
                let params = &output.combos[row];

                // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (params: period={:?}, fast_period={:?}, slow_period={:?})",
                        test, val, bits, row, col, params.period, params.fast_period, params.slow_period
                    );
                }

                // Check for init_matrix_prefixes poison (0x22222222_22222222)
                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (params: period={:?}, fast_period={:?}, slow_period={:?})",
                        test, val, bits, row, col, params.period, params.fast_period, params.slow_period
                    );
                }

                // Check for make_uninit_matrix poison (0x33333333_33333333)
                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (params: period={:?}, fast_period={:?}, slow_period={:?})",
                        test, val, bits, row, col, params.period, params.fast_period, params.slow_period
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

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_maaq_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Strategy 1: Main strategy - test general properties with realistic data
        let main_strat = (
            proptest::collection::vec(
                (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                20..200,
            ),
            2usize..30,  // period
            1usize..10,  // fast_period
            10usize..50, // slow_period
        )
            .prop_filter("valid params", |(data, period, fast_p, slow_p)| {
                *period <= data.len() && *fast_p < *slow_p
            });

        proptest::test_runner::TestRunner::default().run(
            &main_strat,
            |(data, period, fast_p, slow_p)| {
                let params = MaaqParams {
                    period: Some(period),
                    fast_period: Some(fast_p),
                    slow_period: Some(slow_p),
                };
                let input = MaaqInput::from_slice(&data, params.clone());

                // Run with test kernel and scalar reference
                let result = maaq_with_kernel(&input, kernel)?;
                let reference = maaq_with_kernel(&input, Kernel::Scalar)?;

                // Property 1: Output length matches input
                prop_assert_eq!(
                    result.values.len(),
                    data.len(),
                    "Output length {} doesn't match input length {}",
                    result.values.len(),
                    data.len()
                );

                // Property 2: Warmup values should be NaN (ALMA parity)
                let first_valid = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
                let warmup_end = first_valid + period - 1;
                for i in 0..warmup_end.min(data.len()) {
                    prop_assert!(
                        result.values[i].is_nan(),
                        "Warmup value at {} should be NaN, got {}",
                        i,
                        result.values[i]
                    );
                }

                // Property 3: SIMD consistency with scalar (ULP tolerance)
                for i in 0..result.values.len() {
                    let y = result.values[i];
                    let r = reference.values[i];

                    if !y.is_finite() || !r.is_finite() {
                        prop_assert_eq!(
                            y.to_bits(),
                            r.to_bits(),
                            "NaN/Inf mismatch at {}: {} vs {}",
                            i,
                            y,
                            r
                        );
                        continue;
                    }

                    let ulp_diff = y.to_bits().abs_diff(r.to_bits());
                    prop_assert!(
                        (y - r).abs() <= 1e-9 || ulp_diff <= 5,
                        "SIMD mismatch at {}: {} vs {} (ULP={})",
                        i,
                        y,
                        r,
                        ulp_diff
                    );
                }

                // Property 4: Values stay within reasonable bounds
                // MAAQ uses adaptive smoothing bounded by EMA nature
                let data_min = data.iter().copied().fold(f64::INFINITY, f64::min);
                let data_max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let range = (data_max - data_min).abs();
                let tolerance = range * 0.02; // Allow only 2% tolerance since MAAQ is bounded

                for (i, &val) in result.values.iter().enumerate() {
                    if val.is_finite() && i >= period {
                        prop_assert!(
                            val >= data_min - tolerance && val <= data_max + tolerance,
                            "Value {} at index {} outside bounds [{}, {}]",
                            val,
                            i,
                            data_min - tolerance,
                            data_max + tolerance
                        );
                    }
                }

                // Property 5: Constant data produces stable output
                if data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10) && data.len() > period {
                    let constant_val = data[0];
                    for (i, &val) in result.values[period..].iter().enumerate() {
                        prop_assert!(
                            (val - constant_val).abs() < 1e-8,
                            "Constant data should produce constant output, got {} at index {}",
                            val,
                            i + period
                        );
                    }
                }

                Ok(())
            },
        )?;

        // Strategy 2: MAAQ-specific efficiency ratio and adaptive behavior
        let maaq_strat = (
            proptest::collection::vec(
                (-100f64..100f64).prop_filter("finite", |x| x.is_finite()),
                50..100,
            ),
            5usize..15,  // period
            1usize..5,   // fast_period
            20usize..40, // slow_period
        )
            .prop_filter("valid maaq params", |(_data, period, fast_p, slow_p)| {
                *fast_p < *slow_p
            });

        proptest::test_runner::TestRunner::default().run(
            &maaq_strat,
            |(data, period, fast_p, slow_p)| {
                let params = MaaqParams {
                    period: Some(period),
                    fast_period: Some(fast_p),
                    slow_period: Some(slow_p),
                };
                let input = MaaqInput::from_slice(&data, params);
                let result = maaq_with_kernel(&input, kernel)?;

                // Calculate efficiency ratios manually to verify MAAQ behavior
                let fast_sc = 2.0 / (fast_p as f64 + 1.0);
                let slow_sc = 2.0 / (slow_p as f64 + 1.0);

                // After warmup, verify adaptive smoothing behavior
                for i in (period + 1)..data.len() {
                    // Calculate efficiency ratio components
                    let signal = (data[i] - data[i - period]).abs();
                    let noise: f64 = (1..=period)
                        .map(|j| (data[i - j + 1] - data[i - j]).abs())
                        .sum();

                    // MAAQ property: efficiency ratio should be in [0, 1]
                    if noise > f64::EPSILON {
                        let er = signal / noise;
                        prop_assert!(
                            er >= 0.0 && er <= 1.0 + 1e-10,
                            "Efficiency ratio {} out of bounds at index {}",
                            er,
                            i
                        );

                        // When ER is high (trending), output should follow input more closely
                        // When ER is low (noisy), output should be smoother
                        let sc = (er * fast_sc + slow_sc).powi(2);

                        // Smoothing constant should be bounded
                        let min_sc = slow_sc.powi(2);
                        let max_sc = (fast_sc + slow_sc).powi(2);
                        prop_assert!(
                            sc >= min_sc - 1e-10 && sc <= max_sc + 1e-10,
                            "Smoothing constant {} out of bounds [{}..{}] at index {}",
                            sc,
                            min_sc,
                            max_sc,
                            i
                        );
                    }
                }

                // Test adaptive behavior: create trending vs noisy sections
                if data.len() >= period * 3 {
                    // For trending sections (large signal), should track closely
                    let trending_indices: Vec<usize> = (period..data.len())
                        .filter(|&i| {
                            let signal = (data[i] - data[i.saturating_sub(period)]).abs();
                            signal > 10.0 // Strong trend
                        })
                        .collect();

                    for &i in trending_indices.iter().take(5) {
                        let tracking_error = (result.values[i] - data[i]).abs();
                        let price_range = data[i.saturating_sub(period)..=i]
                            .iter()
                            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &v| {
                                (min.min(v), max.max(v))
                            });
                        let local_range = (price_range.1 - price_range.0).abs();

                        // In trending markets, MAAQ should track within 20% of local range
                        prop_assert!(
                            tracking_error <= local_range * 0.2 + 1.0,
                            "Poor tracking in trend at {}: error {} > 20% of range {}",
                            i,
                            tracking_error,
                            local_range
                        );
                    }
                }

                Ok(())
            },
        )?;

        // Strategy 3: Step response - test convergence to new levels
        let step_strat = (
            10usize..30,     // period
            2usize..5,       // fast_period
            20usize..40,     // slow_period
            -100f64..100f64, // initial level
            -100f64..100f64, // final level
        )
            .prop_filter("different levels", |(_p, _f, _s, init, final_level)| {
                (init - final_level).abs() > 1.0
            });

        proptest::test_runner::TestRunner::default().run(
            &step_strat,
            |(period, fast_p, slow_p, initial, final_level)| {
                // Create step function data
                let mut data = vec![initial; 50];
                data.extend(vec![final_level; 50]);

                let params = MaaqParams {
                    period: Some(period),
                    fast_period: Some(fast_p),
                    slow_period: Some(slow_p),
                };
                let input = MaaqInput::from_slice(&data, params);
                let result = maaq_with_kernel(&input, kernel)?;

                // Check that output eventually converges toward the new level
                let last_values = &result.values[90..];
                let convergence_target = final_level;

                for &val in last_values {
                    let distance_to_target = (val - convergence_target).abs();
                    let initial_distance = (initial - final_level).abs();

                    // Should be much closer to final than initial (stricter convergence)
                    prop_assert!(
                        distance_to_target < initial_distance * 0.3,
                        "Failed to converge: value {} too far from target {}",
                        val,
                        convergence_target
                    );
                }

                Ok(())
            },
        )?;

        // Strategy 4: Small data edge cases
        let small_strat = (
            proptest::collection::vec(
                (-100f64..100f64).prop_filter("finite", |x| x.is_finite()),
                1..5,
            ),
            1usize..3, // period
            1usize..3, // fast_period
            3usize..6, // slow_period
        )
            .prop_filter("valid small params", |(data, period, _fast_p, _slow_p)| {
                *period <= data.len()
            });

        proptest::test_runner::TestRunner::default().run(
            &small_strat,
            |(data, period, fast_p, slow_p)| {
                let params = MaaqParams {
                    period: Some(period),
                    fast_period: Some(fast_p),
                    slow_period: Some(slow_p),
                };
                let input = MaaqInput::from_slice(&data, params);

                // Should not panic with small data
                let result = maaq_with_kernel(&input, kernel)?;

                // Basic sanity checks
                prop_assert_eq!(result.values.len(), data.len());

                // First values should match input during warmup
                for i in 0..period.min(data.len()) {
                    if data[i].is_finite() {
                        prop_assert!(
                            (result.values[i] - data[i]).abs() < 1e-10,
                            "Small data warmup mismatch at {}",
                            i
                        );
                    }
                }

                Ok(())
            },
        )?;

        Ok(())
    }

    #[cfg(feature = "proptest")]
    generate_all_maaq_tests!(check_maaq_property);

    // Parity test for native into API vs Vec API
    #[cfg(not(feature = "wasm"))]
    #[test]
    fn test_maaq_into_matches_api() -> Result<(), Box<dyn Error>> {
        // Construct a series with a small NaN prefix and varying values
        let mut data: Vec<f64> = vec![f64::NAN, f64::NAN, f64::NAN];
        for i in 0..256u32 {
            let x = (i as f64).sin() * 0.5 + (i as f64) * 0.1 + ((i % 7) as f64) * 0.01;
            data.push(x);
        }

        let input = MaaqInput::from_slice(&data, MaaqParams::default());

        // Baseline via Vec-returning API
        let baseline = maaq(&input)?.values;

        // Zero-allocation path into preallocated buffer
        let mut out = vec![0.0; data.len()];
        super::maaq_into(&input, &mut out)?;

        assert_eq!(baseline.len(), out.len());

        // Equality: NaN == NaN, otherwise exact or tight epsilon
        for (idx, (a, b)) in baseline.iter().zip(out.iter()).enumerate() {
            let equal = (a.is_nan() && b.is_nan()) || ((a - b).abs() <= 1e-12);
            assert!(equal, "Mismatch at {}: {} vs {}", idx, a, b);
        }

        Ok(())
    }
}

// --- Python bindings ---
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;

// ---- CUDA Python interop helpers (primary-context guard) ----
#[cfg(all(feature = "python", feature = "cuda"))]
pub struct PrimaryCtxGuard {
    dev: i32,
    ctx: cust::sys::CUcontext,
}

#[cfg(all(feature = "python", feature = "cuda"))]
impl PrimaryCtxGuard {
    fn new(device_id: u32) -> Result<Self, cust::error::CudaError> {
        unsafe {
            let mut ctx: cust::sys::CUcontext = core::ptr::null_mut();
            let dev = device_id as i32;
            cust::sys::cuDevicePrimaryCtxRetain(&mut ctx as *mut _, dev)
                .result()
                .map_err(|e| cust::error::CudaError::UnknownError { msg: format!("cuDevicePrimaryCtxRetain: {:?}", e) })?;
            Ok(PrimaryCtxGuard { dev, ctx })
        }
    }
    #[inline]
    unsafe fn push_current(&self) {
        let _ = cust::sys::cuCtxSetCurrent(self.ctx);
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
impl Drop for PrimaryCtxGuard {
    fn drop(&mut self) {
        unsafe { let _ = cust::sys::cuDevicePrimaryCtxRelease_v2(self.dev); }
    }
}

// --- Python CUDA interop (MAAQ-specific device handle) ---
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", name = "DeviceArrayF32Maaq", unsendable)]
pub struct DeviceArrayF32MaaqPy {
    pub(crate) inner: Option<DeviceArrayF32Maaq>,
    device_id: u32,
    pc_guard: Option<PrimaryCtxGuard>,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl DeviceArrayF32MaaqPy {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("buffer already exported via __dlpack__"))?;
        let d = pyo3::types::PyDict::new(py);
        d.set_item("shape", (inner.rows, inner.cols))?;
        d.set_item("typestr", "<f4")?;
        d.set_item(
            "strides",
            (
                inner.cols * std::mem::size_of::<f32>(),
                std::mem::size_of::<f32>(),
            ),
        )?;
        let ptr_val: usize = if inner.rows == 0 || inner.cols == 0 { 0 } else { inner.device_ptr() as usize };
        d.set_item("data", (ptr_val, false))?;
        // Producing stream is synchronized before return; omit stream per CAI v3.
        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> PyResult<(i32, i32)> {
        Ok((2, self.device_id as i32))
    }

    #[pyo3(signature=(_stream=None, max_version=None, _dl_device=None, _copy=None))]
    fn __dlpack__<'py>(
        &mut self,
        py: Python<'py>,
        _stream: Option<pyo3::PyObject>,
        max_version: Option<pyo3::PyObject>,
        _dl_device: Option<pyo3::PyObject>,
        _copy: Option<pyo3::PyObject>,
    ) -> PyResult<PyObject> {
        use crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d;

        // Compute target device id and validate `dl_device` hint if provided.
        let (kdl, alloc_dev) = self.__dlpack_device__()?; // (2, device_id)
        if let Some(dev_obj) = _dl_device.as_ref() {
            if let Ok((dev_ty, dev_id)) = dev_obj.extract::<(i32, i32)>(py) {
                if dev_ty != kdl || dev_id != alloc_dev {
                    let wants_copy = _copy
                        .as_ref()
                        .and_then(|c| c.extract::<bool>(py).ok())
                        .unwrap_or(false);
                    if wants_copy {
                        return Err(PyValueError::new_err(
                            "device copy not implemented for __dlpack__",
                        ));
                    } else {
                        return Err(PyValueError::new_err("dl_device mismatch for __dlpack__"));
                    }
                }
            }
        }
        let _ = _stream;

        // Move VRAM handle out of this wrapper; the DLPack capsule owns it afterwards.
        let inner = self
            .inner
            .take()
            .ok_or_else(|| PyValueError::new_err("buffer already exported via __dlpack__"))?;
        let rows = inner.rows;
        let cols = inner.cols;
        let buf = inner.buf;

        let max_version_bound = max_version.map(|obj| obj.into_bound(py));

        export_f32_cuda_dlpack_2d(py, buf, rows, cols, alloc_dev, max_version_bound)
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
impl Drop for DeviceArrayF32MaaqPy {
    fn drop(&mut self) {
        if let Some(ref pc) = self.pc_guard {
            unsafe { pc.push_current(); }
        }
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "maaq")]
#[pyo3(signature = (data, period, fast_period, slow_period, kernel=None))]
pub fn maaq_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period: usize,
    fast_period: usize,
    slow_period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let kern = validate_kernel(kernel, false)?;
    let params = MaaqParams {
        period: Some(period),
        fast_period: Some(fast_period),
        slow_period: Some(slow_period),
    };

    // Prefer zero-copy for contiguous input; fallback to a single copy for non-contiguous views
    let result_vec: Vec<f64> = if let Ok(slice_in) = data.as_slice() {
        let input = MaaqInput::from_slice(slice_in, params);
        py.allow_threads(|| maaq_with_kernel(&input, kern).map(|o| o.values))
            .map_err(|e| PyValueError::new_err(e.to_string()))?
    } else {
        let owned = data.as_array().to_owned();
        let slice_in = owned.as_slice().expect("owned array should be contiguous");
        let input = MaaqInput::from_slice(slice_in, params);
        py.allow_threads(|| maaq_with_kernel(&input, kern).map(|o| o.values))
            .map_err(|e| PyValueError::new_err(e.to_string()))?
    };

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction(name = "maaq_batch")]
#[pyo3(signature = (data, period_range, fast_period_range, slow_period_range, kernel=None))]
pub fn maaq_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    fast_period_range: (usize, usize, usize),
    slow_period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?; // true for batch operations

    let sweep = MaaqBatchRange {
        period: period_range,
        fast_period: fast_period_range,
        slow_period: slow_period_range,
    };

    // Expand grid to calculate dimensions
    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();

    // Pre-allocate output array (OK for batch operations) with overflow guard
    let total = rows.checked_mul(cols).ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;
    let out_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    // Compute without GIL
    let combos = py
        .allow_threads(|| {
            // Handle kernel selection for batch operations
            let kernel = match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                k => k,
            };
            let simd = match kernel {
                Kernel::Avx512Batch => Kernel::Avx512,
                Kernel::Avx2Batch => Kernel::Avx2,
                Kernel::ScalarBatch => Kernel::Scalar,
                _ => kernel,
            };
            maaq_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
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
    dict.set_item(
        "fast_periods",
        combos
            .iter()
            .map(|p| p.fast_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "slow_periods",
        combos
            .iter()
            .map(|p| p.slow_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    Ok(dict)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "maaq_cuda_batch_dev")]
#[pyo3(signature = (data, period_range, fast_period_range, slow_period_range, device_id=0))]
pub fn maaq_cuda_batch_dev_py(
    py: Python<'_>,
    data: numpy::PyReadonlyArray1<'_, f64>,
    period_range: (usize, usize, usize),
    fast_period_range: (usize, usize, usize),
    slow_period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<DeviceArrayF32MaaqPy> {
    use numpy::PyArrayMethods;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let slice_in = data.as_slice()?;
    let sweep = MaaqBatchRange {
        period: period_range,
        fast_period: fast_period_range,
        slow_period: slow_period_range,
    };
    let data_f32: Vec<f32> = slice_in.iter().map(|&v| v as f32).collect();

    let inner = py.allow_threads(|| {
        let cuda = CudaMaaq::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.maaq_batch_dev_ex(&data_f32, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    let pc = PrimaryCtxGuard::new(device_id as u32).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(DeviceArrayF32MaaqPy { inner: Some(inner), device_id: device_id as u32, pc_guard: Some(pc) })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "maaq_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, period, fast_period, slow_period, device_id=0))]
pub fn maaq_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    data_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    period: usize,
    fast_period: usize,
    slow_period: usize,
    device_id: usize,
) -> PyResult<DeviceArrayF32MaaqPy> {
    use numpy::PyUntypedArrayMethods;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let flat_in = data_tm_f32.as_slice()?;
    let rows = data_tm_f32.shape()[0];
    let cols = data_tm_f32.shape()[1];
    let params = MaaqParams {
        period: Some(period),
        fast_period: Some(fast_period),
        slow_period: Some(slow_period),
    };

    let inner = py.allow_threads(|| {
        let cuda = CudaMaaq::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.maaq_multi_series_one_param_time_major_dev_ex(flat_in, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    let pc = PrimaryCtxGuard::new(device_id as u32).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(DeviceArrayF32MaaqPy { inner: Some(inner), device_id: device_id as u32, pc_guard: Some(pc) })
}

#[cfg(feature = "python")]
#[pyclass(name = "MaaqStream")]
pub struct MaaqStreamPy {
    stream: MaaqStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl MaaqStreamPy {
    #[new]
    pub fn new(period: usize, fast_period: usize, slow_period: usize) -> PyResult<Self> {
        let params = MaaqParams {
            period: Some(period),
            fast_period: Some(fast_period),
            slow_period: Some(slow_period),
        };
        let stream =
            MaaqStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

// ================== WASM Bindings ==================

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};

// ================== WASM Types ==================

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct MaaqBatchConfig {
    pub period_range: (usize, usize, usize),
    pub fast_period_range: (usize, usize, usize),
    pub slow_period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct MaaqBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<MaaqParams>,
    pub rows: usize,
    pub cols: usize,
}

// ================== Safe / Ergonomic API ==================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn maaq_js(
    data: &[f64],
    period: usize,
    fast_period: usize,
    slow_period: usize,
) -> Result<Vec<f64>, JsValue> {
    let params = MaaqParams {
        period: Some(period),
        fast_period: Some(fast_period),
        slow_period: Some(slow_period),
    };
    let input = MaaqInput::from_slice(data, params);

    let mut output = vec![0.0; data.len()]; // Single allocation
    maaq_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn maaq_batch_js(data: &[f64], config: JsValue) -> Result<Vec<f64>, JsValue> {
    let config: MaaqBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let range = MaaqBatchRange {
        period: config.period_range,
        fast_period: config.fast_period_range,
        slow_period: config.slow_period_range,
    };

    match maaq_batch_with_kernel(data, &range, Kernel::Auto) {
        Ok(output) => Ok(output.values),
        Err(e) => Err(JsValue::from_str(&e.to_string())),
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = maaq_batch)]
pub fn maaq_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: MaaqBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let range = MaaqBatchRange {
        period: config.period_range,
        fast_period: config.fast_period_range,
        slow_period: config.slow_period_range,
    };

    let output = maaq_batch_with_kernel(data, &range, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = MaaqBatchJsOutput {
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
pub fn maaq_batch_metadata_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
    fast_period_start: usize,
    fast_period_end: usize,
    fast_period_step: usize,
    slow_period_start: usize,
    slow_period_end: usize,
    slow_period_step: usize,
) -> Vec<f64> {
    let range = MaaqBatchRange {
        period: (period_start, period_end, period_step),
        fast_period: (fast_period_start, fast_period_end, fast_period_step),
        slow_period: (slow_period_start, slow_period_end, slow_period_step),
    };

    let combos = expand_grid(&range);
    let mut metadata = Vec::with_capacity(combos.len() * 3);

    for params in combos {
        metadata.push(params.period.unwrap_or(11) as f64);
        metadata.push(params.fast_period.unwrap_or(2) as f64);
        metadata.push(params.slow_period.unwrap_or(30) as f64);
    }

    metadata
}

// ================== Zero-Copy WASM Helper ==================

/// Write MAAQ values directly to output slice - no allocations
#[inline]
pub fn maaq_into_slice(dst: &mut [f64], input: &MaaqInput, kern: Kernel) -> Result<(), MaaqError> {
    let (data, period, fast_p, slow_p, first, chosen) = maaq_prepare(input, kern)?;

    // Verify output buffer size matches input
    if dst.len() != data.len() {
        return Err(MaaqError::OutputLengthMismatch { expected: data.len(), got: dst.len() });
    }

    // Compute MAAQ values directly into dst
    maaq_compute_into(data, period, fast_p, slow_p, first, chosen, dst)?;

    // ALMA parity: restore NaN prefix
    let warmup_end = first + period - 1;
    for v in &mut dst[..warmup_end] {
        *v = f64::NAN;
    }

    Ok(())
}

// ================== Native Into API (no allocations) ==================

/// Compute MAAQ directly into a caller-provided output buffer.
///
/// - Preserves NaN warmups exactly as the Vec-returning API.
/// - `out.len()` must equal the input length; returns an error on mismatch.
#[cfg(not(feature = "wasm"))]
#[inline]
pub fn maaq_into(input: &MaaqInput, out: &mut [f64]) -> Result<(), MaaqError> {
    maaq_into_slice(out, input, Kernel::Auto)
}

// ================== Fast / Unsafe API (Zero-Copy) ==================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn maaq_alloc(len: usize) -> *mut f64 {
    // Allocate memory for input/output buffer
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec); // Prevent deallocation
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn maaq_free(ptr: *mut f64, len: usize) {
    // Free allocated memory
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn maaq_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
    fast_period: usize,
    slow_period: usize,
) -> Result<(), JsValue> {
    // Check for null pointers
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to maaq_into"));
    }

    unsafe {
        // Create slice from pointer
        let data = std::slice::from_raw_parts(in_ptr, len);

        // Validate inputs
        if period == 0 || period > len {
            return Err(JsValue::from_str("Invalid period"));
        }
        if fast_period == 0 {
            return Err(JsValue::from_str("Invalid fast_period"));
        }
        if slow_period == 0 {
            return Err(JsValue::from_str("Invalid slow_period"));
        }

        // Calculate MAAQ
        let params = MaaqParams {
            period: Some(period),
            fast_period: Some(fast_period),
            slow_period: Some(slow_period),
        };
        let input = MaaqInput::from_slice(data, params);

        // Check for aliasing (input and output buffers are the same)
        if in_ptr == out_ptr {
            // Use temporary buffer to avoid corruption during sliding window computation
            let mut temp = vec![0.0; len];
            maaq_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            // Copy results back to output
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            // Direct computation into output buffer
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            maaq_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

// ================== Optimized Batch Processing ==================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn maaq_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    config: JsValue,
) -> Result<(), JsValue> {
    // Check for null pointers
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to maaq_batch_into"));
    }

    let config: MaaqBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        let range = MaaqBatchRange {
            period: config.period_range,
            fast_period: config.fast_period_range,
            slow_period: config.slow_period_range,
        };

        // Calculate output size
        let combos = expand_grid(&range);
        let total_size = combos.len() * len;

        // Check for aliasing
        if in_ptr == out_ptr {
            // Use temporary buffer
            let mut temp = vec![0.0; total_size];
            maaq_batch_inner_into(data, &range, Kernel::Auto, false, &mut temp)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            let out = std::slice::from_raw_parts_mut(out_ptr, total_size);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, total_size);
            maaq_batch_inner_into(data, &range, Kernel::Auto, false, out)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}
