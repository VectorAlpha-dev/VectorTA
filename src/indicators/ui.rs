//! # Ulcer Index (UI)
//!
//! The Ulcer Index (UI) is a volatility indicator that measures price drawdown from recent highs
//! and focuses on downside risk. It is calculated as the square root of the average of the squared
//! percentage drawdowns from the rolling maximum price within a specified window.
//!
//! ## Parameters
//! - **period**: Window size (number of data points), default 14.
//! - **scalar**: Multiplier applied to drawdown, default 100.0.
//!
//! ## Returns
//! - **Ok(UiOutput)** on success, containing a Vec<f64> matching the input length.
//! - **Err(UiError)** otherwise.
//!
//! ## Developer Notes
//! - Decision: scalar path optimized (monotonic deque + ring-sum). Bitmask-valid fast path for period ≤64 tested but disabled by default here due to neutral/slower results; revisit after profiling. Sqrt input clamped at 0 to avoid FP round-off negatives.
//! - Streaming: O(1) UiStream via monotonic deque (rolling max) + ring-sum (squared drawdowns); behavior matches scalar warmup and validity.
//! - SIMD status: AVX2/AVX512 stubs delegate to scalar (branch-heavy rolling max; SIMD underperforms). Kept for future work.
//! - Batch status: row-specific SIMD not attempted; algorithmic batch optimization groups rows by period and scales a base (scalar=1.0) series per row.
//! - Memory/Perf: uses `alloc_with_nan_prefix` and zero-copy outputs; warmup handling matches alma.rs patterns.
//! - CUDA status: GPU batch and many-series kernels are exposed via `CudaUi`, with primary-context–safe VRAM handles and Python interop through CAI v3 + DLPack v1.x.

#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};

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
use std::mem::MaybeUninit;
use thiserror::Error;

// CUDA Python bindings (VRAM DTO)
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::CudaUi;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::utilities::dlpack_cuda::{make_device_array_py, DeviceArrayF32Py};

impl<'a> AsRef<[f64]> for UiInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            UiData::Slice(slice) => slice,
            UiData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum UiData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct UiOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(serde::Serialize, serde::Deserialize))]
pub struct UiParams {
    pub period: Option<usize>,
    pub scalar: Option<f64>,
}

impl Default for UiParams {
    fn default() -> Self {
        Self {
            period: Some(14),
            scalar: Some(100.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct UiInput<'a> {
    pub data: UiData<'a>,
    pub params: UiParams,
}

impl<'a> UiInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: UiParams) -> Self {
        Self {
            data: UiData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: UiParams) -> Self {
        Self {
            data: UiData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", UiParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
    #[inline]
    pub fn get_scalar(&self) -> f64 {
        self.params.scalar.unwrap_or(100.0)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct UiBuilder {
    period: Option<usize>,
    scalar: Option<f64>,
    kernel: Kernel,
}

impl Default for UiBuilder {
    fn default() -> Self {
        Self {
            period: None,
            scalar: None,
            kernel: Kernel::Auto,
        }
    }
}

impl UiBuilder {
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
    pub fn scalar(mut self, s: f64) -> Self {
        self.scalar = Some(s);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<UiOutput, UiError> {
        let p = UiParams {
            period: self.period,
            scalar: self.scalar,
        };
        let i = UiInput::from_candles(c, "close", p);
        ui_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<UiOutput, UiError> {
        let p = UiParams {
            period: self.period,
            scalar: self.scalar,
        };
        let i = UiInput::from_slice(d, p);
        ui_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<UiStream, UiError> {
        let p = UiParams {
            period: self.period,
            scalar: self.scalar,
        };
        UiStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum UiError {
    #[error("ui: Empty input data")]
    EmptyInputData,
    #[error("ui: All values are NaN.")]
    AllValuesNaN,
    #[error("ui: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("ui: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("ui: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("ui: Invalid scalar: {scalar}")]
    InvalidScalar { scalar: f64 },
    #[error("ui: Invalid range: start = {start}, end = {end}, step = {step}")]
    InvalidRange { start: f64, end: f64, step: f64 },
    #[error("ui: Invalid kernel for batch operation. Expected batch kernel, got: {0:?}")]
    InvalidKernelForBatch(Kernel),
    // Back-compat aliases kept for older callers; new code paths do not emit them.
    #[error("ui: Empty input")]
    EmptyInput,
    #[error("ui: Invalid length: expected = {expected}, actual = {actual}")]
    InvalidLength { expected: usize, actual: usize },
}

#[inline]
pub fn ui(input: &UiInput) -> Result<UiOutput, UiError> {
    ui_with_kernel(input, Kernel::Auto)
}

pub fn ui_with_kernel(input: &UiInput, kernel: Kernel) -> Result<UiOutput, UiError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(UiError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| x.is_finite())
        .ok_or(UiError::AllValuesNaN)?;
    let period = input.get_period();
    let scalar = input.get_scalar();

    if period == 0 || period > len {
        return Err(UiError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(UiError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }
    if !scalar.is_finite() {
        return Err(UiError::InvalidScalar { scalar });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let span = period
        .checked_mul(2)
        .and_then(|v| v.checked_sub(2))
        .ok_or(UiError::InvalidRange {
            start: period as f64,
            end: len as f64,
            step: 2.0,
        })?;
    let warmup = first
        .checked_add(span)
        .ok_or(UiError::InvalidRange {
            start: period as f64,
            end: len as f64,
            step: 2.0,
        })?;
    let mut out = alloc_with_nan_prefix(len, warmup.min(len));

    // no extra clearing needed; prefix already set
    match chosen {
        Kernel::Scalar | Kernel::ScalarBatch => ui_scalar(data, period, scalar, first, &mut out),
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 | Kernel::Avx2Batch => ui_avx2(data, period, scalar, first, &mut out),
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 | Kernel::Avx512Batch => ui_avx512(data, period, scalar, first, &mut out),
        _ => ui_scalar(data, period, scalar, first, &mut out),
    }

    Ok(UiOutput { values: out })
}

/// Write Ulcer Index values into a caller-provided buffer without allocating.
///
/// - Preserves the exact NaN warmup prefix semantics of `ui()`/`ui_with_kernel()`.
/// - The output slice length must equal the input length.
/// - Uses `Kernel::Auto` dispatch for kernel selection.
#[cfg(not(feature = "wasm"))]
pub fn ui_into(input: &UiInput, out: &mut [f64]) -> Result<(), UiError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(UiError::EmptyInputData);
    }

    if out.len() != len {
        return Err(UiError::OutputLengthMismatch {
            expected: len,
            got: out.len(),
        });
    }

    let first = data
        .iter()
        .position(|x| x.is_finite())
        .ok_or(UiError::AllValuesNaN)?;
    let period = input.get_period();
    let scalar = input.get_scalar();

    if period == 0 || period > len {
        return Err(UiError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(UiError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }
    if !scalar.is_finite() {
        return Err(UiError::InvalidScalar { scalar });
    }

    let chosen = detect_best_kernel();

    // Prefill warmup prefix with the crate's quiet-NaN pattern to match alloc_with_nan_prefix
    let span = period
        .checked_mul(2)
        .and_then(|v| v.checked_sub(2))
        .ok_or(UiError::InvalidRange {
            start: period as f64,
            end: len as f64,
            step: 2.0,
        })?;
    let warmup = first
        .checked_add(span)
        .ok_or(UiError::InvalidRange {
            start: period as f64,
            end: len as f64,
            step: 2.0,
        })?;
    let nan_q = f64::from_bits(0x7ff8_0000_0000_0000);
    for v in &mut out[..warmup.min(len)] {
        *v = nan_q;
    }

    match chosen {
        Kernel::Scalar | Kernel::ScalarBatch => ui_scalar(data, period, scalar, first, out),
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 | Kernel::Avx2Batch => ui_avx2(data, period, scalar, first, out),
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 | Kernel::Avx512Batch => ui_avx512(data, period, scalar, first, out),
        _ => ui_scalar(data, period, scalar, first, out),
    }

    Ok(())
}

pub fn ui_scalar(data: &[f64], period: usize, scalar: f64, first: usize, out: &mut [f64]) {
    // Drop-in scalar kernel, aggressively optimized & loop-jammed.
    // - Monotonic rolling-max via a hand-rolled circular deque (no std::collections).
    // - Sliding sum via bitmask for period <= 64; byte ring otherwise.
    // - Unchecked indexing in the hot loop to remove bounds checks.
    // - Uses mul_add where appropriate to allow FMA.
    debug_assert_eq!(out.len(), data.len());
    let len = data.len();
    if len == 0 {
        return;
    }

    // Constants & warmup boundary
    let inv_period = 1.0 / (period as f64);
    let warmup_end = first + (period * 2 - 2);

    // --- Monotonic deque (indices) implemented as circular buffer ---
    // Capacity = period, size tracked explicitly (so head==tail is unambiguously empty).
    let cap = period;
    let mut deq: Vec<usize> = vec![0usize; cap];
    let mut head = 0usize; // index of current front
    let mut tail = 0usize; // slot for next push_back
    let mut dsize = 0usize;

    #[inline(always)]
    fn inc_wrap(x: &mut usize, cap: usize) {
        *x += 1;
        if *x == cap {
            *x = 0;
        }
    }
    #[inline(always)]
    fn dec_wrap(x: &mut usize, cap: usize) {
        if *x == 0 {
            *x = cap - 1;
        } else {
            *x -= 1;
        }
    }

    // --- Sliding window over last `period` squared drawdowns ---
    let mut sq_ring: Vec<f64> = vec![0.0f64; period];
    let mut ring_idx = 0usize;
    let mut sum = 0.0f64;
    let mut count = 0usize;

    // Fast path: validity via u64 bitmask when period <= 64
    // NOTE: Bitmask fast-path was experimentally slower on 10k/100k here;
    // keep code present but disabled for now.
    if false && period <= 64 {
        let mut valid_mask: u64 = 0;

        for i in first..len {
            // Window start index for rolling max
            let start = if i + 1 >= period { i + 1 - period } else { 0 };

            // Expire stale indices from the front
            while dsize != 0 {
                let j = unsafe { *deq.get_unchecked(head) };
                if j < start {
                    inc_wrap(&mut head, cap);
                    dsize -= 1;
                } else {
                    break;
                }
            }

            // Push current index if finite, maintaining descending values in deque
            let xi = unsafe { *data.get_unchecked(i) };
            let xi_finite = xi.is_finite();
            if xi_finite {
                while dsize != 0 {
                    let mut back = tail;
                    dec_wrap(&mut back, cap);
                    let j = unsafe { *deq.get_unchecked(back) };
                    let xj = unsafe { *data.get_unchecked(j) };
                    if xj <= xi {
                        // Pop back
                        tail = back;
                        dsize -= 1;
                    } else {
                        break;
                    }
                }
                // Push back current index
                unsafe { *deq.get_unchecked_mut(tail) = i };
                inc_wrap(&mut tail, cap);
                dsize += 1;
            }

            // Compute squared drawdown when the first rolling max is available
            let mut new_valid = false;
            let mut new_sq: f64 = 0.0;
            if i + 1 >= first + period && dsize != 0 {
                let jmax = unsafe { *deq.get_unchecked(head) };
                let m = unsafe { *data.get_unchecked(jmax) };
                if xi_finite && m.is_finite() && m.abs() > f64::EPSILON {
                    // dd = scalar * (xi - m) / m
                    let dd = (xi - m) * (scalar / m);
                    new_sq = dd.mul_add(dd, 0.0);
                    new_valid = true;
                }
            }

            // Update sliding sum/ring via bitmask
            let bit = 1u64 << ring_idx;
            if (valid_mask & bit) != 0 {
                sum -= unsafe { *sq_ring.get_unchecked(ring_idx) };
                count -= 1;
                valid_mask &= !bit;
            }
            if new_valid {
                sum += new_sq;
                count += 1;
                valid_mask |= bit;
            }
            unsafe { *sq_ring.get_unchecked_mut(ring_idx) = new_sq };

            ring_idx += 1;
            if ring_idx == period {
                ring_idx = 0;
            }

            // Emit only after full warmup; always write (avoid leaving poison)
            if i >= warmup_end {
                let dst = unsafe { out.get_unchecked_mut(i) };
                if count == period {
                    let mut avg = sum * inv_period;
                    if avg < 0.0 {
                        avg = 0.0; // guard tiny negative due to FP round-off
                    }
                    *dst = avg.sqrt();
                } else {
                    *dst = f64::NAN;
                }
            }
        }
        return;
    }

    // Fallback: byte-valid ring for period > 64
    let mut valid_ring: Vec<u8> = vec![0u8; period];

    // Hot loop
    // SAFETY: We perform explicit bounds checks on loop limits; inner accesses use unchecked.
    for i in first..len {
        // Window start index for rolling max
        let start = if i + 1 >= period { i + 1 - period } else { 0 };

        // Expire stale indices from the front
        while dsize != 0 {
            let j = unsafe { *deq.get_unchecked(head) };
            if j < start {
                inc_wrap(&mut head, cap);
                dsize -= 1;
            } else {
                break;
            }
        }

        // Push current index if finite, maintaining descending values in deque
        let xi = unsafe { *data.get_unchecked(i) };
        let xi_finite = xi.is_finite();
        if xi_finite {
            while dsize != 0 {
                let mut back = tail;
                dec_wrap(&mut back, cap);
                let j = unsafe { *deq.get_unchecked(back) };
                let xj = unsafe { *data.get_unchecked(j) };
                if xj <= xi {
                    // Pop back
                    tail = back;
                    dsize -= 1;
                } else {
                    break;
                }
            }
            // Push back current index
            unsafe { *deq.get_unchecked_mut(tail) = i };
            inc_wrap(&mut tail, cap);
            dsize += 1;
        }

        // Compute squared drawdown when the first rolling max is available
        let mut new_valid: u8 = 0;
        let mut new_sq: f64 = 0.0;

        if i + 1 >= first + period && dsize != 0 {
            let jmax = unsafe { *deq.get_unchecked(head) };
            let m = unsafe { *data.get_unchecked(jmax) };
            // Guard against zero/denormal and propagate NaNs/Infs
            if xi_finite && m.is_finite() && m.abs() > f64::EPSILON {
                // dd = scalar * (xi - m) / m
                // Use pre-scaled inverse and FMA-friendly square
                let scaled = scalar / m;
                let diff = xi - m;
                let dd = diff * scaled;
                new_sq = dd.mul_add(dd, 0.0);
                new_valid = 1;
            }
        }

        // Update sliding sum/ring in-place (drop old, add new)
        let old_valid = unsafe { *valid_ring.get_unchecked(ring_idx) };
        if old_valid != 0 {
            sum -= unsafe { *sq_ring.get_unchecked(ring_idx) };
            count -= 1;
        }
        if new_valid != 0 {
            sum += new_sq;
            count += 1;
        }
        unsafe {
            *sq_ring.get_unchecked_mut(ring_idx) = new_sq;
            *valid_ring.get_unchecked_mut(ring_idx) = new_valid;
        }
        ring_idx += 1;
        if ring_idx == period {
            ring_idx = 0;
        }

        // Emit only after full warmup; always write (avoid leaving poison)
        if i >= warmup_end {
            let dst = unsafe { out.get_unchecked_mut(i) };
            if count == period {
                let mut avg = sum * inv_period;
                if avg < 0.0 {
                    avg = 0.0; // guard tiny negative due to FP round-off
                }
                *dst = avg.sqrt();
            } else {
                *dst = f64::NAN;
            }
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn ui_avx512(data: &[f64], period: usize, scalar: f64, first: usize, out: &mut [f64]) {
    unsafe { ui_avx512_short(data, period, scalar, first, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn ui_avx2(data: &[f64], period: usize, scalar: f64, first: usize, out: &mut [f64]) {
    ui_scalar(data, period, scalar, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn ui_avx512_short(
    data: &[f64],
    period: usize,
    scalar: f64,
    first: usize,
    out: &mut [f64],
) {
    ui_scalar(data, period, scalar, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn ui_avx512_long(
    data: &[f64],
    period: usize,
    scalar: f64,
    first: usize,
    out: &mut [f64],
) {
    ui_scalar(data, period, scalar, first, out)
}

#[derive(Debug, Clone)]
pub struct UiStream {
    period: usize,
    scalar: f64,

    // Global stream index of the next value to write (0-based)
    i: usize,

    // Index of the first finite input we have seen (to mirror 'first' in batch/scalar)
    first_finite: Option<usize>,
    // Computed lazily as: first + (period * 2 - 2); mirrors scalar warmup gate
    warmup_end: Option<usize>,

    // --- Rolling max (price) via monotonic deque of indices (circular buffer) ---
    // We store indices into the *global* stream. To retrieve a value for index j,
    // use buffer[j % period]. We only keep indices from the last 'period' samples.
    buffer: Vec<f64>, // ring of last 'period' prices; written at i % period
    deq: Vec<usize>,  // circular array of size 'period' holding indices
    dq_head: usize,   // front pointer
    dq_tail: usize,   // next write position (one past back)
    dq_size: usize,   // number of valid entries

    // --- Last 'period' squared drawdowns (ring) + running sum for O(1) avg ---
    sq_ring: Vec<f64>, // squared drawdowns ring buffer
    ring_idx: usize,   // position to overwrite in sq_ring/valid ring

    // Two validity tracking modes: bitmask (<=64) or byte-ring (>64)
    valid_mask: u64,             // used when period <= 64
    valid_ring: Option<Vec<u8>>, // used when period > 64

    sum_sq: f64,        // running sum of squared drawdowns in the ring
    count_valid: usize, // number of valid slots in the ring (must reach 'period')
}

impl UiStream {
    pub fn try_new(params: UiParams) -> Result<Self, UiError> {
        let period = params.period.unwrap_or(14);
        let scalar = params.scalar.unwrap_or(100.0);
        if period == 0 {
            return Err(UiError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        if !scalar.is_finite() {
            return Err(UiError::InvalidScalar { scalar });
        }

        let use_mask = period <= 64;
        Ok(Self {
            period,
            scalar,

            i: 0,
            first_finite: None,
            warmup_end: None,

            buffer: vec![f64::NAN; period],
            deq: vec![0usize; period],
            dq_head: 0,
            dq_tail: 0,
            dq_size: 0,

            sq_ring: vec![0.0; period],
            ring_idx: 0,

            valid_mask: 0,
            valid_ring: (!use_mask).then(|| vec![0u8; period]),

            sum_sq: 0.0,
            count_valid: 0,
        })
    }

    #[inline(always)]
    fn dq_inc(x: &mut usize, cap: usize) {
        *x += 1;
        if *x == cap {
            *x = 0;
        }
    }
    #[inline(always)]
    fn dq_dec(x: &mut usize, cap: usize) {
        if *x == 0 {
            *x = cap - 1;
        } else {
            *x -= 1;
        }
    }

    /// O(1) amortized update.
    /// Returns Some(UI) when we have a fully valid window; otherwise None (during warmup or invalid data).
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        let p = self.period;
        let cap = p; // deque capacity

        // 1) Write incoming value into the price ring (at position i % p)
        let pos = self.i % p;
        self.buffer[pos] = value;

        // Track the first finite index to mirror 'first' from batch/scalar.
        if self.first_finite.is_none() && value.is_finite() {
            let f = self.i;
            self.first_finite = Some(f);
            self.warmup_end = Some(f + (p * 2 - 2));
        }

        // 2) Expire stale indices from monotonic deque front (indices older than start)
        let start = if self.i + 1 >= p { self.i + 1 - p } else { 0 };
        while self.dq_size != 0 {
            let j = unsafe { *self.deq.get_unchecked(self.dq_head) };
            if j < start {
                Self::dq_inc(&mut self.dq_head, cap);
                self.dq_size -= 1;
            } else {
                break;
            }
        }

        // 3) Push current index into deque (maintain descending values)
        let xi = value;
        let xi_finite = xi.is_finite();
        if xi_finite {
            while self.dq_size != 0 {
                let mut back = self.dq_tail;
                Self::dq_dec(&mut back, cap);
                let j = unsafe { *self.deq.get_unchecked(back) };
                let xj = unsafe { *self.buffer.get_unchecked(j % p) };
                if xj <= xi {
                    // Pop back
                    self.dq_tail = back;
                    self.dq_size -= 1;
                } else {
                    break;
                }
            }
            unsafe {
                *self.deq.get_unchecked_mut(self.dq_tail) = self.i;
            }
            Self::dq_inc(&mut self.dq_tail, cap);
            self.dq_size += 1;
        }

        // 4) Compute new squared drawdown once the first rolling max is attainable
        let mut new_valid = false;
        let mut new_sq = 0.0f64;

        if let Some(first) = self.first_finite {
            if self.i + 1 >= first + p && self.dq_size != 0 {
                let jmax = unsafe { *self.deq.get_unchecked(self.dq_head) };
                let m = unsafe { *self.buffer.get_unchecked(jmax % p) };
                if xi_finite && m.is_finite() && m.abs() > f64::EPSILON {
                    // dd = scalar * (xi - m) / m
                    let dd = (xi - m) * (self.scalar / m);
                    // Square via fused mul-add when available
                    new_sq = dd.mul_add(dd, 0.0);
                    new_valid = true;
                }
            }
        }

        // 5) Update sliding sum/ring in O(1)
        if self.period <= 64 {
            let bit = 1u64 << self.ring_idx;
            if (self.valid_mask & bit) != 0 {
                self.sum_sq -= unsafe { *self.sq_ring.get_unchecked(self.ring_idx) };
                self.count_valid -= 1;
                self.valid_mask &= !bit;
            }
            if new_valid {
                self.sum_sq += new_sq;
                self.count_valid += 1;
                self.valid_mask |= bit;
            }
        } else {
            let vr = self.valid_ring.as_mut().unwrap();
            let was = unsafe { *vr.get_unchecked(self.ring_idx) };
            if was != 0 {
                self.sum_sq -= unsafe { *self.sq_ring.get_unchecked(self.ring_idx) };
                self.count_valid -= 1;
            }
            if new_valid {
                self.sum_sq += new_sq;
                self.count_valid += 1;
            }
            unsafe {
                *vr.get_unchecked_mut(self.ring_idx) = if new_valid { 1 } else { 0 };
            }
        }
        unsafe {
            *self.sq_ring.get_unchecked_mut(self.ring_idx) = new_sq;
        }
        self.ring_idx += 1;
        if self.ring_idx == p {
            self.ring_idx = 0;
        }

        // Bump global index after using it everywhere
        let i_now = self.i;
        self.i = self.i.wrapping_add(1);

        // 6) Emit UI once both (a) warmup gate passed and (b) window fully valid
        if let Some(we) = self.warmup_end {
            if i_now >= we && self.count_valid == p {
                let mut avg = self.sum_sq / (p as f64);
                if avg < 0.0 {
                    avg = 0.0;
                } // clamp tiny negatives from FP error
                  // NOTE: If you enable a fast-math sqrt, call it here instead.
                return Some(avg.sqrt());
            }
        }
        None
    }
}

#[derive(Clone, Debug)]
pub struct UiBatchRange {
    pub period: (usize, usize, usize),
    pub scalar: (f64, f64, f64),
}

impl Default for UiBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 263, 1),
            scalar: (100.0, 100.0, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct UiBatchBuilder {
    range: UiBatchRange,
    kernel: Kernel,
}

impl UiBatchBuilder {
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
    pub fn scalar_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.scalar = (start, end, step);
        self
    }
    #[inline]
    pub fn scalar_static(mut self, s: f64) -> Self {
        self.range.scalar = (s, s, 0.0);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<UiBatchOutput, UiError> {
        ui_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<UiBatchOutput, UiError> {
        UiBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<UiBatchOutput, UiError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<UiBatchOutput, UiError> {
        UiBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn ui_batch_with_kernel(
    data: &[f64],
    sweep: &UiBatchRange,
    k: Kernel,
) -> Result<UiBatchOutput, UiError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => {
            return Err(UiError::InvalidKernelForBatch(other));
        }
    };

    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => Kernel::Scalar, // Default to scalar for any other kernel
    };
    ui_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct UiBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<UiParams>,
    pub rows: usize,
    pub cols: usize,
}

impl UiBatchOutput {
    pub fn row_for_params(&self, p: &UiParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(14) == p.period.unwrap_or(14)
                && (c.scalar.unwrap_or(100.0) - p.scalar.unwrap_or(100.0)).abs() < 1e-12
        })
    }
    pub fn values_for(&self, p: &UiParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &UiBatchRange) -> Result<Vec<UiParams>, UiError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, UiError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            let vals: Vec<usize> = (start..=end).step_by(step).collect();
            if vals.is_empty() {
                return Err(UiError::InvalidRange {
                    start: start as f64,
                    end: end as f64,
                    step: step as f64,
                });
            }
            Ok(vals)
        } else {
            // Support reversed bounds by walking [end, start] and reversing.
            let mut v: Vec<usize> = (end..=start).step_by(step).collect();
            if v.is_empty() {
                return Err(UiError::InvalidRange {
                    start: start as f64,
                    end: end as f64,
                    step: step as f64,
                });
            }
            v.reverse();
            Ok(v)
        }
    }
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, UiError> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return Ok(vec![start]);
        }
        // Guard against invalid step direction that would not converge.
        if (start < end && step <= 0.0) || (start > end && step >= 0.0) {
            return Err(UiError::InvalidRange { start, end, step });
        }
        let mut v = Vec::new();
        let mut x = start;
        let max_iterations: usize = 10_000; // Safety limit
        let mut iterations: usize = 0;
        if start < end {
            while x <= end + 1e-12 {
                if iterations >= max_iterations {
                    return Err(UiError::InvalidRange { start, end, step });
                }
                v.push(x);
                x += step;
                iterations += 1;
            }
        } else {
            // Reversed bounds with negative step.
            while x >= end - 1e-12 {
                if iterations >= max_iterations {
                    return Err(UiError::InvalidRange { start, end, step });
                }
                v.push(x);
                x += step; // step < 0 here
                iterations += 1;
            }
        }
        if v.is_empty() {
            return Err(UiError::InvalidRange { start, end, step });
        }
        Ok(v)
    }

    let periods = axis_usize(r.period)?;
    let scalars = axis_f64(r.scalar)?;
    if periods.is_empty() || scalars.is_empty() {
        return Err(UiError::InvalidRange {
            start: r.period.0 as f64,
            end: r.period.1 as f64,
            step: r.period.2 as f64,
        });
    }
    let combos_len = periods
        .len()
        .checked_mul(scalars.len())
        .ok_or(UiError::InvalidRange {
            start: periods.len() as f64,
            end: scalars.len() as f64,
            step: 0.0,
        })?;
    let mut out = Vec::with_capacity(combos_len);
    for &p in &periods {
        for &s in &scalars {
            out.push(UiParams {
                period: Some(p),
                scalar: Some(s),
            });
        }
    }
    Ok(out)
}

#[inline(always)]
pub fn ui_batch_slice(
    data: &[f64],
    sweep: &UiBatchRange,
    kern: Kernel,
) -> Result<UiBatchOutput, UiError> {
    ui_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn ui_batch_par_slice(
    data: &[f64],
    sweep: &UiBatchRange,
    kern: Kernel,
) -> Result<UiBatchOutput, UiError> {
    ui_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn ui_batch_inner(
    data: &[f64],
    sweep: &UiBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<UiBatchOutput, UiError> {
    // Check for empty input first
    if data.is_empty() {
        return Err(UiError::EmptyInputData);
    }

    let combos = expand_grid(sweep)?;
    if combos.is_empty() {
        return Err(UiError::InvalidRange { start: sweep.period.0 as f64, end: sweep.period.1 as f64, step: sweep.period.2 as f64 });
    }

    // Resolve Kernel::Auto to a concrete kernel
    let kern = match kern {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let first = data
        .iter()
        .position(|x| x.is_finite())
        .ok_or(UiError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    let span = max_p
        .checked_mul(2)
        .and_then(|v| v.checked_sub(2))
        .ok_or(UiError::InvalidRange {
            start: max_p as f64,
            end: data.len() as f64,
            step: 2.0,
        })?;
    let max_warmup = first
        .checked_add(span)
        .ok_or(UiError::InvalidRange {
            start: max_p as f64,
            end: data.len() as f64,
            step: 2.0,
        })?;
    if data.len() <= max_warmup {
        return Err(UiError::NotEnoughValidData {
            needed: max_warmup + 1,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let _total = rows
        .checked_mul(cols)
        .ok_or(UiError::InvalidRange {
            start: rows as f64,
            end: cols as f64,
            step: 0.0,
        })?;
    let mut buf_mu = make_uninit_matrix(rows, cols);
    let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
    let out_slice: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    // Use the grouped row-optimized writer that fills prefixes and rows in one pass.
    let combos = ui_batch_inner_into(data, sweep, kern, parallel, out_slice)?;

    // Convert back to Vec<f64> from ManuallyDrop
    let values_vec = unsafe {
        let ptr = buf_guard.as_mut_ptr() as *mut f64;
        let len = buf_guard.len();
        let cap = buf_guard.capacity();
        core::mem::forget(buf_guard);
        Vec::from_raw_parts(ptr, len, cap)
    };

    Ok(UiBatchOutput {
        values: values_vec,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn ui_row_scalar(data: &[f64], first: usize, period: usize, scalar: f64, out: &mut [f64]) {
    // `out` already has NaN prefix from init_matrix_prefixes
    ui_scalar(data, period, scalar, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
fn ui_row_avx2(data: &[f64], first: usize, period: usize, scalar: f64, out: &mut [f64]) {
    // TODO: Implement actual AVX2 optimizations
    // For now, use the optimized scalar batch version
    ui_row_scalar(data, first, period, scalar, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
fn ui_row_avx512(data: &[f64], first: usize, period: usize, scalar: f64, out: &mut [f64]) {
    // TODO: Implement actual AVX512 optimizations
    // For now, use the optimized scalar batch version
    ui_row_scalar(data, first, period, scalar, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
fn ui_row_avx512_short(data: &[f64], first: usize, period: usize, scalar: f64, out: &mut [f64]) {
    // TODO: Implement actual AVX512 optimizations for short periods
    // For now, use the optimized scalar batch version
    ui_row_scalar(data, first, period, scalar, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
fn ui_row_avx512_long(data: &[f64], first: usize, period: usize, scalar: f64, out: &mut [f64]) {
    // TODO: Implement actual AVX512 optimizations for long periods
    // For now, use the optimized scalar batch version
    ui_row_scalar(data, first, period, scalar, out)
}

// Python bindings

#[cfg(feature = "python")]
#[pyfunction(name = "ui")]
#[pyo3(signature = (data, period, scalar=100.0, kernel=None))]
pub fn ui_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period: usize,
    scalar: f64,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = UiParams {
        period: Some(period),
        scalar: Some(scalar),
    };
    let input = UiInput::from_slice(slice_in, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| ui_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction(name = "ui_batch")]
#[pyo3(signature = (data, period_range, scalar_range=(100.0, 100.0, 0.0), kernel=None))]
pub fn ui_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    scalar_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;

    let sweep = UiBatchRange {
        period: period_range,
        scalar: scalar_range,
    };

    let combos = expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = slice_in.len();

    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("ui_batch: rows * cols overflow"))?;

    let out_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    let kern = validate_kernel(kernel, true)?;

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
                _ => Kernel::Scalar, // Default to scalar for any other kernel
            };
            ui_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
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
    dict.set_item(
        "scalars",
        combos
            .iter()
            .map(|p| p.scalar.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "UiStream")]
pub struct UiStreamPy {
    inner: UiStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl UiStreamPy {
    #[new]
    pub fn new(period: usize, scalar: f64) -> PyResult<Self> {
        let params = UiParams {
            period: Some(period),
            scalar: Some(scalar),
        };
        let inner = UiStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(UiStreamPy { inner })
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.inner.update(value)
    }
}

// Helper function for batch operations with direct output writing
#[inline(always)]
fn ui_batch_inner_into(
    data: &[f64],
    sweep: &UiBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<UiParams>, UiError> {
    // Check for empty input first
    if data.is_empty() {
        return Err(UiError::EmptyInputData);
    }

    let combos = expand_grid(sweep)?;
    if combos.is_empty() {
        return Err(UiError::InvalidRange { start: sweep.period.0 as f64, end: sweep.period.1 as f64, step: sweep.period.2 as f64 });
    }

    let first = data
        .iter()
        .position(|x| x.is_finite())
        .ok_or(UiError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    let span = max_p
        .checked_mul(2)
        .and_then(|v| v.checked_sub(2))
        .ok_or(UiError::InvalidRange {
            start: max_p as f64,
            end: data.len() as f64,
            step: 2.0,
        })?;
    let max_warmup = first
        .checked_add(span)
        .ok_or(UiError::InvalidRange {
            start: max_p as f64,
            end: data.len() as f64,
            step: 2.0,
        })?;
    if data.len() <= max_warmup {
        return Err(UiError::NotEnoughValidData {
            needed: max_warmup + 1,
            valid: data.len() - first,
        });
    }

    let cols = data.len();
    for (row, combo) in combos.iter().enumerate() {
        let period = combo.period.unwrap();
        let span_row = period
            .checked_mul(2)
            .and_then(|v| v.checked_sub(2))
            .ok_or(UiError::InvalidRange {
                start: period as f64,
                end: data.len() as f64,
                step: 2.0,
            })?;
        let warmup = first
            .checked_add(span_row)
            .ok_or(UiError::InvalidRange {
                start: period as f64,
                end: data.len() as f64,
                step: 2.0,
            })?;
        let row_start = row * cols;
        for i in 0..warmup.min(cols) {
            out[row_start + i] = f64::NAN;
        }
    }

    // Row-optimized batch path: group rows by period. For rows sharing the same
    // period but different scalars, we compute a single base series with scalar=1.0
    // and scale results by |scalar| (exact equivalence since scalar is squared
    // inside the average and sqrt is applied at the end).
    use std::collections::BTreeMap;
    let mut by_period: BTreeMap<usize, Vec<(usize, f64)>> = BTreeMap::new();
    for (row, combo) in combos.iter().enumerate() {
        by_period
            .entry(combo.period.unwrap())
            .or_default()
            .push((row, combo.scalar.unwrap()));
    }

    let mut process_group = |(period, rows): (&usize, &Vec<(usize, f64)>)| {
        // Compute base UI with scalar=1.0 once for this period
        let mut base = vec![f64::NAN; cols];
        match kern {
            Kernel::Scalar => ui_row_scalar(data, first, *period, 1.0, &mut base),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => ui_row_avx2(data, first, *period, 1.0, &mut base),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => ui_row_avx512(data, first, *period, 1.0, &mut base),
            _ => ui_row_scalar(data, first, *period, 1.0, &mut base),
        }

        // Scale base by |scalar| for each row in this period group
        for &(row, scalar) in rows.iter() {
            let s = scalar.abs();
            let row_start = row * cols;
            let dst = &mut out[row_start..row_start + cols];
            for i in 0..cols {
                let v = base[i];
                dst[i] = if v.is_finite() { v * s } else { v };
            }
        }
    };

    if parallel {
        // Parallel row-specific optimization: compute one base series (scalar=1.0)
        // per period in parallel, then scale into rows in parallel.
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            use std::collections::HashMap;
            use std::sync::Arc;

            // 1) compute base UI per period (scalar=1.0) in parallel
            let period_keys: Vec<usize> = by_period.keys().copied().collect();
            let base_map: HashMap<usize, Arc<Vec<f64>>> = period_keys
                .par_iter()
                .map(|&p| {
                    let mut base = vec![f64::NAN; cols];
                    match kern {
                        Kernel::Scalar => ui_row_scalar(data, first, p, 1.0, &mut base),
                        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                        Kernel::Avx2 => ui_row_avx2(data, first, p, 1.0, &mut base),
                        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                        Kernel::Avx512 => ui_row_avx512(data, first, p, 1.0, &mut base),
                        _ => ui_row_scalar(data, first, p, 1.0, &mut base),
                    }
                    (p, Arc::new(base))
                })
                .collect();

            // 2) write each row from its period's base series scaled by |scalar|
            out.par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| {
                    let p = combos[row].period.unwrap();
                    let s = combos[row].scalar.unwrap().abs();
                    let base = base_map.get(&p).expect("base series present");
                    // scale base into slice
                    for i in 0..cols {
                        let v = base[i];
                        slice[i] = if v.is_finite() { v * s } else { v };
                    }
                });
        }

        #[cfg(target_arch = "wasm32")]
        {
            // WASM fallback: sequential grouped path
            for entry in by_period.iter() {
                process_group(entry);
            }
        }
    } else {
        for entry in by_period.iter() {
            process_group(entry);
        }
    }

    Ok(combos)
}

// WASM bindings

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// Core helper for writing directly to output slice - no allocations
#[cfg(feature = "wasm")]
pub fn ui_into_slice(dst: &mut [f64], input: &UiInput, kern: Kernel) -> Result<(), UiError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(UiError::EmptyInputData);
    }

    if dst.len() != len {
        return Err(UiError::OutputLengthMismatch {
            expected: len,
            got: dst.len(),
        });
    }

    let period = input.get_period();
    let scalar = input.get_scalar();
    if period == 0 || period > len {
        return Err(UiError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if !scalar.is_finite() {
        return Err(UiError::InvalidScalar { scalar });
    }

    let first = data
        .iter()
        .position(|x| x.is_finite())
        .ok_or(UiError::AllValuesNaN)?;
    if (len - first) < period {
        return Err(UiError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let chosen = match kern {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    // correct prefix
    let warmup = first + (period * 2 - 2);
    for v in &mut dst[..warmup.min(len)] {
        *v = f64::NAN;
    }

    match chosen {
        Kernel::Scalar | Kernel::ScalarBatch => ui_scalar(data, period, scalar, first, dst),
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 | Kernel::Avx2Batch => ui_avx2(data, period, scalar, first, dst),
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 | Kernel::Avx512Batch => ui_avx512(data, period, scalar, first, dst),
        _ => ui_scalar(data, period, scalar, first, dst),
    }
    Ok(())
}

/// Safe API - allocates and returns Vec<f64>
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ui_js(data: &[f64], period: usize, scalar: f64) -> Result<Vec<f64>, JsValue> {
    if !scalar.is_finite() {
        return Err(JsValue::from_str(&format!("Invalid scalar: {}", scalar)));
    }
    let params = UiParams {
        period: Some(period),
        scalar: Some(scalar),
    };
    let input = UiInput::from_slice(data, params);

    let mut output = vec![0.0; data.len()];

    ui_into_slice(&mut output, &input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

/// Fast API with aliasing detection - zero allocations unless aliased
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ui_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
    scalar: f64,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }
    if !scalar.is_finite() {
        return Err(JsValue::from_str(&format!("Invalid scalar: {}", scalar)));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let params = UiParams {
            period: Some(period),
            scalar: Some(scalar),
        };
        let input = UiInput::from_slice(data, params);

        if in_ptr == out_ptr.cast_const() {
            // CRITICAL: Aliasing check
            let mut temp = vec![0.0; len];
            ui_into_slice(&mut temp, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            ui_into_slice(out, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

/// Memory allocation for WASM
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ui_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

/// Memory deallocation for WASM
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ui_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

/// Batch configuration for WASM
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct UiBatchConfig {
    pub period_range: (usize, usize, usize),
    pub scalar_range: (f64, f64, f64),
}

/// Batch output for WASM
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct UiBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<UiParams>,
    pub rows: usize,
    pub cols: usize,
}

/// Batch processing API
#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = ui_batch)]
pub fn ui_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: UiBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = UiBatchRange {
        period: config.period_range,
        scalar: config.scalar_range,
    };

    let output = ui_batch_inner(data, &sweep, detect_best_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = UiBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };

    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

// ---------------- CUDA Python bindings -----------------

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "ui_cuda_batch_dev")]
#[pyo3(signature = (data_f32, period_range, scalar_range=(100.0, 100.0, 0.0), device_id=0))]
pub fn ui_cuda_batch_dev_py(
    py: Python<'_>,
    data_f32: numpy::PyReadonlyArray1<'_, f32>,
    period_range: (usize, usize, usize),
    scalar_range: (f64, f64, f64),
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let slice_in = data_f32.as_slice()?;
    let sweep = UiBatchRange {
        period: period_range,
        scalar: scalar_range,
    };
    let inner = py.allow_threads(|| {
        let cuda = CudaUi::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.ui_batch_dev(slice_in, &sweep)
            .map(|(arr, _)| arr)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    make_device_array_py(device_id, inner)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "ui_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, period, scalar=100.0, device_id=0))]
pub fn ui_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    data_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    period: usize,
    scalar: f64,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    use numpy::PyUntypedArrayMethods;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let flat_in: &[f32] = data_tm_f32.as_slice()?;
    let rows = data_tm_f32.shape()[0];
    let cols = data_tm_f32.shape()[1];
    let params = UiParams {
        period: Some(period),
        scalar: Some(scalar),
    };
    let inner = py.allow_threads(|| {
        let cuda = CudaUi::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.ui_many_series_one_param_time_major_dev(flat_in, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    make_device_array_py(device_id, inner)
}

/// Fast batch API with raw pointers
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ui_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
    scalar_start: f64,
    scalar_end: f64,
    scalar_step: f64,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to ui_batch_into"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        let sweep = UiBatchRange {
            period: (period_start, period_end, period_step),
            scalar: (scalar_start, scalar_end, scalar_step),
        };

        let combos = expand_grid(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let cols = len;

        let total = rows
            .checked_mul(cols)
            .ok_or_else(|| JsValue::from_str("ui_batch_into: rows * cols overflow"))?;

        let out = std::slice::from_raw_parts_mut(out_ptr, total);

        ui_batch_inner_into(data, &sweep, detect_best_kernel(), false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_ui_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = UiParams {
            period: None,
            scalar: None,
        };
        let input = UiInput::from_candles(&candles, "close", default_params);
        let output = ui_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_ui_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = UiParams {
            period: Some(14),
            scalar: Some(100.0),
        };
        let input = UiInput::from_candles(&candles, "close", params);
        let ui_result = ui_with_kernel(&input, kernel)?;
        let expected_last_five_ui = [
            3.514342861283708,
            3.304986039846459,
            3.2011859814326304,
            3.1308860017483373,
            2.909612553474519,
        ];
        assert!(ui_result.values.len() >= 5);
        let start_index = ui_result.values.len() - 5;
        let result_last_five_ui = &ui_result.values[start_index..];
        for (i, &value) in result_last_five_ui.iter().enumerate() {
            let expected_value = expected_last_five_ui[i];
            assert!(
                (value - expected_value).abs() < 1e-6,
                "[{}] UI mismatch at index {}: expected {}, got {}",
                test_name,
                i,
                expected_value,
                value
            );
        }
        let period = 14;
        for i in 0..(period - 1) {
            assert!(ui_result.values[i].is_nan());
        }
        Ok(())
    }

    fn check_ui_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = UiInput::with_default_candles(&candles);
        match input.data {
            UiData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected UiData::Candles"),
        }
        let output = ui_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_ui_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = UiParams {
            period: Some(0),
            scalar: None,
        };
        let input = UiInput::from_slice(&input_data, params);
        let res = ui_with_kernel(&input, kernel);
        assert!(res.is_err());
        Ok(())
    }

    fn check_ui_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = UiParams {
            period: Some(10),
            scalar: None,
        };
        let input = UiInput::from_slice(&data_small, params);
        let res = ui_with_kernel(&input, kernel);
        assert!(res.is_err());
        Ok(())
    }

    fn check_ui_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = UiParams {
            period: Some(14),
            scalar: Some(100.0),
        };
        let input = UiInput::from_slice(&single_point, params);
        let res = ui_with_kernel(&input, kernel);
        assert!(res.is_err());
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_ui_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Define comprehensive parameter combinations
        let test_params = vec![
            UiParams::default(), // period: 14, scalar: 100.0
            UiParams {
                period: Some(2), // minimum viable
                scalar: Some(100.0),
            },
            UiParams {
                period: Some(5), // small
                scalar: Some(50.0),
            },
            UiParams {
                period: Some(10), // medium
                scalar: Some(100.0),
            },
            UiParams {
                period: Some(20), // large
                scalar: Some(200.0),
            },
            UiParams {
                period: Some(50), // very large
                scalar: Some(100.0),
            },
            UiParams {
                period: Some(100), // extreme
                scalar: Some(100.0),
            },
            UiParams {
                period: Some(14), // default period with different scalars
                scalar: Some(1.0),
            },
            UiParams {
                period: Some(14),
                scalar: Some(500.0),
            },
            UiParams {
                period: Some(14),
                scalar: Some(1000.0),
            },
            UiParams {
                period: Some(7), // edge case combinations
                scalar: Some(75.0),
            },
            UiParams {
                period: Some(30),
                scalar: Some(150.0),
            },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = UiInput::from_candles(&candles, "close", params.clone());
            let output = ui_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: period={}, scalar={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(14),
                        params.scalar.unwrap_or(100.0),
                        param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: period={}, scalar={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(14),
                        params.scalar.unwrap_or(100.0),
                        param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: period={}, scalar={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(14),
                        params.scalar.unwrap_or(100.0),
                        param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_ui_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(()) // No-op in release builds
    }

    macro_rules! generate_all_ui_tests {
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

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_ui_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (2usize..=20, 1.0f64..200.0f64).prop_flat_map(|(period, scalar)| {
            let min_data_needed = period * 2 - 2 + 20; // warmup + some data for testing
            (
                prop::collection::vec(
                    (0.001f64..1e6f64)
                        .prop_filter("positive finite", |x| x.is_finite() && *x > 0.0),
                    min_data_needed..400,
                ),
                Just(period),
                Just(scalar),
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, period, scalar)| {
                let params = UiParams {
                    period: Some(period),
                    scalar: Some(scalar),
                };
                let input = UiInput::from_slice(&data, params);

                let UiOutput { values: out } = ui_with_kernel(&input, kernel).unwrap();
                let UiOutput { values: ref_out } = ui_with_kernel(&input, Kernel::Scalar).unwrap();

                // Property 1: Warmup period should be NaN
                let warmup_period = period * 2 - 2;
                for i in 0..warmup_period.min(data.len()) {
                    prop_assert!(
                        out[i].is_nan(),
                        "[{}] Expected NaN during warmup at index {}, got {}",
                        test_name,
                        i,
                        out[i]
                    );
                }

                // Property 2: Non-negativity - UI must always be >= 0
                for (i, &value) in out.iter().enumerate() {
                    if !value.is_nan() {
                        prop_assert!(
                            value >= 0.0,
                            "[{}] UI must be non-negative at index {}: got {}",
                            test_name,
                            i,
                            value
                        );
                    }
                }

                // Property 3: Zero when prices monotonically increase
                let is_monotonic_increase = data.windows(2).all(|w| w[1] >= w[0]);
                if is_monotonic_increase && data.len() > warmup_period {
                    for i in warmup_period..data.len() {
                        prop_assert!(
                            out[i].abs() < 1e-9,
                            "[{}] UI should be ~0 for monotonic increase at index {}: got {}",
                            test_name,
                            i,
                            out[i]
                        );
                    }
                }

                // Property 4: Period=1 edge case - UI should always be 0
                if period == 1 {
                    // With period=1, warmup is 0, so all values should be valid
                    for (i, &value) in out.iter().enumerate() {
                        prop_assert!(
                            value.abs() < 1e-9,
                            "[{}] UI with period=1 should be 0 at index {}: got {}",
                            test_name,
                            i,
                            value
                        );
                    }
                }

                // Property 5: Flat data should give UI = 0
                let is_flat = data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-12);
                if is_flat && data.len() > warmup_period {
                    for i in warmup_period..data.len() {
                        prop_assert!(
                            out[i].abs() < 1e-9,
                            "[{}] UI should be 0 for flat data at index {}: got {}",
                            test_name,
                            i,
                            out[i]
                        );
                    }
                }

                // Property 6: Bounded by theoretical maximum
                // UI = sqrt(avg(squared_percentage_drawdowns)) * scalar
                // Maximum theoretical: all prices drop to near-zero = 100% drawdown
                // UI_max = sqrt(1.0) * scalar = scalar
                for i in warmup_period..data.len() {
                    if !out[i].is_nan() {
                        // UI theoretical max is scalar * 1.0, allow 10% margin for numerical precision
                        prop_assert!(
                            out[i] <= scalar * 1.1,
                            "[{}] UI exceeds theoretical maximum at index {}: UI={}, max={}",
                            test_name,
                            i,
                            out[i],
                            scalar * 1.1
                        );

                        // Also check that UI is finite
                        prop_assert!(
                            out[i].is_finite(),
                            "[{}] UI is not finite at index {}: {}",
                            test_name,
                            i,
                            out[i]
                        );
                    }
                }

                // Property 7: Kernel consistency - all kernels should produce identical results
                for i in 0..data.len() {
                    let y = out[i];
                    let r = ref_out[i];

                    if !y.is_finite() || !r.is_finite() {
                        prop_assert!(
                            y.to_bits() == r.to_bits(),
                            "[{}] finite/NaN mismatch at index {}: {} vs {}",
                            test_name,
                            i,
                            y,
                            r
                        );
                        continue;
                    }

                    let ulp_diff: u64 = y.to_bits().abs_diff(r.to_bits());
                    prop_assert!(
                        (y - r).abs() <= 1e-9 || ulp_diff <= 4,
                        "[{}] kernel mismatch at index {}: {} vs {} (ULP={})",
                        test_name,
                        i,
                        y,
                        r,
                        ulp_diff
                    );
                }

                // Property 8: Determinism - running twice should give identical results
                let UiOutput { values: out2 } = ui_with_kernel(&input, kernel).unwrap();
                for i in 0..data.len() {
                    if out[i].is_finite() && out2[i].is_finite() {
                        prop_assert!(
                            (out[i] - out2[i]).abs() < 1e-12,
                            "[{}] Non-deterministic result at index {}: {} vs {}",
                            test_name,
                            i,
                            out[i],
                            out2[i]
                        );
                    } else {
                        prop_assert!(
                            out[i].to_bits() == out2[i].to_bits(),
                            "[{}] Non-deterministic NaN at index {}",
                            test_name,
                            i
                        );
                    }
                }

                // Property 9: Scalar proportionality
                // Test that doubling scalar doubles the output
                if scalar > 1.0 && scalar < 100.0 {
                    let params2 = UiParams {
                        period: Some(period),
                        scalar: Some(scalar * 2.0),
                    };
                    let input2 = UiInput::from_slice(&data, params2);
                    let UiOutput { values: out_scaled } = ui_with_kernel(&input2, kernel).unwrap();

                    for i in warmup_period..data.len() {
                        if out[i].is_finite() && out_scaled[i].is_finite() && out[i] > 1e-9 {
                            let ratio = out_scaled[i] / out[i];
                            prop_assert!(
                                (ratio - 2.0).abs() < 1e-6,
                                "[{}] Scalar not proportional at index {}: ratio={} (expected 2.0)",
                                test_name,
                                i,
                                ratio
                            );
                        }
                    }
                }

                // Property 10: When data is well-behaved, outputs should stabilize
                // For sufficiently large, stable data, we should get valid UI values
                let has_large_stable_region =
                    data.len() > period * 4 && data.iter().all(|&x| x > 0.1 && x < 1e5);
                if has_large_stable_region {
                    // Count valid outputs after warmup
                    let valid_count = out[warmup_period..]
                        .iter()
                        .filter(|&&x| !x.is_nan())
                        .count();
                    let expected_valid = data.len() - warmup_period;

                    // We should have mostly valid outputs (allow some edge cases)
                    prop_assert!(
                        valid_count as f64 >= expected_valid as f64 * 0.8,
                        "[{}] Too few valid outputs: {} out of {} expected",
                        test_name,
                        valid_count,
                        expected_valid
                    );
                }

                // Property 11: Volatility relationship - UI should increase with volatility
                // Create synthetic high and low volatility periods if we have enough data
                if data.len() > period * 4 {
                    // Find a stable period (low volatility)
                    let mut min_volatility_ui = f64::INFINITY;
                    let mut max_volatility_ui = 0.0;

                    for i in warmup_period..data.len() {
                        if !out[i].is_nan() {
                            // Look at the price range in the window
                            let window_start = i.saturating_sub(period - 1);
                            let window = &data[window_start..=i];
                            let max_price =
                                window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                            let min_price = window.iter().cloned().fold(f64::INFINITY, f64::min);
                            let price_range = (max_price - min_price) / max_price;

                            // Track UI values for different volatility levels
                            if price_range < 0.01 && out[i] < min_volatility_ui {
                                min_volatility_ui = out[i];
                            }
                            if price_range > 0.1 && out[i] > max_volatility_ui {
                                max_volatility_ui = out[i];
                            }
                        }
                    }

                    // If we found both low and high volatility periods, high should have higher UI
                    if min_volatility_ui != f64::INFINITY && max_volatility_ui > 0.0 {
                        prop_assert!(
							max_volatility_ui >= min_volatility_ui,
							"[{}] UI should be higher for volatile periods: low_vol_UI={}, high_vol_UI={}",
							test_name, min_volatility_ui, max_volatility_ui
						);
                    }
                }

                // Property 12: Direct formula verification for simple cases
                // For a window with sufficient drawdown, verify the calculation
                if period <= 5 && data.len() > warmup_period + period {
                    // Find a window where we can manually calculate
                    for i in (warmup_period + period)..data.len().min(warmup_period + period * 2) {
                        if !out[i].is_nan() && out[i] > scalar * 0.01 {
                            // Only verify when UI is meaningful
                            // Calculate the window of interest for the last 'period' UI calculations
                            // UI at position i uses a sliding window approach
                            let mut sum_squared_dd = 0.0;
                            let mut valid_count = 0;

                            // For UI at position i, we need to look at the last 'period' drawdowns
                            for j in 0..period {
                                let pos = i - j;
                                if pos >= period - 1 {
                                    // Find the rolling max for this position
                                    let max_start = pos + 1 - period;
                                    let max_end = pos + 1;
                                    let rolling_max = data[max_start..max_end]
                                        .iter()
                                        .cloned()
                                        .fold(f64::NEG_INFINITY, f64::max);

                                    if rolling_max > 0.0 && !data[pos].is_nan() {
                                        let dd = scalar * (data[pos] - rolling_max) / rolling_max;
                                        sum_squared_dd += dd * dd;
                                        valid_count += 1;
                                    }
                                }
                            }

                            if valid_count == period {
                                let manual_ui = (sum_squared_dd / period as f64).sqrt();
                                // Allow 5% tolerance or small absolute difference for floating point
                                let tolerance = manual_ui * 0.05 + 1e-6;
                                prop_assert!(
									(out[i] - manual_ui).abs() <= tolerance,
									"[{}] Direct formula verification failed at index {}: calculated={}, expected={}, diff={}",
									test_name, i, out[i], manual_ui, (out[i] - manual_ui).abs()
								);
                                break; // Only need to verify once
                            }
                        }
                    }
                }

                // Property 13: Near-zero volatility test
                // When price movements are minimal, UI should approach zero
                let has_low_volatility =
                    data.windows(2).all(|w| (w[1] - w[0]).abs() / w[0] < 0.0001); // Less than 0.01% change
                if has_low_volatility && data.len() > warmup_period {
                    for i in warmup_period..data.len() {
                        if !out[i].is_nan() {
                            prop_assert!(
                                out[i] < scalar * 0.01, // UI should be less than 1% of scalar
                                "[{}] UI too high for near-zero volatility at index {}: UI={}",
                                test_name,
                                i,
                                out[i]
                            );
                        }
                    }
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    generate_all_ui_tests!(
        check_ui_partial_params,
        check_ui_accuracy,
        check_ui_default_candles,
        check_ui_zero_period,
        check_ui_period_exceeds_length,
        check_ui_very_small_dataset,
        check_ui_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_ui_tests!(check_ui_property);

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = UiBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = UiParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            3.514342861283708,
            3.304986039846459,
            3.2011859814326304,
            3.1308860017483373,
            2.909612553474519,
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

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test various parameter sweep configurations
        let test_configs = vec![
            // (period_start, period_end, period_step, scalar_start, scalar_end, scalar_step)
            (2, 10, 2, 100.0, 100.0, 0.0), // Small periods, static scalar
            (5, 25, 5, 50.0, 150.0, 50.0), // Medium periods with scalar sweep
            (30, 60, 15, 100.0, 100.0, 0.0), // Large periods
            (2, 5, 1, 1.0, 100.0, 33.0),   // Dense small range with scalar sweep
            (10, 20, 2, 200.0, 200.0, 0.0), // Medium range, high scalar
            (14, 14, 0, 1.0, 1000.0, 199.0), // Static period with scalar sweep
            (3, 12, 3, 75.0, 125.0, 25.0), // Small to medium with scalar variations
            (50, 100, 25, 100.0, 500.0, 200.0), // Very large periods with scalar sweep
            (7, 21, 7, 50.0, 50.0, 0.0),   // Specific periods, static scalar
        ];

        for (cfg_idx, &(p_start, p_end, p_step, s_start, s_end, s_step)) in
            test_configs.iter().enumerate()
        {
            let output = UiBatchBuilder::new()
                .kernel(kernel)
                .period_range(p_start, p_end, p_step)
                .scalar_range(s_start, s_end, s_step)
                .apply_candles(&c, "close")?;

            for (idx, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;
                let combo = &output.combos[row];

                // Check all three poison patterns with detailed context
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}, scalar={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(14),
                        combo.scalar.unwrap_or(100.0)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}, scalar={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(14),
                        combo.scalar.unwrap_or(100.0)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}, scalar={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(14),
                        combo.scalar.unwrap_or(100.0)
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(
        _test: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(()) // No-op in release builds
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
    gen_batch_tests!(check_batch_no_poison);

    #[test]
    fn test_ui_into_matches_api() -> Result<(), Box<dyn std::error::Error>> {
        // Use the existing CSV candles to match other UI tests
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = UiInput::with_default_candles(&candles);

        // Baseline via existing Vec-returning API
        let baseline = ui(&input)?.values;

        // Preallocate output buffer and compute via new into API
        let mut out = vec![0.0; baseline.len()];
        #[cfg(not(feature = "wasm"))]
        {
            ui_into(&input, &mut out)?;
        }
        #[cfg(feature = "wasm")]
        {
            // In wasm builds, use the slice helper to emulate native into
            ui_into_slice(&mut out, &input, Kernel::Auto)?;
        }

        assert_eq!(baseline.len(), out.len());

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b)
        }
        for i in 0..baseline.len() {
            assert!(
                eq_or_both_nan(baseline[i], out[i]),
                "mismatch at index {i}: baseline={:?}, into={:?}",
                baseline[i],
                out[i]
            );
        }

        Ok(())
    }
}
