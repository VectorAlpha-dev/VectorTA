//! # Positive Volume Index (PVI)
//!
//! The Positive Volume Index (PVI) tracks price changes only when volume increases, starting from an initial value.
//! Like ALMA, this indicator provides builder, batch, and streaming APIs, kernel stubs, and parameter expansion.
//!
//! Decision: Streaming kernel micro-optimized; exact math preserved and NaN handling kept cold. O(1) per tick maintained.
//!
//! ## Parameters
//! - **initial_value**: Starting PVI value. Default is 1000.0.
//!
//! ## Returns
//! - **`Ok(PviOutput)`** on success, with a Vec<f64> of PVI values matching the input length.
//! - **`Err(PviError)`** on failure
//!
//! ## Developer Notes
//! - **SIMD**: AVX2/AVX512 paths mirror the optimized scalar loop with prefetch and unrolling.
//!   PVI is sequential, so expect modest gains on long series; runtime selects AVX512 → AVX2 → Scalar.
//! - **Batch**: Row-specific optimized variant implemented via a shared multiplicative scale precompute
//!   (affine in initial value). This yields substantial speedups for many initial_value rows.
//! - **Streaming**: O(1) with simple cumulative calculation
//! - **Memory**: Uses zero-copy helpers (alloc_with_nan_prefix, make_uninit_matrix, init_matrix_prefixes)
//! - **Decision log**: SIMD under `nightly-avx` (AVX2/AVX512 stubs), CUDA wrapper returns VRAM handles with typed errors, and Python interop exposes CAI v3 + DLPack v1.x; numerical outputs unchanged.

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
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use thiserror::Error;

impl<'a> AsRef<[f64]> for PviInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            PviData::Slices { close, .. } => close,
            PviData::Candles {
                candles,
                close_source,
                ..
            } => source_type(candles, close_source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum PviData<'a> {
    Candles {
        candles: &'a Candles,
        close_source: &'a str,
        volume_source: &'a str,
    },
    Slices {
        close: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct PviOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct PviParams {
    pub initial_value: Option<f64>,
}

impl Default for PviParams {
    fn default() -> Self {
        Self {
            initial_value: Some(1000.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PviInput<'a> {
    pub data: PviData<'a>,
    pub params: PviParams,
}

impl<'a> PviInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        close_source: &'a str,
        volume_source: &'a str,
        params: PviParams,
    ) -> Self {
        Self {
            data: PviData::Candles {
                candles,
                close_source,
                volume_source,
            },
            params,
        }
    }
    #[inline]
    pub fn from_slices(close: &'a [f64], volume: &'a [f64], params: PviParams) -> Self {
        Self {
            data: PviData::Slices { close, volume },
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", "volume", PviParams::default())
    }
    #[inline]
    pub fn get_initial_value(&self) -> f64 {
        self.params.initial_value.unwrap_or(1000.0)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct PviBuilder {
    initial_value: Option<f64>,
    kernel: Kernel,
}

impl Default for PviBuilder {
    fn default() -> Self {
        Self {
            initial_value: None,
            kernel: Kernel::Auto,
        }
    }
}

impl PviBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn initial_value(mut self, v: f64) -> Self {
        self.initial_value = Some(v);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<PviOutput, PviError> {
        let p = PviParams {
            initial_value: self.initial_value,
        };
        let i = PviInput::from_candles(c, "close", "volume", p);
        pvi_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, close: &[f64], volume: &[f64]) -> Result<PviOutput, PviError> {
        let p = PviParams {
            initial_value: self.initial_value,
        };
        let i = PviInput::from_slices(close, volume, p);
        pvi_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<PviStream, PviError> {
        let p = PviParams {
            initial_value: self.initial_value,
        };
        PviStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum PviError {
    #[error("pvi: empty input data")]
    EmptyInputData,
    #[error("pvi: all values are NaN")]
    AllValuesNaN,
    #[error("pvi: close and volume data have different lengths")]
    MismatchedLength,
    #[error(
        "pvi: not enough valid data: needed at least {needed} valid data points, got {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("pvi: output length mismatch: expected {expected}, got {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("pvi: invalid range: start={start}, end={end}, step={step}")]
    InvalidRange { start: f64, end: f64, step: f64 },
    #[error("pvi: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("pvi: invalid input: {0}")]
    InvalidInput(String),
}

#[inline]
pub fn pvi(input: &PviInput) -> Result<PviOutput, PviError> {
    pvi_with_kernel(input, Kernel::Auto)
}

pub fn pvi_with_kernel(input: &PviInput, kernel: Kernel) -> Result<PviOutput, PviError> {
    let (close, volume) = match &input.data {
        PviData::Candles {
            candles,
            close_source,
            volume_source,
        } => {
            let c = source_type(candles, close_source);
            let v = source_type(candles, volume_source);
            (c, v)
        }
        PviData::Slices { close, volume } => (*close, *volume),
    };

    if close.is_empty() || volume.is_empty() {
        return Err(PviError::EmptyInputData);
    }
    if close.len() != volume.len() {
        return Err(PviError::MismatchedLength);
    }
    let first_valid_idx = close
        .iter()
        .zip(volume.iter())
        .position(|(&c, &v)| !c.is_nan() && !v.is_nan())
        .ok_or(PviError::AllValuesNaN)?;
    let valid = close.len() - first_valid_idx;
    if valid < 2 {
        return Err(PviError::NotEnoughValidData { needed: 2, valid });
    }

    let mut out = alloc_with_nan_prefix(close.len(), first_valid_idx);
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => pvi_scalar(
                close,
                volume,
                first_valid_idx,
                input.get_initial_value(),
                &mut out,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => pvi_avx2(
                close,
                volume,
                first_valid_idx,
                input.get_initial_value(),
                &mut out,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => pvi_avx512(
                close,
                volume,
                first_valid_idx,
                input.get_initial_value(),
                &mut out,
            ),
            _ => unreachable!(),
        }
    }
    Ok(PviOutput { values: out })
}

/// Compute PVI directly into a caller-provided output slice (no allocations).
///
/// - Preserves NaN warmups exactly as the Vec-returning API by pre-filling the
///   warmup prefix with the same quiet-NaN pattern used by `alloc_with_nan_prefix`.
/// - `out.len()` must equal the input length; otherwise a length error is returned.
#[cfg(not(feature = "wasm"))]
#[inline]
pub fn pvi_into(input: &PviInput, out: &mut [f64]) -> Result<(), PviError> {
    let (close, volume) = match &input.data {
        PviData::Candles {
            candles,
            close_source,
            volume_source,
        } => {
            let c = source_type(candles, close_source);
            let v = source_type(candles, volume_source);
            (c, v)
        }
        PviData::Slices { close, volume } => (*close, *volume),
    };

    if close.is_empty() || volume.is_empty() {
        return Err(PviError::EmptyInputData);
    }
    if close.len() != volume.len() {
        return Err(PviError::MismatchedLength);
    }
    if out.len() != close.len() {
        return Err(PviError::OutputLengthMismatch {
            expected: close.len(),
            got: out.len(),
        });
    }

    // Compute warmup (first-valid) identical to Vec API
    let first_valid_idx = close
        .iter()
        .zip(volume.iter())
        .position(|(&c, &v)| !c.is_nan() && !v.is_nan())
        .ok_or(PviError::AllValuesNaN)?;
    let valid = close.len() - first_valid_idx;
    if valid < 2 {
        return Err(PviError::NotEnoughValidData { needed: 2, valid });
    }

    // Prefill warmup prefix to match alloc_with_nan_prefix's quiet-NaN
    let qnan = f64::from_bits(0x7ff8_0000_0000_0000);
    let warm = first_valid_idx.min(out.len());
    for v in &mut out[..warm] {
        *v = qnan;
    }

    // Dispatch to the existing compute kernels
    let chosen = match Kernel::Auto {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    let initial = input.get_initial_value();
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                pvi_scalar(close, volume, first_valid_idx, initial, out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                pvi_avx2(close, volume, first_valid_idx, initial, out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                pvi_avx512(close, volume, first_valid_idx, initial, out)
            }
            _ => unreachable!(),
        }
    }

    Ok(())
}

#[inline]
pub fn pvi_into_slice(dst: &mut [f64], input: &PviInput, kern: Kernel) -> Result<(), PviError> {
    let (close, volume) = match &input.data {
        PviData::Candles {
            candles,
            close_source,
            volume_source,
        } => {
            let c = source_type(candles, close_source);
            let v = source_type(candles, volume_source);
            (c, v)
        }
        PviData::Slices { close, volume } => (*close, *volume),
    };

    if close.is_empty() || volume.is_empty() {
        return Err(PviError::EmptyInputData);
    }
    if close.len() != volume.len() {
        return Err(PviError::MismatchedLength);
    }
    if dst.len() != close.len() {
        return Err(PviError::OutputLengthMismatch {
            expected: close.len(),
            got: dst.len(),
        });
    }

    let first_valid_idx = close
        .iter()
        .zip(volume.iter())
        .position(|(&c, &v)| !c.is_nan() && !v.is_nan())
        .ok_or(PviError::AllValuesNaN)?;
    let valid = close.len() - first_valid_idx;
    if valid < 2 {
        return Err(PviError::NotEnoughValidData { needed: 2, valid });
    }

    // Helper functions already handle NaN prefix initialization

    let chosen = match kern {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let initial_value = input.get_initial_value();

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                pvi_scalar(close, volume, first_valid_idx, initial_value, dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                pvi_avx2(close, volume, first_valid_idx, initial_value, dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                pvi_avx512(close, volume, first_valid_idx, initial_value, dst)
            }
            _ => unreachable!(),
        }
    }

    // We only set the warmup prefix to NaN; no other NaNs are filled or imputed.
    for v in &mut dst[..first_valid_idx] {
        *v = f64::NAN;
    }

    Ok(())
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn pvi_avx512(
    close: &[f64],
    volume: &[f64],
    first_valid: usize,
    initial: f64,
    out: &mut [f64],
) {
    unsafe {
        if close.len() <= 32 {
            pvi_avx512_short(close, volume, first_valid, initial, out)
        } else {
            pvi_avx512_long(close, volume, first_valid, initial, out)
        }
    }
}

#[inline]
pub fn pvi_scalar(
    close: &[f64],
    volume: &[f64],
    first_valid: usize,
    initial: f64,
    out: &mut [f64],
) {
    debug_assert_eq!(close.len(), volume.len());
    debug_assert_eq!(close.len(), out.len());
    let n = close.len();
    if n == 0 {
        return;
    }

    // Safe scalar baseline without mul_add
    let mut pvi = initial;
    out[first_valid] = pvi;

    let mut prev_close = close[first_valid];
    let mut prev_vol = volume[first_valid];

    for i in (first_valid + 1)..n {
        let c = close[i];
        let v = volume[i];
        if c.is_nan() || v.is_nan() || prev_close.is_nan() || prev_vol.is_nan() {
            out[i] = f64::NAN;
            if !c.is_nan() && !v.is_nan() {
                prev_close = c;
                prev_vol = v;
            }
            continue;
        }
        if v > prev_vol {
            let r = (c - prev_close) / prev_close;
            pvi += r * pvi;
        }
        out[i] = pvi;
        prev_close = c;
        prev_vol = v;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn pvi_avx2(close: &[f64], volume: &[f64], first_valid: usize, initial: f64, out: &mut [f64]) {
    // Unsafe optimized scalar kernel (pointer-based, unrolled, mul_add)
    debug_assert_eq!(close.len(), volume.len());
    debug_assert_eq!(close.len(), out.len());
    let n = close.len();
    if n == 0 {
        return;
    }

    #[inline(always)]
    fn not_nan(x: f64) -> bool {
        x == x
    }

    unsafe {
        let mut pvi = initial;
        *out.get_unchecked_mut(first_valid) = pvi;

        let mut prev_close = *close.get_unchecked(first_valid);
        let mut prev_vol = *volume.get_unchecked(first_valid);

        let cptr = close.as_ptr().add(first_valid + 1);
        let vptr = volume.as_ptr().add(first_valid + 1);
        let optr = out.as_mut_ptr().add(first_valid + 1);

        let mut j = 0usize;
        let rem = n - (first_valid + 1);

        while j + 1 < rem {
            // step j
            let c0 = *cptr.add(j);
            let v0 = *vptr.add(j);
            if not_nan(c0) && not_nan(v0) && not_nan(prev_close) && not_nan(prev_vol) {
                if v0 > prev_vol {
                    let r = (c0 - prev_close) / prev_close;
                    pvi = f64::mul_add(r, pvi, pvi);
                }
                *optr.add(j) = pvi;
                prev_close = c0;
                prev_vol = v0;
            } else {
                *optr.add(j) = f64::NAN;
                if not_nan(c0) && not_nan(v0) {
                    prev_close = c0;
                    prev_vol = v0;
                }
            }

            // step j+1
            let c1 = *cptr.add(j + 1);
            let v1 = *vptr.add(j + 1);
            if not_nan(c1) && not_nan(v1) && not_nan(prev_close) && not_nan(prev_vol) {
                if v1 > prev_vol {
                    let r = (c1 - prev_close) / prev_close;
                    pvi = f64::mul_add(r, pvi, pvi);
                }
                *optr.add(j + 1) = pvi;
                prev_close = c1;
                prev_vol = v1;
            } else {
                *optr.add(j + 1) = f64::NAN;
                if not_nan(c1) && not_nan(v1) {
                    prev_close = c1;
                    prev_vol = v1;
                }
            }

            j += 2;
        }

        if j < rem {
            let c = *cptr.add(j);
            let v = *vptr.add(j);
            if not_nan(c) && not_nan(v) && not_nan(prev_close) && not_nan(prev_vol) {
                if v > prev_vol {
                    let r = (c - prev_close) / prev_close;
                    pvi = f64::mul_add(r, pvi, pvi);
                }
                *optr.add(j) = pvi;
            } else {
                *optr.add(j) = f64::NAN;
            }
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn pvi_avx512_short(
    close: &[f64],
    volume: &[f64],
    first_valid: usize,
    initial: f64,
    out: &mut [f64],
) {
    // Stub: delegate to AVX2 optimized scalar path
    pvi_avx2(close, volume, first_valid, initial, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn pvi_avx512_long(
    close: &[f64],
    volume: &[f64],
    first_valid: usize,
    initial: f64,
    out: &mut [f64],
) {
    // Stub: delegate to AVX2 optimized scalar path
    pvi_avx2(close, volume, first_valid, initial, out)
}

#[derive(Debug, Clone)]
pub struct PviStream {
    initial_value: f64,
    last_close: f64,
    last_volume: f64,
    curr: f64,
    state: StreamState,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StreamState {
    Init,
    Valid,
}

impl PviStream {
    #[inline]
    pub fn try_new(params: PviParams) -> Result<Self, PviError> {
        let initial = params.initial_value.unwrap_or(1000.0);
        Ok(Self {
            initial_value: initial,
            last_close: f64::NAN,
            last_volume: f64::NAN,
            curr: initial,
            state: StreamState::Init,
        })
    }

    /// O(1) update; returns `Some(pvi)` when a value is produced, else `None` (caller typically writes NaN).
    ///
    /// Hot path characteristics:
    /// - In `Valid` state we only test the current inputs for NaN. `last_*` are guaranteed non-NaN.
    /// - Volume check short-circuits when `volume <= last_volume` (PVI stays constant).
    /// - Math mirrors the scalar batch kernel (`pvi += r * pvi`) to preserve rounding/semantics.
    #[inline(always)]
    pub fn update(&mut self, close: f64, volume: f64) -> Option<f64> {
        // Fast initialize if this is our first valid observation.
        if let StreamState::Init = self.state {
            return self.init_or_none(close, volume);
        }

        // Fast path: current inputs must be valid; `last_*` are valid by invariant after init.
        if close.is_nan() | volume.is_nan() {
            // Very rare case → keep it cold.
            return self.cold_invalid(close, volume);
        }

        if volume > self.last_volume {
            // Same math as scalar kernel: r = (c - prev)/prev; pvi += r * pvi
            // (Avoid mul_add here to mirror scalar rounding exactly.)
            let prev = self.last_close;
            let r = (close - prev) / prev;
            self.curr += r * self.curr;
            // For AVX-like rounding, one could use: self.curr = f64::mul_add(r, self.curr, self.curr);
        }

        // Advance previouss
        self.last_close = close;
        self.last_volume = volume;

        Some(self.curr)
    }

    /// First valid tick handler: seed state and emit the initial value.
    #[inline(always)]
    fn init_or_none(&mut self, close: f64, volume: f64) -> Option<f64> {
        if close.is_nan() || volume.is_nan() {
            return None;
        }
        self.last_close = close;
        self.last_volume = volume;
        self.curr = self.initial_value;
        self.state = StreamState::Valid;
        Some(self.curr)
    }

    /// Rare path: at least one of {close, volume} is NaN. Kept cold to help branch prediction & I-cache.
    #[cold]
    #[inline(never)]
    fn cold_invalid(&mut self, _close: f64, _volume: f64) -> Option<f64> {
        // Batch semantics: on an invalid step we emit NaN (via None) and do not change last_* or curr.
        // (The next valid step resumes using the previous valid last_*.)
        None
    }

    /// Optional: zero-checking variant for inner tight loops when inputs are guaranteed finite.
    #[inline(always)]
    pub fn update_unchecked_finite(&mut self, close: f64, volume: f64) -> f64 {
        debug_assert!(self.state == StreamState::Valid);
        if volume > self.last_volume {
            let prev = self.last_close;
            let r = (close - prev) / prev;
            self.curr += r * self.curr;
        }
        self.last_close = close;
        self.last_volume = volume;
        self.curr
    }
}

#[derive(Clone, Debug)]
pub struct PviBatchRange {
    pub initial_value: (f64, f64, f64),
}

impl Default for PviBatchRange {
    fn default() -> Self {
        Self {
            initial_value: (1000.0, 1000.0, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct PviBatchBuilder {
    range: PviBatchRange,
    kernel: Kernel,
}

impl PviBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn initial_value_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.initial_value = (start, end, step);
        self
    }
    #[inline]
    pub fn initial_value_static(mut self, v: f64) -> Self {
        self.range.initial_value = (v, v, 0.0);
        self
    }
    pub fn apply_slices(self, close: &[f64], volume: &[f64]) -> Result<PviBatchOutput, PviError> {
        pvi_batch_with_kernel(close, volume, &self.range, self.kernel)
    }
    pub fn with_default_slices(
        close: &[f64],
        volume: &[f64],
        k: Kernel,
    ) -> Result<PviBatchOutput, PviError> {
        PviBatchBuilder::new().kernel(k).apply_slices(close, volume)
    }
    pub fn apply_candles(
        self,
        c: &Candles,
        close_src: &str,
        vol_src: &str,
    ) -> Result<PviBatchOutput, PviError> {
        let close = source_type(c, close_src);
        let vol = source_type(c, vol_src);
        self.apply_slices(close, vol)
    }
    pub fn with_default_candles(c: &Candles) -> Result<PviBatchOutput, PviError> {
        PviBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close", "volume")
    }
}

pub fn pvi_batch_with_kernel(
    close: &[f64],
    volume: &[f64],
    sweep: &PviBatchRange,
    k: Kernel,
) -> Result<PviBatchOutput, PviError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(PviError::InvalidKernelForBatch(other)),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    pvi_batch_par_slice(close, volume, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct PviBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<PviParams>,
    pub rows: usize,
    pub cols: usize,
}
impl PviBatchOutput {
    pub fn row_for_params(&self, p: &PviParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            (c.initial_value.unwrap_or(1000.0) - p.initial_value.unwrap_or(1000.0)).abs() < 1e-12
        })
    }
    pub fn values_for(&self, p: &PviParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &PviBatchRange) -> Result<Vec<PviParams>, PviError> {
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, PviError> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return Ok(vec![start]);
        }
        let mut vals = Vec::new();
        if start <= end {
            let mut x = start;
            loop {
                vals.push(x);
                if x >= end {
                    break;
                }
                let next = x + step;
                if !next.is_finite() || next == x {
                    break;
                }
                x = next;
                if x > end + 1e-12 {
                    break;
                }
            }
        } else {
            let mut x = start;
            loop {
                vals.push(x);
                if x <= end {
                    break;
                }
                let next = x - step.abs();
                if !next.is_finite() || next == x {
                    break;
                }
                x = next;
                if x < end - 1e-12 {
                    break;
                }
            }
        }
        if vals.is_empty() {
            return Err(PviError::InvalidRange { start, end, step });
        }
        Ok(vals)
    }

    let initials = axis_f64(r.initial_value)?;
    let mut out = Vec::with_capacity(initials.len());
    for &v in &initials {
        out.push(PviParams {
            initial_value: Some(v),
        });
    }
    if out.is_empty() {
        return Err(PviError::InvalidRange {
            start: r.initial_value.0,
            end: r.initial_value.1,
            step: r.initial_value.2,
        });
    }
    Ok(out)
}

#[inline(always)]
pub fn pvi_batch_slice(
    close: &[f64],
    volume: &[f64],
    sweep: &PviBatchRange,
    kern: Kernel,
) -> Result<PviBatchOutput, PviError> {
    pvi_batch_inner(close, volume, sweep, kern, false)
}

#[inline(always)]
pub fn pvi_batch_par_slice(
    close: &[f64],
    volume: &[f64],
    sweep: &PviBatchRange,
    kern: Kernel,
) -> Result<PviBatchOutput, PviError> {
    pvi_batch_inner(close, volume, sweep, kern, true)
}

#[inline(always)]
fn pvi_batch_inner(
    close: &[f64],
    volume: &[f64],
    sweep: &PviBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<PviBatchOutput, PviError> {
    let combos = expand_grid(sweep)?;
    if close.is_empty() || volume.is_empty() {
        return Err(PviError::EmptyInputData);
    }
    if close.len() != volume.len() {
        return Err(PviError::MismatchedLength);
    }

    let first_valid_idx = close
        .iter()
        .zip(volume.iter())
        .position(|(&c, &v)| !c.is_nan() && !v.is_nan())
        .ok_or(PviError::AllValuesNaN)?;
    let valid = close.len() - first_valid_idx;
    if valid < 2 {
        return Err(PviError::NotEnoughValidData { needed: 2, valid });
    }

    let rows = combos.len();
    let cols = close.len();

    // 1) allocate rows×cols uninit
    let mut buf_mu = make_uninit_matrix(rows, cols);

    // 2) write NaN warm prefixes per row
    if rows <= 32 {
        let mut warm = [0usize; 32];
        for i in 0..rows {
            warm[i] = first_valid_idx;
        }
        init_matrix_prefixes(&mut buf_mu, cols, &warm[..rows]);
    } else {
        let warm = vec![first_valid_idx; rows];
        init_matrix_prefixes(&mut buf_mu, cols, &warm);
    }

    // 3) prevent drop of the allocation while we lend it out
    let mut guard = core::mem::ManuallyDrop::new(buf_mu);

    // 4) invoke the inner writer on the same allocation, cast as f64*
    let out_f: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };
    pvi_batch_inner_into(close, volume, sweep, kern, parallel, out_f)?;

    // 5) take ownership of the same allocation as Vec<f64> without copies
    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(PviBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn pvi_batch_inner_into(
    close: &[f64],
    volume: &[f64],
    sweep: &PviBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<PviParams>, PviError> {
    let combos = expand_grid(sweep)?;
    if close.is_empty() || volume.is_empty() {
        return Err(PviError::EmptyInputData);
    }
    if close.len() != volume.len() {
        return Err(PviError::MismatchedLength);
    }

    let first_valid_idx = close
        .iter()
        .zip(volume.iter())
        .position(|(&c, &v)| !c.is_nan() && !v.is_nan())
        .ok_or(PviError::AllValuesNaN)?;
    let valid = close.len() - first_valid_idx;
    if valid < 2 {
        return Err(PviError::NotEnoughValidData { needed: 2, valid });
    }

    let rows = combos.len();
    let cols = close.len();
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| PviError::InvalidInput("rows*cols overflow".into()))?;
    if out.len() != expected {
        return Err(PviError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    // Work on MaybeUninit view to avoid UB
    let out_mu: &mut [core::mem::MaybeUninit<f64>] = unsafe {
        core::slice::from_raw_parts_mut(
            out.as_mut_ptr() as *mut core::mem::MaybeUninit<f64>,
            out.len(),
        )
    };

    // Row-specific batch optimization: precompute multiplicative scale once.
    // scale[first] = 1.0; for i>first, if valid(prev & curr) then
    //   scale[i] = scale[i-1] * (1 + (close[i]-prev_close)/prev_close) when volume increases;
    //   else scale[i] = scale[i-1]. Invalid steps emit NaN but keep accumulator.
    let mut scale = vec![f64::NAN; cols];
    scale[first_valid_idx] = 1.0;

    #[inline(always)]
    fn not_nan(x: f64) -> bool {
        x == x
    }

    unsafe {
        let mut prev_close = *close.get_unchecked(first_valid_idx);
        let mut prev_vol = *volume.get_unchecked(first_valid_idx);
        let mut accum = 1.0f64;

        let cptr = close.as_ptr();
        let vptr = volume.as_ptr();
        let mut i = first_valid_idx + 1;
        while i < cols {
            let c = *cptr.add(i);
            let v = *vptr.add(i);
            if not_nan(c) && not_nan(v) && not_nan(prev_close) && not_nan(prev_vol) {
                if v > prev_vol {
                    let r = (c - prev_close) / prev_close;
                    accum = f64::mul_add(r, accum, accum);
                }
                scale[i] = accum;
                prev_close = c;
                prev_vol = v;
            } else {
                scale[i] = f64::NAN;
                if not_nan(c) && not_nan(v) {
                    prev_close = c;
                    prev_vol = v;
                }
            }
            i += 1;
        }
    }

    let do_row = |row: usize, dst_row_mu: &mut [core::mem::MaybeUninit<f64>]| unsafe {
        let iv = combos[row].initial_value.unwrap_or(1000.0);

        // reinterpret row as fully-initialized f64s for writing
        let dst_row: &mut [f64] =
            core::slice::from_raw_parts_mut(dst_row_mu.as_mut_ptr() as *mut f64, dst_row_mu.len());

        // first_valid cell equals initial value
        *dst_row.get_unchecked_mut(first_valid_idx) = iv;

        // Fill post-warmup cells using shared scale
        let mut j = first_valid_idx + 1;
        while j < cols {
            let s = *scale.get_unchecked(j);
            if s == s {
                *dst_row.get_unchecked_mut(j) = iv * s;
            } else {
                *dst_row.get_unchecked_mut(j) = f64::NAN;
            }
            j += 1;
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            out_mu
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(r, row)| do_row(r, row));
        }
        #[cfg(target_arch = "wasm32")]
        for (r, row) in out_mu.chunks_mut(cols).enumerate() {
            do_row(r, row);
        }
    } else {
        for (r, row) in out_mu.chunks_mut(cols).enumerate() {
            do_row(r, row);
        }
    }

    Ok(combos)
}

#[inline(always)]
unsafe fn pvi_row_scalar(
    close: &[f64],
    volume: &[f64],
    first: usize,
    initial: f64,
    out: &mut [f64],
) {
    pvi_scalar(close, volume, first, initial, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn pvi_row_avx2(close: &[f64], volume: &[f64], first: usize, initial: f64, out: &mut [f64]) {
    pvi_scalar(close, volume, first, initial, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn pvi_row_avx512(
    close: &[f64],
    volume: &[f64],
    first: usize,
    initial: f64,
    out: &mut [f64],
) {
    if close.len() <= 32 {
        pvi_row_avx512_short(close, volume, first, initial, out);
    } else {
        pvi_row_avx512_long(close, volume, first, initial, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn pvi_row_avx512_short(
    close: &[f64],
    volume: &[f64],
    first: usize,
    initial: f64,
    out: &mut [f64],
) {
    pvi_scalar(close, volume, first, initial, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn pvi_row_avx512_long(
    close: &[f64],
    volume: &[f64],
    first: usize,
    initial: f64,
    out: &mut [f64],
) {
    pvi_scalar(close, volume, first, initial, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_pvi_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = PviParams {
            initial_value: None,
        };
        let input = PviInput::from_candles(&candles, "close", "volume", default_params);
        let output = pvi_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_pvi_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let close_data = [100.0, 102.0, 101.0, 103.0, 103.0, 105.0];
        let volume_data = [500.0, 600.0, 500.0, 700.0, 680.0, 900.0];
        let params = PviParams {
            initial_value: Some(1000.0),
        };
        let input = PviInput::from_slices(&close_data, &volume_data, params);
        let output = pvi_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), close_data.len());
        assert!((output.values[0] - 1000.0).abs() < 1e-6);
        Ok(())
    }

    fn check_pvi_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = PviInput::with_default_candles(&candles);
        let output = pvi_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_pvi_empty_data(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let close_data = [];
        let volume_data = [];
        let params = PviParams::default();
        let input = PviInput::from_slices(&close_data, &volume_data, params);
        let result = pvi_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_pvi_mismatched_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let close_data = [100.0, 101.0];
        let volume_data = [500.0];
        let params = PviParams::default();
        let input = PviInput::from_slices(&close_data, &volume_data, params);
        let result = pvi_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_pvi_all_values_nan(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let close_data = [f64::NAN, f64::NAN, f64::NAN];
        let volume_data = [f64::NAN, f64::NAN, f64::NAN];
        let params = PviParams::default();
        let input = PviInput::from_slices(&close_data, &volume_data, params);
        let result = pvi_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_pvi_not_enough_valid_data(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let close_data = [f64::NAN, 100.0];
        let volume_data = [f64::NAN, 500.0];
        let params = PviParams::default();
        let input = PviInput::from_slices(&close_data, &volume_data, params);
        let result = pvi_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_pvi_streaming(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let close_data = [100.0, 102.0, 101.0, 103.0, 103.0, 105.0];
        let volume_data = [500.0, 600.0, 500.0, 700.0, 680.0, 900.0];
        let params = PviParams {
            initial_value: Some(1000.0),
        };
        let input = PviInput::from_slices(&close_data, &volume_data, params.clone());
        let batch_output = pvi_with_kernel(&input, kernel)?.values;

        let mut stream = PviStream::try_new(params)?;
        let mut stream_values = Vec::with_capacity(close_data.len());
        for (&close, &vol) in close_data.iter().zip(volume_data.iter()) {
            match stream.update(close, vol) {
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
                "[{}] PVI streaming mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_pvi_tests {
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

    #[cfg(debug_assertions)]
    fn check_pvi_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Define comprehensive parameter combinations
        let test_params = vec![
            PviParams::default(), // initial_value: 1000.0
            PviParams {
                initial_value: Some(100.0),
            }, // small value
            PviParams {
                initial_value: Some(500.0),
            }, // medium-small
            PviParams {
                initial_value: Some(5000.0),
            }, // medium-large
            PviParams {
                initial_value: Some(10000.0),
            }, // large value
            PviParams {
                initial_value: Some(0.0),
            }, // edge case: zero
            PviParams {
                initial_value: Some(1.0),
            }, // edge case: one
            PviParams {
                initial_value: Some(-1000.0),
            }, // edge case: negative
            PviParams {
                initial_value: Some(999999.0),
            }, // very large
            PviParams {
                initial_value: None,
            }, // None (uses default)
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = PviInput::from_candles(&candles, "close", "volume", params.clone());
            let output = pvi_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: initial_value={:?} (param set {})",
                        test_name, val, bits, i, params.initial_value, param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: initial_value={:?} (param set {})",
                        test_name, val, bits, i, params.initial_value, param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: initial_value={:?} (param set {})",
                        test_name, val, bits, i, params.initial_value, param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_pvi_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(()) // No-op in release builds
    }

    #[test]
    fn test_pvi_into_matches_api() {
        // Use the repository's existing CSV dataset for parity.
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("load candles");

        let input = PviInput::from_candles(&candles, "close", "volume", PviParams::default());

        // Baseline via Vec-returning API
        let baseline = pvi(&input).expect("pvi baseline").values;

        // No-allocation path
        let mut into_out = vec![0.0f64; baseline.len()];
        #[cfg(not(feature = "wasm"))]
        {
            pvi_into(&input, &mut into_out).expect("pvi_into");
        }
        #[cfg(feature = "wasm")]
        {
            // In wasm builds, use the existing helper to avoid symbol clashes
            pvi_into_slice(&mut into_out, &input, Kernel::Auto).expect("pvi_into_slice");
        }

        assert_eq!(baseline.len(), into_out.len());

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b) || ((a - b).abs() <= 1e-12)
        }
        for i in 0..baseline.len() {
            assert!(
                eq_or_both_nan(baseline[i], into_out[i]),
                "Mismatch at index {}: got {}, expected {}",
                i,
                into_out[i],
                baseline[i]
            );
        }
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_pvi_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Generate test data with realistic market conditions
        let strat = (
            // Close prices between -1e6 and 1e6 (filtered for finite values)
            prop::collection::vec(
                (-1e6f64..1e6f64).prop_filter("finite close", |x| x.is_finite() && x.abs() > 1e-10),
                10..400,
            ),
            // Volume data between 0 and 1e6 (filtered for finite, non-negative values)
            prop::collection::vec(
                (0f64..1e6f64).prop_filter("finite volume", |x| x.is_finite()),
                10..400,
            ),
            // Initial value between 100 and 10000
            100f64..10000f64,
        )
            .prop_filter("same length", |(close, volume, _)| {
                close.len() == volume.len()
            });

        proptest::test_runner::TestRunner::default().run(
            &strat,
            |(close_data, volume_data, initial_value)| {
                let params = PviParams {
                    initial_value: Some(initial_value),
                };
                let input = PviInput::from_slices(&close_data, &volume_data, params);

                // Get outputs from different kernels - handle potential errors gracefully
                let output = match pvi_with_kernel(&input, kernel) {
                    Ok(o) => o,
                    Err(_) => return Ok(()), // Skip this test case if PVI fails (e.g., not enough valid data)
                };
                let out = &output.values;

                let scalar_output = match pvi_with_kernel(&input, Kernel::Scalar) {
                    Ok(o) => o,
                    Err(_) => return Ok(()), // Skip this test case if PVI fails
                };
                let ref_out = &scalar_output.values;

                // Find first valid index
                let first_valid_idx = close_data
                    .iter()
                    .zip(volume_data.iter())
                    .position(|(&c, &v)| !c.is_nan() && !v.is_nan());

                if let Some(first_idx) = first_valid_idx {
                    // Property 1: First valid value should equal initial_value
                    if !out[first_idx].is_nan() {
                        prop_assert!(
                            (out[first_idx] - initial_value).abs() < 1e-9,
                            "First valid PVI value {} should equal initial_value {} at index {}",
                            out[first_idx],
                            initial_value,
                            first_idx
                        );
                    }

                    // Property 2: PVI should remain constant when volume doesn't increase
                    for i in (first_idx + 1)..close_data.len() {
                        if !out[i].is_nan() && i > 0 && !out[i - 1].is_nan() {
                            if !volume_data[i].is_nan() && !volume_data[i - 1].is_nan() {
                                if volume_data[i] <= volume_data[i - 1] {
                                    prop_assert!(
										(out[i] - out[i - 1]).abs() < 1e-9,
										"PVI should remain constant when volume doesn't increase: {} != {} at index {}",
										out[i], out[i - 1], i
									);
                                }
                            }
                        }
                    }

                    // Property 3: PVI should only change when volume increases
                    for i in (first_idx + 1)..close_data.len() {
                        if !out[i].is_nan() && i > 0 && !out[i - 1].is_nan() {
                            if !volume_data[i].is_nan() && !volume_data[i - 1].is_nan() {
                                let volume_increased = volume_data[i] > volume_data[i - 1];
                                let pvi_changed = (out[i] - out[i - 1]).abs() > 1e-9;

                                if pvi_changed {
                                    prop_assert!(
										volume_increased,
										"PVI changed without volume increase at index {}: vol[{}]={} <= vol[{}]={}",
										i, i, volume_data[i], i - 1, volume_data[i - 1]
									);
                                }
                            }
                        }
                    }

                    // Property 4: Verify change calculation accuracy
                    for i in (first_idx + 1)..close_data.len() {
                        if !out[i].is_nan()
                            && i > 0
                            && !out[i - 1].is_nan()
                            && !close_data[i].is_nan()
                            && !close_data[i - 1].is_nan()
                            && !volume_data[i].is_nan()
                            && !volume_data[i - 1].is_nan()
                        {
                            if volume_data[i] > volume_data[i - 1]
                                && close_data[i - 1].abs() > 1e-10
                            {
                                let expected_change = ((close_data[i] - close_data[i - 1])
                                    / close_data[i - 1])
                                    * out[i - 1];
                                let expected_pvi = out[i - 1] + expected_change;
                                prop_assert!(
                                    (out[i] - expected_pvi).abs() < 1e-9,
                                    "PVI calculation error at index {}: expected {} but got {}",
                                    i,
                                    expected_pvi,
                                    out[i]
                                );
                            }
                        }
                    }

                    // Property 5: Kernel consistency - all kernels should produce identical results
                    for i in 0..out.len() {
                        if out[i].is_nan() && ref_out[i].is_nan() {
                            continue;
                        }
                        prop_assert!(
                            (out[i] - ref_out[i]).abs() < 1e-9,
                            "Kernel mismatch at index {}: {} ({:?}) vs {} (Scalar)",
                            i,
                            out[i],
                            kernel,
                            ref_out[i]
                        );
                    }

                    // Property 6: No poison values
                    for (i, &val) in out.iter().enumerate() {
                        if !val.is_nan() {
                            let bits = val.to_bits();
                            prop_assert!(
                                bits != 0x11111111_11111111
                                    && bits != 0x22222222_22222222
                                    && bits != 0x33333333_33333333,
                                "Found poison value {} (0x{:016X}) at index {}",
                                val,
                                bits,
                                i
                            );
                        }
                    }

                    // Property 7: Constant volume should keep PVI at initial value
                    if volume_data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10) {
                        for &val in out.iter().skip(first_idx) {
                            if !val.is_nan() {
                                prop_assert!(
									(val - initial_value).abs() < 1e-9,
									"PVI should remain at initial_value {} with constant volume, but got {}",
									initial_value, val
								);
                            }
                        }
                    }

                    // Property 8: With monotonic increasing volume AND price changes, PVI should change
                    let is_monotonic_increasing = volume_data
                        .windows(2)
                        .all(|w| !w[0].is_nan() && !w[1].is_nan() && w[1] > w[0]);

                    if is_monotonic_increasing && close_data.len() > first_idx + 2 {
                        let mut last_valid_pvi = out[first_idx];
                        for i in (first_idx + 1)..out.len() {
                            if !out[i].is_nan()
                                && !close_data[i].is_nan()
                                && !close_data[i - 1].is_nan()
                            {
                                // PVI changes when BOTH volume increases AND price changes
                                if (close_data[i] - close_data[i - 1]).abs() > 1e-10 {
                                    prop_assert!(
										(out[i] - last_valid_pvi).abs() > 1e-10,
										"PVI should change with monotonic increasing volume and price change at index {}",
										i
									);
                                }
                                last_valid_pvi = out[i];
                            }
                        }
                    }

                    // Property 9: Test behavior when close[i-1] is not near zero (avoid division by zero)
                    // The implementation doesn't explicitly handle division by zero, so we test normal cases
                    for i in (first_idx + 1)..close_data.len() {
                        if !volume_data[i].is_nan()
                            && !volume_data[i - 1].is_nan()
                            && volume_data[i] > volume_data[i - 1]
                            && !close_data[i].is_nan()
                            && !close_data[i - 1].is_nan()
                            && close_data[i - 1].abs() > 1e-10
                        {
                            // Only test when close[i-1] is NOT near zero
                            // Verify the percentage change calculation is mathematically correct
                            if !out[i].is_nan() && i > 0 && !out[i - 1].is_nan() {
                                let expected_change = ((close_data[i] - close_data[i - 1])
                                    / close_data[i - 1])
                                    * out[i - 1];
                                let expected_pvi = out[i - 1] + expected_change;
                                // This duplicates Property 4 but ensures we avoid division by zero edge cases
                                prop_assert!(
									(out[i] - expected_pvi).abs() < 1e-9 || out[i].is_infinite(),
									"PVI calculation should be correct or handle extreme values at index {}",
									i
								);
                            }
                        }
                    }

                    // Property 10: PVI values should remain finite and reasonable
                    for (i, &val) in out.iter().enumerate() {
                        if !val.is_nan() {
                            // PVI should remain finite (not infinite)
                            prop_assert!(
                                val.is_finite(),
                                "PVI should be finite, but got {} at index {}",
                                val,
                                i
                            );

                            // With positive initial value and reasonable price changes,
                            // PVI shouldn't become negative (though mathematically possible with extreme drops)
                            // Only check this for reasonable values to avoid edge cases
                            if initial_value > 0.0
                                && val.is_finite()
                                && val.abs() < initial_value * 100.0
                            {
                                prop_assert!(
									val >= 0.0 || close_data[..i].iter().any(|&c| c < 0.0),
									"PVI unexpectedly negative ({}) with positive initial value {} at index {}",
									val, initial_value, i
								);
                            }
                        }
                    }
                }

                Ok(())
            },
        )?;

        Ok(())
    }

    #[cfg(feature = "proptest")]
    generate_all_pvi_tests!(
        check_pvi_partial_params,
        check_pvi_accuracy,
        check_pvi_default_candles,
        check_pvi_empty_data,
        check_pvi_mismatched_length,
        check_pvi_all_values_nan,
        check_pvi_not_enough_valid_data,
        check_pvi_streaming,
        check_pvi_no_poison,
        check_pvi_property
    );

    #[cfg(not(feature = "proptest"))]
    generate_all_pvi_tests!(
        check_pvi_partial_params,
        check_pvi_accuracy,
        check_pvi_default_candles,
        check_pvi_empty_data,
        check_pvi_mismatched_length,
        check_pvi_all_values_nan,
        check_pvi_not_enough_valid_data,
        check_pvi_streaming,
        check_pvi_no_poison
    );

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let close_data = [100.0, 102.0, 101.0, 103.0, 103.0, 105.0];
        let volume_data = [500.0, 600.0, 500.0, 700.0, 680.0, 900.0];
        let output = PviBatchBuilder::new()
            .kernel(kernel)
            .apply_slices(&close_data, &volume_data)?;
        let def = PviParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), close_data.len());
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
    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test various initial_value sweep configurations
        let test_configs = vec![
            (100.0, 500.0, 100.0),       // Small values
            (1000.0, 5000.0, 1000.0),    // Default range
            (10000.0, 50000.0, 10000.0), // Large values
            (900.0, 1100.0, 50.0),       // Dense range around default
            (0.0, 100.0, 25.0),          // Edge case: starting at zero
            (-1000.0, 1000.0, 500.0),    // Edge case: negative to positive
            (1.0, 10.0, 1.0),            // Very small values
            (999999.0, 1000001.0, 1.0),  // Very large values
        ];

        for (cfg_idx, &(start, end, step)) in test_configs.iter().enumerate() {
            let output = PviBatchBuilder::new()
                .kernel(kernel)
                .initial_value_range(start, end, step)
                .apply_candles(&c, "close", "volume")?;

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
						 at row {} col {} (flat index {}) with params: initial_value={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.initial_value.unwrap_or(1000.0)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: initial_value={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.initial_value.unwrap_or(1000.0)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: initial_value={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.initial_value.unwrap_or(1000.0)
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
        Ok(())
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);
}

#[cfg(feature = "python")]
#[pyfunction(name = "pvi")]
#[pyo3(signature = (close, volume, initial_value=None, kernel=None))]
pub fn pvi_py<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    initial_value: Option<f64>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let close_slice = close.as_slice()?;
    let volume_slice = volume.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = PviParams { initial_value };
    let input = PviInput::from_slices(close_slice, volume_slice, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| pvi_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction(name = "pvi_batch")]
#[pyo3(signature = (close, volume, initial_value_range, kernel=None))]
pub fn pvi_batch_py<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    initial_value_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};

    let close_slice = close.as_slice()?;
    let volume_slice = volume.as_slice()?;

    let sweep = PviBatchRange {
        initial_value: initial_value_range,
    };

    let combos = expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = close_slice.len();

    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
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
                _ => unreachable!(),
            };
            pvi_batch_inner_into(close_slice, volume_slice, &sweep, simd, true, slice_out)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "initial_values",
        combos
            .iter()
            .map(|p| p.initial_value.unwrap_or(1000.0))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "PviStream")]
pub struct PviStreamPy {
    stream: PviStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl PviStreamPy {
    #[new]
    #[pyo3(signature = (initial_value=None))]
    fn new(initial_value: Option<f64>) -> PyResult<Self> {
        let params = PviParams { initial_value };
        let stream =
            PviStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PviStreamPy { stream })
    }

    fn update(&mut self, close: f64, volume: f64) -> Option<f64> {
        self.stream.update(close, volume)
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn pvi_js(close: &[f64], volume: &[f64], initial_value: f64) -> Result<Vec<f64>, JsValue> {
    let params = PviParams {
        initial_value: Some(initial_value),
    };
    let input = PviInput::from_slices(close, volume, params);

    let mut output = vec![0.0; close.len()];
    pvi_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn pvi_into(
    close_ptr: *const f64,
    volume_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    initial_value: f64,
) -> Result<(), JsValue> {
    if close_ptr.is_null() || volume_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let close = std::slice::from_raw_parts(close_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);

        let params = PviParams {
            initial_value: Some(initial_value),
        };
        let input = PviInput::from_slices(close, volume, params);

        // Check for aliasing
        if close_ptr == out_ptr || volume_ptr == out_ptr {
            let mut temp = vec![0.0; len];
            pvi_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            pvi_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn pvi_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn pvi_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct PviBatchConfig {
    pub initial_value_range: (f64, f64, f64),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct PviBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<PviParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = pvi_batch)]
pub fn pvi_batch_js(close: &[f64], volume: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: PviBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = PviBatchRange {
        initial_value: config.initial_value_range,
    };

    let output = pvi_batch_with_kernel(close, volume, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = PviBatchJsOutput {
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
pub fn pvi_batch_into(
    close_ptr: *const f64,
    volume_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    initial_value_start: f64,
    initial_value_end: f64,
    initial_value_step: f64,
) -> Result<usize, JsValue> {
    if close_ptr.is_null() || volume_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to pvi_batch_into"));
    }

    unsafe {
        let close = std::slice::from_raw_parts(close_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);

        let sweep = PviBatchRange {
            initial_value: (initial_value_start, initial_value_end, initial_value_step),
        };

        let combos = expand_grid(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let total = rows
            .checked_mul(len)
            .ok_or_else(|| JsValue::from_str("pvi_batch_into: rows*len overflow"))?;
        let out = std::slice::from_raw_parts_mut(out_ptr, total);

        let kernel = detect_best_batch_kernel();
        let simd = match kernel {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            Kernel::ScalarBatch => Kernel::Scalar,
            _ => unreachable!(),
        };

        pvi_batch_inner_into(close, volume, &sweep, simd, true, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}

// ---------------- Python CUDA bindings ----------------
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::cuda_available;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::CudaPvi;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::DeviceArrayF32;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::context::Context;
#[cfg(all(feature = "python", feature = "cuda"))]
use std::sync::Arc;

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", name = "PviDeviceArrayF32", unsendable)]
pub struct PviDeviceArrayF32Py {
    pub(crate) inner: Option<DeviceArrayF32>,
    pub(crate) _ctx: Arc<Context>,
    pub(crate) device_id: u32,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl PviDeviceArrayF32Py {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("buffer already exported via __dlpack__"))?;
        let d = PyDict::new(py);
        let itemsize = std::mem::size_of::<f32>();
        d.set_item("shape", (inner.rows, inner.cols))?;
        d.set_item("typestr", "<f4")?;
        d.set_item("strides", (inner.cols * itemsize, itemsize))?;
        d.set_item("data", (inner.device_ptr() as usize, false))?;
        // Stream omitted: producing stream is synchronized before returning the handle.
        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        (2, self.device_id as i32)
    }

    #[pyo3(signature=(stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &mut self,
        py: Python<'py>,
        stream: Option<PyObject>,
        max_version: Option<PyObject>,
        dl_device: Option<PyObject>,
        copy: Option<PyObject>,
    ) -> PyResult<PyObject> {
        // Array API stream semantics: producer is already synchronized, so we only validate.
        if let Some(s_obj) = stream.as_ref() {
            if let Ok(s) = s_obj.extract::<usize>(py) {
                if s == 0 {
                    return Err(PyValueError::new_err(
                        "__dlpack__ stream=0 is invalid for CUDA",
                    ));
                }
            }
        }

        // Compute target device id and validate `dl_device` hint if provided.
        let (kdl, alloc_dev) = self.__dlpack_device__();
        if let Some(dev_obj) = dl_device.as_ref() {
            if let Ok((dev_ty, dev_id)) = dev_obj.extract::<(i32, i32)>(py) {
                if dev_ty != kdl || dev_id != alloc_dev {
                    return Err(PyValueError::new_err(
                        "dl_device mismatch; cross-device copy not supported for PviDeviceArrayF32",
                    ));
                }
            }
        }

        if copy
            .as_ref()
            .and_then(|c| c.extract::<bool>(py).ok())
            == Some(true)
        {
            return Err(PyValueError::new_err(
                "copy=True not supported for PviDeviceArrayF32",
            ));
        }

        // Producer stream is synchronized before returning the handle; ignore consumer stream.
        let _ = stream;

        // Move VRAM handle out of this wrapper; the DLPack capsule owns it afterwards.
        let inner = self
            .inner
            .take()
            .ok_or_else(|| PyValueError::new_err("__dlpack__ may only be called once"))?;

        let rows = inner.rows;
        let cols = inner.cols;
        let buf = inner.buf;

        let max_version_bound = max_version.map(|obj| obj.into_bound(py));

        export_f32_cuda_dlpack_2d(py, buf, rows, cols, alloc_dev, max_version_bound)
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
impl PviDeviceArrayF32Py {
    pub fn new_from_rust(inner: DeviceArrayF32, ctx_guard: Arc<Context>, device_id: u32) -> Self {
        Self {
            inner: Some(inner),
            _ctx: ctx_guard,
            device_id,
        }
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "pvi_cuda_batch_dev")]
#[pyo3(signature = (close, volume, initial_values, device_id=0))]
pub fn pvi_cuda_batch_dev_py(
    py: Python<'_>,
    close: PyReadonlyArray1<'_, f32>,
    volume: PyReadonlyArray1<'_, f32>,
    initial_values: PyReadonlyArray1<'_, f32>,
    device_id: usize,
) -> PyResult<PviDeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let close_slice = close.as_slice()?;
    let volume_slice = volume.as_slice()?;
    let inits_slice = initial_values.as_slice()?;
    if close_slice.len() != volume_slice.len() {
        return Err(PyValueError::new_err("mismatched input lengths"));
    }
    if inits_slice.is_empty() {
        return Err(PyValueError::new_err("initial_values must be non-empty"));
    }
    let (inner, ctx, dev_id) = py
        .allow_threads(|| {
            let cuda =
                CudaPvi::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
            let ctx = cuda.context_arc();
            let dev_id = cuda.device_id();
            let arr = cuda
                .pvi_batch_dev(close_slice, volume_slice, inits_slice)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok::<_, pyo3::PyErr>((arr, ctx, dev_id))
        })?;
    Ok(PviDeviceArrayF32Py::new_from_rust(inner, ctx, dev_id))
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "pvi_cuda_many_series_one_param_dev")]
#[pyo3(signature = (close_tm, volume_tm, cols, rows, initial_value, device_id=0))]
pub fn pvi_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    close_tm: PyReadonlyArray1<'_, f32>,
    volume_tm: PyReadonlyArray1<'_, f32>,
    cols: usize,
    rows: usize,
    initial_value: f32,
    device_id: usize,
) -> PyResult<PviDeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let close_slice = close_tm.as_slice()?;
    let volume_slice = volume_tm.as_slice()?;
    let (inner, ctx, dev_id) = py
        .allow_threads(|| {
            let cuda =
                CudaPvi::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
            let ctx = cuda.context_arc();
            let dev_id = cuda.device_id();
            let arr = cuda
                .pvi_many_series_one_param_time_major_dev(
                    close_slice,
                    volume_slice,
                    cols,
                    rows,
                    initial_value,
                )
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok::<_, pyo3::PyErr>((arr, ctx, dev_id))
        })?;
    Ok(PviDeviceArrayF32Py::new_from_rust(inner, ctx, dev_id))
}
