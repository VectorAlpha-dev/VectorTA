//! # Klinger Volume Oscillator (KVO)
//!
//! The Klinger Volume Oscillator (KVO) is designed to capture long-term
//! money flow trends, while remaining sensitive enough to short-term
//! fluctuations. It uses high, low, close prices and volume to measure
//! volume force (VF), then applies two separate EMAs (short and long)
//! to VF and calculates the difference.
//!
//! ## Parameters
//! - **short_period**: The short EMA period (default: 2)
//! - **long_period**: The long EMA period (default: 5)
//!
//! ## Returns
//! - **`Ok(KvoOutput)`** containing a `Vec<f64>` of oscillator values
//!
//! ## Developer Notes
//! ### Implementation Status
//! - SIMD: Single-series AVX2/AVX512 kept as stubs redirecting to scalar due to loop-carried dependencies (trend/CM + EMA) making vectorization across time unprofitable; scalar is fastest and passes all tests.
//! - **AVX2 Kernel**: Stub (calls scalar implementation)
//! - **AVX512 Kernel**: Stub with short/long variants (both call scalar)
//! - **Streaming Update**: O(1) - efficient with maintained EMA states and trend tracking
//! - **Memory Optimization**: Fully optimized with `alloc_with_nan_prefix` for output vectors
//! - **Batch Operations**: Optimized row path by precomputing the shared VF series once (O(N)), then running per-row EMA updates over VF; uses `make_uninit_matrix` and `init_matrix_prefixes`
//!
//! ### TODO - Performance Improvements
//! - [ ] Implement actual AVX2/AVX512 batch SIMD (rows-in-lanes) over shared VF if/when beneficial
//! - [ ] Consider AVX512 short/long specialized row kernels only if measurable wins

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

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::alma::DeviceArrayF32Py;
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
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[derive(Debug, Clone)]
pub enum KvoData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct KvoOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct KvoParams {
    pub short_period: Option<usize>,
    pub long_period: Option<usize>,
}

impl Default for KvoParams {
    fn default() -> Self {
        Self {
            short_period: Some(2),
            long_period: Some(5),
        }
    }
}

#[derive(Debug, Clone)]
pub struct KvoInput<'a> {
    pub data: KvoData<'a>,
    pub params: KvoParams,
}

impl<'a> KvoInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: KvoParams) -> Self {
        Self {
            data: KvoData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
        params: KvoParams,
    ) -> Self {
        Self {
            data: KvoData::Slices {
                high,
                low,
                close,
                volume,
            },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, KvoParams::default())
    }

    #[inline]
    pub fn get_short_period(&self) -> usize {
        self.params.short_period.unwrap_or(2)
    }

    #[inline]
    pub fn get_long_period(&self) -> usize {
        self.params.long_period.unwrap_or(5)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct KvoBuilder {
    short_period: Option<usize>,
    long_period: Option<usize>,
    kernel: Kernel,
}

impl Default for KvoBuilder {
    fn default() -> Self {
        Self {
            short_period: None,
            long_period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl KvoBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn short_period(mut self, n: usize) -> Self {
        self.short_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn long_period(mut self, n: usize) -> Self {
        self.long_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<KvoOutput, KvoError> {
        let params = KvoParams {
            short_period: self.short_period,
            long_period: self.long_period,
        };
        let input = KvoInput::from_candles(c, params);
        kvo_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Result<KvoOutput, KvoError> {
        let params = KvoParams {
            short_period: self.short_period,
            long_period: self.long_period,
        };
        let input = KvoInput::from_slices(high, low, close, volume, params);
        kvo_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<KvoStream, KvoError> {
        let params = KvoParams {
            short_period: self.short_period,
            long_period: self.long_period,
        };
        KvoStream::try_new(params)
    }
}

#[derive(Debug, Error)]
pub enum KvoError {
    #[error("kvo: Empty data provided.")]
    EmptyData,
    #[error("kvo: Invalid period settings: short={short}, long={long}")]
    InvalidPeriod { short: usize, long: usize },
    #[error("kvo: Not enough valid data: found {valid} valid points after the first valid index.")]
    NotEnoughValidData { valid: usize },
    #[error("kvo: All values are NaN.")]
    AllValuesNaN,
    #[error("kvo: Output buffer length mismatch: got={got}, expected={expected}")]
    OutputLenMismatch { got: usize, expected: usize },
}

#[inline]
pub fn kvo(input: &KvoInput) -> Result<KvoOutput, KvoError> {
    kvo_with_kernel(input, Kernel::Auto)
}

pub fn kvo_with_kernel(input: &KvoInput, kernel: Kernel) -> Result<KvoOutput, KvoError> {
    let (high, low, close, volume): (&[f64], &[f64], &[f64], &[f64]) = match &input.data {
        KvoData::Candles { candles } => (
            source_type(candles, "high"),
            source_type(candles, "low"),
            source_type(candles, "close"),
            source_type(candles, "volume"),
        ),
        KvoData::Slices {
            high,
            low,
            close,
            volume,
        } => (*high, *low, *close, *volume),
    };

    if high.is_empty() || low.is_empty() || close.is_empty() || volume.is_empty() {
        return Err(KvoError::EmptyData);
    }

    let short_period = input.get_short_period();
    let long_period = input.get_long_period();
    if short_period < 1 || long_period < short_period {
        return Err(KvoError::InvalidPeriod {
            short: short_period,
            long: long_period,
        });
    }

    let first_valid_idx = high
        .iter()
        .zip(low.iter())
        .zip(close.iter())
        .zip(volume.iter())
        .position(|(((h, l), c), v)| !h.is_nan() && !l.is_nan() && !c.is_nan() && !v.is_nan());
    let first_valid_idx = match first_valid_idx {
        Some(idx) => idx,
        None => return Err(KvoError::AllValuesNaN),
    };

    if (high.len() - first_valid_idx) < 2 {
        return Err(KvoError::NotEnoughValidData {
            valid: high.len() - first_valid_idx,
        });
    }

    let mut out = alloc_with_nan_prefix(high.len(), first_valid_idx + 1);
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => kvo_scalar(
                high,
                low,
                close,
                volume,
                short_period,
                long_period,
                first_valid_idx,
                &mut out,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => kvo_avx2(
                high,
                low,
                close,
                volume,
                short_period,
                long_period,
                first_valid_idx,
                &mut out,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => kvo_avx512(
                high,
                low,
                close,
                volume,
                short_period,
                long_period,
                first_valid_idx,
                &mut out,
            ),
            _ => unreachable!(),
        }
    }
    Ok(KvoOutput { values: out })
}

/// Compute KVO directly into the provided output buffer for zero-copy operations
#[inline]
pub fn kvo_compute_into(out: &mut [f64], input: &KvoInput, kernel: Kernel) -> Result<(), KvoError> {
    let (high, low, close, volume): (&[f64], &[f64], &[f64], &[f64]) = match &input.data {
        KvoData::Candles { candles } => (
            source_type(candles, "high"),
            source_type(candles, "low"),
            source_type(candles, "close"),
            source_type(candles, "volume"),
        ),
        KvoData::Slices {
            high,
            low,
            close,
            volume,
        } => (*high, *low, *close, *volume),
    };

    if high.is_empty() || low.is_empty() || close.is_empty() || volume.is_empty() {
        return Err(KvoError::EmptyData);
    }

    // Validate output buffer length matches input length
    if out.len() != high.len() {
        return Err(KvoError::OutputLenMismatch {
            got: out.len(),
            expected: high.len(),
        });
    }

    let short_period = input.get_short_period();
    let long_period = input.get_long_period();
    if short_period < 1 || long_period < short_period {
        return Err(KvoError::InvalidPeriod {
            short: short_period,
            long: long_period,
        });
    }

    let first_valid_idx = high
        .iter()
        .zip(low.iter())
        .zip(close.iter())
        .zip(volume.iter())
        .position(|(((h, l), c), v)| !h.is_nan() && !l.is_nan() && !c.is_nan() && !v.is_nan());
    let first_valid_idx = match first_valid_idx {
        Some(idx) => idx,
        None => return Err(KvoError::AllValuesNaN),
    };

    if (high.len() - first_valid_idx) < 2 {
        return Err(KvoError::NotEnoughValidData {
            valid: high.len() - first_valid_idx,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => kvo_scalar(
                high,
                low,
                close,
                volume,
                short_period,
                long_period,
                first_valid_idx,
                out,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => kvo_avx2(
                high,
                low,
                close,
                volume,
                short_period,
                long_period,
                first_valid_idx,
                out,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => kvo_avx512(
                high,
                low,
                close,
                volume,
                short_period,
                long_period,
                first_valid_idx,
                out,
            ),
            _ => unreachable!(),
        }
    }

    // Fill warmup period with NaN
    for v in &mut out[..=first_valid_idx] {
        *v = f64::NAN;
    }

    Ok(())
}

/// Compute KVO into slice with API matching alma.rs
#[inline]
pub fn kvo_into_slice(dst: &mut [f64], input: &KvoInput, kern: Kernel) -> Result<(), KvoError> {
    kvo_compute_into(dst, input, kern)
}

#[inline]
pub unsafe fn kvo_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    short_period: usize,
    long_period: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    // Precompute EMA alphas
    let short_alpha = 2.0 / (short_period as f64 + 1.0);
    let long_alpha = 2.0 / (long_period as f64 + 1.0);

    // Raw pointers to elide bounds checks in the hot loop
    let hp = high.as_ptr();
    let lp = low.as_ptr();
    let cp = close.as_ptr();
    let vp = volume.as_ptr();
    let outp = out.as_mut_ptr();

    // Seed state from the first valid bar
    let mut trend: i32 = -1;
    let mut cm: f64 = 0.0;

    let mut prev_hlc =
        *hp.add(first_valid_idx) + *lp.add(first_valid_idx) + *cp.add(first_valid_idx);
    let mut prev_dm = *hp.add(first_valid_idx) - *lp.add(first_valid_idx);

    // EMA state
    let mut short_ema = 0.0f64;
    let mut long_ema = 0.0f64;

    // Main pass
    let mut i = first_valid_idx + 1;
    let len = high.len();
    while i < len {
        // Loads
        let h = *hp.add(i);
        let l = *lp.add(i);
        let c = *cp.add(i);
        let v = *vp.add(i);

        // Aggregates
        let hlc = h + l + c;
        let dm = h - l;

        // Trend + CM update
        if hlc > prev_hlc && trend != 1 {
            trend = 1;
            cm = prev_dm;
        } else if hlc < prev_hlc && trend != 0 {
            trend = 0;
            cm = prev_dm;
        }
        cm += dm;

        // Volume force
        let temp = ((dm / cm) * 2.0 - 1.0).abs();
        let sign = if trend == 1 { 1.0 } else { -1.0 };
        let vf = v * temp * 100.0 * sign;

        // EMA updates
        if i == first_valid_idx + 1 {
            short_ema = vf;
            long_ema = vf;
        } else {
            short_ema += (vf - short_ema) * short_alpha;
            long_ema += (vf - long_ema) * long_alpha;
        }

        // Store
        *outp.add(i) = short_ema - long_ema;

        // Advance state
        prev_hlc = hlc;
        prev_dm = dm;
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn kvo_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    short_period: usize,
    long_period: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    kvo_scalar(
        high,
        low,
        close,
        volume,
        short_period,
        long_period,
        first_valid_idx,
        out,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn kvo_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    short_period: usize,
    long_period: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    if short_period <= 32 && long_period <= 32 {
        kvo_avx512_short(
            high,
            low,
            close,
            volume,
            short_period,
            long_period,
            first_valid_idx,
            out,
        )
    } else {
        kvo_avx512_long(
            high,
            low,
            close,
            volume,
            short_period,
            long_period,
            first_valid_idx,
            out,
        )
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn kvo_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    short_period: usize,
    long_period: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    kvo_scalar(
        high,
        low,
        close,
        volume,
        short_period,
        long_period,
        first_valid_idx,
        out,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn kvo_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    short_period: usize,
    long_period: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    kvo_scalar(
        high,
        low,
        close,
        volume,
        short_period,
        long_period,
        first_valid_idx,
        out,
    )
}

#[derive(Clone, Debug)]
pub struct KvoBatchRange {
    pub short_period: (usize, usize, usize),
    pub long_period: (usize, usize, usize),
}

impl Default for KvoBatchRange {
    fn default() -> Self {
        Self {
            short_period: (2, 10, 1),
            long_period: (5, 20, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct KvoBatchBuilder {
    range: KvoBatchRange,
    kernel: Kernel,
}

impl KvoBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline]
    pub fn short_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.short_period = (start, end, step);
        self
    }
    #[inline]
    pub fn short_static(mut self, v: usize) -> Self {
        self.range.short_period = (v, v, 0);
        self
    }
    #[inline]
    pub fn long_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.long_period = (start, end, step);
        self
    }
    #[inline]
    pub fn long_static(mut self, v: usize) -> Self {
        self.range.long_period = (v, v, 0);
        self
    }

    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Result<KvoBatchOutput, KvoError> {
        kvo_batch_with_kernel(high, low, close, volume, &self.range, self.kernel)
    }
    pub fn with_default_slices(
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
        k: Kernel,
    ) -> Result<KvoBatchOutput, KvoError> {
        KvoBatchBuilder::new()
            .kernel(k)
            .apply_slices(high, low, close, volume)
    }

    pub fn apply_candles(self, c: &Candles) -> Result<KvoBatchOutput, KvoError> {
        let high = source_type(c, "high");
        let low = source_type(c, "low");
        let close = source_type(c, "close");
        let volume = source_type(c, "volume");
        self.apply_slices(high, low, close, volume)
    }

    pub fn with_default_candles(c: &Candles, k: Kernel) -> Result<KvoBatchOutput, KvoError> {
        KvoBatchBuilder::new().kernel(k).apply_candles(c)
    }
}

pub fn kvo_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &KvoBatchRange,
    k: Kernel,
) -> Result<KvoBatchOutput, KvoError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(KvoError::InvalidPeriod { short: 0, long: 0 }),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    kvo_batch_par_slice(high, low, close, volume, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct KvoBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<KvoParams>,
    pub rows: usize,
    pub cols: usize,
}
impl KvoBatchOutput {
    pub fn row_for_params(&self, p: &KvoParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.short_period.unwrap_or(2) == p.short_period.unwrap_or(2)
                && c.long_period.unwrap_or(5) == p.long_period.unwrap_or(5)
        })
    }
    pub fn values_for(&self, p: &KvoParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &KvoBatchRange) -> Vec<KvoParams> {
    fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let shorts = axis(r.short_period);
    let longs = axis(r.long_period);
    let mut out = Vec::with_capacity(shorts.len() * longs.len());
    for &s in &shorts {
        for &l in &longs {
            if s >= 1 && l >= s {
                out.push(KvoParams {
                    short_period: Some(s),
                    long_period: Some(l),
                });
            }
        }
    }
    out
}

#[inline(always)]
pub fn kvo_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &KvoBatchRange,
    kern: Kernel,
) -> Result<KvoBatchOutput, KvoError> {
    kvo_batch_inner(high, low, close, volume, sweep, kern, false)
}
#[inline(always)]
pub fn kvo_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &KvoBatchRange,
    kern: Kernel,
) -> Result<KvoBatchOutput, KvoError> {
    kvo_batch_inner(high, low, close, volume, sweep, kern, true)
}

/// Compute KVO batch directly into the provided output buffer for zero-copy operations
/// The output buffer must have size: rows * cols where rows is the number of parameter combinations
/// and cols is the length of the input data. Data is laid out in row-major order.
#[inline]
pub fn kvo_batch_into_slice(
    out: &mut [f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &KvoBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<Vec<KvoParams>, KvoError> {
    // Validate that all input slices have the same length
    let len = high.len();
    if low.len() != len || close.len() != len || volume.len() != len {
        return Err(KvoError::OutputLenMismatch {
            got: out.len(),
            expected: len,
        });
    }

    // Calculate expected output size
    let combos = expand_grid(sweep);
    let expected_size = combos.len() * len;

    if out.len() != expected_size {
        return Err(KvoError::OutputLenMismatch {
            got: out.len(),
            expected: expected_size,
        });
    }

    kvo_batch_inner_into(high, low, close, volume, sweep, kern, parallel, out)
}

#[inline(always)]
fn kvo_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &KvoBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<KvoBatchOutput, KvoError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(KvoError::InvalidPeriod { short: 0, long: 0 });
    }
    let cols = high.len();
    let rows = combos.len();

    // 1) allocate rows×cols uninit
    let mut buf_mu = make_uninit_matrix(rows, cols);

    // 2) init NaN warmup using first+1 per row, same as before
    let first = high
        .iter()
        .zip(low)
        .zip(close)
        .zip(volume)
        .position(|(((h, l), c), v)| !h.is_nan() && !l.is_nan() && !c.is_nan() && !v.is_nan())
        .ok_or(KvoError::AllValuesNaN)?;
    let warm: Vec<usize> = combos.iter().map(|_| first + 1).collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    // 3) expose &mut [f64] view to write into in-place
    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out_slice: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    // 4) write rows directly into the same allocation
    let simd = match kern {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };
    let combos_back =
        kvo_batch_inner_into(high, low, close, volume, sweep, simd, parallel, out_slice)?;

    // 5) reclaim as Vec<f64> without copying
    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(KvoBatchOutput {
        values,
        combos: combos_back,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn kvo_row_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    first: usize,
    short_period: usize,
    long_period: usize,
    out: &mut [f64],
) {
    kvo_scalar(
        high,
        low,
        close,
        volume,
        short_period,
        long_period,
        first,
        out,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn kvo_row_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    first: usize,
    short_period: usize,
    long_period: usize,
    out: &mut [f64],
) {
    kvo_scalar(
        high,
        low,
        close,
        volume,
        short_period,
        long_period,
        first,
        out,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn kvo_row_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    first: usize,
    short_period: usize,
    long_period: usize,
    out: &mut [f64],
) {
    if short_period <= 32 && long_period <= 32 {
        kvo_row_avx512_short(
            high,
            low,
            close,
            volume,
            first,
            short_period,
            long_period,
            out,
        )
    } else {
        kvo_row_avx512_long(
            high,
            low,
            close,
            volume,
            first,
            short_period,
            long_period,
            out,
        )
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn kvo_row_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    first: usize,
    short_period: usize,
    long_period: usize,
    out: &mut [f64],
) {
    kvo_scalar(
        high,
        low,
        close,
        volume,
        short_period,
        long_period,
        first,
        out,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn kvo_row_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    first: usize,
    short_period: usize,
    long_period: usize,
    out: &mut [f64],
) {
    kvo_scalar(
        high,
        low,
        close,
        volume,
        short_period,
        long_period,
        first,
        out,
    )
}

// Decision: Streaming kernel tightened; semantics preserved (O(1)).
// Caches sign and reduces branches; warmup behavior unchanged.
#[derive(Debug, Clone)]
pub struct KvoStream {
    // periods & smoothing
    short_period: usize,
    long_period: usize,
    short_alpha: f64,
    long_alpha: f64,

    // rolling state
    prev_hlc: f64,
    prev_dm: f64,
    cm: f64,
    trend: i32, // 1 = up, 0 = down (init -1 before we know)
    sign: f64,  // +1.0 for up, -1.0 for down (cached)

    // EMA state
    short_ema: f64,
    long_ema: f64,

    // warmup flags
    first: bool,  // waiting for seed of prev_hlc/prev_dm
    seeded: bool, // EMA seeded with first vf
}

impl KvoStream {
    pub fn try_new(params: KvoParams) -> Result<Self, KvoError> {
        let short_period = params.short_period.unwrap_or(2);
        let long_period = params.long_period.unwrap_or(5);
        if short_period < 1 || long_period < short_period {
            return Err(KvoError::InvalidPeriod {
                short: short_period,
                long: long_period,
            });
        }
        Ok(Self {
            short_period,
            long_period,
            short_alpha: 2.0 / (short_period as f64 + 1.0),
            long_alpha: 2.0 / (long_period as f64 + 1.0),
            prev_hlc: 0.0,
            prev_dm: 0.0,
            cm: 0.0,
            trend: -1,  // unknown at start
            sign: -1.0, // matches trend = -1 default (tie -> -1)
            short_ema: 0.0,
            long_ema: 0.0,
            first: true,
            seeded: false,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> Option<f64> {
        // First bar: seed prev_* and wait for the next bar to start producing values
        if self.first {
            self.prev_hlc = high + low + close;
            self.prev_dm = high - low;
            self.first = false;
            return None;
        }

        // Compute current aggregates
        let hlc = high + low + close;
        let dm = high - low;

        // Trend & CM update (minimize branches, semantics unchanged)
        if hlc > self.prev_hlc {
            if self.trend != 1 {
                self.trend = 1;
                self.cm = self.prev_dm; // reset on flip
                self.sign = 1.0;
            }
        } else if hlc < self.prev_hlc {
            if self.trend != 0 {
                self.trend = 0;
                self.cm = self.prev_dm; // reset on flip
                self.sign = -1.0;
            }
        }
        self.cm += dm;

        // Volume Force (VF): VF = V * | 2*(dm/cm) - 1 | * 100 * sign
        let temp = ((dm / self.cm) * 2.0 - 1.0).abs();
        let vf = volume * temp * 100.0 * self.sign;

        // EMA updates
        if !self.seeded {
            self.short_ema = vf;
            self.long_ema = vf;
            self.seeded = true;
        } else {
            self.short_ema += (vf - self.short_ema) * self.short_alpha;
            self.long_ema += (vf - self.long_ema) * self.long_alpha;
        }

        // advance state
        self.prev_hlc = hlc;
        self.prev_dm = dm;

        Some(self.short_ema - self.long_ema)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;

    fn check_kvo_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = KvoParams {
            short_period: None,
            long_period: None,
        };
        let input = KvoInput::from_candles(&candles, default_params);
        let output = kvo_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_kvo_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = KvoInput::from_candles(&candles, KvoParams::default());
        let result = kvo_with_kernel(&input, kernel)?;
        let expected_last_five = [
            -246.42698280402647,
            530.8651474164992,
            237.2148311016648,
            608.8044103976362,
            -6339.615516805162,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-1,
                "[{}] KVO {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_kvo_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = KvoInput::with_default_candles(&candles);
        match input.data {
            KvoData::Candles { .. } => {}
            _ => panic!("Expected KvoData::Candles"),
        }
        let output = kvo_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_kvo_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = KvoParams {
            short_period: Some(0),
            long_period: Some(5),
        };
        let input = KvoInput::from_candles(&candles, params);
        let res = kvo_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] KVO should fail with zero short period",
            test_name
        );
        Ok(())
    }

    fn check_kvo_period_invalid(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = KvoParams {
            short_period: Some(5),
            long_period: Some(2),
        };
        let input = KvoInput::from_candles(&candles, params);
        let res = kvo_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] KVO should fail with long_period < short_period",
            test_name
        );
        Ok(())
    }

    fn check_kvo_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let mut candles = read_candles_from_csv(file_path)?;
        candles.high.truncate(1);
        candles.low.truncate(1);
        candles.close.truncate(1);
        candles.volume.truncate(1);
        let input = KvoInput::from_candles(&candles, KvoParams::default());
        let res = kvo_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] KVO should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_kvo_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = KvoParams {
            short_period: Some(2),
            long_period: Some(5),
        };
        let first_input = KvoInput::from_candles(&candles, first_params);
        let first_result = kvo_with_kernel(&first_input, kernel)?;
        let second_params = KvoParams {
            short_period: Some(2),
            long_period: Some(5),
        };
        let second_input = KvoInput::from_slices(
            &candles.high,
            &candles.low,
            &candles.close,
            &first_result.values,
            second_params,
        );
        let _ = kvo_with_kernel(&second_input, kernel);
        Ok(())
    }

    fn check_kvo_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = KvoInput::from_candles(&candles, KvoParams::default());
        let res = kvo_with_kernel(&input, kernel)?;
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

    fn check_kvo_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let short = 2;
        let long = 5;

        let input = KvoInput::from_candles(
            &candles,
            KvoParams {
                short_period: Some(short),
                long_period: Some(long),
            },
        );
        let batch_output = kvo_with_kernel(&input, kernel)?.values;

        let mut stream = KvoStream::try_new(KvoParams {
            short_period: Some(short),
            long_period: Some(long),
        })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for ((&h, &l), (&c, &v)) in candles
            .high
            .iter()
            .zip(&candles.low)
            .zip(candles.close.iter().zip(&candles.volume))
        {
            match stream.update(h, l, c, v) {
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
                "[{}] KVO streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_kvo_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let test_params = vec![
            KvoParams::default(), // short_period: 2, long_period: 5
            KvoParams {
                short_period: Some(1),
                long_period: Some(1),
            },
            KvoParams {
                short_period: Some(1),
                long_period: Some(2),
            },
            KvoParams {
                short_period: Some(2),
                long_period: Some(2),
            },
            KvoParams {
                short_period: Some(3),
                long_period: Some(5),
            },
            KvoParams {
                short_period: Some(5),
                long_period: Some(10),
            },
            KvoParams {
                short_period: Some(10),
                long_period: Some(20),
            },
            KvoParams {
                short_period: Some(20),
                long_period: Some(50),
            },
            KvoParams {
                short_period: Some(50),
                long_period: Some(100),
            },
            KvoParams {
                short_period: Some(100),
                long_period: Some(200),
            },
            KvoParams {
                short_period: Some(2),
                long_period: Some(100),
            },
            KvoParams {
                short_period: Some(1),
                long_period: Some(200),
            },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = KvoInput::from_candles(&candles, params.clone());
            let output = kvo_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: short_period={}, long_period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.short_period.unwrap_or(2),
                        params.long_period.unwrap_or(5),
                        param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: short_period={}, long_period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.short_period.unwrap_or(2),
                        params.long_period.unwrap_or(5),
                        param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: short_period={}, long_period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.short_period.unwrap_or(2),
                        params.long_period.unwrap_or(5),
                        param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_kvo_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(()) // No-op in release builds
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_kvo_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        // Strategy for generating realistic OHLCV data
        let strat = (2usize..=10, 5usize..=20) // (short_period, long_period)
            .prop_flat_map(|(short, long)| {
                (
                    // Generate data length between 50 and 500 points
                    prop::collection::vec(
                        (
                            // Generate realistic OHLC prices
                            (100.0f64..10000.0f64).prop_filter("finite", |x| x.is_finite()),
                            0.01f64..0.1f64, // volatility factor
                            // Volume between 100 and 1M
                            (100.0f64..1_000_000.0f64)
                                .prop_filter("positive", |x| *x > 0.0 && x.is_finite()),
                        ),
                        50..=500,
                    ),
                    Just(short),
                    Just(long.max(short)), // Ensure long >= short
                )
            });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(price_data, short_period, long_period)| {
                // Generate OHLCV data from price seeds
                let mut high = Vec::with_capacity(price_data.len());
                let mut low = Vec::with_capacity(price_data.len());
                let mut close = Vec::with_capacity(price_data.len());
                let mut volume = Vec::with_capacity(price_data.len());

                for (base_price, volatility, vol) in &price_data {
                    let range = base_price * volatility;
                    let h = base_price + range * 0.5;
                    let l = base_price - range * 0.5;
                    let c = l + (h - l) * 0.6; // Close slightly above middle

                    high.push(h);
                    low.push(l);
                    close.push(c);
                    volume.push(*vol);
                }

                // Create input
                let params = KvoParams {
                    short_period: Some(short_period),
                    long_period: Some(long_period),
                };
                let input = KvoInput::from_slices(&high, &low, &close, &volume, params.clone());

                // Calculate with specified kernel and scalar reference
                let result = kvo_with_kernel(&input, kernel)?;
                let reference = kvo_with_kernel(&input, Kernel::Scalar)?;

                // Property 1: Kernel consistency - results should match between kernels
                for i in 0..result.values.len() {
                    let val = result.values[i];
                    let ref_val = reference.values[i];

                    if val.is_nan() && ref_val.is_nan() {
                        continue;
                    }

                    if val.is_finite() && ref_val.is_finite() {
                        let ulp_diff = val.to_bits().abs_diff(ref_val.to_bits());
                        prop_assert!(
                            (val - ref_val).abs() <= 1e-9 || ulp_diff <= 8,
                            "[{}] Kernel mismatch at idx {}: {} vs {} (ULP={})",
                            test_name,
                            i,
                            val,
                            ref_val,
                            ulp_diff
                        );
                    } else {
                        prop_assert_eq!(
                            val.is_finite(),
                            ref_val.is_finite(),
                            "[{}] Finite mismatch at idx {}: {} vs {}",
                            test_name,
                            i,
                            val,
                            ref_val
                        );
                    }
                }

                // Find first valid index (all inputs are non-NaN)
                let first_valid_idx = high
                    .iter()
                    .zip(low.iter())
                    .zip(close.iter())
                    .zip(volume.iter())
                    .position(|(((h, l), c), v)| {
                        !h.is_nan() && !l.is_nan() && !c.is_nan() && !v.is_nan()
                    })
                    .unwrap_or(0);

                // Property 2: Warmup period - NaN values only before first_valid_idx + 1
                // KVO needs at least 2 data points to start calculating (outputs start at first_valid_idx + 1)
                for i in 0..result.values.len() {
                    if i < first_valid_idx + 1 {
                        prop_assert!(
                            result.values[i].is_nan(),
                            "[{}] Expected NaN during warmup at idx {}, got {}",
                            test_name,
                            i,
                            result.values[i]
                        );
                    }
                }

                // Property 3: After warmup, all values should be finite (no NaN)
                if result.values.len() > first_valid_idx + 1 {
                    for i in (first_valid_idx + 1)..result.values.len() {
                        prop_assert!(
                            result.values[i].is_finite(),
                            "[{}] Expected finite value after warmup at idx {}, got {}",
                            test_name,
                            i,
                            result.values[i]
                        );
                    }
                }

                // Property 4: For constant prices and volumes, oscillator should converge to near zero
                let all_same_price = high.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10)
                    && low.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10)
                    && close.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10);
                let all_same_volume = volume.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10);

                if all_same_price && all_same_volume && result.values.len() > 100 {
                    // After sufficient data points, oscillator should be near zero
                    let last_values = &result.values[result.values.len() - 10..];
                    for val in last_values {
                        if val.is_finite() {
                            prop_assert!(
                                val.abs() < 0.1,
                                "[{}] Constant data should produce near-zero oscillator, got {}",
                                test_name,
                                val
                            );
                        }
                    }
                }

                // Property 5: Short EMA period effect - shorter period = more responsive
                // When short_period is very small (1-2), oscillator should be more volatile
                if short_period <= 2 && result.values.len() > first_valid_idx + 20 {
                    let valid_values: Vec<f64> = result.values[(first_valid_idx + 2)..]
                        .iter()
                        .filter(|v| v.is_finite())
                        .copied()
                        .collect();

                    if valid_values.len() > 10 {
                        // Calculate standard deviation of changes
                        let changes: Vec<f64> = valid_values
                            .windows(2)
                            .map(|w| (w[1] - w[0]).abs())
                            .collect();

                        let avg_change = changes.iter().sum::<f64>() / changes.len() as f64;

                        // With very short period, expect some movement (not stuck at zero)
                        if !all_same_price {
                            prop_assert!(
                                avg_change > 1e-12,
                                "[{}] Short period {} should produce some oscillator movement",
                                test_name,
                                short_period
                            );
                        }
                    }
                }

                // Property 6: Trend detection - verify clear trends affect oscillator direction
                // Check if we have a clear uptrend or downtrend in the data
                if result.values.len() > first_valid_idx + 20 {
                    // Calculate HLC trend over the last 20 points (if available)
                    let trend_start = result.values.len().saturating_sub(20);
                    let trend_end = result.values.len();

                    if trend_start > first_valid_idx + 1 {
                        let mut hlc_values = Vec::new();
                        for i in trend_start..trend_end {
                            hlc_values.push(high[i] + low[i] + close[i]);
                        }

                        // Check if we have a clear trend (most values increasing or decreasing)
                        let mut up_moves = 0;
                        let mut down_moves = 0;
                        for window in hlc_values.windows(2) {
                            if window[1] > window[0] * 1.001 {
                                // 0.1% threshold
                                up_moves += 1;
                            } else if window[1] < window[0] * 0.999 {
                                down_moves += 1;
                            }
                        }

                        // If we have a strong trend with good volume, check oscillator direction
                        let avg_volume = volume[trend_start..trend_end].iter().sum::<f64>() / 20.0;
                        if avg_volume > 1000.0 {
                            // Get the last few oscillator values
                            let last_oscillator_values =
                                &result.values[trend_end.saturating_sub(5)..trend_end];
                            let avg_oscillator = last_oscillator_values
                                .iter()
                                .filter(|v| v.is_finite())
                                .sum::<f64>()
                                / last_oscillator_values.len() as f64;

                            // Strong uptrend should produce positive oscillator (eventually)
                            if up_moves > 15 && down_moves < 3 {
                                prop_assert!(
									avg_oscillator > -100.0,  // Very loose - just shouldn't be strongly negative
									"[{}] Strong uptrend should not produce strongly negative oscillator: {}",
									test_name, avg_oscillator
								);
                            }
                            // Strong downtrend should produce negative oscillator (eventually)
                            else if down_moves > 15 && up_moves < 3 {
                                prop_assert!(
									avg_oscillator < 100.0,  // Very loose - just shouldn't be strongly positive
									"[{}] Strong downtrend should not produce strongly positive oscillator: {}",
									test_name, avg_oscillator
								);
                            }
                        }
                    }
                }

                // Property 7: Parameter relationship - long_period >= short_period
                prop_assert!(
                    long_period >= short_period,
                    "[{}] Long period {} should be >= short period {}",
                    test_name,
                    long_period,
                    short_period
                );

                // Property 8: Volume impact - zero or very small volume should produce smaller oscillator values
                // Create a test case with very small volume
                let mut small_vol = volume.clone();
                for v in &mut small_vol {
                    *v *= 1e-10; // Make volume extremely small
                }

                let small_vol_input =
                    KvoInput::from_slices(&high, &low, &close, &small_vol, params.clone());
                if let Ok(small_vol_result) = kvo_with_kernel(&small_vol_input, kernel) {
                    // Compare magnitudes - small volume should produce smaller oscillator values
                    for i in (first_valid_idx + 1)..result.values.len() {
                        if result.values[i].is_finite() && small_vol_result.values[i].is_finite() {
                            // Small volume oscillator should be much smaller in magnitude
                            prop_assert!(
								small_vol_result.values[i].abs() <= result.values[i].abs() * 1e-8 + 1e-10,
								"[{}] Small volume should produce smaller oscillator at idx {}: {} vs {}",
								test_name, i, small_vol_result.values[i], result.values[i]
							);
                        }
                    }
                }

                // Property 9: Volume Force calculation bounds
                // The volume force formula includes (dm/cm * 2.0 - 1.0).abs() which should be bounded
                // When cm > 0, this expression should be in range [0, 1] after abs()
                if result.values.len() > first_valid_idx + 10 {
                    // Manually check a few volume force calculations
                    let mut cm = 0.0;
                    let mut trend = -1;
                    let mut prev_hlc =
                        high[first_valid_idx] + low[first_valid_idx] + close[first_valid_idx];

                    for i in (first_valid_idx + 1)..(first_valid_idx + 10).min(high.len()) {
                        let hlc = high[i] + low[i] + close[i];
                        let dm = high[i] - low[i];

                        // Update trend and cm as in the actual implementation
                        if hlc > prev_hlc && trend != 1 {
                            trend = 1;
                            cm = high[i - 1] - low[i - 1];
                        } else if hlc < prev_hlc && trend != 0 {
                            trend = 0;
                            cm = high[i - 1] - low[i - 1];
                        }
                        cm += dm;

                        // Check the volume force component bounds
                        if cm > 1e-10 {
                            // Avoid division by very small numbers
                            let vf_component = (dm / cm * 2.0 - 1.0).abs();
                            prop_assert!(
								vf_component <= 1.0 + 1e-9,  // Allow small numerical error
								"[{}] Volume force component out of bounds at idx {}: {} (dm={}, cm={})",
								test_name, i, vf_component, dm, cm
							);
                        }

                        prev_hlc = hlc;
                    }
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    macro_rules! generate_all_kvo_tests {
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

    generate_all_kvo_tests!(
        check_kvo_partial_params,
        check_kvo_accuracy,
        check_kvo_default_candles,
        check_kvo_zero_period,
        check_kvo_period_invalid,
        check_kvo_very_small_dataset,
        check_kvo_reinput,
        check_kvo_nan_handling,
        check_kvo_streaming,
        check_kvo_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_kvo_tests!(check_kvo_property);

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = KvoBatchBuilder::new().kernel(kernel).apply_candles(&c)?;
        let def = KvoParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [
            -246.42698280402647,
            530.8651474164992,
            237.2148311016648,
            608.8044103976362,
            -6339.615516805162,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-1,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test various parameter sweep configurations
        let test_configs = vec![
            // (short_start, short_end, short_step, long_start, long_end, long_step)
            (1, 5, 1, 2, 10, 2),    // Small periods
            (2, 10, 2, 5, 20, 5),   // Medium periods
            (5, 25, 5, 10, 50, 10), // Large periods
            (1, 3, 1, 1, 5, 1),     // Dense small range
            (10, 20, 2, 20, 40, 4), // Mid-range dense
            (2, 2, 0, 5, 50, 5),    // Static short, varying long
            (1, 10, 1, 20, 20, 0),  // Varying short, static long
        ];

        for (cfg_idx, &(short_start, short_end, short_step, long_start, long_end, long_step)) in
            test_configs.iter().enumerate()
        {
            let output = KvoBatchBuilder::new()
                .kernel(kernel)
                .short_range(short_start, short_end, short_step)
                .long_range(long_start, long_end, long_step)
                .apply_candles(&c)?;

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
						 at row {} col {} (flat index {}) with params: short_period={}, long_period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.short_period.unwrap_or(2),
                        combo.long_period.unwrap_or(5)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: short_period={}, long_period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.short_period.unwrap_or(2),
                        combo.long_period.unwrap_or(5)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: short_period={}, long_period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.short_period.unwrap_or(2),
                        combo.long_period.unwrap_or(5)
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
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
}

#[cfg(feature = "python")]
#[pyfunction(name = "kvo")]
#[pyo3(signature = (high, low, close, volume, short_period=None, long_period=None, kernel=None))]
pub fn kvo_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    short_period: Option<usize>,
    long_period: Option<usize>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let volume_slice = volume.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = KvoParams {
        short_period,
        long_period,
    };
    let input = KvoInput::from_slices(high_slice, low_slice, close_slice, volume_slice, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| kvo_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "KvoStream")]
pub struct KvoStreamPy {
    stream: KvoStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl KvoStreamPy {
    #[new]
    fn new(short_period: Option<usize>, long_period: Option<usize>) -> PyResult<Self> {
        let params = KvoParams {
            short_period,
            long_period,
        };
        let stream =
            KvoStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(KvoStreamPy { stream })
    }

    fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> Option<f64> {
        self.stream.update(high, low, close, volume)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "kvo_batch")]
#[pyo3(signature = (high, low, close, volume, short_range, long_range, kernel=None))]
pub fn kvo_batch_py<'py>(
    py: Python<'py>,
    high: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    close: numpy::PyReadonlyArray1<'py, f64>,
    volume: numpy::PyReadonlyArray1<'py, f64>,
    short_range: (usize, usize, usize),
    long_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};

    let h = high.as_slice()?;
    let l = low.as_slice()?;
    let c = close.as_slice()?;
    let v = volume.as_slice()?;

    // grid + shape
    let sweep = KvoBatchRange {
        short_period: short_range,
        long_period: long_range,
    };
    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = h.len();

    // pre-allocate the exact target buffer in NumPy
    let arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let out_flat: &mut [f64] = unsafe { arr.as_slice_mut()? };

    let kern = validate_kernel(kernel, true)?; // expect batch-ish
    let simd = match kern {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    py.allow_threads(|| kvo_batch_inner_into(h, l, c, v, &sweep, simd, true, out_flat))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", arr.reshape((rows, cols))?)?;
    dict.set_item(
        "shorts",
        combos
            .iter()
            .map(|p| p.short_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "longs",
        combos
            .iter()
            .map(|p| p.long_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    Ok(dict)
}

// ----------------------------- PYTHON CUDA BINDINGS -----------------------------
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "kvo_cuda_batch_dev")]
#[pyo3(signature = (high_f32, low_f32, close_f32, volume_f32, short_range, long_range, device_id=0))]
pub fn kvo_cuda_batch_dev_py<'py>(
    py: Python<'py>,
    high_f32: numpy::PyReadonlyArray1<'py, f32>,
    low_f32: numpy::PyReadonlyArray1<'py, f32>,
    close_f32: numpy::PyReadonlyArray1<'py, f32>,
    volume_f32: numpy::PyReadonlyArray1<'py, f32>,
    short_range: (usize, usize, usize),
    long_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<(DeviceArrayF32Py, Bound<'py, PyDict>)> {
    use crate::cuda::cuda_available;
    use crate::cuda::CudaKvo;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let h = high_f32.as_slice()?;
    let l = low_f32.as_slice()?;
    let c = close_f32.as_slice()?;
    let v = volume_f32.as_slice()?;
    if h.len() != l.len() || h.len() != c.len() || h.len() != v.len() {
        return Err(PyValueError::new_err("inputs must have equal length"));
    }

    let sweep = KvoBatchRange {
        short_period: short_range,
        long_period: long_range,
    };
    let (inner, combos) = py.allow_threads(|| {
        let cuda = CudaKvo::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.kvo_batch_dev(h, l, c, v, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    let dict = PyDict::new(py);
    dict.set_item(
        "shorts",
        combos
            .iter()
            .map(|p| p.short_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "longs",
        combos
            .iter()
            .map(|p| p.long_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    Ok((DeviceArrayF32Py { inner }, dict))
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "kvo_cuda_many_series_one_param_dev")]
#[pyo3(signature = (high_tm_f32, low_tm_f32, close_tm_f32, volume_tm_f32, cols, rows, short_period, long_period, device_id=0))]
pub fn kvo_cuda_many_series_one_param_dev_py<'py>(
    py: Python<'py>,
    high_tm_f32: numpy::PyReadonlyArray1<'py, f32>,
    low_tm_f32: numpy::PyReadonlyArray1<'py, f32>,
    close_tm_f32: numpy::PyReadonlyArray1<'py, f32>,
    volume_tm_f32: numpy::PyReadonlyArray1<'py, f32>,
    cols: usize,
    rows: usize,
    short_period: usize,
    long_period: usize,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    use crate::cuda::CudaKvo;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let h = high_tm_f32.as_slice()?;
    let l = low_tm_f32.as_slice()?;
    let c = close_tm_f32.as_slice()?;
    let v = volume_tm_f32.as_slice()?;
    if h.len() != l.len() || h.len() != c.len() || h.len() != v.len() {
        return Err(PyValueError::new_err("inputs must have equal length"));
    }
    if cols * rows != h.len() {
        return Err(PyValueError::new_err("cols*rows must equal data length"));
    }

    let params = KvoParams {
        short_period: Some(short_period),
        long_period: Some(long_period),
    };
    let inner = py.allow_threads(|| {
        let cuda = CudaKvo::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.kvo_many_series_one_param_time_major_dev(h, l, c, v, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(DeviceArrayF32Py { inner })
}

// Helper function for batch processing that writes directly into the provided slice
#[inline(always)]
fn kvo_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &KvoBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<KvoParams>, KvoError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(KvoError::InvalidPeriod { short: 0, long: 0 });
    }

    // First valid index across all inputs
    let first = high
        .iter()
        .zip(low)
        .zip(close)
        .zip(volume)
        .position(|(((h, l), c), v)| !h.is_nan() && !l.is_nan() && !c.is_nan() && !v.is_nan())
        .ok_or(KvoError::AllValuesNaN)?;

    if high.len() - first < 2 {
        return Err(KvoError::NotEnoughValidData {
            valid: high.len() - first,
        });
    }

    let rows = combos.len();
    let cols = high.len();
    assert_eq!(out.len(), rows * cols, "out buffer has wrong length");

    // Initialize warm prefixes as NaN per row without extra copies
    let out_mu: &mut [MaybeUninit<f64>] = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };
    let warm: Vec<usize> = combos.iter().map(|_| first + 1).collect();
    init_matrix_prefixes(out_mu, cols, &warm);

    // Choose concrete kernel (scalar/avx/avx512) for rows
    let actual = match kern {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    // Precompute VF once for all rows (shared across parameter sets)
    // Warmup semantics: indices 0..=first are considered warmup; VF is only consumed from first+1.
    #[inline(always)]
    fn precompute_vf(
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
        first: usize,
    ) -> Vec<f64> {
        let len = high.len();
        let mut vf = vec![f64::NAN; len];
        if len <= first + 1 {
            return vf;
        }
        // Use raw pointers to keep this pass tight
        unsafe {
            let hp = high.as_ptr();
            let lp = low.as_ptr();
            let cp = close.as_ptr();
            let vp = volume.as_ptr();

            let mut trend: i32 = -1;
            let mut cm: f64 = 0.0;
            let mut prev_hlc = *hp.add(first) + *lp.add(first) + *cp.add(first);
            let mut prev_dm = *hp.add(first) - *lp.add(first);

            let mut i = first + 1;
            while i < len {
                let h = *hp.add(i);
                let l = *lp.add(i);
                let c = *cp.add(i);
                let v = *vp.add(i);

                let hlc = h + l + c;
                let dm = h - l;

                if hlc > prev_hlc && trend != 1 {
                    trend = 1;
                    cm = prev_dm;
                } else if hlc < prev_hlc && trend != 0 {
                    trend = 0;
                    cm = prev_dm;
                }
                cm += dm;

                let temp = ((dm / cm) * 2.0 - 1.0).abs();
                let sign = if trend == 1 { 1.0 } else { -1.0 };
                vf[i] = v * temp * 100.0 * sign;

                prev_hlc = hlc;
                prev_dm = dm;
                i += 1;
            }
        }
        vf
    }

    let vf = precompute_vf(high, low, close, volume, first);

    // Row writer that consumes shared VF and performs only EMA work per parameter set
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| {
        let s = combos[row].short_period.unwrap();
        let l = combos[row].long_period.unwrap();

        let short_alpha = 2.0 / (s as f64 + 1.0);
        let long_alpha = 2.0 / (l as f64 + 1.0);

        // Cast row to &mut [f64] once; we only write into it
        let dst = unsafe {
            std::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len())
        };

        // EMA over shared VF stream
        let mut short_ema = 0.0f64;
        let mut long_ema = 0.0f64;
        // Seed at first+1 to match single-series semantics
        if first + 1 < cols {
            let seed = vf[first + 1];
            short_ema = seed;
            long_ema = seed;
            dst[first + 1] = 0.0; // seed difference = 0.0
            for i in (first + 2)..cols {
                let vfi = vf[i];
                short_ema += (vfi - short_ema) * short_alpha;
                long_ema += (vfi - long_ema) * long_alpha;
                dst[i] = short_ema - long_ema;
            }
        }

        // Respect kernel selection for API parity (Avx paths currently stubbed)
        let _ = actual; // keep variable used for readability; selection handled above if needed
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
        {
            for (r, row) in out_mu.chunks_mut(cols).enumerate() {
                do_row(r, row);
            }
        }
    } else {
        for (r, row) in out_mu.chunks_mut(cols).enumerate() {
            do_row(r, row);
        }
    }

    Ok(combos)
}

// =============================================================================
// WASM BINDINGS
// =============================================================================

/// Helper function for WASM that writes directly to output slice - no allocations
#[cfg(feature = "wasm")]
#[inline]
fn kvo_wasm_into_slice(
    dst: &mut [f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    short_period: usize,
    long_period: usize,
    kern: Kernel,
) -> Result<(), KvoError> {
    // Validate output length
    if dst.len() != high.len()
        || dst.len() != low.len()
        || dst.len() != close.len()
        || dst.len() != volume.len()
    {
        return Err(KvoError::InvalidPeriod { short: 0, long: 0 });
    }

    // Create input
    let params = KvoParams {
        short_period: Some(short_period),
        long_period: Some(long_period),
    };
    let input = KvoInput {
        data: KvoData::Slices {
            high,
            low,
            close,
            volume,
        },
        params,
    };

    // Use existing kvo_compute_into which already handles NaN prefix
    kvo_compute_into(dst, &input, kern)
}

/// Safe WASM API for KVO calculation
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn kvo_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    short_period: usize,
    long_period: usize,
) -> Result<Vec<f64>, JsValue> {
    let mut output = vec![0.0; high.len()]; // Single allocation

    kvo_wasm_into_slice(
        &mut output,
        high,
        low,
        close,
        volume,
        short_period,
        long_period,
        detect_best_kernel(),
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

/// Fast WASM API for KVO with aliasing detection
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn kvo_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    volume_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    short_period: usize,
    long_period: usize,
) -> Result<(), JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || volume_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);

        // Check for aliasing - if any input pointer equals output pointer
        if high_ptr == out_ptr
            || low_ptr == out_ptr
            || close_ptr == out_ptr
            || volume_ptr == out_ptr
        {
            // Need temporary buffer
            let mut temp = vec![0.0; len];
            kvo_wasm_into_slice(
                &mut temp,
                high,
                low,
                close,
                volume,
                short_period,
                long_period,
                detect_best_kernel(),
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            // No aliasing, can write directly
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            kvo_wasm_into_slice(
                out,
                high,
                low,
                close,
                volume,
                short_period,
                long_period,
                detect_best_kernel(),
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

/// Allocate memory for WASM operations
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn kvo_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

/// Free memory allocated by kvo_alloc
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn kvo_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

/// Batch configuration for WASM
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct KvoBatchConfig {
    pub short_period_range: (usize, usize, usize), // (start, end, step)
    pub long_period_range: (usize, usize, usize),  // (start, end, step)
}

/// Batch output for WASM
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct KvoBatchJsOutput {
    pub values: Vec<f64>, // Flattened matrix
    pub combos: Vec<KvoParams>,
    pub rows: usize,
    pub cols: usize,
}

/// Safe batch API for KVO
#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = kvo_batch)]
pub fn kvo_batch_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: KvoBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = KvoBatchRange {
        short_period: config.short_period_range,
        long_period: config.long_period_range,
    };

    let output = kvo_batch_inner(
        high,
        low,
        close,
        volume,
        &sweep,
        detect_best_kernel(),
        false,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = KvoBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };

    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

/// Fast batch API for KVO with raw pointers
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn kvo_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    volume_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    short_period_start: usize,
    short_period_end: usize,
    short_period_step: usize,
    long_period_start: usize,
    long_period_end: usize,
    long_period_step: usize,
) -> Result<usize, JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || volume_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    if short_period_start == 0 || long_period_start == 0 {
        return Err(JsValue::from_str("Period cannot be zero"));
    }

    if short_period_step == 0 || long_period_step == 0 {
        return Err(JsValue::from_str("Step cannot be zero"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);

        let sweep = KvoBatchRange {
            short_period: (short_period_start, short_period_end, short_period_step),
            long_period: (long_period_start, long_period_end, long_period_step),
        };

        // Check for aliasing
        let aliased = high_ptr == out_ptr
            || low_ptr == out_ptr
            || close_ptr == out_ptr
            || volume_ptr == out_ptr;

        if aliased {
            // Need temporary buffer for batch results
            let output = kvo_batch_inner(
                high,
                low,
                close,
                volume,
                &sweep,
                detect_best_kernel(),
                false,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

            let total_size = output.values.len();
            let out = std::slice::from_raw_parts_mut(out_ptr, total_size);
            out.copy_from_slice(&output.values);

            Ok(output.rows)
        } else {
            // Calculate number of rows
            let combos = expand_grid(&sweep);
            let rows = combos.len();
            let total_size = rows * len;

            // Get output slice
            let out = std::slice::from_raw_parts_mut(out_ptr, total_size);

            // Use direct write helper that avoids allocation
            kvo_batch_inner_into(
                high,
                low,
                close,
                volume,
                &sweep,
                detect_best_kernel(),
                false,
                out,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

            Ok(rows)
        }
    }
}
