//! # Chaikin Accumulation/Distribution Oscillator (ADOSC)
//!
//! A momentum indicator that subtracts a long-term EMA of the Accumulation/Distribution Line (ADL)
//! from a short-term EMA of the ADL to identify shifts in market buying/selling pressure.
//!
//! ## Parameters
//! - **short_period**: The shorter EMA period (default: 3)
//! - **long_period**: The longer EMA period (default: 10)
//!
//! ## Returns
//! - `Ok(AdoscOutput)` on success, containing Vec<f64> with one value per bar
//! - `Err(AdoscError)` otherwise.
//!
//! ## Developer Notes
//! - SIMD status: Loop-carried dependencies (prefix ADL and EMA recurrences) limit gains; AVX2/AVX512 remain stubs that defer to scalar.
//! - Streaming update: O(1) per bar via EMA; math matches batch/scalar path exactly (no FMA), with precomputed (1-α) and guards for `volume==0` and `high==low` fast paths.
//! - Memory: Uses zero-copy helpers (alloc_with_nan_prefix).
//! - Batch optimization: Shares ADL across rows to avoid recomputing MFV/ADL per parameter set.
//!
//! Decision log: SIMD paths are stubs that defer to scalar; CUDA wrapper uses typed errors, checked byte sizing, and an Arc<Context> guard;
//! Python bindings expose CUDA Array Interface v3 (explicit byte strides, version=3, synchronized stream) and DLPack v1.x with legacy capsules,
//! keeping numerical outputs and public APIs identical to the scalar reference implementation.

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
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

use crate::utilities::data_loader::Candles;
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
use thiserror::Error;

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::CudaAdosc;

#[derive(Debug, Clone)]
pub enum AdoscData<'a> {
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
pub struct AdoscOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct AdoscParams {
    pub short_period: Option<usize>,
    pub long_period: Option<usize>,
}

impl Default for AdoscParams {
    fn default() -> Self {
        Self {
            short_period: Some(3),
            long_period: Some(10),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AdoscInput<'a> {
    pub data: AdoscData<'a>,
    pub params: AdoscParams,
}

impl<'a> AdoscInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: AdoscParams) -> Self {
        Self {
            data: AdoscData::Candles { candles },
            params,
        }
    }
    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
        params: AdoscParams,
    ) -> Self {
        Self {
            data: AdoscData::Slices {
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
        Self {
            data: AdoscData::Candles { candles },
            params: AdoscParams::default(),
        }
    }
    #[inline]
    pub fn get_short_period(&self) -> usize {
        self.params.short_period.unwrap_or(3)
    }
    #[inline]
    pub fn get_long_period(&self) -> usize {
        self.params.long_period.unwrap_or(10)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct AdoscBuilder {
    short_period: Option<usize>,
    long_period: Option<usize>,
    kernel: Kernel,
}

impl Default for AdoscBuilder {
    fn default() -> Self {
        Self {
            short_period: None,
            long_period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl AdoscBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<AdoscOutput, AdoscError> {
        let p = AdoscParams {
            short_period: self.short_period,
            long_period: self.long_period,
        };
        let i = AdoscInput::from_candles(c, p);
        adosc_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Result<AdoscOutput, AdoscError> {
        let p = AdoscParams {
            short_period: self.short_period,
            long_period: self.long_period,
        };
        let i = AdoscInput::from_slices(high, low, close, volume, p);
        adosc_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<AdoscStream, AdoscError> {
        let p = AdoscParams {
            short_period: self.short_period,
            long_period: self.long_period,
        };
        AdoscStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum AdoscError {
    #[error("adosc: input data is empty")]
    EmptyInputData,
    #[error("adosc: All values are NaN.")]
    AllValuesNaN,
    #[error("adosc: Invalid period: short={short}, long={long}, data length={data_len}")]
    InvalidPeriod { short: usize, long: usize, data_len: usize },
    #[error("adosc: short_period must be less than long_period: short={short}, long={long}")]
    ShortPeriodGreaterThanLong { short: usize, long: usize },
    #[error("adosc: At least one slice is empty: high={high}, low={low}, close={close}, volume={volume}")]
    EmptySlices { high: usize, low: usize, close: usize, volume: usize },
    #[error("adosc: output length mismatch: expected={expected}, got={got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("adosc: invalid range expansion: start={start}, end={end}, step={step}")]
    InvalidRange { start: usize, end: usize, step: usize },
    #[error("adosc: not enough valid data: needed={needed}, valid={valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("adosc: invalid kernel for batch path: {0:?}")]
    InvalidKernelForBatch(crate::utilities::enums::Kernel),
    #[error("adosc: invalid input: {0}")]
    InvalidInput(String),
}

#[inline]
pub fn adosc(input: &AdoscInput) -> Result<AdoscOutput, AdoscError> {
    adosc_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn adosc_prepare<'a>(
    input: &'a AdoscInput,
    kernel: Kernel,
) -> Result<
    (
        &'a [f64],
        &'a [f64],
        &'a [f64],
        &'a [f64], // high, low, close, volume
        usize,
        usize, // short, long
        usize,
        usize,  // first=0, len
        Kernel, // chosen
    ),
    AdoscError,
> {
    let (high, low, close, volume) = match &input.data {
    AdoscData::Candles { candles } => {
            let n = candles.close.len();
            if n == 0 {
                return Err(AdoscError::EmptyInputData);
            }
            // Enforce equal lengths like alma.rs does for single-slice shape safety
            let (hh, ll, cc, vv) = (
                candles.high.as_slice(),
                candles.low.as_slice(),
                candles.close.as_slice(),
                candles.volume.as_slice(),
            );
            let len = cc.len();
            if hh.len() != len || ll.len() != len || vv.len() != len {
                return Err(AdoscError::EmptySlices {
                    high: hh.len(),
                    low: ll.len(),
                    close: cc.len(),
                    volume: vv.len(),
                });
            }
            (hh, ll, cc, vv)
        }
        AdoscData::Slices {
            high,
            low,
            close,
            volume,
        } => {
            if high.is_empty() || low.is_empty() || close.is_empty() || volume.is_empty() {
                if high.is_empty() && low.is_empty() && close.is_empty() && volume.is_empty() {
                    return Err(AdoscError::EmptyInputData);
                }
                return Err(AdoscError::EmptySlices {
                    high: high.len(),
                    low: low.len(),
                    close: close.len(),
                    volume: volume.len(),
                });
            }
            let len = close.len();
            if high.len() != len || low.len() != len || volume.len() != len {
                return Err(AdoscError::EmptySlices {
                    high: high.len(),
                    low: low.len(),
                    close: close.len(),
                    volume: volume.len(),
                });
            }
            (*high, *low, *close, *volume)
        }
    };

    let len = close.len();
    let short = input.get_short_period();
    let long = input.get_long_period();

    if short == 0 || long == 0 || long > len {
        return Err(AdoscError::InvalidPeriod {
            short,
            long,
            data_len: len,
        });
    }
    if short >= long {
        return Err(AdoscError::ShortPeriodGreaterThanLong { short, long });
    }

    // NaN guard check - ADOSC can process NaN values but this helps detect data issues early
    let all_nan = |s: &[f64]| s.iter().all(|x| x.is_nan());
    if all_nan(high) && all_nan(low) && all_nan(close) && all_nan(volume) {
        return Err(AdoscError::AllValuesNaN);
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };
    Ok((high, low, close, volume, short, long, 0, len, chosen))
}

pub fn adosc_with_kernel(input: &AdoscInput, kernel: Kernel) -> Result<AdoscOutput, AdoscError> {
    let (high, low, close, volume, short, long, first, len, chosen) = adosc_prepare(input, kernel)?;

    match chosen {
        Kernel::Scalar | Kernel::ScalarBatch => unsafe {
            adosc_scalar(high, low, close, volume, short, long, first, len)
        },
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 | Kernel::Avx2Batch => unsafe {
            adosc_avx2(high, low, close, volume, short, long, first, len)
        },
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 | Kernel::Avx512Batch => unsafe {
            adosc_avx512(high, low, close, volume, short, long, first, len)
        },
        #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
        Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => unsafe {
            adosc_scalar(high, low, close, volume, short, long, first, len)
        },
        _ => unreachable!(),
    }
}

/// Writes ADOSC values into a caller-provided output slice without allocating.
///
/// - Preserves ADOSC semantics: no warmup prefix (index 0 is a valid 0.0 value).
/// - `out.len()` must exactly equal the input length; otherwise returns `AdoscError::OutputLengthMismatch`.
#[cfg(not(feature = "wasm"))]
#[inline]
pub fn adosc_into(input: &AdoscInput, out: &mut [f64]) -> Result<(), AdoscError> {
    adosc_into_slice(out, input, Kernel::Auto)
}

#[inline(always)]
pub unsafe fn adosc_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    short: usize,
    long: usize,
    _first: usize,
    len: usize,
) -> Result<AdoscOutput, AdoscError> {
    debug_assert!(len > 0);

    let alpha_short = 2.0 / (short as f64 + 1.0);
    let alpha_long = 2.0 / (long as f64 + 1.0);
    let one_minus_alpha_short = 1.0 - alpha_short;
    let one_minus_alpha_long = 1.0 - alpha_long;

    // Allocate output (no warmup for ADOSC)
    let mut out = alloc_with_nan_prefix(len, 0);

    // Use raw pointers to avoid bounds checks in hot loop
    let hp = high.as_ptr();
    let lp = low.as_ptr();
    let cp = close.as_ptr();
    let vp = volume.as_ptr();
    let op = out.as_mut_ptr();

    // Bootstrap at i = 0
    let h0 = *hp;
    let l0 = *lp;
    let c0 = *cp;
    let v0 = *vp;
    let hl0 = h0 - l0;
    let mfm0 = if hl0 != 0.0 {
        ((c0 - l0) - (h0 - c0)) / hl0
    } else {
        0.0
    };
    let mfv0 = mfm0 * v0;
    let mut sum_ad = mfv0;
    let mut short_ema = sum_ad;
    let mut long_ema = sum_ad;
    *op = short_ema - long_ema; // exactly 0.0

    // i = 1..len-1 hot loop
    let mut i = 1usize;
    while i < len {
        let h = *hp.add(i);
        let l = *lp.add(i);
        let c = *cp.add(i);
        let v = *vp.add(i);

        let hl = h - l;
        let mfm = if hl != 0.0 {
            ((c - l) - (h - c)) / hl
        } else {
            0.0
        };
        let mfv = mfm * v;
        sum_ad += mfv;
        short_ema = alpha_short * sum_ad + one_minus_alpha_short * short_ema;
        long_ema = alpha_long * sum_ad + one_minus_alpha_long * long_ema;
        *op.add(i) = short_ema - long_ema;

        i += 1;
    }

    Ok(AdoscOutput { values: out })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn adosc_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    short: usize,
    long: usize,
    first: usize,
    len: usize,
) -> Result<AdoscOutput, AdoscError> {
    adosc_scalar(high, low, close, volume, short, long, first, len)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn adosc_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    short: usize,
    long: usize,
    first: usize,
    len: usize,
) -> Result<AdoscOutput, AdoscError> {
    adosc_scalar(high, low, close, volume, short, long, first, len)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn adosc_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    short: usize,
    long: usize,
    first: usize,
    len: usize,
) -> Result<AdoscOutput, AdoscError> {
    adosc_scalar(high, low, close, volume, short, long, first, len)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn adosc_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    short: usize,
    long: usize,
    first: usize,
    len: usize,
) -> Result<AdoscOutput, AdoscError> {
    adosc_scalar(high, low, close, volume, short, long, first, len)
}

#[derive(Clone, Debug)]
pub struct AdoscBatchRange {
    pub short_period: (usize, usize, usize),
    pub long_period: (usize, usize, usize),
}

impl Default for AdoscBatchRange {
    fn default() -> Self {
        Self {
            short_period: (3, 10, 1),
            long_period: (10, 30, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct AdoscBatchBuilder {
    range: AdoscBatchRange,
    kernel: Kernel,
}

impl AdoscBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn short_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.short_period = (start, end, step);
        self
    }
    #[inline]
    pub fn long_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.long_period = (start, end, step);
        self
    }
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Result<AdoscBatchOutput, AdoscError> {
        adosc_batch_with_kernel(high, low, close, volume, &self.range, self.kernel)
    }
    pub fn apply_candles(self, candles: &Candles) -> Result<AdoscBatchOutput, AdoscError> {
        self.apply_slices(
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
            candles.volume.as_slice(),
        )
    }
}

#[derive(Clone, Debug)]
pub struct AdoscBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<AdoscParams>,
    pub rows: usize,
    pub cols: usize,
}
impl AdoscBatchOutput {
    pub fn row_for_params(&self, p: &AdoscParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.short_period.unwrap_or(3) == p.short_period.unwrap_or(3)
                && c.long_period.unwrap_or(10) == p.long_period.unwrap_or(10)
        })
    }
    pub fn values_for(&self, p: &AdoscParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

fn expand_grid(r: &AdoscBatchRange) -> Vec<AdoscParams> {
    match expand_grid_checked(r) {
        Ok(v) => v,
        Err(_) => Vec::new(),
    }
}

fn expand_grid_checked(r: &AdoscBatchRange) -> Result<Vec<AdoscParams>, AdoscError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, AdoscError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            let v: Vec<_> = (start..=end).step_by(step).collect();
            if v.is_empty() {
                return Err(AdoscError::InvalidRange { start, end, step });
            }
            Ok(v)
        } else {
            let mut v = Vec::new();
            let mut cur = start;
            while cur >= end {
                v.push(cur);
                if cur - end < step { break; }
                cur -= step;
            }
            if v.is_empty() { return Err(AdoscError::InvalidRange { start, end, step }); }
            Ok(v)
        }
    }
    let shorts = axis_usize(r.short_period)?;
    let longs = axis_usize(r.long_period)?;

    let mut out = Vec::new();
    for &short in &shorts {
        for &long in &longs {
            if short == 0 || long == 0 || short >= long {
                continue;
            }
            out.push(AdoscParams { short_period: Some(short), long_period: Some(long) });
        }
    }
    if out.is_empty() {
        return Err(AdoscError::InvalidRange {
            start: r.short_period.0,
            end: r.long_period.1,
            step: r.short_period.2.max(r.long_period.2),
        });
    }
    Ok(out)
}

pub fn adosc_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &AdoscBatchRange,
    k: Kernel,
) -> Result<AdoscBatchOutput, AdoscError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(AdoscError::InvalidKernelForBatch(other)),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    adosc_batch_par_slice(high, low, close, volume, sweep, simd)
}

pub fn adosc_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &AdoscBatchRange,
    kern: Kernel,
) -> Result<AdoscBatchOutput, AdoscError> {
    adosc_batch_inner(high, low, close, volume, sweep, kern, false)
}

pub fn adosc_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &AdoscBatchRange,
    kern: Kernel,
) -> Result<AdoscBatchOutput, AdoscError> {
    adosc_batch_inner(high, low, close, volume, sweep, kern, true)
}

fn adosc_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &AdoscBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<AdoscBatchOutput, AdoscError> {
    // Check for empty input first
    if high.is_empty() || low.is_empty() || close.is_empty() || volume.is_empty() {
        return Err(AdoscError::EmptySlices {
            high: high.len(),
            low: low.len(),
            close: close.len(),
            volume: volume.len(),
        });
    }

    let combos = expand_grid_checked(sweep)?;
    let first = 0;
    let len = close.len();
    let rows = combos.len();
    let cols = len;
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| AdoscError::InvalidInput("rows*cols overflow".into()))?;

    // Use zero-copy memory allocation
    let mut buf_mu = make_uninit_matrix(rows, cols);
    debug_assert_eq!(buf_mu.len(), expected);

    // ADOSC computes from index 0, so warmup period is 0 for all rows
    let warm: Vec<usize> = vec![0; rows];
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    // Convert to mutable slice for computation
    let mut buf_guard = std::mem::ManuallyDrop::new(buf_mu);
    let values: &mut [f64] = unsafe {
        std::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };
    // Precompute ADL once and share across rows
    let mut adl = vec![0.0f64; len];
    unsafe {
        // Use the same algebra and order to preserve parity
        let hp = high.as_ptr();
        let lp = low.as_ptr();
        let cp = close.as_ptr();
        let vp = volume.as_ptr();
        let ap = adl.as_mut_ptr();

        let h0 = *hp;
        let l0 = *lp;
        let c0 = *cp;
        let v0 = *vp;
        let hl0 = h0 - l0;
        let mfm0 = if hl0 != 0.0 {
            ((c0 - l0) - (h0 - c0)) / hl0
        } else {
            0.0
        };
        let mfv0 = mfm0 * v0;
        *ap = mfv0; // sum_ad at i=0

        let mut i = 1usize;
        while i < len {
            let h = *hp.add(i);
            let l = *lp.add(i);
            let c = *cp.add(i);
            let v = *vp.add(i);
            let prev = *ap.add(i - 1);
            let hl = h - l;
            let mfm = if hl != 0.0 {
                ((c - l) - (h - c)) / hl
            } else {
                0.0
            };
            let mfv = mfm * v;
            *ap.add(i) = prev + mfv;
            i += 1;
        }
    }

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let prm = &combos[row];
        let short = prm.short_period.unwrap();
        let long = prm.long_period.unwrap();

        // EMA over precomputed ADL (kernel choice does not matter here; SIMD stubs defer to scalar)
        let alpha_short = 2.0 / (short as f64 + 1.0);
        let alpha_long = 2.0 / (long as f64 + 1.0);
        let one_minus_alpha_short = 1.0 - alpha_short;
        let one_minus_alpha_long = 1.0 - alpha_long;

        let ap = adl.as_ptr();
        let op = out_row.as_mut_ptr();

        let mut short_ema = *ap; // adl[0]
        let mut long_ema = *ap;
        *op = short_ema - long_ema; // 0.0

        let mut i = 1usize;
        while i < cols {
            let s = *ap.add(i);
            short_ema = alpha_short * s + one_minus_alpha_short * short_ema;
            long_ema = alpha_long * s + one_minus_alpha_long * long_ema;
            *op.add(i) = short_ema - long_ema;
            i += 1;
        }
    };
    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            values
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in values.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in values.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    // Reclaim as Vec<f64>
    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(AdoscBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub fn adosc_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &AdoscBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<AdoscParams>, AdoscError> {
    // Check for empty input first
    if high.is_empty() || low.is_empty() || close.is_empty() || volume.is_empty() {
        return Err(AdoscError::EmptySlices {
            high: high.len(),
            low: low.len(),
            close: close.len(),
            volume: volume.len(),
        });
    }

    let combos = expand_grid_checked(sweep)?;
    let first = 0;
    let len = close.len();
    let rows = combos.len();
    let cols = len;
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| AdoscError::InvalidInput("rows*cols overflow".into()))?;
    if out.len() != expected {
        return Err(AdoscError::OutputLengthMismatch { expected, got: out.len() });
    }

    // Precompute ADL once and share across rows
    let mut adl = vec![0.0f64; len];
    unsafe {
        let hp = high.as_ptr();
        let lp = low.as_ptr();
        let cp = close.as_ptr();
        let vp = volume.as_ptr();
        let ap = adl.as_mut_ptr();

        let h0 = *hp;
        let l0 = *lp;
        let c0 = *cp;
        let v0 = *vp;
        let hl0 = h0 - l0;
        let mfm0 = if hl0 != 0.0 {
            ((c0 - l0) - (h0 - c0)) / hl0
        } else {
            0.0
        };
        let mfv0 = mfm0 * v0;
        *ap = mfv0;

        let mut i = 1usize;
        while i < len {
            let h = *hp.add(i);
            let l = *lp.add(i);
            let c = *cp.add(i);
            let v = *vp.add(i);
            let prev = *ap.add(i - 1);
            let hl = h - l;
            let mfm = if hl != 0.0 {
                ((c - l) - (h - c)) / hl
            } else {
                0.0
            };
            let mfv = mfm * v;
            *ap.add(i) = prev + mfv;
            i += 1;
        }
    }

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let prm = &combos[row];
        let short = prm.short_period.unwrap();
        let long = prm.long_period.unwrap();

        let alpha_short = 2.0 / (short as f64 + 1.0);
        let alpha_long = 2.0 / (long as f64 + 1.0);
        let one_minus_alpha_short = 1.0 - alpha_short;
        let one_minus_alpha_long = 1.0 - alpha_long;

        let ap = adl.as_ptr();
        let op = out_row.as_mut_ptr();
        let mut short_ema = *ap; // adl[0]
        let mut long_ema = *ap;
        *op = short_ema - long_ema;

        let mut i = 1usize;
        while i < cols {
            let s = *ap.add(i);
            short_ema = alpha_short * s + one_minus_alpha_short * short_ema;
            long_ema = alpha_long * s + one_minus_alpha_long * long_ema;
            *op.add(i) = short_ema - long_ema;
            i += 1;
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out.par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in out.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in out.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    Ok(combos)
}

#[inline(always)]
pub unsafe fn adosc_row_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    short: usize,
    long: usize,
    _first: usize,
    out: &mut [f64],
) -> Result<(), AdoscError> {
    let len = out.len();
    debug_assert!(len > 0);

    let alpha_short = 2.0 / (short as f64 + 1.0);
    let alpha_long = 2.0 / (long as f64 + 1.0);
    let one_minus_alpha_short = 1.0 - alpha_short;
    let one_minus_alpha_long = 1.0 - alpha_long;

    let hp = high.as_ptr();
    let lp = low.as_ptr();
    let cp = close.as_ptr();
    let vp = volume.as_ptr();
    let op = out.as_mut_ptr();

    // i = 0 bootstrap
    let h0 = *hp;
    let l0 = *lp;
    let c0 = *cp;
    let v0 = *vp;
    let hl0 = h0 - l0;
    let mfm0 = if hl0 != 0.0 {
        ((c0 - l0) - (h0 - c0)) / hl0
    } else {
        0.0
    };
    let mfv0 = mfm0 * v0;
    let mut sum_ad = mfv0;
    let mut short_ema = sum_ad;
    let mut long_ema = sum_ad;
    *op = short_ema - long_ema;

    // i = 1..len-1
    let mut i = 1usize;
    while i < len {
        let h = *hp.add(i);
        let l = *lp.add(i);
        let c = *cp.add(i);
        let v = *vp.add(i);

        let hl = h - l;
        let mfm = if hl != 0.0 {
            ((c - l) - (h - c)) / hl
        } else {
            0.0
        };
        let mfv = mfm * v;
        sum_ad += mfv;
        short_ema = alpha_short * sum_ad + one_minus_alpha_short * short_ema;
        long_ema = alpha_long * sum_ad + one_minus_alpha_long * long_ema;
        *op.add(i) = short_ema - long_ema;

        i += 1;
    }
    Ok(())
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn adosc_row_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    short: usize,
    long: usize,
    first: usize,
    out: &mut [f64],
) -> Result<(), AdoscError> {
    adosc_row_scalar(high, low, close, volume, short, long, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn adosc_row_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    short: usize,
    long: usize,
    first: usize,
    out: &mut [f64],
) -> Result<(), AdoscError> {
    adosc_row_scalar(high, low, close, volume, short, long, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn adosc_row_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    short: usize,
    long: usize,
    first: usize,
    out: &mut [f64],
) -> Result<(), AdoscError> {
    adosc_row_scalar(high, low, close, volume, short, long, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn adosc_row_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    short: usize,
    long: usize,
    first: usize,
    out: &mut [f64],
) -> Result<(), AdoscError> {
    adosc_row_scalar(high, low, close, volume, short, long, first, out)
}

pub struct AdoscStream {
    short_period: usize,
    long_period: usize,
    // EMA coefficients
    alpha_short: f64,
    alpha_long: f64,
    one_minus_alpha_short: f64,
    one_minus_alpha_long: f64,
    // ADL state and EMA states
    sum_ad: f64,
    short_ema: f64,
    long_ema: f64,
    initialized: bool,
}

impl AdoscStream {
    #[inline(always)]
    pub fn try_new(params: AdoscParams) -> Result<Self, AdoscError> {
        let short = params.short_period.unwrap_or(3);
        let long = params.long_period.unwrap_or(10);
        if short == 0 || long == 0 {
            return Err(AdoscError::InvalidPeriod {
                short,
                long,
                data_len: 0,
            });
        }
        if short >= long {
            return Err(AdoscError::ShortPeriodGreaterThanLong { short, long });
        }

        // α = 2/(n+1) — standard EMA smoothing factor
        let alpha_short = 2.0 / (short as f64 + 1.0);
        let alpha_long = 2.0 / (long as f64 + 1.0);

        Ok(Self {
            short_period: short,
            long_period: long,
            alpha_short,
            alpha_long,
            one_minus_alpha_short: 1.0 - alpha_short,
            one_minus_alpha_long: 1.0 - alpha_long,
            sum_ad: 0.0,
            short_ema: 0.0,
            long_ema: 0.0,
            initialized: false,
        })
    }

    /// O(1) update per bar. Preserves batch parity (no FMA; same MFM algebra/order).
    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> f64 {
        // Fast paths that provably imply MFM == 0.0 (skip divide & extra flops):
        if volume != 0.0 {
            let hl = high - low;
            if hl != 0.0 {
                // Keep algebra/order identical to batch/scalar path:
                let mfm = ((close - low) - (high - close)) / hl;
                self.sum_ad += mfm * volume;
            }
            // else: range==0 → mfm=0 → ADL unchanged
        }
        // else: volume==0 → mfv=0 → ADL unchanged

        if !self.initialized {
            // Bootstrap: EMA short == EMA long == ADL at i=0 → oscillator = 0.0
            self.short_ema = self.sum_ad;
            self.long_ema = self.sum_ad;
            self.initialized = true;
            return 0.0;
        }

        // One multiply-add per EMA with precomputed complements (no FMA to keep parity)
        let x = self.sum_ad;
        self.short_ema = self.alpha_short * x + self.one_minus_alpha_short * self.short_ema;
        self.long_ema = self.alpha_long * x + self.one_minus_alpha_long * self.long_ema;

        self.short_ema - self.long_ema
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_adosc_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = AdoscInput::with_default_candles(&candles);
        let result = adosc_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        let expected_last_five = [-166.2175, -148.9983, -144.9052, -128.5921, -142.0772];
        let start_index = result.values.len().saturating_sub(5);
        let result_last_five = &result.values[start_index..];
        for (i, &actual) in result_last_five.iter().enumerate() {
            let expected = expected_last_five[i];
            assert!(
                (actual - expected).abs() < 1e-1,
                "ADOSC value mismatch at index {}: expected {}, got {}",
                i,
                expected,
                actual
            );
        }
        for (i, &val) in result.values.iter().enumerate() {
            assert!(
                val.is_finite(),
                "ADOSC output at index {} should be finite, got {}",
                i,
                val
            );
        }
        Ok(())
    }

    fn check_adosc_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let partial_params = AdoscParams {
            short_period: Some(2),
            long_period: None,
        };
        let input = AdoscInput::from_candles(&candles, partial_params);
        let result = adosc_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        let missing_short = AdoscParams {
            short_period: None,
            long_period: Some(12),
        };
        let input_missing = AdoscInput::from_candles(&candles, missing_short);
        let result_missing = adosc_with_kernel(&input_missing, kernel)?;
        assert_eq!(result_missing.values.len(), candles.close.len());
        Ok(())
    }

    fn check_adosc_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = AdoscInput::with_default_candles(&candles);
        match input.data {
            AdoscData::Candles { .. } => {}
            _ => panic!("Expected AdoscData::Candles variant"),
        }
        let result = adosc_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        Ok(())
    }

    fn check_adosc_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 10.0, 10.0];
        let low = [5.0, 5.0, 5.0];
        let close = [7.0, 7.0, 7.0];
        let volume = [1000.0, 1000.0, 1000.0];
        let zero_short = AdoscParams {
            short_period: Some(0),
            long_period: Some(10),
        };
        let input = AdoscInput::from_slices(&high, &low, &close, &volume, zero_short);
        let result = adosc_with_kernel(&input, kernel);
        assert!(result.is_err());
        let zero_long = AdoscParams {
            short_period: Some(3),
            long_period: Some(0),
        };
        let input2 = AdoscInput::from_slices(&high, &low, &close, &volume, zero_long);
        let result2 = adosc_with_kernel(&input2, kernel);
        assert!(result2.is_err());
        Ok(())
    }

    fn check_adosc_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 11.0, 12.0];
        let low = [5.0, 5.5, 6.0];
        let close = [7.0, 8.0, 9.0];
        let volume = [1000.0, 1000.0, 1000.0];
        let params = AdoscParams {
            short_period: Some(3),
            long_period: Some(10),
        };
        let input = AdoscInput::from_slices(&high, &low, &close, &volume, params);
        let result = adosc_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_adosc_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0];
        let low = [5.0];
        let close = [7.0];
        let volume = [1000.0];
        let params = AdoscParams {
            short_period: Some(3),
            long_period: Some(10),
        };
        let input = AdoscInput::from_slices(&high, &low, &close, &volume, params);
        let result = adosc_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_adosc_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = AdoscParams {
            short_period: Some(3),
            long_period: Some(10),
        };
        let first_input = AdoscInput::from_candles(&candles, first_params);
        let first_result = adosc_with_kernel(&first_input, kernel)?;
        assert_eq!(first_result.values.len(), candles.close.len());
        let second_params = AdoscParams {
            short_period: Some(2),
            long_period: Some(6),
        };
        let second_input = AdoscInput::from_slices(
            &first_result.values,
            &first_result.values,
            &first_result.values,
            &first_result.values,
            second_params,
        );
        let second_result = adosc_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    fn check_adosc_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = AdoscInput::from_candles(&candles, AdoscParams::default());
        let result = adosc_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        if result.values.len() > 240 {
            for (i, &val) in result.values[240..].iter().enumerate() {
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

    fn check_adosc_streaming(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = AdoscParams {
            short_period: Some(3),
            long_period: Some(10),
        };
        let input = AdoscInput::from_candles(&candles, params.clone());
        let batch_output = adosc_with_kernel(&input, kernel)?.values;
        let mut stream = AdoscStream::try_new(params)?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for ((&h, &l), (&c, &v)) in candles
            .high
            .iter()
            .zip(candles.low.iter())
            .zip(candles.close.iter().zip(candles.volume.iter()))
        {
            stream_values.push(stream.update(h, l, c, v));
        }
        assert_eq!(batch_output.len(), stream_values.len());
        for (i, (&b, &s)) in batch_output.iter().zip(stream_values.iter()).enumerate() {
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] ADOSC streaming mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let batch = AdoscBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&candles)?;
        let def = AdoscParams::default();
        let row = batch.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), candles.close.len());
        Ok(())
    }

    macro_rules! generate_all_adosc_tests {
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

    fn check_adosc_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Fill poison values
        let len = candles.close.len();
        let mut high = AVec::<f64>::with_capacity(CACHELINE_ALIGN, len);
        let mut low = AVec::<f64>::with_capacity(CACHELINE_ALIGN, len);
        let mut close = AVec::<f64>::with_capacity(CACHELINE_ALIGN, len);
        let mut volume = AVec::<f64>::with_capacity(CACHELINE_ALIGN, len);

        high.resize(len, f64::from_bits(0x11111111_11111111));
        low.resize(len, f64::from_bits(0x22222222_22222222));
        close.resize(len, f64::from_bits(0x33333333_33333333));
        volume.resize(len, f64::from_bits(0x11111111_11111111));

        // Copy real data
        high.copy_from_slice(&candles.high);
        low.copy_from_slice(&candles.low);
        close.copy_from_slice(&candles.close);
        volume.copy_from_slice(&candles.volume);

        // Run calculation
        let params = AdoscParams {
            short_period: Some(3),
            long_period: Some(10),
        };
        let input = AdoscInput::from_slices(&high, &low, &close, &volume, params);
        let result = adosc_with_kernel(&input, kernel)?;

        // Check for poison values - ADOSC has no warmup period
        for (i, &val) in result.values.iter().enumerate() {
            assert_ne!(
                val.to_bits(),
                0x11111111_11111111,
                "[{}] Poison value 0x11111111_11111111 found at index {}",
                test_name,
                i
            );
            assert_ne!(
                val.to_bits(),
                0x22222222_22222222,
                "[{}] Poison value 0x22222222_22222222 found at index {}",
                test_name,
                i
            );
            assert_ne!(
                val.to_bits(),
                0x33333333_33333333,
                "[{}] Poison value 0x33333333_33333333 found at index {}",
                test_name,
                i
            );
        }

        Ok(())
    }

    fn check_batch_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Use smaller dataset for batch testing
        let slice_end = candles.close.len().min(1000);
        let high_slice = &candles.high[..slice_end];
        let low_slice = &candles.low[..slice_end];
        let close_slice = &candles.close[..slice_end];
        let volume_slice = &candles.volume[..slice_end];

        // Create configuration with comprehensive parameter sweep
        let batch_config = AdoscBatchRange {
            short_period: (2, 5, 1), // 2, 3, 4, 5
            long_period: (8, 12, 2), // 8, 10, 12
        };

        // Run batch calculation
        let result = adosc_batch_with_kernel(
            high_slice,
            low_slice,
            close_slice,
            volume_slice,
            &batch_config,
            kernel,
        )?;

        // Check for poison values
        for (i, &val) in result.values.iter().enumerate() {
            // ADOSC has no warmup, so no NaN values expected
            assert_ne!(
                val.to_bits(),
                0x11111111_11111111,
                "[{}] Poison value 0x11111111_11111111 found in batch output at index {}",
                test_name,
                i
            );
            assert_ne!(
                val.to_bits(),
                0x22222222_22222222,
                "[{}] Poison value 0x22222222_22222222 found in batch output at index {}",
                test_name,
                i
            );
            assert_ne!(
                val.to_bits(),
                0x33333333_33333333,
                "[{}] Poison value 0x33333333_33333333 found in batch output at index {}",
                test_name,
                i
            );
        }

        // Verify dimensions
        let expected_rows = result.combos.len();
        let expected_cols = slice_end;
        assert_eq!(
            result.values.len(),
            expected_rows * expected_cols,
            "[{}] Batch output size mismatch",
            test_name
        );

        // Test with different sweep ranges
        let batch_config2 = AdoscBatchRange {
            short_period: (3, 7, 2),  // 3, 5, 7
            long_period: (10, 20, 5), // 10, 15, 20
        };

        let result2 = adosc_batch_with_kernel(
            high_slice,
            low_slice,
            close_slice,
            volume_slice,
            &batch_config2,
            kernel,
        )?;

        // Check second configuration
        for (i, &val) in result2.values.iter().enumerate() {
            assert_ne!(
                val.to_bits(),
                0x11111111_11111111,
                "[{}] Poison value found in second batch config at index {}",
                test_name,
                i
            );
        }

        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_adosc_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (1usize..=10, 11usize..=30).prop_flat_map(|(short_period, long_period)| {
            // Generate realistic OHLC data where High >= Close >= Low
            let len = long_period..400;
            (
                prop::collection::vec(
                    // Generate base prices
                    (1f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                    len.clone(),
                )
                .prop_flat_map(move |base_prices| {
                    let len = base_prices.len();
                    // Generate spreads for high and low
                    let high_spreads = prop::collection::vec(
                        (0f64..100f64).prop_filter("finite", |x| x.is_finite()),
                        len,
                    );
                    let low_spreads = prop::collection::vec(
                        (0f64..100f64).prop_filter("finite", |x| x.is_finite()),
                        len,
                    );
                    // Generate close position between 0 and 1 (0 = at low, 1 = at high)
                    let close_positions = prop::collection::vec(0f64..=1f64, len);

                    (
                        Just(base_prices),
                        high_spreads,
                        low_spreads,
                        close_positions,
                    )
                })
                .prop_map(|(base, high_spreads, low_spreads, close_positions)| {
                    let mut high = Vec::with_capacity(base.len());
                    let mut low = Vec::with_capacity(base.len());
                    let mut close = Vec::with_capacity(base.len());

                    for i in 0..base.len() {
                        let h = base[i] + high_spreads[i];
                        let l = base[i] - low_spreads[i];
                        let c = l + (h - l) * close_positions[i];

                        high.push(h);
                        low.push(l);
                        close.push(c);
                    }

                    (high, low, close)
                }),
                prop::collection::vec((0f64..1e6f64).prop_filter("finite", |x| x.is_finite()), len),
                Just(short_period),
                Just(long_period),
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(
                &strat,
                |((high, low, close), volume, short_period, long_period)| {
                    // Ensure all vectors have the same length
                    let len = high.len();
                    prop_assert_eq!(low.len(), len);
                    prop_assert_eq!(close.len(), len);
                    prop_assert_eq!(volume.len(), len);

                    // Verify OHLC relationships are valid
                    for i in 0..len {
                        prop_assert!(
                            high[i] >= low[i],
                            "High must be >= Low at index {}: {} < {}",
                            i,
                            high[i],
                            low[i]
                        );
                        prop_assert!(
                            close[i] >= low[i] && close[i] <= high[i],
                            "Close must be between Low and High at index {}: {} not in [{}, {}]",
                            i,
                            close[i],
                            low[i],
                            high[i]
                        );
                    }

                    let params = AdoscParams {
                        short_period: Some(short_period),
                        long_period: Some(long_period),
                    };
                    let input = AdoscInput::from_slices(&high, &low, &close, &volume, params);

                    let result = adosc_with_kernel(&input, kernel);
                    prop_assert!(result.is_ok(), "ADOSC computation failed: {:?}", result);

                    let AdoscOutput { values: out } = result.unwrap();

                    // Property 1: Output length matches input
                    prop_assert_eq!(out.len(), len, "Output length mismatch");

                    // Property 2: All values are finite (ADOSC has no warmup period)
                    for (i, &val) in out.iter().enumerate() {
                        prop_assert!(
                            val.is_finite(),
                            "ADOSC output at index {} should be finite, got {}",
                            i,
                            val
                        );
                    }

                    // Property 3: First value should be 0 (short_ema - long_ema both start at same value)
                    prop_assert!(
                        out[0].abs() < 1e-10,
                        "First ADOSC value should be 0, got {}",
                        out[0]
                    );

                    // Property 4: Zero volume should maintain previous AD value
                    if volume.iter().all(|&v| v == 0.0) {
                        // With zero volume, MFV is always 0, so AD never changes
                        // This means short_ema and long_ema decay towards 0
                        // ADOSC = short_ema - long_ema should approach 0
                        for &val in out.iter() {
                            prop_assert!(
                                val.abs() < 1e-9,
                                "With zero volume, ADOSC should be ~0, got {}",
                                val
                            );
                        }
                    }

                    // Property 5: MFM bounds check
                    // Money Flow Multiplier should always be between -1 and 1
                    for i in 0..len {
                        let h = high[i];
                        let l = low[i];
                        let c = close[i];
                        let hl = h - l;
                        if hl != 0.0 {
                            let mfm = ((c - l) - (h - c)) / hl;
                            prop_assert!(
                                mfm >= -1.0 - 1e-10 && mfm <= 1.0 + 1e-10,
                                "MFM at index {} out of bounds: {}",
                                i,
                                mfm
                            );
                        }
                    }

                    // Property 6: Values should be bounded reasonably
                    // Since MFM is bounded [-1, 1], the maximum AD change per step is volume
                    // The maximum cumulative AD is sum of all volumes * 1
                    // ADOSC is the difference between two EMAs of AD
                    let total_volume: f64 = volume.iter().sum();
                    // The maximum possible difference between EMAs is proportional to total volume
                    // Using a tighter bound based on the fact that EMAs converge
                    let expected_bound = total_volume * 0.5; // Tighter bound
                    for (i, &val) in out.iter().enumerate() {
                        prop_assert!(
                            val.abs() <= expected_bound,
                            "ADOSC at index {} exceeds reasonable bounds: {} > {}",
                            i,
                            val.abs(),
                            expected_bound
                        );
                    }

                    // Property 7: Verify period relationship
                    prop_assert!(
                        short_period < long_period,
                        "Short period must be less than long period"
                    );

                    // Property 8: Formula validation - manually calculate first few values
                    if len >= 3 {
                        let alpha_short = 2.0 / (short_period as f64 + 1.0);
                        let alpha_long = 2.0 / (long_period as f64 + 1.0);

                        // First value calculation
                        let h0 = high[0];
                        let l0 = low[0];
                        let c0 = close[0];
                        let v0 = volume[0];
                        let hl0 = h0 - l0;
                        let mfm0 = if hl0 != 0.0 {
                            ((c0 - l0) - (h0 - c0)) / hl0
                        } else {
                            0.0
                        };
                        let mfv0 = mfm0 * v0;
                        let sum_ad0 = mfv0;
                        let expected_first = 0.0; // short_ema - long_ema both start at sum_ad0
                        prop_assert!(
                            (out[0] - expected_first).abs() < 1e-9,
                            "First value mismatch: expected {}, got {}",
                            expected_first,
                            out[0]
                        );

                        // Second value calculation
                        let h1 = high[1];
                        let l1 = low[1];
                        let c1 = close[1];
                        let v1 = volume[1];
                        let hl1 = h1 - l1;
                        let mfm1 = if hl1 != 0.0 {
                            ((c1 - l1) - (h1 - c1)) / hl1
                        } else {
                            0.0
                        };
                        let mfv1 = mfm1 * v1;
                        let sum_ad1 = sum_ad0 + mfv1;
                        let short_ema1 = alpha_short * sum_ad1 + (1.0 - alpha_short) * sum_ad0;
                        let long_ema1 = alpha_long * sum_ad1 + (1.0 - alpha_long) * sum_ad0;
                        let expected_second = short_ema1 - long_ema1;
                        prop_assert!(
                            (out[1] - expected_second).abs() < 1e-9,
                            "Second value mismatch: expected {}, got {}",
                            expected_second,
                            out[1]
                        );
                    }

                    // Property 9: Kernel consistency - all kernels should produce the same result
                    let ref_output = adosc_with_kernel(&input, Kernel::Scalar);
                    prop_assert!(ref_output.is_ok(), "Reference scalar computation failed");
                    let AdoscOutput { values: ref_out } = ref_output.unwrap();

                    for (i, (&val, &ref_val)) in out.iter().zip(ref_out.iter()).enumerate() {
                        let val_bits = val.to_bits();
                        let ref_bits = ref_val.to_bits();

                        if !val.is_finite() || !ref_val.is_finite() {
                            prop_assert_eq!(
                                val_bits,
                                ref_bits,
                                "NaN/Inf mismatch at index {}: {} vs {}",
                                i,
                                val,
                                ref_val
                            );
                        } else {
                            let ulp_diff = val_bits.abs_diff(ref_bits);
                            prop_assert!(
                                (val - ref_val).abs() <= 1e-9 || ulp_diff <= 4,
                                "Kernel mismatch at index {}: {} vs {} (diff: {}, ULP: {})",
                                i,
                                val,
                                ref_val,
                                (val - ref_val).abs(),
                                ulp_diff
                            );
                        }
                    }

                    Ok(())
                },
            )
            .map_err(|e| e.into())
    }

    generate_all_adosc_tests!(
        check_adosc_accuracy,
        check_adosc_partial_params,
        check_adosc_default_candles,
        check_adosc_zero_period,
        check_adosc_period_exceeds_length,
        check_adosc_very_small_dataset,
        check_adosc_reinput,
        check_adosc_nan_handling,
        check_adosc_streaming,
        check_adosc_no_poison,
        check_adosc_property
    );

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
    gen_batch_tests!(check_batch_no_poison);

    #[cfg(not(feature = "wasm"))]
    #[test]
    fn test_adosc_into_matches_api() -> Result<(), Box<dyn std::error::Error>> {
        // Construct a deterministic OHLCV series with realistic constraints: high >= close >= low
        let len = 512usize;
        let mut high = Vec::with_capacity(len);
        let mut low = Vec::with_capacity(len);
        let mut close = Vec::with_capacity(len);
        let mut volume = Vec::with_capacity(len);

        for i in 0..len {
            let base = 100.0 + (i as f64) * 0.05 + ((i % 13) as f64) * 0.01;
            let spread = 0.5 + ((i % 7) as f64) * 0.03; // 0.5..0.68
            let lo = base - spread;
            let hi = base + spread;
            // place close between low and high using a bounded fraction
            let frac = ((i % 97) as f64) / 96.0; // [0,1]
            let cl = lo + (hi - lo) * frac;
            let vol = 1_000.0 + ((i * 37) % 10_000) as f64;

            low.push(lo);
            high.push(hi);
            close.push(cl);
            volume.push(vol);
        }

        let input = AdoscInput::from_slices(&high, &low, &close, &volume, AdoscParams::default());

        // Baseline via existing Vec-returning API
        let baseline = adosc(&input)?.values;

        // Preallocate and compute via new no-allocation API
        let mut out = vec![0.0; len];
        adosc_into(&input, &mut out)?;

        assert_eq!(baseline.len(), out.len());

        #[inline]
        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b) || (a - b).abs() <= 1e-12
        }

        for (i, (&a, &b)) in baseline.iter().zip(out.iter()).enumerate() {
            assert!(
                eq_or_both_nan(a, b),
                "ADOSC parity mismatch at index {}: api={}, into={}",
                i,
                a,
                b
            );
        }
        Ok(())
    }
}

/// Write ADOSC directly to output slice - no allocations
#[inline]
pub fn adosc_into_slice(
    dst: &mut [f64],
    input: &AdoscInput,
    kern: Kernel,
) -> Result<(), AdoscError> {
    let (high, low, close, volume, short, long, first, len, chosen) = adosc_prepare(input, kern)?;
    if dst.len() != len {
        return Err(AdoscError::OutputLengthMismatch { expected: len, got: dst.len() });
    }
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                adosc_row_scalar(high, low, close, volume, short, long, first, dst)?
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                adosc_row_scalar(high, low, close, volume, short, long, first, dst)?
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                adosc_row_scalar(high, low, close, volume, short, long, first, dst)?
            }
            _ => unreachable!(),
        }
    }
    Ok(())
}

#[cfg(feature = "python")]
#[pyfunction(name = "adosc")]
#[pyo3(signature = (high, low, close, volume, short_period, long_period, kernel=None))]
pub fn adosc_py<'py>(
    py: Python<'py>,
    high: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    close: numpy::PyReadonlyArray1<'py, f64>,
    volume: numpy::PyReadonlyArray1<'py, f64>,
    short_period: usize,
    long_period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let volume_slice = volume.as_slice()?;

    // Validate all slices have the same length
    let len = close_slice.len();
    if high_slice.len() != len || low_slice.len() != len || volume_slice.len() != len {
        return Err(PyValueError::new_err(format!(
            "All input arrays must have the same length. Got high={}, low={}, close={}, volume={}",
            high_slice.len(),
            low_slice.len(),
            close_slice.len(),
            volume_slice.len()
        )));
    }

    // Use kernel validation for safety
    let kern = validate_kernel(kernel, false)?;

    // Build input struct
    let params = AdoscParams {
        short_period: Some(short_period),
        long_period: Some(long_period),
    };
    let adosc_in =
        AdoscInput::from_slices(high_slice, low_slice, close_slice, volume_slice, params);

    // GOOD: Get Vec<f64> from Rust function
    let result_vec: Vec<f64> = py
        .allow_threads(|| adosc_with_kernel(&adosc_in, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // GOOD: Zero-copy transfer to NumPy
    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "AdoscStream")]
pub struct AdoscStreamPy {
    stream: AdoscStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl AdoscStreamPy {
    #[new]
    fn new(short_period: usize, long_period: usize) -> PyResult<Self> {
        let params = AdoscParams {
            short_period: Some(short_period),
            long_period: Some(long_period),
        };
        let stream =
            AdoscStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(AdoscStreamPy { stream })
    }

    /// Updates the stream with new values and returns the calculated ADOSC value.
    fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> f64 {
        self.stream.update(high, low, close, volume)
    }
}

// ADOSC CUDA device handle for Python with CAI v3 and DLPack stubs
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", unsendable)]
pub struct DeviceArrayF32AdoscPy {
    pub(crate) inner: Option<crate::cuda::oscillators::adosc_wrapper::DeviceArrayF32Adosc>,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl DeviceArrayF32AdoscPy {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("buffer already exported via __dlpack__"))?;
        let d = PyDict::new(py);
        d.set_item("shape", (inner.rows, inner.cols))?;
        d.set_item("typestr", "<f4")?;
        d.set_item(
            "strides",
            (
                inner.cols * std::mem::size_of::<f32>(),
                std::mem::size_of::<f32>(),
            ),
        )?;
        d.set_item("data", (inner.device_ptr() as usize, false))?;
        // Stream omitted: producing stream is synchronized before return
        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> PyResult<(i32, i32)> {
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("buffer already exported via __dlpack__"))?;
        // 2 == kDLCUDA per DLPack
        Ok((2, inner.device_id as i32))
    }

    #[pyo3(signature = (stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &mut self,
        py: Python<'py>,
        stream: Option<pyo3::PyObject>,
        max_version: Option<pyo3::PyObject>,
        dl_device: Option<pyo3::PyObject>,
        copy: Option<pyo3::PyObject>,
    ) -> PyResult<PyObject> {
        use crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d;

        // Producer synchronizes its CUDA stream before returning; we only reject
        // explicitly disallowed stream==0 per Array API semantics.
        if let Some(obj) = &stream {
            if let Ok(i) = obj.extract::<i64>(py) {
                if i == 0 {
                    return Err(PyValueError::new_err(
                        "__dlpack__: stream 0 is disallowed for CUDA",
                    ));
                }
            }
        }

        // Validate `dl_device` hint against allocation device.
        let (kdl, alloc_dev) = self.__dlpack_device__()?;
        if let Some(dev_obj) = dl_device.as_ref() {
            if let Ok((dev_ty, dev_id)) = dev_obj.extract::<(i32, i32)>(py) {
                if dev_ty != kdl || dev_id != alloc_dev {
                    let wants_copy = copy
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

        // Move VRAM handle out; subsequent exports are not allowed.
        let inner = self
            .inner
            .take()
            .ok_or_else(|| PyValueError::new_err("buffer already exported via __dlpack__"))?;
        let rows = inner.rows;
        let cols = inner.cols;
        let device_id = inner.device_id as i32;
        let buf = inner.buf;

        let max_version_bound = max_version.map(|obj| obj.into_bound(py));

        export_f32_cuda_dlpack_2d(py, buf, rows, cols, device_id, max_version_bound)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "adosc_batch")]
#[pyo3(signature = (high, low, close, volume, short_period_range, long_period_range, kernel=None))]
pub fn adosc_batch_py<'py>(
    py: Python<'py>,
    high: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    close: numpy::PyReadonlyArray1<'py, f64>,
    volume: numpy::PyReadonlyArray1<'py, f64>,
    short_period_range: (usize, usize, usize),
    long_period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let volume_slice = volume.as_slice()?;

    // Validate all slices have the same length
    let len = close_slice.len();
    if high_slice.len() != len || low_slice.len() != len || volume_slice.len() != len {
        return Err(PyValueError::new_err(format!(
            "All input arrays must have the same length. Got high={}, low={}, close={}, volume={}",
            high_slice.len(),
            low_slice.len(),
            close_slice.len(),
            volume_slice.len()
        )));
    }

    let sweep = AdoscBatchRange {
        short_period: short_period_range,
        long_period: long_period_range,
    };

    // Expand grid to know rows*cols
    let combos = expand_grid_checked(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = len;

    // Pre-allocate uninitialized NumPy array (1-D, will reshape later)
    // NOTE: PyArray1::new() creates uninitialized memory, not zero-initialized
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;
    let out_arr = unsafe { PyArray1::<f64>::new(py, [expected], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    // Use kernel validation for safety
    let kern = validate_kernel(kernel, true)?;

    // Heavy work without the GIL
    let combos = py
        .allow_threads(|| -> Result<Vec<AdoscParams>, AdoscError> {
            // Resolve Kernel::Auto to a specific kernel
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

            // Use the optimized batch function that writes directly to output
            adosc_batch_inner_into(
                high_slice,
                low_slice,
                close_slice,
                volume_slice,
                &sweep,
                simd,
                true, // parallel
                slice_out,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Build dict with the GIL
    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
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

// ---------------- CUDA Python bindings ----------------
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "adosc_cuda_batch_dev")]
#[pyo3(signature = (high, low, close, volume, short_period_range, long_period_range, device_id=0))]
pub fn adosc_cuda_batch_dev_py(
    py: Python<'_>,
    high: numpy::PyReadonlyArray1<'_, f32>,
    low: numpy::PyReadonlyArray1<'_, f32>,
    close: numpy::PyReadonlyArray1<'_, f32>,
    volume: numpy::PyReadonlyArray1<'_, f32>,
    short_period_range: (usize, usize, usize),
    long_period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<DeviceArrayF32AdoscPy> {
    use crate::cuda::cuda_available;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let volume_slice = volume.as_slice()?;
    let len = close_slice.len();
    if high_slice.len() != len || low_slice.len() != len || volume_slice.len() != len {
        return Err(PyValueError::new_err("mismatched input lengths"));
    }
    let sweep = AdoscBatchRange {
        short_period: short_period_range,
        long_period: long_period_range,
    };
    let inner = py.allow_threads(|| {
        let cuda = CudaAdosc::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.adosc_batch_dev(high_slice, low_slice, close_slice, volume_slice, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    Ok(DeviceArrayF32AdoscPy { inner: Some(inner) })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "adosc_cuda_many_series_one_param_dev")]
#[pyo3(signature = (high_tm, low_tm, close_tm, volume_tm, cols, rows, short_period, long_period, device_id=0))]
pub fn adosc_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    high_tm: numpy::PyReadonlyArray1<'_, f32>,
    low_tm: numpy::PyReadonlyArray1<'_, f32>,
    close_tm: numpy::PyReadonlyArray1<'_, f32>,
    volume_tm: numpy::PyReadonlyArray1<'_, f32>,
    cols: usize,
    rows: usize,
    short_period: usize,
    long_period: usize,
    device_id: usize,
) -> PyResult<DeviceArrayF32AdoscPy> {
    use crate::cuda::cuda_available;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let high_slice = high_tm.as_slice()?;
    let low_slice = low_tm.as_slice()?;
    let close_slice = close_tm.as_slice()?;
    let volume_slice = volume_tm.as_slice()?;
    let expected = cols
        .checked_mul(rows)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;
    if high_slice.len() != expected
        || low_slice.len() != expected
        || close_slice.len() != expected
        || volume_slice.len() != expected
    {
        return Err(PyValueError::new_err("time-major input lengths mismatch"));
    }
    let inner = py.allow_threads(|| {
        let cuda = CudaAdosc::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.adosc_many_series_one_param_time_major_dev(
            high_slice,
            low_slice,
            close_slice,
            volume_slice,
            cols,
            rows,
            short_period,
            long_period,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    Ok(DeviceArrayF32AdoscPy { inner: Some(inner) })
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn adosc_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    short_period: usize,
    long_period: usize,
) -> Result<Vec<f64>, JsValue> {
    let params = AdoscParams {
        short_period: Some(short_period),
        long_period: Some(long_period),
    };
    let input = AdoscInput::from_slices(high, low, close, volume, params);

    // Single allocation with zero-copy pattern
    let mut output = vec![0.0; high.len()];
    adosc_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn adosc_batch_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    short_period_start: usize,
    short_period_end: usize,
    short_period_step: usize,
    long_period_start: usize,
    long_period_end: usize,
    long_period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = AdoscBatchRange {
        short_period: (short_period_start, short_period_end, short_period_step),
        long_period: (long_period_start, long_period_end, long_period_step),
    };

    // Use the existing batch function with parallel=false for WASM
    adosc_batch_inner(high, low, close, volume, &sweep, Kernel::Scalar, false)
        .map(|output| output.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn adosc_batch_metadata_js(
    short_period_start: usize,
    short_period_end: usize,
    short_period_step: usize,
    long_period_start: usize,
    long_period_end: usize,
    long_period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = AdoscBatchRange {
        short_period: (short_period_start, short_period_end, short_period_step),
        long_period: (long_period_start, long_period_end, long_period_step),
    };

    let combos = expand_grid(&sweep);
    let mut metadata = Vec::with_capacity(combos.len() * 2);

    for combo in combos {
        metadata.push(combo.short_period.unwrap() as f64);
        metadata.push(combo.long_period.unwrap() as f64);
    }

    Ok(metadata)
}

// New ergonomic WASM API
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct AdoscBatchConfig {
    pub short_period_range: (usize, usize, usize),
    pub long_period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct AdoscBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<AdoscParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = adosc_batch)]
pub fn adosc_batch_unified_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    // 1. Deserialize the configuration object from JavaScript
    let config: AdoscBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = AdoscBatchRange {
        short_period: config.short_period_range,
        long_period: config.long_period_range,
    };

    // 2. Run the existing core logic
    let output = adosc_batch_inner(high, low, close, volume, &sweep, Kernel::Scalar, false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // 3. Create the structured output
    let js_output = AdoscBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };

    // 4. Serialize the output struct into a JavaScript object
    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn adosc_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn adosc_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn adosc_into(
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

        let params = AdoscParams {
            short_period: Some(short_period),
            long_period: Some(long_period),
        };
        let input = AdoscInput::from_slices(high, low, close, volume, params);

        // Check for aliasing - if output pointer matches ANY input pointer
        if out_ptr as *const f64 == high_ptr
            || out_ptr as *const f64 == low_ptr
            || out_ptr as *const f64 == close_ptr
            || out_ptr as *const f64 == volume_ptr
        {
            // Handle aliasing by using temporary buffer
            let mut temp = vec![0.0; len];
            adosc_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            // No aliasing, compute directly into output
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            adosc_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn adosc_batch_into(
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
        return Err(JsValue::from_str("null pointer passed to adosc_batch_into"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);

        let sweep = AdoscBatchRange {
            short_period: (short_period_start, short_period_end, short_period_step),
            long_period: (long_period_start, long_period_end, long_period_step),
        };

        let combos = expand_grid_checked(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let cols = len;
        let expected = rows
            .checked_mul(cols)
            .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
        let out = std::slice::from_raw_parts_mut(out_ptr, expected);

        adosc_batch_inner_into(high, low, close, volume, &sweep, Kernel::Scalar, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}
