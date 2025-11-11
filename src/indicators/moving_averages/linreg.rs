//! # Linear Regression (LINREG)
//!
//! Fits a straight line (y = a + b·x) to recent data over a rolling window and forecasts the next value.
//! Supports kernels (scalar, AVX2, AVX512), streaming, batch grid, custom input sources, error reporting, and parameter builders.
//!
//! ## Parameters
//! - **period**: Look-back window size (default: 14).
//!
//! ## Errors
//! - **AllValuesNaN**: linreg: All input data values are `NaN`.
//! - **InvalidPeriod**: linreg: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: linreg: Not enough valid data points for the requested `period`.
//!
//! ## Returns
//! - **`Ok(LinRegOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(LinRegError)`** otherwise.
//!
//! ## Developer Notes
//! - **AVX2/AVX512 kernels**: Implemented but disabled by default (Auto short-circuits to Scalar). On typical CPUs, AVX downclocks and the small one-time init SIMD gives net slower results at 100k. Revisit with prefix sums-based row kernels.
//! - **Streaming update**: ⚠️ O(n) - `dot_ring()` recalculates regression over full buffer
//! - **Memory optimization**: ✅ Uses `alloc_with_nan_prefix` for zero-copy output allocation
//! - **Current status**: Functional with efficient scalar implementation but missing SIMD optimizations
//! - **Optimization opportunities**:
//!   - Implement AVX2 kernel for vectorized sum calculations
//!   - Implement AVX512 kernel for wider vector processing
//!   - Streaming update could potentially maintain incremental sums for O(1) updates
//!   - Linear regression calculations are well-suited for SIMD parallelization

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::{
    cuda::moving_averages::CudaLinreg, indicators::moving_averages::alma::DeviceArrayF32Py,
};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

// --- DATA, PARAMS, INPUT/OUTPUT STRUCTS ---

#[derive(Debug, Clone)]
pub enum LinRegData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct LinRegOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct LinRegParams {
    pub period: Option<usize>,
}

impl Default for LinRegParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct LinRegInput<'a> {
    pub data: LinRegData<'a>,
    pub params: LinRegParams,
}

impl<'a> AsRef<[f64]> for LinRegInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            LinRegData::Slice(slice) => slice,
            LinRegData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

impl<'a> LinRegInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: LinRegParams) -> Self {
        Self {
            data: LinRegData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: LinRegParams) -> Self {
        Self {
            data: LinRegData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", LinRegParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
}

// --- BUILDER ---

#[derive(Copy, Clone, Debug)]
pub struct LinRegBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for LinRegBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl LinRegBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<LinRegOutput, LinRegError> {
        let p = LinRegParams {
            period: self.period,
        };
        let i = LinRegInput::from_candles(c, "close", p);
        linreg_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<LinRegOutput, LinRegError> {
        let p = LinRegParams {
            period: self.period,
        };
        let i = LinRegInput::from_slice(d, p);
        linreg_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<LinRegStream, LinRegError> {
        let p = LinRegParams {
            period: self.period,
        };
        LinRegStream::try_new(p)
    }
}

// --- ERRORS ---

#[derive(Debug, Error)]
pub enum LinRegError {
    #[error("linreg: No data provided.")]
    EmptyInputData,
    #[error("linreg: All values are NaN.")]
    AllValuesNaN,
    #[error("linreg: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("linreg: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("linreg: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("linreg: Invalid range: start = {start}, end = {end}, step = {step}")]
    InvalidRange { start: usize, end: usize, step: usize },
    #[error("linreg: Invalid kernel for batch API: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("linreg: arithmetic overflow when computing {what}")]
    ArithmeticOverflow { what: &'static str },
}

// --- INDICATOR API ---

#[inline]
pub fn linreg(input: &LinRegInput) -> Result<LinRegOutput, LinRegError> {
    linreg_with_kernel(input, Kernel::Auto)
}

/// Prepare LinReg computation parameters (zero-copy pattern)
#[inline(always)]
fn linreg_prepare<'a>(
    input: &'a LinRegInput,
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
    LinRegError,
> {
    let data: &[f64] = input.as_ref();
    if data.is_empty() {
        return Err(LinRegError::EmptyInputData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(LinRegError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(LinRegError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(LinRegError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        // SIMD underperforms for this indicator; keep Auto on the scalar path
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    Ok((data, period, first, chosen))
}

pub fn linreg_with_kernel(
    input: &LinRegInput,
    kernel: Kernel,
) -> Result<LinRegOutput, LinRegError> {
    let (data, period, first, chosen) = linreg_prepare(input, kernel)?;

    let warm = first + period;
    let mut out = alloc_with_nan_prefix(data.len(), warm);

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => linreg_scalar(data, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => linreg_avx2(data, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => linreg_avx512(data, period, first, &mut out),
            _ => unreachable!(),
        }
    }

    Ok(LinRegOutput { values: out })
}

/// LinReg zero-allocation API: writes results into the caller-provided buffer.
///
/// - Preserves NaN warmups exactly like `linreg()` (Auto short-circuits to Scalar).
/// - The output slice length must equal the input length.
/// - Returns `Ok(())` on success, or the same errors as the Vec-returning API on failure.
#[cfg(not(feature = "wasm"))]
#[inline]
pub fn linreg_into(input: &LinRegInput, out: &mut [f64]) -> Result<(), LinRegError> {
    // Match `linreg()` semantics where Kernel::Auto maps to Scalar for this indicator.
    // Use the existing compute_into path for warmup handling and kernel dispatch.
    linreg_compute_into(input, Kernel::Scalar, out)
}

/// Compute LinReg directly into pre-allocated output slice (zero-copy)
pub fn linreg_compute_into(
    input: &LinRegInput,
    kernel: Kernel,
    out: &mut [f64],
) -> Result<(), LinRegError> {
    let data: &[f64] = input.as_ref();
    if data.is_empty() {
        return Err(LinRegError::EmptyInputData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(LinRegError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(LinRegError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(LinRegError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }
    if out.len() != len {
        return Err(LinRegError::OutputLengthMismatch { expected: len, got: out.len() });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let warm = first + period;
    // Initialize warmup period with NaN
    out[..warm].fill(f64::NAN);

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => linreg_scalar(data, period, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => linreg_avx2(data, period, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => linreg_avx512(data, period, first, out),
            _ => unreachable!(),
        }
    }

    Ok(())
}

#[inline(always)]
fn linreg_scalar(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    // ---- invariant pre‑computations ---------------------------------------------------------
    let period_f = period as f64;
    let x_sum = ((period * (period + 1)) / 2) as f64; // Σx
    let x2_sum = ((period * (period + 1) * (2 * period + 1)) / 6) as f64; // Σx²
    let denom_inv = 1.0 / (period_f * x2_sum - x_sum * x_sum); // 1 / Δ
    let inv_period = 1.0 / period_f;

    // ---- prime running sums with the first (period‑1) points -------------------------------
    let mut y_sum = 0.0;
    let mut xy_sum = 0.0;
    let init_slice = &data[first..first + period - 1];
    let mut k = 1usize;
    for &v in init_slice.iter() {
        y_sum += v;
        xy_sum += (k as f64) * v;
        k += 1;
    }

    // ---- main rolling loop -----------------------------------------------------------------
    let len = data.len();
    let mut idx = first + period - 1; // index of *last* element in the current window
    let mut old_idx = first; // index of the oldest element in the current window
    while idx < len {
        let new_val = data[idx];
        y_sum += new_val; // include newest sample
        xy_sum += new_val * period_f; // its x = period

        // coefficients
        let b = (period_f * xy_sum - x_sum * y_sum) * denom_inv;
        let a = (y_sum - b * x_sum) * inv_period;
        out[idx] = a + b * period_f; // forecast next point (x = period)

        // slide window: remove oldest point and shift indices by ‑1
        xy_sum -= y_sum; // Σ(x·y) → Σ((x‑1)·y)
        y_sum -= data[old_idx]; // drop y_{t‑period+1}

        idx += 1;
        old_idx += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn linreg_avx2(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    use core::arch::x86_64::*;

    // ---- invariants -----------------------------------------------------------
    let pf = period as f64;
    let x_sum = ((period * (period + 1)) / 2) as f64; // Σx
    let x2_sum = ((period * (period + 1) * (2 * period + 1)) / 6) as f64; // Σx²
    let denom_inv = 1.0 / (pf * x2_sum - x_sum * x_sum);
    let inv_pf = 1.0 / pf;

    // ---- vectorized init over first (period-1) -------------------------------
    let mut y_sum = 0.0f64;
    let mut xy_sum = 0.0f64;

    let init_len = period.saturating_sub(1);
    let mut p = data.as_ptr().add(first);

    // process 4 at a time
    let vec_blocks = init_len / 4;
    if vec_blocks > 0 {
        let base = _mm256_setr_pd(1.0, 2.0, 3.0, 4.0);
        let mut off = 0.0f64;
        let mut y_acc = _mm256_set1_pd(0.0);
        let mut xy_acc = _mm256_set1_pd(0.0);

        for _ in 0..vec_blocks {
            let y = _mm256_loadu_pd(p);
            let x = _mm256_add_pd(base, _mm256_set1_pd(off));
            y_acc = _mm256_add_pd(y_acc, y);
            xy_acc = _mm256_fmadd_pd(y, x, xy_acc);
            p = p.add(4);
            off += 4.0;
        }

        // horizontal reduce by store + scalar sum (cheap; one-time)
        let mut buf = [0.0f64; 4];
        _mm256_storeu_pd(buf.as_mut_ptr(), y_acc);
        y_sum += buf.iter().sum::<f64>();
        _mm256_storeu_pd(buf.as_mut_ptr(), xy_acc);
        xy_sum += buf.iter().sum::<f64>();
    }

    // scalar tail for remaining (period-1) % 4
    let tail = init_len & 3;
    let mut k_off = (vec_blocks * 4 + 1) as f64; // next x value
    for _ in 0..tail {
        let v = *p;
        y_sum += v;
        xy_sum += k_off * v;
        k_off += 1.0;
        p = p.add(1);
    }

    // ---- main rolling loop ---------------------------------------------------
    let len = data.len();
    let mut idx = first + period - 1;
    let mut old_idx = first;
    while idx < len {
        let new_v = *data.get_unchecked(idx);
        y_sum += new_v;
        xy_sum = f64::mul_add(pf, new_v, xy_sum);

        let b = (pf * xy_sum - x_sum * y_sum) * denom_inv;
        let a = (y_sum - b * x_sum) * inv_pf;
        *out.get_unchecked_mut(idx) = a + b * pf;

        xy_sum -= y_sum;
        y_sum -= *data.get_unchecked(old_idx);
        idx += 1;
        old_idx += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512dq,fma")]
pub unsafe fn linreg_avx512(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    use core::arch::x86_64::*;

    // ---- invariants -----------------------------------------------------------
    let pf = period as f64;
    let x_sum = ((period * (period + 1)) / 2) as f64; // Σx
    let x2_sum = ((period * (period + 1) * (2 * period + 1)) / 6) as f64; // Σx²
    let denom_inv = 1.0 / (pf * x2_sum - x_sum * x_sum);
    let inv_pf = 1.0 / pf;

    // ---- vectorized init over first (period-1) -------------------------------
    let mut y_sum = 0.0f64;
    let mut xy_sum = 0.0f64;

    let init_len = period.saturating_sub(1);
    let mut p = data.as_ptr().add(first);

    // process 8 at a time
    let vec_blocks = init_len / 8;
    if vec_blocks > 0 {
        let base = _mm512_setr_pd(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let mut off = 0.0f64;
        let mut y_acc = _mm512_set1_pd(0.0);
        let mut xy_acc = _mm512_set1_pd(0.0);

        for _ in 0..vec_blocks {
            let y = _mm512_loadu_pd(p);
            let x = _mm512_add_pd(base, _mm512_set1_pd(off));
            y_acc = _mm512_add_pd(y_acc, y);
            xy_acc = _mm512_fmadd_pd(y, x, xy_acc);
            p = p.add(8);
            off += 8.0;
        }

        // horizontal reduce by store + scalar sum
        let mut buf = [0.0f64; 8];
        _mm512_storeu_pd(buf.as_mut_ptr(), y_acc);
        y_sum += buf.iter().sum::<f64>();
        _mm512_storeu_pd(buf.as_mut_ptr(), xy_acc);
        xy_sum += buf.iter().sum::<f64>();
    }

    // scalar tail for remaining (period-1) % 8
    let tail = init_len & 7;
    let mut k_off = (vec_blocks * 8 + 1) as f64; // next x value
    for _ in 0..tail {
        let v = *p;
        y_sum += v;
        xy_sum += k_off * v;
        k_off += 1.0;
        p = p.add(1);
    }

    // ---- main rolling loop ---------------------------------------------------
    let len = data.len();
    let mut idx = first + period - 1;
    let mut old_idx = first;
    while idx < len {
        let new_v = *data.get_unchecked(idx);
        y_sum += new_v;
        xy_sum = f64::mul_add(pf, new_v, xy_sum);

        let b = (pf * xy_sum - x_sum * y_sum) * denom_inv;
        let a = (y_sum - b * x_sum) * inv_pf;
        *out.get_unchecked_mut(idx) = a + b * pf;

        xy_sum -= y_sum;
        y_sum -= *data.get_unchecked(old_idx);
        idx += 1;
        old_idx += 1;
    }
}

// --- BATCH RANGE/BUILDER/OUTPUT/GRID ---

#[derive(Clone, Debug)]
pub struct LinRegBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for LinRegBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 40, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct LinRegBatchBuilder {
    range: LinRegBatchRange,
    kernel: Kernel,
}

impl LinRegBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.period = (start, end, step);
        self
    }
    pub fn period_static(mut self, p: usize) -> Self {
        self.range.period = (p, p, 0);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<LinRegBatchOutput, LinRegError> {
        linreg_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<LinRegBatchOutput, LinRegError> {
        LinRegBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<LinRegBatchOutput, LinRegError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<LinRegBatchOutput, LinRegError> {
        LinRegBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct LinRegBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<LinRegParams>,
    pub rows: usize,
    pub cols: usize,
}

impl LinRegBatchOutput {
    pub fn row_for_params(&self, p: &LinRegParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
    }
    pub fn values_for(&self, p: &LinRegParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

// --- BATCH MAIN ENTRYPOINTS ---

pub fn linreg_batch_with_kernel(
    data: &[f64],
    sweep: &LinRegBatchRange,
    k: Kernel,
) -> Result<LinRegBatchOutput, LinRegError> {
    let kernel = match k {
        // Row-specific SIMD not implemented; default to ScalarBatch for best results
        Kernel::Auto => Kernel::ScalarBatch,
        other if other.is_batch() => other,
        _ => return Err(LinRegError::InvalidKernelForBatch(k)),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    linreg_batch_par_slice(data, sweep, simd)
}

#[inline(always)]
pub fn linreg_batch_slice(
    data: &[f64],
    sweep: &LinRegBatchRange,
    kern: Kernel,
) -> Result<LinRegBatchOutput, LinRegError> {
    linreg_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn linreg_batch_par_slice(
    data: &[f64],
    sweep: &LinRegBatchRange,
    kern: Kernel,
) -> Result<LinRegBatchOutput, LinRegError> {
    linreg_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn linreg_batch_inner(
    data: &[f64],
    sweep: &LinRegBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<LinRegBatchOutput, LinRegError> {
    // ------------- 0. sanity checks -------------
    let combos = expand_grid_linreg(sweep);
    if combos.is_empty() {
        let (s, e, t) = sweep.period;
        return Err(LinRegError::InvalidRange { start: s, end: e, step: t });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(LinRegError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(LinRegError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    // ------------- 1. matrix set-up -------------
    let rows = combos.len();
    let cols = data.len();
    let _ = rows
        .checked_mul(cols)
        .ok_or(LinRegError::ArithmeticOverflow { what: "rows*cols" })?;

    // per-row prefix length that must stay NaN
    let warm: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap()).collect();

    // allocate rows × cols as MaybeUninit and paint the prefixes
    let mut raw = make_uninit_matrix(rows, cols);
    unsafe { init_matrix_prefixes(&mut raw, cols, &warm) };

    // ------------- 2. per-row worker ------------
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();

        // cast this single row to &mut [f64]
        let out_row =
            core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        match kern {
            Kernel::Scalar => linreg_row_scalar(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => linreg_row_avx2(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => linreg_row_avx512(data, first, period, out_row),
            _ => unreachable!(),
        }
    };

    // ------------- 3. run every row -------------
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

    // ------------- 4. transmute -----------------
    let values: Vec<f64> = unsafe { std::mem::transmute(raw) };

    Ok(LinRegBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

/// Batch compute LinReg directly into pre-allocated output slice (zero-copy)
pub fn linreg_batch_inner_into(
    data: &[f64],
    sweep: &LinRegBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<LinRegParams>, LinRegError> {
    // ------------- 0. prelim checks -------------
    let combos = expand_grid_linreg(sweep);
    if combos.is_empty() {
        let (s, e, t) = sweep.period;
        return Err(LinRegError::InvalidRange { start: s, end: e, step: t });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(LinRegError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(LinRegError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let expected = rows
        .checked_mul(cols)
        .ok_or(LinRegError::ArithmeticOverflow { what: "rows*cols" })?;
    // Validate output slice size
    if out.len() != expected {
        return Err(LinRegError::OutputLengthMismatch { expected, got: out.len() });
    }

    // ------------- 1a. cast output slice to MaybeUninit -------------
    let out_uninit = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };

    // per-row prefix length that must stay NaN
    let warm: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap()).collect();

    // paint the prefixes with NaN
    unsafe { init_matrix_prefixes(out_uninit, cols, &warm) };

    // ------------- 1b. (optional) shared prefix sums over valid region -------------
    // NOTE: A row-optimized variant using prefix sums (S, SP) was evaluated but
    // did not outperform the rolling-update scalar row kernel on this workload.
    // We keep the implementation below (linreg_row_prefix_sums_scalar) for future
    // experiments, but do not compute prefix sums by default to avoid overhead.

    // ------------- 2. per-row worker ------------
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();

        // cast this single row to &mut [f64]
        let out_row =
            core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        match kern {
            // Keep fastest known-good path
            Kernel::Scalar => linreg_row_scalar(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => linreg_row_avx2(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => linreg_row_avx512(data, first, period, out_row),
            _ => unreachable!(),
        }
    };

    // ------------- 3. run every row -------------
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
// --- ROW SCALAR/SIMD (all AVX just call scalar for parity) ---

#[inline(always)]
unsafe fn linreg_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    linreg_scalar(data, period, first, out)
}

#[inline(always)]
unsafe fn linreg_row_prefix_sums_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
    s: &[f64],
    sp: &[f64],
) {
    let len = data.len();
    let pf = period as f64;
    let x_sum = ((period * (period + 1)) / 2) as f64; // Σx
    let x2_sum = ((period * (period + 1) * (2 * period + 1)) / 6) as f64; // Σx²
    let denom_inv = 1.0 / (pf * x2_sum - x_sum * x_sum);
    let inv_pf = 1.0 / pf;

    // Start at the first full window
    let mut idx = first + period - 1;
    while idx < len {
        // 1-based position within the valid region [first..]
        let pos = idx - first + 1;
        let y_sum = s.get_unchecked(pos) - s.get_unchecked(pos - period);
        // Rebase absolute-index prefix sum to local x ∈ [1..period]
        let xy_sum = (sp.get_unchecked(pos) - sp.get_unchecked(pos - period))
            - ((pos - period) as f64) * y_sum;

        let b = (pf * xy_sum - x_sum * y_sum) * denom_inv;
        let a = (y_sum - b * x_sum) * inv_pf;
        *out.get_unchecked_mut(idx) = a + b * pf;

        idx += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn linreg_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    linreg_avx2(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn linreg_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    linreg_avx512(data, period, first, out)
}

// --- STREAM SUPPORT ---

#[derive(Debug, Clone)]
pub struct LinRegStream {
    period: usize,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,
    x_sum: f64,
    x2_sum: f64,
}

impl LinRegStream {
    pub fn try_new(params: LinRegParams) -> Result<Self, LinRegError> {
        let period = params.period.unwrap_or(14);
        if period == 0 {
            return Err(LinRegError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let mut x_sum = 0.0;
        let mut x2_sum = 0.0;
        for i in 1..=period {
            let xi = i as f64;
            x_sum += xi;
            x2_sum += xi * xi;
        }
        Ok(Self {
            period,
            buffer: vec![f64::NAN; period],
            head: 0,
            filled: false,
            x_sum,
            x2_sum,
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
        Some(self.dot_ring())
    }

    #[inline(always)]
    fn dot_ring(&self) -> f64 {
        let mut y_sum = 0.0;
        let mut xy_sum = 0.0;
        for (i, &y) in
            (1..=self.period).zip(self.buffer.iter().cycle().skip(self.head).take(self.period))
        {
            y_sum += y;
            xy_sum += y * (i as f64);
        }
        let pf = self.period as f64;
        let bd = 1.0 / (pf * self.x2_sum - self.x_sum * self.x_sum);
        let b = (pf * xy_sum - self.x_sum * y_sum) * bd;
        let a = (y_sum - b * self.x_sum) / pf;
        a + b * pf
    }
}

// --- BATCH GRID ---

#[inline(always)]
fn round_up8(x: usize) -> usize {
    (x + 7) & !7
}

// --- EXPOSED BATCH EXPANSION ---

#[inline(always)]
pub fn expand_grid_linreg(r: &LinRegBatchRange) -> Vec<LinRegParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        let (lo, hi) = if start <= end { (start, end) } else { (end, start) };
        let mut v = Vec::new();
        let mut x = lo;
        while x <= hi {
            v.push(x);
            match x.checked_add(step) {
                Some(nx) => x = nx,
                None => break,
            }
        }
        v
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(LinRegParams { period: Some(p) });
    }
    out
}

// --- TESTS ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_linreg_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let close_prices = candles.select_candle_field("close")?;
        let params = LinRegParams { period: Some(14) };
        let input = LinRegInput::from_candles(&candles, "close", params);
        let linreg_result = linreg_with_kernel(&input, kernel)?;
        let expected_last_five = [
            58929.37142857143,
            58899.42857142857,
            58918.857142857145,
            59100.6,
            58987.94285714286,
        ];
        assert!(linreg_result.values.len() >= 5);
        assert_eq!(linreg_result.values.len(), close_prices.len());
        let start_index = linreg_result.values.len() - 5;
        let result_last_five = &linreg_result.values[start_index..];
        for (i, &value) in result_last_five.iter().enumerate() {
            let expected_value = expected_last_five[i];
            assert!(
                (value - expected_value).abs() < 1e-1,
                "Mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }
        Ok(())
    }

    fn check_linreg_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = LinRegParams { period: None };
        let input = LinRegInput::from_candles(&candles, "close", default_params);
        let output = linreg_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_linreg_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = LinRegInput::with_default_candles(&candles);
        match input.data {
            LinRegData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected LinRegData::Candles"),
        }
        let output = linreg_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    #[test]
    fn test_linreg_into_matches_api() -> Result<(), Box<dyn Error>> {
        // Build a small but non-trivial input: NaN warmup prefix + trending/oscillating data
        let mut data = Vec::with_capacity(5 + 256);
        for _ in 0..5 {
            data.push(f64::NAN);
        }
        for i in 0..256u32 {
            let x = i as f64;
            let v = (x * 0.137).sin() * 3.0 + x * 0.25; // mild sinusoid on a trend
            data.push(v);
        }

        let params = LinRegParams { period: Some(14) };
        let input = LinRegInput::from_slice(&data, params);

        // Baseline via existing Vec-returning API (Auto -> Scalar for this indicator)
        let baseline = linreg(&input)?.values;

        // Preallocate and compute in-place
        let mut out = vec![0.0; data.len()];
        linreg_into(&input, &mut out)?;

        assert_eq!(out.len(), baseline.len());
        for (i, (&a, &b)) in out.iter().zip(baseline.iter()).enumerate() {
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan(), "NaN parity mismatch at index {}", i);
            } else {
                let diff = (a - b).abs();
                assert!(diff <= 1e-12, "Value mismatch at index {}: {} vs {} (diff={})", i, a, b, diff);
            }
        }
        Ok(())
    }

    fn check_linreg_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = LinRegParams { period: Some(0) };
        let input = LinRegInput::from_slice(&input_data, params);
        let res = linreg_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] LINREG should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_linreg_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = LinRegParams { period: Some(10) };
        let input = LinRegInput::from_slice(&data_small, params);
        let res = linreg_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] LINREG should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_linreg_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = LinRegParams { period: Some(14) };
        let input = LinRegInput::from_slice(&single_point, params);
        let res = linreg_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] LINREG should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_linreg_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = LinRegParams { period: Some(14) };
        let first_input = LinRegInput::from_candles(&candles, "close", first_params);
        let first_result = linreg_with_kernel(&first_input, kernel)?;
        let second_params = LinRegParams { period: Some(10) };
        let second_input = LinRegInput::from_slice(&first_result.values, second_params);
        let second_result = linreg_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    fn check_linreg_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = LinRegInput::from_candles(&candles, "close", LinRegParams { period: Some(14) });
        let res = linreg_with_kernel(&input, kernel)?;
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

    fn check_linreg_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 14;
        let input = LinRegInput::from_candles(
            &candles,
            "close",
            LinRegParams {
                period: Some(period),
            },
        );
        let batch_output = linreg_with_kernel(&input, kernel)?.values;
        let mut stream = LinRegStream::try_new(LinRegParams {
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
                diff < 1e-6,
                "[{}] LINREG streaming mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_linreg_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(#[test] fn [<$test_fn _scalar_f64>]() { let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar); })*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(
                    #[test] fn [<$test_fn _avx2_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2); }
                    #[test] fn [<$test_fn _avx512_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512); }
                )*
            }
        }
    }

    // Check for poison values in single output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_linreg_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test with multiple parameter combinations to better catch uninitialized memory bugs
        let test_periods = vec![2, 5, 10, 14, 20, 30, 50, 100, 200];
        let test_sources = vec!["open", "high", "low", "close", "hl2", "hlc3", "ohlc4"];

        for period in &test_periods {
            for source in &test_sources {
                let input = LinRegInput::from_candles(
                    &candles,
                    source,
                    LinRegParams {
                        period: Some(*period),
                    },
                );
                let output = linreg_with_kernel(&input, kernel)?;

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
    fn check_linreg_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    fn check_linreg_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Load real market data for realistic testing
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let close_data = &candles.close;

        // Strategy: test various parameter combinations with real data slices
        let strat = (
            2usize..=50, // period (must be >= 2 for meaningful regression)
            0usize..close_data.len().saturating_sub(200), // starting index for data slice
            100usize..=200, // length of data slice to use
        );

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(period, start_idx, slice_len)| {
                // Ensure we have valid slice bounds
                let end_idx = (start_idx + slice_len).min(close_data.len());
                if end_idx <= start_idx || end_idx - start_idx < period + 10 {
                    return Ok(()); // Skip invalid combinations
                }

                let data_slice = &close_data[start_idx..end_idx];
                let params = LinRegParams {
                    period: Some(period),
                };
                let input = LinRegInput::from_slice(data_slice, params.clone());

                // Test the specified kernel
                let result = linreg_with_kernel(&input, kernel);

                // Also compute with scalar kernel for reference
                let scalar_result = linreg_with_kernel(&input, Kernel::Scalar);

                // Both should succeed or fail together
                match (result, scalar_result) {
                    (Ok(LinRegOutput { values: out }), Ok(LinRegOutput { values: ref_out })) => {
                        // Verify output length
                        prop_assert_eq!(out.len(), data_slice.len());
                        prop_assert_eq!(ref_out.len(), data_slice.len());

                        // Find first non-NaN value in input
                        let first = data_slice.iter().position(|x| !x.is_nan()).unwrap_or(0);
                        let expected_warmup = first + period;

                        // Find first non-NaN value in output
                        let first_valid = out.iter().position(|x| !x.is_nan());
                        if let Some(first_idx) = first_valid {
                            // Verify warmup period is correct
                            prop_assert_eq!(
                                first_idx,
                                expected_warmup,
                                "First valid at {} but expected warmup is {}",
                                first_idx,
                                expected_warmup
                            );

                            // Check NaN pattern - all values before first_valid should be NaN
                            for i in 0..first_idx {
                                prop_assert!(
                                    out[i].is_nan(),
                                    "Expected NaN at index {} during warmup, got {}",
                                    i,
                                    out[i]
                                );
                            }
                        }

                        // Verify kernel consistency
                        for i in 0..out.len() {
                            let y = out[i];
                            let r = ref_out[i];

                            // Both should be NaN or both should be valid
                            if y.is_nan() {
                                prop_assert!(
                                    r.is_nan(),
                                    "Kernel mismatch at {}: {} vs {}",
                                    i,
                                    y,
                                    r
                                );
                                continue;
                            }

                            // Check finite values
                            prop_assert!(y.is_finite(), "Non-finite value at index {}: {}", i, y);

                            // Compare with scalar reference (allowing for floating-point precision)
                            let ulps_diff = if y == r {
                                0
                            } else {
                                let y_bits = y.to_bits();
                                let r_bits = r.to_bits();
                                ((y_bits as i64) - (r_bits as i64)).unsigned_abs()
                            };

                            prop_assert!(
                                ulps_diff <= 3 || (y - r).abs() < 1e-9,
                                "Kernel mismatch at {}: {} vs {} (diff: {}, ulps: {})",
                                i,
                                y,
                                r,
                                (y - r).abs(),
                                ulps_diff
                            );
                        }

                        // Test specific properties of linear regression
                        if first_valid.is_some() {
                            // Property 1: Test with synthetic linear data
                            let mut linear_data = vec![0.0; period + 5];
                            for i in 0..linear_data.len() {
                                linear_data[i] = 100.0 + i as f64 * 2.0; // y = 100 + 2x
                            }
                            let linear_input =
                                LinRegInput::from_slice(&linear_data, params.clone());
                            if let Ok(LinRegOutput { values: linear_out }) =
                                linreg_with_kernel(&linear_input, kernel)
                            {
                                // For perfectly linear data, prediction should be exact
                                for i in period..linear_data.len() {
                                    if !linear_out[i].is_nan() {
                                        let expected = 100.0 + (i + 1) as f64 * 2.0; // Next value in sequence
                                        prop_assert!(
                                            (linear_out[i] - expected).abs() < 1e-6,
                                            "Linear prediction error at {}: got {} expected {}",
                                            i,
                                            linear_out[i],
                                            expected
                                        );
                                    }
                                }
                            }

                            // Property 2: Test with constant data
                            let constant_val = 42.0;
                            let constant_data = vec![constant_val; period + 5];
                            let const_input = LinRegInput::from_slice(&constant_data, params);
                            if let Ok(LinRegOutput { values: const_out }) =
                                linreg_with_kernel(&const_input, kernel)
                            {
                                // For constant data, output should equal the constant
                                for i in period..constant_data.len() {
                                    if !const_out[i].is_nan() {
                                        prop_assert!(
                                            (const_out[i] - constant_val).abs() < 1e-9,
                                            "Constant prediction error at {}: got {} expected {}",
                                            i,
                                            const_out[i],
                                            constant_val
                                        );
                                    }
                                }
                            }

                            // Property 3: Output bounds check for real data
                            for i in expected_warmup..out.len() {
                                if !out[i].is_nan() {
                                    // Get the window that was used for this calculation
                                    let window_start = i + 1 - period;
                                    let window_end = i + 1;
                                    let window = &data_slice[window_start..window_end];

                                    let min_val =
                                        window.iter().copied().fold(f64::INFINITY, f64::min);
                                    let max_val =
                                        window.iter().copied().fold(f64::NEG_INFINITY, f64::max);

                                    // Linear regression can extrapolate beyond the window bounds,
                                    // but should be within a reasonable range (e.g., 2x the range)
                                    let range = max_val - min_val;
                                    let lower_bound = min_val - range;
                                    let upper_bound = max_val + range;

                                    prop_assert!(
                                        out[i] >= lower_bound && out[i] <= upper_bound,
                                        "Output {} at index {} outside reasonable bounds [{}, {}]",
                                        out[i],
                                        i,
                                        lower_bound,
                                        upper_bound
                                    );
                                }
                            }
                        }

                        Ok(())
                    }
                    (Err(e1), Err(e2)) => {
                        // Both kernels should fail with the same error type
                        prop_assert_eq!(
                            std::mem::discriminant(&e1),
                            std::mem::discriminant(&e2),
                            "Different error types: {:?} vs {:?}",
                            e1,
                            e2
                        );
                        Ok(())
                    }
                    _ => {
                        prop_assert!(
                            false,
                            "Kernel consistency failed - one succeeded, one failed"
                        );
                        Ok(())
                    }
                }
            })
            .map_err(|e| e.into())
    }

    generate_all_linreg_tests!(
        check_linreg_accuracy,
        check_linreg_partial_params,
        check_linreg_default_candles,
        check_linreg_zero_period,
        check_linreg_period_exceeds_length,
        check_linreg_very_small_dataset,
        check_linreg_reinput,
        check_linreg_nan_handling,
        check_linreg_streaming,
        check_linreg_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_linreg_tests!(check_linreg_property);

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = LinRegBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = LinRegParams::default();
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

        // Test batch with multiple parameter combinations to better catch uninitialized memory bugs
        let test_sources = vec!["open", "high", "low", "close", "hl2", "hlc3", "ohlc4"];

        for source in &test_sources {
            // Test with comprehensive period ranges
            let output = LinRegBatchBuilder::new()
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
            let output = LinRegBatchBuilder::new()
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
}

// Python bindings
#[cfg(feature = "python")]
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
use numpy::IntoPyArray;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "linreg", signature = (data, period, kernel=None))]
pub fn linreg_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let params = LinRegParams {
        period: Some(period),
    };
    let input = LinRegInput::from_slice(slice_in, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| linreg_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "linreg_batch", signature = (data, period_range, kernel=None))]
pub fn linreg_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let sweep = LinRegBatchRange {
        period: period_range,
    };

    // Calculate dimensions
    let combos = expand_grid_linreg(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();

    // Pre-allocate output array (OK for batch operations)
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
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

            linreg_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Build result dictionary
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
#[pyfunction(name = "linreg_cuda_batch_dev")]
#[pyo3(signature = (data_f32, period_range, device_id=0))]
pub fn linreg_cuda_batch_dev_py<'py>(
    py: Python<'py>,
    data_f32: PyReadonlyArray1<'py, f32>,
    period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<(DeviceArrayF32Py, Bound<'py, PyDict>)> {
    use crate::cuda::cuda_available;
    use numpy::IntoPyArray;
    use pyo3::types::PyDict;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let slice_in = data_f32.as_slice()?;
    let sweep = LinRegBatchRange {
        period: period_range,
    };

    let (inner, combos, stream_u64) = py.allow_threads(|| {
        let cuda = CudaLinreg::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let res = cuda
            .linreg_batch_dev(slice_in, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, PyErr>((res.0, res.1, cuda.stream_handle_u64()))
    })?;

    let dict = PyDict::new(py);
    let periods: Vec<u64> = combos.iter().map(|c| c.period.unwrap() as u64).collect();
    dict.set_item("periods", periods.into_pyarray(py))?;
    // CUDA Array Interface v3 helper metadata for consumers; no sync performed here.
    dict.set_item("cai_version", 3u64)?;
    dict.set_item("cai_typestr", "<f4")?; // little-endian f32
    dict.set_item("cai_shape", (inner.rows as u64, inner.cols as u64))?;
    dict.set_item("cai_strides_bytes", ((inner.cols as u64) * 4u64, 4u64))?;
    dict.set_item("stream", stream_u64)?; // non-zero CUstream per CAI v3

    Ok((DeviceArrayF32Py { inner }, dict))
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "linreg_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, period, device_id=0))]
pub fn linreg_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    data_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    period: usize,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    use numpy::PyUntypedArrayMethods;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let flat_in = data_tm_f32.as_slice()?;
    let rows = data_tm_f32.shape()[0];
    let cols = data_tm_f32.shape()[1];
    let params = LinRegParams {
        period: Some(period),
    };

    let inner = py.allow_threads(|| {
        let cuda = CudaLinreg::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.linreg_multi_series_one_param_time_major_dev(flat_in, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(DeviceArrayF32Py { inner })
}

#[cfg(feature = "python")]
#[pyclass(name = "LinRegStream")]
pub struct LinRegStreamPy {
    inner: LinRegStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl LinRegStreamPy {
    #[new]
    pub fn new(period: usize) -> PyResult<Self> {
        let params = LinRegParams {
            period: Some(period),
        };
        match LinRegStream::try_new(params) {
            Ok(stream) => Ok(Self { inner: stream }),
            Err(e) => Err(PyValueError::new_err(format!("LinRegStream error: {}", e))),
        }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.inner.update(value)
    }
}

/// Computes LinReg directly into a provided output slice, avoiding allocation.
/// The output slice must be the same length as the input data.
#[inline]
pub fn linreg_into_slice(
    dst: &mut [f64],
    input: &LinRegInput,
    kern: Kernel,
) -> Result<(), LinRegError> {
    let data: &[f64] = input.as_ref();

    // Verify output buffer size matches input
    if dst.len() != data.len() {
        return Err(LinRegError::OutputLengthMismatch { expected: data.len(), got: dst.len() });
    }

    // Use the existing compute_into function
    linreg_compute_into(input, kern, dst)
}

// WASM bindings
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn linreg_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = LinRegParams {
        period: Some(period),
    };
    let input = LinRegInput::from_slice(data, params);

    // Allocate output buffer once
    let mut output = vec![0.0; data.len()];

    // Compute directly into output buffer
    linreg_into_slice(&mut output, &input, Kernel::Scalar)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct LinRegBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = linreg_batch)]
pub fn linreg_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: LinRegBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = LinRegBatchRange {
        period: config.period_range,
    };

    // For WASM, we need to use Kernel::Scalar instead of Auto
    // since WASM doesn't support AVX instructions
    let output = linreg_batch_slice(data, &sweep, Kernel::Scalar)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    serde_wasm_bindgen::to_value(&output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

// ================== Zero-Copy WASM Functions ==================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn linreg_alloc(len: usize) -> *mut f64 {
    // Allocate memory for input/output buffer
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec); // Prevent deallocation
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn linreg_free(ptr: *mut f64, len: usize) {
    // Free allocated memory
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn linreg_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    // Check for null pointers
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to linreg_into"));
    }

    unsafe {
        // Create slice from pointer
        let data = std::slice::from_raw_parts(in_ptr, len);

        // Validate inputs
        if period == 0 || period > len {
            return Err(JsValue::from_str("Invalid period"));
        }

        // Calculate LinReg
        let params = LinRegParams {
            period: Some(period),
        };
        let input = LinRegInput::from_slice(data, params);

        // Check for aliasing (input and output buffers are the same)
        if in_ptr == out_ptr {
            // Use temporary buffer to avoid corruption during sliding window computation
            let mut temp = vec![0.0; len];
            linreg_into_slice(&mut temp, &input, Kernel::Scalar)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            // Copy results back to output
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            // No aliasing, compute directly into output
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            linreg_into_slice(out, &input, Kernel::Scalar)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

// ================== Optimized Batch Processing ==================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn linreg_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to linreg_batch_into",
        ));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        let sweep = LinRegBatchRange {
            period: (period_start, period_end, period_step),
        };

        let combos = expand_grid_linreg(&sweep);
        let rows = combos.len();
        let cols = len;

        let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);

        // Use optimized batch processing with Scalar kernel for WASM
        linreg_batch_inner_into(data, &sweep, Kernel::Scalar, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}
