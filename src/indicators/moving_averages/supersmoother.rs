//! # Super Smoother Filter
//!
//! A double-pole smoothing filter developed by John Ehlers that reduces high-frequency noise
//! while preserving significant trend information. Uses recursive calculations with coefficients
//! derived from the period parameter.
//!
//! ## Implementation Notes
//! Row kernel functions (`supersmoother_row_*`) perform pure computation only - they do not write
//! warmup NaN values. Warmup handling is the responsibility of the caller via helper functions
//! like `alloc_with_nan_prefix` and `init_matrix_prefixes`, or by setting NaNs post-computation.
//!
//! ## Parameters
//! - **period**: Main lookback length (defaults to 14). Must be ≥ 1 and ≤ the data length.
//!
//! ## Returns
//! - `Ok(SuperSmootherOutput)` on success (`values: Vec<f64>` matching input).
//! - `Err(SuperSmootherError)` on validation or computation errors.
//!
//! ## Developer Notes
//! - **AVX2 kernel**: ❌ Stub only - falls back to scalar implementation
//! - **AVX512 kernel**: ❌ Stub only - has short/long path structure but both fall back to scalar
//! - **Decision**: Runtime selection short-circuits to Scalar by default. The IIR dependency chain prevents
//!   effective across-time SIMD; AVX2/AVX512 paths do not beat the optimized scalar by >5% in benches.
//! - **Streaming update**: ✅ O(1) complexity using minimal state (x_prev, y1, y2)
//!   with correct feedback terms (uses previous outputs, not inputs) per Ehlers.
//! - **Memory optimization**: ✅ Uses zero-copy helpers (alloc_with_nan_prefix, make_uninit_matrix) for output vectors
//! - **Row-specific batch**: ❌ Not attempted. Potential future win by precomputing s[i] = x[i] + x[i-1]
//!   once and reusing across rows; current code keeps per-row scalar kernel for simplicity.
//! - **Note**: SIMD acceleration currently stubbed to scalar for API parity

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
    cuda::moving_averages::CudaSuperSmoother, indicators::moving_averages::alma::DeviceArrayF32Py,
};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use std::f64::consts::PI;
use std::mem::MaybeUninit;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum SuperSmootherData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

impl<'a> AsRef<[f64]> for SuperSmootherInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            SuperSmootherData::Slice(sl) => sl,
            SuperSmootherData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SuperSmootherOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct SuperSmootherParams {
    pub period: Option<usize>,
}

impl Default for SuperSmootherParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct SuperSmootherInput<'a> {
    pub data: SuperSmootherData<'a>,
    pub params: SuperSmootherParams,
}

impl<'a> SuperSmootherInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: SuperSmootherParams) -> Self {
        Self {
            data: SuperSmootherData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: SuperSmootherParams) -> Self {
        Self {
            data: SuperSmootherData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", SuperSmootherParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SuperSmootherBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for SuperSmootherBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl SuperSmootherBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<SuperSmootherOutput, SuperSmootherError> {
        let p = SuperSmootherParams {
            period: self.period,
        };
        let i = SuperSmootherInput::from_candles(c, "close", p);
        supersmoother_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<SuperSmootherOutput, SuperSmootherError> {
        let p = SuperSmootherParams {
            period: self.period,
        };
        let i = SuperSmootherInput::from_slice(d, p);
        supersmoother_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<SuperSmootherStream, SuperSmootherError> {
        let p = SuperSmootherParams {
            period: self.period,
        };
        SuperSmootherStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum SuperSmootherError {
    #[error("supersmoother: All values are NaN.")]
    AllValuesNaN,
    #[error("supersmoother: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("supersmoother: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("supersmoother: Empty data provided.")]
    EmptyData,
}

#[inline]
pub fn supersmoother(
    input: &SuperSmootherInput,
) -> Result<SuperSmootherOutput, SuperSmootherError> {
    supersmoother_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
pub fn supersmoother_with_kernel(
    input: &SuperSmootherInput,
    kernel: Kernel,
) -> Result<SuperSmootherOutput, SuperSmootherError> {
    // ---------- 0. validation ----------
    let data: &[f64] = input.as_ref();
    if data.is_empty() {
        return Err(SuperSmootherError::EmptyData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(SuperSmootherError::AllValuesNaN)?;

    let len = data.len();
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(SuperSmootherError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(SuperSmootherError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    // ---------- 1. prepare the output buffer ----------
    //   All indices 0‥warm-1 are guaranteed NaN so the stream version lines up.
    let warm = first + period - 1;
    let mut out = alloc_with_nan_prefix(len, warm);

    // ---------- 2. choose kernel ----------
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    // ---------- 3. do the work ----------
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                supersmoother_row_scalar(data, first, period, &mut out);
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                supersmoother_row_avx2(data, first, period, &mut out);
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                supersmoother_row_avx512(data, first, period, &mut out);
            }
            _ => unreachable!(),
        }
    }

    // ---------- 4. package and return ----------
    Ok(SuperSmootherOutput { values: out })
}

/// Write SuperSmoother values directly to output slice - no allocations
pub fn supersmoother_into_slice(
    dst: &mut [f64],
    input: &SuperSmootherInput,
    kernel: Kernel,
) -> Result<(), SuperSmootherError> {
    let data: &[f64] = input.as_ref();
    if data.is_empty() {
        return Err(SuperSmootherError::EmptyData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(SuperSmootherError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();
    if period == 0 || period > len {
        return Err(SuperSmootherError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(SuperSmootherError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }
    if dst.len() != len {
        return Err(SuperSmootherError::InvalidPeriod {
            period: dst.len(),
            data_len: len,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                supersmoother_row_scalar(data, first, period, dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => supersmoother_row_avx2(data, first, period, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                supersmoother_row_avx512(data, first, period, dst)
            }
            _ => unreachable!(),
        }
    }

    // Set warmup NaNs post-compute, like ALMA.into_slice
    let warm = first + period - 1;
    for v in &mut dst[..warm] {
        *v = f64::NAN;
    }
    Ok(())
}

#[inline]
pub unsafe fn supersmoother_scalar(
    data: &[f64],
    period: usize,
    first: usize,
) -> Result<SuperSmootherOutput, SuperSmootherError> {
    let len = data.len();
    if len == 0 {
        return Err(SuperSmootherError::EmptyData);
    }
    if period == 0 {
        return Err(SuperSmootherError::InvalidPeriod {
            period,
            data_len: len,
        });
    }

    let warm = first + period - 1;
    if warm >= len {
        return Err(SuperSmootherError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    // Allocate once and let the row kernel write from warm onward.
    let mut out = alloc_with_nan_prefix(len, warm);
    supersmoother_row_scalar(data, first, period, &mut out);
    Ok(SuperSmootherOutput { values: out })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn supersmoother_avx2(
    data: &[f64],
    period: usize,
    first: usize,
) -> Result<SuperSmootherOutput, SuperSmootherError> {
    supersmoother_scalar(data, period, first)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn supersmoother_avx512(
    data: &[f64],
    period: usize,
    first: usize,
) -> Result<SuperSmootherOutput, SuperSmootherError> {
    if period <= 32 {
        supersmoother_avx512_short(data, period, first)
    } else {
        supersmoother_avx512_long(data, period, first)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn supersmoother_avx512_short(
    data: &[f64],
    period: usize,
    first: usize,
) -> Result<SuperSmootherOutput, SuperSmootherError> {
    supersmoother_scalar(data, period, first)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn supersmoother_avx512_long(
    data: &[f64],
    period: usize,
    first: usize,
) -> Result<SuperSmootherOutput, SuperSmootherError> {
    supersmoother_scalar(data, period, first)
}

#[derive(Debug, Clone)]
pub struct SuperSmootherStream {
    period: usize,
    // Coefficients
    a: f64,
    a_sq: f64,
    b: f64,
    c: f64,
    // Minimal state for O(1) update
    x_prev: f64, // x[t-1]
    y1: f64,     // y[t-1]
    y2: f64,     // y[t-2]
    seen: u8,    // 0 = no samples, 1 = have x[t-1], >=2 = fully primed
}

impl SuperSmootherStream {
    #[inline]
    pub fn try_new(params: SuperSmootherParams) -> Result<Self, SuperSmootherError> {
        let period = params.period.unwrap_or(14);
        if period == 0 {
            return Err(SuperSmootherError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }

        // Compute coefficients once (no divides in the hot path)
        let inv_p = 1.0 / (period as f64);
        let f = std::f64::consts::SQRT_2 * PI * inv_p;
        let a = (-f).exp();
        let a_sq = a * a;
        let b = 2.0 * a * f.cos();
        let c = 0.5 * (1.0 + a_sq - b);

        Ok(Self {
            period,
            a,
            a_sq,
            b,
            c,
            x_prev: 0.0,
            y1: 0.0,
            y2: 0.0,
            seen: 0,
        })
    }

    /// One-sample O(1) update.
    ///
    /// Returns:
    /// - `None` until we have seen at least two samples (initial conditions).
    /// - `Some(y_t)` thereafter. The first `Some` returned equals the second sample (Ehlers' initial condition).
    ///
    /// Notes:
    /// - `prev` (when provided) must be the previous outputs `(y_{t-1}, y_{t-2})`. If `None`, the internal state is used.
    /// - This streaming API does not inject period-length warmup NaNs; callers align outputs as needed.
    #[inline(always)]
    pub fn update(&mut self, x_t: f64, prev: Option<(f64, f64)>) -> Option<f64> {
        // Ignore non-finite samples in live streams; keep state unchanged
        if !x_t.is_finite() {
            return None;
        }

        match self.seen {
            0 => {
                // First sample: remember as x[t-1]; cannot produce output yet
                self.x_prev = x_t;
                self.seen = 1;
                None
            }
            1 => {
                // Second sample: Ehlers initial condition -> y[t] = x[t]
                // Also set y[t-2] = x[t-1] so that the next step has two outputs
                let y = x_t;
                self.y2 = self.x_prev; // y_{t-2} := x_{t-1}
                self.y1 = y; // y_{t-1} := x_t
                self.x_prev = x_t; // x_{t-1} := x_t
                self.seen = 2;
                Some(y)
            }
            _ => {
                // Fully primed: use external or internal (y_{t-1}, y_{t-2})
                let (mut y_im1, mut y_im2) = (self.y1, self.y2);
                if let Some((p1, p2)) = prev {
                    y_im1 = p1;
                    y_im2 = p2;
                }

                // y_t = c*(x_t + x_{t-1}) + b*y_{t-1} - a^2*y_{t-2}
                let t = f64::mul_add(self.b, y_im1, -self.a_sq * y_im2);
                let y = f64::mul_add(self.c, x_t + self.x_prev, t);

                // Roll state
                self.y2 = y_im1;
                self.y1 = y;
                self.x_prev = x_t;

                Some(y)
            }
        }
    }

    /// Reconfigure the period and reset state.
    #[inline]
    pub fn reconfigure(&mut self, period: usize) -> Result<(), SuperSmootherError> {
        if period == 0 {
            return Err(SuperSmootherError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        self.period = period;

        let inv_p = 1.0 / (period as f64);
        let f = std::f64::consts::SQRT_2 * PI * inv_p;
        let a = (-f).exp();
        self.a = a;
        self.a_sq = a * a;
        self.b = 2.0 * a * f.cos();
        self.c = 0.5 * (1.0 + self.a_sq - self.b);

        self.x_prev = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
        self.seen = 0;
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct SuperSmootherBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for SuperSmootherBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 100, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SuperSmootherBatchBuilder {
    range: SuperSmootherBatchRange,
    kernel: Kernel,
}

impl SuperSmootherBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<SuperSmootherBatchOutput, SuperSmootherError> {
        supersmoother_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(
        data: &[f64],
        k: Kernel,
    ) -> Result<SuperSmootherBatchOutput, SuperSmootherError> {
        SuperSmootherBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(
        self,
        c: &Candles,
        src: &str,
    ) -> Result<SuperSmootherBatchOutput, SuperSmootherError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(
        c: &Candles,
    ) -> Result<SuperSmootherBatchOutput, SuperSmootherError> {
        SuperSmootherBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn supersmoother_batch_with_kernel(
    data: &[f64],
    sweep: &SuperSmootherBatchRange,
    k: Kernel,
) -> Result<SuperSmootherBatchOutput, SuperSmootherError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(SuperSmootherError::InvalidPeriod {
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
    supersmoother_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct SuperSmootherBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<SuperSmootherParams>,
    pub rows: usize,
    pub cols: usize,
}
impl SuperSmootherBatchOutput {
    pub fn row_for_params(&self, p: &SuperSmootherParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
    }
    pub fn values_for(&self, p: &SuperSmootherParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
pub fn expand_grid_supersmoother(r: &SuperSmootherBatchRange) -> Vec<SuperSmootherParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(SuperSmootherParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn supersmoother_batch_slice(
    data: &[f64],
    sweep: &SuperSmootherBatchRange,
    kern: Kernel,
) -> Result<SuperSmootherBatchOutput, SuperSmootherError> {
    supersmoother_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn supersmoother_batch_par_slice(
    data: &[f64],
    sweep: &SuperSmootherBatchRange,
    kern: Kernel,
) -> Result<SuperSmootherBatchOutput, SuperSmootherError> {
    supersmoother_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn supersmoother_batch_inner(
    data: &[f64],
    sweep: &SuperSmootherBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<SuperSmootherBatchOutput, SuperSmootherError> {
    let combos = expand_grid_supersmoother(sweep);
    if combos.is_empty() {
        return Err(SuperSmootherError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(SuperSmootherError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(SuperSmootherError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();

    let mut buf_mu = make_uninit_matrix(rows, cols);
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out_f64: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    supersmoother_batch_inner_into(data, sweep, kern, parallel, out_f64)?;

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(SuperSmootherBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub fn supersmoother_batch_inner_into(
    data: &[f64],
    sweep: &SuperSmootherBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<SuperSmootherParams>, SuperSmootherError> {
    let combos = expand_grid_supersmoother(sweep);
    if combos.is_empty() {
        return Err(SuperSmootherError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(SuperSmootherError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(SuperSmootherError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    debug_assert_eq!(out.len(), rows * cols, "out buffer must be rows*cols");

    // 1) Treat caller's buffer as MaybeUninit and initialize NaN prefixes in one pass
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();

    // Cast the output buffer to MaybeUninit and create a Vec from it temporarily
    let mut out_vec = unsafe {
        Vec::from_raw_parts(
            out.as_mut_ptr() as *mut std::mem::MaybeUninit<f64>,
            out.len(),
            out.len(),
        )
    };
    init_matrix_prefixes(&mut out_vec, cols, &warm);
    // Forget the Vec to avoid double-free
    std::mem::forget(out_vec);

    let out_mu = unsafe {
        core::slice::from_raw_parts_mut(
            out.as_mut_ptr() as *mut std::mem::MaybeUninit<f64>,
            out.len(),
        )
    };

    // 2) Per-row compute into the same memory
    let do_row = |row: usize, dst_mu: &mut [std::mem::MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();
        let out_row =
            core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());
        match kern {
            Kernel::Scalar => supersmoother_row_scalar(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => supersmoother_row_avx2(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => supersmoother_row_avx512(data, first, period, out_row),
            _ => unreachable!(),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            out_mu
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(r, s)| do_row(r, s));
        }
        #[cfg(target_arch = "wasm32")]
        for (r, s) in out_mu.chunks_mut(cols).enumerate() {
            do_row(r, s);
        }
    } else {
        for (r, s) in out_mu.chunks_mut(cols).enumerate() {
            do_row(r, s);
        }
    }

    Ok(combos)
}

/// Row kernel for scalar SuperSmoother computation.
/// This function performs pure computation only - it does NOT write warmup NaN values.
/// The caller is responsible for handling warmup via `alloc_with_nan_prefix` or `init_matrix_prefixes`.
#[inline(always)]
pub unsafe fn supersmoother_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    let len = data.len();
    let warm = first + period - 1;

    // Row kernels do not touch 0..warm; caller handles warmup.
    if len <= warm {
        return;
    }

    // Coefficients (Ehlers 2‑pole SuperSmoother)
    let f = 1.414_f64 * PI / (period as f64);
    let a = (-f).exp();
    let a_sq = a * a;
    let b = 2.0 * a * f.cos();
    let c = 0.5 * (1.0 + a_sq - b);

    // Use raw pointers to avoid bounds checks in the hot loop
    let x_ptr = data.as_ptr();
    let y_ptr = out.as_mut_ptr();

    // Initial conditions at warm and warm+1 (if present)
    *y_ptr.add(warm) = *x_ptr.add(warm);
    if len == warm + 1 {
        return;
    }
    *y_ptr.add(warm + 1) = *x_ptr.add(warm + 1);
    if len == warm + 2 {
        return;
    }

    // Carry state in registers: y[i-2], y[i-1], x[i-1]
    let mut y_im2 = *y_ptr.add(warm);
    let mut y_im1 = *y_ptr.add(warm + 1);
    let mut x_prev = *x_ptr.add(warm + 1);

    // Main recurrence (2x unrolled)
    let mut i = warm + 2;
    let end_even = warm + 2 + ((len - (warm + 2)) & !1);

    while i < end_even {
        // i
        let x_i = *x_ptr.add(i);
        let s0 = f64::mul_add(b, y_im1, -a_sq * y_im2);
        let y0 = f64::mul_add(c, x_i + x_prev, s0);
        *y_ptr.add(i) = y0;

        // i+1
        let x_ip1 = *x_ptr.add(i + 1);
        let s1 = f64::mul_add(b, y0, -a_sq * y_im1);
        let y1 = f64::mul_add(c, x_ip1 + x_i, s1);
        *y_ptr.add(i + 1) = y1;

        // roll
        y_im2 = y0;
        y_im1 = y1;
        x_prev = x_ip1;
        i += 2;
    }

    // Tail
    if i < len {
        let x_i = *x_ptr.add(i);
        let s0 = f64::mul_add(b, y_im1, -a_sq * y_im2);
        let y0 = f64::mul_add(c, x_i + x_prev, s0);
        *y_ptr.add(i) = y0;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn supersmoother_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    supersmoother_row_scalar(data, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn supersmoother_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    if period <= 32 {
        supersmoother_row_avx512_short(data, first, period, out)
    } else {
        supersmoother_row_avx512_long(data, first, period, out)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
pub unsafe fn supersmoother_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    supersmoother_row_scalar(data, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
pub unsafe fn supersmoother_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    supersmoother_row_scalar(data, first, period, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_supersmoother_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = SuperSmootherParams { period: None };
        let input = SuperSmootherInput::from_candles(&candles, "close", default_params);
        let output = supersmoother_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }
    fn check_supersmoother_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = SuperSmootherParams { period: Some(14) };
        let input = SuperSmootherInput::from_candles(&candles, "close", params);
        let result = supersmoother_with_kernel(&input, kernel)?;
        let out_vals = &result.values;
        let expected_last_five = [
            59140.98229179739,
            59172.03593376982,
            59179.40342783722,
            59171.22758152845,
            59127.859841077094,
        ];
        let start_idx = out_vals.len() - 5;
        for (i, &val) in out_vals[start_idx..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-8,
                "[{}] mismatch at idx {}: got {}, expected {}",
                test_name,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }
    fn check_supersmoother_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = SuperSmootherInput::with_default_candles(&candles);
        match input.data {
            SuperSmootherData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected SuperSmootherData::Candles"),
        }
        let output = supersmoother_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }
    fn check_supersmoother_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = SuperSmootherParams { period: Some(0) };
        let input = SuperSmootherInput::from_slice(&input_data, params);
        let res = supersmoother_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] should fail with zero period", test_name);
        Ok(())
    }
    fn check_supersmoother_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = SuperSmootherParams { period: Some(10) };
        let input = SuperSmootherInput::from_slice(&data_small, params);
        let res = supersmoother_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] should fail with period exceeding length",
            test_name
        );
        Ok(())
    }
    fn check_supersmoother_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = SuperSmootherParams { period: Some(14) };
        let input = SuperSmootherInput::from_slice(&single_point, params);
        let res = supersmoother_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] should fail with insufficient data",
            test_name
        );
        Ok(())
    }
    fn check_supersmoother_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = SuperSmootherParams { period: Some(14) };
        let first_input = SuperSmootherInput::from_candles(&candles, "close", first_params);
        let first_result = supersmoother_with_kernel(&first_input, kernel)?;
        let second_params = SuperSmootherParams { period: Some(10) };
        let second_input = SuperSmootherInput::from_slice(&first_result.values, second_params);
        let second_result = supersmoother_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }
    fn check_supersmoother_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = SuperSmootherInput::from_candles(
            &candles,
            "close",
            SuperSmootherParams { period: Some(14) },
        );
        let res = supersmoother_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 240 {
            for (i, &val) in res.values[240..].iter().enumerate() {
                assert!(
                    val.is_finite(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test_name,
                    240 + i
                );
            }
        }
        Ok(())
    }
    macro_rules! generate_all_supersmoother_tests {
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
    fn check_supersmoother_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test multiple parameter combinations to catch uninitialized memory reads
        let test_periods = vec![3, 7, 10, 14, 20, 30, 50, 100, 200];

        for period in test_periods {
            let params = SuperSmootherParams {
                period: Some(period),
            };
            let input = SuperSmootherInput::from_candles(&candles, "close", params);

            // Skip if period is too large for the data
            if period > candles.close.len() {
                continue;
            }

            let output = supersmoother_with_kernel(&input, kernel)?;

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
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} with period {}",
						test_name, val, bits, i, period
					);
                }

                // Check for init_matrix_prefixes poison (0x22222222_22222222)
                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} with period {}",
						test_name, val, bits, i, period
					);
                }

                // Check for make_uninit_matrix poison (0x33333333_33333333)
                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} with period {}",
						test_name, val, bits, i, period
					);
                }
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_supersmoother_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_supersmoother_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Test strategy: generate period first, then data of appropriate length
        let strat = (1usize..=100).prop_flat_map(|period| {
            (
                prop::collection::vec(
                    (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                    period..400,
                ),
                Just(period),
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, period)| {
                let params = SuperSmootherParams {
                    period: Some(period),
                };
                let input = SuperSmootherInput::from_slice(&data, params);

                // Run with test kernel and reference scalar kernel
                let SuperSmootherOutput { values: out } =
                    supersmoother_with_kernel(&input, kernel).unwrap();
                let SuperSmootherOutput { values: ref_out } =
                    supersmoother_with_kernel(&input, Kernel::Scalar).unwrap();

                // Find first non-NaN index
                let first = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
                let warmup = first + period - 1;

                // Property 1: Warmup period values should be NaN
                for i in 0..warmup.min(out.len()) {
                    prop_assert!(out[i].is_nan(), "Expected NaN during warmup at index {}", i);
                }

                // Property 2: Initial conditions - first two values after warmup should match input
                if warmup < data.len() {
                    let tolerance = if period == 1 { 1e-8 } else { 1e-9 };
                    prop_assert!(
                        (out[warmup] - data[warmup]).abs() <= tolerance,
                        "Initial condition 1 failed at index {}: {} vs {}",
                        warmup,
                        out[warmup],
                        data[warmup]
                    );
                }
                if warmup + 1 < data.len() {
                    let tolerance = if period == 1 { 1e-8 } else { 1e-9 };
                    prop_assert!(
                        (out[warmup + 1] - data[warmup + 1]).abs() <= tolerance,
                        "Initial condition 2 failed at index {}: {} vs {}",
                        warmup + 1,
                        out[warmup + 1],
                        data[warmup + 1]
                    );
                }

                // Property 3: Constant input property
                if data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10) && !data.is_empty() {
                    // For constant input, output should converge to that constant
                    let constant_val = data[first];
                    for i in (warmup + 10).min(out.len() - 1)..out.len() {
                        let tolerance = if period == 1 { 1e-8 } else { 1e-6 };
                        prop_assert!(
                            (out[i] - constant_val).abs() <= tolerance,
                            "Constant input property failed at index {}: {} vs {}",
                            i,
                            out[i],
                            constant_val
                        );
                    }
                }

                // Property 4: Cross-kernel consistency
                for i in warmup..out.len() {
                    let y = out[i];
                    let r = ref_out[i];

                    if !y.is_finite() || !r.is_finite() {
                        prop_assert!(
                            y.to_bits() == r.to_bits(),
                            "NaN/infinite mismatch at index {}: {} vs {}",
                            i,
                            y,
                            r
                        );
                    } else {
                        // Allow slightly more tolerance for period=1 due to numerical precision
                        let tolerance = if period == 1 { 1e-8 } else { 1e-9 };
                        let ulp_diff = y.to_bits().abs_diff(r.to_bits());

                        prop_assert!(
                            (y - r).abs() <= tolerance || ulp_diff <= 8,
                            "Cross-kernel mismatch at index {}: {} vs {} (diff={}, ULP={})",
                            i,
                            y,
                            r,
                            (y - r).abs(),
                            ulp_diff
                        );
                    }
                }

                // Property 5: Filter stability - output should not blow up
                for i in warmup..out.len() {
                    prop_assert!(
                        out[i].is_nan() || out[i].abs() < 1e12,
                        "Filter instability detected at index {}: {}",
                        i,
                        out[i]
                    );
                }

                // Property 6: Reasonable bounds check - only for non-sparse data
                // IIR filters can have unbounded overshoot with sparse/impulsive data
                if warmup < out.len() {
                    // Check for sparse data (lots of zeros or near-zeros)
                    let zero_count = data.iter().filter(|&&x| x.abs() < 1e-10).count();
                    let sparsity_ratio = zero_count as f64 / data.len() as f64;

                    // Only check bounds for non-sparse data where behavior is more predictable
                    if sparsity_ratio < 0.3 {
                        let data_min = data[0..=out.len() - 1]
                            .iter()
                            .cloned()
                            .fold(f64::INFINITY, f64::min);
                        let data_max = data[0..=out.len() - 1]
                            .iter()
                            .cloned()
                            .fold(f64::NEG_INFINITY, f64::max);
                        let data_range = (data_max - data_min).abs();
                        let max_magnitude = data_max.abs().max(data_min.abs());

                        for i in warmup..out.len() {
                            if out[i].is_finite() {
                                // Allow up to 12x overshoot for IIR filters with normal data
                                // IIR filters can have significant overshoot even with normal data
                                let bound = 12.0 * max_magnitude + data_range + 1e-3;
                                prop_assert!(
                                    out[i].abs() <= bound,
                                    "Bounds check failed at index {}: |{}| > {}",
                                    i,
                                    out[i],
                                    bound
                                );
                            }
                        }
                    }
                }

                // Property 7: Recursive formula verification (for indices after initial conditions)
                if warmup + 2 < out.len() {
                    // Calculate filter coefficients
                    let a = (-1.414_f64 * std::f64::consts::PI / (period as f64)).exp();
                    let a_sq = a * a;
                    let b = 2.0 * a * (1.414_f64 * std::f64::consts::PI / (period as f64)).cos();
                    let c = (1.0 + a_sq - b) * 0.5;

                    // Verify recursive formula for a few samples
                    let test_start = warmup + 2;
                    let test_end = (test_start + 10).min(out.len());

                    for i in test_start..test_end {
                        let expected =
                            c * (data[i] + data[i - 1]) + b * out[i - 1] - a_sq * out[i - 2];
                        let tolerance = 1e-9 * (1.0 + expected.abs());
                        prop_assert!(
                            (out[i] - expected).abs() <= tolerance,
                            "Recursive formula failed at index {}: {} vs expected {}",
                            i,
                            out[i],
                            expected
                        );
                    }
                }

                // Property 8: Period=1 edge case with proper tolerance
                if period == 1 && warmup + 2 < out.len() {
                    // With period=1, the filter still applies its coefficients
                    // Check that output remains bounded relative to recent input
                    for i in warmup + 2..out.len() {
                        let recent_window = &data[i.saturating_sub(5)..=i];
                        let recent_min =
                            recent_window.iter().cloned().fold(f64::INFINITY, f64::min);
                        let recent_max = recent_window
                            .iter()
                            .cloned()
                            .fold(f64::NEG_INFINITY, f64::max);
                        let recent_range = (recent_max - recent_min).abs();

                        // Allow significant overshoot for period=1 due to filter characteristics
                        let tolerance =
                            recent_range + recent_max.abs().max(recent_min.abs()) + 1e-6;
                        prop_assert!(
                            out[i].abs() <= recent_max.abs() + tolerance,
                            "Period=1 bounds failed at index {}: |{}| exceeds reasonable bounds",
                            i,
                            out[i]
                        );
                    }
                }

                // Property 9: Monotonic response test
                if warmup + 10 < out.len() {
                    // Check if input is monotonically increasing or decreasing in a window
                    let test_window_start = warmup + 2;
                    let test_window_end = (test_window_start + 20).min(out.len() - 1);

                    if test_window_end > test_window_start {
                        let input_window = &data[test_window_start..test_window_end];
                        let is_monotonic_inc =
                            input_window.windows(2).all(|w| w[1] >= w[0] - 1e-10);
                        let is_monotonic_dec =
                            input_window.windows(2).all(|w| w[1] <= w[0] + 1e-10);

                        if is_monotonic_inc || is_monotonic_dec {
                            // Output should generally follow the trend (with some lag and smoothing)
                            let output_window = &out[test_window_start..test_window_end];
                            let input_trend = data[test_window_end - 1] - data[test_window_start];
                            let output_trend = out[test_window_end - 1] - out[test_window_start];

                            // Check that trends have the same sign (allowing for small differences)
                            if input_trend.abs() > 1e-6 {
                                prop_assert!(
                                    input_trend.signum() == output_trend.signum()
                                        || output_trend.abs() < 1e-6,
                                    "Monotonic response failed: input trend {} but output trend {}",
                                    input_trend,
                                    output_trend
                                );
                            }
                        }
                    }
                }

                // Property 10: Poison value detection
                for (i, &val) in out.iter().enumerate() {
                    if val.is_finite() {
                        let bits = val.to_bits();
                        prop_assert!(
                            bits != 0x11111111_11111111
                                && bits != 0x22222222_22222222
                                && bits != 0x33333333_33333333,
                            "Poison value detected at index {}: {} (0x{:016X})",
                            i,
                            val,
                            bits
                        );
                    }
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    #[cfg(feature = "proptest")]
    generate_all_supersmoother_tests!(check_supersmoother_property);

    generate_all_supersmoother_tests!(
        check_supersmoother_partial_params,
        check_supersmoother_accuracy,
        check_supersmoother_default_candles,
        check_supersmoother_zero_period,
        check_supersmoother_period_exceeds_length,
        check_supersmoother_very_small_dataset,
        check_supersmoother_reinput,
        check_supersmoother_nan_handling,
        check_supersmoother_no_poison
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = SuperSmootherBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = SuperSmootherParams::default();
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

        // Test multiple different batch configurations to catch edge cases
        let batch_configs = vec![
            (3, 10, 2),    // Small periods with small step
            (10, 30, 10),  // Medium periods
            (20, 100, 20), // Large periods
            (5, 5, 1),     // Single period (edge case)
            (2, 50, 1),    // Many periods starting from minimum
        ];

        for (start, end, step) in batch_configs {
            // Skip if the largest period exceeds data length
            if end > c.close.len() {
                continue;
            }

            let output = SuperSmootherBatchBuilder::new()
                .kernel(kernel)
                .period_range(start, end, step)
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
                let period = if row < output.combos.len() {
                    output.combos[row].period.unwrap_or(0)
                } else {
                    0
                };

                // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with period {} in batch ({}, {}, {})",
                        test, val, bits, row, col, idx, period, start, end, step
                    );
                }

                // Check for init_matrix_prefixes poison (0x22222222_22222222)
                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {}) with period {} in batch ({}, {}, {})",
                        test, val, bits, row, col, idx, period, start, end, step
                    );
                }

                // Check for make_uninit_matrix poison (0x33333333_33333333)
                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with period {} in batch ({}, {}, {})",
                        test, val, bits, row, col, idx, period, start, end, step
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

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "python")]
#[pyfunction(name = "supersmoother")]
#[pyo3(signature = (data, period, kernel=None))]
/// Compute the SuperSmoother filter (2-pole) of the input data.
///
/// SuperSmoother is a double-pole smoothing filter that reduces high-frequency noise
/// while preserving trend information. It provides better smoothing than EMA with
/// similar lag characteristics.
///
/// Parameters:
/// -----------
/// data : np.ndarray
///     Input data array (float64).
/// period : int
///     Smoothing period, must be >= 1 and <= data length.
/// kernel : str, optional
///     Computation kernel to use: 'auto', 'scalar', 'avx2', 'avx512'.
///     Default is 'auto' which auto-detects the best available.
///
/// Returns:
/// --------
/// np.ndarray
///     Array of SuperSmoother values, same length as input.
///
/// Raises:
/// -------
/// ValueError
///     If inputs are invalid (period = 0, exceeds data length, all NaN, etc).
pub fn supersmoother_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?; // false for single operations

    let params = SuperSmootherParams {
        period: Some(period),
    };
    let ss_in = SuperSmootherInput::from_slice(slice_in, params);

    // Get Vec<f64> from Rust function and zero-copy transfer to NumPy
    let result_vec: Vec<f64> = py
        .allow_threads(|| supersmoother_with_kernel(&ss_in, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "SuperSmootherStream")]
pub struct SuperSmootherStreamPy {
    stream: SuperSmootherStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl SuperSmootherStreamPy {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        let params = SuperSmootherParams {
            period: Some(period),
        };
        let stream = SuperSmootherStream::try_new(params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(SuperSmootherStreamPy { stream })
    }

    /// Updates the stream with a new value and returns the calculated SuperSmoother value.
    /// Returns `None` if the buffer is not yet full.
    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value, None)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "supersmoother_batch")]
#[pyo3(signature = (data, period_start, period_end, period_step, kernel=None))]
/// Compute SuperSmoother for multiple periods in a single pass.
///
/// Parameters:
/// -----------
/// data : np.ndarray
///     Input data array (float64).
/// period_start : int
///     Starting period value.
/// period_end : int
///     Ending period value (inclusive).
/// period_step : int
///     Step size between periods.
/// kernel : str, optional
///     Computation kernel: 'auto', 'scalar', 'avx2', 'avx512'.
///     Default is 'auto' which auto-detects the best available.
///
/// Returns:
/// --------
/// dict
///     Dictionary with 'values' (2D array, rows=periods, cols=data length)
///     and 'periods' array.
pub fn supersmoother_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period_start: usize,
    period_end: usize,
    period_step: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;

    let sweep = SuperSmootherBatchRange {
        period: (period_start, period_end, period_step),
    };

    let combos = expand_grid_supersmoother(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();

    // Pre-allocate output array (correct for batch operations)
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    let kern = validate_kernel(kernel, true)?;

    // Compute without GIL, writing directly to the NumPy array
    let combos = py
        .allow_threads(|| {
            // Handle kernel selection for batch operations
            let kernel = match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                k => k,
            };

            // Map batch kernels to regular kernels
            let simd = match kernel {
                Kernel::Avx512Batch => Kernel::Avx512,
                Kernel::Avx2Batch => Kernel::Avx2,
                Kernel::ScalarBatch => Kernel::Scalar,
                _ => kernel,
            };

            supersmoother_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Build result dictionary
    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;

    // For single-parameter indicators like SuperSmoother
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
#[pyfunction(name = "supersmoother_cuda_batch_dev")]
#[pyo3(signature = (data_f32, period_range, device_id=0))]
pub fn supersmoother_cuda_batch_dev_py<'py>(
    py: Python<'py>,
    data_f32: numpy::PyReadonlyArray1<'py, f32>,
    period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<(DeviceArrayF32Py, Bound<'py, pyo3::types::PyDict>)> {
    use crate::cuda::cuda_available;
    use numpy::IntoPyArray;
    use pyo3::types::PyDict;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let slice_in = data_f32.as_slice()?;
    let sweep = SuperSmootherBatchRange {
        period: period_range,
    };

    let (inner, combos) = py.allow_threads(|| {
        let cuda =
            CudaSuperSmoother::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.supersmoother_batch_dev(slice_in, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    let dict = PyDict::new(py);
    let periods: Vec<u64> = combos.iter().map(|c| c.period.unwrap() as u64).collect();
    dict.set_item("periods", periods.into_pyarray(py))?;

    Ok((DeviceArrayF32Py { inner }, dict))
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "supersmoother_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, period, device_id=0))]
pub fn supersmoother_cuda_many_series_one_param_dev_py(
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
    let params = SuperSmootherParams {
        period: Some(period),
    };

    let inner = py.allow_threads(|| {
        let cuda =
            CudaSuperSmoother::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.supersmoother_multi_series_one_param_time_major_dev(flat_in, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(DeviceArrayF32Py { inner })
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct SuperSmootherBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct SuperSmootherBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<SuperSmootherParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn supersmoother_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = SuperSmootherParams {
        period: Some(period),
    };
    let input = SuperSmootherInput::from_slice(data, params);

    // Allocate output buffer once
    let mut output = vec![0.0; data.len()];

    // Compute directly into output buffer
    supersmoother_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = supersmoother_batch)]
pub fn supersmoother_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: SuperSmootherBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = SuperSmootherBatchRange {
        period: config.period_range,
    };

    // Use non-batch kernel here, like ALMA
    let output = supersmoother_batch_inner(data, &sweep, detect_best_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = SuperSmootherBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };
    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

// Keep the old function for backward compatibility but mark as deprecated
#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[deprecated(since = "1.0.0", note = "Use supersmoother_batch instead")]
pub fn supersmoother_batch_js(
    data: &[f64],
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = SuperSmootherBatchRange {
        period: (period_start, period_end, period_step),
    };

    // Use the existing batch function with parallel=false for WASM
    supersmoother_batch_inner(data, &sweep, Kernel::Scalar, false)
        .map(|output| output.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn supersmoother_batch_metadata_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = SuperSmootherBatchRange {
        period: (period_start, period_end, period_step),
    };

    let combos = expand_grid_supersmoother(&sweep);
    let metadata: Vec<f64> = combos
        .iter()
        .map(|combo| combo.period.unwrap() as f64)
        .collect();

    Ok(metadata)
}

// ================== Zero-Copy WASM Functions ==================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn supersmoother_alloc(len: usize) -> *mut f64 {
    // Allocate memory for input/output buffer
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec); // Prevent deallocation
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn supersmoother_free(ptr: *mut f64, len: usize) {
    // Free allocated memory
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn supersmoother_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    // Check for null pointers
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        // Create slice from pointer
        let data = std::slice::from_raw_parts(in_ptr, len);

        // Validate inputs
        if period == 0 || period > len {
            return Err(JsValue::from_str("Invalid period"));
        }

        // Create input
        let params = SuperSmootherParams {
            period: Some(period),
        };
        let input = SuperSmootherInput::from_slice(data, params);

        if in_ptr == out_ptr as *const f64 {
            // CRITICAL: Aliasing check - in-place operation
            let mut temp = vec![0.0; len];
            supersmoother_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            // Direct write to output buffer
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            supersmoother_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn supersmoother_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to supersmoother_batch_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let sweep = SuperSmootherBatchRange {
            period: (period_start, period_end, period_step),
        };
        let combos = expand_grid_supersmoother(&sweep);
        let rows = combos.len();
        let out_slice = std::slice::from_raw_parts_mut(out_ptr, rows * len);

        // Map batch→simd
        let simd = match detect_best_batch_kernel() {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            Kernel::ScalarBatch => Kernel::Scalar,
            _ => Kernel::Scalar,
        };

        supersmoother_batch_inner_into(data, &sweep, simd, false, out_slice)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}
