//! # Polarized Fractal Efficiency (PFE)
//!
//! Measures the efficiency of price movement over a period, producing signed values
//! (positive = upward efficiency, negative = downward), then smooths with EMA.
//!
//! ## Parameters
//! - **period**: Lookback window (default: 10)
//! - **smoothing**: EMA smoothing window (default: 5)
//!
//! ## Inputs
//! - **data**: Price data or any numeric series
//!
//! ## Returns
//! - **values**: Vector of PFE values with NaN prefix during warmup period
//!
//! ## Developer Notes
//! - **Decision**: Single-series SIMD not enabled; time dependency makes it unhelpful beyond init. Batch path uses shared prefix sums and is fastest.
//! - **AVX2/AVX512 Kernels**: Implemented for row init; runtime selection remains but single-series uses the scalar core.
//! - **Streaming**: Implemented with O(1) update performance (maintains running sum)
//! - **Zero-copy Memory**: Uses alloc_with_nan_prefix and make_uninit_matrix for batch operations
//! - **Decision log**: SIMD disabled for single-series; CUDA batch/many-series wrappers enabled and matched to scalar within FP32 tolerance; Python interop exposes CUDA Array Interface v3 and DLPack v1.x without changing numerical outputs.
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

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
use std::error::Error;
use thiserror::Error;

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::pfe_wrapper::CudaPfe;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::DeviceArrayF32;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::context::Context;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::memory::DeviceBuffer;
#[cfg(all(feature = "python", feature = "cuda"))]
use std::sync::Arc;

impl<'a> AsRef<[f64]> for PfeInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            PfeData::Slice(slice) => slice,
            PfeData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum PfeData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct PfeOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(serde::Serialize, serde::Deserialize))]
pub struct PfeParams {
    pub period: Option<usize>,
    pub smoothing: Option<usize>,
}

impl Default for PfeParams {
    fn default() -> Self {
        Self {
            period: Some(10),
            smoothing: Some(5),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PfeInput<'a> {
    pub data: PfeData<'a>,
    pub params: PfeParams,
}

impl<'a> PfeInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: PfeParams) -> Self {
        Self {
            data: PfeData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: PfeParams) -> Self {
        Self {
            data: PfeData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", PfeParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(10)
    }
    #[inline]
    pub fn get_smoothing(&self) -> usize {
        self.params.smoothing.unwrap_or(5)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct PfeBuilder {
    period: Option<usize>,
    smoothing: Option<usize>,
    kernel: Kernel,
}

impl Default for PfeBuilder {
    fn default() -> Self {
        Self {
            period: None,
            smoothing: None,
            kernel: Kernel::Auto,
        }
    }
}

impl PfeBuilder {
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
    pub fn smoothing(mut self, s: usize) -> Self {
        self.smoothing = Some(s);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<PfeOutput, PfeError> {
        let p = PfeParams {
            period: self.period,
            smoothing: self.smoothing,
        };
        let i = PfeInput::from_candles(c, "close", p);
        pfe_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<PfeOutput, PfeError> {
        let p = PfeParams {
            period: self.period,
            smoothing: self.smoothing,
        };
        let i = PfeInput::from_slice(d, p);
        pfe_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<PfeStream, PfeError> {
        let p = PfeParams {
            period: self.period,
            smoothing: self.smoothing,
        };
        PfeStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum PfeError {
    #[error("pfe: Input data slice is empty.")]
    EmptyInputData,
    #[error("pfe: All values are NaN.")]
    AllValuesNaN,
    #[error("pfe: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("pfe: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("pfe: Invalid smoothing: {smoothing}")]
    InvalidSmoothing { smoothing: usize },
    #[error("pfe: Output length mismatch: expected {expected}, got {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("pfe: invalid range: start={start} end={end} step={step}")]
    InvalidRange { start: usize, end: usize, step: usize },
    #[error("pfe: invalid kernel for batch path: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("pfe: invalid input: {0}")]
    InvalidInput(&'static str),
}

#[inline(always)]
fn pfe_prepare<'a>(
    input: &'a PfeInput,
    k: Kernel,
) -> Result<(&'a [f64], usize, usize, usize, Kernel), PfeError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(PfeError::EmptyInputData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(PfeError::AllValuesNaN)?;
    let period = input.get_period();
    let smoothing = input.get_smoothing();

    if period == 0 || period > len {
        return Err(PfeError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    // Earliest valid output index for PFE is first + period (note: not -1).
    if len - first < period + 1 {
        return Err(PfeError::NotEnoughValidData {
            needed: period + 1,
            valid: len - first,
        });
    }
    if smoothing == 0 {
        return Err(PfeError::InvalidSmoothing { smoothing });
    }

    let chosen = match k {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    Ok((data, period, smoothing, first, chosen))
}

#[inline(always)]
fn pfe_compute_into(
    data: &[f64],
    period: usize,
    smoothing: usize,
    first: usize,
    _kernel: Kernel, // single-series SIMD not beneficial; keep scalar safe path
    out: &mut [f64],
) {
    let len = data.len();
    if len == 0 {
        return;
    }

    // Earliest computable index for PFE
    let start = first + period;
    if start >= len {
        return;
    }

    // Precompute constants
    let p = period as f64;
    let p2 = p * p;
    let alpha = 2.0 / (smoothing as f64 + 1.0);
    let one_minus_alpha = 1.0 - alpha;

    // Maintain a ring buffer of the last `period` short-leg segment lengths sqrt(1 + (ΔP)^2)
    let mut seg = vec![0.0f64; period];
    let mut head = 0usize; // points at oldest element
    let mut denom = 0.0f64;

    // Initialize denominator for t = start
    // steps from k = start - period + 1 ..= start, each step uses data[k] - data[k-1]
    let base = start - period + 1;
    for j in 0..period {
        let k = base + j;
        let d = data[k] - data[k - 1];
        let s = (d.mul_add(d, 1.0)).sqrt();
        seg[j] = s;
        denom += s;
    }

    // EMA state
    let mut ema_started = false;
    let mut ema_val = 0.0f64;

    // Rolling loop: O(n), 1 sqrt per step
    for t in start..len {
        let cur = data[t];
        let past = data[t - period];
        let diff = cur - past;

        // Long leg = sqrt(diff^2 + period^2)
        let long_leg = (diff.mul_add(diff, p2)).sqrt();

        // Compute raw, protect against vanishing denom
        let raw = if denom <= f64::EPSILON {
            0.0
        } else {
            100.0 * (long_leg / denom)
        };
        let signed = if diff > 0.0 { raw } else { -raw };

        // EMA smoothing with standard warmup seed
        let val = if !ema_started {
            ema_started = true;
            ema_val = signed;
            signed
        } else {
            ema_val = alpha.mul_add(signed, one_minus_alpha * ema_val);
            ema_val
        };

        out[t] = val;

        // Prepare denominator for next t (t+1)
        if t + 1 < len {
            let old = seg[head];
            let next = data[t + 1];
            let new_d = next - cur; // step between t and t+1
            let new_s = (new_d.mul_add(new_d, 1.0)).sqrt();
            denom += new_s - old;
            seg[head] = new_s;
            head += 1;
            if head == period {
                head = 0;
            }
        }
    }
}

#[inline]
pub fn pfe(input: &PfeInput) -> Result<PfeOutput, PfeError> {
    pfe_with_kernel(input, Kernel::Auto)
}

pub fn pfe_with_kernel(input: &PfeInput, kernel: Kernel) -> Result<PfeOutput, PfeError> {
    let (data, period, smoothing, first, chosen) = pfe_prepare(input, kernel)?;
    // Warmup count for PFE is first + period
    let mut out = alloc_with_nan_prefix(data.len(), first + period);
    pfe_compute_into(data, period, smoothing, first, chosen, &mut out);
    Ok(PfeOutput { values: out })
}

/// Write PFE values into a caller-provided buffer without allocating.
///
/// - Preserves NaN warmups identical to the Vec-returning API
///   (quiet NaNs for the first `first + period` slots).
/// - The output slice length must equal the input data length.
///
/// Returns `Ok(())` on success or the module's error on invalid inputs.
#[cfg(not(feature = "wasm"))]
#[inline]
pub fn pfe_into(input: &PfeInput, out: &mut [f64]) -> Result<(), PfeError> {
    let (data, period, smoothing, first, chosen) = pfe_prepare(input, Kernel::Auto)?;
    if out.len() != data.len() {
        return Err(PfeError::OutputLengthMismatch {
            expected: data.len(),
            got: out.len(),
        });
    }

    // Prefill warmup prefix with the same quiet-NaN pattern used by
    // `alloc_with_nan_prefix` to ensure exact parity of warmups.
    let warm = (first + period).min(out.len());
    for v in &mut out[..warm] {
        *v = f64::from_bits(0x7ff8_0000_0000_0000);
    }

    // Compute the indicator values into the provided buffer.
    pfe_compute_into(data, period, smoothing, first, chosen, out);
    Ok(())
}

#[inline]
pub fn pfe_into_slice(dst: &mut [f64], input: &PfeInput, k: Kernel) -> Result<(), PfeError> {
    let (data, period, smoothing, first, chosen) = pfe_prepare(input, k)?;
    if dst.len() != data.len() {
        return Err(PfeError::OutputLengthMismatch {
            expected: data.len(),
            got: dst.len(),
        });
    }
    pfe_compute_into(data, period, smoothing, first, chosen, dst);
    // ensure warmup prefix is NaN without bulk fills
    for v in &mut dst[..(first + period)] {
        *v = f64::NAN;
    }
    Ok(())
}

// Removed old pfe_scalar and AVX stubs - now using unified pfe_compute_into

#[inline]
pub fn pfe_batch_with_kernel(
    data: &[f64],
    sweep: &PfeBatchRange,
    k: Kernel,
) -> Result<PfeBatchOutput, PfeError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(PfeError::InvalidKernelForBatch(k));
        }
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    pfe_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct PfeBatchRange {
    pub period: (usize, usize, usize),
    pub smoothing: (usize, usize, usize),
}

impl Default for PfeBatchRange {
    fn default() -> Self {
        Self {
            period: (10, 40, 1),
            smoothing: (5, 10, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct PfeBatchBuilder {
    range: PfeBatchRange,
    kernel: Kernel,
}

impl PfeBatchBuilder {
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
    pub fn smoothing_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.smoothing = (start, end, step);
        self
    }
    #[inline]
    pub fn smoothing_static(mut self, s: usize) -> Self {
        self.range.smoothing = (s, s, 0);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<PfeBatchOutput, PfeError> {
        pfe_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<PfeBatchOutput, PfeError> {
        PfeBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<PfeBatchOutput, PfeError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<PfeBatchOutput, PfeError> {
        PfeBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct PfeBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<PfeParams>,
    pub rows: usize,
    pub cols: usize,
}
impl PfeBatchOutput {
    pub fn row_for_params(&self, p: &PfeParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(10) == p.period.unwrap_or(10)
                && c.smoothing.unwrap_or(5) == p.smoothing.unwrap_or(5)
        })
    }
    pub fn values_for(&self, p: &PfeParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &PfeBatchRange) -> Result<Vec<PfeParams>, PfeError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, PfeError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            let mut v = Vec::new();
            let mut x = start;
            let st = step.max(1);
            loop {
                v.push(x);
                let next = match x.checked_add(st) {
                    Some(nx) if nx > x && nx <= end => nx,
                    _ => break,
                };
                x = next;
            }
            if v.is_empty() {
                return Err(PfeError::InvalidRange {
                    start,
                    end,
                    step,
                });
            }
            return Ok(v);
        }
        let mut v = Vec::new();
        let mut x = start;
        let st = step.max(1);
        loop {
            v.push(x);
            if x <= end {
                break;
            }
            let next = x.saturating_sub(st);
            if next >= x {
                break;
            }
            x = next;
        }
        if v.is_empty() {
            return Err(PfeError::InvalidRange {
                start,
                end,
                step,
            });
        }
        Ok(v)
    }

    let periods = axis_usize(r.period)?;
    let smoothings = axis_usize(r.smoothing)?;

    let cap = periods
        .len()
        .checked_mul(smoothings.len())
        .ok_or(PfeError::InvalidRange {
            start: periods.len(),
            end: smoothings.len(),
            step: 0,
        })?;

    let mut out = Vec::with_capacity(cap);
    for &p in &periods {
        for &s in &smoothings {
            out.push(PfeParams {
                period: Some(p),
                smoothing: Some(s),
            });
        }
    }
    Ok(out)
}

#[inline(always)]
pub fn pfe_batch_slice(
    data: &[f64],
    sweep: &PfeBatchRange,
    kern: Kernel,
) -> Result<PfeBatchOutput, PfeError> {
    pfe_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn pfe_batch_par_slice(
    data: &[f64],
    sweep: &PfeBatchRange,
    kern: Kernel,
) -> Result<PfeBatchOutput, PfeError> {
    pfe_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
pub fn pfe_batch_inner_into(
    data: &[f64],
    sweep: &PfeBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<PfeParams>, PfeError> {
    let combos = expand_grid(sweep)?;
    if combos.is_empty() {
        return Err(PfeError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(PfeError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(PfeError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();

    let expected = rows
        .checked_mul(cols)
        .ok_or(PfeError::InvalidRange {
            start: rows,
            end: cols,
            step: 0,
        })?;
    if out.len() != expected {
        return Err(PfeError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    // Initialize warmup NaNs for all rows efficiently
    for (row, combo) in combos.iter().enumerate() {
        let warmup = first + combo.period.unwrap();
        let row_start = row
            .checked_mul(cols)
            .ok_or(PfeError::InvalidRange {
                start: row,
                end: cols,
                step: 0,
            })?;
        let end = row_start
            .checked_add(warmup.min(cols))
            .ok_or(PfeError::InvalidRange {
                start: row_start,
                end: warmup.min(cols),
                step: 0,
            })?;
        for i in row_start..end {
            out[i] = f64::NAN;
        }
    }

    // Shared prefix sums: prefix[i] = sum_{k=1..i} sqrt(1 + (ΔP_k)^2)
    let mut prefix = vec![0.0f64; cols];
    for i in 1..cols {
        let d = data[i] - data[i - 1];
        let s = (d.mul_add(d, 1.0)).sqrt();
        prefix[i] = prefix[i - 1] + s;
    }

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        let smoothing = combos[row].smoothing.unwrap();
        let _ = pfe_row_scalar_with_prefix(data, &prefix, first, period, smoothing, out_row);
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
fn pfe_batch_inner(
    data: &[f64],
    sweep: &PfeBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<PfeBatchOutput, PfeError> {
    let combos = expand_grid(sweep)?;
    if combos.is_empty() {
        return Err(PfeError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(PfeError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(PfeError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or(PfeError::InvalidRange {
            start: rows,
            end: cols,
            step: 0,
        })?;

    let mut buf_mu = make_uninit_matrix(rows, cols);
    let warm: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap()).collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
    let values_slice: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, total)
    };

    // Shared prefix sums across rows for batch speedup
    let mut prefix = vec![0.0f64; cols];
    for i in 1..cols {
        let d = data[i] - data[i - 1];
        let s = (d.mul_add(d, 1.0)).sqrt();
        prefix[i] = prefix[i - 1] + s;
    }

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        let smoothing = combos[row].smoothing.unwrap();
        let _ = pfe_row_scalar_with_prefix(data, &prefix, first, period, smoothing, out_row);
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            values_slice
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in values_slice.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in values_slice.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(PfeBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn pfe_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    smoothing: usize,
    out_row: &mut [f64],
) {
    let len = data.len();
    let start = first + period;

    let p = period as f64;
    let p2 = p * p;
    let alpha = 2.0 / (smoothing as f64 + 1.0);
    let one_minus_alpha = 1.0 - alpha;

    // Ring buffer of short-leg segment lengths over the last `period` steps
    let mut seg = vec![0.0f64; period];
    let mut head = 0usize;
    let mut denom = 0.0f64;

    // Initialize denominator at `start`
    let base = start - period + 1;
    // Unroll by 4
    let mut j = 0usize;
    let stop = period & !3;
    while j < stop {
        let k0 = base + j;
        let d0 = *data.get_unchecked(k0) - *data.get_unchecked(k0 - 1);
        let k1 = k0 + 1;
        let d1 = *data.get_unchecked(k1) - *data.get_unchecked(k1 - 1);
        let k2 = k1 + 1;
        let d2 = *data.get_unchecked(k2) - *data.get_unchecked(k2 - 1);
        let k3 = k2 + 1;
        let d3 = *data.get_unchecked(k3) - *data.get_unchecked(k3 - 1);

        let s0 = (d0.mul_add(d0, 1.0)).sqrt();
        let s1 = (d1.mul_add(d1, 1.0)).sqrt();
        let s2 = (d2.mul_add(d2, 1.0)).sqrt();
        let s3 = (d3.mul_add(d3, 1.0)).sqrt();

        *seg.get_unchecked_mut(j) = s0;
        *seg.get_unchecked_mut(j + 1) = s1;
        *seg.get_unchecked_mut(j + 2) = s2;
        *seg.get_unchecked_mut(j + 3) = s3;

        denom += s0 + s1 + s2 + s3;
        j += 4;
    }
    while j < period {
        let k = base + j;
        let d = *data.get_unchecked(k) - *data.get_unchecked(k - 1);
        let s = (d.mul_add(d, 1.0)).sqrt();
        *seg.get_unchecked_mut(j) = s;
        denom += s;
        j += 1;
    }

    // EMA state
    let mut ema_started = false;
    let mut ema_val = 0.0f64;

    for t in start..len {
        let cur = *data.get_unchecked(t);
        let past = *data.get_unchecked(t - period);
        let diff = cur - past;

        let long_leg = (diff.mul_add(diff, p2)).sqrt();
        let raw = if denom <= f64::EPSILON {
            0.0
        } else {
            100.0 * (long_leg / denom)
        };
        let signed = if diff > 0.0 { raw } else { -raw };

        let val = if !ema_started {
            ema_started = true;
            ema_val = signed;
            signed
        } else {
            ema_val = alpha.mul_add(signed, one_minus_alpha * ema_val);
            ema_val
        };

        *out_row.get_unchecked_mut(t) = val;

        if t + 1 < len {
            let old = *seg.get_unchecked(head);
            let next = *data.get_unchecked(t + 1);
            let new_d = next - cur;
            let new_s = (new_d.mul_add(new_d, 1.0)).sqrt();
            denom += new_s - old;
            *seg.get_unchecked_mut(head) = new_s;

            head += 1;
            if head == period {
                head = 0;
            }
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn pfe_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    smoothing: usize,
    out: &mut [f64],
) {
    use core::arch::x86_64::*;

    let len = data.len();
    let start = first + period;

    let p = period as f64;
    let p2 = p * p;
    let alpha = 2.0 / (smoothing as f64 + 1.0);
    let one_minus_alpha = 1.0 - alpha;

    // Allocate contiguous ring buffer for segment lengths
    let mut seg = vec![0.0f64; period];
    let mut head = 0usize;

    // Vectorized initialization of denominator at `start`
    let base = start - period + 1;
    let mut denom = 0.0f64;

    let ones = _mm256_set1_pd(1.0);

    let mut j = 0usize;
    let stop = period & !3;
    while j < stop {
        let k0 = base + j;

        let prev = _mm256_loadu_pd(data.as_ptr().add(k0 - 1));
        let curr = _mm256_loadu_pd(data.as_ptr().add(k0));
        let diff = _mm256_sub_pd(curr, prev);
        let sq = _mm256_sqrt_pd(_mm256_add_pd(_mm256_mul_pd(diff, diff), ones));

        _mm256_storeu_pd(seg.as_mut_ptr().add(j), sq);

        let hadd1 = _mm256_hadd_pd(sq, sq);
        let lo = _mm256_extractf128_pd(hadd1, 0);
        let hi = _mm256_extractf128_pd(hadd1, 1);
        let sum2 = _mm_add_pd(lo, hi);
        denom += _mm_cvtsd_f64(sum2);

        j += 4;
    }
    while j < period {
        let k = base + j;
        let d = *data.get_unchecked(k) - *data.get_unchecked(k - 1);
        let s = (d.mul_add(d, 1.0)).sqrt();
        *seg.get_unchecked_mut(j) = s;
        denom += s;
        j += 1;
    }

    // EMA state
    let mut ema_started = false;
    let mut ema_val = 0.0f64;

    for t in start..len {
        let cur = *data.get_unchecked(t);
        let past = *data.get_unchecked(t - period);
        let diff = cur - past;

        let long_leg = (diff.mul_add(diff, p2)).sqrt();
        let raw = if denom <= f64::EPSILON {
            0.0
        } else {
            100.0 * (long_leg / denom)
        };
        let signed = if diff > 0.0 { raw } else { -raw };

        let val = if !ema_started {
            ema_started = true;
            ema_val = signed;
            signed
        } else {
            ema_val = alpha.mul_add(signed, one_minus_alpha * ema_val);
            ema_val
        };

        *out.get_unchecked_mut(t) = val;

        if t + 1 < len {
            let old = *seg.get_unchecked(head);
            let next = *data.get_unchecked(t + 1);
            let new_d = next - cur;
            let new_s = (new_d.mul_add(new_d, 1.0)).sqrt();
            denom += new_s - old;
            *seg.get_unchecked_mut(head) = new_s;

            head += 1;
            if head == period {
                head = 0;
            }
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn pfe_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    smoothing: usize,
    out: &mut [f64],
) {
    use core::arch::x86_64::*;

    let len = data.len();
    let start = first + period;

    let p = period as f64;
    let p2 = p * p;
    let alpha = 2.0 / (smoothing as f64 + 1.0);
    let one_minus_alpha = 1.0 - alpha;

    // Allocate contiguous ring buffer for segment lengths
    let mut seg = vec![0.0f64; period];
    let mut head = 0usize;

    // Vectorized initialization of denominator at `start`
    let base = start - period + 1;
    let mut denom = 0.0f64;

    let ones = _mm512_set1_pd(1.0);

    let mut j = 0usize;
    let stop = period & !7;
    while j < stop {
        let k0 = base + j;

        let prev = _mm512_loadu_pd(data.as_ptr().add(k0 - 1));
        let curr = _mm512_loadu_pd(data.as_ptr().add(k0));
        let diff = _mm512_sub_pd(curr, prev);
        let sq = _mm512_sqrt_pd(_mm512_add_pd(_mm512_mul_pd(diff, diff), ones));

        _mm512_storeu_pd(seg.as_mut_ptr().add(j), sq);

        let lo256 = _mm512_extractf64x4_pd(sq, 0);
        let hi256 = _mm512_extractf64x4_pd(sq, 1);
        let hadd_lo = _mm256_hadd_pd(lo256, lo256);
        let hadd_hi = _mm256_hadd_pd(hi256, hi256);
        let lo_pair = _mm_add_pd(
            _mm256_extractf128_pd(hadd_lo, 0),
            _mm256_extractf128_pd(hadd_lo, 1),
        );
        let hi_pair = _mm_add_pd(
            _mm256_extractf128_pd(hadd_hi, 0),
            _mm256_extractf128_pd(hadd_hi, 1),
        );
        denom += _mm_cvtsd_f64(lo_pair) + _mm_cvtsd_f64(hi_pair);

        j += 8;
    }
    while j < period {
        let k = base + j;
        let d = *data.get_unchecked(k) - *data.get_unchecked(k - 1);
        let s = (d.mul_add(d, 1.0)).sqrt();
        *seg.get_unchecked_mut(j) = s;
        denom += s;
        j += 1;
    }

    // EMA state
    let mut ema_started = false;
    let mut ema_val = 0.0f64;

    for t in start..len {
        let cur = *data.get_unchecked(t);
        let past = *data.get_unchecked(t - period);
        let diff = cur - past;

        let long_leg = (diff.mul_add(diff, p2)).sqrt();
        let raw = if denom <= f64::EPSILON {
            0.0
        } else {
            100.0 * (long_leg / denom)
        };
        let signed = if diff > 0.0 { raw } else { -raw };

        let val = if !ema_started {
            ema_started = true;
            ema_val = signed;
            signed
        } else {
            ema_val = alpha.mul_add(signed, one_minus_alpha * ema_val);
            ema_val
        };

        *out.get_unchecked_mut(t) = val;

        if t + 1 < len {
            let old = *seg.get_unchecked(head);
            let next = *data.get_unchecked(t + 1);
            let new_d = next - cur;
            let new_s = (new_d.mul_add(new_d, 1.0)).sqrt();
            denom += new_s - old;
            *seg.get_unchecked_mut(head) = new_s;

            head += 1;
            if head == period {
                head = 0;
            }
        }
    }
}

// Batch optimization: shared prefix sums of step lengths
#[inline(always)]
unsafe fn pfe_row_scalar_with_prefix(
    data: &[f64],
    prefix: &[f64], // prefix[i] = sum_{k=1..i} sqrt(1 + (ΔP)^2) with ΔP at k
    first: usize,
    period: usize,
    smoothing: usize,
    out_row: &mut [f64],
) {
    let len = data.len();
    let start = first + period;

    let p = period as f64;
    let p2 = p * p;
    let alpha = 2.0 / (smoothing as f64 + 1.0);
    let one_minus_alpha = 1.0 - alpha;

    let mut ema_started = false;
    let mut ema_val = 0.0f64;

    for t in start..len {
        let cur = *data.get_unchecked(t);
        let past = *data.get_unchecked(t - period);
        let diff = cur - past;

        let long_leg = (diff.mul_add(diff, p2)).sqrt();
        let denom = *prefix.get_unchecked(t) - *prefix.get_unchecked(t - period);
        let raw = if denom <= f64::EPSILON {
            0.0
        } else {
            100.0 * (long_leg / denom)
        };
        let signed = if diff > 0.0 { raw } else { -raw };

        let val = if !ema_started {
            ema_started = true;
            ema_val = signed;
            signed
        } else {
            ema_val = alpha.mul_add(signed, one_minus_alpha * ema_val);
            ema_val
        };

        *out_row.get_unchecked_mut(t) = val;
    }
}

// Decision: Streaming kernel uses contiguous Vec rings (prices, segments),
// not VecDeque; exact math retained for test parity. O(1) per tick.

#[derive(Debug, Clone)]
pub struct PfeStream {
    // params
    period: usize,
    smoothing: usize,

    // precomputed constants
    alpha: f64,
    one_minus_alpha: f64,
    p2: f64, // (period as f64)^2

    // ring of last (period + 1) prices
    prices: Vec<f64>,   // capacity = period + 1 (contiguous)
    price_pos: usize,   // index of most-recent element in `prices`
    price_count: usize, // how many prices we have (<= period+1)

    // ring of last `period` short-leg step lengths: sqrt(1 + d^2)
    seg: Vec<f64>,    // capacity = period
    seg_pos: usize,   // next position to overwrite in `seg`
    seg_count: usize, // how many segments we have (<= period)
    short_sum: f64,   // running sum of `seg`

    // step state
    last_price: f64,
    have_last: bool,

    // EMA state
    ema_val: f64,
    started: bool,
}

impl PfeStream {
    pub fn try_new(params: PfeParams) -> Result<Self, PfeError> {
        let period = params.period.unwrap_or(10);
        if period == 0 {
            return Err(PfeError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let smoothing = params.smoothing.unwrap_or(5);
        if smoothing == 0 {
            return Err(PfeError::InvalidSmoothing { smoothing });
        }

        let cap_prices = period + 1;
        let cap_seg = period;
        let alpha = 2.0 / (smoothing as f64 + 1.0);

        Ok(Self {
            period,
            smoothing,
            alpha,
            one_minus_alpha: 1.0 - alpha,
            p2: (period as f64) * (period as f64),

            prices: vec![0.0; cap_prices],
            price_pos: cap_prices - 1, // so first write lands at 0
            price_count: 0,

            seg: if cap_seg > 0 {
                vec![0.0; cap_seg]
            } else {
                Vec::new()
            },
            seg_pos: 0,
            seg_count: 0,
            short_sum: 0.0,

            last_price: 0.0,
            have_last: false,

            ema_val: 0.0,
            started: false,
        })
    }

    /// Push one price. Returns None until `period+1` samples have been seen.
    #[inline(always)]
    pub fn update(&mut self, price: f64) -> Option<f64> {
        // 1) Update short-leg with the new step (t-1 -> t)
        if self.have_last {
            let d = price - self.last_price;
            // exact hypot-like step: sqrt(1 + d^2)
            let new_s = (d.mul_add(d, 1.0)).sqrt();

            if self.seg_count < self.period {
                // still filling the segment ring
                self.short_sum += new_s;
                if self.period > 0 {
                    self.seg[self.seg_pos] = new_s;
                    self.seg_pos += 1;
                    if self.seg_pos == self.period {
                        self.seg_pos = 0;
                    }
                }
                self.seg_count += 1;
            } else if self.period > 0 {
                // steady state: drop oldest, add newest
                let old = self.seg[self.seg_pos];
                self.short_sum += new_s - old;
                self.seg[self.seg_pos] = new_s;
                self.seg_pos += 1;
                if self.seg_pos == self.period {
                    self.seg_pos = 0;
                }
            }
        }
        self.last_price = price;
        self.have_last = true;

        // 2) Insert the new price into the price ring
        let cap_prices = self.period + 1;
        self.price_pos += 1;
        if self.price_pos == cap_prices {
            self.price_pos = 0;
        }
        self.prices[self.price_pos] = price;
        if self.price_count < cap_prices {
            self.price_count += 1;
        }

        // 3) Not enough samples yet? No output.
        if self.price_count < cap_prices {
            return None;
        }

        // 4) Compute diff = P_t - P_{t-period}
        //    Oldest element is the slot after the current (wrap-aware).
        let past_idx = self.price_pos + 1 - ((self.price_pos + 1) / cap_prices) * cap_prices;
        let past = self.prices[past_idx];
        let diff = price - past;

        // 5) Long leg = sqrt(diff^2 + period^2); denom = rolling sum of short legs
        let long_leg = (diff.mul_add(diff, self.p2)).sqrt();
        let denom = self.short_sum;

        // 6) Raw PFE (guard small denom), sign by diff>0
        let inv = if denom <= f64::EPSILON {
            0.0
        } else {
            1.0 / denom
        };
        let raw = 100.0 * long_leg * inv;
        let signed = if diff > 0.0 { raw } else { -raw };

        // 7) EMA smoothing
        let out = if !self.started {
            self.started = true;
            self.ema_val = signed;
            signed
        } else {
            self.ema_val = self
                .alpha
                .mul_add(signed, self.one_minus_alpha * self.ema_val);
            self.ema_val
        };

        Some(out)
    }
}

// ---------------- CUDA Python device handle (CAI v3 + DLPack v1.x) ----------------
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", name = "PfeDeviceArrayF32", unsendable)]
pub struct PfeDeviceArrayF32Py {
    pub(crate) inner: DeviceArrayF32,
    _ctx_guard: Arc<Context>,
    _device_id: u32,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl PfeDeviceArrayF32Py {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let inner = &self.inner;
        let d = PyDict::new(py);
        let itemsize = std::mem::size_of::<f32>();
        d.set_item("shape", (inner.rows, inner.cols))?;
        d.set_item("typestr", "<f4")?;
        d.set_item("strides", (inner.cols * itemsize, itemsize))?;
        d.set_item("data", (inner.device_ptr() as usize, false))?;
        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        (2, self._device_id as i32)
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
        // Compute target device id and validate requested `dl_device` (if any).
        let (kdl, alloc_dev) = self.__dlpack_device__(); // (2, device_id)
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
                        return Err(PyValueError::new_err(
                            "dl_device mismatch for __dlpack__",
                        ));
                    }
                }
            }
        }
        let _ = stream;

        // Move VRAM handle out of this wrapper; the DLPack capsule owns it afterwards.
        let dummy = DeviceBuffer::from_slice(&[])
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let inner = std::mem::replace(
            &mut self.inner,
            DeviceArrayF32 {
                buf: dummy,
                rows: 0,
                cols: 0,
            },
        );

        let rows = inner.rows;
        let cols = inner.cols;
        let buf = inner.buf;

        let max_version_bound = max_version.map(|obj| obj.into_bound(py));
        export_f32_cuda_dlpack_2d(py, buf, rows, cols, alloc_dev, max_version_bound)
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
impl PfeDeviceArrayF32Py {
    pub fn new_from_rust(inner: DeviceArrayF32, ctx_guard: Arc<Context>, device_id: u32) -> Self {
        Self {
            inner,
            _ctx_guard: ctx_guard,
            _device_id: device_id,
        }
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "pfe")]
#[pyo3(signature = (data, period, smoothing, kernel=None))]
pub fn pfe_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period: usize,
    smoothing: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let params = PfeParams {
        period: Some(period),
        smoothing: Some(smoothing),
    };
    let pfe_in = PfeInput::from_slice(slice_in, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| pfe_with_kernel(&pfe_in, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction(name = "pfe_batch")]
#[pyo3(signature = (data, period_range, smoothing_range, kernel=None))]
pub fn pfe_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    smoothing_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;

    let sweep = PfeBatchRange {
        period: period_range,
        smoothing: smoothing_range,
    };

    let combos = expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = slice_in.len();

    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("pfe_batch: rows*cols overflow"))?;
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
                _ => unreachable!(),
            };
            pfe_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
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
        "smoothings",
        combos
            .iter()
            .map(|p| p.smoothing.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    Ok(dict)
}

// ---------------- CUDA Python bindings ----------------
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "pfe_cuda_batch_dev")]
#[pyo3(signature = (data, period_range, smoothing_range, device_id=0))]
pub fn pfe_cuda_batch_dev_py(
    py: Python<'_>,
    data: PyReadonlyArray1<'_, f32>,
    period_range: (usize, usize, usize),
    smoothing_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<PfeDeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let slice_in = data.as_slice()?;
    let sweep = PfeBatchRange {
        period: period_range,
        smoothing: smoothing_range,
    };
    let (inner, ctx, dev_id) = py
        .allow_threads(|| {
            let cuda =
                CudaPfe::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
            let dev = cuda
                .pfe_batch_dev(slice_in, &sweep)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok::<_, PyErr>((dev, cuda.context_arc(), cuda.device_id()))
        })?;
    Ok(PfeDeviceArrayF32Py::new_from_rust(inner, ctx, dev_id))
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "pfe_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm, cols, rows, period, smoothing, device_id=0))]
pub fn pfe_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    data_tm: PyReadonlyArray1<'_, f32>,
    cols: usize,
    rows: usize,
    period: usize,
    smoothing: usize,
    device_id: usize,
) -> PyResult<PfeDeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let tm_slice = data_tm.as_slice()?;
    let (inner, ctx, dev_id) = py
        .allow_threads(|| {
            let cuda =
                CudaPfe::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
            let dev = cuda
                .pfe_many_series_one_param_time_major_dev(tm_slice, cols, rows, period, smoothing)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok::<_, PyErr>((dev, cuda.context_arc(), cuda.device_id()))
        })?;
    Ok(PfeDeviceArrayF32Py::new_from_rust(inner, ctx, dev_id))
}

#[cfg(feature = "python")]
#[pyclass(name = "PfeStream")]
pub struct PfeStreamPy {
    stream: PfeStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl PfeStreamPy {
    #[new]
    fn new(period: usize, smoothing: usize) -> PyResult<Self> {
        let params = PfeParams {
            period: Some(period),
            smoothing: Some(smoothing),
        };
        let stream =
            PfeStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PfeStreamPy { stream })
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn pfe_js(data: &[f64], period: usize, smoothing: usize) -> Result<Vec<f64>, JsValue> {
    let params = PfeParams {
        period: Some(period),
        smoothing: Some(smoothing),
    };
    let input = PfeInput::from_slice(data, params);
    let mut out = vec![0.0; data.len()];
    pfe_into_slice(&mut out, &input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(out)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn pfe_alloc(len: usize) -> *mut f64 {
    let mut v = Vec::<f64>::with_capacity(len);
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn pfe_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn pfe_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
    smoothing: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer"));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let params = PfeParams {
            period: Some(period),
            smoothing: Some(smoothing),
        };
        let input = PfeInput::from_slice(data, params);
        if in_ptr == out_ptr {
            let mut tmp = vec![0.0; len];
            pfe_into_slice(&mut tmp, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            std::slice::from_raw_parts_mut(out_ptr, len).copy_from_slice(&tmp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            pfe_into_slice(out, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct PfeBatchConfig {
    pub period_range: (usize, usize, usize),
    pub smoothing_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct PfeBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<PfeParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = pfe_batch)]
pub fn pfe_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let cfg: PfeBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
    let sweep = PfeBatchRange {
        period: cfg.period_range,
        smoothing: cfg.smoothing_range,
    };
    let out = pfe_batch_inner(data, &sweep, detect_best_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let js = PfeBatchJsOutput {
        values: out.values,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    };
    serde_wasm_bindgen::to_value(&js)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn pfe_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    p_start: usize,
    p_end: usize,
    p_step: usize,
    s_start: usize,
    s_end: usize,
    s_step: usize,
    ) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer"));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let sweep = PfeBatchRange {
            period: (p_start, p_end, p_step),
            smoothing: (s_start, s_end, s_step),
        };
        let combos = expand_grid(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let total = len
            .checked_mul(rows)
            .ok_or_else(|| JsValue::from_str("pfe_batch_into: rows*cols overflow"))?;
        let combos = pfe_batch_inner_into(
            data,
            &sweep,
            detect_best_kernel(),
            false,
            std::slice::from_raw_parts_mut(out_ptr, total),
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(combos.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_pfe_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = PfeParams {
            period: None,
            smoothing: None,
        };
        let input = PfeInput::from_candles(&candles, "close", default_params);
        let output = pfe_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_pfe_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let close_prices = &candles.close;

        let params = PfeParams {
            period: Some(10),
            smoothing: Some(5),
        };
        let input = PfeInput::from_candles(&candles, "close", params);
        let pfe_result = pfe_with_kernel(&input, kernel)?;

        assert_eq!(pfe_result.values.len(), close_prices.len());

        let expected_last_five_pfe = [
            -13.03562252,
            -11.93979855,
            -9.94609862,
            -9.73372410,
            -14.88374798,
        ];
        let start_index = pfe_result.values.len() - 5;
        let result_last_five_pfe = &pfe_result.values[start_index..];
        for (i, &value) in result_last_five_pfe.iter().enumerate() {
            let expected_value = expected_last_five_pfe[i];
            assert!(
                (value - expected_value).abs() < 1e-8,
                "[{}] PFE mismatch at idx {}: got {}, expected {}",
                test_name,
                i,
                value,
                expected_value
            );
        }

        for i in 0..(10 - 1) {
            assert!(pfe_result.values[i].is_nan());
        }

        Ok(())
    }

    fn check_pfe_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = PfeInput::with_default_candles(&candles);
        match input.data {
            PfeData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected PfeData::Candles"),
        }
        let output = pfe_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_pfe_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = PfeParams {
            period: Some(0),
            smoothing: Some(5),
        };
        let input = PfeInput::from_slice(&input_data, params);
        let res = pfe_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] PFE should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_pfe_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = PfeParams {
            period: Some(10),
            smoothing: Some(2),
        };
        let input = PfeInput::from_slice(&data_small, params);
        let res = pfe_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] PFE should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_pfe_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = PfeParams {
            period: Some(10),
            smoothing: Some(2),
        };
        let input = PfeInput::from_slice(&single_point, params);
        let res = pfe_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] PFE should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_pfe_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = PfeParams {
            period: Some(10),
            smoothing: Some(5),
        };
        let first_input = PfeInput::from_candles(&candles, "close", first_params);
        let first_result = pfe_with_kernel(&first_input, kernel)?;

        let second_params = PfeParams {
            period: Some(10),
            smoothing: Some(5),
        };
        let second_input = PfeInput::from_slice(&first_result.values, second_params);
        let second_result = pfe_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());
        for i in 20..second_result.values.len() {
            assert!(
                !second_result.values[i].is_nan(),
                "[{}] Expected no NaN after index 20, but found NaN at index {}",
                test_name,
                i
            );
        }
        Ok(())
    }

    fn check_pfe_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = PfeInput::from_candles(
            &candles,
            "close",
            PfeParams {
                period: Some(10),
                smoothing: Some(5),
            },
        );
        let res = pfe_with_kernel(&input, kernel)?;
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

    fn check_pfe_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let period = 10;
        let smoothing = 5;

        let input = PfeInput::from_candles(
            &candles,
            "close",
            PfeParams {
                period: Some(period),
                smoothing: Some(smoothing),
            },
        );
        let batch_output = pfe_with_kernel(&input, kernel)?.values;

        let mut stream = PfeStream::try_new(PfeParams {
            period: Some(period),
            smoothing: Some(smoothing),
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
                "[{}] PFE streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    #[test]
    fn test_pfe_into_matches_api() -> Result<(), Box<dyn Error>> {
        let mut data = vec![0.0f64; 256];
        for (i, v) in data.iter_mut().enumerate() {
            let x = i as f64;
            *v = 0.05 * x + (x * 0.1).sin() * 2.0 + (x * 0.03).cos();
        }

        let input = PfeInput::from_slice(&data, PfeParams::default());
        let baseline = pfe(&input)?.values;

        let mut into_out = vec![0.0f64; data.len()];
        pfe_into(&input, &mut into_out)?;
        assert_eq!(baseline.len(), into_out.len());

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b)
        }
        for i in 0..baseline.len() {
            assert!(
                eq_or_both_nan(baseline[i], into_out[i]),
                "Mismatch at index {}: into={}, api={}",
                i,
                into_out[i],
                baseline[i]
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_pfe_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Define comprehensive parameter combinations
        let test_params = vec![
            PfeParams::default(), // period: 10, smoothing: 5
            PfeParams {
                period: Some(2),    // minimum viable period
                smoothing: Some(1), // minimum smoothing
            },
            PfeParams {
                period: Some(5),
                smoothing: Some(2),
            },
            PfeParams {
                period: Some(10),
                smoothing: Some(3),
            },
            PfeParams {
                period: Some(14),
                smoothing: Some(5),
            },
            PfeParams {
                period: Some(20),
                smoothing: Some(5),
            },
            PfeParams {
                period: Some(20),
                smoothing: Some(10),
            },
            PfeParams {
                period: Some(50), // large period
                smoothing: Some(15),
            },
            PfeParams {
                period: Some(100), // very large period
                smoothing: Some(20),
            },
            PfeParams {
                period: Some(3),
                smoothing: Some(30), // smoothing > period case
            },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = PfeInput::from_candles(&candles, "close", params.clone());
            let output = pfe_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: period={}, smoothing={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(10),
                        params.smoothing.unwrap_or(5),
                        param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: period={}, smoothing={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(10),
                        params.smoothing.unwrap_or(5),
                        param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: period={}, smoothing={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(10),
                        params.smoothing.unwrap_or(5),
                        param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_pfe_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(()) // No-op in release builds
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_pfe_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // More comprehensive data generation strategy
        let strat = (2usize..=64).prop_flat_map(|period| {
            (
                // Generate varied price data - from very small to very large values
                prop::collection::vec(
                    prop::strategy::Union::new(vec![
                        // Small values
                        (0.01f64..10.0f64).boxed(),
                        // Medium values
                        (10.0f64..1000.0f64).boxed(),
                        // Large values
                        (1000.0f64..100000.0f64).boxed(),
                    ])
                    .prop_filter("finite", |x| x.is_finite()),
                    period + 50..400,
                ),
                Just(period),
                1usize..=20, // smoothing range
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, period, smoothing)| {
                let params = PfeParams {
                    period: Some(period),
                    smoothing: Some(smoothing),
                };
                let input = PfeInput::from_slice(&data, params);

                let PfeOutput { values: out } = pfe_with_kernel(&input, kernel).unwrap();
                let PfeOutput { values: ref_out } =
                    pfe_with_kernel(&input, Kernel::Scalar).unwrap();

                // Basic structure checks
                prop_assert_eq!(
                    out.len(),
                    data.len(),
                    "[{}] Output length mismatch",
                    test_name
                );

                // Warmup period validation
                for i in 0..period {
                    prop_assert!(
                        out[i].is_nan(),
                        "[{}] Expected NaN at index {} during warmup (period={})",
                        test_name,
                        i,
                        period
                    );
                }

                // Main validation loop with proper floating point tolerance
                for i in period..data.len() {
                    let y = out[i];
                    let r = ref_out[i];

                    // Both NaN or both finite
                    if y.is_nan() != r.is_nan() {
                        prop_assert!(
                            false,
                            "[{}] NaN mismatch at index {}: {} vs {}",
                            test_name,
                            i,
                            y,
                            r
                        );
                    }

                    if y.is_finite() {
                        // Theoretical bounds check
                        prop_assert!(
                            y >= -100.0 && y <= 100.0,
                            "[{}] PFE value {} at index {} out of bounds [-100, 100]",
                            test_name,
                            y,
                            i
                        );

                        // Kernel consistency with ULP tolerance
                        let y_bits = y.to_bits();
                        let r_bits = r.to_bits();
                        let ulp_diff = y_bits.abs_diff(r_bits);

                        // Allow small ULP difference for accumulated floating point errors
                        prop_assert!(
                            (y - r).abs() <= 1e-9 || ulp_diff <= 4,
                            "[{}] Kernel mismatch at index {}: {} vs {} (ULP diff: {})",
                            test_name,
                            i,
                            y,
                            r,
                            ulp_diff
                        );
                    }
                }

                // Verify PFE calculation logic for specific patterns
                // 1. Perfect straight line up should have high positive efficiency
                let straight_up: Vec<f64> = (0..100).map(|i| 100.0 + i as f64).collect();
                let straight_params = PfeParams {
                    period: Some(10),
                    smoothing: Some(1), // Minimal smoothing to see raw PFE
                };
                let straight_input = PfeInput::from_slice(&straight_up, straight_params);
                if let Ok(straight_out) = pfe_with_kernel(&straight_input, kernel) {
                    // After warmup, straight line should have efficiency near 100
                    for i in 15..straight_out.values.len() {
                        if straight_out.values[i].is_finite() {
                            prop_assert!(
								straight_out.values[i] > 50.0,
								"[{}] Straight line up should have high positive efficiency, got {} at index {}",
								test_name,
								straight_out.values[i],
								i
							);
                        }
                    }
                }

                // 2. Zigzag pattern should have lower absolute efficiency
                let zigzag: Vec<f64> = (0..100)
                    .map(|i| {
                        if i % 2 == 0 {
                            100.0 + (i as f64)
                        } else {
                            100.0 + (i as f64) - 5.0
                        }
                    })
                    .collect();
                let zigzag_params = PfeParams {
                    period: Some(10),
                    smoothing: Some(1),
                };
                let zigzag_input = PfeInput::from_slice(&zigzag, zigzag_params.clone());
                let straight_input2 = PfeInput::from_slice(&straight_up, zigzag_params);

                if let (Ok(zigzag_out), Ok(straight_out2)) = (
                    pfe_with_kernel(&zigzag_input, kernel),
                    pfe_with_kernel(&straight_input2, kernel),
                ) {
                    // Compare average absolute efficiency
                    let zigzag_avg: f64 = zigzag_out.values[20..50]
                        .iter()
                        .filter(|x| x.is_finite())
                        .map(|x| x.abs())
                        .sum::<f64>()
                        / 30.0;
                    let straight_avg: f64 = straight_out2.values[20..50]
                        .iter()
                        .filter(|x| x.is_finite())
                        .map(|x| x.abs())
                        .sum::<f64>()
                        / 30.0;

                    prop_assert!(
						zigzag_avg < straight_avg,
						"[{}] Zigzag pattern should have lower efficiency ({}) than straight line ({})",
						test_name,
						zigzag_avg,
						straight_avg
					);
                }

                // 3. Test EMA smoothing effect
                if smoothing > 1 && data.len() > period + 30 {
                    // Calculate unsmoothed version
                    let unsmoothed_params = PfeParams {
                        period: Some(period),
                        smoothing: Some(1),
                    };
                    let unsmoothed_input = PfeInput::from_slice(&data, unsmoothed_params);

                    if let Ok(unsmoothed_out) = pfe_with_kernel(&unsmoothed_input, kernel) {
                        // Use a larger window for more stable variance calculation
                        let window_start = period + 10;
                        let window_end = (period + 30).min(out.len());

                        if window_end > window_start {
                            // Calculate variance to compare smoothness
                            let smoothed_variance =
                                calculate_variance(&out[window_start..window_end]);
                            let unsmoothed_variance = calculate_variance(
                                &unsmoothed_out.values[window_start..window_end],
                            );

                            // Check for extreme price jumps in the data
                            let mut has_extreme_jumps = false;
                            for i in 1..data.len() {
                                let ratio = if data[i - 1] != 0.0 {
                                    (data[i] / data[i - 1]).abs()
                                } else {
                                    f64::INFINITY
                                };
                                // If any price changes by more than 100x, consider it extreme
                                if ratio > 100.0 || ratio < 0.01 {
                                    has_extreme_jumps = true;
                                    break;
                                }
                            }

                            // Only apply smoothness check if:
                            // 1. Both variances are finite
                            // 2. There's meaningful variance (not near-constant)
                            // 3. Data doesn't have extreme jumps
                            // 4. Smoothing parameter is reasonable (not too high)
                            if smoothed_variance.is_finite()
                                && unsmoothed_variance.is_finite()
                                && unsmoothed_variance > 1e-6
                                && !has_extreme_jumps
                                && smoothing <= 10
                            {
                                // For normal data, smoothing should reduce variance
                                // Allow up to 50% higher variance as EMA can amplify certain patterns
                                prop_assert!(
									smoothed_variance <= unsmoothed_variance * 1.5,
									"[{}] Smoothed variance ({}) should be <= 1.5x unsmoothed variance ({})",
									test_name,
									smoothed_variance,
									unsmoothed_variance
								);
                            }
                        }
                    }
                }

                // 4. Edge case: period = 2 (minimum viable)
                if period == 2 {
                    // With period=2, we're only looking at 2 points
                    // The calculation should still produce valid results
                    for i in 2..out.len() {
                        if out[i].is_finite() {
                            prop_assert!(
								out[i] >= -100.0 && out[i] <= 100.0,
								"[{}] Period=2 should still produce valid bounded values, got {} at index {}",
								test_name,
								out[i],
								i
							);
                        }
                    }
                }

                // 5. Constant prices verification - mathematically should be -100
                let constant: Vec<f64> = vec![500.0; 50];
                let const_params = PfeParams {
                    period: Some(10),
                    smoothing: Some(1), // No smoothing to see raw value
                };
                let const_input = PfeInput::from_slice(&constant, const_params);
                if let Ok(const_out) = pfe_with_kernel(&const_input, kernel) {
                    // For constant prices:
                    // diff = 0, long_leg = period, short_leg = period * 1 = period
                    // raw_pfe = 100 * (period/period) = 100
                    // signed_pfe = -100 (because diff <= 0)
                    for i in 15..const_out.values.len() {
                        if const_out.values[i].is_finite() {
                            prop_assert!(
								(const_out.values[i] - (-100.0)).abs() < 1e-6,
								"[{}] Constant prices should produce exactly -100, got {} at index {}",
								test_name,
								const_out.values[i],
								i
							);
                        }
                    }
                }

                Ok(())
            })
            .unwrap();

        // Helper function for variance calculation
        fn calculate_variance(values: &[f64]) -> f64 {
            let finite_values: Vec<f64> =
                values.iter().filter(|x| x.is_finite()).copied().collect();

            if finite_values.is_empty() {
                return f64::NAN;
            }

            let mean = finite_values.iter().sum::<f64>() / finite_values.len() as f64;
            let variance = finite_values
                .iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>()
                / finite_values.len() as f64;
            variance
        }

        Ok(())
    }

    macro_rules! generate_all_pfe_tests {
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

    generate_all_pfe_tests!(
        check_pfe_partial_params,
        check_pfe_accuracy,
        check_pfe_default_candles,
        check_pfe_zero_period,
        check_pfe_period_exceeds_length,
        check_pfe_very_small_dataset,
        check_pfe_reinput,
        check_pfe_nan_handling,
        check_pfe_streaming,
        check_pfe_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_pfe_tests!(check_pfe_property);

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = PfeBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = PfeParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            -13.03562252,
            -11.93979855,
            -9.94609862,
            -9.73372410,
            -14.88374798,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-8,
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
            // (period_start, period_end, period_step, smoothing_start, smoothing_end, smoothing_step)
            (2, 10, 2, 1, 5, 1),       // Small periods and smoothing
            (5, 25, 5, 2, 10, 2),      // Medium periods and smoothing
            (30, 60, 15, 5, 20, 5),    // Large periods and smoothing
            (2, 5, 1, 1, 3, 1),        // Dense small range
            (10, 10, 0, 1, 20, 1),     // Static period, varying smoothing
            (2, 100, 10, 5, 5, 0),     // Varying period, static smoothing
            (14, 21, 7, 3, 9, 3),      // Medium focused range
            (50, 100, 25, 10, 30, 10), // Large focused range
        ];

        for (cfg_idx, &(p_start, p_end, p_step, s_start, s_end, s_step)) in
            test_configs.iter().enumerate()
        {
            let output = PfeBatchBuilder::new()
                .kernel(kernel)
                .period_range(p_start, p_end, p_step)
                .smoothing_range(s_start, s_end, s_step)
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
						 at row {} col {} (flat index {}) with params: period={}, smoothing={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(10),
                        combo.smoothing.unwrap_or(5)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}, smoothing={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(10),
                        combo.smoothing.unwrap_or(5)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}, smoothing={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(10),
                        combo.smoothing.unwrap_or(5)
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
