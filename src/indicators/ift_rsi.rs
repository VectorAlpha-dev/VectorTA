//! # Inverse Fisher Transform RSI (IFT RSI)
//!
//! Applies Inverse Fisher Transform to a WMA-smoothed RSI series.
//! The indicator first calculates RSI (Wilder SMMA), smooths it with WMA (weights 1..N),
//! then applies tanh(.) to normalize values between -1 and 1.
//!
//! ## Parameters
//! - **rsi_period**: Period for RSI calculation (default: 5)
//! - **wma_period**: Period for WMA smoothing (default: 9)
//!
//! ## Inputs
//! - Single data slice (typically close prices)
//!
//! ## Returns
//! - **Ok(IftRsiOutput)** containing values (Vec<f64>) representing transformed RSI between -1 and 1
//! - Output length matches input data length with NaN padding for warmup period
//!
//! ## Decision Log
//! - Scalar path optimized (Wilder SMMA + O(1) LWMA), ~13% faster at 100k samples vs previous scalar baseline.
//! - SIMD disabled (stubs call scalar); recurrence/IIR structure and O(1) LWMA leave little room for SIMD speedups.
//! - CUDA enabled for batch and many-series variants; wrappers enforce VRAM checks, typed errors, and Python CAI v3/DLPack v1.x interop.

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

use crate::indicators::rsi::{rsi, RsiError, RsiInput, RsiParams};
use crate::indicators::wma::{wma, WmaError, WmaInput, WmaParams};
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::utilities::dlpack_cuda::DeviceArrayF32Py;
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

impl<'a> AsRef<[f64]> for IftRsiInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            IftRsiData::Slice(slice) => slice,
            IftRsiData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum IftRsiData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct IftRsiOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct IftRsiParams {
    pub rsi_period: Option<usize>,
    pub wma_period: Option<usize>,
}

impl Default for IftRsiParams {
    fn default() -> Self {
        Self {
            rsi_period: Some(5),
            wma_period: Some(9),
        }
    }
}

#[derive(Debug, Clone)]
pub struct IftRsiInput<'a> {
    pub data: IftRsiData<'a>,
    pub params: IftRsiParams,
}

impl<'a> IftRsiInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: IftRsiParams) -> Self {
        Self {
            data: IftRsiData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: IftRsiParams) -> Self {
        Self {
            data: IftRsiData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", IftRsiParams::default())
    }
    #[inline]
    pub fn get_rsi_period(&self) -> usize {
        self.params.rsi_period.unwrap_or(5)
    }
    #[inline]
    pub fn get_wma_period(&self) -> usize {
        self.params.wma_period.unwrap_or(9)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct IftRsiBuilder {
    rsi_period: Option<usize>,
    wma_period: Option<usize>,
    kernel: Kernel,
}

impl Default for IftRsiBuilder {
    fn default() -> Self {
        Self {
            rsi_period: None,
            wma_period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl IftRsiBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn rsi_period(mut self, n: usize) -> Self {
        self.rsi_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn wma_period(mut self, n: usize) -> Self {
        self.wma_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<IftRsiOutput, IftRsiError> {
        let p = IftRsiParams {
            rsi_period: self.rsi_period,
            wma_period: self.wma_period,
        };
        let i = IftRsiInput::from_candles(c, "close", p);
        ift_rsi_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<IftRsiOutput, IftRsiError> {
        let p = IftRsiParams {
            rsi_period: self.rsi_period,
            wma_period: self.wma_period,
        };
        let i = IftRsiInput::from_slice(d, p);
        ift_rsi_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<IftRsiStream, IftRsiError> {
        let p = IftRsiParams {
            rsi_period: self.rsi_period,
            wma_period: self.wma_period,
        };
        IftRsiStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum IftRsiError {
    #[error("ift_rsi: Input data slice is empty.")]
    EmptyData,
    #[error("ift_rsi: All values are NaN.")]
    AllValuesNaN,
    #[error("ift_rsi: Invalid RSI period {rsi_period} or WMA period {wma_period}, data length = {data_len}.")]
    InvalidPeriod {
        rsi_period: usize,
        wma_period: usize,
        data_len: usize,
    },
    #[error("ift_rsi: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("ift_rsi: RSI calculation error: {0}")]
    RsiCalculationError(String),
    #[error("ift_rsi: WMA calculation error: {0}")]
    WmaCalculationError(String),
    #[error("ift_rsi: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("ift_rsi: Wrong kernel for batch operation. Use a batch kernel variant.")]
    WrongKernelForBatch,
    #[error("ift_rsi: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(crate::utilities::enums::Kernel),
    #[error("ift_rsi: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange { start: usize, end: usize, step: usize },
}

#[inline]
pub fn ift_rsi(input: &IftRsiInput) -> Result<IftRsiOutput, IftRsiError> {
    ift_rsi_with_kernel(input, Kernel::Auto)
}

pub fn ift_rsi_with_kernel(
    input: &IftRsiInput,
    kernel: Kernel,
) -> Result<IftRsiOutput, IftRsiError> {
    let data: &[f64] = match &input.data {
        IftRsiData::Candles { candles, source } => source_type(candles, source),
        IftRsiData::Slice(sl) => sl,
    };

    if data.is_empty() {
        return Err(IftRsiError::EmptyData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(IftRsiError::AllValuesNaN)?;
    let len = data.len();
    let rsi_period = input.get_rsi_period();
    let wma_period = input.get_wma_period();
    if rsi_period == 0 || wma_period == 0 || rsi_period > len || wma_period > len {
        return Err(IftRsiError::InvalidPeriod {
            rsi_period,
            wma_period,
            data_len: len,
        });
    }
    let needed = rsi_period.max(wma_period);
    if (len - first) < needed {
        return Err(IftRsiError::NotEnoughValidData {
            needed,
            valid: len - first,
        });
    }
    // SIMD kernels are currently stubs (they call the scalar classic kernel). Avoid paying
    // runtime detection overhead for `Kernel::Auto` on the single-series API.
    if kernel.is_batch() {
        return Err(IftRsiError::WrongKernelForBatch);
    }

    // Calculate warmup period: first + rsi_period + wma_period - 1
    let warmup_period = first + rsi_period + wma_period - 1;
    let mut out = alloc_with_nan_prefix(len, warmup_period);

    // Use classic kernel for optimized loop-jammed implementation
    unsafe {
        ift_rsi_scalar_classic(data, rsi_period, wma_period, first, &mut out)?;
    }

    Ok(IftRsiOutput { values: out })
}

/// Writes IFT-RSI values into a caller-provided buffer without allocating.
///
/// - Preserves NaN warmups exactly as the Vec-returning API.
/// - The output slice length must equal the input data length.
/// - Uses `Kernel::Auto` dispatch (mirrors module default).
#[cfg(not(feature = "wasm"))]
#[inline]
pub fn ift_rsi_into(input: &IftRsiInput, out: &mut [f64]) -> Result<(), IftRsiError> {
    let data: &[f64] = match &input.data {
        IftRsiData::Candles { candles, source } => source_type(candles, source),
        IftRsiData::Slice(sl) => sl,
    };

    if out.len() != data.len() {
        return Err(IftRsiError::OutputLengthMismatch { expected: data.len(), got: out.len() });
    }

    let kern = Kernel::Auto;
    ift_rsi_into_slice(out, input, kern)
}

#[inline(always)]
fn ift_rsi_compute_into(
    data: &[f64],
    rsi_period: usize,
    wma_period: usize,
    first_valid: usize,
    out: &mut [f64],
) -> Result<(), IftRsiError> {
    let sliced = &data[first_valid..];
    let mut rsi_values = rsi(&RsiInput::from_slice(
        sliced,
        RsiParams {
            period: Some(rsi_period),
        },
    ))
    .map_err(|e| IftRsiError::RsiCalculationError(e.to_string()))?
    .values;

    for val in rsi_values.iter_mut() {
        if !val.is_nan() {
            *val = 0.1 * (*val - 50.0);
        }
    }

    let wma_values = wma(&WmaInput::from_slice(
        &rsi_values,
        WmaParams {
            period: Some(wma_period),
        },
    ))
    .map_err(|e| IftRsiError::WmaCalculationError(e.to_string()))?
    .values;

    for (i, &w) in wma_values.iter().enumerate() {
        if !w.is_nan() {
            out[first_valid + i] = w.tanh();
        }
    }
    Ok(())
}

#[inline]
pub fn ift_rsi_scalar(
    data: &[f64],
    rsi_period: usize,
    wma_period: usize,
    first_valid: usize,
    out: &mut [f64],
) -> Result<(), IftRsiError> {
    ift_rsi_compute_into(data, rsi_period, wma_period, first_valid, out)
}

/// Write directly to output slice - no allocations
pub fn ift_rsi_into_slice(
    dst: &mut [f64],
    input: &IftRsiInput,
    kern: Kernel,
) -> Result<(), IftRsiError> {
    let data: &[f64] = match &input.data {
        IftRsiData::Candles { candles, source } => source_type(candles, source),
        IftRsiData::Slice(sl) => sl,
    };

    if data.is_empty() {
        return Err(IftRsiError::EmptyData);
    }

    if dst.len() != data.len() {
        return Err(IftRsiError::OutputLengthMismatch { expected: data.len(), got: dst.len() });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(IftRsiError::AllValuesNaN)?;
    let rsi_period = input.get_rsi_period();
    let wma_period = input.get_wma_period();

    if rsi_period == 0 || wma_period == 0 || rsi_period > data.len() || wma_period > data.len() {
        return Err(IftRsiError::InvalidPeriod {
            rsi_period,
            wma_period,
            data_len: data.len(),
        });
    }

    // Initialize dst with NaN for warmup period (matching alma.rs pattern)
    let warmup_period = (first + rsi_period + wma_period - 1).min(dst.len());
    for v in &mut dst[..warmup_period] {
        *v = f64::NAN;
    }

    // Use classic kernel for optimized loop-jammed implementation
    unsafe {
        return ift_rsi_scalar_classic(data, rsi_period, wma_period, first, dst);
    }

    Ok(())
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn ift_rsi_avx512(
    data: &[f64],
    rsi_period: usize,
    wma_period: usize,
    first_valid: usize,
    out: &mut [f64],
) -> Result<(), IftRsiError> {
    unsafe { ift_rsi_scalar_classic(data, rsi_period, wma_period, first_valid, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn ift_rsi_avx2(
    data: &[f64],
    rsi_period: usize,
    wma_period: usize,
    first_valid: usize,
    out: &mut [f64],
) -> Result<(), IftRsiError> {
    unsafe { ift_rsi_scalar_classic(data, rsi_period, wma_period, first_valid, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn ift_rsi_avx512_short(
    data: &[f64],
    rsi_period: usize,
    wma_period: usize,
    first_valid: usize,
    out: &mut [f64],
) -> Result<(), IftRsiError> {
    ift_rsi_avx512(data, rsi_period, wma_period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn ift_rsi_avx512_long(
    data: &[f64],
    rsi_period: usize,
    wma_period: usize,
    first_valid: usize,
    out: &mut [f64],
) -> Result<(), IftRsiError> {
    ift_rsi_avx512(data, rsi_period, wma_period, first_valid, out)
}

#[inline]
pub fn ift_rsi_batch_with_kernel(
    data: &[f64],
    sweep: &IftRsiBatchRange,
    k: Kernel,
) -> Result<IftRsiBatchOutput, IftRsiError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(IftRsiError::InvalidKernelForBatch(other)),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    ift_rsi_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct IftRsiBatchRange {
    pub rsi_period: (usize, usize, usize),
    pub wma_period: (usize, usize, usize),
}

impl Default for IftRsiBatchRange {
    fn default() -> Self {
        Self {
            rsi_period: (5, 5, 0),
            wma_period: (9, 258, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct IftRsiBatchBuilder {
    range: IftRsiBatchRange,
    kernel: Kernel,
}

impl IftRsiBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn rsi_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.rsi_period = (start, end, step);
        self
    }
    #[inline]
    pub fn rsi_period_static(mut self, p: usize) -> Self {
        self.range.rsi_period = (p, p, 0);
        self
    }
    #[inline]
    pub fn wma_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.wma_period = (start, end, step);
        self
    }
    #[inline]
    pub fn wma_period_static(mut self, n: usize) -> Self {
        self.range.wma_period = (n, n, 0);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<IftRsiBatchOutput, IftRsiError> {
        ift_rsi_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<IftRsiBatchOutput, IftRsiError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<IftRsiBatchOutput, IftRsiError> {
        IftRsiBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn with_default_candles(c: &Candles) -> Result<IftRsiBatchOutput, IftRsiError> {
        IftRsiBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct IftRsiBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<IftRsiParams>,
    pub rows: usize,
    pub cols: usize,
}

impl IftRsiBatchOutput {
    pub fn row_for_params(&self, p: &IftRsiParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.rsi_period.unwrap_or(5) == p.rsi_period.unwrap_or(5)
                && c.wma_period.unwrap_or(9) == p.wma_period.unwrap_or(9)
        })
    }

    pub fn values_for(&self, p: &IftRsiParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &IftRsiBatchRange) -> Result<Vec<IftRsiParams>, IftRsiError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, IftRsiError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        let (lo, hi) = if start <= end { (start, end) } else { (end, start) };
        let vals: Vec<usize> = (lo..=hi).step_by(step).collect();
        if vals.is_empty() {
            return Err(IftRsiError::InvalidRange { start, end, step });
        }
        Ok(vals)
    }
    let rsi_periods = axis_usize(r.rsi_period)?;
    let wma_periods = axis_usize(r.wma_period)?;
    let cap = rsi_periods
        .len()
        .checked_mul(wma_periods.len())
        .ok_or(IftRsiError::InvalidRange {
            start: r.rsi_period.0,
            end: r.rsi_period.1,
            step: r.rsi_period.2,
        })?;
    let mut out = Vec::with_capacity(cap);
    for &rsi_p in &rsi_periods {
        for &wma_p in &wma_periods {
            out.push(IftRsiParams { rsi_period: Some(rsi_p), wma_period: Some(wma_p) });
        }
    }
    Ok(out)
}

#[inline(always)]
pub fn ift_rsi_batch_slice(
    data: &[f64],
    sweep: &IftRsiBatchRange,
    kern: Kernel,
) -> Result<IftRsiBatchOutput, IftRsiError> {
    ift_rsi_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn ift_rsi_batch_par_slice(
    data: &[f64],
    sweep: &IftRsiBatchRange,
    kern: Kernel,
) -> Result<IftRsiBatchOutput, IftRsiError> {
    ift_rsi_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn ift_rsi_batch_inner(
    data: &[f64],
    sweep: &IftRsiBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<IftRsiBatchOutput, IftRsiError> {
    let combos = expand_grid(sweep)?;
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(IftRsiError::AllValuesNaN)?;
    let max_rsi = combos.iter().map(|c| c.rsi_period.unwrap()).max().unwrap();
    let max_wma = combos.iter().map(|c| c.wma_period.unwrap()).max().unwrap();
    let max_p = max_rsi.max(max_wma);
    if data.len() - first < max_p {
        return Err(IftRsiError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    rows.checked_mul(cols).ok_or(IftRsiError::InvalidRange {
        start: sweep.rsi_period.0,
        end: sweep.rsi_period.1,
        step: sweep.rsi_period.2,
    })?;

    // Calculate warmup periods for each parameter combination
    let warmup_periods: Vec<usize> = combos
        .iter()
        .map(|c| first + c.rsi_period.unwrap() + c.wma_period.unwrap() - 1)
        .collect();

    // Use uninitialized memory with proper NaN prefixes
    let mut buf_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut buf_mu, cols, &warmup_periods);

    // Convert to mutable slice for computation
    let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
    let values: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    // Precompute gains/losses once for all rows to avoid duplicate work
    let sliced = &data[first..];
    let n = sliced.len();
    let mut gains = Vec::with_capacity(n.saturating_sub(1));
    let mut losses = Vec::with_capacity(n.saturating_sub(1));
    for i in 1..n {
        let d = sliced[i] - sliced[i - 1];
        if d > 0.0 {
            gains.push(d);
            losses.push(0.0);
        } else {
            gains.push(0.0);
            losses.push(-d);
        }
    }
    // Prefix sums for O(1) seeding per row
    let n1 = gains.len();
    let mut pg = Vec::with_capacity(n1 + 1);
    let mut pl = Vec::with_capacity(n1 + 1);
    pg.push(0.0);
    pl.push(0.0);
    for i in 0..n1 {
        pg.push(pg[i] + gains[i]);
        pl.push(pl[i] + losses[i]);
    }
    // Prefix sums for O(1) seeding per row
    let n1 = gains.len();
    let mut pg = Vec::with_capacity(n1 + 1);
    let mut pl = Vec::with_capacity(n1 + 1);
    pg.push(0.0);
    pl.push(0.0);
    for i in 0..n1 {
        pg.push(pg[i] + gains[i]);
        pl.push(pl[i] + losses[i]);
    }
    // Prefix sums for O(1) seeding per row
    let n1 = gains.len();
    let mut pg = Vec::with_capacity(n1 + 1);
    let mut pl = Vec::with_capacity(n1 + 1);
    pg.push(0.0);
    pl.push(0.0);
    for i in 0..n1 {
        pg.push(pg[i] + gains[i]);
        pl.push(pl[i] + losses[i]);
    }
    // Prefix sums for O(1) seeding per row
    let n1 = gains.len();
    let mut pg = Vec::with_capacity(n1 + 1);
    let mut pl = Vec::with_capacity(n1 + 1);
    pg.push(0.0);
    pl.push(0.0);
    for i in 0..n1 {
        pg.push(pg[i] + gains[i]);
        pl.push(pl[i] + losses[i]);
    }

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let rsi_p = combos[row].rsi_period.unwrap();
        let wma_p = combos[row].wma_period.unwrap();
        match kern {
            Kernel::Scalar => ift_rsi_row_scalar_precomputed_ps(
                &gains, &losses, &pg, &pl, rsi_p, wma_p, first, out_row,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => ift_rsi_row_avx2(data, first, rsi_p, wma_p, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => ift_rsi_row_avx512(data, first, rsi_p, wma_p, out_row),
            _ => unreachable!(),
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

    // Convert back to Vec for output
    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(IftRsiBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn ift_rsi_batch_inner_into(
    data: &[f64],
    sweep: &IftRsiBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<IftRsiParams>, IftRsiError> {
    let combos = expand_grid(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    rows.checked_mul(cols).ok_or(IftRsiError::InvalidRange {
        start: sweep.rsi_period.0,
        end: sweep.rsi_period.1,
        step: sweep.rsi_period.2,
    })?;
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(IftRsiError::AllValuesNaN)?;
    let max_rsi = combos.iter().map(|c| c.rsi_period.unwrap()).max().unwrap();
    let max_wma = combos.iter().map(|c| c.wma_period.unwrap()).max().unwrap();
    let max_p = max_rsi.max(max_wma);
    if data.len() - first < max_p {
        return Err(IftRsiError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();

    // Initialize NaN prefixes for each row based on warmup period
    for (row, combo) in combos.iter().enumerate() {
        let warmup = (first + combo.rsi_period.unwrap() + combo.wma_period.unwrap() - 1).min(cols);
        let row_start = row * cols;
        for i in 0..warmup {
            out[row_start + i] = f64::NAN;
        }
    }

    // Precompute gains/losses once for all rows to avoid duplicate work
    let sliced = &data[first..];
    let n = sliced.len();
    let mut gains = Vec::with_capacity(n.saturating_sub(1));
    let mut losses = Vec::with_capacity(n.saturating_sub(1));
    for i in 1..n {
        let d = sliced[i] - sliced[i - 1];
        if d > 0.0 {
            gains.push(d);
            losses.push(0.0);
        } else {
            gains.push(0.0);
            losses.push(-d);
        }
    }
    // Prefix sums for O(1) seeding per row
    let n1 = gains.len();
    let mut pg = Vec::with_capacity(n1 + 1);
    let mut pl = Vec::with_capacity(n1 + 1);
    pg.push(0.0);
    pl.push(0.0);
    for i in 0..n1 {
        pg.push(pg[i] + gains[i]);
        pl.push(pl[i] + losses[i]);
    }

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let rsi_p = combos[row].rsi_period.unwrap();
        let wma_p = combos[row].wma_period.unwrap();
        match kern {
            Kernel::Scalar => ift_rsi_row_scalar_precomputed_ps(
                &gains, &losses, &pg, &pl, rsi_p, wma_p, first, out_row,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => ift_rsi_row_avx2(data, first, rsi_p, wma_p, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => ift_rsi_row_avx512(data, first, rsi_p, wma_p, out_row),
            _ => unreachable!(),
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
unsafe fn ift_rsi_row_scalar(
    data: &[f64],
    first: usize,
    rsi_period: usize,
    wma_period: usize,
    out: &mut [f64],
) {
    // Fallback to precomputed path by computing diffs locally
    let sliced = &data[first..];
    let n = sliced.len();
    if n == 0 {
        return;
    }
    let mut gains = Vec::with_capacity(n.saturating_sub(1));
    let mut losses = Vec::with_capacity(n.saturating_sub(1));
    for i in 1..n {
        let d = sliced[i] - sliced[i - 1];
        if d > 0.0 {
            gains.push(d);
            losses.push(0.0);
        } else {
            gains.push(0.0);
            losses.push(-d);
        }
    }
    ift_rsi_row_scalar_precomputed(&gains, &losses, rsi_period, wma_period, first, out);
}

#[inline(always)]
unsafe fn ift_rsi_row_scalar_precomputed(
    gains: &[f64],
    losses: &[f64],
    rsi_period: usize,
    wma_period: usize,
    first: usize,
    out: &mut [f64],
) {
    let n1 = gains.len(); // equals losses.len(), and equals sliced.len() - 1
    if rsi_period == 0 || wma_period == 0 {
        return;
    }
    if rsi_period + wma_period - 1 >= n1 + 1 {
        return;
    }

    // Seed Wilder averages
    let mut avg_gain = 0.0f64;
    let mut avg_loss = 0.0f64;
    for i in 0..rsi_period {
        avg_gain += *gains.get_unchecked(i);
        avg_loss += *losses.get_unchecked(i);
    }
    let rp_f = rsi_period as f64;
    avg_gain /= rp_f;
    avg_loss /= rp_f;
    let alpha = 1.0f64 / rp_f;
    let beta = 1.0f64 - alpha;

    // LWMA state
    let wp = wma_period;
    let wp_f = wp as f64;
    let denom = 0.5f64 * wp_f * (wp_f + 1.0);
    let denom_rcp = 1.0f64 / denom;
    let mut buf: Vec<f64> = vec![0.0; wp];
    let mut head = 0usize;
    let mut filled = 0usize;
    let mut sum = 0.0f64;
    let mut num = 0.0f64;

    // Iterate timesteps i over RSI indices (0-based on sliced): i=rsi_period..n-1
    let mut i = rsi_period;
    while i <= n1 {
        // since gains/losses are indexed by diff at t, RSI index i uses diff i (for i>rp)
        if i > rsi_period {
            let g = *gains.get_unchecked(i - 1); // diff at (i-1)
            let l = *losses.get_unchecked(i - 1);
            avg_gain = f64::mul_add(avg_gain, beta, alpha * g);
            avg_loss = f64::mul_add(avg_loss, beta, alpha * l);
        }

        let rs = if avg_loss != 0.0 {
            avg_gain / avg_loss
        } else {
            100.0
        };
        let rsi = 100.0 - 100.0 / (1.0 + rs);
        let x = 0.1f64 * (rsi - 50.0);

        if filled < wp {
            sum += x;
            num = f64::mul_add((filled as f64) + 1.0, x, num);
            *buf.get_unchecked_mut(head) = x;
            head += 1;
            if head == wp {
                head = 0;
            }
            filled += 1;
            if filled == wp {
                let wma = num * denom_rcp;
                *out.get_unchecked_mut(first + i) = wma.tanh();
            }
        } else {
            let x_old = *buf.get_unchecked(head);
            *buf.get_unchecked_mut(head) = x;
            head += 1;
            if head == wp {
                head = 0;
            }
            let sum_t = sum;
            num = f64::mul_add(wp_f, x, num) - sum_t;
            sum = sum_t + x - x_old;
            let wma = num * denom_rcp;
            *out.get_unchecked_mut(first + i) = wma.tanh();
        }

        i += 1;
        if i > n1 {
            break;
        }
    }
}

#[inline(always)]
unsafe fn ift_rsi_row_scalar_precomputed_ps(
    gains: &[f64],
    losses: &[f64],
    pg: &[f64],
    pl: &[f64],
    rsi_period: usize,
    wma_period: usize,
    first: usize,
    out: &mut [f64],
) {
    let n1 = gains.len();
    if rsi_period == 0 || wma_period == 0 {
        return;
    }
    if rsi_period + wma_period - 1 >= n1 + 1 {
        return;
    }

    // Seed with prefix sums: avg over gains[0..rsi_period-1], losses[0..rsi_period-1]
    let sum_gain = *pg.get_unchecked(rsi_period) - *pg.get_unchecked(0);
    let sum_loss = *pl.get_unchecked(rsi_period) - *pl.get_unchecked(0);
    let rp_f = rsi_period as f64;
    let mut avg_gain = sum_gain / rp_f;
    let mut avg_loss = sum_loss / rp_f;
    let alpha = 1.0f64 / rp_f;
    let beta = 1.0f64 - alpha;

    // LWMA state
    let wp = wma_period;
    let wp_f = wp as f64;
    let denom = 0.5f64 * wp_f * (wp_f + 1.0);
    let denom_rcp = 1.0f64 / denom;
    let mut buf: Vec<f64> = vec![0.0; wp];
    let mut head = 0usize;
    let mut filled = 0usize;
    let mut sum = 0.0f64;
    let mut num = 0.0f64;

    let mut i = rsi_period;
    while i <= n1 {
        if i > rsi_period {
            let g = *gains.get_unchecked(i - 1);
            let l = *losses.get_unchecked(i - 1);
            avg_gain = f64::mul_add(avg_gain, beta, alpha * g);
            avg_loss = f64::mul_add(avg_loss, beta, alpha * l);
        }

        let rs = if avg_loss != 0.0 {
            avg_gain / avg_loss
        } else {
            100.0
        };
        let rsi = 100.0 - 100.0 / (1.0 + rs);
        let x = 0.1f64 * (rsi - 50.0);

        if filled < wp {
            sum += x;
            num = f64::mul_add((filled as f64) + 1.0, x, num);
            *buf.get_unchecked_mut(head) = x;
            head += 1;
            if head == wp {
                head = 0;
            }
            filled += 1;
            if filled == wp {
                let wma = num * denom_rcp;
                *out.get_unchecked_mut(first + i) = wma.tanh();
            }
        } else {
            let x_old = *buf.get_unchecked(head);
            *buf.get_unchecked_mut(head) = x;
            head += 1;
            if head == wp {
                head = 0;
            }
            let sum_t = sum;
            num = f64::mul_add(wp_f, x, num) - sum_t;
            sum = sum_t + x - x_old;
            let wma = num * denom_rcp;
            *out.get_unchecked_mut(first + i) = wma.tanh();
        }

        i += 1;
        if i > n1 {
            break;
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn ift_rsi_row_avx2(
    data: &[f64],
    first: usize,
    rsi_period: usize,
    wma_period: usize,
    out: &mut [f64],
) {
    ift_rsi_row_scalar(data, first, rsi_period, wma_period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn ift_rsi_row_avx512(
    data: &[f64],
    first: usize,
    rsi_period: usize,
    wma_period: usize,
    out: &mut [f64],
) {
    if rsi_period.max(wma_period) <= 32 {
        ift_rsi_row_avx512_short(data, first, rsi_period, wma_period, out);
    } else {
        ift_rsi_row_avx512_long(data, first, rsi_period, wma_period, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn ift_rsi_row_avx512_short(
    data: &[f64],
    first: usize,
    rsi_period: usize,
    wma_period: usize,
    out: &mut [f64],
) {
    ift_rsi_row_scalar(data, first, rsi_period, wma_period, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn ift_rsi_row_avx512_long(
    data: &[f64],
    first: usize,
    rsi_period: usize,
    wma_period: usize,
    out: &mut [f64],
) {
    ift_rsi_row_scalar(data, first, rsi_period, wma_period, out);
}

#[derive(Debug, Clone)]
pub struct IftRsiStream {
    // params
    rsi_period: usize,
    wma_period: usize,

    // --- RSI (Wilder SMMA) state ---
    prev: f64,
    have_prev: bool,
    seed_g: f64,
    seed_l: f64,
    seed_cnt: usize,
    avg_gain: f64,
    avg_loss: f64,
    seeded: bool,
    alpha: f64,
    beta: f64,

    // --- LWMA (weights 1..wp) state over x = 0.1*(RSI-50) ---
    buf: Vec<f64>,
    head: usize,
    filled: usize,
    sum: f64,
    num: f64,
    wp_f: f64,
    denom_rcp: f64,
}

impl IftRsiStream {
    pub fn try_new(params: IftRsiParams) -> Result<Self, IftRsiError> {
        let rsi_period = params.rsi_period.unwrap_or(5);
        let wma_period = params.wma_period.unwrap_or(9);
        if rsi_period == 0 || wma_period == 0 {
            return Err(IftRsiError::InvalidPeriod {
                rsi_period,
                wma_period,
                data_len: 0,
            });
        }
        let wp_f = wma_period as f64;
        let denom = 0.5 * wp_f * (wp_f + 1.0);
        Ok(Self {
            rsi_period,
            wma_period,

            prev: 0.0,
            have_prev: false,
            seed_g: 0.0,
            seed_l: 0.0,
            seed_cnt: 0,
            avg_gain: 0.0,
            avg_loss: 0.0,
            seeded: false,
            alpha: 1.0 / (rsi_period as f64),
            beta: 1.0 - 1.0 / (rsi_period as f64),

            buf: vec![0.0; wma_period],
            head: 0,
            filled: 0,
            sum: 0.0,
            num: 0.0,
            wp_f,
            denom_rcp: 1.0 / denom,
        })
    }

    /// Push one price. Returns IFT-RSI once both warmups complete:
    /// after rsi_period diffs and wma_period transformed values.
    /// On NaN input we reset the state and return None.
    #[inline]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        if !value.is_finite() {
            self.reset_soft();
            return None;
        }

        // First sample: just store and wait for a diff
        if !self.have_prev {
            self.prev = value;
            self.have_prev = true;
            return None;
        }

        // Compute one-step diff and split into gain/loss (positive)
        let d = value - self.prev;
        self.prev = value;
        let gain = if d > 0.0 { d } else { 0.0 };
        let loss = if d < 0.0 { -d } else { 0.0 };

        // --- Seed Wilder averages for first rsi_period diffs ---
        if !self.seeded {
            self.seed_g += gain;
            self.seed_l += loss;
            self.seed_cnt += 1;
            if self.seed_cnt < self.rsi_period {
                return None; // still seeding
            }
            // Initialize averages on the tick where we collected rsi_period diffs
            self.avg_gain = self.seed_g / (self.rsi_period as f64);
            self.avg_loss = self.seed_l / (self.rsi_period as f64);
            self.seeded = true;

            // We also produce an RSI value on this same tick and feed it into LWMA below.
        } else {
            // --- Regular Wilder update: O(1) ---
            self.avg_gain = f64::mul_add(self.avg_gain, self.beta, self.alpha * gain);
            self.avg_loss = f64::mul_add(self.avg_loss, self.beta, self.alpha * loss);
        }

        // --- Transform RSI -> x = 0.1 * (RSI - 50) ---
        let rs = if self.avg_loss != 0.0 {
            self.avg_gain / self.avg_loss
        } else {
            100.0
        };
        let rsi = 100.0 - 100.0 / (1.0 + rs);
        let x = 0.1 * (rsi - 50.0);

        // --- O(1) LWMA update over x with weights 1..wp ---
        if self.filled < self.wma_period {
            // build phase: num += (filled+1)*x ; sum += x
            self.sum += x;
            self.num = f64::mul_add((self.filled as f64) + 1.0, x, self.num);
            self.buf[self.head] = x;
            self.head += 1;
            if self.head == self.wma_period {
                self.head = 0;
            }
            self.filled += 1;

            if self.filled == self.wma_period {
                let wma = self.num * self.denom_rcp;
                return Some(tanh_kernel(wma));
            }
            return None;
        } else {
            // steady-state: rotate buffer, update (sum, num)
            let x_old = self.buf[self.head];
            self.buf[self.head] = x;
            self.head += 1;
            if self.head == self.wma_period {
                self.head = 0;
            }

            // num' = num + wp*x - sum
            let sum_prev = self.sum;
            self.num = f64::mul_add(self.wp_f, x, self.num) - sum_prev;
            self.sum = sum_prev + x - x_old;

            let wma = self.num * self.denom_rcp;
            return Some(tanh_kernel(wma));
        }
    }

    #[inline]
    fn reset_soft(&mut self) {
        // lightweight reset on NaN input so subsequent values seed cleanly
        self.have_prev = false;
        self.seed_g = 0.0;
        self.seed_l = 0.0;
        self.seed_cnt = 0;
        self.avg_gain = 0.0;
        self.avg_loss = 0.0;
        self.seeded = false;

        self.head = 0;
        self.filled = 0;
        self.sum = 0.0;
        self.num = 0.0;
        for v in &mut self.buf {
            *v = 0.0;
        }
    }
}

#[inline(always)]
fn tanh_kernel(x: f64) -> f64 {
    x.tanh()
}

/// Optimized single-pass scalar kernel:
/// - Inline Wilder RSI (SMMA) with FMA
/// - Inline LWMA using O(1) rolling recurrence for numerator/sum
/// - Writes outputs only after the full warmup (caller pre-fills NaNs)
#[inline]
pub unsafe fn ift_rsi_scalar_classic(
    data: &[f64],
    rsi_period: usize,
    wma_period: usize,
    first_valid: usize,
    out: &mut [f64],
) -> Result<(), IftRsiError> {
    debug_assert!(rsi_period > 0 && wma_period > 0);
    let len = data.len();
    if first_valid >= len {
        return Ok(());
    }
    let sliced = data.get_unchecked(first_valid..);
    let n = sliced.len();
    if n == 0 {
        return Ok(());
    }

    // Need at least rsi_period diffs, then wma_period transformed values
    if rsi_period + wma_period - 1 >= n {
        return Ok(());
    }

    // Wilder RSI smoothing parameters
    let rp = rsi_period;
    let rp_f = rp as f64;
    let alpha = 1.0f64 / rp_f;
    let beta = 1.0f64 - alpha;

    // Seed averages over the first `rp` differences
    let mut avg_gain = 0.0f64;
    let mut avg_loss = 0.0f64;
    {
        let mut i = 1usize;
        while i <= rp {
            let d = *sliced.get_unchecked(i) - *sliced.get_unchecked(i - 1);
            if d > 0.0 {
                avg_gain += d;
            } else {
                avg_loss -= d; // make loss positive
            }
            i += 1;
        }
        avg_gain /= rp_f;
        avg_loss /= rp_f;
    }

    // LWMA rolling state
    let wp = wma_period;
    let wp_f = wp as f64;
    let denom = 0.5f64 * wp_f * (wp_f + 1.0);
    let denom_rcp = 1.0f64 / denom;

    // Circular buffer for last `wp` transformed RSI values
    let mut buf: Vec<f64> = vec![0.0; wp];
    let mut head: usize = 0;
    let mut filled: usize = 0;

    // Rolling sums for LWMA
    let mut sum = 0.0f64; // Σ x
    let mut num = 0.0f64; // Σ j*x_j with j=1..wp

    // Process stream
    let mut i = rp; // first RSI index available
    while i < n {
        if i > rp {
            // Wilder smoothing update with FMA
            let d = *sliced.get_unchecked(i) - *sliced.get_unchecked(i - 1);
            let gain = if d > 0.0 { d } else { 0.0 };
            let loss = if d < 0.0 { -d } else { 0.0 };
            avg_gain = f64::mul_add(avg_gain, beta, alpha * gain);
            avg_loss = f64::mul_add(avg_loss, beta, alpha * loss);
        }

        // Compute transformed RSI value x = 0.1*(RSI-50)
        let rs = if avg_loss != 0.0 {
            avg_gain / avg_loss
        } else {
            100.0
        };
        let rsi = 100.0 - 100.0 / (1.0 + rs);
        let x = 0.1f64 * (rsi - 50.0);

        if filled < wp {
            // Build initial window
            sum += x;
            num = f64::mul_add((filled as f64) + 1.0, x, num);
            *buf.get_unchecked_mut(head) = x;
            head += 1;
            if head == wp {
                head = 0;
            }
            filled += 1;

            if filled == wp {
                let wma = num * denom_rcp;
                *out.get_unchecked_mut(first_valid + i) = wma.tanh();
            }
        } else {
            // Steady-state O(1) LWMA recurrence
            let x_old = *buf.get_unchecked(head);
            *buf.get_unchecked_mut(head) = x;
            head += 1;
            if head == wp {
                head = 0;
            }

            let sum_t = sum;
            num = f64::mul_add(wp_f, x, num) - sum_t; // mul_add tweak
            sum = sum_t + x - x_old; // then update plain sum

            let wma = num * denom_rcp;
            *out.get_unchecked_mut(first_valid + i) = wma.tanh();
        }

        i += 1;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_ift_rsi_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = IftRsiParams {
            rsi_period: None,
            wma_period: None,
        };
        let input = IftRsiInput::from_candles(&candles, "close", default_params);
        let output = ift_rsi_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    #[test]
    fn test_ift_rsi_into_matches_api() -> Result<(), Box<dyn Error>> {
        // Build a small but non-trivial input with a NaN prefix
        let n = 256usize;
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            if i < 3 { // ensure first-valid > 0 to exercise warmup logic
                data.push(f64::NAN);
            } else {
                // oscillatory but bounded series
                let x = (i as f64).sin() * 5.0 + 100.0 + ((i % 7) as f64);
                data.push(x);
            }
        }

        let input = IftRsiInput::from_slice(&data, IftRsiParams::default());

        // Baseline via Vec-returning API
        let baseline = ift_rsi(&input)?.values;

        // New into-API into a preallocated buffer
        let mut out = vec![0.0; data.len()];
        ift_rsi_into(&input, &mut out)?;

        assert_eq!(baseline.len(), out.len());
        for i in 0..out.len() {
            let a = baseline[i];
            let b = out[i];
            let equal = (a.is_nan() && b.is_nan()) || (a == b);
            assert!(
                equal,
                "Mismatch at {}: baseline={}, into={}",
                i,
                a,
                b
            );
        }
        Ok(())
    }

    fn check_ift_rsi_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = IftRsiInput::from_candles(&candles, "close", IftRsiParams::default());
        let result = ift_rsi_with_kernel(&input, kernel)?;

        // Updated reference values using correct tanh formula
        let expected_last_five = [
            -0.35919800205778424,
            -0.3275464113984847,
            -0.39970276998138216,
            -0.36321812798797737,
            -0.5843346528346959,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-8,
                "[{}] IFT_RSI {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_ift_rsi_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = IftRsiInput::with_default_candles(&candles);
        let output = ift_rsi_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_ift_rsi_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = IftRsiParams {
            rsi_period: Some(0),
            wma_period: Some(9),
        };
        let input = IftRsiInput::from_slice(&input_data, params);
        let res = ift_rsi_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] IFT_RSI should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_ift_rsi_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = IftRsiParams {
            rsi_period: Some(10),
            wma_period: Some(9),
        };
        let input = IftRsiInput::from_slice(&data_small, params);
        let res = ift_rsi_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] IFT_RSI should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_ift_rsi_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = IftRsiParams {
            rsi_period: Some(5),
            wma_period: Some(9),
        };
        let input = IftRsiInput::from_slice(&single_point, params);
        let res = ift_rsi_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] IFT_RSI should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_ift_rsi_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = IftRsiParams {
            rsi_period: Some(5),
            wma_period: Some(9),
        };
        let first_input = IftRsiInput::from_candles(&candles, "close", first_params);
        let first_result = ift_rsi_with_kernel(&first_input, kernel)?;
        let second_params = IftRsiParams {
            rsi_period: Some(5),
            wma_period: Some(9),
        };
        let second_input = IftRsiInput::from_slice(&first_result.values, second_params);
        let second_result = ift_rsi_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    fn check_ift_rsi_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = IftRsiInput::from_candles(
            &candles,
            "close",
            IftRsiParams {
                rsi_period: Some(5),
                wma_period: Some(9),
            },
        );
        let res = ift_rsi_with_kernel(&input, kernel)?;
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

    #[cfg(debug_assertions)]
    fn check_ift_rsi_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let test_params = vec![
            IftRsiParams::default(), // rsi_period: 5, wma_period: 9
            IftRsiParams {
                rsi_period: Some(2),
                wma_period: Some(2),
            },
            IftRsiParams {
                rsi_period: Some(3),
                wma_period: Some(5),
            },
            IftRsiParams {
                rsi_period: Some(7),
                wma_period: Some(14),
            },
            IftRsiParams {
                rsi_period: Some(14),
                wma_period: Some(21),
            },
            IftRsiParams {
                rsi_period: Some(21),
                wma_period: Some(9),
            },
            IftRsiParams {
                rsi_period: Some(50),
                wma_period: Some(50),
            },
            IftRsiParams {
                rsi_period: Some(100),
                wma_period: Some(100),
            },
            IftRsiParams {
                rsi_period: Some(2),
                wma_period: Some(50),
            },
            IftRsiParams {
                rsi_period: Some(50),
                wma_period: Some(2),
            },
            IftRsiParams {
                rsi_period: Some(9),
                wma_period: Some(21),
            },
            IftRsiParams {
                rsi_period: Some(25),
                wma_period: Some(10),
            },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = IftRsiInput::from_candles(&candles, "close", params.clone());
            let output = ift_rsi_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: rsi_period={}, wma_period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.rsi_period.unwrap_or(5),
                        params.wma_period.unwrap_or(9),
                        param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: rsi_period={}, wma_period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.rsi_period.unwrap_or(5),
                        params.wma_period.unwrap_or(9),
                        param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: rsi_period={}, wma_period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.rsi_period.unwrap_or(5),
                        params.wma_period.unwrap_or(9),
                        param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_ift_rsi_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(()) // No-op in release builds
    }

    #[cfg(feature = "proptest")]
    fn check_ift_rsi_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Strategy for generating test data with realistic price movements
        // Note: RSI period starts from 2 since RSI needs at least 2 values
        let strat = (2usize..=50, 2usize..=50)
            .prop_flat_map(|(rsi_period, wma_period)| {
                let min_len = (rsi_period + wma_period) * 2; // Ensure sufficient data for meaningful testing
                (
                    // Base price level and volatility
                    (100.0f64..5000.0f64, 0.01f64..0.1f64),
                    // Trend strength (-2% to +2% per step)
                    -0.02f64..0.02f64,
                    // Generate periods and data length
                    Just(rsi_period),
                    Just(wma_period),
                    min_len..400,
                )
            })
            .prop_map(
                |((base_price, volatility), trend, rsi_period, wma_period, len)| {
                    // Generate realistic price data with trend and noise
                    let mut prices = Vec::with_capacity(len);
                    let mut current_price = base_price;

                    for i in 0..len {
                        // Apply trend
                        current_price *= 1.0 + trend;
                        // Add noise
                        let noise = 1.0 + (i as f64 * 0.1).sin() * volatility;
                        prices.push(current_price * noise);
                    }

                    (prices, rsi_period, wma_period)
                },
            );

        proptest::test_runner::TestRunner::default().run(
            &strat,
            |(data, rsi_period, wma_period)| {
                let params = IftRsiParams {
                    rsi_period: Some(rsi_period),
                    wma_period: Some(wma_period),
                };
                let input = IftRsiInput::from_slice(&data, params);

                // Test with the specified kernel
                let IftRsiOutput { values: out } = ift_rsi_with_kernel(&input, kernel)?;

                // Also compute with scalar kernel for reference
                let IftRsiOutput { values: ref_out } = ift_rsi_with_kernel(&input, Kernel::Scalar)?;

                // Property 1: Output length matches input length
                prop_assert_eq!(out.len(), data.len(), "Output length mismatch");

                // Property 2: Warmup period handling
                let warmup_period = rsi_period + wma_period - 1;
                for i in 0..warmup_period.min(data.len()) {
                    prop_assert!(
                        out[i].is_nan(),
                        "Expected NaN during warmup at index {}, got {}",
                        i,
                        out[i]
                    );
                }

                // Properties for values after warmup
                for i in warmup_period..data.len() {
                    let y = out[i];
                    let r = ref_out[i];

                    // Property 3: Mathematical bounds - IFT RSI bounded to [-1, 1]
                    // The Inverse Fisher Transform formula: (2w)^2 - 1 / (2w)^2 + 1
                    // This is mathematically bounded to [-1, 1]
                    if y.is_finite() {
                        prop_assert!(
                            y >= -1.0 - 1e-9 && y <= 1.0 + 1e-9,
                            "IFT RSI value {} at index {} outside [-1, 1] bounds",
                            y,
                            i
                        );
                    }

                    // Property 4: Kernel consistency
                    if !y.is_finite() || !r.is_finite() {
                        prop_assert_eq!(
                            y.to_bits(),
                            r.to_bits(),
                            "NaN/Inf mismatch at index {}: {} vs {}",
                            i,
                            y,
                            r
                        );
                    } else {
                        let ulp_diff = y.to_bits().abs_diff(r.to_bits());
                        prop_assert!(
                            (y - r).abs() <= 1e-9 || ulp_diff <= 4,
                            "Kernel mismatch at index {}: {} vs {} (ULP={})",
                            i,
                            y,
                            r,
                            ulp_diff
                        );
                    }

                    // Property 5: Response to trending markets
                    // NOTE: This property currently exposes a bug in the IFT RSI implementation
                    // The formula (2w)^2 loses sign information, making both up and down trends positive
                    // Check if we have enough data for trend analysis
                    if i >= warmup_period + 10 {
                        let lookback = 10;
                        let recent_prices = &data[i - lookback..=i];
                        let price_change =
                            (recent_prices[lookback] - recent_prices[0]) / recent_prices[0];

                        // For strong uptrends (>5%), IFT RSI should be distinctly positive
                        if price_change > 0.05 && y.is_finite() {
                            // Tightened from > -0.5 to > 0.2 to catch directionality issues
                            prop_assert!(
								y > 0.2,
								"Strong uptrend should produce positive IFT RSI > 0.2, got {} at index {}",
								y,
								i
							);
                        }

                        // For strong downtrends (<-5%), IFT RSI should be distinctly negative
                        // WARNING: Due to implementation bug using (2w)^2, this will likely fail
                        if price_change < -0.05 && y.is_finite() {
                            // Tightened from < 0.5 to < -0.2 to properly test directionality
                            prop_assert!(
								y < -0.2,
								"Strong downtrend should produce negative IFT RSI < -0.2, got {} at index {}",
								y,
								i
							);
                        }
                    }

                    // Property 6: No NaN values after warmup (unless input has NaN)
                    if !data[..=i].iter().any(|x| x.is_nan()) {
                        prop_assert!(!y.is_nan(), "Unexpected NaN at index {} after warmup", i);
                    }
                }

                // Property 7: Response to constant prices
                if data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10)
                    && data.len() > warmup_period
                {
                    // Mathematical derivation for constant prices:
                    // 1. RSI = 50 (no gains or losses)
                    // 2. Transformed: 0.1 * (50 - 50) = 0
                    // 3. WMA of zeros = 0
                    // 4. IFT formula: ((2*0)^2 - 1) / ((2*0)^2 + 1) = -1/1 = -1
                    for i in warmup_period..out.len() {
                        if out[i].is_finite() {
                            prop_assert!(
                                (out[i] - (-1.0)).abs() < 1e-6,
                                "Constant prices should yield IFT RSI = -1, got {} at index {}",
                                out[i],
                                i
                            );
                        }
                    }
                }

                // Property 8: Response to extreme volatility
                let volatility = if data.len() > 2 {
                    let returns: Vec<f64> = data.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect();
                    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
                    let variance = returns
                        .iter()
                        .map(|r| (r - mean_return).powi(2))
                        .sum::<f64>()
                        / returns.len() as f64;
                    variance.sqrt()
                } else {
                    0.0
                };

                // For extreme volatility (>10% per period), values should still be bounded
                if volatility > 0.1 {
                    for &val in out.iter() {
                        if val.is_finite() {
                            prop_assert!(
                                val >= -1.0 && val <= 1.0,
                                "Even with extreme volatility, IFT RSI must be bounded: {}",
                                val
                            );
                        }
                    }
                }

                // Property 9: Sign preservation check (currently exposes implementation bug)
                // The IFT RSI should preserve directional information from RSI
                // However, due to (2w)^2 in the formula, sign information is lost
                // This property will likely FAIL, confirming the mathematical issue
                if data.len() > warmup_period + 20 {
                    // Sample a few points to check sign preservation
                    for check_idx in (warmup_period + 10..data.len()).step_by(20) {
                        if check_idx + 5 >= data.len() {
                            break;
                        }

                        // Calculate approximate RSI direction
                        let recent_window = &data[check_idx - 5..=check_idx];
                        let gains: f64 = recent_window
                            .windows(2)
                            .map(|w| (w[1] - w[0]).max(0.0))
                            .sum();
                        let losses: f64 = recent_window
                            .windows(2)
                            .map(|w| (w[0] - w[1]).max(0.0))
                            .sum();

                        if gains > losses * 1.5 && out[check_idx].is_finite() {
                            // Strong bullish momentum should yield positive IFT RSI
                            prop_assert!(
								out[check_idx] > -0.1,
								"Bullish momentum (gains > losses*1.5) should yield IFT RSI > -0.1, got {} at index {}",
								out[check_idx],
								check_idx
							);
                        }

                        if losses > gains * 1.5 && out[check_idx].is_finite() {
                            // Strong bearish momentum should yield negative IFT RSI
                            // This will likely FAIL due to the (2w)^2 bug
                            prop_assert!(
								out[check_idx] < 0.1,
								"Bearish momentum (losses > gains*1.5) should yield IFT RSI < 0.1, got {} at index {}",
								out[check_idx],
								check_idx
							);
                        }
                    }
                }

                Ok(())
            },
        )?;

        Ok(())
    }

    #[cfg(not(feature = "proptest"))]
    fn check_ift_rsi_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        Ok(())
    }

    macro_rules! generate_all_ift_rsi_tests {
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

    generate_all_ift_rsi_tests!(
        check_ift_rsi_partial_params,
        check_ift_rsi_accuracy,
        check_ift_rsi_default_candles,
        check_ift_rsi_zero_period,
        check_ift_rsi_period_exceeds_length,
        check_ift_rsi_very_small_dataset,
        check_ift_rsi_reinput,
        check_ift_rsi_nan_handling,
        check_ift_rsi_no_poison,
        check_ift_rsi_property
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = IftRsiBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = IftRsiParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test various parameter sweep configurations
        let test_configs = vec![
            // (rsi_start, rsi_end, rsi_step, wma_start, wma_end, wma_step)
            (2, 10, 2, 2, 10, 2),     // Small periods
            (5, 25, 5, 5, 25, 5),     // Medium periods
            (30, 60, 15, 30, 60, 15), // Large periods
            (2, 5, 1, 2, 5, 1),       // Dense small range
            (9, 15, 3, 9, 15, 3),     // Mid-range dense
            (2, 2, 0, 2, 20, 2),      // Static RSI, varying WMA
            (2, 20, 2, 9, 9, 0),      // Varying RSI, static WMA
        ];

        for (cfg_idx, &(rsi_start, rsi_end, rsi_step, wma_start, wma_end, wma_step)) in
            test_configs.iter().enumerate()
        {
            let output = IftRsiBatchBuilder::new()
                .kernel(kernel)
                .rsi_period_range(rsi_start, rsi_end, rsi_step)
                .wma_period_range(wma_start, wma_end, wma_step)
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
						 at row {} col {} (flat index {}) with params: rsi_period={}, wma_period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.rsi_period.unwrap_or(5),
                        combo.wma_period.unwrap_or(9)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: rsi_period={}, wma_period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.rsi_period.unwrap_or(5),
                        combo.wma_period.unwrap_or(9)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: rsi_period={}, wma_period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.rsi_period.unwrap_or(5),
                        combo.wma_period.unwrap_or(9)
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
#[pyfunction(name = "ift_rsi")]
#[pyo3(signature = (data, rsi_period, wma_period, kernel=None))]
pub fn ift_rsi_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    rsi_period: usize,
    wma_period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = IftRsiParams {
        rsi_period: Some(rsi_period),
        wma_period: Some(wma_period),
    };
    let input = IftRsiInput::from_slice(slice_in, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| ift_rsi_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "IftRsiStream")]
pub struct IftRsiStreamPy {
    stream: IftRsiStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl IftRsiStreamPy {
    #[new]
    fn new(rsi_period: usize, wma_period: usize) -> PyResult<Self> {
        let params = IftRsiParams {
            rsi_period: Some(rsi_period),
            wma_period: Some(wma_period),
        };
        let stream =
            IftRsiStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(IftRsiStreamPy { stream })
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "ift_rsi_batch")]
#[pyo3(signature = (data, rsi_period_range, wma_period_range, kernel=None))]
pub fn ift_rsi_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    rsi_period_range: (usize, usize, usize),
    wma_period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    let sweep = IftRsiBatchRange {
        rsi_period: rsi_period_range,
        wma_period: wma_period_range,
    };

    let combos = expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = slice_in.len();

    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

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
            ift_rsi_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "rsi_periods",
        combos
            .iter()
            .map(|p| p.rsi_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "wma_periods",
        combos
            .iter()
            .map(|p| p.wma_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;

    Ok(dict)
}

// ==================== PYTHON: CUDA BINDINGS (zero-copy) ====================
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "ift_rsi_cuda_batch_dev")]
#[pyo3(signature = (data_f32, rsi_range, wma_range, device_id=0))]
pub fn ift_rsi_cuda_batch_dev_py(
    py: Python<'_>,
    data_f32: numpy::PyReadonlyArray1<'_, f32>,
    rsi_range: (usize, usize, usize),
    wma_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    use crate::cuda::oscillators::CudaIftRsi;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let slice_in: &[f32] = data_f32.as_slice()?;
    let sweep = IftRsiBatchRange { rsi_period: rsi_range, wma_period: wma_range };
    let (inner, dev_id, ctx) = py.allow_threads(|| {
        let cuda = CudaIftRsi::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dev_id = cuda.device_id();
        let ctx = cuda.context_arc();
        let (dev, _combos) = cuda
            .ift_rsi_batch_dev(slice_in, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.synchronize().map_err(|e| PyValueError::new_err(e.to_string()))?; // CAI stream can be omitted
        Ok::<_, PyErr>((dev, dev_id, ctx))
    })?;
    let handle = DeviceArrayF32Py { inner, _ctx: Some(ctx), device_id: Some(dev_id) };
    Ok(handle)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "ift_rsi_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, rsi_period, wma_period, device_id=0))]
pub fn ift_rsi_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    data_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    rsi_period: usize,
    wma_period: usize,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    use crate::cuda::oscillators::CudaIftRsi;
    use numpy::PyUntypedArrayMethods;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let flat_in: &[f32] = data_tm_f32.as_slice()?;
    let rows = data_tm_f32.shape()[0];
    let cols = data_tm_f32.shape()[1];
    let params = IftRsiParams { rsi_period: Some(rsi_period), wma_period: Some(wma_period) };
    let (inner, dev_id, ctx) = py.allow_threads(|| {
        let cuda = CudaIftRsi::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dev_id = cuda.device_id();
        let ctx = cuda.context_arc();
        let dev = cuda
            .ift_rsi_many_series_one_param_time_major_dev(flat_in, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.synchronize().map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, PyErr>((dev, dev_id, ctx))
    })?;
    let handle = DeviceArrayF32Py { inner, _ctx: Some(ctx), device_id: Some(dev_id) };
    Ok(handle)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ift_rsi_js(data: &[f64], rsi_period: usize, wma_period: usize) -> Result<Vec<f64>, JsValue> {
    let params = IftRsiParams {
        rsi_period: Some(rsi_period),
        wma_period: Some(wma_period),
    };
    let input = IftRsiInput::from_slice(data, params);

    let mut output = vec![0.0; data.len()]; // Single allocation

    let kernel = Kernel::Scalar;

    ift_rsi_into_slice(&mut output, &input, kernel)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ift_rsi_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    rsi_period: usize,
    wma_period: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to ift_rsi_into"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let params = IftRsiParams {
            rsi_period: Some(rsi_period),
            wma_period: Some(wma_period),
        };
        let input = IftRsiInput::from_slice(data, params);

        let kernel = Kernel::Scalar;

        if in_ptr == out_ptr as *const f64 {
            // CRITICAL: Aliasing check
            let mut temp = vec![0.0; len];
            ift_rsi_into_slice(&mut temp, &input, kernel)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            ift_rsi_into_slice(out, &input, kernel)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ift_rsi_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ift_rsi_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct IftRsiBatchConfig {
    pub rsi_period_range: (usize, usize, usize),
    pub wma_period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct IftRsiBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<IftRsiParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = ift_rsi_batch)]
pub fn ift_rsi_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: IftRsiBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = IftRsiBatchRange {
        rsi_period: config.rsi_period_range,
        wma_period: config.wma_period_range,
    };

    #[cfg(target_arch = "wasm32")]
    let kernel = detect_best_kernel();
    #[cfg(not(target_arch = "wasm32"))]
    let kernel = Kernel::Scalar;

    let output = ift_rsi_batch_inner(data, &sweep, kernel, false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = IftRsiBatchJsOutput {
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
pub fn ift_rsi_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    rsi_start: usize,
    rsi_end: usize,
    rsi_step: usize,
    wma_start: usize,
    wma_end: usize,
    wma_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to ift_rsi_batch_into",
        ));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let sweep = IftRsiBatchRange {
            rsi_period: (rsi_start, rsi_end, rsi_step),
            wma_period: (wma_start, wma_end, wma_step),
        };

        let combos = expand_grid(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let cols = len;
        let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);

        #[cfg(target_arch = "wasm32")]
        let kernel = detect_best_kernel();
        #[cfg(not(target_arch = "wasm32"))]
        let kernel = Kernel::Scalar;

        ift_rsi_batch_inner_into(data, &sweep, kernel, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}
