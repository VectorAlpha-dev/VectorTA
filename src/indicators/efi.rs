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
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use serde::{Deserialize, Serialize};
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::{cuda_available, CudaEfi, DeviceArrayF32};
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::context::Context;
#[cfg(all(feature = "python", feature = "cuda"))]
use std::sync::Arc;

#[inline(always)]
fn first_valid_diff_index(price: &[f64], volume: &[f64], first_valid_idx: usize) -> usize {
    let mut i = first_valid_idx.saturating_add(1);
    while i < price.len() {
        if !price[i].is_nan() && !price[i - 1].is_nan() && !volume[i].is_nan() {
            return i;
        }
        i += 1;
    }
    price.len()
}

impl<'a> AsRef<[f64]> for EfiInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            EfiData::Candles { candles, source } => source_type(candles, source),
            EfiData::Slice { price, .. } => price,
        }
    }
}

#[derive(Debug, Clone)]
pub enum EfiData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice {
        price: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct EfiOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct EfiParams {
    pub period: Option<usize>,
}

impl Default for EfiParams {
    fn default() -> Self {
        Self { period: Some(13) }
    }
}

#[derive(Debug, Clone)]
pub struct EfiInput<'a> {
    pub data: EfiData<'a>,
    pub params: EfiParams,
}

impl<'a> EfiInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: EfiParams) -> Self {
        Self {
            data: EfiData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slices(price: &'a [f64], volume: &'a [f64], p: EfiParams) -> Self {
        Self {
            data: EfiData::Slice { price, volume },
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", EfiParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(13)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct EfiBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for EfiBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl EfiBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<EfiOutput, EfiError> {
        let p = EfiParams {
            period: self.period,
        };
        let i = EfiInput::from_candles(c, "close", p);
        efi_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(self, price: &[f64], volume: &[f64]) -> Result<EfiOutput, EfiError> {
        let p = EfiParams {
            period: self.period,
        };
        let i = EfiInput::from_slices(price, volume, p);
        efi_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<EfiStream, EfiError> {
        let p = EfiParams {
            period: self.period,
        };
        EfiStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum EfiError {
    #[error("efi: Empty data provided.")]
    EmptyInputData,
    #[error("efi: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("efi: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("efi: All values are NaN.")]
    AllValuesNaN,
    #[error("efi: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("efi: Invalid range expansion: start={start} end={end} step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("efi: Invalid kernel for batch path: {0:?}")]
    InvalidKernelForBatch(crate::utilities::enums::Kernel),
}

#[inline]
pub fn efi(input: &EfiInput) -> Result<EfiOutput, EfiError> {
    efi_with_kernel(input, Kernel::Auto)
}

pub fn efi_with_kernel(input: &EfiInput, kernel: Kernel) -> Result<EfiOutput, EfiError> {
    let (price, volume): (&[f64], &[f64]) = match &input.data {
        EfiData::Candles { candles, source } => (source_type(candles, source), &candles.volume),
        EfiData::Slice { price, volume } => (price, volume),
    };

    if price.is_empty() || volume.is_empty() || price.len() != volume.len() {
        return Err(EfiError::EmptyInputData);
    }

    let len = price.len();
    let period = input.get_period();
    if period == 0 || period > len {
        return Err(EfiError::InvalidPeriod {
            period,
            data_len: len,
        });
    }

    let first = price
        .iter()
        .zip(volume.iter())
        .position(|(p, v)| !p.is_nan() && !v.is_nan())
        .ok_or(EfiError::AllValuesNaN)?;

    if len - first < 2 {
        return Err(EfiError::NotEnoughValidData {
            needed: 2,
            valid: len - first,
        });
    }

    let warm = first_valid_diff_index(price, volume, first);
    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    let mut out = alloc_with_nan_prefix(len, warm);

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                efi_scalar(price, volume, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => efi_avx2(price, volume, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                efi_avx512(price, volume, period, first, &mut out)
            }
            _ => unreachable!(),
        }
    }
    Ok(EfiOutput { values: out })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn efi_into(input: &EfiInput, out: &mut [f64]) -> Result<(), EfiError> {
    efi_into_slice(out, input, Kernel::Auto)
}

pub fn efi_into_slice(dst: &mut [f64], input: &EfiInput, kern: Kernel) -> Result<(), EfiError> {
    let (price, volume): (&[f64], &[f64]) = match &input.data {
        EfiData::Candles { candles, source } => (source_type(candles, source), &candles.volume),
        EfiData::Slice { price, volume } => (price, volume),
    };

    if price.is_empty() || volume.is_empty() || price.len() != volume.len() {
        return Err(EfiError::EmptyInputData);
    }
    let len = price.len();
    if dst.len() != len {
        return Err(EfiError::OutputLengthMismatch {
            expected: len,
            got: dst.len(),
        });
    }

    let period = input.get_period();
    if period == 0 || period > len {
        return Err(EfiError::InvalidPeriod {
            period,
            data_len: len,
        });
    }

    let first = price
        .iter()
        .zip(volume.iter())
        .position(|(p, v)| !p.is_nan() && !v.is_nan())
        .ok_or(EfiError::AllValuesNaN)?;

    if len - first < 2 {
        return Err(EfiError::NotEnoughValidData {
            needed: 2,
            valid: len - first,
        });
    }

    let warm = first_valid_diff_index(price, volume, first);
    let chosen = match kern {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => efi_scalar(price, volume, period, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => efi_avx2(price, volume, period, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => efi_avx512(price, volume, period, first, dst),
            _ => unreachable!(),
        }
    }

    for v in &mut dst[..warm] {
        *v = f64::NAN;
    }
    Ok(())
}

#[inline(always)]
pub fn efi_scalar(
    price: &[f64],
    volume: &[f64],
    period: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    let len = price.len();
    if len == 0 {
        return;
    }

    let start = first_valid_diff_index(price, volume, first_valid_idx);
    if start >= len {
        return;
    }

    let alpha = 2.0 / (period as f64 + 1.0);
    let one_minus_alpha = 1.0 - alpha;

    unsafe {
        let p_ptr = price.as_ptr();
        let v_ptr = volume.as_ptr();
        let o_ptr = out.as_mut_ptr();

        let p_cur = *p_ptr.add(start);
        let p_prev = *p_ptr.add(start - 1);
        let v_cur = *v_ptr.add(start);
        let mut prev = (p_cur - p_prev) * v_cur;
        *o_ptr.add(start) = prev;
        let mut prev_price = p_cur;

        let mut i = start + 1;
        while i < len {
            let pc = *p_ptr.add(i);
            let vc = *v_ptr.add(i);

            let valid = (pc == pc) & (prev_price == prev_price) & (vc == vc);
            if valid {
                let diff = (pc - prev_price) * vc;

                prev = alpha.mul_add(diff, one_minus_alpha * prev);
            }
            *o_ptr.add(i) = prev;
            prev_price = pc;
            i += 1;
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn efi_avx2(
    price: &[f64],
    volume: &[f64],
    period: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    efi_scalar(price, volume, period, first_valid_idx, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn efi_avx512(
    price: &[f64],
    volume: &[f64],
    period: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        unsafe { efi_avx512_short(price, volume, period, first_valid_idx, out) }
    } else {
        unsafe { efi_avx512_long(price, volume, period, first_valid_idx, out) }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn efi_avx512_short(
    price: &[f64],
    volume: &[f64],
    period: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    efi_scalar(price, volume, period, first_valid_idx, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn efi_avx512_long(
    price: &[f64],
    volume: &[f64],
    period: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    efi_scalar(price, volume, period, first_valid_idx, out)
}

#[derive(Debug, Clone)]
pub struct EfiStream {
    period: usize,
    alpha: f64,
    prev: f64,
    filled: bool,
    last_price: f64,
    has_last: bool,
}

impl EfiStream {
    pub fn try_new(params: EfiParams) -> Result<Self, EfiError> {
        let period = params.period.unwrap_or(13);
        if period == 0 {
            return Err(EfiError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        Ok(Self {
            period,
            alpha: 2.0 / (period as f64 + 1.0),
            prev: f64::NAN,
            filled: false,
            last_price: f64::NAN,
            has_last: false,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, price: f64, volume: f64) -> Option<f64> {
        if !self.has_last {
            self.last_price = price;
            self.has_last = true;
            return None;
        }

        let valid = (price == price) & (self.last_price == self.last_price) & (volume == volume);

        if !valid {
            let out = if self.filled { self.prev } else { f64::NAN };
            self.last_price = price;
            return Some(out);
        }

        let diff = (price - self.last_price) * volume;

        let out = if !self.filled {
            self.prev = diff;
            self.filled = true;
            diff
        } else {
            self.prev = self.alpha.mul_add(diff - self.prev, self.prev);
            self.prev
        };

        self.last_price = price;
        Some(out)
    }
}

#[derive(Clone, Debug)]
pub struct EfiBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for EfiBatchRange {
    fn default() -> Self {
        Self {
            period: (13, 262, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct EfiBatchBuilder {
    range: EfiBatchRange,
    kernel: Kernel,
}

impl EfiBatchBuilder {
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

    pub fn apply_slices(self, price: &[f64], volume: &[f64]) -> Result<EfiBatchOutput, EfiError> {
        efi_batch_with_kernel(price, volume, &self.range, self.kernel)
    }

    pub fn with_default_slices(
        price: &[f64],
        volume: &[f64],
        k: Kernel,
    ) -> Result<EfiBatchOutput, EfiError> {
        EfiBatchBuilder::new().kernel(k).apply_slices(price, volume)
    }

    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<EfiBatchOutput, EfiError> {
        let slice = source_type(c, src);
        let volume = &c.volume;
        self.apply_slices(slice, volume)
    }

    pub fn with_default_candles(c: &Candles) -> Result<EfiBatchOutput, EfiError> {
        EfiBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn efi_batch_with_kernel(
    price: &[f64],
    volume: &[f64],
    sweep: &EfiBatchRange,
    k: Kernel,
) -> Result<EfiBatchOutput, EfiError> {
    let kernel = match k {
        Kernel::Auto => Kernel::ScalarBatch,
        other if other.is_batch() => other,
        other => return Err(EfiError::InvalidKernelForBatch(other)),
    };

    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    efi_batch_par_slice(price, volume, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct EfiBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<EfiParams>,
    pub rows: usize,
    pub cols: usize,
}
impl EfiBatchOutput {
    pub fn row_for_params(&self, p: &EfiParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(13) == p.period.unwrap_or(13))
    }
    pub fn values_for(&self, p: &EfiParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &EfiBatchRange) -> Vec<EfiParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        let mut out = Vec::new();
        if start < end {
            let mut v = start;
            while v <= end {
                out.push(v);
                match v.checked_add(step) {
                    Some(n) => {
                        if n == v {
                            break;
                        }
                        v = n;
                    }
                    None => break,
                }
            }
        } else {
            let mut v = start;
            loop {
                out.push(v);
                if v <= end {
                    break;
                }
                let next = v.saturating_sub(step);
                if next == v {
                    break;
                }
                v = next;
            }
        }
        out
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(EfiParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn efi_batch_slice(
    price: &[f64],
    volume: &[f64],
    sweep: &EfiBatchRange,
    kern: Kernel,
) -> Result<EfiBatchOutput, EfiError> {
    efi_batch_inner(price, volume, sweep, kern, false)
}

#[inline(always)]
pub fn efi_batch_par_slice(
    price: &[f64],
    volume: &[f64],
    sweep: &EfiBatchRange,
    kern: Kernel,
) -> Result<EfiBatchOutput, EfiError> {
    efi_batch_inner(price, volume, sweep, kern, true)
}

#[inline(always)]
fn efi_batch_inner(
    price: &[f64],
    volume: &[f64],
    sweep: &EfiBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<EfiBatchOutput, EfiError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(EfiError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
        });
    }

    let first = price
        .iter()
        .zip(volume.iter())
        .position(|(p, v)| !p.is_nan() && !v.is_nan())
        .ok_or(EfiError::AllValuesNaN)?;

    if price.len() - first < 2 {
        return Err(EfiError::NotEnoughValidData {
            needed: 2,
            valid: price.len() - first,
        });
    }

    let rows = combos.len();
    let cols = price.len();

    let _cap = rows.checked_mul(cols).ok_or(EfiError::InvalidRange {
        start: sweep.period.0,
        end: sweep.period.1,
        step: sweep.period.2,
    })?;

    let warm = first_valid_diff_index(price, volume, first);
    let mut buf_mu = make_uninit_matrix(rows, cols);
    let warm_prefixes = vec![warm; rows];
    init_matrix_prefixes(&mut buf_mu, cols, &warm_prefixes);

    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    efi_batch_inner_into(price, volume, sweep, kern, parallel, out)?;

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(EfiBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn efi_row_scalar(
    price: &[f64],
    volume: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    efi_scalar(price, volume, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn efi_row_avx2(
    price: &[f64],
    volume: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    efi_scalar(price, volume, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn efi_row_avx512(
    price: &[f64],
    volume: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        efi_row_avx512_short(price, volume, first, period, out);
    } else {
        efi_row_avx512_long(price, volume, first, period, out);
    }
    _mm_sfence();
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn efi_row_avx512_short(
    price: &[f64],
    volume: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    efi_scalar(price, volume, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn efi_row_avx512_long(
    price: &[f64],
    volume: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    efi_scalar(price, volume, period, first, out);
}

#[cfg(feature = "python")]
#[pyfunction(name = "efi")]
#[pyo3(signature = (price, volume, period, kernel=None))]
pub fn efi_py<'py>(
    py: Python<'py>,
    price: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let price_slice = price.as_slice()?;
    let volume_slice = volume.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = EfiParams {
        period: Some(period),
    };
    let input = EfiInput::from_slices(price_slice, volume_slice, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| efi_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "EfiStream")]
pub struct EfiStreamPy {
    stream: EfiStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl EfiStreamPy {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        let params = EfiParams {
            period: Some(period),
        };
        let stream =
            EfiStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(EfiStreamPy { stream })
    }

    fn update(&mut self, price: f64, volume: f64) -> Option<f64> {
        self.stream.update(price, volume)
    }
}

#[inline(always)]
fn efi_batch_inner_into(
    price: &[f64],
    volume: &[f64],
    sweep: &EfiBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<EfiParams>, EfiError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(EfiError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
        });
    }

    let first = price
        .iter()
        .zip(volume.iter())
        .position(|(p, v)| !p.is_nan() && !v.is_nan())
        .ok_or(EfiError::AllValuesNaN)?;

    if price.len() - first < 2 {
        return Err(EfiError::NotEnoughValidData {
            needed: 2,
            valid: price.len() - first,
        });
    }

    let cols = price.len();
    let warm = first_valid_diff_index(price, volume, first);

    let rows = combos.len();
    let cols = price.len();
    let expected = rows.checked_mul(cols).ok_or(EfiError::InvalidRange {
        start: sweep.period.0,
        end: sweep.period.1,
        step: sweep.period.2,
    })?;
    if out.len() != expected {
        return Err(EfiError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }
    for row in 0..rows {
        let row_start = row * cols;
        for i in 0..warm.min(cols) {
            out[row_start + i] = f64::NAN;
        }
    }

    let out_mu: &mut [std::mem::MaybeUninit<f64>] = unsafe {
        std::slice::from_raw_parts_mut(
            out.as_mut_ptr() as *mut std::mem::MaybeUninit<f64>,
            out.len(),
        )
    };

    let mut fi_raw_mu: Vec<std::mem::MaybeUninit<f64>> = Vec::with_capacity(cols);
    unsafe {
        fi_raw_mu.set_len(cols);
    }
    unsafe {
        let p_ptr = price.as_ptr();
        let v_ptr = volume.as_ptr();
        let r_ptr = fi_raw_mu.as_mut_ptr();
        let mut i = warm;
        while i < cols {
            let pc = *p_ptr.add(i);
            let pp = *p_ptr.add(i - 1);
            let vc = *v_ptr.add(i);
            if (pc == pc) & (pp == pp) & (vc == vc) {
                let val = (pc - pp) * vc;
                std::ptr::write(r_ptr.add(i), std::mem::MaybeUninit::new(val));
            }
            i += 1;
        }
    }

    let row_fn = |row: usize, dst_row_mu: &mut [std::mem::MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();
        let dst: &mut [f64] =
            std::slice::from_raw_parts_mut(dst_row_mu.as_mut_ptr() as *mut f64, dst_row_mu.len());
        efi_row_from_precomputed(price, volume, &fi_raw_mu, warm, period, dst)
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_mu
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(r, s)| row_fn(r, s));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (r, s) in out_mu.chunks_mut(cols).enumerate() {
                row_fn(r, s);
            }
        }
    } else {
        for (r, s) in out_mu.chunks_mut(cols).enumerate() {
            row_fn(r, s);
        }
    }

    Ok(combos)
}

#[inline(always)]
fn efi_row_from_precomputed(
    price: &[f64],
    volume: &[f64],
    fi_raw: &[std::mem::MaybeUninit<f64>],
    start: usize,
    period: usize,
    out: &mut [f64],
) {
    let len = fi_raw.len();
    if start >= len {
        return;
    }
    let alpha = 2.0 / (period as f64 + 1.0);
    let one_minus_alpha = 1.0 - alpha;
    unsafe {
        let r_ptr = fi_raw.as_ptr();
        let o_ptr = out.as_mut_ptr();

        let mut prev = (*r_ptr.add(start)).assume_init();
        *o_ptr.add(start) = prev;
        let mut i = start + 1;
        while i < len {
            let pc = *price.get_unchecked(i);
            let pp = *price.get_unchecked(i - 1);
            let vc = *volume.get_unchecked(i);
            let valid = (pc == pc) & (pp == pp) & (vc == vc);
            if valid {
                let x = (*r_ptr.add(i)).assume_init();
                prev = alpha.mul_add(x, one_minus_alpha * prev);
            }
            *o_ptr.add(i) = prev;
            i += 1;
        }
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "efi_batch")]
#[pyo3(signature = (price, volume, period_range, kernel=None))]
pub fn efi_batch_py<'py>(
    py: Python<'py>,
    price: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let price_slice = price.as_slice()?;
    let volume_slice = volume.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    let sweep = EfiBatchRange {
        period: period_range,
    };

    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = price_slice.len();
    let total = rows.checked_mul(cols).ok_or_else(|| {
        PyValueError::new_err("efi: Invalid range expansion (rows*cols overflow)")
    })?;

    let out_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
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
            efi_batch_inner_into(price_slice, volume_slice, &sweep, simd, true, slice_out)
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

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn efi_js(price: &[f64], volume: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = EfiParams {
        period: Some(period),
    };
    let input = EfiInput::from_slices(price, volume, params);

    let mut output = vec![0.0; price.len()];
    efi_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn efi_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn efi_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn efi_into(
    in_price_ptr: *const f64,
    in_volume_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    if in_price_ptr.is_null() || in_volume_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let price = std::slice::from_raw_parts(in_price_ptr, len);
        let volume = std::slice::from_raw_parts(in_volume_ptr, len);
        let params = EfiParams {
            period: Some(period),
        };
        let input = EfiInput::from_slices(price, volume, params);

        if in_price_ptr == out_ptr || in_volume_ptr == out_ptr {
            let mut temp = vec![0.0; len];
            efi_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            efi_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct EfiBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct EfiBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<EfiParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = efi_batch)]
pub fn efi_batch_js(price: &[f64], volume: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: EfiBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = EfiBatchRange {
        period: config.period_range,
    };

    let output = efi_batch_with_kernel(price, volume, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = EfiBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };

    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", unsendable)]
pub struct EfiDeviceArrayF32Py {
    pub(crate) inner: Option<DeviceArrayF32>,
    pub(crate) ctx: Arc<Context>,
    pub(crate) device_id: i32,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl EfiDeviceArrayF32Py {
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

        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        (2, self.device_id)
    }

    #[pyo3(signature = (stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &mut self,
        py: Python<'py>,
        stream: Option<PyObject>,
        max_version: Option<PyObject>,
        dl_device: Option<PyObject>,
        copy: Option<PyObject>,
    ) -> PyResult<PyObject> {
        let (kdl, alloc_dev) = self.__dlpack_device__();
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
        let _ = stream;

        let inner = self
            .inner
            .take()
            .ok_or_else(|| PyValueError::new_err("buffer already exported via __dlpack__"))?;

        let rows = inner.rows;
        let cols = inner.cols;
        let buf = inner.buf;

        let max_version_bound = max_version.map(|obj| obj.into_bound(py));

        crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d(
            py,
            buf,
            rows,
            cols,
            alloc_dev,
            max_version_bound,
        )
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "efi_cuda_batch_dev")]
#[pyo3(signature = (price_f32, volume_f32, period_range=(13,13,0), device_id=0))]
pub fn efi_cuda_batch_dev_py(
    py: Python<'_>,
    price_f32: numpy::PyReadonlyArray1<'_, f32>,
    volume_f32: numpy::PyReadonlyArray1<'_, f32>,
    period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<EfiDeviceArrayF32Py> {
    use numpy::PyArrayMethods;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let p = price_f32.as_slice()?;
    let v = volume_f32.as_slice()?;
    if p.len() != v.len() {
        return Err(PyValueError::new_err(
            "price and volume must have same length",
        ));
    }
    let sweep = EfiBatchRange {
        period: period_range,
    };
    let (inner, ctx, dev_id) = py.allow_threads(|| {
        let cuda = CudaEfi::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.context_arc();
        let dev_id = cuda.device_id() as i32;
        let arr = cuda
            .efi_batch_dev(p, v, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, PyErr>((arr, ctx, dev_id))
    })?;
    Ok(EfiDeviceArrayF32Py {
        inner: Some(inner),
        ctx,
        device_id: dev_id,
    })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "efi_cuda_many_series_one_param_dev")]
#[pyo3(signature = (prices_tm_f32, volumes_tm_f32, period=13, device_id=0))]
pub fn efi_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    prices_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    volumes_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    period: usize,
    device_id: usize,
) -> PyResult<EfiDeviceArrayF32Py> {
    use numpy::PyArrayMethods;
    use numpy::PyUntypedArrayMethods;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let p_flat = prices_tm_f32.as_slice()?;
    let v_flat = volumes_tm_f32.as_slice()?;
    let shp_p = prices_tm_f32.shape();
    let shp_v = volumes_tm_f32.shape();
    if shp_p.len() != 2 || shp_v.len() != 2 || shp_p != shp_v {
        return Err(PyValueError::new_err(
            "prices_tm and volumes_tm must be same 2D shape",
        ));
    }
    let rows = shp_p[0];
    let cols = shp_p[1];
    let params = EfiParams {
        period: Some(period),
    };
    let (inner, ctx, dev_id) = py.allow_threads(|| {
        let cuda = CudaEfi::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.context_arc();
        let dev_id = cuda.device_id() as i32;
        let arr = cuda
            .efi_many_series_one_param_time_major_dev(p_flat, v_flat, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, PyErr>((arr, ctx, dev_id))
    })?;
    Ok(EfiDeviceArrayF32Py {
        inner: Some(inner),
        ctx,
        device_id: dev_id,
    })
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn efi_batch_into(
    in_price_ptr: *const f64,
    in_volume_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if in_price_ptr.is_null() || in_volume_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to efi_batch_into"));
    }

    unsafe {
        let price = std::slice::from_raw_parts(in_price_ptr, len);
        let volume = std::slice::from_raw_parts(in_volume_ptr, len);

        let sweep = EfiBatchRange {
            period: (period_start, period_end, period_step),
        };

        let combos = expand_grid(&sweep);
        let rows = combos.len();
        let cols = len;
        let total = rows
            .checked_mul(cols)
            .ok_or_else(|| JsValue::from_str("efi_batch_into: rows*cols overflow"))?;

        let out = std::slice::from_raw_parts_mut(out_ptr, total);

        let first = price
            .iter()
            .zip(volume.iter())
            .position(|(p, v)| !p.is_nan() && !v.is_nan())
            .ok_or_else(|| JsValue::from_str("All values are NaN"))?;

        let warm = first_valid_diff_index(price, volume, first);

        for row in 0..rows {
            let row_start = row * cols;
            for i in 0..warm.min(cols) {
                out[row_start + i] = f64::NAN;
            }
        }

        efi_batch_inner_into(price, volume, &sweep, Kernel::Auto, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::utilities::enums::Kernel;

    fn check_efi_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = EfiParams { period: None };
        let input = EfiInput::from_candles(&candles, "close", default_params);
        let output = efi_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_efi_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = EfiInput::from_candles(&candles, "close", EfiParams::default());
        let result = efi_with_kernel(&input, kernel)?;
        let expected_last_five = [
            -44604.382026531224,
            -39811.02321812391,
            -36599.9671820205,
            -29903.28014503471,
            -55406.382981,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1.0,
                "[{}] EFI {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_efi_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let price = [10.0, 20.0, 30.0];
        let volume = [100.0, 200.0, 300.0];
        let params = EfiParams { period: Some(0) };
        let input = EfiInput::from_slices(&price, &volume, params);
        let res = efi_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] EFI should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_efi_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let price = [10.0, 20.0, 30.0];
        let volume = [100.0, 200.0, 300.0];
        let params = EfiParams { period: Some(10) };
        let input = EfiInput::from_slices(&price, &volume, params);
        let res = efi_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] EFI should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_efi_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = EfiInput::from_candles(&candles, "close", EfiParams { period: Some(13) });
        let res = efi_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        Ok(())
    }

    fn check_efi_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 13;
        let input = EfiInput::from_candles(
            &candles,
            "close",
            EfiParams {
                period: Some(period),
            },
        );
        let batch_output = efi_with_kernel(&input, kernel)?.values;
        let mut stream = EfiStream::try_new(EfiParams {
            period: Some(period),
        })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for (&p, &v) in candles.close.iter().zip(&candles.volume) {
            match stream.update(p, v) {
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
                diff < 1.0,
                "[{}] EFI streaming mismatch at idx {}: batch={}, stream={}",
                test_name,
                i,
                b,
                s
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_efi_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let test_params = vec![
            EfiParams::default(),
            EfiParams { period: Some(2) },
            EfiParams { period: Some(5) },
            EfiParams { period: Some(7) },
            EfiParams { period: Some(10) },
            EfiParams { period: Some(20) },
            EfiParams { period: Some(30) },
            EfiParams { period: Some(50) },
            EfiParams { period: Some(100) },
            EfiParams { period: Some(200) },
            EfiParams { period: Some(500) },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = EfiInput::from_candles(&candles, "close", params.clone());
            let output = efi_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: {:?} (param set {})",
                        test_name, val, bits, i, params, param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: {:?} (param set {})",
                        test_name, val, bits, i, params, param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: {:?} (param set {})",
                        test_name, val, bits, i, params, param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_efi_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_efi_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (2usize..=50).prop_flat_map(|period| {
            (
                (100f64..10000f64, 0.01f64..0.05f64, period + 10..400)
                    .prop_flat_map(move |(base_price, volatility, data_len)| {
                        (
                            Just(base_price),
                            Just(volatility),
                            Just(data_len),
                            prop::collection::vec((-1f64..1f64), data_len),
                            prop::collection::vec((0.1f64..10f64), data_len),
                        )
                    })
                    .prop_map(
                        move |(
                            base_price,
                            volatility,
                            data_len,
                            price_changes,
                            volume_multipliers,
                        )| {
                            let mut price = Vec::with_capacity(data_len);
                            let mut volume = Vec::with_capacity(data_len);
                            let mut current_price = base_price;
                            let base_volume = 1000000.0;

                            for i in 0..data_len {
                                let change = price_changes[i] * volatility * current_price;
                                current_price = (current_price + change).max(10.0);
                                price.push(current_price);

                                let daily_volume = base_volume * volume_multipliers[i];
                                volume.push(daily_volume);
                            }

                            (price, volume)
                        },
                    ),
                Just(period),
            )
        });

        proptest::test_runner::TestRunner::default().run(&strat, |((price, volume), period)| {
            let params = EfiParams {
                period: Some(period),
            };
            let input = EfiInput::from_slices(&price, &volume, params);

            let EfiOutput { values: out } = efi_with_kernel(&input, kernel).unwrap();
            let EfiOutput { values: ref_out } = efi_with_kernel(&input, Kernel::Scalar).unwrap();

            prop_assert_eq!(out.len(), price.len(), "Output length mismatch");

            prop_assert!(out[0].is_nan(), "First value should be NaN");

            let constant_start = price
                .windows(3)
                .position(|w| w.iter().all(|&p| (p - w[0]).abs() < 1e-9));

            if let Some(start) = constant_start {
                let mut constant_end = start + 3;
                while constant_end < price.len()
                    && (price[constant_end] - price[start]).abs() < 1e-9
                {
                    constant_end += 1;
                }

                if constant_end - start >= period && constant_end < price.len() {
                    let check_idx = constant_end - 1;
                    if out[check_idx].is_finite() {
                        prop_assert!(
                            out[check_idx].abs() < 1e-6,
                            "EFI should approach 0 for constant price at idx {}: {}",
                            check_idx,
                            out[check_idx]
                        );
                    }
                }
            }

            for i in 0..out.len() {
                let y = out[i];
                let r = ref_out[i];

                let y_bits = y.to_bits();
                let r_bits = r.to_bits();

                if !y.is_finite() || !r.is_finite() {
                    prop_assert_eq!(
                        y_bits,
                        r_bits,
                        "NaN/infinite mismatch at idx {}: {} vs {}",
                        i,
                        y,
                        r
                    );
                    continue;
                }

                let ulp_diff: u64 = y_bits.abs_diff(r_bits);
                prop_assert!(
                    (y - r).abs() <= 1e-9 || ulp_diff <= 4,
                    "Kernel mismatch at idx {}: {} vs {} (ULP={})",
                    i,
                    y,
                    r,
                    ulp_diff
                );
            }

            let alpha = 2.0 / (period as f64 + 1.0);
            for i in 2..out.len() {
                if out[i].is_finite()
                    && out[i - 1].is_finite()
                    && price[i].is_finite()
                    && price[i - 1].is_finite()
                    && volume[i].is_finite()
                {
                    let raw_fi = (price[i] - price[i - 1]) * volume[i];

                    let expected = alpha * raw_fi + (1.0 - alpha) * out[i - 1];

                    if (out[i] - expected).abs() > 1e-9 {
                        if i > period + 1 {
                            prop_assert!(
                                (out[i] - expected).abs() < 1e-6,
                                "EMA smoothing violated at idx {}: got {}, expected {} (diff: {})",
                                i,
                                out[i],
                                expected,
                                (out[i] - expected).abs()
                            );
                        }
                    }
                }
            }

            Ok(())
        })?;

        Ok(())
    }

    macro_rules! generate_all_efi_tests {
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

    generate_all_efi_tests!(
        check_efi_partial_params,
        check_efi_accuracy,
        check_efi_zero_period,
        check_efi_period_exceeds_length,
        check_efi_nan_handling,
        check_efi_streaming,
        check_efi_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_efi_tests!(check_efi_property);

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = EfiBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = EfiParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let test_configs = vec![
            (2, 10, 2),
            (5, 25, 5),
            (30, 100, 10),
            (2, 5, 1),
            (10, 10, 0),
            (13, 13, 0),
            (50, 50, 0),
            (7, 21, 7),
            (100, 200, 50),
        ];

        for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
            let output = EfiBatchBuilder::new()
                .kernel(kernel)
                .period_range(p_start, p_end, p_step)
                .apply_candles(&c, "close")?;

            for (idx, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;
                let combo = &output.combos[row];

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
						at row {} col {} (flat index {}) with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(13)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						at row {} col {} (flat index {}) with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(13)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						at row {} col {} (flat index {}) with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(13)
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
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
    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);

    #[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
    #[test]
    fn test_efi_into_matches_api() -> Result<(), Box<dyn Error>> {
        let len = 256usize;
        let mut price = vec![0.0; len];
        let mut volume = vec![0.0; len];

        price[0] = f64::NAN;
        volume[0] = 1_000.0;
        for i in 1..len {
            price[i] = 100.0 + (i as f64) * 0.5;
            volume[i] = 10_000.0 + (i as f64) * 100.0;
        }

        let input = EfiInput::from_slices(&price, &volume, EfiParams::default());

        let baseline = efi(&input)?.values;

        let mut out = vec![0.0; len];
        efi_into(&input, &mut out)?;

        assert_eq!(baseline.len(), out.len());

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b)
        }

        for i in 0..len {
            assert!(
                eq_or_both_nan(baseline[i], out[i]),
                "Mismatch at index {}: baseline={}, into={}",
                i,
                baseline[i],
                out[i]
            );
        }
        Ok(())
    }
}
