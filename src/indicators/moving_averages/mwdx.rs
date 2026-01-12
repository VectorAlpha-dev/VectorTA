#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::mwdx_wrapper::DeviceArrayF32Mwdx;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::{cuda_available, moving_averages::CudaMwdx};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::memory::DeviceBuffer;
#[cfg(feature = "python")]
use numpy::PyUntypedArrayMethods;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use serde::{Deserialize, Serialize};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

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
use std::mem::MaybeUninit;
use thiserror::Error;

impl<'a> AsRef<[f64]> for MwdxInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            MwdxData::Slice(slice) => slice,
            MwdxData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum MwdxData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct MwdxOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct MwdxParams {
    pub factor: Option<f64>,
}

impl Default for MwdxParams {
    fn default() -> Self {
        Self { factor: Some(0.2) }
    }
}

#[derive(Debug, Clone)]
pub struct MwdxInput<'a> {
    pub data: MwdxData<'a>,
    pub params: MwdxParams,
}

impl<'a> MwdxInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: MwdxParams) -> Self {
        Self {
            data: MwdxData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: MwdxParams) -> Self {
        Self {
            data: MwdxData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", MwdxParams::default())
    }
    #[inline]
    pub fn get_factor(&self) -> f64 {
        self.params.factor.unwrap_or(0.2)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct MwdxBuilder {
    factor: Option<f64>,
    kernel: Kernel,
}

impl Default for MwdxBuilder {
    fn default() -> Self {
        Self {
            factor: None,
            kernel: Kernel::Auto,
        }
    }
}

impl MwdxBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn factor(mut self, x: f64) -> Self {
        self.factor = Some(x);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<MwdxOutput, MwdxError> {
        let p = MwdxParams {
            factor: self.factor,
        };
        let i = MwdxInput::from_candles(c, "close", p);
        mwdx_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<MwdxOutput, MwdxError> {
        let p = MwdxParams {
            factor: self.factor,
        };
        let i = MwdxInput::from_slice(d, p);
        mwdx_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<MwdxStream, MwdxError> {
        let p = MwdxParams {
            factor: self.factor,
        };
        MwdxStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum MwdxError {
    #[error("mwdx: No input data was provided.")]
    EmptyInputData,

    #[allow(dead_code)]
    #[error("mwdx: All values are NaN.")]
    AllValuesNaN,
    #[error("mwdx: Factor must be greater than 0, got {factor}")]
    InvalidFactor { factor: f64 },
    #[error("mwdx: Factor leads to invalid denominator, factor: {factor}")]
    InvalidDenominator { factor: f64 },
    #[error("mwdx: Invalid length - expected {expected}, got {got}")]
    OutputLengthMismatch { expected: usize, got: usize },

    #[allow(dead_code)]
    #[error("mwdx: Not enough valid data: needed {needed}, valid {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("mwdx: invalid range: start={start}, end={end}, step={step}")]
    InvalidRange { start: f64, end: f64, step: f64 },
    #[error("mwdx: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline]
pub fn mwdx(input: &MwdxInput) -> Result<MwdxOutput, MwdxError> {
    mwdx_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn mwdx_prepare<'a>(
    input: &'a MwdxInput,
    kernel: Kernel,
) -> Result<(&'a [f64], f64, usize, Kernel), MwdxError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(MwdxError::EmptyInputData);
    }

    let factor = input.get_factor();
    if factor <= 0.0 || factor.is_nan() || factor.is_infinite() {
        return Err(MwdxError::InvalidFactor { factor });
    }

    let fac = factor;

    let warm = data.iter().position(|x| !x.is_nan()).unwrap_or(len);

    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    Ok((data, fac, warm, chosen))
}

#[inline(always)]
fn mwdx_compute_into(data: &[f64], fac: f64, kernel: Kernel, out: &mut [f64]) {
    unsafe {
        match kernel {
            Kernel::Scalar | Kernel::ScalarBatch => mwdx_scalar(data, fac, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => mwdx_avx2(data, fac, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => mwdx_avx512(data, fac, out),

            #[allow(unreachable_patterns)]
            _ => mwdx_scalar(data, fac, out),
        }
    }
}

pub fn mwdx_with_kernel(input: &MwdxInput, kernel: Kernel) -> Result<MwdxOutput, MwdxError> {
    let (data, fac, warm, chosen) = mwdx_prepare(input, kernel)?;
    let mut out = alloc_with_nan_prefix(data.len(), warm);
    mwdx_compute_into(data, fac, chosen, &mut out);
    Ok(MwdxOutput { values: out })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn mwdx_into(input: &MwdxInput, dst: &mut [f64]) -> Result<(), MwdxError> {
    mwdx_into_slice(dst, input, Kernel::Auto)
}

#[inline]
pub fn mwdx_into_slice(dst: &mut [f64], input: &MwdxInput, kern: Kernel) -> Result<(), MwdxError> {
    let (data, fac, warmup_period, chosen) = mwdx_prepare(input, kern)?;

    if dst.len() != data.len() {
        return Err(MwdxError::OutputLengthMismatch {
            expected: data.len(),
            got: dst.len(),
        });
    }

    mwdx_compute_into(data, fac, chosen, dst);

    Ok(())
}

#[inline(always)]
pub unsafe fn mwdx_scalar(data: &[f64], fac: f64, out: &mut [f64]) {
    let n = data.len();
    if n == 0 {
        return;
    }

    let pin = data.as_ptr();
    let pout = out.as_mut_ptr();

    let mut i = 0usize;
    while i < n {
        let x = *pin.add(i);
        if x.is_nan() {
            pout.add(i).write(f64::NAN);
            i += 1;
        } else {
            break;
        }
    }

    if i == n {
        return;
    }

    let mut prev = *pin.add(i);
    pout.add(i).write(prev);
    i += 1;

    while i + 1 < n {
        let x0 = *pin.add(i);
        let y0 = (x0 - prev).mul_add(fac, prev);
        pout.add(i).write(y0);

        let x1 = *pin.add(i + 1);
        let y1 = (x1 - y0).mul_add(fac, y0);
        pout.add(i + 1).write(y1);

        prev = y1;
        i += 2;
    }

    if i < n {
        let x = *pin.add(i);
        let y = (x - prev).mul_add(fac, prev);
        pout.add(i).write(y);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn mwdx_avx2(data: &[f64], fac: f64, out: &mut [f64]) {
    mwdx_scalar(data, fac, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn mwdx_avx512(data: &[f64], fac: f64, out: &mut [f64]) {
    mwdx_scalar(data, fac, out);
}

#[inline]
pub fn mwdx_batch_with_kernel(
    data: &[f64],
    sweep: &MwdxBatchRange,
    k: Kernel,
) -> Result<MwdxBatchOutput, MwdxError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(MwdxError::InvalidKernelForBatch(other)),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    mwdx_batch_par_slice(data, sweep, simd)
}

#[derive(Debug, Clone)]
pub struct MwdxStream {
    factor: f64,
    fac: f64,
    prev: Option<f64>,
}

impl MwdxStream {
    pub fn try_new(params: MwdxParams) -> Result<Self, MwdxError> {
        let factor = params.factor.unwrap_or(0.2);
        if factor <= 0.0 || factor.is_nan() || factor.is_infinite() {
            return Err(MwdxError::InvalidFactor { factor });
        }

        let fac = factor;
        Ok(Self {
            factor,
            fac,
            prev: None,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> f64 {
        if self.prev.is_none() {
            if value.is_nan() {
                return f64::NAN;
            }
            self.prev = Some(value);
            return value;
        }

        let prev = unsafe { self.prev.unwrap_unchecked() };
        let out = (value - prev).mul_add(self.fac, prev);
        self.prev = Some(out);
        out
    }

    #[inline(always)]
    pub fn update_fast_unchecked(&mut self, value: f64) -> f64 {
        let prev = unsafe { self.prev.unwrap_unchecked() };
        let out = (value - prev).mul_add(self.fac, prev);
        self.prev = Some(out);
        out
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.prev = None;
    }
}

#[derive(Clone, Debug)]
pub struct MwdxBatchRange {
    pub factor: (f64, f64, f64),
}

impl Default for MwdxBatchRange {
    fn default() -> Self {
        Self {
            factor: (0.2, 0.449, 0.001),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MwdxBatchBuilder {
    range: MwdxBatchRange,
    kernel: Kernel,
}

impl MwdxBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline]
    pub fn factor_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.factor = (start, end, step);
        self
    }
    #[inline]
    pub fn factor_static(mut self, x: f64) -> Self {
        self.range.factor = (x, x, 0.0);
        self
    }

    pub fn apply_slice(self, data: &[f64]) -> Result<MwdxBatchOutput, MwdxError> {
        mwdx_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<MwdxBatchOutput, MwdxError> {
        MwdxBatchBuilder::new().kernel(k).apply_slice(data)
    }

    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<MwdxBatchOutput, MwdxError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }

    pub fn with_default_candles(c: &Candles) -> Result<MwdxBatchOutput, MwdxError> {
        MwdxBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct MwdxBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<MwdxParams>,
    pub rows: usize,
    pub cols: usize,
}
impl MwdxBatchOutput {
    pub fn row_for_params(&self, p: &MwdxParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| (c.factor.unwrap_or(0.2) - p.factor.unwrap_or(0.2)).abs() < 1e-12)
    }
    pub fn values_for(&self, p: &MwdxParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &MwdxBatchRange) -> Vec<MwdxParams> {
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Vec<f64> {
        if step == 0.0 || (start - end).abs() < 1e-12 {
            return vec![start];
        }
        let d = step.abs();
        let mut vals = Vec::new();
        if start <= end {
            let mut x = start;
            while x <= end + 1e-12 {
                vals.push(x);
                x += d;
            }
        } else {
            let mut x = start;
            while x + 1e-12 >= end {
                vals.push(x);
                x -= d;
            }
        }
        vals
    }

    let factors = axis_f64(r.factor);
    let mut out = Vec::with_capacity(factors.len());
    for &f in &factors {
        out.push(MwdxParams { factor: Some(f) });
    }
    out
}

#[inline(always)]
pub fn mwdx_batch_slice(
    data: &[f64],
    sweep: &MwdxBatchRange,
    kern: Kernel,
) -> Result<MwdxBatchOutput, MwdxError> {
    mwdx_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn mwdx_batch_par_slice(
    data: &[f64],
    sweep: &MwdxBatchRange,
    kern: Kernel,
) -> Result<MwdxBatchOutput, MwdxError> {
    mwdx_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn mwdx_batch_inner(
    data: &[f64],
    sweep: &MwdxBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<MwdxBatchOutput, MwdxError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(MwdxError::InvalidRange {
            start: sweep.factor.0,
            end: sweep.factor.1,
            step: sweep.factor.2,
        });
    }
    if data.is_empty() {
        return Err(MwdxError::EmptyInputData);
    }

    for combo in &combos {
        let factor = combo.factor.unwrap();
        if factor <= 0.0 || factor.is_nan() || factor.is_infinite() {
            return Err(MwdxError::InvalidFactor { factor });
        }
    }

    let rows = combos.len();
    let cols = data.len();
    let first = data.iter().position(|x| !x.is_nan()).unwrap_or(cols);
    let warm_prefixes = vec![first; rows];

    let _ = rows.checked_mul(cols).ok_or(MwdxError::InvalidRange {
        start: sweep.factor.0,
        end: sweep.factor.1,
        step: sweep.factor.2,
    })?;
    let mut raw = make_uninit_matrix(rows, cols);

    unsafe { init_matrix_prefixes(&mut raw, cols, &warm_prefixes) };

    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let prm = &combos[row];
        let factor = prm.factor.unwrap();

        if factor <= 0.0 || factor.is_nan() || factor.is_infinite() {
            return;
        }
        let fac = factor;

        let out_row =
            core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        match kern {
            Kernel::Scalar => mwdx_row_scalar(data, fac, first, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => mwdx_row_avx2(data, fac, first, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => mwdx_row_avx512(data, fac, first, out_row),

            #[allow(unreachable_patterns)]
            _ => mwdx_row_scalar(data, fac, first, out_row),
        }
    };

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

    let mut guard = core::mem::ManuallyDrop::new(raw);
    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(MwdxBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn mwdx_batch_inner_into(
    data: &[f64],
    sweep: &MwdxBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<MwdxParams>, MwdxError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(MwdxError::InvalidRange {
            start: sweep.factor.0,
            end: sweep.factor.1,
            step: sweep.factor.2,
        });
    }
    if data.is_empty() {
        return Err(MwdxError::EmptyInputData);
    }

    for combo in &combos {
        let factor = combo.factor.unwrap();
        if factor <= 0.0 || factor.is_nan() || factor.is_infinite() {
            return Err(MwdxError::InvalidFactor { factor });
        }
        let val2 = (2.0 / factor) - 1.0;
        if val2 + 1.0 <= 0.0 {
            return Err(MwdxError::InvalidDenominator { factor });
        }
    }

    let rows = combos.len();
    let cols = data.len();
    let expected = rows.checked_mul(cols).ok_or(MwdxError::InvalidRange {
        start: sweep.factor.0,
        end: sweep.factor.1,
        step: sweep.factor.2,
    })?;
    if out.len() != expected {
        return Err(MwdxError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }
    let first = data.iter().position(|x| !x.is_nan()).unwrap_or(cols);
    let warm_prefixes = vec![first; rows];

    let out_uninit = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };

    unsafe { init_matrix_prefixes(out_uninit, cols, &warm_prefixes) };

    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let prm = &combos[row];
        let factor = prm.factor.unwrap();

        if factor <= 0.0 || factor.is_nan() || factor.is_infinite() {
            return;
        }
        let val2 = (2.0 / factor) - 1.0;
        if val2 + 1.0 <= 0.0 {
            return;
        }
        let fac = 2.0 / (val2 + 1.0);

        let out_row =
            core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        match kern {
            Kernel::Scalar => mwdx_row_scalar(data, fac, first, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => mwdx_row_avx2(data, fac, first, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => mwdx_row_avx512(data, fac, first, out_row),

            #[allow(unreachable_patterns)]
            _ => mwdx_row_scalar(data, fac, first, out_row),
        }
    };

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
unsafe fn mwdx_row_scalar(data: &[f64], fac: f64, first: usize, out: &mut [f64]) {
    let n = data.len();
    if n == 0 || first >= n {
        return;
    }

    let pin = data.as_ptr();
    let pout = out.as_mut_ptr();

    let mut i = first;
    let mut prev = *pin.add(i);
    pout.add(i).write(prev);
    i += 1;

    while i + 1 < n {
        let x0 = *pin.add(i);
        let y0 = (x0 - prev).mul_add(fac, prev);
        pout.add(i).write(y0);

        let x1 = *pin.add(i + 1);
        let y1 = (x1 - y0).mul_add(fac, y0);
        pout.add(i + 1).write(y1);

        prev = y1;
        i += 2;
    }

    if i < n {
        let x = *pin.add(i);
        let y = (x - prev).mul_add(fac, prev);
        pout.add(i).write(y);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn mwdx_row_avx2(data: &[f64], fac: f64, first: usize, out: &mut [f64]) {
    mwdx_row_scalar(data, fac, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn mwdx_row_avx512(data: &[f64], fac: f64, first: usize, out: &mut [f64]) {
    mwdx_row_scalar(data, fac, first, out);
}

#[inline(always)]
pub fn expand_grid_mwdx(r: &MwdxBatchRange) -> Vec<MwdxParams> {
    expand_grid(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_mwdx_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = MwdxParams { factor: None };
        let input = MwdxInput::from_candles(&candles, "close", default_params);
        let output = mwdx_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        let params_factor_05 = MwdxParams { factor: Some(0.5) };
        let input2 = MwdxInput::from_candles(&candles, "hl2", params_factor_05);
        let output2 = mwdx_with_kernel(&input2, kernel)?;
        assert_eq!(output2.values.len(), candles.close.len());
        let params_custom = MwdxParams { factor: Some(0.7) };
        let input3 = MwdxInput::from_candles(&candles, "hlc3", params_custom);
        let output3 = mwdx_with_kernel(&input3, kernel)?;
        assert_eq!(output3.values.len(), candles.close.len());
        Ok(())
    }

    fn check_mwdx_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let expected_last_five = [
            59302.181566190935,
            59277.94525295275,
            59230.1562023622,
            59215.124961889764,
            59103.099969511815,
        ];
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = MwdxParams { factor: Some(0.2) };
        let input = MwdxInput::from_candles(&candles, "close", params);
        let result = mwdx_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        assert!(result.values.len() >= 5);
        let start_idx = result.values.len() - 5;
        let actual_last_five = &result.values[start_idx..];
        for (i, &val) in actual_last_five.iter().enumerate() {
            let exp_val = expected_last_five[i];
            assert!(
                (val - exp_val).abs() < 1e-5,
                "[{}] MWDX mismatch at index {}, expected {}, got {}",
                test_name,
                i,
                exp_val,
                val
            );
        }
        Ok(())
    }

    fn check_mwdx_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = MwdxInput::with_default_candles(&candles);
        match input.data {
            MwdxData::Candles { source, .. } => {
                assert_eq!(source, "close");
            }
            _ => panic!("Expected MwdxData::Candles"),
        }
        let output = mwdx_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_mwdx_zero_factor(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = MwdxParams { factor: Some(0.0) };
        let input = MwdxInput::from_slice(&input_data, params);
        let res = mwdx_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] MWDX should fail with zero factor",
            test_name
        );
        Ok(())
    }

    fn check_mwdx_negative_factor(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [10.0, 20.0, 30.0];
        let params = MwdxParams { factor: Some(-0.5) };
        let input = MwdxInput::from_slice(&data, params);
        let result = mwdx_with_kernel(&input, kernel);
        assert!(
            result.is_err(),
            "[{}] MWDX should fail with negative factor",
            test_name
        );
        Ok(())
    }

    fn check_mwdx_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [42.0];
        let params = MwdxParams { factor: Some(0.2) };
        let input = MwdxInput::from_slice(&data, params);
        let result = mwdx_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), data.len());
        assert_eq!(result.values[0], 42.0);
        Ok(())
    }

    fn check_mwdx_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_input =
            MwdxInput::from_candles(&candles, "close", MwdxParams { factor: Some(0.2) });
        let first_result = mwdx_with_kernel(&first_input, kernel)?;
        let second_input =
            MwdxInput::from_slice(&first_result.values, MwdxParams { factor: Some(0.3) });
        let second_result = mwdx_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        for i in 0..second_result.values.len() {
            assert!(
                second_result.values[i].is_finite(),
                "[{}] NaN found at index {}",
                test_name,
                i
            );
        }
        Ok(())
    }

    fn check_mwdx_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = MwdxInput::from_candles(&candles, "close", MwdxParams { factor: Some(0.2) });
        let result = mwdx_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        for (i, &val) in result.values.iter().enumerate() {
            assert!(val.is_finite(), "[{}] NaN found at index {}", test_name, i);
        }
        Ok(())
    }

    macro_rules! generate_all_mwdx_tests {
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
    fn check_mwdx_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let test_cases = vec![
            MwdxParams::default(),
            MwdxParams { factor: Some(0.05) },
            MwdxParams { factor: Some(0.1) },
            MwdxParams { factor: Some(0.2) },
            MwdxParams { factor: Some(0.3) },
            MwdxParams { factor: Some(0.4) },
            MwdxParams { factor: Some(0.5) },
            MwdxParams { factor: Some(0.7) },
            MwdxParams { factor: Some(0.9) },
            MwdxParams { factor: Some(1.0) },
            MwdxParams { factor: Some(0.01) },
            MwdxParams { factor: Some(0.99) },
        ];

        for params in test_cases {
            let input = MwdxInput::from_candles(&candles, "close", params.clone());
            let output = mwdx_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} with params factor={:?}",
                        test_name, val, bits, i, params.factor
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} with params factor={:?}",
                        test_name, val, bits, i, params.factor
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} with params factor={:?}",
						test_name, val, bits, i, params.factor
					);
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_mwdx_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_mwdx_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (0.01f64..=2.0).prop_flat_map(|factor| {
            (
                prop::collection::vec(
                    (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                    1..400,
                ),
                Just(factor),
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, factor)| {
                let params = MwdxParams {
                    factor: Some(factor),
                };
                let input = MwdxInput::from_slice(&data, params);

                let MwdxOutput { values: out } = mwdx_with_kernel(&input, kernel).unwrap();

                prop_assert_eq!(
                    out[0],
                    data[0],
                    "Initial value mismatch: out[0]={} != data[0]={}",
                    out[0],
                    data[0]
                );

                let fac = 2.0 / (2.0 / factor);
                for i in 1..data.len() {
                    let expected = fac * data[i] + (1.0 - fac) * out[i - 1];

                    prop_assert!(
                        (out[i] - expected).abs() < 1e-7,
                        "Formula mismatch at index {}: out[{}]={}, expected={}",
                        i,
                        i,
                        out[i],
                        expected
                    );
                }

                if fac <= 1.0 {
                    for i in 1..data.len() {
                        let hist_min = data[0..=i].iter().cloned().fold(f64::INFINITY, f64::min);
                        let hist_max = data[0..=i]
                            .iter()
                            .cloned()
                            .fold(f64::NEG_INFINITY, f64::max);
                        prop_assert!(
                            out[i] >= hist_min - 1e-9 && out[i] <= hist_max + 1e-9,
                            "Output out of bounds at index {}: {} not in [{}, {}]",
                            i,
                            out[i],
                            hist_min,
                            hist_max
                        );
                    }
                }

                if data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-9) {
                    let target = data[0];
                    let last = out[data.len() - 1];
                    prop_assert!(
                        (last - target).abs() < 1e-6,
                        "Failed to converge to constant input: last={}, target={}",
                        last,
                        target
                    );
                }

                if fac <= 1.0 {
                    let is_increasing = data.windows(2).all(|w| w[1] >= w[0] - 1e-12);
                    let is_decreasing = data.windows(2).all(|w| w[1] <= w[0] + 1e-12);
                    if is_increasing {
                        for i in 1..out.len() {
                            prop_assert!(
                                out[i] >= out[i - 1] - 1e-9,
                                "Monotonic increasing violated at index {}: {} < {}",
                                i,
                                out[i],
                                out[i - 1]
                            );
                        }
                    }
                    if is_decreasing {
                        for i in 1..out.len() {
                            prop_assert!(
                                out[i] <= out[i - 1] + 1e-9,
                                "Monotonic decreasing violated at index {}: {} > {}",
                                i,
                                out[i],
                                out[i - 1]
                            );
                        }
                    }
                }

                if kernel != Kernel::Scalar {
                    let MwdxOutput { values: ref_out } =
                        mwdx_with_kernel(&input, Kernel::Scalar).unwrap();
                    for i in 0..data.len() {
                        let y = out[i];
                        let r = ref_out[i];

                        if !y.is_finite() || !r.is_finite() {
                            prop_assert_eq!(
                                y.to_bits(),
                                r.to_bits(),
                                "Special value mismatch at index {}: {} vs {}",
                                i,
                                y,
                                r
                            );
                            continue;
                        }

                        let ulp_diff = if y.is_finite() && r.is_finite() {
                            let y_bits = y.to_bits() as i64;
                            let r_bits = r.to_bits() as i64;
                            (y_bits - r_bits).abs() as u64
                        } else {
                            0
                        };

                        prop_assert!(
                            (y - r).abs() <= 1e-7 || ulp_diff <= 20,
                            "Cross-kernel mismatch at index {}: {} vs {} (ULP={}, diff={})",
                            i,
                            y,
                            r,
                            ulp_diff,
                            (y - r).abs()
                        );
                    }
                }

                if data.len() > 10 && factor > 0.05 && factor < 0.5 {
                    let input_mean = data.iter().sum::<f64>() / data.len() as f64;
                    let output_mean = out.iter().sum::<f64>() / out.len() as f64;

                    let input_var = data.iter().map(|x| (x - input_mean).powi(2)).sum::<f64>()
                        / data.len() as f64;
                    let output_var = out.iter().map(|x| (x - output_mean).powi(2)).sum::<f64>()
                        / out.len() as f64;

                    if input_var > 1e-6 {
                        prop_assert!(
                            output_var <= input_var * 1.01,
                            "Output variance {} should be less than input variance {}",
                            output_var,
                            input_var
                        );
                    }
                }

                #[cfg(debug_assertions)]
                {
                    for (i, &val) in out.iter().enumerate() {
                        if val.is_nan() {
                            continue;
                        }
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

    #[test]
    fn test_leading_nans_single_series() {
        let data_with_nans = vec![f64::NAN, f64::NAN, f64::NAN, 1.0, 2.0, 3.0, 4.0, 5.0];
        let params = MwdxParams { factor: Some(0.5) };
        let input = MwdxInput::from_slice(&data_with_nans, params);

        let result = mwdx(&input).expect("MWDX should succeed");

        for i in 0..3 {
            assert!(
                result.values[i].is_nan(),
                "Index {} should be NaN but got {}",
                i,
                result.values[i]
            );
        }

        assert!(!result.values[3].is_nan(), "Index 3 should not be NaN");
        assert_eq!(result.values[3], 1.0, "First non-NaN should be 1.0");

        let expected_4 = 0.5 * 2.0 + 0.5 * 1.0;
        let expected_5 = 0.5 * 3.0 + 0.5 * 1.5;

        assert!(
            (result.values[4] - expected_4).abs() < 1e-10,
            "Index 4 mismatch"
        );
        assert!(
            (result.values[5] - expected_5).abs() < 1e-10,
            "Index 5 mismatch"
        );
    }

    #[test]
    fn test_leading_nans_batch() {
        let data_with_nans = vec![f64::NAN, f64::NAN, 10.0, 20.0, 30.0, 40.0];
        let sweep = MwdxBatchRange {
            factor: (0.3, 0.5, 0.2),
        };

        let result = mwdx_batch_slice(&data_with_nans, &sweep, Kernel::Scalar)
            .expect("Batch should succeed");

        for row in 0..result.rows {
            let row_start = row * result.cols;
            let row_values = &result.values[row_start..row_start + result.cols];

            for i in 0..2 {
                assert!(
                    row_values[i].is_nan(),
                    "Row {} index {} should be NaN",
                    row,
                    i
                );
            }

            assert_eq!(row_values[2], 10.0, "Row {} index 2 should be 10.0", row);

            assert!(
                !row_values[3].is_nan(),
                "Row {} index 3 should not be NaN",
                row
            );
        }
    }

    generate_all_mwdx_tests!(
        check_mwdx_partial_params,
        check_mwdx_accuracy,
        check_mwdx_default_candles,
        check_mwdx_zero_factor,
        check_mwdx_negative_factor,
        check_mwdx_very_small_dataset,
        check_mwdx_reinput,
        check_mwdx_nan_handling,
        check_mwdx_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_mwdx_tests!(check_mwdx_property);

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = MwdxBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = MwdxParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            59302.181566190935,
            59277.94525295275,
            59230.1562023622,
            59215.124961889764,
            59103.099969511815,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-5,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
            );
        }
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

        let test_configs = vec![
            (0.05, 0.15, 0.05),
            (0.1, 0.5, 0.1),
            (0.2, 0.6, 0.2),
            (0.3, 0.9, 0.3),
            (0.1, 0.3, 0.05),
            (0.01, 0.1, 0.03),
            (0.5, 0.99, 0.1),
        ];

        for (start, end, step) in test_configs {
            let output = MwdxBatchBuilder::new()
                .kernel(kernel)
                .factor_range(start, end, step)
                .apply_candles(&c, "close")?;

            for (idx, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;
                let params = &output.combos[row];

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (params: factor={:?})",
                        test, val, bits, row, col, params.factor
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (params: factor={:?})",
                        test, val, bits, row, col, params.factor
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (params: factor={:?})",
                        test, val, bits, row, col, params.factor
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

    #[test]
    fn test_mwdx_into_matches_api() -> Result<(), Box<dyn std::error::Error>> {
        let n = 256usize;
        let mut data = vec![0.0f64; n];
        data[0] = f64::NAN;
        data[1] = f64::NAN;
        for i in 2..n {
            let x = i as f64;

            data[i] = (x * 0.015).sin() * 50.0 + (x * 0.003).cos() * 10.0 + x * 0.01;
        }

        let input = MwdxInput::from_slice(&data, MwdxParams::default());

        let baseline = mwdx(&input)?.values;

        let mut out = vec![0.0f64; n];
        mwdx_into(&input, &mut out)?;

        assert_eq!(baseline.len(), out.len());

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b)
        }

        for i in 0..n {
            assert!(
                eq_or_both_nan(baseline[i], out[i]),
                "mwdx_into parity mismatch at {}: baseline={:?}, into={:?}",
                i,
                baseline[i],
                out[i]
            );
        }

        Ok(())
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "mwdx")]
#[pyo3(signature = (data, factor, kernel=None))]
pub fn mwdx_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    factor: f64,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = MwdxParams {
        factor: Some(factor),
    };
    let mwdx_in = MwdxInput::from_slice(slice_in, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| mwdx_with_kernel(&mwdx_in, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "MwdxStream")]
pub struct MwdxStreamPy {
    stream: MwdxStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl MwdxStreamPy {
    #[new]
    fn new(factor: f64) -> PyResult<Self> {
        let params = MwdxParams {
            factor: Some(factor),
        };
        let stream =
            MwdxStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(MwdxStreamPy { stream })
    }

    fn update(&mut self, value: f64) -> f64 {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "mwdx_batch")]
#[pyo3(signature = (data, factor_range, kernel=None))]
pub fn mwdx_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    factor_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let sweep = MwdxBatchRange {
        factor: factor_range,
    };

    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("mwdx: invalid range expansion (overflow)"))?;

    let out_arr = unsafe { PyArray1::<f64>::new(py, [expected], false) };
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

            mwdx_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "factors",
        combos
            .iter()
            .map(|p| p.factor.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    Ok(dict)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", unsendable)]
pub struct DeviceArrayF32MwdxPy {
    pub(crate) inner: DeviceArrayF32Mwdx,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl DeviceArrayF32MwdxPy {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("shape", (self.inner.rows, self.inner.cols))?;
        d.set_item("typestr", "<f4")?;
        d.set_item(
            "strides",
            (
                self.inner.cols * std::mem::size_of::<f32>(),
                std::mem::size_of::<f32>(),
            ),
        )?;
        d.set_item("data", (self.inner.device_ptr() as usize, false))?;

        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        (2, self.inner.device_id as i32)
    }

    #[allow(clippy::too_many_arguments)]
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

        let dummy =
            DeviceBuffer::from_slice(&[]).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = self.inner.ctx.clone();
        let device_id = self.inner.device_id;
        let inner = std::mem::replace(
            &mut self.inner,
            DeviceArrayF32Mwdx {
                buf: dummy,
                rows: 0,
                cols: 0,
                ctx,
                device_id,
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
#[pyfunction(name = "mwdx_cuda_batch_dev")]
#[pyo3(signature = (data_f32, factor_range=(0.2, 0.2, 0.0), device_id=0))]
pub fn mwdx_cuda_batch_dev_py(
    py: Python<'_>,
    data_f32: PyReadonlyArray1<'_, f32>,
    factor_range: (f64, f64, f64),
    device_id: usize,
) -> PyResult<DeviceArrayF32MwdxPy> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let slice_in = data_f32.as_slice()?;
    let sweep = MwdxBatchRange {
        factor: factor_range,
    };

    let inner = py.allow_threads(|| {
        let cuda = CudaMwdx::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.mwdx_batch_dev(slice_in, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(DeviceArrayF32MwdxPy { inner })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "mwdx_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, factor, device_id=0))]
pub fn mwdx_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    data_tm_f32: PyReadonlyArray2<'_, f32>,
    factor: f64,
    device_id: usize,
) -> PyResult<DeviceArrayF32MwdxPy> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let flat = data_tm_f32.as_slice()?;
    let shape = data_tm_f32.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err("expected a 2D array"));
    }
    let series_len = shape[0];
    let num_series = shape[1];
    let params = MwdxParams {
        factor: Some(factor),
    };

    let inner = py.allow_threads(|| {
        let cuda = CudaMwdx::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.mwdx_many_series_one_param_time_major_dev(flat, num_series, series_len, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(DeviceArrayF32MwdxPy { inner })
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn mwdx_js(data: &[f64], factor: f64) -> Result<Vec<f64>, JsValue> {
    let params = MwdxParams {
        factor: Some(factor),
    };
    let input = MwdxInput::from_slice(data, params);

    let mut output = vec![0.0; data.len()];

    mwdx_into_slice(&mut output, &input, Kernel::Scalar)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct MwdxBatchConfig {
    pub factor_range: (f64, f64, f64),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct MwdxBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<MwdxParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = mwdx_batch)]
pub fn mwdx_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: MwdxBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = MwdxBatchRange {
        factor: config.factor_range,
    };

    let output = mwdx_batch_inner(data, &sweep, Kernel::Scalar, false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = MwdxBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };

    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn mwdx_batch_js(
    data: &[f64],
    factor_start: f64,
    factor_end: f64,
    factor_step: f64,
) -> Result<Vec<f64>, JsValue> {
    let sweep = MwdxBatchRange {
        factor: (factor_start, factor_end, factor_step),
    };

    mwdx_batch_inner(data, &sweep, Kernel::Scalar, false)
        .map(|output| output.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn mwdx_batch_metadata_js(
    factor_start: f64,
    factor_end: f64,
    factor_step: f64,
) -> Result<Vec<f64>, JsValue> {
    let sweep = MwdxBatchRange {
        factor: (factor_start, factor_end, factor_step),
    };

    let combos = expand_grid(&sweep);
    let metadata: Vec<f64> = combos.iter().map(|combo| combo.factor.unwrap()).collect();

    Ok(metadata)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn mwdx_batch_rows_cols_js(
    factor_start: f64,
    factor_end: f64,
    factor_step: f64,
    data_len: usize,
) -> Vec<usize> {
    let sweep = MwdxBatchRange {
        factor: (factor_start, factor_end, factor_step),
    };
    let combos = expand_grid(&sweep);
    vec![combos.len(), data_len]
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn mwdx_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn mwdx_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn mwdx_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    factor: f64,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to mwdx_into"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        if len == 0 {
            return Err(JsValue::from_str("Empty data"));
        }

        let params = MwdxParams {
            factor: Some(factor),
        };
        let input = MwdxInput::from_slice(data, params);

        if in_ptr == out_ptr as *const f64 {
            let mut temp = vec![0.0; len];
            mwdx_into_slice(&mut temp, &input, Kernel::Scalar)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            mwdx_into_slice(out, &input, Kernel::Scalar)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn mwdx_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    factor_start: f64,
    factor_end: f64,
    factor_step: f64,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to mwdx_batch_into"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        let sweep = MwdxBatchRange {
            factor: (factor_start, factor_end, factor_step),
        };

        let combos = expand_grid(&sweep);
        let rows = combos.len();
        let total_len = rows * len;

        let out_slice = std::slice::from_raw_parts_mut(out_ptr, total_len);

        for (i, params) in combos.iter().enumerate() {
            let row_start = i * len;
            let row_end = row_start + len;
            let out_row = &mut out_slice[row_start..row_end];

            let input = MwdxInput::from_slice(data, params.clone());
            mwdx_into_slice(out_row, &input, Kernel::Scalar)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(rows)
    }
}
