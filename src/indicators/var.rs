#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
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
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum VarData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

impl<'a> AsRef<[f64]> for VarInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            VarData::Slice(slice) => slice,
            VarData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VarOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct VarParams {
    pub period: Option<usize>,
    pub nbdev: Option<f64>,
}

impl Default for VarParams {
    fn default() -> Self {
        Self {
            period: Some(14),
            nbdev: Some(1.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VarInput<'a> {
    pub data: VarData<'a>,
    pub params: VarParams,
}

impl<'a> VarInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: VarParams) -> Self {
        Self {
            data: VarData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: VarParams) -> Self {
        Self {
            data: VarData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", VarParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
    #[inline]
    pub fn get_nbdev(&self) -> f64 {
        self.params.nbdev.unwrap_or(1.0)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct VarBuilder {
    period: Option<usize>,
    nbdev: Option<f64>,
    kernel: Kernel,
}

impl Default for VarBuilder {
    fn default() -> Self {
        Self {
            period: None,
            nbdev: None,
            kernel: Kernel::Auto,
        }
    }
}

impl VarBuilder {
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
    pub fn nbdev(mut self, x: f64) -> Self {
        self.nbdev = Some(x);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<VarOutput, VarError> {
        let p = VarParams {
            period: self.period,
            nbdev: self.nbdev,
        };
        let i = VarInput::from_candles(c, "close", p);
        var_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<VarOutput, VarError> {
        let p = VarParams {
            period: self.period,
            nbdev: self.nbdev,
        };
        let i = VarInput::from_slice(d, p);
        var_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<VarStream, VarError> {
        let p = VarParams {
            period: self.period,
            nbdev: self.nbdev,
        };
        VarStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum VarError {
    #[error("var: input data is empty (All values are NaN).")]
    EmptyInputData,
    #[error("var: All values are NaN.")]
    AllValuesNaN,
    #[error("var: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("var: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("var: output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("var: invalid range: start = {start}, end = {end}, step = {step}")]
    InvalidRange { start: f64, end: f64, step: f64 },
    #[error("var: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("var: nbdev is NaN or infinite: {nbdev}")]
    InvalidNbdev { nbdev: f64 },
}

#[inline]
pub fn var(input: &VarInput) -> Result<VarOutput, VarError> {
    var_with_kernel(input, Kernel::Auto)
}

pub fn var_with_kernel(input: &VarInput, kernel: Kernel) -> Result<VarOutput, VarError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(VarError::EmptyInputData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(VarError::AllValuesNaN)?;
    let period = input.get_period();
    let nbdev = input.get_nbdev();

    if period == 0 || period > len {
        return Err(VarError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(VarError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }
    if nbdev.is_nan() || nbdev.is_infinite() {
        return Err(VarError::InvalidNbdev { nbdev });
    }

    let chosen = match kernel {
        Kernel::Auto => match detect_best_kernel() {
            Kernel::Avx512 => Kernel::Avx2,
            other => other,
        },
        other => other,
    };

    let mut out = alloc_with_nan_prefix(len, first + period - 1);

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                var_scalar(data, period, first, nbdev, &mut out)?
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => var_avx2(data, period, first, nbdev, &mut out)?,
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                var_avx512(data, period, first, nbdev, &mut out)?
            }
            _ => unreachable!(),
        }
    }

    Ok(VarOutput { values: out })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn var_into(input: &VarInput, out: &mut [f64]) -> Result<(), VarError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(VarError::EmptyInputData);
    }
    if out.len() != len {
        return Err(VarError::OutputLengthMismatch {
            expected: len,
            got: out.len(),
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(VarError::AllValuesNaN)?;
    let period = input.get_period();
    let nbdev = input.get_nbdev();

    if period == 0 || period > len {
        return Err(VarError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(VarError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }
    if nbdev.is_nan() || nbdev.is_infinite() {
        return Err(VarError::InvalidNbdev { nbdev });
    }

    let warmup_end = (first + period - 1).min(len);
    for v in &mut out[..warmup_end] {
        *v = f64::from_bits(0x7ff8_0000_0000_0000);
    }

    let chosen = match detect_best_kernel() {
        Kernel::Avx512 => Kernel::Avx2,
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => var_scalar(data, period, first, nbdev, out)?,
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => var_avx2(data, period, first, nbdev, out)?,
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => var_avx512(data, period, first, nbdev, out)?,
            _ => unreachable!(),
        }
    }

    Ok(())
}

#[inline(always)]
pub fn var_scalar(
    data: &[f64],
    period: usize,
    first: usize,
    nbdev: f64,
    out: &mut [f64],
) -> Result<(), VarError> {
    let len = data.len();
    let nbdev2 = nbdev * nbdev;
    let inv_p = 1.0 / (period as f64);

    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let init = &data[first..first + period];
    let mut it = init.chunks_exact(4);
    for c in &mut it {
        let x0 = c[0];
        let x1 = c[1];
        let x2 = c[2];
        let x3 = c[3];
        sum += x0 + x1 + x2 + x3;
        sum_sq += x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
    }
    for &x in it.remainder() {
        sum += x;
        sum_sq += x * x;
    }

    let idx0 = first + period - 1;
    let mean0 = sum * inv_p;
    out[idx0] = (sum_sq * inv_p - mean0 * mean0) * nbdev2;

    unsafe {
        let mut out_ptr = out.as_mut_ptr().add(idx0 + 1);
        let mut in_new = data.as_ptr().add(first + period);
        let mut in_old = data.as_ptr().add(first);
        let end = data.as_ptr().add(len);

        while in_new < end {
            let new = *in_new;
            let old = *in_old;

            sum += new - old;
            sum_sq += new * new - old * old;

            let mean = sum * inv_p;
            *out_ptr = (sum_sq * inv_p - mean * mean) * nbdev2;

            in_new = in_new.add(1);
            in_old = in_old.add(1);
            out_ptr = out_ptr.add(1);
        }
    }

    Ok(())
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn var_avx2(
    data: &[f64],
    period: usize,
    first: usize,
    nbdev: f64,
    out: &mut [f64],
) -> Result<(), VarError> {
    let len = data.len();
    let nbdev2 = nbdev * nbdev;
    let inv_p = 1.0 / (period as f64);

    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    unsafe {
        let mut idx = first;
        let end = first + period;

        while idx + 4 <= end {
            let v = _mm256_loadu_pd(data.as_ptr().add(idx));
            let v2 = _mm256_mul_pd(v, v);

            let mut lanes: [f64; 4] = core::mem::zeroed();
            _mm256_storeu_pd(lanes.as_mut_ptr(), v);
            let mut t = lanes[0] + lanes[1];
            t = t + lanes[2];
            t = t + lanes[3];
            sum += t;

            _mm256_storeu_pd(lanes.as_mut_ptr(), v2);
            let mut t2 = lanes[0] + lanes[1];
            t2 = t2 + lanes[2];
            t2 = t2 + lanes[3];
            sum_sq += t2;

            idx += 4;
        }

        while idx < end {
            let x = *data.get_unchecked(idx);
            sum += x;
            sum_sq += x * x;
            idx += 1;
        }
    }

    let idx0 = first + period - 1;
    let mean0 = sum * inv_p;
    out[idx0] = (sum_sq * inv_p - mean0 * mean0) * nbdev2;

    unsafe {
        let mut out_ptr = out.as_mut_ptr().add(idx0 + 1);
        let mut in_new = data.as_ptr().add(first + period);
        let mut in_old = data.as_ptr().add(first);
        let end = data.as_ptr().add(len);

        while in_new < end {
            let new = *in_new;
            let old = *in_old;

            sum += new - old;
            sum_sq += new * new - old * old;

            let mean = sum * inv_p;
            *out_ptr = (sum_sq * inv_p - mean * mean) * nbdev2;

            in_new = in_new.add(1);
            in_old = in_old.add(1);
            out_ptr = out_ptr.add(1);
        }
    }

    Ok(())
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn var_avx512(
    data: &[f64],
    period: usize,
    first: usize,
    nbdev: f64,
    out: &mut [f64],
) -> Result<(), VarError> {
    use core::arch::x86_64::*;

    let len = data.len();
    let nbdev2 = nbdev * nbdev;
    let inv_p = 1.0 / (period as f64);

    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;

    unsafe {
        let mut idx = first;
        let end = first + period;
        while idx + 8 <= end {
            let v = _mm512_loadu_pd(data.as_ptr().add(idx));
            let v2 = _mm512_mul_pd(v, v);

            let mut lanes: [f64; 8] = core::mem::zeroed();
            _mm512_storeu_pd(lanes.as_mut_ptr(), v);

            let mut t0 = lanes[0] + lanes[1];
            t0 = t0 + lanes[2];
            t0 = t0 + lanes[3];
            sum += t0;

            let mut t1 = lanes[4] + lanes[5];
            t1 = t1 + lanes[6];
            t1 = t1 + lanes[7];
            sum += t1;

            _mm512_storeu_pd(lanes.as_mut_ptr(), v2);
            let mut u0 = lanes[0] + lanes[1];
            u0 = u0 + lanes[2];
            u0 = u0 + lanes[3];
            sum_sq += u0;
            let mut u1 = lanes[4] + lanes[5];
            u1 = u1 + lanes[6];
            u1 = u1 + lanes[7];
            sum_sq += u1;

            idx += 8;
        }

        while idx + 4 <= end {
            let v4 = _mm256_loadu_pd(data.as_ptr().add(idx));
            let v4sq = _mm256_mul_pd(v4, v4);
            let mut lanes4: [f64; 4] = core::mem::zeroed();
            _mm256_storeu_pd(lanes4.as_mut_ptr(), v4);
            let mut t = lanes4[0] + lanes4[1];
            t = t + lanes4[2];
            t = t + lanes4[3];
            sum += t;
            _mm256_storeu_pd(lanes4.as_mut_ptr(), v4sq);
            let mut u = lanes4[0] + lanes4[1];
            u = u + lanes4[2];
            u = u + lanes4[3];
            sum_sq += u;
            idx += 4;
        }

        while idx < end {
            let x = *data.get_unchecked(idx);
            sum += x;
            sum_sq += x * x;
            idx += 1;
        }

        let idx0 = first + period - 1;
        let mean0 = sum * inv_p;
        out[idx0] = (sum_sq * inv_p - mean0 * mean0) * nbdev2;

        let mut out_ptr = out.as_mut_ptr().add(idx0 + 1);
        let mut in_new = data.as_ptr().add(first + period);
        let mut in_old = data.as_ptr().add(first);
        let end_ptr = data.as_ptr().add(len);

        while in_new < end_ptr {
            let new = *in_new;
            let old = *in_old;

            sum += new - old;
            sum_sq += new * new - old * old;

            let mean = sum * inv_p;
            *out_ptr = (sum_sq * inv_p - mean * mean) * nbdev2;

            in_new = in_new.add(1);
            in_old = in_old.add(1);
            out_ptr = out_ptr.add(1);
        }
    }

    Ok(())
}

#[inline]
pub fn var_into_slice(dst: &mut [f64], input: &VarInput, kern: Kernel) -> Result<(), VarError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(VarError::EmptyInputData);
    }
    if dst.len() != len {
        return Err(VarError::OutputLengthMismatch {
            expected: len,
            got: dst.len(),
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(VarError::AllValuesNaN)?;
    let period = input.get_period();
    let nbdev = input.get_nbdev();

    if period == 0 || period > len {
        return Err(VarError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(VarError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }
    if nbdev.is_nan() || nbdev.is_infinite() {
        return Err(VarError::InvalidNbdev { nbdev });
    }
    let chosen = match kern {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                var_scalar(data, period, first, nbdev, dst)?;
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                var_avx2(data, period, first, nbdev, dst)?;
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                var_avx512(data, period, first, nbdev, dst)?;
            }
            _ => unreachable!(),
        }
    }

    let warmup_end = first + period - 1;
    for v in &mut dst[..warmup_end] {
        *v = f64::NAN;
    }

    Ok(())
}

#[inline(always)]
pub unsafe fn var_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    nbdev: f64,
    out: &mut [f64],
) {
    let len = data.len();
    let inv_p = 1.0 / (period as f64);
    let nbdev2 = nbdev * nbdev;

    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;

    let mut p = data.as_ptr().add(first);
    let end = p.add(period);
    while p.add(4) <= end {
        let x0 = *p;
        let x1 = *p.add(1);
        let x2 = *p.add(2);
        let x3 = *p.add(3);
        sum += x0 + x1 + x2 + x3;
        sum_sq += x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
        p = p.add(4);
    }
    while p < end {
        let x = *p;
        sum += x;
        sum_sq += x * x;
        p = p.add(1);
    }

    let idx0 = first + period - 1;
    let mean0 = sum * inv_p;
    *out.get_unchecked_mut(idx0) = (sum_sq * inv_p - mean0 * mean0) * nbdev2;

    let mut i = idx0 + 1;
    while i + 1 < len {
        {
            let new0 = *data.get_unchecked(i);
            let old0 = *data.get_unchecked(i - period);
            sum += new0 - old0;
            sum_sq += new0 * new0 - old0 * old0;
            let mean = sum * inv_p;
            *out.get_unchecked_mut(i) = (sum_sq * inv_p - mean * mean) * nbdev2;
        }

        {
            let new1 = *data.get_unchecked(i + 1);
            let old1 = *data.get_unchecked(i + 1 - period);
            sum += new1 - old1;
            sum_sq += new1 * new1 - old1 * old1;
            let mean = sum * inv_p;
            *out.get_unchecked_mut(i + 1) = (sum_sq * inv_p - mean * mean) * nbdev2;
        }

        i += 2;
    }

    if i < len {
        let newx = *data.get_unchecked(i);
        let oldx = *data.get_unchecked(i - period);
        sum += newx - oldx;
        sum_sq += newx * newx - oldx * oldx;
        let mean = sum * inv_p;
        *out.get_unchecked_mut(i) = (sum_sq * inv_p - mean * mean) * nbdev2;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn var_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    nbdev: f64,
    out: &mut [f64],
) {
    var_row_scalar(data, first, period, stride, nbdev, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn var_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    nbdev: f64,
    out: &mut [f64],
) {
    if period <= 32 {
        var_row_avx512_short(data, first, period, stride, nbdev, out);
    } else {
        var_row_avx512_long(data, first, period, stride, nbdev, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn var_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    nbdev: f64,
    out: &mut [f64],
) {
    var_row_scalar(data, first, period, stride, nbdev, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn var_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    nbdev: f64,
    out: &mut [f64],
) {
    var_row_scalar(data, first, period, stride, nbdev, out)
}

#[derive(Clone, Debug)]
pub struct VarBatchRange {
    pub period: (usize, usize, usize),
    pub nbdev: (f64, f64, f64),
}

impl Default for VarBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 263, 1),
            nbdev: (1.0, 1.0, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct VarBatchBuilder {
    range: VarBatchRange,
    kernel: Kernel,
}

impl VarBatchBuilder {
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
    pub fn nbdev_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.nbdev = (start, end, step);
        self
    }
    #[inline]
    pub fn nbdev_static(mut self, x: f64) -> Self {
        self.range.nbdev = (x, x, 0.0);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<VarBatchOutput, VarError> {
        var_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<VarBatchOutput, VarError> {
        VarBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<VarBatchOutput, VarError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<VarBatchOutput, VarError> {
        VarBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct VarBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<VarParams>,
    pub rows: usize,
    pub cols: usize,
}
impl VarBatchOutput {
    pub fn row_for_params(&self, p: &VarParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(14) == p.period.unwrap_or(14)
                && (c.nbdev.unwrap_or(1.0) - p.nbdev.unwrap_or(1.0)).abs() < 1e-12
        })
    }
    pub fn values_for(&self, p: &VarParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &VarBatchRange) -> Vec<VarParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        let mut v = Vec::new();
        if start < end {
            let mut x = start;
            while x <= end {
                v.push(x);
                match x.checked_add(step) {
                    Some(next) if next > x => x = next,
                    _ => break,
                }
            }
        } else {
            let mut x = start;
            while x >= end {
                v.push(x);
                match x.checked_sub(step) {
                    Some(next) if next < x => x = next,
                    _ => break,
                }
            }
        }
        v
    }
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Vec<f64> {
        let eps = 1e-12;
        if step.abs() < eps || (start - end).abs() < eps {
            return vec![start];
        }
        let mut v = Vec::new();
        let mut x = start;
        if step > 0.0 {
            if start > end + eps {
                return v;
            }
            while x <= end + eps {
                v.push(x);
                x += step;
            }
        } else {
            if start < end - eps {
                return v;
            }
            while x >= end - eps {
                v.push(x);
                x += step;
            }
        }
        v
    }
    let periods = axis_usize(r.period);
    let nbdevs = axis_f64(r.nbdev);
    let capacity = periods.len().checked_mul(nbdevs.len()).unwrap_or(0);
    let mut out = Vec::with_capacity(capacity);
    for &p in &periods {
        for &n in &nbdevs {
            out.push(VarParams {
                period: Some(p),
                nbdev: Some(n),
            });
        }
    }
    out
}

#[inline(always)]
pub fn var_expand_grid(r: &VarBatchRange) -> Vec<VarParams> {
    expand_grid(r)
}

#[inline(always)]
pub fn var_batch_slice(
    data: &[f64],
    sweep: &VarBatchRange,
    kern: Kernel,
) -> Result<VarBatchOutput, VarError> {
    var_batch_inner(data, sweep, kern, false)
}
#[inline(always)]
pub fn var_batch_par_slice(
    data: &[f64],
    sweep: &VarBatchRange,
    kern: Kernel,
) -> Result<VarBatchOutput, VarError> {
    var_batch_inner(data, sweep, kern, true)
}

fn round_up8(x: usize) -> usize {
    (x + 7) & !7
}

const ENABLE_VAR_BATCH_PREFIX_SUMS: bool = false;

#[inline(always)]
fn make_prefix_sums(data: &[f64], first: usize) -> (Vec<f64>, Vec<f64>) {
    let len = data.len();
    let mut ps = vec![0.0f64; len + 1];
    let mut psq = vec![0.0f64; len + 1];
    let mut s = 0.0f64;
    let mut s2 = 0.0f64;
    for j in first..len {
        let x = data[j];
        s += x;
        s2 += x * x;
        ps[j + 1] = s;
        psq[j + 1] = s2;
    }
    (ps, psq)
}

#[inline(always)]
fn var_batch_inner(
    data: &[f64],
    sweep: &VarBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<VarBatchOutput, VarError> {
    if data.is_empty() {
        return Err(VarError::EmptyInputData);
    }
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(VarError::InvalidRange {
            start: sweep.period.0 as f64,
            end: sweep.period.1 as f64,
            step: sweep.period.2 as f64,
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(VarError::AllValuesNaN)?;
    let max_period = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_period {
        return Err(VarError::NotEnoughValidData {
            needed: max_period,
            valid: data.len() - first,
        });
    }
    let stride = round_up8(max_period);
    let rows = combos.len();
    let cols = data.len();
    let mut buf_mu = make_uninit_matrix(rows, cols);

    let warmup_periods: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warmup_periods);

    let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    if ENABLE_VAR_BATCH_PREFIX_SUMS && matches!(kern, Kernel::Scalar) {
        let (ps, psq) = make_prefix_sums(data, first);
        let compute_row = |row: usize, out_row: &mut [f64]| {
            let period = combos[row].period.unwrap();
            let inv_p = 1.0 / (period as f64);
            let nbdev2 = combos[row].nbdev.unwrap() * combos[row].nbdev.unwrap();
            let start = first + period - 1;
            for i in start..cols {
                let s = ps[i + 1] - ps[i + 1 - period];
                let s2 = psq[i + 1] - psq[i + 1 - period];
                let m = s * inv_p;
                out_row[i] = (s2 * inv_p - m * m) * nbdev2;
            }
        };

        if parallel {
            #[cfg(not(target_arch = "wasm32"))]
            {
                out.par_chunks_mut(cols)
                    .enumerate()
                    .for_each(|(row, slice)| compute_row(row, slice));
            }

            #[cfg(target_arch = "wasm32")]
            {
                for (row, slice) in out.chunks_mut(cols).enumerate() {
                    compute_row(row, slice);
                }
            }
        } else {
            for (row, slice) in out.chunks_mut(cols).enumerate() {
                compute_row(row, slice);
            }
        }
    } else {
        let do_row = |row: usize, out_row: &mut [f64]| unsafe {
            let period = combos[row].period.unwrap();
            let nbdev = combos[row].nbdev.unwrap();
            match kern {
                Kernel::Scalar => var_row_scalar(data, first, period, stride, nbdev, out_row),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2 => var_row_avx2(data, first, period, stride, nbdev, out_row),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512 => var_row_avx512(data, first, period, stride, nbdev, out_row),
                #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
                Kernel::Avx2 | Kernel::Avx512 => {
                    var_row_scalar(data, first, period, stride, nbdev, out_row)
                }
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
    }

    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };
    std::mem::forget(buf_guard);

    Ok(VarBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

pub fn var_batch_with_kernel(
    data: &[f64],
    sweep: &VarBatchRange,
    k: Kernel,
) -> Result<VarBatchOutput, VarError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(VarError::InvalidKernelForBatch(k)),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    var_batch_par_slice(data, sweep, simd)
}

#[derive(Debug, Clone)]
pub struct VarStream {
    period: usize,
    nbdev: f64,
    buffer: Vec<f64>,
    sum: f64,
    sum_sq: f64,
    head: usize,
    filled: bool,
}
impl VarStream {
    pub fn try_new(params: VarParams) -> Result<Self, VarError> {
        let period = params.period.unwrap_or(14);
        if period == 0 {
            return Err(VarError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let nbdev = params.nbdev.unwrap_or(1.0);
        if nbdev.is_nan() || nbdev.is_infinite() {
            return Err(VarError::InvalidNbdev { nbdev });
        }
        Ok(Self {
            period,
            nbdev,
            buffer: vec![f64::NAN; period],
            sum: 0.0,
            sum_sq: 0.0,
            head: 0,
            filled: false,
        })
    }
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        let old = self.buffer[self.head];
        self.buffer[self.head] = value;
        self.head = (self.head + 1) % self.period;
        if !self.filled {
            self.sum += value;
            self.sum_sq += value * value;
            if self.head == 0 {
                self.filled = true;
            } else {
                return None;
            }
        } else {
            self.sum += value - old;
            self.sum_sq += value * value - old * old;
        }
        let inv_p = 1.0 / self.period as f64;
        let mean = self.sum * inv_p;
        let mean_sq = self.sum_sq * inv_p;
        Some((mean_sq - mean * mean) * self.nbdev * self.nbdev)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn var_js(data: &[f64], period: usize, nbdev: f64) -> Result<Vec<f64>, JsValue> {
    let params = VarParams {
        period: Some(period),
        nbdev: Some(nbdev),
    };
    let input = VarInput::from_slice(data, params);

    let mut output = vec![0.0; data.len()];

    var_into_slice(&mut output, &input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn var_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn var_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn var_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
    nbdev: f64,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        if period == 0 || period > len {
            return Err(JsValue::from_str("Invalid period"));
        }

        let params = VarParams {
            period: Some(period),
            nbdev: Some(nbdev),
        };
        let input = VarInput::from_slice(data, params);

        if in_ptr == out_ptr as *const f64 {
            let mut temp = vec![0.0; len];
            var_into_slice(&mut temp, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            var_into_slice(out, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct VarBatchConfig {
    pub period_range: (usize, usize, usize),
    pub nbdev_range: (f64, f64, f64),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct VarBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<VarParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = var_batch)]
pub fn var_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: VarBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = VarBatchRange {
        period: config.period_range,
        nbdev: config.nbdev_range,
    };

    let output = var_batch_inner(data, &sweep, detect_best_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = VarBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };

    serde_wasm_bindgen::to_value(&js_output).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn var_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
    nbdev_start: f64,
    nbdev_end: f64,
    nbdev_step: f64,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        let sweep = VarBatchRange {
            period: (period_start, period_end, period_step),
            nbdev: (nbdev_start, nbdev_end, nbdev_step),
        };

        let combos = expand_grid(&sweep);
        if combos.is_empty() {
            return Err(JsValue::from_str("No valid parameter combinations"));
        }
        let rows = combos.len();
        let cols = len;

        let total = rows
            .checked_mul(cols)
            .ok_or_else(|| JsValue::from_str("rows * cols overflow"))?;
        let out = std::slice::from_raw_parts_mut(out_ptr, total);

        let first = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
        for (r, prm) in combos.iter().enumerate() {
            let warm = (first + prm.period.unwrap() - 1).min(cols);
            let row = &mut out[r * cols..r * cols + warm];
            for v in row {
                *v = f64::NAN;
            }
        }

        let _ = var_batch_inner_into(data, &sweep, detect_best_kernel(), false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use paste::paste;

    #[test]
    fn test_var_into_matches_api() -> Result<(), Box<dyn Error>> {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = VarInput::from_candles(&candles, "close", VarParams::default());

        let VarOutput { values: expected } = var(&input)?;

        let mut out = vec![0.0f64; expected.len()];

        #[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
        {
            var_into(&input, &mut out)?;
        }
        #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
        {
            var_into_slice(&mut out, &input, detect_best_kernel())?;
        }

        assert_eq!(out.len(), expected.len());

        for (i, (a, b)) in out.iter().zip(expected.iter()).enumerate() {
            let eq_or_both_nan = (*a == *b) || (a.is_nan() && b.is_nan());
            assert!(
                eq_or_both_nan,
                "Mismatch at index {}: got {:?}, expected {:?}",
                i, a, b
            );
        }

        Ok(())
    }

    fn check_var_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = VarParams {
            period: None,
            nbdev: None,
        };
        let input = VarInput::from_candles(&candles, "close", default_params);
        let output = var_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_var_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = VarInput::from_candles(&candles, "close", VarParams::default());
        let var_result = var_with_kernel(&input, kernel)?;
        assert_eq!(var_result.values.len(), candles.close.len());
        let expected_last_five = [
            350987.4081501961,
            348493.9183540344,
            302611.06121110916,
            106092.2499871254,
            121941.35202789307,
        ];
        let start_index = var_result.values.len() - 5;
        let result_last_five = &var_result.values[start_index..];
        for (i, &value) in result_last_five.iter().enumerate() {
            let expected_value = expected_last_five[i];
            assert!(
                (value - expected_value).abs() < 1e-1,
                "[{}] VAR mismatch at idx {}: got {}, expected {}",
                test_name,
                i,
                value,
                expected_value
            );
        }
        Ok(())
    }

    fn check_var_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = VarInput::with_default_candles(&candles);
        match input.data {
            VarData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected VarData::Candles"),
        }
        let output = var_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_var_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = VarParams {
            period: Some(0),
            nbdev: None,
        };
        let input = VarInput::from_slice(&input_data, params);
        let res = var_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] VAR should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_var_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = VarParams {
            period: Some(10),
            nbdev: None,
        };
        let input = VarInput::from_slice(&data_small, params);
        let res = var_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] VAR should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_var_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = VarParams {
            period: Some(14),
            nbdev: None,
        };
        let input = VarInput::from_slice(&single_point, params);
        let res = var_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] VAR should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_var_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = VarParams {
            period: Some(14),
            nbdev: Some(1.0),
        };
        let first_input = VarInput::from_candles(&candles, "close", first_params);
        let first_result = var_with_kernel(&first_input, kernel)?;
        let second_params = VarParams {
            period: Some(14),
            nbdev: Some(1.0),
        };
        let second_input = VarInput::from_slice(&first_result.values, second_params);
        let second_result = var_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    fn check_var_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = VarInput::from_candles(
            &candles,
            "close",
            VarParams {
                period: Some(14),
                nbdev: None,
            },
        );
        let res = var_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 30 {
            for (i, &val) in res.values[30..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test_name,
                    30 + i
                );
            }
        }
        Ok(())
    }

    fn check_var_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 14;
        let nbdev = 1.0;
        let input = VarInput::from_candles(
            &candles,
            "close",
            VarParams {
                period: Some(period),
                nbdev: Some(nbdev),
            },
        );
        let batch_output = var_with_kernel(&input, kernel)?.values;
        let mut stream = VarStream::try_new(VarParams {
            period: Some(period),
            nbdev: Some(nbdev),
        })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(var_val) => stream_values.push(var_val),
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
                "[{}] VAR streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
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
    fn check_var_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let test_params = vec![
            VarParams::default(),
            VarParams {
                period: Some(2),
                nbdev: Some(1.0),
            },
            VarParams {
                period: Some(5),
                nbdev: Some(1.0),
            },
            VarParams {
                period: Some(10),
                nbdev: Some(1.0),
            },
            VarParams {
                period: Some(20),
                nbdev: Some(1.0),
            },
            VarParams {
                period: Some(30),
                nbdev: Some(1.0),
            },
            VarParams {
                period: Some(50),
                nbdev: Some(1.0),
            },
            VarParams {
                period: Some(100),
                nbdev: Some(1.0),
            },
            VarParams {
                period: Some(200),
                nbdev: Some(1.0),
            },
            VarParams {
                period: Some(14),
                nbdev: Some(0.5),
            },
            VarParams {
                period: Some(14),
                nbdev: Some(2.0),
            },
            VarParams {
                period: Some(14),
                nbdev: Some(3.0),
            },
            VarParams {
                period: Some(7),
                nbdev: Some(1.5),
            },
            VarParams {
                period: Some(21),
                nbdev: Some(2.5),
            },
            VarParams {
                period: Some(50),
                nbdev: Some(0.75),
            },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = VarInput::from_candles(&candles, "close", params.clone());
            let output = var_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: period={}, nbdev={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(14),
                        params.nbdev.unwrap_or(1.0),
                        param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: period={}, nbdev={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(14),
                        params.nbdev.unwrap_or(1.0),
                        param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: period={}, nbdev={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(14),
                        params.nbdev.unwrap_or(1.0),
                        param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_var_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_var_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (2usize..=64).prop_flat_map(|period| {
            (
                prop::collection::vec(
                    (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                    period.max(10)..400,
                ),
                Just(period),
                0.1f64..3.0f64,
                -100.0f64..100.0f64,
                -1e5f64..1e5f64,
                prop::bool::ANY,
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(
                &strat,
                |(mut data, period, nbdev, trend, intercept, use_special_pattern)| {
                    if use_special_pattern {
                        for (i, val) in data.iter_mut().enumerate() {
                            *val = intercept + trend * (i as f64);
                        }

                        for val in data.iter_mut() {
                            *val += (val.abs() * 0.01).min(10.0)
                                * (if val.is_sign_positive() { 1.0 } else { -1.0 });
                        }
                    }

                    let params = VarParams {
                        period: Some(period),
                        nbdev: Some(nbdev),
                    };
                    let input = VarInput::from_slice(&data, params);

                    let VarOutput { values: out } = var_with_kernel(&input, kernel).unwrap();
                    let VarOutput { values: ref_out } =
                        var_with_kernel(&input, Kernel::Scalar).unwrap();

                    for i in (period - 1)..data.len() {
                        let y = out[i];
                        if !y.is_nan() {
                            prop_assert!(
                                y >= -1e-6,
                                "[{}] Variance should be non-negative at idx {}: got {}",
                                test_name,
                                i,
                                y
                            );
                        }
                    }

                    if period == 2 && !use_special_pattern {
                        for i in (period - 1)..data.len().min(10) {
                            if !out[i].is_nan() {
                                let window = &data[i + 1 - period..=i];
                                let mean = (window[0] + window[1]) / 2.0;
                                let expected =
                                    ((window[0] - mean).powi(2) + (window[1] - mean).powi(2)) / 2.0
                                        * nbdev
                                        * nbdev;

                                let tolerance = (expected.abs() + 1.0) * 1e-8;
                                prop_assert!(
                                    (out[i] - expected).abs() <= tolerance,
                                    "[{}] Period=2 variance mismatch at idx {}: got {} expected {}",
                                    test_name,
                                    i,
                                    out[i],
                                    expected
                                );
                            }
                        }
                    }

                    let is_constant = data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10);
                    if is_constant && data.len() >= period {
                        for i in (period - 1)..data.len() {
                            if !out[i].is_nan() {
                                prop_assert!(
								out[i].abs() <= 1e-6,
								"[{}] Constant data should have near-zero variance at idx {}: got {}",
								test_name, i, out[i]
							);
                            }
                        }
                    }

                    for i in (period - 1)..data.len() {
                        let y = out[i];
                        let r = ref_out[i];

                        if !y.is_finite() || !r.is_finite() {
                            prop_assert!(
                                y.to_bits() == r.to_bits(),
                                "[{}] finite/NaN mismatch at idx {}: {} vs {}",
                                test_name,
                                i,
                                y,
                                r
                            );
                            continue;
                        }

                        let y_bits = y.to_bits();
                        let r_bits = r.to_bits();
                        let ulp_diff: u64 = y_bits.abs_diff(r_bits);

                        prop_assert!(
                            (y - r).abs() <= 1e-9 || ulp_diff <= 4,
                            "[{}] Kernel mismatch at idx {}: {} vs {} (ULP={})",
                            test_name,
                            i,
                            y,
                            r,
                            ulp_diff
                        );
                    }

                    if !use_special_pattern && data.len() >= period && period <= 10 {
                        let idx = period * 2;
                        if idx < data.len() && !out[idx].is_nan() {
                            let window = &data[idx + 1 - period..=idx];
                            let mean: f64 = window.iter().sum::<f64>() / (period as f64);
                            let mean_sq: f64 =
                                window.iter().map(|x| x * x).sum::<f64>() / (period as f64);
                            let expected_var = (mean_sq - mean * mean) * nbdev * nbdev;

                            let tolerance = (expected_var.abs() + 1.0) * 1e-8;
                            prop_assert!(
							(out[idx] - expected_var).abs() <= tolerance,
							"[{}] Mathematical formula mismatch at idx {}: got {} expected {} (diff: {})",
							test_name, idx, out[idx], expected_var, (out[idx] - expected_var).abs()
						);
                        }
                    }

                    Ok(())
                },
            )
            .unwrap();

        Ok(())
    }

    macro_rules! generate_all_var_tests {
        ($($test_fn:ident),*) => {
            paste! {
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
    generate_all_var_tests!(
        check_var_partial_params,
        check_var_accuracy,
        check_var_default_candles,
        check_var_zero_period,
        check_var_period_exceeds_length,
        check_var_very_small_dataset,
        check_var_reinput,
        check_var_nan_handling,
        check_var_streaming,
        check_var_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_var_tests!(check_var_property);

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = VarBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = VarParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [
            350987.4081501961,
            348493.9183540344,
            302611.06121110916,
            106092.2499871254,
            121941.35202789307,
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

        let test_configs = vec![
            ((2, 10, 2), (1.0, 1.0, 0.0)),
            ((10, 30, 5), (1.0, 2.0, 0.5)),
            ((30, 100, 10), (0.5, 1.5, 0.5)),
            ((2, 5, 1), (1.0, 3.0, 1.0)),
            ((14, 14, 0), (1.0, 1.0, 0.0)),
            ((5, 25, 5), (2.0, 2.0, 0.0)),
            ((50, 100, 25), (1.0, 2.0, 0.25)),
            ((14, 28, 7), (0.5, 2.5, 0.5)),
        ];

        for (cfg_idx, &(period_range, nbdev_range)) in test_configs.iter().enumerate() {
            let output = VarBatchBuilder::new()
                .kernel(kernel)
                .period_range(period_range.0, period_range.1, period_range.2)
                .nbdev_range(nbdev_range.0, nbdev_range.1, nbdev_range.2)
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
						 at row {} col {} (flat index {}) with params: period={}, nbdev={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(14),
                        combo.nbdev.unwrap_or(1.0)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}, nbdev={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(14),
                        combo.nbdev.unwrap_or(1.0)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}, nbdev={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(14),
                        combo.nbdev.unwrap_or(1.0)
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
            paste! {
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
    fn test_batch_non_aligned_periods() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let sweep = VarBatchRange {
            period: (7, 7, 0),
            nbdev: (1.0, 1.0, 0.0),
        };

        let result = var_batch_slice(&data, &sweep, Kernel::Scalar);
        assert!(result.is_ok(), "Should handle period=7 with 10 data points");

        let sweep_multi = VarBatchRange {
            period: (5, 7, 1),
            nbdev: (1.0, 1.0, 0.0),
        };

        let result_multi = var_batch_slice(&data, &sweep_multi, Kernel::Scalar);
        assert!(
            result_multi.is_ok(),
            "Should handle periods 5,6,7 with 10 data points"
        );
        let output = result_multi.unwrap();
        assert_eq!(output.rows, 3, "Should have 3 rows for periods 5,6,7");

        let data_short = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result_short = var_batch_slice(&data_short, &sweep, Kernel::Scalar);
        assert!(
            result_short.is_err(),
            "Should reject when data length (6) < period (7)"
        );

        let data_15 = vec![1.0; 15];
        let sweep_15 = VarBatchRange {
            period: (15, 15, 0),
            nbdev: (1.0, 1.0, 0.0),
        };

        let result_15 = var_batch_slice(&data_15, &sweep_15, Kernel::Scalar);
        assert!(
            result_15.is_ok(),
            "Should handle period=15 with exactly 15 data points"
        );
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "var")]
#[pyo3(signature = (data, period=14, nbdev=1.0, kernel=None))]
pub fn var_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period: usize,
    nbdev: f64,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = VarParams {
        period: Some(period),
        nbdev: Some(nbdev),
    };
    let input = VarInput::from_slice(slice_in, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| var_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "VarStream")]
pub struct VarStreamPy {
    stream: VarStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl VarStreamPy {
    #[new]
    fn new(period: usize, nbdev: f64) -> PyResult<Self> {
        let params = VarParams {
            period: Some(period),
            nbdev: Some(nbdev),
        };
        let stream =
            VarStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(VarStreamPy { stream })
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "var_batch")]
#[pyo3(signature = (data, period_range, nbdev_range=(1.0, 1.0, 0.0), kernel=None))]
pub fn var_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    nbdev_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let slice_in = data.as_slice()?;
    let sweep = VarBatchRange {
        period: period_range,
        nbdev: nbdev_range,
    };

    let kern = validate_kernel(kernel, true)?;
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

    let out = py
        .allow_threads(|| var_batch_par_slice(slice_in, &sweep, simd))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let rows = out.rows;
    let cols = out.cols;

    let dict = PyDict::new(py);

    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows * cols overflow"))?;
    let values_2d = unsafe { numpy::PyArray2::<f64>::new(py, [rows, cols], false) };
    let raw_ptr = values_2d.data() as *mut f64;
    let output_slice = unsafe { std::slice::from_raw_parts_mut(raw_ptr, total) };
    output_slice.copy_from_slice(&out.values);

    dict.set_item("values", values_2d)?;
    dict.set_item(
        "periods",
        out.combos
            .iter()
            .map(|p| p.period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "nbdevs",
        out.combos
            .iter()
            .map(|p| p.nbdev.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    Ok(dict)
}

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::cuda_available;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::DeviceArrayF32;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::var_wrapper::CudaVar;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::context::Context;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::memory::DeviceBuffer;
#[cfg(all(feature = "python", feature = "cuda"))]
use std::sync::Arc;

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", name = "VarDeviceArrayF32", unsendable)]
pub struct VarDeviceArrayF32Py {
    pub(crate) inner: DeviceArrayF32,
    pub(crate) _ctx: Arc<Context>,
    pub(crate) device_id: u32,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl VarDeviceArrayF32Py {
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
        (2, self.device_id as i32)
    }

    #[pyo3(signature = (stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        mut slf: pyo3::PyRefMut<'py, Self>,
        py: Python<'py>,
        stream: Option<pyo3::PyObject>,
        max_version: Option<pyo3::PyObject>,
        dl_device: Option<pyo3::PyObject>,
        copy: Option<pyo3::PyObject>,
    ) -> PyResult<PyObject> {
        use cust::memory::DeviceBuffer;

        let (expected_type, expected_dev) = slf.__dlpack_device__();
        if let Some(dev_obj) = dl_device.as_ref() {
            if let Ok((dev_type, dev_id)) = dev_obj.extract::<(i32, i32)>(py) {
                if dev_type != expected_type || dev_id != expected_dev {
                    return Err(PyValueError::new_err("dl_device mismatch for VAR buffer"));
                }
            }
        }

        let wants_copy = copy
            .as_ref()
            .and_then(|c| c.extract::<bool>(py).ok())
            .unwrap_or(false);
        if wants_copy {
            return Err(PyValueError::new_err(
                "copy=True not supported for VAR DLPack export",
            ));
        }

        let _ = stream;

        let dummy =
            DeviceBuffer::from_slice(&[]).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let rows = slf.inner.rows;
        let cols = slf.inner.cols;
        let inner = std::mem::replace(
            &mut slf.inner,
            DeviceArrayF32 {
                buf: dummy,
                rows: 0,
                cols: 0,
            },
        );

        let max_version_bound = max_version.map(|obj| obj.into_bound(py));

        export_f32_cuda_dlpack_2d(py, inner.buf, rows, cols, expected_dev, max_version_bound)
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "var_cuda_batch_dev")]
#[pyo3(signature = (data_f32, period_range, nbdev_range=(1.0,1.0,0.0), device_id=0))]
pub fn var_cuda_batch_dev_py(
    py: Python<'_>,
    data_f32: numpy::PyReadonlyArray1<'_, f32>,
    period_range: (usize, usize, usize),
    nbdev_range: (f32, f32, f32),
    device_id: usize,
) -> PyResult<VarDeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let slice_in = data_f32.as_slice()?;
    let sweep = VarBatchRange {
        period: period_range,
        nbdev: (
            nbdev_range.0 as f64,
            nbdev_range.1 as f64,
            nbdev_range.2 as f64,
        ),
    };
    let (inner, ctx, dev_id) = py.allow_threads(|| {
        let cuda = CudaVar::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.context_arc();
        let dev_id = cuda.device_id();
        cuda.var_batch_dev(slice_in, &sweep)
            .map(|pair| (pair.0, ctx, dev_id))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    Ok(VarDeviceArrayF32Py {
        inner,
        _ctx: ctx,
        device_id: dev_id,
    })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "var_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, cols, rows, period, nbdev=1.0, device_id=0))]
pub fn var_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    data_tm_f32: numpy::PyReadonlyArray1<'_, f32>,
    cols: usize,
    rows: usize,
    period: usize,
    nbdev: f64,
    device_id: usize,
) -> PyResult<VarDeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let slice_tm = data_tm_f32.as_slice()?;
    let params = VarParams {
        period: Some(period),
        nbdev: Some(nbdev),
    };
    let (inner, ctx, dev_id) = py.allow_threads(|| {
        let cuda = CudaVar::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.context_arc();
        let dev_id = cuda.device_id();
        cuda.var_many_series_one_param_time_major_dev(slice_tm, cols, rows, &params)
            .map(|h| (h, ctx, dev_id))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    Ok(VarDeviceArrayF32Py {
        inner,
        _ctx: ctx,
        device_id: dev_id,
    })
}

#[inline(always)]
fn var_batch_inner_into(
    data: &[f64],
    sweep: &VarBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<VarParams>, VarError> {
    if data.is_empty() {
        return Err(VarError::EmptyInputData);
    }
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(VarError::InvalidRange {
            start: sweep.period.0 as f64,
            end: sweep.period.1 as f64,
            step: sweep.period.2 as f64,
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(VarError::AllValuesNaN)?;
    let max_period = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_period {
        return Err(VarError::NotEnoughValidData {
            needed: max_period,
            valid: data.len() - first,
        });
    }
    let stride = round_up8(max_period);
    let cols = data.len();

    if ENABLE_VAR_BATCH_PREFIX_SUMS && matches!(kern, Kernel::Scalar) {
        let (ps, psq) = make_prefix_sums(data, first);
        let compute_row = |row: usize, out_row: &mut [f64]| {
            let period = combos[row].period.unwrap();
            let inv_p = 1.0 / (period as f64);
            let nbdev2 = combos[row].nbdev.unwrap() * combos[row].nbdev.unwrap();
            let start = first + period - 1;
            for i in start..cols {
                let s = ps[i + 1] - ps[i + 1 - period];
                let s2 = psq[i + 1] - psq[i + 1 - period];
                let m = s * inv_p;
                out_row[i] = (s2 * inv_p - m * m) * nbdev2;
            }
        };
        if parallel {
            #[cfg(not(target_arch = "wasm32"))]
            {
                out.par_chunks_mut(cols)
                    .enumerate()
                    .for_each(|(row, slice)| compute_row(row, slice));
            }
            #[cfg(target_arch = "wasm32")]
            {
                for (row, slice) in out.chunks_mut(cols).enumerate() {
                    compute_row(row, slice);
                }
            }
        } else {
            for (row, slice) in out.chunks_mut(cols).enumerate() {
                compute_row(row, slice);
            }
        }
    } else {
        let do_row = |row: usize, out_row: &mut [f64]| unsafe {
            let period = combos[row].period.unwrap();
            let nbdev = combos[row].nbdev.unwrap();
            match kern {
                Kernel::Scalar => var_row_scalar(data, first, period, stride, nbdev, out_row),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2 => var_row_avx2(data, first, period, stride, nbdev, out_row),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512 => var_row_avx512(data, first, period, stride, nbdev, out_row),
                #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
                Kernel::Avx2 | Kernel::Avx512 => {
                    var_row_scalar(data, first, period, stride, nbdev, out_row)
                }
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
    }

    Ok(combos)
}
