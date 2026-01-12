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
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use serde::{Deserialize, Serialize};
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

impl<'a> AsRef<[f64]> for LinearRegInterceptInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            LinearRegInterceptData::Slice(slice) => slice,
            LinearRegInterceptData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum LinearRegInterceptData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct LinearRegInterceptOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct LinearRegInterceptParams {
    pub period: Option<usize>,
}

impl Default for LinearRegInterceptParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct LinearRegInterceptInput<'a> {
    pub data: LinearRegInterceptData<'a>,
    pub params: LinearRegInterceptParams,
}

impl<'a> LinearRegInterceptInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: LinearRegInterceptParams) -> Self {
        Self {
            data: LinearRegInterceptData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: LinearRegInterceptParams) -> Self {
        Self {
            data: LinearRegInterceptData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", LinearRegInterceptParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct LinearRegInterceptBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for LinearRegInterceptBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl LinearRegInterceptBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<LinearRegInterceptOutput, LinearRegInterceptError> {
        let p = LinearRegInterceptParams {
            period: self.period,
        };
        let i = LinearRegInterceptInput::from_candles(c, "close", p);
        linearreg_intercept_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(
        self,
        d: &[f64],
    ) -> Result<LinearRegInterceptOutput, LinearRegInterceptError> {
        let p = LinearRegInterceptParams {
            period: self.period,
        };
        let i = LinearRegInterceptInput::from_slice(d, p);
        linearreg_intercept_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<LinearRegInterceptStream, LinearRegInterceptError> {
        let p = LinearRegInterceptParams {
            period: self.period,
        };
        LinearRegInterceptStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum LinearRegInterceptError {
    #[error("linearreg_intercept: Input data slice is empty.")]
    EmptyInputData,
    #[error("linearreg_intercept: All values are NaN.")]
    AllValuesNaN,
    #[error("linearreg_intercept: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("linearreg_intercept: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("linearreg_intercept: Output length mismatch: expected {expected}, got {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("linearreg_intercept: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("linearreg_intercept: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline]
pub fn linearreg_intercept(
    input: &LinearRegInterceptInput,
) -> Result<LinearRegInterceptOutput, LinearRegInterceptError> {
    linearreg_intercept_with_kernel(input, Kernel::Auto)
}

pub fn linearreg_intercept_with_kernel(
    input: &LinearRegInterceptInput,
    kernel: Kernel,
) -> Result<LinearRegInterceptOutput, LinearRegInterceptError> {
    let data: &[f64] = input.as_ref();

    if data.is_empty() {
        return Err(LinearRegInterceptError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(LinearRegInterceptError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(LinearRegInterceptError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(LinearRegInterceptError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let mut out = alloc_with_nan_prefix(len, first + period - 1);

    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                linearreg_intercept_scalar(data, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                linearreg_intercept_avx2(data, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                linearreg_intercept_avx512(data, period, first, &mut out)
            }
            _ => unreachable!(),
        }
    }

    Ok(LinearRegInterceptOutput { values: out })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn linearreg_intercept_into(
    input: &LinearRegInterceptInput,
    dst: &mut [f64],
) -> Result<(), LinearRegInterceptError> {
    let data: &[f64] = input.as_ref();

    if data.is_empty() {
        return Err(LinearRegInterceptError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(LinearRegInterceptError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(LinearRegInterceptError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(LinearRegInterceptError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    if dst.len() != data.len() {
        return Err(LinearRegInterceptError::OutputLengthMismatch {
            expected: data.len(),
            got: dst.len(),
        });
    }

    let warmup_end = first + period - 1;
    for v in &mut dst[..warmup_end] {
        *v = f64::from_bits(0x7ff8_0000_0000_0000);
    }

    let chosen = detect_best_kernel();

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                linearreg_intercept_scalar(data, period, first, dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => linearreg_intercept_avx2(data, period, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                linearreg_intercept_avx512(data, period, first, dst)
            }
            _ => unreachable!(),
        }
    }

    Ok(())
}

#[inline]
pub fn linearreg_intercept_into_slice(
    dst: &mut [f64],
    input: &LinearRegInterceptInput,
    kern: Kernel,
) -> Result<(), LinearRegInterceptError> {
    let data: &[f64] = input.as_ref();

    if data.is_empty() {
        return Err(LinearRegInterceptError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(LinearRegInterceptError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(LinearRegInterceptError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(LinearRegInterceptError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    if dst.len() != data.len() {
        return Err(LinearRegInterceptError::OutputLengthMismatch {
            expected: data.len(),
            got: dst.len(),
        });
    }

    let chosen = match kern {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                linearreg_intercept_scalar(data, period, first, dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => linearreg_intercept_avx2(data, period, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                linearreg_intercept_avx512(data, period, first, dst)
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

#[inline]
pub fn linearreg_intercept_scalar(data: &[f64], period: usize, first_val: usize, out: &mut [f64]) {
    if period == 1 {
        for i in first_val..data.len() {
            out[i] = data[i];
        }
        return;
    }

    let n = period as f64;
    let inv_n = 1.0 / n;

    let sum_x = 0.5_f64 * n * (n + 1.0);
    let sum_x2 = (n * (n + 1.0) * (2.0 * n + 1.0)) / 6.0;
    let denom = n * sum_x2 - sum_x * sum_x;
    let bd = 1.0 / denom;
    let k = 1.0 - sum_x * inv_n;

    let start = first_val;
    let end = data.len();
    if end == 0 || end < start + period {
        return;
    }

    let mut sum_y = 0.0f64;
    let mut sum_xy = 0.0f64;
    for j in 0..period {
        let y = data[start + j];
        let x = (j as f64) + 1.0;
        sum_y += y;
        sum_xy += y * x;
    }

    let mut i = start + period - 1;
    out[i] = ((n * sum_xy - sum_x * sum_y) * bd) * k + sum_y * inv_n;

    while i + 1 < end {
        let y_in = data[i + 1];
        let y_out = data[i + 1 - period];

        let prev_sum_y = sum_y;
        sum_y = prev_sum_y + y_in - y_out;
        sum_xy = (sum_xy - prev_sum_y) + n * y_in;

        i += 1;
        out[i] = ((n * sum_xy - sum_x * sum_y) * bd) * k + sum_y * inv_n;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn linearreg_intercept_avx512(data: &[f64], period: usize, first_val: usize, out: &mut [f64]) {
    if period <= 32 {
        unsafe { linearreg_intercept_avx512_short(data, period, first_val, out) }
    } else {
        unsafe { linearreg_intercept_avx512_long(data, period, first_val, out) }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn linearreg_intercept_avx2(
    data: &[f64],
    period: usize,
    first_val: usize,
    out: &mut [f64],
) {
    if period == 1 {
        let mut i = first_val;
        let end = data.len();
        let src = data.as_ptr();
        let dst = out.as_mut_ptr();
        while i < end {
            *dst.add(i) = *src.add(i);
            i += 1;
        }
        return;
    }

    let n = period as f64;
    let inv_n = 1.0 / n;
    let sum_x = 0.5_f64 * n * (n + 1.0);
    let sum_x2 = (n * (n + 1.0) * (2.0 * n + 1.0)) / 6.0;
    let denom = n.mul_add(sum_x2, -sum_x * sum_x);
    let bd = 1.0 / denom;
    let k = 1.0 - sum_x * inv_n;

    let start = first_val;
    let end = data.len();
    if end == 0 || end < start + period {
        return;
    }

    let mut sum_y = 0.0f64;
    let mut sum_xy = 0.0f64;
    let base = data.as_ptr().add(start);
    let mut j = 0usize;
    let mut x = 1.0f64;
    while j < period {
        let y = *base.add(j);
        sum_y += y;
        sum_xy = y.mul_add(x, sum_xy);
        x += 1.0;
        j += 1;
    }

    let mut i = start + period - 1;
    let outp = out.as_mut_ptr();
    let mut b = n.mul_add(sum_xy, -sum_x * sum_y) * bd;
    *outp.add(i) = b.mul_add(k, sum_y * inv_n);

    let dptr = data.as_ptr();
    while i + 1 < end {
        let y_in = *dptr.add(i + 1);
        let y_out = *dptr.add(i + 1 - period);

        let prev_sum_y = sum_y;
        sum_y = prev_sum_y + y_in - y_out;
        sum_xy = (sum_xy - prev_sum_y) + n * y_in;

        i += 1;
        b = n.mul_add(sum_xy, -sum_x * sum_y) * bd;
        *outp.add(i) = b.mul_add(k, sum_y * inv_n);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn linearreg_intercept_avx512_short(
    data: &[f64],
    period: usize,
    first_val: usize,
    out: &mut [f64],
) {
    linearreg_intercept_avx2(data, period, first_val, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn linearreg_intercept_avx512_long(
    data: &[f64],
    period: usize,
    first_val: usize,
    out: &mut [f64],
) {
    linearreg_intercept_avx2(data, period, first_val, out)
}

#[derive(Debug, Clone)]
pub struct LinearRegInterceptStream {
    period: usize,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,
    sum_x: f64,
    sum_x2: f64,
    n: f64,
    bd: f64,
    sum_y: f64,
    sum_xy: f64,
}

impl LinearRegInterceptStream {
    #[inline]
    pub fn try_new(params: LinearRegInterceptParams) -> Result<Self, LinearRegInterceptError> {
        let period = params.period.unwrap_or(14);
        if period == 0 {
            return Err(LinearRegInterceptError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }

        let n = period as f64;
        let sum_x = 0.5_f64 * n * (n + 1.0);
        let sum_x2 = (n * (n + 1.0) * (2.0 * n + 1.0)) / 6.0;
        let denom = n * sum_x2 - sum_x * sum_x;
        let bd = if period == 1 { 0.0 } else { 1.0 / denom };

        Ok(Self {
            period,
            buffer: vec![f64::NAN; period],
            head: 0,
            filled: false,
            sum_x,
            sum_x2,
            n,
            bd,
            sum_y: 0.0,
            sum_xy: 0.0,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        if self.period == 1 {
            return Some(value);
        }

        let tail = self.head;
        let y_out = self.buffer[tail];

        self.buffer[tail] = value;
        self.head = if self.head + 1 == self.period {
            0
        } else {
            self.head + 1
        };

        if !self.filled {
            let x = (tail as f64) + 1.0;
            self.sum_y += value;
            self.sum_xy = value.mul_add(x, self.sum_xy);

            if self.head == 0 {
                self.filled = true;
            } else {
                return None;
            }
        } else {
            let sum_y_old = self.sum_y;
            self.sum_y = sum_y_old + value - y_out;

            self.sum_xy = (self.sum_xy - sum_y_old) + self.n * value;
        }

        let inv_n = 1.0 / self.n;
        let k = 1.0 - self.sum_x * inv_n;

        let t = self.n.mul_add(self.sum_xy, -(self.sum_x * self.sum_y));
        let b = t * self.bd;
        let y = self.sum_y.mul_add(inv_n, b * k);
        Some(y)
    }
}

#[derive(Clone, Debug)]
pub struct LinearRegInterceptBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for LinearRegInterceptBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 263, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct LinearRegInterceptBatchBuilder {
    range: LinearRegInterceptBatchRange,
    kernel: Kernel,
}

impl LinearRegInterceptBatchBuilder {
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
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<LinearRegInterceptBatchOutput, LinearRegInterceptError> {
        linearreg_intercept_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(
        data: &[f64],
        k: Kernel,
    ) -> Result<LinearRegInterceptBatchOutput, LinearRegInterceptError> {
        LinearRegInterceptBatchBuilder::new()
            .kernel(k)
            .apply_slice(data)
    }
    pub fn apply_candles(
        self,
        c: &Candles,
        src: &str,
    ) -> Result<LinearRegInterceptBatchOutput, LinearRegInterceptError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(
        c: &Candles,
    ) -> Result<LinearRegInterceptBatchOutput, LinearRegInterceptError> {
        LinearRegInterceptBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn linearreg_intercept_batch_with_kernel(
    data: &[f64],
    sweep: &LinearRegInterceptBatchRange,
    k: Kernel,
) -> Result<LinearRegInterceptBatchOutput, LinearRegInterceptError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(LinearRegInterceptError::InvalidKernelForBatch(other)),
    };

    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    linearreg_intercept_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct LinearRegInterceptBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<LinearRegInterceptParams>,
    pub rows: usize,
    pub cols: usize,
}
impl LinearRegInterceptBatchOutput {
    pub fn row_for_params(&self, p: &LinearRegInterceptParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
    }
    pub fn values_for(&self, p: &LinearRegInterceptParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(
    r: &LinearRegInterceptBatchRange,
) -> Result<Vec<LinearRegInterceptParams>, LinearRegInterceptError> {
    fn axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, LinearRegInterceptError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }

        let mut values = Vec::new();
        let step_u = step;

        if start <= end {
            let mut v = start;
            loop {
                if v > end {
                    break;
                }
                values.push(v);
                match v.checked_add(step_u) {
                    Some(next) => v = next,
                    None => break,
                }
            }
        } else {
            let mut v = start;
            loop {
                if v < end {
                    break;
                }
                values.push(v);
                match v.checked_sub(step_u) {
                    Some(next) => v = next,
                    None => break,
                }
            }
        }

        if values.is_empty() {
            return Err(LinearRegInterceptError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }

        Ok(values)
    }

    let periods = axis_usize(r.period)?;

    let mut out = Vec::with_capacity(periods.len());
    for p in periods {
        out.push(LinearRegInterceptParams { period: Some(p) });
    }
    Ok(out)
}

#[inline(always)]
pub fn linearreg_intercept_batch_slice(
    data: &[f64],
    sweep: &LinearRegInterceptBatchRange,
    kern: Kernel,
) -> Result<LinearRegInterceptBatchOutput, LinearRegInterceptError> {
    linearreg_intercept_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn linearreg_intercept_batch_par_slice(
    data: &[f64],
    sweep: &LinearRegInterceptBatchRange,
    kern: Kernel,
) -> Result<LinearRegInterceptBatchOutput, LinearRegInterceptError> {
    linearreg_intercept_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn linearreg_intercept_batch_inner_into(
    data: &[f64],
    sweep: &LinearRegInterceptBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<LinearRegInterceptParams>, LinearRegInterceptError> {
    if data.is_empty() {
        return Err(LinearRegInterceptError::EmptyInputData);
    }

    let combos = expand_grid(sweep)?;

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(LinearRegInterceptError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(LinearRegInterceptError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let cols = data.len();

    let chosen = match kern {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                linearreg_intercept_row_scalar(data, first, period, out_row)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                linearreg_intercept_row_avx2(data, first, period, out_row)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                linearreg_intercept_row_avx512(data, first, period, out_row)
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

    Ok(combos)
}

#[inline(always)]
fn linearreg_intercept_batch_inner(
    data: &[f64],
    sweep: &LinearRegInterceptBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<LinearRegInterceptBatchOutput, LinearRegInterceptError> {
    if data.is_empty() {
        return Err(LinearRegInterceptError::EmptyInputData);
    }

    let combos = expand_grid(sweep)?;

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(LinearRegInterceptError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(LinearRegInterceptError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();

    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| LinearRegInterceptError::InvalidRange {
            start: sweep.period.0.to_string(),
            end: sweep.period.1.to_string(),
            step: sweep.period.2.to_string(),
        })?;

    let mut buf_mu = make_uninit_matrix(rows, cols);

    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    let mut values = unsafe {
        let ptr = buf_mu.as_mut_ptr() as *mut f64;
        std::mem::forget(buf_mu);
        Vec::from_raw_parts(ptr, total, total)
    };

    let chosen = match kern {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                linearreg_intercept_row_scalar(data, first, period, out_row)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                linearreg_intercept_row_avx2(data, first, period, out_row)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                linearreg_intercept_row_avx512(data, first, period, out_row)
            }
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

    Ok(LinearRegInterceptBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn linearreg_intercept_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    linearreg_intercept_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn linearreg_intercept_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    linearreg_intercept_avx2(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn linearreg_intercept_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    linearreg_intercept_avx512(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn linearreg_intercept_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    linearreg_intercept_avx512_short(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn linearreg_intercept_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    linearreg_intercept_avx512_long(data, period, first, out)
}

#[inline(always)]
fn expand_grid_reg(r: &LinearRegInterceptBatchRange) -> Vec<LinearRegInterceptParams> {
    expand_grid(r).unwrap_or_else(|_| Vec::new())
}

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

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", unsendable)]
pub struct LinearRegInterceptDeviceArrayF32Py {
    pub(crate) buf: Option<DeviceBuffer<f32>>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) ctx: Arc<Context>,
    pub(crate) device_id: u32,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl LinearRegInterceptDeviceArrayF32Py {
    #[getter]
    fn __cuda_array_interface__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let d = pyo3::types::PyDict::new(py);
        d.set_item("shape", (self.rows, self.cols))?;
        d.set_item("typestr", "<f4")?;
        d.set_item(
            "strides",
            (
                self.cols * std::mem::size_of::<f32>(),
                std::mem::size_of::<f32>(),
            ),
        )?;
        let ptr = self
            .buf
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("buffer already exported via __dlpack__"))?
            .as_device_ptr()
            .as_raw() as usize;
        d.set_item("data", (ptr, false))?;

        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        (2, self.device_id as i32)
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

        let buf = self
            .buf
            .take()
            .ok_or_else(|| PyValueError::new_err("__dlpack__ may only be called once"))?;

        let rows = self.rows;
        let cols = self.cols;

        let max_version_bound = max_version.map(|obj| obj.into_bound(py));

        export_f32_cuda_dlpack_2d(py, buf, rows, cols, alloc_dev, max_version_bound)
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "linearreg_intercept_cuda_batch_dev")]
#[pyo3(signature = (data, period_range, device_id=0))]
pub fn linearreg_intercept_cuda_batch_dev_py(
    py: Python<'_>,
    data: numpy::PyReadonlyArray1<'_, f32>,
    period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<LinearRegInterceptDeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    use crate::cuda::CudaLinregIntercept;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let slice = data.as_slice()?;
    let sweep = LinearRegInterceptBatchRange {
        period: period_range,
    };
    let (dev, ctx, dev_id) = py.allow_threads(|| {
        let cuda = CudaLinregIntercept::new(device_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let (dev, _combos) = cuda
            .linearreg_intercept_batch_dev(slice, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.context_arc();
        Ok::<_, pyo3::PyErr>((dev, ctx, cuda.device_id()))
    })?;
    let DeviceArrayF32 { buf, rows, cols } = dev;
    Ok(LinearRegInterceptDeviceArrayF32Py {
        buf: Some(buf),
        rows,
        cols,
        ctx,
        device_id: dev_id,
    })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "linearreg_intercept_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm, cols, rows, period, device_id=0))]
pub fn linearreg_intercept_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    data_tm: numpy::PyReadonlyArray1<'_, f32>,
    cols: usize,
    rows: usize,
    period: usize,
    device_id: usize,
) -> PyResult<LinearRegInterceptDeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    use crate::cuda::CudaLinregIntercept;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let slice = data_tm.as_slice()?;
    let expected = cols
        .checked_mul(rows)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;
    if slice.len() != expected {
        return Err(PyValueError::new_err("time-major input length mismatch"));
    }
    let params = LinearRegInterceptParams {
        period: Some(period),
    };
    let (dev, ctx, dev_id) = py.allow_threads(|| {
        let cuda = CudaLinregIntercept::new(device_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dev = cuda
            .linearreg_intercept_many_series_one_param_time_major_dev(slice, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.context_arc();
        Ok::<_, pyo3::PyErr>((dev, ctx, cuda.device_id()))
    })?;
    Ok(LinearRegInterceptDeviceArrayF32Py {
        buf: Some(dev.buf),
        rows: dev.rows,
        cols: dev.cols,
        ctx,
        device_id: dev_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;

    fn check_linreg_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = LinearRegInterceptParams { period: None };
        let input = LinearRegInterceptInput::from_candles(&candles, "close", default_params);
        let output = linearreg_intercept_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_linreg_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = LinearRegInterceptInput::from_candles(
            &candles,
            "close",
            LinearRegInterceptParams::default(),
        );
        let result = linearreg_intercept_with_kernel(&input, kernel)?;
        let expected_last_five = [
            60000.91428571429,
            59947.142857142855,
            59754.57142857143,
            59318.4,
            59321.91428571429,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-1,
                "[{}] LinReg {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_linreg_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = LinearRegInterceptInput::with_default_candles(&candles);
        match input.data {
            LinearRegInterceptData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected LinearRegInterceptData::Candles"),
        }
        let output = linearreg_intercept_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_linreg_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = LinearRegInterceptParams { period: Some(0) };
        let input = LinearRegInterceptInput::from_slice(&input_data, params);
        let res = linearreg_intercept_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] LinReg should fail with zero period",
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
        let params = LinearRegInterceptParams { period: Some(10) };
        let input = LinearRegInterceptInput::from_slice(&data_small, params);
        let res = linearreg_intercept_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] LinReg should fail with period exceeding length",
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
        let params = LinearRegInterceptParams { period: Some(14) };
        let input = LinearRegInterceptInput::from_slice(&single_point, params);
        let res = linearreg_intercept_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] LinReg should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_linreg_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = LinearRegInterceptParams { period: Some(14) };
        let first_input = LinearRegInterceptInput::from_candles(&candles, "close", first_params);
        let first_result = linearreg_intercept_with_kernel(&first_input, kernel)?;
        let second_params = LinearRegInterceptParams { period: Some(14) };
        let second_input = LinearRegInterceptInput::from_slice(&first_result.values, second_params);
        let second_result = linearreg_intercept_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());

        let start = second_result
            .values
            .iter()
            .position(|v| !v.is_nan())
            .unwrap_or(second_result.values.len());

        for (i, v) in second_result.values[start..].iter().enumerate() {
            assert!(
                !v.is_nan(),
                "[{}] Unexpected NaN at index {} after reinput",
                test_name,
                start + i
            );
        }
        Ok(())
    }

    fn check_linreg_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = LinearRegInterceptInput::from_candles(
            &candles,
            "close",
            LinearRegInterceptParams { period: Some(14) },
        );
        let res = linearreg_intercept_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 40 {
            for (i, &val) in res.values[40..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test_name,
                    40 + i
                );
            }
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_linreg_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let test_params = vec![
            LinearRegInterceptParams::default(),
            LinearRegInterceptParams { period: Some(2) },
            LinearRegInterceptParams { period: Some(5) },
            LinearRegInterceptParams { period: Some(7) },
            LinearRegInterceptParams { period: Some(10) },
            LinearRegInterceptParams { period: Some(20) },
            LinearRegInterceptParams { period: Some(30) },
            LinearRegInterceptParams { period: Some(50) },
            LinearRegInterceptParams { period: Some(100) },
            LinearRegInterceptParams { period: Some(150) },
            LinearRegInterceptParams { period: Some(200) },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = LinearRegInterceptInput::from_candles(&candles, "close", params.clone());
            let output = linearreg_intercept_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(14),
                        param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(14),
                        param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(14),
                        param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_linreg_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_linearreg_intercept_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        fn calculate_expected_linreg_intercept(
            window_start_idx: usize,
            period: usize,
            data_slope: f64,
            data_intercept: f64,
        ) -> f64 {
            data_slope * window_start_idx as f64 + data_intercept
        }

        let strat = (1usize..=100, 50usize..500, 0usize..5, any::<u64>()).prop_map(
            |(period, len, scenario, seed)| {
                let mut rng_state = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                let mut data = Vec::with_capacity(len);

                match scenario {
                    0 => {
                        for _ in 0..len {
                            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
                            let val = (rng_state as f64 / u64::MAX as f64) * 200.0 - 100.0;
                            data.push(val);
                        }
                    }
                    1 => {
                        let constant = 42.0;
                        data.resize(len, constant);
                    }
                    2 => {
                        for i in 0..len {
                            data.push(2.0 * i as f64 + 10.0);
                        }
                    }
                    3 => {
                        for i in 0..len {
                            data.push(-1.5 * i as f64 + 100.0);
                        }
                    }
                    _ => {
                        for i in 0..len {
                            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
                            let noise = ((rng_state as f64 / u64::MAX as f64) - 0.5) * 10.0;
                            data.push(0.5 * i as f64 + 50.0 + noise);
                        }
                    }
                }

                (data, period, scenario)
            },
        );

        proptest::test_runner::TestRunner::default().run(&strat, |(data, period, scenario)| {
            let params = LinearRegInterceptParams {
                period: Some(period),
            };
            let input = LinearRegInterceptInput::from_slice(&data, params);

            let output = linearreg_intercept_with_kernel(&input, kernel)?;

            let ref_output = linearreg_intercept_with_kernel(&input, Kernel::Scalar)?;

            prop_assert_eq!(
                output.values.len(),
                data.len(),
                "[{}] Output length mismatch",
                test_name
            );

            if period == 1 {
                for i in 0..data.len() {
                    let expected = data[i];
                    let actual = output.values[i];
                    prop_assert!(
                        (actual - expected).abs() < 1e-9,
                        "[{}] Period=1: expected {}, got {} at index {}",
                        test_name,
                        expected,
                        actual,
                        i
                    );
                }
            } else {
                for i in 0..(period - 1) {
                    prop_assert!(
                        output.values[i].is_nan(),
                        "[{}] Expected NaN during warmup at index {}",
                        test_name,
                        i
                    );
                }

                if period <= data.len() {
                    prop_assert!(
                        !output.values[period - 1].is_nan(),
                        "[{}] Expected valid value at index {}",
                        test_name,
                        period - 1
                    );
                }
            }

            if scenario == 1 && period < data.len() {
                for i in (period - 1)..data.len() {
                    let intercept = output.values[i];
                    if !intercept.is_nan() {
                        prop_assert!(
                            (intercept - 42.0).abs() < 1e-9,
                            "[{}] Constant data: expected 42.0, got {} at index {}",
                            test_name,
                            intercept,
                            i
                        );
                    }
                }
            }

            if (scenario == 2 || scenario == 3) && period > 1 && period < data.len() {
                let (data_slope, data_intercept) = match scenario {
                    2 => (2.0, 10.0),
                    3 => (-1.5, 100.0),
                    _ => unreachable!(),
                };

                for i in (period - 1)..data.len().min(period * 5) {
                    let actual = output.values[i];
                    if !actual.is_nan() {
                        let window_start = i + 1 - period;
                        let expected = calculate_expected_linreg_intercept(
                            window_start,
                            period,
                            data_slope,
                            data_intercept,
                        );

                        prop_assert!((actual - expected).abs() < 1e-9,
								"[{}] Linear trend (scenario {}): expected {:.6}, got {:.6} at index {} (window start {})",
								test_name, scenario, expected, actual, i, window_start);
                    }
                }
            }

            for i in 0..output.values.len() {
                let y = output.values[i];
                let r = ref_output.values[i];

                let bits = y.to_bits();
                prop_assert!(
                    bits != 0x11111111_11111111
                        && bits != 0x22222222_22222222
                        && bits != 0x33333333_33333333,
                    "[{}] Found poison value at index {}: 0x{:016X}",
                    test_name,
                    i,
                    bits
                );

                if y.is_nan() && r.is_nan() {
                    continue;
                }

                if y.is_finite() && r.is_finite() {
                    prop_assert!(
                        (y - r).abs() <= 1e-9,
                        "[{}] Kernel mismatch at index {}: {} vs {} (diff: {})",
                        test_name,
                        i,
                        y,
                        r,
                        (y - r).abs()
                    );
                } else {
                    prop_assert_eq!(
                        y.is_nan(),
                        r.is_nan(),
                        "[{}] NaN mismatch at index {}: {} vs {}",
                        test_name,
                        i,
                        y,
                        r
                    );
                }
            }

            if period > 1 {
                for i in (period - 1)..output.values.len() {
                    let val = output.values[i];
                    prop_assert!(
                        val.is_finite(),
                        "[{}] Non-finite value {} at index {}",
                        test_name,
                        val,
                        i
                    );
                }
            }

            Ok(())
        })?;

        Ok(())
    }

    macro_rules! generate_all_linreg_tests {
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

    generate_all_linreg_tests!(
        check_linreg_partial_params,
        check_linreg_accuracy,
        check_linreg_default_candles,
        check_linreg_zero_period,
        check_linreg_period_exceeds_length,
        check_linreg_very_small_dataset,
        check_linreg_reinput,
        check_linreg_nan_handling,
        check_linreg_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_linreg_tests!(check_linearreg_intercept_property);

    #[test]
    fn test_linearreg_intercept_into_matches_api() -> Result<(), Box<dyn Error>> {
        let len = 256usize;
        let mut data = Vec::with_capacity(len);
        for i in 0..len {
            if i < 10 {
                data.push(f64::NAN);
            } else {
                let x = i as f64;
                data.push((0.1 * x).sin() * 3.0 + 0.05 * x + 2.0);
            }
        }

        let input = LinearRegInterceptInput::from_slice(&data, LinearRegInterceptParams::default());

        let baseline = linearreg_intercept(&input)?.values;

        let mut out = vec![0.0; data.len()];
        #[allow(unused_variables)]
        {
            #[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
            {
                linearreg_intercept_into(&input, &mut out)?;
            }
            #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
            {
                linearreg_intercept_into_slice(&mut out, &input, Kernel::Auto)?;
            }
        }

        assert_eq!(baseline.len(), out.len());

        fn eq_or_both_nan_eps(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12
        }

        for i in 0..out.len() {
            assert!(
                eq_or_both_nan_eps(baseline[i], out[i]),
                "mismatch at index {}: baseline={} out={}",
                i,
                baseline[i],
                out[i]
            );
        }

        Ok(())
    }

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = LinearRegInterceptBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = LinearRegInterceptParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            60000.91428571429,
            59947.142857142855,
            59754.57142857143,
            59318.4,
            59321.91428571429,
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
            (2, 10, 2),
            (5, 25, 5),
            (10, 10, 0),
            (2, 5, 1),
            (30, 60, 15),
            (2, 14, 3),
            (50, 100, 25),
            (100, 200, 50),
        ];

        for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
            let output = LinearRegInterceptBatchBuilder::new()
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
                        combo.period.unwrap_or(14)
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
                        combo.period.unwrap_or(14)
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
                        combo.period.unwrap_or(14)
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
}

#[cfg(feature = "python")]
#[pyfunction(name = "linearreg_intercept")]
#[pyo3(signature = (data, period, kernel=None))]
pub fn linearreg_intercept_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let params = LinearRegInterceptParams {
        period: Some(period),
    };
    let input = LinearRegInterceptInput::from_slice(slice_in, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| linearreg_intercept_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "LinearRegInterceptStream")]
pub struct LinearRegInterceptStreamPy {
    stream: LinearRegInterceptStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl LinearRegInterceptStreamPy {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        let params = LinearRegInterceptParams {
            period: Some(period),
        };
        let stream = LinearRegInterceptStream::try_new(params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(LinearRegInterceptStreamPy { stream })
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "linearreg_intercept_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
pub fn linearreg_intercept_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;
    let sweep = LinearRegInterceptBatchRange {
        period: period_range,
    };

    let combos = expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = slice_in.len();

    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("linearreg_intercept_batch: rows*cols overflow"))?;

    let out_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    if !combos.is_empty() && cols > 0 {
        let first = slice_in.iter().position(|x| !x.is_nan()).unwrap_or(0);
        for (r, prm) in combos.iter().enumerate() {
            let warm = (first + prm.period.unwrap() - 1).min(cols);
            for v in &mut slice_out[r * cols..r * cols + warm] {
                *v = f64::NAN;
            }
        }
    }

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
            linearreg_intercept_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
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
pub fn linearreg_intercept_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = LinearRegInterceptParams {
        period: Some(period),
    };
    let input = LinearRegInterceptInput::from_slice(data, params);

    let mut output = vec![0.0; data.len()];
    linearreg_intercept_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn linearreg_intercept_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn linearreg_intercept_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn linearreg_intercept_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let params = LinearRegInterceptParams {
            period: Some(period),
        };
        let input = LinearRegInterceptInput::from_slice(data, params);

        if in_ptr == out_ptr {
            let mut temp = vec![0.0; len];
            linearreg_intercept_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            linearreg_intercept_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct LinearRegInterceptBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct LinearRegInterceptBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<LinearRegInterceptParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = linearreg_intercept_batch)]
pub fn linearreg_intercept_batch_unified_js(
    data: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: LinearRegInterceptBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = LinearRegInterceptBatchRange {
        period: config.period_range,
    };

    let batch_output = linearreg_intercept_batch_with_kernel(data, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let rows = batch_output.values.len() / data.len();
    let result = LinearRegInterceptBatchJsOutput {
        values: batch_output.values,
        combos: batch_output.combos,
        rows,
        cols: data.len(),
    };

    serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn linearreg_intercept_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let sweep = LinearRegInterceptBatchRange {
            period: (period_start, period_end, period_step),
        };
        let combos = expand_grid(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let cols = len;
        let total = rows.checked_mul(cols).ok_or_else(|| {
            JsValue::from_str("linearreg_intercept_batch_into: rows*cols overflow")
        })?;

        let first = data.iter().position(|x| !x.is_nan()).unwrap_or(0);

        if in_ptr == out_ptr {
            let mut temp = vec![0.0; total];

            for (r, prm) in combos.iter().enumerate() {
                let warm = (first + prm.period.unwrap() - 1).min(cols);
                temp[r * cols..r * cols + warm].fill(f64::NAN);
            }

            linearreg_intercept_batch_inner_into(data, &sweep, Kernel::Auto, true, &mut temp)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, total);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, total);

            for (r, prm) in combos.iter().enumerate() {
                let warm = (first + prm.period.unwrap() - 1).min(cols);
                out[r * cols..r * cols + warm].fill(f64::NAN);
            }

            linearreg_intercept_batch_inner_into(data, &sweep, Kernel::Auto, true, out)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(rows)
    }
}
