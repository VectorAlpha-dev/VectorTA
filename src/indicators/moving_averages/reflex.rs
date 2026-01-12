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
use crate::cuda::cuda_available;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::CudaReflex;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::DeviceArrayF32;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::utilities::dlpack_cuda::DeviceArrayF32Py;
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

impl<'a> AsRef<[f64]> for ReflexInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            ReflexData::Slice(slice) => slice,
            ReflexData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ReflexData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct ReflexOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
pub struct ReflexParams {
    pub period: Option<usize>,
}

impl Default for ReflexParams {
    fn default() -> Self {
        Self { period: Some(20) }
    }
}

#[derive(Debug, Clone)]
pub struct ReflexInput<'a> {
    pub data: ReflexData<'a>,
    pub params: ReflexParams,
}

impl<'a> ReflexInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: ReflexParams) -> Self {
        Self {
            data: ReflexData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: ReflexParams) -> Self {
        Self {
            data: ReflexData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", ReflexParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(20)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ReflexBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for ReflexBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl ReflexBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<ReflexOutput, ReflexError> {
        let p = ReflexParams {
            period: self.period,
        };
        let i = ReflexInput::from_candles(c, "close", p);
        reflex_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<ReflexOutput, ReflexError> {
        let p = ReflexParams {
            period: self.period,
        };
        let i = ReflexInput::from_slice(d, p);
        reflex_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<ReflexStream, ReflexError> {
        let p = ReflexParams {
            period: self.period,
        };
        ReflexStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum ReflexError {
    #[error("reflex: No data available (input data slice is empty).")]
    EmptyInputData,
    #[error("reflex: All values are NaN.")]
    AllValuesNaN,
    #[error("reflex: period must be >=2 (period = {period}, data length = {data_len})")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("reflex: Not enough data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("reflex: output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("reflex: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("reflex: invalid range: start = {start}, end = {end}, step = {step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
}

#[inline]
pub fn reflex(input: &ReflexInput) -> Result<ReflexOutput, ReflexError> {
    reflex_with_kernel(input, Kernel::Auto)
}

pub fn reflex_with_kernel(
    input: &ReflexInput,
    kernel: Kernel,
) -> Result<ReflexOutput, ReflexError> {
    let (data, period, first, chosen) = reflex_prepare(input, kernel)?;
    let len = data.len();

    let mut out = alloc_with_nan_prefix(len, period);

    reflex_compute_into(data, period, first, chosen, &mut out);

    out[..period.min(len)].fill(0.0);

    Ok(ReflexOutput { values: out })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn reflex_into(input: &ReflexInput, out: &mut [f64]) -> Result<(), ReflexError> {
    reflex_into_slice(out, input, Kernel::Auto)
}

#[inline]
pub fn reflex_into_slice(
    dst: &mut [f64],
    input: &ReflexInput,
    kern: Kernel,
) -> Result<(), ReflexError> {
    let (data, period, first, chosen) = reflex_prepare(input, kern)?;

    if dst.len() != data.len() {
        return Err(ReflexError::OutputLengthMismatch {
            expected: data.len(),
            got: dst.len(),
        });
    }

    reflex_compute_into(data, period, first, chosen, dst);

    let end = period.min(dst.len());
    for x in &mut dst[..end] {
        *x = 0.0;
    }

    Ok(())
}

#[inline]
pub fn reflex_scalar(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    let len = data.len();
    if len < 2 || period < 2 {
        return;
    }

    let half_p = (period / 2).max(1) as f64;
    let a = (-1.414_f64 * std::f64::consts::PI / half_p).exp();
    let a2 = a * a;
    let b = 2.0 * a * (1.414_f64 * std::f64::consts::PI / half_p).cos();
    let c = 0.5 * (1.0 + a2 - b);

    let ring_len = period + 1;
    let mut ssf = vec![0.0_f64; ring_len];

    ssf[0] = data[0];
    if len > 1 {
        ssf[1] = data[1];
    }

    let mut ssf_sum = ssf[0] + ssf[1];

    let inv_p = 1.0 / (period as f64);
    let alpha = 0.5 * (1.0 + inv_p);
    let beta = 1.0 - alpha;

    let mut ms = 0.0_f64;

    let d_ptr = data.as_ptr();
    let o_ptr = out.as_mut_ptr();

    let mut idx_im2 = 0usize;
    let mut idx_im1 = 1usize;
    let mut idx = 2usize;

    unsafe {
        let mut i = 2usize;
        while i < len {
            let di = *d_ptr.add(i);
            let dim1 = *d_ptr.add(i - 1);
            let ssf_im1 = *ssf.get_unchecked(idx_im1);
            let ssf_im2 = *ssf.get_unchecked(idx_im2);

            let t0 = c * (di + dim1);
            let t1 = (-a2).mul_add(ssf_im2, t0);
            let ssf_i = b.mul_add(ssf_im1, t1);

            *ssf.get_unchecked_mut(idx) = ssf_i;

            if i < period {
                ssf_sum += ssf_i;
            } else {
                let mut idx_ip = idx + 1;
                if idx_ip == ring_len {
                    idx_ip = 0;
                }
                let ssf_ip = *ssf.get_unchecked(idx_ip);

                let mean_lp = ssf_sum * inv_p;
                let my_sum = ssf_i.mul_add(beta, ssf_ip * alpha) - mean_lp;

                ms = (0.96_f64).mul_add(ms, 0.04_f64 * (my_sum * my_sum));
                if ms > 0.0 {
                    *o_ptr.add(i) = my_sum / ms.sqrt();
                }

                ssf_sum += ssf_i - ssf_ip;
            }

            idx_im2 = idx_im1;
            idx_im1 = idx;
            idx += 1;
            if idx == ring_len {
                idx = 0;
            }

            i += 1;
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn reflex_avx2(data: &[f64], period: usize, _first: usize, out: &mut [f64]) {
    reflex_scalar(data, period, _first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512dq,fma")]
pub unsafe fn reflex_avx512(data: &[f64], period: usize, _first: usize, out: &mut [f64]) {
    reflex_scalar(data, period, _first, out)
}

#[inline(always)]
fn reflex_prepare<'a>(
    input: &'a ReflexInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, usize, Kernel), ReflexError> {
    let data: &[f64] = match &input.data {
        ReflexData::Candles { candles, source } => source_type(candles, source),
        ReflexData::Slice(sl) => sl,
    };

    let len = data.len();
    if len == 0 {
        return Err(ReflexError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(ReflexError::AllValuesNaN)?;
    let period = input.get_period();

    if period < 2 {
        return Err(ReflexError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if period > (len - first) {
        return Err(ReflexError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    Ok((data, period, first, chosen))
}

#[inline(always)]
fn reflex_compute_into(data: &[f64], period: usize, first: usize, kernel: Kernel, out: &mut [f64]) {
    unsafe {
        match kernel {
            Kernel::Scalar | Kernel::ScalarBatch => reflex_scalar(data, period, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => reflex_avx2(data, period, first, out),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch => reflex_scalar(data, period, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => reflex_avx512(data, period, first, out),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx512 | Kernel::Avx512Batch => reflex_scalar(data, period, first, out),
            _ => unreachable!(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReflexStream {
    period: usize,

    a_sq: f64,
    b: f64,
    c: f64,

    alpha: f64,
    beta: f64,
    inv_p: f64,

    ssf_buf: Vec<f64>,
    head: usize,
    tail: usize,

    ssf_sum: f64,
    last_ms: f64,
    prev_x: f64,
    last_ssf1: f64,
    last_ssf2: f64,
    count: usize,
}

impl ReflexStream {
    #[inline]
    pub fn try_new(params: ReflexParams) -> Result<Self, ReflexError> {
        let period = params.period.unwrap_or(20);
        if period < 2 {
            return Err(ReflexError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }

        let half_p = (period / 2).max(1) as f64;
        let a = (-1.414_f64 * std::f64::consts::PI / half_p).exp();
        let a_sq = a * a;
        let b = 2.0 * a * (1.414_f64 * std::f64::consts::PI / half_p).cos();
        let c = 0.5 * (1.0 + a_sq - b);

        let inv_p = 1.0 / (period as f64);
        let alpha = 0.5 * (1.0 + inv_p);
        let beta = 1.0 - alpha;

        Ok(Self {
            period,
            a_sq,
            b,
            c,
            alpha,
            beta,
            inv_p,

            ssf_buf: vec![0.0; period + 1],
            head: 0,
            tail: 0,

            ssf_sum: 0.0,
            last_ms: 0.0,
            prev_x: 0.0,
            last_ssf1: 0.0,
            last_ssf2: 0.0,
            count: 0,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, x: f64) -> Option<f64> {
        let p = self.period;
        let ring_len = p + 1;
        let t = self.count;

        if t == 0 {
            self.prev_x = x;
            self.last_ssf1 = x;
            self.ssf_buf[self.head] = x;
            self.head += 1;
            if self.head == ring_len {
                self.head = 0;
            }
            self.ssf_sum += x;
            self.count = 1;
            return None;
        }
        if t == 1 {
            self.prev_x = x;
            self.last_ssf2 = self.last_ssf1;
            self.last_ssf1 = x;
            self.ssf_buf[self.head] = x;
            self.head += 1;
            if self.head == ring_len {
                self.head = 0;
            }
            self.ssf_sum += x;
            self.count = 2;
            return None;
        }

        let t0 = self.c * (x + self.prev_x);
        let t1 = (-self.a_sq).mul_add(self.last_ssf2, t0);
        let ssf_t = self.b.mul_add(self.last_ssf1, t1);

        let mut out = None;
        if t >= p {
            let ssf_tp = self.ssf_buf[self.tail];

            let mean_lp = self.ssf_sum * self.inv_p;

            let my_sum = self.beta.mul_add(ssf_t, self.alpha * ssf_tp) - mean_lp;

            let ms = 0.96_f64.mul_add(self.last_ms, 0.04_f64 * (my_sum * my_sum));
            self.last_ms = ms;
            out = if ms > 0.0 {
                Some(my_sum / ms.sqrt())
            } else {
                Some(0.0)
            };

            self.ssf_sum += ssf_t - ssf_tp;
            self.tail += 1;
            if self.tail == ring_len {
                self.tail = 0;
            }
        } else {
            self.ssf_sum += ssf_t;
        }

        self.ssf_buf[self.head] = ssf_t;
        self.head += 1;
        if self.head == ring_len {
            self.head = 0;
        }

        self.prev_x = x;
        self.last_ssf2 = self.last_ssf1;
        self.last_ssf1 = ssf_t;
        self.count = t + 1;

        out
    }
}

#[derive(Clone, Debug)]
pub struct ReflexBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for ReflexBatchRange {
    fn default() -> Self {
        Self {
            period: (20, 269, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ReflexBatchBuilder {
    range: ReflexBatchRange,
    kernel: Kernel,
}

impl ReflexBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<ReflexBatchOutput, ReflexError> {
        reflex_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<ReflexBatchOutput, ReflexError> {
        ReflexBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<ReflexBatchOutput, ReflexError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<ReflexBatchOutput, ReflexError> {
        ReflexBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn reflex_batch_with_kernel(
    data: &[f64],
    sweep: &ReflexBatchRange,
    k: Kernel,
) -> Result<ReflexBatchOutput, ReflexError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(ReflexError::InvalidKernelForBatch(other)),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    reflex_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct ReflexBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<ReflexParams>,
    pub rows: usize,
    pub cols: usize,
}

impl ReflexBatchOutput {
    pub fn row_for_params(&self, p: &ReflexParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(20) == p.period.unwrap_or(20))
    }
    pub fn values_for(&self, p: &ReflexParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid_checked(r: &ReflexBatchRange) -> Result<Vec<ReflexParams>, ReflexError> {
    fn axis_usize(range: (usize, usize, usize)) -> Result<Vec<usize>, ReflexError> {
        let (start, end, step) = range;
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        let mut out = Vec::new();
        if start < end {
            let mut cur = start;
            while cur <= end {
                out.push(cur);
                cur = match cur.checked_add(step) {
                    Some(v) => v,
                    None => break,
                };
            }
        } else {
            let mut cur = start;
            while cur >= end {
                out.push(cur);
                cur = match cur.checked_sub(step) {
                    Some(v) => v,
                    None => break,
                };
                if cur == 0 {
                    break;
                }
            }
        }
        if out.is_empty() {
            return Err(ReflexError::InvalidRange { start, end, step });
        }
        Ok(out)
    }
    let periods = axis_usize(r.period)?;
    Ok(periods
        .into_iter()
        .map(|p| ReflexParams { period: Some(p) })
        .collect())
}

#[inline(always)]
pub fn reflex_batch_slice(
    data: &[f64],
    sweep: &ReflexBatchRange,
    kern: Kernel,
) -> Result<ReflexBatchOutput, ReflexError> {
    reflex_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn reflex_batch_par_slice(
    data: &[f64],
    sweep: &ReflexBatchRange,
    kern: Kernel,
) -> Result<ReflexBatchOutput, ReflexError> {
    reflex_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn reflex_batch_inner(
    data: &[f64],
    sweep: &ReflexBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<ReflexBatchOutput, ReflexError> {
    let combos = expand_grid_checked(sweep)?;
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(ReflexError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(ReflexError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();

    let _total = rows.checked_mul(cols).ok_or(ReflexError::InvalidRange {
        start: rows,
        end: cols,
        step: 0,
    })?;

    let mut buf_mu = make_uninit_matrix(rows, cols);
    let warm: Vec<usize> = combos.iter().map(|c| c.period.unwrap()).collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    let kernel = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        other => other,
    };

    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        other => other,
    };

    let meta = reflex_batch_inner_into(data, sweep, simd, parallel, out)?;

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(ReflexBatchOutput {
        values,
        combos: meta.combos,
        rows: meta.rows,
        cols: meta.cols,
    })
}

#[inline(always)]
unsafe fn reflex_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    reflex_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn reflex_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    reflex_avx2(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn reflex_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    reflex_avx512(data, period, first, out)
}

#[inline(always)]
fn reflex_batch_inner_into(
    data: &[f64],
    sweep: &ReflexBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<ReflexBatchMetadata, ReflexError> {
    let combos = expand_grid_checked(sweep)?;

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(ReflexError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(ReflexError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();

    let expected = rows.checked_mul(cols).ok_or(ReflexError::InvalidRange {
        start: rows,
        end: cols,
        step: 0,
    })?;
    if out.len() != expected {
        return Err(ReflexError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    let do_row = |row: usize, dst: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();

        for x in &mut dst[..period.min(cols)] {
            *x = 0.0;
        }

        match kern {
            Kernel::Scalar => reflex_row_scalar(data, first, period, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => reflex_row_avx2(data, first, period, dst),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 => reflex_row_scalar(data, first, period, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => reflex_row_avx512(data, first, period, dst),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx512 => reflex_row_scalar(data, first, period, dst),
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

    Ok(ReflexBatchMetadata { combos, rows, cols })
}

#[derive(Clone, Debug)]
pub struct ReflexBatchMetadata {
    pub combos: Vec<ReflexParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "python")]
#[pyfunction(name = "reflex")]
#[pyo3(signature = (data, period = 20, kernel = None), text_signature = "(data, period=20, kernel=None)")]
pub fn reflex_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    r#"Compute Reflex indicator.

    Parameters
    ----------
    data : numpy.ndarray
        Input data array
    period : int, default=20
        Period for the indicator (must be >= 2)
    kernel : str, optional
        Kernel to use:
        - 'auto' or None: Auto-detect best kernel (default)
        - 'scalar': Use scalar implementation
        - 'avx2': Use AVX2 implementation (if available)
        - 'avx512': Use AVX512 implementation (if available)

    Returns
    -------
    numpy.ndarray
        Reflex values
    "#;

    use numpy::{IntoPyArray, PyArrayMethods};

    let data_slice = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = ReflexParams {
        period: Some(period),
    };
    let input = ReflexInput::from_slice(data_slice, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| reflex_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction(name = "reflex_batch")]
#[pyo3(signature = (data, periods, kernel = None), text_signature = "(data, periods, kernel=None)")]
pub fn reflex_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    periods: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Py<PyDict>> {
    r#"Compute Reflex indicator for multiple periods.

    Parameters
    ----------
    data : numpy.ndarray
        Input data array
    periods : tuple of int
        (start, end, step) for period range
    kernel : str, optional
        Kernel to use (see reflex() for options)

    Returns
    -------
    dict
        Dictionary with 'values' (2D array) and 'periods' (list)
    "#;

    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let data_slice = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    let range = ReflexBatchRange { period: periods };

    let combos = expand_grid_checked(&range)
        .map_err(|e| PyValueError::new_err(format!("reflex batch error: {}", e)))?;
    let rows = combos.len();
    let cols = data_slice.len();

    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let out_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    let metadata = py
        .allow_threads(|| {
            let kernel = match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                k => k,
            };

            let simd = match kernel {
                Kernel::Avx512Batch => Kernel::Avx512,
                Kernel::Avx2Batch => Kernel::Avx2,
                Kernel::ScalarBatch => Kernel::Scalar,
                other => other,
            };

            reflex_batch_inner_into(data_slice, &range, simd, true, slice_out)
        })
        .map_err(|e| PyValueError::new_err(format!("reflex batch error: {}", e)))?;

    let dict = PyDict::new(py);

    let reshaped = out_arr.reshape([rows, cols])?;
    dict.set_item("values", reshaped)?;

    dict.set_item(
        "periods",
        metadata
            .combos
            .iter()
            .map(|c| c.period.unwrap_or(20) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    Ok(dict.into())
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "reflex_cuda_batch_dev")]
#[pyo3(signature = (data_f32, period_range, device_id=0))]
pub fn reflex_cuda_batch_dev_py(
    py: Python<'_>,
    data_f32: numpy::PyReadonlyArray1<'_, f32>,
    period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let slice_in = data_f32.as_slice()?;
    let sweep = ReflexBatchRange {
        period: period_range,
    };

    let (inner, ctx, dev_id) = py.allow_threads(|| {
        let cuda = CudaReflex::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.context_arc();
        let dev_id = device_id as u32;
        cuda.reflex_batch_dev(slice_in, &sweep)
            .map(|inner| (inner, ctx, dev_id))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    Ok(DeviceArrayF32Py {
        inner,
        _ctx: Some(ctx),
        device_id: Some(dev_id),
    })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "reflex_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, period, device_id=0))]
pub fn reflex_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    data_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    period: usize,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    use numpy::PyUntypedArrayMethods;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let shape = data_tm_f32.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err("expected 2D array"));
    }
    let rows = shape[0];
    let cols = shape[1];
    let flat = data_tm_f32.as_slice()?;

    let (inner, ctx, dev_id) = py.allow_threads(|| {
        let cuda = CudaReflex::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.context_arc();
        let dev_id = device_id as u32;
        cuda.reflex_many_series_one_param_time_major_dev(flat, cols, rows, period)
            .map(|inner| (inner, ctx, dev_id))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    Ok(DeviceArrayF32Py {
        inner,
        _ctx: Some(ctx),
        device_id: Some(dev_id),
    })
}

#[cfg(feature = "python")]
#[pyclass(name = "ReflexStream")]
pub struct ReflexStreamPy {
    inner: ReflexStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl ReflexStreamPy {
    #[new]
    #[pyo3(signature = (period = 20))]
    pub fn new(period: usize) -> PyResult<Self> {
        let params = ReflexParams {
            period: Some(period),
        };
        let inner = ReflexStream::try_new(params)
            .map_err(|e| PyValueError::new_err(format!("reflex stream error: {}", e)))?;
        Ok(Self { inner })
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.inner.update(value)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn reflex_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = ReflexParams {
        period: Some(period),
    };
    let input = ReflexInput::from_slice(data, params);

    let mut output = vec![0.0; data.len()];

    reflex_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn reflex_batch_js(
    data: &[f64],
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let range = ReflexBatchRange {
        period: (period_start, period_end, period_step),
    };

    let output = reflex_batch_with_kernel(data, &range, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&format!("reflex batch error: {}", e)))?;

    Ok(output.values)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn reflex_batch_metadata_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Vec<usize> {
    let range = ReflexBatchRange {
        period: (period_start, period_end, period_step),
    };
    match expand_grid_checked(&range) {
        Ok(combos) => combos.iter().map(|c| c.period.unwrap_or(20)).collect(),
        Err(_) => Vec::new(),
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn reflex_batch_rows_cols_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
    data_len: usize,
) -> Vec<usize> {
    let range = ReflexBatchRange {
        period: (period_start, period_end, period_step),
    };
    let rows = expand_grid_checked(&range).map(|c| c.len()).unwrap_or(0);
    vec![rows, data_len]
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn reflex_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn reflex_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn reflex_into(
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

        if period == 0 || period > len {
            return Err(JsValue::from_str("Invalid period"));
        }

        let params = ReflexParams {
            period: Some(period),
        };
        let input = ReflexInput::from_slice(data, params);

        if in_ptr == out_ptr {
            let mut temp = vec![0.0; len];
            reflex_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            reflex_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn reflex_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer"));
    }
    let data = unsafe { std::slice::from_raw_parts(in_ptr, len) };
    let sweep = ReflexBatchRange {
        period: (period_start, period_end, period_step),
    };

    let combos = expand_grid_checked(&sweep)
        .map_err(|e| JsValue::from_str(&format!("reflex batch error: {}", e)))?;
    let rows = combos.len();
    let cols = len;
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| JsValue::from_str("size overflow"))?;
    let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, total) };

    reflex_batch_inner_into(data, &sweep, detect_best_kernel(), false, out)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_reflex_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = ReflexParams { period: None };
        let input = ReflexInput::from_candles(&candles, "close", default_params);
        let output = reflex_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        let params_period_14 = ReflexParams { period: Some(14) };
        let input2 = ReflexInput::from_candles(&candles, "hl2", params_period_14);
        let output2 = reflex_with_kernel(&input2, kernel)?;
        assert_eq!(output2.values.len(), candles.close.len());
        let params_custom = ReflexParams { period: Some(30) };
        let input3 = ReflexInput::from_candles(&candles, "hlc3", params_custom);
        let output3 = reflex_with_kernel(&input3, kernel)?;
        assert_eq!(output3.values.len(), candles.close.len());
        Ok(())
    }

    fn check_reflex_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = ReflexParams::default();
        let input = ReflexInput::from_candles(&candles, "close", default_params);
        let result = reflex_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        let len = result.values.len();
        let expected_last_five = [
            0.8085220962465361,
            0.445264715886137,
            0.13861699036615063,
            -0.03598639652007061,
            -0.224906760543743,
        ];
        let start_idx = len - 5;
        let last_five = &result.values[start_idx..];
        for (i, &val) in last_five.iter().enumerate() {
            let exp = expected_last_five[i];
            assert!(
                (val - exp).abs() < 1e-7,
                "[{}] Reflex mismatch at idx {}: got {}, expected {}",
                test_name,
                i,
                val,
                exp
            );
        }
        Ok(())
    }

    fn check_reflex_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = ReflexInput::with_default_candles(&candles);
        match input.data {
            ReflexData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected ReflexData::Candles"),
        }
        let output = reflex_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_reflex_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = ReflexParams { period: Some(0) };
        let input = ReflexInput::from_slice(&input_data, params);
        let res = reflex_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Reflex should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_reflex_period_less_than_two(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = ReflexParams { period: Some(1) };
        let input = ReflexInput::from_slice(&input_data, params);
        let res = reflex_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Reflex should fail with period<2",
            test_name
        );
        Ok(())
    }

    fn check_reflex_very_small_data_set(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [42.0];
        let params = ReflexParams { period: Some(2) };
        let input = ReflexInput::from_slice(&input_data, params);
        let res = reflex_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Reflex should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_reflex_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = ReflexParams { period: Some(14) };
        let first_input = ReflexInput::from_candles(&candles, "close", first_params);
        let first_result = reflex_with_kernel(&first_input, kernel)?;
        assert_eq!(first_result.values.len(), candles.close.len());
        let second_params = ReflexParams { period: Some(10) };
        let second_input = ReflexInput::from_slice(&first_result.values, second_params);
        let second_result = reflex_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        for i in 14..second_result.values.len() {
            assert!(second_result.values[i].is_finite());
        }
        Ok(())
    }

    fn check_reflex_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 14;
        let params = ReflexParams {
            period: Some(period),
        };
        let input = ReflexInput::from_candles(&candles, "close", params);
        let result = reflex_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        if result.values.len() > period {
            for i in period..result.values.len() {
                assert!(
                    result.values[i].is_finite(),
                    "[{}] Unexpected NaN at index {}",
                    test_name,
                    i
                );
            }
        }
        Ok(())
    }

    fn check_reflex_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 14;
        let params = ReflexParams {
            period: Some(period),
        };
        let input = ReflexInput::from_candles(&candles, "close", params.clone());
        let batch_output = reflex_with_kernel(&input, kernel)?.values;
        let mut stream = ReflexStream::try_new(params)?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(v) => stream_values.push(v),
                None => stream_values.push(0.0),
            }
        }
        assert_eq!(batch_output.len(), stream_values.len());
        for (i, (&b, &s)) in batch_output.iter().zip(stream_values.iter()).enumerate() {
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] Reflex streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_reflex_tests {
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
    fn check_reflex_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let test_cases = vec![
            ReflexParams { period: Some(20) },
            ReflexParams { period: Some(2) },
            ReflexParams { period: Some(5) },
            ReflexParams { period: Some(10) },
            ReflexParams { period: Some(30) },
            ReflexParams { period: Some(50) },
            ReflexParams { period: Some(15) },
            ReflexParams { period: Some(40) },
            ReflexParams { period: None },
        ];

        for params in test_cases {
            let input = ReflexInput::from_candles(&candles, "close", params);
            let output = reflex_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
                         with params period={:?}",
                        test_name, val, bits, i, params.period
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
                         with params period={:?}",
                        test_name, val, bits, i, params.period
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
                         with params period={:?}",
                        test_name, val, bits, i, params.period
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_reflex_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_reflex_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (2usize..=50).prop_flat_map(|period| {
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
                let params = ReflexParams {
                    period: Some(period),
                };
                let input = ReflexInput::from_slice(&data, params);

                let ReflexOutput { values: out } = reflex_with_kernel(&input, kernel).unwrap();
                let ReflexOutput { values: ref_out } =
                    reflex_with_kernel(&input, Kernel::Scalar).unwrap();

                prop_assert_eq!(out.len(), data.len());

                for i in 0..period.min(data.len()) {
                    prop_assert!(
                        out[i] == 0.0,
                        "[{}] idx {}: expected 0.0 during warmup, got {}",
                        test_name,
                        i,
                        out[i]
                    );
                }

                for i in 0..data.len() {
                    let y = out[i];
                    let r = ref_out[i];

                    if !y.is_finite() || !r.is_finite() {
                        prop_assert_eq!(
                            y.to_bits(),
                            r.to_bits(),
                            "[{}] finite/NaN mismatch idx {}: {} vs {}",
                            test_name,
                            i,
                            y,
                            r
                        );
                        continue;
                    }

                    let ulp_diff = y.to_bits().abs_diff(r.to_bits());
                    prop_assert!(
                        (y - r).abs() <= 1e-9 || ulp_diff <= 4,
                        "[{}] mismatch idx {}: {} vs {} (ULP={})",
                        test_name,
                        i,
                        y,
                        r,
                        ulp_diff
                    );
                }

                for i in period..data.len() {
                    if data[i].abs() < 1e10 {
                        prop_assert!(
                            out[i].is_finite(),
                            "[{}] idx {}: expected finite, got {}",
                            test_name,
                            i,
                            out[i]
                        );
                    }
                }

                if data.windows(2).all(|w| (w[0] - w[1]).abs() < f64::EPSILON) {
                    for i in (period * 2)..data.len() {
                        prop_assert!(
                            out[i].abs() < 0.001,
                            "[{}] idx {}: constant data should yield near-zero, got {}",
                            test_name,
                            i,
                            out[i]
                        );
                    }
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    #[cfg(not(feature = "proptest"))]
    fn check_reflex_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        Ok(())
    }

    generate_all_reflex_tests!(
        check_reflex_partial_params,
        check_reflex_accuracy,
        check_reflex_default_candles,
        check_reflex_zero_period,
        check_reflex_period_less_than_two,
        check_reflex_very_small_data_set,
        check_reflex_reinput,
        check_reflex_nan_handling,
        check_reflex_streaming,
        check_reflex_no_poison,
        check_reflex_property
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = ReflexBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = ReflexParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());

        let expected = [
            0.8085220962465361,
            0.445264715886137,
            0.13861699036615063,
            -0.03598639652007061,
            -0.224906760543743,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-7,
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
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let batch_configs = vec![
            (10, 30, 10),
            (20, 20, 0),
            (2, 10, 2),
            (25, 50, 25),
            (5, 20, 5),
            (15, 45, 15),
            (3, 15, 3),
            (30, 60, 10),
        ];

        for (p_start, p_end, p_step) in batch_configs {
            let output = ReflexBatchBuilder::new()
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
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} \
                         (flat index {}) with params period={:?}",
						test, val, bits, row, col, idx, combo.period
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} \
                         (flat index {}) with params period={:?}",
						test, val, bits, row, col, idx, combo.period
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} \
                         (flat index {}) with params period={:?}",
						test, val, bits, row, col, idx, combo.period
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

    #[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
    #[test]
    fn test_reflex_into_matches_api() -> Result<(), Box<dyn Error>> {
        let mut data = vec![f64::from_bits(0x7ff8_0000_0000_0000); 3];
        data.extend((0..256).map(|i| ((i as f64) * 0.1).sin() * 1.23 + (i as f64) * 0.01));

        let input = ReflexInput::from_slice(&data, ReflexParams::default());

        let baseline = reflex_with_kernel(&input, Kernel::Auto)?.values;

        let mut out = vec![0.0; data.len()];
        super::reflex_into(&input, &mut out)?;

        assert_eq!(baseline.len(), out.len());

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b) || ((a - b).abs() <= 1e-12)
        }

        for i in 0..out.len() {
            assert!(
                eq_or_both_nan(baseline[i], out[i]),
                "mismatch at {}: baseline={} out={}",
                i,
                baseline[i],
                out[i]
            );
        }

        Ok(())
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);
}
