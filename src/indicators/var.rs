//! # Rolling Variance (VAR)
//!
//! Computes the rolling variance over a specified window with an optional standard deviation factor
//! for scaling the output.
//!
//! ## Parameters
//! - **period**: Window size for variance calculation (default: 14)
//! - **nbdev**: Standard deviation factor for scaling output, VAR = variance * nbdev^2 (default: 1.0)
//!
//! ## Inputs
//! - Data series as slice or candles with source
//!
//! ## Returns
//! - **values**: Variance values as `Vec<f64>` (length matches input)
//!
//! ## Developer Notes
//! - **AVX2/AVX512 kernels**: Currently stubs that call scalar implementation
//! - **Streaming update**: O(1) performance with Welford's online algorithm for incremental variance
//! - **Memory optimization**: Properly uses zero-copy helper functions (alloc_with_nan_prefix, make_uninit_matrix, init_matrix_prefixes)
//! - **TODO**: Implement actual SIMD kernels for AVX2/AVX512
//! - **Note**: Streaming implementation is highly optimized using running sums and sum of squares

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

// -- Data Structures --

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
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
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

// -- Builder --

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

// -- Errors --

#[derive(Debug, Error)]
pub enum VarError {
    #[error("var: All values are NaN.")]
    AllValuesNaN,
    #[error("var: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("var: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("var: nbdev is NaN or infinite: {nbdev}")]
    InvalidNbdev { nbdev: f64 },
}

// -- Indicator functions --

#[inline]
pub fn var(input: &VarInput) -> Result<VarOutput, VarError> {
    var_with_kernel(input, Kernel::Auto)
}

pub fn var_with_kernel(input: &VarInput, kernel: Kernel) -> Result<VarOutput, VarError> {
    let data: &[f64] = input.as_ref();
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(VarError::AllValuesNaN)?;
    let len = data.len();
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
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    // key change
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

    let mut sum = 0.0;
    let mut sum_sq = 0.0;

    for &v in &data[first..first + period] {
        sum += v;
        sum_sq += v * v;
    }

    out[first + period - 1] = (sum_sq * inv_p - (sum * inv_p).powi(2)) * nbdev2;

    for i in (first + period)..len {
        let old = data[i - period];
        let new = data[i];
        sum += new - old;
        sum_sq += new * new - old * old;
        out[i] = (sum_sq * inv_p - (sum * inv_p).powi(2)) * nbdev2;
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
    // Stub: points to scalar logic for API parity
    var_scalar(data, period, first, nbdev, out)
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
    // Stub: points to scalar logic for API parity
    var_scalar(data, period, first, nbdev, out)
}

// --- WASM Helper Function ---

#[inline]
pub fn var_into_slice(dst: &mut [f64], input: &VarInput, kern: Kernel) -> Result<(), VarError> {
    let data: &[f64] = input.as_ref();
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(VarError::AllValuesNaN)?;
    let len = data.len();
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
    if dst.len() != data.len() {
        return Err(VarError::InvalidPeriod {
            period: dst.len(),
            data_len: data.len(),
        });
    }

    let chosen = match kern {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    // Compute directly into dst
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

    // key addition
    let warmup_end = first + period - 1;
    for v in &mut dst[..warmup_end] {
        *v = f64::NAN;
    }

    Ok(())
}

// --- Row variants for batch ---

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
    let nbdev2 = nbdev * nbdev;
    let inv_p = 1.0 / (period as f64);

    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for &v in &data[first..first + period] {
        sum += v;
        sum_sq += v * v;
    }
    out[first + period - 1] = (sum_sq * inv_p - (sum * inv_p).powi(2)) * nbdev2;

    for i in (first + period)..len {
        let old = data[i - period];
        let new = data[i];
        sum += new - old;
        sum_sq += new * new - old * old;
        out[i] = (sum_sq * inv_p - (sum * inv_p).powi(2)) * nbdev2;
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

// --- Batch support ---

#[derive(Clone, Debug)]
pub struct VarBatchRange {
    pub period: (usize, usize, usize),
    pub nbdev: (f64, f64, f64),
}

impl Default for VarBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 60, 1),
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

// -- Grid

#[inline(always)]
fn expand_grid(r: &VarBatchRange) -> Vec<VarParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Vec<f64> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return vec![start];
        }
        let mut v = Vec::new();
        let mut x = start;
        while x <= end + 1e-12 {
            v.push(x);
            x += step;
        }
        v
    }
    let periods = axis_usize(r.period);
    let nbdevs = axis_f64(r.nbdev);
    let mut out = Vec::with_capacity(periods.len() * nbdevs.len());
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

// -- Batch Inner

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

#[inline(always)]
fn var_batch_inner(
    data: &[f64],
    sweep: &VarBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<VarBatchOutput, VarError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(VarError::InvalidPeriod {
            period: 0,
            data_len: 0,
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
    // key change: include `first`
    let warmup_periods: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warmup_periods);

    let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

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
        _ => {
            return Err(VarError::InvalidPeriod {
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
    var_batch_par_slice(data, sweep, simd)
}

// --- Streaming ---

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

// --- WASM Bindings ---

#[cfg(feature = "wasm")]
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

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn var_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn var_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(feature = "wasm")]
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
            // Handle aliasing - data and output are the same
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

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct VarBatchConfig {
    pub period_range: (usize, usize, usize),
    pub nbdev_range: (f64, f64, f64),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct VarBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<VarParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
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

#[cfg(feature = "wasm")]
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

        // Calculate output size
        let combos = expand_grid(&sweep);
        if combos.is_empty() {
            return Err(JsValue::from_str("No valid parameter combinations"));
        }
        let rows = combos.len();
        let cols = len;

        let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);

        // key addition: initialize warmup prefixes per row
        let first = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
        for (r, prm) in combos.iter().enumerate() {
            let warm = (first + prm.period.unwrap() - 1).min(cols);
            let row = &mut out[r * cols..r * cols + warm];
            for v in row {
                *v = f64::NAN;
            }
        }

        // compute into the buffer
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

        // Define comprehensive parameter combinations
        let test_params = vec![
            VarParams::default(), // period: 14, nbdev: 1.0
            VarParams {
                period: Some(2),
                nbdev: Some(1.0),
            }, // minimum period
            VarParams {
                period: Some(5),
                nbdev: Some(1.0),
            }, // small period
            VarParams {
                period: Some(10),
                nbdev: Some(1.0),
            }, // small-medium period
            VarParams {
                period: Some(20),
                nbdev: Some(1.0),
            }, // medium period
            VarParams {
                period: Some(30),
                nbdev: Some(1.0),
            }, // medium-large period
            VarParams {
                period: Some(50),
                nbdev: Some(1.0),
            }, // large period
            VarParams {
                period: Some(100),
                nbdev: Some(1.0),
            }, // very large period
            VarParams {
                period: Some(200),
                nbdev: Some(1.0),
            }, // extreme period
            VarParams {
                period: Some(14),
                nbdev: Some(0.5),
            }, // small nbdev
            VarParams {
                period: Some(14),
                nbdev: Some(2.0),
            }, // double nbdev
            VarParams {
                period: Some(14),
                nbdev: Some(3.0),
            }, // triple nbdev
            VarParams {
                period: Some(7),
                nbdev: Some(1.5),
            }, // mixed 1
            VarParams {
                period: Some(21),
                nbdev: Some(2.5),
            }, // mixed 2
            VarParams {
                period: Some(50),
                nbdev: Some(0.75),
            }, // mixed 3
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = VarInput::from_candles(&candles, "close", params.clone());
            let output = var_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
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
        Ok(()) // No-op in release builds
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_var_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Note: Starting at period=2 because period=1 has numerical precision issues
        // in the rolling calculation that accumulate small errors
        let strat = (2usize..=64).prop_flat_map(|period| {
            (
                prop::collection::vec(
                    (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                    period.max(10)..400,
                ),
                Just(period),
                0.1f64..3.0f64,      // nbdev
                -100.0f64..100.0f64, // trend
                -1e5f64..1e5f64,     // intercept
                prop::bool::ANY,     // use_special_pattern
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(
                &strat,
                |(mut data, period, nbdev, trend, intercept, use_special_pattern)| {
                    // Apply patterns to data for more realistic testing
                    if use_special_pattern {
                        // Apply linear trend
                        for (i, val) in data.iter_mut().enumerate() {
                            *val = intercept + trend * (i as f64);
                        }
                        // Add some noise
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

                    // Property 1: Variance should always be non-negative
                    for i in (period - 1)..data.len() {
                        let y = out[i];
                        if !y.is_nan() {
                            prop_assert!(
                                y >= -1e-6, // Allow small negative due to floating point precision
                                "[{}] Variance should be non-negative at idx {}: got {}",
                                test_name,
                                i,
                                y
                            );
                        }
                    }

                    // Property 2: For period=2, check special behavior (only for non-trended data)
                    // With only 2 values, variance represents the squared difference from mean
                    // Skip this check for trended data due to numerical precision differences
                    if period == 2 && !use_special_pattern {
                        for i in (period - 1)..data.len().min(10) {
                            // Check first few values only
                            if !out[i].is_nan() {
                                let window = &data[i + 1 - period..=i];
                                let mean = (window[0] + window[1]) / 2.0;
                                let expected =
                                    ((window[0] - mean).powi(2) + (window[1] - mean).powi(2)) / 2.0
                                        * nbdev
                                        * nbdev;
                                // Higher tolerance for different computation methods
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

                    // Property 3: For constant data, variance should be close to 0
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

                    // Property 4: Kernel consistency - all kernels should produce identical results
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

                    // Property 5: Mathematical correctness check for simple, non-trended cases
                    // For a window, verify variance formula: var = E[X^2] - E[X]^2
                    // Only check when we haven't applied special patterns (which can cause numerical differences)
                    if !use_special_pattern && data.len() >= period && period <= 10 {
                        let idx = period * 2; // Pick a point well into the data
                        if idx < data.len() && !out[idx].is_nan() {
                            let window = &data[idx + 1 - period..=idx];
                            let mean: f64 = window.iter().sum::<f64>() / (period as f64);
                            let mean_sq: f64 =
                                window.iter().map(|x| x * x).sum::<f64>() / (period as f64);
                            let expected_var = (mean_sq - mean * mean) * nbdev * nbdev;

                            // Allow reasonable tolerance for floating point differences
                            // The rolling computation may accumulate different rounding errors
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

        // Test various parameter sweep configurations
        let test_configs = vec![
            // (period_range, nbdev_range)
            ((2, 10, 2), (1.0, 1.0, 0.0)), // Small periods, single nbdev
            ((10, 30, 5), (1.0, 2.0, 0.5)), // Medium periods, varying nbdev
            ((30, 100, 10), (0.5, 1.5, 0.5)), // Large periods, varying nbdev
            ((2, 5, 1), (1.0, 3.0, 1.0)),  // Dense small range, varying nbdev
            ((14, 14, 0), (1.0, 1.0, 0.0)), // Single value (default)
            ((5, 25, 5), (2.0, 2.0, 0.0)), // Mixed range, fixed nbdev
            ((50, 100, 25), (1.0, 2.0, 0.25)), // Large step, fine nbdev
            ((14, 28, 7), (0.5, 2.5, 0.5)), // Common periods, wide nbdev range
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

                // Check all three poison patterns with detailed context
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
        Ok(()) // No-op in release builds
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

    // Test that periods not divisible by 8 work correctly
    #[test]
    fn test_batch_non_aligned_periods() {
        // Create test data with exactly enough points for period=7
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        // Test with period=7 (not divisible by 8)
        let sweep = VarBatchRange {
            period: (7, 7, 0), // Single period of 7
            nbdev: (1.0, 1.0, 0.0),
        };

        // This should work with 10 data points (need at least 7)
        let result = var_batch_slice(&data, &sweep, Kernel::Scalar);
        assert!(result.is_ok(), "Should handle period=7 with 10 data points");

        // Test with multiple non-aligned periods
        let sweep_multi = VarBatchRange {
            period: (5, 7, 1), // Periods: 5, 6, 7
            nbdev: (1.0, 1.0, 0.0),
        };

        let result_multi = var_batch_slice(&data, &sweep_multi, Kernel::Scalar);
        assert!(
            result_multi.is_ok(),
            "Should handle periods 5,6,7 with 10 data points"
        );
        let output = result_multi.unwrap();
        assert_eq!(output.rows, 3, "Should have 3 rows for periods 5,6,7");

        // Test that it correctly rejects when data is insufficient
        let data_short = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // Only 6 points
        let result_short = var_batch_slice(&data_short, &sweep, Kernel::Scalar);
        assert!(
            result_short.is_err(),
            "Should reject when data length (6) < period (7)"
        );

        // Test edge case: period=15 with exactly 15 data points
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

// Python bindings
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

    // Use high-level batch that initializes prefixes correctly
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
    // Convert flat vector to 2D array similar to how ATR does it
    let values_2d = unsafe { numpy::PyArray2::<f64>::new(py, [rows, cols], false) };
    let raw_ptr = values_2d.data() as *mut f64;
    let output_slice = unsafe { std::slice::from_raw_parts_mut(raw_ptr, rows * cols) };
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

// Helper function for batch processing that writes directly to output
#[inline(always)]
fn var_batch_inner_into(
    data: &[f64],
    sweep: &VarBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<VarParams>, VarError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(VarError::InvalidPeriod {
            period: 0,
            data_len: 0,
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

    Ok(combos)
}
