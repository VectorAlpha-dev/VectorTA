//! # Symmetric Weighted Moving Average (SWMA)
//!
//! A triangular weighted moving average centered on the window. More weight is given to center elements, balancing smoothness and responsiveness.
//!
//! ## Parameters
//! - **period**: Window size (number of data points). Default is 5.
//!
//! ## Errors
//! - **AllValuesNaN**: swma: All input data values are `NaN`.
//! - **InvalidPeriod**: swma: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: swma: Not enough valid data points for the requested `period`.
//!
//! ## Returns
//! - **`Ok(SwmaOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(SwmaError)`** otherwise.
//!

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel, alloc_with_nan_prefix, init_matrix_prefixes, make_uninit_matrix};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use thiserror::Error;
use std::mem::MaybeUninit;

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

impl<'a> AsRef<[f64]> for SwmaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            SwmaData::Slice(slice) => slice,
            SwmaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum SwmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct SwmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct SwmaParams {
    pub period: Option<usize>,
}

impl Default for SwmaParams {
    fn default() -> Self {
        Self { period: Some(5) }
    }
}

#[derive(Debug, Clone)]
pub struct SwmaInput<'a> {
    pub data: SwmaData<'a>,
    pub params: SwmaParams,
}

impl<'a> SwmaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: SwmaParams) -> Self {
        Self {
            data: SwmaData::Candles { candles: c, source: s },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: SwmaParams) -> Self {
        Self {
            data: SwmaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", SwmaParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(5)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SwmaBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for SwmaBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl SwmaBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<SwmaOutput, SwmaError> {
        let p = SwmaParams { period: self.period };
        let i = SwmaInput::from_candles(c, "close", p);
        swma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<SwmaOutput, SwmaError> {
        let p = SwmaParams { period: self.period };
        let i = SwmaInput::from_slice(d, p);
        swma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<SwmaStream, SwmaError> {
        let p = SwmaParams { period: self.period };
        SwmaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum SwmaError {
    #[error("swma: All values are NaN.")]
    AllValuesNaN,

    #[error("swma: Invalid period: period = {period}, data length = {data_len}. Period must be between 1 and data length.")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("swma: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn swma(input: &SwmaInput) -> Result<SwmaOutput, SwmaError> {
    swma_with_kernel(input, Kernel::Auto)
}

/// Prepare SWMA computation: validate inputs, build weights, determine kernel
#[inline]
fn swma_prepare<'a>(
    input: &'a SwmaInput,
    kernel: Kernel,
) -> Result<(&'a [f64], AVec<f64>, usize, usize, Kernel), SwmaError> {
    let data: &[f64] = match &input.data {
        SwmaData::Candles { candles, source } => source_type(candles, source),
        SwmaData::Slice(sl) => sl,
    };

    let first = data.iter().position(|x| !x.is_nan()).ok_or(SwmaError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(SwmaError::InvalidPeriod { period, data_len: len });
    }
    if (len - first) < period {
        return Err(SwmaError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let weights: AVec<f64> = {
        let v = build_symmetric_triangle(period);
        let mut out = AVec::with_capacity(CACHELINE_ALIGN, period);
        out.extend_from_slice(&v);
        out
    };

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    Ok((data, weights, period, first, chosen))
}

pub fn swma_with_kernel(input: &SwmaInput, kernel: Kernel) -> Result<SwmaOutput, SwmaError> {
    let (data, weights, period, first, chosen) = swma_prepare(input, kernel)?;
    
    let len = data.len();
    let warm = first + period;
    let mut out = alloc_with_nan_prefix(len, warm);

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                swma_scalar(data, &weights, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                swma_avx2(data, &weights, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                swma_avx512(data, &weights, period, first, &mut out)
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                // Fallback to scalar when AVX is not available
                swma_scalar(data, &weights, period, first, &mut out)
            }
            _ => unreachable!(),
        }
    }

    Ok(SwmaOutput { values: out })
}

#[inline(always)]
fn build_symmetric_triangle(n: usize) -> Vec<f64> {
    let triangle: Vec<f64> = if n == 1 {
        vec![1.0]
    } else if n == 2 {
        vec![1.0, 1.0]
    } else if n % 2 == 0 {
        let half = n / 2;
        let mut front: Vec<f64> = (1..=half).map(|x| x as f64).collect();
        let mut back = front.clone();
        back.reverse();
        front.extend(back);
        front
    
        } else {
        let half_plus = ((n + 1) as f64 / 2.0).floor() as usize;
        let mut front: Vec<f64> = (1..=half_plus).map(|x| x as f64).collect();
        let mut tri = front.clone();
        front.pop();
        front.reverse();
        tri.extend(front);
        tri
    };
    let sum: f64 = triangle.iter().sum();
    triangle.into_iter().map(|x| x / sum).collect()
}

#[inline]
pub fn swma_scalar(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first_val: usize,
    out: &mut [f64],
) {
    assert_eq!(weights.len(), period);
    assert!(out.len() >= data.len());
    let p4 = period & !3;
    for i in (first_val + period - 1)..data.len() {
        let start = i + 1 - period;
        let window = &data[start..start + period];
        let mut sum = 0.0;
        for (d4, w4) in window[..p4].chunks_exact(4).zip(weights[..p4].chunks_exact(4)) {
            sum += d4[0] * w4[0] + d4[1] * w4[1] + d4[2] * w4[2] + d4[3] * w4[3];
        }
        for (d, w) in window[p4..].iter().zip(&weights[p4..]) {
            sum += d * w;
        }
        out[i] = sum;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn swma_avx512(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        unsafe { swma_avx512_short(data, weights, period, first_valid, out) }
    } else {
        unsafe { swma_avx512_long(data, weights, period, first_valid, out) }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn swma_avx2(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    swma_scalar(data, weights, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
unsafe fn swma_avx512_short(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    swma_scalar(data, weights, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512dq,fma")]
unsafe fn swma_avx512_long(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    swma_scalar(data, weights, period, first_valid, out)
}

#[derive(Debug, Clone)]
pub struct SwmaStream {
    period: usize,
    weights: Vec<f64>,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,
}

impl SwmaStream {
    pub fn try_new(params: SwmaParams) -> Result<Self, SwmaError> {
        let period = params.period.unwrap_or(5);
        if period == 0 {
            return Err(SwmaError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let weights = build_symmetric_triangle(period);
        Ok(Self {
            period,
            weights,
            buffer: vec![f64::NAN; period],
            head: 0,
            filled: false,
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
        let mut sum = 0.0;
        let mut idx = self.head;
        for &w in &self.weights {
            sum += w * self.buffer[idx];
            idx = (idx + 1) % self.period;
        }
        sum
    }
}

#[derive(Clone, Debug)]
pub struct SwmaBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for SwmaBatchRange {
    fn default() -> Self {
        Self {
            period: (5, 240, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SwmaBatchBuilder {
    range: SwmaBatchRange,
    kernel: Kernel,
}

impl SwmaBatchBuilder {
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

    pub fn apply_slice(self, data: &[f64]) -> Result<SwmaBatchOutput, SwmaError> {
        swma_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<SwmaBatchOutput, SwmaError> {
        SwmaBatchBuilder::new().kernel(k).apply_slice(data)
    }

    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<SwmaBatchOutput, SwmaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }

    pub fn with_default_candles(c: &Candles) -> Result<SwmaBatchOutput, SwmaError> {
        SwmaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn swma_batch_with_kernel(
    data: &[f64],
    sweep: &SwmaBatchRange,
    k: Kernel,
) -> Result<SwmaBatchOutput, SwmaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(SwmaError::InvalidPeriod {
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
    swma_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct SwmaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<SwmaParams>,
    pub rows: usize,
    pub cols: usize,
}

impl SwmaBatchOutput {
    pub fn row_for_params(&self, p: &SwmaParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(5) == p.period.unwrap_or(5)
        })
    }

    pub fn values_for(&self, p: &SwmaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &SwmaBatchRange) -> Vec<SwmaParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(SwmaParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn swma_batch_slice(
    data: &[f64],
    sweep: &SwmaBatchRange,
    kern: Kernel,
) -> Result<SwmaBatchOutput, SwmaError> {
    swma_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn swma_batch_par_slice(
    data: &[f64],
    sweep: &SwmaBatchRange,
    kern: Kernel,
) -> Result<SwmaBatchOutput, SwmaError> {
    swma_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn swma_batch_inner(
    data: &[f64],
    sweep: &SwmaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<SwmaBatchOutput, SwmaError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(SwmaError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    // Reuse validation logic
    let first = data.iter().position(|x| !x.is_nan()).ok_or(SwmaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    let len = data.len();
    
    if max_p == 0 || max_p > len {
        return Err(SwmaError::InvalidPeriod { period: max_p, data_len: len });
    }
    if (len - first) < max_p {
        return Err(SwmaError::NotEnoughValidData {
            needed: max_p,
            valid: len - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let cap = rows * max_p;
    let mut flat_w = AVec::<f64>::with_capacity(CACHELINE_ALIGN, cap);
    flat_w.resize(cap, 0.0);
    
    // Pre-compute weights for each period
    for (row, combo) in combos.iter().enumerate() {
        let period = combo.period.unwrap();
        let weights = build_symmetric_triangle(period);
        let w_start = row * max_p;
        flat_w[w_start..w_start + period].copy_from_slice(&weights);
    }

    let warm: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap()).collect();
    let mut raw = make_uninit_matrix(rows, cols);
    unsafe { init_matrix_prefixes(&mut raw, cols, &warm) };

    // ------------------------------------------------------------------
    // 2)  Closure that fills one row; it receives &mut [MaybeUninit<f64>]
    // ------------------------------------------------------------------
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();
        let w_ptr  = flat_w.as_ptr().add(row * max_p);

        // Cast just this row to &mut [f64] (now safe to write real values)
        let out_row = core::slice::from_raw_parts_mut(
            dst_mu.as_mut_ptr() as *mut f64,
            dst_mu.len(),
        );

        match kern {
            Kernel::Scalar => swma_row_scalar (data, first, period, w_ptr, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2   => swma_row_avx2   (data, first, period, w_ptr, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => swma_row_avx512 (data, first, period, w_ptr, out_row),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                // Fallback to scalar when AVX is not available
                swma_row_scalar (data, first, period, w_ptr, out_row)
            }
            _ => unreachable!(),
        }
    };

    // ------------------------------------------------------------------
    // 3)  Run every row, writing directly into `raw`
    // ------------------------------------------------------------------
    if parallel {

        #[cfg(not(target_arch = "wasm32"))] {

        raw.par_chunks_mut(cols)

                .enumerate()

                .for_each(|(row, slice)| do_row(row, slice));

        }

        #[cfg(target_arch = "wasm32")] {

        for (row, slice) in raw.chunks_mut(cols).enumerate() {

                    do_row(row, slice);

        }
        }
    } else {
        for (row, slice) in raw.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    // ------------------------------------------------------------------
    // 4)  Transmute the now-initialised buffer into Vec<f64>
    // ------------------------------------------------------------------
    let values: Vec<f64> = unsafe { std::mem::transmute(raw) };

    Ok(SwmaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn swma_batch_inner_into(
    data: &[f64],
    sweep: &SwmaBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<SwmaParams>, SwmaError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(SwmaError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    // Reuse validation logic
    let first = data.iter().position(|x| !x.is_nan()).ok_or(SwmaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    let len = data.len();
    
    if max_p == 0 || max_p > len {
        return Err(SwmaError::InvalidPeriod { period: max_p, data_len: len });
    }
    if (len - first) < max_p {
        return Err(SwmaError::NotEnoughValidData {
            needed: max_p,
            valid: len - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let cap = rows * max_p;
    let mut flat_w = AVec::<f64>::with_capacity(CACHELINE_ALIGN, cap);
    flat_w.resize(cap, 0.0);
    
    // Pre-compute weights for each period
    for (row, combo) in combos.iter().enumerate() {
        let period = combo.period.unwrap();
        let weights = build_symmetric_triangle(period);
        let w_start = row * max_p;
        flat_w[w_start..w_start + period].copy_from_slice(&weights);
    }

    let warm: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap()).collect();
    
    // SAFETY: We're reinterpreting the output slice as MaybeUninit to use the efficient
    // init_matrix_prefixes function. This is safe because:
    // 1. MaybeUninit<T> has the same layout as T
    // 2. We ensure all values are written before the slice is used again
    let out_uninit = unsafe {
        std::slice::from_raw_parts_mut(
            out.as_mut_ptr() as *mut MaybeUninit<f64>,
            out.len()
        )
    };
    
    unsafe { init_matrix_prefixes(out_uninit, cols, &warm) };

    // Closure that fills one row
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();
        let w_ptr  = flat_w.as_ptr().add(row * max_p);

        // Cast just this row to &mut [f64]
        let out_row = core::slice::from_raw_parts_mut(
            dst_mu.as_mut_ptr() as *mut f64,
            dst_mu.len(),
        );

        match kern {
            Kernel::Scalar | Kernel::ScalarBatch => swma_row_scalar(data, first, period, w_ptr, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => swma_row_avx2(data, first, period, w_ptr, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => swma_row_avx512(data, first, period, w_ptr, out_row),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                swma_row_scalar(data, first, period, w_ptr, out_row)
            }
            _ => unreachable!(),
        }
    };

    // Run every row, writing directly into output buffer
    if parallel {
        #[cfg(not(target_arch = "wasm32"))] {
            out_uninit.par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }
        #[cfg(target_arch = "wasm32")] {
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
unsafe fn swma_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    w_ptr: *const f64,
    out: &mut [f64],
) {
    let p4 = period & !3;
    for i in (first + period - 1)..data.len() {
        let start = i + 1 - period;
        let mut sum = 0.0;
        for k in (0..p4).step_by(4) {
            let w = std::slice::from_raw_parts(w_ptr.add(k), 4);
            let d = &data[start + k..start + k + 4];
            sum += d[0] * w[0] + d[1] * w[1] + d[2] * w[2] + d[3] * w[3];
        }
        for k in p4..period {
            sum += *data.get_unchecked(start + k) * *w_ptr.add(k);
        }
        out[i] = sum;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn swma_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    w_ptr: *const f64,
    out: &mut [f64],
) {
    swma_row_scalar(data, first, period, w_ptr, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512dq,fma")]
pub unsafe fn swma_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    w_ptr: *const f64,
    out: &mut [f64],
) {
    if period <= 32 {
        swma_row_avx512_short(data, first, period, w_ptr, out);
    
        } else {
        swma_row_avx512_long(data, first, period, w_ptr, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
unsafe fn swma_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    w_ptr: *const f64,
    out: &mut [f64],
) {
    swma_row_scalar(data, first, period, w_ptr, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512dq,fma")]
unsafe fn swma_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    w_ptr: *const f64,
    out: &mut [f64],
) {
    swma_row_scalar(data, first, period, w_ptr, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_swma_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = SwmaParams { period: None };
        let input = SwmaInput::from_candles(&candles, "close", default_params);
        let output = swma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_swma_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = SwmaInput::from_candles(&candles, "close", SwmaParams::default());
        let result = swma_with_kernel(&input, kernel)?;
        let expected_last_five = [
            59288.22222222222,
            59301.99999999999,
            59247.33333333333,
            59179.88888888889,
            59080.99999999999,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(diff < 1e-8, "[{}] SWMA {:?} mismatch at idx {}: got {}, expected {}", test_name, kernel, i, val, expected_last_five[i]);
        }
        Ok(())
    }

    fn check_swma_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = SwmaInput::with_default_candles(&candles);
        match input.data {
            SwmaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected SwmaData::Candles"),
        }
        let output = swma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_swma_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = SwmaParams { period: Some(0) };
        let input = SwmaInput::from_slice(&input_data, params);
        let res = swma_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] SWMA should fail with zero period", test_name);
        Ok(())
    }

    fn check_swma_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = SwmaParams { period: Some(10) };
        let input = SwmaInput::from_slice(&data_small, params);
        let res = swma_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] SWMA should fail with period exceeding length", test_name);
        Ok(())
    }

    fn check_swma_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = SwmaParams { period: Some(5) };
        let input = SwmaInput::from_slice(&single_point, params);
        let res = swma_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] SWMA should fail with insufficient data", test_name);
        Ok(())
    }

    fn check_swma_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = SwmaParams { period: Some(5) };
        let first_input = SwmaInput::from_candles(&candles, "close", first_params);
        let first_result = swma_with_kernel(&first_input, kernel)?;
        let second_params = SwmaParams { period: Some(3) };
        let second_input = SwmaInput::from_slice(&first_result.values, second_params);
        let second_result = swma_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    fn check_swma_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = SwmaParams { period: Some(5) };
        let input = SwmaInput::from_candles(&candles, "close", params);
        let res = swma_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 240 {
            for (i, &val) in res.values[240..].iter().enumerate() {
                assert!(!val.is_nan(), "[{}] Found unexpected NaN at out-index {}", test_name, 240 + i);
            }
        }
        Ok(())
    }

    fn check_swma_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 5;
        let input = SwmaInput::from_candles(&candles, "close", SwmaParams { period: Some(period) });
        let batch_output = swma_with_kernel(&input, kernel)?.values;
        let mut stream = SwmaStream::try_new(SwmaParams { period: Some(period) })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(swma_val) => stream_values.push(swma_val),
                None => stream_values.push(f64::NAN),
            }
        }
        assert_eq!(batch_output.len(), stream_values.len());
        for (i, (&b, &s)) in batch_output.iter().zip(stream_values.iter()).enumerate() {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(diff < 1e-9, "[{}] SWMA streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}", test_name, i, b, s, diff);
        }
        Ok(())
    }

    macro_rules! generate_all_swma_tests {
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
    fn check_swma_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test with default parameters
        let input = SwmaInput::from_candles(&candles, "close", SwmaParams::default());
        let output = swma_with_kernel(&input, kernel)?;

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
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {}",
                    test_name, val, bits, i
                );
            }

            // Check for init_matrix_prefixes poison (0x22222222_22222222)
            if bits == 0x22222222_22222222 {
                panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {}",
                    test_name, val, bits, i
                );
            }

            // Check for make_uninit_matrix poison (0x33333333_33333333)
            if bits == 0x33333333_33333333 {
                panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {}",
                    test_name, val, bits, i
                );
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_swma_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    generate_all_swma_tests!(
        check_swma_partial_params,
        check_swma_accuracy,
        check_swma_default_candles,
        check_swma_zero_period,
        check_swma_period_exceeds_length,
        check_swma_very_small_dataset,
        check_swma_reinput,
        check_swma_nan_handling,
        check_swma_streaming,
        check_swma_no_poison
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = SwmaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = SwmaParams::default();
        let period = def.period.unwrap_or(5);
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            59288.22222222222,
            59301.99999999999,
            59247.33333333333,
            59179.88888888889,
            59080.99999999999,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate().skip(period) {
            assert!((v - expected[i]).abs() < 1e-8, "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}");
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

    // Check for poison values in batch output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test batch with multiple parameter combinations
        let output = SwmaBatchBuilder::new()
            .kernel(kernel)
            .period_range(3, 9, 3)  // Adjust ranges based on indicator
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

            // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
            if bits == 0x11111111_11111111 {
                panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {})",
                    test, val, bits, row, col, idx
                );
            }

            // Check for init_matrix_prefixes poison (0x22222222_22222222)
            if bits == 0x22222222_22222222 {
                panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {})",
                    test, val, bits, row, col, idx
                );
            }

            // Check for make_uninit_matrix poison (0x33333333_33333333)
            if bits == 0x33333333_33333333 {
                panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {})",
                    test, val, bits, row, col, idx
                );
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);
}

#[cfg(feature = "python")]
#[pyfunction(name = "swma")]
#[pyo3(signature = (data, period, kernel=None))]
/// Compute the Symmetric Weighted Moving Average (SWMA) of the input data.
///
/// SWMA uses triangular weights centered on the window to balance smoothness
/// and responsiveness in the moving average calculation.
///
/// Parameters:
/// -----------
/// data : np.ndarray
///     Input data array (float64).
/// period : int
///     Number of data points in the moving average window.
/// kernel : str, optional
///     Computation kernel to use: 'auto', 'scalar', 'avx2', 'avx512'.
///     Default is 'auto' which auto-detects the best available.
///
/// Returns:
/// --------
/// np.ndarray
///     Array of SWMA values, same length as input.
///
/// Raises:
/// -------
/// ValueError
///     If inputs are invalid (period out of range, empty data, etc).
pub fn swma_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{PyArray1, PyArrayMethods};

    let slice_in = data.as_slice()?; // zero-copy, read-only view

    // Parse kernel string to enum
    let kern = match kernel {
        None | Some("auto") => Kernel::Auto,
        Some("scalar") => Kernel::Scalar,
        Some("avx2") => Kernel::Avx2,
        Some("avx512") => Kernel::Avx512,
        Some(k) => return Err(PyValueError::new_err(format!("Unknown kernel: {}", k))),
    };

    // Build input struct
    let params = SwmaParams { period: Some(period) };
    let swma_in = SwmaInput::from_slice(slice_in, params);

    // Allocate NumPy output buffer
    let out_arr = unsafe { PyArray1::<f64>::new(py, [slice_in.len()], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? }; // safe: we own the array

    // Heavy lifting without the GIL
    let result = py.allow_threads(|| -> Result<(), SwmaError> {
        let output = swma_with_kernel(&swma_in, kern)?;
        slice_out.copy_from_slice(&output.values);
        Ok(())
    });

    match result {
        Ok(()) => Ok(out_arr),
        Err(e) => Err(PyValueError::new_err(e.to_string())),
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "SwmaStream")]
pub struct SwmaStreamPy {
    stream: SwmaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl SwmaStreamPy {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        let params = SwmaParams { period: Some(period) };
        let stream =
            SwmaStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(SwmaStreamPy { stream })
    }

    /// Updates the stream with a new value and returns the calculated SWMA value.
    /// Returns `None` if the buffer is not yet full.
    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "swma_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
/// Compute SWMA for multiple period values in a single pass.
///
/// Parameters:
/// -----------
/// data : np.ndarray
///     Input data array (float64).
/// period_range : tuple
///     (start, end, step) for period values to compute.
/// kernel : str, optional
///     Computation kernel: 'auto', 'scalar', 'avx2', 'avx512'.
///     Default is 'auto' which auto-detects the best available.
///
/// Returns:
/// --------
/// dict
///     Dictionary with 'values' (2D array) and 'periods' array.
pub fn swma_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;

    let sweep = SwmaBatchRange {
        period: period_range,
    };

    // 1. Expand grid once to know rows*cols
    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();

    // 2. Pre-allocate NumPy array (1-D, will reshape later)
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    // Parse kernel string to enum
    let kern = match kernel {
        None | Some("auto") => Kernel::Auto,
        Some("scalar") => Kernel::ScalarBatch,
        Some("avx2") => Kernel::Avx2Batch,
        Some("avx512") => Kernel::Avx512Batch,
        Some(k) => return Err(PyValueError::new_err(format!("Unknown kernel: {}", k))),
    };

    // 3. Heavy work without the GIL
    let combos = py.allow_threads(|| {
        // Resolve Kernel::Auto to a specific kernel
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
        // Use the _into variant that writes directly to our pre-allocated buffer
        swma_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // 4. Build dict with the GIL
    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "periods",
        combos
            .iter()
            .map(|c| c.period.unwrap_or(5))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    Ok(dict)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn swma_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = SwmaParams {
        period: Some(period),
    };
    let input = SwmaInput::from_slice(data, params);

    swma_with_kernel(&input, Kernel::Scalar)
        .map(|o| o.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn swma_batch_js(
    data: &[f64],
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = SwmaBatchRange {
        period: (period_start, period_end, period_step),
    };

    // Use the existing batch function with parallel=false for WASM
    swma_batch_inner(data, &sweep, Kernel::Scalar, false)
        .map(|output| output.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn swma_batch_metadata_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = SwmaBatchRange {
        period: (period_start, period_end, period_step),
    };

    let combos = expand_grid(&sweep);
    let mut metadata = Vec::with_capacity(combos.len());

    for combo in combos {
        metadata.push(combo.period.unwrap_or(5) as f64);
    }

    Ok(metadata)
}
