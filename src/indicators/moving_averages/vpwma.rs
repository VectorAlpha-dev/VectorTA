//! # Variable Power Weighted Moving Average (VPWMA)
//!
//! The Variable Power Weighted Moving Average (VPWMA) adjusts the weights of each
//! price data point in its calculation based on their respective volumes. This
//! means that periods with higher trading volumes have a greater influence on
//! the moving average. By raising the weight to a specified power (`power`),
//! one can control how aggressively recent, high-volume data points dominate
//! the resulting average.
//!
//! ## Parameters
//! - **period**: Number of data points in each calculation window (defaults to 14).
//! - **power**: Exponent applied to the volume-based weight function. Higher
//!   values give more impact to recent, higher-volume data (defaults to 0.382).
//!
//! ## Errors
//! - **AllValuesNaN**: vpwma: All input data values are `NaN`.
//! - **InvalidPeriod**: vpwma: `period` < 2 or exceeds the data length.
//! - **NotEnoughValidData**: vpwma: Not enough valid data points for the requested `period`.
//! - **InvalidPower**: vpwma: `power` is `NaN` or infinite.
//!
//! ## Returns
//! - **`Ok(VpwmaOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(VpwmaError)`** otherwise.
//!

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel, alloc_with_nan_prefix, make_uninit_matrix, init_matrix_prefixes};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use thiserror::Error;
use std::mem::MaybeUninit;

impl<'a> AsRef<[f64]> for VpwmaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            VpwmaData::Slice(slice) => slice,
            VpwmaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum VpwmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct VpwmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct VpwmaParams {
    pub period: Option<usize>,
    pub power: Option<f64>,
}

impl Default for VpwmaParams {
    fn default() -> Self {
        Self {
            period: Some(14),
            power: Some(0.382),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VpwmaInput<'a> {
    pub data: VpwmaData<'a>,
    pub params: VpwmaParams,
}

impl<'a> VpwmaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: VpwmaParams) -> Self {
        Self {
            data: VpwmaData::Candles { candles: c, source: s },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: VpwmaParams) -> Self {
        Self {
            data: VpwmaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", VpwmaParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
    #[inline]
    pub fn get_power(&self) -> f64 {
        self.params.power.unwrap_or(0.382)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct VpwmaBuilder {
    period: Option<usize>,
    power: Option<f64>,
    kernel: Kernel,
}

impl Default for VpwmaBuilder {
    fn default() -> Self {
        Self {
            period: None,
            power: None,
            kernel: Kernel::Auto,
        }
    }
}

impl VpwmaBuilder {
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
    pub fn power(mut self, x: f64) -> Self {
        self.power = Some(x);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<VpwmaOutput, VpwmaError> {
        let p = VpwmaParams { period: self.period, power: self.power };
        let i = VpwmaInput::from_candles(c, "close", p);
        vpwma_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<VpwmaOutput, VpwmaError> {
        let p = VpwmaParams { period: self.period, power: self.power };
        let i = VpwmaInput::from_slice(d, p);
        vpwma_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<VpwmaStream, VpwmaError> {
        let p = VpwmaParams { period: self.period, power: self.power };
        VpwmaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum VpwmaError {
    #[error("vpwma: Input data slice is empty.")]
    EmptyInputData,
    #[error("vpwma: All values are NaN.")]
    AllValuesNaN,
    #[error("vpwma: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("vpwma: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("vpwma: Invalid power: {power}")]
    InvalidPower { power: f64 },
}

#[inline]
pub fn vpwma(input: &VpwmaInput) -> Result<VpwmaOutput, VpwmaError> {
    vpwma_with_kernel(input, Kernel::Auto)
}

pub fn vpwma_with_kernel(input: &VpwmaInput, kernel: Kernel) -> Result<VpwmaOutput, VpwmaError> {
    let data: &[f64] = match &input.data {
        VpwmaData::Candles { candles, source } => source_type(candles, source),
        VpwmaData::Slice(sl) => sl,
    };

    let len = data.len();
    if len == 0 {
        return Err(VpwmaError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(VpwmaError::AllValuesNaN)?;
    let period = input.get_period();
    let power = input.get_power();

    if period < 2 || period > len {
        return Err(VpwmaError::InvalidPeriod { period, data_len: len });
    }
    if (len - first) < period {
        return Err(VpwmaError::NotEnoughValidData { needed: period, valid: len - first });
    }
    if power.is_nan() || power.is_infinite() {
        return Err(VpwmaError::InvalidPower { power });
    }

    // Build exactly (period - 1) weights
    let win_len = period - 1;
    let mut weights: AVec<f64> = AVec::with_capacity(CACHELINE_ALIGN, win_len);
    weights.resize(win_len, 0.0);

    let mut norm = 0.0;
    for k in 0..win_len {
        let w = (period as f64 - k as f64).powf(power);
        weights[k] = w;
        norm += w;
    }
    let inv_norm = 1.0 / norm;

    // VPWMA uses period-1 weights, so warmup is first + period - 1
    let warm = first + period - 1;
    let mut out = alloc_with_nan_prefix(len, warm);

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                vpwma_scalar(data, &weights, period, first, inv_norm, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                vpwma_avx2(data, &weights, period, first, inv_norm, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                vpwma_avx512(data, &weights, period, first, inv_norm, &mut out)
            }
            _ => unreachable!(),
        }
    }

    Ok(VpwmaOutput { values: out })
}
#[inline]
pub fn vpwma_scalar(
    data: &[f64],
    weights: &[f64], // length = (period - 1)
    period: usize,
    first_val: usize,
    inv_norm: f64,
    out: &mut [f64],
) {
    let win_len = period - 1;           // number of weights
    let p4 = win_len & !3;              // largest multiple of 4 <= win_len

    // We start computing at i = first_val + win_len
    for i in (first_val + win_len)..data.len() {
        let mut sum = 0.0;

        // Process in chunks of 4
        for k in (0..p4).step_by(4) {
            sum += data[i - k]         * weights[k]
                 + data[i - (k + 1)]   * weights[k + 1]
                 + data[i - (k + 2)]   * weights[k + 2]
                 + data[i - (k + 3)]   * weights[k + 3];
        }

        // Process any remainder
        for k in p4..win_len {
            sum += data[i - k] * weights[k];
        }

        out[i] = sum * inv_norm;
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn vpwma_avx512(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first_valid: usize,
    inv_norm: f64,
    out: &mut [f64],
) {
    if period <= 32 {
        unsafe { vpwma_avx512_short(data, weights, period, first_valid, inv_norm, out) }
    } else {
        unsafe { vpwma_avx512_long(data, weights, period, first_valid, inv_norm, out) }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn vpwma_avx2(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first_valid: usize,
    inv_norm: f64,
    out: &mut [f64],
) {
    vpwma_scalar(data, weights, period, first_valid, inv_norm, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
unsafe fn vpwma_avx512_short(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first_valid: usize,
    inv_norm: f64,
    out: &mut [f64],
) {
    vpwma_scalar(data, weights, period, first_valid, inv_norm, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512dq,fma")]
unsafe fn vpwma_avx512_long(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first_valid: usize,
    inv_norm: f64,
    out: &mut [f64],
) {
    vpwma_scalar(data, weights, period, first_valid, inv_norm, out);
}

// ------- 3) VpwmaStream (streaming “online” version) -------
#[derive(Debug, Clone)]
pub struct VpwmaStream {
    period: usize,
    weights: Vec<f64>,   // length = (period - 1)
    inv_norm: f64,
    buffer: Vec<f64>,    // length = period
    head: usize,
    filled: bool,
}

impl VpwmaStream {
    pub fn try_new(params: VpwmaParams) -> Result<Self, VpwmaError> {
        let period = params.period.unwrap_or(14);
        if period < 2 {
            return Err(VpwmaError::InvalidPeriod { period, data_len: 0 });
        }
        let power = params.power.unwrap_or(0.382);
        if power.is_nan() || power.is_infinite() {
            return Err(VpwmaError::InvalidPower { power });
        }

        // Build exactly (period - 1) weights
        let win_len = period - 1;
        let mut weights = Vec::with_capacity(win_len);
        let mut norm = 0.0;
        for k in 0..win_len {
            let w = (period as f64 - k as f64).powf(power);
            weights.push(w);
            norm += w;
        }
        let inv_norm = 1.0 / norm;

        Ok(Self {
            period,
            weights,
            inv_norm,
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
        // The most-recently written price is at index (head + period - 1) % period
        let mut idx = (self.head + self.period - 1) % self.period;
        let win_len = self.weights.len(); // = period - 1

        for &w in &self.weights {
            sum += w * self.buffer[idx];
            // Move one step "backwards" in the circular buffer
            idx = (idx + self.period - 1) % self.period;
        }
        sum * self.inv_norm
    }
}

#[derive(Clone, Debug)]
pub struct VpwmaBatchRange {
    pub period: (usize, usize, usize),
    pub power: (f64, f64, f64),
}

impl Default for VpwmaBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 50, 1),
            power: (0.382, 0.382, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct VpwmaBatchBuilder {
    range: VpwmaBatchRange,
    kernel: Kernel,
}

impl VpwmaBatchBuilder {
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
    pub fn power_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.power = (start, end, step);
        self
    }
    #[inline]
    pub fn power_static(mut self, p: f64) -> Self {
        self.range.power = (p, p, 0.0);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<VpwmaBatchOutput, VpwmaError> {
        vpwma_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<VpwmaBatchOutput, VpwmaError> {
        VpwmaBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<VpwmaBatchOutput, VpwmaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<VpwmaBatchOutput, VpwmaError> {
        VpwmaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn vpwma_batch_with_kernel(
    data: &[f64],
    sweep: &VpwmaBatchRange,
    k: Kernel,
) -> Result<VpwmaBatchOutput, VpwmaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(VpwmaError::InvalidPeriod { period: 0, data_len: 0 });
        }
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    vpwma_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct VpwmaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<VpwmaParams>,
    pub rows: usize,
    pub cols: usize,
}
impl VpwmaBatchOutput {
    pub fn row_for_params(&self, p: &VpwmaParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(14) == p.period.unwrap_or(14)
                && (c.power.unwrap_or(0.382) - p.power.unwrap_or(0.382)).abs() < 1e-12
        })
    }
    pub fn values_for(&self, p: &VpwmaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &VpwmaBatchRange) -> Vec<VpwmaParams> {
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
    let powers = axis_f64(r.power);
    let mut out = Vec::with_capacity(periods.len() * powers.len());
    for &p in &periods {
        for &pw in &powers {
            out.push(VpwmaParams {
                period: Some(p),
                power: Some(pw),
            });
        }
    }
    out
}

#[inline(always)]
pub fn vpwma_batch_slice(
    data: &[f64],
    sweep: &VpwmaBatchRange,
    kern: Kernel,
) -> Result<VpwmaBatchOutput, VpwmaError> {
    vpwma_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn vpwma_batch_par_slice(
    data: &[f64],
    sweep: &VpwmaBatchRange,
    kern: Kernel,
) -> Result<VpwmaBatchOutput, VpwmaError> {
    vpwma_batch_inner(data, sweep, kern, true)
}

#[inline]
fn round_up8(x: usize) -> usize {
    (x + 7) & !7
}

#[inline(always)]
fn vpwma_batch_inner(
    data: &[f64],
    sweep: &VpwmaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<VpwmaBatchOutput, VpwmaError> {
    if data.is_empty() {
        return Err(VpwmaError::EmptyInputData);
    }
    
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(VpwmaError::InvalidPeriod { period: 0, data_len: 0 });
    }

    let first = data.iter().position(|x| !x.is_nan()).ok_or(VpwmaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| round_up8(c.period.unwrap())).max().unwrap();
    if data.len() - first < max_p {
        return Err(VpwmaError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let mut inv_norms = vec![0.0; rows];
    let cap = rows * max_p;
    let mut flat_w = AVec::<f64>::with_capacity(CACHELINE_ALIGN, cap);
    flat_w.resize(cap, 0.0);

    // Build, for each combo, exactly (period - 1) weights
    for (row, prm) in combos.iter().enumerate() {
        let period = prm.period.unwrap();
        let power = prm.power.unwrap();
        if power.is_nan() || power.is_infinite() {
            return Err(VpwmaError::InvalidPower { power });
        }
        let win_len = period - 1;
        let mut norm = 0.0;
        for k in 0..win_len {
            let w = (period as f64 - k as f64).powf(power);
            flat_w[row * max_p + k] = w;
            norm += w;
        }
        inv_norms[row] = 1.0 / norm;
    }

    // VPWMA uses period-1 weights, so warmup is first + period - 1
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();

    // --- 2.  allocate an uninitialised rows×cols matrix and stamp the prefixes --
    let mut raw = make_uninit_matrix(rows, cols);
    unsafe { init_matrix_prefixes(&mut raw, cols, &warm) };

    // --- 3.  closure that fills ONE row; it works on &mut [MaybeUninit<f64>] ----
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();
        let w_ptr  = flat_w.as_ptr().add(row * max_p);
        let inv_n  = *inv_norms.get_unchecked(row);

        // Cast just this row to &mut [f64] so the row-kernel can write into it.
        let out_row = core::slice::from_raw_parts_mut(
            dst_mu.as_mut_ptr() as *mut f64,
            dst_mu.len(),
        );

        match kern {
            Kernel::Scalar => vpwma_row_scalar(data, first, period, max_p, w_ptr, inv_n, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2   => vpwma_row_avx2  (data, first, period, max_p, w_ptr, inv_n, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => vpwma_row_avx512(data, first, period, max_p, w_ptr, inv_n, out_row),
            _ => unreachable!(),
        }
    };

    // --- 4.  run every row (parallel or serial) ---------------------------------
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

    // --- 5.  all elements are initialised; transmute into Vec<f64> --------------
    let values: Vec<f64> = unsafe { std::mem::transmute(raw) };

    Ok(VpwmaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn vpwma_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,    // (unused beyond satisfying signature)
    w_ptr: *const f64, // pointer to weights[0..(period-1)]
    inv_n: f64,
    out: &mut [f64],
) {
    let win_len = period - 1;
    let p4 = win_len & !3;

    // Compute from i = first + win_len onward
    for i in (first + win_len)..data.len() {
        let mut sum = 0.0;

        // Process k = 0..(p4-1) in blocks of 4
        for k in (0..p4).step_by(4) {
            sum += *data.get_unchecked(i - k)         * *w_ptr.add(k)
                 + *data.get_unchecked(i - (k + 1))   * *w_ptr.add(k + 1)
                 + *data.get_unchecked(i - (k + 2))   * *w_ptr.add(k + 2)
                 + *data.get_unchecked(i - (k + 3))   * *w_ptr.add(k + 3);
        }

        // Process remainder k = p4..(win_len-1)
        for k in p4..win_len {
            sum += *data.get_unchecked(i - k) * *w_ptr.add(k);
        }

        out[i] = sum * inv_n;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn vpwma_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    w_ptr: *const f64,
    inv_n: f64,
    out: &mut [f64],
) {
    vpwma_row_scalar(data, first, period, stride, w_ptr, inv_n, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma,avx512dq")]
pub unsafe fn vpwma_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    w_ptr: *const f64,
    inv_n: f64,
    out: &mut [f64],
) {
    if period <= 32 {
        vpwma_row_avx512_short(data, first, period, stride, w_ptr, inv_n, out);
    
        } else {
        vpwma_row_avx512_long(data, first, period, stride, w_ptr, inv_n, out);
    }
    _mm_sfence();
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
unsafe fn vpwma_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    w_ptr: *const f64,
    inv_n: f64,
    out: &mut [f64],
) {
    vpwma_row_scalar(data, first, period, _stride, w_ptr, inv_n, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512dq,fma")]
pub(crate) unsafe fn vpwma_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    w_ptr: *const f64,
    inv_n: f64,
    out: &mut [f64],
) {
    vpwma_row_scalar(data, first, period, _stride, w_ptr, inv_n, out)
}

// ----- TESTS -----
#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_vpwma_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = VpwmaParams { period: None, power: None };
        let input = VpwmaInput::from_candles(&candles, "close", default_params);
        let output = vpwma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_vpwma_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = VpwmaInput::from_candles(&candles, "close", VpwmaParams::default());
        let result = vpwma_with_kernel(&input, kernel)?;
        let expected_last_five = [
            59363.927599446455,
            59296.83894519251,
            59196.82476139941,
            59180.8040249446,
            59113.84473799056,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-2,
                "[{}] VPWMA {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_vpwma_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = VpwmaParams { period: Some(0), power: None };
        let input = VpwmaInput::from_slice(&input_data, params);
        let res = vpwma_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] VPWMA should fail with zero period", test_name);
        Ok(())
    }

    fn check_vpwma_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = VpwmaParams { period: Some(10), power: None };
        let input = VpwmaInput::from_slice(&data_small, params);
        let res = vpwma_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] VPWMA should fail with period exceeding length", test_name);
        Ok(())
    }

    fn check_vpwma_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = VpwmaParams { period: Some(2), power: None };
        let input = VpwmaInput::from_slice(&single_point, params);
        let res = vpwma_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] VPWMA should fail with insufficient data", test_name);
        Ok(())
    }

    fn check_vpwma_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = VpwmaParams { period: Some(14), power: None };
        let first_input = VpwmaInput::from_candles(&candles, "close", first_params);
        let first_result = vpwma_with_kernel(&first_input, kernel)?;
        let second_params = VpwmaParams { period: Some(5), power: Some(0.5) };
        let second_input = VpwmaInput::from_slice(&first_result.values, second_params);
        let second_result = vpwma_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        if second_result.values.len() > 240 {
            for i in 240..second_result.values.len() {
                assert!(!second_result.values[i].is_nan());
            }
        }
        Ok(())
    }

    fn check_vpwma_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = VpwmaInput::from_candles(
            &candles,
            "close",
            VpwmaParams { period: Some(14), power: None }
        );
        let res = vpwma_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 50 {
            for (i, &val) in res.values[50..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test_name,
                    50 + i
                );
            }
        }
        Ok(())
    }

    fn check_vpwma_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 14;
        let power = 0.382;
        let input = VpwmaInput::from_candles(
            &candles,
            "close",
            VpwmaParams { period: Some(period), power: Some(power) }
        );
        let batch_output = vpwma_with_kernel(&input, kernel)?.values;
        let mut stream = VpwmaStream::try_new(VpwmaParams { period: Some(period), power: Some(power) })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(vpwma_val) => stream_values.push(vpwma_val),
                None => stream_values.push(f64::NAN),
            }
        }
        assert_eq!(batch_output.len(), stream_values.len());
        for (i, (&b, &s)) in batch_output.iter().zip(stream_values.iter()).enumerate() {
            if b.is_nan() && s.is_nan() { continue; }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] VPWMA streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name, i, b, s, diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_vpwma_tests {
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
    fn check_vpwma_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test with multiple parameter combinations to increase coverage
        let test_periods = vec![2, 5, 10, 14, 30, 50];
        let test_powers = vec![0.1, 0.382, 0.5, 1.0, 2.0];
        let test_sources = vec!["close", "open", "high", "low", "hl2", "hlc3", "ohlc4"];

        for period in test_periods {
            for power in &test_powers {
                for source in &test_sources {
                    let params = VpwmaParams { 
                        period: Some(period),
                        power: Some(*power)
                    };
                    let input = VpwmaInput::from_candles(&candles, source, params);
                    let output = vpwma_with_kernel(&input, kernel)?;

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
                                "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} (period={}, power={}, source={})",
                                test_name, val, bits, i, period, power, source
                            );
                        }

                        // Check for init_matrix_prefixes poison (0x22222222_22222222)
                        if bits == 0x22222222_22222222 {
                            panic!(
                                "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} (period={}, power={}, source={})",
                                test_name, val, bits, i, period, power, source
                            );
                        }

                        // Check for make_uninit_matrix poison (0x33333333_33333333)
                        if bits == 0x33333333_33333333 {
                            panic!(
                                "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} (period={}, power={}, source={})",
                                test_name, val, bits, i, period, power, source
                            );
                        }
                    }
                }
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_vpwma_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    generate_all_vpwma_tests!(
        check_vpwma_partial_params,
        check_vpwma_accuracy,
        check_vpwma_zero_period,
        check_vpwma_period_exceeds_length,
        check_vpwma_very_small_dataset,
        check_vpwma_reinput,
        check_vpwma_nan_handling,
        check_vpwma_streaming,
        check_vpwma_no_poison
    );
    
    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
    skip_if_unsupported!(kernel, test);
    let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
    let c = read_candles_from_csv(file)?;
    let output = VpwmaBatchBuilder::new()
        .kernel(kernel)
        .apply_candles(&c, "close")?;
    let def = VpwmaParams::default();
    let row = output.values_for(&def).expect("default row missing");
    assert_eq!(row.len(), c.close.len());

    let expected = [
        59363.927599446455,
        59296.83894519251,
        59196.82476139941,
        59180.8040249446,
        59113.84473799056,
    ];
    let start = row.len() - 5;
    for (i, &v) in row[start..].iter().enumerate() {
        assert!(
            (v - expected[i]).abs() < 1e-2,
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
// Check for poison values in batch output - only runs in debug mode
#[cfg(debug_assertions)]
fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
    skip_if_unsupported!(kernel, test);

    let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
    let c = read_candles_from_csv(file)?;

    // Test batch with multiple parameter range combinations
    let period_ranges = vec![
        (2, 10, 2),    // Small periods
        (10, 30, 5),   // Medium periods  
        (30, 60, 10),  // Large periods
        (5, 15, 1),    // Dense small range
    ];
    
    let power_ranges = vec![
        (0.1, 0.5, 0.1),   // Low powers
        (0.3, 1.0, 0.2),   // Medium powers
        (1.0, 3.0, 0.5),   // High powers
    ];

    let test_sources = vec!["close", "open", "high", "low", "hl2", "hlc3", "ohlc4"];

    for (p_start, p_end, p_step) in period_ranges {
        for (pow_start, pow_end, pow_step) in &power_ranges {
            for source in &test_sources {
                // Create power values from range
                let mut power_values = vec![];
                let mut current = *pow_start;
                while current <= *pow_end {
                    power_values.push(current);
                    current += pow_step;
                }
                
                let output = VpwmaBatchBuilder::new()
                    .kernel(kernel)
                    .period_range(p_start, p_end, p_step)
                    .power_range(*pow_start, *pow_end, *pow_step)
                    .apply_candles(&c, source)?;

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
                            "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with period_range({},{},{}) power_range({},{},{}) source={}",
                            test, val, bits, row, col, idx, p_start, p_end, p_step, pow_start, pow_end, pow_step, source
                        );
                    }

                    // Check for init_matrix_prefixes poison (0x22222222_22222222)
                    if bits == 0x22222222_22222222 {
                        panic!(
                            "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {}) with period_range({},{},{}) power_range({},{},{}) source={}",
                            test, val, bits, row, col, idx, p_start, p_end, p_step, pow_start, pow_end, pow_step, source
                        );
                    }

                    // Check for make_uninit_matrix poison (0x33333333_33333333)
                    if bits == 0x33333333_33333333 {
                        panic!(
                            "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with period_range({},{},{}) power_range({},{},{}) source={}",
                            test, val, bits, row, col, idx, p_start, p_end, p_step, pow_start, pow_end, pow_step, source
                        );
                    }
                }
            }
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
#[pyfunction(name = "vpwma")]
#[pyo3(signature = (data, period, power, kernel=None))]
/// Compute the Variable Power Weighted Moving Average (VPWMA) of the input data.
///
/// VPWMA adjusts the weights of each price data point based on their respective 
/// positions, with the weight raised to a specified power to control how 
/// aggressively recent data points dominate the average.
///
/// Parameters:
/// -----------
/// data : np.ndarray
///     Input data array (float64).
/// period : int
///     Number of data points in the moving average window (must be >= 2).
/// power : float
///     Exponent applied to the weight function (typically 0.382).
/// kernel : str, optional
///     Computation kernel to use: 'auto', 'scalar', 'avx2', 'avx512'.
///     Default is 'auto' which auto-detects the best available.
///
/// Returns:
/// --------
/// np.ndarray
///     Array of VPWMA values, same length as input.
///
/// Raises:
/// -------
/// ValueError
///     If inputs are invalid (period < 2, power is NaN/infinite, etc).
pub fn vpwma_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
    power: f64,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{PyArray1, PyArrayMethods};

    let slice_in = data.as_slice()?; // zero-copy, read-only view

    // Parse kernel string to enum with CPU feature validation
    let kern = validate_kernel(kernel, false)?;

    // Build input struct
    let params = VpwmaParams {
        period: Some(period),
        power: Some(power),
    };
    let vpwma_in = VpwmaInput::from_slice(slice_in, params);

    // Allocate NumPy output buffer
    let out_arr = unsafe { PyArray1::<f64>::new(py, [slice_in.len()], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? }; // safe: we own the array

    // Heavy lifting without the GIL
    py.allow_threads(|| -> Result<(), VpwmaError> {
        let data: &[f64] = vpwma_in.as_ref();
        let len = data.len();
        if len == 0 {
            return Err(VpwmaError::EmptyInputData);
        }
        let first = data
            .iter()
            .position(|x| !x.is_nan())
            .ok_or(VpwmaError::AllValuesNaN)?;
        let period = vpwma_in.get_period();
        let power = vpwma_in.get_power();

        // Validation
        if period < 2 || period > len {
            return Err(VpwmaError::InvalidPeriod { period, data_len: len });
        }
        if data.len() - first < period {
            return Err(VpwmaError::NotEnoughValidData {
                needed: period,
                valid: data.len() - first,
            });
        }
        if power.is_nan() || power.is_infinite() {
            return Err(VpwmaError::InvalidPower { power });
        }

        // Build weights once
        let win_len = period - 1;
        let weights: AVec<f64> = AVec::from_iter(
            CACHELINE_ALIGN,
            (0..win_len).map(|k| (period as f64 - k as f64).powf(power)),
        );
        let norm: f64 = weights.iter().sum();
        let inv_norm = 1.0 / norm;

        // Kernel auto-detection
        let chosen = match kern {
            Kernel::Auto => detect_best_kernel(),
            k => k,
        };

        // Prefix initialize with NaN (warmup is first + period - 1 for VPWMA)
        slice_out[..first + period - 1].fill(f64::NAN);

        // Compute VPWMA
        unsafe {
            match chosen {
                Kernel::Scalar => {
                    vpwma_scalar(data, &weights, period, first, inv_norm, slice_out)
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2 => vpwma_avx2(data, &weights, period, first, inv_norm, slice_out),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512 => vpwma_avx512(data, &weights, period, first, inv_norm, slice_out),
                #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
                Kernel::Avx2 | Kernel::Avx512 => {
                    vpwma_scalar(data, &weights, period, first, inv_norm, slice_out)
                }
                _ => unreachable!(),
            }
        }

        Ok(())
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(out_arr.into())
}

#[cfg(feature = "python")]
#[pyclass(name = "VpwmaStream")]
pub struct VpwmaStreamPy {
    stream: VpwmaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl VpwmaStreamPy {
    #[new]
    fn new(period: usize, power: f64) -> PyResult<Self> {
        let params = VpwmaParams {
            period: Some(period),
            power: Some(power),
        };
        let stream =
            VpwmaStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(VpwmaStreamPy { stream })
    }

    /// Updates the stream with a new value and returns the calculated VPWMA value.
    /// Returns `None` if the buffer is not yet full.
    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "vpwma_batch")]
#[pyo3(signature = (data, period_range, power_range, kernel=None))]
/// Compute VPWMA for multiple parameter combinations in a single pass.
///
/// Parameters:
/// -----------
/// data : np.ndarray
///     Input data array (float64).
/// period_range : tuple
///     (start, end, step) for period values to compute.
/// power_range : tuple
///     (start, end, step) for power values to compute.
/// kernel : str, optional
///     Computation kernel: 'auto', 'scalar', 'avx2', 'avx512'.
///     Default is 'auto' which auto-detects the best available.
///
/// Returns:
/// --------
/// dict
///     Dictionary with 'values' (2D array), 'periods', and 'powers' arrays.
pub fn vpwma_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    power_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;

    let sweep = VpwmaBatchRange {
        period: period_range,
        power: power_range,
    };

    // Expand grid once to know rows*cols
    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();

    // Pre-allocate NumPy array (1-D, will reshape later)
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    // Parse kernel string to enum with CPU feature validation
    let kern = validate_kernel(kernel, true)?;

    // Heavy work without the GIL
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
        
        // Use the existing batch function
        let output = vpwma_batch_inner(slice_in, &sweep, simd, true)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        // Copy the values to our pre-allocated buffer
        slice_out.copy_from_slice(&output.values);
        Ok::<Vec<VpwmaParams>, PyErr>(output.combos)
    })?;

    // Build result dictionary
    let dict = PyDict::new(py);

    // Reshape the flat array into 2D
    let reshaped = out_arr.reshape([rows, cols])?;
    dict.set_item("values", reshaped)?;

    // Extract periods and powers
    let periods: Vec<usize> = combos.iter().map(|c| c.period.unwrap()).collect();
    let powers: Vec<f64> = combos.iter().map(|c| c.power.unwrap()).collect();

    dict.set_item("periods", periods.into_pyarray(py))?;
    dict.set_item("powers", powers.into_pyarray(py))?;

    Ok(dict)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vpwma_js(data: &[f64], period: usize, power: f64) -> Result<Vec<f64>, JsValue> {
    let params = VpwmaParams {
        period: Some(period),
        power: Some(power),
    };
    let input = VpwmaInput::from_slice(data, params);

    vpwma_with_kernel(&input, Kernel::Scalar)
        .map(|o| o.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vpwma_batch_js(
    data: &[f64],
    period_start: usize,
    period_end: usize,
    period_step: usize,
    power_start: f64,
    power_end: f64,
    power_step: f64,
) -> Result<Vec<f64>, JsValue> {
    let sweep = VpwmaBatchRange {
        period: (period_start, period_end, period_step),
        power: (power_start, power_end, power_step),
    };

    // Use the existing batch function with parallel=false for WASM
    vpwma_batch_inner(data, &sweep, Kernel::Scalar, false)
        .map(|output| output.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn vpwma_batch_metadata_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
    power_start: f64,
    power_end: f64,
    power_step: f64,
) -> Result<Vec<f64>, JsValue> {
    let sweep = VpwmaBatchRange {
        period: (period_start, period_end, period_step),
        power: (power_start, power_end, power_step),
    };

    let combos = expand_grid(&sweep);
    let mut metadata = Vec::with_capacity(combos.len() * 2);

    for combo in combos {
        metadata.push(combo.period.unwrap() as f64);
        metadata.push(combo.power.unwrap());
    }

    Ok(metadata)
}
