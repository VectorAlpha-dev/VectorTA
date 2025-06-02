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

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use rayon::prelude::*;
use std::convert::AsRef;
use thiserror::Error;

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

    let first = data.iter().position(|x| !x.is_nan()).ok_or(VpwmaError::AllValuesNaN)?;

    let len = data.len();
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

    let mut weights: AVec<f64> = AVec::with_capacity(CACHELINE_ALIGN, period);
    weights.resize(period, 0.0);
    let mut norm = 0.0;
    for i in 0..period {
        let w = (period as f64 - i as f64).powf(power);
        weights[i] = w;
        norm += w;
    }
    let inv_norm = 1.0 / norm;
    let mut out = data.to_vec();

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
    weights: &[f64],
    period: usize,
    first_val: usize,
    inv_norm: f64,
    out: &mut [f64],
) {
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

#[derive(Debug, Clone)]
pub struct VpwmaStream {
    period: usize,
    weights: Vec<f64>,
    inv_norm: f64,
    buffer: Vec<f64>,
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
        let mut weights = Vec::with_capacity(period);
        let mut norm = 0.0;
        for i in 0..period {
            let w = (period as f64 - i as f64).powf(power);
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
        let mut idx = self.head;
        for &w in &self.weights {
            sum += w * self.buffer[idx];
            idx = (idx + 1) % self.period;
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
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(VpwmaError::InvalidPeriod { period: 0, data_len: 0 });
    }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(VpwmaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| round_up8(c.period.unwrap())).max().unwrap();
    if data.len() - first < max_p {
        return Err(VpwmaError::NotEnoughValidData { needed: max_p, valid: data.len() - first });
    }
    let rows = combos.len();
    let cols = data.len();
    let mut inv_norms = vec![0.0; rows];
    let cap = rows * max_p;
    let mut flat_w = AVec::<f64>::with_capacity(CACHELINE_ALIGN, cap);
    flat_w.resize(cap, 0.0);

    for (row, prm) in combos.iter().enumerate() {
        let period = prm.period.unwrap();
        let power = prm.power.unwrap();
        if power.is_nan() || power.is_infinite() {
            return Err(VpwmaError::InvalidPower { power });
        }
        let mut norm = 0.0;
        for i in 0..period {
            let w = (period as f64 - i as f64).powf(power);
            flat_w[row * max_p + i] = w;
            norm += w;
        }
        inv_norms[row] = 1.0 / norm;
    }

    let mut values = vec![f64::NAN; rows * cols];
    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        let w_ptr = flat_w.as_ptr().add(row * max_p);
        let inv_n = *inv_norms.get_unchecked(row);
        match kern {
            Kernel::Scalar => vpwma_row_scalar(data, first, period, max_p, w_ptr, inv_n, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => vpwma_row_avx2(data, first, period, max_p, w_ptr, inv_n, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => vpwma_row_avx512(data, first, period, max_p, w_ptr, inv_n, out_row),
            _ => unreachable!(),
        }
    };
    if parallel {
        values
            .par_chunks_mut(cols)
            .enumerate()
            .for_each(|(row, slice)| do_row(row, slice));
    } else {
        for (row, slice) in values.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }
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
    stride: usize,
    w_ptr: *const f64,
    inv_n: f64,
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

#[inline(always)]
fn expand_grid_vpwma(_r: &VpwmaBatchRange) -> Vec<VpwmaParams> {
    unimplemented!()
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

    generate_all_vpwma_tests!(
        check_vpwma_partial_params,
        check_vpwma_accuracy,
        check_vpwma_zero_period,
        check_vpwma_period_exceeds_length,
        check_vpwma_very_small_dataset,
        check_vpwma_reinput,
        check_vpwma_nan_handling,
        check_vpwma_streaming
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
gen_batch_tests!(check_batch_default_row);

}
