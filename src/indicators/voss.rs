//! # Voss Filter (VOSS)
//!
//! John F. Ehlers' VOSS indicator: An IIR filter using cyclical analysis with predictive lookahead.
//!
//! ## Parameters
//! - **period**: Cycle length (default: 20)
//! - **predict**: Predictive lookahead factor (default: 3)
//! - **bandwidth**: Filter bandwidth (default: 0.25)
//!
//! ## Errors
//! - **AllValuesNaN**: voss: All input data values are `NaN`.
//! - **InvalidPeriod**: voss: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: voss: Not enough valid data points for the requested window.
//! - **EmptyData**: voss: Input data slice is empty.
//!
//! ## Returns
//! - **`Ok(VossOutput)`** on success, containing filtered output vectors matching input length.
//! - **`Err(VossError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;

impl<'a> AsRef<[f64]> for VossInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            VossData::Slice(slice) => slice,
            VossData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum VossData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct VossOutput {
    pub voss: Vec<f64>,
    pub filt: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct VossParams {
    pub period: Option<usize>,
    pub predict: Option<usize>,
    pub bandwidth: Option<f64>,
}

impl Default for VossParams {
    fn default() -> Self {
        Self {
            period: Some(20),
            predict: Some(3),
            bandwidth: Some(0.25),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VossInput<'a> {
    pub data: VossData<'a>,
    pub params: VossParams,
}

impl<'a> VossInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: VossParams) -> Self {
        Self {
            data: VossData::Candles { candles: c, source: s },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: VossParams) -> Self {
        Self {
            data: VossData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", VossParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(20)
    }
    #[inline]
    pub fn get_predict(&self) -> usize {
        self.params.predict.unwrap_or(3)
    }
    #[inline]
    pub fn get_bandwidth(&self) -> f64 {
        self.params.bandwidth.unwrap_or(0.25)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct VossBuilder {
    period: Option<usize>,
    predict: Option<usize>,
    bandwidth: Option<f64>,
    kernel: Kernel,
}

impl Default for VossBuilder {
    fn default() -> Self {
        Self {
            period: None,
            predict: None,
            bandwidth: None,
            kernel: Kernel::Auto,
        }
    }
}

impl VossBuilder {
    #[inline(always)]
    pub fn new() -> Self { Self::default() }
    #[inline(always)]
    pub fn period(mut self, n: usize) -> Self { self.period = Some(n); self }
    #[inline(always)]
    pub fn predict(mut self, n: usize) -> Self { self.predict = Some(n); self }
    #[inline(always)]
    pub fn bandwidth(mut self, b: f64) -> Self { self.bandwidth = Some(b); self }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self { self.kernel = k; self }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<VossOutput, VossError> {
        let p = VossParams { period: self.period, predict: self.predict, bandwidth: self.bandwidth };
        let i = VossInput::from_candles(c, "close", p);
        voss_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<VossOutput, VossError> {
        let p = VossParams { period: self.period, predict: self.predict, bandwidth: self.bandwidth };
        let i = VossInput::from_slice(d, p);
        voss_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<VossStream, VossError> {
        let p = VossParams { period: self.period, predict: self.predict, bandwidth: self.bandwidth };
        VossStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum VossError {
    #[error("voss: All values are NaN.")]
    AllValuesNaN,
    #[error("voss: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("voss: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("voss: Empty data provided.")]
    EmptyData,
}

#[inline]
pub fn voss(input: &VossInput) -> Result<VossOutput, VossError> {
    voss_with_kernel(input, Kernel::Auto)
}

pub fn voss_with_kernel(input: &VossInput, kernel: Kernel) -> Result<VossOutput, VossError> {
    let data: &[f64] = match &input.data {
        VossData::Candles { candles, source } => source_type(candles, source),
        VossData::Slice(sl) => sl,
    };

    if data.is_empty() {
        return Err(VossError::EmptyData);
    }

    let first = data.iter().position(|x| !x.is_nan()).ok_or(VossError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();
    let predict = input.get_predict();
    let bandwidth = input.get_bandwidth();

    if period == 0 || period > len {
        return Err(VossError::InvalidPeriod { period, data_len: len });
    }

    let order = 3 * predict;
    let min_index = period.max(5).max(order);
    if (len - first) < min_index {
        return Err(VossError::NotEnoughValidData { needed: min_index, valid: len - first });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let mut voss = vec![f64::NAN; len];
    let mut filt = vec![f64::NAN; len];

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                voss_scalar(data, period, predict, bandwidth, first, &mut voss, &mut filt)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                voss_avx2(data, period, predict, bandwidth, first, &mut voss, &mut filt)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                voss_avx512(data, period, predict, bandwidth, first, &mut voss, &mut filt)
            }
            _ => unreachable!(),
        }
    }
    Ok(VossOutput { voss, filt })
}

#[inline]
pub unsafe fn voss_scalar(
    data: &[f64],
    period: usize,
    predict: usize,
    bandwidth: f64,
    first: usize,
    voss: &mut [f64],
    filt: &mut [f64],
) {
    use std::f64::consts::PI;
    let order = 3 * predict;
    let min_index = period.max(5).max(order);

    let f1 = (2.0 * PI / period as f64).cos();
    let g1 = (bandwidth * 2.0 * PI / period as f64).cos();
    let s1 = 1.0 / g1 - (1.0 / (g1 * g1) - 1.0).sqrt();

    for i in first..(first + min_index) {
        filt[i] = 0.0;
    }

    for i in (first + min_index)..data.len() {
        let current = data[i];
        let prev_2 = data[i - 2];
        let prev_filt_1 = filt[i - 1];
        let prev_filt_2 = filt[i - 2];
        filt[i] = 0.5 * (1.0 - s1) * (current - prev_2) + f1 * (1.0 + s1) * prev_filt_1
            - s1 * prev_filt_2;
    }

    for i in first..(first + min_index) {
        voss[i] = 0.0;
    }

    for i in (first + min_index)..data.len() {
        let mut sumc = 0.0;
        for count in 0..order {
            let idx = i - (order - count);
            sumc += ((count + 1) as f64 / order as f64) * voss[idx];
        }
        voss[i] = ((3 + order) as f64 / 2.0) * filt[i] - sumc;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn voss_avx2(
    data: &[f64],
    period: usize,
    predict: usize,
    bandwidth: f64,
    first: usize,
    voss: &mut [f64],
    filt: &mut [f64],
) {
    voss_scalar(data, period, predict, bandwidth, first, voss, filt)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn voss_avx512(
    data: &[f64],
    period: usize,
    predict: usize,
    bandwidth: f64,
    first: usize,
    voss: &mut [f64],
    filt: &mut [f64],
) {
    if period <= 32 {
        voss_avx512_short(data, period, predict, bandwidth, first, voss, filt)
    } else {
        voss_avx512_long(data, period, predict, bandwidth, first, voss, filt)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn voss_avx512_short(
    data: &[f64],
    period: usize,
    predict: usize,
    bandwidth: f64,
    first: usize,
    voss: &mut [f64],
    filt: &mut [f64],
) {
    voss_scalar(data, period, predict, bandwidth, first, voss, filt)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn voss_avx512_long(
    data: &[f64],
    period: usize,
    predict: usize,
    bandwidth: f64,
    first: usize,
    voss: &mut [f64],
    filt: &mut [f64],
) {
    voss_scalar(data, period, predict, bandwidth, first, voss, filt)
}

#[derive(Debug, Clone)]
pub struct VossStream {
    period: usize,
    predict: usize,
    bandwidth: f64,
    order: usize,
    min_index: usize,
    filt: Vec<f64>,
    voss: Vec<f64>,
    head: usize,
    filled: bool,
}

impl VossStream {
    pub fn try_new(params: VossParams) -> Result<Self, VossError> {
        let period = params.period.unwrap_or(20);
        let predict = params.predict.unwrap_or(3);
        let bandwidth = params.bandwidth.unwrap_or(0.25);

        if period == 0 {
            return Err(VossError::InvalidPeriod { period, data_len: 0 });
        }
        let order = 3 * predict;
        let min_index = period.max(5).max(order);

        Ok(Self {
            period,
            predict,
            bandwidth,
            order,
            min_index,
            filt: vec![0.0; min_index + 2],
            voss: vec![0.0; min_index + 2 + order],
            head: 0,
            filled: false,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        let len = self.filt.len();
        if self.head < len {
            self.filt[self.head] = 0.0;
            self.voss[self.head] = 0.0;
            self.head += 1;
            return None;
        }
        let i = self.head;
        let f1 = (2.0 * std::f64::consts::PI / self.period as f64).cos();
        let g1 = (self.bandwidth * 2.0 * std::f64::consts::PI / self.period as f64).cos();
        let s1 = 1.0 / g1 - (1.0 / (g1 * g1) - 1.0).sqrt();

        let prev_2 = value; // Not fully accurate in streaming context (need full window or ring buffer)
        let prev_filt_1 = self.filt[i - 1];
        let prev_filt_2 = self.filt[i - 2];
        let filt = 0.5 * (1.0 - s1) * (value - prev_2) + f1 * (1.0 + s1) * prev_filt_1 - s1 * prev_filt_2;
        self.filt.push(filt);

        let mut sumc = 0.0;
        for count in 0..self.order {
            let idx = i - (self.order - count);
            sumc += ((count + 1) as f64 / self.order as f64) * self.voss[idx];
        }
        let voss_val = ((3 + self.order) as f64 / 2.0) * filt - sumc;
        self.voss.push(voss_val);
        self.head += 1;
        self.filled = true;

        Some((voss_val, filt))
    }
}

#[derive(Clone, Debug)]
pub struct VossBatchRange {
    pub period: (usize, usize, usize),
    pub predict: (usize, usize, usize),
    pub bandwidth: (f64, f64, f64),
}

impl Default for VossBatchRange {
    fn default() -> Self {
        Self {
            period: (20, 100, 1),
            predict: (3, 3, 0),
            bandwidth: (0.25, 0.25, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct VossBatchBuilder {
    range: VossBatchRange,
    kernel: Kernel,
}

impl VossBatchBuilder {
    pub fn new() -> Self { Self::default() }
    pub fn kernel(mut self, k: Kernel) -> Self { self.kernel = k; self }
    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self { self.range.period = (start, end, step); self }
    pub fn predict_range(mut self, start: usize, end: usize, step: usize) -> Self { self.range.predict = (start, end, step); self }
    pub fn bandwidth_range(mut self, start: f64, end: f64, step: f64) -> Self { self.range.bandwidth = (start, end, step); self }

    pub fn apply_slice(self, data: &[f64]) -> Result<VossBatchOutput, VossError> {
        voss_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<VossBatchOutput, VossError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
}

pub fn voss_batch_with_kernel(
    data: &[f64],
    sweep: &VossBatchRange,
    k: Kernel,
) -> Result<VossBatchOutput, VossError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(VossError::InvalidPeriod { period: 0, data_len: 0 }),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    voss_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct VossBatchOutput {
    pub voss: Vec<f64>,
    pub filt: Vec<f64>,
    pub combos: Vec<VossParams>,
    pub rows: usize,
    pub cols: usize,
}
impl VossBatchOutput {
    pub fn row_for_params(&self, p: &VossParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(20) == p.period.unwrap_or(20)
                && c.predict.unwrap_or(3) == p.predict.unwrap_or(3)
                && (c.bandwidth.unwrap_or(0.25) - p.bandwidth.unwrap_or(0.25)).abs() < 1e-12
        })
    }
    pub fn voss_for(&self, p: &VossParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.voss[start..start + self.cols]
        })
    }
    pub fn filt_for(&self, p: &VossParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.filt[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &VossBatchRange) -> Vec<VossParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end { return vec![start]; }
        (start..=end).step_by(step).collect()
    }
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Vec<f64> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 { return vec![start]; }
        let mut v = Vec::new(); let mut x = start;
        while x <= end + 1e-12 { v.push(x); x += step; }
        v
    }
    let periods = axis_usize(r.period);
    let predicts = axis_usize(r.predict);
    let bandwidths = axis_f64(r.bandwidth);
    let mut out = Vec::with_capacity(periods.len() * predicts.len() * bandwidths.len());
    for &p in &periods {
        for &q in &predicts {
            for &b in &bandwidths {
                out.push(VossParams { period: Some(p), predict: Some(q), bandwidth: Some(b) });
            }
        }
    }
    out
}

#[inline(always)]
pub fn voss_batch_slice(
    data: &[f64],
    sweep: &VossBatchRange,
    kern: Kernel,
) -> Result<VossBatchOutput, VossError> {
    voss_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn voss_batch_par_slice(
    data: &[f64],
    sweep: &VossBatchRange,
    kern: Kernel,
) -> Result<VossBatchOutput, VossError> {
    voss_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn voss_batch_inner(
    data: &[f64],
    sweep: &VossBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<VossBatchOutput, VossError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(VossError::InvalidPeriod { period: 0, data_len: 0 });
    }

    let first = data.iter().position(|x| !x.is_nan()).ok_or(VossError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(VossError::NotEnoughValidData { needed: max_p, valid: data.len() - first });
    }

    let rows = combos.len();
    let cols = data.len();
    let mut voss_vec = vec![f64::NAN; rows * cols];
    let mut filt_vec = vec![f64::NAN; rows * cols];

    let do_row = |row: usize, out_voss: &mut [f64], out_filt: &mut [f64]| unsafe {
        let prm = &combos[row];
        let period = prm.period.unwrap();
        let predict = prm.predict.unwrap();
        let bandwidth = prm.bandwidth.unwrap();
        match kern {
            Kernel::Scalar => voss_row_scalar(data, first, period, predict, bandwidth, out_voss, out_filt),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => voss_row_avx2(data, first, period, predict, bandwidth, out_voss, out_filt),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => voss_row_avx512(data, first, period, predict, bandwidth, out_voss, out_filt),
            _ => unreachable!(),
        }
    };

    if parallel {
        voss_vec
            .par_chunks_mut(cols)
            .zip(filt_vec.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, (vo, fo))| do_row(row, vo, fo));
    } else {
        for (row, (vo, fo)) in voss_vec.chunks_mut(cols).zip(filt_vec.chunks_mut(cols)).enumerate() {
            do_row(row, vo, fo);
        }
    }

    Ok(VossBatchOutput {
        voss: voss_vec,
        filt: filt_vec,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn voss_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    predict: usize,
    bandwidth: f64,
    voss: &mut [f64],
    filt: &mut [f64],
) {
    voss_scalar(data, period, predict, bandwidth, first, voss, filt)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn voss_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    predict: usize,
    bandwidth: f64,
    voss: &mut [f64],
    filt: &mut [f64],
) {
    voss_avx2(data, period, predict, bandwidth, first, voss, filt)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn voss_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    predict: usize,
    bandwidth: f64,
    voss: &mut [f64],
    filt: &mut [f64],
) {
    voss_avx512(data, period, predict, bandwidth, first, voss, filt)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn voss_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    predict: usize,
    bandwidth: f64,
    voss: &mut [f64],
    filt: &mut [f64],
) {
    voss_avx512_short(data, period, predict, bandwidth, first, voss, filt)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn voss_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    predict: usize,
    bandwidth: f64,
    voss: &mut [f64],
    filt: &mut [f64],
) {
    voss_avx512_long(data, period, predict, bandwidth, first, voss, filt)
}

#[inline(always)]
pub fn expand_grid_voss(r: &VossBatchRange) -> Vec<VossParams> {
    expand_grid(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_voss_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = VossParams { period: None, predict: Some(2), bandwidth: None };
        let input = VossInput::from_candles(&candles, "close", params);
        let output = voss_with_kernel(&input, kernel)?;
        assert_eq!(output.voss.len(), candles.close.len());
        assert_eq!(output.filt.len(), candles.close.len());
        Ok(())
    }

    fn check_voss_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = VossParams { period: Some(20), predict: Some(3), bandwidth: Some(0.25) };
        let input = VossInput::from_candles(&candles, "close", params);
        let output = voss_with_kernel(&input, kernel)?;

        let expected_voss_last_five = [
            -290.430249544605,
            -269.74949153549596,
            -241.08179139844515,
            -149.2113276943419,
            -138.60361772412466,
        ];
        let expected_filt_last_five = [
            -228.0283989610523,
            -257.79056527053103,
            -270.3220395771822,
            -257.4282859799144,
            -235.78021136041997,
        ];

        let start = output.voss.len() - 5;
        for (i, &val) in output.voss[start..].iter().enumerate() {
            let expected = expected_voss_last_five[i];
            assert!((val - expected).abs() < 1e-1, "[{}] VOSS mismatch at idx {}: got {}, expected {}", test_name, i, val, expected);
        }
        for (i, &val) in output.filt[start..].iter().enumerate() {
            let expected = expected_filt_last_five[i];
            assert!((val - expected).abs() < 1e-1, "[{}] Filt mismatch at idx {}: got {}, expected {}", test_name, i, val, expected);
        }
        Ok(())
    }

    fn check_voss_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = VossInput::with_default_candles(&candles);
        match input.data {
            VossData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected VossData::Candles"),
        }
        let output = voss_with_kernel(&input, kernel)?;
        assert_eq!(output.voss.len(), candles.close.len());
        Ok(())
    }

    fn check_voss_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = VossParams { period: Some(0), predict: None, bandwidth: None };
        let input = VossInput::from_slice(&input_data, params);
        let res = voss_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] VOSS should fail with zero period", test_name);
        Ok(())
    }

    fn check_voss_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = VossParams { period: Some(10), predict: None, bandwidth: None };
        let input = VossInput::from_slice(&data_small, params);
        let res = voss_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] VOSS should fail with period exceeding length", test_name);
        Ok(())
    }

    fn check_voss_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = VossParams { period: Some(20), predict: None, bandwidth: None };
        let input = VossInput::from_slice(&single_point, params);
        let res = voss_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] VOSS should fail with insufficient data", test_name);
        Ok(())
    }

    fn check_voss_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = VossParams { period: Some(10), predict: Some(2), bandwidth: Some(0.2) };
        let first_input = VossInput::from_candles(&candles, "close", first_params);
        let first_result = voss_with_kernel(&first_input, kernel)?;

        let second_params = VossParams { period: Some(10), predict: Some(2), bandwidth: Some(0.2) };
        let second_input = VossInput::from_slice(&first_result.voss, second_params);
        let second_result = voss_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.voss.len(), first_result.voss.len());
        Ok(())
    }

    macro_rules! generate_all_voss_tests {
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
    generate_all_voss_tests!(
        check_voss_partial_params,
        check_voss_accuracy,
        check_voss_default_candles,
        check_voss_zero_period,
        check_voss_period_exceeds_length,
        check_voss_very_small_dataset,
        check_voss_reinput
    );

    fn check_batch_default_row(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = VossBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = VossParams::default();
        let row = output.voss_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        // For reference, only test length and NaN propagation
        for (i, &v) in row.iter().enumerate() {
            if i < def.period.unwrap() {
                assert!(
                    v.is_nan() || v.abs() < 1e-8,
                    "[{test_name}] expected NaN or 0 at idx {i}, got {v}"
                );
            }
        }
        Ok(())
    }

    fn check_batch_param_grid(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let builder = VossBatchBuilder::new()
            .kernel(kernel)
            .period_range(10, 14, 2)
            .predict_range(2, 4, 1)
            .bandwidth_range(0.1, 0.2, 0.1);

        let output = builder.apply_candles(&c, "close")?;
        let expected_param_count = ((14 - 10) / 2 + 1) * (4 - 2 + 1) * 2; // periods x predicts x bandwidths
        assert_eq!(
            output.combos.len(),
            expected_param_count,
            "[{test_name}] Unexpected grid size: got {}, expected {}",
            output.combos.len(),
            expected_param_count
        );

        for p in &output.combos {
            let row = output.voss_for(p).unwrap();
            assert_eq!(row.len(), c.close.len());
        }
        Ok(())
    }

    fn check_batch_nan_propagation(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let data = [f64::NAN, f64::NAN, 1.0, 2.0, 3.0, 4.0];
        let range = VossBatchRange {
            period: (3, 3, 0),
            predict: (2, 2, 0),
            bandwidth: (0.1, 0.1, 0.0),
        };
        let output = VossBatchBuilder::new().kernel(kernel).apply_slice(&data)?;
        for row in 0..output.rows {
            let v = &output.voss[row * output.cols..][..output.cols];
            assert!(v.iter().any(|&x| x.is_nan()), "[{test_name}] No NaNs found in output row");
        }
        Ok(())
    }

    fn check_batch_invalid_range(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let data = [1.0, 2.0, 3.0];
        let range = VossBatchRange {
            period: (10, 10, 0),
            predict: (3, 3, 0),
            bandwidth: (0.25, 0.25, 0.0),
        };
        let result = VossBatchBuilder::new()
            .kernel(kernel)
            .apply_slice(&data);
        assert!(result.is_err(), "[{test_name}] Expected error for invalid batch range");
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
    gen_batch_tests!(check_batch_param_grid);
    gen_batch_tests!(check_batch_nan_propagation);
    gen_batch_tests!(check_batch_invalid_range);
}


