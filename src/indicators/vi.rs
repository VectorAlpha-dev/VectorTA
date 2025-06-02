//! # Vortex Indicator (VI)
//!
//! Computes the positive (VI+) and negative (VI-) vortex indicators based on a specified period.
//! Supports batch computation, builder pattern, parameter sweeps, AVX2/AVX512 feature stubs, and streaming API.
//!
//! ## Parameters
//! - **period**: Lookback window size (default: 14).
//!
//! ## Errors
//! - **EmptyData**: vi: Input slices are empty.
//! - **InvalidPeriod**: vi: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: vi: Not enough valid data points for the requested `period`.
//! - **AllValuesNaN**: vi: All input data values are `NaN`.
//!
//! ## Returns
//! - **`Ok(ViOutput)`** on success, with `.plus` and `.minus` of length matching the input.
//! - **`Err(ViError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use rayon::prelude::*;
use thiserror::Error;
use std::error::Error;

#[derive(Debug, Clone)]
pub enum ViData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct ViOutput {
    pub plus: Vec<f64>,
    pub minus: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct ViParams {
    pub period: Option<usize>,
}

impl Default for ViParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct ViInput<'a> {
    pub data: ViData<'a>,
    pub params: ViParams,
}

impl<'a> ViInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: ViParams) -> Self {
        Self {
            data: ViData::Candles { candles },
            params,
        }
    }
    #[inline]
    pub fn from_slices(high: &'a [f64], low: &'a [f64], close: &'a [f64], params: ViParams) -> Self {
        Self {
            data: ViData::Slices { high, low, close },
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: ViData::Candles { candles },
            params: ViParams::default(),
        }
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ViBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for ViBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl ViBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<ViOutput, ViError> {
        let p = ViParams { period: self.period };
        let i = ViInput::from_candles(c, p);
        vi_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(self, high: &[f64], low: &[f64], close: &[f64]) -> Result<ViOutput, ViError> {
        let p = ViParams { period: self.period };
        let i = ViInput::from_slices(high, low, close, p);
        vi_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<ViStream, ViError> {
        let p = ViParams { period: self.period };
        ViStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum ViError {
    #[error("vi: Empty data provided.")]
    EmptyData,
    #[error("vi: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("vi: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("vi: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn vi(input: &ViInput) -> Result<ViOutput, ViError> {
    vi_with_kernel(input, Kernel::Auto)
}

pub fn vi_with_kernel(input: &ViInput, kernel: Kernel) -> Result<ViOutput, ViError> {
    let (high, low, close) = match &input.data {
        ViData::Candles { candles } => (
            source_type(candles, "high"),
            source_type(candles, "low"),
            source_type(candles, "close"),
        ),
        ViData::Slices { high, low, close } => (*high, *low, *close),
    };
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(ViError::EmptyData);
    }
    let length = high.len();
    if length != low.len() || length != close.len() {
        return Err(ViError::EmptyData);
    }
    let period = input.get_period();
    if period == 0 || period > length {
        return Err(ViError::InvalidPeriod { period, data_len: length });
    }
    let first_valid_idx =
        (0..length).find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan());
    let first = match first_valid_idx {
        Some(idx) => idx,
        None => return Err(ViError::AllValuesNaN),
    };
    if (length - first) < period {
        return Err(ViError::NotEnoughValidData { needed: period, valid: length - first });
    }
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    let mut plus = vec![f64::NAN; length];
    let mut minus = vec![f64::NAN; length];
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                vi_scalar(high, low, close, period, first, &mut plus, &mut minus)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                vi_avx2(high, low, close, period, first, &mut plus, &mut minus)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                vi_avx512(high, low, close, period, first, &mut plus, &mut minus)
            }
            _ => unreachable!(),
        }
    }
    Ok(ViOutput { plus, minus })
}

#[inline]
pub unsafe fn vi_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first: usize,
    plus: &mut [f64],
    minus: &mut [f64],
) {
    let length = high.len();
    let mut tr = vec![0.0; length];
    let mut vp = vec![0.0; length];
    let mut vm = vec![0.0; length];
    tr[first] = high[first] - low[first];
    for i in (first + 1)..length {
        tr[i] = (high[i] - low[i])
            .max((high[i] - close[i - 1]).abs())
            .max((low[i] - close[i - 1]).abs());
        vp[i] = (high[i] - low[i - 1]).abs();
        vm[i] = (low[i] - high[i - 1]).abs();
    }
    let mut sum_tr = 0.0;
    let mut sum_vp = 0.0;
    let mut sum_vm = 0.0;
    for i in first..(first + period) {
        sum_tr += tr[i];
        sum_vp += vp[i];
        sum_vm += vm[i];
    }
    plus[first + period - 1] = sum_vp / sum_tr;
    minus[first + period - 1] = sum_vm / sum_tr;
    for i in (first + period)..length {
        sum_tr += tr[i] - tr[i - period];
        sum_vp += vp[i] - vp[i - period];
        sum_vm += vm[i] - vm[i - period];
        plus[i] = sum_vp / sum_tr;
        minus[i] = sum_vm / sum_tr;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vi_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first: usize,
    plus: &mut [f64],
    minus: &mut [f64],
) {
    vi_scalar(high, low, close, period, first, plus, minus);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vi_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first: usize,
    plus: &mut [f64],
    minus: &mut [f64],
) {
    if period <= 32 {
        vi_avx512_short(high, low, close, period, first, plus, minus);
    } else {
        vi_avx512_long(high, low, close, period, first, plus, minus);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vi_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first: usize,
    plus: &mut [f64],
    minus: &mut [f64],
) {
    vi_scalar(high, low, close, period, first, plus, minus);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vi_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    first: usize,
    plus: &mut [f64],
    minus: &mut [f64],
) {
    vi_scalar(high, low, close, period, first, plus, minus);
}

#[derive(Debug, Clone)]
pub struct ViStream {
    period: usize,
    tr: Vec<f64>,
    vp: Vec<f64>,
    vm: Vec<f64>,
    idx: usize,
    filled: bool,
    sum_tr: f64,
    sum_vp: f64,
    sum_vm: f64,
}

impl ViStream {
    pub fn try_new(params: ViParams) -> Result<Self, ViError> {
        let period = params.period.unwrap_or(14);
        if period == 0 {
            return Err(ViError::InvalidPeriod { period, data_len: 0 });
        }
        Ok(Self {
            period,
            tr: vec![0.0; period],
            vp: vec![0.0; period],
            vm: vec![0.0; period],
            idx: 0,
            filled: false,
            sum_tr: 0.0,
            sum_vp: 0.0,
            sum_vm: 0.0,
        })
    }
    pub fn update(&mut self, high: f64, low: f64, close: f64, prev_low: f64, prev_high: f64, prev_close: f64) -> Option<(f64, f64)> {
        let i = self.idx % self.period;
        let tr = (high - low)
            .max((high - prev_close).abs())
            .max((low - prev_close).abs());
        let vp = (high - prev_low).abs();
        let vm = (low - prev_high).abs();
        self.sum_tr += tr - self.tr[i];
        self.sum_vp += vp - self.vp[i];
        self.sum_vm += vm - self.vm[i];
        self.tr[i] = tr;
        self.vp[i] = vp;
        self.vm[i] = vm;
        self.idx += 1;
        if !self.filled && self.idx >= self.period {
            self.filled = true;
        }
        if self.filled {
            Some((self.sum_vp / self.sum_tr, self.sum_vm / self.sum_tr))
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub struct ViBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for ViBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 14, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ViBatchBuilder {
    range: ViBatchRange,
    kernel: Kernel,
}

impl ViBatchBuilder {
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
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<ViBatchOutput, ViError> {
        vi_batch_with_kernel(high, low, close, &self.range, self.kernel)
    }
    pub fn apply_candles(self, c: &Candles) -> Result<ViBatchOutput, ViError> {
        let high = source_type(c, "high");
        let low = source_type(c, "low");
        let close = source_type(c, "close");
        self.apply_slices(high, low, close)
    }
}

#[derive(Clone, Debug)]
pub struct ViBatchOutput {
    pub plus: Vec<f64>,
    pub minus: Vec<f64>,
    pub combos: Vec<ViParams>,
    pub rows: usize,
    pub cols: usize,
}
impl ViBatchOutput {
    pub fn row_for_params(&self, p: &ViParams) -> Option<usize> {
        self.combos.iter().position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
    }
    pub fn plus_for(&self, p: &ViParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.plus[start..start + self.cols]
        })
    }
    pub fn minus_for(&self, p: &ViParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.minus[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid_vi(r: &ViBatchRange) -> Vec<ViParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(ViParams { period: Some(p) });
    }
    out
}

pub fn vi_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &ViBatchRange,
    k: Kernel,
) -> Result<ViBatchOutput, ViError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(ViError::InvalidPeriod { period: 0, data_len: 0 });
        }
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    vi_batch_par_slice(high, low, close, sweep, simd)
}

pub fn vi_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &ViBatchRange,
    kern: Kernel,
) -> Result<ViBatchOutput, ViError> {
    vi_batch_inner(high, low, close, sweep, kern, false)
}
pub fn vi_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &ViBatchRange,
    kern: Kernel,
) -> Result<ViBatchOutput, ViError> {
    vi_batch_inner(high, low, close, sweep, kern, true)
}
#[inline(always)]
fn vi_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &ViBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<ViBatchOutput, ViError> {
    let combos = expand_grid_vi(sweep);
    if combos.is_empty() {
        return Err(ViError::InvalidPeriod { period: 0, data_len: 0 });
    }
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(ViError::EmptyData);
    }
    let first = (0..high.len())
        .find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
        .ok_or(ViError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if high.len() - first < max_p {
        return Err(ViError::NotEnoughValidData { needed: max_p, valid: high.len() - first });
    }
    let rows = combos.len();
    let cols = high.len();
    let mut plus = vec![f64::NAN; rows * cols];
    let mut minus = vec![f64::NAN; rows * cols];
    let do_row = |row: usize, plus_row: &mut [f64], minus_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        match kern {
            Kernel::Scalar => vi_row_scalar(high, low, close, first, period, plus_row, minus_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => vi_row_avx2(high, low, close, first, period, plus_row, minus_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => vi_row_avx512(high, low, close, first, period, plus_row, minus_row),
            _ => unreachable!(),
        }
    };
    if parallel {
        plus.par_chunks_mut(cols)
            .zip(minus.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, (p, m))| do_row(row, p, m));
    } else {
        for ((row, p), m) in plus.chunks_mut(cols).enumerate().zip(minus.chunks_mut(cols)) {
            do_row(row, p, m);
        }
    }
    Ok(ViBatchOutput { plus, minus, combos, rows, cols })
}

#[inline(always)]
unsafe fn vi_row_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    plus: &mut [f64],
    minus: &mut [f64],
) {
    vi_scalar(high, low, close, period, first, plus, minus);
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn vi_row_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    plus: &mut [f64],
    minus: &mut [f64],
) {
    vi_scalar(high, low, close, period, first, plus, minus);
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn vi_row_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    plus: &mut [f64],
    minus: &mut [f64],
) {
    if period <= 32 {
        vi_row_avx512_short(high, low, close, first, period, plus, minus);
    } else {
        vi_row_avx512_long(high, low, close, first, period, plus, minus);
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn vi_row_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    plus: &mut [f64],
    minus: &mut [f64],
) {
    vi_scalar(high, low, close, period, first, plus, minus);
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn vi_row_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    period: usize,
    plus: &mut [f64],
    minus: &mut [f64],
) {
    vi_scalar(high, low, close, period, first, plus, minus);
}
#[inline(always)]
fn expand_grid(_r: &ViBatchRange) -> Vec<ViParams> {
    // For full parity with ALMA, but VI has only period for batch
    expand_grid_vi(_r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_vi_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = ViParams { period: None };
        let input = ViInput::from_candles(&candles, default_params);
        let output = vi_with_kernel(&input, kernel)?;
        assert_eq!(output.plus.len(), candles.close.len());
        assert_eq!(output.minus.len(), candles.close.len());
        Ok(())
    }
    fn check_vi_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = ViInput::from_candles(&candles, ViParams::default());
        let result = vi_with_kernel(&input, kernel)?;
        let expected_last_five_plus = [
            0.9970238095238095,
            0.9871071716357775,
            0.9464453759945247,
            0.890897412369242,
            0.9206478557604156,
        ];
        let expected_last_five_minus = [
            1.0097117794486214,
            1.04174053182917,
            1.1152365471811105,
            1.181684712791338,
            1.1894672506875827,
        ];
        let n = result.plus.len();
        let plus_slice = &result.plus[n - 5..];
        let minus_slice = &result.minus[n - 5..];
        for (i, &val) in plus_slice.iter().enumerate() {
            let expected = expected_last_five_plus[i];
            assert!(
                (val - expected).abs() < 1e-8,
                "[{}] VI+ mismatch at idx {}: got {}, expected {}",
                test_name,
                i,
                val,
                expected
            );
        }
        for (i, &val) in minus_slice.iter().enumerate() {
            let expected = expected_last_five_minus[i];
            assert!(
                (val - expected).abs() < 1e-8,
                "[{}] VI- mismatch at idx {}: got {}, expected {}",
                test_name,
                i,
                val,
                expected
            );
        }
        Ok(())
    }
    fn check_vi_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = ViInput::with_default_candles(&candles);
        let output = vi_with_kernel(&input, kernel)?;
        assert_eq!(output.plus.len(), candles.close.len());
        assert_eq!(output.minus.len(), candles.close.len());
        Ok(())
    }
    fn check_vi_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = ViParams { period: Some(0) };
        let input = ViInput::from_slices(&input_data, &input_data, &input_data, params);
        let res = vi_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] VI should fail with zero period",
            test_name
        );
        Ok(())
    }
    fn check_vi_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = ViParams { period: Some(10) };
        let input = ViInput::from_slices(&data_small, &data_small, &data_small, params);
        let res = vi_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] VI should fail with period exceeding length",
            test_name
        );
        Ok(())
    }
    fn check_vi_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = ViParams { period: Some(14) };
        let input = ViInput::from_slices(&single_point, &single_point, &single_point, params);
        let res = vi_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] VI should fail with insufficient data",
            test_name
        );
        Ok(())
    }
    fn check_vi_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = ViInput::from_candles(&candles, ViParams::default());
        let res = vi_with_kernel(&input, kernel)?;
        assert_eq!(res.plus.len(), candles.close.len());
        if res.plus.len() > 20 {
            for (i, &val) in res.plus[20..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test_name,
                    20 + i
                );
            }
        }
        Ok(())
    }
    macro_rules! generate_all_vi_tests {
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
    generate_all_vi_tests!(
        check_vi_partial_params,
        check_vi_accuracy,
        check_vi_default_candles,
        check_vi_zero_period,
        check_vi_period_exceeds_length,
        check_vi_very_small_dataset,
        check_vi_nan_handling
    );
    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = ViBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c)?;
        let def = ViParams::default();
        let row = output.plus_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
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
