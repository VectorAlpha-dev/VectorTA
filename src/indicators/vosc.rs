//! # Volume Oscillator (VOSC)
//!
//! Measures changes in volume trends using two moving averages (short and long).
//!
//! ## Formula
//! ```ignore
//! vosc = 100 * ((short_avg - long_avg) / long_avg)
//! ```
//!
//! ## Parameters
//! - **short_period**: The short window size. Defaults to 2.
//! - **long_period**: The long window size. Defaults to 5.
//!
//! ## Errors
//! - **EmptyData**: vosc: Input data slice is empty.
//! - **InvalidShortPeriod**: vosc: `short_period` is zero or exceeds the data length.
//! - **InvalidLongPeriod**: vosc: `long_period` is zero or exceeds the data length.
//! - **ShortPeriodGreaterThanLongPeriod**: vosc: `short_period` is greater than `long_period`.
//! - **NotEnoughValidData**: vosc: Fewer than `long_period` valid data points after the first valid index.
//! - **AllValuesNaN**: vosc: All input data values are `NaN`.
//!
//! ## Returns
//! - **`Ok(VoscOutput)`** on success, containing a `Vec<f64>` matching input length.
//! - **`Err(VoscError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use rayon::prelude::*;
use thiserror::Error;
use std::convert::AsRef;

impl<'a> AsRef<[f64]> for VoscInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            VoscData::Slice(slice) => slice,
            VoscData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum VoscData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct VoscOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct VoscParams {
    pub short_period: Option<usize>,
    pub long_period: Option<usize>,
}

impl Default for VoscParams {
    fn default() -> Self {
        Self {
            short_period: Some(2),
            long_period: Some(5),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VoscInput<'a> {
    pub data: VoscData<'a>,
    pub params: VoscParams,
}

impl<'a> VoscInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: VoscParams) -> Self {
        Self {
            data: VoscData::Candles { candles: c, source: s },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: VoscParams) -> Self {
        Self {
            data: VoscData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "volume", VoscParams::default())
    }
    #[inline]
    pub fn get_short_period(&self) -> usize {
        self.params.short_period.unwrap_or(2)
    }
    #[inline]
    pub fn get_long_period(&self) -> usize {
        self.params.long_period.unwrap_or(5)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct VoscBuilder {
    short_period: Option<usize>,
    long_period: Option<usize>,
    kernel: Kernel,
}

impl Default for VoscBuilder {
    fn default() -> Self {
        Self {
            short_period: None,
            long_period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl VoscBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn short_period(mut self, n: usize) -> Self {
        self.short_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn long_period(mut self, n: usize) -> Self {
        self.long_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<VoscOutput, VoscError> {
        let p = VoscParams {
            short_period: self.short_period,
            long_period: self.long_period,
        };
        let i = VoscInput::from_candles(c, "volume", p);
        vosc_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<VoscOutput, VoscError> {
        let p = VoscParams {
            short_period: self.short_period,
            long_period: self.long_period,
        };
        let i = VoscInput::from_slice(d, p);
        vosc_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<VoscStream, VoscError> {
        let p = VoscParams {
            short_period: self.short_period,
            long_period: self.long_period,
        };
        VoscStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum VoscError {
    #[error("vosc: Empty data provided for VOSC.")]
    EmptyData,
    #[error("vosc: Invalid short period: short_period = {period}, data length = {data_len}")]
    InvalidShortPeriod { period: usize, data_len: usize },
    #[error("vosc: Invalid long period: long_period = {period}, data length = {data_len}")]
    InvalidLongPeriod { period: usize, data_len: usize },
    #[error("vosc: short_period > long_period")]
    ShortPeriodGreaterThanLongPeriod,
    #[error("vosc: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("vosc: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn vosc(input: &VoscInput) -> Result<VoscOutput, VoscError> {
    vosc_with_kernel(input, Kernel::Auto)
}

pub fn vosc_with_kernel(input: &VoscInput, kernel: Kernel) -> Result<VoscOutput, VoscError> {
    let data: &[f64] = match &input.data {
        VoscData::Candles { candles, source } => source_type(candles, source),
        VoscData::Slice(sl) => sl,
    };

    if data.is_empty() {
        return Err(VoscError::EmptyData);
    }

    let short_period = input.get_short_period();
    let long_period = input.get_long_period();

    if short_period == 0 || short_period > data.len() {
        return Err(VoscError::InvalidShortPeriod {
            period: short_period,
            data_len: data.len(),
        });
    }
    if long_period == 0 || long_period > data.len() {
        return Err(VoscError::InvalidLongPeriod {
            period: long_period,
            data_len: data.len(),
        });
    }
    if short_period > long_period {
        return Err(VoscError::ShortPeriodGreaterThanLongPeriod);
    }

    let first = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(VoscError::AllValuesNaN),
    };
    if (data.len() - first) < long_period {
        return Err(VoscError::NotEnoughValidData {
            needed: long_period,
            valid: data.len() - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let mut out = vec![f64::NAN; data.len()];

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                vosc_scalar(data, short_period, long_period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                vosc_avx2(data, short_period, long_period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                vosc_avx512(data, short_period, long_period, first, &mut out)
            }
            _ => unreachable!(),
        }
    }

    Ok(VoscOutput { values: out })
}

#[inline]
pub fn vosc_scalar(
    data: &[f64],
    short_period: usize,
    long_period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    let mut short_sum = 0.0;
    let mut long_sum = 0.0;
    for i in first_valid..(first_valid + long_period) {
        let v = data[i];
        if i >= (first_valid + long_period - short_period) {
            short_sum += v;
        }
        long_sum += v;
    }

    let short_div = 1.0 / (short_period as f64);
    let long_div = 1.0 / (long_period as f64);
    let init_idx = first_valid + long_period - 1;
    let mut savg = short_sum * short_div;
    let mut lavg = long_sum * long_div;
    out[init_idx] = 100.0 * (savg - lavg) / lavg;

    for i in (first_valid + long_period)..data.len() {
        short_sum += data[i];
        short_sum -= data[i - short_period];
        long_sum += data[i];
        long_sum -= data[i - long_period];

        savg = short_sum * short_div;
        lavg = long_sum * long_div;
        out[i] = 100.0 * (savg - lavg) / lavg;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn vosc_avx512(
    data: &[f64],
    short_period: usize,
    long_period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    vosc_scalar(data, short_period, long_period, first_valid, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn vosc_avx2(
    data: &[f64],
    short_period: usize,
    long_period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    vosc_scalar(data, short_period, long_period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vosc_avx512_short(
    data: &[f64],
    short_period: usize,
    long_period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    vosc_scalar(data, short_period, long_period, first_valid, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vosc_avx512_long(
    data: &[f64],
    short_period: usize,
    long_period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    vosc_scalar(data, short_period, long_period, first_valid, out)
}

#[inline]
pub fn vosc_row_scalar(
    data: &[f64],
    first: usize,
    short_period: usize,
    long_period: usize,
    out: &mut [f64],
) {
    vosc_scalar(data, short_period, long_period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vosc_row_avx2(
    data: &[f64],
    first: usize,
    short_period: usize,
    long_period: usize,
    out: &mut [f64],
) {
    vosc_scalar(data, short_period, long_period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vosc_row_avx512(
    data: &[f64],
    first: usize,
    short_period: usize,
    long_period: usize,
    out: &mut [f64],
) {
    vosc_scalar(data, short_period, long_period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vosc_row_avx512_short(
    data: &[f64],
    first: usize,
    short_period: usize,
    long_period: usize,
    out: &mut [f64],
) {
    vosc_scalar(data, short_period, long_period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vosc_row_avx512_long(
    data: &[f64],
    first: usize,
    short_period: usize,
    long_period: usize,
    out: &mut [f64],
) {
    vosc_scalar(data, short_period, long_period, first, out)
}

#[derive(Debug, Clone)]
pub struct VoscStream {
    short_period: usize,
    long_period: usize,
    short_buf: Vec<f64>,
    long_buf: Vec<f64>,
    short_head: usize,
    long_head: usize,
    short_filled: bool,
    long_filled: bool,
}

impl VoscStream {
    pub fn try_new(params: VoscParams) -> Result<Self, VoscError> {
        let short_period = params.short_period.unwrap_or(2);
        let long_period = params.long_period.unwrap_or(5);
        if short_period == 0 {
            return Err(VoscError::InvalidShortPeriod {
                period: short_period,
                data_len: 0,
            });
        }
        if long_period == 0 {
            return Err(VoscError::InvalidLongPeriod {
                period: long_period,
                data_len: 0,
            });
        }
        if short_period > long_period {
            return Err(VoscError::ShortPeriodGreaterThanLongPeriod);
        }
        Ok(Self {
            short_period,
            long_period,
            short_buf: vec![f64::NAN; short_period],
            long_buf: vec![f64::NAN; long_period],
            short_head: 0,
            long_head: 0,
            short_filled: false,
            long_filled: false,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.short_buf[self.short_head] = value;
        self.long_buf[self.long_head] = value;

        self.short_head = (self.short_head + 1) % self.short_period;
        self.long_head = (self.long_head + 1) % self.long_period;

        if !self.short_filled && self.short_head == 0 {
            self.short_filled = true;
        }
        if !self.long_filled && self.long_head == 0 {
            self.long_filled = true;
        }
        if !self.short_filled || !self.long_filled {
            return None;
        }
        let short_avg = self.short_buf.iter().copied().sum::<f64>() / self.short_period as f64;
        let long_avg = self.long_buf.iter().copied().sum::<f64>() / self.long_period as f64;
        Some(100.0 * (short_avg - long_avg) / long_avg)
    }
}

#[derive(Clone, Debug)]
pub struct VoscBatchRange {
    pub short_period: (usize, usize, usize),
    pub long_period: (usize, usize, usize),
}

impl Default for VoscBatchRange {
    fn default() -> Self {
        Self {
            short_period: (2, 10, 1),
            long_period: (5, 20, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct VoscBatchBuilder {
    range: VoscBatchRange,
    kernel: Kernel,
}

impl VoscBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn short_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.short_period = (start, end, step);
        self
    }
    #[inline]
    pub fn short_period_static(mut self, n: usize) -> Self {
        self.range.short_period = (n, n, 0);
        self
    }
    #[inline]
    pub fn long_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.long_period = (start, end, step);
        self
    }
    #[inline]
    pub fn long_period_static(mut self, n: usize) -> Self {
        self.range.long_period = (n, n, 0);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<VoscBatchOutput, VoscError> {
        vosc_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<VoscBatchOutput, VoscError> {
        VoscBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<VoscBatchOutput, VoscError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<VoscBatchOutput, VoscError> {
        VoscBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "volume")
    }
}

pub fn vosc_batch_with_kernel(
    data: &[f64],
    sweep: &VoscBatchRange,
    k: Kernel,
) -> Result<VoscBatchOutput, VoscError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(VoscError::InvalidLongPeriod {
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
    vosc_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct VoscBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<VoscParams>,
    pub rows: usize,
    pub cols: usize,
}
impl VoscBatchOutput {
    pub fn row_for_params(&self, p: &VoscParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.short_period.unwrap_or(2) == p.short_period.unwrap_or(2)
                && c.long_period.unwrap_or(5) == p.long_period.unwrap_or(5)
        })
    }

    pub fn values_for(&self, p: &VoscParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &VoscBatchRange) -> Vec<VoscParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let shorts = axis_usize(r.short_period);
    let longs = axis_usize(r.long_period);
    let mut out = Vec::with_capacity(shorts.len() * longs.len());
    for &s in &shorts {
        for &l in &longs {
            if s <= l {
                out.push(VoscParams {
                    short_period: Some(s),
                    long_period: Some(l),
                });
            }
        }
    }
    out
}

#[inline(always)]
pub fn vosc_batch_slice(
    data: &[f64],
    sweep: &VoscBatchRange,
    kern: Kernel,
) -> Result<VoscBatchOutput, VoscError> {
    vosc_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn vosc_batch_par_slice(
    data: &[f64],
    sweep: &VoscBatchRange,
    kern: Kernel,
) -> Result<VoscBatchOutput, VoscError> {
    vosc_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn vosc_batch_inner(
    data: &[f64],
    sweep: &VoscBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<VoscBatchOutput, VoscError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(VoscError::InvalidLongPeriod {
            period: 0,
            data_len: 0,
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(VoscError::AllValuesNaN)?;
    let max_long = combos.iter().map(|c| c.long_period.unwrap()).max().unwrap();
    if data.len() - first < max_long {
        return Err(VoscError::NotEnoughValidData {
            needed: max_long,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let mut values = vec![f64::NAN; rows * cols];
    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let short = combos[row].short_period.unwrap();
        let long = combos[row].long_period.unwrap();
        match kern {
            Kernel::Scalar => vosc_row_scalar(data, first, short, long, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => vosc_row_avx2(data, first, short, long, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => vosc_row_avx512(data, first, short, long, out_row),
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

    Ok(VoscBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_vosc_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let volume = candles
            .select_candle_field("volume")
            .expect("Failed to extract volume data");
        let params = VoscParams {
            short_period: Some(2),
            long_period: Some(5),
        };
        let input = VoscInput::from_candles(&candles, "volume", params);
        let vosc_result = vosc_with_kernel(&input, kernel)?;

        assert_eq!(
            vosc_result.values.len(),
            volume.len(),
            "VOSC length mismatch"
        );
        let expected_last_five_vosc = [
            -39.478510754298895,
            -25.886077312645188,
            -21.155087549723756,
            -12.36093768813373,
            48.70809369473075,
        ];
        let start_index = vosc_result.values.len() - 5;
        let result_last_five_vosc = &vosc_result.values[start_index..];
        for (i, &value) in result_last_five_vosc.iter().enumerate() {
            let expected_value = expected_last_five_vosc[i];
            assert!(
                (value - expected_value).abs() < 1e-1,
                "VOSC mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }
        for i in 0..(5 - 1) {
            assert!(vosc_result.values[i].is_nan());
        }

        let default_input = VoscInput::with_default_candles(&candles);
        let default_vosc_result = vosc_with_kernel(&default_input, kernel)?;
        assert_eq!(default_vosc_result.values.len(), volume.len());
        Ok(())
    }

    fn check_vosc_zero_period(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let input_data = [10.0, 20.0, 30.0];
        let params = VoscParams {
            short_period: Some(0),
            long_period: Some(5),
        };
        let input = VoscInput::from_slice(&input_data, params);
        let res = vosc_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] VOSC should fail with zero short_period", test);
        Ok(())
    }

    fn check_vosc_short_gt_long(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let input_data = [10.0, 20.0, 30.0, 40.0, 50.0];
        let params = VoscParams {
            short_period: Some(5),
            long_period: Some(2),
        };
        let input = VoscInput::from_slice(&input_data, params);
        let res = vosc_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] VOSC should fail when short_period > long_period", test);
        Ok(())
    }

    fn check_vosc_not_enough_valid(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let data = [f64::NAN, f64::NAN, 1.0, 2.0, 3.0];
        let params = VoscParams {
            short_period: Some(2),
            long_period: Some(5),
        };
        let input = VoscInput::from_slice(&data, params);
        let res = vosc_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] VOSC should fail with not enough valid data", test);
        Ok(())
    }

    fn check_vosc_all_nan(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let data = [f64::NAN, f64::NAN, f64::NAN];
        let params = VoscParams {
            short_period: Some(2),
            long_period: Some(3),
        };
        let input = VoscInput::from_slice(&data, params);
        let res = vosc_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] VOSC should fail with all NaN", test);
        Ok(())
    }

    fn check_vosc_streaming(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let volume = candles
            .select_candle_field("volume")
            .expect("Failed to extract volume data");
        let short_period = 2;
        let long_period = 5;
        let input = VoscInput::from_candles(
            &candles,
            "volume",
            VoscParams {
                short_period: Some(short_period),
                long_period: Some(long_period),
            },
        );
        let batch_output = vosc_with_kernel(&input, kernel)?.values;

        let mut stream = VoscStream::try_new(VoscParams {
            short_period: Some(short_period),
            long_period: Some(long_period),
        })?;
        let mut stream_values = Vec::with_capacity(volume.len());
        for &v in volume {
            match stream.update(v) {
                Some(val) => stream_values.push(val),
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
                diff < 1e-9,
                "[{}] VOSC streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_vosc_tests {
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

    generate_all_vosc_tests!(
        check_vosc_accuracy,
        check_vosc_zero_period,
        check_vosc_short_gt_long,
        check_vosc_not_enough_valid,
        check_vosc_all_nan,
        check_vosc_streaming
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = VoscBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "volume")?;
        let def = VoscParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.volume.len());

        let expected = [
            -39.478510754298895,
            -25.886077312645188,
            -21.155087549723756,
            -12.36093768813373,
            48.70809369473075,
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
