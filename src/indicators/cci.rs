//! # Commodity Channel Index (CCI)
//!
//! Commodity Channel Index is typically calculated as:
//!
//! ```text
//! CCI_t = (price_t - SMA(price, period)) / (0.015 * MeanAbsoluteDeviation(price, period))
//! ```
//!
//! - **period**: Window size (number of data points). Defaults to 14.
//!
//! ## Errors
//! - **AllValuesNaN**: cci: All input data values are `NaN`.
//! - **InvalidPeriod**: cci: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: cci: Not enough valid data points for the requested `period`.
//!
//! ## Returns
//! - **`Ok(CciOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(CciError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

// ---- Input/Output Structs ----

#[derive(Debug, Clone)]
pub enum CciData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct CciOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct CciParams {
    pub period: Option<usize>,
}

impl Default for CciParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct CciInput<'a> {
    pub data: CciData<'a>,
    pub params: CciParams,
}

impl<'a> AsRef<[f64]> for CciInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            CciData::Slice(slice) => slice,
            CciData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

impl<'a> CciInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: CciParams) -> Self {
        Self {
            data: CciData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: CciParams) -> Self {
        Self {
            data: CciData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "hlc3", CciParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
}

// ---- Builder ----

#[derive(Copy, Clone, Debug)]
pub struct CciBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for CciBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl CciBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<CciOutput, CciError> {
        let p = CciParams {
            period: self.period,
        };
        let i = CciInput::from_candles(c, "hlc3", p);
        cci_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<CciOutput, CciError> {
        let p = CciParams {
            period: self.period,
        };
        let i = CciInput::from_slice(d, p);
        cci_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<CciStream, CciError> {
        let p = CciParams {
            period: self.period,
        };
        CciStream::try_new(p)
    }
}

// ---- Error ----

#[derive(Debug, Error)]
pub enum CciError {
    #[error("cci: All values are NaN.")]
    AllValuesNaN,
    #[error("cci: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("cci: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

// ---- Indicator API ----

#[inline]
pub fn cci(input: &CciInput) -> Result<CciOutput, CciError> {
    cci_with_kernel(input, Kernel::Auto)
}

pub fn cci_with_kernel(input: &CciInput, kernel: Kernel) -> Result<CciOutput, CciError> {
    let data: &[f64] = match &input.data {
        CciData::Candles { candles, source } => source_type(candles, source),
        CciData::Slice(sl) => sl,
    };
    let first = data.iter().position(|x| !x.is_nan()).ok_or(CciError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(CciError::InvalidPeriod { period, data_len: len });
    }
    if (len - first) < period {
        return Err(CciError::NotEnoughValidData { needed: period, valid: len - first });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let mut out = vec![f64::NAN; len];
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                cci_scalar(data, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                cci_avx2(data, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                cci_avx512(data, period, first, &mut out)
            }
            _ => unreachable!(),
        }
    }
    Ok(CciOutput { values: out })
}

// ---- Scalar + AVX2/AVX512 (Stub) ----

#[inline]
pub fn cci_scalar(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    let inv_period = 1.0 / period as f64;
    let mut sum = 0.0;
    for &val in &data[first_valid..(first_valid + period)] {
        sum += val;
    }
    let mut sma = sum * inv_period;
    let mut sum_abs = 0.0;
    for &val in &data[first_valid..(first_valid + period)] {
        sum_abs += (val - sma).abs();
    }
    let first_out = first_valid + period - 1;
    let price = data[first_out];
    out[first_out] = if sum_abs == 0.0 {
        0.0
    } else {
        (price - sma) / (0.015 * (sum_abs * inv_period))
    };

    for i in (first_out + 1)..data.len() {
        let exiting = data[i - period];
        let entering = data[i];
        sum = sum - exiting + entering;
        sma = sum * inv_period;
        sum_abs = 0.0;
        for &val in &data[(i - period + 1)..=i] {
            sum_abs += (val - sma).abs();
        }
        out[i] = if sum_abs == 0.0 {
            0.0
        } else {
            (entering - sma) / (0.015 * (sum_abs * inv_period))
        };
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn cci_avx512(
    data: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        unsafe { cci_avx512_short(data, period, first_valid, out) }
    } else {
        unsafe { cci_avx512_long(data, period, first_valid, out) }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn cci_avx2(
    data: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    cci_scalar(data, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn cci_avx512_short(
    data: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    cci_scalar(data, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn cci_avx512_long(
    data: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    cci_scalar(data, period, first_valid, out)
}

// ---- Row functions (all AVX variants point to correct level) ----

#[inline(always)]
pub unsafe fn cci_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv: f64,
    out: &mut [f64],
) {
    cci_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn cci_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv: f64,
    out: &mut [f64],
) {
    cci_avx2(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn cci_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv: f64,
    out: &mut [f64],
) {
    cci_avx512(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn cci_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv: f64,
    out: &mut [f64],
) {
    cci_avx512_short(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn cci_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv: f64,
    out: &mut [f64],
) {
    cci_avx512_long(data, period, first, out)
}

// ---- Stream ----

#[derive(Debug, Clone)]
pub struct CciStream {
    period: usize,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,
    sum: f64,
    last_sma: f64,
}

impl CciStream {
    pub fn try_new(params: CciParams) -> Result<Self, CciError> {
        let period = params.period.unwrap_or(14);
        if period == 0 {
            return Err(CciError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        Ok(Self {
            period,
            buffer: vec![f64::NAN; period],
            head: 0,
            filled: false,
            sum: 0.0,
            last_sma: 0.0,
        })
    }
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        let old = self.buffer[self.head];
        self.buffer[self.head] = value;
        self.head = (self.head + 1) % self.period;
        if !self.filled && self.head == 0 {
            self.filled = true;
        }
        if !self.filled {
            return None;
        }
        self.sum = self.sum - if old.is_nan() { 0.0 } else { old } + value;
        self.last_sma = self.sum / self.period as f64;
        let mut sum_abs = 0.0;
        for &v in &self.buffer {
            sum_abs += (v - self.last_sma).abs();
        }
        if sum_abs == 0.0 {
            Some(0.0)
        } else {
            Some((value - self.last_sma) / (0.015 * (sum_abs / self.period as f64)))
        }
    }
}

// ---- Batch/Range ----

#[derive(Clone, Debug)]
pub struct CciBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for CciBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 200, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CciBatchBuilder {
    range: CciBatchRange,
    kernel: Kernel,
}

impl CciBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<CciBatchOutput, CciError> {
        cci_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<CciBatchOutput, CciError> {
        CciBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<CciBatchOutput, CciError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<CciBatchOutput, CciError> {
        CciBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "hlc3")
    }
}

#[derive(Clone, Debug)]
pub struct CciBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<CciParams>,
    pub rows: usize,
    pub cols: usize,
}
impl CciBatchOutput {
    pub fn row_for_params(&self, p: &CciParams) -> Option<usize> {
        self.combos.iter().position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
    }
    pub fn values_for(&self, p: &CciParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &CciBatchRange) -> Vec<CciParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(CciParams {
            period: Some(p),
        });
    }
    out
}

pub fn cci_batch_with_kernel(
    data: &[f64],
    sweep: &CciBatchRange,
    k: Kernel,
) -> Result<CciBatchOutput, CciError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(CciError::InvalidPeriod {
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
    cci_batch_par_slice(data, sweep, simd)
}

#[inline(always)]
pub fn cci_batch_slice(
    data: &[f64],
    sweep: &CciBatchRange,
    kern: Kernel,
) -> Result<CciBatchOutput, CciError> {
    cci_batch_inner(data, sweep, kern, false)
}
#[inline(always)]
pub fn cci_batch_par_slice(
    data: &[f64],
    sweep: &CciBatchRange,
    kern: Kernel,
) -> Result<CciBatchOutput, CciError> {
    cci_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn cci_batch_inner(
    data: &[f64],
    sweep: &CciBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<CciBatchOutput, CciError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(CciError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(CciError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(CciError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();
    let mut values = vec![f64::NAN; rows * cols];
    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        match kern {
            Kernel::Scalar => cci_row_scalar(data, first, period, 0, std::ptr::null(), 0.0, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => cci_row_avx2(data, first, period, 0, std::ptr::null(), 0.0, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => cci_row_avx512(data, first, period, 0, std::ptr::null(), 0.0, out_row),
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
    Ok(CciBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

// ---- Tests ----

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;
    use paste::paste;

    fn check_cci_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = CciParams { period: None };
        let input_default = CciInput::from_candles(&candles, "close", default_params);
        let output_default = cci_with_kernel(&input_default, kernel)?;
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_20 = CciParams { period: Some(20) };
        let input_20 = CciInput::from_candles(&candles, "hl2", params_20);
        let output_20 = cci_with_kernel(&input_20, kernel)?;
        assert_eq!(output_20.values.len(), candles.close.len());

        let params_custom = CciParams { period: Some(9) };
        let input_custom = CciInput::from_candles(&candles, "hlc3", params_custom);
        let output_custom = cci_with_kernel(&input_custom, kernel)?;
        assert_eq!(output_custom.values.len(), candles.close.len());
        Ok(())
    }

    fn check_cci_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = CciInput::with_default_candles(&candles);
        let cci_result = cci_with_kernel(&input, kernel)?;
        assert_eq!(cci_result.values.len(), candles.close.len());

        let expected_last_five_cci = [
            -51.55252564125841,
            -43.50326506381541,
            -64.05117302269149,
            -39.05150631680948,
            -152.50523930896998,
        ];

        let start_idx = cci_result.values.len() - 5;
        let last_five_cci = &cci_result.values[start_idx..];
        for (i, &value) in last_five_cci.iter().enumerate() {
            let expected = expected_last_five_cci[i];
            assert!(
                (value - expected).abs() < 1e-6,
                "[{}] CCI mismatch at last five index {}: expected {}, got {}",
                test_name,
                i,
                expected,
                value
            );
        }
        let period: usize = input.get_period();
        for i in 0..(period - 1) {
            assert!(
                cci_result.values[i].is_nan(),
                "Expected NaN at index {} for initial period warm-up",
                i
            );
        }
        Ok(())
    }

    fn check_cci_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = CciInput::with_default_candles(&candles);

        match input.data {
            CciData::Candles { source, .. } => {
                assert_eq!(source, "hlc3", "Expected default source to be 'hlc3'");
            }
            _ => panic!("Expected CciData::Candles variant"),
        }
        let output = cci_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_cci_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = CciParams { period: Some(0) };
        let input = CciInput::from_slice(&input_data, params);
        let res = cci_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] CCI should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_cci_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = CciParams { period: Some(10) };
        let input = CciInput::from_slice(&data_small, params);
        let res = cci_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] CCI should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_cci_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = CciParams { period: Some(9) };
        let input = CciInput::from_slice(&single_point, params);
        let res = cci_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] CCI should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_cci_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = CciParams { period: Some(14) };
        let first_input = CciInput::from_candles(&candles, "close", first_params);
        let first_result = cci_with_kernel(&first_input, kernel)?;

        let second_params = CciParams { period: Some(14) };
        let second_input = CciInput::from_slice(&first_result.values, second_params);
        let second_result = cci_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());
        if second_result.values.len() > 28 {
            for i in 28..second_result.values.len() {
                assert!(
                    !second_result.values[i].is_nan(),
                    "Expected no NaN after index 28, found NaN at index {}",
                    i
                );
            }
        }
        Ok(())
    }

    fn check_cci_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = CciInput::from_candles(&candles, "close", CciParams { period: Some(14) });
        let res = cci_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 240 {
            for (i, &val) in res.values[240..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test_name,
                    240 + i
                );
            }
        }
        Ok(())
    }

    fn check_cci_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let period = 14;
        let input = CciInput::from_candles(
            &candles,
            "close",
            CciParams {
                period: Some(period),
            },
        );
        let batch_output = cci_with_kernel(&input, kernel)?.values;

        let mut stream = CciStream::try_new(CciParams {
            period: Some(period),
        })?;

        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(cci_val) => stream_values.push(cci_val),
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
                "[{}] CCI streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_cci_tests {
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

    generate_all_cci_tests!(
        check_cci_partial_params,
        check_cci_accuracy,
        check_cci_default_candles,
        check_cci_zero_period,
        check_cci_period_exceeds_length,
        check_cci_very_small_dataset,
        check_cci_reinput,
        check_cci_nan_handling,
        check_cci_streaming
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = CciBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = CciParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());

        let expected = [
            -51.55252564125841,
            -43.50326506381541,
            -64.05117302269149,
            -39.05150631680948,
            -152.50523930896998,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-6,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
            );
        }
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
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]),
                                     Kernel::Auto);
                }
            }
        };
    }
    gen_batch_tests!(check_batch_default_row);
}
