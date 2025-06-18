//! # Simple Moving Average (SMA)
//!
//! The most basic form of moving average, summing the last `period` points
//! and dividing by `period`. Useful for smoothing data and trend detection.
//!
//! ## Parameters
//! - **period**: Window size (number of data points).
//!
//! ## Errors
//! - **EmptyData**: sma: Input data slice is empty.
//! - **InvalidPeriod**: sma: `period` is zero or exceeds data length.
//! - **NotEnoughValidData**: sma: Fewer than `period` valid (non-`NaN`) values after the first valid index.
//! - **AllValuesNaN**: sma: All input data are `NaN`.
//!
//! ## Returns
//! - **`Ok(SmaOutput)`** on success, containing a `Vec<f64>` matching the input length.
//! - **`Err(SmaError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel, alloc_with_nan_prefix, init_matrix_prefixes, make_uninit_matrix};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use rayon::prelude::*;
use std::convert::AsRef;
use thiserror::Error;
use std::mem::MaybeUninit;

impl<'a> AsRef<[f64]> for SmaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            SmaData::Slice(slice) => slice,
            SmaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum SmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct SmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct SmaParams {
    pub period: Option<usize>,
}

impl Default for SmaParams {
    fn default() -> Self {
        Self { period: Some(9) }
    }
}

#[derive(Debug, Clone)]
pub struct SmaInput<'a> {
    pub data: SmaData<'a>,
    pub params: SmaParams,
}

impl<'a> SmaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: SmaParams) -> Self {
        Self {
            data: SmaData::Candles { candles: c, source: s },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: SmaParams) -> Self {
        Self {
            data: SmaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", SmaParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(9)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SmaBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for SmaBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl SmaBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<SmaOutput, SmaError> {
        let p = SmaParams { period: self.period };
        let i = SmaInput::from_candles(c, "close", p);
        sma_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<SmaOutput, SmaError> {
        let p = SmaParams { period: self.period };
        let i = SmaInput::from_slice(d, p);
        sma_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<SmaStream, SmaError> {
        let p = SmaParams { period: self.period };
        SmaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum SmaError {
    #[error("sma: Empty data provided for SMA.")]
    EmptyData,
    #[error("sma: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("sma: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("sma: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn sma(input: &SmaInput) -> Result<SmaOutput, SmaError> {
    sma_with_kernel(input, Kernel::Auto)
}

pub fn sma_with_kernel(input: &SmaInput, kernel: Kernel) -> Result<SmaOutput, SmaError> {
    let data: &[f64] = input.as_ref();
    if data.is_empty() {
        return Err(SmaError::EmptyData);
    }
    let period = input.get_period();
    let len = data.len();
    if period == 0 || period > len {
        return Err(SmaError::InvalidPeriod { period, data_len: len });
    }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(SmaError::AllValuesNaN)?;
    if len - first < period {
        return Err(SmaError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let warm = first + period - 1;
    let mut out = alloc_with_nan_prefix(len, warm);
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                sma_scalar(data, period, first, &mut out);
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                sma_avx2(data, period, first, &mut out);
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                sma_avx512(data, period, first, &mut out);
            }
            _ => unreachable!(),
        }
    }

    Ok(SmaOutput { values: out })
}

#[inline(always)]
pub unsafe fn sma_scalar(
    data:      &[f64],
    period:    usize,
    first:     usize,
    out:      &mut [f64],
) {
    debug_assert!(period >= 1);
    debug_assert_eq!(data.len(), out.len());
    let len = data.len();

    let dp = data.as_ptr();
    let op = out.as_mut_ptr();

    let mut sum = 0.0;
    for k in 0..period {
        sum += *dp.add(first + k);
    }
    let inv = 1.0 / (period as f64);

    *op.add(first + period - 1) = sum * inv;

    for i in (first + period)..len {
        sum += *dp.add(i) - *dp.add(i - period);
        *op.add(i) = sum * inv;
    }
}


#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn sma_avx2(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    sma_scalar(data, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn sma_avx512(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        unsafe { sma_avx512_short(data, period, first, out) }
    } else {
        unsafe { sma_avx512_long(data, period, first, out) }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn sma_avx512_short(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    // Stub: call scalar
    sma_scalar(data, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn sma_avx512_long(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    // Stub: call scalar
    sma_scalar(data, period, first, out);
}

#[derive(Debug, Clone)]
pub struct SmaStream {
    period: usize,
    buffer: Vec<f64>,
    head: usize,
    sum: f64,
    filled: bool,
}

impl SmaStream {
    pub fn try_new(params: SmaParams) -> Result<Self, SmaError> {
        let period = params.period.unwrap_or(9);
        if period == 0 {
            return Err(SmaError::InvalidPeriod { period, data_len: 0 });
        }
        Ok(Self {
            period,
            buffer: vec![0.0; period],
            head: 0,
            sum: 0.0,
            filled: false,
        })
    }
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        if !self.filled && self.head == 0 && self.sum == 0.0 {
            self.sum = value;
            self.buffer[self.head] = value;
            self.head = (self.head + 1) % self.period;
            if self.head == 0 {
                self.filled = true;
            }
            return None;
        }
        self.sum += value - self.buffer[self.head];
        self.buffer[self.head] = value;
        self.head = (self.head + 1) % self.period;
        if !self.filled && self.head == 0 {
            self.filled = true;
        }
        if self.filled {
            Some(self.sum / self.period as f64)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub struct SmaBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for SmaBatchRange {
    fn default() -> Self {
        Self {
            period: (9, 240, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SmaBatchBuilder {
    range: SmaBatchRange,
    kernel: Kernel,
}

impl SmaBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<SmaBatchOutput, SmaError> {
        sma_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<SmaBatchOutput, SmaError> {
        SmaBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<SmaBatchOutput, SmaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<SmaBatchOutput, SmaError> {
        SmaBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
    }
}

pub fn sma_batch_with_kernel(
    data: &[f64],
    sweep: &SmaBatchRange,
    k: Kernel,
) -> Result<SmaBatchOutput, SmaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(SmaError::InvalidPeriod {
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
    sma_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct SmaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<SmaParams>,
    pub rows: usize,
    pub cols: usize,
}
impl SmaBatchOutput {
    pub fn row_for_params(&self, p: &SmaParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(9) == p.period.unwrap_or(9)
        })
    }
    pub fn values_for(&self, p: &SmaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &SmaBatchRange) -> Vec<SmaParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(SmaParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn sma_batch_slice(
    data: &[f64],
    sweep: &SmaBatchRange,
    kern: Kernel,
) -> Result<SmaBatchOutput, SmaError> {
    sma_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn sma_batch_par_slice(
    data: &[f64],
    sweep: &SmaBatchRange,
    kern: Kernel,
) -> Result<SmaBatchOutput, SmaError> {
    sma_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn sma_batch_inner(
    data: &[f64],
    sweep: &SmaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<SmaBatchOutput, SmaError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(SmaError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(SmaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(SmaError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)   // first valid SMA index for that row
        .collect();

    let mut raw = make_uninit_matrix(rows, cols);
    unsafe { init_matrix_prefixes(&mut raw, cols, &warm) };

    // ---- closure that writes one row ------------------------------------------
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();

        // cast this row to &mut [f64]
        let out_row = core::slice::from_raw_parts_mut(
            dst_mu.as_mut_ptr() as *mut f64,
            dst_mu.len(),
        );

        match kern {
            Kernel::Scalar => sma_row_scalar(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2   => sma_row_avx2  (data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => sma_row_avx512(data, first, period, out_row),
            _ => unreachable!(),
        }
    };

    // ---- run every row, filling `raw` in-place ---------------------------------
    if parallel {
        raw.par_chunks_mut(cols)
            .enumerate()
            .for_each(|(row, slice)| do_row(row, slice));
    } else {
        for (row, slice) in raw.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    // ---- finished: transmute into a Vec<f64> -----------------------------------
    let values: Vec<f64> = unsafe { std::mem::transmute(raw) };
    Ok(SmaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn sma_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    sma_scalar(data, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn sma_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    sma_avx2(data, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn sma_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        sma_avx512_short(data, period, first, out);
    } else {
        sma_avx512_long(data, period, first, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn sma_row_avx512_short(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    sma_scalar(data, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn sma_row_avx512_long(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    sma_scalar(data, period, first, out);
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;
    fn check_sma_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = SmaParams { period: None };
        let input = SmaInput::from_candles(&candles, "close", default_params);
        let output = sma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }
    fn check_sma_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = SmaParams { period: Some(9) };
        let input = SmaInput::from_candles(&candles, "close", params);
        let result = sma_with_kernel(&input, kernel)?;
        let expected_last_five = [59180.8, 59175.0, 59129.4, 59085.4, 59133.7];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(diff < 1e-1, "[{}] SMA {:?} mismatch at idx {}: got {}, expected {}", test_name, kernel, i, val, expected_last_five[i]);
        }
        Ok(())
    }
    fn check_sma_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = SmaInput::with_default_candles(&candles);
        match input.data {
            SmaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected SmaData::Candles"),
        }
        let output = sma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }
    fn check_sma_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = SmaParams { period: Some(0) };
        let input = SmaInput::from_slice(&input_data, params);
        let res = sma_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] SMA should fail with zero period", test_name);
        Ok(())
    }
    fn check_sma_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = SmaParams { period: Some(10) };
        let input = SmaInput::from_slice(&data_small, params);
        let res = sma_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] SMA should fail with period exceeding length", test_name);
        Ok(())
    }
    fn check_sma_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = SmaParams { period: Some(9) };
        let input = SmaInput::from_slice(&single_point, params);
        let res = sma_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] SMA should fail with insufficient data", test_name);
        Ok(())
    }
    fn check_sma_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = SmaParams { period: Some(14) };
        let first_input = SmaInput::from_candles(&candles, "close", first_params);
        let first_result = sma_with_kernel(&first_input, kernel)?;
        let second_params = SmaParams { period: Some(14) };
        let second_input = SmaInput::from_slice(&first_result.values, second_params);
        let second_result = sma_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }
    fn check_sma_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = SmaInput::from_candles(&candles, "close", SmaParams { period: Some(9) });
        let res = sma_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 240 {
            for (i, &val) in res.values[240..].iter().enumerate() {
                assert!(!val.is_nan(), "[{}] Found unexpected NaN at out-index {}", test_name, 240 + i);
            }
        }
        Ok(())
    }
    fn check_sma_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 9;
        let input = SmaInput::from_candles(&candles, "close", SmaParams { period: Some(period) });
        let batch_output = sma_with_kernel(&input, kernel)?.values;
        let mut stream = SmaStream::try_new(SmaParams { period: Some(period) })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(sma_val) => stream_values.push(sma_val),
                None => stream_values.push(f64::NAN),
            }
        }
        assert_eq!(batch_output.len(), stream_values.len());
        for (i, (&b, &s)) in batch_output.iter().zip(stream_values.iter()).enumerate() {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(diff < 1e-9, "[{}] SMA streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}", test_name, i, b, s, diff);
        }
        Ok(())
    }
    macro_rules! generate_all_sma_tests {
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
    generate_all_sma_tests!(
        check_sma_partial_params,
        check_sma_accuracy,
        check_sma_default_candles,
        check_sma_zero_period,
        check_sma_period_exceeds_length,
        check_sma_very_small_dataset,
        check_sma_reinput,
        check_sma_nan_handling,
        check_sma_streaming
    );
    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = SmaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = SmaParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [59180.8, 59175.0, 59129.4, 59085.4, 59133.7];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!((v - expected[i]).abs() < 1e-1, "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}");
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
