//! # Detrended Price Oscillator (DPO)
//!
//! The Detrended Price Oscillator (DPO) removes trend to highlight cycles by subtracting a
//! centered moving average from a shifted price. Controlled by a single period parameter.
//!
//! ## Parameters
//! - **period**: Window size (number of data points).
//!
//! ## Errors
//! - **AllValuesNaN**: dpo: All input data values are `NaN`.
//! - **InvalidPeriod**: dpo: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: dpo: Not enough valid data points for the requested `period`.
//!
//! ## Returns
//! - **`Ok(DpoOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(DpoError)`** otherwise.
//!
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
use paste::paste;

impl<'a> AsRef<[f64]> for DpoInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            DpoData::Slice(slice) => slice,
            DpoData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum DpoData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct DpoOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct DpoParams {
    pub period: Option<usize>,
}

impl Default for DpoParams {
    fn default() -> Self {
        Self { period: Some(5) }
    }
}

#[derive(Debug, Clone)]
pub struct DpoInput<'a> {
    pub data: DpoData<'a>,
    pub params: DpoParams,
}

impl<'a> DpoInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: DpoParams) -> Self {
        Self {
            data: DpoData::Candles { candles: c, source: s },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: DpoParams) -> Self {
        Self {
            data: DpoData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", DpoParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(5)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct DpoBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for DpoBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl DpoBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<DpoOutput, DpoError> {
        let p = DpoParams {
            period: self.period,
        };
        let i = DpoInput::from_candles(c, "close", p);
        dpo_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<DpoOutput, DpoError> {
        let p = DpoParams {
            period: self.period,
        };
        let i = DpoInput::from_slice(d, p);
        dpo_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<DpoStream, DpoError> {
        let p = DpoParams {
            period: self.period,
        };
        DpoStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum DpoError {
    #[error("dpo: All values are NaN.")]
    AllValuesNaN,

    #[error("dpo: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("dpo: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn dpo(input: &DpoInput) -> Result<DpoOutput, DpoError> {
    dpo_with_kernel(input, Kernel::Auto)
}

pub fn dpo_with_kernel(input: &DpoInput, kernel: Kernel) -> Result<DpoOutput, DpoError> {
    let data: &[f64] = input.as_ref();
    let first = data.iter().position(|x| !x.is_nan()).ok_or(DpoError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(DpoError::InvalidPeriod { period, data_len: len });
    }
    if (len - first) < period {
        return Err(DpoError::NotEnoughValidData { needed: period, valid: len - first });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let mut out = vec![f64::NAN; len];
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                dpo_scalar(data, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                dpo_avx2(data, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                dpo_avx512(data, period, first, &mut out)
            }
            _ => unreachable!(),
        }
    }
    Ok(DpoOutput { values: out })
}

#[inline]
pub fn dpo_scalar(data: &[f64], period: usize, first_val: usize, out: &mut [f64]) {
    let back = period / 2 + 1;
    let mut sum = 0.0;
    let scale = 1.0 / (period as f64);

    for i in first_val..data.len() {
        sum += data[i];
        if i >= first_val + period {
            sum -= data[i - period];
        }
        if i >= first_val + period - 1 && i >= back {
            out[i] = data[i - back] - (sum * scale);
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn dpo_avx512(data: &[f64], period: usize, first_val: usize, out: &mut [f64]) {
    unsafe { dpo_scalar(data, period, first_val, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn dpo_avx2(data: &[f64], period: usize, first_val: usize, out: &mut [f64]) {
    unsafe { dpo_scalar(data, period, first_val, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn dpo_avx512_short(data: &[f64], period: usize, first_val: usize, out: &mut [f64]) {
    unsafe { dpo_scalar(data, period, first_val, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn dpo_avx512_long(data: &[f64], period: usize, first_val: usize, out: &mut [f64]) {
    unsafe { dpo_scalar(data, period, first_val, out) }
}

#[inline]
pub fn dpo_batch_with_kernel(
    data: &[f64],
    sweep: &DpoBatchRange,
    k: Kernel,
) -> Result<DpoBatchOutput, DpoError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(DpoError::InvalidPeriod {
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
    dpo_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct DpoBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for DpoBatchRange {
    fn default() -> Self {
        Self {
            period: (5, 60, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct DpoBatchBuilder {
    range: DpoBatchRange,
    kernel: Kernel,
}

impl DpoBatchBuilder {
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

    pub fn apply_slice(self, data: &[f64]) -> Result<DpoBatchOutput, DpoError> {
        dpo_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<DpoBatchOutput, DpoError> {
        DpoBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<DpoBatchOutput, DpoError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<DpoBatchOutput, DpoError> {
        DpoBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct DpoBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<DpoParams>,
    pub rows: usize,
    pub cols: usize,
}

impl DpoBatchOutput {
    pub fn row_for_params(&self, p: &DpoParams) -> Option<usize> {
        self.combos.iter().position(|c| c.period.unwrap_or(5) == p.period.unwrap_or(5))
    }

    pub fn values_for(&self, p: &DpoParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &DpoBatchRange) -> Vec<DpoParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(DpoParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn dpo_batch_slice(
    data: &[f64],
    sweep: &DpoBatchRange,
    kern: Kernel,
) -> Result<DpoBatchOutput, DpoError> {
    dpo_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn dpo_batch_par_slice(
    data: &[f64],
    sweep: &DpoBatchRange,
    kern: Kernel,
) -> Result<DpoBatchOutput, DpoError> {
    dpo_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn dpo_batch_inner(
    data: &[f64],
    sweep: &DpoBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<DpoBatchOutput, DpoError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(DpoError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(DpoError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(DpoError::NotEnoughValidData {
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
            Kernel::Scalar => dpo_row_scalar(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => dpo_row_avx2(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => dpo_row_avx512(data, first, period, out_row),
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

    Ok(DpoBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn dpo_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    dpo_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn dpo_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    dpo_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn dpo_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    if period <= 32 {
        dpo_row_avx512_short(data, first, period, out);
    } else {
        dpo_row_avx512_long(data, first, period, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn dpo_row_avx512_short(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    dpo_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn dpo_row_avx512_long(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    dpo_scalar(data, period, first, out)
}

#[derive(Debug, Clone)]
pub struct DpoStream {
    period: usize,
    buf: Vec<f64>,
    sum: f64,
    head: usize,
    filled: bool,
}

impl DpoStream {
    pub fn try_new(params: DpoParams) -> Result<Self, DpoError> {
        let period = params.period.unwrap_or(5);
        if period == 0 {
            return Err(DpoError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        Ok(Self {
            period,
            buf: vec![f64::NAN; period],
            sum: 0.0,
            head: 0,
            filled: false,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        if !self.filled && self.head == self.period - 1 {
            self.filled = true;
        }
        if !self.buf[self.head].is_nan() {
            self.sum -= self.buf[self.head];
        }
        self.buf[self.head] = value;
        self.sum += value;
        self.head = (self.head + 1) % self.period;
        if !self.filled {
            return None;
        }
        let back = self.period / 2 + 1;
        let idx = (self.head + self.period - back) % self.period;
        Some(self.buf[idx] - (self.sum / self.period as f64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_dpo_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = DpoParams { period: None };
        let input = DpoInput::from_candles(&candles, "close", default_params);
        let output = dpo_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_dpo_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = DpoInput::from_candles(&candles, "close", DpoParams { period: Some(5) });
        let result = dpo_with_kernel(&input, kernel)?;
        let expected_last_five = [
            65.3999999999287,
            131.3999999999287,
            32.599999999925785,
            98.3999999999287,
            117.99999999992724,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-1,
                "[{}] DPO {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_dpo_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = DpoInput::with_default_candles(&candles);
        match input.data {
            DpoData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected DpoData::Candles"),
        }
        let output = dpo_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_dpo_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = DpoParams { period: Some(0) };
        let input = DpoInput::from_slice(&input_data, params);
        let res = dpo_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] DPO should fail with zero period", test_name);
        Ok(())
    }

    fn check_dpo_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = DpoParams { period: Some(10) };
        let input = DpoInput::from_slice(&data_small, params);
        let res = dpo_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] DPO should fail with period exceeding length", test_name);
        Ok(())
    }

    fn check_dpo_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = DpoParams { period: Some(5) };
        let input = DpoInput::from_slice(&single_point, params);
        let res = dpo_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] DPO should fail with insufficient data", test_name);
        Ok(())
    }

    fn check_dpo_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = DpoParams { period: Some(5) };
        let first_input = DpoInput::from_candles(&candles, "close", first_params);
        let first_result = dpo_with_kernel(&first_input, kernel)?;
        let second_params = DpoParams { period: Some(5) };
        let second_input = DpoInput::from_slice(&first_result.values, second_params);
        let second_result = dpo_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    fn check_dpo_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = DpoInput::from_candles(&candles, "close", DpoParams { period: Some(5) });
        let res = dpo_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 20 {
            for (i, &val) in res.values[20..].iter().enumerate() {
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

    fn check_dpo_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 5;
        let input = DpoInput::from_candles(&candles, "close", DpoParams { period: Some(period) });
        let batch_output = dpo_with_kernel(&input, kernel)?.values;
        let mut stream = DpoStream::try_new(DpoParams { period: Some(period) })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(dpo_val) => stream_values.push(dpo_val),
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
                "[{}] DPO streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_dpo_tests {
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

    generate_all_dpo_tests!(
        check_dpo_partial_params,
        check_dpo_accuracy,
        check_dpo_default_candles,
        check_dpo_zero_period,
        check_dpo_period_exceeds_length,
        check_dpo_very_small_dataset,
        check_dpo_reinput,
        check_dpo_nan_handling,
        check_dpo_streaming
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = DpoBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = DpoParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            65.3999999999287,
            131.3999999999287,
            32.599999999925785,
            98.3999999999287,
            117.99999999992724,
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
}
