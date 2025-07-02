//! # Exponential Moving Average (EMA)
//!
//! The Exponential Moving Average (EMA) provides a moving average that
//! places a greater weight and significance on the most recent data points.
//! The EMA reacts faster to recent price changes than the simple moving average (SMA).
//!
//! ## Parameters
//! - **period**: Window size (number of data points).
//!
//! ## Errors
//! - **AllValuesNaN**: ema: All input data values are `NaN`.
//! - **InvalidPeriod**: ema: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: ema: Not enough valid data points for the requested `period`.
//!
//! ## Returns
//! - **`Ok(EmaOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(EmaError)`** otherwise.

use crate::utilities::aligned_vector::AlignedVec;
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::mem::MaybeUninit;
use thiserror::Error;

impl<'a> AsRef<[f64]> for EmaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            EmaData::Slice(slice) => slice,
            EmaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum EmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct EmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct EmaParams {
    pub period: Option<usize>,
}

impl Default for EmaParams {
    fn default() -> Self {
        Self { period: Some(9) }
    }
}

#[derive(Debug, Clone)]
pub struct EmaInput<'a> {
    pub data: EmaData<'a>,
    pub params: EmaParams,
}

impl<'a> EmaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: EmaParams) -> Self {
        Self {
            data: EmaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: EmaParams) -> Self {
        Self {
            data: EmaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", EmaParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(9)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct EmaBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for EmaBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl EmaBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<EmaOutput, EmaError> {
        let p = EmaParams {
            period: self.period,
        };
        let i = EmaInput::from_candles(c, "close", p);
        ema_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<EmaOutput, EmaError> {
        let p = EmaParams {
            period: self.period,
        };
        let i = EmaInput::from_slice(d, p);
        ema_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<EmaStream, EmaError> {
        let p = EmaParams {
            period: self.period,
        };
        EmaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum EmaError {
    #[error("ema: Input data slice is empty.")]
    EmptyInputData,
    #[error("ema: All values are NaN.")]
    AllValuesNaN,
    #[error("ema: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("ema: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn ema(input: &EmaInput) -> Result<EmaOutput, EmaError> {
    ema_with_kernel(input, Kernel::Auto)
}

pub fn ema_with_kernel(input: &EmaInput, kernel: Kernel) -> Result<EmaOutput, EmaError> {
    let data: &[f64] = match &input.data {
        EmaData::Candles { candles, source } => source_type(candles, source),
        EmaData::Slice(sl) => sl,
    };

    let len = data.len();
    if len == 0 {
        return Err(EmaError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(EmaError::AllValuesNaN)?;
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(EmaError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(EmaError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    let mut out = alloc_with_nan_prefix(len, first);
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => ema_scalar(data, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => ema_avx2(data, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => ema_avx512(data, period, first, &mut out),
            _ => unreachable!(),
        }
    }
}

#[inline(always)]
pub unsafe fn ema_scalar(
    data: &[f64],
    period: usize,
    first_val: usize,
    out: &mut Vec<f64>,
) -> Result<EmaOutput, EmaError> {
    let len = data.len();
    let alpha = 2.0 / (period as f64 + 1.0);
    let one_m = 1.0 - alpha;

    debug_assert_eq!(out.len(), len);

    let mut prev = *data.get_unchecked(first_val);
    *out.get_unchecked_mut(first_val) = prev;

    let mut src = data.as_ptr().add(first_val + 1);
    let mut dst = out.as_mut_ptr().add(first_val + 1);
    for _ in (first_val + 1)..len {
        let x = *src;
        prev = one_m.mul_add(prev, alpha * x);
        *dst = prev;
        src = src.add(1);
        dst = dst.add(1);
    }
    let values = std::mem::take(out);
    Ok(EmaOutput { values })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn ema_avx2(
    data: &[f64],
    period: usize,
    first_val: usize,
    out: &mut Vec<f64>,
) -> Result<EmaOutput, EmaError> {
    ema_scalar(data, period, first_val, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn ema_avx512(
    data: &[f64],
    period: usize,
    first_val: usize,
    out: &mut Vec<f64>,
) -> Result<EmaOutput, EmaError> {
    ema_scalar(data, period, first_val, out)
}

#[derive(Debug, Clone)]
pub struct EmaStream {
    period: usize,
    alpha: f64,
    beta: f64,
    count: usize,
    mean: f64,
}

impl EmaStream {
    pub fn try_new(params: EmaParams) -> Result<Self, EmaError> {
        let period = params.period.unwrap_or(9);
        if period == 0 {
            return Err(EmaError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let alpha = 2.0 / (period as f64 + 1.0);
        Ok(Self {
            period,
            alpha,
            beta: 1.0 - alpha,
            count: 0,
            mean: f64::NAN,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, x: f64) -> Option<f64> {
        if !x.is_finite() {
            return None;
        }

        self.count += 1;
        if self.count == 1 {
            self.mean = x;
        } else if self.count > self.period {
            self.mean = self.beta.mul_add(self.mean, self.alpha * x);
        
            } else {
            self.mean = ((self.count as f64 - 1.0) * self.mean + x) / self.count as f64;
        }

        Some(self.mean)
    }
}

#[derive(Clone, Debug)]
pub struct EmaBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for EmaBatchRange {
    fn default() -> Self {
        Self {
            period: (9, 240, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct EmaBatchBuilder {
    range: EmaBatchRange,
    kernel: Kernel,
}

impl EmaBatchBuilder {
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

    pub fn apply_slice(self, data: &[f64]) -> Result<EmaBatchOutput, EmaError> {
        ema_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<EmaBatchOutput, EmaError> {
        EmaBatchBuilder::new().kernel(k).apply_slice(data)
    }

    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<EmaBatchOutput, EmaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }

    pub fn with_default_candles(c: &Candles) -> Result<EmaBatchOutput, EmaError> {
        EmaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn ema_batch_with_kernel(
    data: &[f64],
    sweep: &EmaBatchRange,
    k: Kernel,
) -> Result<EmaBatchOutput, EmaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(EmaError::InvalidPeriod {
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
    ema_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct EmaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<EmaParams>,
    pub rows: usize,
    pub cols: usize,
}
impl EmaBatchOutput {
    pub fn row_for_params(&self, p: &EmaParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(9) == p.period.unwrap_or(9))
    }

    pub fn values_for(&self, p: &EmaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &EmaBatchRange) -> Vec<EmaParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }

    let periods = axis_usize(r.period);

    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(EmaParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn ema_batch_slice(
    data: &[f64],
    sweep: &EmaBatchRange,
    kern: Kernel,
) -> Result<EmaBatchOutput, EmaError> {
    ema_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn ema_batch_par_slice(
    data: &[f64],
    sweep: &EmaBatchRange,
    kern: Kernel,
) -> Result<EmaBatchOutput, EmaError> {
    ema_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn ema_batch_inner(
    data: &[f64],
    sweep: &EmaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<EmaBatchOutput, EmaError> {
    // ------------ boiler-plate unchanged -----------------------------------
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(EmaError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    if data.is_empty() {
        return Err(EmaError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(EmaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(EmaError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();

    // ------------ 1. allocate & warm-up prefixes ---------------------------
    let mut raw = make_uninit_matrix(rows, cols);
    let warm: Vec<usize> = vec![first; rows]; // same warm-up for every row
    unsafe { init_matrix_prefixes(&mut raw, cols, &warm) };

    // ------------ 2. row-kernel closure on MaybeUninit rows ---------------
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();

        // cast *this* row slice to &mut [f64] for the kernel
        let dst = core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        match kern {
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => ema_row_avx512(data, first, period, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => ema_row_avx2(data, first, period, dst),
            _ => ema_row_scalar(data, first, period, dst),
        }
    };

    // ------------ 3. run rows in parallel or serial ------------------------
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

    } else {
        for (row, slice) in raw.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    // ------------ 4. soundly transmute to Vec<f64> -------------------------
    let values: Vec<f64> = unsafe { std::mem::transmute(raw) };

    Ok(EmaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn ema_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    let alpha = 2.0 / (period as f64 + 1.0);
    out[first] = data[first];
    for i in (first + 1)..data.len() {
        out[i] = alpha * data[i] + (1.0 - alpha) * out[i - 1];
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn ema_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    ema_row_scalar(data, first, period, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn ema_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    ema_row_scalar(data, first, period, out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use proptest::prelude::*;

    fn check_ema_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = EmaParams { period: None };
        let input = EmaInput::from_candles(&candles, "close", default_params);
        let output = ema_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_ema_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = EmaInput::from_candles(&candles, "close", EmaParams::default());
        let result = ema_with_kernel(&input, kernel)?;
        let expected_last_five = [59302.2, 59277.9, 59230.2, 59215.1, 59103.1];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-1,
                "[{}] EMA {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_ema_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = EmaInput::with_default_candles(&candles);
        match input.data {
            EmaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected EmaData::Candles"),
        }
        let output = ema_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_ema_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = EmaParams { period: Some(0) };
        let input = EmaInput::from_slice(&input_data, params);
        let res = ema_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] EMA should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_ema_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = EmaParams { period: Some(10) };
        let input = EmaInput::from_slice(&data_small, params);
        let res = ema_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] EMA should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_ema_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = EmaParams { period: Some(9) };
        let input = EmaInput::from_slice(&single_point, params);
        let res = ema_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] EMA should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_ema_empty_input(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: [f64; 0] = [];
        let input = EmaInput::from_slice(&empty, EmaParams::default());
        let res = ema_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(EmaError::EmptyInputData)),
            "[{}] EMA should fail with empty input",
            test_name
        );
        Ok(())
    }

    fn check_ema_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = EmaParams { period: Some(9) };
        let first_input = EmaInput::from_candles(&candles, "close", first_params);
        let first_result = ema_with_kernel(&first_input, kernel)?;

        let second_params = EmaParams { period: Some(5) };
        let second_input = EmaInput::from_slice(&first_result.values, second_params);
        let second_result = ema_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());
        if second_result.values.len() > 240 {
            for (i, &val) in second_result.values[240..].iter().enumerate() {
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

    fn check_ema_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = EmaInput::from_candles(&candles, "close", EmaParams { period: Some(9) });
        let res = ema_with_kernel(&input, kernel)?;
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

    fn check_ema_streaming(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let period = 9;
        let warm_up = 240;

        let input = EmaInput::from_candles(
            &candles,
            "close",
            EmaParams {
                period: Some(period),
            },
        );
        let batch_output = ema_with_kernel(&input, kernel)?.values;

        let mut stream = EmaStream::try_new(EmaParams {
            period: Some(period),
        })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());

        for &price in &candles.close {
            stream_values.push(stream.update(price).unwrap_or(f64::NAN));
        }

        assert_eq!(batch_output.len(), stream_values.len());

        for (i, (&b, &s)) in batch_output
            .iter()
            .zip(&stream_values)
            .enumerate()
            .skip(warm_up)
        {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] EMA streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    fn check_ema_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let strat = (
            proptest::collection::vec(
                (-1e6f64..1e6).prop_filter("finite", |x| x.is_finite()),
                30..200,
            ),
            3usize..30,
        );

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, period)| {
                let params = EmaParams {
                    period: Some(period),
                };
                let input = EmaInput::from_slice(&data, params);
                let EmaOutput { values: out } = ema_with_kernel(&input, kernel).unwrap();

                for i in (period - 1)..data.len() {
                    let window = &data[..=i];
                    let lo = window.iter().cloned().fold(f64::INFINITY, f64::min);
                    let hi = window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let y = out[i];

                    prop_assert!(
                        y.is_nan() || (y >= lo - 1e-9 && y <= hi + 1e-9),
                        "idx {i}: {y} not in [{lo}, {hi}]",
                    );
                }
                Ok(())
            })
            .unwrap();

        Ok(())
    }

    macro_rules! generate_all_ema_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(
                    #[test]
                    fn [<$test_fn _scalar_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar);
                    }

                    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                    #[test]
                    fn [<$test_fn _avx2_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2);
                    }

                    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                    #[test]
                    fn [<$test_fn _avx512_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512);
                    }
                )*
            }
        }
    }

    generate_all_ema_tests!(
        check_ema_partial_params,
        check_ema_accuracy,
        check_ema_default_candles,
        check_ema_zero_period,
        check_ema_period_exceeds_length,
        check_ema_very_small_dataset,
        check_ema_empty_input,
        check_ema_reinput,
        check_ema_nan_handling,
        check_ema_streaming,
        check_ema_property
    );

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = EmaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = EmaParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [59302.2, 59277.9, 59230.2, 59215.1, 59103.1];
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
                #[test]
                fn [<$fn_name _scalar>]() {
                    let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test]
                fn [<$fn_name _avx2>]() {
                    let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test]
                fn [<$fn_name _avx512>]() {
                    let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch);
                }
                #[test]
                fn [<$fn_name _auto_detect>]() {
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto);
                }
            }
        };
    }
    gen_batch_tests!(check_batch_default_row);
}
