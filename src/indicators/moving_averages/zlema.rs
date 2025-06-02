//! # Zero Lag Exponential Moving Average (ZLEMA)
//!
//! ZLEMA is a moving average designed to reduce lag by de-lagging the input before EMA calculation.
//! Supports kernel (SIMD) selection and batch/grid computation with streaming support.
//!
//! ## Parameters
//! - **period**: Lookback window (>= 1, defaults to 14).
//!
//! ## Errors
//! - **AllValuesNaN**: All input values are `NaN`.
//! - **InvalidPeriod**: `period` is 0 or exceeds data length.
//!
//! ## Returns
//! - **`Ok(ZlemaOutput)`** on success, containing a `Vec<f64>`.
//! - **`Err(ZlemaError)`** otherwise.

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

impl<'a> AsRef<[f64]> for ZlemaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            ZlemaData::Slice(slice) => slice,
            ZlemaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ZlemaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct ZlemaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct ZlemaParams {
    pub period: Option<usize>,
}

impl Default for ZlemaParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct ZlemaInput<'a> {
    pub data: ZlemaData<'a>,
    pub params: ZlemaParams,
}

impl<'a> ZlemaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: ZlemaParams) -> Self {
        Self {
            data: ZlemaData::Candles { candles: c, source: s },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: ZlemaParams) -> Self {
        Self {
            data: ZlemaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", ZlemaParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ZlemaBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for ZlemaBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl ZlemaBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<ZlemaOutput, ZlemaError> {
        let p = ZlemaParams { period: self.period };
        let i = ZlemaInput::from_candles(c, "close", p);
        zlema_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<ZlemaOutput, ZlemaError> {
        let p = ZlemaParams { period: self.period };
        let i = ZlemaInput::from_slice(d, p);
        zlema_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<ZlemaStream, ZlemaError> {
        let p = ZlemaParams { period: self.period };
        ZlemaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum ZlemaError {
    #[error("zlema: All values are NaN.")]
    AllValuesNaN,
    #[error("zlema: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
}

#[inline]
pub fn zlema(input: &ZlemaInput) -> Result<ZlemaOutput, ZlemaError> {
    zlema_with_kernel(input, Kernel::Auto)
}

pub fn zlema_with_kernel(input: &ZlemaInput, kernel: Kernel) -> Result<ZlemaOutput, ZlemaError> {
    let data: &[f64] = match &input.data {
        ZlemaData::Candles { candles, source } => source_type(candles, source),
        ZlemaData::Slice(sl) => sl,
    };

    let first = data.iter().position(|x| !x.is_nan()).ok_or(ZlemaError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(ZlemaError::InvalidPeriod { period, data_len: len });
    }
    if (len - first) < period {
        return Err(ZlemaError::InvalidPeriod { period, data_len: len - first });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                zlema_scalar(data, period, first, &mut vec![0.0; 0]) // we don't use output here, see below
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                zlema_avx2(data, period, first, &mut vec![0.0; 0])
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                zlema_avx512(data, period, first, &mut vec![0.0; 0])
            }
            _ => unreachable!(),
        }
    }
}

#[inline]
pub fn zlema_scalar(
    data: &[f64],
    period: usize,
    first_val: usize,
    _out: &mut [f64],
) -> Result<ZlemaOutput, ZlemaError> {
    let len = data.len();
    let lag = (period - 1) / 2;
    let alpha = 2.0 / (period as f64 + 1.0);
    let mut zlema_values = vec![f64::NAN; len];
    let mut last_ema = data[first_val];
    zlema_values[first_val] = last_ema;
    for i in (first_val + 1)..len {
        let val = if i < lag { data[i] } else { 2.0 * data[i] - data[i - lag] };
        last_ema = alpha * val + (1.0 - alpha) * last_ema;
        zlema_values[i] = last_ema;
    }
    Ok(ZlemaOutput { values: zlema_values })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn zlema_avx512(
    data: &[f64],
    period: usize,
    first_val: usize,
    out: &mut [f64],
) -> Result<ZlemaOutput, ZlemaError> {
    zlema_scalar(data, period, first_val, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn zlema_avx2(
    data: &[f64],
    period: usize,
    first_val: usize,
    out: &mut [f64],
) -> Result<ZlemaOutput, ZlemaError> {
    zlema_scalar(data, period, first_val, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn zlema_avx512_short(
    data: &[f64],
    period: usize,
    first_val: usize,
    out: &mut [f64],
) -> Result<ZlemaOutput, ZlemaError> {
    zlema_scalar(data, period, first_val, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn zlema_avx512_long(
    data: &[f64],
    period: usize,
    first_val: usize,
    out: &mut [f64],
) -> Result<ZlemaOutput, ZlemaError> {
    zlema_scalar(data, period, first_val, out)
}

#[inline]
pub fn zlema_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    let len = data.len();
    let lag = (period - 1) / 2;
    let alpha = 2.0 / (period as f64 + 1.0);
    let mut last_ema = data[first];
    out[first] = last_ema;
    for i in (first + 1)..len {
        let val = if i < lag { data[i] } else { 2.0 * data[i] - data[i - lag] };
        last_ema = alpha * val + (1.0 - alpha) * last_ema;
        out[i] = last_ema;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn zlema_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    w_ptr: *const f64,
    inv_n: f64,
    out: &mut [f64],
) {
    zlema_row_scalar(data, first, period, stride, w_ptr, inv_n, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn zlema_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    w_ptr: *const f64,
    inv_n: f64,
    out: &mut [f64],
) {
    zlema_row_scalar(data, first, period, stride, w_ptr, inv_n, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn zlema_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    w_ptr: *const f64,
    inv_n: f64,
    out: &mut [f64],
) {
    zlema_row_scalar(data, first, period, stride, w_ptr, inv_n, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn zlema_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    w_ptr: *const f64,
    inv_n: f64,
    out: &mut [f64],
) {
    zlema_row_scalar(data, first, period, stride, w_ptr, inv_n, out)
}

#[derive(Debug, Clone)]
pub struct ZlemaStream {
    period: usize,
    lag: usize,
    alpha: f64,
    last_ema: f64,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,
}

impl ZlemaStream {
    pub fn try_new(params: ZlemaParams) -> Result<Self, ZlemaError> {
        let period = params.period.unwrap_or(14);
        if period == 0 {
            return Err(ZlemaError::InvalidPeriod { period, data_len: 0 });
        }
        Ok(Self {
            period,
            lag: (period - 1) / 2,
            alpha: 2.0 / (period as f64 + 1.0),
            last_ema: f64::NAN,
            buffer: vec![f64::NAN; period],
            head: 0,
            filled: false,
        })
    }
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.buffer[self.head] = value;
        let lag_idx = (self.head + self.period - self.lag) % self.period;
        self.head = (self.head + 1) % self.period;

        if !self.filled && self.head == 0 {
            self.filled = true;
        }
        let val = if !self.filled { value } else { 2.0 * value - self.buffer[lag_idx] };
        if self.last_ema.is_nan() {
            self.last_ema = val;
        } else {
            self.last_ema = self.alpha * val + (1.0 - self.alpha) * self.last_ema;
        }
        Some(self.last_ema)
    }
}

#[derive(Clone, Debug)]
pub struct ZlemaBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for ZlemaBatchRange {
    fn default() -> Self {
        Self { period: (14, 40, 1) }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ZlemaBatchBuilder {
    range: ZlemaBatchRange,
    kernel: Kernel,
}

impl ZlemaBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<ZlemaBatchOutput, ZlemaError> {
        zlema_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<ZlemaBatchOutput, ZlemaError> {
        ZlemaBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<ZlemaBatchOutput, ZlemaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<ZlemaBatchOutput, ZlemaError> {
        ZlemaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn zlema_batch_with_kernel(
    data: &[f64],
    sweep: &ZlemaBatchRange,
    k: Kernel,
) -> Result<ZlemaBatchOutput, ZlemaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(ZlemaError::InvalidPeriod { period: 0, data_len: 0 });
        }
    };

    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    zlema_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct ZlemaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<ZlemaParams>,
    pub rows: usize,
    pub cols: usize,
}

impl ZlemaBatchOutput {
    pub fn row_for_params(&self, p: &ZlemaParams) -> Option<usize> {
        self.combos.iter().position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
    }
    pub fn values_for(&self, p: &ZlemaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &ZlemaBatchRange) -> Vec<ZlemaParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(ZlemaParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn zlema_batch_slice(
    data: &[f64],
    sweep: &ZlemaBatchRange,
    kern: Kernel,
) -> Result<ZlemaBatchOutput, ZlemaError> {
    zlema_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn zlema_batch_par_slice(
    data: &[f64],
    sweep: &ZlemaBatchRange,
    kern: Kernel,
) -> Result<ZlemaBatchOutput, ZlemaError> {
    zlema_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn zlema_batch_inner(
    data: &[f64],
    sweep: &ZlemaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<ZlemaBatchOutput, ZlemaError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(ZlemaError::InvalidPeriod { period: 0, data_len: 0 });
    }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(ZlemaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(ZlemaError::InvalidPeriod { period: max_p, data_len: data.len() - first });
    }
    let rows = combos.len();
    let cols = data.len();
    let mut values = vec![f64::NAN; rows * cols];
    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        match kern {
            Kernel::Scalar => zlema_row_scalar(data, first, period, 0, std::ptr::null(), 0.0, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => zlema_row_avx2(data, first, period, 0, std::ptr::null(), 0.0, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => zlema_row_avx512(data, first, period, 0, std::ptr::null(), 0.0, out_row),
            _ => unreachable!(),
        }
    };
    if parallel {
        values.par_chunks_mut(cols).enumerate().for_each(|(row, slice)| do_row(row, slice));
    } else {
        for (row, slice) in values.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }
    Ok(ZlemaBatchOutput { values, combos, rows, cols })
}

#[inline(always)]
pub fn expand_grid_zlema(r: &ZlemaBatchRange) -> Vec<ZlemaParams> {
    expand_grid(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_zlema_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = ZlemaParams { period: None };
        let input = ZlemaInput::from_candles(&candles, "close", default_params);
        let output = zlema_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_zlema_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = ZlemaInput::from_candles(&candles, "close", ZlemaParams::default());
        let result = zlema_with_kernel(&input, kernel)?;
        let expected_last_five = [59015.1, 59165.2, 59168.1, 59147.0, 58978.9];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(diff < 1e-1, "[{}] ZLEMA {:?} mismatch at idx {}: got {}, expected {}", test_name, kernel, i, val, expected_last_five[i]);
        }
        Ok(())
    }

    fn check_zlema_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = ZlemaParams { period: Some(0) };
        let input = ZlemaInput::from_slice(&input_data, params);
        let res = zlema_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] ZLEMA should fail with zero period", test_name);
        Ok(())
    }

    fn check_zlema_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = ZlemaParams { period: Some(10) };
        let input = ZlemaInput::from_slice(&data_small, params);
        let res = zlema_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] ZLEMA should fail with period exceeding length", test_name);
        Ok(())
    }

    fn check_zlema_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = ZlemaParams { period: Some(14) };
        let input = ZlemaInput::from_slice(&single_point, params);
        let res = zlema_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] ZLEMA should fail with insufficient data", test_name);
        Ok(())
    }

    fn check_zlema_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = ZlemaParams { period: Some(21) };
        let first_input = ZlemaInput::from_candles(&candles, "close", first_params);
        let first_result = zlema_with_kernel(&first_input, kernel)?;
        let second_params = ZlemaParams { period: Some(14) };
        let second_input = ZlemaInput::from_slice(&first_result.values, second_params);
        let second_result = zlema_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        for (idx, &val) in second_result.values.iter().enumerate().skip(14) {
            assert!(val.is_finite(), "NaN found at index {}", idx);
        }
        Ok(())
    }

    fn check_zlema_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = ZlemaInput::from_candles(&candles, "close", ZlemaParams::default());
        let res = zlema_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 20 {
            for (i, &val) in res.values[20..].iter().enumerate() {
                assert!(!val.is_nan(), "[{}] Found unexpected NaN at out-index {}", test_name, 20 + i);
            }
        }
        Ok(())
    }

    macro_rules! generate_all_zlema_tests {
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

    generate_all_zlema_tests!(
        check_zlema_partial_params,
        check_zlema_accuracy,
        check_zlema_zero_period,
        check_zlema_period_exceeds_length,
        check_zlema_very_small_dataset,
        check_zlema_reinput,
        check_zlema_nan_handling
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = ZlemaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = ZlemaParams::default();
        let row = output.values_for(&def).expect("default row missing");
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
