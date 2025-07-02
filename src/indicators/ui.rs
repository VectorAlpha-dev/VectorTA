//! # Ulcer Index (UI)
//!
//! The Ulcer Index (UI) is a volatility indicator that measures price drawdown from recent highs
//! and focuses on downside risk. It is calculated as the square root of the average of the squared
//! percentage drawdowns from the rolling maximum price within a specified window.
//!
//! ## Parameters
//! - **period**: Window size (number of data points), default 14.
//! - **scalar**: Multiplier applied to drawdown, default 100.0.
//!
//! ## Errors
//! - **UiAllValuesNaN**: All input values are NaN.
//! - **UiInvalidPeriod**: `period` is zero or exceeds data length.
//! - **UiNotEnoughValidData**: Not enough valid data points for period.
//!
//! ## Returns
//! - **Ok(UiOutput)** on success, containing a Vec<f64> matching the input length.
//! - **Err(UiError)** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use crate::utilities::enums::Kernel;
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use thiserror::Error;

impl<'a> AsRef<[f64]> for UiInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            UiData::Slice(slice) => slice,
            UiData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum UiData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct UiOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct UiParams {
    pub period: Option<usize>,
    pub scalar: Option<f64>,
}

impl Default for UiParams {
    fn default() -> Self {
        Self {
            period: Some(14),
            scalar: Some(100.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct UiInput<'a> {
    pub data: UiData<'a>,
    pub params: UiParams,
}

impl<'a> UiInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: UiParams) -> Self {
        Self {
            data: UiData::Candles { candles: c, source: s },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: UiParams) -> Self {
        Self {
            data: UiData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", UiParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
    #[inline]
    pub fn get_scalar(&self) -> f64 {
        self.params.scalar.unwrap_or(100.0)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct UiBuilder {
    period: Option<usize>,
    scalar: Option<f64>,
    kernel: Kernel,
}

impl Default for UiBuilder {
    fn default() -> Self {
        Self {
            period: None,
            scalar: None,
            kernel: Kernel::Auto,
        }
    }
}

impl UiBuilder {
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
    pub fn scalar(mut self, s: f64) -> Self {
        self.scalar = Some(s);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<UiOutput, UiError> {
        let p = UiParams { period: self.period, scalar: self.scalar };
        let i = UiInput::from_candles(c, "close", p);
        ui_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<UiOutput, UiError> {
        let p = UiParams { period: self.period, scalar: self.scalar };
        let i = UiInput::from_slice(d, p);
        ui_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<UiStream, UiError> {
        let p = UiParams { period: self.period, scalar: self.scalar };
        UiStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum UiError {
    #[error("ui: All values are NaN.")]
    AllValuesNaN,
    #[error("ui: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("ui: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn ui(input: &UiInput) -> Result<UiOutput, UiError> {
    ui_with_kernel(input, Kernel::Auto)
}

pub fn ui_with_kernel(input: &UiInput, kernel: Kernel) -> Result<UiOutput, UiError> {
    let data: &[f64] = input.as_ref();
    let first = data.iter().position(|x| !x.is_nan()).ok_or(UiError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();
    let scalar = input.get_scalar();

    if period == 0 || period > len {
        return Err(UiError::InvalidPeriod { period, data_len: len });
    }
    if (len - first) < period {
        return Err(UiError::NotEnoughValidData { needed: period, valid: len - first });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let mut out = vec![f64::NAN; len];

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                ui_scalar(data, period, scalar, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                ui_avx2(data, period, scalar, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                ui_avx512(data, period, scalar, first, &mut out)
            }
            _ => unreachable!(),
        }
    }

    Ok(UiOutput { values: out })
}

#[inline]
pub fn ui_scalar(
    data: &[f64],
    period: usize,
    scalar: f64,
    first: usize,
    out: &mut [f64],
) {
    let len = data.len();
    let mut rolling_max: AVec<f64, aligned_vec::ConstAlign<CACHELINE_ALIGN>> = AVec::with_capacity(CACHELINE_ALIGN, len);
    rolling_max.resize(len, f64::NAN);

    // Rolling max calculation
    for i in first..len {
        if i < period - 1 {
            continue;
        }
        let mut max = f64::NAN;
        for j in (i + 1 - period)..=i {
            let v = data[j];
            if !v.is_nan() && (max.is_nan() || v > max) {
                max = v;
            }
        }
        rolling_max[i] = max;
    }
    
    let mut squared_drawdowns: AVec<f64, aligned_vec::ConstAlign<CACHELINE_ALIGN>> = AVec::with_capacity(CACHELINE_ALIGN, len);
    squared_drawdowns.resize(len, f64::NAN);

    for i in (first + period - 1)..len {
        if !rolling_max[i].is_nan() && !data[i].is_nan() && rolling_max[i] != 0.0 {
            let dd = scalar * (data[i] - rolling_max[i]) / rolling_max[i];
            squared_drawdowns[i] = dd * dd;
        }
    }

    let mut sum = 0.0;
    let mut count = 0usize;
    for i in (first + period - 1)..len {
        if squared_drawdowns[i].is_nan() {
            continue;
        }
        sum += squared_drawdowns[i];
        count += 1;
        if count >= period {
            break;
        }
    }

    if count == period {
        out[first + period * 2 - 2] = (sum / period as f64).sqrt();
    }

    for i in (first + period * 2 - 1)..len {
        if squared_drawdowns[i].is_nan() || squared_drawdowns[i - period].is_nan() {
            continue;
        }
        sum += squared_drawdowns[i] - squared_drawdowns[i - period];
        out[i] = (sum / period as f64).sqrt();
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn ui_avx512(
    data: &[f64],
    period: usize,
    scalar: f64,
    first: usize,
    out: &mut [f64],
) {
    unsafe { ui_avx512_short(data, period, scalar, first, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn ui_avx2(
    data: &[f64],
    period: usize,
    scalar: f64,
    first: usize,
    out: &mut [f64],
) {
    ui_scalar(data, period, scalar, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn ui_avx512_short(
    data: &[f64],
    period: usize,
    scalar: f64,
    first: usize,
    out: &mut [f64],
) {
    ui_scalar(data, period, scalar, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn ui_avx512_long(
    data: &[f64],
    period: usize,
    scalar: f64,
    first: usize,
    out: &mut [f64],
) {
    ui_scalar(data, period, scalar, first, out)
}

#[derive(Debug, Clone)]
pub struct UiStream {
    period: usize,
    scalar: f64,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,
    rolling_max: Vec<f64>,
    idx: usize,
}

impl UiStream {
    pub fn try_new(params: UiParams) -> Result<Self, UiError> {
        let period = params.period.unwrap_or(14);
        let scalar = params.scalar.unwrap_or(100.0);
        if period == 0 {
            return Err(UiError::InvalidPeriod { period, data_len: 0 });
        }
        Ok(Self {
            period,
            scalar,
            buffer: vec![f64::NAN; period],
            head: 0,
            filled: false,
            rolling_max: vec![f64::NAN; period],
            idx: 0,
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

        // Rolling max
        let mut max = f64::NAN;
        for &v in &self.buffer {
            if !v.is_nan() && (max.is_nan() || v > max) {
                max = v;
            }
        }
        self.rolling_max[self.idx % self.period] = max;

        let roll_max = max;
        let dd = if !value.is_nan() && !roll_max.is_nan() && roll_max != 0.0 {
            let d = self.scalar * (value - roll_max) / roll_max;
            d * d
        
            } else {
            f64::NAN
        };

        // Windowed sum
        let mut sum = 0.0;
        let mut valid = 0usize;
        for &sq in &self.rolling_max {
            if !sq.is_nan() {
                sum += sq;
                valid += 1;
            }
        }

        self.idx += 1;
        if valid == self.period {
            Some((sum / self.period as f64).sqrt())
        
            } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub struct UiBatchRange {
    pub period: (usize, usize, usize),
    pub scalar: (f64, f64, f64),
}

impl Default for UiBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 60, 1),
            scalar: (100.0, 100.0, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct UiBatchBuilder {
    range: UiBatchRange,
    kernel: Kernel,
}

impl UiBatchBuilder {
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
    pub fn scalar_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.scalar = (start, end, step);
        self
    }
    #[inline]
    pub fn scalar_static(mut self, s: f64) -> Self {
        self.range.scalar = (s, s, 0.0);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<UiBatchOutput, UiError> {
        ui_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<UiBatchOutput, UiError> {
        UiBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<UiBatchOutput, UiError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<UiBatchOutput, UiError> {
        UiBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
    }
}

pub fn ui_batch_with_kernel(
    data: &[f64],
    sweep: &UiBatchRange,
    k: Kernel,
) -> Result<UiBatchOutput, UiError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(UiError::InvalidPeriod { period: 0, data_len: 0 }),
    };

    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    ui_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct UiBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<UiParams>,
    pub rows: usize,
    pub cols: usize,
}

impl UiBatchOutput {
    pub fn row_for_params(&self, p: &UiParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(14) == p.period.unwrap_or(14)
                && (c.scalar.unwrap_or(100.0) - p.scalar.unwrap_or(100.0)).abs() < 1e-12
        })
    }
    pub fn values_for(&self, p: &UiParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &UiBatchRange) -> Vec<UiParams> {
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
    let scalars = axis_f64(r.scalar);
    let mut out = Vec::with_capacity(periods.len() * scalars.len());
    for &p in &periods {
        for &s in &scalars {
            out.push(UiParams {
                period: Some(p),
                scalar: Some(s),
            });
        }
    }
    out
}

#[inline(always)]
pub fn ui_batch_slice(
    data: &[f64],
    sweep: &UiBatchRange,
    kern: Kernel,
) -> Result<UiBatchOutput, UiError> {
    ui_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn ui_batch_par_slice(
    data: &[f64],
    sweep: &UiBatchRange,
    kern: Kernel,
) -> Result<UiBatchOutput, UiError> {
    ui_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn ui_batch_inner(
    data: &[f64],
    sweep: &UiBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<UiBatchOutput, UiError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(UiError::InvalidPeriod { period: 0, data_len: 0 });
    }

    let first = data.iter().position(|x| !x.is_nan()).ok_or(UiError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(UiError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let mut values = vec![f64::NAN; rows * cols];

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        let scalar = combos[row].scalar.unwrap();
        match kern {
            Kernel::Scalar => ui_row_scalar(data, first, period, scalar, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => ui_row_avx2(data, first, period, scalar, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => ui_row_avx512(data, first, period, scalar, out_row),
            _ => unreachable!(),
        }
    };

    if parallel {


        #[cfg(not(target_arch = "wasm32"))] {


        values.par_chunks_mut(cols).enumerate().for_each(|(row, slice)| do_row(row, slice));


        }


        #[cfg(target_arch = "wasm32")] {


        for (row, slice) in values.chunks_mut(cols).enumerate() {


                    do_row(row, slice);


        }

        }
    } else {
        for (row, slice) in values.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    Ok(UiBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn ui_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    scalar: f64,
    out: &mut [f64],
) {
    ui_scalar(data, period, scalar, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn ui_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    scalar: f64,
    out: &mut [f64],
) {
    ui_scalar(data, period, scalar, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn ui_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    scalar: f64,
    out: &mut [f64],
) {
    ui_scalar(data, period, scalar, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn ui_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    scalar: f64,
    out: &mut [f64],
) {
    ui_scalar(data, period, scalar, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn ui_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    scalar: f64,
    out: &mut [f64],
) {
    ui_scalar(data, period, scalar, first, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_ui_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = UiParams { period: None, scalar: None };
        let input = UiInput::from_candles(&candles, "close", default_params);
        let output = ui_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_ui_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = UiParams { period: Some(14), scalar: Some(100.0) };
        let input = UiInput::from_candles(&candles, "close", params);
        let ui_result = ui_with_kernel(&input, kernel)?;
        let expected_last_five_ui = [
            3.514342861283708,
            3.304986039846459,
            3.2011859814326304,
            3.1308860017483373,
            2.909612553474519,
        ];
        assert!(ui_result.values.len() >= 5);
        let start_index = ui_result.values.len() - 5;
        let result_last_five_ui = &ui_result.values[start_index..];
        for (i, &value) in result_last_five_ui.iter().enumerate() {
            let expected_value = expected_last_five_ui[i];
            assert!(
                (value - expected_value).abs() < 1e-6,
                "[{}] UI mismatch at index {}: expected {}, got {}",
                test_name, i, expected_value, value
            );
        }
        let period = 14;
        for i in 0..(period - 1) {
            assert!(ui_result.values[i].is_nan());
        }
        Ok(())
    }

    fn check_ui_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = UiInput::with_default_candles(&candles);
        match input.data {
            UiData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected UiData::Candles"),
        }
        let output = ui_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_ui_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = UiParams { period: Some(0), scalar: None };
        let input = UiInput::from_slice(&input_data, params);
        let res = ui_with_kernel(&input, kernel);
        assert!(res.is_err());
        Ok(())
    }

    fn check_ui_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = UiParams { period: Some(10), scalar: None };
        let input = UiInput::from_slice(&data_small, params);
        let res = ui_with_kernel(&input, kernel);
        assert!(res.is_err());
        Ok(())
    }

    fn check_ui_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = UiParams { period: Some(14), scalar: Some(100.0) };
        let input = UiInput::from_slice(&single_point, params);
        let res = ui_with_kernel(&input, kernel);
        assert!(res.is_err());
        Ok(())
    }

    macro_rules! generate_all_ui_tests {
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

    generate_all_ui_tests!(
        check_ui_partial_params,
        check_ui_accuracy,
        check_ui_default_candles,
        check_ui_zero_period,
        check_ui_period_exceeds_length,
        check_ui_very_small_dataset
    );


    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = UiBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = UiParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            3.514342861283708,
            3.304986039846459,
            3.2011859814326304,
            3.1308860017483373,
            2.909612553474519,
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


