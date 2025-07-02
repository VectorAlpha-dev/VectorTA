//! # Pascal Weighted Moving Average (PWMA)
//!
//! A weighted moving average using Pascal’s triangle coefficients for weights.
//!
//! ## Parameters
//! - **period**: Window size (number of data points). Defaults to 5.
//!
//! ## Errors
//! - **AllValuesNaN**: pwma: All input data values are `NaN`.
//! - **InvalidPeriod**: pwma: `period` is zero or exceeds data length.
//! - **PascalWeightsSumZero**: pwma: The computed Pascal weights sum to zero (unexpected).
//!
//! ## Returns
//! - **`Ok(PwmaOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(PwmaError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_kernel, detect_best_batch_kernel, alloc_with_nan_prefix, init_matrix_prefixes, make_uninit_matrix};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;
use std::mem::MaybeUninit;

impl<'a> AsRef<[f64]> for PwmaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            PwmaData::Slice(slice) => slice,
            PwmaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum PwmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct PwmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct PwmaParams {
    pub period: Option<usize>,
}

impl Default for PwmaParams {
    fn default() -> Self {
        Self { period: Some(5) }
    }
}

#[derive(Debug, Clone)]
pub struct PwmaInput<'a> {
    pub data: PwmaData<'a>,
    pub params: PwmaParams,
}

impl<'a> PwmaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: PwmaParams) -> Self {
        Self {
            data: PwmaData::Candles { candles: c, source: s },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: PwmaParams) -> Self {
        Self {
            data: PwmaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", PwmaParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(5)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct PwmaBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for PwmaBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl PwmaBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<PwmaOutput, PwmaError> {
        let p = PwmaParams { period: self.period };
        let i = PwmaInput::from_candles(c, "close", p);
        pwma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<PwmaOutput, PwmaError> {
        let p = PwmaParams { period: self.period };
        let i = PwmaInput::from_slice(d, p);
        pwma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<PwmaStream, PwmaError> {
        let p = PwmaParams { period: self.period };
        PwmaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum PwmaError {
    #[error("pwma: All values are NaN.")]
    AllValuesNaN,
    #[error("pwma: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("pwma: Pascal weights sum to zero for period = {period}")]
    PascalWeightsSumZero { period: usize },
}

#[inline]
pub fn pwma(input: &PwmaInput) -> Result<PwmaOutput, PwmaError> {
    pwma_with_kernel(input, Kernel::Auto)
}

pub fn pwma_with_kernel(input: &PwmaInput, kernel: Kernel) -> Result<PwmaOutput, PwmaError> {
    let data: &[f64] = match &input.data {
        PwmaData::Candles { candles, source } => source_type(candles, source),
        PwmaData::Slice(sl) => sl,
    };

    let first = data.iter().position(|x| !x.is_nan()).ok_or(PwmaError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(PwmaError::InvalidPeriod { period, data_len: len });
    }

    let weights = pascal_weights(period)?;

    let warm = first + period;
    let mut out = alloc_with_nan_prefix(len, warm);

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                pwma_scalar(data, &weights, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                pwma_avx2(data, &weights, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                pwma_avx512(data, &weights, period, first, &mut out)
            }
            _ => unreachable!(),
        }
    }

    Ok(PwmaOutput { values: out })
}

#[inline]
pub fn pwma_scalar(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    assert_eq!(weights.len(), period, "weights.len() must equal `period`");
    assert!(out.len() >= data.len(), "`out` must be at least as long as `data`");

    for i in (first + period - 1)..data.len() {
        let start = i + 1 - period;
        let window = &data[start..start + period];
        let mut sum = 0.0;
        for (d, w) in window.iter().zip(weights.iter()) {
            sum += d * w;
        }
        out[i] = sum;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn pwma_avx512(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        unsafe { pwma_avx512_short(data, weights, period, first, out) }
    } else {
        unsafe { pwma_avx512_long(data, weights, period, first, out) }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn pwma_avx2(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    pwma_scalar(data, weights, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn pwma_avx512_short(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    pwma_scalar(data, weights, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn pwma_avx512_long(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    pwma_scalar(data, weights, period, first, out)
}

#[derive(Debug, Clone)]
pub struct PwmaStream {
    period: usize,
    weights: Vec<f64>,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,
}

impl PwmaStream {
    pub fn try_new(params: PwmaParams) -> Result<Self, PwmaError> {
        let period = params.period.unwrap_or(5);
        if period == 0 {
            return Err(PwmaError::InvalidPeriod { period, data_len: 0 });
        }
        let weights = pascal_weights(period)?;
        Ok(Self {
            period,
            weights,
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
        sum
    }
}

#[derive(Clone, Debug)]
pub struct PwmaBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for PwmaBatchRange {
    fn default() -> Self {
        Self {
            period: (5, 30, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct PwmaBatchBuilder {
    range: PwmaBatchRange,
    kernel: Kernel,
}

impl PwmaBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<PwmaBatchOutput, PwmaError> {
        pwma_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<PwmaBatchOutput, PwmaError> {
        PwmaBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<PwmaBatchOutput, PwmaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<PwmaBatchOutput, PwmaError> {
        PwmaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub struct PwmaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<PwmaParams>,
    pub rows: usize,
    pub cols: usize,
}
impl PwmaBatchOutput {
    pub fn row_for_params(&self, p: &PwmaParams) -> Option<usize> {
        self.combos.iter().position(|c| c.period.unwrap_or(5) == p.period.unwrap_or(5))
    }
    pub fn values_for(&self, p: &PwmaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid_pwma(r: &PwmaBatchRange) -> Vec<PwmaParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(PwmaParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn pwma_batch_slice(
    data: &[f64],
    sweep: &PwmaBatchRange,
    kern: Kernel,
) -> Result<PwmaBatchOutput, PwmaError> {
    pwma_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn pwma_batch_par_slice(
    data: &[f64],
    sweep: &PwmaBatchRange,
    kern: Kernel,
) -> Result<PwmaBatchOutput, PwmaError> {
    pwma_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
pub fn pwma_batch_with_kernel(
    data: &[f64],
    sweep: &PwmaBatchRange,
    k: Kernel,
) -> Result<PwmaBatchOutput, PwmaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(PwmaError::InvalidPeriod {
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
    pwma_batch_par_slice(data, sweep, simd)
}

#[inline(always)]
fn pwma_batch_inner(
    data: &[f64],
    sweep: &PwmaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<PwmaBatchOutput, PwmaError> {
    let combos = expand_grid_pwma(sweep);
    if combos.is_empty() {
        return Err(PwmaError::InvalidPeriod { period: 0, data_len: 0 });
    }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(PwmaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(PwmaError::InvalidPeriod { period: max_p, data_len: data.len() });
    }
    let rows = combos.len();
    let cols = data.len();
    let mut weights = AVec::<f64>::with_capacity(CACHELINE_ALIGN, rows * max_p);
    weights.resize(rows * max_p, 0.0);
    for (row, prm) in combos.iter().enumerate() {
        let period = prm.period.unwrap();
        let row_weights = pascal_weights(period)?;
        for (i, w) in row_weights.iter().enumerate() {
            weights[row * max_p + i] = *w;
        }
    }
    let warm: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap()).collect();

    let mut raw = make_uninit_matrix(rows, cols);          // Vec<MaybeUninit<f64>>
    unsafe { init_matrix_prefixes(&mut raw, cols, &warm) }; // write NaN prefixes

    // --- closure that fills a single row -------------------------------------
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();
        let w_ptr  = weights.as_ptr().add(row * max_p);

        // reinterpret this row as &mut [f64]
        let out_row = core::slice::from_raw_parts_mut(
            dst_mu.as_mut_ptr() as *mut f64,
            dst_mu.len(),
        );

        match kern {
            Kernel::Scalar => pwma_row_scalar(data, first, period, max_p, w_ptr, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2   => pwma_row_avx2  (data, first, period, max_p, w_ptr, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => pwma_row_avx512(data, first, period, max_p, w_ptr, out_row),
            _ => unreachable!(),
        }
    };

    // --- run the rows in parallel or serial ----------------------------------
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

    // --- transmute to fully-initialised Vec<f64> ------------------------------
    let values: Vec<f64> = unsafe { std::mem::transmute(raw) };

    Ok(PwmaBatchOutput { values, combos, rows, cols })
}

#[inline(always)]
unsafe fn pwma_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    w_ptr: *const f64,
    out: &mut [f64],
) {
    for i in (first + period - 1)..data.len() {
        let start = i + 1 - period;
        let mut sum = 0.0;
        for k in 0..period {
            sum += *data.get_unchecked(start + k) * *w_ptr.add(k);
        }
        out[i] = sum;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
unsafe fn pwma_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    w_ptr: *const f64,
    out: &mut [f64],
) {
    pwma_row_scalar(data, first, period, stride, w_ptr, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn pwma_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    w_ptr: *const f64,
    out: &mut [f64],
) {
    if period <= 32 {
        pwma_row_avx512_short(data, first, period, stride, w_ptr, out);
    
        } else {
        pwma_row_avx512_long(data, first, period, stride, w_ptr, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
unsafe fn pwma_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    w_ptr: *const f64,
    out: &mut [f64],
) {
    pwma_row_scalar(data, first, period, stride, w_ptr, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
unsafe fn pwma_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    w_ptr: *const f64,
    out: &mut [f64],
) {
    pwma_row_scalar(data, first, period, stride, w_ptr, out)
}

#[inline]
fn pascal_weights(period: usize) -> Result<Vec<f64>, PwmaError> {
    if period == 0 {
        return Err(PwmaError::InvalidPeriod { period, data_len: 0 });
    }
    let n = period - 1;
    let mut row = Vec::with_capacity(period);
    for r in 0..=n {
        let c = combination_f64(n, r);
        row.push(c);
    }
    let sum: f64 = row.iter().sum();
    if sum == 0.0 {
        return Err(PwmaError::PascalWeightsSumZero { period });
    }
    for val in row.iter_mut() {
        *val /= sum;
    }
    Ok(row)
}

#[inline]
fn combination_f64(n: usize, r: usize) -> f64 {
    let r = r.min(n - r);
    if r == 0 { return 1.0; }
    let mut result = 1.0;
    for i in 0..r {
        result *= (n - i) as f64;
        result /= (i + 1) as f64;
    }
    result
}

// ---- Tests (macro parity, batch tests, kernel detection) ----

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_pwma_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = PwmaParams { period: None };
        let input = PwmaInput::from_candles(&candles, "close", default_params);
        let output = pwma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_pwma_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let expected_last_five = [59313.25, 59309.6875, 59249.3125, 59175.625, 59094.875];
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = PwmaInput::from_candles(&candles, "close", PwmaParams::default());
        let result = pwma_with_kernel(&input, kernel)?;
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-3,
                "[{}] PWMA {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_pwma_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = PwmaInput::with_default_candles(&candles);
        match input.data {
            PwmaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected PwmaData::Candles"),
        }
        let output = pwma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_pwma_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = PwmaParams { period: Some(0) };
        let input = PwmaInput::from_slice(&input_data, params);
        let res = pwma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] PWMA should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_pwma_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = PwmaParams { period: Some(10) };
        let input = PwmaInput::from_slice(&data_small, params);
        let res = pwma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] PWMA should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_pwma_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = PwmaParams { period: Some(5) };
        let input = PwmaInput::from_slice(&single_point, params);
        let res = pwma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] PWMA should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_pwma_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = PwmaParams { period: Some(5) };
        let first_input = PwmaInput::from_candles(&candles, "close", first_params);
        let first_result = pwma_with_kernel(&first_input, kernel)?;
        let second_params = PwmaParams { period: Some(3) };
        let second_input = PwmaInput::from_slice(&first_result.values, second_params);
        let second_result = pwma_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    fn check_pwma_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = PwmaInput::from_candles(&candles, "close", PwmaParams { period: Some(5) });
        let res = pwma_with_kernel(&input, kernel)?;
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

    fn check_pwma_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 5;
        let input = PwmaInput::from_candles(&candles, "close", PwmaParams { period: Some(period) });
        let batch_output = pwma_with_kernel(&input, kernel)?.values;
        let mut stream = PwmaStream::try_new(PwmaParams { period: Some(period) })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(val) => stream_values.push(val),
                None => stream_values.push(f64::NAN),
            }
        }
        assert_eq!(batch_output.len(), stream_values.len());
        for (i, (&b, &s)) in batch_output.iter().zip(stream_values.iter()).enumerate() {
            if b.is_nan() && s.is_nan() { continue; }
            let diff = (b - s).abs();
            assert!(diff < 1e-9, "[{}] PWMA streaming mismatch at idx {}: batch={}, stream={}", test_name, i, b, s);
        }
        Ok(())
    }

    macro_rules! generate_all_pwma_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(#[test] fn [<$test_fn _scalar_f64>]() { let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar); })*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(#[test] fn [<$test_fn _avx2_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2); })*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(#[test] fn [<$test_fn _avx512_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512); })*
            }
        }
    }
    generate_all_pwma_tests!(
        check_pwma_partial_params,
        check_pwma_accuracy,
        check_pwma_default_candles,
        check_pwma_zero_period,
        check_pwma_period_exceeds_length,
        check_pwma_very_small_dataset,
        check_pwma_reinput,
        check_pwma_nan_handling,
        check_pwma_streaming
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = PwmaBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;
        let def = PwmaParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        Ok(())
    }

    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste::paste! {
                #[test] fn [<$fn_name _scalar>]() { let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch); }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx2>]() { let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch); }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx512>]() { let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch); }
                #[test] fn [<$fn_name _auto_detect>]() { let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto); }
            }
        };
    }
    gen_batch_tests!(check_batch_default_row);
}
