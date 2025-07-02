//! # Normalized Moving Average (NMA)
//!
//! A technique that computes an adaptive moving average by transforming input
//! values into log space and weighting differences between consecutive values.
//! The weighting ratio depends on a series of square-root increments. This design
//! aims to normalize large price changes without oversmoothing small fluctuations.
//!
//! ## Parameters
//! - **period**: Window size (number of data points, default: 40)
//!
//! ## Errors
//! - **AllValuesNaN**: nma: All input data values are `NaN`.
//! - **InvalidPeriod**: nma: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: nma: Not enough valid data points for the requested `period`.
//!
//! ## Returns
//! - **`Ok(NmaOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(NmaError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes, make_uninit_matrix, alloc_with_nan_prefix};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;
use std::mem::MaybeUninit;

impl<'a> AsRef<[f64]> for NmaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            NmaData::Slice(slice) => slice,
            NmaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum NmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct NmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct NmaParams {
    pub period: Option<usize>,
}

impl Default for NmaParams {
    fn default() -> Self {
        Self { period: Some(40) }
    }
}

#[derive(Debug, Clone)]
pub struct NmaInput<'a> {
    pub data: NmaData<'a>,
    pub params: NmaParams,
}

impl<'a> NmaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: NmaParams) -> Self {
        Self {
            data: NmaData::Candles { candles: c, source: s },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: NmaParams) -> Self {
        Self {
            data: NmaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", NmaParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(40)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct NmaBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for NmaBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl NmaBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<NmaOutput, NmaError> {
        let p = NmaParams {
            period: self.period,
        };
        let i = NmaInput::from_candles(c, "close", p);
        nma_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<NmaOutput, NmaError> {
        let p = NmaParams {
            period: self.period,
        };
        let i = NmaInput::from_slice(d, p);
        nma_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<NmaStream, NmaError> {
        let p = NmaParams {
            period: self.period,
        };
        NmaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum NmaError {
    #[error("nma: All values are NaN.")]
    AllValuesNaN,
    #[error("nma: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("nma: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn nma(input: &NmaInput) -> Result<NmaOutput, NmaError> {
    nma_with_kernel(input, Kernel::Auto)
}

pub fn nma_with_kernel(input: &NmaInput, kernel: Kernel) -> Result<NmaOutput, NmaError> {
    let data: &[f64] = match &input.data {
        NmaData::Candles { candles, source } => source_type(candles, source),
        NmaData::Slice(sl) => sl,
    };

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(NmaError::AllValuesNaN)?;

    let len = data.len();
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(NmaError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < (period + 1) {
        return Err(NmaError::NotEnoughValidData {
            needed: period + 1,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let warm  = first + period;
    let mut out = alloc_with_nan_prefix(len, warm);

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                nma_scalar(data, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                nma_avx2(data, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                nma_avx512(data, period, first, &mut out)
            }
            _ => unreachable!(),
        }
    }

    Ok(NmaOutput { values: out })
}

#[inline]
pub fn nma_scalar(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    let len = data.len();

    let mut ln_values = Vec::with_capacity(len);
    ln_values.extend(data.iter().map(|&val| {
        let clamped = val.max(1e-10);
        clamped.ln() * 1000.0
    }));

    let mut sqrt_diffs = Vec::with_capacity(period);
    for i in 0..period {
        let s0 = (i as f64).sqrt();
        let s1 = ((i + 1) as f64).sqrt();
        sqrt_diffs.push(s1 - s0);
    }

    for j in (first + period)..len {
        let mut num = 0.0;
        let mut denom = 0.0;

        for i in 0..period {
            let oi = (ln_values[j - i] - ln_values[j - i - 1]).abs();
            num += oi * sqrt_diffs[i];
            denom += oi;
        }

        let ratio = if denom == 0.0 { 0.0 } else { num / denom };

        let i = period - 1;
        out[j] = data[j - i] * ratio + data[j - i - 1] * (1.0 - ratio);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn nma_avx512(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    unsafe { nma_scalar(data, period, first, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn nma_avx2(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    unsafe { nma_scalar(data, period, first, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn nma_avx512_short(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    nma_avx512(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn nma_avx512_long(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    nma_avx512(data, period, first, out)
}

#[inline(always)]
pub fn nma_batch_with_kernel(
    data: &[f64],
    sweep: &NmaBatchRange,
    k: Kernel,
) -> Result<NmaBatchOutput, NmaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(NmaError::InvalidPeriod {
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
    nma_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct NmaBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for NmaBatchRange {
    fn default() -> Self {
        Self {
            period: (40, 100, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct NmaBatchBuilder {
    range: NmaBatchRange,
    kernel: Kernel,
}

impl NmaBatchBuilder {
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

    pub fn apply_slice(self, data: &[f64]) -> Result<NmaBatchOutput, NmaError> {
        nma_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<NmaBatchOutput, NmaError> {
        NmaBatchBuilder::new().kernel(k).apply_slice(data)
    }

    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<NmaBatchOutput, NmaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }

    pub fn with_default_candles(c: &Candles) -> Result<NmaBatchOutput, NmaError> {
        NmaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct NmaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<NmaParams>,
    pub rows: usize,
    pub cols: usize,
}

impl NmaBatchOutput {
    pub fn row_for_params(&self, p: &NmaParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(40) == p.period.unwrap_or(40)
        })
    }

    pub fn values_for(&self, p: &NmaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &NmaBatchRange) -> Vec<NmaParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);

    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(NmaParams {
            period: Some(p),
        });
    }
    out
}

#[inline(always)]
pub fn nma_batch_slice(
    data: &[f64],
    sweep: &NmaBatchRange,
    kern: Kernel,
) -> Result<NmaBatchOutput, NmaError> {
    nma_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn nma_batch_par_slice(
    data: &[f64],
    sweep: &NmaBatchRange,
    kern: Kernel,
) -> Result<NmaBatchOutput, NmaError> {
    nma_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn nma_batch_inner(
    data: &[f64],
    sweep: &NmaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<NmaBatchOutput, NmaError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(NmaError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(NmaError::AllValuesNaN)?;
    let max_p = combos
        .iter()
        .map(|c| c.period.unwrap())
        .max()
        .unwrap();
    if data.len() - first < max_p + 1 {
        return Err(NmaError::NotEnoughValidData {
            needed: max_p + 1,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();

    let warm: Vec<usize> = combos.iter()
        .map(|c| first + c.period.unwrap())
        .collect();

    let mut raw = make_uninit_matrix(rows, cols);
    unsafe { init_matrix_prefixes(&mut raw, cols, &warm) };

    // ---------------------------------------------------------------------
    // 2. closure that fills one row (works with MaybeUninit<f64>)
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();

        // cast just this row to &mut [f64] so we can call the usual kernel
        let out_row = core::slice::from_raw_parts_mut(
            dst_mu.as_mut_ptr() as *mut f64,
            dst_mu.len(),
        );

        match kern {
            Kernel::Scalar     => nma_row_scalar(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2       => nma_row_avx2   (data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512     => nma_row_avx512 (data, first, period, out_row),
            _ => unreachable!(),
        }
    };

    // ---------------------------------------------------------------------
    // 3. run every row, writing directly into `raw`
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
        }
    } else {
        for (row, slice) in raw.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    // ---------------------------------------------------------------------
    // 4. everything is now initialised – transmute to Vec<f64>
    let values: Vec<f64> = unsafe { std::mem::transmute(raw) };

    Ok(NmaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn nma_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    nma_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn nma_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    nma_row_scalar(data, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn nma_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        nma_row_avx512_short(data, first, period, out);
    
        } else {
        nma_row_avx512_long(data, first, period, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn nma_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    nma_row_scalar(data, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn nma_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    nma_row_scalar(data, first, period, out)
}

#[derive(Debug, Clone)]
pub struct NmaStream {
    period: usize,
    ln_values: Vec<f64>,
    sqrt_diffs: Vec<f64>,
    buffer: Vec<f64>,
    ln_buffer: Vec<f64>,
    head: usize,
    filled: bool,
}

impl NmaStream {
    pub fn try_new(params: NmaParams) -> Result<Self, NmaError> {
        let period = params.period.unwrap_or(40);
        if period == 0 {
            return Err(NmaError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let mut sqrt_diffs = Vec::with_capacity(period);
        for i in 0..period {
            let s0 = (i as f64).sqrt();
            let s1 = ((i + 1) as f64).sqrt();
            sqrt_diffs.push(s1 - s0);
        }
        Ok(Self {
            period,
            ln_values: vec![f64::NAN; period + 1],
            sqrt_diffs,
            buffer: vec![f64::NAN; period + 1],
            ln_buffer: vec![f64::NAN; period + 1],
            head: 0,
            filled: false,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        let ln_val = value.max(1e-10).ln() * 1000.0;
        self.buffer[self.head] = value;
        self.ln_buffer[self.head] = ln_val;
        self.head = (self.head + 1) % (self.period + 1);

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
        let mut num = 0.0;
        let mut denom = 0.0;
        let mut idx = self.head;
        for i in 0..self.period {
            let curr = self.ln_buffer[(idx + self.period) % (self.period + 1)];
            let prev = self.ln_buffer[(idx + self.period - 1) % (self.period + 1)];
            let oi = (curr - prev).abs();
            num += oi * self.sqrt_diffs[i];
            denom += oi;
            idx = (idx + self.period) % (self.period + 1);
        }
        let ratio = if denom == 0.0 { 0.0 } else { num / denom };
        let val_idx = (self.head + 1) % (self.period + 1);
        let i = self.period - 1;
        let x1 = self.buffer[(val_idx + self.period - i) % (self.period + 1)];
        let x2 = self.buffer[(val_idx + self.period - i - 1) % (self.period + 1)];
        x1 * ratio + x2 * (1.0 - ratio)
    }
}

// Expand grid for batch
#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_nma_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = NmaParams { period: None };
        let input = NmaInput::from_candles(&candles, "close", default_params);
        let output = nma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_nma_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = NmaInput::from_candles(&candles, "close", NmaParams::default());
        let nma_result = nma_with_kernel(&input, kernel)?;

        let expected_last_five_nma = [
            64320.486018271724,
            64227.95719984426,
            64180.9249333126,
            63966.35530620797,
            64039.04719192334,
        ];
        let start_index = nma_result.values.len() - 5;
        let result_last_five_nma = &nma_result.values[start_index..];
        for (i, &value) in result_last_five_nma.iter().enumerate() {
            let expected_value = expected_last_five_nma[i];
            assert!(
                (value - expected_value).abs() < 1e-3,
                "[{}] NMA value mismatch at last-5 index {}: expected {}, got {}",
                test_name,
                i,
                expected_value,
                value
            );
        }
        Ok(())
    }

    fn check_nma_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = NmaInput::with_default_candles(&candles);
        match input.data {
            NmaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected NmaData::Candles"),
        }
        let output = nma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_nma_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = NmaParams { period: Some(0) };
        let input = NmaInput::from_slice(&input_data, params);
        let res = nma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] NMA should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_nma_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = NmaParams { period: Some(10) };
        let input = NmaInput::from_slice(&data_small, params);
        let res = nma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] NMA should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_nma_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = NmaParams { period: Some(40) };
        let input = NmaInput::from_slice(&single_point, params);
        let res = nma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] NMA should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_nma_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = NmaParams { period: Some(40) };
        let first_input = NmaInput::from_candles(&candles, "close", first_params);
        let first_result = nma_with_kernel(&first_input, kernel)?;
        let second_params = NmaParams { period: Some(20) };
        let second_input = NmaInput::from_slice(&first_result.values, second_params);
        let second_result = nma_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        if second_result.values.len() > 240 {
            for i in 240..second_result.values.len() {
                assert!(second_result.values[i].is_finite());
            }
        }
        Ok(())
    }

    fn check_nma_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = NmaInput::from_candles(
            &candles,
            "close",
            NmaParams { period: Some(40) },
        );
        let res = nma_with_kernel(&input, kernel)?;
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

    macro_rules! generate_all_nma_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(#[test]
                fn [<$test_fn _scalar_f64>]() {
                    let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar);
                })*
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

    generate_all_nma_tests!(
        check_nma_partial_params,
        check_nma_accuracy,
        check_nma_default_candles,
        check_nma_zero_period,
        check_nma_period_exceeds_length,
        check_nma_very_small_dataset,
        check_nma_reinput,
        check_nma_nan_handling
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = NmaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = NmaParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            64320.486018271724,
            64227.95719984426,
            64180.9249333126,
            63966.35530620797,
            64039.04719192334,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-3,
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
