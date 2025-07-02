//! # Wilder’s Moving Average (Wilders)
//!
//! A moving average introduced by J. Welles Wilder, commonly used in indicators such as
//! the Average Directional Index (ADX). Places a heavier emphasis on new data than an SMA,
//! but less so than an EMA. Features include kernel selection, batch calculation, AVX stubs,
//! and streaming in parity with alma.rs.
//!
//! ## Parameters
//! - **period**: Window size (number of data points).
//!
//! ## Errors
//! - **AllValuesNaN**: wilders: All input data values are `NaN`.
//! - **InvalidPeriod**: wilders: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: wilders: Not enough valid data points for the requested `period`.
//!
//! ## Returns
//! - **`Ok(WildersOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(WildersError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel, alloc_with_nan_prefix, init_matrix_prefixes, make_uninit_matrix};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;
use std::mem::MaybeUninit;

// --- Input/Output/Params/Builder Structs ---

#[derive(Debug, Clone)]
pub enum WildersData<'a> {
    Candles { candles: &'a Candles, source: &'a str },
    Slice(&'a [f64]),
}

impl<'a> AsRef<[f64]> for WildersInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            WildersData::Slice(slice) => slice,
            WildersData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub struct WildersOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct WildersParams {
    pub period: Option<usize>,
}

impl Default for WildersParams {
    fn default() -> Self {
        Self { period: Some(5) }
    }
}

#[derive(Debug, Clone)]
pub struct WildersInput<'a> {
    pub data: WildersData<'a>,
    pub params: WildersParams,
}

impl<'a> WildersInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: WildersParams) -> Self {
        Self { data: WildersData::Candles { candles: c, source: s }, params: p }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: WildersParams) -> Self {
        Self { data: WildersData::Slice(sl), params: p }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", WildersParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(5)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct WildersBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for WildersBuilder {
    fn default() -> Self {
        Self { period: None, kernel: Kernel::Auto }
    }
}

impl WildersBuilder {
    #[inline(always)]
    pub fn new() -> Self { Self::default() }
    #[inline(always)]
    pub fn period(mut self, n: usize) -> Self {
        self.period = Some(n); self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k; self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<WildersOutput, WildersError> {
        let p = WildersParams { period: self.period };
        let i = WildersInput::from_candles(c, "close", p);
        wilders_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<WildersOutput, WildersError> {
        let p = WildersParams { period: self.period };
        let i = WildersInput::from_slice(d, p);
        wilders_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<WildersStream, WildersError> {
        let p = WildersParams { period: self.period };
        WildersStream::try_new(p)
    }
}

// --- Error Handling ---

#[derive(Debug, Error)]
pub enum WildersError {
    #[error("wilders: All values are NaN.")]
    AllValuesNaN,
    #[error("wilders: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("wilders: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

// --- API parity main function & kernel dispatch ---

#[inline]
pub fn wilders(input: &WildersInput) -> Result<WildersOutput, WildersError> {
    wilders_with_kernel(input, Kernel::Auto)
}

pub fn wilders_with_kernel(input: &WildersInput, kernel: Kernel) -> Result<WildersOutput, WildersError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    let period = input.get_period();

    let first = data.iter().position(|x| !x.is_nan()).ok_or(WildersError::AllValuesNaN)?;
    if period == 0 || period > len {
        return Err(WildersError::InvalidPeriod { period, data_len: len });
    }
    if (len - first) < period {
        return Err(WildersError::NotEnoughValidData { needed: period, valid: len - first });
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
                wilders_scalar(data, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                wilders_avx2(data, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                wilders_avx512(data, period, first, &mut out)
            }
            _ => unreachable!(),
        }
    }
    Ok(WildersOutput { values: out })
}

// --- Scalar calculation (core logic) ---

#[inline]
pub fn wilders_scalar(
    data: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    let mut sum = 0.0;
    for i in 0..period {
        sum += data[first_valid + i];
    }
    let wma_start = first_valid + period - 1;
    let mut val = sum / (period as f64);
    out[wma_start] = val;
    let alpha = 1.0 / (period as f64);
    for i in (wma_start + 1)..data.len() {
        val = (data[i] - val) * alpha + val;
        out[i] = val;
    }
}

// --- AVX2 and AVX512 stubs (API parity, always point to scalar) ---

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn wilders_avx2(
    data: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    wilders_scalar(data, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn wilders_avx512(
    data: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        wilders_avx512_short(data, period, first_valid, out)
    
        } else {
        wilders_avx512_long(data, period, first_valid, out)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn wilders_avx512_short(
    data: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    wilders_scalar(data, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn wilders_avx512_long(
    data: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    wilders_scalar(data, period, first_valid, out)
}

// --- Streaming (WildersStream) ---

#[derive(Debug, Clone)]
pub struct WildersStream {
    period: usize,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,
    last: f64,
    started: bool,
}

impl WildersStream {
    pub fn try_new(params: WildersParams) -> Result<Self, WildersError> {
        let period = params.period.unwrap_or(5);
        if period == 0 {
            return Err(WildersError::InvalidPeriod { period, data_len: 0 });
        }
        Ok(Self {
            period,
            buffer: vec![f64::NAN; period],
            head: 0,
            filled: false,
            last: f64::NAN,
            started: false,
        })
    }
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.buffer[self.head] = value;
        self.head = (self.head + 1) % self.period;
        if !self.filled && self.head == 0 { self.filled = true; }
        if !self.filled { return None; }
        if !self.started {
            let sum: f64 = self.buffer.iter().copied().sum();
            self.last = sum / (self.period as f64);
            self.started = true;
            Some(self.last)
        
            } else {
            self.last = (value - self.last) / (self.period as f64) + self.last;
            Some(self.last)
        }
    }
}

// --- Batch Ranges, Builder, Output, Batch Apply ---

#[derive(Clone, Debug)]
pub struct WildersBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for WildersBatchRange {
    fn default() -> Self {
        Self { period: (5, 24, 1) }
    }
}

#[derive(Clone, Debug, Default)]
pub struct WildersBatchBuilder {
    range: WildersBatchRange,
    kernel: Kernel,
}

impl WildersBatchBuilder {
    pub fn new() -> Self { Self::default() }
    pub fn kernel(mut self, k: Kernel) -> Self { self.kernel = k; self }
    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.period = (start, end, step); self
    }
    pub fn period_static(mut self, p: usize) -> Self {
        self.range.period = (p, p, 0); self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<WildersBatchOutput, WildersError> {
        wilders_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<WildersBatchOutput, WildersError> {
        WildersBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<WildersBatchOutput, WildersError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<WildersBatchOutput, WildersError> {
        WildersBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn wilders_batch_with_kernel(
    data: &[f64],
    sweep: &WildersBatchRange,
    k: Kernel,
) -> Result<WildersBatchOutput, WildersError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(WildersError::InvalidPeriod { period: 0, data_len: 0 }),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    wilders_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct WildersBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<WildersParams>,
    pub rows: usize,
    pub cols: usize,
}
impl WildersBatchOutput {
    pub fn row_for_params(&self, p: &WildersParams) -> Option<usize> {
        self.combos.iter().position(|c| c.period.unwrap_or(5) == p.period.unwrap_or(5))
    }
    pub fn values_for(&self, p: &WildersParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

// --- Batch Internals & Grid Expansion ---

#[inline(always)]
fn expand_grid(r: &WildersBatchRange) -> Vec<WildersParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end { return vec![start]; }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(WildersParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn wilders_batch_slice(
    data: &[f64],
    sweep: &WildersBatchRange,
    kern: Kernel,
) -> Result<WildersBatchOutput, WildersError> {
    wilders_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn wilders_batch_par_slice(
    data: &[f64],
    sweep: &WildersBatchRange,
    kern: Kernel,
) -> Result<WildersBatchOutput, WildersError> {
    wilders_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn wilders_batch_inner(
    data: &[f64],
    sweep: &WildersBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<WildersBatchOutput, WildersError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(WildersError::InvalidPeriod { period: 0, data_len: 0 });
    }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(WildersError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(WildersError::NotEnoughValidData { needed: max_p, valid: data.len() - first });
    }
    let rows = combos.len();
    let cols = data.len();
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();

    // -----------------------------------------
    // 2. allocate rows×cols uninitialised
    //    and paint the NaN prefixes
    // -----------------------------------------
    let mut raw = make_uninit_matrix(rows, cols);
    unsafe { init_matrix_prefixes(&mut raw, cols, &warm) };

    // -----------------------------------------
    // 3. helper that fills a single row
    // -----------------------------------------
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();

        // transmute this row to &mut [f64]
        let out_row = std::slice::from_raw_parts_mut(
            dst_mu.as_mut_ptr() as *mut f64,
            dst_mu.len(),
        );

        match kern {
            Kernel::Scalar => wilders_row_scalar(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2   => wilders_row_avx2  (data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => wilders_row_avx512(data, first, period, out_row),
            _ => unreachable!(),
        }
    };

    // -----------------------------------------
    // 4. run every row
    // -----------------------------------------
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

    // -----------------------------------------
    // 5. convert to Vec<f64> now that everything
    //    has been fully initialised
    // -----------------------------------------
    let values: Vec<f64> = unsafe { std::mem::transmute(raw) };
    Ok(WildersBatchOutput { values, combos, rows, cols })
}

// --- Row functions for batch (all just call scalar or AVX stubs) ---

#[inline(always)]
pub unsafe fn wilders_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    wilders_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn wilders_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    wilders_avx2(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn wilders_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        wilders_row_avx512_short(data, first, period, out)
    
        } else {
        wilders_row_avx512_long(data, first, period, out)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn wilders_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    wilders_avx512_short(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn wilders_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    wilders_avx512_long(data, period, first, out)
}

// --- Unit Tests (feature parity with alma.rs) ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_wilders_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = WildersParams { period: None };
        let input = WildersInput::from_candles(&candles, "close", default_params);
        let output = wilders_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_wilders_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = WildersInput::from_candles(&candles, "close", WildersParams { period: Some(5) });
        let result = wilders_with_kernel(&input, kernel)?;
        let expected_last_five = [
            59302.18156619092,
            59277.94525295273,
            59230.15620236219,
            59215.12496188975,
            59103.0999695118,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(diff < 1e-8, "[{}] Wilders {:?} mismatch at idx {}: got {}, expected {}", test_name, kernel, i, val, expected_last_five[i]);
        }
        Ok(())
    }

    fn check_wilders_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = WildersInput::with_default_candles(&candles);
        match input.data {
            WildersData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected WildersData::Candles"),
        }
        let output = wilders_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_wilders_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = WildersParams { period: Some(0) };
        let input = WildersInput::from_slice(&input_data, params);
        let res = wilders_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] Wilders should fail with zero period", test_name);
        Ok(())
    }

    fn check_wilders_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = WildersParams { period: Some(10) };
        let input = WildersInput::from_slice(&data_small, params);
        let res = wilders_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] Wilders should fail with period exceeding length", test_name);
        Ok(())
    }

    fn check_wilders_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = WildersParams { period: Some(1) };
        let input = WildersInput::from_slice(&single_point, params);
        let res = wilders_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), single_point.len());
        Ok(())
    }

    fn check_wilders_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = WildersParams { period: Some(5) };
        let first_input = WildersInput::from_candles(&candles, "close", first_params);
        let first_result = wilders_with_kernel(&first_input, kernel)?;

        let second_params = WildersParams { period: Some(10) };
        let second_input = WildersInput::from_slice(&first_result.values, second_params);
        let second_result = wilders_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    fn check_wilders_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = WildersInput::from_candles(&candles, "close", WildersParams { period: Some(5) });
        let res = wilders_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 240 {
            for (i, &val) in res.values[240..].iter().enumerate() {
                assert!(!val.is_nan(), "[{}] Found unexpected NaN at out-index {}", test_name, 240 + i);
            }
        }
        Ok(())
    }

    fn check_wilders_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let period = 5;
        let input = WildersInput::from_candles(&candles, "close", WildersParams { period: Some(period) });
        let batch_output = wilders_with_kernel(&input, kernel)?.values;

        let mut stream = WildersStream::try_new(WildersParams { period: Some(period) })?;
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
            assert!(diff < 1e-9, "[{}] Wilders streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}", test_name, i, b, s, diff);
        }
        Ok(())
    }

    macro_rules! generate_all_wilders_tests {
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
    generate_all_wilders_tests!(
        check_wilders_partial_params,
        check_wilders_accuracy,
        check_wilders_default_candles,
        check_wilders_zero_period,
        check_wilders_period_exceeds_length,
        check_wilders_very_small_dataset,
        check_wilders_reinput,
        check_wilders_nan_handling,
        check_wilders_streaming
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = WildersBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = WildersParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        // Last five expected Wilder’s values for period = 5
        let expected = [
            59302.18156619092,
            59277.94525295273,
            59230.15620236219,
            59215.12496188975,
            59103.0999695118,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-8,
                "[{}] default-row mismatch at idx {}: {} vs {:?}",
                test,
                i,
                v,
                expected
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
