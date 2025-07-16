//! # Chande Momentum Oscillator (CMO)
//!
//! A momentum oscillator that compares the sum of recent gains to recent losses over a given period.
//! The CMO oscillates between +100 and -100, with values near +100 indicating strong upward momentum and values near -100 indicating strong downward momentum.
//!
//! ## Parameters
//! - **period**: Window size (number of data points, default: 14).
//! - **source**: Candle field (e.g., `"close"`, default: `"close"`).
//!
//! ## Errors
//! - **EmptyData**: cmo: Input data slice is empty.
//! - **InvalidPeriod**: cmo: Period is zero or exceeds data length.
//! - **AllValuesNaN**: cmo: All input data values are `NaN`.
//! - **NotEnoughValidData**: cmo: Not enough valid data points for the requested period.
//!
//! ## Returns
//! - **`Ok(CmoOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(CmoError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

impl<'a> AsRef<[f64]> for CmoInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            CmoData::Slice(slice) => slice,
            CmoData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum CmoData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct CmoOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct CmoParams {
    pub period: Option<usize>,
}

impl Default for CmoParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct CmoInput<'a> {
    pub data: CmoData<'a>,
    pub params: CmoParams,
}

impl<'a> CmoInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: CmoParams) -> Self {
        Self {
            data: CmoData::Candles { candles: c, source: s },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: CmoParams) -> Self {
        Self {
            data: CmoData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", CmoParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct CmoBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for CmoBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl CmoBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<CmoOutput, CmoError> {
        let p = CmoParams { period: self.period };
        let i = CmoInput::from_candles(c, "close", p);
        cmo_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<CmoOutput, CmoError> {
        let p = CmoParams { period: self.period };
        let i = CmoInput::from_slice(d, p);
        cmo_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<CmoStream, CmoError> {
        let p = CmoParams { period: self.period };
        CmoStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum CmoError {
    #[error("cmo: Empty data provided.")]
    EmptyData,

    #[error("cmo: Invalid period: period={period}, data_len={data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("cmo: All values are NaN.")]
    AllValuesNaN,

    #[error("cmo: Not enough valid data: needed={needed}, valid={valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn cmo(input: &CmoInput) -> Result<CmoOutput, CmoError> {
    cmo_with_kernel(input, Kernel::Auto)
}

pub fn cmo_with_kernel(input: &CmoInput, kernel: Kernel) -> Result<CmoOutput, CmoError> {
    let data: &[f64] = match &input.data {
        CmoData::Candles { candles, source } => source_type(candles, source),
        CmoData::Slice(sl) => sl,
    };

    if data.is_empty() {
        return Err(CmoError::EmptyData);
    }

    let period = input.get_period();
    let len = data.len();

    if period == 0 || period > len {
        return Err(CmoError::InvalidPeriod { period, data_len: len });
    }

    let first = data.iter().position(|x| !x.is_nan()).ok_or(CmoError::AllValuesNaN)?;

    if (len - first) < period {
        return Err(CmoError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let mut out = alloc_with_nan_prefix(len, first + period);

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                cmo_scalar(data, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                cmo_avx2(data, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                cmo_avx512(data, period, first, &mut out)
            }
            _ => unreachable!(),
        }
    }
    Ok(CmoOutput { values: out })
}

#[inline]
pub fn cmo_scalar(
    data: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    let mut avg_gain = 0.0;
    let mut avg_loss = 0.0;
    let mut prev_price = data[first_valid];

    let start_loop = first_valid + 1;
    let init_end = first_valid + period;

    let period_f = period as f64;
    let period_m1 = (period - 1) as f64;
    let inv_period = 1.0 / period_f;

    for i in start_loop..data.len() {
        let curr = data[i];
        let diff = curr - prev_price;
        prev_price = curr;

        let abs_diff = diff.abs();
        let gain = 0.5 * (diff + abs_diff);
        let loss = 0.5 * (abs_diff - diff);

        if i <= init_end {
            avg_gain += gain;
            avg_loss += loss;
            if i == init_end {
                avg_gain *= inv_period;
                avg_loss *= inv_period;
                let sum_gl = avg_gain + avg_loss;
                out[i] = if sum_gl != 0.0 {
                    100.0 * ((avg_gain - avg_loss) / sum_gl)
                } else {
                    0.0
                };
            }
        } else {
            avg_gain *= period_m1;
            avg_loss *= period_m1;
            avg_gain += gain;
            avg_loss += loss;
            avg_gain *= inv_period;
            avg_loss *= inv_period;
            let sum_gl = avg_gain + avg_loss;
            out[i] = if sum_gl != 0.0 {
                100.0 * ((avg_gain - avg_loss) / sum_gl)
            } else {
                0.0
            };
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn cmo_avx512(
    data: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    unsafe {
        if period <= 32 {
            cmo_avx512_short(data, period, first_valid, out)
        
            } else {
            cmo_avx512_long(data, period, first_valid, out)
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn cmo_avx2(
    data: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    // AVX2 stub: use scalar
    cmo_scalar(data, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn cmo_avx512_short(
    data: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    // AVX512 short stub: use scalar
    cmo_scalar(data, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn cmo_avx512_long(
    data: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    // AVX512 long stub: use scalar
    cmo_scalar(data, period, first_valid, out)
}

#[inline(always)]
pub fn cmo_batch_with_kernel(
    data: &[f64],
    sweep: &CmoBatchRange,
    k: Kernel,
) -> Result<CmoBatchOutput, CmoError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(CmoError::InvalidPeriod {
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
    cmo_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct CmoBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for CmoBatchRange {
    fn default() -> Self {
        Self { period: (14, 40, 1) }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CmoBatchBuilder {
    range: CmoBatchRange,
    kernel: Kernel,
}

impl CmoBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<CmoBatchOutput, CmoError> {
        cmo_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<CmoBatchOutput, CmoError> {
        CmoBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<CmoBatchOutput, CmoError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<CmoBatchOutput, CmoError> {
        CmoBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct CmoBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<CmoParams>,
    pub rows: usize,
    pub cols: usize,
}

impl CmoBatchOutput {
    pub fn row_for_params(&self, p: &CmoParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(14) == p.period.unwrap_or(14)
        })
    }
    pub fn values_for(&self, p: &CmoParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &CmoBatchRange) -> Vec<CmoParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(CmoParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn cmo_batch_slice(
    data: &[f64],
    sweep: &CmoBatchRange,
    kern: Kernel,
) -> Result<CmoBatchOutput, CmoError> {
    cmo_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn cmo_batch_par_slice(
    data: &[f64],
    sweep: &CmoBatchRange,
    kern: Kernel,
) -> Result<CmoBatchOutput, CmoError> {
    cmo_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn cmo_batch_inner(
    data: &[f64],
    sweep: &CmoBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<CmoBatchOutput, CmoError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(CmoError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(CmoError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(CmoError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();
    
    // Step 1: Allocate uninitialized matrix
    let mut buf_mu = make_uninit_matrix(rows, cols);
    
    // Step 2: Calculate warmup periods for each row
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap())
        .collect();
    
    // Step 3: Initialize NaN prefixes for each row
    init_matrix_prefixes(&mut buf_mu, cols, &warm);
    
    // Step 4: Convert to mutable slice for computation
    let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
        )
    };

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        match kern {
            Kernel::Scalar => cmo_row_scalar(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => cmo_row_avx2(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => cmo_row_avx512(data, first, period, out_row),
            _ => unreachable!(),
        }
    };

    if parallel {


        #[cfg(not(target_arch = "wasm32"))] {


        out.par_chunks_mut(cols).enumerate().for_each(|(row, slice)| do_row(row, slice));


        }


        #[cfg(target_arch = "wasm32")] {


        for (row, slice) in out.chunks_mut(cols).enumerate() {


                    do_row(row, slice);


        }


        }
    } else {
        for (row, slice) in out.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }
    
    // Step 5: Reclaim as Vec<f64>
    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity()
        )
    };
    
    Ok(CmoBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn cmo_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    cmo_scalar(data, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn cmo_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    cmo_avx2(data, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn cmo_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        cmo_row_avx512_short(data, first, period, out);
    
        } else {
        cmo_row_avx512_long(data, first, period, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn cmo_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    cmo_avx512_short(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn cmo_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    cmo_avx512_long(data, period, first, out)
}

#[derive(Debug, Clone)]
pub struct CmoStream {
    period: usize,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,
    avg_gain: f64,
    avg_loss: f64,
    prev: f64,
    started: bool,
}

impl CmoStream {
    pub fn try_new(params: CmoParams) -> Result<Self, CmoError> {
        let period = params.period.unwrap_or(14);
        if period == 0 {
            return Err(CmoError::InvalidPeriod { period, data_len: 0 });
        }
        Ok(Self {
            period,
            buffer: alloc_with_nan_prefix(period, period),
            head: 0,
            filled: false,
            avg_gain: 0.0,
            avg_loss: 0.0,
            prev: 0.0,
            started: false,
        })
    }
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        if !self.started {
            self.prev = value;
            self.started = true;
            return None;
        }
        let diff = value - self.prev;
        self.prev = value;
        let abs_diff = diff.abs();
        let gain = 0.5 * (diff + abs_diff);
        let loss = 0.5 * (abs_diff - diff);

        if !self.filled {
            self.avg_gain += gain;
            self.avg_loss += loss;
            self.head += 1;
            if self.head == self.period {
                self.avg_gain /= self.period as f64;
                self.avg_loss /= self.period as f64;
                self.filled = true;
                let sum_gl = self.avg_gain + self.avg_loss;
                return Some(if sum_gl != 0.0 {
                    100.0 * ((self.avg_gain - self.avg_loss) / sum_gl)
                
                    } else {
                    0.0
                });
            }
            return None;
        }
        let period_f = self.period as f64;
        let period_m1 = (self.period - 1) as f64;
        self.avg_gain *= period_m1;
        self.avg_loss *= period_m1;
        self.avg_gain += gain;
        self.avg_loss += loss;
        self.avg_gain /= period_f;
        self.avg_loss /= period_f;
        let sum_gl = self.avg_gain + self.avg_loss;
        Some(if sum_gl != 0.0 {
            100.0 * ((self.avg_gain - self.avg_loss) / sum_gl)
        } else {
            0.0
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_cmo_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = CmoParams { period: None };
        let input = CmoInput::from_candles(&candles, "close", default_params);
        let output = cmo_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        let params_10 = CmoParams { period: Some(10) };
        let input_10 = CmoInput::from_candles(&candles, "hl2", params_10);
        let output_10 = cmo_with_kernel(&input_10, kernel)?;
        assert_eq!(output_10.values.len(), candles.close.len());
        Ok(())
    }

    fn check_cmo_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = CmoParams { period: Some(14) };
        let input = CmoInput::from_candles(&candles, "close", params);
        let cmo_result = cmo_with_kernel(&input, kernel)?;
        let expected_last_five = [
            -13.152504931406101,
            -14.649876201213106,
            -16.760170709240303,
            -14.274505732779227,
            -21.984038127126716,
        ];
        let start_idx = cmo_result.values.len() - 5;
        let last_five = &cmo_result.values[start_idx..];
        for (i, &actual) in last_five.iter().enumerate() {
            let expected = expected_last_five[i];
            assert!(
                (actual - expected).abs() < 1e-6,
                "[{}] CMO mismatch at final 5 index {}: expected {}, got {}",
                test_name,
                i,
                expected,
                actual
            );
        }
        Ok(())
    }

    fn check_cmo_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = CmoInput::with_default_candles(&candles);
        match input.data {
            CmoData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected CmoData::Candles variant"),
        }
        let output = cmo_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_cmo_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [10.0, 20.0, 30.0];
        let params = CmoParams { period: Some(0) };
        let input = CmoInput::from_slice(&data, params);
        let result = cmo_with_kernel(&input, kernel);
        assert!(result.is_err(), "[{}] Expected error for period=0", test_name);
        Ok(())
    }

    fn check_cmo_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [10.0, 20.0, 30.0];
        let params = CmoParams { period: Some(10) };
        let input = CmoInput::from_slice(&data, params);
        let result = cmo_with_kernel(&input, kernel);
        assert!(result.is_err(), "[{}] Expected error for period>data.len()", test_name);
        Ok(())
    }

    fn check_cmo_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single = [42.0];
        let params = CmoParams { period: Some(14) };
        let input = CmoInput::from_slice(&single, params);
        let result = cmo_with_kernel(&input, kernel);
        assert!(result.is_err(), "[{}] Expected error for insufficient data", test_name);
        Ok(())
    }

    fn check_cmo_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = CmoParams { period: Some(14) };
        let first_input = CmoInput::from_candles(&candles, "close", first_params);
        let first_result = cmo_with_kernel(&first_input, kernel)?;
        let second_params = CmoParams { period: Some(14) };
        let second_input = CmoInput::from_slice(&first_result.values, second_params);
        let second_result = cmo_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        for i in 28..second_result.values.len() {
            assert!(
                !second_result.values[i].is_nan(),
                "[{}] Expected no NaN after index 28, found NaN at {}",
                test_name,
                i
            );
        }
        Ok(())
    }

    // Check for poison values in single output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_cmo_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        
        // Test with multiple parameter combinations to increase coverage
        let test_periods = vec![7, 14, 21, 28];
        
        for period in test_periods {
            let params = CmoParams { period: Some(period) };
            let input = CmoInput::from_candles(&candles, "close", params);
            let output = cmo_with_kernel(&input, kernel)?;
            
            // Check every value for poison patterns
            for (i, &val) in output.values.iter().enumerate() {
                // Skip NaN values as they're expected in the warmup period
                if val.is_nan() {
                    continue;
                }
                
                let bits = val.to_bits();
                
                // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} with period {}",
                        test_name, val, bits, i, period
                    );
                }
                
                // Check for init_matrix_prefixes poison (0x22222222_22222222)
                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} with period {}",
                        test_name, val, bits, i, period
                    );
                }
                
                // Check for make_uninit_matrix poison (0x33333333_33333333)
                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} with period {}",
                        test_name, val, bits, i, period
                    );
                }
            }
        }
        
        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_cmo_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    macro_rules! generate_all_cmo_tests {
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

    generate_all_cmo_tests!(
        check_cmo_partial_params,
        check_cmo_accuracy,
        check_cmo_default_candles,
        check_cmo_zero_period,
        check_cmo_period_exceeds_length,
        check_cmo_very_small_dataset,
        check_cmo_reinput,
        check_cmo_no_poison
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = CmoBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;
        let def = CmoParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        Ok(())
    }

    // Check for poison values in batch output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        
        // Test batch with multiple parameter combinations
        let output = CmoBatchBuilder::new()
            .kernel(kernel)
            .period_range(7, 28, 7)  // Test periods: 7, 14, 21, 28
            .apply_candles(&c, "close")?;
        
        // Check every value in the entire batch matrix for poison patterns
        for (idx, &val) in output.values.iter().enumerate() {
            // Skip NaN values as they're expected in warmup periods
            if val.is_nan() {
                continue;
            }
            
            let bits = val.to_bits();
            let row = idx / output.cols;
            let col = idx % output.cols;
            
            // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
            if bits == 0x11111111_11111111 {
                panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {})",
                    test, val, bits, row, col, idx
                );
            }
            
            // Check for init_matrix_prefixes poison (0x22222222_22222222)
            if bits == 0x22222222_22222222 {
                panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {})",
                    test, val, bits, row, col, idx
                );
            }
            
            // Check for make_uninit_matrix poison (0x33333333_33333333)
            if bits == 0x33333333_33333333 {
                panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {})",
                    test, val, bits, row, col, idx
                );
            }
        }
        
        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
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
    gen_batch_tests!(check_batch_no_poison);
}
