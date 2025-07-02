//! # Square Root Weighted Moving Average (SRWMA)
//!
//! A moving average variant that assigns weights proportional to the square root
//! of the distance from the current bar. This approach provides a moderate
//! emphasis on more recent data while still accounting for older points, thereby
//! reducing noise without excessively lagging.
//!
//! ## Parameters
//! - **period**: The look-back window size used for weighting (defaults to 14).
//!
//! ## Errors
//! - **AllValuesNaN**: srwma: All input data values are `NaN`.
//! - **InvalidPeriod**: srwma: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: srwma: Not enough valid data points for the requested `period`.
//!
//! ## Returns
//! - **`Ok(SrwmaOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(SrwmaError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel, alloc_with_nan_prefix, init_matrix_prefixes, make_uninit_matrix};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use thiserror::Error;
use std::convert::AsRef;
use std::mem::MaybeUninit;

#[derive(Debug, Clone)]
pub enum SrwmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

impl<'a> AsRef<[f64]> for SrwmaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            SrwmaData::Slice(slice) => slice,
            SrwmaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SrwmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct SrwmaParams {
    pub period: Option<usize>,
}

impl Default for SrwmaParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct SrwmaInput<'a> {
    pub data: SrwmaData<'a>,
    pub params: SrwmaParams,
}

impl<'a> SrwmaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: SrwmaParams) -> Self {
        Self {
            data: SrwmaData::Candles { candles: c, source: s },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: SrwmaParams) -> Self {
        Self {
            data: SrwmaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", SrwmaParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SrwmaBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for SrwmaBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl SrwmaBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<SrwmaOutput, SrwmaError> {
        let p = SrwmaParams {
            period: self.period,
        };
        let i = SrwmaInput::from_candles(c, "close", p);
        srwma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<SrwmaOutput, SrwmaError> {
        let p = SrwmaParams {
            period: self.period,
        };
        let i = SrwmaInput::from_slice(d, p);
        srwma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<SrwmaStream, SrwmaError> {
        let p = SrwmaParams {
            period: self.period,
        };
        SrwmaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum SrwmaError {
    #[error("srwma: All values are NaN.")]
    AllValuesNaN,
    #[error("srwma: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("srwma: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn srwma(input: &SrwmaInput) -> Result<SrwmaOutput, SrwmaError> {
    srwma_with_kernel(input, Kernel::Auto)
}

pub fn srwma_with_kernel(
    input: &SrwmaInput,
    kernel: Kernel,
) -> Result<SrwmaOutput, SrwmaError> {
    let data: &[f64] = match &input.data {
        SrwmaData::Candles { candles, source } => source_type(candles, source),
        SrwmaData::Slice(sl) => sl,
    };

    let first = data.iter().position(|x| !x.is_nan()).ok_or(SrwmaError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(SrwmaError::InvalidPeriod { period, data_len: len });
    }
    if (len - first) < period + 1 {
        return Err(SrwmaError::NotEnoughValidData {
            needed: period + 1,
            valid: len - first,
        });
    }

    let weight_len = period - 1;
    let mut weights: AVec<f64> = AVec::with_capacity(CACHELINE_ALIGN, weight_len);
    weights.resize(weight_len, 0.0);
    let mut norm = 0.0;
    for i in 0..weight_len {
        let w = ((period - i) as f64).sqrt();
        weights[i] = w;
        norm += w;
    }
    let inv_norm = 1.0 / norm;

    let warm = first + period + 1;
    let mut out = alloc_with_nan_prefix(len, warm);

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                srwma_scalar(data, &weights, period, first, inv_norm, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                srwma_avx2(data, &weights, period, first, inv_norm, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                srwma_avx512(data, &weights, period, first, inv_norm, &mut out)
            }
            _ => unreachable!(),
        }
    }

    Ok(SrwmaOutput { values: out })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn srwma_avx512(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first_valid: usize,
    inv_norm: f64,
    out: &mut [f64],
) {
    if period <= 32 {
        unsafe { srwma_avx512_short(data, weights, period, first_valid, inv_norm, out) }
    } else {
        unsafe { srwma_avx512_long(data, weights, period, first_valid, inv_norm, out) }
    }
}

#[inline]
pub fn srwma_scalar(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first_val: usize,
    inv_norm: f64,
    out: &mut [f64],
) {
    assert_eq!(weights.len(), period - 1, "weights.len() must be period - 1");
    assert!(out.len() >= data.len(), "`out` must be at least as long as `data`");

    let wlen = period - 1;
    let start_idx = first_val + period + 1;
    let len = data.len();

    for i in start_idx..len {
        let mut sum = 0.0;
        for k in 0..wlen {
            let d = data[i - k];
            let w = weights[k];
            sum += d * w;
        }
        out[i] = sum * inv_norm;
    }
}


#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn srwma_avx2(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first_valid: usize,
    inv_norm: f64,
    out: &mut [f64],
) {
    srwma_scalar(data, weights, period, first_valid, inv_norm, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn srwma_avx512_short(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first_valid: usize,
    inv_norm: f64,
    out: &mut [f64],
) {
    srwma_scalar(data, weights, period, first_valid, inv_norm, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn srwma_avx512_long(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first_valid: usize,
    inv_norm: f64,
    out: &mut [f64],
) {
    srwma_scalar(data, weights, period, first_valid, inv_norm, out)
}

#[inline]
pub fn srwma_batch_with_kernel(
    data: &[f64],
    sweep: &SrwmaBatchRange,
    k: Kernel,
) -> Result<SrwmaBatchOutput, SrwmaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(SrwmaError::InvalidPeriod {
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
    srwma_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct SrwmaBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for SrwmaBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 50, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SrwmaBatchBuilder {
    range: SrwmaBatchRange,
    kernel: Kernel,
}

impl SrwmaBatchBuilder {
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

    pub fn apply_slice(self, data: &[f64]) -> Result<SrwmaBatchOutput, SrwmaError> {
        srwma_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<SrwmaBatchOutput, SrwmaError> {
        SrwmaBatchBuilder::new().kernel(k).apply_slice(data)
    }

    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<SrwmaBatchOutput, SrwmaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }

    pub fn with_default_candles(c: &Candles) -> Result<SrwmaBatchOutput, SrwmaError> {
        SrwmaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct SrwmaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<SrwmaParams>,
    pub rows: usize,
    pub cols: usize,
}
impl SrwmaBatchOutput {
    pub fn row_for_params(&self, p: &SrwmaParams) -> Option<usize> {
        self.combos.iter().position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
    }
    pub fn values_for(&self, p: &SrwmaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &SrwmaBatchRange) -> Vec<SrwmaParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(SrwmaParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn srwma_batch_slice(
    data: &[f64],
    sweep: &SrwmaBatchRange,
    kern: Kernel,
) -> Result<SrwmaBatchOutput, SrwmaError> {
    srwma_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn srwma_batch_par_slice(
    data: &[f64],
    sweep: &SrwmaBatchRange,
    kern: Kernel,
) -> Result<SrwmaBatchOutput, SrwmaError> {
    srwma_batch_inner(data, sweep, kern, true)
}

#[inline]
fn round_up8(x: usize) -> usize {
    (x + 7) & !7
}
use std::alloc::{alloc, dealloc, Layout};

#[inline(always)]
fn srwma_batch_inner(
    data: &[f64],
    sweep: &SrwmaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<SrwmaBatchOutput, SrwmaError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(SrwmaError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(SrwmaError::AllValuesNaN)?;
    let len = data.len();

    let max_wlen = combos.iter().map(|c| c.period.unwrap() - 1).max().unwrap();
    let rows = combos.len();
    let cols = len;

    if combos
        .iter()
        .any(|c| (len - first) < (c.period.unwrap() + 1))
    {
        let needed = combos.iter().map(|c| c.period.unwrap() + 1).max().unwrap();
        return Err(SrwmaError::NotEnoughValidData {
            needed,
            valid: len - first,
        });
    }

    let mut inv_norms = vec![0.0; rows];
    let cap = rows * max_wlen;
    let mut flat_w = AVec::<f64>::with_capacity(CACHELINE_ALIGN, cap);
    flat_w.resize(cap, 0.0);
    let flat_slice = flat_w.as_mut_slice();

    for (row, prm) in combos.iter().enumerate() {
        let period = prm.period.unwrap();
        let wlen = period - 1;
        let mut norm = 0.0;
        let base = row * max_wlen;
        for i in 0..wlen {
            let w = ((period - i) as f64).sqrt();
            flat_slice[base + i] = w;
            norm += w;
        }
        inv_norms[row] = 1.0 / norm;
    }

    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() + 1)   // +1 because SRWMA starts at first+period+1
        .collect();

    let mut raw = make_uninit_matrix(rows, cols);
    unsafe { init_matrix_prefixes(&mut raw, cols, &warm) };

    // ---------- 2. per-row worker ---------------------------------------
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();
        let w_ptr  = flat_slice.as_ptr().add(row * max_wlen);
        let inv_n  = *inv_norms.get_unchecked(row);

        // treat this row as &mut [f64]
        let out_row = core::slice::from_raw_parts_mut(
            dst_mu.as_mut_ptr() as *mut f64,
            dst_mu.len(),
        );

        match kern {
            Kernel::Scalar => srwma_row_scalar(data, first, period, max_wlen, w_ptr, inv_n, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2   => srwma_row_avx2  (data, first, period, max_wlen, w_ptr, inv_n, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => srwma_row_avx512(data, first, period, max_wlen, w_ptr, inv_n, out_row),
            _              => unreachable!(),
        }
    };

    // ---------- 3. fill every row directly into `raw` --------------------
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

    // ---------- 4. finished – convert to Vec<f64> ------------------------
    let values: Vec<f64> = unsafe { std::mem::transmute(raw) };

    Ok(SrwmaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}



#[inline(always)]
unsafe fn srwma_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    w_ptr: *const f64,
    inv_n: f64,
    out: &mut [f64],
) {
    let wlen = period - 1;
    let len = data.len();
    let start_idx = first + period + 1;

    for i in start_idx..len {
        let mut sum = 0.0;
        for k in 0..wlen {
            sum += *data.get_unchecked(i - k) * *w_ptr.add(k);
        }
        out[i] = sum * inv_n;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn srwma_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    w_ptr: *const f64,
    inv_n: f64,
    out: &mut [f64],
) {
    srwma_row_scalar(data, first, period, stride, w_ptr, inv_n, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn srwma_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    w_ptr: *const f64,
    inv_n: f64,
    out: &mut [f64],
) {
    if period <= 32 {
        srwma_row_avx512_short(data, first, period, stride, w_ptr, inv_n, out);
    
        } else {
        srwma_row_avx512_long(data, first, period, stride, w_ptr, inv_n, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn srwma_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    w_ptr: *const f64,
    inv_n: f64,
    out: &mut [f64],
) {
    srwma_row_scalar(data, first, period, _stride, w_ptr, inv_n, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn srwma_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    w_ptr: *const f64,
    inv_n: f64,
    out: &mut [f64],
) {
    srwma_row_scalar(data, first, period, _stride, w_ptr, inv_n, out);
}


pub struct SrwmaStream {
    period: usize,
    weights: Vec<f64>,
    sum_weights: f64,
    data_history: Vec<f64>,
}

impl SrwmaStream {
    pub fn try_new(params: SrwmaParams) -> Result<Self, SrwmaError> {
        let period = params.period.unwrap_or(14);
        if period == 0 {
            return Err(SrwmaError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let wlen = period - 1;
        let mut weights = Vec::with_capacity(wlen);
        let mut sumw = 0.0;
        for i in 0..wlen {
            let w = ((period - i) as f64).sqrt();
            weights.push(w);
            sumw += w;
        }

        Ok(Self {
            period,
            weights,
            sum_weights: sumw,
            data_history: Vec::new(),
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.data_history.push(value);
        let idx = self.data_history.len() - 1;

        if idx + 1 <= self.period + 1 {
            return None;
        }

        let wlen = self.period - 1;
        let mut sum = 0.0;
        for k in 0..wlen {
            let data_idx = idx - k;
            sum += self.data_history[data_idx] * self.weights[k];
        }
        Some(sum / self.sum_weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_srwma_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = SrwmaParams { period: None };
        let input = SrwmaInput::from_candles(&candles, "close", default_params);
        let output = srwma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_srwma_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = SrwmaInput::from_candles(&candles, "close", SrwmaParams::default());
        let result = srwma_with_kernel(&input, kernel)?;
        let expected_last_five = [
            59344.28384704595,
            59282.09151629659,
            59192.76580529367,
            59178.04767548977,
            59110.03801260874,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-8,
                "[{}] SRWMA {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_srwma_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = SrwmaInput::with_default_candles(&candles);
        match input.data {
            SrwmaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected SrwmaData::Candles"),
        }
        let output = srwma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_srwma_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = SrwmaParams { period: Some(0) };
        let input = SrwmaInput::from_slice(&input_data, params);
        let res = srwma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] SRWMA should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_srwma_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = SrwmaParams { period: Some(10) };
        let input = SrwmaInput::from_slice(&data_small, params);
        let res = srwma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] SRWMA should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_srwma_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = SrwmaParams { period: Some(3) };
        let input = SrwmaInput::from_slice(&single_point, params);
        let res = srwma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] SRWMA should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_srwma_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = SrwmaParams { period: Some(14) };
        let first_input = SrwmaInput::from_candles(&candles, "close", first_params);
        let first_result = srwma_with_kernel(&first_input, kernel)?;

        let second_params = SrwmaParams { period: Some(5) };
        let second_input = SrwmaInput::from_slice(&first_result.values, second_params);
        let second_result = srwma_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());
        for i in 50..second_result.values.len() {
            assert!(second_result.values[i].is_finite());
        }
        Ok(())
    }

    fn check_srwma_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = SrwmaInput::from_candles(
            &candles,
            "close",
            SrwmaParams { period: Some(14) },
        );
        let res = srwma_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 50 {
            for (i, &val) in res.values[50..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test_name,
                    50 + i
                );
            }
        }
        Ok(())
    }

    fn check_srwma_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let period = 14;
        let input = SrwmaInput::from_candles(
            &candles,
            "close",
            SrwmaParams { period: Some(period) },
        );
        let batch_output = srwma_with_kernel(&input, kernel)?.values;

        let mut stream = SrwmaStream::try_new(SrwmaParams { period: Some(period) })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(srwma_val) => stream_values.push(srwma_val),
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
                "[{}] SRWMA streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_srwma_tests {
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
    generate_all_srwma_tests!(
        check_srwma_partial_params,
        check_srwma_accuracy,
        check_srwma_default_candles,
        check_srwma_zero_period,
        check_srwma_period_exceeds_length,
        check_srwma_very_small_dataset,
        check_srwma_reinput,
        check_srwma_nan_handling,
        check_srwma_streaming
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = SrwmaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = SrwmaParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            59344.28384704595,
            59282.09151629659,
            59192.76580529367,
            59178.04767548977,
            59110.03801260874,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-8,
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
