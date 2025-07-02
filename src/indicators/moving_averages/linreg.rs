//! # Linear Regression (LINREG)
//!
//! Fits a straight line (y = a + b·x) to recent data over a rolling window and forecasts the next value.
//! Supports kernels (scalar, AVX2, AVX512), streaming, batch grid, custom input sources, error reporting, and parameter builders.
//!
//! ## Parameters
//! - **period**: Look-back window size (default: 14).
//!
//! ## Errors
//! - **AllValuesNaN**: linreg: All input data values are `NaN`.
//! - **InvalidPeriod**: linreg: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: linreg: Not enough valid data points for the requested `period`.
//!
//! ## Returns
//! - **`Ok(LinRegOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(LinRegError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel, make_uninit_matrix, alloc_with_nan_prefix, init_matrix_prefixes};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;
use std::mem::MaybeUninit;  

// --- DATA, PARAMS, INPUT/OUTPUT STRUCTS ---

#[derive(Debug, Clone)]
pub enum LinRegData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct LinRegOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct LinRegParams {
    pub period: Option<usize>,
}

impl Default for LinRegParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct LinRegInput<'a> {
    pub data: LinRegData<'a>,
    pub params: LinRegParams,
}

impl<'a> AsRef<[f64]> for LinRegInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            LinRegData::Slice(slice) => slice,
            LinRegData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

impl<'a> LinRegInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: LinRegParams) -> Self {
        Self {
            data: LinRegData::Candles { candles: c, source: s },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: LinRegParams) -> Self {
        Self {
            data: LinRegData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", LinRegParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
}

// --- BUILDER ---

#[derive(Copy, Clone, Debug)]
pub struct LinRegBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for LinRegBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl LinRegBuilder {
    #[inline(always)]
    pub fn new() -> Self { Self::default() }
    #[inline(always)]
    pub fn period(mut self, n: usize) -> Self { self.period = Some(n); self }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self { self.kernel = k; self }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<LinRegOutput, LinRegError> {
        let p = LinRegParams { period: self.period };
        let i = LinRegInput::from_candles(c, "close", p);
        linreg_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<LinRegOutput, LinRegError> {
        let p = LinRegParams { period: self.period };
        let i = LinRegInput::from_slice(d, p);
        linreg_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<LinRegStream, LinRegError> {
        let p = LinRegParams { period: self.period };
        LinRegStream::try_new(p)
    }
}

// --- ERRORS ---

#[derive(Debug, Error)]
pub enum LinRegError {
    #[error("linreg: All values are NaN.")]
    AllValuesNaN,
    #[error("linreg: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("linreg: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

// --- INDICATOR API ---

#[inline]
pub fn linreg(input: &LinRegInput) -> Result<LinRegOutput, LinRegError> {
    linreg_with_kernel(input, Kernel::Auto)
}

pub fn linreg_with_kernel(input: &LinRegInput, kernel: Kernel) -> Result<LinRegOutput, LinRegError> {
    let data: &[f64] = input.as_ref();
    let first = data.iter().position(|x| !x.is_nan()).ok_or(LinRegError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(LinRegError::InvalidPeriod { period, data_len: len });
    }
    if (len - first) < period {
        return Err(LinRegError::NotEnoughValidData { needed: period, valid: len - first });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let warm  = first + period;
    let mut out = alloc_with_nan_prefix(len, warm);

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => linreg_scalar(data, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => linreg_avx2(data, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => linreg_avx512(data, period, first, &mut out),
            _ => unreachable!(),
        }
    }

    Ok(LinRegOutput { values: out })
}


#[inline]
fn linreg_scalar(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    // ---- invariant pre‑computations ---------------------------------------------------------
    let period_f = period as f64;
    let x_sum   = ((period * (period + 1)) / 2)            as f64;         // Σx
    let x2_sum  = ((period * (period + 1) * (2 * period + 1)) / 6) as f64; // Σx²
    let denom_inv = 1.0 / (period_f * x2_sum - x_sum * x_sum);            // 1 / Δ
    let inv_period = 1.0 / period_f;

    // ---- prime running sums with the first (period‑1) points -------------------------------
    let mut y_sum  = 0.0;
    let mut xy_sum = 0.0;
    {
        let init_slice = &data[first .. first + period - 1];
        // k = 0‑based here; (k+1) gives us x ∈ [1, period‑1]
        for (k, &v) in init_slice.iter().enumerate() {
            let x = (k + 1) as f64;
            y_sum  += v;
            xy_sum += v * x;
        }
    }

    // ---- main rolling loop -----------------------------------------------------------------
    let mut idx = first + period - 1;          // index of *last* element in the current window
    while idx < data.len() {
        let new_val = data[idx];
        y_sum  += new_val;                     // include newest sample
        xy_sum += new_val * period_f;          // its x = period

        // coefficients
        let b = (period_f * xy_sum - x_sum * y_sum) * denom_inv;
        let a = (y_sum - b * x_sum) * inv_period;
        out[idx] = a + b * period_f;           // forecast next point (x = period)

        // slide window: remove oldest point and shift indices by ‑1
        xy_sum -= y_sum;                       // Σ(x·y) → Σ((x‑1)·y)
        y_sum  -= data[idx + 1 - period];      // drop y_{t‑period+1}

        idx += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn linreg_avx2(
    data: &[f64], period: usize, first: usize, out: &mut [f64],
) {
    linreg_scalar(data, period, first, out);
}


#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512dq,fma")]
pub unsafe fn linreg_avx512(
    data: &[f64],
    period: usize,
    first: usize,
    out:   &mut [f64],
) {
    linreg_scalar(data, period, first, out);
}


// --- BATCH RANGE/BUILDER/OUTPUT/GRID ---

#[derive(Clone, Debug)]
pub struct LinRegBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for LinRegBatchRange {
    fn default() -> Self {
        Self { period: (14, 40, 1) }
    }
}

#[derive(Clone, Debug, Default)]
pub struct LinRegBatchBuilder {
    range: LinRegBatchRange,
    kernel: Kernel,
}

impl LinRegBatchBuilder {
    pub fn new() -> Self { Self::default() }
    pub fn kernel(mut self, k: Kernel) -> Self { self.kernel = k; self }
    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.period = (start, end, step); self
    }
    pub fn period_static(mut self, p: usize) -> Self {
        self.range.period = (p, p, 0); self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<LinRegBatchOutput, LinRegError> {
        linreg_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<LinRegBatchOutput, LinRegError> {
        LinRegBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<LinRegBatchOutput, LinRegError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<LinRegBatchOutput, LinRegError> {
        LinRegBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct LinRegBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<LinRegParams>,
    pub rows: usize,
    pub cols: usize,
}

impl LinRegBatchOutput {
    pub fn row_for_params(&self, p: &LinRegParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(14) == p.period.unwrap_or(14)
        })
    }
    pub fn values_for(&self, p: &LinRegParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

// --- BATCH MAIN ENTRYPOINTS ---

pub fn linreg_batch_with_kernel(
    data: &[f64],
    sweep: &LinRegBatchRange,
    k: Kernel,
) -> Result<LinRegBatchOutput, LinRegError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(LinRegError::InvalidPeriod { period: 0, data_len: 0 });
        }
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    linreg_batch_par_slice(data, sweep, simd)
}

#[inline(always)]
pub fn linreg_batch_slice(
    data: &[f64],
    sweep: &LinRegBatchRange,
    kern: Kernel,
) -> Result<LinRegBatchOutput, LinRegError> {
    linreg_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn linreg_batch_par_slice(
    data: &[f64],
    sweep: &LinRegBatchRange,
    kern: Kernel,
) -> Result<LinRegBatchOutput, LinRegError> {
    linreg_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn linreg_batch_inner(
    data: &[f64],
    sweep: &LinRegBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<LinRegBatchOutput, LinRegError> {
    // ------------- 0. sanity checks -------------
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(LinRegError::InvalidPeriod { period: 0, data_len: 0 });
    }
    let first = data.iter().position(|x| !x.is_nan())
        .ok_or(LinRegError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(LinRegError::NotEnoughValidData {
            needed: max_p,
            valid : data.len() - first,
        });
    }

    // ------------- 1. matrix set-up -------------
    let rows = combos.len();
    let cols = data.len();

    // per-row prefix length that must stay NaN
    let warm: Vec<usize> =
        combos.iter().map(|c| first + c.period.unwrap()).collect();

    // allocate rows × cols as MaybeUninit and paint the prefixes
    let mut raw = make_uninit_matrix(rows, cols);
    unsafe { init_matrix_prefixes(&mut raw, cols, &warm) };

    // ------------- 2. per-row worker ------------
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period  = combos[row].period.unwrap();

        // cast this single row to &mut [f64]
        let out_row = core::slice::from_raw_parts_mut(
            dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        match kern {
            Kernel::Scalar => linreg_row_scalar (data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2   => linreg_row_avx2   (data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => linreg_row_avx512 (data, first, period, out_row),
            _ => unreachable!(),
        }
    };

    // ------------- 3. run every row -------------
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

    // ------------- 4. transmute -----------------
    let values: Vec<f64> = unsafe { std::mem::transmute(raw) };

    Ok(LinRegBatchOutput { values, combos, rows, cols })
}
// --- ROW SCALAR/SIMD (all AVX just call scalar for parity) ---

#[inline(always)]
unsafe fn linreg_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    linreg_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn linreg_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    linreg_avx2(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn linreg_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    linreg_avx512(data, period, first, out)
}

// --- STREAM SUPPORT ---

#[derive(Debug, Clone)]
pub struct LinRegStream {
    period: usize,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,
    x_sum: f64,
    x2_sum: f64,
}

impl LinRegStream {
    pub fn try_new(params: LinRegParams) -> Result<Self, LinRegError> {
        let period = params.period.unwrap_or(14);
        if period == 0 {
            return Err(LinRegError::InvalidPeriod { period, data_len: 0 });
        }
        let mut x_sum = 0.0;
        let mut x2_sum = 0.0;
        for i in 1..=period {
            let xi = i as f64;
            x_sum += xi;
            x2_sum += xi * xi;
        }
        Ok(Self {
            period,
            buffer: vec![f64::NAN; period],
            head: 0,
            filled: false,
            x_sum,
            x2_sum,
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
        let mut y_sum = 0.0;
        let mut xy_sum = 0.0;
        for (i, &y) in (1..=self.period).zip(self.buffer.iter().cycle().skip(self.head).take(self.period)) {
            y_sum += y;
            xy_sum += y * (i as f64);
        }
        let pf = self.period as f64;
        let bd = 1.0 / (pf * self.x2_sum - self.x_sum * self.x_sum);
        let b = (pf * xy_sum - self.x_sum * y_sum) * bd;
        let a = (y_sum - b * self.x_sum) / pf;
        a + b * pf
    }
}

// --- BATCH GRID ---

#[inline(always)]
fn round_up8(x: usize) -> usize { (x + 7) & !7 }

// --- EXPOSED BATCH EXPANSION ---

#[inline(always)]
fn expand_grid(r: &LinRegBatchRange) -> Vec<LinRegParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(LinRegParams { period: Some(p) });
    }
    out
}


// --- TESTS ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_linreg_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let close_prices = candles.select_candle_field("close")?;
        let params = LinRegParams { period: Some(14) };
        let input = LinRegInput::from_candles(&candles, "close", params);
        let linreg_result = linreg_with_kernel(&input, kernel)?;
        let expected_last_five = [
            58929.37142857143,
            58899.42857142857,
            58918.857142857145,
            59100.6,
            58987.94285714286,
        ];
        assert!(linreg_result.values.len() >= 5);
        assert_eq!(linreg_result.values.len(), close_prices.len());
        let start_index = linreg_result.values.len() - 5;
        let result_last_five = &linreg_result.values[start_index..];
        for (i, &value) in result_last_five.iter().enumerate() {
            let expected_value = expected_last_five[i];
            assert!(
                (value - expected_value).abs() < 1e-1,
                "Mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }
        Ok(())
    }

    fn check_linreg_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = LinRegParams { period: None };
        let input = LinRegInput::from_candles(&candles, "close", default_params);
        let output = linreg_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_linreg_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = LinRegInput::with_default_candles(&candles);
        match input.data {
            LinRegData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected LinRegData::Candles"),
        }
        let output = linreg_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_linreg_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = LinRegParams { period: Some(0) };
        let input = LinRegInput::from_slice(&input_data, params);
        let res = linreg_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] LINREG should fail with zero period", test_name);
        Ok(())
    }

    fn check_linreg_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = LinRegParams { period: Some(10) };
        let input = LinRegInput::from_slice(&data_small, params);
        let res = linreg_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] LINREG should fail with period exceeding length", test_name);
        Ok(())
    }

    fn check_linreg_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = LinRegParams { period: Some(14) };
        let input = LinRegInput::from_slice(&single_point, params);
        let res = linreg_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] LINREG should fail with insufficient data", test_name);
        Ok(())
    }

    fn check_linreg_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = LinRegParams { period: Some(14) };
        let first_input = LinRegInput::from_candles(&candles, "close", first_params);
        let first_result = linreg_with_kernel(&first_input, kernel)?;
        let second_params = LinRegParams { period: Some(10) };
        let second_input = LinRegInput::from_slice(&first_result.values, second_params);
        let second_result = linreg_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    fn check_linreg_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = LinRegInput::from_candles(
            &candles,
            "close",
            LinRegParams { period: Some(14) },
        );
        let res = linreg_with_kernel(&input, kernel)?;
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

    fn check_linreg_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 14;
        let input = LinRegInput::from_candles(
            &candles,
            "close",
            LinRegParams { period: Some(period) },
        );
        let batch_output = linreg_with_kernel(&input, kernel)?.values;
        let mut stream = LinRegStream::try_new(LinRegParams { period: Some(period) })?;
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
            assert!(
                diff < 1e-6,
                "[{}] LINREG streaming mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name, i, b, s, diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_linreg_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(#[test] fn [<$test_fn _scalar_f64>]() { let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar); })*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(
                    #[test] fn [<$test_fn _avx2_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2); }
                    #[test] fn [<$test_fn _avx512_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512); }
                )*
            }
        }
    }

    generate_all_linreg_tests!(
        check_linreg_accuracy,
        check_linreg_partial_params,
        check_linreg_default_candles,
        check_linreg_zero_period,
        check_linreg_period_exceeds_length,
        check_linreg_very_small_dataset,
        check_linreg_reinput,
        check_linreg_nan_handling,
        check_linreg_streaming
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = LinRegBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;
        let def = LinRegParams::default();
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
