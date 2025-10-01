//! # Triple Exponential Moving Average (TEMA)
//!
//! Applies three exponential moving averages in succession to reduce lag and noise.
//! TEMA is calculated as: `TEMA = 3*EMA1 - 3*EMA2 + EMA3`, with all EMAs using the same period.
//!
//! ## Parameters
//! - **period**: Window size (number of data points, must be ≥ 1).
//!
//! ## Errors
//! - **AllValuesNaN**: tema: All input data values are `NaN`.
//! - **InvalidPeriod**: tema: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: tema: Not enough valid data points for the requested `period`.
//!
//! ## Returns
//! - **`Ok(TemaOutput)`** on success, containing a `Vec<f64>` matching the input length.
//! - **`Err(TemaError)`** otherwise.
//!
//! ## Developer Notes
//! - **Decision note (SIMD)**: Single-series SIMD is disabled by design.
//!   TEMA is three cascaded IIR filters with loop-carried dependencies,
//!   so AVX2/AVX512 over time provides no consistent >5% win. Runtime
//!   AVX entries delegate to the scalar path for exact numeric parity.
//! - **Streaming update**: O(1) - maintains three EMA states with simple update calculations
//! - **Memory optimization**: Uses `alloc_with_nan_prefix` for zero-copy allocation
//! - **Current status**: Scalar optimized; SIMD and batch SIMD paths delegate to scalar
//! - **Bench (100k, native cpu)**: scalar ~218.7µs → ~215.5µs (~1.5%)
//! - **Optimization opportunities**:
//!   - Implement vectorized AVX2/AVX512 kernels for parallel EMA calculations
//!   - Consider SIMD for batch processing multiple TEMA values simultaneously
//!   - Optimize EMA coefficient calculations with FMA instructions

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::cuda_available;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::CudaTema;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::alma::DeviceArrayF32Py;
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
#[cfg(all(feature = "python", feature = "cuda"))]
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

// ========== Input Data Types ==========

#[derive(Debug, Clone)]
pub enum TemaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct TemaInput<'a> {
    pub data: TemaData<'a>,
    pub params: TemaParams,
}

impl<'a> TemaInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: TemaParams) -> Self {
        Self {
            data: TemaData::Candles { candles, source },
            params,
        }
    }
    #[inline]
    pub fn from_slice(slice: &'a [f64], params: TemaParams) -> Self {
        Self {
            data: TemaData::Slice(slice),
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", TemaParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(9)
    }
}

impl<'a> AsRef<[f64]> for TemaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            TemaData::Slice(slice) => slice,
            TemaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

// ========== Parameter Structs ==========

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct TemaParams {
    pub period: Option<usize>,
}

impl Default for TemaParams {
    fn default() -> Self {
        Self { period: Some(9) }
    }
}

// ========== Output ==========

#[derive(Debug, Clone)]
pub struct TemaOutput {
    pub values: Vec<f64>,
}

// ========== Error Types ==========

#[derive(Debug, Error)]
pub enum TemaError {
    #[error("tema: Input data slice is empty.")]
    EmptyInputData,
    #[error("tema: All values are NaN.")]
    AllValuesNaN,
    #[error("tema: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("tema: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

// ========== Builder ==========

#[derive(Copy, Clone, Debug)]
pub struct TemaBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for TemaBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl TemaBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<TemaOutput, TemaError> {
        let p = TemaParams {
            period: self.period,
        };
        let i = TemaInput::from_candles(c, "close", p);
        tema_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<TemaOutput, TemaError> {
        let p = TemaParams {
            period: self.period,
        };
        let i = TemaInput::from_slice(d, p);
        tema_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<TemaStream, TemaError> {
        let p = TemaParams {
            period: self.period,
        };
        TemaStream::try_new(p)
    }
}

// ========== Indicator API ==========

#[inline]
pub fn tema(input: &TemaInput) -> Result<TemaOutput, TemaError> {
    tema_with_kernel(input, Kernel::Auto)
}

pub fn tema_with_kernel(input: &TemaInput, kernel: Kernel) -> Result<TemaOutput, TemaError> {
    let (data, period, first, len, chosen) = tema_prepare(input, kernel)?;
    let lookback = (period - 1) * 3;
    let warm = first + lookback;

    let mut out = alloc_with_nan_prefix(len, warm);
    tema_compute_into(data, period, first, chosen, &mut out);
    Ok(TemaOutput { values: out })
}

// ========== Scalar Implementation ==========

#[inline]
pub fn tema_scalar(data: &[f64], period: usize, first_val: usize, out: &mut [f64]) {
    debug_assert_eq!(data.len(), out.len());

    let n = data.len();
    if n == 0 {
        return;
    }

    // Coefficients: per = 2/(N+1); per1 = 1-per
    let per = 2.0 / (period as f64 + 1.0);
    let per1 = 1.0 - per;

    // Warmup thresholds hoisted out of the loop
    let p1 = period - 1;
    let start2 = first_val + p1; // when EMA2 becomes valid
    let start3 = first_val + (p1 << 1); // when EMA3 becomes valid
    let start_out = first_val + p1 * 3; // when TEMA becomes valid

    // Fast path: period == 1 -> EMA alpha = 1, so TEMA == price from first_val onward
    if period == 1 {
        // Prefix < first_val is already NaN from alloc_with_nan_prefix
        out[first_val..n].copy_from_slice(&data[first_val..n]);
        return;
    }

    // Initialize EMA state from the first valid element
    let mut ema1 = data[first_val];
    let mut ema2 = 0.0f64; // becomes valid at start2
    let mut ema3 = 0.0f64; // becomes valid at start3

    for i in first_val..n {
        let price = data[i];

        // EMA1 update
        ema1 = ema1 * per1 + price * per;

        // EMA2 warmup and update
        if i == start2 {
            ema2 = ema1;
        }
        if i >= start2 {
            ema2 = ema2 * per1 + ema1 * per;
        }

        // EMA3 warmup and update
        if i == start3 {
            ema3 = ema2;
        }
        if i >= start3 {
            ema3 = ema3 * per1 + ema2 * per;
        }

        // Output once all three EMAs are valid
        if i >= start_out {
            // Keep operation order identical to streaming path for bitwise parity
            out[i] = 3.0 * ema1 - 3.0 * ema2 + ema3;
        }
    }
}

// ========== AVX2/AVX512 Stubs ==========

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn tema_avx512(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    unsafe { tema_avx512_long(data, period, first_valid, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn tema_avx2(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    tema_scalar(data, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn tema_avx512_short(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    tema_scalar(data, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn tema_avx512_long(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    tema_scalar(data, period, first_valid, out)
}

// ========== Streaming ==========

#[derive(Debug, Clone)]
pub struct TemaStream {
    period: usize,
    buf: Vec<f64>,
    ema1: f64,
    ema2: f64,
    ema3: f64,
    pos: usize,
    filled: bool,
    step: usize,
    per: f64,
    per1: f64,
    valid: usize,
}

impl TemaStream {
    pub fn try_new(params: TemaParams) -> Result<Self, TemaError> {
        let period = params.period.unwrap_or(9);
        if period == 0 {
            return Err(TemaError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        Ok(Self {
            period,
            buf: vec![f64::NAN; period],
            ema1: f64::NAN,
            ema2: 0.0,
            ema3: 0.0,
            pos: 0,
            filled: false,
            step: 0,
            per: 2.0 / (period as f64 + 1.0),
            per1: 1.0 - (2.0 / (period as f64 + 1.0)),
            valid: 0,
        })
    }
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        if self.filled {
            self.ema1 = self.ema1 * self.per1 + value * self.per;
            self.ema2 = self.ema2 * self.per1 + self.ema1 * self.per;
            self.ema3 = self.ema3 * self.per1 + self.ema2 * self.per;
            let tema_val = 3.0 * self.ema1 - 3.0 * self.ema2 + self.ema3;
            return Some(tema_val);
        }

        if self.valid == 0 {
            self.ema1 = value;
            self.valid += 1;
            self.buf[self.pos] = value;
            self.pos = (self.pos + 1) % self.period;
            return None;
        }

        self.ema1 = self.ema1 * self.per1 + value * self.per;
        self.buf[self.pos] = value;
        self.pos = (self.pos + 1) % self.period;
        self.valid += 1;

        if self.valid == self.period {
            self.ema2 = self.ema1;
        } else if self.valid > self.period {
            self.ema2 = self.ema2 * self.per1 + self.ema1 * self.per;
        }

        if self.valid == 2 * self.period - 1 {
            self.ema3 = self.ema2;
        } else if self.valid > 2 * self.period - 1 {
            self.ema3 = self.ema3 * self.per1 + self.ema2 * self.per;
        }

        if self.valid > (self.period - 1) * 3 {
            self.filled = true;
            let tema_val = 3.0 * self.ema1 - 3.0 * self.ema2 + self.ema3;
            Some(tema_val)
        } else {
            None
        }
    }
}

// ========== Batch Processing ==========

#[derive(Clone, Debug)]
pub struct TemaBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for TemaBatchRange {
    fn default() -> Self {
        Self {
            period: (9, 240, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TemaBatchBuilder {
    range: TemaBatchRange,
    kernel: Kernel,
}

impl TemaBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<TemaBatchOutput, TemaError> {
        tema_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<TemaBatchOutput, TemaError> {
        TemaBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<TemaBatchOutput, TemaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<TemaBatchOutput, TemaError> {
        TemaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn tema_batch_with_kernel(
    data: &[f64],
    sweep: &TemaBatchRange,
    k: Kernel,
) -> Result<TemaBatchOutput, TemaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(TemaError::InvalidPeriod {
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
    tema_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct TemaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<TemaParams>,
    pub rows: usize,
    pub cols: usize,
}
impl TemaBatchOutput {
    pub fn row_for_params(&self, p: &TemaParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(9) == p.period.unwrap_or(9))
    }
    pub fn values_for(&self, p: &TemaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &TemaBatchRange) -> Vec<TemaParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(TemaParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn tema_batch_slice(
    data: &[f64],
    sweep: &TemaBatchRange,
    kern: Kernel,
) -> Result<TemaBatchOutput, TemaError> {
    tema_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn tema_batch_par_slice(
    data: &[f64],
    sweep: &TemaBatchRange,
    kern: Kernel,
) -> Result<TemaBatchOutput, TemaError> {
    tema_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn tema_batch_inner(
    data: &[f64],
    sweep: &TemaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<TemaBatchOutput, TemaError> {
    // ---------- 0. parameter checks ----------
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(TemaError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(TemaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(TemaError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    // ---------- 1. matrix dimensions ----------
    let rows = combos.len();
    let cols = data.len();

    // Resolve the kernel if it's Auto
    let actual_kern = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    };

    // ---------- 2. build per-row warm-up lengths ----------
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| (first + (c.period.unwrap() - 1) * 3).min(cols)) // (period-1)*3 matches tema_scalar, clamped to cols
        .collect();

    // ---------- 3. allocate rows×cols uninitialised, fill NaN prefixes ----------
    let mut raw = make_uninit_matrix(rows, cols);
    unsafe { init_matrix_prefixes(&mut raw, cols, &warm) };

    // ---------- 4. closure that fills ONE row ----------
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();

        // cast this row to &mut [f64] so the row-kernel can write normally
        let out_row =
            core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        match actual_kern {
            Kernel::Scalar | Kernel::ScalarBatch => tema_row_scalar(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => tema_row_avx2(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => tema_row_avx512(data, first, period, out_row),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                tema_row_scalar(data, first, period, out_row)
            }
            Kernel::Auto => unreachable!("Auto kernel should have been resolved"),
        }
    };

    // ---------- 5. run all rows (optionally in parallel) ----------
    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            raw.par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in raw.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in raw.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    // ---------- 6. safe Vec rebind like ALMA ----------
    let mut guard = core::mem::ManuallyDrop::new(raw);
    let values: Vec<f64> = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    // ---------- 7. package ----------
    Ok(TemaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn tema_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    tema_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
unsafe fn tema_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    tema_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
unsafe fn tema_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    if period <= 32 {
        tema_row_avx512_short(data, first, period, out)
    } else {
        tema_row_avx512_long(data, first, period, out)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
unsafe fn tema_row_avx512_short(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    tema_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
unsafe fn tema_row_avx512_long(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    tema_scalar(data, period, first, out)
}

// ========== Unit Tests ==========

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_tema_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = TemaParams { period: None };
        let input = TemaInput::from_candles(&candles, "close", default_params);
        let output = tema_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }
    fn check_tema_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = TemaInput::from_candles(&candles, "close", TemaParams::default());
        let result = tema_with_kernel(&input, kernel)?;
        let expected_last_five = [
            59281.895570662884,
            59257.25021607971,
            59172.23342859784,
            59175.218345941066,
            58934.24395798363,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-8,
                "[{}] TEMA {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }
    fn check_tema_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = TemaInput::with_default_candles(&candles);
        match input.data {
            TemaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected TemaData::Candles"),
        }
        let output = tema_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }
    fn check_tema_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = TemaParams { period: Some(0) };
        let input = TemaInput::from_slice(&input_data, params);
        let res = tema_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] TEMA should fail with zero period",
            test_name
        );
        Ok(())
    }
    fn check_tema_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data: [f64; 0] = [];
        let params = TemaParams { period: Some(9) };
        let input = TemaInput::from_slice(&input_data, params);
        let res = tema_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] TEMA should fail with empty input",
            test_name
        );
        if let Err(e) = res {
            assert!(
                matches!(e, TemaError::EmptyInputData),
                "[{}] Expected EmptyInputData error",
                test_name
            );
        }
        Ok(())
    }
    fn check_tema_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = TemaParams { period: Some(10) };
        let input = TemaInput::from_slice(&data_small, params);
        let res = tema_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] TEMA should fail with period exceeding length",
            test_name
        );
        Ok(())
    }
    fn check_tema_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = TemaParams { period: Some(9) };
        let input = TemaInput::from_slice(&single_point, params);
        let res = tema_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] TEMA should fail with insufficient data",
            test_name
        );
        Ok(())
    }
    fn check_tema_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = TemaParams { period: Some(9) };
        let first_input = TemaInput::from_candles(&candles, "close", first_params);
        let first_result = tema_with_kernel(&first_input, kernel)?;
        let second_params = TemaParams { period: Some(9) };
        let second_input = TemaInput::from_slice(&first_result.values, second_params);
        let second_result = tema_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }
    fn check_tema_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = TemaInput::from_candles(&candles, "close", TemaParams { period: Some(9) });
        let res = tema_with_kernel(&input, kernel)?;
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
    fn check_tema_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 9;
        let input = TemaInput::from_candles(
            &candles,
            "close",
            TemaParams {
                period: Some(period),
            },
        );
        let batch_output = tema_with_kernel(&input, kernel)?.values;
        let mut stream = TemaStream::try_new(TemaParams {
            period: Some(period),
        })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(val) => stream_values.push(val),
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
                "[{}] TEMA streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_tema_tests {
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

    // Check for poison values in single output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_tema_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test with multiple parameter combinations to better catch uninitialized memory bugs
        let test_periods = vec![5, 9, 14, 20, 50, 100, 200];

        for &period in &test_periods {
            let params = TemaParams {
                period: Some(period),
            };
            let input = TemaInput::from_candles(&candles, "close", params);

            // Skip if we don't have enough data for this period
            if candles.close.len() < period {
                continue;
            }

            let output = match tema_with_kernel(&input, kernel) {
                Ok(o) => o,
                Err(_) => continue, // Skip if this period causes an error
            };

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
    fn check_tema_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    fn check_tema_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Load real market data for realistic testing
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let close_data = &candles.close;

        // Strategy: test various parameter combinations with real data slices
        // TEMA uses triple exponential smoothing, typically with moderate periods
        let strat = (
            2usize..=50,                                  // period (TEMA typical range)
            0usize..close_data.len().saturating_sub(200), // starting index
            100usize..=200,                               // length of data slice to use
        );

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(period, start_idx, slice_len)| {
                // Ensure we have valid slice bounds
                let end_idx = (start_idx + slice_len).min(close_data.len());
                if end_idx <= start_idx || end_idx - start_idx < period * 3 + 10 {
                    return Ok(()); // Skip invalid combinations (need enough data for triple smoothing)
                }

                let data_slice = &close_data[start_idx..end_idx];
                let params = TemaParams {
                    period: Some(period),
                };
                let input = TemaInput::from_slice(data_slice, params);

                // Test the specified kernel
                let result = tema_with_kernel(&input, kernel);

                // Also compute with scalar kernel for reference
                let scalar_result = tema_with_kernel(&input, Kernel::Scalar);

                // Both should succeed or fail together
                match (result, scalar_result) {
                    (Ok(TemaOutput { values: out }), Ok(TemaOutput { values: ref_out })) => {
                        // Verify output length
                        prop_assert_eq!(out.len(), data_slice.len());
                        prop_assert_eq!(ref_out.len(), data_slice.len());

                        // Find first non-NaN value
                        let first = data_slice.iter().position(|x| !x.is_nan()).unwrap_or(0);
                        let lookback = (period - 1) * 3;
                        let expected_warmup = first + lookback; // TEMA warmup: first + (period - 1) * 3

                        // Check NaN pattern during warmup
                        for i in 0..expected_warmup.min(out.len()) {
                            prop_assert!(
                                out[i].is_nan(),
                                "Expected NaN at index {} during warmup, got {}",
                                i,
                                out[i]
                            );
                        }

                        // Test exponential smoothing properties
                        let multiplier = 2.0 / (period as f64 + 1.0);
                        prop_assert!(
                            multiplier > 0.0 && multiplier <= 1.0,
                            "EMA multiplier should be in (0, 1]: {}",
                            multiplier
                        );

                        // Test specific properties for valid outputs
                        for i in expected_warmup..out.len() {
                            let y = out[i];
                            let r = ref_out[i];

                            // Both should be valid
                            prop_assert!(!y.is_nan(), "Unexpected NaN at index {}", i);
                            prop_assert!(y.is_finite(), "Non-finite value at index {}: {}", i, y);

                            // Kernel consistency check
                            let y_bits = y.to_bits();
                            let r_bits = r.to_bits();

                            if !y.is_finite() || !r.is_finite() {
                                prop_assert_eq!(
                                    y_bits,
                                    r_bits,
                                    "NaN/Inf mismatch at {}: {} vs {}",
                                    i,
                                    y,
                                    r
                                );
                                continue;
                            }

                            // ULP difference check for floating-point precision
                            let ulp_diff: u64 = y_bits.abs_diff(r_bits);
                            prop_assert!(
                                (y - r).abs() <= 1e-9 || ulp_diff <= 5,
                                "Kernel mismatch at {}: {} vs {} (ULP={})",
                                i,
                                y,
                                r,
                                ulp_diff
                            );

                            // Note: TEMA can legitimately exceed window bounds due to its formula (3*EMA1 - 3*EMA2 + EMA3)
                            // The triple exponential smoothing amplifies recent price movements, which means:
                            // - In strong uptrends, TEMA can overshoot the maximum by ~10-20%
                            // - In strong downtrends, TEMA can undershoot the minimum by ~10-20%
                            // This is expected behavior, not a calculation error, so we don't enforce bounds checking
                        }

                        // Test constant data property
                        let const_value = 100.0;
                        let const_data = vec![const_value; period * 4];
                        let const_input = TemaInput::from_slice(
                            &const_data,
                            TemaParams {
                                period: Some(period),
                            },
                        );
                        if let Ok(TemaOutput { values: const_out }) =
                            tema_with_kernel(&const_input, kernel)
                        {
                            // After warmup, TEMA of constant data should converge to the constant
                            let const_warmup = lookback; // No NaN in input, so warmup is just lookback
                            for (i, &val) in const_out.iter().enumerate() {
                                if i >= const_warmup && !val.is_nan() {
                                    prop_assert!(
										(val - const_value).abs() < 1e-9,
										"TEMA of constant data should equal the constant at {}: got {}",
										i, val
									);
                                }
                            }
                        }

                        // Test streaming consistency
                        if period <= 20 {
                            // Only test smaller periods for speed
                            let mut stream = TemaStream::try_new(TemaParams {
                                period: Some(period),
                            })
                            .unwrap();
                            let mut stream_values = Vec::with_capacity(data_slice.len());

                            for &price in data_slice {
                                match stream.update(price) {
                                    Some(val) => stream_values.push(val),
                                    None => stream_values.push(f64::NAN),
                                }
                            }

                            // Compare streaming output with batch output
                            for (i, (&batch_val, &stream_val)) in
                                out.iter().zip(stream_values.iter()).enumerate()
                            {
                                if batch_val.is_nan() && stream_val.is_nan() {
                                    continue;
                                }
                                if !batch_val.is_nan() && !stream_val.is_nan() {
                                    prop_assert!(
                                        (batch_val - stream_val).abs() < 1e-9,
                                        "Streaming mismatch at {}: batch={} vs stream={}",
                                        i,
                                        batch_val,
                                        stream_val
                                    );
                                }
                            }
                        }

                        // Test that EMA relationship holds for some data points
                        // TEMA = 3*EMA1 - 3*EMA2 + EMA3
                        // We can't directly verify this without computing the EMAs,
                        // but we can check that TEMA responds appropriately to trends

                        // For a strongly trending section, TEMA should amplify the trend
                        if data_slice.len() > period * 2 {
                            // Find a trending section
                            let trend_start = expected_warmup;
                            let trend_end = (trend_start + period).min(data_slice.len());

                            if trend_end > trend_start + 3 {
                                let trend_data = &data_slice[trend_start..trend_end];
                                let is_uptrend =
                                    trend_data.windows(2).filter(|w| w[1] > w[0]).count()
                                        > trend_data.windows(2).filter(|w| w[1] < w[0]).count();

                                if is_uptrend {
                                    // In an uptrend, TEMA often leads price (can be above current price)
                                    // This is because TEMA = 3*EMA1 - 3*EMA2 + EMA3 amplifies recent movements
                                    // We just check that TEMA is responding to the trend, not lagging excessively
                                    let last_price = data_slice[trend_end - 1];
                                    let tema_value = out[trend_end - 1];

                                    // TEMA should be reasonably close to the price, not wildly divergent
                                    let price_range = trend_data
                                        .iter()
                                        .cloned()
                                        .fold(f64::NEG_INFINITY, f64::max)
                                        - trend_data.iter().cloned().fold(f64::INFINITY, f64::min);
                                    prop_assert!(
										(tema_value - last_price).abs() < price_range * 1.5,
										"TEMA diverged too much from price: TEMA={}, price={}, range={}",
										tema_value, last_price, price_range
									);
                                }
                            }
                        }
                    }
                    (Err(e1), Err(e2)) => {
                        // Both kernels should fail with similar errors
                        prop_assert_eq!(
                            std::mem::discriminant(&e1),
                            std::mem::discriminant(&e2),
                            "Different error types: {:?} vs {:?}",
                            e1,
                            e2
                        );
                    }
                    _ => {
                        prop_assert!(
                            false,
                            "Kernel consistency failure: one succeeded, one failed"
                        );
                    }
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    generate_all_tema_tests!(
        check_tema_partial_params,
        check_tema_accuracy,
        check_tema_default_candles,
        check_tema_zero_period,
        check_tema_empty_input,
        check_tema_period_exceeds_length,
        check_tema_very_small_dataset,
        check_tema_reinput,
        check_tema_nan_handling,
        check_tema_streaming,
        check_tema_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_tema_tests!(check_tema_property);

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = TemaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = TemaParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [
            59281.895570662884,
            59257.25021607971,
            59172.23342859784,
            59175.218345941066,
            58934.24395798363,
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

    // Check for poison values in batch output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test multiple batch configurations with different period ranges
        let test_configs = vec![
            (5, 15, 2),    // Small periods with fine steps
            (10, 50, 5),   // Medium periods
            (20, 100, 10), // Large periods
            (50, 200, 25), // Very large periods
            (3, 3, 1),     // Single small period
            (150, 150, 1), // Single large period
        ];

        for (start, end, step) in test_configs {
            let output = TemaBatchBuilder::new()
                .kernel(kernel)
                .period_range(start, end, step)
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
                let period = output
                    .combos
                    .get(row)
                    .map(|p| p.period.unwrap_or(0))
                    .unwrap_or(0);

                // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (period {}, flat index {})",
                        test, val, bits, row, col, period, idx
                    );
                }

                // Check for init_matrix_prefixes poison (0x22222222_22222222)
                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (period {}, flat index {})",
                        test, val, bits, row, col, period, idx
                    );
                }

                // Check for make_uninit_matrix poison (0x33333333_33333333)
                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (period {}, flat index {})",
                        test, val, bits, row, col, period, idx
                    );
                }
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);
}

// ========== Helper Functions for Bindings ==========

/// Centralized validation and preparation for TEMA calculation
#[inline]
fn tema_prepare<'a>(
    input: &'a TemaInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, usize, usize, Kernel), TemaError> {
    let data: &[f64] = match &input.data {
        TemaData::Candles { candles, source } => source_type(candles, source),
        TemaData::Slice(sl) => sl,
    };

    if data.is_empty() {
        return Err(TemaError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(TemaError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(TemaError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(TemaError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    Ok((data, period, first, len, chosen))
}

/// Compute TEMA directly into pre-allocated output buffer
#[inline]
fn tema_compute_into(data: &[f64], period: usize, first: usize, chosen: Kernel, out: &mut [f64]) {
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => tema_scalar(data, period, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => tema_avx2(data, period, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => tema_avx512(data, period, first, out),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                // Fallback to scalar when AVX is not available
                tema_scalar(data, period, first, out)
            }
            Kernel::Auto => unreachable!(),
        }
    }
}

/// Optimized batch calculation that writes directly to pre-allocated buffer
#[inline(always)]
fn tema_batch_inner_into(
    data: &[f64],
    sweep: &TemaBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<TemaParams>, TemaError> {
    // ---------- 0. parameter checks ----------
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(TemaError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    if data.is_empty() {
        return Err(TemaError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(TemaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(TemaError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    // ---------- 1. matrix dimensions ----------
    let rows = combos.len();
    let cols = data.len();

    // Resolve the kernel if it's Auto
    let actual_kern = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    };

    // ---------- 2. build per-row warm-up lengths ----------
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| (first + (c.period.unwrap() - 1) * 3).min(cols)) // (period-1)*3 matches tema_scalar, clamped to cols
        .collect();

    // ---------- 3. reinterpret output slice as MaybeUninit for efficient initialization ----------
    let out_uninit = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };

    unsafe { init_matrix_prefixes(out_uninit, cols, &warm) };

    // ---------- 4. closure that fills ONE row ----------
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();

        // cast this row to &mut [f64] so the row-kernel can write normally
        let out_row =
            core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        match actual_kern {
            Kernel::Scalar | Kernel::ScalarBatch => tema_row_scalar(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => tema_row_avx2(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => tema_row_avx512(data, first, period, out_row),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                tema_row_scalar(data, first, period, out_row)
            }
            Kernel::Auto => unreachable!("Auto kernel should have been resolved"),
        }
    };

    // ---------- 5. run all rows (optionally in parallel) ----------
    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_uninit
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in out_uninit.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in out_uninit.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    Ok(combos)
}

// ========== Python Bindings ==========

#[cfg(feature = "python")]
#[pyfunction(name = "tema")]
#[pyo3(signature = (data, period, kernel=None))]
/// Compute the Triple Exponential Moving Average (TEMA) of the input data.
///
/// TEMA applies three exponential moving averages in succession to reduce lag and noise.
/// It is calculated as: TEMA = 3*EMA1 - 3*EMA2 + EMA3.
///
/// Parameters:
/// -----------
/// data : np.ndarray
///     Input data array (float64).
/// period : int
///     Number of data points in the moving average window (must be >= 1).
/// kernel : str, optional
///     Computation kernel to use: 'auto', 'scalar', 'avx2', 'avx512'.
///     Default is 'auto' which auto-detects the best available.
///
/// Returns:
/// --------
/// np.ndarray
///     Array of TEMA values, same length as input.
///
/// Raises:
/// -------
/// ValueError
///     If inputs are invalid (period is zero, exceeds data length, etc).
pub fn tema_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?; // Validate before allow_threads

    let params = TemaParams {
        period: Some(period),
    };
    let tema_in = TemaInput::from_slice(slice_in, params);

    // Get Vec<f64> from Rust function
    let result_vec: Vec<f64> = py
        .allow_threads(|| tema_with_kernel(&tema_in, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Zero-copy transfer to NumPy
    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "TemaStream")]
pub struct TemaStreamPy {
    stream: TemaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl TemaStreamPy {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        let params = TemaParams {
            period: Some(period),
        };
        let stream =
            TemaStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(TemaStreamPy { stream })
    }

    /// Updates the stream with a new value and returns the calculated TEMA value.
    /// Returns `None` if the buffer is not yet full.
    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "tema_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
/// Compute TEMA for multiple period values in a single pass.
///
/// Parameters:
/// -----------
/// data : np.ndarray
///     Input data array (float64).
/// period_range : tuple
///     (start, end, step) for period values to compute.
/// kernel : str, optional
///     Computation kernel: 'auto', 'scalar', 'avx2', 'avx512'.
///     Default is 'auto' which auto-detects the best available.
///
/// Returns:
/// --------
/// dict
///     Dictionary with 'values' (2D array) and 'periods' arrays.
pub fn tema_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?; // true for batch operations

    let sweep = TemaBatchRange {
        period: period_range,
    };

    // Calculate dimensions
    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();

    // Pre-allocate output array (OK for batch operations)
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    // Compute without GIL
    let combos = py
        .allow_threads(|| {
            // Handle kernel selection for batch operations
            let kernel = match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                k => k,
            };

            // Map batch kernels to regular kernels
            let simd = match kernel {
                Kernel::Avx512Batch => Kernel::Avx512,
                Kernel::Avx2Batch => Kernel::Avx2,
                Kernel::ScalarBatch => Kernel::Scalar,
                _ => kernel,
            };

            tema_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Build result dictionary
    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "periods",
        combos
            .iter()
            .map(|p| p.period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    Ok(dict)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "tema_cuda_batch_dev")]
#[pyo3(signature = (data_f32, period_range, device_id=0))]
pub fn tema_cuda_batch_dev_py(
    py: Python<'_>,
    data_f32: PyReadonlyArray1<'_, f32>,
    period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let slice_in = data_f32.as_slice()?;
    let sweep = TemaBatchRange {
        period: period_range,
    };

    let inner = py.allow_threads(|| {
        let cuda = CudaTema::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.tema_batch_dev(slice_in, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(DeviceArrayF32Py { inner })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "tema_cuda_many_series_one_param_dev")]
#[pyo3(signature = (prices_tm_f32, period, device_id=0))]
pub fn tema_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    prices_tm_f32: PyReadonlyArray2<'_, f32>,
    period: usize,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    use numpy::PyUntypedArrayMethods;

    let rows = prices_tm_f32.shape()[0];
    let cols = prices_tm_f32.shape()[1];

    let prices_flat = prices_tm_f32.as_slice()?;

    let inner = py.allow_threads(|| {
        let cuda = CudaTema::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.tema_many_series_one_param_time_major_dev(prices_flat, cols, rows, period)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(DeviceArrayF32Py { inner })
}

// ========== WASM Bindings ==========

#[inline]
pub fn tema_into_slice(dst: &mut [f64], input: &TemaInput, kern: Kernel) -> Result<(), TemaError> {
    let (data, period, first, len, chosen) = tema_prepare(input, kern)?;
    if dst.len() != len {
        return Err(TemaError::InvalidPeriod {
            period: dst.len(),
            data_len: len,
        });
    }
    tema_compute_into(data, period, first, chosen, dst);
    let warm = first + (period - 1) * 3;
    for v in &mut dst[..warm] {
        *v = f64::NAN;
    }
    Ok(())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn tema_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    // Validate inputs first
    if data.is_empty() {
        return Err(JsValue::from_str("Input data slice is empty"));
    }
    if period == 0 || period > data.len() {
        return Err(JsValue::from_str(&format!(
            "Invalid period: {} (data length: {})",
            period,
            data.len()
        )));
    }

    // Check if all values are NaN
    if data.iter().all(|&x| x.is_nan()) {
        return Err(JsValue::from_str("All values are NaN"));
    }

    // Check if there's enough valid data after NaN values
    let first_valid = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
    let valid_count = data.len() - first_valid;
    if valid_count < period {
        return Err(JsValue::from_str(&format!(
            "Not enough valid data: need {} but only {} valid values after NaN values",
            period, valid_count
        )));
    }

    // Special case for WASM: when the warmup period equals or exceeds data length
    // from the first valid index, return all NaN values instead of hitting
    // an unreachable panic in the kernel selection code
    if period > 1 {
        let lookback = (period - 1) * 3;
        if first_valid + lookback >= data.len() {
            // Return all NaN values
            return Ok(vec![f64::NAN; data.len()]);
        }
    }

    let params = TemaParams {
        period: Some(period),
    };
    let input = TemaInput::from_slice(data, params);

    let mut output = vec![0.0; data.len()];

    tema_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct TemaBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct TemaBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<TemaParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = tema_batch)]
pub fn tema_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: TemaBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = TemaBatchRange {
        period: config.period_range,
    };

    let output = tema_batch_inner(data, &sweep, Kernel::Auto, false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = TemaBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };

    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[deprecated(since = "1.0.0", note = "Use tema_batch instead")]
pub fn tema_batch_js(
    data: &[f64],
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = TemaBatchRange {
        period: (period_start, period_end, period_step),
    };

    // Use the existing batch function with parallel=false for WASM
    tema_batch_inner(data, &sweep, Kernel::Scalar, false)
        .map(|output| output.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn tema_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn tema_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn tema_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to tema_into"));
    }

    // Validate inputs
    if len == 0 {
        return Err(JsValue::from_str("Input data slice is empty"));
    }
    if period == 0 || period > len {
        return Err(JsValue::from_str(&format!(
            "Invalid period: {} (data length: {})",
            period, len
        )));
    }

    let data = unsafe { std::slice::from_raw_parts(in_ptr, len) };

    // Special case for WASM: when the warmup period equals or exceeds data length
    // AND there are valid data points, fill with NaN values instead of hitting
    // an unreachable panic in the kernel selection code
    if period > 1 && !data.iter().all(|&x| x.is_nan()) {
        let lookback = (period - 1) * 3;
        if lookback >= len {
            // Fill output with NaN values
            let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, len) };
            out.fill(f64::NAN);
            return Ok(());
        }
    }

    let params = TemaParams {
        period: Some(period),
    };
    let input = TemaInput::from_slice(data, params);

    if in_ptr == out_ptr {
        // In-place computation: need temporary buffer
        let mut temp = vec![0.0; len];
        tema_into_slice(&mut temp, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, len) };
        out.copy_from_slice(&temp);
    } else {
        // Direct computation into output buffer
        let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, len) };
        tema_into_slice(out, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }

    Ok(())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn tema_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to tema_batch_into"));
    }

    let data = unsafe { std::slice::from_raw_parts(in_ptr, len) };
    let sweep = TemaBatchRange {
        period: (period_start, period_end, period_step),
    };

    // Get the number of parameter combinations
    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = data.len();
    let total_size = rows * cols;

    // Safety check
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, total_size) };

    // Call the inner function directly with the output slice
    tema_batch_inner_into(data, &sweep, Kernel::Auto, false, out_slice)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Return the number of rows (parameter combinations)
    Ok(rows)
}
