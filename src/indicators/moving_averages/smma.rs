//! # Smoothed Moving Average (SMMA)
//!
//! The Smoothed Moving Average (SMMA) uses a recursive smoothing formula where the first value
//! is the mean of the first `period` points and subsequent values use a blend of the prior SMMA
//! and the new point. API matches ALMA in structure and extensibility, supporting streaming,
//! parameter sweeps, and SIMD feature stubs for AVX2/AVX512 kernels.
//!
//! ## Parameters
//! - **period**: Window size (number of data points).
//!
//! ## Errors
//! - **AllValuesNaN**: smma: All input data values are `NaN`.
//! - **InvalidPeriod**: smma: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: smma: Not enough valid data points for the requested `period`.
//!
//! ## Returns
//! - **`Ok(SmmaOutput)`** on success, containing a `Vec<f64>` matching the input.
//! - **`Err(SmmaError)`** otherwise.

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel, alloc_with_nan_prefix, init_matrix_prefixes, make_uninit_matrix};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use thiserror::Error;
use std::mem::MaybeUninit;

impl<'a> AsRef<[f64]> for SmmaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            SmmaData::Slice(slice) => slice,
            SmmaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum SmmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct SmmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct SmmaParams {
    pub period: Option<usize>,
}

impl Default for SmmaParams {
    fn default() -> Self {
        Self { period: Some(7) }
    }
}

#[derive(Debug, Clone)]
pub struct SmmaInput<'a> {
    pub data: SmmaData<'a>,
    pub params: SmmaParams,
}

impl<'a> SmmaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: SmmaParams) -> Self {
        Self { data: SmmaData::Candles { candles: c, source: s }, params: p }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: SmmaParams) -> Self {
        Self { data: SmmaData::Slice(sl), params: p }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", SmmaParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(7)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SmmaBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for SmmaBuilder {
    fn default() -> Self {
        Self { period: None, kernel: Kernel::Auto }
    }
}

impl SmmaBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<SmmaOutput, SmmaError> {
        let p = SmmaParams { period: self.period };
        let i = SmmaInput::from_candles(c, "close", p);
        smma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<SmmaOutput, SmmaError> {
        let p = SmmaParams { period: self.period };
        let i = SmmaInput::from_slice(d, p);
        smma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<SmmaStream, SmmaError> {
        let p = SmmaParams { period: self.period };
        SmmaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum SmmaError {
    #[error("smma: Input data slice is empty.")]
    EmptyInputData,
    #[error("smma: All values are NaN.")]
    AllValuesNaN,
    #[error("smma: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("smma: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn smma(input: &SmmaInput) -> Result<SmmaOutput, SmmaError> {
    smma_with_kernel(input, Kernel::Auto)
}

pub fn smma_with_kernel(input: &SmmaInput, kernel: Kernel) -> Result<SmmaOutput, SmmaError> {
    let data: &[f64] = match &input.data {
        SmmaData::Candles { candles, source } => source_type(candles, source),
        SmmaData::Slice(sl) => sl,
    };

    // Check for empty input first
    if data.is_empty() {
        return Err(SmmaError::EmptyInputData);
    }

    let first = data.iter().position(|x| !x.is_nan()).ok_or(SmmaError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(SmmaError::InvalidPeriod { period, data_len: len });
    }
    if (len - first) < period {
        return Err(SmmaError::NotEnoughValidData { needed: period, valid: len - first });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let warm = first + period;
    let mut out = alloc_with_nan_prefix(len, warm);

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                smma_scalar(data, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                smma_avx2(data, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                smma_avx512(data, period, first, &mut out)
            }
            _ => unreachable!(),
        }
    }
    Ok(SmmaOutput { values: out })
}

#[inline]
pub fn smma_scalar(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    let end = first + period;
    let sum: f64 = data[first..end].iter().sum();
    let mut prev = sum / period as f64;
    out[end - 1] = prev;
    for i in end..data.len() {
        prev = (prev * (period as f64 - 1.0) + data[i]) / (period as f64);
        out[i] = prev;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn smma_avx512(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    unsafe { smma_avx512_long(data, period, first, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn smma_avx2(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    smma_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn smma_avx512_short(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    smma_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn smma_avx512_long(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    smma_scalar(data, period, first, out)
}

#[inline(always)]
fn smma_prepare<'a>(
    input: &'a SmmaInput,
    kernel: Kernel,
) -> Result<
    (
        /*data*/ &'a [f64],
        /*period*/ usize,
        /*first*/ usize,
        /*chosen*/ Kernel,
    ),
    SmmaError,
> {
    let data: &[f64] = input.as_ref();
    
    // Check for empty input first
    if data.is_empty() {
        return Err(SmmaError::EmptyInputData);
    }
    
    let first = data.iter().position(|x| !x.is_nan()).ok_or(SmmaError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(SmmaError::InvalidPeriod { period, data_len: len });
    }
    if (len - first) < period {
        return Err(SmmaError::NotEnoughValidData { needed: period, valid: len - first });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    Ok((data, period, first, chosen))
}

#[inline(always)]
fn smma_compute_into(
    data: &[f64],
    period: usize,
    first: usize,
    kernel: Kernel,
    out: &mut [f64],
) {
    unsafe {
        match kernel {
            Kernel::Scalar | Kernel::ScalarBatch => {
                smma_scalar(data, period, first, out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                smma_avx2(data, period, first, out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                smma_avx512(data, period, first, out)
            }
            _ => unreachable!(),
        }
    }
}

#[inline]
pub fn expand_grid(r: &SmmaBatchRange) -> Vec<SmmaParams> {
    let axis_usize = |(start, end, step): (usize, usize, usize)| {
        if step == 0 || start == end { vec![start] } else { (start..=end).step_by(step).collect() }
    };
    axis_usize(r.period).into_iter().map(|p| SmmaParams { period: Some(p) }).collect()
}

#[derive(Debug, Clone)]
pub struct SmmaStream {
    period: usize,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,
    value: f64,
}

impl SmmaStream {
    pub fn try_new(params: SmmaParams) -> Result<Self, SmmaError> {
        let period = params.period.unwrap_or(7);
        if period == 0 {
            return Err(SmmaError::InvalidPeriod { period, data_len: 0 });
        }
        Ok(Self {
            period,
            buffer: vec![f64::NAN; period],
            head: 0,
            filled: false,
            value: f64::NAN,
        })
    }
    #[inline(always)]
    pub fn update(&mut self, v: f64) -> Option<f64> {
        self.buffer[self.head] = v;
        self.head = (self.head + 1) % self.period;
        if !self.filled && self.head == 0 {
            self.filled = true;
            let sum: f64 = self.buffer.iter().sum();
            self.value = sum / self.period as f64;
            return Some(self.value);
        }
        if self.filled {
            self.value = (self.value * (self.period as f64 - 1.0) + v) / (self.period as f64);
            Some(self.value)
        
            } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub struct SmmaBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for SmmaBatchRange {
    fn default() -> Self {
        Self { period: (7, 100, 1) }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SmmaBatchBuilder {
    range: SmmaBatchRange,
    kernel: Kernel,
}

impl SmmaBatchBuilder {
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

    pub fn apply_slice(self, data: &[f64]) -> Result<SmmaBatchOutput, SmmaError> {
        smma_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<SmmaBatchOutput, SmmaError> {
        SmmaBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<SmmaBatchOutput, SmmaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<SmmaBatchOutput, SmmaError> {
        SmmaBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
    }
}

pub fn smma_batch_with_kernel(data: &[f64], sweep: &SmmaBatchRange, k: Kernel) -> Result<SmmaBatchOutput, SmmaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(SmmaError::InvalidPeriod { period: 0, data_len: 0 })
        }
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    smma_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct SmmaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<SmmaParams>,
    pub rows: usize,
    pub cols: usize,
}
impl SmmaBatchOutput {
    pub fn row_for_params(&self, p: &SmmaParams) -> Option<usize> {
        self.combos.iter().position(|c| c.period.unwrap_or(7) == p.period.unwrap_or(7))
    }
    pub fn values_for(&self, p: &SmmaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
pub fn smma_batch_slice(data: &[f64], sweep: &SmmaBatchRange, kern: Kernel) -> Result<SmmaBatchOutput, SmmaError> {
    smma_batch_inner(data, sweep, kern, false)
}
#[inline(always)]
pub fn smma_batch_par_slice(data: &[f64], sweep: &SmmaBatchRange, kern: Kernel) -> Result<SmmaBatchOutput, SmmaError> {
    smma_batch_inner(data, sweep, kern, true)
}
#[inline(always)]
fn smma_batch_inner(data: &[f64], sweep: &SmmaBatchRange, kern: Kernel, parallel: bool) -> Result<SmmaBatchOutput, SmmaError> {
    // Check for empty input first
    if data.is_empty() {
        return Err(SmmaError::EmptyInputData);
    }
    
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(SmmaError::InvalidPeriod { period: 0, data_len: 0 });
    }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(SmmaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(SmmaError::NotEnoughValidData { needed: max_p, valid: data.len() - first });
    }
    let rows = combos.len();
    let cols = data.len();

    // ------------------------------------------------------------------
    // 1.  Figure out how long each row’s NaN prefix should be
    // ------------------------------------------------------------------
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap())
        .collect();

    // ------------------------------------------------------------------
    // 2.  Allocate rows × cols uninitialised and write the NaN prefixes
    // ------------------------------------------------------------------
    let mut raw = make_uninit_matrix(rows, cols);
    unsafe { init_matrix_prefixes(&mut raw, cols, &warm); }

    // ------------------------------------------------------------------
    // 3.  Closure that computes ONE row in-place
    //     - receives the row as &mut [MaybeUninit<f64>]
    //     - casts it to &mut [f64] after the prefix
    // ------------------------------------------------------------------
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();

        // cast this row to plain f64s
        let out_row = core::slice::from_raw_parts_mut(
            dst_mu.as_mut_ptr() as *mut f64,
            dst_mu.len(),
        );

        match kern {
            Kernel::Scalar => smma_row_scalar(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2   => smma_row_avx2  (data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => smma_row_avx512(data, first, period, out_row),
            _ => unreachable!(),
        }
    };

    // ------------------------------------------------------------------
    // 4.  Run every row (optionally in parallel) straight into `raw`
    // ------------------------------------------------------------------
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

    // ------------------------------------------------------------------
    // 5.  All elements are now initialised – transmute to Vec<f64>
    // ------------------------------------------------------------------
    let values: Vec<f64> = unsafe { core::mem::transmute(raw) };

    Ok(SmmaBatchOutput { values, combos, rows, cols })
}

#[inline(always)]
fn smma_batch_inner_into(
    data: &[f64],
    sweep: &SmmaBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<SmmaParams>, SmmaError> {
    // Check for empty input first
    if data.is_empty() {
        return Err(SmmaError::EmptyInputData);
    }
    
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(SmmaError::InvalidPeriod { period: 0, data_len: 0 });
    }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(SmmaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(SmmaError::NotEnoughValidData { needed: max_p, valid: data.len() - first });
    }
    
    let rows = combos.len();
    let cols = data.len();
    
    // Collect warm-up lengths per row
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap())
        .collect();
    
    // SAFETY: We're reinterpreting the output slice as MaybeUninit to use the efficient
    // init_matrix_prefixes function. This is safe because:
    // 1. MaybeUninit<T> has the same layout as T
    // 2. We ensure all values are written before the slice is used again
    let out_uninit = unsafe {
        std::slice::from_raw_parts_mut(
            out.as_mut_ptr() as *mut MaybeUninit<f64>,
            out.len()
        )
    };
    
    unsafe { init_matrix_prefixes(out_uninit, cols, &warm) };
    
    // Closure that writes one row
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();
        
        // Cast the row slice to f64
        let dst = core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());
        
        match kern {
            Kernel::Scalar | Kernel::ScalarBatch => {
                smma_row_scalar(data, first, period, dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                smma_row_avx2(data, first, period, dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                smma_row_avx512(data, first, period, dst)
            }
            _ => unreachable!(),
        }
    };
    
    // Run every row kernel
    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_uninit.par_chunks_mut(cols)
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

#[inline(always)]
pub unsafe fn smma_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    smma_scalar(data, period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn smma_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    smma_scalar(data, period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn smma_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    if period <= 32 {
        smma_row_avx512_short(data, first, period, out);
    
        } else {
        smma_row_avx512_long(data, first, period, out);
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn smma_row_avx512_short(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    smma_scalar(data, period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn smma_row_avx512_long(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    smma_scalar(data, period, first, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;

    fn check_smma_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = SmmaInput::from_candles(&candles, "close", SmmaParams { period: None });
        let output = smma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }
    fn check_smma_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = SmmaInput::from_candles(&candles, "close", SmmaParams::default());
        let result = smma_with_kernel(&input, kernel)?;
        let expected_last_five = [59434.4, 59398.2, 59346.9, 59319.4, 59224.5];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(diff < 1e-1, "[{}] SMMA {:?} mismatch at idx {}: got {}, expected {}", test_name, kernel, i, val, expected_last_five[i]);
        }
        Ok(())
    }
    fn check_smma_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = SmmaInput::with_default_candles(&candles);
        match input.data {
            SmmaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected SmmaData::Candles"),
        }
        let output = smma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }
    fn check_smma_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = SmmaParams { period: Some(0) };
        let input = SmmaInput::from_slice(&input_data, params);
        let res = smma_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] SMMA should fail with zero period", test_name);
        Ok(())
    }
    fn check_smma_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = SmmaParams { period: Some(10) };
        let input = SmmaInput::from_slice(&data_small, params);
        let res = smma_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] SMMA should fail with period exceeding length", test_name);
        Ok(())
    }
    fn check_smma_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = SmmaParams { period: Some(9) };
        let input = SmmaInput::from_slice(&single_point, params);
        let res = smma_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] SMMA should fail with insufficient data", test_name);
        Ok(())
    }
    fn check_smma_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: Vec<f64> = vec![];
        let params = SmmaParams { period: Some(7) };
        let input = SmmaInput::from_slice(&empty, params);
        let res = smma_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] SMMA should fail with empty input", test_name);
        if let Err(SmmaError::EmptyInputData) = res {
            // Good, expected error type
        } else {
            panic!("[{}] Expected EmptyInputData error, got {:?}", test_name, res);
        }
        Ok(())
    }
    fn check_smma_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = SmmaParams { period: Some(7) };
        let first_input = SmmaInput::from_candles(&candles, "close", first_params);
        let first_result = smma_with_kernel(&first_input, kernel)?;
        let second_params = SmmaParams { period: Some(5) };
        let second_input = SmmaInput::from_slice(&first_result.values, second_params);
        let second_result = smma_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }
    fn check_smma_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = SmmaInput::from_candles(&candles, "close", SmmaParams { period: Some(7) });
        let res = smma_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 240 {
            for (i, &val) in res.values[240..].iter().enumerate() {
                assert!(!val.is_nan(), "[{}] Found unexpected NaN at out-index {}", test_name, 240 + i);
            }
        }
        Ok(())
    }
    fn check_smma_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 7;
        let input = SmmaInput::from_candles(&candles, "close", SmmaParams { period: Some(period) });
        let batch_output = smma_with_kernel(&input, kernel)?.values;
        let mut stream = SmmaStream::try_new(SmmaParams { period: Some(period) })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(smma_val) => stream_values.push(smma_val),
                None => stream_values.push(f64::NAN),
            }
        }
        assert_eq!(batch_output.len(), stream_values.len());
        for (i, (&b, &s)) in batch_output.iter().zip(stream_values.iter()).enumerate() {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(diff < 1e-9, "[{}] SMMA streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}", test_name, i, b, s, diff);
        }
        Ok(())
    }
    #[cfg(feature = "proptest")]
    fn check_smma_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Property test strategy: generate period and matching data length
        let strat = (1usize..=64) // period
            .prop_flat_map(|period| {
                (
                    prop::collection::vec(
                        (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                        period..400, // len >= period
                    ),
                    Just(period),
                )
            });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, period)| {
                let params = SmmaParams { period: Some(period) };
                let input = SmmaInput::from_slice(&data, params);
                
                match smma_with_kernel(&input, kernel) {
                    Ok(output) => {
                        // Property 1: Output length equals input length
                        prop_assert_eq!(output.values.len(), data.len());
                        
                        // Property 2: First period-1 values are NaN
                        for i in 0..period-1 {
                            prop_assert!(output.values[i].is_nan(), 
                                "Expected NaN at index {} but got {}", i, output.values[i]);
                        }
                        
                        // Property 3: Values after warmup are finite (not NaN or inf)
                        for i in period-1..output.values.len() {
                            prop_assert!(output.values[i].is_finite(), 
                                "Expected finite value at index {} but got {}", i, output.values[i]);
                        }
                        
                        // Property 4: SMMA is bounded by min/max of input window
                        if let Some(first_valid) = data.iter().position(|&x| !x.is_nan()) {
                            for i in (first_valid + period - 1)..output.values.len() {
                                let window_start = i.saturating_sub(period - 1);
                                let window = &data[window_start..=i];
                                if let (Some(&min), Some(&max)) = (
                                    window.iter().filter(|x| x.is_finite()).min_by(|a, b| a.partial_cmp(b).unwrap()),
                                    window.iter().filter(|x| x.is_finite()).max_by(|a, b| a.partial_cmp(b).unwrap())
                                ) {
                                    prop_assert!(output.values[i] >= min && output.values[i] <= max,
                                        "SMMA value {} at index {} outside bounds [{}, {}]", 
                                        output.values[i], i, min, max);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        // If it errors, it should be for a valid reason
                        prop_assert!(
                            matches!(e, SmmaError::EmptyInputData | 
                                       SmmaError::AllValuesNaN | 
                                       SmmaError::InvalidPeriod { .. } |
                                       SmmaError::NotEnoughValidData { .. }),
                            "Unexpected error type: {:?}", e
                        );
                    }
                }
                Ok(())
            })?;
        
        Ok(())
    }

    // Check for poison values in single output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_smma_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test with default parameters
        let input = SmmaInput::from_candles(&candles, "close", SmmaParams::default());
        let output = smma_with_kernel(&input, kernel)?;

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
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {}",
                    test_name, val, bits, i
                );
            }

            // Check for init_matrix_prefixes poison (0x22222222_22222222)
            if bits == 0x22222222_22222222 {
                panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {}",
                    test_name, val, bits, i
                );
            }

            // Check for make_uninit_matrix poison (0x33333333_33333333)
            if bits == 0x33333333_33333333 {
                panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {}",
                    test_name, val, bits, i
                );
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_smma_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    macro_rules! generate_all_smma_tests {
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
    generate_all_smma_tests!(
        check_smma_partial_params,
        check_smma_accuracy,
        check_smma_default_candles,
        check_smma_zero_period,
        check_smma_period_exceeds_length,
        check_smma_very_small_dataset,
        check_smma_empty_input,
        check_smma_reinput,
        check_smma_nan_handling,
        check_smma_streaming,
        check_smma_no_poison
    );
    
    #[cfg(feature = "proptest")]
    generate_all_smma_tests!(check_smma_property);
    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = SmmaBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;
        let def = SmmaParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [59434.4, 59398.2, 59346.9, 59319.4, 59224.5];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!((v - expected[i]).abs() < 1e-1, "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}");
        }
        Ok(())
    }

    // Check for poison values in batch output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test batch with multiple parameter combinations
        let output = SmmaBatchBuilder::new()
            .kernel(kernel)
            .period_range(10, 30, 10)
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
    fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
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

#[cfg(feature = "python")]
#[pyfunction(name = "smma")]
#[pyo3(signature = (data, period, kernel=None))]
/// Compute the Smoothed Moving Average (SMMA) of the input data.
///
/// SMMA uses a recursive smoothing formula where the first value is the mean
/// of the first `period` points and subsequent values use a blend of the prior
/// SMMA and the new point.
///
/// Parameters:
/// -----------
/// data : np.ndarray
///     Input data array (float64).
/// period : int
///     Number of data points in the moving average window.
/// kernel : str, optional
///     Computation kernel to use: 'auto', 'scalar', 'avx2', 'avx512'.
///     Default is 'auto' which auto-detects the best available.
///
/// Returns:
/// --------
/// np.ndarray
///     Array of SMMA values, same length as input.
///
/// Raises:
/// -------
/// ValueError
///     If inputs are invalid (period <= 0, insufficient data, etc).
pub fn smma_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let slice_in = data.as_slice()?;
    
    // Parse kernel string to enum
    let kern = match kernel {
        None | Some("auto") => Kernel::Auto,
        Some("scalar") => Kernel::Scalar,
        Some("avx2") => Kernel::Avx2,
        Some("avx512") => Kernel::Avx512,
        Some(k) => return Err(PyValueError::new_err(format!("Unknown kernel: {}", k))),
    };
    
    // Build input struct
    let params = SmmaParams { period: Some(period) };
    let smma_in = SmmaInput::from_slice(slice_in, params);
    
    // Allocate NumPy output buffer
    let out_arr = unsafe { PyArray1::<f64>::new(py, [slice_in.len()], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };
    
    // Heavy lifting without the GIL
    py.allow_threads(|| -> Result<(), SmmaError> {
        let (data, period, first, chosen) = smma_prepare(&smma_in, kern)?;
        // Initialize entire output with NaN first
        slice_out.fill(f64::NAN);
        // Compute SMMA starting from the appropriate index
        smma_compute_into(data, period, first, chosen, slice_out);
        Ok(())
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    
    Ok(out_arr.into())
}

#[cfg(feature = "python")]
#[pyclass(name = "SmmaStream")]
pub struct SmmaStreamPy {
    stream: SmmaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl SmmaStreamPy {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        let params = SmmaParams { period: Some(period) };
        let stream = SmmaStream::try_new(params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(SmmaStreamPy { stream })
    }
    
    /// Updates the stream with a new value and returns the calculated SMMA value.
    /// Returns `None` if the buffer is not yet full.
    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "smma_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
/// Compute SMMA for multiple period values in a single pass.
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
pub fn smma_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let slice_in = data.as_slice()?;
    
    let sweep = SmmaBatchRange { period: period_range };
    
    // Expand grid to know rows*cols
    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();
    
    // Pre-allocate NumPy array
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };
    
    // Parse kernel string to enum
    let kern = match kernel {
        None | Some("auto") => Kernel::Auto,
        Some("scalar") => Kernel::ScalarBatch,
        Some("avx2") => Kernel::Avx2Batch,
        Some("avx512") => Kernel::Avx512Batch,
        Some(k) => return Err(PyValueError::new_err(format!("Unknown kernel: {}", k))),
    };
    
    // Heavy work without the GIL
    let combos = py.allow_threads(|| {
        // Resolve Kernel::Auto to a specific kernel
        let kernel = match kern {
            Kernel::Auto => detect_best_batch_kernel(),
            k => k,
        };
        let simd = match kernel {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            Kernel::ScalarBatch => Kernel::Scalar,
            _ => unreachable!(),
        };
        // Use the _into variant that writes directly to our pre-allocated buffer
        smma_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    
    // Build dict with the GIL
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
    
    Ok(dict.into())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "smma")]
/// Compute the Smoothed Moving Average (SMMA) of the input data.
pub fn smma_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = SmmaParams { period: Some(period) };
    let input = SmmaInput::from_slice(data, params);
    
    match smma(&input) {
        Ok(output) => Ok(output.values),
        Err(e) => Err(JsValue::from_str(&e.to_string())),
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "smma_batch")]
/// Compute SMMA for multiple period values in a single pass.
pub fn smma_batch_js(
    data: &[f64],
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = SmmaBatchRange {
        period: (period_start, period_end, period_step),
    };
    
    smma_batch_with_kernel(data, &sweep, Kernel::Auto)
        .map(|output| output.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "smma_batch_metadata")]
/// Get metadata about the batch computation (periods used)
pub fn smma_batch_metadata_js(
    period_start: usize,
    period_end: usize,
    period_step: usize
) -> Vec<usize> {
    let range = SmmaBatchRange {
        period: (period_start, period_end, period_step),
    };
    let combos = expand_grid(&range);
    combos.iter().map(|c| c.period.unwrap_or(7)).collect()
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "smma_batch_rows_cols")]
/// Get the dimensions of the batch output
pub fn smma_batch_rows_cols_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
    data_len: usize
) -> Vec<usize> {
    let range = SmmaBatchRange {
        period: (period_start, period_end, period_step),
    };
    let combos = expand_grid(&range);
    vec![combos.len(), data_len]
}

