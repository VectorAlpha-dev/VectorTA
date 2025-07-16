//! # Jump Step Average (JSA)
//!
//! A simple smoothing indicator: each point is averaged with the value at `period` steps in the past.
//!
//! ## Parameters
//! - **period**: Look-back length for smoothing (default: 30).
//!
//! ## Errors
//! - **AllValuesNaN**: All input values are `NaN`.
//! - **InvalidPeriod**: `period` is zero or exceeds data length.
//! - **NotEnoughValidData**: Not enough valid data points for `period`.
//!
//! ## Returns
//! - **`Ok(JsaOutput)`** with `values: Vec<f64>`.
//! - **`Err(JsaError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel, alloc_with_nan_prefix, init_matrix_prefixes, make_uninit_matrix};
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

impl<'a> AsRef<[f64]> for JsaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            JsaData::Slice(slice) => slice,
            JsaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum JsaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct JsaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct JsaParams {
    pub period: Option<usize>,
}

impl Default for JsaParams {
    fn default() -> Self {
        Self { period: Some(30) }
    }
}

#[derive(Debug, Clone)]
pub struct JsaInput<'a> {
    pub data: JsaData<'a>,
    pub params: JsaParams,
}

impl<'a> JsaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: JsaParams) -> Self {
        Self {
            data: JsaData::Candles { candles: c, source: s },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: JsaParams) -> Self {
        Self {
            data: JsaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", JsaParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(30)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct JsaBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for JsaBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl JsaBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<JsaOutput, JsaError> {
        let p = JsaParams { period: self.period };
        let i = JsaInput::from_candles(c, "close", p);
        jsa_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<JsaOutput, JsaError> {
        let p = JsaParams { period: self.period };
        let i = JsaInput::from_slice(d, p);
        jsa_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<JsaStream, JsaError> {
        let p = JsaParams { period: self.period };
        JsaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum JsaError {
    #[error("jsa: Input data slice is empty.")]
    EmptyInputData,

    #[error("jsa: All values are NaN.")]
    AllValuesNaN,

    #[error("jsa: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("jsa: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },

    #[error("jsa: Invalid output buffer size: expected = {expected}, actual = {actual}")]
    InvalidOutputBuffer { expected: usize, actual: usize },
}

#[inline]
pub fn jsa(input: &JsaInput) -> Result<JsaOutput, JsaError> {
    jsa_with_kernel(input, Kernel::Auto)
}

pub fn jsa_with_kernel(input: &JsaInput, kernel: Kernel) -> Result<JsaOutput, JsaError> {
    let data: &[f64] = match &input.data {
        JsaData::Candles { candles, source } => source_type(candles, source),
        JsaData::Slice(sl) => sl,
    };

    // Check for empty input data
    if data.is_empty() {
        return Err(JsaError::EmptyInputData);
    }

    let first = data.iter().position(|x| !x.is_nan()).ok_or(JsaError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(JsaError::InvalidPeriod { period, data_len: len });
    }
    if (len - first) < period {
        return Err(JsaError::NotEnoughValidData { needed: period, valid: len - first });
    }

    let warm = first + period;
    let mut out = alloc_with_nan_prefix(len, warm);
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => jsa_scalar(data, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => jsa_avx2(data, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => jsa_avx512(data, period, first, &mut out),
            _ => unreachable!(),
        }
    }
    Ok(JsaOutput { values: out })
}

#[inline]
pub fn jsa_into(input: &JsaInput, out: &mut [f64]) -> Result<(), JsaError> {
    jsa_with_kernel_into(input, Kernel::Auto, out)
}

pub fn jsa_with_kernel_into(
    input: &JsaInput,
    kernel: Kernel,
    out: &mut [f64],
) -> Result<(), JsaError> {
    let data: &[f64] = match &input.data {
        JsaData::Candles { candles, source } => source_type(candles, source),
        JsaData::Slice(sl) => sl,
    };
    
    // Check for empty input data
    if data.is_empty() {
        return Err(JsaError::EmptyInputData);
    }
    
    let len = data.len();
    
    // Ensure output buffer is the correct size
    if out.len() != len {
        return Err(JsaError::InvalidOutputBuffer {
            expected: len,
            actual: out.len(),
        });
    }

    let first = data.iter().position(|x| !x.is_nan()).ok_or(JsaError::AllValuesNaN)?;
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(JsaError::InvalidPeriod { period, data_len: len });
    }
    if (len - first) < period {
        return Err(JsaError::NotEnoughValidData { needed: period, valid: len - first });
    }

    let warm = first + period;
    
    // Initialize NaN prefix
    out[..warm].fill(f64::NAN);
    
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => jsa_scalar(data, period, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => jsa_avx2(data, period, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => jsa_avx512(data, period, first, out),
            _ => unreachable!(),
        }
    }
    Ok(())
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn jsa_avx512(
    data: &[f64],
    period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        unsafe { jsa_avx512_short(data, period, first_valid, out) }
    } else {
        unsafe { jsa_avx512_long(data, period, first_valid, out) }
    }
}

#[inline]
pub fn jsa_scalar(
    data: &[f64],
    period: usize,
    first_val: usize,
    out: &mut [f64],
) {
    for i in (first_val + period)..data.len() {
        out[i] = (data[i] + data[i - period]) * 0.5;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
pub unsafe fn jsa_avx2(
    data: &[f64],
    period: usize,
    first_val: usize,
    out: &mut [f64],
) {
    // Just call scalar, as AVX2 stub
    jsa_scalar(data, period, first_val, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
pub unsafe fn jsa_avx512_short(
    data: &[f64],
    period: usize,
    first_val: usize,
    out: &mut [f64],
) {
    // Just call scalar, as AVX512-short stub
    jsa_scalar(data, period, first_val, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
pub unsafe fn jsa_avx512_long(
    data: &[f64],
    period: usize,
    first_val: usize,
    out: &mut [f64],
) {
    // Just call scalar, as AVX512-long stub
    jsa_scalar(data, period, first_val, out)
}

#[derive(Debug, Clone)]
pub struct JsaStream {
    period: usize,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,
}

impl JsaStream {
    pub fn try_new(params: JsaParams) -> Result<Self, JsaError> {
        let period = params.period.unwrap_or(30);
        if period == 0 {
            return Err(JsaError::InvalidPeriod { period, data_len: 0 });
        }
        Ok(Self {
            period,
            buffer: vec![f64::NAN; period],
            head: 0,
            filled: false,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        let out = if self.filled {
            let past = self.buffer[self.head];
            Some((value + past) * 0.5)
        } else {
            None
        };
        self.buffer[self.head] = value;
        self.head = (self.head + 1) % self.period;
        if !self.filled && self.head == 0 {
            self.filled = true;
        }
        out
    }
}

#[derive(Clone, Debug)]
pub struct JsaBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for JsaBatchRange {
    fn default() -> Self {
        Self {
            period: (30, 120, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct JsaBatchBuilder {
    range: JsaBatchRange,
    kernel: Kernel,
}

impl JsaBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<JsaBatchOutput, JsaError> {
        jsa_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<JsaBatchOutput, JsaError> {
        JsaBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<JsaBatchOutput, JsaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<JsaBatchOutput, JsaError> {
        JsaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn jsa_batch_with_kernel(
    data: &[f64],
    sweep: &JsaBatchRange,
    k: Kernel,
) -> Result<JsaBatchOutput, JsaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(JsaError::InvalidPeriod {
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
    jsa_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct JsaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<JsaParams>,
    pub rows: usize,
    pub cols: usize,
}
impl JsaBatchOutput {
    pub fn row_for_params(&self, p: &JsaParams) -> Option<usize> {
        self.combos.iter().position(|c| c.period.unwrap_or(30) == p.period.unwrap_or(30))
    }
    pub fn values_for(&self, p: &JsaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &JsaBatchRange) -> Vec<JsaParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(JsaParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn jsa_batch_slice(
    data: &[f64],
    sweep: &JsaBatchRange,
    kern: Kernel,
) -> Result<JsaBatchOutput, JsaError> {
    jsa_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn jsa_batch_par_slice(
    data: &[f64],
    sweep: &JsaBatchRange,
    kern: Kernel,
) -> Result<JsaBatchOutput, JsaError> {
    jsa_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn jsa_batch_inner(
    data: &[f64],
    sweep: &JsaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<JsaBatchOutput, JsaError> {
    // Check for empty input data
    if data.is_empty() {
        return Err(JsaError::EmptyInputData);
    }
    
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(JsaError::InvalidPeriod { period: 0, data_len: 0 });
    }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(JsaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(JsaError::NotEnoughValidData { needed: max_p, valid: data.len() - first });
    }
    let rows = combos.len();
    let cols = data.len();
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap())
        .collect();

    // -----------------------------------
    // 2.  allocate rows × cols as MaybeUninit
    // -----------------------------------
    let mut raw = make_uninit_matrix(rows, cols);
    // fill each row’s warm prefix with quiet-NaNs
    unsafe { init_matrix_prefixes(&mut raw, cols, &warm) };

    // -----------------------------------
    // 3.  per-row worker (writes into MaybeUninit)
    // -----------------------------------
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period  = combos[row].period.unwrap();

        // cast this row to &mut [f64] for the SIMD/scalar kernels
        let out_row = core::slice::from_raw_parts_mut(
            dst_mu.as_mut_ptr() as *mut f64,
            dst_mu.len(),
        );

        match kern {
            Kernel::Scalar => jsa_row_scalar(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2   => jsa_row_avx2   (data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => jsa_row_avx512 (data, first, period, out_row),
            _ => unreachable!(),
        }
    };

    // -----------------------------------
    // 4.  run all rows, filling `raw`
    // -----------------------------------
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

    // -----------------------------------
    // 5.  transmute to Vec<f64> (all cells written)
    // -----------------------------------
    let values: Vec<f64> = unsafe { std::mem::transmute(raw) };

    Ok(JsaBatchOutput { values, combos, rows, cols })
}

#[inline(always)]
fn jsa_batch_inner_into(
    data: &[f64],
    sweep: &JsaBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<(Vec<JsaParams>, usize, usize), JsaError> {
    // Check for empty input data
    if data.is_empty() {
        return Err(JsaError::EmptyInputData);
    }
    
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(JsaError::InvalidPeriod { period: 0, data_len: 0 });
    }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(JsaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(JsaError::NotEnoughValidData { needed: max_p, valid: data.len() - first });
    }
    let rows = combos.len();
    let cols = data.len();
    
    // Ensure output buffer is the correct size
    if out.len() != rows * cols {
        return Err(JsaError::InvalidOutputBuffer {
            expected: rows * cols,
            actual: out.len(),
        });
    }
    
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap())
        .collect();

    // Cast output to MaybeUninit for initialization
    let out_uninit = unsafe {
        std::slice::from_raw_parts_mut(
            out.as_mut_ptr() as *mut MaybeUninit<f64>,
            out.len()
        )
    };
    unsafe { init_matrix_prefixes(out_uninit, cols, &warm) };

    // ---------- closure that fills ONE row ---------------------------
    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();

        match kern {
            Kernel::Scalar => jsa_row_scalar(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => jsa_row_avx2(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => jsa_row_avx512(data, first, period, out_row),
            _ => unreachable!(),
        }
    };

    // ---------- run every row ----------------------------------------
    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out.par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in out.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in out.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    Ok((combos, rows, cols))
}

#[inline(always)]
unsafe fn jsa_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    for i in (first + period)..data.len() {
        out[i] = (data[i] + data[i - period]) * 0.5;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn jsa_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    // AVX2 stub - call scalar
    jsa_row_scalar(data, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn jsa_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        jsa_row_avx512_short(data, first, period, out);
    } else {
        jsa_row_avx512_long(data, first, period, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn jsa_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    // AVX512-short stub - call scalar
    jsa_row_scalar(data, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn jsa_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    // AVX512-long stub - call scalar
    jsa_row_scalar(data, first, period, out)
}

#[inline(always)]
pub fn expand_grid_jsa(r: &JsaBatchRange) -> Vec<JsaParams> {
    expand_grid(r)
}

// Python bindings
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, IntoPyArray};
#[cfg(feature = "python")]
use pyo3::types::PyDict;

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "jsa")]
pub fn jsa_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let slice_in = data.as_slice()?; // zero-copy, read-only view
    
    // Pre-allocate NumPy output buffer
    let out_arr = unsafe { PyArray1::<f64>::new(py, [slice_in.len()], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? }; // safe: we own the array
    
    // Prepare JSA input
    let params = JsaParams {
        period: Some(period),
    };
    let input = JsaInput::from_slice(slice_in, params);
    
    // Heavy lifting without the GIL
    py.allow_threads(|| -> Result<(), JsaError> {
        jsa_into(&input, slice_out)
    })
    .map_err(|e| PyValueError::new_err(format!("JSA error: {}", e)))?;
    
    Ok(out_arr)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "jsa_batch")]
pub fn jsa_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let slice_in = data.as_slice()?; // zero-copy, read-only view
    let sweep = JsaBatchRange {
        period: (period_start, period_end, period_step),
    };
    
    // Expand grid to get all combinations
    let combos = expand_grid(&sweep);
    if combos.is_empty() {
        return Err(PyValueError::new_err("Invalid parameter ranges"));
    }
    
    let rows = combos.len();
    let cols = slice_in.len();
    
    // Pre-allocate NumPy array (1-D, will reshape later)
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? }; // safe: we own the array
    
    // Heavy work without the GIL
    let (_, final_rows, final_cols) = py.allow_threads(|| -> Result<(Vec<JsaParams>, usize, usize), JsaError> {
        // Detect best kernel
        let kernel = match detect_best_batch_kernel() {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            _ => Kernel::Scalar,
        };
        
        // Use the new _into function with parallel=true
        jsa_batch_inner_into(slice_in, &sweep, kernel, true, slice_out)
    })
    .map_err(|e| PyValueError::new_err(format!("JSA batch error: {}", e)))?;
    
    // Extract metadata and convert to NumPy arrays for zero-copy
    let periods = combos
        .iter()
        .map(|c| c.period.unwrap())
        .collect::<Vec<_>>()
        .into_pyarray(py);
    
    // Reshape to 2D
    let out_2d = out_arr.reshape((final_rows, final_cols))?;
    
    // Create dictionary output
    let dict = PyDict::new(py);
    dict.set_item("values", out_2d)?;
    dict.set_item("periods", periods)?;
    
    Ok(dict)
}

// Note: jsa_batch_with_metadata_py is no longer needed since jsa_batch_py now returns metadata in the dictionary

// Note: jsa_batch_2d_py is no longer needed since jsa_batch_py now returns a 2D array in the dictionary

#[cfg(feature = "python")]
#[pyclass(name = "JsaStream")]
pub struct JsaStreamPy {
    inner: JsaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl JsaStreamPy {
    #[new]
    pub fn new(period: usize) -> PyResult<Self> {
        let params = JsaParams {
            period: Some(period),
        };
        match JsaStream::try_new(params) {
            Ok(stream) => Ok(Self { inner: stream }),
            Err(e) => Err(PyValueError::new_err(format!("JsaStream error: {}", e))),
        }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.inner.update(value)
    }
}

// WASM bindings
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn jsa_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = JsaParams {
        period: Some(period),
    };
    let input = JsaInput::from_slice(data, params);
    match jsa_with_kernel(&input, Kernel::Scalar) {
        Ok(output) => Ok(output.values),
        Err(e) => Err(JsValue::from_str(&format!("JSA error: {}", e))),
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn jsa_batch_js(
    data: &[f64],
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = JsaBatchRange {
        period: (period_start, period_end, period_step),
    };
    
    // Use the existing batch function with parallel=false for WASM
    jsa_batch_inner(data, &sweep, Kernel::Scalar, false)
        .map(|output| output.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn jsa_batch_metadata_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Vec<f64> {
    let periods: Vec<usize> = if period_step == 0 || period_start == period_end {
        vec![period_start]
    } else {
        (period_start..=period_end).step_by(period_step).collect()
    };
    
    let mut result = Vec::new();
    for &period in &periods {
        result.push(period as f64);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_jsa_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = JsaParams { period: None };
        let input = JsaInput::from_candles(&candles, "close", default_params);
        let output = jsa_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_jsa_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let expected_last_five = [61640.0, 61418.0, 61240.0, 61060.5, 60889.5];
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = JsaParams::default();
        let input = JsaInput::from_candles(&candles, "close", default_params);
        let result = jsa_with_kernel(&input, kernel)?;
        let start_idx = result.values.len() - 5;
        for (i, &val) in result.values[start_idx..].iter().enumerate() {
            let expected = expected_last_five[i];
            assert!((val - expected).abs() < 1e-5, "[{}] mismatch idx {}: got {}, expected {}", test_name, i, val, expected);
        }
        Ok(())
    }

    fn check_jsa_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = JsaParams { period: Some(0) };
        let input = JsaInput::from_slice(&input_data, params);
        let res = jsa_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] JSA should fail with zero period", test_name);
        Ok(())
    }

    fn check_jsa_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = JsaParams { period: Some(10) };
        let input = JsaInput::from_slice(&data_small, params);
        let res = jsa_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] JSA should fail with period exceeding length", test_name);
        Ok(())
    }

    fn check_jsa_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = JsaParams { period: Some(5) };
        let input = JsaInput::from_slice(&single_point, params);
        let res = jsa_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] JSA should fail with insufficient data", test_name);
        Ok(())
    }

    fn check_jsa_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = JsaParams { period: Some(10) };
        let first_input = JsaInput::from_candles(&candles, "close", first_params);
        let first_result = jsa_with_kernel(&first_input, kernel)?;

        let second_params = JsaParams { period: Some(5) };
        let second_input = JsaInput::from_slice(&first_result.values, second_params);
        let second_result = jsa_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());
        for i in 30..second_result.values.len() {
            assert!(second_result.values[i].is_finite(), "[{}] NaN at idx {}", test_name, i);
        }
        Ok(())
    }

    fn check_jsa_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 30;
        let input = JsaInput::from_candles(&candles, "close", JsaParams { period: Some(period) });
        let batch_output = jsa_with_kernel(&input, kernel)?.values;

        let mut stream = JsaStream::try_new(JsaParams { period: Some(period) })?;
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
            assert!(diff < 1e-9, "[{}] streaming mismatch at idx {}: batch={}, stream={}, diff={}", test_name, i, b, s, diff);
        }
        Ok(())
    }

    macro_rules! generate_all_jsa_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $( #[test]
                    fn [<$test_fn _scalar_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar);
                    }
                )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $( #[test]
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
    fn check_jsa_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test with multiple parameter combinations to better catch uninitialized memory bugs
        let test_periods = vec![2, 5, 10, 14, 20, 30, 50, 100, 200];
        let test_sources = vec!["open", "high", "low", "close", "hl2", "hlc3", "ohlc4"];

        for period in &test_periods {
            for source in &test_sources {
                let input = JsaInput::from_candles(
                    &candles,
                    source,
                    JsaParams {
                        period: Some(*period),
                    },
                );
                let output = jsa_with_kernel(&input, kernel)?;

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
                            "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} with period={}, source={}",
                            test_name, val, bits, i, period, source
                        );
                    }

                    // Check for init_matrix_prefixes poison (0x22222222_22222222)
                    if bits == 0x22222222_22222222 {
                        panic!(
                            "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} with period={}, source={}",
                            test_name, val, bits, i, period, source
                        );
                    }

                    // Check for make_uninit_matrix poison (0x33333333_33333333)
                    if bits == 0x33333333_33333333 {
                        panic!(
                            "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} with period={}, source={}",
                            test_name, val, bits, i, period, source
                        );
                    }
                }
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_jsa_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    generate_all_jsa_tests!(
        check_jsa_partial_params,
        check_jsa_accuracy,
        check_jsa_zero_period,
        check_jsa_period_exceeds_length,
        check_jsa_very_small_dataset,
        check_jsa_reinput,
        check_jsa_streaming,
        check_jsa_no_poison
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = JsaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = JsaParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());

        let expected = [61640.0, 61418.0, 61240.0, 61060.5, 60889.5];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!((v - expected[i]).abs() < 1e-5, "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}");
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

        // Test batch with multiple parameter combinations to better catch uninitialized memory bugs
        let test_sources = vec!["open", "high", "low", "close", "hl2", "hlc3", "ohlc4"];
        
        for source in &test_sources {
            // Test with comprehensive period ranges
            let output = JsaBatchBuilder::new()
                .kernel(kernel)
                .period_range(2, 200, 3)  // Wide range: 2 to 200 with step 3
                .apply_candles(&c, source)?;

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
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with source={}",
                        test, val, bits, row, col, idx, source
                    );
                }

                // Check for init_matrix_prefixes poison (0x22222222_22222222)
                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {}) with source={}",
                        test, val, bits, row, col, idx, source
                    );
                }

                // Check for make_uninit_matrix poison (0x33333333_33333333)
                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with source={}",
                        test, val, bits, row, col, idx, source
                    );
                }
            }
        }

        // Also test edge cases with very small and very large periods
        let edge_case_ranges = vec![(2, 5, 1), (190, 200, 2), (50, 100, 10)];
        for (start, end, step) in edge_case_ranges {
            let output = JsaBatchBuilder::new()
                .kernel(kernel)
                .period_range(start, end, step)
                .apply_candles(&c, "close")?;

            for (idx, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;

                if bits == 0x11111111_11111111 || bits == 0x22222222_22222222 || bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found poison value {} (0x{:016X}) at row {} col {} with range ({},{},{})",
                        test, val, bits, row, col, start, end, step
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
