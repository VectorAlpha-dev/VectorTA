//! # Tilson T3 Moving Average (T3)
//!
//! A specialized moving average that applies multiple iterations of an
//! exponential smoothing algorithm, enhanced by a volume factor (`v_factor`)
//! parameter. API matches alma.rs. SIMD/AVX variants forward to scalar logic by default.
//!
//! ## Parameters
//! - **period**: The look-back period for smoothing (defaults to 5).
//! - **volume_factor**: Controls the depth of the T3 smoothing. Range [0.0, 1.0], higher values = more smoothing (default 0.0).
//!
//! ## Errors
//! - **AllValuesNaN**: tilson: All input data values are `NaN`.
//! - **InvalidPeriod**: tilson: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: tilson: Not enough valid data points for the requested `period`.
//! - **InvalidVolumeFactor**: tilson: `volume_factor` is `NaN` or infinite.
//!
//! ## Returns
//! - **`Ok(TilsonOutput)`** on success, containing a `Vec<f64>` matching the input.
//! - **`Err(TilsonError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel, make_uninit_matrix, init_matrix_prefixes, alloc_with_nan_prefix};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;
use std::mem::MaybeUninit;

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

impl<'a> AsRef<[f64]> for TilsonInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            TilsonData::Slice(slice) => slice,
            TilsonData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum TilsonData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct TilsonOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct TilsonParams {
    pub period: Option<usize>,
    pub volume_factor: Option<f64>,
}

impl Default for TilsonParams {
    fn default() -> Self {
        Self {
            period: Some(5),
            volume_factor: Some(0.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TilsonInput<'a> {
    pub data: TilsonData<'a>,
    pub params: TilsonParams,
}

impl<'a> TilsonInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: TilsonParams) -> Self {
        Self {
            data: TilsonData::Candles { candles: c, source: s },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: TilsonParams) -> Self {
        Self {
            data: TilsonData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", TilsonParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(5)
    }
    #[inline]
    pub fn get_volume_factor(&self) -> f64 {
        self.params.volume_factor.unwrap_or(0.0)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct TilsonBuilder {
    period: Option<usize>,
    volume_factor: Option<f64>,
    kernel: Kernel,
}

impl Default for TilsonBuilder {
    fn default() -> Self {
        Self {
            period: None,
            volume_factor: None,
            kernel: Kernel::Auto,
        }
    }
}

impl TilsonBuilder {
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
    pub fn volume_factor(mut self, v: f64) -> Self {
        self.volume_factor = Some(v);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<TilsonOutput, TilsonError> {
        let p = TilsonParams {
            period: self.period,
            volume_factor: self.volume_factor,
        };
        let i = TilsonInput::from_candles(c, "close", p);
        tilson_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<TilsonOutput, TilsonError> {
        let p = TilsonParams {
            period: self.period,
            volume_factor: self.volume_factor,
        };
        let i = TilsonInput::from_slice(d, p);
        tilson_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<TilsonStream, TilsonError> {
        let p = TilsonParams {
            period: self.period,
            volume_factor: self.volume_factor,
        };
        TilsonStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum TilsonError {
    #[error("tilson: Input data slice is empty.")]
    EmptyInputData,

    #[error("tilson: All values are NaN.")]
    AllValuesNaN,

    #[error("tilson: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("tilson: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },

    #[error("tilson: Invalid volume factor: {v_factor}")]
    InvalidVolumeFactor { v_factor: f64 },
}

#[inline]
pub fn tilson(input: &TilsonInput) -> Result<TilsonOutput, TilsonError> {
    tilson_with_kernel(input, Kernel::Auto)
}

pub fn tilson_with_kernel(input: &TilsonInput, kernel: Kernel) -> Result<TilsonOutput, TilsonError> {
    let data: &[f64] = match &input.data {
        TilsonData::Candles { candles, source } => source_type(candles, source),
        TilsonData::Slice(sl) => sl,
    };

    if data.is_empty() {
        return Err(TilsonError::EmptyInputData);
    }

    let first = data.iter().position(|x| !x.is_nan()).ok_or(TilsonError::AllValuesNaN)?;

    let len = data.len();
    let period = input.get_period();
    let v_factor = input.get_volume_factor();

    if period == 0 || period > len {
        return Err(TilsonError::InvalidPeriod { period, data_len: len });
    }
    if (len - first) < period {
        return Err(TilsonError::NotEnoughValidData { needed: period, valid: len - first });
    }
    if v_factor.is_nan() || v_factor.is_infinite() {
        return Err(TilsonError::InvalidVolumeFactor { v_factor });
    }

    let lookback_total = 6 * (period - 1);
    if (len - first) < lookback_total + 1 {
        return Err(TilsonError::NotEnoughValidData { needed: lookback_total + 1, valid: len - first });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let lookback_total = 6 * (period - 1);        // first real value appears here
    let warm           = first + lookback_total;
    let mut out        = alloc_with_nan_prefix(len, warm);

    // ----------- run the chosen kernel, filling `out` in-place
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch =>
                tilson_scalar (data, period, v_factor, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2   | Kernel::Avx2Batch   =>
                tilson_avx2 (data, period, v_factor, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch =>
                tilson_avx512(data, period, v_factor, first, &mut out),
            _ => unreachable!(),
        }?;    // <-- OK!
    }

    Ok(TilsonOutput { values: out })
}

#[inline]
pub fn tilson_scalar(
    data:        &[f64],
    period:      usize,
    v_factor:    f64,
    first_valid: usize,
    out:         &mut [f64],
) -> Result<(), TilsonError> {
    let len            = data.len();
    let lookback_total = 6 * (period - 1);
    debug_assert_eq!(len, out.len());

    if len == 0 || period == 0 || v_factor.is_nan() || v_factor.is_infinite() || len - first_valid < period {
        return Err(TilsonError::InvalidPeriod { period, data_len: len });
    }
    if lookback_total + first_valid >= len {
        return Err(TilsonError::NotEnoughValidData { needed: lookback_total + 1, valid: len - first_valid });
    }

    let k = 2.0 / (period as f64 + 1.0);
    let one_minus_k = 1.0 - k;

    let temp = v_factor * v_factor;
    let c1 = -(temp * v_factor);
    let c2 = 3.0 * (temp - c1);
    let c3 = -6.0 * temp - 3.0 * (v_factor - c1);
    let c4 = 1.0 + 3.0 * v_factor - c1 + 3.0 * temp;

    let mut today = 0_usize;
    let mut temp_real;
    let mut e1;
    let mut e2;
    let mut e3;
    let mut e4;
    let mut e5;
    let mut e6;

    temp_real = 0.0;
    for i in 0..period {
        temp_real += data[first_valid + today + i];
    }
    e1 = temp_real / (period as f64);
    today += period;

    temp_real = e1;
    for _ in 1..period {
        e1 = k * data[first_valid + today] + one_minus_k * e1;
        temp_real += e1;
        today += 1;
    }
    e2 = temp_real / (period as f64);

    temp_real = e2;
    for _ in 1..period {
        e1 = k * data[first_valid + today] + one_minus_k * e1;
        e2 = k * e1 + one_minus_k * e2;
        temp_real += e2;
        today += 1;
    }
    e3 = temp_real / (period as f64);

    temp_real = e3;
    for _ in 1..period {
        e1 = k * data[first_valid + today] + one_minus_k * e1;
        e2 = k * e1 + one_minus_k * e2;
        e3 = k * e2 + one_minus_k * e3;
        temp_real += e3;
        today += 1;
    }
    e4 = temp_real / (period as f64);

    temp_real = e4;
    for _ in 1..period {
        e1 = k * data[first_valid + today] + one_minus_k * e1;
        e2 = k * e1 + one_minus_k * e2;
        e3 = k * e2 + one_minus_k * e3;
        e4 = k * e3 + one_minus_k * e4;
        temp_real += e4;
        today += 1;
    }
    e5 = temp_real / (period as f64);

    temp_real = e5;
    for _ in 1..period {
        e1 = k * data[first_valid + today] + one_minus_k * e1;
        e2 = k * e1 + one_minus_k * e2;
        e3 = k * e2 + one_minus_k * e3;
        e4 = k * e3 + one_minus_k * e4;
        e5 = k * e4 + one_minus_k * e5;
        temp_real += e5;
        today += 1;
    }
    e6 = temp_real / (period as f64);

    let start_idx = first_valid + lookback_total;
    let end_idx = len - 1;

    let mut idx = start_idx;
    if idx < len {
        out[idx]    = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3;
    }
    idx += 1;

    while (first_valid + today) <= end_idx {
        e1 = k * data[first_valid + today] + one_minus_k * e1;
        e2 = k * e1 + one_minus_k * e2;
        e3 = k * e2 + one_minus_k * e3;
        e4 = k * e3 + one_minus_k * e4;
        e5 = k * e4 + one_minus_k * e5;
        e6 = k * e5 + one_minus_k * e6;

        if idx < len {
            out[idx]    = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3;
        }

        today += 1;
        idx += 1;
    }

    Ok(())
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn tilson_avx512(
    data: &[f64],
    period: usize,
    v_factor: f64,
    first_valid: usize,
    out: &mut [f64],
) -> Result<(), TilsonError> {
    tilson_scalar(data, period, v_factor, first_valid, out)
}

#[inline]
pub fn tilson_avx2(
    data: &[f64],
    period: usize,
    v_factor: f64,
    first_valid: usize,
    out: &mut [f64],
) -> Result<(), TilsonError> {
    tilson_scalar(data, period, v_factor, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn tilson_avx512_short(
    data: &[f64],
    period: usize,
    v_factor: f64,
    first_valid: usize,
    out: &mut [f64],
) -> Result<(), TilsonError> {
    tilson_scalar(data, period, v_factor, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn tilson_avx512_long(
    data: &[f64],
    period: usize,
    v_factor: f64,
    first_valid: usize,
    out: &mut [f64],
) -> Result<(), TilsonError> {
    tilson_scalar(data, period, v_factor, first_valid, out)
}

#[inline]
pub fn tilson_batch_with_kernel(
    data: &[f64],
    sweep: &TilsonBatchRange,
    k: Kernel,
) -> Result<TilsonBatchOutput, TilsonError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(TilsonError::InvalidPeriod { period: 0, data_len: 0 });
        }
    };

    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    tilson_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct TilsonBatchRange {
    pub period: (usize, usize, usize),
    pub volume_factor: (f64, f64, f64),
}

impl Default for TilsonBatchRange {
    fn default() -> Self {
        Self {
            period: (5, 40, 1),
            volume_factor: (0.0, 1.0, 0.1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TilsonBatchBuilder {
    range: TilsonBatchRange,
    kernel: Kernel,
}

impl TilsonBatchBuilder {
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
    pub fn volume_factor_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.volume_factor = (start, end, step);
        self
    }
    #[inline]
    pub fn volume_factor_static(mut self, v: f64) -> Self {
        self.range.volume_factor = (v, v, 0.0);
        self
    }

    pub fn apply_slice(self, data: &[f64]) -> Result<TilsonBatchOutput, TilsonError> {
        tilson_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<TilsonBatchOutput, TilsonError> {
        TilsonBatchBuilder::new().kernel(k).apply_slice(data)
    }

    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<TilsonBatchOutput, TilsonError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }

    pub fn with_default_candles(c: &Candles) -> Result<TilsonBatchOutput, TilsonError> {
        TilsonBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct TilsonBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<TilsonParams>,
    pub rows: usize,
    pub cols: usize,
}
impl TilsonBatchOutput {
    pub fn row_for_params(&self, p: &TilsonParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(5) == p.period.unwrap_or(5)
                && (c.volume_factor.unwrap_or(0.0) - p.volume_factor.unwrap_or(0.0)).abs() < 1e-12
        })
    }
    pub fn values_for(&self, p: &TilsonParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &TilsonBatchRange) -> Vec<TilsonParams> {
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
    let v_factors = axis_f64(r.volume_factor);

    let mut out = Vec::with_capacity(periods.len() * v_factors.len());
    for &p in &periods {
        for &v in &v_factors {
            out.push(TilsonParams {
                period: Some(p),
                volume_factor: Some(v),
            });
        }
    }
    out
}

#[inline(always)]
pub fn tilson_batch_slice(
    data: &[f64],
    sweep: &TilsonBatchRange,
    kern: Kernel,
) -> Result<TilsonBatchOutput, TilsonError> {
    tilson_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn tilson_batch_par_slice(
    data: &[f64],
    sweep: &TilsonBatchRange,
    kern: Kernel,
) -> Result<TilsonBatchOutput, TilsonError> {
    tilson_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn tilson_batch_inner(
    data: &[f64],
    sweep: &TilsonBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<TilsonBatchOutput, TilsonError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(TilsonError::InvalidPeriod { period: 0, data_len: 0 });
    }
    
    if data.is_empty() {
        return Err(TilsonError::EmptyInputData);
    }
    
    let first = data.iter().position(|x| !x.is_nan()).ok_or(TilsonError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < 6 * (max_p - 1) + 1 {
        return Err(TilsonError::NotEnoughValidData {
            needed: 6 * (max_p - 1) + 1,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + 6 * (c.period.unwrap() - 1))
        .collect();

    // ------------- 1. allocate uninitialised & stamp NaN prefixes
    let mut raw = make_uninit_matrix(rows, cols);
    unsafe { init_matrix_prefixes(&mut raw, cols, &warm) };

    // ------------- 2. worker that fills ONE row -------------
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period   = combos[row].period.unwrap();
        let v_factor = combos[row].volume_factor.unwrap();

        // cast this row to &mut [f64]
        let out_row = core::slice::from_raw_parts_mut(
            dst_mu.as_mut_ptr() as *mut f64,
            dst_mu.len(),
        );
        tilson_row_scalar(data, first, period, v_factor, out_row);
    };

    // ------------- 3. run every row in (parallel) iterator ---
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

    // ------------- 4. transmute to a plain Vec<f64> ----------
    let values: Vec<f64> = unsafe { std::mem::transmute(raw) };

    Ok(TilsonBatchOutput { values, combos, rows, cols })
}

#[inline(always)]
pub unsafe fn tilson_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    v_factor: f64,
    out: &mut [f64],
) {
    tilson_scalar(data, period, v_factor, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn tilson_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    v_factor: f64,
    out: &mut [f64],
) {
    tilson_row_scalar(data, first, period, v_factor, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn tilson_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    v_factor: f64,
    out: &mut [f64],
) {
    tilson_row_scalar(data, first, period, v_factor, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn tilson_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    v_factor: f64,
    out: &mut [f64],
) {
    tilson_row_scalar(data, first, period, v_factor, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn tilson_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    v_factor: f64,
    out: &mut [f64],
) {
    tilson_row_scalar(data, first, period, v_factor, out);
}

#[derive(Debug, Clone)]
pub struct TilsonStream {
    period: usize,
    v_factor: f64,
    e: [f64; 6],
    k: f64,
    one_minus_k: f64,
    head: usize,
    buffer: Vec<f64>,
    filled: bool,
    lookback_total: usize,
    seen: usize,
    history: Vec<f64>,
}

impl TilsonStream {
    pub fn try_new(params: TilsonParams) -> Result<Self, TilsonError> {
        let period = params.period.unwrap_or(5);
        let v_factor = params.volume_factor.unwrap_or(0.0);
        if period == 0 {
            return Err(TilsonError::InvalidPeriod { period, data_len: 0 });
        }
        if v_factor.is_nan() || v_factor.is_infinite() {
            return Err(TilsonError::InvalidVolumeFactor { v_factor });
        }
        let lookback_total = 6 * (period - 1);
        Ok(Self {
            period,
            v_factor,
            e: [0.0; 6],
            k: 2.0 / (period as f64 + 1.0),
            one_minus_k: 1.0 - 2.0 / (period as f64 + 1.0),
            head: 0,
            buffer: vec![f64::NAN; period],
            filled: false,
            lookback_total,
            seen: 0,
            history: Vec::new(),
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.buffer[self.head] = value;
        self.head = (self.head + 1) % self.period;
        self.history.push(value);
        self.seen += 1;
        if !self.filled && self.head == 0 {
            self.filled = true;
        }
        if !self.filled {
            return None;
        }
        if self.seen <= self.lookback_total {
            self.dot_ring();
            return None;
        }
        let params = TilsonParams {
            period: Some(self.period),
            volume_factor: Some(self.v_factor),
        };
        let input = TilsonInput::from_slice(&self.history, params);
        match tilson_with_kernel(&input, Kernel::Scalar) {
            Ok(out) => out.values.last().copied(),
            Err(_) => None,
        }
    }

    #[inline(always)]
    fn dot_ring(&mut self) -> f64 {
        let idx = if self.head == 0 { self.period - 1 } else { self.head - 1 };
        let mut val = self.buffer[idx];
        let mut e = &mut self.e;
        e[0] = self.k * val + self.one_minus_k * e[0];
        for i in 1..6 {
            e[i] = self.k * e[i - 1] + self.one_minus_k * e[i];
        }
        let temp = self.v_factor * self.v_factor;
        let c1 = -(temp * self.v_factor);
        let c2 = 3.0 * (temp - c1);
        let c3 = -6.0 * temp - 3.0 * (self.v_factor - c1);
        let c4 = 1.0 + 3.0 * self.v_factor - c1 + 3.0 * temp;
        c1 * e[5] + c2 * e[4] + c3 * e[3] + c4 * e[2]
    }
    
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_tilson_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = TilsonParams { period: None, volume_factor: None };
        let input = TilsonInput::from_candles(&candles, "close", default_params);
        let output = tilson_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_tilson_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = TilsonInput::from_candles(&candles, "close", TilsonParams::default());
        let result = tilson_with_kernel(&input, kernel)?;
        let expected_last_five = [
            59304.716332473254,
            59283.56868015526,
            59261.16173577631,
            59240.25895948583,
            59203.544843167765,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-8,
                "[{}] TILSON {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_tilson_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = TilsonInput::with_default_candles(&candles);
        match input.data {
            TilsonData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected TilsonData::Candles"),
        }
        let output = tilson_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_tilson_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = TilsonParams { period: Some(0), volume_factor: None };
        let input = TilsonInput::from_slice(&input_data, params);
        let res = tilson_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] TILSON should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_tilson_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data: [f64; 0] = [];
        let params = TilsonParams { period: Some(5), volume_factor: Some(0.0) };
        let input = TilsonInput::from_slice(&input_data, params);
        let res = tilson_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] TILSON should fail with empty input", test_name);
        if let Err(e) = res {
            assert!(matches!(e, TilsonError::EmptyInputData), "[{}] Expected EmptyInputData error", test_name);
        }
        Ok(())
    }

    fn check_tilson_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = TilsonParams { period: Some(10), volume_factor: None };
        let input = TilsonInput::from_slice(&data_small, params);
        let res = tilson_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] TILSON should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_tilson_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = TilsonParams { period: Some(9), volume_factor: None };
        let input = TilsonInput::from_slice(&single_point, params);
        let res = tilson_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] TILSON should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_tilson_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = TilsonParams { period: Some(5), volume_factor: None };
        let first_input = TilsonInput::from_candles(&candles, "close", first_params);
        let first_result = tilson_with_kernel(&first_input, kernel)?;

        let second_params = TilsonParams { period: Some(3), volume_factor: Some(0.7) };
        let second_input = TilsonInput::from_slice(&first_result.values, second_params);
        let second_result = tilson_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());
        for i in 240..second_result.values.len() {
            assert!(second_result.values[i].is_finite());
        }
        Ok(())
    }

    fn check_tilson_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = TilsonInput::from_candles(
            &candles,
            "close",
            TilsonParams { period: Some(5), volume_factor: Some(0.0) },
        );
        let res = tilson_with_kernel(&input, kernel)?;
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

    fn check_tilson_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let period = 5;
        let v_factor = 0.0;

        let input = TilsonInput::from_candles(
            &candles,
            "close",
            TilsonParams { period: Some(period), volume_factor: Some(v_factor) },
        );
        let batch_output = tilson_with_kernel(&input, kernel)?.values;

        let mut stream = TilsonStream::try_new(TilsonParams {
            period: Some(period),
            volume_factor: Some(v_factor),
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
                "[{}] TILSON streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_tilson_tests {
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
    fn check_tilson_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test with default parameters
        let input = TilsonInput::from_candles(&candles, "close", TilsonParams::default());
        let output = tilson_with_kernel(&input, kernel)?;

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
    fn check_tilson_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    generate_all_tilson_tests!(
        check_tilson_partial_params,
        check_tilson_accuracy,
        check_tilson_default_candles,
        check_tilson_zero_period,
        check_tilson_empty_input,
        check_tilson_period_exceeds_length,
        check_tilson_very_small_dataset,
        check_tilson_reinput,
        check_tilson_nan_handling,
        check_tilson_streaming,
        check_tilson_no_poison
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = TilsonBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = TilsonParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            59304.716332473254,
            59283.56868015526,
            59261.16173577631,
            59240.25895948583,
            59203.544843167765,
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

        // Test batch with multiple parameter combinations
        let output = TilsonBatchBuilder::new()
            .kernel(kernel)
            .period_range(10, 30, 10)  // Adjust ranges based on indicator
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

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);
}

// ========== Helper Functions for Bindings ==========

/// Centralized validation and preparation for Tilson calculation
#[inline]
fn tilson_prepare<'a>(
    input: &'a TilsonInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, f64, usize, usize, Kernel), TilsonError> {
    let data: &[f64] = input.as_ref();
    
    if data.is_empty() {
        return Err(TilsonError::EmptyInputData);
    }
    
    let first = data.iter().position(|x| !x.is_nan()).ok_or(TilsonError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();
    let v_factor = input.get_volume_factor();
    
    if period == 0 || period > len {
        return Err(TilsonError::InvalidPeriod { period, data_len: len });
    }
    if (len - first) < period {
        return Err(TilsonError::NotEnoughValidData { needed: period, valid: len - first });
    }
    if v_factor.is_nan() || v_factor.is_infinite() {
        return Err(TilsonError::InvalidVolumeFactor { v_factor });
    }
    
    let lookback_total = 6 * (period - 1);
    if (len - first) < lookback_total + 1 {
        return Err(TilsonError::NotEnoughValidData { needed: lookback_total + 1, valid: len - first });
    }
    
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    
    Ok((data, period, v_factor, first, len, chosen))
}

/// Compute Tilson directly into pre-allocated output buffer
#[inline]
fn tilson_compute_into(
    data: &[f64],
    period: usize,
    v_factor: f64,
    first: usize,
    chosen: Kernel,
    out: &mut [f64],
) -> Result<(), TilsonError> {
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                tilson_scalar(data, period, v_factor, first, out)?
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                tilson_avx2(data, period, v_factor, first, out)?
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                tilson_avx512(data, period, v_factor, first, out)?
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                // Fallback to scalar when AVX is not available
                tilson_scalar(data, period, v_factor, first, out)?
            }
            Kernel::Auto => unreachable!(),
        }
    }
    Ok(())
}

/// Optimized batch calculation that writes directly to pre-allocated buffer
#[inline(always)]
fn tilson_batch_inner_into(
    data: &[f64],
    sweep: &TilsonBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<TilsonParams>, TilsonError> {
    // ---------- 0. parameter checks ----------
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(TilsonError::InvalidPeriod { period: 0, data_len: 0 });
    }
    
    if data.is_empty() {
        return Err(TilsonError::EmptyInputData);
    }
    
    let first = data.iter().position(|x| !x.is_nan()).ok_or(TilsonError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    
    if data.len() - first < 6 * (max_p - 1) + 1 {
        return Err(TilsonError::NotEnoughValidData {
            needed: 6 * (max_p - 1) + 1,
            valid: data.len() - first,
        });
    }
    
    // ---------- 1. matrix dimensions ----------
    let rows = combos.len();
    let cols = data.len();
    
    // ---------- 2. build per-row warm-up lengths ----------
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| (6 * (c.period.unwrap() - 1) + first).min(cols))
        .collect();
    
    // ---------- 3. reinterpret output slice as MaybeUninit for efficient initialization ----------
    let out_uninit = unsafe {
        std::slice::from_raw_parts_mut(
            out.as_mut_ptr() as *mut MaybeUninit<f64>,
            out.len()
        )
    };
    
    unsafe { init_matrix_prefixes(out_uninit, cols, &warm) };
    
    // ---------- 4. closure that fills ONE row ----------
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();
        let v_factor = combos[row].volume_factor.unwrap();
        
        // cast this row to &mut [f64] so the row-kernel can write normally
        let out_row = core::slice::from_raw_parts_mut(
            dst_mu.as_mut_ptr() as *mut f64,
            dst_mu.len(),
        );
        
        match kern {
            Kernel::Scalar | Kernel::ScalarBatch => tilson_row_scalar(data, first, period, v_factor, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => tilson_row_avx2(data, first, period, v_factor, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => tilson_row_avx512(data, first, period, v_factor, out_row),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx512 | Kernel::Avx2Batch | Kernel::Avx512Batch => tilson_row_scalar(data, first, period, v_factor, out_row),
            _ => unreachable!(),
        }
    };
    
    // ---------- 5. run all rows (optionally in parallel) ----------
    if parallel {
        #[cfg(not(target_arch = "wasm32"))] {
            out_uninit.par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }
        #[cfg(target_arch = "wasm32")] {
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
#[pyfunction(name = "tilson")]
#[pyo3(signature = (data, period, volume_factor=None, kernel=None))]
/// Compute the Tilson T3 Moving Average of the input data.
///
/// The Tilson T3 is a moving average with reduced lag achieved through multiple
/// iterations of exponential smoothing.
///
/// Parameters:
/// -----------
/// data : np.ndarray
///     Input data array (float64).
/// period : int
///     Number of data points in the moving average window (must be >= 1).
/// volume_factor : float, optional
///     Controls the depth of T3 smoothing. Range [0.0, 1.0].
///     Default is 0.0. Higher values = more smoothing.
/// kernel : str, optional
///     Computation kernel to use: 'auto', 'scalar', 'avx2', 'avx512'.
///     Default is 'auto' which auto-detects the best available.
///
/// Returns:
/// --------
/// np.ndarray
///     Array of Tilson T3 values, same length as input.
///
/// Raises:
/// -------
/// ValueError
///     If inputs are invalid (period is zero, exceeds data length, etc).
pub fn tilson_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
    volume_factor: Option<f64>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{PyArray1, PyArrayMethods};
    
    let slice_in = data.as_slice()?; // zero-copy, read-only view
    
    // Parse kernel string to enum
    let kern = match kernel {
        None | Some("auto") => Kernel::Auto,
        Some("scalar") => Kernel::Scalar,
        Some("avx2") => Kernel::Avx2,
        Some("avx512") => Kernel::Avx512,
        Some(k) => return Err(PyValueError::new_err(format!("Unknown kernel: {}", k))),
    };
    
    // ---------- build input struct -------------------------------------------------
    let params = TilsonParams { 
        period: Some(period),
        volume_factor: volume_factor.or(Some(0.0))
    };
    let tilson_in = TilsonInput::from_slice(slice_in, params);
    
    // ---------- allocate NumPy output buffer ---------------------------------------
    let out_arr = unsafe { PyArray1::<f64>::new(py, [slice_in.len()], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? }; // safe: we own the array
    
    // ---------- heavy lifting without the GIL --------------------------------------
    py.allow_threads(|| -> Result<(), TilsonError> {
        let (data, period, v_factor, first, _len, chosen) = tilson_prepare(&tilson_in, kern)?;
        
        // Initialize NaN prefix
        let lookback = 6 * (period - 1);
        let warm = (first + lookback).min(slice_out.len());
        slice_out[..warm].fill(f64::NAN);
        
        // Compute Tilson
        tilson_compute_into(data, period, v_factor, first, chosen, slice_out)?;
        Ok(())
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?; // unify error type
    
    Ok(out_arr.into())
}

#[cfg(feature = "python")]
#[pyclass(name = "TilsonStream")]
pub struct TilsonStreamPy {
    stream: TilsonStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl TilsonStreamPy {
    #[new]
    fn new(period: usize, volume_factor: Option<f64>) -> PyResult<Self> {
        let params = TilsonParams { 
            period: Some(period),
            volume_factor: volume_factor.or(Some(0.0))
        };
        let stream = TilsonStream::try_new(params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(TilsonStreamPy { stream })
    }
    
    /// Updates the stream with a new value and returns the calculated Tilson value.
    /// Returns `None` if the buffer is not yet full.
    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "tilson_batch")]
#[pyo3(signature = (data, period_range, volume_factor_range=None, kernel=None))]
/// Compute Tilson T3 for multiple parameter combinations in a single pass.
///
/// Parameters:
/// -----------
/// data : np.ndarray
///     Input data array (float64).
/// period_range : tuple
///     (start, end, step) for period values to compute.
/// volume_factor_range : tuple, optional
///     (start, end, step) for volume_factor values. Default is (0.0, 0.0, 0.0).
/// kernel : str, optional
///     Computation kernel: 'auto', 'scalar', 'avx2', 'avx512'.
///     Default is 'auto' which auto-detects the best available.
///
/// Returns:
/// --------
/// dict
///     Dictionary with 'values' (2D array), 'periods', and 'volume_factors' arrays.
pub fn tilson_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    volume_factor_range: Option<(f64, f64, f64)>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;
    
    let slice_in = data.as_slice()?;
    
    let sweep = TilsonBatchRange {
        period: period_range,
        volume_factor: volume_factor_range.unwrap_or((0.0, 0.0, 0.0)),
    };
    
    // 1. Expand grid once to know rows*cols
    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();
    
    // 2. Pre-allocate NumPy array (1-D, will reshape later)
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
    
    // 3. Heavy work without the GIL
    let combos = py.allow_threads(|| {
        // Resolve Kernel::Auto to a specific kernel
        let kernel = match kern {
            Kernel::Auto => detect_best_batch_kernel(),
            k => k,
        };
        
        // Determine if we should use parallel processing
        let parallel = !cfg!(target_arch = "wasm32") && rows > 1 && cols > 1000;
        
        tilson_batch_inner_into(slice_in, &sweep, kernel, parallel, slice_out)?;
        Ok::<Vec<TilsonParams>, TilsonError>(combos)
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    
    // 4. Reshape the flat output array to 2D
    let reshaped = out_arr.reshape([rows, cols])?;
    
    // 5. Extract periods and volume_factors as separate arrays
    let periods: Vec<usize> = combos.iter().map(|c| c.period.unwrap()).collect();
    let v_factors: Vec<f64> = combos.iter().map(|c| c.volume_factor.unwrap()).collect();
    
    // 6. Create output dictionary
    let dict = PyDict::new(py);
    dict.set_item("values", reshaped)?;
    dict.set_item("periods", periods.into_pyarray(py))?;
    dict.set_item("volume_factors", v_factors.into_pyarray(py))?;
    
    Ok(dict.into())
}

// ========== WASM Bindings ==========

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = tilson_js)]
/// Compute the Tilson T3 Moving Average.
///
/// # Arguments
/// * `data` - Input data array
/// * `period` - Period (must be >= 1)
/// * `volume_factor` - Volume factor (0.0 to 1.0), defaults to 0.0
///
/// # Returns
/// Array of Tilson values, same length as input
pub fn tilson_js(data: &[f64], period: usize, volume_factor: Option<f64>) -> Result<Vec<f64>, JsValue> {
    let params = TilsonParams { 
        period: Some(period),
        volume_factor: volume_factor.or(Some(0.0))
    };
    let input = TilsonInput::from_slice(data, params);
    
    tilson(&input)
        .map(|output| output.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = tilson_batch_js)]
/// Compute Tilson for multiple parameter combinations in a single pass.
///
/// # Arguments
/// * `data` - Input data array
/// * `period_start`, `period_end`, `period_step` - Period range parameters
/// * `v_factor_start`, `v_factor_end`, `v_factor_step` - Volume factor range parameters
///
/// # Returns
/// Flattened array of values (row-major order)
pub fn tilson_batch_js(
    data: &[f64],
    period_start: usize,
    period_end: usize,
    period_step: usize,
    v_factor_start: f64,
    v_factor_end: f64,
    v_factor_step: f64,
) -> Result<Vec<f64>, JsValue> {
    let sweep = TilsonBatchRange {
        period: (period_start, period_end, period_step),
        volume_factor: (v_factor_start, v_factor_end, v_factor_step),
    };
    
    let output = tilson_batch_with_kernel(data, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    
    Ok(output.values)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = tilson_batch_metadata_js)]
/// Get metadata about batch computation.
///
/// # Arguments
/// * Period and volume factor range parameters (same as tilson_batch_js)
///
/// # Returns
/// Array containing [periods array, volume_factors array] flattened
pub fn tilson_batch_metadata_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
    v_factor_start: f64,
    v_factor_end: f64,
    v_factor_step: f64,
) -> Vec<f64> {
    let sweep = TilsonBatchRange {
        period: (period_start, period_end, period_step),
        volume_factor: (v_factor_start, v_factor_end, v_factor_step),
    };
    
    let combos = expand_grid(&sweep);
    let mut result = Vec::with_capacity(combos.len() * 2);
    
    // First, all periods
    for combo in &combos {
        result.push(combo.period.unwrap() as f64);
    }
    
    // Then, all volume factors
    for combo in &combos {
        result.push(combo.volume_factor.unwrap());
    }
    
    result
}
