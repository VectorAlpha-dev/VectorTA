//! # 3-Pole SuperSmoother Filter
//!
//! Three-pole smoothing filter (John Ehlers). Strong noise suppression, responsive to trend, configurable by period.
//!
//! ## Parameters
//! - **period**: Smoothing period, >= 1.
//!
//! ## Errors
//! - **AllValuesNaN**: supersmoother_3_pole: All input values are NaN.
//! - **InvalidPeriod**: supersmoother_3_pole: `period` is zero or < 1 or exceeds data length.
//! - **NotEnoughValidData**: supersmoother_3_pole: Not enough valid data for the requested period.
//!
//! ## Returns
//! - **Ok(SuperSmoother3PoleOutput)** on success, values Vec<f64> matching input length.
//! - **Err(SuperSmoother3PoleError)** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel, alloc_with_nan_prefix, make_uninit_matrix, init_matrix_prefixes};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;
use std::f64::consts::PI;
use std::mem::MaybeUninit;

// Input and Output Types

#[derive(Debug, Clone)]
pub enum SuperSmoother3PoleData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

impl<'a> AsRef<[f64]> for SuperSmoother3PoleInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            SuperSmoother3PoleData::Slice(slice) => slice,
            SuperSmoother3PoleData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SuperSmoother3PoleOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct SuperSmoother3PoleParams {
    pub period: Option<usize>,
}

impl Default for SuperSmoother3PoleParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct SuperSmoother3PoleInput<'a> {
    pub data: SuperSmoother3PoleData<'a>,
    pub params: SuperSmoother3PoleParams,
}

impl<'a> SuperSmoother3PoleInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: SuperSmoother3PoleParams) -> Self {
        Self {
            data: SuperSmoother3PoleData::Candles { candles: c, source: s },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: SuperSmoother3PoleParams) -> Self {
        Self {
            data: SuperSmoother3PoleData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", SuperSmoother3PoleParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
}

// Builder Pattern

#[derive(Copy, Clone, Debug)]
pub struct SuperSmoother3PoleBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for SuperSmoother3PoleBuilder {
    fn default() -> Self {
        Self { period: None, kernel: Kernel::Auto }
    }
}

impl SuperSmoother3PoleBuilder {
    #[inline(always)]
    pub fn new() -> Self { Self::default() }
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
    pub fn apply(self, c: &Candles) -> Result<SuperSmoother3PoleOutput, SuperSmoother3PoleError> {
        let p = SuperSmoother3PoleParams { period: self.period };
        let i = SuperSmoother3PoleInput::from_candles(c, "close", p);
        supersmoother_3_pole_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<SuperSmoother3PoleOutput, SuperSmoother3PoleError> {
        let p = SuperSmoother3PoleParams { period: self.period };
        let i = SuperSmoother3PoleInput::from_slice(d, p);
        supersmoother_3_pole_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<SuperSmoother3PoleStream, SuperSmoother3PoleError> {
        let p = SuperSmoother3PoleParams { period: self.period };
        SuperSmoother3PoleStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum SuperSmoother3PoleError {
    #[error("supersmoother_3_pole: All values are NaN.")]
    AllValuesNaN,
    #[error("supersmoother_3_pole: Invalid period: period = {period}")]
    InvalidPeriod { period: usize },
    #[error("supersmoother_3_pole: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

// Main Entrypoint

#[inline]
pub fn supersmoother_3_pole(input: &SuperSmoother3PoleInput) -> Result<SuperSmoother3PoleOutput, SuperSmoother3PoleError> {
    supersmoother_3_pole_with_kernel(input, Kernel::Auto)
}

pub fn supersmoother_3_pole_with_kernel(
    input: &SuperSmoother3PoleInput,
    kernel: Kernel,
) -> Result<SuperSmoother3PoleOutput, SuperSmoother3PoleError> {
    let data: &[f64] = input.as_ref();
    let first = data.iter().position(|x| !x.is_nan()).ok_or(SuperSmoother3PoleError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(SuperSmoother3PoleError::InvalidPeriod { period });
    }
    if (len - first) < period {
        return Err(SuperSmoother3PoleError::NotEnoughValidData { needed: period, valid: len - first });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    let warm   = first + period;
    let mut out = alloc_with_nan_prefix(len, warm);

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                supersmoother_3_pole_scalar(data, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                supersmoother_3_pole_avx2(data, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                supersmoother_3_pole_avx512(data, period, first, &mut out)
            }
            _ => unreachable!(),
        }
    }

    Ok(SuperSmoother3PoleOutput { values: out })
}

// Scalar reference implementation (original logic)
#[inline(always)]
pub unsafe fn supersmoother_3_pole_scalar(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    let n = data.len();
    if n == 0 { return; }

    let a = (-PI / period as f64).exp();
    let b = 2.0 * a * (1.738_f64 * PI / period as f64).cos();
    let c = a * a;

    let coef_source = 1.0 - c * c - b + b * c;
    let coef_prev1 = b + c;
    let coef_prev2 = -c - b * c;
    let coef_prev3 = c * c;

    if n > 0 {
        out[0] = data[0];
    }
    if n > 1 {
        out[1] = data[1];
    }
    if n > 2 {
        out[2] = data[2];
    }
    for i in 3..n {
        let d_i = data[i];
        let o_im1 = out[i - 1];
        let o_im2 = out[i - 2];
        let o_im3 = out[i - 3];
        out[i] = coef_source * d_i + coef_prev1 * o_im1 + coef_prev2 * o_im2 + coef_prev3 * o_im3;
    }
}

// AVX2/AVX512 stubs (point to scalar)
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn supersmoother_3_pole_avx2(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    supersmoother_3_pole_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn supersmoother_3_pole_avx512(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        unsafe { supersmoother_3_pole_avx512_short(data, period, first, out) }
    } else {
        unsafe { supersmoother_3_pole_avx512_long(data, period, first, out) }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn supersmoother_3_pole_avx512_short(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    supersmoother_3_pole_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn supersmoother_3_pole_avx512_long(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    supersmoother_3_pole_scalar(data, period, first, out)
}

// Row functions (API parity)
#[inline(always)]
pub unsafe fn supersmoother_3_pole_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    supersmoother_3_pole_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn supersmoother_3_pole_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    w_ptr: *const f64,
    inv_n: f64,
    out: &mut [f64],
) {
    supersmoother_3_pole_row_scalar(data, first, period, stride, w_ptr, inv_n, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn supersmoother_3_pole_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    w_ptr: *const f64,
    inv_n: f64,
    out: &mut [f64],
) {
    supersmoother_3_pole_row_scalar(data, first, period, stride, w_ptr, inv_n, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn supersmoother_3_pole_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    w_ptr: *const f64,
    inv_n: f64,
    out: &mut [f64],
) {
    supersmoother_3_pole_row_scalar(data, first, period, stride, w_ptr, inv_n, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn supersmoother_3_pole_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    w_ptr: *const f64,
    inv_n: f64,
    out: &mut [f64],
) {
    supersmoother_3_pole_row_scalar(data, first, period, stride, w_ptr, inv_n, out)
}

// Stream/Stateful

#[derive(Debug, Clone)]
pub struct SuperSmoother3PoleStream {
    period: usize,
    buffer: Vec<f64>,
    idx: usize,
    filled: usize,
    a: f64,
    b: f64,
    c: f64,
    coef_source: f64,
    coef_prev1: f64,
    coef_prev2: f64,
    coef_prev3: f64,
}

impl SuperSmoother3PoleStream {
    pub fn try_new(params: SuperSmoother3PoleParams) -> Result<Self, SuperSmoother3PoleError> {
        let period = params.period.unwrap_or(14);
        if period == 0 {
            return Err(SuperSmoother3PoleError::InvalidPeriod { period });
        }
        let a = (-PI / period as f64).exp();
        let b = 2.0 * a * (1.738_f64 * PI / period as f64).cos();
        let c = a * a;
        let coef_source = 1.0 - c * c - b + b * c;
        let coef_prev1 = b + c;
        let coef_prev2 = -c - b * c;
        let coef_prev3 = c * c;
        Ok(Self {
            period,
            buffer: vec![f64::NAN; 3],
            idx: 0,
            filled: 0,
            a,
            b,
            c,
            coef_source,
            coef_prev1,
            coef_prev2,
            coef_prev3,
        })
    }
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> f64 {
        if self.filled < 3 {
            self.buffer[self.filled] = value;
            self.filled += 1;
            return value;
        }
        let next = self.coef_source * value
            + self.coef_prev1 * self.buffer[(self.idx + 2) % 3]
            + self.coef_prev2 * self.buffer[(self.idx + 1) % 3]
            + self.coef_prev3 * self.buffer[self.idx % 3];
        self.buffer[self.idx] = next;
        self.idx = (self.idx + 1) % 3;
        next
    }
}

// Batch Range/Builder/Output

#[derive(Clone, Debug)]
pub struct SuperSmoother3PoleBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for SuperSmoother3PoleBatchRange {
    fn default() -> Self {
        Self { period: (14, 14, 0) }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SuperSmoother3PoleBatchBuilder {
    range: SuperSmoother3PoleBatchRange,
    kernel: Kernel,
}

impl SuperSmoother3PoleBatchBuilder {
    pub fn new() -> Self { Self::default() }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k; self
    }
    #[inline]
    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.period = (start, end, step); self
    }
    #[inline]
    pub fn period_static(mut self, p: usize) -> Self {
        self.range.period = (p, p, 0); self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<SuperSmoother3PoleBatchOutput, SuperSmoother3PoleError> {
        supersmoother_3_pole_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<SuperSmoother3PoleBatchOutput, SuperSmoother3PoleError> {
        SuperSmoother3PoleBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<SuperSmoother3PoleBatchOutput, SuperSmoother3PoleError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<SuperSmoother3PoleBatchOutput, SuperSmoother3PoleError> {
        SuperSmoother3PoleBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
    }
}

pub fn supersmoother_3_pole_batch_with_kernel(
    data: &[f64],
    sweep: &SuperSmoother3PoleBatchRange,
    k: Kernel,
) -> Result<SuperSmoother3PoleBatchOutput, SuperSmoother3PoleError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(SuperSmoother3PoleError::InvalidPeriod { period: 0 }),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    supersmoother_3_pole_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct SuperSmoother3PoleBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<SuperSmoother3PoleParams>,
    pub rows: usize,
    pub cols: usize,
}
impl SuperSmoother3PoleBatchOutput {
    pub fn row_for_params(&self, p: &SuperSmoother3PoleParams) -> Option<usize> {
        self.combos.iter().position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
    }
    pub fn values_for(&self, p: &SuperSmoother3PoleParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &SuperSmoother3PoleBatchRange) -> Vec<SuperSmoother3PoleParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(SuperSmoother3PoleParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn supersmoother_3_pole_batch_slice(
    data: &[f64],
    sweep: &SuperSmoother3PoleBatchRange,
    kern: Kernel,
) -> Result<SuperSmoother3PoleBatchOutput, SuperSmoother3PoleError> {
    supersmoother_3_pole_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn supersmoother_3_pole_batch_par_slice(
    data: &[f64],
    sweep: &SuperSmoother3PoleBatchRange,
    kern: Kernel,
) -> Result<SuperSmoother3PoleBatchOutput, SuperSmoother3PoleError> {
    supersmoother_3_pole_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn supersmoother_3_pole_batch_inner(
    data: &[f64],
    sweep: &SuperSmoother3PoleBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<SuperSmoother3PoleBatchOutput, SuperSmoother3PoleError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(SuperSmoother3PoleError::InvalidPeriod { period: 0 });
    }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(SuperSmoother3PoleError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(SuperSmoother3PoleError::NotEnoughValidData { needed: max_p, valid: data.len() - first });
    }
    let rows = combos.len();
    let cols = data.len();
    let warm: Vec<usize> = combos.iter()
                                .map(|c| first + c.period.unwrap())
                                .collect();

    let mut raw = make_uninit_matrix(rows, cols);
    unsafe { init_matrix_prefixes(&mut raw, cols, &warm) };

    // ---------- 2.  Row-level closure (accepts &mut [MaybeUninit<f64>]) ----
    let do_row = |row: usize, dst_mu: &mut [std::mem::MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();

        // transmute just this row to &mut [f64]
        let out_row = core::slice::from_raw_parts_mut(
            dst_mu.as_mut_ptr() as *mut f64,
            dst_mu.len(),
        );

        match kern {
            Kernel::Scalar => supersmoother_3_pole_row_scalar(
                data, first, period, 0, std::ptr::null(), 0.0, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2   => supersmoother_3_pole_row_avx2(
                data, first, period, 0, std::ptr::null(), 0.0, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => supersmoother_3_pole_row_avx512(
                data, first, period, 0, std::ptr::null(), 0.0, out_row),
            _ => unreachable!(),
        }
    };

    // ---------- 3.  Run rows in serial / parallel --------------------------
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

    // ---------- 4.  Finalise: convert Vec<MaybeUninit<f64>> → Vec<f64> -----
    let values: Vec<f64> = unsafe { std::mem::transmute(raw) };
    Ok(SuperSmoother3PoleBatchOutput { values, combos, rows, cols })
}

#[inline(always)]
fn expand_grid_supersmoother(r: &SuperSmoother3PoleBatchRange) -> Vec<SuperSmoother3PoleParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    periods.into_iter().map(|p| SuperSmoother3PoleParams { period: Some(p) }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_supersmoother_3_pole_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = SuperSmoother3PoleParams { period: None };
        let input = SuperSmoother3PoleInput::from_candles(&candles, "close", default_params);
        let output = supersmoother_3_pole_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_supersmoother_3_pole_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = SuperSmoother3PoleParams { period: Some(14) };
        let input = SuperSmoother3PoleInput::from_candles(&candles, "close", params);
        let result = supersmoother_3_pole_with_kernel(&input, kernel)?;
        let values = &result.values;
        let expected_last_five = [
            59072.13481064446,
            59089.08032603,
            59111.35711851466,
            59133.14402399381,
            59121.91820047289,
        ];
        assert!(values.len() >= 5);
        let start_idx = values.len() - 5;
        let last_five = &values[start_idx..];
        for (i, (&actual, &expected)) in last_five.iter().zip(expected_last_five.iter()).enumerate() {
            let diff = (actual - expected).abs();
            assert!(diff < 1e-8, "3-pole SuperSmoother mismatch at index {}: expected {}, got {}, diff {}", i, expected, actual, diff);
        }
        Ok(())
    }

    fn check_supersmoother_3_pole_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [10.0, 20.0, 30.0];
        let params = SuperSmoother3PoleParams { period: Some(0) };
        let input = SuperSmoother3PoleInput::from_slice(&data, params);
        let res = supersmoother_3_pole_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] SS3Pole should fail with zero period", test_name);
        Ok(())
    }

    fn check_supersmoother_3_pole_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [10.0, 20.0, 30.0];
        let params = SuperSmoother3PoleParams { period: Some(10) };
        let input = SuperSmoother3PoleInput::from_slice(&data, params);
        let res = supersmoother_3_pole_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] SS3Pole should fail with period exceeding length", test_name);
        Ok(())
    }

    fn check_supersmoother_3_pole_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [10.0, 20.0];
        let params = SuperSmoother3PoleParams { period: Some(14) };
        let input = SuperSmoother3PoleInput::from_slice(&data, params);
        let result = supersmoother_3_pole_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), data.len());
        assert_eq!(result.values[0], 10.0);
        if result.values.len() > 1 {
            assert_eq!(result.values[1], 20.0);
        }
        Ok(())
    }

    fn check_supersmoother_3_pole_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_input = SuperSmoother3PoleInput::from_candles(&candles, "close", SuperSmoother3PoleParams { period: Some(14) });
        let first_result = supersmoother_3_pole_with_kernel(&first_input, kernel)?;
        let second_input = SuperSmoother3PoleInput::from_slice(&first_result.values, SuperSmoother3PoleParams { period: Some(7) });
        let second_result = supersmoother_3_pole_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    fn check_supersmoother_3_pole_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = SuperSmoother3PoleInput::from_candles(&candles, "close", SuperSmoother3PoleParams { period: Some(14) });
        let result = supersmoother_3_pole_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        for (idx, &val) in result.values.iter().enumerate() {
            assert!(val.is_finite(), "NaN found at index {}", idx);
        }
        Ok(())
    }

    fn check_supersmoother_3_pole_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 14;
        let input = SuperSmoother3PoleInput::from_candles(&candles, "close", SuperSmoother3PoleParams { period: Some(period) });
        let batch_output = supersmoother_3_pole_with_kernel(&input, kernel)?.values;
        let mut stream = SuperSmoother3PoleStream::try_new(SuperSmoother3PoleParams { period: Some(period) })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            let ss_val = stream.update(price);
            stream_values.push(ss_val);
        }
        assert_eq!(batch_output.len(), stream_values.len());
        for (i, (&b, &s)) in batch_output.iter().zip(stream_values.iter()).enumerate() {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(diff < 1e-9, "[{}] SS3Pole streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}", test_name, i, b, s, diff);
        }
        Ok(())
    }

    macro_rules! generate_all_ss3pole_tests {
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
    fn check_supersmoother_3_pole_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test with default parameters
        let input = SuperSmoother3PoleInput::from_candles(&candles, "close", SuperSmoother3PoleParams::default());
        let output = supersmoother_3_pole_with_kernel(&input, kernel)?;

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
    fn check_supersmoother_3_pole_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    generate_all_ss3pole_tests!(
        check_supersmoother_3_pole_partial_params,
        check_supersmoother_3_pole_accuracy,
        check_supersmoother_3_pole_zero_period,
        check_supersmoother_3_pole_period_exceeds_length,
        check_supersmoother_3_pole_very_small_dataset,
        check_supersmoother_3_pole_reinput,
        check_supersmoother_3_pole_nan_handling,
        check_supersmoother_3_pole_streaming,
        check_supersmoother_3_pole_no_poison
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = SuperSmoother3PoleBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = SuperSmoother3PoleParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [
            59072.13481064446,
            59089.08032603,
            59111.35711851466,
            59133.14402399381,
            59121.91820047289,
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
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]),
                                     Kernel::Auto);
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
        let output = SuperSmoother3PoleBatchBuilder::new()
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
    fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);
}

// Python bindings
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use numpy;

#[cfg(feature = "python")]
#[pyfunction(name = "supersmoother_3_pole")]
#[pyo3(signature = (data, period, kernel=None))]
/// Compute the 3-Pole SuperSmoother filter of the input data.
///
/// The 3-Pole SuperSmoother is a smoothing filter developed by John Ehlers
/// that provides strong noise suppression while remaining responsive to trend changes.
///
/// Parameters:
/// -----------
/// data : np.ndarray
///     Input data array (float64).
/// period : int
///     The smoothing period (must be >= 1).
/// kernel : str, optional
///     Computation kernel to use: 'auto', 'scalar', 'avx2', 'avx512'.
///     Default is 'auto' which auto-detects the best available.
///
/// Returns:
/// --------
/// np.ndarray
///     Array of SuperSmoother3Pole values, same length as input.
///
/// Raises:
/// -------
/// ValueError
///     If inputs are invalid (period <= 0, period > data length, all NaN data).
pub fn supersmoother_3_pole_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
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

    // Build input struct
    let params = SuperSmoother3PoleParams { period: Some(period) };
    let ss3p_in = SuperSmoother3PoleInput::from_slice(slice_in, params);

    // Allocate NumPy output buffer
    let out_arr = unsafe { PyArray1::<f64>::new(py, [slice_in.len()], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? }; // safe: we own the array

    // Heavy lifting without the GIL
    py.allow_threads(|| -> Result<(), SuperSmoother3PoleError> {
        let result = supersmoother_3_pole_with_kernel(&ss3p_in, kern)?;
        slice_out.copy_from_slice(&result.values);
        Ok(())
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(out_arr.into())
}

#[cfg(feature = "python")]
#[pyclass(name = "SuperSmoother3PoleStream")]
pub struct SuperSmoother3PoleStreamPy {
    stream: SuperSmoother3PoleStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl SuperSmoother3PoleStreamPy {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        let params = SuperSmoother3PoleParams { period: Some(period) };
        let stream = SuperSmoother3PoleStream::try_new(params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(SuperSmoother3PoleStreamPy { stream })
    }

    /// Updates the stream with a new value and returns the calculated SuperSmoother3Pole value.
    fn update(&mut self, value: f64) -> f64 {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "supersmoother_3_pole_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
/// Compute SuperSmoother3Pole for multiple period combinations in a single pass.
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
///     Dictionary with 'values' (2D array) and 'periods' array.
pub fn supersmoother_3_pole_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;

    let sweep = SuperSmoother3PoleBatchRange {
        period: period_range,
    };

    // Parse kernel string to enum
    let kern = match kernel {
        None | Some("auto") => Kernel::Auto,
        Some("scalar") => Kernel::ScalarBatch,
        Some("avx2") => Kernel::Avx2Batch,
        Some("avx512") => Kernel::Avx512Batch,
        Some(k) => return Err(PyValueError::new_err(format!("Unknown kernel: {}", k))),
    };

    // Run batch computation without GIL
    let output = py
        .allow_threads(|| supersmoother_3_pole_batch_with_kernel(slice_in, &sweep, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Create return dictionary
    let dict = PyDict::new(py);

    // Values as 2D array
    let values_array = output
        .values
        .into_pyarray(py)
        .reshape([output.rows, output.cols])?;
    dict.set_item("values", values_array)?;

    // Periods array
    let periods: Vec<f64> = output
        .combos
        .iter()
        .map(|c| c.period.unwrap() as f64)
        .collect();
    dict.set_item("periods", periods.into_pyarray(py))?;

    Ok(dict.into())
}

// WASM bindings
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn supersmoother_3_pole_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = SuperSmoother3PoleParams { period: Some(period) };
    let input = SuperSmoother3PoleInput::from_slice(data, params);

    supersmoother_3_pole_with_kernel(&input, Kernel::Scalar)
        .map(|o| o.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn supersmoother_3_pole_batch_js(
    data: &[f64],
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = SuperSmoother3PoleBatchRange {
        period: (period_start, period_end, period_step),
    };

    // Use the existing batch function with parallel=false for WASM
    supersmoother_3_pole_batch_inner(data, &sweep, Kernel::Scalar, false)
        .map(|output| output.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn supersmoother_3_pole_batch_metadata_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = SuperSmoother3PoleBatchRange {
        period: (period_start, period_end, period_step),
    };

    let combos = expand_grid(&sweep);
    let mut metadata = Vec::with_capacity(combos.len());

    for combo in combos {
        metadata.push(combo.period.unwrap() as f64);
    }

    Ok(metadata)
}
