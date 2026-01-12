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
use crate::utilities::dlpack_cuda::{make_device_array_py, DeviceArrayF32Py};
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
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use serde::{Deserialize, Serialize};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

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

#[derive(Debug, Clone, Copy)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct TemaParams {
    pub period: Option<usize>,
}

impl Default for TemaParams {
    fn default() -> Self {
        Self { period: Some(9) }
    }
}

#[derive(Debug, Clone)]
pub struct TemaOutput {
    pub values: Vec<f64>,
}

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
    #[error("tema: Output length mismatch: expected {expected}, got {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("tema: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("tema: Invalid kernel for batch path: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("tema: Invalid input: {0}")]
    InvalidInput(String),
}

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

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn tema_into(input: &TemaInput, out: &mut [f64]) -> Result<(), TemaError> {
    let (data, period, first, len, chosen) = tema_prepare(input, Kernel::Auto)?;

    if out.len() != len {
        return Err(TemaError::OutputLengthMismatch {
            expected: len,
            got: out.len(),
        });
    }

    let warm = first + (period - 1) * 3;
    let qnan = f64::from_bits(0x7ff8_0000_0000_0000);
    let pref = warm.min(out.len());
    for v in &mut out[..pref] {
        *v = qnan;
    }

    tema_compute_into(data, period, first, chosen, out);

    Ok(())
}

#[inline]
pub fn tema_scalar(data: &[f64], period: usize, first_val: usize, out: &mut [f64]) {
    debug_assert_eq!(data.len(), out.len());

    let n = data.len();
    if n == 0 {
        return;
    }

    let per = 2.0 / (period as f64 + 1.0);
    let per1 = 1.0 - per;

    let p1 = period - 1;
    let start2 = first_val + p1;
    let start3 = first_val + (p1 << 1);
    let start_out = first_val + p1 * 3;

    if period == 1 {
        out[first_val..n].copy_from_slice(&data[first_val..n]);
        return;
    }

    let mut ema1 = data[first_val];
    let mut ema2 = 0.0f64;
    let mut ema3 = 0.0f64;

    let end0 = start2.min(n);
    for i in first_val..end0 {
        let price = data[i];
        ema1 = ema1 * per1 + price * per;
    }

    if start2 < n {
        let price = data[start2];
        ema1 = ema1 * per1 + price * per;
        ema2 = ema1;
        ema2 = ema2 * per1 + ema1 * per;

        let end1 = start3.min(n);
        for i in (start2 + 1)..end1 {
            let price = data[i];
            ema1 = ema1 * per1 + price * per;
            ema2 = ema2 * per1 + ema1 * per;
        }

        if start3 < n {
            let price = data[start3];
            ema1 = ema1 * per1 + price * per;
            ema2 = ema2 * per1 + ema1 * per;
            ema3 = ema2;
            ema3 = ema3 * per1 + ema2 * per;

            let end2 = start_out.min(n);
            for i in (start3 + 1)..end2 {
                let price = data[i];
                ema1 = ema1 * per1 + price * per;
                ema2 = ema2 * per1 + ema1 * per;
                ema3 = ema3 * per1 + ema2 * per;
            }

            for i in start_out..n {
                let price = data[i];
                ema1 = ema1 * per1 + price * per;
                ema2 = ema2 * per1 + ema1 * per;
                ema3 = ema3 * per1 + ema2 * per;

                out[i] = 3.0 * ema1 - 3.0 * ema2 + ema3;
            }
        }
    }
}

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

#[derive(Debug, Clone)]
pub struct TemaStream {
    period: usize,

    alpha: f64,
    one_minus_alpha: f64,

    ema1: f64,
    ema2: f64,
    ema3: f64,

    count: usize,

    start2: usize,
    start3: usize,
    lookback: usize,

    ready: bool,
}

impl TemaStream {
    #[inline]
    pub fn try_new(params: TemaParams) -> Result<Self, TemaError> {
        let period = params.period.unwrap_or(9);
        if period == 0 {
            return Err(TemaError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }

        let alpha = 2.0 / (period as f64 + 1.0);
        let one_minus_alpha = 1.0 - alpha;

        let start2 = period;

        let start3 = period + period - 1;

        let lookback = (period - 1) * 3;

        Ok(Self {
            period,
            alpha,
            one_minus_alpha,
            ema1: f64::NAN,
            ema2: 0.0,
            ema3: 0.0,
            count: 0,
            start2,
            start3,
            lookback,
            ready: false,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, x: f64) -> Option<f64> {
        if self.ready {
            self.ema1 = self.ema1 * self.one_minus_alpha + x * self.alpha;
            self.ema2 = self.ema2 * self.one_minus_alpha + self.ema1 * self.alpha;
            self.ema3 = self.ema3 * self.one_minus_alpha + self.ema2 * self.alpha;
            return Some(3.0 * self.ema1 - 3.0 * self.ema2 + self.ema3);
        }

        if self.count == 0 {
            if x.is_nan() {
                return None;
            }
            self.ema1 = x;
            self.count = 1;

            if self.period == 1 {
                self.ema2 = self.ema1;
                self.ema3 = self.ema2;
                self.ready = true;
                return Some(self.ema1);
            }
            return None;
        }

        self.ema1 = self.ema1 * self.one_minus_alpha + x * self.alpha;
        self.count += 1;

        if self.count == self.start2 {
            self.ema2 = self.ema1;
        } else if self.count > self.start2 {
            self.ema2 = self.ema2 * self.one_minus_alpha + self.ema1 * self.alpha;
        }

        if self.count == self.start3 {
            self.ema3 = self.ema2;
        } else if self.count > self.start3 {
            self.ema3 = self.ema3 * self.one_minus_alpha + self.ema2 * self.alpha;
        }

        if self.count > self.lookback {
            self.ready = true;
            Some(3.0 * self.ema1 - 3.0 * self.ema2 + self.ema3)
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn update_unchecked(&mut self, x: f64) -> f64 {
        debug_assert!(self.ready, "call update() until Some(..) first");
        self.ema1 = self.ema1 * self.one_minus_alpha + x * self.alpha;
        self.ema2 = self.ema2 * self.one_minus_alpha + self.ema1 * self.alpha;
        self.ema3 = self.ema3 * self.one_minus_alpha + self.ema2 * self.alpha;
        3.0 * self.ema1 - 3.0 * self.ema2 + self.ema3
    }
}

#[derive(Clone, Debug)]
pub struct TemaBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for TemaBatchRange {
    fn default() -> Self {
        Self {
            period: (9, 258, 1),
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
        other => return Err(TemaError::InvalidKernelForBatch(other)),
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
fn expand_grid(r: &TemaBatchRange) -> Result<Vec<TemaParams>, TemaError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, TemaError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }

        if start > end {
            let mut v = Vec::new();
            let mut cur = start;
            while cur >= end {
                v.push(cur);

                if let Some(next) = cur.checked_sub(step) {
                    if next == cur {
                        break;
                    }
                    cur = next;
                } else {
                    break;
                }
                if cur == usize::MAX {
                    break;
                }
                if cur < end {
                    break;
                }
            }
            if v.is_empty() {
                return Err(TemaError::InvalidRange { start, end, step });
            }
            return Ok(v);
        }
        let it = (start..=end).step_by(step);
        let v: Vec<usize> = it.collect();
        if v.is_empty() {
            return Err(TemaError::InvalidRange { start, end, step });
        }
        Ok(v)
    }
    let periods = axis_usize(r.period)?;
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(TemaParams { period: Some(p) });
    }
    Ok(out)
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
    let combos = expand_grid(sweep)?;
    if combos.is_empty() {
        return Err(TemaError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
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

    let rows = combos.len();
    let cols = data.len();

    let _total = rows
        .checked_mul(cols)
        .ok_or_else(|| TemaError::InvalidInput("rows * cols overflow".into()))?;

    let actual_kern = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    };

    let warm: Vec<usize> = combos
        .iter()
        .map(|c| (first + (c.period.unwrap() - 1) * 3).min(cols))
        .collect();

    let mut raw = make_uninit_matrix(rows, cols);
    unsafe { init_matrix_prefixes(&mut raw, cols, &warm) };

    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();

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

    let mut guard = core::mem::ManuallyDrop::new(raw);
    let values: Vec<f64> = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_tema_into_matches_api() -> Result<(), Box<dyn Error>> {
        let len = 256usize;
        let mut data = vec![f64::NAN; len];
        for i in 2..len {
            let x = i as f64;
            data[i] = (x * 0.31337).sin() * 10.0 + x * 0.01;
        }

        let input = TemaInput::from_slice(&data, TemaParams::default());

        let baseline = tema(&input)?.values;

        let mut out = vec![0.0; len];
        #[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
        {
            tema_into(&input, &mut out)?;
        }
        #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
        {
            tema_into_slice(&mut out, &input, Kernel::Auto)?;
        }

        assert_eq!(baseline.len(), out.len());

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12
        }

        for (i, (&a, &b)) in baseline.iter().zip(out.iter()).enumerate() {
            assert!(
                eq_or_both_nan(a, b),
                "parity mismatch at idx {}: api={} into={} diff={}",
                i,
                a,
                b,
                (a - b).abs()
            );
        }

        Ok(())
    }

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

    #[cfg(debug_assertions)]
    fn check_tema_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let test_periods = vec![5, 9, 14, 20, 50, 100, 200];

        for &period in &test_periods {
            let params = TemaParams {
                period: Some(period),
            };
            let input = TemaInput::from_candles(&candles, "close", params);

            if candles.close.len() < period {
                continue;
            }

            let output = match tema_with_kernel(&input, kernel) {
                Ok(o) => o,
                Err(_) => continue,
            };

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} with period {}",
						test_name, val, bits, i, period
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} with period {}",
						test_name, val, bits, i, period
					);
                }

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

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let close_data = &candles.close;

        let strat = (
            2usize..=50,
            0usize..close_data.len().saturating_sub(200),
            100usize..=200,
        );

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(period, start_idx, slice_len)| {
                let end_idx = (start_idx + slice_len).min(close_data.len());
                if end_idx <= start_idx || end_idx - start_idx < period * 3 + 10 {
                    return Ok(());
                }

                let data_slice = &close_data[start_idx..end_idx];
                let params = TemaParams {
                    period: Some(period),
                };
                let input = TemaInput::from_slice(data_slice, params);

                let result = tema_with_kernel(&input, kernel);

                let scalar_result = tema_with_kernel(&input, Kernel::Scalar);

                match (result, scalar_result) {
                    (Ok(TemaOutput { values: out }), Ok(TemaOutput { values: ref_out })) => {
                        prop_assert_eq!(out.len(), data_slice.len());
                        prop_assert_eq!(ref_out.len(), data_slice.len());

                        let first = data_slice.iter().position(|x| !x.is_nan()).unwrap_or(0);
                        let lookback = (period - 1) * 3;
                        let expected_warmup = first + lookback;

                        for i in 0..expected_warmup.min(out.len()) {
                            prop_assert!(
                                out[i].is_nan(),
                                "Expected NaN at index {} during warmup, got {}",
                                i,
                                out[i]
                            );
                        }

                        let multiplier = 2.0 / (period as f64 + 1.0);
                        prop_assert!(
                            multiplier > 0.0 && multiplier <= 1.0,
                            "EMA multiplier should be in (0, 1]: {}",
                            multiplier
                        );

                        for i in expected_warmup..out.len() {
                            let y = out[i];
                            let r = ref_out[i];

                            prop_assert!(!y.is_nan(), "Unexpected NaN at index {}", i);
                            prop_assert!(y.is_finite(), "Non-finite value at index {}: {}", i, y);

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

                            let ulp_diff: u64 = y_bits.abs_diff(r_bits);
                            prop_assert!(
                                (y - r).abs() <= 1e-9 || ulp_diff <= 5,
                                "Kernel mismatch at {}: {} vs {} (ULP={})",
                                i,
                                y,
                                r,
                                ulp_diff
                            );
                        }

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
                            let const_warmup = lookback;
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

                        if period <= 20 {
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

                        if data_slice.len() > period * 2 {
                            let trend_start = expected_warmup;
                            let trend_end = (trend_start + period).min(data_slice.len());

                            if trend_end > trend_start + 3 {
                                let trend_data = &data_slice[trend_start..trend_end];
                                let is_uptrend =
                                    trend_data.windows(2).filter(|w| w[1] > w[0]).count()
                                        > trend_data.windows(2).filter(|w| w[1] < w[0]).count();

                                if is_uptrend {
                                    let last_price = data_slice[trend_end - 1];
                                    let tema_value = out[trend_end - 1];

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

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let test_configs = vec![
            (5, 15, 2),
            (10, 50, 5),
            (20, 100, 10),
            (50, 200, 25),
            (3, 3, 1),
            (150, 150, 1),
        ];

        for (start, end, step) in test_configs {
            let output = TemaBatchBuilder::new()
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
                let period = output
                    .combos
                    .get(row)
                    .map(|p| p.period.unwrap_or(0))
                    .unwrap_or(0);

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (period {}, flat index {})",
                        test, val, bits, row, col, period, idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (period {}, flat index {})",
                        test, val, bits, row, col, period, idx
                    );
                }

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

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);
}

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
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    Ok((data, period, first, len, chosen))
}

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
                tema_scalar(data, period, first, out)
            }
            Kernel::Auto => unreachable!(),
        }
    }
}

#[inline(always)]
fn tema_batch_inner_into(
    data: &[f64],
    sweep: &TemaBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<TemaParams>, TemaError> {
    let combos = expand_grid(sweep)?;
    if combos.is_empty() {
        return Err(TemaError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
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

    let rows = combos.len();
    let cols = data.len();
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| TemaError::InvalidInput("rows * cols overflow".into()))?;
    if out.len() != expected {
        return Err(TemaError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    let actual_kern = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    };

    let warm: Vec<usize> = combos
        .iter()
        .map(|c| (first + (c.period.unwrap() - 1) * 3).min(cols))
        .collect();

    let out_uninit = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };

    unsafe { init_matrix_prefixes(out_uninit, cols, &warm) };

    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();

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

#[cfg(feature = "python")]
#[pyfunction(name = "tema")]
#[pyo3(signature = (data, period, kernel=None))]

pub fn tema_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let kern = validate_kernel(kernel, false)?;

    let params = TemaParams {
        period: Some(period),
    };
    let result_vec: Vec<f64> = if let Ok(slice_in) = data.as_slice() {
        let tema_in = TemaInput::from_slice(slice_in, params);
        py.allow_threads(|| tema_with_kernel(&tema_in, kern).map(|o| o.values))
            .map_err(|e| PyValueError::new_err(e.to_string()))?
    } else {
        let owned = data.as_array().to_owned();
        let slice_in = owned.as_slice().expect("owned array should be contiguous");
        let tema_in = TemaInput::from_slice(slice_in, params);
        py.allow_threads(|| tema_with_kernel(&tema_in, kern).map(|o| o.values))
            .map_err(|e| PyValueError::new_err(e.to_string()))?
    };

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

    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "tema_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]

pub fn tema_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    let sweep = TemaBatchRange {
        period: period_range,
    };

    let combos = expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = slice_in.len();

    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows * cols overflow"))?;
    let out_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    let combos = py
        .allow_threads(|| {
            let kernel = match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                k => k,
            };

            let simd = match kernel {
                Kernel::Avx512Batch => Kernel::Avx512,
                Kernel::Avx2Batch => Kernel::Avx2,
                Kernel::ScalarBatch => Kernel::Scalar,
                _ => kernel,
            };

            tema_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

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

    let inner = py.allow_threads(|| -> Result<_, PyErr> {
        let cuda = CudaTema::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let inner = cuda
            .tema_batch_dev(slice_in, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(inner)
    })?;
    make_device_array_py(device_id, inner)
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

    let inner = py.allow_threads(|| -> Result<_, PyErr> {
        let cuda = CudaTema::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let inner = cuda
            .tema_many_series_one_param_time_major_dev(prices_flat, cols, rows, period)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(inner)
    })?;
    make_device_array_py(device_id, inner)
}

#[inline]
pub fn tema_into_slice(dst: &mut [f64], input: &TemaInput, kern: Kernel) -> Result<(), TemaError> {
    let (data, period, first, len, chosen) = tema_prepare(input, kern)?;
    if dst.len() != len {
        return Err(TemaError::OutputLengthMismatch {
            expected: len,
            got: dst.len(),
        });
    }
    tema_compute_into(data, period, first, chosen, dst);
    let warm = first + (period - 1) * 3;
    let pref = warm.min(len);
    for v in &mut dst[..pref] {
        *v = f64::NAN;
    }
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn tema_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
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

    if data.iter().all(|&x| x.is_nan()) {
        return Err(JsValue::from_str("All values are NaN"));
    }

    let first_valid = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
    let valid_count = data.len() - first_valid;
    if valid_count < period {
        return Err(JsValue::from_str(&format!(
            "Not enough valid data: need {} but only {} valid values after NaN values",
            period, valid_count
        )));
    }

    if period > 1 {
        let lookback = (period - 1) * 3;
        if first_valid + lookback >= data.len() {
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

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct TemaBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct TemaBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<TemaParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
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

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
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

    tema_batch_inner(data, &sweep, Kernel::Scalar, false)
        .map(|output| output.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn tema_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn tema_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
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

    if period > 1 && !data.iter().all(|&x| x.is_nan()) {
        let lookback = (period - 1) * 3;
        if lookback >= len {
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
        let mut temp = vec![0.0; len];
        tema_into_slice(&mut temp, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, len) };
        out.copy_from_slice(&temp);
    } else {
        let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, len) };
        tema_into_slice(out, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }

    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
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

    let combos = expand_grid(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let cols = data.len();
    let total_size = rows * cols;

    let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, total_size) };

    tema_batch_inner_into(data, &sweep, Kernel::Auto, false, out_slice)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(rows)
}
