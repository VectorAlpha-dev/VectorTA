use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use serde::{Deserialize, Serialize};
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::CudaMeanAd;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::alma::{make_device_array_py, DeviceArrayF32Py};

impl<'a> AsRef<[f64]> for MeanAdInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            MeanAdData::Slice(slice) => slice,
            MeanAdData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum MeanAdData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct MeanAdOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct MeanAdParams {
    pub period: Option<usize>,
}

impl Default for MeanAdParams {
    fn default() -> Self {
        Self { period: Some(5) }
    }
}

#[derive(Debug, Clone)]
pub struct MeanAdInput<'a> {
    pub data: MeanAdData<'a>,
    pub params: MeanAdParams,
}

impl<'a> MeanAdInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: MeanAdParams) -> Self {
        Self {
            data: MeanAdData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: MeanAdParams) -> Self {
        Self {
            data: MeanAdData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", MeanAdParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(5)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct MeanAdBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for MeanAdBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl MeanAdBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<MeanAdOutput, MeanAdError> {
        let p = MeanAdParams {
            period: self.period,
        };
        let i = MeanAdInput::from_candles(c, "close", p);
        mean_ad_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<MeanAdOutput, MeanAdError> {
        let p = MeanAdParams {
            period: self.period,
        };
        let i = MeanAdInput::from_slice(d, p);
        mean_ad_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<MeanAdStream, MeanAdError> {
        let p = MeanAdParams {
            period: self.period,
        };
        MeanAdStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum MeanAdError {
    #[error("mean_ad: Empty data provided.")]
    EmptyInputData,
    #[error("mean_ad: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("mean_ad: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("mean_ad: All values are NaN.")]
    AllValuesNaN,
    #[error("mean_ad: Output length mismatch: expected {expected}, got {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("mean_ad: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("mean_ad: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(crate::utilities::enums::Kernel),
}

#[inline]
pub fn mean_ad(input: &MeanAdInput) -> Result<MeanAdOutput, MeanAdError> {
    mean_ad_with_kernel(input, Kernel::Auto)
}

pub fn mean_ad_with_kernel(
    input: &MeanAdInput,
    kernel: Kernel,
) -> Result<MeanAdOutput, MeanAdError> {
    let data: &[f64] = match &input.data {
        MeanAdData::Candles { candles, source } => source_type(candles, source),
        MeanAdData::Slice(sl) => sl,
    };

    if data.is_empty() {
        return Err(MeanAdError::EmptyInputData);
    }

    let period = input.get_period();
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(MeanAdError::AllValuesNaN)?;

    if period == 0 || period > data.len() {
        return Err(MeanAdError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }
    if (data.len() - first) < period {
        return Err(MeanAdError::NotEnoughValidData {
            needed: period,
            valid: data.len() - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    match chosen {
        Kernel::Scalar | Kernel::ScalarBatch => mean_ad_scalar(data, period, first),
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 | Kernel::Avx2Batch => mean_ad_avx2(data, period, first),
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 | Kernel::Avx512Batch => mean_ad_avx512(data, period, first),
        _ => unreachable!(),
    }
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn mean_ad_into(input: &MeanAdInput, out: &mut [f64]) -> Result<(), MeanAdError> {
    mean_ad_into_slice(out, input, Kernel::Auto)
}

#[inline]
pub fn mean_ad_scalar(
    data: &[f64],
    period: usize,
    first: usize,
) -> Result<MeanAdOutput, MeanAdError> {
    if period == 0 {
        return Err(MeanAdError::InvalidPeriod {
            period: 0,
            data_len: data.len(),
        });
    }
    if first > data.len() {
        return Err(MeanAdError::NotEnoughValidData {
            needed: first,
            valid: data.len(),
        });
    }

    let n = data.len();

    if first + period > n {
        let out = alloc_with_nan_prefix(n, n);
        return Ok(MeanAdOutput { values: out });
    }

    let inv_p = 1.0f64 / (period as f64);

    let warmup_end = first + (period << 1) - 2;
    let mut out = alloc_with_nan_prefix(n, warmup_end.min(n));

    let mut sum = 0.0f64;
    for i in first..(first + period) {
        sum += data[i];
    }
    let mut sma = sum * inv_p;

    let mut residual_buffer = vec![0.0f64; period];
    let mut buffer_index = 0usize;
    let mut residual_sum = 0.0f64;

    let start_t = first + period - 1;
    let fill_t_end = (start_t + period - 1).min(n.saturating_sub(1));
    for t in start_t..=fill_t_end {
        let residual = (data[t] - sma).abs();
        residual_buffer[buffer_index] = residual;
        residual_sum += residual;
        buffer_index += 1;
        if buffer_index == period {
            buffer_index = 0;
        }
        if t + 1 < n {
            sum += data[t + 1] - data[t + 1 - period];
            sma = sum * inv_p;
        }
    }

    if warmup_end < n {
        out[warmup_end] = residual_sum * inv_p;
    }

    let mut t = start_t + period;
    while t < n {
        let residual = (data[t] - sma).abs();

        let old = residual_buffer[buffer_index];
        residual_sum += residual - old;
        residual_buffer[buffer_index] = residual;
        buffer_index += 1;
        if buffer_index == period {
            buffer_index = 0;
        }

        out[t] = residual_sum * inv_p;

        if t + 1 < n {
            sum += data[t + 1] - data[t + 1 - period];
            sma = sum * inv_p;
        }
        t += 1;
    }

    Ok(MeanAdOutput { values: out })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn mean_ad_avx2(
    data: &[f64],
    period: usize,
    first: usize,
) -> Result<MeanAdOutput, MeanAdError> {
    mean_ad_scalar(data, period, first)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn mean_ad_avx512(
    data: &[f64],
    period: usize,
    first: usize,
) -> Result<MeanAdOutput, MeanAdError> {
    mean_ad_scalar(data, period, first)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn mean_ad_avx512_short(
    data: &[f64],
    period: usize,
    first: usize,
) -> Result<MeanAdOutput, MeanAdError> {
    mean_ad_scalar(data, period, first)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn mean_ad_avx512_long(
    data: &[f64],
    period: usize,
    first: usize,
) -> Result<MeanAdOutput, MeanAdError> {
    mean_ad_scalar(data, period, first)
}

#[inline(always)]
pub fn mean_ad_batch_with_kernel(
    data: &[f64],
    sweep: &MeanAdBatchRange,
    k: Kernel,
) -> Result<MeanAdBatchOutput, MeanAdError> {
    let kernel = match k {
        Kernel::Auto => Kernel::ScalarBatch,
        other if other.is_batch() => other,
        other => return Err(MeanAdError::InvalidKernelForBatch(other)),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    mean_ad_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct MeanAdBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for MeanAdBatchRange {
    fn default() -> Self {
        Self {
            period: (5, 254, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MeanAdBatchBuilder {
    range: MeanAdBatchRange,
    kernel: Kernel,
}

impl MeanAdBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<MeanAdBatchOutput, MeanAdError> {
        mean_ad_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<MeanAdBatchOutput, MeanAdError> {
        MeanAdBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<MeanAdBatchOutput, MeanAdError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<MeanAdBatchOutput, MeanAdError> {
        MeanAdBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct MeanAdBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<MeanAdParams>,
    pub rows: usize,
    pub cols: usize,
}
impl MeanAdBatchOutput {
    pub fn row_for_params(&self, p: &MeanAdParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(5) == p.period.unwrap_or(5))
    }
    pub fn values_for(&self, p: &MeanAdParams) -> Option<&[f64]> {
        self.row_for_params(p).and_then(|row| {
            let start = row.checked_mul(self.cols)?;
            self.values.get(start..start + self.cols)
        })
    }
}

#[inline(always)]
fn expand_grid(r: &MeanAdBatchRange) -> Result<Vec<MeanAdParams>, MeanAdError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, MeanAdError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            let st = step.max(1);
            let v: Vec<usize> = (start..=end).step_by(st).collect();
            if v.is_empty() {
                return Err(MeanAdError::InvalidRange {
                    start: start.to_string(),
                    end: end.to_string(),
                    step: step.to_string(),
                });
            }
            return Ok(v);
        }

        let mut v = Vec::new();
        let mut x = start as isize;
        let end_i = end as isize;
        let st = (step as isize).max(1);
        while x >= end_i {
            v.push(x as usize);
            x -= st;
        }
        if v.is_empty() {
            return Err(MeanAdError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        Ok(v)
    }

    let periods = axis_usize(r.period)?;
    if periods.is_empty() {
        return Err(MeanAdError::InvalidRange {
            start: r.period.0.to_string(),
            end: r.period.1.to_string(),
            step: r.period.2.to_string(),
        });
    }

    let mut out = Vec::with_capacity(periods.len());
    for p in periods {
        out.push(MeanAdParams { period: Some(p) });
    }
    Ok(out)
}

#[inline(always)]
pub fn mean_ad_batch_slice(
    data: &[f64],
    sweep: &MeanAdBatchRange,
    kern: Kernel,
) -> Result<MeanAdBatchOutput, MeanAdError> {
    mean_ad_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn mean_ad_batch_par_slice(
    data: &[f64],
    sweep: &MeanAdBatchRange,
    kern: Kernel,
) -> Result<MeanAdBatchOutput, MeanAdError> {
    mean_ad_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn mean_ad_batch_inner_into(
    data: &[f64],
    sweep: &MeanAdBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<MeanAdParams>, MeanAdError> {
    if data.is_empty() {
        return Err(MeanAdError::EmptyInputData);
    }
    let combos = expand_grid(sweep)?;
    if combos.iter().any(|c| c.period.unwrap_or(0) == 0) {
        return Err(MeanAdError::InvalidPeriod {
            period: 0,
            data_len: data.len(),
        });
    }
    let expected =
        combos
            .len()
            .checked_mul(data.len())
            .ok_or_else(|| MeanAdError::OutputLengthMismatch {
                expected: usize::MAX,
                got: out.len(),
            })?;
    if out.len() != expected {
        return Err(MeanAdError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(MeanAdError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if max_p > data.len() {
        return Err(MeanAdError::InvalidPeriod {
            period: max_p,
            data_len: data.len(),
        });
    }
    if data.len() - first < max_p {
        return Err(MeanAdError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();

    let chosen = match kern {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    let do_row = |row: usize, out_row: &mut [f64]| {
        let period = combos[row].period.unwrap();
        match chosen {
            Kernel::Scalar => mean_ad_row_scalar(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => mean_ad_row_avx2(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => mean_ad_row_avx512(data, first, period, out_row),
            _ => mean_ad_row_scalar(data, first, period, out_row),
        }
    };

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

    Ok(combos)
}

#[inline(always)]
fn mean_ad_batch_inner(
    data: &[f64],
    sweep: &MeanAdBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<MeanAdBatchOutput, MeanAdError> {
    if data.is_empty() {
        return Err(MeanAdError::EmptyInputData);
    }
    let combos = expand_grid(sweep)?;
    if combos.iter().any(|c| c.period.unwrap_or(0) == 0) {
        return Err(MeanAdError::InvalidPeriod {
            period: 0,
            data_len: data.len(),
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(MeanAdError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if max_p > data.len() {
        return Err(MeanAdError::InvalidPeriod {
            period: max_p,
            data_len: data.len(),
        });
    }
    if data.len() - first < max_p {
        return Err(MeanAdError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();

    let warmup_periods: Vec<usize> = combos
        .iter()
        .map(|c| first + 2 * c.period.unwrap() - 2)
        .collect();

    let mut buf_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut buf_mu, cols, &warmup_periods);

    let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
    let values_ptr = buf_guard.as_mut_ptr() as *mut f64;
    let values_slice: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(values_ptr, buf_guard.len()) };

    let chosen = match kern {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    let do_row = |row: usize, out_row: &mut [f64]| {
        let period = combos[row].period.unwrap();

        let warmup_end = first + 2 * period - 2;
        for i in 0..warmup_end.min(out_row.len()) {
            out_row[i] = f64::NAN;
        }
        match chosen {
            Kernel::Scalar => mean_ad_row_scalar(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => mean_ad_row_avx2(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => mean_ad_row_avx512(data, first, period, out_row),
            _ => mean_ad_row_scalar(data, first, period, out_row),
        }
    };
    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            values_slice
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in values_slice.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in values_slice.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(MeanAdBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub fn mean_ad_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    if first + period > data.len() {
        return;
    }

    let n = data.len();
    let inv_p = 1.0f64 / (period as f64);

    let mut sum = 0.0f64;
    for i in first..(first + period) {
        sum += data[i];
    }
    let mut sma = sum * inv_p;

    let mut residual_buffer = vec![0.0f64; period];
    let mut buffer_index = 0usize;
    let mut residual_sum = 0.0f64;

    let start_t = first + period - 1;
    let fill_t_end = (start_t + period - 1).min(n.saturating_sub(1));
    for t in start_t..=fill_t_end {
        let residual = (data[t] - sma).abs();
        residual_buffer[buffer_index] = residual;
        residual_sum += residual;
        buffer_index += 1;
        if buffer_index == period {
            buffer_index = 0;
        }
        if t + 1 < n {
            sum += data[t + 1] - data[t + 1 - period];
            sma = sum * inv_p;
        }
    }

    let first_output = first + (period << 1) - 2;
    if first_output < n {
        out[first_output] = residual_sum * inv_p;
    }

    let mut t = start_t + period;
    while t < n {
        let residual = (data[t] - sma).abs();

        let old = residual_buffer[buffer_index];
        residual_sum += residual - old;
        residual_buffer[buffer_index] = residual;
        buffer_index += 1;
        if buffer_index == period {
            buffer_index = 0;
        }

        out[t] = residual_sum * inv_p;

        if t + 1 < n {
            sum += data[t + 1] - data[t + 1 - period];
            sma = sum * inv_p;
        }
        t += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn mean_ad_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    mean_ad_row_scalar(data, first, period, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn mean_ad_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    if period <= 32 {
        mean_ad_row_avx512_short(data, first, period, out);
    } else {
        mean_ad_row_avx512_long(data, first, period, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn mean_ad_row_avx512_short(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    mean_ad_row_scalar(data, first, period, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn mean_ad_row_avx512_long(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    mean_ad_row_scalar(data, first, period, out);
}

#[derive(Debug, Clone)]
pub struct MeanAdStream {
    period: usize,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,
    mean_buffer: Vec<f64>,
    mean_head: usize,
    mean_filled: bool,
    mean: f64,
    mad: f64,
}

impl MeanAdStream {
    pub fn try_new(params: MeanAdParams) -> Result<Self, MeanAdError> {
        let period = params.period.unwrap_or(5);
        if period == 0 {
            return Err(MeanAdError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        Ok(Self {
            period,
            buffer: vec![f64::NAN; period],
            head: 0,
            filled: false,
            mean_buffer: vec![f64::NAN; period],
            mean_head: 0,
            mean_filled: false,
            mean: 0.0,
            mad: 0.0,
        })
    }
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        let p = self.period;
        let inv_p = 1.0f64 / (p as f64);

        let price_idx = self.head;
        let old_x = self.buffer[price_idx];

        let resid_idx = self.mean_head;
        let old_r = self.mean_buffer[resid_idx];

        self.buffer[price_idx] = value;
        let next_price_idx = price_idx + 1;
        let wrapped_prices = next_price_idx == p;
        self.head = if wrapped_prices { 0 } else { next_price_idx };

        let just_filled_prices = !self.filled && wrapped_prices;
        if just_filled_prices {
            self.filled = true;
        }

        if !self.filled {
            self.mean += value;
            return None;
        }

        let sum_t = if just_filled_prices {
            self.mean + value
        } else {
            self.mean + value - old_x
        };
        let mean_t = sum_t * inv_p;

        self.mean = sum_t;

        let resid_t = (value - mean_t).abs();

        self.mean_buffer[resid_idx] = resid_t;
        let next_resid_idx = resid_idx + 1;
        let wrapped_resids = next_resid_idx == p;
        self.mean_head = if wrapped_resids { 0 } else { next_resid_idx };

        if !self.mean_filled {
            self.mad += resid_t;
            if wrapped_resids {
                self.mean_filled = true;
                return Some(self.mad * inv_p);
            }
            return None;
        }

        self.mad = self.mad + resid_t - old_r;
        Some(self.mad * inv_p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use paste::paste;

    fn check_mean_ad_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = MeanAdParams { period: None };
        let input = MeanAdInput::from_candles(&candles, "close", default_params);
        let output = mean_ad_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_mean_ad_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = MeanAdInput::from_candles(&candles, "hl2", MeanAdParams { period: Some(5) });
        let result = mean_ad_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        let expected_last_five = [
            199.71999999999971,
            104.14000000000087,
            133.4,
            100.54000000000087,
            117.98000000000029,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-1,
                "[{}] MeanAd {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_mean_ad_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = MeanAdInput::with_default_candles(&candles);
        match input.data {
            MeanAdData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected MeanAdData::Candles"),
        }
        let output = mean_ad_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_mean_ad_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = MeanAdParams { period: Some(0) };
        let input = MeanAdInput::from_slice(&input_data, params);
        let res = mean_ad_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] MeanAd should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_mean_ad_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = MeanAdParams { period: Some(10) };
        let input = MeanAdInput::from_slice(&data_small, params);
        let res = mean_ad_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] MeanAd should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_mean_ad_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = MeanAdParams { period: Some(5) };
        let input = MeanAdInput::from_slice(&single_point, params);
        let res = mean_ad_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] MeanAd should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_mean_ad_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = MeanAdParams { period: Some(5) };
        let first_input = MeanAdInput::from_candles(&candles, "close", first_params);
        let first_result = mean_ad_with_kernel(&first_input, kernel)?;
        let params2 = MeanAdParams { period: Some(3) };
        let second_input = MeanAdInput::from_slice(&first_result.values, params2);
        let second_result = mean_ad_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    fn check_mean_ad_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = MeanAdInput::from_candles(&candles, "close", MeanAdParams { period: Some(5) });
        let res = mean_ad_with_kernel(&input, kernel)?;
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

    fn check_mean_ad_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 5;
        let input = MeanAdInput::from_candles(
            &candles,
            "close",
            MeanAdParams {
                period: Some(period),
            },
        );
        let batch_output = mean_ad_with_kernel(&input, kernel)?.values;
        let mut stream = MeanAdStream::try_new(MeanAdParams {
            period: Some(period),
        })?;

        let mut stream_uninit: Vec<MaybeUninit<f64>> = Vec::with_capacity(candles.close.len());
        unsafe {
            stream_uninit.set_len(candles.close.len());
        }

        for (i, &price) in candles.close.iter().enumerate() {
            let val = match stream.update(price) {
                Some(mean_ad_val) => mean_ad_val,
                None => f64::NAN,
            };
            stream_uninit[i] = MaybeUninit::new(val);
        }

        let stream_values = unsafe {
            let ptr = stream_uninit.as_mut_ptr() as *mut f64;
            let len = stream_uninit.len();
            let cap = stream_uninit.capacity();
            std::mem::forget(stream_uninit);
            Vec::from_raw_parts(ptr, len, cap)
        };
        assert_eq!(batch_output.len(), stream_values.len());
        for (i, (&b, &s)) in batch_output.iter().zip(stream_values.iter()).enumerate() {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] MeanAd streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_mean_ad_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let test_params = vec![
            MeanAdParams::default(),
            MeanAdParams { period: Some(2) },
            MeanAdParams { period: Some(3) },
            MeanAdParams { period: Some(5) },
            MeanAdParams { period: Some(7) },
            MeanAdParams { period: Some(10) },
            MeanAdParams { period: Some(14) },
            MeanAdParams { period: Some(20) },
            MeanAdParams { period: Some(30) },
            MeanAdParams { period: Some(50) },
            MeanAdParams { period: Some(100) },
            MeanAdParams { period: Some(200) },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = MeanAdInput::from_candles(&candles, "close", params.clone());
            let output = mean_ad_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(5),
                        param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(5),
                        param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(5),
                        param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_mean_ad_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    macro_rules! generate_all_mean_ad_tests {
        ($($test_fn:ident),*) => {
            paste! {
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
    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_mean_ad_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (2usize..=64)
            .prop_flat_map(|period| {
                (
                    prop::collection::vec(
                        (10.0f64..1000.0f64).prop_filter("finite", |x| x.is_finite()),
                        period..400,
                    ),
                    Just(period),
                    prop::bool::weighted(0.1),
                )
            })
            .prop_map(|(mut data, period, make_constant)| {
                if make_constant && data.len() > 0 {
                    let constant_val = data[0];
                    data.iter_mut().for_each(|v| *v = constant_val);
                }
                (data, period)
            });

        proptest::test_runner::TestRunner::default().run(&strat, |(data, period)| {
            let params = MeanAdParams {
                period: Some(period),
            };
            let input = MeanAdInput::from_slice(&data, params);

            let MeanAdOutput { values: out } = mean_ad_with_kernel(&input, kernel)?;
            let MeanAdOutput { values: ref_out } = mean_ad_with_kernel(&input, Kernel::Scalar)?;

            prop_assert_eq!(
                out.len(),
                data.len(),
                "[{}] Output length mismatch",
                test_name
            );

            let first_valid = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
            let warmup_period = first_valid + 2 * period - 2;

            for i in 0..warmup_period.min(out.len()) {
                prop_assert!(
                    out[i].is_nan(),
                    "[{}] Expected NaN during warmup at index {}, got {}",
                    test_name,
                    i,
                    out[i]
                );
            }

            for i in warmup_period..out.len() {
                if !out[i].is_nan() {
                    prop_assert!(
                        out[i] >= -1e-10,
                        "[{}] MAD should be non-negative at index {}: got {}",
                        test_name,
                        i,
                        out[i]
                    );
                }
            }

            for i in 0..out.len() {
                let y = out[i];
                let r = ref_out[i];

                if y.is_nan() || r.is_nan() {
                    prop_assert!(
                        y.is_nan() && r.is_nan(),
                        "[{}] NaN mismatch at index {}: kernel={}, scalar={}",
                        test_name,
                        i,
                        y,
                        r
                    );
                    continue;
                }

                let y_bits = y.to_bits();
                let r_bits = r.to_bits();
                let ulp_diff = y_bits.abs_diff(r_bits);

                prop_assert!(
                    (y - r).abs() <= 1e-9 || ulp_diff <= 5,
                    "[{}] Kernel mismatch at index {}: kernel={}, scalar={}, diff={}, ULP={}",
                    test_name,
                    i,
                    y,
                    r,
                    (y - r).abs(),
                    ulp_diff
                );
            }

            let is_constant = data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10);
            if is_constant && out.len() > warmup_period {
                for i in warmup_period..out.len() {
                    if !out[i].is_nan() {
                        prop_assert!(
                            out[i].abs() <= 1e-9,
                            "[{}] MAD should be ~0 for constant data at index {}: got {}",
                            test_name,
                            i,
                            out[i]
                        );
                    }
                }
            }

            let is_linear_monotonic = if data.len() >= 3 {
                let diffs: Vec<f64> = data.windows(2).map(|w| w[1] - w[0]).collect();
                let first_diff = diffs[0];
                diffs.iter().all(|&d| (d - first_diff).abs() < 1e-9)
            } else {
                false
            };

            if is_linear_monotonic && out.len() > warmup_period + period {
                for i in (warmup_period + 1)..out.len() {
                    if !out[i].is_nan() && !out[i - 1].is_nan() && out[i - 1] > 1e-10 {
                        let change_ratio = (out[i] - out[i - 1]).abs() / out[i - 1];

                        prop_assert!(
								change_ratio <= 0.1,
								"[{}] MAD changes too much for linear data at index {}: {} -> {} ({:.2}% change)",
								test_name,
								i,
								out[i-1],
								out[i],
								change_ratio * 100.0
							);
                    }
                }
            }

            for i in warmup_period..out.len() {
                if out[i].is_nan() || i < period {
                    continue;
                }

                let window_start = i + 1 - period;
                let window = &data[window_start..=i];

                let window_min = window.iter().cloned().fold(f64::INFINITY, f64::min);
                let window_max = window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let window_range = window_max - window_min;

                prop_assert!(
                    out[i] <= window_range / 2.0 + 1e-9,
                    "[{}] MAD exceeds half window range at index {}: MAD={}, window_range/2={}",
                    test_name,
                    i,
                    out[i],
                    window_range / 2.0
                );
            }

            if period == data.len() && out.len() > warmup_period {
                let non_nan_count = out.iter().filter(|&&v| !v.is_nan()).count();
                prop_assert!(
                    non_nan_count <= 1,
                    "[{}] With period={}, expected at most 1 non-NaN value, got {}",
                    test_name,
                    period,
                    non_nan_count
                );
            }

            if period == 2 && out.len() > warmup_period {
                for i in warmup_period..out.len() {
                    if !out[i].is_nan() && i >= 1 {
                        let expected = (data[i] - data[i - 1]).abs() / 2.0;
                        prop_assert!(
                            (out[i] - expected).abs() <= 1e-9,
                            "[{}] For period=2 at index {}, MAD should be {}, got {}",
                            test_name,
                            i,
                            expected,
                            out[i]
                        );
                    }
                }
            }

            Ok(())
        })?;

        Ok(())
    }

    generate_all_mean_ad_tests!(
        check_mean_ad_partial_params,
        check_mean_ad_accuracy,
        check_mean_ad_default_candles,
        check_mean_ad_zero_period,
        check_mean_ad_period_exceeds_length,
        check_mean_ad_very_small_dataset,
        check_mean_ad_reinput,
        check_mean_ad_nan_handling,
        check_mean_ad_streaming,
        check_mean_ad_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_mean_ad_tests!(check_mean_ad_property);

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = MeanAdBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "hl2")?;

        let def = MeanAdParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            199.71999999999971,
            104.14000000000087,
            133.4,
            100.54000000000087,
            117.98000000000029,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-1,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let test_configs = vec![
            (2, 10, 2),
            (5, 25, 5),
            (30, 60, 15),
            (2, 5, 1),
            (10, 50, 10),
            (3, 15, 3),
            (20, 30, 2),
            (7, 21, 7),
            (100, 200, 50),
        ];

        for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
            let output = MeanAdBatchBuilder::new()
                .kernel(kernel)
                .period_range(p_start, p_end, p_step)
                .apply_candles(&c, "close")?;

            for (idx, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;
                let combo = &output.combos[row];

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(5)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(5)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(5)
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

    #[test]
    fn test_mean_ad_into_matches_api() -> Result<(), Box<dyn Error>> {
        let n = 256usize;
        let mut data = Vec::with_capacity(n);
        data.push(f64::NAN);
        data.push(f64::NAN);
        data.push(f64::NAN);
        for i in 3..n {
            let t = i as f64;
            let v = (t * 0.01).mul_add((t * 0.1).sin() * 100.0, (t * 0.05).cos() * 0.5);
            data.push(v);
        }

        let input = MeanAdInput::from_slice(&data, MeanAdParams::default());
        let baseline = mean_ad(&input)?.values;

        let mut into_out = vec![0.0; baseline.len()];
        #[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
        {
            mean_ad_into(&input, &mut into_out)?;
        }
        #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
        {
            mean_ad_into_slice(&mut into_out, &input, Kernel::Auto)?;
        }

        assert_eq!(baseline.len(), into_out.len());
        #[inline]
        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b) || (a - b).abs() <= 1e-12
        }
        for (i, (a, b)) in baseline.iter().zip(into_out.iter()).enumerate() {
            assert!(
                eq_or_both_nan(*a, *b),
                "divergence at idx {}: api={}, into={}",
                i,
                a,
                b
            );
        }
        Ok(())
    }

    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste! {
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
#[pyfunction(name = "mean_ad")]
#[pyo3(signature = (data, period, kernel=None))]
pub fn mean_ad_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = MeanAdParams {
        period: Some(period),
    };
    let input = MeanAdInput::from_slice(slice_in, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| mean_ad_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction(name = "mean_ad_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
pub fn mean_ad_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    let sweep = MeanAdBatchRange {
        period: period_range,
    };

    let combos_for_shape = expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos_for_shape.len();
    let cols = slice_in.len();

    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("mean_ad_batch_py: size overflow"))?;
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
                _ => unreachable!(),
            };
            mean_ad_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
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
#[pyfunction(name = "mean_ad_cuda_batch_dev")]
#[pyo3(signature = (data_f32, period_range, device_id=0))]
pub fn mean_ad_cuda_batch_dev_py<'py>(
    py: Python<'py>,
    data_f32: numpy::PyReadonlyArray1<'py, f32>,
    period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    if !crate::cuda::cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let slice_in = data_f32.as_slice()?;
    let sweep = MeanAdBatchRange {
        period: period_range,
    };
    let inner = py.allow_threads(|| {
        let cuda = CudaMeanAd::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.mean_ad_batch_dev(slice_in, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    make_device_array_py(device_id, inner)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "mean_ad_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, cols, rows, period, device_id=0))]
pub fn mean_ad_cuda_many_series_one_param_dev_py<'py>(
    py: Python<'py>,
    data_tm_f32: numpy::PyReadonlyArray1<'py, f32>,
    cols: usize,
    rows: usize,
    period: usize,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    if !crate::cuda::cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let slice_in = data_tm_f32.as_slice()?;
    let params = MeanAdParams {
        period: Some(period),
    };
    let inner = py.allow_threads(|| {
        let cuda = CudaMeanAd::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.mean_ad_many_series_one_param_time_major_dev(slice_in, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    make_device_array_py(device_id, inner)
}

#[cfg(feature = "python")]
#[pyclass(name = "MeanAdStream")]
pub struct MeanAdStreamPy {
    stream: MeanAdStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl MeanAdStreamPy {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        let params = MeanAdParams {
            period: Some(period),
        };
        let stream =
            MeanAdStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(MeanAdStreamPy { stream })
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

pub fn mean_ad_into_slice(
    dst: &mut [f64],
    input: &MeanAdInput,
    kern: Kernel,
) -> Result<(), MeanAdError> {
    let data = input.as_ref();
    let period = match input.params.period {
        Some(p) if p > 0 => p,
        _ => {
            return Err(MeanAdError::InvalidPeriod {
                period: input.params.period.unwrap_or(0),
                data_len: data.len(),
            })
        }
    };

    if data.is_empty() {
        return Err(MeanAdError::EmptyInputData);
    }
    if period > data.len() {
        return Err(MeanAdError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }
    if dst.len() != data.len() {
        return Err(MeanAdError::OutputLengthMismatch {
            expected: data.len(),
            got: dst.len(),
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(MeanAdError::AllValuesNaN)?;

    if (data.len() - first) < period {
        return Err(MeanAdError::NotEnoughValidData {
            needed: period,
            valid: data.len() - first,
        });
    }

    let chosen = match kern {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    let warmup_end = first + (period << 1) - 2;
    let warmup_end = warmup_end.min(dst.len());
    if warmup_end > 0 {
        dst[..warmup_end].fill(f64::NAN);
    }

    match chosen {
        Kernel::Scalar | Kernel::ScalarBatch => mean_ad_row_scalar(data, first, period, dst),
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 | Kernel::Avx2Batch => mean_ad_row_avx2(data, first, period, dst),
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 | Kernel::Avx512Batch => mean_ad_row_avx512(data, first, period, dst),
        _ => unreachable!(),
    }

    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn mean_ad_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = MeanAdParams {
        period: Some(period),
    };
    let input = MeanAdInput::from_slice(data, params);

    let mut output = vec![0.0; data.len()];
    mean_ad_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn mean_ad_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to mean_ad_into"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let params = MeanAdParams {
            period: Some(period),
        };
        let input = MeanAdInput::from_slice(data, params);

        if in_ptr == out_ptr {
            let mut temp = vec![0.0; len];
            mean_ad_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            mean_ad_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn mean_ad_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn mean_ad_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct MeanAdBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct MeanAdBatchJsOutput {
    pub values: Vec<f64>,
    pub periods: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = mean_ad_batch)]
pub fn mean_ad_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: MeanAdBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = MeanAdBatchRange {
        period: config.period_range,
    };

    let output = mean_ad_batch_inner(data, &sweep, Kernel::Auto, false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = MeanAdBatchJsOutput {
        values: output.values,
        periods: output.combos.iter().map(|p| p.period.unwrap()).collect(),
        rows: output.rows,
        cols: output.cols,
    };

    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn mean_ad_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to mean_ad_batch_into",
        ));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        let sweep = MeanAdBatchRange {
            period: (period_start, period_end, period_step),
        };

        let combos = expand_grid(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let cols = len;

        if out_ptr as usize % 8 != 0 {
            return Err(JsValue::from_str("Output pointer must be 8-byte aligned"));
        }

        let total = rows
            .checked_mul(cols)
            .ok_or_else(|| JsValue::from_str("mean_ad_batch_into: size overflow"))?;
        let out = std::slice::from_raw_parts_mut(out_ptr, total);

        let kernel = detect_best_batch_kernel();
        let simd = match kernel {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            Kernel::ScalarBatch => Kernel::Scalar,
            _ => unreachable!(),
        };

        mean_ad_batch_inner_into(data, &sweep, simd, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}
