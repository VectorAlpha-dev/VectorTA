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
//! ## Returns
//! - **`Ok(SrwmaOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(SrwmaError)`** otherwise.
//!
//! ## Developer Notes
//! - **AVX2 kernel**: ❌ Stub only - falls back to scalar implementation
//! - **AVX512 kernel**: ⚠️ Partially stubbed - has short/long path structure but both call scalar
//! - **Streaming update**: ⚠️ O(n) complexity - iterates through all period weights for each update
//!   - TODO: Could optimize to O(1) with incremental weight updates using square root properties
//! - **Memory optimization**: ✅ Uses zero-copy helpers (alloc_with_nan_prefix, make_uninit_matrix) for output vectors
//! - **TODO**: Implement AVX2 kernel and complete AVX512 short/long path implementations for vectorized weighted averaging

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
use std::mem::MaybeUninit;
use thiserror::Error;

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
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
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
            data: SrwmaData::Candles {
                candles: c,
                source: s,
            },
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
    #[error("srwma: Input data slice is empty.")]
    EmptyInputData,
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

pub fn srwma_with_kernel(input: &SrwmaInput, kernel: Kernel) -> Result<SrwmaOutput, SrwmaError> {
    let data: &[f64] = match &input.data {
        SrwmaData::Candles { candles, source } => source_type(candles, source),
        SrwmaData::Slice(sl) => sl,
    };

    let len = data.len();
    if len == 0 {
        return Err(SrwmaError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(SrwmaError::AllValuesNaN)?;
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(SrwmaError::InvalidPeriod {
            period,
            data_len: len,
        });
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
    assert_eq!(
        weights.len(),
        period - 1,
        "weights.len() must be period - 1"
    );
    assert!(
        out.len() >= data.len(),
        "`out` must be at least as long as `data`"
    );

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
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
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

    let len = data.len();
    if len == 0 {
        return Err(SrwmaError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(SrwmaError::AllValuesNaN)?;

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
        .map(|c| first + c.period.unwrap() + 1) // +1 because SRWMA starts at first+period+1
        .collect();

    let mut raw = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut raw, cols, &warm);

    // ---------- 2. per-row worker ---------------------------------------
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();
        let w_ptr = flat_slice.as_ptr().add(row * max_wlen);
        let inv_n = *inv_norms.get_unchecked(row);

        // treat this row as &mut [f64]
        let out_row =
            core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        match kern {
            Kernel::Scalar => {
                srwma_row_scalar(data, first, period, max_wlen, w_ptr, inv_n, out_row)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => srwma_row_avx2(data, first, period, max_wlen, w_ptr, inv_n, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => {
                srwma_row_avx512(data, first, period, max_wlen, w_ptr, inv_n, out_row)
            }
            _ => unreachable!(),
        }
    };

    // ---------- 3. fill every row directly into `raw` --------------------
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

    // ---------- 4. finished – convert to Vec<f64> ------------------------
    let mut guard = core::mem::ManuallyDrop::new(raw);
    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(SrwmaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub fn srwma_batch_inner_into(
    data: &[f64],
    sweep: &SrwmaBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<SrwmaParams>, SrwmaError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(SrwmaError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    let len = data.len();
    if len == 0 {
        return Err(SrwmaError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(SrwmaError::AllValuesNaN)?;

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

    // Warmup per row
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() + 1)
        .collect();

    // Cast `out` to MaybeUninit and initialize warm prefixes with NaN
    let out_mu: &mut [MaybeUninit<f64>] = unsafe {
        core::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };
    init_matrix_prefixes(out_mu, cols, &warm);

    // Worker writes each row directly
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();
        let w_ptr = flat_slice.as_ptr().add(row * max_wlen);
        let inv_n = *inv_norms.get_unchecked(row);
        // Treat this row as &mut [f64]
        let out_row =
            core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        match kern {
            Kernel::Scalar | Kernel::ScalarBatch => {
                srwma_row_scalar(data, first, period, max_wlen, w_ptr, inv_n, out_row)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                srwma_row_avx2(data, first, period, max_wlen, w_ptr, inv_n, out_row)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                srwma_row_avx512(data, first, period, max_wlen, w_ptr, inv_n, out_row)
            }
            _ => unreachable!(),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out_mu
            .par_chunks_mut(cols)
            .enumerate()
            .for_each(|(r, row_mu)| do_row(r, row_mu));
        #[cfg(target_arch = "wasm32")]
        for (r, row_mu) in out_mu.chunks_mut(cols).enumerate() {
            do_row(r, row_mu);
        }
    } else {
        for (r, row_mu) in out_mu.chunks_mut(cols).enumerate() {
            do_row(r, row_mu);
        }
    }

    Ok(combos)
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

#[derive(Debug, Clone)]
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
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_srwma_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = SrwmaParams { period: None };
        let input = SrwmaInput::from_candles(&candles, "close", default_params);
        let output = srwma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_srwma_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
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

    fn check_srwma_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
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

    fn check_srwma_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
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

    fn check_srwma_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
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

    fn check_srwma_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
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

    fn check_srwma_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
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

    fn check_srwma_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = SrwmaInput::from_candles(&candles, "close", SrwmaParams { period: Some(14) });
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

    fn check_srwma_streaming(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let period = 14;
        let input = SrwmaInput::from_candles(
            &candles,
            "close",
            SrwmaParams {
                period: Some(period),
            },
        );
        let batch_output = srwma_with_kernel(&input, kernel)?.values;

        let mut stream = SrwmaStream::try_new(SrwmaParams {
            period: Some(period),
        })?;
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
    // Check for poison values in single output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_srwma_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test with multiple parameter combinations to increase chance of catching bugs
        let test_periods = vec![
            3,   // Small period
            5,   // Small period
            14,  // Default period
            30,  // Medium period
            50,  // Large period
            100, // Very large period
            200, // Extra large period
        ];

        for &period in &test_periods {
            // Skip if period would be too large for the data
            if period > candles.close.len() {
                continue;
            }

            let input = SrwmaInput::from_candles(
                &candles,
                "close",
                SrwmaParams {
                    period: Some(period),
                },
            );
            let output = srwma_with_kernel(&input, kernel)?;

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
    fn check_srwma_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_srwma_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Single comprehensive strategy - SRWMA requires period >= 2
        let strat = (2usize..=50).prop_flat_map(|period| {
            (
                prop::collection::vec(
                    (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                    period + 2..400, // Need at least period+2 for valid output
                ),
                Just(period),
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, period)| {
                let params = SrwmaParams {
                    period: Some(period),
                };
                let input = SrwmaInput::from_slice(&data, params);

                // Compute with specified kernel and scalar reference
                let SrwmaOutput { values: out } = srwma_with_kernel(&input, kernel).unwrap();
                let SrwmaOutput { values: ref_out } =
                    srwma_with_kernel(&input, Kernel::Scalar).unwrap();

                // Property 1: Output length must match input
                prop_assert_eq!(out.len(), data.len(), "Output length mismatch");

                // SRWMA warmup is period + 1 (assuming clean input data)
                let warmup_end = period + 1;

                // Property 2: Warmup period values must be NaN
                for i in 0..warmup_end.min(out.len()) {
                    prop_assert!(
                        out[i].is_nan(),
                        "Expected NaN during warmup at index {}, got {}",
                        i,
                        out[i]
                    );
                }

                // Property 3: Values after warmup should be finite and within bounds
                for i in warmup_end..data.len() {
                    let window_start = i + 1 - period;
                    let window_end = i;
                    let window = &data[window_start..=window_end];

                    let lo = window.iter().cloned().fold(f64::INFINITY, f64::min);
                    let hi = window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let y = out[i];
                    let r = ref_out[i];

                    // Property 4: Output within window bounds
                    prop_assert!(
                        y.is_nan() || (y >= lo - 1e-9 && y <= hi + 1e-9),
                        "idx {}: {} ∉ [{}, {}]",
                        i,
                        y,
                        lo,
                        hi
                    );

                    // Property 5: SIMD consistency check
                    if !y.is_finite() || !r.is_finite() {
                        prop_assert_eq!(
                            y.to_bits(),
                            r.to_bits(),
                            "NaN/finite mismatch at idx {}",
                            i
                        );
                        continue;
                    }

                    let ulp_diff = y.to_bits().abs_diff(r.to_bits());
                    prop_assert!(
                        (y - r).abs() <= 1e-9 || ulp_diff <= 4,
                        "SIMD mismatch at idx {}: {} vs {} (ULP={})",
                        i,
                        y,
                        r,
                        ulp_diff
                    );
                }

                // Property 6: Constant data should produce near-constant output
                if data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10) && data.len() > warmup_end {
                    // For constant data, all outputs after warmup should equal the input constant value
                    let const_val = data[0]; // All values are the same, use first one
                    for i in warmup_end..out.len() {
                        if out[i].is_finite() {
                            prop_assert!(
                                (out[i] - const_val).abs() < 1e-6,
                                "Constant data test failed at idx {}: expected {}, got {}",
                                i,
                                const_val,
                                out[i]
                            );
                        }
                    }
                }

                // Property 7: Period=2 edge case (minimum valid period)
                if period == 2 && data.len() > warmup_end {
                    // With period=2, weights are [sqrt(2), sqrt(1)] = [1.414.., 1.0]
                    // Output should be weighted average favoring more recent value
                    for i in warmup_end..out.len() {
                        if out[i].is_finite() {
                            let expected_range_lo = data[i - 1].min(data[i]) - 1e-9;
                            let expected_range_hi = data[i - 1].max(data[i]) + 1e-9;
                            prop_assert!(
                                out[i] >= expected_range_lo && out[i] <= expected_range_hi,
                                "Period=2 test failed at idx {}: {} not in [{}, {}]",
                                i,
                                out[i],
                                expected_range_lo,
                                expected_range_hi
                            );
                        }
                    }
                }

                // Property 8: Monotonicity test for trending data
                // If input is strictly increasing/decreasing, output should follow trend
                let is_increasing = data.windows(2).all(|w| w[1] >= w[0]);
                let is_decreasing = data.windows(2).all(|w| w[1] <= w[0]);

                if (is_increasing || is_decreasing) && data.len() > warmup_end + 10 {
                    // Check trend preservation in stable part (skip immediate post-warmup)
                    for i in (warmup_end + 5)..out.len() - 1 {
                        if out[i].is_finite() && out[i + 1].is_finite() {
                            if is_increasing {
                                prop_assert!(
                                    out[i + 1] >= out[i] - 1e-9,
                                    "Monotonic increase violated at idx {}: {} > {}",
                                    i,
                                    out[i],
                                    out[i + 1]
                                );
                            } else if is_decreasing {
                                prop_assert!(
                                    out[i + 1] <= out[i] + 1e-9,
                                    "Monotonic decrease violated at idx {}: {} < {}",
                                    i,
                                    out[i],
                                    out[i + 1]
                                );
                            }
                        }
                    }
                }

                Ok(())
            })
            .unwrap();

        Ok(())
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
        check_srwma_streaming,
        check_srwma_no_poison,
        check_srwma_property
    );

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
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

    // Check for poison values in batch output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test multiple batch configurations to increase detection coverage
        let batch_configs = vec![
            (3, 10, 1),    // Small range with step 1
            (5, 25, 5),    // Small range with step 5
            (10, 30, 10),  // Medium range with larger step
            (14, 100, 7),  // Default start with step 7
            (50, 200, 50), // Very large periods
            (2, 8, 2),     // Edge case: very small range
        ];

        for (start, end, step) in batch_configs {
            // Skip configurations that would exceed data length
            if start > c.close.len() {
                continue;
            }

            let output = SrwmaBatchBuilder::new()
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
                let period = output.combos[row].period.unwrap_or(0);

                // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {}) for period {} in range ({}, {}, {})",
                        test, val, bits, row, col, idx, period, start, end, step
                    );
                }

                // Check for init_matrix_prefixes poison (0x22222222_22222222)
                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {}) for period {} in range ({}, {}, {})",
                        test, val, bits, row, col, idx, period, start, end, step
                    );
                }

                // Check for make_uninit_matrix poison (0x33333333_33333333)
                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {}) for period {} in range ({}, {}, {})",
                        test, val, bits, row, col, idx, period, start, end, step
                    );
                }
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(
        _test: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);
}

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::cuda_available;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::CudaSrwma;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::alma::DeviceArrayF32Py;
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(feature = "python")]
use numpy::PyUntypedArrayMethods;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
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

#[cfg(feature = "python")]
#[pyfunction(name = "srwma")]
#[pyo3(signature = (data, period, kernel=None))]
/// Compute the Square Root Weighted Moving Average (SRWMA) of the input data.
///
/// SRWMA assigns weights proportional to the square root of the distance from
/// the current bar, providing moderate emphasis on recent data while reducing noise.
///
/// Parameters:
/// -----------
/// data : np.ndarray
///     Input data array (float64).
/// period : int
///     The look-back window size for weighting (must be >= 2).
/// kernel : str, optional
///     Computation kernel to use: 'auto', 'scalar', 'avx2', 'avx512'.
///     Default is 'auto' which auto-detects the best available.
///
/// Returns:
/// --------
/// np.ndarray
///     Array of SRWMA values, same length as input.
///
/// Raises:
/// -------
/// ValueError
///     If inputs are invalid (period < 2, data too short, etc).
pub fn srwma_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = SrwmaParams {
        period: Some(period),
    };
    let srwma_in = SrwmaInput::from_slice(slice_in, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| srwma_with_kernel(&srwma_in, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "SrwmaStream")]
pub struct SrwmaStreamPy {
    stream: SrwmaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl SrwmaStreamPy {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        let params = SrwmaParams {
            period: Some(period),
        };
        let stream =
            SrwmaStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(SrwmaStreamPy { stream })
    }

    /// Updates the stream with a new value and returns the calculated SRWMA value.
    /// Returns `None` if the buffer is not yet full.
    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "srwma_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
/// Compute SRWMA for multiple period values in a single pass.
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
pub fn srwma_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    let sweep = SrwmaBatchRange {
        period: period_range,
    };

    // Expand grid to know dimensions
    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();

    // Pre-allocate NumPy array
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    // Heavy computation without GIL
    let combos = py
        .allow_threads(|| {
            // Resolve kernel
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

            // Process batch directly into pre-allocated buffer
            srwma_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Build return dictionary
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
#[pyfunction(name = "srwma_cuda_batch_dev")]
#[pyo3(signature = (data_f32, period_range=(14, 50, 2), device_id=0))]
pub fn srwma_cuda_batch_dev_py(
    py: Python<'_>,
    data_f32: PyReadonlyArray1<'_, f32>,
    period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let slice_in = data_f32.as_slice()?;
    let sweep = SrwmaBatchRange {
        period: period_range,
    };

    let inner = py.allow_threads(|| {
        let cuda = CudaSrwma::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.srwma_batch_dev(slice_in, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(DeviceArrayF32Py { inner })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "srwma_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, period, device_id=0))]
pub fn srwma_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    data_tm_f32: PyReadonlyArray2<'_, f32>,
    period: usize,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    if period < 2 {
        return Err(PyValueError::new_err("period must be >= 2"));
    }

    let flat = data_tm_f32.as_slice()?;
    let shape = data_tm_f32.shape();
    let rows = shape[0];
    let cols = shape[1];
    let params = SrwmaParams {
        period: Some(period),
    };

    let inner = py.allow_threads(|| {
        let cuda = CudaSrwma::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.srwma_many_series_one_param_time_major_dev(flat, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(DeviceArrayF32Py { inner })
}

/// Helper function to compute SRWMA directly into output slice - zero allocations
#[inline]
pub fn srwma_into_slice(
    dst: &mut [f64],
    input: &SrwmaInput,
    kern: Kernel,
) -> Result<(), SrwmaError> {
    let data = match &input.data {
        SrwmaData::Slice(s) => *s,
        SrwmaData::Candles { candles, source } => source_type(candles, source),
    };

    if data.is_empty() {
        return Err(SrwmaError::EmptyInputData);
    }

    let period = input.params.period.unwrap_or(14);

    // Validate parameters
    if period == 0 {
        return Err(SrwmaError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }

    if data.len() < period {
        return Err(SrwmaError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }

    if dst.len() != data.len() {
        return Err(SrwmaError::InvalidPeriod {
            period: dst.len(),
            data_len: data.len(),
        });
    }

    // Find first valid (non-NaN) value
    let first = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => {
            // All values are NaN
            return Err(SrwmaError::AllValuesNaN);
        }
    };

    // Prepare weights
    let wlen = period - 1;
    let mut weights = Vec::with_capacity(wlen);
    let mut sumw = 0.0;
    for i in 0..wlen {
        let w = ((period - i) as f64).sqrt();
        weights.push(w);
        sumw += w;
    }
    let inv_norm = 1.0 / sumw;

    // Choose best kernel
    let chosen = match kern {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    // Compute directly into dst
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                srwma_scalar(data, &weights, period, first, inv_norm, dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                srwma_avx2(data, &weights, period, first, inv_norm, dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                srwma_avx512(data, &weights, period, first, inv_norm, dst)
            }
            _ => unreachable!(),
        }
    }

    // Fill warmup period with NaN
    let warmup_end = first + period + 1;
    for v in &mut dst[..warmup_end] {
        *v = f64::NAN;
    }

    Ok(())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn srwma_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = SrwmaParams {
        period: Some(period),
    };
    let input = SrwmaInput::from_slice(data, params);

    // Allocate output buffer once
    let mut output = vec![0.0; data.len()];

    // Compute directly into output buffer
    srwma_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct SrwmaBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct SrwmaBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<SrwmaParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = srwma_batch)]
pub fn srwma_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: SrwmaBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = SrwmaBatchRange {
        period: config.period_range,
    };

    let output = srwma_batch_inner(data, &sweep, detect_best_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let combos = expand_grid(&sweep);

    let js_output = SrwmaBatchJsOutput {
        values: output.values,
        combos,
        rows: output.rows,
        cols: output.cols,
    };

    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Failed to serialize output: {}", e)))
}

// Keep the old API for backwards compatibility
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn srwma_batch_js(
    data: &[f64],
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = SrwmaBatchRange {
        period: (period_start, period_end, period_step),
    };

    // Use batch function with parallel=false for WASM
    srwma_batch_inner(data, &sweep, detect_best_kernel(), false)
        .map(|output| output.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn srwma_batch_metadata_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<u32>, JsValue> {
    let sweep = SrwmaBatchRange {
        period: (period_start, period_end, period_step),
    };

    let combos = expand_grid(&sweep);
    Ok(combos.iter().map(|p| p.period.unwrap() as u32).collect())
}

// ================== Zero-Copy WASM Functions ==================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn srwma_alloc(len: usize) -> *mut f64 {
    // Allocate memory for input/output buffer
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec); // Prevent deallocation
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn srwma_free(ptr: *mut f64, len: usize) {
    // Free allocated memory
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn srwma_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    // Check for null pointers
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let params = SrwmaParams {
            period: Some(period),
        };
        let input = SrwmaInput::from_slice(data, params);

        // CRITICAL: Check for aliasing
        if in_ptr == out_ptr {
            // Handle in-place operation
            let mut temp = vec![0.0; len];
            srwma_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            // Direct computation
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            srwma_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

// ================== Optimized Batch Processing ==================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn srwma_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let sweep = SrwmaBatchRange {
            period: (period_start, period_end, period_step),
        };

        // Calculate output matrix dimensions
        let combos = expand_grid(&sweep);
        let rows = combos.len();
        let total_size = rows * len;

        // Get output slice
        let out = std::slice::from_raw_parts_mut(out_ptr, total_size);

        // Use batch_inner_into for direct computation
        srwma_batch_inner_into(data, &sweep, detect_best_kernel(), false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}
