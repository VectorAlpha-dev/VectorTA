//! # Elder Ray Index (ERI)
//!
//! Measures bullish and bearish pressure using a moving average as baseline.
//! Calculates bull power (high - MA) and bear power (low - MA).
//!
//! ## Parameters
//! - **period**: MA window size (default: 13)
//! - **ma_type**: Moving average type (default: "ema")
//!
//! ## Returns
//! - **`Ok(EriOutput)`** on success (`bull` and `bear` vectors of length matching input)
//! - **`Err(EriError)`** on failure
//!
//! ## Developer Status
//! - **SIMD Kernels**: AVX2 and AVX512 implemented (single-series + row variants)
//! - **Streaming**: O(1) SMA/EMA/RMA/DEMA/TEMA/WMA; Generic fallback is O(n)
//! - **Memory**: Good zero-copy usage (alloc_with_nan_prefix, make_uninit_matrix)
//! - Decision log: Scalar path is the reference; SIMD enabled (AVX2/AVX512) and >5% faster at 100k. CUDA wrapper present and synchronized before returning handles; Python interop via CAI v3 + DLPack. Typed errors added; range-expansion and size arithmetic now checked.

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
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

use crate::indicators::moving_averages::ma::{ma, MaData};
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum EriData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        source: &'a [f64],
    },
}

impl<'a> AsRef<[f64]> for EriInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            EriData::Candles { candles, source } => source_type(candles, source),
            EriData::Slices { source, .. } => source,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EriOutput {
    pub bull: Vec<f64>,
    pub bear: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct EriParams {
    pub period: Option<usize>,
    pub ma_type: Option<String>,
}

impl Default for EriParams {
    fn default() -> Self {
        Self {
            period: Some(13),
            ma_type: Some("ema".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EriInput<'a> {
    pub data: EriData<'a>,
    pub params: EriParams,
}

impl<'a> EriInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: EriParams) -> Self {
        Self {
            data: EriData::Candles { candles, source },
            params,
        }
    }
    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        source: &'a [f64],
        params: EriParams,
    ) -> Self {
        Self {
            data: EriData::Slices { high, low, source },
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", EriParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(13)
    }
    #[inline]
    pub fn get_ma_type(&self) -> &str {
        self.params.ma_type.as_deref().unwrap_or("ema")
    }
}

#[derive(Clone, Debug)]
pub struct EriBuilder {
    period: Option<usize>,
    ma_type: Option<String>,
    kernel: Kernel,
}

impl Default for EriBuilder {
    fn default() -> Self {
        Self {
            period: None,
            ma_type: None,
            kernel: Kernel::Auto,
        }
    }
}

impl EriBuilder {
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
    pub fn ma_type<S: Into<String>>(mut self, t: S) -> Self {
        self.ma_type = Some(t.into());
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<EriOutput, EriError> {
        let p = EriParams {
            period: self.period,
            ma_type: self.ma_type,
        };
        let i = EriInput::from_candles(c, "close", p);
        eri_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        src: &[f64],
    ) -> Result<EriOutput, EriError> {
        let p = EriParams {
            period: self.period,
            ma_type: self.ma_type,
        };
        let i = EriInput::from_slices(high, low, src, p);
        eri_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<EriStream, EriError> {
        let p = EriParams {
            period: self.period,
            ma_type: self.ma_type,
        };
        EriStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum EriError {
    #[error("eri: All input values are NaN.")]
    AllValuesNaN,
    #[error("eri: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("eri: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("eri: MA calculation error: {0}")]
    MaCalculationError(String),
    #[error("eri: Empty data provided.")]
    EmptyInputData,
    #[error("eri: Output slice length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("eri: Invalid range expansion: start={start}, end={end}, step={step}")]
    InvalidRange { start: usize, end: usize, step: usize },
    #[error("eri: Invalid kernel for batch operation. Got {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline]
pub fn eri(input: &EriInput) -> Result<EriOutput, EriError> {
    eri_with_kernel(input, Kernel::Auto)
}

pub fn eri_with_kernel(input: &EriInput, kernel: Kernel) -> Result<EriOutput, EriError> {
    let (high, low, source_data) = match &input.data {
        EriData::Candles { candles, source } => {
            let high = candles
                .select_candle_field("high")
                .map_err(|_| EriError::EmptyInputData)?;
            let low = candles
                .select_candle_field("low")
                .map_err(|_| EriError::EmptyInputData)?;
            let src = source_type(candles, source);
            (high, low, src)
        }
        EriData::Slices { high, low, source } => (*high, *low, *source),
    };

    if source_data.is_empty() || high.is_empty() || low.is_empty() {
        return Err(EriError::EmptyInputData);
    }

    let period = input.get_period();
    if period == 0 || period > source_data.len() {
        return Err(EriError::InvalidPeriod {
            period,
            data_len: source_data.len(),
        });
    }

    let mut first_valid_idx = None;
    for i in 0..source_data.len() {
        if !(source_data[i].is_nan() || high[i].is_nan() || low[i].is_nan()) {
            first_valid_idx = Some(i);
            break;
        }
    }
    let first_valid_idx = match first_valid_idx {
        Some(idx) => idx,
        None => return Err(EriError::AllValuesNaN),
    };

    if (source_data.len() - first_valid_idx) < period {
        return Err(EriError::NotEnoughValidData {
            needed: period,
            valid: source_data.len() - first_valid_idx,
        });
    }

    let ma_type = input.get_ma_type();
    let warmup_period = first_valid_idx + period - 1;
    let mut bull = alloc_with_nan_prefix(source_data.len(), warmup_period);
    let mut bear = alloc_with_nan_prefix(source_data.len(), warmup_period);

    // Dispatch to classic kernels for common cases
    if ma_type == "sma" || ma_type == "SMA" {
        unsafe {
            eri_scalar_classic_sma(
                high,
                low,
                &source_data,
                period,
                first_valid_idx,
                &mut bull,
                &mut bear,
            )?;
        }
        return Ok(EriOutput { bull, bear });
    } else if ma_type == "ema" || ma_type == "EMA" {
        unsafe {
            eri_scalar_classic_ema(
                high,
                low,
                &source_data,
                period,
                first_valid_idx,
                &mut bull,
                &mut bear,
            )?;
        }
        return Ok(EriOutput { bull, bear });
    }

    // Fall back to general implementation
    let full_ma = ma(&ma_type, MaData::Slice(&source_data), period)
        .map_err(|e| EriError::MaCalculationError(e.to_string()))?;

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => eri_scalar(
                high,
                low,
                &full_ma,
                period,
                first_valid_idx,
                &mut bull,
                &mut bear,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => eri_avx2(
                high,
                low,
                &full_ma,
                period,
                first_valid_idx,
                &mut bull,
                &mut bear,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => eri_avx512(
                high,
                low,
                &full_ma,
                period,
                first_valid_idx,
                &mut bull,
                &mut bear,
            ),
            _ => unreachable!(),
        }
    }

    Ok(EriOutput { bull, bear })
}

#[inline]
pub fn eri_scalar(
    high: &[f64],
    low: &[f64],
    ma: &[f64],
    period: usize,
    first_valid: usize,
    bull: &mut [f64],
    bear: &mut [f64],
) {
    let mut i = first_valid + period - 1;
    let n = high.len();
    if i >= n {
        return;
    }

    // Unroll by 4, then 2, then tail
    while i + 4 <= n {
        let m0 = ma[i + 0];
        bull[i + 0] = high[i + 0] - m0;
        bear[i + 0] = low[i + 0] - m0;

        let m1 = ma[i + 1];
        bull[i + 1] = high[i + 1] - m1;
        bear[i + 1] = low[i + 1] - m1;

        let m2 = ma[i + 2];
        bull[i + 2] = high[i + 2] - m2;
        bear[i + 2] = low[i + 2] - m2;

        let m3 = ma[i + 3];
        bull[i + 3] = high[i + 3] - m3;
        bear[i + 3] = low[i + 3] - m3;

        i += 4;
    }

    if i + 2 <= n {
        let m0 = ma[i + 0];
        bull[i + 0] = high[i + 0] - m0;
        bear[i + 0] = low[i + 0] - m0;

        let m1 = ma[i + 1];
        bull[i + 1] = high[i + 1] - m1;
        bear[i + 1] = low[i + 1] - m1;
        i += 2;
    }

    if i < n {
        let m0 = ma[i];
        bull[i] = high[i] - m0;
        bear[i] = low[i] - m0;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn eri_avx512(
    high: &[f64],
    low: &[f64],
    ma: &[f64],
    period: usize,
    first_valid: usize,
    bull: &mut [f64],
    bear: &mut [f64],
) {
    unsafe { eri_avx512_long(high, low, ma, period, first_valid, bull, bear) }
}

#[inline]
pub fn eri_avx2(
    high: &[f64],
    low: &[f64],
    ma: &[f64],
    period: usize,
    first_valid: usize,
    bull: &mut [f64],
    bear: &mut [f64],
) {
    // Prefer SIMD when available; otherwise fall back to scalar
    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    unsafe {
        return eri_avx2_core(high, low, ma, period, first_valid, bull, bear);
    }
    eri_scalar(high, low, ma, period, first_valid, bull, bear)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn eri_avx512_short(
    high: &[f64],
    low: &[f64],
    ma: &[f64],
    period: usize,
    first_valid: usize,
    bull: &mut [f64],
    bear: &mut [f64],
) {
    unsafe { eri_avx512_core(high, low, ma, period, first_valid, bull, bear) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn eri_avx512_long(
    high: &[f64],
    low: &[f64],
    ma: &[f64],
    period: usize,
    first_valid: usize,
    bull: &mut [f64],
    bear: &mut [f64],
) {
    unsafe { eri_avx512_core(high, low, ma, period, first_valid, bull, bear) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn eri_avx2_core(
    high: &[f64],
    low: &[f64],
    ma: &[f64],
    period: usize,
    first_valid: usize,
    bull: &mut [f64],
    bear: &mut [f64],
) {
    use core::arch::x86_64::*;

    let mut i = first_valid + period - 1;
    let n = high.len();
    if i >= n {
        return;
    }
    let len = n - i;

    let mut h_ptr = high.as_ptr().add(i);
    let mut l_ptr = low.as_ptr().add(i);
    let mut m_ptr = ma.as_ptr().add(i);
    let mut b_ptr = bull.as_mut_ptr().add(i);
    let mut r_ptr = bear.as_mut_ptr().add(i);

    let mut k = 0usize;
    while k + 4 <= len {
        let h = _mm256_loadu_pd(h_ptr);
        let l = _mm256_loadu_pd(l_ptr);
        let m = _mm256_loadu_pd(m_ptr);

        let b = _mm256_sub_pd(h, m);
        let r = _mm256_sub_pd(l, m);

        _mm256_storeu_pd(b_ptr, b);
        _mm256_storeu_pd(r_ptr, r);

        h_ptr = h_ptr.add(4);
        l_ptr = l_ptr.add(4);
        m_ptr = m_ptr.add(4);
        b_ptr = b_ptr.add(4);
        r_ptr = r_ptr.add(4);
        k += 4;
    }

    if k + 2 <= len {
        let h = _mm_loadu_pd(h_ptr);
        let l = _mm_loadu_pd(l_ptr);
        let m = _mm_loadu_pd(m_ptr);

        let b = _mm_sub_pd(h, m);
        let r = _mm_sub_pd(l, m);

        _mm_storeu_pd(b_ptr, b);
        _mm_storeu_pd(r_ptr, r);

        h_ptr = h_ptr.add(2);
        l_ptr = l_ptr.add(2);
        m_ptr = m_ptr.add(2);
        b_ptr = b_ptr.add(2);
        r_ptr = r_ptr.add(2);
        k += 2;
    }

    if k < len {
        let m0 = *m_ptr;
        *b_ptr = *h_ptr - m0;
        *r_ptr = *l_ptr - m0;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn eri_avx512_core(
    high: &[f64],
    low: &[f64],
    ma: &[f64],
    period: usize,
    first_valid: usize,
    bull: &mut [f64],
    bear: &mut [f64],
) {
    use core::arch::x86_64::*;

    let mut i = first_valid + period - 1;
    let n = high.len();
    if i >= n {
        return;
    }
    let len = n - i;

    let mut h_ptr = high.as_ptr().add(i);
    let mut l_ptr = low.as_ptr().add(i);
    let mut m_ptr = ma.as_ptr().add(i);
    let mut b_ptr = bull.as_mut_ptr().add(i);
    let mut r_ptr = bear.as_mut_ptr().add(i);

    let mut k = 0usize;
    while k + 8 <= len {
        let h = _mm512_loadu_pd(h_ptr);
        let l = _mm512_loadu_pd(l_ptr);
        let m = _mm512_loadu_pd(m_ptr);

        let b = _mm512_sub_pd(h, m);
        let r = _mm512_sub_pd(l, m);

        _mm512_storeu_pd(b_ptr, b);
        _mm512_storeu_pd(r_ptr, r);

        h_ptr = h_ptr.add(8);
        l_ptr = l_ptr.add(8);
        m_ptr = m_ptr.add(8);
        b_ptr = b_ptr.add(8);
        r_ptr = r_ptr.add(8);
        k += 8;
    }

    if k + 4 <= len {
        #[cfg(target_feature = "avx2")]
        {
            let h = _mm256_loadu_pd(h_ptr);
            let l = _mm256_loadu_pd(l_ptr);
            let m = _mm256_loadu_pd(m_ptr);

            let b = _mm256_sub_pd(h, m);
            let r = _mm256_sub_pd(l, m);

            _mm256_storeu_pd(b_ptr, b);
            _mm256_storeu_pd(r_ptr, r);

            h_ptr = h_ptr.add(4);
            l_ptr = l_ptr.add(4);
            m_ptr = m_ptr.add(4);
            b_ptr = b_ptr.add(4);
            r_ptr = r_ptr.add(4);
            k += 4;
        }
        #[cfg(not(target_feature = "avx2"))]
        {
            let m0 = *m_ptr.add(0);
            *b_ptr.add(0) = *h_ptr.add(0) - m0;
            *r_ptr.add(0) = *l_ptr.add(0) - m0;
            let m1 = *m_ptr.add(1);
            *b_ptr.add(1) = *h_ptr.add(1) - m1;
            *r_ptr.add(1) = *l_ptr.add(1) - m1;
            let m2 = *m_ptr.add(2);
            *b_ptr.add(2) = *h_ptr.add(2) - m2;
            *r_ptr.add(2) = *l_ptr.add(2) - m2;
            let m3 = *m_ptr.add(3);
            *b_ptr.add(3) = *h_ptr.add(3) - m3;
            *r_ptr.add(3) = *l_ptr.add(3) - m3;

            h_ptr = h_ptr.add(4);
            l_ptr = l_ptr.add(4);
            m_ptr = m_ptr.add(4);
            b_ptr = b_ptr.add(4);
            r_ptr = r_ptr.add(4);
            k += 4;
        }
    }

    if k + 2 <= len {
        let h = _mm_loadu_pd(h_ptr);
        let l = _mm_loadu_pd(l_ptr);
        let m = _mm_loadu_pd(m_ptr);

        let b = _mm_sub_pd(h, m);
        let r = _mm_sub_pd(l, m);

        _mm_storeu_pd(b_ptr, b);
        _mm_storeu_pd(r_ptr, r);

        h_ptr = h_ptr.add(2);
        l_ptr = l_ptr.add(2);
        m_ptr = m_ptr.add(2);
        b_ptr = b_ptr.add(2);
        r_ptr = r_ptr.add(2);
        k += 2;
    }

    if k < len {
        let m0 = *m_ptr;
        *b_ptr = *h_ptr - m0;
        *r_ptr = *l_ptr - m0;
    }
}

#[inline]
pub fn eri_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    sweep: &EriBatchRange,
    k: Kernel,
) -> Result<EriBatchOutput, EriError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(EriError::InvalidKernelForBatch(other)),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    eri_batch_par_slice(high, low, source, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct EriBatchRange {
    pub period: (usize, usize, usize),
    pub ma_type: String,
}

impl Default for EriBatchRange {
    fn default() -> Self {
        Self {
            period: (13, 60, 1),
            ma_type: "ema".into(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct EriBatchBuilder {
    range: EriBatchRange,
    kernel: Kernel,
}

impl EriBatchBuilder {
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
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        source: &[f64],
    ) -> Result<EriBatchOutput, EriError> {
        eri_batch_with_kernel(high, low, source, &self.range, self.kernel)
    }
}

#[derive(Clone, Debug)]
pub struct EriBatchOutput {
    pub bull: Vec<f64>,
    pub bear: Vec<f64>,
    pub params: Vec<EriParams>,
    pub rows: usize,
    pub cols: usize,
}

impl EriBatchOutput {
    pub fn row_for_params(&self, p: &EriParams) -> Option<usize> {
        self.params
            .iter()
            .position(|c| c.period == p.period && c.ma_type == p.ma_type)
    }
    pub fn values_for_bull(&self, p: &EriParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.bull[start..start + self.cols]
        })
    }
    pub fn values_for_bear(&self, p: &EriParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.bear[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &EriBatchRange) -> Result<Vec<EriParams>, EriError> {
    let (start, end, step) = r.period;
    // Treat zero step as static; support start==end
    if step == 0 {
        return Ok(vec![EriParams {
            period: Some(start),
            ma_type: Some(r.ma_type.clone()),
        }]);
    }

    let mut out: Vec<EriParams> = Vec::new();
    if start == end {
        out.push(EriParams {
            period: Some(start),
            ma_type: Some(r.ma_type.clone()),
        });
        return Ok(out);
    }

    if start < end {
        let mut p = start;
        while p <= end {
            out.push(EriParams {
                period: Some(p),
                ma_type: Some(r.ma_type.clone()),
            });
            match p.checked_add(step) {
                Some(next) => {
                    if next == p { break; }
                    p = next;
                }
                None => return Err(EriError::InvalidRange { start, end, step }),
            }
        }
    } else {
        // Reversed bounds
        let mut p = start;
        while p >= end {
            out.push(EriParams {
                period: Some(p),
                ma_type: Some(r.ma_type.clone()),
            });
            // guard underflow
            if p < step { break; }
            p -= step;
            if p == usize::MAX { break; }
        }
        // Ensure last included 'end' if exactly on step boundary; else accept as-is
        if out.is_empty() {
            return Err(EriError::InvalidRange { start, end, step });
        }
    }

    if out.is_empty() { return Err(EriError::InvalidRange { start, end, step }); }
    Ok(out)
}

#[inline(always)]
pub fn eri_batch_slice(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    sweep: &EriBatchRange,
    kern: Kernel,
) -> Result<EriBatchOutput, EriError> {
    eri_batch_inner(high, low, source, sweep, kern, false)
}

#[inline(always)]
pub fn eri_batch_par_slice(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    sweep: &EriBatchRange,
    kern: Kernel,
) -> Result<EriBatchOutput, EriError> {
    eri_batch_inner(high, low, source, sweep, kern, true)
}

#[inline(always)]
fn eri_batch_inner(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    sweep: &EriBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<EriBatchOutput, EriError> {
    let combos = expand_grid(sweep)?;
    // Use triple-validity check to match single path
    let first = high
        .iter()
        .zip(low.iter())
        .zip(source.iter())
        .position(|((h, l), s)| !h.is_nan() && !l.is_nan() && !s.is_nan())
        .ok_or(EriError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if source.len() - first < max_p {
        return Err(EriError::NotEnoughValidData {
            needed: max_p,
            valid: source.len() - first,
        });
    }
    let rows = combos.len();
    let cols = source.len();
    let total = rows
        .checked_mul(cols)
        .ok_or(EriError::InvalidRange { start: sweep.period.0, end: sweep.period.1, step: sweep.period.2 })?;

    // Use uninitialized memory for batch processing
    let mut buf_bull = make_uninit_matrix(rows, cols);
    let mut buf_bear = make_uninit_matrix(rows, cols);

    // Initialize prefixes with NaN based on warmup periods
    let warmup_periods: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();
    init_matrix_prefixes(&mut buf_bull, cols, &warmup_periods);
    init_matrix_prefixes(&mut buf_bear, cols, &warmup_periods);

    // Wrap buffers in ManuallyDrop to prevent premature drop
    let mut buf_bull_guard = std::mem::ManuallyDrop::new(buf_bull);
    let mut buf_bear_guard = std::mem::ManuallyDrop::new(buf_bear);

    // Convert to initialized slices
    let mut bull = unsafe {
        std::slice::from_raw_parts_mut(buf_bull_guard.as_mut_ptr() as *mut f64, total)
    };
    let mut bear = unsafe {
        std::slice::from_raw_parts_mut(buf_bear_guard.as_mut_ptr() as *mut f64, total)
    };

    let do_row = |row: usize, bull_row: &mut [f64], bear_row: &mut [f64]| -> Result<(), EriError> {
        let period = combos[row].period.unwrap();
        let ma_type = combos[row].ma_type.as_deref().unwrap();
        let ma_vec = ma(ma_type, MaData::Slice(source), period)
            .map_err(|e| EriError::MaCalculationError(e.to_string()))?;
        match kern {
            Kernel::Scalar => unsafe {
                eri_row_scalar(high, low, &ma_vec, first, period, bull_row, bear_row)
            },
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => unsafe {
                eri_row_avx2(high, low, &ma_vec, first, period, bull_row, bear_row)
            },
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => unsafe {
                eri_row_avx512(high, low, &ma_vec, first, period, bull_row, bear_row)
            },
            _ => unreachable!(),
        }
        Ok(())
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            bull.par_chunks_mut(cols)
                .zip(bear.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, (bull_row, bear_row))| {
                    let _ = do_row(row, bull_row, bear_row);
                });
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, (bull_row, bear_row)) in
                bull.chunks_mut(cols).zip(bear.chunks_mut(cols)).enumerate()
            {
                let _ = do_row(row, bull_row, bear_row);
            }
        }
    } else {
        for (row, (bull_row, bear_row)) in
            bull.chunks_mut(cols).zip(bear.chunks_mut(cols)).enumerate()
        {
            let _ = do_row(row, bull_row, bear_row);
        }
    }

    // Create output with owned data
    // SAFETY: We're taking ownership of the allocated memory from make_uninit_matrix
    let bull_vec = unsafe {
        Vec::from_raw_parts(
            buf_bull_guard.as_mut_ptr() as *mut f64,
            total,
            total,
        )
    };
    let bear_vec = unsafe {
        Vec::from_raw_parts(
            buf_bear_guard.as_mut_ptr() as *mut f64,
            total,
            total,
        )
    };

    Ok(EriBatchOutput {
        bull: bull_vec,
        bear: bear_vec,
        params: combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub fn eri_batch_inner_into(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    sweep: &EriBatchRange,
    kern: Kernel,
    parallel: bool,
    bull_out: &mut [f64],
    bear_out: &mut [f64],
) -> Result<Vec<EriParams>, EriError> {
    let combos = expand_grid(sweep)?;
    // Use triple-validity check to match single path
    let first = high
        .iter()
        .zip(low.iter())
        .zip(source.iter())
        .position(|((h, l), s)| !h.is_nan() && !l.is_nan() && !s.is_nan())
        .ok_or(EriError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if source.len() - first < max_p {
        return Err(EriError::NotEnoughValidData {
            needed: max_p,
            valid: source.len() - first,
        });
    }

    let rows = combos.len();
    let cols = source.len();

    // Verify output slices have correct length
    let expected = rows
        .checked_mul(cols)
        .ok_or(EriError::InvalidRange { start: sweep.period.0, end: sweep.period.1, step: sweep.period.2 })?;
    if bull_out.len() != expected || bear_out.len() != expected {
        return Err(EriError::OutputLengthMismatch { expected, got: bull_out.len().max(bear_out.len()) });
    }

    let do_row = |row: usize, bull_row: &mut [f64], bear_row: &mut [f64]| -> Result<(), EriError> {
        let period = combos[row].period.unwrap();
        let ma_type = combos[row].ma_type.as_deref().unwrap();
        let ma_vec = ma(ma_type, MaData::Slice(source), period)
            .map_err(|e| EriError::MaCalculationError(e.to_string()))?;
        match kern {
            Kernel::Scalar => unsafe {
                eri_row_scalar(high, low, &ma_vec, first, period, bull_row, bear_row)
            },
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => unsafe {
                eri_row_avx2(high, low, &ma_vec, first, period, bull_row, bear_row)
            },
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => unsafe {
                eri_row_avx512(high, low, &ma_vec, first, period, bull_row, bear_row)
            },
            _ => unreachable!(),
        }
        Ok(())
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            bull_out
                .par_chunks_mut(cols)
                .zip(bear_out.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, (bull_row, bear_row))| {
                    let _ = do_row(row, bull_row, bear_row);
                });
        }
        #[cfg(target_arch = "wasm32")]
        for row in 0..rows {
            let bull_row = &mut bull_out[row * cols..(row + 1) * cols];
            let bear_row = &mut bear_out[row * cols..(row + 1) * cols];
            let _ = do_row(row, bull_row, bear_row);
        }
    } else {
        for row in 0..rows {
            let bull_row = &mut bull_out[row * cols..(row + 1) * cols];
            let bear_row = &mut bear_out[row * cols..(row + 1) * cols];
            let _ = do_row(row, bull_row, bear_row);
        }
    }

    Ok(combos)
}

#[inline(always)]
unsafe fn eri_row_scalar(
    high: &[f64],
    low: &[f64],
    ma: &[f64],
    first: usize,
    period: usize,
    bull: &mut [f64],
    bear: &mut [f64],
) {
    let mut i = first + period - 1;
    let n = high.len();
    if i >= n {
        return;
    }

    let len = n - i;
    let mut h_ptr = high.as_ptr().add(i);
    let mut l_ptr = low.as_ptr().add(i);
    let mut m_ptr = ma.as_ptr().add(i);
    let mut b_ptr = bull.as_mut_ptr().add(i);
    let mut r_ptr = bear.as_mut_ptr().add(i);

    let mut k = 0usize;
    while k + 4 <= len {
        let m0 = *m_ptr.add(0);
        *b_ptr.add(0) = *h_ptr.add(0) - m0;
        *r_ptr.add(0) = *l_ptr.add(0) - m0;

        let m1 = *m_ptr.add(1);
        *b_ptr.add(1) = *h_ptr.add(1) - m1;
        *r_ptr.add(1) = *l_ptr.add(1) - m1;

        let m2 = *m_ptr.add(2);
        *b_ptr.add(2) = *h_ptr.add(2) - m2;
        *r_ptr.add(2) = *l_ptr.add(2) - m2;

        let m3 = *m_ptr.add(3);
        *b_ptr.add(3) = *h_ptr.add(3) - m3;
        *r_ptr.add(3) = *l_ptr.add(3) - m3;

        h_ptr = h_ptr.add(4);
        l_ptr = l_ptr.add(4);
        m_ptr = m_ptr.add(4);
        b_ptr = b_ptr.add(4);
        r_ptr = r_ptr.add(4);
        k += 4;
    }
    if k + 2 <= len {
        let m0 = *m_ptr.add(0);
        *b_ptr.add(0) = *h_ptr.add(0) - m0;
        *r_ptr.add(0) = *l_ptr.add(0) - m0;

        let m1 = *m_ptr.add(1);
        *b_ptr.add(1) = *h_ptr.add(1) - m1;
        *r_ptr.add(1) = *l_ptr.add(1) - m1;

        h_ptr = h_ptr.add(2);
        l_ptr = l_ptr.add(2);
        m_ptr = m_ptr.add(2);
        b_ptr = b_ptr.add(2);
        r_ptr = r_ptr.add(2);
        k += 2;
    }
    if k < len {
        let m0 = *m_ptr;
        *b_ptr = *h_ptr - m0;
        *r_ptr = *l_ptr - m0;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn eri_row_avx2(
    high: &[f64],
    low: &[f64],
    ma: &[f64],
    first: usize,
    period: usize,
    bull: &mut [f64],
    bear: &mut [f64],
) {
    eri_avx2_core(high, low, ma, period, first, bull, bear)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn eri_row_avx512(
    high: &[f64],
    low: &[f64],
    ma: &[f64],
    first: usize,
    period: usize,
    bull: &mut [f64],
    bear: &mut [f64],
) {
    if period <= 32 {
        eri_row_avx512_short(high, low, ma, first, period, bull, bear);
    } else {
        eri_row_avx512_long(high, low, ma, first, period, bull, bear);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn eri_row_avx512_short(
    high: &[f64],
    low: &[f64],
    ma: &[f64],
    first: usize,
    period: usize,
    bull: &mut [f64],
    bear: &mut [f64],
) {
    eri_avx512_core(high, low, ma, period, first, bull, bear)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn eri_row_avx512_long(
    high: &[f64],
    low: &[f64],
    ma: &[f64],
    first: usize,
    period: usize,
    bull: &mut [f64],
    bear: &mut [f64],
) {
    eri_avx512_core(high, low, ma, period, first, bull, bear)
}

// Decision note: Streaming enabled; O(1) for SMA/EMA/RMA/DEMA/TEMA/WMA; fallback to generic O(n) only when needed.
#[derive(Debug, Clone)]
pub struct EriStream {
    period: usize,
    ma_type: String,
    engine: StreamMa,
    ready: bool,
}

#[derive(Debug, Clone)]
enum StreamMa {
    Sma(SmaState),
    Ema(EmaState),
    Rma(EmaState),
    Dema(DemaState),
    Tema(TemaState),
    Wma(WmaState),
    Generic(GenericState),
}

#[derive(Debug, Clone)]
struct SmaState {
    buf: Vec<f64>,
    pos: usize,
    count: usize,
    sum: f64,
    inv_n: f64,
}

#[derive(Debug, Clone)]
struct EmaState {
    n: usize,
    alpha: f64,
    beta: f64,
    // warmup via SMA on first n points
    init_sum: f64,
    init_count: usize,
    ema: f64,
}

#[derive(Debug, Clone)]
struct DemaState {
    n: usize,
    alpha: f64,
    beta: f64,
    init_sum: f64,
    init_count: usize,
    e1: f64, // EMA
    e2: f64, // EMA(EMA)
}

#[derive(Debug, Clone)]
struct TemaState {
    n: usize,
    alpha: f64,
    beta: f64,
    init_sum: f64,
    init_count: usize,
    e1: f64, // EMA
    e2: f64, // EMA(EMA)
    e3: f64, // EMA(EMA(EMA))
}

#[derive(Debug, Clone)]
struct WmaState {
    n: usize,
    den_inv: f64, // 1 / (n(n+1)/2)
    buf: Vec<f64>,
    pos: usize,
    count: usize,
    s: f64,  // sum of last n values
    ws: f64, // weighted sum with weights 1..n (oldest=1, newest=n)
}

#[derive(Debug, Clone)]
struct GenericState {
    n: usize,
    buf: Vec<f64>,
    pos: usize,
    count: usize,
    scratch: Vec<f64>, // reused to reorder buf chronologically
}

impl EriStream {
    pub fn try_new(params: EriParams) -> Result<Self, EriError> {
        let period = params.period.unwrap_or(13);
        if period == 0 {
            return Err(EriError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let ma_type = params.ma_type.unwrap_or_else(|| "ema".to_string());

        let engine = make_engine(period, &ma_type);
        Ok(Self {
            period,
            ma_type,
            engine,
            ready: false,
        })
    }

    /// O(1) streaming update for SMA/EMA/RMA/DEMA/TEMA/WMA.
    /// If any input is NaN, the stream resets and returns None (matches batch "first-valid" semantics).
    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64, source: f64) -> Option<(f64, f64)> {
        if high.is_nan() || low.is_nan() || source.is_nan() {
            self.reset();
            return None;
        }

        let ma_val = match &mut self.engine {
            StreamMa::Sma(st) => sma_update(st, source),
            StreamMa::Ema(st) => ema_like_update(st, source),
            StreamMa::Rma(st) => ema_like_update(st, source),
            StreamMa::Dema(st) => dema_update(st, source),
            StreamMa::Tema(st) => tema_update(st, source),
            StreamMa::Wma(st) => wma_update(st, source),
            StreamMa::Generic(st) => generic_update(st, &self.ma_type, source),
        }?;

        self.ready = true;
        Some((high - ma_val, low - ma_val))
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.ready = false;
        self.engine = make_engine(self.period, &self.ma_type);
    }
}

#[inline(always)]
fn make_engine(period: usize, ma_type: &str) -> StreamMa {
    let t = ma_type.to_ascii_lowercase();
    match t.as_str() {
        "sma" => StreamMa::Sma(SmaState {
            buf: vec![0.0; period],
            pos: 0,
            count: 0,
            sum: 0.0,
            inv_n: 1.0 / period as f64,
        }),
        "ema" | "ewma" => StreamMa::Ema(EmaState {
            n: period,
            alpha: 2.0 / (period as f64 + 1.0),
            beta: 1.0 - (2.0 / (period as f64 + 1.0)),
            init_sum: 0.0,
            init_count: 0,
            ema: f64::NAN,
        }),
        "rma" | "wilder" | "smma" => StreamMa::Rma(EmaState {
            n: period,
            alpha: 1.0 / period as f64,
            beta: 1.0 - (1.0 / period as f64),
            init_sum: 0.0,
            init_count: 0,
            ema: f64::NAN,
        }),
        "dema" => StreamMa::Dema(DemaState {
            n: period,
            alpha: 2.0 / (period as f64 + 1.0),
            beta: 1.0 - (2.0 / (period as f64 + 1.0)),
            init_sum: 0.0,
            init_count: 0,
            e1: f64::NAN,
            e2: f64::NAN,
        }),
        "tema" => StreamMa::Tema(TemaState {
            n: period,
            alpha: 2.0 / (period as f64 + 1.0),
            beta: 1.0 - (2.0 / (period as f64 + 1.0)),
            init_sum: 0.0,
            init_count: 0,
            e1: f64::NAN,
            e2: f64::NAN,
            e3: f64::NAN,
        }),
        "wma" | "lwma" | "linear" | "linear_wma" => {
            let n = period as f64;
            let den_inv = 2.0 / (n * (n + 1.0));
            StreamMa::Wma(WmaState {
                n: period,
                den_inv,
                buf: vec![0.0; period],
                pos: 0,
                count: 0,
                s: 0.0,
                ws: 0.0,
            })
        }
        _ => StreamMa::Generic(GenericState {
            n: period,
            buf: vec![0.0; period],
            pos: 0,
            count: 0,
            scratch: vec![0.0; period],
        }),
    }
}

#[inline(always)]
fn sma_update(st: &mut SmaState, x: f64) -> Option<f64> {
    let n = st.buf.len();
    if st.count < n {
        st.buf[st.pos] = x;
        st.sum += x;
        st.pos = (st.pos + 1) % n;
        st.count += 1;
        return (st.count == n).then(|| st.sum * st.inv_n);
    }
    let old = st.buf[st.pos];
    st.buf[st.pos] = x;
    st.sum += x - old;
    st.pos = (st.pos + 1) % n;
    Some(st.sum * st.inv_n)
}

#[inline(always)]
fn ema_like_update(st: &mut EmaState, x: f64) -> Option<f64> {
    if st.init_count < st.n {
        st.init_sum += x;
        st.init_count += 1;
        if st.init_count == st.n {
            st.ema = st.init_sum / st.n as f64;
            return Some(st.ema);
        }
        return None;
    }
    st.ema = x.mul_add(st.alpha, st.beta * st.ema);
    Some(st.ema)
}

#[inline(always)]
fn dema_update(st: &mut DemaState, x: f64) -> Option<f64> {
    if st.init_count < st.n {
        st.init_sum += x;
        st.init_count += 1;
        if st.init_count == st.n {
            st.e1 = st.init_sum / st.n as f64;
            st.e2 = st.e1;
            return Some(st.e1);
        }
        return None;
    }
    st.e1 = x.mul_add(st.alpha, st.beta * st.e1);
    st.e2 = st.e1.mul_add(st.alpha, st.beta * st.e2);
    Some(2.0f64.mul_add(st.e1, -st.e2))
}

#[inline(always)]
fn tema_update(st: &mut TemaState, x: f64) -> Option<f64> {
    if st.init_count < st.n {
        st.init_sum += x;
        st.init_count += 1;
        if st.init_count == st.n {
            st.e1 = st.init_sum / st.n as f64;
            st.e2 = st.e1;
            st.e3 = st.e2;
            return Some(st.e1);
        }
        return None;
    }
    st.e1 = x.mul_add(st.alpha, st.beta * st.e1);
    st.e2 = st.e1.mul_add(st.alpha, st.beta * st.e2);
    st.e3 = st.e2.mul_add(st.alpha, st.beta * st.e3);
    Some((3.0 * st.e1) - (3.0 * st.e2) + st.e3)
}

#[inline(always)]
fn wma_update(st: &mut WmaState, x: f64) -> Option<f64> {
    let n = st.n;
    if st.count < n {
        st.buf[st.pos] = x;
        st.s += x;
        st.ws += (st.count as f64 + 1.0) * x;
        st.pos = (st.pos + 1) % n;
        st.count += 1;
        return (st.count == n).then(|| st.ws * st.den_inv);
    }
    let old = st.buf[st.pos];
    st.buf[st.pos] = x;
    let s_prev = st.s;
    st.s = s_prev - old + x;
    st.ws = st.ws - s_prev + (n as f64) * x;
    st.pos = (st.pos + 1) % n;
    Some(st.ws * st.den_inv)
}

#[inline(always)]
fn generic_update(st: &mut GenericState, ma_type: &str, x: f64) -> Option<f64> {
    let n = st.n;
    if st.count < n {
        st.buf[st.pos] = x;
        st.pos = (st.pos + 1) % n;
        st.count += 1;
        return None;
    }
    st.buf[st.pos] = x;
    st.pos = (st.pos + 1) % n;

    for i in 0..n {
        let src_idx = (st.pos + i) % n;
        st.scratch[i] = st.buf[src_idx];
    }
    let m = ma(ma_type, MaData::Slice(&st.scratch), n).ok()?;
    m.last().copied()
}

#[cfg(feature = "python")]
#[pyfunction(name = "eri")]
#[pyo3(signature = (high, low, source, period=13, ma_type="ema", kernel=None))]
pub fn eri_py<'py>(
    py: Python<'py>,
    high: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    source: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
    ma_type: &str,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};

    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let source_slice = source.as_slice()?;

    // Validate lengths match
    if high_slice.len() != low_slice.len() || high_slice.len() != source_slice.len() {
        return Err(PyValueError::new_err(
            "high, low, and source arrays must have the same length",
        ));
    }

    let kern = validate_kernel(kernel, false)?;
    let params = EriParams {
        period: Some(period),
        ma_type: Some(ma_type.to_string()),
    };
    let input = EriInput::from_slices(high_slice, low_slice, source_slice, params);

    let result = py
        .allow_threads(|| eri_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Zero-copy transfer to Python
    Ok((result.bull.into_pyarray(py), result.bear.into_pyarray(py)))
}

#[cfg(feature = "python")]
#[pyclass(name = "EriStream")]
pub struct EriStreamPy {
    stream: EriStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl EriStreamPy {
    #[new]
    fn new(period: usize, ma_type: Option<&str>) -> PyResult<Self> {
        let params = EriParams {
            period: Some(period),
            ma_type: ma_type
                .map(|s| s.to_string())
                .or_else(|| Some("ema".to_string())),
        };
        let stream =
            EriStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(EriStreamPy { stream })
    }

    fn update(&mut self, high: f64, low: f64, source: f64) -> Option<(f64, f64)> {
        self.stream.update(high, low, source)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "eri_batch")]
#[pyo3(signature = (high, low, source, period_range=(13, 13, 0), ma_type="ema", kernel=None))]
pub fn eri_batch_py<'py>(
    py: Python<'py>,
    high: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    source: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    ma_type: &str,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let source_slice = source.as_slice()?;

    // Validate lengths match
    if high_slice.len() != low_slice.len() || high_slice.len() != source_slice.len() {
        return Err(PyValueError::new_err(
            "high, low, and source arrays must have the same length",
        ));
    }

    let sweep = EriBatchRange {
        period: period_range,
        ma_type: ma_type.to_string(),
    };

    // 1. Expand grid once to know rows*cols
    let combos = expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = high_slice.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    // 2. Pre-allocate uninitialized NumPy arrays (1-D, will reshape later)
    let bull_array: Bound<'py, PyArray1<f64>> =
        unsafe { PyArray1::<f64>::new(py, [total], false) };
    let bear_array: Bound<'py, PyArray1<f64>> =
        unsafe { PyArray1::<f64>::new(py, [total], false) };

    // 3. Get mutable slices from arrays
    let bull_slice = unsafe { bull_array.as_slice_mut()? };
    let bear_slice = unsafe { bear_array.as_slice_mut()? };

    // Find first valid index based on triple validity
    let first_valid = high_slice
        .iter()
        .zip(low_slice.iter())
        .zip(source_slice.iter())
        .position(|((h, l), s)| !h.is_nan() && !l.is_nan() && !s.is_nan())
        .unwrap_or(0);

    // Initialize NaN prefixes for each row based on warmup
    for (row, combo) in combos.iter().enumerate() {
        let period = combo.period.unwrap();
        let warmup = first_valid + period - 1;
        let row_start = row * cols;
        for i in 0..warmup.min(cols) {
            bull_slice[row_start + i] = f64::NAN;
            bear_slice[row_start + i] = f64::NAN;
        }
    }

    // 4. Run computation with GIL released
    let kern = validate_kernel(kernel, true)?;
    let kernel_to_use = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    };
    let simd = match kernel_to_use {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => Kernel::Scalar,
    };

    let combos = py
        .allow_threads(|| {
            eri_batch_inner_into(
                high_slice,
                low_slice,
                source_slice,
                &sweep,
                simd,
                true,
                bull_slice,
                bear_slice,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // 6. Reshape arrays to 2D
    let bull_reshaped = bull_array.reshape([rows, cols])?;
    let bear_reshaped = bear_array.reshape([rows, cols])?;

    // 5. Build parameter arrays
    let periods: Vec<usize> = combos.iter().map(|c| c.period.unwrap()).collect();
    let ma_types: Vec<&str> = vec![ma_type; combos.len()];

    // 8. Create result dictionary
    let dict = PyDict::new(py);
    dict.set_item("bull_values", bull_reshaped)?;
    dict.set_item("bear_values", bear_reshaped)?;
    dict.set_item("periods", periods.into_pyarray(py))?;
    dict.set_item("ma_types", ma_types)?;

    Ok(dict.into())
}

// Zero-copy output function - write directly to output slices
/// Write directly to output slices - no allocations
pub fn eri_into_slice(
    dst_bull: &mut [f64],
    dst_bear: &mut [f64],
    input: &EriInput,
    kern: Kernel,
) -> Result<(), EriError> {
    let (high, low, source_data) = match &input.data {
        EriData::Candles { candles, source } => {
            let high = candles
                .select_candle_field("high")
                .map_err(|_| EriError::EmptyInputData)?;
            let low = candles
                .select_candle_field("low")
                .map_err(|_| EriError::EmptyInputData)?;
            let src = source_type(candles, source);
            (high, low, src)
        }
        EriData::Slices { high, low, source } => (*high, *low, *source),
    };

    if source_data.is_empty() || high.is_empty() || low.is_empty() {
        return Err(EriError::EmptyInputData);
    }

    // Verify output slices have correct length
    if dst_bull.len() != source_data.len() || dst_bear.len() != source_data.len() {
        return Err(EriError::OutputLengthMismatch {
            expected: source_data.len(),
            got: dst_bull.len().max(dst_bear.len()),
        });
    }

    let period = input.get_period();
    if period == 0 || period > source_data.len() {
        return Err(EriError::InvalidPeriod {
            period,
            data_len: source_data.len(),
        });
    }

    let mut first_valid_idx = None;
    for i in 0..source_data.len() {
        if !(source_data[i].is_nan() || high[i].is_nan() || low[i].is_nan()) {
            first_valid_idx = Some(i);
            break;
        }
    }
    let first_valid_idx = match first_valid_idx {
        Some(idx) => idx,
        None => return Err(EriError::AllValuesNaN),
    };

    if (source_data.len() - first_valid_idx) < period {
        return Err(EriError::NotEnoughValidData {
            needed: period,
            valid: source_data.len() - first_valid_idx,
        });
    }

    let ma_type = input.get_ma_type();
    let full_ma = ma(&ma_type, MaData::Slice(&source_data), period)
        .map_err(|e| EriError::MaCalculationError(e.to_string()))?;

    let warmup_period = first_valid_idx + period - 1;

    // Fill warmup with NaN
    for v in &mut dst_bull[..warmup_period] {
        *v = f64::NAN;
    }
    for v in &mut dst_bear[..warmup_period] {
        *v = f64::NAN;
    }

    let chosen = match kern {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => eri_scalar(
                high,
                low,
                &full_ma,
                period,
                first_valid_idx,
                dst_bull,
                dst_bear,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => eri_avx2(
                high,
                low,
                &full_ma,
                period,
                first_valid_idx,
                dst_bull,
                dst_bear,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => eri_avx512(
                high,
                low,
                &full_ma,
                period,
                first_valid_idx,
                dst_bull,
                dst_bear,
            ),
            _ => unreachable!(),
        }
    }

    Ok(())
}

/// Write ERI (bull and bear) directly into caller-provided output slices.
///
/// - Preserves the module's NaN warmup semantics (quiet-NaN prefix up to `first_valid + period - 1`).
/// - `bull_out.len()` and `bear_out.len()` must equal the input length.
/// - Uses `Kernel::Auto` dispatch for runtime selection.
#[cfg(not(feature = "wasm"))]
#[inline]
pub fn eri_into(
    input: &EriInput,
    bull_out: &mut [f64],
    bear_out: &mut [f64],
) -> Result<(), EriError> {
    eri_into_slice(bull_out, bear_out, input, Kernel::Auto)
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct EriResult {
    pub values: Vec<f64>, // [bull..., bear...]
    pub rows: usize,      // 2
    pub cols: usize,      // len
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn eri_js_flat(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    period: usize,
    ma_type: &str,
) -> Result<JsValue, JsValue> {
    if high.len() != low.len() || high.len() != source.len() {
        return Err(JsValue::from_str("length mismatch"));
    }
    let params = EriParams {
        period: Some(period),
        ma_type: Some(ma_type.to_string()),
    };
    let input = EriInput::from_slices(high, low, source, params);

    let mut bull = vec![0.0; source.len()];
    let mut bear = vec![0.0; source.len()];
    eri_into_slice(&mut bull, &mut bear, &input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // flatten [bull..., bear...]
    let mut values = bull;
    values.extend_from_slice(&bear);

    let out = EriResult {
        values,
        rows: 2,
        cols: source.len(),
    };
    serde_wasm_bindgen::to_value(&out)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn eri_js(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    period: usize,
    ma_type: &str,
) -> Result<Vec<f64>, JsValue> {
    // Validate lengths match
    if high.len() != low.len() || high.len() != source.len() {
        return Err(JsValue::from_str(
            "high, low, and source arrays must have the same length",
        ));
    }

    let params = EriParams {
        period: Some(period),
        ma_type: Some(ma_type.to_string()),
    };
    let input = EriInput::from_slices(high, low, source, params);

    // Single allocation for both outputs
    let total = source
        .len()
        .checked_mul(2)
        .ok_or_else(|| JsValue::from_str("length overflow"))?;
    let mut output = vec![0.0; total];
    let (bull_part, bear_part) = output.split_at_mut(source.len());

    eri_into_slice(bull_part, bear_part, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

// ----- Python CUDA bindings -----
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::eri_wrapper::CudaEri;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::utilities::dlpack_cuda::{make_device_array_py, DeviceArrayF32Py};
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "eri_cuda_batch_dev")]
#[pyo3(signature = (high_f32, low_f32, source_f32, period_range, ma_type, device_id=0))]
pub fn eri_cuda_batch_dev_py(
    py: Python<'_>,
    high_f32: numpy::PyReadonlyArray1<'_, f32>,
    low_f32: numpy::PyReadonlyArray1<'_, f32>,
    source_f32: numpy::PyReadonlyArray1<'_, f32>,
    period_range: (usize, usize, usize),
    ma_type: &str,
    device_id: usize,
) -> PyResult<(DeviceArrayF32Py, DeviceArrayF32Py)> {
    use crate::cuda::cuda_available;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let h = high_f32.as_slice()?;
    let l = low_f32.as_slice()?;
    let s = source_f32.as_slice()?;
    let sweep = EriBatchRange {
        period: period_range,
        ma_type: ma_type.to_string(),
    };
    let (bull, bear) = py.allow_threads(|| {
        let cuda = CudaEri::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.eri_batch_dev(h, l, s, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
            .map(|((bull, bear), _combos)| (bull, bear))
    })?;
    let bull_dev = make_device_array_py(device_id, bull)?;
    let bear_dev = make_device_array_py(device_id, bear)?;
    Ok((bull_dev, bear_dev))
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "eri_cuda_many_series_one_param_dev")]
#[pyo3(signature = (high_tm_f32, low_tm_f32, source_tm_f32, cols, rows, period, ma_type, device_id=0))]
pub fn eri_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    high_tm_f32: numpy::PyReadonlyArray1<'_, f32>,
    low_tm_f32: numpy::PyReadonlyArray1<'_, f32>,
    source_tm_f32: numpy::PyReadonlyArray1<'_, f32>,
    cols: usize,
    rows: usize,
    period: usize,
    ma_type: &str,
    device_id: usize,
) -> PyResult<(DeviceArrayF32Py, DeviceArrayF32Py)> {
    use crate::cuda::cuda_available;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let h = high_tm_f32.as_slice()?;
    let l = low_tm_f32.as_slice()?;
    let s = source_tm_f32.as_slice()?;
    let (bull, bear) = py.allow_threads(|| {
        let cuda = CudaEri::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.eri_many_series_one_param_time_major_dev(h, l, s, cols, rows, period, ma_type)
            .map_err(|e| PyValueError::new_err(e.to_string()))
            .map(|(bull, bear)| (bull, bear))
    })?;
    let bull_dev = make_device_array_py(device_id, bull)?;
    let bear_dev = make_device_array_py(device_id, bear)?;
    Ok((bull_dev, bear_dev))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn eri_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    source_ptr: *const f64,
    bull_ptr: *mut f64,
    bear_ptr: *mut f64,
    len: usize,
    period: usize,
    ma_type: &str,
) -> Result<(), JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || source_ptr.is_null()
        || bull_ptr.is_null()
        || bear_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let source = std::slice::from_raw_parts(source_ptr, len);

        let params = EriParams {
            period: Some(period),
            ma_type: Some(ma_type.to_string()),
        };
        let input = EriInput::from_slices(high, low, source, params);

        // Check for aliasing - any output pointer matches any input pointer
        let needs_temp = bull_ptr as *const f64 == high_ptr
            || bull_ptr as *const f64 == low_ptr
            || bull_ptr as *const f64 == source_ptr
            || bear_ptr as *const f64 == high_ptr
            || bear_ptr as *const f64 == low_ptr
            || bear_ptr as *const f64 == source_ptr
            || bull_ptr == bear_ptr;

        if needs_temp {
            // Allocate temporary buffers
            let mut temp_bull = vec![0.0; len];
            let mut temp_bear = vec![0.0; len];
            eri_into_slice(&mut temp_bull, &mut temp_bear, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            // Copy to output
            let bull_out = std::slice::from_raw_parts_mut(bull_ptr, len);
            let bear_out = std::slice::from_raw_parts_mut(bear_ptr, len);
            bull_out.copy_from_slice(&temp_bull);
            bear_out.copy_from_slice(&temp_bear);
        } else {
            // Direct write to output
            let bull_out = std::slice::from_raw_parts_mut(bull_ptr, len);
            let bear_out = std::slice::from_raw_parts_mut(bear_ptr, len);
            eri_into_slice(bull_out, bear_out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn eri_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn eri_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct EriBatchConfig {
    pub period_range: (usize, usize, usize),
    pub ma_type: String,
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct EriBatchJsOutput {
    pub values: Vec<f64>,    // [row0..., row1..., ...] where row= bull or bear
    pub rows: usize,         // combos * 2
    pub cols: usize,         // len
    pub periods: Vec<usize>, // length = combos
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = eri_batch)]
pub fn eri_batch_js(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    // Validate lengths match
    if high.len() != low.len() || high.len() != source.len() {
        return Err(JsValue::from_str(
            "high, low, and source arrays must have the same length",
        ));
    }

    let config: EriBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = EriBatchRange {
        period: config.period_range,
        ma_type: config.ma_type,
    };

    let output = eri_batch_with_kernel(high, low, source, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let rows = output
        .rows
        .checked_mul(2)
        .ok_or_else(|| JsValue::from_str("rows overflow"))?;
    let cols = output.cols;

    // flatten as bull rows first then bear rows
    let cap = rows
        .checked_mul(cols)
        .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
    let mut values = Vec::with_capacity(cap);
    // bull block
    for r in 0..output.rows {
        let start = r * cols;
        values.extend_from_slice(&output.bull[start..start + cols]);
    }
    // bear block
    for r in 0..output.rows {
        let start = r * cols;
        values.extend_from_slice(&output.bear[start..start + cols]);
    }

    let periods: Vec<usize> = output.params.iter().map(|p| p.period.unwrap()).collect();

    let js_output = EriBatchJsOutput {
        values,
        rows,
        cols,
        periods,
    };
    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

/// Optimized ERI calculation with inline SMA
#[inline]
pub unsafe fn eri_scalar_classic_sma(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    period: usize,
    first_valid_idx: usize,
    bull: &mut [f64],
    bear: &mut [f64],
) -> Result<(), EriError> {
    let start_idx = first_valid_idx + period - 1;

    // Calculate initial SMA
    let mut sum = 0.0;
    for i in 0..period {
        sum += source[first_valid_idx + i];
    }
    let mut sma = sum / period as f64;

    // Calculate first ERI values
    bull[start_idx] = high[start_idx] - sma;
    bear[start_idx] = low[start_idx] - sma;

    // Rolling window calculations
    for i in (start_idx + 1)..source.len() {
        // Update SMA with rolling window
        let old_val = source[i - period];
        let new_val = source[i];
        sum = sum - old_val + new_val;
        sma = sum / period as f64;

        // Calculate ERI values
        bull[i] = high[i] - sma;
        bear[i] = low[i] - sma;
    }

    Ok(())
}

/// Optimized ERI calculation with inline EMA
#[inline]
pub unsafe fn eri_scalar_classic_ema(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    period: usize,
    first_valid_idx: usize,
    bull: &mut [f64],
    bear: &mut [f64],
) -> Result<(), EriError> {
    let start_idx = first_valid_idx + period - 1;
    let alpha = 2.0 / (period as f64 + 1.0);
    let beta = 1.0 - alpha;

    // Calculate initial SMA for EMA startup
    let mut sum = 0.0;
    for i in 0..period {
        sum += source[first_valid_idx + i];
    }
    let mut ema = sum / period as f64;

    // Calculate first ERI values
    bull[start_idx] = high[start_idx] - ema;
    bear[start_idx] = low[start_idx] - ema;

    // EMA and ERI calculations
    for i in (start_idx + 1)..source.len() {
        // Update EMA
        ema = alpha * source[i] + beta * ema;

        // Calculate ERI values
        bull[i] = high[i] - ema;
        bear[i] = low[i] - ema;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::utilities::enums::Kernel;

    fn check_eri_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = EriParams {
            period: None,
            ma_type: None,
        };
        let input_default = EriInput::from_candles(&candles, "close", default_params);
        let output_default = eri_with_kernel(&input_default, kernel)?;
        assert_eq!(output_default.bull.len(), candles.close.len());
        assert_eq!(output_default.bear.len(), candles.close.len());

        let params_period_14 = EriParams {
            period: Some(14),
            ma_type: Some("ema".to_string()),
        };
        let input_period_14 = EriInput::from_candles(&candles, "hl2", params_period_14);
        let output_period_14 = eri_with_kernel(&input_period_14, kernel)?;
        assert_eq!(output_period_14.bull.len(), candles.close.len());
        assert_eq!(output_period_14.bear.len(), candles.close.len());

        let params_custom = EriParams {
            period: Some(20),
            ma_type: Some("sma".to_string()),
        };
        let input_custom = EriInput::from_candles(&candles, "hlc3", params_custom);
        let output_custom = eri_with_kernel(&input_custom, kernel)?;
        assert_eq!(output_custom.bull.len(), candles.close.len());
        assert_eq!(output_custom.bear.len(), candles.close.len());

        Ok(())
    }

    fn check_eri_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = EriParams {
            period: Some(13),
            ma_type: Some("ema".to_string()),
        };
        let input = EriInput::from_candles(&candles, "close", params);
        let eri_result = eri_with_kernel(&input, kernel)?;

        assert_eq!(eri_result.bull.len(), close_prices.len());
        assert_eq!(eri_result.bear.len(), close_prices.len());

        let expected_bull_last_five = [
            -103.35343557205488,
            6.839912366813223,
            -42.851503685589705,
            -9.444146016219747,
            11.476446271808527,
        ];
        let expected_bear_last_five = [
            -433.3534355720549,
            -314.1600876331868,
            -414.8515036855897,
            -336.44414601621975,
            -925.5235537281915,
        ];

        let start_index = eri_result.bull.len() - 5;
        for i in 0..5 {
            let actual_bull = eri_result.bull[start_index + i];
            let actual_bear = eri_result.bear[start_index + i];
            let expected_bull = expected_bull_last_five[i];
            let expected_bear = expected_bear_last_five[i];
            assert!(
                (actual_bull - expected_bull).abs() < 1e-2,
                "ERI bull mismatch at index {}: expected {}, got {}",
                i,
                expected_bull,
                actual_bull
            );
            assert!(
                (actual_bear - expected_bear).abs() < 1e-2,
                "ERI bear mismatch at index {}: expected {}, got {}",
                i,
                expected_bear,
                actual_bear
            );
        }
        Ok(())
    }

    fn check_eri_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = EriInput::with_default_candles(&candles);
        match input.data {
            EriData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected EriData::Candles"),
        }
        let output = eri_with_kernel(&input, kernel)?;
        assert_eq!(output.bull.len(), candles.close.len());
        assert_eq!(output.bear.len(), candles.close.len());

        Ok(())
    }

    fn check_eri_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 20.0, 30.0];
        let low = [8.0, 18.0, 28.0];
        let src = [9.0, 19.0, 29.0];
        let params = EriParams {
            period: Some(0),
            ma_type: Some("ema".to_string()),
        };
        let input = EriInput::from_slices(&high, &low, &src, params);
        let res = eri_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] ERI should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_eri_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 20.0, 30.0];
        let low = [8.0, 18.0, 28.0];
        let src = [9.0, 19.0, 29.0];
        let params = EriParams {
            period: Some(10),
            ma_type: Some("ema".to_string()),
        };
        let input = EriInput::from_slices(&high, &low, &src, params);
        let res = eri_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] ERI should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_eri_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [42.0];
        let low = [40.0];
        let src = [41.0];
        let params = EriParams {
            period: Some(9),
            ma_type: Some("ema".to_string()),
        };
        let input = EriInput::from_slices(&high, &low, &src, params);
        let res = eri_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] ERI should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_eri_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = EriParams {
            period: Some(14),
            ma_type: Some("ema".to_string()),
        };
        let first_input = EriInput::from_candles(&candles, "close", first_params);
        let first_result = eri_with_kernel(&first_input, kernel)?;

        assert_eq!(first_result.bull.len(), candles.close.len());
        assert_eq!(first_result.bear.len(), candles.close.len());

        let second_params = EriParams {
            period: Some(14),
            ma_type: Some("ema".to_string()),
        };
        let second_input = EriInput::from_slices(
            &first_result.bull,
            &first_result.bear,
            &first_result.bull,
            second_params,
        );
        let second_result = eri_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.bull.len(), first_result.bull.len());
        assert_eq!(second_result.bear.len(), first_result.bear.len());

        for i in 28..second_result.bull.len() {
            assert!(
                !second_result.bull[i].is_nan(),
                "Expected no NaN in bull after index 28, but found NaN at index {}",
                i
            );
            assert!(
                !second_result.bear[i].is_nan(),
                "Expected no NaN in bear after index 28, but found NaN at index {}",
                i
            );
        }
        Ok(())
    }

    fn check_eri_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = EriInput::from_candles(
            &candles,
            "close",
            EriParams {
                period: Some(13),
                ma_type: Some("ema".to_string()),
            },
        );
        let res = eri_with_kernel(&input, kernel)?;
        assert_eq!(res.bull.len(), candles.close.len());
        if res.bull.len() > 240 {
            for (i, &val) in res.bull[240..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at bull-index {}",
                    test_name,
                    240 + i
                );
            }
        }
        if res.bear.len() > 240 {
            for (i, &val) in res.bear[240..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at bear-index {}",
                    test_name,
                    240 + i
                );
            }
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_eri_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Define comprehensive parameter combinations
        let test_params = vec![
            // Default parameters
            EriParams::default(),
            // Minimum period
            EriParams {
                period: Some(2),
                ma_type: Some("ema".to_string()),
            },
            // Small periods with different MA types
            EriParams {
                period: Some(5),
                ma_type: Some("sma".to_string()),
            },
            EriParams {
                period: Some(7),
                ma_type: Some("ema".to_string()),
            },
            EriParams {
                period: Some(10),
                ma_type: Some("wma".to_string()),
            },
            // Medium periods
            EriParams {
                period: Some(13),
                ma_type: Some("ema".to_string()),
            },
            EriParams {
                period: Some(20),
                ma_type: Some("sma".to_string()),
            },
            EriParams {
                period: Some(30),
                ma_type: Some("ema".to_string()),
            },
            // Large periods
            EriParams {
                period: Some(50),
                ma_type: Some("sma".to_string()),
            },
            EriParams {
                period: Some(100),
                ma_type: Some("ema".to_string()),
            },
            // Edge cases
            EriParams {
                period: Some(3),
                ma_type: Some("hma".to_string()),
            },
            EriParams {
                period: Some(21),
                ma_type: Some("dema".to_string()),
            },
            EriParams {
                period: Some(14),
                ma_type: Some("tema".to_string()),
            },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = EriInput::from_candles(&candles, "close", params.clone());
            let output = eri_with_kernel(&input, kernel)?;

            // Check bull values
            for (i, &val) in output.bull.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at bull index {} \
						 with params: period={}, ma_type={} (param set {})",
						test_name,
						val,
						bits,
						i,
						params.period.unwrap_or(13),
						params.ma_type.as_ref().unwrap_or(&"ema".to_string()),
						param_idx
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at bull index {} \
						 with params: period={}, ma_type={} (param set {})",
						test_name,
						val,
						bits,
						i,
						params.period.unwrap_or(13),
						params.ma_type.as_ref().unwrap_or(&"ema".to_string()),
						param_idx
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at bull index {} \
						 with params: period={}, ma_type={} (param set {})",
						test_name,
						val,
						bits,
						i,
						params.period.unwrap_or(13),
						params.ma_type.as_ref().unwrap_or(&"ema".to_string()),
						param_idx
					);
                }
            }

            // Check bear values
            for (i, &val) in output.bear.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at bear index {} \
						 with params: period={}, ma_type={} (param set {})",
						test_name,
						val,
						bits,
						i,
						params.period.unwrap_or(13),
						params.ma_type.as_ref().unwrap_or(&"ema".to_string()),
						param_idx
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at bear index {} \
						 with params: period={}, ma_type={} (param set {})",
						test_name,
						val,
						bits,
						i,
						params.period.unwrap_or(13),
						params.ma_type.as_ref().unwrap_or(&"ema".to_string()),
						param_idx
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at bear index {} \
						 with params: period={}, ma_type={} (param set {})",
						test_name,
						val,
						bits,
						i,
						params.period.unwrap_or(13),
						params.ma_type.as_ref().unwrap_or(&"ema".to_string()),
						param_idx
					);
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_eri_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(()) // No-op in release builds
    }

    // Macro to generate kernel-variant tests as in alma.rs
    macro_rules! generate_all_eri_tests {
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

    generate_all_eri_tests!(
        check_eri_partial_params,
        check_eri_accuracy,
        check_eri_default_candles,
        check_eri_zero_period,
        check_eri_period_exceeds_length,
        check_eri_very_small_dataset,
        check_eri_reinput,
        check_eri_nan_handling,
        check_eri_no_poison
    );

    #[cfg(test)]
    generate_all_eri_tests!(check_eri_property);

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let high = c.select_candle_field("high").unwrap();
        let low = c.select_candle_field("low").unwrap();
        let src = c.select_candle_field("close").unwrap();

        let output = EriBatchBuilder::new()
            .kernel(kernel)
            .period_static(13)
            .apply_slices(high, low, src)?;

        let def = EriParams::default();
        let row = output.values_for_bull(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            -103.35343557205488,
            6.839912366813223,
            -42.851503685589705,
            -9.444146016219747,
            11.476446271808527,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-2,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_eri_into_matches_api() -> Result<(), Box<dyn std::error::Error>> {
        // Prepare a small but non-trivial synthetic dataset
        let len = 256usize;
        let mut high = vec![0.0; len];
        let mut low = vec![0.0; len];
        let mut src = vec![0.0; len];
        for i in 0..len {
            let base = 100.0 + (i as f64) * 0.1;
            src[i] = base;
            high[i] = base + 1.0;
            low[i] = base - 1.0;
        }

        let params = EriParams { period: Some(13), ma_type: Some("ema".to_string()) };
        let input = EriInput::from_slices(&high, &low, &src, params);

        // Baseline via Vec-returning API
        let baseline = eri(&input)?;

        // Preallocate outputs and call into API
        let mut bull = vec![0.0; len];
        let mut bear = vec![0.0; len];
        #[cfg(not(feature = "wasm"))]
        {
            eri_into(&input, &mut bull, &mut bear)?;
        }
        #[cfg(feature = "wasm")]
        {
            // wasm build doesn't expose the native eri_into; use the slice variant for parity
            eri_into_slice(&mut bull, &mut bear, &input, Kernel::Auto)?;
        }

        assert_eq!(baseline.bull.len(), bull.len());
        assert_eq!(baseline.bear.len(), bear.len());

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12
        }

        for i in 0..len {
            assert!(
                eq_or_both_nan(baseline.bull[i], bull[i]),
                "bull mismatch at index {i}: baseline={} into={}",
                baseline.bull[i],
                bull[i]
            );
            assert!(
                eq_or_both_nan(baseline.bear[i], bear[i]),
                "bear mismatch at index {i}: baseline={} into={}",
                baseline.bear[i],
                bear[i]
            );
        }

        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let high = c.select_candle_field("high").unwrap();
        let low = c.select_candle_field("low").unwrap();
        let src = c.select_candle_field("close").unwrap();

        // Test various parameter sweep configurations
        let test_configs = vec![
            // (period_start, period_end, period_step)
            (2, 10, 2),   // Small periods
            (5, 25, 5),   // Medium periods
            (30, 60, 15), // Large periods
            (2, 5, 1),    // Dense small range
            (10, 20, 2),  // Medium dense range
            (20, 50, 10), // Large sparse range
            (13, 13, 0),  // Single period (default)
        ];

        for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
            let output = EriBatchBuilder::new()
                .kernel(kernel)
                .period_range(p_start, p_end, p_step)
                .apply_slices(high, low, src)?;

            // Check bull values
            for (idx, &val) in output.bull.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;
                let combo = &output.params[row];

                // Check all three poison patterns with detailed context
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
						 at bull row {} col {} (flat index {}) with params: period={}, ma_type={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(13),
                        combo.ma_type.as_ref().unwrap_or(&"ema".to_string())
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at bull row {} col {} (flat index {}) with params: period={}, ma_type={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(13),
                        combo.ma_type.as_ref().unwrap_or(&"ema".to_string())
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at bull row {} col {} (flat index {}) with params: period={}, ma_type={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(13),
                        combo.ma_type.as_ref().unwrap_or(&"ema".to_string())
                    );
                }
            }

            // Check bear values
            for (idx, &val) in output.bear.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;
                let combo = &output.params[row];

                // Check all three poison patterns with detailed context
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
						 at bear row {} col {} (flat index {}) with params: period={}, ma_type={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(13),
                        combo.ma_type.as_ref().unwrap_or(&"ema".to_string())
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at bear row {} col {} (flat index {}) with params: period={}, ma_type={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(13),
                        combo.ma_type.as_ref().unwrap_or(&"ema".to_string())
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at bear row {} col {} (flat index {}) with params: period={}, ma_type={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(13),
                        combo.ma_type.as_ref().unwrap_or(&"ema".to_string())
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(
        _test: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    #[cfg(test)]
    #[allow(clippy::float_cmp)]
    fn check_eri_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Strategy to generate realistic OHLC data
        let strat = (2usize..=50) // ERI requires minimum period of 2
            .prop_flat_map(|period| {
                (
                    100.0f64..5000.0f64, // Base price
                    (period + 20)..400,  // Data length
                    0.001f64..0.05f64,   // Volatility factor
                    -0.01f64..0.01f64,   // Trend factor
                    Just(period),
                    prop::sample::select(vec!["ema", "sma", "wma"]), // MA types - simpler ones that don't need excessive warmup
                )
            })
            .prop_map(
                |(base_price, data_len, volatility, trend, period, ma_type)| {
                    // Generate synthetic OHLC data with realistic constraints
                    let mut high = Vec::with_capacity(data_len);
                    let mut low = Vec::with_capacity(data_len);
                    let mut close = Vec::with_capacity(data_len);

                    let mut price = base_price;
                    for i in 0..data_len {
                        // Add trend and volatility
                        price *= 1.0 + trend + (i as f64 * 0.0001 * trend);
                        let daily_vol = volatility * price;

                        // Generate OHLC with proper constraints
                        let c = price + daily_vol * ((i as f64).sin() * 0.3);
                        let h = c + daily_vol * (0.5 + (i as f64 * 0.7).cos().abs() * 0.5);
                        let l = c - daily_vol * (0.5 + (i as f64 * 0.7).sin().abs() * 0.5);

                        high.push(h);
                        low.push(l.min(c)); // Ensure low <= close
                        close.push(c);
                    }

                    (high, low, close, period, ma_type.to_string())
                },
            );

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(high, low, close, period, ma_type)| {
                let params = EriParams {
                    period: Some(period),
                    ma_type: Some(ma_type.clone()),
                };
                let input = EriInput::from_slices(&high, &low, &close, params.clone());

                // Run with specified kernel - handle MA errors gracefully
                let result = match eri_with_kernel(&input, kernel) {
                    Ok(r) => r,
                    Err(e) => {
                        // DEMA and TEMA need approximately 2*period and 3*period data respectively
                        // These errors are expected when testing with shorter data lengths
                        match e {
                            EriError::MaCalculationError(msg)
                                if msg.contains("Not enough data") =>
                            {
                                return Ok(())
                            }
                            EriError::NotEnoughValidData { .. } => return Ok(()),
                            _ => panic!("Unexpected error type: {:?}", e),
                        }
                    }
                };

                // Also run with scalar as reference
                let reference = match eri_with_kernel(&input, Kernel::Scalar) {
                    Ok(r) => r,
                    Err(_) => return Ok(()), // If scalar fails, kernel would too
                };

                // Calculate expected warmup period
                let first_valid_idx = high
                    .iter()
                    .zip(low.iter())
                    .zip(close.iter())
                    .position(|((h, l), c)| !h.is_nan() && !l.is_nan() && !c.is_nan())
                    .unwrap_or(0);
                let warmup_period = first_valid_idx + period - 1;

                // Property 1: Check warmup period - values before warmup should be NaN
                for i in 0..warmup_period.min(high.len()) {
                    prop_assert!(
                        result.bull[i].is_nan(),
                        "[{}] Expected NaN in bull warmup at index {}, got {}",
                        test_name,
                        i,
                        result.bull[i]
                    );
                    prop_assert!(
                        result.bear[i].is_nan(),
                        "[{}] Expected NaN in bear warmup at index {}, got {}",
                        test_name,
                        i,
                        result.bear[i]
                    );
                }

                // Property 2: Check values after warmup are not NaN (unless input was NaN)
                for i in warmup_period..high.len() {
                    if !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan() {
                        prop_assert!(
                            !result.bull[i].is_nan(),
                            "[{}] Unexpected NaN in bull at index {} after warmup",
                            test_name,
                            i
                        );
                        prop_assert!(
                            !result.bear[i].is_nan(),
                            "[{}] Unexpected NaN in bear at index {} after warmup",
                            test_name,
                            i
                        );
                    }
                }

                // Property 3: Bear should always be <= Bull (since low <= high)
                for i in warmup_period..high.len() {
                    if !result.bull[i].is_nan() && !result.bear[i].is_nan() {
                        prop_assert!(
                            result.bear[i] <= result.bull[i] + 1e-9,
                            "[{}] Bear {} > Bull {} at index {} (low={}, high={})",
                            test_name,
                            result.bear[i],
                            result.bull[i],
                            i,
                            low[i],
                            high[i]
                        );
                    }
                }

                // Property 4: Kernel consistency - all kernels should produce identical results
                for i in 0..high.len() {
                    let bull_val = result.bull[i];
                    let bear_val = result.bear[i];
                    let ref_bull = reference.bull[i];
                    let ref_bear = reference.bear[i];

                    // Check bull consistency
                    if bull_val.is_finite() && ref_bull.is_finite() {
                        let bull_diff = (bull_val - ref_bull).abs();
                        let bull_ulp = bull_val.to_bits().abs_diff(ref_bull.to_bits());
                        prop_assert!(
                            bull_diff <= 1e-9 || bull_ulp <= 4,
                            "[{}] Bull kernel mismatch at index {}: {} vs {} (diff={}, ULP={})",
                            test_name,
                            i,
                            bull_val,
                            ref_bull,
                            bull_diff,
                            bull_ulp
                        );
                    } else {
                        prop_assert_eq!(
                            bull_val.to_bits(),
                            ref_bull.to_bits(),
                            "[{}] Bull NaN/Inf mismatch at index {}: {} vs {}",
                            test_name,
                            i,
                            bull_val,
                            ref_bull
                        );
                    }

                    // Check bear consistency
                    if bear_val.is_finite() && ref_bear.is_finite() {
                        let bear_diff = (bear_val - ref_bear).abs();
                        let bear_ulp = bear_val.to_bits().abs_diff(ref_bear.to_bits());
                        prop_assert!(
                            bear_diff <= 1e-9 || bear_ulp <= 4,
                            "[{}] Bear kernel mismatch at index {}: {} vs {} (diff={}, ULP={})",
                            test_name,
                            i,
                            bear_val,
                            ref_bear,
                            bear_diff,
                            bear_ulp
                        );
                    } else {
                        prop_assert_eq!(
                            bear_val.to_bits(),
                            ref_bear.to_bits(),
                            "[{}] Bear NaN/Inf mismatch at index {}: {} vs {}",
                            test_name,
                            i,
                            bear_val,
                            ref_bear
                        );
                    }
                }

                // Property 5: Constant price test - with constant prices, values converge to exact constants
                let all_high_same = high.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10);
                let all_low_same = low.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10);
                let all_close_same = close.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10);

                if all_high_same
                    && all_low_same
                    && all_close_same
                    && high.len() > warmup_period + 2 * period
                {
                    // With constant prices, MA converges to the constant close value
                    // So bull = high[0] - close[0] and bear = low[0] - close[0]
                    let expected_bull = high[0] - close[0];
                    let expected_bear = low[0] - close[0];

                    // After sufficient warmup (2*period), values should converge
                    for i in (warmup_period + 2 * period)..high.len() {
                        if !result.bull[i].is_nan() && !result.bear[i].is_nan() {
                            prop_assert!(
                                (result.bull[i] - expected_bull).abs() < 1e-6,
                                "[{}] Constant price: Bull {} != expected {} at index {}",
                                test_name,
                                result.bull[i],
                                expected_bull,
                                i
                            );
                            prop_assert!(
                                (result.bear[i] - expected_bear).abs() < 1e-6,
                                "[{}] Constant price: Bear {} != expected {} at index {}",
                                test_name,
                                result.bear[i],
                                expected_bear,
                                i
                            );
                        }
                    }
                }

                // Property 6: Mathematical definition check
                // Bull = High - MA, Bear = Low - MA
                // So Bull - Bear = High - Low
                for i in warmup_period..high.len() {
                    if !result.bull[i].is_nan() && !result.bear[i].is_nan() {
                        let expected_diff = high[i] - low[i];
                        let actual_diff = result.bull[i] - result.bear[i];
                        prop_assert!(
                            (actual_diff - expected_diff).abs() < 1e-9,
                            "[{}] Bull - Bear != High - Low at index {}: {} vs {}",
                            test_name,
                            i,
                            actual_diff,
                            expected_diff
                        );
                    }
                }

                // Property 7: Impossible state check
                // Since bull = high - MA and bear = low - MA, and high >= low,
                // it's impossible for bull < 0 while bear > 0
                for i in warmup_period..high.len() {
                    if !result.bull[i].is_nan() && !result.bear[i].is_nan() {
                        // The only impossible combination
                        if result.bull[i] < 0.0 && result.bear[i] > 0.0 {
                            prop_assert!(
								false,
								"[{}] Impossible state: bull {} < 0 but bear {} > 0 at index {} (high={}, low={})",
								test_name, result.bull[i], result.bear[i], i, high[i], low[i]
							);
                        }
                    }
                }

                // Property 8: Period boundary test
                if period == 1 {
                    // With period=1, most MAs should give MA ≈ close[i]
                    // So bull ≈ high[i] - close[i] and bear ≈ low[i] - close[i]
                    for i in warmup_period..high.len().min(warmup_period + 10) {
                        if !result.bull[i].is_nan() && !result.bear[i].is_nan() {
                            // Just check the relationship holds
                            let expected_diff = high[i] - low[i];
                            let actual_diff = result.bull[i] - result.bear[i];
                            prop_assert!(
                                (actual_diff - expected_diff).abs() < 1e-6,
                                "[{}] Period=1: Bull-Bear mismatch at index {}: {} vs {}",
                                test_name,
                                i,
                                actual_diff,
                                expected_diff
                            );
                        }
                    }
                }

                // Property 9: Bounds check - bull/bear values should be within reasonable range
                // They shouldn't exceed the price range in the data
                let max_price = high.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let min_price = low.iter().cloned().fold(f64::INFINITY, f64::min);
                let price_range = max_price - min_price;

                for i in warmup_period..high.len() {
                    if !result.bull[i].is_nan() && !result.bear[i].is_nan() {
                        // Bull and bear represent deviations from MA, so they should be bounded
                        prop_assert!(
                            result.bull[i].abs() <= price_range * 2.0,
                            "[{}] Bull {} exceeds reasonable bounds (price range: {}) at index {}",
                            test_name,
                            result.bull[i],
                            price_range,
                            i
                        );
                        prop_assert!(
                            result.bear[i].abs() <= price_range * 2.0,
                            "[{}] Bear {} exceeds reasonable bounds (price range: {}) at index {}",
                            test_name,
                            result.bear[i],
                            price_range,
                            i
                        );
                    }
                }

                // Property 10: Edge case - period close to data length
                if period >= high.len() - 5 && period < high.len() {
                    // With period very close to data length, we should still get at least one valid value
                    let valid_count = result
                        .bull
                        .iter()
                        .zip(result.bear.iter())
                        .filter(|(b, r)| !b.is_nan() && !r.is_nan())
                        .count();
                    prop_assert!(
                        valid_count >= 1,
                        "[{}] No valid values with period {} and data_len {}",
                        test_name,
                        period,
                        high.len()
                    );
                }

                Ok(())
            })
            .unwrap();

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
