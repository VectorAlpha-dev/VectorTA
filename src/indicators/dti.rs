#[cfg(all(feature = "python", feature = "cuda"))]
use cust::memory::DeviceBuffer;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use serde::{Deserialize, Serialize};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum DtiData<'a> {
    Candles { candles: &'a Candles },
    Slices { high: &'a [f64], low: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct DtiOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct DtiParams {
    pub r: Option<usize>,
    pub s: Option<usize>,
    pub u: Option<usize>,
}

impl Default for DtiParams {
    fn default() -> Self {
        Self {
            r: Some(14),
            s: Some(10),
            u: Some(5),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DtiInput<'a> {
    pub data: DtiData<'a>,
    pub params: DtiParams,
}

impl<'a> DtiInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: DtiParams) -> Self {
        Self {
            data: DtiData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(high: &'a [f64], low: &'a [f64], params: DtiParams) -> Self {
        Self {
            data: DtiData::Slices { high, low },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, DtiParams::default())
    }

    #[inline]
    pub fn get_r(&self) -> usize {
        self.params.r.unwrap_or(14)
    }
    #[inline]
    pub fn get_s(&self) -> usize {
        self.params.s.unwrap_or(10)
    }
    #[inline]
    pub fn get_u(&self) -> usize {
        self.params.u.unwrap_or(5)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct DtiBuilder {
    r: Option<usize>,
    s: Option<usize>,
    u: Option<usize>,
    kernel: Kernel,
}

impl Default for DtiBuilder {
    fn default() -> Self {
        Self {
            r: None,
            s: None,
            u: None,
            kernel: Kernel::Auto,
        }
    }
}

impl DtiBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn r(mut self, n: usize) -> Self {
        self.r = Some(n);
        self
    }
    #[inline(always)]
    pub fn s(mut self, n: usize) -> Self {
        self.s = Some(n);
        self
    }
    #[inline(always)]
    pub fn u(mut self, n: usize) -> Self {
        self.u = Some(n);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<DtiOutput, DtiError> {
        let p = DtiParams {
            r: self.r,
            s: self.s,
            u: self.u,
        };
        let i = DtiInput::from_candles(c, p);
        dti_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<DtiOutput, DtiError> {
        let p = DtiParams {
            r: self.r,
            s: self.s,
            u: self.u,
        };
        let i = DtiInput::from_slices(high, low, p);
        dti_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<DtiStream, DtiError> {
        let p = DtiParams {
            r: self.r,
            s: self.s,
            u: self.u,
        };
        DtiStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum DtiError {
    #[error("dti: Input data slice is empty.")]
    EmptyInputData,
    #[error("dti: Candle field error: {0}")]
    CandleFieldError(String),
    #[error("dti: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("dti: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("dti: All high/low values are NaN.")]
    AllValuesNaN,
    #[error("dti: Length mismatch: high length = {high}, low length = {low}")]
    LengthMismatch { high: usize, low: usize },
    #[error("dti: output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("dti: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("dti: invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
}

#[inline]
pub fn dti(input: &DtiInput) -> Result<DtiOutput, DtiError> {
    dti_with_kernel(input, Kernel::Auto)
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn dti_into(input: &DtiInput, out: &mut [f64]) -> Result<(), DtiError> {
    dti_into_slice(out, input, Kernel::Auto)
}

#[inline]
pub fn dti_into_slice(dst: &mut [f64], input: &DtiInput, kern: Kernel) -> Result<(), DtiError> {
    let (high, low) = match &input.data {
        DtiData::Candles { candles } => {
            let high = candles
                .select_candle_field("high")
                .map_err(|e| DtiError::CandleFieldError(e.to_string()))?;
            let low = candles
                .select_candle_field("low")
                .map_err(|e| DtiError::CandleFieldError(e.to_string()))?;
            (high, low)
        }
        DtiData::Slices { high, low } => (*high, *low),
    };

    if high.is_empty() || low.is_empty() {
        return Err(DtiError::EmptyInputData);
    }
    let len = high.len();
    if low.len() != len {
        return Err(DtiError::LengthMismatch {
            high: high.len(),
            low: low.len(),
        });
    }
    if dst.len() != len {
        return Err(DtiError::OutputLengthMismatch {
            expected: len,
            got: dst.len(),
        });
    }

    let first_valid_idx = (0..len)
        .find(|&i| !high[i].is_nan() && !low[i].is_nan())
        .ok_or(DtiError::AllValuesNaN)?;

    let r = input.get_r();
    let s = input.get_s();
    let u = input.get_u();
    for &p in &[r, s, u] {
        if p == 0 || p > len {
            return Err(DtiError::InvalidPeriod {
                period: p,
                data_len: len,
            });
        }
        if len - first_valid_idx < p {
            return Err(DtiError::NotEnoughValidData {
                needed: p,
                valid: len - first_valid_idx,
            });
        }
    }

    let chosen = match kern {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    unsafe {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            match chosen {
                Kernel::Scalar | Kernel::ScalarBatch => {
                    dti_simd128(high, low, r, s, u, first_valid_idx, dst)
                }
                _ => dti_scalar(high, low, r, s, u, first_valid_idx, dst),
            }
        }
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                dti_scalar(high, low, r, s, u, first_valid_idx, dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => dti_avx2(high, low, r, s, u, first_valid_idx, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                dti_avx512(high, low, r, s, u, first_valid_idx, dst)
            }
            _ => unreachable!(),
        }
    }

    for v in &mut dst[..=first_valid_idx] {
        *v = f64::NAN;
    }
    Ok(())
}

pub fn dti_with_kernel(input: &DtiInput, kernel: Kernel) -> Result<DtiOutput, DtiError> {
    let (high, low) = match &input.data {
        DtiData::Candles { candles } => {
            let high = candles
                .select_candle_field("high")
                .map_err(|e| DtiError::CandleFieldError(e.to_string()))?;
            let low = candles
                .select_candle_field("low")
                .map_err(|e| DtiError::CandleFieldError(e.to_string()))?;
            (high, low)
        }
        DtiData::Slices { high, low } => (*high, *low),
    };

    if high.is_empty() || low.is_empty() {
        return Err(DtiError::EmptyInputData);
    }
    let len = high.len();
    if low.len() != len {
        return Err(DtiError::LengthMismatch {
            high: high.len(),
            low: low.len(),
        });
    }

    let first_valid_idx = match (0..len).find(|&i| !high[i].is_nan() && !low[i].is_nan()) {
        Some(idx) => idx,
        None => return Err(DtiError::AllValuesNaN),
    };

    let r = input.get_r();
    let s = input.get_s();
    let u = input.get_u();

    for &period in &[r, s, u] {
        if period == 0 || period > len {
            return Err(DtiError::InvalidPeriod {
                period,
                data_len: len,
            });
        }
        if (len - first_valid_idx) < period {
            return Err(DtiError::NotEnoughValidData {
                needed: period,
                valid: len - first_valid_idx,
            });
        }
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let mut out = alloc_with_nan_prefix(len, first_valid_idx + 1);
    unsafe {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            match chosen {
                Kernel::Scalar | Kernel::ScalarBatch => {
                    dti_simd128(high, low, r, s, u, first_valid_idx, &mut out)
                }
                _ => dti_scalar(high, low, r, s, u, first_valid_idx, &mut out),
            }
            return Ok(DtiOutput { values: out });
        }

        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                dti_scalar(high, low, r, s, u, first_valid_idx, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                dti_avx2(high, low, r, s, u, first_valid_idx, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                dti_avx512(high, low, r, s, u, first_valid_idx, &mut out)
            }
            _ => unreachable!(),
        }
    }
    Ok(DtiOutput { values: out })
}

#[inline]
pub fn dti_scalar(
    high: &[f64],
    low: &[f64],
    r: usize,
    s: usize,
    u: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    let len = high.len();
    let alpha_r = 2.0 / (r as f64 + 1.0);
    let alpha_s = 2.0 / (s as f64 + 1.0);
    let alpha_u = 2.0 / (u as f64 + 1.0);

    let alpha_r_1 = 1.0 - alpha_r;
    let alpha_s_1 = 1.0 - alpha_s;
    let alpha_u_1 = 1.0 - alpha_u;

    let mut e0_r = 0.0;
    let mut e0_s = 0.0;
    let mut e0_u = 0.0;
    let mut e1_r = 0.0;
    let mut e1_s = 0.0;
    let mut e1_u = 0.0;

    out[first_valid_idx] = f64::NAN;
    for i in (first_valid_idx + 1)..len {
        let dh = high[i] - high[i - 1];
        let dl = low[i] - low[i - 1];
        let x_hmu = if dh > 0.0 { dh } else { 0.0 };
        let x_lmd = if dl < 0.0 { -dl } else { 0.0 };
        let x_price = x_hmu - x_lmd;
        let x_price_abs = x_price.abs();

        e0_r = alpha_r * x_price + alpha_r_1 * e0_r;
        e0_s = alpha_s * e0_r + alpha_s_1 * e0_s;
        e0_u = alpha_u * e0_s + alpha_u_1 * e0_u;

        e1_r = alpha_r * x_price_abs + alpha_r_1 * e1_r;
        e1_s = alpha_s * e1_r + alpha_s_1 * e1_s;
        e1_u = alpha_u * e1_s + alpha_u_1 * e1_u;

        if !e1_u.is_nan() && e1_u != 0.0 {
            out[i] = 100.0 * e0_u / e1_u;
        } else {
            out[i] = 0.0;
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn dti_simd128(
    high: &[f64],
    low: &[f64],
    r: usize,
    s: usize,
    u: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    dti_scalar(high, low, r, s, u, first_valid_idx, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn dti_avx2(
    high: &[f64],
    low: &[f64],
    r: usize,
    s: usize,
    u: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    use core::arch::x86_64::*;

    let len = high.len();
    let start = first_valid_idx + 1;
    if start >= len {
        return;
    }

    let alpha_r = 2.0 / (r as f64 + 1.0);
    let alpha_s = 2.0 / (s as f64 + 1.0);
    let alpha_u = 2.0 / (u as f64 + 1.0);
    let ar1 = 1.0 - alpha_r;
    let as1 = 1.0 - alpha_s;
    let au1 = 1.0 - alpha_u;

    unsafe {
        let vr_a = _mm_set1_pd(alpha_r);
        let vs_a = _mm_set1_pd(alpha_s);
        let vu_a = _mm_set1_pd(alpha_u);
        let vr_b = _mm_set1_pd(ar1);
        let vs_b = _mm_set1_pd(as1);
        let vu_b = _mm_set1_pd(au1);

        let mut v_er = _mm_set1_pd(0.0);
        let mut v_es = _mm_set1_pd(0.0);
        let mut v_eu = _mm_set1_pd(0.0);

        let ph = high.as_ptr();
        let pl = low.as_ptr();
        let po = out.as_mut_ptr();
        let half = 0.5f64;
        let hundred = 100.0f64;

        let mut i = start;
        while i < len {
            let hi0 = *ph.add(i);
            let hi_1 = *ph.add(i - 1);
            let lo0 = *pl.add(i);
            let lo_1 = *pl.add(i - 1);

            let dh = hi0 - hi_1;
            let dl = lo0 - lo_1;

            let x = half * (dh.abs() + dh) - half * (dl.abs() - dl);
            let ax = x.abs();

            let vx = _mm_set_pd(ax, x);

            v_er = _mm_add_pd(_mm_mul_pd(vr_a, vx), _mm_mul_pd(vr_b, v_er));
            v_es = _mm_add_pd(_mm_mul_pd(vs_a, v_er), _mm_mul_pd(vs_b, v_es));
            v_eu = _mm_add_pd(_mm_mul_pd(vu_a, v_es), _mm_mul_pd(vu_b, v_eu));

            let mut tmp = [0.0f64; 2];
            _mm_storeu_pd(tmp.as_mut_ptr(), v_eu);
            let num = tmp[0];
            let den = tmp[1];

            *po.add(i) = if !den.is_nan() && den != 0.0 {
                hundred * num / den
            } else {
                0.0
            };

            i += 1;
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn dti_avx512(
    high: &[f64],
    low: &[f64],
    r: usize,
    s: usize,
    u: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    dti_avx2(high, low, r, s, u, first_valid_idx, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn dti_avx512_short(
    high: &[f64],
    low: &[f64],
    r: usize,
    s: usize,
    u: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    dti_avx2(high, low, r, s, u, first_valid_idx, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn dti_avx512_long(
    high: &[f64],
    low: &[f64],
    r: usize,
    s: usize,
    u: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    dti_avx2(high, low, r, s, u, first_valid_idx, out)
}

#[inline(always)]
pub fn dti_row_scalar(
    high: &[f64],
    low: &[f64],
    r: usize,
    s: usize,
    u: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    dti_scalar(high, low, r, s, u, first_valid_idx, out)
}

#[inline(always)]
fn dti_precompute_base(high: &[f64], low: &[f64], start: usize) -> (AVec<f64>, AVec<f64>) {
    let len = high.len();
    let mut x = AVec::<f64>::new(CACHELINE_ALIGN);
    let mut ax = AVec::<f64>::new(CACHELINE_ALIGN);
    x.resize(len, 0.0);
    ax.resize(len, 0.0);
    for i in (start)..len {
        let dh = high[i] - high[i - 1];
        let dl = low[i] - low[i - 1];
        let x_hmu = if dh > 0.0 { dh } else { 0.0 };
        let x_lmd = if dl < 0.0 { -dl } else { 0.0 };
        let v = x_hmu - x_lmd;
        x[i] = v;
        ax[i] = v.abs();
    }
    (x, ax)
}

#[inline(always)]
fn dti_row_scalar_from_base(
    x: &[f64],
    ax: &[f64],
    r: usize,
    s: usize,
    u: usize,
    start: usize,
    out: &mut [f64],
) {
    let len = x.len();
    if start >= len {
        return;
    }
    let alpha_r = 2.0 / (r as f64 + 1.0);
    let alpha_s = 2.0 / (s as f64 + 1.0);
    let alpha_u = 2.0 / (u as f64 + 1.0);
    let ar1 = 1.0 - alpha_r;
    let as1 = 1.0 - alpha_s;
    let au1 = 1.0 - alpha_u;
    let mut e0_r = 0.0;
    let mut e0_s = 0.0;
    let mut e0_u = 0.0;
    let mut e1_r = 0.0;
    let mut e1_s = 0.0;
    let mut e1_u = 0.0;
    for i in start..len {
        let xi = x[i];
        let axi = ax[i];
        e0_r = alpha_r * xi + ar1 * e0_r;
        e0_s = alpha_s * e0_r + as1 * e0_s;
        e0_u = alpha_u * e0_s + au1 * e0_u;
        e1_r = alpha_r * axi + ar1 * e1_r;
        e1_s = alpha_s * e1_r + as1 * e1_s;
        e1_u = alpha_u * e1_s + au1 * e1_u;
        out[i] = if !e1_u.is_nan() && e1_u != 0.0 {
            100.0 * e0_u / e1_u
        } else {
            0.0
        };
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
fn dti_row_avx2_from_base(
    x: &[f64],
    ax: &[f64],
    r: usize,
    s: usize,
    u: usize,
    start: usize,
    out: &mut [f64],
) {
    unsafe {
        use core::arch::x86_64::*;
        let len = x.len();
        if start >= len {
            return;
        }
        let alpha_r = 2.0 / (r as f64 + 1.0);
        let alpha_s = 2.0 / (s as f64 + 1.0);
        let alpha_u = 2.0 / (u as f64 + 1.0);
        let ar1 = 1.0 - alpha_r;
        let as1 = 1.0 - alpha_s;
        let au1 = 1.0 - alpha_u;

        let vr_a = _mm_set1_pd(alpha_r);
        let vs_a = _mm_set1_pd(alpha_s);
        let vu_a = _mm_set1_pd(alpha_u);
        let vr_b = _mm_set1_pd(ar1);
        let vs_b = _mm_set1_pd(as1);
        let vu_b = _mm_set1_pd(au1);

        let mut v_er = _mm_set1_pd(0.0);
        let mut v_es = _mm_set1_pd(0.0);
        let mut v_eu = _mm_set1_pd(0.0);
        let px = x.as_ptr();
        let pax = ax.as_ptr();
        let po = out.as_mut_ptr();
        let hundred = 100.0f64;
        let mut i = start;
        while i < len {
            let xi = *px.add(i);
            let axi = *pax.add(i);
            let vx = _mm_set_pd(axi, xi);
            v_er = _mm_add_pd(_mm_mul_pd(vr_a, vx), _mm_mul_pd(vr_b, v_er));
            v_es = _mm_add_pd(_mm_mul_pd(vs_a, v_er), _mm_mul_pd(vs_b, v_es));
            v_eu = _mm_add_pd(_mm_mul_pd(vu_a, v_es), _mm_mul_pd(vu_b, v_eu));
            let mut tmp = [0.0f64; 2];
            _mm_storeu_pd(tmp.as_mut_ptr(), v_eu);
            let num = tmp[0];
            let den = tmp[1];
            *po.add(i) = if !den.is_nan() && den != 0.0 {
                hundred * num / den
            } else {
                0.0
            };
            i += 1;
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
fn dti_row_avx512_from_base(
    x: &[f64],
    ax: &[f64],
    r: usize,
    s: usize,
    u: usize,
    start: usize,
    out: &mut [f64],
) {
    dti_row_avx2_from_base(x, ax, r, s, u, start, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn dti_row_avx2(
    high: &[f64],
    low: &[f64],
    r: usize,
    s: usize,
    u: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    dti_avx2(high, low, r, s, u, first_valid_idx, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn dti_row_avx512(
    high: &[f64],
    low: &[f64],
    r: usize,
    s: usize,
    u: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    dti_avx512(high, low, r, s, u, first_valid_idx, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn dti_row_avx512_short(
    high: &[f64],
    low: &[f64],
    r: usize,
    s: usize,
    u: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    unsafe { dti_avx512_short(high, low, r, s, u, first_valid_idx, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn dti_row_avx512_long(
    high: &[f64],
    low: &[f64],
    r: usize,
    s: usize,
    u: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    unsafe { dti_avx512_long(high, low, r, s, u, first_valid_idx, out) }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
pub fn dti_row_simd128(
    high: &[f64],
    low: &[f64],
    r: usize,
    s: usize,
    u: usize,
    first_valid_idx: usize,
    out: &mut [f64],
) {
    unsafe { dti_simd128(high, low, r, s, u, first_valid_idx, out) }
}

#[derive(Debug, Clone)]
pub struct DtiStream {
    r: usize,
    s: usize,
    u: usize,
    alpha_r: f64,
    alpha_s: f64,
    alpha_u: f64,
    alpha_r_1: f64,
    alpha_s_1: f64,
    alpha_u_1: f64,
    e0_r: f64,
    e0_s: f64,
    e0_u: f64,
    e1_r: f64,
    e1_s: f64,
    e1_u: f64,
    last_high: Option<f64>,
    last_low: Option<f64>,
    initialized: bool,
}

impl DtiStream {
    pub fn try_new(params: DtiParams) -> Result<Self, DtiError> {
        let r = params.r.unwrap_or(14);
        let s = params.s.unwrap_or(10);
        let u = params.u.unwrap_or(5);
        if r == 0 || s == 0 || u == 0 {
            return Err(DtiError::InvalidPeriod {
                period: 0,
                data_len: 0,
            });
        }
        let alpha_r = 2.0 / (r as f64 + 1.0);
        let alpha_s = 2.0 / (s as f64 + 1.0);
        let alpha_u = 2.0 / (u as f64 + 1.0);
        let alpha_r_1 = 1.0 - alpha_r;
        let alpha_s_1 = 1.0 - alpha_s;
        let alpha_u_1 = 1.0 - alpha_u;
        Ok(Self {
            r,
            s,
            u,
            alpha_r,
            alpha_s,
            alpha_u,
            alpha_r_1,
            alpha_s_1,
            alpha_u_1,
            e0_r: 0.0,
            e0_s: 0.0,
            e0_u: 0.0,
            e1_r: 0.0,
            e1_s: 0.0,
            e1_u: 0.0,
            last_high: None,
            last_low: None,
            initialized: false,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64) -> Option<f64> {
        if let (Some(last_h), Some(last_l)) = (self.last_high, self.last_low) {
            let dh = high - last_h;
            let dl = low - last_l;
            let x_hmu = if dh > 0.0 { dh } else { 0.0 };
            let x_lmd = if dl < 0.0 { -dl } else { 0.0 };
            let x_price = x_hmu - x_lmd;
            let x_price_abs = x_price.abs();

            self.e0_r = (x_price - self.e0_r).mul_add(self.alpha_r, self.e0_r);
            self.e0_s = (self.e0_r - self.e0_s).mul_add(self.alpha_s, self.e0_s);
            self.e0_u = (self.e0_s - self.e0_u).mul_add(self.alpha_u, self.e0_u);

            self.e1_r = (x_price_abs - self.e1_r).mul_add(self.alpha_r, self.e1_r);
            self.e1_s = (self.e1_r - self.e1_s).mul_add(self.alpha_s, self.e1_s);
            self.e1_u = (self.e1_s - self.e1_u).mul_add(self.alpha_u, self.e1_u);

            self.last_high = Some(high);
            self.last_low = Some(low);

            if !self.e1_u.is_nan() && self.e1_u != 0.0 {
                Some(fast_div_approx(self.e0_u * 100.0, self.e1_u))
            } else {
                Some(0.0)
            }
        } else {
            self.last_high = Some(high);
            self.last_low = Some(low);
            self.initialized = true;
            None
        }
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.e0_r = 0.0;
        self.e0_s = 0.0;
        self.e0_u = 0.0;
        self.e1_r = 0.0;
        self.e1_s = 0.0;
        self.e1_u = 0.0;
        self.last_high = None;
        self.last_low = None;
        self.initialized = false;
    }

    #[inline(always)]
    pub fn update_delta(&mut self, dh: f64, dl: f64) -> Option<f64> {
        if let (Some(prev_h), Some(prev_l)) = (self.last_high, self.last_low) {
            let high = prev_h + dh;
            let low = prev_l + dl;

            let x_hmu = if dh > 0.0 { dh } else { 0.0 };
            let x_lmd = if dl < 0.0 { -dl } else { 0.0 };
            let x_price = x_hmu - x_lmd;
            let x_price_abs = x_price.abs();

            self.e0_r = (x_price - self.e0_r).mul_add(self.alpha_r, self.e0_r);
            self.e0_s = (self.e0_r - self.e0_s).mul_add(self.alpha_s, self.e0_s);
            self.e0_u = (self.e0_s - self.e0_u).mul_add(self.alpha_u, self.e0_u);

            self.e1_r = (x_price_abs - self.e1_r).mul_add(self.alpha_r, self.e1_r);
            self.e1_s = (self.e1_r - self.e1_s).mul_add(self.alpha_s, self.e1_s);
            self.e1_u = (self.e1_s - self.e1_u).mul_add(self.alpha_u, self.e1_u);

            self.last_high = Some(high);
            self.last_low = Some(low);

            if !self.e1_u.is_nan() && self.e1_u != 0.0 {
                Some(fast_div_approx(self.e0_u * 100.0, self.e1_u))
            } else {
                Some(0.0)
            }
        } else {
            None
        }
    }
}

#[inline(always)]
fn fast_div_approx(num: f64, den: f64) -> f64 {
    debug_assert!(den != 0.0);

    let ad = den.abs();
    if ad <= f32::MAX as f64 && ad >= f32::MIN_POSITIVE as f64 {
        let r0 = (1.0f32 / den as f32) as f64;
        let r1 = r0 * (2.0 - den * r0);
        num * r1
    } else {
        num / den
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "dti")]
#[pyo3(signature = (high, low, r, s, u, kernel=None))]
pub fn dti_py<'py>(
    py: Python<'py>,
    high: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    r: usize,
    s: usize,
    u: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = DtiParams {
        r: Some(r),
        s: Some(s),
        u: Some(u),
    };
    let input = DtiInput::from_slices(high_slice, low_slice, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| dti_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "DtiStream")]
pub struct DtiStreamPy {
    stream: DtiStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl DtiStreamPy {
    #[new]
    fn new(r: usize, s: usize, u: usize) -> PyResult<Self> {
        let params = DtiParams {
            r: Some(r),
            s: Some(s),
            u: Some(u),
        };
        let stream =
            DtiStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(DtiStreamPy { stream })
    }

    fn update(&mut self, high: f64, low: f64) -> Option<f64> {
        self.stream.update(high, low)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "dti_batch")]
#[pyo3(signature = (high, low, r_range, s_range, u_range, kernel=None))]
pub fn dti_batch_py<'py>(
    py: Python<'py>,
    high: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    r_range: (usize, usize, usize),
    s_range: (usize, usize, usize),
    u_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;

    let sweep = DtiBatchRange {
        r: r_range,
        s: s_range,
        u: u_range,
    };

    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = high_slice.len();

    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    let kern = validate_kernel(kernel, true)?;

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
            dti_batch_inner_into(high_slice, low_slice, &sweep, simd, true, slice_out)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "r",
        combos
            .iter()
            .map(|p| p.r.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "s",
        combos
            .iter()
            .map(|p| p.s.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "u",
        combos
            .iter()
            .map(|p| p.u.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    Ok(dict)
}

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::oscillators::dti_wrapper::DeviceArrayF32Dti;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::oscillators::CudaDti;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d;

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", unsendable)]
pub struct DtiDeviceArrayF32Py {
    pub(crate) inner: DeviceArrayF32Dti,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl DtiDeviceArrayF32Py {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("shape", (self.inner.rows, self.inner.cols))?;
        d.set_item("typestr", "<f4")?;
        d.set_item(
            "strides",
            (
                self.inner.cols * std::mem::size_of::<f32>(),
                std::mem::size_of::<f32>(),
            ),
        )?;
        d.set_item("data", (self.inner.device_ptr() as usize, false))?;

        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> PyResult<(i32, i32)> {
        let mut device_ordinal: i32 = self.inner.device_id as i32;
        unsafe {
            let attr = cust::sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL;
            let mut value = std::mem::MaybeUninit::<i32>::uninit();
            let err = cust::sys::cuPointerGetAttribute(
                value.as_mut_ptr() as *mut std::ffi::c_void,
                attr,
                self.inner.buf.as_device_ptr().as_raw(),
            );
            if err == cust::sys::CUresult::CUDA_SUCCESS {
                device_ordinal = value.assume_init();
            } else {
                let _ = cust::sys::cuCtxGetDevice(&mut device_ordinal);
            }
        }
        Ok((2, device_ordinal))
    }

    #[pyo3(signature = (stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &mut self,
        py: Python<'py>,
        stream: Option<pyo3::PyObject>,
        max_version: Option<pyo3::PyObject>,
        dl_device: Option<pyo3::PyObject>,
        copy: Option<pyo3::PyObject>,
    ) -> PyResult<PyObject> {
        let (kdl, alloc_dev) = self.__dlpack_device__()?;
        if let Some(dev_obj) = dl_device.as_ref() {
            if let Ok((dev_ty, dev_id)) = dev_obj.extract::<(i32, i32)>(py) {
                if dev_ty != kdl || dev_id != alloc_dev {
                    let wants_copy = copy
                        .as_ref()
                        .and_then(|c| c.extract::<bool>(py).ok())
                        .unwrap_or(false);
                    if wants_copy {
                        return Err(PyValueError::new_err(
                            "device copy not implemented for __dlpack__",
                        ));
                    } else {
                        return Err(PyValueError::new_err("dl_device mismatch for __dlpack__"));
                    }
                }
            }
        }
        let _ = stream;

        let ctx_arc = self.inner.ctx.clone();
        let dummy =
            DeviceBuffer::from_slice(&[]).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let inner = std::mem::replace(
            &mut self.inner,
            DeviceArrayF32Dti {
                buf: dummy,
                rows: 0,
                cols: 0,
                ctx: ctx_arc,
                device_id: alloc_dev as u32,
            },
        );

        let rows = inner.rows;
        let cols = inner.cols;
        let buf = inner.buf;

        let max_version_bound = max_version.map(|obj| obj.into_bound(py));

        export_f32_cuda_dlpack_2d(py, buf, rows, cols, alloc_dev, max_version_bound)
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "dti_cuda_batch_dev")]
#[pyo3(signature = (high_f32, low_f32, r_range, s_range, u_range, device_id=0))]
pub fn dti_cuda_batch_dev_py<'py>(
    py: Python<'py>,
    high_f32: numpy::PyReadonlyArray1<'py, f32>,
    low_f32: numpy::PyReadonlyArray1<'py, f32>,
    r_range: (usize, usize, usize),
    s_range: (usize, usize, usize),
    u_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<(DtiDeviceArrayF32Py, Bound<'py, pyo3::types::PyDict>)> {
    use crate::cuda::cuda_available;
    use numpy::IntoPyArray;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let high = high_f32.as_slice()?;
    let low = low_f32.as_slice()?;
    let sweep = DtiBatchRange {
        r: r_range,
        s: s_range,
        u: u_range,
    };
    let (inner, combos) = py.allow_threads(|| {
        let cuda = CudaDti::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.dti_batch_dev(high, low, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    let dict = PyDict::new(py);
    let rr: Vec<u64> = combos.iter().map(|c| c.r.unwrap() as u64).collect();
    let ss: Vec<u64> = combos.iter().map(|c| c.s.unwrap() as u64).collect();
    let uu: Vec<u64> = combos.iter().map(|c| c.u.unwrap() as u64).collect();
    dict.set_item("r", rr.into_pyarray(py))?;
    dict.set_item("s", ss.into_pyarray(py))?;
    dict.set_item("u", uu.into_pyarray(py))?;
    Ok((DtiDeviceArrayF32Py { inner }, dict))
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "dti_cuda_many_series_one_param_dev")]
#[pyo3(signature = (high_tm_f32, low_tm_f32, r, s, u, device_id=0))]
pub fn dti_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    high_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    low_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    r: usize,
    s: usize,
    u: usize,
    device_id: usize,
) -> PyResult<DtiDeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    use numpy::PyUntypedArrayMethods;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let high_flat = high_tm_f32.as_slice()?;
    let low_flat = low_tm_f32.as_slice()?;
    let rows = high_tm_f32.shape()[0];
    let cols = high_tm_f32.shape()[1];
    let elems = cols
        .checked_mul(rows)
        .ok_or_else(|| PyValueError::new_err("matrix size overflow"))?;
    if low_tm_f32.shape() != [rows, cols] {
        return Err(PyValueError::new_err("high/low shapes mismatch"));
    }
    if high_tm_f32.len() != elems || low_tm_f32.len() != elems {
        return Err(PyValueError::new_err("high/low flattened sizes mismatch"));
    }
    let params = DtiParams {
        r: Some(r),
        s: Some(s),
        u: Some(u),
    };
    let inner = py.allow_threads(|| {
        let cuda = CudaDti::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.dti_many_series_one_param_time_major_dev(high_flat, low_flat, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    Ok(DtiDeviceArrayF32Py { inner })
}

#[derive(Clone, Debug)]
pub struct DtiBatchRange {
    pub r: (usize, usize, usize),
    pub s: (usize, usize, usize),
    pub u: (usize, usize, usize),
}

impl Default for DtiBatchRange {
    fn default() -> Self {
        Self {
            r: (14, 263, 1),
            s: (10, 10, 0),
            u: (5, 5, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct DtiBatchBuilder {
    range: DtiBatchRange,
    kernel: Kernel,
}

impl DtiBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn r_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.r = (start, end, step);
        self
    }
    #[inline]
    pub fn r_static(mut self, p: usize) -> Self {
        self.range.r = (p, p, 0);
        self
    }
    #[inline]
    pub fn s_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.s = (start, end, step);
        self
    }
    #[inline]
    pub fn s_static(mut self, x: usize) -> Self {
        self.range.s = (x, x, 0);
        self
    }
    #[inline]
    pub fn u_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.u = (start, end, step);
        self
    }
    #[inline]
    pub fn u_static(mut self, s: usize) -> Self {
        self.range.u = (s, s, 0);
        self
    }
    pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<DtiBatchOutput, DtiError> {
        dti_batch_with_kernel(high, low, &self.range, self.kernel)
    }
    pub fn with_default_slices(
        high: &[f64],
        low: &[f64],
        k: Kernel,
    ) -> Result<DtiBatchOutput, DtiError> {
        DtiBatchBuilder::new().kernel(k).apply_slices(high, low)
    }
    pub fn apply_candles(self, c: &Candles) -> Result<DtiBatchOutput, DtiError> {
        let high = c
            .select_candle_field("high")
            .map_err(|e| DtiError::CandleFieldError(e.to_string()))?;
        let low = c
            .select_candle_field("low")
            .map_err(|e| DtiError::CandleFieldError(e.to_string()))?;
        self.apply_slices(high, low)
    }
    pub fn with_default_candles(c: &Candles) -> Result<DtiBatchOutput, DtiError> {
        DtiBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c)
    }
}

#[derive(Clone, Debug)]
pub struct DtiBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<DtiParams>,
    pub rows: usize,
    pub cols: usize,
}

impl DtiBatchOutput {
    pub fn row_for_params(&self, p: &DtiParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.r.unwrap_or(14) == p.r.unwrap_or(14)
                && c.s.unwrap_or(10) == p.s.unwrap_or(10)
                && c.u.unwrap_or(5) == p.u.unwrap_or(5)
        })
    }
    pub fn values_for(&self, p: &DtiParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &DtiBatchRange) -> Vec<DtiParams> {
    #[inline(always)]
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        let mut v = Vec::new();
        if start < end {
            let mut cur = start;
            while cur <= end {
                v.push(cur);
                match cur.checked_add(step) {
                    Some(nxt) => {
                        if nxt > end {
                            break;
                        }
                        cur = nxt;
                    }
                    None => break,
                }
            }
        } else {
            let mut cur = start;
            loop {
                if cur < end {
                    break;
                }
                v.push(cur);
                if cur == end {
                    break;
                }
                match cur.checked_sub(step) {
                    Some(nxt) => {
                        if nxt < end {
                            break;
                        }
                        cur = nxt;
                    }
                    None => break,
                }
            }
        }
        v
    }

    let rr = axis_usize(r.r);
    let ss = axis_usize(r.s);
    let uu = axis_usize(r.u);

    let cap = rr
        .len()
        .checked_mul(ss.len())
        .and_then(|x| x.checked_mul(uu.len()))
        .unwrap_or(0);
    let mut out = Vec::with_capacity(cap);
    for &rv in &rr {
        for &sv in &ss {
            for &uv in &uu {
                out.push(DtiParams {
                    r: Some(rv),
                    s: Some(sv),
                    u: Some(uv),
                });
            }
        }
    }
    out
}

#[inline(always)]
pub fn dti_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    sweep: &DtiBatchRange,
    k: Kernel,
) -> Result<DtiBatchOutput, DtiError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(DtiError::InvalidKernelForBatch(other)),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    dti_batch_par_slice(high, low, sweep, simd)
}

#[inline(always)]
pub fn dti_batch_slice(
    high: &[f64],
    low: &[f64],
    sweep: &DtiBatchRange,
    kern: Kernel,
) -> Result<DtiBatchOutput, DtiError> {
    dti_batch_inner(high, low, sweep, kern, false)
}

#[inline(always)]
pub fn dti_batch_par_slice(
    high: &[f64],
    low: &[f64],
    sweep: &DtiBatchRange,
    kern: Kernel,
) -> Result<DtiBatchOutput, DtiError> {
    dti_batch_inner(high, low, sweep, kern, true)
}

#[inline(always)]
fn dti_batch_inner(
    high: &[f64],
    low: &[f64],
    sweep: &DtiBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<DtiBatchOutput, DtiError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(DtiError::InvalidRange {
            start: 0,
            end: 0,
            step: 0,
        });
    }
    let len = high.len();
    if low.len() != len {
        return Err(DtiError::LengthMismatch {
            high: high.len(),
            low: low.len(),
        });
    }
    let first_valid = (0..len)
        .find(|&i| !high[i].is_nan() && !low[i].is_nan())
        .ok_or(DtiError::AllValuesNaN)?;
    let max_p = combos
        .iter()
        .map(|c| c.r.unwrap().max(c.s.unwrap()).max(c.u.unwrap()))
        .max()
        .unwrap();
    if len - first_valid < max_p {
        return Err(DtiError::NotEnoughValidData {
            needed: max_p,
            valid: len - first_valid,
        });
    }
    let rows = combos.len();
    let cols = len;
    let expected = rows.checked_mul(cols).ok_or(DtiError::InvalidRange {
        start: 0,
        end: 0,
        step: 0,
    })?;

    let warmup_periods: Vec<usize> = combos.iter().map(|_| first_valid + 1).collect();

    let mut buf_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut buf_mu, cols, &warmup_periods);

    let uninit_ptr = buf_mu.as_mut_ptr();
    let values = unsafe { std::slice::from_raw_parts_mut(uninit_ptr as *mut f64, expected) };

    let start = first_valid + 1;
    let (x_base, ax_base) = dti_precompute_base(high, low, start);

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let prm = &combos[row];
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            if matches!(kern, Kernel::Scalar | Kernel::Auto) {
                dti_row_simd128(
                    high,
                    low,
                    prm.r.unwrap(),
                    prm.s.unwrap(),
                    prm.u.unwrap(),
                    first_valid,
                    out_row,
                );
                return;
            }
        }

        match kern {
            Kernel::Scalar | Kernel::Auto => dti_row_scalar_from_base(
                &x_base,
                &ax_base,
                prm.r.unwrap(),
                prm.s.unwrap(),
                prm.u.unwrap(),
                start,
                out_row,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => dti_row_avx2_from_base(
                &x_base,
                &ax_base,
                prm.r.unwrap(),
                prm.s.unwrap(),
                prm.u.unwrap(),
                start,
                out_row,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => dti_row_avx512_from_base(
                &x_base,
                &ax_base,
                prm.r.unwrap(),
                prm.s.unwrap(),
                prm.u.unwrap(),
                start,
                out_row,
            ),
            _ => dti_row_scalar_from_base(
                &x_base,
                &ax_base,
                prm.r.unwrap(),
                prm.s.unwrap(),
                prm.u.unwrap(),
                start,
                out_row,
            ),
        }
    };
    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            values
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in values.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in values.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    let values = unsafe { Vec::from_raw_parts(uninit_ptr as *mut f64, expected, expected) };
    std::mem::forget(buf_mu);

    Ok(DtiBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub fn dti_batch_inner_into(
    high: &[f64],
    low: &[f64],
    sweep: &DtiBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<DtiParams>, DtiError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(DtiError::InvalidRange {
            start: 0,
            end: 0,
            step: 0,
        });
    }
    let len = high.len();
    if low.len() != len {
        return Err(DtiError::LengthMismatch {
            high: high.len(),
            low: low.len(),
        });
    }
    let first_valid = (0..len)
        .find(|&i| !high[i].is_nan() && !low[i].is_nan())
        .ok_or(DtiError::AllValuesNaN)?;
    let max_p = combos
        .iter()
        .map(|c| c.r.unwrap().max(c.s.unwrap()).max(c.u.unwrap()))
        .max()
        .unwrap();
    if len - first_valid < max_p {
        return Err(DtiError::NotEnoughValidData {
            needed: max_p,
            valid: len - first_valid,
        });
    }

    let rows = combos.len();
    let cols = len;
    let expected = rows.checked_mul(cols).ok_or(DtiError::InvalidRange {
        start: 0,
        end: 0,
        step: 0,
    })?;
    if out.len() != expected {
        return Err(DtiError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    let warm: Vec<usize> = vec![first_valid + 1; rows];
    let out_mu: &mut [std::mem::MaybeUninit<f64>] = unsafe {
        std::slice::from_raw_parts_mut(
            out.as_mut_ptr() as *mut std::mem::MaybeUninit<f64>,
            out.len(),
        )
    };
    init_matrix_prefixes(out_mu, cols, &warm);

    let values: &mut [f64] =
        unsafe { std::slice::from_raw_parts_mut(out_mu.as_mut_ptr() as *mut f64, out_mu.len()) };

    let do_row = |row: usize, row_slice: &mut [f64]| unsafe {
        let prm = &combos[row];
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            if matches!(kern, Kernel::Scalar | Kernel::Auto) {
                dti_row_simd128(
                    high,
                    low,
                    prm.r.unwrap(),
                    prm.s.unwrap(),
                    prm.u.unwrap(),
                    first_valid,
                    row_slice,
                );
                return;
            }
        }
        match kern {
            Kernel::Scalar | Kernel::Auto => dti_row_scalar(
                high,
                low,
                prm.r.unwrap(),
                prm.s.unwrap(),
                prm.u.unwrap(),
                first_valid,
                row_slice,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => dti_row_avx2(
                high,
                low,
                prm.r.unwrap(),
                prm.s.unwrap(),
                prm.u.unwrap(),
                first_valid,
                row_slice,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => dti_row_avx512(
                high,
                low,
                prm.r.unwrap(),
                prm.s.unwrap(),
                prm.u.unwrap(),
                first_valid,
                row_slice,
            ),
            _ => dti_row_scalar(
                high,
                low,
                prm.r.unwrap(),
                prm.s.unwrap(),
                prm.u.unwrap(),
                first_valid,
                row_slice,
            ),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        values
            .par_chunks_mut(cols)
            .enumerate()
            .for_each(|(row, sl)| do_row(row, sl));
        #[cfg(target_arch = "wasm32")]
        for (row, sl) in values.chunks_mut(cols).enumerate() {
            do_row(row, sl);
        }
    } else {
        for (row, sl) in values.chunks_mut(cols).enumerate() {
            do_row(row, sl);
        }
    }

    Ok(combos)
}

#[inline(always)]
pub fn expand_grid_dti(r: &DtiBatchRange) -> Vec<DtiParams> {
    expand_grid(r)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn dti_js(
    high: &[f64],
    low: &[f64],
    r: usize,
    s: usize,
    u: usize,
) -> Result<Vec<f64>, JsValue> {
    let params = DtiParams {
        r: Some(r),
        s: Some(s),
        u: Some(u),
    };
    let data = DtiData::Slices { high, low };
    let input = DtiInput { data, params };

    let output =
        dti_with_kernel(&input, Kernel::Auto).map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output.values)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn dti_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    r: usize,
    s: usize,
    u: usize,
) -> Result<(), JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let params = DtiParams {
            r: Some(r),
            s: Some(s),
            u: Some(u),
        };
        let data = DtiData::Slices { high, low };
        let input = DtiInput { data, params };

        if out_ptr as *const f64 == high_ptr || out_ptr as *const f64 == low_ptr {
            let mut temp = vec![0.0; len];
            dti_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            dti_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn dti_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn dti_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct DtiBatchConfig {
    pub r_range: (usize, usize, usize),
    pub s_range: (usize, usize, usize),
    pub u_range: (usize, usize, usize),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct DtiBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<DtiParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = dti_batch)]
pub fn dti_batch_js(high: &[f64], low: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: DtiBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = DtiBatchRange {
        r: config.r_range,
        s: config.s_range,
        u: config.u_range,
    };

    let out = dti_batch_inner(high, low, &sweep, detect_best_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_out = DtiBatchJsOutput {
        values: out.values,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    };
    serde_wasm_bindgen::to_value(&js_out)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn dti_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    r_start: usize,
    r_end: usize,
    r_step: usize,
    s_start: usize,
    s_end: usize,
    s_step: usize,
    u_start: usize,
    u_end: usize,
    u_step: usize,
) -> Result<usize, JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided to dti_batch_into"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);

        let sweep = DtiBatchRange {
            r: (r_start, r_end, r_step),
            s: (s_start, s_end, s_step),
            u: (u_start, u_end, u_step),
        };

        fn axis_count(start: usize, end: usize, step: usize) -> usize {
            if step == 0 || start == end {
                return 1;
            }
            if start < end {
                ((end - start) / step) + 1
            } else {
                ((start - end) / step) + 1
            }
        }
        let r_count = axis_count(r_start, r_end, r_step);
        let s_count = axis_count(s_start, s_end, s_step);
        let u_count = axis_count(u_start, u_end, u_step);
        let total_rows = r_count
            .checked_mul(s_count)
            .and_then(|x| x.checked_mul(u_count))
            .ok_or(JsValue::from_str(
                "range expansion overflow in dti_batch_into",
            ))?;
        let total_len = total_rows
            .checked_mul(len)
            .ok_or(JsValue::from_str("size overflow in dti_batch_into"))?;

        let out = std::slice::from_raw_parts_mut(out_ptr, total_len);

        if out_ptr as *const f64 == high_ptr || out_ptr as *const f64 == low_ptr {
            let mut temp_values = vec![0.0; total_len];
            let combos =
                dti_batch_inner_into(high, low, &sweep, Kernel::Auto, false, &mut temp_values)
                    .map_err(|e| JsValue::from_str(&e.to_string()))?;
            out.copy_from_slice(&temp_values);
        } else {
            dti_batch_inner_into(high, low, &sweep, Kernel::Auto, false, out)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(total_rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    fn check_dti_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = DtiParams {
            r: None,
            s: None,
            u: None,
        };
        let input = DtiInput::from_candles(&candles, default_params);
        let output = dti_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }
    fn check_dti_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = DtiInput::with_default_candles(&candles);
        let result = dti_with_kernel(&input, kernel)?;
        let expected_last_five = [
            -39.0091620347991,
            -39.75219264093014,
            -40.53941417932286,
            -41.2787749205189,
            -42.93758699380749,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-6,
                "[{}] DTI {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }
    fn check_dti_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = DtiInput::with_default_candles(&candles);
        let output = dti_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }
    fn check_dti_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 11.0, 12.0];
        let low = [9.0, 10.0, 11.0];
        let params = DtiParams {
            r: Some(0),
            s: Some(10),
            u: Some(5),
        };
        let input = DtiInput::from_slices(&high, &low, params);
        let res = dti_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] DTI should fail with zero period",
            test_name
        );
        Ok(())
    }
    fn check_dti_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 11.0];
        let low = [9.0, 10.0];
        let params = DtiParams {
            r: Some(14),
            s: Some(10),
            u: Some(5),
        };
        let input = DtiInput::from_slices(&high, &low, params);
        let res = dti_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] DTI should fail with period exceeding length",
            test_name
        );
        Ok(())
    }
    fn check_dti_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [f64::NAN, f64::NAN, f64::NAN];
        let low = [f64::NAN, f64::NAN, f64::NAN];
        let params = DtiParams::default();
        let input = DtiInput::from_slices(&high, &low, params);
        let res = dti_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] DTI should fail with all NaN", test_name);
        Ok(())
    }
    fn check_dti_empty_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high: [f64; 0] = [];
        let low: [f64; 0] = [];
        let params = DtiParams::default();
        let input = DtiInput::from_slices(&high, &low, params);
        let res = dti_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] DTI should fail with empty data",
            test_name
        );
        Ok(())
    }
    fn check_dti_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let high = candles.select_candle_field("high")?;
        let low = candles.select_candle_field("low")?;
        let params = DtiParams::default();
        let input = DtiInput::from_slices(high, low, params.clone());
        let batch_output = dti_with_kernel(&input, kernel)?.values;
        let mut stream = DtiStream::try_new(params)?;
        let mut stream_values = Vec::with_capacity(high.len());
        for (&h, &l) in high.iter().zip(low.iter()) {
            match stream.update(h, l) {
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
                "[{}] DTI streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    fn check_dti_length_mismatch(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let high = vec![10.0, 11.0, 12.0];
        let low = vec![9.0, 10.0];
        let params = DtiParams::default();
        let input = DtiInput::from_slices(&high, &low, params);

        let result = dti_with_kernel(&input, kernel);
        assert!(
            result.is_err(),
            "[{}] Should fail with mismatched lengths",
            test_name
        );

        match result.unwrap_err() {
            DtiError::LengthMismatch { high: h, low: l } => {
                assert_eq!(h, 3, "[{}] High length should be 3", test_name);
                assert_eq!(l, 2, "[{}] Low length should be 2", test_name);
            }
            e => panic!(
                "[{}] Expected LengthMismatch error, got: {:?}",
                test_name, e
            ),
        }

        let mut out = vec![0.0; high.len()];
        let result = dti_into_slice(&mut out, &input, kernel);
        assert!(
            result.is_err(),
            "[{}] dti_into_slice should fail with mismatched lengths",
            test_name
        );

        match result.unwrap_err() {
            DtiError::LengthMismatch { high: h, low: l } => {
                assert_eq!(h, 3, "[{}] High length should be 3", test_name);
                assert_eq!(l, 2, "[{}] Low length should be 2", test_name);
            }
            e => panic!(
                "[{}] Expected LengthMismatch error from dti_into_slice, got: {:?}",
                test_name, e
            ),
        }

        let sweep = DtiBatchRange::default();
        let result = dti_batch_inner(&high, &low, &sweep, kernel, false);
        assert!(
            result.is_err(),
            "[{}] Batch should fail with mismatched lengths",
            test_name
        );

        match result.unwrap_err() {
            DtiError::LengthMismatch { high: h, low: l } => {
                assert_eq!(h, 3, "[{}] Batch high length should be 3", test_name);
                assert_eq!(l, 2, "[{}] Batch low length should be 2", test_name);
            }
            e => panic!(
                "[{}] Expected LengthMismatch error from batch, got: {:?}",
                test_name, e
            ),
        }

        Ok(())
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    fn check_dti_wasm_kernel_fallback(
        test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        let high = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let low = vec![9.0, 10.0, 11.0, 12.0, 13.0];
        let params = DtiParams::default();
        let input = DtiInput::from_slices(&high, &low, params);

        let scalar_result = dti_with_kernel(&input, Kernel::Scalar)?;

        let avx2_result = dti_with_kernel(&input, Kernel::Avx2)?;
        assert_eq!(
            scalar_result.values.len(),
            avx2_result.values.len(),
            "[{}] AVX2 fallback length mismatch",
            test_name
        );
        for (i, (s, a)) in scalar_result
            .values
            .iter()
            .zip(avx2_result.values.iter())
            .enumerate()
        {
            if s.is_nan() && a.is_nan() {
                continue;
            }
            assert!(
                (s - a).abs() < 1e-10,
                "[{}] AVX2 fallback mismatch at {}: scalar={}, avx2={}",
                test_name,
                i,
                s,
                a
            );
        }

        let avx512_result = dti_with_kernel(&input, Kernel::Avx512)?;
        assert_eq!(
            scalar_result.values.len(),
            avx512_result.values.len(),
            "[{}] AVX512 fallback length mismatch",
            test_name
        );
        for (i, (s, a)) in scalar_result
            .values
            .iter()
            .zip(avx512_result.values.iter())
            .enumerate()
        {
            if s.is_nan() && a.is_nan() {
                continue;
            }
            assert!(
                (s - a).abs() < 1e-10,
                "[{}] AVX512 fallback mismatch at {}: scalar={}, avx512={}",
                test_name,
                i,
                s,
                a
            );
        }

        let mut out_scalar = vec![0.0; high.len()];
        let mut out_avx = vec![0.0; high.len()];

        dti_into_slice(&mut out_scalar, &input, Kernel::Scalar)?;
        dti_into_slice(&mut out_avx, &input, Kernel::Avx2)?;

        for (i, (s, a)) in out_scalar.iter().zip(out_avx.iter()).enumerate() {
            if s.is_nan() && a.is_nan() {
                continue;
            }
            assert!(
                (s - a).abs() < 1e-10,
                "[{}] dti_into_slice AVX2 fallback mismatch at {}",
                test_name,
                i
            );
        }

        Ok(())
    }

    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    fn check_dti_wasm_kernel_fallback(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_dti_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let test_params = vec![
            DtiParams::default(),
            DtiParams {
                r: Some(1),
                s: Some(1),
                u: Some(1),
            },
            DtiParams {
                r: Some(5),
                s: Some(5),
                u: Some(5),
            },
            DtiParams {
                r: Some(20),
                s: Some(15),
                u: Some(10),
            },
            DtiParams {
                r: Some(50),
                s: Some(30),
                u: Some(20),
            },
            DtiParams {
                r: Some(100),
                s: Some(50),
                u: Some(25),
            },
            DtiParams {
                r: Some(14),
                s: Some(5),
                u: Some(20),
            },
            DtiParams {
                r: Some(30),
                s: Some(10),
                u: Some(5),
            },
            DtiParams {
                r: Some(10),
                s: Some(20),
                u: Some(15),
            },
            DtiParams {
                r: Some(2),
                s: Some(10),
                u: Some(5),
            },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = DtiInput::from_candles(&candles, params.clone());
            let output = dti_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: r={}, s={}, u={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.r.unwrap_or(14),
                        params.s.unwrap_or(10),
                        params.u.unwrap_or(5),
                        param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: r={}, s={}, u={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.r.unwrap_or(14),
                        params.s.unwrap_or(10),
                        params.u.unwrap_or(5),
                        param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: r={}, s={}, u={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.r.unwrap_or(14),
                        params.s.unwrap_or(10),
                        params.u.unwrap_or(5),
                        param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_dti_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    fn check_dti_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (2usize..=50)
            .prop_flat_map(|max_period| {
                let min_len = max_period * 3;
                (
                    (100.0f64..5000.0f64, 0.01f64..0.1f64),
                    (
                        1usize..=max_period,
                        1usize..=max_period,
                        1usize..=max_period,
                    ),
                    min_len..400,
                )
            })
            .prop_flat_map(|((base_price, volatility), (r, s, u), len)| {
                let price_changes = prop::collection::vec((-1.0f64..1.0f64), len);

                (
                    Just((base_price, volatility)),
                    Just((r, s, u)),
                    price_changes,
                )
            })
            .prop_map(|((base_price, volatility), (r, s, u), changes)| {
                let len = changes.len();
                let mut high = Vec::with_capacity(len);
                let mut low = Vec::with_capacity(len);
                let mut current_price = base_price;

                for change_factor in changes {
                    let change = change_factor * volatility * current_price;
                    current_price = (current_price + change).max(10.0);

                    let daily_range = current_price * volatility * (0.5 + change_factor.abs());
                    let mid_adjustment = change_factor * daily_range * 0.25;

                    let daily_high = current_price + daily_range / 2.0 + mid_adjustment;
                    let daily_low = current_price - daily_range / 2.0 + mid_adjustment;

                    high.push(daily_high);
                    low.push(daily_low.max(1.0));
                }

                (high, low, r, s, u)
            });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(high, low, r, s, u)| {
                let params = DtiParams {
                    r: Some(r),
                    s: Some(s),
                    u: Some(u),
                };
                let input = DtiInput::from_slices(&high, &low, params);

                let result = dti_with_kernel(&input, kernel);
                prop_assert!(result.is_ok(), "DTI computation failed: {:?}", result);
                let DtiOutput { values: out } = result.unwrap();

                let DtiOutput { values: ref_out } =
                    dti_with_kernel(&input, Kernel::Scalar).unwrap();

                prop_assert_eq!(out.len(), high.len(), "Output length mismatch");

                prop_assert!(out[0].is_nan(), "First value should be NaN");

                let finite_values: Vec<f64> =
                    out.iter().copied().filter(|v| v.is_finite()).collect();
                if !finite_values.is_empty() {
                    let max_val = finite_values
                        .iter()
                        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    let min_val = finite_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));

                    prop_assert!(
                        max_val <= 100.0001 && min_val >= -100.0001,
                        "DTI values exceed mathematical bounds: [{:.6}, {:.6}]",
                        min_val,
                        max_val
                    );
                }

                for i in 0..out.len() {
                    let y = out[i];
                    let r = ref_out[i];

                    if !y.is_finite() || !r.is_finite() {
                        prop_assert_eq!(
                            y.to_bits(),
                            r.to_bits(),
                            "NaN/finite mismatch at idx {}: {} vs {}",
                            i,
                            y,
                            r
                        );
                    } else {
                        let diff = (y - r).abs();
                        let ulp_diff = y.to_bits().abs_diff(r.to_bits());
                        prop_assert!(
                            diff <= 1e-9 || ulp_diff <= 10,
                            "Kernel mismatch at idx {}: {} vs {} (diff={}, ulp={})",
                            i,
                            y,
                            r,
                            diff,
                            ulp_diff
                        );
                    }
                }

                let is_strong_uptrend = high
                    .windows(5)
                    .all(|w| w.windows(2).all(|pair| pair[1] > pair[0] * 1.001))
                    && low
                        .windows(5)
                        .all(|w| w.windows(2).all(|pair| pair[1] > pair[0] * 1.001));

                let is_strong_downtrend = high
                    .windows(5)
                    .all(|w| w.windows(2).all(|pair| pair[1] < pair[0] * 0.999))
                    && low
                        .windows(5)
                        .all(|w| w.windows(2).all(|pair| pair[1] < pair[0] * 0.999));

                if (is_strong_uptrend || is_strong_downtrend) && out.len() > r + s + u + 10 {
                    let later_values: Vec<f64> = out[out.len() - 10..]
                        .iter()
                        .copied()
                        .filter(|v| v.is_finite())
                        .collect();
                    if later_values.len() >= 5 {
                        let avg = later_values.iter().sum::<f64>() / later_values.len() as f64;
                        if is_strong_uptrend {
                            prop_assert!(
                                avg > 0.0,
                                "DTI should be positive in strong uptrend: avg={:.2}",
                                avg
                            );
                        }
                        if is_strong_downtrend {
                            prop_assert!(
                                avg < 0.0,
                                "DTI should be negative in strong downtrend: avg={:.2}",
                                avg
                            );
                        }
                    }
                }

                #[cfg(debug_assertions)]
                for (i, &val) in out.iter().enumerate() {
                    if val.is_nan() {
                        continue;
                    }
                    let bits = val.to_bits();
                    prop_assert!(
                        bits != 0x11111111_11111111
                            && bits != 0x22222222_22222222
                            && bits != 0x33333333_33333333,
                        "Found poison value at index {}: {} (0x{:016X})",
                        i,
                        val,
                        bits
                    );
                }

                if r == 1 && s == 1 && u == 1 && out.len() > 10 {
                    let responsive_values: Vec<f64> = out[2..10]
                        .iter()
                        .copied()
                        .filter(|v| v.is_finite() && v.abs() > 0.0)
                        .collect();
                    prop_assert!(
                        !responsive_values.is_empty(),
                        "DTI with period=1 should produce non-zero values quickly"
                    );
                }

                let is_zero_volatility = high
                    .iter()
                    .zip(low.iter())
                    .all(|(h, l)| (h - l).abs() < 1e-10);
                if is_zero_volatility && out.len() > 2 {
                    for i in 2..out.len() {
                        if out[i].is_finite() {
                            prop_assert!(
                                out[i].abs() < 1e-10,
                                "DTI should be 0 with zero volatility at index {}: {}",
                                i,
                                out[i]
                            );
                        }
                    }
                }

                if high.len() >= 10 {
                    let spreads: Vec<f64> =
                        high.iter().zip(low.iter()).map(|(h, l)| h - l).collect();
                    let first_spread = spreads[0];
                    let is_constant_spread = spreads
                        .iter()
                        .all(|&s| (s - first_spread).abs() < first_spread * 0.01);

                    let high_changes: Vec<f64> =
                        high.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
                    let low_changes: Vec<f64> =
                        low.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
                    let avg_high_change =
                        high_changes.iter().sum::<f64>() / high_changes.len() as f64;
                    let avg_low_change = low_changes.iter().sum::<f64>() / low_changes.len() as f64;
                    let is_stable =
                        avg_high_change < high[0] * 0.001 && avg_low_change < low[0] * 0.001;

                    if is_constant_spread && is_stable && out.len() > r + s + u + 5 {
                        let last_values: Vec<f64> = out[out.len() - 5..]
                            .iter()
                            .copied()
                            .filter(|v| v.is_finite())
                            .collect();
                        if last_values.len() >= 3 {
                            let avg_abs = last_values.iter().map(|v| v.abs()).sum::<f64>()
                                / last_values.len() as f64;
                            prop_assert!(
								avg_abs < 10.0,
								"DTI should converge near 0 with constant spread and stable prices: avg_abs={:.2}",
								avg_abs
							);
                        }
                    }
                }

                if high.len() >= 10 {
                    let high_rising = high
                        .windows(5)
                        .all(|w| w.windows(2).all(|pair| pair[1] >= pair[0]));
                    let low_rising = low
                        .windows(5)
                        .all(|w| w.windows(2).all(|pair| pair[1] >= pair[0]));
                    let high_falling = high
                        .windows(5)
                        .all(|w| w.windows(2).all(|pair| pair[1] <= pair[0]));
                    let low_falling = low
                        .windows(5)
                        .all(|w| w.windows(2).all(|pair| pair[1] <= pair[0]));

                    if (high_rising && low_rising) && out.len() > 10 {
                        let mid_to_end: Vec<f64> = out[out.len() / 2..]
                            .iter()
                            .copied()
                            .filter(|v| v.is_finite())
                            .collect();
                        if mid_to_end.len() >= 3 {
                            let positive_count = mid_to_end.iter().filter(|&&v| v > 0.0).count();
                            prop_assert!(
                                positive_count >= mid_to_end.len() / 2,
                                "DTI should be mostly positive when prices consistently rise"
                            );
                        }
                    }

                    if (high_falling && low_falling) && out.len() > 10 {
                        let mid_to_end: Vec<f64> = out[out.len() / 2..]
                            .iter()
                            .copied()
                            .filter(|v| v.is_finite())
                            .collect();
                        if mid_to_end.len() >= 3 {
                            let negative_count = mid_to_end.iter().filter(|&&v| v < 0.0).count();
                            prop_assert!(
                                negative_count >= mid_to_end.len() / 2,
                                "DTI should be mostly negative when prices consistently fall"
                            );
                        }
                    }
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    #[cfg(not(feature = "proptest"))]
    fn check_dti_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        Ok(())
    }

    macro_rules! generate_all_dti_tests {
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
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                $(
                    #[test]
                    fn [<$test_fn _simd128_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _simd128_f64>]), Kernel::Scalar);
                    }
                )*
            }
        }
    }
    generate_all_dti_tests!(
        check_dti_partial_params,
        check_dti_accuracy,
        check_dti_default_candles,
        check_dti_zero_period,
        check_dti_period_exceeds_length,
        check_dti_all_nan,
        check_dti_empty_data,
        check_dti_streaming,
        check_dti_length_mismatch,
        check_dti_wasm_kernel_fallback,
        check_dti_no_poison,
        check_dti_property
    );
    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = DtiBatchBuilder::new().kernel(kernel).apply_candles(&c)?;
        let def = DtiParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [
            -39.0091620347991,
            -39.75219264093014,
            -40.53941417932286,
            -41.2787749205189,
            -42.93758699380749,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-6,
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
            (1, 5, 1, 1, 5, 1, 1, 5, 1),
            (5, 15, 5, 5, 15, 5, 5, 15, 5),
            (10, 30, 10, 10, 20, 10, 5, 15, 5),
            (14, 14, 0, 10, 10, 0, 1, 10, 1),
            (1, 20, 5, 10, 10, 0, 5, 5, 0),
            (20, 50, 15, 15, 30, 15, 10, 20, 10),
            (2, 6, 2, 8, 12, 2, 3, 9, 3),
            (50, 100, 25, 30, 60, 30, 20, 40, 20),
            (14, 14, 0, 5, 20, 5, 5, 20, 5),
            (5, 5, 0, 5, 5, 0, 1, 10, 1),
        ];

        for (cfg_idx, &(r_start, r_end, r_step, s_start, s_end, s_step, u_start, u_end, u_step)) in
            test_configs.iter().enumerate()
        {
            let output = DtiBatchBuilder::new()
                .kernel(kernel)
                .r_range(r_start, r_end, r_step)
                .s_range(s_start, s_end, s_step)
                .u_range(u_start, u_end, u_step)
                .apply_candles(&c)?;

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
						 at row {} col {} (flat index {}) with params: r={}, s={}, u={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.r.unwrap_or(14),
                        combo.s.unwrap_or(10),
                        combo.u.unwrap_or(5)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: r={}, s={}, u={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.r.unwrap_or(14),
                        combo.s.unwrap_or(10),
                        combo.u.unwrap_or(5)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: r={}, s={}, u={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.r.unwrap_or(14),
                        combo.s.unwrap_or(10),
                        combo.u.unwrap_or(5)
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
    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);

    #[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
    #[test]
    fn test_dti_into_matches_api() -> Result<(), Box<dyn Error>> {
        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b)
        }

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = DtiInput::with_default_candles(&candles);

        let baseline = dti(&input)?;

        let mut out = vec![0.0f64; candles.close.len()];
        dti_into(&input, &mut out)?;

        assert_eq!(baseline.values.len(), out.len());
        for i in 0..out.len() {
            assert!(
                eq_or_both_nan(baseline.values[i], out[i]),
                "Mismatch at index {}: api={} into={}",
                i,
                baseline.values[i],
                out[i]
            );
        }
        Ok(())
    }
}
