#[cfg(all(feature = "python", feature = "cuda"))]
use crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::context::Context;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::memory::DeviceBuffer;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
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

#[cfg(all(feature = "python", feature = "cuda"))]
use std::sync::Arc;

use crate::utilities::data_loader::Candles;
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
use std::error::Error;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum SarData<'a> {
    Candles { candles: &'a Candles },
    Slices { high: &'a [f64], low: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct SarOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct SarParams {
    pub acceleration: Option<f64>,
    pub maximum: Option<f64>,
}

impl Default for SarParams {
    fn default() -> Self {
        Self {
            acceleration: Some(0.02),
            maximum: Some(0.2),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SarInput<'a> {
    pub data: SarData<'a>,
    pub params: SarParams,
}

impl<'a> SarInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: SarParams) -> Self {
        Self {
            data: SarData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(high: &'a [f64], low: &'a [f64], params: SarParams) -> Self {
        Self {
            data: SarData::Slices { high, low },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: SarData::Candles { candles },
            params: SarParams::default(),
        }
    }

    #[inline]
    pub fn get_acceleration(&self) -> f64 {
        self.params.acceleration.unwrap_or(0.02)
    }

    #[inline]
    pub fn get_maximum(&self) -> f64 {
        self.params.maximum.unwrap_or(0.2)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SarBuilder {
    acceleration: Option<f64>,
    maximum: Option<f64>,
    kernel: Kernel,
}

impl Default for SarBuilder {
    fn default() -> Self {
        Self {
            acceleration: None,
            maximum: None,
            kernel: Kernel::Auto,
        }
    }
}

impl SarBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn acceleration(mut self, v: f64) -> Self {
        self.acceleration = Some(v);
        self
    }
    #[inline(always)]
    pub fn maximum(mut self, v: f64) -> Self {
        self.maximum = Some(v);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<SarOutput, SarError> {
        let params = SarParams {
            acceleration: self.acceleration,
            maximum: self.maximum,
        };
        let input = SarInput::from_candles(c, params);
        sar_with_kernel(&input, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<SarOutput, SarError> {
        let params = SarParams {
            acceleration: self.acceleration,
            maximum: self.maximum,
        };
        let input = SarInput::from_slices(high, low, params);
        sar_with_kernel(&input, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<SarStream, SarError> {
        let params = SarParams {
            acceleration: self.acceleration,
            maximum: self.maximum,
        };
        SarStream::try_new(params)
    }
}

#[derive(Debug, Error)]
pub enum SarError {
    #[error("sar: Empty data provided.")]
    EmptyInputData,
    #[error("sar: All values are NaN.")]
    AllValuesNaN,
    #[error("sar: Not enough valid data. needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("sar: Invalid acceleration: {acceleration}")]
    InvalidAcceleration { acceleration: f64 },
    #[error("sar: Invalid maximum: {maximum}")]
    InvalidMaximum { maximum: f64 },
    #[error("sar: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("sar: Invalid parameter range: start = {start}, end = {end}, step = {step}")]
    InvalidRange { start: f64, end: f64, step: f64 },
    #[error("sar: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline]
pub fn sar(input: &SarInput) -> Result<SarOutput, SarError> {
    sar_with_kernel(input, Kernel::Auto)
}

pub fn sar_with_kernel(input: &SarInput, kernel: Kernel) -> Result<SarOutput, SarError> {
    let (high, low) = match &input.data {
        SarData::Candles { candles } => (candles.high.as_slice(), candles.low.as_slice()),
        SarData::Slices { high, low } => (*high, *low),
    };

    if high.is_empty() || low.is_empty() {
        return Err(SarError::EmptyInputData);
    }

    let min_len = high.len().min(low.len());
    let (high, low) = (&high[..min_len], &low[..min_len]);

    let first_valid_idx = high
        .iter()
        .zip(low.iter())
        .position(|(&h, &l)| !h.is_nan() && !l.is_nan());
    let first = match first_valid_idx {
        Some(idx) => idx,
        None => return Err(SarError::AllValuesNaN),
    };

    if (high.len() - first) < 2 {
        return Err(SarError::NotEnoughValidData {
            needed: 2,
            valid: high.len() - first,
        });
    }

    let acceleration = input.get_acceleration();
    let maximum = input.get_maximum();

    if !(acceleration > 0.0) || acceleration.is_nan() || acceleration.is_infinite() {
        return Err(SarError::InvalidAcceleration { acceleration });
    }
    if !(maximum > 0.0) || maximum.is_nan() || maximum.is_infinite() {
        return Err(SarError::InvalidMaximum { maximum });
    }

    let mut out = alloc_with_nan_prefix(high.len(), first + 1);

    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                sar_scalar(high, low, first, acceleration, maximum, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                sar_avx2(high, low, first, acceleration, maximum, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                sar_avx512(high, low, first, acceleration, maximum, &mut out)
            }
            _ => unreachable!(),
        }
    }

    Ok(SarOutput { values: out })
}

#[inline]
pub fn sar_into_slice(dst: &mut [f64], input: &SarInput, kern: Kernel) -> Result<(), SarError> {
    let (high, low) = match &input.data {
        SarData::Candles { candles } => (candles.high.as_slice(), candles.low.as_slice()),
        SarData::Slices { high, low } => (*high, *low),
    };

    if high.is_empty() || low.is_empty() {
        return Err(SarError::EmptyInputData);
    }

    let expected_len = high.len().min(low.len());
    if dst.len() != expected_len {
        return Err(SarError::OutputLengthMismatch {
            expected: expected_len,
            got: dst.len(),
        });
    }

    let (high, low) = (&high[..expected_len], &low[..expected_len]);

    let first_valid_idx = high
        .iter()
        .zip(low.iter())
        .position(|(&h, &l)| !h.is_nan() && !l.is_nan());
    let first = match first_valid_idx {
        Some(idx) => idx,
        None => return Err(SarError::AllValuesNaN),
    };

    if (high.len() - first) < 2 {
        return Err(SarError::NotEnoughValidData {
            valid: high.len() - first,
            needed: 2,
        });
    }

    let acceleration = input.params.acceleration.unwrap_or(0.02);
    let maximum = input.params.maximum.unwrap_or(0.2);

    if acceleration <= 0.0 || acceleration.is_nan() || acceleration.is_infinite() {
        return Err(SarError::InvalidAcceleration { acceleration });
    }
    if maximum <= 0.0 || maximum.is_nan() || maximum.is_infinite() {
        return Err(SarError::InvalidMaximum { maximum });
    }

    for v in &mut dst[..first.saturating_add(1)] {
        *v = f64::from_bits(0x7ff8_0000_0000_0000);
    }

    let chosen = match kern {
        Kernel::Auto => Kernel::Scalar,
        x => x,
    };

    match chosen {
        Kernel::Scalar => sar_scalar(high, low, first, acceleration, maximum, dst),
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 | Kernel::Avx2Batch => sar_avx2(high, low, first, acceleration, maximum, dst),
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 | Kernel::Avx512Batch => {
            sar_avx512(high, low, first, acceleration, maximum, dst)
        }

        _ => sar_scalar(high, low, first, acceleration, maximum, dst),
    }
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn sar_into(input: &SarInput, out: &mut [f64]) -> Result<(), SarError> {
    sar_into_slice(out, input, Kernel::Auto)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn sar_avx512(
    high: &[f64],
    low: &[f64],
    first_valid: usize,
    acceleration: f64,
    maximum: f64,
    out: &mut [f64],
) {
    sar_avx2(high, low, first_valid, acceleration, maximum, out)
}

#[inline]
pub fn sar_scalar(
    high: &[f64],
    low: &[f64],
    first: usize,
    acceleration: f64,
    maximum: f64,
    out: &mut [f64],
) {
    let len = high.len();
    let i0 = first;
    let i1 = i0 + 1;

    if i1 >= len || i1 >= low.len() || i1 >= out.len() {
        return;
    }

    let h0 = high[i0];
    let h1 = high[i1];
    let l0 = low[i0];
    let l1 = low[i1];

    let mut trend_up = h1 > h0;
    let mut sar = if trend_up { l0 } else { h0 };
    let mut ep = if trend_up { h1 } else { l1 };
    let mut acc = acceleration;

    out[i0] = f64::NAN;
    out[i1] = sar;

    let mut low_prev2 = l0;
    let mut low_prev = l1;
    let mut high_prev2 = h0;
    let mut high_prev = h1;

    let mut i = i1 + 1;
    while i < len {
        let hi = high[i];
        let lo = low[i];

        let mut next_sar = acc.mul_add(ep - sar, sar);

        if trend_up {
            if lo < next_sar {
                trend_up = false;
                next_sar = ep;
                ep = lo;
                acc = acceleration;
            } else {
                if hi > ep {
                    ep = hi;
                    acc = (acc + acceleration).min(maximum);
                }
                next_sar = next_sar.min(low_prev).min(low_prev2);
            }
        } else {
            if hi > next_sar {
                trend_up = true;
                next_sar = ep;
                ep = hi;
                acc = acceleration;
            } else {
                if lo < ep {
                    ep = lo;
                    acc = (acc + acceleration).min(maximum);
                }
                next_sar = next_sar.max(high_prev).max(high_prev2);
            }
        }

        out[i] = next_sar;
        sar = next_sar;

        low_prev2 = low_prev;
        low_prev = lo;
        high_prev2 = high_prev;
        high_prev = hi;

        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn sar_avx2(
    high: &[f64],
    low: &[f64],
    first_valid: usize,
    acceleration: f64,
    maximum: f64,
    out: &mut [f64],
) {
    let len = high.len();
    let i0 = first_valid;
    let i1 = i0 + 1;

    if i1 >= len || i1 >= low.len() || i1 >= out.len() {
        return;
    }

    unsafe {
        let h0 = *high.get_unchecked(i0);
        let h1 = *high.get_unchecked(i1);
        let l0 = *low.get_unchecked(i0);
        let l1 = *low.get_unchecked(i1);

        let mut trend_up = h1 > h0;
        let mut sar = if trend_up { l0 } else { h0 };
        let mut ep = if trend_up { h1 } else { l1 };
        let mut acc = acceleration;

        *out.get_unchecked_mut(i0) = f64::NAN;
        *out.get_unchecked_mut(i1) = sar;

        let mut low_prev2 = l0;
        let mut low_prev = l1;
        let mut high_prev2 = h0;
        let mut high_prev = h1;

        let mut i = i1 + 1;
        while i < len {
            let hi = *high.get_unchecked(i);
            let lo = *low.get_unchecked(i);

            let mut next_sar = acc.mul_add(ep - sar, sar);

            if trend_up {
                if lo < next_sar {
                    trend_up = false;
                    next_sar = ep;
                    ep = lo;
                    acc = acceleration;
                } else {
                    if hi > ep {
                        ep = hi;
                        acc = (acc + acceleration).min(maximum);
                    }
                    next_sar = next_sar.min(low_prev).min(low_prev2);
                }
            } else {
                if hi > next_sar {
                    trend_up = true;
                    next_sar = ep;
                    ep = hi;
                    acc = acceleration;
                } else {
                    if lo < ep {
                        ep = lo;
                        acc = (acc + acceleration).min(maximum);
                    }
                    next_sar = next_sar.max(high_prev).max(high_prev2);
                }
            }

            *out.get_unchecked_mut(i) = next_sar;
            sar = next_sar;

            low_prev2 = low_prev;
            low_prev = lo;
            high_prev2 = high_prev;
            high_prev = hi;

            i += 1;
        }
    }
}

#[cfg(all(feature = "simd128", target_arch = "wasm32"))]
#[inline]
pub fn sar_simd128(
    high: &[f64],
    low: &[f64],
    first_valid: usize,
    acceleration: f64,
    maximum: f64,
    out: &mut [f64],
) {
    sar_scalar(high, low, first_valid, acceleration, maximum, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn sar_avx512_short(
    high: &[f64],
    low: &[f64],
    first_valid: usize,
    acceleration: f64,
    maximum: f64,
    out: &mut [f64],
) {
    sar_avx2(high, low, first_valid, acceleration, maximum, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn sar_avx512_long(
    high: &[f64],
    low: &[f64],
    first_valid: usize,
    acceleration: f64,
    maximum: f64,
    out: &mut [f64],
) {
    sar_avx2(high, low, first_valid, acceleration, maximum, out)
}

#[derive(Debug, Clone)]
pub struct SarStream {
    acceleration: f64,
    maximum: f64,
    state: Option<StreamState>,

    idx: usize,
}

#[derive(Debug, Clone)]
struct StreamState {
    trend_up: bool,
    sar: f64,
    ep: f64,
    acc: f64,

    prev_high: f64,
    prev_high2: f64,
    prev_low: f64,
    prev_low2: f64,
}

impl SarStream {
    pub fn try_new(params: SarParams) -> Result<Self, SarError> {
        let acceleration = params.acceleration.unwrap_or(0.02);
        let maximum = params.maximum.unwrap_or(0.2);

        if !(acceleration > 0.0) || !acceleration.is_finite() {
            return Err(SarError::InvalidAcceleration { acceleration });
        }
        if !(maximum > 0.0) || !maximum.is_finite() {
            return Err(SarError::InvalidMaximum { maximum });
        }

        Ok(Self {
            acceleration,
            maximum,
            state: None,
            idx: 0,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64) -> Option<f64> {
        if !high.is_finite() || !low.is_finite() {
            return None;
        }

        match self.state.as_mut() {
            None => {
                self.state = Some(StreamState {
                    trend_up: false,
                    sar: f64::NAN,
                    ep: f64::NAN,
                    acc: self.acceleration,
                    prev_high: high,
                    prev_high2: high,
                    prev_low: low,
                    prev_low2: low,
                });
                self.idx = 1;
                None
            }

            Some(st) if self.idx == 1 => {
                let trend_up = high > st.prev_high;

                let sar = if trend_up { st.prev_low } else { st.prev_high };
                let ep = if trend_up { high } else { low };

                st.prev_high2 = st.prev_high;
                st.prev_low2 = st.prev_low;
                st.prev_high = high;
                st.prev_low = low;

                st.trend_up = trend_up;
                st.sar = sar;
                st.ep = ep;
                st.acc = self.acceleration;

                self.idx = 2;
                Some(sar)
            }

            Some(st) => {
                let mut next_sar = st.acc.mul_add(st.ep - st.sar, st.sar);

                if st.trend_up {
                    if low < next_sar {
                        st.trend_up = false;
                        next_sar = st.ep;
                        st.ep = low;
                        st.acc = self.acceleration;
                    } else {
                        if high > st.ep {
                            st.ep = high;
                            st.acc = (st.acc + self.acceleration).min(self.maximum);
                        }
                        next_sar = min3(next_sar, st.prev_low, st.prev_low2);
                    }
                } else {
                    if high > next_sar {
                        st.trend_up = true;
                        next_sar = st.ep;
                        st.ep = high;
                        st.acc = self.acceleration;
                    } else {
                        if low < st.ep {
                            st.ep = low;
                            st.acc = (st.acc + self.acceleration).min(self.maximum);
                        }
                        next_sar = max3(next_sar, st.prev_high, st.prev_high2);
                    }
                }

                st.prev_high2 = st.prev_high;
                st.prev_low2 = st.prev_low;
                st.prev_high = high;
                st.prev_low = low;

                st.sar = next_sar;
                self.idx += 1;
                Some(next_sar)
            }
        }
    }
}

#[inline(always)]
fn min3(a: f64, b: f64, c: f64) -> f64 {
    a.min(b.min(c))
}

#[inline(always)]
fn max3(a: f64, b: f64, c: f64) -> f64 {
    a.max(b.max(c))
}

#[derive(Clone, Debug)]
pub struct SarBatchRange {
    pub acceleration: (f64, f64, f64),
    pub maximum: (f64, f64, f64),
}

impl Default for SarBatchRange {
    fn default() -> Self {
        Self {
            acceleration: (0.02, 0.02, 0.0),
            maximum: (0.2, 0.449, 0.001),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SarBatchBuilder {
    range: SarBatchRange,
    kernel: Kernel,
}

impl SarBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    pub fn acceleration_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.acceleration = (start, end, step);
        self
    }
    pub fn acceleration_static(mut self, x: f64) -> Self {
        self.range.acceleration = (x, x, 0.0);
        self
    }
    pub fn maximum_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.maximum = (start, end, step);
        self
    }
    pub fn maximum_static(mut self, x: f64) -> Self {
        self.range.maximum = (x, x, 0.0);
        self
    }

    pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<SarBatchOutput, SarError> {
        sar_batch_with_kernel(high, low, &self.range, self.kernel)
    }

    pub fn with_default_slices(
        high: &[f64],
        low: &[f64],
        k: Kernel,
    ) -> Result<SarBatchOutput, SarError> {
        SarBatchBuilder::new().kernel(k).apply_slices(high, low)
    }

    pub fn apply_candles(self, c: &Candles) -> Result<SarBatchOutput, SarError> {
        self.apply_slices(&c.high, &c.low)
    }

    pub fn with_default_candles(c: &Candles) -> Result<SarBatchOutput, SarError> {
        SarBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c)
    }
}

pub fn sar_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    sweep: &SarBatchRange,
    k: Kernel,
) -> Result<SarBatchOutput, SarError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(SarError::InvalidKernelForBatch(other)),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    sar_batch_par_slice(high, low, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct SarBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<SarParams>,
    pub rows: usize,
    pub cols: usize,
}
impl SarBatchOutput {
    pub fn row_for_params(&self, p: &SarParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            (c.acceleration.unwrap_or(0.02) - p.acceleration.unwrap_or(0.02)).abs() < 1e-12
                && (c.maximum.unwrap_or(0.2) - p.maximum.unwrap_or(0.2)).abs() < 1e-12
        })
    }
    pub fn values_for(&self, p: &SarParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn axis_f64_checked(axis: (f64, f64, f64)) -> Result<Vec<f64>, SarError> {
    let (start, end, step) = axis;
    if !step.is_finite() {
        return Err(SarError::InvalidRange { start, end, step });
    }
    if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
        return Ok(vec![start]);
    }

    let mut v = Vec::new();
    let tol = step.abs() * 1e-12;

    if step > 0.0 {
        if start <= end {
            let mut x = start;
            while x <= end + tol {
                v.push(x);
                x += step;
            }
        } else {
            let mut x = start;
            while x >= end - tol {
                v.push(x);
                x -= step;
            }
        }
    } else {
        if start >= end {
            let mut x = start;
            while x >= end - tol {
                v.push(x);
                x += step;
            }
        } else {
            return Err(SarError::InvalidRange { start, end, step });
        }
    }

    if v.is_empty() {
        Err(SarError::InvalidRange { start, end, step })
    } else {
        Ok(v)
    }
}

#[inline(always)]
fn expand_grid(r: &SarBatchRange) -> Result<Vec<SarParams>, SarError> {
    let accs = axis_f64_checked(r.acceleration)?;
    let maxs = axis_f64_checked(r.maximum)?;

    let capacity = accs
        .len()
        .checked_mul(maxs.len())
        .ok_or(SarError::InvalidRange {
            start: r.acceleration.0,
            end: r.acceleration.1,
            step: r.acceleration.2,
        })?;

    let mut out = Vec::with_capacity(capacity);
    for &a in &accs {
        for &m in &maxs {
            out.push(SarParams {
                acceleration: Some(a),
                maximum: Some(m),
            });
        }
    }
    Ok(out)
}

#[inline(always)]
pub fn sar_batch_slice(
    high: &[f64],
    low: &[f64],
    sweep: &SarBatchRange,
    kern: Kernel,
) -> Result<SarBatchOutput, SarError> {
    sar_batch_inner(high, low, sweep, kern, false)
}

#[inline(always)]
pub fn sar_batch_par_slice(
    high: &[f64],
    low: &[f64],
    sweep: &SarBatchRange,
    kern: Kernel,
) -> Result<SarBatchOutput, SarError> {
    sar_batch_inner(high, low, sweep, kern, true)
}

#[inline(always)]
fn sar_batch_inner(
    high: &[f64],
    low: &[f64],
    sweep: &SarBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<SarBatchOutput, SarError> {
    let combos = expand_grid(sweep)?;

    let min_len = high.len().min(low.len());
    let (high, low) = (&high[..min_len], &low[..min_len]);
    let first = high
        .iter()
        .zip(low.iter())
        .position(|(&h, &l)| !h.is_nan() && !l.is_nan())
        .ok_or(SarError::AllValuesNaN)?;

    if high.len() - first < 2 {
        return Err(SarError::NotEnoughValidData {
            needed: 2,
            valid: high.len() - first,
        });
    }
    let rows = combos.len();
    let cols = high.len();

    let mut buf_mu = make_uninit_matrix(rows, cols);

    let warm = vec![first + 1; rows];
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
    let values: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let p = &combos[row];
        match kern {
            Kernel::Scalar => sar_row_scalar(
                high,
                low,
                first,
                p.acceleration.unwrap(),
                p.maximum.unwrap(),
                out_row,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => sar_row_avx2(
                high,
                low,
                first,
                p.acceleration.unwrap(),
                p.maximum.unwrap(),
                out_row,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => sar_row_avx512(
                high,
                low,
                first,
                p.acceleration.unwrap(),
                p.maximum.unwrap(),
                out_row,
            ),
            _ => unreachable!(),
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

    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(SarBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn sar_row_scalar(
    high: &[f64],
    low: &[f64],
    first: usize,
    acceleration: f64,
    maximum: f64,
    out: &mut [f64],
) {
    sar_scalar(high, low, first, acceleration, maximum, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn sar_row_avx2(
    high: &[f64],
    low: &[f64],
    first: usize,
    acceleration: f64,
    maximum: f64,
    out: &mut [f64],
) {
    sar_avx2(high, low, first, acceleration, maximum, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn sar_row_avx512(
    high: &[f64],
    low: &[f64],
    first: usize,
    acceleration: f64,
    maximum: f64,
    out: &mut [f64],
) {
    sar_avx2(high, low, first, acceleration, maximum, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn sar_row_avx512_short(
    high: &[f64],
    low: &[f64],
    first: usize,
    acceleration: f64,
    maximum: f64,
    out: &mut [f64],
) {
    sar_avx2(high, low, first, acceleration, maximum, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn sar_row_avx512_long(
    high: &[f64],
    low: &[f64],
    first: usize,
    acceleration: f64,
    maximum: f64,
    out: &mut [f64],
) {
    sar_avx2(high, low, first, acceleration, maximum, out)
}

#[inline(always)]
fn expand_grid_for_test(r: &SarBatchRange) -> Vec<SarParams> {
    expand_grid(r).expect("expand_grid_for_test should not fail")
}

#[cfg(feature = "python")]
#[pyfunction(name = "sar")]
#[pyo3(signature = (high, low, acceleration=None, maximum=None, kernel=None))]
pub fn sar_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    acceleration: Option<f64>,
    maximum: Option<f64>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;

    let min_len = high_slice.len().min(low_slice.len());
    let high_trimmed = &high_slice[..min_len];
    let low_trimmed = &low_slice[..min_len];

    let kern = validate_kernel(kernel, false)?;

    let params = SarParams {
        acceleration,
        maximum,
    };
    let input = SarInput::from_slices(high_trimmed, low_trimmed, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| sar_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "SarStream")]
pub struct SarStreamPy {
    stream: SarStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl SarStreamPy {
    #[new]
    #[pyo3(signature = (acceleration=None, maximum=None))]
    fn new(acceleration: Option<f64>, maximum: Option<f64>) -> PyResult<Self> {
        let params = SarParams {
            acceleration,
            maximum,
        };
        let stream =
            SarStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(SarStreamPy { stream })
    }

    fn update(&mut self, high: f64, low: f64) -> Option<f64> {
        self.stream.update(high, low)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "sar_batch")]
#[pyo3(signature = (high, low, acceleration_range, maximum_range, kernel=None))]
pub fn sar_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    acceleration_range: (f64, f64, f64),
    maximum_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;

    let min_len = high_slice.len().min(low_slice.len());
    let (high_slice, low_slice) = (&high_slice[..min_len], &low_slice[..min_len]);

    let sweep = SarBatchRange {
        acceleration: acceleration_range,
        maximum: maximum_range,
    };
    let combos = expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = min_len;
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("sar_batch_py: size overflow in rows*cols"))?;

    let out_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    let kern = validate_kernel(kernel, true)?;
    py.allow_threads(|| {
        let k = match kern {
            Kernel::Auto => detect_best_batch_kernel(),
            k => k,
        };
        sar_batch_inner_into_noalloc(high_slice, low_slice, &sweep, k, true, slice_out)
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "accelerations",
        combos
            .iter()
            .map(|p| p.acceleration.unwrap_or(0.02))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "maximums",
        combos
            .iter()
            .map(|p| p.maximum.unwrap_or(0.2))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    Ok(dict)
}

#[cfg(feature = "python")]
fn sar_batch_inner_into_noalloc(
    high: &[f64],
    low: &[f64],
    sweep: &SarBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<SarParams>, SarError> {
    let combos = expand_grid(sweep)?;

    let first = high
        .iter()
        .zip(low.iter())
        .position(|(&h, &l)| !h.is_nan() && !l.is_nan())
        .ok_or(SarError::AllValuesNaN)?;
    if high.len() - first < 2 {
        return Err(SarError::NotEnoughValidData {
            needed: 2,
            valid: high.len() - first,
        });
    }

    let rows = combos.len();
    let cols = high.len();
    let expected = rows.checked_mul(cols).ok_or(SarError::InvalidRange {
        start: sweep.acceleration.0,
        end: sweep.acceleration.1,
        step: sweep.acceleration.2,
    })?;
    if out.len() != expected {
        return Err(SarError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    unsafe {
        let out_mu = std::slice::from_raw_parts_mut(
            out.as_mut_ptr() as *mut std::mem::MaybeUninit<f64>,
            out.len(),
        );
        let warm: Vec<usize> = vec![first + 1; rows];
        init_matrix_prefixes(out_mu, cols, &warm);
    }

    let simd = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    };
    let exec = |row: usize, row_out: &mut [f64]| {
        let p = &combos[row];
        match simd {
            Kernel::Scalar | Kernel::ScalarBatch => unsafe {
                sar_row_scalar(
                    high,
                    low,
                    first,
                    p.acceleration.unwrap(),
                    p.maximum.unwrap(),
                    row_out,
                );
            },
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => unsafe {
                sar_row_avx2(
                    high,
                    low,
                    first,
                    p.acceleration.unwrap(),
                    p.maximum.unwrap(),
                    row_out,
                );
            },
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => unsafe {
                sar_row_avx512(
                    high,
                    low,
                    first,
                    p.acceleration.unwrap(),
                    p.maximum.unwrap(),
                    row_out,
                );
            },
            _ => unsafe {
                sar_row_scalar(
                    high,
                    low,
                    first,
                    p.acceleration.unwrap(),
                    p.maximum.unwrap(),
                    row_out,
                );
            },
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            out.par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, r)| exec(row, r));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, r) in out.chunks_mut(cols).enumerate() {
                exec(row, r);
            }
        }
    } else {
        for (row, r) in out.chunks_mut(cols).enumerate() {
            exec(row, r);
        }
    }

    Ok(combos)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn sar_js(
    high: &[f64],
    low: &[f64],
    acceleration: f64,
    maximum: f64,
) -> Result<Vec<f64>, JsValue> {
    let min_len = high.len().min(low.len());
    let (high, low) = (&high[..min_len], &low[..min_len]);

    let params = SarParams {
        acceleration: Some(acceleration),
        maximum: Some(maximum),
    };
    let input = SarInput::from_slices(high, low, params);

    let mut output = vec![0.0; min_len];

    sar_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", unsendable)]
pub struct SarDeviceArrayF32Py {
    pub(crate) buf: Option<DeviceBuffer<f32>>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) _ctx: Arc<Context>,
    pub(crate) device_id: u32,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl SarDeviceArrayF32Py {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("shape", (self.rows, self.cols))?;
        d.set_item("typestr", "<f4")?;
        d.set_item(
            "strides",
            (
                self.cols * std::mem::size_of::<f32>(),
                std::mem::size_of::<f32>(),
            ),
        )?;
        let buf = self
            .buf
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("buffer already exported via __dlpack__"))?;
        let ptr = buf.as_device_ptr().as_raw() as usize;
        d.set_item("data", (ptr, false))?;

        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        (2, self.device_id as i32)
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
        let (kdl, alloc_dev) = self.__dlpack_device__();
        if let Some(dev_obj) = dl_device.as_ref() {
            if let Ok((dev_ty, dev_id)) = dev_obj.extract::<(i32, i32)>(py) {
                if dev_ty != kdl || dev_id != alloc_dev {
                    let wants_copy = copy
                        .as_ref()
                        .and_then(|c| c.extract::<bool>(py).ok())
                        .unwrap_or(false);
                    if wants_copy {
                        return Err(PyValueError::new_err(
                            "cross-device DLPack copy is not implemented for SarDeviceArrayF32Py",
                        ));
                    } else {
                        return Err(PyValueError::new_err(
                            "requested dl_device does not match SAR producer device",
                        ));
                    }
                }
            }
        }
        let _ = stream;

        let buf = self
            .buf
            .take()
            .ok_or_else(|| PyValueError::new_err("__dlpack__ may only be called once"))?;

        let rows = self.rows;
        let cols = self.cols;

        let max_version_bound = max_version.map(|obj| obj.into_bound(py));

        export_f32_cuda_dlpack_2d(py, buf, rows, cols, alloc_dev, max_version_bound)
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "sar_cuda_batch_dev")]
#[pyo3(signature = (high_f32, low_f32, acceleration_range, maximum_range, device_id=0))]
pub fn sar_cuda_batch_dev_py(
    py: Python<'_>,
    high_f32: numpy::PyReadonlyArray1<'_, f32>,
    low_f32: numpy::PyReadonlyArray1<'_, f32>,
    acceleration_range: (f64, f64, f64),
    maximum_range: (f64, f64, f64),
    device_id: usize,
) -> PyResult<SarDeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    use crate::cuda::CudaSar;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let h = high_f32.as_slice()?;
    let l = low_f32.as_slice()?;
    if h.len() != l.len() {
        return Err(PyValueError::new_err("high/low length mismatch"));
    }
    let sweep = SarBatchRange {
        acceleration: acceleration_range,
        maximum: maximum_range,
    };
    let (buf, rows, cols, ctx, dev_id) = py.allow_threads(|| {
        let cuda = CudaSar::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let (dev, _combos) = cuda
            .sar_batch_dev(h, l, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.context_arc();
        Ok::<_, pyo3::PyErr>((dev.buf, dev.rows, dev.cols, ctx, cuda.device_id()))
    })?;
    Ok(SarDeviceArrayF32Py {
        buf: Some(buf),
        rows,
        cols,
        _ctx: ctx,
        device_id: dev_id,
    })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "sar_cuda_many_series_one_param_dev")]
#[pyo3(signature = (high_tm_f32, low_tm_f32, cols, rows, acceleration, maximum, device_id=0))]
pub fn sar_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    high_tm_f32: numpy::PyReadonlyArray1<'_, f32>,
    low_tm_f32: numpy::PyReadonlyArray1<'_, f32>,
    cols: usize,
    rows: usize,
    acceleration: f64,
    maximum: f64,
    device_id: usize,
) -> PyResult<SarDeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    use crate::cuda::CudaSar;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let h = high_tm_f32.as_slice()?;
    let l = low_tm_f32.as_slice()?;
    let expected = cols.checked_mul(rows).ok_or_else(|| {
        PyValueError::new_err("sar_cuda_many_series_one_param_dev: size overflow in cols*rows")
    })?;
    if expected != h.len() || h.len() != l.len() {
        return Err(PyValueError::new_err(
            "time‑major inputs must be equal length and cols*rows",
        ));
    }
    let params = SarParams {
        acceleration: Some(acceleration),
        maximum: Some(maximum),
    };
    let (buf, r_out, c_out, ctx, dev_id) = py.allow_threads(|| {
        let cuda = CudaSar::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dev = cuda
            .sar_many_series_one_param_time_major_dev(h, l, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.context_arc();
        Ok::<_, pyo3::PyErr>((dev.buf, dev.rows, dev.cols, ctx, cuda.device_id()))
    })?;
    Ok(SarDeviceArrayF32Py {
        buf: Some(buf),
        rows: r_out,
        cols: c_out,
        _ctx: ctx,
        device_id: dev_id,
    })
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn sar_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    acceleration: f64,
    maximum: f64,
) -> Result<(), JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to sar_into"));
    }

    unsafe {
        let high_slice = std::slice::from_raw_parts(high_ptr, len);
        let low_slice = std::slice::from_raw_parts(low_ptr, len);

        let params = SarParams {
            acceleration: Some(acceleration),
            maximum: Some(maximum),
        };
        let input = SarInput::from_slices(high_slice, low_slice, params);

        if high_ptr == out_ptr || low_ptr == out_ptr {
            let mut temp = vec![0.0; len];
            sar_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            sar_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn sar_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn sar_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn sar_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    acc_start: f64,
    acc_end: f64,
    acc_step: f64,
    max_start: f64,
    max_end: f64,
    max_step: f64,
) -> Result<usize, JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to sar_batch_into"));
    }
    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let sweep = SarBatchRange {
            acceleration: (acc_start, acc_end, acc_step),
            maximum: (max_start, max_end, max_step),
        };
        let combos = expand_grid(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let cols = len;
        let total = rows
            .checked_mul(cols)
            .ok_or_else(|| JsValue::from_str("sar_batch_into: size overflow in rows*cols"))?;

        let out = std::slice::from_raw_parts_mut(out_ptr, total);

        sar_batch_inner_into_noalloc_wasm(high, low, &sweep, Kernel::Scalar, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
fn sar_batch_inner_into_noalloc_wasm(
    high: &[f64],
    low: &[f64],
    sweep: &SarBatchRange,
    _kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<SarParams>, SarError> {
    let combos = expand_grid(sweep)?;

    let min_len = high.len().min(low.len());
    let (high, low) = (&high[..min_len], &low[..min_len]);

    let first = high
        .iter()
        .zip(low.iter())
        .position(|(&h, &l)| !h.is_nan() && !l.is_nan())
        .ok_or(SarError::AllValuesNaN)?;
    if high.len() - first < 2 {
        return Err(SarError::NotEnoughValidData {
            needed: 2,
            valid: high.len() - first,
        });
    }

    let rows = combos.len();
    let cols = high.len();
    let expected = rows.checked_mul(cols).ok_or(SarError::InvalidRange {
        start: sweep.acceleration.0,
        end: sweep.acceleration.1,
        step: sweep.acceleration.2,
    })?;
    if out.len() != expected {
        return Err(SarError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    unsafe {
        let out_mu = std::slice::from_raw_parts_mut(
            out.as_mut_ptr() as *mut std::mem::MaybeUninit<f64>,
            out.len(),
        );
        let warm: Vec<usize> = vec![first + 1; rows];
        init_matrix_prefixes(out_mu, cols, &warm);
    }

    let exec = |row: usize, row_out: &mut [f64]| unsafe {
        let p = &combos[row];
        sar_row_scalar(
            high,
            low,
            first,
            p.acceleration.unwrap(),
            p.maximum.unwrap(),
            row_out,
        );
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            out.par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, r)| exec(row, r));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, r) in out.chunks_mut(cols).enumerate() {
                exec(row, r);
            }
        }
    } else {
        for (row, r) in out.chunks_mut(cols).enumerate() {
            exec(row, r);
        }
    }

    Ok(combos)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct SarBatchConfig {
    pub acceleration_range: (f64, f64, f64),
    pub maximum_range: (f64, f64, f64),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct SarBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<SarParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = sar_batch)]
pub fn sar_batch_unified_js(
    high: &[f64],
    low: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: SarBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = SarBatchRange {
        acceleration: config.acceleration_range,
        maximum: config.maximum_range,
    };

    let kernel = if cfg!(target_arch = "wasm32") {
        Kernel::Scalar
    } else {
        detect_best_batch_kernel()
    };
    let output = sar_batch_inner(high, low, &sweep, kernel, false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = SarBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };

    serde_wasm_bindgen::to_value(&js_output).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_sar_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = SarParams {
            acceleration: None,
            maximum: None,
        };
        let input = SarInput::from_candles(&candles, default_params);
        let output = sar_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_sar_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = SarInput::from_candles(&candles, SarParams::default());
        let result = sar_with_kernel(&input, kernel)?;
        let expected_last_five = [
            60370.00224209362,
            60220.362107568006,
            60079.70038111392,
            59947.478358247085,
            59823.189656752256,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-4,
                "[{}] SAR {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_sar_from_slices(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [50000.0, 50500.0, 51000.0];
        let low = [49000.0, 49500.0, 49900.0];
        let params = SarParams::default();
        let input = SarInput::from_slices(&high, &low, params);
        let result = sar_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), high.len());
        Ok(())
    }

    #[test]
    fn test_sar_into_slice_trims_mismatched_lengths() -> Result<(), Box<dyn Error>> {
        let high = [50000.0, 50500.0, 51000.0, 52000.0];
        let low = [49000.0, 49500.0, 49900.0];
        let params = SarParams::default();
        let input = SarInput::from_slices(&high, &low, params);

        let expected = sar_with_kernel(&input, Kernel::Scalar)?;
        assert_eq!(expected.values.len(), low.len());

        let mut out = vec![0.0; low.len()];
        sar_into_slice(&mut out, &input, Kernel::Scalar)?;

        for i in 0..out.len() {
            let a = out[i];
            let b = expected.values[i];
            assert!(
                (a.is_nan() && b.is_nan()) || a == b,
                "mismatch at {}: {} vs {}",
                i,
                a,
                b
            );
        }

        Ok(())
    }

    fn check_sar_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [f64::NAN, f64::NAN, f64::NAN];
        let low = [f64::NAN, f64::NAN, f64::NAN];
        let params = SarParams::default();
        let input = SarInput::from_slices(&high, &low, params);
        let result = sar_with_kernel(&input, kernel);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("All values are NaN"));
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_sar_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let test_params = vec![
            SarParams::default(),
            SarParams {
                acceleration: Some(0.001),
                maximum: Some(0.001),
            },
            SarParams {
                acceleration: Some(0.01),
                maximum: Some(0.1),
            },
            SarParams {
                acceleration: Some(0.02),
                maximum: Some(0.3),
            },
            SarParams {
                acceleration: Some(0.05),
                maximum: Some(0.2),
            },
            SarParams {
                acceleration: Some(0.05),
                maximum: Some(0.5),
            },
            SarParams {
                acceleration: Some(0.1),
                maximum: Some(0.5),
            },
            SarParams {
                acceleration: Some(0.1),
                maximum: Some(0.9),
            },
            SarParams {
                acceleration: Some(0.2),
                maximum: Some(0.9),
            },
            SarParams {
                acceleration: Some(0.001),
                maximum: Some(0.9),
            },
            SarParams {
                acceleration: Some(0.2),
                maximum: Some(0.01),
            },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = SarInput::from_candles(&candles, params.clone());
            let output = sar_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: acceleration={}, maximum={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.acceleration.unwrap_or(0.02),
                        params.maximum.unwrap_or(0.2),
                        param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: acceleration={}, maximum={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.acceleration.unwrap_or(0.02),
                        params.maximum.unwrap_or(0.2),
                        param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: acceleration={}, maximum={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.acceleration.unwrap_or(0.02),
                        params.maximum.unwrap_or(0.2),
                        param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_sar_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_sar_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (0.001f64..0.5f64)
            .prop_flat_map(|acceleration| (Just(acceleration), acceleration..1.0f64))
            .prop_flat_map(|(acceleration, maximum)| {
                (
                    prop::collection::vec(
                        (1.0f64..1e6f64).prop_filter("finite price", |x| x.is_finite() && *x > 0.0),
                        10..400,
                    ),
                    Just(acceleration),
                    Just(maximum),
                    0.001f64..0.1f64,
                )
            });

        proptest::test_runner::TestRunner::default().run(
            &strat,
            |(base_prices, acceleration, maximum, volatility)| {
                let mut high = Vec::with_capacity(base_prices.len());
                let mut low = Vec::with_capacity(base_prices.len());

                let mut spread_factor = 1.0;
                for price in &base_prices {
                    spread_factor = (spread_factor + (price % 0.1 - 0.05) * 0.2)
                        .max(0.5)
                        .min(2.0);
                    let spread = price * volatility * spread_factor;
                    high.push(price + spread);
                    low.push(price - spread);
                }

                let params = SarParams {
                    acceleration: Some(acceleration),
                    maximum: Some(maximum),
                };
                let input = SarInput::from_slices(&high, &low, params.clone());

                let SarOutput { values: out } = sar_with_kernel(&input, kernel).unwrap();

                let SarOutput { values: ref_out } =
                    sar_with_kernel(&input, Kernel::Scalar).unwrap();

                for i in 1..out.len() {
                    if !out[i].is_nan() {
                        let min_price = low.iter().cloned().fold(f64::INFINITY, f64::min);
                        let max_price = high.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                        prop_assert!(
                            out[i] >= min_price - 1e-9 && out[i] <= max_price + 1e-9,
                            "SAR[{}] = {} is outside range [{}, {}]",
                            i,
                            out[i],
                            min_price,
                            max_price
                        );
                    }
                }

                prop_assert!(
                    out[0].is_nan(),
                    "First SAR value should be NaN during warmup, got {}",
                    out[0]
                );

                for i in 0..out.len() {
                    let y = out[i];
                    let r = ref_out[i];

                    if y.is_nan() {
                        prop_assert!(
                            r.is_nan(),
                            "NaN mismatch at index {}: test={}, ref={}",
                            i,
                            y,
                            r
                        );
                    } else if r.is_nan() {
                        prop_assert!(
                            y.is_nan(),
                            "NaN mismatch at index {}: test={}, ref={}",
                            i,
                            y,
                            r
                        );
                    } else {
                        let diff = (y - r).abs();
                        prop_assert!(
                            diff < 1e-9,
                            "Kernel mismatch at index {}: test={}, ref={}, diff={}",
                            i,
                            y,
                            r,
                            diff
                        );
                    }
                }

                if out.len() > 10 {
                    let mut last_movement = 0.0;
                    let mut increasing_count = 0;

                    for i in 2..out.len().min(20) {
                        if !out[i].is_nan() && !out[i - 1].is_nan() {
                            let movement = (out[i] - out[i - 1]).abs();
                            if movement > last_movement {
                                increasing_count += 1;
                            }
                            last_movement = movement;
                        }
                    }

                    prop_assert!(
                        increasing_count > 0 || out.len() < 5,
                        "SAR acceleration never increases (count: {})",
                        increasing_count
                    );
                }

                let strong_uptrend = high.windows(2).all(|w| w[1] > w[0] + 1e-9)
                    && low.windows(2).all(|w| w[1] > w[0] + 1e-9);
                if strong_uptrend && out.len() > 10 {
                    let start = out.len() * 3 / 4;
                    for i in start..out.len() {
                        if !out[i].is_nan() {
                            prop_assert!(
                                out[i] <= low[i] + 1e-6,
                                "In uptrend, SAR[{}] = {} should be <= low[{}] = {}",
                                i,
                                out[i],
                                i,
                                low[i]
                            );
                        }
                    }
                }

                let strong_downtrend = high.windows(2).all(|w| w[1] < w[0] - 1e-9)
                    && low.windows(2).all(|w| w[1] < w[0] - 1e-9);
                if strong_downtrend && out.len() > 10 {
                    let start = out.len() * 3 / 4;
                    for i in start..out.len() {
                        if !out[i].is_nan() {
                            prop_assert!(
                                out[i] >= high[i] - 1e-6,
                                "In downtrend, SAR[{}] = {} should be >= high[{}] = {}",
                                i,
                                out[i],
                                i,
                                high[i]
                            );
                        }
                    }
                }

                if out.len() > 5 {
                    for i in 2..out.len() {
                        if !out[i].is_nan() && !out[i - 1].is_nan() {
                            let jump = (out[i] - out[i - 1]).abs();
                            let avg_price = (high[i] + low[i]) / 2.0;

                            if jump > avg_price * 0.05 {
                                let prev_below = out[i - 1] < low[i - 1];
                                let curr_below = out[i] < low[i];

                                prop_assert!(
                                    prev_below != curr_below || jump > avg_price * 0.03,
                                    "Large SAR jump without proper reversal at index {}",
                                    i
                                );
                            }
                        }
                    }
                }

                if base_prices.windows(2).all(|w| w[1] > w[0]) && out.len() > 20 {
                    let quarter = out.len() / 4;
                    let first_quarter: Vec<f64> = out[quarter..quarter * 2]
                        .iter()
                        .filter(|v| !v.is_nan())
                        .cloned()
                        .collect();
                    let last_quarter: Vec<f64> = out[quarter * 3..]
                        .iter()
                        .filter(|v| !v.is_nan())
                        .cloned()
                        .collect();

                    if !first_quarter.is_empty() && !last_quarter.is_empty() {
                        let first_avg =
                            first_quarter.iter().sum::<f64>() / first_quarter.len() as f64;
                        let last_avg = last_quarter.iter().sum::<f64>() / last_quarter.len() as f64;

                        prop_assert!(
							last_avg >= first_avg * 0.95,
							"For increasing prices, SAR should generally trend up: first_avg={}, last_avg={}",
							first_avg, last_avg
						);
                    }
                }

                #[cfg(debug_assertions)]
                {
                    for (i, &val) in out.iter().enumerate() {
                        if !val.is_nan() {
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
                    }
                }

                Ok(())
            },
        )?;

        Ok(())
    }

    macro_rules! generate_all_sar_tests {
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

    generate_all_sar_tests!(
        check_sar_partial_params,
        check_sar_accuracy,
        check_sar_from_slices,
        check_sar_all_nan,
        check_sar_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_sar_tests!(check_sar_property);

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = SarBatchBuilder::new().kernel(kernel).apply_candles(&c)?;

        let def = SarParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            60370.00224209362,
            60220.362107568006,
            60079.70038111392,
            59947.478358247085,
            59823.189656752256,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-4,
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
            (0.01, 0.05, 0.01, 0.1, 0.3, 0.1),
            (0.02, 0.1, 0.02, 0.2, 0.5, 0.1),
            (0.05, 0.2, 0.05, 0.3, 0.9, 0.2),
            (0.001, 0.005, 0.001, 0.05, 0.1, 0.05),
            (0.02, 0.02, 0.0, 0.2, 0.2, 0.0),
            (0.1, 0.2, 0.025, 0.5, 0.9, 0.1),
            (0.001, 0.01, 0.003, 0.1, 0.5, 0.2),
        ];

        for (cfg_idx, &(a_start, a_end, a_step, m_start, m_end, m_step)) in
            test_configs.iter().enumerate()
        {
            let output = SarBatchBuilder::new()
                .kernel(kernel)
                .acceleration_range(a_start, a_end, a_step)
                .maximum_range(m_start, m_end, m_step)
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
						 at row {} col {} (flat index {}) with params: acceleration={}, maximum={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.acceleration.unwrap_or(0.02),
                        combo.maximum.unwrap_or(0.2)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: acceleration={}, maximum={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.acceleration.unwrap_or(0.02),
                        combo.maximum.unwrap_or(0.2)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: acceleration={}, maximum={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.acceleration.unwrap_or(0.02),
                        combo.maximum.unwrap_or(0.2)
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
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto);
                }
            }
        };
    }
    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);

    #[test]
    fn test_sar_into_matches_api() -> Result<(), Box<dyn Error>> {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = SarInput::from_candles(&candles, SarParams::default());

        let SarOutput { values: expected } = sar(&input)?;

        let mut actual = vec![0.0; candles.high.len()];
        #[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
        {
            sar_into(&input, &mut actual)?;
        }
        #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
        {
            sar_into_slice(&mut actual, &input, Kernel::Auto)?;
        }

        assert_eq!(expected.len(), actual.len());

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b)
        }

        for i in 0..expected.len() {
            assert!(
                eq_or_both_nan(expected[i], actual[i]),
                "Mismatch at index {}: expected {:?}, got {:?}",
                i,
                expected[i],
                actual[i]
            );
        }
        Ok(())
    }
}
