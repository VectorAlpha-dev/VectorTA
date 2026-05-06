#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(feature = "python")]
use pyo3::wrap_pyfunction;

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use serde::{Deserialize, Serialize};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use serde_wasm_bindgen;
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{alloc_uninit_f64, detect_best_batch_kernel, make_uninit_matrix};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::mem::MaybeUninit;
use thiserror::Error;

const DEFAULT_PERIOD: usize = 10;
const DEFAULT_HOT: f64 = 0.7;
const DEFAULT_T3_MODE: usize = 0;

impl<'a> AsRef<[f64]> for EmaDeviationCorrectedT3Input<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            EmaDeviationCorrectedT3Data::Slice(slice) => slice,
            EmaDeviationCorrectedT3Data::Candles { candles, source } => {
                source_type(candles, source)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum EmaDeviationCorrectedT3Data<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct EmaDeviationCorrectedT3Output {
    pub corrected: Vec<f64>,
    pub t3: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct EmaDeviationCorrectedT3Params {
    pub period: Option<usize>,
    pub hot: Option<f64>,
    pub t3_mode: Option<usize>,
}

impl Default for EmaDeviationCorrectedT3Params {
    fn default() -> Self {
        Self {
            period: Some(DEFAULT_PERIOD),
            hot: Some(DEFAULT_HOT),
            t3_mode: Some(DEFAULT_T3_MODE),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EmaDeviationCorrectedT3Input<'a> {
    pub data: EmaDeviationCorrectedT3Data<'a>,
    pub params: EmaDeviationCorrectedT3Params,
}

impl<'a> EmaDeviationCorrectedT3Input<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: EmaDeviationCorrectedT3Params,
    ) -> Self {
        Self {
            data: EmaDeviationCorrectedT3Data::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(data: &'a [f64], params: EmaDeviationCorrectedT3Params) -> Self {
        Self {
            data: EmaDeviationCorrectedT3Data::Slice(data),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", EmaDeviationCorrectedT3Params::default())
    }

    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(DEFAULT_PERIOD)
    }

    #[inline]
    pub fn get_hot(&self) -> f64 {
        self.params.hot.unwrap_or(DEFAULT_HOT)
    }

    #[inline]
    pub fn get_t3_mode(&self) -> usize {
        self.params.t3_mode.unwrap_or(DEFAULT_T3_MODE)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct EmaDeviationCorrectedT3Builder {
    period: Option<usize>,
    hot: Option<f64>,
    t3_mode: Option<usize>,
    kernel: Kernel,
}

impl Default for EmaDeviationCorrectedT3Builder {
    fn default() -> Self {
        Self {
            period: None,
            hot: None,
            t3_mode: None,
            kernel: Kernel::Auto,
        }
    }
}

impl EmaDeviationCorrectedT3Builder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn period(mut self, value: usize) -> Self {
        self.period = Some(value);
        self
    }

    #[inline(always)]
    pub fn hot(mut self, value: f64) -> Self {
        self.hot = Some(value);
        self
    }

    #[inline(always)]
    pub fn t3_mode(mut self, value: usize) -> Self {
        self.t3_mode = Some(value);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<EmaDeviationCorrectedT3Output, EmaDeviationCorrectedT3Error> {
        let input = EmaDeviationCorrectedT3Input::from_candles(
            candles,
            "close",
            EmaDeviationCorrectedT3Params {
                period: self.period,
                hot: self.hot,
                t3_mode: self.t3_mode,
            },
        );
        ema_deviation_corrected_t3_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<EmaDeviationCorrectedT3Output, EmaDeviationCorrectedT3Error> {
        let input = EmaDeviationCorrectedT3Input::from_slice(
            data,
            EmaDeviationCorrectedT3Params {
                period: self.period,
                hot: self.hot,
                t3_mode: self.t3_mode,
            },
        );
        ema_deviation_corrected_t3_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<EmaDeviationCorrectedT3Stream, EmaDeviationCorrectedT3Error> {
        EmaDeviationCorrectedT3Stream::try_new(EmaDeviationCorrectedT3Params {
            period: self.period,
            hot: self.hot,
            t3_mode: self.t3_mode,
        })
    }
}

#[derive(Debug, Error)]
pub enum EmaDeviationCorrectedT3Error {
    #[error("ema_deviation_corrected_t3: Input data slice is empty.")]
    EmptyInputData,

    #[error("ema_deviation_corrected_t3: All values are NaN.")]
    AllValuesNaN,

    #[error(
        "ema_deviation_corrected_t3: Invalid period: period = {period}, data length = {data_len}"
    )]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("ema_deviation_corrected_t3: Invalid hot value: {hot}")]
    InvalidHot { hot: f64 },

    #[error("ema_deviation_corrected_t3: Invalid T3 mode: {mode}")]
    InvalidT3Mode { mode: usize },

    #[error(
        "ema_deviation_corrected_t3: Output length mismatch: expected = {expected}, got = {got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },

    #[error(
        "ema_deviation_corrected_t3: Invalid integer range: start={start}, end={end}, step={step}"
    )]
    InvalidRangeUsize {
        start: usize,
        end: usize,
        step: usize,
    },

    #[error(
        "ema_deviation_corrected_t3: Invalid float range: start={start}, end={end}, step={step}"
    )]
    InvalidRangeF64 { start: f64, end: f64, step: f64 },

    #[error("ema_deviation_corrected_t3: Invalid kernel for batch path: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline(always)]
fn alpha_t3(period: usize, mode: usize) -> Result<f64, EmaDeviationCorrectedT3Error> {
    match mode {
        0 => Ok(2.0 / (2.0 + (period as f64 - 1.0) / 2.0)),
        1 => Ok(2.0 / (1.0 + period as f64)),
        _ => Err(EmaDeviationCorrectedT3Error::InvalidT3Mode { mode }),
    }
}

#[inline(always)]
fn correction_variance_scale(period: usize) -> f64 {
    period as f64 / (period.saturating_sub(1).max(1) as f64)
}

#[inline(always)]
fn compute_coefficients(hot: f64) -> (f64, f64, f64, f64) {
    let hot2 = hot * hot;
    let hot3 = hot2 * hot;
    let c1 = -hot3;
    let c2 = 3.0 * hot2 + 3.0 * hot3;
    let c3 = -6.0 * hot2 - 3.0 * hot - 3.0 * hot3;
    let c4 = 1.0 + 3.0 * hot + hot3 + 3.0 * hot2;
    (c1, c2, c3, c4)
}

#[inline(always)]
fn prepare_input<'a>(
    input: &'a EmaDeviationCorrectedT3Input<'a>,
) -> Result<(&'a [f64], usize, f64, usize), EmaDeviationCorrectedT3Error> {
    let data = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(EmaDeviationCorrectedT3Error::EmptyInputData);
    }
    if data.iter().all(|v| !v.is_finite()) {
        return Err(EmaDeviationCorrectedT3Error::AllValuesNaN);
    }

    let (period, hot, mode) = validate_params_for_len(&input.params, len)?;

    Ok((data, period, hot, mode))
}

#[inline(always)]
fn validate_params_for_len(
    params: &EmaDeviationCorrectedT3Params,
    len: usize,
) -> Result<(usize, f64, usize), EmaDeviationCorrectedT3Error> {
    let period = params.period.unwrap_or(DEFAULT_PERIOD);
    if period == 0 || period > len {
        return Err(EmaDeviationCorrectedT3Error::InvalidPeriod {
            period,
            data_len: len,
        });
    }

    let hot = params.hot.unwrap_or(DEFAULT_HOT);
    if !hot.is_finite() {
        return Err(EmaDeviationCorrectedT3Error::InvalidHot { hot });
    }

    let mode = params.t3_mode.unwrap_or(DEFAULT_T3_MODE);
    let _ = alpha_t3(period, mode)?;

    Ok((period, hot, mode))
}

#[inline(always)]
pub fn ema_deviation_corrected_t3(
    input: &EmaDeviationCorrectedT3Input,
) -> Result<EmaDeviationCorrectedT3Output, EmaDeviationCorrectedT3Error> {
    ema_deviation_corrected_t3_with_kernel(input, Kernel::Auto)
}

pub fn ema_deviation_corrected_t3_with_kernel(
    input: &EmaDeviationCorrectedT3Input,
    _kernel: Kernel,
) -> Result<EmaDeviationCorrectedT3Output, EmaDeviationCorrectedT3Error> {
    let (data, period, hot, mode) = prepare_input(input)?;
    let len = data.len();
    let mut corrected = alloc_uninit_f64(len);
    let mut t3 = alloc_uninit_f64(len);
    compute_into_slices(data, period, hot, mode, &mut corrected, &mut t3)?;
    Ok(EmaDeviationCorrectedT3Output { corrected, t3 })
}

#[inline(always)]
fn compute_into_slices(
    data: &[f64],
    period: usize,
    hot: f64,
    mode: usize,
    corrected: &mut [f64],
    t3: &mut [f64],
) -> Result<(), EmaDeviationCorrectedT3Error> {
    let alpha_t3 = alpha_t3(period, mode)?;
    let alpha_ema = 2.0 / (1.0 + period as f64);
    let variance_scale = correction_variance_scale(period);
    let (c1, c2, c3, c4) = compute_coefficients(hot);

    let mut t0 = 0.0;
    let mut t1 = 0.0;
    let mut t2 = 0.0;
    let mut t3s = 0.0;
    let mut t4 = 0.0;
    let mut t5 = 0.0;
    let mut ema0 = 0.0;
    let mut ema1 = 0.0;
    let mut corr = 0.0;
    let mut seeded_ema = false;

    for i in 0..data.len() {
        let value = data[i];
        if !value.is_finite() {
            t0 = 0.0;
            t1 = 0.0;
            t2 = 0.0;
            t3s = 0.0;
            t4 = 0.0;
            t5 = 0.0;
            ema0 = 0.0;
            ema1 = 0.0;
            corr = 0.0;
            seeded_ema = false;
            corrected[i] = f64::NAN;
            t3[i] = f64::NAN;
            continue;
        }

        t0 += alpha_t3 * (value - t0);
        t1 += alpha_t3 * (t0 - t1);
        t2 += alpha_t3 * (t1 - t2);
        t3s += alpha_t3 * (t2 - t3s);
        t4 += alpha_t3 * (t3s - t4);
        t5 += alpha_t3 * (t4 - t5);

        let t3_value = c1 * t5 + c2 * t4 + c3 * t3s + c4 * t2;

        let price_sq = value * value;
        if seeded_ema {
            ema0 += alpha_ema * (value - ema0);
            ema1 += alpha_ema * (price_sq - ema1);
        } else {
            ema0 = value;
            ema1 = price_sq;
            seeded_ema = true;
        }

        let variance_sq = (ema1 - ema0 * ema0).max(0.0) * variance_scale;
        let v2 = (corr - t3_value) * (corr - t3_value);
        let c = if v2 < variance_sq || v2 == 0.0 {
            0.0
        } else {
            1.0 - variance_sq / v2
        };
        corr += c * (t3_value - corr);

        corrected[i] = corr;
        t3[i] = t3_value;
    }
    Ok(())
}

#[inline]
pub fn ema_deviation_corrected_t3_into_slices_with_kernel(
    corrected: &mut [f64],
    t3: &mut [f64],
    input: &EmaDeviationCorrectedT3Input,
    _kernel: Kernel,
) -> Result<(), EmaDeviationCorrectedT3Error> {
    let (data, period, hot, mode) = prepare_input(input)?;
    let len = data.len();
    if corrected.len() != len || t3.len() != len {
        return Err(EmaDeviationCorrectedT3Error::OutputLengthMismatch {
            expected: len,
            got: corrected.len().min(t3.len()),
        });
    }

    compute_into_slices(data, period, hot, mode, corrected, t3)
}

#[inline]
pub fn ema_deviation_corrected_t3_into_slices(
    corrected: &mut [f64],
    t3: &mut [f64],
    input: &EmaDeviationCorrectedT3Input,
) -> Result<(), EmaDeviationCorrectedT3Error> {
    ema_deviation_corrected_t3_into_slices_with_kernel(corrected, t3, input, Kernel::Auto)
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn ema_deviation_corrected_t3_into(
    input: &EmaDeviationCorrectedT3Input,
    corrected: &mut [f64],
    t3: &mut [f64],
) -> Result<(), EmaDeviationCorrectedT3Error> {
    ema_deviation_corrected_t3_into_slices(corrected, t3, input)
}

#[derive(Debug, Clone)]
pub struct EmaDeviationCorrectedT3Stream {
    alpha_t3: f64,
    alpha_ema: f64,
    variance_scale: f64,
    c1: f64,
    c2: f64,
    c3: f64,
    c4: f64,
    t0: f64,
    t1: f64,
    t2: f64,
    t3s: f64,
    t4: f64,
    t5: f64,
    ema0: f64,
    ema1: f64,
    corr: f64,
    seeded_ema: bool,
}

impl EmaDeviationCorrectedT3Stream {
    pub fn try_new(
        params: EmaDeviationCorrectedT3Params,
    ) -> Result<Self, EmaDeviationCorrectedT3Error> {
        let period = params.period.unwrap_or(DEFAULT_PERIOD);
        if period == 0 {
            return Err(EmaDeviationCorrectedT3Error::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let hot = params.hot.unwrap_or(DEFAULT_HOT);
        if !hot.is_finite() {
            return Err(EmaDeviationCorrectedT3Error::InvalidHot { hot });
        }
        let mode = params.t3_mode.unwrap_or(DEFAULT_T3_MODE);
        let alpha_t3 = alpha_t3(period, mode)?;
        let alpha_ema = 2.0 / (1.0 + period as f64);
        let variance_scale = correction_variance_scale(period);
        let (c1, c2, c3, c4) = compute_coefficients(hot);

        Ok(Self {
            alpha_t3,
            alpha_ema,
            variance_scale,
            c1,
            c2,
            c3,
            c4,
            t0: 0.0,
            t1: 0.0,
            t2: 0.0,
            t3s: 0.0,
            t4: 0.0,
            t5: 0.0,
            ema0: 0.0,
            ema1: 0.0,
            corr: 0.0,
            seeded_ema: false,
        })
    }

    #[inline]
    pub fn reset(&mut self) {
        self.t0 = 0.0;
        self.t1 = 0.0;
        self.t2 = 0.0;
        self.t3s = 0.0;
        self.t4 = 0.0;
        self.t5 = 0.0;
        self.ema0 = 0.0;
        self.ema1 = 0.0;
        self.corr = 0.0;
        self.seeded_ema = false;
    }

    #[inline]
    pub fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        if !value.is_finite() {
            self.reset();
            return None;
        }

        let prev_t0 = self.t0;
        self.t0 = prev_t0 + self.alpha_t3 * (value - prev_t0);

        let prev_t1 = self.t1;
        self.t1 = prev_t1 + self.alpha_t3 * (self.t0 - prev_t1);

        let prev_t2 = self.t2;
        self.t2 = prev_t2 + self.alpha_t3 * (self.t1 - prev_t2);

        let prev_t3 = self.t3s;
        self.t3s = prev_t3 + self.alpha_t3 * (self.t2 - prev_t3);

        let prev_t4 = self.t4;
        self.t4 = prev_t4 + self.alpha_t3 * (self.t3s - prev_t4);

        let prev_t5 = self.t5;
        self.t5 = prev_t5 + self.alpha_t3 * (self.t4 - prev_t5);

        let t3_value =
            self.c1 * self.t5 + self.c2 * self.t4 + self.c3 * self.t3s + self.c4 * self.t2;

        let price_sq = value * value;
        if !self.seeded_ema {
            self.ema0 = value;
            self.ema1 = price_sq;
            self.seeded_ema = true;
        } else {
            self.ema0 += self.alpha_ema * (value - self.ema0);
            self.ema1 += self.alpha_ema * (price_sq - self.ema1);
        }

        let variance_sq = (self.ema1 - self.ema0 * self.ema0).max(0.0) * self.variance_scale;
        let prev_corr = self.corr;
        let v2 = (prev_corr - t3_value) * (prev_corr - t3_value);
        let c = if v2 < variance_sq || v2 == 0.0 {
            0.0
        } else {
            1.0 - variance_sq / v2
        };
        self.corr = prev_corr + c * (t3_value - prev_corr);

        Some((t3_value, self.corr))
    }
}

#[derive(Clone, Debug)]
pub struct EmaDeviationCorrectedT3BatchRange {
    pub period: (usize, usize, usize),
    pub hot: (f64, f64, f64),
    pub t3_mode: (usize, usize, usize),
}

impl Default for EmaDeviationCorrectedT3BatchRange {
    fn default() -> Self {
        Self {
            period: (DEFAULT_PERIOD, DEFAULT_PERIOD, 0),
            hot: (DEFAULT_HOT, DEFAULT_HOT, 0.0),
            t3_mode: (DEFAULT_T3_MODE, DEFAULT_T3_MODE, 0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct EmaDeviationCorrectedT3BatchBuilder {
    range: EmaDeviationCorrectedT3BatchRange,
    kernel: Kernel,
}

impl Default for EmaDeviationCorrectedT3BatchBuilder {
    fn default() -> Self {
        Self {
            range: EmaDeviationCorrectedT3BatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl EmaDeviationCorrectedT3BatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.period = (start, end, step);
        self
    }

    pub fn hot_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.hot = (start, end, step);
        self
    }

    pub fn t3_mode_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.t3_mode = (start, end, step);
        self
    }

    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<EmaDeviationCorrectedT3BatchOutput, EmaDeviationCorrectedT3Error> {
        ema_deviation_corrected_t3_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<EmaDeviationCorrectedT3BatchOutput, EmaDeviationCorrectedT3Error> {
        self.apply_slice(source_type(candles, source))
    }
}

#[derive(Clone, Debug)]
pub struct EmaDeviationCorrectedT3BatchOutput {
    pub corrected: Vec<f64>,
    pub t3: Vec<f64>,
    pub combos: Vec<EmaDeviationCorrectedT3Params>,
    pub rows: usize,
    pub cols: usize,
}

#[inline(always)]
fn axis_usize(range: (usize, usize, usize)) -> Result<Vec<usize>, EmaDeviationCorrectedT3Error> {
    let (start, end, step) = range;
    if step == 0 || start == end {
        return Ok(vec![start]);
    }
    let mut out = Vec::new();
    if start < end {
        let mut cur = start;
        loop {
            out.push(cur);
            match cur.checked_add(step) {
                Some(next) if next <= end => cur = next,
                _ => break,
            }
        }
    } else {
        let mut cur = start;
        loop {
            out.push(cur);
            if cur <= end || cur < step {
                break;
            }
            cur -= step;
            if cur < end {
                break;
            }
        }
    }
    if out.is_empty() {
        return Err(EmaDeviationCorrectedT3Error::InvalidRangeUsize { start, end, step });
    }
    Ok(out)
}

#[inline(always)]
fn axis_f64(range: (f64, f64, f64)) -> Result<Vec<f64>, EmaDeviationCorrectedT3Error> {
    let (start, end, step) = range;
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(EmaDeviationCorrectedT3Error::InvalidRangeF64 { start, end, step });
    }
    if step == 0.0 || start == end {
        return Ok(vec![start]);
    }
    if step < 0.0 {
        return Err(EmaDeviationCorrectedT3Error::InvalidRangeF64 { start, end, step });
    }

    let mut out = Vec::new();
    let eps = step.abs() * 1e-12 + 1e-12;
    if start < end {
        let mut cur = start;
        while cur <= end + eps {
            out.push(cur);
            cur += step;
        }
    } else {
        let mut cur = start;
        while cur >= end - eps {
            out.push(cur);
            cur -= step;
        }
    }

    if out.is_empty() {
        return Err(EmaDeviationCorrectedT3Error::InvalidRangeF64 { start, end, step });
    }
    Ok(out)
}

#[inline(always)]
fn to_batch_kernel(kernel: Kernel) -> Result<Kernel, EmaDeviationCorrectedT3Error> {
    match kernel {
        Kernel::Auto => Ok(detect_best_batch_kernel()),
        value if value.is_batch() => Ok(value),
        value => Err(EmaDeviationCorrectedT3Error::InvalidKernelForBatch(value)),
    }
}

#[inline(always)]
pub fn expand_grid_ema_deviation_corrected_t3(
    range: &EmaDeviationCorrectedT3BatchRange,
) -> Result<Vec<EmaDeviationCorrectedT3Params>, EmaDeviationCorrectedT3Error> {
    let periods = axis_usize(range.period)?;
    let hots = axis_f64(range.hot)?;
    let modes = axis_usize(range.t3_mode)?;

    let mut combos = Vec::with_capacity(periods.len() * hots.len() * modes.len());
    for period in periods {
        for &hot in &hots {
            for &mode in &modes {
                let _ = alpha_t3(period, mode)?;
                combos.push(EmaDeviationCorrectedT3Params {
                    period: Some(period),
                    hot: Some(hot),
                    t3_mode: Some(mode),
                });
            }
        }
    }

    Ok(combos)
}

#[inline(always)]
pub fn ema_deviation_corrected_t3_batch_slice(
    data: &[f64],
    range: &EmaDeviationCorrectedT3BatchRange,
    kernel: Kernel,
) -> Result<EmaDeviationCorrectedT3BatchOutput, EmaDeviationCorrectedT3Error> {
    ema_deviation_corrected_t3_batch_inner(data, range, kernel, false)
}

#[inline(always)]
pub fn ema_deviation_corrected_t3_batch_par_slice(
    data: &[f64],
    range: &EmaDeviationCorrectedT3BatchRange,
    kernel: Kernel,
) -> Result<EmaDeviationCorrectedT3BatchOutput, EmaDeviationCorrectedT3Error> {
    ema_deviation_corrected_t3_batch_inner(data, range, kernel, true)
}

#[inline(always)]
fn ema_deviation_corrected_t3_batch_inner(
    data: &[f64],
    range: &EmaDeviationCorrectedT3BatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<EmaDeviationCorrectedT3BatchOutput, EmaDeviationCorrectedT3Error> {
    if data.is_empty() {
        return Err(EmaDeviationCorrectedT3Error::EmptyInputData);
    }
    if data.iter().all(|v| !v.is_finite()) {
        return Err(EmaDeviationCorrectedT3Error::AllValuesNaN);
    }
    let _ = to_batch_kernel(kernel)?;
    let combos = expand_grid_ema_deviation_corrected_t3(range)?;
    let rows = combos.len();
    let cols = data.len();
    for combo in &combos {
        let _ = validate_params_for_len(combo, cols)?;
    }

    let mut corrected_mu = make_uninit_matrix(rows, cols);
    let mut t3_mu = make_uninit_matrix(rows, cols);

    let row_fn = |row: usize,
                  corrected_dst: &mut [MaybeUninit<f64>],
                  t3_dst: &mut [MaybeUninit<f64>]| {
        let corrected_out =
            unsafe { std::slice::from_raw_parts_mut(corrected_dst.as_mut_ptr() as *mut f64, cols) };
        let t3_out =
            unsafe { std::slice::from_raw_parts_mut(t3_dst.as_mut_ptr() as *mut f64, cols) };
        let (period, hot, mode) =
            validate_params_for_len(&combos[row], cols).expect("validated batch params");
        compute_into_slices(data, period, hot, mode, corrected_out, t3_out)
            .expect("validated batch params");
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            corrected_mu
                .par_chunks_mut(cols)
                .zip(t3_mu.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, (corrected_slice, t3_slice))| {
                    row_fn(row, corrected_slice, t3_slice)
                });
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, (corrected_slice, t3_slice)) in corrected_mu
                .chunks_mut(cols)
                .zip(t3_mu.chunks_mut(cols))
                .enumerate()
            {
                row_fn(row, corrected_slice, t3_slice);
            }
        }
    } else {
        for (row, (corrected_slice, t3_slice)) in corrected_mu
            .chunks_mut(cols)
            .zip(t3_mu.chunks_mut(cols))
            .enumerate()
        {
            row_fn(row, corrected_slice, t3_slice);
        }
    }

    let corrected = unsafe {
        Vec::from_raw_parts(
            corrected_mu.as_mut_ptr() as *mut f64,
            corrected_mu.len(),
            corrected_mu.capacity(),
        )
    };
    let t3 = unsafe {
        Vec::from_raw_parts(
            t3_mu.as_mut_ptr() as *mut f64,
            t3_mu.len(),
            t3_mu.capacity(),
        )
    };
    std::mem::forget(corrected_mu);
    std::mem::forget(t3_mu);

    Ok(EmaDeviationCorrectedT3BatchOutput {
        corrected,
        t3,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub fn ema_deviation_corrected_t3_batch_inner_into(
    data: &[f64],
    range: &EmaDeviationCorrectedT3BatchRange,
    kernel: Kernel,
    parallel: bool,
    corrected_out: &mut [f64],
    t3_out: &mut [f64],
) -> Result<Vec<EmaDeviationCorrectedT3Params>, EmaDeviationCorrectedT3Error> {
    if data.is_empty() {
        return Err(EmaDeviationCorrectedT3Error::EmptyInputData);
    }
    if data.iter().all(|v| !v.is_finite()) {
        return Err(EmaDeviationCorrectedT3Error::AllValuesNaN);
    }
    let _ = to_batch_kernel(kernel)?;
    let combos = expand_grid_ema_deviation_corrected_t3(range)?;
    let rows = combos.len();
    let cols = data.len();
    for combo in &combos {
        let _ = validate_params_for_len(combo, cols)?;
    }
    let expected = rows.saturating_mul(cols);
    if corrected_out.len() != expected || t3_out.len() != expected {
        return Err(EmaDeviationCorrectedT3Error::OutputLengthMismatch {
            expected,
            got: corrected_out.len().min(t3_out.len()),
        });
    }

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            corrected_out
                .par_chunks_mut(cols)
                .zip(t3_out.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, (corrected_slice, t3_slice))| {
                    let (period, hot, mode) = validate_params_for_len(&combos[row], cols)
                        .expect("validated batch params");
                    compute_into_slices(data, period, hot, mode, corrected_slice, t3_slice)
                        .expect("validated batch params");
                });
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, (corrected_slice, t3_slice)) in corrected_out
                .chunks_mut(cols)
                .zip(t3_out.chunks_mut(cols))
                .enumerate()
            {
                let (period, hot, mode) = validate_params_for_len(&combos[row], cols)?;
                compute_into_slices(data, period, hot, mode, corrected_slice, t3_slice)?;
            }
        }
    } else {
        for (row, (corrected_slice, t3_slice)) in corrected_out
            .chunks_mut(cols)
            .zip(t3_out.chunks_mut(cols))
            .enumerate()
        {
            let (period, hot, mode) = validate_params_for_len(&combos[row], cols)?;
            compute_into_slices(data, period, hot, mode, corrected_slice, t3_slice)?;
        }
    }

    Ok(combos)
}

#[inline(always)]
pub fn ema_deviation_corrected_t3_batch_with_kernel(
    data: &[f64],
    range: &EmaDeviationCorrectedT3BatchRange,
    kernel: Kernel,
) -> Result<EmaDeviationCorrectedT3BatchOutput, EmaDeviationCorrectedT3Error> {
    let _ = to_batch_kernel(kernel)?;
    ema_deviation_corrected_t3_batch_par_slice(data, range, Kernel::ScalarBatch)
}

#[cfg(feature = "python")]
#[pyfunction(name = "ema_deviation_corrected_t3")]
#[pyo3(signature = (data, period=DEFAULT_PERIOD, hot=DEFAULT_HOT, t3_mode=DEFAULT_T3_MODE, kernel=None))]
pub fn ema_deviation_corrected_t3_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period: usize,
    hot: f64,
    t3_mode: usize,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let slice = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let input = EmaDeviationCorrectedT3Input::from_slice(
        slice,
        EmaDeviationCorrectedT3Params {
            period: Some(period),
            hot: Some(hot),
            t3_mode: Some(t3_mode),
        },
    );
    let out = py
        .allow_threads(|| ema_deviation_corrected_t3_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((out.corrected.into_pyarray(py), out.t3.into_pyarray(py)))
}

#[cfg(feature = "python")]
#[pyclass(name = "EmaDeviationCorrectedT3Stream")]
pub struct EmaDeviationCorrectedT3StreamPy {
    inner: EmaDeviationCorrectedT3Stream,
}

#[cfg(feature = "python")]
#[pymethods]
impl EmaDeviationCorrectedT3StreamPy {
    #[new]
    #[pyo3(signature = (period=DEFAULT_PERIOD, hot=DEFAULT_HOT, t3_mode=DEFAULT_T3_MODE))]
    fn new(period: usize, hot: f64, t3_mode: usize) -> PyResult<Self> {
        let inner = EmaDeviationCorrectedT3Stream::try_new(EmaDeviationCorrectedT3Params {
            period: Some(period),
            hot: Some(hot),
            t3_mode: Some(t3_mode),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        self.inner
            .update(value)
            .map(|(t3, corrected)| (corrected, t3))
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "ema_deviation_corrected_t3_batch")]
#[pyo3(signature = (data, period_range=(DEFAULT_PERIOD, DEFAULT_PERIOD, 0), hot_range=(DEFAULT_HOT, DEFAULT_HOT, 0.0), t3_mode_range=(DEFAULT_T3_MODE, DEFAULT_T3_MODE, 0), kernel=None))]
pub fn ema_deviation_corrected_t3_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    hot_range: (f64, f64, f64),
    t3_mode_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let slice = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let range = EmaDeviationCorrectedT3BatchRange {
        period: period_range,
        hot: hot_range,
        t3_mode: t3_mode_range,
    };
    let combos = expand_grid_ema_deviation_corrected_t3(&range)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = slice.len();

    let corrected_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let t3_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let corrected_slice = unsafe { corrected_arr.as_slice_mut()? };
    let t3_slice = unsafe { t3_arr.as_slice_mut()? };

    let combos = py
        .allow_threads(|| {
            ema_deviation_corrected_t3_batch_inner_into(
                slice,
                &range,
                kern,
                false,
                corrected_slice,
                t3_slice,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("corrected", corrected_arr.reshape((rows, cols))?)?;
    dict.set_item("t3", t3_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "periods",
        combos
            .iter()
            .map(|p| p.period.unwrap_or(DEFAULT_PERIOD) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "hots",
        combos
            .iter()
            .map(|p| p.hot.unwrap_or(DEFAULT_HOT))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "t3_modes",
        combos
            .iter()
            .map(|p| p.t3_mode.unwrap_or(DEFAULT_T3_MODE) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_ema_deviation_corrected_t3_module(
    m: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ema_deviation_corrected_t3_py, m)?)?;
    m.add_function(wrap_pyfunction!(ema_deviation_corrected_t3_batch_py, m)?)?;
    m.add_class::<EmaDeviationCorrectedT3StreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
#[derive(Serialize, Deserialize)]
pub struct EmaDeviationCorrectedT3Result {
    values: Vec<f64>,
    rows: usize,
    cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
impl EmaDeviationCorrectedT3Result {
    #[wasm_bindgen(getter)]
    pub fn values(&self) -> Vec<f64> {
        self.values.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[wasm_bindgen(getter)]
    pub fn cols(&self) -> usize {
        self.cols
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ema_deviation_corrected_t3_js(
    data: &[f64],
    period: usize,
    hot: f64,
    t3_mode: usize,
) -> Result<EmaDeviationCorrectedT3Result, JsValue> {
    let input = EmaDeviationCorrectedT3Input::from_slice(
        data,
        EmaDeviationCorrectedT3Params {
            period: Some(period),
            hot: Some(hot),
            t3_mode: Some(t3_mode),
        },
    );
    let out = ema_deviation_corrected_t3(&input).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let len = data.len();
    let mut values = Vec::with_capacity(2 * len);
    values.extend_from_slice(&out.corrected);
    values.extend_from_slice(&out.t3);
    Ok(EmaDeviationCorrectedT3Result {
        values,
        rows: 2,
        cols: len,
    })
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct EmaDeviationCorrectedT3BatchConfig {
    pub period_range: (usize, usize, usize),
    pub hot_range: (f64, f64, f64),
    pub t3_mode_range: (usize, usize, usize),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct EmaDeviationCorrectedT3BatchJsOutput {
    pub corrected: Vec<f64>,
    pub t3: Vec<f64>,
    pub periods: Vec<usize>,
    pub hots: Vec<f64>,
    pub t3_modes: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ema_deviation_corrected_t3_batch_js(
    data: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: EmaDeviationCorrectedT3BatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let range = EmaDeviationCorrectedT3BatchRange {
        period: config.period_range,
        hot: config.hot_range,
        t3_mode: config.t3_mode_range,
    };
    let out = ema_deviation_corrected_t3_batch_with_kernel(data, &range, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let js_output = EmaDeviationCorrectedT3BatchJsOutput {
        corrected: out.corrected,
        t3: out.t3,
        periods: out
            .combos
            .iter()
            .map(|p| p.period.unwrap_or(DEFAULT_PERIOD))
            .collect(),
        hots: out
            .combos
            .iter()
            .map(|p| p.hot.unwrap_or(DEFAULT_HOT))
            .collect(),
        t3_modes: out
            .combos
            .iter()
            .map(|p| p.t3_mode.unwrap_or(DEFAULT_T3_MODE))
            .collect(),
        rows: out.rows,
        cols: out.cols,
    };
    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ema_deviation_corrected_t3_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ema_deviation_corrected_t3_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ema_deviation_corrected_t3_into_host(
    data: &[f64],
    corrected_ptr: *mut f64,
    t3_ptr: *mut f64,
    period: usize,
    hot: f64,
    t3_mode: usize,
) -> Result<(), JsValue> {
    if corrected_ptr.is_null() || t3_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to ema_deviation_corrected_t3_into_host",
        ));
    }
    let input = EmaDeviationCorrectedT3Input::from_slice(
        data,
        EmaDeviationCorrectedT3Params {
            period: Some(period),
            hot: Some(hot),
            t3_mode: Some(t3_mode),
        },
    );
    let corrected = unsafe { std::slice::from_raw_parts_mut(corrected_ptr, data.len()) };
    let t3 = unsafe { std::slice::from_raw_parts_mut(t3_ptr, data.len()) };
    ema_deviation_corrected_t3_into_slices(corrected, t3, &input)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ema_deviation_corrected_t3_into(
    in_ptr: *const f64,
    corrected_ptr: *mut f64,
    t3_ptr: *mut f64,
    len: usize,
    period: usize,
    hot: f64,
    t3_mode: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || corrected_ptr.is_null() || t3_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to ema_deviation_corrected_t3_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let corrected = std::slice::from_raw_parts_mut(corrected_ptr, len);
        let t3 = std::slice::from_raw_parts_mut(t3_ptr, len);
        let input = EmaDeviationCorrectedT3Input::from_slice(
            data,
            EmaDeviationCorrectedT3Params {
                period: Some(period),
                hot: Some(hot),
                t3_mode: Some(t3_mode),
            },
        );
        ema_deviation_corrected_t3_into_slices(corrected, t3, &input)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ema_deviation_corrected_t3_batch_into(
    data: &[f64],
    corrected_ptr: *mut f64,
    t3_ptr: *mut f64,
    config: JsValue,
) -> Result<usize, JsValue> {
    if corrected_ptr.is_null() || t3_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to ema_deviation_corrected_t3_batch_into",
        ));
    }
    let config: EmaDeviationCorrectedT3BatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let range = EmaDeviationCorrectedT3BatchRange {
        period: config.period_range,
        hot: config.hot_range,
        t3_mode: config.t3_mode_range,
    };
    let combos = expand_grid_ema_deviation_corrected_t3(&range)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let cols = data.len();
    let corrected_out = unsafe { std::slice::from_raw_parts_mut(corrected_ptr, rows * cols) };
    let t3_out = unsafe { std::slice::from_raw_parts_mut(t3_ptr, rows * cols) };
    ema_deviation_corrected_t3_batch_inner_into(
        data,
        &range,
        Kernel::Auto,
        false,
        corrected_out,
        t3_out,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ema_deviation_corrected_t3_output_into_js(
    data: &[f64],
    period: usize,
    hot: f64,
    t3_mode: usize,
    out: &js_sys::Float64Array,
) -> Result<usize, JsValue> {
    let result = ema_deviation_corrected_t3_js(data, period, hot, t3_mode)?;
    crate::write_wasm_f64_output(
        "ema_deviation_corrected_t3_output_into_js",
        &result.values,
        out,
    )
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ema_deviation_corrected_t3_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = ema_deviation_corrected_t3_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "ema_deviation_corrected_t3_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_input_has_matching_corrected_and_t3() {
        let data = vec![100.0; 32];
        let input = EmaDeviationCorrectedT3Input::from_slice(
            &data,
            EmaDeviationCorrectedT3Params::default(),
        );
        let out = ema_deviation_corrected_t3(&input).expect("edct3");
        for i in 0..data.len() {
            assert!(
                (out.corrected[i] - out.t3[i]).abs() < 1e-12,
                "constant input mismatch at {i}: {} vs {}",
                out.corrected[i],
                out.t3[i]
            );
        }
    }

    #[test]
    fn stream_matches_batch_with_nan_reset() {
        let data = vec![1.0, 2.0, 3.0, 4.0, f64::NAN, 5.0, 6.0, 7.0, 8.0];
        let params = EmaDeviationCorrectedT3Params {
            period: Some(5),
            ..Default::default()
        };
        let input = EmaDeviationCorrectedT3Input::from_slice(&data, params.clone());
        let out = ema_deviation_corrected_t3(&input).expect("edct3");
        let mut stream = EmaDeviationCorrectedT3Stream::try_new(params).expect("stream");
        let mut corrected = Vec::with_capacity(data.len());
        let mut t3 = Vec::with_capacity(data.len());
        for value in data {
            if let Some((t3_value, corrected_value)) = stream.update(value) {
                corrected.push(corrected_value);
                t3.push(t3_value);
            } else {
                corrected.push(f64::NAN);
                t3.push(f64::NAN);
            }
        }
        assert_eq!(corrected.len(), out.corrected.len());
        for i in 0..corrected.len() {
            if corrected[i].is_nan()
                && out.corrected[i].is_nan()
                && t3[i].is_nan()
                && out.t3[i].is_nan()
            {
                continue;
            }
            assert!(
                (corrected[i] - out.corrected[i]).abs() < 1e-12,
                "corrected mismatch at {i}"
            );
            assert!((t3[i] - out.t3[i]).abs() < 1e-12, "t3 mismatch at {i}");
        }
    }

    #[test]
    fn batch_single_param_matches_single() {
        let data: Vec<f64> = (1..=64).map(|v| v as f64).collect();
        let input = EmaDeviationCorrectedT3Input::from_slice(
            &data,
            EmaDeviationCorrectedT3Params {
                period: Some(10),
                hot: Some(0.7),
                t3_mode: Some(0),
            },
        );
        let single = ema_deviation_corrected_t3(&input).expect("single");
        let batch = ema_deviation_corrected_t3_batch_with_kernel(
            &data,
            &EmaDeviationCorrectedT3BatchRange {
                period: (10, 10, 0),
                hot: (0.7, 0.7, 0.0),
                t3_mode: (0, 0, 0),
            },
            Kernel::Auto,
        )
        .expect("batch");
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, data.len());
        assert_eq!(batch.corrected, single.corrected);
        assert_eq!(batch.t3, single.t3);
    }

    #[test]
    fn invalid_t3_mode_rejected() {
        let data: Vec<f64> = (1..=16).map(|v| v as f64).collect();
        let input = EmaDeviationCorrectedT3Input::from_slice(
            &data,
            EmaDeviationCorrectedT3Params {
                period: Some(10),
                hot: Some(0.7),
                t3_mode: Some(2),
            },
        );
        assert!(ema_deviation_corrected_t3(&input).is_err());
    }
}
