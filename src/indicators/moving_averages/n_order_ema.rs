use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

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

const DEFAULT_PERIOD: f64 = 9.0;
const DEFAULT_ORDER: usize = 1;
const DEFAULT_EMA_STYLE: &str = "ema";
const DEFAULT_IIR_STYLE: &str = "impulse_matched";
const MAX_ORDER: usize = 64;

impl<'a> AsRef<[f64]> for NOrderEmaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            NOrderEmaData::Slice(slice) => slice,
            NOrderEmaData::Candles { candles, source } => match *source {
                "open" => &candles.open,
                "high" => &candles.high,
                "low" => &candles.low,
                "close" => &candles.close,
                "volume" => &candles.volume,
                "hl2" => &candles.hl2,
                "hlc3" => &candles.hlc3,
                "ohlc4" => &candles.ohlc4,
                "hlcc4" | "hlcc" => &candles.hlcc4,
                _ => source_type(candles, source),
            },
        }
    }
}

#[derive(Debug, Clone)]
pub enum NOrderEmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct NOrderEmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    serde(rename_all = "snake_case")
)]
pub enum NOrderEmaStyle {
    Ema,
    Dema,
    Hema,
    Tema,
}

impl NOrderEmaStyle {
    #[inline(always)]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ema => "ema",
            Self::Dema => "dema",
            Self::Hema => "hema",
            Self::Tema => "tema",
        }
    }

    #[inline(always)]
    pub fn from_str(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "ema" => Some(Self::Ema),
            "dema" => Some(Self::Dema),
            "hema" => Some(Self::Hema),
            "tema" => Some(Self::Tema),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    serde(rename_all = "snake_case")
)]
pub enum NOrderEmaIirStyle {
    AllPole,
    ImpulseMatched,
    MatchedZ,
    Bilinear,
}

impl NOrderEmaIirStyle {
    #[inline(always)]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AllPole => "all_pole",
            Self::ImpulseMatched => "impulse_matched",
            Self::MatchedZ => "matched_z",
            Self::Bilinear => "bilinear",
        }
    }

    #[inline(always)]
    pub fn from_str(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "all_pole" | "allpole" => Some(Self::AllPole),
            "impulse_matched" | "impulse-matched" | "impulsematched" => Some(Self::ImpulseMatched),
            "matched_z" | "matchedz" | "matched_z_transform" | "matched-z-transform" => {
                Some(Self::MatchedZ)
            }
            "bilinear" | "bilinear_transform" | "bilinear-transform" => Some(Self::Bilinear),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct NOrderEmaParams {
    pub period: Option<f64>,
    pub order: Option<usize>,
    pub ema_style: Option<String>,
    pub iir_style: Option<String>,
}

impl Default for NOrderEmaParams {
    fn default() -> Self {
        Self {
            period: Some(DEFAULT_PERIOD),
            order: Some(DEFAULT_ORDER),
            ema_style: Some(DEFAULT_EMA_STYLE.to_string()),
            iir_style: Some(DEFAULT_IIR_STYLE.to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct NOrderEmaInput<'a> {
    pub data: NOrderEmaData<'a>,
    pub params: NOrderEmaParams,
}

impl<'a> NOrderEmaInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: NOrderEmaParams) -> Self {
        Self {
            data: NOrderEmaData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(data: &'a [f64], params: NOrderEmaParams) -> Self {
        Self {
            data: NOrderEmaData::Slice(data),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", NOrderEmaParams::default())
    }
}

#[derive(Copy, Clone, Debug)]
pub struct NOrderEmaBuilder {
    period: Option<f64>,
    order: Option<usize>,
    ema_style: Option<NOrderEmaStyle>,
    iir_style: Option<NOrderEmaIirStyle>,
    kernel: Kernel,
}

impl Default for NOrderEmaBuilder {
    fn default() -> Self {
        Self {
            period: None,
            order: None,
            ema_style: None,
            iir_style: None,
            kernel: Kernel::Auto,
        }
    }
}

impl NOrderEmaBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn period(mut self, value: f64) -> Self {
        self.period = Some(value);
        self
    }

    #[inline(always)]
    pub fn order(mut self, value: usize) -> Self {
        self.order = Some(value);
        self
    }

    #[inline(always)]
    pub fn ema_style(mut self, value: NOrderEmaStyle) -> Self {
        self.ema_style = Some(value);
        self
    }

    #[inline(always)]
    pub fn iir_style(mut self, value: NOrderEmaIirStyle) -> Self {
        self.iir_style = Some(value);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }
}

#[derive(Clone, Debug)]
pub struct NOrderEmaBatchRange {
    pub period: (f64, f64, f64),
    pub order: (usize, usize, usize),
}

#[derive(Clone, Debug)]
pub struct NOrderEmaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<NOrderEmaParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Debug)]
pub struct NOrderEmaBatchBuilder {
    period_range: (f64, f64, f64),
    order_range: (usize, usize, usize),
    ema_style: Option<NOrderEmaStyle>,
    iir_style: Option<NOrderEmaIirStyle>,
    kernel: Kernel,
}

impl Default for NOrderEmaBatchBuilder {
    fn default() -> Self {
        Self {
            period_range: (DEFAULT_PERIOD, DEFAULT_PERIOD, 0.0),
            order_range: (DEFAULT_ORDER, DEFAULT_ORDER, 0),
            ema_style: None,
            iir_style: None,
            kernel: Kernel::Auto,
        }
    }
}

impl NOrderEmaBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Debug, Error)]
pub enum NOrderEmaError {
    #[error("n_order_ema: Input data slice is empty.")]
    EmptyInputData,
    #[error("n_order_ema: All values are NaN.")]
    AllValuesNaN,
    #[error("n_order_ema: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: f64, data_len: usize },
    #[error("n_order_ema: Invalid order: {order}")]
    InvalidOrder { order: usize },
    #[error("n_order_ema: Invalid ema_style: {value}")]
    InvalidEmaStyle { value: String },
    #[error("n_order_ema: Invalid iir_style: {value}")]
    InvalidIirStyle { value: String },
    #[error("n_order_ema: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("n_order_ema: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("n_order_ema: Invalid float range: start = {start}, end = {end}, step = {step}")]
    InvalidRangeF64 { start: f64, end: f64, step: f64 },
    #[error("n_order_ema: Invalid integer range: start = {start}, end = {end}, step = {step}")]
    InvalidRangeUsize {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("n_order_ema: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("n_order_ema: Mismatched output length: dst_len = {dst_len}, expected = {expected}")]
    MismatchedOutputLen { dst_len: usize, expected: usize },
}

#[derive(Clone, Copy, Debug)]
struct ResolvedParams {
    period: f64,
    order: usize,
    ema_style: NOrderEmaStyle,
    iir_style: NOrderEmaIirStyle,
}

#[derive(Clone, Debug)]
struct IirCoefficients {
    a: Vec<f64>,
    b: Vec<f64>,
}

#[derive(Clone, Debug)]
struct IirCoreFilter {
    coeffs: IirCoefficients,
    x_hist: Vec<f64>,
    y_hist: Vec<f64>,
    first_value: Option<f64>,
}

impl IirCoreFilter {
    #[inline(always)]
    fn new(period: f64, order: usize, style: NOrderEmaIirStyle) -> Self {
        let coeffs = build_coefficients(period, order, style);
        Self {
            x_hist: Vec::with_capacity(coeffs.b.len().saturating_sub(1)),
            y_hist: Vec::with_capacity(coeffs.a.len()),
            coeffs,
            first_value: None,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.x_hist.clear();
        self.y_hist.clear();
        self.first_value = None;
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<f64> {
        if !value.is_finite() {
            self.reset();
            return None;
        }

        let first = *self.first_value.get_or_insert(value);
        let mut acc = self.coeffs.b[0] * value;

        for m in 1..self.coeffs.b.len() {
            let x = self.x_hist.get(m - 1).copied().unwrap_or(first);
            acc += self.coeffs.b[m] * x;
        }
        for k in 0..self.coeffs.a.len() {
            let y = self.y_hist.get(k).copied().unwrap_or(first);
            acc -= self.coeffs.a[k] * y;
        }

        if self.coeffs.b.len() > 1 {
            self.x_hist.insert(0, value);
            if self.x_hist.len() > self.coeffs.b.len() - 1 {
                self.x_hist.pop();
            }
        }
        if !self.coeffs.a.is_empty() {
            self.y_hist.insert(0, acc);
            if self.y_hist.len() > self.coeffs.a.len() {
                self.y_hist.pop();
            }
        }

        Some(acc)
    }
}

#[derive(Clone, Debug)]
enum NOrderEmaState {
    Ema {
        f1: IirCoreFilter,
    },
    Dema {
        f1: IirCoreFilter,
        f2: IirCoreFilter,
    },
    Tema {
        f1: IirCoreFilter,
        f2: IirCoreFilter,
        f3: IirCoreFilter,
    },
    Hema {
        short: IirCoreFilter,
        base: IirCoreFilter,
        final_filter: IirCoreFilter,
    },
}

#[derive(Clone, Debug)]
pub struct NOrderEmaStream {
    state: NOrderEmaState,
    warmup: usize,
    count: usize,
}

impl NOrderEmaStream {
    pub fn try_new(params: NOrderEmaParams) -> Result<Self, NOrderEmaError> {
        let resolved = resolve_params(&params, 0)?;
        Ok(Self::new_resolved(resolved))
    }

    #[inline(always)]
    fn new_resolved(resolved: ResolvedParams) -> Self {
        let period = resolved.period;
        let order = resolved.order;
        let style = resolved.iir_style;
        let state = match resolved.ema_style {
            NOrderEmaStyle::Ema => NOrderEmaState::Ema {
                f1: IirCoreFilter::new(period, order, style),
            },
            NOrderEmaStyle::Dema => NOrderEmaState::Dema {
                f1: IirCoreFilter::new(period, order, style),
                f2: IirCoreFilter::new(period, order, style),
            },
            NOrderEmaStyle::Tema => NOrderEmaState::Tema {
                f1: IirCoreFilter::new(period, order, style),
                f2: IirCoreFilter::new(period, order, style),
                f3: IirCoreFilter::new(period, order, style),
            },
            NOrderEmaStyle::Hema => NOrderEmaState::Hema {
                short: IirCoreFilter::new((period * 0.5).max(1.0), order, style),
                base: IirCoreFilter::new(period, order, style),
                final_filter: IirCoreFilter::new(period.sqrt().max(1.0), order, style),
            },
        };

        Self {
            state,
            warmup: warmup_len(resolved),
            count: 0,
        }
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.count = 0;
        match &mut self.state {
            NOrderEmaState::Ema { f1 } => f1.reset(),
            NOrderEmaState::Dema { f1, f2 } => {
                f1.reset();
                f2.reset();
            }
            NOrderEmaState::Tema { f1, f2, f3 } => {
                f1.reset();
                f2.reset();
                f3.reset();
            }
            NOrderEmaState::Hema {
                short,
                base,
                final_filter,
            } => {
                short.reset();
                base.reset();
                final_filter.reset();
            }
        }
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        if !value.is_finite() {
            self.reset();
            return None;
        }

        let out = match &mut self.state {
            NOrderEmaState::Ema { f1 } => f1.update(value).unwrap(),
            NOrderEmaState::Dema { f1, f2 } => {
                let e1 = f1.update(value).unwrap();
                let e2 = f2.update(e1).unwrap();
                2.0 * e1 - e2
            }
            NOrderEmaState::Tema { f1, f2, f3 } => {
                let e1 = f1.update(value).unwrap();
                let e2 = f2.update(e1).unwrap();
                let e3 = f3.update(e2).unwrap();
                3.0 * (e1 - e2) + e3
            }
            NOrderEmaState::Hema {
                short,
                base,
                final_filter,
            } => {
                let e1 = short.update(value).unwrap();
                let e2 = base.update(value).unwrap();
                final_filter.update(2.0 * e1 - e2).unwrap()
            }
        };

        self.count += 1;
        if self.count > self.warmup {
            Some(out)
        } else {
            None
        }
    }
}

#[inline(always)]
fn ceil_period(period: f64) -> usize {
    period.ceil().max(1.0) as usize
}

#[inline(always)]
fn base_lookback(period: f64, order: usize) -> usize {
    order.saturating_mul(ceil_period(period).saturating_sub(1))
}

#[inline(always)]
fn warmup_len(params: ResolvedParams) -> usize {
    match params.ema_style {
        NOrderEmaStyle::Ema => base_lookback(params.period, params.order),
        NOrderEmaStyle::Dema => base_lookback(params.period, params.order),
        NOrderEmaStyle::Tema => base_lookback(params.period, params.order).saturating_mul(3),
        NOrderEmaStyle::Hema => {
            let short = base_lookback((params.period * 0.5).max(1.0), params.order);
            let base = base_lookback(params.period, params.order);
            let final_lb = base_lookback(params.period.sqrt().max(1.0), params.order);
            short.max(base).saturating_add(final_lb)
        }
    }
}

#[inline(always)]
fn required_valid_len(params: ResolvedParams) -> usize {
    let base = base_lookback(params.period, params.order);
    match params.ema_style {
        NOrderEmaStyle::Ema => warmup_len(params).saturating_add(1),
        NOrderEmaStyle::Dema => base
            .saturating_mul(2)
            .max(warmup_len(params).saturating_add(1)),
        NOrderEmaStyle::Tema => warmup_len(params).saturating_add(1),
        NOrderEmaStyle::Hema => warmup_len(params).saturating_add(1),
    }
}

#[inline(always)]
fn binomial(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let kk = k.min(n - k);
    let mut out = 1.0f64;
    for i in 0..kk {
        out *= (n - i) as f64;
        out /= (i + 1) as f64;
    }
    out
}

#[inline(always)]
fn clamp_omega(w: f64) -> f64 {
    w.clamp(1.0e-6, std::f64::consts::PI - 1.0e-6)
}

fn build_coefficients(period: f64, order: usize, style: NOrderEmaIirStyle) -> IirCoefficients {
    let fc = 2.0 / (period + 1.0);
    let mut a = Vec::with_capacity(order);
    let mut b = Vec::new();
    match style {
        NOrderEmaIirStyle::AllPole => {
            let r = 1.0 - fc;
            for k in 1..=order {
                a.push(binomial(order, k) * (-r).powi(k as i32));
            }
            b.push(fc.powi(order as i32));
            if order > 1 {
                b.resize(order, 0.0);
            }
        }
        NOrderEmaIirStyle::ImpulseMatched => {
            let r = 1.0 - fc;
            for k in 1..=order {
                a.push(binomial(order, k) * (-r).powi(k as i32));
            }
            let mut sum = 0.0;
            for m in 0..order {
                sum += binomial(m + order - 1, order - 1) * r.powi(m as i32);
            }
            let s = if sum != 0.0 { 1.0 / sum } else { 1.0 };
            for m in 0..order {
                b.push(
                    fc.powi(order as i32)
                        * binomial(m + order - 1, order - 1)
                        * r.powi(m as i32)
                        * s,
                );
            }
        }
        NOrderEmaIirStyle::MatchedZ => {
            let p = 1.0 - fc;
            for k in 1..=order {
                a.push(binomial(order, k) * (-p).powi(k as i32));
            }
            let dc_denom = (1.0 - p).powi(order as i32);
            let mut sum = 0.0;
            for m in 0..order {
                sum += binomial(order - 1, m);
            }
            let g = if sum != 0.0 { dc_denom / sum } else { 0.0 };
            for m in 0..order {
                b.push(binomial(order - 1, m) * g);
            }
        }
        NOrderEmaIirStyle::Bilinear => {
            let fc_d = 2.0 / period.max(1.0);
            let w_c = clamp_omega(fc_d);
            let k = (w_c * 0.5).tan();
            let q = (k - 1.0) / (k + 1.0);
            for idx in 1..=order {
                a.push(binomial(order, idx) * q.powi(idx as i32));
            }
            let g = (k / (k + 1.0)).powi(order as i32);
            for m in 0..=order {
                b.push(binomial(order, m) * g);
            }
        }
    }
    IirCoefficients { a, b }
}

#[inline(always)]
fn resolve_params(params: &NOrderEmaParams, len: usize) -> Result<ResolvedParams, NOrderEmaError> {
    let period = params.period.unwrap_or(DEFAULT_PERIOD);
    if !period.is_finite() || period < 1.0 || (len > 0 && period.ceil() as usize > len) {
        return Err(NOrderEmaError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    let order = params.order.unwrap_or(DEFAULT_ORDER);
    if order == 0 || order > MAX_ORDER {
        return Err(NOrderEmaError::InvalidOrder { order });
    }
    let ema_style =
        NOrderEmaStyle::from_str(params.ema_style.as_deref().unwrap_or(DEFAULT_EMA_STYLE))
            .ok_or_else(|| NOrderEmaError::InvalidEmaStyle {
                value: params
                    .ema_style
                    .clone()
                    .unwrap_or_else(|| DEFAULT_EMA_STYLE.to_string()),
            })?;
    let iir_style =
        NOrderEmaIirStyle::from_str(params.iir_style.as_deref().unwrap_or(DEFAULT_IIR_STYLE))
            .ok_or_else(|| NOrderEmaError::InvalidIirStyle {
                value: params
                    .iir_style
                    .clone()
                    .unwrap_or_else(|| DEFAULT_IIR_STYLE.to_string()),
            })?;

    Ok(ResolvedParams {
        period,
        order,
        ema_style,
        iir_style,
    })
}

#[inline(always)]
fn validate_input(data: &[f64], resolved: ResolvedParams) -> Result<(), NOrderEmaError> {
    if data.is_empty() {
        return Err(NOrderEmaError::EmptyInputData);
    }
    let needed = required_valid_len(resolved);
    let mut best = 0usize;
    let mut cur = 0usize;
    let mut any = false;
    for &value in data {
        if value.is_finite() {
            any = true;
            cur += 1;
            if cur >= needed {
                return Ok(());
            }
            if cur > best {
                best = cur;
            }
        } else {
            cur = 0;
        }
    }
    if !any {
        return Err(NOrderEmaError::AllValuesNaN);
    }
    if best < needed {
        return Err(NOrderEmaError::NotEnoughValidData {
            needed,
            valid: best,
        });
    }
    Ok(())
}

#[inline]
pub fn n_order_ema(input: &NOrderEmaInput) -> Result<NOrderEmaOutput, NOrderEmaError> {
    n_order_ema_with_kernel(input, Kernel::Auto)
}

pub fn n_order_ema_with_kernel(
    input: &NOrderEmaInput,
    _kernel: Kernel,
) -> Result<NOrderEmaOutput, NOrderEmaError> {
    let data = input.as_ref();
    let resolved = resolve_params(&input.params, data.len())?;
    validate_input(data, resolved)?;
    let warmup = warmup_len(resolved);
    let mut output = alloc_with_nan_prefix(data.len(), warmup.min(data.len()));
    n_order_ema_compute_into(data, resolved, &mut output);
    Ok(NOrderEmaOutput { values: output })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn n_order_ema_into(input: &NOrderEmaInput, out: &mut [f64]) -> Result<(), NOrderEmaError> {
    n_order_ema_into_slice(out, input, Kernel::Auto)
}

pub fn n_order_ema_into_slice(
    out: &mut [f64],
    input: &NOrderEmaInput,
    _kernel: Kernel,
) -> Result<(), NOrderEmaError> {
    let data = input.as_ref();
    if out.len() != data.len() {
        return Err(NOrderEmaError::OutputLengthMismatch {
            expected: data.len(),
            got: out.len(),
        });
    }
    let resolved = resolve_params(&input.params, data.len())?;
    validate_input(data, resolved)?;
    n_order_ema_compute_into(data, resolved, out);
    Ok(())
}

#[inline(always)]
fn n_order_ema_compute_into(data: &[f64], resolved: ResolvedParams, out: &mut [f64]) {
    if is_default_ema(resolved) {
        n_order_ema_default_ema_into(data, out);
        return;
    }

    let mut stream = NOrderEmaStream::new_resolved(resolved);
    for (dst, &value) in out.iter_mut().zip(data.iter()) {
        *dst = stream.update(value).unwrap_or(f64::NAN);
    }
}

#[inline(always)]
fn is_default_ema(resolved: ResolvedParams) -> bool {
    resolved.period == DEFAULT_PERIOD
        && resolved.order == DEFAULT_ORDER
        && resolved.ema_style == NOrderEmaStyle::Ema
        && resolved.iir_style == NOrderEmaIirStyle::ImpulseMatched
}

#[inline(always)]
fn n_order_ema_default_ema_into(data: &[f64], out: &mut [f64]) {
    let mut prev = 0.0;
    let mut initialized = false;
    let mut count = 0usize;

    for (dst, &value) in out.iter_mut().zip(data.iter()) {
        if !value.is_finite() {
            initialized = false;
            count = 0;
            *dst = f64::NAN;
            continue;
        }

        if initialized {
            let old = prev;
            let mut acc = 0.2 * value;
            acc -= -0.8 * old;
            prev = acc;
            count += 1;
        } else {
            let mut acc = 0.2 * value;
            acc -= -0.8 * value;
            prev = acc;
            initialized = true;
            count = 1;
        }

        *dst = if count > 8 { prev } else { f64::NAN };
    }
}

#[inline(always)]
fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, NOrderEmaError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(NOrderEmaError::InvalidRangeF64 { start, end, step });
    }
    if step == 0.0 || (start - end).abs() <= f64::EPSILON {
        return Ok(vec![start]);
    }
    let ascending = end >= start;
    if (ascending && step < 0.0) || (!ascending && step > 0.0) {
        return Err(NOrderEmaError::InvalidRangeF64 { start, end, step });
    }
    let mut values = Vec::new();
    let mut cur = start;
    if ascending {
        while cur <= end + 1.0e-12 {
            values.push(cur);
            cur += step;
        }
    } else {
        while cur >= end - 1.0e-12 {
            values.push(cur);
            cur += step;
        }
    }
    if values.is_empty() {
        return Err(NOrderEmaError::InvalidRangeF64 { start, end, step });
    }
    Ok(values)
}

#[inline(always)]
fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, NOrderEmaError> {
    if step == 0 || start == end {
        return Ok(vec![start]);
    }
    let mut values = Vec::new();
    if start < end {
        for value in (start..=end).step_by(step) {
            values.push(value);
        }
    } else {
        let mut cur = start;
        while cur >= end {
            values.push(cur);
            if cur < step {
                break;
            }
            cur -= step;
        }
    }
    if values.is_empty() {
        return Err(NOrderEmaError::InvalidRangeUsize { start, end, step });
    }
    Ok(values)
}

pub fn expand_grid_n_order_ema(
    sweep: &NOrderEmaBatchRange,
    fixed: &NOrderEmaParams,
) -> Result<Vec<NOrderEmaParams>, NOrderEmaError> {
    let periods = axis_f64(sweep.period)?;
    let orders = axis_usize(sweep.order)?;
    let mut combos = Vec::with_capacity(periods.len().saturating_mul(orders.len()));
    for &period in &periods {
        for &order in &orders {
            combos.push(NOrderEmaParams {
                period: Some(period),
                order: Some(order),
                ema_style: Some(
                    fixed
                        .ema_style
                        .clone()
                        .unwrap_or_else(|| DEFAULT_EMA_STYLE.to_string()),
                ),
                iir_style: Some(
                    fixed
                        .iir_style
                        .clone()
                        .unwrap_or_else(|| DEFAULT_IIR_STYLE.to_string()),
                ),
            });
        }
    }
    Ok(combos)
}

pub fn n_order_ema_batch_with_kernel(
    data: &[f64],
    sweep: &NOrderEmaBatchRange,
    fixed: &NOrderEmaParams,
    kernel: Kernel,
) -> Result<NOrderEmaBatchOutput, NOrderEmaError> {
    let input = NOrderEmaInput::from_slice(data, fixed.clone());
    n_order_ema_batch_from_input_with_kernel(&input, sweep, kernel)
}

pub fn n_order_ema_batch_from_input_with_kernel(
    input: &NOrderEmaInput,
    sweep: &NOrderEmaBatchRange,
    kernel: Kernel,
) -> Result<NOrderEmaBatchOutput, NOrderEmaError> {
    let data = input.as_ref();
    let combos = expand_grid_n_order_ema(sweep, &input.params)?;
    let rows = combos.len();
    let cols = data.len();
    let mut warmups = Vec::with_capacity(rows);
    for combo in &combos {
        let resolved = resolve_params(combo, cols)?;
        validate_input(data, resolved)?;
        warmups.push(warmup_len(resolved));
    }
    let mut raw = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut raw, cols, &warmups);
    let mut values =
        unsafe { Vec::from_raw_parts(raw.as_mut_ptr() as *mut f64, raw.len(), raw.capacity()) };
    std::mem::forget(raw);
    n_order_ema_batch_inner_into(data, &combos, kernel, true, &mut values)?;
    Ok(NOrderEmaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

fn n_order_ema_batch_inner_into(
    data: &[f64],
    combos: &[NOrderEmaParams],
    kernel: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<(), NOrderEmaError> {
    match kernel {
        Kernel::Auto
        | Kernel::Scalar
        | Kernel::ScalarBatch
        | Kernel::Avx2
        | Kernel::Avx2Batch
        | Kernel::Avx512
        | Kernel::Avx512Batch => {}
        other => return Err(NOrderEmaError::InvalidKernelForBatch(other)),
    }

    let cols = data.len();
    let expected = combos.len().saturating_mul(cols);
    if out.len() != expected {
        return Err(NOrderEmaError::MismatchedOutputLen {
            dst_len: out.len(),
            expected,
        });
    }

    let worker = |row: usize, dst: &mut [f64]| -> Result<(), NOrderEmaError> {
        let input = NOrderEmaInput::from_slice(data, combos[row].clone());
        n_order_ema_into_slice(dst, &input, Kernel::Scalar)
    };

    if parallel && combos.len() > 1 {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out.par_chunks_mut(cols)
                .enumerate()
                .try_for_each(|(row, dst)| worker(row, dst))?;
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, dst) in out.chunks_mut(cols).enumerate() {
                worker(row, dst)?;
            }
        }
    } else {
        for (row, dst) in out.chunks_mut(cols).enumerate() {
            worker(row, dst)?;
        }
    }
    Ok(())
}

impl NOrderEmaBuilder {
    #[inline(always)]
    pub fn apply(self, candles: &Candles) -> Result<NOrderEmaOutput, NOrderEmaError> {
        n_order_ema_with_kernel(
            &NOrderEmaInput::from_candles(
                candles,
                "close",
                NOrderEmaParams {
                    period: self.period,
                    order: self.order,
                    ema_style: self.ema_style.map(|v| v.as_str().to_string()),
                    iir_style: self.iir_style.map(|v| v.as_str().to_string()),
                },
            ),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slice(self, data: &[f64]) -> Result<NOrderEmaOutput, NOrderEmaError> {
        n_order_ema_with_kernel(
            &NOrderEmaInput::from_slice(
                data,
                NOrderEmaParams {
                    period: self.period,
                    order: self.order,
                    ema_style: self.ema_style.map(|v| v.as_str().to_string()),
                    iir_style: self.iir_style.map(|v| v.as_str().to_string()),
                },
            ),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<NOrderEmaStream, NOrderEmaError> {
        NOrderEmaStream::try_new(NOrderEmaParams {
            period: self.period,
            order: self.order,
            ema_style: self.ema_style.map(|v| v.as_str().to_string()),
            iir_style: self.iir_style.map(|v| v.as_str().to_string()),
        })
    }
}

impl NOrderEmaBatchBuilder {
    #[inline(always)]
    pub fn period_range(mut self, value: (f64, f64, f64)) -> Self {
        self.period_range = value;
        self
    }

    #[inline(always)]
    pub fn order_range(mut self, value: (usize, usize, usize)) -> Self {
        self.order_range = value;
        self
    }

    #[inline(always)]
    pub fn ema_style(mut self, value: NOrderEmaStyle) -> Self {
        self.ema_style = Some(value);
        self
    }

    #[inline(always)]
    pub fn iir_style(mut self, value: NOrderEmaIirStyle) -> Self {
        self.iir_style = Some(value);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline(always)]
    pub fn apply_slice(self, data: &[f64]) -> Result<NOrderEmaBatchOutput, NOrderEmaError> {
        n_order_ema_batch_with_kernel(
            data,
            &NOrderEmaBatchRange {
                period: self.period_range,
                order: self.order_range,
            },
            &NOrderEmaParams {
                period: None,
                order: None,
                ema_style: self.ema_style.map(|v| v.as_str().to_string()),
                iir_style: self.iir_style.map(|v| v.as_str().to_string()),
            },
            self.kernel,
        )
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "n_order_ema")]
#[pyo3(signature = (data, period=DEFAULT_PERIOD, order=DEFAULT_ORDER, ema_style=DEFAULT_EMA_STYLE, iir_style=DEFAULT_IIR_STYLE, kernel=None))]
pub fn n_order_ema_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period: f64,
    order: usize,
    ema_style: &str,
    iir_style: &str,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let slice = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let input = NOrderEmaInput::from_slice(
        slice,
        NOrderEmaParams {
            period: Some(period),
            order: Some(order),
            ema_style: Some(ema_style.to_string()),
            iir_style: Some(iir_style.to_string()),
        },
    );
    let values = py
        .allow_threads(|| n_order_ema_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(values.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction(name = "n_order_ema_batch")]
#[pyo3(signature = (data, period_range, order_range=(DEFAULT_ORDER, DEFAULT_ORDER, 0), ema_style=DEFAULT_EMA_STYLE, iir_style=DEFAULT_IIR_STYLE, kernel=None))]
pub fn n_order_ema_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period_range: (f64, f64, f64),
    order_range: (usize, usize, usize),
    ema_style: &str,
    iir_style: &str,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let slice = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let batch_kernel = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        other => other,
    };
    let output = py
        .allow_threads(|| {
            n_order_ema_batch_with_kernel(
                slice,
                &NOrderEmaBatchRange {
                    period: period_range,
                    order: order_range,
                },
                &NOrderEmaParams {
                    period: None,
                    order: None,
                    ema_style: Some(ema_style.to_string()),
                    iir_style: Some(iir_style.to_string()),
                },
                batch_kernel,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item(
        "values",
        output
            .values
            .into_pyarray(py)
            .reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "periods",
        output
            .combos
            .iter()
            .map(|p| p.period.unwrap_or(DEFAULT_PERIOD))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "orders",
        output
            .combos
            .iter()
            .map(|p| p.order.unwrap_or(DEFAULT_ORDER) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", output.rows)?;
    dict.set_item("cols", output.cols)?;
    Ok(dict.into())
}

#[cfg(feature = "python")]
#[pyclass(name = "NOrderEmaStream")]
pub struct NOrderEmaStreamPy {
    stream: NOrderEmaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl NOrderEmaStreamPy {
    #[new]
    #[pyo3(signature = (period=DEFAULT_PERIOD, order=DEFAULT_ORDER, ema_style=DEFAULT_EMA_STYLE, iir_style=DEFAULT_IIR_STYLE))]
    pub fn new(period: f64, order: usize, ema_style: &str, iir_style: &str) -> PyResult<Self> {
        let stream = NOrderEmaStream::try_new(NOrderEmaParams {
            period: Some(period),
            order: Some(order),
            ema_style: Some(ema_style.to_string()),
            iir_style: Some(iir_style.to_string()),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    pub fn update(&mut self, value: f64) -> f64 {
        self.stream.update(value).unwrap_or(f64::NAN)
    }
}

#[cfg(feature = "python")]
pub fn register_n_order_ema_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(n_order_ema_py, m)?)?;
    m.add_function(wrap_pyfunction!(n_order_ema_batch_py, m)?)?;
    m.add_class::<NOrderEmaStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn n_order_ema_js(
    data: &[f64],
    period: f64,
    order: usize,
    ema_style: &str,
    iir_style: &str,
) -> Result<Vec<f64>, JsValue> {
    let input = NOrderEmaInput::from_slice(
        data,
        NOrderEmaParams {
            period: Some(period),
            order: Some(order),
            ema_style: Some(ema_style.to_string()),
            iir_style: Some(iir_style.to_string()),
        },
    );
    let mut output = alloc_with_nan_prefix(data.len(), 0);
    n_order_ema_into_slice(&mut output, &input, Kernel::Scalar)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(output)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct NOrderEmaBatchConfig {
    pub period_range: (f64, f64, f64),
    pub order_range: (usize, usize, usize),
    pub ema_style: Option<String>,
    pub iir_style: Option<String>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct NOrderEmaBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<NOrderEmaParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn n_order_ema_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: NOrderEmaBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
    let output = n_order_ema_batch_with_kernel(
        data,
        &NOrderEmaBatchRange {
            period: config.period_range,
            order: config.order_range,
        },
        &NOrderEmaParams {
            period: None,
            order: None,
            ema_style: Some(
                config
                    .ema_style
                    .unwrap_or_else(|| DEFAULT_EMA_STYLE.to_string()),
            ),
            iir_style: Some(
                config
                    .iir_style
                    .unwrap_or_else(|| DEFAULT_IIR_STYLE.to_string()),
            ),
        },
        Kernel::ScalarBatch,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    serde_wasm_bindgen::to_value(&NOrderEmaBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn n_order_ema_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn n_order_ema_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn n_order_ema_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: f64,
    order: usize,
    ema_style: &str,
    iir_style: &str,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to n_order_ema_into"));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let input = NOrderEmaInput::from_slice(
            data,
            NOrderEmaParams {
                period: Some(period),
                order: Some(order),
                ema_style: Some(ema_style.to_string()),
                iir_style: Some(iir_style.to_string()),
            },
        );
        if in_ptr == out_ptr {
            let mut tmp = alloc_with_nan_prefix(len, 0);
            n_order_ema_into_slice(&mut tmp, &input, Kernel::Scalar)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            std::slice::from_raw_parts_mut(out_ptr, len).copy_from_slice(&tmp);
        } else {
            n_order_ema_into_slice(
                std::slice::from_raw_parts_mut(out_ptr, len),
                &input,
                Kernel::Scalar,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn n_order_ema_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: f64,
    period_end: f64,
    period_step: f64,
    order_start: usize,
    order_end: usize,
    order_step: usize,
    ema_style: &str,
    iir_style: &str,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to n_order_ema_batch_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let sweep = NOrderEmaBatchRange {
            period: (period_start, period_end, period_step),
            order: (order_start, order_end, order_step),
        };
        let combos = expand_grid_n_order_ema(
            &sweep,
            &NOrderEmaParams {
                period: None,
                order: None,
                ema_style: Some(ema_style.to_string()),
                iir_style: Some(iir_style.to_string()),
            },
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        n_order_ema_batch_inner_into(
            data,
            &combos,
            Kernel::ScalarBatch,
            false,
            std::slice::from_raw_parts_mut(out_ptr, rows * len),
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn n_order_ema_output_into_js(
    data: &[f64],
    period: f64,
    order: usize,
    ema_style: &str,
    iir_style: &str,
    out: &js_sys::Float64Array,
) -> Result<usize, JsValue> {
    let values = n_order_ema_js(data, period, order, ema_style, iir_style)?;
    crate::write_wasm_f64_output("n_order_ema_output_into_js", &values, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn n_order_ema_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = n_order_ema_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs("n_order_ema_batch_output_into_js", &value, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::moving_averages::ma::{ma, ma_with_kernel, MaData};
    use crate::indicators::moving_averages::ma_batch::{
        ma_batch_with_kernel_and_typed_params, MaBatchParamKV, MaBatchParamValue,
    };

    fn sample_data(len: usize) -> Vec<f64> {
        (0..len)
            .map(|i| 100.0 + i as f64 * 0.2 + (i as f64 * 0.17).sin() * 3.0)
            .collect()
    }

    #[test]
    fn n_order_ema_constant_series_stays_constant() -> Result<(), Box<dyn Error>> {
        let data = vec![42.0; 128];
        let out = n_order_ema(&NOrderEmaInput::from_slice(
            &data,
            NOrderEmaParams::default(),
        ))?
        .values;
        for value in out.into_iter().skip(32) {
            assert!((value - 42.0).abs() <= 1e-12);
        }
        Ok(())
    }

    #[test]
    fn n_order_ema_into_matches_api() -> Result<(), Box<dyn Error>> {
        let data = sample_data(256);
        let input = NOrderEmaInput::from_slice(
            &data,
            NOrderEmaParams {
                period: Some(12.5),
                order: Some(3),
                ema_style: Some("tema".to_string()),
                iir_style: Some("matched_z".to_string()),
            },
        );
        let baseline = n_order_ema(&input)?.values;
        let mut out = vec![0.0; data.len()];
        n_order_ema_into_slice(&mut out, &input, Kernel::Auto)?;
        for (a, b) in baseline.iter().zip(out.iter()) {
            assert!((a.is_nan() && b.is_nan()) || (*a - *b).abs() <= 1e-12);
        }
        Ok(())
    }

    #[test]
    fn n_order_ema_stream_matches_batch() -> Result<(), Box<dyn Error>> {
        let data = sample_data(256);
        let params = NOrderEmaParams {
            period: Some(15.0),
            order: Some(2),
            ema_style: Some("hema".to_string()),
            iir_style: Some("bilinear".to_string()),
        };
        let batch = n_order_ema(&NOrderEmaInput::from_slice(&data, params.clone()))?.values;
        let mut stream = NOrderEmaStream::try_new(params)?;
        let streamed = data
            .iter()
            .map(|&v| stream.update(v).unwrap_or(f64::NAN))
            .collect::<Vec<_>>();
        for (a, b) in batch.iter().zip(streamed.iter()) {
            assert!((a.is_nan() && b.is_nan()) || (*a - *b).abs() <= 1e-12);
        }
        Ok(())
    }

    #[test]
    fn n_order_ema_batch_rows_match_combos() -> Result<(), Box<dyn Error>> {
        let data = sample_data(128);
        let out = n_order_ema_batch_with_kernel(
            &data,
            &NOrderEmaBatchRange {
                period: (9.0, 11.0, 1.0),
                order: (1, 2, 1),
            },
            &NOrderEmaParams {
                period: None,
                order: None,
                ema_style: Some("ema".to_string()),
                iir_style: Some("impulse_matched".to_string()),
            },
            Kernel::ScalarBatch,
        )?;
        assert_eq!(out.rows, 6);
        assert_eq!(out.cols, data.len());
        assert_eq!(out.values.len(), out.rows * out.cols);
        Ok(())
    }

    #[test]
    fn n_order_ema_ma_dispatch_matches_direct() -> Result<(), Box<dyn Error>> {
        let data = sample_data(128);
        let direct = n_order_ema(&NOrderEmaInput::from_slice(
            &data,
            NOrderEmaParams {
                period: Some(10.0),
                order: Some(1),
                ema_style: Some("ema".to_string()),
                iir_style: Some("impulse_matched".to_string()),
            },
        ))?
        .values;
        let dispatched = ma("n_order_ema", MaData::Slice(&data), 10)?;
        assert_eq!(direct.len(), dispatched.len());
        for (a, b) in direct.iter().zip(dispatched.iter()) {
            assert!((a.is_nan() && b.is_nan()) || (*a - *b).abs() <= 1e-12);
        }
        let dispatched_kernel =
            ma_with_kernel("n_order_ema", MaData::Slice(&data), 10, Kernel::Auto)?;
        for (a, b) in direct.iter().zip(dispatched_kernel.iter()) {
            assert!((a.is_nan() && b.is_nan()) || (*a - *b).abs() <= 1e-12);
        }
        Ok(())
    }

    #[test]
    fn n_order_ema_ma_batch_typed_params_match_direct() -> Result<(), Box<dyn Error>> {
        let data = sample_data(96);
        let params = [
            MaBatchParamKV {
                key: "order",
                value: MaBatchParamValue::Int(2),
            },
            MaBatchParamKV {
                key: "ema_style",
                value: MaBatchParamValue::EnumString("tema"),
            },
            MaBatchParamKV {
                key: "iir_style",
                value: MaBatchParamValue::EnumString("matched_z"),
            },
        ];
        let got = ma_batch_with_kernel_and_typed_params(
            "n_order_ema",
            MaData::Slice(&data),
            (10, 12, 2),
            Kernel::Auto,
            &params,
        )?;
        assert_eq!(got.rows, 2);
        assert_eq!(got.cols, data.len());
        Ok(())
    }
}
