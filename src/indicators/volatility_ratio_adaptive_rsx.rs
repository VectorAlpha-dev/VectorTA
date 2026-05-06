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

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_uninit_f64, detect_best_batch_kernel, init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

const DEFAULT_PERIOD: usize = 14;
const DEFAULT_SPEED: f64 = 0.5;
const DEFAULT_SOURCE: &str = "close";

#[inline(always)]
fn source_slice<'a>(candles: &'a Candles, source: &str) -> &'a [f64] {
    match source {
        DEFAULT_SOURCE => &candles.close,
        "open" => &candles.open,
        "high" => &candles.high,
        "low" => &candles.low,
        "volume" => &candles.volume,
        "hl2" => &candles.hl2,
        "hlc3" => &candles.hlc3,
        "ohlc4" => &candles.ohlc4,
        "hlcc4" | "hlcc" => &candles.hlcc4,
        _ => source_type(candles, source),
    }
}

impl<'a> AsRef<[f64]> for VolatilityRatioAdaptiveRsxInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            VolatilityRatioAdaptiveRsxData::Slice(slice) => slice,
            VolatilityRatioAdaptiveRsxData::Candles { candles, source } => {
                source_slice(candles, source)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum VolatilityRatioAdaptiveRsxData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct VolatilityRatioAdaptiveRsxOutput {
    pub line: Vec<f64>,
    pub signal: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct VolatilityRatioAdaptiveRsxParams {
    pub period: Option<usize>,
    pub speed: Option<f64>,
}

impl Default for VolatilityRatioAdaptiveRsxParams {
    fn default() -> Self {
        Self {
            period: Some(DEFAULT_PERIOD),
            speed: Some(DEFAULT_SPEED),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VolatilityRatioAdaptiveRsxInput<'a> {
    pub data: VolatilityRatioAdaptiveRsxData<'a>,
    pub params: VolatilityRatioAdaptiveRsxParams,
}

impl<'a> VolatilityRatioAdaptiveRsxInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: VolatilityRatioAdaptiveRsxParams,
    ) -> Self {
        Self {
            data: VolatilityRatioAdaptiveRsxData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: VolatilityRatioAdaptiveRsxParams) -> Self {
        Self {
            data: VolatilityRatioAdaptiveRsxData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(
            candles,
            DEFAULT_SOURCE,
            VolatilityRatioAdaptiveRsxParams::default(),
        )
    }

    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(DEFAULT_PERIOD)
    }

    #[inline]
    pub fn get_speed(&self) -> f64 {
        self.params.speed.unwrap_or(DEFAULT_SPEED)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct VolatilityRatioAdaptiveRsxBuilder {
    period: Option<usize>,
    speed: Option<f64>,
    kernel: Kernel,
}

impl Default for VolatilityRatioAdaptiveRsxBuilder {
    fn default() -> Self {
        Self {
            period: None,
            speed: None,
            kernel: Kernel::Auto,
        }
    }
}

impl VolatilityRatioAdaptiveRsxBuilder {
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
    pub fn speed(mut self, value: f64) -> Self {
        self.speed = Some(value);
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
    ) -> Result<VolatilityRatioAdaptiveRsxOutput, VolatilityRatioAdaptiveRsxError> {
        let params = VolatilityRatioAdaptiveRsxParams {
            period: self.period,
            speed: self.speed,
        };
        let input = VolatilityRatioAdaptiveRsxInput::from_candles(candles, DEFAULT_SOURCE, params);
        volatility_ratio_adaptive_rsx_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<VolatilityRatioAdaptiveRsxOutput, VolatilityRatioAdaptiveRsxError> {
        let params = VolatilityRatioAdaptiveRsxParams {
            period: self.period,
            speed: self.speed,
        };
        let input = VolatilityRatioAdaptiveRsxInput::from_slice(data, params);
        volatility_ratio_adaptive_rsx_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<VolatilityRatioAdaptiveRsxStream, VolatilityRatioAdaptiveRsxError> {
        let params = VolatilityRatioAdaptiveRsxParams {
            period: self.period,
            speed: self.speed,
        };
        VolatilityRatioAdaptiveRsxStream::try_new(params)
    }
}

#[derive(Debug, Error)]
pub enum VolatilityRatioAdaptiveRsxError {
    #[error("volatility_ratio_adaptive_rsx: Input data slice is empty.")]
    EmptyInputData,
    #[error("volatility_ratio_adaptive_rsx: All source values are invalid.")]
    AllValuesNaN,
    #[error("volatility_ratio_adaptive_rsx: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("volatility_ratio_adaptive_rsx: Invalid speed: {speed}")]
    InvalidSpeed { speed: f64 },
    #[error(
        "volatility_ratio_adaptive_rsx: Not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "volatility_ratio_adaptive_rsx: Output length mismatch: expected = {expected}, got = {got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("volatility_ratio_adaptive_rsx: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("volatility_ratio_adaptive_rsx: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("volatility_ratio_adaptive_rsx: Invalid input: {0}")]
    InvalidInput(&'static str),
}

#[inline(always)]
fn is_valid_source(v: f64) -> bool {
    v.is_finite()
}

#[inline(always)]
fn first_valid_source(data: &[f64]) -> Option<usize> {
    data.iter().position(|&v| is_valid_source(v))
}

#[inline(always)]
fn line_warmup(period: usize, first: usize) -> usize {
    first + (2 * period) - 2
}

#[inline(always)]
fn signal_warmup(period: usize, first: usize) -> usize {
    first + (2 * period) - 1
}

#[inline(always)]
fn biased_std_from_sums(sum: f64, sum_sq: f64, period: usize) -> f64 {
    let n = period as f64;
    let centered = (sum_sq - (sum * sum) / n).max(0.0);
    (centered / n).sqrt()
}

#[inline(always)]
fn nz(v: f64) -> f64 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

#[inline(always)]
fn push_window_sum_sumsq(
    window: &mut [f64],
    head: &mut usize,
    count: &mut usize,
    valid: &mut usize,
    sum: &mut f64,
    sum_sq: &mut f64,
    value: f64,
) {
    if *count == window.len() {
        let old = window[*head];
        if old.is_finite() {
            *valid -= 1;
            *sum -= old;
            *sum_sq -= old * old;
        }
    } else {
        *count += 1;
    }

    window[*head] = value;
    *head += 1;
    if *head == window.len() {
        *head = 0;
    }

    if value.is_finite() {
        *valid += 1;
        *sum += value;
        *sum_sq += value * value;
    }
}

#[inline(always)]
fn push_window_sum(
    window: &mut [f64],
    head: &mut usize,
    count: &mut usize,
    valid: &mut usize,
    sum: &mut f64,
    value: f64,
) {
    if *count == window.len() {
        let old = window[*head];
        if old.is_finite() {
            *valid -= 1;
            *sum -= old;
        }
    } else {
        *count += 1;
    }

    window[*head] = value;
    *head += 1;
    if *head == window.len() {
        *head = 0;
    }

    if value.is_finite() {
        *valid += 1;
        *sum += value;
    }
}

#[derive(Clone, Debug)]
struct VrarsxState {
    prev_src_out: f64,
    prev_line: f64,
    price_window: Vec<f64>,
    price_head: usize,
    price_count: usize,
    price_valid: usize,
    price_sum: f64,
    price_sum_sq: f64,
    dev_window: Vec<f64>,
    dev_head: usize,
    dev_count: usize,
    dev_valid: usize,
    dev_sum: f64,
    f28: f64,
    f30: f64,
    f38: f64,
    f40: f64,
    f48: f64,
    f50: f64,
    f58: f64,
    f60: f64,
    f68: f64,
    f70: f64,
    f78: f64,
    f80: f64,
}

impl VrarsxState {
    #[inline]
    fn new(period: usize) -> Self {
        Self {
            prev_src_out: f64::NAN,
            prev_line: f64::NAN,
            price_window: vec![f64::NAN; period],
            price_head: 0,
            price_count: 0,
            price_valid: 0,
            price_sum: 0.0,
            price_sum_sq: 0.0,
            dev_window: vec![f64::NAN; period],
            dev_head: 0,
            dev_count: 0,
            dev_valid: 0,
            dev_sum: 0.0,
            f28: f64::NAN,
            f30: f64::NAN,
            f38: f64::NAN,
            f40: f64::NAN,
            f48: f64::NAN,
            f50: f64::NAN,
            f58: f64::NAN,
            f60: f64::NAN,
            f68: f64::NAN,
            f70: f64::NAN,
            f78: f64::NAN,
            f80: f64::NAN,
        }
    }

    #[inline(always)]
    fn update(&mut self, value: f64, period: usize, speed: f64) -> (f64, f64) {
        let src_out = if is_valid_source(value) {
            100.0 * value
        } else {
            f64::NAN
        };

        push_window_sum_sumsq(
            &mut self.price_window,
            &mut self.price_head,
            &mut self.price_count,
            &mut self.price_valid,
            &mut self.price_sum,
            &mut self.price_sum_sq,
            value,
        );

        let dev = if self.price_count == period && self.price_valid == period {
            biased_std_from_sums(self.price_sum, self.price_sum_sq, period)
        } else {
            f64::NAN
        };

        push_window_sum(
            &mut self.dev_window,
            &mut self.dev_head,
            &mut self.dev_count,
            &mut self.dev_valid,
            &mut self.dev_sum,
            dev,
        );

        let devavg = if self.dev_count == period && self.dev_valid == period {
            self.dev_sum / period as f64
        } else {
            f64::NAN
        };

        let vol_ratio = if dev.is_finite() && devavg.is_finite() && devavg != 0.0 {
            dev / devavg
        } else {
            f64::NAN
        };

        let adaptive_len = if vol_ratio.is_finite() && vol_ratio > 0.0 {
            ((period as f64) / vol_ratio).trunc()
        } else {
            f64::NAN
        };

        let kg = if adaptive_len.is_finite() {
            3.0 / (adaptive_len + 2.0)
        } else {
            f64::NAN
        };
        let hg = if kg.is_finite() { 1.0 - kg } else { f64::NAN };

        let mom0 = if src_out.is_finite() && self.prev_src_out.is_finite() {
            src_out - self.prev_src_out
        } else {
            f64::NAN
        };
        let moa0 = if mom0.is_finite() {
            mom0.abs()
        } else {
            f64::NAN
        };
        let spdp1 = speed + 1.0;

        let f28 = if kg.is_finite() && hg.is_finite() && mom0.is_finite() {
            kg * mom0 + hg * nz(self.f28)
        } else {
            f64::NAN
        };
        let f30 = if kg.is_finite() && hg.is_finite() && f28.is_finite() {
            hg * nz(self.f30) + kg * f28
        } else {
            f64::NAN
        };
        let mom1 = if f28.is_finite() && f30.is_finite() {
            f28 * spdp1 - f30 * speed
        } else {
            f64::NAN
        };

        let f38 = if kg.is_finite() && hg.is_finite() && mom1.is_finite() {
            hg * nz(self.f38) + kg * mom1
        } else {
            f64::NAN
        };
        let f40 = if kg.is_finite() && hg.is_finite() && f38.is_finite() {
            kg * f38 + hg * nz(self.f40)
        } else {
            f64::NAN
        };
        let mom2 = if f38.is_finite() && f40.is_finite() {
            f38 * spdp1 - f40 * speed
        } else {
            f64::NAN
        };

        let f48 = if kg.is_finite() && hg.is_finite() && mom2.is_finite() {
            hg * nz(self.f48) + kg * mom2
        } else {
            f64::NAN
        };
        let f50 = if kg.is_finite() && hg.is_finite() && f48.is_finite() {
            kg * f48 + hg * nz(self.f50)
        } else {
            f64::NAN
        };
        let mom_out = if f48.is_finite() && f50.is_finite() {
            f48 * spdp1 - f50 * speed
        } else {
            f64::NAN
        };

        let f58 = if kg.is_finite() && hg.is_finite() && moa0.is_finite() {
            hg * nz(self.f58) + kg * moa0
        } else {
            f64::NAN
        };
        let f60 = if kg.is_finite() && hg.is_finite() && f58.is_finite() {
            kg * f58 + hg * nz(self.f60)
        } else {
            f64::NAN
        };
        let moa1 = if f58.is_finite() && f60.is_finite() {
            f58 * spdp1 - f60 * speed
        } else {
            f64::NAN
        };

        let f68 = if kg.is_finite() && hg.is_finite() && moa1.is_finite() {
            hg * nz(self.f68) + kg * moa1
        } else {
            f64::NAN
        };
        let f70 = if kg.is_finite() && hg.is_finite() && f68.is_finite() {
            kg * f68 + hg * nz(self.f70)
        } else {
            f64::NAN
        };
        let moa2 = if f68.is_finite() && f70.is_finite() {
            f68 * spdp1 - f70 * speed
        } else {
            f64::NAN
        };

        let f78 = if kg.is_finite() && hg.is_finite() && moa2.is_finite() {
            hg * nz(self.f78) + kg * moa2
        } else {
            f64::NAN
        };
        let f80 = if kg.is_finite() && hg.is_finite() && f78.is_finite() {
            kg * f78 + hg * nz(self.f80)
        } else {
            f64::NAN
        };
        let moa_out = if f78.is_finite() && f80.is_finite() {
            f78 * spdp1 - f80 * speed
        } else {
            f64::NAN
        };

        let line = if mom_out.is_finite() && moa_out.is_finite() && moa_out != 0.0 {
            ((mom_out / moa_out + 1.0) * 50.0).clamp(0.0, 100.0)
        } else {
            f64::NAN
        };
        let signal = self.prev_line;

        self.prev_src_out = src_out;
        self.prev_line = line;
        self.f28 = f28;
        self.f30 = f30;
        self.f38 = f38;
        self.f40 = f40;
        self.f48 = f48;
        self.f50 = f50;
        self.f58 = f58;
        self.f60 = f60;
        self.f68 = f68;
        self.f70 = f70;
        self.f78 = f78;
        self.f80 = f80;

        (line, signal)
    }
}

#[inline(always)]
fn vrarsx_compute_into(
    data: &[f64],
    period: usize,
    speed: f64,
    line: &mut [f64],
    signal: &mut [f64],
) {
    let mut state = VrarsxState::new(period);
    for (i, &value) in data.iter().enumerate() {
        let (line_i, signal_i) = state.update(value, period, speed);
        line[i] = line_i;
        signal[i] = signal_i;
    }
}

#[inline(always)]
fn vrarsx_prepare<'a>(
    input: &'a VolatilityRatioAdaptiveRsxInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, f64, usize, Kernel), VolatilityRatioAdaptiveRsxError> {
    let data = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(VolatilityRatioAdaptiveRsxError::EmptyInputData);
    }

    let first = first_valid_source(data).ok_or(VolatilityRatioAdaptiveRsxError::AllValuesNaN)?;
    let period = input.get_period();
    let speed = input.get_speed();

    if period == 0 || period > len {
        return Err(VolatilityRatioAdaptiveRsxError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if !speed.is_finite() || !(0.0..=1.0).contains(&speed) {
        return Err(VolatilityRatioAdaptiveRsxError::InvalidSpeed { speed });
    }

    let needed = (2 * period).saturating_sub(1);
    let valid = len - first;
    if valid < needed {
        return Err(VolatilityRatioAdaptiveRsxError::NotEnoughValidData { needed, valid });
    }

    let chosen = kernel.to_non_batch();
    Ok((data, period, speed, first, chosen))
}

#[inline]
pub fn volatility_ratio_adaptive_rsx(
    input: &VolatilityRatioAdaptiveRsxInput,
) -> Result<VolatilityRatioAdaptiveRsxOutput, VolatilityRatioAdaptiveRsxError> {
    volatility_ratio_adaptive_rsx_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn volatility_ratio_adaptive_rsx_with_kernel(
    input: &VolatilityRatioAdaptiveRsxInput,
    kernel: Kernel,
) -> Result<VolatilityRatioAdaptiveRsxOutput, VolatilityRatioAdaptiveRsxError> {
    let (data, period, speed, first, _chosen) = vrarsx_prepare(input, kernel)?;
    let _ = (line_warmup(period, first), signal_warmup(period, first));
    let mut line = alloc_uninit_f64(data.len());
    let mut signal = alloc_uninit_f64(data.len());
    vrarsx_compute_into(data, period, speed, &mut line, &mut signal);
    Ok(VolatilityRatioAdaptiveRsxOutput { line, signal })
}

#[inline]
pub fn volatility_ratio_adaptive_rsx_into_slice(
    dst_line: &mut [f64],
    dst_signal: &mut [f64],
    input: &VolatilityRatioAdaptiveRsxInput,
    kernel: Kernel,
) -> Result<(), VolatilityRatioAdaptiveRsxError> {
    let (data, period, speed, _first, _chosen) = vrarsx_prepare(input, kernel)?;
    if dst_line.len() != data.len() || dst_signal.len() != data.len() {
        return Err(VolatilityRatioAdaptiveRsxError::OutputLengthMismatch {
            expected: data.len(),
            got: dst_line.len().max(dst_signal.len()),
        });
    }
    vrarsx_compute_into(data, period, speed, dst_line, dst_signal);
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn volatility_ratio_adaptive_rsx_into(
    input: &VolatilityRatioAdaptiveRsxInput,
    out_line: &mut [f64],
    out_signal: &mut [f64],
) -> Result<(), VolatilityRatioAdaptiveRsxError> {
    volatility_ratio_adaptive_rsx_into_slice(out_line, out_signal, input, Kernel::Auto)
}

#[derive(Clone, Debug)]
pub struct VolatilityRatioAdaptiveRsxStream {
    period: usize,
    speed: f64,
    state: VrarsxState,
}

impl VolatilityRatioAdaptiveRsxStream {
    pub fn try_new(
        params: VolatilityRatioAdaptiveRsxParams,
    ) -> Result<Self, VolatilityRatioAdaptiveRsxError> {
        let period = params.period.unwrap_or(14);
        let speed = params.speed.unwrap_or(0.5);
        if period == 0 {
            return Err(VolatilityRatioAdaptiveRsxError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        if !speed.is_finite() || !(0.0..=1.0).contains(&speed) {
            return Err(VolatilityRatioAdaptiveRsxError::InvalidSpeed { speed });
        }
        Ok(Self {
            period,
            speed,
            state: VrarsxState::new(period),
        })
    }

    #[inline]
    pub fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        let out = self.state.update(value, self.period, self.speed);
        if out.0.is_finite() || out.1.is_finite() {
            Some(out)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub struct VolatilityRatioAdaptiveRsxBatchRange {
    pub period: (usize, usize, usize),
    pub speed: (f64, f64, f64),
}

impl Default for VolatilityRatioAdaptiveRsxBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 56, 1),
            speed: (0.5, 0.5, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct VolatilityRatioAdaptiveRsxBatchBuilder {
    range: VolatilityRatioAdaptiveRsxBatchRange,
    kernel: Kernel,
}

impl VolatilityRatioAdaptiveRsxBatchBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline]
    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.period = (start, end, step);
        self
    }

    #[inline]
    pub fn period_static(mut self, value: usize) -> Self {
        self.range.period = (value, value, 0);
        self
    }

    #[inline]
    pub fn speed_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.speed = (start, end, step);
        self
    }

    #[inline]
    pub fn speed_static(mut self, value: f64) -> Self {
        self.range.speed = (value, value, 0.0);
        self
    }

    #[inline]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<VolatilityRatioAdaptiveRsxBatchOutput, VolatilityRatioAdaptiveRsxError> {
        volatility_ratio_adaptive_rsx_batch_with_kernel(data, &self.range, self.kernel)
    }

    #[inline]
    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<VolatilityRatioAdaptiveRsxBatchOutput, VolatilityRatioAdaptiveRsxError> {
        volatility_ratio_adaptive_rsx_batch_with_kernel(
            source_type(candles, source),
            &self.range,
            self.kernel,
        )
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct VolatilityRatioAdaptiveRsxBatchConfig {
    pub period_range: Vec<usize>,
    pub speed_range: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct VolatilityRatioAdaptiveRsxBatchOutput {
    pub line: Vec<f64>,
    pub signal: Vec<f64>,
    pub combos: Vec<VolatilityRatioAdaptiveRsxParams>,
    pub rows: usize,
    pub cols: usize,
}

impl VolatilityRatioAdaptiveRsxBatchOutput {
    #[inline]
    pub fn row_for_params(&self, params: &VolatilityRatioAdaptiveRsxParams) -> Option<usize> {
        self.combos.iter().position(|combo| {
            combo.period.unwrap_or(14) == params.period.unwrap_or(14)
                && combo.speed.unwrap_or(0.5).to_bits() == params.speed.unwrap_or(0.5).to_bits()
        })
    }

    #[inline]
    pub fn line_for(&self, params: &VolatilityRatioAdaptiveRsxParams) -> Option<&[f64]> {
        self.row_for_params(params).and_then(|row| {
            let start = row * self.cols;
            self.line.get(start..start + self.cols)
        })
    }

    #[inline]
    pub fn signal_for(&self, params: &VolatilityRatioAdaptiveRsxParams) -> Option<&[f64]> {
        self.row_for_params(params).and_then(|row| {
            let start = row * self.cols;
            self.signal.get(start..start + self.cols)
        })
    }
}

#[inline]
pub fn expand_grid_volatility_ratio_adaptive_rsx(
    range: &VolatilityRatioAdaptiveRsxBatchRange,
) -> Result<Vec<VolatilityRatioAdaptiveRsxParams>, VolatilityRatioAdaptiveRsxError> {
    fn axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, VolatilityRatioAdaptiveRsxError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start <= end {
            let mut out = Vec::new();
            let mut x = start;
            while x <= end {
                out.push(x);
                match x.checked_add(step.max(1)) {
                    Some(next) if next > x => x = next,
                    _ => break,
                }
            }
            if out.is_empty() {
                return Err(VolatilityRatioAdaptiveRsxError::InvalidRange {
                    start: start.to_string(),
                    end: end.to_string(),
                    step: step.to_string(),
                });
            }
            Ok(out)
        } else {
            let mut out = Vec::new();
            let mut x = start;
            while x >= end {
                out.push(x);
                if x == end {
                    break;
                }
                let next = x.saturating_sub(step.max(1));
                if next == x || next < end {
                    break;
                }
                x = next;
            }
            if out.is_empty() {
                return Err(VolatilityRatioAdaptiveRsxError::InvalidRange {
                    start: start.to_string(),
                    end: end.to_string(),
                    step: step.to_string(),
                });
            }
            Ok(out)
        }
    }

    fn axis_f64(
        (start, end, step): (f64, f64, f64),
    ) -> Result<Vec<f64>, VolatilityRatioAdaptiveRsxError> {
        if !start.is_finite() || !end.is_finite() || !step.is_finite() {
            return Err(VolatilityRatioAdaptiveRsxError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return Ok(vec![start]);
        }
        let mut out = Vec::new();
        if start < end {
            let st = step.abs();
            let mut x = start;
            while x <= end + 1e-12 {
                out.push(x);
                x += st;
            }
        } else {
            let st = -step.abs();
            let mut x = start;
            while x >= end - 1e-12 {
                out.push(x);
                x += st;
            }
        }
        if out.is_empty() {
            return Err(VolatilityRatioAdaptiveRsxError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        Ok(out)
    }

    let periods = axis_usize(range.period)?;
    let speeds = axis_f64(range.speed)?;
    let mut combos = Vec::with_capacity(periods.len().saturating_mul(speeds.len()));
    for period in periods {
        for speed in speeds.iter().copied() {
            combos.push(VolatilityRatioAdaptiveRsxParams {
                period: Some(period),
                speed: Some(speed),
            });
        }
    }
    Ok(combos)
}

#[inline]
pub fn volatility_ratio_adaptive_rsx_batch_with_kernel(
    data: &[f64],
    sweep: &VolatilityRatioAdaptiveRsxBatchRange,
    kernel: Kernel,
) -> Result<VolatilityRatioAdaptiveRsxBatchOutput, VolatilityRatioAdaptiveRsxError> {
    let batch = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => {
            return Err(VolatilityRatioAdaptiveRsxError::InvalidKernelForBatch(
                other,
            ))
        }
    };
    volatility_ratio_adaptive_rsx_batch_par_slice(data, sweep, batch.to_non_batch())
}

#[inline(always)]
pub fn volatility_ratio_adaptive_rsx_batch_slice(
    data: &[f64],
    sweep: &VolatilityRatioAdaptiveRsxBatchRange,
    kernel: Kernel,
) -> Result<VolatilityRatioAdaptiveRsxBatchOutput, VolatilityRatioAdaptiveRsxError> {
    volatility_ratio_adaptive_rsx_batch_inner(data, sweep, kernel, false)
}

#[inline(always)]
pub fn volatility_ratio_adaptive_rsx_batch_par_slice(
    data: &[f64],
    sweep: &VolatilityRatioAdaptiveRsxBatchRange,
    kernel: Kernel,
) -> Result<VolatilityRatioAdaptiveRsxBatchOutput, VolatilityRatioAdaptiveRsxError> {
    volatility_ratio_adaptive_rsx_batch_inner(data, sweep, kernel, true)
}

#[inline(always)]
fn volatility_ratio_adaptive_rsx_batch_inner(
    data: &[f64],
    sweep: &VolatilityRatioAdaptiveRsxBatchRange,
    _kernel: Kernel,
    parallel: bool,
) -> Result<VolatilityRatioAdaptiveRsxBatchOutput, VolatilityRatioAdaptiveRsxError> {
    let combos = expand_grid_volatility_ratio_adaptive_rsx(sweep)?;
    if data.is_empty() {
        return Err(VolatilityRatioAdaptiveRsxError::EmptyInputData);
    }
    let first = first_valid_source(data).ok_or(VolatilityRatioAdaptiveRsxError::AllValuesNaN)?;
    let max_needed = combos
        .iter()
        .map(|p| (2 * p.period.unwrap_or(14)).saturating_sub(1))
        .max()
        .unwrap_or(0);
    if data.len() - first < max_needed {
        return Err(VolatilityRatioAdaptiveRsxError::NotEnoughValidData {
            needed: max_needed,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let mut line_mu = make_uninit_matrix(rows, cols);
    let mut signal_mu = make_uninit_matrix(rows, cols);
    let line_warmups: Vec<usize> = combos
        .iter()
        .map(|p| line_warmup(p.period.unwrap_or(14), first))
        .collect();
    let signal_warmups: Vec<usize> = combos
        .iter()
        .map(|p| signal_warmup(p.period.unwrap_or(14), first))
        .collect();
    init_matrix_prefixes(&mut line_mu, cols, &line_warmups);
    init_matrix_prefixes(&mut signal_mu, cols, &signal_warmups);

    let mut line_guard = ManuallyDrop::new(line_mu);
    let mut signal_guard = ManuallyDrop::new(signal_mu);
    let line = unsafe {
        core::slice::from_raw_parts_mut(line_guard.as_mut_ptr() as *mut f64, line_guard.len())
    };
    let signal = unsafe {
        core::slice::from_raw_parts_mut(signal_guard.as_mut_ptr() as *mut f64, signal_guard.len())
    };

    volatility_ratio_adaptive_rsx_batch_inner_into(
        data,
        sweep,
        Kernel::Scalar,
        parallel,
        line,
        signal,
    )?;

    let line_values = unsafe {
        Vec::from_raw_parts(
            line_guard.as_mut_ptr() as *mut f64,
            line_guard.len(),
            line_guard.capacity(),
        )
    };
    let signal_values = unsafe {
        Vec::from_raw_parts(
            signal_guard.as_mut_ptr() as *mut f64,
            signal_guard.len(),
            signal_guard.capacity(),
        )
    };

    Ok(VolatilityRatioAdaptiveRsxBatchOutput {
        line: line_values,
        signal: signal_values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn volatility_ratio_adaptive_rsx_batch_inner_into(
    data: &[f64],
    sweep: &VolatilityRatioAdaptiveRsxBatchRange,
    _kernel: Kernel,
    parallel: bool,
    out_line: &mut [f64],
    out_signal: &mut [f64],
) -> Result<Vec<VolatilityRatioAdaptiveRsxParams>, VolatilityRatioAdaptiveRsxError> {
    let combos = expand_grid_volatility_ratio_adaptive_rsx(sweep)?;
    if data.is_empty() {
        return Err(VolatilityRatioAdaptiveRsxError::EmptyInputData);
    }
    let first = first_valid_source(data).ok_or(VolatilityRatioAdaptiveRsxError::AllValuesNaN)?;
    let max_needed = combos
        .iter()
        .map(|p| (2 * p.period.unwrap_or(14)).saturating_sub(1))
        .max()
        .unwrap_or(0);
    if data.len() - first < max_needed {
        return Err(VolatilityRatioAdaptiveRsxError::NotEnoughValidData {
            needed: max_needed,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or(VolatilityRatioAdaptiveRsxError::InvalidInput(
            "rows*cols overflow",
        ))?;
    if out_line.len() != total || out_signal.len() != total {
        return Err(VolatilityRatioAdaptiveRsxError::OutputLengthMismatch {
            expected: total,
            got: out_line.len().max(out_signal.len()),
        });
    }

    unsafe {
        let out_line_mu =
            std::slice::from_raw_parts_mut(out_line.as_mut_ptr() as *mut MaybeUninit<f64>, total);
        let out_signal_mu =
            std::slice::from_raw_parts_mut(out_signal.as_mut_ptr() as *mut MaybeUninit<f64>, total);
        let line_warmups: Vec<usize> = combos
            .iter()
            .map(|p| line_warmup(p.period.unwrap_or(14), first))
            .collect();
        let signal_warmups: Vec<usize> = combos
            .iter()
            .map(|p| signal_warmup(p.period.unwrap_or(14), first))
            .collect();
        init_matrix_prefixes(out_line_mu, cols, &line_warmups);
        init_matrix_prefixes(out_signal_mu, cols, &signal_warmups);
    }

    let do_row = |row: usize, line_row: &mut [f64], signal_row: &mut [f64]| {
        let params = &combos[row];
        vrarsx_compute_into(
            data,
            params.period.unwrap_or(14),
            params.speed.unwrap_or(0.5),
            line_row,
            signal_row,
        );
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_line
                .par_chunks_mut(cols)
                .zip(out_signal.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, (line_row, signal_row))| do_row(row, line_row, signal_row));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, (line_row, signal_row)) in out_line
                .chunks_mut(cols)
                .zip(out_signal.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, line_row, signal_row);
            }
        }
    } else {
        for (row, (line_row, signal_row)) in out_line
            .chunks_mut(cols)
            .zip(out_signal.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, line_row, signal_row);
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "volatility_ratio_adaptive_rsx")]
#[pyo3(signature = (data, period=14, speed=0.5, kernel=None))]
pub fn volatility_ratio_adaptive_rsx_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period: usize,
    speed: f64,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let data = data.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = VolatilityRatioAdaptiveRsxInput::from_slice(
        data,
        VolatilityRatioAdaptiveRsxParams {
            period: Some(period),
            speed: Some(speed),
        },
    );
    let output = py
        .allow_threads(|| volatility_ratio_adaptive_rsx_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((output.line.into_pyarray(py), output.signal.into_pyarray(py)))
}

#[cfg(feature = "python")]
#[pyclass(name = "VolatilityRatioAdaptiveRsxStream")]
pub struct VolatilityRatioAdaptiveRsxStreamPy {
    stream: VolatilityRatioAdaptiveRsxStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl VolatilityRatioAdaptiveRsxStreamPy {
    #[new]
    #[pyo3(signature = (period=14, speed=0.5))]
    fn new(period: usize, speed: f64) -> PyResult<Self> {
        let stream = VolatilityRatioAdaptiveRsxStream::try_new(VolatilityRatioAdaptiveRsxParams {
            period: Some(period),
            speed: Some(speed),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "volatility_ratio_adaptive_rsx_batch")]
#[pyo3(signature = (data, period_range, speed_range, kernel=None))]
pub fn volatility_ratio_adaptive_rsx_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    speed_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let sweep = VolatilityRatioAdaptiveRsxBatchRange {
        period: period_range,
        speed: speed_range,
    };
    let combos = expand_grid_volatility_ratio_adaptive_rsx(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;
    let line_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let signal_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let line_out = unsafe { line_arr.as_slice_mut()? };
    let signal_out = unsafe { signal_arr.as_slice_mut()? };
    let kernel = validate_kernel(kernel, true)?;

    py.allow_threads(|| {
        let batch_kernel = match kernel {
            Kernel::Auto => detect_best_batch_kernel(),
            other => other,
        };
        volatility_ratio_adaptive_rsx_batch_inner_into(
            data,
            &sweep,
            batch_kernel.to_non_batch(),
            true,
            line_out,
            signal_out,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("line", line_arr.reshape((rows, cols))?)?;
    dict.set_item("signal", signal_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "periods",
        combos
            .iter()
            .map(|p| p.period.unwrap_or(14) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "speeds",
        combos
            .iter()
            .map(|p| p.speed.unwrap_or(0.5))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_volatility_ratio_adaptive_rsx_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(volatility_ratio_adaptive_rsx_py, m)?)?;
    m.add_function(wrap_pyfunction!(volatility_ratio_adaptive_rsx_batch_py, m)?)?;
    m.add_class::<VolatilityRatioAdaptiveRsxStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "volatility_ratio_adaptive_rsx_js")]
pub fn volatility_ratio_adaptive_rsx_js(
    data: &[f64],
    period: usize,
    speed: f64,
) -> Result<JsValue, JsValue> {
    let input = VolatilityRatioAdaptiveRsxInput::from_slice(
        data,
        VolatilityRatioAdaptiveRsxParams {
            period: Some(period),
            speed: Some(speed),
        },
    );
    let mut line = vec![0.0; data.len()];
    let mut signal = vec![0.0; data.len()];
    volatility_ratio_adaptive_rsx_into_slice(&mut line, &mut signal, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("line"),
        &serde_wasm_bindgen::to_value(&line).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("signal"),
        &serde_wasm_bindgen::to_value(&signal).unwrap(),
    )?;
    Ok(obj.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "volatility_ratio_adaptive_rsx_batch_js")]
pub fn volatility_ratio_adaptive_rsx_batch_js(
    data: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: VolatilityRatioAdaptiveRsxBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.period_range.len() != 3 || config.speed_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: ranges must have exactly 3 elements [start, end, step]",
        ));
    }

    let sweep = VolatilityRatioAdaptiveRsxBatchRange {
        period: (
            config.period_range[0],
            config.period_range[1],
            config.period_range[2],
        ),
        speed: (
            config.speed_range[0],
            config.speed_range[1],
            config.speed_range[2],
        ),
    };
    let combos = expand_grid_volatility_ratio_adaptive_rsx(&sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
    let mut line = vec![0.0; total];
    let mut signal = vec![0.0; total];
    volatility_ratio_adaptive_rsx_batch_inner_into(
        data,
        &sweep,
        Kernel::Scalar,
        false,
        &mut line,
        &mut signal,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("line"),
        &serde_wasm_bindgen::to_value(&line).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("signal"),
        &serde_wasm_bindgen::to_value(&signal).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("rows"),
        &JsValue::from_f64(rows as f64),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("cols"),
        &JsValue::from_f64(cols as f64),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("combos"),
        &serde_wasm_bindgen::to_value(&combos).unwrap(),
    )?;
    Ok(obj.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn volatility_ratio_adaptive_rsx_alloc(len: usize) -> *mut f64 {
    let mut v = Vec::<f64>::with_capacity(2 * len);
    let ptr = v.as_mut_ptr();
    std::mem::forget(v);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn volatility_ratio_adaptive_rsx_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, 0, 2 * len);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn volatility_ratio_adaptive_rsx_into(
    data_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
    speed: f64,
) -> Result<(), JsValue> {
    if data_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to volatility_ratio_adaptive_rsx_into",
        ));
    }

    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, 2 * len);
        let (line, signal) = out.split_at_mut(len);
        let input = VolatilityRatioAdaptiveRsxInput::from_slice(
            data,
            VolatilityRatioAdaptiveRsxParams {
                period: Some(period),
                speed: Some(speed),
            },
        );
        volatility_ratio_adaptive_rsx_into_slice(line, signal, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "volatility_ratio_adaptive_rsx_into_host")]
pub fn volatility_ratio_adaptive_rsx_into_host(
    data: &[f64],
    out_ptr: *mut f64,
    period: usize,
    speed: f64,
) -> Result<(), JsValue> {
    if out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to volatility_ratio_adaptive_rsx_into_host",
        ));
    }
    unsafe {
        let out = std::slice::from_raw_parts_mut(out_ptr, 2 * data.len());
        let (line, signal) = out.split_at_mut(data.len());
        let input = VolatilityRatioAdaptiveRsxInput::from_slice(
            data,
            VolatilityRatioAdaptiveRsxParams {
                period: Some(period),
                speed: Some(speed),
            },
        );
        volatility_ratio_adaptive_rsx_into_slice(line, signal, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn volatility_ratio_adaptive_rsx_batch_into(
    data_ptr: *const f64,
    line_ptr: *mut f64,
    signal_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
    speed_start: f64,
    speed_end: f64,
    speed_step: f64,
) -> Result<usize, JsValue> {
    if data_ptr.is_null() || line_ptr.is_null() || signal_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to volatility_ratio_adaptive_rsx_batch_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let sweep = VolatilityRatioAdaptiveRsxBatchRange {
            period: (period_start, period_end, period_step),
            speed: (speed_start, speed_end, speed_step),
        };
        let combos = expand_grid_volatility_ratio_adaptive_rsx(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let total = rows
            .checked_mul(len)
            .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
        let line = std::slice::from_raw_parts_mut(line_ptr, total);
        let signal = std::slice::from_raw_parts_mut(signal_ptr, total);
        volatility_ratio_adaptive_rsx_batch_inner_into(
            data,
            &sweep,
            Kernel::Scalar,
            false,
            line,
            signal,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn volatility_ratio_adaptive_rsx_output_into_js(
    data: &[f64],
    period: usize,
    speed: f64,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = volatility_ratio_adaptive_rsx_js(data, period, speed)?;
    crate::write_wasm_object_f64_outputs(
        "volatility_ratio_adaptive_rsx_output_into_js",
        &value,
        out,
    )
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn volatility_ratio_adaptive_rsx_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = volatility_ratio_adaptive_rsx_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "volatility_ratio_adaptive_rsx_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn series_close(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.len() == b.len()
            && a.iter().zip(b.iter()).all(|(&x, &y)| {
                (x.is_nan() && y.is_nan())
                    || (x.is_finite() && y.is_finite() && (x - y).abs() <= tol)
            })
    }

    fn sample_data() -> Vec<f64> {
        vec![
            100.0, 100.8, 101.6, 101.1, 102.4, 103.0, 102.2, 103.6, 104.1, 103.4, 104.9, 105.7,
            105.2, 106.8, 107.4, 106.6, 108.0, 108.9, 108.1, 109.7, 110.3, 109.8, 111.1, 112.0,
            111.5, 113.2, 113.8, 113.0, 114.7, 115.1, 114.4, 116.0, 116.8, 116.1, 117.5, 118.2,
            117.7, 119.1, 119.9, 119.0, 120.6, 121.3, 120.7, 122.2, 122.8, 122.0, 123.6, 124.4,
        ]
    }

    fn naive_vrarsx(data: &[f64], period: usize, speed: f64) -> (Vec<f64>, Vec<f64>) {
        let n = data.len();
        let mut dev = vec![f64::NAN; n];
        for i in 0..n {
            if i + 1 < period {
                continue;
            }
            let start = i + 1 - period;
            let window = &data[start..=i];
            if window.iter().all(|v| v.is_finite()) {
                let sum: f64 = window.iter().sum();
                let sum_sq: f64 = window.iter().map(|v| v * v).sum();
                dev[i] = biased_std_from_sums(sum, sum_sq, period);
            }
        }

        let mut devavg = vec![f64::NAN; n];
        for i in 0..n {
            if i + 1 < period {
                continue;
            }
            let start = i + 1 - period;
            let window = &dev[start..=i];
            if window.iter().all(|v| v.is_finite()) {
                devavg[i] = window.iter().sum::<f64>() / period as f64;
            }
        }

        let mut line = vec![f64::NAN; n];
        let mut signal = vec![f64::NAN; n];
        let mut prev_src_out = f64::NAN;
        let mut prev_line = f64::NAN;
        let mut f28 = f64::NAN;
        let mut f30 = f64::NAN;
        let mut f38 = f64::NAN;
        let mut f40 = f64::NAN;
        let mut f48 = f64::NAN;
        let mut f50 = f64::NAN;
        let mut f58 = f64::NAN;
        let mut f60 = f64::NAN;
        let mut f68 = f64::NAN;
        let mut f70 = f64::NAN;
        let mut f78 = f64::NAN;
        let mut f80 = f64::NAN;

        for i in 0..n {
            let src_out = if data[i].is_finite() {
                100.0 * data[i]
            } else {
                f64::NAN
            };
            let ratio = if dev[i].is_finite() && devavg[i].is_finite() && devavg[i] != 0.0 {
                dev[i] / devavg[i]
            } else {
                f64::NAN
            };
            let adaptive_len = if ratio.is_finite() && ratio > 0.0 {
                ((period as f64) / ratio).trunc()
            } else {
                f64::NAN
            };
            let kg = if adaptive_len.is_finite() {
                3.0 / (adaptive_len + 2.0)
            } else {
                f64::NAN
            };
            let hg = if kg.is_finite() { 1.0 - kg } else { f64::NAN };
            let mom0 = if src_out.is_finite() && prev_src_out.is_finite() {
                src_out - prev_src_out
            } else {
                f64::NAN
            };
            let moa0 = if mom0.is_finite() {
                mom0.abs()
            } else {
                f64::NAN
            };
            let spdp1 = speed + 1.0;

            let nf28 = if kg.is_finite() && hg.is_finite() && mom0.is_finite() {
                kg * mom0 + hg * nz(f28)
            } else {
                f64::NAN
            };
            let nf30 = if kg.is_finite() && hg.is_finite() && nf28.is_finite() {
                hg * nz(f30) + kg * nf28
            } else {
                f64::NAN
            };
            let mom1 = if nf28.is_finite() && nf30.is_finite() {
                nf28 * spdp1 - nf30 * speed
            } else {
                f64::NAN
            };

            let nf38 = if kg.is_finite() && hg.is_finite() && mom1.is_finite() {
                hg * nz(f38) + kg * mom1
            } else {
                f64::NAN
            };
            let nf40 = if kg.is_finite() && hg.is_finite() && nf38.is_finite() {
                kg * nf38 + hg * nz(f40)
            } else {
                f64::NAN
            };
            let mom2 = if nf38.is_finite() && nf40.is_finite() {
                nf38 * spdp1 - nf40 * speed
            } else {
                f64::NAN
            };

            let nf48 = if kg.is_finite() && hg.is_finite() && mom2.is_finite() {
                hg * nz(f48) + kg * mom2
            } else {
                f64::NAN
            };
            let nf50 = if kg.is_finite() && hg.is_finite() && nf48.is_finite() {
                kg * nf48 + hg * nz(f50)
            } else {
                f64::NAN
            };
            let mom_out = if nf48.is_finite() && nf50.is_finite() {
                nf48 * spdp1 - nf50 * speed
            } else {
                f64::NAN
            };

            let nf58 = if kg.is_finite() && hg.is_finite() && moa0.is_finite() {
                hg * nz(f58) + kg * moa0
            } else {
                f64::NAN
            };
            let nf60 = if kg.is_finite() && hg.is_finite() && nf58.is_finite() {
                kg * nf58 + hg * nz(f60)
            } else {
                f64::NAN
            };
            let moa1 = if nf58.is_finite() && nf60.is_finite() {
                nf58 * spdp1 - nf60 * speed
            } else {
                f64::NAN
            };

            let nf68 = if kg.is_finite() && hg.is_finite() && moa1.is_finite() {
                hg * nz(f68) + kg * moa1
            } else {
                f64::NAN
            };
            let nf70 = if kg.is_finite() && hg.is_finite() && nf68.is_finite() {
                kg * nf68 + hg * nz(f70)
            } else {
                f64::NAN
            };
            let moa2 = if nf68.is_finite() && nf70.is_finite() {
                nf68 * spdp1 - nf70 * speed
            } else {
                f64::NAN
            };

            let nf78 = if kg.is_finite() && hg.is_finite() && moa2.is_finite() {
                hg * nz(f78) + kg * moa2
            } else {
                f64::NAN
            };
            let nf80 = if kg.is_finite() && hg.is_finite() && nf78.is_finite() {
                kg * nf78 + hg * nz(f80)
            } else {
                f64::NAN
            };
            let moa_out = if nf78.is_finite() && nf80.is_finite() {
                nf78 * spdp1 - nf80 * speed
            } else {
                f64::NAN
            };

            signal[i] = prev_line;
            line[i] = if mom_out.is_finite() && moa_out.is_finite() && moa_out != 0.0 {
                ((mom_out / moa_out + 1.0) * 50.0).clamp(0.0, 100.0)
            } else {
                f64::NAN
            };

            prev_line = line[i];
            prev_src_out = src_out;
            f28 = nf28;
            f30 = nf30;
            f38 = nf38;
            f40 = nf40;
            f48 = nf48;
            f50 = nf50;
            f58 = nf58;
            f60 = nf60;
            f68 = nf68;
            f70 = nf70;
            f78 = nf78;
            f80 = nf80;
        }

        (line, signal)
    }

    #[test]
    fn volatility_ratio_adaptive_rsx_matches_naive() {
        let data = sample_data();
        let input = VolatilityRatioAdaptiveRsxInput::from_slice(
            &data,
            VolatilityRatioAdaptiveRsxParams {
                period: Some(6),
                speed: Some(0.5),
            },
        );
        let out = volatility_ratio_adaptive_rsx(&input).expect("vrarsx output");
        let (exp_line, exp_signal) = naive_vrarsx(&data, 6, 0.5);
        assert!(series_close(&out.line, &exp_line, 1e-12));
        assert!(series_close(&out.signal, &exp_signal, 1e-12));
    }

    #[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
    #[test]
    fn volatility_ratio_adaptive_rsx_into_matches_api() -> Result<(), Box<dyn Error>> {
        let data = sample_data();
        let input = VolatilityRatioAdaptiveRsxInput::from_slice(
            &data,
            VolatilityRatioAdaptiveRsxParams {
                period: Some(6),
                speed: Some(0.5),
            },
        );
        let direct = volatility_ratio_adaptive_rsx(&input)?;
        let mut line = vec![0.0; data.len()];
        let mut signal = vec![0.0; data.len()];
        volatility_ratio_adaptive_rsx_into(&input, &mut line, &mut signal)?;
        assert!(series_close(&direct.line, &line, 1e-12));
        assert!(series_close(&direct.signal, &signal, 1e-12));
        Ok(())
    }

    #[test]
    fn volatility_ratio_adaptive_rsx_stream_matches_batch() {
        let data = sample_data();
        let params = VolatilityRatioAdaptiveRsxParams {
            period: Some(6),
            speed: Some(0.5),
        };
        let input = VolatilityRatioAdaptiveRsxInput::from_slice(&data, params.clone());
        let batch = volatility_ratio_adaptive_rsx(&input).expect("batch output");
        let mut stream = VolatilityRatioAdaptiveRsxStream::try_new(params).expect("stream");
        let mut line = Vec::with_capacity(data.len());
        let mut signal = Vec::with_capacity(data.len());
        for &value in &data {
            match stream.update(value) {
                Some((v0, v1)) => {
                    line.push(v0);
                    signal.push(v1);
                }
                None => {
                    line.push(f64::NAN);
                    signal.push(f64::NAN);
                }
            }
        }
        assert!(series_close(&batch.line, &line, 1e-12));
        assert!(series_close(&batch.signal, &signal, 1e-12));
    }

    #[test]
    fn volatility_ratio_adaptive_rsx_batch_single_param_matches_single() {
        let data = sample_data();
        let sweep = VolatilityRatioAdaptiveRsxBatchRange {
            period: (6, 6, 0),
            speed: (0.5, 0.5, 0.0),
        };
        let batch = volatility_ratio_adaptive_rsx_batch_with_kernel(&data, &sweep, Kernel::Auto)
            .expect("batch output");
        let input = VolatilityRatioAdaptiveRsxInput::from_slice(
            &data,
            VolatilityRatioAdaptiveRsxParams {
                period: Some(6),
                speed: Some(0.5),
            },
        );
        let single = volatility_ratio_adaptive_rsx(&input).expect("single output");
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, data.len());
        assert!(series_close(&batch.line, &single.line, 1e-12));
        assert!(series_close(&batch.signal, &single.signal, 1e-12));
    }

    #[test]
    fn volatility_ratio_adaptive_rsx_batch_sweep_matches_singles() {
        let data = sample_data();
        let sweep = VolatilityRatioAdaptiveRsxBatchRange {
            period: (5, 6, 1),
            speed: (0.4, 0.6, 0.2),
        };
        let batch = volatility_ratio_adaptive_rsx_batch_with_kernel(&data, &sweep, Kernel::Auto)
            .expect("batch output");
        assert_eq!(batch.rows, 4);
        for params in &batch.combos {
            let row = batch.row_for_params(params).expect("row");
            let start = row * batch.cols;
            let end = start + batch.cols;
            let input = VolatilityRatioAdaptiveRsxInput::from_slice(&data, params.clone());
            let single = volatility_ratio_adaptive_rsx(&input).expect("single");
            assert!(series_close(&batch.line[start..end], &single.line, 1e-12));
            assert!(series_close(
                &batch.signal[start..end],
                &single.signal,
                1e-12
            ));
        }
    }

    #[test]
    fn volatility_ratio_adaptive_rsx_rejects_invalid_speed() {
        let data = sample_data();
        let input = VolatilityRatioAdaptiveRsxInput::from_slice(
            &data,
            VolatilityRatioAdaptiveRsxParams {
                period: Some(6),
                speed: Some(1.5),
            },
        );
        let err = volatility_ratio_adaptive_rsx(&input).expect_err("invalid speed");
        assert!(matches!(
            err,
            VolatilityRatioAdaptiveRsxError::InvalidSpeed { .. }
        ));
    }
}
