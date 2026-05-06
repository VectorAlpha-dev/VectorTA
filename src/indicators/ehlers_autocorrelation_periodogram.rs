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
use wasm_bindgen::prelude::*;

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::f64::consts::{PI, SQRT_2};
use std::mem::ManuallyDrop;
use thiserror::Error;

const DEFAULT_MIN_PERIOD: usize = 8;
const DEFAULT_MAX_PERIOD: usize = 48;
const DEFAULT_AVG_LENGTH: usize = 3;
const DEFAULT_ENHANCE: bool = true;

impl<'a> AsRef<[f64]> for EhlersAutocorrelationPeriodogramInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            EhlersAutocorrelationPeriodogramData::Slice(slice) => slice,
            EhlersAutocorrelationPeriodogramData::Candles { candles, source } => {
                source_type(candles, source)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum EhlersAutocorrelationPeriodogramData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct EhlersAutocorrelationPeriodogramOutput {
    pub dominant_cycle: Vec<f64>,
    pub normalized_power: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct EhlersAutocorrelationPeriodogramParams {
    pub min_period: Option<usize>,
    pub max_period: Option<usize>,
    pub avg_length: Option<usize>,
    pub enhance: Option<bool>,
}

impl Default for EhlersAutocorrelationPeriodogramParams {
    fn default() -> Self {
        Self {
            min_period: Some(DEFAULT_MIN_PERIOD),
            max_period: Some(DEFAULT_MAX_PERIOD),
            avg_length: Some(DEFAULT_AVG_LENGTH),
            enhance: Some(DEFAULT_ENHANCE),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EhlersAutocorrelationPeriodogramInput<'a> {
    pub data: EhlersAutocorrelationPeriodogramData<'a>,
    pub params: EhlersAutocorrelationPeriodogramParams,
}

impl<'a> EhlersAutocorrelationPeriodogramInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: EhlersAutocorrelationPeriodogramParams,
    ) -> Self {
        Self {
            data: EhlersAutocorrelationPeriodogramData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: EhlersAutocorrelationPeriodogramParams) -> Self {
        Self {
            data: EhlersAutocorrelationPeriodogramData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(
            candles,
            "close",
            EhlersAutocorrelationPeriodogramParams::default(),
        )
    }
}

#[derive(Copy, Clone, Debug)]
pub struct EhlersAutocorrelationPeriodogramBuilder {
    min_period: Option<usize>,
    max_period: Option<usize>,
    avg_length: Option<usize>,
    enhance: Option<bool>,
    kernel: Kernel,
}

impl Default for EhlersAutocorrelationPeriodogramBuilder {
    fn default() -> Self {
        Self {
            min_period: None,
            max_period: None,
            avg_length: None,
            enhance: None,
            kernel: Kernel::Auto,
        }
    }
}

impl EhlersAutocorrelationPeriodogramBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn min_period(mut self, min_period: usize) -> Self {
        self.min_period = Some(min_period);
        self
    }

    #[inline]
    pub fn max_period(mut self, max_period: usize) -> Self {
        self.max_period = Some(max_period);
        self
    }

    #[inline]
    pub fn avg_length(mut self, avg_length: usize) -> Self {
        self.avg_length = Some(avg_length);
        self
    }

    #[inline]
    pub fn enhance(mut self, enhance: bool) -> Self {
        self.enhance = Some(enhance);
        self
    }

    #[inline]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline]
    pub fn apply(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<EhlersAutocorrelationPeriodogramOutput, EhlersAutocorrelationPeriodogramError> {
        let input = EhlersAutocorrelationPeriodogramInput::from_candles(
            candles,
            source,
            EhlersAutocorrelationPeriodogramParams {
                min_period: self.min_period,
                max_period: self.max_period,
                avg_length: self.avg_length,
                enhance: self.enhance,
            },
        );
        ehlers_autocorrelation_periodogram_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<EhlersAutocorrelationPeriodogramOutput, EhlersAutocorrelationPeriodogramError> {
        let input = EhlersAutocorrelationPeriodogramInput::from_slice(
            data,
            EhlersAutocorrelationPeriodogramParams {
                min_period: self.min_period,
                max_period: self.max_period,
                avg_length: self.avg_length,
                enhance: self.enhance,
            },
        );
        ehlers_autocorrelation_periodogram_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn into_stream(
        self,
    ) -> Result<EhlersAutocorrelationPeriodogramStream, EhlersAutocorrelationPeriodogramError> {
        EhlersAutocorrelationPeriodogramStream::try_new(EhlersAutocorrelationPeriodogramParams {
            min_period: self.min_period,
            max_period: self.max_period,
            avg_length: self.avg_length,
            enhance: self.enhance,
        })
    }
}

#[derive(Debug, Error)]
pub enum EhlersAutocorrelationPeriodogramError {
    #[error("ehlers_autocorrelation_periodogram: Input data slice is empty.")]
    EmptyInputData,
    #[error("ehlers_autocorrelation_periodogram: All values are NaN.")]
    AllValuesNaN,
    #[error("ehlers_autocorrelation_periodogram: Invalid min_period: {min_period}")]
    InvalidMinPeriod { min_period: usize },
    #[error("ehlers_autocorrelation_periodogram: Invalid max_period: max_period = {max_period}, data length = {data_len}")]
    InvalidMaxPeriod { max_period: usize, data_len: usize },
    #[error("ehlers_autocorrelation_periodogram: Invalid period order: min_period = {min_period}, max_period = {max_period}")]
    InvalidPeriodOrder {
        min_period: usize,
        max_period: usize,
    },
    #[error("ehlers_autocorrelation_periodogram: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("ehlers_autocorrelation_periodogram: Output length mismatch: expected = {expected}, dominant_cycle = {dominant_cycle_got}, normalized_power = {normalized_power_got}")]
    OutputLengthMismatch {
        expected: usize,
        dominant_cycle_got: usize,
        normalized_power_got: usize,
    },
    #[error(
        "ehlers_autocorrelation_periodogram: Invalid range: start={start}, end={end}, step={step}"
    )]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("ehlers_autocorrelation_periodogram: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ResolvedParams {
    min_period: usize,
    max_period: usize,
    avg_length: usize,
    enhance: bool,
}

#[derive(Debug, Clone)]
pub struct EhlersAutocorrelationPeriodogramStream {
    params: ResolvedParams,
    prev_price_1: f64,
    prev_price_2: f64,
    hp_prev_1: f64,
    hp_prev_2: f64,
    filt_prev_1: f64,
    filt_prev_2: f64,
    filt_history: Vec<f64>,
    corr: Vec<f64>,
    power: Vec<f64>,
    smooth: Vec<f64>,
    cos_table: Vec<f64>,
    sin_table: Vec<f64>,
    trig_stride: usize,
    hp_coef: f64,
    hp_prev1_coef: f64,
    hp_prev2_coef: f64,
    filt_c1: f64,
    filt_c2: f64,
    filt_c3: f64,
    decay: f64,
    dom: f64,
    max_pwr: f64,
    e: f64,
    warmup_bias: bool,
    bars_seen: usize,
}

impl EhlersAutocorrelationPeriodogramStream {
    pub fn try_new(
        params: EhlersAutocorrelationPeriodogramParams,
    ) -> Result<Self, EhlersAutocorrelationPeriodogramError> {
        let params = resolve_params(&params, 0)?;
        Ok(Self::new_resolved(params))
    }

    #[inline]
    fn new_resolved(params: ResolvedParams) -> Self {
        let size = params.max_period + 1;
        let alpha_hp = highpass_alpha(params.max_period);
        let one_minus = 1.0 - alpha_hp;
        let a1 = (-SQRT_2 * PI / params.min_period as f64).exp();
        let b1 = 2.0 * a1 * (SQRT_2 * PI / params.min_period as f64).cos();
        let c2 = b1;
        let c3 = -(a1 * a1);
        let diff = (params.max_period - params.min_period) as f64;
        let decay = if diff > 0.0 {
            10.0_f64.powf(-0.15 / diff)
        } else {
            1.0
        };
        let trig_stride = params.max_period + 1;
        let trig_len = trig_stride * trig_stride;
        let mut cos_table = vec![0.0; trig_len];
        let mut sin_table = vec![0.0; trig_len];
        for period in params.min_period..=params.max_period {
            let period_f = period as f64;
            let base = period * trig_stride;
            for n in 2..=params.max_period {
                let angle = 2.0 * PI * n as f64 / period_f;
                cos_table[base + n] = angle.cos();
                sin_table[base + n] = angle.sin();
            }
        }
        Self {
            params,
            prev_price_1: 0.0,
            prev_price_2: 0.0,
            hp_prev_1: 0.0,
            hp_prev_2: 0.0,
            filt_prev_1: 0.0,
            filt_prev_2: 0.0,
            filt_history: Vec::with_capacity(params.max_period + corr_window_for_params(params)),
            corr: vec![0.0; size],
            power: vec![0.0; size],
            smooth: vec![0.0; size],
            cos_table,
            sin_table,
            trig_stride,
            hp_coef: (1.0 - alpha_hp * 0.5).powi(2),
            hp_prev1_coef: 2.0 * one_minus,
            hp_prev2_coef: one_minus.powi(2),
            filt_c1: 1.0 - c2 - c3,
            filt_c2: c2,
            filt_c3: c3,
            decay,
            dom: (params.min_period + params.max_period) as f64 * 0.5,
            max_pwr: 0.0,
            e: 1.0,
            warmup_bias: true,
            bars_seen: 0,
        }
    }

    #[inline]
    fn reset(&mut self) {
        *self = Self::new_resolved(self.params);
    }

    #[inline]
    pub fn get_warmup_period(&self) -> usize {
        warmup_period(self.params)
    }

    #[inline(always)]
    fn filt_back(&self, back: usize) -> f64 {
        let len = self.filt_history.len();
        if back >= len {
            0.0
        } else {
            self.filt_history[len - 1 - back]
        }
    }

    pub fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        if !value.is_finite() {
            self.reset();
            return None;
        }

        let hp = self.hp_coef * (value - 2.0 * self.prev_price_1 + self.prev_price_2)
            + self.hp_prev1_coef * self.hp_prev_1
            - self.hp_prev2_coef * self.hp_prev_2;

        let filt = self.filt_c1 * (hp + self.hp_prev_1) * 0.5
            + self.filt_c2 * self.filt_prev_1
            + self.filt_c3 * self.filt_prev_2;

        self.prev_price_2 = self.prev_price_1;
        self.prev_price_1 = value;
        self.hp_prev_2 = self.hp_prev_1;
        self.hp_prev_1 = hp;
        self.filt_prev_2 = self.filt_prev_1;
        self.filt_prev_1 = filt;
        self.filt_history.push(filt);
        self.bars_seen += 1;

        self.corr[0] = 0.0;
        if self.params.max_period >= 1 {
            self.corr[1] = 0.0;
        }
        if self.params.avg_length == 3 {
            let x0 = self.filt_back(0);
            let x1 = self.filt_back(1);
            let x2 = self.filt_back(2);
            let sx = x0 + x1 + x2;
            let sxx = x0 * x0 + x1 * x1 + x2 * x2;
            for lag in 2..=self.params.max_period {
                let y0 = self.filt_back(lag);
                let y1 = self.filt_back(lag + 1);
                let y2 = self.filt_back(lag + 2);
                let sy = y0 + y1 + y2;
                let syy = y0 * y0 + y1 * y1 + y2 * y2;
                let sxy = x0 * y0 + x1 * y1 + x2 * y2;
                let denom_x = 3.0 * sxx - sx * sx;
                let denom_y = 3.0 * syy - sy * sy;
                let denom = denom_x * denom_y;
                self.corr[lag] = if denom > 0.0 {
                    (3.0 * sxy - sx * sy) / denom.sqrt()
                } else {
                    0.0
                };
            }
        } else {
            for lag in 2..=self.params.max_period {
                let window = corr_window(self.params.avg_length, lag);
                let mut sx = 0.0;
                let mut sy = 0.0;
                let mut sxx = 0.0;
                let mut syy = 0.0;
                let mut sxy = 0.0;
                for k in 0..window {
                    let x = self.filt_back(k);
                    let y = self.filt_back(lag + k);
                    sx += x;
                    sy += y;
                    sxx += x * x;
                    syy += y * y;
                    sxy += x * y;
                }
                let valid = window as f64;
                let denom_x = valid * sxx - sx * sx;
                let denom_y = valid * syy - sy * sy;
                let denom = denom_x * denom_y;
                self.corr[lag] = if denom > 0.0 {
                    (valid * sxy - sx * sy) / denom.sqrt()
                } else {
                    0.0
                };
            }
        }

        let mut local_max_pwr = 0.0;
        for period in self.params.min_period..=self.params.max_period {
            let mut cos_acc = 0.0;
            let mut sin_acc = 0.0;
            let trig_base = period * self.trig_stride;
            for n in 2..=self.params.max_period {
                let corr = self.corr[n];
                cos_acc += corr * self.cos_table[trig_base + n];
                sin_acc += corr * self.sin_table[trig_base + n];
            }
            let sq = cos_acc * cos_acc + sin_acc * sin_acc;
            let smooth = 0.2 * sq * sq + 0.8 * self.smooth[period];
            self.smooth[period] = smooth;
            if smooth > local_max_pwr {
                local_max_pwr = smooth;
            }
        }

        if local_max_pwr > self.max_pwr {
            self.max_pwr = local_max_pwr;
        } else {
            self.max_pwr *= self.decay;
        }

        let mut weighted = 0.0;
        let mut sum_weight = 0.0;
        for period in self.params.min_period..=self.params.max_period {
            let mut pwr = if self.max_pwr > 0.0 {
                self.smooth[period] / self.max_pwr
            } else {
                0.0
            };
            if self.params.enhance {
                pwr = pwr.powi(3);
            }
            self.power[period] = pwr;
            if pwr >= 0.5 {
                weighted += period as f64 * pwr;
                sum_weight += pwr;
            }
        }

        let base = if sum_weight >= 0.25 {
            weighted / sum_weight
        } else {
            self.dom
        };
        self.dom += 0.2 * (base - self.dom);
        if self.warmup_bias {
            self.e *= 0.8;
            let c = 1.0 / (1.0 - self.e);
            self.dom *= c;
            self.warmup_bias = self.e > 1e-10;
        }

        if self.bars_seen <= self.get_warmup_period() {
            return None;
        }

        let dom_idx = self
            .dom
            .round()
            .clamp(self.params.min_period as f64, self.params.max_period as f64)
            as usize;
        Some((self.dom, self.power[dom_idx]))
    }
}

#[inline]
fn highpass_alpha(max_period: usize) -> f64 {
    let angle = SQRT_2 * PI / max_period as f64;
    (angle.cos() + angle.sin() - 1.0) / angle.cos()
}

#[inline]
fn corr_window(avg_length: usize, lag: usize) -> usize {
    if avg_length == 0 {
        lag.max(2)
    } else {
        avg_length.max(2)
    }
}

#[inline]
fn corr_window_for_params(params: ResolvedParams) -> usize {
    corr_window(params.avg_length, params.max_period)
}

#[inline]
fn warmup_period(params: ResolvedParams) -> usize {
    params.max_period + corr_window_for_params(params) - 1
}

#[inline(always)]
fn first_valid_value(data: &[f64]) -> usize {
    let mut i = 0usize;
    while i < data.len() {
        if data[i].is_finite() {
            break;
        }
        i += 1;
    }
    i.min(data.len())
}

#[inline(always)]
fn count_valid_values(data: &[f64]) -> usize {
    data.iter().filter(|v| v.is_finite()).count()
}

#[inline]
fn resolve_params(
    params: &EhlersAutocorrelationPeriodogramParams,
    data_len: usize,
) -> Result<ResolvedParams, EhlersAutocorrelationPeriodogramError> {
    let min_period = params.min_period.unwrap_or(DEFAULT_MIN_PERIOD);
    if min_period < 3 {
        return Err(EhlersAutocorrelationPeriodogramError::InvalidMinPeriod { min_period });
    }

    let max_period = params.max_period.unwrap_or(DEFAULT_MAX_PERIOD);
    if max_period <= min_period || (data_len != 0 && max_period > data_len) {
        if max_period <= min_period {
            return Err(EhlersAutocorrelationPeriodogramError::InvalidPeriodOrder {
                min_period,
                max_period,
            });
        }
        return Err(EhlersAutocorrelationPeriodogramError::InvalidMaxPeriod {
            max_period,
            data_len,
        });
    }

    Ok(ResolvedParams {
        min_period,
        max_period,
        avg_length: params.avg_length.unwrap_or(DEFAULT_AVG_LENGTH),
        enhance: params.enhance.unwrap_or(DEFAULT_ENHANCE),
    })
}

#[inline]
fn prepare_input<'a>(
    input: &'a EhlersAutocorrelationPeriodogramInput<'a>,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, ResolvedParams, Kernel), EhlersAutocorrelationPeriodogramError> {
    let data = input.as_ref();
    if data.is_empty() {
        return Err(EhlersAutocorrelationPeriodogramError::EmptyInputData);
    }
    let first = first_valid_value(data);
    if first >= data.len() {
        return Err(EhlersAutocorrelationPeriodogramError::AllValuesNaN);
    }
    let params = resolve_params(&input.params, data.len())?;
    let valid = count_valid_values(data);
    let needed = warmup_period(params) + 1;
    if valid < needed {
        return Err(EhlersAutocorrelationPeriodogramError::NotEnoughValidData { needed, valid });
    }
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other.to_non_batch(),
    };
    Ok((data, first, params, chosen))
}

#[inline(always)]
fn row_from_slice(
    data: &[f64],
    params: &EhlersAutocorrelationPeriodogramParams,
    dominant_cycle_out: &mut [f64],
    normalized_power_out: &mut [f64],
) -> Result<(), EhlersAutocorrelationPeriodogramError> {
    let mut stream = EhlersAutocorrelationPeriodogramStream::try_new(params.clone())?;
    for i in 0..data.len() {
        match stream.update(data[i]) {
            Some((dominant_cycle, normalized_power)) => {
                dominant_cycle_out[i] = dominant_cycle;
                normalized_power_out[i] = normalized_power;
            }
            None => {
                dominant_cycle_out[i] = f64::NAN;
                normalized_power_out[i] = f64::NAN;
            }
        }
    }
    Ok(())
}

#[inline]
pub fn ehlers_autocorrelation_periodogram(
    input: &EhlersAutocorrelationPeriodogramInput,
) -> Result<EhlersAutocorrelationPeriodogramOutput, EhlersAutocorrelationPeriodogramError> {
    ehlers_autocorrelation_periodogram_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn ehlers_autocorrelation_periodogram_with_kernel(
    input: &EhlersAutocorrelationPeriodogramInput,
    kernel: Kernel,
) -> Result<EhlersAutocorrelationPeriodogramOutput, EhlersAutocorrelationPeriodogramError> {
    let (data, first, params, _chosen) = prepare_input(input, kernel)?;
    let mut dominant_cycle = alloc_with_nan_prefix(data.len(), first);
    let mut normalized_power = alloc_with_nan_prefix(data.len(), first);
    row_from_slice(
        data,
        &EhlersAutocorrelationPeriodogramParams {
            min_period: Some(params.min_period),
            max_period: Some(params.max_period),
            avg_length: Some(params.avg_length),
            enhance: Some(params.enhance),
        },
        &mut dominant_cycle,
        &mut normalized_power,
    )?;
    Ok(EhlersAutocorrelationPeriodogramOutput {
        dominant_cycle,
        normalized_power,
    })
}

#[inline]
pub fn ehlers_autocorrelation_periodogram_into_slices(
    dominant_cycle_out: &mut [f64],
    normalized_power_out: &mut [f64],
    input: &EhlersAutocorrelationPeriodogramInput,
    kernel: Kernel,
) -> Result<(), EhlersAutocorrelationPeriodogramError> {
    let (data, _first, params, _chosen) = prepare_input(input, kernel)?;
    if dominant_cycle_out.len() != data.len() || normalized_power_out.len() != data.len() {
        return Err(
            EhlersAutocorrelationPeriodogramError::OutputLengthMismatch {
                expected: data.len(),
                dominant_cycle_got: dominant_cycle_out.len(),
                normalized_power_got: normalized_power_out.len(),
            },
        );
    }
    row_from_slice(
        data,
        &EhlersAutocorrelationPeriodogramParams {
            min_period: Some(params.min_period),
            max_period: Some(params.max_period),
            avg_length: Some(params.avg_length),
            enhance: Some(params.enhance),
        },
        dominant_cycle_out,
        normalized_power_out,
    )
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn ehlers_autocorrelation_periodogram_into(
    input: &EhlersAutocorrelationPeriodogramInput,
    dominant_cycle_out: &mut [f64],
    normalized_power_out: &mut [f64],
) -> Result<(), EhlersAutocorrelationPeriodogramError> {
    ehlers_autocorrelation_periodogram_into_slices(
        dominant_cycle_out,
        normalized_power_out,
        input,
        Kernel::Auto,
    )
}

#[derive(Clone, Debug)]
pub struct EhlersAutocorrelationPeriodogramBatchRange {
    pub min_period: (usize, usize, usize),
    pub max_period: (usize, usize, usize),
    pub avg_length: (usize, usize, usize),
    pub enhance: bool,
}

impl Default for EhlersAutocorrelationPeriodogramBatchRange {
    fn default() -> Self {
        Self {
            min_period: (DEFAULT_MIN_PERIOD, DEFAULT_MIN_PERIOD, 0),
            max_period: (DEFAULT_MAX_PERIOD, DEFAULT_MAX_PERIOD, 0),
            avg_length: (DEFAULT_AVG_LENGTH, DEFAULT_AVG_LENGTH, 0),
            enhance: DEFAULT_ENHANCE,
        }
    }
}

#[derive(Clone, Debug)]
pub struct EhlersAutocorrelationPeriodogramBatchBuilder {
    range: EhlersAutocorrelationPeriodogramBatchRange,
    kernel: Kernel,
}

impl Default for EhlersAutocorrelationPeriodogramBatchBuilder {
    fn default() -> Self {
        Self {
            range: EhlersAutocorrelationPeriodogramBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl EhlersAutocorrelationPeriodogramBatchBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn min_period_range(mut self, range: (usize, usize, usize)) -> Self {
        self.range.min_period = range;
        self
    }

    #[inline]
    pub fn max_period_range(mut self, range: (usize, usize, usize)) -> Self {
        self.range.max_period = range;
        self
    }

    #[inline]
    pub fn avg_length_range(mut self, range: (usize, usize, usize)) -> Self {
        self.range.avg_length = range;
        self
    }

    #[inline]
    pub fn enhance(mut self, enhance: bool) -> Self {
        self.range.enhance = enhance;
        self
    }

    #[inline]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<EhlersAutocorrelationPeriodogramBatchOutput, EhlersAutocorrelationPeriodogramError>
    {
        ehlers_autocorrelation_periodogram_batch_with_kernel(data, &self.range, self.kernel)
    }

    #[inline]
    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<EhlersAutocorrelationPeriodogramBatchOutput, EhlersAutocorrelationPeriodogramError>
    {
        self.apply_slice(source_type(candles, source))
    }
}

#[derive(Clone, Debug)]
pub struct EhlersAutocorrelationPeriodogramBatchOutput {
    pub dominant_cycle: Vec<f64>,
    pub normalized_power: Vec<f64>,
    pub combos: Vec<EhlersAutocorrelationPeriodogramParams>,
    pub rows: usize,
    pub cols: usize,
}

#[inline(always)]
fn axis_usize(
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, EhlersAutocorrelationPeriodogramError> {
    if step == 0 || start == end {
        return Ok(vec![start]);
    }
    let step = step.max(1);
    if start < end {
        let mut out = Vec::new();
        let mut x = start;
        while x <= end {
            out.push(x);
            match x.checked_add(step) {
                Some(next) if next != x => x = next,
                _ => break,
            }
        }
        if out.is_empty() {
            return Err(EhlersAutocorrelationPeriodogramError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        Ok(out)
    } else {
        let mut out = Vec::new();
        let mut x = start;
        loop {
            out.push(x);
            if x == end {
                break;
            }
            let next = x.saturating_sub(step);
            if next == x || next < end {
                break;
            }
            x = next;
        }
        if out.is_empty() {
            return Err(EhlersAutocorrelationPeriodogramError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        Ok(out)
    }
}

#[inline(always)]
fn expand_grid(
    range: &EhlersAutocorrelationPeriodogramBatchRange,
) -> Result<Vec<EhlersAutocorrelationPeriodogramParams>, EhlersAutocorrelationPeriodogramError> {
    let mins = axis_usize(range.min_period)?;
    let maxes = axis_usize(range.max_period)?;
    let avgs = axis_usize(range.avg_length)?;

    let mut out = Vec::with_capacity(mins.len() * maxes.len() * avgs.len());
    for &min_period in &mins {
        for &max_period in &maxes {
            for &avg_length in &avgs {
                out.push(EhlersAutocorrelationPeriodogramParams {
                    min_period: Some(min_period),
                    max_period: Some(max_period),
                    avg_length: Some(avg_length),
                    enhance: Some(range.enhance),
                });
            }
        }
    }
    Ok(out)
}

#[inline]
pub fn ehlers_autocorrelation_periodogram_batch_with_kernel(
    data: &[f64],
    sweep: &EhlersAutocorrelationPeriodogramBatchRange,
    kernel: Kernel,
) -> Result<EhlersAutocorrelationPeriodogramBatchOutput, EhlersAutocorrelationPeriodogramError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => {
            return Err(EhlersAutocorrelationPeriodogramError::InvalidKernelForBatch(other));
        }
    };
    ehlers_autocorrelation_periodogram_batch_inner(data, sweep, batch_kernel.to_non_batch(), false)
}

#[inline]
pub fn ehlers_autocorrelation_periodogram_batch_slice(
    data: &[f64],
    sweep: &EhlersAutocorrelationPeriodogramBatchRange,
) -> Result<EhlersAutocorrelationPeriodogramBatchOutput, EhlersAutocorrelationPeriodogramError> {
    ehlers_autocorrelation_periodogram_batch_with_kernel(data, sweep, Kernel::Auto)
}

#[inline]
pub fn ehlers_autocorrelation_periodogram_batch_par_slice(
    data: &[f64],
    sweep: &EhlersAutocorrelationPeriodogramBatchRange,
) -> Result<EhlersAutocorrelationPeriodogramBatchOutput, EhlersAutocorrelationPeriodogramError> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let kernel = detect_best_batch_kernel().to_non_batch();
        return ehlers_autocorrelation_periodogram_batch_inner(data, sweep, kernel, true);
    }
    #[cfg(target_arch = "wasm32")]
    {
        ehlers_autocorrelation_periodogram_batch_inner(data, sweep, detect_best_kernel(), false)
    }
}

pub fn ehlers_autocorrelation_periodogram_batch_inner(
    data: &[f64],
    sweep: &EhlersAutocorrelationPeriodogramBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<EhlersAutocorrelationPeriodogramBatchOutput, EhlersAutocorrelationPeriodogramError> {
    if data.is_empty() {
        return Err(EhlersAutocorrelationPeriodogramError::EmptyInputData);
    }
    let first = first_valid_value(data);
    if first >= data.len() {
        return Err(EhlersAutocorrelationPeriodogramError::AllValuesNaN);
    }
    let combos = expand_grid(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    let total = rows.checked_mul(cols).ok_or(
        EhlersAutocorrelationPeriodogramError::OutputLengthMismatch {
            expected: usize::MAX,
            dominant_cycle_got: 0,
            normalized_power_got: 0,
        },
    )?;

    let valid = count_valid_values(data);
    let mut warms = Vec::with_capacity(rows);
    for combo in &combos {
        let params = resolve_params(combo, cols)?;
        let needed = warmup_period(params) + 1;
        if valid < needed {
            return Err(EhlersAutocorrelationPeriodogramError::NotEnoughValidData {
                needed,
                valid,
            });
        }
        warms.push((first + warmup_period(params)).min(cols));
    }

    let mut dominant_mu = make_uninit_matrix(rows, cols);
    let mut power_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut dominant_mu, cols, &warms);
    init_matrix_prefixes(&mut power_mu, cols, &warms);

    let mut dominant_guard = ManuallyDrop::new(dominant_mu);
    let mut power_guard = ManuallyDrop::new(power_mu);
    let dominant_out =
        unsafe { std::slice::from_raw_parts_mut(dominant_guard.as_mut_ptr() as *mut f64, total) };
    let power_out =
        unsafe { std::slice::from_raw_parts_mut(power_guard.as_mut_ptr() as *mut f64, total) };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            dominant_out
                .par_chunks_mut(cols)
                .zip(power_out.par_chunks_mut(cols))
                .zip(combos.par_iter())
                .for_each(|((dst_dom, dst_pwr), combo)| {
                    let _ = row_from_slice(data, combo, dst_dom, dst_pwr);
                });
        }
    } else {
        let _ = kernel;
        for (row, combo) in combos.iter().enumerate() {
            let start = row * cols;
            let end = start + cols;
            row_from_slice(
                data,
                combo,
                &mut dominant_out[start..end],
                &mut power_out[start..end],
            )?;
        }
    }

    let dominant_cycle = unsafe {
        Vec::from_raw_parts(
            dominant_guard.as_mut_ptr() as *mut f64,
            dominant_guard.len(),
            dominant_guard.capacity(),
        )
    };
    let normalized_power = unsafe {
        Vec::from_raw_parts(
            power_guard.as_mut_ptr() as *mut f64,
            power_guard.len(),
            power_guard.capacity(),
        )
    };
    core::mem::forget(dominant_guard);
    core::mem::forget(power_guard);

    Ok(EhlersAutocorrelationPeriodogramBatchOutput {
        dominant_cycle,
        normalized_power,
        combos,
        rows,
        cols,
    })
}

pub fn ehlers_autocorrelation_periodogram_batch_inner_into(
    data: &[f64],
    sweep: &EhlersAutocorrelationPeriodogramBatchRange,
    kernel: Kernel,
    dominant_cycle_out: &mut [f64],
    normalized_power_out: &mut [f64],
) -> Result<Vec<EhlersAutocorrelationPeriodogramParams>, EhlersAutocorrelationPeriodogramError> {
    let out = ehlers_autocorrelation_periodogram_batch_inner(data, sweep, kernel, false)?;
    let total = out.rows * out.cols;
    if dominant_cycle_out.len() != total || normalized_power_out.len() != total {
        return Err(
            EhlersAutocorrelationPeriodogramError::OutputLengthMismatch {
                expected: total,
                dominant_cycle_got: dominant_cycle_out.len(),
                normalized_power_got: normalized_power_out.len(),
            },
        );
    }
    dominant_cycle_out.copy_from_slice(&out.dominant_cycle);
    normalized_power_out.copy_from_slice(&out.normalized_power);
    Ok(out.combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "ehlers_autocorrelation_periodogram")]
#[pyo3(signature = (data, min_period=None, max_period=None, avg_length=None, enhance=None, kernel=None))]
pub fn ehlers_autocorrelation_periodogram_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    min_period: Option<usize>,
    max_period: Option<usize>,
    avg_length: Option<usize>,
    enhance: Option<bool>,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let data = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let input = EhlersAutocorrelationPeriodogramInput::from_slice(
        data,
        EhlersAutocorrelationPeriodogramParams {
            min_period,
            max_period,
            avg_length,
            enhance,
        },
    );
    let out = py
        .allow_threads(|| ehlers_autocorrelation_periodogram_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        out.dominant_cycle.into_pyarray(py),
        out.normalized_power.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyclass(name = "EhlersAutocorrelationPeriodogramStream")]
pub struct EhlersAutocorrelationPeriodogramStreamPy {
    inner: EhlersAutocorrelationPeriodogramStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl EhlersAutocorrelationPeriodogramStreamPy {
    #[new]
    #[pyo3(signature = (min_period=DEFAULT_MIN_PERIOD, max_period=DEFAULT_MAX_PERIOD, avg_length=DEFAULT_AVG_LENGTH, enhance=DEFAULT_ENHANCE))]
    fn new(
        min_period: usize,
        max_period: usize,
        avg_length: usize,
        enhance: bool,
    ) -> PyResult<Self> {
        let inner = EhlersAutocorrelationPeriodogramStream::try_new(
            EhlersAutocorrelationPeriodogramParams {
                min_period: Some(min_period),
                max_period: Some(max_period),
                avg_length: Some(avg_length),
                enhance: Some(enhance),
            },
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        self.inner.update(value)
    }

    #[getter]
    fn warmup_period(&self) -> usize {
        self.inner.get_warmup_period()
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "ehlers_autocorrelation_periodogram_batch")]
#[pyo3(signature = (data, min_period_range=(DEFAULT_MIN_PERIOD, DEFAULT_MIN_PERIOD, 0), max_period_range=(DEFAULT_MAX_PERIOD, DEFAULT_MAX_PERIOD, 0), avg_length_range=(DEFAULT_AVG_LENGTH, DEFAULT_AVG_LENGTH, 0), enhance=DEFAULT_ENHANCE, kernel=None))]
pub fn ehlers_autocorrelation_periodogram_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    min_period_range: (usize, usize, usize),
    max_period_range: (usize, usize, usize),
    avg_length_range: (usize, usize, usize),
    enhance: bool,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let sweep = EhlersAutocorrelationPeriodogramBatchRange {
        min_period: min_period_range,
        max_period: max_period_range,
        avg_length: avg_length_range,
        enhance,
    };
    let combos = expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let dom_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let pwr_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let dom_slice = unsafe { dom_arr.as_slice_mut()? };
    let pwr_slice = unsafe { pwr_arr.as_slice_mut()? };

    let combos = py
        .allow_threads(|| {
            let batch = match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                other => other,
            };
            ehlers_autocorrelation_periodogram_batch_inner_into(
                data,
                &sweep,
                batch.to_non_batch(),
                dom_slice,
                pwr_slice,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("dominant_cycle", dom_arr.reshape((rows, cols))?)?;
    dict.set_item("normalized_power", pwr_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "min_periods",
        combos
            .iter()
            .map(|p| p.min_period.unwrap_or(DEFAULT_MIN_PERIOD) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "max_periods",
        combos
            .iter()
            .map(|p| p.max_period.unwrap_or(DEFAULT_MAX_PERIOD) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "avg_lengths",
        combos
            .iter()
            .map(|p| p.avg_length.unwrap_or(DEFAULT_AVG_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "enhance_flags",
        combos
            .iter()
            .map(|p| p.enhance.unwrap_or(DEFAULT_ENHANCE))
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_ehlers_autocorrelation_periodogram_module(
    module: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(
        ehlers_autocorrelation_periodogram_py,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        ehlers_autocorrelation_periodogram_batch_py,
        module
    )?)?;
    module.add_class::<EhlersAutocorrelationPeriodogramStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "ehlers_autocorrelation_periodogram_js")]
pub fn ehlers_autocorrelation_periodogram_js(
    data: &[f64],
    min_period: usize,
    max_period: usize,
    avg_length: usize,
    enhance: bool,
) -> Result<JsValue, JsValue> {
    let input = EhlersAutocorrelationPeriodogramInput::from_slice(
        data,
        EhlersAutocorrelationPeriodogramParams {
            min_period: Some(min_period),
            max_period: Some(max_period),
            avg_length: Some(avg_length),
            enhance: Some(enhance),
        },
    );
    let out = ehlers_autocorrelation_periodogram(&input)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let result = js_sys::Object::new();

    let dom = js_sys::Float64Array::new_with_length(out.dominant_cycle.len() as u32);
    dom.copy_from(&out.dominant_cycle);
    js_sys::Reflect::set(&result, &JsValue::from_str("dominant_cycle"), &dom)?;

    let pwr = js_sys::Float64Array::new_with_length(out.normalized_power.len() as u32);
    pwr.copy_from(&out.normalized_power);
    js_sys::Reflect::set(&result, &JsValue::from_str("normalized_power"), &pwr)?;

    Ok(result.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_autocorrelation_periodogram_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_autocorrelation_periodogram_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "ehlers_autocorrelation_periodogram_into")]
pub fn ehlers_autocorrelation_periodogram_into_js(
    data_ptr: *const f64,
    dominant_cycle_ptr: *mut f64,
    normalized_power_ptr: *mut f64,
    len: usize,
    min_period: usize,
    max_period: usize,
    avg_length: usize,
    enhance: bool,
) -> Result<(), JsValue> {
    if data_ptr.is_null() || dominant_cycle_ptr.is_null() || normalized_power_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let input = EhlersAutocorrelationPeriodogramInput::from_slice(
            data,
            EhlersAutocorrelationPeriodogramParams {
                min_period: Some(min_period),
                max_period: Some(max_period),
                avg_length: Some(avg_length),
                enhance: Some(enhance),
            },
        );
        let alias = data_ptr == dominant_cycle_ptr
            || data_ptr == normalized_power_ptr
            || dominant_cycle_ptr == normalized_power_ptr;
        if alias {
            let mut dom_tmp = vec![0.0; len];
            let mut pwr_tmp = vec![0.0; len];
            ehlers_autocorrelation_periodogram_into_slices(
                &mut dom_tmp,
                &mut pwr_tmp,
                &input,
                Kernel::Auto,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
            std::slice::from_raw_parts_mut(dominant_cycle_ptr, len).copy_from_slice(&dom_tmp);
            std::slice::from_raw_parts_mut(normalized_power_ptr, len).copy_from_slice(&pwr_tmp);
        } else {
            ehlers_autocorrelation_periodogram_into_slices(
                std::slice::from_raw_parts_mut(dominant_cycle_ptr, len),
                std::slice::from_raw_parts_mut(normalized_power_ptr, len),
                &input,
                Kernel::Auto,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct EhlersAutocorrelationPeriodogramBatchConfig {
    pub min_period_range: (usize, usize, usize),
    pub max_period_range: Option<(usize, usize, usize)>,
    pub avg_length_range: Option<(usize, usize, usize)>,
    pub enhance: Option<bool>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct EhlersAutocorrelationPeriodogramBatchJsOutput {
    pub dominant_cycle: Vec<f64>,
    pub normalized_power: Vec<f64>,
    pub combos: Vec<EhlersAutocorrelationPeriodogramParams>,
    pub min_periods: Vec<usize>,
    pub max_periods: Vec<usize>,
    pub avg_lengths: Vec<usize>,
    pub enhance_flags: Vec<bool>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "ehlers_autocorrelation_periodogram_batch_js")]
pub fn ehlers_autocorrelation_periodogram_batch_js(
    data: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let cfg: EhlersAutocorrelationPeriodogramBatchConfig =
        serde_wasm_bindgen::from_value(config)
            .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let sweep = EhlersAutocorrelationPeriodogramBatchRange {
        min_period: cfg.min_period_range,
        max_period: cfg
            .max_period_range
            .unwrap_or((DEFAULT_MAX_PERIOD, DEFAULT_MAX_PERIOD, 0)),
        avg_length: cfg
            .avg_length_range
            .unwrap_or((DEFAULT_AVG_LENGTH, DEFAULT_AVG_LENGTH, 0)),
        enhance: cfg.enhance.unwrap_or(DEFAULT_ENHANCE),
    };
    let out = ehlers_autocorrelation_periodogram_batch_inner(
        data,
        &sweep,
        detect_best_batch_kernel().to_non_batch(),
        false,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&EhlersAutocorrelationPeriodogramBatchJsOutput {
        min_periods: out
            .combos
            .iter()
            .map(|p| p.min_period.unwrap_or(DEFAULT_MIN_PERIOD))
            .collect(),
        max_periods: out
            .combos
            .iter()
            .map(|p| p.max_period.unwrap_or(DEFAULT_MAX_PERIOD))
            .collect(),
        avg_lengths: out
            .combos
            .iter()
            .map(|p| p.avg_length.unwrap_or(DEFAULT_AVG_LENGTH))
            .collect(),
        enhance_flags: out
            .combos
            .iter()
            .map(|p| p.enhance.unwrap_or(DEFAULT_ENHANCE))
            .collect(),
        dominant_cycle: out.dominant_cycle,
        normalized_power: out.normalized_power,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "ehlers_autocorrelation_periodogram_batch_into")]
pub fn ehlers_autocorrelation_periodogram_batch_into_js(
    data_ptr: *const f64,
    dominant_cycle_ptr: *mut f64,
    normalized_power_ptr: *mut f64,
    len: usize,
    min_start: usize,
    min_end: usize,
    min_step: usize,
    max_start: usize,
    max_end: usize,
    max_step: usize,
    avg_start: usize,
    avg_end: usize,
    avg_step: usize,
    enhance: bool,
) -> Result<usize, JsValue> {
    if data_ptr.is_null() || dominant_cycle_ptr.is_null() || normalized_power_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }
    let sweep = EhlersAutocorrelationPeriodogramBatchRange {
        min_period: (min_start, min_end, min_step),
        max_period: (max_start, max_end, max_step),
        avg_length: (avg_start, avg_end, avg_step),
        enhance,
    };
    let combos = expand_grid(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let total = rows
        .checked_mul(len)
        .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;

    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let alias = data_ptr == dominant_cycle_ptr
            || data_ptr == normalized_power_ptr
            || dominant_cycle_ptr == normalized_power_ptr;
        if alias {
            let mut dom_tmp = vec![0.0; total];
            let mut pwr_tmp = vec![0.0; total];
            ehlers_autocorrelation_periodogram_batch_inner_into(
                data,
                &sweep,
                detect_best_batch_kernel().to_non_batch(),
                &mut dom_tmp,
                &mut pwr_tmp,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
            std::slice::from_raw_parts_mut(dominant_cycle_ptr, total).copy_from_slice(&dom_tmp);
            std::slice::from_raw_parts_mut(normalized_power_ptr, total).copy_from_slice(&pwr_tmp);
        } else {
            ehlers_autocorrelation_periodogram_batch_inner_into(
                data,
                &sweep,
                detect_best_batch_kernel().to_non_batch(),
                std::slice::from_raw_parts_mut(dominant_cycle_ptr, total),
                std::slice::from_raw_parts_mut(normalized_power_ptr, total),
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }
    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_autocorrelation_periodogram_output_into_js(
    data: &[f64],
    min_period: usize,
    max_period: usize,
    avg_length: usize,
    enhance: bool,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value =
        ehlers_autocorrelation_periodogram_js(data, min_period, max_period, avg_length, enhance)?;
    crate::write_wasm_object_f64_outputs(
        "ehlers_autocorrelation_periodogram_output_into_js",
        &value,
        out,
    )
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_autocorrelation_periodogram_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = ehlers_autocorrelation_periodogram_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "ehlers_autocorrelation_periodogram_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq_or_nan(a: f64, b: f64) -> bool {
        (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-9
    }

    fn cycle_data(len: usize, period: f64) -> Vec<f64> {
        (0..len)
            .map(|i| {
                let phase = 2.0 * PI * i as f64 / period;
                phase.sin() + 0.15 * (phase * 0.5).cos()
            })
            .collect()
    }

    #[test]
    fn eacp_output_contract_on_cycle() {
        let data = cycle_data(256, 20.0);
        let input = EhlersAutocorrelationPeriodogramInput::from_slice(
            &data,
            EhlersAutocorrelationPeriodogramParams::default(),
        );
        let out = ehlers_autocorrelation_periodogram(&input).unwrap();
        assert_eq!(out.dominant_cycle.len(), data.len());
        assert_eq!(out.normalized_power.len(), data.len());
        assert!(out.dominant_cycle[..50].iter().all(|v| v.is_nan()));
        let last_dc = *out.dominant_cycle.last().unwrap();
        let last_pwr = *out.normalized_power.last().unwrap();
        assert!(last_dc.is_finite());
        assert!(last_pwr.is_finite());
        assert!(last_dc > 15.0 && last_dc < 25.0);
        assert!((0.0..=1.0).contains(&last_pwr));
    }

    #[test]
    fn eacp_rejects_invalid_params() {
        let data = cycle_data(64, 16.0);
        let input = EhlersAutocorrelationPeriodogramInput::from_slice(
            &data,
            EhlersAutocorrelationPeriodogramParams {
                min_period: Some(2),
                max_period: Some(32),
                avg_length: Some(3),
                enhance: Some(true),
            },
        );
        assert!(matches!(
            ehlers_autocorrelation_periodogram(&input),
            Err(EhlersAutocorrelationPeriodogramError::InvalidMinPeriod { .. })
        ));
    }

    #[test]
    fn eacp_stream_matches_batch_with_reset() {
        let mut data = cycle_data(192, 18.0);
        data[90] = f64::NAN;
        let params = EhlersAutocorrelationPeriodogramParams {
            min_period: Some(8),
            max_period: Some(32),
            avg_length: Some(3),
            enhance: Some(true),
        };
        let batch = ehlers_autocorrelation_periodogram(
            &EhlersAutocorrelationPeriodogramInput::from_slice(&data, params.clone()),
        )
        .unwrap();
        let mut stream = EhlersAutocorrelationPeriodogramStream::try_new(params).unwrap();
        let mut dom = Vec::with_capacity(data.len());
        let mut pwr = Vec::with_capacity(data.len());
        for &value in &data {
            match stream.update(value) {
                Some((a, b)) => {
                    dom.push(a);
                    pwr.push(b);
                }
                None => {
                    dom.push(f64::NAN);
                    pwr.push(f64::NAN);
                }
            }
        }
        assert_eq!(stream.get_warmup_period(), 34);
        for i in 0..data.len() {
            assert!(approx_eq_or_nan(dom[i], batch.dominant_cycle[i]));
            assert!(approx_eq_or_nan(pwr[i], batch.normalized_power[i]));
        }
        assert!(batch.dominant_cycle[90].is_nan());
        assert!(batch.dominant_cycle[120].is_nan());
        assert!(batch.dominant_cycle.last().unwrap().is_finite());
    }

    #[test]
    fn eacp_batch_matches_single() {
        let data = cycle_data(160, 22.0);
        let sweep = EhlersAutocorrelationPeriodogramBatchRange {
            min_period: (8, 8, 0),
            max_period: (48, 48, 0),
            avg_length: (3, 3, 0),
            enhance: true,
        };
        let batch = ehlers_autocorrelation_periodogram_batch_with_kernel(
            &data,
            &sweep,
            Kernel::ScalarBatch,
        )
        .unwrap();
        let single =
            ehlers_autocorrelation_periodogram(&EhlersAutocorrelationPeriodogramInput::from_slice(
                &data,
                EhlersAutocorrelationPeriodogramParams::default(),
            ))
            .unwrap();
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, data.len());
        for i in 0..data.len() {
            assert!(approx_eq_or_nan(
                batch.dominant_cycle[i],
                single.dominant_cycle[i]
            ));
            assert!(approx_eq_or_nan(
                batch.normalized_power[i],
                single.normalized_power[i]
            ));
        }
    }

    #[test]
    fn eacp_batch_metadata_and_into() {
        let data = cycle_data(160, 21.0);
        let sweep = EhlersAutocorrelationPeriodogramBatchRange {
            min_period: (8, 10, 2),
            max_period: (24, 28, 4),
            avg_length: (3, 3, 0),
            enhance: false,
        };
        let batch =
            ehlers_autocorrelation_periodogram_batch_inner(&data, &sweep, Kernel::Scalar, false)
                .unwrap();
        assert_eq!(batch.rows, 4);
        let total = batch.rows * batch.cols;
        let mut dom = vec![0.0; total];
        let mut pwr = vec![0.0; total];
        let combos = ehlers_autocorrelation_periodogram_batch_inner_into(
            &data,
            &sweep,
            Kernel::Scalar,
            &mut dom,
            &mut pwr,
        )
        .unwrap();
        assert_eq!(combos, batch.combos);
        for i in 0..total {
            assert!(approx_eq_or_nan(dom[i], batch.dominant_cycle[i]));
            assert!(approx_eq_or_nan(pwr[i], batch.normalized_power[i]));
        }
    }
}
