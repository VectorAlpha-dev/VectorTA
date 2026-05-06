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

use crate::indicators::atr::{AtrParams, AtrStream};
use crate::utilities::data_loader::Candles;
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::collections::VecDeque;
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

const DEFAULT_GAUSSIAN_LENGTH: usize = 15;
const DEFAULT_POLES: usize = 3;
const DEFAULT_SMOOTHING_LENGTH: usize = 22;
const DEFAULT_LINREG_OFFSET: usize = 7;
const SUPERTREND_FACTOR: f64 = 0.15;
const SUPERTREND_ATR_PERIOD: usize = 21;

#[derive(Debug, Clone)]
pub enum SmoothedGaussianTrendFilterData<'a> {
    Candles(&'a Candles),
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct SmoothedGaussianTrendFilterOutput {
    pub filter: Vec<f64>,
    pub supertrend: Vec<f64>,
    pub trend: Vec<f64>,
    pub ranging: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct SmoothedGaussianTrendFilterParams {
    pub gaussian_length: Option<usize>,
    pub poles: Option<usize>,
    pub smoothing_length: Option<usize>,
    pub linreg_offset: Option<usize>,
}

impl Default for SmoothedGaussianTrendFilterParams {
    fn default() -> Self {
        Self {
            gaussian_length: Some(DEFAULT_GAUSSIAN_LENGTH),
            poles: Some(DEFAULT_POLES),
            smoothing_length: Some(DEFAULT_SMOOTHING_LENGTH),
            linreg_offset: Some(DEFAULT_LINREG_OFFSET),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SmoothedGaussianTrendFilterInput<'a> {
    pub data: SmoothedGaussianTrendFilterData<'a>,
    pub params: SmoothedGaussianTrendFilterParams,
}

impl<'a> SmoothedGaussianTrendFilterInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: SmoothedGaussianTrendFilterParams) -> Self {
        Self {
            data: SmoothedGaussianTrendFilterData::Candles(candles),
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: SmoothedGaussianTrendFilterParams,
    ) -> Self {
        Self {
            data: SmoothedGaussianTrendFilterData::Slices { high, low, close },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, SmoothedGaussianTrendFilterParams::default())
    }

    #[inline]
    pub fn get_gaussian_length(&self) -> usize {
        self.params
            .gaussian_length
            .unwrap_or(DEFAULT_GAUSSIAN_LENGTH)
    }

    #[inline]
    pub fn get_poles(&self) -> usize {
        self.params.poles.unwrap_or(DEFAULT_POLES)
    }

    #[inline]
    pub fn get_smoothing_length(&self) -> usize {
        self.params
            .smoothing_length
            .unwrap_or(DEFAULT_SMOOTHING_LENGTH)
    }

    #[inline]
    pub fn get_linreg_offset(&self) -> usize {
        self.params.linreg_offset.unwrap_or(DEFAULT_LINREG_OFFSET)
    }

    #[inline]
    pub fn as_refs(&'a self) -> (&'a [f64], &'a [f64], &'a [f64]) {
        match &self.data {
            SmoothedGaussianTrendFilterData::Candles(candles) => (
                candles.high.as_slice(),
                candles.low.as_slice(),
                candles.close.as_slice(),
            ),
            SmoothedGaussianTrendFilterData::Slices { high, low, close } => (*high, *low, *close),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SmoothedGaussianTrendFilterBuilder {
    gaussian_length: Option<usize>,
    poles: Option<usize>,
    smoothing_length: Option<usize>,
    linreg_offset: Option<usize>,
    kernel: Kernel,
}

impl Default for SmoothedGaussianTrendFilterBuilder {
    fn default() -> Self {
        Self {
            gaussian_length: None,
            poles: None,
            smoothing_length: None,
            linreg_offset: None,
            kernel: Kernel::Auto,
        }
    }
}

impl SmoothedGaussianTrendFilterBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn gaussian_length(mut self, value: usize) -> Self {
        self.gaussian_length = Some(value);
        self
    }

    #[inline]
    pub fn poles(mut self, value: usize) -> Self {
        self.poles = Some(value);
        self
    }

    #[inline]
    pub fn smoothing_length(mut self, value: usize) -> Self {
        self.smoothing_length = Some(value);
        self
    }

    #[inline]
    pub fn linreg_offset(mut self, value: usize) -> Self {
        self.linreg_offset = Some(value);
        self
    }

    #[inline]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<SmoothedGaussianTrendFilterOutput, SmoothedGaussianTrendFilterError> {
        let input = SmoothedGaussianTrendFilterInput::from_candles(
            candles,
            SmoothedGaussianTrendFilterParams {
                gaussian_length: self.gaussian_length,
                poles: self.poles,
                smoothing_length: self.smoothing_length,
                linreg_offset: self.linreg_offset,
            },
        );
        smoothed_gaussian_trend_filter_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<SmoothedGaussianTrendFilterOutput, SmoothedGaussianTrendFilterError> {
        let input = SmoothedGaussianTrendFilterInput::from_slices(
            high,
            low,
            close,
            SmoothedGaussianTrendFilterParams {
                gaussian_length: self.gaussian_length,
                poles: self.poles,
                smoothing_length: self.smoothing_length,
                linreg_offset: self.linreg_offset,
            },
        );
        smoothed_gaussian_trend_filter_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn into_stream(
        self,
    ) -> Result<SmoothedGaussianTrendFilterStream, SmoothedGaussianTrendFilterError> {
        SmoothedGaussianTrendFilterStream::try_new(SmoothedGaussianTrendFilterParams {
            gaussian_length: self.gaussian_length,
            poles: self.poles,
            smoothing_length: self.smoothing_length,
            linreg_offset: self.linreg_offset,
        })
    }
}

#[derive(Debug, Error)]
pub enum SmoothedGaussianTrendFilterError {
    #[error("smoothed_gaussian_trend_filter: Empty input data.")]
    EmptyInputData,
    #[error(
        "smoothed_gaussian_trend_filter: Input length mismatch: high={high}, low={low}, close={close}"
    )]
    DataLengthMismatch {
        high: usize,
        low: usize,
        close: usize,
    },
    #[error("smoothed_gaussian_trend_filter: All input values are invalid.")]
    AllValuesNaN,
    #[error(
        "smoothed_gaussian_trend_filter: Invalid gaussian_length: gaussian_length = {gaussian_length}, data length = {data_len}"
    )]
    InvalidGaussianLength {
        gaussian_length: usize,
        data_len: usize,
    },
    #[error("smoothed_gaussian_trend_filter: Invalid poles: expected 1..4, got {poles}")]
    InvalidPoles { poles: usize },
    #[error(
        "smoothed_gaussian_trend_filter: Invalid smoothing_length: smoothing_length = {smoothing_length}, data length = {data_len}"
    )]
    InvalidSmoothingLength {
        smoothing_length: usize,
        data_len: usize,
    },
    #[error(
        "smoothed_gaussian_trend_filter: Not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "smoothed_gaussian_trend_filter: Output length mismatch: expected={expected}, got={got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error(
        "smoothed_gaussian_trend_filter: Invalid range: start={start}, end={end}, step={step}"
    )]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("smoothed_gaussian_trend_filter: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline(always)]
fn valid_bar(high: f64, low: f64, close: f64) -> bool {
    high.is_finite() && low.is_finite() && close.is_finite() && high >= low
}

#[inline(always)]
fn first_valid_bar(high: &[f64], low: &[f64], close: &[f64]) -> Option<usize> {
    (0..close.len()).find(|&i| valid_bar(high[i], low[i], close[i]))
}

#[inline(always)]
fn normalize_kernel(kernel: Kernel) -> Kernel {
    match kernel {
        Kernel::Auto => detect_best_kernel(),
        other if other.is_batch() => other.to_non_batch(),
        other => other,
    }
}

#[inline(always)]
fn validate_lengths(
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> Result<(), SmoothedGaussianTrendFilterError> {
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(SmoothedGaussianTrendFilterError::EmptyInputData);
    }
    if high.len() != low.len() || low.len() != close.len() {
        return Err(SmoothedGaussianTrendFilterError::DataLengthMismatch {
            high: high.len(),
            low: low.len(),
            close: close.len(),
        });
    }
    Ok(())
}

#[inline(always)]
fn validate_params(
    gaussian_length: usize,
    poles: usize,
    smoothing_length: usize,
    len: usize,
) -> Result<(), SmoothedGaussianTrendFilterError> {
    if gaussian_length == 0 || gaussian_length > len {
        return Err(SmoothedGaussianTrendFilterError::InvalidGaussianLength {
            gaussian_length,
            data_len: len,
        });
    }
    if !(1..=4).contains(&poles) {
        return Err(SmoothedGaussianTrendFilterError::InvalidPoles { poles });
    }
    if smoothing_length == 0 || smoothing_length > len {
        return Err(SmoothedGaussianTrendFilterError::InvalidSmoothingLength {
            smoothing_length,
            data_len: len,
        });
    }
    Ok(())
}

#[inline(always)]
fn gaussian_alpha(length: usize, poles: usize) -> f64 {
    let freq = (2.0 * std::f64::consts::PI) / length as f64;
    let factor_b = (1.0 - freq.cos()) / (1.414_f64.powf(2.0 / poles as f64) - 1.0);
    -factor_b + (factor_b * factor_b + 2.0 * factor_b).sqrt()
}

#[derive(Clone, Debug)]
struct GaussianPoleState {
    poles: usize,
    alpha: f64,
    history: [f64; 4],
}

impl GaussianPoleState {
    #[inline]
    fn new(length: usize, poles: usize) -> Self {
        Self {
            poles,
            alpha: gaussian_alpha(length, poles),
            history: [0.0; 4],
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.history = [0.0; 4];
    }

    #[inline]
    fn update(&mut self, input: f64) -> f64 {
        let alpha = self.alpha;
        let oma = 1.0 - alpha;
        let out = match self.poles {
            1 => alpha * input + oma * self.history[0],
            2 => alpha * alpha * input + 2.0 * oma * self.history[0] - oma * oma * self.history[1],
            3 => {
                let oma2 = oma * oma;
                let oma3 = oma2 * oma;
                alpha * alpha * alpha * input + 3.0 * oma * self.history[0]
                    - 3.0 * oma2 * self.history[1]
                    + oma3 * self.history[2]
            }
            _ => {
                let oma2 = oma * oma;
                let oma3 = oma2 * oma;
                let oma4 = oma3 * oma;
                let alpha4 = alpha * alpha * alpha * alpha;
                alpha4 * input + 4.0 * oma * self.history[0] - 6.0 * oma2 * self.history[1]
                    + 4.0 * oma3 * self.history[2]
                    - oma4 * self.history[3]
            }
        };
        self.history[3] = self.history[2];
        self.history[2] = self.history[1];
        self.history[1] = self.history[0];
        self.history[0] = out;
        out
    }
}

#[derive(Clone, Debug)]
struct LinRegOffsetState {
    period: usize,
    offset: usize,
    buffer: VecDeque<f64>,
    x_sum: f64,
    denom_inv: f64,
}

impl LinRegOffsetState {
    #[inline]
    fn new(period: usize, offset: usize) -> Self {
        let mut x_sum = 0.0;
        let mut x2_sum = 0.0;
        for i in 1..=period {
            let x = i as f64;
            x_sum += x;
            x2_sum += x * x;
        }
        let pf = period as f64;
        let denom_inv = 1.0 / (pf * x2_sum - x_sum * x_sum);
        Self {
            period,
            offset,
            buffer: VecDeque::with_capacity(period),
            x_sum,
            denom_inv,
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.buffer.clear();
    }

    #[inline]
    fn update(&mut self, value: f64) -> Option<f64> {
        if self.buffer.len() == self.period {
            self.buffer.pop_front();
        }
        self.buffer.push_back(value);
        if self.buffer.len() < self.period {
            return None;
        }

        let mut y_sum = 0.0;
        let mut xy_sum = 0.0;
        for (idx, &y) in self.buffer.iter().enumerate() {
            let x = (idx + 1) as f64;
            y_sum += y;
            xy_sum += x * y;
        }

        let pf = self.period as f64;
        let b = (pf * xy_sum - self.x_sum * y_sum) * self.denom_inv;
        let a = (y_sum - b * self.x_sum) / pf;
        let projected_x = pf - self.offset as f64;
        Some(a + b * projected_x)
    }
}

#[derive(Clone, Debug)]
struct PineSupertrendState {
    factor: f64,
    prev_src: f64,
    prev_upper: f64,
    prev_lower: f64,
    prev_supertrend: f64,
    prev_atr_valid: bool,
    initialized: bool,
}

impl PineSupertrendState {
    #[inline]
    fn new(factor: f64) -> Self {
        Self {
            factor,
            prev_src: f64::NAN,
            prev_upper: f64::NAN,
            prev_lower: f64::NAN,
            prev_supertrend: f64::NAN,
            prev_atr_valid: false,
            initialized: false,
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.prev_src = f64::NAN;
        self.prev_upper = f64::NAN;
        self.prev_lower = f64::NAN;
        self.prev_supertrend = f64::NAN;
        self.prev_atr_valid = false;
        self.initialized = false;
    }

    #[inline]
    fn update(&mut self, src: f64, atr: f64) -> f64 {
        let mut upper = src + self.factor * atr;
        let mut lower = src - self.factor * atr;

        let prev_upper = if self.initialized {
            self.prev_upper
        } else {
            upper
        };
        let prev_lower = if self.initialized {
            self.prev_lower
        } else {
            lower
        };
        let prev_src = if self.initialized { self.prev_src } else { src };

        if !(lower > prev_lower || prev_src < prev_lower) {
            lower = prev_lower;
        }
        if !(upper < prev_upper || prev_src > prev_upper) {
            upper = prev_upper;
        }

        let direction = if !self.prev_atr_valid {
            1.0
        } else if self.prev_supertrend == self.prev_upper {
            if src > upper {
                -1.0
            } else {
                1.0
            }
        } else if src < lower {
            1.0
        } else {
            -1.0
        };

        let supertrend = if direction == -1.0 { lower } else { upper };

        self.prev_src = src;
        self.prev_upper = upper;
        self.prev_lower = lower;
        self.prev_supertrend = supertrend;
        self.prev_atr_valid = true;
        self.initialized = true;

        supertrend
    }
}

#[derive(Clone, Debug)]
pub struct SmoothedGaussianTrendFilterStream {
    gaussian: GaussianPoleState,
    linreg: LinRegOffsetState,
    atr: AtrStream,
    supertrend: PineSupertrendState,
    prev_final: f64,
    has_prev_final: bool,
}

impl SmoothedGaussianTrendFilterStream {
    pub fn try_new(
        params: SmoothedGaussianTrendFilterParams,
    ) -> Result<Self, SmoothedGaussianTrendFilterError> {
        let gaussian_length = params.gaussian_length.unwrap_or(DEFAULT_GAUSSIAN_LENGTH);
        let poles = params.poles.unwrap_or(DEFAULT_POLES);
        let smoothing_length = params.smoothing_length.unwrap_or(DEFAULT_SMOOTHING_LENGTH);
        let linreg_offset = params.linreg_offset.unwrap_or(DEFAULT_LINREG_OFFSET);

        validate_params(
            gaussian_length,
            poles,
            smoothing_length,
            smoothing_length
                .max(gaussian_length)
                .max(SUPERTREND_ATR_PERIOD),
        )?;

        let atr = AtrStream::try_new(AtrParams {
            length: Some(SUPERTREND_ATR_PERIOD),
        })
        .map_err(
            |_| SmoothedGaussianTrendFilterError::InvalidGaussianLength {
                gaussian_length,
                data_len: smoothing_length
                    .max(gaussian_length)
                    .max(SUPERTREND_ATR_PERIOD),
            },
        )?;

        Ok(Self {
            gaussian: GaussianPoleState::new(gaussian_length, poles),
            linreg: LinRegOffsetState::new(smoothing_length, linreg_offset),
            atr,
            supertrend: PineSupertrendState::new(SUPERTREND_FACTOR),
            prev_final: f64::NAN,
            has_prev_final: false,
        })
    }

    #[inline]
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64, f64, f64)> {
        if !valid_bar(high, low, close) {
            self.gaussian.reset();
            self.linreg.reset();
            self.supertrend.reset();
            self.prev_final = f64::NAN;
            self.has_prev_final = false;
            self.atr = AtrStream::try_new(AtrParams {
                length: Some(SUPERTREND_ATR_PERIOD),
            })
            .ok()?;
            return None;
        }

        let atr = self.atr.update(high, low, close);
        let gaussian = self.gaussian.update(close);
        let final_value = self.linreg.update(gaussian)?;
        let atr = atr?;
        let supertrend = self.supertrend.update(final_value, atr);
        let trend = if final_value > supertrend { 1.0 } else { -1.0 };
        let slope_trend = if self.has_prev_final && final_value > self.prev_final {
            1.0
        } else {
            -1.0
        };
        let ranging = if slope_trend * trend < 0.0 { 1.0 } else { 0.0 };
        self.prev_final = final_value;
        self.has_prev_final = true;
        Some((final_value, supertrend, trend, ranging))
    }
}

#[inline]
pub fn smoothed_gaussian_trend_filter(
    input: &SmoothedGaussianTrendFilterInput<'_>,
) -> Result<SmoothedGaussianTrendFilterOutput, SmoothedGaussianTrendFilterError> {
    smoothed_gaussian_trend_filter_with_kernel(input, Kernel::Auto)
}

fn smoothed_gaussian_trend_filter_row_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    gaussian_length: usize,
    poles: usize,
    smoothing_length: usize,
    linreg_offset: usize,
    out_filter: &mut [f64],
    out_supertrend: &mut [f64],
    out_trend: &mut [f64],
    out_ranging: &mut [f64],
) {
    out_filter.fill(f64::NAN);
    out_supertrend.fill(f64::NAN);
    out_trend.fill(f64::NAN);
    out_ranging.fill(f64::NAN);

    let mut gaussian = GaussianPoleState::new(gaussian_length, poles);
    let mut linreg = LinRegOffsetState::new(smoothing_length, linreg_offset);
    let mut atr = AtrStream::try_new(AtrParams {
        length: Some(SUPERTREND_ATR_PERIOD),
    })
    .expect("valid ATR params");
    let mut supertrend = PineSupertrendState::new(SUPERTREND_FACTOR);
    let mut prev_final = f64::NAN;
    let mut has_prev_final = false;

    for i in 0..close.len() {
        if !valid_bar(high[i], low[i], close[i]) {
            gaussian.reset();
            linreg.reset();
            supertrend.reset();
            atr = AtrStream::try_new(AtrParams {
                length: Some(SUPERTREND_ATR_PERIOD),
            })
            .expect("valid ATR params");
            prev_final = f64::NAN;
            has_prev_final = false;
            continue;
        }

        let atr_value = atr.update(high[i], low[i], close[i]);
        let gaussian_value = gaussian.update(close[i]);
        let final_value = match linreg.update(gaussian_value) {
            Some(value) => value,
            None => continue,
        };
        out_filter[i] = final_value;

        let atr_value = match atr_value {
            Some(value) => value,
            None => continue,
        };
        let supertrend_value = supertrend.update(final_value, atr_value);
        out_supertrend[i] = supertrend_value;

        let trend = if final_value > supertrend_value {
            1.0
        } else {
            -1.0
        };
        let slope_trend = if has_prev_final && final_value > prev_final {
            1.0
        } else {
            -1.0
        };
        let ranging = if slope_trend * trend < 0.0 { 1.0 } else { 0.0 };

        out_trend[i] = trend;
        out_ranging[i] = ranging;
        prev_final = final_value;
        has_prev_final = true;
    }
}

fn smoothed_gaussian_trend_filter_filter_row_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    gaussian_length: usize,
    poles: usize,
    smoothing_length: usize,
    linreg_offset: usize,
    out_filter: &mut [f64],
) {
    out_filter.fill(f64::NAN);

    let mut gaussian = GaussianPoleState::new(gaussian_length, poles);
    let mut linreg = LinRegOffsetState::new(smoothing_length, linreg_offset);

    for i in 0..close.len() {
        if !valid_bar(high[i], low[i], close[i]) {
            gaussian.reset();
            linreg.reset();
            continue;
        }

        let gaussian_value = gaussian.update(close[i]);
        if let Some(final_value) = linreg.update(gaussian_value) {
            out_filter[i] = final_value;
        }
    }
}

pub(crate) fn smoothed_gaussian_trend_filter_filter_with_kernel(
    input: &SmoothedGaussianTrendFilterInput<'_>,
    kernel: Kernel,
) -> Result<Vec<f64>, SmoothedGaussianTrendFilterError> {
    let (high, low, close) = input.as_refs();
    validate_lengths(high, low, close)?;
    let len = close.len();
    let gaussian_length = input.get_gaussian_length();
    let poles = input.get_poles();
    let smoothing_length = input.get_smoothing_length();
    let linreg_offset = input.get_linreg_offset();
    validate_params(gaussian_length, poles, smoothing_length, len)?;

    let first_valid =
        first_valid_bar(high, low, close).ok_or(SmoothedGaussianTrendFilterError::AllValuesNaN)?;
    let valid = len - first_valid;
    let needed = smoothing_length.max(SUPERTREND_ATR_PERIOD);
    if valid < needed {
        return Err(SmoothedGaussianTrendFilterError::NotEnoughValidData { needed, valid });
    }

    let _chosen = normalize_kernel(kernel);
    let mut filter = alloc_with_nan_prefix(len, first_valid);
    smoothed_gaussian_trend_filter_filter_row_scalar(
        high,
        low,
        close,
        gaussian_length,
        poles,
        smoothing_length,
        linreg_offset,
        &mut filter,
    );
    Ok(filter)
}

pub fn smoothed_gaussian_trend_filter_with_kernel(
    input: &SmoothedGaussianTrendFilterInput<'_>,
    kernel: Kernel,
) -> Result<SmoothedGaussianTrendFilterOutput, SmoothedGaussianTrendFilterError> {
    let (high, low, close) = input.as_refs();
    validate_lengths(high, low, close)?;
    let len = close.len();
    let gaussian_length = input.get_gaussian_length();
    let poles = input.get_poles();
    let smoothing_length = input.get_smoothing_length();
    let linreg_offset = input.get_linreg_offset();
    validate_params(gaussian_length, poles, smoothing_length, len)?;

    let first_valid =
        first_valid_bar(high, low, close).ok_or(SmoothedGaussianTrendFilterError::AllValuesNaN)?;
    let valid = len - first_valid;
    let needed = smoothing_length.max(SUPERTREND_ATR_PERIOD);
    if valid < needed {
        return Err(SmoothedGaussianTrendFilterError::NotEnoughValidData { needed, valid });
    }

    let _chosen = normalize_kernel(kernel);
    let mut filter = alloc_with_nan_prefix(len, first_valid);
    let mut supertrend = alloc_with_nan_prefix(len, first_valid);
    let mut trend = alloc_with_nan_prefix(len, first_valid);
    let mut ranging = alloc_with_nan_prefix(len, first_valid);
    smoothed_gaussian_trend_filter_row_scalar(
        high,
        low,
        close,
        gaussian_length,
        poles,
        smoothing_length,
        linreg_offset,
        &mut filter,
        &mut supertrend,
        &mut trend,
        &mut ranging,
    );

    Ok(SmoothedGaussianTrendFilterOutput {
        filter,
        supertrend,
        trend,
        ranging,
    })
}

pub fn smoothed_gaussian_trend_filter_into_slice(
    out_filter: &mut [f64],
    out_supertrend: &mut [f64],
    out_trend: &mut [f64],
    out_ranging: &mut [f64],
    input: &SmoothedGaussianTrendFilterInput<'_>,
    kernel: Kernel,
) -> Result<(), SmoothedGaussianTrendFilterError> {
    let (high, low, close) = input.as_refs();
    validate_lengths(high, low, close)?;
    let len = close.len();
    if out_filter.len() != len
        || out_supertrend.len() != len
        || out_trend.len() != len
        || out_ranging.len() != len
    {
        return Err(SmoothedGaussianTrendFilterError::OutputLengthMismatch {
            expected: len,
            got: out_filter
                .len()
                .min(out_supertrend.len())
                .min(out_trend.len())
                .min(out_ranging.len()),
        });
    }

    let gaussian_length = input.get_gaussian_length();
    let poles = input.get_poles();
    let smoothing_length = input.get_smoothing_length();
    let linreg_offset = input.get_linreg_offset();
    validate_params(gaussian_length, poles, smoothing_length, len)?;

    let first_valid =
        first_valid_bar(high, low, close).ok_or(SmoothedGaussianTrendFilterError::AllValuesNaN)?;
    let valid = len - first_valid;
    let needed = smoothing_length.max(SUPERTREND_ATR_PERIOD);
    if valid < needed {
        return Err(SmoothedGaussianTrendFilterError::NotEnoughValidData { needed, valid });
    }

    let _chosen = normalize_kernel(kernel);
    out_filter[..first_valid].fill(f64::NAN);
    out_supertrend[..first_valid].fill(f64::NAN);
    out_trend[..first_valid].fill(f64::NAN);
    out_ranging[..first_valid].fill(f64::NAN);
    smoothed_gaussian_trend_filter_row_scalar(
        high,
        low,
        close,
        gaussian_length,
        poles,
        smoothing_length,
        linreg_offset,
        out_filter,
        out_supertrend,
        out_trend,
        out_ranging,
    );
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn smoothed_gaussian_trend_filter_into(
    input: &SmoothedGaussianTrendFilterInput<'_>,
    out_filter: &mut [f64],
    out_supertrend: &mut [f64],
    out_trend: &mut [f64],
    out_ranging: &mut [f64],
) -> Result<(), SmoothedGaussianTrendFilterError> {
    smoothed_gaussian_trend_filter_into_slice(
        out_filter,
        out_supertrend,
        out_trend,
        out_ranging,
        input,
        Kernel::Auto,
    )
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct SmoothedGaussianTrendFilterBatchRange {
    pub gaussian_length: (usize, usize, usize),
    pub poles: (usize, usize, usize),
    pub smoothing_length: (usize, usize, usize),
    pub linreg_offset: (usize, usize, usize),
}

impl Default for SmoothedGaussianTrendFilterBatchRange {
    fn default() -> Self {
        Self {
            gaussian_length: (DEFAULT_GAUSSIAN_LENGTH, DEFAULT_GAUSSIAN_LENGTH, 0),
            poles: (DEFAULT_POLES, DEFAULT_POLES, 0),
            smoothing_length: (DEFAULT_SMOOTHING_LENGTH, DEFAULT_SMOOTHING_LENGTH, 0),
            linreg_offset: (DEFAULT_LINREG_OFFSET, DEFAULT_LINREG_OFFSET, 0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SmoothedGaussianTrendFilterBatchOutput {
    pub filter: Vec<f64>,
    pub supertrend: Vec<f64>,
    pub trend: Vec<f64>,
    pub ranging: Vec<f64>,
    pub combos: Vec<SmoothedGaussianTrendFilterParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Debug)]
pub struct SmoothedGaussianTrendFilterBatchBuilder {
    range: SmoothedGaussianTrendFilterBatchRange,
    kernel: Kernel,
}

impl Default for SmoothedGaussianTrendFilterBatchBuilder {
    fn default() -> Self {
        Self {
            range: SmoothedGaussianTrendFilterBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl SmoothedGaussianTrendFilterBatchBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn gaussian_length_range(mut self, value: (usize, usize, usize)) -> Self {
        self.range.gaussian_length = value;
        self
    }

    #[inline]
    pub fn poles_range(mut self, value: (usize, usize, usize)) -> Self {
        self.range.poles = value;
        self
    }

    #[inline]
    pub fn smoothing_length_range(mut self, value: (usize, usize, usize)) -> Self {
        self.range.smoothing_length = value;
        self
    }

    #[inline]
    pub fn linreg_offset_range(mut self, value: (usize, usize, usize)) -> Self {
        self.range.linreg_offset = value;
        self
    }

    #[inline]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<SmoothedGaussianTrendFilterBatchOutput, SmoothedGaussianTrendFilterError> {
        smoothed_gaussian_trend_filter_batch_with_kernel(
            &candles.high,
            &candles.low,
            &candles.close,
            &self.range,
            self.kernel,
        )
    }
}

pub fn expand_grid_smoothed_gaussian_trend_filter(
    range: &SmoothedGaussianTrendFilterBatchRange,
) -> Result<Vec<SmoothedGaussianTrendFilterParams>, SmoothedGaussianTrendFilterError> {
    fn axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, SmoothedGaussianTrendFilterError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }

        let mut out = Vec::new();
        if start <= end {
            let mut x = start;
            while x <= end {
                out.push(x);
                x = x.saturating_add(step);
                if step == 0 {
                    break;
                }
            }
        } else {
            let mut x = start;
            while x >= end {
                out.push(x);
                let next = x.saturating_sub(step);
                if next == x {
                    break;
                }
                x = next;
                if x < end {
                    break;
                }
            }
        }

        if out.is_empty() {
            return Err(SmoothedGaussianTrendFilterError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        Ok(out)
    }

    let gaussian_lengths = axis_usize(range.gaussian_length)?;
    let poles = axis_usize(range.poles)?;
    let smoothing_lengths = axis_usize(range.smoothing_length)?;
    let linreg_offsets = axis_usize(range.linreg_offset)?;

    let cap = gaussian_lengths
        .len()
        .checked_mul(poles.len())
        .and_then(|value| value.checked_mul(smoothing_lengths.len()))
        .and_then(|value| value.checked_mul(linreg_offsets.len()))
        .ok_or(SmoothedGaussianTrendFilterError::InvalidRange {
            start: range.gaussian_length.0.to_string(),
            end: range.gaussian_length.1.to_string(),
            step: range.gaussian_length.2.to_string(),
        })?;

    let mut out = Vec::with_capacity(cap);
    for &gaussian_length in &gaussian_lengths {
        for &pole in &poles {
            for &smoothing_length in &smoothing_lengths {
                for &linreg_offset in &linreg_offsets {
                    out.push(SmoothedGaussianTrendFilterParams {
                        gaussian_length: Some(gaussian_length),
                        poles: Some(pole),
                        smoothing_length: Some(smoothing_length),
                        linreg_offset: Some(linreg_offset),
                    });
                }
            }
        }
    }
    Ok(out)
}

#[inline]
pub fn smoothed_gaussian_trend_filter_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &SmoothedGaussianTrendFilterBatchRange,
    kernel: Kernel,
) -> Result<SmoothedGaussianTrendFilterBatchOutput, SmoothedGaussianTrendFilterError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => {
            return Err(SmoothedGaussianTrendFilterError::InvalidKernelForBatch(
                other,
            ))
        }
    };
    smoothed_gaussian_trend_filter_batch_par_slice(
        high,
        low,
        close,
        sweep,
        batch_kernel.to_non_batch(),
    )
}

#[inline]
pub fn smoothed_gaussian_trend_filter_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &SmoothedGaussianTrendFilterBatchRange,
    kernel: Kernel,
) -> Result<SmoothedGaussianTrendFilterBatchOutput, SmoothedGaussianTrendFilterError> {
    smoothed_gaussian_trend_filter_batch_inner(high, low, close, sweep, kernel, false)
}

#[inline]
pub fn smoothed_gaussian_trend_filter_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &SmoothedGaussianTrendFilterBatchRange,
    kernel: Kernel,
) -> Result<SmoothedGaussianTrendFilterBatchOutput, SmoothedGaussianTrendFilterError> {
    smoothed_gaussian_trend_filter_batch_inner(high, low, close, sweep, kernel, true)
}

fn smoothed_gaussian_trend_filter_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &SmoothedGaussianTrendFilterBatchRange,
    _kernel: Kernel,
    parallel: bool,
) -> Result<SmoothedGaussianTrendFilterBatchOutput, SmoothedGaussianTrendFilterError> {
    validate_lengths(high, low, close)?;
    let combos = expand_grid_smoothed_gaussian_trend_filter(sweep)?;
    for params in &combos {
        validate_params(
            params.gaussian_length.unwrap_or(DEFAULT_GAUSSIAN_LENGTH),
            params.poles.unwrap_or(DEFAULT_POLES),
            params.smoothing_length.unwrap_or(DEFAULT_SMOOTHING_LENGTH),
            close.len(),
        )?;
    }

    let first_valid =
        first_valid_bar(high, low, close).ok_or(SmoothedGaussianTrendFilterError::AllValuesNaN)?;
    let rows = combos.len();
    let cols = close.len();
    let total =
        rows.checked_mul(cols)
            .ok_or(SmoothedGaussianTrendFilterError::OutputLengthMismatch {
                expected: usize::MAX,
                got: 0,
            })?;

    let mut filter_matrix = make_uninit_matrix(rows, cols);
    let mut supertrend_matrix = make_uninit_matrix(rows, cols);
    let mut trend_matrix = make_uninit_matrix(rows, cols);
    let mut ranging_matrix = make_uninit_matrix(rows, cols);
    let warmups = vec![first_valid; rows];
    init_matrix_prefixes(&mut filter_matrix, cols, &warmups);
    init_matrix_prefixes(&mut supertrend_matrix, cols, &warmups);
    init_matrix_prefixes(&mut trend_matrix, cols, &warmups);
    init_matrix_prefixes(&mut ranging_matrix, cols, &warmups);

    let mut filter_guard = ManuallyDrop::new(filter_matrix);
    let mut supertrend_guard = ManuallyDrop::new(supertrend_matrix);
    let mut trend_guard = ManuallyDrop::new(trend_matrix);
    let mut ranging_guard = ManuallyDrop::new(ranging_matrix);

    let filter_mu: &mut [MaybeUninit<f64>] =
        unsafe { std::slice::from_raw_parts_mut(filter_guard.as_mut_ptr(), filter_guard.len()) };
    let supertrend_mu: &mut [MaybeUninit<f64>] = unsafe {
        std::slice::from_raw_parts_mut(supertrend_guard.as_mut_ptr(), supertrend_guard.len())
    };
    let trend_mu: &mut [MaybeUninit<f64>] =
        unsafe { std::slice::from_raw_parts_mut(trend_guard.as_mut_ptr(), trend_guard.len()) };
    let ranging_mu: &mut [MaybeUninit<f64>] =
        unsafe { std::slice::from_raw_parts_mut(ranging_guard.as_mut_ptr(), ranging_guard.len()) };

    let do_row = |row: usize,
                  row_filter: &mut [MaybeUninit<f64>],
                  row_supertrend: &mut [MaybeUninit<f64>],
                  row_trend: &mut [MaybeUninit<f64>],
                  row_ranging: &mut [MaybeUninit<f64>]| {
        let params = &combos[row];
        let dst_filter =
            unsafe { std::slice::from_raw_parts_mut(row_filter.as_mut_ptr() as *mut f64, cols) };
        let dst_supertrend = unsafe {
            std::slice::from_raw_parts_mut(row_supertrend.as_mut_ptr() as *mut f64, cols)
        };
        let dst_trend =
            unsafe { std::slice::from_raw_parts_mut(row_trend.as_mut_ptr() as *mut f64, cols) };
        let dst_ranging =
            unsafe { std::slice::from_raw_parts_mut(row_ranging.as_mut_ptr() as *mut f64, cols) };
        smoothed_gaussian_trend_filter_row_scalar(
            high,
            low,
            close,
            params.gaussian_length.unwrap_or(DEFAULT_GAUSSIAN_LENGTH),
            params.poles.unwrap_or(DEFAULT_POLES),
            params.smoothing_length.unwrap_or(DEFAULT_SMOOTHING_LENGTH),
            params.linreg_offset.unwrap_or(DEFAULT_LINREG_OFFSET),
            dst_filter,
            dst_supertrend,
            dst_trend,
            dst_ranging,
        );
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        filter_mu
            .par_chunks_mut(cols)
            .zip(supertrend_mu.par_chunks_mut(cols))
            .zip(trend_mu.par_chunks_mut(cols))
            .zip(ranging_mu.par_chunks_mut(cols))
            .enumerate()
            .for_each(
                |(row, (((row_filter, row_supertrend), row_trend), row_ranging))| {
                    do_row(row, row_filter, row_supertrend, row_trend, row_ranging)
                },
            );

        #[cfg(target_arch = "wasm32")]
        for (row, (((row_filter, row_supertrend), row_trend), row_ranging)) in filter_mu
            .chunks_mut(cols)
            .zip(supertrend_mu.chunks_mut(cols))
            .zip(trend_mu.chunks_mut(cols))
            .zip(ranging_mu.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, row_filter, row_supertrend, row_trend, row_ranging);
        }
    } else {
        for (row, (((row_filter, row_supertrend), row_trend), row_ranging)) in filter_mu
            .chunks_mut(cols)
            .zip(supertrend_mu.chunks_mut(cols))
            .zip(trend_mu.chunks_mut(cols))
            .zip(ranging_mu.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, row_filter, row_supertrend, row_trend, row_ranging);
        }
    }

    let filter = unsafe {
        Vec::from_raw_parts(
            filter_guard.as_mut_ptr() as *mut f64,
            total,
            filter_guard.capacity(),
        )
    };
    let supertrend = unsafe {
        Vec::from_raw_parts(
            supertrend_guard.as_mut_ptr() as *mut f64,
            total,
            supertrend_guard.capacity(),
        )
    };
    let trend = unsafe {
        Vec::from_raw_parts(
            trend_guard.as_mut_ptr() as *mut f64,
            total,
            trend_guard.capacity(),
        )
    };
    let ranging = unsafe {
        Vec::from_raw_parts(
            ranging_guard.as_mut_ptr() as *mut f64,
            total,
            ranging_guard.capacity(),
        )
    };

    Ok(SmoothedGaussianTrendFilterBatchOutput {
        filter,
        supertrend,
        trend,
        ranging,
        combos,
        rows,
        cols,
    })
}

fn smoothed_gaussian_trend_filter_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &SmoothedGaussianTrendFilterBatchRange,
    kernel: Kernel,
    parallel: bool,
    out_filter: &mut [f64],
    out_supertrend: &mut [f64],
    out_trend: &mut [f64],
    out_ranging: &mut [f64],
) -> Result<Vec<SmoothedGaussianTrendFilterParams>, SmoothedGaussianTrendFilterError> {
    validate_lengths(high, low, close)?;
    let combos = expand_grid_smoothed_gaussian_trend_filter(sweep)?;
    for params in &combos {
        validate_params(
            params.gaussian_length.unwrap_or(DEFAULT_GAUSSIAN_LENGTH),
            params.poles.unwrap_or(DEFAULT_POLES),
            params.smoothing_length.unwrap_or(DEFAULT_SMOOTHING_LENGTH),
            close.len(),
        )?;
    }

    let rows = combos.len();
    let cols = close.len();
    let total =
        rows.checked_mul(cols)
            .ok_or(SmoothedGaussianTrendFilterError::OutputLengthMismatch {
                expected: usize::MAX,
                got: 0,
            })?;
    if out_filter.len() != total
        || out_supertrend.len() != total
        || out_trend.len() != total
        || out_ranging.len() != total
    {
        return Err(SmoothedGaussianTrendFilterError::OutputLengthMismatch {
            expected: total,
            got: out_filter
                .len()
                .min(out_supertrend.len())
                .min(out_trend.len())
                .min(out_ranging.len()),
        });
    }

    let _kernel = kernel;
    let do_row = |row: usize,
                  dst_filter: &mut [f64],
                  dst_supertrend: &mut [f64],
                  dst_trend: &mut [f64],
                  dst_ranging: &mut [f64]| {
        let params = &combos[row];
        smoothed_gaussian_trend_filter_row_scalar(
            high,
            low,
            close,
            params.gaussian_length.unwrap_or(DEFAULT_GAUSSIAN_LENGTH),
            params.poles.unwrap_or(DEFAULT_POLES),
            params.smoothing_length.unwrap_or(DEFAULT_SMOOTHING_LENGTH),
            params.linreg_offset.unwrap_or(DEFAULT_LINREG_OFFSET),
            dst_filter,
            dst_supertrend,
            dst_trend,
            dst_ranging,
        );
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out_filter
            .par_chunks_mut(cols)
            .zip(out_supertrend.par_chunks_mut(cols))
            .zip(out_trend.par_chunks_mut(cols))
            .zip(out_ranging.par_chunks_mut(cols))
            .enumerate()
            .for_each(
                |(row, (((dst_filter, dst_supertrend), dst_trend), dst_ranging))| {
                    do_row(row, dst_filter, dst_supertrend, dst_trend, dst_ranging)
                },
            );

        #[cfg(target_arch = "wasm32")]
        for (row, (((dst_filter, dst_supertrend), dst_trend), dst_ranging)) in out_filter
            .chunks_mut(cols)
            .zip(out_supertrend.chunks_mut(cols))
            .zip(out_trend.chunks_mut(cols))
            .zip(out_ranging.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, dst_filter, dst_supertrend, dst_trend, dst_ranging);
        }
    } else {
        for (row, (((dst_filter, dst_supertrend), dst_trend), dst_ranging)) in out_filter
            .chunks_mut(cols)
            .zip(out_supertrend.chunks_mut(cols))
            .zip(out_trend.chunks_mut(cols))
            .zip(out_ranging.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, dst_filter, dst_supertrend, dst_trend, dst_ranging);
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "smoothed_gaussian_trend_filter")]
#[pyo3(signature = (high, low, close, gaussian_length=DEFAULT_GAUSSIAN_LENGTH, poles=DEFAULT_POLES, smoothing_length=DEFAULT_SMOOTHING_LENGTH, linreg_offset=DEFAULT_LINREG_OFFSET, kernel=None))]
pub fn smoothed_gaussian_trend_filter_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    gaussian_length: usize,
    poles: usize,
    smoothing_length: usize,
    linreg_offset: usize,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let input = SmoothedGaussianTrendFilterInput::from_slices(
        high,
        low,
        close,
        SmoothedGaussianTrendFilterParams {
            gaussian_length: Some(gaussian_length),
            poles: Some(poles),
            smoothing_length: Some(smoothing_length),
            linreg_offset: Some(linreg_offset),
        },
    );
    let kernel = validate_kernel(kernel, false)?;
    let out = py
        .allow_threads(|| smoothed_gaussian_trend_filter_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        out.filter.into_pyarray(py),
        out.supertrend.into_pyarray(py),
        out.trend.into_pyarray(py),
        out.ranging.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyclass(name = "SmoothedGaussianTrendFilterStream")]
pub struct SmoothedGaussianTrendFilterStreamPy {
    stream: SmoothedGaussianTrendFilterStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl SmoothedGaussianTrendFilterStreamPy {
    #[new]
    #[pyo3(signature = (gaussian_length=DEFAULT_GAUSSIAN_LENGTH, poles=DEFAULT_POLES, smoothing_length=DEFAULT_SMOOTHING_LENGTH, linreg_offset=DEFAULT_LINREG_OFFSET))]
    fn new(
        gaussian_length: usize,
        poles: usize,
        smoothing_length: usize,
        linreg_offset: usize,
    ) -> PyResult<Self> {
        let stream =
            SmoothedGaussianTrendFilterStream::try_new(SmoothedGaussianTrendFilterParams {
                gaussian_length: Some(gaussian_length),
                poles: Some(poles),
                smoothing_length: Some(smoothing_length),
                linreg_offset: Some(linreg_offset),
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64, f64, f64)> {
        self.stream.update(high, low, close)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "smoothed_gaussian_trend_filter_batch")]
#[pyo3(signature = (high, low, close, gaussian_length_range, poles_range, smoothing_length_range, linreg_offset_range, kernel=None))]
pub fn smoothed_gaussian_trend_filter_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    gaussian_length_range: (usize, usize, usize),
    poles_range: (usize, usize, usize),
    smoothing_length_range: (usize, usize, usize),
    linreg_offset_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let sweep = SmoothedGaussianTrendFilterBatchRange {
        gaussian_length: gaussian_length_range,
        poles: poles_range,
        smoothing_length: smoothing_length_range,
        linreg_offset: linreg_offset_range,
    };
    let combos = expand_grid_smoothed_gaussian_trend_filter(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let filter_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let supertrend_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let trend_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let ranging_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_filter = unsafe { filter_arr.as_slice_mut()? };
    let out_supertrend = unsafe { supertrend_arr.as_slice_mut()? };
    let out_trend = unsafe { trend_arr.as_slice_mut()? };
    let out_ranging = unsafe { ranging_arr.as_slice_mut()? };
    let kernel = validate_kernel(kernel, true)?;

    py.allow_threads(|| {
        let batch_kernel = match kernel {
            Kernel::Auto => detect_best_batch_kernel(),
            other => other,
        };
        smoothed_gaussian_trend_filter_batch_inner_into(
            high,
            low,
            close,
            &sweep,
            batch_kernel.to_non_batch(),
            true,
            out_filter,
            out_supertrend,
            out_trend,
            out_ranging,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let gaussian_lengths: Vec<usize> = combos
        .iter()
        .map(|params| params.gaussian_length.unwrap_or(DEFAULT_GAUSSIAN_LENGTH))
        .collect();
    let poles: Vec<usize> = combos
        .iter()
        .map(|params| params.poles.unwrap_or(DEFAULT_POLES))
        .collect();
    let smoothing_lengths: Vec<usize> = combos
        .iter()
        .map(|params| params.smoothing_length.unwrap_or(DEFAULT_SMOOTHING_LENGTH))
        .collect();
    let linreg_offsets: Vec<usize> = combos
        .iter()
        .map(|params| params.linreg_offset.unwrap_or(DEFAULT_LINREG_OFFSET))
        .collect();

    let dict = PyDict::new(py);
    dict.set_item("filter", filter_arr.reshape((rows, cols))?)?;
    dict.set_item("supertrend", supertrend_arr.reshape((rows, cols))?)?;
    dict.set_item("trend", trend_arr.reshape((rows, cols))?)?;
    dict.set_item("ranging", ranging_arr.reshape((rows, cols))?)?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    dict.set_item("gaussian_lengths", gaussian_lengths.into_pyarray(py))?;
    dict.set_item("poles", poles.into_pyarray(py))?;
    dict.set_item("smoothing_lengths", smoothing_lengths.into_pyarray(py))?;
    dict.set_item("linreg_offsets", linreg_offsets.into_pyarray(py))?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_smoothed_gaussian_trend_filter_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(smoothed_gaussian_trend_filter_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        smoothed_gaussian_trend_filter_batch_py,
        m
    )?)?;
    m.add_class::<SmoothedGaussianTrendFilterStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SmoothedGaussianTrendFilterWasmOutput {
    filter: Vec<f64>,
    supertrend: Vec<f64>,
    trend: Vec<f64>,
    ranging: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "smoothed_gaussian_trend_filter")]
pub fn smoothed_gaussian_trend_filter_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    gaussian_length: usize,
    poles: usize,
    smoothing_length: usize,
    linreg_offset: usize,
) -> Result<JsValue, JsValue> {
    let input = SmoothedGaussianTrendFilterInput::from_slices(
        high,
        low,
        close,
        SmoothedGaussianTrendFilterParams {
            gaussian_length: Some(gaussian_length),
            poles: Some(poles),
            smoothing_length: Some(smoothing_length),
            linreg_offset: Some(linreg_offset),
        },
    );
    let out = smoothed_gaussian_trend_filter_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&SmoothedGaussianTrendFilterWasmOutput {
        filter: out.filter,
        supertrend: out.supertrend,
        trend: out.trend,
        ranging: out.ranging,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn smoothed_gaussian_trend_filter_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    gaussian_length: usize,
    poles: usize,
    smoothing_length: usize,
    linreg_offset: usize,
) -> Result<(), JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to smoothed_gaussian_trend_filter_into",
        ));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len * 4);
        let (out_filter, rest) = out.split_at_mut(len);
        let (out_supertrend, rest) = rest.split_at_mut(len);
        let (out_trend, out_ranging) = rest.split_at_mut(len);
        let input = SmoothedGaussianTrendFilterInput::from_slices(
            high,
            low,
            close,
            SmoothedGaussianTrendFilterParams {
                gaussian_length: Some(gaussian_length),
                poles: Some(poles),
                smoothing_length: Some(smoothing_length),
                linreg_offset: Some(linreg_offset),
            },
        );
        smoothed_gaussian_trend_filter_into_slice(
            out_filter,
            out_supertrend,
            out_trend,
            out_ranging,
            &input,
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "smoothed_gaussian_trend_filter_into_host")]
pub fn smoothed_gaussian_trend_filter_into_host(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    out_ptr: *mut f64,
    gaussian_length: usize,
    poles: usize,
    smoothing_length: usize,
    linreg_offset: usize,
) -> Result<(), JsValue> {
    if out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to smoothed_gaussian_trend_filter_into_host",
        ));
    }

    unsafe {
        let out = std::slice::from_raw_parts_mut(out_ptr, close.len() * 4);
        let (out_filter, rest) = out.split_at_mut(close.len());
        let (out_supertrend, rest) = rest.split_at_mut(close.len());
        let (out_trend, out_ranging) = rest.split_at_mut(close.len());
        let input = SmoothedGaussianTrendFilterInput::from_slices(
            high,
            low,
            close,
            SmoothedGaussianTrendFilterParams {
                gaussian_length: Some(gaussian_length),
                poles: Some(poles),
                smoothing_length: Some(smoothing_length),
                linreg_offset: Some(linreg_offset),
            },
        );
        smoothed_gaussian_trend_filter_into_slice(
            out_filter,
            out_supertrend,
            out_trend,
            out_ranging,
            &input,
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn smoothed_gaussian_trend_filter_alloc(len: usize) -> *mut f64 {
    let mut buf = vec![0.0_f64; len * 4];
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn smoothed_gaussian_trend_filter_free(ptr: *mut f64, len: usize) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let _ = Vec::from_raw_parts(ptr, 0, len * 4);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SmoothedGaussianTrendFilterBatchConfig {
    gaussian_length_range: Vec<usize>,
    poles_range: Vec<usize>,
    smoothing_length_range: Vec<usize>,
    linreg_offset_range: Vec<usize>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SmoothedGaussianTrendFilterBatchOutputWasm {
    filter: Vec<f64>,
    supertrend: Vec<f64>,
    trend: Vec<f64>,
    ranging: Vec<f64>,
    rows: usize,
    cols: usize,
    combos: Vec<SmoothedGaussianTrendFilterParams>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "smoothed_gaussian_trend_filter_batch")]
pub fn smoothed_gaussian_trend_filter_batch_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: SmoothedGaussianTrendFilterBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.gaussian_length_range.len() != 3
        || config.poles_range.len() != 3
        || config.smoothing_length_range.len() != 3
        || config.linreg_offset_range.len() != 3
    {
        return Err(JsValue::from_str(
            "Invalid config: ranges must have exactly 3 elements [start, end, step]",
        ));
    }

    let sweep = SmoothedGaussianTrendFilterBatchRange {
        gaussian_length: (
            config.gaussian_length_range[0],
            config.gaussian_length_range[1],
            config.gaussian_length_range[2],
        ),
        poles: (
            config.poles_range[0],
            config.poles_range[1],
            config.poles_range[2],
        ),
        smoothing_length: (
            config.smoothing_length_range[0],
            config.smoothing_length_range[1],
            config.smoothing_length_range[2],
        ),
        linreg_offset: (
            config.linreg_offset_range[0],
            config.linreg_offset_range[1],
            config.linreg_offset_range[2],
        ),
    };

    let out =
        smoothed_gaussian_trend_filter_batch_inner(high, low, close, &sweep, Kernel::Auto, false)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&SmoothedGaussianTrendFilterBatchOutputWasm {
        filter: out.filter,
        supertrend: out.supertrend,
        trend: out.trend,
        ranging: out.ranging,
        rows: out.rows,
        cols: out.cols,
        combos: out.combos,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn smoothed_gaussian_trend_filter_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    filter_ptr: *mut f64,
    supertrend_ptr: *mut f64,
    trend_ptr: *mut f64,
    ranging_ptr: *mut f64,
    len: usize,
    gaussian_length_start: usize,
    gaussian_length_end: usize,
    gaussian_length_step: usize,
    poles_start: usize,
    poles_end: usize,
    poles_step: usize,
    smoothing_length_start: usize,
    smoothing_length_end: usize,
    smoothing_length_step: usize,
    linreg_offset_start: usize,
    linreg_offset_end: usize,
    linreg_offset_step: usize,
) -> Result<usize, JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || filter_ptr.is_null()
        || supertrend_ptr.is_null()
        || trend_ptr.is_null()
        || ranging_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to smoothed_gaussian_trend_filter_batch_into",
        ));
    }

    let sweep = SmoothedGaussianTrendFilterBatchRange {
        gaussian_length: (
            gaussian_length_start,
            gaussian_length_end,
            gaussian_length_step,
        ),
        poles: (poles_start, poles_end, poles_step),
        smoothing_length: (
            smoothing_length_start,
            smoothing_length_end,
            smoothing_length_step,
        ),
        linreg_offset: (linreg_offset_start, linreg_offset_end, linreg_offset_step),
    };
    let combos = expand_grid_smoothed_gaussian_trend_filter(&sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let total = rows
        .checked_mul(len)
        .ok_or_else(|| JsValue::from_str("rows*len overflow"))?;

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let out_filter = std::slice::from_raw_parts_mut(filter_ptr, total);
        let out_supertrend = std::slice::from_raw_parts_mut(supertrend_ptr, total);
        let out_trend = std::slice::from_raw_parts_mut(trend_ptr, total);
        let out_ranging = std::slice::from_raw_parts_mut(ranging_ptr, total);
        smoothed_gaussian_trend_filter_batch_inner_into(
            high,
            low,
            close,
            &sweep,
            Kernel::Auto,
            false,
            out_filter,
            out_supertrend,
            out_trend,
            out_ranging,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }

    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn smoothed_gaussian_trend_filter_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    gaussian_length: usize,
    poles: usize,
    smoothing_length: usize,
    linreg_offset: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = smoothed_gaussian_trend_filter_js(
        high,
        low,
        close,
        gaussian_length,
        poles,
        smoothing_length,
        linreg_offset,
    )?;
    crate::write_wasm_object_f64_outputs(
        "smoothed_gaussian_trend_filter_output_into_js",
        &value,
        out,
    )
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn smoothed_gaussian_trend_filter_batch_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = smoothed_gaussian_trend_filter_batch_js(high, low, close, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "smoothed_gaussian_trend_filter_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::dispatch::cpu_batch::compute_cpu_batch;
    use crate::indicators::dispatch::{
        IndicatorBatchRequest, IndicatorDataRef, IndicatorParamSet, ParamKV, ParamValue,
    };
    use crate::utilities::data_loader::read_candles_from_csv;

    fn load_candles() -> Candles {
        read_candles_from_csv("src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv").expect("candles")
    }

    fn assert_series_eq(lhs: &[f64], rhs: &[f64]) {
        assert_eq!(lhs.len(), rhs.len());
        for (idx, (&a, &b)) in lhs.iter().zip(rhs.iter()).enumerate() {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert!(
                (a - b).abs() <= 1.0e-10,
                "mismatch at index {idx}: left={a}, right={b}"
            );
        }
    }

    #[test]
    fn smoothed_gaussian_trend_filter_invalid_params() {
        let candles = load_candles();
        let input = SmoothedGaussianTrendFilterInput::from_candles(
            &candles,
            SmoothedGaussianTrendFilterParams {
                gaussian_length: Some(0),
                poles: Some(3),
                smoothing_length: Some(22),
                linreg_offset: Some(7),
            },
        );
        assert!(matches!(
            smoothed_gaussian_trend_filter(&input),
            Err(SmoothedGaussianTrendFilterError::InvalidGaussianLength { .. })
        ));

        let input = SmoothedGaussianTrendFilterInput::from_candles(
            &candles,
            SmoothedGaussianTrendFilterParams {
                gaussian_length: Some(15),
                poles: Some(5),
                smoothing_length: Some(22),
                linreg_offset: Some(7),
            },
        );
        assert!(matches!(
            smoothed_gaussian_trend_filter(&input),
            Err(SmoothedGaussianTrendFilterError::InvalidPoles { .. })
        ));
    }

    #[test]
    fn smoothed_gaussian_trend_filter_output_contract() -> Result<(), Box<dyn std::error::Error>> {
        let candles = load_candles();
        let input = SmoothedGaussianTrendFilterInput::with_default_candles(&candles);
        let out = smoothed_gaussian_trend_filter(&input)?;
        assert_eq!(out.filter.len(), candles.close.len());
        assert_eq!(out.supertrend.len(), candles.close.len());
        assert_eq!(out.trend.len(), candles.close.len());
        assert_eq!(out.ranging.len(), candles.close.len());
        assert!(out.filter.iter().any(|v| v.is_finite()));
        assert!(out.supertrend.iter().any(|v| v.is_finite()));
        assert!(out.trend.iter().any(|v| v.is_finite()));
        assert!(out.ranging.iter().any(|v| v.is_finite()));
        Ok(())
    }

    #[test]
    fn smoothed_gaussian_trend_filter_into_matches_api() -> Result<(), Box<dyn std::error::Error>> {
        let candles = load_candles();
        let input = SmoothedGaussianTrendFilterInput::with_default_candles(&candles);
        let base = smoothed_gaussian_trend_filter(&input)?;
        let len = candles.close.len();
        let mut filter = vec![f64::NAN; len];
        let mut supertrend = vec![f64::NAN; len];
        let mut trend = vec![f64::NAN; len];
        let mut ranging = vec![f64::NAN; len];
        smoothed_gaussian_trend_filter_into(
            &input,
            &mut filter,
            &mut supertrend,
            &mut trend,
            &mut ranging,
        )?;
        assert_series_eq(&base.filter, &filter);
        assert_series_eq(&base.supertrend, &supertrend);
        assert_series_eq(&base.trend, &trend);
        assert_series_eq(&base.ranging, &ranging);
        Ok(())
    }

    #[test]
    fn smoothed_gaussian_trend_filter_stream_matches_batch(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let candles = load_candles();
        let input = SmoothedGaussianTrendFilterInput::with_default_candles(&candles);
        let batch = smoothed_gaussian_trend_filter(&input)?;
        let mut stream = SmoothedGaussianTrendFilterStream::try_new(
            SmoothedGaussianTrendFilterParams::default(),
        )?;
        let mut filter = Vec::with_capacity(candles.close.len());
        let mut supertrend = Vec::with_capacity(candles.close.len());
        let mut trend = Vec::with_capacity(candles.close.len());
        let mut ranging = Vec::with_capacity(candles.close.len());

        for i in 0..candles.close.len() {
            match stream.update(candles.high[i], candles.low[i], candles.close[i]) {
                Some((f, s, t, r)) => {
                    filter.push(f);
                    supertrend.push(s);
                    trend.push(t);
                    ranging.push(r);
                }
                None => {
                    filter.push(f64::NAN);
                    supertrend.push(f64::NAN);
                    trend.push(f64::NAN);
                    ranging.push(f64::NAN);
                }
            }
        }

        assert_series_eq(&batch.filter, &filter);
        assert_series_eq(&batch.supertrend, &supertrend);
        assert_series_eq(&batch.trend, &trend);
        assert_series_eq(&batch.ranging, &ranging);
        Ok(())
    }

    #[test]
    fn smoothed_gaussian_trend_filter_batch_single_matches_single(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let candles = load_candles();
        let batch = smoothed_gaussian_trend_filter_batch_with_kernel(
            &candles.high,
            &candles.low,
            &candles.close,
            &SmoothedGaussianTrendFilterBatchRange::default(),
            Kernel::ScalarBatch,
        )?;
        let single = smoothed_gaussian_trend_filter(
            &SmoothedGaussianTrendFilterInput::with_default_candles(&candles),
        )?;
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, candles.close.len());
        assert_series_eq(&batch.filter, &single.filter);
        assert_series_eq(&batch.supertrend, &single.supertrend);
        assert_series_eq(&batch.trend, &single.trend);
        assert_series_eq(&batch.ranging, &single.ranging);
        Ok(())
    }

    #[test]
    fn smoothed_gaussian_trend_filter_dispatch_matches_direct(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let candles = load_candles();
        let direct = smoothed_gaussian_trend_filter(
            &SmoothedGaussianTrendFilterInput::with_default_candles(&candles),
        )?;
        let req = IndicatorBatchRequest {
            indicator_id: "smoothed_gaussian_trend_filter",
            output_id: Some("filter"),
            data: IndicatorDataRef::Candles {
                candles: &candles,
                source: None,
            },
            combos: &[IndicatorParamSet {
                params: &[
                    ParamKV {
                        key: "gaussian_length",
                        value: ParamValue::Int(15),
                    },
                    ParamKV {
                        key: "poles",
                        value: ParamValue::Int(3),
                    },
                    ParamKV {
                        key: "smoothing_length",
                        value: ParamValue::Int(22),
                    },
                    ParamKV {
                        key: "linreg_offset",
                        value: ParamValue::Int(7),
                    },
                ],
            }],
            kernel: Kernel::ScalarBatch,
        };
        let out = compute_cpu_batch(req)?;
        assert_eq!(out.rows, 1);
        assert_eq!(out.cols, candles.close.len());
        assert_series_eq(&out.values_f64.unwrap(), &direct.filter);
        Ok(())
    }
}
