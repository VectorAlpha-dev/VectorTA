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

use crate::indicators::dispatch::{
    compute_cpu_batch, IndicatorBatchRequest, IndicatorDataRef, IndicatorParamSet, ParamKV,
    ParamValue,
};
use crate::indicators::moving_averages::ema::{EmaParams, EmaStream};
use crate::indicators::moving_averages::hma::{HmaParams, HmaStream};
use crate::indicators::moving_averages::sma::{SmaParams, SmaStream};
use crate::indicators::stddev::{StdDevParams, StdDevStream};
use crate::utilities::data_loader::Candles;
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::collections::VecDeque;
use std::str::FromStr;
use thiserror::Error;

const DEFAULT_MA_TYPE: MovingAverageCrossProbabilityMaType =
    MovingAverageCrossProbabilityMaType::Ema;
const DEFAULT_SMOOTHING_WINDOW: usize = 7;
const DEFAULT_SLOW_LENGTH: usize = 30;
const DEFAULT_FAST_LENGTH: usize = 14;
const DEFAULT_RESOLUTION: usize = 50;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize),
    serde(rename_all = "snake_case")
)]
pub enum MovingAverageCrossProbabilityMaType {
    Ema,
    Sma,
}

impl Default for MovingAverageCrossProbabilityMaType {
    fn default() -> Self {
        DEFAULT_MA_TYPE
    }
}

impl MovingAverageCrossProbabilityMaType {
    #[inline(always)]
    fn as_str(self) -> &'static str {
        match self {
            Self::Ema => "ema",
            Self::Sma => "sma",
        }
    }
}

impl FromStr for MovingAverageCrossProbabilityMaType {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "ema" => Ok(Self::Ema),
            "sma" => Ok(Self::Sma),
            _ => Err(format!("invalid ma_type: {value}")),
        }
    }
}

#[derive(Debug, Clone)]
pub enum MovingAverageCrossProbabilityData<'a> {
    Candles { candles: &'a Candles },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct MovingAverageCrossProbabilityOutput {
    pub value: Vec<f64>,
    pub slow_ma: Vec<f64>,
    pub fast_ma: Vec<f64>,
    pub forecast: Vec<f64>,
    pub upper: Vec<f64>,
    pub lower: Vec<f64>,
    pub direction: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct MovingAverageCrossProbabilityParams {
    pub ma_type: Option<MovingAverageCrossProbabilityMaType>,
    pub smoothing_window: Option<usize>,
    pub slow_length: Option<usize>,
    pub fast_length: Option<usize>,
    pub resolution: Option<usize>,
}

impl Default for MovingAverageCrossProbabilityParams {
    fn default() -> Self {
        Self {
            ma_type: Some(DEFAULT_MA_TYPE),
            smoothing_window: Some(DEFAULT_SMOOTHING_WINDOW),
            slow_length: Some(DEFAULT_SLOW_LENGTH),
            fast_length: Some(DEFAULT_FAST_LENGTH),
            resolution: Some(DEFAULT_RESOLUTION),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MovingAverageCrossProbabilityInput<'a> {
    pub data: MovingAverageCrossProbabilityData<'a>,
    pub params: MovingAverageCrossProbabilityParams,
}

impl<'a> MovingAverageCrossProbabilityInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: MovingAverageCrossProbabilityParams) -> Self {
        Self {
            data: MovingAverageCrossProbabilityData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slice(data: &'a [f64], params: MovingAverageCrossProbabilityParams) -> Self {
        Self {
            data: MovingAverageCrossProbabilityData::Slice(data),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, MovingAverageCrossProbabilityParams::default())
    }
}

#[derive(Copy, Clone, Debug)]
pub struct MovingAverageCrossProbabilityBuilder {
    ma_type: Option<MovingAverageCrossProbabilityMaType>,
    smoothing_window: Option<usize>,
    slow_length: Option<usize>,
    fast_length: Option<usize>,
    resolution: Option<usize>,
    kernel: Kernel,
}

impl Default for MovingAverageCrossProbabilityBuilder {
    fn default() -> Self {
        Self {
            ma_type: None,
            smoothing_window: None,
            slow_length: None,
            fast_length: None,
            resolution: None,
            kernel: Kernel::Auto,
        }
    }
}

impl MovingAverageCrossProbabilityBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn ma_type(mut self, value: MovingAverageCrossProbabilityMaType) -> Self {
        self.ma_type = Some(value);
        self
    }

    #[inline(always)]
    pub fn smoothing_window(mut self, value: usize) -> Self {
        self.smoothing_window = Some(value);
        self
    }

    #[inline(always)]
    pub fn slow_length(mut self, value: usize) -> Self {
        self.slow_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn fast_length(mut self, value: usize) -> Self {
        self.fast_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn resolution(mut self, value: usize) -> Self {
        self.resolution = Some(value);
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
    ) -> Result<MovingAverageCrossProbabilityOutput, MovingAverageCrossProbabilityError> {
        let input = MovingAverageCrossProbabilityInput::from_candles(
            candles,
            MovingAverageCrossProbabilityParams {
                ma_type: self.ma_type,
                smoothing_window: self.smoothing_window,
                slow_length: self.slow_length,
                fast_length: self.fast_length,
                resolution: self.resolution,
            },
        );
        moving_average_cross_probability_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<MovingAverageCrossProbabilityOutput, MovingAverageCrossProbabilityError> {
        let input = MovingAverageCrossProbabilityInput::from_slice(
            data,
            MovingAverageCrossProbabilityParams {
                ma_type: self.ma_type,
                smoothing_window: self.smoothing_window,
                slow_length: self.slow_length,
                fast_length: self.fast_length,
                resolution: self.resolution,
            },
        );
        moving_average_cross_probability_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<MovingAverageCrossProbabilityStream, MovingAverageCrossProbabilityError> {
        MovingAverageCrossProbabilityStream::try_new(MovingAverageCrossProbabilityParams {
            ma_type: self.ma_type,
            smoothing_window: self.smoothing_window,
            slow_length: self.slow_length,
            fast_length: self.fast_length,
            resolution: self.resolution,
        })
    }
}

#[derive(Debug, Error)]
pub enum MovingAverageCrossProbabilityError {
    #[error("moving_average_cross_probability: Input data slice is empty.")]
    EmptyInputData,
    #[error("moving_average_cross_probability: All values are NaN.")]
    AllValuesNaN,
    #[error("moving_average_cross_probability: Invalid smoothing_window: {smoothing_window}")]
    InvalidSmoothingWindow { smoothing_window: usize },
    #[error("moving_average_cross_probability: Invalid slow_length: {slow_length}")]
    InvalidSlowLength { slow_length: usize },
    #[error("moving_average_cross_probability: Invalid fast_length: {fast_length}")]
    InvalidFastLength { fast_length: usize },
    #[error("moving_average_cross_probability: Invalid resolution: {resolution}")]
    InvalidResolution { resolution: usize },
    #[error("moving_average_cross_probability: Invalid length order: fast_length={fast_length}, slow_length={slow_length}")]
    InvalidLengthOrder {
        fast_length: usize,
        slow_length: usize,
    },
    #[error(
        "moving_average_cross_probability: Output length mismatch: expected={expected}, got={got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error(
        "moving_average_cross_probability: Invalid range: start={start}, end={end}, step={step}"
    )]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("moving_average_cross_probability: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Debug, Clone)]
struct ResolvedParams {
    ma_type: MovingAverageCrossProbabilityMaType,
    smoothing_window: usize,
    slow_length: usize,
    fast_length: usize,
    resolution: usize,
    history_window_len: usize,
    slow_alpha: f64,
    slow_beta: f64,
    fast_alpha: f64,
    fast_beta: f64,
    slow_ma_warmup: usize,
    fast_ma_warmup: usize,
    direction_warmup: usize,
    forecast_warmup: usize,
    probability_warmup: usize,
}

#[derive(Debug, Clone)]
enum CurrentMaStream {
    Ema(EmaStream),
    Sma(SmaStream),
}

impl CurrentMaStream {
    #[inline(always)]
    fn try_new(
        ma_type: MovingAverageCrossProbabilityMaType,
        period: usize,
    ) -> Result<Self, MovingAverageCrossProbabilityError> {
        match ma_type {
            MovingAverageCrossProbabilityMaType::Ema => Ok(Self::Ema(
                EmaStream::try_new(EmaParams {
                    period: Some(period),
                })
                .map_err(|_| {
                    MovingAverageCrossProbabilityError::InvalidSlowLength {
                        slow_length: period,
                    }
                })?,
            )),
            MovingAverageCrossProbabilityMaType::Sma => Ok(Self::Sma(
                SmaStream::try_new(SmaParams {
                    period: Some(period),
                })
                .map_err(|_| {
                    MovingAverageCrossProbabilityError::InvalidSlowLength {
                        slow_length: period,
                    }
                })?,
            )),
        }
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<f64> {
        match self {
            Self::Ema(stream) => stream.update(value),
            Self::Sma(stream) => stream.update(value),
        }
    }
}

#[inline(always)]
fn resolve_params(
    params: &MovingAverageCrossProbabilityParams,
) -> Result<ResolvedParams, MovingAverageCrossProbabilityError> {
    let ma_type = params.ma_type.unwrap_or(DEFAULT_MA_TYPE);
    let smoothing_window = params.smoothing_window.unwrap_or(DEFAULT_SMOOTHING_WINDOW);
    let slow_length = params.slow_length.unwrap_or(DEFAULT_SLOW_LENGTH);
    let fast_length = params.fast_length.unwrap_or(DEFAULT_FAST_LENGTH);
    let resolution = params.resolution.unwrap_or(DEFAULT_RESOLUTION);

    if smoothing_window < 2 {
        return Err(MovingAverageCrossProbabilityError::InvalidSmoothingWindow {
            smoothing_window,
        });
    }
    if slow_length < 2 {
        return Err(MovingAverageCrossProbabilityError::InvalidSlowLength { slow_length });
    }
    if fast_length == 0 {
        return Err(MovingAverageCrossProbabilityError::InvalidFastLength { fast_length });
    }
    if slow_length <= fast_length {
        return Err(MovingAverageCrossProbabilityError::InvalidLengthOrder {
            fast_length,
            slow_length,
        });
    }
    if resolution < 2 {
        return Err(MovingAverageCrossProbabilityError::InvalidResolution { resolution });
    }

    let sqrt_len = (smoothing_window as f64).sqrt().floor() as usize;
    let forecast_warmup = smoothing_window + sqrt_len - 1;
    let slow_ma_warmup = slow_length - 1;
    let fast_ma_warmup = fast_length - 1;
    let direction_warmup = slow_ma_warmup.max(fast_ma_warmup);
    let probability_warmup = forecast_warmup.max(direction_warmup).max(2 * slow_length);

    Ok(ResolvedParams {
        ma_type,
        smoothing_window,
        slow_length,
        fast_length,
        resolution,
        history_window_len: 2 * slow_length + 1,
        slow_alpha: 2.0 / (slow_length as f64 + 1.0),
        slow_beta: 1.0 - 2.0 / (slow_length as f64 + 1.0),
        fast_alpha: 2.0 / (fast_length as f64 + 1.0),
        fast_beta: 1.0 - 2.0 / (fast_length as f64 + 1.0),
        slow_ma_warmup,
        fast_ma_warmup,
        direction_warmup,
        forecast_warmup,
        probability_warmup,
    })
}

#[inline(always)]
fn extract_slice<'a>(
    input: &'a MovingAverageCrossProbabilityInput<'a>,
) -> Result<&'a [f64], MovingAverageCrossProbabilityError> {
    let data = match &input.data {
        MovingAverageCrossProbabilityData::Candles { candles } => candles.close.as_slice(),
        MovingAverageCrossProbabilityData::Slice(values) => *values,
    };
    if data.is_empty() {
        return Err(MovingAverageCrossProbabilityError::EmptyInputData);
    }
    if !data.iter().any(|v| v.is_finite()) {
        return Err(MovingAverageCrossProbabilityError::AllValuesNaN);
    }
    Ok(data)
}

#[inline(always)]
fn check_output_len(
    out: &[f64],
    expected: usize,
) -> Result<(), MovingAverageCrossProbabilityError> {
    if out.len() != expected {
        return Err(MovingAverageCrossProbabilityError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }
    Ok(())
}

#[inline(always)]
fn truncated_ema_from_window(window: &VecDeque<f64>, alpha: f64, beta: f64) -> Option<f64> {
    let mut iter = window.iter().rev();
    let mut ema = *iter.next()?;
    if !ema.is_finite() {
        return None;
    }
    for value in iter {
        if !value.is_finite() {
            return None;
        }
        ema = alpha.mul_add(*value, beta * ema);
    }
    Some(ema)
}

#[inline(always)]
fn truncated_ema_pair_from_window(
    window: &VecDeque<f64>,
    slow_alpha: f64,
    slow_beta: f64,
    fast_alpha: f64,
    fast_beta: f64,
) -> Option<(f64, f64)> {
    let mut iter = window.iter().rev();
    let first = *iter.next()?;
    if !first.is_finite() {
        return None;
    }
    let mut slow = first;
    let mut fast = first;
    for value in iter {
        if !value.is_finite() {
            return None;
        }
        slow = slow_alpha.mul_add(*value, slow_beta * slow);
        fast = fast_alpha.mul_add(*value, fast_beta * fast);
    }
    Some((slow, fast))
}

#[inline(always)]
fn truncated_ema_pair_from_slice(
    values: &[f64],
    slow_alpha: f64,
    slow_beta: f64,
    fast_alpha: f64,
    fast_beta: f64,
) -> (f64, f64) {
    let mut slow = values[0];
    let mut fast = values[0];
    for value in &values[1..] {
        slow = slow_alpha.mul_add(*value, slow_beta * slow);
        fast = fast_alpha.mul_add(*value, fast_beta * fast);
    }
    (slow, fast)
}

#[inline(always)]
fn count_crosses_by_probe<F>(resolution: usize, mut crossed: F) -> usize
where
    F: FnMut(usize) -> bool,
{
    let first = crossed(0);
    let last_idx = resolution - 1;
    let last = crossed(last_idx);

    if first == last {
        return if first { resolution } else { 0 };
    }

    let mut lo = 0usize;
    let mut hi = last_idx;
    while lo + 1 < hi {
        let mid = (lo + hi) >> 1;
        if crossed(mid) == first {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    if first {
        lo + 1
    } else {
        resolution - hi
    }
}

#[inline(always)]
fn count_ema_crosses_estimated(
    params: &ResolvedParams,
    lower: f64,
    step: f64,
    direction: f64,
    slow_current: f64,
    fast_current: f64,
) -> usize {
    let slow_base = params
        .slow_alpha
        .mul_add(lower, params.slow_beta * slow_current);
    let fast_base = params
        .fast_alpha
        .mul_add(lower, params.fast_beta * fast_current);
    let base = slow_base - fast_base;
    let slope = (params.slow_alpha - params.fast_alpha) * step;
    let crossed = |idx: usize| {
        let price = lower + step * idx as f64;
        let slow_future = params
            .slow_alpha
            .mul_add(price, params.slow_beta * slow_current);
        let fast_future = params
            .fast_alpha
            .mul_add(price, params.fast_beta * fast_current);
        if direction < 0.0 {
            slow_future > fast_future
        } else {
            slow_future <= fast_future
        }
    };

    if slope == 0.0 || !slope.is_finite() {
        return if crossed(0) { params.resolution } else { 0 };
    }

    let estimate = (-base / slope).floor();
    let mut hits = if direction < 0.0 {
        (estimate as isize + 1).clamp(0, params.resolution as isize) as usize
    } else {
        let first_true = (estimate as isize + 1).clamp(0, params.resolution as isize) as usize;
        params.resolution - first_true
    };

    if direction < 0.0 {
        while hits > 0 && !crossed(hits - 1) {
            hits -= 1;
        }
        while hits < params.resolution && crossed(hits) {
            hits += 1;
        }
    } else {
        while hits > 0 && !crossed(params.resolution - hits) {
            hits -= 1;
        }
        while hits < params.resolution {
            let idx = params.resolution - hits - 1;
            if crossed(idx) {
                hits += 1;
            } else {
                break;
            }
        }
    }

    hits
}

#[inline(always)]
fn probability_from_window(
    window: &VecDeque<f64>,
    params: &ResolvedParams,
    lower: f64,
    upper: f64,
    direction: f64,
) -> Option<f64> {
    let step = (upper - lower) / (params.resolution - 1) as f64;
    let mut hits = 0usize;

    match params.ma_type {
        MovingAverageCrossProbabilityMaType::Ema => {
            let (slow_current, fast_current) = truncated_ema_pair_from_window(
                window,
                params.slow_alpha,
                params.slow_beta,
                params.fast_alpha,
                params.fast_beta,
            )?;
            hits = count_ema_crosses_estimated(
                params,
                lower,
                step,
                direction,
                slow_current,
                fast_current,
            );
        }
        MovingAverageCrossProbabilityMaType::Sma => {
            let slow_needed = params.slow_length.saturating_sub(1);
            let fast_needed = params.fast_length.saturating_sub(1);
            let mut slow_sum = 0.0;
            let mut fast_sum = 0.0;
            for (idx, value) in window.iter().enumerate() {
                if idx < slow_needed {
                    slow_sum += *value;
                }
                if idx < fast_needed {
                    fast_sum += *value;
                }
                if idx >= slow_needed && idx >= fast_needed {
                    break;
                }
            }
            hits = count_crosses_by_probe(params.resolution, |idx| {
                let price = lower + step * idx as f64;
                let slow_future = (price + slow_sum) / params.slow_length as f64;
                let fast_future = (price + fast_sum) / params.fast_length as f64;
                let crossed = if direction < 0.0 {
                    slow_future > fast_future
                } else {
                    slow_future <= fast_future
                };
                crossed
            });
        }
    }

    Some(100.0 * hits as f64 / params.resolution as f64)
}

#[inline(always)]
fn moving_average_cross_probability_compute_into(
    data: &[f64],
    params: &ResolvedParams,
    out_value: &mut [f64],
    out_slow_ma: &mut [f64],
    out_fast_ma: &mut [f64],
    out_forecast: &mut [f64],
    out_upper: &mut [f64],
    out_lower: &mut [f64],
    out_direction: &mut [f64],
) -> Result<(), MovingAverageCrossProbabilityError> {
    let len = data.len();
    check_output_len(out_value, len)?;
    check_output_len(out_slow_ma, len)?;
    check_output_len(out_fast_ma, len)?;
    check_output_len(out_forecast, len)?;
    check_output_len(out_upper, len)?;
    check_output_len(out_lower, len)?;
    check_output_len(out_direction, len)?;

    if params.ma_type == MovingAverageCrossProbabilityMaType::Ema
        && data.iter().all(|value| value.is_finite())
    {
        return moving_average_cross_probability_ema_finite_compute_into(
            data,
            params,
            out_value,
            out_slow_ma,
            out_fast_ma,
            out_forecast,
            out_upper,
            out_lower,
            out_direction,
        );
    }

    out_value.fill(f64::NAN);
    out_slow_ma.fill(f64::NAN);
    out_fast_ma.fill(f64::NAN);
    out_forecast.fill(f64::NAN);
    out_upper.fill(f64::NAN);
    out_lower.fill(f64::NAN);
    out_direction.fill(f64::NAN);

    let mut slow_stream = CurrentMaStream::try_new(params.ma_type, params.slow_length)?;
    let mut fast_stream = CurrentMaStream::try_new(params.ma_type, params.fast_length)?;
    let mut hma_stream = HmaStream::try_new(HmaParams {
        period: Some(params.smoothing_window),
    })
    .map_err(
        |_| MovingAverageCrossProbabilityError::InvalidSmoothingWindow {
            smoothing_window: params.smoothing_window,
        },
    )?;
    let mut stddev_stream = StdDevStream::try_new(StdDevParams {
        period: Some(params.smoothing_window),
        nbdev: Some(4.0),
    })
    .map_err(
        |_| MovingAverageCrossProbabilityError::InvalidSmoothingWindow {
            smoothing_window: params.smoothing_window,
        },
    )?;

    let mut history: VecDeque<f64> = VecDeque::with_capacity(params.history_window_len);
    let mut invalid_history = 0usize;
    let mut previous_hma = f64::NAN;

    for (idx, &value) in data.iter().enumerate() {
        if history.len() == params.history_window_len {
            if let Some(old) = history.pop_back() {
                if !old.is_finite() {
                    invalid_history = invalid_history.saturating_sub(1);
                }
            }
        }
        history.push_front(value);
        if !value.is_finite() {
            invalid_history += 1;
        }

        let slow_ma = slow_stream.update(value).unwrap_or(f64::NAN);
        let fast_ma = fast_stream.update(value).unwrap_or(f64::NAN);
        let current_hma = hma_stream.update(value).unwrap_or(f64::NAN);
        let current_std = stddev_stream.update(value).unwrap_or(f64::NAN);

        out_slow_ma[idx] = slow_ma;
        out_fast_ma[idx] = fast_ma;

        let direction = if slow_ma.is_finite() && fast_ma.is_finite() {
            if fast_ma > slow_ma {
                -1.0
            } else {
                1.0
            }
        } else {
            f64::NAN
        };
        out_direction[idx] = direction;

        if current_hma.is_finite() && previous_hma.is_finite() && current_std.is_finite() {
            let forecast = current_hma + (current_hma - previous_hma);
            out_forecast[idx] = forecast;
            out_upper[idx] = forecast + current_std;
            out_lower[idx] = forecast - current_std;

            if direction.is_finite()
                && history.len() == params.history_window_len
                && invalid_history == 0
                && out_upper[idx].is_finite()
                && out_lower[idx].is_finite()
            {
                if let Some(probability) = probability_from_window(
                    &history,
                    params,
                    out_lower[idx],
                    out_upper[idx],
                    direction,
                ) {
                    out_value[idx] = probability;
                }
            }
        }

        previous_hma = current_hma;
    }

    Ok(())
}

#[inline(always)]
fn moving_average_cross_probability_ema_finite_compute_into(
    data: &[f64],
    params: &ResolvedParams,
    out_value: &mut [f64],
    out_slow_ma: &mut [f64],
    out_fast_ma: &mut [f64],
    out_forecast: &mut [f64],
    out_upper: &mut [f64],
    out_lower: &mut [f64],
    out_direction: &mut [f64],
) -> Result<(), MovingAverageCrossProbabilityError> {
    let mut slow_stream = EmaStream::try_new(EmaParams {
        period: Some(params.slow_length),
    })
    .map_err(|_| MovingAverageCrossProbabilityError::InvalidSlowLength {
        slow_length: params.slow_length,
    })?;
    let mut fast_stream = EmaStream::try_new(EmaParams {
        period: Some(params.fast_length),
    })
    .map_err(|_| MovingAverageCrossProbabilityError::InvalidFastLength {
        fast_length: params.fast_length,
    })?;
    let mut hma_stream = HmaStream::try_new(HmaParams {
        period: Some(params.smoothing_window),
    })
    .map_err(
        |_| MovingAverageCrossProbabilityError::InvalidSmoothingWindow {
            smoothing_window: params.smoothing_window,
        },
    )?;
    let mut stddev_stream = StdDevStream::try_new(StdDevParams {
        period: Some(params.smoothing_window),
        nbdev: Some(4.0),
    })
    .map_err(
        |_| MovingAverageCrossProbabilityError::InvalidSmoothingWindow {
            smoothing_window: params.smoothing_window,
        },
    )?;

    let history_len = params.history_window_len;
    let slow_drop_scale = params.slow_beta.powi(history_len as i32);
    let fast_drop_scale = params.fast_beta.powi(history_len as i32);
    let mut slow_probability_ema = f64::NAN;
    let mut fast_probability_ema = f64::NAN;
    let mut previous_hma = f64::NAN;

    for (idx, &value) in data.iter().enumerate() {
        let slow_ma = slow_stream.update(value).unwrap_or(f64::NAN);
        let fast_ma = fast_stream.update(value).unwrap_or(f64::NAN);
        let current_hma = hma_stream.update(value).unwrap_or(f64::NAN);
        let current_std = stddev_stream.update(value).unwrap_or(f64::NAN);

        out_slow_ma[idx] = slow_ma;
        out_fast_ma[idx] = fast_ma;

        let direction = if slow_ma.is_finite() && fast_ma.is_finite() {
            if fast_ma > slow_ma {
                -1.0
            } else {
                1.0
            }
        } else {
            f64::NAN
        };
        out_direction[idx] = direction;

        let mut forecast = f64::NAN;
        let mut upper = f64::NAN;
        let mut lower = f64::NAN;
        let mut probability = f64::NAN;

        if idx + 1 == history_len {
            let pair = truncated_ema_pair_from_slice(
                &data[..=idx],
                params.slow_alpha,
                params.slow_beta,
                params.fast_alpha,
                params.fast_beta,
            );
            slow_probability_ema = pair.0;
            fast_probability_ema = pair.1;
        } else if idx + 1 > history_len {
            let dropped = data[idx - history_len];
            let new_oldest = data[idx + 1 - history_len];
            slow_probability_ema = params
                .slow_alpha
                .mul_add(value, params.slow_beta * slow_probability_ema)
                + slow_drop_scale * (new_oldest - dropped);
            fast_probability_ema = params
                .fast_alpha
                .mul_add(value, params.fast_beta * fast_probability_ema)
                + fast_drop_scale * (new_oldest - dropped);
        }

        if current_hma.is_finite() && previous_hma.is_finite() && current_std.is_finite() {
            forecast = current_hma + (current_hma - previous_hma);
            upper = forecast + current_std;
            lower = forecast - current_std;

            if direction.is_finite() && idx + 1 >= history_len {
                let step = (upper - lower) / (params.resolution - 1) as f64;
                let hits = count_ema_crosses_estimated(
                    params,
                    lower,
                    step,
                    direction,
                    slow_probability_ema,
                    fast_probability_ema,
                );
                probability = 100.0 * hits as f64 / params.resolution as f64;
            }
        }

        out_value[idx] = probability;
        out_forecast[idx] = forecast;
        out_upper[idx] = upper;
        out_lower[idx] = lower;

        previous_hma = current_hma;
    }

    Ok(())
}

#[inline]
pub fn moving_average_cross_probability(
    input: &MovingAverageCrossProbabilityInput,
) -> Result<MovingAverageCrossProbabilityOutput, MovingAverageCrossProbabilityError> {
    moving_average_cross_probability_with_kernel(input, Kernel::Auto)
}

pub fn moving_average_cross_probability_with_kernel(
    input: &MovingAverageCrossProbabilityInput,
    _kernel: Kernel,
) -> Result<MovingAverageCrossProbabilityOutput, MovingAverageCrossProbabilityError> {
    let data = extract_slice(input)?;
    let params = resolve_params(&input.params)?;
    let len = data.len();

    let mut value = alloc_with_nan_prefix(len, params.probability_warmup.min(len));
    let mut slow_ma = alloc_with_nan_prefix(len, params.slow_ma_warmup.min(len));
    let mut fast_ma = alloc_with_nan_prefix(len, params.fast_ma_warmup.min(len));
    let mut forecast = alloc_with_nan_prefix(len, params.forecast_warmup.min(len));
    let mut upper = alloc_with_nan_prefix(len, params.forecast_warmup.min(len));
    let mut lower = alloc_with_nan_prefix(len, params.forecast_warmup.min(len));
    let mut direction = alloc_with_nan_prefix(len, params.direction_warmup.min(len));

    moving_average_cross_probability_compute_into(
        data,
        &params,
        &mut value,
        &mut slow_ma,
        &mut fast_ma,
        &mut forecast,
        &mut upper,
        &mut lower,
        &mut direction,
    )?;

    Ok(MovingAverageCrossProbabilityOutput {
        value,
        slow_ma,
        fast_ma,
        forecast,
        upper,
        lower,
        direction,
    })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn moving_average_cross_probability_into(
    input: &MovingAverageCrossProbabilityInput,
    out_value: &mut [f64],
    out_slow_ma: &mut [f64],
    out_fast_ma: &mut [f64],
    out_forecast: &mut [f64],
    out_upper: &mut [f64],
    out_lower: &mut [f64],
    out_direction: &mut [f64],
) -> Result<(), MovingAverageCrossProbabilityError> {
    moving_average_cross_probability_into_slice(
        out_value,
        out_slow_ma,
        out_fast_ma,
        out_forecast,
        out_upper,
        out_lower,
        out_direction,
        input,
        Kernel::Auto,
    )
}

pub fn moving_average_cross_probability_into_slice(
    out_value: &mut [f64],
    out_slow_ma: &mut [f64],
    out_fast_ma: &mut [f64],
    out_forecast: &mut [f64],
    out_upper: &mut [f64],
    out_lower: &mut [f64],
    out_direction: &mut [f64],
    input: &MovingAverageCrossProbabilityInput,
    _kernel: Kernel,
) -> Result<(), MovingAverageCrossProbabilityError> {
    let data = extract_slice(input)?;
    let params = resolve_params(&input.params)?;
    moving_average_cross_probability_compute_into(
        data,
        &params,
        out_value,
        out_slow_ma,
        out_fast_ma,
        out_forecast,
        out_upper,
        out_lower,
        out_direction,
    )
}

#[derive(Debug)]
pub struct MovingAverageCrossProbabilityStream {
    params: ResolvedParams,
    slow_stream: CurrentMaStream,
    fast_stream: CurrentMaStream,
    hma_stream: HmaStream,
    stddev_stream: StdDevStream,
    history: VecDeque<f64>,
    invalid_history: usize,
    previous_hma: f64,
}

impl MovingAverageCrossProbabilityStream {
    pub fn try_new(
        params: MovingAverageCrossProbabilityParams,
    ) -> Result<Self, MovingAverageCrossProbabilityError> {
        let params = resolve_params(&params)?;
        Ok(Self {
            slow_stream: CurrentMaStream::try_new(params.ma_type, params.slow_length)?,
            fast_stream: CurrentMaStream::try_new(params.ma_type, params.fast_length)?,
            hma_stream: HmaStream::try_new(HmaParams {
                period: Some(params.smoothing_window),
            })
            .map_err(|_| {
                MovingAverageCrossProbabilityError::InvalidSmoothingWindow {
                    smoothing_window: params.smoothing_window,
                }
            })?,
            stddev_stream: StdDevStream::try_new(StdDevParams {
                period: Some(params.smoothing_window),
                nbdev: Some(4.0),
            })
            .map_err(|_| {
                MovingAverageCrossProbabilityError::InvalidSmoothingWindow {
                    smoothing_window: params.smoothing_window,
                }
            })?,
            history: VecDeque::with_capacity(params.history_window_len),
            invalid_history: 0,
            previous_hma: f64::NAN,
            params,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> (f64, f64, f64, f64, f64, f64, f64) {
        if self.history.len() == self.params.history_window_len {
            if let Some(old) = self.history.pop_back() {
                if !old.is_finite() {
                    self.invalid_history = self.invalid_history.saturating_sub(1);
                }
            }
        }
        self.history.push_front(value);
        if !value.is_finite() {
            self.invalid_history += 1;
        }

        let slow_ma = self.slow_stream.update(value).unwrap_or(f64::NAN);
        let fast_ma = self.fast_stream.update(value).unwrap_or(f64::NAN);
        let current_hma = self.hma_stream.update(value).unwrap_or(f64::NAN);
        let current_std = self.stddev_stream.update(value).unwrap_or(f64::NAN);
        let direction = if slow_ma.is_finite() && fast_ma.is_finite() {
            if fast_ma > slow_ma {
                -1.0
            } else {
                1.0
            }
        } else {
            f64::NAN
        };

        let mut forecast = f64::NAN;
        let mut upper = f64::NAN;
        let mut lower = f64::NAN;
        let mut probability = f64::NAN;
        if current_hma.is_finite() && self.previous_hma.is_finite() && current_std.is_finite() {
            forecast = current_hma + (current_hma - self.previous_hma);
            upper = forecast + current_std;
            lower = forecast - current_std;
            if direction.is_finite()
                && self.history.len() == self.params.history_window_len
                && self.invalid_history == 0
            {
                probability =
                    probability_from_window(&self.history, &self.params, lower, upper, direction)
                        .unwrap_or(f64::NAN);
            }
        }
        self.previous_hma = current_hma;

        (
            probability,
            slow_ma,
            fast_ma,
            forecast,
            upper,
            lower,
            direction,
        )
    }
}

#[derive(Clone, Debug)]
pub struct MovingAverageCrossProbabilityBatchRange {
    pub smoothing_window: (usize, usize, usize),
    pub slow_length: (usize, usize, usize),
    pub fast_length: (usize, usize, usize),
    pub resolution: (usize, usize, usize),
    pub ma_type: MovingAverageCrossProbabilityMaType,
}

impl Default for MovingAverageCrossProbabilityBatchRange {
    fn default() -> Self {
        Self {
            smoothing_window: (DEFAULT_SMOOTHING_WINDOW, DEFAULT_SMOOTHING_WINDOW, 0),
            slow_length: (DEFAULT_SLOW_LENGTH, DEFAULT_SLOW_LENGTH, 0),
            fast_length: (DEFAULT_FAST_LENGTH, DEFAULT_FAST_LENGTH, 0),
            resolution: (DEFAULT_RESOLUTION, DEFAULT_RESOLUTION, 0),
            ma_type: DEFAULT_MA_TYPE,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MovingAverageCrossProbabilityBatchBuilder {
    range: MovingAverageCrossProbabilityBatchRange,
    kernel: Kernel,
}

impl MovingAverageCrossProbabilityBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline(always)]
    pub fn smoothing_window_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.smoothing_window = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn slow_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.slow_length = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn fast_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.fast_length = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn resolution_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.resolution = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn ma_type(mut self, value: MovingAverageCrossProbabilityMaType) -> Self {
        self.range.ma_type = value;
        self
    }

    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<MovingAverageCrossProbabilityBatchOutput, MovingAverageCrossProbabilityError> {
        moving_average_cross_probability_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<MovingAverageCrossProbabilityBatchOutput, MovingAverageCrossProbabilityError> {
        self.apply_slice(candles.close.as_slice())
    }
}

#[derive(Clone, Debug)]
pub struct MovingAverageCrossProbabilityBatchOutput {
    pub value: Vec<f64>,
    pub slow_ma: Vec<f64>,
    pub fast_ma: Vec<f64>,
    pub forecast: Vec<f64>,
    pub upper: Vec<f64>,
    pub lower: Vec<f64>,
    pub direction: Vec<f64>,
    pub combos: Vec<MovingAverageCrossProbabilityParams>,
    pub rows: usize,
    pub cols: usize,
}

fn axis_usize(
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, MovingAverageCrossProbabilityError> {
    if step == 0 || start == end {
        return Ok(vec![start]);
    }
    let mut values = Vec::new();
    if start < end {
        let mut current = start;
        while current <= end {
            values.push(current);
            let next = current.saturating_add(step);
            if next <= current {
                break;
            }
            current = next;
        }
    } else {
        let mut current = start;
        while current >= end {
            values.push(current);
            let next = current.saturating_sub(step);
            if next == current {
                break;
            }
            current = next;
            if current == 0 && end > 0 {
                break;
            }
        }
    }
    if values.is_empty() {
        return Err(MovingAverageCrossProbabilityError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(values)
}

pub fn moving_average_cross_probability_expand_grid(
    range: &MovingAverageCrossProbabilityBatchRange,
) -> Result<Vec<MovingAverageCrossProbabilityParams>, MovingAverageCrossProbabilityError> {
    let smoothing_windows = axis_usize(range.smoothing_window)?;
    let slow_lengths = axis_usize(range.slow_length)?;
    let fast_lengths = axis_usize(range.fast_length)?;
    let resolutions = axis_usize(range.resolution)?;

    let cap = smoothing_windows
        .len()
        .checked_mul(slow_lengths.len())
        .and_then(|v| v.checked_mul(fast_lengths.len()))
        .and_then(|v| v.checked_mul(resolutions.len()))
        .ok_or_else(|| MovingAverageCrossProbabilityError::InvalidRange {
            start: "grid".to_string(),
            end: "overflow".to_string(),
            step: "n/a".to_string(),
        })?;

    let mut out = Vec::with_capacity(cap);
    for &smoothing_window in &smoothing_windows {
        for &slow_length in &slow_lengths {
            for &fast_length in &fast_lengths {
                for &resolution in &resolutions {
                    out.push(MovingAverageCrossProbabilityParams {
                        ma_type: Some(range.ma_type),
                        smoothing_window: Some(smoothing_window),
                        slow_length: Some(slow_length),
                        fast_length: Some(fast_length),
                        resolution: Some(resolution),
                    });
                }
            }
        }
    }
    Ok(out)
}

#[inline(always)]
fn batch_shape(rows: usize, cols: usize) -> Result<usize, MovingAverageCrossProbabilityError> {
    rows.checked_mul(cols)
        .ok_or_else(|| MovingAverageCrossProbabilityError::InvalidRange {
            start: rows.to_string(),
            end: cols.to_string(),
            step: "overflow".to_string(),
        })
}

#[inline(always)]
pub fn moving_average_cross_probability_batch_slice(
    data: &[f64],
    range: &MovingAverageCrossProbabilityBatchRange,
    _kernel: Kernel,
) -> Result<MovingAverageCrossProbabilityBatchOutput, MovingAverageCrossProbabilityError> {
    moving_average_cross_probability_batch_inner(data, range, false)
}

#[inline(always)]
pub fn moving_average_cross_probability_batch_par_slice(
    data: &[f64],
    range: &MovingAverageCrossProbabilityBatchRange,
    _kernel: Kernel,
) -> Result<MovingAverageCrossProbabilityBatchOutput, MovingAverageCrossProbabilityError> {
    moving_average_cross_probability_batch_inner(data, range, true)
}

pub fn moving_average_cross_probability_batch_with_kernel(
    data: &[f64],
    range: &MovingAverageCrossProbabilityBatchRange,
    kernel: Kernel,
) -> Result<MovingAverageCrossProbabilityBatchOutput, MovingAverageCrossProbabilityError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => {
            return Err(MovingAverageCrossProbabilityError::InvalidKernelForBatch(
                other,
            ))
        }
    };
    moving_average_cross_probability_batch_inner(data, range, batch_kernel.is_batch())
}

fn moving_average_cross_probability_batch_inner(
    data: &[f64],
    range: &MovingAverageCrossProbabilityBatchRange,
    parallel: bool,
) -> Result<MovingAverageCrossProbabilityBatchOutput, MovingAverageCrossProbabilityError> {
    let combos = moving_average_cross_probability_expand_grid(range)?;
    let rows = combos.len();
    let cols = data.len();
    let total = batch_shape(rows, cols)?;

    let mut value_buf = make_uninit_matrix(rows, cols);
    let mut slow_buf = make_uninit_matrix(rows, cols);
    let mut fast_buf = make_uninit_matrix(rows, cols);
    let mut forecast_buf = make_uninit_matrix(rows, cols);
    let mut upper_buf = make_uninit_matrix(rows, cols);
    let mut lower_buf = make_uninit_matrix(rows, cols);
    let mut direction_buf = make_uninit_matrix(rows, cols);

    let mut value_warmups = Vec::with_capacity(rows);
    let mut slow_warmups = Vec::with_capacity(rows);
    let mut fast_warmups = Vec::with_capacity(rows);
    let mut forecast_warmups = Vec::with_capacity(rows);
    let mut direction_warmups = Vec::with_capacity(rows);
    for combo in &combos {
        let resolved = resolve_params(combo)?;
        value_warmups.push(resolved.probability_warmup);
        slow_warmups.push(resolved.slow_ma_warmup);
        fast_warmups.push(resolved.fast_ma_warmup);
        forecast_warmups.push(resolved.forecast_warmup);
        direction_warmups.push(resolved.direction_warmup);
    }

    init_matrix_prefixes(&mut value_buf, cols, &value_warmups);
    init_matrix_prefixes(&mut slow_buf, cols, &slow_warmups);
    init_matrix_prefixes(&mut fast_buf, cols, &fast_warmups);
    init_matrix_prefixes(&mut forecast_buf, cols, &forecast_warmups);
    init_matrix_prefixes(&mut upper_buf, cols, &forecast_warmups);
    init_matrix_prefixes(&mut lower_buf, cols, &forecast_warmups);
    init_matrix_prefixes(&mut direction_buf, cols, &direction_warmups);

    let mut value = unsafe {
        Vec::from_raw_parts(
            value_buf.as_mut_ptr() as *mut f64,
            total,
            value_buf.capacity(),
        )
    };
    let mut slow_ma = unsafe {
        Vec::from_raw_parts(
            slow_buf.as_mut_ptr() as *mut f64,
            total,
            slow_buf.capacity(),
        )
    };
    let mut fast_ma = unsafe {
        Vec::from_raw_parts(
            fast_buf.as_mut_ptr() as *mut f64,
            total,
            fast_buf.capacity(),
        )
    };
    let mut forecast = unsafe {
        Vec::from_raw_parts(
            forecast_buf.as_mut_ptr() as *mut f64,
            total,
            forecast_buf.capacity(),
        )
    };
    let mut upper = unsafe {
        Vec::from_raw_parts(
            upper_buf.as_mut_ptr() as *mut f64,
            total,
            upper_buf.capacity(),
        )
    };
    let mut lower = unsafe {
        Vec::from_raw_parts(
            lower_buf.as_mut_ptr() as *mut f64,
            total,
            lower_buf.capacity(),
        )
    };
    let mut direction = unsafe {
        Vec::from_raw_parts(
            direction_buf.as_mut_ptr() as *mut f64,
            total,
            direction_buf.capacity(),
        )
    };
    std::mem::forget(value_buf);
    std::mem::forget(slow_buf);
    std::mem::forget(fast_buf);
    std::mem::forget(forecast_buf);
    std::mem::forget(upper_buf);
    std::mem::forget(lower_buf);
    std::mem::forget(direction_buf);

    moving_average_cross_probability_batch_inner_into(
        data,
        range,
        parallel,
        &mut value,
        &mut slow_ma,
        &mut fast_ma,
        &mut forecast,
        &mut upper,
        &mut lower,
        &mut direction,
    )?;

    Ok(MovingAverageCrossProbabilityBatchOutput {
        value,
        slow_ma,
        fast_ma,
        forecast,
        upper,
        lower,
        direction,
        combos,
        rows,
        cols,
    })
}

pub fn moving_average_cross_probability_batch_into_slice(
    out_value: &mut [f64],
    out_slow_ma: &mut [f64],
    out_fast_ma: &mut [f64],
    out_forecast: &mut [f64],
    out_upper: &mut [f64],
    out_lower: &mut [f64],
    out_direction: &mut [f64],
    data: &[f64],
    range: &MovingAverageCrossProbabilityBatchRange,
    kernel: Kernel,
) -> Result<(), MovingAverageCrossProbabilityError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => {
            return Err(MovingAverageCrossProbabilityError::InvalidKernelForBatch(
                other,
            ))
        }
    };
    moving_average_cross_probability_batch_inner_into(
        data,
        range,
        batch_kernel.is_batch(),
        out_value,
        out_slow_ma,
        out_fast_ma,
        out_forecast,
        out_upper,
        out_lower,
        out_direction,
    )?;
    Ok(())
}

fn moving_average_cross_probability_batch_inner_into(
    data: &[f64],
    range: &MovingAverageCrossProbabilityBatchRange,
    parallel: bool,
    out_value: &mut [f64],
    out_slow_ma: &mut [f64],
    out_fast_ma: &mut [f64],
    out_forecast: &mut [f64],
    out_upper: &mut [f64],
    out_lower: &mut [f64],
    out_direction: &mut [f64],
) -> Result<Vec<MovingAverageCrossProbabilityParams>, MovingAverageCrossProbabilityError> {
    if data.is_empty() {
        return Err(MovingAverageCrossProbabilityError::EmptyInputData);
    }
    if !data.iter().any(|value| value.is_finite()) {
        return Err(MovingAverageCrossProbabilityError::AllValuesNaN);
    }
    let combos = moving_average_cross_probability_expand_grid(range)?;
    let rows = combos.len();
    let cols = data.len();
    let total = batch_shape(rows, cols)?;
    check_output_len(out_value, total)?;
    check_output_len(out_slow_ma, total)?;
    check_output_len(out_fast_ma, total)?;
    check_output_len(out_forecast, total)?;
    check_output_len(out_upper, total)?;
    check_output_len(out_lower, total)?;
    check_output_len(out_direction, total)?;

    #[cfg(not(target_arch = "wasm32"))]
    if parallel {
        let results: Vec<Result<(), MovingAverageCrossProbabilityError>> = out_value
            .par_chunks_mut(cols)
            .zip(out_slow_ma.par_chunks_mut(cols))
            .zip(out_fast_ma.par_chunks_mut(cols))
            .zip(out_forecast.par_chunks_mut(cols))
            .zip(out_upper.par_chunks_mut(cols))
            .zip(out_lower.par_chunks_mut(cols))
            .zip(out_direction.par_chunks_mut(cols))
            .zip(combos.par_iter())
            .map(
                |(
                    (
                        (((((value_row, slow_row), fast_row), forecast_row), upper_row), lower_row),
                        direction_row,
                    ),
                    combo,
                )| {
                    let params = resolve_params(combo)?;
                    moving_average_cross_probability_compute_into(
                        data,
                        &params,
                        value_row,
                        slow_row,
                        fast_row,
                        forecast_row,
                        upper_row,
                        lower_row,
                        direction_row,
                    )
                },
            )
            .collect();
        for result in results {
            result?;
        }
    }

    if !parallel || cfg!(target_arch = "wasm32") {
        for (row, combo) in combos.iter().enumerate() {
            let start = row * cols;
            let end = start + cols;
            let params = resolve_params(combo)?;
            moving_average_cross_probability_compute_into(
                data,
                &params,
                &mut out_value[start..end],
                &mut out_slow_ma[start..end],
                &mut out_fast_ma[start..end],
                &mut out_forecast[start..end],
                &mut out_upper[start..end],
                &mut out_lower[start..end],
                &mut out_direction[start..end],
            )?;
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "moving_average_cross_probability")]
#[pyo3(signature = (
    data,
    ma_type="ema",
    smoothing_window=7,
    slow_length=30,
    fast_length=14,
    resolution=50,
    kernel=None
))]
pub fn moving_average_cross_probability_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    ma_type: &str,
    smoothing_window: usize,
    slow_length: usize,
    fast_length: usize,
    resolution: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let input = MovingAverageCrossProbabilityInput::from_slice(
        data,
        MovingAverageCrossProbabilityParams {
            ma_type: Some(
                MovingAverageCrossProbabilityMaType::from_str(ma_type)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            ),
            smoothing_window: Some(smoothing_window),
            slow_length: Some(slow_length),
            fast_length: Some(fast_length),
            resolution: Some(resolution),
        },
    );
    let kernel = validate_kernel(kernel, false)?;
    let out = py
        .allow_threads(|| moving_average_cross_probability_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dict = PyDict::new(py);
    dict.set_item("value", out.value.into_pyarray(py))?;
    dict.set_item("slow_ma", out.slow_ma.into_pyarray(py))?;
    dict.set_item("fast_ma", out.fast_ma.into_pyarray(py))?;
    dict.set_item("forecast", out.forecast.into_pyarray(py))?;
    dict.set_item("upper", out.upper.into_pyarray(py))?;
    dict.set_item("lower", out.lower.into_pyarray(py))?;
    dict.set_item("direction", out.direction.into_pyarray(py))?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "MovingAverageCrossProbabilityStream")]
pub struct MovingAverageCrossProbabilityStreamPy {
    stream: MovingAverageCrossProbabilityStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl MovingAverageCrossProbabilityStreamPy {
    #[new]
    #[pyo3(signature = (
        ma_type="ema",
        smoothing_window=7,
        slow_length=30,
        fast_length=14,
        resolution=50
    ))]
    fn new(
        ma_type: &str,
        smoothing_window: usize,
        slow_length: usize,
        fast_length: usize,
        resolution: usize,
    ) -> PyResult<Self> {
        let stream =
            MovingAverageCrossProbabilityStream::try_new(MovingAverageCrossProbabilityParams {
                ma_type: Some(
                    MovingAverageCrossProbabilityMaType::from_str(ma_type)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?,
                ),
                smoothing_window: Some(smoothing_window),
                slow_length: Some(slow_length),
                fast_length: Some(fast_length),
                resolution: Some(resolution),
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, value: f64) -> (f64, f64, f64, f64, f64, f64, f64) {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "moving_average_cross_probability_batch")]
#[pyo3(signature = (
    data,
    smoothing_window_range=(7,7,0),
    slow_length_range=(30,30,0),
    fast_length_range=(14,14,0),
    resolution_range=(50,50,0),
    ma_type="ema",
    kernel=None
))]
pub fn moving_average_cross_probability_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    smoothing_window_range: (usize, usize, usize),
    slow_length_range: (usize, usize, usize),
    fast_length_range: (usize, usize, usize),
    resolution_range: (usize, usize, usize),
    ma_type: &str,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let sweep = MovingAverageCrossProbabilityBatchRange {
        smoothing_window: smoothing_window_range,
        slow_length: slow_length_range,
        fast_length: fast_length_range,
        resolution: resolution_range,
        ma_type: MovingAverageCrossProbabilityMaType::from_str(ma_type)
            .map_err(|e| PyValueError::new_err(e.to_string()))?,
    };
    let combos = moving_average_cross_probability_expand_grid(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let out_value = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_slow_ma = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_fast_ma = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_forecast = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_upper = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_lower = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_direction = unsafe { PyArray1::<f64>::new(py, [total], false) };

    let value_slice = unsafe { out_value.as_slice_mut()? };
    let slow_slice = unsafe { out_slow_ma.as_slice_mut()? };
    let fast_slice = unsafe { out_fast_ma.as_slice_mut()? };
    let forecast_slice = unsafe { out_forecast.as_slice_mut()? };
    let upper_slice = unsafe { out_upper.as_slice_mut()? };
    let lower_slice = unsafe { out_lower.as_slice_mut()? };
    let direction_slice = unsafe { out_direction.as_slice_mut()? };
    let kernel = validate_kernel(kernel, true)?;

    py.allow_threads(|| {
        let batch_kernel = match kernel {
            Kernel::Auto => detect_best_batch_kernel(),
            other => other,
        };
        moving_average_cross_probability_batch_inner_into(
            data,
            &sweep,
            batch_kernel.is_batch(),
            value_slice,
            slow_slice,
            fast_slice,
            forecast_slice,
            upper_slice,
            lower_slice,
            direction_slice,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("value", out_value.reshape((rows, cols))?)?;
    dict.set_item("slow_ma", out_slow_ma.reshape((rows, cols))?)?;
    dict.set_item("fast_ma", out_fast_ma.reshape((rows, cols))?)?;
    dict.set_item("forecast", out_forecast.reshape((rows, cols))?)?;
    dict.set_item("upper", out_upper.reshape((rows, cols))?)?;
    dict.set_item("lower", out_lower.reshape((rows, cols))?)?;
    dict.set_item("direction", out_direction.reshape((rows, cols))?)?;
    dict.set_item(
        "smoothing_windows",
        combos
            .iter()
            .map(|combo| combo.smoothing_window.unwrap_or(DEFAULT_SMOOTHING_WINDOW))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "slow_lengths",
        combos
            .iter()
            .map(|combo| combo.slow_length.unwrap_or(DEFAULT_SLOW_LENGTH))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "fast_lengths",
        combos
            .iter()
            .map(|combo| combo.fast_length.unwrap_or(DEFAULT_FAST_LENGTH))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "resolutions",
        combos
            .iter()
            .map(|combo| combo.resolution.unwrap_or(DEFAULT_RESOLUTION))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_moving_average_cross_probability_module(
    m: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(moving_average_cross_probability_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        moving_average_cross_probability_batch_py,
        m
    )?)?;
    m.add_class::<MovingAverageCrossProbabilityStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct MovingAverageCrossProbabilityJsOutput {
    pub value: Vec<f64>,
    pub slow_ma: Vec<f64>,
    pub fast_ma: Vec<f64>,
    pub forecast: Vec<f64>,
    pub upper: Vec<f64>,
    pub lower: Vec<f64>,
    pub direction: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
fn js_vec3_to_usize(name: &str, values: &[f64]) -> Result<(usize, usize, usize), JsValue> {
    if values.len() != 3 {
        return Err(JsValue::from_str(&format!(
            "Invalid config: {name} must have exactly 3 elements [start, end, step]"
        )));
    }
    let mut out = [0usize; 3];
    for (idx, value) in values.iter().enumerate() {
        if !value.is_finite() || *value < 0.0 || value.fract() != 0.0 {
            return Err(JsValue::from_str(&format!(
                "Invalid config: {name} values must be non-negative integers"
            )));
        }
        out[idx] = *value as usize;
    }
    Ok((out[0], out[1], out[2]))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "moving_average_cross_probability_js")]
pub fn moving_average_cross_probability_js(
    data: &[f64],
    ma_type: String,
    smoothing_window: usize,
    slow_length: usize,
    fast_length: usize,
    resolution: usize,
) -> Result<JsValue, JsValue> {
    let input = MovingAverageCrossProbabilityInput::from_slice(
        data,
        MovingAverageCrossProbabilityParams {
            ma_type: Some(
                MovingAverageCrossProbabilityMaType::from_str(&ma_type)
                    .map_err(|e| JsValue::from_str(&e))?,
            ),
            smoothing_window: Some(smoothing_window),
            slow_length: Some(slow_length),
            fast_length: Some(fast_length),
            resolution: Some(resolution),
        },
    );
    let out = moving_average_cross_probability_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&MovingAverageCrossProbabilityJsOutput {
        value: out.value,
        slow_ma: out.slow_ma,
        fast_ma: out.fast_ma,
        forecast: out.forecast,
        upper: out.upper,
        lower: out.lower,
        direction: out.direction,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct MovingAverageCrossProbabilityBatchConfig {
    pub smoothing_window_range: Vec<f64>,
    pub slow_length_range: Vec<f64>,
    pub fast_length_range: Vec<f64>,
    pub resolution_range: Vec<f64>,
    pub ma_type: Option<String>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct MovingAverageCrossProbabilityBatchJsOutput {
    pub value: Vec<f64>,
    pub slow_ma: Vec<f64>,
    pub fast_ma: Vec<f64>,
    pub forecast: Vec<f64>,
    pub upper: Vec<f64>,
    pub lower: Vec<f64>,
    pub direction: Vec<f64>,
    pub smoothing_windows: Vec<usize>,
    pub slow_lengths: Vec<usize>,
    pub fast_lengths: Vec<usize>,
    pub resolutions: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "moving_average_cross_probability_batch_js")]
pub fn moving_average_cross_probability_batch_js(
    data: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: MovingAverageCrossProbabilityBatchConfig =
        serde_wasm_bindgen::from_value(config)
            .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let ma_type = config
        .ma_type
        .as_deref()
        .map(MovingAverageCrossProbabilityMaType::from_str)
        .transpose()
        .map_err(|e| JsValue::from_str(&e))?
        .unwrap_or(DEFAULT_MA_TYPE);
    let sweep = MovingAverageCrossProbabilityBatchRange {
        smoothing_window: js_vec3_to_usize(
            "smoothing_window_range",
            &config.smoothing_window_range,
        )?,
        slow_length: js_vec3_to_usize("slow_length_range", &config.slow_length_range)?,
        fast_length: js_vec3_to_usize("fast_length_range", &config.fast_length_range)?,
        resolution: js_vec3_to_usize("resolution_range", &config.resolution_range)?,
        ma_type,
    };
    let out = moving_average_cross_probability_batch_with_kernel(data, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&MovingAverageCrossProbabilityBatchJsOutput {
        value: out.value,
        slow_ma: out.slow_ma,
        fast_ma: out.fast_ma,
        forecast: out.forecast,
        upper: out.upper,
        lower: out.lower,
        direction: out.direction,
        smoothing_windows: out
            .combos
            .iter()
            .map(|combo| combo.smoothing_window.unwrap_or(DEFAULT_SMOOTHING_WINDOW))
            .collect(),
        slow_lengths: out
            .combos
            .iter()
            .map(|combo| combo.slow_length.unwrap_or(DEFAULT_SLOW_LENGTH))
            .collect(),
        fast_lengths: out
            .combos
            .iter()
            .map(|combo| combo.fast_length.unwrap_or(DEFAULT_FAST_LENGTH))
            .collect(),
        resolutions: out
            .combos
            .iter()
            .map(|combo| combo.resolution.unwrap_or(DEFAULT_RESOLUTION))
            .collect(),
        rows: out.rows,
        cols: out.cols,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn moving_average_cross_probability_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn moving_average_cross_probability_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn moving_average_cross_probability_into(
    in_ptr: *const f64,
    out_value_ptr: *mut f64,
    out_slow_ma_ptr: *mut f64,
    out_fast_ma_ptr: *mut f64,
    out_forecast_ptr: *mut f64,
    out_upper_ptr: *mut f64,
    out_lower_ptr: *mut f64,
    out_direction_ptr: *mut f64,
    len: usize,
    ma_type: String,
    smoothing_window: usize,
    slow_length: usize,
    fast_length: usize,
    resolution: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null()
        || out_value_ptr.is_null()
        || out_slow_ma_ptr.is_null()
        || out_fast_ma_ptr.is_null()
        || out_forecast_ptr.is_null()
        || out_upper_ptr.is_null()
        || out_lower_ptr.is_null()
        || out_direction_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to moving_average_cross_probability_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let out_value = std::slice::from_raw_parts_mut(out_value_ptr, len);
        let out_slow_ma = std::slice::from_raw_parts_mut(out_slow_ma_ptr, len);
        let out_fast_ma = std::slice::from_raw_parts_mut(out_fast_ma_ptr, len);
        let out_forecast = std::slice::from_raw_parts_mut(out_forecast_ptr, len);
        let out_upper = std::slice::from_raw_parts_mut(out_upper_ptr, len);
        let out_lower = std::slice::from_raw_parts_mut(out_lower_ptr, len);
        let out_direction = std::slice::from_raw_parts_mut(out_direction_ptr, len);
        let input = MovingAverageCrossProbabilityInput::from_slice(
            data,
            MovingAverageCrossProbabilityParams {
                ma_type: Some(
                    MovingAverageCrossProbabilityMaType::from_str(&ma_type)
                        .map_err(|e| JsValue::from_str(&e))?,
                ),
                smoothing_window: Some(smoothing_window),
                slow_length: Some(slow_length),
                fast_length: Some(fast_length),
                resolution: Some(resolution),
            },
        );
        moving_average_cross_probability_into_slice(
            out_value,
            out_slow_ma,
            out_fast_ma,
            out_forecast,
            out_upper,
            out_lower,
            out_direction,
            &input,
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn moving_average_cross_probability_batch_into(
    in_ptr: *const f64,
    out_value_ptr: *mut f64,
    out_slow_ma_ptr: *mut f64,
    out_fast_ma_ptr: *mut f64,
    out_forecast_ptr: *mut f64,
    out_upper_ptr: *mut f64,
    out_lower_ptr: *mut f64,
    out_direction_ptr: *mut f64,
    len: usize,
    smoothing_window_start: usize,
    smoothing_window_end: usize,
    smoothing_window_step: usize,
    slow_length_start: usize,
    slow_length_end: usize,
    slow_length_step: usize,
    fast_length_start: usize,
    fast_length_end: usize,
    fast_length_step: usize,
    resolution_start: usize,
    resolution_end: usize,
    resolution_step: usize,
    ma_type: String,
) -> Result<usize, JsValue> {
    if in_ptr.is_null()
        || out_value_ptr.is_null()
        || out_slow_ma_ptr.is_null()
        || out_fast_ma_ptr.is_null()
        || out_forecast_ptr.is_null()
        || out_upper_ptr.is_null()
        || out_lower_ptr.is_null()
        || out_direction_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to moving_average_cross_probability_batch_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let sweep = MovingAverageCrossProbabilityBatchRange {
            smoothing_window: (
                smoothing_window_start,
                smoothing_window_end,
                smoothing_window_step,
            ),
            slow_length: (slow_length_start, slow_length_end, slow_length_step),
            fast_length: (fast_length_start, fast_length_end, fast_length_step),
            resolution: (resolution_start, resolution_end, resolution_step),
            ma_type: MovingAverageCrossProbabilityMaType::from_str(&ma_type)
                .map_err(|e| JsValue::from_str(&e))?,
        };
        let combos = moving_average_cross_probability_expand_grid(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let total = rows.checked_mul(len).ok_or_else(|| {
            JsValue::from_str("rows*cols overflow in moving_average_cross_probability_batch_into")
        })?;
        let out_value = std::slice::from_raw_parts_mut(out_value_ptr, total);
        let out_slow_ma = std::slice::from_raw_parts_mut(out_slow_ma_ptr, total);
        let out_fast_ma = std::slice::from_raw_parts_mut(out_fast_ma_ptr, total);
        let out_forecast = std::slice::from_raw_parts_mut(out_forecast_ptr, total);
        let out_upper = std::slice::from_raw_parts_mut(out_upper_ptr, total);
        let out_lower = std::slice::from_raw_parts_mut(out_lower_ptr, total);
        let out_direction = std::slice::from_raw_parts_mut(out_direction_ptr, total);
        moving_average_cross_probability_batch_into_slice(
            out_value,
            out_slow_ma,
            out_fast_ma,
            out_forecast,
            out_upper,
            out_lower,
            out_direction,
            data,
            &sweep,
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn moving_average_cross_probability_output_into_js(
    data: &[f64],
    ma_type: String,
    smoothing_window: usize,
    slow_length: usize,
    fast_length: usize,
    resolution: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = moving_average_cross_probability_js(
        data,
        ma_type,
        smoothing_window,
        slow_length,
        fast_length,
        resolution,
    )?;
    crate::write_wasm_object_f64_outputs(
        "moving_average_cross_probability_output_into_js",
        &value,
        out,
    )
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn moving_average_cross_probability_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = moving_average_cross_probability_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "moving_average_cross_probability_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data(len: usize) -> Vec<f64> {
        (0..len)
            .map(|i| {
                let x = i as f64;
                100.0 + x * 0.13 + (x * 0.11).sin() * 2.4 + (x * 0.03).cos() * 0.7
            })
            .collect()
    }

    fn sample_candles(len: usize) -> Candles {
        let close = sample_data(len);
        let open = close.iter().map(|v| v - 0.3).collect::<Vec<_>>();
        let high = close.iter().map(|v| v + 0.8).collect::<Vec<_>>();
        let low = close.iter().map(|v| v - 0.9).collect::<Vec<_>>();
        let volume = vec![1_000.0; len];
        let timestamp = (0..len as i64).collect::<Vec<_>>();
        Candles::new(timestamp, open, high, low, close, volume)
    }

    fn assert_vec_close(lhs: &[f64], rhs: &[f64]) {
        assert_eq!(lhs.len(), rhs.len());
        for (idx, (a, b)) in lhs.iter().zip(rhs.iter()).enumerate() {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            let diff = (a - b).abs();
            assert!(diff <= 1e-9, "mismatch at {idx}: {a} vs {b}");
        }
    }

    fn sma(data: &[f64], period: usize) -> Vec<f64> {
        let mut out = vec![f64::NAN; data.len()];
        let mut sum = 0.0;
        for i in 0..data.len() {
            sum += data[i];
            if i + 1 >= period {
                if i + 1 > period {
                    sum -= data[i - period];
                }
                out[i] = sum / period as f64;
            }
        }
        out
    }

    fn ema(data: &[f64], period: usize) -> Vec<f64> {
        let mut out = vec![f64::NAN; data.len()];
        let alpha = 2.0 / (period as f64 + 1.0);
        let mut seed_sum = 0.0;
        let mut current = f64::NAN;
        for i in 0..data.len() {
            seed_sum += data[i];
            if i + 1 < period {
                continue;
            }
            if i + 1 == period {
                current = seed_sum / period as f64;
            } else {
                current = alpha.mul_add(data[i], (1.0 - alpha) * current);
            }
            out[i] = current;
        }
        out
    }

    fn wma_window(data: &[f64]) -> f64 {
        let mut num = 0.0;
        let mut den = 0.0;
        for (idx, value) in data.iter().enumerate() {
            let w = (idx + 1) as f64;
            num += *value * w;
            den += w;
        }
        num / den
    }

    fn hma(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let half = period / 2;
        let sqrt_len = (period as f64).sqrt().floor() as usize;
        let mut diff = vec![f64::NAN; n];
        let mut out = vec![f64::NAN; n];
        for i in 0..n {
            if i + 1 >= period && i + 1 >= half {
                let full = wma_window(&data[i + 1 - period..=i]);
                let half_val = wma_window(&data[i + 1 - half..=i]);
                diff[i] = 2.0 * half_val - full;
            }
            if i + 1 >= period + sqrt_len - 1 {
                let start = i + 1 - sqrt_len;
                let window = diff[start..=i].iter().copied().collect::<Vec<_>>();
                if window.iter().all(|v| v.is_finite()) {
                    out[i] = wma_window(&window);
                }
            }
        }
        out
    }

    fn stddev4(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut out = vec![f64::NAN; n];
        for i in period.saturating_sub(1)..n {
            let window = &data[i + 1 - period..=i];
            let mean = window.iter().sum::<f64>() / period as f64;
            let var = window
                .iter()
                .map(|v| {
                    let d = *v - mean;
                    d * d
                })
                .sum::<f64>()
                / period as f64;
            out[i] = var.sqrt() * 4.0;
        }
        out
    }

    fn array_sma(temp: &[f64], period: usize) -> f64 {
        temp[..period].iter().sum::<f64>() / period as f64
    }

    fn array_ema(temp: &[f64], period: usize) -> f64 {
        let alpha = 2.0 / (period as f64 + 1.0);
        let mut ema = *temp.last().unwrap();
        for idx in (0..temp.len() - 1).rev() {
            ema = alpha.mul_add(temp[idx], (1.0 - alpha) * ema);
        }
        ema
    }

    fn manual_reference(
        data: &[f64],
        ma_type: MovingAverageCrossProbabilityMaType,
        smoothing_window: usize,
        slow_length: usize,
        fast_length: usize,
        resolution: usize,
    ) -> MovingAverageCrossProbabilityOutput {
        let n = data.len();
        let mut value = vec![f64::NAN; n];
        let slow_ma = match ma_type {
            MovingAverageCrossProbabilityMaType::Ema => ema(data, slow_length),
            MovingAverageCrossProbabilityMaType::Sma => sma(data, slow_length),
        };
        let fast_ma = match ma_type {
            MovingAverageCrossProbabilityMaType::Ema => ema(data, fast_length),
            MovingAverageCrossProbabilityMaType::Sma => sma(data, fast_length),
        };
        let price = hma(data, smoothing_window);
        let stdev = stddev4(data, smoothing_window);
        let mut forecast = vec![f64::NAN; n];
        let mut upper = vec![f64::NAN; n];
        let mut lower = vec![f64::NAN; n];
        let mut direction = vec![f64::NAN; n];

        for i in 0..n {
            if slow_ma[i].is_finite() && fast_ma[i].is_finite() {
                direction[i] = if fast_ma[i] > slow_ma[i] { -1.0 } else { 1.0 };
            }
            if i > 0 && price[i].is_finite() && price[i - 1].is_finite() && stdev[i].is_finite() {
                forecast[i] = price[i] + (price[i] - price[i - 1]);
                upper[i] = forecast[i] + stdev[i];
                lower[i] = forecast[i] - stdev[i];
            }
            if i < 2 * slow_length
                || !direction[i].is_finite()
                || !upper[i].is_finite()
                || !lower[i].is_finite()
            {
                continue;
            }
            let mut memory = Vec::with_capacity(2 * slow_length + 1);
            for j in 0..=2 * slow_length {
                memory.push(data[i - j]);
            }
            let seg = (upper[i] - lower[i]) / (resolution - 1) as f64;
            let mut hits = 0usize;
            for k in 0..resolution {
                let possibility = lower[i] + seg * k as f64;
                let mut temp = Vec::with_capacity(memory.len() + 1);
                temp.push(possibility);
                temp.extend_from_slice(&memory);
                let slow_future = match ma_type {
                    MovingAverageCrossProbabilityMaType::Ema => array_ema(&temp, slow_length),
                    MovingAverageCrossProbabilityMaType::Sma => array_sma(&temp, slow_length),
                };
                let fast_future = match ma_type {
                    MovingAverageCrossProbabilityMaType::Ema => array_ema(&temp, fast_length),
                    MovingAverageCrossProbabilityMaType::Sma => array_sma(&temp, fast_length),
                };
                let crossed = if direction[i] < 0.0 {
                    slow_future > fast_future
                } else {
                    slow_future <= fast_future
                };
                if crossed {
                    hits += 1;
                }
            }
            value[i] = 100.0 * hits as f64 / resolution as f64;
        }

        MovingAverageCrossProbabilityOutput {
            value,
            slow_ma,
            fast_ma,
            forecast,
            upper,
            lower,
            direction,
        }
    }

    #[test]
    fn manual_reference_matches_ema_single() {
        let data = sample_data(220);
        let input = MovingAverageCrossProbabilityInput::from_slice(
            &data,
            MovingAverageCrossProbabilityParams::default(),
        );
        let out = moving_average_cross_probability(&input).unwrap();
        let expected = manual_reference(&data, DEFAULT_MA_TYPE, 7, 30, 14, 50);
        assert_vec_close(&out.value, &expected.value);
        assert_vec_close(&out.slow_ma, &expected.slow_ma);
        assert_vec_close(&out.fast_ma, &expected.fast_ma);
        assert_vec_close(&out.forecast, &expected.forecast);
        assert_vec_close(&out.upper, &expected.upper);
        assert_vec_close(&out.lower, &expected.lower);
        assert_vec_close(&out.direction, &expected.direction);
    }

    #[test]
    fn manual_reference_matches_sma_single() {
        let data = sample_data(220);
        let params = MovingAverageCrossProbabilityParams {
            ma_type: Some(MovingAverageCrossProbabilityMaType::Sma),
            ..MovingAverageCrossProbabilityParams::default()
        };
        let input = MovingAverageCrossProbabilityInput::from_slice(&data, params);
        let out = moving_average_cross_probability(&input).unwrap();
        let expected = manual_reference(
            &data,
            MovingAverageCrossProbabilityMaType::Sma,
            7,
            30,
            14,
            50,
        );
        assert_vec_close(&out.value, &expected.value);
    }

    #[test]
    fn stream_matches_batch() {
        let data = sample_data(200);
        let params = MovingAverageCrossProbabilityParams {
            ma_type: Some(MovingAverageCrossProbabilityMaType::Sma),
            smoothing_window: Some(8),
            slow_length: Some(26),
            fast_length: Some(11),
            resolution: Some(40),
        };
        let input = MovingAverageCrossProbabilityInput::from_slice(&data, params.clone());
        let batch = moving_average_cross_probability(&input).unwrap();
        let mut stream = MovingAverageCrossProbabilityStream::try_new(params).unwrap();
        let mut value = Vec::with_capacity(data.len());
        let mut slow_ma = Vec::with_capacity(data.len());
        let mut fast_ma = Vec::with_capacity(data.len());
        let mut forecast = Vec::with_capacity(data.len());
        let mut upper = Vec::with_capacity(data.len());
        let mut lower = Vec::with_capacity(data.len());
        let mut direction = Vec::with_capacity(data.len());
        for item in data {
            let (v, s, f, fc, u, l, d) = stream.update(item);
            value.push(v);
            slow_ma.push(s);
            fast_ma.push(f);
            forecast.push(fc);
            upper.push(u);
            lower.push(l);
            direction.push(d);
        }
        assert_vec_close(&value, &batch.value);
        assert_vec_close(&slow_ma, &batch.slow_ma);
        assert_vec_close(&fast_ma, &batch.fast_ma);
        assert_vec_close(&forecast, &batch.forecast);
        assert_vec_close(&upper, &batch.upper);
        assert_vec_close(&lower, &batch.lower);
        assert_vec_close(&direction, &batch.direction);
    }

    #[test]
    fn batch_first_row_matches_single() {
        let data = sample_data(180);
        let sweep = MovingAverageCrossProbabilityBatchRange {
            smoothing_window: (7, 8, 1),
            slow_length: (30, 30, 0),
            fast_length: (14, 14, 0),
            resolution: (50, 50, 0),
            ma_type: MovingAverageCrossProbabilityMaType::Ema,
        };
        let batch = moving_average_cross_probability_batch_with_kernel(&data, &sweep, Kernel::Auto)
            .unwrap();
        let input = MovingAverageCrossProbabilityInput::from_slice(
            &data,
            MovingAverageCrossProbabilityParams::default(),
        );
        let single = moving_average_cross_probability(&input).unwrap();
        assert_eq!(batch.rows, 2);
        assert_eq!(batch.cols, data.len());
        assert_vec_close(&batch.value[..data.len()], &single.value);
        assert_vec_close(&batch.slow_ma[..data.len()], &single.slow_ma);
        assert_vec_close(&batch.fast_ma[..data.len()], &single.fast_ma);
    }

    #[test]
    fn invalid_length_order_fails() {
        let data = sample_data(96);
        let input = MovingAverageCrossProbabilityInput::from_slice(
            &data,
            MovingAverageCrossProbabilityParams {
                slow_length: Some(10),
                fast_length: Some(14),
                ..MovingAverageCrossProbabilityParams::default()
            },
        );
        let err = moving_average_cross_probability(&input).unwrap_err();
        assert!(matches!(
            err,
            MovingAverageCrossProbabilityError::InvalidLengthOrder { .. }
        ));
    }

    #[test]
    fn cpu_dispatch_matches_direct() {
        let candles = sample_candles(180);
        let combos = [IndicatorParamSet {
            params: &[
                ParamKV {
                    key: "ma_type",
                    value: ParamValue::EnumString("ema"),
                },
                ParamKV {
                    key: "smoothing_window",
                    value: ParamValue::Int(7),
                },
                ParamKV {
                    key: "slow_length",
                    value: ParamValue::Int(30),
                },
                ParamKV {
                    key: "fast_length",
                    value: ParamValue::Int(14),
                },
                ParamKV {
                    key: "resolution",
                    value: ParamValue::Int(50),
                },
            ],
        }];
        let dispatched = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "moving_average_cross_probability",
            output_id: Some("value"),
            data: IndicatorDataRef::Candles {
                candles: &candles,
                source: Some("close"),
            },
            combos: &combos,
            kernel: Kernel::Auto,
        })
        .unwrap();
        let direct =
            moving_average_cross_probability(&MovingAverageCrossProbabilityInput::from_candles(
                &candles,
                MovingAverageCrossProbabilityParams::default(),
            ))
            .unwrap();
        assert_vec_close(dispatched.values_f64.as_ref().unwrap(), &direct.value);
    }
}
