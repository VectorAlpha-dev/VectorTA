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
use crate::utilities::helpers::{alloc_with_nan_prefix, detect_best_batch_kernel};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::collections::VecDeque;
use std::sync::OnceLock;
use thiserror::Error;

const DEFAULT_PERIOD: usize = 100;
const DEFAULT_STEEPNESS: f64 = 2.5;
const DEFAULT_MA_TYPE: &str = "ema";
const DEFAULT_SMOOTH: usize = 10;
const DEFAULT_MOMENTUM_WEIGHT: f64 = 1.2;
const DEFAULT_LONG_THRESHOLD: f64 = 0.5;
const DEFAULT_SHORT_THRESHOLD: f64 = -0.5;
const SLOPE_LOOKBACK: usize = 10;
const ANNUALIZATION: f64 = 252.0;

#[derive(Debug, Clone)]
pub enum LogarithmicMovingAverageData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice {
        data: &'a [f64],
        volume: Option<&'a [f64]>,
    },
}

#[derive(Debug, Clone)]
pub struct LogarithmicMovingAverageOutput {
    pub lma: Vec<f64>,
    pub signal: Vec<f64>,
    pub position: Vec<f64>,
    pub momentum_confirmed: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct LogarithmicMovingAverageParams {
    pub period: Option<usize>,
    pub steepness: Option<f64>,
    pub ma_type: Option<String>,
    pub smooth: Option<usize>,
    pub momentum_weight: Option<f64>,
    pub long_threshold: Option<f64>,
    pub short_threshold: Option<f64>,
}

impl Default for LogarithmicMovingAverageParams {
    fn default() -> Self {
        Self {
            period: Some(DEFAULT_PERIOD),
            steepness: Some(DEFAULT_STEEPNESS),
            ma_type: Some(DEFAULT_MA_TYPE.to_string()),
            smooth: Some(DEFAULT_SMOOTH),
            momentum_weight: Some(DEFAULT_MOMENTUM_WEIGHT),
            long_threshold: Some(DEFAULT_LONG_THRESHOLD),
            short_threshold: Some(DEFAULT_SHORT_THRESHOLD),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LogarithmicMovingAverageInput<'a> {
    pub data: LogarithmicMovingAverageData<'a>,
    pub params: LogarithmicMovingAverageParams,
}

impl<'a> LogarithmicMovingAverageInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: LogarithmicMovingAverageParams,
    ) -> Self {
        Self {
            data: LogarithmicMovingAverageData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(data: &'a [f64], params: LogarithmicMovingAverageParams) -> Self {
        Self {
            data: LogarithmicMovingAverageData::Slice { data, volume: None },
            params,
        }
    }

    #[inline]
    pub fn from_slice_with_volume(
        data: &'a [f64],
        volume: &'a [f64],
        params: LogarithmicMovingAverageParams,
    ) -> Self {
        Self {
            data: LogarithmicMovingAverageData::Slice {
                data,
                volume: Some(volume),
            },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", LogarithmicMovingAverageParams::default())
    }

    #[inline(always)]
    fn prices(&self) -> &'a [f64] {
        match &self.data {
            LogarithmicMovingAverageData::Candles { candles, source } => {
                logarithmic_moving_average_source_type(candles, source)
            }
            LogarithmicMovingAverageData::Slice { data, .. } => data,
        }
    }

    #[inline(always)]
    fn volumes(&self) -> Option<&'a [f64]> {
        match &self.data {
            LogarithmicMovingAverageData::Candles { candles, .. } => Some(&candles.volume),
            LogarithmicMovingAverageData::Slice { volume, .. } => *volume,
        }
    }

    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(DEFAULT_PERIOD)
    }

    #[inline]
    pub fn get_steepness(&self) -> f64 {
        self.params.steepness.unwrap_or(DEFAULT_STEEPNESS)
    }

    #[inline]
    pub fn ma_type_str(&self) -> &str {
        self.params.ma_type.as_deref().unwrap_or(DEFAULT_MA_TYPE)
    }

    #[inline]
    pub fn get_smooth(&self) -> usize {
        self.params.smooth.unwrap_or(DEFAULT_SMOOTH)
    }

    #[inline]
    pub fn get_momentum_weight(&self) -> f64 {
        self.params
            .momentum_weight
            .unwrap_or(DEFAULT_MOMENTUM_WEIGHT)
    }

    #[inline]
    pub fn get_long_threshold(&self) -> f64 {
        self.params.long_threshold.unwrap_or(DEFAULT_LONG_THRESHOLD)
    }

    #[inline]
    pub fn get_short_threshold(&self) -> f64 {
        self.params
            .short_threshold
            .unwrap_or(DEFAULT_SHORT_THRESHOLD)
    }
}

#[inline(always)]
fn logarithmic_moving_average_source_type<'a>(candles: &'a Candles, source: &str) -> &'a [f64] {
    match source {
        "close" => &candles.close,
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

#[derive(Copy, Clone, Debug)]
pub struct LogarithmicMovingAverageBuilder {
    period: Option<usize>,
    steepness: Option<f64>,
    smooth: Option<usize>,
    momentum_weight: Option<f64>,
    long_threshold: Option<f64>,
    short_threshold: Option<f64>,
    kernel: Kernel,
}

impl Default for LogarithmicMovingAverageBuilder {
    fn default() -> Self {
        Self {
            period: None,
            steepness: None,
            smooth: None,
            momentum_weight: None,
            long_threshold: None,
            short_threshold: None,
            kernel: Kernel::Auto,
        }
    }
}

impl LogarithmicMovingAverageBuilder {
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
    pub fn steepness(mut self, value: f64) -> Self {
        self.steepness = Some(value);
        self
    }

    #[inline(always)]
    pub fn smooth(mut self, value: usize) -> Self {
        self.smooth = Some(value);
        self
    }

    #[inline(always)]
    pub fn momentum_weight(mut self, value: f64) -> Self {
        self.momentum_weight = Some(value);
        self
    }

    #[inline(always)]
    pub fn long_threshold(mut self, value: f64) -> Self {
        self.long_threshold = Some(value);
        self
    }

    #[inline(always)]
    pub fn short_threshold(mut self, value: f64) -> Self {
        self.short_threshold = Some(value);
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
    ) -> Result<LogarithmicMovingAverageOutput, LogarithmicMovingAverageError> {
        let input = LogarithmicMovingAverageInput::from_candles(
            candles,
            "close",
            LogarithmicMovingAverageParams {
                period: self.period,
                steepness: self.steepness,
                ma_type: Some(DEFAULT_MA_TYPE.to_string()),
                smooth: self.smooth,
                momentum_weight: self.momentum_weight,
                long_threshold: self.long_threshold,
                short_threshold: self.short_threshold,
            },
        );
        logarithmic_moving_average_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<LogarithmicMovingAverageOutput, LogarithmicMovingAverageError> {
        let input = LogarithmicMovingAverageInput::from_slice(
            data,
            LogarithmicMovingAverageParams {
                period: self.period,
                steepness: self.steepness,
                ma_type: Some(DEFAULT_MA_TYPE.to_string()),
                smooth: self.smooth,
                momentum_weight: self.momentum_weight,
                long_threshold: self.long_threshold,
                short_threshold: self.short_threshold,
            },
        );
        logarithmic_moving_average_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice_with_volume(
        self,
        data: &[f64],
        volume: &[f64],
        ma_type: &str,
    ) -> Result<LogarithmicMovingAverageOutput, LogarithmicMovingAverageError> {
        let input = LogarithmicMovingAverageInput::from_slice_with_volume(
            data,
            volume,
            LogarithmicMovingAverageParams {
                period: self.period,
                steepness: self.steepness,
                ma_type: Some(ma_type.to_string()),
                smooth: self.smooth,
                momentum_weight: self.momentum_weight,
                long_threshold: self.long_threshold,
                short_threshold: self.short_threshold,
            },
        );
        logarithmic_moving_average_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream<T: Into<String>>(
        self,
        ma_type: T,
    ) -> Result<LogarithmicMovingAverageStream, LogarithmicMovingAverageError> {
        LogarithmicMovingAverageStream::try_new(LogarithmicMovingAverageParams {
            period: self.period,
            steepness: self.steepness,
            ma_type: Some(ma_type.into()),
            smooth: self.smooth,
            momentum_weight: self.momentum_weight,
            long_threshold: self.long_threshold,
            short_threshold: self.short_threshold,
        })
    }
}

#[derive(Debug, Error)]
pub enum LogarithmicMovingAverageError {
    #[error("logarithmic_moving_average: Input data slice is empty.")]
    EmptyInputData,
    #[error("logarithmic_moving_average: All values are NaN.")]
    AllValuesNaN,
    #[error(
        "logarithmic_moving_average: Invalid period: period = {period}, data length = {data_len}"
    )]
    InvalidPeriod { period: usize, data_len: usize },
    #[error(
        "logarithmic_moving_average: Invalid smooth length: smooth = {smooth}, data length = {data_len}"
    )]
    InvalidSmooth { smooth: usize, data_len: usize },
    #[error("logarithmic_moving_average: Invalid steepness: {steepness}")]
    InvalidSteepness { steepness: f64 },
    #[error("logarithmic_moving_average: Invalid MA type: {ma_type}")]
    InvalidMaType { ma_type: String },
    #[error("logarithmic_moving_average: Invalid momentum_weight: {momentum_weight}")]
    InvalidMomentumWeight { momentum_weight: f64 },
    #[error(
        "logarithmic_moving_average: Invalid thresholds: long_threshold = {long_threshold}, short_threshold = {short_threshold}"
    )]
    InvalidThresholds {
        long_threshold: f64,
        short_threshold: f64,
    },
    #[error(
        "logarithmic_moving_average: Data length mismatch: data = {data_len}, volume = {volume_len}"
    )]
    DataLengthMismatch { data_len: usize, volume_len: usize },
    #[error("logarithmic_moving_average: VWMA smoothing requires volume data.")]
    MissingVolumeForVwma,
    #[error(
        "logarithmic_moving_average: Not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "logarithmic_moving_average: Output length mismatch: expected = {expected}, got = {got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("logarithmic_moving_average: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("logarithmic_moving_average: Invalid kernel for batch path: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Clone, Debug)]
struct PreparedParams {
    period: usize,
    steepness: f64,
    ma_type: String,
    smooth: usize,
    momentum_weight: f64,
    long_threshold: f64,
    short_threshold: f64,
}

#[derive(Debug, Clone)]
pub struct LogarithmicMovingAverageBatchRange {
    pub period: (usize, usize, usize),
    pub steepness: (f64, f64, f64),
    pub smooth: (usize, usize, usize),
    pub momentum_weight: (f64, f64, f64),
    pub long_threshold: (f64, f64, f64),
    pub short_threshold: (f64, f64, f64),
    pub ma_type: String,
}

impl Default for LogarithmicMovingAverageBatchRange {
    fn default() -> Self {
        Self {
            period: (DEFAULT_PERIOD, DEFAULT_PERIOD, 0),
            steepness: (DEFAULT_STEEPNESS, DEFAULT_STEEPNESS, 0.0),
            smooth: (DEFAULT_SMOOTH, DEFAULT_SMOOTH, 0),
            momentum_weight: (DEFAULT_MOMENTUM_WEIGHT, DEFAULT_MOMENTUM_WEIGHT, 0.0),
            long_threshold: (DEFAULT_LONG_THRESHOLD, DEFAULT_LONG_THRESHOLD, 0.0),
            short_threshold: (DEFAULT_SHORT_THRESHOLD, DEFAULT_SHORT_THRESHOLD, 0.0),
            ma_type: DEFAULT_MA_TYPE.to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LogarithmicMovingAverageBatchOutput {
    pub lma: Vec<f64>,
    pub signal: Vec<f64>,
    pub position: Vec<f64>,
    pub momentum_confirmed: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
    pub combos: Vec<LogarithmicMovingAverageParams>,
}

#[derive(Clone, Debug)]
pub struct LogarithmicMovingAverageBatchBuilder {
    range: LogarithmicMovingAverageBatchRange,
    kernel: Kernel,
}

impl Default for LogarithmicMovingAverageBatchBuilder {
    fn default() -> Self {
        Self {
            range: LogarithmicMovingAverageBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl LogarithmicMovingAverageBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn period(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.period = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn steepness(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.steepness = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn smooth(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.smooth = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn momentum_weight(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.momentum_weight = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn long_threshold(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.long_threshold = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn short_threshold(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.short_threshold = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn ma_type<T: Into<String>>(mut self, value: T) -> Self {
        self.range.ma_type = value.into();
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<LogarithmicMovingAverageBatchOutput, LogarithmicMovingAverageError> {
        logarithmic_moving_average_batch_with_kernel(data, None, &self.range, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice_with_volume(
        self,
        data: &[f64],
        volume: &[f64],
    ) -> Result<LogarithmicMovingAverageBatchOutput, LogarithmicMovingAverageError> {
        logarithmic_moving_average_batch_with_kernel(data, Some(volume), &self.range, self.kernel)
    }

    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<LogarithmicMovingAverageBatchOutput, LogarithmicMovingAverageError> {
        logarithmic_moving_average_batch_with_kernel(
            &candles.close,
            Some(&candles.volume),
            &self.range,
            self.kernel,
        )
    }
}

#[derive(Debug, Clone)]
pub struct LogarithmicMovingAverageStream {
    params: LogarithmicMovingAverageParams,
    data: Vec<f64>,
    volume: Vec<f64>,
}

impl LogarithmicMovingAverageStream {
    pub fn try_new(
        params: LogarithmicMovingAverageParams,
    ) -> Result<Self, LogarithmicMovingAverageError> {
        let _ = prepare_param_values(
            params.period.unwrap_or(DEFAULT_PERIOD),
            params.steepness.unwrap_or(DEFAULT_STEEPNESS),
            params.ma_type.as_deref().unwrap_or(DEFAULT_MA_TYPE),
            params.smooth.unwrap_or(DEFAULT_SMOOTH),
            params.momentum_weight.unwrap_or(DEFAULT_MOMENTUM_WEIGHT),
            params.long_threshold.unwrap_or(DEFAULT_LONG_THRESHOLD),
            params.short_threshold.unwrap_or(DEFAULT_SHORT_THRESHOLD),
        )?;
        Ok(Self {
            params,
            data: Vec::new(),
            volume: Vec::new(),
        })
    }

    pub fn update(&mut self, value: f64, volume: Option<f64>) -> Option<(f64, f64, f64, f64)> {
        self.data.push(value);
        self.volume.push(volume.unwrap_or(f64::NAN));
        let input = if self.volume.iter().all(|v| v.is_nan()) {
            LogarithmicMovingAverageInput::from_slice(&self.data, self.params.clone())
        } else {
            LogarithmicMovingAverageInput::from_slice_with_volume(
                &self.data,
                &self.volume,
                self.params.clone(),
            )
        };
        let out = logarithmic_moving_average(&input).ok()?;
        let idx = out.signal.len().checked_sub(1)?;
        let lma = *out.lma.get(idx)?;
        let signal = *out.signal.get(idx)?;
        let position = *out.position.get(idx)?;
        let momentum_confirmed = *out.momentum_confirmed.get(idx)?;
        if lma.is_nan() || signal.is_nan() || position.is_nan() || momentum_confirmed.is_nan() {
            None
        } else {
            Some((lma, signal, position, momentum_confirmed))
        }
    }
}

#[inline]
pub fn logarithmic_moving_average(
    input: &LogarithmicMovingAverageInput,
) -> Result<LogarithmicMovingAverageOutput, LogarithmicMovingAverageError> {
    logarithmic_moving_average_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn longest_finite_run(data: &[f64]) -> usize {
    let mut best = 0usize;
    let mut cur = 0usize;
    for &value in data {
        if value.is_finite() {
            cur += 1;
            best = best.max(cur);
        } else {
            cur = 0;
        }
    }
    best
}

#[inline(always)]
fn normalize_ma_type(value: &str) -> Result<String, LogarithmicMovingAverageError> {
    let normalized = value.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "ema" | "sma" | "wma" | "rma" | "vwma" => Ok(normalized),
        _ => Err(LogarithmicMovingAverageError::InvalidMaType {
            ma_type: value.to_string(),
        }),
    }
}

fn prepare_input(
    input: &LogarithmicMovingAverageInput,
) -> Result<PreparedParams, LogarithmicMovingAverageError> {
    let prepared = prepare_param_values(
        input.get_period(),
        input.get_steepness(),
        input.ma_type_str(),
        input.get_smooth(),
        input.get_momentum_weight(),
        input.get_long_threshold(),
        input.get_short_threshold(),
    )?;
    let prices = input.prices();
    if prices.is_empty() {
        return Err(LogarithmicMovingAverageError::EmptyInputData);
    }
    if prices.iter().all(|x| x.is_nan()) {
        return Err(LogarithmicMovingAverageError::AllValuesNaN);
    }

    if prepared.period > prices.len() {
        return Err(LogarithmicMovingAverageError::InvalidPeriod {
            period: prepared.period,
            data_len: prices.len(),
        });
    }
    if prepared.smooth > prices.len() {
        return Err(LogarithmicMovingAverageError::InvalidSmooth {
            smooth: prepared.smooth,
            data_len: prices.len(),
        });
    }
    if prepared.ma_type == "vwma" {
        let volume = input
            .volumes()
            .ok_or(LogarithmicMovingAverageError::MissingVolumeForVwma)?;
        if volume.len() != prices.len() {
            return Err(LogarithmicMovingAverageError::DataLengthMismatch {
                data_len: prices.len(),
                volume_len: volume.len(),
            });
        }
    } else if let Some(volume) = input.volumes() {
        if !volume.is_empty() && volume.len() != prices.len() {
            return Err(LogarithmicMovingAverageError::DataLengthMismatch {
                data_len: prices.len(),
                volume_len: volume.len(),
            });
        }
    }

    let longest = longest_finite_run(prices);
    if longest < prepared.period {
        return Err(LogarithmicMovingAverageError::NotEnoughValidData {
            needed: prepared.period,
            valid: longest,
        });
    }

    Ok(prepared)
}

fn prepare_param_values(
    period: usize,
    steepness: f64,
    ma_type: &str,
    smooth: usize,
    momentum_weight: f64,
    long_threshold: f64,
    short_threshold: f64,
) -> Result<PreparedParams, LogarithmicMovingAverageError> {
    if period == 0 {
        return Err(LogarithmicMovingAverageError::InvalidPeriod {
            period,
            data_len: 0,
        });
    }
    if smooth == 0 {
        return Err(LogarithmicMovingAverageError::InvalidSmooth {
            smooth,
            data_len: 0,
        });
    }
    if !steepness.is_finite() || steepness <= 0.0 {
        return Err(LogarithmicMovingAverageError::InvalidSteepness { steepness });
    }
    if !momentum_weight.is_finite() || momentum_weight <= 0.0 {
        return Err(LogarithmicMovingAverageError::InvalidMomentumWeight { momentum_weight });
    }
    if !long_threshold.is_finite()
        || !short_threshold.is_finite()
        || long_threshold <= short_threshold
    {
        return Err(LogarithmicMovingAverageError::InvalidThresholds {
            long_threshold,
            short_threshold,
        });
    }
    Ok(PreparedParams {
        period,
        steepness,
        ma_type: normalize_ma_type(ma_type)?,
        smooth,
        momentum_weight,
        long_threshold,
        short_threshold,
    })
}

#[inline(always)]
fn compute_weights(period: usize, steepness: f64) -> (Vec<f64>, f64) {
    let mut weights = Vec::with_capacity(period);
    let mut total = 0.0;
    for i in 0..period {
        let log_arg = ((i as f64) + steepness).max(2.0);
        let weight = 1.0 / log_arg.ln().powi(2);
        weights.push(weight);
        total += weight;
    }
    (weights, total)
}

fn default_weights() -> &'static (Vec<f64>, f64) {
    static WEIGHTS: OnceLock<(Vec<f64>, f64)> = OnceLock::new();
    WEIGHTS.get_or_init(|| compute_weights(DEFAULT_PERIOD, DEFAULT_STEEPNESS))
}

fn compute_lma(prices: &[f64], period: usize, steepness: f64, out: &mut [f64]) {
    let owned_weights;
    let (weights, total_weight) =
        if period == DEFAULT_PERIOD && steepness.to_bits() == DEFAULT_STEEPNESS.to_bits() {
            let cached = default_weights();
            (cached.0.as_slice(), cached.1)
        } else {
            owned_weights = compute_weights(period, steepness);
            (owned_weights.0.as_slice(), owned_weights.1)
        };
    let mut run = 0usize;
    for (i, &price) in prices.iter().enumerate() {
        if price.is_finite() {
            run += 1;
        } else {
            run = 0;
            continue;
        }
        if run < period {
            continue;
        }
        let mut acc = 0.0;
        let mut k = 0usize;
        unsafe {
            let price_ptr = prices.as_ptr().add(i);
            let weight_ptr = weights.as_ptr();
            while k + 7 < period {
                acc += *price_ptr.sub(k) * *weight_ptr.add(k)
                    + *price_ptr.sub(k + 1) * *weight_ptr.add(k + 1)
                    + *price_ptr.sub(k + 2) * *weight_ptr.add(k + 2)
                    + *price_ptr.sub(k + 3) * *weight_ptr.add(k + 3)
                    + *price_ptr.sub(k + 4) * *weight_ptr.add(k + 4)
                    + *price_ptr.sub(k + 5) * *weight_ptr.add(k + 5)
                    + *price_ptr.sub(k + 6) * *weight_ptr.add(k + 6)
                    + *price_ptr.sub(k + 7) * *weight_ptr.add(k + 7);
                k += 8;
            }
            while k < period {
                acc += *price_ptr.sub(k) * *weight_ptr.add(k);
                k += 1;
            }
        }
        out[i] = acc / total_weight;
    }
}

fn compute_log_momentum(prices: &[f64], period: usize, out: &mut [f64]) {
    let mut ring = vec![0.0; period];
    let mut head = 0usize;
    let mut count = 0usize;
    let mut sum = 0.0;

    for i in 1..prices.len() {
        let prev = prices[i - 1];
        let curr = prices[i];
        if !prev.is_finite() || !curr.is_finite() || prev <= 0.0 || curr <= 0.0 {
            head = 0;
            count = 0;
            sum = 0.0;
            continue;
        }
        let ret = (curr / prev).ln();
        if count < period {
            ring[count] = ret;
            count += 1;
            sum += ret;
            if count < period {
                continue;
            }
            head = 0;
        } else {
            let old = ring[head];
            sum -= old;
            ring[head] = ret;
            sum += ret;
            head += 1;
            if head == period {
                head = 0;
            }
        }
        out[i] = sum * ANNUALIZATION / (period as f64);
    }
}

fn compute_r_squared(prices: &[f64], period: usize, out: &mut [f64]) {
    let sum_x = (period * (period - 1) / 2) as f64;
    let sum_x2 = ((period - 1) * period * (2 * period - 1) / 6) as f64;
    let mut window: VecDeque<f64> = VecDeque::with_capacity(period);
    let mut sum_y = 0.0;
    let mut sum_y2 = 0.0;
    let mut sum_xy = 0.0;

    for (i, &price) in prices.iter().enumerate() {
        if !price.is_finite() || price <= 0.0 {
            window.clear();
            sum_y = 0.0;
            sum_y2 = 0.0;
            sum_xy = 0.0;
            continue;
        }

        let y = price.ln();
        if window.len() < period {
            window.push_back(y);
            let idx = (window.len() - 1) as f64;
            sum_y += y;
            sum_y2 += y * y;
            sum_xy += idx * y;
            if window.len() < period {
                continue;
            }
        } else {
            let oldest = window.pop_front().unwrap();
            let prev_sum_y = sum_y;
            sum_y -= oldest;
            sum_y2 -= oldest * oldest;
            sum_xy = sum_xy - prev_sum_y + oldest + ((period - 1) as f64) * y;
            window.push_back(y);
            sum_y += y;
            sum_y2 += y * y;
        }

        if period <= 10 {
            out[i] = 0.0;
            continue;
        }

        let n = period as f64;
        let denom_y = n * sum_y2 - sum_y * sum_y;
        let denom = ((n * sum_x2 - sum_x * sum_x) * denom_y).sqrt();
        let correlation = if denom.is_finite() && denom != 0.0 {
            ((n * sum_xy - sum_x * sum_y) / denom).clamp(-1.0, 1.0)
        } else {
            0.0
        };
        out[i] = (correlation * correlation).clamp(0.0, 1.0);
    }
}

fn compute_raw_signal(
    lma: &[f64],
    log_momentum: &[f64],
    r_squared: &[f64],
    momentum_weight: f64,
    out: &mut [f64],
) {
    for i in 0..lma.len() {
        if i < SLOPE_LOOKBACK {
            continue;
        }
        let current = lma[i];
        let prev = lma[i - SLOPE_LOOKBACK];
        let momentum = log_momentum[i];
        let quality = r_squared[i];
        if !current.is_finite()
            || !prev.is_finite()
            || prev == 0.0
            || !momentum.is_finite()
            || !quality.is_finite()
        {
            continue;
        }
        let slope = ((current - prev) / prev) * 100.0;
        let mut signal = slope * (0.5 + quality * 0.5);
        if signal.signum() == momentum.signum() && momentum.abs() > 0.01 {
            signal *= momentum_weight;
        }
        out[i] = signal;
    }
}

fn smooth_sma(signal: &[f64], smooth: usize, out: &mut [f64]) {
    let mut ring = vec![0.0; smooth];
    let mut head = 0usize;
    let mut count = 0usize;
    let mut sum = 0.0;
    for (i, &value) in signal.iter().enumerate() {
        if !value.is_finite() {
            head = 0;
            count = 0;
            sum = 0.0;
            continue;
        }
        if count < smooth {
            ring[count] = value;
            count += 1;
            sum += value;
            if count < smooth {
                continue;
            }
            head = 0;
        } else {
            let old = ring[head];
            sum -= old;
            ring[head] = value;
            sum += value;
            head += 1;
            if head == smooth {
                head = 0;
            }
        }
        out[i] = sum / (smooth as f64);
    }
}

fn smooth_ema_like(signal: &[f64], smooth: usize, alpha: f64, out: &mut [f64]) {
    let mut window = VecDeque::with_capacity(smooth);
    let mut sum = 0.0;
    let mut seeded = false;
    let mut prev = f64::NAN;
    for (i, &value) in signal.iter().enumerate() {
        if !value.is_finite() {
            window.clear();
            sum = 0.0;
            seeded = false;
            prev = f64::NAN;
            continue;
        }
        if !seeded {
            window.push_back(value);
            sum += value;
            if window.len() < smooth {
                continue;
            }
            if window.len() > smooth {
                let old = window.pop_front().unwrap();
                sum -= old;
            }
            prev = sum / (smooth as f64);
            out[i] = prev;
            seeded = true;
            continue;
        }
        prev += alpha * (value - prev);
        out[i] = prev;
    }
}

fn smooth_wma(signal: &[f64], smooth: usize, out: &mut [f64]) {
    let mut ring = vec![0.0; smooth];
    let mut head = 0usize;
    let mut count = 0usize;
    let mut sum = 0.0;
    let mut weighted = 0.0;
    let denom = (smooth * (smooth + 1) / 2) as f64;

    for (i, &value) in signal.iter().enumerate() {
        if !value.is_finite() {
            head = 0;
            count = 0;
            sum = 0.0;
            weighted = 0.0;
            continue;
        }
        if count < smooth {
            ring[count] = value;
            count += 1;
            sum += value;
            weighted += (count as f64) * value;
            if count < smooth {
                continue;
            }
            head = 0;
        } else {
            let old = ring[head];
            let prev_sum = sum;
            sum -= old;
            ring[head] = value;
            sum += value;
            weighted = weighted - prev_sum + old + (smooth as f64) * value;
            head += 1;
            if head == smooth {
                head = 0;
            }
        }
        out[i] = weighted / denom;
    }
}

fn smooth_vwma(
    signal: &[f64],
    volume: &[f64],
    smooth: usize,
    out: &mut [f64],
) -> Result<(), LogarithmicMovingAverageError> {
    if signal.len() != volume.len() {
        return Err(LogarithmicMovingAverageError::DataLengthMismatch {
            data_len: signal.len(),
            volume_len: volume.len(),
        });
    }

    let mut ring_sv = vec![0.0; smooth];
    let mut ring_v = vec![0.0; smooth];
    let mut head = 0usize;
    let mut count = 0usize;
    let mut sum_sv = 0.0;
    let mut sum_v = 0.0;

    for i in 0..signal.len() {
        let s = signal[i];
        let v = volume[i];
        if !s.is_finite() || !v.is_finite() {
            head = 0;
            count = 0;
            sum_sv = 0.0;
            sum_v = 0.0;
            continue;
        }
        let sv = s * v;
        if count < smooth {
            ring_sv[count] = sv;
            ring_v[count] = v;
            count += 1;
            sum_sv += sv;
            sum_v += v;
            if count < smooth {
                continue;
            }
            head = 0;
        } else {
            sum_sv -= ring_sv[head];
            sum_v -= ring_v[head];
            ring_sv[head] = sv;
            ring_v[head] = v;
            sum_sv += sv;
            sum_v += v;
            head += 1;
            if head == smooth {
                head = 0;
            }
        }
        if sum_v != 0.0 {
            out[i] = sum_sv / sum_v;
        }
    }
    Ok(())
}

fn finalize_outputs(
    signal: &[f64],
    log_momentum: &[f64],
    long_threshold: f64,
    short_threshold: f64,
    out_position: &mut [f64],
    out_momentum_confirmed: &mut [f64],
) {
    for i in 0..signal.len() {
        let value = signal[i];
        if !value.is_finite() {
            continue;
        }
        out_position[i] = if value > long_threshold {
            1.0
        } else if value < short_threshold {
            -1.0
        } else {
            0.0
        };
        let momentum = log_momentum[i];
        if momentum.is_finite() {
            out_momentum_confirmed[i] = if value.signum() == momentum.signum()
                && value.abs() > long_threshold.abs() * 0.5
            {
                1.0
            } else {
                0.0
            };
        }
    }
}

fn logarithmic_moving_average_compute_into(
    prices: &[f64],
    volume: Option<&[f64]>,
    params: &PreparedParams,
    out_lma: &mut [f64],
    out_signal: &mut [f64],
    out_position: &mut [f64],
    out_momentum_confirmed: &mut [f64],
) -> Result<(), LogarithmicMovingAverageError> {
    out_lma.fill(f64::NAN);
    out_signal.fill(f64::NAN);
    out_position.fill(f64::NAN);
    out_momentum_confirmed.fill(f64::NAN);

    let mut log_momentum = alloc_with_nan_prefix(prices.len(), prices.len());
    let mut r_squared = alloc_with_nan_prefix(prices.len(), prices.len());
    let mut raw_signal = alloc_with_nan_prefix(prices.len(), prices.len());

    compute_lma(prices, params.period, params.steepness, out_lma);
    compute_log_momentum(prices, params.period, &mut log_momentum);
    compute_r_squared(prices, params.period, &mut r_squared);
    compute_raw_signal(
        out_lma,
        &log_momentum,
        &r_squared,
        params.momentum_weight,
        &mut raw_signal,
    );

    match params.ma_type.as_str() {
        "ema" => smooth_ema_like(
            &raw_signal,
            params.smooth,
            2.0 / ((params.smooth as f64) + 1.0),
            out_signal,
        ),
        "sma" => smooth_sma(&raw_signal, params.smooth, out_signal),
        "wma" => smooth_wma(&raw_signal, params.smooth, out_signal),
        "rma" => smooth_ema_like(
            &raw_signal,
            params.smooth,
            1.0 / (params.smooth as f64),
            out_signal,
        ),
        "vwma" => smooth_vwma(
            &raw_signal,
            volume.ok_or(LogarithmicMovingAverageError::MissingVolumeForVwma)?,
            params.smooth,
            out_signal,
        )?,
        other => {
            return Err(LogarithmicMovingAverageError::InvalidMaType {
                ma_type: other.to_string(),
            });
        }
    }

    finalize_outputs(
        out_signal,
        &log_momentum,
        params.long_threshold,
        params.short_threshold,
        out_position,
        out_momentum_confirmed,
    );
    Ok(())
}

pub fn logarithmic_moving_average_into_slice(
    out_lma: &mut [f64],
    out_signal: &mut [f64],
    out_position: &mut [f64],
    out_momentum_confirmed: &mut [f64],
    input: &LogarithmicMovingAverageInput,
    _kernel: Kernel,
) -> Result<(), LogarithmicMovingAverageError> {
    let prices = input.prices();
    let volume = input.volumes();
    let params = prepare_input(input)?;

    let expected = prices.len();
    if out_lma.len() != expected
        || out_signal.len() != expected
        || out_position.len() != expected
        || out_momentum_confirmed.len() != expected
    {
        return Err(LogarithmicMovingAverageError::OutputLengthMismatch {
            expected,
            got: out_lma
                .len()
                .max(out_signal.len())
                .max(out_position.len())
                .max(out_momentum_confirmed.len()),
        });
    }

    logarithmic_moving_average_compute_into(
        prices,
        volume,
        &params,
        out_lma,
        out_signal,
        out_position,
        out_momentum_confirmed,
    )
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn logarithmic_moving_average_into(
    input: &LogarithmicMovingAverageInput,
    out_lma: &mut [f64],
    out_signal: &mut [f64],
    out_position: &mut [f64],
    out_momentum_confirmed: &mut [f64],
) -> Result<(), LogarithmicMovingAverageError> {
    logarithmic_moving_average_into_slice(
        out_lma,
        out_signal,
        out_position,
        out_momentum_confirmed,
        input,
        Kernel::Auto,
    )
}

pub fn logarithmic_moving_average_with_kernel(
    input: &LogarithmicMovingAverageInput,
    kernel: Kernel,
) -> Result<LogarithmicMovingAverageOutput, LogarithmicMovingAverageError> {
    let len = input.prices().len();
    let mut lma = alloc_with_nan_prefix(len, 0);
    let mut signal = alloc_with_nan_prefix(len, 0);
    let mut position = alloc_with_nan_prefix(len, 0);
    let mut momentum_confirmed = alloc_with_nan_prefix(len, 0);
    logarithmic_moving_average_into_slice(
        &mut lma,
        &mut signal,
        &mut position,
        &mut momentum_confirmed,
        input,
        kernel,
    )?;
    Ok(LogarithmicMovingAverageOutput {
        lma,
        signal,
        position,
        momentum_confirmed,
    })
}

fn expand_axis_usize(
    range: (usize, usize, usize),
) -> Result<Vec<usize>, LogarithmicMovingAverageError> {
    let (start, end, step) = range;
    if start > end {
        return Err(LogarithmicMovingAverageError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if start == end {
        return Ok(vec![start]);
    }
    if step == 0 {
        return Err(LogarithmicMovingAverageError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    let mut out = Vec::new();
    let mut value = start;
    while value <= end {
        out.push(value);
        match value.checked_add(step) {
            Some(next) if next > value => value = next,
            _ => break,
        }
    }
    Ok(out)
}

fn expand_axis_f64(range: (f64, f64, f64)) -> Result<Vec<f64>, LogarithmicMovingAverageError> {
    let (start, end, step) = range;
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(LogarithmicMovingAverageError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if start > end {
        return Err(LogarithmicMovingAverageError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if (start - end).abs() <= f64::EPSILON {
        return Ok(vec![start]);
    }
    if step <= 0.0 {
        return Err(LogarithmicMovingAverageError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    let mut out = Vec::new();
    let mut value = start;
    let limit = end + step * 1e-9;
    while value <= limit {
        out.push(value.min(end));
        value += step;
    }
    Ok(out)
}

pub fn expand_grid_logarithmic_moving_average(
    range: &LogarithmicMovingAverageBatchRange,
) -> Result<Vec<LogarithmicMovingAverageParams>, LogarithmicMovingAverageError> {
    let periods = expand_axis_usize(range.period)?;
    let steepnesses = expand_axis_f64(range.steepness)?;
    let smooths = expand_axis_usize(range.smooth)?;
    let momentum_weights = expand_axis_f64(range.momentum_weight)?;
    let long_thresholds = expand_axis_f64(range.long_threshold)?;
    let short_thresholds = expand_axis_f64(range.short_threshold)?;
    let ma_type = normalize_ma_type(&range.ma_type)?;

    let mut out = Vec::new();
    for &period in &periods {
        for &steepness in &steepnesses {
            for &smooth in &smooths {
                for &momentum_weight in &momentum_weights {
                    for &long_threshold in &long_thresholds {
                        for &short_threshold in &short_thresholds {
                            out.push(LogarithmicMovingAverageParams {
                                period: Some(period),
                                steepness: Some(steepness),
                                ma_type: Some(ma_type.clone()),
                                smooth: Some(smooth),
                                momentum_weight: Some(momentum_weight),
                                long_threshold: Some(long_threshold),
                                short_threshold: Some(short_threshold),
                            });
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

fn logarithmic_moving_average_batch_inner_into(
    prices: &[f64],
    volume: Option<&[f64]>,
    sweep: &LogarithmicMovingAverageBatchRange,
    kernel: Kernel,
    parallel: bool,
    out_lma: &mut [f64],
    out_signal: &mut [f64],
    out_position: &mut [f64],
    out_momentum_confirmed: &mut [f64],
) -> Result<Vec<LogarithmicMovingAverageParams>, LogarithmicMovingAverageError> {
    if prices.is_empty() {
        return Err(LogarithmicMovingAverageError::EmptyInputData);
    }
    let combos = expand_grid_logarithmic_moving_average(sweep)?;
    let cols = prices.len();
    let rows = combos.len();
    let total =
        rows.checked_mul(cols)
            .ok_or(LogarithmicMovingAverageError::OutputLengthMismatch {
                expected: usize::MAX,
                got: 0,
            })?;
    if out_lma.len() != total
        || out_signal.len() != total
        || out_position.len() != total
        || out_momentum_confirmed.len() != total
    {
        return Err(LogarithmicMovingAverageError::OutputLengthMismatch {
            expected: total,
            got: out_lma
                .len()
                .max(out_signal.len())
                .max(out_position.len())
                .max(out_momentum_confirmed.len()),
        });
    }
    let _kernel = kernel;

    for params in &combos {
        let probe = LogarithmicMovingAverageInput {
            data: LogarithmicMovingAverageData::Slice {
                data: prices,
                volume,
            },
            params: params.clone(),
        };
        let _ = prepare_input(&probe)?;
    }

    let do_row = |row: usize,
                  lma_row: &mut [f64],
                  signal_row: &mut [f64],
                  position_row: &mut [f64],
                  momentum_row: &mut [f64]| {
        let row_input = LogarithmicMovingAverageInput {
            data: LogarithmicMovingAverageData::Slice {
                data: prices,
                volume,
            },
            params: combos[row].clone(),
        };
        let prepared = prepare_input(&row_input).unwrap();
        logarithmic_moving_average_compute_into(
            prices,
            volume,
            &prepared,
            lma_row,
            signal_row,
            position_row,
            momentum_row,
        )
        .unwrap();
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out_lma
            .par_chunks_mut(cols)
            .zip(out_signal.par_chunks_mut(cols))
            .zip(out_position.par_chunks_mut(cols))
            .zip(out_momentum_confirmed.par_chunks_mut(cols))
            .enumerate()
            .for_each(
                |(row, (((lma_row, signal_row), position_row), momentum_row))| {
                    do_row(row, lma_row, signal_row, position_row, momentum_row)
                },
            );

        #[cfg(target_arch = "wasm32")]
        for (row, (((lma_row, signal_row), position_row), momentum_row)) in out_lma
            .chunks_mut(cols)
            .zip(out_signal.chunks_mut(cols))
            .zip(out_position.chunks_mut(cols))
            .zip(out_momentum_confirmed.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, lma_row, signal_row, position_row, momentum_row);
        }
    } else {
        for (row, (((lma_row, signal_row), position_row), momentum_row)) in out_lma
            .chunks_mut(cols)
            .zip(out_signal.chunks_mut(cols))
            .zip(out_position.chunks_mut(cols))
            .zip(out_momentum_confirmed.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, lma_row, signal_row, position_row, momentum_row);
        }
    }

    Ok(combos)
}

pub fn logarithmic_moving_average_batch_slice(
    prices: &[f64],
    volume: Option<&[f64]>,
    sweep: &LogarithmicMovingAverageBatchRange,
    kernel: Kernel,
) -> Result<LogarithmicMovingAverageBatchOutput, LogarithmicMovingAverageError> {
    let combos = expand_grid_logarithmic_moving_average(sweep)?;
    let rows = combos.len();
    let cols = prices.len();
    let total =
        rows.checked_mul(cols)
            .ok_or(LogarithmicMovingAverageError::OutputLengthMismatch {
                expected: usize::MAX,
                got: 0,
            })?;
    let mut lma = alloc_with_nan_prefix(total, total);
    let mut signal = alloc_with_nan_prefix(total, total);
    let mut position = alloc_with_nan_prefix(total, total);
    let mut momentum_confirmed = alloc_with_nan_prefix(total, total);
    let combos = logarithmic_moving_average_batch_inner_into(
        prices,
        volume,
        sweep,
        kernel,
        false,
        &mut lma,
        &mut signal,
        &mut position,
        &mut momentum_confirmed,
    )?;
    Ok(LogarithmicMovingAverageBatchOutput {
        lma,
        signal,
        position,
        momentum_confirmed,
        rows,
        cols,
        combos,
    })
}

pub fn logarithmic_moving_average_batch_par_slice(
    prices: &[f64],
    volume: Option<&[f64]>,
    sweep: &LogarithmicMovingAverageBatchRange,
    kernel: Kernel,
) -> Result<LogarithmicMovingAverageBatchOutput, LogarithmicMovingAverageError> {
    let combos = expand_grid_logarithmic_moving_average(sweep)?;
    let rows = combos.len();
    let cols = prices.len();
    let total =
        rows.checked_mul(cols)
            .ok_or(LogarithmicMovingAverageError::OutputLengthMismatch {
                expected: usize::MAX,
                got: 0,
            })?;
    let mut lma = alloc_with_nan_prefix(total, total);
    let mut signal = alloc_with_nan_prefix(total, total);
    let mut position = alloc_with_nan_prefix(total, total);
    let mut momentum_confirmed = alloc_with_nan_prefix(total, total);
    let combos = logarithmic_moving_average_batch_inner_into(
        prices,
        volume,
        sweep,
        kernel,
        true,
        &mut lma,
        &mut signal,
        &mut position,
        &mut momentum_confirmed,
    )?;
    Ok(LogarithmicMovingAverageBatchOutput {
        lma,
        signal,
        position,
        momentum_confirmed,
        rows,
        cols,
        combos,
    })
}

pub fn logarithmic_moving_average_batch_with_kernel(
    prices: &[f64],
    volume: Option<&[f64]>,
    sweep: &LogarithmicMovingAverageBatchRange,
    kernel: Kernel,
) -> Result<LogarithmicMovingAverageBatchOutput, LogarithmicMovingAverageError> {
    match kernel {
        Kernel::Scalar
        | Kernel::ScalarBatch
        | Kernel::Auto
        | Kernel::Avx2
        | Kernel::Avx512
        | Kernel::Avx2Batch
        | Kernel::Avx512Batch => {}
        other => return Err(LogarithmicMovingAverageError::InvalidKernelForBatch(other)),
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        logarithmic_moving_average_batch_par_slice(prices, volume, sweep, kernel)
    }
    #[cfg(target_arch = "wasm32")]
    {
        logarithmic_moving_average_batch_slice(prices, volume, sweep, kernel)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "logarithmic_moving_average")]
#[pyo3(signature = (data, period=DEFAULT_PERIOD, steepness=DEFAULT_STEEPNESS, ma_type=DEFAULT_MA_TYPE, smooth=DEFAULT_SMOOTH, momentum_weight=DEFAULT_MOMENTUM_WEIGHT, long_threshold=DEFAULT_LONG_THRESHOLD, short_threshold=DEFAULT_SHORT_THRESHOLD, volume=None, kernel=None))]
pub fn logarithmic_moving_average_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period: usize,
    steepness: f64,
    ma_type: &str,
    smooth: usize,
    momentum_weight: f64,
    long_threshold: f64,
    short_threshold: f64,
    volume: Option<PyReadonlyArray1<'py, f64>>,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let data = data.as_slice()?;
    let volume_slice = volume.as_ref().map(|v| v.as_slice()).transpose()?;
    let input = match volume_slice {
        Some(v) => LogarithmicMovingAverageInput::from_slice_with_volume(
            data,
            v,
            LogarithmicMovingAverageParams {
                period: Some(period),
                steepness: Some(steepness),
                ma_type: Some(ma_type.to_string()),
                smooth: Some(smooth),
                momentum_weight: Some(momentum_weight),
                long_threshold: Some(long_threshold),
                short_threshold: Some(short_threshold),
            },
        ),
        None => LogarithmicMovingAverageInput::from_slice(
            data,
            LogarithmicMovingAverageParams {
                period: Some(period),
                steepness: Some(steepness),
                ma_type: Some(ma_type.to_string()),
                smooth: Some(smooth),
                momentum_weight: Some(momentum_weight),
                long_threshold: Some(long_threshold),
                short_threshold: Some(short_threshold),
            },
        ),
    };
    let kernel = validate_kernel(kernel, false)?;
    let out = py
        .allow_threads(|| logarithmic_moving_average_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        out.lma.into_pyarray(py),
        out.signal.into_pyarray(py),
        out.position.into_pyarray(py),
        out.momentum_confirmed.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyclass(name = "LogarithmicMovingAverageStream")]
pub struct LogarithmicMovingAverageStreamPy {
    stream: LogarithmicMovingAverageStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl LogarithmicMovingAverageStreamPy {
    #[new]
    #[pyo3(signature = (period=DEFAULT_PERIOD, steepness=DEFAULT_STEEPNESS, ma_type=DEFAULT_MA_TYPE, smooth=DEFAULT_SMOOTH, momentum_weight=DEFAULT_MOMENTUM_WEIGHT, long_threshold=DEFAULT_LONG_THRESHOLD, short_threshold=DEFAULT_SHORT_THRESHOLD))]
    fn new(
        period: usize,
        steepness: f64,
        ma_type: &str,
        smooth: usize,
        momentum_weight: f64,
        long_threshold: f64,
        short_threshold: f64,
    ) -> PyResult<Self> {
        let stream = LogarithmicMovingAverageStream::try_new(LogarithmicMovingAverageParams {
            period: Some(period),
            steepness: Some(steepness),
            ma_type: Some(ma_type.to_string()),
            smooth: Some(smooth),
            momentum_weight: Some(momentum_weight),
            long_threshold: Some(long_threshold),
            short_threshold: Some(short_threshold),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    #[pyo3(signature = (value, volume=None))]
    fn update(&mut self, value: f64, volume: Option<f64>) -> Option<(f64, f64, f64, f64)> {
        self.stream.update(value, volume)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "logarithmic_moving_average_batch")]
#[pyo3(signature = (data, period_range, steepness_range, smooth_range, momentum_weight_range, long_threshold_range, short_threshold_range, ma_type=DEFAULT_MA_TYPE, volume=None, kernel=None))]
pub fn logarithmic_moving_average_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    steepness_range: (f64, f64, f64),
    smooth_range: (usize, usize, usize),
    momentum_weight_range: (f64, f64, f64),
    long_threshold_range: (f64, f64, f64),
    short_threshold_range: (f64, f64, f64),
    ma_type: &str,
    volume: Option<PyReadonlyArray1<'py, f64>>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let volume = volume.as_ref().map(|v| v.as_slice()).transpose()?;
    let sweep = LogarithmicMovingAverageBatchRange {
        period: period_range,
        steepness: steepness_range,
        smooth: smooth_range,
        momentum_weight: momentum_weight_range,
        long_threshold: long_threshold_range,
        short_threshold: short_threshold_range,
        ma_type: ma_type.to_string(),
    };
    let combos = expand_grid_logarithmic_moving_average(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;
    let lma_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let signal_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let position_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let momentum_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_lma = unsafe { lma_arr.as_slice_mut()? };
    let out_signal = unsafe { signal_arr.as_slice_mut()? };
    let out_position = unsafe { position_arr.as_slice_mut()? };
    let out_momentum = unsafe { momentum_arr.as_slice_mut()? };
    let kernel = validate_kernel(kernel, true)?;

    py.allow_threads(|| {
        let batch_kernel = match kernel {
            Kernel::Auto => detect_best_batch_kernel(),
            other => other,
        };
        logarithmic_moving_average_batch_inner_into(
            data,
            volume,
            &sweep,
            batch_kernel.to_non_batch(),
            true,
            out_lma,
            out_signal,
            out_position,
            out_momentum,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let periods: Vec<u64> = combos
        .iter()
        .map(|params| params.period.unwrap_or(DEFAULT_PERIOD) as u64)
        .collect();
    let steepnesses: Vec<f64> = combos
        .iter()
        .map(|params| params.steepness.unwrap_or(DEFAULT_STEEPNESS))
        .collect();
    let smooths: Vec<u64> = combos
        .iter()
        .map(|params| params.smooth.unwrap_or(DEFAULT_SMOOTH) as u64)
        .collect();
    let momentum_weights: Vec<f64> = combos
        .iter()
        .map(|params| params.momentum_weight.unwrap_or(DEFAULT_MOMENTUM_WEIGHT))
        .collect();
    let long_thresholds: Vec<f64> = combos
        .iter()
        .map(|params| params.long_threshold.unwrap_or(DEFAULT_LONG_THRESHOLD))
        .collect();
    let short_thresholds: Vec<f64> = combos
        .iter()
        .map(|params| params.short_threshold.unwrap_or(DEFAULT_SHORT_THRESHOLD))
        .collect();

    let dict = PyDict::new(py);
    dict.set_item("lma", lma_arr.reshape((rows, cols))?)?;
    dict.set_item("signal", signal_arr.reshape((rows, cols))?)?;
    dict.set_item("position", position_arr.reshape((rows, cols))?)?;
    dict.set_item("momentum_confirmed", momentum_arr.reshape((rows, cols))?)?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    dict.set_item("periods", periods.into_pyarray(py))?;
    dict.set_item("steepnesses", steepnesses.into_pyarray(py))?;
    dict.set_item("smooths", smooths.into_pyarray(py))?;
    dict.set_item("momentum_weights", momentum_weights.into_pyarray(py))?;
    dict.set_item("long_thresholds", long_thresholds.into_pyarray(py))?;
    dict.set_item("short_thresholds", short_thresholds.into_pyarray(py))?;
    dict.set_item("ma_type", ma_type)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_logarithmic_moving_average_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(logarithmic_moving_average_py, m)?)?;
    m.add_function(wrap_pyfunction!(logarithmic_moving_average_batch_py, m)?)?;
    m.add_class::<LogarithmicMovingAverageStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LogarithmicMovingAverageJsOutput {
    lma: Vec<f64>,
    signal: Vec<f64>,
    position: Vec<f64>,
    momentum_confirmed: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LogarithmicMovingAverageBatchConfig {
    period_range: Vec<usize>,
    steepness_range: Vec<f64>,
    smooth_range: Vec<usize>,
    momentum_weight_range: Vec<f64>,
    long_threshold_range: Vec<f64>,
    short_threshold_range: Vec<f64>,
    ma_type: String,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LogarithmicMovingAverageBatchJsOutput {
    lma: Vec<f64>,
    signal: Vec<f64>,
    position: Vec<f64>,
    momentum_confirmed: Vec<f64>,
    rows: usize,
    cols: usize,
    combos: Vec<LogarithmicMovingAverageParams>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "logarithmic_moving_average")]
pub fn logarithmic_moving_average_js(
    data: &[f64],
    volume: &[f64],
    period: usize,
    steepness: f64,
    ma_type: &str,
    smooth: usize,
    momentum_weight: f64,
    long_threshold: f64,
    short_threshold: f64,
) -> Result<JsValue, JsValue> {
    let input = if volume.is_empty() {
        LogarithmicMovingAverageInput::from_slice(
            data,
            LogarithmicMovingAverageParams {
                period: Some(period),
                steepness: Some(steepness),
                ma_type: Some(ma_type.to_string()),
                smooth: Some(smooth),
                momentum_weight: Some(momentum_weight),
                long_threshold: Some(long_threshold),
                short_threshold: Some(short_threshold),
            },
        )
    } else {
        LogarithmicMovingAverageInput::from_slice_with_volume(
            data,
            volume,
            LogarithmicMovingAverageParams {
                period: Some(period),
                steepness: Some(steepness),
                ma_type: Some(ma_type.to_string()),
                smooth: Some(smooth),
                momentum_weight: Some(momentum_weight),
                long_threshold: Some(long_threshold),
                short_threshold: Some(short_threshold),
            },
        )
    };
    let out = logarithmic_moving_average(&input).map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&LogarithmicMovingAverageJsOutput {
        lma: out.lma,
        signal: out.signal,
        position: out.position,
        momentum_confirmed: out.momentum_confirmed,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn logarithmic_moving_average_into(
    data_ptr: *const f64,
    volume_ptr: *const f64,
    volume_len: usize,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
    steepness: f64,
    ma_type: &str,
    smooth: usize,
    momentum_weight: f64,
    long_threshold: f64,
    short_threshold: f64,
) -> Result<(), JsValue> {
    if data_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to logarithmic_moving_average_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let volume = if volume_ptr.is_null() || volume_len == 0 {
            None
        } else {
            Some(std::slice::from_raw_parts(volume_ptr, volume_len))
        };
        let out = std::slice::from_raw_parts_mut(out_ptr, len * 4);
        let (out_lma, rest) = out.split_at_mut(len);
        let (out_signal, rest) = rest.split_at_mut(len);
        let (out_position, out_momentum) = rest.split_at_mut(len);
        let input = match volume {
            Some(v) => LogarithmicMovingAverageInput::from_slice_with_volume(
                data,
                v,
                LogarithmicMovingAverageParams {
                    period: Some(period),
                    steepness: Some(steepness),
                    ma_type: Some(ma_type.to_string()),
                    smooth: Some(smooth),
                    momentum_weight: Some(momentum_weight),
                    long_threshold: Some(long_threshold),
                    short_threshold: Some(short_threshold),
                },
            ),
            None => LogarithmicMovingAverageInput::from_slice(
                data,
                LogarithmicMovingAverageParams {
                    period: Some(period),
                    steepness: Some(steepness),
                    ma_type: Some(ma_type.to_string()),
                    smooth: Some(smooth),
                    momentum_weight: Some(momentum_weight),
                    long_threshold: Some(long_threshold),
                    short_threshold: Some(short_threshold),
                },
            ),
        };
        logarithmic_moving_average_into_slice(
            out_lma,
            out_signal,
            out_position,
            out_momentum,
            &input,
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "logarithmic_moving_average_into_host")]
pub fn logarithmic_moving_average_into_host(
    data: &[f64],
    volume: &[f64],
    out_ptr: *mut f64,
    period: usize,
    steepness: f64,
    ma_type: &str,
    smooth: usize,
    momentum_weight: f64,
    long_threshold: f64,
    short_threshold: f64,
) -> Result<(), JsValue> {
    if out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to logarithmic_moving_average_into_host",
        ));
    }
    unsafe {
        let out = std::slice::from_raw_parts_mut(out_ptr, data.len() * 4);
        let (out_lma, rest) = out.split_at_mut(data.len());
        let (out_signal, rest) = rest.split_at_mut(data.len());
        let (out_position, out_momentum) = rest.split_at_mut(data.len());
        let input = if volume.is_empty() {
            LogarithmicMovingAverageInput::from_slice(
                data,
                LogarithmicMovingAverageParams {
                    period: Some(period),
                    steepness: Some(steepness),
                    ma_type: Some(ma_type.to_string()),
                    smooth: Some(smooth),
                    momentum_weight: Some(momentum_weight),
                    long_threshold: Some(long_threshold),
                    short_threshold: Some(short_threshold),
                },
            )
        } else {
            LogarithmicMovingAverageInput::from_slice_with_volume(
                data,
                volume,
                LogarithmicMovingAverageParams {
                    period: Some(period),
                    steepness: Some(steepness),
                    ma_type: Some(ma_type.to_string()),
                    smooth: Some(smooth),
                    momentum_weight: Some(momentum_weight),
                    long_threshold: Some(long_threshold),
                    short_threshold: Some(short_threshold),
                },
            )
        };
        logarithmic_moving_average_into_slice(
            out_lma,
            out_signal,
            out_position,
            out_momentum,
            &input,
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn logarithmic_moving_average_alloc(len: usize) -> *mut f64 {
    let mut buf = vec![0.0_f64; len * 4];
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn logarithmic_moving_average_free(ptr: *mut f64, len: usize) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let _ = Vec::from_raw_parts(ptr, 0, len * 4);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "logarithmic_moving_average_batch")]
pub fn logarithmic_moving_average_batch_js(
    data: &[f64],
    volume: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: LogarithmicMovingAverageBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.period_range.len() != 3
        || config.steepness_range.len() != 3
        || config.smooth_range.len() != 3
        || config.momentum_weight_range.len() != 3
        || config.long_threshold_range.len() != 3
        || config.short_threshold_range.len() != 3
    {
        return Err(JsValue::from_str(
            "Invalid config: ranges must have exactly 3 elements [start, end, step]",
        ));
    }
    let sweep = LogarithmicMovingAverageBatchRange {
        period: (
            config.period_range[0],
            config.period_range[1],
            config.period_range[2],
        ),
        steepness: (
            config.steepness_range[0],
            config.steepness_range[1],
            config.steepness_range[2],
        ),
        smooth: (
            config.smooth_range[0],
            config.smooth_range[1],
            config.smooth_range[2],
        ),
        momentum_weight: (
            config.momentum_weight_range[0],
            config.momentum_weight_range[1],
            config.momentum_weight_range[2],
        ),
        long_threshold: (
            config.long_threshold_range[0],
            config.long_threshold_range[1],
            config.long_threshold_range[2],
        ),
        short_threshold: (
            config.short_threshold_range[0],
            config.short_threshold_range[1],
            config.short_threshold_range[2],
        ),
        ma_type: config.ma_type,
    };
    let batch = logarithmic_moving_average_batch_slice(
        data,
        if volume.is_empty() {
            None
        } else {
            Some(volume)
        },
        &sweep,
        Kernel::Scalar,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&LogarithmicMovingAverageBatchJsOutput {
        lma: batch.lma,
        signal: batch.signal,
        position: batch.position,
        momentum_confirmed: batch.momentum_confirmed,
        rows: batch.rows,
        cols: batch.cols,
        combos: batch.combos,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn logarithmic_moving_average_batch_into(
    data_ptr: *const f64,
    volume_ptr: *const f64,
    volume_len: usize,
    lma_ptr: *mut f64,
    signal_ptr: *mut f64,
    position_ptr: *mut f64,
    momentum_confirmed_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
    steepness_start: f64,
    steepness_end: f64,
    steepness_step: f64,
    smooth_start: usize,
    smooth_end: usize,
    smooth_step: usize,
    momentum_weight_start: f64,
    momentum_weight_end: f64,
    momentum_weight_step: f64,
    long_threshold_start: f64,
    long_threshold_end: f64,
    long_threshold_step: f64,
    short_threshold_start: f64,
    short_threshold_end: f64,
    short_threshold_step: f64,
    ma_type: &str,
) -> Result<usize, JsValue> {
    if data_ptr.is_null()
        || lma_ptr.is_null()
        || signal_ptr.is_null()
        || position_ptr.is_null()
        || momentum_confirmed_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to logarithmic_moving_average_batch_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let volume = if volume_ptr.is_null() || volume_len == 0 {
            None
        } else {
            Some(std::slice::from_raw_parts(volume_ptr, volume_len))
        };
        let sweep = LogarithmicMovingAverageBatchRange {
            period: (period_start, period_end, period_step),
            steepness: (steepness_start, steepness_end, steepness_step),
            smooth: (smooth_start, smooth_end, smooth_step),
            momentum_weight: (
                momentum_weight_start,
                momentum_weight_end,
                momentum_weight_step,
            ),
            long_threshold: (
                long_threshold_start,
                long_threshold_end,
                long_threshold_step,
            ),
            short_threshold: (
                short_threshold_start,
                short_threshold_end,
                short_threshold_step,
            ),
            ma_type: ma_type.to_string(),
        };
        let combos = expand_grid_logarithmic_moving_average(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let total = rows
            .checked_mul(len)
            .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
        let out_lma = std::slice::from_raw_parts_mut(lma_ptr, total);
        let out_signal = std::slice::from_raw_parts_mut(signal_ptr, total);
        let out_position = std::slice::from_raw_parts_mut(position_ptr, total);
        let out_momentum = std::slice::from_raw_parts_mut(momentum_confirmed_ptr, total);
        logarithmic_moving_average_batch_inner_into(
            data,
            volume,
            &sweep,
            Kernel::Scalar,
            false,
            out_lma,
            out_signal,
            out_position,
            out_momentum,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn logarithmic_moving_average_output_into_js(
    data: &[f64],
    volume: &[f64],
    period: usize,
    steepness: f64,
    ma_type: &str,
    smooth: usize,
    momentum_weight: f64,
    long_threshold: f64,
    short_threshold: f64,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = logarithmic_moving_average_js(
        data,
        volume,
        period,
        steepness,
        ma_type,
        smooth,
        momentum_weight,
        long_threshold,
        short_threshold,
    )?;
    crate::write_wasm_object_f64_outputs("logarithmic_moving_average_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn logarithmic_moving_average_batch_output_into_js(
    data: &[f64],
    volume: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = logarithmic_moving_average_batch_js(data, volume, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "logarithmic_moving_average_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::dispatch::{
        compute_cpu_batch, IndicatorBatchRequest, IndicatorDataRef, IndicatorParamSet, ParamKV,
        ParamValue,
    };

    fn sample_series(len: usize) -> (Vec<f64>, Vec<f64>) {
        let data: Vec<f64> = (0..len)
            .map(|i| 100.0 + (i as f64) * 0.15 + ((i as f64) * 0.07).sin())
            .collect();
        let volume: Vec<f64> = (0..len).map(|i| 1000.0 + (i % 17) as f64 * 25.0).collect();
        (data, volume)
    }

    fn assert_close(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (idx, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            if x.is_nan() && y.is_nan() {
                continue;
            }
            assert!(
                (x - y).abs() <= tol,
                "mismatch at index {idx}: {x} vs {y} with tol {tol}"
            );
        }
    }

    #[test]
    fn logarithmic_moving_average_output_contract() -> Result<(), Box<dyn std::error::Error>> {
        let (data, _) = sample_series(256);
        let input = LogarithmicMovingAverageInput::from_slice(
            &data,
            LogarithmicMovingAverageParams::default(),
        );
        let out = logarithmic_moving_average(&input)?;
        assert_eq!(out.lma.len(), data.len());
        assert_eq!(out.signal.len(), data.len());
        assert!(out.lma[..99].iter().all(|v| v.is_nan()));
        assert!(out.signal.iter().any(|v| v.is_finite()));
        Ok(())
    }

    #[test]
    fn logarithmic_moving_average_vwma_requires_volume() {
        let (data, _) = sample_series(128);
        let input = LogarithmicMovingAverageInput::from_slice(
            &data,
            LogarithmicMovingAverageParams {
                ma_type: Some("vwma".to_string()),
                ..Default::default()
            },
        );
        let err = logarithmic_moving_average(&input).unwrap_err();
        assert!(matches!(
            err,
            LogarithmicMovingAverageError::MissingVolumeForVwma
        ));
    }

    #[test]
    fn logarithmic_moving_average_stream_matches_batch() -> Result<(), Box<dyn std::error::Error>> {
        let (data, volume) = sample_series(220);
        let params = LogarithmicMovingAverageParams {
            ma_type: Some("vwma".to_string()),
            ..Default::default()
        };
        let input =
            LogarithmicMovingAverageInput::from_slice_with_volume(&data, &volume, params.clone());
        let batch = logarithmic_moving_average(&input)?;
        let mut stream = LogarithmicMovingAverageStream::try_new(params)?;
        let mut streamed_signal = Vec::with_capacity(data.len());
        for i in 0..data.len() {
            let value = stream.update(data[i], Some(volume[i]));
            streamed_signal.push(value.map(|(_, signal, _, _)| signal).unwrap_or(f64::NAN));
        }
        assert_close(&batch.signal, &streamed_signal, 1e-10);
        Ok(())
    }

    #[test]
    fn logarithmic_moving_average_batch_single_param_matches_single(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (data, volume) = sample_series(196);
        let single =
            logarithmic_moving_average(&LogarithmicMovingAverageInput::from_slice_with_volume(
                &data,
                &volume,
                LogarithmicMovingAverageParams {
                    ma_type: Some("vwma".to_string()),
                    ..Default::default()
                },
            ))?;
        let batch = logarithmic_moving_average_batch_with_kernel(
            &data,
            Some(&volume),
            &LogarithmicMovingAverageBatchRange {
                ma_type: "vwma".to_string(),
                ..Default::default()
            },
            Kernel::Auto,
        )?;
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, data.len());
        assert_close(&single.signal, &batch.signal[..data.len()], 1e-10);
        Ok(())
    }

    #[test]
    fn logarithmic_moving_average_dispatch_matches_direct() -> Result<(), Box<dyn std::error::Error>>
    {
        let (data, volume) = sample_series(192);
        let direct =
            logarithmic_moving_average(&LogarithmicMovingAverageInput::from_slice_with_volume(
                &data,
                &volume,
                LogarithmicMovingAverageParams {
                    ma_type: Some("vwma".to_string()),
                    ..Default::default()
                },
            ))?;
        let params = vec![
            ParamKV {
                key: "ma_type",
                value: ParamValue::EnumString("vwma"),
            },
            ParamKV {
                key: "period",
                value: ParamValue::Int(DEFAULT_PERIOD as i64),
            },
            ParamKV {
                key: "steepness",
                value: ParamValue::Float(DEFAULT_STEEPNESS),
            },
            ParamKV {
                key: "smooth",
                value: ParamValue::Int(DEFAULT_SMOOTH as i64),
            },
            ParamKV {
                key: "momentum_weight",
                value: ParamValue::Float(DEFAULT_MOMENTUM_WEIGHT),
            },
            ParamKV {
                key: "long_threshold",
                value: ParamValue::Float(DEFAULT_LONG_THRESHOLD),
            },
            ParamKV {
                key: "short_threshold",
                value: ParamValue::Float(DEFAULT_SHORT_THRESHOLD),
            },
        ];
        let combos = [IndicatorParamSet { params: &params }];
        let out = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "logarithmic_moving_average",
            output_id: Some("signal"),
            data: IndicatorDataRef::CloseVolume {
                close: &data,
                volume: &volume,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        })?;
        let got = out.values_f64.expect("expected f64 output");
        assert_close(&direct.signal, &got, 1e-10);
        Ok(())
    }
}
