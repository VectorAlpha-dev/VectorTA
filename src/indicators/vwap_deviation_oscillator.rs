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

use crate::utilities::data_loader::Candles;
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::detect_best_batch_kernel;
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::collections::{HashMap, VecDeque};
use thiserror::Error;

const DEFAULT_ROLLING_PERIOD: usize = 20;
const DEFAULT_ROLLING_DAYS: usize = 30;
const DEFAULT_Z_WINDOW: usize = 50;
const DEFAULT_PCT_VOL_LOOKBACK: usize = 100;
const DEFAULT_PCT_MIN_SIGMA: f64 = 0.1;
const DEFAULT_ABS_VOL_LOOKBACK: usize = 100;
const DAY_MS: i64 = 86_400_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub enum VwapDeviationSessionMode {
    FourHours,
    Daily,
    Weekly,
    RollingBars,
    RollingDays,
}

impl Default for VwapDeviationSessionMode {
    fn default() -> Self {
        Self::RollingBars
    }
}

impl std::str::FromStr for VwapDeviationSessionMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "4_hours" | "4h" | "4 hours" => Ok(Self::FourHours),
            "daily" | "1d" | "day" => Ok(Self::Daily),
            "weekly" | "1w" | "week" => Ok(Self::Weekly),
            "rolling_bars" | "rolling bars" | "rolling (lookback: bars)" => Ok(Self::RollingBars),
            "rolling_days" | "rolling days" | "rolling (lookback: days)" => Ok(Self::RollingDays),
            _ => Err(format!("Unknown session_mode: {s}")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub enum VwapDeviationMode {
    Percent,
    Absolute,
    ZScore,
}

impl Default for VwapDeviationMode {
    fn default() -> Self {
        Self::Absolute
    }
}

impl std::str::FromStr for VwapDeviationMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "percent" => Ok(Self::Percent),
            "absolute" => Ok(Self::Absolute),
            "zscore" | "z-score" => Ok(Self::ZScore),
            _ => Err(format!("Unknown deviation_mode: {s}")),
        }
    }
}

#[derive(Debug, Clone)]
pub enum VwapDeviationOscillatorData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        timestamps: &'a [i64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct VwapDeviationOscillatorOutput {
    pub osc: Vec<f64>,
    pub std1: Vec<f64>,
    pub std2: Vec<f64>,
    pub std3: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VwapDeviationOscillatorOutputField {
    Osc,
    Std1,
    Std2,
    Std3,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct VwapDeviationOscillatorParams {
    pub session_mode: Option<VwapDeviationSessionMode>,
    pub rolling_period: Option<usize>,
    pub rolling_days: Option<usize>,
    pub use_close: Option<bool>,
    pub deviation_mode: Option<VwapDeviationMode>,
    pub z_window: Option<usize>,
    pub pct_vol_lookback: Option<usize>,
    pub pct_min_sigma: Option<f64>,
    pub abs_vol_lookback: Option<usize>,
}

impl Default for VwapDeviationOscillatorParams {
    fn default() -> Self {
        Self {
            session_mode: Some(VwapDeviationSessionMode::RollingBars),
            rolling_period: Some(DEFAULT_ROLLING_PERIOD),
            rolling_days: Some(DEFAULT_ROLLING_DAYS),
            use_close: Some(false),
            deviation_mode: Some(VwapDeviationMode::Absolute),
            z_window: Some(DEFAULT_Z_WINDOW),
            pct_vol_lookback: Some(DEFAULT_PCT_VOL_LOOKBACK),
            pct_min_sigma: Some(DEFAULT_PCT_MIN_SIGMA),
            abs_vol_lookback: Some(DEFAULT_ABS_VOL_LOOKBACK),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VwapDeviationOscillatorInput<'a> {
    pub data: VwapDeviationOscillatorData<'a>,
    pub params: VwapDeviationOscillatorParams,
}

impl<'a> VwapDeviationOscillatorInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: VwapDeviationOscillatorParams) -> Self {
        Self {
            data: VwapDeviationOscillatorData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        timestamps: &'a [i64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
        params: VwapDeviationOscillatorParams,
    ) -> Self {
        Self {
            data: VwapDeviationOscillatorData::Slices {
                timestamps,
                high,
                low,
                close,
                volume,
            },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, VwapDeviationOscillatorParams::default())
    }

    #[inline]
    pub fn get_session_mode(&self) -> VwapDeviationSessionMode {
        self.params.session_mode.unwrap_or_default()
    }

    #[inline]
    pub fn get_rolling_period(&self) -> usize {
        self.params.rolling_period.unwrap_or(DEFAULT_ROLLING_PERIOD)
    }

    #[inline]
    pub fn get_rolling_days(&self) -> usize {
        self.params.rolling_days.unwrap_or(DEFAULT_ROLLING_DAYS)
    }

    #[inline]
    pub fn get_use_close(&self) -> bool {
        self.params.use_close.unwrap_or(false)
    }

    #[inline]
    pub fn get_deviation_mode(&self) -> VwapDeviationMode {
        self.params.deviation_mode.unwrap_or_default()
    }

    #[inline]
    pub fn get_z_window(&self) -> usize {
        self.params.z_window.unwrap_or(DEFAULT_Z_WINDOW)
    }

    #[inline]
    pub fn get_pct_vol_lookback(&self) -> usize {
        self.params
            .pct_vol_lookback
            .unwrap_or(DEFAULT_PCT_VOL_LOOKBACK)
    }

    #[inline]
    pub fn get_pct_min_sigma(&self) -> f64 {
        self.params.pct_min_sigma.unwrap_or(DEFAULT_PCT_MIN_SIGMA)
    }

    #[inline]
    pub fn get_abs_vol_lookback(&self) -> usize {
        self.params
            .abs_vol_lookback
            .unwrap_or(DEFAULT_ABS_VOL_LOOKBACK)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct VwapDeviationOscillatorBuilder {
    session_mode: Option<VwapDeviationSessionMode>,
    rolling_period: Option<usize>,
    rolling_days: Option<usize>,
    use_close: Option<bool>,
    deviation_mode: Option<VwapDeviationMode>,
    z_window: Option<usize>,
    pct_vol_lookback: Option<usize>,
    pct_min_sigma: Option<f64>,
    abs_vol_lookback: Option<usize>,
    kernel: Kernel,
}

impl Default for VwapDeviationOscillatorBuilder {
    fn default() -> Self {
        Self {
            session_mode: None,
            rolling_period: None,
            rolling_days: None,
            use_close: None,
            deviation_mode: None,
            z_window: None,
            pct_vol_lookback: None,
            pct_min_sigma: None,
            abs_vol_lookback: None,
            kernel: Kernel::Auto,
        }
    }
}

impl VwapDeviationOscillatorBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn session_mode(mut self, value: VwapDeviationSessionMode) -> Self {
        self.session_mode = Some(value);
        self
    }

    #[inline(always)]
    pub fn rolling_period(mut self, value: usize) -> Self {
        self.rolling_period = Some(value);
        self
    }

    #[inline(always)]
    pub fn rolling_days(mut self, value: usize) -> Self {
        self.rolling_days = Some(value);
        self
    }

    #[inline(always)]
    pub fn use_close(mut self, value: bool) -> Self {
        self.use_close = Some(value);
        self
    }

    #[inline(always)]
    pub fn deviation_mode(mut self, value: VwapDeviationMode) -> Self {
        self.deviation_mode = Some(value);
        self
    }

    #[inline(always)]
    pub fn z_window(mut self, value: usize) -> Self {
        self.z_window = Some(value);
        self
    }

    #[inline(always)]
    pub fn pct_vol_lookback(mut self, value: usize) -> Self {
        self.pct_vol_lookback = Some(value);
        self
    }

    #[inline(always)]
    pub fn pct_min_sigma(mut self, value: f64) -> Self {
        self.pct_min_sigma = Some(value);
        self
    }

    #[inline(always)]
    pub fn abs_vol_lookback(mut self, value: usize) -> Self {
        self.abs_vol_lookback = Some(value);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }
}

#[derive(Debug, Error)]
pub enum VwapDeviationOscillatorError {
    #[error("vwap_deviation_oscillator: Input data slice is empty.")]
    EmptyInputData,
    #[error("vwap_deviation_oscillator: All values are NaN.")]
    AllValuesNaN,
    #[error("vwap_deviation_oscillator: Inconsistent slice lengths: timestamps={timestamps_len}, high={high_len}, low={low_len}, close={close_len}, volume={volume_len}")]
    InconsistentSliceLengths {
        timestamps_len: usize,
        high_len: usize,
        low_len: usize,
        close_len: usize,
        volume_len: usize,
    },
    #[error("vwap_deviation_oscillator: Invalid rolling_period: {rolling_period}")]
    InvalidRollingPeriod { rolling_period: usize },
    #[error("vwap_deviation_oscillator: Invalid rolling_days: {rolling_days}")]
    InvalidRollingDays { rolling_days: usize },
    #[error("vwap_deviation_oscillator: Invalid z_window: {z_window}")]
    InvalidZWindow { z_window: usize },
    #[error("vwap_deviation_oscillator: Invalid pct_vol_lookback: {pct_vol_lookback}")]
    InvalidPctVolLookback { pct_vol_lookback: usize },
    #[error("vwap_deviation_oscillator: Invalid pct_min_sigma: {pct_min_sigma}")]
    InvalidPctMinSigma { pct_min_sigma: f64 },
    #[error("vwap_deviation_oscillator: Invalid abs_vol_lookback: {abs_vol_lookback}")]
    InvalidAbsVolLookback { abs_vol_lookback: usize },
    #[error(
        "vwap_deviation_oscillator: Output length mismatch: expected = {expected}, got = {got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("vwap_deviation_oscillator: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("vwap_deviation_oscillator: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BaseKey {
    session_mode: VwapDeviationSessionMode,
    rolling_period: usize,
    rolling_days: usize,
    use_close: bool,
}

#[derive(Debug, Clone)]
struct NormalizedParams {
    session_mode: VwapDeviationSessionMode,
    rolling_period: usize,
    rolling_days: usize,
    use_close: bool,
    deviation_mode: VwapDeviationMode,
    z_window: usize,
    pct_vol_lookback: usize,
    pct_min_sigma: f64,
    abs_vol_lookback: usize,
}

impl NormalizedParams {
    #[inline(always)]
    fn base_key(&self) -> BaseKey {
        BaseKey {
            session_mode: self.session_mode,
            rolling_period: self.rolling_period,
            rolling_days: self.rolling_days,
            use_close: self.use_close,
        }
    }

    #[inline(always)]
    fn into_params(self) -> VwapDeviationOscillatorParams {
        VwapDeviationOscillatorParams {
            session_mode: Some(self.session_mode),
            rolling_period: Some(self.rolling_period),
            rolling_days: Some(self.rolling_days),
            use_close: Some(self.use_close),
            deviation_mode: Some(self.deviation_mode),
            z_window: Some(self.z_window),
            pct_vol_lookback: Some(self.pct_vol_lookback),
            pct_min_sigma: Some(self.pct_min_sigma),
            abs_vol_lookback: Some(self.abs_vol_lookback),
        }
    }
}

#[derive(Debug, Clone)]
struct BaseSeries {
    resid_abs: Vec<f64>,
    resid_pct: Vec<f64>,
}

#[derive(Debug, Clone)]
struct RollingFiniteWindow {
    period: usize,
    values: Vec<f64>,
    head: usize,
    count: usize,
    sum: f64,
    sumsq: f64,
}

impl RollingFiniteWindow {
    #[inline(always)]
    fn new(period: usize) -> Self {
        Self {
            period,
            values: vec![0.0; period],
            head: 0,
            count: 0,
            sum: 0.0,
            sumsq: 0.0,
        }
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        if value.is_finite() {
            if self.count == self.period {
                let old = self.values[self.head];
                self.sum -= old;
                self.sumsq -= old * old;
            } else {
                self.count += 1;
            }
            self.values[self.head] = value;
            self.sum += value;
            self.sumsq += value * value;
            self.head += 1;
            if self.head == self.period {
                self.head = 0;
            }
        }
        self.stats()
    }

    #[inline(always)]
    fn stats(&self) -> Option<(f64, f64)> {
        if self.count != self.period {
            return None;
        }
        let n = self.period as f64;
        let mean = self.sum / n;
        let var = (self.sumsq / n - mean * mean).max(0.0);
        Some((mean, var.sqrt()))
    }
}

#[derive(Debug, Clone)]
enum StreamVwapState {
    Anchored {
        session_mode: VwapDeviationSessionMode,
        current_period_id: Option<i64>,
        sum_pv: f64,
        sum_vol: f64,
    },
    RollingBars {
        period: usize,
        entries: Vec<(f64, f64)>,
        head: usize,
        count: usize,
        sum_pv: f64,
        sum_vol: f64,
    },
    RollingDays {
        days_ms: i64,
        entries: VecDeque<(i64, f64, f64)>,
        sum_pv: f64,
        sum_vol: f64,
    },
}

#[derive(Debug, Clone)]
pub struct VwapDeviationOscillatorStream {
    params: NormalizedParams,
    vwap_state: StreamVwapState,
    z_stats: RollingFiniteWindow,
    pct_stats: RollingFiniteWindow,
    abs_stats: RollingFiniteWindow,
}

#[inline(always)]
fn validate_rolling_period(rolling_period: usize) -> Result<usize, VwapDeviationOscillatorError> {
    if rolling_period == 0 {
        return Err(VwapDeviationOscillatorError::InvalidRollingPeriod { rolling_period });
    }
    Ok(rolling_period)
}

#[inline(always)]
fn validate_rolling_days(rolling_days: usize) -> Result<usize, VwapDeviationOscillatorError> {
    if rolling_days == 0 {
        return Err(VwapDeviationOscillatorError::InvalidRollingDays { rolling_days });
    }
    Ok(rolling_days)
}

#[inline(always)]
fn validate_z_window(z_window: usize) -> Result<usize, VwapDeviationOscillatorError> {
    if z_window < 5 {
        return Err(VwapDeviationOscillatorError::InvalidZWindow { z_window });
    }
    Ok(z_window)
}

#[inline(always)]
fn validate_pct_vol_lookback(
    pct_vol_lookback: usize,
) -> Result<usize, VwapDeviationOscillatorError> {
    if pct_vol_lookback < 10 {
        return Err(VwapDeviationOscillatorError::InvalidPctVolLookback { pct_vol_lookback });
    }
    Ok(pct_vol_lookback)
}

#[inline(always)]
fn validate_pct_min_sigma(pct_min_sigma: f64) -> Result<f64, VwapDeviationOscillatorError> {
    if !pct_min_sigma.is_finite() || pct_min_sigma < 0.01 {
        return Err(VwapDeviationOscillatorError::InvalidPctMinSigma { pct_min_sigma });
    }
    Ok(pct_min_sigma)
}

#[inline(always)]
fn validate_abs_vol_lookback(
    abs_vol_lookback: usize,
) -> Result<usize, VwapDeviationOscillatorError> {
    if abs_vol_lookback < 10 {
        return Err(VwapDeviationOscillatorError::InvalidAbsVolLookback { abs_vol_lookback });
    }
    Ok(abs_vol_lookback)
}

#[inline(always)]
fn price_ref(high: f64, low: f64, close: f64, use_close: bool) -> f64 {
    if use_close {
        close
    } else if high.is_finite() && low.is_finite() && close.is_finite() {
        (high + low + close) / 3.0
    } else {
        f64::NAN
    }
}

#[inline(always)]
fn period_id(timestamp_ms: i64, session_mode: VwapDeviationSessionMode) -> i64 {
    let sec = timestamp_ms.div_euclid(1000);
    match session_mode {
        VwapDeviationSessionMode::FourHours => sec.div_euclid(3600).div_euclid(4),
        VwapDeviationSessionMode::Daily => sec.div_euclid(86_400),
        VwapDeviationSessionMode::Weekly => (sec.div_euclid(86_400) + 3).div_euclid(7),
        VwapDeviationSessionMode::RollingBars | VwapDeviationSessionMode::RollingDays => 0,
    }
}

#[inline(always)]
fn scale_or_nan(value: f64, factor: f64) -> f64 {
    if value.is_finite() {
        value * factor
    } else {
        f64::NAN
    }
}

#[inline(always)]
fn guarded_sigma(stats: &mut RollingFiniteWindow, value: f64, guard: f64) -> f64 {
    match stats.update(value) {
        Some((_, std)) if std.is_finite() => std.max(guard),
        _ => f64::NAN,
    }
}

#[inline(always)]
fn zscore_value(stats: &mut RollingFiniteWindow, value: f64) -> f64 {
    let Some((mean, std)) = stats.update(value) else {
        return f64::NAN;
    };
    if !value.is_finite() || std == 0.0 || !std.is_finite() {
        return f64::NAN;
    }
    (value - mean) / std
}

impl NormalizedParams {
    #[inline(always)]
    fn from_input(
        input: &VwapDeviationOscillatorInput,
    ) -> Result<Self, VwapDeviationOscillatorError> {
        Ok(Self {
            session_mode: input.get_session_mode(),
            rolling_period: validate_rolling_period(input.get_rolling_period())?,
            rolling_days: validate_rolling_days(input.get_rolling_days())?,
            use_close: input.get_use_close(),
            deviation_mode: input.get_deviation_mode(),
            z_window: validate_z_window(input.get_z_window())?,
            pct_vol_lookback: validate_pct_vol_lookback(input.get_pct_vol_lookback())?,
            pct_min_sigma: validate_pct_min_sigma(input.get_pct_min_sigma())?,
            abs_vol_lookback: validate_abs_vol_lookback(input.get_abs_vol_lookback())?,
        })
    }

    #[inline(always)]
    fn from_params(
        params: &VwapDeviationOscillatorParams,
    ) -> Result<Self, VwapDeviationOscillatorError> {
        Self::from_input(&VwapDeviationOscillatorInput {
            data: VwapDeviationOscillatorData::Slices {
                timestamps: &[],
                high: &[],
                low: &[],
                close: &[],
                volume: &[],
            },
            params: params.clone(),
        })
    }
}

#[inline(always)]
fn extract_input<'a>(
    input: &'a VwapDeviationOscillatorInput<'a>,
) -> Result<(&'a [i64], &'a [f64], &'a [f64], &'a [f64], &'a [f64]), VwapDeviationOscillatorError> {
    let (timestamps, high, low, close, volume) = match &input.data {
        VwapDeviationOscillatorData::Candles { candles } => (
            candles.timestamp.as_slice(),
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
            candles.volume.as_slice(),
        ),
        VwapDeviationOscillatorData::Slices {
            timestamps,
            high,
            low,
            close,
            volume,
        } => (*timestamps, *high, *low, *close, *volume),
    };
    if timestamps.is_empty()
        || high.is_empty()
        || low.is_empty()
        || close.is_empty()
        || volume.is_empty()
    {
        return Err(VwapDeviationOscillatorError::EmptyInputData);
    }
    if timestamps.len() != high.len()
        || timestamps.len() != low.len()
        || timestamps.len() != close.len()
        || timestamps.len() != volume.len()
    {
        return Err(VwapDeviationOscillatorError::InconsistentSliceLengths {
            timestamps_len: timestamps.len(),
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
            volume_len: volume.len(),
        });
    }
    if !high
        .iter()
        .zip(low.iter())
        .zip(close.iter())
        .zip(volume.iter())
        .any(|(((h, l), c), v)| h.is_finite() || l.is_finite() || c.is_finite() || v.is_finite())
    {
        return Err(VwapDeviationOscillatorError::AllValuesNaN);
    }
    Ok((timestamps, high, low, close, volume))
}

#[inline(always)]
fn validate_raw_slices(
    timestamps: &[i64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
) -> Result<(), VwapDeviationOscillatorError> {
    if timestamps.is_empty()
        || high.is_empty()
        || low.is_empty()
        || close.is_empty()
        || volume.is_empty()
    {
        return Err(VwapDeviationOscillatorError::EmptyInputData);
    }
    if timestamps.len() != high.len()
        || timestamps.len() != low.len()
        || timestamps.len() != close.len()
        || timestamps.len() != volume.len()
    {
        return Err(VwapDeviationOscillatorError::InconsistentSliceLengths {
            timestamps_len: timestamps.len(),
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
            volume_len: volume.len(),
        });
    }
    if !high
        .iter()
        .zip(low.iter())
        .zip(close.iter())
        .zip(volume.iter())
        .any(|(((h, l), c), v)| h.is_finite() || l.is_finite() || c.is_finite() || v.is_finite())
    {
        return Err(VwapDeviationOscillatorError::AllValuesNaN);
    }
    Ok(())
}

fn compute_base_series_filtered(
    timestamps: &[i64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    key: BaseKey,
    need_abs: bool,
    need_pct: bool,
) -> BaseSeries {
    let len = close.len();
    let mut resid_abs = if need_abs {
        vec![f64::NAN; len]
    } else {
        Vec::new()
    };
    let mut resid_pct = if need_pct {
        vec![f64::NAN; len]
    } else {
        Vec::new()
    };

    match key.session_mode {
        VwapDeviationSessionMode::RollingBars => {
            let period = key.rolling_period;
            let mut entries = vec![(0.0, 0.0); period];
            let mut head = 0usize;
            let mut count = 0usize;
            let mut sum_pv = 0.0;
            let mut sum_vol = 0.0;
            for i in 0..len {
                let pr = price_ref(high[i], low[i], close[i], key.use_close);
                let contrib = if pr.is_finite() && volume[i].is_finite() {
                    (pr * volume[i], volume[i])
                } else {
                    (0.0, 0.0)
                };
                if count == period {
                    let (old_pv, old_vol) = entries[head];
                    sum_pv -= old_pv;
                    sum_vol -= old_vol;
                } else {
                    count += 1;
                }
                entries[head] = contrib;
                sum_pv += contrib.0;
                sum_vol += contrib.1;
                head += 1;
                if head == period {
                    head = 0;
                }
                let vwap = if sum_vol != 0.0 {
                    sum_pv / sum_vol
                } else {
                    f64::NAN
                };
                if pr.is_finite() && vwap.is_finite() {
                    let residual = pr - vwap;
                    if need_abs {
                        resid_abs[i] = residual;
                    }
                    if need_pct && vwap != 0.0 {
                        resid_pct[i] = 100.0 * (pr / vwap - 1.0);
                    }
                }
            }
        }
        VwapDeviationSessionMode::RollingDays => {
            let mut entries: VecDeque<(i64, f64, f64)> = VecDeque::new();
            let mut sum_pv = 0.0;
            let mut sum_vol = 0.0;
            let days_ms = (key.rolling_days as i64).saturating_mul(DAY_MS);
            for i in 0..len {
                let pr = price_ref(high[i], low[i], close[i], key.use_close);
                let contrib = if pr.is_finite() && volume[i].is_finite() {
                    (pr * volume[i], volume[i])
                } else {
                    (0.0, 0.0)
                };
                entries.push_back((timestamps[i], contrib.0, contrib.1));
                sum_pv += contrib.0;
                sum_vol += contrib.1;
                let cutoff = timestamps[i].saturating_sub(days_ms);
                while entries
                    .front()
                    .map(|(ts, _, _)| *ts < cutoff)
                    .unwrap_or(false)
                {
                    if let Some((_, old_pv, old_vol)) = entries.pop_front() {
                        sum_pv -= old_pv;
                        sum_vol -= old_vol;
                    }
                }
                let vwap = if sum_vol != 0.0 {
                    sum_pv / sum_vol
                } else {
                    f64::NAN
                };
                if pr.is_finite() && vwap.is_finite() {
                    let residual = pr - vwap;
                    if need_abs {
                        resid_abs[i] = residual;
                    }
                    if need_pct && vwap != 0.0 {
                        resid_pct[i] = 100.0 * (pr / vwap - 1.0);
                    }
                }
            }
        }
        VwapDeviationSessionMode::FourHours
        | VwapDeviationSessionMode::Daily
        | VwapDeviationSessionMode::Weekly => {
            let mut last_id: Option<i64> = None;
            let mut sum_pv = 0.0;
            let mut sum_vol = 0.0;
            for i in 0..len {
                let id = period_id(timestamps[i], key.session_mode);
                if last_id.map(|prev| prev != id).unwrap_or(true) {
                    last_id = Some(id);
                    sum_pv = 0.0;
                    sum_vol = 0.0;
                }
                let pr = price_ref(high[i], low[i], close[i], key.use_close);
                if pr.is_finite() && volume[i].is_finite() {
                    sum_pv += pr * volume[i];
                    sum_vol += volume[i];
                }
                let vwap = if sum_vol != 0.0 {
                    sum_pv / sum_vol
                } else {
                    f64::NAN
                };
                if pr.is_finite() && vwap.is_finite() {
                    let residual = pr - vwap;
                    if need_abs {
                        resid_abs[i] = residual;
                    }
                    if need_pct && vwap != 0.0 {
                        resid_pct[i] = 100.0 * (pr / vwap - 1.0);
                    }
                }
            }
        }
    }

    BaseSeries {
        resid_abs,
        resid_pct,
    }
}

fn compute_base_series(
    timestamps: &[i64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    key: BaseKey,
) -> BaseSeries {
    compute_base_series_filtered(timestamps, high, low, close, volume, key, true, true)
}

#[inline(always)]
fn base_residual_needs(deviation_mode: VwapDeviationMode) -> (bool, bool) {
    match deviation_mode {
        VwapDeviationMode::Percent => (false, true),
        VwapDeviationMode::Absolute | VwapDeviationMode::ZScore => (true, false),
    }
}

fn compute_outputs_from_base(
    base: &BaseSeries,
    params: &NormalizedParams,
    out_osc: &mut [f64],
    out_std1: &mut [f64],
    out_std2: &mut [f64],
    out_std3: &mut [f64],
) {
    match params.deviation_mode {
        VwapDeviationMode::Absolute => {
            let mut stats = RollingFiniteWindow::new(params.abs_vol_lookback);
            for i in 0..base.resid_abs.len() {
                let osc = base.resid_abs[i];
                let std1 = guarded_sigma(&mut stats, osc, 1.0);
                out_osc[i] = osc;
                out_std1[i] = std1;
                out_std2[i] = scale_or_nan(std1, 2.0);
                out_std3[i] = scale_or_nan(std1, 3.0);
            }
        }
        VwapDeviationMode::Percent => {
            let mut stats = RollingFiniteWindow::new(params.pct_vol_lookback);
            for i in 0..base.resid_pct.len() {
                let osc = base.resid_pct[i];
                let std1 = guarded_sigma(&mut stats, osc, params.pct_min_sigma);
                out_osc[i] = osc;
                out_std1[i] = std1;
                out_std2[i] = scale_or_nan(std1, 2.0);
                out_std3[i] = scale_or_nan(std1, 3.0);
            }
        }
        VwapDeviationMode::ZScore => {
            let mut stats = RollingFiniteWindow::new(params.z_window);
            for i in 0..base.resid_abs.len() {
                out_osc[i] = zscore_value(&mut stats, base.resid_abs[i]);
                out_std1[i] = 1.0;
                out_std2[i] = 2.0;
                out_std3[i] = 3.0;
            }
        }
    }
}

fn compute_output_field_from_base(
    base: &BaseSeries,
    params: &NormalizedParams,
    out: &mut [f64],
    field: VwapDeviationOscillatorOutputField,
) {
    match params.deviation_mode {
        VwapDeviationMode::Absolute => match field {
            VwapDeviationOscillatorOutputField::Osc => out.copy_from_slice(&base.resid_abs),
            VwapDeviationOscillatorOutputField::Std1
            | VwapDeviationOscillatorOutputField::Std2
            | VwapDeviationOscillatorOutputField::Std3 => {
                let factor = match field {
                    VwapDeviationOscillatorOutputField::Std1 => 1.0,
                    VwapDeviationOscillatorOutputField::Std2 => 2.0,
                    VwapDeviationOscillatorOutputField::Std3 => 3.0,
                    VwapDeviationOscillatorOutputField::Osc => unreachable!(),
                };
                let mut stats = RollingFiniteWindow::new(params.abs_vol_lookback);
                for i in 0..base.resid_abs.len() {
                    let std1 = guarded_sigma(&mut stats, base.resid_abs[i], 1.0);
                    out[i] = if factor == 1.0 {
                        std1
                    } else {
                        scale_or_nan(std1, factor)
                    };
                }
            }
        },
        VwapDeviationMode::Percent => match field {
            VwapDeviationOscillatorOutputField::Osc => out.copy_from_slice(&base.resid_pct),
            VwapDeviationOscillatorOutputField::Std1
            | VwapDeviationOscillatorOutputField::Std2
            | VwapDeviationOscillatorOutputField::Std3 => {
                let factor = match field {
                    VwapDeviationOscillatorOutputField::Std1 => 1.0,
                    VwapDeviationOscillatorOutputField::Std2 => 2.0,
                    VwapDeviationOscillatorOutputField::Std3 => 3.0,
                    VwapDeviationOscillatorOutputField::Osc => unreachable!(),
                };
                let mut stats = RollingFiniteWindow::new(params.pct_vol_lookback);
                for i in 0..base.resid_pct.len() {
                    let std1 = guarded_sigma(&mut stats, base.resid_pct[i], params.pct_min_sigma);
                    out[i] = if factor == 1.0 {
                        std1
                    } else {
                        scale_or_nan(std1, factor)
                    };
                }
            }
        },
        VwapDeviationMode::ZScore => match field {
            VwapDeviationOscillatorOutputField::Osc => {
                let mut stats = RollingFiniteWindow::new(params.z_window);
                for i in 0..base.resid_abs.len() {
                    out[i] = zscore_value(&mut stats, base.resid_abs[i]);
                }
            }
            VwapDeviationOscillatorOutputField::Std1 => out.fill(1.0),
            VwapDeviationOscillatorOutputField::Std2 => out.fill(2.0),
            VwapDeviationOscillatorOutputField::Std3 => out.fill(3.0),
        },
    }
}

#[inline]
pub fn vwap_deviation_oscillator(
    input: &VwapDeviationOscillatorInput,
) -> Result<VwapDeviationOscillatorOutput, VwapDeviationOscillatorError> {
    vwap_deviation_oscillator_with_kernel(input, Kernel::Auto)
}

pub fn vwap_deviation_oscillator_with_kernel(
    input: &VwapDeviationOscillatorInput,
    kernel: Kernel,
) -> Result<VwapDeviationOscillatorOutput, VwapDeviationOscillatorError> {
    let (timestamps, high, low, close, volume) = extract_input(input)?;
    let params = NormalizedParams::from_input(input)?;
    let _chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other.to_non_batch(),
    };
    let len = close.len();
    let mut osc = vec![f64::NAN; len];
    let mut std1 = vec![f64::NAN; len];
    let mut std2 = vec![f64::NAN; len];
    let mut std3 = vec![f64::NAN; len];
    let (need_abs, need_pct) = base_residual_needs(params.deviation_mode);
    let base = compute_base_series_filtered(
        timestamps,
        high,
        low,
        close,
        volume,
        params.base_key(),
        need_abs,
        need_pct,
    );
    compute_outputs_from_base(&base, &params, &mut osc, &mut std1, &mut std2, &mut std3);
    Ok(VwapDeviationOscillatorOutput {
        osc,
        std1,
        std2,
        std3,
    })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn vwap_deviation_oscillator_into(
    timestamps: &[i64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    out_osc: &mut [f64],
    out_std1: &mut [f64],
    out_std2: &mut [f64],
    out_std3: &mut [f64],
    params: VwapDeviationOscillatorParams,
    kernel: Kernel,
) -> Result<(), VwapDeviationOscillatorError> {
    let input =
        VwapDeviationOscillatorInput::from_slices(timestamps, high, low, close, volume, params);
    vwap_deviation_oscillator_into_slice(out_osc, out_std1, out_std2, out_std3, &input, kernel)
}

pub fn vwap_deviation_oscillator_into_slice(
    out_osc: &mut [f64],
    out_std1: &mut [f64],
    out_std2: &mut [f64],
    out_std3: &mut [f64],
    input: &VwapDeviationOscillatorInput,
    kernel: Kernel,
) -> Result<(), VwapDeviationOscillatorError> {
    let (timestamps, high, low, close, volume) = extract_input(input)?;
    let params = NormalizedParams::from_input(input)?;
    let _chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other.to_non_batch(),
    };
    let len = close.len();
    if out_osc.len() != len {
        return Err(VwapDeviationOscillatorError::OutputLengthMismatch {
            expected: len,
            got: out_osc.len(),
        });
    }
    if out_std1.len() != len {
        return Err(VwapDeviationOscillatorError::OutputLengthMismatch {
            expected: len,
            got: out_std1.len(),
        });
    }
    if out_std2.len() != len {
        return Err(VwapDeviationOscillatorError::OutputLengthMismatch {
            expected: len,
            got: out_std2.len(),
        });
    }
    if out_std3.len() != len {
        return Err(VwapDeviationOscillatorError::OutputLengthMismatch {
            expected: len,
            got: out_std3.len(),
        });
    }
    let (need_abs, need_pct) = base_residual_needs(params.deviation_mode);
    let base = compute_base_series_filtered(
        timestamps,
        high,
        low,
        close,
        volume,
        params.base_key(),
        need_abs,
        need_pct,
    );
    compute_outputs_from_base(&base, &params, out_osc, out_std1, out_std2, out_std3);
    Ok(())
}

pub fn vwap_deviation_oscillator_output_into_slice(
    out: &mut [f64],
    input: &VwapDeviationOscillatorInput,
    kernel: Kernel,
    field: VwapDeviationOscillatorOutputField,
) -> Result<(), VwapDeviationOscillatorError> {
    let (timestamps, high, low, close, volume) = extract_input(input)?;
    let params = NormalizedParams::from_input(input)?;
    let _chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other.to_non_batch(),
    };
    let len = close.len();
    if out.len() != len {
        return Err(VwapDeviationOscillatorError::OutputLengthMismatch {
            expected: len,
            got: out.len(),
        });
    }
    if params.deviation_mode == VwapDeviationMode::ZScore {
        match field {
            VwapDeviationOscillatorOutputField::Std1 => {
                out.fill(1.0);
                return Ok(());
            }
            VwapDeviationOscillatorOutputField::Std2 => {
                out.fill(2.0);
                return Ok(());
            }
            VwapDeviationOscillatorOutputField::Std3 => {
                out.fill(3.0);
                return Ok(());
            }
            VwapDeviationOscillatorOutputField::Osc => {}
        }
    }
    let (need_abs, need_pct) = base_residual_needs(params.deviation_mode);
    let base = compute_base_series_filtered(
        timestamps,
        high,
        low,
        close,
        volume,
        params.base_key(),
        need_abs,
        need_pct,
    );
    compute_output_field_from_base(&base, &params, out, field);
    Ok(())
}

impl VwapDeviationOscillatorBuilder {
    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<VwapDeviationOscillatorOutput, VwapDeviationOscillatorError> {
        let input = VwapDeviationOscillatorInput::from_candles(
            candles,
            VwapDeviationOscillatorParams {
                session_mode: self.session_mode,
                rolling_period: self.rolling_period,
                rolling_days: self.rolling_days,
                use_close: self.use_close,
                deviation_mode: self.deviation_mode,
                z_window: self.z_window,
                pct_vol_lookback: self.pct_vol_lookback,
                pct_min_sigma: self.pct_min_sigma,
                abs_vol_lookback: self.abs_vol_lookback,
            },
        );
        vwap_deviation_oscillator_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        timestamps: &[i64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Result<VwapDeviationOscillatorOutput, VwapDeviationOscillatorError> {
        let input = VwapDeviationOscillatorInput::from_slices(
            timestamps,
            high,
            low,
            close,
            volume,
            VwapDeviationOscillatorParams {
                session_mode: self.session_mode,
                rolling_period: self.rolling_period,
                rolling_days: self.rolling_days,
                use_close: self.use_close,
                deviation_mode: self.deviation_mode,
                z_window: self.z_window,
                pct_vol_lookback: self.pct_vol_lookback,
                pct_min_sigma: self.pct_min_sigma,
                abs_vol_lookback: self.abs_vol_lookback,
            },
        );
        vwap_deviation_oscillator_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<VwapDeviationOscillatorStream, VwapDeviationOscillatorError> {
        VwapDeviationOscillatorStream::try_new(VwapDeviationOscillatorParams {
            session_mode: self.session_mode,
            rolling_period: self.rolling_period,
            rolling_days: self.rolling_days,
            use_close: self.use_close,
            deviation_mode: self.deviation_mode,
            z_window: self.z_window,
            pct_vol_lookback: self.pct_vol_lookback,
            pct_min_sigma: self.pct_min_sigma,
            abs_vol_lookback: self.abs_vol_lookback,
        })
    }
}

impl VwapDeviationOscillatorStream {
    pub fn try_new(
        params: VwapDeviationOscillatorParams,
    ) -> Result<Self, VwapDeviationOscillatorError> {
        let params = NormalizedParams::from_params(&params)?;
        let vwap_state = match params.session_mode {
            VwapDeviationSessionMode::FourHours
            | VwapDeviationSessionMode::Daily
            | VwapDeviationSessionMode::Weekly => StreamVwapState::Anchored {
                session_mode: params.session_mode,
                current_period_id: None,
                sum_pv: 0.0,
                sum_vol: 0.0,
            },
            VwapDeviationSessionMode::RollingBars => StreamVwapState::RollingBars {
                period: params.rolling_period,
                entries: vec![(0.0, 0.0); params.rolling_period],
                head: 0,
                count: 0,
                sum_pv: 0.0,
                sum_vol: 0.0,
            },
            VwapDeviationSessionMode::RollingDays => StreamVwapState::RollingDays {
                days_ms: (params.rolling_days as i64).saturating_mul(DAY_MS),
                entries: VecDeque::new(),
                sum_pv: 0.0,
                sum_vol: 0.0,
            },
        };
        Ok(Self {
            z_stats: RollingFiniteWindow::new(params.z_window),
            pct_stats: RollingFiniteWindow::new(params.pct_vol_lookback),
            abs_stats: RollingFiniteWindow::new(params.abs_vol_lookback),
            params,
            vwap_state,
        })
    }

    #[inline]
    pub fn update(
        &mut self,
        timestamp: i64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> (f64, f64, f64, f64) {
        let pr = price_ref(high, low, close, self.params.use_close);
        let contrib = if pr.is_finite() && volume.is_finite() {
            (pr * volume, volume)
        } else {
            (0.0, 0.0)
        };
        let vwap = match &mut self.vwap_state {
            StreamVwapState::Anchored {
                session_mode,
                current_period_id,
                sum_pv,
                sum_vol,
            } => {
                let id = period_id(timestamp, *session_mode);
                if current_period_id.map(|prev| prev != id).unwrap_or(true) {
                    *current_period_id = Some(id);
                    *sum_pv = 0.0;
                    *sum_vol = 0.0;
                }
                *sum_pv += contrib.0;
                *sum_vol += contrib.1;
                if *sum_vol != 0.0 {
                    *sum_pv / *sum_vol
                } else {
                    f64::NAN
                }
            }
            StreamVwapState::RollingBars {
                period,
                entries,
                head,
                count,
                sum_pv,
                sum_vol,
            } => {
                if *count == *period {
                    let (old_pv, old_vol) = entries[*head];
                    *sum_pv -= old_pv;
                    *sum_vol -= old_vol;
                } else {
                    *count += 1;
                }
                entries[*head] = contrib;
                *sum_pv += contrib.0;
                *sum_vol += contrib.1;
                *head += 1;
                if *head == *period {
                    *head = 0;
                }
                if *sum_vol != 0.0 {
                    *sum_pv / *sum_vol
                } else {
                    f64::NAN
                }
            }
            StreamVwapState::RollingDays {
                days_ms,
                entries,
                sum_pv,
                sum_vol,
            } => {
                entries.push_back((timestamp, contrib.0, contrib.1));
                *sum_pv += contrib.0;
                *sum_vol += contrib.1;
                let cutoff = timestamp.saturating_sub(*days_ms);
                while entries
                    .front()
                    .map(|(ts, _, _)| *ts < cutoff)
                    .unwrap_or(false)
                {
                    if let Some((_, old_pv, old_vol)) = entries.pop_front() {
                        *sum_pv -= old_pv;
                        *sum_vol -= old_vol;
                    }
                }
                if *sum_vol != 0.0 {
                    *sum_pv / *sum_vol
                } else {
                    f64::NAN
                }
            }
        };
        let resid_abs = if pr.is_finite() && vwap.is_finite() {
            pr - vwap
        } else {
            f64::NAN
        };
        let resid_pct = if pr.is_finite() && vwap.is_finite() && vwap != 0.0 {
            100.0 * (pr / vwap - 1.0)
        } else {
            f64::NAN
        };
        match self.params.deviation_mode {
            VwapDeviationMode::Absolute => {
                let std1 = guarded_sigma(&mut self.abs_stats, resid_abs, 1.0);
                (
                    resid_abs,
                    std1,
                    scale_or_nan(std1, 2.0),
                    scale_or_nan(std1, 3.0),
                )
            }
            VwapDeviationMode::Percent => {
                let std1 = guarded_sigma(&mut self.pct_stats, resid_pct, self.params.pct_min_sigma);
                (
                    resid_pct,
                    std1,
                    scale_or_nan(std1, 2.0),
                    scale_or_nan(std1, 3.0),
                )
            }
            VwapDeviationMode::ZScore => {
                (zscore_value(&mut self.z_stats, resid_abs), 1.0, 2.0, 3.0)
            }
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct VwapDeviationOscillatorBatchRange {
    pub rolling_period: (usize, usize, usize),
    pub rolling_days: (usize, usize, usize),
    pub z_window: (usize, usize, usize),
    pub pct_vol_lookback: (usize, usize, usize),
    pub pct_min_sigma: (f64, f64, f64),
    pub abs_vol_lookback: (usize, usize, usize),
    pub session_mode: VwapDeviationSessionMode,
    pub use_close: bool,
    pub deviation_mode: VwapDeviationMode,
}

impl Default for VwapDeviationOscillatorBatchRange {
    fn default() -> Self {
        Self {
            rolling_period: (DEFAULT_ROLLING_PERIOD, DEFAULT_ROLLING_PERIOD, 0),
            rolling_days: (DEFAULT_ROLLING_DAYS, DEFAULT_ROLLING_DAYS, 0),
            z_window: (DEFAULT_Z_WINDOW, DEFAULT_Z_WINDOW, 0),
            pct_vol_lookback: (DEFAULT_PCT_VOL_LOOKBACK, DEFAULT_PCT_VOL_LOOKBACK, 0),
            pct_min_sigma: (DEFAULT_PCT_MIN_SIGMA, DEFAULT_PCT_MIN_SIGMA, 0.0),
            abs_vol_lookback: (DEFAULT_ABS_VOL_LOOKBACK, DEFAULT_ABS_VOL_LOOKBACK, 0),
            session_mode: VwapDeviationSessionMode::RollingBars,
            use_close: false,
            deviation_mode: VwapDeviationMode::Absolute,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VwapDeviationOscillatorBatchOutput {
    pub osc: Vec<f64>,
    pub std1: Vec<f64>,
    pub std2: Vec<f64>,
    pub std3: Vec<f64>,
    pub combos: Vec<VwapDeviationOscillatorParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Copy, Clone, Debug)]
pub struct VwapDeviationOscillatorBatchBuilder {
    range: VwapDeviationOscillatorBatchRange,
    kernel: Kernel,
}

impl Default for VwapDeviationOscillatorBatchBuilder {
    fn default() -> Self {
        Self {
            range: VwapDeviationOscillatorBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl VwapDeviationOscillatorBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn session_mode(mut self, value: VwapDeviationSessionMode) -> Self {
        self.range.session_mode = value;
        self
    }

    #[inline(always)]
    pub fn use_close(mut self, value: bool) -> Self {
        self.range.use_close = value;
        self
    }

    #[inline(always)]
    pub fn deviation_mode(mut self, value: VwapDeviationMode) -> Self {
        self.range.deviation_mode = value;
        self
    }

    #[inline(always)]
    pub fn rolling_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.rolling_period = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn rolling_days_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.rolling_days = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn z_window_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.z_window = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn pct_vol_lookback_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.pct_vol_lookback = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn pct_min_sigma_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.pct_min_sigma = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn abs_vol_lookback_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.abs_vol_lookback = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<VwapDeviationOscillatorBatchOutput, VwapDeviationOscillatorError> {
        vwap_deviation_oscillator_batch_with_kernel(
            candles.timestamp.as_slice(),
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
            candles.volume.as_slice(),
            &self.range,
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        timestamps: &[i64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Result<VwapDeviationOscillatorBatchOutput, VwapDeviationOscillatorError> {
        vwap_deviation_oscillator_batch_with_kernel(
            timestamps,
            high,
            low,
            close,
            volume,
            &self.range,
            self.kernel,
        )
    }
}

#[inline(always)]
fn axis_usize(
    start: usize,
    end: usize,
    step: usize,
) -> Result<Vec<usize>, VwapDeviationOscillatorError> {
    if start == end {
        return Ok(vec![start]);
    }
    if step == 0 {
        return Err(VwapDeviationOscillatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    let mut out = Vec::new();
    if start < end {
        let mut x = start;
        while x <= end {
            out.push(x);
            match x.checked_add(step) {
                Some(next) => x = next,
                None => break,
            }
        }
    } else {
        let mut x = start;
        while x >= end {
            out.push(x);
            match x.checked_sub(step) {
                Some(next) => x = next,
                None => break,
            }
            if x > start {
                break;
            }
        }
    }
    if out.is_empty() {
        return Err(VwapDeviationOscillatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

#[inline(always)]
fn axis_f64(start: f64, end: f64, step: f64) -> Result<Vec<f64>, VwapDeviationOscillatorError> {
    if (start - end).abs() <= 1e-12 {
        return Ok(vec![start]);
    }
    if !step.is_finite() || step <= 0.0 {
        return Err(VwapDeviationOscillatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    let mut out = Vec::new();
    let mut x = start;
    if start < end {
        while x <= end + 1e-12 {
            out.push(x);
            x += step;
        }
    } else {
        while x >= end - 1e-12 {
            out.push(x);
            x -= step;
        }
    }
    if out.is_empty() {
        return Err(VwapDeviationOscillatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

pub fn expand_grid(
    range: &VwapDeviationOscillatorBatchRange,
) -> Result<Vec<VwapDeviationOscillatorParams>, VwapDeviationOscillatorError> {
    let rolling_periods = axis_usize(
        range.rolling_period.0,
        range.rolling_period.1,
        range.rolling_period.2,
    )?;
    let rolling_days = axis_usize(
        range.rolling_days.0,
        range.rolling_days.1,
        range.rolling_days.2,
    )?;
    let z_windows = axis_usize(range.z_window.0, range.z_window.1, range.z_window.2)?;
    let pct_vol_lookbacks = axis_usize(
        range.pct_vol_lookback.0,
        range.pct_vol_lookback.1,
        range.pct_vol_lookback.2,
    )?;
    let pct_min_sigmas = axis_f64(
        range.pct_min_sigma.0,
        range.pct_min_sigma.1,
        range.pct_min_sigma.2,
    )?;
    let abs_vol_lookbacks = axis_usize(
        range.abs_vol_lookback.0,
        range.abs_vol_lookback.1,
        range.abs_vol_lookback.2,
    )?;
    let mut out = Vec::with_capacity(
        rolling_periods.len()
            * rolling_days.len()
            * z_windows.len()
            * pct_vol_lookbacks.len()
            * pct_min_sigmas.len()
            * abs_vol_lookbacks.len(),
    );
    for &rolling_period in &rolling_periods {
        for &rolling_days in &rolling_days {
            for &z_window in &z_windows {
                for &pct_vol_lookback in &pct_vol_lookbacks {
                    for &pct_min_sigma in &pct_min_sigmas {
                        for &abs_vol_lookback in &abs_vol_lookbacks {
                            out.push(VwapDeviationOscillatorParams {
                                session_mode: Some(range.session_mode),
                                rolling_period: Some(rolling_period),
                                rolling_days: Some(rolling_days),
                                use_close: Some(range.use_close),
                                deviation_mode: Some(range.deviation_mode),
                                z_window: Some(z_window),
                                pct_vol_lookback: Some(pct_vol_lookback),
                                pct_min_sigma: Some(pct_min_sigma),
                                abs_vol_lookback: Some(abs_vol_lookback),
                            });
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

pub fn vwap_deviation_oscillator_batch_with_kernel(
    timestamps: &[i64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &VwapDeviationOscillatorBatchRange,
    kernel: Kernel,
) -> Result<VwapDeviationOscillatorBatchOutput, VwapDeviationOscillatorError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(VwapDeviationOscillatorError::InvalidKernelForBatch(kernel)),
    };
    vwap_deviation_oscillator_batch_par_slice(
        timestamps,
        high,
        low,
        close,
        volume,
        sweep,
        batch_kernel.to_non_batch(),
    )
}

#[inline(always)]
pub fn vwap_deviation_oscillator_batch_slice(
    timestamps: &[i64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &VwapDeviationOscillatorBatchRange,
    kernel: Kernel,
) -> Result<VwapDeviationOscillatorBatchOutput, VwapDeviationOscillatorError> {
    vwap_deviation_oscillator_batch_inner(
        timestamps, high, low, close, volume, sweep, kernel, false,
    )
}

#[inline(always)]
pub fn vwap_deviation_oscillator_batch_par_slice(
    timestamps: &[i64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &VwapDeviationOscillatorBatchRange,
    kernel: Kernel,
) -> Result<VwapDeviationOscillatorBatchOutput, VwapDeviationOscillatorError> {
    vwap_deviation_oscillator_batch_inner(timestamps, high, low, close, volume, sweep, kernel, true)
}

fn vwap_deviation_oscillator_batch_inner(
    timestamps: &[i64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &VwapDeviationOscillatorBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<VwapDeviationOscillatorBatchOutput, VwapDeviationOscillatorError> {
    validate_raw_slices(timestamps, high, low, close, volume)?;
    let combos = expand_grid(sweep)?;
    let rows = combos.len();
    let cols = close.len();
    let total =
        rows.checked_mul(cols)
            .ok_or_else(|| VwapDeviationOscillatorError::InvalidRange {
                start: rows.to_string(),
                end: cols.to_string(),
                step: "rows*cols".to_string(),
            })?;
    let mut osc = vec![f64::NAN; total];
    let mut std1 = vec![f64::NAN; total];
    let mut std2 = vec![f64::NAN; total];
    let mut std3 = vec![f64::NAN; total];
    vwap_deviation_oscillator_batch_inner_into(
        timestamps, high, low, close, volume, sweep, kernel, parallel, &mut osc, &mut std1,
        &mut std2, &mut std3,
    )?;
    Ok(VwapDeviationOscillatorBatchOutput {
        osc,
        std1,
        std2,
        std3,
        combos,
        rows,
        cols,
    })
}

pub fn vwap_deviation_oscillator_batch_into_slice(
    out_osc: &mut [f64],
    out_std1: &mut [f64],
    out_std2: &mut [f64],
    out_std3: &mut [f64],
    timestamps: &[i64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &VwapDeviationOscillatorBatchRange,
    kernel: Kernel,
) -> Result<(), VwapDeviationOscillatorError> {
    vwap_deviation_oscillator_batch_inner_into(
        timestamps, high, low, close, volume, sweep, kernel, false, out_osc, out_std1, out_std2,
        out_std3,
    )?;
    Ok(())
}

fn vwap_deviation_oscillator_batch_inner_into(
    timestamps: &[i64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &VwapDeviationOscillatorBatchRange,
    kernel: Kernel,
    parallel: bool,
    out_osc: &mut [f64],
    out_std1: &mut [f64],
    out_std2: &mut [f64],
    out_std3: &mut [f64],
) -> Result<Vec<VwapDeviationOscillatorParams>, VwapDeviationOscillatorError> {
    validate_raw_slices(timestamps, high, low, close, volume)?;
    let combos = expand_grid(sweep)?;
    let params = combos
        .iter()
        .map(NormalizedParams::from_params)
        .collect::<Result<Vec<_>, _>>()?;
    let rows = params.len();
    let cols = close.len();
    let expected =
        rows.checked_mul(cols)
            .ok_or_else(|| VwapDeviationOscillatorError::InvalidRange {
                start: rows.to_string(),
                end: cols.to_string(),
                step: "rows*cols".to_string(),
            })?;
    for out in [&*out_osc, &*out_std1, &*out_std2, &*out_std3] {
        if out.len() != expected {
            return Err(VwapDeviationOscillatorError::OutputLengthMismatch {
                expected,
                got: out.len(),
            });
        }
    }
    let _chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other.to_non_batch(),
    };

    let mut base_map: HashMap<BaseKey, BaseSeries> = HashMap::new();
    for p in &params {
        base_map.entry(p.base_key()).or_insert_with(|| {
            compute_base_series(timestamps, high, low, close, volume, p.base_key())
        });
    }
    let do_row = |row: usize,
                  dst_osc: &mut [f64],
                  dst_std1: &mut [f64],
                  dst_std2: &mut [f64],
                  dst_std3: &mut [f64]| {
        let p = &params[row];
        let base = base_map.get(&p.base_key()).unwrap();
        compute_outputs_from_base(base, p, dst_osc, dst_std1, dst_std2, dst_std3);
        Ok::<(), VwapDeviationOscillatorError>(())
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_osc
                .par_chunks_mut(cols)
                .zip(out_std1.par_chunks_mut(cols))
                .zip(out_std2.par_chunks_mut(cols))
                .zip(out_std3.par_chunks_mut(cols))
                .enumerate()
                .try_for_each(|(row, (((dst_osc, dst_std1), dst_std2), dst_std3))| {
                    do_row(row, dst_osc, dst_std1, dst_std2, dst_std3)
                })?;
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, (((dst_osc, dst_std1), dst_std2), dst_std3)) in out_osc
                .chunks_mut(cols)
                .zip(out_std1.chunks_mut(cols))
                .zip(out_std2.chunks_mut(cols))
                .zip(out_std3.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, dst_osc, dst_std1, dst_std2, dst_std3)?;
            }
        }
    } else {
        for (row, (((dst_osc, dst_std1), dst_std2), dst_std3)) in out_osc
            .chunks_mut(cols)
            .zip(out_std1.chunks_mut(cols))
            .zip(out_std2.chunks_mut(cols))
            .zip(out_std3.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, dst_osc, dst_std1, dst_std2, dst_std3)?;
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "vwap_deviation_oscillator")]
#[pyo3(signature = (timestamps, high, low, close, volume, session_mode="rolling_bars", rolling_period=20, rolling_days=30, use_close=false, deviation_mode="absolute", z_window=50, pct_vol_lookback=100, pct_min_sigma=0.1, abs_vol_lookback=100, kernel=None))]
pub fn vwap_deviation_oscillator_py<'py>(
    py: Python<'py>,
    timestamps: PyReadonlyArray1<'py, i64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    session_mode: &str,
    rolling_period: usize,
    rolling_days: usize,
    use_close: bool,
    deviation_mode: &str,
    z_window: usize,
    pct_vol_lookback: usize,
    pct_min_sigma: f64,
    abs_vol_lookback: usize,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let ts = timestamps.as_slice()?;
    let h = high.as_slice()?;
    let l = low.as_slice()?;
    let c = close.as_slice()?;
    let v = volume.as_slice()?;
    if ts.len() != h.len() || ts.len() != l.len() || ts.len() != c.len() || ts.len() != v.len() {
        return Err(PyValueError::new_err(
            "timestamp/high/low/close/volume slice length mismatch",
        ));
    }
    let session_mode = session_mode
        .parse::<VwapDeviationSessionMode>()
        .map_err(PyValueError::new_err)?;
    let deviation_mode = deviation_mode
        .parse::<VwapDeviationMode>()
        .map_err(PyValueError::new_err)?;
    let kern = validate_kernel(kernel, false)?;
    let input = VwapDeviationOscillatorInput::from_slices(
        ts,
        h,
        l,
        c,
        v,
        VwapDeviationOscillatorParams {
            session_mode: Some(session_mode),
            rolling_period: Some(rolling_period),
            rolling_days: Some(rolling_days),
            use_close: Some(use_close),
            deviation_mode: Some(deviation_mode),
            z_window: Some(z_window),
            pct_vol_lookback: Some(pct_vol_lookback),
            pct_min_sigma: Some(pct_min_sigma),
            abs_vol_lookback: Some(abs_vol_lookback),
        },
    );
    let out = py
        .allow_threads(|| vwap_deviation_oscillator_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        out.osc.into_pyarray(py),
        out.std1.into_pyarray(py),
        out.std2.into_pyarray(py),
        out.std3.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyclass(name = "VwapDeviationOscillatorStream")]
pub struct VwapDeviationOscillatorStreamPy {
    stream: VwapDeviationOscillatorStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl VwapDeviationOscillatorStreamPy {
    #[new]
    #[pyo3(signature = (session_mode="rolling_bars", rolling_period=20, rolling_days=30, use_close=false, deviation_mode="absolute", z_window=50, pct_vol_lookback=100, pct_min_sigma=0.1, abs_vol_lookback=100))]
    fn new(
        session_mode: &str,
        rolling_period: usize,
        rolling_days: usize,
        use_close: bool,
        deviation_mode: &str,
        z_window: usize,
        pct_vol_lookback: usize,
        pct_min_sigma: f64,
        abs_vol_lookback: usize,
    ) -> PyResult<Self> {
        let session_mode = session_mode
            .parse::<VwapDeviationSessionMode>()
            .map_err(PyValueError::new_err)?;
        let deviation_mode = deviation_mode
            .parse::<VwapDeviationMode>()
            .map_err(PyValueError::new_err)?;
        let stream = VwapDeviationOscillatorStream::try_new(VwapDeviationOscillatorParams {
            session_mode: Some(session_mode),
            rolling_period: Some(rolling_period),
            rolling_days: Some(rolling_days),
            use_close: Some(use_close),
            deviation_mode: Some(deviation_mode),
            z_window: Some(z_window),
            pct_vol_lookback: Some(pct_vol_lookback),
            pct_min_sigma: Some(pct_min_sigma),
            abs_vol_lookback: Some(abs_vol_lookback),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(
        &mut self,
        timestamp: i64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> (f64, f64, f64, f64) {
        self.stream.update(timestamp, high, low, close, volume)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "vwap_deviation_oscillator_batch")]
#[pyo3(signature = (timestamps, high, low, close, volume, session_mode="rolling_bars", use_close=false, deviation_mode="absolute", rolling_period_range=(20,20,0), rolling_days_range=(30,30,0), z_window_range=(50,50,0), pct_vol_lookback_range=(100,100,0), pct_min_sigma_range=(0.1,0.1,0.0), abs_vol_lookback_range=(100,100,0), kernel=None))]
pub fn vwap_deviation_oscillator_batch_py<'py>(
    py: Python<'py>,
    timestamps: PyReadonlyArray1<'py, i64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    session_mode: &str,
    use_close: bool,
    deviation_mode: &str,
    rolling_period_range: (usize, usize, usize),
    rolling_days_range: (usize, usize, usize),
    z_window_range: (usize, usize, usize),
    pct_vol_lookback_range: (usize, usize, usize),
    pct_min_sigma_range: (f64, f64, f64),
    abs_vol_lookback_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let ts = timestamps.as_slice()?;
    let h = high.as_slice()?;
    let l = low.as_slice()?;
    let c = close.as_slice()?;
    let v = volume.as_slice()?;
    if ts.len() != h.len() || ts.len() != l.len() || ts.len() != c.len() || ts.len() != v.len() {
        return Err(PyValueError::new_err(
            "timestamp/high/low/close/volume slice length mismatch",
        ));
    }
    let session_mode = session_mode
        .parse::<VwapDeviationSessionMode>()
        .map_err(PyValueError::new_err)?;
    let deviation_mode = deviation_mode
        .parse::<VwapDeviationMode>()
        .map_err(PyValueError::new_err)?;
    let sweep = VwapDeviationOscillatorBatchRange {
        rolling_period: rolling_period_range,
        rolling_days: rolling_days_range,
        z_window: z_window_range,
        pct_vol_lookback: pct_vol_lookback_range,
        pct_min_sigma: pct_min_sigma_range,
        abs_vol_lookback: abs_vol_lookback_range,
        session_mode,
        use_close,
        deviation_mode,
    };
    let combos = expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = c.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let osc_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let std1_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let std2_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let std3_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let osc_out = unsafe { osc_arr.as_slice_mut()? };
    let std1_out = unsafe { std1_arr.as_slice_mut()? };
    let std2_out = unsafe { std2_arr.as_slice_mut()? };
    let std3_out = unsafe { std3_arr.as_slice_mut()? };

    let kern = validate_kernel(kernel, true)?;
    py.allow_threads(|| {
        let batch = match kern {
            Kernel::Auto => detect_best_batch_kernel(),
            other => other,
        };
        vwap_deviation_oscillator_batch_inner_into(
            ts,
            h,
            l,
            c,
            v,
            &sweep,
            batch.to_non_batch(),
            true,
            osc_out,
            std1_out,
            std2_out,
            std3_out,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("osc", osc_arr.reshape((rows, cols))?)?;
    dict.set_item("std1", std1_arr.reshape((rows, cols))?)?;
    dict.set_item("std2", std2_arr.reshape((rows, cols))?)?;
    dict.set_item("std3", std3_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "rolling_periods",
        combos
            .iter()
            .map(|p| p.rolling_period.unwrap_or(DEFAULT_ROLLING_PERIOD) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "rolling_days",
        combos
            .iter()
            .map(|p| p.rolling_days.unwrap_or(DEFAULT_ROLLING_DAYS) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "z_windows",
        combos
            .iter()
            .map(|p| p.z_window.unwrap_or(DEFAULT_Z_WINDOW) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "pct_vol_lookbacks",
        combos
            .iter()
            .map(|p| p.pct_vol_lookback.unwrap_or(DEFAULT_PCT_VOL_LOOKBACK) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "pct_min_sigmas",
        combos
            .iter()
            .map(|p| p.pct_min_sigma.unwrap_or(DEFAULT_PCT_MIN_SIGMA))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "abs_vol_lookbacks",
        combos
            .iter()
            .map(|p| p.abs_vol_lookback.unwrap_or(DEFAULT_ABS_VOL_LOOKBACK) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_vwap_deviation_oscillator_module(
    m: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(vwap_deviation_oscillator_py, m)?)?;
    m.add_function(wrap_pyfunction!(vwap_deviation_oscillator_batch_py, m)?)?;
    m.add_class::<VwapDeviationOscillatorStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
fn f64_timestamps_to_i64(ts: &[f64]) -> Result<Vec<i64>, JsValue> {
    let mut out = Vec::with_capacity(ts.len());
    for &t in ts {
        if !t.is_finite() {
            return Err(JsValue::from_str("invalid timestamp"));
        }
        out.push(t as i64);
    }
    Ok(out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "vwap_deviation_oscillator_js")]
pub fn vwap_deviation_oscillator_js(
    timestamps: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    session_mode: &str,
    rolling_period: usize,
    rolling_days: usize,
    use_close: bool,
    deviation_mode: &str,
    z_window: usize,
    pct_vol_lookback: usize,
    pct_min_sigma: f64,
    abs_vol_lookback: usize,
) -> Result<JsValue, JsValue> {
    if timestamps.len() != high.len()
        || timestamps.len() != low.len()
        || timestamps.len() != close.len()
        || timestamps.len() != volume.len()
    {
        return Err(JsValue::from_str(
            "timestamp/high/low/close/volume slice length mismatch",
        ));
    }
    let ts = f64_timestamps_to_i64(timestamps)?;
    let session_mode = session_mode
        .parse::<VwapDeviationSessionMode>()
        .map_err(|e| JsValue::from_str(&e))?;
    let deviation_mode = deviation_mode
        .parse::<VwapDeviationMode>()
        .map_err(|e| JsValue::from_str(&e))?;
    let input = VwapDeviationOscillatorInput::from_slices(
        &ts,
        high,
        low,
        close,
        volume,
        VwapDeviationOscillatorParams {
            session_mode: Some(session_mode),
            rolling_period: Some(rolling_period),
            rolling_days: Some(rolling_days),
            use_close: Some(use_close),
            deviation_mode: Some(deviation_mode),
            z_window: Some(z_window),
            pct_vol_lookback: Some(pct_vol_lookback),
            pct_min_sigma: Some(pct_min_sigma),
            abs_vol_lookback: Some(abs_vol_lookback),
        },
    );
    let out = vwap_deviation_oscillator_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("osc"),
        &serde_wasm_bindgen::to_value(&out.osc).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("std1"),
        &serde_wasm_bindgen::to_value(&out.std1).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("std2"),
        &serde_wasm_bindgen::to_value(&out.std2).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("std3"),
        &serde_wasm_bindgen::to_value(&out.std3).unwrap(),
    )?;
    Ok(obj.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct VwapDeviationOscillatorBatchConfig {
    pub session_mode: String,
    pub use_close: bool,
    pub deviation_mode: String,
    pub rolling_period_range: Vec<usize>,
    pub rolling_days_range: Vec<usize>,
    pub z_window_range: Vec<usize>,
    pub pct_vol_lookback_range: Vec<usize>,
    pub pct_min_sigma_range: Vec<f64>,
    pub abs_vol_lookback_range: Vec<usize>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct VwapDeviationOscillatorBatchJsOutput {
    pub osc: Vec<f64>,
    pub std1: Vec<f64>,
    pub std2: Vec<f64>,
    pub std3: Vec<f64>,
    pub combos: Vec<VwapDeviationOscillatorParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "vwap_deviation_oscillator_batch_js")]
pub fn vwap_deviation_oscillator_batch_js(
    timestamps: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    if timestamps.len() != high.len()
        || timestamps.len() != low.len()
        || timestamps.len() != close.len()
        || timestamps.len() != volume.len()
    {
        return Err(JsValue::from_str(
            "timestamp/high/low/close/volume slice length mismatch",
        ));
    }
    let ts = f64_timestamps_to_i64(timestamps)?;
    let config: VwapDeviationOscillatorBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.rolling_period_range.len() != 3
        || config.rolling_days_range.len() != 3
        || config.z_window_range.len() != 3
        || config.pct_vol_lookback_range.len() != 3
        || config.pct_min_sigma_range.len() != 3
        || config.abs_vol_lookback_range.len() != 3
    {
        return Err(JsValue::from_str(
            "Invalid config: each range must have exactly 3 elements [start, end, step]",
        ));
    }
    let session_mode = config
        .session_mode
        .parse::<VwapDeviationSessionMode>()
        .map_err(|e| JsValue::from_str(&e))?;
    let deviation_mode = config
        .deviation_mode
        .parse::<VwapDeviationMode>()
        .map_err(|e| JsValue::from_str(&e))?;
    let out = vwap_deviation_oscillator_batch_with_kernel(
        &ts,
        high,
        low,
        close,
        volume,
        &VwapDeviationOscillatorBatchRange {
            session_mode,
            use_close: config.use_close,
            deviation_mode,
            rolling_period: (
                config.rolling_period_range[0],
                config.rolling_period_range[1],
                config.rolling_period_range[2],
            ),
            rolling_days: (
                config.rolling_days_range[0],
                config.rolling_days_range[1],
                config.rolling_days_range[2],
            ),
            z_window: (
                config.z_window_range[0],
                config.z_window_range[1],
                config.z_window_range[2],
            ),
            pct_vol_lookback: (
                config.pct_vol_lookback_range[0],
                config.pct_vol_lookback_range[1],
                config.pct_vol_lookback_range[2],
            ),
            pct_min_sigma: (
                config.pct_min_sigma_range[0],
                config.pct_min_sigma_range[1],
                config.pct_min_sigma_range[2],
            ),
            abs_vol_lookback: (
                config.abs_vol_lookback_range[0],
                config.abs_vol_lookback_range[1],
                config.abs_vol_lookback_range[2],
            ),
        },
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&VwapDeviationOscillatorBatchJsOutput {
        osc: out.osc,
        std1: out.std1,
        std2: out.std2,
        std3: out.std3,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn vwap_deviation_oscillator_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn vwap_deviation_oscillator_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn vwap_deviation_oscillator_into(
    timestamps_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    volume_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    session_mode: &str,
    rolling_period: usize,
    rolling_days: usize,
    use_close: bool,
    deviation_mode: &str,
    z_window: usize,
    pct_vol_lookback: usize,
    pct_min_sigma: f64,
    abs_vol_lookback: usize,
) -> Result<(), JsValue> {
    if timestamps_ptr.is_null()
        || high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || volume_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to vwap_deviation_oscillator_into",
        ));
    }
    unsafe {
        let ts_f64 = std::slice::from_raw_parts(timestamps_ptr, len);
        let ts = f64_timestamps_to_i64(ts_f64)?;
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, 4 * len);
        let (out_osc, rest) = out.split_at_mut(len);
        let (out_std1, rest) = rest.split_at_mut(len);
        let (out_std2, out_std3) = rest.split_at_mut(len);
        let session_mode = session_mode
            .parse::<VwapDeviationSessionMode>()
            .map_err(|e| JsValue::from_str(&e))?;
        let deviation_mode = deviation_mode
            .parse::<VwapDeviationMode>()
            .map_err(|e| JsValue::from_str(&e))?;
        let input = VwapDeviationOscillatorInput::from_slices(
            &ts,
            high,
            low,
            close,
            volume,
            VwapDeviationOscillatorParams {
                session_mode: Some(session_mode),
                rolling_period: Some(rolling_period),
                rolling_days: Some(rolling_days),
                use_close: Some(use_close),
                deviation_mode: Some(deviation_mode),
                z_window: Some(z_window),
                pct_vol_lookback: Some(pct_vol_lookback),
                pct_min_sigma: Some(pct_min_sigma),
                abs_vol_lookback: Some(abs_vol_lookback),
            },
        );
        vwap_deviation_oscillator_into_slice(
            out_osc,
            out_std1,
            out_std2,
            out_std3,
            &input,
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn vwap_deviation_oscillator_batch_into(
    timestamps_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    volume_ptr: *const f64,
    osc_ptr: *mut f64,
    std1_ptr: *mut f64,
    std2_ptr: *mut f64,
    std3_ptr: *mut f64,
    len: usize,
    session_mode: &str,
    use_close: bool,
    deviation_mode: &str,
    rolling_period_start: usize,
    rolling_period_end: usize,
    rolling_period_step: usize,
    rolling_days_start: usize,
    rolling_days_end: usize,
    rolling_days_step: usize,
    z_window_start: usize,
    z_window_end: usize,
    z_window_step: usize,
    pct_vol_lookback_start: usize,
    pct_vol_lookback_end: usize,
    pct_vol_lookback_step: usize,
    pct_min_sigma_start: f64,
    pct_min_sigma_end: f64,
    pct_min_sigma_step: f64,
    abs_vol_lookback_start: usize,
    abs_vol_lookback_end: usize,
    abs_vol_lookback_step: usize,
) -> Result<usize, JsValue> {
    if timestamps_ptr.is_null()
        || high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || volume_ptr.is_null()
        || osc_ptr.is_null()
        || std1_ptr.is_null()
        || std2_ptr.is_null()
        || std3_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to vwap_deviation_oscillator_batch_into",
        ));
    }
    unsafe {
        let ts_f64 = std::slice::from_raw_parts(timestamps_ptr, len);
        let ts = f64_timestamps_to_i64(ts_f64)?;
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);
        let session_mode = session_mode
            .parse::<VwapDeviationSessionMode>()
            .map_err(|e| JsValue::from_str(&e))?;
        let deviation_mode = deviation_mode
            .parse::<VwapDeviationMode>()
            .map_err(|e| JsValue::from_str(&e))?;
        let sweep = VwapDeviationOscillatorBatchRange {
            session_mode,
            use_close,
            deviation_mode,
            rolling_period: (
                rolling_period_start,
                rolling_period_end,
                rolling_period_step,
            ),
            rolling_days: (rolling_days_start, rolling_days_end, rolling_days_step),
            z_window: (z_window_start, z_window_end, z_window_step),
            pct_vol_lookback: (
                pct_vol_lookback_start,
                pct_vol_lookback_end,
                pct_vol_lookback_step,
            ),
            pct_min_sigma: (pct_min_sigma_start, pct_min_sigma_end, pct_min_sigma_step),
            abs_vol_lookback: (
                abs_vol_lookback_start,
                abs_vol_lookback_end,
                abs_vol_lookback_step,
            ),
        };
        let combos = expand_grid(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let total = rows
            .checked_mul(len)
            .ok_or_else(|| JsValue::from_str("rows*len overflow"))?;
        let out_osc = std::slice::from_raw_parts_mut(osc_ptr, total);
        let out_std1 = std::slice::from_raw_parts_mut(std1_ptr, total);
        let out_std2 = std::slice::from_raw_parts_mut(std2_ptr, total);
        let out_std3 = std::slice::from_raw_parts_mut(std3_ptr, total);
        vwap_deviation_oscillator_batch_inner_into(
            &ts,
            high,
            low,
            close,
            volume,
            &sweep,
            Kernel::Scalar,
            false,
            out_osc,
            out_std1,
            out_std2,
            out_std3,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn vwap_deviation_oscillator_output_into_js(
    timestamps: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    session_mode: &str,
    rolling_period: usize,
    rolling_days: usize,
    use_close: bool,
    deviation_mode: &str,
    z_window: usize,
    pct_vol_lookback: usize,
    pct_min_sigma: f64,
    abs_vol_lookback: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = vwap_deviation_oscillator_js(
        timestamps,
        high,
        low,
        close,
        volume,
        session_mode,
        rolling_period,
        rolling_days,
        use_close,
        deviation_mode,
        z_window,
        pct_vol_lookback,
        pct_min_sigma,
        abs_vol_lookback,
    )?;
    crate::write_wasm_object_f64_outputs("vwap_deviation_oscillator_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn vwap_deviation_oscillator_batch_output_into_js(
    timestamps: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = vwap_deviation_oscillator_batch_js(timestamps, high, low, close, volume, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "vwap_deviation_oscillator_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_ohlcv(n: usize) -> (Vec<i64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut timestamps = Vec::with_capacity(n);
        let mut high = Vec::with_capacity(n);
        let mut low = Vec::with_capacity(n);
        let mut close = Vec::with_capacity(n);
        let mut volume = Vec::with_capacity(n);
        let mut base = 100.0;
        for i in 0..n {
            timestamps.push(1_700_000_000_000i64 + (i as i64) * 14_400_000);
            let c = base + ((i as f64) * 0.07).sin() * 2.0 + (i as f64) * 0.03;
            let h = c + 1.0 + ((i as f64) * 0.03).cos().abs();
            let l = c - 1.1 - ((i as f64) * 0.05).sin().abs();
            let v = 1000.0 + (i as f64) * 3.0 + ((i as f64) * 0.11).cos() * 35.0;
            high.push(h);
            low.push(l);
            close.push(c);
            volume.push(v.max(1.0));
            base = c;
        }
        (timestamps, high, low, close, volume)
    }

    fn manual_std_ignore_nan(values: &[f64], period: usize, guard: f64) -> Vec<f64> {
        let mut out = vec![f64::NAN; values.len()];
        let mut stats = RollingFiniteWindow::new(period);
        for i in 0..values.len() {
            out[i] = guarded_sigma(&mut stats, values[i], guard);
        }
        out
    }

    #[test]
    fn absolute_mode_matches_manual_sigma_series() {
        let (timestamps, high, low, close, volume) = sample_ohlcv(160);
        let input = VwapDeviationOscillatorInput::from_slices(
            &timestamps,
            &high,
            &low,
            &close,
            &volume,
            VwapDeviationOscillatorParams {
                session_mode: Some(VwapDeviationSessionMode::RollingBars),
                rolling_period: Some(20),
                rolling_days: Some(30),
                use_close: Some(false),
                deviation_mode: Some(VwapDeviationMode::Absolute),
                z_window: Some(50),
                pct_vol_lookback: Some(100),
                pct_min_sigma: Some(0.1),
                abs_vol_lookback: Some(20),
            },
        );
        let out = vwap_deviation_oscillator(&input).unwrap();
        let params = NormalizedParams::from_input(&input).unwrap();
        let base =
            compute_base_series(&timestamps, &high, &low, &close, &volume, params.base_key());
        let expected_std1 = manual_std_ignore_nan(&base.resid_abs, 20, 1.0);
        for i in 0..out.osc.len() {
            if out.osc[i].is_finite() && base.resid_abs[i].is_finite() {
                assert!((out.osc[i] - base.resid_abs[i]).abs() <= 1e-12);
            }
            if out.std1[i].is_nan() && expected_std1[i].is_nan() {
                continue;
            }
            assert!((out.std1[i] - expected_std1[i]).abs() <= 1e-12);
        }
    }

    #[test]
    fn stream_matches_batch() {
        let (timestamps, high, low, close, volume) = sample_ohlcv(128);
        let params = VwapDeviationOscillatorParams {
            session_mode: Some(VwapDeviationSessionMode::RollingDays),
            rolling_period: Some(20),
            rolling_days: Some(5),
            use_close: Some(true),
            deviation_mode: Some(VwapDeviationMode::Percent),
            z_window: Some(50),
            pct_vol_lookback: Some(25),
            pct_min_sigma: Some(0.1),
            abs_vol_lookback: Some(100),
        };
        let input = VwapDeviationOscillatorInput::from_slices(
            &timestamps,
            &high,
            &low,
            &close,
            &volume,
            params.clone(),
        );
        let batch = vwap_deviation_oscillator(&input).unwrap();
        let mut stream = VwapDeviationOscillatorStream::try_new(params).unwrap();
        let mut osc = Vec::with_capacity(close.len());
        let mut std1 = Vec::with_capacity(close.len());
        let mut std2 = Vec::with_capacity(close.len());
        let mut std3 = Vec::with_capacity(close.len());
        for i in 0..close.len() {
            let row = stream.update(timestamps[i], high[i], low[i], close[i], volume[i]);
            osc.push(row.0);
            std1.push(row.1);
            std2.push(row.2);
            std3.push(row.3);
        }
        for (lhs, rhs) in osc.iter().zip(batch.osc.iter()) {
            if lhs.is_nan() && rhs.is_nan() {
                continue;
            }
            assert!((lhs - rhs).abs() <= 1e-12);
        }
        for (lhs, rhs) in std1.iter().zip(batch.std1.iter()) {
            if lhs.is_nan() && rhs.is_nan() {
                continue;
            }
            assert!((lhs - rhs).abs() <= 1e-12);
        }
        for (lhs, rhs) in std2.iter().zip(batch.std2.iter()) {
            if lhs.is_nan() && rhs.is_nan() {
                continue;
            }
            assert!((lhs - rhs).abs() <= 1e-12);
        }
        for (lhs, rhs) in std3.iter().zip(batch.std3.iter()) {
            if lhs.is_nan() && rhs.is_nan() {
                continue;
            }
            assert!((lhs - rhs).abs() <= 1e-12);
        }
    }

    #[test]
    fn batch_first_row_matches_single() {
        let (timestamps, high, low, close, volume) = sample_ohlcv(96);
        let batch = vwap_deviation_oscillator_batch_with_kernel(
            &timestamps,
            &high,
            &low,
            &close,
            &volume,
            &VwapDeviationOscillatorBatchRange {
                session_mode: VwapDeviationSessionMode::RollingBars,
                use_close: false,
                deviation_mode: VwapDeviationMode::ZScore,
                rolling_period: (20, 22, 2),
                rolling_days: (30, 30, 0),
                z_window: (25, 25, 0),
                pct_vol_lookback: (100, 100, 0),
                pct_min_sigma: (0.1, 0.1, 0.0),
                abs_vol_lookback: (100, 100, 0),
            },
            Kernel::Auto,
        )
        .unwrap();
        let input = VwapDeviationOscillatorInput::from_slices(
            &timestamps,
            &high,
            &low,
            &close,
            &volume,
            VwapDeviationOscillatorParams {
                session_mode: Some(VwapDeviationSessionMode::RollingBars),
                rolling_period: Some(20),
                rolling_days: Some(30),
                use_close: Some(false),
                deviation_mode: Some(VwapDeviationMode::ZScore),
                z_window: Some(25),
                pct_vol_lookback: Some(100),
                pct_min_sigma: Some(0.1),
                abs_vol_lookback: Some(100),
            },
        );
        let single = vwap_deviation_oscillator(&input).unwrap();
        let cols = close.len();
        assert_eq!(batch.rows, 2);
        for i in 0..cols {
            let lhs = batch.osc[i];
            let rhs = single.osc[i];
            if lhs.is_nan() && rhs.is_nan() {
                continue;
            }
            assert!((lhs - rhs).abs() <= 1e-12);
        }
    }

    #[test]
    fn into_slice_matches_single() {
        let (timestamps, high, low, close, volume) = sample_ohlcv(80);
        let input = VwapDeviationOscillatorInput::from_slices(
            &timestamps,
            &high,
            &low,
            &close,
            &volume,
            VwapDeviationOscillatorParams::default(),
        );
        let single = vwap_deviation_oscillator(&input).unwrap();
        let mut osc = vec![f64::NAN; close.len()];
        let mut std1 = vec![f64::NAN; close.len()];
        let mut std2 = vec![f64::NAN; close.len()];
        let mut std3 = vec![f64::NAN; close.len()];
        vwap_deviation_oscillator_into_slice(
            &mut osc,
            &mut std1,
            &mut std2,
            &mut std3,
            &input,
            Kernel::Auto,
        )
        .unwrap();
        for i in 0..close.len() {
            if osc[i].is_nan() && single.osc[i].is_nan() {
                continue;
            }
            assert!((osc[i] - single.osc[i]).abs() <= 1e-12);
        }
    }

    #[test]
    fn invalid_z_window_is_rejected() {
        let (timestamps, high, low, close, volume) = sample_ohlcv(32);
        let input = VwapDeviationOscillatorInput::from_slices(
            &timestamps,
            &high,
            &low,
            &close,
            &volume,
            VwapDeviationOscillatorParams {
                z_window: Some(4),
                ..VwapDeviationOscillatorParams::default()
            },
        );
        let err = vwap_deviation_oscillator(&input).unwrap_err();
        assert!(matches!(
            err,
            VwapDeviationOscillatorError::InvalidZWindow { z_window: 4 }
        ));
    }
}
