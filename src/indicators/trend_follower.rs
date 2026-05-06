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

use crate::indicators::moving_averages::ema::{ema_with_kernel, EmaInput, EmaParams};
use crate::indicators::moving_averages::linreg::{
    linreg_with_kernel, LinRegInput, LinRegParams, LinRegStream,
};
use crate::indicators::moving_averages::sma::{sma_with_kernel, SmaInput, SmaParams};
use crate::indicators::moving_averages::vwma::{vwma_with_kernel, VwmaInput, VwmaParams};
use crate::indicators::moving_averages::wilders::{
    wilders_with_kernel, WildersInput, WildersParams,
};
use crate::indicators::moving_averages::wma::{wma_with_kernel, WmaInput, WmaParams};
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
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

const CHANNEL_WINDOW: usize = 280;

#[derive(Debug, Clone)]
pub struct TrendFollowerOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct TrendFollowerParams {
    pub matype: Option<String>,
    pub trend_period: Option<usize>,
    pub ma_period: Option<usize>,
    pub channel_rate_percent: Option<f64>,
    pub use_linear_regression: Option<bool>,
    pub linear_regression_period: Option<usize>,
}

impl Default for TrendFollowerParams {
    fn default() -> Self {
        Self {
            matype: Some("ema".to_string()),
            trend_period: Some(20),
            ma_period: Some(20),
            channel_rate_percent: Some(1.0),
            use_linear_regression: Some(true),
            linear_regression_period: Some(5),
        }
    }
}

#[derive(Debug, Clone)]
pub enum TrendFollowerData<'a> {
    Candles(&'a Candles),
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct TrendFollowerInput<'a> {
    pub data: TrendFollowerData<'a>,
    pub params: TrendFollowerParams,
}

impl<'a> TrendFollowerInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: TrendFollowerParams) -> Self {
        Self {
            data: TrendFollowerData::Candles(candles),
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
        params: TrendFollowerParams,
    ) -> Self {
        Self {
            data: TrendFollowerData::Slices {
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
        Self::from_candles(candles, TrendFollowerParams::default())
    }

    #[inline]
    pub fn as_slices(&self) -> (&[f64], &[f64], &[f64], &[f64]) {
        match &self.data {
            TrendFollowerData::Candles(candles) => (
                candles.high.as_slice(),
                candles.low.as_slice(),
                candles.close.as_slice(),
                candles.volume.as_slice(),
            ),
            TrendFollowerData::Slices {
                high,
                low,
                close,
                volume,
            } => (*high, *low, *close, *volume),
        }
    }

    #[inline]
    pub fn get_matype(&self) -> &str {
        self.params.matype.as_deref().unwrap_or("ema")
    }

    #[inline]
    pub fn get_trend_period(&self) -> usize {
        self.params.trend_period.unwrap_or(20)
    }

    #[inline]
    pub fn get_ma_period(&self) -> usize {
        self.params.ma_period.unwrap_or(20)
    }

    #[inline]
    pub fn get_channel_rate_percent(&self) -> f64 {
        self.params.channel_rate_percent.unwrap_or(1.0)
    }

    #[inline]
    pub fn get_use_linear_regression(&self) -> bool {
        self.params.use_linear_regression.unwrap_or(true)
    }

    #[inline]
    pub fn get_linear_regression_period(&self) -> usize {
        self.params.linear_regression_period.unwrap_or(5)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum TrendFollowerMaType {
    Ema,
    Sma,
    Rma,
    Wma,
    Vwma,
}

impl TrendFollowerMaType {
    #[inline]
    fn as_str(self) -> &'static str {
        match self {
            Self::Ema => "ema",
            Self::Sma => "sma",
            Self::Rma => "rma",
            Self::Wma => "wma",
            Self::Vwma => "vwma",
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct TrendFollowerResolvedParams {
    matype: TrendFollowerMaType,
    trend_period: usize,
    ma_period: usize,
    channel_rate_fraction: f64,
    use_linear_regression: bool,
    linear_regression_period: usize,
}

#[derive(Clone, Debug)]
enum TrendFollowerBaseMaStream {
    Ema(EmaState),
    Sma(SmaState),
    Rma(RmaState),
    Wma(WmaState),
    Vwma(VwmaState),
}

impl TrendFollowerBaseMaStream {
    fn new(matype: TrendFollowerMaType, period: usize) -> Self {
        match matype {
            TrendFollowerMaType::Ema => Self::Ema(EmaState::new(period)),
            TrendFollowerMaType::Sma => Self::Sma(SmaState::new(period)),
            TrendFollowerMaType::Rma => Self::Rma(RmaState::new(period)),
            TrendFollowerMaType::Wma => Self::Wma(WmaState::new(period)),
            TrendFollowerMaType::Vwma => Self::Vwma(VwmaState::new(period)),
        }
    }

    fn update(&mut self, value: f64, volume: f64) -> Option<f64> {
        match self {
            Self::Ema(state) => state.update(value),
            Self::Sma(state) => state.update(value),
            Self::Rma(state) => state.update(value),
            Self::Wma(state) => state.update(value),
            Self::Vwma(state) => state.update(value, volume),
        }
    }
}

#[derive(Clone, Debug)]
struct EmaState {
    period: usize,
    alpha: f64,
    beta: f64,
    value: Option<f64>,
    valid_count: usize,
}

impl EmaState {
    fn new(period: usize) -> Self {
        Self {
            period,
            alpha: 2.0 / (period as f64 + 1.0),
            beta: 1.0 - 2.0 / (period as f64 + 1.0),
            value: None,
            valid_count: 0,
        }
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        if !value.is_finite() {
            return None;
        }
        let next = match self.value {
            None => {
                self.valid_count = 1;
                value
            }
            Some(prev) if self.valid_count < self.period => {
                self.valid_count += 1;
                let vc = self.valid_count as f64;
                ((vc - 1.0) * prev + value) / vc
            }
            Some(prev) => self.beta.mul_add(prev, self.alpha * value),
        };
        self.value = Some(next);
        Some(next)
    }
}

#[derive(Clone, Debug)]
struct SmaState {
    period: usize,
    buffer: Vec<f64>,
    head: usize,
    filled: usize,
    sum: f64,
}

impl SmaState {
    fn new(period: usize) -> Self {
        Self {
            period,
            buffer: vec![0.0; period],
            head: 0,
            filled: 0,
            sum: 0.0,
        }
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        if !value.is_finite() {
            return None;
        }
        if self.filled == self.period {
            self.sum -= self.buffer[self.head];
        } else {
            self.filled += 1;
        }
        self.buffer[self.head] = value;
        self.sum += value;
        self.head = (self.head + 1) % self.period;
        if self.filled == self.period {
            Some(self.sum / self.period as f64)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
struct RmaState {
    period: usize,
    buffer: Vec<f64>,
    head: usize,
    filled: usize,
    sum: f64,
    value: Option<f64>,
}

impl RmaState {
    fn new(period: usize) -> Self {
        Self {
            period,
            buffer: vec![0.0; period],
            head: 0,
            filled: 0,
            sum: 0.0,
            value: None,
        }
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        if !value.is_finite() {
            return None;
        }
        if let Some(prev) = self.value {
            let next = prev + (value - prev) / self.period as f64;
            self.value = Some(next);
            return Some(next);
        }
        self.buffer[self.head] = value;
        self.sum += value;
        self.head = (self.head + 1) % self.period;
        self.filled += 1;
        if self.filled == self.period {
            let next = self.sum / self.period as f64;
            self.value = Some(next);
            Some(next)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
struct WmaState {
    period: usize,
    buffer: Vec<f64>,
    head: usize,
    filled: usize,
}

impl WmaState {
    fn new(period: usize) -> Self {
        Self {
            period,
            buffer: vec![0.0; period],
            head: 0,
            filled: 0,
        }
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        if !value.is_finite() {
            return None;
        }
        self.buffer[self.head] = value;
        self.head = (self.head + 1) % self.period;
        if self.filled < self.period {
            self.filled += 1;
        }
        if self.filled < self.period {
            return None;
        }
        let mut acc = 0.0;
        let mut weight_sum = 0.0;
        for i in 0..self.period {
            let idx = (self.head + i) % self.period;
            let weight = (i + 1) as f64;
            acc += self.buffer[idx] * weight;
            weight_sum += weight;
        }
        Some(acc / weight_sum)
    }
}

#[derive(Clone, Debug)]
struct VwmaState {
    period: usize,
    prices: Vec<f64>,
    volumes: Vec<f64>,
    head: usize,
    filled: usize,
    sum_pv: f64,
    sum_v: f64,
}

impl VwmaState {
    fn new(period: usize) -> Self {
        Self {
            period,
            prices: vec![0.0; period],
            volumes: vec![0.0; period],
            head: 0,
            filled: 0,
            sum_pv: 0.0,
            sum_v: 0.0,
        }
    }

    fn update(&mut self, price: f64, volume: f64) -> Option<f64> {
        if !(price.is_finite() && volume.is_finite()) {
            return None;
        }
        if self.filled == self.period {
            self.sum_pv -= self.prices[self.head] * self.volumes[self.head];
            self.sum_v -= self.volumes[self.head];
        } else {
            self.filled += 1;
        }
        self.prices[self.head] = price;
        self.volumes[self.head] = volume;
        self.sum_pv += price * volume;
        self.sum_v += volume;
        self.head = (self.head + 1) % self.period;
        if self.filled == self.period && self.sum_v != 0.0 {
            Some(self.sum_pv / self.sum_v)
        } else {
            None
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct TrendFollowerBuilder {
    trend_period: Option<usize>,
    ma_period: Option<usize>,
    channel_rate_percent: Option<f64>,
    use_linear_regression: Option<bool>,
    linear_regression_period: Option<usize>,
    kernel: Kernel,
}

impl Default for TrendFollowerBuilder {
    fn default() -> Self {
        Self {
            trend_period: None,
            ma_period: None,
            channel_rate_percent: None,
            use_linear_regression: None,
            linear_regression_period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl TrendFollowerBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn trend_period(mut self, value: usize) -> Self {
        self.trend_period = Some(value);
        self
    }

    #[inline]
    pub fn ma_period(mut self, value: usize) -> Self {
        self.ma_period = Some(value);
        self
    }

    #[inline]
    pub fn channel_rate_percent(mut self, value: f64) -> Self {
        self.channel_rate_percent = Some(value);
        self
    }

    #[inline]
    pub fn use_linear_regression(mut self, value: bool) -> Self {
        self.use_linear_regression = Some(value);
        self
    }

    #[inline]
    pub fn linear_regression_period(mut self, value: usize) -> Self {
        self.linear_regression_period = Some(value);
        self
    }

    #[inline]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline]
    fn params(self, matype: &str) -> TrendFollowerParams {
        TrendFollowerParams {
            matype: Some(matype.to_string()),
            trend_period: self.trend_period,
            ma_period: self.ma_period,
            channel_rate_percent: self.channel_rate_percent,
            use_linear_regression: self.use_linear_regression,
            linear_regression_period: self.linear_regression_period,
        }
    }

    #[inline]
    pub fn apply(self, candles: &Candles) -> Result<TrendFollowerOutput, TrendFollowerError> {
        let input = TrendFollowerInput::from_candles(candles, self.params("ema"));
        trend_follower_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn apply_with_matype(
        self,
        candles: &Candles,
        matype: &str,
    ) -> Result<TrendFollowerOutput, TrendFollowerError> {
        let input = TrendFollowerInput::from_candles(candles, self.params(matype));
        trend_follower_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
        matype: &str,
    ) -> Result<TrendFollowerOutput, TrendFollowerError> {
        let input = TrendFollowerInput::from_slices(high, low, close, volume, self.params(matype));
        trend_follower_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn into_stream(self, matype: &str) -> Result<TrendFollowerStream, TrendFollowerError> {
        TrendFollowerStream::try_new(self.params(matype))
    }
}

#[derive(Debug, Error)]
pub enum TrendFollowerError {
    #[error("trend_follower: Empty input data.")]
    EmptyInputData,
    #[error(
        "trend_follower: Data length mismatch: high={high_len}, low={low_len}, close={close_len}, volume={volume_len}"
    )]
    DataLengthMismatch {
        high_len: usize,
        low_len: usize,
        close_len: usize,
        volume_len: usize,
    },
    #[error("trend_follower: All values are invalid.")]
    AllValuesNaN,
    #[error("trend_follower: Invalid MA type: {matype}")]
    InvalidMaType { matype: String },
    #[error("trend_follower: Invalid trend period: {trend_period}")]
    InvalidTrendPeriod { trend_period: usize },
    #[error("trend_follower: Invalid MA period: {ma_period}, data length = {data_len}")]
    InvalidMaPeriod { ma_period: usize, data_len: usize },
    #[error(
        "trend_follower: Invalid linear regression period: {linear_regression_period}, data length = {data_len}"
    )]
    InvalidLinearRegressionPeriod {
        linear_regression_period: usize,
        data_len: usize,
    },
    #[error("trend_follower: Invalid channel rate percent: {channel_rate_percent}")]
    InvalidChannelRatePercent { channel_rate_percent: f64 },
    #[error("trend_follower: Moving average computation failed: {0}")]
    MovingAverageError(String),
    #[error("trend_follower: Linear regression computation failed: {0}")]
    LinearRegressionError(String),
    #[error("trend_follower: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("trend_follower: Invalid integer range: start={start}, end={end}, step={step}")]
    InvalidRangeUsize {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("trend_follower: Invalid float range: start={start}, end={end}, step={step}")]
    InvalidRangeF64 { start: f64, end: f64, step: f64 },
    #[error("trend_follower: Invalid kernel for batch path: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline]
fn parse_matype(matype: &str) -> Result<TrendFollowerMaType, TrendFollowerError> {
    if matype.eq_ignore_ascii_case("ema") {
        return Ok(TrendFollowerMaType::Ema);
    }
    if matype.eq_ignore_ascii_case("sma") {
        return Ok(TrendFollowerMaType::Sma);
    }
    if matype.eq_ignore_ascii_case("rma") {
        return Ok(TrendFollowerMaType::Rma);
    }
    if matype.eq_ignore_ascii_case("wma") {
        return Ok(TrendFollowerMaType::Wma);
    }
    if matype.eq_ignore_ascii_case("vwma") {
        return Ok(TrendFollowerMaType::Vwma);
    }
    Err(TrendFollowerError::InvalidMaType {
        matype: matype.to_string(),
    })
}

#[inline]
fn resolve_params(
    input: &TrendFollowerInput<'_>,
    data_len: usize,
) -> Result<TrendFollowerResolvedParams, TrendFollowerError> {
    let trend_period = input.get_trend_period();
    if trend_period < 1 {
        return Err(TrendFollowerError::InvalidTrendPeriod { trend_period });
    }

    let ma_period = input.get_ma_period();
    if ma_period == 0 || ma_period > data_len {
        return Err(TrendFollowerError::InvalidMaPeriod {
            ma_period,
            data_len,
        });
    }

    let linear_regression_period = input.get_linear_regression_period();
    if input.get_use_linear_regression()
        && (linear_regression_period < 2 || linear_regression_period > data_len)
    {
        return Err(TrendFollowerError::InvalidLinearRegressionPeriod {
            linear_regression_period,
            data_len,
        });
    }

    let channel_rate_percent = input.get_channel_rate_percent();
    if !channel_rate_percent.is_finite() || channel_rate_percent <= 0.0 {
        return Err(TrendFollowerError::InvalidChannelRatePercent {
            channel_rate_percent,
        });
    }

    Ok(TrendFollowerResolvedParams {
        matype: parse_matype(input.get_matype())?,
        trend_period,
        ma_period,
        channel_rate_fraction: channel_rate_percent * 0.01,
        use_linear_regression: input.get_use_linear_regression(),
        linear_regression_period,
    })
}

#[inline]
fn first_valid_bar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    needs_volume: bool,
) -> Option<usize> {
    (0..high.len()).find(|&i| {
        high[i].is_finite()
            && low[i].is_finite()
            && close[i].is_finite()
            && (!needs_volume || volume[i].is_finite())
    })
}

#[inline]
fn data_is_clean(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    first: usize,
    needs_volume: bool,
) -> bool {
    for i in first..high.len() {
        if !(high[i].is_finite() && low[i].is_finite() && close[i].is_finite()) {
            return false;
        }
        if needs_volume && !volume[i].is_finite() {
            return false;
        }
    }
    true
}

#[inline]
fn trend_follower_prepare<'a>(
    input: &'a TrendFollowerInput<'a>,
) -> Result<
    (
        &'a [f64],
        &'a [f64],
        &'a [f64],
        &'a [f64],
        TrendFollowerResolvedParams,
        usize,
    ),
    TrendFollowerError,
> {
    let (high, low, close, volume) = input.as_slices();
    if high.is_empty() {
        return Err(TrendFollowerError::EmptyInputData);
    }
    if high.len() != low.len() || high.len() != close.len() || high.len() != volume.len() {
        return Err(TrendFollowerError::DataLengthMismatch {
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
            volume_len: volume.len(),
        });
    }
    let params = resolve_params(input, high.len())?;
    let first = first_valid_bar(
        high,
        low,
        close,
        volume,
        params.matype == TrendFollowerMaType::Vwma,
    )
    .ok_or(TrendFollowerError::AllValuesNaN)?;
    Ok((high, low, close, volume, params, first))
}

#[inline]
fn compute_ma_series(
    close: &[f64],
    volume: &[f64],
    params: TrendFollowerResolvedParams,
    kernel: Kernel,
) -> Result<Vec<f64>, TrendFollowerError> {
    match params.matype {
        TrendFollowerMaType::Ema => ema_with_kernel(
            &EmaInput::from_slice(
                close,
                EmaParams {
                    period: Some(params.ma_period),
                },
            ),
            kernel,
        )
        .map(|out| out.values)
        .map_err(|e| TrendFollowerError::MovingAverageError(e.to_string())),
        TrendFollowerMaType::Sma => sma_with_kernel(
            &SmaInput::from_slice(
                close,
                SmaParams {
                    period: Some(params.ma_period),
                },
            ),
            kernel,
        )
        .map(|out| out.values)
        .map_err(|e| TrendFollowerError::MovingAverageError(e.to_string())),
        TrendFollowerMaType::Rma => wilders_with_kernel(
            &WildersInput::from_slice(
                close,
                WildersParams {
                    period: Some(params.ma_period),
                },
            ),
            kernel,
        )
        .map(|out| out.values)
        .map_err(|e| TrendFollowerError::MovingAverageError(e.to_string())),
        TrendFollowerMaType::Wma => wma_with_kernel(
            &WmaInput::from_slice(
                close,
                WmaParams {
                    period: Some(params.ma_period),
                },
            ),
            kernel,
        )
        .map(|out| out.values)
        .map_err(|e| TrendFollowerError::MovingAverageError(e.to_string())),
        TrendFollowerMaType::Vwma => vwma_with_kernel(
            &VwmaInput::from_slice(
                close,
                volume,
                VwmaParams {
                    period: Some(params.ma_period),
                },
            ),
            kernel,
        )
        .map(|out| out.values)
        .map_err(|e| TrendFollowerError::MovingAverageError(e.to_string())),
    }
}

#[inline]
fn push_max(queue: &mut VecDeque<(usize, f64)>, idx: usize, value: f64, window: usize) {
    let min_idx = idx.saturating_add(1).saturating_sub(window);
    while let Some(&(old_idx, _)) = queue.front() {
        if old_idx < min_idx {
            queue.pop_front();
        } else {
            break;
        }
    }
    while let Some(&(_, old_value)) = queue.back() {
        if old_value <= value {
            queue.pop_back();
        } else {
            break;
        }
    }
    queue.push_back((idx, value));
}

#[inline]
fn push_min(queue: &mut VecDeque<(usize, f64)>, idx: usize, value: f64, window: usize) {
    let min_idx = idx.saturating_add(1).saturating_sub(window);
    while let Some(&(old_idx, _)) = queue.front() {
        if old_idx < min_idx {
            queue.pop_front();
        } else {
            break;
        }
    }
    while let Some(&(_, old_value)) = queue.back() {
        if old_value >= value {
            queue.pop_back();
        } else {
            break;
        }
    }
    queue.push_back((idx, value));
}

#[inline]
fn evict_front(queue: &mut VecDeque<(usize, f64)>, idx: usize, window: usize) {
    let min_idx = idx.saturating_add(1).saturating_sub(window);
    while let Some(&(old_idx, _)) = queue.front() {
        if old_idx < min_idx {
            queue.pop_front();
        } else {
            break;
        }
    }
}

#[derive(Clone, Debug)]
struct MonoQueue {
    idx: Vec<usize>,
    vals: Vec<f64>,
    head: usize,
    tail: usize,
    count: usize,
}

impl MonoQueue {
    fn new(window: usize) -> Self {
        let cap = window.max(1) + 1;
        Self {
            idx: vec![0; cap],
            vals: vec![0.0; cap],
            head: 0,
            tail: 0,
            count: 0,
        }
    }

    #[inline(always)]
    fn cap(&self) -> usize {
        self.idx.len()
    }

    #[inline(always)]
    fn prev_pos(&self) -> usize {
        if self.tail == 0 {
            self.cap() - 1
        } else {
            self.tail - 1
        }
    }

    #[inline(always)]
    fn evict(&mut self, idx: usize, window: usize) {
        let min_idx = idx.saturating_add(1).saturating_sub(window);
        while self.count > 0 && self.idx[self.head] < min_idx {
            self.head += 1;
            if self.head == self.cap() {
                self.head = 0;
            }
            self.count -= 1;
        }
    }

    #[inline(always)]
    fn push_max(&mut self, idx: usize, value: f64) {
        while self.count > 0 {
            let back = self.prev_pos();
            if self.vals[back] <= value {
                self.tail = back;
                self.count -= 1;
            } else {
                break;
            }
        }
        self.idx[self.tail] = idx;
        self.vals[self.tail] = value;
        self.tail += 1;
        if self.tail == self.cap() {
            self.tail = 0;
        }
        self.count += 1;
    }

    #[inline(always)]
    fn push_min(&mut self, idx: usize, value: f64) {
        while self.count > 0 {
            let back = self.prev_pos();
            if self.vals[back] >= value {
                self.tail = back;
                self.count -= 1;
            } else {
                break;
            }
        }
        self.idx[self.tail] = idx;
        self.vals[self.tail] = value;
        self.tail += 1;
        if self.tail == self.cap() {
            self.tail = 0;
        }
        self.count += 1;
    }

    #[inline(always)]
    fn front_value(&self) -> Option<f64> {
        if self.count == 0 {
            None
        } else {
            Some(self.vals[self.head])
        }
    }
}

fn trend_follower_compute_clean_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    params: TrendFollowerResolvedParams,
    first: usize,
    kernel: Kernel,
    out: &mut [f64],
) -> Result<(), TrendFollowerError> {
    let base_ma = compute_ma_series(close, volume, params, kernel)?;
    let trend_ma = if params.use_linear_regression {
        linreg_with_kernel(
            &LinRegInput::from_slice(
                &base_ma,
                LinRegParams {
                    period: Some(params.linear_regression_period),
                },
            ),
            kernel,
        )
        .map(|series| series.values)
        .map_err(|e| TrendFollowerError::LinearRegressionError(e.to_string()))?
    } else {
        base_ma
    };

    let mut high_max = MonoQueue::new(CHANNEL_WINDOW);
    let mut low_min = MonoQueue::new(CHANNEL_WINDOW);
    let mut ma_max = MonoQueue::new(params.trend_period.max(1));
    let mut ma_min = MonoQueue::new(params.trend_period.max(1));

    for i in first..high.len() {
        high_max.evict(i, CHANNEL_WINDOW);
        low_min.evict(i, CHANNEL_WINDOW);
        ma_max.evict(i, params.trend_period);
        ma_min.evict(i, params.trend_period);

        high_max.push_max(i, high[i]);
        low_min.push_min(i, low[i]);

        let ma = trend_ma[i];
        if ma.is_finite() {
            ma_max.push_max(i, ma);
            ma_min.push_min(i, ma);
        }

        let (hh, ll) = match (ma_max.front_value(), ma_min.front_value()) {
            (Some(hh), Some(ll)) => (hh, ll),
            _ => continue,
        };
        let (channel_high, channel_low) = match (high_max.front_value(), low_min.front_value()) {
            (Some(hi), Some(lo)) => (hi, lo),
            _ => continue,
        };
        let chan = (channel_high - channel_low) * params.channel_rate_fraction;
        if !ma.is_finite() || !chan.is_finite() || chan == 0.0 {
            out[i] = f64::NAN;
            continue;
        }

        let diff = (hh - ll).abs();
        let trend = if diff > chan {
            if ma > ll + chan {
                1.0
            } else if ma < hh - chan {
                -1.0
            } else {
                0.0
            }
        } else {
            0.0
        };
        out[i] = trend * diff / chan;
    }

    Ok(())
}

fn trend_follower_compute_fallback_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    input: &TrendFollowerInput<'_>,
    out: &mut [f64],
) -> Result<(), TrendFollowerError> {
    let mut stream = TrendFollowerStream::try_new(input.params.clone())?;
    for i in 0..high.len() {
        out[i] = stream
            .update_reset_on_nan(high[i], low[i], close[i], volume[i])
            .unwrap_or(f64::NAN);
    }
    Ok(())
}

fn trend_follower_compute_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    input: &TrendFollowerInput<'_>,
    params: TrendFollowerResolvedParams,
    first: usize,
    kernel: Kernel,
    out: &mut [f64],
) -> Result<(), TrendFollowerError> {
    if data_is_clean(
        high,
        low,
        close,
        volume,
        first,
        params.matype == TrendFollowerMaType::Vwma,
    ) {
        trend_follower_compute_clean_into(high, low, close, volume, params, first, kernel, out)
    } else {
        trend_follower_compute_fallback_into(high, low, close, volume, input, out)
    }
}

#[inline(always)]
fn trend_follower_single_kernel(kernel: Kernel) -> Kernel {
    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    {
        if matches!(kernel, Kernel::Auto) && std::arch::is_x86_feature_detected!("avx2") {
            return Kernel::Avx2;
        }
    }
    kernel
}

#[inline]
pub fn trend_follower(
    input: &TrendFollowerInput<'_>,
) -> Result<TrendFollowerOutput, TrendFollowerError> {
    trend_follower_with_kernel(input, Kernel::Auto)
}

pub fn trend_follower_with_kernel(
    input: &TrendFollowerInput<'_>,
    kernel: Kernel,
) -> Result<TrendFollowerOutput, TrendFollowerError> {
    let (high, low, close, volume, params, first) = trend_follower_prepare(input)?;
    let kernel = trend_follower_single_kernel(kernel);
    let mut out = alloc_with_nan_prefix(close.len(), close.len());
    trend_follower_compute_into(
        high, low, close, volume, input, params, first, kernel, &mut out,
    )?;
    Ok(TrendFollowerOutput { values: out })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn trend_follower_into(
    input: &TrendFollowerInput<'_>,
    out: &mut [f64],
) -> Result<(), TrendFollowerError> {
    trend_follower_into_slice(out, input, Kernel::Auto)
}

pub fn trend_follower_into_slice(
    out: &mut [f64],
    input: &TrendFollowerInput<'_>,
    kernel: Kernel,
) -> Result<(), TrendFollowerError> {
    let (high, low, close, volume, params, first) = trend_follower_prepare(input)?;
    let kernel = trend_follower_single_kernel(kernel);
    if out.len() != close.len() {
        return Err(TrendFollowerError::OutputLengthMismatch {
            expected: close.len(),
            got: out.len(),
        });
    }
    out.fill(f64::NAN);
    trend_follower_compute_into(high, low, close, volume, input, params, first, kernel, out)
}

#[derive(Clone, Debug)]
pub struct TrendFollowerStream {
    matype: TrendFollowerMaType,
    trend_period: usize,
    ma_period: usize,
    channel_rate_fraction: f64,
    use_linear_regression: bool,
    linear_regression_period: usize,
    ma_stream: TrendFollowerBaseMaStream,
    linreg_stream: Option<LinRegStream>,
    index: usize,
    high_max: VecDeque<(usize, f64)>,
    low_min: VecDeque<(usize, f64)>,
    ma_max: VecDeque<(usize, f64)>,
    ma_min: VecDeque<(usize, f64)>,
}

impl TrendFollowerStream {
    #[inline]
    pub fn try_new(params: TrendFollowerParams) -> Result<Self, TrendFollowerError> {
        let input = TrendFollowerInput::from_slices(&[1.0], &[1.0], &[1.0], &[1.0], params);
        let resolved = resolve_params(&input, usize::MAX)?;
        let ma_stream = TrendFollowerBaseMaStream::new(resolved.matype, resolved.ma_period);
        let linreg_stream = if resolved.use_linear_regression {
            Some(
                LinRegStream::try_new(LinRegParams {
                    period: Some(resolved.linear_regression_period),
                })
                .map_err(|e| TrendFollowerError::LinearRegressionError(e.to_string()))?,
            )
        } else {
            None
        };
        Ok(Self {
            matype: resolved.matype,
            trend_period: resolved.trend_period,
            ma_period: resolved.ma_period,
            channel_rate_fraction: resolved.channel_rate_fraction,
            use_linear_regression: resolved.use_linear_regression,
            linear_regression_period: resolved.linear_regression_period,
            ma_stream,
            linreg_stream,
            index: 0,
            high_max: VecDeque::with_capacity(CHANNEL_WINDOW),
            low_min: VecDeque::with_capacity(CHANNEL_WINDOW),
            ma_max: VecDeque::with_capacity(resolved.trend_period.max(1)),
            ma_min: VecDeque::with_capacity(resolved.trend_period.max(1)),
        })
    }

    #[inline]
    pub fn reset(&mut self) -> Result<(), TrendFollowerError> {
        self.ma_stream = TrendFollowerBaseMaStream::new(self.matype, self.ma_period);
        self.linreg_stream = if self.use_linear_regression {
            Some(
                LinRegStream::try_new(LinRegParams {
                    period: Some(self.linear_regression_period),
                })
                .map_err(|e| TrendFollowerError::LinearRegressionError(e.to_string()))?,
            )
        } else {
            None
        };
        self.index = 0;
        self.high_max.clear();
        self.low_min.clear();
        self.ma_max.clear();
        self.ma_min.clear();
        Ok(())
    }

    #[inline]
    pub fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> Option<f64> {
        let needs_volume = self.matype == TrendFollowerMaType::Vwma;
        if !(high.is_finite() && low.is_finite() && close.is_finite())
            || (needs_volume && !volume.is_finite())
        {
            return None;
        }

        let idx = self.index;
        evict_front(&mut self.high_max, idx, CHANNEL_WINDOW);
        evict_front(&mut self.low_min, idx, CHANNEL_WINDOW);
        evict_front(&mut self.ma_max, idx, self.trend_period);
        evict_front(&mut self.ma_min, idx, self.trend_period);

        push_max(&mut self.high_max, idx, high, CHANNEL_WINDOW);
        push_min(&mut self.low_min, idx, low, CHANNEL_WINDOW);

        let base_ma = self.ma_stream.update(close, volume);
        let ma = if self.use_linear_regression {
            match (base_ma, self.linreg_stream.as_mut()) {
                (Some(value), Some(stream)) => stream.update(value),
                _ => None,
            }
        } else {
            base_ma
        };

        self.index = idx + 1;

        let Some(ma) = ma else {
            return None;
        };
        if ma.is_finite() {
            push_max(&mut self.ma_max, idx, ma, self.trend_period);
            push_min(&mut self.ma_min, idx, ma, self.trend_period);
        } else {
            return Some(f64::NAN);
        }

        let (hh, ll) = match (self.ma_max.front(), self.ma_min.front()) {
            (Some((_, hh)), Some((_, ll))) => (*hh, *ll),
            _ => return None,
        };
        let (channel_high, channel_low) = match (self.high_max.front(), self.low_min.front()) {
            (Some((_, hi)), Some((_, lo))) => (*hi, *lo),
            _ => return None,
        };
        let chan = (channel_high - channel_low) * self.channel_rate_fraction;
        if !chan.is_finite() || chan == 0.0 {
            return Some(f64::NAN);
        }

        let diff = (hh - ll).abs();
        let trend = if diff > chan {
            if ma > ll + chan {
                1.0
            } else if ma < hh - chan {
                -1.0
            } else {
                0.0
            }
        } else {
            0.0
        };
        Some(trend * diff / chan)
    }

    #[inline]
    pub fn update_reset_on_nan(
        &mut self,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Option<f64> {
        let needs_volume = self.matype == TrendFollowerMaType::Vwma;
        if !(high.is_finite() && low.is_finite() && close.is_finite())
            || (needs_volume && !volume.is_finite())
        {
            let _ = self.reset();
            return None;
        }
        self.update(high, low, close, volume)
    }
}

#[derive(Clone, Debug)]
pub struct TrendFollowerBatchRange {
    pub trend_period: (usize, usize, usize),
    pub ma_period: (usize, usize, usize),
    pub channel_rate_percent: (f64, f64, f64),
    pub linear_regression_period: (usize, usize, usize),
    pub matype: (String, String, String),
    pub use_linear_regression: bool,
}

impl Default for TrendFollowerBatchRange {
    fn default() -> Self {
        Self {
            trend_period: (20, 20, 0),
            ma_period: (20, 20, 0),
            channel_rate_percent: (1.0, 1.0, 0.0),
            linear_regression_period: (5, 5, 0),
            matype: ("ema".to_string(), "ema".to_string(), String::new()),
            use_linear_regression: true,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TrendFollowerBatchBuilder {
    range: TrendFollowerBatchRange,
    kernel: Kernel,
}

impl TrendFollowerBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    pub fn trend_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.trend_period = (start, end, step);
        self
    }

    pub fn ma_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.ma_period = (start, end, step);
        self
    }

    pub fn channel_rate_percent_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.channel_rate_percent = (start, end, step);
        self
    }

    pub fn linear_regression_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.linear_regression_period = (start, end, step);
        self
    }

    pub fn matype_static<S: Into<String>>(mut self, value: S) -> Self {
        let value = value.into();
        self.range.matype = (value.clone(), value, String::new());
        self
    }

    pub fn use_linear_regression(mut self, value: bool) -> Self {
        self.range.use_linear_regression = value;
        self
    }

    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Result<TrendFollowerBatchOutput, TrendFollowerError> {
        trend_follower_batch_with_kernel(high, low, close, volume, &self.range, self.kernel)
    }

    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<TrendFollowerBatchOutput, TrendFollowerError> {
        self.apply_slices(&candles.high, &candles.low, &candles.close, &candles.volume)
    }
}

#[derive(Clone, Debug)]
pub struct TrendFollowerBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<TrendFollowerParams>,
    pub rows: usize,
    pub cols: usize,
}

impl TrendFollowerBatchOutput {
    pub fn row_for_params(&self, params: &TrendFollowerParams) -> Option<usize> {
        let matype = params
            .matype
            .as_deref()
            .unwrap_or("ema")
            .to_ascii_lowercase();
        self.combos.iter().position(|combo| {
            combo.trend_period.unwrap_or(20) == params.trend_period.unwrap_or(20)
                && combo.ma_period.unwrap_or(20) == params.ma_period.unwrap_or(20)
                && (combo.channel_rate_percent.unwrap_or(1.0)
                    - params.channel_rate_percent.unwrap_or(1.0))
                .abs()
                    <= 1e-12
                && combo.use_linear_regression.unwrap_or(true)
                    == params.use_linear_regression.unwrap_or(true)
                && combo.linear_regression_period.unwrap_or(5)
                    == params.linear_regression_period.unwrap_or(5)
                && combo
                    .matype
                    .as_deref()
                    .unwrap_or("ema")
                    .eq_ignore_ascii_case(&matype)
        })
    }

    pub fn values_for(&self, params: &TrendFollowerParams) -> Option<&[f64]> {
        self.row_for_params(params).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline]
fn axis_usize(range: (usize, usize, usize)) -> Result<Vec<usize>, TrendFollowerError> {
    let (start, end, step) = range;
    if start == 0 || end == 0 {
        return Err(TrendFollowerError::InvalidRangeUsize { start, end, step });
    }
    if step == 0 || start == end {
        return Ok(vec![start]);
    }
    let mut out = Vec::new();
    if start < end {
        let mut value = start;
        while value <= end {
            out.push(value);
            match value.checked_add(step) {
                Some(next) if next > value => value = next,
                _ => break,
            }
        }
    } else {
        let mut value = start;
        while value >= end {
            out.push(value);
            if value < end + step {
                break;
            }
            value = value.saturating_sub(step);
            if value == 0 {
                break;
            }
        }
    }
    if out.is_empty() {
        return Err(TrendFollowerError::InvalidRangeUsize { start, end, step });
    }
    Ok(out)
}

#[inline]
fn axis_f64(range: (f64, f64, f64)) -> Result<Vec<f64>, TrendFollowerError> {
    let (start, end, step) = range;
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(TrendFollowerError::InvalidRangeF64 { start, end, step });
    }
    if step == 0.0 || (start - end).abs() <= 1e-12 {
        return Ok(vec![start]);
    }
    if step < 0.0 {
        return Err(TrendFollowerError::InvalidRangeF64 { start, end, step });
    }
    let mut out = Vec::new();
    if start < end {
        let mut value = start;
        while value <= end + 1e-12 {
            out.push(value);
            value += step;
        }
    } else {
        let mut value = start;
        while value >= end - 1e-12 {
            out.push(value);
            value -= step;
        }
    }
    if out.is_empty() {
        return Err(TrendFollowerError::InvalidRangeF64 { start, end, step });
    }
    Ok(out)
}

#[inline]
fn axis_string(range: (String, String, String)) -> Vec<String> {
    if range.0.eq_ignore_ascii_case(&range.1) {
        vec![range.0]
    } else {
        vec![range.0, range.1]
    }
}

pub fn expand_grid_trend_follower(
    range: &TrendFollowerBatchRange,
) -> Result<Vec<TrendFollowerParams>, TrendFollowerError> {
    let trend_periods = axis_usize(range.trend_period)?;
    let ma_periods = axis_usize(range.ma_period)?;
    let channel_rates = axis_f64(range.channel_rate_percent)?;
    let linear_regression_periods = axis_usize(range.linear_regression_period)?;
    let matypes = axis_string(range.matype.clone());

    let mut out = Vec::new();
    for trend_period in &trend_periods {
        for ma_period in &ma_periods {
            for channel_rate_percent in &channel_rates {
                for linear_regression_period in &linear_regression_periods {
                    for matype in &matypes {
                        out.push(TrendFollowerParams {
                            matype: Some(matype.to_ascii_lowercase()),
                            trend_period: Some(*trend_period),
                            ma_period: Some(*ma_period),
                            channel_rate_percent: Some(*channel_rate_percent),
                            use_linear_regression: Some(range.use_linear_regression),
                            linear_regression_period: Some(*linear_regression_period),
                        });
                    }
                }
            }
        }
    }
    Ok(out)
}

pub fn trend_follower_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    range: &TrendFollowerBatchRange,
    kernel: Kernel,
) -> Result<TrendFollowerBatchOutput, TrendFollowerError> {
    let batch_kernel = match kernel {
        Kernel::Auto => Kernel::ScalarBatch,
        other if other.is_batch() => other,
        other => return Err(TrendFollowerError::InvalidKernelForBatch(other)),
    };
    trend_follower_batch_impl(
        high,
        low,
        close,
        volume,
        range,
        batch_kernel.to_non_batch(),
        true,
    )
}

pub fn trend_follower_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    range: &TrendFollowerBatchRange,
) -> Result<TrendFollowerBatchOutput, TrendFollowerError> {
    trend_follower_batch_impl(high, low, close, volume, range, Kernel::Scalar, false)
}

pub fn trend_follower_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    range: &TrendFollowerBatchRange,
) -> Result<TrendFollowerBatchOutput, TrendFollowerError> {
    trend_follower_batch_impl(high, low, close, volume, range, Kernel::Scalar, true)
}

fn trend_follower_batch_impl(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    range: &TrendFollowerBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<TrendFollowerBatchOutput, TrendFollowerError> {
    if high.len() != low.len() || high.len() != close.len() || high.len() != volume.len() {
        return Err(TrendFollowerError::DataLengthMismatch {
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
            volume_len: volume.len(),
        });
    }
    if high.is_empty() {
        return Err(TrendFollowerError::EmptyInputData);
    }

    let combos = expand_grid_trend_follower(range)?;
    let rows = combos.len();
    let cols = close.len();
    let mut matrix = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut matrix, cols, &vec![cols; rows]);

    let mut guard = ManuallyDrop::new(matrix);
    let out_mu: &mut [MaybeUninit<f64>] =
        unsafe { std::slice::from_raw_parts_mut(guard.as_mut_ptr(), guard.len()) };

    let do_row = |row: usize, row_mu: &mut [MaybeUninit<f64>]| {
        let out = unsafe {
            std::slice::from_raw_parts_mut(row_mu.as_mut_ptr() as *mut f64, row_mu.len())
        };
        let input = TrendFollowerInput::from_slices(high, low, close, volume, combos[row].clone());
        let _ = trend_follower_into_slice(out, &input, kernel);
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out_mu
            .par_chunks_mut(cols)
            .enumerate()
            .for_each(|(row, row_mu)| do_row(row, row_mu));
        #[cfg(target_arch = "wasm32")]
        for (row, row_mu) in out_mu.chunks_mut(cols).enumerate() {
            do_row(row, row_mu);
        }
    } else {
        for (row, row_mu) in out_mu.chunks_mut(cols).enumerate() {
            do_row(row, row_mu);
        }
    }

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(TrendFollowerBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

fn trend_follower_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    range: &TrendFollowerBatchRange,
    kernel: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<(), TrendFollowerError> {
    if high.len() != low.len() || high.len() != close.len() || high.len() != volume.len() {
        return Err(TrendFollowerError::DataLengthMismatch {
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
            volume_len: volume.len(),
        });
    }
    let combos = expand_grid_trend_follower(range)?;
    let rows = combos.len();
    let cols = close.len();
    if rows.checked_mul(cols) != Some(out.len()) {
        return Err(TrendFollowerError::OutputLengthMismatch {
            expected: rows * cols,
            got: out.len(),
        });
    }

    for row_out in out.chunks_mut(cols) {
        row_out.fill(f64::NAN);
    }

    let do_row = |row: usize, row_out: &mut [f64]| {
        let input = TrendFollowerInput::from_slices(high, low, close, volume, combos[row].clone());
        let _ = trend_follower_into_slice(row_out, &input, kernel);
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out.par_chunks_mut(cols)
            .enumerate()
            .for_each(|(row, row_out)| do_row(row, row_out));
        #[cfg(target_arch = "wasm32")]
        for (row, row_out) in out.chunks_mut(cols).enumerate() {
            do_row(row, row_out);
        }
    } else {
        for (row, row_out) in out.chunks_mut(cols).enumerate() {
            do_row(row, row_out);
        }
    }

    Ok(())
}

#[cfg(feature = "python")]
#[pyfunction(name = "trend_follower")]
#[pyo3(signature = (high, low, close, volume, matype="ema", trend_period=20, ma_period=20, channel_rate_percent=1.0, use_linear_regression=true, linear_regression_period=5, kernel=None))]
pub fn trend_follower_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    matype: &str,
    trend_period: usize,
    ma_period: usize,
    channel_rate_percent: f64,
    use_linear_regression: bool,
    linear_regression_period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let volume = volume.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = TrendFollowerInput::from_slices(
        high,
        low,
        close,
        volume,
        TrendFollowerParams {
            matype: Some(matype.to_string()),
            trend_period: Some(trend_period),
            ma_period: Some(ma_period),
            channel_rate_percent: Some(channel_rate_percent),
            use_linear_regression: Some(use_linear_regression),
            linear_regression_period: Some(linear_regression_period),
        },
    );
    let output = py
        .allow_threads(|| trend_follower_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(output.values.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "TrendFollowerStream")]
pub struct TrendFollowerStreamPy {
    stream: TrendFollowerStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl TrendFollowerStreamPy {
    #[new]
    #[pyo3(signature = (matype="ema", trend_period=20, ma_period=20, channel_rate_percent=1.0, use_linear_regression=true, linear_regression_period=5))]
    fn new(
        matype: &str,
        trend_period: usize,
        ma_period: usize,
        channel_rate_percent: f64,
        use_linear_regression: bool,
        linear_regression_period: usize,
    ) -> PyResult<Self> {
        let stream = TrendFollowerStream::try_new(TrendFollowerParams {
            matype: Some(matype.to_string()),
            trend_period: Some(trend_period),
            ma_period: Some(ma_period),
            channel_rate_percent: Some(channel_rate_percent),
            use_linear_regression: Some(use_linear_regression),
            linear_regression_period: Some(linear_regression_period),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> Option<f64> {
        self.stream.update_reset_on_nan(high, low, close, volume)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "trend_follower_batch")]
#[pyo3(signature = (high, low, close, volume, trend_period_range=(20, 20, 0), ma_period_range=(20, 20, 0), channel_rate_percent_range=(1.0, 1.0, 0.0), linear_regression_period_range=(5, 5, 0), matype="ema", use_linear_regression=true, kernel=None))]
pub fn trend_follower_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    trend_period_range: (usize, usize, usize),
    ma_period_range: (usize, usize, usize),
    channel_rate_percent_range: (f64, f64, f64),
    linear_regression_period_range: (usize, usize, usize),
    matype: &str,
    use_linear_regression: bool,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let volume = volume.as_slice()?;
    let range = TrendFollowerBatchRange {
        trend_period: trend_period_range,
        ma_period: ma_period_range,
        channel_rate_percent: channel_rate_percent_range,
        linear_regression_period: linear_regression_period_range,
        matype: (matype.to_string(), matype.to_string(), String::new()),
        use_linear_regression,
    };
    let combos =
        expand_grid_trend_follower(&range).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;
    let arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out = unsafe { arr.as_slice_mut()? };
    let kernel = validate_kernel(kernel, true)?;

    py.allow_threads(|| {
        let batch_kernel = match kernel {
            Kernel::Auto => detect_best_batch_kernel(),
            other => other,
        };
        trend_follower_batch_inner_into(
            high,
            low,
            close,
            volume,
            &range,
            batch_kernel.to_non_batch(),
            true,
            out,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", arr.reshape((rows, cols))?)?;
    dict.set_item(
        "trend_periods",
        combos
            .iter()
            .map(|params| params.trend_period.unwrap_or(20) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "ma_periods",
        combos
            .iter()
            .map(|params| params.ma_period.unwrap_or(20) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "channel_rate_percents",
        combos
            .iter()
            .map(|params| params.channel_rate_percent.unwrap_or(1.0))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "linear_regression_periods",
        combos
            .iter()
            .map(|params| params.linear_regression_period.unwrap_or(5) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "matypes",
        combos
            .iter()
            .map(|params| params.matype.as_deref().unwrap_or("ema").to_string())
            .collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "use_linear_regression",
        combos
            .iter()
            .map(|params| params.use_linear_regression.unwrap_or(true))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_trend_follower_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(trend_follower_py, m)?)?;
    m.add_function(wrap_pyfunction!(trend_follower_batch_py, m)?)?;
    m.add_class::<TrendFollowerStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TrendFollowerBatchConfig {
    trend_period_range: Vec<usize>,
    ma_period_range: Vec<usize>,
    channel_rate_percent_range: Vec<f64>,
    linear_regression_period_range: Vec<usize>,
    matype: String,
    use_linear_regression: bool,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TrendFollowerBatchJsOutput {
    values: Vec<f64>,
    rows: usize,
    cols: usize,
    combos: Vec<TrendFollowerParams>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "trend_follower_js")]
pub fn trend_follower_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    matype: &str,
    trend_period: usize,
    ma_period: usize,
    channel_rate_percent: f64,
    use_linear_regression: bool,
    linear_regression_period: usize,
) -> Result<Vec<f64>, JsValue> {
    let input = TrendFollowerInput::from_slices(
        high,
        low,
        close,
        volume,
        TrendFollowerParams {
            matype: Some(matype.to_string()),
            trend_period: Some(trend_period),
            ma_period: Some(ma_period),
            channel_rate_percent: Some(channel_rate_percent),
            use_linear_regression: Some(use_linear_regression),
            linear_regression_period: Some(linear_regression_period),
        },
    );
    trend_follower(&input)
        .map(|out| out.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "trend_follower_batch_js")]
pub fn trend_follower_batch_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: TrendFollowerBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.trend_period_range.len() != 3
        || config.ma_period_range.len() != 3
        || config.channel_rate_percent_range.len() != 3
        || config.linear_regression_period_range.len() != 3
    {
        return Err(JsValue::from_str(
            "Invalid config: all *_range fields must have exactly 3 elements",
        ));
    }
    let range = TrendFollowerBatchRange {
        trend_period: (
            config.trend_period_range[0],
            config.trend_period_range[1],
            config.trend_period_range[2],
        ),
        ma_period: (
            config.ma_period_range[0],
            config.ma_period_range[1],
            config.ma_period_range[2],
        ),
        channel_rate_percent: (
            config.channel_rate_percent_range[0],
            config.channel_rate_percent_range[1],
            config.channel_rate_percent_range[2],
        ),
        linear_regression_period: (
            config.linear_regression_period_range[0],
            config.linear_regression_period_range[1],
            config.linear_regression_period_range[2],
        ),
        matype: (config.matype.clone(), config.matype, String::new()),
        use_linear_regression: config.use_linear_regression,
    };
    let batch = trend_follower_batch_slice(high, low, close, volume, &range)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&TrendFollowerBatchJsOutput {
        values: batch.values,
        rows: batch.rows,
        cols: batch.cols,
        combos: batch.combos,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn trend_follower_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn trend_follower_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, 0, len);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn trend_follower_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    volume_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    matype: &str,
    trend_period: usize,
    ma_period: usize,
    channel_rate_percent: f64,
    use_linear_regression: bool,
    linear_regression_period: usize,
) -> Result<(), JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || volume_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to trend_follower_into",
        ));
    }
    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len);
        let input = TrendFollowerInput::from_slices(
            high,
            low,
            close,
            volume,
            TrendFollowerParams {
                matype: Some(matype.to_string()),
                trend_period: Some(trend_period),
                ma_period: Some(ma_period),
                channel_rate_percent: Some(channel_rate_percent),
                use_linear_regression: Some(use_linear_regression),
                linear_regression_period: Some(linear_regression_period),
            },
        );
        trend_follower_into_slice(out, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "trend_follower_into_host")]
pub fn trend_follower_into_host(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    out_ptr: *mut f64,
    matype: &str,
    trend_period: usize,
    ma_period: usize,
    channel_rate_percent: f64,
    use_linear_regression: bool,
    linear_regression_period: usize,
) -> Result<(), JsValue> {
    if out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to trend_follower_into_host",
        ));
    }
    unsafe {
        let out = std::slice::from_raw_parts_mut(out_ptr, close.len());
        let input = TrendFollowerInput::from_slices(
            high,
            low,
            close,
            volume,
            TrendFollowerParams {
                matype: Some(matype.to_string()),
                trend_period: Some(trend_period),
                ma_period: Some(ma_period),
                channel_rate_percent: Some(channel_rate_percent),
                use_linear_regression: Some(use_linear_regression),
                linear_regression_period: Some(linear_regression_period),
            },
        );
        trend_follower_into_slice(out, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn trend_follower_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    volume_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    trend_period_start: usize,
    trend_period_end: usize,
    trend_period_step: usize,
    ma_period_start: usize,
    ma_period_end: usize,
    ma_period_step: usize,
    channel_rate_percent_start: f64,
    channel_rate_percent_end: f64,
    channel_rate_percent_step: f64,
    linear_regression_period_start: usize,
    linear_regression_period_end: usize,
    linear_regression_period_step: usize,
    matype: &str,
    use_linear_regression: bool,
) -> Result<usize, JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || volume_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to trend_follower_batch_into",
        ));
    }
    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);
        let range = TrendFollowerBatchRange {
            trend_period: (trend_period_start, trend_period_end, trend_period_step),
            ma_period: (ma_period_start, ma_period_end, ma_period_step),
            channel_rate_percent: (
                channel_rate_percent_start,
                channel_rate_percent_end,
                channel_rate_percent_step,
            ),
            linear_regression_period: (
                linear_regression_period_start,
                linear_regression_period_end,
                linear_regression_period_step,
            ),
            matype: (matype.to_string(), matype.to_string(), String::new()),
            use_linear_regression,
        };
        let combos =
            expand_grid_trend_follower(&range).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let out = std::slice::from_raw_parts_mut(out_ptr, rows * len);
        trend_follower_batch_inner_into(
            high,
            low,
            close,
            volume,
            &range,
            Kernel::Scalar,
            false,
            out,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn trend_follower_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    matype: &str,
    trend_period: usize,
    ma_period: usize,
    channel_rate_percent: f64,
    use_linear_regression: bool,
    linear_regression_period: usize,
    out: &js_sys::Float64Array,
) -> Result<usize, JsValue> {
    let values = trend_follower_js(
        high,
        low,
        close,
        volume,
        matype,
        trend_period,
        ma_period,
        channel_rate_percent,
        use_linear_regression,
        linear_regression_period,
    )?;
    crate::write_wasm_f64_output("trend_follower_output_into_js", &values, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn trend_follower_batch_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = trend_follower_batch_js(high, low, close, volume, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "trend_follower_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_ohlcv(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut high = Vec::with_capacity(len);
        let mut low = Vec::with_capacity(len);
        let mut close = Vec::with_capacity(len);
        let mut volume = Vec::with_capacity(len);
        for i in 0..len {
            let base = 100.0 + i as f64 * 0.19 + (i as f64 * 0.13).sin() * 1.8;
            let c = base + (i as f64 * 0.027).cos() * 0.6;
            high.push(c + 1.2 + (i as f64 * 0.05).sin().abs());
            low.push(c - 1.1 - (i as f64 * 0.04).cos().abs());
            close.push(c);
            volume.push(1000.0 + i as f64 * 11.0 + (i % 9) as f64 * 17.0);
        }
        (high, low, close, volume)
    }

    fn assert_close(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            if a[i].is_nan() || b[i].is_nan() {
                assert!(a[i].is_nan() && b[i].is_nan(), "nan mismatch at {i}");
            } else {
                assert!(
                    (a[i] - b[i]).abs() <= 1e-9,
                    "value mismatch at {i}: {} vs {}",
                    a[i],
                    b[i]
                );
            }
        }
    }

    #[test]
    fn trend_follower_into_matches_api() {
        let (high, low, close, volume) = sample_ohlcv(128);
        let input = TrendFollowerInput::from_slices(
            &high,
            &low,
            &close,
            &volume,
            TrendFollowerParams::default(),
        );
        let direct = trend_follower(&input).unwrap();
        let mut out = vec![f64::NAN; close.len()];
        trend_follower_into_slice(&mut out, &input, Kernel::Auto).unwrap();
        assert_close(&direct.values, &out);
    }

    #[test]
    fn trend_follower_stream_matches_batch_with_nan_gap() {
        let (mut high, mut low, mut close, mut volume) = sample_ohlcv(128);
        high[48] = f64::NAN;
        low[48] = f64::NAN;
        close[48] = f64::NAN;
        volume[48] = f64::NAN;
        let input = TrendFollowerInput::from_slices(
            &high,
            &low,
            &close,
            &volume,
            TrendFollowerParams::default(),
        );
        let batch = trend_follower(&input).unwrap();
        let mut stream = TrendFollowerStream::try_new(TrendFollowerParams::default()).unwrap();
        let mut collected = Vec::with_capacity(close.len());
        for i in 0..close.len() {
            collected.push(
                stream
                    .update_reset_on_nan(high[i], low[i], close[i], volume[i])
                    .unwrap_or(f64::NAN),
            );
        }
        assert_close(&batch.values, &collected);
    }

    #[test]
    fn trend_follower_batch_single_param_matches_single() {
        let (high, low, close, volume) = sample_ohlcv(128);
        let params = TrendFollowerParams {
            matype: Some("wma".to_string()),
            trend_period: Some(14),
            ma_period: Some(9),
            channel_rate_percent: Some(1.1),
            use_linear_regression: Some(false),
            linear_regression_period: Some(5),
        };
        let single = trend_follower(&TrendFollowerInput::from_slices(
            &high,
            &low,
            &close,
            &volume,
            params.clone(),
        ))
        .unwrap();
        let batch = trend_follower_batch_with_kernel(
            &high,
            &low,
            &close,
            &volume,
            &TrendFollowerBatchRange {
                trend_period: (14, 14, 0),
                ma_period: (9, 9, 0),
                channel_rate_percent: (1.1, 1.1, 0.0),
                linear_regression_period: (5, 5, 0),
                matype: ("wma".to_string(), "wma".to_string(), String::new()),
                use_linear_regression: false,
            },
            Kernel::Auto,
        )
        .unwrap();
        assert_eq!(batch.rows, 1);
        assert_close(&single.values, &batch.values[..close.len()]);
    }

    #[test]
    fn trend_follower_vwma_depends_on_volume() {
        let (high, low, close, volume) = sample_ohlcv(96);
        let mut volume_b = volume.clone();
        volume_b.reverse();
        let params = TrendFollowerParams {
            matype: Some("vwma".to_string()),
            trend_period: Some(20),
            ma_period: Some(12),
            channel_rate_percent: Some(1.0),
            use_linear_regression: Some(false),
            linear_regression_period: Some(5),
        };
        let a = trend_follower(&TrendFollowerInput::from_slices(
            &high,
            &low,
            &close,
            &volume,
            params.clone(),
        ))
        .unwrap();
        let b = trend_follower(&TrendFollowerInput::from_slices(
            &high, &low, &close, &volume_b, params,
        ))
        .unwrap();
        assert!(a
            .values
            .iter()
            .zip(&b.values)
            .any(|(x, y)| x.is_finite() && y.is_finite() && (x - y).abs() > 1e-9));
    }

    #[test]
    fn trend_follower_invalid_matype_rejected() {
        let (high, low, close, volume) = sample_ohlcv(64);
        let err = trend_follower(&TrendFollowerInput::from_slices(
            &high,
            &low,
            &close,
            &volume,
            TrendFollowerParams {
                matype: Some("hma".to_string()),
                ..TrendFollowerParams::default()
            },
        ))
        .unwrap_err();
        assert!(matches!(err, TrendFollowerError::InvalidMaType { .. }));
    }
}
