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

use crate::utilities::data_loader::Candles;
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_uninit_f64, detect_best_batch_kernel, init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

const DEFAULT_SOURCE: &str = "hlc3";
const DEFAULT_LENGTH: usize = 21;
const DEFAULT_ATR_LENGTH: usize = 14;
const DEFAULT_USE_ATR: bool = true;
const DEFAULT_TP_AGGRESSIVENESS: &str = "low";
const MULT1: f64 = 0.618;
const MULT2: f64 = 1.0;
const MULT3: f64 = 1.618;
const MULT4: f64 = 2.618;
const FLOAT_TOL: f64 = 1e-12;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SourceKind {
    Open,
    High,
    Low,
    Close,
    Hl2,
    Hlc3,
    Ohlc4,
    Hlcc4,
}

impl SourceKind {
    #[inline(always)]
    fn parse(value: &str) -> Option<Self> {
        if value.eq_ignore_ascii_case("open") {
            Some(Self::Open)
        } else if value.eq_ignore_ascii_case("high") {
            Some(Self::High)
        } else if value.eq_ignore_ascii_case("low") {
            Some(Self::Low)
        } else if value.eq_ignore_ascii_case("close") {
            Some(Self::Close)
        } else if value.eq_ignore_ascii_case("hl2") {
            Some(Self::Hl2)
        } else if value.eq_ignore_ascii_case("hlc3") {
            Some(Self::Hlc3)
        } else if value.eq_ignore_ascii_case("ohlc4") {
            Some(Self::Ohlc4)
        } else if value.eq_ignore_ascii_case("hlcc4") || value.eq_ignore_ascii_case("hlcc") {
            Some(Self::Hlcc4)
        } else {
            None
        }
    }

    #[inline(always)]
    fn as_str(self) -> &'static str {
        match self {
            Self::Open => "open",
            Self::High => "high",
            Self::Low => "low",
            Self::Close => "close",
            Self::Hl2 => "hl2",
            Self::Hlc3 => "hlc3",
            Self::Ohlc4 => "ohlc4",
            Self::Hlcc4 => "hlcc4",
        }
    }

    #[inline(always)]
    fn needs_open(self) -> bool {
        matches!(self, Self::Open | Self::Ohlc4)
    }

    #[inline(always)]
    fn value(self, open: f64, high: f64, low: f64, close: f64) -> f64 {
        match self {
            Self::Open => open,
            Self::High => high,
            Self::Low => low,
            Self::Close => close,
            Self::Hl2 => 0.5 * (high + low),
            Self::Hlc3 => (high + low + close) / 3.0,
            Self::Ohlc4 => (open + high + low + close) * 0.25,
            Self::Hlcc4 => (high + low + close + close) * 0.25,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TpAggressiveness {
    Low,
    Medium,
    High,
}

impl TpAggressiveness {
    #[inline(always)]
    fn parse(value: &str) -> Option<Self> {
        if value.eq_ignore_ascii_case("low") {
            Some(Self::Low)
        } else if value.eq_ignore_ascii_case("medium") {
            Some(Self::Medium)
        } else if value.eq_ignore_ascii_case("high") {
            Some(Self::High)
        } else {
            None
        }
    }

    #[inline(always)]
    fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }
}

#[derive(Debug, Clone)]
pub enum FibonacciEntryBandsData<'a> {
    Candles(&'a Candles),
    Slices {
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct FibonacciEntryBandsOutput {
    pub basis: Vec<f64>,
    pub trend: Vec<f64>,
    pub upper_0618: Vec<f64>,
    pub upper_1000: Vec<f64>,
    pub upper_1618: Vec<f64>,
    pub upper_2618: Vec<f64>,
    pub lower_0618: Vec<f64>,
    pub lower_1000: Vec<f64>,
    pub lower_1618: Vec<f64>,
    pub lower_2618: Vec<f64>,
    pub tp_long_band: Vec<f64>,
    pub tp_short_band: Vec<f64>,
    pub long_entry: Vec<f64>,
    pub short_entry: Vec<f64>,
    pub rejection_long: Vec<f64>,
    pub rejection_short: Vec<f64>,
    pub long_bounce: Vec<f64>,
    pub short_bounce: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
pub struct FibonacciEntryBandsPoint {
    pub basis: f64,
    pub trend: f64,
    pub upper_0618: f64,
    pub upper_1000: f64,
    pub upper_1618: f64,
    pub upper_2618: f64,
    pub lower_0618: f64,
    pub lower_1000: f64,
    pub lower_1618: f64,
    pub lower_2618: f64,
    pub tp_long_band: f64,
    pub tp_short_band: f64,
    pub long_entry: f64,
    pub short_entry: f64,
    pub rejection_long: f64,
    pub rejection_short: f64,
    pub long_bounce: f64,
    pub short_bounce: f64,
}

impl FibonacciEntryBandsPoint {
    #[inline(always)]
    fn nan() -> Self {
        Self {
            basis: f64::NAN,
            trend: f64::NAN,
            upper_0618: f64::NAN,
            upper_1000: f64::NAN,
            upper_1618: f64::NAN,
            upper_2618: f64::NAN,
            lower_0618: f64::NAN,
            lower_1000: f64::NAN,
            lower_1618: f64::NAN,
            lower_2618: f64::NAN,
            tp_long_band: f64::NAN,
            tp_short_band: f64::NAN,
            long_entry: f64::NAN,
            short_entry: f64::NAN,
            rejection_long: f64::NAN,
            rejection_short: f64::NAN,
            long_bounce: f64::NAN,
            short_bounce: f64::NAN,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct FibonacciEntryBandsParams {
    pub source: Option<String>,
    pub length: Option<usize>,
    pub atr_length: Option<usize>,
    pub use_atr: Option<bool>,
    pub tp_aggressiveness: Option<String>,
}

impl Default for FibonacciEntryBandsParams {
    fn default() -> Self {
        Self {
            source: Some(DEFAULT_SOURCE.to_string()),
            length: Some(DEFAULT_LENGTH),
            atr_length: Some(DEFAULT_ATR_LENGTH),
            use_atr: Some(DEFAULT_USE_ATR),
            tp_aggressiveness: Some(DEFAULT_TP_AGGRESSIVENESS.to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FibonacciEntryBandsInput<'a> {
    pub data: FibonacciEntryBandsData<'a>,
    pub params: FibonacciEntryBandsParams,
}

impl<'a> FibonacciEntryBandsInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: FibonacciEntryBandsParams) -> Self {
        Self {
            data: FibonacciEntryBandsData::Candles(candles),
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: FibonacciEntryBandsParams,
    ) -> Self {
        Self {
            data: FibonacciEntryBandsData::Slices {
                open,
                high,
                low,
                close,
            },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, FibonacciEntryBandsParams::default())
    }

    #[inline]
    pub fn as_slices(&self) -> (&'a [f64], &'a [f64], &'a [f64], &'a [f64]) {
        match &self.data {
            FibonacciEntryBandsData::Candles(candles) => {
                (&candles.open, &candles.high, &candles.low, &candles.close)
            }
            FibonacciEntryBandsData::Slices {
                open,
                high,
                low,
                close,
            } => (open, high, low, close),
        }
    }
}

#[derive(Clone, Debug)]
pub struct FibonacciEntryBandsBuilder {
    source: Option<SourceKind>,
    length: Option<usize>,
    atr_length: Option<usize>,
    use_atr: Option<bool>,
    tp_aggressiveness: Option<TpAggressiveness>,
    kernel: Kernel,
}

impl Default for FibonacciEntryBandsBuilder {
    fn default() -> Self {
        Self {
            source: None,
            length: None,
            atr_length: None,
            use_atr: None,
            tp_aggressiveness: None,
            kernel: Kernel::Auto,
        }
    }
}

#[derive(Debug, Error)]
pub enum FibonacciEntryBandsError {
    #[error("fibonacci_entry_bands: Input data slice is empty.")]
    EmptyInputData,
    #[error("fibonacci_entry_bands: All values are NaN.")]
    AllValuesNaN,
    #[error(
        "fibonacci_entry_bands: Inconsistent slice lengths: open={open_len}, high={high_len}, low={low_len}, close={close_len}"
    )]
    InconsistentSliceLengths {
        open_len: usize,
        high_len: usize,
        low_len: usize,
        close_len: usize,
    },
    #[error("fibonacci_entry_bands: Invalid length: length = {length}, data length = {data_len}")]
    InvalidLength { length: usize, data_len: usize },
    #[error(
        "fibonacci_entry_bands: Invalid atr_length: atr_length = {atr_length}, data length = {data_len}"
    )]
    InvalidAtrLength { atr_length: usize, data_len: usize },
    #[error(
        "fibonacci_entry_bands: Invalid source: {source_name}. Supported: open, high, low, close, hl2, hlc3, ohlc4, hlcc4"
    )]
    InvalidSource { source_name: String },
    #[error(
        "fibonacci_entry_bands: Invalid TP aggressiveness: {tp_aggressiveness}. Supported: low, medium, high"
    )]
    InvalidTpAggressiveness { tp_aggressiveness: String },
    #[error("fibonacci_entry_bands: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("fibonacci_entry_bands: Output length mismatch: expected = {expected}")]
    OutputLengthMismatch { expected: usize },
    #[error("fibonacci_entry_bands: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("fibonacci_entry_bands: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Debug, Clone, Copy)]
struct ResolvedParams {
    source: SourceKind,
    length: usize,
    atr_length: usize,
    use_atr: bool,
    tp_aggressiveness: TpAggressiveness,
    ema_alpha: f64,
}

#[derive(Debug, Clone)]
struct RollingStdev {
    values: Vec<f64>,
    head: usize,
    count: usize,
    sum: f64,
    sum_sq: f64,
}

impl RollingStdev {
    #[inline]
    fn new(length: usize) -> Self {
        Self {
            values: vec![0.0; length.max(1)],
            head: 0,
            count: 0,
            sum: 0.0,
            sum_sq: 0.0,
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.head = 0;
        self.count = 0;
        self.sum = 0.0;
        self.sum_sq = 0.0;
    }

    #[inline]
    fn update(&mut self, value: f64) -> Option<f64> {
        let len = self.values.len();
        if self.count == len {
            let old = self.values[self.head];
            self.sum -= old;
            self.sum_sq -= old * old;
        } else {
            self.count += 1;
        }
        self.values[self.head] = value;
        self.head += 1;
        if self.head == len {
            self.head = 0;
        }
        self.sum += value;
        self.sum_sq += value * value;
        if self.count < len {
            return None;
        }
        let mean = self.sum / len as f64;
        Some((self.sum_sq / len as f64 - mean * mean).max(0.0).sqrt())
    }
}

#[derive(Debug, Clone)]
pub struct FibonacciEntryBandsStream {
    params: ResolvedParams,
    ema1: Option<f64>,
    ema2: Option<f64>,
    prev_basis: Option<f64>,
    prev_prev_basis: Option<f64>,
    trend: f64,
    prev_close: Option<f64>,
    atr_sum: f64,
    atr_count: usize,
    atr_value: Option<f64>,
    stdev: RollingStdev,
    prev_tp_long_band: Option<f64>,
    prev_tp_short_band: Option<f64>,
}

impl FibonacciEntryBandsBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn source(mut self, value: &str) -> Result<Self, FibonacciEntryBandsError> {
        self.source = Some(parse_source(value)?);
        Ok(self)
    }

    #[inline]
    pub fn length(mut self, value: usize) -> Self {
        self.length = Some(value);
        self
    }

    #[inline]
    pub fn atr_length(mut self, value: usize) -> Self {
        self.atr_length = Some(value);
        self
    }

    #[inline]
    pub fn use_atr(mut self, value: bool) -> Self {
        self.use_atr = Some(value);
        self
    }

    #[inline]
    pub fn tp_aggressiveness(mut self, value: &str) -> Result<Self, FibonacciEntryBandsError> {
        self.tp_aggressiveness = Some(parse_tp_aggressiveness(value)?);
        Ok(self)
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
    ) -> Result<FibonacciEntryBandsOutput, FibonacciEntryBandsError> {
        let input = FibonacciEntryBandsInput::from_candles(
            candles,
            FibonacciEntryBandsParams {
                source: Some(self.source.unwrap_or(SourceKind::Hlc3).as_str().to_string()),
                length: self.length,
                atr_length: self.atr_length,
                use_atr: self.use_atr,
                tp_aggressiveness: Some(
                    self.tp_aggressiveness
                        .unwrap_or(TpAggressiveness::Low)
                        .as_str()
                        .to_string(),
                ),
            },
        );
        fibonacci_entry_bands_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn apply_slices(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<FibonacciEntryBandsOutput, FibonacciEntryBandsError> {
        let input = FibonacciEntryBandsInput::from_slices(
            open,
            high,
            low,
            close,
            FibonacciEntryBandsParams {
                source: Some(self.source.unwrap_or(SourceKind::Hlc3).as_str().to_string()),
                length: self.length,
                atr_length: self.atr_length,
                use_atr: self.use_atr,
                tp_aggressiveness: Some(
                    self.tp_aggressiveness
                        .unwrap_or(TpAggressiveness::Low)
                        .as_str()
                        .to_string(),
                ),
            },
        );
        fibonacci_entry_bands_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn into_stream(self) -> Result<FibonacciEntryBandsStream, FibonacciEntryBandsError> {
        FibonacciEntryBandsStream::try_new(FibonacciEntryBandsParams {
            source: Some(self.source.unwrap_or(SourceKind::Hlc3).as_str().to_string()),
            length: self.length,
            atr_length: self.atr_length,
            use_atr: self.use_atr,
            tp_aggressiveness: Some(
                self.tp_aggressiveness
                    .unwrap_or(TpAggressiveness::Low)
                    .as_str()
                    .to_string(),
            ),
        })
    }
}

impl FibonacciEntryBandsStream {
    #[inline]
    pub fn try_new(params: FibonacciEntryBandsParams) -> Result<Self, FibonacciEntryBandsError> {
        let resolved = resolve_params(&params, None)?;
        Ok(Self::new_resolved(resolved))
    }

    #[inline]
    fn new_resolved(params: ResolvedParams) -> Self {
        Self {
            params,
            ema1: None,
            ema2: None,
            prev_basis: None,
            prev_prev_basis: None,
            trend: 0.0,
            prev_close: None,
            atr_sum: 0.0,
            atr_count: 0,
            atr_value: None,
            stdev: RollingStdev::new(params.length),
            prev_tp_long_band: None,
            prev_tp_short_band: None,
        }
    }

    #[inline]
    pub fn reset(&mut self) {
        self.stdev.reset();
        *self = Self::new_resolved(self.params);
    }

    #[inline]
    pub fn get_warmup_period(&self) -> usize {
        let vol_warmup = if self.params.use_atr {
            self.params.atr_length.saturating_sub(1)
        } else {
            self.params.length.saturating_sub(1)
        };
        vol_warmup.max(2)
    }

    #[inline]
    pub fn update(
        &mut self,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
    ) -> Option<FibonacciEntryBandsPoint> {
        if !valid_bar(self.params.source, open, high, low, close) {
            self.reset();
            return None;
        }

        Some(self.update_valid(open, high, low, close))
    }

    #[inline]
    fn update_valid(
        &mut self,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
    ) -> FibonacciEntryBandsPoint {
        let source = self.params.source.value(open, high, low, close);
        let ema1 = match self.ema1 {
            Some(prev) => prev + self.params.ema_alpha * (source - prev),
            None => source,
        };
        self.ema1 = Some(ema1);
        let basis = match self.ema2 {
            Some(prev) => prev + self.params.ema_alpha * (ema1 - prev),
            None => ema1,
        };
        self.ema2 = Some(basis);

        let long_entry = if let (Some(prev_basis), Some(prev_prev_basis)) =
            (self.prev_basis, self.prev_prev_basis)
        {
            let curr_delta = basis - prev_basis;
            let prev_delta = prev_basis - prev_prev_basis;
            if curr_delta > FLOAT_TOL && prev_delta <= FLOAT_TOL {
                1.0
            } else {
                0.0
            }
        } else {
            f64::NAN
        };

        let short_entry = if let (Some(prev_basis), Some(prev_prev_basis)) =
            (self.prev_basis, self.prev_prev_basis)
        {
            let curr_delta = basis - prev_basis;
            let prev_delta = prev_basis - prev_prev_basis;
            if curr_delta < -FLOAT_TOL && prev_delta >= -FLOAT_TOL {
                1.0
            } else {
                0.0
            }
        } else {
            f64::NAN
        };

        if let Some(prev_basis) = self.prev_basis {
            if basis > prev_basis + FLOAT_TOL {
                self.trend = 1.0;
            } else if basis < prev_basis - FLOAT_TOL {
                self.trend = -1.0;
            }
        }

        let vol = if self.params.use_atr {
            self.update_atr(high, low, close)
        } else {
            self.stdev.update(source)
        };

        let mut point = FibonacciEntryBandsPoint {
            basis,
            trend: self.trend,
            upper_0618: f64::NAN,
            upper_1000: f64::NAN,
            upper_1618: f64::NAN,
            upper_2618: f64::NAN,
            lower_0618: f64::NAN,
            lower_1000: f64::NAN,
            lower_1618: f64::NAN,
            lower_2618: f64::NAN,
            tp_long_band: f64::NAN,
            tp_short_band: f64::NAN,
            long_entry,
            short_entry,
            rejection_long: f64::NAN,
            rejection_short: f64::NAN,
            long_bounce: if self.trend > 0.0 {
                if low < basis - FLOAT_TOL && close > basis + FLOAT_TOL && !matches_true(long_entry)
                {
                    1.0
                } else {
                    0.0
                }
            } else {
                0.0
            },
            short_bounce: if self.trend < 0.0 {
                if high > basis + FLOAT_TOL
                    && close < basis - FLOAT_TOL
                    && !matches_true(short_entry)
                {
                    1.0
                } else {
                    0.0
                }
            } else {
                0.0
            },
        };

        if let Some(volatility) = vol.filter(|v| v.is_finite()) {
            point.upper_0618 = basis + volatility * MULT1;
            point.upper_1000 = basis + volatility * MULT2;
            point.upper_1618 = basis + volatility * MULT3;
            point.upper_2618 = basis + volatility * MULT4;
            point.lower_0618 = basis - volatility * MULT1;
            point.lower_1000 = basis - volatility * MULT2;
            point.lower_1618 = basis - volatility * MULT3;
            point.lower_2618 = basis - volatility * MULT4;

            let (tp_long_band, tp_short_band) = match self.params.tp_aggressiveness {
                TpAggressiveness::Low => (point.lower_2618, point.upper_2618),
                TpAggressiveness::Medium => (point.lower_1000, point.upper_1000),
                TpAggressiveness::High => (point.lower_0618, point.upper_0618),
            };
            point.tp_long_band = tp_long_band;
            point.tp_short_band = tp_short_band;

            point.rejection_long = if self.trend < 0.0
                && crossunder(close, tp_long_band, self.prev_close, self.prev_tp_long_band)
            {
                1.0
            } else {
                0.0
            };

            point.rejection_short = if self.trend > 0.0
                && crossover(
                    close,
                    tp_short_band,
                    self.prev_close,
                    self.prev_tp_short_band,
                ) {
                1.0
            } else {
                0.0
            };
        }

        self.prev_prev_basis = self.prev_basis;
        self.prev_basis = Some(basis);
        self.prev_close = Some(close);
        self.prev_tp_long_band = finite_option(point.tp_long_band);
        self.prev_tp_short_band = finite_option(point.tp_short_band);

        point
    }

    #[inline]
    fn update_atr(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        let tr = match self.prev_close {
            Some(prev_close) => {
                let hl = high - low;
                let hc = (high - prev_close).abs();
                let lc = (low - prev_close).abs();
                hl.max(hc).max(lc)
            }
            None => high - low,
        };
        if self.atr_count < self.params.atr_length {
            self.atr_count += 1;
            self.atr_sum += tr;
            if self.atr_count == self.params.atr_length {
                self.atr_value = Some(self.atr_sum / self.params.atr_length as f64);
            }
        } else if let Some(prev) = self.atr_value {
            self.atr_value = Some(
                ((self.params.atr_length - 1) as f64 * prev + tr) / self.params.atr_length as f64,
            );
        }
        self.atr_value
    }
}

#[inline(always)]
fn finite_option(value: f64) -> Option<f64> {
    if value.is_finite() {
        Some(value)
    } else {
        None
    }
}

#[inline(always)]
fn matches_true(value: f64) -> bool {
    value.is_finite() && value > 0.5
}

#[inline(always)]
fn crossover(current_a: f64, current_b: f64, prev_a: Option<f64>, prev_b: Option<f64>) -> bool {
    current_a.is_finite()
        && current_b.is_finite()
        && current_a > current_b + FLOAT_TOL
        && prev_a
            .zip(prev_b)
            .map_or(false, |(a, b)| a <= b + FLOAT_TOL)
}

#[inline(always)]
fn crossunder(current_a: f64, current_b: f64, prev_a: Option<f64>, prev_b: Option<f64>) -> bool {
    current_a.is_finite()
        && current_b.is_finite()
        && current_a < current_b - FLOAT_TOL
        && prev_a
            .zip(prev_b)
            .map_or(false, |(a, b)| a >= b - FLOAT_TOL)
}

#[inline(always)]
fn parse_source(value: &str) -> Result<SourceKind, FibonacciEntryBandsError> {
    SourceKind::parse(value).ok_or_else(|| FibonacciEntryBandsError::InvalidSource {
        source_name: value.to_string(),
    })
}

#[inline(always)]
fn parse_tp_aggressiveness(value: &str) -> Result<TpAggressiveness, FibonacciEntryBandsError> {
    TpAggressiveness::parse(value).ok_or_else(|| {
        FibonacciEntryBandsError::InvalidTpAggressiveness {
            tp_aggressiveness: value.to_string(),
        }
    })
}

#[inline(always)]
fn valid_bar(source: SourceKind, open: f64, high: f64, low: f64, close: f64) -> bool {
    high.is_finite()
        && low.is_finite()
        && close.is_finite()
        && (!source.needs_open() || open.is_finite())
}

#[inline(always)]
fn first_valid_ohlc(
    source: SourceKind,
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> usize {
    let len = close.len();
    let mut i = 0usize;
    while i < len {
        if valid_bar(source, open[i], high[i], low[i], close[i]) {
            return i;
        }
        i += 1;
    }
    len
}

#[inline(always)]
fn max_consecutive_valid_ohlc(
    source: SourceKind,
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> usize {
    let len = close.len();
    let mut best = 0usize;
    let mut run = 0usize;
    let mut i = 0usize;
    while i < len {
        if valid_bar(source, open[i], high[i], low[i], close[i]) {
            run += 1;
            if run > best {
                best = run;
            }
        } else {
            run = 0;
        }
        i += 1;
    }
    best
}

#[inline]
fn resolve_params(
    params: &FibonacciEntryBandsParams,
    data_len: Option<usize>,
) -> Result<ResolvedParams, FibonacciEntryBandsError> {
    let source = parse_source(params.source.as_deref().unwrap_or(DEFAULT_SOURCE))?;
    let length = params.length.unwrap_or(DEFAULT_LENGTH);
    let atr_length = params.atr_length.unwrap_or(DEFAULT_ATR_LENGTH);
    let use_atr = params.use_atr.unwrap_or(DEFAULT_USE_ATR);
    let tp_aggressiveness = parse_tp_aggressiveness(
        params
            .tp_aggressiveness
            .as_deref()
            .unwrap_or(DEFAULT_TP_AGGRESSIVENESS),
    )?;
    let data_len = data_len.unwrap_or(0);

    if length == 0 || (data_len != 0 && length > data_len) {
        return Err(FibonacciEntryBandsError::InvalidLength { length, data_len });
    }
    if atr_length == 0 || (data_len != 0 && atr_length > data_len) {
        return Err(FibonacciEntryBandsError::InvalidAtrLength {
            atr_length,
            data_len,
        });
    }

    Ok(ResolvedParams {
        source,
        length,
        atr_length,
        use_atr,
        tp_aggressiveness,
        ema_alpha: 2.0 / (length as f64 + 1.0),
    })
}

#[inline(always)]
fn output_vec(len: usize) -> Vec<f64> {
    alloc_uninit_f64(len)
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn write_point(
    point: FibonacciEntryBandsPoint,
    idx: usize,
    basis: &mut [f64],
    trend: &mut [f64],
    upper_0618: &mut [f64],
    upper_1000: &mut [f64],
    upper_1618: &mut [f64],
    upper_2618: &mut [f64],
    lower_0618: &mut [f64],
    lower_1000: &mut [f64],
    lower_1618: &mut [f64],
    lower_2618: &mut [f64],
    tp_long_band: &mut [f64],
    tp_short_band: &mut [f64],
    long_entry: &mut [f64],
    short_entry: &mut [f64],
    rejection_long: &mut [f64],
    rejection_short: &mut [f64],
    long_bounce: &mut [f64],
    short_bounce: &mut [f64],
) {
    basis[idx] = point.basis;
    trend[idx] = point.trend;
    upper_0618[idx] = point.upper_0618;
    upper_1000[idx] = point.upper_1000;
    upper_1618[idx] = point.upper_1618;
    upper_2618[idx] = point.upper_2618;
    lower_0618[idx] = point.lower_0618;
    lower_1000[idx] = point.lower_1000;
    lower_1618[idx] = point.lower_1618;
    lower_2618[idx] = point.lower_2618;
    tp_long_band[idx] = point.tp_long_band;
    tp_short_band[idx] = point.tp_short_band;
    long_entry[idx] = point.long_entry;
    short_entry[idx] = point.short_entry;
    rejection_long[idx] = point.rejection_long;
    rejection_short[idx] = point.rejection_short;
    long_bounce[idx] = point.long_bounce;
    short_bounce[idx] = point.short_bounce;
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn fibonacci_entry_bands_row_from_slices(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    params: ResolvedParams,
    basis: &mut [f64],
    trend: &mut [f64],
    upper_0618: &mut [f64],
    upper_1000: &mut [f64],
    upper_1618: &mut [f64],
    upper_2618: &mut [f64],
    lower_0618: &mut [f64],
    lower_1000: &mut [f64],
    lower_1618: &mut [f64],
    lower_2618: &mut [f64],
    tp_long_band: &mut [f64],
    tp_short_band: &mut [f64],
    long_entry: &mut [f64],
    short_entry: &mut [f64],
    rejection_long: &mut [f64],
    rejection_short: &mut [f64],
    long_bounce: &mut [f64],
    short_bounce: &mut [f64],
) {
    let mut stream = FibonacciEntryBandsStream::new_resolved(params);
    for i in 0..close.len() {
        let point = stream
            .update(open[i], high[i], low[i], close[i])
            .unwrap_or_else(FibonacciEntryBandsPoint::nan);
        write_point(
            point,
            i,
            basis,
            trend,
            upper_0618,
            upper_1000,
            upper_1618,
            upper_2618,
            lower_0618,
            lower_1000,
            lower_1618,
            lower_2618,
            tp_long_band,
            tp_short_band,
            long_entry,
            short_entry,
            rejection_long,
            rejection_short,
            long_bounce,
            short_bounce,
        );
    }
}

#[inline]
fn fibonacci_entry_bands_prepare<'a>(
    input: &'a FibonacciEntryBandsInput,
    _kernel: Kernel,
) -> Result<(&'a [f64], &'a [f64], &'a [f64], &'a [f64], ResolvedParams), FibonacciEntryBandsError>
{
    let (open, high, low, close) = input.as_slices();
    if open.is_empty() || high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(FibonacciEntryBandsError::EmptyInputData);
    }
    if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
        return Err(FibonacciEntryBandsError::InconsistentSliceLengths {
            open_len: open.len(),
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }

    let params = resolve_params(&input.params, Some(close.len()))?;
    let first = first_valid_ohlc(params.source, open, high, low, close);
    if first >= close.len() {
        return Err(FibonacciEntryBandsError::AllValuesNaN);
    }

    let valid = max_consecutive_valid_ohlc(params.source, open, high, low, close);
    let needed = if params.use_atr {
        params.atr_length
    } else {
        params.length
    };
    if valid < needed {
        return Err(FibonacciEntryBandsError::NotEnoughValidData { needed, valid });
    }

    Ok((open, high, low, close, params))
}

#[inline]
pub fn fibonacci_entry_bands(
    input: &FibonacciEntryBandsInput,
) -> Result<FibonacciEntryBandsOutput, FibonacciEntryBandsError> {
    fibonacci_entry_bands_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn fibonacci_entry_bands_with_kernel(
    input: &FibonacciEntryBandsInput,
    kernel: Kernel,
) -> Result<FibonacciEntryBandsOutput, FibonacciEntryBandsError> {
    let (open, high, low, close, params) = fibonacci_entry_bands_prepare(input, kernel)?;
    let len = close.len();
    let mut out = FibonacciEntryBandsOutput {
        basis: output_vec(len),
        trend: output_vec(len),
        upper_0618: output_vec(len),
        upper_1000: output_vec(len),
        upper_1618: output_vec(len),
        upper_2618: output_vec(len),
        lower_0618: output_vec(len),
        lower_1000: output_vec(len),
        lower_1618: output_vec(len),
        lower_2618: output_vec(len),
        tp_long_band: output_vec(len),
        tp_short_band: output_vec(len),
        long_entry: output_vec(len),
        short_entry: output_vec(len),
        rejection_long: output_vec(len),
        rejection_short: output_vec(len),
        long_bounce: output_vec(len),
        short_bounce: output_vec(len),
    };
    fibonacci_entry_bands_row_from_slices(
        open,
        high,
        low,
        close,
        params,
        &mut out.basis,
        &mut out.trend,
        &mut out.upper_0618,
        &mut out.upper_1000,
        &mut out.upper_1618,
        &mut out.upper_2618,
        &mut out.lower_0618,
        &mut out.lower_1000,
        &mut out.lower_1618,
        &mut out.lower_2618,
        &mut out.tp_long_band,
        &mut out.tp_short_band,
        &mut out.long_entry,
        &mut out.short_entry,
        &mut out.rejection_long,
        &mut out.rejection_short,
        &mut out.long_bounce,
        &mut out.short_bounce,
    );
    Ok(out)
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub fn fibonacci_entry_bands_into_slices(
    basis: &mut [f64],
    trend: &mut [f64],
    upper_0618: &mut [f64],
    upper_1000: &mut [f64],
    upper_1618: &mut [f64],
    upper_2618: &mut [f64],
    lower_0618: &mut [f64],
    lower_1000: &mut [f64],
    lower_1618: &mut [f64],
    lower_2618: &mut [f64],
    tp_long_band: &mut [f64],
    tp_short_band: &mut [f64],
    long_entry: &mut [f64],
    short_entry: &mut [f64],
    rejection_long: &mut [f64],
    rejection_short: &mut [f64],
    long_bounce: &mut [f64],
    short_bounce: &mut [f64],
    input: &FibonacciEntryBandsInput,
    kernel: Kernel,
) -> Result<(), FibonacciEntryBandsError> {
    let expected = input.as_slices().3.len();
    if basis.len() != expected
        || trend.len() != expected
        || upper_0618.len() != expected
        || upper_1000.len() != expected
        || upper_1618.len() != expected
        || upper_2618.len() != expected
        || lower_0618.len() != expected
        || lower_1000.len() != expected
        || lower_1618.len() != expected
        || lower_2618.len() != expected
        || tp_long_band.len() != expected
        || tp_short_band.len() != expected
        || long_entry.len() != expected
        || short_entry.len() != expected
        || rejection_long.len() != expected
        || rejection_short.len() != expected
        || long_bounce.len() != expected
        || short_bounce.len() != expected
    {
        return Err(FibonacciEntryBandsError::OutputLengthMismatch { expected });
    }

    let (open, high, low, close, params) = fibonacci_entry_bands_prepare(input, kernel)?;
    fibonacci_entry_bands_row_from_slices(
        open,
        high,
        low,
        close,
        params,
        basis,
        trend,
        upper_0618,
        upper_1000,
        upper_1618,
        upper_2618,
        lower_0618,
        lower_1000,
        lower_1618,
        lower_2618,
        tp_long_band,
        tp_short_band,
        long_entry,
        short_entry,
        rejection_long,
        rejection_short,
        long_bounce,
        short_bounce,
    );
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn fibonacci_entry_bands_into(
    input: &FibonacciEntryBandsInput,
    basis: &mut [f64],
    trend: &mut [f64],
    upper_0618: &mut [f64],
    upper_1000: &mut [f64],
    upper_1618: &mut [f64],
    upper_2618: &mut [f64],
    lower_0618: &mut [f64],
    lower_1000: &mut [f64],
    lower_1618: &mut [f64],
    lower_2618: &mut [f64],
    tp_long_band: &mut [f64],
    tp_short_band: &mut [f64],
    long_entry: &mut [f64],
    short_entry: &mut [f64],
    rejection_long: &mut [f64],
    rejection_short: &mut [f64],
    long_bounce: &mut [f64],
    short_bounce: &mut [f64],
) -> Result<(), FibonacciEntryBandsError> {
    fibonacci_entry_bands_into_slices(
        basis,
        trend,
        upper_0618,
        upper_1000,
        upper_1618,
        upper_2618,
        lower_0618,
        lower_1000,
        lower_1618,
        lower_2618,
        tp_long_band,
        tp_short_band,
        long_entry,
        short_entry,
        rejection_long,
        rejection_short,
        long_bounce,
        short_bounce,
        input,
        Kernel::Auto,
    )
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct FibonacciEntryBandsBatchRange {
    pub length: (usize, usize, usize),
    pub atr_length: (usize, usize, usize),
    pub source: String,
    pub use_atr: bool,
    pub tp_aggressiveness: String,
}

impl Default for FibonacciEntryBandsBatchRange {
    fn default() -> Self {
        Self {
            length: (DEFAULT_LENGTH, DEFAULT_LENGTH, 0),
            atr_length: (DEFAULT_ATR_LENGTH, DEFAULT_ATR_LENGTH, 0),
            source: DEFAULT_SOURCE.to_string(),
            use_atr: DEFAULT_USE_ATR,
            tp_aggressiveness: DEFAULT_TP_AGGRESSIVENESS.to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FibonacciEntryBandsBatchOutput {
    pub basis: Vec<f64>,
    pub trend: Vec<f64>,
    pub upper_0618: Vec<f64>,
    pub upper_1000: Vec<f64>,
    pub upper_1618: Vec<f64>,
    pub upper_2618: Vec<f64>,
    pub lower_0618: Vec<f64>,
    pub lower_1000: Vec<f64>,
    pub lower_1618: Vec<f64>,
    pub lower_2618: Vec<f64>,
    pub tp_long_band: Vec<f64>,
    pub tp_short_band: Vec<f64>,
    pub long_entry: Vec<f64>,
    pub short_entry: Vec<f64>,
    pub rejection_long: Vec<f64>,
    pub rejection_short: Vec<f64>,
    pub long_bounce: Vec<f64>,
    pub short_bounce: Vec<f64>,
    pub combos: Vec<FibonacciEntryBandsParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Debug, Default)]
pub struct FibonacciEntryBandsBatchBuilder {
    range: FibonacciEntryBandsBatchRange,
    kernel: Kernel,
}

impl FibonacciEntryBandsBatchBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline]
    pub fn length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.length = (start, end, step);
        self
    }

    #[inline]
    pub fn atr_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.atr_length = (start, end, step);
        self
    }

    #[inline]
    pub fn source(mut self, value: &str) -> Self {
        self.range.source = value.to_string();
        self
    }

    #[inline]
    pub fn use_atr(mut self, value: bool) -> Self {
        self.range.use_atr = value;
        self
    }

    #[inline]
    pub fn tp_aggressiveness(mut self, value: &str) -> Self {
        self.range.tp_aggressiveness = value.to_string();
        self
    }

    #[inline]
    pub fn apply_slices(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<FibonacciEntryBandsBatchOutput, FibonacciEntryBandsError> {
        fibonacci_entry_bands_batch_with_kernel(open, high, low, close, &self.range, self.kernel)
    }

    #[inline]
    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<FibonacciEntryBandsBatchOutput, FibonacciEntryBandsError> {
        fibonacci_entry_bands_batch_with_kernel(
            &candles.open,
            &candles.high,
            &candles.low,
            &candles.close,
            &self.range,
            self.kernel,
        )
    }
}

#[inline]
fn expand_usize_range(
    name: &str,
    range: (usize, usize, usize),
) -> Result<Vec<usize>, FibonacciEntryBandsError> {
    let (start, end, step) = range;
    if start > end {
        return Err(FibonacciEntryBandsError::InvalidRange {
            start: format!("{name}={start}"),
            end: format!("{name}={end}"),
            step: format!("{name}={step}"),
        });
    }
    if start == end {
        return Ok(vec![start]);
    }
    if step == 0 {
        return Err(FibonacciEntryBandsError::InvalidRange {
            start: format!("{name}={start}"),
            end: format!("{name}={end}"),
            step: format!("{name}={step}"),
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

#[inline]
fn expand_grid(
    sweep: &FibonacciEntryBandsBatchRange,
) -> Result<Vec<FibonacciEntryBandsParams>, FibonacciEntryBandsError> {
    let _ = parse_source(&sweep.source)?;
    let _ = parse_tp_aggressiveness(&sweep.tp_aggressiveness)?;
    let lengths = expand_usize_range("length", sweep.length)?;
    let atr_lengths = expand_usize_range("atr_length", sweep.atr_length)?;
    let mut combos = Vec::with_capacity(lengths.len().saturating_mul(atr_lengths.len()));
    for &length in &lengths {
        for &atr_length in &atr_lengths {
            combos.push(FibonacciEntryBandsParams {
                source: Some(sweep.source.clone()),
                length: Some(length),
                atr_length: Some(atr_length),
                use_atr: Some(sweep.use_atr),
                tp_aggressiveness: Some(sweep.tp_aggressiveness.clone()),
            });
        }
    }
    Ok(combos)
}

#[inline(always)]
unsafe fn assume_init_vec(
    mut guard: ManuallyDrop<Vec<MaybeUninit<f64>>>,
    total: usize,
) -> Vec<f64> {
    Vec::from_raw_parts(guard.as_mut_ptr() as *mut f64, total, guard.capacity())
}

#[inline]
fn validate_batch_kernel(kernel: Kernel) -> Result<Kernel, FibonacciEntryBandsError> {
    let chosen = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other => other,
    };
    match chosen {
        Kernel::Auto
        | Kernel::ScalarBatch
        | Kernel::Avx2Batch
        | Kernel::Avx512Batch
        | Kernel::Scalar
        | Kernel::Avx2
        | Kernel::Avx512 => Ok(chosen.to_non_batch()),
        other => Err(FibonacciEntryBandsError::InvalidKernelForBatch(other)),
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub fn fibonacci_entry_bands_batch_with_kernel(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &FibonacciEntryBandsBatchRange,
    kernel: Kernel,
) -> Result<FibonacciEntryBandsBatchOutput, FibonacciEntryBandsError> {
    let resolved = expand_grid(sweep)?;
    if open.is_empty() || high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(FibonacciEntryBandsError::EmptyInputData);
    }
    if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
        return Err(FibonacciEntryBandsError::InconsistentSliceLengths {
            open_len: open.len(),
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }
    let rows = resolved.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or(FibonacciEntryBandsError::OutputLengthMismatch {
            expected: usize::MAX,
        })?;
    let _kernel = validate_batch_kernel(kernel)?;
    let resolved_params = resolved
        .iter()
        .map(|combo| resolve_params(combo, Some(cols)))
        .collect::<Result<Vec<_>, _>>()?;
    let zero_prefixes = vec![0usize; rows];

    macro_rules! alloc_matrix {
        ($name:ident, $guard:ident, $out:ident) => {
            let mut $name = make_uninit_matrix(rows, cols);
            init_matrix_prefixes(&mut $name, cols, &zero_prefixes);
            let mut $guard = ManuallyDrop::new($name);
            let $out =
                unsafe { std::slice::from_raw_parts_mut($guard.as_mut_ptr() as *mut f64, total) };
        };
    }

    alloc_matrix!(basis_mu, basis_guard, basis_out);
    alloc_matrix!(trend_mu, trend_guard, trend_out);
    alloc_matrix!(upper_0618_mu, upper_0618_guard, upper_0618_out);
    alloc_matrix!(upper_1000_mu, upper_1000_guard, upper_1000_out);
    alloc_matrix!(upper_1618_mu, upper_1618_guard, upper_1618_out);
    alloc_matrix!(upper_2618_mu, upper_2618_guard, upper_2618_out);
    alloc_matrix!(lower_0618_mu, lower_0618_guard, lower_0618_out);
    alloc_matrix!(lower_1000_mu, lower_1000_guard, lower_1000_out);
    alloc_matrix!(lower_1618_mu, lower_1618_guard, lower_1618_out);
    alloc_matrix!(lower_2618_mu, lower_2618_guard, lower_2618_out);
    alloc_matrix!(tp_long_mu, tp_long_guard, tp_long_out);
    alloc_matrix!(tp_short_mu, tp_short_guard, tp_short_out);
    alloc_matrix!(long_entry_mu, long_entry_guard, long_entry_out);
    alloc_matrix!(short_entry_mu, short_entry_guard, short_entry_out);
    alloc_matrix!(rejection_long_mu, rejection_long_guard, rejection_long_out);
    alloc_matrix!(
        rejection_short_mu,
        rejection_short_guard,
        rejection_short_out
    );
    alloc_matrix!(long_bounce_mu, long_bounce_guard, long_bounce_out);
    alloc_matrix!(short_bounce_mu, short_bounce_guard, short_bounce_out);

    #[cfg(not(target_arch = "wasm32"))]
    {
        let basis_ptr = basis_out.as_mut_ptr() as usize;
        let trend_ptr = trend_out.as_mut_ptr() as usize;
        let upper_0618_ptr = upper_0618_out.as_mut_ptr() as usize;
        let upper_1000_ptr = upper_1000_out.as_mut_ptr() as usize;
        let upper_1618_ptr = upper_1618_out.as_mut_ptr() as usize;
        let upper_2618_ptr = upper_2618_out.as_mut_ptr() as usize;
        let lower_0618_ptr = lower_0618_out.as_mut_ptr() as usize;
        let lower_1000_ptr = lower_1000_out.as_mut_ptr() as usize;
        let lower_1618_ptr = lower_1618_out.as_mut_ptr() as usize;
        let lower_2618_ptr = lower_2618_out.as_mut_ptr() as usize;
        let tp_long_ptr = tp_long_out.as_mut_ptr() as usize;
        let tp_short_ptr = tp_short_out.as_mut_ptr() as usize;
        let long_entry_ptr = long_entry_out.as_mut_ptr() as usize;
        let short_entry_ptr = short_entry_out.as_mut_ptr() as usize;
        let rejection_long_ptr = rejection_long_out.as_mut_ptr() as usize;
        let rejection_short_ptr = rejection_short_out.as_mut_ptr() as usize;
        let long_bounce_ptr = long_bounce_out.as_mut_ptr() as usize;
        let short_bounce_ptr = short_bounce_out.as_mut_ptr() as usize;

        resolved
            .par_iter()
            .zip(resolved_params.par_iter())
            .enumerate()
            .for_each(|(row, (_combo, params))| unsafe {
                let start = row * cols;
                fibonacci_entry_bands_row_from_slices(
                    open,
                    high,
                    low,
                    close,
                    *params,
                    &mut std::slice::from_raw_parts_mut(basis_ptr as *mut f64, total)
                        [start..start + cols],
                    &mut std::slice::from_raw_parts_mut(trend_ptr as *mut f64, total)
                        [start..start + cols],
                    &mut std::slice::from_raw_parts_mut(upper_0618_ptr as *mut f64, total)
                        [start..start + cols],
                    &mut std::slice::from_raw_parts_mut(upper_1000_ptr as *mut f64, total)
                        [start..start + cols],
                    &mut std::slice::from_raw_parts_mut(upper_1618_ptr as *mut f64, total)
                        [start..start + cols],
                    &mut std::slice::from_raw_parts_mut(upper_2618_ptr as *mut f64, total)
                        [start..start + cols],
                    &mut std::slice::from_raw_parts_mut(lower_0618_ptr as *mut f64, total)
                        [start..start + cols],
                    &mut std::slice::from_raw_parts_mut(lower_1000_ptr as *mut f64, total)
                        [start..start + cols],
                    &mut std::slice::from_raw_parts_mut(lower_1618_ptr as *mut f64, total)
                        [start..start + cols],
                    &mut std::slice::from_raw_parts_mut(lower_2618_ptr as *mut f64, total)
                        [start..start + cols],
                    &mut std::slice::from_raw_parts_mut(tp_long_ptr as *mut f64, total)
                        [start..start + cols],
                    &mut std::slice::from_raw_parts_mut(tp_short_ptr as *mut f64, total)
                        [start..start + cols],
                    &mut std::slice::from_raw_parts_mut(long_entry_ptr as *mut f64, total)
                        [start..start + cols],
                    &mut std::slice::from_raw_parts_mut(short_entry_ptr as *mut f64, total)
                        [start..start + cols],
                    &mut std::slice::from_raw_parts_mut(rejection_long_ptr as *mut f64, total)
                        [start..start + cols],
                    &mut std::slice::from_raw_parts_mut(rejection_short_ptr as *mut f64, total)
                        [start..start + cols],
                    &mut std::slice::from_raw_parts_mut(long_bounce_ptr as *mut f64, total)
                        [start..start + cols],
                    &mut std::slice::from_raw_parts_mut(short_bounce_ptr as *mut f64, total)
                        [start..start + cols],
                );
            });
    }

    #[cfg(target_arch = "wasm32")]
    {
        for (row, params) in resolved_params.iter().enumerate() {
            let start = row * cols;
            fibonacci_entry_bands_row_from_slices(
                open,
                high,
                low,
                close,
                *params,
                &mut basis_out[start..start + cols],
                &mut trend_out[start..start + cols],
                &mut upper_0618_out[start..start + cols],
                &mut upper_1000_out[start..start + cols],
                &mut upper_1618_out[start..start + cols],
                &mut upper_2618_out[start..start + cols],
                &mut lower_0618_out[start..start + cols],
                &mut lower_1000_out[start..start + cols],
                &mut lower_1618_out[start..start + cols],
                &mut lower_2618_out[start..start + cols],
                &mut tp_long_out[start..start + cols],
                &mut tp_short_out[start..start + cols],
                &mut long_entry_out[start..start + cols],
                &mut short_entry_out[start..start + cols],
                &mut rejection_long_out[start..start + cols],
                &mut rejection_short_out[start..start + cols],
                &mut long_bounce_out[start..start + cols],
                &mut short_bounce_out[start..start + cols],
            );
        }
    }

    Ok(FibonacciEntryBandsBatchOutput {
        basis: unsafe { assume_init_vec(basis_guard, total) },
        trend: unsafe { assume_init_vec(trend_guard, total) },
        upper_0618: unsafe { assume_init_vec(upper_0618_guard, total) },
        upper_1000: unsafe { assume_init_vec(upper_1000_guard, total) },
        upper_1618: unsafe { assume_init_vec(upper_1618_guard, total) },
        upper_2618: unsafe { assume_init_vec(upper_2618_guard, total) },
        lower_0618: unsafe { assume_init_vec(lower_0618_guard, total) },
        lower_1000: unsafe { assume_init_vec(lower_1000_guard, total) },
        lower_1618: unsafe { assume_init_vec(lower_1618_guard, total) },
        lower_2618: unsafe { assume_init_vec(lower_2618_guard, total) },
        tp_long_band: unsafe { assume_init_vec(tp_long_guard, total) },
        tp_short_band: unsafe { assume_init_vec(tp_short_guard, total) },
        long_entry: unsafe { assume_init_vec(long_entry_guard, total) },
        short_entry: unsafe { assume_init_vec(short_entry_guard, total) },
        rejection_long: unsafe { assume_init_vec(rejection_long_guard, total) },
        rejection_short: unsafe { assume_init_vec(rejection_short_guard, total) },
        long_bounce: unsafe { assume_init_vec(long_bounce_guard, total) },
        short_bounce: unsafe { assume_init_vec(short_bounce_guard, total) },
        combos: resolved,
        rows,
        cols,
    })
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub fn fibonacci_entry_bands_batch_inner_into(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &FibonacciEntryBandsBatchRange,
    kernel: Kernel,
    basis: &mut [f64],
    trend: &mut [f64],
    upper_0618: &mut [f64],
    upper_1000: &mut [f64],
    upper_1618: &mut [f64],
    upper_2618: &mut [f64],
    lower_0618: &mut [f64],
    lower_1000: &mut [f64],
    lower_1618: &mut [f64],
    lower_2618: &mut [f64],
    tp_long_band: &mut [f64],
    tp_short_band: &mut [f64],
    long_entry: &mut [f64],
    short_entry: &mut [f64],
    rejection_long: &mut [f64],
    rejection_short: &mut [f64],
    long_bounce: &mut [f64],
    short_bounce: &mut [f64],
) -> Result<Vec<FibonacciEntryBandsParams>, FibonacciEntryBandsError> {
    let out = fibonacci_entry_bands_batch_with_kernel(open, high, low, close, sweep, kernel)?;
    let total = out.rows * out.cols;
    if basis.len() != total
        || trend.len() != total
        || upper_0618.len() != total
        || upper_1000.len() != total
        || upper_1618.len() != total
        || upper_2618.len() != total
        || lower_0618.len() != total
        || lower_1000.len() != total
        || lower_1618.len() != total
        || lower_2618.len() != total
        || tp_long_band.len() != total
        || tp_short_band.len() != total
        || long_entry.len() != total
        || short_entry.len() != total
        || rejection_long.len() != total
        || rejection_short.len() != total
        || long_bounce.len() != total
        || short_bounce.len() != total
    {
        return Err(FibonacciEntryBandsError::OutputLengthMismatch { expected: total });
    }

    basis.copy_from_slice(&out.basis);
    trend.copy_from_slice(&out.trend);
    upper_0618.copy_from_slice(&out.upper_0618);
    upper_1000.copy_from_slice(&out.upper_1000);
    upper_1618.copy_from_slice(&out.upper_1618);
    upper_2618.copy_from_slice(&out.upper_2618);
    lower_0618.copy_from_slice(&out.lower_0618);
    lower_1000.copy_from_slice(&out.lower_1000);
    lower_1618.copy_from_slice(&out.lower_1618);
    lower_2618.copy_from_slice(&out.lower_2618);
    tp_long_band.copy_from_slice(&out.tp_long_band);
    tp_short_band.copy_from_slice(&out.tp_short_band);
    long_entry.copy_from_slice(&out.long_entry);
    short_entry.copy_from_slice(&out.short_entry);
    rejection_long.copy_from_slice(&out.rejection_long);
    rejection_short.copy_from_slice(&out.rejection_short);
    long_bounce.copy_from_slice(&out.long_bounce);
    short_bounce.copy_from_slice(&out.short_bounce);
    Ok(out.combos)
}

#[cfg(feature = "python")]
fn output_to_py_dict<'py>(
    py: Python<'py>,
    output: FibonacciEntryBandsOutput,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("basis", output.basis.into_pyarray(py))?;
    dict.set_item("trend", output.trend.into_pyarray(py))?;
    dict.set_item("upper_0618", output.upper_0618.into_pyarray(py))?;
    dict.set_item("upper_1000", output.upper_1000.into_pyarray(py))?;
    dict.set_item("upper_1618", output.upper_1618.into_pyarray(py))?;
    dict.set_item("upper_2618", output.upper_2618.into_pyarray(py))?;
    dict.set_item("lower_0618", output.lower_0618.into_pyarray(py))?;
    dict.set_item("lower_1000", output.lower_1000.into_pyarray(py))?;
    dict.set_item("lower_1618", output.lower_1618.into_pyarray(py))?;
    dict.set_item("lower_2618", output.lower_2618.into_pyarray(py))?;
    dict.set_item("tp_long_band", output.tp_long_band.into_pyarray(py))?;
    dict.set_item("tp_short_band", output.tp_short_band.into_pyarray(py))?;
    dict.set_item("long_entry", output.long_entry.into_pyarray(py))?;
    dict.set_item("short_entry", output.short_entry.into_pyarray(py))?;
    dict.set_item("rejection_long", output.rejection_long.into_pyarray(py))?;
    dict.set_item("rejection_short", output.rejection_short.into_pyarray(py))?;
    dict.set_item("long_bounce", output.long_bounce.into_pyarray(py))?;
    dict.set_item("short_bounce", output.short_bounce.into_pyarray(py))?;
    Ok(dict)
}

#[cfg(feature = "python")]
fn point_to_py_dict<'py>(
    py: Python<'py>,
    point: FibonacciEntryBandsPoint,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("basis", point.basis)?;
    dict.set_item("trend", point.trend)?;
    dict.set_item("upper_0618", point.upper_0618)?;
    dict.set_item("upper_1000", point.upper_1000)?;
    dict.set_item("upper_1618", point.upper_1618)?;
    dict.set_item("upper_2618", point.upper_2618)?;
    dict.set_item("lower_0618", point.lower_0618)?;
    dict.set_item("lower_1000", point.lower_1000)?;
    dict.set_item("lower_1618", point.lower_1618)?;
    dict.set_item("lower_2618", point.lower_2618)?;
    dict.set_item("tp_long_band", point.tp_long_band)?;
    dict.set_item("tp_short_band", point.tp_short_band)?;
    dict.set_item("long_entry", point.long_entry)?;
    dict.set_item("short_entry", point.short_entry)?;
    dict.set_item("rejection_long", point.rejection_long)?;
    dict.set_item("rejection_short", point.rejection_short)?;
    dict.set_item("long_bounce", point.long_bounce)?;
    dict.set_item("short_bounce", point.short_bounce)?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "FibonacciEntryBandsStream")]
pub struct FibonacciEntryBandsStreamPy {
    stream: FibonacciEntryBandsStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl FibonacciEntryBandsStreamPy {
    #[new]
    #[pyo3(signature = (source=DEFAULT_SOURCE, length=DEFAULT_LENGTH, atr_length=DEFAULT_ATR_LENGTH, use_atr=DEFAULT_USE_ATR, tp_aggressiveness=DEFAULT_TP_AGGRESSIVENESS))]
    fn new(
        source: &str,
        length: usize,
        atr_length: usize,
        use_atr: bool,
        tp_aggressiveness: &str,
    ) -> PyResult<Self> {
        Ok(Self {
            stream: FibonacciEntryBandsStream::try_new(FibonacciEntryBandsParams {
                source: Some(source.to_string()),
                length: Some(length),
                atr_length: Some(atr_length),
                use_atr: Some(use_atr),
                tp_aggressiveness: Some(tp_aggressiveness.to_string()),
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?,
        })
    }

    fn update<'py>(
        &mut self,
        py: Python<'py>,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        self.stream
            .update(open, high, low, close)
            .map(|point| point_to_py_dict(py, point))
            .transpose()
    }

    fn reset(&mut self) {
        self.stream.reset();
    }

    #[getter]
    fn warmup_period(&self) -> usize {
        self.stream.get_warmup_period()
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "fibonacci_entry_bands")]
#[pyo3(signature = (open, high, low, close, source=DEFAULT_SOURCE, length=DEFAULT_LENGTH, atr_length=DEFAULT_ATR_LENGTH, use_atr=DEFAULT_USE_ATR, tp_aggressiveness=DEFAULT_TP_AGGRESSIVENESS, kernel=None))]
pub fn fibonacci_entry_bands_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    source: &str,
    length: usize,
    atr_length: usize,
    use_atr: bool,
    tp_aggressiveness: &str,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let input = FibonacciEntryBandsInput::from_slices(
        open.as_slice()?,
        high.as_slice()?,
        low.as_slice()?,
        close.as_slice()?,
        FibonacciEntryBandsParams {
            source: Some(source.to_string()),
            length: Some(length),
            atr_length: Some(atr_length),
            use_atr: Some(use_atr),
            tp_aggressiveness: Some(tp_aggressiveness.to_string()),
        },
    );
    let kern = validate_kernel(kernel, false)?;
    let output = py
        .allow_threads(|| fibonacci_entry_bands_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    output_to_py_dict(py, output)
}

#[cfg(feature = "python")]
#[pyfunction(name = "fibonacci_entry_bands_batch")]
#[pyo3(signature = (open, high, low, close, length_range=(DEFAULT_LENGTH, DEFAULT_LENGTH, 0), atr_length_range=(DEFAULT_ATR_LENGTH, DEFAULT_ATR_LENGTH, 0), source=DEFAULT_SOURCE, use_atr=DEFAULT_USE_ATR, tp_aggressiveness=DEFAULT_TP_AGGRESSIVENESS, kernel=None))]
pub fn fibonacci_entry_bands_batch_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    atr_length_range: (usize, usize, usize),
    source: &str,
    use_atr: bool,
    tp_aggressiveness: &str,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let open = open.as_slice()?;
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let sweep = FibonacciEntryBandsBatchRange {
        length: length_range,
        atr_length: atr_length_range,
        source: source.to_string(),
        use_atr,
        tp_aggressiveness: tp_aggressiveness.to_string(),
    };
    let combos = expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    macro_rules! arr {
        ($name:ident) => {
            let $name = unsafe { PyArray1::<f64>::new(py, [total], false) };
        };
    }

    arr!(basis_arr);
    arr!(trend_arr);
    arr!(upper_0618_arr);
    arr!(upper_1000_arr);
    arr!(upper_1618_arr);
    arr!(upper_2618_arr);
    arr!(lower_0618_arr);
    arr!(lower_1000_arr);
    arr!(lower_1618_arr);
    arr!(lower_2618_arr);
    arr!(tp_long_arr);
    arr!(tp_short_arr);
    arr!(long_entry_arr);
    arr!(short_entry_arr);
    arr!(rejection_long_arr);
    arr!(rejection_short_arr);
    arr!(long_bounce_arr);
    arr!(short_bounce_arr);

    let basis_slice = unsafe { basis_arr.as_slice_mut()? };
    let trend_slice = unsafe { trend_arr.as_slice_mut()? };
    let upper_0618_slice = unsafe { upper_0618_arr.as_slice_mut()? };
    let upper_1000_slice = unsafe { upper_1000_arr.as_slice_mut()? };
    let upper_1618_slice = unsafe { upper_1618_arr.as_slice_mut()? };
    let upper_2618_slice = unsafe { upper_2618_arr.as_slice_mut()? };
    let lower_0618_slice = unsafe { lower_0618_arr.as_slice_mut()? };
    let lower_1000_slice = unsafe { lower_1000_arr.as_slice_mut()? };
    let lower_1618_slice = unsafe { lower_1618_arr.as_slice_mut()? };
    let lower_2618_slice = unsafe { lower_2618_arr.as_slice_mut()? };
    let tp_long_slice = unsafe { tp_long_arr.as_slice_mut()? };
    let tp_short_slice = unsafe { tp_short_arr.as_slice_mut()? };
    let long_entry_slice = unsafe { long_entry_arr.as_slice_mut()? };
    let short_entry_slice = unsafe { short_entry_arr.as_slice_mut()? };
    let rejection_long_slice = unsafe { rejection_long_arr.as_slice_mut()? };
    let rejection_short_slice = unsafe { rejection_short_arr.as_slice_mut()? };
    let long_bounce_slice = unsafe { long_bounce_arr.as_slice_mut()? };
    let short_bounce_slice = unsafe { short_bounce_arr.as_slice_mut()? };

    let combos = py
        .allow_threads(|| {
            fibonacci_entry_bands_batch_inner_into(
                open,
                high,
                low,
                close,
                &sweep,
                kern,
                basis_slice,
                trend_slice,
                upper_0618_slice,
                upper_1000_slice,
                upper_1618_slice,
                upper_2618_slice,
                lower_0618_slice,
                lower_1000_slice,
                lower_1618_slice,
                lower_2618_slice,
                tp_long_slice,
                tp_short_slice,
                long_entry_slice,
                short_entry_slice,
                rejection_long_slice,
                rejection_short_slice,
                long_bounce_slice,
                short_bounce_slice,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("basis", basis_arr.reshape((rows, cols))?)?;
    dict.set_item("trend", trend_arr.reshape((rows, cols))?)?;
    dict.set_item("upper_0618", upper_0618_arr.reshape((rows, cols))?)?;
    dict.set_item("upper_1000", upper_1000_arr.reshape((rows, cols))?)?;
    dict.set_item("upper_1618", upper_1618_arr.reshape((rows, cols))?)?;
    dict.set_item("upper_2618", upper_2618_arr.reshape((rows, cols))?)?;
    dict.set_item("lower_0618", lower_0618_arr.reshape((rows, cols))?)?;
    dict.set_item("lower_1000", lower_1000_arr.reshape((rows, cols))?)?;
    dict.set_item("lower_1618", lower_1618_arr.reshape((rows, cols))?)?;
    dict.set_item("lower_2618", lower_2618_arr.reshape((rows, cols))?)?;
    dict.set_item("tp_long_band", tp_long_arr.reshape((rows, cols))?)?;
    dict.set_item("tp_short_band", tp_short_arr.reshape((rows, cols))?)?;
    dict.set_item("long_entry", long_entry_arr.reshape((rows, cols))?)?;
    dict.set_item("short_entry", short_entry_arr.reshape((rows, cols))?)?;
    dict.set_item("rejection_long", rejection_long_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "rejection_short",
        rejection_short_arr.reshape((rows, cols))?,
    )?;
    dict.set_item("long_bounce", long_bounce_arr.reshape((rows, cols))?)?;
    dict.set_item("short_bounce", short_bounce_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "lengths",
        combos
            .iter()
            .map(|combo| combo.length.unwrap_or(DEFAULT_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "atr_lengths",
        combos
            .iter()
            .map(|combo| combo.atr_length.unwrap_or(DEFAULT_ATR_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "sources",
        combos
            .iter()
            .map(|combo| combo.source.as_deref().unwrap_or(DEFAULT_SOURCE))
            .collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "use_atr_flags",
        combos
            .iter()
            .map(|combo| combo.use_atr.unwrap_or(DEFAULT_USE_ATR))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "tp_aggressiveness_values",
        combos
            .iter()
            .map(|combo| {
                combo
                    .tp_aggressiveness
                    .clone()
                    .unwrap_or_else(|| DEFAULT_TP_AGGRESSIVENESS.to_string())
            })
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_fibonacci_entry_bands_module(
    module: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(fibonacci_entry_bands_py, module)?)?;
    module.add_function(wrap_pyfunction!(fibonacci_entry_bands_batch_py, module)?)?;
    module.add_class::<FibonacciEntryBandsStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct FibonacciEntryBandsJsOutput {
    pub basis: Vec<f64>,
    pub trend: Vec<f64>,
    pub upper_0618: Vec<f64>,
    pub upper_1000: Vec<f64>,
    pub upper_1618: Vec<f64>,
    pub upper_2618: Vec<f64>,
    pub lower_0618: Vec<f64>,
    pub lower_1000: Vec<f64>,
    pub lower_1618: Vec<f64>,
    pub lower_2618: Vec<f64>,
    pub tp_long_band: Vec<f64>,
    pub tp_short_band: Vec<f64>,
    pub long_entry: Vec<f64>,
    pub short_entry: Vec<f64>,
    pub rejection_long: Vec<f64>,
    pub rejection_short: Vec<f64>,
    pub long_bounce: Vec<f64>,
    pub short_bounce: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "fibonacci_entry_bands_js")]
pub fn fibonacci_entry_bands_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &str,
    length: usize,
    atr_length: usize,
    use_atr: bool,
    tp_aggressiveness: &str,
) -> Result<JsValue, JsValue> {
    let input = FibonacciEntryBandsInput::from_slices(
        open,
        high,
        low,
        close,
        FibonacciEntryBandsParams {
            source: Some(source.to_string()),
            length: Some(length),
            atr_length: Some(atr_length),
            use_atr: Some(use_atr),
            tp_aggressiveness: Some(tp_aggressiveness.to_string()),
        },
    );
    let out = fibonacci_entry_bands(&input).map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&FibonacciEntryBandsJsOutput {
        basis: out.basis,
        trend: out.trend,
        upper_0618: out.upper_0618,
        upper_1000: out.upper_1000,
        upper_1618: out.upper_1618,
        upper_2618: out.upper_2618,
        lower_0618: out.lower_0618,
        lower_1000: out.lower_1000,
        lower_1618: out.lower_1618,
        lower_2618: out.lower_2618,
        tp_long_band: out.tp_long_band,
        tp_short_band: out.tp_short_band,
        long_entry: out.long_entry,
        short_entry: out.short_entry,
        rejection_long: out.rejection_long,
        rejection_short: out.rejection_short,
        long_bounce: out.long_bounce,
        short_bounce: out.short_bounce,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn fibonacci_entry_bands_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn fibonacci_entry_bands_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
fn has_duplicate_ptrs(ptrs: &[usize]) -> bool {
    for i in 0..ptrs.len() {
        for j in (i + 1)..ptrs.len() {
            if ptrs[i] == ptrs[j] {
                return true;
            }
        }
    }
    false
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn fibonacci_entry_bands_into(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    basis_ptr: *mut f64,
    trend_ptr: *mut f64,
    upper_0618_ptr: *mut f64,
    upper_1000_ptr: *mut f64,
    upper_1618_ptr: *mut f64,
    upper_2618_ptr: *mut f64,
    lower_0618_ptr: *mut f64,
    lower_1000_ptr: *mut f64,
    lower_1618_ptr: *mut f64,
    lower_2618_ptr: *mut f64,
    tp_long_ptr: *mut f64,
    tp_short_ptr: *mut f64,
    long_entry_ptr: *mut f64,
    short_entry_ptr: *mut f64,
    rejection_long_ptr: *mut f64,
    rejection_short_ptr: *mut f64,
    long_bounce_ptr: *mut f64,
    short_bounce_ptr: *mut f64,
    len: usize,
    source: &str,
    length: usize,
    atr_length: usize,
    use_atr: bool,
    tp_aggressiveness: &str,
) -> Result<(), JsValue> {
    let ptrs = [
        open_ptr as usize,
        high_ptr as usize,
        low_ptr as usize,
        close_ptr as usize,
        basis_ptr as usize,
        trend_ptr as usize,
        upper_0618_ptr as usize,
        upper_1000_ptr as usize,
        upper_1618_ptr as usize,
        upper_2618_ptr as usize,
        lower_0618_ptr as usize,
        lower_1000_ptr as usize,
        lower_1618_ptr as usize,
        lower_2618_ptr as usize,
        tp_long_ptr as usize,
        tp_short_ptr as usize,
        long_entry_ptr as usize,
        short_entry_ptr as usize,
        rejection_long_ptr as usize,
        rejection_short_ptr as usize,
        long_bounce_ptr as usize,
        short_bounce_ptr as usize,
    ];
    if ptrs.iter().any(|ptr| *ptr == 0) {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let input = FibonacciEntryBandsInput::from_slices(
            open,
            high,
            low,
            close,
            FibonacciEntryBandsParams {
                source: Some(source.to_string()),
                length: Some(length),
                atr_length: Some(atr_length),
                use_atr: Some(use_atr),
                tp_aggressiveness: Some(tp_aggressiveness.to_string()),
            },
        );
        let output_ptrs = &ptrs[4..];
        let need_temp = output_ptrs.iter().any(|ptr| {
            *ptr == open_ptr as usize
                || *ptr == high_ptr as usize
                || *ptr == low_ptr as usize
                || *ptr == close_ptr as usize
        }) || has_duplicate_ptrs(output_ptrs);

        if need_temp {
            let mut basis = vec![0.0; len];
            let mut trend = vec![0.0; len];
            let mut upper_0618 = vec![0.0; len];
            let mut upper_1000 = vec![0.0; len];
            let mut upper_1618 = vec![0.0; len];
            let mut upper_2618 = vec![0.0; len];
            let mut lower_0618 = vec![0.0; len];
            let mut lower_1000 = vec![0.0; len];
            let mut lower_1618 = vec![0.0; len];
            let mut lower_2618 = vec![0.0; len];
            let mut tp_long = vec![0.0; len];
            let mut tp_short = vec![0.0; len];
            let mut long_entry = vec![0.0; len];
            let mut short_entry = vec![0.0; len];
            let mut rejection_long = vec![0.0; len];
            let mut rejection_short = vec![0.0; len];
            let mut long_bounce = vec![0.0; len];
            let mut short_bounce = vec![0.0; len];
            fibonacci_entry_bands_into_slices(
                &mut basis,
                &mut trend,
                &mut upper_0618,
                &mut upper_1000,
                &mut upper_1618,
                &mut upper_2618,
                &mut lower_0618,
                &mut lower_1000,
                &mut lower_1618,
                &mut lower_2618,
                &mut tp_long,
                &mut tp_short,
                &mut long_entry,
                &mut short_entry,
                &mut rejection_long,
                &mut rejection_short,
                &mut long_bounce,
                &mut short_bounce,
                &input,
                Kernel::Auto,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
            std::slice::from_raw_parts_mut(basis_ptr, len).copy_from_slice(&basis);
            std::slice::from_raw_parts_mut(trend_ptr, len).copy_from_slice(&trend);
            std::slice::from_raw_parts_mut(upper_0618_ptr, len).copy_from_slice(&upper_0618);
            std::slice::from_raw_parts_mut(upper_1000_ptr, len).copy_from_slice(&upper_1000);
            std::slice::from_raw_parts_mut(upper_1618_ptr, len).copy_from_slice(&upper_1618);
            std::slice::from_raw_parts_mut(upper_2618_ptr, len).copy_from_slice(&upper_2618);
            std::slice::from_raw_parts_mut(lower_0618_ptr, len).copy_from_slice(&lower_0618);
            std::slice::from_raw_parts_mut(lower_1000_ptr, len).copy_from_slice(&lower_1000);
            std::slice::from_raw_parts_mut(lower_1618_ptr, len).copy_from_slice(&lower_1618);
            std::slice::from_raw_parts_mut(lower_2618_ptr, len).copy_from_slice(&lower_2618);
            std::slice::from_raw_parts_mut(tp_long_ptr, len).copy_from_slice(&tp_long);
            std::slice::from_raw_parts_mut(tp_short_ptr, len).copy_from_slice(&tp_short);
            std::slice::from_raw_parts_mut(long_entry_ptr, len).copy_from_slice(&long_entry);
            std::slice::from_raw_parts_mut(short_entry_ptr, len).copy_from_slice(&short_entry);
            std::slice::from_raw_parts_mut(rejection_long_ptr, len)
                .copy_from_slice(&rejection_long);
            std::slice::from_raw_parts_mut(rejection_short_ptr, len)
                .copy_from_slice(&rejection_short);
            std::slice::from_raw_parts_mut(long_bounce_ptr, len).copy_from_slice(&long_bounce);
            std::slice::from_raw_parts_mut(short_bounce_ptr, len).copy_from_slice(&short_bounce);
        } else {
            fibonacci_entry_bands_into_slices(
                std::slice::from_raw_parts_mut(basis_ptr, len),
                std::slice::from_raw_parts_mut(trend_ptr, len),
                std::slice::from_raw_parts_mut(upper_0618_ptr, len),
                std::slice::from_raw_parts_mut(upper_1000_ptr, len),
                std::slice::from_raw_parts_mut(upper_1618_ptr, len),
                std::slice::from_raw_parts_mut(upper_2618_ptr, len),
                std::slice::from_raw_parts_mut(lower_0618_ptr, len),
                std::slice::from_raw_parts_mut(lower_1000_ptr, len),
                std::slice::from_raw_parts_mut(lower_1618_ptr, len),
                std::slice::from_raw_parts_mut(lower_2618_ptr, len),
                std::slice::from_raw_parts_mut(tp_long_ptr, len),
                std::slice::from_raw_parts_mut(tp_short_ptr, len),
                std::slice::from_raw_parts_mut(long_entry_ptr, len),
                std::slice::from_raw_parts_mut(short_entry_ptr, len),
                std::slice::from_raw_parts_mut(rejection_long_ptr, len),
                std::slice::from_raw_parts_mut(rejection_short_ptr, len),
                std::slice::from_raw_parts_mut(long_bounce_ptr, len),
                std::slice::from_raw_parts_mut(short_bounce_ptr, len),
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
pub struct FibonacciEntryBandsBatchJsConfig {
    pub length_range: Option<(usize, usize, usize)>,
    pub atr_length_range: Option<(usize, usize, usize)>,
    pub source: Option<String>,
    pub use_atr: Option<bool>,
    pub tp_aggressiveness: Option<String>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct FibonacciEntryBandsBatchJsOutput {
    pub basis: Vec<f64>,
    pub trend: Vec<f64>,
    pub upper_0618: Vec<f64>,
    pub upper_1000: Vec<f64>,
    pub upper_1618: Vec<f64>,
    pub upper_2618: Vec<f64>,
    pub lower_0618: Vec<f64>,
    pub lower_1000: Vec<f64>,
    pub lower_1618: Vec<f64>,
    pub lower_2618: Vec<f64>,
    pub tp_long_band: Vec<f64>,
    pub tp_short_band: Vec<f64>,
    pub long_entry: Vec<f64>,
    pub short_entry: Vec<f64>,
    pub rejection_long: Vec<f64>,
    pub rejection_short: Vec<f64>,
    pub long_bounce: Vec<f64>,
    pub short_bounce: Vec<f64>,
    pub combos: Vec<FibonacciEntryBandsParams>,
    pub lengths: Vec<usize>,
    pub atr_lengths: Vec<usize>,
    pub sources: Vec<String>,
    pub use_atr_flags: Vec<bool>,
    pub tp_aggressiveness_values: Vec<String>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "fibonacci_entry_bands_batch_js")]
pub fn fibonacci_entry_bands_batch_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: FibonacciEntryBandsBatchJsConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let sweep = FibonacciEntryBandsBatchRange {
        length: config
            .length_range
            .unwrap_or((DEFAULT_LENGTH, DEFAULT_LENGTH, 0)),
        atr_length: config
            .atr_length_range
            .unwrap_or((DEFAULT_ATR_LENGTH, DEFAULT_ATR_LENGTH, 0)),
        source: config.source.unwrap_or_else(|| DEFAULT_SOURCE.to_string()),
        use_atr: config.use_atr.unwrap_or(DEFAULT_USE_ATR),
        tp_aggressiveness: config
            .tp_aggressiveness
            .unwrap_or_else(|| DEFAULT_TP_AGGRESSIVENESS.to_string()),
    };
    let out = fibonacci_entry_bands_batch_with_kernel(open, high, low, close, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&FibonacciEntryBandsBatchJsOutput {
        lengths: out
            .combos
            .iter()
            .map(|combo| combo.length.unwrap_or(DEFAULT_LENGTH))
            .collect(),
        atr_lengths: out
            .combos
            .iter()
            .map(|combo| combo.atr_length.unwrap_or(DEFAULT_ATR_LENGTH))
            .collect(),
        sources: out
            .combos
            .iter()
            .map(|combo| {
                combo
                    .source
                    .clone()
                    .unwrap_or_else(|| DEFAULT_SOURCE.to_string())
            })
            .collect(),
        use_atr_flags: out
            .combos
            .iter()
            .map(|combo| combo.use_atr.unwrap_or(DEFAULT_USE_ATR))
            .collect(),
        tp_aggressiveness_values: out
            .combos
            .iter()
            .map(|combo| {
                combo
                    .tp_aggressiveness
                    .clone()
                    .unwrap_or_else(|| DEFAULT_TP_AGGRESSIVENESS.to_string())
            })
            .collect(),
        basis: out.basis,
        trend: out.trend,
        upper_0618: out.upper_0618,
        upper_1000: out.upper_1000,
        upper_1618: out.upper_1618,
        upper_2618: out.upper_2618,
        lower_0618: out.lower_0618,
        lower_1000: out.lower_1000,
        lower_1618: out.lower_1618,
        lower_2618: out.lower_2618,
        tp_long_band: out.tp_long_band,
        tp_short_band: out.tp_short_band,
        long_entry: out.long_entry,
        short_entry: out.short_entry,
        rejection_long: out.rejection_long,
        rejection_short: out.rejection_short,
        long_bounce: out.long_bounce,
        short_bounce: out.short_bounce,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn fibonacci_entry_bands_batch_into(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    basis_ptr: *mut f64,
    trend_ptr: *mut f64,
    upper_0618_ptr: *mut f64,
    upper_1000_ptr: *mut f64,
    upper_1618_ptr: *mut f64,
    upper_2618_ptr: *mut f64,
    lower_0618_ptr: *mut f64,
    lower_1000_ptr: *mut f64,
    lower_1618_ptr: *mut f64,
    lower_2618_ptr: *mut f64,
    tp_long_ptr: *mut f64,
    tp_short_ptr: *mut f64,
    long_entry_ptr: *mut f64,
    short_entry_ptr: *mut f64,
    rejection_long_ptr: *mut f64,
    rejection_short_ptr: *mut f64,
    long_bounce_ptr: *mut f64,
    short_bounce_ptr: *mut f64,
    len: usize,
    length_start: usize,
    length_end: usize,
    length_step: usize,
    atr_length_start: usize,
    atr_length_end: usize,
    atr_length_step: usize,
    source: &str,
    use_atr: bool,
    tp_aggressiveness: &str,
) -> Result<usize, JsValue> {
    let ptrs = [
        open_ptr as usize,
        high_ptr as usize,
        low_ptr as usize,
        close_ptr as usize,
        basis_ptr as usize,
        trend_ptr as usize,
        upper_0618_ptr as usize,
        upper_1000_ptr as usize,
        upper_1618_ptr as usize,
        upper_2618_ptr as usize,
        lower_0618_ptr as usize,
        lower_1000_ptr as usize,
        lower_1618_ptr as usize,
        lower_2618_ptr as usize,
        tp_long_ptr as usize,
        tp_short_ptr as usize,
        long_entry_ptr as usize,
        short_entry_ptr as usize,
        rejection_long_ptr as usize,
        rejection_short_ptr as usize,
        long_bounce_ptr as usize,
        short_bounce_ptr as usize,
    ];
    if ptrs.iter().any(|ptr| *ptr == 0) {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    let sweep = FibonacciEntryBandsBatchRange {
        length: (length_start, length_end, length_step),
        atr_length: (atr_length_start, atr_length_end, atr_length_step),
        source: source.to_string(),
        use_atr,
        tp_aggressiveness: tp_aggressiveness.to_string(),
    };
    let rows = expand_grid(&sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?
        .len();
    let total = rows
        .checked_mul(len)
        .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;

    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let output_ptrs = &ptrs[4..];
        let need_temp = output_ptrs.iter().any(|ptr| {
            *ptr == open_ptr as usize
                || *ptr == high_ptr as usize
                || *ptr == low_ptr as usize
                || *ptr == close_ptr as usize
        }) || has_duplicate_ptrs(output_ptrs);

        if need_temp {
            let mut basis = vec![0.0; total];
            let mut trend = vec![0.0; total];
            let mut upper_0618 = vec![0.0; total];
            let mut upper_1000 = vec![0.0; total];
            let mut upper_1618 = vec![0.0; total];
            let mut upper_2618 = vec![0.0; total];
            let mut lower_0618 = vec![0.0; total];
            let mut lower_1000 = vec![0.0; total];
            let mut lower_1618 = vec![0.0; total];
            let mut lower_2618 = vec![0.0; total];
            let mut tp_long = vec![0.0; total];
            let mut tp_short = vec![0.0; total];
            let mut long_entry = vec![0.0; total];
            let mut short_entry = vec![0.0; total];
            let mut rejection_long = vec![0.0; total];
            let mut rejection_short = vec![0.0; total];
            let mut long_bounce = vec![0.0; total];
            let mut short_bounce = vec![0.0; total];
            let rows = fibonacci_entry_bands_batch_inner_into(
                open,
                high,
                low,
                close,
                &sweep,
                Kernel::Auto,
                &mut basis,
                &mut trend,
                &mut upper_0618,
                &mut upper_1000,
                &mut upper_1618,
                &mut upper_2618,
                &mut lower_0618,
                &mut lower_1000,
                &mut lower_1618,
                &mut lower_2618,
                &mut tp_long,
                &mut tp_short,
                &mut long_entry,
                &mut short_entry,
                &mut rejection_long,
                &mut rejection_short,
                &mut long_bounce,
                &mut short_bounce,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?
            .len();
            std::slice::from_raw_parts_mut(basis_ptr, total).copy_from_slice(&basis);
            std::slice::from_raw_parts_mut(trend_ptr, total).copy_from_slice(&trend);
            std::slice::from_raw_parts_mut(upper_0618_ptr, total).copy_from_slice(&upper_0618);
            std::slice::from_raw_parts_mut(upper_1000_ptr, total).copy_from_slice(&upper_1000);
            std::slice::from_raw_parts_mut(upper_1618_ptr, total).copy_from_slice(&upper_1618);
            std::slice::from_raw_parts_mut(upper_2618_ptr, total).copy_from_slice(&upper_2618);
            std::slice::from_raw_parts_mut(lower_0618_ptr, total).copy_from_slice(&lower_0618);
            std::slice::from_raw_parts_mut(lower_1000_ptr, total).copy_from_slice(&lower_1000);
            std::slice::from_raw_parts_mut(lower_1618_ptr, total).copy_from_slice(&lower_1618);
            std::slice::from_raw_parts_mut(lower_2618_ptr, total).copy_from_slice(&lower_2618);
            std::slice::from_raw_parts_mut(tp_long_ptr, total).copy_from_slice(&tp_long);
            std::slice::from_raw_parts_mut(tp_short_ptr, total).copy_from_slice(&tp_short);
            std::slice::from_raw_parts_mut(long_entry_ptr, total).copy_from_slice(&long_entry);
            std::slice::from_raw_parts_mut(short_entry_ptr, total).copy_from_slice(&short_entry);
            std::slice::from_raw_parts_mut(rejection_long_ptr, total)
                .copy_from_slice(&rejection_long);
            std::slice::from_raw_parts_mut(rejection_short_ptr, total)
                .copy_from_slice(&rejection_short);
            std::slice::from_raw_parts_mut(long_bounce_ptr, total).copy_from_slice(&long_bounce);
            std::slice::from_raw_parts_mut(short_bounce_ptr, total).copy_from_slice(&short_bounce);
            Ok(rows)
        } else {
            let rows = fibonacci_entry_bands_batch_inner_into(
                open,
                high,
                low,
                close,
                &sweep,
                Kernel::Auto,
                std::slice::from_raw_parts_mut(basis_ptr, total),
                std::slice::from_raw_parts_mut(trend_ptr, total),
                std::slice::from_raw_parts_mut(upper_0618_ptr, total),
                std::slice::from_raw_parts_mut(upper_1000_ptr, total),
                std::slice::from_raw_parts_mut(upper_1618_ptr, total),
                std::slice::from_raw_parts_mut(upper_2618_ptr, total),
                std::slice::from_raw_parts_mut(lower_0618_ptr, total),
                std::slice::from_raw_parts_mut(lower_1000_ptr, total),
                std::slice::from_raw_parts_mut(lower_1618_ptr, total),
                std::slice::from_raw_parts_mut(lower_2618_ptr, total),
                std::slice::from_raw_parts_mut(tp_long_ptr, total),
                std::slice::from_raw_parts_mut(tp_short_ptr, total),
                std::slice::from_raw_parts_mut(long_entry_ptr, total),
                std::slice::from_raw_parts_mut(short_entry_ptr, total),
                std::slice::from_raw_parts_mut(rejection_long_ptr, total),
                std::slice::from_raw_parts_mut(rejection_short_ptr, total),
                std::slice::from_raw_parts_mut(long_bounce_ptr, total),
                std::slice::from_raw_parts_mut(short_bounce_ptr, total),
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?
            .len();
            Ok(rows)
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn fibonacci_entry_bands_output_into_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &str,
    length: usize,
    atr_length: usize,
    use_atr: bool,
    tp_aggressiveness: &str,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = fibonacci_entry_bands_js(
        open,
        high,
        low,
        close,
        source,
        length,
        atr_length,
        use_atr,
        tp_aggressiveness,
    )?;
    crate::write_wasm_object_f64_outputs("fibonacci_entry_bands_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn fibonacci_entry_bands_batch_output_into_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = fibonacci_entry_bands_batch_js(open, high, low, close, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "fibonacci_entry_bands_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_ohlc(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let open: Vec<f64> = (0..len)
            .map(|i| {
                let x = i as f64;
                100.0 + x * 0.05 + (x * 0.09).sin() * 1.8 + (x * 0.021).cos() * 0.7
            })
            .collect();
        let close: Vec<f64> = open
            .iter()
            .enumerate()
            .map(|(i, &o)| o + (i as f64 * 0.13).cos() * 0.9)
            .collect();
        let high: Vec<f64> = open
            .iter()
            .zip(close.iter())
            .enumerate()
            .map(|(i, (&o, &c))| o.max(c) + 0.7 + (i as f64 * 0.05).sin().abs() * 0.2)
            .collect();
        let low: Vec<f64> = open
            .iter()
            .zip(close.iter())
            .enumerate()
            .map(|(i, (&o, &c))| o.min(c) - 0.7 - (i as f64 * 0.04).cos().abs() * 0.2)
            .collect();
        (open, high, low, close)
    }

    fn assert_series_eq(left: &[f64], right: &[f64], tol: f64) {
        assert_eq!(left.len(), right.len());
        for (&lhs, &rhs) in left.iter().zip(right.iter()) {
            assert!(
                (lhs.is_nan() && rhs.is_nan()) || (lhs - rhs).abs() <= tol,
                "series mismatch: left={lhs:?}, right={rhs:?}"
            );
        }
    }

    #[test]
    fn fibonacci_entry_bands_output_contract() {
        let (open, high, low, close) = sample_ohlc(320);
        let out = fibonacci_entry_bands(&FibonacciEntryBandsInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            FibonacciEntryBandsParams::default(),
        ))
        .unwrap();
        assert_eq!(out.basis.len(), close.len());
        assert_eq!(out.upper_2618.len(), close.len());
        assert_eq!(out.long_entry.len(), close.len());
        assert!(out.basis.iter().any(|v| v.is_finite()));
    }

    #[test]
    fn fibonacci_entry_bands_rejects_invalid_params() {
        let (open, high, low, close) = sample_ohlc(32);
        let err = fibonacci_entry_bands(&FibonacciEntryBandsInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            FibonacciEntryBandsParams {
                source: Some("bad".to_string()),
                ..FibonacciEntryBandsParams::default()
            },
        ))
        .unwrap_err();
        assert!(matches!(
            err,
            FibonacciEntryBandsError::InvalidSource { .. }
        ));

        let err = fibonacci_entry_bands(&FibonacciEntryBandsInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            FibonacciEntryBandsParams {
                length: Some(0),
                ..FibonacciEntryBandsParams::default()
            },
        ))
        .unwrap_err();
        assert!(matches!(
            err,
            FibonacciEntryBandsError::InvalidLength { .. }
        ));
    }

    #[test]
    fn fibonacci_entry_bands_stream_matches_batch_with_reset() {
        let (mut open, mut high, mut low, mut close) = sample_ohlc(220);
        open[110] = f64::NAN;
        high[110] = f64::NAN;
        low[110] = f64::NAN;
        close[110] = f64::NAN;

        let params = FibonacciEntryBandsParams {
            source: Some("hlc3".to_string()),
            length: Some(16),
            atr_length: Some(11),
            use_atr: Some(true),
            tp_aggressiveness: Some("high".to_string()),
        };
        let batch = fibonacci_entry_bands(&FibonacciEntryBandsInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            params.clone(),
        ))
        .unwrap();
        let mut stream = FibonacciEntryBandsStream::try_new(params).unwrap();

        let mut basis = Vec::with_capacity(close.len());
        let mut trend = Vec::with_capacity(close.len());
        let mut tp_long = Vec::with_capacity(close.len());
        for i in 0..close.len() {
            if let Some(point) = stream.update(open[i], high[i], low[i], close[i]) {
                basis.push(point.basis);
                trend.push(point.trend);
                tp_long.push(point.tp_long_band);
            } else {
                basis.push(f64::NAN);
                trend.push(f64::NAN);
                tp_long.push(f64::NAN);
            }
        }

        assert_series_eq(&basis, &batch.basis, 1e-12);
        assert_series_eq(&trend, &batch.trend, 1e-12);
        assert_series_eq(&tp_long, &batch.tp_long_band, 1e-12);
    }

    #[test]
    fn fibonacci_entry_bands_into_matches_api() {
        let (open, high, low, close) = sample_ohlc(192);
        let input = FibonacciEntryBandsInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            FibonacciEntryBandsParams {
                source: Some("hlc3".to_string()),
                length: Some(14),
                atr_length: Some(9),
                use_atr: Some(false),
                tp_aggressiveness: Some("low".to_string()),
            },
        );
        let direct = fibonacci_entry_bands(&input).unwrap();
        let mut basis = vec![0.0; close.len()];
        let mut trend = vec![0.0; close.len()];
        let mut upper_0618 = vec![0.0; close.len()];
        let mut upper_1000 = vec![0.0; close.len()];
        let mut upper_1618 = vec![0.0; close.len()];
        let mut upper_2618 = vec![0.0; close.len()];
        let mut lower_0618 = vec![0.0; close.len()];
        let mut lower_1000 = vec![0.0; close.len()];
        let mut lower_1618 = vec![0.0; close.len()];
        let mut lower_2618 = vec![0.0; close.len()];
        let mut tp_long = vec![0.0; close.len()];
        let mut tp_short = vec![0.0; close.len()];
        let mut long_entry = vec![0.0; close.len()];
        let mut short_entry = vec![0.0; close.len()];
        let mut rejection_long = vec![0.0; close.len()];
        let mut rejection_short = vec![0.0; close.len()];
        let mut long_bounce = vec![0.0; close.len()];
        let mut short_bounce = vec![0.0; close.len()];

        fibonacci_entry_bands_into(
            &input,
            &mut basis,
            &mut trend,
            &mut upper_0618,
            &mut upper_1000,
            &mut upper_1618,
            &mut upper_2618,
            &mut lower_0618,
            &mut lower_1000,
            &mut lower_1618,
            &mut lower_2618,
            &mut tp_long,
            &mut tp_short,
            &mut long_entry,
            &mut short_entry,
            &mut rejection_long,
            &mut rejection_short,
            &mut long_bounce,
            &mut short_bounce,
        )
        .unwrap();

        assert_series_eq(&basis, &direct.basis, 1e-12);
        assert_series_eq(&upper_2618, &direct.upper_2618, 1e-12);
        assert_series_eq(&short_bounce, &direct.short_bounce, 1e-12);
    }

    #[test]
    fn fibonacci_entry_bands_batch_single_param_matches_single() {
        let (open, high, low, close) = sample_ohlc(160);
        let sweep = FibonacciEntryBandsBatchRange {
            length: (17, 17, 0),
            atr_length: (10, 10, 0),
            source: "hlc3".to_string(),
            use_atr: true,
            tp_aggressiveness: "medium".to_string(),
        };
        let batch = fibonacci_entry_bands_batch_with_kernel(
            &open,
            &high,
            &low,
            &close,
            &sweep,
            Kernel::Auto,
        )
        .unwrap();
        let single = fibonacci_entry_bands(&FibonacciEntryBandsInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            FibonacciEntryBandsParams {
                source: Some("hlc3".to_string()),
                length: Some(17),
                atr_length: Some(10),
                use_atr: Some(true),
                tp_aggressiveness: Some("medium".to_string()),
            },
        ))
        .unwrap();
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, close.len());
        assert_series_eq(&batch.basis[..close.len()], &single.basis, 1e-12);
        assert_series_eq(
            &batch.rejection_short[..close.len()],
            &single.rejection_short,
            1e-12,
        );
    }
}
