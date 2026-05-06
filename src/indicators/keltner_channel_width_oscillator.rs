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
    alloc_with_nan_prefix, detect_best_batch_kernel, init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

impl<'a> AsRef<[f64]> for KeltnerChannelWidthOscillatorInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            KeltnerChannelWidthOscillatorData::Candles { candles, source } => {
                kcwo_source(candles, source)
            }
            KeltnerChannelWidthOscillatorData::Slices { source, .. } => source,
        }
    }
}

#[inline(always)]
fn kcwo_source<'a>(candles: &'a Candles, source: &str) -> &'a [f64] {
    if source.eq_ignore_ascii_case("close") {
        &candles.close
    } else {
        source_type(candles, source)
    }
}

#[derive(Debug, Clone)]
pub enum KeltnerChannelWidthOscillatorData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        source: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct KeltnerChannelWidthOscillatorOutput {
    pub kbw: Vec<f64>,
    pub kbw_sma: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct KeltnerChannelWidthOscillatorParams {
    pub length: Option<usize>,
    pub multiplier: Option<f64>,
    pub use_exponential: Option<bool>,
    pub bands_style: Option<String>,
    pub atr_length: Option<usize>,
}

impl Default for KeltnerChannelWidthOscillatorParams {
    fn default() -> Self {
        Self {
            length: Some(20),
            multiplier: Some(2.0),
            use_exponential: Some(true),
            bands_style: Some("Average True Range".to_string()),
            atr_length: Some(10),
        }
    }
}

#[derive(Debug, Clone)]
pub struct KeltnerChannelWidthOscillatorInput<'a> {
    pub data: KeltnerChannelWidthOscillatorData<'a>,
    pub params: KeltnerChannelWidthOscillatorParams,
}

impl<'a> KeltnerChannelWidthOscillatorInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: KeltnerChannelWidthOscillatorParams,
    ) -> Self {
        Self {
            data: KeltnerChannelWidthOscillatorData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        source: &'a [f64],
        params: KeltnerChannelWidthOscillatorParams,
    ) -> Self {
        Self {
            data: KeltnerChannelWidthOscillatorData::Slices {
                high,
                low,
                close,
                source,
            },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(
            candles,
            "close",
            KeltnerChannelWidthOscillatorParams::default(),
        )
    }

    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(20)
    }

    #[inline]
    pub fn get_multiplier(&self) -> f64 {
        self.params.multiplier.unwrap_or(2.0)
    }

    #[inline]
    pub fn get_use_exponential(&self) -> bool {
        self.params.use_exponential.unwrap_or(true)
    }

    #[inline]
    pub fn bands_style_str(&self) -> &str {
        self.params
            .bands_style
            .as_deref()
            .unwrap_or("Average True Range")
    }

    #[inline]
    pub fn get_atr_length(&self) -> usize {
        self.params.atr_length.unwrap_or(10)
    }

    #[inline]
    pub fn as_refs(&'a self) -> (&'a [f64], &'a [f64], &'a [f64], &'a [f64]) {
        match &self.data {
            KeltnerChannelWidthOscillatorData::Candles { candles, source } => (
                candles.high.as_slice(),
                candles.low.as_slice(),
                candles.close.as_slice(),
                kcwo_source(candles, source),
            ),
            KeltnerChannelWidthOscillatorData::Slices {
                high,
                low,
                close,
                source,
            } => (*high, *low, *close, *source),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct KeltnerChannelWidthOscillatorBuilder {
    length: Option<usize>,
    multiplier: Option<f64>,
    use_exponential: Option<bool>,
    bands_style: Option<KeltnerWidthBandsStyle>,
    atr_length: Option<usize>,
    kernel: Kernel,
}

impl Default for KeltnerChannelWidthOscillatorBuilder {
    fn default() -> Self {
        Self {
            length: None,
            multiplier: None,
            use_exponential: None,
            bands_style: None,
            atr_length: None,
            kernel: Kernel::Auto,
        }
    }
}

impl KeltnerChannelWidthOscillatorBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn length(mut self, value: usize) -> Self {
        self.length = Some(value);
        self
    }

    #[inline(always)]
    pub fn multiplier(mut self, value: f64) -> Self {
        self.multiplier = Some(value);
        self
    }

    #[inline(always)]
    pub fn use_exponential(mut self, value: bool) -> Self {
        self.use_exponential = Some(value);
        self
    }

    #[inline(always)]
    pub fn bands_style(mut self, value: &str) -> Result<Self, KeltnerChannelWidthOscillatorError> {
        self.bands_style = Some(parse_bands_style(value)?);
        Ok(self)
    }

    #[inline(always)]
    pub fn atr_length(mut self, value: usize) -> Self {
        self.atr_length = Some(value);
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
    ) -> Result<KeltnerChannelWidthOscillatorOutput, KeltnerChannelWidthOscillatorError> {
        let params = KeltnerChannelWidthOscillatorParams {
            length: self.length,
            multiplier: self.multiplier,
            use_exponential: self.use_exponential,
            bands_style: Some(
                self.bands_style
                    .unwrap_or(KeltnerWidthBandsStyle::AverageTrueRange)
                    .as_str()
                    .to_string(),
            ),
            atr_length: self.atr_length,
        };
        let input = KeltnerChannelWidthOscillatorInput::from_candles(candles, "close", params);
        keltner_channel_width_oscillator_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        source: &[f64],
    ) -> Result<KeltnerChannelWidthOscillatorOutput, KeltnerChannelWidthOscillatorError> {
        let params = KeltnerChannelWidthOscillatorParams {
            length: self.length,
            multiplier: self.multiplier,
            use_exponential: self.use_exponential,
            bands_style: Some(
                self.bands_style
                    .unwrap_or(KeltnerWidthBandsStyle::AverageTrueRange)
                    .as_str()
                    .to_string(),
            ),
            atr_length: self.atr_length,
        };
        let input =
            KeltnerChannelWidthOscillatorInput::from_slices(high, low, close, source, params);
        keltner_channel_width_oscillator_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<KeltnerChannelWidthOscillatorStream, KeltnerChannelWidthOscillatorError> {
        KeltnerChannelWidthOscillatorStream::try_new(KeltnerChannelWidthOscillatorParams {
            length: self.length,
            multiplier: self.multiplier,
            use_exponential: self.use_exponential,
            bands_style: Some(
                self.bands_style
                    .unwrap_or(KeltnerWidthBandsStyle::AverageTrueRange)
                    .as_str()
                    .to_string(),
            ),
            atr_length: self.atr_length,
        })
    }
}

#[derive(Debug, Error)]
pub enum KeltnerChannelWidthOscillatorError {
    #[error("keltner_channel_width_oscillator: Empty input data.")]
    EmptyInputData,
    #[error(
        "keltner_channel_width_oscillator: Data length mismatch across high, low, close, and source."
    )]
    DataLengthMismatch,
    #[error("keltner_channel_width_oscillator: All OHLC/source values are invalid.")]
    AllValuesNaN,
    #[error(
        "keltner_channel_width_oscillator: Invalid length: length = {length}, data length = {data_len}"
    )]
    InvalidLength { length: usize, data_len: usize },
    #[error(
        "keltner_channel_width_oscillator: Invalid ATR length: atr_length = {atr_length}, data length = {data_len}"
    )]
    InvalidAtrLength { atr_length: usize, data_len: usize },
    #[error("keltner_channel_width_oscillator: Invalid multiplier: multiplier = {multiplier}")]
    InvalidMultiplier { multiplier: f64 },
    #[error("keltner_channel_width_oscillator: Invalid bands style: {bands_style}")]
    InvalidBandsStyle { bands_style: String },
    #[error(
        "keltner_channel_width_oscillator: Not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "keltner_channel_width_oscillator: Output length mismatch: expected = {expected}, got = {got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error(
        "keltner_channel_width_oscillator: Invalid range: start={start}, end={end}, step={step}"
    )]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("keltner_channel_width_oscillator: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KeltnerWidthBandsStyle {
    AverageTrueRange,
    TrueRange,
    Range,
}

impl KeltnerWidthBandsStyle {
    #[inline(always)]
    fn as_str(self) -> &'static str {
        match self {
            Self::AverageTrueRange => "Average True Range",
            Self::TrueRange => "True Range",
            Self::Range => "Range",
        }
    }
}

#[inline(always)]
fn parse_bands_style(
    value: &str,
) -> Result<KeltnerWidthBandsStyle, KeltnerChannelWidthOscillatorError> {
    let normalized = value.trim().to_ascii_uppercase();
    match normalized.as_str() {
        "AVERAGE TRUE RANGE" | "ATR" => Ok(KeltnerWidthBandsStyle::AverageTrueRange),
        "TRUE RANGE" | "TR" => Ok(KeltnerWidthBandsStyle::TrueRange),
        "RANGE" => Ok(KeltnerWidthBandsStyle::Range),
        _ => Err(KeltnerChannelWidthOscillatorError::InvalidBandsStyle {
            bands_style: value.to_string(),
        }),
    }
}

#[inline(always)]
fn normalize_bands_style(value: &str) -> Result<String, KeltnerChannelWidthOscillatorError> {
    Ok(parse_bands_style(value)?.as_str().to_string())
}

#[inline(always)]
fn is_valid_bar(high: f64, low: f64, close: f64, source: f64) -> bool {
    high.is_finite() && low.is_finite() && close.is_finite() && source.is_finite() && high >= low
}

#[inline(always)]
fn first_valid_bar(high: &[f64], low: &[f64], close: &[f64], source: &[f64]) -> Option<usize> {
    (0..source.len()).find(|&i| is_valid_bar(high[i], low[i], close[i], source[i]))
}

#[inline(always)]
fn width_needed_bars(
    length: usize,
    atr_length: usize,
    bands_style: KeltnerWidthBandsStyle,
) -> usize {
    match bands_style {
        KeltnerWidthBandsStyle::AverageTrueRange => length.max(atr_length),
        KeltnerWidthBandsStyle::TrueRange | KeltnerWidthBandsStyle::Range => length,
    }
}

#[inline(always)]
fn kbw_warmup(
    length: usize,
    atr_length: usize,
    bands_style: KeltnerWidthBandsStyle,
    first: usize,
) -> usize {
    first + width_needed_bars(length, atr_length, bands_style) - 1
}

#[inline(always)]
fn kbw_sma_warmup(
    length: usize,
    atr_length: usize,
    bands_style: KeltnerWidthBandsStyle,
    first: usize,
) -> usize {
    kbw_warmup(length, atr_length, bands_style, first) + length - 1
}

#[derive(Clone, Debug)]
struct RollingSmaState {
    period: usize,
    count: usize,
    head: usize,
    sum: f64,
    buffer: Vec<f64>,
}

impl RollingSmaState {
    #[inline]
    fn new(period: usize) -> Self {
        Self {
            period,
            count: 0,
            head: 0,
            sum: 0.0,
            buffer: vec![0.0; period.max(1)],
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.count = 0;
        self.head = 0;
        self.sum = 0.0;
        self.buffer.fill(0.0);
    }

    #[inline]
    fn update(&mut self, value: f64) -> Option<f64> {
        if self.count < self.period {
            self.buffer[self.count] = value;
            self.sum += value;
            self.count += 1;
            if self.count == self.period {
                return Some(self.sum / self.period as f64);
            }
            return None;
        }

        let old = self.buffer[self.head];
        self.buffer[self.head] = value;
        self.sum += value - old;
        self.head += 1;
        if self.head == self.period {
            self.head = 0;
        }
        Some(self.sum / self.period as f64)
    }
}

#[derive(Clone, Debug)]
struct SeededEmaState {
    period: usize,
    alpha: f64,
    count: usize,
    sum: f64,
    value: f64,
    seeded: bool,
}

impl SeededEmaState {
    #[inline]
    fn new(period: usize) -> Self {
        Self {
            period,
            alpha: 2.0 / (period as f64 + 1.0),
            count: 0,
            sum: 0.0,
            value: f64::NAN,
            seeded: false,
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.count = 0;
        self.sum = 0.0;
        self.value = f64::NAN;
        self.seeded = false;
    }

    #[inline]
    fn update(&mut self, value: f64) -> Option<f64> {
        if !self.seeded {
            self.sum += value;
            self.count += 1;
            if self.count == self.period {
                self.value = self.sum / self.period as f64;
                self.seeded = true;
                return Some(self.value);
            }
            return None;
        }

        self.value = self.alpha.mul_add(value - self.value, self.value);
        Some(self.value)
    }
}

#[derive(Clone, Debug)]
struct SeededRmaState {
    period: usize,
    alpha: f64,
    count: usize,
    sum: f64,
    value: f64,
    seeded: bool,
}

impl SeededRmaState {
    #[inline]
    fn new(period: usize) -> Self {
        Self {
            period,
            alpha: 1.0 / period as f64,
            count: 0,
            sum: 0.0,
            value: f64::NAN,
            seeded: false,
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.count = 0;
        self.sum = 0.0;
        self.value = f64::NAN;
        self.seeded = false;
    }

    #[inline]
    fn update(&mut self, value: f64) -> Option<f64> {
        if !self.seeded {
            self.sum += value;
            self.count += 1;
            if self.count == self.period {
                self.value = self.sum / self.period as f64;
                self.seeded = true;
                return Some(self.value);
            }
            return None;
        }

        self.value = self.alpha.mul_add(value - self.value, self.value);
        Some(self.value)
    }
}

#[derive(Clone, Debug)]
struct TrueRangeState {
    prev_close: Option<f64>,
}

impl TrueRangeState {
    #[inline]
    fn new() -> Self {
        Self { prev_close: None }
    }

    #[inline]
    fn reset(&mut self) {
        self.prev_close = None;
    }

    #[inline]
    fn update(&mut self, high: f64, low: f64, close: f64) -> f64 {
        let tr = match self.prev_close {
            Some(prev_close) => {
                let up = if high > prev_close { high } else { prev_close };
                let dn = if low < prev_close { low } else { prev_close };
                up - dn
            }
            None => high - low,
        };
        self.prev_close = Some(close);
        tr
    }
}

#[derive(Clone, Debug)]
pub struct KeltnerChannelWidthOscillatorStream {
    multiplier: f64,
    center_use_exponential: bool,
    bands_style: KeltnerWidthBandsStyle,
    center_sma: RollingSmaState,
    center_ema: SeededEmaState,
    true_range: TrueRangeState,
    atr_rma: SeededRmaState,
    range_rma: SeededRmaState,
    width_sma: RollingSmaState,
}

impl KeltnerChannelWidthOscillatorStream {
    #[inline]
    pub fn try_new(
        params: KeltnerChannelWidthOscillatorParams,
    ) -> Result<Self, KeltnerChannelWidthOscillatorError> {
        let length = params.length.unwrap_or(20);
        if length == 0 {
            return Err(KeltnerChannelWidthOscillatorError::InvalidLength {
                length,
                data_len: 0,
            });
        }

        let atr_length = params.atr_length.unwrap_or(10);
        if atr_length == 0 {
            return Err(KeltnerChannelWidthOscillatorError::InvalidAtrLength {
                atr_length,
                data_len: 0,
            });
        }

        let multiplier = params.multiplier.unwrap_or(2.0);
        if !multiplier.is_finite() || multiplier < 0.0 {
            return Err(KeltnerChannelWidthOscillatorError::InvalidMultiplier { multiplier });
        }

        let bands_style = parse_bands_style(
            params
                .bands_style
                .as_deref()
                .unwrap_or("Average True Range"),
        )?;

        Ok(Self {
            multiplier,
            center_use_exponential: params.use_exponential.unwrap_or(true),
            bands_style,
            center_sma: RollingSmaState::new(length),
            center_ema: SeededEmaState::new(length),
            true_range: TrueRangeState::new(),
            atr_rma: SeededRmaState::new(atr_length),
            range_rma: SeededRmaState::new(length),
            width_sma: RollingSmaState::new(length),
        })
    }

    #[inline]
    fn reset(&mut self) {
        self.center_sma.reset();
        self.center_ema.reset();
        self.true_range.reset();
        self.atr_rma.reset();
        self.range_rma.reset();
        self.width_sma.reset();
    }

    #[inline]
    pub fn update(&mut self, high: f64, low: f64, close: f64, source: f64) -> Option<(f64, f64)> {
        if !is_valid_bar(high, low, close, source) {
            return None;
        }

        let middle = if self.center_use_exponential {
            self.center_ema.update(source)
        } else {
            self.center_sma.update(source)
        };

        let range = match self.bands_style {
            KeltnerWidthBandsStyle::AverageTrueRange => {
                let tr = self.true_range.update(high, low, close);
                self.atr_rma.update(tr)
            }
            KeltnerWidthBandsStyle::TrueRange => Some(self.true_range.update(high, low, close)),
            KeltnerWidthBandsStyle::Range => self.range_rma.update(high - low),
        };

        let (middle, range) = match (middle, range) {
            (Some(middle), Some(range))
                if middle.is_finite() && range.is_finite() && middle != 0.0 =>
            {
                (middle, range)
            }
            (Some(_), Some(_)) => return Some((f64::NAN, f64::NAN)),
            _ => return None,
        };

        let kbw = (2.0 * self.multiplier * range) / middle;
        let kbw_sma = self.width_sma.update(kbw).unwrap_or(f64::NAN);
        Some((kbw, kbw_sma))
    }

    #[inline]
    pub fn update_reset_on_nan(
        &mut self,
        high: f64,
        low: f64,
        close: f64,
        source: f64,
    ) -> Option<(f64, f64)> {
        if !is_valid_bar(high, low, close, source) {
            self.reset();
            return None;
        }
        self.update(high, low, close, source)
    }
}

#[inline(always)]
fn keltner_channel_width_oscillator_prepare<'a>(
    input: &'a KeltnerChannelWidthOscillatorInput,
) -> Result<
    (
        &'a [f64],
        &'a [f64],
        &'a [f64],
        &'a [f64],
        usize,
        f64,
        bool,
        KeltnerWidthBandsStyle,
        usize,
        usize,
    ),
    KeltnerChannelWidthOscillatorError,
> {
    let (high, low, close, source) = input.as_refs();
    let data_len = source.len();
    if data_len == 0 {
        return Err(KeltnerChannelWidthOscillatorError::EmptyInputData);
    }
    if high.len() != data_len || low.len() != data_len || close.len() != data_len {
        return Err(KeltnerChannelWidthOscillatorError::DataLengthMismatch);
    }

    let length = input.get_length();
    if length == 0 || length > data_len {
        return Err(KeltnerChannelWidthOscillatorError::InvalidLength { length, data_len });
    }

    let multiplier = input.get_multiplier();
    if !multiplier.is_finite() || multiplier < 0.0 {
        return Err(KeltnerChannelWidthOscillatorError::InvalidMultiplier { multiplier });
    }

    let atr_length = input.get_atr_length();
    if atr_length == 0 {
        return Err(KeltnerChannelWidthOscillatorError::InvalidAtrLength {
            atr_length,
            data_len,
        });
    }

    let bands_style = parse_bands_style(input.bands_style_str())?;
    if bands_style == KeltnerWidthBandsStyle::AverageTrueRange && atr_length > data_len {
        return Err(KeltnerChannelWidthOscillatorError::InvalidAtrLength {
            atr_length,
            data_len,
        });
    }

    let first = first_valid_bar(high, low, close, source)
        .ok_or(KeltnerChannelWidthOscillatorError::AllValuesNaN)?;
    let valid = data_len - first;
    let needed = width_needed_bars(length, atr_length, bands_style);
    if valid < needed {
        return Err(KeltnerChannelWidthOscillatorError::NotEnoughValidData { needed, valid });
    }

    Ok((
        high,
        low,
        close,
        source,
        length,
        multiplier,
        input.get_use_exponential(),
        bands_style,
        atr_length,
        first,
    ))
}

#[inline(always)]
fn keltner_channel_width_oscillator_compute_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    length: usize,
    multiplier: f64,
    use_exponential: bool,
    bands_style: KeltnerWidthBandsStyle,
    atr_length: usize,
    out_kbw: &mut [f64],
    out_kbw_sma: &mut [f64],
) {
    if use_exponential
        && bands_style == KeltnerWidthBandsStyle::AverageTrueRange
        && length == 20
        && atr_length == 10
    {
        keltner_channel_width_oscillator_default_ema_atr_into(
            high,
            low,
            close,
            source,
            multiplier,
            out_kbw,
            out_kbw_sma,
        );
        return;
    }

    let mut stream = KeltnerChannelWidthOscillatorStream {
        multiplier,
        center_use_exponential: use_exponential,
        bands_style,
        center_sma: RollingSmaState::new(length),
        center_ema: SeededEmaState::new(length),
        true_range: TrueRangeState::new(),
        atr_rma: SeededRmaState::new(atr_length),
        range_rma: SeededRmaState::new(length),
        width_sma: RollingSmaState::new(length),
    };

    for i in 0..source.len() {
        if let Some((kbw, kbw_sma)) =
            stream.update_reset_on_nan(high[i], low[i], close[i], source[i])
        {
            out_kbw[i] = kbw;
            out_kbw_sma[i] = kbw_sma;
        }
    }
}

#[inline]
fn keltner_channel_width_oscillator_default_ema_atr_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    multiplier: f64,
    out_kbw: &mut [f64],
    out_kbw_sma: &mut [f64],
) {
    let len = source.len();
    let ema_alpha = 2.0f64 / 21.0;
    let rma_alpha = 0.1f64;
    let width_scale = 2.0 * multiplier;

    let mut ema_count = 0usize;
    let mut ema_sum = 0.0;
    let mut ema_value = f64::NAN;
    let mut ema_seeded = false;

    let mut atr_count = 0usize;
    let mut atr_sum = 0.0;
    let mut atr_value = f64::NAN;
    let mut atr_seeded = false;

    let mut width_buffer = [0.0f64; 20];
    let mut width_count = 0usize;
    let mut width_head = 0usize;
    let mut width_sum = 0.0;

    let mut prev_close = f64::NAN;
    let mut has_prev_close = false;

    let mut i = 0usize;
    while i < len {
        let high_value = high[i];
        let low_value = low[i];
        let close_value = close[i];
        let source_value = source[i];

        if !is_valid_bar(high_value, low_value, close_value, source_value) {
            ema_count = 0;
            ema_sum = 0.0;
            ema_value = f64::NAN;
            ema_seeded = false;
            atr_count = 0;
            atr_sum = 0.0;
            atr_value = f64::NAN;
            atr_seeded = false;
            width_count = 0;
            width_head = 0;
            width_sum = 0.0;
            width_buffer.fill(0.0);
            prev_close = f64::NAN;
            has_prev_close = false;
            i += 1;
            continue;
        }

        let middle = if !ema_seeded {
            ema_sum += source_value;
            ema_count += 1;
            if ema_count == 20 {
                ema_value = ema_sum / 20.0;
                ema_seeded = true;
                Some(ema_value)
            } else {
                None
            }
        } else {
            ema_value = ema_alpha.mul_add(source_value - ema_value, ema_value);
            Some(ema_value)
        };

        let tr = if has_prev_close {
            let up = if high_value > prev_close {
                high_value
            } else {
                prev_close
            };
            let dn = if low_value < prev_close {
                low_value
            } else {
                prev_close
            };
            up - dn
        } else {
            high_value - low_value
        };
        prev_close = close_value;
        has_prev_close = true;

        let range = if !atr_seeded {
            atr_sum += tr;
            atr_count += 1;
            if atr_count == 10 {
                atr_value = atr_sum / 10.0;
                atr_seeded = true;
                Some(atr_value)
            } else {
                None
            }
        } else {
            atr_value = rma_alpha.mul_add(tr - atr_value, atr_value);
            Some(atr_value)
        };

        if let (Some(middle), Some(range)) = (middle, range) {
            let kbw = if middle.is_finite() && range.is_finite() && middle != 0.0 {
                (width_scale * range) / middle
            } else {
                f64::NAN
            };
            out_kbw[i] = kbw;

            if width_count < 20 {
                width_buffer[width_count] = kbw;
                width_sum += kbw;
                width_count += 1;
                out_kbw_sma[i] = if width_count == 20 {
                    width_sum / 20.0
                } else {
                    f64::NAN
                };
            } else {
                let old = width_buffer[width_head];
                width_buffer[width_head] = kbw;
                width_sum += kbw - old;
                width_head += 1;
                if width_head == 20 {
                    width_head = 0;
                }
                out_kbw_sma[i] = width_sum / 20.0;
            }
        }

        i += 1;
    }
}

#[inline]
pub fn keltner_channel_width_oscillator(
    input: &KeltnerChannelWidthOscillatorInput,
) -> Result<KeltnerChannelWidthOscillatorOutput, KeltnerChannelWidthOscillatorError> {
    keltner_channel_width_oscillator_with_kernel(input, Kernel::Auto)
}

pub fn keltner_channel_width_oscillator_with_kernel(
    input: &KeltnerChannelWidthOscillatorInput,
    _kernel: Kernel,
) -> Result<KeltnerChannelWidthOscillatorOutput, KeltnerChannelWidthOscillatorError> {
    let (
        high,
        low,
        close,
        source,
        length,
        multiplier,
        use_exponential,
        bands_style,
        atr_length,
        first,
    ) = keltner_channel_width_oscillator_prepare(input)?;

    let kbw_prefix = kbw_warmup(length, atr_length, bands_style, first).min(source.len());
    let kbw_sma_prefix = kbw_sma_warmup(length, atr_length, bands_style, first).min(source.len());
    let mut kbw = alloc_with_nan_prefix(source.len(), kbw_prefix);
    let mut kbw_sma = alloc_with_nan_prefix(source.len(), kbw_sma_prefix);
    keltner_channel_width_oscillator_compute_into(
        high,
        low,
        close,
        source,
        length,
        multiplier,
        use_exponential,
        bands_style,
        atr_length,
        &mut kbw,
        &mut kbw_sma,
    );
    Ok(KeltnerChannelWidthOscillatorOutput { kbw, kbw_sma })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn keltner_channel_width_oscillator_into(
    input: &KeltnerChannelWidthOscillatorInput,
    out_kbw: &mut [f64],
    out_kbw_sma: &mut [f64],
) -> Result<(), KeltnerChannelWidthOscillatorError> {
    keltner_channel_width_oscillator_into_slice(out_kbw, out_kbw_sma, input, Kernel::Auto)
}

pub fn keltner_channel_width_oscillator_into_slice(
    out_kbw: &mut [f64],
    out_kbw_sma: &mut [f64],
    input: &KeltnerChannelWidthOscillatorInput,
    _kernel: Kernel,
) -> Result<(), KeltnerChannelWidthOscillatorError> {
    let (
        high,
        low,
        close,
        source,
        length,
        multiplier,
        use_exponential,
        bands_style,
        atr_length,
        _first,
    ) = keltner_channel_width_oscillator_prepare(input)?;

    if out_kbw.len() != source.len() || out_kbw_sma.len() != source.len() {
        return Err(KeltnerChannelWidthOscillatorError::OutputLengthMismatch {
            expected: source.len(),
            got: out_kbw.len().max(out_kbw_sma.len()),
        });
    }

    out_kbw.fill(f64::NAN);
    out_kbw_sma.fill(f64::NAN);
    keltner_channel_width_oscillator_compute_into(
        high,
        low,
        close,
        source,
        length,
        multiplier,
        use_exponential,
        bands_style,
        atr_length,
        out_kbw,
        out_kbw_sma,
    );
    Ok(())
}

#[derive(Clone, Debug)]
pub struct KeltnerChannelWidthOscillatorBatchRange {
    pub length: (usize, usize, usize),
    pub multiplier: (f64, f64, f64),
    pub atr_length: (usize, usize, usize),
    pub use_exponential: Option<bool>,
    pub bands_style: Option<String>,
}

impl Default for KeltnerChannelWidthOscillatorBatchRange {
    fn default() -> Self {
        Self {
            length: (20, 20, 0),
            multiplier: (2.0, 2.0, 0.0),
            atr_length: (10, 10, 0),
            use_exponential: Some(true),
            bands_style: Some("Average True Range".to_string()),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct KeltnerChannelWidthOscillatorBatchBuilder {
    range: KeltnerChannelWidthOscillatorBatchRange,
    kernel: Kernel,
}

impl KeltnerChannelWidthOscillatorBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline]
    pub fn length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.length = (start, end, step);
        self
    }

    #[inline]
    pub fn multiplier_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.multiplier = (start, end, step);
        self
    }

    #[inline]
    pub fn atr_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.atr_length = (start, end, step);
        self
    }

    #[inline]
    pub fn use_exponential(mut self, value: bool) -> Self {
        self.range.use_exponential = Some(value);
        self
    }

    #[inline]
    pub fn bands_style(mut self, value: &str) -> Result<Self, KeltnerChannelWidthOscillatorError> {
        self.range.bands_style = Some(normalize_bands_style(value)?);
        Ok(self)
    }

    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        source: &[f64],
    ) -> Result<KeltnerChannelWidthOscillatorBatchOutput, KeltnerChannelWidthOscillatorError> {
        keltner_channel_width_oscillator_batch_with_kernel(
            high,
            low,
            close,
            source,
            &self.range,
            self.kernel,
        )
    }

    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<KeltnerChannelWidthOscillatorBatchOutput, KeltnerChannelWidthOscillatorError> {
        self.apply_slices(
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
            source_type(candles, source),
        )
    }
}

#[derive(Clone, Debug)]
pub struct KeltnerChannelWidthOscillatorBatchOutput {
    pub kbw: Vec<f64>,
    pub kbw_sma: Vec<f64>,
    pub combos: Vec<KeltnerChannelWidthOscillatorParams>,
    pub rows: usize,
    pub cols: usize,
}

impl KeltnerChannelWidthOscillatorBatchOutput {
    pub fn row_for_params(&self, params: &KeltnerChannelWidthOscillatorParams) -> Option<usize> {
        let target_length = params.length.unwrap_or(20);
        let target_multiplier = params.multiplier.unwrap_or(2.0);
        let target_use_exponential = params.use_exponential.unwrap_or(true);
        let target_bands_style = normalize_bands_style(
            params
                .bands_style
                .as_deref()
                .unwrap_or("Average True Range"),
        )
        .ok()?;
        let target_atr_length = params.atr_length.unwrap_or(10);

        self.combos.iter().position(|combo| {
            combo.length.unwrap_or(20) == target_length
                && combo.multiplier.unwrap_or(2.0).to_bits() == target_multiplier.to_bits()
                && combo.use_exponential.unwrap_or(true) == target_use_exponential
                && combo.bands_style.as_deref().unwrap_or("Average True Range")
                    == target_bands_style
                && combo.atr_length.unwrap_or(10) == target_atr_length
        })
    }
}

fn axis_usize(
    range: (usize, usize, usize),
) -> Result<Vec<usize>, KeltnerChannelWidthOscillatorError> {
    let (start, end, step) = range;
    if step == 0 || start == end {
        return Ok(vec![start]);
    }

    let mut out = Vec::new();
    if start < end {
        let mut value = start;
        while value <= end {
            out.push(value);
            value = value.checked_add(step).ok_or(
                KeltnerChannelWidthOscillatorError::InvalidRange {
                    start: start.to_string(),
                    end: end.to_string(),
                    step: step.to_string(),
                },
            )?;
        }
    } else {
        let mut value = start;
        while value >= end {
            out.push(value);
            if value < end.saturating_add(step) {
                break;
            }
            value = value.saturating_sub(step);
            if value == 0 {
                break;
            }
        }
    }

    if out.is_empty() {
        return Err(KeltnerChannelWidthOscillatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

fn axis_f64(range: (f64, f64, f64)) -> Result<Vec<f64>, KeltnerChannelWidthOscillatorError> {
    let (start, end, step) = range;
    if !start.is_finite() || !end.is_finite() || !step.is_finite() || step < 0.0 {
        return Err(KeltnerChannelWidthOscillatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if step == 0.0 || start == end {
        return Ok(vec![start]);
    }

    let mut out = Vec::new();
    let mut value = start;
    let epsilon = step.abs() * 1e-9 + 1e-12;
    if start < end {
        while value <= end + epsilon {
            out.push(value);
            value += step;
        }
    } else {
        while value >= end - epsilon {
            out.push(value);
            value -= step;
        }
    }

    if out.is_empty() {
        return Err(KeltnerChannelWidthOscillatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

pub fn expand_grid_keltner_channel_width_oscillator(
    sweep: &KeltnerChannelWidthOscillatorBatchRange,
) -> Result<Vec<KeltnerChannelWidthOscillatorParams>, KeltnerChannelWidthOscillatorError> {
    let lengths = axis_usize(sweep.length)?;
    let multipliers = axis_f64(sweep.multiplier)?;
    let atr_lengths = axis_usize(sweep.atr_length)?;
    let use_exponential = sweep.use_exponential.unwrap_or(true);
    let bands_style =
        normalize_bands_style(sweep.bands_style.as_deref().unwrap_or("Average True Range"))?;

    let mut out = Vec::with_capacity(lengths.len() * multipliers.len() * atr_lengths.len());
    for &length in &lengths {
        for &multiplier in &multipliers {
            for &atr_length in &atr_lengths {
                out.push(KeltnerChannelWidthOscillatorParams {
                    length: Some(length),
                    multiplier: Some(multiplier),
                    use_exponential: Some(use_exponential),
                    bands_style: Some(bands_style.clone()),
                    atr_length: Some(atr_length),
                });
            }
        }
    }
    Ok(out)
}

pub fn keltner_channel_width_oscillator_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    sweep: &KeltnerChannelWidthOscillatorBatchRange,
    kernel: Kernel,
) -> Result<KeltnerChannelWidthOscillatorBatchOutput, KeltnerChannelWidthOscillatorError> {
    let batch_kernel = match kernel {
        Kernel::Auto => Kernel::ScalarBatch,
        other if other.is_batch() => other,
        other => {
            return Err(KeltnerChannelWidthOscillatorError::InvalidKernelForBatch(
                other,
            ))
        }
    };
    keltner_channel_width_oscillator_batch_impl(
        high,
        low,
        close,
        source,
        sweep,
        batch_kernel.to_non_batch(),
        true,
    )
}

pub fn keltner_channel_width_oscillator_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    sweep: &KeltnerChannelWidthOscillatorBatchRange,
) -> Result<KeltnerChannelWidthOscillatorBatchOutput, KeltnerChannelWidthOscillatorError> {
    keltner_channel_width_oscillator_batch_impl(
        high,
        low,
        close,
        source,
        sweep,
        Kernel::Scalar,
        false,
    )
}

pub fn keltner_channel_width_oscillator_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    sweep: &KeltnerChannelWidthOscillatorBatchRange,
) -> Result<KeltnerChannelWidthOscillatorBatchOutput, KeltnerChannelWidthOscillatorError> {
    keltner_channel_width_oscillator_batch_impl(
        high,
        low,
        close,
        source,
        sweep,
        Kernel::Scalar,
        true,
    )
}

fn keltner_channel_width_oscillator_batch_impl(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    sweep: &KeltnerChannelWidthOscillatorBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<KeltnerChannelWidthOscillatorBatchOutput, KeltnerChannelWidthOscillatorError> {
    let combos = expand_grid_keltner_channel_width_oscillator(sweep)?;
    let rows = combos.len();
    let cols = source.len();
    if cols == 0 {
        return Err(KeltnerChannelWidthOscillatorError::EmptyInputData);
    }
    if high.len() != cols || low.len() != cols || close.len() != cols {
        return Err(KeltnerChannelWidthOscillatorError::DataLengthMismatch);
    }

    let first = first_valid_bar(high, low, close, source)
        .ok_or(KeltnerChannelWidthOscillatorError::AllValuesNaN)?;

    for params in &combos {
        let bands_style = parse_bands_style(
            params
                .bands_style
                .as_deref()
                .unwrap_or("Average True Range"),
        )?;
        let needed = width_needed_bars(
            params.length.unwrap_or(20),
            params.atr_length.unwrap_or(10),
            bands_style,
        );
        let valid = cols - first;
        if valid < needed {
            return Err(KeltnerChannelWidthOscillatorError::NotEnoughValidData { needed, valid });
        }
    }

    let kbw_warmups: Vec<usize> = combos
        .iter()
        .map(|params| {
            kbw_warmup(
                params.length.unwrap_or(20),
                params.atr_length.unwrap_or(10),
                parse_bands_style(
                    params
                        .bands_style
                        .as_deref()
                        .unwrap_or("Average True Range"),
                )
                .unwrap_or(KeltnerWidthBandsStyle::AverageTrueRange),
                first,
            )
            .min(cols)
        })
        .collect();
    let kbw_sma_warmups: Vec<usize> = combos
        .iter()
        .map(|params| {
            kbw_sma_warmup(
                params.length.unwrap_or(20),
                params.atr_length.unwrap_or(10),
                parse_bands_style(
                    params
                        .bands_style
                        .as_deref()
                        .unwrap_or("Average True Range"),
                )
                .unwrap_or(KeltnerWidthBandsStyle::AverageTrueRange),
                first,
            )
            .min(cols)
        })
        .collect();

    let mut kbw_matrix = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut kbw_matrix, cols, &kbw_warmups);
    let mut kbw_sma_matrix = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut kbw_sma_matrix, cols, &kbw_sma_warmups);

    let mut kbw_guard = ManuallyDrop::new(kbw_matrix);
    let mut kbw_sma_guard = ManuallyDrop::new(kbw_sma_matrix);
    let kbw_mu: &mut [MaybeUninit<f64>] =
        unsafe { std::slice::from_raw_parts_mut(kbw_guard.as_mut_ptr(), kbw_guard.len()) };
    let kbw_sma_mu: &mut [MaybeUninit<f64>] =
        unsafe { std::slice::from_raw_parts_mut(kbw_sma_guard.as_mut_ptr(), kbw_sma_guard.len()) };

    let do_row = |row: usize,
                  row_kbw_mu: &mut [MaybeUninit<f64>],
                  row_kbw_sma_mu: &mut [MaybeUninit<f64>]| {
        let params = &combos[row];
        let bands_style = parse_bands_style(
            params
                .bands_style
                .as_deref()
                .unwrap_or("Average True Range"),
        )
        .unwrap_or(KeltnerWidthBandsStyle::AverageTrueRange);
        let dst_kbw = unsafe {
            std::slice::from_raw_parts_mut(row_kbw_mu.as_mut_ptr() as *mut f64, row_kbw_mu.len())
        };
        let dst_kbw_sma = unsafe {
            std::slice::from_raw_parts_mut(
                row_kbw_sma_mu.as_mut_ptr() as *mut f64,
                row_kbw_sma_mu.len(),
            )
        };
        keltner_channel_width_oscillator_compute_into(
            high,
            low,
            close,
            source,
            params.length.unwrap_or(20),
            params.multiplier.unwrap_or(2.0),
            params.use_exponential.unwrap_or(true),
            bands_style,
            params.atr_length.unwrap_or(10),
            dst_kbw,
            dst_kbw_sma,
        );
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        kbw_mu
            .par_chunks_mut(cols)
            .zip(kbw_sma_mu.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, (row_kbw, row_kbw_sma))| do_row(row, row_kbw, row_kbw_sma));
        #[cfg(target_arch = "wasm32")]
        for (row, (row_kbw, row_kbw_sma)) in kbw_mu
            .chunks_mut(cols)
            .zip(kbw_sma_mu.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, row_kbw, row_kbw_sma);
        }
    } else {
        for (row, (row_kbw, row_kbw_sma)) in kbw_mu
            .chunks_mut(cols)
            .zip(kbw_sma_mu.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, row_kbw, row_kbw_sma);
        }
    }

    let kbw = unsafe {
        Vec::from_raw_parts(
            kbw_guard.as_mut_ptr() as *mut f64,
            kbw_guard.len(),
            kbw_guard.capacity(),
        )
    };
    let kbw_sma = unsafe {
        Vec::from_raw_parts(
            kbw_sma_guard.as_mut_ptr() as *mut f64,
            kbw_sma_guard.len(),
            kbw_sma_guard.capacity(),
        )
    };

    Ok(KeltnerChannelWidthOscillatorBatchOutput {
        kbw,
        kbw_sma,
        combos,
        rows,
        cols,
    })
}

fn keltner_channel_width_oscillator_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    sweep: &KeltnerChannelWidthOscillatorBatchRange,
    _kernel: Kernel,
    parallel: bool,
    out_kbw: &mut [f64],
    out_kbw_sma: &mut [f64],
) -> Result<(), KeltnerChannelWidthOscillatorError> {
    let combos = expand_grid_keltner_channel_width_oscillator(sweep)?;
    let rows = combos.len();
    let cols = source.len();
    if out_kbw.len() != rows * cols || out_kbw_sma.len() != rows * cols {
        return Err(KeltnerChannelWidthOscillatorError::OutputLengthMismatch {
            expected: rows * cols,
            got: out_kbw.len().max(out_kbw_sma.len()),
        });
    }

    let first = first_valid_bar(high, low, close, source)
        .ok_or(KeltnerChannelWidthOscillatorError::AllValuesNaN)?;
    for (row, params) in combos.iter().enumerate() {
        let bands_style = parse_bands_style(
            params
                .bands_style
                .as_deref()
                .unwrap_or("Average True Range"),
        )?;
        let needed = width_needed_bars(
            params.length.unwrap_or(20),
            params.atr_length.unwrap_or(10),
            bands_style,
        );
        let valid = cols - first;
        if valid < needed {
            return Err(KeltnerChannelWidthOscillatorError::NotEnoughValidData { needed, valid });
        }
        let row_kbw = &mut out_kbw[row * cols..(row + 1) * cols];
        let row_kbw_sma = &mut out_kbw_sma[row * cols..(row + 1) * cols];
        row_kbw.fill(f64::NAN);
        row_kbw_sma.fill(f64::NAN);
    }

    let do_row = |row: usize, row_kbw: &mut [f64], row_kbw_sma: &mut [f64]| {
        let params = &combos[row];
        let bands_style = parse_bands_style(
            params
                .bands_style
                .as_deref()
                .unwrap_or("Average True Range"),
        )
        .unwrap_or(KeltnerWidthBandsStyle::AverageTrueRange);
        keltner_channel_width_oscillator_compute_into(
            high,
            low,
            close,
            source,
            params.length.unwrap_or(20),
            params.multiplier.unwrap_or(2.0),
            params.use_exponential.unwrap_or(true),
            bands_style,
            params.atr_length.unwrap_or(10),
            row_kbw,
            row_kbw_sma,
        );
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out_kbw
            .par_chunks_mut(cols)
            .zip(out_kbw_sma.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, (row_kbw, row_kbw_sma))| do_row(row, row_kbw, row_kbw_sma));
        #[cfg(target_arch = "wasm32")]
        for (row, (row_kbw, row_kbw_sma)) in out_kbw
            .chunks_mut(cols)
            .zip(out_kbw_sma.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, row_kbw, row_kbw_sma);
        }
    } else {
        for (row, (row_kbw, row_kbw_sma)) in out_kbw
            .chunks_mut(cols)
            .zip(out_kbw_sma.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, row_kbw, row_kbw_sma);
        }
    }

    Ok(())
}

#[cfg(feature = "python")]
#[pyfunction(name = "keltner_channel_width_oscillator")]
#[pyo3(signature = (high, low, close, source, length=20, multiplier=2.0, use_exponential=true, bands_style="Average True Range", atr_length=10, kernel=None))]
pub fn keltner_channel_width_oscillator_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    source: PyReadonlyArray1<'py, f64>,
    length: usize,
    multiplier: f64,
    use_exponential: bool,
    bands_style: &str,
    atr_length: usize,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let source = source.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = KeltnerChannelWidthOscillatorInput::from_slices(
        high,
        low,
        close,
        source,
        KeltnerChannelWidthOscillatorParams {
            length: Some(length),
            multiplier: Some(multiplier),
            use_exponential: Some(use_exponential),
            bands_style: Some(
                normalize_bands_style(bands_style)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            ),
            atr_length: Some(atr_length),
        },
    );
    let output = py
        .allow_threads(|| keltner_channel_width_oscillator_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((output.kbw.into_pyarray(py), output.kbw_sma.into_pyarray(py)))
}

#[cfg(feature = "python")]
#[pyclass(name = "KeltnerChannelWidthOscillatorStream")]
pub struct KeltnerChannelWidthOscillatorStreamPy {
    stream: KeltnerChannelWidthOscillatorStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl KeltnerChannelWidthOscillatorStreamPy {
    #[new]
    #[pyo3(signature = (length=20, multiplier=2.0, use_exponential=true, bands_style="Average True Range", atr_length=10))]
    fn new(
        length: usize,
        multiplier: f64,
        use_exponential: bool,
        bands_style: &str,
        atr_length: usize,
    ) -> PyResult<Self> {
        let stream =
            KeltnerChannelWidthOscillatorStream::try_new(KeltnerChannelWidthOscillatorParams {
                length: Some(length),
                multiplier: Some(multiplier),
                use_exponential: Some(use_exponential),
                bands_style: Some(
                    normalize_bands_style(bands_style)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?,
                ),
                atr_length: Some(atr_length),
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, high: f64, low: f64, close: f64, source: f64) -> Option<(f64, f64)> {
        self.stream.update_reset_on_nan(high, low, close, source)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "keltner_channel_width_oscillator_batch")]
#[pyo3(signature = (high, low, close, source, length_range, multiplier_range=(2.0, 2.0, 0.0), atr_length_range=(10, 10, 0), use_exponential=true, bands_style="Average True Range", kernel=None))]
pub fn keltner_channel_width_oscillator_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    source: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    multiplier_range: (f64, f64, f64),
    atr_length_range: (usize, usize, usize),
    use_exponential: bool,
    bands_style: &str,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let source = source.as_slice()?;
    let sweep = KeltnerChannelWidthOscillatorBatchRange {
        length: length_range,
        multiplier: multiplier_range,
        atr_length: atr_length_range,
        use_exponential: Some(use_exponential),
        bands_style: Some(
            normalize_bands_style(bands_style).map_err(|e| PyValueError::new_err(e.to_string()))?,
        ),
    };
    let combos = expand_grid_keltner_channel_width_oscillator(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = source.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;
    let kbw_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let kbw_sma_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_kbw = unsafe { kbw_arr.as_slice_mut()? };
    let out_kbw_sma = unsafe { kbw_sma_arr.as_slice_mut()? };
    let kernel = validate_kernel(kernel, true)?;

    py.allow_threads(|| {
        let batch_kernel = match kernel {
            Kernel::Auto => detect_best_batch_kernel(),
            other => other,
        };
        keltner_channel_width_oscillator_batch_inner_into(
            high,
            low,
            close,
            source,
            &sweep,
            batch_kernel.to_non_batch(),
            true,
            out_kbw,
            out_kbw_sma,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("kbw", kbw_arr.reshape((rows, cols))?)?;
    dict.set_item("kbw_sma", kbw_sma_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "lengths",
        combos
            .iter()
            .map(|params| params.length.unwrap_or(20) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "multipliers",
        combos
            .iter()
            .map(|params| params.multiplier.unwrap_or(2.0))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "use_exponential",
        combos
            .iter()
            .map(|params| params.use_exponential.unwrap_or(true))
            .collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "bands_styles",
        combos
            .iter()
            .map(|params| {
                params
                    .bands_style
                    .clone()
                    .unwrap_or_else(|| "Average True Range".to_string())
            })
            .collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "atr_lengths",
        combos
            .iter()
            .map(|params| params.atr_length.unwrap_or(10) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_keltner_channel_width_oscillator_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(keltner_channel_width_oscillator_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        keltner_channel_width_oscillator_batch_py,
        m
    )?)?;
    m.add_class::<KeltnerChannelWidthOscillatorStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct KeltnerChannelWidthOscillatorJsOutput {
    kbw: Vec<f64>,
    kbw_sma: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct KeltnerChannelWidthOscillatorBatchConfig {
    length_range: Vec<usize>,
    multiplier_range: Vec<f64>,
    atr_length_range: Vec<usize>,
    use_exponential: Option<bool>,
    bands_style: Option<String>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct KeltnerChannelWidthOscillatorBatchJsOutput {
    kbw: Vec<f64>,
    kbw_sma: Vec<f64>,
    rows: usize,
    cols: usize,
    combos: Vec<KeltnerChannelWidthOscillatorParams>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "keltner_channel_width_oscillator_js")]
pub fn keltner_channel_width_oscillator_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    length: usize,
    multiplier: f64,
    use_exponential: bool,
    bands_style: &str,
    atr_length: usize,
) -> Result<JsValue, JsValue> {
    let input = KeltnerChannelWidthOscillatorInput::from_slices(
        high,
        low,
        close,
        source,
        KeltnerChannelWidthOscillatorParams {
            length: Some(length),
            multiplier: Some(multiplier),
            use_exponential: Some(use_exponential),
            bands_style: Some(
                normalize_bands_style(bands_style)
                    .map_err(|e| JsValue::from_str(&e.to_string()))?,
            ),
            atr_length: Some(atr_length),
        },
    );
    let output =
        keltner_channel_width_oscillator(&input).map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&KeltnerChannelWidthOscillatorJsOutput {
        kbw: output.kbw,
        kbw_sma: output.kbw_sma,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "keltner_channel_width_oscillator_batch_js")]
pub fn keltner_channel_width_oscillator_batch_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: KeltnerChannelWidthOscillatorBatchConfig =
        serde_wasm_bindgen::from_value(config)
            .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.length_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: length_range must have exactly 3 elements [start, end, step]",
        ));
    }
    if config.multiplier_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: multiplier_range must have exactly 3 elements [start, end, step]",
        ));
    }
    if config.atr_length_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: atr_length_range must have exactly 3 elements [start, end, step]",
        ));
    }
    let sweep = KeltnerChannelWidthOscillatorBatchRange {
        length: (
            config.length_range[0],
            config.length_range[1],
            config.length_range[2],
        ),
        multiplier: (
            config.multiplier_range[0],
            config.multiplier_range[1],
            config.multiplier_range[2],
        ),
        atr_length: (
            config.atr_length_range[0],
            config.atr_length_range[1],
            config.atr_length_range[2],
        ),
        use_exponential: Some(config.use_exponential.unwrap_or(true)),
        bands_style: Some(
            normalize_bands_style(
                config
                    .bands_style
                    .as_deref()
                    .unwrap_or("Average True Range"),
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?,
        ),
    };
    let batch = keltner_channel_width_oscillator_batch_slice(high, low, close, source, &sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&KeltnerChannelWidthOscillatorBatchJsOutput {
        kbw: batch.kbw,
        kbw_sma: batch.kbw_sma,
        rows: batch.rows,
        cols: batch.cols,
        combos: batch.combos,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn keltner_channel_width_oscillator_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len * 2);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn keltner_channel_width_oscillator_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, 0, len * 2);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn keltner_channel_width_oscillator_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    source_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length: usize,
    multiplier: f64,
    use_exponential: bool,
    bands_style: String,
    atr_length: usize,
) -> Result<(), JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || source_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to keltner_channel_width_oscillator_into",
        ));
    }
    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let source = std::slice::from_raw_parts(source_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len * 2);
        let (out_kbw, out_kbw_sma) = out.split_at_mut(len);
        let input = KeltnerChannelWidthOscillatorInput::from_slices(
            high,
            low,
            close,
            source,
            KeltnerChannelWidthOscillatorParams {
                length: Some(length),
                multiplier: Some(multiplier),
                use_exponential: Some(use_exponential),
                bands_style: Some(bands_style),
                atr_length: Some(atr_length),
            },
        );
        keltner_channel_width_oscillator_into_slice(out_kbw, out_kbw_sma, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "keltner_channel_width_oscillator_into_host")]
pub fn keltner_channel_width_oscillator_into_host(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    out_ptr: *mut f64,
    length: usize,
    multiplier: f64,
    use_exponential: bool,
    bands_style: &str,
    atr_length: usize,
) -> Result<(), JsValue> {
    if out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to keltner_channel_width_oscillator_into_host",
        ));
    }
    unsafe {
        let out = std::slice::from_raw_parts_mut(out_ptr, source.len() * 2);
        let (out_kbw, out_kbw_sma) = out.split_at_mut(source.len());
        let input = KeltnerChannelWidthOscillatorInput::from_slices(
            high,
            low,
            close,
            source,
            KeltnerChannelWidthOscillatorParams {
                length: Some(length),
                multiplier: Some(multiplier),
                use_exponential: Some(use_exponential),
                bands_style: Some(
                    normalize_bands_style(bands_style)
                        .map_err(|e| JsValue::from_str(&e.to_string()))?,
                ),
                atr_length: Some(atr_length),
            },
        );
        keltner_channel_width_oscillator_into_slice(out_kbw, out_kbw_sma, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn keltner_channel_width_oscillator_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    source_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length_start: usize,
    length_end: usize,
    length_step: usize,
    multiplier_start: f64,
    multiplier_end: f64,
    multiplier_step: f64,
    atr_length_start: usize,
    atr_length_end: usize,
    atr_length_step: usize,
    use_exponential: bool,
    bands_style: String,
) -> Result<usize, JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || source_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to keltner_channel_width_oscillator_batch_into",
        ));
    }
    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let source = std::slice::from_raw_parts(source_ptr, len);
        let sweep = KeltnerChannelWidthOscillatorBatchRange {
            length: (length_start, length_end, length_step),
            multiplier: (multiplier_start, multiplier_end, multiplier_step),
            atr_length: (atr_length_start, atr_length_end, atr_length_step),
            use_exponential: Some(use_exponential),
            bands_style: Some(bands_style),
        };
        let combos = expand_grid_keltner_channel_width_oscillator(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let out = std::slice::from_raw_parts_mut(out_ptr, rows * len * 2);
        let (out_kbw, out_kbw_sma) = out.split_at_mut(rows * len);
        keltner_channel_width_oscillator_batch_inner_into(
            high,
            low,
            close,
            source,
            &sweep,
            Kernel::Scalar,
            false,
            out_kbw,
            out_kbw_sma,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn keltner_channel_width_oscillator_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    length: usize,
    multiplier: f64,
    use_exponential: bool,
    bands_style: &str,
    atr_length: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = keltner_channel_width_oscillator_js(
        high,
        low,
        close,
        source,
        length,
        multiplier,
        use_exponential,
        bands_style,
        atr_length,
    )?;
    crate::write_wasm_object_f64_outputs(
        "keltner_channel_width_oscillator_output_into_js",
        &value,
        out,
    )
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn keltner_channel_width_oscillator_batch_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = keltner_channel_width_oscillator_batch_js(high, low, close, source, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "keltner_channel_width_oscillator_batch_output_into_js",
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

    fn sample_ohlc(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut high = Vec::with_capacity(len);
        let mut low = Vec::with_capacity(len);
        let mut close = Vec::with_capacity(len);
        let mut source = Vec::with_capacity(len);
        for i in 0..len {
            let base = 100.0 + i as f64 * 0.11 + (i as f64 * 0.17).sin() * 1.8;
            let width = 1.2 + (i as f64 * 0.07).cos().abs();
            let lo = base - width;
            let hi = base + width;
            let cl = base + (i as f64 * 0.13).sin() * 0.35;
            high.push(hi);
            low.push(lo);
            close.push(cl);
            source.push((hi + lo + cl) / 3.0);
        }
        (high, low, close, source)
    }

    fn assert_close(lhs: &[f64], rhs: &[f64]) {
        assert_eq!(lhs.len(), rhs.len());
        for (i, (&l, &r)) in lhs.iter().zip(rhs.iter()).enumerate() {
            if l.is_nan() && r.is_nan() {
                continue;
            }
            let diff = (l - r).abs();
            assert!(
                diff <= 1e-10,
                "mismatch at {i}: lhs={l} rhs={r} diff={diff}"
            );
        }
    }

    #[test]
    fn keltner_channel_width_oscillator_output_contract() {
        let (high, low, close, source) = sample_ohlc(160);
        let input = KeltnerChannelWidthOscillatorInput::from_slices(
            &high,
            &low,
            &close,
            &source,
            KeltnerChannelWidthOscillatorParams::default(),
        );
        let out = keltner_channel_width_oscillator(&input).expect("indicator");
        assert_eq!(out.kbw.len(), source.len());
        assert_eq!(out.kbw_sma.len(), source.len());
        assert!(out.kbw[..19].iter().all(|x| x.is_nan()));
        assert!(out.kbw_sma[..38].iter().all(|x| x.is_nan()));
        assert!(out.kbw[19..].iter().any(|x| x.is_finite()));
        assert!(out.kbw_sma[38..].iter().any(|x| x.is_finite()));
    }

    #[test]
    fn keltner_channel_width_oscillator_into_matches_api() {
        let (high, low, close, source) = sample_ohlc(144);
        let input = KeltnerChannelWidthOscillatorInput::from_slices(
            &high,
            &low,
            &close,
            &source,
            KeltnerChannelWidthOscillatorParams::default(),
        );
        let baseline = keltner_channel_width_oscillator(&input).expect("baseline");
        let mut kbw = vec![0.0; source.len()];
        let mut kbw_sma = vec![0.0; source.len()];
        keltner_channel_width_oscillator_into(&input, &mut kbw, &mut kbw_sma).expect("into");
        assert_close(&kbw, &baseline.kbw);
        assert_close(&kbw_sma, &baseline.kbw_sma);
    }

    #[test]
    fn keltner_channel_width_oscillator_stream_matches_batch() {
        let (high, low, close, source) = sample_ohlc(192);
        let params = KeltnerChannelWidthOscillatorParams {
            length: Some(16),
            multiplier: Some(1.8),
            use_exponential: Some(false),
            bands_style: Some("Range".to_string()),
            atr_length: Some(10),
        };
        let input = KeltnerChannelWidthOscillatorInput::from_slices(
            &high,
            &low,
            &close,
            &source,
            params.clone(),
        );
        let batch = keltner_channel_width_oscillator(&input).expect("batch");
        let mut stream = KeltnerChannelWidthOscillatorStream::try_new(params).expect("stream");
        let mut kbw = Vec::with_capacity(source.len());
        let mut kbw_sma = Vec::with_capacity(source.len());
        for i in 0..source.len() {
            match stream.update_reset_on_nan(high[i], low[i], close[i], source[i]) {
                Some((w, s)) => {
                    kbw.push(w);
                    kbw_sma.push(s);
                }
                None => {
                    kbw.push(f64::NAN);
                    kbw_sma.push(f64::NAN);
                }
            }
        }
        assert_close(&kbw, &batch.kbw);
        assert_close(&kbw_sma, &batch.kbw_sma);
    }

    #[test]
    fn keltner_channel_width_oscillator_batch_single_param_matches_single() {
        let (high, low, close, source) = sample_ohlc(176);
        let sweep = KeltnerChannelWidthOscillatorBatchRange::default();
        let batch = keltner_channel_width_oscillator_batch_with_kernel(
            &high,
            &low,
            &close,
            &source,
            &sweep,
            Kernel::Auto,
        )
        .expect("batch");
        let input = KeltnerChannelWidthOscillatorInput::from_slices(
            &high,
            &low,
            &close,
            &source,
            KeltnerChannelWidthOscillatorParams::default(),
        );
        let direct = keltner_channel_width_oscillator(&input).expect("direct");
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, source.len());
        assert_close(&batch.kbw[..source.len()], &direct.kbw);
        assert_close(&batch.kbw_sma[..source.len()], &direct.kbw_sma);
    }

    #[test]
    fn keltner_channel_width_oscillator_rejects_invalid_bands_style() {
        let (high, low, close, source) = sample_ohlc(64);
        let input = KeltnerChannelWidthOscillatorInput::from_slices(
            &high,
            &low,
            &close,
            &source,
            KeltnerChannelWidthOscillatorParams {
                bands_style: Some("bogus".to_string()),
                ..KeltnerChannelWidthOscillatorParams::default()
            },
        );
        let err = keltner_channel_width_oscillator(&input).unwrap_err();
        assert!(matches!(
            err,
            KeltnerChannelWidthOscillatorError::InvalidBandsStyle { .. }
        ));
    }

    #[test]
    fn keltner_channel_width_oscillator_dispatch_matches_direct() {
        let (high, low, _close, source) = sample_ohlc(160);
        let combo = [
            ParamKV {
                key: "length",
                value: ParamValue::Int(20),
            },
            ParamKV {
                key: "multiplier",
                value: ParamValue::Float(2.0),
            },
            ParamKV {
                key: "use_exponential",
                value: ParamValue::Bool(true),
            },
            ParamKV {
                key: "bands_style",
                value: ParamValue::EnumString("Average True Range"),
            },
            ParamKV {
                key: "atr_length",
                value: ParamValue::Int(10),
            },
        ];
        let combos = [IndicatorParamSet { params: &combo }];

        let req_kbw = IndicatorBatchRequest {
            indicator_id: "keltner_channel_width_oscillator",
            output_id: Some("kbw"),
            data: IndicatorDataRef::Ohlc {
                open: &source,
                high: &high,
                low: &low,
                close: &source,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out_kbw = compute_cpu_batch(req_kbw).expect("dispatch kbw");

        let req_kbw_sma = IndicatorBatchRequest {
            indicator_id: "keltner_channel_width_oscillator",
            output_id: Some("kbw_sma"),
            data: IndicatorDataRef::Ohlc {
                open: &source,
                high: &high,
                low: &low,
                close: &source,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out_kbw_sma = compute_cpu_batch(req_kbw_sma).expect("dispatch kbw_sma");

        let input = KeltnerChannelWidthOscillatorInput::from_slices(
            &high,
            &low,
            &source,
            &source,
            KeltnerChannelWidthOscillatorParams::default(),
        );
        let direct = keltner_channel_width_oscillator(&input).expect("direct");
        assert_eq!(out_kbw.rows, 1);
        assert_eq!(out_kbw.cols, source.len());
        assert_eq!(out_kbw_sma.rows, 1);
        assert_eq!(out_kbw_sma.cols, source.len());
        assert_close(&out_kbw.values_f64.expect("values"), &direct.kbw);
        assert_close(&out_kbw_sma.values_f64.expect("values"), &direct.kbw_sma);
    }
}
