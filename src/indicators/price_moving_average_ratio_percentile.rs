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

use crate::indicators::moving_averages::ema::{ema_into_slice, EmaInput, EmaParams};
use crate::indicators::moving_averages::hma::{hma_into_slice, HmaInput, HmaParams};
use crate::indicators::moving_averages::sma::{sma_into_slice, SmaInput, SmaParams};
use crate::indicators::moving_averages::vwma::{vwma_into_slice, VwmaInput, VwmaParams};
use crate::indicators::moving_averages::wilders::{
    wilders_into_slice, WildersInput, WildersParams,
};
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{alloc_uninit_f64, detect_best_batch_kernel};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use thiserror::Error;

const DEFAULT_SOURCE: &str = "close";
const DEFAULT_MA_LENGTH: usize = 20;
const DEFAULT_PMARP_LOOKBACK: usize = 350;
const DEFAULT_SIGNAL_MA_LENGTH: usize = 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub enum PriceMovingAverageRatioPercentileMaType {
    Sma,
    Ema,
    Hma,
    Rma,
    Vwma,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub enum PriceMovingAverageRatioPercentileLineMode {
    Pmar,
    Pmarp,
}

#[derive(Debug, Clone)]
pub enum PriceMovingAverageRatioPercentileData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slices {
        price: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct PriceMovingAverageRatioPercentileOutput {
    pub pmar: Vec<f64>,
    pub pmarp: Vec<f64>,
    pub plotline: Vec<f64>,
    pub signal: Vec<f64>,
    pub pmar_high: Vec<f64>,
    pub pmar_low: Vec<f64>,
    pub scaled_pmar: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct PriceMovingAverageRatioPercentileParams {
    pub ma_length: Option<usize>,
    pub ma_type: Option<PriceMovingAverageRatioPercentileMaType>,
    pub pmarp_lookback: Option<usize>,
    pub signal_ma_length: Option<usize>,
    pub signal_ma_type: Option<PriceMovingAverageRatioPercentileMaType>,
    pub line_mode: Option<PriceMovingAverageRatioPercentileLineMode>,
}

#[derive(Debug, Clone)]
pub struct PriceMovingAverageRatioPercentileInput<'a> {
    pub data: PriceMovingAverageRatioPercentileData<'a>,
    pub params: PriceMovingAverageRatioPercentileParams,
}

#[derive(Debug, Error)]
pub enum PriceMovingAverageRatioPercentileError {
    #[error("price_moving_average_ratio_percentile: Input data slice is empty.")]
    EmptyInputData,
    #[error("price_moving_average_ratio_percentile: All values are NaN.")]
    AllValuesNaN,
    #[error("price_moving_average_ratio_percentile: Inconsistent slice lengths: price={price_len}, volume={volume_len}")]
    InconsistentSliceLengths { price_len: usize, volume_len: usize },
    #[error("price_moving_average_ratio_percentile: Invalid period `{name}`: {value}")]
    InvalidPeriod { name: String, value: usize },
    #[error("price_moving_average_ratio_percentile: Not enough valid data: needed={needed}, valid={valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("price_moving_average_ratio_percentile: Output length mismatch: expected={expected}, got={got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("price_moving_average_ratio_percentile: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("price_moving_average_ratio_percentile: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("price_moving_average_ratio_percentile: {context} failed: {details}")]
    MovingAverageComputation { context: String, details: String },
}

impl Default for PriceMovingAverageRatioPercentileMaType {
    fn default() -> Self {
        Self::Vwma
    }
}

impl PriceMovingAverageRatioPercentileMaType {
    #[inline(always)]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sma => "sma",
            Self::Ema => "ema",
            Self::Hma => "hma",
            Self::Rma => "rma",
            Self::Vwma => "vwma",
        }
    }
}

impl std::str::FromStr for PriceMovingAverageRatioPercentileMaType {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "sma" => Ok(Self::Sma),
            "ema" => Ok(Self::Ema),
            "hma" => Ok(Self::Hma),
            "rma" => Ok(Self::Rma),
            "vwma" => Ok(Self::Vwma),
            other => Err(format!("Unknown MA type: {other}")),
        }
    }
}

impl Default for PriceMovingAverageRatioPercentileLineMode {
    fn default() -> Self {
        Self::Pmarp
    }
}

impl PriceMovingAverageRatioPercentileLineMode {
    #[inline(always)]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pmar => "pmar",
            Self::Pmarp => "pmarp",
        }
    }
}

impl std::str::FromStr for PriceMovingAverageRatioPercentileLineMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "pmar" | "price moving average ratio" => Ok(Self::Pmar),
            "pmarp" | "price moving average ratio percentile" => Ok(Self::Pmarp),
            other => Err(format!("Unknown line mode: {other}")),
        }
    }
}

impl Default for PriceMovingAverageRatioPercentileParams {
    fn default() -> Self {
        Self {
            ma_length: Some(DEFAULT_MA_LENGTH),
            ma_type: Some(PriceMovingAverageRatioPercentileMaType::Vwma),
            pmarp_lookback: Some(DEFAULT_PMARP_LOOKBACK),
            signal_ma_length: Some(DEFAULT_SIGNAL_MA_LENGTH),
            signal_ma_type: Some(PriceMovingAverageRatioPercentileMaType::Sma),
            line_mode: Some(PriceMovingAverageRatioPercentileLineMode::Pmarp),
        }
    }
}

impl<'a> PriceMovingAverageRatioPercentileInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: PriceMovingAverageRatioPercentileParams,
    ) -> Self {
        Self {
            data: PriceMovingAverageRatioPercentileData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        price: &'a [f64],
        volume: &'a [f64],
        params: PriceMovingAverageRatioPercentileParams,
    ) -> Self {
        Self {
            data: PriceMovingAverageRatioPercentileData::Slices { price, volume },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(
            candles,
            DEFAULT_SOURCE,
            PriceMovingAverageRatioPercentileParams::default(),
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct ValidatedParams {
    ma_length: usize,
    ma_type: PriceMovingAverageRatioPercentileMaType,
    pmarp_lookback: usize,
    signal_ma_length: usize,
    signal_ma_type: PriceMovingAverageRatioPercentileMaType,
    line_mode: PriceMovingAverageRatioPercentileLineMode,
}

impl ValidatedParams {
    fn from_params(
        params: &PriceMovingAverageRatioPercentileParams,
    ) -> Result<Self, PriceMovingAverageRatioPercentileError> {
        let out = Self {
            ma_length: params.ma_length.unwrap_or(DEFAULT_MA_LENGTH),
            ma_type: params.ma_type.unwrap_or_default(),
            pmarp_lookback: params.pmarp_lookback.unwrap_or(DEFAULT_PMARP_LOOKBACK),
            signal_ma_length: params.signal_ma_length.unwrap_or(DEFAULT_SIGNAL_MA_LENGTH),
            signal_ma_type: params
                .signal_ma_type
                .unwrap_or(PriceMovingAverageRatioPercentileMaType::Sma),
            line_mode: params.line_mode.unwrap_or_default(),
        };
        for (name, value) in [
            ("ma_length", out.ma_length),
            ("pmarp_lookback", out.pmarp_lookback),
            ("signal_ma_length", out.signal_ma_length),
        ] {
            if value == 0 {
                return Err(PriceMovingAverageRatioPercentileError::InvalidPeriod {
                    name: name.to_string(),
                    value,
                });
            }
        }
        Ok(out)
    }

    fn into_params(self) -> PriceMovingAverageRatioPercentileParams {
        PriceMovingAverageRatioPercentileParams {
            ma_length: Some(self.ma_length),
            ma_type: Some(self.ma_type),
            pmarp_lookback: Some(self.pmarp_lookback),
            signal_ma_length: Some(self.signal_ma_length),
            signal_ma_type: Some(self.signal_ma_type),
            line_mode: Some(self.line_mode),
        }
    }
}

#[derive(Clone, Debug)]
pub struct PriceMovingAverageRatioPercentileBuilder {
    source: Option<String>,
    ma_length: Option<usize>,
    ma_type: Option<PriceMovingAverageRatioPercentileMaType>,
    pmarp_lookback: Option<usize>,
    signal_ma_length: Option<usize>,
    signal_ma_type: Option<PriceMovingAverageRatioPercentileMaType>,
    line_mode: Option<PriceMovingAverageRatioPercentileLineMode>,
    kernel: Kernel,
}

impl Default for PriceMovingAverageRatioPercentileBuilder {
    fn default() -> Self {
        Self {
            source: None,
            ma_length: None,
            ma_type: None,
            pmarp_lookback: None,
            signal_ma_length: None,
            signal_ma_type: None,
            line_mode: None,
            kernel: Kernel::Auto,
        }
    }
}

impl PriceMovingAverageRatioPercentileBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn source(mut self, value: impl Into<String>) -> Self {
        self.source = Some(value.into());
        self
    }

    #[inline(always)]
    pub fn ma_length(mut self, value: usize) -> Self {
        self.ma_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn ma_type(mut self, value: PriceMovingAverageRatioPercentileMaType) -> Self {
        self.ma_type = Some(value);
        self
    }

    #[inline(always)]
    pub fn pmarp_lookback(mut self, value: usize) -> Self {
        self.pmarp_lookback = Some(value);
        self
    }

    #[inline(always)]
    pub fn signal_ma_length(mut self, value: usize) -> Self {
        self.signal_ma_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn signal_ma_type(mut self, value: PriceMovingAverageRatioPercentileMaType) -> Self {
        self.signal_ma_type = Some(value);
        self
    }

    #[inline(always)]
    pub fn line_mode(mut self, value: PriceMovingAverageRatioPercentileLineMode) -> Self {
        self.line_mode = Some(value);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    fn params(&self) -> PriceMovingAverageRatioPercentileParams {
        PriceMovingAverageRatioPercentileParams {
            ma_length: self.ma_length,
            ma_type: self.ma_type,
            pmarp_lookback: self.pmarp_lookback,
            signal_ma_length: self.signal_ma_length,
            signal_ma_type: self.signal_ma_type,
            line_mode: self.line_mode,
        }
    }

    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<PriceMovingAverageRatioPercentileOutput, PriceMovingAverageRatioPercentileError>
    {
        let input = PriceMovingAverageRatioPercentileInput::from_candles(
            candles,
            self.source.as_deref().unwrap_or(DEFAULT_SOURCE),
            self.params(),
        );
        price_moving_average_ratio_percentile_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        price: &[f64],
        volume: &[f64],
    ) -> Result<PriceMovingAverageRatioPercentileOutput, PriceMovingAverageRatioPercentileError>
    {
        let input =
            PriceMovingAverageRatioPercentileInput::from_slices(price, volume, self.params());
        price_moving_average_ratio_percentile_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<PriceMovingAverageRatioPercentileStream, PriceMovingAverageRatioPercentileError>
    {
        PriceMovingAverageRatioPercentileStream::try_new(self.params())
    }
}

#[inline(always)]
fn extract_price_volume<'a>(
    input: &'a PriceMovingAverageRatioPercentileInput<'a>,
) -> Result<(&'a [f64], &'a [f64]), PriceMovingAverageRatioPercentileError> {
    let (price, volume) = match &input.data {
        PriceMovingAverageRatioPercentileData::Candles { candles, source } => {
            (source_type(candles, source), candles.volume.as_slice())
        }
        PriceMovingAverageRatioPercentileData::Slices { price, volume } => (*price, *volume),
    };

    if price.is_empty() || volume.is_empty() {
        return Err(PriceMovingAverageRatioPercentileError::EmptyInputData);
    }
    if price.len() != volume.len() {
        return Err(
            PriceMovingAverageRatioPercentileError::InconsistentSliceLengths {
                price_len: price.len(),
                volume_len: volume.len(),
            },
        );
    }
    if !price
        .iter()
        .zip(volume.iter())
        .any(|(&p, &v)| p.is_finite() && v.is_finite())
    {
        return Err(PriceMovingAverageRatioPercentileError::AllValuesNaN);
    }
    Ok((price, volume))
}

#[inline(always)]
fn count_valid_price_volume(price: &[f64], volume: &[f64]) -> usize {
    price
        .iter()
        .zip(volume.iter())
        .filter(|(p, v)| p.is_finite() && v.is_finite())
        .count()
}

fn compute_ma_series(
    price: &[f64],
    volume: &[f64],
    period: usize,
    ma_type: PriceMovingAverageRatioPercentileMaType,
    kernel: Kernel,
    context: &str,
) -> Result<Vec<f64>, PriceMovingAverageRatioPercentileError> {
    let mut out = vec![f64::NAN; price.len()];
    if ma_type == PriceMovingAverageRatioPercentileMaType::Hma && period == 1 {
        out.copy_from_slice(price);
        return Ok(out);
    }

    match ma_type {
        PriceMovingAverageRatioPercentileMaType::Sma => sma_into_slice(
            &mut out,
            &SmaInput::from_slice(
                price,
                SmaParams {
                    period: Some(period),
                },
            ),
            kernel,
        )
        .map_err(
            |e| PriceMovingAverageRatioPercentileError::MovingAverageComputation {
                context: context.to_string(),
                details: e.to_string(),
            },
        )?,
        PriceMovingAverageRatioPercentileMaType::Ema => ema_into_slice(
            &mut out,
            &EmaInput::from_slice(
                price,
                EmaParams {
                    period: Some(period),
                },
            ),
            kernel,
        )
        .map_err(
            |e| PriceMovingAverageRatioPercentileError::MovingAverageComputation {
                context: context.to_string(),
                details: e.to_string(),
            },
        )?,
        PriceMovingAverageRatioPercentileMaType::Hma => hma_into_slice(
            &mut out,
            &HmaInput::from_slice(
                price,
                HmaParams {
                    period: Some(period),
                },
            ),
            kernel,
        )
        .map_err(
            |e| PriceMovingAverageRatioPercentileError::MovingAverageComputation {
                context: context.to_string(),
                details: e.to_string(),
            },
        )?,
        PriceMovingAverageRatioPercentileMaType::Rma => wilders_into_slice(
            &mut out,
            &WildersInput::from_slice(
                price,
                WildersParams {
                    period: Some(period),
                },
            ),
            kernel,
        )
        .map_err(
            |e| PriceMovingAverageRatioPercentileError::MovingAverageComputation {
                context: context.to_string(),
                details: e.to_string(),
            },
        )?,
        PriceMovingAverageRatioPercentileMaType::Vwma => vwma_into_slice(
            &mut out,
            &VwmaInput::from_slice(
                price,
                volume,
                VwmaParams {
                    period: Some(period),
                },
            ),
            kernel,
        )
        .map_err(
            |e| PriceMovingAverageRatioPercentileError::MovingAverageComputation {
                context: context.to_string(),
                details: e.to_string(),
            },
        )?,
    };

    Ok(out)
}

#[inline(always)]
fn scaled_pmar_value(pmar: f64, pmar_high: f64, pmar_low: f64) -> f64 {
    if pmar >= 1.0 {
        let denom = pmar_high - 1.0;
        if denom.abs() <= 1e-12 {
            50.0
        } else {
            (((pmar - 1.0) * (100.0 / denom)) / 2.0) + 50.0
        }
    } else {
        let denom = 1.0 - pmar_low;
        if denom.abs() <= 1e-12 {
            50.0
        } else {
            ((pmar - pmar_low) * (100.0 / denom)) / 2.0
        }
    }
}

#[inline(always)]
fn insert_pmar_window(sorted: &mut Vec<f64>, invalid_count: &mut usize, value: f64) {
    let value = value.abs();
    if value.is_finite() {
        let pos = sorted.partition_point(|probe| probe.total_cmp(&value).is_lt());
        sorted.insert(pos, value);
    } else {
        *invalid_count += 1;
    }
}

#[inline(always)]
fn remove_pmar_window(sorted: &mut Vec<f64>, invalid_count: &mut usize, value: f64) {
    let value = value.abs();
    if value.is_finite() {
        let pos = sorted
            .binary_search_by(|probe| probe.total_cmp(&value))
            .expect("pmar percentile window lost a finite value");
        sorted.remove(pos);
    } else {
        *invalid_count -= 1;
    }
}

fn compute_pmarp_percentile(
    pmar_out: &[f64],
    ma_length: usize,
    lookback_limit: usize,
    pmarp_out: &mut [f64],
) {
    let mut sorted = Vec::with_capacity(lookback_limit.min(pmar_out.len()));
    let mut invalid_count = 0usize;

    for i in 0..pmar_out.len() {
        if i >= ma_length {
            let current = pmar_out[i].abs();
            let lookback = i.min(lookback_limit);
            if current.is_finite() && lookback != 0 {
                let le_count = sorted.partition_point(|value| *value <= current);
                pmarp_out[i] = ((le_count + invalid_count) as f64 / lookback as f64) * 100.0;
            }
        }

        if i >= lookback_limit {
            remove_pmar_window(
                &mut sorted,
                &mut invalid_count,
                pmar_out[i - lookback_limit],
            );
        }
        insert_pmar_window(&mut sorted, &mut invalid_count, pmar_out[i]);
    }
}

#[allow(clippy::too_many_arguments)]
fn compute_core(
    price: &[f64],
    volume: &[f64],
    params: ValidatedParams,
    kernel: Kernel,
    pmar_out: &mut [f64],
    pmarp_out: &mut [f64],
    plotline_out: &mut [f64],
    signal_out: &mut [f64],
    pmar_high_out: &mut [f64],
    pmar_low_out: &mut [f64],
    scaled_pmar_out: &mut [f64],
) -> Result<(), PriceMovingAverageRatioPercentileError> {
    pmar_out.fill(f64::NAN);
    pmarp_out.fill(f64::NAN);
    plotline_out.fill(f64::NAN);
    signal_out.fill(f64::NAN);
    pmar_high_out.fill(f64::NAN);
    pmar_low_out.fill(f64::NAN);
    scaled_pmar_out.fill(f64::NAN);

    let ma = compute_ma_series(
        price,
        volume,
        params.ma_length,
        params.ma_type,
        kernel,
        "pmar ma",
    )?;

    let mut seen_pmar = false;
    let mut pmar_high: f64 = 1.0;
    let mut pmar_low: f64 = 1.0;
    for i in 0..price.len() {
        let m = ma[i];
        let p = price[i];
        if p.is_finite() && m.is_finite() && m != 0.0 {
            let pmar = p / m;
            pmar_out[i] = pmar;
            pmar_high = pmar_high.max(pmar);
            pmar_low = pmar_low.min(pmar);
            seen_pmar = true;
        }
        if seen_pmar {
            pmar_high_out[i] = pmar_high;
            pmar_low_out[i] = pmar_low;
            if pmar_out[i].is_finite() {
                scaled_pmar_out[i] = scaled_pmar_value(pmar_out[i], pmar_high, pmar_low);
            }
        }
    }

    if !seen_pmar {
        return Err(PriceMovingAverageRatioPercentileError::NotEnoughValidData {
            needed: params.ma_length,
            valid: count_valid_price_volume(price, volume),
        });
    }

    compute_pmarp_percentile(pmar_out, params.ma_length, params.pmarp_lookback, pmarp_out);

    let signal_source = match params.line_mode {
        PriceMovingAverageRatioPercentileLineMode::Pmar => &*pmar_out,
        PriceMovingAverageRatioPercentileLineMode::Pmarp => &*pmarp_out,
    };
    let signal = compute_ma_series(
        signal_source,
        volume,
        params.signal_ma_length,
        params.signal_ma_type,
        kernel,
        "signal ma",
    )?;
    signal_out.copy_from_slice(&signal);

    match params.line_mode {
        PriceMovingAverageRatioPercentileLineMode::Pmar => plotline_out.copy_from_slice(pmar_out),
        PriceMovingAverageRatioPercentileLineMode::Pmarp => plotline_out.copy_from_slice(pmarp_out),
    }
    Ok(())
}

#[inline]
pub fn price_moving_average_ratio_percentile(
    input: &PriceMovingAverageRatioPercentileInput,
) -> Result<PriceMovingAverageRatioPercentileOutput, PriceMovingAverageRatioPercentileError> {
    price_moving_average_ratio_percentile_with_kernel(input, Kernel::Auto)
}

pub fn price_moving_average_ratio_percentile_with_kernel(
    input: &PriceMovingAverageRatioPercentileInput,
    kernel: Kernel,
) -> Result<PriceMovingAverageRatioPercentileOutput, PriceMovingAverageRatioPercentileError> {
    let (price, volume) = extract_price_volume(input)?;
    let params = ValidatedParams::from_params(&input.params)?;
    let kernel = kernel.to_non_batch();

    let mut pmar = alloc_uninit_f64(price.len());
    let mut pmarp = alloc_uninit_f64(price.len());
    let mut plotline = alloc_uninit_f64(price.len());
    let mut signal = alloc_uninit_f64(price.len());
    let mut pmar_high = alloc_uninit_f64(price.len());
    let mut pmar_low = alloc_uninit_f64(price.len());
    let mut scaled_pmar = alloc_uninit_f64(price.len());

    compute_core(
        price,
        volume,
        params,
        kernel,
        &mut pmar,
        &mut pmarp,
        &mut plotline,
        &mut signal,
        &mut pmar_high,
        &mut pmar_low,
        &mut scaled_pmar,
    )?;

    Ok(PriceMovingAverageRatioPercentileOutput {
        pmar,
        pmarp,
        plotline,
        signal,
        pmar_high,
        pmar_low,
        scaled_pmar,
    })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[allow(clippy::too_many_arguments)]
pub fn price_moving_average_ratio_percentile_into(
    input: &PriceMovingAverageRatioPercentileInput,
    pmar: &mut [f64],
    pmarp: &mut [f64],
    plotline: &mut [f64],
    signal: &mut [f64],
    pmar_high: &mut [f64],
    pmar_low: &mut [f64],
    scaled_pmar: &mut [f64],
) -> Result<(), PriceMovingAverageRatioPercentileError> {
    price_moving_average_ratio_percentile_into_slice(
        pmar,
        pmarp,
        plotline,
        signal,
        pmar_high,
        pmar_low,
        scaled_pmar,
        input,
        Kernel::Auto,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn price_moving_average_ratio_percentile_into_slice(
    pmar: &mut [f64],
    pmarp: &mut [f64],
    plotline: &mut [f64],
    signal: &mut [f64],
    pmar_high: &mut [f64],
    pmar_low: &mut [f64],
    scaled_pmar: &mut [f64],
    input: &PriceMovingAverageRatioPercentileInput,
    kernel: Kernel,
) -> Result<(), PriceMovingAverageRatioPercentileError> {
    let (price, volume) = extract_price_volume(input)?;
    let expected = price.len();
    for dst in [
        pmar as &[f64],
        pmarp as &[f64],
        plotline as &[f64],
        signal as &[f64],
        pmar_high as &[f64],
        pmar_low as &[f64],
        scaled_pmar as &[f64],
    ] {
        if dst.len() != expected {
            return Err(
                PriceMovingAverageRatioPercentileError::OutputLengthMismatch {
                    expected,
                    got: dst.len(),
                },
            );
        }
    }
    compute_core(
        price,
        volume,
        ValidatedParams::from_params(&input.params)?,
        kernel.to_non_batch(),
        pmar,
        pmarp,
        plotline,
        signal,
        pmar_high,
        pmar_low,
        scaled_pmar,
    )
}

#[derive(Debug, Clone)]
pub struct PriceMovingAverageRatioPercentileStream {
    params: PriceMovingAverageRatioPercentileParams,
    price_history: Vec<f64>,
    volume_history: Vec<f64>,
}

impl PriceMovingAverageRatioPercentileStream {
    pub fn try_new(
        params: PriceMovingAverageRatioPercentileParams,
    ) -> Result<Self, PriceMovingAverageRatioPercentileError> {
        let _ = ValidatedParams::from_params(&params)?;
        Ok(Self {
            params,
            price_history: Vec::new(),
            volume_history: Vec::new(),
        })
    }

    #[allow(clippy::type_complexity)]
    pub fn update(&mut self, price: f64, volume: f64) -> (f64, f64, f64, f64, f64, f64, f64) {
        self.price_history.push(price);
        self.volume_history.push(volume);
        let input = PriceMovingAverageRatioPercentileInput::from_slices(
            &self.price_history,
            &self.volume_history,
            self.params.clone(),
        );
        match price_moving_average_ratio_percentile(&input) {
            Ok(out) => {
                let i = self.price_history.len() - 1;
                (
                    out.pmar[i],
                    out.pmarp[i],
                    out.plotline[i],
                    out.signal[i],
                    out.pmar_high[i],
                    out.pmar_low[i],
                    out.scaled_pmar[i],
                )
            }
            Err(_) => (
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
            ),
        }
    }

    #[inline(always)]
    pub fn get_warmup_period(&self) -> usize {
        self.params.ma_length.unwrap_or(DEFAULT_MA_LENGTH).max(
            self.params
                .signal_ma_length
                .unwrap_or(DEFAULT_SIGNAL_MA_LENGTH),
        )
    }
}

#[derive(Clone, Debug)]
pub struct PriceMovingAverageRatioPercentileBatchRange {
    pub ma_length: (usize, usize, usize),
    pub pmarp_lookback: (usize, usize, usize),
    pub signal_ma_length: (usize, usize, usize),
    pub ma_type: Option<PriceMovingAverageRatioPercentileMaType>,
    pub signal_ma_type: Option<PriceMovingAverageRatioPercentileMaType>,
    pub line_mode: Option<PriceMovingAverageRatioPercentileLineMode>,
}

impl Default for PriceMovingAverageRatioPercentileBatchRange {
    fn default() -> Self {
        Self {
            ma_length: (DEFAULT_MA_LENGTH, DEFAULT_MA_LENGTH, 0),
            pmarp_lookback: (DEFAULT_PMARP_LOOKBACK, DEFAULT_PMARP_LOOKBACK, 0),
            signal_ma_length: (DEFAULT_SIGNAL_MA_LENGTH, DEFAULT_SIGNAL_MA_LENGTH, 0),
            ma_type: Some(PriceMovingAverageRatioPercentileMaType::Vwma),
            signal_ma_type: Some(PriceMovingAverageRatioPercentileMaType::Sma),
            line_mode: Some(PriceMovingAverageRatioPercentileLineMode::Pmarp),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct PriceMovingAverageRatioPercentileBatchBuilder {
    range: PriceMovingAverageRatioPercentileBatchRange,
    kernel: Kernel,
}

#[derive(Clone, Debug)]
pub struct PriceMovingAverageRatioPercentileBatchOutput {
    pub pmar: Vec<f64>,
    pub pmarp: Vec<f64>,
    pub plotline: Vec<f64>,
    pub signal: Vec<f64>,
    pub pmar_high: Vec<f64>,
    pub pmar_low: Vec<f64>,
    pub scaled_pmar: Vec<f64>,
    pub combos: Vec<PriceMovingAverageRatioPercentileParams>,
    pub rows: usize,
    pub cols: usize,
}

impl PriceMovingAverageRatioPercentileBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    pub fn ma_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.ma_length = (start, end, step);
        self
    }

    pub fn pmarp_lookback_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.pmarp_lookback = (start, end, step);
        self
    }

    pub fn signal_ma_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.signal_ma_length = (start, end, step);
        self
    }

    pub fn ma_type(mut self, value: PriceMovingAverageRatioPercentileMaType) -> Self {
        self.range.ma_type = Some(value);
        self
    }

    pub fn signal_ma_type(mut self, value: PriceMovingAverageRatioPercentileMaType) -> Self {
        self.range.signal_ma_type = Some(value);
        self
    }

    pub fn line_mode(mut self, value: PriceMovingAverageRatioPercentileLineMode) -> Self {
        self.range.line_mode = Some(value);
        self
    }

    pub fn apply_slices(
        self,
        price: &[f64],
        volume: &[f64],
    ) -> Result<PriceMovingAverageRatioPercentileBatchOutput, PriceMovingAverageRatioPercentileError>
    {
        price_moving_average_ratio_percentile_batch_with_kernel(
            price,
            volume,
            &self.range,
            self.kernel,
        )
    }
}

impl PriceMovingAverageRatioPercentileBatchOutput {
    pub fn row_for_params(
        &self,
        params: &PriceMovingAverageRatioPercentileParams,
    ) -> Option<usize> {
        let ma_length = params.ma_length.unwrap_or(DEFAULT_MA_LENGTH);
        let ma_type = params.ma_type.unwrap_or_default();
        let pmarp_lookback = params.pmarp_lookback.unwrap_or(DEFAULT_PMARP_LOOKBACK);
        let signal_ma_length = params.signal_ma_length.unwrap_or(DEFAULT_SIGNAL_MA_LENGTH);
        let signal_ma_type = params
            .signal_ma_type
            .unwrap_or(PriceMovingAverageRatioPercentileMaType::Sma);
        let line_mode = params.line_mode.unwrap_or_default();
        self.combos.iter().position(|combo| {
            combo.ma_length.unwrap_or(DEFAULT_MA_LENGTH) == ma_length
                && combo.ma_type.unwrap_or_default() == ma_type
                && combo.pmarp_lookback.unwrap_or(DEFAULT_PMARP_LOOKBACK) == pmarp_lookback
                && combo.signal_ma_length.unwrap_or(DEFAULT_SIGNAL_MA_LENGTH) == signal_ma_length
                && combo
                    .signal_ma_type
                    .unwrap_or(PriceMovingAverageRatioPercentileMaType::Sma)
                    == signal_ma_type
                && combo.line_mode.unwrap_or_default() == line_mode
        })
    }
}

#[inline(always)]
fn axis_usize(
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, PriceMovingAverageRatioPercentileError> {
    if start == end {
        return Ok(vec![start]);
    }
    if step == 0 {
        return Err(PriceMovingAverageRatioPercentileError::InvalidRange {
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
        return Err(PriceMovingAverageRatioPercentileError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

pub fn expand_grid(
    range: &PriceMovingAverageRatioPercentileBatchRange,
) -> Result<Vec<PriceMovingAverageRatioPercentileParams>, PriceMovingAverageRatioPercentileError> {
    let ma_type = range.ma_type.unwrap_or_default();
    let signal_ma_type = range
        .signal_ma_type
        .unwrap_or(PriceMovingAverageRatioPercentileMaType::Sma);
    let line_mode = range.line_mode.unwrap_or_default();
    let ma_lengths = axis_usize(range.ma_length)?;
    let lookbacks = axis_usize(range.pmarp_lookback)?;
    let signal_lengths = axis_usize(range.signal_ma_length)?;
    let mut out = Vec::with_capacity(ma_lengths.len() * lookbacks.len() * signal_lengths.len());
    for ma_length in ma_lengths {
        for pmarp_lookback in &lookbacks {
            for signal_ma_length in &signal_lengths {
                out.push(PriceMovingAverageRatioPercentileParams {
                    ma_length: Some(ma_length),
                    ma_type: Some(ma_type),
                    pmarp_lookback: Some(*pmarp_lookback),
                    signal_ma_length: Some(*signal_ma_length),
                    signal_ma_type: Some(signal_ma_type),
                    line_mode: Some(line_mode),
                });
            }
        }
    }
    Ok(out)
}

fn batch_compute_rows(
    price: &[f64],
    volume: &[f64],
    sweep: &PriceMovingAverageRatioPercentileBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<
    (
        Vec<PriceMovingAverageRatioPercentileParams>,
        Vec<PriceMovingAverageRatioPercentileOutput>,
    ),
    PriceMovingAverageRatioPercentileError,
> {
    let combos = expand_grid(sweep)?;
    let kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel().to_non_batch(),
        other if other.is_batch() => other.to_non_batch(),
        other => other.to_non_batch(),
    };
    let compute = |params: &PriceMovingAverageRatioPercentileParams| {
        let input =
            PriceMovingAverageRatioPercentileInput::from_slices(price, volume, params.clone());
        price_moving_average_ratio_percentile_with_kernel(&input, kernel)
    };
    let rows = if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            combos
                .par_iter()
                .map(compute)
                .collect::<Result<Vec<_>, _>>()?
        }
        #[cfg(target_arch = "wasm32")]
        {
            combos.iter().map(compute).collect::<Result<Vec<_>, _>>()?
        }
    } else {
        combos.iter().map(compute).collect::<Result<Vec<_>, _>>()?
    };
    Ok((combos, rows))
}

fn flatten_rows(
    rows: &[PriceMovingAverageRatioPercentileOutput],
    cols: usize,
) -> PriceMovingAverageRatioPercentileOutput {
    let total = rows.len() * cols;
    let mut pmar = vec![f64::NAN; total];
    let mut pmarp = vec![f64::NAN; total];
    let mut plotline = vec![f64::NAN; total];
    let mut signal = vec![f64::NAN; total];
    let mut pmar_high = vec![f64::NAN; total];
    let mut pmar_low = vec![f64::NAN; total];
    let mut scaled_pmar = vec![f64::NAN; total];
    for (row_idx, row) in rows.iter().enumerate() {
        let start = row_idx * cols;
        let end = start + cols;
        pmar[start..end].copy_from_slice(&row.pmar);
        pmarp[start..end].copy_from_slice(&row.pmarp);
        plotline[start..end].copy_from_slice(&row.plotline);
        signal[start..end].copy_from_slice(&row.signal);
        pmar_high[start..end].copy_from_slice(&row.pmar_high);
        pmar_low[start..end].copy_from_slice(&row.pmar_low);
        scaled_pmar[start..end].copy_from_slice(&row.scaled_pmar);
    }
    PriceMovingAverageRatioPercentileOutput {
        pmar,
        pmarp,
        plotline,
        signal,
        pmar_high,
        pmar_low,
        scaled_pmar,
    }
}

pub fn price_moving_average_ratio_percentile_batch_with_kernel(
    price: &[f64],
    volume: &[f64],
    sweep: &PriceMovingAverageRatioPercentileBatchRange,
    kernel: Kernel,
) -> Result<PriceMovingAverageRatioPercentileBatchOutput, PriceMovingAverageRatioPercentileError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(PriceMovingAverageRatioPercentileError::InvalidKernelForBatch(kernel));
        }
    };
    price_moving_average_ratio_percentile_batch_par_slice(
        price,
        volume,
        sweep,
        batch_kernel.to_non_batch(),
    )
}

#[inline(always)]
pub fn price_moving_average_ratio_percentile_batch_slice(
    price: &[f64],
    volume: &[f64],
    sweep: &PriceMovingAverageRatioPercentileBatchRange,
    kernel: Kernel,
) -> Result<PriceMovingAverageRatioPercentileBatchOutput, PriceMovingAverageRatioPercentileError> {
    let _ = extract_price_volume(&PriceMovingAverageRatioPercentileInput::from_slices(
        price,
        volume,
        PriceMovingAverageRatioPercentileParams::default(),
    ))?;
    let (combos, rows) = batch_compute_rows(price, volume, sweep, kernel, false)?;
    let flat = flatten_rows(&rows, price.len());
    Ok(PriceMovingAverageRatioPercentileBatchOutput {
        pmar: flat.pmar,
        pmarp: flat.pmarp,
        plotline: flat.plotline,
        signal: flat.signal,
        pmar_high: flat.pmar_high,
        pmar_low: flat.pmar_low,
        scaled_pmar: flat.scaled_pmar,
        rows: combos.len(),
        cols: price.len(),
        combos,
    })
}

#[inline(always)]
pub fn price_moving_average_ratio_percentile_batch_par_slice(
    price: &[f64],
    volume: &[f64],
    sweep: &PriceMovingAverageRatioPercentileBatchRange,
    kernel: Kernel,
) -> Result<PriceMovingAverageRatioPercentileBatchOutput, PriceMovingAverageRatioPercentileError> {
    let _ = extract_price_volume(&PriceMovingAverageRatioPercentileInput::from_slices(
        price,
        volume,
        PriceMovingAverageRatioPercentileParams::default(),
    ))?;
    let (combos, rows) = batch_compute_rows(price, volume, sweep, kernel, true)?;
    let flat = flatten_rows(&rows, price.len());
    Ok(PriceMovingAverageRatioPercentileBatchOutput {
        pmar: flat.pmar,
        pmarp: flat.pmarp,
        plotline: flat.plotline,
        signal: flat.signal,
        pmar_high: flat.pmar_high,
        pmar_low: flat.pmar_low,
        scaled_pmar: flat.scaled_pmar,
        rows: combos.len(),
        cols: price.len(),
        combos,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn price_moving_average_ratio_percentile_batch_into_slice(
    pmar_out: &mut [f64],
    pmarp_out: &mut [f64],
    plotline_out: &mut [f64],
    signal_out: &mut [f64],
    pmar_high_out: &mut [f64],
    pmar_low_out: &mut [f64],
    scaled_pmar_out: &mut [f64],
    price: &[f64],
    volume: &[f64],
    sweep: &PriceMovingAverageRatioPercentileBatchRange,
    kernel: Kernel,
) -> Result<(), PriceMovingAverageRatioPercentileError> {
    let combos = expand_grid(sweep)?;
    let expected = combos.len().checked_mul(price.len()).ok_or_else(|| {
        PriceMovingAverageRatioPercentileError::InvalidRange {
            start: combos.len().to_string(),
            end: price.len().to_string(),
            step: "rows*cols".to_string(),
        }
    })?;
    for dst in [
        pmar_out as &[f64],
        pmarp_out as &[f64],
        plotline_out as &[f64],
        signal_out as &[f64],
        pmar_high_out as &[f64],
        pmar_low_out as &[f64],
        scaled_pmar_out as &[f64],
    ] {
        if dst.len() != expected {
            return Err(
                PriceMovingAverageRatioPercentileError::OutputLengthMismatch {
                    expected,
                    got: dst.len(),
                },
            );
        }
    }
    let (_combos, rows) = batch_compute_rows(price, volume, sweep, kernel, true)?;
    let cols = price.len();
    for (row_idx, row) in rows.iter().enumerate() {
        let start = row_idx * cols;
        let end = start + cols;
        pmar_out[start..end].copy_from_slice(&row.pmar);
        pmarp_out[start..end].copy_from_slice(&row.pmarp);
        plotline_out[start..end].copy_from_slice(&row.plotline);
        signal_out[start..end].copy_from_slice(&row.signal);
        pmar_high_out[start..end].copy_from_slice(&row.pmar_high);
        pmar_low_out[start..end].copy_from_slice(&row.pmar_low);
        scaled_pmar_out[start..end].copy_from_slice(&row.scaled_pmar);
    }
    Ok(())
}

#[cfg(feature = "python")]
#[pyfunction(name = "price_moving_average_ratio_percentile")]
#[pyo3(signature = (
    price,
    volume,
    ma_length=20,
    ma_type="vwma",
    pmarp_lookback=350,
    signal_ma_length=20,
    signal_ma_type="sma",
    line_mode="pmarp",
    kernel=None
))]
pub fn price_moving_average_ratio_percentile_py<'py>(
    py: Python<'py>,
    price: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    ma_length: usize,
    ma_type: &str,
    pmarp_lookback: usize,
    signal_ma_length: usize,
    signal_ma_type: &str,
    line_mode: &str,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let price = price.as_slice()?;
    let volume = volume.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = PriceMovingAverageRatioPercentileInput::from_slices(
        price,
        volume,
        PriceMovingAverageRatioPercentileParams {
            ma_length: Some(ma_length),
            ma_type: Some(
                ma_type
                    .parse::<PriceMovingAverageRatioPercentileMaType>()
                    .map_err(PyValueError::new_err)?,
            ),
            pmarp_lookback: Some(pmarp_lookback),
            signal_ma_length: Some(signal_ma_length),
            signal_ma_type: Some(
                signal_ma_type
                    .parse::<PriceMovingAverageRatioPercentileMaType>()
                    .map_err(PyValueError::new_err)?,
            ),
            line_mode: Some(
                line_mode
                    .parse::<PriceMovingAverageRatioPercentileLineMode>()
                    .map_err(PyValueError::new_err)?,
            ),
        },
    );
    let out = py
        .allow_threads(|| price_moving_average_ratio_percentile_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dict = PyDict::new(py);
    dict.set_item("pmar", out.pmar.into_pyarray(py))?;
    dict.set_item("pmarp", out.pmarp.into_pyarray(py))?;
    dict.set_item("plotline", out.plotline.into_pyarray(py))?;
    dict.set_item("signal", out.signal.into_pyarray(py))?;
    dict.set_item("pmar_high", out.pmar_high.into_pyarray(py))?;
    dict.set_item("pmar_low", out.pmar_low.into_pyarray(py))?;
    dict.set_item("scaled_pmar", out.scaled_pmar.into_pyarray(py))?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "PriceMovingAverageRatioPercentileStream")]
pub struct PriceMovingAverageRatioPercentileStreamPy {
    stream: PriceMovingAverageRatioPercentileStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl PriceMovingAverageRatioPercentileStreamPy {
    #[new]
    #[pyo3(signature = (
        ma_length=20,
        ma_type="vwma",
        pmarp_lookback=350,
        signal_ma_length=20,
        signal_ma_type="sma",
        line_mode="pmarp"
    ))]
    fn new(
        ma_length: usize,
        ma_type: &str,
        pmarp_lookback: usize,
        signal_ma_length: usize,
        signal_ma_type: &str,
        line_mode: &str,
    ) -> PyResult<Self> {
        let stream = PriceMovingAverageRatioPercentileStream::try_new(
            PriceMovingAverageRatioPercentileParams {
                ma_length: Some(ma_length),
                ma_type: Some(
                    ma_type
                        .parse::<PriceMovingAverageRatioPercentileMaType>()
                        .map_err(PyValueError::new_err)?,
                ),
                pmarp_lookback: Some(pmarp_lookback),
                signal_ma_length: Some(signal_ma_length),
                signal_ma_type: Some(
                    signal_ma_type
                        .parse::<PriceMovingAverageRatioPercentileMaType>()
                        .map_err(PyValueError::new_err)?,
                ),
                line_mode: Some(
                    line_mode
                        .parse::<PriceMovingAverageRatioPercentileLineMode>()
                        .map_err(PyValueError::new_err)?,
                ),
            },
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update<'py>(
        &mut self,
        py: Python<'py>,
        price: f64,
        volume: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let values = self.stream.update(price, volume);
        let dict = PyDict::new(py);
        dict.set_item("pmar", values.0)?;
        dict.set_item("pmarp", values.1)?;
        dict.set_item("plotline", values.2)?;
        dict.set_item("signal", values.3)?;
        dict.set_item("pmar_high", values.4)?;
        dict.set_item("pmar_low", values.5)?;
        dict.set_item("scaled_pmar", values.6)?;
        Ok(dict)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "price_moving_average_ratio_percentile_batch")]
#[pyo3(signature = (
    price,
    volume,
    ma_length_range=(20,20,0),
    ma_type="vwma",
    pmarp_lookback_range=(350,350,0),
    signal_ma_length_range=(20,20,0),
    signal_ma_type="sma",
    line_mode="pmarp",
    kernel=None
))]
pub fn price_moving_average_ratio_percentile_batch_py<'py>(
    py: Python<'py>,
    price: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    ma_length_range: (usize, usize, usize),
    ma_type: &str,
    pmarp_lookback_range: (usize, usize, usize),
    signal_ma_length_range: (usize, usize, usize),
    signal_ma_type: &str,
    line_mode: &str,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let price = price.as_slice()?;
    let volume = volume.as_slice()?;
    let kernel = validate_kernel(kernel, true)?;
    let sweep = PriceMovingAverageRatioPercentileBatchRange {
        ma_length: ma_length_range,
        pmarp_lookback: pmarp_lookback_range,
        signal_ma_length: signal_ma_length_range,
        ma_type: Some(
            ma_type
                .parse::<PriceMovingAverageRatioPercentileMaType>()
                .map_err(PyValueError::new_err)?,
        ),
        signal_ma_type: Some(
            signal_ma_type
                .parse::<PriceMovingAverageRatioPercentileMaType>()
                .map_err(PyValueError::new_err)?,
        ),
        line_mode: Some(
            line_mode
                .parse::<PriceMovingAverageRatioPercentileLineMode>()
                .map_err(PyValueError::new_err)?,
        ),
    };
    let out = py
        .allow_threads(|| {
            price_moving_average_ratio_percentile_batch_with_kernel(price, volume, &sweep, kernel)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dict = PyDict::new(py);
    dict.set_item(
        "pmar",
        out.pmar.into_pyarray(py).reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "pmarp",
        out.pmarp.into_pyarray(py).reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "plotline",
        out.plotline
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "signal",
        out.signal.into_pyarray(py).reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "pmar_high",
        out.pmar_high
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "pmar_low",
        out.pmar_low
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "scaled_pmar",
        out.scaled_pmar
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item("rows", out.rows)?;
    dict.set_item("cols", out.cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_price_moving_average_ratio_percentile_module(
    m: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        price_moving_average_ratio_percentile_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        price_moving_average_ratio_percentile_batch_py,
        m
    )?)?;
    m.add_class::<PriceMovingAverageRatioPercentileStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct PriceMovingAverageRatioPercentileJsOutput {
    pub pmar: Vec<f64>,
    pub pmarp: Vec<f64>,
    pub plotline: Vec<f64>,
    pub signal: Vec<f64>,
    pub pmar_high: Vec<f64>,
    pub pmar_low: Vec<f64>,
    pub scaled_pmar: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct PriceMovingAverageRatioPercentileBatchConfig {
    pub ma_length_range: Vec<f64>,
    pub pmarp_lookback_range: Vec<f64>,
    pub signal_ma_length_range: Vec<f64>,
    pub ma_type: Option<String>,
    pub signal_ma_type: Option<String>,
    pub line_mode: Option<String>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct PriceMovingAverageRatioPercentileBatchJsOutput {
    pub pmar: Vec<f64>,
    pub pmarp: Vec<f64>,
    pub plotline: Vec<f64>,
    pub signal: Vec<f64>,
    pub pmar_high: Vec<f64>,
    pub pmar_low: Vec<f64>,
    pub scaled_pmar: Vec<f64>,
    pub ma_lengths: Vec<usize>,
    pub pmarp_lookbacks: Vec<usize>,
    pub signal_ma_lengths: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
fn js_vec3_to_usize(name: &str, values: &[f64]) -> Result<(usize, usize, usize), JsValue> {
    if values.len() != 3 {
        return Err(JsValue::from_str(&format!(
            "Invalid config: {name} must have exactly 3 elements [start, end, step]"
        )));
    }
    let mut out = [0usize; 3];
    for (i, value) in values.iter().copied().enumerate() {
        if !value.is_finite() || value < 0.0 {
            return Err(JsValue::from_str(&format!(
                "Invalid config: {name}[{i}] must be a finite non-negative whole number"
            )));
        }
        let rounded = value.round();
        if (value - rounded).abs() > 1e-9 {
            return Err(JsValue::from_str(&format!(
                "Invalid config: {name}[{i}] must be a whole number"
            )));
        }
        out[i] = rounded as usize;
    }
    Ok((out[0], out[1], out[2]))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "price_moving_average_ratio_percentile_js")]
pub fn price_moving_average_ratio_percentile_js(
    price: &[f64],
    volume: &[f64],
    ma_length: usize,
    ma_type: &str,
    pmarp_lookback: usize,
    signal_ma_length: usize,
    signal_ma_type: &str,
    line_mode: &str,
) -> Result<JsValue, JsValue> {
    let input = PriceMovingAverageRatioPercentileInput::from_slices(
        price,
        volume,
        PriceMovingAverageRatioPercentileParams {
            ma_length: Some(ma_length),
            ma_type: Some(
                ma_type
                    .parse::<PriceMovingAverageRatioPercentileMaType>()
                    .map_err(|e| JsValue::from_str(&e))?,
            ),
            pmarp_lookback: Some(pmarp_lookback),
            signal_ma_length: Some(signal_ma_length),
            signal_ma_type: Some(
                signal_ma_type
                    .parse::<PriceMovingAverageRatioPercentileMaType>()
                    .map_err(|e| JsValue::from_str(&e))?,
            ),
            line_mode: Some(
                line_mode
                    .parse::<PriceMovingAverageRatioPercentileLineMode>()
                    .map_err(|e| JsValue::from_str(&e))?,
            ),
        },
    );
    let out = price_moving_average_ratio_percentile_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&PriceMovingAverageRatioPercentileJsOutput {
        pmar: out.pmar,
        pmarp: out.pmarp,
        plotline: out.plotline,
        signal: out.signal,
        pmar_high: out.pmar_high,
        pmar_low: out.pmar_low,
        scaled_pmar: out.scaled_pmar,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "price_moving_average_ratio_percentile_batch_js")]
pub fn price_moving_average_ratio_percentile_batch_js(
    price: &[f64],
    volume: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: PriceMovingAverageRatioPercentileBatchConfig =
        serde_wasm_bindgen::from_value(config)
            .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let sweep = PriceMovingAverageRatioPercentileBatchRange {
        ma_length: js_vec3_to_usize("ma_length_range", &config.ma_length_range)?,
        pmarp_lookback: js_vec3_to_usize("pmarp_lookback_range", &config.pmarp_lookback_range)?,
        signal_ma_length: js_vec3_to_usize(
            "signal_ma_length_range",
            &config.signal_ma_length_range,
        )?,
        ma_type: Some(
            config
                .ma_type
                .unwrap_or_else(|| "vwma".to_string())
                .parse::<PriceMovingAverageRatioPercentileMaType>()
                .map_err(|e| JsValue::from_str(&e))?,
        ),
        signal_ma_type: Some(
            config
                .signal_ma_type
                .unwrap_or_else(|| "sma".to_string())
                .parse::<PriceMovingAverageRatioPercentileMaType>()
                .map_err(|e| JsValue::from_str(&e))?,
        ),
        line_mode: Some(
            config
                .line_mode
                .unwrap_or_else(|| "pmarp".to_string())
                .parse::<PriceMovingAverageRatioPercentileLineMode>()
                .map_err(|e| JsValue::from_str(&e))?,
        ),
    };
    let out = price_moving_average_ratio_percentile_batch_with_kernel(
        price,
        volume,
        &sweep,
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let ma_lengths = out
        .combos
        .iter()
        .map(|combo| combo.ma_length.unwrap_or(DEFAULT_MA_LENGTH))
        .collect();
    let pmarp_lookbacks = out
        .combos
        .iter()
        .map(|combo| combo.pmarp_lookback.unwrap_or(DEFAULT_PMARP_LOOKBACK))
        .collect();
    let signal_ma_lengths = out
        .combos
        .iter()
        .map(|combo| combo.signal_ma_length.unwrap_or(DEFAULT_SIGNAL_MA_LENGTH))
        .collect();
    serde_wasm_bindgen::to_value(&PriceMovingAverageRatioPercentileBatchJsOutput {
        pmar: out.pmar,
        pmarp: out.pmarp,
        plotline: out.plotline,
        signal: out.signal,
        pmar_high: out.pmar_high,
        pmar_low: out.pmar_low,
        scaled_pmar: out.scaled_pmar,
        ma_lengths,
        pmarp_lookbacks,
        signal_ma_lengths,
        rows: out.rows,
        cols: out.cols,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn price_moving_average_ratio_percentile_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn price_moving_average_ratio_percentile_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn price_moving_average_ratio_percentile_into(
    price_ptr: *const f64,
    volume_ptr: *const f64,
    pmar_ptr: *mut f64,
    pmarp_ptr: *mut f64,
    plotline_ptr: *mut f64,
    signal_ptr: *mut f64,
    pmar_high_ptr: *mut f64,
    pmar_low_ptr: *mut f64,
    scaled_pmar_ptr: *mut f64,
    len: usize,
    ma_length: usize,
    ma_type: &str,
    pmarp_lookback: usize,
    signal_ma_length: usize,
    signal_ma_type: &str,
    line_mode: &str,
) -> Result<(), JsValue> {
    if price_ptr.is_null()
        || volume_ptr.is_null()
        || pmar_ptr.is_null()
        || pmarp_ptr.is_null()
        || plotline_ptr.is_null()
        || signal_ptr.is_null()
        || pmar_high_ptr.is_null()
        || pmar_low_ptr.is_null()
        || scaled_pmar_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }
    let ma_type = ma_type
        .parse::<PriceMovingAverageRatioPercentileMaType>()
        .map_err(|e| JsValue::from_str(&e))?;
    let signal_ma_type = signal_ma_type
        .parse::<PriceMovingAverageRatioPercentileMaType>()
        .map_err(|e| JsValue::from_str(&e))?;
    let line_mode = line_mode
        .parse::<PriceMovingAverageRatioPercentileLineMode>()
        .map_err(|e| JsValue::from_str(&e))?;
    unsafe {
        let price = std::slice::from_raw_parts(price_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);
        let input = PriceMovingAverageRatioPercentileInput::from_slices(
            price,
            volume,
            PriceMovingAverageRatioPercentileParams {
                ma_length: Some(ma_length),
                ma_type: Some(ma_type),
                pmarp_lookback: Some(pmarp_lookback),
                signal_ma_length: Some(signal_ma_length),
                signal_ma_type: Some(signal_ma_type),
                line_mode: Some(line_mode),
            },
        );
        price_moving_average_ratio_percentile_into_slice(
            std::slice::from_raw_parts_mut(pmar_ptr, len),
            std::slice::from_raw_parts_mut(pmarp_ptr, len),
            std::slice::from_raw_parts_mut(plotline_ptr, len),
            std::slice::from_raw_parts_mut(signal_ptr, len),
            std::slice::from_raw_parts_mut(pmar_high_ptr, len),
            std::slice::from_raw_parts_mut(pmar_low_ptr, len),
            std::slice::from_raw_parts_mut(scaled_pmar_ptr, len),
            &input,
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn price_moving_average_ratio_percentile_batch_into(
    price_ptr: *const f64,
    volume_ptr: *const f64,
    pmar_ptr: *mut f64,
    pmarp_ptr: *mut f64,
    plotline_ptr: *mut f64,
    signal_ptr: *mut f64,
    pmar_high_ptr: *mut f64,
    pmar_low_ptr: *mut f64,
    scaled_pmar_ptr: *mut f64,
    len: usize,
    ma_start: usize,
    ma_end: usize,
    ma_step: usize,
    lookback_start: usize,
    lookback_end: usize,
    lookback_step: usize,
    signal_start: usize,
    signal_end: usize,
    signal_step: usize,
    ma_type: &str,
    signal_ma_type: &str,
    line_mode: &str,
) -> Result<usize, JsValue> {
    if price_ptr.is_null()
        || volume_ptr.is_null()
        || pmar_ptr.is_null()
        || pmarp_ptr.is_null()
        || plotline_ptr.is_null()
        || signal_ptr.is_null()
        || pmar_high_ptr.is_null()
        || pmar_low_ptr.is_null()
        || scaled_pmar_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to price_moving_average_ratio_percentile_batch_into",
        ));
    }
    let sweep = PriceMovingAverageRatioPercentileBatchRange {
        ma_length: (ma_start, ma_end, ma_step),
        pmarp_lookback: (lookback_start, lookback_end, lookback_step),
        signal_ma_length: (signal_start, signal_end, signal_step),
        ma_type: Some(
            ma_type
                .parse::<PriceMovingAverageRatioPercentileMaType>()
                .map_err(|e| JsValue::from_str(&e))?,
        ),
        signal_ma_type: Some(
            signal_ma_type
                .parse::<PriceMovingAverageRatioPercentileMaType>()
                .map_err(|e| JsValue::from_str(&e))?,
        ),
        line_mode: Some(
            line_mode
                .parse::<PriceMovingAverageRatioPercentileLineMode>()
                .map_err(|e| JsValue::from_str(&e))?,
        ),
    };
    let rows = expand_grid(&sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?
        .len();
    let total = rows.checked_mul(len).ok_or_else(|| {
        JsValue::from_str("rows*cols overflow in price_moving_average_ratio_percentile_batch_into")
    })?;
    unsafe {
        let price = std::slice::from_raw_parts(price_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);
        price_moving_average_ratio_percentile_batch_into_slice(
            std::slice::from_raw_parts_mut(pmar_ptr, total),
            std::slice::from_raw_parts_mut(pmarp_ptr, total),
            std::slice::from_raw_parts_mut(plotline_ptr, total),
            std::slice::from_raw_parts_mut(signal_ptr, total),
            std::slice::from_raw_parts_mut(pmar_high_ptr, total),
            std::slice::from_raw_parts_mut(pmar_low_ptr, total),
            std::slice::from_raw_parts_mut(scaled_pmar_ptr, total),
            price,
            volume,
            &sweep,
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }
    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn price_moving_average_ratio_percentile_output_into_js(
    price: &[f64],
    volume: &[f64],
    ma_length: usize,
    ma_type: &str,
    pmarp_lookback: usize,
    signal_ma_length: usize,
    signal_ma_type: &str,
    line_mode: &str,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = price_moving_average_ratio_percentile_js(
        price,
        volume,
        ma_length,
        ma_type,
        pmarp_lookback,
        signal_ma_length,
        signal_ma_type,
        line_mode,
    )?;
    crate::write_wasm_object_f64_outputs(
        "price_moving_average_ratio_percentile_output_into_js",
        &value,
        out,
    )
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn price_moving_average_ratio_percentile_batch_output_into_js(
    price: &[f64],
    volume: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = price_moving_average_ratio_percentile_batch_js(price, volume, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "price_moving_average_ratio_percentile_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_series_eq(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for (&lhs, &rhs) in a.iter().zip(b.iter()) {
            if lhs.is_nan() && rhs.is_nan() {
                continue;
            }
            assert!((lhs - rhs).abs() <= 1e-12, "lhs={lhs}, rhs={rhs}");
        }
    }

    fn sample_price_volume() -> (Vec<f64>, Vec<f64>) {
        let price: Vec<f64> = (0..180)
            .map(|i| 100.0 + (i as f64 * 0.13).sin() * 2.0 + i as f64 * 0.04)
            .collect();
        let volume: Vec<f64> = (0..180)
            .map(|i| 1_000.0 + (i as f64 * 0.11).cos() * 75.0 + i as f64 * 3.0)
            .collect();
        (price, volume)
    }

    #[test]
    fn pmarp_matches_manual_counting() {
        let (price, volume) = sample_price_volume();
        let input = PriceMovingAverageRatioPercentileInput::from_slices(
            &price,
            &volume,
            PriceMovingAverageRatioPercentileParams {
                ma_length: Some(20),
                ma_type: Some(PriceMovingAverageRatioPercentileMaType::Sma),
                pmarp_lookback: Some(30),
                signal_ma_length: Some(5),
                signal_ma_type: Some(PriceMovingAverageRatioPercentileMaType::Sma),
                line_mode: Some(PriceMovingAverageRatioPercentileLineMode::Pmarp),
            },
        );
        let out = price_moving_average_ratio_percentile(&input).unwrap();

        for i in 20..price.len() {
            if !out.pmar[i].is_finite() {
                continue;
            }
            let lookback = i.min(30);
            if lookback == 0 {
                continue;
            }
            let current = out.pmar[i].abs();
            let mut count = 0usize;
            for offset in 1..=lookback {
                let prev = out.pmar[i - offset].abs();
                if !(prev.is_finite() && prev > current) {
                    count += 1;
                }
            }
            let expected = (count as f64 / lookback as f64) * 100.0;
            assert!((out.pmarp[i] - expected).abs() <= 1e-12);
        }
    }

    #[test]
    fn into_slice_matches_owned_output() {
        let (price, volume) = sample_price_volume();
        let params = PriceMovingAverageRatioPercentileParams {
            ma_length: Some(20),
            ma_type: Some(PriceMovingAverageRatioPercentileMaType::Vwma),
            pmarp_lookback: Some(40),
            signal_ma_length: Some(10),
            signal_ma_type: Some(PriceMovingAverageRatioPercentileMaType::Ema),
            line_mode: Some(PriceMovingAverageRatioPercentileLineMode::Pmar),
        };
        let input = PriceMovingAverageRatioPercentileInput::from_slices(&price, &volume, params);
        let expected = price_moving_average_ratio_percentile(&input).unwrap();
        let mut pmar = vec![f64::NAN; price.len()];
        let mut pmarp = vec![f64::NAN; price.len()];
        let mut plotline = vec![f64::NAN; price.len()];
        let mut signal = vec![f64::NAN; price.len()];
        let mut pmar_high = vec![f64::NAN; price.len()];
        let mut pmar_low = vec![f64::NAN; price.len()];
        let mut scaled_pmar = vec![f64::NAN; price.len()];

        price_moving_average_ratio_percentile_into_slice(
            &mut pmar,
            &mut pmarp,
            &mut plotline,
            &mut signal,
            &mut pmar_high,
            &mut pmar_low,
            &mut scaled_pmar,
            &input,
            Kernel::Auto,
        )
        .unwrap();

        assert_series_eq(&pmar, &expected.pmar);
        assert_series_eq(&pmarp, &expected.pmarp);
        assert_series_eq(&plotline, &expected.plotline);
        assert_series_eq(&signal, &expected.signal);
        assert_series_eq(&pmar_high, &expected.pmar_high);
        assert_series_eq(&pmar_low, &expected.pmar_low);
        assert_series_eq(&scaled_pmar, &expected.scaled_pmar);
    }

    #[test]
    fn stream_last_value_matches_batch() {
        let (price, volume) = sample_price_volume();
        let params = PriceMovingAverageRatioPercentileParams {
            ma_length: Some(20),
            ma_type: Some(PriceMovingAverageRatioPercentileMaType::Vwma),
            pmarp_lookback: Some(35),
            signal_ma_length: Some(7),
            signal_ma_type: Some(PriceMovingAverageRatioPercentileMaType::Sma),
            line_mode: Some(PriceMovingAverageRatioPercentileLineMode::Pmarp),
        };
        let input =
            PriceMovingAverageRatioPercentileInput::from_slices(&price, &volume, params.clone());
        let batch = price_moving_average_ratio_percentile(&input).unwrap();
        let mut stream = PriceMovingAverageRatioPercentileStream::try_new(params).unwrap();
        let mut last = None;
        for (&p, &v) in price.iter().zip(volume.iter()) {
            last = Some(stream.update(p, v));
        }
        let last = last.unwrap();
        let i = price.len() - 1;
        assert_eq!(last.0, batch.pmar[i]);
        assert_eq!(last.1, batch.pmarp[i]);
        assert_eq!(last.2, batch.plotline[i]);
        assert_eq!(last.3, batch.signal[i]);
        assert_eq!(last.4, batch.pmar_high[i]);
        assert_eq!(last.5, batch.pmar_low[i]);
        assert_eq!(last.6, batch.scaled_pmar[i]);
    }

    #[test]
    fn batch_first_row_matches_single() {
        let (price, volume) = sample_price_volume();
        let sweep = PriceMovingAverageRatioPercentileBatchRange {
            ma_length: (20, 22, 2),
            pmarp_lookback: (30, 30, 0),
            signal_ma_length: (5, 5, 0),
            ma_type: Some(PriceMovingAverageRatioPercentileMaType::Sma),
            signal_ma_type: Some(PriceMovingAverageRatioPercentileMaType::Sma),
            line_mode: Some(PriceMovingAverageRatioPercentileLineMode::Pmarp),
        };
        let batch = price_moving_average_ratio_percentile_batch_with_kernel(
            &price,
            &volume,
            &sweep,
            Kernel::Auto,
        )
        .unwrap();
        assert_eq!(batch.rows, 2);
        assert_eq!(batch.cols, price.len());

        let single = price_moving_average_ratio_percentile(
            &PriceMovingAverageRatioPercentileInput::from_slices(
                &price,
                &volume,
                PriceMovingAverageRatioPercentileParams {
                    ma_length: Some(20),
                    ma_type: Some(PriceMovingAverageRatioPercentileMaType::Sma),
                    pmarp_lookback: Some(30),
                    signal_ma_length: Some(5),
                    signal_ma_type: Some(PriceMovingAverageRatioPercentileMaType::Sma),
                    line_mode: Some(PriceMovingAverageRatioPercentileLineMode::Pmarp),
                },
            ),
        )
        .unwrap();

        assert_series_eq(&batch.pmar[..price.len()], single.pmar.as_slice());
        assert_series_eq(&batch.pmarp[..price.len()], single.pmarp.as_slice());
        assert_series_eq(&batch.plotline[..price.len()], single.plotline.as_slice());
        assert_series_eq(&batch.signal[..price.len()], single.signal.as_slice());
    }

    #[test]
    fn invalid_period_is_rejected() {
        let (price, volume) = sample_price_volume();
        let input = PriceMovingAverageRatioPercentileInput::from_slices(
            &price,
            &volume,
            PriceMovingAverageRatioPercentileParams {
                ma_length: Some(0),
                ..PriceMovingAverageRatioPercentileParams::default()
            },
        );
        let err = price_moving_average_ratio_percentile(&input).unwrap_err();
        assert!(matches!(
            err,
            PriceMovingAverageRatioPercentileError::InvalidPeriod { .. }
        ));
    }
}
