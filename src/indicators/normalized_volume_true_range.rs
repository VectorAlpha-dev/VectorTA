#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};
#[cfg(feature = "python")]
use pyo3::wrap_pyfunction;

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use serde::{Deserialize, Serialize};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

use crate::utilities::data_loader::Candles;
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, make_uninit_matrix};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::mem::ManuallyDrop;
use thiserror::Error;

const DEFAULT_OUTLIER_RANGE: f64 = 5.0;
const DEFAULT_ATR_LENGTH: usize = 14;
const DEFAULT_VOLUME_LENGTH: usize = 14;
const MIN_LENGTH: usize = 2;
const MIN_OUTLIER_RANGE: f64 = 0.5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize),
    serde(rename_all = "snake_case")
)]
pub enum NormalizedVolumeTrueRangeStyle {
    Body,
    Hl,
    Delta,
}

impl Default for NormalizedVolumeTrueRangeStyle {
    fn default() -> Self {
        Self::Body
    }
}

impl NormalizedVolumeTrueRangeStyle {
    #[inline(always)]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Body => "body",
            Self::Hl => "hl",
            Self::Delta => "delta",
        }
    }
}

impl std::str::FromStr for NormalizedVolumeTrueRangeStyle {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "body" => Ok(Self::Body),
            "hl" | "high_low" | "high/low" => Ok(Self::Hl),
            "delta" | "close_delta" | "close/close" => Ok(Self::Delta),
            other => Err(format!(
                "normalized_volume_true_range: invalid true_range_style: {other}"
            )),
        }
    }
}

#[derive(Debug, Clone)]
pub enum NormalizedVolumeTrueRangeData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct NormalizedVolumeTrueRangeOutput {
    pub normalized_volume: Vec<f64>,
    pub normalized_true_range: Vec<f64>,
    pub baseline: Vec<f64>,
    pub atr: Vec<f64>,
    pub average_volume: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct NormalizedVolumeTrueRangeParams {
    pub true_range_style: Option<NormalizedVolumeTrueRangeStyle>,
    pub outlier_range: Option<f64>,
    pub atr_length: Option<usize>,
    pub volume_length: Option<usize>,
}

impl Default for NormalizedVolumeTrueRangeParams {
    fn default() -> Self {
        Self {
            true_range_style: Some(NormalizedVolumeTrueRangeStyle::Body),
            outlier_range: Some(DEFAULT_OUTLIER_RANGE),
            atr_length: Some(DEFAULT_ATR_LENGTH),
            volume_length: Some(DEFAULT_VOLUME_LENGTH),
        }
    }
}

#[derive(Debug, Clone)]
pub struct NormalizedVolumeTrueRangeInput<'a> {
    pub data: NormalizedVolumeTrueRangeData<'a>,
    pub params: NormalizedVolumeTrueRangeParams,
}

impl<'a> NormalizedVolumeTrueRangeInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: NormalizedVolumeTrueRangeParams) -> Self {
        Self {
            data: NormalizedVolumeTrueRangeData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
        params: NormalizedVolumeTrueRangeParams,
    ) -> Self {
        Self {
            data: NormalizedVolumeTrueRangeData::Slices {
                open,
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
        Self::from_candles(candles, NormalizedVolumeTrueRangeParams::default())
    }

    #[inline(always)]
    pub fn get_true_range_style(&self) -> NormalizedVolumeTrueRangeStyle {
        self.params.true_range_style.unwrap_or_default()
    }

    #[inline(always)]
    pub fn get_outlier_range(&self) -> f64 {
        self.params.outlier_range.unwrap_or(DEFAULT_OUTLIER_RANGE)
    }

    #[inline(always)]
    pub fn get_atr_length(&self) -> usize {
        self.params.atr_length.unwrap_or(DEFAULT_ATR_LENGTH)
    }

    #[inline(always)]
    pub fn get_volume_length(&self) -> usize {
        self.params.volume_length.unwrap_or(DEFAULT_VOLUME_LENGTH)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct NormalizedVolumeTrueRangeBuilder {
    true_range_style: Option<NormalizedVolumeTrueRangeStyle>,
    outlier_range: Option<f64>,
    atr_length: Option<usize>,
    volume_length: Option<usize>,
    kernel: Kernel,
}

impl Default for NormalizedVolumeTrueRangeBuilder {
    fn default() -> Self {
        Self {
            true_range_style: None,
            outlier_range: None,
            atr_length: None,
            volume_length: None,
            kernel: Kernel::Auto,
        }
    }
}

impl NormalizedVolumeTrueRangeBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn true_range_style(mut self, style: NormalizedVolumeTrueRangeStyle) -> Self {
        self.true_range_style = Some(style);
        self
    }

    #[inline(always)]
    pub fn outlier_range(mut self, outlier_range: f64) -> Self {
        self.outlier_range = Some(outlier_range);
        self
    }

    #[inline(always)]
    pub fn atr_length(mut self, atr_length: usize) -> Self {
        self.atr_length = Some(atr_length);
        self
    }

    #[inline(always)]
    pub fn volume_length(mut self, volume_length: usize) -> Self {
        self.volume_length = Some(volume_length);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline(always)]
    fn params(self) -> NormalizedVolumeTrueRangeParams {
        NormalizedVolumeTrueRangeParams {
            true_range_style: self.true_range_style,
            outlier_range: self.outlier_range,
            atr_length: self.atr_length,
            volume_length: self.volume_length,
        }
    }

    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<NormalizedVolumeTrueRangeOutput, NormalizedVolumeTrueRangeError> {
        let input = NormalizedVolumeTrueRangeInput::from_candles(candles, self.params());
        normalized_volume_true_range_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Result<NormalizedVolumeTrueRangeOutput, NormalizedVolumeTrueRangeError> {
        let input = NormalizedVolumeTrueRangeInput::from_slices(
            open,
            high,
            low,
            close,
            volume,
            self.params(),
        );
        normalized_volume_true_range_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<NormalizedVolumeTrueRangeStream, NormalizedVolumeTrueRangeError> {
        NormalizedVolumeTrueRangeStream::try_new(self.params())
    }
}

#[derive(Debug, Error)]
pub enum NormalizedVolumeTrueRangeError {
    #[error("normalized_volume_true_range: input data slice is empty.")]
    EmptyInputData,
    #[error("normalized_volume_true_range: all values are NaN.")]
    AllValuesNaN,
    #[error(
        "normalized_volume_true_range: invalid outlier_range: {outlier_range}. Expected >= 0.5."
    )]
    InvalidOutlierRange { outlier_range: f64 },
    #[error("normalized_volume_true_range: invalid atr_length: {atr_length}. Expected >= 2.")]
    InvalidAtrLength { atr_length: usize },
    #[error(
        "normalized_volume_true_range: invalid volume_length: {volume_length}. Expected >= 2."
    )]
    InvalidVolumeLength { volume_length: usize },
    #[error("normalized_volume_true_range: inconsistent slice lengths: open={open_len}, high={high_len}, low={low_len}, close={close_len}, volume={volume_len}")]
    InconsistentSliceLengths {
        open_len: usize,
        high_len: usize,
        low_len: usize,
        close_len: usize,
        volume_len: usize,
    },
    #[error(
        "normalized_volume_true_range: output length mismatch: expected = {expected}, got = {got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("normalized_volume_true_range: invalid outlier range sweep: start={start}, end={end}, step={step}")]
    InvalidOutlierRangeSweep { start: f64, end: f64, step: f64 },
    #[error("normalized_volume_true_range: invalid atr length sweep: start={start}, end={end}, step={step}")]
    InvalidAtrLengthSweep {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("normalized_volume_true_range: invalid volume length sweep: start={start}, end={end}, step={step}")]
    InvalidVolumeLengthSweep {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("normalized_volume_true_range: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Debug, Clone, Copy)]
struct PreparedNormalizedVolumeTrueRange<'a> {
    open: &'a [f64],
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    volume: &'a [f64],
    len: usize,
    style: NormalizedVolumeTrueRangeStyle,
    outlier_range: f64,
    atr_length: usize,
    volume_length: usize,
}

#[derive(Debug, Clone)]
struct PositiveDeviationState {
    variance_sum: f64,
    qualifying_count: usize,
    current: f64,
}

impl Default for PositiveDeviationState {
    fn default() -> Self {
        Self {
            variance_sum: 0.0,
            qualifying_count: 0,
            current: f64::NAN,
        }
    }
}

impl PositiveDeviationState {
    #[inline(always)]
    fn update(&mut self, source: f64, average: f64) -> f64 {
        if source > average {
            let delta = source - average;
            self.variance_sum += delta * delta;
            self.qualifying_count += 1;
        }
        if self.qualifying_count >= 2 {
            self.current = (self.variance_sum / (self.qualifying_count - 1) as f64).sqrt();
        }
        self.current
    }
}

#[derive(Debug, Clone)]
struct FilledSmaState {
    len: usize,
    first_value: f64,
    ready: bool,
    ring: Vec<f64>,
    head: usize,
    sum: f64,
}

impl FilledSmaState {
    #[inline]
    fn new(len: usize) -> Self {
        Self {
            len,
            first_value: f64::NAN,
            ready: false,
            ring: vec![0.0; len],
            head: 0,
            sum: 0.0,
        }
    }

    #[inline]
    fn update(&mut self, value: f64) -> f64 {
        if !self.ready {
            if !value.is_finite() {
                return f64::NAN;
            }
            self.first_value = value;
            self.ready = true;
            self.ring.fill(value);
            self.sum = value * self.len as f64;
            self.head = 1 % self.len;
            return value;
        }

        let sanitized = if value.is_finite() {
            value
        } else {
            self.first_value
        };
        let old = self.ring[self.head];
        self.ring[self.head] = sanitized;
        self.sum += sanitized - old;
        self.head += 1;
        if self.head == self.len {
            self.head = 0;
        }
        self.sum / self.len as f64
    }
}

#[derive(Debug, Clone)]
struct NormalizedVolumeTrueRangeCore {
    style: NormalizedVolumeTrueRangeStyle,
    outlier_range: f64,
    abs_sum: f64,
    volume_sum: f64,
    count: usize,
    abs_positive_deviation: PositiveDeviationState,
    volume_positive_deviation: PositiveDeviationState,
    atr_sma: FilledSmaState,
    volume_sma: FilledSmaState,
    prev_close: f64,
    have_prev_close: bool,
}

impl NormalizedVolumeTrueRangeCore {
    #[inline]
    fn try_new(
        params: &NormalizedVolumeTrueRangeParams,
    ) -> Result<Self, NormalizedVolumeTrueRangeError> {
        let style = params.true_range_style.unwrap_or_default();
        let outlier_range = params.outlier_range.unwrap_or(DEFAULT_OUTLIER_RANGE);
        let atr_length = params.atr_length.unwrap_or(DEFAULT_ATR_LENGTH);
        let volume_length = params.volume_length.unwrap_or(DEFAULT_VOLUME_LENGTH);

        if !outlier_range.is_finite() || outlier_range < MIN_OUTLIER_RANGE {
            return Err(NormalizedVolumeTrueRangeError::InvalidOutlierRange { outlier_range });
        }
        if atr_length < MIN_LENGTH {
            return Err(NormalizedVolumeTrueRangeError::InvalidAtrLength { atr_length });
        }
        if volume_length < MIN_LENGTH {
            return Err(NormalizedVolumeTrueRangeError::InvalidVolumeLength { volume_length });
        }

        Ok(Self {
            style,
            outlier_range,
            abs_sum: 0.0,
            volume_sum: 0.0,
            count: 0,
            abs_positive_deviation: PositiveDeviationState::default(),
            volume_positive_deviation: PositiveDeviationState::default(),
            atr_sma: FilledSmaState::new(atr_length),
            volume_sma: FilledSmaState::new(volume_length),
            prev_close: f64::NAN,
            have_prev_close: false,
        })
    }

    #[inline(always)]
    fn update(
        &mut self,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Option<(f64, f64, f64, f64, f64)> {
        let valid = match self.style {
            NormalizedVolumeTrueRangeStyle::Body => {
                open.is_finite() && close.is_finite() && volume.is_finite()
            }
            NormalizedVolumeTrueRangeStyle::Hl => {
                high.is_finite() && low.is_finite() && volume.is_finite()
            }
            NormalizedVolumeTrueRangeStyle::Delta => close.is_finite() && volume.is_finite(),
        };
        if !valid {
            if close.is_finite() {
                self.prev_close = close;
                self.have_prev_close = true;
            }
            return None;
        }

        let prev_close = if self.have_prev_close {
            self.prev_close
        } else {
            close
        };
        let (start, finish) = match self.style {
            NormalizedVolumeTrueRangeStyle::Body => (open, close),
            NormalizedVolumeTrueRangeStyle::Hl => (low, high),
            NormalizedVolumeTrueRangeStyle::Delta => (prev_close, close),
        };

        self.prev_close = close;
        self.have_prev_close = true;

        let denom = start.min(finish);
        if !denom.is_finite() || denom <= 0.0 {
            return None;
        }

        let abs_percent = (finish - start).abs() / denom;
        if !abs_percent.is_finite() {
            return None;
        }

        self.count += 1;
        self.abs_sum += abs_percent;
        self.volume_sum += volume;

        let count_f64 = self.count as f64;
        let avg_abs_percent = self.abs_sum / count_f64;
        let avg_volume = self.volume_sum / count_f64;

        let p_stdev_abs_percent = self
            .abs_positive_deviation
            .update(abs_percent, avg_abs_percent);
        let p_stdev_volume = self.volume_positive_deviation.update(volume, avg_volume);

        let abs_percent_max = if p_stdev_abs_percent.is_finite() {
            avg_abs_percent + p_stdev_abs_percent * self.outlier_range
        } else {
            f64::NAN
        };

        let normalized_avg_percent = if abs_percent_max.is_finite() && abs_percent_max > 0.0 {
            avg_abs_percent / abs_percent_max
        } else {
            f64::NAN
        };

        let scale_factor = if normalized_avg_percent.is_finite()
            && normalized_avg_percent > 0.0
            && normalized_avg_percent < 1.0
            && p_stdev_volume.is_finite()
            && p_stdev_volume > 0.0
        {
            avg_volume * (1.0 - normalized_avg_percent) / (normalized_avg_percent * p_stdev_volume)
        } else {
            f64::NAN
        };

        let max_volume = if scale_factor.is_finite() && p_stdev_volume.is_finite() {
            avg_volume + p_stdev_volume * scale_factor
        } else {
            f64::NAN
        };

        let normalized_abs_percent = if abs_percent_max.is_finite() && abs_percent_max > 0.0 {
            abs_percent.min(abs_percent_max) / abs_percent_max
        } else {
            f64::NAN
        };

        let normalized_volume_ratio = if max_volume.is_finite() && max_volume > 0.0 {
            volume.min(max_volume) / max_volume
        } else {
            f64::NAN
        };

        let normalized_avg_volume_ratio = if max_volume.is_finite() && max_volume > 0.0 {
            avg_volume / max_volume
        } else {
            f64::NAN
        };

        let normalized_volume = normalized_volume_ratio * 100.0;
        let normalized_true_range = normalized_abs_percent * 100.0;
        let baseline = normalized_avg_volume_ratio * 100.0;
        let atr = self.atr_sma.update(normalized_true_range);
        let average_volume = self.volume_sma.update(normalized_volume);

        if !(normalized_volume.is_finite()
            && normalized_true_range.is_finite()
            && baseline.is_finite()
            && atr.is_finite()
            && average_volume.is_finite())
        {
            return None;
        }

        Some((
            normalized_volume,
            normalized_true_range,
            baseline,
            atr,
            average_volume,
        ))
    }
}

#[inline(always)]
fn normalize_single_kernel(kernel: Kernel) -> Kernel {
    match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    }
}

#[inline(always)]
fn normalize_single_kernel_to_scalar(kernel: Kernel) -> Kernel {
    match normalize_single_kernel(kernel) {
        Kernel::Auto
        | Kernel::Scalar
        | Kernel::ScalarBatch
        | Kernel::Avx2
        | Kernel::Avx2Batch
        | Kernel::Avx512
        | Kernel::Avx512Batch => Kernel::Scalar,
    }
}

#[inline(always)]
fn validate_params(
    params: &NormalizedVolumeTrueRangeParams,
) -> Result<(), NormalizedVolumeTrueRangeError> {
    let outlier_range = params.outlier_range.unwrap_or(DEFAULT_OUTLIER_RANGE);
    if !outlier_range.is_finite() || outlier_range < MIN_OUTLIER_RANGE {
        return Err(NormalizedVolumeTrueRangeError::InvalidOutlierRange { outlier_range });
    }

    let atr_length = params.atr_length.unwrap_or(DEFAULT_ATR_LENGTH);
    if atr_length < MIN_LENGTH {
        return Err(NormalizedVolumeTrueRangeError::InvalidAtrLength { atr_length });
    }

    let volume_length = params.volume_length.unwrap_or(DEFAULT_VOLUME_LENGTH);
    if volume_length < MIN_LENGTH {
        return Err(NormalizedVolumeTrueRangeError::InvalidVolumeLength { volume_length });
    }

    Ok(())
}

#[inline(always)]
fn prepare_input<'a>(
    input: &'a NormalizedVolumeTrueRangeInput<'a>,
) -> Result<PreparedNormalizedVolumeTrueRange<'a>, NormalizedVolumeTrueRangeError> {
    let (open, high, low, close, volume) = match &input.data {
        NormalizedVolumeTrueRangeData::Candles { candles } => (
            candles.open.as_slice(),
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
            candles.volume.as_slice(),
        ),
        NormalizedVolumeTrueRangeData::Slices {
            open,
            high,
            low,
            close,
            volume,
        } => {
            if open.len() != high.len()
                || high.len() != low.len()
                || low.len() != close.len()
                || close.len() != volume.len()
            {
                return Err(NormalizedVolumeTrueRangeError::InconsistentSliceLengths {
                    open_len: open.len(),
                    high_len: high.len(),
                    low_len: low.len(),
                    close_len: close.len(),
                    volume_len: volume.len(),
                });
            }
            (*open, *high, *low, *close, *volume)
        }
    };

    let len = close.len();
    if len == 0 {
        return Err(NormalizedVolumeTrueRangeError::EmptyInputData);
    }

    validate_params(&input.params)?;

    let style = input.get_true_range_style();
    let any_valid = (0..len).any(|idx| match style {
        NormalizedVolumeTrueRangeStyle::Body => {
            open[idx].is_finite() && close[idx].is_finite() && volume[idx].is_finite()
        }
        NormalizedVolumeTrueRangeStyle::Hl => {
            high[idx].is_finite() && low[idx].is_finite() && volume[idx].is_finite()
        }
        NormalizedVolumeTrueRangeStyle::Delta => close[idx].is_finite() && volume[idx].is_finite(),
    });
    if !any_valid {
        return Err(NormalizedVolumeTrueRangeError::AllValuesNaN);
    }

    Ok(PreparedNormalizedVolumeTrueRange {
        open,
        high,
        low,
        close,
        volume,
        len,
        style,
        outlier_range: input.get_outlier_range(),
        atr_length: input.get_atr_length(),
        volume_length: input.get_volume_length(),
    })
}

#[inline]
fn ensure_output_len(expected: usize, got: usize) -> Result<(), NormalizedVolumeTrueRangeError> {
    if expected == got {
        Ok(())
    } else {
        Err(NormalizedVolumeTrueRangeError::OutputLengthMismatch { expected, got })
    }
}

#[inline]
fn compute_into_slices(
    prepared: PreparedNormalizedVolumeTrueRange<'_>,
    normalized_volume: &mut [f64],
    normalized_true_range: &mut [f64],
    baseline: &mut [f64],
    atr: &mut [f64],
    average_volume: &mut [f64],
) -> Result<(), NormalizedVolumeTrueRangeError> {
    let len = prepared.len;
    ensure_output_len(len, normalized_volume.len())?;
    ensure_output_len(len, normalized_true_range.len())?;
    ensure_output_len(len, baseline.len())?;
    ensure_output_len(len, atr.len())?;
    ensure_output_len(len, average_volume.len())?;

    let mut core = NormalizedVolumeTrueRangeCore::try_new(&NormalizedVolumeTrueRangeParams {
        true_range_style: Some(prepared.style),
        outlier_range: Some(prepared.outlier_range),
        atr_length: Some(prepared.atr_length),
        volume_length: Some(prepared.volume_length),
    })?;

    for idx in 0..len {
        if let Some((nv, ntr, base, atr_value, avg_vol)) = core.update(
            prepared.open[idx],
            prepared.high[idx],
            prepared.low[idx],
            prepared.close[idx],
            prepared.volume[idx],
        ) {
            normalized_volume[idx] = nv;
            normalized_true_range[idx] = ntr;
            baseline[idx] = base;
            atr[idx] = atr_value;
            average_volume[idx] = avg_vol;
        } else {
            normalized_volume[idx] = f64::NAN;
            normalized_true_range[idx] = f64::NAN;
            baseline[idx] = f64::NAN;
            atr[idx] = f64::NAN;
            average_volume[idx] = f64::NAN;
        }
    }

    Ok(())
}

#[inline]
pub fn normalized_volume_true_range(
    input: &NormalizedVolumeTrueRangeInput<'_>,
) -> Result<NormalizedVolumeTrueRangeOutput, NormalizedVolumeTrueRangeError> {
    normalized_volume_true_range_with_kernel(input, Kernel::Auto)
}

pub fn normalized_volume_true_range_with_kernel(
    input: &NormalizedVolumeTrueRangeInput<'_>,
    kernel: Kernel,
) -> Result<NormalizedVolumeTrueRangeOutput, NormalizedVolumeTrueRangeError> {
    let _kernel = normalize_single_kernel_to_scalar(kernel);
    let prepared = prepare_input(input)?;
    let len = prepared.len;
    let mut normalized_volume = vec![0.0; len];
    let mut normalized_true_range = vec![0.0; len];
    let mut baseline = vec![0.0; len];
    let mut atr = vec![0.0; len];
    let mut average_volume = vec![0.0; len];
    compute_into_slices(
        prepared,
        &mut normalized_volume,
        &mut normalized_true_range,
        &mut baseline,
        &mut atr,
        &mut average_volume,
    )?;
    Ok(NormalizedVolumeTrueRangeOutput {
        normalized_volume,
        normalized_true_range,
        baseline,
        atr,
        average_volume,
    })
}

#[inline]
pub fn normalized_volume_true_range_into_slice(
    normalized_volume: &mut [f64],
    normalized_true_range: &mut [f64],
    baseline: &mut [f64],
    atr: &mut [f64],
    average_volume: &mut [f64],
    input: &NormalizedVolumeTrueRangeInput<'_>,
    kernel: Kernel,
) -> Result<(), NormalizedVolumeTrueRangeError> {
    let _kernel = normalize_single_kernel_to_scalar(kernel);
    let prepared = prepare_input(input)?;
    compute_into_slices(
        prepared,
        normalized_volume,
        normalized_true_range,
        baseline,
        atr,
        average_volume,
    )
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn normalized_volume_true_range_into(
    input: &NormalizedVolumeTrueRangeInput<'_>,
    normalized_volume: &mut [f64],
    normalized_true_range: &mut [f64],
    baseline: &mut [f64],
    atr: &mut [f64],
    average_volume: &mut [f64],
) -> Result<(), NormalizedVolumeTrueRangeError> {
    normalized_volume_true_range_into_slice(
        normalized_volume,
        normalized_true_range,
        baseline,
        atr,
        average_volume,
        input,
        Kernel::Auto,
    )
}

#[derive(Debug, Clone)]
pub struct NormalizedVolumeTrueRangeStream {
    params: NormalizedVolumeTrueRangeParams,
    core: NormalizedVolumeTrueRangeCore,
}

impl NormalizedVolumeTrueRangeStream {
    pub fn try_new(
        params: NormalizedVolumeTrueRangeParams,
    ) -> Result<Self, NormalizedVolumeTrueRangeError> {
        let core = NormalizedVolumeTrueRangeCore::try_new(&params)?;
        Ok(Self { params, core })
    }

    #[inline]
    pub fn update(
        &mut self,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Option<(f64, f64, f64, f64, f64)> {
        self.core.update(open, high, low, close, volume)
    }

    pub fn reset(&mut self) {
        self.core = NormalizedVolumeTrueRangeCore::try_new(&self.params)
            .expect("normalized_volume_true_range stream reset should revalidate existing params");
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NormalizedVolumeTrueRangeBatchRange {
    pub outlier_range: (f64, f64, f64),
    pub atr_length: (usize, usize, usize),
    pub volume_length: (usize, usize, usize),
    pub true_range_style: Option<NormalizedVolumeTrueRangeStyle>,
}

impl Default for NormalizedVolumeTrueRangeBatchRange {
    fn default() -> Self {
        Self {
            outlier_range: (DEFAULT_OUTLIER_RANGE, DEFAULT_OUTLIER_RANGE, 0.0),
            atr_length: (DEFAULT_ATR_LENGTH, DEFAULT_ATR_LENGTH, 0),
            volume_length: (DEFAULT_VOLUME_LENGTH, DEFAULT_VOLUME_LENGTH, 0),
            true_range_style: Some(NormalizedVolumeTrueRangeStyle::Body),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NormalizedVolumeTrueRangeBatchBuilder {
    range: NormalizedVolumeTrueRangeBatchRange,
    kernel: Kernel,
}

impl Default for NormalizedVolumeTrueRangeBatchBuilder {
    fn default() -> Self {
        Self {
            range: NormalizedVolumeTrueRangeBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl NormalizedVolumeTrueRangeBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn outlier_range_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.outlier_range = (start, end, step);
        self
    }

    pub fn outlier_range_static(mut self, outlier_range: f64) -> Self {
        self.range.outlier_range = (outlier_range, outlier_range, 0.0);
        self
    }

    pub fn atr_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.atr_length = (start, end, step);
        self
    }

    pub fn atr_length_static(mut self, atr_length: usize) -> Self {
        self.range.atr_length = (atr_length, atr_length, 0);
        self
    }

    pub fn volume_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.volume_length = (start, end, step);
        self
    }

    pub fn volume_length_static(mut self, volume_length: usize) -> Self {
        self.range.volume_length = (volume_length, volume_length, 0);
        self
    }

    pub fn true_range_style(mut self, style: NormalizedVolumeTrueRangeStyle) -> Self {
        self.range.true_range_style = Some(style);
        self
    }

    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    pub fn apply_slices(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Result<NormalizedVolumeTrueRangeBatchOutput, NormalizedVolumeTrueRangeError> {
        normalized_volume_true_range_batch_with_kernel(
            open,
            high,
            low,
            close,
            volume,
            &self.range,
            self.kernel,
        )
    }

    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<NormalizedVolumeTrueRangeBatchOutput, NormalizedVolumeTrueRangeError> {
        normalized_volume_true_range_batch_with_kernel(
            candles.open.as_slice(),
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
            candles.volume.as_slice(),
            &self.range,
            self.kernel,
        )
    }
}

#[derive(Debug, Clone)]
pub struct NormalizedVolumeTrueRangeBatchOutput {
    pub normalized_volume: Vec<f64>,
    pub normalized_true_range: Vec<f64>,
    pub baseline: Vec<f64>,
    pub atr: Vec<f64>,
    pub average_volume: Vec<f64>,
    pub combos: Vec<NormalizedVolumeTrueRangeParams>,
    pub rows: usize,
    pub cols: usize,
}

impl NormalizedVolumeTrueRangeBatchOutput {
    pub fn row_for_params(&self, params: &NormalizedVolumeTrueRangeParams) -> Option<usize> {
        let outlier = params.outlier_range.unwrap_or(DEFAULT_OUTLIER_RANGE);
        let atr_length = params.atr_length.unwrap_or(DEFAULT_ATR_LENGTH);
        let volume_length = params.volume_length.unwrap_or(DEFAULT_VOLUME_LENGTH);
        let style = params.true_range_style.unwrap_or_default();
        self.combos.iter().position(|combo| {
            (combo.outlier_range.unwrap_or(DEFAULT_OUTLIER_RANGE) - outlier).abs() <= 1e-12
                && combo.atr_length.unwrap_or(DEFAULT_ATR_LENGTH) == atr_length
                && combo.volume_length.unwrap_or(DEFAULT_VOLUME_LENGTH) == volume_length
                && combo.true_range_style.unwrap_or_default() == style
        })
    }
}

pub fn expand_grid_normalized_volume_true_range(
    range: &NormalizedVolumeTrueRangeBatchRange,
) -> Result<Vec<NormalizedVolumeTrueRangeParams>, NormalizedVolumeTrueRangeError> {
    fn float_axis(
        (start, end, step): (f64, f64, f64),
    ) -> Result<Vec<f64>, NormalizedVolumeTrueRangeError> {
        if !start.is_finite() || !end.is_finite() || !step.is_finite() {
            return Err(NormalizedVolumeTrueRangeError::InvalidOutlierRangeSweep {
                start,
                end,
                step,
            });
        }
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return Ok(vec![start]);
        }

        let mut values = Vec::new();
        let step_abs = step.abs();
        if start < end {
            let mut current = start;
            while current <= end + 1e-12 {
                values.push(current);
                current += step_abs;
            }
        } else {
            let mut current = start;
            while current + 1e-12 >= end {
                values.push(current);
                current -= step_abs;
            }
        }

        if values.is_empty() {
            return Err(NormalizedVolumeTrueRangeError::InvalidOutlierRangeSweep {
                start,
                end,
                step,
            });
        }
        Ok(values)
    }

    fn usize_axis(
        (start, end, step): (usize, usize, usize),
        make_error: fn(usize, usize, usize) -> NormalizedVolumeTrueRangeError,
    ) -> Result<Vec<usize>, NormalizedVolumeTrueRangeError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        let mut values = Vec::new();
        if start < end {
            let mut current = start;
            while current <= end {
                values.push(current);
                match current.checked_add(step) {
                    Some(next) => current = next,
                    None => break,
                }
            }
        } else {
            let mut current = start;
            while current >= end {
                values.push(current);
                if current < step {
                    break;
                }
                current -= step;
            }
        }
        if values.is_empty() {
            return Err(make_error(start, end, step));
        }
        Ok(values)
    }

    let styles = vec![range.true_range_style.unwrap_or_default()];
    let outlier_values = float_axis(range.outlier_range)?;
    let atr_values = usize_axis(range.atr_length, |start, end, step| {
        NormalizedVolumeTrueRangeError::InvalidAtrLengthSweep { start, end, step }
    })?;
    let volume_values = usize_axis(range.volume_length, |start, end, step| {
        NormalizedVolumeTrueRangeError::InvalidVolumeLengthSweep { start, end, step }
    })?;

    let mut combos = Vec::with_capacity(
        styles.len() * outlier_values.len() * atr_values.len() * volume_values.len(),
    );
    for style in styles {
        for &outlier_range in &outlier_values {
            for &atr_length in &atr_values {
                for &volume_length in &volume_values {
                    let params = NormalizedVolumeTrueRangeParams {
                        true_range_style: Some(style),
                        outlier_range: Some(outlier_range),
                        atr_length: Some(atr_length),
                        volume_length: Some(volume_length),
                    };
                    validate_params(&params)?;
                    combos.push(params);
                }
            }
        }
    }
    Ok(combos)
}

#[inline(always)]
pub fn normalized_volume_true_range_batch_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &NormalizedVolumeTrueRangeBatchRange,
) -> Result<NormalizedVolumeTrueRangeBatchOutput, NormalizedVolumeTrueRangeError> {
    normalized_volume_true_range_batch_inner(
        open,
        high,
        low,
        close,
        volume,
        sweep,
        Kernel::Scalar,
        false,
    )
}

#[inline(always)]
pub fn normalized_volume_true_range_batch_par_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &NormalizedVolumeTrueRangeBatchRange,
) -> Result<NormalizedVolumeTrueRangeBatchOutput, NormalizedVolumeTrueRangeError> {
    normalized_volume_true_range_batch_inner(
        open,
        high,
        low,
        close,
        volume,
        sweep,
        Kernel::Scalar,
        true,
    )
}

pub fn normalized_volume_true_range_batch_with_kernel(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &NormalizedVolumeTrueRangeBatchRange,
    kernel: Kernel,
) -> Result<NormalizedVolumeTrueRangeBatchOutput, NormalizedVolumeTrueRangeError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(NormalizedVolumeTrueRangeError::InvalidKernelForBatch(other)),
    };
    let scalar_kernel = match batch_kernel {
        Kernel::ScalarBatch => Kernel::Scalar,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::Avx512Batch => Kernel::Avx512,
        _ => unreachable!(),
    };
    normalized_volume_true_range_batch_inner(
        open,
        high,
        low,
        close,
        volume,
        sweep,
        scalar_kernel,
        !matches!(batch_kernel, Kernel::ScalarBatch),
    )
}

fn normalized_volume_true_range_batch_inner(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &NormalizedVolumeTrueRangeBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<NormalizedVolumeTrueRangeBatchOutput, NormalizedVolumeTrueRangeError> {
    let _kernel = normalize_single_kernel_to_scalar(kernel);
    let input = NormalizedVolumeTrueRangeInput::from_slices(
        open,
        high,
        low,
        close,
        volume,
        NormalizedVolumeTrueRangeParams {
            true_range_style: sweep.true_range_style,
            outlier_range: Some(sweep.outlier_range.0),
            atr_length: Some(sweep.atr_length.0),
            volume_length: Some(sweep.volume_length.0),
        },
    );
    let prepared = prepare_input(&input)?;
    let combos = expand_grid_normalized_volume_true_range(sweep)?;
    let rows = combos.len();
    let cols = prepared.len;

    let normalized_volume_mu = make_uninit_matrix(rows, cols);
    let normalized_true_range_mu = make_uninit_matrix(rows, cols);
    let baseline_mu = make_uninit_matrix(rows, cols);
    let atr_mu = make_uninit_matrix(rows, cols);
    let average_volume_mu = make_uninit_matrix(rows, cols);

    let mut normalized_volume_guard = ManuallyDrop::new(normalized_volume_mu);
    let mut normalized_true_range_guard = ManuallyDrop::new(normalized_true_range_mu);
    let mut baseline_guard = ManuallyDrop::new(baseline_mu);
    let mut atr_guard = ManuallyDrop::new(atr_mu);
    let mut average_volume_guard = ManuallyDrop::new(average_volume_mu);

    let normalized_volume_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(
            normalized_volume_guard.as_mut_ptr() as *mut f64,
            normalized_volume_guard.len(),
        )
    };
    let normalized_true_range_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(
            normalized_true_range_guard.as_mut_ptr() as *mut f64,
            normalized_true_range_guard.len(),
        )
    };
    let baseline_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(
            baseline_guard.as_mut_ptr() as *mut f64,
            baseline_guard.len(),
        )
    };
    let atr_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(atr_guard.as_mut_ptr() as *mut f64, atr_guard.len())
    };
    let average_volume_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(
            average_volume_guard.as_mut_ptr() as *mut f64,
            average_volume_guard.len(),
        )
    };

    let do_row = |row: usize,
                  normalized_volume_row: &mut [f64],
                  normalized_true_range_row: &mut [f64],
                  baseline_row: &mut [f64],
                  atr_row: &mut [f64],
                  average_volume_row: &mut [f64]| {
        let combo = &combos[row];
        let prepared_row = PreparedNormalizedVolumeTrueRange {
            open: prepared.open,
            high: prepared.high,
            low: prepared.low,
            close: prepared.close,
            volume: prepared.volume,
            len: cols,
            style: combo.true_range_style.unwrap_or_default(),
            outlier_range: combo.outlier_range.unwrap_or(DEFAULT_OUTLIER_RANGE),
            atr_length: combo.atr_length.unwrap_or(DEFAULT_ATR_LENGTH),
            volume_length: combo.volume_length.unwrap_or(DEFAULT_VOLUME_LENGTH),
        };
        compute_into_slices(
            prepared_row,
            normalized_volume_row,
            normalized_true_range_row,
            baseline_row,
            atr_row,
            average_volume_row,
        )
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            normalized_volume_out
                .par_chunks_mut(cols)
                .zip(normalized_true_range_out.par_chunks_mut(cols))
                .zip(baseline_out.par_chunks_mut(cols))
                .zip(atr_out.par_chunks_mut(cols))
                .zip(average_volume_out.par_chunks_mut(cols))
                .enumerate()
                .try_for_each(
                    |(
                        row,
                        (
                            (
                                ((normalized_volume_row, normalized_true_range_row), baseline_row),
                                atr_row,
                            ),
                            average_volume_row,
                        ),
                    )| {
                        do_row(
                            row,
                            normalized_volume_row,
                            normalized_true_range_row,
                            baseline_row,
                            atr_row,
                            average_volume_row,
                        )
                    },
                )?;
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (
                row,
                (
                    (((normalized_volume_row, normalized_true_range_row), baseline_row), atr_row),
                    average_volume_row,
                ),
            ) in normalized_volume_out
                .chunks_mut(cols)
                .zip(normalized_true_range_out.chunks_mut(cols))
                .zip(baseline_out.chunks_mut(cols))
                .zip(atr_out.chunks_mut(cols))
                .zip(average_volume_out.chunks_mut(cols))
                .enumerate()
            {
                do_row(
                    row,
                    normalized_volume_row,
                    normalized_true_range_row,
                    baseline_row,
                    atr_row,
                    average_volume_row,
                )?;
            }
        }
    } else {
        for (
            row,
            (
                (((normalized_volume_row, normalized_true_range_row), baseline_row), atr_row),
                average_volume_row,
            ),
        ) in normalized_volume_out
            .chunks_mut(cols)
            .zip(normalized_true_range_out.chunks_mut(cols))
            .zip(baseline_out.chunks_mut(cols))
            .zip(atr_out.chunks_mut(cols))
            .zip(average_volume_out.chunks_mut(cols))
            .enumerate()
        {
            do_row(
                row,
                normalized_volume_row,
                normalized_true_range_row,
                baseline_row,
                atr_row,
                average_volume_row,
            )?;
        }
    }

    let normalized_volume = unsafe {
        Vec::from_raw_parts(
            normalized_volume_guard.as_mut_ptr() as *mut f64,
            normalized_volume_guard.len(),
            normalized_volume_guard.capacity(),
        )
    };
    let normalized_true_range = unsafe {
        Vec::from_raw_parts(
            normalized_true_range_guard.as_mut_ptr() as *mut f64,
            normalized_true_range_guard.len(),
            normalized_true_range_guard.capacity(),
        )
    };
    let baseline = unsafe {
        Vec::from_raw_parts(
            baseline_guard.as_mut_ptr() as *mut f64,
            baseline_guard.len(),
            baseline_guard.capacity(),
        )
    };
    let atr = unsafe {
        Vec::from_raw_parts(
            atr_guard.as_mut_ptr() as *mut f64,
            atr_guard.len(),
            atr_guard.capacity(),
        )
    };
    let average_volume = unsafe {
        Vec::from_raw_parts(
            average_volume_guard.as_mut_ptr() as *mut f64,
            average_volume_guard.len(),
            average_volume_guard.capacity(),
        )
    };

    Ok(NormalizedVolumeTrueRangeBatchOutput {
        normalized_volume,
        normalized_true_range,
        baseline,
        atr,
        average_volume,
        combos,
        rows,
        cols,
    })
}

#[cfg(feature = "python")]
fn parse_style_py(style: Option<&str>) -> PyResult<Option<NormalizedVolumeTrueRangeStyle>> {
    match style {
        Some(value) => value
            .parse::<NormalizedVolumeTrueRangeStyle>()
            .map(Some)
            .map_err(PyValueError::new_err),
        None => Ok(None),
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "normalized_volume_true_range")]
#[pyo3(signature = (open, high, low, close, volume, true_range_style=None, outlier_range=None, atr_length=None, volume_length=None, *, kernel=None))]
pub fn normalized_volume_true_range_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    true_range_style: Option<&str>,
    outlier_range: Option<f64>,
    atr_length: Option<usize>,
    volume_length: Option<usize>,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let kernel = validate_kernel(kernel, false)?;
    let style = parse_style_py(true_range_style)?;
    let input = NormalizedVolumeTrueRangeInput::from_slices(
        open.as_slice()?,
        high.as_slice()?,
        low.as_slice()?,
        close.as_slice()?,
        volume.as_slice()?,
        NormalizedVolumeTrueRangeParams {
            true_range_style: style,
            outlier_range,
            atr_length,
            volume_length,
        },
    );
    let out = py
        .allow_threads(|| normalized_volume_true_range_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        out.normalized_volume.into_pyarray(py),
        out.normalized_true_range.into_pyarray(py),
        out.baseline.into_pyarray(py),
        out.atr.into_pyarray(py),
        out.average_volume.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyclass(name = "NormalizedVolumeTrueRangeStream")]
pub struct NormalizedVolumeTrueRangeStreamPy {
    inner: NormalizedVolumeTrueRangeStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl NormalizedVolumeTrueRangeStreamPy {
    #[new]
    #[pyo3(signature = (true_range_style=None, outlier_range=None, atr_length=None, volume_length=None))]
    pub fn new(
        true_range_style: Option<&str>,
        outlier_range: Option<f64>,
        atr_length: Option<usize>,
        volume_length: Option<usize>,
    ) -> PyResult<Self> {
        let inner = NormalizedVolumeTrueRangeStream::try_new(NormalizedVolumeTrueRangeParams {
            true_range_style: parse_style_py(true_range_style)?,
            outlier_range,
            atr_length,
            volume_length,
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    pub fn update(
        &mut self,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Option<(f64, f64, f64, f64, f64)> {
        self.inner.update(open, high, low, close, volume)
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "normalized_volume_true_range_batch")]
#[pyo3(signature = (open, high, low, close, volume, outlier_range_range=None, atr_length_range=None, volume_length_range=None, true_range_style=None, *, kernel=None))]
pub fn normalized_volume_true_range_batch_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    outlier_range_range: Option<(f64, f64, f64)>,
    atr_length_range: Option<(usize, usize, usize)>,
    volume_length_range: Option<(usize, usize, usize)>,
    true_range_style: Option<&str>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let kernel = validate_kernel(kernel, true)?;
    let style = parse_style_py(true_range_style)?;
    let out = normalized_volume_true_range_batch_with_kernel(
        open.as_slice()?,
        high.as_slice()?,
        low.as_slice()?,
        close.as_slice()?,
        volume.as_slice()?,
        &NormalizedVolumeTrueRangeBatchRange {
            outlier_range: outlier_range_range.unwrap_or((
                DEFAULT_OUTLIER_RANGE,
                DEFAULT_OUTLIER_RANGE,
                0.0,
            )),
            atr_length: atr_length_range.unwrap_or((DEFAULT_ATR_LENGTH, DEFAULT_ATR_LENGTH, 0)),
            volume_length: volume_length_range.unwrap_or((
                DEFAULT_VOLUME_LENGTH,
                DEFAULT_VOLUME_LENGTH,
                0,
            )),
            true_range_style: style,
        },
        kernel,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item(
        "normalized_volume",
        out.normalized_volume
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "normalized_true_range",
        out.normalized_true_range
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "baseline",
        out.baseline
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "atr",
        out.atr.into_pyarray(py).reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "average_volume",
        out.average_volume
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "outlier_ranges",
        out.combos
            .iter()
            .map(|combo| combo.outlier_range.unwrap_or(DEFAULT_OUTLIER_RANGE))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "atr_lengths",
        out.combos
            .iter()
            .map(|combo| combo.atr_length.unwrap_or(DEFAULT_ATR_LENGTH))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "volume_lengths",
        out.combos
            .iter()
            .map(|combo| combo.volume_length.unwrap_or(DEFAULT_VOLUME_LENGTH))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "true_range_styles",
        PyList::new(
            py,
            out.combos
                .iter()
                .map(|combo| combo.true_range_style.unwrap_or_default().as_str()),
        )?,
    )?;
    dict.set_item("rows", out.rows)?;
    dict.set_item("cols", out.cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_normalized_volume_true_range_module(
    m: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(normalized_volume_true_range_py, m)?)?;
    m.add_function(wrap_pyfunction!(normalized_volume_true_range_batch_py, m)?)?;
    m.add_class::<NormalizedVolumeTrueRangeStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
fn parse_style_js(
    style: Option<String>,
) -> Result<Option<NormalizedVolumeTrueRangeStyle>, JsValue> {
    match style {
        Some(value) => value
            .parse::<NormalizedVolumeTrueRangeStyle>()
            .map(Some)
            .map_err(|e| JsValue::from_str(&e)),
        None => Ok(None),
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
struct NormalizedVolumeTrueRangeJsOutput {
    normalized_volume: Vec<f64>,
    normalized_true_range: Vec<f64>,
    baseline: Vec<f64>,
    atr: Vec<f64>,
    average_volume: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
struct NormalizedVolumeTrueRangeStreamJsOutput {
    normalized_volume: f64,
    normalized_true_range: f64,
    baseline: f64,
    atr: f64,
    average_volume: f64,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct NormalizedVolumeTrueRangeBatchConfig {
    pub true_range_style: Option<String>,
    pub outlier_range_range: (f64, f64, f64),
    pub atr_length_range: (usize, usize, usize),
    pub volume_length_range: (usize, usize, usize),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct NormalizedVolumeTrueRangeBatchJsOutput {
    pub normalized_volume: Vec<f64>,
    pub normalized_true_range: Vec<f64>,
    pub baseline: Vec<f64>,
    pub atr: Vec<f64>,
    pub average_volume: Vec<f64>,
    pub combos: Vec<NormalizedVolumeTrueRangeParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = normalized_volume_true_range_js)]
pub fn normalized_volume_true_range_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    true_range_style: Option<String>,
    outlier_range: Option<f64>,
    atr_length: Option<usize>,
    volume_length: Option<usize>,
) -> Result<JsValue, JsValue> {
    let input = NormalizedVolumeTrueRangeInput::from_slices(
        open,
        high,
        low,
        close,
        volume,
        NormalizedVolumeTrueRangeParams {
            true_range_style: parse_style_js(true_range_style)?,
            outlier_range,
            atr_length,
            volume_length,
        },
    );
    let out =
        normalized_volume_true_range(&input).map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&NormalizedVolumeTrueRangeJsOutput {
        normalized_volume: out.normalized_volume,
        normalized_true_range: out.normalized_true_range,
        baseline: out.baseline,
        atr: out.atr,
        average_volume: out.average_volume,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = normalized_volume_true_range_batch)]
pub fn normalized_volume_true_range_batch_unified_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: NormalizedVolumeTrueRangeBatchConfig =
        serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let out = normalized_volume_true_range_batch_with_kernel(
        open,
        high,
        low,
        close,
        volume,
        &NormalizedVolumeTrueRangeBatchRange {
            true_range_style: parse_style_js(config.true_range_style)?,
            outlier_range: config.outlier_range_range,
            atr_length: config.atr_length_range,
            volume_length: config.volume_length_range,
        },
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&NormalizedVolumeTrueRangeBatchJsOutput {
        normalized_volume: out.normalized_volume,
        normalized_true_range: out.normalized_true_range,
        baseline: out.baseline,
        atr: out.atr,
        average_volume: out.average_volume,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = normalized_volume_true_range_alloc)]
pub fn normalized_volume_true_range_alloc(len: usize) -> *mut f64 {
    let mut values = vec![0.0; len];
    let ptr = values.as_mut_ptr();
    std::mem::forget(values);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = normalized_volume_true_range_free)]
pub fn normalized_volume_true_range_free(ptr: *mut f64, len: usize) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        drop(Vec::from_raw_parts(ptr, 0, len));
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = normalized_volume_true_range_into)]
pub fn normalized_volume_true_range_into(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    volume_ptr: *const f64,
    normalized_volume_ptr: *mut f64,
    normalized_true_range_ptr: *mut f64,
    baseline_ptr: *mut f64,
    atr_ptr: *mut f64,
    average_volume_ptr: *mut f64,
    len: usize,
    true_range_style: Option<String>,
    outlier_range: Option<f64>,
    atr_length: Option<usize>,
    volume_length: Option<usize>,
) -> Result<(), JsValue> {
    if open_ptr.is_null()
        || high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || volume_ptr.is_null()
        || normalized_volume_ptr.is_null()
        || normalized_true_range_ptr.is_null()
        || baseline_ptr.is_null()
        || atr_ptr.is_null()
        || average_volume_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to normalized_volume_true_range_into",
        ));
    }

    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);
        let input = NormalizedVolumeTrueRangeInput::from_slices(
            open,
            high,
            low,
            close,
            volume,
            NormalizedVolumeTrueRangeParams {
                true_range_style: parse_style_js(true_range_style)?,
                outlier_range,
                atr_length,
                volume_length,
            },
        );

        let input_alias = [
            normalized_volume_ptr as *const f64,
            normalized_true_range_ptr as *const f64,
            baseline_ptr as *const f64,
            atr_ptr as *const f64,
            average_volume_ptr as *const f64,
        ]
        .iter()
        .any(|&ptr| {
            ptr == open_ptr
                || ptr == high_ptr
                || ptr == low_ptr
                || ptr == close_ptr
                || ptr == volume_ptr
        });
        let output_alias = normalized_volume_ptr == normalized_true_range_ptr
            || normalized_volume_ptr == baseline_ptr
            || normalized_volume_ptr == atr_ptr
            || normalized_volume_ptr == average_volume_ptr
            || normalized_true_range_ptr == baseline_ptr
            || normalized_true_range_ptr == atr_ptr
            || normalized_true_range_ptr == average_volume_ptr
            || baseline_ptr == atr_ptr
            || baseline_ptr == average_volume_ptr
            || atr_ptr == average_volume_ptr;

        if input_alias || output_alias {
            let mut normalized_volume = vec![0.0; len];
            let mut normalized_true_range = vec![0.0; len];
            let mut baseline = vec![0.0; len];
            let mut atr = vec![0.0; len];
            let mut average_volume = vec![0.0; len];
            normalized_volume_true_range_into_slice(
                &mut normalized_volume,
                &mut normalized_true_range,
                &mut baseline,
                &mut atr,
                &mut average_volume,
                &input,
                Kernel::Auto,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
            std::slice::from_raw_parts_mut(normalized_volume_ptr, len)
                .copy_from_slice(&normalized_volume);
            std::slice::from_raw_parts_mut(normalized_true_range_ptr, len)
                .copy_from_slice(&normalized_true_range);
            std::slice::from_raw_parts_mut(baseline_ptr, len).copy_from_slice(&baseline);
            std::slice::from_raw_parts_mut(atr_ptr, len).copy_from_slice(&atr);
            std::slice::from_raw_parts_mut(average_volume_ptr, len)
                .copy_from_slice(&average_volume);
            return Ok(());
        }

        normalized_volume_true_range_into_slice(
            std::slice::from_raw_parts_mut(normalized_volume_ptr, len),
            std::slice::from_raw_parts_mut(normalized_true_range_ptr, len),
            std::slice::from_raw_parts_mut(baseline_ptr, len),
            std::slice::from_raw_parts_mut(atr_ptr, len),
            std::slice::from_raw_parts_mut(average_volume_ptr, len),
            &input,
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = normalized_volume_true_range_batch_into)]
pub fn normalized_volume_true_range_batch_into(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    volume_ptr: *const f64,
    normalized_volume_ptr: *mut f64,
    normalized_true_range_ptr: *mut f64,
    baseline_ptr: *mut f64,
    atr_ptr: *mut f64,
    average_volume_ptr: *mut f64,
    len: usize,
    outlier_range_start: f64,
    outlier_range_end: f64,
    outlier_range_step: f64,
    atr_length_start: usize,
    atr_length_end: usize,
    atr_length_step: usize,
    volume_length_start: usize,
    volume_length_end: usize,
    volume_length_step: usize,
    true_range_style: Option<String>,
) -> Result<usize, JsValue> {
    if open_ptr.is_null()
        || high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || volume_ptr.is_null()
        || normalized_volume_ptr.is_null()
        || normalized_true_range_ptr.is_null()
        || baseline_ptr.is_null()
        || atr_ptr.is_null()
        || average_volume_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to normalized_volume_true_range_batch_into",
        ));
    }

    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);
        let sweep = NormalizedVolumeTrueRangeBatchRange {
            true_range_style: parse_style_js(true_range_style)?,
            outlier_range: (outlier_range_start, outlier_range_end, outlier_range_step),
            atr_length: (atr_length_start, atr_length_end, atr_length_step),
            volume_length: (volume_length_start, volume_length_end, volume_length_step),
        };
        let combos = expand_grid_normalized_volume_true_range(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let total = rows
            .checked_mul(len)
            .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;

        let input_alias = [
            normalized_volume_ptr as *const f64,
            normalized_true_range_ptr as *const f64,
            baseline_ptr as *const f64,
            atr_ptr as *const f64,
            average_volume_ptr as *const f64,
        ]
        .iter()
        .any(|&ptr| {
            ptr == open_ptr
                || ptr == high_ptr
                || ptr == low_ptr
                || ptr == close_ptr
                || ptr == volume_ptr
        });
        let output_alias = normalized_volume_ptr == normalized_true_range_ptr
            || normalized_volume_ptr == baseline_ptr
            || normalized_volume_ptr == atr_ptr
            || normalized_volume_ptr == average_volume_ptr
            || normalized_true_range_ptr == baseline_ptr
            || normalized_true_range_ptr == atr_ptr
            || normalized_true_range_ptr == average_volume_ptr
            || baseline_ptr == atr_ptr
            || baseline_ptr == average_volume_ptr
            || atr_ptr == average_volume_ptr;

        let out = normalized_volume_true_range_batch_with_kernel(
            open,
            high,
            low,
            close,
            volume,
            &sweep,
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

        if input_alias || output_alias {
            std::slice::from_raw_parts_mut(normalized_volume_ptr, total)
                .copy_from_slice(&out.normalized_volume);
            std::slice::from_raw_parts_mut(normalized_true_range_ptr, total)
                .copy_from_slice(&out.normalized_true_range);
            std::slice::from_raw_parts_mut(baseline_ptr, total).copy_from_slice(&out.baseline);
            std::slice::from_raw_parts_mut(atr_ptr, total).copy_from_slice(&out.atr);
            std::slice::from_raw_parts_mut(average_volume_ptr, total)
                .copy_from_slice(&out.average_volume);
            return Ok(rows);
        }

        std::slice::from_raw_parts_mut(normalized_volume_ptr, total)
            .copy_from_slice(&out.normalized_volume);
        std::slice::from_raw_parts_mut(normalized_true_range_ptr, total)
            .copy_from_slice(&out.normalized_true_range);
        std::slice::from_raw_parts_mut(baseline_ptr, total).copy_from_slice(&out.baseline);
        std::slice::from_raw_parts_mut(atr_ptr, total).copy_from_slice(&out.atr);
        std::slice::from_raw_parts_mut(average_volume_ptr, total)
            .copy_from_slice(&out.average_volume);
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub struct NormalizedVolumeTrueRangeStreamWasm {
    inner: NormalizedVolumeTrueRangeStream,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
impl NormalizedVolumeTrueRangeStreamWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(
        true_range_style: Option<String>,
        outlier_range: Option<f64>,
        atr_length: Option<usize>,
        volume_length: Option<usize>,
    ) -> Result<NormalizedVolumeTrueRangeStreamWasm, JsValue> {
        Ok(Self {
            inner: NormalizedVolumeTrueRangeStream::try_new(NormalizedVolumeTrueRangeParams {
                true_range_style: parse_style_js(true_range_style)?,
                outlier_range,
                atr_length,
                volume_length,
            })
            .map_err(|e| JsValue::from_str(&e.to_string()))?,
        })
    }

    pub fn update(
        &mut self,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Result<JsValue, JsValue> {
        match self.inner.update(open, high, low, close, volume) {
            Some((normalized_volume, normalized_true_range, baseline, atr, average_volume)) => {
                serde_wasm_bindgen::to_value(&NormalizedVolumeTrueRangeStreamJsOutput {
                    normalized_volume,
                    normalized_true_range,
                    baseline,
                    atr,
                    average_volume,
                })
                .map_err(|e| JsValue::from_str(&e.to_string()))
            }
            None => Ok(JsValue::NULL),
        }
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn normalized_volume_true_range_output_into_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    true_range_style: Option<String>,
    outlier_range: Option<f64>,
    atr_length: Option<usize>,
    volume_length: Option<usize>,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = normalized_volume_true_range_js(
        open,
        high,
        low,
        close,
        volume,
        true_range_style,
        outlier_range,
        atr_length,
        volume_length,
    )?;
    crate::write_wasm_object_f64_outputs("normalized_volume_true_range_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn normalized_volume_true_range_batch_unified_output_into_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value =
        normalized_volume_true_range_batch_unified_js(open, high, low, close, volume, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "normalized_volume_true_range_batch_unified_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use std::error::Error;

    fn naive_reference(
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
        params: &NormalizedVolumeTrueRangeParams,
    ) -> NormalizedVolumeTrueRangeOutput {
        let len = close.len();
        let mut normalized_volume = vec![f64::NAN; len];
        let mut normalized_true_range = vec![f64::NAN; len];
        let mut baseline = vec![f64::NAN; len];
        let mut atr = vec![f64::NAN; len];
        let mut average_volume = vec![f64::NAN; len];
        let mut stream = NormalizedVolumeTrueRangeStream::try_new(params.clone()).unwrap();
        for idx in 0..len {
            if let Some((nv, ntr, base, atr_value, avg_vol)) =
                stream.update(open[idx], high[idx], low[idx], close[idx], volume[idx])
            {
                normalized_volume[idx] = nv;
                normalized_true_range[idx] = ntr;
                baseline[idx] = base;
                atr[idx] = atr_value;
                average_volume[idx] = avg_vol;
            }
        }
        NormalizedVolumeTrueRangeOutput {
            normalized_volume,
            normalized_true_range,
            baseline,
            atr,
            average_volume,
        }
    }

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (lhs, rhs) in actual.iter().zip(expected.iter()) {
            assert!((lhs.is_nan() && rhs.is_nan()) || (lhs - rhs).abs() <= 1e-12);
        }
    }

    fn sample_ohlcv() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let len = 256;
        let mut open = Vec::with_capacity(len);
        let mut high = Vec::with_capacity(len);
        let mut low = Vec::with_capacity(len);
        let mut close = Vec::with_capacity(len);
        let mut volume = Vec::with_capacity(len);
        let mut prev_close = 100.0;
        for idx in 0..len {
            let x = idx as f64;
            let drift = x * 0.015;
            let body = (x * 0.11).sin() * 1.6;
            let spread = 2.0 + (x * 0.07).cos().abs();
            let open_value = prev_close + (x * 0.03).sin() * 0.6;
            let close_value = 100.0 + drift + body;
            let high_value = open_value.max(close_value) + spread * 0.55;
            let low_value = open_value.min(close_value) - spread * 0.45;
            let volume_value = 1_000_000.0 + (x * 0.17).sin().abs() * 350_000.0 + x * 800.0;
            open.push(open_value);
            high.push(high_value);
            low.push(low_value);
            close.push(close_value);
            volume.push(volume_value);
            prev_close = close_value;
        }
        (open, high, low, close, volume)
    }

    #[test]
    fn normalized_volume_true_range_matches_naive_body() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close, volume) = sample_ohlcv();
        let params = NormalizedVolumeTrueRangeParams::default();
        let input = NormalizedVolumeTrueRangeInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            &volume,
            params.clone(),
        );
        let actual = normalized_volume_true_range(&input)?;
        let expected = naive_reference(&open, &high, &low, &close, &volume, &params);
        assert_close(&actual.normalized_volume, &expected.normalized_volume);
        assert_close(
            &actual.normalized_true_range,
            &expected.normalized_true_range,
        );
        assert_close(&actual.baseline, &expected.baseline);
        assert_close(&actual.atr, &expected.atr);
        assert_close(&actual.average_volume, &expected.average_volume);
        Ok(())
    }

    #[test]
    fn normalized_volume_true_range_into_matches_api() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close, volume) = sample_ohlcv();
        let input = NormalizedVolumeTrueRangeInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            &volume,
            NormalizedVolumeTrueRangeParams {
                true_range_style: Some(NormalizedVolumeTrueRangeStyle::Delta),
                outlier_range: Some(4.5),
                atr_length: Some(10),
                volume_length: Some(7),
            },
        );
        let expected = normalized_volume_true_range(&input)?;
        let mut normalized_volume = vec![0.0; close.len()];
        let mut normalized_true_range = vec![0.0; close.len()];
        let mut baseline = vec![0.0; close.len()];
        let mut atr = vec![0.0; close.len()];
        let mut average_volume = vec![0.0; close.len()];
        normalized_volume_true_range_into(
            &input,
            &mut normalized_volume,
            &mut normalized_true_range,
            &mut baseline,
            &mut atr,
            &mut average_volume,
        )?;
        assert_close(&normalized_volume, &expected.normalized_volume);
        assert_close(&normalized_true_range, &expected.normalized_true_range);
        assert_close(&baseline, &expected.baseline);
        assert_close(&atr, &expected.atr);
        assert_close(&average_volume, &expected.average_volume);
        Ok(())
    }

    #[test]
    fn normalized_volume_true_range_batch_matches_single() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close, volume) = sample_ohlcv();
        let batch = normalized_volume_true_range_batch_with_kernel(
            &open,
            &high,
            &low,
            &close,
            &volume,
            &NormalizedVolumeTrueRangeBatchRange {
                true_range_style: Some(NormalizedVolumeTrueRangeStyle::Body),
                outlier_range: (4.0, 5.0, 1.0),
                atr_length: (8, 10, 2),
                volume_length: (5, 5, 0),
            },
            Kernel::ScalarBatch,
        )?;
        for (row, combo) in batch.combos.iter().enumerate() {
            let single =
                normalized_volume_true_range(&NormalizedVolumeTrueRangeInput::from_slices(
                    &open,
                    &high,
                    &low,
                    &close,
                    &volume,
                    combo.clone(),
                ))?;
            let start = row * batch.cols;
            let end = start + batch.cols;
            assert_close(
                &batch.normalized_volume[start..end],
                &single.normalized_volume,
            );
            assert_close(
                &batch.normalized_true_range[start..end],
                &single.normalized_true_range,
            );
            assert_close(&batch.baseline[start..end], &single.baseline);
            assert_close(&batch.atr[start..end], &single.atr);
            assert_close(&batch.average_volume[start..end], &single.average_volume);
        }
        Ok(())
    }

    #[test]
    fn normalized_volume_true_range_fixture_has_values() -> Result<(), Box<dyn Error>> {
        let candles = read_candles_from_csv("src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv")?;
        let out = normalized_volume_true_range(
            &NormalizedVolumeTrueRangeInput::with_default_candles(&candles),
        )?;
        assert_eq!(out.normalized_volume.len(), candles.close.len());
        assert!(out
            .normalized_volume
            .iter()
            .skip(32)
            .any(|value| value.is_finite()));
        assert!(out
            .normalized_true_range
            .iter()
            .skip(32)
            .any(|value| value.is_finite()));
        assert!(out.baseline.iter().skip(32).any(|value| value.is_finite()));
        assert!(out.atr.iter().skip(32).any(|value| value.is_finite()));
        assert!(out
            .average_volume
            .iter()
            .skip(32)
            .any(|value| value.is_finite()));
        Ok(())
    }
}
