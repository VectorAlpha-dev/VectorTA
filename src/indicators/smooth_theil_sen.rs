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
    alloc_with_nan_prefix, detect_best_batch_kernel, init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::collections::VecDeque;
use std::mem::ManuallyDrop;
use std::str::FromStr;
use thiserror::Error;

const DEFAULT_SOURCE: &str = "close";
const DEFAULT_LENGTH: usize = 25;
const DEFAULT_OFFSET: usize = 0;
const DEFAULT_MULTIPLIER: f64 = 2.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize),
    serde(rename_all = "snake_case")
)]
pub enum SmoothTheilSenStatStyle {
    Mean,
    SmoothMedian,
    Median,
}

impl Default for SmoothTheilSenStatStyle {
    fn default() -> Self {
        Self::SmoothMedian
    }
}

impl SmoothTheilSenStatStyle {
    #[inline(always)]
    fn blend(self) -> f64 {
        match self {
            Self::Mean => 1.0,
            Self::SmoothMedian => 0.5,
            Self::Median => 0.0,
        }
    }

    #[inline(always)]
    fn as_str(self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::SmoothMedian => "smooth_median",
            Self::Median => "median",
        }
    }
}

impl FromStr for SmoothTheilSenStatStyle {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let normalized = value.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "mean" => Ok(Self::Mean),
            "smooth_median" | "smooth median" => Ok(Self::SmoothMedian),
            "median" => Ok(Self::Median),
            _ => Err(format!("invalid stat_style: {value}")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize),
    serde(rename_all = "snake_case")
)]
pub enum SmoothTheilSenDeviationType {
    Mad,
    Rmsd,
}

impl Default for SmoothTheilSenDeviationType {
    fn default() -> Self {
        Self::Mad
    }
}

impl SmoothTheilSenDeviationType {
    #[inline(always)]
    fn as_str(self) -> &'static str {
        match self {
            Self::Mad => "mad",
            Self::Rmsd => "rmsd",
        }
    }
}

impl FromStr for SmoothTheilSenDeviationType {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let normalized = value.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "mad" | "median_absolute_deviation" => Ok(Self::Mad),
            "rmsd" | "root_mean_square_deviation" => Ok(Self::Rmsd),
            _ => Err(format!("invalid deviation_style: {value}")),
        }
    }
}

#[derive(Debug, Clone)]
pub enum SmoothTheilSenData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct SmoothTheilSenOutput {
    pub value: Vec<f64>,
    pub upper: Vec<f64>,
    pub lower: Vec<f64>,
    pub slope: Vec<f64>,
    pub intercept: Vec<f64>,
    pub deviation: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct SmoothTheilSenParams {
    pub length: Option<usize>,
    pub offset: Option<usize>,
    pub multiplier: Option<f64>,
    pub slope_style: Option<SmoothTheilSenStatStyle>,
    pub residual_style: Option<SmoothTheilSenStatStyle>,
    pub deviation_style: Option<SmoothTheilSenDeviationType>,
    pub mad_style: Option<SmoothTheilSenStatStyle>,
    pub include_prediction_in_deviation: Option<bool>,
}

impl Default for SmoothTheilSenParams {
    fn default() -> Self {
        Self {
            length: Some(DEFAULT_LENGTH),
            offset: Some(DEFAULT_OFFSET),
            multiplier: Some(DEFAULT_MULTIPLIER),
            slope_style: Some(SmoothTheilSenStatStyle::SmoothMedian),
            residual_style: Some(SmoothTheilSenStatStyle::SmoothMedian),
            deviation_style: Some(SmoothTheilSenDeviationType::Mad),
            mad_style: Some(SmoothTheilSenStatStyle::SmoothMedian),
            include_prediction_in_deviation: Some(false),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SmoothTheilSenInput<'a> {
    pub data: SmoothTheilSenData<'a>,
    pub params: SmoothTheilSenParams,
}

impl<'a> SmoothTheilSenInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: SmoothTheilSenParams,
    ) -> Self {
        Self {
            data: SmoothTheilSenData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(data: &'a [f64], params: SmoothTheilSenParams) -> Self {
        Self {
            data: SmoothTheilSenData::Slice(data),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, DEFAULT_SOURCE, SmoothTheilSenParams::default())
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SmoothTheilSenBuilder {
    source: Option<&'static str>,
    length: Option<usize>,
    offset: Option<usize>,
    multiplier: Option<f64>,
    slope_style: Option<SmoothTheilSenStatStyle>,
    residual_style: Option<SmoothTheilSenStatStyle>,
    deviation_style: Option<SmoothTheilSenDeviationType>,
    mad_style: Option<SmoothTheilSenStatStyle>,
    include_prediction_in_deviation: Option<bool>,
    kernel: Kernel,
}

impl Default for SmoothTheilSenBuilder {
    fn default() -> Self {
        Self {
            source: None,
            length: None,
            offset: None,
            multiplier: None,
            slope_style: None,
            residual_style: None,
            deviation_style: None,
            mad_style: None,
            include_prediction_in_deviation: None,
            kernel: Kernel::Auto,
        }
    }
}

impl SmoothTheilSenBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn source(mut self, value: &'static str) -> Self {
        self.source = Some(value);
        self
    }

    #[inline(always)]
    pub fn length(mut self, value: usize) -> Self {
        self.length = Some(value);
        self
    }

    #[inline(always)]
    pub fn offset(mut self, value: usize) -> Self {
        self.offset = Some(value);
        self
    }

    #[inline(always)]
    pub fn multiplier(mut self, value: f64) -> Self {
        self.multiplier = Some(value);
        self
    }

    #[inline(always)]
    pub fn slope_style(mut self, value: SmoothTheilSenStatStyle) -> Self {
        self.slope_style = Some(value);
        self
    }

    #[inline(always)]
    pub fn residual_style(mut self, value: SmoothTheilSenStatStyle) -> Self {
        self.residual_style = Some(value);
        self
    }

    #[inline(always)]
    pub fn deviation_style(mut self, value: SmoothTheilSenDeviationType) -> Self {
        self.deviation_style = Some(value);
        self
    }

    #[inline(always)]
    pub fn mad_style(mut self, value: SmoothTheilSenStatStyle) -> Self {
        self.mad_style = Some(value);
        self
    }

    #[inline(always)]
    pub fn include_prediction_in_deviation(mut self, value: bool) -> Self {
        self.include_prediction_in_deviation = Some(value);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline(always)]
    pub fn apply(self, candles: &Candles) -> Result<SmoothTheilSenOutput, SmoothTheilSenError> {
        let input = SmoothTheilSenInput::from_candles(
            candles,
            self.source.unwrap_or(DEFAULT_SOURCE),
            SmoothTheilSenParams {
                length: self.length,
                offset: self.offset,
                multiplier: self.multiplier,
                slope_style: self.slope_style,
                residual_style: self.residual_style,
                deviation_style: self.deviation_style,
                mad_style: self.mad_style,
                include_prediction_in_deviation: self.include_prediction_in_deviation,
            },
        );
        smooth_theil_sen_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, data: &[f64]) -> Result<SmoothTheilSenOutput, SmoothTheilSenError> {
        let input = SmoothTheilSenInput::from_slice(
            data,
            SmoothTheilSenParams {
                length: self.length,
                offset: self.offset,
                multiplier: self.multiplier,
                slope_style: self.slope_style,
                residual_style: self.residual_style,
                deviation_style: self.deviation_style,
                mad_style: self.mad_style,
                include_prediction_in_deviation: self.include_prediction_in_deviation,
            },
        );
        smooth_theil_sen_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<SmoothTheilSenStream, SmoothTheilSenError> {
        SmoothTheilSenStream::try_new(SmoothTheilSenParams {
            length: self.length,
            offset: self.offset,
            multiplier: self.multiplier,
            slope_style: self.slope_style,
            residual_style: self.residual_style,
            deviation_style: self.deviation_style,
            mad_style: self.mad_style,
            include_prediction_in_deviation: self.include_prediction_in_deviation,
        })
    }
}

#[derive(Debug, Error)]
pub enum SmoothTheilSenError {
    #[error("smooth_theil_sen: Input data slice is empty.")]
    EmptyInputData,
    #[error("smooth_theil_sen: All values are NaN.")]
    AllValuesNaN,
    #[error("smooth_theil_sen: Invalid length: {length}")]
    InvalidLength { length: usize },
    #[error("smooth_theil_sen: Invalid multiplier: {multiplier}")]
    InvalidMultiplier { multiplier: f64 },
    #[error("smooth_theil_sen: Invalid source: {source_name}")]
    InvalidSource { source_name: String },
    #[error("smooth_theil_sen: Not enough valid data: needed={needed}, valid={valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("smooth_theil_sen: Output length mismatch: expected={expected}, got={got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("smooth_theil_sen: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("smooth_theil_sen: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Clone, Copy, Debug)]
struct ResolvedParams {
    length: usize,
    offset: usize,
    multiplier: f64,
    slope_style: SmoothTheilSenStatStyle,
    residual_style: SmoothTheilSenStatStyle,
    deviation_style: SmoothTheilSenDeviationType,
    mad_style: SmoothTheilSenStatStyle,
    include_prediction_in_deviation: bool,
}

#[derive(Clone, Debug)]
struct KernelCache {
    slope_weights: Option<Vec<f64>>,
    residual_weights: Option<Vec<f64>>,
    error_weights: Option<Vec<f64>>,
}

#[derive(Clone, Debug)]
struct WorkBuffers {
    slopes: Vec<f64>,
    residuals: Vec<f64>,
    errors: Vec<f64>,
}

#[derive(Clone, Copy, Debug)]
struct PointOutput {
    value: f64,
    upper: f64,
    lower: f64,
    slope: f64,
    intercept: f64,
    deviation: f64,
}

#[inline(always)]
fn extract_slice<'a>(input: &'a SmoothTheilSenInput<'a>) -> Result<&'a [f64], SmoothTheilSenError> {
    let data = match &input.data {
        SmoothTheilSenData::Candles { candles, source } => {
            let source_values = match *source {
                "open" => &candles.open,
                "high" => &candles.high,
                "low" => &candles.low,
                "close" => &candles.close,
                "volume" => &candles.volume,
                "hl2" => &candles.hl2,
                "hlc3" => &candles.hlc3,
                "ohlc4" => &candles.ohlc4,
                "hlcc4" | "hlcc" => &candles.hlcc4,
                _ => source_type(candles, source),
            };
            if source_values.is_empty() {
                return Err(SmoothTheilSenError::InvalidSource {
                    source_name: (*source).to_string(),
                });
            }
            source_values
        }
        SmoothTheilSenData::Slice(data) => *data,
    };
    if data.is_empty() {
        return Err(SmoothTheilSenError::EmptyInputData);
    }
    Ok(data)
}

#[inline(always)]
fn first_valid(data: &[f64]) -> Option<usize> {
    data.iter().position(|v| v.is_finite())
}

#[inline(always)]
fn resolve_params(params: &SmoothTheilSenParams) -> Result<ResolvedParams, SmoothTheilSenError> {
    let length = params.length.unwrap_or(DEFAULT_LENGTH);
    if length < 2 {
        return Err(SmoothTheilSenError::InvalidLength { length });
    }
    let multiplier = params.multiplier.unwrap_or(DEFAULT_MULTIPLIER);
    if !multiplier.is_finite() || multiplier < 0.0 {
        return Err(SmoothTheilSenError::InvalidMultiplier { multiplier });
    }
    Ok(ResolvedParams {
        length,
        offset: params.offset.unwrap_or(DEFAULT_OFFSET),
        multiplier,
        slope_style: params.slope_style.unwrap_or_default(),
        residual_style: params.residual_style.unwrap_or_default(),
        deviation_style: params.deviation_style.unwrap_or_default(),
        mad_style: params.mad_style.unwrap_or_default(),
        include_prediction_in_deviation: params.include_prediction_in_deviation.unwrap_or(false),
    })
}

#[inline(always)]
fn warmup_bars(params: &ResolvedParams) -> usize {
    params.length + params.offset - 1
}

#[inline(always)]
fn validate_input<'a>(
    input: &'a SmoothTheilSenInput<'a>,
    kernel: Kernel,
) -> Result<(&'a [f64], ResolvedParams, usize, usize, Kernel), SmoothTheilSenError> {
    let data = extract_slice(input)?;
    let params = resolve_params(&input.params)?;
    let first = first_valid(data).ok_or(SmoothTheilSenError::AllValuesNaN)?;
    let needed = params.length + params.offset;
    let valid = data.len().saturating_sub(first);
    if valid < needed {
        return Err(SmoothTheilSenError::NotEnoughValidData { needed, valid });
    }
    Ok((
        data,
        params,
        first,
        first + warmup_bars(&params),
        kernel.to_non_batch(),
    ))
}

#[inline(always)]
fn pairwise_count(length: usize) -> usize {
    length * (length - 1) / 2
}

#[inline(always)]
fn exponential_interpolation(k: f64, endpoint: f64) -> f64 {
    let clamp = k.clamp(0.0, 1.0);
    let min = 0.5;
    (endpoint - min) * 1024.0f64.powf(clamp - 1.0) + min
}

#[inline(always)]
fn gaussian_kernel(source: f64, bandwidth: f64) -> f64 {
    let ratio = source / bandwidth;
    (-(ratio * ratio) / 4.0).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

fn kernel_weights(size: usize, style: SmoothTheilSenStatStyle) -> Option<Vec<f64>> {
    if style != SmoothTheilSenStatStyle::SmoothMedian || size == 0 {
        return None;
    }
    let width = exponential_interpolation(style.blend(), size as f64);
    let center = (size as f64 - 1.0) * 0.5;
    let mut weights = Vec::with_capacity(size);
    let mut normalization = 0.0;
    for i in 0..size {
        let position = i as f64 - center;
        let weight = gaussian_kernel(position, width);
        weights.push(weight);
        normalization += weight;
    }
    if normalization != 0.0 {
        for weight in &mut weights {
            *weight /= normalization;
        }
    }
    Some(weights)
}

#[inline(always)]
fn median_in_place(values: &mut [f64]) -> f64 {
    values.sort_unstable_by(f64::total_cmp);
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) * 0.5
    }
}

#[inline(always)]
fn estimator(values: &mut [f64], style: SmoothTheilSenStatStyle, weights: Option<&[f64]>) -> f64 {
    match style {
        SmoothTheilSenStatStyle::Mean => values.iter().sum::<f64>() / values.len() as f64,
        SmoothTheilSenStatStyle::Median => median_in_place(values),
        SmoothTheilSenStatStyle::SmoothMedian => {
            values.sort_unstable_by(f64::total_cmp);
            values
                .iter()
                .zip(weights.expect("smooth_median requires weights"))
                .map(|(value, weight)| value * weight)
                .sum()
        }
    }
}

#[inline(always)]
fn build_kernel_cache(params: &ResolvedParams) -> KernelCache {
    let error_len =
        params.length + usize::from(params.include_prediction_in_deviation) * params.offset;
    KernelCache {
        slope_weights: kernel_weights(pairwise_count(params.length), params.slope_style),
        residual_weights: kernel_weights(params.length, params.residual_style),
        error_weights: kernel_weights(error_len, params.mad_style),
    }
}

#[inline(always)]
fn create_work_buffers(params: &ResolvedParams) -> WorkBuffers {
    WorkBuffers {
        slopes: Vec::with_capacity(pairwise_count(params.length)),
        residuals: Vec::with_capacity(params.length),
        errors: Vec::with_capacity(
            params.length + usize::from(params.include_prediction_in_deviation) * params.offset,
        ),
    }
}

struct DefaultWorkBuffers {
    slopes: [f64; 300],
    residuals: [f64; 25],
    errors: [f64; 25],
}

impl Default for DefaultWorkBuffers {
    #[inline(always)]
    fn default() -> Self {
        Self {
            slopes: [0.0; 300],
            residuals: [0.0; 25],
            errors: [0.0; 25],
        }
    }
}

#[inline(always)]
fn is_default_params(params: &ResolvedParams) -> bool {
    params.length == DEFAULT_LENGTH
        && params.offset == DEFAULT_OFFSET
        && params.multiplier == DEFAULT_MULTIPLIER
        && params.slope_style == SmoothTheilSenStatStyle::SmoothMedian
        && params.residual_style == SmoothTheilSenStatStyle::SmoothMedian
        && params.deviation_style == SmoothTheilSenDeviationType::Mad
        && params.mad_style == SmoothTheilSenStatStyle::SmoothMedian
        && !params.include_prediction_in_deviation
}

#[inline(always)]
fn output_len_check(out: &[f64], len: usize) -> Result<(), SmoothTheilSenError> {
    if out.len() != len {
        return Err(SmoothTheilSenError::OutputLengthMismatch {
            expected: len,
            got: out.len(),
        });
    }
    Ok(())
}

#[inline(always)]
fn fill_prefixes(
    warmup: usize,
    out_value: &mut [f64],
    out_upper: &mut [f64],
    out_lower: &mut [f64],
    out_slope: &mut [f64],
    out_intercept: &mut [f64],
    out_deviation: &mut [f64],
) {
    let qnan = f64::NAN;
    out_value[..warmup].fill(qnan);
    out_upper[..warmup].fill(qnan);
    out_lower[..warmup].fill(qnan);
    out_slope[..warmup].fill(qnan);
    out_intercept[..warmup].fill(qnan);
    out_deviation[..warmup].fill(qnan);
}

#[inline(always)]
fn required_finite_segment(
    data: &[f64],
    idx: usize,
    params: &ResolvedParams,
) -> Option<(usize, usize)> {
    let base = idx.checked_sub(params.offset)?;
    let start = base + 1 - params.length;
    let end = if params.include_prediction_in_deviation {
        idx
    } else {
        base
    };
    Some((start, end))
}

#[inline(always)]
fn segment_all_finite(data: &[f64], start: usize, end: usize) -> bool {
    data[start..=end].iter().all(|value| value.is_finite())
}

#[inline(always)]
fn smooth_weighted_sorted(values: &mut [f64], weights: &[f64]) -> f64 {
    values.sort_unstable_by(f64::total_cmp);
    values
        .iter()
        .zip(weights)
        .map(|(value, weight)| value * weight)
        .sum()
}

fn compute_default_point(
    data: &[f64],
    idx: usize,
    cache: &KernelCache,
    work: &mut DefaultWorkBuffers,
) -> PointOutput {
    let base = idx;
    let mut n = 0usize;
    for i in 0..24 {
        let value_i = data[base - i];
        for j in i + 1..25 {
            let value_j = data[base - j];
            work.slopes[n] = (value_j - value_i) / (j - i) as f64;
            n += 1;
        }
    }
    let beta_1 = smooth_weighted_sorted(
        &mut work.slopes,
        cache
            .slope_weights
            .as_deref()
            .expect("default slope weights"),
    );

    for j in 0..25 {
        work.residuals[j] = data[base - j] - beta_1 * j as f64;
    }
    let beta_0 = smooth_weighted_sorted(
        &mut work.residuals,
        cache
            .residual_weights
            .as_deref()
            .expect("default residual weights"),
    );

    let predicted = beta_0;
    for point in 0..25 {
        let predicted_point = beta_0 + beta_1 * point as f64;
        work.errors[point] = (data[idx - point] - predicted_point).abs();
    }
    let deviation = smooth_weighted_sorted(
        &mut work.errors,
        cache
            .error_weights
            .as_deref()
            .expect("default error weights"),
    ) * DEFAULT_MULTIPLIER;

    PointOutput {
        value: predicted,
        upper: predicted + deviation,
        lower: predicted - deviation,
        slope: beta_1,
        intercept: beta_0,
        deviation,
    }
}

fn compute_point(
    data: &[f64],
    idx: usize,
    params: &ResolvedParams,
    cache: &KernelCache,
    work: &mut WorkBuffers,
) -> Option<PointOutput> {
    let (start, end) = required_finite_segment(data, idx, params)?;
    if !segment_all_finite(data, start, end) {
        return None;
    }

    let base = idx - params.offset;
    work.slopes.clear();
    for i in 0..params.length - 1 {
        let value_i = data[base - i];
        for j in i + 1..params.length {
            let value_j = data[base - j];
            work.slopes.push((value_j - value_i) / (j - i) as f64);
        }
    }
    let beta_1 = estimator(
        work.slopes.as_mut_slice(),
        params.slope_style,
        cache.slope_weights.as_deref(),
    );

    work.residuals.clear();
    for j in 0..params.length {
        work.residuals.push(data[base - j] - beta_1 * j as f64);
    }
    let beta_0 = estimator(
        work.residuals.as_mut_slice(),
        params.residual_style,
        cache.residual_weights.as_deref(),
    );

    let predicted = beta_0 - beta_1 * params.offset as f64;

    let deviation = match params.deviation_style {
        SmoothTheilSenDeviationType::Mad => {
            work.errors.clear();
            let start_point = if params.include_prediction_in_deviation {
                -(params.offset as isize)
            } else {
                0
            };
            for point in start_point..=(params.length as isize - 1) {
                let source_idx = (idx as isize - params.offset as isize - point) as usize;
                let predicted_point = beta_0 + beta_1 * point as f64;
                work.errors.push((data[source_idx] - predicted_point).abs());
            }
            estimator(
                work.errors.as_mut_slice(),
                params.mad_style,
                cache.error_weights.as_deref(),
            ) * params.multiplier
        }
        SmoothTheilSenDeviationType::Rmsd => {
            let start_point = if params.include_prediction_in_deviation {
                -(params.offset as isize)
            } else {
                0
            };
            let mut square_errors = 0.0;
            let mut count = 0usize;
            for point in start_point..=(params.length as isize - 1) {
                let source_idx = (idx as isize - params.offset as isize - point) as usize;
                let predicted_point = beta_0 + beta_1 * point as f64;
                let error = data[source_idx] - predicted_point;
                square_errors += error * error;
                count += 1;
            }
            square_errors
                .sqrt()
                .mul_add(params.multiplier / (count as f64).sqrt(), 0.0)
        }
    };

    Some(PointOutput {
        value: predicted,
        upper: predicted + deviation,
        lower: predicted - deviation,
        slope: beta_1,
        intercept: beta_0,
        deviation,
    })
}

fn smooth_theil_sen_default_all_finite_into(
    data: &[f64],
    params: &ResolvedParams,
    warmup: usize,
    out_value: &mut [f64],
    out_upper: &mut [f64],
    out_lower: &mut [f64],
    out_slope: &mut [f64],
    out_intercept: &mut [f64],
    out_deviation: &mut [f64],
) {
    let cache = build_kernel_cache(params);
    let mut work = DefaultWorkBuffers::default();
    for idx in warmup..data.len() {
        let point = compute_default_point(data, idx, &cache, &mut work);
        out_value[idx] = point.value;
        out_upper[idx] = point.upper;
        out_lower[idx] = point.lower;
        out_slope[idx] = point.slope;
        out_intercept[idx] = point.intercept;
        out_deviation[idx] = point.deviation;
    }
}

fn smooth_theil_sen_compute_into(
    data: &[f64],
    params: &ResolvedParams,
    warmup: usize,
    out_value: &mut [f64],
    out_upper: &mut [f64],
    out_lower: &mut [f64],
    out_slope: &mut [f64],
    out_intercept: &mut [f64],
    out_deviation: &mut [f64],
) -> Result<(), SmoothTheilSenError> {
    let len = data.len();
    output_len_check(out_value, len)?;
    output_len_check(out_upper, len)?;
    output_len_check(out_lower, len)?;
    output_len_check(out_slope, len)?;
    output_len_check(out_intercept, len)?;
    output_len_check(out_deviation, len)?;

    if is_default_params(params) && data.iter().all(|value| value.is_finite()) {
        smooth_theil_sen_default_all_finite_into(
            data,
            params,
            warmup,
            out_value,
            out_upper,
            out_lower,
            out_slope,
            out_intercept,
            out_deviation,
        );
        return Ok(());
    }

    let cache = build_kernel_cache(params);
    let mut work = create_work_buffers(params);
    for idx in warmup..len {
        let Some(point) = compute_point(data, idx, params, &cache, &mut work) else {
            continue;
        };
        out_value[idx] = point.value;
        out_upper[idx] = point.upper;
        out_lower[idx] = point.lower;
        out_slope[idx] = point.slope;
        out_intercept[idx] = point.intercept;
        out_deviation[idx] = point.deviation;
    }
    Ok(())
}

#[inline]
pub fn smooth_theil_sen(
    input: &SmoothTheilSenInput,
) -> Result<SmoothTheilSenOutput, SmoothTheilSenError> {
    smooth_theil_sen_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn smooth_theil_sen_with_kernel(
    input: &SmoothTheilSenInput,
    kernel: Kernel,
) -> Result<SmoothTheilSenOutput, SmoothTheilSenError> {
    let (data, params, _first, warmup, _kernel) = validate_input(input, kernel)?;
    let mut value = alloc_with_nan_prefix(data.len(), warmup);
    let mut upper = alloc_with_nan_prefix(data.len(), warmup);
    let mut lower = alloc_with_nan_prefix(data.len(), warmup);
    let mut slope = alloc_with_nan_prefix(data.len(), warmup);
    let mut intercept = alloc_with_nan_prefix(data.len(), warmup);
    let mut deviation = alloc_with_nan_prefix(data.len(), warmup);
    smooth_theil_sen_compute_into(
        data,
        &params,
        warmup,
        &mut value,
        &mut upper,
        &mut lower,
        &mut slope,
        &mut intercept,
        &mut deviation,
    )?;
    Ok(SmoothTheilSenOutput {
        value,
        upper,
        lower,
        slope,
        intercept,
        deviation,
    })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn smooth_theil_sen_into(
    out_value: &mut [f64],
    out_upper: &mut [f64],
    out_lower: &mut [f64],
    out_slope: &mut [f64],
    out_intercept: &mut [f64],
    out_deviation: &mut [f64],
    input: &SmoothTheilSenInput,
    kernel: Kernel,
) -> Result<(), SmoothTheilSenError> {
    smooth_theil_sen_into_slice(
        out_value,
        out_upper,
        out_lower,
        out_slope,
        out_intercept,
        out_deviation,
        input,
        kernel,
    )
}

#[inline]
pub fn smooth_theil_sen_into_slice(
    out_value: &mut [f64],
    out_upper: &mut [f64],
    out_lower: &mut [f64],
    out_slope: &mut [f64],
    out_intercept: &mut [f64],
    out_deviation: &mut [f64],
    input: &SmoothTheilSenInput,
    kernel: Kernel,
) -> Result<(), SmoothTheilSenError> {
    let (data, params, _first, warmup, _kernel) = validate_input(input, kernel)?;
    let len = data.len();
    output_len_check(out_value, len)?;
    output_len_check(out_upper, len)?;
    output_len_check(out_lower, len)?;
    output_len_check(out_slope, len)?;
    output_len_check(out_intercept, len)?;
    output_len_check(out_deviation, len)?;

    if is_default_params(&params) && data.iter().all(|value| value.is_finite()) {
        fill_prefixes(
            warmup,
            out_value,
            out_upper,
            out_lower,
            out_slope,
            out_intercept,
            out_deviation,
        );
        smooth_theil_sen_default_all_finite_into(
            data,
            &params,
            warmup,
            out_value,
            out_upper,
            out_lower,
            out_slope,
            out_intercept,
            out_deviation,
        );
        return Ok(());
    }

    out_value.fill(f64::NAN);
    out_upper.fill(f64::NAN);
    out_lower.fill(f64::NAN);
    out_slope.fill(f64::NAN);
    out_intercept.fill(f64::NAN);
    out_deviation.fill(f64::NAN);
    smooth_theil_sen_compute_into(
        data,
        &params,
        warmup,
        out_value,
        out_upper,
        out_lower,
        out_slope,
        out_intercept,
        out_deviation,
    )
}

#[derive(Clone, Debug)]
pub struct SmoothTheilSenStream {
    params: ResolvedParams,
    window: VecDeque<f64>,
}

impl SmoothTheilSenStream {
    pub fn try_new(params: SmoothTheilSenParams) -> Result<Self, SmoothTheilSenError> {
        Ok(Self {
            params: resolve_params(&params)?,
            window: VecDeque::new(),
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> (f64, f64, f64, f64, f64, f64) {
        let needed = self.params.length + self.params.offset;
        self.window.push_back(value);
        if self.window.len() > needed {
            self.window.pop_front();
        }
        if self.window.len() < needed {
            return (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN);
        }
        let buffer: Vec<f64> = self.window.iter().copied().collect();
        let cache = build_kernel_cache(&self.params);
        let mut work = create_work_buffers(&self.params);
        let Some(point) = compute_point(&buffer, buffer.len() - 1, &self.params, &cache, &mut work)
        else {
            return (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN);
        };
        (
            point.value,
            point.upper,
            point.lower,
            point.slope,
            point.intercept,
            point.deviation,
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SmoothTheilSenBatchRange {
    pub length: (usize, usize, usize),
    pub offset: (usize, usize, usize),
    pub multiplier: (f64, f64, f64),
    pub slope_style: SmoothTheilSenStatStyle,
    pub residual_style: SmoothTheilSenStatStyle,
    pub deviation_style: SmoothTheilSenDeviationType,
    pub mad_style: SmoothTheilSenStatStyle,
    pub include_prediction_in_deviation: bool,
}

impl Default for SmoothTheilSenBatchRange {
    fn default() -> Self {
        Self {
            length: (DEFAULT_LENGTH, DEFAULT_LENGTH, 0),
            offset: (DEFAULT_OFFSET, DEFAULT_OFFSET, 0),
            multiplier: (DEFAULT_MULTIPLIER, DEFAULT_MULTIPLIER, 0.0),
            slope_style: SmoothTheilSenStatStyle::SmoothMedian,
            residual_style: SmoothTheilSenStatStyle::SmoothMedian,
            deviation_style: SmoothTheilSenDeviationType::Mad,
            mad_style: SmoothTheilSenStatStyle::SmoothMedian,
            include_prediction_in_deviation: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SmoothTheilSenBatchOutput {
    pub value: Vec<f64>,
    pub upper: Vec<f64>,
    pub lower: Vec<f64>,
    pub slope: Vec<f64>,
    pub intercept: Vec<f64>,
    pub deviation: Vec<f64>,
    pub combos: Vec<SmoothTheilSenParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Copy, Clone, Debug)]
pub struct SmoothTheilSenBatchBuilder {
    source: Option<&'static str>,
    range: SmoothTheilSenBatchRange,
    kernel: Kernel,
}

impl Default for SmoothTheilSenBatchBuilder {
    fn default() -> Self {
        Self {
            source: None,
            range: SmoothTheilSenBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl SmoothTheilSenBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn source(mut self, value: &'static str) -> Self {
        self.source = Some(value);
        self
    }

    #[inline(always)]
    pub fn length_range(mut self, value: (usize, usize, usize)) -> Self {
        self.range.length = value;
        self
    }

    #[inline(always)]
    pub fn offset_range(mut self, value: (usize, usize, usize)) -> Self {
        self.range.offset = value;
        self
    }

    #[inline(always)]
    pub fn multiplier_range(mut self, value: (f64, f64, f64)) -> Self {
        self.range.multiplier = value;
        self
    }

    #[inline(always)]
    pub fn slope_style(mut self, value: SmoothTheilSenStatStyle) -> Self {
        self.range.slope_style = value;
        self
    }

    #[inline(always)]
    pub fn residual_style(mut self, value: SmoothTheilSenStatStyle) -> Self {
        self.range.residual_style = value;
        self
    }

    #[inline(always)]
    pub fn deviation_style(mut self, value: SmoothTheilSenDeviationType) -> Self {
        self.range.deviation_style = value;
        self
    }

    #[inline(always)]
    pub fn mad_style(mut self, value: SmoothTheilSenStatStyle) -> Self {
        self.range.mad_style = value;
        self
    }

    #[inline(always)]
    pub fn include_prediction_in_deviation(mut self, value: bool) -> Self {
        self.range.include_prediction_in_deviation = value;
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
    ) -> Result<SmoothTheilSenBatchOutput, SmoothTheilSenError> {
        smooth_theil_sen_batch_with_kernel(
            source_type(candles, self.source.unwrap_or(DEFAULT_SOURCE)),
            &self.range,
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<SmoothTheilSenBatchOutput, SmoothTheilSenError> {
        smooth_theil_sen_batch_with_kernel(data, &self.range, self.kernel)
    }
}

#[inline(always)]
fn expand_usize_range(
    start: usize,
    end: usize,
    step: usize,
) -> Result<Vec<usize>, SmoothTheilSenError> {
    if step == 0 {
        if start != end {
            return Err(SmoothTheilSenError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        return Ok(vec![start]);
    }
    if start > end {
        return Err(SmoothTheilSenError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    let mut out = Vec::new();
    let mut current = start;
    while current <= end {
        out.push(current);
        if out.len() > 1_000_000 {
            return Err(SmoothTheilSenError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        current = current.saturating_add(step);
        if current == usize::MAX && current != end {
            break;
        }
    }
    Ok(out)
}

#[inline(always)]
fn expand_float_range(start: f64, end: f64, step: f64) -> Result<Vec<f64>, SmoothTheilSenError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(SmoothTheilSenError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if step == 0.0 {
        if (start - end).abs() > 1e-12 {
            return Err(SmoothTheilSenError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        return Ok(vec![start]);
    }
    if start > end || step < 0.0 {
        return Err(SmoothTheilSenError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    let mut out = Vec::new();
    let mut current = start;
    while current <= end + 1e-12 {
        out.push(current);
        if out.len() > 1_000_000 {
            return Err(SmoothTheilSenError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        current += step;
    }
    Ok(out)
}

pub fn smooth_theil_sen_expand_grid(
    sweep: &SmoothTheilSenBatchRange,
) -> Result<Vec<SmoothTheilSenParams>, SmoothTheilSenError> {
    let lengths = expand_usize_range(sweep.length.0, sweep.length.1, sweep.length.2)?;
    let offsets = expand_usize_range(sweep.offset.0, sweep.offset.1, sweep.offset.2)?;
    let multipliers =
        expand_float_range(sweep.multiplier.0, sweep.multiplier.1, sweep.multiplier.2)?;
    let mut out = Vec::with_capacity(lengths.len() * offsets.len() * multipliers.len());
    for length in lengths {
        for offset in &offsets {
            for multiplier in &multipliers {
                out.push(SmoothTheilSenParams {
                    length: Some(length),
                    offset: Some(*offset),
                    multiplier: Some(*multiplier),
                    slope_style: Some(sweep.slope_style),
                    residual_style: Some(sweep.residual_style),
                    deviation_style: Some(sweep.deviation_style),
                    mad_style: Some(sweep.mad_style),
                    include_prediction_in_deviation: Some(sweep.include_prediction_in_deviation),
                });
            }
        }
    }
    Ok(out)
}

#[inline(always)]
fn batch_shape(rows: usize, cols: usize) -> Result<usize, SmoothTheilSenError> {
    rows.checked_mul(cols)
        .ok_or_else(|| SmoothTheilSenError::InvalidRange {
            start: rows.to_string(),
            end: cols.to_string(),
            step: "rows*cols".to_string(),
        })
}

fn validate_raw_data(data: &[f64]) -> Result<usize, SmoothTheilSenError> {
    if data.is_empty() {
        return Err(SmoothTheilSenError::EmptyInputData);
    }
    first_valid(data).ok_or(SmoothTheilSenError::AllValuesNaN)
}

pub fn smooth_theil_sen_batch_with_kernel(
    data: &[f64],
    sweep: &SmoothTheilSenBatchRange,
    kernel: Kernel,
) -> Result<SmoothTheilSenBatchOutput, SmoothTheilSenError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(SmoothTheilSenError::InvalidKernelForBatch(kernel)),
    };
    smooth_theil_sen_batch_par_slice(data, sweep, batch_kernel.to_non_batch())
}

#[inline(always)]
pub fn smooth_theil_sen_batch_slice(
    data: &[f64],
    sweep: &SmoothTheilSenBatchRange,
    kernel: Kernel,
) -> Result<SmoothTheilSenBatchOutput, SmoothTheilSenError> {
    smooth_theil_sen_batch_inner(data, sweep, kernel, false)
}

#[inline(always)]
pub fn smooth_theil_sen_batch_par_slice(
    data: &[f64],
    sweep: &SmoothTheilSenBatchRange,
    kernel: Kernel,
) -> Result<SmoothTheilSenBatchOutput, SmoothTheilSenError> {
    smooth_theil_sen_batch_inner(data, sweep, kernel, true)
}

fn smooth_theil_sen_batch_inner(
    data: &[f64],
    sweep: &SmoothTheilSenBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<SmoothTheilSenBatchOutput, SmoothTheilSenError> {
    let combos = smooth_theil_sen_expand_grid(sweep)?;
    let first = validate_raw_data(data)?;
    let rows = combos.len();
    let cols = data.len();
    let total = batch_shape(rows, cols)?;
    let mut warmups = Vec::with_capacity(rows);
    for combo in &combos {
        let params = resolve_params(combo)?;
        let needed = params.length + params.offset;
        let valid = data.len().saturating_sub(first);
        if valid < needed {
            return Err(SmoothTheilSenError::NotEnoughValidData { needed, valid });
        }
        warmups.push(first + warmup_bars(&params));
    }

    let mut value_buf = make_uninit_matrix(rows, cols);
    let mut upper_buf = make_uninit_matrix(rows, cols);
    let mut lower_buf = make_uninit_matrix(rows, cols);
    let mut slope_buf = make_uninit_matrix(rows, cols);
    let mut intercept_buf = make_uninit_matrix(rows, cols);
    let mut deviation_buf = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut value_buf, cols, &warmups);
    init_matrix_prefixes(&mut upper_buf, cols, &warmups);
    init_matrix_prefixes(&mut lower_buf, cols, &warmups);
    init_matrix_prefixes(&mut slope_buf, cols, &warmups);
    init_matrix_prefixes(&mut intercept_buf, cols, &warmups);
    init_matrix_prefixes(&mut deviation_buf, cols, &warmups);

    let mut value_guard = ManuallyDrop::new(value_buf);
    let mut upper_guard = ManuallyDrop::new(upper_buf);
    let mut lower_guard = ManuallyDrop::new(lower_buf);
    let mut slope_guard = ManuallyDrop::new(slope_buf);
    let mut intercept_guard = ManuallyDrop::new(intercept_buf);
    let mut deviation_guard = ManuallyDrop::new(deviation_buf);

    let out_value: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(value_guard.as_mut_ptr() as *mut f64, value_guard.len())
    };
    let out_upper: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(upper_guard.as_mut_ptr() as *mut f64, upper_guard.len())
    };
    let out_lower: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(lower_guard.as_mut_ptr() as *mut f64, lower_guard.len())
    };
    let out_slope: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(slope_guard.as_mut_ptr() as *mut f64, slope_guard.len())
    };
    let out_intercept: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(
            intercept_guard.as_mut_ptr() as *mut f64,
            intercept_guard.len(),
        )
    };
    let out_deviation: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(
            deviation_guard.as_mut_ptr() as *mut f64,
            deviation_guard.len(),
        )
    };

    smooth_theil_sen_batch_inner_into(
        data,
        sweep,
        kernel,
        parallel,
        out_value,
        out_upper,
        out_lower,
        out_slope,
        out_intercept,
        out_deviation,
    )?;

    let value = unsafe {
        Vec::from_raw_parts(
            value_guard.as_mut_ptr() as *mut f64,
            total,
            value_guard.capacity(),
        )
    };
    let upper = unsafe {
        Vec::from_raw_parts(
            upper_guard.as_mut_ptr() as *mut f64,
            total,
            upper_guard.capacity(),
        )
    };
    let lower = unsafe {
        Vec::from_raw_parts(
            lower_guard.as_mut_ptr() as *mut f64,
            total,
            lower_guard.capacity(),
        )
    };
    let slope = unsafe {
        Vec::from_raw_parts(
            slope_guard.as_mut_ptr() as *mut f64,
            total,
            slope_guard.capacity(),
        )
    };
    let intercept = unsafe {
        Vec::from_raw_parts(
            intercept_guard.as_mut_ptr() as *mut f64,
            total,
            intercept_guard.capacity(),
        )
    };
    let deviation = unsafe {
        Vec::from_raw_parts(
            deviation_guard.as_mut_ptr() as *mut f64,
            total,
            deviation_guard.capacity(),
        )
    };

    Ok(SmoothTheilSenBatchOutput {
        value,
        upper,
        lower,
        slope,
        intercept,
        deviation,
        combos,
        rows,
        cols,
    })
}

pub fn smooth_theil_sen_batch_into_slice(
    out_value: &mut [f64],
    out_upper: &mut [f64],
    out_lower: &mut [f64],
    out_slope: &mut [f64],
    out_intercept: &mut [f64],
    out_deviation: &mut [f64],
    data: &[f64],
    sweep: &SmoothTheilSenBatchRange,
    kernel: Kernel,
) -> Result<(), SmoothTheilSenError> {
    smooth_theil_sen_batch_inner_into(
        data,
        sweep,
        kernel,
        false,
        out_value,
        out_upper,
        out_lower,
        out_slope,
        out_intercept,
        out_deviation,
    )?;
    Ok(())
}

fn smooth_theil_sen_batch_inner_into(
    data: &[f64],
    sweep: &SmoothTheilSenBatchRange,
    _kernel: Kernel,
    parallel: bool,
    out_value: &mut [f64],
    out_upper: &mut [f64],
    out_lower: &mut [f64],
    out_slope: &mut [f64],
    out_intercept: &mut [f64],
    out_deviation: &mut [f64],
) -> Result<Vec<SmoothTheilSenParams>, SmoothTheilSenError> {
    let combos = smooth_theil_sen_expand_grid(sweep)?;
    let first = validate_raw_data(data)?;
    let rows = combos.len();
    let cols = data.len();
    let total = batch_shape(rows, cols)?;
    if out_value.len() != total {
        return Err(SmoothTheilSenError::OutputLengthMismatch {
            expected: total,
            got: out_value.len(),
        });
    }
    if out_upper.len() != total {
        return Err(SmoothTheilSenError::OutputLengthMismatch {
            expected: total,
            got: out_upper.len(),
        });
    }
    if out_lower.len() != total {
        return Err(SmoothTheilSenError::OutputLengthMismatch {
            expected: total,
            got: out_lower.len(),
        });
    }
    if out_slope.len() != total {
        return Err(SmoothTheilSenError::OutputLengthMismatch {
            expected: total,
            got: out_slope.len(),
        });
    }
    if out_intercept.len() != total {
        return Err(SmoothTheilSenError::OutputLengthMismatch {
            expected: total,
            got: out_intercept.len(),
        });
    }
    if out_deviation.len() != total {
        return Err(SmoothTheilSenError::OutputLengthMismatch {
            expected: total,
            got: out_deviation.len(),
        });
    }

    out_value.fill(f64::NAN);
    out_upper.fill(f64::NAN);
    out_lower.fill(f64::NAN);
    out_slope.fill(f64::NAN);
    out_intercept.fill(f64::NAN);
    out_deviation.fill(f64::NAN);

    let do_row = |row: usize,
                  dst_value: &mut [f64],
                  dst_upper: &mut [f64],
                  dst_lower: &mut [f64],
                  dst_slope: &mut [f64],
                  dst_intercept: &mut [f64],
                  dst_dev: &mut [f64]|
     -> Result<(), SmoothTheilSenError> {
        let params = resolve_params(&combos[row])?;
        let needed = params.length + params.offset;
        let valid = data.len().saturating_sub(first);
        if valid < needed {
            return Err(SmoothTheilSenError::NotEnoughValidData { needed, valid });
        }
        smooth_theil_sen_compute_into(
            data,
            &params,
            first + warmup_bars(&params),
            dst_value,
            dst_upper,
            dst_lower,
            dst_slope,
            dst_intercept,
            dst_dev,
        )
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_value
                .par_chunks_mut(cols)
                .zip(out_upper.par_chunks_mut(cols))
                .zip(out_lower.par_chunks_mut(cols))
                .zip(out_slope.par_chunks_mut(cols))
                .zip(out_intercept.par_chunks_mut(cols))
                .zip(out_deviation.par_chunks_mut(cols))
                .enumerate()
                .try_for_each(
                    |(
                        row,
                        (
                            ((((value_row, upper_row), lower_row), slope_row), intercept_row),
                            deviation_row,
                        ),
                    )| {
                        do_row(
                            row,
                            value_row,
                            upper_row,
                            lower_row,
                            slope_row,
                            intercept_row,
                            deviation_row,
                        )
                    },
                )?;
        }
        #[cfg(target_arch = "wasm32")]
        {
            for row in 0..rows {
                let start = row * cols;
                let end = start + cols;
                do_row(
                    row,
                    &mut out_value[start..end],
                    &mut out_upper[start..end],
                    &mut out_lower[start..end],
                    &mut out_slope[start..end],
                    &mut out_intercept[start..end],
                    &mut out_deviation[start..end],
                )?;
            }
        }
    } else {
        for row in 0..rows {
            let start = row * cols;
            let end = start + cols;
            do_row(
                row,
                &mut out_value[start..end],
                &mut out_upper[start..end],
                &mut out_lower[start..end],
                &mut out_slope[start..end],
                &mut out_intercept[start..end],
                &mut out_deviation[start..end],
            )?;
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "smooth_theil_sen")]
#[pyo3(signature = (data, length=25, offset=0, multiplier=2.0, slope_style="smooth_median", residual_style="smooth_median", deviation_style="mad", mad_style="smooth_median", include_prediction_in_deviation=false, kernel=None))]
pub fn smooth_theil_sen_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length: usize,
    offset: usize,
    multiplier: f64,
    slope_style: &str,
    residual_style: &str,
    deviation_style: &str,
    mad_style: &str,
    include_prediction_in_deviation: bool,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let params = SmoothTheilSenParams {
        length: Some(length),
        offset: Some(offset),
        multiplier: Some(multiplier),
        slope_style: Some(
            SmoothTheilSenStatStyle::from_str(slope_style)
                .map_err(|_| PyValueError::new_err("Invalid slope_style"))?,
        ),
        residual_style: Some(
            SmoothTheilSenStatStyle::from_str(residual_style)
                .map_err(|_| PyValueError::new_err("Invalid residual_style"))?,
        ),
        deviation_style: Some(
            SmoothTheilSenDeviationType::from_str(deviation_style)
                .map_err(|_| PyValueError::new_err("Invalid deviation_style"))?,
        ),
        mad_style: Some(
            SmoothTheilSenStatStyle::from_str(mad_style)
                .map_err(|_| PyValueError::new_err("Invalid mad_style"))?,
        ),
        include_prediction_in_deviation: Some(include_prediction_in_deviation),
    };
    let input = SmoothTheilSenInput::from_slice(data, params);
    let kernel = validate_kernel(kernel, false)?;
    let out = py
        .allow_threads(|| smooth_theil_sen_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dict = PyDict::new(py);
    dict.set_item("value", out.value.into_pyarray(py))?;
    dict.set_item("upper", out.upper.into_pyarray(py))?;
    dict.set_item("lower", out.lower.into_pyarray(py))?;
    dict.set_item("slope", out.slope.into_pyarray(py))?;
    dict.set_item("intercept", out.intercept.into_pyarray(py))?;
    dict.set_item("deviation", out.deviation.into_pyarray(py))?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "SmoothTheilSenStream")]
pub struct SmoothTheilSenStreamPy {
    stream: SmoothTheilSenStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl SmoothTheilSenStreamPy {
    #[new]
    #[pyo3(signature = (length=25, offset=0, multiplier=2.0, slope_style="smooth_median", residual_style="smooth_median", deviation_style="mad", mad_style="smooth_median", include_prediction_in_deviation=false))]
    fn new(
        length: usize,
        offset: usize,
        multiplier: f64,
        slope_style: &str,
        residual_style: &str,
        deviation_style: &str,
        mad_style: &str,
        include_prediction_in_deviation: bool,
    ) -> PyResult<Self> {
        let stream = SmoothTheilSenStream::try_new(SmoothTheilSenParams {
            length: Some(length),
            offset: Some(offset),
            multiplier: Some(multiplier),
            slope_style: Some(
                SmoothTheilSenStatStyle::from_str(slope_style)
                    .map_err(|_| PyValueError::new_err("Invalid slope_style"))?,
            ),
            residual_style: Some(
                SmoothTheilSenStatStyle::from_str(residual_style)
                    .map_err(|_| PyValueError::new_err("Invalid residual_style"))?,
            ),
            deviation_style: Some(
                SmoothTheilSenDeviationType::from_str(deviation_style)
                    .map_err(|_| PyValueError::new_err("Invalid deviation_style"))?,
            ),
            mad_style: Some(
                SmoothTheilSenStatStyle::from_str(mad_style)
                    .map_err(|_| PyValueError::new_err("Invalid mad_style"))?,
            ),
            include_prediction_in_deviation: Some(include_prediction_in_deviation),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, value: f64) -> (f64, f64, f64, f64, f64, f64) {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "smooth_theil_sen_batch")]
#[pyo3(signature = (data, length_range=(25,25,0), offset_range=(0,0,0), multiplier_range=(2.0,2.0,0.0), slope_style="smooth_median", residual_style="smooth_median", deviation_style="mad", mad_style="smooth_median", include_prediction_in_deviation=false, kernel=None))]
pub fn smooth_theil_sen_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    offset_range: (usize, usize, usize),
    multiplier_range: (f64, f64, f64),
    slope_style: &str,
    residual_style: &str,
    deviation_style: &str,
    mad_style: &str,
    include_prediction_in_deviation: bool,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let sweep = SmoothTheilSenBatchRange {
        length: length_range,
        offset: offset_range,
        multiplier: multiplier_range,
        slope_style: SmoothTheilSenStatStyle::from_str(slope_style)
            .map_err(|_| PyValueError::new_err("Invalid slope_style"))?,
        residual_style: SmoothTheilSenStatStyle::from_str(residual_style)
            .map_err(|_| PyValueError::new_err("Invalid residual_style"))?,
        deviation_style: SmoothTheilSenDeviationType::from_str(deviation_style)
            .map_err(|_| PyValueError::new_err("Invalid deviation_style"))?,
        mad_style: SmoothTheilSenStatStyle::from_str(mad_style)
            .map_err(|_| PyValueError::new_err("Invalid mad_style"))?,
        include_prediction_in_deviation,
    };
    let combos =
        smooth_theil_sen_expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let out_value = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_upper = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_lower = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_slope = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_intercept = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_deviation = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let value_slice = unsafe { out_value.as_slice_mut()? };
    let upper_slice = unsafe { out_upper.as_slice_mut()? };
    let lower_slice = unsafe { out_lower.as_slice_mut()? };
    let slope_slice = unsafe { out_slope.as_slice_mut()? };
    let intercept_slice = unsafe { out_intercept.as_slice_mut()? };
    let deviation_slice = unsafe { out_deviation.as_slice_mut()? };
    let kernel = validate_kernel(kernel, true)?;

    py.allow_threads(|| {
        let batch_kernel = match kernel {
            Kernel::Auto => detect_best_batch_kernel(),
            other => other,
        };
        smooth_theil_sen_batch_inner_into(
            data,
            &sweep,
            batch_kernel.to_non_batch(),
            true,
            value_slice,
            upper_slice,
            lower_slice,
            slope_slice,
            intercept_slice,
            deviation_slice,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("value", out_value.reshape((rows, cols))?)?;
    dict.set_item("upper", out_upper.reshape((rows, cols))?)?;
    dict.set_item("lower", out_lower.reshape((rows, cols))?)?;
    dict.set_item("slope", out_slope.reshape((rows, cols))?)?;
    dict.set_item("intercept", out_intercept.reshape((rows, cols))?)?;
    dict.set_item("deviation", out_deviation.reshape((rows, cols))?)?;
    dict.set_item(
        "lengths",
        combos
            .iter()
            .map(|combo| combo.length.unwrap_or(DEFAULT_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "offsets",
        combos
            .iter()
            .map(|combo| combo.offset.unwrap_or(DEFAULT_OFFSET) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "multipliers",
        combos
            .iter()
            .map(|combo| combo.multiplier.unwrap_or(DEFAULT_MULTIPLIER))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_smooth_theil_sen_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(smooth_theil_sen_py, m)?)?;
    m.add_function(wrap_pyfunction!(smooth_theil_sen_batch_py, m)?)?;
    m.add_class::<SmoothTheilSenStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct SmoothTheilSenJsOutput {
    pub value: Vec<f64>,
    pub upper: Vec<f64>,
    pub lower: Vec<f64>,
    pub slope: Vec<f64>,
    pub intercept: Vec<f64>,
    pub deviation: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
fn parse_stat_style(value: String) -> Result<SmoothTheilSenStatStyle, JsValue> {
    SmoothTheilSenStatStyle::from_str(&value).map_err(|_| JsValue::from_str("Invalid stat style"))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
fn parse_deviation_style(value: String) -> Result<SmoothTheilSenDeviationType, JsValue> {
    SmoothTheilSenDeviationType::from_str(&value)
        .map_err(|_| JsValue::from_str("Invalid deviation style"))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "smooth_theil_sen_js")]
pub fn smooth_theil_sen_js(
    data: &[f64],
    length: usize,
    offset: usize,
    multiplier: f64,
    slope_style: String,
    residual_style: String,
    deviation_style: String,
    mad_style: String,
    include_prediction_in_deviation: bool,
) -> Result<JsValue, JsValue> {
    let input = SmoothTheilSenInput::from_slice(
        data,
        SmoothTheilSenParams {
            length: Some(length),
            offset: Some(offset),
            multiplier: Some(multiplier),
            slope_style: Some(parse_stat_style(slope_style)?),
            residual_style: Some(parse_stat_style(residual_style)?),
            deviation_style: Some(parse_deviation_style(deviation_style)?),
            mad_style: Some(parse_stat_style(mad_style)?),
            include_prediction_in_deviation: Some(include_prediction_in_deviation),
        },
    );
    let out = smooth_theil_sen_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&SmoothTheilSenJsOutput {
        value: out.value,
        upper: out.upper,
        lower: out.lower,
        slope: out.slope,
        intercept: out.intercept,
        deviation: out.deviation,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct SmoothTheilSenBatchConfig {
    pub length_range: Vec<usize>,
    pub offset_range: Vec<usize>,
    pub multiplier_range: Vec<f64>,
    pub slope_style: Option<String>,
    pub residual_style: Option<String>,
    pub deviation_style: Option<String>,
    pub mad_style: Option<String>,
    pub include_prediction_in_deviation: Option<bool>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct SmoothTheilSenBatchJsOutput {
    pub value: Vec<f64>,
    pub upper: Vec<f64>,
    pub lower: Vec<f64>,
    pub slope: Vec<f64>,
    pub intercept: Vec<f64>,
    pub deviation: Vec<f64>,
    pub lengths: Vec<usize>,
    pub offsets: Vec<usize>,
    pub multipliers: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
fn js_vec3_to_usize(name: &str, values: &[usize]) -> Result<(usize, usize, usize), JsValue> {
    if values.len() != 3 {
        return Err(JsValue::from_str(&format!(
            "Invalid config: {name} must have exactly 3 elements [start, end, step]"
        )));
    }
    Ok((values[0], values[1], values[2]))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
fn js_vec3_to_f64(name: &str, values: &[f64]) -> Result<(f64, f64, f64), JsValue> {
    if values.len() != 3 {
        return Err(JsValue::from_str(&format!(
            "Invalid config: {name} must have exactly 3 elements [start, end, step]"
        )));
    }
    if !values.iter().all(|v| v.is_finite()) {
        return Err(JsValue::from_str(&format!(
            "Invalid config: {name} entries must be finite numbers"
        )));
    }
    Ok((values[0], values[1], values[2]))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "smooth_theil_sen_batch_js")]
pub fn smooth_theil_sen_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: SmoothTheilSenBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let sweep = SmoothTheilSenBatchRange {
        length: js_vec3_to_usize("length_range", &config.length_range)?,
        offset: js_vec3_to_usize("offset_range", &config.offset_range)?,
        multiplier: js_vec3_to_f64("multiplier_range", &config.multiplier_range)?,
        slope_style: parse_stat_style(
            config
                .slope_style
                .unwrap_or_else(|| SmoothTheilSenStatStyle::SmoothMedian.as_str().to_string()),
        )?,
        residual_style: parse_stat_style(
            config
                .residual_style
                .unwrap_or_else(|| SmoothTheilSenStatStyle::SmoothMedian.as_str().to_string()),
        )?,
        deviation_style: parse_deviation_style(
            config
                .deviation_style
                .unwrap_or_else(|| SmoothTheilSenDeviationType::Mad.as_str().to_string()),
        )?,
        mad_style: parse_stat_style(
            config
                .mad_style
                .unwrap_or_else(|| SmoothTheilSenStatStyle::SmoothMedian.as_str().to_string()),
        )?,
        include_prediction_in_deviation: config.include_prediction_in_deviation.unwrap_or(false),
    };
    let out = smooth_theil_sen_batch_with_kernel(data, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let lengths = out
        .combos
        .iter()
        .map(|combo| combo.length.unwrap_or(DEFAULT_LENGTH))
        .collect();
    let offsets = out
        .combos
        .iter()
        .map(|combo| combo.offset.unwrap_or(DEFAULT_OFFSET))
        .collect();
    let multipliers = out
        .combos
        .iter()
        .map(|combo| combo.multiplier.unwrap_or(DEFAULT_MULTIPLIER))
        .collect();
    serde_wasm_bindgen::to_value(&SmoothTheilSenBatchJsOutput {
        value: out.value,
        upper: out.upper,
        lower: out.lower,
        slope: out.slope,
        intercept: out.intercept,
        deviation: out.deviation,
        lengths,
        offsets,
        multipliers,
        rows: out.rows,
        cols: out.cols,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn smooth_theil_sen_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn smooth_theil_sen_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn smooth_theil_sen_into(
    data_ptr: *const f64,
    out_value_ptr: *mut f64,
    out_upper_ptr: *mut f64,
    out_lower_ptr: *mut f64,
    out_slope_ptr: *mut f64,
    out_intercept_ptr: *mut f64,
    out_deviation_ptr: *mut f64,
    len: usize,
    length: usize,
    offset: usize,
    multiplier: f64,
    slope_style: String,
    residual_style: String,
    deviation_style: String,
    mad_style: String,
    include_prediction_in_deviation: bool,
) -> Result<(), JsValue> {
    if data_ptr.is_null()
        || out_value_ptr.is_null()
        || out_upper_ptr.is_null()
        || out_lower_ptr.is_null()
        || out_slope_ptr.is_null()
        || out_intercept_ptr.is_null()
        || out_deviation_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }
    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let out_value = std::slice::from_raw_parts_mut(out_value_ptr, len);
        let out_upper = std::slice::from_raw_parts_mut(out_upper_ptr, len);
        let out_lower = std::slice::from_raw_parts_mut(out_lower_ptr, len);
        let out_slope = std::slice::from_raw_parts_mut(out_slope_ptr, len);
        let out_intercept = std::slice::from_raw_parts_mut(out_intercept_ptr, len);
        let out_deviation = std::slice::from_raw_parts_mut(out_deviation_ptr, len);
        let input = SmoothTheilSenInput::from_slice(
            data,
            SmoothTheilSenParams {
                length: Some(length),
                offset: Some(offset),
                multiplier: Some(multiplier),
                slope_style: Some(parse_stat_style(slope_style)?),
                residual_style: Some(parse_stat_style(residual_style)?),
                deviation_style: Some(parse_deviation_style(deviation_style)?),
                mad_style: Some(parse_stat_style(mad_style)?),
                include_prediction_in_deviation: Some(include_prediction_in_deviation),
            },
        );
        smooth_theil_sen_into_slice(
            out_value,
            out_upper,
            out_lower,
            out_slope,
            out_intercept,
            out_deviation,
            &input,
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn smooth_theil_sen_batch_into(
    data_ptr: *const f64,
    out_value_ptr: *mut f64,
    out_upper_ptr: *mut f64,
    out_lower_ptr: *mut f64,
    out_slope_ptr: *mut f64,
    out_intercept_ptr: *mut f64,
    out_deviation_ptr: *mut f64,
    len: usize,
    length_start: usize,
    length_end: usize,
    length_step: usize,
    offset_start: usize,
    offset_end: usize,
    offset_step: usize,
    multiplier_start: f64,
    multiplier_end: f64,
    multiplier_step: f64,
    slope_style: String,
    residual_style: String,
    deviation_style: String,
    mad_style: String,
    include_prediction_in_deviation: bool,
) -> Result<usize, JsValue> {
    if data_ptr.is_null()
        || out_value_ptr.is_null()
        || out_upper_ptr.is_null()
        || out_lower_ptr.is_null()
        || out_slope_ptr.is_null()
        || out_intercept_ptr.is_null()
        || out_deviation_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to smooth_theil_sen_batch_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let sweep = SmoothTheilSenBatchRange {
            length: (length_start, length_end, length_step),
            offset: (offset_start, offset_end, offset_step),
            multiplier: (multiplier_start, multiplier_end, multiplier_step),
            slope_style: parse_stat_style(slope_style)?,
            residual_style: parse_stat_style(residual_style)?,
            deviation_style: parse_deviation_style(deviation_style)?,
            mad_style: parse_stat_style(mad_style)?,
            include_prediction_in_deviation,
        };
        let combos =
            smooth_theil_sen_expand_grid(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let total = rows.checked_mul(len).ok_or_else(|| {
            JsValue::from_str("rows*cols overflow in smooth_theil_sen_batch_into")
        })?;
        let out_value = std::slice::from_raw_parts_mut(out_value_ptr, total);
        let out_upper = std::slice::from_raw_parts_mut(out_upper_ptr, total);
        let out_lower = std::slice::from_raw_parts_mut(out_lower_ptr, total);
        let out_slope = std::slice::from_raw_parts_mut(out_slope_ptr, total);
        let out_intercept = std::slice::from_raw_parts_mut(out_intercept_ptr, total);
        let out_deviation = std::slice::from_raw_parts_mut(out_deviation_ptr, total);
        smooth_theil_sen_batch_into_slice(
            out_value,
            out_upper,
            out_lower,
            out_slope,
            out_intercept,
            out_deviation,
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
pub fn smooth_theil_sen_output_into_js(
    data: &[f64],
    length: usize,
    offset: usize,
    multiplier: f64,
    slope_style: String,
    residual_style: String,
    deviation_style: String,
    mad_style: String,
    include_prediction_in_deviation: bool,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = smooth_theil_sen_js(
        data,
        length,
        offset,
        multiplier,
        slope_style,
        residual_style,
        deviation_style,
        mad_style,
        include_prediction_in_deviation,
    )?;
    crate::write_wasm_object_f64_outputs("smooth_theil_sen_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn smooth_theil_sen_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = smooth_theil_sen_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "smooth_theil_sen_batch_output_into_js",
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

    fn sample_data(len: usize) -> Vec<f64> {
        (0..len)
            .map(|i| 100.0 + (i as f64) * 0.12 + ((i as f64) * 0.17).sin() * 1.5)
            .collect()
    }

    fn assert_vec_close_with_nan(left: &[f64], right: &[f64]) {
        assert_eq!(left.len(), right.len());
        for (a, b) in left.iter().zip(right.iter()) {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert!((a - b).abs() < 1e-10, "left={a} right={b}");
        }
    }

    #[test]
    fn perfect_linear_series_has_zero_deviation() {
        let data: Vec<f64> = (0..128).map(|i| i as f64).collect();
        let input = SmoothTheilSenInput::from_slice(
            &data,
            SmoothTheilSenParams {
                length: Some(25),
                ..SmoothTheilSenParams::default()
            },
        );
        let out = smooth_theil_sen(&input).unwrap();
        for i in 24..data.len() {
            assert!((out.value[i] - data[i]).abs() < 1e-10);
            assert!(out.deviation[i].abs() < 1e-10);
            assert!((out.slope[i] + 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn stream_matches_batch() {
        let data = sample_data(160);
        let params = SmoothTheilSenParams {
            length: Some(21),
            offset: Some(2),
            multiplier: Some(1.5),
            slope_style: Some(SmoothTheilSenStatStyle::SmoothMedian),
            residual_style: Some(SmoothTheilSenStatStyle::Median),
            deviation_style: Some(SmoothTheilSenDeviationType::Mad),
            mad_style: Some(SmoothTheilSenStatStyle::SmoothMedian),
            include_prediction_in_deviation: Some(true),
        };
        let batch =
            smooth_theil_sen(&SmoothTheilSenInput::from_slice(&data, params.clone())).unwrap();
        let mut stream = SmoothTheilSenStream::try_new(params).unwrap();
        for (i, value) in data.iter().copied().enumerate() {
            let got = stream.update(value);
            let expected = [
                batch.value[i],
                batch.upper[i],
                batch.lower[i],
                batch.slope[i],
                batch.intercept[i],
                batch.deviation[i],
            ];
            let actual = [got.0, got.1, got.2, got.3, got.4, got.5];
            for (left, right) in actual.into_iter().zip(expected) {
                if left.is_nan() && right.is_nan() {
                    continue;
                }
                assert!((left - right).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn batch_first_row_matches_single() {
        let data = sample_data(144);
        let single = smooth_theil_sen(&SmoothTheilSenInput::from_slice(
            &data,
            SmoothTheilSenParams {
                length: Some(21),
                offset: Some(1),
                multiplier: Some(1.5),
                slope_style: Some(SmoothTheilSenStatStyle::Mean),
                residual_style: Some(SmoothTheilSenStatStyle::SmoothMedian),
                deviation_style: Some(SmoothTheilSenDeviationType::Rmsd),
                mad_style: Some(SmoothTheilSenStatStyle::Median),
                include_prediction_in_deviation: Some(false),
            },
        ))
        .unwrap();
        let batch = smooth_theil_sen_batch_with_kernel(
            &data,
            &SmoothTheilSenBatchRange {
                length: (21, 23, 2),
                offset: (1, 1, 0),
                multiplier: (1.5, 1.5, 0.0),
                slope_style: SmoothTheilSenStatStyle::Mean,
                residual_style: SmoothTheilSenStatStyle::SmoothMedian,
                deviation_style: SmoothTheilSenDeviationType::Rmsd,
                mad_style: SmoothTheilSenStatStyle::Median,
                include_prediction_in_deviation: false,
            },
            Kernel::Auto,
        )
        .unwrap();
        assert_vec_close_with_nan(&batch.value[..data.len()], single.value.as_slice());
        assert_vec_close_with_nan(&batch.upper[..data.len()], single.upper.as_slice());
        assert_vec_close_with_nan(&batch.lower[..data.len()], single.lower.as_slice());
        assert_vec_close_with_nan(&batch.slope[..data.len()], single.slope.as_slice());
        assert_vec_close_with_nan(&batch.intercept[..data.len()], single.intercept.as_slice());
        assert_vec_close_with_nan(&batch.deviation[..data.len()], single.deviation.as_slice());
    }

    #[test]
    fn invalid_length_fails() {
        let data = sample_data(32);
        let input = SmoothTheilSenInput::from_slice(
            &data,
            SmoothTheilSenParams {
                length: Some(1),
                ..SmoothTheilSenParams::default()
            },
        );
        match smooth_theil_sen(&input) {
            Err(SmoothTheilSenError::InvalidLength { length }) => assert_eq!(length, 1),
            other => panic!("unexpected result: {other:?}"),
        }
    }

    #[test]
    fn cpu_dispatch_matches_direct_output() {
        let data = sample_data(128);
        let expected = smooth_theil_sen(&SmoothTheilSenInput::from_slice(
            &data,
            SmoothTheilSenParams {
                length: Some(21),
                offset: Some(2),
                multiplier: Some(1.75),
                slope_style: Some(SmoothTheilSenStatStyle::SmoothMedian),
                residual_style: Some(SmoothTheilSenStatStyle::SmoothMedian),
                deviation_style: Some(SmoothTheilSenDeviationType::Mad),
                mad_style: Some(SmoothTheilSenStatStyle::Median),
                include_prediction_in_deviation: Some(true),
            },
        ))
        .unwrap();
        let combos = [IndicatorParamSet {
            params: &[
                ParamKV {
                    key: "length",
                    value: ParamValue::Int(21),
                },
                ParamKV {
                    key: "offset",
                    value: ParamValue::Int(2),
                },
                ParamKV {
                    key: "multiplier",
                    value: ParamValue::Float(1.75),
                },
                ParamKV {
                    key: "slope_style",
                    value: ParamValue::EnumString("smooth_median"),
                },
                ParamKV {
                    key: "residual_style",
                    value: ParamValue::EnumString("smooth_median"),
                },
                ParamKV {
                    key: "deviation_style",
                    value: ParamValue::EnumString("mad"),
                },
                ParamKV {
                    key: "mad_style",
                    value: ParamValue::EnumString("median"),
                },
                ParamKV {
                    key: "include_prediction_in_deviation",
                    value: ParamValue::Bool(true),
                },
            ],
        }];
        let out = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "smooth_theil_sen",
            output_id: Some("value"),
            data: IndicatorDataRef::Slice { values: &data },
            combos: &combos,
            kernel: Kernel::Auto,
        })
        .unwrap();
        assert_vec_close_with_nan(
            out.values_f64.unwrap().as_slice(),
            expected.value.as_slice(),
        );
    }
}
