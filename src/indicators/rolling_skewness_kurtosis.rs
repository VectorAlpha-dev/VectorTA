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
use std::error::Error;
use thiserror::Error;

impl<'a> AsRef<[f64]> for RollingSkewnessKurtosisInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            RollingSkewnessKurtosisData::Slice(slice) => slice,
            RollingSkewnessKurtosisData::Candles { candles, source } => {
                if *source == "close" {
                    &candles.close
                } else {
                    source_type(candles, source)
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum RollingSkewnessKurtosisData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct RollingSkewnessKurtosisOutput {
    pub skewness: Vec<f64>,
    pub kurtosis: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct RollingSkewnessKurtosisParams {
    pub length: Option<usize>,
    pub smooth_length: Option<usize>,
}

impl Default for RollingSkewnessKurtosisParams {
    fn default() -> Self {
        Self {
            length: Some(50),
            smooth_length: Some(3),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RollingSkewnessKurtosisInput<'a> {
    pub data: RollingSkewnessKurtosisData<'a>,
    pub params: RollingSkewnessKurtosisParams,
}

impl<'a> RollingSkewnessKurtosisInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: RollingSkewnessKurtosisParams,
    ) -> Self {
        Self {
            data: RollingSkewnessKurtosisData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: RollingSkewnessKurtosisParams) -> Self {
        Self {
            data: RollingSkewnessKurtosisData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", RollingSkewnessKurtosisParams::default())
    }

    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(50)
    }

    #[inline]
    pub fn get_smooth_length(&self) -> usize {
        self.params.smooth_length.unwrap_or(3)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct RollingSkewnessKurtosisBuilder {
    length: Option<usize>,
    smooth_length: Option<usize>,
    kernel: Kernel,
}

impl Default for RollingSkewnessKurtosisBuilder {
    fn default() -> Self {
        Self {
            length: None,
            smooth_length: None,
            kernel: Kernel::Auto,
        }
    }
}

impl RollingSkewnessKurtosisBuilder {
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
    pub fn smooth_length(mut self, value: usize) -> Self {
        self.smooth_length = Some(value);
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
    ) -> Result<RollingSkewnessKurtosisOutput, RollingSkewnessKurtosisError> {
        let params = RollingSkewnessKurtosisParams {
            length: self.length,
            smooth_length: self.smooth_length,
        };
        rolling_skewness_kurtosis_with_kernel(
            &RollingSkewnessKurtosisInput::from_candles(candles, "close", params),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<RollingSkewnessKurtosisOutput, RollingSkewnessKurtosisError> {
        let params = RollingSkewnessKurtosisParams {
            length: self.length,
            smooth_length: self.smooth_length,
        };
        rolling_skewness_kurtosis_with_kernel(
            &RollingSkewnessKurtosisInput::from_candles(candles, source, params),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<RollingSkewnessKurtosisOutput, RollingSkewnessKurtosisError> {
        let params = RollingSkewnessKurtosisParams {
            length: self.length,
            smooth_length: self.smooth_length,
        };
        rolling_skewness_kurtosis_with_kernel(
            &RollingSkewnessKurtosisInput::from_slice(data, params),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<RollingSkewnessKurtosisStream, RollingSkewnessKurtosisError> {
        RollingSkewnessKurtosisStream::try_new(RollingSkewnessKurtosisParams {
            length: self.length,
            smooth_length: self.smooth_length,
        })
    }
}

#[derive(Debug, Error)]
pub enum RollingSkewnessKurtosisError {
    #[error("rolling_skewness_kurtosis: Input data slice is empty.")]
    EmptyInputData,
    #[error("rolling_skewness_kurtosis: All values are NaN.")]
    AllValuesNaN,
    #[error(
        "rolling_skewness_kurtosis: Invalid length: length = {length}, data length = {data_len}"
    )]
    InvalidLength { length: usize, data_len: usize },
    #[error("rolling_skewness_kurtosis: Invalid smooth_length: {smooth_length}")]
    InvalidSmoothLength { smooth_length: usize },
    #[error(
        "rolling_skewness_kurtosis: Not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "rolling_skewness_kurtosis: Output length mismatch: expected = {expected}, got = {got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("rolling_skewness_kurtosis: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("rolling_skewness_kurtosis: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error(
        "rolling_skewness_kurtosis: Output length mismatch: dst = {dst_len}, expected = {expected_len}"
    )]
    MismatchedOutputLen { dst_len: usize, expected_len: usize },
    #[error("rolling_skewness_kurtosis: Invalid input: {msg}")]
    InvalidInput { msg: String },
}

#[derive(Debug, Clone)]
struct SmaState {
    period: usize,
    buf: Vec<f64>,
    head: usize,
    count: usize,
    sum: f64,
}

impl SmaState {
    #[inline(always)]
    fn new(period: usize) -> Self {
        Self {
            period,
            buf: vec![0.0; period.max(1)],
            head: 0,
            count: 0,
            sum: 0.0,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.head = 0;
        self.count = 0;
        self.sum = 0.0;
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<f64> {
        if !value.is_finite() {
            self.reset();
            return None;
        }
        if self.period == 1 {
            return Some(value);
        }
        if self.count < self.period {
            self.buf[self.head] = value;
            self.head = (self.head + 1) % self.period;
            self.count += 1;
            self.sum += value;
            if self.count < self.period {
                None
            } else {
                Some(self.sum / self.period as f64)
            }
        } else {
            let old = self.buf[self.head];
            self.buf[self.head] = value;
            self.head = (self.head + 1) % self.period;
            self.sum += value - old;
            Some(self.sum / self.period as f64)
        }
    }
}

#[derive(Debug, Clone)]
pub struct RollingSkewnessKurtosisStream {
    length: usize,
    smooth_length: usize,
    source_buf: Vec<f64>,
    head: usize,
    count: usize,
    sum1: f64,
    sum2: f64,
    sum3: f64,
    sum4: f64,
    skew_sma: SmaState,
    kurt_sma: SmaState,
}

impl RollingSkewnessKurtosisStream {
    #[inline(always)]
    pub fn try_new(
        params: RollingSkewnessKurtosisParams,
    ) -> Result<Self, RollingSkewnessKurtosisError> {
        let length = params.length.unwrap_or(50);
        let smooth_length = params.smooth_length.unwrap_or(3);
        validate_params(length, smooth_length, usize::MAX)?;
        Ok(Self {
            length,
            smooth_length,
            source_buf: vec![0.0; length.max(1)],
            head: 0,
            count: 0,
            sum1: 0.0,
            sum2: 0.0,
            sum3: 0.0,
            sum4: 0.0,
            skew_sma: SmaState::new(smooth_length),
            kurt_sma: SmaState::new(smooth_length),
        })
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.head = 0;
        self.count = 0;
        self.sum1 = 0.0;
        self.sum2 = 0.0;
        self.sum3 = 0.0;
        self.sum4 = 0.0;
        self.skew_sma.reset();
        self.kurt_sma.reset();
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        if !value.is_finite() {
            self.reset();
            return None;
        }

        if self.count == self.length {
            self.sum1 += value - self.source_buf[self.head];
        } else {
            self.count += 1;
            self.sum1 += value;
        }
        self.source_buf[self.head] = value;
        self.head = (self.head + 1) % self.length;

        if self.count < self.length {
            return None;
        }

        let n = self.length as f64;
        let mean = self.sum1 / n;
        let mut m2 = 0.0;
        let mut m3 = 0.0;
        let mut m4 = 0.0;
        for &window_value in self.source_buf.iter().take(self.length) {
            let dev = window_value - mean;
            let dev2 = dev * dev;
            m2 += dev2;
            m3 += dev2 * dev;
            m4 += dev2 * dev2;
        }
        m2 /= n;
        if !m2.is_finite() || m2 <= f64::EPSILON {
            self.skew_sma.reset();
            self.kurt_sma.reset();
            return None;
        }
        let sigma = m2.sqrt();
        let skew_raw = (m3 / n) / (sigma * sigma * sigma);
        let kurt_raw = (m4 / n) / (m2 * m2) - 3.0;
        let skewness = self.skew_sma.update(skew_raw);
        let kurtosis = self.kurt_sma.update(kurt_raw);
        match (skewness, kurtosis) {
            (Some(skewness), Some(kurtosis)) => Some((skewness, kurtosis)),
            _ => None,
        }
    }

    #[inline(always)]
    pub fn get_warmup_period(&self) -> usize {
        warmup_prefix(self.length, self.smooth_length)
    }
}

#[inline(always)]
fn warmup_needed(length: usize, smooth_length: usize) -> usize {
    length.saturating_add(smooth_length).saturating_sub(1)
}

#[inline(always)]
fn warmup_prefix(length: usize, smooth_length: usize) -> usize {
    warmup_needed(length, smooth_length).saturating_sub(1)
}

#[inline(always)]
fn longest_valid_run(data: &[f64]) -> usize {
    let mut best = 0usize;
    let mut cur = 0usize;
    for &value in data {
        if value.is_finite() {
            cur += 1;
            if cur > best {
                best = cur;
            }
        } else {
            cur = 0;
        }
    }
    best
}

#[inline(always)]
fn validate_params(
    length: usize,
    smooth_length: usize,
    data_len: usize,
) -> Result<(), RollingSkewnessKurtosisError> {
    if length == 0 || (data_len != usize::MAX && length > data_len) {
        return Err(RollingSkewnessKurtosisError::InvalidLength { length, data_len });
    }
    if smooth_length == 0 {
        return Err(RollingSkewnessKurtosisError::InvalidSmoothLength { smooth_length });
    }
    Ok(())
}

#[inline(always)]
fn validate_common(
    data: &[f64],
    length: usize,
    smooth_length: usize,
) -> Result<usize, RollingSkewnessKurtosisError> {
    if data.is_empty() {
        return Err(RollingSkewnessKurtosisError::EmptyInputData);
    }
    validate_params(length, smooth_length, data.len())?;
    let max_run = longest_valid_run(data);
    if max_run == 0 {
        return Err(RollingSkewnessKurtosisError::AllValuesNaN);
    }
    let needed = warmup_needed(length, smooth_length);
    if max_run < needed {
        return Err(RollingSkewnessKurtosisError::NotEnoughValidData {
            needed,
            valid: max_run,
        });
    }
    Ok(max_run)
}

#[inline(always)]
fn update_sma3(
    value: f64,
    buf: &mut [f64; 3],
    head: &mut usize,
    count: &mut usize,
    sum: &mut f64,
) -> Option<f64> {
    if *count < 3 {
        buf[*head] = value;
        *head += 1;
        if *head == 3 {
            *head = 0;
        }
        *count += 1;
        *sum += value;
        if *count < 3 {
            None
        } else {
            Some(*sum / 3.0)
        }
    } else {
        let old = buf[*head];
        buf[*head] = value;
        *head += 1;
        if *head == 3 {
            *head = 0;
        }
        *sum += value - old;
        Some(*sum / 3.0)
    }
}

#[inline(always)]
fn compute_row_50_3_all_finite(data: &[f64], out_skewness: &mut [f64], out_kurtosis: &mut [f64]) {
    let prefix = warmup_prefix(50, 3).min(data.len());
    out_skewness[..prefix].fill(f64::NAN);
    out_kurtosis[..prefix].fill(f64::NAN);

    let mut source_buf = [0.0; 50];
    let mut head = 0usize;
    let mut count = 0usize;
    let mut sum1 = 0.0;
    let mut skew_buf = [0.0; 3];
    let mut kurt_buf = [0.0; 3];
    let mut skew_head = 0usize;
    let mut kurt_head = 0usize;
    let mut skew_count = 0usize;
    let mut kurt_count = 0usize;
    let mut skew_sum = 0.0;
    let mut kurt_sum = 0.0;

    for i in 0..data.len() {
        let value = data[i];
        if count == 50 {
            sum1 += value - source_buf[head];
        } else {
            count += 1;
            sum1 += value;
        }
        source_buf[head] = value;
        head += 1;
        if head == 50 {
            head = 0;
        }

        if count < 50 {
            continue;
        }

        let mean = sum1 / 50.0;
        let mut m2 = 0.0;
        let mut m3 = 0.0;
        let mut m4 = 0.0;
        for &window_value in &source_buf {
            let dev = window_value - mean;
            let dev2 = dev * dev;
            m2 += dev2;
            m3 += dev2 * dev;
            m4 += dev2 * dev2;
        }
        m2 /= 50.0;
        if !m2.is_finite() || m2 <= f64::EPSILON {
            skew_head = 0;
            kurt_head = 0;
            skew_count = 0;
            kurt_count = 0;
            skew_sum = 0.0;
            kurt_sum = 0.0;
            if i >= prefix {
                out_skewness[i] = f64::NAN;
                out_kurtosis[i] = f64::NAN;
            }
            continue;
        }
        let sigma = m2.sqrt();
        let skew_raw = (m3 / 50.0) / (sigma * sigma * sigma);
        let kurt_raw = (m4 / 50.0) / (m2 * m2) - 3.0;
        let skewness = update_sma3(
            skew_raw,
            &mut skew_buf,
            &mut skew_head,
            &mut skew_count,
            &mut skew_sum,
        );
        let kurtosis = update_sma3(
            kurt_raw,
            &mut kurt_buf,
            &mut kurt_head,
            &mut kurt_count,
            &mut kurt_sum,
        );
        if let (Some(skewness), Some(kurtosis)) = (skewness, kurtosis) {
            out_skewness[i] = skewness;
            out_kurtosis[i] = kurtosis;
        }
    }
}

#[inline(always)]
fn compute_row(
    data: &[f64],
    length: usize,
    smooth_length: usize,
    all_finite: bool,
    out_skewness: &mut [f64],
    out_kurtosis: &mut [f64],
) {
    if all_finite && length == 50 && smooth_length == 3 {
        compute_row_50_3_all_finite(data, out_skewness, out_kurtosis);
        return;
    }

    if all_finite {
        let prefix = warmup_prefix(length, smooth_length).min(data.len());
        out_skewness[..prefix].fill(f64::NAN);
        out_kurtosis[..prefix].fill(f64::NAN);
    } else {
        out_skewness.fill(f64::NAN);
        out_kurtosis.fill(f64::NAN);
    }
    let mut stream = RollingSkewnessKurtosisStream::try_new(RollingSkewnessKurtosisParams {
        length: Some(length),
        smooth_length: Some(smooth_length),
    })
    .expect("validated params");

    for i in 0..data.len() {
        if let Some((skewness, kurtosis)) = stream.update(data[i]) {
            out_skewness[i] = skewness;
            out_kurtosis[i] = kurtosis;
        } else if all_finite && i >= warmup_prefix(length, smooth_length) {
            out_skewness[i] = f64::NAN;
            out_kurtosis[i] = f64::NAN;
        }
    }
}

pub fn rolling_skewness_kurtosis(
    input: &RollingSkewnessKurtosisInput,
) -> Result<RollingSkewnessKurtosisOutput, RollingSkewnessKurtosisError> {
    rolling_skewness_kurtosis_with_kernel(input, Kernel::Auto)
}

pub fn rolling_skewness_kurtosis_with_kernel(
    input: &RollingSkewnessKurtosisInput,
    kernel: Kernel,
) -> Result<RollingSkewnessKurtosisOutput, RollingSkewnessKurtosisError> {
    let data = input.as_ref();
    let length = input.get_length();
    let smooth_length = input.get_smooth_length();
    let max_run = validate_common(data, length, smooth_length)?;

    let _chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    let prefix = warmup_prefix(length, smooth_length);
    let mut skewness = alloc_with_nan_prefix(data.len(), prefix);
    let mut kurtosis = alloc_with_nan_prefix(data.len(), prefix);
    compute_row(
        data,
        length,
        smooth_length,
        max_run == data.len(),
        &mut skewness,
        &mut kurtosis,
    );
    Ok(RollingSkewnessKurtosisOutput { skewness, kurtosis })
}

pub fn rolling_skewness_kurtosis_into_slice(
    dst_skewness: &mut [f64],
    dst_kurtosis: &mut [f64],
    input: &RollingSkewnessKurtosisInput,
    kernel: Kernel,
) -> Result<(), RollingSkewnessKurtosisError> {
    let data = input.as_ref();
    let length = input.get_length();
    let smooth_length = input.get_smooth_length();
    let max_run = validate_common(data, length, smooth_length)?;
    if dst_skewness.len() != data.len() {
        return Err(RollingSkewnessKurtosisError::OutputLengthMismatch {
            expected: data.len(),
            got: dst_skewness.len(),
        });
    }
    if dst_kurtosis.len() != data.len() {
        return Err(RollingSkewnessKurtosisError::OutputLengthMismatch {
            expected: data.len(),
            got: dst_kurtosis.len(),
        });
    }

    let _chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    compute_row(
        data,
        length,
        smooth_length,
        max_run == data.len(),
        dst_skewness,
        dst_kurtosis,
    );
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn rolling_skewness_kurtosis_into(
    input: &RollingSkewnessKurtosisInput,
    out_skewness: &mut [f64],
    out_kurtosis: &mut [f64],
) -> Result<(), RollingSkewnessKurtosisError> {
    rolling_skewness_kurtosis_into_slice(out_skewness, out_kurtosis, input, Kernel::Auto)
}

#[derive(Debug, Clone, Copy)]
pub struct RollingSkewnessKurtosisBatchRange {
    pub length: (usize, usize, usize),
    pub smooth_length: (usize, usize, usize),
}

impl Default for RollingSkewnessKurtosisBatchRange {
    fn default() -> Self {
        Self {
            length: (50, 50, 0),
            smooth_length: (3, 3, 0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RollingSkewnessKurtosisBatchOutput {
    pub skewness: Vec<f64>,
    pub kurtosis: Vec<f64>,
    pub combos: Vec<RollingSkewnessKurtosisParams>,
    pub rows: usize,
    pub cols: usize,
}

impl RollingSkewnessKurtosisBatchOutput {
    pub fn row_for_params(&self, params: &RollingSkewnessKurtosisParams) -> Option<usize> {
        let length = params.length.unwrap_or(50);
        let smooth_length = params.smooth_length.unwrap_or(3);
        self.combos.iter().position(|combo| {
            combo.length.unwrap_or(50) == length
                && combo.smooth_length.unwrap_or(3) == smooth_length
        })
    }

    pub fn skewness_for(&self, params: &RollingSkewnessKurtosisParams) -> Option<&[f64]> {
        self.row_for_params(params).and_then(|row| {
            let start = row.checked_mul(self.cols)?;
            self.skewness.get(start..start + self.cols)
        })
    }

    pub fn kurtosis_for(&self, params: &RollingSkewnessKurtosisParams) -> Option<&[f64]> {
        self.row_for_params(params).and_then(|row| {
            let start = row.checked_mul(self.cols)?;
            self.kurtosis.get(start..start + self.cols)
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RollingSkewnessKurtosisBatchBuilder {
    range: RollingSkewnessKurtosisBatchRange,
    kernel: Kernel,
}

impl Default for RollingSkewnessKurtosisBatchBuilder {
    fn default() -> Self {
        Self {
            range: RollingSkewnessKurtosisBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl RollingSkewnessKurtosisBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline(always)]
    pub fn length_range(mut self, value: (usize, usize, usize)) -> Self {
        self.range.length = value;
        self
    }

    #[inline(always)]
    pub fn smooth_length_range(mut self, value: (usize, usize, usize)) -> Self {
        self.range.smooth_length = value;
        self
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<RollingSkewnessKurtosisBatchOutput, RollingSkewnessKurtosisError> {
        rolling_skewness_kurtosis_batch_with_kernel(data, &self.range, self.kernel)
    }

    #[inline(always)]
    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<RollingSkewnessKurtosisBatchOutput, RollingSkewnessKurtosisError> {
        rolling_skewness_kurtosis_batch_with_kernel(
            source_type(candles, source),
            &self.range,
            self.kernel,
        )
    }
}

#[inline(always)]
fn expand_axis(range: (usize, usize, usize)) -> Result<Vec<usize>, RollingSkewnessKurtosisError> {
    let (start, end, step) = range;
    if start == 0 {
        return Err(RollingSkewnessKurtosisError::InvalidRange { start, end, step });
    }
    if step == 0 {
        return Ok(vec![start]);
    }
    if start > end {
        return Err(RollingSkewnessKurtosisError::InvalidRange { start, end, step });
    }
    let mut out = Vec::new();
    let mut cur = start;
    loop {
        out.push(cur);
        if cur >= end {
            break;
        }
        let next =
            cur.checked_add(step)
                .ok_or_else(|| RollingSkewnessKurtosisError::InvalidInput {
                    msg: "rolling_skewness_kurtosis: range step overflow".to_string(),
                })?;
        if next <= cur {
            return Err(RollingSkewnessKurtosisError::InvalidRange { start, end, step });
        }
        cur = next.min(end);
    }
    Ok(out)
}

#[inline(always)]
fn expand_grid_checked(
    range: &RollingSkewnessKurtosisBatchRange,
) -> Result<Vec<RollingSkewnessKurtosisParams>, RollingSkewnessKurtosisError> {
    let lengths = expand_axis(range.length)?;
    let smooth_lengths = expand_axis(range.smooth_length)?;
    let total = lengths
        .len()
        .checked_mul(smooth_lengths.len())
        .ok_or_else(|| RollingSkewnessKurtosisError::InvalidInput {
            msg: "rolling_skewness_kurtosis: parameter grid size overflow".to_string(),
        })?;
    let mut out = Vec::with_capacity(total);
    for &length in &lengths {
        for &smooth_length in &smooth_lengths {
            out.push(RollingSkewnessKurtosisParams {
                length: Some(length),
                smooth_length: Some(smooth_length),
            });
        }
    }
    Ok(out)
}

pub fn expand_grid_rolling_skewness_kurtosis(
    range: &RollingSkewnessKurtosisBatchRange,
) -> Vec<RollingSkewnessKurtosisParams> {
    expand_grid_checked(range).unwrap_or_default()
}

pub fn rolling_skewness_kurtosis_batch_with_kernel(
    data: &[f64],
    sweep: &RollingSkewnessKurtosisBatchRange,
    kernel: Kernel,
) -> Result<RollingSkewnessKurtosisBatchOutput, RollingSkewnessKurtosisError> {
    rolling_skewness_kurtosis_batch_inner(data, sweep, kernel, true)
}

pub fn rolling_skewness_kurtosis_batch_slice(
    data: &[f64],
    sweep: &RollingSkewnessKurtosisBatchRange,
    kernel: Kernel,
) -> Result<RollingSkewnessKurtosisBatchOutput, RollingSkewnessKurtosisError> {
    rolling_skewness_kurtosis_batch_inner(data, sweep, kernel, false)
}

pub fn rolling_skewness_kurtosis_batch_par_slice(
    data: &[f64],
    sweep: &RollingSkewnessKurtosisBatchRange,
    kernel: Kernel,
) -> Result<RollingSkewnessKurtosisBatchOutput, RollingSkewnessKurtosisError> {
    rolling_skewness_kurtosis_batch_inner(data, sweep, kernel, true)
}

fn rolling_skewness_kurtosis_batch_inner(
    data: &[f64],
    sweep: &RollingSkewnessKurtosisBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<RollingSkewnessKurtosisBatchOutput, RollingSkewnessKurtosisError> {
    let combos = expand_grid_checked(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    let total =
        rows.checked_mul(cols)
            .ok_or_else(|| RollingSkewnessKurtosisError::InvalidInput {
                msg: "rolling_skewness_kurtosis: rows*cols overflow in batch".to_string(),
            })?;

    if data.is_empty() {
        return Err(RollingSkewnessKurtosisError::EmptyInputData);
    }

    let mut max_needed = 0usize;
    let warmups: Vec<usize> = combos
        .iter()
        .map(|params| {
            let length = params.length.unwrap_or(50);
            let smooth_length = params.smooth_length.unwrap_or(3);
            max_needed = max_needed.max(warmup_needed(length, smooth_length));
            warmup_prefix(length, smooth_length)
        })
        .collect();

    let max_run = longest_valid_run(data);
    if max_run == 0 {
        return Err(RollingSkewnessKurtosisError::AllValuesNaN);
    }
    if max_run < max_needed {
        return Err(RollingSkewnessKurtosisError::NotEnoughValidData {
            needed: max_needed,
            valid: max_run,
        });
    }

    let mut skewness_mu = make_uninit_matrix(rows, cols);
    let mut kurtosis_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut skewness_mu, cols, &warmups);
    init_matrix_prefixes(&mut kurtosis_mu, cols, &warmups);

    let mut skewness = unsafe {
        Vec::from_raw_parts(
            skewness_mu.as_mut_ptr() as *mut f64,
            skewness_mu.len(),
            skewness_mu.capacity(),
        )
    };
    let mut kurtosis = unsafe {
        Vec::from_raw_parts(
            kurtosis_mu.as_mut_ptr() as *mut f64,
            kurtosis_mu.len(),
            kurtosis_mu.capacity(),
        )
    };
    std::mem::forget(skewness_mu);
    std::mem::forget(kurtosis_mu);
    debug_assert_eq!(skewness.len(), total);
    debug_assert_eq!(kurtosis.len(), total);

    rolling_skewness_kurtosis_batch_inner_into(
        data,
        sweep,
        kernel,
        parallel,
        &mut skewness,
        &mut kurtosis,
    )?;

    Ok(RollingSkewnessKurtosisBatchOutput {
        skewness,
        kurtosis,
        combos,
        rows,
        cols,
    })
}

fn rolling_skewness_kurtosis_batch_inner_into(
    data: &[f64],
    sweep: &RollingSkewnessKurtosisBatchRange,
    kernel: Kernel,
    parallel: bool,
    out_skewness: &mut [f64],
    out_kurtosis: &mut [f64],
) -> Result<Vec<RollingSkewnessKurtosisParams>, RollingSkewnessKurtosisError> {
    match kernel {
        Kernel::Auto
        | Kernel::Scalar
        | Kernel::ScalarBatch
        | Kernel::Avx2
        | Kernel::Avx2Batch
        | Kernel::Avx512
        | Kernel::Avx512Batch => {}
        other => return Err(RollingSkewnessKurtosisError::InvalidKernelForBatch(other)),
    }

    let combos = expand_grid_checked(sweep)?;
    let len = data.len();
    if len == 0 {
        return Err(RollingSkewnessKurtosisError::EmptyInputData);
    }
    let total = combos.len().checked_mul(len).ok_or_else(|| {
        RollingSkewnessKurtosisError::InvalidInput {
            msg: "rolling_skewness_kurtosis: rows*cols overflow in batch_into".to_string(),
        }
    })?;
    if out_skewness.len() != total {
        return Err(RollingSkewnessKurtosisError::MismatchedOutputLen {
            dst_len: out_skewness.len(),
            expected_len: total,
        });
    }
    if out_kurtosis.len() != total {
        return Err(RollingSkewnessKurtosisError::MismatchedOutputLen {
            dst_len: out_kurtosis.len(),
            expected_len: total,
        });
    }

    let mut max_needed = 0usize;
    for params in &combos {
        max_needed = max_needed.max(warmup_needed(
            params.length.unwrap_or(50),
            params.smooth_length.unwrap_or(3),
        ));
    }
    let max_run = longest_valid_run(data);
    if max_run == 0 {
        return Err(RollingSkewnessKurtosisError::AllValuesNaN);
    }
    if max_run < max_needed {
        return Err(RollingSkewnessKurtosisError::NotEnoughValidData {
            needed: max_needed,
            valid: max_run,
        });
    }

    let _chosen = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other => other,
    };

    let worker = |row: usize, dst_skewness: &mut [f64], dst_kurtosis: &mut [f64]| {
        let params = &combos[row];
        compute_row(
            data,
            params.length.unwrap_or(50),
            params.smooth_length.unwrap_or(3),
            max_run == data.len(),
            dst_skewness,
            dst_kurtosis,
        );
    };

    if parallel && combos.len() > 1 {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_skewness
                .par_chunks_mut(len)
                .zip(out_kurtosis.par_chunks_mut(len))
                .enumerate()
                .for_each(|(row, (dst_skewness, dst_kurtosis))| {
                    worker(row, dst_skewness, dst_kurtosis)
                });
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, (dst_skewness, dst_kurtosis)) in out_skewness
                .chunks_mut(len)
                .zip(out_kurtosis.chunks_mut(len))
                .enumerate()
            {
                worker(row, dst_skewness, dst_kurtosis);
            }
        }
    } else {
        for (row, (dst_skewness, dst_kurtosis)) in out_skewness
            .chunks_mut(len)
            .zip(out_kurtosis.chunks_mut(len))
            .enumerate()
        {
            worker(row, dst_skewness, dst_kurtosis);
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "rolling_skewness_kurtosis")]
#[pyo3(signature = (data, length=50, smooth_length=3, kernel=None))]
pub fn rolling_skewness_kurtosis_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length: usize,
    smooth_length: usize,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let data = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let input = RollingSkewnessKurtosisInput::from_slice(
        data,
        RollingSkewnessKurtosisParams {
            length: Some(length),
            smooth_length: Some(smooth_length),
        },
    );
    let out = py
        .allow_threads(|| rolling_skewness_kurtosis_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((out.skewness.into_pyarray(py), out.kurtosis.into_pyarray(py)))
}

#[cfg(feature = "python")]
#[pyclass(name = "RollingSkewnessKurtosisStream")]
pub struct RollingSkewnessKurtosisStreamPy {
    stream: RollingSkewnessKurtosisStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl RollingSkewnessKurtosisStreamPy {
    #[new]
    #[pyo3(signature = (length=50, smooth_length=3))]
    fn new(length: usize, smooth_length: usize) -> PyResult<Self> {
        let stream = RollingSkewnessKurtosisStream::try_new(RollingSkewnessKurtosisParams {
            length: Some(length),
            smooth_length: Some(smooth_length),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        self.stream.update(value)
    }

    fn reset(&mut self) {
        self.stream.reset();
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "rolling_skewness_kurtosis_batch")]
#[pyo3(signature = (data, length_range=(50,50,0), smooth_length_range=(3,3,0), kernel=None))]
pub fn rolling_skewness_kurtosis_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    smooth_length_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let output = py
        .allow_threads(|| {
            rolling_skewness_kurtosis_batch_with_kernel(
                data,
                &RollingSkewnessKurtosisBatchRange {
                    length: length_range,
                    smooth_length: smooth_length_range,
                },
                kern,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item(
        "skewness",
        output
            .skewness
            .into_pyarray(py)
            .reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "kurtosis",
        output
            .kurtosis
            .into_pyarray(py)
            .reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "lengths",
        output
            .combos
            .iter()
            .map(|params| params.length.unwrap_or(50) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "smooth_lengths",
        output
            .combos
            .iter()
            .map(|params| params.smooth_length.unwrap_or(3) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", output.rows)?;
    dict.set_item("cols", output.cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_rolling_skewness_kurtosis_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rolling_skewness_kurtosis_py, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_skewness_kurtosis_batch_py, m)?)?;
    m.add_class::<RollingSkewnessKurtosisStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollingSkewnessKurtosisBatchConfig {
    pub length_range: Vec<usize>,
    pub smooth_length_range: Vec<usize>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = rolling_skewness_kurtosis_js)]
pub fn rolling_skewness_kurtosis_js(
    data: &[f64],
    length: usize,
    smooth_length: usize,
) -> Result<JsValue, JsValue> {
    let input = RollingSkewnessKurtosisInput::from_slice(
        data,
        RollingSkewnessKurtosisParams {
            length: Some(length),
            smooth_length: Some(smooth_length),
        },
    );
    let out = rolling_skewness_kurtosis_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("skewness"),
        &serde_wasm_bindgen::to_value(&out.skewness).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("kurtosis"),
        &serde_wasm_bindgen::to_value(&out.kurtosis).unwrap(),
    )?;
    Ok(obj.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = rolling_skewness_kurtosis_batch_js)]
pub fn rolling_skewness_kurtosis_batch_js(
    data: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: RollingSkewnessKurtosisBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.length_range.len() != 3 || config.smooth_length_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: every range must have exactly 3 elements [start, end, step]",
        ));
    }

    let out = rolling_skewness_kurtosis_batch_with_kernel(
        data,
        &RollingSkewnessKurtosisBatchRange {
            length: (
                config.length_range[0],
                config.length_range[1],
                config.length_range[2],
            ),
            smooth_length: (
                config.smooth_length_range[0],
                config.smooth_length_range[1],
                config.smooth_length_range[2],
            ),
        },
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("skewness"),
        &serde_wasm_bindgen::to_value(&out.skewness).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("kurtosis"),
        &serde_wasm_bindgen::to_value(&out.kurtosis).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("rows"),
        &JsValue::from_f64(out.rows as f64),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("cols"),
        &JsValue::from_f64(out.cols as f64),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("combos"),
        &serde_wasm_bindgen::to_value(&out.combos).unwrap(),
    )?;
    Ok(obj.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn rolling_skewness_kurtosis_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(2 * len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn rolling_skewness_kurtosis_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, 2 * len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn rolling_skewness_kurtosis_into(
    data_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length: usize,
    smooth_length: usize,
) -> Result<(), JsValue> {
    if data_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to rolling_skewness_kurtosis_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, 2 * len);
        let (dst_skewness, dst_kurtosis) = out.split_at_mut(len);
        let input = RollingSkewnessKurtosisInput::from_slice(
            data,
            RollingSkewnessKurtosisParams {
                length: Some(length),
                smooth_length: Some(smooth_length),
            },
        );
        rolling_skewness_kurtosis_into_slice(dst_skewness, dst_kurtosis, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn rolling_skewness_kurtosis_batch_into(
    data_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length_start: usize,
    length_end: usize,
    length_step: usize,
    smooth_length_start: usize,
    smooth_length_end: usize,
    smooth_length_step: usize,
) -> Result<usize, JsValue> {
    if data_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to rolling_skewness_kurtosis_batch_into",
        ));
    }
    let sweep = RollingSkewnessKurtosisBatchRange {
        length: (length_start, length_end, length_step),
        smooth_length: (smooth_length_start, smooth_length_end, smooth_length_step),
    };
    let combos = expand_grid_checked(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let total = rows
        .checked_mul(len)
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| {
            JsValue::from_str("rows*cols overflow in rolling_skewness_kurtosis_batch_into")
        })?;
    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        let split = rows * len;
        let (dst_skewness, dst_kurtosis) = out.split_at_mut(split);
        rolling_skewness_kurtosis_batch_inner_into(
            data,
            &sweep,
            Kernel::Auto,
            false,
            dst_skewness,
            dst_kurtosis,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }
    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn rolling_skewness_kurtosis_output_into_js(
    data: &[f64],
    length: usize,
    smooth_length: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = rolling_skewness_kurtosis_js(data, length, smooth_length)?;
    crate::write_wasm_object_f64_outputs("rolling_skewness_kurtosis_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn rolling_skewness_kurtosis_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = rolling_skewness_kurtosis_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "rolling_skewness_kurtosis_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::dispatch::{
        compute_cpu, IndicatorComputeRequest, IndicatorDataRef, ParamKV, ParamValue,
    };

    fn sample_data(len: usize) -> Vec<f64> {
        (0..len)
            .map(|i| {
                let x = i as f64;
                100.0 + x * 0.05 + (x * 0.11).sin() * 2.7 + (x * 0.031).cos() * 0.9
            })
            .collect()
    }

    fn sma_series(values: &[f64], period: usize) -> Vec<f64> {
        if period == 1 {
            return values.to_vec();
        }
        let mut out = vec![f64::NAN; values.len()];
        let mut sum = 0.0;
        let mut count = 0usize;
        let mut buf = vec![0.0; period];
        let mut head = 0usize;
        for (i, &value) in values.iter().enumerate() {
            if !value.is_finite() {
                sum = 0.0;
                count = 0;
                head = 0;
                continue;
            }
            if count < period {
                buf[head] = value;
                sum += value;
                count += 1;
                head = (head + 1) % period;
                if count == period {
                    out[i] = sum / period as f64;
                }
            } else {
                let old = buf[head];
                buf[head] = value;
                head = (head + 1) % period;
                sum += value - old;
                out[i] = sum / period as f64;
            }
        }
        out
    }

    fn naive_indicator(data: &[f64], length: usize, smooth_length: usize) -> (Vec<f64>, Vec<f64>) {
        let mut raw_skew = vec![f64::NAN; data.len()];
        let mut raw_kurt = vec![f64::NAN; data.len()];
        for i in (length - 1)..data.len() {
            let window = &data[i + 1 - length..=i];
            if window.iter().any(|v| !v.is_finite()) {
                continue;
            }
            let n = length as f64;
            let mean = window.iter().sum::<f64>() / n;
            let mut m2 = 0.0;
            let mut m3 = 0.0;
            let mut m4 = 0.0;
            for &value in window {
                let dev = value - mean;
                let dev2 = dev * dev;
                m2 += dev2;
                m3 += dev2 * dev;
                m4 += dev2 * dev2;
            }
            m2 /= n;
            if m2 <= f64::EPSILON {
                continue;
            }
            let sigma = m2.sqrt();
            raw_skew[i] = (m3 / n) / (sigma * sigma * sigma);
            raw_kurt[i] = (m4 / n) / (m2 * m2) - 3.0;
        }
        (
            sma_series(&raw_skew, smooth_length),
            sma_series(&raw_kurt, smooth_length),
        )
    }

    fn assert_series_close(left: &[f64], right: &[f64], tol: f64) {
        assert_eq!(left.len(), right.len());
        for (&a, &b) in left.iter().zip(right.iter()) {
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan(), "left={a} right={b}");
            } else {
                assert!((a - b).abs() <= tol, "left={a} right={b}");
            }
        }
    }

    #[test]
    fn rolling_skewness_kurtosis_matches_naive() -> Result<(), Box<dyn Error>> {
        let data = sample_data(256);
        let input = RollingSkewnessKurtosisInput::from_slice(
            &data,
            RollingSkewnessKurtosisParams {
                length: Some(50),
                smooth_length: Some(3),
            },
        );
        let out = rolling_skewness_kurtosis_with_kernel(&input, Kernel::Scalar)?;
        let (expected_skew, expected_kurt) = naive_indicator(&data, 50, 3);
        assert_series_close(&out.skewness, &expected_skew, 1e-8);
        assert_series_close(&out.kurtosis, &expected_kurt, 1e-8);
        Ok(())
    }

    #[test]
    fn rolling_skewness_kurtosis_into_matches_api() -> Result<(), Box<dyn Error>> {
        let data = sample_data(220);
        let input = RollingSkewnessKurtosisInput::from_slice(
            &data,
            RollingSkewnessKurtosisParams {
                length: Some(40),
                smooth_length: Some(4),
            },
        );
        let base = rolling_skewness_kurtosis(&input)?;
        let mut skew = vec![0.0; data.len()];
        let mut kurt = vec![0.0; data.len()];
        rolling_skewness_kurtosis_into_slice(&mut skew, &mut kurt, &input, Kernel::Auto)?;
        assert_series_close(&base.skewness, &skew, 1e-12);
        assert_series_close(&base.kurtosis, &kurt, 1e-12);
        Ok(())
    }

    #[test]
    fn rolling_skewness_kurtosis_into_overwrites_stale_constant_data() -> Result<(), Box<dyn Error>>
    {
        let data = vec![42.0; 96];
        let input = RollingSkewnessKurtosisInput::from_slice(
            &data,
            RollingSkewnessKurtosisParams {
                length: Some(50),
                smooth_length: Some(3),
            },
        );
        let mut skew = vec![123.0; data.len()];
        let mut kurt = vec![456.0; data.len()];
        rolling_skewness_kurtosis_into_slice(&mut skew, &mut kurt, &input, Kernel::Auto)?;
        assert!(skew.iter().all(|v| v.is_nan()));
        assert!(kurt.iter().all(|v| v.is_nan()));
        Ok(())
    }

    #[test]
    fn rolling_skewness_kurtosis_stream_matches_batch() -> Result<(), Box<dyn Error>> {
        let data = sample_data(256);
        let batch = rolling_skewness_kurtosis(&RollingSkewnessKurtosisInput::from_slice(
            &data,
            RollingSkewnessKurtosisParams {
                length: Some(50),
                smooth_length: Some(3),
            },
        ))?;

        let mut stream = RollingSkewnessKurtosisStream::try_new(RollingSkewnessKurtosisParams {
            length: Some(50),
            smooth_length: Some(3),
        })?;
        let mut skew = vec![f64::NAN; data.len()];
        let mut kurt = vec![f64::NAN; data.len()];
        for (i, &value) in data.iter().enumerate() {
            if let Some((s, k)) = stream.update(value) {
                skew[i] = s;
                kurt[i] = k;
            }
        }

        assert_series_close(&batch.skewness, &skew, 1e-12);
        assert_series_close(&batch.kurtosis, &kurt, 1e-12);
        Ok(())
    }

    #[test]
    fn rolling_skewness_kurtosis_batch_single_matches_single() -> Result<(), Box<dyn Error>> {
        let data = sample_data(192);
        let single = rolling_skewness_kurtosis(&RollingSkewnessKurtosisInput::from_slice(
            &data,
            RollingSkewnessKurtosisParams {
                length: Some(30),
                smooth_length: Some(2),
            },
        ))?;

        let batch = rolling_skewness_kurtosis_batch_with_kernel(
            &data,
            &RollingSkewnessKurtosisBatchRange {
                length: (30, 30, 0),
                smooth_length: (2, 2, 0),
            },
            Kernel::Auto,
        )?;

        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, data.len());
        assert_series_close(&batch.skewness[..data.len()], &single.skewness, 1e-12);
        assert_series_close(&batch.kurtosis[..data.len()], &single.kurtosis, 1e-12);
        Ok(())
    }

    #[test]
    fn rolling_skewness_kurtosis_rejects_invalid_params() {
        let data = sample_data(64);

        let err = rolling_skewness_kurtosis(&RollingSkewnessKurtosisInput::from_slice(
            &data,
            RollingSkewnessKurtosisParams {
                length: Some(0),
                smooth_length: Some(3),
            },
        ))
        .unwrap_err();
        assert!(matches!(
            err,
            RollingSkewnessKurtosisError::InvalidLength { .. }
        ));

        let err = rolling_skewness_kurtosis(&RollingSkewnessKurtosisInput::from_slice(
            &data,
            RollingSkewnessKurtosisParams {
                length: Some(10),
                smooth_length: Some(0),
            },
        ))
        .unwrap_err();
        assert!(matches!(
            err,
            RollingSkewnessKurtosisError::InvalidSmoothLength { .. }
        ));

        let err = rolling_skewness_kurtosis_batch_with_kernel(
            &data,
            &RollingSkewnessKurtosisBatchRange {
                length: (10, 5, 1),
                smooth_length: (3, 3, 0),
            },
            Kernel::Auto,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            RollingSkewnessKurtosisError::InvalidRange { .. }
        ));
    }

    #[test]
    fn rolling_skewness_kurtosis_dispatch_compute_returns_outputs() -> Result<(), Box<dyn Error>> {
        let data = sample_data(200);
        let req_skew = IndicatorComputeRequest {
            indicator_id: "rolling_skewness_kurtosis",
            output_id: Some("skewness"),
            data: IndicatorDataRef::Slice { values: &data },
            params: &[
                ParamKV {
                    key: "length",
                    value: ParamValue::Int(50),
                },
                ParamKV {
                    key: "smooth_length",
                    value: ParamValue::Int(3),
                },
            ],
            kernel: Kernel::Auto,
        };
        let out_skew = compute_cpu(req_skew)?;
        assert_eq!(out_skew.output_id, "skewness");
        assert_eq!(out_skew.rows, 1);
        assert_eq!(out_skew.cols, data.len());

        let req_kurt = IndicatorComputeRequest {
            indicator_id: "rolling_skewness_kurtosis",
            output_id: Some("kurtosis"),
            data: IndicatorDataRef::Slice { values: &data },
            params: &[
                ParamKV {
                    key: "length",
                    value: ParamValue::Int(50),
                },
                ParamKV {
                    key: "smooth_length",
                    value: ParamValue::Int(3),
                },
            ],
            kernel: Kernel::Auto,
        };
        let out_kurt = compute_cpu(req_kurt)?;
        assert_eq!(out_kurt.output_id, "kurtosis");
        assert_eq!(out_kurt.rows, 1);
        assert_eq!(out_kurt.cols, data.len());
        Ok(())
    }
}
