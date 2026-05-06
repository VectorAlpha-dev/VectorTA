#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray1};
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
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

const DEFAULT_LENGTH: usize = 20;
const DEFAULT_MAX_LAG: usize = 99;
const TEST_SIGNAL_PERIOD: f64 = 30.0;

impl<'a> AsRef<[f64]> for AutocorrelationIndicatorInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            AutocorrelationIndicatorData::Slice(slice) => slice,
            AutocorrelationIndicatorData::Candles { candles, source } => {
                source_type(candles, source)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum AutocorrelationIndicatorData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct AutocorrelationIndicatorOutput {
    pub filtered: Vec<f64>,
    pub correlations: Vec<f64>,
    pub lag_count: usize,
    pub cols: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutocorrelationIndicatorOutputField {
    Filtered,
    Correlation { lag: usize },
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct AutocorrelationIndicatorParams {
    pub length: Option<usize>,
    pub max_lag: Option<usize>,
    pub use_test_signal: Option<bool>,
}

impl Default for AutocorrelationIndicatorParams {
    fn default() -> Self {
        Self {
            length: Some(DEFAULT_LENGTH),
            max_lag: Some(DEFAULT_MAX_LAG),
            use_test_signal: Some(false),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AutocorrelationIndicatorInput<'a> {
    pub data: AutocorrelationIndicatorData<'a>,
    pub params: AutocorrelationIndicatorParams,
}

impl<'a> AutocorrelationIndicatorInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: AutocorrelationIndicatorParams,
    ) -> Self {
        Self {
            data: AutocorrelationIndicatorData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: AutocorrelationIndicatorParams) -> Self {
        Self {
            data: AutocorrelationIndicatorData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", AutocorrelationIndicatorParams::default())
    }

    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(DEFAULT_LENGTH)
    }

    #[inline]
    pub fn get_max_lag(&self) -> usize {
        self.params.max_lag.unwrap_or(DEFAULT_MAX_LAG)
    }

    #[inline]
    pub fn get_use_test_signal(&self) -> bool {
        self.params.use_test_signal.unwrap_or(false)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct AutocorrelationIndicatorBuilder {
    length: Option<usize>,
    max_lag: Option<usize>,
    use_test_signal: Option<bool>,
    kernel: Kernel,
}

impl Default for AutocorrelationIndicatorBuilder {
    fn default() -> Self {
        Self {
            length: None,
            max_lag: None,
            use_test_signal: None,
            kernel: Kernel::Auto,
        }
    }
}

impl AutocorrelationIndicatorBuilder {
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
    pub fn max_lag(mut self, value: usize) -> Self {
        self.max_lag = Some(value);
        self
    }

    #[inline(always)]
    pub fn use_test_signal(mut self, value: bool) -> Self {
        self.use_test_signal = Some(value);
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
    ) -> Result<AutocorrelationIndicatorOutput, AutocorrelationIndicatorError> {
        let params = AutocorrelationIndicatorParams {
            length: self.length,
            max_lag: self.max_lag,
            use_test_signal: self.use_test_signal,
        };
        autocorrelation_indicator_with_kernel(
            &AutocorrelationIndicatorInput::from_candles(candles, "close", params),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<AutocorrelationIndicatorOutput, AutocorrelationIndicatorError> {
        let params = AutocorrelationIndicatorParams {
            length: self.length,
            max_lag: self.max_lag,
            use_test_signal: self.use_test_signal,
        };
        autocorrelation_indicator_with_kernel(
            &AutocorrelationIndicatorInput::from_candles(candles, source, params),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<AutocorrelationIndicatorOutput, AutocorrelationIndicatorError> {
        let params = AutocorrelationIndicatorParams {
            length: self.length,
            max_lag: self.max_lag,
            use_test_signal: self.use_test_signal,
        };
        autocorrelation_indicator_with_kernel(
            &AutocorrelationIndicatorInput::from_slice(data, params),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<AutocorrelationIndicatorStream, AutocorrelationIndicatorError> {
        AutocorrelationIndicatorStream::try_new(AutocorrelationIndicatorParams {
            length: self.length,
            max_lag: self.max_lag,
            use_test_signal: self.use_test_signal,
        })
    }
}

#[derive(Debug, Error)]
pub enum AutocorrelationIndicatorError {
    #[error("autocorrelation_indicator: Input data slice is empty.")]
    EmptyInputData,
    #[error("autocorrelation_indicator: All values are NaN.")]
    AllValuesNaN,
    #[error(
        "autocorrelation_indicator: Invalid length: length = {length}, data length = {data_len}"
    )]
    InvalidLength { length: usize, data_len: usize },
    #[error("autocorrelation_indicator: Invalid max_lag: {max_lag}")]
    InvalidMaxLag { max_lag: usize },
    #[error(
        "autocorrelation_indicator: Not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "autocorrelation_indicator: Filtered output length mismatch: expected = {expected}, got = {got}"
    )]
    FilteredOutputLengthMismatch { expected: usize, got: usize },
    #[error(
        "autocorrelation_indicator: Correlations output length mismatch: expected = {expected}, got = {got}"
    )]
    CorrelationsOutputLengthMismatch { expected: usize, got: usize },
    #[error("autocorrelation_indicator: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("autocorrelation_indicator: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("autocorrelation_indicator: Invalid input: {msg}")]
    InvalidInput { msg: String },
}

#[derive(Debug, Clone)]
pub struct AutocorrelationIndicatorStreamPoint {
    pub filtered: f64,
    pub correlations: Vec<f64>,
}

#[derive(Debug, Clone)]
struct UltimateSmootherState {
    c1: f64,
    c2: f64,
    c3: f64,
    count: usize,
    prev_src1: f64,
    prev_src2: f64,
    prev_us1: f64,
    prev_us2: f64,
}

impl UltimateSmootherState {
    #[inline(always)]
    fn new(period: usize) -> Self {
        let period_f = period as f64;
        let a1 = (-1.414 * std::f64::consts::PI / period_f).exp();
        let c2 = 2.0 * a1 * (1.414 * std::f64::consts::PI / period_f).cos();
        let c3 = -a1 * a1;
        let c1 = (1.0 + c2 - c3) * 0.25;
        Self {
            c1,
            c2,
            c3,
            count: 0,
            prev_src1: f64::NAN,
            prev_src2: f64::NAN,
            prev_us1: f64::NAN,
            prev_us2: f64::NAN,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.count = 0;
        self.prev_src1 = f64::NAN;
        self.prev_src2 = f64::NAN;
        self.prev_us1 = f64::NAN;
        self.prev_us2 = f64::NAN;
    }

    #[inline(always)]
    fn update(&mut self, src: f64) -> f64 {
        let out = if self.count >= 4 {
            (1.0 - self.c1) * src + (2.0 * self.c1 - self.c2) * self.prev_src1
                - (self.c1 + self.c3) * self.prev_src2
                + self.c2 * self.prev_us1
                + self.c3 * self.prev_us2
        } else {
            src
        };
        self.prev_src2 = self.prev_src1;
        self.prev_src1 = src;
        self.prev_us2 = self.prev_us1;
        self.prev_us1 = out;
        self.count += 1;
        out
    }
}

#[derive(Debug, Clone)]
pub struct AutocorrelationIndicatorStream {
    length: usize,
    max_lag: usize,
    use_test_signal: bool,
    smoother: UltimateSmootherState,
    filtered_history: Vec<f64>,
    next_index: usize,
}

impl AutocorrelationIndicatorStream {
    #[inline(always)]
    pub fn try_new(
        params: AutocorrelationIndicatorParams,
    ) -> Result<Self, AutocorrelationIndicatorError> {
        let length = params.length.unwrap_or(DEFAULT_LENGTH);
        let max_lag = params.max_lag.unwrap_or(DEFAULT_MAX_LAG);
        validate_params(length, max_lag, usize::MAX)?;
        Ok(Self {
            length,
            max_lag,
            use_test_signal: params.use_test_signal.unwrap_or(false),
            smoother: UltimateSmootherState::new(length),
            filtered_history: Vec::new(),
            next_index: 0,
        })
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.smoother.reset();
        self.filtered_history.clear();
        self.next_index = 0;
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<AutocorrelationIndicatorStreamPoint> {
        if !self.use_test_signal && !value.is_finite() {
            self.reset();
            return None;
        }

        let src = if self.use_test_signal {
            (2.0 * std::f64::consts::PI * self.next_index as f64 / TEST_SIGNAL_PERIOD).sin()
        } else {
            value
        };
        let filtered = self.smoother.update(src);
        self.filtered_history.push(filtered);
        self.next_index += 1;

        let len = self.filtered_history.len();
        let mut correlations = vec![f64::NAN; self.max_lag];
        let window = self.length as f64;
        let t = len - 1;
        for lag in 1..=self.max_lag {
            if t + 1 < self.length + lag {
                continue;
            }
            let start_x = t + 1 - self.length;
            let start_y = start_x - lag;
            let mut sx = 0.0;
            let mut sy = 0.0;
            let mut sxx = 0.0;
            let mut syy = 0.0;
            let mut sxy = 0.0;
            for j in 0..self.length {
                let x = self.filtered_history[start_x + j];
                let y = self.filtered_history[start_y + j];
                sx += x;
                sy += y;
                sxx += x * x;
                syy += y * y;
                sxy += x * y;
            }
            let ca1 = window * sxx - sx * sx;
            let ca2 = window * syy - sy * sy;
            correlations[lag - 1] = if ca1 > 0.0 && ca2 > 0.0 {
                let ca3 = window * sxy - sx * sy;
                ca3 / (ca1 * ca2).sqrt()
            } else {
                0.0
            };
        }

        Some(AutocorrelationIndicatorStreamPoint {
            filtered,
            correlations,
        })
    }

    #[inline(always)]
    pub fn get_warmup_period(&self) -> usize {
        self.length.saturating_add(self.max_lag).saturating_sub(1)
    }
}

#[derive(Debug, Clone)]
pub struct AutocorrelationIndicatorBatchRange {
    pub length: (usize, usize, usize),
    pub max_lag: Option<usize>,
    pub use_test_signal: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct AutocorrelationIndicatorBatchOutput {
    pub filtered: Vec<f64>,
    pub correlations: Vec<f64>,
    pub combos: Vec<AutocorrelationIndicatorParams>,
    pub rows: usize,
    pub cols: usize,
    pub lag_count: usize,
}

#[derive(Copy, Clone, Debug)]
pub struct AutocorrelationIndicatorBatchBuilder {
    length_range: (usize, usize, usize),
    max_lag: Option<usize>,
    use_test_signal: Option<bool>,
    kernel: Kernel,
}

impl Default for AutocorrelationIndicatorBatchBuilder {
    fn default() -> Self {
        Self {
            length_range: (DEFAULT_LENGTH, DEFAULT_LENGTH, 0),
            max_lag: Some(DEFAULT_MAX_LAG),
            use_test_signal: Some(false),
            kernel: Kernel::Auto,
        }
    }
}

impl AutocorrelationIndicatorBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn length_range(mut self, value: (usize, usize, usize)) -> Self {
        self.length_range = value;
        self
    }

    #[inline(always)]
    pub fn max_lag(mut self, value: usize) -> Self {
        self.max_lag = Some(value);
        self
    }

    #[inline(always)]
    pub fn use_test_signal(mut self, value: bool) -> Self {
        self.use_test_signal = Some(value);
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
    ) -> Result<AutocorrelationIndicatorBatchOutput, AutocorrelationIndicatorError> {
        autocorrelation_indicator_batch_with_kernel(
            data,
            &AutocorrelationIndicatorBatchRange {
                length: self.length_range,
                max_lag: self.max_lag,
                use_test_signal: self.use_test_signal,
            },
            self.kernel,
        )
    }
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
    max_lag: usize,
    data_len: usize,
) -> Result<(), AutocorrelationIndicatorError> {
    if length == 0 || (data_len != usize::MAX && length > data_len) {
        return Err(AutocorrelationIndicatorError::InvalidLength { length, data_len });
    }
    if max_lag == 0 {
        return Err(AutocorrelationIndicatorError::InvalidMaxLag { max_lag });
    }
    Ok(())
}

#[inline(always)]
fn validate_common(
    data: &[f64],
    length: usize,
    max_lag: usize,
    use_test_signal: bool,
) -> Result<(), AutocorrelationIndicatorError> {
    if data.is_empty() {
        return Err(AutocorrelationIndicatorError::EmptyInputData);
    }
    validate_params(length, max_lag, data.len())?;
    if use_test_signal {
        return Ok(());
    }
    let max_run = longest_valid_run(data);
    if max_run == 0 {
        return Err(AutocorrelationIndicatorError::AllValuesNaN);
    }
    if max_run < length {
        return Err(AutocorrelationIndicatorError::NotEnoughValidData {
            needed: length,
            valid: max_run,
        });
    }
    Ok(())
}

#[inline(always)]
fn filter_series(data: &[f64], length: usize, use_test_signal: bool, out: &mut [f64]) {
    out.fill(f64::NAN);
    let mut smoother = UltimateSmootherState::new(length);
    for (i, dst) in out.iter_mut().enumerate() {
        let raw = if use_test_signal {
            (2.0 * std::f64::consts::PI * i as f64 / TEST_SIGNAL_PERIOD).sin()
        } else {
            data[i]
        };
        if !raw.is_finite() {
            smoother.reset();
            continue;
        }
        *dst = smoother.update(raw);
    }
}

#[inline(always)]
fn compute_segment_correlations(
    filtered: &[f64],
    length: usize,
    max_lag: usize,
    cols: usize,
    offset: usize,
    correlations: &mut [f64],
) {
    if filtered.len() < length + 1 {
        return;
    }
    let mut prefix = vec![0.0; filtered.len() + 1];
    let mut prefix_sq = vec![0.0; filtered.len() + 1];
    for (i, &value) in filtered.iter().enumerate() {
        prefix[i + 1] = prefix[i] + value;
        prefix_sq[i + 1] = prefix_sq[i] + value * value;
    }

    let length_f = length as f64;
    for lag in 1..=max_lag {
        if filtered.len() < length + lag {
            continue;
        }
        let row_start = (lag - 1) * cols;
        let row = &mut correlations[row_start..row_start + cols];
        let out = &mut row[offset..offset + filtered.len()];
        let t0 = lag + length - 1;
        let mut cross = 0.0;
        for j in 0..length {
            cross += filtered[lag + j] * filtered[j];
        }
        for t in t0..filtered.len() {
            let x_start = t + 1 - length;
            let y_start = x_start - lag;
            let sx = prefix[t + 1] - prefix[x_start];
            let sxx = prefix_sq[t + 1] - prefix_sq[x_start];
            let sy = prefix[y_start + length] - prefix[y_start];
            let syy = prefix_sq[y_start + length] - prefix_sq[y_start];
            let ca1 = length_f * sxx - sx * sx;
            let ca2 = length_f * syy - sy * sy;
            out[t] = if ca1 > 0.0 && ca2 > 0.0 {
                let ca3 = length_f * cross - sx * sy;
                ca3 / (ca1 * ca2).sqrt()
            } else {
                0.0
            };
            if t + 1 < filtered.len() {
                cross +=
                    filtered[t + 1] * filtered[t + 1 - lag] - filtered[x_start] * filtered[y_start];
            }
        }
    }
}

#[inline(always)]
fn compute_segment_correlation_lag(
    filtered: &[f64],
    length: usize,
    lag: usize,
    offset: usize,
    out: &mut [f64],
) {
    if lag == 0 || filtered.len() < length + lag {
        return;
    }
    let mut prefix = vec![0.0; filtered.len() + 1];
    let mut prefix_sq = vec![0.0; filtered.len() + 1];
    for (i, &value) in filtered.iter().enumerate() {
        prefix[i + 1] = prefix[i] + value;
        prefix_sq[i + 1] = prefix_sq[i] + value * value;
    }

    let length_f = length as f64;
    let t0 = lag + length - 1;
    let mut cross = 0.0;
    for j in 0..length {
        cross += filtered[lag + j] * filtered[j];
    }
    for t in t0..filtered.len() {
        let x_start = t + 1 - length;
        let y_start = x_start - lag;
        let sx = prefix[t + 1] - prefix[x_start];
        let sxx = prefix_sq[t + 1] - prefix_sq[x_start];
        let sy = prefix[y_start + length] - prefix[y_start];
        let syy = prefix_sq[y_start + length] - prefix_sq[y_start];
        let ca1 = length_f * sxx - sx * sx;
        let ca2 = length_f * syy - sy * sy;
        out[offset + t] = if ca1 > 0.0 && ca2 > 0.0 {
            let ca3 = length_f * cross - sx * sy;
            ca3 / (ca1 * ca2).sqrt()
        } else {
            0.0
        };
        if t + 1 < filtered.len() {
            cross +=
                filtered[t + 1] * filtered[t + 1 - lag] - filtered[x_start] * filtered[y_start];
        }
    }
}

#[inline(always)]
fn compute_full(
    data: &[f64],
    length: usize,
    max_lag: usize,
    use_test_signal: bool,
    filtered_out: &mut [f64],
    correlations_out: &mut [f64],
) {
    filter_series(data, length, use_test_signal, filtered_out);
    correlations_out.fill(f64::NAN);

    let mut seg_start = 0usize;
    while seg_start < filtered_out.len() {
        while seg_start < filtered_out.len() && !filtered_out[seg_start].is_finite() {
            seg_start += 1;
        }
        if seg_start >= filtered_out.len() {
            break;
        }
        let mut seg_end = seg_start + 1;
        while seg_end < filtered_out.len() && filtered_out[seg_end].is_finite() {
            seg_end += 1;
        }
        compute_segment_correlations(
            &filtered_out[seg_start..seg_end],
            length,
            max_lag,
            filtered_out.len(),
            seg_start,
            correlations_out,
        );
        seg_start = seg_end;
    }
}

#[inline]
pub fn autocorrelation_indicator(
    input: &AutocorrelationIndicatorInput,
) -> Result<AutocorrelationIndicatorOutput, AutocorrelationIndicatorError> {
    autocorrelation_indicator_with_kernel(input, Kernel::Auto)
}

pub fn autocorrelation_indicator_with_kernel(
    input: &AutocorrelationIndicatorInput,
    kernel: Kernel,
) -> Result<AutocorrelationIndicatorOutput, AutocorrelationIndicatorError> {
    let data = input.as_ref();
    let length = input.get_length();
    let max_lag = input.get_max_lag();
    let use_test_signal = input.get_use_test_signal();
    validate_common(data, length, max_lag, use_test_signal)?;

    let _chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let mut filtered = alloc_with_nan_prefix(data.len(), 0);
    filtered.resize(data.len(), f64::NAN);
    let mut correlations = vec![f64::NAN; data.len() * max_lag];
    compute_full(
        data,
        length,
        max_lag,
        use_test_signal,
        &mut filtered,
        &mut correlations,
    );

    Ok(AutocorrelationIndicatorOutput {
        filtered,
        correlations,
        lag_count: max_lag,
        cols: data.len(),
    })
}

pub fn autocorrelation_indicator_into_slice(
    filtered_out: &mut [f64],
    correlations_out: &mut [f64],
    input: &AutocorrelationIndicatorInput,
    kernel: Kernel,
) -> Result<(), AutocorrelationIndicatorError> {
    let data = input.as_ref();
    let length = input.get_length();
    let max_lag = input.get_max_lag();
    let use_test_signal = input.get_use_test_signal();
    validate_common(data, length, max_lag, use_test_signal)?;

    if filtered_out.len() != data.len() {
        return Err(
            AutocorrelationIndicatorError::FilteredOutputLengthMismatch {
                expected: data.len(),
                got: filtered_out.len(),
            },
        );
    }
    let expected_corr = data.len().checked_mul(max_lag).ok_or_else(|| {
        AutocorrelationIndicatorError::InvalidInput {
            msg: "autocorrelation_indicator: correlations size overflow".to_string(),
        }
    })?;
    if correlations_out.len() != expected_corr {
        return Err(
            AutocorrelationIndicatorError::CorrelationsOutputLengthMismatch {
                expected: expected_corr,
                got: correlations_out.len(),
            },
        );
    }

    let _chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    compute_full(
        data,
        length,
        max_lag,
        use_test_signal,
        filtered_out,
        correlations_out,
    );
    Ok(())
}

pub fn autocorrelation_indicator_output_into_slice(
    dst: &mut [f64],
    input: &AutocorrelationIndicatorInput,
    kernel: Kernel,
    field: AutocorrelationIndicatorOutputField,
) -> Result<(), AutocorrelationIndicatorError> {
    let data = input.as_ref();
    let length = input.get_length();
    let max_lag = input.get_max_lag();
    let use_test_signal = input.get_use_test_signal();
    validate_common(data, length, max_lag, use_test_signal)?;
    if dst.len() != data.len() {
        return Err(
            AutocorrelationIndicatorError::FilteredOutputLengthMismatch {
                expected: data.len(),
                got: dst.len(),
            },
        );
    }
    let _chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    match field {
        AutocorrelationIndicatorOutputField::Filtered => {
            filter_series(data, length, use_test_signal, dst);
        }
        AutocorrelationIndicatorOutputField::Correlation { lag } => {
            if lag == 0 || lag > max_lag {
                return Err(AutocorrelationIndicatorError::InvalidMaxLag { max_lag: lag });
            }
            let mut filtered = alloc_with_nan_prefix(data.len(), 0);
            filter_series(data, length, use_test_signal, &mut filtered);
            dst.fill(f64::NAN);

            let mut seg_start = 0usize;
            while seg_start < filtered.len() {
                while seg_start < filtered.len() && !filtered[seg_start].is_finite() {
                    seg_start += 1;
                }
                if seg_start >= filtered.len() {
                    break;
                }
                let mut seg_end = seg_start + 1;
                while seg_end < filtered.len() && filtered[seg_end].is_finite() {
                    seg_end += 1;
                }
                compute_segment_correlation_lag(
                    &filtered[seg_start..seg_end],
                    length,
                    lag,
                    seg_start,
                    dst,
                );
                seg_start = seg_end;
            }
        }
    }
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn autocorrelation_indicator_into(
    input: &AutocorrelationIndicatorInput,
    filtered_out: &mut [f64],
    correlations_out: &mut [f64],
) -> Result<(), AutocorrelationIndicatorError> {
    autocorrelation_indicator_into_slice(filtered_out, correlations_out, input, Kernel::Auto)
}

#[inline(always)]
fn expand_grid_checked(
    sweep: &AutocorrelationIndicatorBatchRange,
) -> Result<Vec<AutocorrelationIndicatorParams>, AutocorrelationIndicatorError> {
    let (start, end, step) = sweep.length;
    if start == 0 || end == 0 || end < start || (start != end && step == 0) {
        return Err(AutocorrelationIndicatorError::InvalidRange { start, end, step });
    }
    let max_lag = sweep.max_lag.unwrap_or(DEFAULT_MAX_LAG);
    let use_test_signal = sweep.use_test_signal.unwrap_or(false);
    let mut combos = Vec::new();
    if start == end {
        combos.push(AutocorrelationIndicatorParams {
            length: Some(start),
            max_lag: Some(max_lag),
            use_test_signal: Some(use_test_signal),
        });
        return Ok(combos);
    }
    let mut current = start;
    while current <= end {
        combos.push(AutocorrelationIndicatorParams {
            length: Some(current),
            max_lag: Some(max_lag),
            use_test_signal: Some(use_test_signal),
        });
        match current.checked_add(step) {
            Some(next) if next > current => current = next,
            _ => break,
        }
    }
    Ok(combos)
}

pub fn expand_grid_autocorrelation_indicator(
    sweep: &AutocorrelationIndicatorBatchRange,
) -> Result<Vec<AutocorrelationIndicatorParams>, AutocorrelationIndicatorError> {
    expand_grid_checked(sweep)
}

pub fn autocorrelation_indicator_batch_with_kernel(
    data: &[f64],
    sweep: &AutocorrelationIndicatorBatchRange,
    kernel: Kernel,
) -> Result<AutocorrelationIndicatorBatchOutput, AutocorrelationIndicatorError> {
    autocorrelation_indicator_batch_inner(data, sweep, kernel, true)
}

pub fn autocorrelation_indicator_batch_slice(
    data: &[f64],
    sweep: &AutocorrelationIndicatorBatchRange,
    kernel: Kernel,
) -> Result<AutocorrelationIndicatorBatchOutput, AutocorrelationIndicatorError> {
    autocorrelation_indicator_batch_inner(data, sweep, kernel, false)
}

pub fn autocorrelation_indicator_batch_par_slice(
    data: &[f64],
    sweep: &AutocorrelationIndicatorBatchRange,
    kernel: Kernel,
) -> Result<AutocorrelationIndicatorBatchOutput, AutocorrelationIndicatorError> {
    autocorrelation_indicator_batch_inner(data, sweep, kernel, true)
}

fn autocorrelation_indicator_batch_inner(
    data: &[f64],
    sweep: &AutocorrelationIndicatorBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<AutocorrelationIndicatorBatchOutput, AutocorrelationIndicatorError> {
    let combos = expand_grid_checked(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    let max_lag = sweep.max_lag.unwrap_or(DEFAULT_MAX_LAG);
    let total_filtered =
        rows.checked_mul(cols)
            .ok_or_else(|| AutocorrelationIndicatorError::InvalidInput {
                msg: "autocorrelation_indicator: rows*cols overflow in batch".to_string(),
            })?;
    let total_corr = total_filtered.checked_mul(max_lag).ok_or_else(|| {
        AutocorrelationIndicatorError::InvalidInput {
            msg: "autocorrelation_indicator: rows*lags*cols overflow in batch".to_string(),
        }
    })?;

    let mut filtered_mu = make_uninit_matrix(rows, cols);
    let mut correlations_mu = make_uninit_matrix(rows * max_lag, cols);
    let mut filtered = unsafe {
        Vec::from_raw_parts(
            filtered_mu.as_mut_ptr() as *mut f64,
            filtered_mu.len(),
            filtered_mu.capacity(),
        )
    };
    let mut correlations = unsafe {
        Vec::from_raw_parts(
            correlations_mu.as_mut_ptr() as *mut f64,
            correlations_mu.len(),
            correlations_mu.capacity(),
        )
    };
    std::mem::forget(filtered_mu);
    std::mem::forget(correlations_mu);

    debug_assert_eq!(filtered.len(), total_filtered);
    debug_assert_eq!(correlations.len(), total_corr);

    autocorrelation_indicator_batch_inner_into(
        data,
        sweep,
        kernel,
        parallel,
        &mut filtered,
        &mut correlations,
    )?;

    Ok(AutocorrelationIndicatorBatchOutput {
        filtered,
        correlations,
        combos,
        rows,
        cols,
        lag_count: max_lag,
    })
}

fn autocorrelation_indicator_batch_inner_into(
    data: &[f64],
    sweep: &AutocorrelationIndicatorBatchRange,
    kernel: Kernel,
    parallel: bool,
    filtered_out: &mut [f64],
    correlations_out: &mut [f64],
) -> Result<Vec<AutocorrelationIndicatorParams>, AutocorrelationIndicatorError> {
    match kernel {
        Kernel::Auto
        | Kernel::Scalar
        | Kernel::ScalarBatch
        | Kernel::Avx2
        | Kernel::Avx2Batch
        | Kernel::Avx512
        | Kernel::Avx512Batch => {}
        other => return Err(AutocorrelationIndicatorError::InvalidKernelForBatch(other)),
    }

    let combos = expand_grid_checked(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    let max_lag = sweep.max_lag.unwrap_or(DEFAULT_MAX_LAG);
    validate_common(
        data,
        combos
            .iter()
            .map(|params| params.length.unwrap_or(DEFAULT_LENGTH))
            .max()
            .unwrap_or(DEFAULT_LENGTH),
        max_lag,
        sweep.use_test_signal.unwrap_or(false),
    )?;

    let expected_filtered =
        rows.checked_mul(cols)
            .ok_or_else(|| AutocorrelationIndicatorError::InvalidInput {
                msg: "autocorrelation_indicator: rows*cols overflow in batch_into".to_string(),
            })?;
    let expected_corr = expected_filtered.checked_mul(max_lag).ok_or_else(|| {
        AutocorrelationIndicatorError::InvalidInput {
            msg: "autocorrelation_indicator: rows*lags*cols overflow in batch_into".to_string(),
        }
    })?;
    if filtered_out.len() != expected_filtered {
        return Err(
            AutocorrelationIndicatorError::FilteredOutputLengthMismatch {
                expected: expected_filtered,
                got: filtered_out.len(),
            },
        );
    }
    if correlations_out.len() != expected_corr {
        return Err(
            AutocorrelationIndicatorError::CorrelationsOutputLengthMismatch {
                expected: expected_corr,
                got: correlations_out.len(),
            },
        );
    }

    let _chosen = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other => other,
    };

    let worker = |row: usize, dst_filtered: &mut [f64], dst_corr: &mut [f64]| {
        let params = &combos[row];
        compute_full(
            data,
            params.length.unwrap_or(DEFAULT_LENGTH),
            params.max_lag.unwrap_or(DEFAULT_MAX_LAG),
            params.use_test_signal.unwrap_or(false),
            dst_filtered,
            dst_corr,
        );
    };

    if parallel && rows > 1 {
        #[cfg(not(target_arch = "wasm32"))]
        {
            filtered_out
                .par_chunks_mut(cols)
                .zip(correlations_out.par_chunks_mut(max_lag * cols))
                .enumerate()
                .for_each(|(row, (dst_filtered, dst_corr))| worker(row, dst_filtered, dst_corr));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, (dst_filtered, dst_corr)) in filtered_out
                .chunks_mut(cols)
                .zip(correlations_out.chunks_mut(max_lag * cols))
                .enumerate()
            {
                worker(row, dst_filtered, dst_corr);
            }
        }
    } else {
        for (row, (dst_filtered, dst_corr)) in filtered_out
            .chunks_mut(cols)
            .zip(correlations_out.chunks_mut(max_lag * cols))
            .enumerate()
        {
            worker(row, dst_filtered, dst_corr);
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "autocorrelation_indicator")]
#[pyo3(signature = (data, length=DEFAULT_LENGTH, max_lag=DEFAULT_MAX_LAG, use_test_signal=false, kernel=None))]
pub fn autocorrelation_indicator_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length: usize,
    max_lag: usize,
    use_test_signal: bool,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let input = AutocorrelationIndicatorInput::from_slice(
        data,
        AutocorrelationIndicatorParams {
            length: Some(length),
            max_lag: Some(max_lag),
            use_test_signal: Some(use_test_signal),
        },
    );
    let out = py
        .allow_threads(|| autocorrelation_indicator_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("filtered", out.filtered.into_pyarray(py))?;
    dict.set_item(
        "correlations",
        out.correlations
            .into_pyarray(py)
            .reshape((out.lag_count, out.cols))?,
    )?;
    dict.set_item("lag_count", out.lag_count)?;
    dict.set_item("cols", out.cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "AutocorrelationIndicatorStream")]
pub struct AutocorrelationIndicatorStreamPy {
    stream: AutocorrelationIndicatorStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl AutocorrelationIndicatorStreamPy {
    #[new]
    #[pyo3(signature = (length=DEFAULT_LENGTH, max_lag=DEFAULT_MAX_LAG, use_test_signal=false))]
    fn new(length: usize, max_lag: usize, use_test_signal: bool) -> PyResult<Self> {
        let stream = AutocorrelationIndicatorStream::try_new(AutocorrelationIndicatorParams {
            length: Some(length),
            max_lag: Some(max_lag),
            use_test_signal: Some(use_test_signal),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, value: f64) -> Option<(f64, Vec<f64>)> {
        self.stream
            .update(value)
            .map(|point| (point.filtered, point.correlations))
    }

    fn reset(&mut self) {
        self.stream.reset();
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "autocorrelation_indicator_batch")]
#[pyo3(signature = (data, length_range=(DEFAULT_LENGTH, DEFAULT_LENGTH, 0), max_lag=DEFAULT_MAX_LAG, use_test_signal=false, kernel=None))]
pub fn autocorrelation_indicator_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    max_lag: usize,
    use_test_signal: bool,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let output = py
        .allow_threads(|| {
            autocorrelation_indicator_batch_with_kernel(
                data,
                &AutocorrelationIndicatorBatchRange {
                    length: length_range,
                    max_lag: Some(max_lag),
                    use_test_signal: Some(use_test_signal),
                },
                kern,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item(
        "filtered",
        output
            .filtered
            .into_pyarray(py)
            .reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "correlations",
        output.correlations.into_pyarray(py).reshape((
            output.rows,
            output.lag_count,
            output.cols,
        ))?,
    )?;
    dict.set_item(
        "lengths",
        output
            .combos
            .iter()
            .map(|params| params.length.unwrap_or(DEFAULT_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", output.rows)?;
    dict.set_item("cols", output.cols)?;
    dict.set_item("lag_count", output.lag_count)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_autocorrelation_indicator_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(autocorrelation_indicator_py, m)?)?;
    m.add_function(wrap_pyfunction!(autocorrelation_indicator_batch_py, m)?)?;
    m.add_class::<AutocorrelationIndicatorStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutocorrelationIndicatorBatchConfig {
    pub length_range: Vec<usize>,
    pub max_lag: usize,
    pub use_test_signal: bool,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = autocorrelation_indicator_js)]
pub fn autocorrelation_indicator_js(
    data: &[f64],
    length: usize,
    max_lag: usize,
    use_test_signal: bool,
) -> Result<JsValue, JsValue> {
    let input = AutocorrelationIndicatorInput::from_slice(
        data,
        AutocorrelationIndicatorParams {
            length: Some(length),
            max_lag: Some(max_lag),
            use_test_signal: Some(use_test_signal),
        },
    );
    let out = autocorrelation_indicator_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("filtered"),
        &serde_wasm_bindgen::to_value(&out.filtered).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("correlations"),
        &serde_wasm_bindgen::to_value(&out.correlations).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("lag_count"),
        &JsValue::from_f64(out.lag_count as f64),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("cols"),
        &JsValue::from_f64(out.cols as f64),
    )?;
    Ok(obj.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = autocorrelation_indicator_batch_js)]
pub fn autocorrelation_indicator_batch_js(
    data: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: AutocorrelationIndicatorBatchConfig =
        serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&e.to_string()))?;
    if config.length_range.len() != 3 {
        return Err(JsValue::from_str(
            "length_range must have exactly 3 elements [start, end, step]",
        ));
    }
    let out = autocorrelation_indicator_batch_with_kernel(
        data,
        &AutocorrelationIndicatorBatchRange {
            length: (
                config.length_range[0],
                config.length_range[1],
                config.length_range[2],
            ),
            max_lag: Some(config.max_lag),
            use_test_signal: Some(config.use_test_signal),
        },
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("filtered"),
        &serde_wasm_bindgen::to_value(&out.filtered).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("correlations"),
        &serde_wasm_bindgen::to_value(&out.correlations).unwrap(),
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
        &JsValue::from_str("lag_count"),
        &JsValue::from_f64(out.lag_count as f64),
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
pub fn autocorrelation_indicator_alloc(len: usize, max_lag: usize) -> *mut f64 {
    let total = len
        .checked_mul(max_lag.saturating_add(1))
        .expect("autocorrelation_indicator_alloc overflow");
    let mut buf = vec![0.0; total];
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn autocorrelation_indicator_free(ptr: *mut f64, len: usize, max_lag: usize) {
    if ptr.is_null() {
        return;
    }
    let total = len
        .checked_mul(max_lag.saturating_add(1))
        .expect("autocorrelation_indicator_free overflow");
    unsafe {
        let _ = Vec::from_raw_parts(ptr, 0, total);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn autocorrelation_indicator_into(
    data_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length: usize,
    max_lag: usize,
    use_test_signal: bool,
) -> Result<(), JsValue> {
    if data_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to autocorrelation_indicator_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let total = len
            .checked_mul(max_lag.saturating_add(1))
            .ok_or_else(|| JsValue::from_str("size overflow in autocorrelation_indicator_into"))?;
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        let (filtered_out, correlations_out) = out.split_at_mut(len);
        let input = AutocorrelationIndicatorInput::from_slice(
            data,
            AutocorrelationIndicatorParams {
                length: Some(length),
                max_lag: Some(max_lag),
                use_test_signal: Some(use_test_signal),
            },
        );
        autocorrelation_indicator_into_slice(filtered_out, correlations_out, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn autocorrelation_indicator_batch_into(
    data_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length_start: usize,
    length_end: usize,
    length_step: usize,
    max_lag: usize,
    use_test_signal: bool,
) -> Result<usize, JsValue> {
    if data_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to autocorrelation_indicator_batch_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let sweep = AutocorrelationIndicatorBatchRange {
            length: (length_start, length_end, length_step),
            max_lag: Some(max_lag),
            use_test_signal: Some(use_test_signal),
        };
        let combos = expand_grid_checked(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let filtered_total = rows
            .checked_mul(len)
            .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
        let corr_total = filtered_total
            .checked_mul(max_lag)
            .ok_or_else(|| JsValue::from_str("rows*lags*cols overflow"))?;
        let total = filtered_total
            .checked_add(corr_total)
            .ok_or_else(|| JsValue::from_str("total output overflow"))?;
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        let (filtered_out, correlations_out) = out.split_at_mut(filtered_total);
        autocorrelation_indicator_batch_inner_into(
            data,
            &sweep,
            Kernel::Auto,
            false,
            filtered_out,
            correlations_out,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn autocorrelation_indicator_output_into_js(
    data: &[f64],
    length: usize,
    max_lag: usize,
    use_test_signal: bool,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = autocorrelation_indicator_js(data, length, max_lag, use_test_signal)?;
    crate::write_wasm_object_f64_outputs("autocorrelation_indicator_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn autocorrelation_indicator_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = autocorrelation_indicator_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "autocorrelation_indicator_batch_output_into_js",
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
            .map(|i| {
                let x = i as f64;
                x.sin() + 0.3 * (x / 3.0).cos() + 0.01 * x
            })
            .collect()
    }

    fn naive_output(data: &[f64], length: usize, max_lag: usize) -> AutocorrelationIndicatorOutput {
        let mut filtered = vec![f64::NAN; data.len()];
        filter_series(data, length, false, &mut filtered);
        let mut correlations = vec![f64::NAN; data.len() * max_lag];
        for lag in 1..=max_lag {
            let row = &mut correlations[(lag - 1) * data.len()..lag * data.len()];
            for t in 0..data.len() {
                if t + 1 < length + lag {
                    continue;
                }
                let x_start = t + 1 - length;
                let y_start = x_start - lag;
                let mut sx = 0.0;
                let mut sy = 0.0;
                let mut sxx = 0.0;
                let mut syy = 0.0;
                let mut sxy = 0.0;
                for j in 0..length {
                    let x = filtered[x_start + j];
                    let y = filtered[y_start + j];
                    sx += x;
                    sy += y;
                    sxx += x * x;
                    syy += y * y;
                    sxy += x * y;
                }
                let lf = length as f64;
                let ca1 = lf * sxx - sx * sx;
                let ca2 = lf * syy - sy * sy;
                row[t] = if ca1 > 0.0 && ca2 > 0.0 {
                    let ca3 = lf * sxy - sx * sy;
                    ca3 / (ca1 * ca2).sqrt()
                } else {
                    0.0
                };
            }
        }
        AutocorrelationIndicatorOutput {
            filtered,
            correlations,
            lag_count: max_lag,
            cols: data.len(),
        }
    }

    fn assert_close(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (lhs, rhs) in a.iter().zip(b.iter()) {
            if lhs.is_nan() || rhs.is_nan() {
                assert!(lhs.is_nan() && rhs.is_nan());
            } else {
                assert!((lhs - rhs).abs() <= tol);
            }
        }
    }

    #[test]
    fn autocorrelation_indicator_matches_naive() -> Result<(), Box<dyn Error>> {
        let data = sample_data(160);
        let input = AutocorrelationIndicatorInput::from_slice(
            &data,
            AutocorrelationIndicatorParams {
                length: Some(20),
                max_lag: Some(12),
                use_test_signal: Some(false),
            },
        );
        let out = autocorrelation_indicator_with_kernel(&input, Kernel::Scalar)?;
        let naive = naive_output(&data, 20, 12);
        assert_close(&out.filtered, &naive.filtered, 1e-12);
        assert_close(&out.correlations, &naive.correlations, 1e-12);
        Ok(())
    }

    #[test]
    fn autocorrelation_indicator_into_matches_api() -> Result<(), Box<dyn Error>> {
        let data = sample_data(140);
        let input = AutocorrelationIndicatorInput::from_slice(
            &data,
            AutocorrelationIndicatorParams {
                length: Some(18),
                max_lag: Some(10),
                use_test_signal: Some(false),
            },
        );
        let base = autocorrelation_indicator(&input)?;
        let mut filtered = vec![f64::NAN; data.len()];
        let mut correlations = vec![f64::NAN; data.len() * 10];
        autocorrelation_indicator_into_slice(
            &mut filtered,
            &mut correlations,
            &input,
            Kernel::Auto,
        )?;
        assert_close(&filtered, &base.filtered, 1e-12);
        assert_close(&correlations, &base.correlations, 1e-12);
        Ok(())
    }

    #[test]
    fn autocorrelation_indicator_stream_matches_batch() -> Result<(), Box<dyn Error>> {
        let data = sample_data(150);
        let batch = autocorrelation_indicator(&AutocorrelationIndicatorInput::from_slice(
            &data,
            AutocorrelationIndicatorParams {
                length: Some(16),
                max_lag: Some(8),
                use_test_signal: Some(false),
            },
        ))?;
        let mut stream = AutocorrelationIndicatorStream::try_new(AutocorrelationIndicatorParams {
            length: Some(16),
            max_lag: Some(8),
            use_test_signal: Some(false),
        })?;
        let mut filtered = Vec::with_capacity(data.len());
        let mut correlations = vec![f64::NAN; data.len() * 8];
        for (i, &value) in data.iter().enumerate() {
            let point = stream.update(value).expect("finite stream output");
            filtered.push(point.filtered);
            for lag in 0..8 {
                correlations[lag * data.len() + i] = point.correlations[lag];
            }
        }
        assert_close(&filtered, &batch.filtered, 1e-12);
        assert_close(&correlations, &batch.correlations, 1e-12);
        Ok(())
    }

    #[test]
    fn autocorrelation_indicator_batch_single_matches_single() -> Result<(), Box<dyn Error>> {
        let data = sample_data(128);
        let single = autocorrelation_indicator(&AutocorrelationIndicatorInput::from_slice(
            &data,
            AutocorrelationIndicatorParams {
                length: Some(20),
                max_lag: Some(6),
                use_test_signal: Some(false),
            },
        ))?;
        let batch = autocorrelation_indicator_batch_with_kernel(
            &data,
            &AutocorrelationIndicatorBatchRange {
                length: (20, 20, 0),
                max_lag: Some(6),
                use_test_signal: Some(false),
            },
            Kernel::Scalar,
        )?;
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, data.len());
        assert_eq!(batch.lag_count, 6);
        assert_close(&batch.filtered, &single.filtered, 1e-12);
        assert_close(&batch.correlations, &single.correlations, 1e-12);
        Ok(())
    }

    #[test]
    fn autocorrelation_indicator_rejects_invalid_params() {
        let data = sample_data(64);
        let err = autocorrelation_indicator(&AutocorrelationIndicatorInput::from_slice(
            &data,
            AutocorrelationIndicatorParams {
                length: Some(0),
                max_lag: Some(8),
                use_test_signal: Some(false),
            },
        ))
        .unwrap_err();
        assert!(matches!(
            err,
            AutocorrelationIndicatorError::InvalidLength { .. }
        ));

        let err = autocorrelation_indicator_batch_with_kernel(
            &data,
            &AutocorrelationIndicatorBatchRange {
                length: (20, 10, 0),
                max_lag: Some(8),
                use_test_signal: Some(false),
            },
            Kernel::Auto,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            AutocorrelationIndicatorError::InvalidRange { .. }
        ));
    }

    #[test]
    fn autocorrelation_indicator_dispatch_compute_returns_selected_series(
    ) -> Result<(), Box<dyn Error>> {
        let data = sample_data(96);
        let params = [
            ParamKV {
                key: "length",
                value: ParamValue::Int(18),
            },
            ParamKV {
                key: "lag",
                value: ParamValue::Int(4),
            },
        ];
        let combos = [IndicatorParamSet { params: &params }];
        let req = IndicatorBatchRequest {
            indicator_id: "autocorrelation_indicator",
            output_id: Some("correlation"),
            combos: &combos,
            data: IndicatorDataRef::Slice { values: &data },
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req)?;
        let values = out.values_f64.unwrap();
        let direct = autocorrelation_indicator(&AutocorrelationIndicatorInput::from_slice(
            &data,
            AutocorrelationIndicatorParams {
                length: Some(18),
                max_lag: Some(4),
                use_test_signal: Some(false),
            },
        ))?;
        let row = &direct.correlations[3 * data.len()..4 * data.len()];
        assert_close(&values, row, 1e-12);
        Ok(())
    }
}
