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
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use thiserror::Error;

const DEFAULT_SOURCE: &str = "close";

impl<'a> AsRef<[f64]> for NonlinearRegressionZeroLagMovingAverageInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            NonlinearRegressionZeroLagMovingAverageData::Slice(slice) => slice,
            NonlinearRegressionZeroLagMovingAverageData::Candles { candles, source } => {
                source_type(candles, source)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum NonlinearRegressionZeroLagMovingAverageData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct NonlinearRegressionZeroLagMovingAverageOutput {
    pub value: Vec<f64>,
    pub signal: Vec<f64>,
    pub long_signal: Vec<f64>,
    pub short_signal: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct NonlinearRegressionZeroLagMovingAverageParams {
    pub zlma_period: Option<usize>,
    pub regression_period: Option<usize>,
}

impl Default for NonlinearRegressionZeroLagMovingAverageParams {
    fn default() -> Self {
        Self {
            zlma_period: Some(15),
            regression_period: Some(15),
        }
    }
}

#[derive(Debug, Clone)]
pub struct NonlinearRegressionZeroLagMovingAverageInput<'a> {
    pub data: NonlinearRegressionZeroLagMovingAverageData<'a>,
    pub params: NonlinearRegressionZeroLagMovingAverageParams,
}

impl<'a> NonlinearRegressionZeroLagMovingAverageInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: NonlinearRegressionZeroLagMovingAverageParams,
    ) -> Self {
        Self {
            data: NonlinearRegressionZeroLagMovingAverageData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(
        data: &'a [f64],
        params: NonlinearRegressionZeroLagMovingAverageParams,
    ) -> Self {
        Self {
            data: NonlinearRegressionZeroLagMovingAverageData::Slice(data),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(
            candles,
            DEFAULT_SOURCE,
            NonlinearRegressionZeroLagMovingAverageParams::default(),
        )
    }

    #[inline]
    pub fn get_zlma_period(&self) -> usize {
        self.params.zlma_period.unwrap_or(15)
    }

    #[inline]
    pub fn get_regression_period(&self) -> usize {
        self.params.regression_period.unwrap_or(15)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct NonlinearRegressionZeroLagMovingAverageBuilder {
    zlma_period: Option<usize>,
    regression_period: Option<usize>,
    kernel: Kernel,
}

impl Default for NonlinearRegressionZeroLagMovingAverageBuilder {
    fn default() -> Self {
        Self {
            zlma_period: None,
            regression_period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl NonlinearRegressionZeroLagMovingAverageBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn zlma_period(mut self, value: usize) -> Self {
        self.zlma_period = Some(value);
        self
    }

    #[inline(always)]
    pub fn regression_period(mut self, value: usize) -> Self {
        self.regression_period = Some(value);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline(always)]
    fn build_params(self) -> NonlinearRegressionZeroLagMovingAverageParams {
        NonlinearRegressionZeroLagMovingAverageParams {
            zlma_period: self.zlma_period,
            regression_period: self.regression_period,
        }
    }

    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<
        NonlinearRegressionZeroLagMovingAverageOutput,
        NonlinearRegressionZeroLagMovingAverageError,
    > {
        nonlinear_regression_zero_lag_moving_average_with_kernel(
            &NonlinearRegressionZeroLagMovingAverageInput::from_candles(
                candles,
                DEFAULT_SOURCE,
                self.build_params(),
            ),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<
        NonlinearRegressionZeroLagMovingAverageOutput,
        NonlinearRegressionZeroLagMovingAverageError,
    > {
        nonlinear_regression_zero_lag_moving_average_with_kernel(
            &NonlinearRegressionZeroLagMovingAverageInput::from_candles(
                candles,
                source,
                self.build_params(),
            ),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<
        NonlinearRegressionZeroLagMovingAverageOutput,
        NonlinearRegressionZeroLagMovingAverageError,
    > {
        nonlinear_regression_zero_lag_moving_average_with_kernel(
            &NonlinearRegressionZeroLagMovingAverageInput::from_slice(data, self.build_params()),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<
        NonlinearRegressionZeroLagMovingAverageStream,
        NonlinearRegressionZeroLagMovingAverageError,
    > {
        NonlinearRegressionZeroLagMovingAverageStream::try_new(self.build_params())
    }
}

#[derive(Debug, Error)]
pub enum NonlinearRegressionZeroLagMovingAverageError {
    #[error("nonlinear_regression_zero_lag_moving_average: Input data slice is empty.")]
    EmptyInputData,
    #[error("nonlinear_regression_zero_lag_moving_average: All values are NaN.")]
    AllValuesNaN,
    #[error("nonlinear_regression_zero_lag_moving_average: Invalid zlma_period: {zlma_period}")]
    InvalidZlmaPeriod { zlma_period: usize },
    #[error(
        "nonlinear_regression_zero_lag_moving_average: Invalid regression_period: {regression_period}"
    )]
    InvalidRegressionPeriod { regression_period: usize },
    #[error(
        "nonlinear_regression_zero_lag_moving_average: Not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "nonlinear_regression_zero_lag_moving_average: Output length mismatch: expected = {expected}, got = {got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error(
        "nonlinear_regression_zero_lag_moving_average: Invalid range: start={start}, end={end}, step={step}"
    )]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("nonlinear_regression_zero_lag_moving_average: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error(
        "nonlinear_regression_zero_lag_moving_average: Output length mismatch: dst = {dst_len}, expected = {expected_len}"
    )]
    MismatchedOutputLen { dst_len: usize, expected_len: usize },
    #[error("nonlinear_regression_zero_lag_moving_average: Invalid input: {msg}")]
    InvalidInput { msg: String },
}

#[derive(Clone, Copy, Debug)]
struct ResolvedInput<'a> {
    data: &'a [f64],
    zlma_period: usize,
    regression_period: usize,
    warmup: usize,
}

#[inline(always)]
fn valid_run_until(data: &[f64], needed: usize) -> usize {
    let mut best = 0usize;
    let mut current = 0usize;
    for &value in data {
        if value.is_nan() {
            current = 0;
        } else {
            current += 1;
            if current > best {
                best = current;
                if best >= needed {
                    return best;
                }
            }
        }
    }
    best
}

#[inline(always)]
fn required_valid_count(
    zlma_period: usize,
    regression_period: usize,
) -> Result<usize, NonlinearRegressionZeroLagMovingAverageError> {
    zlma_period
        .checked_mul(2)
        .and_then(|value| value.checked_add(regression_period))
        .and_then(|value| value.checked_sub(2))
        .ok_or_else(
            || NonlinearRegressionZeroLagMovingAverageError::InvalidInput {
                msg: "required valid count overflow".to_string(),
            },
        )
}

#[inline(always)]
fn resolve_input<'a>(
    input: &'a NonlinearRegressionZeroLagMovingAverageInput,
) -> Result<ResolvedInput<'a>, NonlinearRegressionZeroLagMovingAverageError> {
    let data = input.as_ref();
    if data.is_empty() {
        return Err(NonlinearRegressionZeroLagMovingAverageError::EmptyInputData);
    }
    if !data.iter().any(|value| !value.is_nan()) {
        return Err(NonlinearRegressionZeroLagMovingAverageError::AllValuesNaN);
    }

    let zlma_period = input.get_zlma_period();
    if zlma_period == 0 {
        return Err(
            NonlinearRegressionZeroLagMovingAverageError::InvalidZlmaPeriod { zlma_period },
        );
    }

    let regression_period = input.get_regression_period();
    if regression_period == 0 {
        return Err(
            NonlinearRegressionZeroLagMovingAverageError::InvalidRegressionPeriod {
                regression_period,
            },
        );
    }

    let needed = required_valid_count(zlma_period, regression_period)?;
    let valid = valid_run_until(data, needed);
    if valid < needed {
        return Err(
            NonlinearRegressionZeroLagMovingAverageError::NotEnoughValidData { needed, valid },
        );
    }

    Ok(ResolvedInput {
        data,
        zlma_period,
        regression_period,
        warmup: needed.saturating_sub(1),
    })
}

#[derive(Clone, Copy, Debug)]
struct RegressionConstants {
    period: usize,
    period_f64: f64,
    avg_x: f64,
    avg_x2: f64,
    sx2: f64,
    sxx: f64,
    sxx2: f64,
    sx2x2: f64,
    denom: f64,
}

impl RegressionConstants {
    #[inline]
    fn new(period: usize) -> Self {
        let period_f64 = period as f64;
        let mut sx = 0.0;
        let mut sx2 = 0.0;
        let mut sx3 = 0.0;
        let mut sx4 = 0.0;
        for x in 0..period {
            let xf = x as f64;
            let x2 = xf * xf;
            sx += xf;
            sx2 += x2;
            sx3 += x2 * xf;
            sx4 += x2 * x2;
        }
        let avg_x = sx / period_f64;
        let avg_x2 = avg_x * avg_x;
        let sxx = sx2 - period_f64 * avg_x2;
        let sxx2 = sx3 - avg_x * sx2 - avg_x2 * sx + period_f64 * avg_x2 * avg_x;
        let sx2x2 = sx4 - 2.0 * avg_x2 * sx2 + period_f64 * avg_x2 * avg_x2;
        let denom = sxx * sx2x2 - sxx2 * sxx2;
        Self {
            period,
            period_f64,
            avg_x,
            avg_x2,
            sx2,
            sxx,
            sxx2,
            sx2x2,
            denom,
        }
    }
}

#[derive(Clone, Debug)]
struct RollingWma {
    period: usize,
    denominator: f64,
    ring: Vec<f64>,
    head: usize,
    count: usize,
    sum: f64,
    weighted_sum: f64,
}

impl RollingWma {
    #[inline]
    fn new(period: usize) -> Self {
        Self {
            period,
            denominator: (period * (period + 1) / 2) as f64,
            ring: vec![0.0; period.max(1)],
            head: 0,
            count: 0,
            sum: 0.0,
            weighted_sum: 0.0,
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.head = 0;
        self.count = 0;
        self.sum = 0.0;
        self.weighted_sum = 0.0;
    }

    #[inline]
    fn update(&mut self, value: f64) -> Option<f64> {
        if value.is_nan() {
            self.reset();
            return None;
        }

        if self.count < self.period {
            let index = (self.head + self.count) % self.ring.len();
            self.ring[index] = value;
            self.count += 1;
            self.sum += value;
            self.weighted_sum += self.count as f64 * value;
            if self.count == self.period {
                return Some(self.weighted_sum / self.denominator);
            }
            return None;
        }

        let oldest = self.ring[self.head];
        let old_sum = self.sum;
        self.weighted_sum = self.weighted_sum - old_sum + self.period as f64 * value;
        self.sum = old_sum - oldest + value;
        self.ring[self.head] = value;
        self.head += 1;
        if self.head == self.ring.len() {
            self.head = 0;
        }

        Some(self.weighted_sum / self.denominator)
    }
}

#[derive(Clone, Debug)]
struct QuadraticRegressionStream {
    constants: RegressionConstants,
    ring: Vec<f64>,
    head: usize,
    count: usize,
    sy: f64,
    sxy: f64,
    sx2y: f64,
}

impl QuadraticRegressionStream {
    #[inline]
    fn new(period: usize) -> Self {
        Self {
            constants: RegressionConstants::new(period),
            ring: vec![0.0; period.max(1)],
            head: 0,
            count: 0,
            sy: 0.0,
            sxy: 0.0,
            sx2y: 0.0,
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.head = 0;
        self.count = 0;
        self.sy = 0.0;
        self.sxy = 0.0;
        self.sx2y = 0.0;
    }

    #[inline]
    fn recompute_sums(&mut self) {
        self.sy = 0.0;
        self.sxy = 0.0;
        self.sx2y = 0.0;
        for k in 0..self.constants.period {
            let idx = (self.head + self.constants.period - 1 - k) % self.constants.period;
            let value = self.ring[idx];
            let kf = k as f64;
            self.sy += value;
            self.sxy += kf * value;
            self.sx2y += kf * kf * value;
        }
    }

    #[inline]
    fn output_value(&self) -> f64 {
        let avg_y = self.sy / self.constants.period_f64;
        if self.constants.denom.abs() <= f64::EPSILON {
            return avg_y;
        }
        let sxy = self.sxy - self.constants.avg_x * self.sy;
        let syx2 = self.sx2y - avg_y * self.constants.sx2;
        let b = (sxy * self.constants.sx2x2 - syx2 * self.constants.sxx2) / self.constants.denom;
        let c = (self.constants.sxx * syx2 - self.constants.sxx2 * sxy) / self.constants.denom;
        let a = avg_y - b * self.constants.avg_x - c * self.constants.avg_x2;
        a + c
    }

    #[inline]
    fn update(&mut self, value: f64) -> Option<f64> {
        if value.is_nan() {
            self.reset();
            return None;
        }

        if self.count < self.constants.period {
            let index = (self.head + self.count) % self.ring.len();
            self.ring[index] = value;
            self.count += 1;
            if self.count == self.constants.period {
                self.recompute_sums();
                return Some(self.output_value());
            }
            return None;
        }

        let oldest = self.ring[self.head];
        let old_sy = self.sy;
        let old_sxy = self.sxy;
        let carry = old_sy - oldest;

        self.ring[self.head] = value;
        self.head += 1;
        if self.head == self.ring.len() {
            self.head = 0;
        }

        self.sy = old_sy - oldest + value;
        self.sxy = old_sxy + carry;
        self.sx2y = self.sx2y + 2.0 * old_sxy + carry;

        Some(self.output_value())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct NonlinearRegressionZeroLagMovingAveragePoint {
    pub value: f64,
    pub signal: f64,
    pub long_signal: f64,
    pub short_signal: f64,
}

#[derive(Clone, Debug)]
pub struct NonlinearRegressionZeroLagMovingAverageStream {
    zlma_period: usize,
    regression_period: usize,
    warmup_period: usize,
    first_wma: RollingWma,
    second_wma: RollingWma,
    regression: QuadraticRegressionStream,
    prev_value: Option<f64>,
    prev_signal: Option<f64>,
}

impl NonlinearRegressionZeroLagMovingAverageStream {
    #[inline]
    pub fn try_new(
        params: NonlinearRegressionZeroLagMovingAverageParams,
    ) -> Result<Self, NonlinearRegressionZeroLagMovingAverageError> {
        let zlma_period = params.zlma_period.unwrap_or(15);
        if zlma_period == 0 {
            return Err(
                NonlinearRegressionZeroLagMovingAverageError::InvalidZlmaPeriod { zlma_period },
            );
        }
        let regression_period = params.regression_period.unwrap_or(15);
        if regression_period == 0 {
            return Err(
                NonlinearRegressionZeroLagMovingAverageError::InvalidRegressionPeriod {
                    regression_period,
                },
            );
        }
        let warmup = required_valid_count(zlma_period, regression_period)?.saturating_sub(1);
        Ok(Self {
            zlma_period,
            regression_period,
            warmup_period: warmup,
            first_wma: RollingWma::new(zlma_period),
            second_wma: RollingWma::new(zlma_period),
            regression: QuadraticRegressionStream::new(regression_period),
            prev_value: None,
            prev_signal: None,
        })
    }

    #[inline]
    pub fn update(&mut self, value: f64) -> Option<NonlinearRegressionZeroLagMovingAveragePoint> {
        if value.is_nan() {
            self.reset();
            return None;
        }

        let first = self.first_wma.update(value)?;
        let second = self.second_wma.update(first)?;
        let zl_value = 2.0 * first - second;
        let reg_value = self.regression.update(zl_value)?;
        let signal = self.prev_value.unwrap_or(f64::NAN);
        let long_signal =
            if let (Some(prev_value), Some(prev_signal)) = (self.prev_value, self.prev_signal) {
                if reg_value > signal && prev_value <= prev_signal {
                    1.0
                } else {
                    0.0
                }
            } else {
                0.0
            };
        let short_signal =
            if let (Some(prev_value), Some(prev_signal)) = (self.prev_value, self.prev_signal) {
                if reg_value < signal && prev_value >= prev_signal {
                    1.0
                } else {
                    0.0
                }
            } else {
                0.0
            };

        self.prev_signal = Some(signal);
        self.prev_value = Some(reg_value);

        Some(NonlinearRegressionZeroLagMovingAveragePoint {
            value: reg_value,
            signal,
            long_signal,
            short_signal,
        })
    }

    #[inline]
    pub fn reset(&mut self) {
        self.first_wma.reset();
        self.second_wma.reset();
        self.regression.reset();
        self.prev_value = None;
        self.prev_signal = None;
    }

    #[inline]
    pub fn get_warmup_period(&self) -> usize {
        self.warmup_period
    }
}

#[inline(always)]
fn compute_row(
    data: &[f64],
    zlma_period: usize,
    regression_period: usize,
    value: &mut [f64],
    signal: &mut [f64],
    long_signal: &mut [f64],
    short_signal: &mut [f64],
) {
    let mut stream = NonlinearRegressionZeroLagMovingAverageStream {
        zlma_period,
        regression_period,
        warmup_period: required_valid_count(zlma_period, regression_period)
            .unwrap_or(1)
            .saturating_sub(1),
        first_wma: RollingWma::new(zlma_period),
        second_wma: RollingWma::new(zlma_period),
        regression: QuadraticRegressionStream::new(regression_period),
        prev_value: None,
        prev_signal: None,
    };

    for i in 0..data.len() {
        value[i] = f64::NAN;
        signal[i] = f64::NAN;
        long_signal[i] = 0.0;
        short_signal[i] = 0.0;
        if let Some(point) = stream.update(data[i]) {
            value[i] = point.value;
            signal[i] = point.signal;
            long_signal[i] = point.long_signal;
            short_signal[i] = point.short_signal;
        }
    }
}

#[inline]
pub fn nonlinear_regression_zero_lag_moving_average(
    input: &NonlinearRegressionZeroLagMovingAverageInput,
) -> Result<
    NonlinearRegressionZeroLagMovingAverageOutput,
    NonlinearRegressionZeroLagMovingAverageError,
> {
    nonlinear_regression_zero_lag_moving_average_with_kernel(input, Kernel::Auto)
}

pub fn nonlinear_regression_zero_lag_moving_average_with_kernel(
    input: &NonlinearRegressionZeroLagMovingAverageInput,
    kernel: Kernel,
) -> Result<
    NonlinearRegressionZeroLagMovingAverageOutput,
    NonlinearRegressionZeroLagMovingAverageError,
> {
    let resolved = resolve_input(input)?;
    let _chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        Kernel::Scalar
        | Kernel::ScalarBatch
        | Kernel::Avx2
        | Kernel::Avx2Batch
        | Kernel::Avx512
        | Kernel::Avx512Batch => kernel,
        other => {
            return Err(NonlinearRegressionZeroLagMovingAverageError::InvalidInput {
                msg: format!("unsupported kernel: {other:?}"),
            })
        }
    };

    let len = resolved.data.len();
    let mut value = alloc_with_nan_prefix(len, resolved.warmup.min(len));
    let mut signal = alloc_with_nan_prefix(len, resolved.warmup.min(len));
    let mut long_signal = vec![0.0; len];
    let mut short_signal = vec![0.0; len];

    compute_row(
        resolved.data,
        resolved.zlma_period,
        resolved.regression_period,
        &mut value,
        &mut signal,
        &mut long_signal,
        &mut short_signal,
    );

    Ok(NonlinearRegressionZeroLagMovingAverageOutput {
        value,
        signal,
        long_signal,
        short_signal,
    })
}

#[inline]
pub fn nonlinear_regression_zero_lag_moving_average_into_slice(
    dst_value: &mut [f64],
    dst_signal: &mut [f64],
    dst_long_signal: &mut [f64],
    dst_short_signal: &mut [f64],
    input: &NonlinearRegressionZeroLagMovingAverageInput,
    kernel: Kernel,
) -> Result<(), NonlinearRegressionZeroLagMovingAverageError> {
    let resolved = resolve_input(input)?;
    if dst_value.len() != resolved.data.len() {
        return Err(
            NonlinearRegressionZeroLagMovingAverageError::OutputLengthMismatch {
                expected: resolved.data.len(),
                got: dst_value.len(),
            },
        );
    }
    if dst_signal.len() != resolved.data.len() {
        return Err(
            NonlinearRegressionZeroLagMovingAverageError::OutputLengthMismatch {
                expected: resolved.data.len(),
                got: dst_signal.len(),
            },
        );
    }
    if dst_long_signal.len() != resolved.data.len() {
        return Err(
            NonlinearRegressionZeroLagMovingAverageError::OutputLengthMismatch {
                expected: resolved.data.len(),
                got: dst_long_signal.len(),
            },
        );
    }
    if dst_short_signal.len() != resolved.data.len() {
        return Err(
            NonlinearRegressionZeroLagMovingAverageError::OutputLengthMismatch {
                expected: resolved.data.len(),
                got: dst_short_signal.len(),
            },
        );
    }

    let _chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        Kernel::Scalar
        | Kernel::ScalarBatch
        | Kernel::Avx2
        | Kernel::Avx2Batch
        | Kernel::Avx512
        | Kernel::Avx512Batch => kernel,
        other => {
            return Err(NonlinearRegressionZeroLagMovingAverageError::InvalidInput {
                msg: format!("unsupported kernel: {other:?}"),
            })
        }
    };

    compute_row(
        resolved.data,
        resolved.zlma_period,
        resolved.regression_period,
        dst_value,
        dst_signal,
        dst_long_signal,
        dst_short_signal,
    );
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn nonlinear_regression_zero_lag_moving_average_into(
    input: &NonlinearRegressionZeroLagMovingAverageInput,
    dst_value: &mut [f64],
    dst_signal: &mut [f64],
    dst_long_signal: &mut [f64],
    dst_short_signal: &mut [f64],
) -> Result<(), NonlinearRegressionZeroLagMovingAverageError> {
    nonlinear_regression_zero_lag_moving_average_into_slice(
        dst_value,
        dst_signal,
        dst_long_signal,
        dst_short_signal,
        input,
        Kernel::Auto,
    )
}

#[derive(Debug, Clone, Copy)]
pub struct NonlinearRegressionZeroLagMovingAverageBatchRange {
    pub zlma_period: (usize, usize, usize),
    pub regression_period: (usize, usize, usize),
}

impl Default for NonlinearRegressionZeroLagMovingAverageBatchRange {
    fn default() -> Self {
        Self {
            zlma_period: (15, 15, 0),
            regression_period: (15, 15, 0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct NonlinearRegressionZeroLagMovingAverageBatchOutput {
    pub value: Vec<f64>,
    pub signal: Vec<f64>,
    pub long_signal: Vec<f64>,
    pub short_signal: Vec<f64>,
    pub combos: Vec<NonlinearRegressionZeroLagMovingAverageParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct NonlinearRegressionZeroLagMovingAverageBatchBuilder {
    range: NonlinearRegressionZeroLagMovingAverageBatchRange,
    kernel: Kernel,
}

impl Default for NonlinearRegressionZeroLagMovingAverageBatchBuilder {
    fn default() -> Self {
        Self {
            range: NonlinearRegressionZeroLagMovingAverageBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl NonlinearRegressionZeroLagMovingAverageBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn zlma_period_range(mut self, value: (usize, usize, usize)) -> Self {
        self.range.zlma_period = value;
        self
    }

    #[inline(always)]
    pub fn regression_period_range(mut self, value: (usize, usize, usize)) -> Self {
        self.range.regression_period = value;
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
    ) -> Result<
        NonlinearRegressionZeroLagMovingAverageBatchOutput,
        NonlinearRegressionZeroLagMovingAverageError,
    > {
        nonlinear_regression_zero_lag_moving_average_batch_with_kernel(
            data,
            &self.range,
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<
        NonlinearRegressionZeroLagMovingAverageBatchOutput,
        NonlinearRegressionZeroLagMovingAverageError,
    > {
        nonlinear_regression_zero_lag_moving_average_batch_with_kernel(
            source_type(candles, source),
            &self.range,
            self.kernel,
        )
    }
}

#[inline(always)]
fn expand_axis(
    range: (usize, usize, usize),
) -> Result<Vec<usize>, NonlinearRegressionZeroLagMovingAverageError> {
    let (start, end, step) = range;
    if start == 0 {
        return Err(NonlinearRegressionZeroLagMovingAverageError::InvalidRange {
            start,
            end,
            step,
        });
    }
    if step == 0 {
        return Ok(vec![start]);
    }
    if start > end {
        return Err(NonlinearRegressionZeroLagMovingAverageError::InvalidRange {
            start,
            end,
            step,
        });
    }
    let mut out = Vec::new();
    let mut current = start;
    loop {
        out.push(current);
        if current >= end {
            break;
        }
        let next = current.checked_add(step).ok_or_else(|| {
            NonlinearRegressionZeroLagMovingAverageError::InvalidInput {
                msg: "range step overflow".to_string(),
            }
        })?;
        if next <= current {
            return Err(NonlinearRegressionZeroLagMovingAverageError::InvalidRange {
                start,
                end,
                step,
            });
        }
        current = next.min(end);
    }
    Ok(out)
}

fn expand_grid_checked(
    sweep: &NonlinearRegressionZeroLagMovingAverageBatchRange,
) -> Result<
    Vec<NonlinearRegressionZeroLagMovingAverageParams>,
    NonlinearRegressionZeroLagMovingAverageError,
> {
    let zlma_periods = expand_axis(sweep.zlma_period)?;
    let regression_periods = expand_axis(sweep.regression_period)?;
    let total = zlma_periods
        .len()
        .checked_mul(regression_periods.len())
        .ok_or_else(
            || NonlinearRegressionZeroLagMovingAverageError::InvalidInput {
                msg: "parameter grid size overflow".to_string(),
            },
        )?;
    let mut combos = Vec::with_capacity(total);
    for &zlma_period in &zlma_periods {
        for &regression_period in &regression_periods {
            combos.push(NonlinearRegressionZeroLagMovingAverageParams {
                zlma_period: Some(zlma_period),
                regression_period: Some(regression_period),
            });
        }
    }
    Ok(combos)
}

pub fn expand_grid_nonlinear_regression_zero_lag_moving_average(
    sweep: &NonlinearRegressionZeroLagMovingAverageBatchRange,
) -> Result<
    Vec<NonlinearRegressionZeroLagMovingAverageParams>,
    NonlinearRegressionZeroLagMovingAverageError,
> {
    expand_grid_checked(sweep)
}

#[inline(always)]
fn alloc_nan_matrix(rows: usize, cols: usize, warmups: &[usize]) -> Vec<f64> {
    let mut matrix = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut matrix, cols, warmups);
    let out = unsafe {
        Vec::from_raw_parts(
            matrix.as_mut_ptr() as *mut f64,
            matrix.len(),
            matrix.capacity(),
        )
    };
    std::mem::forget(matrix);
    out
}

pub fn nonlinear_regression_zero_lag_moving_average_batch_with_kernel(
    data: &[f64],
    sweep: &NonlinearRegressionZeroLagMovingAverageBatchRange,
    kernel: Kernel,
) -> Result<
    NonlinearRegressionZeroLagMovingAverageBatchOutput,
    NonlinearRegressionZeroLagMovingAverageError,
> {
    match kernel {
        Kernel::Auto
        | Kernel::Scalar
        | Kernel::ScalarBatch
        | Kernel::Avx2
        | Kernel::Avx2Batch
        | Kernel::Avx512
        | Kernel::Avx512Batch => {}
        other => {
            return Err(NonlinearRegressionZeroLagMovingAverageError::InvalidKernelForBatch(other))
        }
    }

    let combos = expand_grid_checked(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    let total = rows.checked_mul(cols).ok_or_else(|| {
        NonlinearRegressionZeroLagMovingAverageError::InvalidInput {
            msg: "rows*cols overflow in batch".to_string(),
        }
    })?;

    let warmups = combos
        .iter()
        .map(|params| {
            required_valid_count(
                params.zlma_period.unwrap_or(15),
                params.regression_period.unwrap_or(15),
            )
            .unwrap_or(cols)
            .saturating_sub(1)
            .min(cols)
        })
        .collect::<Vec<_>>();

    let mut value = alloc_nan_matrix(rows, cols, &warmups);
    let mut signal = alloc_nan_matrix(rows, cols, &warmups);
    let mut long_signal = vec![0.0; total];
    let mut short_signal = vec![0.0; total];

    let _chosen = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other => other,
    };

    let worker = |row: usize,
                  dst_value: &mut [f64],
                  dst_signal: &mut [f64],
                  dst_long: &mut [f64],
                  dst_short: &mut [f64]| {
        let params = &combos[row];
        compute_row(
            data,
            params.zlma_period.unwrap_or(15),
            params.regression_period.unwrap_or(15),
            dst_value,
            dst_signal,
            dst_long,
            dst_short,
        );
    };

    #[cfg(not(target_arch = "wasm32"))]
    {
        value
            .par_chunks_mut(cols)
            .zip(signal.par_chunks_mut(cols))
            .zip(long_signal.par_chunks_mut(cols))
            .zip(short_signal.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, (((dst_value, dst_signal), dst_long), dst_short))| {
                worker(row, dst_value, dst_signal, dst_long, dst_short);
            });
    }

    #[cfg(target_arch = "wasm32")]
    {
        for (row, (((dst_value, dst_signal), dst_long), dst_short)) in value
            .chunks_mut(cols)
            .zip(signal.chunks_mut(cols))
            .zip(long_signal.chunks_mut(cols))
            .zip(short_signal.chunks_mut(cols))
            .enumerate()
        {
            worker(row, dst_value, dst_signal, dst_long, dst_short);
        }
    }

    Ok(NonlinearRegressionZeroLagMovingAverageBatchOutput {
        value,
        signal,
        long_signal,
        short_signal,
        combos,
        rows,
        cols,
    })
}

pub fn nonlinear_regression_zero_lag_moving_average_batch_slice(
    data: &[f64],
    sweep: &NonlinearRegressionZeroLagMovingAverageBatchRange,
    kernel: Kernel,
) -> Result<
    NonlinearRegressionZeroLagMovingAverageBatchOutput,
    NonlinearRegressionZeroLagMovingAverageError,
> {
    nonlinear_regression_zero_lag_moving_average_batch_with_kernel(data, sweep, kernel)
}

pub fn nonlinear_regression_zero_lag_moving_average_batch_par_slice(
    data: &[f64],
    sweep: &NonlinearRegressionZeroLagMovingAverageBatchRange,
    kernel: Kernel,
) -> Result<
    NonlinearRegressionZeroLagMovingAverageBatchOutput,
    NonlinearRegressionZeroLagMovingAverageError,
> {
    nonlinear_regression_zero_lag_moving_average_batch_with_kernel(data, sweep, kernel)
}

#[cfg(feature = "python")]
#[pyfunction(name = "nonlinear_regression_zero_lag_moving_average")]
#[pyo3(signature = (data, zlma_period=15, regression_period=15, kernel=None))]
pub fn nonlinear_regression_zero_lag_moving_average_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    zlma_period: usize,
    regression_period: usize,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let data = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let input = NonlinearRegressionZeroLagMovingAverageInput::from_slice(
        data,
        NonlinearRegressionZeroLagMovingAverageParams {
            zlma_period: Some(zlma_period),
            regression_period: Some(regression_period),
        },
    );
    let out = py
        .allow_threads(|| nonlinear_regression_zero_lag_moving_average_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        out.value.into_pyarray(py),
        out.signal.into_pyarray(py),
        out.long_signal.into_pyarray(py),
        out.short_signal.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyclass(name = "NonlinearRegressionZeroLagMovingAverageStream")]
pub struct NonlinearRegressionZeroLagMovingAverageStreamPy {
    stream: NonlinearRegressionZeroLagMovingAverageStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl NonlinearRegressionZeroLagMovingAverageStreamPy {
    #[new]
    #[pyo3(signature = (zlma_period=15, regression_period=15))]
    fn new(zlma_period: usize, regression_period: usize) -> PyResult<Self> {
        let stream = NonlinearRegressionZeroLagMovingAverageStream::try_new(
            NonlinearRegressionZeroLagMovingAverageParams {
                zlma_period: Some(zlma_period),
                regression_period: Some(regression_period),
            },
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, value: f64) -> Option<(f64, f64, f64, f64)> {
        self.stream.update(value).map(|point| {
            (
                point.value,
                point.signal,
                point.long_signal,
                point.short_signal,
            )
        })
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
#[pyfunction(name = "nonlinear_regression_zero_lag_moving_average_batch")]
#[pyo3(signature = (data, zlma_period_range=(15, 15, 0), regression_period_range=(15, 15, 0), kernel=None))]
pub fn nonlinear_regression_zero_lag_moving_average_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    zlma_period_range: (usize, usize, usize),
    regression_period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let out = py
        .allow_threads(|| {
            nonlinear_regression_zero_lag_moving_average_batch_with_kernel(
                data,
                &NonlinearRegressionZeroLagMovingAverageBatchRange {
                    zlma_period: zlma_period_range,
                    regression_period: regression_period_range,
                },
                kern,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item(
        "value",
        out.value.into_pyarray(py).reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "signal",
        out.signal.into_pyarray(py).reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "long_signal",
        out.long_signal
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "short_signal",
        out.short_signal
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "zlma_periods",
        out.combos
            .iter()
            .map(|combo| combo.zlma_period.unwrap_or(15) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "regression_periods",
        out.combos
            .iter()
            .map(|combo| combo.regression_period.unwrap_or(15) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", out.rows)?;
    dict.set_item("cols", out.cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_nonlinear_regression_zero_lag_moving_average_module(
    m: &Bound<'_, PyModule>,
) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        nonlinear_regression_zero_lag_moving_average_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        nonlinear_regression_zero_lag_moving_average_batch_py,
        m
    )?)?;
    m.add_class::<NonlinearRegressionZeroLagMovingAverageStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NonlinearRegressionZeroLagMovingAverageBatchConfig {
    pub zlma_period_range: Vec<usize>,
    pub regression_period_range: Vec<usize>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = nonlinear_regression_zero_lag_moving_average_js)]
pub fn nonlinear_regression_zero_lag_moving_average_js(
    data: &[f64],
    zlma_period: usize,
    regression_period: usize,
) -> Result<JsValue, JsValue> {
    let input = NonlinearRegressionZeroLagMovingAverageInput::from_slice(
        data,
        NonlinearRegressionZeroLagMovingAverageParams {
            zlma_period: Some(zlma_period),
            regression_period: Some(regression_period),
        },
    );
    let out = nonlinear_regression_zero_lag_moving_average_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("value"),
        &serde_wasm_bindgen::to_value(&out.value).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("signal"),
        &serde_wasm_bindgen::to_value(&out.signal).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("long_signal"),
        &serde_wasm_bindgen::to_value(&out.long_signal).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("short_signal"),
        &serde_wasm_bindgen::to_value(&out.short_signal).unwrap(),
    )?;
    Ok(obj.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = nonlinear_regression_zero_lag_moving_average_batch_js)]
pub fn nonlinear_regression_zero_lag_moving_average_batch_js(
    data: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: NonlinearRegressionZeroLagMovingAverageBatchConfig =
        serde_wasm_bindgen::from_value(config)
            .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.zlma_period_range.len() != 3 || config.regression_period_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: every range must have exactly 3 elements [start, end, step]",
        ));
    }
    let out = nonlinear_regression_zero_lag_moving_average_batch_with_kernel(
        data,
        &NonlinearRegressionZeroLagMovingAverageBatchRange {
            zlma_period: (
                config.zlma_period_range[0],
                config.zlma_period_range[1],
                config.zlma_period_range[2],
            ),
            regression_period: (
                config.regression_period_range[0],
                config.regression_period_range[1],
                config.regression_period_range[2],
            ),
        },
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("value"),
        &serde_wasm_bindgen::to_value(&out.value).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("signal"),
        &serde_wasm_bindgen::to_value(&out.signal).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("long_signal"),
        &serde_wasm_bindgen::to_value(&out.long_signal).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("short_signal"),
        &serde_wasm_bindgen::to_value(&out.short_signal).unwrap(),
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
pub fn nonlinear_regression_zero_lag_moving_average_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(4 * len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn nonlinear_regression_zero_lag_moving_average_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, 4 * len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn nonlinear_regression_zero_lag_moving_average_into(
    data_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    zlma_period: usize,
    regression_period: usize,
) -> Result<(), JsValue> {
    if data_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to nonlinear_regression_zero_lag_moving_average_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, 4 * len);
        let (dst_value, rest) = out.split_at_mut(len);
        let (dst_signal, rest) = rest.split_at_mut(len);
        let (dst_long_signal, dst_short_signal) = rest.split_at_mut(len);

        let input = NonlinearRegressionZeroLagMovingAverageInput::from_slice(
            data,
            NonlinearRegressionZeroLagMovingAverageParams {
                zlma_period: Some(zlma_period),
                regression_period: Some(regression_period),
            },
        );
        nonlinear_regression_zero_lag_moving_average_into_slice(
            dst_value,
            dst_signal,
            dst_long_signal,
            dst_short_signal,
            &input,
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn nonlinear_regression_zero_lag_moving_average_batch_into(
    data_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    zlma_period_start: usize,
    zlma_period_end: usize,
    zlma_period_step: usize,
    regression_period_start: usize,
    regression_period_end: usize,
    regression_period_step: usize,
) -> Result<usize, JsValue> {
    if data_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to nonlinear_regression_zero_lag_moving_average_batch_into",
        ));
    }

    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let sweep = NonlinearRegressionZeroLagMovingAverageBatchRange {
            zlma_period: (zlma_period_start, zlma_period_end, zlma_period_step),
            regression_period: (
                regression_period_start,
                regression_period_end,
                regression_period_step,
            ),
        };
        let combos = expand_grid_checked(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let total = rows
            .checked_mul(len)
            .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
        let out = std::slice::from_raw_parts_mut(out_ptr, 4 * total);
        let (dst_value, rest) = out.split_at_mut(total);
        let (dst_signal, rest) = rest.split_at_mut(total);
        let (dst_long_signal, dst_short_signal) = rest.split_at_mut(total);

        let batch = nonlinear_regression_zero_lag_moving_average_batch_with_kernel(
            data,
            &sweep,
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

        dst_value.copy_from_slice(&batch.value);
        dst_signal.copy_from_slice(&batch.signal);
        dst_long_signal.copy_from_slice(&batch.long_signal);
        dst_short_signal.copy_from_slice(&batch.short_signal);
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn nonlinear_regression_zero_lag_moving_average_output_into_js(
    data: &[f64],
    zlma_period: usize,
    regression_period: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value =
        nonlinear_regression_zero_lag_moving_average_js(data, zlma_period, regression_period)?;
    crate::write_wasm_object_f64_outputs(
        "nonlinear_regression_zero_lag_moving_average_output_into_js",
        &value,
        out,
    )
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn nonlinear_regression_zero_lag_moving_average_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = nonlinear_regression_zero_lag_moving_average_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "nonlinear_regression_zero_lag_moving_average_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data(len: usize) -> Vec<f64> {
        (0..len)
            .map(|i| {
                let x = i as f64;
                (x * 0.07).sin() + 0.35 * (x * 0.013).cos() + x * 0.001
            })
            .collect()
    }

    #[test]
    fn nonlinear_regression_zero_lag_moving_average_output_contract() {
        let data = sample_data(256);
        let input = NonlinearRegressionZeroLagMovingAverageInput::from_slice(
            &data,
            NonlinearRegressionZeroLagMovingAverageParams::default(),
        );
        let out = nonlinear_regression_zero_lag_moving_average(&input).unwrap();

        assert_eq!(out.value.len(), data.len());
        assert_eq!(out.signal.len(), data.len());
        assert_eq!(out.long_signal.len(), data.len());
        assert_eq!(out.short_signal.len(), data.len());
        assert_eq!(out.value.iter().position(|v| !v.is_nan()), Some(42));
        assert_eq!(out.signal.iter().position(|v| !v.is_nan()), Some(43));
    }

    #[test]
    fn nonlinear_regression_zero_lag_moving_average_invalid_params() {
        let data = sample_data(128);
        let input = NonlinearRegressionZeroLagMovingAverageInput::from_slice(
            &data,
            NonlinearRegressionZeroLagMovingAverageParams {
                zlma_period: Some(0),
                regression_period: Some(15),
            },
        );
        assert!(matches!(
            nonlinear_regression_zero_lag_moving_average(&input),
            Err(NonlinearRegressionZeroLagMovingAverageError::InvalidZlmaPeriod { .. })
        ));
    }

    #[test]
    fn nonlinear_regression_zero_lag_moving_average_stream_matches_batch() {
        let data = sample_data(320);
        let input = NonlinearRegressionZeroLagMovingAverageInput::from_slice(
            &data,
            NonlinearRegressionZeroLagMovingAverageParams {
                zlma_period: Some(17),
                regression_period: Some(11),
            },
        );
        let batch = nonlinear_regression_zero_lag_moving_average(&input).unwrap();
        let mut stream =
            NonlinearRegressionZeroLagMovingAverageStream::try_new(input.params.clone()).unwrap();

        let mut value = Vec::with_capacity(data.len());
        let mut signal = Vec::with_capacity(data.len());
        let mut long_signal = Vec::with_capacity(data.len());
        let mut short_signal = Vec::with_capacity(data.len());

        for sample in data {
            if let Some(point) = stream.update(sample) {
                value.push(point.value);
                signal.push(point.signal);
                long_signal.push(point.long_signal);
                short_signal.push(point.short_signal);
            } else {
                value.push(f64::NAN);
                signal.push(f64::NAN);
                long_signal.push(0.0);
                short_signal.push(0.0);
            }
        }

        for i in 0..value.len() {
            let equal_value = (value[i].is_nan() && batch.value[i].is_nan())
                || (value[i] - batch.value[i]).abs() < 1e-12;
            let equal_signal = (signal[i].is_nan() && batch.signal[i].is_nan())
                || (signal[i] - batch.signal[i]).abs() < 1e-12;
            assert!(equal_value, "value mismatch at {i}");
            assert!(equal_signal, "signal mismatch at {i}");
            assert!((long_signal[i] - batch.long_signal[i]).abs() < 1e-12);
            assert!((short_signal[i] - batch.short_signal[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn nonlinear_regression_zero_lag_moving_average_into_matches_api() {
        let data = sample_data(300);
        let input = NonlinearRegressionZeroLagMovingAverageInput::from_slice(
            &data,
            NonlinearRegressionZeroLagMovingAverageParams::default(),
        );
        let baseline = nonlinear_regression_zero_lag_moving_average(&input).unwrap();

        let mut value = vec![0.0; data.len()];
        let mut signal = vec![0.0; data.len()];
        let mut long_signal = vec![0.0; data.len()];
        let mut short_signal = vec![0.0; data.len()];
        nonlinear_regression_zero_lag_moving_average_into(
            &input,
            &mut value,
            &mut signal,
            &mut long_signal,
            &mut short_signal,
        )
        .unwrap();

        for i in 0..data.len() {
            let value_match = (value[i].is_nan() && baseline.value[i].is_nan())
                || (value[i] - baseline.value[i]).abs() < 1e-12;
            let signal_match = (signal[i].is_nan() && baseline.signal[i].is_nan())
                || (signal[i] - baseline.signal[i]).abs() < 1e-12;
            assert!(value_match, "value mismatch at {i}");
            assert!(signal_match, "signal mismatch at {i}");
            assert!((long_signal[i] - baseline.long_signal[i]).abs() < 1e-12);
            assert!((short_signal[i] - baseline.short_signal[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn nonlinear_regression_zero_lag_moving_average_batch_single_matches_single() {
        let data = sample_data(280);
        let batch = nonlinear_regression_zero_lag_moving_average_batch_with_kernel(
            &data,
            &NonlinearRegressionZeroLagMovingAverageBatchRange {
                zlma_period: (15, 15, 0),
                regression_period: (15, 15, 0),
            },
            Kernel::Auto,
        )
        .unwrap();
        let single = nonlinear_regression_zero_lag_moving_average(
            &NonlinearRegressionZeroLagMovingAverageInput::from_slice(
                &data,
                NonlinearRegressionZeroLagMovingAverageParams::default(),
            ),
        )
        .unwrap();

        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, data.len());
        for i in 0..data.len() {
            let value_match = (batch.value[i].is_nan() && single.value[i].is_nan())
                || (batch.value[i] - single.value[i]).abs() < 1e-12;
            assert!(value_match, "batch value mismatch at {i}");
        }
    }
}
