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

use crate::indicators::moving_averages::linreg::{
    linreg_with_kernel, LinRegInput, LinRegParams, LinRegStream,
};
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
use std::convert::AsRef;
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

impl<'a> AsRef<[f64]> for LinearRegressionIntensityInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            LinearRegressionIntensityData::Slice(slice) => slice,
            LinearRegressionIntensityData::Candles { candles, source } => {
                linear_regression_intensity_source_type(candles, source)
            }
        }
    }
}

#[inline(always)]
fn linear_regression_intensity_source_type<'a>(candles: &'a Candles, source: &str) -> &'a [f64] {
    match source {
        "close" => &candles.close,
        "open" => &candles.open,
        "high" => &candles.high,
        "low" => &candles.low,
        "volume" => &candles.volume,
        "hl2" => &candles.hl2,
        "hlc3" => &candles.hlc3,
        "ohlc4" => &candles.ohlc4,
        "hlcc4" | "hlcc" => &candles.hlcc4,
        _ => source_type(candles, source),
    }
}

#[derive(Debug, Clone)]
pub enum LinearRegressionIntensityData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct LinearRegressionIntensityOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct LinearRegressionIntensityParams {
    pub lookback_period: Option<usize>,
    pub range_tolerance: Option<f64>,
    pub linreg_length: Option<usize>,
}

impl Default for LinearRegressionIntensityParams {
    fn default() -> Self {
        Self {
            lookback_period: Some(12),
            range_tolerance: Some(90.0),
            linreg_length: Some(90),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LinearRegressionIntensityInput<'a> {
    pub data: LinearRegressionIntensityData<'a>,
    pub params: LinearRegressionIntensityParams,
}

impl<'a> LinearRegressionIntensityInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: LinearRegressionIntensityParams,
    ) -> Self {
        Self {
            data: LinearRegressionIntensityData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: LinearRegressionIntensityParams) -> Self {
        Self {
            data: LinearRegressionIntensityData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", LinearRegressionIntensityParams::default())
    }

    #[inline]
    pub fn get_lookback_period(&self) -> usize {
        self.params.lookback_period.unwrap_or(12)
    }

    #[inline]
    pub fn get_range_tolerance(&self) -> f64 {
        self.params.range_tolerance.unwrap_or(90.0)
    }

    #[inline]
    pub fn get_linreg_length(&self) -> usize {
        self.params.linreg_length.unwrap_or(90)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct LinearRegressionIntensityBuilder {
    lookback_period: Option<usize>,
    range_tolerance: Option<f64>,
    linreg_length: Option<usize>,
    kernel: Kernel,
}

impl Default for LinearRegressionIntensityBuilder {
    fn default() -> Self {
        Self {
            lookback_period: None,
            range_tolerance: None,
            linreg_length: None,
            kernel: Kernel::Auto,
        }
    }
}

impl LinearRegressionIntensityBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn lookback_period(mut self, value: usize) -> Self {
        self.lookback_period = Some(value);
        self
    }

    #[inline(always)]
    pub fn range_tolerance(mut self, value: f64) -> Self {
        self.range_tolerance = Some(value);
        self
    }

    #[inline(always)]
    pub fn linreg_length(mut self, value: usize) -> Self {
        self.linreg_length = Some(value);
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
    ) -> Result<LinearRegressionIntensityOutput, LinearRegressionIntensityError> {
        self.apply_source(candles, "close")
    }

    #[inline(always)]
    pub fn apply_source(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<LinearRegressionIntensityOutput, LinearRegressionIntensityError> {
        let input = LinearRegressionIntensityInput::from_candles(
            candles,
            source,
            LinearRegressionIntensityParams {
                lookback_period: self.lookback_period,
                range_tolerance: self.range_tolerance,
                linreg_length: self.linreg_length,
            },
        );
        linear_regression_intensity_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<LinearRegressionIntensityOutput, LinearRegressionIntensityError> {
        let input = LinearRegressionIntensityInput::from_slice(
            data,
            LinearRegressionIntensityParams {
                lookback_period: self.lookback_period,
                range_tolerance: self.range_tolerance,
                linreg_length: self.linreg_length,
            },
        );
        linear_regression_intensity_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<LinearRegressionIntensityStream, LinearRegressionIntensityError> {
        LinearRegressionIntensityStream::try_new(LinearRegressionIntensityParams {
            lookback_period: self.lookback_period,
            range_tolerance: self.range_tolerance,
            linreg_length: self.linreg_length,
        })
    }
}

#[derive(Debug, Error)]
pub enum LinearRegressionIntensityError {
    #[error("linear_regression_intensity: Input data slice is empty.")]
    EmptyInputData,
    #[error("linear_regression_intensity: All values are NaN.")]
    AllValuesNaN,
    #[error(
        "linear_regression_intensity: Invalid lookback period: lookback_period = {lookback_period}, data length = {data_len}"
    )]
    InvalidLookbackPeriod {
        lookback_period: usize,
        data_len: usize,
    },
    #[error(
        "linear_regression_intensity: Invalid range tolerance: range_tolerance = {range_tolerance}"
    )]
    InvalidRangeTolerance { range_tolerance: f64 },
    #[error(
        "linear_regression_intensity: Invalid linreg length: linreg_length = {linreg_length}, data length = {data_len}"
    )]
    InvalidLinregLength {
        linreg_length: usize,
        data_len: usize,
    },
    #[error(
        "linear_regression_intensity: Not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "linear_regression_intensity: Output length mismatch: expected = {expected}, got = {got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("linear_regression_intensity: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRangeUsize {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("linear_regression_intensity: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRangeF64 { start: f64, end: f64, step: f64 },
    #[error("linear_regression_intensity: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Copy, Clone, Debug)]
struct ResolvedParams {
    lookback_period: usize,
    range_tolerance: f64,
    linreg_length: usize,
}

#[inline(always)]
fn first_valid_index(data: &[f64]) -> Option<usize> {
    data.iter().position(|x| x.is_finite())
}

#[inline(always)]
fn total_combinations(lookback_period: usize) -> usize {
    lookback_period.saturating_mul(lookback_period.saturating_sub(1)) / 2
}

#[inline(always)]
fn trend_to_intensity(trend: i64, total_combinations: usize) -> f64 {
    if total_combinations == 0 {
        0.0
    } else {
        trend as f64 / total_combinations as f64
    }
}

#[inline(always)]
fn pair_sign(later: f64, earlier: f64) -> i64 {
    if later > earlier {
        1
    } else if later < earlier {
        -1
    } else {
        0
    }
}

#[inline(always)]
fn warmup_prefix(first: usize, params: ResolvedParams) -> usize {
    first
        .saturating_add(params.linreg_length)
        .saturating_add(params.lookback_period)
        .saturating_sub(2)
}

#[inline]
fn linear_regression_intensity_prepare<'a>(
    input: &'a LinearRegressionIntensityInput,
) -> Result<(&'a [f64], usize, ResolvedParams), LinearRegressionIntensityError> {
    let data = input.as_ref();
    let data_len = data.len();
    if data_len == 0 {
        return Err(LinearRegressionIntensityError::EmptyInputData);
    }

    let first = first_valid_index(data).ok_or(LinearRegressionIntensityError::AllValuesNaN)?;
    let valid = data_len - first;
    let lookback_period = input.get_lookback_period();
    let range_tolerance = input.get_range_tolerance();
    let linreg_length = input.get_linreg_length();

    if lookback_period == 0 || lookback_period > data_len {
        return Err(LinearRegressionIntensityError::InvalidLookbackPeriod {
            lookback_period,
            data_len,
        });
    }
    if !range_tolerance.is_finite() || !(0.0..=100.0).contains(&range_tolerance) {
        return Err(LinearRegressionIntensityError::InvalidRangeTolerance { range_tolerance });
    }
    if linreg_length == 0 || linreg_length > data_len {
        return Err(LinearRegressionIntensityError::InvalidLinregLength {
            linreg_length,
            data_len,
        });
    }

    let needed = linreg_length + lookback_period - 1;
    if valid < needed {
        return Err(LinearRegressionIntensityError::NotEnoughValidData { needed, valid });
    }

    Ok((
        data,
        first,
        ResolvedParams {
            lookback_period,
            range_tolerance,
            linreg_length,
        },
    ))
}

#[inline]
fn naive_window_trend(window: &VecDeque<f64>) -> i64 {
    let mut trend = 0i64;
    let len = window.len();
    for i in 0..len.saturating_sub(1) {
        let a = window[i];
        for j in (i + 1)..len {
            let b = window[j];
            if a != b {
                trend += if b > a { 1 } else { -1 };
            }
        }
    }
    trend
}

#[inline]
fn compute_fallback(data: &[f64], params: ResolvedParams, out: &mut [f64]) {
    let total = total_combinations(params.lookback_period);
    let mut linreg_stream = LinRegStream::try_new(LinRegParams {
        period: Some(params.linreg_length),
    })
    .expect("validated linreg params");
    let mut window = VecDeque::with_capacity(params.lookback_period);

    for (index, &value) in data.iter().enumerate() {
        if !value.is_finite() {
            linreg_stream = LinRegStream::try_new(LinRegParams {
                period: Some(params.linreg_length),
            })
            .expect("validated linreg params");
            window.clear();
            continue;
        }
        let Some(lr) = linreg_stream.update(value) else {
            continue;
        };
        if !lr.is_finite() {
            window.clear();
            continue;
        }
        window.push_back(lr);
        if window.len() > params.lookback_period {
            window.pop_front();
        }
        if window.len() != params.lookback_period {
            continue;
        }

        let trend = naive_window_trend(&window);
        out[index] = trend_to_intensity(trend, total);
    }
}

#[derive(Clone, Debug)]
struct Fenwick {
    tree: Vec<i32>,
}

impl Fenwick {
    #[inline]
    fn new(size: usize) -> Self {
        Self {
            tree: vec![0; size + 1],
        }
    }

    #[inline]
    fn add(&mut self, mut index: usize, delta: i32) {
        while index < self.tree.len() {
            self.tree[index] += delta;
            index += index & (!index + 1);
        }
    }

    #[inline]
    fn prefix_sum(&self, mut index: usize) -> i32 {
        let mut sum = 0;
        while index > 0 {
            sum += self.tree[index];
            index &= index - 1;
        }
        sum
    }
}

#[inline]
fn rank_of(values: &[f64], value: f64) -> usize {
    values
        .binary_search_by(|probe| probe.total_cmp(&value))
        .expect("rank present")
        + 1
}

#[inline]
fn compute_fast_from_linreg(linreg: &[f64], params: ResolvedParams, out: &mut [f64]) {
    let Some(first_lr) = first_valid_index(linreg) else {
        return;
    };
    let total = total_combinations(params.lookback_period);

    if params.lookback_period == 1 {
        for value in &mut out[first_lr..] {
            *value = 0.0;
        }
        return;
    }

    let mut unique = linreg[first_lr..]
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    unique.sort_by(|a, b| a.total_cmp(b));
    unique.dedup();

    let first_out = first_lr + params.lookback_period - 1;
    let mut fenwick = Fenwick::new(unique.len());
    let mut window = VecDeque::with_capacity(params.lookback_period);
    let mut trend = 0i64;

    for &value in &linreg[first_lr..=first_out] {
        let rank = rank_of(&unique, value);
        let current = window.len() as i64;
        let less = fenwick.prefix_sum(rank - 1) as i64;
        let greater = current - fenwick.prefix_sum(rank) as i64;
        trend += less - greater;
        fenwick.add(rank, 1);
        window.push_back(rank);
    }
    out[first_out] = trend_to_intensity(trend, total);

    for index in (first_out + 1)..linreg.len() {
        let old_rank = window.pop_front().expect("window");
        fenwick.add(old_rank, -1);
        let remaining = (params.lookback_period - 1) as i64;
        let less_old = fenwick.prefix_sum(old_rank - 1) as i64;
        let greater_old = remaining - fenwick.prefix_sum(old_rank) as i64;
        trend -= greater_old - less_old;

        let value = linreg[index];
        let rank = rank_of(&unique, value);
        let less_new = fenwick.prefix_sum(rank - 1) as i64;
        let greater_new = remaining - fenwick.prefix_sum(rank) as i64;
        trend += less_new - greater_new;
        fenwick.add(rank, 1);
        window.push_back(rank);
        out[index] = trend_to_intensity(trend, total);
    }
}

#[inline]
fn compute_fused_small_lookback(
    data: &[f64],
    first: usize,
    params: ResolvedParams,
    out: &mut [f64],
) {
    let period = params.linreg_length;
    let lookback = params.lookback_period;
    let period_f = period as f64;
    let x_sum = ((period * (period + 1)) / 2) as f64;
    let x2_sum = ((period * (period + 1) * (2 * period + 1)) / 6) as f64;
    let denom_inv = 1.0 / (period_f * x2_sum - x_sum * x_sum);
    let inv_period = 1.0 / period_f;

    let mut y_sum = 0.0;
    let mut xy_sum = 0.0;
    let init_slice = &data[first..first + period - 1];
    let mut k = 1usize;
    for &value in init_slice {
        y_sum += value;
        xy_sum += (k as f64) * value;
        k += 1;
    }

    let total = total_combinations(lookback);
    let mut window = vec![0.0; lookback];
    let mut head = 0usize;
    let mut count = 0usize;
    let mut trend = 0i64;
    let mut idx = first + period - 1;
    let mut old_idx = first;

    while idx < data.len() {
        let new_value = data[idx];
        y_sum += new_value;
        xy_sum += new_value * period_f;

        let b = (period_f * xy_sum - x_sum * y_sum) * denom_inv;
        let a = (y_sum - b * x_sum) * inv_period;
        let lr = a + b * period_f;

        if lookback == 1 {
            out[idx] = 0.0;
        } else if count < lookback {
            for &prior in &window[..count] {
                trend += pair_sign(lr, prior);
            }
            window[count] = lr;
            count += 1;
            if count == lookback {
                out[idx] = trend_to_intensity(trend, total);
            }
        } else {
            let old = window[head];
            let mut remove_delta = 0i64;
            let mut add_delta = 0i64;
            let mut pos = head + 1;
            if pos == lookback {
                pos = 0;
            }
            for _ in 1..lookback {
                let value = window[pos];
                remove_delta += pair_sign(value, old);
                add_delta += pair_sign(lr, value);
                pos += 1;
                if pos == lookback {
                    pos = 0;
                }
            }
            trend += add_delta - remove_delta;
            window[head] = lr;
            head += 1;
            if head == lookback {
                head = 0;
            }
            out[idx] = trend_to_intensity(trend, total);
        }

        xy_sum -= y_sum;
        y_sum -= data[old_idx];
        idx += 1;
        old_idx += 1;
    }
}

#[inline]
fn linear_regression_intensity_compute_into(
    data: &[f64],
    params: ResolvedParams,
    kernel: Kernel,
    out: &mut [f64],
) -> Result<(), LinearRegressionIntensityError> {
    if params.lookback_period <= 64 {
        if let Some(first) = first_valid_index(data) {
            if data[first..].iter().all(|value| value.is_finite()) {
                compute_fused_small_lookback(data, first, params, out);
                return Ok(());
            }
        }
    }

    let linreg = linreg_with_kernel(
        &LinRegInput::from_slice(
            data,
            LinRegParams {
                period: Some(params.linreg_length),
            },
        ),
        kernel,
    )
    .map_err(|err| match err {
        crate::indicators::moving_averages::linreg::LinRegError::EmptyInputData => {
            LinearRegressionIntensityError::EmptyInputData
        }
        crate::indicators::moving_averages::linreg::LinRegError::AllValuesNaN => {
            LinearRegressionIntensityError::AllValuesNaN
        }
        crate::indicators::moving_averages::linreg::LinRegError::InvalidPeriod {
            period,
            data_len,
        } => LinearRegressionIntensityError::InvalidLinregLength {
            linreg_length: period,
            data_len,
        },
        crate::indicators::moving_averages::linreg::LinRegError::NotEnoughValidData {
            needed,
            valid,
        } => LinearRegressionIntensityError::NotEnoughValidData { needed, valid },
        _ => LinearRegressionIntensityError::AllValuesNaN,
    })?
    .values;

    let Some(first_lr) = first_valid_index(&linreg) else {
        return Ok(());
    };
    if linreg[first_lr..].iter().all(|value| value.is_finite()) {
        compute_fast_from_linreg(&linreg, params, out);
    } else {
        compute_fallback(data, params, out);
    }
    Ok(())
}

#[inline]
pub fn linear_regression_intensity(
    input: &LinearRegressionIntensityInput,
) -> Result<LinearRegressionIntensityOutput, LinearRegressionIntensityError> {
    linear_regression_intensity_with_kernel(input, Kernel::Auto)
}

pub fn linear_regression_intensity_with_kernel(
    input: &LinearRegressionIntensityInput,
    kernel: Kernel,
) -> Result<LinearRegressionIntensityOutput, LinearRegressionIntensityError> {
    let (data, first, params) = linear_regression_intensity_prepare(input)?;
    let mut out = alloc_with_nan_prefix(data.len(), warmup_prefix(first, params));
    linear_regression_intensity_compute_into(data, params, kernel, &mut out)?;
    Ok(LinearRegressionIntensityOutput { values: out })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn linear_regression_intensity_into(
    input: &LinearRegressionIntensityInput,
    out: &mut [f64],
) -> Result<(), LinearRegressionIntensityError> {
    linear_regression_intensity_into_slice(out, input, Kernel::Auto)
}

pub fn linear_regression_intensity_into_slice(
    out: &mut [f64],
    input: &LinearRegressionIntensityInput,
    kernel: Kernel,
) -> Result<(), LinearRegressionIntensityError> {
    let (data, _first, params) = linear_regression_intensity_prepare(input)?;
    if out.len() != data.len() {
        return Err(LinearRegressionIntensityError::OutputLengthMismatch {
            expected: data.len(),
            got: out.len(),
        });
    }
    out.fill(f64::NAN);
    linear_regression_intensity_compute_into(data, params, kernel, out)
}

#[derive(Clone, Debug)]
pub struct LinearRegressionIntensityStream {
    lookback_period: usize,
    linreg_length: usize,
    linreg_stream: LinRegStream,
    window: VecDeque<f64>,
}

impl LinearRegressionIntensityStream {
    #[inline]
    pub fn try_new(
        params: LinearRegressionIntensityParams,
    ) -> Result<Self, LinearRegressionIntensityError> {
        let lookback_period = params.lookback_period.unwrap_or(12);
        let range_tolerance = params.range_tolerance.unwrap_or(90.0);
        let linreg_length = params.linreg_length.unwrap_or(90);

        if lookback_period == 0 {
            return Err(LinearRegressionIntensityError::InvalidLookbackPeriod {
                lookback_period,
                data_len: 0,
            });
        }
        if !range_tolerance.is_finite() || !(0.0..=100.0).contains(&range_tolerance) {
            return Err(LinearRegressionIntensityError::InvalidRangeTolerance { range_tolerance });
        }
        if linreg_length == 0 {
            return Err(LinearRegressionIntensityError::InvalidLinregLength {
                linreg_length,
                data_len: 0,
            });
        }

        Ok(Self {
            lookback_period,
            linreg_length,
            linreg_stream: LinRegStream::try_new(LinRegParams {
                period: Some(linreg_length),
            })
            .map_err(|_| LinearRegressionIntensityError::InvalidLinregLength {
                linreg_length,
                data_len: 0,
            })?,
            window: VecDeque::with_capacity(lookback_period),
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        if !value.is_finite() {
            self.linreg_stream = LinRegStream::try_new(LinRegParams {
                period: Some(self.linreg_length),
            })
            .expect("validated linreg length");
            self.window.clear();
            return None;
        }
        let linreg_value = self.linreg_stream.update(value)?;
        if !linreg_value.is_finite() {
            self.window.clear();
            return None;
        }
        self.window.push_back(linreg_value);
        if self.window.len() > self.lookback_period {
            self.window.pop_front();
        }
        if self.window.len() != self.lookback_period {
            return None;
        }
        Some(trend_to_intensity(
            naive_window_trend(&self.window),
            total_combinations(self.lookback_period),
        ))
    }

    #[inline(always)]
    pub fn update_reset_on_nan(&mut self, value: f64) -> Option<f64> {
        self.update(value)
    }
}

#[derive(Clone, Debug)]
pub struct LinearRegressionIntensityBatchRange {
    pub lookback_period: (usize, usize, usize),
    pub range_tolerance: (f64, f64, f64),
    pub linreg_length: (usize, usize, usize),
}

impl Default for LinearRegressionIntensityBatchRange {
    fn default() -> Self {
        Self {
            lookback_period: (12, 12, 0),
            range_tolerance: (90.0, 90.0, 0.0),
            linreg_length: (90, 90, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct LinearRegressionIntensityBatchBuilder {
    range: LinearRegressionIntensityBatchRange,
    kernel: Kernel,
}

impl LinearRegressionIntensityBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    pub fn lookback_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.lookback_period = (start, end, step);
        self
    }

    pub fn range_tolerance_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.range_tolerance = (start, end, step);
        self
    }

    pub fn linreg_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.linreg_length = (start, end, step);
        self
    }

    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<LinearRegressionIntensityBatchOutput, LinearRegressionIntensityError> {
        linear_regression_intensity_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<LinearRegressionIntensityBatchOutput, LinearRegressionIntensityError> {
        self.apply_slice(linear_regression_intensity_source_type(candles, source))
    }
}

#[derive(Clone, Debug)]
pub struct LinearRegressionIntensityBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<LinearRegressionIntensityParams>,
    pub rows: usize,
    pub cols: usize,
}

impl LinearRegressionIntensityBatchOutput {
    pub fn row_for_params(&self, params: &LinearRegressionIntensityParams) -> Option<usize> {
        self.combos.iter().position(|combo| {
            combo.lookback_period.unwrap_or(12) == params.lookback_period.unwrap_or(12)
                && (combo.range_tolerance.unwrap_or(90.0) - params.range_tolerance.unwrap_or(90.0))
                    .abs()
                    <= 1e-12
                && combo.linreg_length.unwrap_or(90) == params.linreg_length.unwrap_or(90)
        })
    }

    pub fn values_for(&self, params: &LinearRegressionIntensityParams) -> Option<&[f64]> {
        self.row_for_params(params).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline]
fn axis_usize(range: (usize, usize, usize)) -> Result<Vec<usize>, LinearRegressionIntensityError> {
    let (start, end, step) = range;
    if start < 1 || end < 1 {
        return Err(LinearRegressionIntensityError::InvalidRangeUsize { start, end, step });
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
        return Err(LinearRegressionIntensityError::InvalidRangeUsize { start, end, step });
    }
    Ok(out)
}

#[inline]
fn axis_f64(range: (f64, f64, f64)) -> Result<Vec<f64>, LinearRegressionIntensityError> {
    let (start, end, step) = range;
    if !start.is_finite() || !end.is_finite() || !step.is_finite() || step < 0.0 {
        return Err(LinearRegressionIntensityError::InvalidRangeF64 { start, end, step });
    }
    if step == 0.0 || (start - end).abs() <= 1e-12 {
        return Ok(vec![start]);
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
        return Err(LinearRegressionIntensityError::InvalidRangeF64 { start, end, step });
    }
    Ok(out)
}

pub fn expand_grid_linear_regression_intensity(
    sweep: &LinearRegressionIntensityBatchRange,
) -> Result<Vec<LinearRegressionIntensityParams>, LinearRegressionIntensityError> {
    let lookback_periods = axis_usize(sweep.lookback_period)?;
    let range_tolerances = axis_f64(sweep.range_tolerance)?;
    let linreg_lengths = axis_usize(sweep.linreg_length)?;

    let mut out = Vec::new();
    for lookback_period in &lookback_periods {
        for range_tolerance in &range_tolerances {
            for linreg_length in &linreg_lengths {
                out.push(LinearRegressionIntensityParams {
                    lookback_period: Some(*lookback_period),
                    range_tolerance: Some(*range_tolerance),
                    linreg_length: Some(*linreg_length),
                });
            }
        }
    }
    Ok(out)
}

pub fn linear_regression_intensity_batch_with_kernel(
    data: &[f64],
    sweep: &LinearRegressionIntensityBatchRange,
    kernel: Kernel,
) -> Result<LinearRegressionIntensityBatchOutput, LinearRegressionIntensityError> {
    let batch_kernel = match kernel {
        Kernel::Auto => Kernel::ScalarBatch,
        other if other.is_batch() => other,
        other => return Err(LinearRegressionIntensityError::InvalidKernelForBatch(other)),
    };
    linear_regression_intensity_batch_impl(data, sweep, batch_kernel.to_non_batch(), true)
}

pub fn linear_regression_intensity_batch_slice(
    data: &[f64],
    sweep: &LinearRegressionIntensityBatchRange,
) -> Result<LinearRegressionIntensityBatchOutput, LinearRegressionIntensityError> {
    linear_regression_intensity_batch_impl(data, sweep, Kernel::Scalar, false)
}

pub fn linear_regression_intensity_batch_par_slice(
    data: &[f64],
    sweep: &LinearRegressionIntensityBatchRange,
) -> Result<LinearRegressionIntensityBatchOutput, LinearRegressionIntensityError> {
    linear_regression_intensity_batch_impl(data, sweep, Kernel::Scalar, true)
}

fn linear_regression_intensity_batch_impl(
    data: &[f64],
    sweep: &LinearRegressionIntensityBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<LinearRegressionIntensityBatchOutput, LinearRegressionIntensityError> {
    let combos = expand_grid_linear_regression_intensity(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    if cols == 0 {
        return Err(LinearRegressionIntensityError::EmptyInputData);
    }

    for combo in &combos {
        let input = LinearRegressionIntensityInput::from_slice(data, combo.clone());
        let _ = linear_regression_intensity_prepare(&input)?;
    }

    let first = first_valid_index(data).ok_or(LinearRegressionIntensityError::AllValuesNaN)?;
    let warmups = combos
        .iter()
        .map(|params| {
            warmup_prefix(
                first,
                ResolvedParams {
                    lookback_period: params.lookback_period.unwrap_or(12),
                    range_tolerance: params.range_tolerance.unwrap_or(90.0),
                    linreg_length: params.linreg_length.unwrap_or(90),
                },
            )
        })
        .collect::<Vec<_>>();

    let mut matrix = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut matrix, cols, &warmups);
    let mut guard = ManuallyDrop::new(matrix);
    let out_mu: &mut [MaybeUninit<f64>] =
        unsafe { std::slice::from_raw_parts_mut(guard.as_mut_ptr(), guard.len()) };

    let do_row = |row: usize, row_mu: &mut [MaybeUninit<f64>]| {
        let params = ResolvedParams {
            lookback_period: combos[row].lookback_period.unwrap_or(12),
            range_tolerance: combos[row].range_tolerance.unwrap_or(90.0),
            linreg_length: combos[row].linreg_length.unwrap_or(90),
        };
        let row_out = unsafe {
            std::slice::from_raw_parts_mut(row_mu.as_mut_ptr() as *mut f64, row_mu.len())
        };
        linear_regression_intensity_compute_into(data, params, kernel, row_out)
            .expect("batch row validated");
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

    Ok(LinearRegressionIntensityBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

fn linear_regression_intensity_batch_inner_into(
    data: &[f64],
    sweep: &LinearRegressionIntensityBatchRange,
    kernel: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<(), LinearRegressionIntensityError> {
    let combos = expand_grid_linear_regression_intensity(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    for combo in &combos {
        let input = LinearRegressionIntensityInput::from_slice(data, combo.clone());
        let _ = linear_regression_intensity_prepare(&input)?;
    }
    let expected =
        rows.checked_mul(cols)
            .ok_or(LinearRegressionIntensityError::OutputLengthMismatch {
                expected: usize::MAX,
                got: out.len(),
            })?;
    if expected != out.len() {
        return Err(LinearRegressionIntensityError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    for row_out in out.chunks_mut(cols) {
        row_out.fill(f64::NAN);
    }

    let do_row = |row: usize, row_out: &mut [f64]| {
        let params = ResolvedParams {
            lookback_period: combos[row].lookback_period.unwrap_or(12),
            range_tolerance: combos[row].range_tolerance.unwrap_or(90.0),
            linreg_length: combos[row].linreg_length.unwrap_or(90),
        };
        linear_regression_intensity_compute_into(data, params, kernel, row_out)
            .expect("batch row validated");
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
#[pyfunction(name = "linear_regression_intensity")]
#[pyo3(signature = (data, lookback_period=12, range_tolerance=90.0, linreg_length=90, kernel=None))]
pub fn linear_regression_intensity_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    lookback_period: usize,
    range_tolerance: f64,
    linreg_length: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = data.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = LinearRegressionIntensityInput::from_slice(
        data,
        LinearRegressionIntensityParams {
            lookback_period: Some(lookback_period),
            range_tolerance: Some(range_tolerance),
            linreg_length: Some(linreg_length),
        },
    );
    let output = py
        .allow_threads(|| linear_regression_intensity_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(output.values.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "LinearRegressionIntensityStream")]
pub struct LinearRegressionIntensityStreamPy {
    stream: LinearRegressionIntensityStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl LinearRegressionIntensityStreamPy {
    #[new]
    #[pyo3(signature = (lookback_period=12, range_tolerance=90.0, linreg_length=90))]
    fn new(lookback_period: usize, range_tolerance: f64, linreg_length: usize) -> PyResult<Self> {
        let stream = LinearRegressionIntensityStream::try_new(LinearRegressionIntensityParams {
            lookback_period: Some(lookback_period),
            range_tolerance: Some(range_tolerance),
            linreg_length: Some(linreg_length),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update_reset_on_nan(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "linear_regression_intensity_batch")]
#[pyo3(signature = (data, lookback_period_range=(12, 12, 0), range_tolerance_range=(90.0, 90.0, 0.0), linreg_length_range=(90, 90, 0), kernel=None))]
pub fn linear_regression_intensity_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    lookback_period_range: (usize, usize, usize),
    range_tolerance_range: (f64, f64, f64),
    linreg_length_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let sweep = LinearRegressionIntensityBatchRange {
        lookback_period: lookback_period_range,
        range_tolerance: range_tolerance_range,
        linreg_length: linreg_length_range,
    };
    let combos = expand_grid_linear_regression_intensity(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = data.len();
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
        linear_regression_intensity_batch_inner_into(
            data,
            &sweep,
            batch_kernel.to_non_batch(),
            true,
            out,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", arr.reshape((rows, cols))?)?;
    dict.set_item(
        "lookback_periods",
        combos
            .iter()
            .map(|params| params.lookback_period.unwrap_or(12) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "range_tolerances",
        combos
            .iter()
            .map(|params| params.range_tolerance.unwrap_or(90.0))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "linreg_lengths",
        combos
            .iter()
            .map(|params| params.linreg_length.unwrap_or(90) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_linear_regression_intensity_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(linear_regression_intensity_py, m)?)?;
    m.add_function(wrap_pyfunction!(linear_regression_intensity_batch_py, m)?)?;
    m.add_class::<LinearRegressionIntensityStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LinearRegressionIntensityBatchConfig {
    lookback_period_range: Vec<usize>,
    range_tolerance_range: Vec<f64>,
    linreg_length_range: Vec<usize>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LinearRegressionIntensityBatchJsOutput {
    values: Vec<f64>,
    rows: usize,
    cols: usize,
    combos: Vec<LinearRegressionIntensityParams>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "linear_regression_intensity_js")]
pub fn linear_regression_intensity_js(
    data: &[f64],
    lookback_period: usize,
    range_tolerance: f64,
    linreg_length: usize,
) -> Result<Vec<f64>, JsValue> {
    let input = LinearRegressionIntensityInput::from_slice(
        data,
        LinearRegressionIntensityParams {
            lookback_period: Some(lookback_period),
            range_tolerance: Some(range_tolerance),
            linreg_length: Some(linreg_length),
        },
    );
    let mut out = vec![0.0; data.len()];
    linear_regression_intensity_into_slice(&mut out, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "linear_regression_intensity_batch_js")]
pub fn linear_regression_intensity_batch_js(
    data: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: LinearRegressionIntensityBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.lookback_period_range.len() != 3
        || config.range_tolerance_range.len() != 3
        || config.linreg_length_range.len() != 3
    {
        return Err(JsValue::from_str(
            "Invalid config: each range must have exactly 3 elements [start, end, step]",
        ));
    }
    let sweep = LinearRegressionIntensityBatchRange {
        lookback_period: (
            config.lookback_period_range[0],
            config.lookback_period_range[1],
            config.lookback_period_range[2],
        ),
        range_tolerance: (
            config.range_tolerance_range[0],
            config.range_tolerance_range[1],
            config.range_tolerance_range[2],
        ),
        linreg_length: (
            config.linreg_length_range[0],
            config.linreg_length_range[1],
            config.linreg_length_range[2],
        ),
    };
    let output = linear_regression_intensity_batch_slice(data, &sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&LinearRegressionIntensityBatchJsOutput {
        values: output.values,
        rows: output.rows,
        cols: output.cols,
        combos: output.combos,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn linear_regression_intensity_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn linear_regression_intensity_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, 0, len);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn linear_regression_intensity_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    lookback_period: usize,
    range_tolerance: f64,
    linreg_length: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to linear_regression_intensity_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len);
        let input = LinearRegressionIntensityInput::from_slice(
            data,
            LinearRegressionIntensityParams {
                lookback_period: Some(lookback_period),
                range_tolerance: Some(range_tolerance),
                linreg_length: Some(linreg_length),
            },
        );
        linear_regression_intensity_into_slice(out, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "linear_regression_intensity_into_host")]
pub fn linear_regression_intensity_into_host(
    data: &[f64],
    out_ptr: *mut f64,
    lookback_period: usize,
    range_tolerance: f64,
    linreg_length: usize,
) -> Result<(), JsValue> {
    if out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to linear_regression_intensity_into_host",
        ));
    }
    unsafe {
        let out = std::slice::from_raw_parts_mut(out_ptr, data.len());
        let input = LinearRegressionIntensityInput::from_slice(
            data,
            LinearRegressionIntensityParams {
                lookback_period: Some(lookback_period),
                range_tolerance: Some(range_tolerance),
                linreg_length: Some(linreg_length),
            },
        );
        linear_regression_intensity_into_slice(out, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn linear_regression_intensity_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    lookback_period_start: usize,
    lookback_period_end: usize,
    lookback_period_step: usize,
    range_tolerance_start: f64,
    range_tolerance_end: f64,
    range_tolerance_step: f64,
    linreg_length_start: usize,
    linreg_length_end: usize,
    linreg_length_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to linear_regression_intensity_batch_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let sweep = LinearRegressionIntensityBatchRange {
            lookback_period: (
                lookback_period_start,
                lookback_period_end,
                lookback_period_step,
            ),
            range_tolerance: (
                range_tolerance_start,
                range_tolerance_end,
                range_tolerance_step,
            ),
            linreg_length: (linreg_length_start, linreg_length_end, linreg_length_step),
        };
        let combos = expand_grid_linear_regression_intensity(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let out = std::slice::from_raw_parts_mut(out_ptr, rows * len);
        linear_regression_intensity_batch_inner_into(data, &sweep, Kernel::Scalar, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn linear_regression_intensity_output_into_js(
    data: &[f64],
    lookback_period: usize,
    range_tolerance: f64,
    linreg_length: usize,
    out: &js_sys::Float64Array,
) -> Result<usize, JsValue> {
    let values =
        linear_regression_intensity_js(data, lookback_period, range_tolerance, linreg_length)?;
    crate::write_wasm_f64_output("linear_regression_intensity_output_into_js", &values, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn linear_regression_intensity_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = linear_regression_intensity_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "linear_regression_intensity_batch_output_into_js",
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
        let mut out = Vec::with_capacity(len);
        for i in 0..len {
            let trend = 100.0 + i as f64 * 0.17;
            let wave = (i as f64 * 0.19).sin() * 2.8 + (i as f64 * 0.041).cos() * 1.3;
            out.push(trend + wave);
        }
        out
    }

    fn naive_linear_regression_intensity(
        data: &[f64],
        lookback_period: usize,
        linreg_length: usize,
    ) -> Vec<f64> {
        let linreg = linreg_with_kernel(
            &LinRegInput::from_slice(
                data,
                LinRegParams {
                    period: Some(linreg_length),
                },
            ),
            Kernel::Scalar,
        )
        .expect("linreg")
        .values;
        let total = total_combinations(lookback_period);
        let mut out = vec![f64::NAN; data.len()];

        for index in 0..data.len() {
            if index + 1 < lookback_period {
                continue;
            }
            let start = index + 1 - lookback_period;
            let window = &linreg[start..=index];
            if window.iter().any(|value| !value.is_finite()) {
                continue;
            }
            let mut trend = 0i64;
            for i in 0..lookback_period.saturating_sub(1) {
                for j in (i + 1)..lookback_period {
                    if window[i] != window[j] {
                        trend += if window[j] > window[i] { 1 } else { -1 };
                    }
                }
            }
            out[index] = trend_to_intensity(trend, total);
        }
        out
    }

    fn assert_close(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            if a[i].is_nan() || b[i].is_nan() {
                assert!(
                    a[i].is_nan() && b[i].is_nan(),
                    "nan mismatch at {i}: {} vs {}",
                    a[i],
                    b[i]
                );
            } else {
                assert!(
                    (a[i] - b[i]).abs() <= 1e-10,
                    "mismatch at {i}: {} vs {}",
                    a[i],
                    b[i]
                );
            }
        }
    }

    #[test]
    fn linear_regression_intensity_matches_naive() {
        let data = sample_data(256);
        let input = LinearRegressionIntensityInput::from_slice(
            &data,
            LinearRegressionIntensityParams {
                lookback_period: Some(12),
                range_tolerance: Some(90.0),
                linreg_length: Some(30),
            },
        );
        let out = linear_regression_intensity(&input).expect("indicator");
        let expected = naive_linear_regression_intensity(&data, 12, 30);
        assert_close(&out.values, &expected);
    }

    #[test]
    fn linear_regression_intensity_into_matches_api() {
        let data = sample_data(192);
        let input = LinearRegressionIntensityInput::from_slice(
            &data,
            LinearRegressionIntensityParams {
                lookback_period: Some(10),
                range_tolerance: Some(85.0),
                linreg_length: Some(24),
            },
        );
        let baseline = linear_regression_intensity(&input).expect("baseline");
        let mut out = vec![0.0; data.len()];
        linear_regression_intensity_into(&input, &mut out).expect("into");
        assert_close(&baseline.values, &out);
    }

    #[test]
    fn linear_regression_intensity_stream_matches_batch() {
        let data = sample_data(192);
        let batch = linear_regression_intensity(&LinearRegressionIntensityInput::from_slice(
            &data,
            LinearRegressionIntensityParams {
                lookback_period: Some(12),
                range_tolerance: Some(90.0),
                linreg_length: Some(24),
            },
        ))
        .expect("batch");
        let mut stream =
            LinearRegressionIntensityStream::try_new(LinearRegressionIntensityParams {
                lookback_period: Some(12),
                range_tolerance: Some(90.0),
                linreg_length: Some(24),
            })
            .expect("stream");
        let mut values = Vec::with_capacity(data.len());
        for value in data {
            values.push(stream.update_reset_on_nan(value).unwrap_or(f64::NAN));
        }
        assert_close(&batch.values, &values);
    }

    #[test]
    fn linear_regression_intensity_batch_single_param_matches_single() {
        let data = sample_data(192);
        let sweep = LinearRegressionIntensityBatchRange {
            lookback_period: (12, 12, 0),
            range_tolerance: (90.0, 90.0, 0.0),
            linreg_length: (24, 24, 0),
        };
        let batch =
            linear_regression_intensity_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)
                .expect("batch");
        let single = linear_regression_intensity(&LinearRegressionIntensityInput::from_slice(
            &data,
            LinearRegressionIntensityParams {
                lookback_period: Some(12),
                range_tolerance: Some(90.0),
                linreg_length: Some(24),
            },
        ))
        .expect("single");
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, data.len());
        assert_close(&batch.values, &single.values);
    }

    #[test]
    fn linear_regression_intensity_rejects_invalid_tolerance() {
        let data = sample_data(64);
        let err = linear_regression_intensity(&LinearRegressionIntensityInput::from_slice(
            &data,
            LinearRegressionIntensityParams {
                lookback_period: Some(12),
                range_tolerance: Some(120.0),
                linreg_length: Some(24),
            },
        ))
        .expect_err("invalid tolerance");
        assert!(matches!(
            err,
            LinearRegressionIntensityError::InvalidRangeTolerance { .. }
        ));
    }

    #[test]
    fn linear_regression_intensity_dispatch_matches_direct() {
        let data = sample_data(192);
        let params = [
            ParamKV {
                key: "lookback_period",
                value: ParamValue::Int(12),
            },
            ParamKV {
                key: "range_tolerance",
                value: ParamValue::Float(90.0),
            },
            ParamKV {
                key: "linreg_length",
                value: ParamValue::Int(24),
            },
        ];
        let combos = [IndicatorParamSet { params: &params }];
        let out = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "linear_regression_intensity",
            output_id: Some("value"),
            data: IndicatorDataRef::Slice { values: &data },
            combos: &combos,
            kernel: Kernel::ScalarBatch,
        })
        .expect("dispatch");
        let direct = linear_regression_intensity(&LinearRegressionIntensityInput::from_slice(
            &data,
            LinearRegressionIntensityParams {
                lookback_period: Some(12),
                range_tolerance: Some(90.0),
                linreg_length: Some(24),
            },
        ))
        .expect("direct");
        assert_eq!(out.rows, 1);
        assert_eq!(out.cols, data.len());
        assert_close(out.values_f64.as_ref().expect("values"), &direct.values);
    }
}
