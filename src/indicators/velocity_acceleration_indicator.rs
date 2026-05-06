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
    alloc_uninit_f64, detect_best_batch_kernel, init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::mem::ManuallyDrop;
use thiserror::Error;

const DEFAULT_LENGTH: usize = 21;
const DEFAULT_SMOOTH_LENGTH: usize = 5;
const DEFAULT_SOURCE: &str = "hlcc4";
const DEFAULT_WMA_DENOMINATOR: f64 =
    (DEFAULT_SMOOTH_LENGTH * (DEFAULT_SMOOTH_LENGTH + 1) / 2) as f64;

impl<'a> AsRef<[f64]> for VelocityAccelerationIndicatorInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            VelocityAccelerationIndicatorData::Slice(slice) => slice,
            VelocityAccelerationIndicatorData::Candles { candles, source } => match *source {
                DEFAULT_SOURCE | "hlcc" => &candles.hlcc4,
                "open" => &candles.open,
                "high" => &candles.high,
                "low" => &candles.low,
                "close" => &candles.close,
                "volume" => &candles.volume,
                "hl2" => &candles.hl2,
                "hlc3" => &candles.hlc3,
                "ohlc4" => &candles.ohlc4,
                _ => source_type(candles, source),
            },
        }
    }
}

#[derive(Debug, Clone)]
pub enum VelocityAccelerationIndicatorData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct VelocityAccelerationIndicatorOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct VelocityAccelerationIndicatorParams {
    pub length: Option<usize>,
    pub smooth_length: Option<usize>,
}

impl Default for VelocityAccelerationIndicatorParams {
    fn default() -> Self {
        Self {
            length: Some(DEFAULT_LENGTH),
            smooth_length: Some(DEFAULT_SMOOTH_LENGTH),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VelocityAccelerationIndicatorInput<'a> {
    pub data: VelocityAccelerationIndicatorData<'a>,
    pub params: VelocityAccelerationIndicatorParams,
}

impl<'a> VelocityAccelerationIndicatorInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: VelocityAccelerationIndicatorParams,
    ) -> Self {
        Self {
            data: VelocityAccelerationIndicatorData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: VelocityAccelerationIndicatorParams) -> Self {
        Self {
            data: VelocityAccelerationIndicatorData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(
            candles,
            DEFAULT_SOURCE,
            VelocityAccelerationIndicatorParams::default(),
        )
    }

    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(DEFAULT_LENGTH)
    }

    #[inline]
    pub fn get_smooth_length(&self) -> usize {
        self.params.smooth_length.unwrap_or(DEFAULT_SMOOTH_LENGTH)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct VelocityAccelerationIndicatorBuilder {
    length: Option<usize>,
    smooth_length: Option<usize>,
    kernel: Kernel,
}

impl Default for VelocityAccelerationIndicatorBuilder {
    fn default() -> Self {
        Self {
            length: None,
            smooth_length: None,
            kernel: Kernel::Auto,
        }
    }
}

impl VelocityAccelerationIndicatorBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn length(mut self, length: usize) -> Self {
        self.length = Some(length);
        self
    }

    #[inline]
    pub fn smooth_length(mut self, smooth_length: usize) -> Self {
        self.smooth_length = Some(smooth_length);
        self
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
        source: &str,
    ) -> Result<VelocityAccelerationIndicatorOutput, VelocityAccelerationIndicatorError> {
        let input = VelocityAccelerationIndicatorInput::from_candles(
            candles,
            source,
            VelocityAccelerationIndicatorParams {
                length: self.length,
                smooth_length: self.smooth_length,
            },
        );
        velocity_acceleration_indicator_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<VelocityAccelerationIndicatorOutput, VelocityAccelerationIndicatorError> {
        let input = VelocityAccelerationIndicatorInput::from_slice(
            data,
            VelocityAccelerationIndicatorParams {
                length: self.length,
                smooth_length: self.smooth_length,
            },
        );
        velocity_acceleration_indicator_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn into_stream(
        self,
    ) -> Result<VelocityAccelerationIndicatorStream, VelocityAccelerationIndicatorError> {
        VelocityAccelerationIndicatorStream::try_new(VelocityAccelerationIndicatorParams {
            length: self.length,
            smooth_length: self.smooth_length,
        })
    }
}

#[derive(Debug, Error)]
pub enum VelocityAccelerationIndicatorError {
    #[error("velocity_acceleration_indicator: Input data slice is empty.")]
    EmptyInputData,
    #[error("velocity_acceleration_indicator: All values are NaN.")]
    AllValuesNaN,
    #[error("velocity_acceleration_indicator: Invalid length: {length}")]
    InvalidLength { length: usize },
    #[error("velocity_acceleration_indicator: Invalid smooth_length: {smooth_length}")]
    InvalidSmoothLength { smooth_length: usize },
    #[error(
        "velocity_acceleration_indicator: Not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "velocity_acceleration_indicator: Output length mismatch: expected = {expected}, got = {got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error(
        "velocity_acceleration_indicator: Invalid range: start={start}, end={end}, step={step}"
    )]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("velocity_acceleration_indicator: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Debug, Clone)]
struct LagHistory {
    values: Vec<f64>,
    next: usize,
    count: usize,
}

impl LagHistory {
    #[inline]
    fn new(size: usize) -> Self {
        Self {
            values: vec![0.0; size],
            next: 0,
            count: 0,
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.next = 0;
        self.count = 0;
    }

    #[inline]
    fn weighted_past_sum(&self) -> f64 {
        let len = self.values.len();
        let upto = self.count.min(len);
        let mut sum = 0.0;
        for lag in 1..=upto {
            let idx = if self.next >= lag {
                self.next - lag
            } else {
                len + self.next - lag
            };
            sum += self.values[idx] / lag as f64;
        }
        sum
    }

    #[inline]
    fn push(&mut self, value: f64) {
        if self.values.is_empty() {
            return;
        }
        self.values[self.next] = value;
        self.next += 1;
        if self.next == self.values.len() {
            self.next = 0;
        }
        if self.count < self.values.len() {
            self.count += 1;
        }
    }
}

#[derive(Debug, Clone)]
struct WmaState {
    values: Vec<f64>,
    next: usize,
    count: usize,
    sum: f64,
    weighted_sum: f64,
    denominator: f64,
}

impl WmaState {
    #[inline]
    fn new(length: usize) -> Self {
        let length = length.max(1);
        Self {
            values: vec![0.0; length],
            next: 0,
            count: 0,
            sum: 0.0,
            weighted_sum: 0.0,
            denominator: (length * (length + 1) / 2) as f64,
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.next = 0;
        self.count = 0;
        self.sum = 0.0;
        self.weighted_sum = 0.0;
    }

    #[inline]
    fn len(&self) -> usize {
        self.values.len()
    }

    #[inline]
    fn update(&mut self, value: f64) -> Option<f64> {
        let len = self.len();
        if len == 1 {
            self.values[0] = value;
            self.count = 1;
            self.sum = value;
            self.weighted_sum = value;
            return Some(value);
        }

        if self.count < len {
            self.values[self.next] = value;
            self.count += 1;
            self.next += 1;
            if self.next == len {
                self.next = 0;
            }
            self.sum += value;
            self.weighted_sum += self.count as f64 * value;
            if self.count < len {
                None
            } else {
                Some(self.weighted_sum / self.denominator)
            }
        } else {
            let old = self.values[self.next];
            let prev_sum = self.sum;
            self.values[self.next] = value;
            self.next += 1;
            if self.next == len {
                self.next = 0;
            }
            self.sum = prev_sum - old + value;
            self.weighted_sum = self.weighted_sum - prev_sum + len as f64 * value;
            Some(self.weighted_sum / self.denominator)
        }
    }
}

#[derive(Debug, Clone)]
pub struct VelocityAccelerationIndicatorStream {
    harmonic_sum: f64,
    inv_length: f64,
    source_history: LagHistory,
    wma: WmaState,
    acceleration_history: LagHistory,
}

impl VelocityAccelerationIndicatorStream {
    #[inline]
    pub fn try_new(
        params: VelocityAccelerationIndicatorParams,
    ) -> Result<Self, VelocityAccelerationIndicatorError> {
        let length = params.length.unwrap_or(DEFAULT_LENGTH);
        if length < 2 {
            return Err(VelocityAccelerationIndicatorError::InvalidLength { length });
        }
        let smooth_length = params.smooth_length.unwrap_or(DEFAULT_SMOOTH_LENGTH);
        if smooth_length == 0 {
            return Err(VelocityAccelerationIndicatorError::InvalidSmoothLength { smooth_length });
        }

        let mut harmonic_sum = 0.0;
        for i in 1..=length {
            harmonic_sum += 1.0 / i as f64;
        }

        Ok(Self {
            harmonic_sum,
            inv_length: 1.0 / length as f64,
            source_history: LagHistory::new(length),
            wma: WmaState::new(smooth_length),
            acceleration_history: LagHistory::new(length),
        })
    }

    #[inline]
    pub fn reset(&mut self) {
        self.source_history.reset();
        self.wma.reset();
        self.acceleration_history.reset();
    }

    #[inline]
    pub fn get_warmup_period(&self) -> usize {
        self.wma.len().saturating_sub(1)
    }

    #[inline]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        if !value.is_finite() {
            self.reset();
            return None;
        }

        let velocity = velocity_value(
            value,
            &self.source_history,
            self.harmonic_sum,
            self.inv_length,
        );
        self.source_history.push(value);

        let velocity_avg = self.wma.update(velocity)?;
        let acceleration = velocity_value(
            velocity_avg,
            &self.acceleration_history,
            self.harmonic_sum,
            self.inv_length,
        );
        self.acceleration_history.push(velocity_avg);
        Some(acceleration)
    }
}

#[inline(always)]
fn velocity_value(current: f64, history: &LagHistory, harmonic_sum: f64, inv_length: f64) -> f64 {
    (current * harmonic_sum - history.weighted_past_sum()) * inv_length
}

#[inline(always)]
fn fixed_lag_weighted_past_sum(values: &[f64; DEFAULT_LENGTH], next: usize, count: usize) -> f64 {
    let upto = count.min(DEFAULT_LENGTH);
    let mut sum = 0.0;
    let direct = upto.min(next);
    for lag in 1..=direct {
        sum += values[next - lag] / lag as f64;
    }
    for lag in direct + 1..=upto {
        sum += values[DEFAULT_LENGTH + next - lag] / lag as f64;
    }
    sum
}

#[inline(always)]
fn fixed_lag_push(
    values: &mut [f64; DEFAULT_LENGTH],
    next: &mut usize,
    count: &mut usize,
    value: f64,
) {
    values[*next] = value;
    *next += 1;
    if *next == DEFAULT_LENGTH {
        *next = 0;
    }
    if *count < DEFAULT_LENGTH {
        *count += 1;
    }
}

#[inline(always)]
fn fixed_velocity_value(
    current: f64,
    values: &[f64; DEFAULT_LENGTH],
    next: usize,
    count: usize,
    harmonic_sum: f64,
    inv_length: f64,
) -> f64 {
    (current * harmonic_sum - fixed_lag_weighted_past_sum(values, next, count)) * inv_length
}

#[inline(always)]
fn fixed_wma_update(
    values: &mut [f64; DEFAULT_SMOOTH_LENGTH],
    next: &mut usize,
    count: &mut usize,
    sum: &mut f64,
    weighted_sum: &mut f64,
    value: f64,
) -> Option<f64> {
    if *count < DEFAULT_SMOOTH_LENGTH {
        values[*next] = value;
        *count += 1;
        *next += 1;
        if *next == DEFAULT_SMOOTH_LENGTH {
            *next = 0;
        }
        *sum += value;
        *weighted_sum += *count as f64 * value;
        if *count < DEFAULT_SMOOTH_LENGTH {
            None
        } else {
            Some(*weighted_sum / DEFAULT_WMA_DENOMINATOR)
        }
    } else {
        let old = values[*next];
        let prev_sum = *sum;
        values[*next] = value;
        *next += 1;
        if *next == DEFAULT_SMOOTH_LENGTH {
            *next = 0;
        }
        *sum = prev_sum - old + value;
        *weighted_sum = *weighted_sum - prev_sum + DEFAULT_SMOOTH_LENGTH as f64 * value;
        Some(*weighted_sum / DEFAULT_WMA_DENOMINATOR)
    }
}

#[inline(always)]
fn first_valid_value(data: &[f64]) -> usize {
    let mut i = 0usize;
    while i < data.len() {
        if data[i].is_finite() {
            return i;
        }
        i += 1;
    }
    data.len()
}

#[inline(always)]
fn max_consecutive_valid_values(data: &[f64]) -> usize {
    let mut best = 0usize;
    let mut run = 0usize;
    for &value in data {
        if value.is_finite() {
            run += 1;
            if run > best {
                best = run;
            }
        } else {
            run = 0;
        }
    }
    best
}

#[inline(always)]
fn resolve_params(
    params: &VelocityAccelerationIndicatorParams,
) -> Result<(usize, usize), VelocityAccelerationIndicatorError> {
    let length = params.length.unwrap_or(DEFAULT_LENGTH);
    if length < 2 {
        return Err(VelocityAccelerationIndicatorError::InvalidLength { length });
    }
    let smooth_length = params.smooth_length.unwrap_or(DEFAULT_SMOOTH_LENGTH);
    if smooth_length == 0 {
        return Err(VelocityAccelerationIndicatorError::InvalidSmoothLength { smooth_length });
    }
    Ok((length, smooth_length))
}

#[inline(always)]
fn velocity_acceleration_indicator_prepare<'a>(
    input: &'a VelocityAccelerationIndicatorInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, usize, usize, Kernel), VelocityAccelerationIndicatorError> {
    let data = input.as_ref();
    if data.is_empty() {
        return Err(VelocityAccelerationIndicatorError::EmptyInputData);
    }

    let first = first_valid_value(data);
    if first >= data.len() {
        return Err(VelocityAccelerationIndicatorError::AllValuesNaN);
    }

    let (length, smooth_length) = resolve_params(&input.params)?;
    let valid = max_consecutive_valid_values(data);
    if valid < smooth_length {
        return Err(VelocityAccelerationIndicatorError::NotEnoughValidData {
            needed: smooth_length,
            valid,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other.to_non_batch(),
    };
    Ok((data, length, smooth_length, first, chosen))
}

#[inline(always)]
fn compute_row(data: &[f64], length: usize, smooth_length: usize, out: &mut [f64]) {
    if length == DEFAULT_LENGTH && smooth_length == DEFAULT_SMOOTH_LENGTH {
        compute_row_default(data, out);
        return;
    }

    let mut stream =
        VelocityAccelerationIndicatorStream::try_new(VelocityAccelerationIndicatorParams {
            length: Some(length),
            smooth_length: Some(smooth_length),
        })
        .unwrap();
    for (slot, &value) in out.iter_mut().zip(data.iter()) {
        *slot = stream.update(value).unwrap_or(f64::NAN);
    }
}

#[inline(always)]
fn compute_row_default(data: &[f64], out: &mut [f64]) {
    let mut harmonic_sum = 0.0;
    for i in 1..=DEFAULT_LENGTH {
        harmonic_sum += 1.0 / i as f64;
    }
    let inv_length = 1.0 / DEFAULT_LENGTH as f64;

    let mut source_history = [0.0; DEFAULT_LENGTH];
    let mut source_next = 0usize;
    let mut source_count = 0usize;
    let mut wma_values = [0.0; DEFAULT_SMOOTH_LENGTH];
    let mut wma_next = 0usize;
    let mut wma_count = 0usize;
    let mut wma_sum = 0.0;
    let mut wma_weighted_sum = 0.0;
    let mut acceleration_history = [0.0; DEFAULT_LENGTH];
    let mut acceleration_next = 0usize;
    let mut acceleration_count = 0usize;

    for (slot, &value) in out.iter_mut().zip(data.iter()) {
        if !value.is_finite() {
            source_next = 0;
            source_count = 0;
            wma_next = 0;
            wma_count = 0;
            wma_sum = 0.0;
            wma_weighted_sum = 0.0;
            acceleration_next = 0;
            acceleration_count = 0;
            *slot = f64::NAN;
            continue;
        }

        let velocity = fixed_velocity_value(
            value,
            &source_history,
            source_next,
            source_count,
            harmonic_sum,
            inv_length,
        );
        fixed_lag_push(
            &mut source_history,
            &mut source_next,
            &mut source_count,
            value,
        );

        if let Some(velocity_avg) = fixed_wma_update(
            &mut wma_values,
            &mut wma_next,
            &mut wma_count,
            &mut wma_sum,
            &mut wma_weighted_sum,
            velocity,
        ) {
            let acceleration = fixed_velocity_value(
                velocity_avg,
                &acceleration_history,
                acceleration_next,
                acceleration_count,
                harmonic_sum,
                inv_length,
            );
            fixed_lag_push(
                &mut acceleration_history,
                &mut acceleration_next,
                &mut acceleration_count,
                velocity_avg,
            );
            *slot = acceleration;
        } else {
            *slot = f64::NAN;
        }
    }
}

#[inline]
pub fn velocity_acceleration_indicator(
    input: &VelocityAccelerationIndicatorInput,
) -> Result<VelocityAccelerationIndicatorOutput, VelocityAccelerationIndicatorError> {
    velocity_acceleration_indicator_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn velocity_acceleration_indicator_with_kernel(
    input: &VelocityAccelerationIndicatorInput,
    kernel: Kernel,
) -> Result<VelocityAccelerationIndicatorOutput, VelocityAccelerationIndicatorError> {
    let (data, length, smooth_length, first, _chosen) =
        velocity_acceleration_indicator_prepare(input, kernel)?;
    let _ = first;
    let mut values = alloc_uninit_f64(data.len());
    compute_row(data, length, smooth_length, &mut values);
    Ok(VelocityAccelerationIndicatorOutput { values })
}

#[inline]
pub fn velocity_acceleration_indicator_into_slice(
    out: &mut [f64],
    input: &VelocityAccelerationIndicatorInput,
    kernel: Kernel,
) -> Result<(), VelocityAccelerationIndicatorError> {
    let (data, length, smooth_length, _first, _chosen) =
        velocity_acceleration_indicator_prepare(input, kernel)?;
    if out.len() != data.len() {
        return Err(VelocityAccelerationIndicatorError::OutputLengthMismatch {
            expected: data.len(),
            got: out.len(),
        });
    }
    compute_row(data, length, smooth_length, out);
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn velocity_acceleration_indicator_into(
    input: &VelocityAccelerationIndicatorInput,
    out: &mut [f64],
) -> Result<(), VelocityAccelerationIndicatorError> {
    velocity_acceleration_indicator_into_slice(out, input, Kernel::Auto)
}

#[derive(Clone, Debug)]
pub struct VelocityAccelerationIndicatorBatchRange {
    pub length: (usize, usize, usize),
    pub smooth_length: (usize, usize, usize),
}

impl Default for VelocityAccelerationIndicatorBatchRange {
    fn default() -> Self {
        Self {
            length: (DEFAULT_LENGTH, DEFAULT_LENGTH, 0),
            smooth_length: (DEFAULT_SMOOTH_LENGTH, DEFAULT_SMOOTH_LENGTH, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct VelocityAccelerationIndicatorBatchBuilder {
    range: VelocityAccelerationIndicatorBatchRange,
    kernel: Kernel,
}

impl VelocityAccelerationIndicatorBatchBuilder {
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
    pub fn length_static(mut self, length: usize) -> Self {
        self.range.length = (length, length, 0);
        self
    }

    #[inline]
    pub fn smooth_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.smooth_length = (start, end, step);
        self
    }

    #[inline]
    pub fn smooth_length_static(mut self, smooth_length: usize) -> Self {
        self.range.smooth_length = (smooth_length, smooth_length, 0);
        self
    }

    #[inline]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<VelocityAccelerationIndicatorBatchOutput, VelocityAccelerationIndicatorError> {
        velocity_acceleration_indicator_batch_with_kernel(data, &self.range, self.kernel)
    }

    #[inline]
    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<VelocityAccelerationIndicatorBatchOutput, VelocityAccelerationIndicatorError> {
        self.apply_slice(source_type(candles, source))
    }
}

#[derive(Clone, Debug)]
pub struct VelocityAccelerationIndicatorBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<VelocityAccelerationIndicatorParams>,
    pub rows: usize,
    pub cols: usize,
}

impl VelocityAccelerationIndicatorBatchOutput {
    #[inline]
    pub fn row_for_params(&self, params: &VelocityAccelerationIndicatorParams) -> Option<usize> {
        self.combos.iter().position(|combo| {
            combo.length.unwrap_or(DEFAULT_LENGTH) == params.length.unwrap_or(DEFAULT_LENGTH)
                && combo.smooth_length.unwrap_or(DEFAULT_SMOOTH_LENGTH)
                    == params.smooth_length.unwrap_or(DEFAULT_SMOOTH_LENGTH)
        })
    }

    #[inline]
    pub fn values_for(&self, params: &VelocityAccelerationIndicatorParams) -> Option<&[f64]> {
        self.row_for_params(params).and_then(|row| {
            row.checked_mul(self.cols)
                .and_then(|start| self.values.get(start..start + self.cols))
        })
    }
}

#[inline(always)]
fn expand_axis_usize(
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, VelocityAccelerationIndicatorError> {
    if step == 0 || start == end {
        return Ok(vec![start]);
    }

    let mut out = Vec::new();
    if start < end {
        let mut x = start;
        while x <= end {
            out.push(x);
            let next = x.saturating_add(step);
            if next == x {
                break;
            }
            x = next;
        }
    } else {
        let mut x = start;
        loop {
            out.push(x);
            if x == end {
                break;
            }
            let next = x.saturating_sub(step);
            if next == x || next < end {
                break;
            }
            x = next;
        }
    }

    if out.is_empty() {
        return Err(VelocityAccelerationIndicatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

#[inline(always)]
fn expand_grid_velocity_acceleration_indicator(
    sweep: &VelocityAccelerationIndicatorBatchRange,
) -> Result<Vec<VelocityAccelerationIndicatorParams>, VelocityAccelerationIndicatorError> {
    let lengths = expand_axis_usize(sweep.length)?;
    let smooth_lengths = expand_axis_usize(sweep.smooth_length)?;

    let mut combos = Vec::with_capacity(lengths.len() * smooth_lengths.len());
    for length in lengths {
        for smooth_length in smooth_lengths.iter().copied() {
            let combo = VelocityAccelerationIndicatorParams {
                length: Some(length),
                smooth_length: Some(smooth_length),
            };
            let _ = resolve_params(&combo)?;
            combos.push(combo);
        }
    }
    Ok(combos)
}

#[inline]
pub fn velocity_acceleration_indicator_batch_with_kernel(
    data: &[f64],
    sweep: &VelocityAccelerationIndicatorBatchRange,
    kernel: Kernel,
) -> Result<VelocityAccelerationIndicatorBatchOutput, VelocityAccelerationIndicatorError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => {
            return Err(VelocityAccelerationIndicatorError::InvalidKernelForBatch(
                other,
            ))
        }
    };
    velocity_acceleration_indicator_batch_par_slice(data, sweep, batch_kernel.to_non_batch())
}

#[inline]
pub fn velocity_acceleration_indicator_batch_slice(
    data: &[f64],
    sweep: &VelocityAccelerationIndicatorBatchRange,
    kernel: Kernel,
) -> Result<VelocityAccelerationIndicatorBatchOutput, VelocityAccelerationIndicatorError> {
    velocity_acceleration_indicator_batch_inner(data, sweep, kernel, false)
}

#[inline]
pub fn velocity_acceleration_indicator_batch_par_slice(
    data: &[f64],
    sweep: &VelocityAccelerationIndicatorBatchRange,
    kernel: Kernel,
) -> Result<VelocityAccelerationIndicatorBatchOutput, VelocityAccelerationIndicatorError> {
    velocity_acceleration_indicator_batch_inner(data, sweep, kernel, true)
}

#[inline(always)]
pub fn velocity_acceleration_indicator_batch_inner(
    data: &[f64],
    sweep: &VelocityAccelerationIndicatorBatchRange,
    _kernel: Kernel,
    parallel: bool,
) -> Result<VelocityAccelerationIndicatorBatchOutput, VelocityAccelerationIndicatorError> {
    let combos = expand_grid_velocity_acceleration_indicator(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    if cols == 0 {
        return Err(VelocityAccelerationIndicatorError::EmptyInputData);
    }

    let first = first_valid_value(data);
    if first >= cols {
        return Err(VelocityAccelerationIndicatorError::AllValuesNaN);
    }

    let valid = max_consecutive_valid_values(data);
    let max_smooth_length = combos
        .iter()
        .map(|combo| combo.smooth_length.unwrap_or(DEFAULT_SMOOTH_LENGTH))
        .max()
        .unwrap_or(0);
    if valid < max_smooth_length {
        return Err(VelocityAccelerationIndicatorError::NotEnoughValidData {
            needed: max_smooth_length,
            valid,
        });
    }

    let mut buf_mu = make_uninit_matrix(rows, cols);
    let warmups: Vec<usize> = combos
        .iter()
        .map(|combo| {
            first.saturating_add(
                combo
                    .smooth_length
                    .unwrap_or(DEFAULT_SMOOTH_LENGTH)
                    .saturating_sub(1),
            )
        })
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warmups);

    let mut guard = ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { std::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out.par_chunks_mut(cols)
            .enumerate()
            .for_each(|(row, out_row)| {
                let combo = &combos[row];
                compute_row(
                    data,
                    combo.length.unwrap_or(DEFAULT_LENGTH),
                    combo.smooth_length.unwrap_or(DEFAULT_SMOOTH_LENGTH),
                    out_row,
                );
            });

        #[cfg(target_arch = "wasm32")]
        for (row, out_row) in out.chunks_mut(cols).enumerate() {
            let combo = &combos[row];
            compute_row(
                data,
                combo.length.unwrap_or(DEFAULT_LENGTH),
                combo.smooth_length.unwrap_or(DEFAULT_SMOOTH_LENGTH),
                out_row,
            );
        }
    } else {
        for (row, out_row) in out.chunks_mut(cols).enumerate() {
            let combo = &combos[row];
            compute_row(
                data,
                combo.length.unwrap_or(DEFAULT_LENGTH),
                combo.smooth_length.unwrap_or(DEFAULT_SMOOTH_LENGTH),
                out_row,
            );
        }
    }

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(VelocityAccelerationIndicatorBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub fn velocity_acceleration_indicator_batch_inner_into(
    data: &[f64],
    sweep: &VelocityAccelerationIndicatorBatchRange,
    _kernel: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<VelocityAccelerationIndicatorParams>, VelocityAccelerationIndicatorError> {
    let combos = expand_grid_velocity_acceleration_indicator(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    if cols == 0 {
        return Err(VelocityAccelerationIndicatorError::EmptyInputData);
    }

    let total = rows.checked_mul(cols).ok_or_else(|| {
        VelocityAccelerationIndicatorError::OutputLengthMismatch {
            expected: usize::MAX,
            got: out.len(),
        }
    })?;
    if out.len() != total {
        return Err(VelocityAccelerationIndicatorError::OutputLengthMismatch {
            expected: total,
            got: out.len(),
        });
    }

    let first = first_valid_value(data);
    if first >= cols {
        return Err(VelocityAccelerationIndicatorError::AllValuesNaN);
    }

    let valid = max_consecutive_valid_values(data);
    let max_smooth_length = combos
        .iter()
        .map(|combo| combo.smooth_length.unwrap_or(DEFAULT_SMOOTH_LENGTH))
        .max()
        .unwrap_or(0);
    if valid < max_smooth_length {
        return Err(VelocityAccelerationIndicatorError::NotEnoughValidData {
            needed: max_smooth_length,
            valid,
        });
    }

    let warmups: Vec<usize> = combos
        .iter()
        .map(|combo| {
            first.saturating_add(
                combo
                    .smooth_length
                    .unwrap_or(DEFAULT_SMOOTH_LENGTH)
                    .saturating_sub(1),
            )
        })
        .collect();

    for (row, out_row) in out.chunks_mut(cols).enumerate() {
        let warmup = warmups[row].min(cols);
        out_row[..warmup].fill(f64::NAN);
    }

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out.par_chunks_mut(cols)
            .enumerate()
            .for_each(|(row, out_row)| {
                let combo = &combos[row];
                compute_row(
                    data,
                    combo.length.unwrap_or(DEFAULT_LENGTH),
                    combo.smooth_length.unwrap_or(DEFAULT_SMOOTH_LENGTH),
                    out_row,
                );
            });

        #[cfg(target_arch = "wasm32")]
        for (row, out_row) in out.chunks_mut(cols).enumerate() {
            let combo = &combos[row];
            compute_row(
                data,
                combo.length.unwrap_or(DEFAULT_LENGTH),
                combo.smooth_length.unwrap_or(DEFAULT_SMOOTH_LENGTH),
                out_row,
            );
        }
    } else {
        for (row, out_row) in out.chunks_mut(cols).enumerate() {
            let combo = &combos[row];
            compute_row(
                data,
                combo.length.unwrap_or(DEFAULT_LENGTH),
                combo.smooth_length.unwrap_or(DEFAULT_SMOOTH_LENGTH),
                out_row,
            );
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "velocity_acceleration_indicator")]
#[pyo3(signature = (data, length=DEFAULT_LENGTH, smooth_length=DEFAULT_SMOOTH_LENGTH, kernel=None))]
pub fn velocity_acceleration_indicator_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length: usize,
    smooth_length: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let slice = data.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = VelocityAccelerationIndicatorInput::from_slice(
        slice,
        VelocityAccelerationIndicatorParams {
            length: Some(length),
            smooth_length: Some(smooth_length),
        },
    );
    let output = py
        .allow_threads(|| velocity_acceleration_indicator_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(output.values.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "VelocityAccelerationIndicatorStream")]
pub struct VelocityAccelerationIndicatorStreamPy {
    inner: VelocityAccelerationIndicatorStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl VelocityAccelerationIndicatorStreamPy {
    #[new]
    #[pyo3(signature = (length=DEFAULT_LENGTH, smooth_length=DEFAULT_SMOOTH_LENGTH))]
    fn new(length: usize, smooth_length: usize) -> PyResult<Self> {
        let inner =
            VelocityAccelerationIndicatorStream::try_new(VelocityAccelerationIndicatorParams {
                length: Some(length),
                smooth_length: Some(smooth_length),
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        self.inner.update(value)
    }

    #[getter]
    fn warmup_period(&self) -> usize {
        self.inner.get_warmup_period()
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "velocity_acceleration_indicator_batch")]
#[pyo3(signature = (data, length_range=(DEFAULT_LENGTH, DEFAULT_LENGTH, 0), smooth_length_range=(DEFAULT_SMOOTH_LENGTH, DEFAULT_SMOOTH_LENGTH, 0), kernel=None))]
pub fn velocity_acceleration_indicator_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    smooth_length_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let slice = data.as_slice()?;
    let kernel = validate_kernel(kernel, true)?;
    let sweep = VelocityAccelerationIndicatorBatchRange {
        length: length_range,
        smooth_length: smooth_length_range,
    };
    let combos = expand_grid_velocity_acceleration_indicator(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = slice.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let out_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_slice = unsafe { out_arr.as_slice_mut()? };

    let combos = py
        .allow_threads(|| {
            let batch = match kernel {
                Kernel::Auto => detect_best_batch_kernel(),
                other => other,
            };
            velocity_acceleration_indicator_batch_inner_into(
                slice,
                &sweep,
                batch.to_non_batch(),
                true,
                out_slice,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "lengths",
        combos
            .iter()
            .map(|combo| combo.length.unwrap_or(DEFAULT_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "smooth_lengths",
        combos
            .iter()
            .map(|combo| combo.smooth_length.unwrap_or(DEFAULT_SMOOTH_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_velocity_acceleration_indicator_module(
    module: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(
        velocity_acceleration_indicator_py,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        velocity_acceleration_indicator_batch_py,
        module
    )?)?;
    module.add_class::<VelocityAccelerationIndicatorStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "velocity_acceleration_indicator_js")]
pub fn velocity_acceleration_indicator_js(
    data: &[f64],
    length: usize,
    smooth_length: usize,
) -> Result<Vec<f64>, JsValue> {
    let input = VelocityAccelerationIndicatorInput::from_slice(
        data,
        VelocityAccelerationIndicatorParams {
            length: Some(length),
            smooth_length: Some(smooth_length),
        },
    );
    let mut output = vec![0.0; data.len()];
    velocity_acceleration_indicator_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(output)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn velocity_acceleration_indicator_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn velocity_acceleration_indicator_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn velocity_acceleration_indicator_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length: usize,
    smooth_length: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let input = VelocityAccelerationIndicatorInput::from_slice(
            data,
            VelocityAccelerationIndicatorParams {
                length: Some(length),
                smooth_length: Some(smooth_length),
            },
        );
        if in_ptr == out_ptr {
            let mut tmp = vec![0.0; len];
            velocity_acceleration_indicator_into_slice(&mut tmp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            std::slice::from_raw_parts_mut(out_ptr, len).copy_from_slice(&tmp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            velocity_acceleration_indicator_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct VelocityAccelerationIndicatorBatchConfig {
    pub length_range: (usize, usize, usize),
    pub smooth_length_range: Option<(usize, usize, usize)>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct VelocityAccelerationIndicatorBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<VelocityAccelerationIndicatorParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "velocity_acceleration_indicator_batch_js")]
pub fn velocity_acceleration_indicator_batch_js(
    data: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: VelocityAccelerationIndicatorBatchConfig =
        serde_wasm_bindgen::from_value(config)
            .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let sweep = VelocityAccelerationIndicatorBatchRange {
        length: config.length_range,
        smooth_length: config.smooth_length_range.unwrap_or((
            DEFAULT_SMOOTH_LENGTH,
            DEFAULT_SMOOTH_LENGTH,
            0,
        )),
    };
    let output = velocity_acceleration_indicator_batch_with_kernel(data, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&VelocityAccelerationIndicatorBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn velocity_acceleration_indicator_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length_start: usize,
    length_end: usize,
    length_step: usize,
    smooth_length_start: usize,
    smooth_length_end: usize,
    smooth_length_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let sweep = VelocityAccelerationIndicatorBatchRange {
            length: (length_start, length_end, length_step),
            smooth_length: (smooth_length_start, smooth_length_end, smooth_length_step),
        };
        let combos = expand_grid_velocity_acceleration_indicator(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let total = rows
            .checked_mul(len)
            .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        velocity_acceleration_indicator_batch_inner_into(data, &sweep, Kernel::Auto, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn velocity_acceleration_indicator_output_into_js(
    data: &[f64],
    length: usize,
    smooth_length: usize,
    out: &js_sys::Float64Array,
) -> Result<usize, JsValue> {
    let values = velocity_acceleration_indicator_js(data, length, smooth_length)?;
    crate::write_wasm_f64_output(
        "velocity_acceleration_indicator_output_into_js",
        &values,
        out,
    )
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn velocity_acceleration_indicator_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = velocity_acceleration_indicator_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "velocity_acceleration_indicator_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data(length: usize) -> Vec<f64> {
        (0..length)
            .map(|i| {
                let x = i as f64;
                100.0 + x * 0.05 + (x * 0.13).sin() * 2.5 + (x * 0.04).cos() * 0.8
            })
            .collect()
    }

    fn sample_candles(length: usize) -> Candles {
        let mut open = Vec::with_capacity(length);
        let mut high = Vec::with_capacity(length);
        let mut low = Vec::with_capacity(length);
        let mut close = Vec::with_capacity(length);
        for i in 0..length {
            let x = i as f64;
            let o = 100.0 + x * 0.04 + (x * 0.09).sin();
            let c = o + (x * 0.11).cos() * 0.9;
            let h = o.max(c) + 0.6 + (x * 0.03).sin().abs() * 0.2;
            let l = o.min(c) - 0.6 - (x * 0.05).cos().abs() * 0.2;
            open.push(o);
            high.push(h);
            low.push(l);
            close.push(c);
        }
        Candles::new(
            (0..length as i64).collect(),
            open,
            high,
            low,
            close,
            vec![1_000.0; length],
        )
    }

    fn assert_series_eq(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (&a, &e) in actual.iter().zip(expected.iter()) {
            assert!(
                (a.is_nan() && e.is_nan()) || (a - e).abs() <= 1e-12,
                "expected {e:?}, got {a:?}"
            );
        }
    }

    #[test]
    fn velocity_acceleration_indicator_output_contract() {
        let data = sample_data(256);
        let input = VelocityAccelerationIndicatorInput::from_slice(
            &data,
            VelocityAccelerationIndicatorParams::default(),
        );
        let out = velocity_acceleration_indicator(&input).unwrap();
        assert_eq!(out.values.len(), data.len());
        assert_eq!(
            out.values.iter().position(|v| v.is_finite()).unwrap(),
            DEFAULT_SMOOTH_LENGTH - 1
        );
        assert!(out.values.last().unwrap().is_finite());
    }

    #[test]
    fn velocity_acceleration_indicator_length_can_exceed_data_len() {
        let data = sample_data(8);
        let input = VelocityAccelerationIndicatorInput::from_slice(
            &data,
            VelocityAccelerationIndicatorParams {
                length: Some(21),
                smooth_length: Some(1),
            },
        );
        let out = velocity_acceleration_indicator(&input).unwrap();
        assert!(out.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn velocity_acceleration_indicator_rejects_invalid_parameters() {
        let data = sample_data(32);
        let err = velocity_acceleration_indicator(&VelocityAccelerationIndicatorInput::from_slice(
            &data,
            VelocityAccelerationIndicatorParams {
                length: Some(1),
                smooth_length: Some(5),
            },
        ))
        .unwrap_err();
        assert!(matches!(
            err,
            VelocityAccelerationIndicatorError::InvalidLength { .. }
        ));

        let err = velocity_acceleration_indicator(&VelocityAccelerationIndicatorInput::from_slice(
            &data,
            VelocityAccelerationIndicatorParams {
                length: Some(21),
                smooth_length: Some(0),
            },
        ))
        .unwrap_err();
        assert!(matches!(
            err,
            VelocityAccelerationIndicatorError::InvalidSmoothLength { .. }
        ));
    }

    #[test]
    fn velocity_acceleration_indicator_builder_supports_candles() {
        let candles = sample_candles(160);
        let output = VelocityAccelerationIndicatorBuilder::new()
            .length(21)
            .smooth_length(5)
            .apply(&candles, "hlcc4")
            .unwrap();
        assert_eq!(output.values.len(), candles.close.len());
        assert!(output.values.last().unwrap().is_finite());
    }

    #[test]
    fn velocity_acceleration_indicator_stream_matches_batch_with_reset() {
        let mut data = sample_data(180);
        data[90] = f64::NAN;

        let input = VelocityAccelerationIndicatorInput::from_slice(
            &data,
            VelocityAccelerationIndicatorParams::default(),
        );
        let batch = velocity_acceleration_indicator(&input).unwrap();
        let mut stream = VelocityAccelerationIndicatorStream::try_new(
            VelocityAccelerationIndicatorParams::default(),
        )
        .unwrap();
        let mut streamed = Vec::with_capacity(data.len());
        for &value in &data {
            streamed.push(stream.update(value).unwrap_or(f64::NAN));
        }
        assert_series_eq(&batch.values, &streamed);
    }

    #[test]
    fn velocity_acceleration_indicator_into_matches_api() {
        let data = sample_data(192);
        let input = VelocityAccelerationIndicatorInput::from_slice(
            &data,
            VelocityAccelerationIndicatorParams::default(),
        );
        let direct = velocity_acceleration_indicator(&input).unwrap();
        let mut out = vec![0.0; data.len()];
        velocity_acceleration_indicator_into(&input, &mut out).unwrap();
        assert_series_eq(&direct.values, &out);
    }

    #[test]
    fn velocity_acceleration_indicator_batch_single_param_matches_single() {
        let data = sample_data(192);
        let batch = velocity_acceleration_indicator_batch_with_kernel(
            &data,
            &VelocityAccelerationIndicatorBatchRange::default(),
            Kernel::ScalarBatch,
        )
        .unwrap();
        let single =
            velocity_acceleration_indicator(&VelocityAccelerationIndicatorInput::from_slice(
                &data,
                VelocityAccelerationIndicatorParams::default(),
            ))
            .unwrap();
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, data.len());
        assert_series_eq(&batch.values[..data.len()], &single.values);
    }

    #[test]
    fn velocity_acceleration_indicator_batch_metadata() {
        let data = sample_data(128);
        let batch = velocity_acceleration_indicator_batch_with_kernel(
            &data,
            &VelocityAccelerationIndicatorBatchRange {
                length: (21, 23, 2),
                smooth_length: (4, 5, 1),
            },
            Kernel::ScalarBatch,
        )
        .unwrap();
        assert_eq!(batch.rows, 4);
        assert_eq!(batch.cols, data.len());
        assert_eq!(batch.combos.len(), 4);
    }
}
