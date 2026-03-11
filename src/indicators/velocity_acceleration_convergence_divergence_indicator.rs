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
use std::error::Error;
use thiserror::Error;

impl<'a> AsRef<[f64]> for VelocityAccelerationConvergenceDivergenceIndicatorInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            VelocityAccelerationConvergenceDivergenceIndicatorData::Slice(slice) => slice,
            VelocityAccelerationConvergenceDivergenceIndicatorData::Candles { candles, source } => {
                source_type(candles, source)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum VelocityAccelerationConvergenceDivergenceIndicatorData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct VelocityAccelerationConvergenceDivergenceIndicatorOutput {
    pub vacd: Vec<f64>,
    pub signal: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct VelocityAccelerationConvergenceDivergenceIndicatorParams {
    pub length: Option<usize>,
    pub smooth_length: Option<usize>,
}

impl Default for VelocityAccelerationConvergenceDivergenceIndicatorParams {
    fn default() -> Self {
        Self {
            length: Some(21),
            smooth_length: Some(5),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VelocityAccelerationConvergenceDivergenceIndicatorInput<'a> {
    pub data: VelocityAccelerationConvergenceDivergenceIndicatorData<'a>,
    pub params: VelocityAccelerationConvergenceDivergenceIndicatorParams,
}

impl<'a> VelocityAccelerationConvergenceDivergenceIndicatorInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: VelocityAccelerationConvergenceDivergenceIndicatorParams,
    ) -> Self {
        Self {
            data: VelocityAccelerationConvergenceDivergenceIndicatorData::Candles {
                candles,
                source,
            },
            params,
        }
    }

    #[inline]
    pub fn from_slice(
        slice: &'a [f64],
        params: VelocityAccelerationConvergenceDivergenceIndicatorParams,
    ) -> Self {
        Self {
            data: VelocityAccelerationConvergenceDivergenceIndicatorData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(
            candles,
            "hlcc4",
            VelocityAccelerationConvergenceDivergenceIndicatorParams::default(),
        )
    }

    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(21)
    }

    #[inline]
    pub fn get_smooth_length(&self) -> usize {
        self.params.smooth_length.unwrap_or(5)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct VelocityAccelerationConvergenceDivergenceIndicatorBuilder {
    length: Option<usize>,
    smooth_length: Option<usize>,
    kernel: Kernel,
}

impl Default for VelocityAccelerationConvergenceDivergenceIndicatorBuilder {
    fn default() -> Self {
        Self {
            length: None,
            smooth_length: None,
            kernel: Kernel::Auto,
        }
    }
}

impl VelocityAccelerationConvergenceDivergenceIndicatorBuilder {
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
    ) -> Result<
        VelocityAccelerationConvergenceDivergenceIndicatorOutput,
        VelocityAccelerationConvergenceDivergenceIndicatorError,
    > {
        let params = VelocityAccelerationConvergenceDivergenceIndicatorParams {
            length: self.length,
            smooth_length: self.smooth_length,
        };
        velocity_acceleration_convergence_divergence_indicator_with_kernel(
            &VelocityAccelerationConvergenceDivergenceIndicatorInput::from_candles(
                candles, "hlcc4", params,
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
        VelocityAccelerationConvergenceDivergenceIndicatorOutput,
        VelocityAccelerationConvergenceDivergenceIndicatorError,
    > {
        let params = VelocityAccelerationConvergenceDivergenceIndicatorParams {
            length: self.length,
            smooth_length: self.smooth_length,
        };
        velocity_acceleration_convergence_divergence_indicator_with_kernel(
            &VelocityAccelerationConvergenceDivergenceIndicatorInput::from_candles(
                candles, source, params,
            ),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<
        VelocityAccelerationConvergenceDivergenceIndicatorOutput,
        VelocityAccelerationConvergenceDivergenceIndicatorError,
    > {
        let params = VelocityAccelerationConvergenceDivergenceIndicatorParams {
            length: self.length,
            smooth_length: self.smooth_length,
        };
        velocity_acceleration_convergence_divergence_indicator_with_kernel(
            &VelocityAccelerationConvergenceDivergenceIndicatorInput::from_slice(data, params),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<
        VelocityAccelerationConvergenceDivergenceIndicatorStream,
        VelocityAccelerationConvergenceDivergenceIndicatorError,
    > {
        VelocityAccelerationConvergenceDivergenceIndicatorStream::try_new(
            VelocityAccelerationConvergenceDivergenceIndicatorParams {
                length: self.length,
                smooth_length: self.smooth_length,
            },
        )
    }
}

#[derive(Debug, Error)]
pub enum VelocityAccelerationConvergenceDivergenceIndicatorError {
    #[error("velocity_acceleration_convergence_divergence_indicator: Input data slice is empty.")]
    EmptyInputData,
    #[error("velocity_acceleration_convergence_divergence_indicator: All values are NaN.")]
    AllValuesNaN,
    #[error("velocity_acceleration_convergence_divergence_indicator: Invalid length: {length}")]
    InvalidLength { length: usize },
    #[error(
        "velocity_acceleration_convergence_divergence_indicator: Invalid smooth_length: {smooth_length}"
    )]
    InvalidSmoothLength { smooth_length: usize },
    #[error(
        "velocity_acceleration_convergence_divergence_indicator: Not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "velocity_acceleration_convergence_divergence_indicator: Output length mismatch: expected = {expected}, got = {got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error(
        "velocity_acceleration_convergence_divergence_indicator: Invalid range: start={start}, end={end}, step={step}"
    )]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error(
        "velocity_acceleration_convergence_divergence_indicator: Invalid kernel for batch: {0:?}"
    )]
    InvalidKernelForBatch(Kernel),
    #[error(
        "velocity_acceleration_convergence_divergence_indicator: Output length mismatch: dst = {dst_len}, expected = {expected_len}"
    )]
    MismatchedOutputLen { dst_len: usize, expected_len: usize },
    #[error("velocity_acceleration_convergence_divergence_indicator: Invalid input: {msg}")]
    InvalidInput { msg: String },
}

#[inline(always)]
fn validate_length(
    length: usize,
) -> Result<(), VelocityAccelerationConvergenceDivergenceIndicatorError> {
    if length < 2 {
        return Err(
            VelocityAccelerationConvergenceDivergenceIndicatorError::InvalidLength { length },
        );
    }
    Ok(())
}

#[inline(always)]
fn validate_smooth_length(
    smooth_length: usize,
) -> Result<(), VelocityAccelerationConvergenceDivergenceIndicatorError> {
    if smooth_length == 0 {
        return Err(
            VelocityAccelerationConvergenceDivergenceIndicatorError::InvalidSmoothLength {
                smooth_length,
            },
        );
    }
    Ok(())
}

#[inline(always)]
fn longest_valid_run(data: &[f64]) -> usize {
    let mut best = 0usize;
    let mut cur = 0usize;
    for &value in data {
        if value.is_finite() {
            cur += 1;
            best = best.max(cur);
        } else {
            cur = 0;
        }
    }
    best
}

#[inline(always)]
fn validate_common(
    data: &[f64],
    length: usize,
    smooth_length: usize,
) -> Result<(), VelocityAccelerationConvergenceDivergenceIndicatorError> {
    if data.is_empty() {
        return Err(VelocityAccelerationConvergenceDivergenceIndicatorError::EmptyInputData);
    }
    validate_length(length)?;
    validate_smooth_length(smooth_length)?;
    let valid = longest_valid_run(data);
    if valid == 0 {
        return Err(VelocityAccelerationConvergenceDivergenceIndicatorError::AllValuesNaN);
    }
    if valid < smooth_length {
        return Err(
            VelocityAccelerationConvergenceDivergenceIndicatorError::NotEnoughValidData {
                needed: smooth_length,
                valid,
            },
        );
    }
    Ok(())
}

#[inline(always)]
fn warmup_prefix(smooth_length: usize) -> usize {
    smooth_length.saturating_sub(1)
}

#[inline(always)]
fn classify_signal(vacd: f64, prev_vacd_nz: f64) -> f64 {
    if vacd > 0.0 {
        if vacd > prev_vacd_nz {
            2.0
        } else {
            1.0
        }
    } else if vacd < 0.0 {
        if vacd < prev_vacd_nz {
            -2.0
        } else {
            -1.0
        }
    } else {
        0.0
    }
}

#[inline(always)]
fn compute_velocity_current(history: &[f64], current: f64, length: usize) -> f64 {
    let mut sum = 0.0;
    for i in 1..=length {
        let prev = if history.len() >= i {
            history[history.len() - i]
        } else {
            0.0
        };
        sum += (current - prev) / i as f64;
    }
    sum / length as f64
}

#[inline(always)]
fn compute_wma_tail(history: &[f64], period: usize) -> f64 {
    let start = history.len() - period;
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for (offset, &value) in history[start..].iter().enumerate() {
        let weight = (offset + 1) as f64;
        numerator += value * weight;
        denominator += weight;
    }
    numerator / denominator
}

#[derive(Debug, Clone)]
pub struct VelocityAccelerationConvergenceDivergenceIndicatorStream {
    length: usize,
    smooth_length: usize,
    source_history: Vec<f64>,
    raw_velocity_history: Vec<f64>,
    velocity_avg_history: Vec<f64>,
    prev_vacd: f64,
    has_prev_vacd: bool,
}

impl VelocityAccelerationConvergenceDivergenceIndicatorStream {
    #[inline(always)]
    pub fn try_new(
        params: VelocityAccelerationConvergenceDivergenceIndicatorParams,
    ) -> Result<Self, VelocityAccelerationConvergenceDivergenceIndicatorError> {
        let length = params.length.unwrap_or(21);
        let smooth_length = params.smooth_length.unwrap_or(5);
        validate_length(length)?;
        validate_smooth_length(smooth_length)?;
        Ok(Self {
            length,
            smooth_length,
            source_history: Vec::new(),
            raw_velocity_history: Vec::new(),
            velocity_avg_history: Vec::new(),
            prev_vacd: f64::NAN,
            has_prev_vacd: false,
        })
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.source_history.clear();
        self.raw_velocity_history.clear();
        self.velocity_avg_history.clear();
        self.prev_vacd = f64::NAN;
        self.has_prev_vacd = false;
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        if !value.is_finite() {
            self.reset();
            return None;
        }

        let raw_velocity = compute_velocity_current(&self.source_history, value, self.length);
        self.source_history.push(value);
        self.raw_velocity_history.push(raw_velocity);

        if self.raw_velocity_history.len() < self.smooth_length {
            return None;
        }

        let velocity_avg = compute_wma_tail(&self.raw_velocity_history, self.smooth_length);
        let acceleration =
            compute_velocity_current(&self.velocity_avg_history, velocity_avg, self.length);
        let vacd = velocity_avg - acceleration;
        let signal = classify_signal(
            vacd,
            if self.has_prev_vacd {
                self.prev_vacd
            } else {
                0.0
            },
        );
        self.velocity_avg_history.push(velocity_avg);
        self.prev_vacd = vacd;
        self.has_prev_vacd = true;
        Some((vacd, signal))
    }

    #[inline(always)]
    pub fn get_warmup_period(&self) -> usize {
        warmup_prefix(self.smooth_length)
    }
}

#[inline(always)]
fn compute_row(
    data: &[f64],
    length: usize,
    smooth_length: usize,
    vacd: &mut [f64],
    signal: &mut [f64],
) {
    vacd.fill(f64::NAN);
    signal.fill(f64::NAN);
    let mut stream = VelocityAccelerationConvergenceDivergenceIndicatorStream::try_new(
        VelocityAccelerationConvergenceDivergenceIndicatorParams {
            length: Some(length),
            smooth_length: Some(smooth_length),
        },
    )
    .expect("validated params");
    for (i, &value) in data.iter().enumerate() {
        if let Some((out_vacd, out_signal)) = stream.update(value) {
            vacd[i] = out_vacd;
            signal[i] = out_signal;
        }
    }
}

pub fn velocity_acceleration_convergence_divergence_indicator(
    input: &VelocityAccelerationConvergenceDivergenceIndicatorInput,
) -> Result<
    VelocityAccelerationConvergenceDivergenceIndicatorOutput,
    VelocityAccelerationConvergenceDivergenceIndicatorError,
> {
    velocity_acceleration_convergence_divergence_indicator_with_kernel(input, Kernel::Auto)
}

pub fn velocity_acceleration_convergence_divergence_indicator_with_kernel(
    input: &VelocityAccelerationConvergenceDivergenceIndicatorInput,
    kernel: Kernel,
) -> Result<
    VelocityAccelerationConvergenceDivergenceIndicatorOutput,
    VelocityAccelerationConvergenceDivergenceIndicatorError,
> {
    let data = input.as_ref();
    let length = input.get_length();
    let smooth_length = input.get_smooth_length();
    validate_common(data, length, smooth_length)?;

    let _chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let prefix = warmup_prefix(smooth_length);
    let mut vacd = alloc_with_nan_prefix(data.len(), prefix);
    let mut signal = alloc_with_nan_prefix(data.len(), prefix);
    compute_row(data, length, smooth_length, &mut vacd, &mut signal);
    Ok(VelocityAccelerationConvergenceDivergenceIndicatorOutput { vacd, signal })
}

pub fn velocity_acceleration_convergence_divergence_indicator_into_slice(
    dst_vacd: &mut [f64],
    dst_signal: &mut [f64],
    input: &VelocityAccelerationConvergenceDivergenceIndicatorInput,
    kernel: Kernel,
) -> Result<(), VelocityAccelerationConvergenceDivergenceIndicatorError> {
    let data = input.as_ref();
    let length = input.get_length();
    let smooth_length = input.get_smooth_length();
    validate_common(data, length, smooth_length)?;
    if dst_vacd.len() != data.len() {
        return Err(
            VelocityAccelerationConvergenceDivergenceIndicatorError::OutputLengthMismatch {
                expected: data.len(),
                got: dst_vacd.len(),
            },
        );
    }
    if dst_signal.len() != data.len() {
        return Err(
            VelocityAccelerationConvergenceDivergenceIndicatorError::OutputLengthMismatch {
                expected: data.len(),
                got: dst_signal.len(),
            },
        );
    }

    let _chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    compute_row(data, length, smooth_length, dst_vacd, dst_signal);
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn velocity_acceleration_convergence_divergence_indicator_into(
    input: &VelocityAccelerationConvergenceDivergenceIndicatorInput,
    out_vacd: &mut [f64],
    out_signal: &mut [f64],
) -> Result<(), VelocityAccelerationConvergenceDivergenceIndicatorError> {
    velocity_acceleration_convergence_divergence_indicator_into_slice(
        out_vacd,
        out_signal,
        input,
        Kernel::Auto,
    )
}

#[derive(Debug, Clone, Copy)]
pub struct VelocityAccelerationConvergenceDivergenceIndicatorBatchRange {
    pub length: (usize, usize, usize),
    pub smooth_length: (usize, usize, usize),
}

impl Default for VelocityAccelerationConvergenceDivergenceIndicatorBatchRange {
    fn default() -> Self {
        Self {
            length: (21, 21, 0),
            smooth_length: (5, 5, 0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VelocityAccelerationConvergenceDivergenceIndicatorBatchOutput {
    pub vacd: Vec<f64>,
    pub signal: Vec<f64>,
    pub combos: Vec<VelocityAccelerationConvergenceDivergenceIndicatorParams>,
    pub rows: usize,
    pub cols: usize,
}

impl VelocityAccelerationConvergenceDivergenceIndicatorBatchOutput {
    pub fn row_for_params(
        &self,
        params: &VelocityAccelerationConvergenceDivergenceIndicatorParams,
    ) -> Option<usize> {
        let length = params.length.unwrap_or(21);
        let smooth_length = params.smooth_length.unwrap_or(5);
        self.combos.iter().position(|combo| {
            combo.length.unwrap_or(21) == length
                && combo.smooth_length.unwrap_or(5) == smooth_length
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VelocityAccelerationConvergenceDivergenceIndicatorBatchBuilder {
    range: VelocityAccelerationConvergenceDivergenceIndicatorBatchRange,
    kernel: Kernel,
}

impl Default for VelocityAccelerationConvergenceDivergenceIndicatorBatchBuilder {
    fn default() -> Self {
        Self {
            range: VelocityAccelerationConvergenceDivergenceIndicatorBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl VelocityAccelerationConvergenceDivergenceIndicatorBatchBuilder {
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
    ) -> Result<
        VelocityAccelerationConvergenceDivergenceIndicatorBatchOutput,
        VelocityAccelerationConvergenceDivergenceIndicatorError,
    > {
        velocity_acceleration_convergence_divergence_indicator_batch_with_kernel(
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
        VelocityAccelerationConvergenceDivergenceIndicatorBatchOutput,
        VelocityAccelerationConvergenceDivergenceIndicatorError,
    > {
        velocity_acceleration_convergence_divergence_indicator_batch_with_kernel(
            source_type(candles, source),
            &self.range,
            self.kernel,
        )
    }
}

#[inline(always)]
fn expand_axis(
    range: (usize, usize, usize),
    is_smooth: bool,
) -> Result<Vec<usize>, VelocityAccelerationConvergenceDivergenceIndicatorError> {
    let (start, end, step) = range;
    if is_smooth {
        validate_smooth_length(start)?;
    } else {
        validate_length(start)?;
    }
    if step == 0 {
        return Ok(vec![start]);
    }
    if start > end {
        return Err(
            VelocityAccelerationConvergenceDivergenceIndicatorError::InvalidRange {
                start,
                end,
                step,
            },
        );
    }
    let mut values = Vec::new();
    let mut cur = start;
    loop {
        values.push(cur);
        if cur >= end {
            break;
        }
        let next = cur.checked_add(step).ok_or_else(|| {
            VelocityAccelerationConvergenceDivergenceIndicatorError::InvalidInput {
                msg: "velocity_acceleration_convergence_divergence_indicator: range step overflow"
                    .to_string(),
            }
        })?;
        if next <= cur {
            return Err(
                VelocityAccelerationConvergenceDivergenceIndicatorError::InvalidRange {
                    start,
                    end,
                    step,
                },
            );
        }
        cur = next.min(end);
    }
    Ok(values)
}

#[inline(always)]
fn expand_grid_checked(
    range: &VelocityAccelerationConvergenceDivergenceIndicatorBatchRange,
) -> Result<
    Vec<VelocityAccelerationConvergenceDivergenceIndicatorParams>,
    VelocityAccelerationConvergenceDivergenceIndicatorError,
> {
    let lengths = expand_axis(range.length, false)?;
    let smooth_lengths = expand_axis(range.smooth_length, true)?;
    let mut combos = Vec::with_capacity(lengths.len() * smooth_lengths.len());
    for &length in &lengths {
        for &smooth_length in &smooth_lengths {
            combos.push(VelocityAccelerationConvergenceDivergenceIndicatorParams {
                length: Some(length),
                smooth_length: Some(smooth_length),
            });
        }
    }
    Ok(combos)
}

pub fn expand_grid_velocity_acceleration_convergence_divergence_indicator(
    range: &VelocityAccelerationConvergenceDivergenceIndicatorBatchRange,
) -> Vec<VelocityAccelerationConvergenceDivergenceIndicatorParams> {
    expand_grid_checked(range).unwrap_or_default()
}

pub fn velocity_acceleration_convergence_divergence_indicator_batch_with_kernel(
    data: &[f64],
    sweep: &VelocityAccelerationConvergenceDivergenceIndicatorBatchRange,
    kernel: Kernel,
) -> Result<
    VelocityAccelerationConvergenceDivergenceIndicatorBatchOutput,
    VelocityAccelerationConvergenceDivergenceIndicatorError,
> {
    velocity_acceleration_convergence_divergence_indicator_batch_inner(data, sweep, kernel, true)
}

pub fn velocity_acceleration_convergence_divergence_indicator_batch_slice(
    data: &[f64],
    sweep: &VelocityAccelerationConvergenceDivergenceIndicatorBatchRange,
    kernel: Kernel,
) -> Result<
    VelocityAccelerationConvergenceDivergenceIndicatorBatchOutput,
    VelocityAccelerationConvergenceDivergenceIndicatorError,
> {
    velocity_acceleration_convergence_divergence_indicator_batch_inner(data, sweep, kernel, false)
}

pub fn velocity_acceleration_convergence_divergence_indicator_batch_par_slice(
    data: &[f64],
    sweep: &VelocityAccelerationConvergenceDivergenceIndicatorBatchRange,
    kernel: Kernel,
) -> Result<
    VelocityAccelerationConvergenceDivergenceIndicatorBatchOutput,
    VelocityAccelerationConvergenceDivergenceIndicatorError,
> {
    velocity_acceleration_convergence_divergence_indicator_batch_inner(data, sweep, kernel, true)
}

fn velocity_acceleration_convergence_divergence_indicator_batch_inner(
    data: &[f64],
    sweep: &VelocityAccelerationConvergenceDivergenceIndicatorBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<
    VelocityAccelerationConvergenceDivergenceIndicatorBatchOutput,
    VelocityAccelerationConvergenceDivergenceIndicatorError,
> {
    let combos = expand_grid_checked(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    let total = rows.checked_mul(cols).ok_or_else(|| {
        VelocityAccelerationConvergenceDivergenceIndicatorError::InvalidInput {
            msg: "velocity_acceleration_convergence_divergence_indicator: rows*cols overflow in batch"
                .to_string(),
        }
    })?;
    if data.is_empty() {
        return Err(VelocityAccelerationConvergenceDivergenceIndicatorError::EmptyInputData);
    }
    let valid = longest_valid_run(data);
    if valid == 0 {
        return Err(VelocityAccelerationConvergenceDivergenceIndicatorError::AllValuesNaN);
    }
    let mut max_needed = 0usize;
    let warmups: Vec<usize> = combos
        .iter()
        .map(|params| {
            let smooth_length = params.smooth_length.unwrap_or(5);
            max_needed = max_needed.max(smooth_length);
            warmup_prefix(smooth_length)
        })
        .collect();
    if valid < max_needed {
        return Err(
            VelocityAccelerationConvergenceDivergenceIndicatorError::NotEnoughValidData {
                needed: max_needed,
                valid,
            },
        );
    }

    let mut vacd_mu = make_uninit_matrix(rows, cols);
    let mut signal_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut vacd_mu, cols, &warmups);
    init_matrix_prefixes(&mut signal_mu, cols, &warmups);

    let mut vacd = unsafe {
        Vec::from_raw_parts(
            vacd_mu.as_mut_ptr() as *mut f64,
            vacd_mu.len(),
            vacd_mu.capacity(),
        )
    };
    let mut signal = unsafe {
        Vec::from_raw_parts(
            signal_mu.as_mut_ptr() as *mut f64,
            signal_mu.len(),
            signal_mu.capacity(),
        )
    };
    std::mem::forget(vacd_mu);
    std::mem::forget(signal_mu);
    debug_assert_eq!(vacd.len(), total);
    debug_assert_eq!(signal.len(), total);

    velocity_acceleration_convergence_divergence_indicator_batch_inner_into(
        data,
        sweep,
        kernel,
        parallel,
        &mut vacd,
        &mut signal,
    )?;

    Ok(
        VelocityAccelerationConvergenceDivergenceIndicatorBatchOutput {
            vacd,
            signal,
            combos,
            rows,
            cols,
        },
    )
}

fn velocity_acceleration_convergence_divergence_indicator_batch_inner_into(
    data: &[f64],
    sweep: &VelocityAccelerationConvergenceDivergenceIndicatorBatchRange,
    kernel: Kernel,
    parallel: bool,
    out_vacd: &mut [f64],
    out_signal: &mut [f64],
) -> Result<
    Vec<VelocityAccelerationConvergenceDivergenceIndicatorParams>,
    VelocityAccelerationConvergenceDivergenceIndicatorError,
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
            return Err(
                VelocityAccelerationConvergenceDivergenceIndicatorError::InvalidKernelForBatch(
                    other,
                ),
            )
        }
    }

    let combos = expand_grid_checked(sweep)?;
    let len = data.len();
    if len == 0 {
        return Err(VelocityAccelerationConvergenceDivergenceIndicatorError::EmptyInputData);
    }
    let total = combos.len().checked_mul(len).ok_or_else(|| {
        VelocityAccelerationConvergenceDivergenceIndicatorError::InvalidInput {
            msg: "velocity_acceleration_convergence_divergence_indicator: rows*cols overflow in batch_into"
                .to_string(),
        }
    })?;
    if out_vacd.len() != total {
        return Err(
            VelocityAccelerationConvergenceDivergenceIndicatorError::MismatchedOutputLen {
                dst_len: out_vacd.len(),
                expected_len: total,
            },
        );
    }
    if out_signal.len() != total {
        return Err(
            VelocityAccelerationConvergenceDivergenceIndicatorError::MismatchedOutputLen {
                dst_len: out_signal.len(),
                expected_len: total,
            },
        );
    }

    let valid = longest_valid_run(data);
    if valid == 0 {
        return Err(VelocityAccelerationConvergenceDivergenceIndicatorError::AllValuesNaN);
    }
    let max_needed = combos
        .iter()
        .map(|params| params.smooth_length.unwrap_or(5))
        .max()
        .unwrap_or(0);
    if valid < max_needed {
        return Err(
            VelocityAccelerationConvergenceDivergenceIndicatorError::NotEnoughValidData {
                needed: max_needed,
                valid,
            },
        );
    }

    let _chosen = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other => other,
    };

    let worker = |row: usize, dst_vacd: &mut [f64], dst_signal: &mut [f64]| {
        let params = &combos[row];
        compute_row(
            data,
            params.length.unwrap_or(21),
            params.smooth_length.unwrap_or(5),
            dst_vacd,
            dst_signal,
        );
    };

    if parallel && combos.len() > 1 {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_vacd
                .par_chunks_mut(len)
                .zip(out_signal.par_chunks_mut(len))
                .enumerate()
                .for_each(|(row, (dst_vacd, dst_signal))| worker(row, dst_vacd, dst_signal));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, (dst_vacd, dst_signal)) in out_vacd
                .chunks_mut(len)
                .zip(out_signal.chunks_mut(len))
                .enumerate()
            {
                worker(row, dst_vacd, dst_signal);
            }
        }
    } else {
        for (row, (dst_vacd, dst_signal)) in out_vacd
            .chunks_mut(len)
            .zip(out_signal.chunks_mut(len))
            .enumerate()
        {
            worker(row, dst_vacd, dst_signal);
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "velocity_acceleration_convergence_divergence_indicator")]
#[pyo3(signature = (data, length=21, smooth_length=5, kernel=None))]
pub fn velocity_acceleration_convergence_divergence_indicator_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length: usize,
    smooth_length: usize,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let data = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let input = VelocityAccelerationConvergenceDivergenceIndicatorInput::from_slice(
        data,
        VelocityAccelerationConvergenceDivergenceIndicatorParams {
            length: Some(length),
            smooth_length: Some(smooth_length),
        },
    );
    let out = py
        .allow_threads(|| {
            velocity_acceleration_convergence_divergence_indicator_with_kernel(&input, kern)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((out.vacd.into_pyarray(py), out.signal.into_pyarray(py)))
}

#[cfg(feature = "python")]
#[pyclass(name = "VelocityAccelerationConvergenceDivergenceIndicatorStream")]
pub struct VelocityAccelerationConvergenceDivergenceIndicatorStreamPy {
    stream: VelocityAccelerationConvergenceDivergenceIndicatorStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl VelocityAccelerationConvergenceDivergenceIndicatorStreamPy {
    #[new]
    #[pyo3(signature = (length=21, smooth_length=5))]
    fn new(length: usize, smooth_length: usize) -> PyResult<Self> {
        let stream = VelocityAccelerationConvergenceDivergenceIndicatorStream::try_new(
            VelocityAccelerationConvergenceDivergenceIndicatorParams {
                length: Some(length),
                smooth_length: Some(smooth_length),
            },
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        self.stream.update(value)
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
#[pyfunction(name = "velocity_acceleration_convergence_divergence_indicator_batch")]
#[pyo3(signature = (data, length_range=(21, 21, 0), smooth_length_range=(5, 5, 0), kernel=None))]
pub fn velocity_acceleration_convergence_divergence_indicator_batch_py<'py>(
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
            velocity_acceleration_convergence_divergence_indicator_batch_with_kernel(
                data,
                &VelocityAccelerationConvergenceDivergenceIndicatorBatchRange {
                    length: length_range,
                    smooth_length: smooth_length_range,
                },
                kern,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item(
        "vacd",
        output
            .vacd
            .into_pyarray(py)
            .reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "signal",
        output
            .signal
            .into_pyarray(py)
            .reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "lengths",
        output
            .combos
            .iter()
            .map(|params| params.length.unwrap_or(21) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "smooth_lengths",
        output
            .combos
            .iter()
            .map(|params| params.smooth_length.unwrap_or(5) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", output.rows)?;
    dict.set_item("cols", output.cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_velocity_acceleration_convergence_divergence_indicator_module(
    m: &Bound<'_, PyModule>,
) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        velocity_acceleration_convergence_divergence_indicator_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        velocity_acceleration_convergence_divergence_indicator_batch_py,
        m
    )?)?;
    m.add_class::<VelocityAccelerationConvergenceDivergenceIndicatorStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VelocityAccelerationConvergenceDivergenceIndicatorBatchConfig {
    pub length_range: Vec<usize>,
    pub smooth_length_range: Vec<usize>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = velocity_acceleration_convergence_divergence_indicator_js)]
pub fn velocity_acceleration_convergence_divergence_indicator_js(
    data: &[f64],
    length: usize,
    smooth_length: usize,
) -> Result<JsValue, JsValue> {
    let input = VelocityAccelerationConvergenceDivergenceIndicatorInput::from_slice(
        data,
        VelocityAccelerationConvergenceDivergenceIndicatorParams {
            length: Some(length),
            smooth_length: Some(smooth_length),
        },
    );
    let out =
        velocity_acceleration_convergence_divergence_indicator_with_kernel(&input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("vacd"),
        &serde_wasm_bindgen::to_value(&out.vacd).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("signal"),
        &serde_wasm_bindgen::to_value(&out.signal).unwrap(),
    )?;
    Ok(obj.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = velocity_acceleration_convergence_divergence_indicator_batch_js)]
pub fn velocity_acceleration_convergence_divergence_indicator_batch_js(
    data: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: VelocityAccelerationConvergenceDivergenceIndicatorBatchConfig =
        serde_wasm_bindgen::from_value(config)
            .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.length_range.len() != 3 || config.smooth_length_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: every range must have exactly 3 elements [start, end, step]",
        ));
    }

    let out = velocity_acceleration_convergence_divergence_indicator_batch_with_kernel(
        data,
        &VelocityAccelerationConvergenceDivergenceIndicatorBatchRange {
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
        &JsValue::from_str("vacd"),
        &serde_wasm_bindgen::to_value(&out.vacd).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("signal"),
        &serde_wasm_bindgen::to_value(&out.signal).unwrap(),
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
pub fn velocity_acceleration_convergence_divergence_indicator_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(2 * len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn velocity_acceleration_convergence_divergence_indicator_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 2 * len, 2 * len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn velocity_acceleration_convergence_divergence_indicator_into(
    data_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length: usize,
    smooth_length: usize,
) -> Result<(), JsValue> {
    if data_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to velocity_acceleration_convergence_divergence_indicator_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, 2 * len);
        let (dst_vacd, dst_signal) = out.split_at_mut(len);
        let input = VelocityAccelerationConvergenceDivergenceIndicatorInput::from_slice(
            data,
            VelocityAccelerationConvergenceDivergenceIndicatorParams {
                length: Some(length),
                smooth_length: Some(smooth_length),
            },
        );
        velocity_acceleration_convergence_divergence_indicator_into_slice(
            dst_vacd,
            dst_signal,
            &input,
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn velocity_acceleration_convergence_divergence_indicator_batch_into(
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
            "null pointer passed to velocity_acceleration_convergence_divergence_indicator_batch_into",
        ));
    }
    let sweep = VelocityAccelerationConvergenceDivergenceIndicatorBatchRange {
        length: (length_start, length_end, length_step),
        smooth_length: (smooth_length_start, smooth_length_end, smooth_length_step),
    };
    let combos = expand_grid_checked(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let total = rows
        .checked_mul(len)
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| {
            JsValue::from_str(
                "rows*cols overflow in velocity_acceleration_convergence_divergence_indicator_batch_into",
            )
        })?;
    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        let split = rows * len;
        let (dst_vacd, dst_signal) = out.split_at_mut(split);
        velocity_acceleration_convergence_divergence_indicator_batch_inner_into(
            data,
            &sweep,
            Kernel::Auto,
            false,
            dst_vacd,
            dst_signal,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::dispatch::{
        compute_cpu, IndicatorComputeRequest, IndicatorDataRef, ParamKV, ParamValue,
    };

    fn sample_ohlc(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let close = (0..len)
            .map(|i| {
                let x = i as f64;
                100.0 + x * 0.18 + (x * 0.21).sin() * 2.3 + (x * 0.07).cos() * 0.9
            })
            .collect::<Vec<_>>();
        let open = close
            .iter()
            .enumerate()
            .map(|(i, &c)| c - (i as f64 * 0.15).sin() * 0.7 - 0.2)
            .collect::<Vec<_>>();
        let high = open
            .iter()
            .zip(close.iter())
            .map(|(&o, &c)| o.max(c) + 0.6)
            .collect::<Vec<_>>();
        let low = open
            .iter()
            .zip(close.iter())
            .map(|(&o, &c)| o.min(c) - 0.6)
            .collect::<Vec<_>>();
        let hlcc4 = high
            .iter()
            .zip(low.iter())
            .zip(close.iter())
            .map(|((&h, &l), &c)| (h + l + 2.0 * c) * 0.25)
            .collect::<Vec<_>>();
        (open, high, low, close, hlcc4)
    }

    fn assert_series_close(left: &[f64], right: &[f64], tol: f64) {
        assert_eq!(left.len(), right.len());
        for (&a, &b) in left.iter().zip(right.iter()) {
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan());
            } else {
                assert!((a - b).abs() <= tol, "left={a} right={b}");
            }
        }
    }

    #[test]
    fn velocity_acceleration_convergence_divergence_indicator_output_contract(
    ) -> Result<(), Box<dyn Error>> {
        let (_open, _high, _low, _close, hlcc4) = sample_ohlc(256);
        let input = VelocityAccelerationConvergenceDivergenceIndicatorInput::from_slice(
            &hlcc4,
            VelocityAccelerationConvergenceDivergenceIndicatorParams {
                length: Some(21),
                smooth_length: Some(5),
            },
        );
        let out = velocity_acceleration_convergence_divergence_indicator(&input)?;
        assert_eq!(out.vacd.len(), hlcc4.len());
        assert_eq!(out.signal.len(), hlcc4.len());
        assert_eq!(out.vacd.iter().position(|v| v.is_finite()), Some(4));
        assert_eq!(out.signal.iter().position(|v| v.is_finite()), Some(4));
        Ok(())
    }

    #[test]
    fn velocity_acceleration_convergence_divergence_indicator_into_matches_api(
    ) -> Result<(), Box<dyn Error>> {
        let (_open, _high, _low, _close, hlcc4) = sample_ohlc(220);
        let input = VelocityAccelerationConvergenceDivergenceIndicatorInput::from_slice(
            &hlcc4,
            VelocityAccelerationConvergenceDivergenceIndicatorParams {
                length: Some(21),
                smooth_length: Some(5),
            },
        );
        let base = velocity_acceleration_convergence_divergence_indicator(&input)?;
        let mut vacd = vec![f64::NAN; hlcc4.len()];
        let mut signal = vec![f64::NAN; hlcc4.len()];
        velocity_acceleration_convergence_divergence_indicator_into_slice(
            &mut vacd,
            &mut signal,
            &input,
            Kernel::Auto,
        )?;
        assert_series_close(&base.vacd, &vacd, 1e-12);
        assert_series_close(&base.signal, &signal, 1e-12);
        Ok(())
    }

    #[test]
    fn velocity_acceleration_convergence_divergence_indicator_stream_matches_batch(
    ) -> Result<(), Box<dyn Error>> {
        let (_open, _high, _low, _close, hlcc4) = sample_ohlc(256);
        let batch = velocity_acceleration_convergence_divergence_indicator(
            &VelocityAccelerationConvergenceDivergenceIndicatorInput::from_slice(
                &hlcc4,
                VelocityAccelerationConvergenceDivergenceIndicatorParams {
                    length: Some(21),
                    smooth_length: Some(5),
                },
            ),
        )?;

        let mut stream = VelocityAccelerationConvergenceDivergenceIndicatorStream::try_new(
            VelocityAccelerationConvergenceDivergenceIndicatorParams {
                length: Some(21),
                smooth_length: Some(5),
            },
        )?;
        let mut vacd = Vec::with_capacity(hlcc4.len());
        let mut signal = Vec::with_capacity(hlcc4.len());
        for &value in &hlcc4 {
            if let Some((a, b)) = stream.update(value) {
                vacd.push(a);
                signal.push(b);
            } else {
                vacd.push(f64::NAN);
                signal.push(f64::NAN);
            }
        }
        assert_series_close(&batch.vacd, &vacd, 1e-12);
        assert_series_close(&batch.signal, &signal, 1e-12);
        Ok(())
    }

    #[test]
    fn velocity_acceleration_convergence_divergence_indicator_batch_single_matches_single(
    ) -> Result<(), Box<dyn Error>> {
        let (_open, _high, _low, _close, hlcc4) = sample_ohlc(240);
        let single = velocity_acceleration_convergence_divergence_indicator(
            &VelocityAccelerationConvergenceDivergenceIndicatorInput::from_slice(
                &hlcc4,
                VelocityAccelerationConvergenceDivergenceIndicatorParams {
                    length: Some(21),
                    smooth_length: Some(5),
                },
            ),
        )?;
        let batch = velocity_acceleration_convergence_divergence_indicator_batch_with_kernel(
            &hlcc4,
            &VelocityAccelerationConvergenceDivergenceIndicatorBatchRange {
                length: (21, 21, 0),
                smooth_length: (5, 5, 0),
            },
            Kernel::Auto,
        )?;
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, hlcc4.len());
        assert_series_close(&single.vacd, &batch.vacd, 1e-12);
        assert_series_close(&single.signal, &batch.signal, 1e-12);
        Ok(())
    }

    #[test]
    fn velocity_acceleration_convergence_divergence_indicator_rejects_invalid_params() {
        let (_open, _high, _low, _close, hlcc4) = sample_ohlc(64);
        let err = velocity_acceleration_convergence_divergence_indicator(
            &VelocityAccelerationConvergenceDivergenceIndicatorInput::from_slice(
                &hlcc4,
                VelocityAccelerationConvergenceDivergenceIndicatorParams {
                    length: Some(1),
                    smooth_length: Some(5),
                },
            ),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            VelocityAccelerationConvergenceDivergenceIndicatorError::InvalidLength { .. }
        ));
    }

    #[test]
    fn velocity_acceleration_convergence_divergence_indicator_dispatch_compute_returns_outputs(
    ) -> Result<(), Box<dyn Error>> {
        let (open, high, low, close, _hlcc4) = sample_ohlc(192);
        for output_id in ["vacd", "signal"] {
            let req = IndicatorComputeRequest {
                indicator_id: "velocity_acceleration_convergence_divergence_indicator",
                output_id: Some(output_id),
                data: IndicatorDataRef::Ohlc {
                    open: &open,
                    high: &high,
                    low: &low,
                    close: &close,
                },
                params: &[
                    ParamKV {
                        key: "length",
                        value: ParamValue::Int(21),
                    },
                    ParamKV {
                        key: "smooth_length",
                        value: ParamValue::Int(5),
                    },
                ],
                kernel: Kernel::Auto,
            };
            let out = compute_cpu(req)?;
            assert_eq!(out.output_id, output_id);
            assert_eq!(out.rows, 1);
            assert_eq!(out.cols, close.len());
        }
        Ok(())
    }
}
