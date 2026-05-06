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
    alloc_uninit_f64, detect_best_batch_kernel, init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::collections::VecDeque;
use std::convert::AsRef;
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

const DEFAULT_LENGTH: usize = 20;
const DEFAULT_SMOOTH: usize = 10;

impl<'a> AsRef<[f64]> for ForwardBackwardExponentialOscillatorInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            ForwardBackwardExponentialOscillatorData::Slice(slice) => slice,
            ForwardBackwardExponentialOscillatorData::Candles { candles, source } => {
                match *source {
                    "open" => &candles.open,
                    "high" => &candles.high,
                    "low" => &candles.low,
                    "close" => &candles.close,
                    "volume" => &candles.volume,
                    _ => source_type(candles, source),
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum ForwardBackwardExponentialOscillatorData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct ForwardBackwardExponentialOscillatorOutput {
    pub forward_backward: Vec<f64>,
    pub backward: Vec<f64>,
    pub histogram: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct ForwardBackwardExponentialOscillatorParams {
    pub length: Option<usize>,
    pub smooth: Option<usize>,
}

impl Default for ForwardBackwardExponentialOscillatorParams {
    fn default() -> Self {
        Self {
            length: Some(DEFAULT_LENGTH),
            smooth: Some(DEFAULT_SMOOTH),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ForwardBackwardExponentialOscillatorInput<'a> {
    pub data: ForwardBackwardExponentialOscillatorData<'a>,
    pub params: ForwardBackwardExponentialOscillatorParams,
}

impl<'a> ForwardBackwardExponentialOscillatorInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: ForwardBackwardExponentialOscillatorParams,
    ) -> Self {
        Self {
            data: ForwardBackwardExponentialOscillatorData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(
        slice: &'a [f64],
        params: ForwardBackwardExponentialOscillatorParams,
    ) -> Self {
        Self {
            data: ForwardBackwardExponentialOscillatorData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(
            candles,
            "close",
            ForwardBackwardExponentialOscillatorParams::default(),
        )
    }

    #[inline(always)]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(DEFAULT_LENGTH)
    }

    #[inline(always)]
    pub fn get_smooth(&self) -> usize {
        self.params.smooth.unwrap_or(DEFAULT_SMOOTH)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ForwardBackwardExponentialOscillatorBuilder {
    length: Option<usize>,
    smooth: Option<usize>,
    kernel: Kernel,
}

impl Default for ForwardBackwardExponentialOscillatorBuilder {
    fn default() -> Self {
        Self {
            length: None,
            smooth: None,
            kernel: Kernel::Auto,
        }
    }
}

impl ForwardBackwardExponentialOscillatorBuilder {
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
    pub fn smooth(mut self, value: usize) -> Self {
        self.smooth = Some(value);
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
    ) -> Result<ForwardBackwardExponentialOscillatorOutput, ForwardBackwardExponentialOscillatorError>
    {
        let input = ForwardBackwardExponentialOscillatorInput::from_candles(
            candles,
            "close",
            ForwardBackwardExponentialOscillatorParams {
                length: self.length,
                smooth: self.smooth,
            },
        );
        forward_backward_exponential_oscillator_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<ForwardBackwardExponentialOscillatorOutput, ForwardBackwardExponentialOscillatorError>
    {
        let input = ForwardBackwardExponentialOscillatorInput::from_slice(
            data,
            ForwardBackwardExponentialOscillatorParams {
                length: self.length,
                smooth: self.smooth,
            },
        );
        forward_backward_exponential_oscillator_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<ForwardBackwardExponentialOscillatorStream, ForwardBackwardExponentialOscillatorError>
    {
        ForwardBackwardExponentialOscillatorStream::try_new(
            ForwardBackwardExponentialOscillatorParams {
                length: self.length,
                smooth: self.smooth,
            },
        )
    }
}

#[derive(Debug, Error)]
pub enum ForwardBackwardExponentialOscillatorError {
    #[error("forward_backward_exponential_oscillator: input data slice is empty")]
    EmptyInputData,
    #[error("forward_backward_exponential_oscillator: all values are NaN")]
    AllValuesNaN,
    #[error(
        "forward_backward_exponential_oscillator: invalid length: length = {length}, data length = {data_len}"
    )]
    InvalidLength { length: usize, data_len: usize },
    #[error(
        "forward_backward_exponential_oscillator: invalid smooth: smooth = {smooth}, data length = {data_len}"
    )]
    InvalidSmooth { smooth: usize, data_len: usize },
    #[error(
        "forward_backward_exponential_oscillator: not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "forward_backward_exponential_oscillator: output length mismatch: expected {expected}, got {got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error(
        "forward_backward_exponential_oscillator: invalid range: start={start}, end={end}, step={step}"
    )]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("forward_backward_exponential_oscillator: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Clone, Debug)]
struct PreparedInput<'a> {
    data: &'a [f64],
    len: usize,
    length: usize,
    alpha: f64,
    warmup_forward_backward: usize,
    warmup_backward: usize,
}

#[derive(Clone, Debug)]
struct RollingDiffWindow {
    period: usize,
    values: VecDeque<f64>,
    sum: f64,
    abs_sum: f64,
}

impl RollingDiffWindow {
    #[inline(always)]
    fn new(period: usize) -> Self {
        Self {
            period,
            values: VecDeque::with_capacity(period.max(1)),
            sum: 0.0,
            abs_sum: 0.0,
        }
    }

    #[inline(always)]
    fn clear(&mut self) {
        self.values.clear();
        self.sum = 0.0;
        self.abs_sum = 0.0;
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        self.values.push_back(value);
        self.sum += value;
        self.abs_sum += value.abs();
        if self.values.len() > self.period {
            if let Some(removed) = self.values.pop_front() {
                self.sum -= removed;
                self.abs_sum -= removed.abs();
            }
        }
        if self.values.len() == self.period {
            Some((self.sum, self.abs_sum))
        } else {
            None
        }
    }
}

#[inline(always)]
fn ema_step(prev: &mut Option<f64>, value: f64, alpha: f64) -> f64 {
    let next = match *prev {
        Some(last) => alpha * value + (1.0 - alpha) * last,
        None => value,
    };
    *prev = Some(next);
    next
}

#[inline(always)]
fn push_window(window: &mut VecDeque<f64>, period: usize, value: f64) {
    if window.len() == period {
        window.pop_front();
    }
    window.push_back(value);
}

#[inline(always)]
fn compute_forward_backward_value(window: &VecDeque<f64>, alpha: f64) -> f64 {
    let Some(&current) = window.back() else {
        return f64::NAN;
    };
    let mut ema2 = current;
    let mut prev = ema2;
    let mut num = 0.0;
    let mut den = 0.0;

    for value in window.iter().rev().skip(1) {
        ema2 += alpha * (*value - ema2);
        let dt = prev - ema2;
        num += dt;
        den += dt.abs();
        prev = ema2;
    }

    if den != 0.0 {
        num / den * 50.0 + 50.0
    } else {
        f64::NAN
    }
}

#[inline]
pub fn forward_backward_exponential_oscillator(
    input: &ForwardBackwardExponentialOscillatorInput<'_>,
) -> Result<ForwardBackwardExponentialOscillatorOutput, ForwardBackwardExponentialOscillatorError> {
    forward_backward_exponential_oscillator_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn forward_backward_exponential_oscillator_with_kernel(
    input: &ForwardBackwardExponentialOscillatorInput<'_>,
    kernel: Kernel,
) -> Result<ForwardBackwardExponentialOscillatorOutput, ForwardBackwardExponentialOscillatorError> {
    let prepared = prepare_input(input, kernel)?;
    let mut forward_backward = alloc_uninit_f64(prepared.len);
    let mut backward = alloc_uninit_f64(prepared.len);
    let mut histogram = alloc_uninit_f64(prepared.len);
    compute_into_slices(
        &prepared,
        &mut forward_backward,
        &mut backward,
        &mut histogram,
    )?;
    Ok(ForwardBackwardExponentialOscillatorOutput {
        forward_backward,
        backward,
        histogram,
    })
}

#[inline]
pub fn forward_backward_exponential_oscillator_into(
    input: &ForwardBackwardExponentialOscillatorInput<'_>,
    forward_backward: &mut [f64],
    backward: &mut [f64],
    histogram: &mut [f64],
) -> Result<(), ForwardBackwardExponentialOscillatorError> {
    forward_backward_exponential_oscillator_into_slices(
        input,
        Kernel::Auto,
        forward_backward,
        backward,
        histogram,
    )
}

#[inline]
pub fn forward_backward_exponential_oscillator_into_slices(
    input: &ForwardBackwardExponentialOscillatorInput<'_>,
    kernel: Kernel,
    forward_backward: &mut [f64],
    backward: &mut [f64],
    histogram: &mut [f64],
) -> Result<(), ForwardBackwardExponentialOscillatorError> {
    let prepared = prepare_input(input, kernel)?;
    if forward_backward.len() != prepared.len
        || backward.len() != prepared.len
        || histogram.len() != prepared.len
    {
        return Err(
            ForwardBackwardExponentialOscillatorError::OutputLengthMismatch {
                expected: prepared.len,
                got: *[forward_backward.len(), backward.len(), histogram.len()]
                    .iter()
                    .min()
                    .unwrap_or(&0),
            },
        );
    }
    compute_into_slices(&prepared, forward_backward, backward, histogram)
}

#[inline(always)]
fn prepare_input<'a>(
    input: &'a ForwardBackwardExponentialOscillatorInput<'_>,
    kernel: Kernel,
) -> Result<PreparedInput<'a>, ForwardBackwardExponentialOscillatorError> {
    let data = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(ForwardBackwardExponentialOscillatorError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|value| value.is_finite())
        .ok_or(ForwardBackwardExponentialOscillatorError::AllValuesNaN)?;

    let length = input.get_length();
    let smooth = input.get_smooth();

    if length == 0 || length > len {
        return Err(ForwardBackwardExponentialOscillatorError::InvalidLength {
            length,
            data_len: len,
        });
    }
    if smooth == 0 {
        return Err(ForwardBackwardExponentialOscillatorError::InvalidSmooth {
            smooth,
            data_len: len,
        });
    }

    let needed = length.max(2);
    let mut valid = 0usize;
    for value in &data[first..] {
        if value.is_finite() {
            valid += 1;
            if valid >= needed {
                break;
            }
        }
    }
    if valid < needed {
        return Err(
            ForwardBackwardExponentialOscillatorError::NotEnoughValidData { needed, valid },
        );
    }

    let _ = kernel;

    Ok(PreparedInput {
        data,
        len,
        length,
        alpha: 2.0 / (smooth as f64 + 1.0),
        warmup_forward_backward: first + length.saturating_sub(1),
        warmup_backward: first + length,
    })
}

#[inline(always)]
fn compute_into_slices(
    prepared: &PreparedInput<'_>,
    dst_forward_backward: &mut [f64],
    dst_backward: &mut [f64],
    dst_histogram: &mut [f64],
) -> Result<(), ForwardBackwardExponentialOscillatorError> {
    if dst_forward_backward.len() != prepared.len
        || dst_backward.len() != prepared.len
        || dst_histogram.len() != prepared.len
    {
        return Err(
            ForwardBackwardExponentialOscillatorError::OutputLengthMismatch {
                expected: prepared.len,
                got: *[
                    dst_forward_backward.len(),
                    dst_backward.len(),
                    dst_histogram.len(),
                ]
                .iter()
                .min()
                .unwrap_or(&0),
            },
        );
    }

    let mut ema1_state = None;
    let mut ema2_state = None;
    let mut prev_ema2 = None;
    let mut ema1_window = VecDeque::with_capacity(prepared.length.max(1));
    let mut diff_window = RollingDiffWindow::new(prepared.length);

    for i in 0..prepared.len {
        let value = prepared.data[i];
        let mut forward_backward_value = f64::NAN;
        let mut backward_value = f64::NAN;
        let mut histogram_value = f64::NAN;

        if value.is_finite() {
            let ema1 = ema_step(&mut ema1_state, value, prepared.alpha);
            push_window(&mut ema1_window, prepared.length, ema1);

            if ema1_window.len() == prepared.length {
                let fb = compute_forward_backward_value(&ema1_window, prepared.alpha);
                if fb.is_finite() {
                    forward_backward_value = fb;
                }
            }

            let ema2 = ema_step(&mut ema2_state, ema1, prepared.alpha);
            if let Some(last_ema2) = prev_ema2 {
                if let Some((num, den)) = diff_window.update(ema2 - last_ema2) {
                    if den != 0.0 {
                        let bw = num / den * 50.0 + 50.0;
                        backward_value = bw;
                        if forward_backward_value.is_finite() {
                            histogram_value = (forward_backward_value - bw) * 0.25 + 50.0;
                        }
                    }
                }
            }
            prev_ema2 = Some(ema2);
        } else {
            ema1_state = None;
            ema2_state = None;
            prev_ema2 = None;
            ema1_window.clear();
            diff_window.clear();
        }

        dst_forward_backward[i] = forward_backward_value;
        dst_backward[i] = backward_value;
        dst_histogram[i] = histogram_value;
    }

    Ok(())
}

#[derive(Clone, Debug)]
pub struct ForwardBackwardExponentialOscillatorBatchRange {
    pub length: (usize, usize, usize),
    pub smooth: (usize, usize, usize),
}

impl Default for ForwardBackwardExponentialOscillatorBatchRange {
    fn default() -> Self {
        Self {
            length: (DEFAULT_LENGTH, DEFAULT_LENGTH, 0),
            smooth: (DEFAULT_SMOOTH, DEFAULT_SMOOTH, 0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ForwardBackwardExponentialOscillatorBatchOutput {
    pub forward_backward: Vec<f64>,
    pub backward: Vec<f64>,
    pub histogram: Vec<f64>,
    pub combos: Vec<ForwardBackwardExponentialOscillatorParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Debug)]
pub struct ForwardBackwardExponentialOscillatorBatchBuilder {
    range: ForwardBackwardExponentialOscillatorBatchRange,
    kernel: Kernel,
}

impl Default for ForwardBackwardExponentialOscillatorBatchBuilder {
    fn default() -> Self {
        Self {
            range: ForwardBackwardExponentialOscillatorBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl ForwardBackwardExponentialOscillatorBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn range(mut self, value: ForwardBackwardExponentialOscillatorBatchRange) -> Self {
        self.range = value;
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
        ForwardBackwardExponentialOscillatorBatchOutput,
        ForwardBackwardExponentialOscillatorError,
    > {
        forward_backward_exponential_oscillator_batch_with_kernel(data, &self.range, self.kernel)
    }

    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<
        ForwardBackwardExponentialOscillatorBatchOutput,
        ForwardBackwardExponentialOscillatorError,
    > {
        self.apply_slice(candles.close.as_slice())
    }
}

fn axis_usize(
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, ForwardBackwardExponentialOscillatorError> {
    if step == 0 || start == end {
        return Ok(vec![start]);
    }
    let mut out = Vec::new();
    if start <= end {
        let mut current = start;
        while current <= end {
            out.push(current);
            match current.checked_add(step) {
                Some(next) => current = next,
                None => break,
            }
        }
    } else {
        let mut current = start;
        while current >= end {
            out.push(current);
            match current.checked_sub(step) {
                Some(next) => current = next,
                None => break,
            }
            if current < end {
                break;
            }
        }
    }
    if out.is_empty() {
        return Err(ForwardBackwardExponentialOscillatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

fn expand_grid(
    range: &ForwardBackwardExponentialOscillatorBatchRange,
) -> Result<
    Vec<ForwardBackwardExponentialOscillatorParams>,
    ForwardBackwardExponentialOscillatorError,
> {
    let lengths = axis_usize(range.length)?;
    let smooths = axis_usize(range.smooth)?;
    let total = lengths.len().checked_mul(smooths.len()).ok_or_else(|| {
        ForwardBackwardExponentialOscillatorError::InvalidRange {
            start: range.length.0.to_string(),
            end: range.length.1.to_string(),
            step: range.length.2.to_string(),
        }
    })?;

    let mut out = Vec::with_capacity(total);
    for &length in &lengths {
        for &smooth in &smooths {
            out.push(ForwardBackwardExponentialOscillatorParams {
                length: Some(length),
                smooth: Some(smooth),
            });
        }
    }
    Ok(out)
}

#[inline]
pub fn forward_backward_exponential_oscillator_batch_with_kernel(
    data: &[f64],
    range: &ForwardBackwardExponentialOscillatorBatchRange,
    kernel: Kernel,
) -> Result<
    ForwardBackwardExponentialOscillatorBatchOutput,
    ForwardBackwardExponentialOscillatorError,
> {
    if data.is_empty() {
        return Err(ForwardBackwardExponentialOscillatorError::EmptyInputData);
    }
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        value if value.is_batch() => value,
        _ => return Err(ForwardBackwardExponentialOscillatorError::InvalidKernelForBatch(kernel)),
    };
    let single_kernel = batch_kernel.to_non_batch();
    let combos = expand_grid(range)?;
    let rows = combos.len();
    let cols = data.len();

    let first = data
        .iter()
        .position(|value| value.is_finite())
        .ok_or(ForwardBackwardExponentialOscillatorError::AllValuesNaN)?;
    let fb_warmups: Vec<usize> = combos
        .iter()
        .map(|combo| first + combo.length.unwrap_or(DEFAULT_LENGTH).saturating_sub(1))
        .collect();
    let bw_warmups: Vec<usize> = combos
        .iter()
        .map(|combo| first + combo.length.unwrap_or(DEFAULT_LENGTH))
        .collect();

    let mut fb_mu = make_uninit_matrix(rows, cols);
    let mut bw_mu = make_uninit_matrix(rows, cols);
    let mut hist_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut fb_mu, cols, &fb_warmups);
    init_matrix_prefixes(&mut bw_mu, cols, &bw_warmups);
    init_matrix_prefixes(&mut hist_mu, cols, &bw_warmups);

    let mut fb_guard = ManuallyDrop::new(fb_mu);
    let mut bw_guard = ManuallyDrop::new(bw_mu);
    let mut hist_guard = ManuallyDrop::new(hist_mu);
    let fb_all = unsafe { mu_slice_as_f64_slice_mut(&mut fb_guard) };
    let bw_all = unsafe { mu_slice_as_f64_slice_mut(&mut bw_guard) };
    let hist_all = unsafe { mu_slice_as_f64_slice_mut(&mut hist_guard) };

    let run_row = |row: usize,
                   fb_row: &mut [f64],
                   bw_row: &mut [f64],
                   hist_row: &mut [f64]|
     -> Result<(), ForwardBackwardExponentialOscillatorError> {
        let input =
            ForwardBackwardExponentialOscillatorInput::from_slice(data, combos[row].clone());
        forward_backward_exponential_oscillator_into_slices(
            &input,
            single_kernel,
            fb_row,
            bw_row,
            hist_row,
        )
    };

    #[cfg(not(target_arch = "wasm32"))]
    {
        fb_all
            .par_chunks_mut(cols)
            .zip(bw_all.par_chunks_mut(cols))
            .zip(hist_all.par_chunks_mut(cols))
            .enumerate()
            .try_for_each(|(row, ((fb_row, bw_row), hist_row))| {
                run_row(row, fb_row, bw_row, hist_row)
            })?;
    }

    #[cfg(target_arch = "wasm32")]
    {
        for row in 0..rows {
            let start = row * cols;
            let end = start + cols;
            run_row(
                row,
                &mut fb_all[start..end],
                &mut bw_all[start..end],
                &mut hist_all[start..end],
            )?;
        }
    }

    Ok(ForwardBackwardExponentialOscillatorBatchOutput {
        forward_backward: unsafe { vec_f64_from_mu_guard(fb_guard) },
        backward: unsafe { vec_f64_from_mu_guard(bw_guard) },
        histogram: unsafe { vec_f64_from_mu_guard(hist_guard) },
        combos,
        rows,
        cols,
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ForwardBackwardExponentialOscillatorStreamOutput {
    pub forward_backward: f64,
    pub backward: f64,
    pub histogram: f64,
}

#[derive(Debug, Clone)]
pub struct ForwardBackwardExponentialOscillatorStream {
    length: usize,
    alpha: f64,
    ema1_state: Option<f64>,
    ema2_state: Option<f64>,
    prev_ema2: Option<f64>,
    ema1_window: VecDeque<f64>,
    diff_window: RollingDiffWindow,
}

impl ForwardBackwardExponentialOscillatorStream {
    pub fn try_new(
        params: ForwardBackwardExponentialOscillatorParams,
    ) -> Result<Self, ForwardBackwardExponentialOscillatorError> {
        let length = params.length.unwrap_or(DEFAULT_LENGTH);
        let smooth = params.smooth.unwrap_or(DEFAULT_SMOOTH);
        if length == 0 {
            return Err(ForwardBackwardExponentialOscillatorError::InvalidLength {
                length,
                data_len: 0,
            });
        }
        if smooth == 0 {
            return Err(ForwardBackwardExponentialOscillatorError::InvalidSmooth {
                smooth,
                data_len: 0,
            });
        }
        Ok(Self {
            length,
            alpha: 2.0 / (smooth as f64 + 1.0),
            ema1_state: None,
            ema2_state: None,
            prev_ema2: None,
            ema1_window: VecDeque::with_capacity(length.max(1)),
            diff_window: RollingDiffWindow::new(length),
        })
    }

    #[inline(always)]
    pub fn update(
        &mut self,
        value: f64,
    ) -> Option<ForwardBackwardExponentialOscillatorStreamOutput> {
        if !value.is_finite() {
            self.ema1_state = None;
            self.ema2_state = None;
            self.prev_ema2 = None;
            self.ema1_window.clear();
            self.diff_window.clear();
            return None;
        }

        let ema1 = ema_step(&mut self.ema1_state, value, self.alpha);
        push_window(&mut self.ema1_window, self.length, ema1);
        let forward_backward = if self.ema1_window.len() == self.length {
            compute_forward_backward_value(&self.ema1_window, self.alpha)
        } else {
            f64::NAN
        };

        let ema2 = ema_step(&mut self.ema2_state, ema1, self.alpha);
        let mut backward = f64::NAN;
        if let Some(last_ema2) = self.prev_ema2 {
            if let Some((num, den)) = self.diff_window.update(ema2 - last_ema2) {
                if den != 0.0 {
                    backward = num / den * 50.0 + 50.0;
                }
            }
        }
        self.prev_ema2 = Some(ema2);

        if !forward_backward.is_finite() && !backward.is_finite() {
            return None;
        }
        let histogram = if forward_backward.is_finite() && backward.is_finite() {
            (forward_backward - backward) * 0.25 + 50.0
        } else {
            f64::NAN
        };

        Some(ForwardBackwardExponentialOscillatorStreamOutput {
            forward_backward,
            backward,
            histogram,
        })
    }
}

#[inline(always)]
unsafe fn mu_slice_as_f64_slice_mut(buf: &mut ManuallyDrop<Vec<MaybeUninit<f64>>>) -> &mut [f64] {
    core::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut f64, buf.len())
}

#[inline(always)]
unsafe fn vec_f64_from_mu_guard(buf: ManuallyDrop<Vec<MaybeUninit<f64>>>) -> Vec<f64> {
    let mut buf = buf;
    Vec::from_raw_parts(buf.as_mut_ptr() as *mut f64, buf.len(), buf.capacity())
}

#[cfg(feature = "python")]
#[pyfunction(name = "forward_backward_exponential_oscillator")]
#[pyo3(signature = (data, length=DEFAULT_LENGTH, smooth=DEFAULT_SMOOTH, kernel=None))]
pub fn forward_backward_exponential_oscillator_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length: usize,
    smooth: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = ForwardBackwardExponentialOscillatorInput::from_slice(
        data,
        ForwardBackwardExponentialOscillatorParams {
            length: Some(length),
            smooth: Some(smooth),
        },
    );
    let output = py
        .allow_threads(|| forward_backward_exponential_oscillator_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dict = PyDict::new(py);
    dict.set_item("forward_backward", output.forward_backward.into_pyarray(py))?;
    dict.set_item("backward", output.backward.into_pyarray(py))?;
    dict.set_item("histogram", output.histogram.into_pyarray(py))?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyfunction(name = "forward_backward_exponential_oscillator_batch")]
#[pyo3(signature = (data, length_range, smooth_range, kernel=None))]
pub fn forward_backward_exponential_oscillator_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    smooth_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let kernel = validate_kernel(kernel, true)?;
    let output = py
        .allow_threads(|| {
            forward_backward_exponential_oscillator_batch_with_kernel(
                data,
                &ForwardBackwardExponentialOscillatorBatchRange {
                    length: length_range,
                    smooth: smooth_range,
                },
                kernel,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let total = output.rows * output.cols;
    let arrays = [
        unsafe { PyArray1::<f64>::new(py, [total], false) },
        unsafe { PyArray1::<f64>::new(py, [total], false) },
        unsafe { PyArray1::<f64>::new(py, [total], false) },
    ];
    unsafe { arrays[0].as_slice_mut()? }.copy_from_slice(&output.forward_backward);
    unsafe { arrays[1].as_slice_mut()? }.copy_from_slice(&output.backward);
    unsafe { arrays[2].as_slice_mut()? }.copy_from_slice(&output.histogram);

    let dict = PyDict::new(py);
    dict.set_item(
        "forward_backward",
        arrays[0].reshape((output.rows, output.cols))?,
    )?;
    dict.set_item("backward", arrays[1].reshape((output.rows, output.cols))?)?;
    dict.set_item("histogram", arrays[2].reshape((output.rows, output.cols))?)?;
    dict.set_item(
        "lengths",
        output
            .combos
            .iter()
            .map(|combo| combo.length.unwrap_or(DEFAULT_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "smooths",
        output
            .combos
            .iter()
            .map(|combo| combo.smooth.unwrap_or(DEFAULT_SMOOTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", output.rows)?;
    dict.set_item("cols", output.cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "ForwardBackwardExponentialOscillatorStream")]
pub struct ForwardBackwardExponentialOscillatorStreamPy {
    stream: ForwardBackwardExponentialOscillatorStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl ForwardBackwardExponentialOscillatorStreamPy {
    #[new]
    #[pyo3(signature = (length=DEFAULT_LENGTH, smooth=DEFAULT_SMOOTH))]
    fn new(length: usize, smooth: usize) -> PyResult<Self> {
        let stream = ForwardBackwardExponentialOscillatorStream::try_new(
            ForwardBackwardExponentialOscillatorParams {
                length: Some(length),
                smooth: Some(smooth),
            },
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, value: f64) -> Option<(f64, f64, f64)> {
        self.stream
            .update(value)
            .map(|output| (output.forward_backward, output.backward, output.histogram))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct ForwardBackwardExponentialOscillatorJsOutput {
    pub forward_backward: Vec<f64>,
    pub backward: Vec<f64>,
    pub histogram: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = forward_backward_exponential_oscillator_js)]
pub fn forward_backward_exponential_oscillator_js(
    data: &[f64],
    length: usize,
    smooth: usize,
) -> Result<JsValue, JsValue> {
    let input = ForwardBackwardExponentialOscillatorInput::from_slice(
        data,
        ForwardBackwardExponentialOscillatorParams {
            length: Some(length),
            smooth: Some(smooth),
        },
    );
    let output = forward_backward_exponential_oscillator_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&ForwardBackwardExponentialOscillatorJsOutput {
        forward_backward: output.forward_backward,
        backward: output.backward,
        histogram: output.histogram,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct ForwardBackwardExponentialOscillatorBatchConfig {
    pub length_range: (usize, usize, usize),
    pub smooth_range: (usize, usize, usize),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct ForwardBackwardExponentialOscillatorBatchJsOutput {
    pub forward_backward: Vec<f64>,
    pub backward: Vec<f64>,
    pub histogram: Vec<f64>,
    pub lengths: Vec<usize>,
    pub smooths: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = forward_backward_exponential_oscillator_batch)]
pub fn forward_backward_exponential_oscillator_batch_js(
    data: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let cfg: ForwardBackwardExponentialOscillatorBatchConfig =
        serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let output = forward_backward_exponential_oscillator_batch_with_kernel(
        data,
        &ForwardBackwardExponentialOscillatorBatchRange {
            length: cfg.length_range,
            smooth: cfg.smooth_range,
        },
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    serde_wasm_bindgen::to_value(&ForwardBackwardExponentialOscillatorBatchJsOutput {
        forward_backward: output.forward_backward,
        backward: output.backward,
        histogram: output.histogram,
        lengths: output
            .combos
            .iter()
            .map(|combo| combo.length.unwrap_or(DEFAULT_LENGTH))
            .collect(),
        smooths: output
            .combos
            .iter()
            .map(|combo| combo.smooth.unwrap_or(DEFAULT_SMOOTH))
            .collect(),
        rows: output.rows,
        cols: output.cols,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn forward_backward_exponential_oscillator_output_into_js(
    data: &[f64],
    length: usize,
    smooth: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = forward_backward_exponential_oscillator_js(data, length, smooth)?;
    crate::write_wasm_object_f64_outputs(
        "forward_backward_exponential_oscillator_output_into_js",
        &value,
        out,
    )
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn forward_backward_exponential_oscillator_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = forward_backward_exponential_oscillator_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "forward_backward_exponential_oscillator_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data() -> Vec<f64> {
        vec![
            100.0, 101.0, 102.5, 101.8, 103.2, 104.1, 103.7, 105.4, 106.2, 105.8, 107.1, 108.0,
            107.6, 108.8, 109.4, 108.9, 110.3, 111.2, 110.7, 112.0, 112.6, 112.1, 113.4, 114.0,
            113.5, 114.7, 115.1, 114.8, 116.0, 116.4, 116.1, 117.0, 117.8, 117.2, 118.4, 119.1,
            118.6, 119.7, 120.2, 119.8, 121.0, 121.6, 121.1, 122.3, 123.0, 122.4, 123.5, 124.1,
        ]
    }

    #[test]
    fn forward_backward_exponential_oscillator_into_matches_single() {
        let data = sample_data();
        let input = ForwardBackwardExponentialOscillatorInput::from_slice(
            &data,
            ForwardBackwardExponentialOscillatorParams {
                length: Some(20),
                smooth: Some(10),
            },
        );
        let out = forward_backward_exponential_oscillator_with_kernel(&input, Kernel::Scalar)
            .expect("single");
        let mut fb = vec![0.0; data.len()];
        let mut bw = vec![0.0; data.len()];
        let mut hist = vec![0.0; data.len()];
        forward_backward_exponential_oscillator_into_slices(
            &input,
            Kernel::Scalar,
            &mut fb,
            &mut bw,
            &mut hist,
        )
        .expect("into");

        for i in 0..data.len() {
            if out.forward_backward[i].is_nan() {
                assert!(fb[i].is_nan());
            } else {
                assert!((out.forward_backward[i] - fb[i]).abs() <= 1e-12);
            }
            if out.backward[i].is_nan() {
                assert!(bw[i].is_nan());
            } else {
                assert!((out.backward[i] - bw[i]).abs() <= 1e-12);
            }
            if out.histogram[i].is_nan() {
                assert!(hist[i].is_nan());
            } else {
                assert!((out.histogram[i] - hist[i]).abs() <= 1e-12);
            }
        }
    }

    #[test]
    fn forward_backward_exponential_oscillator_stream_matches_batch_points() {
        let data = sample_data();
        let params = ForwardBackwardExponentialOscillatorParams {
            length: Some(20),
            smooth: Some(10),
        };
        let input = ForwardBackwardExponentialOscillatorInput::from_slice(&data, params.clone());
        let batch = forward_backward_exponential_oscillator(&input).expect("batch");
        let mut stream =
            ForwardBackwardExponentialOscillatorStream::try_new(params).expect("stream");

        for i in 0..data.len() {
            let point = stream.update(data[i]);
            if let Some(point) = point {
                if batch.forward_backward[i].is_nan() {
                    assert!(point.forward_backward.is_nan());
                } else {
                    assert!((point.forward_backward - batch.forward_backward[i]).abs() <= 1e-12);
                }
                if batch.backward[i].is_nan() {
                    assert!(point.backward.is_nan());
                } else {
                    assert!((point.backward - batch.backward[i]).abs() <= 1e-12);
                }
                if batch.histogram[i].is_nan() {
                    assert!(point.histogram.is_nan());
                } else {
                    assert!((point.histogram - batch.histogram[i]).abs() <= 1e-12);
                }
            } else {
                assert!(batch.forward_backward[i].is_nan());
                assert!(batch.backward[i].is_nan());
                assert!(batch.histogram[i].is_nan());
            }
        }
    }

    #[test]
    fn forward_backward_exponential_oscillator_batch_first_row_matches_single() {
        let data = sample_data();
        let batch = forward_backward_exponential_oscillator_batch_with_kernel(
            &data,
            &ForwardBackwardExponentialOscillatorBatchRange {
                length: (20, 22, 2),
                smooth: (10, 12, 2),
            },
            Kernel::ScalarBatch,
        )
        .expect("batch");
        assert_eq!(batch.rows, 4);
        assert_eq!(batch.cols, data.len());

        let single = forward_backward_exponential_oscillator(
            &ForwardBackwardExponentialOscillatorInput::from_slice(
                &data,
                ForwardBackwardExponentialOscillatorParams {
                    length: Some(20),
                    smooth: Some(10),
                },
            ),
        )
        .expect("single");

        for i in 0..data.len() {
            let batch_fb = batch.forward_backward[i];
            let batch_bw = batch.backward[i];
            let batch_hist = batch.histogram[i];
            if single.forward_backward[i].is_nan() {
                assert!(batch_fb.is_nan());
            } else {
                assert!((single.forward_backward[i] - batch_fb).abs() <= 1e-12);
            }
            if single.backward[i].is_nan() {
                assert!(batch_bw.is_nan());
            } else {
                assert!((single.backward[i] - batch_bw).abs() <= 1e-12);
            }
            if single.histogram[i].is_nan() {
                assert!(batch_hist.is_nan());
            } else {
                assert!((single.histogram[i] - batch_hist).abs() <= 1e-12);
            }
        }
    }

    #[test]
    fn forward_backward_exponential_oscillator_rejects_invalid_inputs() {
        let data = sample_data();
        let err = forward_backward_exponential_oscillator(
            &ForwardBackwardExponentialOscillatorInput::from_slice(
                &data,
                ForwardBackwardExponentialOscillatorParams {
                    length: Some(0),
                    smooth: Some(10),
                },
            ),
        )
        .expect_err("invalid length");
        assert!(matches!(
            err,
            ForwardBackwardExponentialOscillatorError::InvalidLength { .. }
        ));

        let err = forward_backward_exponential_oscillator(
            &ForwardBackwardExponentialOscillatorInput::from_slice(
                &data,
                ForwardBackwardExponentialOscillatorParams {
                    length: Some(20),
                    smooth: Some(0),
                },
            ),
        )
        .expect_err("invalid smooth");
        assert!(matches!(
            err,
            ForwardBackwardExponentialOscillatorError::InvalidSmooth { .. }
        ));
    }
}
