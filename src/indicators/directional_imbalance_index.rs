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

use crate::utilities::data_loader::Candles;
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::collections::VecDeque;
use std::error::Error;
use thiserror::Error;

const DEFAULT_LENGTH: usize = 10;
const DEFAULT_PERIOD: usize = 70;

#[derive(Debug, Clone)]
pub enum DirectionalImbalanceIndexData<'a> {
    Candles { candles: &'a Candles },
    Slices { high: &'a [f64], low: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct DirectionalImbalanceIndexOutput {
    pub up: Vec<f64>,
    pub down: Vec<f64>,
    pub bulls: Vec<f64>,
    pub bears: Vec<f64>,
    pub upper: Vec<f64>,
    pub lower: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectionalImbalanceIndexOutputField {
    Up,
    Down,
    Bulls,
    Bears,
    Upper,
    Lower,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct DirectionalImbalanceIndexParams {
    pub length: Option<usize>,
    pub period: Option<usize>,
}

impl Default for DirectionalImbalanceIndexParams {
    fn default() -> Self {
        Self {
            length: Some(DEFAULT_LENGTH),
            period: Some(DEFAULT_PERIOD),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DirectionalImbalanceIndexInput<'a> {
    pub data: DirectionalImbalanceIndexData<'a>,
    pub params: DirectionalImbalanceIndexParams,
}

impl<'a> DirectionalImbalanceIndexInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: DirectionalImbalanceIndexParams) -> Self {
        Self {
            data: DirectionalImbalanceIndexData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        params: DirectionalImbalanceIndexParams,
    ) -> Self {
        Self {
            data: DirectionalImbalanceIndexData::Slices { high, low },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, DirectionalImbalanceIndexParams::default())
    }

    #[inline(always)]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(DEFAULT_LENGTH)
    }

    #[inline(always)]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(DEFAULT_PERIOD)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct DirectionalImbalanceIndexBuilder {
    length: Option<usize>,
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for DirectionalImbalanceIndexBuilder {
    fn default() -> Self {
        Self {
            length: None,
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl DirectionalImbalanceIndexBuilder {
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
    pub fn period(mut self, value: usize) -> Self {
        self.period = Some(value);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline(always)]
    fn build_params(self) -> DirectionalImbalanceIndexParams {
        DirectionalImbalanceIndexParams {
            length: self.length,
            period: self.period,
        }
    }

    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<DirectionalImbalanceIndexOutput, DirectionalImbalanceIndexError> {
        directional_imbalance_index_with_kernel(
            &DirectionalImbalanceIndexInput::from_candles(candles, self.build_params()),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
    ) -> Result<DirectionalImbalanceIndexOutput, DirectionalImbalanceIndexError> {
        directional_imbalance_index_with_kernel(
            &DirectionalImbalanceIndexInput::from_slices(high, low, self.build_params()),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<DirectionalImbalanceIndexStream, DirectionalImbalanceIndexError> {
        DirectionalImbalanceIndexStream::try_new(self.build_params())
    }
}

#[derive(Debug, Error)]
pub enum DirectionalImbalanceIndexError {
    #[error("directional_imbalance_index: Input data slice is empty.")]
    EmptyInputData,
    #[error(
        "directional_imbalance_index: Input length mismatch: high = {high_len}, low = {low_len}"
    )]
    InputLengthMismatch { high_len: usize, low_len: usize },
    #[error("directional_imbalance_index: All values are NaN.")]
    AllValuesNaN,
    #[error("directional_imbalance_index: Invalid length: {length}")]
    InvalidLength { length: usize },
    #[error("directional_imbalance_index: Invalid period: {period}")]
    InvalidPeriod { period: usize },
    #[error(
        "directional_imbalance_index: Output length mismatch: expected = {expected}, got = {got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error(
        "directional_imbalance_index: Invalid range: {field} start={start} end={end} step={step}"
    )]
    InvalidRange {
        field: &'static str,
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("directional_imbalance_index: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error(
        "directional_imbalance_index: Output length mismatch: dst = {dst_len}, expected = {expected_len}"
    )]
    MismatchedOutputLen { dst_len: usize, expected_len: usize },
    #[error("directional_imbalance_index: Invalid input: {msg}")]
    InvalidInput { msg: String },
}

#[derive(Debug, Clone, Copy)]
struct DirectionalImbalanceIndexResolved {
    length: usize,
    period: usize,
}

#[derive(Debug, Clone)]
pub struct DirectionalImbalanceIndexPoint {
    pub up: f64,
    pub down: f64,
    pub bulls: f64,
    pub bears: f64,
    pub upper: f64,
    pub lower: f64,
}

#[derive(Debug, Clone)]
pub struct DirectionalImbalanceIndexStream {
    length: usize,
    period: usize,
    index: usize,
    max_deque: VecDeque<(usize, f64)>,
    min_deque: VecDeque<(usize, f64)>,
    up_hits: VecDeque<f64>,
    down_hits: VecDeque<f64>,
    up_sum: f64,
    down_sum: f64,
}

impl DirectionalImbalanceIndexStream {
    #[inline(always)]
    pub fn try_new(
        params: DirectionalImbalanceIndexParams,
    ) -> Result<Self, DirectionalImbalanceIndexError> {
        let resolved = resolve_params(&params)?;
        Ok(Self {
            length: resolved.length,
            period: resolved.period,
            index: 0,
            max_deque: VecDeque::with_capacity(resolved.length.saturating_add(1)),
            min_deque: VecDeque::with_capacity(resolved.length.saturating_add(1)),
            up_hits: VecDeque::with_capacity(resolved.period),
            down_hits: VecDeque::with_capacity(resolved.period),
            up_sum: 0.0,
            down_sum: 0.0,
        })
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.index = 0;
        self.max_deque.clear();
        self.min_deque.clear();
        self.up_hits.clear();
        self.down_hits.clear();
        self.up_sum = 0.0;
        self.down_sum = 0.0;
    }

    #[inline(always)]
    fn trim_deques(&mut self) {
        let min_index = self.index.saturating_sub(self.length);
        while let Some(&(idx, _)) = self.max_deque.front() {
            if idx < min_index {
                self.max_deque.pop_front();
            } else {
                break;
            }
        }
        while let Some(&(idx, _)) = self.min_deque.front() {
            if idx < min_index {
                self.min_deque.pop_front();
            } else {
                break;
            }
        }
    }

    #[inline(always)]
    fn push_high(&mut self, high: f64) {
        while let Some((_, value)) = self.max_deque.back() {
            if *value <= high {
                self.max_deque.pop_back();
            } else {
                break;
            }
        }
        self.max_deque.push_back((self.index, high));
    }

    #[inline(always)]
    fn push_low(&mut self, low: f64) {
        while let Some((_, value)) = self.min_deque.back() {
            if *value >= low {
                self.min_deque.pop_back();
            } else {
                break;
            }
        }
        self.min_deque.push_back((self.index, low));
    }

    #[inline(always)]
    fn push_hit(queue: &mut VecDeque<f64>, sum: &mut f64, value: f64, period: usize) {
        queue.push_back(value);
        *sum += value;
        if queue.len() > period {
            if let Some(evicted) = queue.pop_front() {
                *sum -= evicted;
            }
        }
    }

    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64) -> Option<DirectionalImbalanceIndexPoint> {
        if !is_valid_pair(high, low) {
            self.reset();
            return None;
        }

        self.push_high(high);
        self.push_low(low);
        self.trim_deques();

        let upper = self.max_deque.front().map(|&(_, value)| value)?;
        let lower = self.min_deque.front().map(|&(_, value)| value)?;

        let up_hit = if high == upper { 1.0 } else { 0.0 };
        let down_hit = if low == lower { 1.0 } else { 0.0 };
        Self::push_hit(&mut self.up_hits, &mut self.up_sum, up_hit, self.period);
        Self::push_hit(
            &mut self.down_hits,
            &mut self.down_sum,
            down_hit,
            self.period,
        );

        let up = self.up_sum;
        let down = self.down_sum;
        let total = up + down;
        let (bulls, bears) = if total > 0.0 {
            ((up / total) * 100.0, (down / total) * 100.0)
        } else {
            (f64::NAN, f64::NAN)
        };

        self.index = self.index.saturating_add(1);

        Some(DirectionalImbalanceIndexPoint {
            up,
            down,
            bulls,
            bears,
            upper,
            lower,
        })
    }

    #[inline(always)]
    pub fn get_warmup_period(&self) -> usize {
        0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectionalImbalanceIndexBatchRange {
    pub length: (usize, usize, usize),
    pub period: (usize, usize, usize),
}

impl Default for DirectionalImbalanceIndexBatchRange {
    fn default() -> Self {
        Self {
            length: (DEFAULT_LENGTH, DEFAULT_LENGTH, 0),
            period: (DEFAULT_PERIOD, DEFAULT_PERIOD, 0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DirectionalImbalanceIndexBatchOutput {
    pub up: Vec<f64>,
    pub down: Vec<f64>,
    pub bulls: Vec<f64>,
    pub bears: Vec<f64>,
    pub upper: Vec<f64>,
    pub lower: Vec<f64>,
    pub combos: Vec<DirectionalImbalanceIndexParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct DirectionalImbalanceIndexBatchBuilder {
    range: DirectionalImbalanceIndexBatchRange,
    kernel: Kernel,
}

impl Default for DirectionalImbalanceIndexBatchBuilder {
    fn default() -> Self {
        Self {
            range: DirectionalImbalanceIndexBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl DirectionalImbalanceIndexBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn length_range(mut self, value: (usize, usize, usize)) -> Self {
        self.range.length = value;
        self
    }

    #[inline(always)]
    pub fn period_range(mut self, value: (usize, usize, usize)) -> Self {
        self.range.period = value;
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
    ) -> Result<DirectionalImbalanceIndexBatchOutput, DirectionalImbalanceIndexError> {
        directional_imbalance_index_batch_with_kernel(
            high,
            low,
            &self.range,
            &DirectionalImbalanceIndexParams {
                length: None,
                period: None,
            },
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<DirectionalImbalanceIndexBatchOutput, DirectionalImbalanceIndexError> {
        directional_imbalance_index_batch_from_input_with_kernel(
            &DirectionalImbalanceIndexInput::from_candles(
                candles,
                DirectionalImbalanceIndexParams {
                    length: None,
                    period: None,
                },
            ),
            &self.range,
            self.kernel,
        )
    }
}

#[inline(always)]
fn resolve_params(
    params: &DirectionalImbalanceIndexParams,
) -> Result<DirectionalImbalanceIndexResolved, DirectionalImbalanceIndexError> {
    let length = params.length.unwrap_or(DEFAULT_LENGTH);
    if length == 0 {
        return Err(DirectionalImbalanceIndexError::InvalidLength { length });
    }
    let period = params.period.unwrap_or(DEFAULT_PERIOD);
    if period == 0 {
        return Err(DirectionalImbalanceIndexError::InvalidPeriod { period });
    }
    Ok(DirectionalImbalanceIndexResolved { length, period })
}

#[inline(always)]
fn input_slices<'a>(
    input: &'a DirectionalImbalanceIndexInput<'a>,
) -> Result<(&'a [f64], &'a [f64]), DirectionalImbalanceIndexError> {
    match &input.data {
        DirectionalImbalanceIndexData::Candles { candles } => {
            Ok((candles.high.as_slice(), candles.low.as_slice()))
        }
        DirectionalImbalanceIndexData::Slices { high, low } => Ok((high, low)),
    }
}

#[inline(always)]
fn is_valid_pair(high: f64, low: f64) -> bool {
    high.is_finite() && low.is_finite()
}

#[inline(always)]
fn any_valid_pair(high: &[f64], low: &[f64]) -> bool {
    high.iter()
        .zip(low.iter())
        .any(|(&h, &l)| is_valid_pair(h, l))
}

fn validate_common(
    high: &[f64],
    low: &[f64],
    params: &DirectionalImbalanceIndexParams,
) -> Result<DirectionalImbalanceIndexResolved, DirectionalImbalanceIndexError> {
    if high.is_empty() || low.is_empty() {
        return Err(DirectionalImbalanceIndexError::EmptyInputData);
    }
    if high.len() != low.len() {
        return Err(DirectionalImbalanceIndexError::InputLengthMismatch {
            high_len: high.len(),
            low_len: low.len(),
        });
    }
    let resolved = resolve_params(params)?;
    if !any_valid_pair(high, low) {
        return Err(DirectionalImbalanceIndexError::AllValuesNaN);
    }
    Ok(resolved)
}

#[inline(always)]
fn write_point(
    i: usize,
    point: Option<DirectionalImbalanceIndexPoint>,
    out_up: &mut [f64],
    out_down: &mut [f64],
    out_bulls: &mut [f64],
    out_bears: &mut [f64],
    out_upper: &mut [f64],
    out_lower: &mut [f64],
) {
    if let Some(point) = point {
        out_up[i] = point.up;
        out_down[i] = point.down;
        out_bulls[i] = point.bulls;
        out_bears[i] = point.bears;
        out_upper[i] = point.upper;
        out_lower[i] = point.lower;
    } else {
        out_up[i] = f64::NAN;
        out_down[i] = f64::NAN;
        out_bulls[i] = f64::NAN;
        out_bears[i] = f64::NAN;
        out_upper[i] = f64::NAN;
        out_lower[i] = f64::NAN;
    }
}

fn compute_into(
    high: &[f64],
    low: &[f64],
    resolved: DirectionalImbalanceIndexResolved,
    out_up: &mut [f64],
    out_down: &mut [f64],
    out_bulls: &mut [f64],
    out_bears: &mut [f64],
    out_upper: &mut [f64],
    out_lower: &mut [f64],
) {
    let mut stream = DirectionalImbalanceIndexStream::try_new(DirectionalImbalanceIndexParams {
        length: Some(resolved.length),
        period: Some(resolved.period),
    })
    .expect("validated params");

    for i in 0..high.len() {
        let point = stream.update(high[i], low[i]);
        write_point(
            i, point, out_up, out_down, out_bulls, out_bears, out_upper, out_lower,
        );
    }
}

fn compute_selected_into(
    high: &[f64],
    low: &[f64],
    resolved: DirectionalImbalanceIndexResolved,
    field: DirectionalImbalanceIndexOutputField,
    out: &mut [f64],
) {
    let mut stream = DirectionalImbalanceIndexStream::try_new(DirectionalImbalanceIndexParams {
        length: Some(resolved.length),
        period: Some(resolved.period),
    })
    .expect("validated params");

    for i in 0..high.len() {
        out[i] = match stream.update(high[i], low[i]) {
            Some(point) => match field {
                DirectionalImbalanceIndexOutputField::Up => point.up,
                DirectionalImbalanceIndexOutputField::Down => point.down,
                DirectionalImbalanceIndexOutputField::Bulls => point.bulls,
                DirectionalImbalanceIndexOutputField::Bears => point.bears,
                DirectionalImbalanceIndexOutputField::Upper => point.upper,
                DirectionalImbalanceIndexOutputField::Lower => point.lower,
            },
            None => f64::NAN,
        };
    }
}

#[inline]
pub fn directional_imbalance_index(
    input: &DirectionalImbalanceIndexInput,
) -> Result<DirectionalImbalanceIndexOutput, DirectionalImbalanceIndexError> {
    directional_imbalance_index_with_kernel(input, Kernel::Auto)
}

pub fn directional_imbalance_index_with_kernel(
    input: &DirectionalImbalanceIndexInput,
    kernel: Kernel,
) -> Result<DirectionalImbalanceIndexOutput, DirectionalImbalanceIndexError> {
    let (high, low) = input_slices(input)?;
    let resolved = validate_common(high, low, &input.params)?;
    let len = high.len();
    let _chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let mut up = alloc_with_nan_prefix(len, 0);
    let mut down = alloc_with_nan_prefix(len, 0);
    let mut bulls = alloc_with_nan_prefix(len, 0);
    let mut bears = alloc_with_nan_prefix(len, 0);
    let mut upper = alloc_with_nan_prefix(len, 0);
    let mut lower = alloc_with_nan_prefix(len, 0);

    compute_into(
        high, low, resolved, &mut up, &mut down, &mut bulls, &mut bears, &mut upper, &mut lower,
    );

    Ok(DirectionalImbalanceIndexOutput {
        up,
        down,
        bulls,
        bears,
        upper,
        lower,
    })
}

pub fn directional_imbalance_index_into_slice(
    out_up: &mut [f64],
    out_down: &mut [f64],
    out_bulls: &mut [f64],
    out_bears: &mut [f64],
    out_upper: &mut [f64],
    out_lower: &mut [f64],
    input: &DirectionalImbalanceIndexInput,
    kernel: Kernel,
) -> Result<(), DirectionalImbalanceIndexError> {
    let (high, low) = input_slices(input)?;
    let resolved = validate_common(high, low, &input.params)?;
    let len = high.len();
    for dst in [
        &*out_up,
        &*out_down,
        &*out_bulls,
        &*out_bears,
        &*out_upper,
        &*out_lower,
    ] {
        if dst.len() != len {
            return Err(DirectionalImbalanceIndexError::OutputLengthMismatch {
                expected: len,
                got: dst.len(),
            });
        }
    }

    let _chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    compute_into(
        high, low, resolved, out_up, out_down, out_bulls, out_bears, out_upper, out_lower,
    );
    Ok(())
}

pub fn directional_imbalance_index_output_into_slice(
    out: &mut [f64],
    input: &DirectionalImbalanceIndexInput,
    kernel: Kernel,
    field: DirectionalImbalanceIndexOutputField,
) -> Result<(), DirectionalImbalanceIndexError> {
    let (high, low) = input_slices(input)?;
    let resolved = validate_common(high, low, &input.params)?;
    let len = high.len();
    if out.len() != len {
        return Err(DirectionalImbalanceIndexError::OutputLengthMismatch {
            expected: len,
            got: out.len(),
        });
    }

    let _chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    compute_selected_into(high, low, resolved, field, out);
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn directional_imbalance_index_into(
    out_up: &mut [f64],
    out_down: &mut [f64],
    out_bulls: &mut [f64],
    out_bears: &mut [f64],
    out_upper: &mut [f64],
    out_lower: &mut [f64],
    input: &DirectionalImbalanceIndexInput,
) -> Result<(), DirectionalImbalanceIndexError> {
    directional_imbalance_index_into_slice(
        out_up,
        out_down,
        out_bulls,
        out_bears,
        out_upper,
        out_lower,
        input,
        Kernel::Auto,
    )
}

#[inline(always)]
fn expand_axis(
    field: &'static str,
    range: (usize, usize, usize),
) -> Result<Vec<usize>, DirectionalImbalanceIndexError> {
    let (start, end, step) = range;
    if start == 0 {
        return Err(DirectionalImbalanceIndexError::InvalidRange {
            field,
            start,
            end,
            step,
        });
    }
    if step == 0 {
        return Ok(vec![start]);
    }
    if start > end {
        return Err(DirectionalImbalanceIndexError::InvalidRange {
            field,
            start,
            end,
            step,
        });
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
                .ok_or_else(|| DirectionalImbalanceIndexError::InvalidInput {
                    msg: format!("directional_imbalance_index: {field} range step overflow"),
                })?;
        if next <= cur {
            return Err(DirectionalImbalanceIndexError::InvalidRange {
                field,
                start,
                end,
                step,
            });
        }
        cur = next.min(end);
    }
    Ok(out)
}

fn expand_grid_checked(
    sweep: &DirectionalImbalanceIndexBatchRange,
    _fixed: &DirectionalImbalanceIndexParams,
) -> Result<Vec<DirectionalImbalanceIndexParams>, DirectionalImbalanceIndexError> {
    let lengths = expand_axis("length", sweep.length)?;
    let periods = expand_axis("period", sweep.period)?;
    let total = lengths.len().checked_mul(periods.len()).ok_or_else(|| {
        DirectionalImbalanceIndexError::InvalidInput {
            msg: "directional_imbalance_index: parameter grid size overflow".to_string(),
        }
    })?;
    let mut combos = Vec::with_capacity(total);
    for &length in &lengths {
        for &period in &periods {
            combos.push(DirectionalImbalanceIndexParams {
                length: Some(length),
                period: Some(period),
            });
        }
    }
    Ok(combos)
}

pub fn expand_grid_directional_imbalance_index(
    sweep: &DirectionalImbalanceIndexBatchRange,
    fixed: &DirectionalImbalanceIndexParams,
) -> Result<Vec<DirectionalImbalanceIndexParams>, DirectionalImbalanceIndexError> {
    expand_grid_checked(sweep, fixed)
}

pub fn directional_imbalance_index_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    sweep: &DirectionalImbalanceIndexBatchRange,
    fixed: &DirectionalImbalanceIndexParams,
    kernel: Kernel,
) -> Result<DirectionalImbalanceIndexBatchOutput, DirectionalImbalanceIndexError> {
    let input = DirectionalImbalanceIndexInput::from_slices(high, low, fixed.clone());
    directional_imbalance_index_batch_from_input_with_kernel(&input, sweep, kernel)
}

pub fn directional_imbalance_index_batch_from_input_with_kernel(
    input: &DirectionalImbalanceIndexInput,
    sweep: &DirectionalImbalanceIndexBatchRange,
    kernel: Kernel,
) -> Result<DirectionalImbalanceIndexBatchOutput, DirectionalImbalanceIndexError> {
    directional_imbalance_index_batch_inner(input, sweep, kernel, true)
}

pub fn directional_imbalance_index_batch_slice(
    high: &[f64],
    low: &[f64],
    sweep: &DirectionalImbalanceIndexBatchRange,
    fixed: &DirectionalImbalanceIndexParams,
    kernel: Kernel,
) -> Result<DirectionalImbalanceIndexBatchOutput, DirectionalImbalanceIndexError> {
    let input = DirectionalImbalanceIndexInput::from_slices(high, low, fixed.clone());
    directional_imbalance_index_batch_inner(&input, sweep, kernel, false)
}

pub fn directional_imbalance_index_batch_par_slice(
    high: &[f64],
    low: &[f64],
    sweep: &DirectionalImbalanceIndexBatchRange,
    fixed: &DirectionalImbalanceIndexParams,
    kernel: Kernel,
) -> Result<DirectionalImbalanceIndexBatchOutput, DirectionalImbalanceIndexError> {
    let input = DirectionalImbalanceIndexInput::from_slices(high, low, fixed.clone());
    directional_imbalance_index_batch_inner(&input, sweep, kernel, true)
}

fn directional_imbalance_index_batch_inner(
    input: &DirectionalImbalanceIndexInput,
    sweep: &DirectionalImbalanceIndexBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<DirectionalImbalanceIndexBatchOutput, DirectionalImbalanceIndexError> {
    let (high, low) = input_slices(input)?;
    validate_common(high, low, &input.params)?;
    let combos = expand_grid_checked(sweep, &input.params)?;
    let rows = combos.len();
    let cols = high.len();
    let total =
        rows.checked_mul(cols)
            .ok_or_else(|| DirectionalImbalanceIndexError::InvalidInput {
                msg: "directional_imbalance_index: rows*cols overflow in batch".to_string(),
            })?;

    let warmups = vec![0usize; rows];
    let mut up_mu = make_uninit_matrix(rows, cols);
    let mut down_mu = make_uninit_matrix(rows, cols);
    let mut bulls_mu = make_uninit_matrix(rows, cols);
    let mut bears_mu = make_uninit_matrix(rows, cols);
    let mut upper_mu = make_uninit_matrix(rows, cols);
    let mut lower_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut up_mu, cols, &warmups);
    init_matrix_prefixes(&mut down_mu, cols, &warmups);
    init_matrix_prefixes(&mut bulls_mu, cols, &warmups);
    init_matrix_prefixes(&mut bears_mu, cols, &warmups);
    init_matrix_prefixes(&mut upper_mu, cols, &warmups);
    init_matrix_prefixes(&mut lower_mu, cols, &warmups);

    let mut up = unsafe {
        Vec::from_raw_parts(
            up_mu.as_mut_ptr() as *mut f64,
            up_mu.len(),
            up_mu.capacity(),
        )
    };
    let mut down = unsafe {
        Vec::from_raw_parts(
            down_mu.as_mut_ptr() as *mut f64,
            down_mu.len(),
            down_mu.capacity(),
        )
    };
    let mut bulls = unsafe {
        Vec::from_raw_parts(
            bulls_mu.as_mut_ptr() as *mut f64,
            bulls_mu.len(),
            bulls_mu.capacity(),
        )
    };
    let mut bears = unsafe {
        Vec::from_raw_parts(
            bears_mu.as_mut_ptr() as *mut f64,
            bears_mu.len(),
            bears_mu.capacity(),
        )
    };
    let mut upper = unsafe {
        Vec::from_raw_parts(
            upper_mu.as_mut_ptr() as *mut f64,
            upper_mu.len(),
            upper_mu.capacity(),
        )
    };
    let mut lower = unsafe {
        Vec::from_raw_parts(
            lower_mu.as_mut_ptr() as *mut f64,
            lower_mu.len(),
            lower_mu.capacity(),
        )
    };
    std::mem::forget(up_mu);
    std::mem::forget(down_mu);
    std::mem::forget(bulls_mu);
    std::mem::forget(bears_mu);
    std::mem::forget(upper_mu);
    std::mem::forget(lower_mu);
    debug_assert_eq!(up.len(), total);
    debug_assert_eq!(down.len(), total);
    debug_assert_eq!(bulls.len(), total);
    debug_assert_eq!(bears.len(), total);
    debug_assert_eq!(upper.len(), total);
    debug_assert_eq!(lower.len(), total);

    directional_imbalance_index_batch_inner_into(
        high, low, &combos, kernel, parallel, &mut up, &mut down, &mut bulls, &mut bears,
        &mut upper, &mut lower,
    )?;

    Ok(DirectionalImbalanceIndexBatchOutput {
        up,
        down,
        bulls,
        bears,
        upper,
        lower,
        combos,
        rows,
        cols,
    })
}

fn directional_imbalance_index_batch_inner_into(
    high: &[f64],
    low: &[f64],
    combos: &[DirectionalImbalanceIndexParams],
    kernel: Kernel,
    parallel: bool,
    out_up: &mut [f64],
    out_down: &mut [f64],
    out_bulls: &mut [f64],
    out_bears: &mut [f64],
    out_upper: &mut [f64],
    out_lower: &mut [f64],
) -> Result<(), DirectionalImbalanceIndexError> {
    match kernel {
        Kernel::Auto
        | Kernel::Scalar
        | Kernel::ScalarBatch
        | Kernel::Avx2
        | Kernel::Avx2Batch
        | Kernel::Avx512
        | Kernel::Avx512Batch => {}
        other => return Err(DirectionalImbalanceIndexError::InvalidKernelForBatch(other)),
    }

    let len = high.len();
    let total = combos.len().checked_mul(len).ok_or_else(|| {
        DirectionalImbalanceIndexError::InvalidInput {
            msg: "directional_imbalance_index: rows*cols overflow in batch_into".to_string(),
        }
    })?;
    for dst in [
        &*out_up,
        &*out_down,
        &*out_bulls,
        &*out_bears,
        &*out_upper,
        &*out_lower,
    ] {
        if dst.len() != total {
            return Err(DirectionalImbalanceIndexError::MismatchedOutputLen {
                dst_len: dst.len(),
                expected_len: total,
            });
        }
    }

    if !any_valid_pair(high, low) {
        return Err(DirectionalImbalanceIndexError::AllValuesNaN);
    }

    let _chosen = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other => other,
    };

    let worker = |row: usize,
                  dst_up: &mut [f64],
                  dst_down: &mut [f64],
                  dst_bulls: &mut [f64],
                  dst_bears: &mut [f64],
                  dst_upper: &mut [f64],
                  dst_lower: &mut [f64]| {
        let resolved = resolve_params(&combos[row]).expect("validated combo");
        compute_into(
            high, low, resolved, dst_up, dst_down, dst_bulls, dst_bears, dst_upper, dst_lower,
        );
    };

    macro_rules! run_rows {
        ($iter:expr) => {
            for (row, (((((dst_up, dst_down), dst_bulls), dst_bears), dst_upper), dst_lower)) in
                $iter.enumerate()
            {
                worker(
                    row, dst_up, dst_down, dst_bulls, dst_bears, dst_upper, dst_lower,
                );
            }
        };
    }

    if parallel && combos.len() > 1 {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_up
                .par_chunks_mut(len)
                .zip(out_down.par_chunks_mut(len))
                .zip(out_bulls.par_chunks_mut(len))
                .zip(out_bears.par_chunks_mut(len))
                .zip(out_upper.par_chunks_mut(len))
                .zip(out_lower.par_chunks_mut(len))
                .enumerate()
                .for_each(
                    |(
                        row,
                        (((((dst_up, dst_down), dst_bulls), dst_bears), dst_upper), dst_lower),
                    )| {
                        worker(
                            row, dst_up, dst_down, dst_bulls, dst_bears, dst_upper, dst_lower,
                        );
                    },
                );
        }
        #[cfg(target_arch = "wasm32")]
        {
            run_rows!(out_up
                .chunks_mut(len)
                .zip(out_down.chunks_mut(len))
                .zip(out_bulls.chunks_mut(len))
                .zip(out_bears.chunks_mut(len))
                .zip(out_upper.chunks_mut(len))
                .zip(out_lower.chunks_mut(len)));
        }
    } else {
        run_rows!(out_up
            .chunks_mut(len)
            .zip(out_down.chunks_mut(len))
            .zip(out_bulls.chunks_mut(len))
            .zip(out_bears.chunks_mut(len))
            .zip(out_upper.chunks_mut(len))
            .zip(out_lower.chunks_mut(len)));
    }

    Ok(())
}

#[cfg(feature = "python")]
#[pyfunction(
    name = "directional_imbalance_index",
    signature = (high, low, length=DEFAULT_LENGTH, period=DEFAULT_PERIOD, kernel=None)
)]
pub fn directional_imbalance_index_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    length: usize,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let input = DirectionalImbalanceIndexInput::from_slices(
        high,
        low,
        DirectionalImbalanceIndexParams {
            length: Some(length),
            period: Some(period),
        },
    );
    let out = py
        .allow_threads(|| directional_imbalance_index_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        out.up.into_pyarray(py),
        out.down.into_pyarray(py),
        out.bulls.into_pyarray(py),
        out.bears.into_pyarray(py),
        out.upper.into_pyarray(py),
        out.lower.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyclass(name = "DirectionalImbalanceIndexStream")]
pub struct DirectionalImbalanceIndexStreamPy {
    inner: DirectionalImbalanceIndexStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl DirectionalImbalanceIndexStreamPy {
    #[new]
    #[pyo3(signature = (length=DEFAULT_LENGTH, period=DEFAULT_PERIOD))]
    fn new(length: usize, period: usize) -> PyResult<Self> {
        let inner = DirectionalImbalanceIndexStream::try_new(DirectionalImbalanceIndexParams {
            length: Some(length),
            period: Some(period),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn update(&mut self, high: f64, low: f64) -> Option<(f64, f64, f64, f64, f64, f64)> {
        self.inner.update(high, low).map(|point| {
            (
                point.up,
                point.down,
                point.bulls,
                point.bears,
                point.upper,
                point.lower,
            )
        })
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

#[cfg(feature = "python")]
#[pyfunction(
    name = "directional_imbalance_index_batch",
    signature = (high, low, length_range=(DEFAULT_LENGTH, DEFAULT_LENGTH, 0), period_range=(DEFAULT_PERIOD, DEFAULT_PERIOD, 0), kernel=None)
)]
pub fn directional_imbalance_index_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let out = py
        .allow_threads(|| {
            directional_imbalance_index_batch_with_kernel(
                high,
                low,
                &DirectionalImbalanceIndexBatchRange {
                    length: length_range,
                    period: period_range,
                },
                &DirectionalImbalanceIndexParams {
                    length: None,
                    period: None,
                },
                kern,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("up", out.up.into_pyarray(py).reshape((out.rows, out.cols))?)?;
    dict.set_item(
        "down",
        out.down.into_pyarray(py).reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "bulls",
        out.bulls.into_pyarray(py).reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "bears",
        out.bears.into_pyarray(py).reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "upper",
        out.upper.into_pyarray(py).reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "lower",
        out.lower.into_pyarray(py).reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "lengths",
        out.combos
            .iter()
            .map(|combo| combo.length.unwrap_or(DEFAULT_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "periods",
        out.combos
            .iter()
            .map(|combo| combo.period.unwrap_or(DEFAULT_PERIOD) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", out.rows)?;
    dict.set_item("cols", out.cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_directional_imbalance_index_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(directional_imbalance_index_py, m)?)?;
    m.add_function(wrap_pyfunction!(directional_imbalance_index_batch_py, m)?)?;
    m.add_class::<DirectionalImbalanceIndexStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectionalImbalanceIndexBatchConfig {
    pub length_range: Vec<usize>,
    pub period_range: Vec<usize>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = directional_imbalance_index_js)]
pub fn directional_imbalance_index_js(
    high: &[f64],
    low: &[f64],
    length: usize,
    period: usize,
) -> Result<JsValue, JsValue> {
    let out = directional_imbalance_index_with_kernel(
        &DirectionalImbalanceIndexInput::from_slices(
            high,
            low,
            DirectionalImbalanceIndexParams {
                length: Some(length),
                period: Some(period),
            },
        ),
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("up"),
        &serde_wasm_bindgen::to_value(&out.up).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("down"),
        &serde_wasm_bindgen::to_value(&out.down).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("bulls"),
        &serde_wasm_bindgen::to_value(&out.bulls).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("bears"),
        &serde_wasm_bindgen::to_value(&out.bears).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("upper"),
        &serde_wasm_bindgen::to_value(&out.upper).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("lower"),
        &serde_wasm_bindgen::to_value(&out.lower).unwrap(),
    )?;
    Ok(obj.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = directional_imbalance_index_batch_js)]
pub fn directional_imbalance_index_batch_js(
    high: &[f64],
    low: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: DirectionalImbalanceIndexBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.length_range.len() != 3 || config.period_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: every range must have exactly 3 elements [start, end, step]",
        ));
    }

    let out = directional_imbalance_index_batch_with_kernel(
        high,
        low,
        &DirectionalImbalanceIndexBatchRange {
            length: (
                config.length_range[0],
                config.length_range[1],
                config.length_range[2],
            ),
            period: (
                config.period_range[0],
                config.period_range[1],
                config.period_range[2],
            ),
        },
        &DirectionalImbalanceIndexParams {
            length: None,
            period: None,
        },
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("up"),
        &serde_wasm_bindgen::to_value(&out.up).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("down"),
        &serde_wasm_bindgen::to_value(&out.down).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("bulls"),
        &serde_wasm_bindgen::to_value(&out.bulls).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("bears"),
        &serde_wasm_bindgen::to_value(&out.bears).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("upper"),
        &serde_wasm_bindgen::to_value(&out.upper).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("lower"),
        &serde_wasm_bindgen::to_value(&out.lower).unwrap(),
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
pub fn directional_imbalance_index_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(6 * len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn directional_imbalance_index_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, 6 * len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn directional_imbalance_index_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length: usize,
    period: usize,
) -> Result<(), JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to directional_imbalance_index_into",
        ));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, 6 * len);
        let (dst_up, rest) = out.split_at_mut(len);
        let (dst_down, rest) = rest.split_at_mut(len);
        let (dst_bulls, rest) = rest.split_at_mut(len);
        let (dst_bears, rest) = rest.split_at_mut(len);
        let (dst_upper, dst_lower) = rest.split_at_mut(len);
        directional_imbalance_index_into_slice(
            dst_up,
            dst_down,
            dst_bulls,
            dst_bears,
            dst_upper,
            dst_lower,
            &DirectionalImbalanceIndexInput::from_slices(
                high,
                low,
                DirectionalImbalanceIndexParams {
                    length: Some(length),
                    period: Some(period),
                },
            ),
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn directional_imbalance_index_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length_start: usize,
    length_end: usize,
    length_step: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to directional_imbalance_index_batch_into",
        ));
    }

    let sweep = DirectionalImbalanceIndexBatchRange {
        length: (length_start, length_end, length_step),
        period: (period_start, period_end, period_step),
    };
    let fixed = DirectionalImbalanceIndexParams {
        length: None,
        period: None,
    };
    let combos =
        expand_grid_checked(&sweep, &fixed).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let total = rows
        .checked_mul(len)
        .and_then(|value| value.checked_mul(6))
        .ok_or_else(|| {
            JsValue::from_str("rows*cols overflow in directional_imbalance_index_batch_into")
        })?;

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        let split = rows * len;
        let (dst_up, rest) = out.split_at_mut(split);
        let (dst_down, rest) = rest.split_at_mut(split);
        let (dst_bulls, rest) = rest.split_at_mut(split);
        let (dst_bears, rest) = rest.split_at_mut(split);
        let (dst_upper, dst_lower) = rest.split_at_mut(split);
        directional_imbalance_index_batch_inner_into(
            high,
            low,
            &combos,
            Kernel::Auto,
            false,
            dst_up,
            dst_down,
            dst_bulls,
            dst_bears,
            dst_upper,
            dst_lower,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }
    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn directional_imbalance_index_output_into_js(
    high: &[f64],
    low: &[f64],
    length: usize,
    period: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = directional_imbalance_index_js(high, low, length, period)?;
    crate::write_wasm_object_f64_outputs("directional_imbalance_index_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn directional_imbalance_index_batch_output_into_js(
    high: &[f64],
    low: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = directional_imbalance_index_batch_js(high, low, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "directional_imbalance_index_batch_output_into_js",
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

    fn sample_high_low(len: usize) -> (Vec<f64>, Vec<f64>) {
        let high: Vec<f64> = (0..len)
            .map(|i| {
                let x = i as f64;
                100.0 + x * 0.03 + (x * 0.11).sin() * 2.0 + (x * 0.037).cos() * 0.4
            })
            .collect();
        let low: Vec<f64> = high
            .iter()
            .enumerate()
            .map(|(i, &h)| h - 1.2 - ((i as f64) * 0.07).cos().abs() * 0.3)
            .collect();
        (high, low)
    }

    fn naive(
        high: &[f64],
        low: &[f64],
        length: usize,
        period: usize,
    ) -> DirectionalImbalanceIndexOutput {
        let mut out = DirectionalImbalanceIndexOutput {
            up: vec![f64::NAN; high.len()],
            down: vec![f64::NAN; high.len()],
            bulls: vec![f64::NAN; high.len()],
            bears: vec![f64::NAN; high.len()],
            upper: vec![f64::NAN; high.len()],
            lower: vec![f64::NAN; high.len()],
        };
        let mut up_hits = VecDeque::new();
        let mut down_hits = VecDeque::new();
        let mut up_sum = 0.0;
        let mut down_sum = 0.0;
        for i in 0..high.len() {
            if !is_valid_pair(high[i], low[i]) {
                up_hits.clear();
                down_hits.clear();
                up_sum = 0.0;
                down_sum = 0.0;
                continue;
            }
            let start = i.saturating_sub(length);
            let mut upper = f64::NEG_INFINITY;
            let mut lower = f64::INFINITY;
            for j in start..=i {
                upper = upper.max(high[j]);
                lower = lower.min(low[j]);
            }
            let up_hit = if high[i] == upper { 1.0 } else { 0.0 };
            let down_hit = if low[i] == lower { 1.0 } else { 0.0 };
            up_hits.push_back(up_hit);
            down_hits.push_back(down_hit);
            up_sum += up_hit;
            down_sum += down_hit;
            if up_hits.len() > period {
                up_sum -= up_hits.pop_front().unwrap_or(0.0);
            }
            if down_hits.len() > period {
                down_sum -= down_hits.pop_front().unwrap_or(0.0);
            }
            out.up[i] = up_sum;
            out.down[i] = down_sum;
            let total = up_sum + down_sum;
            if total > 0.0 {
                out.bulls[i] = (up_sum / total) * 100.0;
                out.bears[i] = (down_sum / total) * 100.0;
            }
            out.upper[i] = upper;
            out.lower[i] = lower;
        }
        out
    }

    #[test]
    fn directional_imbalance_index_matches_naive() -> Result<(), Box<dyn Error>> {
        let (high, low) = sample_high_low(256);
        let input = DirectionalImbalanceIndexInput::from_slices(
            &high,
            &low,
            DirectionalImbalanceIndexParams {
                length: Some(10),
                period: Some(70),
            },
        );
        let out = directional_imbalance_index_with_kernel(&input, Kernel::Scalar)?;
        let expected = naive(&high, &low, 10, 70);
        assert_eq!(out.up, expected.up);
        assert_eq!(out.down, expected.down);
        assert_eq!(out.bulls, expected.bulls);
        assert_eq!(out.bears, expected.bears);
        assert_eq!(out.upper, expected.upper);
        assert_eq!(out.lower, expected.lower);
        Ok(())
    }

    #[test]
    fn directional_imbalance_index_stream_matches_batch() -> Result<(), Box<dyn Error>> {
        let (high, low) = sample_high_low(192);
        let params = DirectionalImbalanceIndexParams {
            length: Some(7),
            period: Some(20),
        };
        let batch = directional_imbalance_index(&DirectionalImbalanceIndexInput::from_slices(
            &high,
            &low,
            params.clone(),
        ))?;
        let mut stream = DirectionalImbalanceIndexStream::try_new(params)?;
        let mut up = Vec::with_capacity(high.len());
        let mut down = Vec::with_capacity(high.len());
        let mut bulls = Vec::with_capacity(high.len());
        let mut bears = Vec::with_capacity(high.len());
        let mut upper = Vec::with_capacity(high.len());
        let mut lower = Vec::with_capacity(high.len());
        for i in 0..high.len() {
            if let Some(point) = stream.update(high[i], low[i]) {
                up.push(point.up);
                down.push(point.down);
                bulls.push(point.bulls);
                bears.push(point.bears);
                upper.push(point.upper);
                lower.push(point.lower);
            } else {
                up.push(f64::NAN);
                down.push(f64::NAN);
                bulls.push(f64::NAN);
                bears.push(f64::NAN);
                upper.push(f64::NAN);
                lower.push(f64::NAN);
            }
        }
        assert_eq!(up, batch.up);
        assert_eq!(down, batch.down);
        assert_eq!(bulls, batch.bulls);
        assert_eq!(bears, batch.bears);
        assert_eq!(upper, batch.upper);
        assert_eq!(lower, batch.lower);
        Ok(())
    }

    #[test]
    fn directional_imbalance_index_into_slice_matches_direct() -> Result<(), Box<dyn Error>> {
        let (high, low) = sample_high_low(144);
        let input = DirectionalImbalanceIndexInput::from_slices(
            &high,
            &low,
            DirectionalImbalanceIndexParams {
                length: Some(5),
                period: Some(18),
            },
        );
        let baseline = directional_imbalance_index(&input)?;
        let mut up = alloc_with_nan_prefix(high.len(), 0);
        let mut down = alloc_with_nan_prefix(high.len(), 0);
        let mut bulls = alloc_with_nan_prefix(high.len(), 0);
        let mut bears = alloc_with_nan_prefix(high.len(), 0);
        let mut upper = alloc_with_nan_prefix(high.len(), 0);
        let mut lower = alloc_with_nan_prefix(high.len(), 0);
        directional_imbalance_index_into_slice(
            &mut up,
            &mut down,
            &mut bulls,
            &mut bears,
            &mut upper,
            &mut lower,
            &input,
            Kernel::Auto,
        )?;
        assert_eq!(up, baseline.up);
        assert_eq!(down, baseline.down);
        assert_eq!(bulls, baseline.bulls);
        assert_eq!(bears, baseline.bears);
        assert_eq!(upper, baseline.upper);
        assert_eq!(lower, baseline.lower);
        Ok(())
    }

    #[test]
    fn directional_imbalance_index_batch_and_dispatch_outputs() -> Result<(), Box<dyn Error>> {
        let (high, low) = sample_high_low(128);
        let batch = directional_imbalance_index_batch_with_kernel(
            &high,
            &low,
            &DirectionalImbalanceIndexBatchRange {
                length: (6, 10, 2),
                period: (14, 14, 0),
            },
            &DirectionalImbalanceIndexParams {
                length: None,
                period: None,
            },
            Kernel::Auto,
        )?;
        assert_eq!(batch.rows, 3);
        assert_eq!(batch.cols, high.len());
        assert_eq!(batch.combos[0].length, Some(6));
        assert_eq!(batch.combos[1].length, Some(8));
        assert_eq!(batch.combos[2].length, Some(10));
        assert_eq!(batch.combos[0].period, Some(14));

        let combos = [IndicatorParamSet {
            params: &[
                ParamKV {
                    key: "length",
                    value: ParamValue::Int(6),
                },
                ParamKV {
                    key: "period",
                    value: ParamValue::Int(14),
                },
            ],
        }];
        let out = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "directional_imbalance_index",
            output_id: Some("bulls"),
            data: IndicatorDataRef::HighLow {
                high: &high,
                low: &low,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        })?;
        let values = out.values_f64.expect("f64 output");
        assert_eq!(values.len(), high.len());
        assert_eq!(values, batch.bulls[..high.len()].to_vec());
        Ok(())
    }
}
