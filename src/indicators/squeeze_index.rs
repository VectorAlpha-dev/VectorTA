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
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::mem::ManuallyDrop;
use thiserror::Error;

const DEFAULT_CONV: f64 = 50.0;
const DEFAULT_LENGTH: usize = 20;
const FLOAT_TOL: f64 = 1e-12;

impl<'a> AsRef<[f64]> for SqueezeIndexInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            SqueezeIndexData::Slice(slice) => slice,
            SqueezeIndexData::Candles { candles, source } => squeeze_source_type(candles, source),
        }
    }
}

#[inline(always)]
fn squeeze_source_type<'a>(candles: &'a Candles, source: &str) -> &'a [f64] {
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
pub enum SqueezeIndexData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct SqueezeIndexOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct SqueezeIndexParams {
    pub conv: Option<f64>,
    pub length: Option<usize>,
}

impl Default for SqueezeIndexParams {
    fn default() -> Self {
        Self {
            conv: Some(DEFAULT_CONV),
            length: Some(DEFAULT_LENGTH),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SqueezeIndexInput<'a> {
    pub data: SqueezeIndexData<'a>,
    pub params: SqueezeIndexParams,
}

impl<'a> SqueezeIndexInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: SqueezeIndexParams) -> Self {
        Self {
            data: SqueezeIndexData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: SqueezeIndexParams) -> Self {
        Self {
            data: SqueezeIndexData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", SqueezeIndexParams::default())
    }

    #[inline]
    pub fn get_conv(&self) -> f64 {
        self.params.conv.unwrap_or(DEFAULT_CONV)
    }

    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(DEFAULT_LENGTH)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SqueezeIndexBuilder {
    conv: Option<f64>,
    length: Option<usize>,
    kernel: Kernel,
}

impl Default for SqueezeIndexBuilder {
    fn default() -> Self {
        Self {
            conv: None,
            length: None,
            kernel: Kernel::Auto,
        }
    }
}

impl SqueezeIndexBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn conv(mut self, conv: f64) -> Self {
        self.conv = Some(conv);
        self
    }

    #[inline]
    pub fn length(mut self, length: usize) -> Self {
        self.length = Some(length);
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
    ) -> Result<SqueezeIndexOutput, SqueezeIndexError> {
        let input = SqueezeIndexInput::from_candles(
            candles,
            source,
            SqueezeIndexParams {
                conv: self.conv,
                length: self.length,
            },
        );
        squeeze_index_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn apply_slice(self, data: &[f64]) -> Result<SqueezeIndexOutput, SqueezeIndexError> {
        let input = SqueezeIndexInput::from_slice(
            data,
            SqueezeIndexParams {
                conv: self.conv,
                length: self.length,
            },
        );
        squeeze_index_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn into_stream(self) -> Result<SqueezeIndexStream, SqueezeIndexError> {
        SqueezeIndexStream::try_new(SqueezeIndexParams {
            conv: self.conv,
            length: self.length,
        })
    }
}

#[derive(Debug, Error)]
pub enum SqueezeIndexError {
    #[error("squeeze_index: Input data slice is empty.")]
    EmptyInputData,
    #[error("squeeze_index: All values are NaN.")]
    AllValuesNaN,
    #[error("squeeze_index: Invalid conv: {conv}")]
    InvalidConv { conv: f64 },
    #[error("squeeze_index: Invalid length: length = {length}, data length = {data_len}")]
    InvalidLength { length: usize, data_len: usize },
    #[error("squeeze_index: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("squeeze_index: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("squeeze_index: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("squeeze_index: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Debug, Clone, Copy)]
struct ResolvedParams {
    conv: f64,
    inv_conv: f64,
    length: usize,
    length_f: f64,
    sum_y: f64,
    denom_y: f64,
    last_index_f: f64,
}

#[derive(Debug, Clone)]
pub struct SqueezeIndexStream {
    params: ResolvedParams,
    max_state: f64,
    min_state: f64,
    ring_vals: Vec<f64>,
    ring_valid: Vec<u8>,
    head: usize,
    filled: usize,
    valid_count: usize,
    sum_x: f64,
    sum_x2: f64,
    weighted: f64,
}

impl SqueezeIndexStream {
    pub fn try_new(params: SqueezeIndexParams) -> Result<Self, SqueezeIndexError> {
        let params = resolve_params(&params, 0)?;
        Ok(Self::new_resolved(params))
    }

    #[inline]
    fn new_resolved(params: ResolvedParams) -> Self {
        Self {
            params,
            max_state: 0.0,
            min_state: 0.0,
            ring_vals: vec![0.0; params.length],
            ring_valid: vec![0u8; params.length],
            head: 0,
            filled: 0,
            valid_count: 0,
            sum_x: 0.0,
            sum_x2: 0.0,
            weighted: 0.0,
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.max_state = 0.0;
        self.min_state = 0.0;
        self.ring_vals.fill(0.0);
        self.ring_valid.fill(0);
        self.head = 0;
        self.filled = 0;
        self.valid_count = 0;
        self.sum_x = 0.0;
        self.sum_x2 = 0.0;
        self.weighted = 0.0;
    }

    #[inline]
    pub fn get_warmup_period(&self) -> usize {
        self.params.length.saturating_sub(1)
    }

    #[inline]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        if !value.is_finite() {
            self.reset();
            return None;
        }

        let inv_conv = self.params.inv_conv;
        let max_next = value.max(self.max_state - (self.max_state - value) * inv_conv);
        let min_next = value.min(self.min_state + (value - self.min_state) * inv_conv);
        self.max_state = max_next;
        self.min_state = min_next;

        let spread = max_next - min_next;
        let diff = if spread > 0.0 { spread.ln() } else { f64::NAN };
        self.push_diff(diff)
    }

    #[inline]
    fn push_diff(&mut self, diff: f64) -> Option<f64> {
        let is_valid = diff.is_finite();
        let value = if is_valid { diff } else { 0.0 };
        let n = self.params.length;

        if self.filled < n {
            let pos = self.filled;
            self.ring_vals[pos] = value;
            self.ring_valid[pos] = if is_valid { 1 } else { 0 };
            self.sum_x += value;
            self.sum_x2 += value * value;
            self.weighted += pos as f64 * value;
            if is_valid {
                self.valid_count += 1;
            }
            self.filled += 1;
            if self.filled < n {
                return None;
            }
        } else {
            let old_value = self.ring_vals[self.head];
            let old_valid = self.ring_valid[self.head] as usize;
            let old_sum = self.sum_x;

            self.weighted = self.weighted - old_sum + old_value + self.params.last_index_f * value;
            self.sum_x = old_sum - old_value + value;
            self.sum_x2 = self.sum_x2 - old_value * old_value + value * value;
            self.valid_count = self.valid_count + if is_valid { 1 } else { 0 } - old_valid;

            self.ring_vals[self.head] = value;
            self.ring_valid[self.head] = if is_valid { 1 } else { 0 };
            self.head += 1;
            if self.head == n {
                self.head = 0;
            }
        }

        if self.valid_count != n {
            return Some(f64::NAN);
        }

        Some(psi_from_corr(
            self.sum_x,
            self.sum_x2,
            self.weighted,
            self.params,
        ))
    }
}

#[inline]
pub fn squeeze_index(input: &SqueezeIndexInput) -> Result<SqueezeIndexOutput, SqueezeIndexError> {
    squeeze_index_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn valid_value(value: f64) -> bool {
    value.is_finite()
}

#[inline(always)]
fn count_valid_values(data: &[f64]) -> usize {
    data.iter().filter(|&&v| valid_value(v)).count()
}

#[inline(always)]
fn first_valid_value(data: &[f64]) -> usize {
    let mut i = 0usize;
    while i < data.len() {
        if valid_value(data[i]) {
            return i;
        }
        i += 1;
    }
    data.len()
}

#[inline(always)]
fn psi_from_corr(sum_x: f64, sum_x2: f64, weighted: f64, params: ResolvedParams) -> f64 {
    let denom_x = params.length_f * sum_x2 - sum_x * sum_x;
    let denom_y = params.denom_y;
    let denom = denom_x * denom_y;
    if denom <= 0.0 || !denom.is_finite() {
        return f64::NAN;
    }
    let corr = (params.length_f * weighted - sum_x * params.sum_y) / denom.sqrt();
    -50.0 * corr + 50.0
}

#[inline]
fn resolve_params(
    params: &SqueezeIndexParams,
    data_len: usize,
) -> Result<ResolvedParams, SqueezeIndexError> {
    let conv = params.conv.unwrap_or(DEFAULT_CONV);
    if !conv.is_finite() || conv <= 1.0 {
        return Err(SqueezeIndexError::InvalidConv { conv });
    }
    let length = params.length.unwrap_or(DEFAULT_LENGTH);
    if length == 0 || (data_len > 0 && length > data_len) {
        return Err(SqueezeIndexError::InvalidLength { length, data_len });
    }
    let length_f = length as f64;
    let sum_y = length_f * (length_f - 1.0) * 0.5;
    let sum_y2 = (length_f - 1.0) * length_f * (2.0 * length_f - 1.0) / 6.0;
    let denom_y = length_f * sum_y2 - sum_y * sum_y;
    Ok(ResolvedParams {
        conv,
        inv_conv: 1.0 / conv,
        length,
        length_f,
        sum_y,
        denom_y,
        last_index_f: length.saturating_sub(1) as f64,
    })
}

#[inline]
fn squeeze_index_prepare<'a>(
    input: &'a SqueezeIndexInput,
    kernel: Kernel,
) -> Result<(&'a [f64], ResolvedParams, Kernel, bool), SqueezeIndexError> {
    let data = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(SqueezeIndexError::EmptyInputData);
    }

    let first = first_valid_value(data);
    if first >= len {
        return Err(SqueezeIndexError::AllValuesNaN);
    }

    let params = resolve_params(&input.params, len)?;
    let valid = count_valid_values(data);
    if valid < params.length {
        return Err(SqueezeIndexError::NotEnoughValidData {
            needed: params.length,
            valid,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other.to_non_batch(),
    };

    Ok((data, params, chosen, valid == len))
}

#[inline(always)]
fn squeeze_index_row_from_slice(
    data: &[f64],
    params: ResolvedParams,
    all_finite: bool,
    out: &mut [f64],
) {
    if all_finite && squeeze_index_row_all_finite(data, params, out) {
        return;
    }

    let warm = params.length.saturating_sub(1).min(out.len());
    out[..warm].fill(f64::NAN);
    let mut stream = SqueezeIndexStream::new_resolved(params);
    for (i, &value) in data.iter().enumerate() {
        if let Some(out_value) = stream.update(value) {
            out[i] = out_value;
        } else if i >= warm {
            out[i] = f64::NAN;
        }
    }
}

#[inline(always)]
fn squeeze_index_row_all_finite(data: &[f64], params: ResolvedParams, out: &mut [f64]) -> bool {
    if params.length <= 64 {
        let mut ring_vals = [0.0_f64; 64];
        return squeeze_index_row_all_finite_with_ring(
            data,
            params,
            out,
            &mut ring_vals[..params.length],
        );
    }
    let mut ring_vals = vec![0.0; params.length];
    squeeze_index_row_all_finite_with_ring(data, params, out, &mut ring_vals)
}

#[inline(always)]
fn squeeze_index_row_all_finite_with_ring(
    data: &[f64],
    params: ResolvedParams,
    out: &mut [f64],
    ring_vals: &mut [f64],
) -> bool {
    let warm = params.length.saturating_sub(1).min(out.len());
    out[..warm].fill(f64::NAN);

    let n = params.length;
    let mut head = 0usize;
    let mut filled = 0usize;
    let mut sum_x = 0.0;
    let mut sum_x2 = 0.0;
    let mut weighted = 0.0;
    let mut max_state = 0.0;
    let mut min_state = 0.0;

    for i in 0..data.len() {
        let value = data[i];
        let max_next = value.max(max_state - (max_state - value) * params.inv_conv);
        let min_next = value.min(min_state + (value - min_state) * params.inv_conv);
        max_state = max_next;
        min_state = min_next;

        let spread = max_next - min_next;
        if spread <= 0.0 || !spread.is_finite() {
            return false;
        }
        let diff = spread.ln();
        if !diff.is_finite() {
            return false;
        }

        if filled < n {
            ring_vals[filled] = diff;
            sum_x += diff;
            sum_x2 += diff * diff;
            weighted += filled as f64 * diff;
            filled += 1;
            if filled == n {
                out[i] = psi_from_corr(sum_x, sum_x2, weighted, params);
            }
        } else {
            let old_value = ring_vals[head];
            let old_sum = sum_x;
            weighted = weighted - old_sum + old_value + params.last_index_f * diff;
            sum_x = old_sum - old_value + diff;
            sum_x2 = sum_x2 - old_value * old_value + diff * diff;
            ring_vals[head] = diff;
            head += 1;
            if head == n {
                head = 0;
            }
            out[i] = psi_from_corr(sum_x, sum_x2, weighted, params);
        }
    }

    true
}

#[inline]
pub fn squeeze_index_with_kernel(
    input: &SqueezeIndexInput,
    kernel: Kernel,
) -> Result<SqueezeIndexOutput, SqueezeIndexError> {
    let (data, params, _chosen, all_finite) = squeeze_index_prepare(input, kernel)?;
    let mut values = alloc_with_nan_prefix(data.len(), params.length.saturating_sub(1));
    squeeze_index_row_from_slice(data, params, all_finite, &mut values);
    Ok(SqueezeIndexOutput { values })
}

#[inline]
pub fn squeeze_index_into_slice(
    dst: &mut [f64],
    input: &SqueezeIndexInput,
    kernel: Kernel,
) -> Result<(), SqueezeIndexError> {
    let (data, params, _chosen, all_finite) = squeeze_index_prepare(input, kernel)?;
    if dst.len() != data.len() {
        return Err(SqueezeIndexError::OutputLengthMismatch {
            expected: data.len(),
            got: dst.len(),
        });
    }
    squeeze_index_row_from_slice(data, params, all_finite, dst);
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn squeeze_index_into(
    input: &SqueezeIndexInput,
    out: &mut [f64],
) -> Result<(), SqueezeIndexError> {
    squeeze_index_into_slice(out, input, Kernel::Auto)
}

#[derive(Clone, Debug)]
pub struct SqueezeIndexBatchRange {
    pub conv: (f64, f64, f64),
    pub length: (usize, usize, usize),
}

impl Default for SqueezeIndexBatchRange {
    fn default() -> Self {
        Self {
            conv: (DEFAULT_CONV, DEFAULT_CONV, 0.0),
            length: (DEFAULT_LENGTH, DEFAULT_LENGTH, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SqueezeIndexBatchBuilder {
    range: SqueezeIndexBatchRange,
    kernel: Kernel,
}

impl SqueezeIndexBatchBuilder {
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
    pub fn conv_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.conv = (start, end, step);
        self
    }

    #[inline]
    pub fn conv_static(mut self, conv: f64) -> Self {
        self.range.conv = (conv, conv, 0.0);
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
    pub fn apply_slice(self, data: &[f64]) -> Result<SqueezeIndexBatchOutput, SqueezeIndexError> {
        squeeze_index_batch_with_kernel(data, &self.range, self.kernel)
    }

    #[inline]
    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<SqueezeIndexBatchOutput, SqueezeIndexError> {
        self.apply_slice(squeeze_source_type(candles, source))
    }

    #[inline]
    pub fn with_default_candles(
        candles: &Candles,
    ) -> Result<SqueezeIndexBatchOutput, SqueezeIndexError> {
        SqueezeIndexBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(candles, "close")
    }
}

#[derive(Clone, Debug)]
pub struct SqueezeIndexBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<SqueezeIndexParams>,
    pub rows: usize,
    pub cols: usize,
}

impl SqueezeIndexBatchOutput {
    pub fn row_for_params(&self, params: &SqueezeIndexParams) -> Option<usize> {
        self.combos.iter().position(|combo| {
            (combo.conv.unwrap_or(DEFAULT_CONV) - params.conv.unwrap_or(DEFAULT_CONV)).abs()
                <= FLOAT_TOL
                && combo.length.unwrap_or(DEFAULT_LENGTH) == params.length.unwrap_or(DEFAULT_LENGTH)
        })
    }

    pub fn values_for(&self, params: &SqueezeIndexParams) -> Option<&[f64]> {
        self.row_for_params(params).and_then(|row| {
            row.checked_mul(self.cols)
                .and_then(|start| self.values.get(start..start + self.cols))
        })
    }
}

fn expand_axis_usize(
    start: usize,
    end: usize,
    step: usize,
) -> Result<Vec<usize>, SqueezeIndexError> {
    if start == end {
        return Ok(vec![start]);
    }
    if step == 0 {
        return Err(SqueezeIndexError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
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
        while x >= end {
            out.push(x);
            let next = x.saturating_sub(step);
            if next == x {
                break;
            }
            x = next;
        }
    }

    if out.is_empty() {
        return Err(SqueezeIndexError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

fn expand_axis_f64(start: f64, end: f64, step: f64) -> Result<Vec<f64>, SqueezeIndexError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(SqueezeIndexError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if (start - end).abs() <= FLOAT_TOL {
        return Ok(vec![start]);
    }
    if step == 0.0 {
        return Err(SqueezeIndexError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }

    let mut out = Vec::new();
    if start < end {
        let mut x = start;
        while x <= end + FLOAT_TOL {
            out.push(x);
            x += step.abs();
        }
    } else {
        let mut x = start;
        while x >= end - FLOAT_TOL {
            out.push(x);
            x -= step.abs();
        }
    }

    if out.is_empty() {
        return Err(SqueezeIndexError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

fn expand_grid_squeeze_index(
    range: &SqueezeIndexBatchRange,
) -> Result<Vec<SqueezeIndexParams>, SqueezeIndexError> {
    let convs = expand_axis_f64(range.conv.0, range.conv.1, range.conv.2)?;
    let lengths = expand_axis_usize(range.length.0, range.length.1, range.length.2)?;

    let mut combos = Vec::with_capacity(convs.len().saturating_mul(lengths.len()));
    for &conv in &convs {
        if !conv.is_finite() || conv <= 1.0 {
            return Err(SqueezeIndexError::InvalidConv { conv });
        }
        for &length in &lengths {
            if length == 0 {
                return Err(SqueezeIndexError::InvalidLength {
                    length,
                    data_len: 0,
                });
            }
            combos.push(SqueezeIndexParams {
                conv: Some(conv),
                length: Some(length),
            });
        }
    }
    Ok(combos)
}

#[inline]
pub fn squeeze_index_batch_with_kernel(
    data: &[f64],
    sweep: &SqueezeIndexBatchRange,
    kernel: Kernel,
) -> Result<SqueezeIndexBatchOutput, SqueezeIndexError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(SqueezeIndexError::InvalidKernelForBatch(other)),
    };
    squeeze_index_batch_par_slice(data, sweep, batch_kernel.to_non_batch())
}

#[inline]
pub fn squeeze_index_batch_slice(
    data: &[f64],
    sweep: &SqueezeIndexBatchRange,
    kernel: Kernel,
) -> Result<SqueezeIndexBatchOutput, SqueezeIndexError> {
    squeeze_index_batch_inner(data, sweep, kernel, false)
}

#[inline]
pub fn squeeze_index_batch_par_slice(
    data: &[f64],
    sweep: &SqueezeIndexBatchRange,
    kernel: Kernel,
) -> Result<SqueezeIndexBatchOutput, SqueezeIndexError> {
    squeeze_index_batch_inner(data, sweep, kernel, true)
}

#[inline(always)]
fn squeeze_index_batch_inner(
    data: &[f64],
    sweep: &SqueezeIndexBatchRange,
    _kernel: Kernel,
    parallel: bool,
) -> Result<SqueezeIndexBatchOutput, SqueezeIndexError> {
    let combos = expand_grid_squeeze_index(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    if cols == 0 {
        return Err(SqueezeIndexError::EmptyInputData);
    }

    let first = first_valid_value(data);
    if first >= cols {
        return Err(SqueezeIndexError::AllValuesNaN);
    }
    let valid = count_valid_values(data);
    let all_finite = valid == cols;
    let max_length = combos
        .iter()
        .map(|combo| combo.length.unwrap_or(DEFAULT_LENGTH))
        .max()
        .unwrap_or(0);
    if valid < max_length {
        return Err(SqueezeIndexError::NotEnoughValidData {
            needed: max_length,
            valid,
        });
    }

    let mut buf_mu = make_uninit_matrix(rows, cols);
    let warmups: Vec<usize> = combos
        .iter()
        .map(|combo| combo.length.unwrap_or(DEFAULT_LENGTH).saturating_sub(1))
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
                let params = resolve_params(&combos[row], cols).unwrap();
                squeeze_index_row_from_slice(data, params, all_finite, out_row);
            });

        #[cfg(target_arch = "wasm32")]
        for (row, out_row) in out.chunks_mut(cols).enumerate() {
            let params = resolve_params(&combos[row], cols).unwrap();
            squeeze_index_row_from_slice(data, params, all_finite, out_row);
        }
    } else {
        for (row, out_row) in out.chunks_mut(cols).enumerate() {
            let params = resolve_params(&combos[row], cols).unwrap();
            squeeze_index_row_from_slice(data, params, all_finite, out_row);
        }
    }

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(SqueezeIndexBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub fn squeeze_index_batch_inner_into(
    data: &[f64],
    sweep: &SqueezeIndexBatchRange,
    _kernel: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<SqueezeIndexParams>, SqueezeIndexError> {
    let combos = expand_grid_squeeze_index(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    if cols == 0 {
        return Err(SqueezeIndexError::EmptyInputData);
    }

    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| SqueezeIndexError::OutputLengthMismatch {
            expected: usize::MAX,
            got: out.len(),
        })?;
    if out.len() != total {
        return Err(SqueezeIndexError::OutputLengthMismatch {
            expected: total,
            got: out.len(),
        });
    }

    let first = first_valid_value(data);
    if first >= cols {
        return Err(SqueezeIndexError::AllValuesNaN);
    }
    let valid = count_valid_values(data);
    let all_finite = valid == cols;
    let max_length = combos
        .iter()
        .map(|combo| combo.length.unwrap_or(DEFAULT_LENGTH))
        .max()
        .unwrap_or(0);
    if valid < max_length {
        return Err(SqueezeIndexError::NotEnoughValidData {
            needed: max_length,
            valid,
        });
    }

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out.par_chunks_mut(cols)
            .enumerate()
            .for_each(|(row, out_row)| {
                let params = resolve_params(&combos[row], cols).unwrap();
                squeeze_index_row_from_slice(data, params, all_finite, out_row);
            });

        #[cfg(target_arch = "wasm32")]
        for (row, out_row) in out.chunks_mut(cols).enumerate() {
            let params = resolve_params(&combos[row], cols).unwrap();
            squeeze_index_row_from_slice(data, params, all_finite, out_row);
        }
    } else {
        for (row, out_row) in out.chunks_mut(cols).enumerate() {
            let params = resolve_params(&combos[row], cols).unwrap();
            squeeze_index_row_from_slice(data, params, all_finite, out_row);
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "squeeze_index")]
#[pyo3(signature = (data, conv=DEFAULT_CONV, length=DEFAULT_LENGTH, kernel=None))]
pub fn squeeze_index_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    conv: f64,
    length: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = data.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = SqueezeIndexInput::from_slice(
        data,
        SqueezeIndexParams {
            conv: Some(conv),
            length: Some(length),
        },
    );
    let output = py
        .allow_threads(|| squeeze_index_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(output.values.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "SqueezeIndexStream")]
pub struct SqueezeIndexStreamPy {
    stream: SqueezeIndexStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl SqueezeIndexStreamPy {
    #[new]
    #[pyo3(signature = (conv=DEFAULT_CONV, length=DEFAULT_LENGTH))]
    fn new(conv: f64, length: usize) -> PyResult<Self> {
        let stream = SqueezeIndexStream::try_new(SqueezeIndexParams {
            conv: Some(conv),
            length: Some(length),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "squeeze_index_batch")]
#[pyo3(signature = (data, conv_range, length_range, kernel=None))]
pub fn squeeze_index_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    conv_range: (f64, f64, f64),
    length_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let kernel = validate_kernel(kernel, true)?;
    let sweep = SqueezeIndexBatchRange {
        conv: conv_range,
        length: length_range,
    };

    let combos =
        expand_grid_squeeze_index(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let out_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    let combos = py
        .allow_threads(|| {
            let batch = match kernel {
                Kernel::Auto => detect_best_batch_kernel(),
                other => other,
            };
            squeeze_index_batch_inner_into(data, &sweep, batch.to_non_batch(), true, slice_out)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "convs",
        combos
            .iter()
            .map(|combo| combo.conv.unwrap_or(DEFAULT_CONV))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "lengths",
        combos
            .iter()
            .map(|combo| combo.length.unwrap_or(DEFAULT_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_squeeze_index_module(module: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(squeeze_index_py, module)?)?;
    module.add_function(wrap_pyfunction!(squeeze_index_batch_py, module)?)?;
    module.add_class::<SqueezeIndexStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "squeeze_index_js")]
pub fn squeeze_index_js(data: &[f64], conv: f64, length: usize) -> Result<Vec<f64>, JsValue> {
    let input = SqueezeIndexInput::from_slice(
        data,
        SqueezeIndexParams {
            conv: Some(conv),
            length: Some(length),
        },
    );
    let mut output = vec![0.0; data.len()];
    squeeze_index_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(output)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn squeeze_index_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn squeeze_index_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn squeeze_index_into(
    data_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    conv: f64,
    length: usize,
) -> Result<(), JsValue> {
    if data_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let input = SqueezeIndexInput::from_slice(
            data,
            SqueezeIndexParams {
                conv: Some(conv),
                length: Some(length),
            },
        );
        if data_ptr == out_ptr {
            let mut tmp = vec![0.0; len];
            squeeze_index_into_slice(&mut tmp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            std::slice::from_raw_parts_mut(out_ptr, len).copy_from_slice(&tmp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            squeeze_index_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct SqueezeIndexBatchConfig {
    pub conv_range: (f64, f64, f64),
    pub length_range: (usize, usize, usize),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct SqueezeIndexBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<SqueezeIndexParams>,
    pub convs: Vec<f64>,
    pub lengths: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "squeeze_index_batch_js")]
pub fn squeeze_index_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: SqueezeIndexBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let sweep = SqueezeIndexBatchRange {
        conv: config.conv_range,
        length: config.length_range,
    };
    let output = squeeze_index_batch_inner(data, &sweep, detect_best_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    serde_wasm_bindgen::to_value(&SqueezeIndexBatchJsOutput {
        convs: output
            .combos
            .iter()
            .map(|combo| combo.conv.unwrap_or(DEFAULT_CONV))
            .collect(),
        lengths: output
            .combos
            .iter()
            .map(|combo| combo.length.unwrap_or(DEFAULT_LENGTH))
            .collect(),
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn squeeze_index_batch_into(
    data_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    conv_start: f64,
    conv_end: f64,
    conv_step: f64,
    length_start: usize,
    length_end: usize,
    length_step: usize,
) -> Result<usize, JsValue> {
    if data_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    let sweep = SqueezeIndexBatchRange {
        conv: (conv_start, conv_end, conv_step),
        length: (length_start, length_end, length_step),
    };
    let combos =
        expand_grid_squeeze_index(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();

    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let total = rows
            .checked_mul(len)
            .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        squeeze_index_batch_inner_into(data, &sweep, detect_best_kernel(), false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }

    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn squeeze_index_output_into_js(
    data: &[f64],
    conv: f64,
    length: usize,
    out: &js_sys::Float64Array,
) -> Result<usize, JsValue> {
    let values = squeeze_index_js(data, conv, length)?;
    crate::write_wasm_f64_output("squeeze_index_output_into_js", &values, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn squeeze_index_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = squeeze_index_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs("squeeze_index_batch_output_into_js", &value, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use std::error::Error;

    fn load_close() -> Result<Vec<f64>, Box<dyn Error>> {
        let candles = read_candles_from_csv("src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv")?;
        Ok(candles.close)
    }

    fn approx_eq_or_nan(lhs: &[f64], rhs: &[f64], tol: f64) {
        assert_eq!(lhs.len(), rhs.len());
        for (i, (&a, &b)) in lhs.iter().zip(rhs.iter()).enumerate() {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert!(
                (a - b).abs() <= tol,
                "mismatch at {i}: lhs={a} rhs={b} tol={tol}"
            );
        }
    }

    #[test]
    fn squeeze_index_output_contract() -> Result<(), Box<dyn Error>> {
        let close = load_close()?;
        let input = SqueezeIndexInput::from_slice(
            &close,
            SqueezeIndexParams {
                conv: Some(50.0),
                length: Some(20),
            },
        );
        let out = squeeze_index_with_kernel(&input, Kernel::Scalar)?;
        assert_eq!(out.values.len(), close.len());
        let first_valid = out.values.iter().position(|v| !v.is_nan()).unwrap();
        assert!(first_valid >= 19);
        assert!(out.values[first_valid..].iter().any(|v| v.is_finite()));
        Ok(())
    }

    #[test]
    fn squeeze_index_auto_matches_scalar() -> Result<(), Box<dyn Error>> {
        let close = load_close()?;
        let input = SqueezeIndexInput::from_slice(
            &close,
            SqueezeIndexParams {
                conv: Some(50.0),
                length: Some(20),
            },
        );
        let auto = squeeze_index_with_kernel(&input, Kernel::Auto)?;
        let scalar = squeeze_index_with_kernel(&input, Kernel::Scalar)?;
        approx_eq_or_nan(&auto.values, &scalar.values, 1e-12);
        Ok(())
    }

    #[test]
    fn squeeze_index_stream_matches_batch_and_recovers_after_nan() -> Result<(), Box<dyn Error>> {
        let mut close = load_close()?;
        close[64] = f64::NAN;
        let batch = squeeze_index_with_kernel(
            &SqueezeIndexInput::from_slice(
                &close,
                SqueezeIndexParams {
                    conv: Some(50.0),
                    length: Some(10),
                },
            ),
            Kernel::Scalar,
        )?;
        let mut stream = SqueezeIndexStream::try_new(SqueezeIndexParams {
            conv: Some(50.0),
            length: Some(10),
        })?;
        let mut streamed = vec![f64::NAN; close.len()];
        for (i, &value) in close.iter().enumerate() {
            if let Some(out) = stream.update(value) {
                streamed[i] = out;
            }
        }
        approx_eq_or_nan(&streamed, &batch.values, 1e-12);
        assert!(streamed[64].is_nan());
        assert!(streamed[72].is_nan());
        assert!(streamed[74].is_finite());
        Ok(())
    }

    #[test]
    fn squeeze_index_batch_single_param_matches_single() -> Result<(), Box<dyn Error>> {
        let close = load_close()?;
        let batch = squeeze_index_batch_with_kernel(
            &close,
            &SqueezeIndexBatchRange {
                conv: (50.0, 50.0, 0.0),
                length: (20, 20, 0),
            },
            Kernel::Auto,
        )?;
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, close.len());
        let single = squeeze_index_with_kernel(
            &SqueezeIndexInput::from_slice(
                &close,
                SqueezeIndexParams {
                    conv: Some(50.0),
                    length: Some(20),
                },
            ),
            Kernel::Auto,
        )?;
        approx_eq_or_nan(&batch.values, &single.values, 1e-12);
        Ok(())
    }
}
