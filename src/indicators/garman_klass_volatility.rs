#[cfg(all(feature = "python", feature = "cuda"))]
pub use crate::utilities::dlpack_cuda::{make_device_array_py, DeviceArrayF32Py};

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
    alloc_uninit_f64, alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel,
    init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::mem::ManuallyDrop;
use thiserror::Error;

const GK_COEFF: f64 = 2.0 * std::f64::consts::LN_2 - 1.0;

#[derive(Debug, Clone)]
pub enum GarmanKlassVolatilityData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct GarmanKlassVolatilityOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct GarmanKlassVolatilityParams {
    pub lookback: Option<usize>,
}

impl Default for GarmanKlassVolatilityParams {
    fn default() -> Self {
        Self { lookback: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct GarmanKlassVolatilityInput<'a> {
    pub data: GarmanKlassVolatilityData<'a>,
    pub params: GarmanKlassVolatilityParams,
}

impl<'a> GarmanKlassVolatilityInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: GarmanKlassVolatilityParams) -> Self {
        Self {
            data: GarmanKlassVolatilityData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: GarmanKlassVolatilityParams,
    ) -> Self {
        Self {
            data: GarmanKlassVolatilityData::Slices {
                open,
                high,
                low,
                close,
            },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, GarmanKlassVolatilityParams::default())
    }

    #[inline]
    pub fn get_lookback(&self) -> usize {
        self.params.lookback.unwrap_or(14)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct GarmanKlassVolatilityBuilder {
    lookback: Option<usize>,
    kernel: Kernel,
}

impl Default for GarmanKlassVolatilityBuilder {
    fn default() -> Self {
        Self {
            lookback: None,
            kernel: Kernel::Auto,
        }
    }
}

impl GarmanKlassVolatilityBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn lookback(mut self, lookback: usize) -> Self {
        self.lookback = Some(lookback);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<GarmanKlassVolatilityOutput, GarmanKlassVolatilityError> {
        let input = GarmanKlassVolatilityInput::from_candles(
            candles,
            GarmanKlassVolatilityParams {
                lookback: self.lookback,
            },
        );
        garman_klass_volatility_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<GarmanKlassVolatilityOutput, GarmanKlassVolatilityError> {
        let input = GarmanKlassVolatilityInput::from_slices(
            open,
            high,
            low,
            close,
            GarmanKlassVolatilityParams {
                lookback: self.lookback,
            },
        );
        garman_klass_volatility_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<GarmanKlassVolatilityStream, GarmanKlassVolatilityError> {
        GarmanKlassVolatilityStream::try_new(GarmanKlassVolatilityParams {
            lookback: self.lookback,
        })
    }
}

#[derive(Debug, Error)]
pub enum GarmanKlassVolatilityError {
    #[error("garman_klass_volatility: Input data slice is empty.")]
    EmptyInputData,
    #[error("garman_klass_volatility: All values are NaN or non-positive.")]
    AllValuesNaN,
    #[error(
        "garman_klass_volatility: Invalid lookback: lookback = {lookback}, data length = {data_len}"
    )]
    InvalidLookback { lookback: usize, data_len: usize },
    #[error("garman_klass_volatility: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("garman_klass_volatility: Inconsistent slice lengths: open={open_len}, high={high_len}, low={low_len}, close={close_len}")]
    InconsistentSliceLengths {
        open_len: usize,
        high_len: usize,
        low_len: usize,
        close_len: usize,
    },
    #[error("garman_klass_volatility: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("garman_klass_volatility: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("garman_klass_volatility: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Debug, Clone)]
pub struct GarmanKlassVolatilityStream {
    lookback: usize,
    terms: Vec<f64>,
    valid: Vec<u8>,
    idx: usize,
    cnt: usize,
    valid_count: usize,
    sum_terms: f64,
}

impl GarmanKlassVolatilityStream {
    pub fn try_new(
        params: GarmanKlassVolatilityParams,
    ) -> Result<GarmanKlassVolatilityStream, GarmanKlassVolatilityError> {
        let lookback = params.lookback.unwrap_or(14);
        if lookback == 0 {
            return Err(GarmanKlassVolatilityError::InvalidLookback {
                lookback,
                data_len: 0,
            });
        }
        Ok(Self {
            lookback,
            terms: vec![0.0; lookback],
            valid: vec![0u8; lookback],
            idx: 0,
            cnt: 0,
            valid_count: 0,
            sum_terms: 0.0,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, open: f64, high: f64, low: f64, close: f64) -> Option<f64> {
        if self.cnt >= self.lookback {
            let old_idx = self.idx;
            if self.valid[old_idx] != 0 {
                self.valid_count = self.valid_count.saturating_sub(1);
                self.sum_terms -= self.terms[old_idx];
            }
        } else {
            self.cnt += 1;
        }

        if valid_ohlc_bar(open, high, low, close) {
            let term = gk_term(open, high, low, close);
            self.terms[self.idx] = term;
            self.valid[self.idx] = 1;
            self.valid_count += 1;
            self.sum_terms += term;
        } else {
            self.terms[self.idx] = 0.0;
            self.valid[self.idx] = 0;
        }

        self.idx += 1;
        if self.idx == self.lookback {
            self.idx = 0;
        }

        if self.cnt < self.lookback || self.valid_count != self.lookback {
            return None;
        }

        let mut variance = self.sum_terms / self.lookback as f64;
        if variance < 0.0 {
            variance = 0.0;
        }
        Some(variance.sqrt())
    }

    #[inline(always)]
    pub fn get_warmup_period(&self) -> usize {
        self.lookback.saturating_sub(1)
    }
}

#[inline]
pub fn garman_klass_volatility(
    input: &GarmanKlassVolatilityInput,
) -> Result<GarmanKlassVolatilityOutput, GarmanKlassVolatilityError> {
    garman_klass_volatility_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn valid_ohlc_bar(open: f64, high: f64, low: f64, close: f64) -> bool {
    open.is_finite()
        && high.is_finite()
        && low.is_finite()
        && close.is_finite()
        && open > 0.0
        && high > 0.0
        && low > 0.0
        && close > 0.0
}

#[inline(always)]
fn gk_term(open: f64, high: f64, low: f64, close: f64) -> f64 {
    let hl = (high / low).ln();
    let co = (close / open).ln();
    0.5 * hl * hl - GK_COEFF * co * co
}

#[inline(always)]
fn first_valid_ohlc(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> usize {
    let len = close.len();
    let mut i = 0usize;
    while i < len {
        if valid_ohlc_bar(open[i], high[i], low[i], close[i]) {
            break;
        }
        i += 1;
    }
    i.min(len)
}

#[inline(always)]
fn count_valid_ohlc(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> usize {
    let mut count = 0usize;
    for i in 0..close.len() {
        if valid_ohlc_bar(open[i], high[i], low[i], close[i]) {
            count += 1;
        }
    }
    count
}

#[inline(always)]
fn validity_summary(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> (usize, usize, bool) {
    let len = close.len();
    let mut first = len;
    let mut valid = 0usize;
    let mut all_valid = true;
    for i in 0..len {
        if valid_ohlc_bar(open[i], high[i], low[i], close[i]) {
            if first == len {
                first = i;
            }
            valid += 1;
        } else {
            all_valid = false;
        }
    }
    (first, valid, all_valid)
}

#[inline(always)]
fn build_prefix_terms(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> (Vec<u32>, Vec<f64>) {
    let len = close.len();
    let mut prefix_valid = vec![0u32; len + 1];
    let mut prefix_sum = vec![0.0f64; len + 1];

    for i in 0..len {
        if valid_ohlc_bar(open[i], high[i], low[i], close[i]) {
            prefix_valid[i + 1] = prefix_valid[i] + 1;
            prefix_sum[i + 1] = prefix_sum[i] + gk_term(open[i], high[i], low[i], close[i]);
        } else {
            prefix_valid[i + 1] = prefix_valid[i];
            prefix_sum[i + 1] = prefix_sum[i];
        }
    }

    (prefix_valid, prefix_sum)
}

#[inline(always)]
fn build_prefix_sum_terms_all_valid(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> Vec<f64> {
    let len = close.len();
    let mut prefix_sum = vec![0.0f64; len + 1];
    for i in 0..len {
        prefix_sum[i + 1] = prefix_sum[i] + gk_term(open[i], high[i], low[i], close[i]);
    }
    prefix_sum
}

#[inline(always)]
fn gk_row_from_prefix(
    prefix_valid: &[u32],
    prefix_sum: &[f64],
    lookback: usize,
    first: usize,
    out: &mut [f64],
) {
    let warmup = first.saturating_add(lookback.saturating_sub(1));
    let lookback_u32 = lookback as u32;
    let inv_lb = 1.0 / lookback as f64;

    for (t, slot) in out.iter_mut().enumerate() {
        if t < warmup {
            *slot = f64::NAN;
            continue;
        }

        let window_start = t + 1 - lookback;
        let valid_count = prefix_valid[t + 1] - prefix_valid[window_start];
        if valid_count != lookback_u32 {
            *slot = f64::NAN;
            continue;
        }

        let mut variance = (prefix_sum[t + 1] - prefix_sum[window_start]) * inv_lb;
        if variance < 0.0 {
            variance = 0.0;
        }
        *slot = variance.sqrt();
    }
}

#[inline(always)]
fn gk_row_from_prefix_all_valid(
    prefix_sum: &[f64],
    lookback: usize,
    first: usize,
    out: &mut [f64],
) {
    let warmup = first.saturating_add(lookback.saturating_sub(1));
    let inv_lb = 1.0 / lookback as f64;

    for (t, slot) in out.iter_mut().enumerate() {
        if t < warmup {
            *slot = f64::NAN;
            continue;
        }

        let window_start = t + 1 - lookback;
        let mut variance = (prefix_sum[t + 1] - prefix_sum[window_start]) * inv_lb;
        if variance < 0.0 {
            variance = 0.0;
        }
        *slot = variance.sqrt();
    }
}

#[inline(always)]
fn garman_klass_prepare<'a>(
    input: &'a GarmanKlassVolatilityInput,
    _kernel: Kernel,
) -> Result<
    (
        &'a [f64],
        &'a [f64],
        &'a [f64],
        &'a [f64],
        usize,
        usize,
        bool,
    ),
    GarmanKlassVolatilityError,
> {
    let (open, high, low, close): (&[f64], &[f64], &[f64], &[f64]) = match &input.data {
        GarmanKlassVolatilityData::Candles { candles } => {
            (&candles.open, &candles.high, &candles.low, &candles.close)
        }
        GarmanKlassVolatilityData::Slices {
            open,
            high,
            low,
            close,
        } => (open, high, low, close),
    };

    let len = close.len();
    if len == 0 {
        return Err(GarmanKlassVolatilityError::EmptyInputData);
    }
    if open.len() != len || high.len() != len || low.len() != len {
        return Err(GarmanKlassVolatilityError::InconsistentSliceLengths {
            open_len: open.len(),
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }

    let (first, valid, all_valid) = validity_summary(open, high, low, close);
    if first >= len {
        return Err(GarmanKlassVolatilityError::AllValuesNaN);
    }

    let lookback = input.get_lookback();
    if lookback == 0 || lookback > len {
        return Err(GarmanKlassVolatilityError::InvalidLookback {
            lookback,
            data_len: len,
        });
    }

    if valid < lookback {
        return Err(GarmanKlassVolatilityError::NotEnoughValidData {
            needed: lookback,
            valid,
        });
    }

    Ok((open, high, low, close, lookback, first, all_valid))
}

#[inline]
pub fn garman_klass_volatility_with_kernel(
    input: &GarmanKlassVolatilityInput,
    kernel: Kernel,
) -> Result<GarmanKlassVolatilityOutput, GarmanKlassVolatilityError> {
    let (open, high, low, close, lookback, first, all_valid) = garman_klass_prepare(input, kernel)?;
    let len = close.len();
    let mut values = alloc_uninit_f64(len);
    if all_valid {
        let prefix_sum = build_prefix_sum_terms_all_valid(open, high, low, close);
        gk_row_from_prefix_all_valid(&prefix_sum, lookback, first, &mut values);
    } else {
        let (prefix_valid, prefix_sum) = build_prefix_terms(open, high, low, close);
        gk_row_from_prefix(&prefix_valid, &prefix_sum, lookback, first, &mut values);
    }
    Ok(GarmanKlassVolatilityOutput { values })
}

#[inline]
pub fn garman_klass_volatility_into_slice(
    dst: &mut [f64],
    input: &GarmanKlassVolatilityInput,
    kernel: Kernel,
) -> Result<(), GarmanKlassVolatilityError> {
    let (open, high, low, close, lookback, first, all_valid) = garman_klass_prepare(input, kernel)?;
    let expected = close.len();
    if dst.len() != expected {
        return Err(GarmanKlassVolatilityError::OutputLengthMismatch {
            expected,
            got: dst.len(),
        });
    }
    if all_valid {
        let prefix_sum = build_prefix_sum_terms_all_valid(open, high, low, close);
        gk_row_from_prefix_all_valid(&prefix_sum, lookback, first, dst);
    } else {
        let (prefix_valid, prefix_sum) = build_prefix_terms(open, high, low, close);
        gk_row_from_prefix(&prefix_valid, &prefix_sum, lookback, first, dst);
    }
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn garman_klass_volatility_into(
    input: &GarmanKlassVolatilityInput,
    out: &mut [f64],
) -> Result<(), GarmanKlassVolatilityError> {
    garman_klass_volatility_into_slice(out, input, Kernel::Auto)
}

#[derive(Clone, Debug)]
pub struct GarmanKlassVolatilityBatchRange {
    pub lookback: (usize, usize, usize),
}

impl Default for GarmanKlassVolatilityBatchRange {
    fn default() -> Self {
        Self {
            lookback: (14, 252, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct GarmanKlassVolatilityBatchBuilder {
    range: GarmanKlassVolatilityBatchRange,
    kernel: Kernel,
}

impl GarmanKlassVolatilityBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline]
    pub fn lookback_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.lookback = (start, end, step);
        self
    }

    #[inline]
    pub fn lookback_static(mut self, lookback: usize) -> Self {
        self.range.lookback = (lookback, lookback, 0);
        self
    }

    pub fn apply_slices(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<GarmanKlassVolatilityBatchOutput, GarmanKlassVolatilityError> {
        garman_klass_volatility_batch_with_kernel(open, high, low, close, &self.range, self.kernel)
    }

    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<GarmanKlassVolatilityBatchOutput, GarmanKlassVolatilityError> {
        self.apply_slices(&candles.open, &candles.high, &candles.low, &candles.close)
    }

    pub fn with_default_candles(
        candles: &Candles,
    ) -> Result<GarmanKlassVolatilityBatchOutput, GarmanKlassVolatilityError> {
        GarmanKlassVolatilityBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(candles)
    }
}

#[derive(Clone, Debug)]
pub struct GarmanKlassVolatilityBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<GarmanKlassVolatilityParams>,
    pub rows: usize,
    pub cols: usize,
}

impl GarmanKlassVolatilityBatchOutput {
    pub fn row_for_params(&self, params: &GarmanKlassVolatilityParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|combo| combo.lookback.unwrap_or(14) == params.lookback.unwrap_or(14))
    }

    pub fn values_for(&self, params: &GarmanKlassVolatilityParams) -> Option<&[f64]> {
        self.row_for_params(params).and_then(|row| {
            row.checked_mul(self.cols)
                .and_then(|start| self.values.get(start..start + self.cols))
        })
    }
}

#[inline(always)]
fn expand_grid_garman_klass(
    range: &GarmanKlassVolatilityBatchRange,
) -> Result<Vec<GarmanKlassVolatilityParams>, GarmanKlassVolatilityError> {
    fn axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, GarmanKlassVolatilityError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        let step = step.max(1);
        if start < end {
            let mut out = Vec::new();
            let mut x = start;
            while x <= end {
                out.push(x);
                match x.checked_add(step) {
                    Some(next) if next != x => x = next,
                    _ => break,
                }
            }
            if out.is_empty() {
                return Err(GarmanKlassVolatilityError::InvalidRange {
                    start: start.to_string(),
                    end: end.to_string(),
                    step: step.to_string(),
                });
            }
            Ok(out)
        } else {
            let mut out = Vec::new();
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
            if out.is_empty() {
                return Err(GarmanKlassVolatilityError::InvalidRange {
                    start: start.to_string(),
                    end: end.to_string(),
                    step: step.to_string(),
                });
            }
            Ok(out)
        }
    }

    Ok(axis_usize(range.lookback)?
        .into_iter()
        .map(|lookback| GarmanKlassVolatilityParams {
            lookback: Some(lookback),
        })
        .collect())
}

pub fn garman_klass_volatility_batch_with_kernel(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &GarmanKlassVolatilityBatchRange,
    kernel: Kernel,
) -> Result<GarmanKlassVolatilityBatchOutput, GarmanKlassVolatilityError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(GarmanKlassVolatilityError::InvalidKernelForBatch(other)),
    };
    garman_klass_volatility_batch_par_slice(
        open,
        high,
        low,
        close,
        sweep,
        batch_kernel.to_non_batch(),
    )
}

#[inline(always)]
pub fn garman_klass_volatility_batch_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &GarmanKlassVolatilityBatchRange,
    kernel: Kernel,
) -> Result<GarmanKlassVolatilityBatchOutput, GarmanKlassVolatilityError> {
    garman_klass_volatility_batch_inner(open, high, low, close, sweep, kernel, false)
}

#[inline(always)]
pub fn garman_klass_volatility_batch_par_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &GarmanKlassVolatilityBatchRange,
    kernel: Kernel,
) -> Result<GarmanKlassVolatilityBatchOutput, GarmanKlassVolatilityError> {
    garman_klass_volatility_batch_inner(open, high, low, close, sweep, kernel, true)
}

#[inline(always)]
fn garman_klass_volatility_batch_inner(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &GarmanKlassVolatilityBatchRange,
    _kernel: Kernel,
    parallel: bool,
) -> Result<GarmanKlassVolatilityBatchOutput, GarmanKlassVolatilityError> {
    let combos = expand_grid_garman_klass(sweep)?;
    let len = close.len();
    if len == 0 {
        return Err(GarmanKlassVolatilityError::EmptyInputData);
    }
    if open.len() != len || high.len() != len || low.len() != len {
        return Err(GarmanKlassVolatilityError::InconsistentSliceLengths {
            open_len: open.len(),
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }

    let first = first_valid_ohlc(open, high, low, close);
    if first >= len {
        return Err(GarmanKlassVolatilityError::AllValuesNaN);
    }

    let valid = count_valid_ohlc(open, high, low, close);
    let max_lookback = combos
        .iter()
        .map(|combo| combo.lookback.unwrap_or(14))
        .max()
        .unwrap_or(0);
    if max_lookback == 0 || valid < max_lookback {
        return Err(GarmanKlassVolatilityError::NotEnoughValidData {
            needed: max_lookback,
            valid,
        });
    }

    let rows = combos.len();
    let cols = len;
    let mut buf_mu = make_uninit_matrix(rows, cols);
    let warmups: Vec<usize> = combos
        .iter()
        .map(|combo| first.saturating_add(combo.lookback.unwrap_or(14).saturating_sub(1)))
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warmups);

    let mut guard = ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };
    let (prefix_valid, prefix_sum) = build_prefix_terms(open, high, low, close);

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out.par_chunks_mut(cols)
            .enumerate()
            .for_each(|(row, out_row)| {
                let lookback = combos[row].lookback.unwrap_or(14);
                gk_row_from_prefix(&prefix_valid, &prefix_sum, lookback, first, out_row);
            });

        #[cfg(target_arch = "wasm32")]
        for (row, out_row) in out.chunks_mut(cols).enumerate() {
            let lookback = combos[row].lookback.unwrap_or(14);
            gk_row_from_prefix(&prefix_valid, &prefix_sum, lookback, first, out_row);
        }
    } else {
        for (row, out_row) in out.chunks_mut(cols).enumerate() {
            let lookback = combos[row].lookback.unwrap_or(14);
            gk_row_from_prefix(&prefix_valid, &prefix_sum, lookback, first, out_row);
        }
    }

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(GarmanKlassVolatilityBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[cfg(feature = "python")]
#[pyfunction(name = "garman_klass_volatility")]
#[pyo3(signature = (open, high, low, close, lookback=14, kernel=None))]
pub fn garman_klass_volatility_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    lookback: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let open = open.as_slice()?;
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
        return Err(PyValueError::new_err("OHLC slice length mismatch"));
    }

    let kernel = validate_kernel(kernel, false)?;
    let input = GarmanKlassVolatilityInput::from_slices(
        open,
        high,
        low,
        close,
        GarmanKlassVolatilityParams {
            lookback: Some(lookback),
        },
    );
    let output = py
        .allow_threads(|| garman_klass_volatility_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(output.values.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "GarmanKlassVolatilityStream")]
pub struct GarmanKlassVolatilityStreamPy {
    stream: GarmanKlassVolatilityStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl GarmanKlassVolatilityStreamPy {
    #[new]
    fn new(lookback: usize) -> PyResult<Self> {
        let stream = GarmanKlassVolatilityStream::try_new(GarmanKlassVolatilityParams {
            lookback: Some(lookback),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, open: f64, high: f64, low: f64, close: f64) -> Option<f64> {
        self.stream.update(open, high, low, close)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "garman_klass_volatility_batch")]
#[pyo3(signature = (open, high, low, close, lookback_range, kernel=None))]
pub fn garman_klass_volatility_batch_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    lookback_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let open = open.as_slice()?;
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
        return Err(PyValueError::new_err("OHLC slice length mismatch"));
    }

    let sweep = GarmanKlassVolatilityBatchRange {
        lookback: lookback_range,
    };
    let output = {
        let kernel = validate_kernel(kernel, true)?;
        py.allow_threads(|| {
            let batch = match kernel {
                Kernel::Auto => detect_best_batch_kernel(),
                other => other,
            };
            garman_klass_volatility_batch_inner(
                open,
                high,
                low,
                close,
                &sweep,
                batch.to_non_batch(),
                true,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?
    };

    let dict = PyDict::new(py);
    dict.set_item(
        "values",
        output
            .values
            .into_pyarray(py)
            .reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "lookbacks",
        output
            .combos
            .iter()
            .map(|combo| combo.lookback.unwrap_or(14) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", output.rows)?;
    dict.set_item("cols", output.cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_garman_klass_volatility_module(
    module: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(garman_klass_volatility_py, module)?)?;
    module.add_function(wrap_pyfunction!(garman_klass_volatility_batch_py, module)?)?;
    module.add_class::<GarmanKlassVolatilityStreamPy>()?;
    Ok(())
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "garman_klass_volatility_cuda_batch_dev")]
#[pyo3(signature = (open_f32, high_f32, low_f32, close_f32, lookback_range, device_id=0))]
pub fn garman_klass_volatility_cuda_batch_dev_py<'py>(
    py: Python<'py>,
    open_f32: PyReadonlyArray1<'py, f32>,
    high_f32: PyReadonlyArray1<'py, f32>,
    low_f32: PyReadonlyArray1<'py, f32>,
    close_f32: PyReadonlyArray1<'py, f32>,
    lookback_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<(DeviceArrayF32Py, Bound<'py, PyDict>)> {
    use crate::cuda::{cuda_available, CudaGarmanKlassVolatility};

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let open = open_f32.as_slice()?;
    let high = high_f32.as_slice()?;
    let low = low_f32.as_slice()?;
    let close = close_f32.as_slice()?;
    let sweep = GarmanKlassVolatilityBatchRange {
        lookback: lookback_range,
    };
    let result = py.allow_threads(|| {
        let cuda = CudaGarmanKlassVolatility::new(device_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.garman_klass_volatility_batch_dev(open, high, low, close, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    let dict = PyDict::new(py);
    dict.set_item(
        "lookbacks",
        result
            .combos
            .iter()
            .map(|combo| combo.lookback.unwrap_or(14) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    Ok((make_device_array_py(device_id, result.outputs)?, dict))
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "garman_klass_volatility_cuda_many_series_one_param_dev")]
#[pyo3(signature = (open_tm_f32, high_tm_f32, low_tm_f32, close_tm_f32, cols, rows, lookback=14, device_id=0))]
pub fn garman_klass_volatility_cuda_many_series_one_param_dev_py<'py>(
    py: Python<'py>,
    open_tm_f32: PyReadonlyArray1<'py, f32>,
    high_tm_f32: PyReadonlyArray1<'py, f32>,
    low_tm_f32: PyReadonlyArray1<'py, f32>,
    close_tm_f32: PyReadonlyArray1<'py, f32>,
    cols: usize,
    rows: usize,
    lookback: usize,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    use crate::cuda::{cuda_available, CudaGarmanKlassVolatility};

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let open = open_tm_f32.as_slice()?;
    let high = high_tm_f32.as_slice()?;
    let low = low_tm_f32.as_slice()?;
    let close = close_tm_f32.as_slice()?;
    let inner = py.allow_threads(|| {
        let cuda = CudaGarmanKlassVolatility::new(device_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.garman_klass_volatility_many_series_one_param_time_major_dev(
            open, high, low, close, cols, rows, lookback,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    make_device_array_py(device_id, inner)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "garman_klass_volatility_js")]
pub fn garman_klass_volatility_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    lookback: usize,
) -> Result<Vec<f64>, JsValue> {
    let input = GarmanKlassVolatilityInput::from_slices(
        open,
        high,
        low,
        close,
        GarmanKlassVolatilityParams {
            lookback: Some(lookback),
        },
    );
    let mut output = vec![0.0; close.len()];
    garman_klass_volatility_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(output)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn garman_klass_volatility_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn garman_klass_volatility_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn garman_klass_volatility_into(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    lookback: usize,
) -> Result<(), JsValue> {
    if open_ptr.is_null()
        || high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let input = GarmanKlassVolatilityInput::from_slices(
            open,
            high,
            low,
            close,
            GarmanKlassVolatilityParams {
                lookback: Some(lookback),
            },
        );

        if open_ptr == out_ptr || high_ptr == out_ptr || low_ptr == out_ptr || close_ptr == out_ptr
        {
            let mut tmp = vec![0.0; len];
            garman_klass_volatility_into_slice(&mut tmp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            std::slice::from_raw_parts_mut(out_ptr, len).copy_from_slice(&tmp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            garman_klass_volatility_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct GarmanKlassVolatilityBatchConfig {
    pub lookback_range: (usize, usize, usize),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct GarmanKlassVolatilityBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<GarmanKlassVolatilityParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "garman_klass_volatility_batch_js")]
pub fn garman_klass_volatility_batch_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: GarmanKlassVolatilityBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let sweep = GarmanKlassVolatilityBatchRange {
        lookback: config.lookback_range,
    };
    let output = garman_klass_volatility_batch_inner(
        open,
        high,
        low,
        close,
        &sweep,
        detect_best_kernel(),
        false,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&GarmanKlassVolatilityBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn garman_klass_volatility_batch_into(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    lookback_start: usize,
    lookback_end: usize,
    lookback_step: usize,
) -> Result<usize, JsValue> {
    if open_ptr.is_null()
        || high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    let sweep = GarmanKlassVolatilityBatchRange {
        lookback: (lookback_start, lookback_end, lookback_step),
    };
    let combos = expand_grid_garman_klass(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();

    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let total = rows
            .checked_mul(len)
            .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        let batch = garman_klass_volatility_batch_inner(
            open,
            high,
            low,
            close,
            &sweep,
            detect_best_kernel(),
            false,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        out.copy_from_slice(&batch.values);
    }

    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn garman_klass_volatility_output_into_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    lookback: usize,
    out: &js_sys::Float64Array,
) -> Result<usize, JsValue> {
    let values = garman_klass_volatility_js(open, high, low, close, lookback)?;
    crate::write_wasm_f64_output("garman_klass_volatility_output_into_js", &values, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn garman_klass_volatility_batch_output_into_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = garman_klass_volatility_batch_js(open, high, low, close, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "garman_klass_volatility_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_ohlc(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut open = vec![f64::NAN; len];
        let mut high = vec![f64::NAN; len];
        let mut low = vec![f64::NAN; len];
        let mut close = vec![f64::NAN; len];
        let mut prev = 100.0;
        for i in 2..len {
            let x = i as f64;
            let o = (prev + (x * 0.021).sin() * 1.5 + 0.03 * x).max(1.0);
            let c = (o + (x * 0.017).cos() * 0.8).max(1.0);
            let h = o.max(c) + 0.5 + (x * 0.011).sin().abs() * 0.2;
            let l = (o.min(c) - 0.45 - (x * 0.013).cos().abs() * 0.15).max(0.01);
            open[i] = o;
            high[i] = h;
            low[i] = l;
            close[i] = c;
            prev = c;
        }
        (open, high, low, close)
    }

    #[test]
    fn gk_output_contract() {
        let (open, high, low, close) = sample_ohlc(128);
        let input = GarmanKlassVolatilityInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            GarmanKlassVolatilityParams { lookback: Some(14) },
        );
        let out = garman_klass_volatility(&input).expect("gk");
        assert_eq!(out.values.len(), close.len());
        assert!(out.values.iter().any(|v| v.is_finite()));
        let first_valid = out
            .values
            .iter()
            .position(|v| v.is_finite())
            .expect("first valid");
        assert!(first_valid >= 15);
    }

    #[test]
    fn gk_into_matches_api() {
        let (open, high, low, close) = sample_ohlc(192);
        let input = GarmanKlassVolatilityInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            GarmanKlassVolatilityParams { lookback: Some(20) },
        );
        let api = garman_klass_volatility(&input).expect("api");
        let mut out = vec![0.0; close.len()];
        garman_klass_volatility_into(&input, &mut out).expect("into");
        for i in 0..out.len() {
            if api.values[i].is_nan() {
                assert!(out[i].is_nan(), "expected NaN at index {i}");
            } else {
                assert!(
                    (api.values[i] - out[i]).abs() <= 1e-12,
                    "into mismatch at {i}: {} vs {}",
                    api.values[i],
                    out[i]
                );
            }
        }
    }

    #[test]
    fn gk_stream_matches_batch() {
        let (open, high, low, close) = sample_ohlc(160);
        let input = GarmanKlassVolatilityInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            GarmanKlassVolatilityParams { lookback: Some(12) },
        );
        let batch = garman_klass_volatility(&input).expect("batch");
        let mut stream = GarmanKlassVolatilityStream::try_new(GarmanKlassVolatilityParams {
            lookback: Some(12),
        })
        .expect("stream");
        let mut streamed = Vec::with_capacity(close.len());
        for i in 0..close.len() {
            streamed.push(
                stream
                    .update(open[i], high[i], low[i], close[i])
                    .unwrap_or(f64::NAN),
            );
        }
        for i in 0..streamed.len() {
            if batch.values[i].is_nan() {
                assert!(streamed[i].is_nan(), "stream index {i}");
            } else {
                assert!(
                    (batch.values[i] - streamed[i]).abs() <= 1e-12,
                    "stream mismatch at {i}: {} vs {}",
                    batch.values[i],
                    streamed[i]
                );
            }
        }
    }

    #[test]
    fn gk_batch_single_param_matches_single() {
        let (open, high, low, close) = sample_ohlc(200);
        let single_input = GarmanKlassVolatilityInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            GarmanKlassVolatilityParams { lookback: Some(16) },
        );
        let single = garman_klass_volatility(&single_input).expect("single");
        let batch = garman_klass_volatility_batch_with_kernel(
            &open,
            &high,
            &low,
            &close,
            &GarmanKlassVolatilityBatchRange {
                lookback: (16, 16, 0),
            },
            Kernel::ScalarBatch,
        )
        .expect("batch");
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, close.len());
        for i in 0..batch.values.len() {
            if single.values[i].is_nan() {
                assert!(batch.values[i].is_nan(), "expected NaN at index {i}");
            } else {
                assert!(
                    (batch.values[i] - single.values[i]).abs() <= 1e-12,
                    "batch mismatch at {i}: {} vs {}",
                    batch.values[i],
                    single.values[i]
                );
            }
        }
    }

    #[test]
    fn gk_internal_invalid_bar_produces_nan_window_and_recovers() {
        let (mut open, mut high, mut low, mut close) = sample_ohlc(80);
        open[30] = f64::NAN;
        high[30] = f64::NAN;
        low[30] = f64::NAN;
        close[30] = f64::NAN;

        let input = GarmanKlassVolatilityInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            GarmanKlassVolatilityParams { lookback: Some(10) },
        );
        let out = garman_klass_volatility(&input).expect("gk");
        assert!(out.values[30].is_nan());
        assert!(out.values[39].is_nan());
        assert!(out.values[40].is_finite());
    }

    #[test]
    fn gk_rejects_invalid_lookback() {
        let (open, high, low, close) = sample_ohlc(8);
        let input = GarmanKlassVolatilityInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            GarmanKlassVolatilityParams { lookback: Some(0) },
        );
        let err = garman_klass_volatility(&input).unwrap_err();
        match err {
            GarmanKlassVolatilityError::InvalidLookback { lookback, .. } => {
                assert_eq!(lookback, 0);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
