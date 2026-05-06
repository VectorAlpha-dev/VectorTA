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

#[derive(Debug, Clone)]
pub enum BullPowerVsBearPowerData<'a> {
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
pub struct BullPowerVsBearPowerOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct BullPowerVsBearPowerParams {
    pub period: Option<usize>,
}

impl Default for BullPowerVsBearPowerParams {
    fn default() -> Self {
        Self { period: Some(5) }
    }
}

#[derive(Debug, Clone)]
pub struct BullPowerVsBearPowerInput<'a> {
    pub data: BullPowerVsBearPowerData<'a>,
    pub params: BullPowerVsBearPowerParams,
}

impl<'a> BullPowerVsBearPowerInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: BullPowerVsBearPowerParams) -> Self {
        Self {
            data: BullPowerVsBearPowerData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: BullPowerVsBearPowerParams,
    ) -> Self {
        Self {
            data: BullPowerVsBearPowerData::Slices {
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
        Self::from_candles(candles, BullPowerVsBearPowerParams::default())
    }

    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(5)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct BullPowerVsBearPowerBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for BullPowerVsBearPowerBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl BullPowerVsBearPowerBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn period(mut self, period: usize) -> Self {
        self.period = Some(period);
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
    ) -> Result<BullPowerVsBearPowerOutput, BullPowerVsBearPowerError> {
        let input = BullPowerVsBearPowerInput::from_candles(
            candles,
            BullPowerVsBearPowerParams {
                period: self.period,
            },
        );
        bull_power_vs_bear_power_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<BullPowerVsBearPowerOutput, BullPowerVsBearPowerError> {
        let input = BullPowerVsBearPowerInput::from_slices(
            open,
            high,
            low,
            close,
            BullPowerVsBearPowerParams {
                period: self.period,
            },
        );
        bull_power_vs_bear_power_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<BullPowerVsBearPowerStream, BullPowerVsBearPowerError> {
        BullPowerVsBearPowerStream::try_new(BullPowerVsBearPowerParams {
            period: self.period,
        })
    }
}

#[derive(Debug, Error)]
pub enum BullPowerVsBearPowerError {
    #[error("bull_power_vs_bear_power: Input data slice is empty.")]
    EmptyInputData,
    #[error("bull_power_vs_bear_power: All values are NaN or have zero close.")]
    AllValuesNaN,
    #[error(
        "bull_power_vs_bear_power: Invalid period: period = {period}, data length = {data_len}"
    )]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("bull_power_vs_bear_power: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("bull_power_vs_bear_power: Inconsistent slice lengths: open={open_len}, high={high_len}, low={low_len}, close={close_len}")]
    InconsistentSliceLengths {
        open_len: usize,
        high_len: usize,
        low_len: usize,
        close_len: usize,
    },
    #[error(
        "bull_power_vs_bear_power: Output length mismatch: expected = {expected}, got = {got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("bull_power_vs_bear_power: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("bull_power_vs_bear_power: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Debug, Clone)]
pub struct BullPowerVsBearPowerStream {
    period: usize,
    alpha: f64,
    beta: f64,
    count: usize,
    mean: f64,
}

impl BullPowerVsBearPowerStream {
    pub fn try_new(
        params: BullPowerVsBearPowerParams,
    ) -> Result<BullPowerVsBearPowerStream, BullPowerVsBearPowerError> {
        let period = params.period.unwrap_or(5);
        if period == 0 {
            return Err(BullPowerVsBearPowerError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let alpha = 2.0 / (period as f64 + 1.0);
        Ok(Self {
            period,
            alpha,
            beta: 1.0 - alpha,
            count: 0,
            mean: f64::NAN,
        })
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.count = 0;
        self.mean = f64::NAN;
    }

    #[inline(always)]
    pub fn update(&mut self, open: f64, high: f64, low: f64, close: f64) -> Option<f64> {
        if !valid_ohlc_bar(open, high, low, close) {
            self.reset();
            return None;
        }

        let value = bbpower_value(open, high, low, close);
        self.count += 1;
        if self.count == 1 {
            self.mean = value;
        } else if self.count <= self.period {
            let c = self.count as f64;
            self.mean = ((c - 1.0) * self.mean + value) / c;
        } else {
            self.mean = self.beta.mul_add(self.mean, self.alpha * value);
        }

        if self.count >= self.period {
            Some(self.mean)
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn get_warmup_period(&self) -> usize {
        self.period.saturating_sub(1)
    }
}

#[inline]
pub fn bull_power_vs_bear_power(
    input: &BullPowerVsBearPowerInput,
) -> Result<BullPowerVsBearPowerOutput, BullPowerVsBearPowerError> {
    bull_power_vs_bear_power_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn valid_ohlc_bar(open: f64, high: f64, low: f64, close: f64) -> bool {
    open.is_finite() && high.is_finite() && low.is_finite() && close.is_finite() && close != 0.0
}

#[inline(always)]
fn bbpower_value(open: f64, high: f64, low: f64, close: f64) -> f64 {
    ((high + low) - (2.0 * open)) * (100.0 / close)
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
fn first_and_valid_ohlc(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> (usize, usize) {
    let mut first = close.len();
    let mut count = 0usize;
    for i in 0..close.len() {
        if valid_ohlc_bar(open[i], high[i], low[i], close[i]) {
            if first == close.len() {
                first = i;
            }
            count += 1;
        }
    }
    (first, count)
}

#[inline(always)]
fn build_raw_series(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<u8>) {
    let len = close.len();
    let mut values = vec![0.0; len];
    let mut valid = vec![0u8; len];
    for i in 0..len {
        if valid_ohlc_bar(open[i], high[i], low[i], close[i]) {
            values[i] = bbpower_value(open[i], high[i], low[i], close[i]);
            valid[i] = 1;
        }
    }
    (values, valid)
}

#[inline(always)]
fn bbpower_row_from_ohlc(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    out: &mut [f64],
) {
    let alpha = 2.0 / (period as f64 + 1.0);
    let beta = 1.0 - alpha;
    let mut count = 0usize;
    let mut mean = f64::NAN;

    for i in 0..close.len() {
        if !valid_ohlc_bar(open[i], high[i], low[i], close[i]) {
            out[i] = f64::NAN;
            count = 0;
            mean = f64::NAN;
            continue;
        }

        let value = bbpower_value(open[i], high[i], low[i], close[i]);
        count += 1;
        if count == 1 {
            mean = value;
        } else if count <= period {
            let c = count as f64;
            mean = ((c - 1.0) * mean + value) / c;
        } else {
            mean = beta.mul_add(mean, alpha * value);
        }

        out[i] = if count >= period { mean } else { f64::NAN };
    }
}

#[inline(always)]
fn bbpower_row_from_raw(raw: &[f64], valid: &[u8], period: usize, out: &mut [f64]) {
    let alpha = 2.0 / (period as f64 + 1.0);
    let beta = 1.0 - alpha;
    let mut count = 0usize;
    let mut mean = f64::NAN;

    for i in 0..raw.len() {
        if valid[i] == 0 {
            out[i] = f64::NAN;
            count = 0;
            mean = f64::NAN;
            continue;
        }

        let value = raw[i];
        count += 1;
        if count == 1 {
            mean = value;
        } else if count <= period {
            let c = count as f64;
            mean = ((c - 1.0) * mean + value) / c;
        } else {
            mean = beta.mul_add(mean, alpha * value);
        }

        out[i] = if count >= period { mean } else { f64::NAN };
    }
}

#[inline(always)]
fn bull_power_vs_bear_power_prepare<'a>(
    input: &'a BullPowerVsBearPowerInput,
    kernel: Kernel,
) -> Result<
    (
        &'a [f64],
        &'a [f64],
        &'a [f64],
        &'a [f64],
        usize,
        usize,
        Kernel,
    ),
    BullPowerVsBearPowerError,
> {
    let (open, high, low, close): (&[f64], &[f64], &[f64], &[f64]) = match &input.data {
        BullPowerVsBearPowerData::Candles { candles } => {
            (&candles.open, &candles.high, &candles.low, &candles.close)
        }
        BullPowerVsBearPowerData::Slices {
            open,
            high,
            low,
            close,
        } => (open, high, low, close),
    };

    let len = close.len();
    if len == 0 {
        return Err(BullPowerVsBearPowerError::EmptyInputData);
    }
    if open.len() != len || high.len() != len || low.len() != len {
        return Err(BullPowerVsBearPowerError::InconsistentSliceLengths {
            open_len: open.len(),
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }

    let (first, valid) = first_and_valid_ohlc(open, high, low, close);
    if first >= len {
        return Err(BullPowerVsBearPowerError::AllValuesNaN);
    }

    let period = input.get_period();
    if period == 0 || period > len {
        return Err(BullPowerVsBearPowerError::InvalidPeriod {
            period,
            data_len: len,
        });
    }

    if valid < period {
        return Err(BullPowerVsBearPowerError::NotEnoughValidData {
            needed: period,
            valid,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other.to_non_batch(),
    };

    Ok((open, high, low, close, period, first, chosen))
}

#[inline]
pub fn bull_power_vs_bear_power_with_kernel(
    input: &BullPowerVsBearPowerInput,
    kernel: Kernel,
) -> Result<BullPowerVsBearPowerOutput, BullPowerVsBearPowerError> {
    let (open, high, low, close, period, first, _chosen) =
        bull_power_vs_bear_power_prepare(input, kernel)?;
    let len = close.len();
    let _warmup = first.saturating_add(period.saturating_sub(1));
    let mut values = alloc_uninit_f64(len);
    bbpower_row_from_ohlc(open, high, low, close, period, &mut values);
    Ok(BullPowerVsBearPowerOutput { values })
}

#[inline]
pub fn bull_power_vs_bear_power_into_slice(
    dst: &mut [f64],
    input: &BullPowerVsBearPowerInput,
    kernel: Kernel,
) -> Result<(), BullPowerVsBearPowerError> {
    let (open, high, low, close, period, _first, _chosen) =
        bull_power_vs_bear_power_prepare(input, kernel)?;
    let expected = close.len();
    if dst.len() != expected {
        return Err(BullPowerVsBearPowerError::OutputLengthMismatch {
            expected,
            got: dst.len(),
        });
    }
    bbpower_row_from_ohlc(open, high, low, close, period, dst);
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn bull_power_vs_bear_power_into(
    input: &BullPowerVsBearPowerInput,
    out: &mut [f64],
) -> Result<(), BullPowerVsBearPowerError> {
    bull_power_vs_bear_power_into_slice(out, input, Kernel::Auto)
}

#[derive(Clone, Debug)]
pub struct BullPowerVsBearPowerBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for BullPowerVsBearPowerBatchRange {
    fn default() -> Self {
        Self {
            period: (5, 252, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct BullPowerVsBearPowerBatchBuilder {
    range: BullPowerVsBearPowerBatchRange,
    kernel: Kernel,
}

impl BullPowerVsBearPowerBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline]
    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.period = (start, end, step);
        self
    }

    #[inline]
    pub fn period_static(mut self, period: usize) -> Self {
        self.range.period = (period, period, 0);
        self
    }

    pub fn apply_slices(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<BullPowerVsBearPowerBatchOutput, BullPowerVsBearPowerError> {
        bull_power_vs_bear_power_batch_with_kernel(open, high, low, close, &self.range, self.kernel)
    }

    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<BullPowerVsBearPowerBatchOutput, BullPowerVsBearPowerError> {
        self.apply_slices(&candles.open, &candles.high, &candles.low, &candles.close)
    }

    pub fn with_default_candles(
        candles: &Candles,
    ) -> Result<BullPowerVsBearPowerBatchOutput, BullPowerVsBearPowerError> {
        BullPowerVsBearPowerBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(candles)
    }
}

#[derive(Clone, Debug)]
pub struct BullPowerVsBearPowerBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<BullPowerVsBearPowerParams>,
    pub rows: usize,
    pub cols: usize,
}

impl BullPowerVsBearPowerBatchOutput {
    pub fn row_for_params(&self, params: &BullPowerVsBearPowerParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|combo| combo.period.unwrap_or(5) == params.period.unwrap_or(5))
    }

    pub fn values_for(&self, params: &BullPowerVsBearPowerParams) -> Option<&[f64]> {
        self.row_for_params(params).and_then(|row| {
            row.checked_mul(self.cols)
                .and_then(|start| self.values.get(start..start + self.cols))
        })
    }
}

#[inline(always)]
fn expand_grid_bull_power_vs_bear_power(
    range: &BullPowerVsBearPowerBatchRange,
) -> Result<Vec<BullPowerVsBearPowerParams>, BullPowerVsBearPowerError> {
    fn axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, BullPowerVsBearPowerError> {
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
                return Err(BullPowerVsBearPowerError::InvalidRange {
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
                return Err(BullPowerVsBearPowerError::InvalidRange {
                    start: start.to_string(),
                    end: end.to_string(),
                    step: step.to_string(),
                });
            }
            Ok(out)
        }
    }

    let periods = axis_usize(range.period)?;
    if let Some(&period) = periods.iter().find(|&&period| period == 0) {
        return Err(BullPowerVsBearPowerError::InvalidPeriod {
            period,
            data_len: 0,
        });
    }
    Ok(periods
        .into_iter()
        .map(|period| BullPowerVsBearPowerParams {
            period: Some(period),
        })
        .collect())
}

pub fn bull_power_vs_bear_power_batch_with_kernel(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &BullPowerVsBearPowerBatchRange,
    kernel: Kernel,
) -> Result<BullPowerVsBearPowerBatchOutput, BullPowerVsBearPowerError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(BullPowerVsBearPowerError::InvalidKernelForBatch(other)),
    };
    bull_power_vs_bear_power_batch_par_slice(
        open,
        high,
        low,
        close,
        sweep,
        batch_kernel.to_non_batch(),
    )
}

#[inline(always)]
pub fn bull_power_vs_bear_power_batch_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &BullPowerVsBearPowerBatchRange,
    kernel: Kernel,
) -> Result<BullPowerVsBearPowerBatchOutput, BullPowerVsBearPowerError> {
    bull_power_vs_bear_power_batch_inner(open, high, low, close, sweep, kernel, false)
}

#[inline(always)]
pub fn bull_power_vs_bear_power_batch_par_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &BullPowerVsBearPowerBatchRange,
    kernel: Kernel,
) -> Result<BullPowerVsBearPowerBatchOutput, BullPowerVsBearPowerError> {
    bull_power_vs_bear_power_batch_inner(open, high, low, close, sweep, kernel, true)
}

#[inline(always)]
fn bull_power_vs_bear_power_batch_inner(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &BullPowerVsBearPowerBatchRange,
    _kernel: Kernel,
    parallel: bool,
) -> Result<BullPowerVsBearPowerBatchOutput, BullPowerVsBearPowerError> {
    let combos = expand_grid_bull_power_vs_bear_power(sweep)?;
    let len = close.len();
    if len == 0 {
        return Err(BullPowerVsBearPowerError::EmptyInputData);
    }
    if open.len() != len || high.len() != len || low.len() != len {
        return Err(BullPowerVsBearPowerError::InconsistentSliceLengths {
            open_len: open.len(),
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }

    let first = first_valid_ohlc(open, high, low, close);
    if first >= len {
        return Err(BullPowerVsBearPowerError::AllValuesNaN);
    }

    let valid = count_valid_ohlc(open, high, low, close);
    let max_period = combos
        .iter()
        .map(|combo| combo.period.unwrap_or(5))
        .max()
        .unwrap_or(0);
    if max_period == 0 || valid < max_period {
        return Err(BullPowerVsBearPowerError::NotEnoughValidData {
            needed: max_period,
            valid,
        });
    }

    let rows = combos.len();
    let cols = len;
    let mut buf_mu = make_uninit_matrix(rows, cols);
    let warmups: Vec<usize> = combos
        .iter()
        .map(|combo| first.saturating_add(combo.period.unwrap_or(5).saturating_sub(1)))
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warmups);

    let mut guard = ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };
    let (raw, valid_flags) = build_raw_series(open, high, low, close);

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out.par_chunks_mut(cols)
            .enumerate()
            .for_each(|(row, out_row)| {
                bbpower_row_from_raw(&raw, &valid_flags, combos[row].period.unwrap_or(5), out_row);
            });

        #[cfg(target_arch = "wasm32")]
        for (row, out_row) in out.chunks_mut(cols).enumerate() {
            bbpower_row_from_raw(&raw, &valid_flags, combos[row].period.unwrap_or(5), out_row);
        }
    } else {
        for (row, out_row) in out.chunks_mut(cols).enumerate() {
            bbpower_row_from_raw(&raw, &valid_flags, combos[row].period.unwrap_or(5), out_row);
        }
    }

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(BullPowerVsBearPowerBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub fn bull_power_vs_bear_power_batch_inner_into(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &BullPowerVsBearPowerBatchRange,
    _kernel: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<BullPowerVsBearPowerParams>, BullPowerVsBearPowerError> {
    let combos = expand_grid_bull_power_vs_bear_power(sweep)?;
    let rows = combos.len();
    let cols = close.len();
    if cols == 0 {
        return Err(BullPowerVsBearPowerError::EmptyInputData);
    }
    if open.len() != cols || high.len() != cols || low.len() != cols {
        return Err(BullPowerVsBearPowerError::InconsistentSliceLengths {
            open_len: open.len(),
            high_len: high.len(),
            low_len: low.len(),
            close_len: cols,
        });
    }
    let total =
        rows.checked_mul(cols)
            .ok_or_else(|| BullPowerVsBearPowerError::OutputLengthMismatch {
                expected: usize::MAX,
                got: out.len(),
            })?;
    if out.len() != total {
        return Err(BullPowerVsBearPowerError::OutputLengthMismatch {
            expected: total,
            got: out.len(),
        });
    }

    let first = first_valid_ohlc(open, high, low, close);
    if first >= cols {
        return Err(BullPowerVsBearPowerError::AllValuesNaN);
    }
    let valid = count_valid_ohlc(open, high, low, close);
    let max_period = combos
        .iter()
        .map(|combo| combo.period.unwrap_or(5))
        .max()
        .unwrap_or(0);
    if valid < max_period {
        return Err(BullPowerVsBearPowerError::NotEnoughValidData {
            needed: max_period,
            valid,
        });
    }

    let (raw, valid_flags) = build_raw_series(open, high, low, close);

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out.par_chunks_mut(cols)
            .enumerate()
            .for_each(|(row, out_row)| {
                bbpower_row_from_raw(&raw, &valid_flags, combos[row].period.unwrap_or(5), out_row);
            });

        #[cfg(target_arch = "wasm32")]
        for (row, out_row) in out.chunks_mut(cols).enumerate() {
            bbpower_row_from_raw(&raw, &valid_flags, combos[row].period.unwrap_or(5), out_row);
        }
    } else {
        for (row, out_row) in out.chunks_mut(cols).enumerate() {
            bbpower_row_from_raw(&raw, &valid_flags, combos[row].period.unwrap_or(5), out_row);
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "bull_power_vs_bear_power")]
#[pyo3(signature = (open, high, low, close, period=5, kernel=None))]
pub fn bull_power_vs_bear_power_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    period: usize,
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
    let input = BullPowerVsBearPowerInput::from_slices(
        open,
        high,
        low,
        close,
        BullPowerVsBearPowerParams {
            period: Some(period),
        },
    );
    let output = py
        .allow_threads(|| bull_power_vs_bear_power_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(output.values.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "BullPowerVsBearPowerStream")]
pub struct BullPowerVsBearPowerStreamPy {
    stream: BullPowerVsBearPowerStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl BullPowerVsBearPowerStreamPy {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        let stream = BullPowerVsBearPowerStream::try_new(BullPowerVsBearPowerParams {
            period: Some(period),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, open: f64, high: f64, low: f64, close: f64) -> Option<f64> {
        self.stream.update(open, high, low, close)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "bull_power_vs_bear_power_batch")]
#[pyo3(signature = (open, high, low, close, period_range, kernel=None))]
pub fn bull_power_vs_bear_power_batch_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let open = open.as_slice()?;
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
        return Err(PyValueError::new_err("OHLC slice length mismatch"));
    }

    let kernel = validate_kernel(kernel, true)?;
    let sweep = BullPowerVsBearPowerBatchRange {
        period: period_range,
    };
    let combos = expand_grid_bull_power_vs_bear_power(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = close.len();
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
            bull_power_vs_bear_power_batch_inner_into(
                open,
                high,
                low,
                close,
                &sweep,
                batch.to_non_batch(),
                true,
                slice_out,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "periods",
        combos
            .iter()
            .map(|combo| combo.period.unwrap_or(5) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_bull_power_vs_bear_power_module(
    module: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(bull_power_vs_bear_power_py, module)?)?;
    module.add_function(wrap_pyfunction!(bull_power_vs_bear_power_batch_py, module)?)?;
    module.add_class::<BullPowerVsBearPowerStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "bull_power_vs_bear_power_js")]
pub fn bull_power_vs_bear_power_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
) -> Result<Vec<f64>, JsValue> {
    let input = BullPowerVsBearPowerInput::from_slices(
        open,
        high,
        low,
        close,
        BullPowerVsBearPowerParams {
            period: Some(period),
        },
    );
    let mut output = vec![0.0; close.len()];
    bull_power_vs_bear_power_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(output)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn bull_power_vs_bear_power_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn bull_power_vs_bear_power_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn bull_power_vs_bear_power_into(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
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
        let input = BullPowerVsBearPowerInput::from_slices(
            open,
            high,
            low,
            close,
            BullPowerVsBearPowerParams {
                period: Some(period),
            },
        );
        if open_ptr == out_ptr || high_ptr == out_ptr || low_ptr == out_ptr || close_ptr == out_ptr
        {
            let mut tmp = vec![0.0; len];
            bull_power_vs_bear_power_into_slice(&mut tmp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            std::slice::from_raw_parts_mut(out_ptr, len).copy_from_slice(&tmp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            bull_power_vs_bear_power_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct BullPowerVsBearPowerBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct BullPowerVsBearPowerBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<BullPowerVsBearPowerParams>,
    pub periods: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "bull_power_vs_bear_power_batch_js")]
pub fn bull_power_vs_bear_power_batch_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: BullPowerVsBearPowerBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let sweep = BullPowerVsBearPowerBatchRange {
        period: config.period_range,
    };
    let output = bull_power_vs_bear_power_batch_inner(
        open,
        high,
        low,
        close,
        &sweep,
        detect_best_kernel(),
        false,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&BullPowerVsBearPowerBatchJsOutput {
        periods: output
            .combos
            .iter()
            .map(|combo| combo.period.unwrap_or(5))
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
pub fn bull_power_vs_bear_power_batch_into(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if open_ptr.is_null()
        || high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    let sweep = BullPowerVsBearPowerBatchRange {
        period: (period_start, period_end, period_step),
    };
    let combos = expand_grid_bull_power_vs_bear_power(&sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
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
        bull_power_vs_bear_power_batch_inner_into(
            open,
            high,
            low,
            close,
            &sweep,
            detect_best_kernel(),
            false,
            out,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }

    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn bull_power_vs_bear_power_output_into_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
    out: &js_sys::Float64Array,
) -> Result<usize, JsValue> {
    let values = bull_power_vs_bear_power_js(open, high, low, close, period)?;
    crate::write_wasm_f64_output("bull_power_vs_bear_power_output_into_js", &values, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn bull_power_vs_bear_power_batch_output_into_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = bull_power_vs_bear_power_batch_js(open, high, low, close, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "bull_power_vs_bear_power_batch_output_into_js",
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
            let o = prev + (x * 0.031).sin() * 1.25 + 0.02 * x;
            let c = o + (x * 0.027).cos() * 0.9;
            let h = o.max(c) + 0.45 + (x * 0.011).sin().abs() * 0.18;
            let l = o.min(c) - 0.40 - (x * 0.013).cos().abs() * 0.14;
            open[i] = o;
            high[i] = h;
            low[i] = l;
            close[i] = c;
            prev = c;
        }
        (open, high, low, close)
    }

    #[test]
    fn bull_power_vs_bear_power_output_contract() {
        let (open, high, low, close) = sample_ohlc(128);
        let input = BullPowerVsBearPowerInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            BullPowerVsBearPowerParams { period: Some(5) },
        );
        let out = bull_power_vs_bear_power(&input).expect("bbpower");
        assert_eq!(out.values.len(), close.len());
        let first_valid = out
            .values
            .iter()
            .position(|v| v.is_finite())
            .expect("first valid");
        assert!(first_valid >= 6);
    }

    #[test]
    fn bull_power_vs_bear_power_auto_matches_scalar() {
        let (open, high, low, close) = sample_ohlc(192);
        let input = BullPowerVsBearPowerInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            BullPowerVsBearPowerParams { period: Some(9) },
        );
        let auto = bull_power_vs_bear_power_with_kernel(&input, Kernel::Auto).expect("auto");
        let scalar = bull_power_vs_bear_power_with_kernel(&input, Kernel::Scalar).expect("scalar");
        for i in 0..close.len() {
            if auto.values[i].is_nan() {
                assert!(scalar.values[i].is_nan(), "expected NaN at {i}");
            } else {
                assert!((auto.values[i] - scalar.values[i]).abs() <= 1e-12);
            }
        }
    }

    #[test]
    fn bull_power_vs_bear_power_into_matches_api() {
        let (open, high, low, close) = sample_ohlc(192);
        let input = BullPowerVsBearPowerInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            BullPowerVsBearPowerParams { period: Some(7) },
        );
        let api = bull_power_vs_bear_power(&input).expect("api");
        let mut out = vec![0.0; close.len()];
        bull_power_vs_bear_power_into(&input, &mut out).expect("into");
        for i in 0..out.len() {
            if api.values[i].is_nan() {
                assert!(out[i].is_nan(), "expected NaN at index {i}");
            } else {
                assert!((api.values[i] - out[i]).abs() <= 1e-12);
            }
        }
    }

    #[test]
    fn bull_power_vs_bear_power_stream_matches_batch() {
        let (open, high, low, close) = sample_ohlc(160);
        let input = BullPowerVsBearPowerInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            BullPowerVsBearPowerParams { period: Some(6) },
        );
        let batch = bull_power_vs_bear_power(&input).expect("batch");
        let mut stream =
            BullPowerVsBearPowerStream::try_new(BullPowerVsBearPowerParams { period: Some(6) })
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
                assert!(streamed[i].is_nan(), "expected NaN at {i}");
            } else {
                assert!((batch.values[i] - streamed[i]).abs() <= 1e-12);
            }
        }
    }

    #[test]
    fn bull_power_vs_bear_power_batch_single_param_matches_single() {
        let (open, high, low, close) = sample_ohlc(200);
        let single = bull_power_vs_bear_power(&BullPowerVsBearPowerInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            BullPowerVsBearPowerParams { period: Some(5) },
        ))
        .expect("single");
        let batch = bull_power_vs_bear_power_batch_with_kernel(
            &open,
            &high,
            &low,
            &close,
            &BullPowerVsBearPowerBatchRange { period: (5, 5, 0) },
            Kernel::ScalarBatch,
        )
        .expect("batch");
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, close.len());
        for i in 0..single.values.len() {
            if single.values[i].is_nan() {
                assert!(batch.values[i].is_nan(), "expected NaN at {i}");
            } else {
                assert!((single.values[i] - batch.values[i]).abs() <= 1e-12);
            }
        }
    }

    #[test]
    fn bull_power_vs_bear_power_invalid_window_recovers() {
        let (open, high, low, mut close) = sample_ohlc(80);
        close[30] = f64::NAN;
        let input = BullPowerVsBearPowerInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            BullPowerVsBearPowerParams { period: Some(5) },
        );
        let out = bull_power_vs_bear_power(&input).expect("bbpower");
        assert!(out.values[30].is_nan());
        assert!(out.values[34].is_nan());
        assert!(out.values[35].is_finite());
    }

    #[test]
    fn bull_power_vs_bear_power_rejects_invalid_period() {
        let (open, high, low, close) = sample_ohlc(8);
        let input = BullPowerVsBearPowerInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            BullPowerVsBearPowerParams { period: Some(0) },
        );
        let err = bull_power_vs_bear_power(&input).unwrap_err();
        match err {
            BullPowerVsBearPowerError::InvalidPeriod { period, .. } => assert_eq!(period, 0),
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
