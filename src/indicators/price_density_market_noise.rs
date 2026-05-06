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
    alloc_with_nan_prefix, detect_best_batch_kernel, init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::collections::VecDeque;
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum PriceDensityMarketNoiseData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct PriceDensityMarketNoiseOutput {
    pub price_density: Vec<f64>,
    pub price_density_percent: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct PriceDensityMarketNoiseParams {
    pub length: Option<usize>,
    pub eval_period: Option<usize>,
}

impl Default for PriceDensityMarketNoiseParams {
    fn default() -> Self {
        Self {
            length: Some(14),
            eval_period: Some(200),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PriceDensityMarketNoiseInput<'a> {
    pub data: PriceDensityMarketNoiseData<'a>,
    pub params: PriceDensityMarketNoiseParams,
}

impl<'a> PriceDensityMarketNoiseInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: PriceDensityMarketNoiseParams) -> Self {
        Self {
            data: PriceDensityMarketNoiseData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: PriceDensityMarketNoiseParams,
    ) -> Self {
        Self {
            data: PriceDensityMarketNoiseData::Slices { high, low, close },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, PriceDensityMarketNoiseParams::default())
    }

    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(14)
    }

    #[inline]
    pub fn get_eval_period(&self) -> usize {
        self.params.eval_period.unwrap_or(200)
    }

    #[inline]
    pub fn as_refs(&'a self) -> (&'a [f64], &'a [f64], &'a [f64]) {
        match &self.data {
            PriceDensityMarketNoiseData::Candles { candles } => (
                candles.high.as_slice(),
                candles.low.as_slice(),
                candles.close.as_slice(),
            ),
            PriceDensityMarketNoiseData::Slices { high, low, close } => (*high, *low, *close),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct PriceDensityMarketNoiseBuilder {
    length: Option<usize>,
    eval_period: Option<usize>,
    kernel: Kernel,
}

impl Default for PriceDensityMarketNoiseBuilder {
    fn default() -> Self {
        Self {
            length: None,
            eval_period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl PriceDensityMarketNoiseBuilder {
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
    pub fn eval_period(mut self, value: usize) -> Self {
        self.eval_period = Some(value);
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
    ) -> Result<PriceDensityMarketNoiseOutput, PriceDensityMarketNoiseError> {
        let input = PriceDensityMarketNoiseInput::from_candles(
            candles,
            PriceDensityMarketNoiseParams {
                length: self.length,
                eval_period: self.eval_period,
            },
        );
        price_density_market_noise_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<PriceDensityMarketNoiseOutput, PriceDensityMarketNoiseError> {
        let input = PriceDensityMarketNoiseInput::from_slices(
            high,
            low,
            close,
            PriceDensityMarketNoiseParams {
                length: self.length,
                eval_period: self.eval_period,
            },
        );
        price_density_market_noise_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<PriceDensityMarketNoiseStream, PriceDensityMarketNoiseError> {
        PriceDensityMarketNoiseStream::try_new(PriceDensityMarketNoiseParams {
            length: self.length,
            eval_period: self.eval_period,
        })
    }
}

#[derive(Debug, Error)]
pub enum PriceDensityMarketNoiseError {
    #[error("price_density_market_noise: Empty input data.")]
    EmptyInputData,
    #[error("price_density_market_noise: Data length mismatch across high, low, and close.")]
    DataLengthMismatch,
    #[error("price_density_market_noise: All values are NaN.")]
    AllValuesNaN,
    #[error(
        "price_density_market_noise: Invalid length: length = {length}, data length = {data_len}"
    )]
    InvalidLength { length: usize, data_len: usize },
    #[error(
        "price_density_market_noise: Invalid eval period: eval_period = {eval_period}, data length = {data_len}"
    )]
    InvalidEvalPeriod { eval_period: usize, data_len: usize },
    #[error(
        "price_density_market_noise: Not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "price_density_market_noise: Output length mismatch: expected = {expected}, got = {got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("price_density_market_noise: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("price_density_market_noise: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline(always)]
fn valid_bar(high: f64, low: f64, close: f64) -> bool {
    high.is_finite() && low.is_finite() && close.is_finite()
}

#[inline(always)]
fn first_valid_bar(high: &[f64], low: &[f64], close: &[f64]) -> Option<usize> {
    (0..close.len()).find(|&i| valid_bar(high[i], low[i], close[i]))
}

#[inline(always)]
fn count_valid_from(high: &[f64], low: &[f64], close: &[f64], start: usize) -> usize {
    (start..close.len())
        .filter(|&i| valid_bar(high[i], low[i], close[i]))
        .count()
}

#[inline(always)]
fn price_density_warmup(length: usize, first: usize) -> usize {
    first + length - 1
}

#[inline(always)]
fn price_density_percent_warmup(length: usize, eval_period: usize, first: usize) -> usize {
    price_density_warmup(length, first) + eval_period - 1
}

#[inline(always)]
fn true_range(high: f64, low: f64, prev_close: Option<f64>) -> f64 {
    match prev_close {
        Some(prev) => (high - low)
            .max((high - prev).abs())
            .max((low - prev).abs()),
        None => high - low,
    }
}

#[inline(always)]
fn sorted_remove_one(sorted: &mut Vec<f64>, value: f64) {
    let pos = sorted.partition_point(|x| *x < value);
    if pos < sorted.len() && sorted[pos] == value {
        sorted.remove(pos);
        return;
    }
    if let Ok(idx) = sorted.binary_search_by(|x| x.total_cmp(&value)) {
        sorted.remove(idx);
    }
}

#[derive(Clone, Debug)]
pub struct PriceDensityMarketNoiseStream {
    length: usize,
    eval_period: usize,
    index: usize,
    prev_close: Option<f64>,
    tr_window: Vec<f64>,
    tr_head: usize,
    tr_len: usize,
    tr_sum: f64,
    high_max: VecDeque<(usize, f64)>,
    low_min: VecDeque<(usize, f64)>,
    pd_window: Vec<f64>,
    pd_head: usize,
    pd_len: usize,
    pd_sorted: Vec<f64>,
    invalid_pd: usize,
}

impl PriceDensityMarketNoiseStream {
    #[inline]
    fn from_parts(length: usize, eval_period: usize) -> Self {
        Self {
            length,
            eval_period,
            index: 0,
            prev_close: None,
            tr_window: vec![0.0; length.max(1)],
            tr_head: 0,
            tr_len: 0,
            tr_sum: 0.0,
            high_max: VecDeque::with_capacity(length.max(1)),
            low_min: VecDeque::with_capacity(length.max(1)),
            pd_window: vec![f64::NAN; eval_period.max(1)],
            pd_head: 0,
            pd_len: 0,
            pd_sorted: Vec::with_capacity(eval_period.max(1)),
            invalid_pd: 0,
        }
    }

    #[inline]
    pub fn try_new(
        params: PriceDensityMarketNoiseParams,
    ) -> Result<Self, PriceDensityMarketNoiseError> {
        let length = params.length.unwrap_or(14);
        let eval_period = params.eval_period.unwrap_or(200);
        if length == 0 {
            return Err(PriceDensityMarketNoiseError::InvalidLength {
                length,
                data_len: 0,
            });
        }
        if eval_period == 0 {
            return Err(PriceDensityMarketNoiseError::InvalidEvalPeriod {
                eval_period,
                data_len: 0,
            });
        }
        Ok(Self::from_parts(length, eval_period))
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.index = 0;
        self.prev_close = None;
        self.tr_window.fill(0.0);
        self.tr_head = 0;
        self.tr_len = 0;
        self.tr_sum = 0.0;
        self.high_max.clear();
        self.low_min.clear();
        self.pd_window.fill(f64::NAN);
        self.pd_head = 0;
        self.pd_len = 0;
        self.pd_sorted.clear();
        self.invalid_pd = 0;
    }

    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64)> {
        if !valid_bar(high, low, close) {
            return None;
        }

        let current_index = self.index;
        self.index += 1;

        let tr = true_range(high, low, self.prev_close);
        self.prev_close = Some(close);

        if self.tr_len < self.length {
            self.tr_window[self.tr_len] = tr;
            self.tr_sum += tr;
            self.tr_len += 1;
        } else {
            let old = self.tr_window[self.tr_head];
            self.tr_window[self.tr_head] = tr;
            self.tr_sum += tr - old;
            self.tr_head += 1;
            if self.tr_head == self.length {
                self.tr_head = 0;
            }
        }

        while let Some(&(_, value)) = self.high_max.back() {
            if value <= high {
                self.high_max.pop_back();
            } else {
                break;
            }
        }
        self.high_max.push_back((current_index, high));

        while let Some(&(_, value)) = self.low_min.back() {
            if value >= low {
                self.low_min.pop_back();
            } else {
                break;
            }
        }
        self.low_min.push_back((current_index, low));

        let expire_before = current_index.saturating_add(1).saturating_sub(self.length);
        while let Some(&(idx, _)) = self.high_max.front() {
            if idx < expire_before {
                self.high_max.pop_front();
            } else {
                break;
            }
        }
        while let Some(&(idx, _)) = self.low_min.front() {
            if idx < expire_before {
                self.low_min.pop_front();
            } else {
                break;
            }
        }

        if self.tr_len < self.length {
            return None;
        }

        let denom = self.high_max.front().map(|x| x.1).unwrap_or(high)
            - self.low_min.front().map(|x| x.1).unwrap_or(low);
        let price_density = if denom > 0.0 {
            self.tr_sum / denom
        } else {
            f64::NAN
        };

        if self.pd_len == self.eval_period {
            let old = self.pd_window[self.pd_head];
            if old.is_finite() {
                sorted_remove_one(&mut self.pd_sorted, old);
            } else {
                self.invalid_pd = self.invalid_pd.saturating_sub(1);
            }
        } else {
            self.pd_len += 1;
        }

        self.pd_window[self.pd_head] = price_density;
        if price_density.is_finite() {
            let pos = self.pd_sorted.partition_point(|x| *x <= price_density);
            self.pd_sorted.insert(pos, price_density);
        } else {
            self.invalid_pd += 1;
        }

        self.pd_head += 1;
        if self.pd_head == self.eval_period {
            self.pd_head = 0;
        }

        let price_density_percent =
            if self.pd_len == self.eval_period && self.invalid_pd == 0 && price_density.is_finite()
            {
                let rank = self.pd_sorted.partition_point(|x| *x <= price_density);
                (rank as f64 / self.eval_period as f64) * 100.0
            } else {
                f64::NAN
            };

        Some((price_density, price_density_percent))
    }

    #[inline(always)]
    pub fn update_reset_on_nan(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64)> {
        if !valid_bar(high, low, close) {
            self.reset();
            return None;
        }
        self.update(high, low, close)
    }
}

#[inline(always)]
fn price_density_market_noise_prepare<'a>(
    input: &'a PriceDensityMarketNoiseInput,
) -> Result<(&'a [f64], &'a [f64], &'a [f64], usize, usize, usize), PriceDensityMarketNoiseError> {
    let (high, low, close) = input.as_refs();
    let data_len = close.len();
    if data_len == 0 {
        return Err(PriceDensityMarketNoiseError::EmptyInputData);
    }
    if high.len() != data_len || low.len() != data_len {
        return Err(PriceDensityMarketNoiseError::DataLengthMismatch);
    }

    let length = input.get_length();
    if length == 0 || length > data_len {
        return Err(PriceDensityMarketNoiseError::InvalidLength { length, data_len });
    }

    let eval_period = input.get_eval_period();
    if eval_period == 0 {
        return Err(PriceDensityMarketNoiseError::InvalidEvalPeriod {
            eval_period,
            data_len,
        });
    }

    let first =
        first_valid_bar(high, low, close).ok_or(PriceDensityMarketNoiseError::AllValuesNaN)?;
    let valid = count_valid_from(high, low, close, first);
    if valid < length {
        return Err(PriceDensityMarketNoiseError::NotEnoughValidData {
            needed: length,
            valid,
        });
    }

    Ok((high, low, close, length, eval_period, first))
}

#[inline(always)]
fn price_density_market_noise_compute_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
    eval_period: usize,
    _kernel: Kernel,
    out_price_density: &mut [f64],
    out_price_density_percent: &mut [f64],
) {
    let mut stream = PriceDensityMarketNoiseStream::from_parts(length, eval_period);
    for i in 0..close.len() {
        match stream.update_reset_on_nan(high[i], low[i], close[i]) {
            Some((pd, pd_percent)) => {
                out_price_density[i] = pd;
                out_price_density_percent[i] = pd_percent;
            }
            None => {
                out_price_density[i] = f64::NAN;
                out_price_density_percent[i] = f64::NAN;
            }
        }
    }
}

#[inline]
pub fn price_density_market_noise(
    input: &PriceDensityMarketNoiseInput,
) -> Result<PriceDensityMarketNoiseOutput, PriceDensityMarketNoiseError> {
    price_density_market_noise_with_kernel(input, Kernel::Auto)
}

pub fn price_density_market_noise_with_kernel(
    input: &PriceDensityMarketNoiseInput,
    kernel: Kernel,
) -> Result<PriceDensityMarketNoiseOutput, PriceDensityMarketNoiseError> {
    let (high, low, close, length, eval_period, first) = price_density_market_noise_prepare(input)?;
    let _ = first;
    let mut price_density = alloc_with_nan_prefix(close.len(), 0);
    let mut price_density_percent = alloc_with_nan_prefix(close.len(), 0);
    price_density_market_noise_compute_into(
        high,
        low,
        close,
        length,
        eval_period,
        kernel,
        &mut price_density,
        &mut price_density_percent,
    );
    Ok(PriceDensityMarketNoiseOutput {
        price_density,
        price_density_percent,
    })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn price_density_market_noise_into(
    input: &PriceDensityMarketNoiseInput,
    out_price_density: &mut [f64],
    out_price_density_percent: &mut [f64],
) -> Result<(), PriceDensityMarketNoiseError> {
    price_density_market_noise_into_slice(
        out_price_density,
        out_price_density_percent,
        input,
        Kernel::Auto,
    )
}

pub fn price_density_market_noise_into_slice(
    out_price_density: &mut [f64],
    out_price_density_percent: &mut [f64],
    input: &PriceDensityMarketNoiseInput,
    kernel: Kernel,
) -> Result<(), PriceDensityMarketNoiseError> {
    let (high, low, close, length, eval_period, _first) =
        price_density_market_noise_prepare(input)?;
    if out_price_density.len() != close.len() || out_price_density_percent.len() != close.len() {
        return Err(PriceDensityMarketNoiseError::OutputLengthMismatch {
            expected: close.len(),
            got: out_price_density.len().max(out_price_density_percent.len()),
        });
    }

    price_density_market_noise_compute_into(
        high,
        low,
        close,
        length,
        eval_period,
        kernel,
        out_price_density,
        out_price_density_percent,
    );
    Ok(())
}

#[derive(Clone, Debug)]
pub struct PriceDensityMarketNoiseBatchRange {
    pub length: (usize, usize, usize),
    pub eval_period: (usize, usize, usize),
}

impl Default for PriceDensityMarketNoiseBatchRange {
    fn default() -> Self {
        Self {
            length: (14, 14, 0),
            eval_period: (200, 200, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct PriceDensityMarketNoiseBatchBuilder {
    range: PriceDensityMarketNoiseBatchRange,
    kernel: Kernel,
}

impl PriceDensityMarketNoiseBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

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
    pub fn eval_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.eval_period = (start, end, step);
        self
    }

    #[inline]
    pub fn length_static(mut self, length: usize) -> Self {
        self.range.length = (length, length, 0);
        self
    }

    #[inline]
    pub fn eval_period_static(mut self, eval_period: usize) -> Self {
        self.range.eval_period = (eval_period, eval_period, 0);
        self
    }

    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<PriceDensityMarketNoiseBatchOutput, PriceDensityMarketNoiseError> {
        price_density_market_noise_batch_with_kernel(high, low, close, &self.range, self.kernel)
    }

    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<PriceDensityMarketNoiseBatchOutput, PriceDensityMarketNoiseError> {
        self.apply_slices(
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
        )
    }
}

#[derive(Clone, Debug)]
pub struct PriceDensityMarketNoiseBatchOutput {
    pub price_density: Vec<f64>,
    pub price_density_percent: Vec<f64>,
    pub combos: Vec<PriceDensityMarketNoiseParams>,
    pub rows: usize,
    pub cols: usize,
}

impl PriceDensityMarketNoiseBatchOutput {
    pub fn row_for_params(&self, params: &PriceDensityMarketNoiseParams) -> Option<usize> {
        let target_length = params.length.unwrap_or(14);
        let target_eval_period = params.eval_period.unwrap_or(200);
        self.combos.iter().position(|combo| {
            combo.length.unwrap_or(14) == target_length
                && combo.eval_period.unwrap_or(200) == target_eval_period
        })
    }
}

fn axis_usize(range: (usize, usize, usize)) -> Result<Vec<usize>, PriceDensityMarketNoiseError> {
    let (start, end, step) = range;
    if start == 0 || end == 0 {
        return Err(PriceDensityMarketNoiseError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
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
            if value < end.saturating_add(step) {
                break;
            }
            value = value.saturating_sub(step);
            if value == 0 {
                break;
            }
        }
    }

    if out.is_empty() {
        return Err(PriceDensityMarketNoiseError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

pub fn expand_grid_price_density_market_noise(
    sweep: &PriceDensityMarketNoiseBatchRange,
) -> Result<Vec<PriceDensityMarketNoiseParams>, PriceDensityMarketNoiseError> {
    let lengths = axis_usize(sweep.length)?;
    let eval_periods = axis_usize(sweep.eval_period)?;
    let mut out = Vec::with_capacity(lengths.len() * eval_periods.len());
    for length in lengths {
        for &eval_period in &eval_periods {
            out.push(PriceDensityMarketNoiseParams {
                length: Some(length),
                eval_period: Some(eval_period),
            });
        }
    }
    Ok(out)
}

pub fn price_density_market_noise_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &PriceDensityMarketNoiseBatchRange,
    kernel: Kernel,
) -> Result<PriceDensityMarketNoiseBatchOutput, PriceDensityMarketNoiseError> {
    let batch_kernel = match kernel {
        Kernel::Auto => Kernel::ScalarBatch,
        other if other.is_batch() => other,
        other => return Err(PriceDensityMarketNoiseError::InvalidKernelForBatch(other)),
    };
    price_density_market_noise_batch_impl(
        high,
        low,
        close,
        sweep,
        batch_kernel.to_non_batch(),
        true,
    )
}

pub fn price_density_market_noise_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &PriceDensityMarketNoiseBatchRange,
) -> Result<PriceDensityMarketNoiseBatchOutput, PriceDensityMarketNoiseError> {
    price_density_market_noise_batch_impl(high, low, close, sweep, Kernel::Scalar, false)
}

pub fn price_density_market_noise_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &PriceDensityMarketNoiseBatchRange,
) -> Result<PriceDensityMarketNoiseBatchOutput, PriceDensityMarketNoiseError> {
    price_density_market_noise_batch_impl(high, low, close, sweep, Kernel::Scalar, true)
}

fn price_density_market_noise_batch_impl(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &PriceDensityMarketNoiseBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<PriceDensityMarketNoiseBatchOutput, PriceDensityMarketNoiseError> {
    let combos = expand_grid_price_density_market_noise(sweep)?;
    let rows = combos.len();
    let cols = close.len();
    if cols == 0 {
        return Err(PriceDensityMarketNoiseError::EmptyInputData);
    }
    if high.len() != cols || low.len() != cols {
        return Err(PriceDensityMarketNoiseError::DataLengthMismatch);
    }

    for params in &combos {
        let input = PriceDensityMarketNoiseInput::from_slices(high, low, close, params.clone());
        price_density_market_noise_prepare(&input)?;
    }

    let first = first_valid_bar(high, low, close).unwrap_or(cols);
    let price_density_warmups: Vec<usize> = combos
        .iter()
        .map(|params| price_density_warmup(params.length.unwrap_or(14), first).min(cols))
        .collect();
    let price_density_percent_warmups: Vec<usize> = combos
        .iter()
        .map(|params| {
            price_density_percent_warmup(
                params.length.unwrap_or(14),
                params.eval_period.unwrap_or(200),
                first,
            )
            .min(cols)
        })
        .collect();

    let mut price_density_matrix = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut price_density_matrix, cols, &price_density_warmups);
    let mut price_density_percent_matrix = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(
        &mut price_density_percent_matrix,
        cols,
        &price_density_percent_warmups,
    );

    let mut price_density_guard = ManuallyDrop::new(price_density_matrix);
    let mut price_density_percent_guard = ManuallyDrop::new(price_density_percent_matrix);
    let price_density_mu: &mut [MaybeUninit<f64>] = unsafe {
        std::slice::from_raw_parts_mut(price_density_guard.as_mut_ptr(), price_density_guard.len())
    };
    let price_density_percent_mu: &mut [MaybeUninit<f64>] = unsafe {
        std::slice::from_raw_parts_mut(
            price_density_percent_guard.as_mut_ptr(),
            price_density_percent_guard.len(),
        )
    };

    let do_row = |row: usize,
                  row_price_density_mu: &mut [MaybeUninit<f64>],
                  row_price_density_percent_mu: &mut [MaybeUninit<f64>]| {
        let params = &combos[row];
        let dst_price_density = unsafe {
            std::slice::from_raw_parts_mut(
                row_price_density_mu.as_mut_ptr() as *mut f64,
                row_price_density_mu.len(),
            )
        };
        let dst_price_density_percent = unsafe {
            std::slice::from_raw_parts_mut(
                row_price_density_percent_mu.as_mut_ptr() as *mut f64,
                row_price_density_percent_mu.len(),
            )
        };
        price_density_market_noise_compute_into(
            high,
            low,
            close,
            params.length.unwrap_or(14),
            params.eval_period.unwrap_or(200),
            kernel,
            dst_price_density,
            dst_price_density_percent,
        );
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        price_density_mu
            .par_chunks_mut(cols)
            .zip(price_density_percent_mu.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, (row_price_density, row_price_density_percent))| {
                do_row(row, row_price_density, row_price_density_percent)
            });
        #[cfg(target_arch = "wasm32")]
        for (row, (row_price_density, row_price_density_percent)) in price_density_mu
            .chunks_mut(cols)
            .zip(price_density_percent_mu.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, row_price_density, row_price_density_percent);
        }
    } else {
        for (row, (row_price_density, row_price_density_percent)) in price_density_mu
            .chunks_mut(cols)
            .zip(price_density_percent_mu.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, row_price_density, row_price_density_percent);
        }
    }

    let price_density = unsafe {
        Vec::from_raw_parts(
            price_density_guard.as_mut_ptr() as *mut f64,
            price_density_guard.len(),
            price_density_guard.capacity(),
        )
    };
    let price_density_percent = unsafe {
        Vec::from_raw_parts(
            price_density_percent_guard.as_mut_ptr() as *mut f64,
            price_density_percent_guard.len(),
            price_density_percent_guard.capacity(),
        )
    };

    Ok(PriceDensityMarketNoiseBatchOutput {
        price_density,
        price_density_percent,
        combos,
        rows,
        cols,
    })
}

fn price_density_market_noise_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &PriceDensityMarketNoiseBatchRange,
    kernel: Kernel,
    parallel: bool,
    out_price_density: &mut [f64],
    out_price_density_percent: &mut [f64],
) -> Result<(), PriceDensityMarketNoiseError> {
    let combos = expand_grid_price_density_market_noise(sweep)?;
    let rows = combos.len();
    let cols = close.len();
    if out_price_density.len() != rows * cols || out_price_density_percent.len() != rows * cols {
        return Err(PriceDensityMarketNoiseError::OutputLengthMismatch {
            expected: rows * cols,
            got: out_price_density.len().max(out_price_density_percent.len()),
        });
    }

    for params in &combos {
        let input = PriceDensityMarketNoiseInput::from_slices(high, low, close, params.clone());
        price_density_market_noise_prepare(&input)?;
    }

    for row in 0..rows {
        let row_price_density = &mut out_price_density[row * cols..(row + 1) * cols];
        let row_price_density_percent =
            &mut out_price_density_percent[row * cols..(row + 1) * cols];
        row_price_density.fill(f64::NAN);
        row_price_density_percent.fill(f64::NAN);
    }

    let do_row =
        |row: usize, row_price_density: &mut [f64], row_price_density_percent: &mut [f64]| {
            let params = &combos[row];
            price_density_market_noise_compute_into(
                high,
                low,
                close,
                params.length.unwrap_or(14),
                params.eval_period.unwrap_or(200),
                kernel,
                row_price_density,
                row_price_density_percent,
            );
        };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out_price_density
            .par_chunks_mut(cols)
            .zip(out_price_density_percent.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, (row_price_density, row_price_density_percent))| {
                do_row(row, row_price_density, row_price_density_percent)
            });
        #[cfg(target_arch = "wasm32")]
        for (row, (row_price_density, row_price_density_percent)) in out_price_density
            .chunks_mut(cols)
            .zip(out_price_density_percent.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, row_price_density, row_price_density_percent);
        }
    } else {
        for (row, (row_price_density, row_price_density_percent)) in out_price_density
            .chunks_mut(cols)
            .zip(out_price_density_percent.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, row_price_density, row_price_density_percent);
        }
    }

    Ok(())
}

#[cfg(feature = "python")]
#[pyfunction(name = "price_density_market_noise")]
#[pyo3(signature = (high, low, close, length=14, eval_period=200, kernel=None))]
pub fn price_density_market_noise_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    length: usize,
    eval_period: usize,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = PriceDensityMarketNoiseInput::from_slices(
        high,
        low,
        close,
        PriceDensityMarketNoiseParams {
            length: Some(length),
            eval_period: Some(eval_period),
        },
    );
    let output = py
        .allow_threads(|| price_density_market_noise_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        output.price_density.into_pyarray(py),
        output.price_density_percent.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyclass(name = "PriceDensityMarketNoiseStream")]
pub struct PriceDensityMarketNoiseStreamPy {
    stream: PriceDensityMarketNoiseStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl PriceDensityMarketNoiseStreamPy {
    #[new]
    #[pyo3(signature = (length=14, eval_period=200))]
    fn new(length: usize, eval_period: usize) -> PyResult<Self> {
        let stream = PriceDensityMarketNoiseStream::try_new(PriceDensityMarketNoiseParams {
            length: Some(length),
            eval_period: Some(eval_period),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64)> {
        self.stream.update_reset_on_nan(high, low, close)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "price_density_market_noise_batch")]
#[pyo3(signature = (high, low, close, length_range, eval_period_range, kernel=None))]
pub fn price_density_market_noise_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    eval_period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let sweep = PriceDensityMarketNoiseBatchRange {
        length: length_range,
        eval_period: eval_period_range,
    };
    let combos = expand_grid_price_density_market_noise(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;
    let price_density_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let price_density_percent_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_price_density = unsafe { price_density_arr.as_slice_mut()? };
    let out_price_density_percent = unsafe { price_density_percent_arr.as_slice_mut()? };
    let kernel = validate_kernel(kernel, true)?;

    py.allow_threads(|| {
        let batch_kernel = match kernel {
            Kernel::Auto => detect_best_batch_kernel(),
            other => other,
        };
        price_density_market_noise_batch_inner_into(
            high,
            low,
            close,
            &sweep,
            batch_kernel.to_non_batch(),
            true,
            out_price_density,
            out_price_density_percent,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("price_density", price_density_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "price_density_percent",
        price_density_percent_arr.reshape((rows, cols))?,
    )?;
    dict.set_item(
        "lengths",
        combos
            .iter()
            .map(|p| p.length.unwrap_or(14) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "eval_periods",
        combos
            .iter()
            .map(|p| p.eval_period.unwrap_or(200) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_price_density_market_noise_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(price_density_market_noise_py, m)?)?;
    m.add_function(wrap_pyfunction!(price_density_market_noise_batch_py, m)?)?;
    m.add_class::<PriceDensityMarketNoiseStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PriceDensityMarketNoiseJsOutput {
    price_density: Vec<f64>,
    price_density_percent: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PriceDensityMarketNoiseBatchConfig {
    length_range: Vec<usize>,
    eval_period_range: Vec<usize>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PriceDensityMarketNoiseBatchJsOutput {
    price_density: Vec<f64>,
    price_density_percent: Vec<f64>,
    rows: usize,
    cols: usize,
    combos: Vec<PriceDensityMarketNoiseParams>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "price_density_market_noise_js")]
pub fn price_density_market_noise_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
    eval_period: usize,
) -> Result<JsValue, JsValue> {
    let input = PriceDensityMarketNoiseInput::from_slices(
        high,
        low,
        close,
        PriceDensityMarketNoiseParams {
            length: Some(length),
            eval_period: Some(eval_period),
        },
    );
    let output =
        price_density_market_noise(&input).map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&PriceDensityMarketNoiseJsOutput {
        price_density: output.price_density,
        price_density_percent: output.price_density_percent,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "price_density_market_noise_batch_js")]
pub fn price_density_market_noise_batch_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: PriceDensityMarketNoiseBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.length_range.len() != 3 || config.eval_period_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: ranges must have exactly 3 elements [start, end, step]",
        ));
    }
    let sweep = PriceDensityMarketNoiseBatchRange {
        length: (
            config.length_range[0],
            config.length_range[1],
            config.length_range[2],
        ),
        eval_period: (
            config.eval_period_range[0],
            config.eval_period_range[1],
            config.eval_period_range[2],
        ),
    };
    let batch = price_density_market_noise_batch_slice(high, low, close, &sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&PriceDensityMarketNoiseBatchJsOutput {
        price_density: batch.price_density,
        price_density_percent: batch.price_density_percent,
        rows: batch.rows,
        cols: batch.cols,
        combos: batch.combos,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn price_density_market_noise_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len * 2);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn price_density_market_noise_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, 0, len * 2);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn price_density_market_noise_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length: usize,
    eval_period: usize,
) -> Result<(), JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to price_density_market_noise_into",
        ));
    }
    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len * 2);
        let (out_price_density, out_price_density_percent) = out.split_at_mut(len);
        let input = PriceDensityMarketNoiseInput::from_slices(
            high,
            low,
            close,
            PriceDensityMarketNoiseParams {
                length: Some(length),
                eval_period: Some(eval_period),
            },
        );
        price_density_market_noise_into_slice(
            out_price_density,
            out_price_density_percent,
            &input,
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "price_density_market_noise_into_host")]
pub fn price_density_market_noise_into_host(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    out_ptr: *mut f64,
    length: usize,
    eval_period: usize,
) -> Result<(), JsValue> {
    if out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to price_density_market_noise_into_host",
        ));
    }
    unsafe {
        let out = std::slice::from_raw_parts_mut(out_ptr, close.len() * 2);
        let (out_price_density, out_price_density_percent) = out.split_at_mut(close.len());
        let input = PriceDensityMarketNoiseInput::from_slices(
            high,
            low,
            close,
            PriceDensityMarketNoiseParams {
                length: Some(length),
                eval_period: Some(eval_period),
            },
        );
        price_density_market_noise_into_slice(
            out_price_density,
            out_price_density_percent,
            &input,
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn price_density_market_noise_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length_start: usize,
    length_end: usize,
    length_step: usize,
    eval_period_start: usize,
    eval_period_end: usize,
    eval_period_step: usize,
) -> Result<usize, JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to price_density_market_noise_batch_into",
        ));
    }
    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let sweep = PriceDensityMarketNoiseBatchRange {
            length: (length_start, length_end, length_step),
            eval_period: (eval_period_start, eval_period_end, eval_period_step),
        };
        let combos = expand_grid_price_density_market_noise(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let out = std::slice::from_raw_parts_mut(out_ptr, rows * len * 2);
        let (out_price_density, out_price_density_percent) = out.split_at_mut(rows * len);
        price_density_market_noise_batch_inner_into(
            high,
            low,
            close,
            &sweep,
            Kernel::Scalar,
            false,
            out_price_density,
            out_price_density_percent,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn price_density_market_noise_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
    eval_period: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = price_density_market_noise_js(high, low, close, length, eval_period)?;
    crate::write_wasm_object_f64_outputs("price_density_market_noise_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn price_density_market_noise_batch_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = price_density_market_noise_batch_js(high, low, close, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "price_density_market_noise_batch_output_into_js",
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

    fn assert_close(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            let x = a[i];
            let y = b[i];
            if x.is_nan() || y.is_nan() {
                assert!(x.is_nan() && y.is_nan(), "nan mismatch at {i}: {x} vs {y}");
            } else {
                assert!((x - y).abs() <= 1e-10, "mismatch at {i}: {x} vs {y}");
            }
        }
    }

    fn sample_hlc(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut high = Vec::with_capacity(len);
        let mut low = Vec::with_capacity(len);
        let mut close = Vec::with_capacity(len);
        for i in 0..len {
            let base = 100.0 + i as f64 * 0.07 + (i as f64 * 0.11).sin() * 1.8;
            let cl = base + (i as f64 * 0.19).cos() * 0.4;
            let hi = cl + 0.9 + (i as f64 * 0.13).sin().abs();
            let lo = cl - 0.8 - (i as f64 * 0.17).cos().abs();
            high.push(hi);
            low.push(lo);
            close.push(cl);
        }
        (high, low, close)
    }

    fn sample_ohlc(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let (high, low, close) = sample_hlc(len);
        let mut open = Vec::with_capacity(len);
        for i in 0..len {
            open.push((high[i] + low[i] + close[i]) / 3.0);
        }
        (open, high, low, close)
    }

    fn naive_expected(
        high: &[f64],
        low: &[f64],
        close: &[f64],
        length: usize,
        eval_period: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut pd = vec![f64::NAN; n];
        let mut pd_percent = vec![f64::NAN; n];
        let mut tr = vec![f64::NAN; n];

        for i in 0..n {
            tr[i] = if i == 0 {
                high[i] - low[i]
            } else {
                (high[i] - low[i])
                    .max((high[i] - close[i - 1]).abs())
                    .max((low[i] - close[i - 1]).abs())
            };
        }

        for i in (length - 1)..n {
            let start = i + 1 - length;
            let tr_sum: f64 = tr[start..=i].iter().sum();
            let highest = high[start..=i]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let lowest = low[start..=i].iter().copied().fold(f64::INFINITY, f64::min);
            let denom = highest - lowest;
            pd[i] = if denom > 0.0 {
                tr_sum / denom
            } else {
                f64::NAN
            };
        }

        for i in 0..n {
            if i + 1 < eval_period || !pd[i].is_finite() {
                continue;
            }
            let start = i + 1 - eval_period;
            let window = &pd[start..=i];
            if window.iter().all(|x| x.is_finite()) {
                let rank = window.iter().filter(|&&x| x <= pd[i]).count();
                pd_percent[i] = rank as f64 / eval_period as f64 * 100.0;
            }
        }

        (pd, pd_percent)
    }

    #[test]
    fn price_density_market_noise_matches_naive() {
        let (high, low, close) = sample_hlc(256);
        let input = PriceDensityMarketNoiseInput::from_slices(
            &high,
            &low,
            &close,
            PriceDensityMarketNoiseParams {
                length: Some(14),
                eval_period: Some(40),
            },
        );
        let out = price_density_market_noise(&input).expect("indicator");
        let (expected_pd, expected_percent) = naive_expected(&high, &low, &close, 14, 40);
        assert_close(&out.price_density, &expected_pd);
        assert_close(&out.price_density_percent, &expected_percent);
    }

    #[test]
    fn price_density_market_noise_into_matches_api() {
        let (high, low, close) = sample_hlc(192);
        let input = PriceDensityMarketNoiseInput::from_slices(
            &high,
            &low,
            &close,
            PriceDensityMarketNoiseParams {
                length: Some(12),
                eval_period: Some(30),
            },
        );
        let baseline = price_density_market_noise(&input).expect("baseline");
        let mut pd = vec![0.0; close.len()];
        let mut pd_percent = vec![0.0; close.len()];
        price_density_market_noise_into(&input, &mut pd, &mut pd_percent).expect("into");
        assert_close(&baseline.price_density, &pd);
        assert_close(&baseline.price_density_percent, &pd_percent);
    }

    #[test]
    fn price_density_market_noise_stream_matches_batch() {
        let (high, low, close) = sample_hlc(192);
        let batch = price_density_market_noise(&PriceDensityMarketNoiseInput::from_slices(
            &high,
            &low,
            &close,
            PriceDensityMarketNoiseParams {
                length: Some(10),
                eval_period: Some(25),
            },
        ))
        .expect("batch");
        let mut stream = PriceDensityMarketNoiseStream::try_new(PriceDensityMarketNoiseParams {
            length: Some(10),
            eval_period: Some(25),
        })
        .expect("stream");
        let mut pd = vec![f64::NAN; close.len()];
        let mut pd_percent = vec![f64::NAN; close.len()];
        for i in 0..close.len() {
            if let Some((a, b)) = stream.update_reset_on_nan(high[i], low[i], close[i]) {
                pd[i] = a;
                pd_percent[i] = b;
            }
        }
        assert_close(&batch.price_density, &pd);
        assert_close(&batch.price_density_percent, &pd_percent);
    }

    #[test]
    fn price_density_market_noise_batch_single_param_matches_single() {
        let (high, low, close) = sample_hlc(160);
        let batch = price_density_market_noise_batch_with_kernel(
            &high,
            &low,
            &close,
            &PriceDensityMarketNoiseBatchRange {
                length: (12, 12, 0),
                eval_period: (24, 24, 0),
            },
            Kernel::ScalarBatch,
        )
        .expect("batch");
        let input = PriceDensityMarketNoiseInput::from_slices(
            &high,
            &low,
            &close,
            PriceDensityMarketNoiseParams {
                length: Some(12),
                eval_period: Some(24),
            },
        );
        let direct =
            price_density_market_noise_with_kernel(&input, Kernel::Scalar).expect("direct");
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, close.len());
        assert_close(&batch.price_density[..close.len()], &direct.price_density);
        assert_close(
            &batch.price_density_percent[..close.len()],
            &direct.price_density_percent,
        );
    }

    #[test]
    fn price_density_market_noise_rejects_invalid_eval_period() {
        let (high, low, close) = sample_hlc(32);
        let input = PriceDensityMarketNoiseInput::from_slices(
            &high,
            &low,
            &close,
            PriceDensityMarketNoiseParams {
                length: Some(10),
                eval_period: Some(0),
            },
        );
        let err = price_density_market_noise(&input).unwrap_err();
        assert!(matches!(
            err,
            PriceDensityMarketNoiseError::InvalidEvalPeriod { .. }
        ));
    }

    #[test]
    fn price_density_market_noise_dispatch_matches_direct() {
        let (open, high, low, close) = sample_ohlc(160);
        let combo = [
            ParamKV {
                key: "length",
                value: ParamValue::Int(12),
            },
            ParamKV {
                key: "eval_period",
                value: ParamValue::Int(24),
            },
        ];
        let combos = [IndicatorParamSet { params: &combo }];

        let req = IndicatorBatchRequest {
            indicator_id: "price_density_market_noise",
            output_id: Some("price_density"),
            data: IndicatorDataRef::Ohlc {
                open: &open,
                high: &high,
                low: &low,
                close: &close,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).expect("dispatch");

        let input = PriceDensityMarketNoiseInput::from_slices(
            &high,
            &low,
            &close,
            PriceDensityMarketNoiseParams {
                length: Some(12),
                eval_period: Some(24),
            },
        );
        let direct = price_density_market_noise(&input).expect("direct");
        assert_eq!(out.rows, 1);
        assert_eq!(out.cols, close.len());
        let values = out.values_f64.expect("values");
        assert_close(&values, &direct.price_density);
    }
}
