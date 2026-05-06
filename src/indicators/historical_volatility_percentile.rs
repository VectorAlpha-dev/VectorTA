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
use std::error::Error;
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

impl<'a> AsRef<[f64]> for HistoricalVolatilityPercentileInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            HistoricalVolatilityPercentileData::Slice(slice) => slice,
            HistoricalVolatilityPercentileData::Candles { candles, source } => {
                source_type(candles, source)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum HistoricalVolatilityPercentileData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct HistoricalVolatilityPercentileOutput {
    pub hvp: Vec<f64>,
    pub hvp_sma: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistoricalVolatilityPercentileOutputField {
    Hvp,
    HvpSma,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct HistoricalVolatilityPercentileParams {
    pub length: Option<usize>,
    pub annual_length: Option<usize>,
}

impl Default for HistoricalVolatilityPercentileParams {
    fn default() -> Self {
        Self {
            length: Some(21),
            annual_length: Some(252),
        }
    }
}

#[derive(Debug, Clone)]
pub struct HistoricalVolatilityPercentileInput<'a> {
    pub data: HistoricalVolatilityPercentileData<'a>,
    pub params: HistoricalVolatilityPercentileParams,
}

impl<'a> HistoricalVolatilityPercentileInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: HistoricalVolatilityPercentileParams,
    ) -> Self {
        Self {
            data: HistoricalVolatilityPercentileData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: HistoricalVolatilityPercentileParams) -> Self {
        Self {
            data: HistoricalVolatilityPercentileData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(
            candles,
            "close",
            HistoricalVolatilityPercentileParams::default(),
        )
    }

    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(21)
    }

    #[inline]
    pub fn get_annual_length(&self) -> usize {
        self.params.annual_length.unwrap_or(252)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct HistoricalVolatilityPercentileBuilder {
    length: Option<usize>,
    annual_length: Option<usize>,
    kernel: Kernel,
}

impl Default for HistoricalVolatilityPercentileBuilder {
    fn default() -> Self {
        Self {
            length: None,
            annual_length: None,
            kernel: Kernel::Auto,
        }
    }
}

impl HistoricalVolatilityPercentileBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn length(mut self, n: usize) -> Self {
        self.length = Some(n);
        self
    }

    #[inline(always)]
    pub fn annual_length(mut self, n: usize) -> Self {
        self.annual_length = Some(n);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<HistoricalVolatilityPercentileOutput, HistoricalVolatilityPercentileError> {
        let params = HistoricalVolatilityPercentileParams {
            length: self.length,
            annual_length: self.annual_length,
        };
        let input = HistoricalVolatilityPercentileInput::from_candles(candles, "close", params);
        historical_volatility_percentile_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<HistoricalVolatilityPercentileOutput, HistoricalVolatilityPercentileError> {
        let params = HistoricalVolatilityPercentileParams {
            length: self.length,
            annual_length: self.annual_length,
        };
        let input = HistoricalVolatilityPercentileInput::from_slice(data, params);
        historical_volatility_percentile_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<HistoricalVolatilityPercentileStream, HistoricalVolatilityPercentileError> {
        let params = HistoricalVolatilityPercentileParams {
            length: self.length,
            annual_length: self.annual_length,
        };
        HistoricalVolatilityPercentileStream::try_new(params)
    }
}

#[derive(Debug, Error)]
pub enum HistoricalVolatilityPercentileError {
    #[error("historical_volatility_percentile: Input data slice is empty.")]
    EmptyInputData,
    #[error("historical_volatility_percentile: All source values are invalid.")]
    AllValuesNaN,
    #[error("historical_volatility_percentile: Invalid length: length = {length}, data length = {data_len}")]
    InvalidLength { length: usize, data_len: usize },
    #[error(
        "historical_volatility_percentile: Invalid annual length: annual_length = {annual_length}"
    )]
    InvalidAnnualLength { annual_length: usize },
    #[error("historical_volatility_percentile: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("historical_volatility_percentile: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error(
        "historical_volatility_percentile: Invalid range: start={start}, end={end}, step={step}"
    )]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("historical_volatility_percentile: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("historical_volatility_percentile: Invalid input: {0}")]
    InvalidInput(&'static str),
}

#[inline(always)]
fn is_valid_source(v: f64) -> bool {
    v.is_finite() && v > 0.0
}

#[inline(always)]
fn first_valid_source(data: &[f64]) -> Option<usize> {
    data.iter().position(|&v| is_valid_source(v))
}

#[inline(always)]
fn hvp_warmup(length: usize, annual_length: usize, first: usize) -> usize {
    first + length + annual_length - 2
}

#[inline(always)]
fn hvp_sma_warmup(length: usize, annual_length: usize, first: usize) -> usize {
    first + annual_length + (2 * length) - 3
}

#[inline(always)]
fn build_returns(data: &[f64]) -> Vec<f64> {
    let mut out = vec![f64::NAN; data.len()];
    let mut prev_valid: Option<f64> = None;
    for (i, &v) in data.iter().enumerate() {
        if is_valid_source(v) {
            out[i] = match prev_valid {
                Some(prev) => (v / prev).ln(),
                None => 0.0,
            };
            prev_valid = Some(v);
        } else {
            prev_valid = None;
        }
    }
    out
}

#[inline(always)]
fn hv_from_sums(sum: f64, sum_sq: f64, length: usize, annual_length: usize) -> f64 {
    let n = length as f64;
    let mean = sum / n;
    let centered = (sum_sq - (mean * mean * n)).max(0.0);
    let sample_var = centered / ((length - 1) as f64);
    sample_var.sqrt() * (annual_length as f64).sqrt()
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

#[inline(always)]
fn hvp_compute_into(
    returns: &[f64],
    length: usize,
    annual_length: usize,
    first: usize,
    out_hvp: &mut [f64],
    out_hvp_sma: &mut [f64],
) {
    let n = returns.len();
    let hv_start = first + length - 1;
    if hv_start >= n {
        return;
    }

    let mut ret_sum = 0.0f64;
    let mut ret_sum_sq = 0.0f64;
    let mut invalid_ret = 0usize;
    for &r in &returns[first..=hv_start] {
        if r.is_finite() {
            ret_sum += r;
            ret_sum_sq += r * r;
        } else {
            invalid_ret += 1;
        }
    }

    let mut hv_window = vec![f64::NAN; annual_length];
    let mut hv_sorted = Vec::with_capacity(annual_length);
    let mut hv_head = 0usize;
    let mut hv_len = 0usize;
    let mut invalid_hv = 0usize;

    let mut sma_window = vec![f64::NAN; length];
    let mut sma_head = 0usize;
    let mut sma_len = 0usize;
    let mut invalid_sma = 0usize;
    let mut sma_sum = 0.0f64;

    for i in hv_start..n {
        if i > hv_start {
            let out_idx = i - length;
            let r_old = returns[out_idx];
            if r_old.is_finite() {
                ret_sum -= r_old;
                ret_sum_sq -= r_old * r_old;
            } else {
                invalid_ret -= 1;
            }

            let r_new = returns[i];
            if r_new.is_finite() {
                ret_sum += r_new;
                ret_sum_sq += r_new * r_new;
            } else {
                invalid_ret += 1;
            }
        }

        let hv = if invalid_ret == 0 {
            hv_from_sums(ret_sum, ret_sum_sq, length, annual_length)
        } else {
            f64::NAN
        };

        if hv_len == annual_length {
            let old = hv_window[hv_head];
            if old.is_finite() {
                sorted_remove_one(&mut hv_sorted, old);
            } else {
                invalid_hv -= 1;
            }
        }

        hv_window[hv_head] = hv;
        if hv.is_finite() {
            let pos = hv_sorted.partition_point(|x| *x < hv);
            hv_sorted.insert(pos, hv);
        } else {
            invalid_hv += 1;
        }

        hv_head += 1;
        if hv_head == annual_length {
            hv_head = 0;
        }
        if hv_len < annual_length {
            hv_len += 1;
        }

        let hvp = if hv_len == annual_length && invalid_hv == 0 && hv.is_finite() {
            let rank = hv_sorted.partition_point(|x| *x < hv);
            let value = (rank as f64 / annual_length as f64) * 100.0;
            out_hvp[i] = value;
            value
        } else {
            out_hvp[i] = f64::NAN;
            f64::NAN
        };

        if sma_len == length {
            let old = sma_window[sma_head];
            if old.is_finite() {
                sma_sum -= old;
            } else {
                invalid_sma -= 1;
            }
        }

        sma_window[sma_head] = hvp;
        if hvp.is_finite() {
            sma_sum += hvp;
        } else {
            invalid_sma += 1;
        }

        sma_head += 1;
        if sma_head == length {
            sma_head = 0;
        }
        if sma_len < length {
            sma_len += 1;
        }

        if sma_len == length && invalid_sma == 0 {
            out_hvp_sma[i] = sma_sum / length as f64;
        } else {
            out_hvp_sma[i] = f64::NAN;
        }
    }
}

#[inline(always)]
fn hvp_prepare<'a>(
    input: &'a HistoricalVolatilityPercentileInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, usize, usize, Kernel), HistoricalVolatilityPercentileError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(HistoricalVolatilityPercentileError::EmptyInputData);
    }

    let first =
        first_valid_source(data).ok_or(HistoricalVolatilityPercentileError::AllValuesNaN)?;
    let length = input.get_length();
    let annual_length = input.get_annual_length();

    if length < 2 || length > len {
        return Err(HistoricalVolatilityPercentileError::InvalidLength {
            length,
            data_len: len,
        });
    }
    if annual_length == 0 {
        return Err(HistoricalVolatilityPercentileError::InvalidAnnualLength { annual_length });
    }

    let needed = length + annual_length - 1;
    let valid = len - first;
    if valid < needed {
        return Err(HistoricalVolatilityPercentileError::NotEnoughValidData { needed, valid });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other.to_non_batch(),
    };
    Ok((data, length, annual_length, first, chosen))
}

#[inline]
pub fn historical_volatility_percentile(
    input: &HistoricalVolatilityPercentileInput,
) -> Result<HistoricalVolatilityPercentileOutput, HistoricalVolatilityPercentileError> {
    historical_volatility_percentile_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn historical_volatility_percentile_with_kernel(
    input: &HistoricalVolatilityPercentileInput,
    kernel: Kernel,
) -> Result<HistoricalVolatilityPercentileOutput, HistoricalVolatilityPercentileError> {
    let (data, length, annual_length, first, _chosen) = hvp_prepare(input, kernel)?;
    let returns = build_returns(data);
    let mut hvp = alloc_with_nan_prefix(data.len(), hvp_warmup(length, annual_length, first));
    let mut hvp_sma =
        alloc_with_nan_prefix(data.len(), hvp_sma_warmup(length, annual_length, first));
    hvp_compute_into(
        &returns,
        length,
        annual_length,
        first,
        &mut hvp,
        &mut hvp_sma,
    );
    Ok(HistoricalVolatilityPercentileOutput { hvp, hvp_sma })
}

#[inline]
pub fn historical_volatility_percentile_into_slice(
    dst_hvp: &mut [f64],
    dst_hvp_sma: &mut [f64],
    input: &HistoricalVolatilityPercentileInput,
    kernel: Kernel,
) -> Result<(), HistoricalVolatilityPercentileError> {
    let (data, length, annual_length, first, _chosen) = hvp_prepare(input, kernel)?;
    if dst_hvp.len() != data.len() || dst_hvp_sma.len() != data.len() {
        return Err(HistoricalVolatilityPercentileError::OutputLengthMismatch {
            expected: data.len(),
            got: dst_hvp.len().max(dst_hvp_sma.len()),
        });
    }
    dst_hvp.fill(f64::NAN);
    dst_hvp_sma.fill(f64::NAN);
    let returns = build_returns(data);
    hvp_compute_into(&returns, length, annual_length, first, dst_hvp, dst_hvp_sma);
    Ok(())
}

#[inline]
pub fn historical_volatility_percentile_output_into_slice(
    out: &mut [f64],
    input: &HistoricalVolatilityPercentileInput,
    kernel: Kernel,
    field: HistoricalVolatilityPercentileOutputField,
) -> Result<(), HistoricalVolatilityPercentileError> {
    let (data, length, annual_length, first, _chosen) = hvp_prepare(input, kernel)?;
    if out.len() != data.len() {
        return Err(HistoricalVolatilityPercentileError::OutputLengthMismatch {
            expected: data.len(),
            got: out.len(),
        });
    }
    out.fill(f64::NAN);
    let returns = build_returns(data);
    match field {
        HistoricalVolatilityPercentileOutputField::Hvp => {
            let mut hvp_sma = alloc_with_nan_prefix(
                data.len(),
                hvp_sma_warmup(length, annual_length, first).min(data.len()),
            );
            hvp_compute_into(&returns, length, annual_length, first, out, &mut hvp_sma);
        }
        HistoricalVolatilityPercentileOutputField::HvpSma => {
            let mut hvp =
                alloc_with_nan_prefix(data.len(), hvp_warmup(length, annual_length, first));
            hvp_compute_into(&returns, length, annual_length, first, &mut hvp, out);
        }
    }
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn historical_volatility_percentile_into(
    input: &HistoricalVolatilityPercentileInput,
    out_hvp: &mut [f64],
    out_hvp_sma: &mut [f64],
) -> Result<(), HistoricalVolatilityPercentileError> {
    historical_volatility_percentile_into_slice(out_hvp, out_hvp_sma, input, Kernel::Auto)
}

#[derive(Debug, Clone)]
pub struct HistoricalVolatilityPercentileStream {
    length: usize,
    annual_length: usize,
    prev_valid: Option<f64>,
    ret_window: Vec<f64>,
    ret_head: usize,
    ret_len: usize,
    ret_sum: f64,
    ret_sum_sq: f64,
    invalid_ret: usize,
    hv_window: Vec<f64>,
    hv_sorted: Vec<f64>,
    hv_head: usize,
    hv_len: usize,
    invalid_hv: usize,
    sma_window: Vec<f64>,
    sma_head: usize,
    sma_len: usize,
    sma_sum: f64,
    invalid_sma: usize,
}

impl HistoricalVolatilityPercentileStream {
    #[inline]
    pub fn try_new(
        params: HistoricalVolatilityPercentileParams,
    ) -> Result<Self, HistoricalVolatilityPercentileError> {
        let length = params.length.unwrap_or(21);
        let annual_length = params.annual_length.unwrap_or(252);
        if length < 2 {
            return Err(HistoricalVolatilityPercentileError::InvalidLength {
                length,
                data_len: 0,
            });
        }
        if annual_length == 0 {
            return Err(HistoricalVolatilityPercentileError::InvalidAnnualLength { annual_length });
        }

        Ok(Self {
            length,
            annual_length,
            prev_valid: None,
            ret_window: vec![f64::NAN; length],
            ret_head: 0,
            ret_len: 0,
            ret_sum: 0.0,
            ret_sum_sq: 0.0,
            invalid_ret: 0,
            hv_window: vec![f64::NAN; annual_length],
            hv_sorted: Vec::with_capacity(annual_length),
            hv_head: 0,
            hv_len: 0,
            invalid_hv: 0,
            sma_window: vec![f64::NAN; length],
            sma_head: 0,
            sma_len: 0,
            sma_sum: 0.0,
            invalid_sma: 0,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        let ret = if is_valid_source(value) {
            let r = match self.prev_valid {
                Some(prev) => (value / prev).ln(),
                None => 0.0,
            };
            self.prev_valid = Some(value);
            r
        } else {
            self.prev_valid = None;
            f64::NAN
        };

        if self.ret_len == self.length {
            let old = self.ret_window[self.ret_head];
            if old.is_finite() {
                self.ret_sum -= old;
                self.ret_sum_sq -= old * old;
            } else {
                self.invalid_ret -= 1;
            }
        }

        self.ret_window[self.ret_head] = ret;
        if ret.is_finite() {
            self.ret_sum += ret;
            self.ret_sum_sq += ret * ret;
        } else {
            self.invalid_ret += 1;
        }

        self.ret_head += 1;
        if self.ret_head == self.length {
            self.ret_head = 0;
        }
        if self.ret_len < self.length {
            self.ret_len += 1;
        }

        if self.ret_len < self.length {
            return None;
        }

        let hv = if self.invalid_ret == 0 {
            hv_from_sums(
                self.ret_sum,
                self.ret_sum_sq,
                self.length,
                self.annual_length,
            )
        } else {
            f64::NAN
        };

        if self.hv_len == self.annual_length {
            let old = self.hv_window[self.hv_head];
            if old.is_finite() {
                sorted_remove_one(&mut self.hv_sorted, old);
            } else {
                self.invalid_hv -= 1;
            }
        }

        self.hv_window[self.hv_head] = hv;
        if hv.is_finite() {
            let pos = self.hv_sorted.partition_point(|x| *x < hv);
            self.hv_sorted.insert(pos, hv);
        } else {
            self.invalid_hv += 1;
        }

        self.hv_head += 1;
        if self.hv_head == self.annual_length {
            self.hv_head = 0;
        }
        if self.hv_len < self.annual_length {
            self.hv_len += 1;
        }

        let hvp = if self.hv_len == self.annual_length && self.invalid_hv == 0 && hv.is_finite() {
            let rank = self.hv_sorted.partition_point(|x| *x < hv);
            (rank as f64 / self.annual_length as f64) * 100.0
        } else {
            f64::NAN
        };

        if self.sma_len == self.length {
            let old = self.sma_window[self.sma_head];
            if old.is_finite() {
                self.sma_sum -= old;
            } else {
                self.invalid_sma -= 1;
            }
        }

        self.sma_window[self.sma_head] = hvp;
        if hvp.is_finite() {
            self.sma_sum += hvp;
        } else {
            self.invalid_sma += 1;
        }

        self.sma_head += 1;
        if self.sma_head == self.length {
            self.sma_head = 0;
        }
        if self.sma_len < self.length {
            self.sma_len += 1;
        }

        let hvp_sma = if self.sma_len == self.length && self.invalid_sma == 0 {
            self.sma_sum / self.length as f64
        } else {
            f64::NAN
        };

        if hvp.is_finite() {
            Some((hvp, hvp_sma))
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn get_warmup_period(&self) -> usize {
        self.length + self.annual_length - 1
    }
}

#[derive(Clone, Debug)]
pub struct HistoricalVolatilityPercentileBatchRange {
    pub length: (usize, usize, usize),
    pub annual_length: (usize, usize, usize),
}

impl Default for HistoricalVolatilityPercentileBatchRange {
    fn default() -> Self {
        Self {
            length: (21, 21, 0),
            annual_length: (252, 252, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct HistoricalVolatilityPercentileBatchBuilder {
    range: HistoricalVolatilityPercentileBatchRange,
    kernel: Kernel,
}

impl HistoricalVolatilityPercentileBatchBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline]
    pub fn length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.length = (start, end, step);
        self
    }

    #[inline]
    pub fn length_static(mut self, value: usize) -> Self {
        self.range.length = (value, value, 0);
        self
    }

    #[inline]
    pub fn annual_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.annual_length = (start, end, step);
        self
    }

    #[inline]
    pub fn annual_length_static(mut self, value: usize) -> Self {
        self.range.annual_length = (value, value, 0);
        self
    }

    #[inline]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<HistoricalVolatilityPercentileBatchOutput, HistoricalVolatilityPercentileError>
    {
        historical_volatility_percentile_batch_with_kernel(data, &self.range, self.kernel)
    }

    #[inline]
    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<HistoricalVolatilityPercentileBatchOutput, HistoricalVolatilityPercentileError>
    {
        historical_volatility_percentile_batch_with_kernel(
            source_type(candles, source),
            &self.range,
            self.kernel,
        )
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct HistoricalVolatilityPercentileBatchConfig {
    pub length_range: Vec<usize>,
    pub annual_length_range: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct HistoricalVolatilityPercentileBatchOutput {
    pub hvp: Vec<f64>,
    pub hvp_sma: Vec<f64>,
    pub combos: Vec<HistoricalVolatilityPercentileParams>,
    pub rows: usize,
    pub cols: usize,
}

impl HistoricalVolatilityPercentileBatchOutput {
    #[inline]
    pub fn row_for_params(&self, params: &HistoricalVolatilityPercentileParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.length.unwrap_or(21) == params.length.unwrap_or(21)
                && c.annual_length.unwrap_or(252) == params.annual_length.unwrap_or(252)
        })
    }

    #[inline]
    pub fn hvp_for(&self, params: &HistoricalVolatilityPercentileParams) -> Option<&[f64]> {
        self.row_for_params(params).and_then(|row| {
            let start = row * self.cols;
            self.hvp.get(start..start + self.cols)
        })
    }

    #[inline]
    pub fn hvp_sma_for(&self, params: &HistoricalVolatilityPercentileParams) -> Option<&[f64]> {
        self.row_for_params(params).and_then(|row| {
            let start = row * self.cols;
            self.hvp_sma.get(start..start + self.cols)
        })
    }
}

#[inline]
pub fn expand_grid_historical_volatility_percentile(
    range: &HistoricalVolatilityPercentileBatchRange,
) -> Result<Vec<HistoricalVolatilityPercentileParams>, HistoricalVolatilityPercentileError> {
    fn axis(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, HistoricalVolatilityPercentileError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start <= end {
            let mut out = Vec::new();
            let mut x = start;
            while x <= end {
                out.push(x);
                match x.checked_add(step) {
                    Some(next) if next > x => x = next,
                    _ => break,
                }
            }
            if out.is_empty() {
                return Err(HistoricalVolatilityPercentileError::InvalidRange {
                    start: start.to_string(),
                    end: end.to_string(),
                    step: step.to_string(),
                });
            }
            Ok(out)
        } else {
            let mut out = Vec::new();
            let mut x = start;
            while x >= end {
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
                return Err(HistoricalVolatilityPercentileError::InvalidRange {
                    start: start.to_string(),
                    end: end.to_string(),
                    step: step.to_string(),
                });
            }
            Ok(out)
        }
    }

    let lengths = axis(range.length)?;
    let annual_lengths = axis(range.annual_length)?;
    let mut combos = Vec::with_capacity(lengths.len() * annual_lengths.len());
    for length in lengths {
        for annual_length in annual_lengths.iter().copied() {
            combos.push(HistoricalVolatilityPercentileParams {
                length: Some(length),
                annual_length: Some(annual_length),
            });
        }
    }
    Ok(combos)
}

#[inline]
pub fn historical_volatility_percentile_batch_with_kernel(
    data: &[f64],
    sweep: &HistoricalVolatilityPercentileBatchRange,
    kernel: Kernel,
) -> Result<HistoricalVolatilityPercentileBatchOutput, HistoricalVolatilityPercentileError> {
    let batch = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => {
            return Err(HistoricalVolatilityPercentileError::InvalidKernelForBatch(
                other,
            ))
        }
    };
    historical_volatility_percentile_batch_par_slice(data, sweep, batch.to_non_batch())
}

#[inline(always)]
pub fn historical_volatility_percentile_batch_slice(
    data: &[f64],
    sweep: &HistoricalVolatilityPercentileBatchRange,
    kernel: Kernel,
) -> Result<HistoricalVolatilityPercentileBatchOutput, HistoricalVolatilityPercentileError> {
    historical_volatility_percentile_batch_inner(data, sweep, kernel, false)
}

#[inline(always)]
pub fn historical_volatility_percentile_batch_par_slice(
    data: &[f64],
    sweep: &HistoricalVolatilityPercentileBatchRange,
    kernel: Kernel,
) -> Result<HistoricalVolatilityPercentileBatchOutput, HistoricalVolatilityPercentileError> {
    historical_volatility_percentile_batch_inner(data, sweep, kernel, true)
}

#[inline(always)]
fn historical_volatility_percentile_batch_inner(
    data: &[f64],
    sweep: &HistoricalVolatilityPercentileBatchRange,
    _kernel: Kernel,
    parallel: bool,
) -> Result<HistoricalVolatilityPercentileBatchOutput, HistoricalVolatilityPercentileError> {
    let combos = expand_grid_historical_volatility_percentile(sweep)?;
    if data.is_empty() {
        return Err(HistoricalVolatilityPercentileError::EmptyInputData);
    }
    let first =
        first_valid_source(data).ok_or(HistoricalVolatilityPercentileError::AllValuesNaN)?;
    let max_needed = combos
        .iter()
        .map(|p| p.length.unwrap_or(21) + p.annual_length.unwrap_or(252) - 1)
        .max()
        .unwrap_or(0);
    if data.len() - first < max_needed {
        return Err(HistoricalVolatilityPercentileError::NotEnoughValidData {
            needed: max_needed,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let mut hvp_mu = make_uninit_matrix(rows, cols);
    let mut hvp_sma_mu = make_uninit_matrix(rows, cols);
    let warmups_hvp: Vec<usize> = combos
        .iter()
        .map(|p| {
            hvp_warmup(
                p.length.unwrap_or(21),
                p.annual_length.unwrap_or(252),
                first,
            )
        })
        .collect();
    let warmups_hvp_sma: Vec<usize> = combos
        .iter()
        .map(|p| {
            hvp_sma_warmup(
                p.length.unwrap_or(21),
                p.annual_length.unwrap_or(252),
                first,
            )
        })
        .collect();
    init_matrix_prefixes(&mut hvp_mu, cols, &warmups_hvp);
    init_matrix_prefixes(&mut hvp_sma_mu, cols, &warmups_hvp_sma);

    let mut hvp_guard = ManuallyDrop::new(hvp_mu);
    let mut hvp_sma_guard = ManuallyDrop::new(hvp_sma_mu);
    let hvp = unsafe {
        core::slice::from_raw_parts_mut(hvp_guard.as_mut_ptr() as *mut f64, hvp_guard.len())
    };
    let hvp_sma = unsafe {
        core::slice::from_raw_parts_mut(hvp_sma_guard.as_mut_ptr() as *mut f64, hvp_sma_guard.len())
    };

    historical_volatility_percentile_batch_inner_into(
        data,
        sweep,
        Kernel::Scalar,
        parallel,
        hvp,
        hvp_sma,
    )?;

    let hvp_values = unsafe {
        Vec::from_raw_parts(
            hvp_guard.as_mut_ptr() as *mut f64,
            hvp_guard.len(),
            hvp_guard.capacity(),
        )
    };
    let hvp_sma_values = unsafe {
        Vec::from_raw_parts(
            hvp_sma_guard.as_mut_ptr() as *mut f64,
            hvp_sma_guard.len(),
            hvp_sma_guard.capacity(),
        )
    };

    Ok(HistoricalVolatilityPercentileBatchOutput {
        hvp: hvp_values,
        hvp_sma: hvp_sma_values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn historical_volatility_percentile_batch_inner_into(
    data: &[f64],
    sweep: &HistoricalVolatilityPercentileBatchRange,
    _kernel: Kernel,
    parallel: bool,
    out_hvp: &mut [f64],
    out_hvp_sma: &mut [f64],
) -> Result<Vec<HistoricalVolatilityPercentileParams>, HistoricalVolatilityPercentileError> {
    let combos = expand_grid_historical_volatility_percentile(sweep)?;
    if data.is_empty() {
        return Err(HistoricalVolatilityPercentileError::EmptyInputData);
    }
    let first =
        first_valid_source(data).ok_or(HistoricalVolatilityPercentileError::AllValuesNaN)?;
    let max_needed = combos
        .iter()
        .map(|p| p.length.unwrap_or(21) + p.annual_length.unwrap_or(252) - 1)
        .max()
        .unwrap_or(0);
    if data.len() - first < max_needed {
        return Err(HistoricalVolatilityPercentileError::NotEnoughValidData {
            needed: max_needed,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or(HistoricalVolatilityPercentileError::InvalidInput(
            "rows*cols overflow",
        ))?;
    if out_hvp.len() != total || out_hvp_sma.len() != total {
        return Err(HistoricalVolatilityPercentileError::OutputLengthMismatch {
            expected: total,
            got: out_hvp.len().max(out_hvp_sma.len()),
        });
    }

    let out_hvp_mu = unsafe {
        core::slice::from_raw_parts_mut(out_hvp.as_mut_ptr() as *mut MaybeUninit<f64>, total)
    };
    let out_hvp_sma_mu = unsafe {
        core::slice::from_raw_parts_mut(out_hvp_sma.as_mut_ptr() as *mut MaybeUninit<f64>, total)
    };
    let warmups_hvp: Vec<usize> = combos
        .iter()
        .map(|p| {
            hvp_warmup(
                p.length.unwrap_or(21),
                p.annual_length.unwrap_or(252),
                first,
            )
        })
        .collect();
    let warmups_hvp_sma: Vec<usize> = combos
        .iter()
        .map(|p| {
            hvp_sma_warmup(
                p.length.unwrap_or(21),
                p.annual_length.unwrap_or(252),
                first,
            )
        })
        .collect();
    init_matrix_prefixes(out_hvp_mu, cols, &warmups_hvp);
    init_matrix_prefixes(out_hvp_sma_mu, cols, &warmups_hvp_sma);

    let returns = build_returns(data);
    let do_row = |row: usize, hvp_row: &mut [f64], hvp_sma_row: &mut [f64]| {
        let params = &combos[row];
        hvp_compute_into(
            &returns,
            params.length.unwrap_or(21),
            params.annual_length.unwrap_or(252),
            first,
            hvp_row,
            hvp_sma_row,
        );
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_hvp
                .par_chunks_mut(cols)
                .zip(out_hvp_sma.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, (hvp_row, hvp_sma_row))| do_row(row, hvp_row, hvp_sma_row));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, (hvp_row, hvp_sma_row)) in out_hvp
                .chunks_mut(cols)
                .zip(out_hvp_sma.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, hvp_row, hvp_sma_row);
            }
        }
    } else {
        for (row, (hvp_row, hvp_sma_row)) in out_hvp
            .chunks_mut(cols)
            .zip(out_hvp_sma.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, hvp_row, hvp_sma_row);
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "historical_volatility_percentile")]
#[pyo3(signature = (data, length=21, annual_length=252, kernel=None))]
pub fn historical_volatility_percentile_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length: usize,
    annual_length: usize,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let data = data.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let params = HistoricalVolatilityPercentileParams {
        length: Some(length),
        annual_length: Some(annual_length),
    };
    let input = HistoricalVolatilityPercentileInput::from_slice(data, params);
    let output = py
        .allow_threads(|| historical_volatility_percentile_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((output.hvp.into_pyarray(py), output.hvp_sma.into_pyarray(py)))
}

#[cfg(feature = "python")]
#[pyclass(name = "HistoricalVolatilityPercentileStream")]
pub struct HistoricalVolatilityPercentileStreamPy {
    stream: HistoricalVolatilityPercentileStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl HistoricalVolatilityPercentileStreamPy {
    #[new]
    #[pyo3(signature = (length=21, annual_length=252))]
    fn new(length: usize, annual_length: usize) -> PyResult<Self> {
        let params = HistoricalVolatilityPercentileParams {
            length: Some(length),
            annual_length: Some(annual_length),
        };
        let stream = HistoricalVolatilityPercentileStream::try_new(params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "historical_volatility_percentile_batch")]
#[pyo3(signature = (data, length_range, annual_length_range, kernel=None))]
pub fn historical_volatility_percentile_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    annual_length_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let sweep = HistoricalVolatilityPercentileBatchRange {
        length: length_range,
        annual_length: annual_length_range,
    };
    let combos = expand_grid_historical_volatility_percentile(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;
    let hvp_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let hvp_sma_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let hvp_out = unsafe { hvp_arr.as_slice_mut()? };
    let hvp_sma_out = unsafe { hvp_sma_arr.as_slice_mut()? };
    let kernel = validate_kernel(kernel, true)?;

    py.allow_threads(|| {
        let batch_kernel = match kernel {
            Kernel::Auto => detect_best_batch_kernel(),
            other => other,
        };
        historical_volatility_percentile_batch_inner_into(
            data,
            &sweep,
            batch_kernel.to_non_batch(),
            true,
            hvp_out,
            hvp_sma_out,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("hvp", hvp_arr.reshape((rows, cols))?)?;
    dict.set_item("hvp_sma", hvp_sma_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "lengths",
        combos
            .iter()
            .map(|p| p.length.unwrap_or(21) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "annual_lengths",
        combos
            .iter()
            .map(|p| p.annual_length.unwrap_or(252) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_historical_volatility_percentile_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(historical_volatility_percentile_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        historical_volatility_percentile_batch_py,
        m
    )?)?;
    m.add_class::<HistoricalVolatilityPercentileStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "historical_volatility_percentile_js")]
pub fn historical_volatility_percentile_js(
    data: &[f64],
    length: usize,
    annual_length: usize,
) -> Result<JsValue, JsValue> {
    let params = HistoricalVolatilityPercentileParams {
        length: Some(length),
        annual_length: Some(annual_length),
    };
    let input = HistoricalVolatilityPercentileInput::from_slice(data, params);
    let mut hvp = vec![0.0; data.len()];
    let mut hvp_sma = vec![0.0; data.len()];
    historical_volatility_percentile_into_slice(&mut hvp, &mut hvp_sma, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("hvp"),
        &serde_wasm_bindgen::to_value(&hvp).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("hvp_sma"),
        &serde_wasm_bindgen::to_value(&hvp_sma).unwrap(),
    )?;
    Ok(obj.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "historical_volatility_percentile_batch_js")]
pub fn historical_volatility_percentile_batch_js(
    data: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: HistoricalVolatilityPercentileBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.length_range.len() != 3 || config.annual_length_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: ranges must have exactly 3 elements [start, end, step]",
        ));
    }

    let sweep = HistoricalVolatilityPercentileBatchRange {
        length: (
            config.length_range[0],
            config.length_range[1],
            config.length_range[2],
        ),
        annual_length: (
            config.annual_length_range[0],
            config.annual_length_range[1],
            config.annual_length_range[2],
        ),
    };
    let combos = expand_grid_historical_volatility_percentile(&sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
    let mut hvp = vec![0.0; total];
    let mut hvp_sma = vec![0.0; total];
    historical_volatility_percentile_batch_inner_into(
        data,
        &sweep,
        Kernel::Scalar,
        false,
        &mut hvp,
        &mut hvp_sma,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("hvp"),
        &serde_wasm_bindgen::to_value(&hvp).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("hvp_sma"),
        &serde_wasm_bindgen::to_value(&hvp_sma).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("rows"),
        &JsValue::from_f64(rows as f64),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("cols"),
        &JsValue::from_f64(cols as f64),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("combos"),
        &serde_wasm_bindgen::to_value(&combos).unwrap(),
    )?;
    Ok(obj.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn historical_volatility_percentile_alloc(len: usize) -> *mut f64 {
    let mut v = Vec::<f64>::with_capacity(2 * len);
    let ptr = v.as_mut_ptr();
    std::mem::forget(v);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn historical_volatility_percentile_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, 0, 2 * len);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn historical_volatility_percentile_into(
    data_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length: usize,
    annual_length: usize,
) -> Result<(), JsValue> {
    if data_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to historical_volatility_percentile_into",
        ));
    }

    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, 2 * len);
        let (hvp, hvp_sma) = out.split_at_mut(len);
        let params = HistoricalVolatilityPercentileParams {
            length: Some(length),
            annual_length: Some(annual_length),
        };
        let input = HistoricalVolatilityPercentileInput::from_slice(data, params);
        historical_volatility_percentile_into_slice(hvp, hvp_sma, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "historical_volatility_percentile_into_host")]
pub fn historical_volatility_percentile_into_host(
    data: &[f64],
    out_ptr: *mut f64,
    length: usize,
    annual_length: usize,
) -> Result<(), JsValue> {
    if out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to historical_volatility_percentile_into_host",
        ));
    }
    unsafe {
        let out = std::slice::from_raw_parts_mut(out_ptr, 2 * data.len());
        let (hvp, hvp_sma) = out.split_at_mut(data.len());
        let params = HistoricalVolatilityPercentileParams {
            length: Some(length),
            annual_length: Some(annual_length),
        };
        let input = HistoricalVolatilityPercentileInput::from_slice(data, params);
        historical_volatility_percentile_into_slice(hvp, hvp_sma, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn historical_volatility_percentile_batch_into(
    data_ptr: *const f64,
    hvp_ptr: *mut f64,
    hvp_sma_ptr: *mut f64,
    len: usize,
    length_start: usize,
    length_end: usize,
    length_step: usize,
    annual_length_start: usize,
    annual_length_end: usize,
    annual_length_step: usize,
) -> Result<usize, JsValue> {
    if data_ptr.is_null() || hvp_ptr.is_null() || hvp_sma_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to historical_volatility_percentile_batch_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let sweep = HistoricalVolatilityPercentileBatchRange {
            length: (length_start, length_end, length_step),
            annual_length: (annual_length_start, annual_length_end, annual_length_step),
        };
        let combos = expand_grid_historical_volatility_percentile(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let total = rows
            .checked_mul(len)
            .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
        let hvp = std::slice::from_raw_parts_mut(hvp_ptr, total);
        let hvp_sma = std::slice::from_raw_parts_mut(hvp_sma_ptr, total);
        historical_volatility_percentile_batch_inner_into(
            data,
            &sweep,
            Kernel::Scalar,
            false,
            hvp,
            hvp_sma,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn historical_volatility_percentile_output_into_js(
    data: &[f64],
    length: usize,
    annual_length: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = historical_volatility_percentile_js(data, length, annual_length)?;
    crate::write_wasm_object_f64_outputs(
        "historical_volatility_percentile_output_into_js",
        &value,
        out,
    )
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn historical_volatility_percentile_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = historical_volatility_percentile_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "historical_volatility_percentile_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn series_close(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.len() == b.len()
            && a.iter().zip(b.iter()).all(|(&x, &y)| {
                (x.is_nan() && y.is_nan())
                    || (x.is_finite() && y.is_finite() && (x - y).abs() <= tol)
            })
    }

    fn naive_hvp(data: &[f64], length: usize, annual_length: usize) -> (Vec<f64>, Vec<f64>) {
        let n = data.len();
        let returns = build_returns(data);
        let mut hv = vec![f64::NAN; n];
        for i in 0..n {
            if i + 1 < length {
                continue;
            }
            let start = i + 1 - length;
            let window = &returns[start..=i];
            if window.iter().all(|x| x.is_finite()) {
                let sum: f64 = window.iter().sum();
                let sum_sq: f64 = window.iter().map(|x| x * x).sum();
                hv[i] = hv_from_sums(sum, sum_sq, length, annual_length);
            }
        }

        let mut hvp = vec![f64::NAN; n];
        for i in 0..n {
            if i + 1 < annual_length {
                continue;
            }
            let start = i + 1 - annual_length;
            let window = &hv[start..=i];
            if window.iter().all(|x| x.is_finite()) {
                let count = window.iter().filter(|&&x| x < hv[i]).count();
                hvp[i] = count as f64 / annual_length as f64 * 100.0;
            }
        }

        let mut hvp_sma = vec![f64::NAN; n];
        for i in 0..n {
            if i + 1 < length {
                continue;
            }
            let start = i + 1 - length;
            let window = &hvp[start..=i];
            if window.iter().all(|x| x.is_finite()) {
                hvp_sma[i] = window.iter().sum::<f64>() / length as f64;
            }
        }
        (hvp, hvp_sma)
    }

    fn sample_data() -> Vec<f64> {
        vec![
            100.0, 101.0, 102.5, 101.8, 103.2, 104.1, 103.7, 105.3, 104.9, 106.0, 107.4, 106.8,
            108.1, 109.5, 108.9, 110.2, 111.0, 110.4, 112.3, 113.1, 112.6, 114.4, 115.8, 115.1,
            116.9, 118.0, 117.4, 119.3, 120.1, 119.8, 121.5, 122.6, 122.0, 123.9, 124.8, 124.2,
        ]
    }

    #[test]
    fn historical_volatility_percentile_matches_naive() {
        let data = sample_data();
        let params = HistoricalVolatilityPercentileParams {
            length: Some(5),
            annual_length: Some(10),
        };
        let input = HistoricalVolatilityPercentileInput::from_slice(&data, params);
        let out = historical_volatility_percentile(&input).expect("hvp output");
        let (exp_hvp, exp_hvp_sma) = naive_hvp(&data, 5, 10);
        assert!(series_close(&out.hvp, &exp_hvp, 1e-12));
        assert!(series_close(&out.hvp_sma, &exp_hvp_sma, 1e-12));
    }

    #[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
    #[test]
    fn historical_volatility_percentile_into_matches_api() -> Result<(), Box<dyn Error>> {
        let data = sample_data();
        let params = HistoricalVolatilityPercentileParams {
            length: Some(5),
            annual_length: Some(10),
        };
        let input = HistoricalVolatilityPercentileInput::from_slice(&data, params);
        let direct = historical_volatility_percentile(&input)?;
        let mut hvp = vec![0.0; data.len()];
        let mut hvp_sma = vec![0.0; data.len()];
        historical_volatility_percentile_into(&input, &mut hvp, &mut hvp_sma)?;
        assert!(series_close(&direct.hvp, &hvp, 1e-12));
        assert!(series_close(&direct.hvp_sma, &hvp_sma, 1e-12));
        Ok(())
    }

    #[test]
    fn historical_volatility_percentile_stream_matches_batch() {
        let data = sample_data();
        let params = HistoricalVolatilityPercentileParams {
            length: Some(5),
            annual_length: Some(10),
        };
        let input = HistoricalVolatilityPercentileInput::from_slice(&data, params.clone());
        let batch = historical_volatility_percentile(&input).expect("batch output");
        let mut stream =
            HistoricalVolatilityPercentileStream::try_new(params).expect("stream output");
        let mut hvp = Vec::with_capacity(data.len());
        let mut hvp_sma = Vec::with_capacity(data.len());
        for &value in &data {
            match stream.update(value) {
                Some((v0, v1)) => {
                    hvp.push(v0);
                    hvp_sma.push(v1);
                }
                None => {
                    hvp.push(f64::NAN);
                    hvp_sma.push(f64::NAN);
                }
            }
        }
        assert!(series_close(&batch.hvp, &hvp, 1e-12));
        assert!(series_close(&batch.hvp_sma, &hvp_sma, 1e-12));
    }

    #[test]
    fn historical_volatility_percentile_batch_single_param_matches_single() {
        let data = sample_data();
        let sweep = HistoricalVolatilityPercentileBatchRange {
            length: (5, 5, 0),
            annual_length: (10, 10, 0),
        };
        let batch = historical_volatility_percentile_batch_with_kernel(&data, &sweep, Kernel::Auto)
            .expect("batch output");
        let input = HistoricalVolatilityPercentileInput::from_slice(
            &data,
            HistoricalVolatilityPercentileParams {
                length: Some(5),
                annual_length: Some(10),
            },
        );
        let single = historical_volatility_percentile(&input).expect("single output");
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, data.len());
        assert!(series_close(&batch.hvp, &single.hvp, 1e-12));
        assert!(series_close(&batch.hvp_sma, &single.hvp_sma, 1e-12));
    }

    #[test]
    fn historical_volatility_percentile_batch_sweep_matches_singles() {
        let data = sample_data();
        let sweep = HistoricalVolatilityPercentileBatchRange {
            length: (4, 5, 1),
            annual_length: (8, 10, 2),
        };
        let batch = historical_volatility_percentile_batch_with_kernel(&data, &sweep, Kernel::Auto)
            .expect("batch output");
        assert_eq!(batch.rows, 4);
        for params in &batch.combos {
            let row = batch.row_for_params(params).expect("row");
            let start = row * batch.cols;
            let end = start + batch.cols;
            let input = HistoricalVolatilityPercentileInput::from_slice(&data, params.clone());
            let single = historical_volatility_percentile(&input).expect("single");
            assert!(series_close(&batch.hvp[start..end], &single.hvp, 1e-12));
            assert!(series_close(
                &batch.hvp_sma[start..end],
                &single.hvp_sma,
                1e-12
            ));
        }
    }

    #[test]
    fn historical_volatility_percentile_rejects_invalid_length() {
        let data = sample_data();
        let input = HistoricalVolatilityPercentileInput::from_slice(
            &data,
            HistoricalVolatilityPercentileParams {
                length: Some(1),
                annual_length: Some(10),
            },
        );
        let err = historical_volatility_percentile(&input).expect_err("invalid length");
        assert!(matches!(
            err,
            HistoricalVolatilityPercentileError::InvalidLength { .. }
        ));
    }
}
