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
    alloc_uninit_f64, alloc_with_nan_prefix, detect_best_batch_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum VolatilityQualityIndexData<'a> {
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
pub struct VolatilityQualityIndexOutput {
    pub vqi_sum: Vec<f64>,
    pub fast_sma: Vec<f64>,
    pub slow_sma: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct VolatilityQualityIndexParams {
    pub fast_length: Option<usize>,
    pub slow_length: Option<usize>,
}

impl Default for VolatilityQualityIndexParams {
    fn default() -> Self {
        Self {
            fast_length: Some(9),
            slow_length: Some(200),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VolatilityQualityIndexInput<'a> {
    pub data: VolatilityQualityIndexData<'a>,
    pub params: VolatilityQualityIndexParams,
}

impl<'a> VolatilityQualityIndexInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: VolatilityQualityIndexParams) -> Self {
        Self {
            data: VolatilityQualityIndexData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: VolatilityQualityIndexParams,
    ) -> Self {
        Self {
            data: VolatilityQualityIndexData::Slices {
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
        Self::from_candles(candles, VolatilityQualityIndexParams::default())
    }

    #[inline]
    pub fn get_fast_length(&self) -> usize {
        self.params.fast_length.unwrap_or(9)
    }

    #[inline]
    pub fn get_slow_length(&self) -> usize {
        self.params.slow_length.unwrap_or(200)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct VolatilityQualityIndexBuilder {
    fast_length: Option<usize>,
    slow_length: Option<usize>,
    kernel: Kernel,
}

impl Default for VolatilityQualityIndexBuilder {
    fn default() -> Self {
        Self {
            fast_length: None,
            slow_length: None,
            kernel: Kernel::Auto,
        }
    }
}

impl VolatilityQualityIndexBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn fast_length(mut self, value: usize) -> Self {
        self.fast_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn slow_length(mut self, value: usize) -> Self {
        self.slow_length = Some(value);
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
    ) -> Result<VolatilityQualityIndexOutput, VolatilityQualityIndexError> {
        let input = VolatilityQualityIndexInput::from_candles(
            candles,
            VolatilityQualityIndexParams {
                fast_length: self.fast_length,
                slow_length: self.slow_length,
            },
        );
        volatility_quality_index_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<VolatilityQualityIndexOutput, VolatilityQualityIndexError> {
        let input = VolatilityQualityIndexInput::from_slices(
            open,
            high,
            low,
            close,
            VolatilityQualityIndexParams {
                fast_length: self.fast_length,
                slow_length: self.slow_length,
            },
        );
        volatility_quality_index_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<VolatilityQualityIndexStream, VolatilityQualityIndexError> {
        VolatilityQualityIndexStream::try_new(VolatilityQualityIndexParams {
            fast_length: self.fast_length,
            slow_length: self.slow_length,
        })
    }
}

#[derive(Debug, Error)]
pub enum VolatilityQualityIndexError {
    #[error("volatility_quality_index: Input data slice is empty.")]
    EmptyInputData,
    #[error("volatility_quality_index: All values are NaN.")]
    AllValuesNaN,
    #[error("volatility_quality_index: Inconsistent slice lengths: open={open_len}, high={high_len}, low={low_len}, close={close_len}")]
    InconsistentSliceLengths {
        open_len: usize,
        high_len: usize,
        low_len: usize,
        close_len: usize,
    },
    #[error("volatility_quality_index: Invalid fast_length: {fast_length}")]
    InvalidFastLength { fast_length: usize },
    #[error("volatility_quality_index: Invalid slow_length: {slow_length}")]
    InvalidSlowLength { slow_length: usize },
    #[error(
        "volatility_quality_index: Output length mismatch: expected = {expected}, got = {got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("volatility_quality_index: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("volatility_quality_index: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Debug, Clone)]
pub struct VolatilityQualityIndexStream {
    prev_close: f64,
    prev_vqi_t: f64,
    cumulative: f64,
    fast: RunningSma,
    slow: RunningSma,
}

#[derive(Debug, Clone)]
struct RunningSma {
    period: usize,
    sum: f64,
    values: Vec<f64>,
    head: usize,
    count: usize,
}

impl RunningSma {
    #[inline]
    fn new(period: usize) -> Self {
        Self {
            period,
            sum: 0.0,
            values: vec![0.0; period],
            head: 0,
            count: 0,
        }
    }

    #[inline]
    fn update(&mut self, value: f64) -> f64 {
        if self.count == self.period {
            self.sum -= self.values[self.head];
        } else {
            self.count += 1;
        }
        self.values[self.head] = value;
        self.sum += value;
        self.head += 1;
        if self.head == self.period {
            self.head = 0;
        }
        if self.count == self.period {
            self.sum / self.period as f64
        } else {
            f64::NAN
        }
    }
}

impl VolatilityQualityIndexStream {
    pub fn try_new(
        params: VolatilityQualityIndexParams,
    ) -> Result<Self, VolatilityQualityIndexError> {
        let fast_length = validate_fast_length(params.fast_length.unwrap_or(9))?;
        let slow_length = validate_slow_length(params.slow_length.unwrap_or(200))?;
        Ok(Self {
            prev_close: f64::NAN,
            prev_vqi_t: 0.0,
            cumulative: 0.0,
            fast: RunningSma::new(fast_length),
            slow: RunningSma::new(slow_length),
        })
    }

    #[inline]
    pub fn update(&mut self, open: f64, high: f64, low: f64, close: f64) -> (f64, f64, f64) {
        let (vqi_t, raw) =
            compute_vqi_point(self.prev_close, self.prev_vqi_t, open, high, low, close);
        self.prev_vqi_t = vqi_t;
        self.prev_close = close;
        self.cumulative += raw;
        (
            self.cumulative,
            self.fast.update(self.cumulative),
            self.slow.update(self.cumulative),
        )
    }
}

#[inline(always)]
fn validate_fast_length(fast_length: usize) -> Result<usize, VolatilityQualityIndexError> {
    if fast_length == 0 {
        return Err(VolatilityQualityIndexError::InvalidFastLength { fast_length });
    }
    Ok(fast_length)
}

#[inline(always)]
fn validate_slow_length(slow_length: usize) -> Result<usize, VolatilityQualityIndexError> {
    if slow_length == 0 {
        return Err(VolatilityQualityIndexError::InvalidSlowLength { slow_length });
    }
    Ok(slow_length)
}

#[inline(always)]
fn extract_ohlc<'a>(
    input: &'a VolatilityQualityIndexInput<'a>,
) -> Result<(&'a [f64], &'a [f64], &'a [f64], &'a [f64]), VolatilityQualityIndexError> {
    let (open, high, low, close) = match &input.data {
        VolatilityQualityIndexData::Candles { candles } => (
            candles.open.as_slice(),
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
        ),
        VolatilityQualityIndexData::Slices {
            open,
            high,
            low,
            close,
        } => (*open, *high, *low, *close),
    };
    if open.is_empty() || high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(VolatilityQualityIndexError::EmptyInputData);
    }
    if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
        return Err(VolatilityQualityIndexError::InconsistentSliceLengths {
            open_len: open.len(),
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }
    Ok((open, high, low, close))
}

#[inline(always)]
fn compute_vqi_point(
    prev_close: f64,
    prev_vqi_t: f64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
) -> (f64, f64) {
    let range = high - low;
    let tr = if high.is_finite() && low.is_finite() {
        if prev_close.is_finite() {
            let mut tr = range;
            let hc = (high - prev_close).abs();
            if hc > tr {
                tr = hc;
            }
            let lc = (low - prev_close).abs();
            if lc > tr {
                tr = lc;
            }
            tr
        } else {
            range
        }
    } else {
        f64::NAN
    };

    let vqi_t = if prev_close.is_finite()
        && open.is_finite()
        && high.is_finite()
        && low.is_finite()
        && close.is_finite()
        && tr.is_finite()
        && tr != 0.0
        && range.is_finite()
        && range != 0.0
    {
        0.5 * (((close - prev_close) / tr) + ((close - open) / range))
    } else {
        prev_vqi_t
    };

    let raw = if prev_close.is_finite() && open.is_finite() && close.is_finite() {
        vqi_t.abs() * 0.5 * ((close - prev_close) + (close - open))
    } else {
        0.0
    };

    (vqi_t, raw)
}

#[inline(always)]
fn compute_vqi_sum_series(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    let len = close.len();
    let mut out = alloc_uninit_f64(len);
    compute_vqi_sum_series_into(open, high, low, close, &mut out);
    out
}

#[inline(always)]
fn compute_vqi_sum_series_into(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    out: &mut [f64],
) {
    let mut prev_close = f64::NAN;
    let mut prev_vqi_t = 0.0;
    let mut cumulative = 0.0;
    for i in 0..close.len() {
        let (vqi_t, raw) =
            compute_vqi_point(prev_close, prev_vqi_t, open[i], high[i], low[i], close[i]);
        prev_vqi_t = vqi_t;
        prev_close = close[i];
        cumulative += raw;
        out[i] = cumulative;
    }
}

#[inline(always)]
fn sma_into(src: &[f64], period: usize, dst: &mut [f64]) {
    let len = src.len();
    let warm = period.saturating_sub(1).min(len);
    if warm > 0 {
        dst[..warm].fill(f64::NAN);
    }
    if period > len {
        return;
    }
    let mut sum = 0.0;
    for &value in &src[..period] {
        sum += value;
    }
    dst[period - 1] = sum / period as f64;
    for i in period..len {
        sum += src[i] - src[i - period];
        dst[i] = sum / period as f64;
    }
}

#[inline]
pub fn volatility_quality_index(
    input: &VolatilityQualityIndexInput,
) -> Result<VolatilityQualityIndexOutput, VolatilityQualityIndexError> {
    volatility_quality_index_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn volatility_quality_index_with_kernel(
    input: &VolatilityQualityIndexInput,
    kernel: Kernel,
) -> Result<VolatilityQualityIndexOutput, VolatilityQualityIndexError> {
    let (open, high, low, close) = extract_ohlc(input)?;
    let fast_length = validate_fast_length(input.get_fast_length())?;
    let slow_length = validate_slow_length(input.get_slow_length())?;
    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other.to_non_batch(),
    };
    let _ = chosen;
    let mut vqi_sum = alloc_uninit_f64(close.len());
    compute_vqi_sum_series_into(open, high, low, close, &mut vqi_sum);
    let mut fast_sma = alloc_uninit_f64(close.len());
    let mut slow_sma = alloc_uninit_f64(close.len());
    sma_into(&vqi_sum, fast_length, &mut fast_sma);
    sma_into(&vqi_sum, slow_length, &mut slow_sma);
    Ok(VolatilityQualityIndexOutput {
        vqi_sum,
        fast_sma,
        slow_sma,
    })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn volatility_quality_index_into(
    input: &VolatilityQualityIndexInput,
    out_vqi_sum: &mut [f64],
    out_fast_sma: &mut [f64],
    out_slow_sma: &mut [f64],
) -> Result<(), VolatilityQualityIndexError> {
    volatility_quality_index_into_slice(
        out_vqi_sum,
        out_fast_sma,
        out_slow_sma,
        input,
        Kernel::Auto,
    )
}

#[inline]
pub fn volatility_quality_index_into_slice(
    out_vqi_sum: &mut [f64],
    out_fast_sma: &mut [f64],
    out_slow_sma: &mut [f64],
    input: &VolatilityQualityIndexInput,
    kernel: Kernel,
) -> Result<(), VolatilityQualityIndexError> {
    let (open, high, low, close) = extract_ohlc(input)?;
    let len = close.len();
    if out_vqi_sum.len() != len {
        return Err(VolatilityQualityIndexError::OutputLengthMismatch {
            expected: len,
            got: out_vqi_sum.len(),
        });
    }
    if out_fast_sma.len() != len {
        return Err(VolatilityQualityIndexError::OutputLengthMismatch {
            expected: len,
            got: out_fast_sma.len(),
        });
    }
    if out_slow_sma.len() != len {
        return Err(VolatilityQualityIndexError::OutputLengthMismatch {
            expected: len,
            got: out_slow_sma.len(),
        });
    }
    let fast_length = validate_fast_length(input.get_fast_length())?;
    let slow_length = validate_slow_length(input.get_slow_length())?;
    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other.to_non_batch(),
    };
    let _ = chosen;
    compute_vqi_sum_series_into(open, high, low, close, out_vqi_sum);
    sma_into(out_vqi_sum, fast_length, out_fast_sma);
    sma_into(out_vqi_sum, slow_length, out_slow_sma);
    Ok(())
}

#[derive(Copy, Clone, Debug)]
pub struct VolatilityQualityIndexBatchRange {
    pub fast_length: (usize, usize, usize),
    pub slow_length: (usize, usize, usize),
}

impl Default for VolatilityQualityIndexBatchRange {
    fn default() -> Self {
        Self {
            fast_length: (9, 9, 0),
            slow_length: (200, 200, 0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VolatilityQualityIndexBatchOutput {
    pub vqi_sum: Vec<f64>,
    pub fast_sma: Vec<f64>,
    pub slow_sma: Vec<f64>,
    pub combos: Vec<VolatilityQualityIndexParams>,
    pub rows: usize,
    pub cols: usize,
}

impl VolatilityQualityIndexBatchOutput {
    pub fn row_for_params(&self, params: &VolatilityQualityIndexParams) -> Option<usize> {
        let fast_length = params.fast_length.unwrap_or(9);
        let slow_length = params.slow_length.unwrap_or(200);
        self.combos.iter().position(|combo| {
            combo.fast_length.unwrap_or(9) == fast_length
                && combo.slow_length.unwrap_or(200) == slow_length
        })
    }

    pub fn vqi_sum_for(&self, params: &VolatilityQualityIndexParams) -> Option<&[f64]> {
        self.row_for_params(params).and_then(|row| {
            let start = row * self.cols;
            self.vqi_sum.get(start..start + self.cols)
        })
    }

    pub fn fast_sma_for(&self, params: &VolatilityQualityIndexParams) -> Option<&[f64]> {
        self.row_for_params(params).and_then(|row| {
            let start = row * self.cols;
            self.fast_sma.get(start..start + self.cols)
        })
    }

    pub fn slow_sma_for(&self, params: &VolatilityQualityIndexParams) -> Option<&[f64]> {
        self.row_for_params(params).and_then(|row| {
            let start = row * self.cols;
            self.slow_sma.get(start..start + self.cols)
        })
    }

    pub fn values_for(
        &self,
        params: &VolatilityQualityIndexParams,
    ) -> Option<(&[f64], &[f64], &[f64])> {
        self.row_for_params(params).map(|row| {
            let start = row * self.cols;
            (
                &self.vqi_sum[start..start + self.cols],
                &self.fast_sma[start..start + self.cols],
                &self.slow_sma[start..start + self.cols],
            )
        })
    }
}

#[derive(Copy, Clone, Debug)]
pub struct VolatilityQualityIndexBatchBuilder {
    range: VolatilityQualityIndexBatchRange,
    kernel: Kernel,
}

impl Default for VolatilityQualityIndexBatchBuilder {
    fn default() -> Self {
        Self {
            range: VolatilityQualityIndexBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl VolatilityQualityIndexBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn fast_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.fast_length = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn fast_length_static(mut self, value: usize) -> Self {
        self.range.fast_length = (value, value, 0);
        self
    }

    #[inline(always)]
    pub fn slow_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.slow_length = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn slow_length_static(mut self, value: usize) -> Self {
        self.range.slow_length = (value, value, 0);
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
    ) -> Result<VolatilityQualityIndexBatchOutput, VolatilityQualityIndexError> {
        volatility_quality_index_batch_with_kernel(
            candles.open.as_slice(),
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
            &self.range,
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<VolatilityQualityIndexBatchOutput, VolatilityQualityIndexError> {
        self.apply(candles)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<VolatilityQualityIndexBatchOutput, VolatilityQualityIndexError> {
        volatility_quality_index_batch_with_kernel(open, high, low, close, &self.range, self.kernel)
    }
}

#[inline(always)]
fn axis_usize(
    start: usize,
    end: usize,
    step: usize,
) -> Result<Vec<usize>, VolatilityQualityIndexError> {
    if start == end {
        return Ok(vec![start]);
    }
    if step == 0 {
        return Err(VolatilityQualityIndexError::InvalidRange {
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
            match x.checked_add(step) {
                Some(next) => x = next,
                None => break,
            }
        }
    } else {
        let mut x = start;
        while x >= end {
            out.push(x);
            match x.checked_sub(step) {
                Some(next) => x = next,
                None => break,
            }
            if x > start {
                break;
            }
        }
    }
    if out.is_empty() {
        return Err(VolatilityQualityIndexError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

#[inline(always)]
pub fn expand_grid(
    range: &VolatilityQualityIndexBatchRange,
) -> Result<Vec<VolatilityQualityIndexParams>, VolatilityQualityIndexError> {
    let fast_values = axis_usize(
        range.fast_length.0,
        range.fast_length.1,
        range.fast_length.2,
    )?;
    let slow_values = axis_usize(
        range.slow_length.0,
        range.slow_length.1,
        range.slow_length.2,
    )?;
    let mut out = Vec::with_capacity(fast_values.len() * slow_values.len());
    for fast_length in fast_values {
        for &slow_length in &slow_values {
            out.push(VolatilityQualityIndexParams {
                fast_length: Some(fast_length),
                slow_length: Some(slow_length),
            });
        }
    }
    Ok(out)
}

pub fn volatility_quality_index_batch_with_kernel(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &VolatilityQualityIndexBatchRange,
    kernel: Kernel,
) -> Result<VolatilityQualityIndexBatchOutput, VolatilityQualityIndexError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(VolatilityQualityIndexError::InvalidKernelForBatch(kernel)),
    };
    volatility_quality_index_batch_par_slice(
        open,
        high,
        low,
        close,
        sweep,
        batch_kernel.to_non_batch(),
    )
}

#[inline(always)]
pub fn volatility_quality_index_batch_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &VolatilityQualityIndexBatchRange,
    kernel: Kernel,
) -> Result<VolatilityQualityIndexBatchOutput, VolatilityQualityIndexError> {
    volatility_quality_index_batch_inner(open, high, low, close, sweep, kernel, false)
}

#[inline(always)]
pub fn volatility_quality_index_batch_par_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &VolatilityQualityIndexBatchRange,
    kernel: Kernel,
) -> Result<VolatilityQualityIndexBatchOutput, VolatilityQualityIndexError> {
    volatility_quality_index_batch_inner(open, high, low, close, sweep, kernel, true)
}

fn volatility_quality_index_batch_inner(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &VolatilityQualityIndexBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<VolatilityQualityIndexBatchOutput, VolatilityQualityIndexError> {
    let combos = expand_grid(sweep)?;
    if open.is_empty() || high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(VolatilityQualityIndexError::EmptyInputData);
    }
    if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
        return Err(VolatilityQualityIndexError::InconsistentSliceLengths {
            open_len: open.len(),
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }
    if !open
        .iter()
        .zip(high.iter())
        .zip(low.iter())
        .zip(close.iter())
        .any(|(((o, h), l), c)| o.is_finite() || h.is_finite() || l.is_finite() || c.is_finite())
    {
        return Err(VolatilityQualityIndexError::AllValuesNaN);
    }
    let rows = combos.len();
    let cols = close.len();
    let mut vqi_sum = vec![0.0; rows * cols];
    let mut fast_sma = vec![f64::NAN; rows * cols];
    let mut slow_sma = vec![f64::NAN; rows * cols];

    volatility_quality_index_batch_inner_into(
        open,
        high,
        low,
        close,
        sweep,
        kernel,
        parallel,
        &mut vqi_sum,
        &mut fast_sma,
        &mut slow_sma,
    )?;

    Ok(VolatilityQualityIndexBatchOutput {
        vqi_sum,
        fast_sma,
        slow_sma,
        combos,
        rows,
        cols,
    })
}

pub fn volatility_quality_index_batch_into_slice(
    out_vqi_sum: &mut [f64],
    out_fast_sma: &mut [f64],
    out_slow_sma: &mut [f64],
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &VolatilityQualityIndexBatchRange,
    kernel: Kernel,
) -> Result<(), VolatilityQualityIndexError> {
    volatility_quality_index_batch_inner_into(
        open,
        high,
        low,
        close,
        sweep,
        kernel,
        false,
        out_vqi_sum,
        out_fast_sma,
        out_slow_sma,
    )?;
    Ok(())
}

fn volatility_quality_index_batch_inner_into(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &VolatilityQualityIndexBatchRange,
    kernel: Kernel,
    parallel: bool,
    out_vqi_sum: &mut [f64],
    out_fast_sma: &mut [f64],
    out_slow_sma: &mut [f64],
) -> Result<Vec<VolatilityQualityIndexParams>, VolatilityQualityIndexError> {
    let combos = expand_grid(sweep)?;
    if open.is_empty() || high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(VolatilityQualityIndexError::EmptyInputData);
    }
    if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
        return Err(VolatilityQualityIndexError::InconsistentSliceLengths {
            open_len: open.len(),
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }
    let rows = combos.len();
    let cols = close.len();
    let expected =
        rows.checked_mul(cols)
            .ok_or_else(|| VolatilityQualityIndexError::InvalidRange {
                start: rows.to_string(),
                end: cols.to_string(),
                step: "rows*cols".to_string(),
            })?;
    if out_vqi_sum.len() != expected {
        return Err(VolatilityQualityIndexError::OutputLengthMismatch {
            expected,
            got: out_vqi_sum.len(),
        });
    }
    if out_fast_sma.len() != expected {
        return Err(VolatilityQualityIndexError::OutputLengthMismatch {
            expected,
            got: out_fast_sma.len(),
        });
    }
    if out_slow_sma.len() != expected {
        return Err(VolatilityQualityIndexError::OutputLengthMismatch {
            expected,
            got: out_slow_sma.len(),
        });
    }
    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other.to_non_batch(),
    };
    let _ = chosen;

    let vqi_sum = compute_vqi_sum_series(open, high, low, close);
    let fast_lengths: Vec<usize> = combos
        .iter()
        .map(|combo| validate_fast_length(combo.fast_length.unwrap_or(9)))
        .collect::<Result<_, _>>()?;
    let slow_lengths: Vec<usize> = combos
        .iter()
        .map(|combo| validate_slow_length(combo.slow_length.unwrap_or(200)))
        .collect::<Result<_, _>>()?;

    let do_row = |row: usize,
                  dst_vqi_sum: &mut [f64],
                  dst_fast_sma: &mut [f64],
                  dst_slow_sma: &mut [f64]| {
        dst_vqi_sum.copy_from_slice(&vqi_sum);
        sma_into(&vqi_sum, fast_lengths[row], dst_fast_sma);
        sma_into(&vqi_sum, slow_lengths[row], dst_slow_sma);
        Ok::<(), VolatilityQualityIndexError>(())
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_vqi_sum
                .par_chunks_mut(cols)
                .zip(out_fast_sma.par_chunks_mut(cols))
                .zip(out_slow_sma.par_chunks_mut(cols))
                .enumerate()
                .try_for_each(|(row, ((dst_vqi_sum, dst_fast_sma), dst_slow_sma))| {
                    do_row(row, dst_vqi_sum, dst_fast_sma, dst_slow_sma)
                })?;
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, ((dst_vqi_sum, dst_fast_sma), dst_slow_sma)) in out_vqi_sum
                .chunks_mut(cols)
                .zip(out_fast_sma.chunks_mut(cols))
                .zip(out_slow_sma.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, dst_vqi_sum, dst_fast_sma, dst_slow_sma)?;
            }
        }
    } else {
        for (row, ((dst_vqi_sum, dst_fast_sma), dst_slow_sma)) in out_vqi_sum
            .chunks_mut(cols)
            .zip(out_fast_sma.chunks_mut(cols))
            .zip(out_slow_sma.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, dst_vqi_sum, dst_fast_sma, dst_slow_sma)?;
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "volatility_quality_index")]
#[pyo3(signature = (open, high, low, close, fast_length=9, slow_length=200, kernel=None))]
pub fn volatility_quality_index_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    fast_length: usize,
    slow_length: usize,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let o = open.as_slice()?;
    let h = high.as_slice()?;
    let l = low.as_slice()?;
    let c = close.as_slice()?;
    if o.len() != h.len() || o.len() != l.len() || o.len() != c.len() {
        return Err(PyValueError::new_err("OHLC slice length mismatch"));
    }
    let kern = validate_kernel(kernel, false)?;
    let input = VolatilityQualityIndexInput::from_slices(
        o,
        h,
        l,
        c,
        VolatilityQualityIndexParams {
            fast_length: Some(fast_length),
            slow_length: Some(slow_length),
        },
    );
    let out = py
        .allow_threads(|| volatility_quality_index_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        out.vqi_sum.into_pyarray(py),
        out.fast_sma.into_pyarray(py),
        out.slow_sma.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyclass(name = "VolatilityQualityIndexStream")]
pub struct VolatilityQualityIndexStreamPy {
    stream: VolatilityQualityIndexStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl VolatilityQualityIndexStreamPy {
    #[new]
    #[pyo3(signature = (fast_length=9, slow_length=200))]
    fn new(fast_length: usize, slow_length: usize) -> PyResult<Self> {
        let stream = VolatilityQualityIndexStream::try_new(VolatilityQualityIndexParams {
            fast_length: Some(fast_length),
            slow_length: Some(slow_length),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, open: f64, high: f64, low: f64, close: f64) -> (f64, f64, f64) {
        self.stream.update(open, high, low, close)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "volatility_quality_index_batch")]
#[pyo3(signature = (open, high, low, close, fast_length_range=(9,9,0), slow_length_range=(200,200,0), kernel=None))]
pub fn volatility_quality_index_batch_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    fast_length_range: (usize, usize, usize),
    slow_length_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let o = open.as_slice()?;
    let h = high.as_slice()?;
    let l = low.as_slice()?;
    let c = close.as_slice()?;
    if o.len() != h.len() || o.len() != l.len() || o.len() != c.len() {
        return Err(PyValueError::new_err("OHLC slice length mismatch"));
    }
    let sweep = VolatilityQualityIndexBatchRange {
        fast_length: fast_length_range,
        slow_length: slow_length_range,
    };
    let combos = expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = c.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let vqi_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let fast_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let slow_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let vqi_out = unsafe { vqi_arr.as_slice_mut()? };
    let fast_out = unsafe { fast_arr.as_slice_mut()? };
    let slow_out = unsafe { slow_arr.as_slice_mut()? };

    let kern = validate_kernel(kernel, true)?;
    py.allow_threads(|| {
        let batch = match kern {
            Kernel::Auto => detect_best_batch_kernel(),
            other => other,
        };
        volatility_quality_index_batch_inner_into(
            o,
            h,
            l,
            c,
            &sweep,
            batch.to_non_batch(),
            true,
            vqi_out,
            fast_out,
            slow_out,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("vqi_sum", vqi_arr.reshape((rows, cols))?)?;
    dict.set_item("fast_sma", fast_arr.reshape((rows, cols))?)?;
    dict.set_item("slow_sma", slow_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "fast_lengths",
        combos
            .iter()
            .map(|p| p.fast_length.unwrap_or(9) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "slow_lengths",
        combos
            .iter()
            .map(|p| p.slow_length.unwrap_or(200) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_volatility_quality_index_module(
    m: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(volatility_quality_index_py, m)?)?;
    m.add_function(wrap_pyfunction!(volatility_quality_index_batch_py, m)?)?;
    m.add_class::<VolatilityQualityIndexStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "volatility_quality_index_js")]
pub fn volatility_quality_index_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    fast_length: usize,
    slow_length: usize,
) -> Result<JsValue, JsValue> {
    if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
        return Err(JsValue::from_str("OHLC slice length mismatch"));
    }
    let input = VolatilityQualityIndexInput::from_slices(
        open,
        high,
        low,
        close,
        VolatilityQualityIndexParams {
            fast_length: Some(fast_length),
            slow_length: Some(slow_length),
        },
    );
    let out = volatility_quality_index_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("vqi_sum"),
        &serde_wasm_bindgen::to_value(&out.vqi_sum).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("fast_sma"),
        &serde_wasm_bindgen::to_value(&out.fast_sma).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("slow_sma"),
        &serde_wasm_bindgen::to_value(&out.slow_sma).unwrap(),
    )?;
    Ok(obj.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct VolatilityQualityIndexBatchConfig {
    pub fast_length_range: Vec<usize>,
    pub slow_length_range: Vec<usize>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct VolatilityQualityIndexBatchJsOutput {
    pub vqi_sum: Vec<f64>,
    pub fast_sma: Vec<f64>,
    pub slow_sma: Vec<f64>,
    pub combos: Vec<VolatilityQualityIndexParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "volatility_quality_index_batch_js")]
pub fn volatility_quality_index_batch_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
        return Err(JsValue::from_str("OHLC slice length mismatch"));
    }
    let config: VolatilityQualityIndexBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.fast_length_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: fast_length_range must have exactly 3 elements [start, end, step]",
        ));
    }
    if config.slow_length_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: slow_length_range must have exactly 3 elements [start, end, step]",
        ));
    }
    let out = volatility_quality_index_batch_with_kernel(
        open,
        high,
        low,
        close,
        &VolatilityQualityIndexBatchRange {
            fast_length: (
                config.fast_length_range[0],
                config.fast_length_range[1],
                config.fast_length_range[2],
            ),
            slow_length: (
                config.slow_length_range[0],
                config.slow_length_range[1],
                config.slow_length_range[2],
            ),
        },
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&VolatilityQualityIndexBatchJsOutput {
        vqi_sum: out.vqi_sum,
        fast_sma: out.fast_sma,
        slow_sma: out.slow_sma,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn volatility_quality_index_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn volatility_quality_index_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn volatility_quality_index_into(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    fast_length: usize,
    slow_length: usize,
) -> Result<(), JsValue> {
    if open_ptr.is_null()
        || high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to volatility_quality_index_into",
        ));
    }
    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, 3 * len);
        let (out_vqi_sum, rest) = out.split_at_mut(len);
        let (out_fast_sma, out_slow_sma) = rest.split_at_mut(len);
        let input = VolatilityQualityIndexInput::from_slices(
            open,
            high,
            low,
            close,
            VolatilityQualityIndexParams {
                fast_length: Some(fast_length),
                slow_length: Some(slow_length),
            },
        );
        volatility_quality_index_into_slice(
            out_vqi_sum,
            out_fast_sma,
            out_slow_sma,
            &input,
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn volatility_quality_index_batch_into(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    vqi_sum_ptr: *mut f64,
    fast_sma_ptr: *mut f64,
    slow_sma_ptr: *mut f64,
    len: usize,
    fast_start: usize,
    fast_end: usize,
    fast_step: usize,
    slow_start: usize,
    slow_end: usize,
    slow_step: usize,
) -> Result<usize, JsValue> {
    if open_ptr.is_null()
        || high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || vqi_sum_ptr.is_null()
        || fast_sma_ptr.is_null()
        || slow_sma_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to volatility_quality_index_batch_into",
        ));
    }
    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let sweep = VolatilityQualityIndexBatchRange {
            fast_length: (fast_start, fast_end, fast_step),
            slow_length: (slow_start, slow_end, slow_step),
        };
        let combos = expand_grid(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let total = rows
            .checked_mul(len)
            .ok_or_else(|| JsValue::from_str("rows*len overflow"))?;
        let out_vqi_sum = std::slice::from_raw_parts_mut(vqi_sum_ptr, total);
        let out_fast_sma = std::slice::from_raw_parts_mut(fast_sma_ptr, total);
        let out_slow_sma = std::slice::from_raw_parts_mut(slow_sma_ptr, total);
        volatility_quality_index_batch_inner_into(
            open,
            high,
            low,
            close,
            &sweep,
            Kernel::Scalar,
            false,
            out_vqi_sum,
            out_fast_sma,
            out_slow_sma,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn volatility_quality_index_output_into_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    fast_length: usize,
    slow_length: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = volatility_quality_index_js(open, high, low, close, fast_length, slow_length)?;
    crate::write_wasm_object_f64_outputs("volatility_quality_index_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn volatility_quality_index_batch_output_into_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = volatility_quality_index_batch_js(open, high, low, close, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "volatility_quality_index_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_ohlc(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut open = Vec::with_capacity(n);
        let mut high = Vec::with_capacity(n);
        let mut low = Vec::with_capacity(n);
        let mut close = Vec::with_capacity(n);
        let mut price = 100.0;
        for i in 0..n {
            let drift = 0.25 + (i as f64) * 0.005;
            let o = price;
            let c = price + drift;
            let h = o.max(c) + 0.4;
            let l = o.min(c) - 0.3;
            open.push(o);
            high.push(h);
            low.push(l);
            close.push(c);
            price = c;
        }
        (open, high, low, close)
    }

    fn manual_vqi(
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        fast_length: usize,
        slow_length: usize,
    ) -> VolatilityQualityIndexOutput {
        let vqi_sum = compute_vqi_sum_series(open, high, low, close);
        let mut fast_sma =
            alloc_with_nan_prefix(close.len(), fast_length.saturating_sub(1).min(close.len()));
        let mut slow_sma =
            alloc_with_nan_prefix(close.len(), slow_length.saturating_sub(1).min(close.len()));
        sma_into(&vqi_sum, fast_length, &mut fast_sma);
        sma_into(&vqi_sum, slow_length, &mut slow_sma);
        VolatilityQualityIndexOutput {
            vqi_sum,
            fast_sma,
            slow_sma,
        }
    }

    fn assert_close(lhs: &[f64], rhs: &[f64], eps: f64) {
        assert_eq!(lhs.len(), rhs.len());
        for i in 0..lhs.len() {
            let a = lhs[i];
            let b = rhs[i];
            assert!(
                (a.is_nan() && b.is_nan()) || (a - b).abs() <= eps,
                "mismatch at {i}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn volatility_quality_index_matches_manual_reference() {
        let (open, high, low, close) = sample_ohlc(256);
        let input = VolatilityQualityIndexInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            VolatilityQualityIndexParams {
                fast_length: Some(9),
                slow_length: Some(21),
            },
        );
        let out = volatility_quality_index(&input).unwrap();
        let manual = manual_vqi(&open, &high, &low, &close, 9, 21);
        assert_close(&out.vqi_sum, &manual.vqi_sum, 1e-12);
        assert_close(&out.fast_sma, &manual.fast_sma, 1e-12);
        assert_close(&out.slow_sma, &manual.slow_sma, 1e-12);
    }

    #[test]
    fn volatility_quality_index_stream_matches_batch() {
        let (open, high, low, close) = sample_ohlc(128);
        let input = VolatilityQualityIndexInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            VolatilityQualityIndexParams {
                fast_length: Some(9),
                slow_length: Some(21),
            },
        );
        let batch = volatility_quality_index(&input).unwrap();
        let mut stream = VolatilityQualityIndexStream::try_new(input.params.clone()).unwrap();
        let mut vqi_sum = Vec::with_capacity(close.len());
        let mut fast_sma = Vec::with_capacity(close.len());
        let mut slow_sma = Vec::with_capacity(close.len());
        for i in 0..close.len() {
            let (vqi, fast, slow) = stream.update(open[i], high[i], low[i], close[i]);
            vqi_sum.push(vqi);
            fast_sma.push(fast);
            slow_sma.push(slow);
        }
        assert_close(&vqi_sum, &batch.vqi_sum, 1e-12);
        assert_close(&fast_sma, &batch.fast_sma, 1e-12);
        assert_close(&slow_sma, &batch.slow_sma, 1e-12);
    }

    #[test]
    fn volatility_quality_index_batch_rows_match_single() {
        let (open, high, low, close) = sample_ohlc(96);
        let sweep = VolatilityQualityIndexBatchRange {
            fast_length: (9, 11, 2),
            slow_length: (20, 24, 4),
        };
        let batch = volatility_quality_index_batch_with_kernel(
            &open,
            &high,
            &low,
            &close,
            &sweep,
            Kernel::Auto,
        )
        .unwrap();
        assert_eq!(batch.rows, 4);
        assert_eq!(batch.cols, close.len());
        let first = VolatilityQualityIndexInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            batch.combos[0].clone(),
        );
        let direct = volatility_quality_index(&first).unwrap();
        assert_close(&batch.vqi_sum[..close.len()], &direct.vqi_sum, 1e-12);
        assert_close(&batch.fast_sma[..close.len()], &direct.fast_sma, 1e-12);
        assert_close(&batch.slow_sma[..close.len()], &direct.slow_sma, 1e-12);
    }

    #[test]
    fn volatility_quality_index_into_slice_matches_single() {
        let (open, high, low, close) = sample_ohlc(64);
        let input = VolatilityQualityIndexInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            VolatilityQualityIndexParams {
                fast_length: Some(9),
                slow_length: Some(21),
            },
        );
        let direct = volatility_quality_index(&input).unwrap();
        let mut vqi_sum = vec![0.0; close.len()];
        let mut fast_sma = alloc_with_nan_prefix(close.len(), 8);
        let mut slow_sma = alloc_with_nan_prefix(close.len(), 20);
        volatility_quality_index_into_slice(
            &mut vqi_sum,
            &mut fast_sma,
            &mut slow_sma,
            &input,
            Kernel::Auto,
        )
        .unwrap();
        assert_close(&vqi_sum, &direct.vqi_sum, 1e-12);
        assert_close(&fast_sma, &direct.fast_sma, 1e-12);
        assert_close(&slow_sma, &direct.slow_sma, 1e-12);
    }

    #[test]
    fn volatility_quality_index_invalid_lengths_error() {
        let (open, high, low, close) = sample_ohlc(32);
        let input = VolatilityQualityIndexInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            VolatilityQualityIndexParams {
                fast_length: Some(0),
                slow_length: Some(21),
            },
        );
        assert!(matches!(
            volatility_quality_index(&input),
            Err(VolatilityQualityIndexError::InvalidFastLength { .. })
        ));
    }

    #[test]
    fn volatility_quality_index_matches_hand_values() {
        let open = [10.0, 11.0, 12.0, 12.0];
        let high = [12.0, 13.0, 12.0, 15.0];
        let low = [9.0, 10.0, 12.0, 11.0];
        let close = [11.0, 12.0, 12.0, 14.0];
        let input = VolatilityQualityIndexInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            VolatilityQualityIndexParams {
                fast_length: Some(2),
                slow_length: Some(3),
            },
        );
        let out = volatility_quality_index(&input).unwrap();
        let expected_sum = [0.0, 1.0 / 3.0, 1.0 / 3.0, 4.0 / 3.0];
        let expected_fast = [f64::NAN, 1.0 / 6.0, 1.0 / 3.0, 5.0 / 6.0];
        let expected_slow = [f64::NAN, f64::NAN, 2.0 / 9.0, 2.0 / 3.0];
        assert_close(&out.vqi_sum, &expected_sum, 1e-12);
        assert_close(&out.fast_sma, &expected_fast, 1e-12);
        assert_close(&out.slow_sma, &expected_slow, 1e-12);
    }
}
