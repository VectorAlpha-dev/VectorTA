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

use crate::utilities::data_loader::Candles;
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::mem::ManuallyDrop;
use thiserror::Error;

const DEFAULT_THRESHOLD_LEVEL: f64 = 0.35;
const DEFAULT_EMA_PERIOD: usize = 14;

#[derive(Debug, Clone)]
pub enum DailyFactorData<'a> {
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
pub struct DailyFactorOutput {
    pub value: Vec<f64>,
    pub ema: Vec<f64>,
    pub signal: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DailyFactorOutputField {
    Value,
    Ema,
    Signal,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct DailyFactorParams {
    pub threshold_level: Option<f64>,
}

impl Default for DailyFactorParams {
    fn default() -> Self {
        Self {
            threshold_level: Some(DEFAULT_THRESHOLD_LEVEL),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DailyFactorInput<'a> {
    pub data: DailyFactorData<'a>,
    pub params: DailyFactorParams,
}

impl<'a> DailyFactorInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: DailyFactorParams) -> Self {
        Self {
            data: DailyFactorData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: DailyFactorParams,
    ) -> Self {
        Self {
            data: DailyFactorData::Slices {
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
        Self::from_candles(candles, DailyFactorParams::default())
    }
}

#[derive(Copy, Clone, Debug)]
pub struct DailyFactorBuilder {
    threshold_level: Option<f64>,
    kernel: Kernel,
}

impl Default for DailyFactorBuilder {
    fn default() -> Self {
        Self {
            threshold_level: None,
            kernel: Kernel::Auto,
        }
    }
}

impl DailyFactorBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn threshold_level(mut self, value: f64) -> Self {
        self.threshold_level = Some(value);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline(always)]
    pub fn apply(self, candles: &Candles) -> Result<DailyFactorOutput, DailyFactorError> {
        let input = DailyFactorInput::from_candles(
            candles,
            DailyFactorParams {
                threshold_level: self.threshold_level,
            },
        );
        daily_factor_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<DailyFactorOutput, DailyFactorError> {
        let input = DailyFactorInput::from_slices(
            open,
            high,
            low,
            close,
            DailyFactorParams {
                threshold_level: self.threshold_level,
            },
        );
        daily_factor_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<DailyFactorStream, DailyFactorError> {
        DailyFactorStream::try_new(DailyFactorParams {
            threshold_level: self.threshold_level,
        })
    }
}

#[derive(Debug, Error)]
pub enum DailyFactorError {
    #[error("daily_factor: Input data slice is empty.")]
    EmptyInputData,
    #[error("daily_factor: All values are NaN.")]
    AllValuesNaN,
    #[error("daily_factor: Inconsistent slice lengths: open={open_len}, high={high_len}, low={low_len}, close={close_len}")]
    InconsistentSliceLengths {
        open_len: usize,
        high_len: usize,
        low_len: usize,
        close_len: usize,
    },
    #[error("daily_factor: Invalid threshold_level: {threshold_level}")]
    InvalidThresholdLevel { threshold_level: f64 },
    #[error("daily_factor: Output length mismatch: expected={expected}, got={got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("daily_factor: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("daily_factor: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Clone, Copy, Debug)]
struct ResolvedParams {
    threshold_level: f64,
}

#[inline(always)]
fn ema_alpha() -> f64 {
    2.0 / (DEFAULT_EMA_PERIOD as f64 + 1.0)
}

#[inline(always)]
fn extract_ohlc<'a>(
    input: &'a DailyFactorInput<'a>,
) -> Result<(&'a [f64], &'a [f64], &'a [f64], &'a [f64]), DailyFactorError> {
    let (open, high, low, close) = match &input.data {
        DailyFactorData::Candles { candles } => (
            candles.open.as_slice(),
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
        ),
        DailyFactorData::Slices {
            open,
            high,
            low,
            close,
        } => (*open, *high, *low, *close),
    };
    if open.is_empty() || high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(DailyFactorError::EmptyInputData);
    }
    if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
        return Err(DailyFactorError::InconsistentSliceLengths {
            open_len: open.len(),
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }
    Ok((open, high, low, close))
}

#[inline(always)]
fn first_valid_ohlc(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Option<usize> {
    (0..close.len()).find(|&i| {
        open[i].is_finite() && high[i].is_finite() && low[i].is_finite() && close[i].is_finite()
    })
}

#[inline(always)]
fn resolve_params(params: &DailyFactorParams) -> Result<ResolvedParams, DailyFactorError> {
    let threshold_level = params.threshold_level.unwrap_or(DEFAULT_THRESHOLD_LEVEL);
    if !threshold_level.is_finite() || !(0.0..=1.0).contains(&threshold_level) {
        return Err(DailyFactorError::InvalidThresholdLevel { threshold_level });
    }
    Ok(ResolvedParams { threshold_level })
}

#[inline(always)]
fn validate_input<'a>(
    input: &'a DailyFactorInput<'a>,
    kernel: Kernel,
) -> Result<
    (
        &'a [f64],
        &'a [f64],
        &'a [f64],
        &'a [f64],
        ResolvedParams,
        usize,
        Kernel,
    ),
    DailyFactorError,
> {
    let (open, high, low, close) = extract_ohlc(input)?;
    let params = resolve_params(&input.params)?;
    let first = first_valid_ohlc(open, high, low, close).ok_or(DailyFactorError::AllValuesNaN)?;
    Ok((open, high, low, close, params, first, kernel.to_non_batch()))
}

#[inline(always)]
fn compute_signal(value: f64, ema: f64, close: f64, threshold_level: f64) -> f64 {
    if !(value.is_finite() && ema.is_finite() && close.is_finite()) {
        return f64::NAN;
    }
    if value > threshold_level && close > ema {
        2.0
    } else if value > threshold_level && close < ema {
        -2.0
    } else if close > ema {
        1.0
    } else if close < ema {
        -1.0
    } else {
        0.0
    }
}

#[inline(always)]
fn compute_base_into(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    out_value: &mut [f64],
    out_ema: &mut [f64],
) -> Result<(), DailyFactorError> {
    let len = close.len();
    if out_value.len() != len {
        return Err(DailyFactorError::OutputLengthMismatch {
            expected: len,
            got: out_value.len(),
        });
    }
    if out_ema.len() != len {
        return Err(DailyFactorError::OutputLengthMismatch {
            expected: len,
            got: out_ema.len(),
        });
    }

    let alpha = ema_alpha();
    let mut prev_open = f64::NAN;
    let mut prev_high = f64::NAN;
    let mut prev_low = f64::NAN;
    let mut prev_close = f64::NAN;
    let mut prev_ema = f64::NAN;
    let mut has_prev = false;

    for i in first..len {
        let o = open[i];
        let h = high[i];
        let l = low[i];
        let c = close[i];
        if !(o.is_finite() && h.is_finite() && l.is_finite() && c.is_finite()) {
            out_value[i] = f64::NAN;
            out_ema[i] = f64::NAN;
            continue;
        }

        let ema = if prev_ema.is_finite() {
            prev_ema + alpha * (c - prev_ema)
        } else {
            c
        };
        let value = if has_prev {
            let range = prev_high - prev_low;
            if range.is_finite() && range != 0.0 {
                (prev_open - prev_close).abs() / range
            } else {
                0.0
            }
        } else {
            0.0
        };

        out_value[i] = value;
        out_ema[i] = ema;
        prev_open = o;
        prev_high = h;
        prev_low = l;
        prev_close = c;
        prev_ema = ema;
        has_prev = true;
    }

    Ok(())
}

#[inline(always)]
fn compute_signal_into(
    value: &[f64],
    ema: &[f64],
    close: &[f64],
    threshold_level: f64,
    out_signal: &mut [f64],
) -> Result<(), DailyFactorError> {
    let len = close.len();
    if value.len() != len || ema.len() != len {
        return Err(DailyFactorError::OutputLengthMismatch {
            expected: len,
            got: value.len().min(ema.len()),
        });
    }
    if out_signal.len() != len {
        return Err(DailyFactorError::OutputLengthMismatch {
            expected: len,
            got: out_signal.len(),
        });
    }
    for i in 0..len {
        out_signal[i] = compute_signal(value[i], ema[i], close[i], threshold_level);
    }
    Ok(())
}

#[inline]
pub fn daily_factor(input: &DailyFactorInput) -> Result<DailyFactorOutput, DailyFactorError> {
    daily_factor_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn daily_factor_with_kernel(
    input: &DailyFactorInput,
    kernel: Kernel,
) -> Result<DailyFactorOutput, DailyFactorError> {
    let (open, high, low, close, params, first, _kernel) = validate_input(input, kernel)?;
    let mut value = alloc_with_nan_prefix(close.len(), first);
    let mut ema = alloc_with_nan_prefix(close.len(), first);
    let mut signal = alloc_with_nan_prefix(close.len(), first);
    compute_base_into(open, high, low, close, first, &mut value, &mut ema)?;
    compute_signal_into(&value, &ema, close, params.threshold_level, &mut signal)?;
    Ok(DailyFactorOutput { value, ema, signal })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn daily_factor_into(
    out_value: &mut [f64],
    out_ema: &mut [f64],
    out_signal: &mut [f64],
    input: &DailyFactorInput,
    kernel: Kernel,
) -> Result<(), DailyFactorError> {
    daily_factor_into_slice(out_value, out_ema, out_signal, input, kernel)
}

#[inline]
pub fn daily_factor_into_slice(
    out_value: &mut [f64],
    out_ema: &mut [f64],
    out_signal: &mut [f64],
    input: &DailyFactorInput,
    kernel: Kernel,
) -> Result<(), DailyFactorError> {
    let (open, high, low, close, params, first, _kernel) = validate_input(input, kernel)?;
    out_value.fill(f64::NAN);
    out_ema.fill(f64::NAN);
    out_signal.fill(f64::NAN);
    compute_base_into(open, high, low, close, first, out_value, out_ema)?;
    compute_signal_into(
        out_value,
        out_ema,
        close,
        params.threshold_level,
        out_signal,
    )
}

#[inline]
pub fn daily_factor_output_into_slice(
    dst: &mut [f64],
    input: &DailyFactorInput,
    kernel: Kernel,
    field: DailyFactorOutputField,
) -> Result<(), DailyFactorError> {
    let (open, high, low, close, params, first, _kernel) = validate_input(input, kernel)?;
    if dst.len() != close.len() {
        return Err(DailyFactorError::OutputLengthMismatch {
            expected: close.len(),
            got: dst.len(),
        });
    }
    dst.fill(f64::NAN);

    let alpha = ema_alpha();
    let mut prev_open = f64::NAN;
    let mut prev_high = f64::NAN;
    let mut prev_low = f64::NAN;
    let mut prev_close = f64::NAN;
    let mut prev_ema = f64::NAN;
    let mut has_prev = false;

    for i in first..close.len() {
        let o = open[i];
        let h = high[i];
        let l = low[i];
        let c = close[i];
        if !(o.is_finite() && h.is_finite() && l.is_finite() && c.is_finite()) {
            continue;
        }

        let ema = if prev_ema.is_finite() {
            prev_ema + alpha * (c - prev_ema)
        } else {
            c
        };
        let value = if has_prev {
            let range = prev_high - prev_low;
            if range.is_finite() && range != 0.0 {
                (prev_open - prev_close).abs() / range
            } else {
                0.0
            }
        } else {
            0.0
        };

        dst[i] = match field {
            DailyFactorOutputField::Value => value,
            DailyFactorOutputField::Ema => ema,
            DailyFactorOutputField::Signal => compute_signal(value, ema, c, params.threshold_level),
        };

        prev_open = o;
        prev_high = h;
        prev_low = l;
        prev_close = c;
        prev_ema = ema;
        has_prev = true;
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct DailyFactorStream {
    params: ResolvedParams,
    prev_open: f64,
    prev_high: f64,
    prev_low: f64,
    prev_close: f64,
    prev_ema: f64,
    has_prev: bool,
}

impl DailyFactorStream {
    pub fn try_new(params: DailyFactorParams) -> Result<Self, DailyFactorError> {
        Ok(Self {
            params: resolve_params(&params)?,
            prev_open: f64::NAN,
            prev_high: f64::NAN,
            prev_low: f64::NAN,
            prev_close: f64::NAN,
            prev_ema: f64::NAN,
            has_prev: false,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, open: f64, high: f64, low: f64, close: f64) -> (f64, f64, f64) {
        if !(open.is_finite() && high.is_finite() && low.is_finite() && close.is_finite()) {
            return (f64::NAN, f64::NAN, f64::NAN);
        }

        let ema = if self.prev_ema.is_finite() {
            self.prev_ema + ema_alpha() * (close - self.prev_ema)
        } else {
            close
        };
        let value = if self.has_prev {
            let range = self.prev_high - self.prev_low;
            if range.is_finite() && range != 0.0 {
                (self.prev_open - self.prev_close).abs() / range
            } else {
                0.0
            }
        } else {
            0.0
        };
        let signal = compute_signal(value, ema, close, self.params.threshold_level);

        self.prev_open = open;
        self.prev_high = high;
        self.prev_low = low;
        self.prev_close = close;
        self.prev_ema = ema;
        self.has_prev = true;

        (value, ema, signal)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DailyFactorBatchRange {
    pub threshold_level: (f64, f64, f64),
}

#[derive(Clone, Debug)]
pub struct DailyFactorBatchOutput {
    pub value: Vec<f64>,
    pub ema: Vec<f64>,
    pub signal: Vec<f64>,
    pub combos: Vec<DailyFactorParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Copy, Clone, Debug)]
pub struct DailyFactorBatchBuilder {
    threshold_level: (f64, f64, f64),
    kernel: Kernel,
}

impl Default for DailyFactorBatchBuilder {
    fn default() -> Self {
        Self {
            threshold_level: (DEFAULT_THRESHOLD_LEVEL, DEFAULT_THRESHOLD_LEVEL, 0.0),
            kernel: Kernel::Auto,
        }
    }
}

impl DailyFactorBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn threshold_level_range(mut self, value: (f64, f64, f64)) -> Self {
        self.threshold_level = value;
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline(always)]
    pub fn apply(self, candles: &Candles) -> Result<DailyFactorBatchOutput, DailyFactorError> {
        daily_factor_batch_with_kernel(
            candles.open.as_slice(),
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
            &DailyFactorBatchRange {
                threshold_level: self.threshold_level,
            },
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<DailyFactorBatchOutput, DailyFactorError> {
        daily_factor_batch_with_kernel(
            open,
            high,
            low,
            close,
            &DailyFactorBatchRange {
                threshold_level: self.threshold_level,
            },
            self.kernel,
        )
    }
}

#[inline(always)]
fn expand_float_range(start: f64, end: f64, step: f64) -> Result<Vec<f64>, DailyFactorError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(DailyFactorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if step == 0.0 {
        if (start - end).abs() > 1e-12 {
            return Err(DailyFactorError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        return Ok(vec![start]);
    }
    if start > end || step < 0.0 {
        return Err(DailyFactorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    let mut out = Vec::new();
    let mut current = start;
    while current <= end + 1e-12 {
        out.push(current);
        if out.len() > 1_000_000 {
            return Err(DailyFactorError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        current += step;
    }
    Ok(out)
}

pub fn expand_grid(
    sweep: &DailyFactorBatchRange,
) -> Result<Vec<DailyFactorParams>, DailyFactorError> {
    let levels = expand_float_range(
        sweep.threshold_level.0,
        sweep.threshold_level.1,
        sweep.threshold_level.2,
    )?;
    let mut out = Vec::with_capacity(levels.len());
    for threshold_level in levels {
        out.push(DailyFactorParams {
            threshold_level: Some(threshold_level),
        });
    }
    Ok(out)
}

#[inline(always)]
fn validate_raw_slices(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> Result<usize, DailyFactorError> {
    if open.is_empty() || high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(DailyFactorError::EmptyInputData);
    }
    if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
        return Err(DailyFactorError::InconsistentSliceLengths {
            open_len: open.len(),
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }
    first_valid_ohlc(open, high, low, close).ok_or(DailyFactorError::AllValuesNaN)
}

#[inline(always)]
fn batch_shape(rows: usize, cols: usize) -> Result<usize, DailyFactorError> {
    rows.checked_mul(cols)
        .ok_or_else(|| DailyFactorError::InvalidRange {
            start: rows.to_string(),
            end: cols.to_string(),
            step: "rows*cols".to_string(),
        })
}

pub fn daily_factor_batch_with_kernel(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &DailyFactorBatchRange,
    kernel: Kernel,
) -> Result<DailyFactorBatchOutput, DailyFactorError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(DailyFactorError::InvalidKernelForBatch(kernel)),
    };
    daily_factor_batch_par_slice(open, high, low, close, sweep, batch_kernel.to_non_batch())
}

#[inline(always)]
pub fn daily_factor_batch_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &DailyFactorBatchRange,
    kernel: Kernel,
) -> Result<DailyFactorBatchOutput, DailyFactorError> {
    daily_factor_batch_inner(open, high, low, close, sweep, kernel, false)
}

#[inline(always)]
pub fn daily_factor_batch_par_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &DailyFactorBatchRange,
    kernel: Kernel,
) -> Result<DailyFactorBatchOutput, DailyFactorError> {
    daily_factor_batch_inner(open, high, low, close, sweep, kernel, true)
}

fn daily_factor_batch_inner(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &DailyFactorBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<DailyFactorBatchOutput, DailyFactorError> {
    let combos = expand_grid(sweep)?;
    let first = validate_raw_slices(open, high, low, close)?;
    let rows = combos.len();
    let cols = close.len();
    let total = batch_shape(rows, cols)?;
    let warmups = vec![first; rows];

    let mut value_buf = make_uninit_matrix(rows, cols);
    let mut ema_buf = make_uninit_matrix(rows, cols);
    let mut signal_buf = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut value_buf, cols, &warmups);
    init_matrix_prefixes(&mut ema_buf, cols, &warmups);
    init_matrix_prefixes(&mut signal_buf, cols, &warmups);

    let mut value_guard = ManuallyDrop::new(value_buf);
    let mut ema_guard = ManuallyDrop::new(ema_buf);
    let mut signal_guard = ManuallyDrop::new(signal_buf);
    let out_value: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(value_guard.as_mut_ptr() as *mut f64, value_guard.len())
    };
    let out_ema: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(ema_guard.as_mut_ptr() as *mut f64, ema_guard.len())
    };
    let out_signal: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(signal_guard.as_mut_ptr() as *mut f64, signal_guard.len())
    };

    daily_factor_batch_inner_into(
        open, high, low, close, sweep, kernel, parallel, out_value, out_ema, out_signal,
    )?;

    let value = unsafe {
        Vec::from_raw_parts(
            value_guard.as_mut_ptr() as *mut f64,
            total,
            value_guard.capacity(),
        )
    };
    let ema = unsafe {
        Vec::from_raw_parts(
            ema_guard.as_mut_ptr() as *mut f64,
            total,
            ema_guard.capacity(),
        )
    };
    let signal = unsafe {
        Vec::from_raw_parts(
            signal_guard.as_mut_ptr() as *mut f64,
            total,
            signal_guard.capacity(),
        )
    };

    Ok(DailyFactorBatchOutput {
        value,
        ema,
        signal,
        combos,
        rows,
        cols,
    })
}

pub fn daily_factor_batch_into_slice(
    out_value: &mut [f64],
    out_ema: &mut [f64],
    out_signal: &mut [f64],
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &DailyFactorBatchRange,
    kernel: Kernel,
) -> Result<(), DailyFactorError> {
    daily_factor_batch_inner_into(
        open, high, low, close, sweep, kernel, false, out_value, out_ema, out_signal,
    )?;
    Ok(())
}

fn daily_factor_batch_inner_into(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &DailyFactorBatchRange,
    _kernel: Kernel,
    parallel: bool,
    out_value: &mut [f64],
    out_ema: &mut [f64],
    out_signal: &mut [f64],
) -> Result<Vec<DailyFactorParams>, DailyFactorError> {
    let combos = expand_grid(sweep)?;
    let first = validate_raw_slices(open, high, low, close)?;
    let rows = combos.len();
    let cols = close.len();
    let total = batch_shape(rows, cols)?;
    if out_value.len() != total {
        return Err(DailyFactorError::OutputLengthMismatch {
            expected: total,
            got: out_value.len(),
        });
    }
    if out_ema.len() != total {
        return Err(DailyFactorError::OutputLengthMismatch {
            expected: total,
            got: out_ema.len(),
        });
    }
    if out_signal.len() != total {
        return Err(DailyFactorError::OutputLengthMismatch {
            expected: total,
            got: out_signal.len(),
        });
    }

    let mut base_value = alloc_with_nan_prefix(cols, first);
    let mut base_ema = alloc_with_nan_prefix(cols, first);
    compute_base_into(
        open,
        high,
        low,
        close,
        first,
        &mut base_value,
        &mut base_ema,
    )?;
    let thresholds: Vec<f64> = combos
        .iter()
        .map(|combo| resolve_params(combo).map(|p| p.threshold_level))
        .collect::<Result<Vec<_>, _>>()?;

    let do_row =
        |row: usize, value_dst: &mut [f64], ema_dst: &mut [f64], signal_dst: &mut [f64]| {
            value_dst.copy_from_slice(&base_value);
            ema_dst.copy_from_slice(&base_ema);
            compute_signal_into(&base_value, &base_ema, close, thresholds[row], signal_dst)
        };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_value
                .par_chunks_mut(cols)
                .zip(out_ema.par_chunks_mut(cols))
                .zip(out_signal.par_chunks_mut(cols))
                .enumerate()
                .try_for_each(|(row, ((value_dst, ema_dst), signal_dst))| {
                    do_row(row, value_dst, ema_dst, signal_dst)
                })?;
        }
        #[cfg(target_arch = "wasm32")]
        {
            for row in 0..rows {
                let start = row * cols;
                let end = start + cols;
                do_row(
                    row,
                    &mut out_value[start..end],
                    &mut out_ema[start..end],
                    &mut out_signal[start..end],
                )?;
            }
        }
    } else {
        for row in 0..rows {
            let start = row * cols;
            let end = start + cols;
            do_row(
                row,
                &mut out_value[start..end],
                &mut out_ema[start..end],
                &mut out_signal[start..end],
            )?;
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "daily_factor")]
#[pyo3(signature = (open, high, low, close, threshold_level=0.35, kernel=None))]
pub fn daily_factor_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    threshold_level: f64,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let open = open.as_slice()?;
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let input = DailyFactorInput::from_slices(
        open,
        high,
        low,
        close,
        DailyFactorParams {
            threshold_level: Some(threshold_level),
        },
    );
    let kernel = validate_kernel(kernel, false)?;
    let out = py
        .allow_threads(|| daily_factor_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dict = PyDict::new(py);
    dict.set_item("value", out.value.into_pyarray(py))?;
    dict.set_item("ema", out.ema.into_pyarray(py))?;
    dict.set_item("signal", out.signal.into_pyarray(py))?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "DailyFactorStream")]
pub struct DailyFactorStreamPy {
    stream: DailyFactorStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl DailyFactorStreamPy {
    #[new]
    #[pyo3(signature = (threshold_level=0.35))]
    fn new(threshold_level: f64) -> PyResult<Self> {
        let stream = DailyFactorStream::try_new(DailyFactorParams {
            threshold_level: Some(threshold_level),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, open: f64, high: f64, low: f64, close: f64) -> (f64, f64, f64) {
        self.stream.update(open, high, low, close)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "daily_factor_batch")]
#[pyo3(signature = (open, high, low, close, threshold_level_range=(0.35,0.35,0.0), kernel=None))]
pub fn daily_factor_batch_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    threshold_level_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let open = open.as_slice()?;
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let sweep = DailyFactorBatchRange {
        threshold_level: threshold_level_range,
    };
    let combos = expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let out_value = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_ema = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_signal = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let value_slice = unsafe { out_value.as_slice_mut()? };
    let ema_slice = unsafe { out_ema.as_slice_mut()? };
    let signal_slice = unsafe { out_signal.as_slice_mut()? };
    let kernel = validate_kernel(kernel, true)?;

    py.allow_threads(|| {
        let batch_kernel = match kernel {
            Kernel::Auto => detect_best_batch_kernel(),
            other => other,
        };
        daily_factor_batch_inner_into(
            open,
            high,
            low,
            close,
            &sweep,
            batch_kernel.to_non_batch(),
            true,
            value_slice,
            ema_slice,
            signal_slice,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("value", out_value.reshape((rows, cols))?)?;
    dict.set_item("ema", out_ema.reshape((rows, cols))?)?;
    dict.set_item("signal", out_signal.reshape((rows, cols))?)?;
    dict.set_item(
        "threshold_levels",
        combos
            .iter()
            .map(|combo| combo.threshold_level.unwrap_or(DEFAULT_THRESHOLD_LEVEL))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_daily_factor_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(daily_factor_py, m)?)?;
    m.add_function(wrap_pyfunction!(daily_factor_batch_py, m)?)?;
    m.add_class::<DailyFactorStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct DailyFactorJsOutput {
    pub value: Vec<f64>,
    pub ema: Vec<f64>,
    pub signal: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "daily_factor_js")]
pub fn daily_factor_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    threshold_level: f64,
) -> Result<JsValue, JsValue> {
    let input = DailyFactorInput::from_slices(
        open,
        high,
        low,
        close,
        DailyFactorParams {
            threshold_level: Some(threshold_level),
        },
    );
    let out = daily_factor_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&DailyFactorJsOutput {
        value: out.value,
        ema: out.ema,
        signal: out.signal,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct DailyFactorBatchConfig {
    pub threshold_level_range: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct DailyFactorBatchJsOutput {
    pub value: Vec<f64>,
    pub ema: Vec<f64>,
    pub signal: Vec<f64>,
    pub threshold_levels: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
fn js_vec3_to_f64(name: &str, values: &[f64]) -> Result<(f64, f64, f64), JsValue> {
    if values.len() != 3 {
        return Err(JsValue::from_str(&format!(
            "Invalid config: {name} must have exactly 3 elements [start, end, step]"
        )));
    }
    if !values.iter().all(|v| v.is_finite()) {
        return Err(JsValue::from_str(&format!(
            "Invalid config: {name} entries must be finite numbers"
        )));
    }
    Ok((values[0], values[1], values[2]))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "daily_factor_batch_js")]
pub fn daily_factor_batch_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: DailyFactorBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let sweep = DailyFactorBatchRange {
        threshold_level: js_vec3_to_f64("threshold_level_range", &config.threshold_level_range)?,
    };
    let out = daily_factor_batch_with_kernel(open, high, low, close, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let threshold_levels = out
        .combos
        .iter()
        .map(|combo| combo.threshold_level.unwrap_or(DEFAULT_THRESHOLD_LEVEL))
        .collect();
    serde_wasm_bindgen::to_value(&DailyFactorBatchJsOutput {
        value: out.value,
        ema: out.ema,
        signal: out.signal,
        threshold_levels,
        rows: out.rows,
        cols: out.cols,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn daily_factor_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn daily_factor_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn daily_factor_into(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_value_ptr: *mut f64,
    out_ema_ptr: *mut f64,
    out_signal_ptr: *mut f64,
    len: usize,
    threshold_level: f64,
) -> Result<(), JsValue> {
    if open_ptr.is_null()
        || high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || out_value_ptr.is_null()
        || out_ema_ptr.is_null()
        || out_signal_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }
    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let out_value = std::slice::from_raw_parts_mut(out_value_ptr, len);
        let out_ema = std::slice::from_raw_parts_mut(out_ema_ptr, len);
        let out_signal = std::slice::from_raw_parts_mut(out_signal_ptr, len);
        let input = DailyFactorInput::from_slices(
            open,
            high,
            low,
            close,
            DailyFactorParams {
                threshold_level: Some(threshold_level),
            },
        );
        daily_factor_into_slice(out_value, out_ema, out_signal, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn daily_factor_batch_into(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_value_ptr: *mut f64,
    out_ema_ptr: *mut f64,
    out_signal_ptr: *mut f64,
    len: usize,
    threshold_level_start: f64,
    threshold_level_end: f64,
    threshold_level_step: f64,
) -> Result<usize, JsValue> {
    if open_ptr.is_null()
        || high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || out_value_ptr.is_null()
        || out_ema_ptr.is_null()
        || out_signal_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to daily_factor_batch_into",
        ));
    }
    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let sweep = DailyFactorBatchRange {
            threshold_level: (
                threshold_level_start,
                threshold_level_end,
                threshold_level_step,
            ),
        };
        let combos = expand_grid(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let total = rows
            .checked_mul(len)
            .ok_or_else(|| JsValue::from_str("rows*cols overflow in daily_factor_batch_into"))?;
        let out_value = std::slice::from_raw_parts_mut(out_value_ptr, total);
        let out_ema = std::slice::from_raw_parts_mut(out_ema_ptr, total);
        let out_signal = std::slice::from_raw_parts_mut(out_signal_ptr, total);
        daily_factor_batch_into_slice(
            out_value,
            out_ema,
            out_signal,
            open,
            high,
            low,
            close,
            &sweep,
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn daily_factor_output_into_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    threshold_level: f64,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = daily_factor_js(open, high, low, close, threshold_level)?;
    crate::write_wasm_object_f64_outputs("daily_factor_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn daily_factor_batch_output_into_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = daily_factor_batch_js(open, high, low, close, config)?;
    crate::write_wasm_selected_object_f64_outputs("daily_factor_batch_output_into_js", &value, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manual_daily_factor(
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        threshold_level: f64,
    ) -> DailyFactorOutput {
        let len = close.len();
        let mut value = vec![f64::NAN; len];
        let mut ema = vec![f64::NAN; len];
        let mut signal = vec![f64::NAN; len];
        let first = first_valid_ohlc(open, high, low, close).unwrap();
        compute_base_into(open, high, low, close, first, &mut value, &mut ema).unwrap();
        compute_signal_into(&value, &ema, close, threshold_level, &mut signal).unwrap();
        DailyFactorOutput { value, ema, signal }
    }

    fn sample_ohlc(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let open: Vec<f64> = (0..n)
            .map(|i| 100.0 + ((i as f64) * 0.17).sin() * 1.4 + (i as f64) * 0.03)
            .collect();
        let close: Vec<f64> = open
            .iter()
            .enumerate()
            .map(|(i, &o)| o + ((i as f64) * 0.11).cos() * 0.85)
            .collect();
        let high: Vec<f64> = open
            .iter()
            .zip(close.iter())
            .enumerate()
            .map(|(i, (&o, &c))| o.max(c) + 0.9 + ((i as f64) * 0.07).sin().abs())
            .collect();
        let low: Vec<f64> = open
            .iter()
            .zip(close.iter())
            .enumerate()
            .map(|(i, (&o, &c))| o.min(c) - 0.8 - ((i as f64) * 0.09).cos().abs())
            .collect();
        (open, high, low, close)
    }

    #[test]
    fn matches_manual_reference() {
        let (open, high, low, close) = sample_ohlc(96);
        let input = DailyFactorInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            DailyFactorParams {
                threshold_level: Some(0.35),
            },
        );
        let out = daily_factor(&input).unwrap();
        let expected = manual_daily_factor(&open, &high, &low, &close, 0.35);
        assert_eq!(out.value.len(), expected.value.len());
        for i in 0..close.len() {
            let got = out.value[i];
            let want = expected.value[i];
            assert!(
                (got.is_nan() && want.is_nan()) || (got - want).abs() <= 1e-12,
                "value mismatch at {i}: {got} vs {want}"
            );
            let got = out.ema[i];
            let want = expected.ema[i];
            assert!(
                (got.is_nan() && want.is_nan()) || (got - want).abs() <= 1e-12,
                "ema mismatch at {i}: {got} vs {want}"
            );
            let got = out.signal[i];
            let want = expected.signal[i];
            assert!(
                (got.is_nan() && want.is_nan()) || (got - want).abs() <= 1e-12,
                "signal mismatch at {i}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn stream_matches_batch() {
        let (open, high, low, close) = sample_ohlc(80);
        let input = DailyFactorInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            DailyFactorParams {
                threshold_level: Some(0.35),
            },
        );
        let batch = daily_factor(&input).unwrap();
        let mut stream = DailyFactorStream::try_new(DailyFactorParams {
            threshold_level: Some(0.35),
        })
        .unwrap();
        for i in 0..close.len() {
            let (value, ema, signal) = stream.update(open[i], high[i], low[i], close[i]);
            let cmp = |got: f64, want: f64| {
                (got.is_nan() && want.is_nan()) || (got - want).abs() <= 1e-12
            };
            assert!(cmp(value, batch.value[i]));
            assert!(cmp(ema, batch.ema[i]));
            assert!(cmp(signal, batch.signal[i]));
        }
    }

    #[test]
    fn batch_first_row_matches_single() {
        let (open, high, low, close) = sample_ohlc(72);
        let sweep = DailyFactorBatchRange {
            threshold_level: (0.35, 0.45, 0.10),
        };
        let out = daily_factor_batch_with_kernel(&open, &high, &low, &close, &sweep, Kernel::Auto)
            .unwrap();
        assert_eq!(out.rows, 2);
        assert_eq!(out.cols, close.len());
        let single = manual_daily_factor(&open, &high, &low, &close, 0.35);
        let end = close.len();
        for i in 0..end {
            let got = out.value[i];
            let want = single.value[i];
            assert!((got.is_nan() && want.is_nan()) || (got - want).abs() <= 1e-12);
            let got = out.ema[i];
            let want = single.ema[i];
            assert!((got.is_nan() && want.is_nan()) || (got - want).abs() <= 1e-12);
            let got = out.signal[i];
            let want = single.signal[i];
            assert!((got.is_nan() && want.is_nan()) || (got - want).abs() <= 1e-12);
        }
    }

    #[test]
    fn invalid_threshold_level_fails() {
        let (open, high, low, close) = sample_ohlc(16);
        let input = DailyFactorInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            DailyFactorParams {
                threshold_level: Some(1.5),
            },
        );
        let err = daily_factor(&input).unwrap_err();
        assert!(matches!(
            err,
            DailyFactorError::InvalidThresholdLevel { .. }
        ));
    }
}
