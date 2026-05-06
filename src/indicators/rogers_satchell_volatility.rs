#[cfg(all(feature = "python", feature = "cuda"))]
pub use crate::utilities::dlpack_cuda::{make_device_array_py, DeviceArrayF32Py};

#[cfg(all(feature = "python", feature = "cuda"))]
use numpy::PyReadonlyArray2;
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
use std::collections::HashMap;
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum RogersSatchellVolatilityData<'a> {
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
pub struct RogersSatchellVolatilityOutput {
    pub rs: Vec<f64>,
    pub signal: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct RogersSatchellVolatilityParams {
    pub lookback: Option<usize>,
    pub signal_length: Option<usize>,
}

impl Default for RogersSatchellVolatilityParams {
    fn default() -> Self {
        Self {
            lookback: Some(8),
            signal_length: Some(8),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RogersSatchellVolatilityInput<'a> {
    pub data: RogersSatchellVolatilityData<'a>,
    pub params: RogersSatchellVolatilityParams,
}

impl<'a> RogersSatchellVolatilityInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: RogersSatchellVolatilityParams) -> Self {
        Self {
            data: RogersSatchellVolatilityData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: RogersSatchellVolatilityParams,
    ) -> Self {
        Self {
            data: RogersSatchellVolatilityData::Slices {
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
        Self::from_candles(candles, RogersSatchellVolatilityParams::default())
    }

    #[inline]
    pub fn get_lookback(&self) -> usize {
        self.params.lookback.unwrap_or(8)
    }

    #[inline]
    pub fn get_signal_length(&self) -> usize {
        self.params.signal_length.unwrap_or(8)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct RogersSatchellVolatilityBuilder {
    lookback: Option<usize>,
    signal_length: Option<usize>,
    kernel: Kernel,
}

impl Default for RogersSatchellVolatilityBuilder {
    fn default() -> Self {
        Self {
            lookback: None,
            signal_length: None,
            kernel: Kernel::Auto,
        }
    }
}

impl RogersSatchellVolatilityBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn lookback(mut self, value: usize) -> Self {
        self.lookback = Some(value);
        self
    }

    #[inline(always)]
    pub fn signal_length(mut self, value: usize) -> Self {
        self.signal_length = Some(value);
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
    ) -> Result<RogersSatchellVolatilityOutput, RogersSatchellVolatilityError> {
        let params = RogersSatchellVolatilityParams {
            lookback: self.lookback,
            signal_length: self.signal_length,
        };
        let input = RogersSatchellVolatilityInput::from_candles(candles, params);
        rogers_satchell_volatility_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<RogersSatchellVolatilityOutput, RogersSatchellVolatilityError> {
        let params = RogersSatchellVolatilityParams {
            lookback: self.lookback,
            signal_length: self.signal_length,
        };
        let input = RogersSatchellVolatilityInput::from_slices(open, high, low, close, params);
        rogers_satchell_volatility_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<RogersSatchellVolatilityStream, RogersSatchellVolatilityError> {
        let params = RogersSatchellVolatilityParams {
            lookback: self.lookback,
            signal_length: self.signal_length,
        };
        RogersSatchellVolatilityStream::try_new(params)
    }
}

#[derive(Debug, Error)]
pub enum RogersSatchellVolatilityError {
    #[error("rogers_satchell_volatility: Input data slice is empty.")]
    EmptyInputData,
    #[error("rogers_satchell_volatility: No valid OHLC values were found.")]
    NoValidInputData,
    #[error(
        "rogers_satchell_volatility: Invalid lookback: lookback = {lookback}, data length = {data_len}"
    )]
    InvalidLookback { lookback: usize, data_len: usize },
    #[error("rogers_satchell_volatility: Invalid signal length: signal_length = {signal_length}")]
    InvalidSignalLength { signal_length: usize },
    #[error(
        "rogers_satchell_volatility: Not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("rogers_satchell_volatility: Inconsistent slice lengths: open={open_len}, high={high_len}, low={low_len}, close={close_len}")]
    InconsistentSliceLengths {
        open_len: usize,
        high_len: usize,
        low_len: usize,
        close_len: usize,
    },
    #[error(
        "rogers_satchell_volatility: Output length mismatch: expected = {expected}, got = {got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("rogers_satchell_volatility: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("rogers_satchell_volatility: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Debug, Clone)]
pub struct RogersSatchellVolatilityStream {
    lookback: usize,
    signal_length: usize,
    term_ring: Vec<Option<f64>>,
    term_sum: f64,
    term_valid: usize,
    term_idx: usize,
    term_count: usize,
    signal_ring: Vec<Option<f64>>,
    signal_sum: f64,
    signal_valid: usize,
    signal_idx: usize,
    signal_count: usize,
}

impl RogersSatchellVolatilityStream {
    #[inline(always)]
    pub fn try_new(
        params: RogersSatchellVolatilityParams,
    ) -> Result<Self, RogersSatchellVolatilityError> {
        let lookback = params.lookback.unwrap_or(8);
        if lookback == 0 {
            return Err(RogersSatchellVolatilityError::InvalidLookback {
                lookback,
                data_len: 0,
            });
        }
        let signal_length = params.signal_length.unwrap_or(8);
        if signal_length == 0 {
            return Err(RogersSatchellVolatilityError::InvalidSignalLength { signal_length });
        }

        Ok(Self {
            lookback,
            signal_length,
            term_ring: vec![None; lookback],
            term_sum: 0.0,
            term_valid: 0,
            term_idx: 0,
            term_count: 0,
            signal_ring: vec![None; signal_length],
            signal_sum: 0.0,
            signal_valid: 0,
            signal_idx: 0,
            signal_count: 0,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, open: f64, high: f64, low: f64, close: f64) -> Option<(f64, f64)> {
        let term = rs_component(open, high, low, close);

        if self.term_count == self.lookback {
            if let Some(old) = self.term_ring[self.term_idx] {
                self.term_sum -= old;
                self.term_valid -= 1;
            }
        } else {
            self.term_count += 1;
        }
        self.term_ring[self.term_idx] = term;
        if let Some(value) = term {
            self.term_sum += value;
            self.term_valid += 1;
        }
        self.term_idx += 1;
        if self.term_idx == self.lookback {
            self.term_idx = 0;
        }

        let rs_value = if self.term_count == self.lookback && self.term_valid == self.lookback {
            let mut variance = self.term_sum / self.lookback as f64;
            if variance < 0.0 {
                variance = 0.0;
            }
            Some(variance.sqrt())
        } else {
            None
        };

        if self.signal_count == self.signal_length {
            if let Some(old) = self.signal_ring[self.signal_idx] {
                self.signal_sum -= old;
                self.signal_valid -= 1;
            }
        } else {
            self.signal_count += 1;
        }
        self.signal_ring[self.signal_idx] = rs_value;
        if let Some(value) = rs_value {
            self.signal_sum += value;
            self.signal_valid += 1;
        }
        self.signal_idx += 1;
        if self.signal_idx == self.signal_length {
            self.signal_idx = 0;
        }

        rs_value.map(|rs| {
            let signal = if self.signal_count == self.signal_length
                && self.signal_valid == self.signal_length
            {
                self.signal_sum / self.signal_length as f64
            } else {
                f64::NAN
            };
            (rs, signal)
        })
    }

    #[inline(always)]
    pub fn get_warmup_period(&self) -> usize {
        self.lookback + self.signal_length - 1
    }
}

#[inline]
pub fn rogers_satchell_volatility(
    input: &RogersSatchellVolatilityInput,
) -> Result<RogersSatchellVolatilityOutput, RogersSatchellVolatilityError> {
    rogers_satchell_volatility_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn validate_ohlc(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

#[inline(always)]
fn rs_component(open: f64, high: f64, low: f64, close: f64) -> Option<f64> {
    if !validate_ohlc(open) || !validate_ohlc(high) || !validate_ohlc(low) || !validate_ohlc(close)
    {
        return None;
    }
    Some(rs_component_valid(open, high, low, close))
}

#[inline(always)]
fn rs_component_valid(open: f64, high: f64, low: f64, close: f64) -> f64 {
    let high_close = (high / close).ln();
    let low_close = (low / close).ln();
    let close_open = (close / open).ln();
    high_close * (high_close + close_open) + low_close * (low_close + close_open)
}

#[inline(always)]
fn prepare_input<'a>(
    input: &'a RogersSatchellVolatilityInput,
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
        bool,
    ),
    RogersSatchellVolatilityError,
> {
    let (open, high, low, close): (&[f64], &[f64], &[f64], &[f64]) = match &input.data {
        RogersSatchellVolatilityData::Candles { candles } => {
            (&candles.open, &candles.high, &candles.low, &candles.close)
        }
        RogersSatchellVolatilityData::Slices {
            open,
            high,
            low,
            close,
        } => (open, high, low, close),
    };

    let len = close.len();
    if len == 0 {
        return Err(RogersSatchellVolatilityError::EmptyInputData);
    }
    if open.len() != len || high.len() != len || low.len() != len {
        return Err(RogersSatchellVolatilityError::InconsistentSliceLengths {
            open_len: open.len(),
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }

    let lookback = input.get_lookback();
    if lookback == 0 || lookback > len {
        return Err(RogersSatchellVolatilityError::InvalidLookback {
            lookback,
            data_len: len,
        });
    }

    let signal_length = input.get_signal_length();
    if signal_length == 0 {
        return Err(RogersSatchellVolatilityError::InvalidSignalLength { signal_length });
    }

    let mut valid = 0usize;
    for i in 0..len {
        if validate_ohlc(open[i])
            && validate_ohlc(high[i])
            && validate_ohlc(low[i])
            && validate_ohlc(close[i])
        {
            valid += 1;
        }
    }
    if valid == 0 {
        return Err(RogersSatchellVolatilityError::NoValidInputData);
    }
    if valid < lookback {
        return Err(RogersSatchellVolatilityError::NotEnoughValidData {
            needed: lookback,
            valid,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        value => value.to_non_batch(),
    };

    Ok((
        open,
        high,
        low,
        close,
        lookback,
        signal_length,
        chosen,
        valid == len,
    ))
}

#[inline(always)]
fn build_term_prefixes(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> (Vec<usize>, Vec<f64>) {
    let len = close.len();
    let mut prefix_valid = vec![0usize; len + 1];
    let mut prefix_sum = vec![0.0f64; len + 1];

    for i in 0..len {
        prefix_valid[i + 1] = prefix_valid[i];
        prefix_sum[i + 1] = prefix_sum[i];
        if let Some(term) = rs_component(open[i], high[i], low[i], close[i]) {
            prefix_valid[i + 1] += 1;
            prefix_sum[i + 1] += term;
        }
    }

    (prefix_valid, prefix_sum)
}

#[inline(always)]
fn compute_rs_from_prefix(
    prefix_valid: &[usize],
    prefix_sum: &[f64],
    lookback: usize,
    out: &mut [f64],
) {
    let len = out.len();
    if len == 0 {
        return;
    }
    let warm = lookback.saturating_sub(1).min(len);
    for value in &mut out[..warm] {
        *value = f64::NAN;
    }
    for t in warm..len {
        let end = t + 1;
        let start = end - lookback;
        if prefix_valid[end] - prefix_valid[start] == lookback {
            let mut variance = (prefix_sum[end] - prefix_sum[start]) / lookback as f64;
            if variance < 0.0 {
                variance = 0.0;
            }
            out[t] = variance.sqrt();
        } else {
            out[t] = f64::NAN;
        }
    }
}

#[inline(always)]
fn compute_signal_from_rs(rs: &[f64], signal_length: usize, out: &mut [f64]) {
    let len = rs.len();
    if len == 0 {
        return;
    }
    let mut prefix_valid = vec![0usize; len + 1];
    let mut prefix_sum = vec![0.0f64; len + 1];
    for i in 0..len {
        prefix_valid[i + 1] = prefix_valid[i];
        prefix_sum[i + 1] = prefix_sum[i];
        let value = rs[i];
        if value.is_finite() {
            prefix_valid[i + 1] += 1;
            prefix_sum[i + 1] += value;
        }
    }

    let warm = signal_length.saturating_sub(1).min(len);
    for value in &mut out[..warm] {
        *value = f64::NAN;
    }
    for t in warm..len {
        let end = t + 1;
        let start = end - signal_length;
        if prefix_valid[end] - prefix_valid[start] == signal_length {
            out[t] = (prefix_sum[end] - prefix_sum[start]) / signal_length as f64;
        } else {
            out[t] = f64::NAN;
        }
    }
}

#[inline(always)]
fn compute_all_valid_rolling(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    lookback: usize,
    signal_length: usize,
    out_rs: &mut [f64],
    out_signal: &mut [f64],
) {
    if lookback == 8 && signal_length == 8 {
        compute_all_valid_rolling_fixed::<8, 8>(open, high, low, close, out_rs, out_signal);
        return;
    }

    let len = close.len();
    let rs_warm = lookback.saturating_sub(1).min(len);
    out_rs[..rs_warm].fill(f64::NAN);
    let signal_warm = lookback
        .saturating_sub(1)
        .saturating_add(signal_length.saturating_sub(1))
        .min(len);
    out_signal[..signal_warm].fill(f64::NAN);

    let mut term_ring = vec![0.0; lookback];
    let mut signal_ring = vec![0.0; signal_length];
    let inv_lookback = 1.0 / lookback as f64;
    let inv_signal = 1.0 / signal_length as f64;
    let mut term_sum = 0.0;
    let mut term_idx = 0usize;
    let mut term_count = 0usize;
    let mut signal_sum = 0.0;
    let mut signal_idx = 0usize;
    let mut signal_count = 0usize;

    for i in 0..len {
        let term = rs_component_valid(open[i], high[i], low[i], close[i]);
        if term_count == lookback {
            term_sum -= term_ring[term_idx];
        } else {
            term_count += 1;
        }
        term_ring[term_idx] = term;
        term_sum += term;
        term_idx += 1;
        if term_idx == lookback {
            term_idx = 0;
        }

        if term_count == lookback {
            let mut variance = term_sum * inv_lookback;
            if variance < 0.0 {
                variance = 0.0;
            }
            let rs = variance.sqrt();
            out_rs[i] = rs;

            if signal_count == signal_length {
                signal_sum -= signal_ring[signal_idx];
            } else {
                signal_count += 1;
            }
            signal_ring[signal_idx] = rs;
            signal_sum += rs;
            signal_idx += 1;
            if signal_idx == signal_length {
                signal_idx = 0;
            }

            if signal_count == signal_length {
                out_signal[i] = signal_sum * inv_signal;
            }
        }
    }
}

#[inline(always)]
fn compute_all_valid_rolling_fixed<const LOOKBACK: usize, const SIGNAL: usize>(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    out_rs: &mut [f64],
    out_signal: &mut [f64],
) {
    let len = close.len();
    let rs_warm = LOOKBACK.saturating_sub(1).min(len);
    out_rs[..rs_warm].fill(f64::NAN);
    let signal_warm = LOOKBACK
        .saturating_sub(1)
        .saturating_add(SIGNAL.saturating_sub(1))
        .min(len);
    out_signal[..signal_warm].fill(f64::NAN);

    let mut term_ring = [0.0; LOOKBACK];
    let mut signal_ring = [0.0; SIGNAL];
    let inv_lookback = 1.0 / LOOKBACK as f64;
    let inv_signal = 1.0 / SIGNAL as f64;
    let mut term_sum = 0.0;
    let mut term_idx = 0usize;
    let mut term_count = 0usize;
    let mut signal_sum = 0.0;
    let mut signal_idx = 0usize;
    let mut signal_count = 0usize;

    for i in 0..len {
        let term = rs_component_valid(open[i], high[i], low[i], close[i]);
        if term_count == LOOKBACK {
            term_sum -= term_ring[term_idx];
        } else {
            term_count += 1;
        }
        term_ring[term_idx] = term;
        term_sum += term;
        term_idx += 1;
        if term_idx == LOOKBACK {
            term_idx = 0;
        }

        if term_count == LOOKBACK {
            let mut variance = term_sum * inv_lookback;
            if variance < 0.0 {
                variance = 0.0;
            }
            let rs = variance.sqrt();
            out_rs[i] = rs;

            if signal_count == SIGNAL {
                signal_sum -= signal_ring[signal_idx];
            } else {
                signal_count += 1;
            }
            signal_ring[signal_idx] = rs;
            signal_sum += rs;
            signal_idx += 1;
            if signal_idx == SIGNAL {
                signal_idx = 0;
            }

            if signal_count == SIGNAL {
                out_signal[i] = signal_sum * inv_signal;
            }
        }
    }
}

#[inline(always)]
fn rogers_satchell_volatility_compute_into(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    lookback: usize,
    signal_length: usize,
    _kernel: Kernel,
    all_valid: bool,
    out_rs: &mut [f64],
    out_signal: &mut [f64],
) {
    if all_valid {
        compute_all_valid_rolling(
            open,
            high,
            low,
            close,
            lookback,
            signal_length,
            out_rs,
            out_signal,
        );
        return;
    }

    let (prefix_valid, prefix_sum) = build_term_prefixes(open, high, low, close);
    compute_rs_from_prefix(&prefix_valid, &prefix_sum, lookback, out_rs);
    compute_signal_from_rs(out_rs, signal_length, out_signal);
}

#[inline]
pub fn rogers_satchell_volatility_with_kernel(
    input: &RogersSatchellVolatilityInput,
    kernel: Kernel,
) -> Result<RogersSatchellVolatilityOutput, RogersSatchellVolatilityError> {
    let (open, high, low, close, lookback, signal_length, chosen, all_valid) =
        prepare_input(input, kernel)?;
    let len = close.len();

    let mut rs = alloc_with_nan_prefix(len, lookback.saturating_sub(1));
    let signal_warm = lookback
        .saturating_sub(1)
        .saturating_add(signal_length.saturating_sub(1));
    let mut signal = alloc_with_nan_prefix(len, signal_warm);

    rogers_satchell_volatility_compute_into(
        open,
        high,
        low,
        close,
        lookback,
        signal_length,
        chosen,
        all_valid,
        &mut rs,
        &mut signal,
    );

    Ok(RogersSatchellVolatilityOutput { rs, signal })
}

#[inline]
pub fn rogers_satchell_volatility_into_slice(
    out_rs: &mut [f64],
    out_signal: &mut [f64],
    input: &RogersSatchellVolatilityInput,
    kernel: Kernel,
) -> Result<(), RogersSatchellVolatilityError> {
    let (open, high, low, close, lookback, signal_length, chosen, all_valid) =
        prepare_input(input, kernel)?;
    let expected = close.len();
    if out_rs.len() != expected || out_signal.len() != expected {
        return Err(RogersSatchellVolatilityError::OutputLengthMismatch {
            expected,
            got: out_rs.len().max(out_signal.len()),
        });
    }

    rogers_satchell_volatility_compute_into(
        open,
        high,
        low,
        close,
        lookback,
        signal_length,
        chosen,
        all_valid,
        out_rs,
        out_signal,
    );
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn rogers_satchell_volatility_into(
    input: &RogersSatchellVolatilityInput,
    out_rs: &mut [f64],
    out_signal: &mut [f64],
) -> Result<(), RogersSatchellVolatilityError> {
    rogers_satchell_volatility_into_slice(out_rs, out_signal, input, Kernel::Auto)
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct RogersSatchellVolatilityBatchRange {
    pub lookback: (usize, usize, usize),
    pub signal_length: (usize, usize, usize),
}

impl Default for RogersSatchellVolatilityBatchRange {
    fn default() -> Self {
        Self {
            lookback: (8, 252, 1),
            signal_length: (8, 8, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct RogersSatchellVolatilityBatchBuilder {
    range: RogersSatchellVolatilityBatchRange,
    kernel: Kernel,
}

impl RogersSatchellVolatilityBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline]
    pub fn lookback_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.lookback = (start, end, step);
        self
    }

    #[inline]
    pub fn lookback_static(mut self, value: usize) -> Self {
        self.range.lookback = (value, value, 0);
        self
    }

    #[inline]
    pub fn signal_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.signal_length = (start, end, step);
        self
    }

    #[inline]
    pub fn signal_length_static(mut self, value: usize) -> Self {
        self.range.signal_length = (value, value, 0);
        self
    }

    pub fn apply_slices(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<RogersSatchellVolatilityBatchOutput, RogersSatchellVolatilityError> {
        rogers_satchell_volatility_batch_with_kernel(
            open,
            high,
            low,
            close,
            &self.range,
            self.kernel,
        )
    }

    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<RogersSatchellVolatilityBatchOutput, RogersSatchellVolatilityError> {
        self.apply_slices(&candles.open, &candles.high, &candles.low, &candles.close)
    }
}

#[derive(Clone, Debug)]
pub struct RogersSatchellVolatilityBatchOutput {
    pub rs: Vec<f64>,
    pub signal: Vec<f64>,
    pub combos: Vec<RogersSatchellVolatilityParams>,
    pub rows: usize,
    pub cols: usize,
}

impl RogersSatchellVolatilityBatchOutput {
    pub fn row_for_params(&self, params: &RogersSatchellVolatilityParams) -> Option<usize> {
        let lookback = params.lookback.unwrap_or(8);
        let signal_length = params.signal_length.unwrap_or(8);
        self.combos.iter().position(|combo| {
            combo.lookback.unwrap_or(8) == lookback
                && combo.signal_length.unwrap_or(8) == signal_length
        })
    }

    pub fn rs_for(&self, params: &RogersSatchellVolatilityParams) -> Option<&[f64]> {
        self.row_for_params(params).and_then(|row| {
            row.checked_mul(self.cols)
                .and_then(|start| self.rs.get(start..start + self.cols))
        })
    }

    pub fn signal_for(&self, params: &RogersSatchellVolatilityParams) -> Option<&[f64]> {
        self.row_for_params(params).and_then(|row| {
            row.checked_mul(self.cols)
                .and_then(|start| self.signal.get(start..start + self.cols))
        })
    }
}

#[inline(always)]
fn axis_usize(
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, RogersSatchellVolatilityError> {
    if step == 0 || start == end {
        return Ok(vec![start]);
    }
    let step = step.max(1);
    if start < end {
        let mut values = Vec::new();
        let mut current = start;
        while current <= end {
            values.push(current);
            match current.checked_add(step) {
                Some(next) if next != current => current = next,
                _ => break,
            }
        }
        if values.is_empty() {
            return Err(RogersSatchellVolatilityError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        Ok(values)
    } else {
        let mut values = Vec::new();
        let mut current = start;
        loop {
            values.push(current);
            if current == end {
                break;
            }
            let next = current.saturating_sub(step);
            if next == current || next < end {
                break;
            }
            current = next;
        }
        if values.is_empty() {
            return Err(RogersSatchellVolatilityError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        Ok(values)
    }
}

#[inline(always)]
fn expand_grid_rogers_satchell(
    range: &RogersSatchellVolatilityBatchRange,
) -> Result<Vec<RogersSatchellVolatilityParams>, RogersSatchellVolatilityError> {
    let lookbacks = axis_usize(range.lookback)?;
    let signal_lengths = axis_usize(range.signal_length)?;
    let total = lookbacks
        .len()
        .checked_mul(signal_lengths.len())
        .ok_or_else(|| RogersSatchellVolatilityError::InvalidRange {
            start: "lookback".to_string(),
            end: "signal_length".to_string(),
            step: "overflow".to_string(),
        })?;
    let mut combos = Vec::with_capacity(total);
    for &lookback in &lookbacks {
        for &signal_length in &signal_lengths {
            combos.push(RogersSatchellVolatilityParams {
                lookback: Some(lookback),
                signal_length: Some(signal_length),
            });
        }
    }
    Ok(combos)
}

#[inline(always)]
fn fill_row_from_cache(
    combo: &RogersSatchellVolatilityParams,
    raw_cache: &HashMap<usize, Vec<f64>>,
    signal_cache: &HashMap<(usize, usize), Vec<f64>>,
    rs_row: &mut [f64],
    signal_row: &mut [f64],
) {
    let lookback = combo.lookback.unwrap_or(8);
    let signal_length = combo.signal_length.unwrap_or(8);
    rs_row.copy_from_slice(raw_cache.get(&lookback).unwrap());
    signal_row.copy_from_slice(signal_cache.get(&(lookback, signal_length)).unwrap());
}

#[inline(always)]
fn rogers_satchell_volatility_batch_inner_into(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &RogersSatchellVolatilityBatchRange,
    kernel: Kernel,
    parallel: bool,
    out_rs: &mut [f64],
    out_signal: &mut [f64],
) -> Result<Vec<RogersSatchellVolatilityParams>, RogersSatchellVolatilityError> {
    let len = close.len();
    if len == 0 {
        return Err(RogersSatchellVolatilityError::EmptyInputData);
    }
    if open.len() != len || high.len() != len || low.len() != len {
        return Err(RogersSatchellVolatilityError::InconsistentSliceLengths {
            open_len: open.len(),
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }

    let combos = expand_grid_rogers_satchell(sweep)?;
    let rows = combos.len();
    let expected = rows.checked_mul(len).ok_or_else(|| {
        RogersSatchellVolatilityError::OutputLengthMismatch {
            expected: usize::MAX,
            got: out_rs.len().max(out_signal.len()),
        }
    })?;
    if out_rs.len() != expected || out_signal.len() != expected {
        return Err(RogersSatchellVolatilityError::OutputLengthMismatch {
            expected,
            got: out_rs.len().max(out_signal.len()),
        });
    }

    let kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel().to_non_batch(),
        value => value.to_non_batch(),
    };

    let max_lookback = combos
        .iter()
        .map(|combo| combo.lookback.unwrap_or(8))
        .max()
        .unwrap_or(8);
    let valid = open
        .iter()
        .zip(high.iter())
        .zip(low.iter())
        .zip(close.iter())
        .filter(|(((o, h), l), c)| {
            validate_ohlc(**o) && validate_ohlc(**h) && validate_ohlc(**l) && validate_ohlc(**c)
        })
        .count();
    if valid == 0 {
        return Err(RogersSatchellVolatilityError::NoValidInputData);
    }
    if valid < max_lookback {
        return Err(RogersSatchellVolatilityError::NotEnoughValidData {
            needed: max_lookback,
            valid,
        });
    }

    for combo in &combos {
        let lookback = combo.lookback.unwrap_or(8);
        if lookback == 0 || lookback > len {
            return Err(RogersSatchellVolatilityError::InvalidLookback {
                lookback,
                data_len: len,
            });
        }
        let signal_length = combo.signal_length.unwrap_or(8);
        if signal_length == 0 {
            return Err(RogersSatchellVolatilityError::InvalidSignalLength { signal_length });
        }
    }

    let (prefix_valid, prefix_sum) = build_term_prefixes(open, high, low, close);
    let mut raw_cache: HashMap<usize, Vec<f64>> = HashMap::new();
    let mut signal_cache: HashMap<(usize, usize), Vec<f64>> = HashMap::new();

    for combo in &combos {
        let lookback = combo.lookback.unwrap_or(8);
        raw_cache.entry(lookback).or_insert_with(|| {
            let mut row = alloc_with_nan_prefix(len, lookback.saturating_sub(1));
            compute_rs_from_prefix(&prefix_valid, &prefix_sum, lookback, &mut row);
            row
        });
        let signal_length = combo.signal_length.unwrap_or(8);
        signal_cache
            .entry((lookback, signal_length))
            .or_insert_with(|| {
                let raw = raw_cache.get(&lookback).unwrap();
                let warm = lookback
                    .saturating_sub(1)
                    .saturating_add(signal_length.saturating_sub(1));
                let mut row = alloc_with_nan_prefix(len, warm);
                compute_signal_from_rs(raw, signal_length, &mut row);
                row
            });
    }

    #[cfg(not(target_arch = "wasm32"))]
    if parallel {
        out_rs
            .par_chunks_mut(len)
            .zip(out_signal.par_chunks_mut(len))
            .zip(combos.par_iter())
            .for_each(|((rs_row, signal_row), combo)| {
                fill_row_from_cache(combo, &raw_cache, &signal_cache, rs_row, signal_row);
            });
        let _ = kernel;
        return Ok(combos);
    }

    for ((rs_row, signal_row), combo) in out_rs
        .chunks_mut(len)
        .zip(out_signal.chunks_mut(len))
        .zip(combos.iter())
    {
        fill_row_from_cache(combo, &raw_cache, &signal_cache, rs_row, signal_row);
    }
    let _ = kernel;
    Ok(combos)
}

#[inline]
pub fn rogers_satchell_volatility_batch_with_kernel(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &RogersSatchellVolatilityBatchRange,
    kernel: Kernel,
) -> Result<RogersSatchellVolatilityBatchOutput, RogersSatchellVolatilityError> {
    let combos = expand_grid_rogers_satchell(sweep)?;
    if combos.is_empty() {
        return Err(RogersSatchellVolatilityError::InvalidRange {
            start: "range".to_string(),
            end: "range".to_string(),
            step: "empty".to_string(),
        });
    }
    let rows = combos.len();
    let cols = close.len();

    let rs_warm: Vec<usize> = combos
        .iter()
        .map(|combo| combo.lookback.unwrap_or(8).saturating_sub(1).min(cols))
        .collect();
    let signal_warm: Vec<usize> = combos
        .iter()
        .map(|combo| {
            combo
                .lookback
                .unwrap_or(8)
                .saturating_sub(1)
                .saturating_add(combo.signal_length.unwrap_or(8).saturating_sub(1))
                .min(cols)
        })
        .collect();

    let mut rs_mu = make_uninit_matrix(rows, cols);
    let mut signal_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut rs_mu, cols, &rs_warm);
    init_matrix_prefixes(&mut signal_mu, cols, &signal_warm);

    let mut rs_guard = ManuallyDrop::new(rs_mu);
    let mut signal_guard = ManuallyDrop::new(signal_mu);
    let rs_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(rs_guard.as_mut_ptr() as *mut f64, rs_guard.len())
    };
    let signal_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(signal_guard.as_mut_ptr() as *mut f64, signal_guard.len())
    };

    let resolved = rogers_satchell_volatility_batch_inner_into(
        open, high, low, close, sweep, kernel, true, rs_out, signal_out,
    )?;

    let rs = unsafe {
        Vec::from_raw_parts(
            rs_guard.as_mut_ptr() as *mut f64,
            rs_guard.len(),
            rs_guard.capacity(),
        )
    };
    let signal = unsafe {
        Vec::from_raw_parts(
            signal_guard.as_mut_ptr() as *mut f64,
            signal_guard.len(),
            signal_guard.capacity(),
        )
    };

    Ok(RogersSatchellVolatilityBatchOutput {
        rs,
        signal,
        combos: resolved,
        rows,
        cols,
    })
}

#[inline]
pub fn rogers_satchell_volatility_batch_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &RogersSatchellVolatilityBatchRange,
    kernel: Kernel,
) -> Result<RogersSatchellVolatilityBatchOutput, RogersSatchellVolatilityError> {
    rogers_satchell_volatility_batch_with_kernel(open, high, low, close, sweep, kernel)
}

#[inline]
pub fn rogers_satchell_volatility_batch_par_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &RogersSatchellVolatilityBatchRange,
    kernel: Kernel,
) -> Result<RogersSatchellVolatilityBatchOutput, RogersSatchellVolatilityError> {
    rogers_satchell_volatility_batch_with_kernel(open, high, low, close, sweep, kernel)
}

#[cfg(feature = "python")]
#[pyfunction(name = "rogers_satchell_volatility")]
#[pyo3(signature = (open, high, low, close, lookback=8, signal_length=8, kernel=None))]
pub fn rogers_satchell_volatility_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    lookback: usize,
    signal_length: usize,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let open = open.as_slice()?;
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
        return Err(PyValueError::new_err("OHLC slice length mismatch"));
    }

    let kernel = validate_kernel(kernel, false)?;
    let input = RogersSatchellVolatilityInput::from_slices(
        open,
        high,
        low,
        close,
        RogersSatchellVolatilityParams {
            lookback: Some(lookback),
            signal_length: Some(signal_length),
        },
    );
    let out = py
        .allow_threads(|| rogers_satchell_volatility_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((out.rs.into_pyarray(py), out.signal.into_pyarray(py)))
}

#[cfg(feature = "python")]
#[pyclass(name = "RogersSatchellVolatilityStream")]
pub struct RogersSatchellVolatilityStreamPy {
    stream: RogersSatchellVolatilityStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl RogersSatchellVolatilityStreamPy {
    #[new]
    fn new(lookback: usize, signal_length: usize) -> PyResult<Self> {
        let stream = RogersSatchellVolatilityStream::try_new(RogersSatchellVolatilityParams {
            lookback: Some(lookback),
            signal_length: Some(signal_length),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, open: f64, high: f64, low: f64, close: f64) -> Option<(f64, f64)> {
        self.stream.update(open, high, low, close)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "rogers_satchell_volatility_batch")]
#[pyo3(signature = (open, high, low, close, lookback_range=(8,8,0), signal_length_range=(8,8,0), kernel=None))]
pub fn rogers_satchell_volatility_batch_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    lookback_range: (usize, usize, usize),
    signal_length_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let open = open.as_slice()?;
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
        return Err(PyValueError::new_err("OHLC slice length mismatch"));
    }

    let sweep = RogersSatchellVolatilityBatchRange {
        lookback: lookback_range,
        signal_length: signal_length_range,
    };
    let combos =
        expand_grid_rogers_satchell(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let rs_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let signal_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let rs_out = unsafe { rs_arr.as_slice_mut()? };
    let signal_out = unsafe { signal_arr.as_slice_mut()? };

    let kernel = validate_kernel(kernel, true)?;
    py.allow_threads(|| {
        rogers_satchell_volatility_batch_inner_into(
            open, high, low, close, &sweep, kernel, true, rs_out, signal_out,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("rs", rs_arr.reshape((rows, cols))?)?;
    dict.set_item("signal", signal_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "lookbacks",
        combos
            .iter()
            .map(|combo| combo.lookback.unwrap_or(8) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "signal_lengths",
        combos
            .iter()
            .map(|combo| combo.signal_length.unwrap_or(8) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "rogers_satchell_volatility_cuda_batch_dev")]
#[pyo3(signature = (open_f32, high_f32, low_f32, close_f32, lookback_range=(8,8,0), signal_length_range=(8,8,0), device_id=0))]
pub fn rogers_satchell_volatility_cuda_batch_dev_py<'py>(
    py: Python<'py>,
    open_f32: PyReadonlyArray1<'py, f32>,
    high_f32: PyReadonlyArray1<'py, f32>,
    low_f32: PyReadonlyArray1<'py, f32>,
    close_f32: PyReadonlyArray1<'py, f32>,
    lookback_range: (usize, usize, usize),
    signal_length_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<Bound<'py, PyDict>> {
    use crate::cuda::cuda_available;
    use crate::cuda::CudaRogersSatchellVolatility;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let open = open_f32.as_slice()?;
    let high = high_f32.as_slice()?;
    let low = low_f32.as_slice()?;
    let close = close_f32.as_slice()?;
    if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
        return Err(PyValueError::new_err("OHLC slice length mismatch"));
    }

    let sweep = RogersSatchellVolatilityBatchRange {
        lookback: lookback_range,
        signal_length: signal_length_range,
    };
    let combos =
        expand_grid_rogers_satchell(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    let (result, ctx, dev_id) = py.allow_threads(|| {
        let cuda = CudaRogersSatchellVolatility::new(device_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.context_arc_clone();
        let dev_id = cuda.device_id();
        cuda.rogers_satchell_volatility_batch_dev(open, high, low, close, &sweep)
            .map(|result| (result, ctx, dev_id))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    let rows = result.outputs.rs.rows;
    let cols = result.outputs.rs.cols;

    dict.set_item(
        "rs",
        DeviceArrayF32Py {
            inner: result.outputs.rs,
            _ctx: Some(ctx.clone()),
            device_id: Some(dev_id),
        },
    )?;
    dict.set_item(
        "signal",
        DeviceArrayF32Py {
            inner: result.outputs.signal,
            _ctx: Some(ctx),
            device_id: Some(dev_id),
        },
    )?;
    dict.set_item(
        "lookbacks",
        combos
            .iter()
            .map(|combo| combo.lookback.unwrap_or(8) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "signal_lengths",
        combos
            .iter()
            .map(|combo| combo.signal_length.unwrap_or(8) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "rogers_satchell_volatility_cuda_many_series_one_param_dev")]
#[pyo3(signature = (open_tm_f32, high_tm_f32, low_tm_f32, close_tm_f32, lookback=8, signal_length=8, device_id=0))]
pub fn rogers_satchell_volatility_cuda_many_series_one_param_dev_py<'py>(
    py: Python<'py>,
    open_tm_f32: PyReadonlyArray2<'py, f32>,
    high_tm_f32: PyReadonlyArray2<'py, f32>,
    low_tm_f32: PyReadonlyArray2<'py, f32>,
    close_tm_f32: PyReadonlyArray2<'py, f32>,
    lookback: usize,
    signal_length: usize,
    device_id: usize,
) -> PyResult<Bound<'py, PyDict>> {
    use crate::cuda::cuda_available;
    use crate::cuda::CudaRogersSatchellVolatility;
    use numpy::PyUntypedArrayMethods;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let rows = close_tm_f32.shape()[0];
    let cols = close_tm_f32.shape()[1];
    if open_tm_f32.shape() != high_tm_f32.shape()
        || open_tm_f32.shape() != low_tm_f32.shape()
        || open_tm_f32.shape() != close_tm_f32.shape()
    {
        return Err(PyValueError::new_err("OHLC matrix shape mismatch"));
    }

    let open = open_tm_f32.as_slice()?;
    let high = high_tm_f32.as_slice()?;
    let low = low_tm_f32.as_slice()?;
    let close = close_tm_f32.as_slice()?;

    let dict = PyDict::new(py);
    let (result, ctx, dev_id) = py.allow_threads(|| {
        let cuda = CudaRogersSatchellVolatility::new(device_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.context_arc_clone();
        let dev_id = cuda.device_id();
        cuda.rogers_satchell_volatility_many_series_one_param_time_major_dev(
            open,
            high,
            low,
            close,
            cols,
            rows,
            lookback,
            signal_length,
        )
        .map(|result| (result, ctx, dev_id))
        .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    dict.set_item(
        "rs",
        DeviceArrayF32Py {
            inner: result.rs,
            _ctx: Some(ctx.clone()),
            device_id: Some(dev_id),
        },
    )?;
    dict.set_item(
        "signal",
        DeviceArrayF32Py {
            inner: result.signal,
            _ctx: Some(ctx),
            device_id: Some(dev_id),
        },
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_rogers_satchell_volatility_module(
    m: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rogers_satchell_volatility_py, m)?)?;
    m.add_function(wrap_pyfunction!(rogers_satchell_volatility_batch_py, m)?)?;
    m.add_class::<RogersSatchellVolatilityStreamPy>()?;

    #[cfg(feature = "cuda")]
    {
        m.add_class::<DeviceArrayF32Py>()?;
        m.add_function(wrap_pyfunction!(
            rogers_satchell_volatility_cuda_batch_dev_py,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(
            rogers_satchell_volatility_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "rogers_satchell_volatility_js")]
pub fn rogers_satchell_volatility_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    lookback: usize,
    signal_length: usize,
) -> Result<JsValue, JsValue> {
    if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
        return Err(JsValue::from_str("OHLC slice length mismatch"));
    }

    let input = RogersSatchellVolatilityInput::from_slices(
        open,
        high,
        low,
        close,
        RogersSatchellVolatilityParams {
            lookback: Some(lookback),
            signal_length: Some(signal_length),
        },
    );
    let mut rs = vec![0.0; close.len()];
    let mut signal = vec![0.0; close.len()];
    rogers_satchell_volatility_into_slice(&mut rs, &mut signal, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("rs"),
        &serde_wasm_bindgen::to_value(&rs).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("signal"),
        &serde_wasm_bindgen::to_value(&signal).unwrap(),
    )?;
    Ok(obj.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct RogersSatchellVolatilityBatchConfig {
    pub lookback_range: Vec<usize>,
    pub signal_length_range: Vec<usize>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "rogers_satchell_volatility_batch_js")]
pub fn rogers_satchell_volatility_batch_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
        return Err(JsValue::from_str("OHLC slice length mismatch"));
    }

    let config: RogersSatchellVolatilityBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.lookback_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: lookback_range must have exactly 3 elements [start, end, step]",
        ));
    }
    if config.signal_length_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: signal_length_range must have exactly 3 elements [start, end, step]",
        ));
    }

    let sweep = RogersSatchellVolatilityBatchRange {
        lookback: (
            config.lookback_range[0],
            config.lookback_range[1],
            config.lookback_range[2],
        ),
        signal_length: (
            config.signal_length_range[0],
            config.signal_length_range[1],
            config.signal_length_range[2],
        ),
    };
    let combos =
        expand_grid_rogers_satchell(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;

    let mut rs = vec![0.0; total];
    let mut signal = vec![0.0; total];
    rogers_satchell_volatility_batch_inner_into(
        open,
        high,
        low,
        close,
        &sweep,
        detect_best_kernel(),
        false,
        &mut rs,
        &mut signal,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("rs"),
        &serde_wasm_bindgen::to_value(&rs).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("signal"),
        &serde_wasm_bindgen::to_value(&signal).unwrap(),
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
pub fn rogers_satchell_volatility_alloc(len: usize) -> *mut f64 {
    let mut values = Vec::<f64>::with_capacity(2 * len);
    let ptr = values.as_mut_ptr();
    std::mem::forget(values);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn rogers_satchell_volatility_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, 0, 2 * len);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn rogers_satchell_volatility_into(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    lookback: usize,
    signal_length: usize,
) -> Result<(), JsValue> {
    if open_ptr.is_null()
        || high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to rogers_satchell_volatility_into",
        ));
    }

    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, 2 * len);
        let (rs, signal) = out.split_at_mut(len);

        let input = RogersSatchellVolatilityInput::from_slices(
            open,
            high,
            low,
            close,
            RogersSatchellVolatilityParams {
                lookback: Some(lookback),
                signal_length: Some(signal_length),
            },
        );
        rogers_satchell_volatility_into_slice(rs, signal, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn rogers_satchell_volatility_batch_into(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    rs_ptr: *mut f64,
    signal_ptr: *mut f64,
    len: usize,
    lookback_start: usize,
    lookback_end: usize,
    lookback_step: usize,
    signal_length_start: usize,
    signal_length_end: usize,
    signal_length_step: usize,
) -> Result<usize, JsValue> {
    if open_ptr.is_null()
        || high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || rs_ptr.is_null()
        || signal_ptr.is_null()
    {
        return Err(JsValue::from_str("null pointer"));
    }

    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);

        let sweep = RogersSatchellVolatilityBatchRange {
            lookback: (lookback_start, lookback_end, lookback_step),
            signal_length: (signal_length_start, signal_length_end, signal_length_step),
        };
        let combos =
            expand_grid_rogers_satchell(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let cols = len;
        let rs_out = std::slice::from_raw_parts_mut(rs_ptr, rows * cols);
        let signal_out = std::slice::from_raw_parts_mut(signal_ptr, rows * cols);

        rogers_satchell_volatility_batch_inner_into(
            open,
            high,
            low,
            close,
            &sweep,
            detect_best_kernel(),
            false,
            rs_out,
            signal_out,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn rogers_satchell_volatility_output_into_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    lookback: usize,
    signal_length: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = rogers_satchell_volatility_js(open, high, low, close, lookback, signal_length)?;
    crate::write_wasm_object_f64_outputs("rogers_satchell_volatility_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn rogers_satchell_volatility_batch_output_into_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = rogers_satchell_volatility_batch_js(open, high, low, close, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "rogers_satchell_volatility_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    fn sample_ohlc(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut open = vec![0.0; len];
        let mut high = vec![0.0; len];
        let mut low = vec![0.0; len];
        let mut close = vec![0.0; len];
        let mut prev = 100.0;
        for i in 0..len {
            let x = i as f64;
            let o = (prev + (x * 0.17).sin() * 0.9 + 0.03 * x).max(1.0);
            let c = (o + (x * 0.11).cos() * 0.7).max(1.0);
            let h = o.max(c) + 0.4 + (x * 0.07).cos().abs() * 0.1;
            let l = (o.min(c) - 0.35 - (x * 0.13).sin().abs() * 0.08).max(0.01);
            open[i] = o;
            high[i] = h;
            low[i] = l;
            close[i] = c.max(0.01);
            prev = close[i];
        }
        (open, high, low, close)
    }

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        if a.is_nan() && b.is_nan() {
            return true;
        }
        (a - b).abs() <= tol
    }

    fn direct_rs_component(open: f64, high: f64, low: f64, close: f64) -> f64 {
        (high / close).ln() * (high / open).ln() + (low / close).ln() * (low / open).ln()
    }

    #[test]
    fn test_rs_component_rewrite_matches_direct_formula() {
        for i in 1..128 {
            let x = i as f64;
            let open = 90.0 + x * 0.31 + (x * 0.17).sin();
            let close = open + (x * 0.11).cos() * 0.8;
            let high = open.max(close) + 0.25 + (x * 0.07).sin().abs();
            let low = (open.min(close) - 0.2 - (x * 0.13).cos().abs() * 0.3).max(0.01);
            assert!(approx_eq(
                rs_component_valid(open, high, low, close),
                direct_rs_component(open, high, low, close),
                1e-14
            ));
        }
    }

    #[test]
    fn test_into_slice_overwrites_stale_buffers_all_valid() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close) = sample_ohlc(40);
        let input = RogersSatchellVolatilityInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            RogersSatchellVolatilityParams::default(),
        );
        let mut rs = vec![123.0; close.len()];
        let mut signal = vec![456.0; close.len()];
        rogers_satchell_volatility_into_slice(&mut rs, &mut signal, &input, Kernel::Auto)?;

        assert!(rs[..7].iter().all(|v| v.is_nan()));
        assert!(signal[..14].iter().all(|v| v.is_nan()));
        assert!(rs[7..].iter().all(|v| v.is_finite()));
        assert!(signal[14..].iter().all(|v| v.is_finite()));
        assert!(!rs.iter().any(|v| *v == 123.0));
        assert!(!signal.iter().any(|v| *v == 456.0));
        Ok(())
    }

    #[test]
    fn test_output_contract() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close) = sample_ohlc(128);
        let input = RogersSatchellVolatilityInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            RogersSatchellVolatilityParams {
                lookback: Some(8),
                signal_length: Some(8),
            },
        );
        let out = rogers_satchell_volatility(&input)?;
        assert_eq!(out.rs.len(), close.len());
        assert_eq!(out.signal.len(), close.len());
        assert!(out.rs[..7].iter().all(|v| v.is_nan()));
        assert!(out.signal[..14].iter().all(|v| v.is_nan()));
        assert!(out.rs.iter().skip(32).all(|v| v.is_finite()));
        assert!(out.signal.iter().skip(48).all(|v| v.is_finite()));
        Ok(())
    }

    #[test]
    fn test_into_slice_matches_safe_api() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close) = sample_ohlc(96);
        let input = RogersSatchellVolatilityInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            RogersSatchellVolatilityParams {
                lookback: Some(10),
                signal_length: Some(6),
            },
        );
        let baseline = rogers_satchell_volatility_with_kernel(&input, Kernel::Scalar)?;
        let mut rs = vec![0.0; close.len()];
        let mut signal = vec![0.0; close.len()];
        rogers_satchell_volatility_into_slice(&mut rs, &mut signal, &input, Kernel::Auto)?;
        for i in 0..close.len() {
            assert!(approx_eq(baseline.rs[i], rs[i], 1e-12));
            assert!(approx_eq(baseline.signal[i], signal[i], 1e-12));
        }
        Ok(())
    }

    #[test]
    fn test_stream_matches_batch() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close) = sample_ohlc(96);
        let input = RogersSatchellVolatilityInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            RogersSatchellVolatilityParams {
                lookback: Some(9),
                signal_length: Some(5),
            },
        );
        let batch = rogers_satchell_volatility(&input)?;
        let mut stream = RogersSatchellVolatilityStream::try_new(RogersSatchellVolatilityParams {
            lookback: Some(9),
            signal_length: Some(5),
        })?;
        let mut rs = Vec::with_capacity(close.len());
        let mut signal = Vec::with_capacity(close.len());
        for i in 0..close.len() {
            if let Some((r, s)) = stream.update(open[i], high[i], low[i], close[i]) {
                rs.push(r);
                signal.push(s);
            } else {
                rs.push(f64::NAN);
                signal.push(f64::NAN);
            }
        }
        for i in 0..close.len() {
            assert!(approx_eq(batch.rs[i], rs[i], 1e-12));
            assert!(approx_eq(batch.signal[i], signal[i], 1e-12));
        }
        Ok(())
    }

    #[test]
    fn test_batch_single_param_matches_single() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close) = sample_ohlc(128);
        let sweep = RogersSatchellVolatilityBatchRange {
            lookback: (12, 12, 0),
            signal_length: (5, 5, 0),
        };
        let batch = rogers_satchell_volatility_batch_with_kernel(
            &open,
            &high,
            &low,
            &close,
            &sweep,
            Kernel::ScalarBatch,
        )?;
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, close.len());
        let input = RogersSatchellVolatilityInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            RogersSatchellVolatilityParams {
                lookback: Some(12),
                signal_length: Some(5),
            },
        );
        let single = rogers_satchell_volatility_with_kernel(&input, Kernel::Scalar)?;
        for i in 0..close.len() {
            assert!(approx_eq(batch.rs[i], single.rs[i], 1e-12));
            assert!(approx_eq(batch.signal[i], single.signal[i], 1e-12));
        }
        Ok(())
    }

    #[test]
    fn test_validation_errors() {
        let (open, high, low, close) = sample_ohlc(16);
        let input = RogersSatchellVolatilityInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            RogersSatchellVolatilityParams {
                lookback: Some(0),
                signal_length: Some(8),
            },
        );
        assert!(matches!(
            rogers_satchell_volatility(&input),
            Err(RogersSatchellVolatilityError::InvalidLookback { .. })
        ));

        let input = RogersSatchellVolatilityInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            RogersSatchellVolatilityParams {
                lookback: Some(8),
                signal_length: Some(0),
            },
        );
        assert!(matches!(
            rogers_satchell_volatility(&input),
            Err(RogersSatchellVolatilityError::InvalidSignalLength { .. })
        ));
    }
}
