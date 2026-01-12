#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(feature = "python")]
use pyo3::{exceptions::PyValueError, prelude::*};

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use serde::{Deserialize, Serialize};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

use crate::indicators::moving_averages::ma::{ma, MaData};
use crate::indicators::utility_functions::{max_rolling, min_rolling};
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::collections::VecDeque;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum StochData<'a> {
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
pub struct StochOutput {
    pub k: Vec<f64>,
    pub d: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct StochParams {
    pub fastk_period: Option<usize>,
    pub slowk_period: Option<usize>,
    pub slowk_ma_type: Option<String>,
    pub slowd_period: Option<usize>,
    pub slowd_ma_type: Option<String>,
}

impl Default for StochParams {
    fn default() -> Self {
        Self {
            fastk_period: Some(14),
            slowk_period: Some(3),
            slowk_ma_type: Some("sma".to_string()),
            slowd_period: Some(3),
            slowd_ma_type: Some("sma".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StochInput<'a> {
    pub data: StochData<'a>,
    pub params: StochParams,
}

impl<'a> StochInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, p: StochParams) -> Self {
        Self {
            data: StochData::Candles { candles: c },
            params: p,
        }
    }
    #[inline]
    pub fn from_slices(high: &'a [f64], low: &'a [f64], close: &'a [f64], p: StochParams) -> Self {
        Self {
            data: StochData::Slices { high, low, close },
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, StochParams::default())
    }
    #[inline]
    pub fn get_fastk_period(&self) -> usize {
        self.params.fastk_period.unwrap_or(14)
    }
    #[inline]
    pub fn get_slowk_period(&self) -> usize {
        self.params.slowk_period.unwrap_or(3)
    }
    #[inline]
    pub fn get_slowk_ma_type(&self) -> String {
        self.params
            .slowk_ma_type
            .clone()
            .unwrap_or_else(|| "sma".to_string())
    }
    #[inline]
    pub fn get_slowd_period(&self) -> usize {
        self.params.slowd_period.unwrap_or(3)
    }
    #[inline]
    pub fn get_slowd_ma_type(&self) -> String {
        self.params
            .slowd_ma_type
            .clone()
            .unwrap_or_else(|| "sma".to_string())
    }
}

#[derive(Copy, Clone, Debug)]
pub struct StochBuilder {
    fastk_period: Option<usize>,
    slowk_period: Option<usize>,
    slowk_ma_type: Option<&'static str>,
    slowd_period: Option<usize>,
    slowd_ma_type: Option<&'static str>,
    kernel: Kernel,
}

impl Default for StochBuilder {
    fn default() -> Self {
        Self {
            fastk_period: None,
            slowk_period: None,
            slowk_ma_type: None,
            slowd_period: None,
            slowd_ma_type: None,
            kernel: Kernel::Auto,
        }
    }
}

impl StochBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn fastk_period(mut self, n: usize) -> Self {
        self.fastk_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn slowk_period(mut self, n: usize) -> Self {
        self.slowk_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn slowk_ma_type(mut self, t: &'static str) -> Self {
        self.slowk_ma_type = Some(t);
        self
    }
    #[inline(always)]
    pub fn slowd_period(mut self, n: usize) -> Self {
        self.slowd_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn slowd_ma_type(mut self, t: &'static str) -> Self {
        self.slowd_ma_type = Some(t);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<StochOutput, StochError> {
        let p = StochParams {
            fastk_period: self.fastk_period,
            slowk_period: self.slowk_period,
            slowk_ma_type: self.slowk_ma_type.map(|s| s.to_string()),
            slowd_period: self.slowd_period,
            slowd_ma_type: self.slowd_ma_type.map(|s| s.to_string()),
        };
        let i = StochInput::from_candles(c, p);
        stoch_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<StochOutput, StochError> {
        let p = StochParams {
            fastk_period: self.fastk_period,
            slowk_period: self.slowk_period,
            slowk_ma_type: self.slowk_ma_type.map(|s| s.to_string()),
            slowd_period: self.slowd_period,
            slowd_ma_type: self.slowd_ma_type.map(|s| s.to_string()),
        };
        let i = StochInput::from_slices(high, low, close, p);
        stoch_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<StochStream, StochError> {
        let p = StochParams {
            fastk_period: self.fastk_period,
            slowk_period: self.slowk_period,
            slowk_ma_type: self.slowk_ma_type.map(|s| s.to_string()),
            slowd_period: self.slowd_period,
            slowd_ma_type: self.slowd_ma_type.map(|s| s.to_string()),
        };
        StochStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum StochError {
    #[error("stoch: Empty data provided.")]
    EmptyInputData,
    #[error("stoch: Mismatched length.")]
    MismatchedLength,
    #[error("stoch: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("stoch: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("stoch: All values are NaN.")]
    AllValuesNaN,
    #[error("stoch: Output length mismatch: expected {expected}, got {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("stoch: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("stoch: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("stoch: {0}")]
    Other(String),
}

#[inline]
pub fn stoch(input: &StochInput) -> Result<StochOutput, StochError> {
    stoch_with_kernel(input, Kernel::Auto)
}

pub fn stoch_with_kernel(input: &StochInput, kernel: Kernel) -> Result<StochOutput, StochError> {
    let (high, low, close) = match &input.data {
        StochData::Candles { candles } => {
            let high = candles
                .select_candle_field("high")
                .map_err(|e| StochError::Other(e.to_string()))?;
            let low = candles
                .select_candle_field("low")
                .map_err(|e| StochError::Other(e.to_string()))?;
            let close = candles
                .select_candle_field("close")
                .map_err(|e| StochError::Other(e.to_string()))?;
            (high, low, close)
        }
        StochData::Slices { high, low, close } => (*high, *low, *close),
    };

    let data_len = high.len();
    if data_len == 0 || low.is_empty() || close.is_empty() {
        return Err(StochError::EmptyInputData);
    }
    if data_len != low.len() || data_len != close.len() {
        return Err(StochError::MismatchedLength);
    }

    let fastk_period = input.get_fastk_period();
    let slowk_period = input.get_slowk_period();
    let slowd_period = input.get_slowd_period();

    if fastk_period == 0 || fastk_period > data_len {
        return Err(StochError::InvalidPeriod {
            period: fastk_period,
            data_len,
        });
    }
    if slowk_period == 0 || slowk_period > data_len {
        return Err(StochError::InvalidPeriod {
            period: slowk_period,
            data_len,
        });
    }
    if slowd_period == 0 || slowd_period > data_len {
        return Err(StochError::InvalidPeriod {
            period: slowd_period,
            data_len,
        });
    }

    let first_valid_idx = high
        .iter()
        .zip(low.iter())
        .zip(close.iter())
        .position(|((h, l), c)| !h.is_nan() && !l.is_nan() && !c.is_nan())
        .ok_or(StochError::AllValuesNaN)?;

    if (data_len - first_valid_idx) < fastk_period {
        return Err(StochError::NotEnoughValidData {
            needed: fastk_period,
            valid: data_len - first_valid_idx,
        });
    }

    let mut hh = alloc_with_nan_prefix(data_len, first_valid_idx + fastk_period - 1);
    let mut ll = alloc_with_nan_prefix(data_len, first_valid_idx + fastk_period - 1);

    let max_vals = max_rolling(&high[first_valid_idx..], fastk_period)
        .map_err(|e| StochError::Other(e.to_string()))?;
    let min_vals = min_rolling(&low[first_valid_idx..], fastk_period)
        .map_err(|e| StochError::Other(e.to_string()))?;

    for (i, &val) in max_vals.iter().enumerate() {
        hh[i + first_valid_idx] = val;
    }
    for (i, &val) in min_vals.iter().enumerate() {
        ll[i + first_valid_idx] = val;
    }

    let mut k_raw = alloc_with_nan_prefix(data_len, first_valid_idx + fastk_period - 1);

    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => stoch_scalar(
                high,
                low,
                close,
                &hh,
                &ll,
                fastk_period,
                first_valid_idx,
                &mut k_raw,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => stoch_avx2(
                high,
                low,
                close,
                &hh,
                &ll,
                fastk_period,
                first_valid_idx,
                &mut k_raw,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => stoch_avx512(
                high,
                low,
                close,
                &hh,
                &ll,
                fastk_period,
                first_valid_idx,
                &mut k_raw,
            ),
            _ => unreachable!(),
        }
    }

    let slowk_ma_type = input.get_slowk_ma_type();
    let slowd_ma_type = input.get_slowd_ma_type();

    let k_first_valid = first_valid_idx + fastk_period - 1;
    if (slowk_ma_type == "sma" || slowk_ma_type == "SMA")
        && (slowd_ma_type == "sma" || slowd_ma_type == "SMA")
    {
        return stoch_classic_sma(&k_raw, slowk_period, slowd_period, k_first_valid);
    } else if (slowk_ma_type == "ema" || slowk_ma_type == "EMA")
        && (slowd_ma_type == "ema" || slowd_ma_type == "EMA")
    {
        return stoch_classic_ema(&k_raw, slowk_period, slowd_period, k_first_valid);
    }

    let k_vec = ma(&slowk_ma_type, MaData::Slice(&k_raw), slowk_period)
        .map_err(|e| StochError::Other(e.to_string()))?;
    let d_vec = ma(&slowd_ma_type, MaData::Slice(&k_vec), slowd_period)
        .map_err(|e| StochError::Other(e.to_string()))?;
    Ok(StochOutput { k: k_vec, d: d_vec })
}

pub fn stoch_into_slices(
    out_k: &mut [f64],
    out_d: &mut [f64],
    input: &StochInput,
    kernel: Kernel,
) -> Result<(), StochError> {
    let StochOutput { k, d } = stoch_with_kernel(input, kernel)?;
    if out_k.len() != k.len() {
        return Err(StochError::OutputLengthMismatch {
            expected: k.len(),
            got: out_k.len(),
        });
    }
    if out_d.len() != d.len() {
        return Err(StochError::OutputLengthMismatch {
            expected: d.len(),
            got: out_d.len(),
        });
    }
    out_k.copy_from_slice(&k);
    out_d.copy_from_slice(&d);
    Ok(())
}

#[inline]
fn prefill_nan_prefix(dst: &mut [f64], warm: usize) {
    let warm = warm.min(dst.len());
    for v in &mut dst[..warm] {
        *v = f64::from_bits(0x7ff8_0000_0000_0000);
    }
}

#[inline]
fn stoch_compute_into(
    input: &StochInput,
    out_k: &mut [f64],
    out_d: &mut [f64],
    kernel: Kernel,
) -> Result<(), StochError> {
    let (high, low, close) = match &input.data {
        StochData::Candles { candles } => {
            let high = candles
                .select_candle_field("high")
                .map_err(|e| StochError::Other(e.to_string()))?;
            let low = candles
                .select_candle_field("low")
                .map_err(|e| StochError::Other(e.to_string()))?;
            let close = candles
                .select_candle_field("close")
                .map_err(|e| StochError::Other(e.to_string()))?;
            (high, low, close)
        }
        StochData::Slices { high, low, close } => (*high, *low, *close),
    };

    let len = high.len();
    if len == 0 || low.is_empty() || close.is_empty() {
        return Err(StochError::EmptyInputData);
    }
    if len != low.len() || len != close.len() {
        return Err(StochError::MismatchedLength);
    }
    if out_k.len() != len {
        return Err(StochError::OutputLengthMismatch {
            expected: len,
            got: out_k.len(),
        });
    }
    if out_d.len() != len {
        return Err(StochError::OutputLengthMismatch {
            expected: len,
            got: out_d.len(),
        });
    }

    let fastk_period = input.get_fastk_period();
    let slowk_period = input.get_slowk_period();
    let slowd_period = input.get_slowd_period();

    if fastk_period == 0 || fastk_period > len {
        return Err(StochError::InvalidPeriod {
            period: fastk_period,
            data_len: len,
        });
    }
    if slowk_period == 0 || slowk_period > len {
        return Err(StochError::InvalidPeriod {
            period: slowk_period,
            data_len: len,
        });
    }
    if slowd_period == 0 || slowd_period > len {
        return Err(StochError::InvalidPeriod {
            period: slowd_period,
            data_len: len,
        });
    }

    let first = high
        .iter()
        .zip(low.iter())
        .zip(close.iter())
        .position(|((h, l), c)| !h.is_nan() && !l.is_nan() && !c.is_nan())
        .ok_or(StochError::AllValuesNaN)?;

    if (len - first) < fastk_period {
        return Err(StochError::NotEnoughValidData {
            needed: fastk_period,
            valid: len - first,
        });
    }

    let slowk_ma_type = input.get_slowk_ma_type();
    let slowd_ma_type = input.get_slowd_ma_type();
    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    if (slowk_ma_type == "sma" || slowk_ma_type == "SMA")
        && (slowd_ma_type == "sma" || slowd_ma_type == "SMA")
        && matches!(chosen, Kernel::Scalar | Kernel::ScalarBatch)
    {
        return stoch_classic_sma_into_single_pass(
            high,
            low,
            close,
            fastk_period,
            slowk_period,
            slowd_period,
            first,
            out_k,
            out_d,
        );
    }

    let mut hh = alloc_with_nan_prefix(len, first + fastk_period - 1);
    let mut ll = alloc_with_nan_prefix(len, first + fastk_period - 1);
    let highs =
        max_rolling(&high[first..], fastk_period).map_err(|e| StochError::Other(e.to_string()))?;
    let lows =
        min_rolling(&low[first..], fastk_period).map_err(|e| StochError::Other(e.to_string()))?;
    for (i, &v) in highs.iter().enumerate() {
        hh[first + i] = v;
    }
    for (i, &v) in lows.iter().enumerate() {
        ll[first + i] = v;
    }

    let mut k_raw = alloc_with_nan_prefix(len, first + fastk_period - 1);

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                stoch_scalar(high, low, close, &hh, &ll, fastk_period, first, &mut k_raw)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                stoch_avx2(high, low, close, &hh, &ll, fastk_period, first, &mut k_raw)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                stoch_avx512(high, low, close, &hh, &ll, fastk_period, first, &mut k_raw)
            }
            _ => unreachable!(),
        }
    }

    let k_first_valid = first + fastk_period - 1;

    if (slowk_ma_type == "sma" || slowk_ma_type == "SMA")
        && (slowd_ma_type == "sma" || slowd_ma_type == "SMA")
    {
        prefill_nan_prefix(out_k, k_first_valid + slowk_period - 1);
        prefill_nan_prefix(out_d, k_first_valid + slowk_period + slowd_period - 2);

        let mut sum_k = 0.0;
        let k_start = k_first_valid;
        for i in k_start..(k_start + slowk_period).min(len) {
            if !k_raw[i].is_nan() {
                sum_k += k_raw[i];
            }
        }
        if k_start + slowk_period - 1 < len {
            out_k[k_start + slowk_period - 1] = sum_k / slowk_period as f64;
        }
        for i in (k_start + slowk_period)..len {
            let old = k_raw[i - slowk_period];
            let newv = k_raw[i];
            if !old.is_nan() {
                sum_k -= old;
            }
            if !newv.is_nan() {
                sum_k += newv;
            }
            out_k[i] = sum_k / slowk_period as f64;
        }

        let mut sum_d = 0.0;
        let d_start = k_first_valid + slowk_period - 1;
        for i in d_start..(d_start + slowd_period).min(len) {
            if !out_k[i].is_nan() {
                sum_d += out_k[i];
            }
        }
        if d_start + slowd_period - 1 < len {
            out_d[d_start + slowd_period - 1] = sum_d / slowd_period as f64;
        }
        for i in (d_start + slowd_period)..len {
            let old = out_k[i - slowd_period];
            let newv = out_k[i];
            if !old.is_nan() {
                sum_d -= old;
            }
            if !newv.is_nan() {
                sum_d += newv;
            }
            out_d[i] = sum_d / slowd_period as f64;
        }
        return Ok(());
    }

    if (slowk_ma_type == "ema" || slowk_ma_type == "EMA")
        && (slowd_ma_type == "ema" || slowd_ma_type == "EMA")
    {
        prefill_nan_prefix(out_k, k_first_valid + slowk_period - 1);
        prefill_nan_prefix(out_d, k_first_valid + slowk_period + slowd_period - 2);

        let alpha_k = 2.0 / (slowk_period as f64 + 1.0);
        let one_minus_alpha_k = 1.0 - alpha_k;
        let k_warm = k_first_valid + slowk_period - 1;
        let mut sum_k = 0.0;
        let mut cnt_k = 0;
        for i in k_first_valid..(k_first_valid + slowk_period).min(len) {
            if !k_raw[i].is_nan() {
                sum_k += k_raw[i];
                cnt_k += 1;
            }
        }
        if cnt_k > 0 && k_warm < len {
            let mut ema_k = sum_k / cnt_k as f64;
            out_k[k_warm] = ema_k;
            for i in (k_warm + 1)..len {
                if !k_raw[i].is_nan() {
                    ema_k = alpha_k * k_raw[i] + one_minus_alpha_k * ema_k;
                }
                out_k[i] = ema_k;
            }
        } else {
            for i in k_warm..len {
                out_k[i] = f64::from_bits(0x7ff8_0000_0000_0000);
            }
        }

        let alpha_d = 2.0 / (slowd_period as f64 + 1.0);
        let one_minus_alpha_d = 1.0 - alpha_d;
        let d_warm = k_first_valid + slowk_period + slowd_period - 2;
        let d_start = k_first_valid + slowk_period - 1;
        let mut sum_d = 0.0;
        let mut cnt_d = 0;
        for i in d_start..(d_start + slowd_period).min(len) {
            if !out_k[i].is_nan() {
                sum_d += out_k[i];
                cnt_d += 1;
            }
        }
        if cnt_d > 0 && d_warm < len {
            let mut ema_d = sum_d / cnt_d as f64;
            out_d[d_warm] = ema_d;
            for i in (d_warm + 1)..len {
                if !out_k[i].is_nan() {
                    ema_d = alpha_d * out_k[i] + one_minus_alpha_d * ema_d;
                }
                out_d[i] = ema_d;
            }
        } else {
            for i in d_warm..len {
                out_d[i] = f64::from_bits(0x7ff8_0000_0000_0000);
            }
        }
        return Ok(());
    }

    let k_vec = ma(&slowk_ma_type, MaData::Slice(&k_raw), slowk_period)
        .map_err(|e| StochError::Other(e.to_string()))?;
    let d_vec = ma(&slowd_ma_type, MaData::Slice(&k_vec), slowd_period)
        .map_err(|e| StochError::Other(e.to_string()))?;
    out_k.copy_from_slice(&k_vec);
    out_d.copy_from_slice(&d_vec);
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn stoch_into(
    input: &StochInput,
    out_k: &mut [f64],
    out_d: &mut [f64],
) -> Result<(), StochError> {
    #[cfg(test)]
    {
        stoch_into_slices(out_k, out_d, input, Kernel::Auto)
    }
    #[cfg(not(test))]
    {
        stoch_compute_into(input, out_k, out_d, Kernel::Auto)
    }
}

fn stoch_classic_sma_into_single_pass(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    fastk_period: usize,
    slowk_period: usize,
    slowd_period: usize,
    first: usize,
    out_k: &mut [f64],
    out_d: &mut [f64],
) -> Result<(), StochError> {
    let len = close.len();

    let k_first_valid = first + fastk_period - 1;
    let k_warm = k_first_valid + slowk_period - 1;
    let d_warm = k_first_valid + slowk_period + slowd_period - 2;

    prefill_nan_prefix(out_k, k_warm);
    prefill_nan_prefix(out_d, d_warm);

    let mut trail = first;
    let mut maxi = first;
    let mut mini = first;
    let mut max = high[first];
    let mut min = low[first];

    let mut k_buf = vec![0.0f64; slowk_period];
    let mut k_pos: usize = 0;
    let mut k_sum = 0.0f64;
    let mut k_count: usize = 0;

    let mut d_buf = vec![0.0f64; slowd_period];
    let mut d_pos: usize = 0;
    let mut d_sum = 0.0f64;
    let mut d_count: usize = 0;

    const SCALE: f64 = 100.0;
    const EPS: f64 = f64::EPSILON;

    for i in first..len {
        if i >= first + fastk_period {
            trail += 1;
        }

        let bar_h = high[i];
        if maxi < trail {
            maxi = trail;
            max = high[maxi];
            let mut j = trail;
            while j < i {
                j += 1;
                let v = high[j];
                if v >= max {
                    max = v;
                    maxi = j;
                }
            }
        } else if bar_h >= max {
            maxi = i;
            max = bar_h;
        }

        let bar_l = low[i];
        if mini < trail {
            mini = trail;
            min = low[mini];
            let mut j = trail;
            while j < i {
                j += 1;
                let v = low[j];
                if v <= min {
                    min = v;
                    mini = j;
                }
            }
        } else if bar_l <= min {
            mini = i;
            min = bar_l;
        }

        if i < k_first_valid {
            continue;
        }

        let c = close[i];
        let denom = max - min;
        let k_raw = if denom.abs() < EPS {
            50.0
        } else {
            (c - min).mul_add(SCALE / denom, 0.0)
        };

        if k_count >= slowk_period {
            k_sum -= k_buf[k_pos];
        }
        k_buf[k_pos] = k_raw;
        k_sum += k_raw;
        k_count += 1;
        k_pos += 1;
        if k_pos == slowk_period {
            k_pos = 0;
        }

        if i >= k_warm {
            let k_sma = k_sum / slowk_period as f64;
            out_k[i] = k_sma;

            if d_count >= slowd_period {
                d_sum -= d_buf[d_pos];
            }
            d_buf[d_pos] = k_sma;
            d_sum += k_sma;
            d_count += 1;
            d_pos += 1;
            if d_pos == slowd_period {
                d_pos = 0;
            }

            if i >= d_warm {
                out_d[i] = d_sum / slowd_period as f64;
            }
        }
    }

    Ok(())
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn stoch_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    hh: &[f64],
    ll: &[f64],
    fastk_period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    if fastk_period <= 32 {
        unsafe { stoch_avx512_short(high, low, close, hh, ll, fastk_period, first_valid, out) }
    } else {
        unsafe { stoch_avx512_long(high, low, close, hh, ll, fastk_period, first_valid, out) }
    }
}

#[inline]
pub fn stoch_scalar(
    _high: &[f64],
    _low: &[f64],
    close: &[f64],
    hh: &[f64],
    ll: &[f64],
    fastk_period: usize,
    first_val: usize,
    out: &mut [f64],
) {
    let start = first_val + fastk_period - 1;
    if start >= close.len() {
        return;
    }

    const SCALE: f64 = 100.0;
    const EPS: f64 = f64::EPSILON;

    let c = &close[start..];
    let h = &hh[start..];
    let l = &ll[start..];
    let outv = &mut out[start..];

    for (o, (&cv, (&hv, &lv))) in outv.iter_mut().zip(c.iter().zip(h.iter().zip(l.iter()))) {
        let d = hv - lv;
        *o = if d.abs() < EPS {
            50.0
        } else {
            (cv - lv).mul_add(SCALE / d, 0.0)
        };
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn stoch_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    hh: &[f64],
    ll: &[f64],
    fastk_period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    unsafe { stoch_avx2_impl(high, low, close, hh, ll, fastk_period, first_valid, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn stoch_avx2_impl(
    _high: &[f64],
    _low: &[f64],
    close: &[f64],
    hh: &[f64],
    ll: &[f64],
    fastk_period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    let start = first_valid + fastk_period - 1;
    if start >= close.len() {
        return;
    }

    let n = close.len() - start;
    let mut i = 0usize;

    let c_ptr = close.as_ptr().add(start);
    let h_ptr = hh.as_ptr().add(start);
    let l_ptr = ll.as_ptr().add(start);
    let o_ptr = out.as_mut_ptr().add(start);

    const STEP: usize = 4;
    let vec_end = n & !(STEP - 1);

    let scale = _mm256_set1_pd(100.0);
    let fifty = _mm256_set1_pd(50.0);
    let eps = _mm256_set1_pd(f64::EPSILON);
    let sign_mask = _mm256_set1_pd(-0.0);

    while i + STEP <= vec_end {
        let c0 = _mm256_loadu_pd(c_ptr.add(i));
        let h0 = _mm256_loadu_pd(h_ptr.add(i));
        let l0 = _mm256_loadu_pd(l_ptr.add(i));
        let d0 = _mm256_sub_pd(h0, l0);
        let n0 = _mm256_sub_pd(c0, l0);
        let a0 = _mm256_andnot_pd(sign_mask, d0);
        let m0 = _mm256_cmp_pd(a0, eps, _CMP_LT_OQ);
        let inv0 = _mm256_div_pd(scale, d0);
        let v0 = _mm256_mul_pd(n0, inv0);
        let o0 = _mm256_blendv_pd(v0, fifty, m0);

        if i + 2 * STEP <= vec_end {
            let c1 = _mm256_loadu_pd(c_ptr.add(i + STEP));
            let h1 = _mm256_loadu_pd(h_ptr.add(i + STEP));
            let l1 = _mm256_loadu_pd(l_ptr.add(i + STEP));
            let d1 = _mm256_sub_pd(h1, l1);
            let n1 = _mm256_sub_pd(c1, l1);
            let a1 = _mm256_andnot_pd(sign_mask, d1);
            let m1 = _mm256_cmp_pd(a1, eps, _CMP_LT_OQ);
            let inv1 = _mm256_div_pd(scale, d1);
            let v1 = _mm256_mul_pd(n1, inv1);
            let o1 = _mm256_blendv_pd(v1, fifty, m1);

            _mm256_storeu_pd(o_ptr.add(i), o0);
            _mm256_storeu_pd(o_ptr.add(i + STEP), o1);
            i += 2 * STEP;
        } else {
            _mm256_storeu_pd(o_ptr.add(i), o0);
            i += STEP;
        }
    }

    while i < n {
        let c = *c_ptr.add(i);
        let l = *l_ptr.add(i);
        let d = *h_ptr.add(i) - l;
        *o_ptr.add(i) = if d.abs() < f64::EPSILON {
            50.0
        } else {
            (c - l) * (100.0 / d)
        };
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn stoch_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    hh: &[f64],
    ll: &[f64],
    fastk_period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    unsafe { stoch_avx512_impl(high, low, close, hh, ll, fastk_period, first_valid, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn stoch_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    hh: &[f64],
    ll: &[f64],
    fastk_period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    unsafe { stoch_avx512_impl(high, low, close, hh, ll, fastk_period, first_valid, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn stoch_avx512_impl(
    _high: &[f64],
    _low: &[f64],
    close: &[f64],
    hh: &[f64],
    ll: &[f64],
    fastk_period: usize,
    first_valid: usize,
    out: &mut [f64],
) {
    let start = first_valid + fastk_period - 1;
    if start >= close.len() {
        return;
    }

    let n = close.len() - start;

    let c_ptr = close.as_ptr().add(start);
    let h_ptr = hh.as_ptr().add(start);
    let l_ptr = ll.as_ptr().add(start);
    let o_ptr = out.as_mut_ptr().add(start);

    const STEP: usize = 8;
    let vec_end = n & !(STEP - 1);

    let scale = _mm512_set1_pd(100.0);
    let fifty = _mm512_set1_pd(50.0);
    let eps = _mm512_set1_pd(f64::EPSILON);
    let sign_mask = _mm512_set1_pd(-0.0);

    let mut i = 0usize;
    while i + STEP <= vec_end {
        let c0 = _mm512_loadu_pd(c_ptr.add(i));
        let h0 = _mm512_loadu_pd(h_ptr.add(i));
        let l0 = _mm512_loadu_pd(l_ptr.add(i));
        let d0 = _mm512_sub_pd(h0, l0);
        let n0 = _mm512_sub_pd(c0, l0);
        let a0 = _mm512_andnot_pd(sign_mask, d0);
        let m0: __mmask8 = _mm512_cmp_pd_mask(a0, eps, _CMP_LT_OQ);
        let inv0 = _mm512_div_pd(scale, d0);
        let v0 = _mm512_mul_pd(n0, inv0);
        let o0 = _mm512_mask_blend_pd(m0, v0, fifty);

        if i + 2 * STEP <= vec_end {
            let c1 = _mm512_loadu_pd(c_ptr.add(i + STEP));
            let h1 = _mm512_loadu_pd(h_ptr.add(i + STEP));
            let l1 = _mm512_loadu_pd(l_ptr.add(i + STEP));
            let d1 = _mm512_sub_pd(h1, l1);
            let n1 = _mm512_sub_pd(c1, l1);
            let a1 = _mm512_andnot_pd(sign_mask, d1);
            let m1: __mmask8 = _mm512_cmp_pd_mask(a1, eps, _CMP_LT_OQ);
            let inv1 = _mm512_div_pd(scale, d1);
            let v1 = _mm512_mul_pd(n1, inv1);
            let o1 = _mm512_mask_blend_pd(m1, v1, fifty);

            _mm512_storeu_pd(o_ptr.add(i), o0);
            _mm512_storeu_pd(o_ptr.add(i + STEP), o1);
            i += 2 * STEP;
        } else {
            _mm512_storeu_pd(o_ptr.add(i), o0);
            i += STEP;
        }
    }

    while i < n {
        let c = *c_ptr.add(i);
        let l = *l_ptr.add(i);
        let d = *h_ptr.add(i) - l;
        *o_ptr.add(i) = if d.abs() < f64::EPSILON {
            50.0
        } else {
            (c - l) * (100.0 / d)
        };
        i += 1;
    }
}

#[derive(Clone, Debug)]
pub struct StochBatchRange {
    pub fastk_period: (usize, usize, usize),
    pub slowk_period: (usize, usize, usize),
    pub slowk_ma_type: (String, String, f64),
    pub slowd_period: (usize, usize, usize),
    pub slowd_ma_type: (String, String, f64),
}

impl Default for StochBatchRange {
    fn default() -> Self {
        Self {
            fastk_period: (14, 263, 1),
            slowk_period: (3, 3, 0),
            slowk_ma_type: ("sma".to_string(), "sma".to_string(), 0.0),
            slowd_period: (3, 3, 0),
            slowd_ma_type: ("sma".to_string(), "sma".to_string(), 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct StochBatchBuilder {
    range: StochBatchRange,
    kernel: Kernel,
}

impl StochBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn fastk_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.fastk_period = (start, end, step);
        self
    }
    pub fn fastk_period_static(mut self, p: usize) -> Self {
        self.range.fastk_period = (p, p, 0);
        self
    }
    pub fn slowk_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.slowk_period = (start, end, step);
        self
    }
    pub fn slowk_period_static(mut self, p: usize) -> Self {
        self.range.slowk_period = (p, p, 0);
        self
    }
    pub fn slowk_ma_type_static(mut self, t: &str) -> Self {
        self.range.slowk_ma_type = (t.to_string(), t.to_string(), 0.0);
        self
    }
    pub fn slowd_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.slowd_period = (start, end, step);
        self
    }
    pub fn slowd_period_static(mut self, p: usize) -> Self {
        self.range.slowd_period = (p, p, 0);
        self
    }
    pub fn slowd_ma_type_static(mut self, t: &str) -> Self {
        self.range.slowd_ma_type = (t.to_string(), t.to_string(), 0.0);
        self
    }

    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<StochBatchOutput, StochError> {
        stoch_batch_with_kernel(high, low, close, &self.range, self.kernel)
    }
    pub fn apply_candles(self, c: &Candles) -> Result<StochBatchOutput, StochError> {
        let high = source_type(c, "high");
        let low = source_type(c, "low");
        let close = source_type(c, "close");
        self.apply_slices(high, low, close)
    }
}

pub fn stoch_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &StochBatchRange,
    k: Kernel,
) -> Result<StochBatchOutput, StochError> {
    let kernel = match k {
        Kernel::Auto => Kernel::ScalarBatch,
        other if other.is_batch() => other,
        other => return Err(StochError::InvalidKernelForBatch(other)),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    stoch_batch_par_slice(high, low, close, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct StochBatchOutput {
    pub k: Vec<f64>,
    pub d: Vec<f64>,
    pub combos: Vec<StochParams>,
    pub rows: usize,
    pub cols: usize,
}
impl StochBatchOutput {
    pub fn row_for_params(&self, p: &StochParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.fastk_period == p.fastk_period
                && c.slowk_period == p.slowk_period
                && c.slowk_ma_type == p.slowk_ma_type
                && c.slowd_period == p.slowd_period
                && c.slowd_ma_type == p.slowd_ma_type
        })
    }
    pub fn values_for(&self, p: &StochParams) -> Option<(&[f64], &[f64])> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            (
                &self.k[start..start + self.cols],
                &self.d[start..start + self.cols],
            )
        })
    }
}

#[inline(always)]
fn expand_grid(r: &StochBatchRange) -> Result<Vec<StochParams>, StochError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, StochError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }

        let mut v = Vec::new();
        if start < end {
            let mut x = start;
            loop {
                v.push(x);
                match x.checked_add(step) {
                    Some(next) if next <= end => x = next,
                    Some(_) | None => break,
                }
            }
        } else {
            let mut x = start;
            loop {
                v.push(x);
                match x.checked_sub(step) {
                    Some(next) if next >= end => x = next,
                    Some(_) | None => break,
                }
            }
        }

        if v.is_empty() {
            Err(StochError::InvalidRange { start, end, step })
        } else {
            Ok(v)
        }
    }
    fn axis_str((start, end, _): (String, String, f64)) -> Vec<String> {
        if start == end {
            vec![start]
        } else {
            vec![start, end]
        }
    }
    let fastk_periods = axis_usize(r.fastk_period)?;
    let slowk_periods = axis_usize(r.slowk_period)?;
    let slowk_types = axis_str(r.slowk_ma_type.clone());
    let slowd_periods = axis_usize(r.slowd_period)?;
    let slowd_types = axis_str(r.slowd_ma_type.clone());

    let combos_len = fastk_periods
        .len()
        .checked_mul(slowk_periods.len())
        .and_then(|v| v.checked_mul(slowk_types.len()))
        .and_then(|v| v.checked_mul(slowd_periods.len()))
        .and_then(|v| v.checked_mul(slowd_types.len()))
        .ok_or(StochError::InvalidRange {
            start: r.fastk_period.0,
            end: r.fastk_period.1,
            step: r.fastk_period.2,
        })?;

    let mut out = Vec::with_capacity(combos_len);
    for &fkp in &fastk_periods {
        for &skp in &slowk_periods {
            for skt in &slowk_types {
                for &sdp in &slowd_periods {
                    for sdt in &slowd_types {
                        out.push(StochParams {
                            fastk_period: Some(fkp),
                            slowk_period: Some(skp),
                            slowk_ma_type: Some(skt.clone()),
                            slowd_period: Some(sdp),
                            slowd_ma_type: Some(sdt.clone()),
                        });
                    }
                }
            }
        }
    }
    Ok(out)
}

#[inline(always)]
pub fn stoch_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &StochBatchRange,
    kern: Kernel,
) -> Result<StochBatchOutput, StochError> {
    stoch_batch_inner(high, low, close, sweep, kern, false)
}

#[inline(always)]
pub fn stoch_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &StochBatchRange,
    kern: Kernel,
) -> Result<StochBatchOutput, StochError> {
    stoch_batch_inner(high, low, close, sweep, kern, true)
}

#[inline(always)]
fn stoch_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &StochBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<StochBatchOutput, StochError> {
    let combos = expand_grid(sweep)?;

    let n = high.len();
    if n == 0 || low.len() != n || close.len() != n {
        return Err(StochError::MismatchedLength);
    }

    let first = high
        .iter()
        .zip(low.iter())
        .zip(close.iter())
        .position(|((h, l), c)| !h.is_nan() && !l.is_nan() && !c.is_nan())
        .ok_or(StochError::AllValuesNaN)?;
    let max_fkp = combos
        .iter()
        .map(|c| c.fastk_period.unwrap())
        .max()
        .unwrap();
    if n - first < max_fkp {
        return Err(StochError::NotEnoughValidData {
            needed: max_fkp,
            valid: n - first,
        });
    }

    let rows = combos.len();
    let cols = n;

    rows.checked_mul(cols).ok_or(StochError::InvalidRange {
        start: rows,
        end: cols,
        step: 0,
    })?;

    let mut k_mu = make_uninit_matrix(rows, cols);
    let mut d_mu = make_uninit_matrix(rows, cols);

    let warm_k: Vec<usize> = combos
        .iter()
        .map(|c| first + c.fastk_period.unwrap() - 1)
        .collect();
    init_matrix_prefixes(&mut k_mu, cols, &warm_k);
    init_matrix_prefixes(&mut d_mu, cols, &warm_k);

    let mut k_guard = core::mem::ManuallyDrop::new(k_mu);
    let mut d_guard = core::mem::ManuallyDrop::new(d_mu);
    let k_mat: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(k_guard.as_mut_ptr() as *mut f64, k_guard.len()) };
    let d_mat: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(d_guard.as_mut_ptr() as *mut f64, d_guard.len()) };

    use std::collections::HashMap;
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for (row, prm) in combos.iter().enumerate() {
        groups
            .entry(prm.fastk_period.unwrap())
            .or_default()
            .push(row);
    }

    let mut compute_k_raw = |fkp: usize| -> Vec<f64> {
        let mut hh = alloc_with_nan_prefix(cols, first + fkp - 1);
        let mut ll = alloc_with_nan_prefix(cols, first + fkp - 1);
        let highs = max_rolling(&high[first..], fkp).unwrap();
        let lows = min_rolling(&low[first..], fkp).unwrap();
        for (i, &v) in highs.iter().enumerate() {
            hh[first + i] = v;
        }
        for (i, &v) in lows.iter().enumerate() {
            ll[first + i] = v;
        }

        let mut k_raw = alloc_with_nan_prefix(cols, first + fkp - 1);
        unsafe {
            match kern {
                Kernel::Scalar => {
                    stoch_row_scalar(high, low, close, &hh, &ll, fkp, first, &mut k_raw)
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2 => stoch_row_avx2(high, low, close, &hh, &ll, fkp, first, &mut k_raw),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512 => {
                    stoch_row_avx512(high, low, close, &hh, &ll, fkp, first, &mut k_raw)
                }
                _ => unreachable!(),
            }
        }
        k_raw
    };

    for (fkp, rows_in_group) in groups {
        let k_raw = compute_k_raw(fkp);
        for &row in &rows_in_group {
            let prm = &combos[row];
            let k_vec = ma(
                prm.slowk_ma_type.as_ref().unwrap(),
                MaData::Slice(&k_raw),
                prm.slowk_period.unwrap(),
            )
            .unwrap();
            let d_vec = ma(
                prm.slowd_ma_type.as_ref().unwrap(),
                MaData::Slice(&k_vec),
                prm.slowd_period.unwrap(),
            )
            .unwrap();
            let start = row * cols;
            let dst_k = &mut k_mat[start..start + cols];
            let dst_d = &mut d_mat[start..start + cols];
            dst_k.copy_from_slice(&k_vec);
            dst_d.copy_from_slice(&d_vec);
        }
    }

    let k = unsafe {
        Vec::from_raw_parts(
            k_guard.as_mut_ptr() as *mut f64,
            k_guard.len(),
            k_guard.capacity(),
        )
    };
    let d = unsafe {
        Vec::from_raw_parts(
            d_guard.as_mut_ptr() as *mut f64,
            d_guard.len(),
            d_guard.capacity(),
        )
    };
    core::mem::forget(k_guard);
    core::mem::forget(d_guard);

    Ok(StochBatchOutput {
        k,
        d,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn stoch_row_scalar(
    _high: &[f64],
    _low: &[f64],
    close: &[f64],
    hh: &[f64],
    ll: &[f64],
    fastk_period: usize,
    first: usize,
    out: &mut [f64],
) {
    let start = first + fastk_period - 1;
    if start >= close.len() {
        return;
    }

    const SCALE: f64 = 100.0;
    const EPS: f64 = f64::EPSILON;

    let c = &close[start..];
    let h = &hh[start..];
    let l = &ll[start..];
    let outv = &mut out[start..];

    for (o, (&cv, (&hv, &lv))) in outv.iter_mut().zip(c.iter().zip(h.iter().zip(l.iter()))) {
        let d = hv - lv;
        *o = if d.abs() < EPS {
            50.0
        } else {
            (cv - lv).mul_add(SCALE / d, 0.0)
        };
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn stoch_row_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    hh: &[f64],
    ll: &[f64],
    fastk_period: usize,
    first: usize,
    out: &mut [f64],
) {
    stoch_row_avx2_impl(high, low, close, hh, ll, fastk_period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn stoch_row_avx2_impl(
    _high: &[f64],
    _low: &[f64],
    close: &[f64],
    hh: &[f64],
    ll: &[f64],
    fastk_period: usize,
    first: usize,
    out: &mut [f64],
) {
    let start = first + fastk_period - 1;
    if start >= close.len() {
        return;
    }
    let n = close.len() - start;

    let mut i = 0usize;
    let c_ptr = close.as_ptr().add(start);
    let h_ptr = hh.as_ptr().add(start);
    let l_ptr = ll.as_ptr().add(start);
    let o_ptr = out.as_mut_ptr().add(start);

    const STEP: usize = 4;
    let vec_end = n & !(STEP - 1);

    let scale = _mm256_set1_pd(100.0);
    let fifty = _mm256_set1_pd(50.0);
    let eps = _mm256_set1_pd(f64::EPSILON);
    let sign_mask = _mm256_set1_pd(-0.0);

    while i + STEP <= vec_end {
        let c0 = _mm256_loadu_pd(c_ptr.add(i));
        let h0 = _mm256_loadu_pd(h_ptr.add(i));
        let l0 = _mm256_loadu_pd(l_ptr.add(i));
        let d0 = _mm256_sub_pd(h0, l0);
        let n0 = _mm256_sub_pd(c0, l0);
        let a0 = _mm256_andnot_pd(sign_mask, d0);
        let m0 = _mm256_cmp_pd(a0, eps, _CMP_LT_OQ);
        let inv0 = _mm256_div_pd(scale, d0);
        let v0 = _mm256_mul_pd(n0, inv0);
        let o0 = _mm256_blendv_pd(v0, fifty, m0);

        if i + 2 * STEP <= vec_end {
            let c1 = _mm256_loadu_pd(c_ptr.add(i + STEP));
            let h1 = _mm256_loadu_pd(h_ptr.add(i + STEP));
            let l1 = _mm256_loadu_pd(l_ptr.add(i + STEP));
            let d1 = _mm256_sub_pd(h1, l1);
            let n1 = _mm256_sub_pd(c1, l1);
            let a1 = _mm256_andnot_pd(sign_mask, d1);
            let m1 = _mm256_cmp_pd(a1, eps, _CMP_LT_OQ);
            let inv1 = _mm256_div_pd(scale, d1);
            let v1 = _mm256_mul_pd(n1, inv1);
            let o1 = _mm256_blendv_pd(v1, fifty, m1);

            _mm256_storeu_pd(o_ptr.add(i), o0);
            _mm256_storeu_pd(o_ptr.add(i + STEP), o1);
            i += 2 * STEP;
        } else {
            _mm256_storeu_pd(o_ptr.add(i), o0);
            i += STEP;
        }
    }

    while i < n {
        let c = *c_ptr.add(i);
        let l = *l_ptr.add(i);
        let d = *h_ptr.add(i) - l;
        *o_ptr.add(i) = if d.abs() < f64::EPSILON {
            50.0
        } else {
            (c - l) * (100.0 / d)
        };
        i += 1;
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn stoch_row_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    hh: &[f64],
    ll: &[f64],
    fastk_period: usize,
    first: usize,
    out: &mut [f64],
) {
    if fastk_period <= 32 {
        stoch_row_avx512_short(high, low, close, hh, ll, fastk_period, first, out)
    } else {
        stoch_row_avx512_long(high, low, close, hh, ll, fastk_period, first, out)
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn stoch_row_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    hh: &[f64],
    ll: &[f64],
    fastk_period: usize,
    first: usize,
    out: &mut [f64],
) {
    stoch_row_avx512_impl(high, low, close, hh, ll, fastk_period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn stoch_row_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    hh: &[f64],
    ll: &[f64],
    fastk_period: usize,
    first: usize,
    out: &mut [f64],
) {
    stoch_row_avx512_impl(high, low, close, hh, ll, fastk_period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn stoch_row_avx512_impl(
    _high: &[f64],
    _low: &[f64],
    close: &[f64],
    hh: &[f64],
    ll: &[f64],
    fastk_period: usize,
    first: usize,
    out: &mut [f64],
) {
    let start = first + fastk_period - 1;
    if start >= close.len() {
        return;
    }
    let n = close.len() - start;

    let c_ptr = close.as_ptr().add(start);
    let h_ptr = hh.as_ptr().add(start);
    let l_ptr = ll.as_ptr().add(start);
    let o_ptr = out.as_mut_ptr().add(start);

    const STEP: usize = 8;
    let vec_end = n & !(STEP - 1);

    let scale = _mm512_set1_pd(100.0);
    let fifty = _mm512_set1_pd(50.0);
    let eps = _mm512_set1_pd(f64::EPSILON);
    let sign_mask = _mm512_set1_pd(-0.0);

    let mut i = 0usize;
    while i + STEP <= vec_end {
        let c0 = _mm512_loadu_pd(c_ptr.add(i));
        let h0 = _mm512_loadu_pd(h_ptr.add(i));
        let l0 = _mm512_loadu_pd(l_ptr.add(i));
        let d0 = _mm512_sub_pd(h0, l0);
        let n0 = _mm512_sub_pd(c0, l0);
        let a0 = _mm512_andnot_pd(sign_mask, d0);
        let m0: __mmask8 = _mm512_cmp_pd_mask(a0, eps, _CMP_LT_OQ);
        let inv0 = _mm512_div_pd(scale, d0);
        let v0 = _mm512_mul_pd(n0, inv0);
        let o0 = _mm512_mask_blend_pd(m0, v0, fifty);

        if i + 2 * STEP <= vec_end {
            let c1 = _mm512_loadu_pd(c_ptr.add(i + STEP));
            let h1 = _mm512_loadu_pd(h_ptr.add(i + STEP));
            let l1 = _mm512_loadu_pd(l_ptr.add(i + STEP));
            let d1 = _mm512_sub_pd(h1, l1);
            let n1 = _mm512_sub_pd(c1, l1);
            let a1 = _mm512_andnot_pd(sign_mask, d1);
            let m1: __mmask8 = _mm512_cmp_pd_mask(a1, eps, _CMP_LT_OQ);
            let inv1 = _mm512_div_pd(scale, d1);
            let v1 = _mm512_mul_pd(n1, inv1);
            let o1 = _mm512_mask_blend_pd(m1, v1, fifty);

            _mm512_storeu_pd(o_ptr.add(i), o0);
            _mm512_storeu_pd(o_ptr.add(i + STEP), o1);
            i += 2 * STEP;
        } else {
            _mm512_storeu_pd(o_ptr.add(i), o0);
            i += STEP;
        }
    }

    while i < n {
        let c = *c_ptr.add(i);
        let l = *l_ptr.add(i);
        let d = *h_ptr.add(i) - l;
        *o_ptr.add(i) = if d.abs() < f64::EPSILON {
            50.0
        } else {
            (c - l) * (100.0 / d)
        };
        i += 1;
    }
}

#[derive(Debug, Clone)]
struct DeqEntry {
    val: f64,
    idx: usize,
}

#[derive(Debug, Clone)]
pub struct StochStream {
    fastk_period: usize,
    slowk_period: usize,
    slowk_ma_type: String,
    slowd_period: usize,
    slowd_ma_type: String,

    maxq: VecDeque<DeqEntry>,
    minq: VecDeque<DeqEntry>,
    t: usize,
    have_window: bool,

    k_sma_buf: Vec<f64>,
    k_sma_sum: f64,
    k_sma_head: usize,
    k_sma_count: usize,

    k_ema: Option<f64>,
    k_ema_seed_sum: f64,
    k_ema_seed_count: usize,
    alpha_k: f64,

    d_sma_buf: Vec<f64>,
    d_sma_sum: f64,
    d_sma_head: usize,
    d_sma_count: usize,

    d_ema: Option<f64>,
    d_ema_seed_sum: f64,
    d_ema_seed_count: usize,
    alpha_d: f64,

    k_stream: Option<Vec<f64>>,
    d_stream: Option<Vec<f64>>,
}

impl StochStream {
    pub fn try_new(params: StochParams) -> Result<Self, StochError> {
        let fastk_period = params.fastk_period.unwrap_or(14);
        let slowk_period = params.slowk_period.unwrap_or(3);
        let slowd_period = params.slowd_period.unwrap_or(3);
        if fastk_period == 0 || slowk_period == 0 || slowd_period == 0 {
            return Err(StochError::InvalidPeriod {
                period: 0,
                data_len: 0,
            });
        }

        let slowk_ma_type = params.slowk_ma_type.unwrap_or_else(|| "sma".to_string());
        let slowd_ma_type = params.slowd_ma_type.unwrap_or_else(|| "sma".to_string());

        let alpha_k = 2.0 / (slowk_period as f64 + 1.0);
        let alpha_d = 2.0 / (slowd_period as f64 + 1.0);

        Ok(Self {
            fastk_period,
            slowk_period,
            slowk_ma_type,
            slowd_period,
            slowd_ma_type,

            maxq: VecDeque::with_capacity(fastk_period),
            minq: VecDeque::with_capacity(fastk_period),
            t: 0,
            have_window: false,

            k_sma_buf: vec![f64::NAN; slowk_period.max(1)],
            k_sma_sum: 0.0,
            k_sma_head: 0,
            k_sma_count: 0,

            k_ema: None,
            k_ema_seed_sum: 0.0,
            k_ema_seed_count: 0,
            alpha_k,

            d_sma_buf: vec![f64::NAN; slowd_period.max(1)],
            d_sma_sum: 0.0,
            d_sma_head: 0,
            d_sma_count: 0,

            d_ema: None,
            d_ema_seed_sum: 0.0,
            d_ema_seed_count: 0,
            alpha_d,

            k_stream: None,
            d_stream: None,
        })
    }

    #[inline(always)]
    fn evict_older_than(dq: &mut VecDeque<DeqEntry>, min_idx: usize) {
        while let Some(front) = dq.front() {
            if front.idx < min_idx {
                dq.pop_front();
            } else {
                break;
            }
        }
    }

    #[inline(always)]
    fn push_maxq(&mut self, val: f64, idx: usize) {
        while let Some(back) = self.maxq.back() {
            if back.val <= val {
                self.maxq.pop_back();
            } else {
                break;
            }
        }
        self.maxq.push_back(DeqEntry { val, idx });
    }

    #[inline(always)]
    fn push_minq(&mut self, val: f64, idx: usize) {
        while let Some(back) = self.minq.back() {
            if back.val >= val {
                self.minq.pop_back();
            } else {
                break;
            }
        }
        self.minq.push_back(DeqEntry { val, idx });
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64)> {
        if !high.is_finite() || !low.is_finite() || !close.is_finite() {
            return None;
        }

        let idx = self.t;
        self.t = self.t.wrapping_add(1);

        self.push_maxq(high, idx);
        self.push_minq(low, idx);

        let seen = idx + 1;
        if seen >= self.fastk_period {
            let window_start = seen - self.fastk_period;
            Self::evict_older_than(&mut self.maxq, window_start);
            Self::evict_older_than(&mut self.minq, window_start);
            self.have_window = true;
        }

        if !self.have_window {
            return None;
        }

        debug_assert!(!self.maxq.is_empty() && !self.minq.is_empty());
        let hh = self.maxq.front().unwrap().val;
        let ll = self.minq.front().unwrap().val;

        const SCALE: f64 = 100.0;
        const EPS: f64 = f64::EPSILON;

        let denom = hh - ll;
        let k_raw = if denom.abs() < EPS {
            50.0
        } else {
            (close - ll).mul_add(SCALE / denom, 0.0)
        };

        let k_last = if self.slowk_ma_type.eq_ignore_ascii_case("sma") {
            if self.slowk_period == 1 {
                k_raw
            } else if self.k_sma_count < self.slowk_period {
                self.k_sma_sum += k_raw;
                self.k_sma_buf[self.k_sma_head] = k_raw;
                self.k_sma_head = (self.k_sma_head + 1) % self.slowk_period;
                self.k_sma_count += 1;
                if self.k_sma_count == self.slowk_period {
                    self.k_sma_sum / self.slowk_period as f64
                } else {
                    f64::NAN
                }
            } else {
                let old = self.k_sma_buf[self.k_sma_head];
                self.k_sma_sum += k_raw - old;
                self.k_sma_buf[self.k_sma_head] = k_raw;
                self.k_sma_head = (self.k_sma_head + 1) % self.slowk_period;
                self.k_sma_sum / self.slowk_period as f64
            }
        } else if self.slowk_ma_type.eq_ignore_ascii_case("ema") {
            if self.slowk_period == 1 {
                self.k_ema = Some(k_raw);
                k_raw
            } else if self.k_ema.is_none() {
                self.k_ema_seed_sum += k_raw;
                self.k_ema_seed_count += 1;
                if self.k_ema_seed_count == self.slowk_period {
                    let seed = self.k_ema_seed_sum / self.slowk_period as f64;
                    self.k_ema = Some(seed);
                    seed
                } else {
                    f64::NAN
                }
            } else {
                let prev = self.k_ema.unwrap();
                let ema = prev + self.alpha_k * (k_raw - prev);
                self.k_ema = Some(ema);
                ema
            }
        } else {
            let mut k_vec = self
                .k_stream
                .take()
                .unwrap_or_else(|| vec![f64::NAN; self.slowk_period]);
            k_vec.remove(0);
            k_vec.push(k_raw);
            self.k_stream = Some(k_vec.clone());

            match ma(
                &self.slowk_ma_type,
                MaData::Slice(&k_vec),
                self.slowk_period,
            ) {
                Ok(slowk) => *slowk.last().unwrap_or(&f64::NAN),
                Err(_) => k_raw,
            }
        };

        let d_last = if self.slowd_ma_type.eq_ignore_ascii_case("sma") {
            if self.slowd_period == 1 {
                k_last
            } else if !k_last.is_finite() {
                f64::NAN
            } else if self.d_sma_count < self.slowd_period {
                self.d_sma_sum += k_last;
                self.d_sma_buf[self.d_sma_head] = k_last;
                self.d_sma_head = (self.d_sma_head + 1) % self.slowd_period;
                self.d_sma_count += 1;
                if self.d_sma_count == self.slowd_period {
                    self.d_sma_sum / self.slowd_period as f64
                } else {
                    f64::NAN
                }
            } else {
                let old = self.d_sma_buf[self.d_sma_head];
                self.d_sma_sum += k_last - old;
                self.d_sma_buf[self.d_sma_head] = k_last;
                self.d_sma_head = (self.d_sma_head + 1) % self.slowd_period;
                self.d_sma_sum / self.slowd_period as f64
            }
        } else if self.slowd_ma_type.eq_ignore_ascii_case("ema") {
            if self.slowd_period == 1 {
                self.d_ema = Some(k_last);
                k_last
            } else if !k_last.is_finite() {
                f64::NAN
            } else if self.d_ema.is_none() {
                self.d_ema_seed_sum += k_last;
                self.d_ema_seed_count += 1;
                if self.d_ema_seed_count == self.slowd_period {
                    let seed = self.d_ema_seed_sum / self.slowd_period as f64;
                    self.d_ema = Some(seed);
                    seed
                } else {
                    f64::NAN
                }
            } else {
                let prev = self.d_ema.unwrap();
                let ema = prev + self.alpha_d * (k_last - prev);
                self.d_ema = Some(ema);
                ema
            }
        } else {
            let mut d_vec = self
                .d_stream
                .take()
                .unwrap_or_else(|| vec![f64::NAN; self.slowd_period]);
            d_vec.remove(0);
            d_vec.push(k_last);
            self.d_stream = Some(d_vec.clone());

            match ma(
                &self.slowd_ma_type,
                MaData::Slice(&d_vec),
                self.slowd_period,
            ) {
                Ok(slowd) => *slowd.last().unwrap_or(&f64::NAN),
                Err(_) => k_last,
            }
        };

        Some((k_last, d_last))
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::context::Context;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::memory::DeviceBuffer;
#[cfg(all(feature = "python", feature = "cuda"))]
use std::sync::Arc;

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(
    module = "ta_indicators.cuda",
    name = "StochDeviceArrayF32",
    unsendable
)]
pub struct StochDeviceArrayF32Py {
    pub(crate) buf: Option<DeviceBuffer<f32>>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) _ctx: Arc<Context>,
    pub(crate) device_id: u32,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl StochDeviceArrayF32Py {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("shape", (self.rows, self.cols))?;
        d.set_item("typestr", "<f4")?;
        d.set_item(
            "strides",
            (
                self.cols * std::mem::size_of::<f32>(),
                std::mem::size_of::<f32>(),
            ),
        )?;
        let buf = self
            .buf
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("buffer already exported via __dlpack__"))?;
        let ptr = buf.as_device_ptr().as_raw() as usize;
        d.set_item("data", (ptr, false))?;
        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        (2, self.device_id as i32)
    }

    #[pyo3(signature = (stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &mut self,
        py: Python<'py>,
        stream: Option<pyo3::PyObject>,
        max_version: Option<pyo3::PyObject>,
        dl_device: Option<pyo3::PyObject>,
        copy: Option<pyo3::PyObject>,
    ) -> PyResult<PyObject> {
        let (kdl, alloc_dev) = self.__dlpack_device__();
        if let Some(dev_obj) = dl_device.as_ref() {
            if let Ok((dev_ty, dev_id)) = dev_obj.extract::<(i32, i32)>(py) {
                if dev_ty != kdl || dev_id != alloc_dev {
                    let wants_copy = copy
                        .as_ref()
                        .and_then(|c| c.extract::<bool>(py).ok())
                        .unwrap_or(false);
                    if wants_copy {
                        return Err(PyValueError::new_err(
                            "stoch: device copy not implemented for __dlpack__",
                        ));
                    } else {
                        return Err(PyValueError::new_err(
                            "stoch: requested dl_device does not match buffer device",
                        ));
                    }
                }
            }
        }
        let _ = stream;

        if let Some(copy_obj) = copy.as_ref() {
            let do_copy: bool = copy_obj.extract(py)?;
            if do_copy {
                return Err(PyValueError::new_err(
                    "stoch: __dlpack__(copy=True) not supported",
                ));
            }
        }

        let buf = self
            .buf
            .take()
            .ok_or_else(|| PyValueError::new_err("__dlpack__ may only be called once"))?;

        let rows = self.rows;
        let cols = self.cols;

        let max_version_bound = max_version.map(|obj| obj.into_bound(py));

        export_f32_cuda_dlpack_2d(py, buf, rows, cols, alloc_dev, max_version_bound)
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "stoch_cuda_batch_dev")]
#[pyo3(signature = (high_f32, low_f32, close_f32, fastk_period=(14,14,0), slowk_period=(3,3,0), slowd_period=(3,3,0), slowk_ma_type="sma", slowd_ma_type="sma", device_id=0))]
pub fn stoch_cuda_batch_dev_py(
    py: Python<'_>,
    high_f32: numpy::PyReadonlyArray1<'_, f32>,
    low_f32: numpy::PyReadonlyArray1<'_, f32>,
    close_f32: numpy::PyReadonlyArray1<'_, f32>,
    fastk_period: (usize, usize, usize),
    slowk_period: (usize, usize, usize),
    slowd_period: (usize, usize, usize),
    slowk_ma_type: &str,
    slowd_ma_type: &str,
    device_id: usize,
) -> PyResult<(StochDeviceArrayF32Py, StochDeviceArrayF32Py)> {
    use crate::cuda::cuda_available;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let h = high_f32.as_slice()?;
    let l = low_f32.as_slice()?;
    let c = close_f32.as_slice()?;
    let sweep = StochBatchRange {
        fastk_period,
        slowk_period,
        slowk_ma_type: (slowk_ma_type.to_string(), slowk_ma_type.to_string(), 0.0),
        slowd_period,
        slowd_ma_type: (slowd_ma_type.to_string(), slowd_ma_type.to_string(), 0.0),
    };
    let (k_buf, d_buf, rows, cols, ctx, dev_id) = py.allow_threads(|| {
        let cuda = crate::cuda::oscillators::CudaStoch::new(device_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let batch = cuda
            .stoch_batch_dev(h, l, c, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.context_arc();
        Ok::<_, PyErr>((
            batch.k.buf,
            batch.d.buf,
            batch.k.rows,
            batch.k.cols,
            ctx,
            cuda.device_id(),
        ))
    })?;
    Ok((
        StochDeviceArrayF32Py {
            buf: Some(k_buf),
            rows,
            cols,
            _ctx: ctx.clone(),
            device_id: dev_id,
        },
        StochDeviceArrayF32Py {
            buf: Some(d_buf),
            rows,
            cols,
            _ctx: ctx,
            device_id: dev_id,
        },
    ))
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "stoch_cuda_many_series_one_param_dev")]
#[pyo3(signature = (high_tm_f32, low_tm_f32, close_tm_f32, cols, rows, fastk_period=14, slowk_period=3, slowd_period=3, slowk_ma_type="sma", slowd_ma_type="sma", device_id=0))]
pub fn stoch_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    high_tm_f32: numpy::PyReadonlyArray1<'_, f32>,
    low_tm_f32: numpy::PyReadonlyArray1<'_, f32>,
    close_tm_f32: numpy::PyReadonlyArray1<'_, f32>,
    cols: usize,
    rows: usize,
    fastk_period: usize,
    slowk_period: usize,
    slowd_period: usize,
    slowk_ma_type: &str,
    slowd_ma_type: &str,
    device_id: usize,
) -> PyResult<(StochDeviceArrayF32Py, StochDeviceArrayF32Py)> {
    use crate::cuda::cuda_available;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let h = high_tm_f32.as_slice()?;
    let l = low_tm_f32.as_slice()?;
    let c = close_tm_f32.as_slice()?;
    let params = StochParams {
        fastk_period: Some(fastk_period),
        slowk_period: Some(slowk_period),
        slowk_ma_type: Some(slowk_ma_type.to_string()),
        slowd_period: Some(slowd_period),
        slowd_ma_type: Some(slowd_ma_type.to_string()),
    };
    let (k_dev, d_dev, ctx, dev_id) = py.allow_threads(|| {
        let cuda = crate::cuda::oscillators::CudaStoch::new(device_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let (k, d) = cuda
            .stoch_many_series_one_param_time_major_dev(h, l, c, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.context_arc();
        Ok::<_, PyErr>((k.buf, d.buf, ctx, cuda.device_id()))
    })?;
    Ok((
        StochDeviceArrayF32Py {
            buf: Some(k_dev),
            rows,
            cols,
            _ctx: ctx.clone(),
            device_id: dev_id,
        },
        StochDeviceArrayF32Py {
            buf: Some(d_dev),
            rows,
            cols,
            _ctx: ctx,
            device_id: dev_id,
        },
    ))
}

#[cfg(feature = "python")]
#[pyfunction(name = "stoch")]
#[pyo3(signature = (high, low, close, fastk_period=14, slowk_period=3, slowk_ma_type="sma", slowd_period=3, slowd_ma_type="sma", kernel=None))]
pub fn stoch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    fastk_period: usize,
    slowk_period: usize,
    slowk_ma_type: &str,
    slowd_period: usize,
    slowd_ma_type: &str,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let hi = high.as_slice()?;
    let lo = low.as_slice()?;
    let cl = close.as_slice()?;
    let params = StochParams {
        fastk_period: Some(fastk_period),
        slowk_period: Some(slowk_period),
        slowk_ma_type: Some(slowk_ma_type.to_string()),
        slowd_period: Some(slowd_period),
        slowd_ma_type: Some(slowd_ma_type.to_string()),
    };
    let kern = validate_kernel(kernel, false)?;
    let input = StochInput::from_slices(hi, lo, cl, params);
    let out = py
        .allow_threads(|| stoch_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((out.k.into_pyarray(py), out.d.into_pyarray(py)))
}

#[cfg(feature = "python")]
#[pyfunction(name = "stoch_batch")]
#[pyo3(signature = (high, low, close, fastk_range, slowk_range, slowk_ma_type, slowd_range, slowd_ma_type, kernel=None))]
pub fn stoch_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    fastk_range: (usize, usize, usize),
    slowk_range: (usize, usize, usize),
    slowk_ma_type: &str,
    slowd_range: (usize, usize, usize),
    slowd_ma_type: &str,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let hi = high.as_slice()?;
    let lo = low.as_slice()?;
    let cl = close.as_slice()?;

    let sweep = StochBatchRange {
        fastk_period: fastk_range,
        slowk_period: slowk_range,
        slowk_ma_type: (slowk_ma_type.to_string(), slowk_ma_type.to_string(), 0.0),
        slowd_period: slowd_range,
        slowd_ma_type: (slowd_ma_type.to_string(), slowd_ma_type.to_string(), 0.0),
    };

    let kern = validate_kernel(kernel, true)?;
    let out = py
        .allow_threads(|| stoch_batch_with_kernel(hi, lo, cl, &sweep, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let rows = out.rows;
    let cols = out.cols;
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("stoch_batch: size overflow in rows*cols"))?;

    let dict = PyDict::new(py);

    let k_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let d_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    unsafe { k_arr.as_slice_mut()? }.copy_from_slice(&out.k);
    unsafe { d_arr.as_slice_mut()? }.copy_from_slice(&out.d);

    dict.set_item("k", k_arr.reshape((rows, cols))?)?;
    dict.set_item("d", d_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "fastk_periods",
        out.combos
            .iter()
            .map(|p| p.fastk_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "slowk_periods",
        out.combos
            .iter()
            .map(|p| p.slowk_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "slowk_types",
        out.combos
            .iter()
            .map(|p| p.slowk_ma_type.as_deref().unwrap_or("sma"))
            .collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "slowd_periods",
        out.combos
            .iter()
            .map(|p| p.slowd_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "slowd_types",
        out.combos
            .iter()
            .map(|p| p.slowd_ma_type.as_deref().unwrap_or("sma"))
            .collect::<Vec<_>>(),
    )?;

    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "StochStream")]
pub struct StochStreamPy {
    stream: StochStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl StochStreamPy {
    #[new]
    fn new(
        fastk_period: usize,
        slowk_period: usize,
        slowk_ma_type: &str,
        slowd_period: usize,
        slowd_ma_type: &str,
    ) -> PyResult<Self> {
        let params = StochParams {
            fastk_period: Some(fastk_period),
            slowk_period: Some(slowk_period),
            slowk_ma_type: Some(slowk_ma_type.to_string()),
            slowd_period: Some(slowd_period),
            slowd_ma_type: Some(slowd_ma_type.to_string()),
        };
        Ok(Self {
            stream: StochStream::try_new(params)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        })
    }
    fn update(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64)> {
        self.stream.update(high, low, close)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct StochResult {
    pub values: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = stoch)]
pub fn stoch_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    fastk_period: usize,
    slowk_period: usize,
    slowk_ma_type: &str,
    slowd_period: usize,
    slowd_ma_type: &str,
) -> Result<JsValue, JsValue> {
    let params = StochParams {
        fastk_period: Some(fastk_period),
        slowk_period: Some(slowk_period),
        slowk_ma_type: Some(slowk_ma_type.to_string()),
        slowd_period: Some(slowd_period),
        slowd_ma_type: Some(slowd_ma_type.to_string()),
    };
    let input = StochInput::from_slices(high, low, close, params);
    let out = stoch_with_kernel(&input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let mut values = out.k;
    values.extend_from_slice(&out.d);
    serde_wasm_bindgen::to_value(&StochResult {
        values,
        rows: 2,
        cols: high.len(),
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct StochBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<StochParams>,
    pub rows_per_combo: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = stoch_batch)]
pub fn stoch_batch_unified_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    fastk_start: usize,
    fastk_end: usize,
    fastk_step: usize,
    slowk_start: usize,
    slowk_end: usize,
    slowk_step: usize,
    slowk_ma_type: &str,
    slowd_start: usize,
    slowd_end: usize,
    slowd_step: usize,
    slowd_ma_type: &str,
) -> Result<JsValue, JsValue> {
    let sweep = StochBatchRange {
        fastk_period: (fastk_start, fastk_end, fastk_step),
        slowk_period: (slowk_start, slowk_end, slowk_step),
        slowk_ma_type: (slowk_ma_type.to_string(), slowk_ma_type.to_string(), 0.0),
        slowd_period: (slowd_start, slowd_end, slowd_step),
        slowd_ma_type: (slowd_ma_type.to_string(), slowd_ma_type.to_string(), 0.0),
    };
    let out = stoch_batch_inner(high, low, close, &sweep, detect_best_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let mut values = out.k.clone();
    values.extend_from_slice(&out.d);
    let js = StochBatchJsOutput {
        values,
        combos: out.combos,
        rows_per_combo: 2,
        cols: out.cols,
    };
    serde_wasm_bindgen::to_value(&js)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn stoch_alloc(len: usize) -> *mut f64 {
    let mut v = Vec::<f64>::with_capacity(len);
    let ptr = v.as_mut_ptr();
    core::mem::forget(v);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn stoch_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = stoch_into)]
pub fn stoch_into_js(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    len: usize,
    fastk_period: usize,
    slowk_period: usize,
    slowk_ma_type: &str,
    slowd_period: usize,
    slowd_ma_type: &str,
    out_k_ptr: *mut f64,
    out_d_ptr: *mut f64,
) -> Result<(), JsValue> {
    if [high_ptr, low_ptr, close_ptr, out_k_ptr, out_d_ptr]
        .iter()
        .any(|p| p.is_null())
    {
        return Err(JsValue::from_str("null pointer"));
    }
    unsafe {
        let hi = core::slice::from_raw_parts(high_ptr, len);
        let lo = core::slice::from_raw_parts(low_ptr, len);
        let cl = core::slice::from_raw_parts(close_ptr, len);
        let mut ok = core::slice::from_raw_parts_mut(out_k_ptr, len);
        let mut od = core::slice::from_raw_parts_mut(out_d_ptr, len);
        let params = StochParams {
            fastk_period: Some(fastk_period),
            slowk_period: Some(slowk_period),
            slowk_ma_type: Some(slowk_ma_type.to_string()),
            slowd_period: Some(slowd_period),
            slowd_ma_type: Some(slowd_ma_type.to_string()),
        };
        let input = StochInput::from_slices(hi, lo, cl, params);
        stoch_into_slices(&mut ok, &mut od, &input, detect_best_kernel())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::utilities::enums::Kernel;

    fn check_stoch_partial_params(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = StochParams::default();
        let input = StochInput::from_candles(&candles, default_params);
        let output = stoch_with_kernel(&input, kernel)?;
        assert_eq!(output.k.len(), candles.close.len());
        assert_eq!(output.d.len(), candles.close.len());
        Ok(())
    }
    fn check_stoch_accuracy(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = StochInput::from_candles(&candles, StochParams::default());
        let result = stoch_with_kernel(&input, kernel)?;
        assert_eq!(result.k.len(), candles.close.len());
        assert_eq!(result.d.len(), candles.close.len());
        let last_five_k = [
            42.51122827572717,
            40.13864479593807,
            37.853934778363374,
            37.337021714266086,
            36.26053890551548,
        ];
        let last_five_d = [
            41.36561869426493,
            41.7691857059163,
            40.16793595000925,
            38.44320042952222,
            37.15049846604803,
        ];
        let k_slice = &result.k[result.k.len() - 5..];
        let d_slice = &result.d[result.d.len() - 5..];
        for i in 0..5 {
            assert!(
                (k_slice[i] - last_five_k[i]).abs() < 1e-6,
                "Mismatch in K at {}: got {}, expected {}",
                i,
                k_slice[i],
                last_five_k[i]
            );
            assert!(
                (d_slice[i] - last_five_d[i]).abs() < 1e-6,
                "Mismatch in D at {}: got {}, expected {}",
                i,
                d_slice[i],
                last_five_d[i]
            );
        }
        Ok(())
    }
    fn check_stoch_default_candles(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = StochInput::with_default_candles(&candles);
        let output = stoch_with_kernel(&input, kernel)?;
        assert_eq!(output.k.len(), candles.close.len());
        assert_eq!(output.d.len(), candles.close.len());
        Ok(())
    }
    fn check_stoch_zero_period(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let high = [10.0, 11.0, 12.0];
        let low = [9.0, 9.5, 10.5];
        let close = [9.5, 10.6, 11.5];
        let params = StochParams {
            fastk_period: Some(0),
            ..Default::default()
        };
        let input = StochInput::from_slices(&high, &low, &close, params);
        let result = stoch_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }
    fn check_stoch_period_exceeds_length(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let high = [10.0, 11.0, 12.0];
        let low = [9.0, 9.5, 10.5];
        let close = [9.5, 10.6, 11.5];
        let params = StochParams {
            fastk_period: Some(10),
            ..Default::default()
        };
        let input = StochInput::from_slices(&high, &low, &close, params);
        let result = stoch_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }
    fn check_stoch_all_nan(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let nan_data = [f64::NAN, f64::NAN, f64::NAN];
        let params = StochParams::default();
        let input = StochInput::from_slices(&nan_data, &nan_data, &nan_data, params);
        let result = stoch_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_stoch_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let test_params = vec![
            StochParams::default(),
            StochParams {
                fastk_period: Some(2),
                slowk_period: Some(1),
                slowd_period: Some(1),
                slowk_ma_type: Some("sma".to_string()),
                slowd_ma_type: Some("sma".to_string()),
            },
            StochParams {
                fastk_period: Some(5),
                slowk_period: Some(2),
                slowd_period: Some(2),
                slowk_ma_type: Some("sma".to_string()),
                slowd_ma_type: Some("sma".to_string()),
            },
            StochParams {
                fastk_period: Some(10),
                slowk_period: Some(5),
                slowd_period: Some(3),
                slowk_ma_type: Some("ema".to_string()),
                slowd_ma_type: Some("ema".to_string()),
            },
            StochParams {
                fastk_period: Some(14),
                slowk_period: Some(5),
                slowd_period: Some(5),
                slowk_ma_type: Some("sma".to_string()),
                slowd_ma_type: Some("ema".to_string()),
            },
            StochParams {
                fastk_period: Some(20),
                slowk_period: Some(3),
                slowd_period: Some(3),
                slowk_ma_type: Some("sma".to_string()),
                slowd_ma_type: Some("sma".to_string()),
            },
            StochParams {
                fastk_period: Some(50),
                slowk_period: Some(10),
                slowd_period: Some(10),
                slowk_ma_type: Some("ema".to_string()),
                slowd_ma_type: Some("sma".to_string()),
            },
            StochParams {
                fastk_period: Some(100),
                slowk_period: Some(20),
                slowd_period: Some(15),
                slowk_ma_type: Some("sma".to_string()),
                slowd_ma_type: Some("sma".to_string()),
            },
            StochParams {
                fastk_period: Some(7),
                slowk_period: Some(1),
                slowd_period: Some(7),
                slowk_ma_type: Some("sma".to_string()),
                slowd_ma_type: Some("ema".to_string()),
            },
            StochParams {
                fastk_period: Some(3),
                slowk_period: Some(3),
                slowd_period: Some(1),
                slowk_ma_type: Some("ema".to_string()),
                slowd_ma_type: Some("sma".to_string()),
            },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = StochInput::from_candles(&candles, params.clone());
            let output = stoch_with_kernel(&input, kernel)?;

            for (i, &val) in output.k.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in K values \
						 with params: fastk_period={}, slowk_period={}, slowd_period={}, \
						 slowk_ma_type={}, slowd_ma_type={} (param set {})",
						test_name, val, bits, i,
						params.fastk_period.unwrap_or(14),
						params.slowk_period.unwrap_or(3),
						params.slowd_period.unwrap_or(3),
						params.slowk_ma_type.as_deref().unwrap_or("sma"),
						params.slowd_ma_type.as_deref().unwrap_or("sma"),
						param_idx
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in K values \
						 with params: fastk_period={}, slowk_period={}, slowd_period={}, \
						 slowk_ma_type={}, slowd_ma_type={} (param set {})",
						test_name, val, bits, i,
						params.fastk_period.unwrap_or(14),
						params.slowk_period.unwrap_or(3),
						params.slowd_period.unwrap_or(3),
						params.slowk_ma_type.as_deref().unwrap_or("sma"),
						params.slowd_ma_type.as_deref().unwrap_or("sma"),
						param_idx
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in K values \
						 with params: fastk_period={}, slowk_period={}, slowd_period={}, \
						 slowk_ma_type={}, slowd_ma_type={} (param set {})",
						test_name, val, bits, i,
						params.fastk_period.unwrap_or(14),
						params.slowk_period.unwrap_or(3),
						params.slowd_period.unwrap_or(3),
						params.slowk_ma_type.as_deref().unwrap_or("sma"),
						params.slowd_ma_type.as_deref().unwrap_or("sma"),
						param_idx
					);
                }
            }

            for (i, &val) in output.d.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} in D values \
						 with params: fastk_period={}, slowk_period={}, slowd_period={}, \
						 slowk_ma_type={}, slowd_ma_type={} (param set {})",
						test_name, val, bits, i,
						params.fastk_period.unwrap_or(14),
						params.slowk_period.unwrap_or(3),
						params.slowd_period.unwrap_or(3),
						params.slowk_ma_type.as_deref().unwrap_or("sma"),
						params.slowd_ma_type.as_deref().unwrap_or("sma"),
						param_idx
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} in D values \
						 with params: fastk_period={}, slowk_period={}, slowd_period={}, \
						 slowk_ma_type={}, slowd_ma_type={} (param set {})",
						test_name, val, bits, i,
						params.fastk_period.unwrap_or(14),
						params.slowk_period.unwrap_or(3),
						params.slowd_period.unwrap_or(3),
						params.slowk_ma_type.as_deref().unwrap_or("sma"),
						params.slowd_ma_type.as_deref().unwrap_or("sma"),
						param_idx
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} in D values \
						 with params: fastk_period={}, slowk_period={}, slowd_period={}, \
						 slowk_ma_type={}, slowd_ma_type={} (param set {})",
						test_name, val, bits, i,
						params.fastk_period.unwrap_or(14),
						params.slowk_period.unwrap_or(3),
						params.slowd_period.unwrap_or(3),
						params.slowk_ma_type.as_deref().unwrap_or("sma"),
						params.slowd_ma_type.as_deref().unwrap_or("sma"),
						param_idx
					);
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_stoch_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_stoch_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (2usize..=50)
            .prop_flat_map(|fastk_period| {
                (
                    prop::collection::vec(
                        (1.0f64..1000.0f64, 0.001f64..0.1f64),
                        fastk_period.max(10)..400,
                    ),
                    Just(fastk_period),
                    1usize..=10,
                    1usize..=10,
                    prop::bool::ANY,
                    -0.01f64..0.01f64,
                    prop::bool::ANY,
                )
            })
            .prop_flat_map(
                |(
                    price_vol_pairs,
                    fastk_period,
                    slowk_period,
                    slowd_period,
                    use_ema,
                    trend,
                    is_flat,
                )| {
                    let len = price_vol_pairs.len();
                    (
                        Just((
                            price_vol_pairs,
                            fastk_period,
                            slowk_period,
                            slowd_period,
                            use_ema,
                            trend,
                            is_flat,
                        )),
                        prop::collection::vec(-1.0f64..1.0f64, len),
                        prop::collection::vec(0.0f64..1.0f64, len),
                    )
                },
            )
            .prop_map(
                |(
                    (
                        price_vol_pairs,
                        fastk_period,
                        slowk_period,
                        slowd_period,
                        use_ema,
                        trend,
                        is_flat,
                    ),
                    close_factors,
                    beta_params,
                )| {
                    let mut high = Vec::with_capacity(price_vol_pairs.len());
                    let mut low = Vec::with_capacity(price_vol_pairs.len());
                    let mut close = Vec::with_capacity(price_vol_pairs.len());

                    let mut cumulative_trend = 1.0;

                    for (i, ((base_price, volatility), (close_factor, beta))) in price_vol_pairs
                        .into_iter()
                        .zip(close_factors.into_iter().zip(beta_params))
                        .enumerate()
                    {
                        cumulative_trend *= 1.0 + trend;
                        let trended_price = base_price * cumulative_trend;

                        if is_flat {
                            let flat_price = if i == 0 { base_price } else { high[0] };
                            high.push(flat_price);
                            low.push(flat_price);
                            close.push(flat_price);
                        } else {
                            let spread = trended_price * volatility;
                            let h = trended_price + spread;
                            let l = (trended_price - spread).max(0.01);

                            let beta_factor = if beta < 0.5 {
                                2.0 * beta * beta
                            } else {
                                1.0 - 2.0 * (1.0 - beta) * (1.0 - beta)
                            };

                            let close_position = close_factor * 0.5 + beta_factor * 0.5;
                            let c = l + (h - l) * ((close_position + 1.0) / 2.0);

                            high.push(h);
                            low.push(l);
                            close.push(c.clamp(l, h));
                        }
                    }

                    let ma_type = if use_ema { "ema" } else { "sma" };

                    (
                        high,
                        low,
                        close,
                        fastk_period,
                        slowk_period,
                        slowd_period,
                        ma_type.to_string(),
                        is_flat,
                    )
                },
            );

        proptest::test_runner::TestRunner::default().run(
            &strat,
            |(high, low, close, fastk_period, slowk_period, slowd_period, ma_type, is_flat)| {
                let params = StochParams {
                    fastk_period: Some(fastk_period),
                    slowk_period: Some(slowk_period),
                    slowk_ma_type: Some(ma_type.clone()),
                    slowd_period: Some(slowd_period),
                    slowd_ma_type: Some(ma_type.clone()),
                };

                let input = StochInput::from_slices(&high, &low, &close, params.clone());

                let result = stoch_with_kernel(&input, kernel)?;

                let ref_result = stoch_with_kernel(&input, Kernel::Scalar)?;

                prop_assert_eq!(result.k.len(), high.len());
                prop_assert_eq!(result.d.len(), high.len());

                let warmup_k = fastk_period - 1;
                let warmup_slowk = if ma_type == "ema" {
                    0
                } else {
                    slowk_period - 1
                };
                let warmup_slowd = if ma_type == "ema" {
                    0
                } else {
                    slowd_period - 1
                };
                let expected_warmup = warmup_k
                    .max(warmup_k + warmup_slowk)
                    .max(warmup_k + warmup_slowk + warmup_slowd);

                for i in 0..warmup_k.min(high.len()) {
                    prop_assert!(
                        result.k[i].is_nan(),
                        "K[{}] should be NaN during initial warmup but was {}",
                        i,
                        result.k[i]
                    );
                    prop_assert!(
                        result.d[i].is_nan(),
                        "D[{}] should be NaN during initial warmup but was {}",
                        i,
                        result.d[i]
                    );
                }

                for i in expected_warmup..high.len() {
                    let k_val = result.k[i];
                    let d_val = result.d[i];
                    let ref_k = ref_result.k[i];
                    let ref_d = ref_result.d[i];

                    if !k_val.is_nan() {
                        prop_assert!(
                            k_val >= -1e-9 && k_val <= 100.0 + 1e-9,
                            "K[{}] = {} is outside [0, 100] range",
                            i,
                            k_val
                        );
                    }

                    if !d_val.is_nan() {
                        prop_assert!(
                            d_val >= -1e-9 && d_val <= 100.0 + 1e-9,
                            "D[{}] = {} is outside [0, 100] range",
                            i,
                            d_val
                        );
                    }

                    if k_val.is_finite() && ref_k.is_finite() {
                        let k_diff = (k_val - ref_k).abs();
                        let k_ulp_diff = k_val.to_bits().abs_diff(ref_k.to_bits());
                        prop_assert!(
                            k_diff <= 1e-9 || k_ulp_diff <= 4,
                            "K mismatch at [{}]: {} vs {} (diff={}, ULP={})",
                            i,
                            k_val,
                            ref_k,
                            k_diff,
                            k_ulp_diff
                        );
                    }

                    if d_val.is_finite() && ref_d.is_finite() {
                        let d_diff = (d_val - ref_d).abs();
                        let d_ulp_diff = d_val.to_bits().abs_diff(ref_d.to_bits());
                        prop_assert!(
                            d_diff <= 1e-9 || d_ulp_diff <= 4,
                            "D mismatch at [{}]: {} vs {} (diff={}, ULP={})",
                            i,
                            d_val,
                            ref_d,
                            d_diff,
                            d_ulp_diff
                        );
                    }

                    if i >= fastk_period - 1 && !k_val.is_nan() {
                        let window_start = i + 1 - fastk_period;
                        let window_high = &high[window_start..=i];
                        let window_low = &low[window_start..=i];

                        let max_h = window_high
                            .iter()
                            .cloned()
                            .fold(f64::NEG_INFINITY, f64::max);
                        let min_l = window_low.iter().cloned().fold(f64::INFINITY, f64::min);

                        if is_flat || (max_h - min_l).abs() < f64::EPSILON {
                            prop_assert!(
                                (k_val - 50.0).abs() < 1e-6,
                                "K[{}] = {} should be 50 in flat market",
                                i,
                                k_val
                            );
                        } else {
                            if (close[i] - max_h).abs() < 1e-10 {
                                let expected_min = if slowk_period == 1 { 99.0 } else { 85.0 };
                                prop_assert!(
									k_val >= expected_min,
									"K[{}] = {} should be >= {} when close equals highest high (slowk_period={})",
									i, k_val, expected_min, slowk_period
								);
                            }

                            if (close[i] - min_l).abs() < 1e-10 {
                                let expected_max = if slowk_period == 1 { 1.0 } else { 15.0 };
                                prop_assert!(
									k_val <= expected_max,
									"K[{}] = {} should be <= {} when close equals lowest low (slowk_period={})",
									i, k_val, expected_max, slowk_period
								);
                            }
                        }
                    }
                }

                let k_valid: Vec<f64> =
                    result.k.iter().filter(|x| x.is_finite()).copied().collect();
                let d_valid: Vec<f64> =
                    result.d.iter().filter(|x| x.is_finite()).copied().collect();

                if k_valid.len() > 10 && d_valid.len() > 10 && !is_flat {
                    let k_mean = k_valid.iter().sum::<f64>() / k_valid.len() as f64;
                    let d_mean = d_valid.iter().sum::<f64>() / d_valid.len() as f64;

                    let k_var = k_valid.iter().map(|x| (x - k_mean).powi(2)).sum::<f64>()
                        / k_valid.len() as f64;
                    let d_var = d_valid.iter().map(|x| (x - d_mean).powi(2)).sum::<f64>()
                        / d_valid.len() as f64;

                    if slowd_period > 1 && k_var > 1e-6 {
                        prop_assert!(
							d_var <= k_var * 1.01,
							"D variance {} should be <= K variance {} (smoothing effect with slowd_period={})",
							d_var, k_var, slowd_period
						);
                    }

                    if slowd_period == 1 {
                        for i in expected_warmup..result.k.len() {
                            if result.k[i].is_finite() && result.d[i].is_finite() {
                                prop_assert!(
                                    (result.k[i] - result.d[i]).abs() < 1e-9,
                                    "When slowd_period=1, D[{}]={} should equal K[{}]={}",
                                    i,
                                    result.d[i],
                                    i,
                                    result.k[i]
                                );
                            }
                        }
                    }
                }

                if !is_flat && high.len() > fastk_period + 10 {
                    let opposite_ma_type = if ma_type == "sma" { "ema" } else { "sma" };
                    let opposite_params = StochParams {
                        fastk_period: Some(fastk_period),
                        slowk_period: Some(slowk_period),
                        slowk_ma_type: Some(opposite_ma_type.to_string()),
                        slowd_period: Some(slowd_period),
                        slowd_ma_type: Some(opposite_ma_type.to_string()),
                    };

                    let opposite_input =
                        StochInput::from_slices(&high, &low, &close, opposite_params);
                    let opposite_result = stoch_with_kernel(&opposite_input, kernel)?;

                    let mut diff_count = 0;
                    let mut total_valid = 0;
                    for i in expected_warmup..result.k.len() {
                        if result.k[i].is_finite() && opposite_result.k[i].is_finite() {
                            total_valid += 1;
                            if (result.k[i] - opposite_result.k[i]).abs() > 1e-6 {
                                diff_count += 1;
                            }
                        }
                    }

                    if total_valid > 10 && slowk_period > 1 {
                        let diff_ratio = diff_count as f64 / total_valid as f64;
                        prop_assert!(
							diff_ratio >= 0.8,
							"SMA and EMA should produce different results: only {}/{} values differ ({}%)",
							diff_count, total_valid, (diff_ratio * 100.0) as i32
						);
                    }
                }

                Ok(())
            },
        )?;

        Ok(())
    }

    macro_rules! generate_all_stoch_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $( #[test] fn [<$test_fn _scalar_f64>]() { let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar); } )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $( #[test] fn [<$test_fn _avx2_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2); } )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $( #[test] fn [<$test_fn _avx512_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512); } )*
            }
        }
    }
    generate_all_stoch_tests!(
        check_stoch_partial_params,
        check_stoch_accuracy,
        check_stoch_default_candles,
        check_stoch_zero_period,
        check_stoch_period_exceeds_length,
        check_stoch_all_nan,
        check_stoch_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_stoch_tests!(check_stoch_property);
    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = StochBatchBuilder::new().kernel(kernel).apply_candles(&c)?;

        let def = StochParams::default();
        let (row_k, row_d) = output.values_for(&def).expect("default row missing");

        assert_eq!(row_k.len(), c.close.len());
        assert_eq!(row_d.len(), c.close.len());

        let expected_k = [
            42.51122827572717,
            40.13864479593807,
            37.853934778363374,
            37.337021714266086,
            36.26053890551548,
        ];
        let expected_d = [
            41.36561869426493,
            41.7691857059163,
            40.16793595000925,
            38.44320042952222,
            37.15049846604803,
        ];
        let start = row_k.len() - 5;
        for (i, &v) in row_k[start..].iter().enumerate() {
            assert!(
                (v - expected_k[i]).abs() < 1e-6,
                "[{test}] default-row K mismatch at idx {i}: {v} vs {expected_k:?}"
            );
        }
        for (i, &v) in row_d[start..].iter().enumerate() {
            assert!(
                (v - expected_d[i]).abs() < 1e-6,
                "[{test}] default-row D mismatch at idx {i}: {v} vs {expected_d:?}"
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let test_configs = vec![
            (2, 10, 2, 1, 5, 1, 1, 5, 1),
            (5, 25, 5, 2, 10, 2, 2, 10, 2),
            (10, 30, 10, 3, 9, 3, 3, 9, 3),
            (14, 14, 0, 1, 5, 1, 1, 5, 1),
            (2, 5, 1, 3, 3, 0, 3, 3, 0),
            (20, 50, 15, 5, 15, 5, 5, 15, 5),
            (7, 21, 7, 2, 6, 2, 2, 6, 2),
            (3, 12, 3, 1, 3, 1, 1, 3, 1),
        ];

        for (
            cfg_idx,
            &(fk_start, fk_end, fk_step, sk_start, sk_end, sk_step, sd_start, sd_end, sd_step),
        ) in test_configs.iter().enumerate()
        {
            let output = StochBatchBuilder::new()
                .kernel(kernel)
                .fastk_period_range(fk_start, fk_end, fk_step)
                .slowk_period_range(sk_start, sk_end, sk_step)
                .slowd_period_range(sd_start, sd_end, sd_step)
                .slowk_ma_type_static("sma")
                .slowd_ma_type_static("sma")
                .apply_candles(&c)?;

            for (idx, &val) in output.k.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;
                let combo = &output.combos[row];

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in K values with params: \
						 fastk_period={}, slowk_period={}, slowd_period={}, \
						 slowk_ma_type={}, slowd_ma_type={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.fastk_period.unwrap_or(14),
                        combo.slowk_period.unwrap_or(3),
                        combo.slowd_period.unwrap_or(3),
                        combo.slowk_ma_type.as_deref().unwrap_or("sma"),
                        combo.slowd_ma_type.as_deref().unwrap_or("sma")
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in K values with params: \
						 fastk_period={}, slowk_period={}, slowd_period={}, \
						 slowk_ma_type={}, slowd_ma_type={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.fastk_period.unwrap_or(14),
                        combo.slowk_period.unwrap_or(3),
                        combo.slowd_period.unwrap_or(3),
                        combo.slowk_ma_type.as_deref().unwrap_or("sma"),
                        combo.slowd_ma_type.as_deref().unwrap_or("sma")
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in K values with params: \
						 fastk_period={}, slowk_period={}, slowd_period={}, \
						 slowk_ma_type={}, slowd_ma_type={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.fastk_period.unwrap_or(14),
                        combo.slowk_period.unwrap_or(3),
                        combo.slowd_period.unwrap_or(3),
                        combo.slowk_ma_type.as_deref().unwrap_or("sma"),
                        combo.slowd_ma_type.as_deref().unwrap_or("sma")
                    );
                }
            }

            for (idx, &val) in output.d.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;
                let combo = &output.combos[row];

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in D values with params: \
						 fastk_period={}, slowk_period={}, slowd_period={}, \
						 slowk_ma_type={}, slowd_ma_type={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.fastk_period.unwrap_or(14),
                        combo.slowk_period.unwrap_or(3),
                        combo.slowd_period.unwrap_or(3),
                        combo.slowk_ma_type.as_deref().unwrap_or("sma"),
                        combo.slowd_ma_type.as_deref().unwrap_or("sma")
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in D values with params: \
						 fastk_period={}, slowk_period={}, slowd_period={}, \
						 slowk_ma_type={}, slowd_ma_type={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.fastk_period.unwrap_or(14),
                        combo.slowk_period.unwrap_or(3),
                        combo.slowd_period.unwrap_or(3),
                        combo.slowk_ma_type.as_deref().unwrap_or("sma"),
                        combo.slowd_ma_type.as_deref().unwrap_or("sma")
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in D values with params: \
						 fastk_period={}, slowk_period={}, slowd_period={}, \
						 slowk_ma_type={}, slowd_ma_type={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.fastk_period.unwrap_or(14),
                        combo.slowk_period.unwrap_or(3),
                        combo.slowd_period.unwrap_or(3),
                        combo.slowk_ma_type.as_deref().unwrap_or("sma"),
                        combo.slowd_ma_type.as_deref().unwrap_or("sma")
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste::paste! {
                #[test] fn [<$fn_name _scalar>]()      {
                    let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx2>]()        {
                    let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx512>]()      {
                    let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch);
                }
                #[test] fn [<$fn_name _auto_detect>]() {
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto);
                }
            }
        };
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);

    fn eq_or_both_nan(a: f64, b: f64) -> bool {
        (a.is_nan() && b.is_nan()) || (a == b)
    }

    #[test]
    fn test_stoch_into_matches_api() -> Result<(), Box<dyn Error>> {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = StochInput::with_default_candles(&candles);

        let baseline = stoch(&input)?;

        let mut out_k = vec![0.0; baseline.k.len()];
        let mut out_d = vec![0.0; baseline.d.len()];

        #[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
        {
            stoch_into(&input, &mut out_k, &mut out_d)?;
        }
        #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
        {
            stoch_into_slices(&mut out_k, &mut out_d, &input, detect_best_kernel())?;
        }

        assert_eq!(out_k.len(), baseline.k.len());
        assert_eq!(out_d.len(), baseline.d.len());
        for i in 0..out_k.len() {
            assert!(
                eq_or_both_nan(out_k[i], baseline.k[i]),
                "K mismatch at {}: got {}, expected {}",
                i,
                out_k[i],
                baseline.k[i]
            );
            assert!(
                eq_or_both_nan(out_d[i], baseline.d[i]),
                "D mismatch at {}: got {}, expected {}",
                i,
                out_d[i],
                baseline.d[i]
            );
        }
        Ok(())
    }

    #[test]
    fn test_stoch_compute_into_matches_api() -> Result<(), Box<dyn Error>> {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = StochInput::with_default_candles(&candles);

        let baseline = stoch(&input)?;

        let mut out_k = vec![0.0; baseline.k.len()];
        let mut out_d = vec![0.0; baseline.d.len()];
        stoch_compute_into(&input, &mut out_k, &mut out_d, Kernel::Auto)?;

        assert_eq!(out_k.len(), baseline.k.len());
        assert_eq!(out_d.len(), baseline.d.len());
        for i in 0..out_k.len() {
            assert!(
                eq_or_both_nan(out_k[i], baseline.k[i]),
                "K mismatch at {}: got {}, expected {}",
                i,
                out_k[i],
                baseline.k[i]
            );
            assert!(
                eq_or_both_nan(out_d[i], baseline.d[i]),
                "D mismatch at {}: got {}, expected {}",
                i,
                out_d[i],
                baseline.d[i]
            );
        }
        Ok(())
    }
}

#[inline]
fn stoch_classic_sma(
    k_raw: &[f64],
    slowk_period: usize,
    slowd_period: usize,
    first_valid_idx: usize,
) -> Result<StochOutput, StochError> {
    let len = k_raw.len();
    let mut k_vec = alloc_with_nan_prefix(len, first_valid_idx + slowk_period - 1);
    let mut d_vec = alloc_with_nan_prefix(len, first_valid_idx + slowk_period + slowd_period - 2);

    let mut sum_k = 0.0;
    let k_start = first_valid_idx;

    for i in k_start..(k_start + slowk_period).min(len) {
        if !k_raw[i].is_nan() {
            sum_k += k_raw[i];
        }
    }
    if k_start + slowk_period - 1 < len {
        k_vec[k_start + slowk_period - 1] = sum_k / slowk_period as f64;
    }

    for i in (k_start + slowk_period)..len {
        let old_val = k_raw[i - slowk_period];
        let new_val = k_raw[i];
        if !old_val.is_nan() {
            sum_k -= old_val;
        }
        if !new_val.is_nan() {
            sum_k += new_val;
        }
        k_vec[i] = sum_k / slowk_period as f64;
    }

    let mut sum_d = 0.0;
    let d_start = first_valid_idx + slowk_period - 1;

    for i in d_start..(d_start + slowd_period).min(len) {
        if !k_vec[i].is_nan() {
            sum_d += k_vec[i];
        }
    }
    if d_start + slowd_period - 1 < len {
        d_vec[d_start + slowd_period - 1] = sum_d / slowd_period as f64;
    }

    for i in (d_start + slowd_period)..len {
        let old_val = k_vec[i - slowd_period];
        let new_val = k_vec[i];
        if !old_val.is_nan() {
            sum_d -= old_val;
        }
        if !new_val.is_nan() {
            sum_d += new_val;
        }
        d_vec[i] = sum_d / slowd_period as f64;
    }

    Ok(StochOutput { k: k_vec, d: d_vec })
}

#[inline]
fn stoch_classic_ema(
    k_raw: &[f64],
    slowk_period: usize,
    slowd_period: usize,
    first_valid_idx: usize,
) -> Result<StochOutput, StochError> {
    let len = k_raw.len();
    let mut k_vec = alloc_with_nan_prefix(len, first_valid_idx + slowk_period - 1);
    let mut d_vec = alloc_with_nan_prefix(len, first_valid_idx + slowk_period + slowd_period - 2);

    let alpha_k = 2.0 / (slowk_period as f64 + 1.0);
    let one_minus_alpha_k = 1.0 - alpha_k;

    let k_warmup = first_valid_idx + slowk_period - 1;
    let mut sum_k = 0.0;
    let mut count_k = 0;
    for i in first_valid_idx..(first_valid_idx + slowk_period).min(len) {
        if !k_raw[i].is_nan() {
            sum_k += k_raw[i];
            count_k += 1;
        }
    }

    if count_k > 0 && k_warmup < len {
        let mut ema_k = sum_k / count_k as f64;
        k_vec[k_warmup] = ema_k;

        for i in (k_warmup + 1)..len {
            if !k_raw[i].is_nan() {
                ema_k = alpha_k * k_raw[i] + one_minus_alpha_k * ema_k;
            }
            k_vec[i] = ema_k;
        }
    } else {
        for i in k_warmup..len {
            k_vec[i] = f64::NAN;
        }
    }

    let alpha_d = 2.0 / (slowd_period as f64 + 1.0);
    let one_minus_alpha_d = 1.0 - alpha_d;

    let d_warmup = first_valid_idx + slowk_period + slowd_period - 2;
    let d_start = first_valid_idx + slowk_period - 1;
    let mut sum_d = 0.0;
    let mut count_d = 0;
    for i in d_start..(d_start + slowd_period).min(len) {
        if !k_vec[i].is_nan() {
            sum_d += k_vec[i];
            count_d += 1;
        }
    }

    if count_d > 0 && d_warmup < len {
        let mut ema_d = sum_d / count_d as f64;
        d_vec[d_warmup] = ema_d;

        for i in (d_warmup + 1)..len {
            if !k_vec[i].is_nan() {
                ema_d = alpha_d * k_vec[i] + one_minus_alpha_d * ema_d;
            }
            d_vec[i] = ema_d;
        }
    } else {
        for i in d_warmup..len {
            d_vec[i] = f64::NAN;
        }
    }

    Ok(StochOutput { k: k_vec, d: d_vec })
}
