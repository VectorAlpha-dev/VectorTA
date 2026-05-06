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
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::mem::ManuallyDrop;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum VwapZscoreWithSignalsData<'a> {
    Candles { candles: &'a Candles },
    Slices { close: &'a [f64], volume: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct VwapZscoreWithSignalsOutput {
    pub zvwap: Vec<f64>,
    pub support_signal: Vec<f64>,
    pub resistance_signal: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VwapZscoreWithSignalsOutputField {
    Zvwap,
    SupportSignal,
    ResistanceSignal,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct VwapZscoreWithSignalsParams {
    pub length: Option<usize>,
    pub upper_bottom: Option<f64>,
    pub lower_bottom: Option<f64>,
}

impl Default for VwapZscoreWithSignalsParams {
    fn default() -> Self {
        Self {
            length: Some(20),
            upper_bottom: Some(2.5),
            lower_bottom: Some(-2.5),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VwapZscoreWithSignalsInput<'a> {
    pub data: VwapZscoreWithSignalsData<'a>,
    pub params: VwapZscoreWithSignalsParams,
}

impl<'a> VwapZscoreWithSignalsInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: VwapZscoreWithSignalsParams) -> Self {
        Self {
            data: VwapZscoreWithSignalsData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        close: &'a [f64],
        volume: &'a [f64],
        params: VwapZscoreWithSignalsParams,
    ) -> Self {
        Self {
            data: VwapZscoreWithSignalsData::Slices { close, volume },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, VwapZscoreWithSignalsParams::default())
    }

    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(20)
    }

    #[inline]
    pub fn get_upper_bottom(&self) -> f64 {
        self.params.upper_bottom.unwrap_or(2.5)
    }

    #[inline]
    pub fn get_lower_bottom(&self) -> f64 {
        self.params.lower_bottom.unwrap_or(-2.5)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct VwapZscoreWithSignalsBuilder {
    length: Option<usize>,
    upper_bottom: Option<f64>,
    lower_bottom: Option<f64>,
    kernel: Kernel,
}

impl Default for VwapZscoreWithSignalsBuilder {
    fn default() -> Self {
        Self {
            length: None,
            upper_bottom: None,
            lower_bottom: None,
            kernel: Kernel::Auto,
        }
    }
}

impl VwapZscoreWithSignalsBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn length(mut self, length: usize) -> Self {
        self.length = Some(length);
        self
    }

    #[inline]
    pub fn upper_bottom(mut self, upper_bottom: f64) -> Self {
        self.upper_bottom = Some(upper_bottom);
        self
    }

    #[inline]
    pub fn lower_bottom(mut self, lower_bottom: f64) -> Self {
        self.lower_bottom = Some(lower_bottom);
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
    ) -> Result<VwapZscoreWithSignalsOutput, VwapZscoreWithSignalsError> {
        let input = VwapZscoreWithSignalsInput::from_candles(
            candles,
            VwapZscoreWithSignalsParams {
                length: self.length,
                upper_bottom: self.upper_bottom,
                lower_bottom: self.lower_bottom,
            },
        );
        vwap_zscore_with_signals_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn apply_slices(
        self,
        close: &[f64],
        volume: &[f64],
    ) -> Result<VwapZscoreWithSignalsOutput, VwapZscoreWithSignalsError> {
        let input = VwapZscoreWithSignalsInput::from_slices(
            close,
            volume,
            VwapZscoreWithSignalsParams {
                length: self.length,
                upper_bottom: self.upper_bottom,
                lower_bottom: self.lower_bottom,
            },
        );
        vwap_zscore_with_signals_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn into_stream(self) -> Result<VwapZscoreWithSignalsStream, VwapZscoreWithSignalsError> {
        VwapZscoreWithSignalsStream::try_new(VwapZscoreWithSignalsParams {
            length: self.length,
            upper_bottom: self.upper_bottom,
            lower_bottom: self.lower_bottom,
        })
    }
}

#[derive(Debug, Error)]
pub enum VwapZscoreWithSignalsError {
    #[error("vwap_zscore_with_signals: Input data slice is empty.")]
    EmptyInputData,
    #[error("vwap_zscore_with_signals: All values are NaN.")]
    AllValuesNaN,
    #[error(
        "vwap_zscore_with_signals: Invalid length: length = {length}, data length = {data_len}"
    )]
    InvalidLength { length: usize, data_len: usize },
    #[error("vwap_zscore_with_signals: Invalid upper_bottom: {upper_bottom}. Must be finite.")]
    InvalidUpperBottom { upper_bottom: f64 },
    #[error("vwap_zscore_with_signals: Invalid lower_bottom: {lower_bottom}. Must be finite.")]
    InvalidLowerBottom { lower_bottom: f64 },
    #[error(
        "vwap_zscore_with_signals: Inconsistent slice lengths: close={close_len}, volume={volume_len}"
    )]
    InconsistentSliceLengths { close_len: usize, volume_len: usize },
    #[error("vwap_zscore_with_signals: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "vwap_zscore_with_signals: Output length mismatch: expected = {expected}, zvwap = {zvwap_got}, support_signal = {support_got}, resistance_signal = {resistance_got}"
    )]
    OutputLengthMismatch {
        expected: usize,
        zvwap_got: usize,
        support_got: usize,
        resistance_got: usize,
    },
    #[error("vwap_zscore_with_signals: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("vwap_zscore_with_signals: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Debug, Clone)]
pub struct VwapZscoreWithSignalsStream {
    length: usize,
    upper_bottom: f64,
    lower_bottom: f64,
    pv_values: Vec<f64>,
    vol_values: Vec<f64>,
    pv_valid: Vec<u8>,
    idx: usize,
    count: usize,
    valid_count: usize,
    pv_sum: f64,
    vol_sum: f64,
    dev_values: Vec<f64>,
    dev_valid: Vec<u8>,
    dev_idx: usize,
    dev_count: usize,
    dev_valid_count: usize,
    dev_sum: f64,
}

impl VwapZscoreWithSignalsStream {
    pub fn try_new(
        params: VwapZscoreWithSignalsParams,
    ) -> Result<VwapZscoreWithSignalsStream, VwapZscoreWithSignalsError> {
        let length = params.length.unwrap_or(20);
        if length == 0 {
            return Err(VwapZscoreWithSignalsError::InvalidLength {
                length,
                data_len: 0,
            });
        }
        let upper_bottom = params.upper_bottom.unwrap_or(2.5);
        if !upper_bottom.is_finite() {
            return Err(VwapZscoreWithSignalsError::InvalidUpperBottom { upper_bottom });
        }
        let lower_bottom = params.lower_bottom.unwrap_or(-2.5);
        if !lower_bottom.is_finite() {
            return Err(VwapZscoreWithSignalsError::InvalidLowerBottom { lower_bottom });
        }

        Ok(Self {
            length,
            upper_bottom,
            lower_bottom,
            pv_values: vec![0.0; length],
            vol_values: vec![0.0; length],
            pv_valid: vec![0u8; length],
            idx: 0,
            count: 0,
            valid_count: 0,
            pv_sum: 0.0,
            vol_sum: 0.0,
            dev_values: vec![0.0; length],
            dev_valid: vec![0u8; length],
            dev_idx: 0,
            dev_count: 0,
            dev_valid_count: 0,
            dev_sum: 0.0,
        })
    }

    #[inline]
    pub fn update(&mut self, close: f64, volume: f64) -> Option<(f64, f64, f64)> {
        if self.count >= self.length {
            let old_idx = self.idx;
            if self.pv_valid[old_idx] != 0 {
                self.valid_count = self.valid_count.saturating_sub(1);
                self.pv_sum -= self.pv_values[old_idx];
                self.vol_sum -= self.vol_values[old_idx];
            }
        } else {
            self.count += 1;
        }

        if valid_close_volume_bar(close, volume) {
            let pv = close * volume;
            self.pv_values[self.idx] = pv;
            self.vol_values[self.idx] = volume;
            self.pv_valid[self.idx] = 1;
            self.valid_count += 1;
            self.pv_sum += pv;
            self.vol_sum += volume;
        } else {
            self.pv_values[self.idx] = 0.0;
            self.vol_values[self.idx] = 0.0;
            self.pv_valid[self.idx] = 0;
        }
        self.idx += 1;
        if self.idx == self.length {
            self.idx = 0;
        }

        if self.dev_count >= self.length {
            let old_idx = self.dev_idx;
            if self.dev_valid[old_idx] != 0 {
                self.dev_valid_count = self.dev_valid_count.saturating_sub(1);
                self.dev_sum -= self.dev_values[old_idx];
            }
        } else {
            self.dev_count += 1;
        }

        let mut mean = f64::NAN;
        if self.count >= self.length && self.valid_count == self.length && self.vol_sum > 0.0 {
            mean = self.pv_sum / self.vol_sum;
            let dev = (close - mean) * (close - mean);
            self.dev_values[self.dev_idx] = dev;
            self.dev_valid[self.dev_idx] = 1;
            self.dev_valid_count += 1;
            self.dev_sum += dev;
        } else {
            self.dev_values[self.dev_idx] = 0.0;
            self.dev_valid[self.dev_idx] = 0;
        }
        self.dev_idx += 1;
        if self.dev_idx == self.length {
            self.dev_idx = 0;
        }

        if self.dev_count < self.length {
            return None;
        }
        if self.dev_valid_count != self.length || !mean.is_finite() {
            return Some((f64::NAN, f64::NAN, f64::NAN));
        }

        let variance = (self.dev_sum / self.length as f64).max(0.0);
        let sd = variance.sqrt();
        if !sd.is_finite() || sd <= 0.0 {
            return Some((f64::NAN, f64::NAN, f64::NAN));
        }

        let zvwap = (close - mean) / sd;
        let support = if zvwap < self.lower_bottom { 1.0 } else { 0.0 };
        let resistance = if zvwap > self.upper_bottom { 1.0 } else { 0.0 };
        Some((zvwap, support, resistance))
    }

    #[inline]
    pub fn get_warmup_period(&self) -> usize {
        self.length.saturating_mul(2).saturating_sub(2)
    }
}

#[inline]
pub fn vwap_zscore_with_signals(
    input: &VwapZscoreWithSignalsInput,
) -> Result<VwapZscoreWithSignalsOutput, VwapZscoreWithSignalsError> {
    vwap_zscore_with_signals_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn valid_close_volume_bar(close: f64, volume: f64) -> bool {
    close.is_finite() && volume.is_finite() && volume >= 0.0
}

#[inline(always)]
fn first_valid_close_volume(close: &[f64], volume: &[f64]) -> usize {
    let len = close.len();
    let mut i = 0usize;
    while i < len {
        if valid_close_volume_bar(close[i], volume[i]) {
            return i;
        }
        i += 1;
    }
    len
}

#[inline(always)]
fn count_valid_close_volume(close: &[f64], volume: &[f64]) -> usize {
    let mut count = 0usize;
    for i in 0..close.len() {
        if valid_close_volume_bar(close[i], volume[i]) {
            count += 1;
        }
    }
    count
}

#[inline(always)]
fn support_signal_value(zvwap: f64, lower_bottom: f64) -> f64 {
    if zvwap.is_finite() && zvwap < lower_bottom {
        1.0
    } else if zvwap.is_finite() {
        0.0
    } else {
        f64::NAN
    }
}

#[inline(always)]
fn resistance_signal_value(zvwap: f64, upper_bottom: f64) -> f64 {
    if zvwap.is_finite() && zvwap > upper_bottom {
        1.0
    } else if zvwap.is_finite() {
        0.0
    } else {
        f64::NAN
    }
}

#[inline(always)]
fn vwap_zscore_with_signals_row_from_slices(
    close: &[f64],
    volume: &[f64],
    length: usize,
    upper_bottom: f64,
    lower_bottom: f64,
    zvwap_out: &mut [f64],
    support_out: &mut [f64],
    resistance_out: &mut [f64],
) {
    let mut pv_values = vec![0.0f64; length];
    let mut vol_values = vec![0.0f64; length];
    let mut pv_valid = vec![0u8; length];
    let mut idx = 0usize;
    let mut count = 0usize;
    let mut valid_count = 0usize;
    let mut pv_sum = 0.0f64;
    let mut vol_sum = 0.0f64;

    let mut dev_values = vec![0.0f64; length];
    let mut dev_valid = vec![0u8; length];
    let mut dev_idx = 0usize;
    let mut dev_count = 0usize;
    let mut dev_valid_count = 0usize;
    let mut dev_sum = 0.0f64;

    for i in 0..close.len() {
        if count >= length {
            let old_idx = idx;
            if pv_valid[old_idx] != 0 {
                valid_count = valid_count.saturating_sub(1);
                pv_sum -= pv_values[old_idx];
                vol_sum -= vol_values[old_idx];
            }
        } else {
            count += 1;
        }

        if valid_close_volume_bar(close[i], volume[i]) {
            let pv = close[i] * volume[i];
            pv_values[idx] = pv;
            vol_values[idx] = volume[i];
            pv_valid[idx] = 1;
            valid_count += 1;
            pv_sum += pv;
            vol_sum += volume[i];
        } else {
            pv_values[idx] = 0.0;
            vol_values[idx] = 0.0;
            pv_valid[idx] = 0;
        }
        idx += 1;
        if idx == length {
            idx = 0;
        }

        if dev_count >= length {
            let old_idx = dev_idx;
            if dev_valid[old_idx] != 0 {
                dev_valid_count = dev_valid_count.saturating_sub(1);
                dev_sum -= dev_values[old_idx];
            }
        } else {
            dev_count += 1;
        }

        let mut mean = f64::NAN;
        if count >= length && valid_count == length && vol_sum > 0.0 {
            mean = pv_sum / vol_sum;
            let dev = (close[i] - mean) * (close[i] - mean);
            dev_values[dev_idx] = dev;
            dev_valid[dev_idx] = 1;
            dev_valid_count += 1;
            dev_sum += dev;
        } else {
            dev_values[dev_idx] = 0.0;
            dev_valid[dev_idx] = 0;
        }
        dev_idx += 1;
        if dev_idx == length {
            dev_idx = 0;
        }

        if dev_count < length || dev_valid_count != length || !mean.is_finite() {
            continue;
        }

        let variance = (dev_sum / length as f64).max(0.0);
        let sd = variance.sqrt();
        if !sd.is_finite() || sd <= 0.0 {
            continue;
        }

        let zvwap = (close[i] - mean) / sd;
        zvwap_out[i] = zvwap;
        support_out[i] = support_signal_value(zvwap, lower_bottom);
        resistance_out[i] = resistance_signal_value(zvwap, upper_bottom);
    }
}

#[inline(always)]
fn vwap_zscore_with_signals_output_row_from_slices(
    close: &[f64],
    volume: &[f64],
    length: usize,
    upper_bottom: f64,
    lower_bottom: f64,
    out: &mut [f64],
    field: VwapZscoreWithSignalsOutputField,
) {
    let mut pv_values = vec![0.0f64; length];
    let mut vol_values = vec![0.0f64; length];
    let mut pv_valid = vec![0u8; length];
    let mut idx = 0usize;
    let mut count = 0usize;
    let mut valid_count = 0usize;
    let mut pv_sum = 0.0f64;
    let mut vol_sum = 0.0f64;

    let mut dev_values = vec![0.0f64; length];
    let mut dev_valid = vec![0u8; length];
    let mut dev_idx = 0usize;
    let mut dev_count = 0usize;
    let mut dev_valid_count = 0usize;
    let mut dev_sum = 0.0f64;

    for i in 0..close.len() {
        if count >= length {
            let old_idx = idx;
            if pv_valid[old_idx] != 0 {
                valid_count = valid_count.saturating_sub(1);
                pv_sum -= pv_values[old_idx];
                vol_sum -= vol_values[old_idx];
            }
        } else {
            count += 1;
        }

        if valid_close_volume_bar(close[i], volume[i]) {
            let pv = close[i] * volume[i];
            pv_values[idx] = pv;
            vol_values[idx] = volume[i];
            pv_valid[idx] = 1;
            valid_count += 1;
            pv_sum += pv;
            vol_sum += volume[i];
        } else {
            pv_values[idx] = 0.0;
            vol_values[idx] = 0.0;
            pv_valid[idx] = 0;
        }
        idx += 1;
        if idx == length {
            idx = 0;
        }

        if dev_count >= length {
            let old_idx = dev_idx;
            if dev_valid[old_idx] != 0 {
                dev_valid_count = dev_valid_count.saturating_sub(1);
                dev_sum -= dev_values[old_idx];
            }
        } else {
            dev_count += 1;
        }

        let mut mean = f64::NAN;
        if count >= length && valid_count == length && vol_sum > 0.0 {
            mean = pv_sum / vol_sum;
            let dev = (close[i] - mean) * (close[i] - mean);
            dev_values[dev_idx] = dev;
            dev_valid[dev_idx] = 1;
            dev_valid_count += 1;
            dev_sum += dev;
        } else {
            dev_values[dev_idx] = 0.0;
            dev_valid[dev_idx] = 0;
        }
        dev_idx += 1;
        if dev_idx == length {
            dev_idx = 0;
        }

        let mut value = f64::NAN;
        if dev_count >= length && dev_valid_count == length && mean.is_finite() {
            let variance = (dev_sum / length as f64).max(0.0);
            let sd = variance.sqrt();
            if sd.is_finite() && sd > 0.0 {
                let zvwap = (close[i] - mean) / sd;
                value = match field {
                    VwapZscoreWithSignalsOutputField::Zvwap => zvwap,
                    VwapZscoreWithSignalsOutputField::SupportSignal => {
                        if zvwap < lower_bottom {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    VwapZscoreWithSignalsOutputField::ResistanceSignal => {
                        if zvwap > upper_bottom {
                            1.0
                        } else {
                            0.0
                        }
                    }
                };
            }
        }
        out[i] = value;
    }
}

#[inline(always)]
fn vwap_zscore_with_signals_prepare<'a>(
    input: &'a VwapZscoreWithSignalsInput,
    kernel: Kernel,
) -> Result<(&'a [f64], &'a [f64], usize, f64, f64, usize, Kernel), VwapZscoreWithSignalsError> {
    let (close, volume) = match &input.data {
        VwapZscoreWithSignalsData::Candles { candles } => {
            (&candles.close[..], candles.volume.as_slice())
        }
        VwapZscoreWithSignalsData::Slices { close, volume } => {
            if close.len() != volume.len() {
                return Err(VwapZscoreWithSignalsError::InconsistentSliceLengths {
                    close_len: close.len(),
                    volume_len: volume.len(),
                });
            }
            (*close, *volume)
        }
    };

    let len = close.len();
    if len == 0 {
        return Err(VwapZscoreWithSignalsError::EmptyInputData);
    }

    let first = first_valid_close_volume(close, volume);
    if first >= len {
        return Err(VwapZscoreWithSignalsError::AllValuesNaN);
    }

    let length = input.get_length();
    if length == 0 || length > len {
        return Err(VwapZscoreWithSignalsError::InvalidLength {
            length,
            data_len: len,
        });
    }

    let upper_bottom = input.get_upper_bottom();
    if !upper_bottom.is_finite() {
        return Err(VwapZscoreWithSignalsError::InvalidUpperBottom { upper_bottom });
    }
    let lower_bottom = input.get_lower_bottom();
    if !lower_bottom.is_finite() {
        return Err(VwapZscoreWithSignalsError::InvalidLowerBottom { lower_bottom });
    }

    let valid = count_valid_close_volume(close, volume);
    let needed = length.saturating_mul(2).saturating_sub(1);
    if valid < needed {
        return Err(VwapZscoreWithSignalsError::NotEnoughValidData { needed, valid });
    }

    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other.to_non_batch(),
    };

    Ok((
        close,
        volume,
        length,
        upper_bottom,
        lower_bottom,
        first,
        chosen,
    ))
}

#[inline]
pub fn vwap_zscore_with_signals_with_kernel(
    input: &VwapZscoreWithSignalsInput,
    kernel: Kernel,
) -> Result<VwapZscoreWithSignalsOutput, VwapZscoreWithSignalsError> {
    let (close, volume, length, upper_bottom, lower_bottom, first, _chosen) =
        vwap_zscore_with_signals_prepare(input, kernel)?;
    let warmup = first
        .saturating_add(length.saturating_mul(2))
        .saturating_sub(2);
    let mut zvwap = alloc_with_nan_prefix(close.len(), warmup);
    let mut support_signal = alloc_with_nan_prefix(close.len(), warmup);
    let mut resistance_signal = alloc_with_nan_prefix(close.len(), warmup);
    vwap_zscore_with_signals_row_from_slices(
        close,
        volume,
        length,
        upper_bottom,
        lower_bottom,
        &mut zvwap,
        &mut support_signal,
        &mut resistance_signal,
    );
    Ok(VwapZscoreWithSignalsOutput {
        zvwap,
        support_signal,
        resistance_signal,
    })
}

#[inline]
pub fn vwap_zscore_with_signals_into_slices(
    zvwap_out: &mut [f64],
    support_out: &mut [f64],
    resistance_out: &mut [f64],
    input: &VwapZscoreWithSignalsInput,
    kernel: Kernel,
) -> Result<(), VwapZscoreWithSignalsError> {
    let (close, volume, length, upper_bottom, lower_bottom, _first, _chosen) =
        vwap_zscore_with_signals_prepare(input, kernel)?;
    if zvwap_out.len() != close.len()
        || support_out.len() != close.len()
        || resistance_out.len() != close.len()
    {
        return Err(VwapZscoreWithSignalsError::OutputLengthMismatch {
            expected: close.len(),
            zvwap_got: zvwap_out.len(),
            support_got: support_out.len(),
            resistance_got: resistance_out.len(),
        });
    }
    zvwap_out.fill(f64::NAN);
    support_out.fill(f64::NAN);
    resistance_out.fill(f64::NAN);
    vwap_zscore_with_signals_row_from_slices(
        close,
        volume,
        length,
        upper_bottom,
        lower_bottom,
        zvwap_out,
        support_out,
        resistance_out,
    );
    Ok(())
}

pub fn vwap_zscore_with_signals_output_into_slice(
    out: &mut [f64],
    input: &VwapZscoreWithSignalsInput,
    kernel: Kernel,
    field: VwapZscoreWithSignalsOutputField,
) -> Result<(), VwapZscoreWithSignalsError> {
    let (close, volume, length, upper_bottom, lower_bottom, _first, _chosen) =
        vwap_zscore_with_signals_prepare(input, kernel)?;
    if out.len() != close.len() {
        return Err(VwapZscoreWithSignalsError::OutputLengthMismatch {
            expected: close.len(),
            zvwap_got: out.len(),
            support_got: out.len(),
            resistance_got: out.len(),
        });
    }
    vwap_zscore_with_signals_output_row_from_slices(
        close,
        volume,
        length,
        upper_bottom,
        lower_bottom,
        out,
        field,
    );
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn vwap_zscore_with_signals_into(
    input: &VwapZscoreWithSignalsInput,
    zvwap_out: &mut [f64],
    support_out: &mut [f64],
    resistance_out: &mut [f64],
) -> Result<(), VwapZscoreWithSignalsError> {
    vwap_zscore_with_signals_into_slices(
        zvwap_out,
        support_out,
        resistance_out,
        input,
        Kernel::Auto,
    )
}

#[derive(Clone, Debug)]
pub struct VwapZscoreWithSignalsBatchRange {
    pub length: (usize, usize, usize),
    pub upper_bottom: (f64, f64, f64),
    pub lower_bottom: (f64, f64, f64),
}

impl Default for VwapZscoreWithSignalsBatchRange {
    fn default() -> Self {
        Self {
            length: (20, 252, 1),
            upper_bottom: (2.5, 2.5, 0.0),
            lower_bottom: (-2.5, -2.5, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct VwapZscoreWithSignalsBatchBuilder {
    range: VwapZscoreWithSignalsBatchRange,
    kernel: Kernel,
}

impl VwapZscoreWithSignalsBatchBuilder {
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
    pub fn upper_bottom_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.upper_bottom = (start, end, step);
        self
    }

    #[inline]
    pub fn upper_bottom_static(mut self, upper_bottom: f64) -> Self {
        self.range.upper_bottom = (upper_bottom, upper_bottom, 0.0);
        self
    }

    #[inline]
    pub fn lower_bottom_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.lower_bottom = (start, end, step);
        self
    }

    #[inline]
    pub fn lower_bottom_static(mut self, lower_bottom: f64) -> Self {
        self.range.lower_bottom = (lower_bottom, lower_bottom, 0.0);
        self
    }

    #[inline]
    pub fn apply_slices(
        self,
        close: &[f64],
        volume: &[f64],
    ) -> Result<VwapZscoreWithSignalsBatchOutput, VwapZscoreWithSignalsError> {
        vwap_zscore_with_signals_batch_with_kernel(close, volume, &self.range, self.kernel)
    }

    #[inline]
    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<VwapZscoreWithSignalsBatchOutput, VwapZscoreWithSignalsError> {
        self.apply_slices(&candles.close, &candles.volume)
    }
}

#[derive(Clone, Debug)]
pub struct VwapZscoreWithSignalsBatchOutput {
    pub zvwap: Vec<f64>,
    pub support_signal: Vec<f64>,
    pub resistance_signal: Vec<f64>,
    pub combos: Vec<VwapZscoreWithSignalsParams>,
    pub rows: usize,
    pub cols: usize,
}

#[inline(always)]
fn expand_axis_usize(
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, VwapZscoreWithSignalsError> {
    if step == 0 || start == end {
        return Ok(vec![start]);
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
    }
    if out.is_empty() {
        return Err(VwapZscoreWithSignalsError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

#[inline(always)]
fn expand_axis_f64(
    (start, end, step): (f64, f64, f64),
) -> Result<Vec<f64>, VwapZscoreWithSignalsError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(VwapZscoreWithSignalsError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
        return Ok(vec![start]);
    }
    let mut out = Vec::new();
    if start < end {
        let st = step.abs();
        let mut x = start;
        while x <= end + 1e-12 {
            out.push(x);
            x += st;
        }
    } else {
        let st = -step.abs();
        let mut x = start;
        while x >= end - 1e-12 {
            out.push(x);
            x += st;
        }
    }
    if out.is_empty() {
        return Err(VwapZscoreWithSignalsError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

#[inline(always)]
fn expand_grid_vwap_zscore_with_signals(
    range: &VwapZscoreWithSignalsBatchRange,
) -> Result<Vec<VwapZscoreWithSignalsParams>, VwapZscoreWithSignalsError> {
    let lengths = expand_axis_usize(range.length)?;
    if let Some(&bad) = lengths.iter().find(|&&x| x == 0) {
        return Err(VwapZscoreWithSignalsError::InvalidLength {
            length: bad,
            data_len: 0,
        });
    }
    let uppers = expand_axis_f64(range.upper_bottom)?;
    if let Some(&bad) = uppers.iter().find(|&&x| !x.is_finite()) {
        return Err(VwapZscoreWithSignalsError::InvalidUpperBottom { upper_bottom: bad });
    }
    let lowers = expand_axis_f64(range.lower_bottom)?;
    if let Some(&bad) = lowers.iter().find(|&&x| !x.is_finite()) {
        return Err(VwapZscoreWithSignalsError::InvalidLowerBottom { lower_bottom: bad });
    }

    let mut out = Vec::with_capacity(lengths.len() * uppers.len() * lowers.len());
    for &length in &lengths {
        for &upper_bottom in &uppers {
            for &lower_bottom in &lowers {
                out.push(VwapZscoreWithSignalsParams {
                    length: Some(length),
                    upper_bottom: Some(upper_bottom),
                    lower_bottom: Some(lower_bottom),
                });
            }
        }
    }
    Ok(out)
}

#[inline]
pub fn vwap_zscore_with_signals_batch_with_kernel(
    close: &[f64],
    volume: &[f64],
    sweep: &VwapZscoreWithSignalsBatchRange,
    kernel: Kernel,
) -> Result<VwapZscoreWithSignalsBatchOutput, VwapZscoreWithSignalsError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(VwapZscoreWithSignalsError::InvalidKernelForBatch(other)),
    };
    vwap_zscore_with_signals_batch_par_slice(close, volume, sweep, batch_kernel.to_non_batch())
}

#[inline]
pub fn vwap_zscore_with_signals_batch_slice(
    close: &[f64],
    volume: &[f64],
    sweep: &VwapZscoreWithSignalsBatchRange,
    kernel: Kernel,
) -> Result<VwapZscoreWithSignalsBatchOutput, VwapZscoreWithSignalsError> {
    vwap_zscore_with_signals_batch_inner(close, volume, sweep, kernel, false)
}

#[inline]
pub fn vwap_zscore_with_signals_batch_par_slice(
    close: &[f64],
    volume: &[f64],
    sweep: &VwapZscoreWithSignalsBatchRange,
    kernel: Kernel,
) -> Result<VwapZscoreWithSignalsBatchOutput, VwapZscoreWithSignalsError> {
    vwap_zscore_with_signals_batch_inner(close, volume, sweep, kernel, true)
}

#[inline(always)]
fn vwap_zscore_with_signals_batch_inner(
    close: &[f64],
    volume: &[f64],
    sweep: &VwapZscoreWithSignalsBatchRange,
    _kernel: Kernel,
    parallel: bool,
) -> Result<VwapZscoreWithSignalsBatchOutput, VwapZscoreWithSignalsError> {
    let combos = expand_grid_vwap_zscore_with_signals(sweep)?;
    let rows = combos.len();
    let cols = close.len();
    if cols == 0 {
        return Err(VwapZscoreWithSignalsError::EmptyInputData);
    }
    if volume.len() != cols {
        return Err(VwapZscoreWithSignalsError::InconsistentSliceLengths {
            close_len: close.len(),
            volume_len: volume.len(),
        });
    }
    let first = first_valid_close_volume(close, volume);
    if first >= cols {
        return Err(VwapZscoreWithSignalsError::AllValuesNaN);
    }
    let valid = count_valid_close_volume(close, volume);
    let max_needed = combos
        .iter()
        .map(|combo| {
            combo
                .length
                .unwrap_or(20)
                .saturating_mul(2)
                .saturating_sub(1)
        })
        .max()
        .unwrap_or(0);
    if valid < max_needed {
        return Err(VwapZscoreWithSignalsError::NotEnoughValidData {
            needed: max_needed,
            valid,
        });
    }

    let mut zvwap_mu = make_uninit_matrix(rows, cols);
    let mut support_mu = make_uninit_matrix(rows, cols);
    let mut resistance_mu = make_uninit_matrix(rows, cols);
    let warmups: Vec<usize> = combos
        .iter()
        .map(|combo| {
            first
                .saturating_add(combo.length.unwrap_or(20).saturating_mul(2))
                .saturating_sub(2)
                .min(cols)
        })
        .collect();
    init_matrix_prefixes(&mut zvwap_mu, cols, &warmups);
    init_matrix_prefixes(&mut support_mu, cols, &warmups);
    init_matrix_prefixes(&mut resistance_mu, cols, &warmups);

    let mut zvwap_guard = ManuallyDrop::new(zvwap_mu);
    let mut support_guard = ManuallyDrop::new(support_mu);
    let mut resistance_guard = ManuallyDrop::new(resistance_mu);
    let zvwap_out = unsafe {
        std::slice::from_raw_parts_mut(zvwap_guard.as_mut_ptr() as *mut f64, zvwap_guard.len())
    };
    let support_out = unsafe {
        std::slice::from_raw_parts_mut(support_guard.as_mut_ptr() as *mut f64, support_guard.len())
    };
    let resistance_out = unsafe {
        std::slice::from_raw_parts_mut(
            resistance_guard.as_mut_ptr() as *mut f64,
            resistance_guard.len(),
        )
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        zvwap_out
            .par_chunks_mut(cols)
            .zip(support_out.par_chunks_mut(cols))
            .zip(resistance_out.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, ((dst_z, dst_s), dst_r))| {
                let combo = &combos[row];
                vwap_zscore_with_signals_row_from_slices(
                    close,
                    volume,
                    combo.length.unwrap_or(20),
                    combo.upper_bottom.unwrap_or(2.5),
                    combo.lower_bottom.unwrap_or(-2.5),
                    dst_z,
                    dst_s,
                    dst_r,
                );
            });

        #[cfg(target_arch = "wasm32")]
        for row in 0..rows {
            let start = row * cols;
            let end = start + cols;
            let combo = &combos[row];
            vwap_zscore_with_signals_row_from_slices(
                close,
                volume,
                combo.length.unwrap_or(20),
                combo.upper_bottom.unwrap_or(2.5),
                combo.lower_bottom.unwrap_or(-2.5),
                &mut zvwap_out[start..end],
                &mut support_out[start..end],
                &mut resistance_out[start..end],
            );
        }
    } else {
        for row in 0..rows {
            let start = row * cols;
            let end = start + cols;
            let combo = &combos[row];
            vwap_zscore_with_signals_row_from_slices(
                close,
                volume,
                combo.length.unwrap_or(20),
                combo.upper_bottom.unwrap_or(2.5),
                combo.lower_bottom.unwrap_or(-2.5),
                &mut zvwap_out[start..end],
                &mut support_out[start..end],
                &mut resistance_out[start..end],
            );
        }
    }

    let zvwap = unsafe {
        Vec::from_raw_parts(
            zvwap_guard.as_mut_ptr() as *mut f64,
            zvwap_guard.len(),
            zvwap_guard.capacity(),
        )
    };
    let support_signal = unsafe {
        Vec::from_raw_parts(
            support_guard.as_mut_ptr() as *mut f64,
            support_guard.len(),
            support_guard.capacity(),
        )
    };
    let resistance_signal = unsafe {
        Vec::from_raw_parts(
            resistance_guard.as_mut_ptr() as *mut f64,
            resistance_guard.len(),
            resistance_guard.capacity(),
        )
    };

    Ok(VwapZscoreWithSignalsBatchOutput {
        zvwap,
        support_signal,
        resistance_signal,
        combos,
        rows,
        cols,
    })
}

#[inline]
pub fn vwap_zscore_with_signals_batch_inner_into(
    close: &[f64],
    volume: &[f64],
    sweep: &VwapZscoreWithSignalsBatchRange,
    kernel: Kernel,
    zvwap_out: &mut [f64],
    support_out: &mut [f64],
    resistance_out: &mut [f64],
) -> Result<Vec<VwapZscoreWithSignalsParams>, VwapZscoreWithSignalsError> {
    let out = vwap_zscore_with_signals_batch_inner(close, volume, sweep, kernel, false)?;
    let total = out.rows * out.cols;
    if zvwap_out.len() != total || support_out.len() != total || resistance_out.len() != total {
        return Err(VwapZscoreWithSignalsError::OutputLengthMismatch {
            expected: total,
            zvwap_got: zvwap_out.len(),
            support_got: support_out.len(),
            resistance_got: resistance_out.len(),
        });
    }
    zvwap_out.copy_from_slice(&out.zvwap);
    support_out.copy_from_slice(&out.support_signal);
    resistance_out.copy_from_slice(&out.resistance_signal);
    Ok(out.combos)
}

#[cfg(feature = "python")]
#[pyclass(name = "VwapZscoreWithSignalsStream")]
pub struct VwapZscoreWithSignalsStreamPy {
    inner: VwapZscoreWithSignalsStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl VwapZscoreWithSignalsStreamPy {
    #[new]
    #[pyo3(signature = (length=20, upper_bottom=2.5, lower_bottom=-2.5))]
    fn new(length: usize, upper_bottom: f64, lower_bottom: f64) -> PyResult<Self> {
        let inner = VwapZscoreWithSignalsStream::try_new(VwapZscoreWithSignalsParams {
            length: Some(length),
            upper_bottom: Some(upper_bottom),
            lower_bottom: Some(lower_bottom),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn update(&mut self, close: f64, volume: f64) -> Option<(f64, f64, f64)> {
        self.inner.update(close, volume)
    }

    fn warmup_period(&self) -> usize {
        self.inner.get_warmup_period()
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "vwap_zscore_with_signals")]
#[pyo3(signature = (close, volume, length=20, upper_bottom=2.5, lower_bottom=-2.5, kernel=None))]
pub fn vwap_zscore_with_signals_py<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    length: usize,
    upper_bottom: f64,
    lower_bottom: f64,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let close = close.as_slice()?;
    let volume = volume.as_slice()?;
    if close.len() != volume.len() {
        return Err(PyValueError::new_err("Close/volume slice length mismatch"));
    }
    let kern = validate_kernel(kernel, false)?;
    let input = VwapZscoreWithSignalsInput::from_slices(
        close,
        volume,
        VwapZscoreWithSignalsParams {
            length: Some(length),
            upper_bottom: Some(upper_bottom),
            lower_bottom: Some(lower_bottom),
        },
    );
    let out = py
        .allow_threads(|| vwap_zscore_with_signals_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        out.zvwap.into_pyarray(py),
        out.support_signal.into_pyarray(py),
        out.resistance_signal.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyfunction(name = "vwap_zscore_with_signals_batch")]
#[pyo3(signature = (
    close,
    volume,
    length_range=(20, 20, 0),
    upper_bottom_range=(2.5, 2.5, 0.0),
    lower_bottom_range=(-2.5, -2.5, 0.0),
    kernel=None
))]
pub fn vwap_zscore_with_signals_batch_py<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    upper_bottom_range: (f64, f64, f64),
    lower_bottom_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let close = close.as_slice()?;
    let volume = volume.as_slice()?;
    if close.len() != volume.len() {
        return Err(PyValueError::new_err("Close/volume slice length mismatch"));
    }
    let kern = validate_kernel(kernel, true)?;
    let sweep = VwapZscoreWithSignalsBatchRange {
        length: length_range,
        upper_bottom: upper_bottom_range,
        lower_bottom: lower_bottom_range,
    };
    let combos = expand_grid_vwap_zscore_with_signals(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let zvwap_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let support_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let resistance_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let zvwap_slice = unsafe { zvwap_arr.as_slice_mut()? };
    let support_slice = unsafe { support_arr.as_slice_mut()? };
    let resistance_slice = unsafe { resistance_arr.as_slice_mut()? };

    let combos = py
        .allow_threads(|| {
            let batch = match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                other => other,
            };
            vwap_zscore_with_signals_batch_inner_into(
                close,
                volume,
                &sweep,
                batch.to_non_batch(),
                zvwap_slice,
                support_slice,
                resistance_slice,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("zvwap", zvwap_arr.reshape((rows, cols))?)?;
    dict.set_item("support_signal", support_arr.reshape((rows, cols))?)?;
    dict.set_item("resistance_signal", resistance_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "lengths",
        combos
            .iter()
            .map(|combo| combo.length.unwrap_or(20) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "upper_bottoms",
        combos
            .iter()
            .map(|combo| combo.upper_bottom.unwrap_or(2.5))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "lower_bottoms",
        combos
            .iter()
            .map(|combo| combo.lower_bottom.unwrap_or(-2.5))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_vwap_zscore_with_signals_module(
    module: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(vwap_zscore_with_signals_py, module)?)?;
    module.add_function(wrap_pyfunction!(vwap_zscore_with_signals_batch_py, module)?)?;
    module.add_class::<VwapZscoreWithSignalsStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "vwap_zscore_with_signals_js")]
pub fn vwap_zscore_with_signals_js(
    close: &[f64],
    volume: &[f64],
    length: usize,
    upper_bottom: f64,
    lower_bottom: f64,
) -> Result<JsValue, JsValue> {
    let input = VwapZscoreWithSignalsInput::from_slices(
        close,
        volume,
        VwapZscoreWithSignalsParams {
            length: Some(length),
            upper_bottom: Some(upper_bottom),
            lower_bottom: Some(lower_bottom),
        },
    );
    let out = vwap_zscore_with_signals(&input).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let result = js_sys::Object::new();

    let zvwap = js_sys::Float64Array::new_with_length(out.zvwap.len() as u32);
    zvwap.copy_from(&out.zvwap);
    js_sys::Reflect::set(&result, &JsValue::from_str("zvwap"), &zvwap)?;

    let support = js_sys::Float64Array::new_with_length(out.support_signal.len() as u32);
    support.copy_from(&out.support_signal);
    js_sys::Reflect::set(&result, &JsValue::from_str("support_signal"), &support)?;

    let resistance = js_sys::Float64Array::new_with_length(out.resistance_signal.len() as u32);
    resistance.copy_from(&out.resistance_signal);
    js_sys::Reflect::set(
        &result,
        &JsValue::from_str("resistance_signal"),
        &resistance,
    )?;

    Ok(result.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn vwap_zscore_with_signals_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn vwap_zscore_with_signals_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn vwap_zscore_with_signals_into(
    close_ptr: *const f64,
    volume_ptr: *const f64,
    zvwap_ptr: *mut f64,
    support_ptr: *mut f64,
    resistance_ptr: *mut f64,
    len: usize,
    length: usize,
    upper_bottom: f64,
    lower_bottom: f64,
) -> Result<(), JsValue> {
    if close_ptr.is_null()
        || volume_ptr.is_null()
        || zvwap_ptr.is_null()
        || support_ptr.is_null()
        || resistance_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let close = std::slice::from_raw_parts(close_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);
        let input = VwapZscoreWithSignalsInput::from_slices(
            close,
            volume,
            VwapZscoreWithSignalsParams {
                length: Some(length),
                upper_bottom: Some(upper_bottom),
                lower_bottom: Some(lower_bottom),
            },
        );
        let alias = close_ptr == zvwap_ptr
            || close_ptr == support_ptr
            || close_ptr == resistance_ptr
            || volume_ptr == zvwap_ptr
            || volume_ptr == support_ptr
            || volume_ptr == resistance_ptr;
        if alias {
            let mut zvwap_tmp = vec![0.0; len];
            let mut support_tmp = vec![0.0; len];
            let mut resistance_tmp = vec![0.0; len];
            vwap_zscore_with_signals_into_slices(
                &mut zvwap_tmp,
                &mut support_tmp,
                &mut resistance_tmp,
                &input,
                Kernel::Auto,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
            std::slice::from_raw_parts_mut(zvwap_ptr, len).copy_from_slice(&zvwap_tmp);
            std::slice::from_raw_parts_mut(support_ptr, len).copy_from_slice(&support_tmp);
            std::slice::from_raw_parts_mut(resistance_ptr, len).copy_from_slice(&resistance_tmp);
        } else {
            let zvwap_out = std::slice::from_raw_parts_mut(zvwap_ptr, len);
            let support_out = std::slice::from_raw_parts_mut(support_ptr, len);
            let resistance_out = std::slice::from_raw_parts_mut(resistance_ptr, len);
            vwap_zscore_with_signals_into_slices(
                zvwap_out,
                support_out,
                resistance_out,
                &input,
                Kernel::Auto,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct VwapZscoreWithSignalsBatchConfig {
    pub length_range: (usize, usize, usize),
    pub upper_bottom_range: Option<(f64, f64, f64)>,
    pub lower_bottom_range: Option<(f64, f64, f64)>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct VwapZscoreWithSignalsBatchJsOutput {
    pub zvwap: Vec<f64>,
    pub support_signal: Vec<f64>,
    pub resistance_signal: Vec<f64>,
    pub combos: Vec<VwapZscoreWithSignalsParams>,
    pub lengths: Vec<usize>,
    pub upper_bottoms: Vec<f64>,
    pub lower_bottoms: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "vwap_zscore_with_signals_batch_js")]
pub fn vwap_zscore_with_signals_batch_js(
    close: &[f64],
    volume: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: VwapZscoreWithSignalsBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let sweep = VwapZscoreWithSignalsBatchRange {
        length: config.length_range,
        upper_bottom: config.upper_bottom_range.unwrap_or((2.5, 2.5, 0.0)),
        lower_bottom: config.lower_bottom_range.unwrap_or((-2.5, -2.5, 0.0)),
    };
    let out =
        vwap_zscore_with_signals_batch_inner(close, volume, &sweep, detect_best_kernel(), false)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&VwapZscoreWithSignalsBatchJsOutput {
        lengths: out
            .combos
            .iter()
            .map(|combo| combo.length.unwrap_or(20))
            .collect(),
        upper_bottoms: out
            .combos
            .iter()
            .map(|combo| combo.upper_bottom.unwrap_or(2.5))
            .collect(),
        lower_bottoms: out
            .combos
            .iter()
            .map(|combo| combo.lower_bottom.unwrap_or(-2.5))
            .collect(),
        zvwap: out.zvwap,
        support_signal: out.support_signal,
        resistance_signal: out.resistance_signal,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn vwap_zscore_with_signals_batch_into(
    close_ptr: *const f64,
    volume_ptr: *const f64,
    zvwap_ptr: *mut f64,
    support_ptr: *mut f64,
    resistance_ptr: *mut f64,
    len: usize,
    length_start: usize,
    length_end: usize,
    length_step: usize,
    upper_bottom_start: f64,
    upper_bottom_end: f64,
    upper_bottom_step: f64,
    lower_bottom_start: f64,
    lower_bottom_end: f64,
    lower_bottom_step: f64,
) -> Result<usize, JsValue> {
    if close_ptr.is_null()
        || volume_ptr.is_null()
        || zvwap_ptr.is_null()
        || support_ptr.is_null()
        || resistance_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }
    let sweep = VwapZscoreWithSignalsBatchRange {
        length: (length_start, length_end, length_step),
        upper_bottom: (upper_bottom_start, upper_bottom_end, upper_bottom_step),
        lower_bottom: (lower_bottom_start, lower_bottom_end, lower_bottom_step),
    };
    let combos = expand_grid_vwap_zscore_with_signals(&sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    unsafe {
        let close = std::slice::from_raw_parts(close_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);
        let total = rows
            .checked_mul(len)
            .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
        let zvwap_out = std::slice::from_raw_parts_mut(zvwap_ptr, total);
        let support_out = std::slice::from_raw_parts_mut(support_ptr, total);
        let resistance_out = std::slice::from_raw_parts_mut(resistance_ptr, total);
        vwap_zscore_with_signals_batch_inner_into(
            close,
            volume,
            &sweep,
            detect_best_kernel(),
            zvwap_out,
            support_out,
            resistance_out,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }
    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn vwap_zscore_with_signals_output_into_js(
    close: &[f64],
    volume: &[f64],
    length: usize,
    upper_bottom: f64,
    lower_bottom: f64,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = vwap_zscore_with_signals_js(close, volume, length, upper_bottom, lower_bottom)?;
    crate::write_wasm_object_f64_outputs("vwap_zscore_with_signals_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn vwap_zscore_with_signals_batch_output_into_js(
    close: &[f64],
    volume: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = vwap_zscore_with_signals_batch_js(close, volume, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "vwap_zscore_with_signals_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use std::error::Error;

    fn load_close_volume() -> Result<(Vec<f64>, Vec<f64>), Box<dyn Error>> {
        let candles = read_candles_from_csv("src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv")?;
        Ok((candles.close, candles.volume))
    }

    fn assert_series_eq(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (&a, &b) in actual.iter().zip(expected.iter()) {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert!((a - b).abs() <= 1e-10, "expected {b}, got {a}");
        }
    }

    #[test]
    fn vwap_zscore_with_signals_output_contract() -> Result<(), Box<dyn Error>> {
        let (close, volume) = load_close_volume()?;
        let input = VwapZscoreWithSignalsInput::from_slices(
            &close,
            &volume,
            VwapZscoreWithSignalsParams::default(),
        );
        let out = vwap_zscore_with_signals_with_kernel(&input, Kernel::Scalar)?;
        assert_eq!(out.zvwap.len(), close.len());
        assert_eq!(out.support_signal.len(), close.len());
        assert_eq!(out.resistance_signal.len(), close.len());
        assert!(out.zvwap.iter().any(|v| v.is_finite()));
        assert!(out.support_signal.iter().any(|v| v.is_finite()));
        assert!(out.resistance_signal.iter().any(|v| v.is_finite()));
        Ok(())
    }

    #[test]
    fn vwap_zscore_with_signals_auto_matches_scalar() -> Result<(), Box<dyn Error>> {
        let (close, volume) = load_close_volume()?;
        let input = VwapZscoreWithSignalsInput::from_slices(
            &close,
            &volume,
            VwapZscoreWithSignalsParams {
                length: Some(17),
                upper_bottom: Some(2.25),
                lower_bottom: Some(-2.25),
            },
        );
        let auto = vwap_zscore_with_signals_with_kernel(&input, Kernel::Auto)?;
        let scalar = vwap_zscore_with_signals_with_kernel(&input, Kernel::Scalar)?;
        assert_series_eq(&auto.zvwap, &scalar.zvwap);
        assert_series_eq(&auto.support_signal, &scalar.support_signal);
        assert_series_eq(&auto.resistance_signal, &scalar.resistance_signal);
        Ok(())
    }

    #[test]
    fn vwap_zscore_with_signals_into_matches_api() -> Result<(), Box<dyn Error>> {
        let (close, volume) = load_close_volume()?;
        let input = VwapZscoreWithSignalsInput::from_slices(
            &close,
            &volume,
            VwapZscoreWithSignalsParams::default(),
        );
        let expected = vwap_zscore_with_signals(&input)?;
        let mut zvwap = vec![f64::NAN; close.len()];
        let mut support = vec![f64::NAN; close.len()];
        let mut resistance = vec![f64::NAN; close.len()];
        vwap_zscore_with_signals_into(&input, &mut zvwap, &mut support, &mut resistance)?;
        assert_series_eq(&zvwap, &expected.zvwap);
        assert_series_eq(&support, &expected.support_signal);
        assert_series_eq(&resistance, &expected.resistance_signal);
        Ok(())
    }

    #[test]
    fn vwap_zscore_with_signals_stream_matches_batch() -> Result<(), Box<dyn Error>> {
        let (close, volume) = load_close_volume()?;
        let input = VwapZscoreWithSignalsInput::from_slices(
            &close,
            &volume,
            VwapZscoreWithSignalsParams::default(),
        );
        let batch = vwap_zscore_with_signals_with_kernel(&input, Kernel::Scalar)?;
        let mut stream = VwapZscoreWithSignalsStream::try_new(input.params.clone())?;
        let mut zvwap = Vec::with_capacity(close.len());
        let mut support = Vec::with_capacity(close.len());
        let mut resistance = Vec::with_capacity(close.len());
        for (&c, &v) in close.iter().zip(volume.iter()) {
            if let Some((z, s, r)) = stream.update(c, v) {
                zvwap.push(z);
                support.push(s);
                resistance.push(r);
            } else {
                zvwap.push(f64::NAN);
                support.push(f64::NAN);
                resistance.push(f64::NAN);
            }
        }
        assert_series_eq(&zvwap, &batch.zvwap);
        assert_series_eq(&support, &batch.support_signal);
        assert_series_eq(&resistance, &batch.resistance_signal);
        Ok(())
    }

    #[test]
    fn vwap_zscore_with_signals_batch_single_matches_single() -> Result<(), Box<dyn Error>> {
        let (close, volume) = load_close_volume()?;
        let close = &close[..256];
        let volume = &volume[..256];
        let single = vwap_zscore_with_signals_with_kernel(
            &VwapZscoreWithSignalsInput::from_slices(
                close,
                volume,
                VwapZscoreWithSignalsParams::default(),
            ),
            Kernel::Scalar,
        )?;
        let batch = vwap_zscore_with_signals_batch_with_kernel(
            close,
            volume,
            &VwapZscoreWithSignalsBatchRange {
                length: (20, 20, 0),
                upper_bottom: (2.5, 2.5, 0.0),
                lower_bottom: (-2.5, -2.5, 0.0),
            },
            Kernel::Auto,
        )?;
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, close.len());
        assert_series_eq(&batch.zvwap[..close.len()], &single.zvwap);
        assert_series_eq(&batch.support_signal[..close.len()], &single.support_signal);
        assert_series_eq(
            &batch.resistance_signal[..close.len()],
            &single.resistance_signal,
        );
        Ok(())
    }
}
