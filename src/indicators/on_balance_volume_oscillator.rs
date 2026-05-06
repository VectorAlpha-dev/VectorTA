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
use crate::utilities::helpers::{alloc_with_nan_prefix, detect_best_batch_kernel};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use thiserror::Error;

impl<'a> AsRef<[f64]> for OnBalanceVolumeOscillatorInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            OnBalanceVolumeOscillatorData::Candles { candles, source } => match *source {
                "close" => candles.close.as_slice(),
                _ => source_type(candles, source),
            },
            OnBalanceVolumeOscillatorData::Slices { source, .. } => source,
        }
    }
}

#[derive(Debug, Clone)]
pub enum OnBalanceVolumeOscillatorData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slices {
        source: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct OnBalanceVolumeOscillatorOutput {
    pub line: Vec<f64>,
    pub signal: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct OnBalanceVolumeOscillatorParams {
    pub obv_length: Option<usize>,
    pub ema_length: Option<usize>,
}

impl Default for OnBalanceVolumeOscillatorParams {
    fn default() -> Self {
        Self {
            obv_length: Some(20),
            ema_length: Some(9),
        }
    }
}

#[derive(Debug, Clone)]
pub struct OnBalanceVolumeOscillatorInput<'a> {
    pub data: OnBalanceVolumeOscillatorData<'a>,
    pub params: OnBalanceVolumeOscillatorParams,
}

impl<'a> OnBalanceVolumeOscillatorInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: OnBalanceVolumeOscillatorParams,
    ) -> Self {
        Self {
            data: OnBalanceVolumeOscillatorData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        source: &'a [f64],
        volume: &'a [f64],
        params: OnBalanceVolumeOscillatorParams,
    ) -> Self {
        Self {
            data: OnBalanceVolumeOscillatorData::Slices { source, volume },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", OnBalanceVolumeOscillatorParams::default())
    }

    #[inline]
    pub fn get_obv_length(&self) -> usize {
        self.params.obv_length.unwrap_or(20)
    }

    #[inline]
    pub fn get_ema_length(&self) -> usize {
        self.params.ema_length.unwrap_or(9)
    }

    #[inline]
    pub fn as_refs(&'a self) -> (&'a [f64], &'a [f64]) {
        match &self.data {
            OnBalanceVolumeOscillatorData::Candles { candles, source } => (
                match *source {
                    "close" => candles.close.as_slice(),
                    _ => source_type(candles, source),
                },
                candles.volume.as_slice(),
            ),
            OnBalanceVolumeOscillatorData::Slices { source, volume } => (*source, *volume),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct OnBalanceVolumeOscillatorBuilder {
    obv_length: Option<usize>,
    ema_length: Option<usize>,
    kernel: Kernel,
}

impl Default for OnBalanceVolumeOscillatorBuilder {
    fn default() -> Self {
        Self {
            obv_length: None,
            ema_length: None,
            kernel: Kernel::Auto,
        }
    }
}

impl OnBalanceVolumeOscillatorBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn obv_length(mut self, value: usize) -> Self {
        self.obv_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn ema_length(mut self, value: usize) -> Self {
        self.ema_length = Some(value);
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
    ) -> Result<OnBalanceVolumeOscillatorOutput, OnBalanceVolumeOscillatorError> {
        self.apply_source(candles, "close")
    }

    #[inline(always)]
    pub fn apply_source(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<OnBalanceVolumeOscillatorOutput, OnBalanceVolumeOscillatorError> {
        let input = OnBalanceVolumeOscillatorInput::from_candles(
            candles,
            source,
            OnBalanceVolumeOscillatorParams {
                obv_length: self.obv_length,
                ema_length: self.ema_length,
            },
        );
        on_balance_volume_oscillator_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        source: &[f64],
        volume: &[f64],
    ) -> Result<OnBalanceVolumeOscillatorOutput, OnBalanceVolumeOscillatorError> {
        let input = OnBalanceVolumeOscillatorInput::from_slices(
            source,
            volume,
            OnBalanceVolumeOscillatorParams {
                obv_length: self.obv_length,
                ema_length: self.ema_length,
            },
        );
        on_balance_volume_oscillator_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<OnBalanceVolumeOscillatorStream, OnBalanceVolumeOscillatorError> {
        OnBalanceVolumeOscillatorStream::try_new(OnBalanceVolumeOscillatorParams {
            obv_length: self.obv_length,
            ema_length: self.ema_length,
        })
    }
}

#[derive(Debug, Error)]
pub enum OnBalanceVolumeOscillatorError {
    #[error("on_balance_volume_oscillator: Empty input data.")]
    EmptyInputData,
    #[error(
        "on_balance_volume_oscillator: Data length mismatch: source_len = {source_len}, volume_len = {volume_len}"
    )]
    DataLengthMismatch {
        source_len: usize,
        volume_len: usize,
    },
    #[error("on_balance_volume_oscillator: All values are NaN.")]
    AllValuesNaN,
    #[error(
        "on_balance_volume_oscillator: Invalid OBV length: obv_length = {obv_length}, data length = {data_len}"
    )]
    InvalidObvLength { obv_length: usize, data_len: usize },
    #[error("on_balance_volume_oscillator: Invalid EMA length: ema_length = {ema_length}")]
    InvalidEmaLength { ema_length: usize },
    #[error(
        "on_balance_volume_oscillator: Not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "on_balance_volume_oscillator: Output length mismatch: expected = {expected}, got = {got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("on_balance_volume_oscillator: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("on_balance_volume_oscillator: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline(always)]
fn is_valid_bar(source: f64, volume: f64) -> bool {
    source.is_finite() && volume.is_finite()
}

#[inline(always)]
fn first_and_valid_run_until(
    source: &[f64],
    volume: &[f64],
    needed: usize,
) -> (Option<usize>, usize) {
    let mut first = None;
    let mut best = 0usize;
    let mut run = 0usize;
    for i in 0..source.len() {
        if is_valid_bar(source[i], volume[i]) {
            if first.is_none() {
                first = Some(i);
            }
            run += 1;
            if run > best {
                best = run;
                if best >= needed {
                    break;
                }
            }
        } else {
            run = 0;
        }
    }
    (first, best)
}

#[inline(always)]
fn first_valid_bar(source: &[f64], volume: &[f64]) -> Option<usize> {
    (0..source.len()).find(|&i| is_valid_bar(source[i], volume[i]))
}

#[inline(always)]
fn max_valid_run_length(source: &[f64], volume: &[f64]) -> usize {
    first_and_valid_run_until(source, volume, usize::MAX).1
}

#[inline(always)]
fn line_warmup(obv_length: usize, first: usize) -> usize {
    first + obv_length - 1
}

#[inline(always)]
fn signal_warmup(obv_length: usize, first: usize) -> usize {
    line_warmup(obv_length, first)
}

#[derive(Clone, Debug)]
struct RollingWindowSum {
    period: usize,
    count: usize,
    head: usize,
    sum: f64,
    buffer: Vec<f64>,
}

impl RollingWindowSum {
    #[inline]
    fn new(period: usize) -> Self {
        Self {
            period,
            count: 0,
            head: 0,
            sum: 0.0,
            buffer: vec![0.0; period.max(1)],
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.count = 0;
        self.head = 0;
        self.sum = 0.0;
        self.buffer.fill(0.0);
    }

    #[inline]
    fn update(&mut self, value: f64) -> Option<f64> {
        if self.count < self.period {
            self.buffer[self.count] = value;
            self.sum += value;
            self.count += 1;
            if self.count == self.period {
                return Some(self.sum);
            }
            return None;
        }

        let old = self.buffer[self.head];
        self.buffer[self.head] = value;
        self.sum += value - old;
        self.head += 1;
        if self.head == self.period {
            self.head = 0;
        }
        Some(self.sum)
    }
}

#[derive(Clone, Debug)]
pub struct OnBalanceVolumeOscillatorStream {
    alpha: f64,
    prev_source: f64,
    has_prev_source: bool,
    signed_window: RollingWindowSum,
    volume_window: RollingWindowSum,
    ema_value: f64,
    ema_initialized: bool,
}

impl OnBalanceVolumeOscillatorStream {
    #[inline]
    fn from_parts(obv_length: usize, ema_length: usize) -> Self {
        let alpha = 2.0 / (ema_length as f64 + 1.0);
        Self {
            alpha,
            prev_source: f64::NAN,
            has_prev_source: false,
            signed_window: RollingWindowSum::new(obv_length),
            volume_window: RollingWindowSum::new(obv_length),
            ema_value: f64::NAN,
            ema_initialized: false,
        }
    }

    #[inline]
    pub fn try_new(
        params: OnBalanceVolumeOscillatorParams,
    ) -> Result<Self, OnBalanceVolumeOscillatorError> {
        let obv_length = params.obv_length.unwrap_or(20);
        let ema_length = params.ema_length.unwrap_or(9);
        if obv_length == 0 {
            return Err(OnBalanceVolumeOscillatorError::InvalidObvLength {
                obv_length,
                data_len: 0,
            });
        }
        if ema_length == 0 {
            return Err(OnBalanceVolumeOscillatorError::InvalidEmaLength { ema_length });
        }
        Ok(Self::from_parts(obv_length, ema_length))
    }

    #[inline]
    pub fn reset(&mut self) {
        self.prev_source = f64::NAN;
        self.has_prev_source = false;
        self.signed_window.reset();
        self.volume_window.reset();
        self.ema_value = f64::NAN;
        self.ema_initialized = false;
    }

    #[inline]
    pub fn update(&mut self, source: f64, volume: f64) -> Option<(f64, f64)> {
        if !is_valid_bar(source, volume) {
            return None;
        }

        let signed_volume = if self.has_prev_source {
            let sign =
                ((source > self.prev_source) as i32 - (source < self.prev_source) as i32) as f64;
            self.prev_source = source;
            volume * sign
        } else {
            self.prev_source = source;
            self.has_prev_source = true;
            0.0
        };

        let signed_sum = self.signed_window.update(signed_volume);
        let volume_sum = self.volume_window.update(volume);
        let (signed_sum, volume_sum) = match (signed_sum, volume_sum) {
            (Some(signed_sum), Some(volume_sum)) => (signed_sum, volume_sum),
            _ => return None,
        };
        let line = if volume_sum == 0.0 {
            f64::NAN
        } else {
            signed_sum / volume_sum
        };

        let signal = if line.is_finite() {
            if self.ema_initialized {
                self.ema_value += self.alpha * (line - self.ema_value);
            } else {
                self.ema_value = line;
                self.ema_initialized = true;
            }
            self.ema_value
        } else {
            f64::NAN
        };

        Some((line, signal))
    }

    #[inline]
    pub fn update_reset_on_nan(&mut self, source: f64, volume: f64) -> Option<(f64, f64)> {
        if !is_valid_bar(source, volume) {
            self.reset();
            return None;
        }
        self.update(source, volume)
    }
}

#[inline(always)]
fn on_balance_volume_oscillator_prepare<'a>(
    input: &'a OnBalanceVolumeOscillatorInput,
) -> Result<(&'a [f64], &'a [f64], usize, usize, usize), OnBalanceVolumeOscillatorError> {
    let (source, volume) = input.as_refs();
    let data_len = source.len();
    if data_len == 0 {
        return Err(OnBalanceVolumeOscillatorError::EmptyInputData);
    }
    if volume.len() != data_len {
        return Err(OnBalanceVolumeOscillatorError::DataLengthMismatch {
            source_len: data_len,
            volume_len: volume.len(),
        });
    }

    let obv_length = input.get_obv_length();
    if obv_length == 0 || obv_length > data_len {
        return Err(OnBalanceVolumeOscillatorError::InvalidObvLength {
            obv_length,
            data_len,
        });
    }

    let ema_length = input.get_ema_length();
    if ema_length == 0 {
        return Err(OnBalanceVolumeOscillatorError::InvalidEmaLength { ema_length });
    }

    let (first, valid) = first_and_valid_run_until(source, volume, obv_length);
    let first = first.ok_or(OnBalanceVolumeOscillatorError::AllValuesNaN)?;
    if valid < obv_length {
        return Err(OnBalanceVolumeOscillatorError::NotEnoughValidData {
            needed: obv_length,
            valid,
        });
    }

    Ok((source, volume, obv_length, ema_length, first))
}

#[inline(always)]
fn on_balance_volume_oscillator_compute_default_20_9_into(
    source: &[f64],
    volume: &[f64],
    out_line: &mut [f64],
    out_signal: &mut [f64],
) {
    let mut prev_source = f64::NAN;
    let mut has_prev_source = false;
    let mut signed_buffer = [0.0f64; 20];
    let mut volume_buffer = [0.0f64; 20];
    let mut count = 0usize;
    let mut head = 0usize;
    let mut signed_sum = 0.0;
    let mut volume_sum = 0.0;
    let mut ema_value = f64::NAN;
    let mut ema_initialized = false;

    let mut i = 0usize;
    while i < source.len() {
        let value = unsafe { *source.get_unchecked(i) };
        let vol = unsafe { *volume.get_unchecked(i) };
        if !is_valid_bar(value, vol) {
            prev_source = f64::NAN;
            has_prev_source = false;
            count = 0;
            head = 0;
            signed_sum = 0.0;
            volume_sum = 0.0;
            ema_value = f64::NAN;
            ema_initialized = false;
            unsafe {
                *out_line.get_unchecked_mut(i) = f64::NAN;
                *out_signal.get_unchecked_mut(i) = f64::NAN;
            }
            i += 1;
            continue;
        }

        let signed_volume = if has_prev_source {
            let sign = ((value > prev_source) as i32 - (value < prev_source) as i32) as f64;
            prev_source = value;
            vol * sign
        } else {
            prev_source = value;
            has_prev_source = true;
            0.0
        };

        let ready = if count < 20 {
            signed_buffer[count] = signed_volume;
            volume_buffer[count] = vol;
            signed_sum += signed_volume;
            volume_sum += vol;
            count += 1;
            count == 20
        } else {
            let old_signed = signed_buffer[head];
            let old_volume = volume_buffer[head];
            signed_buffer[head] = signed_volume;
            volume_buffer[head] = vol;
            signed_sum += signed_volume - old_signed;
            volume_sum += vol - old_volume;
            head += 1;
            if head == 20 {
                head = 0;
            }
            true
        };

        if ready {
            let line = if volume_sum == 0.0 {
                f64::NAN
            } else {
                signed_sum / volume_sum
            };
            let signal = if line.is_finite() {
                if ema_initialized {
                    ema_value += 0.2 * (line - ema_value);
                } else {
                    ema_value = line;
                    ema_initialized = true;
                }
                ema_value
            } else {
                f64::NAN
            };
            unsafe {
                *out_line.get_unchecked_mut(i) = line;
                *out_signal.get_unchecked_mut(i) = signal;
            }
        } else {
            unsafe {
                *out_line.get_unchecked_mut(i) = f64::NAN;
                *out_signal.get_unchecked_mut(i) = f64::NAN;
            }
        }
        i += 1;
    }
}

#[inline(always)]
fn on_balance_volume_oscillator_compute_into(
    source: &[f64],
    volume: &[f64],
    obv_length: usize,
    ema_length: usize,
    _kernel: Kernel,
    out_line: &mut [f64],
    out_signal: &mut [f64],
) {
    if obv_length == 20 && ema_length == 9 {
        on_balance_volume_oscillator_compute_default_20_9_into(
            source, volume, out_line, out_signal,
        );
        return;
    }

    let mut stream = OnBalanceVolumeOscillatorStream::from_parts(obv_length, ema_length);
    for i in 0..source.len() {
        match stream.update_reset_on_nan(source[i], volume[i]) {
            Some((line, signal)) => {
                out_line[i] = line;
                out_signal[i] = signal;
            }
            None => {
                out_line[i] = f64::NAN;
                out_signal[i] = f64::NAN;
            }
        }
    }
}

#[inline(always)]
pub fn on_balance_volume_oscillator(
    input: &OnBalanceVolumeOscillatorInput,
) -> Result<OnBalanceVolumeOscillatorOutput, OnBalanceVolumeOscillatorError> {
    on_balance_volume_oscillator_with_kernel(input, Kernel::Scalar)
}

pub fn on_balance_volume_oscillator_with_kernel(
    input: &OnBalanceVolumeOscillatorInput,
    kernel: Kernel,
) -> Result<OnBalanceVolumeOscillatorOutput, OnBalanceVolumeOscillatorError> {
    let (source, volume, obv_length, ema_length, first) =
        on_balance_volume_oscillator_prepare(input)?;

    let mut line = alloc_with_nan_prefix(source.len(), 0);
    let mut signal = alloc_with_nan_prefix(source.len(), 0);
    on_balance_volume_oscillator_compute_into(
        source,
        volume,
        obv_length,
        ema_length,
        kernel,
        &mut line,
        &mut signal,
    );

    Ok(OnBalanceVolumeOscillatorOutput { line, signal })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn on_balance_volume_oscillator_into(
    input: &OnBalanceVolumeOscillatorInput,
    out_line: &mut [f64],
    out_signal: &mut [f64],
) -> Result<(), OnBalanceVolumeOscillatorError> {
    on_balance_volume_oscillator_into_slice(out_line, out_signal, input, Kernel::Auto)
}

pub fn on_balance_volume_oscillator_into_slice(
    out_line: &mut [f64],
    out_signal: &mut [f64],
    input: &OnBalanceVolumeOscillatorInput,
    kernel: Kernel,
) -> Result<(), OnBalanceVolumeOscillatorError> {
    let (source, volume, obv_length, ema_length, _) = on_balance_volume_oscillator_prepare(input)?;
    if out_line.len() != source.len() || out_signal.len() != source.len() {
        return Err(OnBalanceVolumeOscillatorError::OutputLengthMismatch {
            expected: source.len(),
            got: out_line.len().max(out_signal.len()),
        });
    }

    on_balance_volume_oscillator_compute_into(
        source, volume, obv_length, ema_length, kernel, out_line, out_signal,
    );
    Ok(())
}

#[derive(Clone, Debug)]
pub struct OnBalanceVolumeOscillatorBatchRange {
    pub obv_length: (usize, usize, usize),
    pub ema_length: (usize, usize, usize),
}

impl Default for OnBalanceVolumeOscillatorBatchRange {
    fn default() -> Self {
        Self {
            obv_length: (20, 20, 0),
            ema_length: (9, 9, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct OnBalanceVolumeOscillatorBatchBuilder {
    range: OnBalanceVolumeOscillatorBatchRange,
    kernel: Kernel,
}

impl OnBalanceVolumeOscillatorBatchBuilder {
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
    pub fn obv_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.obv_length = (start, end, step);
        self
    }

    #[inline]
    pub fn obv_length_static(mut self, value: usize) -> Self {
        self.range.obv_length = (value, value, 0);
        self
    }

    #[inline]
    pub fn ema_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.ema_length = (start, end, step);
        self
    }

    #[inline]
    pub fn ema_length_static(mut self, value: usize) -> Self {
        self.range.ema_length = (value, value, 0);
        self
    }

    #[inline]
    pub fn apply_source(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<OnBalanceVolumeOscillatorBatchOutput, OnBalanceVolumeOscillatorError> {
        self.apply_slices(source_type(candles, source), source_type(candles, "volume"))
    }

    #[inline]
    pub fn apply_slices(
        self,
        source: &[f64],
        volume: &[f64],
    ) -> Result<OnBalanceVolumeOscillatorBatchOutput, OnBalanceVolumeOscillatorError> {
        on_balance_volume_oscillator_batch_with_kernel(source, volume, &self.range, self.kernel)
    }
}

#[derive(Clone, Debug)]
pub struct OnBalanceVolumeOscillatorBatchOutput {
    pub line: Vec<f64>,
    pub signal: Vec<f64>,
    pub combos: Vec<OnBalanceVolumeOscillatorParams>,
    pub rows: usize,
    pub cols: usize,
}

impl OnBalanceVolumeOscillatorBatchOutput {
    #[inline]
    pub fn row_for_params(&self, params: &OnBalanceVolumeOscillatorParams) -> Option<usize> {
        let target_obv = params.obv_length.unwrap_or(20);
        let target_ema = params.ema_length.unwrap_or(9);
        self.combos.iter().position(|combo| {
            combo.obv_length.unwrap_or(20) == target_obv
                && combo.ema_length.unwrap_or(9) == target_ema
        })
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct OnBalanceVolumeOscillatorBatchConfig {
    pub obv_length_range: Vec<usize>,
    pub ema_length_range: Vec<usize>,
}

#[inline]
fn axis_usize(
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, OnBalanceVolumeOscillatorError> {
    if start == 0 || end == 0 {
        return Err(OnBalanceVolumeOscillatorError::InvalidRange {
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
        return Err(OnBalanceVolumeOscillatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

#[inline]
pub fn expand_grid_on_balance_volume_oscillator(
    sweep: &OnBalanceVolumeOscillatorBatchRange,
) -> Result<Vec<OnBalanceVolumeOscillatorParams>, OnBalanceVolumeOscillatorError> {
    let obv_lengths = axis_usize(sweep.obv_length)?;
    let ema_lengths = axis_usize(sweep.ema_length)?;
    let mut combos = Vec::with_capacity(obv_lengths.len().saturating_mul(ema_lengths.len()));
    for obv_length in obv_lengths {
        for ema_length in ema_lengths.iter().copied() {
            combos.push(OnBalanceVolumeOscillatorParams {
                obv_length: Some(obv_length),
                ema_length: Some(ema_length),
            });
        }
    }
    Ok(combos)
}

#[inline]
pub fn on_balance_volume_oscillator_batch_with_kernel(
    source: &[f64],
    volume: &[f64],
    sweep: &OnBalanceVolumeOscillatorBatchRange,
    kernel: Kernel,
) -> Result<OnBalanceVolumeOscillatorBatchOutput, OnBalanceVolumeOscillatorError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(OnBalanceVolumeOscillatorError::InvalidKernelForBatch(other)),
    };
    on_balance_volume_oscillator_batch_impl(
        source,
        volume,
        sweep,
        batch_kernel.to_non_batch(),
        true,
    )
}

#[inline]
pub fn on_balance_volume_oscillator_batch_slice(
    source: &[f64],
    volume: &[f64],
    sweep: &OnBalanceVolumeOscillatorBatchRange,
) -> Result<OnBalanceVolumeOscillatorBatchOutput, OnBalanceVolumeOscillatorError> {
    on_balance_volume_oscillator_batch_impl(source, volume, sweep, Kernel::Scalar, false)
}

#[inline]
pub fn on_balance_volume_oscillator_batch_par_slice(
    source: &[f64],
    volume: &[f64],
    sweep: &OnBalanceVolumeOscillatorBatchRange,
) -> Result<OnBalanceVolumeOscillatorBatchOutput, OnBalanceVolumeOscillatorError> {
    on_balance_volume_oscillator_batch_impl(source, volume, sweep, Kernel::Scalar, true)
}

fn on_balance_volume_oscillator_batch_impl(
    source: &[f64],
    volume: &[f64],
    sweep: &OnBalanceVolumeOscillatorBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<OnBalanceVolumeOscillatorBatchOutput, OnBalanceVolumeOscillatorError> {
    let combos = expand_grid_on_balance_volume_oscillator(sweep)?;
    let rows = combos.len();
    let cols = source.len();

    if cols == 0 {
        return Err(OnBalanceVolumeOscillatorError::EmptyInputData);
    }
    if volume.len() != cols {
        return Err(OnBalanceVolumeOscillatorError::DataLengthMismatch {
            source_len: cols,
            volume_len: volume.len(),
        });
    }

    let valid = max_valid_run_length(source, volume);
    first_valid_bar(source, volume).ok_or(OnBalanceVolumeOscillatorError::AllValuesNaN)?;
    for params in &combos {
        let obv_length = params.obv_length.unwrap_or(20);
        if obv_length == 0 || obv_length > cols {
            return Err(OnBalanceVolumeOscillatorError::InvalidObvLength {
                obv_length,
                data_len: cols,
            });
        }
        let ema_length = params.ema_length.unwrap_or(9);
        if ema_length == 0 {
            return Err(OnBalanceVolumeOscillatorError::InvalidEmaLength { ema_length });
        }
        if valid < obv_length {
            return Err(OnBalanceVolumeOscillatorError::NotEnoughValidData {
                needed: obv_length,
                valid,
            });
        }
    }

    let total =
        rows.checked_mul(cols)
            .ok_or(OnBalanceVolumeOscillatorError::OutputLengthMismatch {
                expected: usize::MAX,
                got: 0,
            })?;
    let mut line = vec![f64::NAN; total];
    let mut signal = vec![f64::NAN; total];

    on_balance_volume_oscillator_batch_inner_into(
        source,
        volume,
        sweep,
        kernel,
        parallel,
        &mut line,
        &mut signal,
    )?;

    Ok(OnBalanceVolumeOscillatorBatchOutput {
        line,
        signal,
        combos,
        rows,
        cols,
    })
}

fn on_balance_volume_oscillator_batch_inner_into(
    source: &[f64],
    volume: &[f64],
    sweep: &OnBalanceVolumeOscillatorBatchRange,
    kernel: Kernel,
    parallel: bool,
    out_line: &mut [f64],
    out_signal: &mut [f64],
) -> Result<Vec<OnBalanceVolumeOscillatorParams>, OnBalanceVolumeOscillatorError> {
    let combos = expand_grid_on_balance_volume_oscillator(sweep)?;
    let rows = combos.len();
    let cols = source.len();
    let total =
        rows.checked_mul(cols)
            .ok_or(OnBalanceVolumeOscillatorError::OutputLengthMismatch {
                expected: usize::MAX,
                got: 0,
            })?;
    if out_line.len() != total || out_signal.len() != total {
        return Err(OnBalanceVolumeOscillatorError::OutputLengthMismatch {
            expected: total,
            got: out_line.len().max(out_signal.len()),
        });
    }
    if cols == 0 {
        return Err(OnBalanceVolumeOscillatorError::EmptyInputData);
    }
    if volume.len() != cols {
        return Err(OnBalanceVolumeOscillatorError::DataLengthMismatch {
            source_len: cols,
            volume_len: volume.len(),
        });
    }

    let valid = max_valid_run_length(source, volume);
    first_valid_bar(source, volume).ok_or(OnBalanceVolumeOscillatorError::AllValuesNaN)?;
    for params in &combos {
        let obv_length = params.obv_length.unwrap_or(20);
        if obv_length == 0 || obv_length > cols {
            return Err(OnBalanceVolumeOscillatorError::InvalidObvLength {
                obv_length,
                data_len: cols,
            });
        }
        let ema_length = params.ema_length.unwrap_or(9);
        if ema_length == 0 {
            return Err(OnBalanceVolumeOscillatorError::InvalidEmaLength { ema_length });
        }
        if valid < obv_length {
            return Err(OnBalanceVolumeOscillatorError::NotEnoughValidData {
                needed: obv_length,
                valid,
            });
        }
    }

    let do_row = |row: usize, line_row: &mut [f64], signal_row: &mut [f64]| {
        let params = &combos[row];
        on_balance_volume_oscillator_compute_into(
            source,
            volume,
            params.obv_length.unwrap_or(20),
            params.ema_length.unwrap_or(9),
            kernel,
            line_row,
            signal_row,
        );
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out_line
            .par_chunks_mut(cols)
            .zip(out_signal.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, (line_row, signal_row))| do_row(row, line_row, signal_row));
        #[cfg(target_arch = "wasm32")]
        for (row, (line_row, signal_row)) in out_line
            .chunks_mut(cols)
            .zip(out_signal.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, line_row, signal_row);
        }
    } else {
        for (row, (line_row, signal_row)) in out_line
            .chunks_mut(cols)
            .zip(out_signal.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, line_row, signal_row);
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "on_balance_volume_oscillator")]
#[pyo3(signature = (source, volume, obv_length=20, ema_length=9, kernel=None))]
pub fn on_balance_volume_oscillator_py<'py>(
    py: Python<'py>,
    source: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    obv_length: usize,
    ema_length: usize,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let source = source.as_slice()?;
    let volume = volume.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = OnBalanceVolumeOscillatorInput::from_slices(
        source,
        volume,
        OnBalanceVolumeOscillatorParams {
            obv_length: Some(obv_length),
            ema_length: Some(ema_length),
        },
    );
    let output = py
        .allow_threads(|| on_balance_volume_oscillator_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((output.line.into_pyarray(py), output.signal.into_pyarray(py)))
}

#[cfg(feature = "python")]
#[pyclass(name = "OnBalanceVolumeOscillatorStream")]
pub struct OnBalanceVolumeOscillatorStreamPy {
    stream: OnBalanceVolumeOscillatorStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl OnBalanceVolumeOscillatorStreamPy {
    #[new]
    #[pyo3(signature = (obv_length=20, ema_length=9))]
    fn new(obv_length: usize, ema_length: usize) -> PyResult<Self> {
        let stream = OnBalanceVolumeOscillatorStream::try_new(OnBalanceVolumeOscillatorParams {
            obv_length: Some(obv_length),
            ema_length: Some(ema_length),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, source: f64, volume: f64) -> Option<(f64, f64)> {
        self.stream.update_reset_on_nan(source, volume)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "on_balance_volume_oscillator_batch")]
#[pyo3(signature = (source, volume, obv_length_range, ema_length_range, kernel=None))]
pub fn on_balance_volume_oscillator_batch_py<'py>(
    py: Python<'py>,
    source: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    obv_length_range: (usize, usize, usize),
    ema_length_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let source = source.as_slice()?;
    let volume = volume.as_slice()?;
    let sweep = OnBalanceVolumeOscillatorBatchRange {
        obv_length: obv_length_range,
        ema_length: ema_length_range,
    };
    let combos = expand_grid_on_balance_volume_oscillator(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = source.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;
    let line_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let signal_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let line_out = unsafe { line_arr.as_slice_mut()? };
    let signal_out = unsafe { signal_arr.as_slice_mut()? };
    let kernel = validate_kernel(kernel, true)?;

    py.allow_threads(|| {
        let batch_kernel = match kernel {
            Kernel::Auto => detect_best_batch_kernel(),
            other => other,
        };
        on_balance_volume_oscillator_batch_inner_into(
            source,
            volume,
            &sweep,
            batch_kernel.to_non_batch(),
            true,
            line_out,
            signal_out,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("line", line_arr.reshape((rows, cols))?)?;
    dict.set_item("signal", signal_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "obv_lengths",
        combos
            .iter()
            .map(|p| p.obv_length.unwrap_or(20) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "ema_lengths",
        combos
            .iter()
            .map(|p| p.ema_length.unwrap_or(9) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "on_balance_volume_oscillator_js")]
pub fn on_balance_volume_oscillator_js(
    source: &[f64],
    volume: &[f64],
    obv_length: usize,
    ema_length: usize,
) -> Result<JsValue, JsValue> {
    let input = OnBalanceVolumeOscillatorInput::from_slices(
        source,
        volume,
        OnBalanceVolumeOscillatorParams {
            obv_length: Some(obv_length),
            ema_length: Some(ema_length),
        },
    );
    let mut line = vec![0.0; source.len()];
    let mut signal = vec![0.0; source.len()];
    on_balance_volume_oscillator_into_slice(&mut line, &mut signal, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("line"),
        &serde_wasm_bindgen::to_value(&line).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("signal"),
        &serde_wasm_bindgen::to_value(&signal).unwrap(),
    )?;
    Ok(obj.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "on_balance_volume_oscillator_batch_js")]
pub fn on_balance_volume_oscillator_batch_js(
    source: &[f64],
    volume: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: OnBalanceVolumeOscillatorBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.obv_length_range.len() != 3 || config.ema_length_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: ranges must have exactly 3 elements [start, end, step]",
        ));
    }

    let sweep = OnBalanceVolumeOscillatorBatchRange {
        obv_length: (
            config.obv_length_range[0],
            config.obv_length_range[1],
            config.obv_length_range[2],
        ),
        ema_length: (
            config.ema_length_range[0],
            config.ema_length_range[1],
            config.ema_length_range[2],
        ),
    };
    let combos = expand_grid_on_balance_volume_oscillator(&sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let cols = source.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
    let mut line = vec![0.0; total];
    let mut signal = vec![0.0; total];
    on_balance_volume_oscillator_batch_inner_into(
        source,
        volume,
        &sweep,
        Kernel::Scalar,
        false,
        &mut line,
        &mut signal,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("line"),
        &serde_wasm_bindgen::to_value(&line).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("signal"),
        &serde_wasm_bindgen::to_value(&signal).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("obv_lengths"),
        &serde_wasm_bindgen::to_value(
            &combos
                .iter()
                .map(|p| p.obv_length.unwrap_or(20))
                .collect::<Vec<_>>(),
        )
        .unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("ema_lengths"),
        &serde_wasm_bindgen::to_value(
            &combos
                .iter()
                .map(|p| p.ema_length.unwrap_or(9))
                .collect::<Vec<_>>(),
        )
        .unwrap(),
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
    Ok(obj.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn on_balance_volume_oscillator_alloc(len: usize) -> *mut f64 {
    let mut v = Vec::<f64>::with_capacity(len);
    let ptr = v.as_mut_ptr();
    std::mem::forget(v);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn on_balance_volume_oscillator_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, 0, len);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn on_balance_volume_oscillator_into(
    source_ptr: *const f64,
    volume_ptr: *const f64,
    line_ptr: *mut f64,
    signal_ptr: *mut f64,
    len: usize,
    obv_length: usize,
    ema_length: usize,
) -> Result<(), JsValue> {
    if source_ptr.is_null() || volume_ptr.is_null() || line_ptr.is_null() || signal_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to on_balance_volume_oscillator_into",
        ));
    }
    unsafe {
        let source = std::slice::from_raw_parts(source_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);
        let line = std::slice::from_raw_parts_mut(line_ptr, len);
        let signal = std::slice::from_raw_parts_mut(signal_ptr, len);
        let input = OnBalanceVolumeOscillatorInput::from_slices(
            source,
            volume,
            OnBalanceVolumeOscillatorParams {
                obv_length: Some(obv_length),
                ema_length: Some(ema_length),
            },
        );
        on_balance_volume_oscillator_into_slice(line, signal, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "on_balance_volume_oscillator_into_host")]
pub fn on_balance_volume_oscillator_into_host(
    source: &[f64],
    volume: &[f64],
    line_ptr: *mut f64,
    signal_ptr: *mut f64,
    obv_length: usize,
    ema_length: usize,
) -> Result<(), JsValue> {
    if line_ptr.is_null() || signal_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to on_balance_volume_oscillator_into_host",
        ));
    }
    unsafe {
        let line = std::slice::from_raw_parts_mut(line_ptr, source.len());
        let signal = std::slice::from_raw_parts_mut(signal_ptr, source.len());
        let input = OnBalanceVolumeOscillatorInput::from_slices(
            source,
            volume,
            OnBalanceVolumeOscillatorParams {
                obv_length: Some(obv_length),
                ema_length: Some(ema_length),
            },
        );
        on_balance_volume_oscillator_into_slice(line, signal, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn on_balance_volume_oscillator_batch_into(
    source_ptr: *const f64,
    volume_ptr: *const f64,
    line_ptr: *mut f64,
    signal_ptr: *mut f64,
    len: usize,
    obv_length_start: usize,
    obv_length_end: usize,
    obv_length_step: usize,
    ema_length_start: usize,
    ema_length_end: usize,
    ema_length_step: usize,
) -> Result<usize, JsValue> {
    if source_ptr.is_null() || volume_ptr.is_null() || line_ptr.is_null() || signal_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to on_balance_volume_oscillator_batch_into",
        ));
    }
    unsafe {
        let source = std::slice::from_raw_parts(source_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);
        let sweep = OnBalanceVolumeOscillatorBatchRange {
            obv_length: (obv_length_start, obv_length_end, obv_length_step),
            ema_length: (ema_length_start, ema_length_end, ema_length_step),
        };
        let combos = expand_grid_on_balance_volume_oscillator(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let total = rows
            .checked_mul(len)
            .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
        let line = std::slice::from_raw_parts_mut(line_ptr, total);
        let signal = std::slice::from_raw_parts_mut(signal_ptr, total);
        on_balance_volume_oscillator_batch_inner_into(
            source,
            volume,
            &sweep,
            Kernel::Scalar,
            false,
            line,
            signal,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn on_balance_volume_oscillator_output_into_js(
    source: &[f64],
    volume: &[f64],
    obv_length: usize,
    ema_length: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = on_balance_volume_oscillator_js(source, volume, obv_length, ema_length)?;
    crate::write_wasm_object_f64_outputs("on_balance_volume_oscillator_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn on_balance_volume_oscillator_batch_output_into_js(
    source: &[f64],
    volume: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = on_balance_volume_oscillator_batch_js(source, volume, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "on_balance_volume_oscillator_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    fn naive_obv_osc(
        source: &[f64],
        volume: &[f64],
        obv_length: usize,
        ema_length: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut line = vec![f64::NAN; source.len()];
        let mut signal = vec![f64::NAN; source.len()];
        let alpha = 2.0 / (ema_length as f64 + 1.0);
        let mut num_buf = vec![0.0; obv_length];
        let mut den_buf = vec![0.0; obv_length];
        let mut count = 0usize;
        let mut head = 0usize;
        let mut num_sum = 0.0;
        let mut den_sum = 0.0;
        let mut prev_source = f64::NAN;
        let mut has_prev = false;
        let mut ema = f64::NAN;
        let mut ema_init = false;

        for i in 0..source.len() {
            let s = source[i];
            let v = volume[i];
            if !is_valid_bar(s, v) {
                count = 0;
                head = 0;
                num_sum = 0.0;
                den_sum = 0.0;
                num_buf.fill(0.0);
                den_buf.fill(0.0);
                prev_source = f64::NAN;
                has_prev = false;
                ema = f64::NAN;
                ema_init = false;
                continue;
            }

            let signed = if has_prev {
                let sign = ((s > prev_source) as i32 - (s < prev_source) as i32) as f64;
                prev_source = s;
                v * sign
            } else {
                prev_source = s;
                has_prev = true;
                0.0
            };

            if count < obv_length {
                num_buf[count] = signed;
                den_buf[count] = v;
                num_sum += signed;
                den_sum += v;
                count += 1;
            } else {
                num_sum += signed - num_buf[head];
                den_sum += v - den_buf[head];
                num_buf[head] = signed;
                den_buf[head] = v;
                head += 1;
                if head == obv_length {
                    head = 0;
                }
            }

            if count == obv_length {
                let curr_line = if den_sum == 0.0 {
                    f64::NAN
                } else {
                    num_sum / den_sum
                };
                line[i] = curr_line;
                signal[i] = if curr_line.is_finite() {
                    if ema_init {
                        ema += alpha * (curr_line - ema);
                    } else {
                        ema = curr_line;
                        ema_init = true;
                    }
                    ema
                } else {
                    f64::NAN
                };
            }
        }

        (line, signal)
    }

    #[test]
    fn on_balance_volume_oscillator_matches_naive_small_sample() {
        let source = [10.0, 11.0, 12.0, 11.0, 10.0, 12.0, 13.0, 13.0];
        let volume = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0];
        let input = OnBalanceVolumeOscillatorInput::from_slices(
            &source,
            &volume,
            OnBalanceVolumeOscillatorParams {
                obv_length: Some(3),
                ema_length: Some(2),
            },
        );

        let out = on_balance_volume_oscillator(&input).expect("obv oscillator output");
        let (expected_line, expected_signal) = naive_obv_osc(&source, &volume, 3, 2);
        for i in 0..source.len() {
            if expected_line[i].is_nan() {
                assert!(out.line[i].is_nan(), "line[{i}] expected NaN");
            } else {
                assert!(
                    (out.line[i] - expected_line[i]).abs() < 1e-12,
                    "line[{i}] actual={} expected={}",
                    out.line[i],
                    expected_line[i]
                );
            }
            if expected_signal[i].is_nan() {
                assert!(out.signal[i].is_nan(), "signal[{i}] expected NaN");
            } else {
                assert!(
                    (out.signal[i] - expected_signal[i]).abs() < 1e-12,
                    "signal[{i}] actual={} expected={}",
                    out.signal[i],
                    expected_signal[i]
                );
            }
        }
    }

    #[test]
    fn on_balance_volume_oscillator_into_matches_api() -> Result<(), Box<dyn Error>> {
        let source = [100.0, 101.0, 102.0, 101.0, 103.0, 104.0, 103.0, 105.0];
        let volume = [1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0];
        let input = OnBalanceVolumeOscillatorInput::from_slices(
            &source,
            &volume,
            OnBalanceVolumeOscillatorParams {
                obv_length: Some(4),
                ema_length: Some(3),
            },
        );
        let direct = on_balance_volume_oscillator(&input)?;
        let mut line = vec![0.0; source.len()];
        let mut signal = vec![0.0; source.len()];
        on_balance_volume_oscillator_into(&input, &mut line, &mut signal)?;
        for i in 0..line.len() {
            if direct.line[i].is_nan() {
                assert!(line[i].is_nan(), "line[{i}] expected NaN");
            } else {
                assert!((line[i] - direct.line[i]).abs() < 1e-12);
            }
            if direct.signal[i].is_nan() {
                assert!(signal[i].is_nan(), "signal[{i}] expected NaN");
            } else {
                assert!((signal[i] - direct.signal[i]).abs() < 1e-12);
            }
        }
        Ok(())
    }

    #[test]
    fn on_balance_volume_oscillator_stream_matches_batch_with_nan_reset() {
        let source = [
            100.0,
            101.0,
            102.0,
            103.0,
            f64::NAN,
            100.0,
            99.0,
            101.0,
            102.0,
            103.0,
        ];
        let volume = [1.0, 2.0, 3.0, 4.0, f64::NAN, 5.0, 6.0, 7.0, 8.0, 9.0];
        let input = OnBalanceVolumeOscillatorInput::from_slices(
            &source,
            &volume,
            OnBalanceVolumeOscillatorParams {
                obv_length: Some(3),
                ema_length: Some(2),
            },
        );
        let batch = on_balance_volume_oscillator(&input).expect("batch output");
        let mut stream =
            OnBalanceVolumeOscillatorStream::try_new(OnBalanceVolumeOscillatorParams {
                obv_length: Some(3),
                ema_length: Some(2),
            })
            .expect("stream");

        let mut line = Vec::with_capacity(source.len());
        let mut signal = Vec::with_capacity(source.len());
        for i in 0..source.len() {
            match stream.update_reset_on_nan(source[i], volume[i]) {
                Some((l, s)) => {
                    line.push(l);
                    signal.push(s);
                }
                None => {
                    line.push(f64::NAN);
                    signal.push(f64::NAN);
                }
            }
        }

        for i in 0..line.len() {
            if batch.line[i].is_nan() {
                assert!(line[i].is_nan(), "line[{i}] expected NaN");
            } else {
                assert!((line[i] - batch.line[i]).abs() < 1e-12);
            }
            if batch.signal[i].is_nan() {
                assert!(signal[i].is_nan(), "signal[{i}] expected NaN");
            } else {
                assert!((signal[i] - batch.signal[i]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn on_balance_volume_oscillator_batch_single_param_matches_single() {
        let source = [100.0, 101.0, 99.0, 102.0, 103.0, 104.0, 102.0, 105.0];
        let volume = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0];
        let single = on_balance_volume_oscillator(&OnBalanceVolumeOscillatorInput::from_slices(
            &source,
            &volume,
            OnBalanceVolumeOscillatorParams {
                obv_length: Some(4),
                ema_length: Some(3),
            },
        ))
        .expect("single");

        let batch = on_balance_volume_oscillator_batch_with_kernel(
            &source,
            &volume,
            &OnBalanceVolumeOscillatorBatchRange {
                obv_length: (4, 4, 0),
                ema_length: (3, 3, 0),
            },
            Kernel::Auto,
        )
        .expect("batch");

        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, source.len());
        for i in 0..source.len() {
            if single.line[i].is_nan() {
                assert!(batch.line[i].is_nan(), "line[{i}] expected NaN");
            } else {
                assert!((batch.line[i] - single.line[i]).abs() < 1e-12);
            }
            if single.signal[i].is_nan() {
                assert!(batch.signal[i].is_nan(), "signal[{i}] expected NaN");
            } else {
                assert!((batch.signal[i] - single.signal[i]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn on_balance_volume_oscillator_rejects_invalid_obv_length() {
        let source = [1.0, 2.0, 3.0];
        let volume = [1.0, 1.0, 1.0];
        let input = OnBalanceVolumeOscillatorInput::from_slices(
            &source,
            &volume,
            OnBalanceVolumeOscillatorParams {
                obv_length: Some(0),
                ema_length: Some(9),
            },
        );
        match on_balance_volume_oscillator(&input) {
            Err(OnBalanceVolumeOscillatorError::InvalidObvLength { obv_length, .. }) => {
                assert_eq!(obv_length, 0);
            }
            other => panic!("unexpected result: {other:?}"),
        }
    }
}
