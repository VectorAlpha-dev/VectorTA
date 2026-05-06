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
    alloc_uninit_f64, detect_best_batch_kernel, init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum ImpulseMacdData<'a> {
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
pub struct ImpulseMacdOutput {
    pub impulse_macd: Vec<f64>,
    pub impulse_histo: Vec<f64>,
    pub signal: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct ImpulseMacdParams {
    pub length_ma: Option<usize>,
    pub length_signal: Option<usize>,
}

impl Default for ImpulseMacdParams {
    fn default() -> Self {
        Self {
            length_ma: Some(34),
            length_signal: Some(9),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ImpulseMacdInput<'a> {
    pub data: ImpulseMacdData<'a>,
    pub params: ImpulseMacdParams,
}

impl<'a> ImpulseMacdInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: ImpulseMacdParams) -> Self {
        Self {
            data: ImpulseMacdData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: ImpulseMacdParams,
    ) -> Self {
        Self {
            data: ImpulseMacdData::Slices { high, low, close },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, ImpulseMacdParams::default())
    }

    #[inline]
    pub fn get_length_ma(&self) -> usize {
        self.params.length_ma.unwrap_or(34)
    }

    #[inline]
    pub fn get_length_signal(&self) -> usize {
        self.params.length_signal.unwrap_or(9)
    }

    #[inline]
    pub fn as_refs(&'a self) -> (&'a [f64], &'a [f64], &'a [f64]) {
        match &self.data {
            ImpulseMacdData::Candles { candles } => (
                candles.high.as_slice(),
                candles.low.as_slice(),
                candles.close.as_slice(),
            ),
            ImpulseMacdData::Slices { high, low, close } => (*high, *low, *close),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ImpulseMacdBuilder {
    length_ma: Option<usize>,
    length_signal: Option<usize>,
    kernel: Kernel,
}

impl Default for ImpulseMacdBuilder {
    fn default() -> Self {
        Self {
            length_ma: None,
            length_signal: None,
            kernel: Kernel::Auto,
        }
    }
}

impl ImpulseMacdBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn length_ma(mut self, value: usize) -> Self {
        self.length_ma = Some(value);
        self
    }

    #[inline(always)]
    pub fn length_signal(mut self, value: usize) -> Self {
        self.length_signal = Some(value);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline(always)]
    pub fn apply(self, candles: &Candles) -> Result<ImpulseMacdOutput, ImpulseMacdError> {
        let input = ImpulseMacdInput::from_candles(
            candles,
            ImpulseMacdParams {
                length_ma: self.length_ma,
                length_signal: self.length_signal,
            },
        );
        impulse_macd_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<ImpulseMacdOutput, ImpulseMacdError> {
        let input = ImpulseMacdInput::from_slices(
            high,
            low,
            close,
            ImpulseMacdParams {
                length_ma: self.length_ma,
                length_signal: self.length_signal,
            },
        );
        impulse_macd_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<ImpulseMacdStream, ImpulseMacdError> {
        ImpulseMacdStream::try_new(ImpulseMacdParams {
            length_ma: self.length_ma,
            length_signal: self.length_signal,
        })
    }
}

#[derive(Debug, Error)]
pub enum ImpulseMacdError {
    #[error("impulse_macd: Empty input data.")]
    EmptyInputData,
    #[error("impulse_macd: Data length mismatch across high, low, and close.")]
    DataLengthMismatch,
    #[error("impulse_macd: All OHLC values are invalid.")]
    AllValuesNaN,
    #[error("impulse_macd: Invalid length_ma: length_ma = {length_ma}, data length = {data_len}")]
    InvalidLengthMa { length_ma: usize, data_len: usize },
    #[error(
        "impulse_macd: Invalid length_signal: length_signal = {length_signal}, data length = {data_len}"
    )]
    InvalidLengthSignal {
        length_signal: usize,
        data_len: usize,
    },
    #[error("impulse_macd: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("impulse_macd: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("impulse_macd: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("impulse_macd: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline(always)]
fn valid_bar(high: f64, low: f64, close: f64) -> bool {
    high.is_finite() && low.is_finite() && close.is_finite() && high >= low
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

#[derive(Clone, Debug)]
struct SmmaState {
    period: usize,
    count: usize,
    sum: f64,
    value: f64,
    ready: bool,
}

impl SmmaState {
    #[inline(always)]
    fn new(period: usize) -> Self {
        Self {
            period,
            count: 0,
            sum: 0.0,
            value: f64::NAN,
            ready: false,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.count = 0;
        self.sum = 0.0;
        self.value = f64::NAN;
        self.ready = false;
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<f64> {
        if self.period == 1 {
            self.value = value;
            self.ready = true;
            return Some(value);
        }
        if !self.ready {
            self.sum += value;
            self.count += 1;
            if self.count == self.period {
                self.value = self.sum / self.period as f64;
                self.ready = true;
                return Some(self.value);
            }
            return None;
        }
        let p = self.period as f64;
        self.value = (self.value * (p - 1.0) + value) / p;
        Some(self.value)
    }
}

#[derive(Clone, Debug)]
struct EmaState {
    alpha: f64,
    value: Option<f64>,
}

impl EmaState {
    #[inline(always)]
    fn new(period: usize) -> Self {
        Self {
            alpha: 2.0 / (period as f64 + 1.0),
            value: None,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.value = None;
    }

    #[inline(always)]
    fn update(&mut self, x: f64) -> f64 {
        let next = match self.value {
            Some(prev) => self.alpha.mul_add(x, (1.0 - self.alpha) * prev),
            None => x,
        };
        self.value = Some(next);
        next
    }
}

#[derive(Clone, Debug)]
struct SmaState {
    period: usize,
    buf: Vec<f64>,
    head: usize,
    len: usize,
    sum: f64,
}

impl SmaState {
    #[inline(always)]
    fn new(period: usize) -> Self {
        Self {
            period,
            buf: vec![0.0; period.max(1)],
            head: 0,
            len: 0,
            sum: 0.0,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.head = 0;
        self.len = 0;
        self.sum = 0.0;
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<f64> {
        if self.period == 1 {
            self.buf[0] = value;
            self.len = 1;
            self.sum = value;
            return Some(value);
        }
        if self.len < self.period {
            self.buf[self.len] = value;
            self.len += 1;
            self.sum += value;
            if self.len == self.period {
                return Some(self.sum / self.period as f64);
            }
            return None;
        }
        let old = self.buf[self.head];
        self.buf[self.head] = value;
        self.head += 1;
        if self.head == self.period {
            self.head = 0;
        }
        self.sum += value - old;
        Some(self.sum / self.period as f64)
    }
}

#[derive(Clone, Debug)]
pub struct ImpulseMacdStream {
    hi_smma: SmmaState,
    lo_smma: SmmaState,
    ema1: EmaState,
    ema2: EmaState,
    signal_sma: SmaState,
}

impl ImpulseMacdStream {
    #[inline(always)]
    fn from_parts(length_ma: usize, length_signal: usize) -> Self {
        Self {
            hi_smma: SmmaState::new(length_ma),
            lo_smma: SmmaState::new(length_ma),
            ema1: EmaState::new(length_ma),
            ema2: EmaState::new(length_ma),
            signal_sma: SmaState::new(length_signal),
        }
    }

    #[inline]
    pub fn try_new(params: ImpulseMacdParams) -> Result<Self, ImpulseMacdError> {
        let length_ma = params.length_ma.unwrap_or(34);
        let length_signal = params.length_signal.unwrap_or(9);
        if length_ma == 0 {
            return Err(ImpulseMacdError::InvalidLengthMa {
                length_ma,
                data_len: 0,
            });
        }
        if length_signal == 0 {
            return Err(ImpulseMacdError::InvalidLengthSignal {
                length_signal,
                data_len: 0,
            });
        }
        Ok(Self::from_parts(length_ma, length_signal))
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.hi_smma.reset();
        self.lo_smma.reset();
        self.ema1.reset();
        self.ema2.reset();
        self.signal_sma.reset();
    }

    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64, f64)> {
        let src = (high + low + close) / 3.0;
        let hi = self.hi_smma.update(high);
        let lo = self.lo_smma.update(low);
        let ema1 = self.ema1.update(src);
        let ema2 = self.ema2.update(ema1);
        let mi = ema1 + (ema1 - ema2);
        let md = match (hi, lo) {
            (Some(hi), Some(lo)) if mi > hi => mi - hi,
            (Some(_), Some(lo)) if mi < lo => mi - lo,
            _ => 0.0,
        };
        let signal = self.signal_sma.update(md);
        let signal_value = signal.unwrap_or(f64::NAN);
        let hist = if signal_value.is_finite() {
            md - signal_value
        } else {
            f64::NAN
        };
        Some((md, hist, signal_value))
    }

    #[inline(always)]
    pub fn update_reset_on_nan(
        &mut self,
        high: f64,
        low: f64,
        close: f64,
    ) -> Option<(f64, f64, f64)> {
        if !valid_bar(high, low, close) {
            self.reset();
            return None;
        }
        self.update(high, low, close)
    }
}

#[inline(always)]
fn impulse_macd_warmup(first: usize) -> usize {
    first
}

#[inline(always)]
fn signal_warmup(first: usize, length_signal: usize) -> usize {
    first + length_signal - 1
}

#[inline(always)]
fn impulse_macd_prepare<'a>(
    input: &'a ImpulseMacdInput,
) -> Result<(&'a [f64], &'a [f64], &'a [f64], usize, usize, usize), ImpulseMacdError> {
    let (high, low, close) = input.as_refs();
    let data_len = close.len();
    if data_len == 0 {
        return Err(ImpulseMacdError::EmptyInputData);
    }
    if high.len() != data_len || low.len() != data_len {
        return Err(ImpulseMacdError::DataLengthMismatch);
    }

    let length_ma = input.get_length_ma();
    if length_ma == 0 || length_ma > data_len {
        return Err(ImpulseMacdError::InvalidLengthMa {
            length_ma,
            data_len,
        });
    }

    let length_signal = input.get_length_signal();
    if length_signal == 0 || length_signal > data_len {
        return Err(ImpulseMacdError::InvalidLengthSignal {
            length_signal,
            data_len,
        });
    }

    let first = first_valid_bar(high, low, close).ok_or(ImpulseMacdError::AllValuesNaN)?;
    let valid = count_valid_from(high, low, close, first);
    let needed = length_ma.max(length_signal);
    if valid < needed {
        return Err(ImpulseMacdError::NotEnoughValidData { needed, valid });
    }

    Ok((high, low, close, length_ma, length_signal, first))
}

#[inline(always)]
fn impulse_macd_compute_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length_ma: usize,
    length_signal: usize,
    _kernel: Kernel,
    out_impulse_macd: &mut [f64],
    out_impulse_histo: &mut [f64],
    out_signal: &mut [f64],
) {
    let mut stream = ImpulseMacdStream::from_parts(length_ma, length_signal);
    for i in 0..close.len() {
        match stream.update_reset_on_nan(high[i], low[i], close[i]) {
            Some((md, hist, signal)) => {
                out_impulse_macd[i] = md;
                out_impulse_histo[i] = hist;
                out_signal[i] = signal;
            }
            None => {
                out_impulse_macd[i] = f64::NAN;
                out_impulse_histo[i] = f64::NAN;
                out_signal[i] = f64::NAN;
            }
        }
    }
}

#[inline(always)]
fn is_default_impulse_macd_params(length_ma: usize, length_signal: usize) -> bool {
    length_ma == 34 && length_signal == 9
}

#[inline]
fn impulse_macd_compute_default_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    out_impulse_macd: &mut [f64],
    out_impulse_histo: &mut [f64],
    out_signal: &mut [f64],
) {
    const LENGTH_MA: usize = 34;
    const LENGTH_SIGNAL: usize = 9;
    const MA_F: f64 = 34.0;
    const MA_M1: f64 = 33.0;
    const EMA_ALPHA: f64 = 2.0 / 35.0;
    const EMA_BETA: f64 = 1.0 - EMA_ALPHA;
    const SIGNAL_RCP: f64 = 1.0 / 9.0;

    let mut hi_count = 0usize;
    let mut hi_sum = 0.0;
    let mut hi_value = f64::NAN;
    let mut hi_ready = false;

    let mut lo_count = 0usize;
    let mut lo_sum = 0.0;
    let mut lo_value = f64::NAN;
    let mut lo_ready = false;

    let mut ema1_value = 0.0;
    let mut ema1_ready = false;
    let mut ema2_value = 0.0;
    let mut ema2_ready = false;

    let mut signal_buf = [0.0; LENGTH_SIGNAL];
    let mut signal_head = 0usize;
    let mut signal_len = 0usize;
    let mut signal_sum = 0.0;

    for i in 0..close.len() {
        let h = high[i];
        let l = low[i];
        let c = close[i];
        if !valid_bar(h, l, c) {
            hi_count = 0;
            hi_sum = 0.0;
            hi_value = f64::NAN;
            hi_ready = false;
            lo_count = 0;
            lo_sum = 0.0;
            lo_value = f64::NAN;
            lo_ready = false;
            ema1_ready = false;
            ema2_ready = false;
            signal_head = 0;
            signal_len = 0;
            signal_sum = 0.0;
            out_impulse_macd[i] = f64::NAN;
            out_impulse_histo[i] = f64::NAN;
            out_signal[i] = f64::NAN;
            continue;
        }

        if !hi_ready {
            hi_sum += h;
            hi_count += 1;
            if hi_count == LENGTH_MA {
                hi_value = hi_sum / MA_F;
                hi_ready = true;
            }
        } else {
            hi_value = (hi_value * MA_M1 + h) / MA_F;
        }

        if !lo_ready {
            lo_sum += l;
            lo_count += 1;
            if lo_count == LENGTH_MA {
                lo_value = lo_sum / MA_F;
                lo_ready = true;
            }
        } else {
            lo_value = (lo_value * MA_M1 + l) / MA_F;
        }

        let src = (h + l + c) / 3.0;
        let ema1_next = if ema1_ready {
            EMA_ALPHA.mul_add(src, EMA_BETA * ema1_value)
        } else {
            ema1_ready = true;
            src
        };
        ema1_value = ema1_next;

        let ema2_next = if ema2_ready {
            EMA_ALPHA.mul_add(ema1_next, EMA_BETA * ema2_value)
        } else {
            ema2_ready = true;
            ema1_next
        };
        ema2_value = ema2_next;

        let mi = ema1_next + (ema1_next - ema2_next);
        let md = if hi_ready && lo_ready {
            if mi > hi_value {
                mi - hi_value
            } else if mi < lo_value {
                mi - lo_value
            } else {
                0.0
            }
        } else {
            0.0
        };
        out_impulse_macd[i] = md;

        if signal_len < LENGTH_SIGNAL {
            signal_buf[signal_len] = md;
            signal_len += 1;
            signal_sum += md;
            if signal_len == LENGTH_SIGNAL {
                let sig = signal_sum * SIGNAL_RCP;
                out_signal[i] = sig;
                out_impulse_histo[i] = md - sig;
            } else {
                out_signal[i] = f64::NAN;
                out_impulse_histo[i] = f64::NAN;
            }
        } else {
            let old = signal_buf[signal_head];
            signal_buf[signal_head] = md;
            signal_head += 1;
            if signal_head == LENGTH_SIGNAL {
                signal_head = 0;
            }
            signal_sum += md - old;
            let sig = signal_sum * SIGNAL_RCP;
            out_signal[i] = sig;
            out_impulse_histo[i] = md - sig;
        }
    }
}

#[inline]
pub fn impulse_macd(input: &ImpulseMacdInput) -> Result<ImpulseMacdOutput, ImpulseMacdError> {
    impulse_macd_with_kernel(input, Kernel::Auto)
}

pub fn impulse_macd_with_kernel(
    input: &ImpulseMacdInput,
    kernel: Kernel,
) -> Result<ImpulseMacdOutput, ImpulseMacdError> {
    let (high, low, close, length_ma, length_signal, first) = impulse_macd_prepare(input)?;
    let mut impulse_macd_values = alloc_uninit_f64(close.len());
    let mut impulse_histo = alloc_uninit_f64(close.len());
    let mut signal = alloc_uninit_f64(close.len());

    if is_default_impulse_macd_params(length_ma, length_signal) {
        impulse_macd_compute_default_into(
            high,
            low,
            close,
            &mut impulse_macd_values,
            &mut impulse_histo,
            &mut signal,
        );
    } else {
        impulse_macd_compute_into(
            high,
            low,
            close,
            length_ma,
            length_signal,
            kernel,
            &mut impulse_macd_values,
            &mut impulse_histo,
            &mut signal,
        );
    }

    Ok(ImpulseMacdOutput {
        impulse_macd: impulse_macd_values,
        impulse_histo,
        signal,
    })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn impulse_macd_into(
    input: &ImpulseMacdInput,
    out_impulse_macd: &mut [f64],
    out_impulse_histo: &mut [f64],
    out_signal: &mut [f64],
) -> Result<(), ImpulseMacdError> {
    impulse_macd_into_slice(
        out_impulse_macd,
        out_impulse_histo,
        out_signal,
        input,
        Kernel::Auto,
    )
}

pub fn impulse_macd_into_slice(
    out_impulse_macd: &mut [f64],
    out_impulse_histo: &mut [f64],
    out_signal: &mut [f64],
    input: &ImpulseMacdInput,
    kernel: Kernel,
) -> Result<(), ImpulseMacdError> {
    let (high, low, close, length_ma, length_signal, _first) = impulse_macd_prepare(input)?;
    if out_impulse_macd.len() != close.len()
        || out_impulse_histo.len() != close.len()
        || out_signal.len() != close.len()
    {
        return Err(ImpulseMacdError::OutputLengthMismatch {
            expected: close.len(),
            got: out_impulse_macd
                .len()
                .max(out_impulse_histo.len())
                .max(out_signal.len()),
        });
    }

    impulse_macd_compute_into(
        high,
        low,
        close,
        length_ma,
        length_signal,
        kernel,
        out_impulse_macd,
        out_impulse_histo,
        out_signal,
    );
    Ok(())
}

#[derive(Clone, Debug)]
pub struct ImpulseMacdBatchRange {
    pub length_ma: (usize, usize, usize),
    pub length_signal: (usize, usize, usize),
}

impl Default for ImpulseMacdBatchRange {
    fn default() -> Self {
        Self {
            length_ma: (34, 34, 0),
            length_signal: (9, 9, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ImpulseMacdBatchBuilder {
    range: ImpulseMacdBatchRange,
    kernel: Kernel,
}

impl ImpulseMacdBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline]
    pub fn length_ma_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.length_ma = (start, end, step);
        self
    }

    #[inline]
    pub fn length_signal_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.length_signal = (start, end, step);
        self
    }

    #[inline]
    pub fn length_ma_static(mut self, value: usize) -> Self {
        self.range.length_ma = (value, value, 0);
        self
    }

    #[inline]
    pub fn length_signal_static(mut self, value: usize) -> Self {
        self.range.length_signal = (value, value, 0);
        self
    }

    #[inline]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<ImpulseMacdBatchOutput, ImpulseMacdError> {
        impulse_macd_batch_with_kernel(high, low, close, &self.range, self.kernel)
    }

    #[inline]
    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<ImpulseMacdBatchOutput, ImpulseMacdError> {
        self.apply_slices(
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
        )
    }
}

#[derive(Clone, Debug)]
pub struct ImpulseMacdBatchOutput {
    pub impulse_macd: Vec<f64>,
    pub impulse_histo: Vec<f64>,
    pub signal: Vec<f64>,
    pub combos: Vec<ImpulseMacdParams>,
    pub rows: usize,
    pub cols: usize,
}

impl ImpulseMacdBatchOutput {
    pub fn row_for_params(&self, params: &ImpulseMacdParams) -> Option<usize> {
        let target_length_ma = params.length_ma.unwrap_or(34);
        let target_length_signal = params.length_signal.unwrap_or(9);
        self.combos.iter().position(|combo| {
            combo.length_ma.unwrap_or(34) == target_length_ma
                && combo.length_signal.unwrap_or(9) == target_length_signal
        })
    }
}

fn axis_usize(range: (usize, usize, usize)) -> Result<Vec<usize>, ImpulseMacdError> {
    let (start, end, step) = range;
    if start == 0 || end == 0 {
        return Err(ImpulseMacdError::InvalidRange { start, end, step });
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
        return Err(ImpulseMacdError::InvalidRange { start, end, step });
    }
    Ok(out)
}

pub fn expand_grid_impulse_macd(
    sweep: &ImpulseMacdBatchRange,
) -> Result<Vec<ImpulseMacdParams>, ImpulseMacdError> {
    let length_mas = axis_usize(sweep.length_ma)?;
    let length_signals = axis_usize(sweep.length_signal)?;
    let mut out = Vec::with_capacity(length_mas.len() * length_signals.len());
    for length_ma in length_mas {
        for &length_signal in &length_signals {
            out.push(ImpulseMacdParams {
                length_ma: Some(length_ma),
                length_signal: Some(length_signal),
            });
        }
    }
    Ok(out)
}

pub fn impulse_macd_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &ImpulseMacdBatchRange,
    kernel: Kernel,
) -> Result<ImpulseMacdBatchOutput, ImpulseMacdError> {
    let batch_kernel = match kernel {
        Kernel::Auto => Kernel::ScalarBatch,
        other if other.is_batch() => other,
        other => return Err(ImpulseMacdError::InvalidKernelForBatch(other)),
    };
    impulse_macd_batch_impl(high, low, close, sweep, batch_kernel.to_non_batch(), true)
}

pub fn impulse_macd_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &ImpulseMacdBatchRange,
) -> Result<ImpulseMacdBatchOutput, ImpulseMacdError> {
    impulse_macd_batch_impl(high, low, close, sweep, Kernel::Scalar, false)
}

pub fn impulse_macd_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &ImpulseMacdBatchRange,
) -> Result<ImpulseMacdBatchOutput, ImpulseMacdError> {
    impulse_macd_batch_impl(high, low, close, sweep, Kernel::Scalar, true)
}

fn impulse_macd_batch_impl(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &ImpulseMacdBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<ImpulseMacdBatchOutput, ImpulseMacdError> {
    let combos = expand_grid_impulse_macd(sweep)?;
    let rows = combos.len();
    let cols = close.len();
    if cols == 0 {
        return Err(ImpulseMacdError::EmptyInputData);
    }
    if high.len() != cols || low.len() != cols {
        return Err(ImpulseMacdError::DataLengthMismatch);
    }

    for params in &combos {
        let input = ImpulseMacdInput::from_slices(high, low, close, params.clone());
        impulse_macd_prepare(&input)?;
    }

    let first = first_valid_bar(high, low, close).unwrap_or(cols);
    let md_warmups: Vec<usize> = combos
        .iter()
        .map(|_| impulse_macd_warmup(first).min(cols))
        .collect();
    let signal_warmups: Vec<usize> = combos
        .iter()
        .map(|params| signal_warmup(first, params.length_signal.unwrap_or(9)).min(cols))
        .collect();

    let mut md_matrix = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut md_matrix, cols, &md_warmups);
    let mut hist_matrix = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut hist_matrix, cols, &signal_warmups);
    let mut signal_matrix = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut signal_matrix, cols, &signal_warmups);

    let mut md_guard = ManuallyDrop::new(md_matrix);
    let mut hist_guard = ManuallyDrop::new(hist_matrix);
    let mut signal_guard = ManuallyDrop::new(signal_matrix);
    let md_mu: &mut [MaybeUninit<f64>] =
        unsafe { std::slice::from_raw_parts_mut(md_guard.as_mut_ptr(), md_guard.len()) };
    let hist_mu: &mut [MaybeUninit<f64>] =
        unsafe { std::slice::from_raw_parts_mut(hist_guard.as_mut_ptr(), hist_guard.len()) };
    let signal_mu: &mut [MaybeUninit<f64>] =
        unsafe { std::slice::from_raw_parts_mut(signal_guard.as_mut_ptr(), signal_guard.len()) };

    let do_row = |row: usize,
                  row_md_mu: &mut [MaybeUninit<f64>],
                  row_hist_mu: &mut [MaybeUninit<f64>],
                  row_signal_mu: &mut [MaybeUninit<f64>]| {
        let params = &combos[row];
        let dst_md = unsafe {
            std::slice::from_raw_parts_mut(row_md_mu.as_mut_ptr() as *mut f64, row_md_mu.len())
        };
        let dst_hist = unsafe {
            std::slice::from_raw_parts_mut(row_hist_mu.as_mut_ptr() as *mut f64, row_hist_mu.len())
        };
        let dst_signal = unsafe {
            std::slice::from_raw_parts_mut(
                row_signal_mu.as_mut_ptr() as *mut f64,
                row_signal_mu.len(),
            )
        };
        impulse_macd_compute_into(
            high,
            low,
            close,
            params.length_ma.unwrap_or(34),
            params.length_signal.unwrap_or(9),
            kernel,
            dst_md,
            dst_hist,
            dst_signal,
        );
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        md_mu
            .par_chunks_mut(cols)
            .zip(hist_mu.par_chunks_mut(cols))
            .zip(signal_mu.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, ((row_md, row_hist), row_signal))| {
                do_row(row, row_md, row_hist, row_signal)
            });
        #[cfg(target_arch = "wasm32")]
        for (row, ((row_md, row_hist), row_signal)) in md_mu
            .chunks_mut(cols)
            .zip(hist_mu.chunks_mut(cols))
            .zip(signal_mu.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, row_md, row_hist, row_signal);
        }
    } else {
        for (row, ((row_md, row_hist), row_signal)) in md_mu
            .chunks_mut(cols)
            .zip(hist_mu.chunks_mut(cols))
            .zip(signal_mu.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, row_md, row_hist, row_signal);
        }
    }

    let impulse_macd = unsafe {
        Vec::from_raw_parts(
            md_guard.as_mut_ptr() as *mut f64,
            md_guard.len(),
            md_guard.capacity(),
        )
    };
    let impulse_histo = unsafe {
        Vec::from_raw_parts(
            hist_guard.as_mut_ptr() as *mut f64,
            hist_guard.len(),
            hist_guard.capacity(),
        )
    };
    let signal = unsafe {
        Vec::from_raw_parts(
            signal_guard.as_mut_ptr() as *mut f64,
            signal_guard.len(),
            signal_guard.capacity(),
        )
    };

    Ok(ImpulseMacdBatchOutput {
        impulse_macd,
        impulse_histo,
        signal,
        combos,
        rows,
        cols,
    })
}

fn impulse_macd_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &ImpulseMacdBatchRange,
    kernel: Kernel,
    parallel: bool,
    out_impulse_macd: &mut [f64],
    out_impulse_histo: &mut [f64],
    out_signal: &mut [f64],
) -> Result<(), ImpulseMacdError> {
    let combos = expand_grid_impulse_macd(sweep)?;
    let rows = combos.len();
    let cols = close.len();
    if cols == 0 {
        return Err(ImpulseMacdError::EmptyInputData);
    }
    if high.len() != cols || low.len() != cols {
        return Err(ImpulseMacdError::DataLengthMismatch);
    }
    for params in &combos {
        let input = ImpulseMacdInput::from_slices(high, low, close, params.clone());
        impulse_macd_prepare(&input)?;
    }
    let expected = rows * cols;
    if out_impulse_macd.len() != expected
        || out_impulse_histo.len() != expected
        || out_signal.len() != expected
    {
        return Err(ImpulseMacdError::OutputLengthMismatch {
            expected,
            got: out_impulse_macd
                .len()
                .max(out_impulse_histo.len())
                .max(out_signal.len()),
        });
    }

    let do_row = |row: usize, dst_md: &mut [f64], dst_hist: &mut [f64], dst_signal: &mut [f64]| {
        let params = &combos[row];
        impulse_macd_compute_into(
            high,
            low,
            close,
            params.length_ma.unwrap_or(34),
            params.length_signal.unwrap_or(9),
            kernel,
            dst_md,
            dst_hist,
            dst_signal,
        );
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out_impulse_macd
            .par_chunks_mut(cols)
            .zip(out_impulse_histo.par_chunks_mut(cols))
            .zip(out_signal.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, ((dst_md, dst_hist), dst_signal))| {
                do_row(row, dst_md, dst_hist, dst_signal)
            });
        #[cfg(target_arch = "wasm32")]
        for (row, ((dst_md, dst_hist), dst_signal)) in out_impulse_macd
            .chunks_mut(cols)
            .zip(out_impulse_histo.chunks_mut(cols))
            .zip(out_signal.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, dst_md, dst_hist, dst_signal);
        }
    } else {
        for (row, ((dst_md, dst_hist), dst_signal)) in out_impulse_macd
            .chunks_mut(cols)
            .zip(out_impulse_histo.chunks_mut(cols))
            .zip(out_signal.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, dst_md, dst_hist, dst_signal);
        }
    }
    Ok(())
}

#[cfg(feature = "python")]
#[pyfunction(name = "impulse_macd")]
#[pyo3(signature = (high, low, close, length_ma=34, length_signal=9, kernel=None))]
pub fn impulse_macd_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    length_ma: usize,
    length_signal: usize,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let input = ImpulseMacdInput::from_slices(
        high,
        low,
        close,
        ImpulseMacdParams {
            length_ma: Some(length_ma),
            length_signal: Some(length_signal),
        },
    );
    let kernel = validate_kernel(kernel, false)?;
    let out = py
        .allow_threads(|| impulse_macd_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        out.impulse_macd.into_pyarray(py),
        out.impulse_histo.into_pyarray(py),
        out.signal.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyclass(name = "ImpulseMacdStream")]
pub struct ImpulseMacdStreamPy {
    stream: ImpulseMacdStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl ImpulseMacdStreamPy {
    #[new]
    #[pyo3(signature = (length_ma=34, length_signal=9))]
    fn new(length_ma: usize, length_signal: usize) -> PyResult<Self> {
        let stream = ImpulseMacdStream::try_new(ImpulseMacdParams {
            length_ma: Some(length_ma),
            length_signal: Some(length_signal),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64, f64)> {
        self.stream.update_reset_on_nan(high, low, close)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "impulse_macd_batch")]
#[pyo3(signature = (high, low, close, length_ma_range, length_signal_range, kernel=None))]
pub fn impulse_macd_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    length_ma_range: (usize, usize, usize),
    length_signal_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let sweep = ImpulseMacdBatchRange {
        length_ma: length_ma_range,
        length_signal: length_signal_range,
    };
    let combos =
        expand_grid_impulse_macd(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;
    let arr_md = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let arr_hist = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let arr_signal = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_md = unsafe { arr_md.as_slice_mut()? };
    let out_hist = unsafe { arr_hist.as_slice_mut()? };
    let out_signal = unsafe { arr_signal.as_slice_mut()? };
    let kernel = validate_kernel(kernel, true)?;

    py.allow_threads(|| {
        let batch_kernel = match kernel {
            Kernel::Auto => detect_best_batch_kernel(),
            other => other,
        };
        impulse_macd_batch_inner_into(
            high,
            low,
            close,
            &sweep,
            batch_kernel.to_non_batch(),
            true,
            out_md,
            out_hist,
            out_signal,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("impulse_macd", arr_md.reshape((rows, cols))?)?;
    dict.set_item("impulse_histo", arr_hist.reshape((rows, cols))?)?;
    dict.set_item("signal", arr_signal.reshape((rows, cols))?)?;
    dict.set_item(
        "length_mas",
        combos
            .iter()
            .map(|params| params.length_ma.unwrap_or(34) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "length_signals",
        combos
            .iter()
            .map(|params| params.length_signal.unwrap_or(9) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_impulse_macd_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(impulse_macd_py, m)?)?;
    m.add_function(wrap_pyfunction!(impulse_macd_batch_py, m)?)?;
    m.add_class::<ImpulseMacdStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ImpulseMacdJsOutput {
    impulse_macd: Vec<f64>,
    impulse_histo: Vec<f64>,
    signal: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ImpulseMacdBatchConfig {
    length_ma_range: Vec<usize>,
    length_signal_range: Vec<usize>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ImpulseMacdBatchJsOutput {
    impulse_macd: Vec<f64>,
    impulse_histo: Vec<f64>,
    signal: Vec<f64>,
    rows: usize,
    cols: usize,
    combos: Vec<ImpulseMacdParams>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "impulse_macd_js")]
pub fn impulse_macd_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length_ma: usize,
    length_signal: usize,
) -> Result<JsValue, JsValue> {
    let input = ImpulseMacdInput::from_slices(
        high,
        low,
        close,
        ImpulseMacdParams {
            length_ma: Some(length_ma),
            length_signal: Some(length_signal),
        },
    );
    let out = impulse_macd_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&ImpulseMacdJsOutput {
        impulse_macd: out.impulse_macd,
        impulse_histo: out.impulse_histo,
        signal: out.signal,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "impulse_macd_batch_js")]
pub fn impulse_macd_batch_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: ImpulseMacdBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.length_ma_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: length_ma_range must have exactly 3 elements [start, end, step]",
        ));
    }
    if config.length_signal_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: length_signal_range must have exactly 3 elements [start, end, step]",
        ));
    }
    let sweep = ImpulseMacdBatchRange {
        length_ma: (
            config.length_ma_range[0],
            config.length_ma_range[1],
            config.length_ma_range[2],
        ),
        length_signal: (
            config.length_signal_range[0],
            config.length_signal_range[1],
            config.length_signal_range[2],
        ),
    };
    let batch = impulse_macd_batch_slice(high, low, close, &sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&ImpulseMacdBatchJsOutput {
        impulse_macd: batch.impulse_macd,
        impulse_histo: batch.impulse_histo,
        signal: batch.signal,
        rows: batch.rows,
        cols: batch.cols,
        combos: batch.combos,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn impulse_macd_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len * 3);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn impulse_macd_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, 0, len * 3);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn impulse_macd_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length_ma: usize,
    length_signal: usize,
) -> Result<(), JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to impulse_macd_into",
        ));
    }
    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len * 3);
        let (out_md, rest) = out.split_at_mut(len);
        let (out_hist, out_signal) = rest.split_at_mut(len);
        let input = ImpulseMacdInput::from_slices(
            high,
            low,
            close,
            ImpulseMacdParams {
                length_ma: Some(length_ma),
                length_signal: Some(length_signal),
            },
        );
        impulse_macd_into_slice(out_md, out_hist, out_signal, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "impulse_macd_into_host")]
pub fn impulse_macd_into_host(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    out_ptr: *mut f64,
    length_ma: usize,
    length_signal: usize,
) -> Result<(), JsValue> {
    if out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to impulse_macd_into_host",
        ));
    }
    unsafe {
        let out = std::slice::from_raw_parts_mut(out_ptr, close.len() * 3);
        let (out_md, rest) = out.split_at_mut(close.len());
        let (out_hist, out_signal) = rest.split_at_mut(close.len());
        let input = ImpulseMacdInput::from_slices(
            high,
            low,
            close,
            ImpulseMacdParams {
                length_ma: Some(length_ma),
                length_signal: Some(length_signal),
            },
        );
        impulse_macd_into_slice(out_md, out_hist, out_signal, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn impulse_macd_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length_ma_start: usize,
    length_ma_end: usize,
    length_ma_step: usize,
    length_signal_start: usize,
    length_signal_end: usize,
    length_signal_step: usize,
) -> Result<usize, JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to impulse_macd_batch_into",
        ));
    }
    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let sweep = ImpulseMacdBatchRange {
            length_ma: (length_ma_start, length_ma_end, length_ma_step),
            length_signal: (length_signal_start, length_signal_end, length_signal_step),
        };
        let combos =
            expand_grid_impulse_macd(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let out = std::slice::from_raw_parts_mut(out_ptr, rows * len * 3);
        let (out_md, rest) = out.split_at_mut(rows * len);
        let (out_hist, out_signal) = rest.split_at_mut(rows * len);
        impulse_macd_batch_inner_into(
            high,
            low,
            close,
            &sweep,
            Kernel::Scalar,
            false,
            out_md,
            out_hist,
            out_signal,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn impulse_macd_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length_ma: usize,
    length_signal: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = impulse_macd_js(high, low, close, length_ma, length_signal)?;
    crate::write_wasm_object_f64_outputs("impulse_macd_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn impulse_macd_batch_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = impulse_macd_batch_js(high, low, close, config)?;
    crate::write_wasm_selected_object_f64_outputs("impulse_macd_batch_output_into_js", &value, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::dispatch::{
        compute_cpu_batch, IndicatorBatchRequest, IndicatorDataRef, IndicatorParamSet, ParamKV,
        ParamValue,
    };

    fn sample_ohlc(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut high = Vec::with_capacity(len);
        let mut low = Vec::with_capacity(len);
        let mut close = Vec::with_capacity(len);
        for i in 0..len {
            let base = 100.0 + i as f64 * 0.06 + (i as f64 * 0.17).sin() * 2.4;
            let cl = base + (i as f64 * 0.11).cos() * 0.9;
            let hi = cl + 1.3 + (i as f64 * 0.07).sin().abs();
            let lo = cl - 1.1 - (i as f64 * 0.05).cos().abs();
            high.push(hi);
            low.push(lo);
            close.push(cl);
        }
        (high, low, close)
    }

    fn assert_close_nan(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for i in 0..actual.len() {
            let a = actual[i];
            let e = expected[i];
            if a.is_nan() || e.is_nan() {
                assert!(
                    a.is_nan() && e.is_nan(),
                    "nan mismatch at {i}: got {a}, expected {e}"
                );
            } else {
                assert!(
                    (a - e).abs() <= 1e-10,
                    "mismatch at {i}: got {a}, expected {e}"
                );
            }
        }
    }

    fn naive_expected(
        high: &[f64],
        low: &[f64],
        close: &[f64],
        length_ma: usize,
        length_signal: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut md = vec![f64::NAN; close.len()];
        let mut hist = vec![f64::NAN; close.len()];
        let mut signal = vec![f64::NAN; close.len()];

        let mut hi_smma = SmmaState::new(length_ma);
        let mut lo_smma = SmmaState::new(length_ma);
        let mut ema1 = EmaState::new(length_ma);
        let mut ema2 = EmaState::new(length_ma);
        let mut signal_sma = SmaState::new(length_signal);

        for i in 0..close.len() {
            if !valid_bar(high[i], low[i], close[i]) {
                hi_smma.reset();
                lo_smma.reset();
                ema1.reset();
                ema2.reset();
                signal_sma.reset();
                continue;
            }
            let src = (high[i] + low[i] + close[i]) / 3.0;
            let hi = hi_smma.update(high[i]);
            let lo = lo_smma.update(low[i]);
            let e1 = ema1.update(src);
            let e2 = ema2.update(e1);
            let mi = e1 + (e1 - e2);
            let now_md = match (hi, lo) {
                (Some(hi), Some(lo)) if mi > hi => mi - hi,
                (Some(_), Some(lo)) if mi < lo => mi - lo,
                _ => 0.0,
            };
            md[i] = now_md;
            if let Some(sig) = signal_sma.update(now_md) {
                signal[i] = sig;
                hist[i] = now_md - sig;
            }
        }

        (md, hist, signal)
    }

    #[test]
    fn impulse_macd_matches_naive() {
        let (high, low, close) = sample_ohlc(256);
        let input = ImpulseMacdInput::from_slices(
            &high,
            &low,
            &close,
            ImpulseMacdParams {
                length_ma: Some(34),
                length_signal: Some(9),
            },
        );
        let out = impulse_macd(&input).expect("indicator");
        let (expected_md, expected_hist, expected_signal) =
            naive_expected(&high, &low, &close, 34, 9);
        assert_close_nan(&out.impulse_macd, &expected_md);
        assert_close_nan(&out.impulse_histo, &expected_hist);
        assert_close_nan(&out.signal, &expected_signal);
    }

    #[test]
    fn impulse_macd_into_matches_api() {
        let (high, low, close) = sample_ohlc(192);
        let input = ImpulseMacdInput::from_slices(
            &high,
            &low,
            &close,
            ImpulseMacdParams {
                length_ma: Some(21),
                length_signal: Some(7),
            },
        );
        let out = impulse_macd(&input).expect("baseline");
        let mut md = vec![0.0; close.len()];
        let mut hist = vec![0.0; close.len()];
        let mut signal = vec![0.0; close.len()];
        impulse_macd_into(&input, &mut md, &mut hist, &mut signal).expect("into");
        assert_close_nan(&md, &out.impulse_macd);
        assert_close_nan(&hist, &out.impulse_histo);
        assert_close_nan(&signal, &out.signal);
    }

    #[test]
    fn impulse_macd_stream_matches_batch() {
        let (high, low, close) = sample_ohlc(192);
        let input = ImpulseMacdInput::from_slices(
            &high,
            &low,
            &close,
            ImpulseMacdParams {
                length_ma: Some(34),
                length_signal: Some(9),
            },
        );
        let batch = impulse_macd(&input).expect("batch");
        let mut stream = ImpulseMacdStream::try_new(ImpulseMacdParams {
            length_ma: Some(34),
            length_signal: Some(9),
        })
        .expect("stream");
        let mut md = Vec::with_capacity(close.len());
        let mut hist = Vec::with_capacity(close.len());
        let mut signal = Vec::with_capacity(close.len());
        for i in 0..close.len() {
            match stream.update_reset_on_nan(high[i], low[i], close[i]) {
                Some((a, b, c)) => {
                    md.push(a);
                    hist.push(b);
                    signal.push(c);
                }
                None => {
                    md.push(f64::NAN);
                    hist.push(f64::NAN);
                    signal.push(f64::NAN);
                }
            }
        }
        assert_close_nan(&md, &batch.impulse_macd);
        assert_close_nan(&hist, &batch.impulse_histo);
        assert_close_nan(&signal, &batch.signal);
    }

    #[test]
    fn impulse_macd_batch_single_param_matches_single() {
        let (high, low, close) = sample_ohlc(160);
        let sweep = ImpulseMacdBatchRange {
            length_ma: (34, 34, 0),
            length_signal: (9, 9, 0),
        };
        let batch =
            impulse_macd_batch_with_kernel(&high, &low, &close, &sweep, Kernel::ScalarBatch)
                .expect("batch");
        let input = ImpulseMacdInput::from_slices(
            &high,
            &low,
            &close,
            ImpulseMacdParams {
                length_ma: Some(34),
                length_signal: Some(9),
            },
        );
        let out = impulse_macd(&input).expect("single");
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, close.len());
        assert_close_nan(&batch.impulse_macd[..close.len()], &out.impulse_macd);
        assert_close_nan(&batch.impulse_histo[..close.len()], &out.impulse_histo);
        assert_close_nan(&batch.signal[..close.len()], &out.signal);
    }

    #[test]
    fn impulse_macd_rejects_invalid_length_signal() {
        let (high, low, close) = sample_ohlc(32);
        let input = ImpulseMacdInput::from_slices(
            &high,
            &low,
            &close,
            ImpulseMacdParams {
                length_ma: Some(20),
                length_signal: Some(0),
            },
        );
        let err = impulse_macd(&input).expect_err("invalid");
        assert!(matches!(err, ImpulseMacdError::InvalidLengthSignal { .. }));
    }

    #[test]
    fn impulse_macd_dispatch_matches_direct() {
        let (high, low, close) = sample_ohlc(180);
        let params = [
            ParamKV {
                key: "length_ma",
                value: ParamValue::Int(34),
            },
            ParamKV {
                key: "length_signal",
                value: ParamValue::Int(9),
            },
        ];
        let combos = [IndicatorParamSet { params: &params }];
        let out = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "impulse_macd",
            output_id: Some("impulse_macd"),
            data: IndicatorDataRef::Ohlc {
                open: &close,
                high: &high,
                low: &low,
                close: &close,
            },
            combos: &combos,
            kernel: Kernel::ScalarBatch,
        })
        .expect("dispatch");
        let direct = impulse_macd(&ImpulseMacdInput::from_slices(
            &high,
            &low,
            &close,
            ImpulseMacdParams {
                length_ma: Some(34),
                length_signal: Some(9),
            },
        ))
        .expect("direct");
        assert_eq!(out.rows, 1);
        assert_eq!(out.cols, close.len());
        assert_close_nan(
            out.values_f64.as_ref().expect("values"),
            &direct.impulse_macd,
        );
    }
}
