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
use crate::utilities::helpers::{alloc_with_nan_prefix, make_uninit_matrix};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;

use std::collections::VecDeque;
use std::mem::{ManuallyDrop, MaybeUninit};

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use thiserror::Error;

const DEFAULT_EMO_DIVISOR: usize = 10_000;
const DEFAULT_EMO_LENGTH: usize = 14;
const DEFAULT_FAST_LENGTH: usize = 12;
const DEFAULT_SLOW_LENGTH: usize = 26;
const DEFAULT_MFI_LENGTH: usize = 20;
const DEFAULT_BB_LENGTH: usize = 20;
const DEFAULT_BB_MULTIPLIER: f64 = 2.0;
const DEFAULT_CCI_LENGTH: usize = 14;
const DEFAULT_DPO_LENGTH: usize = 18;
const DEFAULT_ROC_LENGTH: usize = 10;
const DEFAULT_RSI_LENGTH: usize = 14;
const DEFAULT_STOCH_LENGTH: usize = 14;
const DEFAULT_STOCH_D_LENGTH: usize = 3;
const DEFAULT_STOCH_K_LENGTH: usize = 1;
const DEFAULT_SMA_LENGTH: usize = 10;
const DPO_DELAY: usize = 10;

#[derive(Debug, Clone)]
pub enum InsyncIndexData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct InsyncIndexOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct InsyncIndexParams {
    pub emo_divisor: Option<usize>,
    pub emo_length: Option<usize>,
    pub fast_length: Option<usize>,
    pub slow_length: Option<usize>,
    pub mfi_length: Option<usize>,
    pub bb_length: Option<usize>,
    pub bb_multiplier: Option<f64>,
    pub cci_length: Option<usize>,
    pub dpo_length: Option<usize>,
    pub roc_length: Option<usize>,
    pub rsi_length: Option<usize>,
    pub stoch_length: Option<usize>,
    pub stoch_d_length: Option<usize>,
    pub stoch_k_length: Option<usize>,
    pub sma_length: Option<usize>,
}

impl Default for InsyncIndexParams {
    fn default() -> Self {
        Self {
            emo_divisor: Some(DEFAULT_EMO_DIVISOR),
            emo_length: Some(DEFAULT_EMO_LENGTH),
            fast_length: Some(DEFAULT_FAST_LENGTH),
            slow_length: Some(DEFAULT_SLOW_LENGTH),
            mfi_length: Some(DEFAULT_MFI_LENGTH),
            bb_length: Some(DEFAULT_BB_LENGTH),
            bb_multiplier: Some(DEFAULT_BB_MULTIPLIER),
            cci_length: Some(DEFAULT_CCI_LENGTH),
            dpo_length: Some(DEFAULT_DPO_LENGTH),
            roc_length: Some(DEFAULT_ROC_LENGTH),
            rsi_length: Some(DEFAULT_RSI_LENGTH),
            stoch_length: Some(DEFAULT_STOCH_LENGTH),
            stoch_d_length: Some(DEFAULT_STOCH_D_LENGTH),
            stoch_k_length: Some(DEFAULT_STOCH_K_LENGTH),
            sma_length: Some(DEFAULT_SMA_LENGTH),
        }
    }
}

#[derive(Debug, Clone)]
pub struct InsyncIndexInput<'a> {
    pub data: InsyncIndexData<'a>,
    pub params: InsyncIndexParams,
}

impl<'a> InsyncIndexInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: InsyncIndexParams) -> Self {
        Self {
            data: InsyncIndexData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
        params: InsyncIndexParams,
    ) -> Self {
        Self {
            data: InsyncIndexData::Slices {
                high,
                low,
                close,
                volume,
            },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, InsyncIndexParams::default())
    }

    #[inline]
    pub fn as_refs(&'a self) -> (&'a [f64], &'a [f64], &'a [f64], &'a [f64]) {
        match &self.data {
            InsyncIndexData::Candles { candles } => (
                candles.high.as_slice(),
                candles.low.as_slice(),
                candles.close.as_slice(),
                candles.volume.as_slice(),
            ),
            InsyncIndexData::Slices {
                high,
                low,
                close,
                volume,
            } => (*high, *low, *close, *volume),
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct ResolvedParams {
    emo_divisor: usize,
    emo_length: usize,
    fast_length: usize,
    slow_length: usize,
    mfi_length: usize,
    bb_length: usize,
    bb_multiplier: f64,
    cci_length: usize,
    dpo_length: usize,
    roc_length: usize,
    rsi_length: usize,
    stoch_length: usize,
    stoch_d_length: usize,
    stoch_k_length: usize,
    sma_length: usize,
}

#[derive(Clone, Debug)]
pub struct InsyncIndexBuilder {
    params: InsyncIndexParams,
    kernel: Kernel,
}

impl Default for InsyncIndexBuilder {
    fn default() -> Self {
        Self {
            params: InsyncIndexParams::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl InsyncIndexBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn emo_divisor(mut self, value: usize) -> Self {
        self.params.emo_divisor = Some(value);
        self
    }

    #[inline(always)]
    pub fn emo_length(mut self, value: usize) -> Self {
        self.params.emo_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn fast_length(mut self, value: usize) -> Self {
        self.params.fast_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn slow_length(mut self, value: usize) -> Self {
        self.params.slow_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn mfi_length(mut self, value: usize) -> Self {
        self.params.mfi_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn bb_length(mut self, value: usize) -> Self {
        self.params.bb_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn bb_multiplier(mut self, value: f64) -> Self {
        self.params.bb_multiplier = Some(value);
        self
    }

    #[inline(always)]
    pub fn cci_length(mut self, value: usize) -> Self {
        self.params.cci_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn dpo_length(mut self, value: usize) -> Self {
        self.params.dpo_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn roc_length(mut self, value: usize) -> Self {
        self.params.roc_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn rsi_length(mut self, value: usize) -> Self {
        self.params.rsi_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn stoch_length(mut self, value: usize) -> Self {
        self.params.stoch_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn stoch_d_length(mut self, value: usize) -> Self {
        self.params.stoch_d_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn stoch_k_length(mut self, value: usize) -> Self {
        self.params.stoch_k_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn sma_length(mut self, value: usize) -> Self {
        self.params.sma_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline(always)]
    pub fn apply(self, candles: &Candles) -> Result<InsyncIndexOutput, InsyncIndexError> {
        let input = InsyncIndexInput::from_candles(candles, self.params);
        insync_index_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Result<InsyncIndexOutput, InsyncIndexError> {
        let input = InsyncIndexInput::from_slices(high, low, close, volume, self.params);
        insync_index_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<InsyncIndexStream, InsyncIndexError> {
        InsyncIndexStream::try_new(self.params)
    }
}

#[derive(Debug, Error)]
pub enum InsyncIndexError {
    #[error("insync_index: Empty input data.")]
    EmptyInputData,
    #[error("insync_index: Data length mismatch across high, low, close, and volume.")]
    DataLengthMismatch,
    #[error("insync_index: All OHLCV values are invalid.")]
    AllValuesNaN,
    #[error("insync_index: Invalid parameter {name}={value}.")]
    InvalidPeriod { name: &'static str, value: usize },
    #[error("insync_index: Invalid parameter {name}={value}.")]
    InvalidFloat { name: &'static str, value: f64 },
    #[error("insync_index: Output length mismatch: expected = {expected}, got = {got}.")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("insync_index: Invalid range for {name}: start={start}, end={end}, step={step}.")]
    InvalidRange {
        name: &'static str,
        start: String,
        end: String,
        step: String,
    },
    #[error("insync_index: Invalid kernel for batch: {0:?}.")]
    InvalidKernelForBatch(Kernel),
}

#[inline(always)]
fn resolve_params(params: &InsyncIndexParams) -> Result<ResolvedParams, InsyncIndexError> {
    let resolved = ResolvedParams {
        emo_divisor: params.emo_divisor.unwrap_or(DEFAULT_EMO_DIVISOR),
        emo_length: params.emo_length.unwrap_or(DEFAULT_EMO_LENGTH),
        fast_length: params.fast_length.unwrap_or(DEFAULT_FAST_LENGTH),
        slow_length: params.slow_length.unwrap_or(DEFAULT_SLOW_LENGTH),
        mfi_length: params.mfi_length.unwrap_or(DEFAULT_MFI_LENGTH),
        bb_length: params.bb_length.unwrap_or(DEFAULT_BB_LENGTH),
        bb_multiplier: params.bb_multiplier.unwrap_or(DEFAULT_BB_MULTIPLIER),
        cci_length: params.cci_length.unwrap_or(DEFAULT_CCI_LENGTH),
        dpo_length: params.dpo_length.unwrap_or(DEFAULT_DPO_LENGTH),
        roc_length: params.roc_length.unwrap_or(DEFAULT_ROC_LENGTH),
        rsi_length: params.rsi_length.unwrap_or(DEFAULT_RSI_LENGTH),
        stoch_length: params.stoch_length.unwrap_or(DEFAULT_STOCH_LENGTH),
        stoch_d_length: params.stoch_d_length.unwrap_or(DEFAULT_STOCH_D_LENGTH),
        stoch_k_length: params.stoch_k_length.unwrap_or(DEFAULT_STOCH_K_LENGTH),
        sma_length: params.sma_length.unwrap_or(DEFAULT_SMA_LENGTH),
    };

    let periods = [
        ("emo_divisor", resolved.emo_divisor),
        ("emo_length", resolved.emo_length),
        ("fast_length", resolved.fast_length),
        ("slow_length", resolved.slow_length),
        ("mfi_length", resolved.mfi_length),
        ("bb_length", resolved.bb_length),
        ("cci_length", resolved.cci_length),
        ("dpo_length", resolved.dpo_length),
        ("roc_length", resolved.roc_length),
        ("rsi_length", resolved.rsi_length),
        ("stoch_length", resolved.stoch_length),
        ("stoch_d_length", resolved.stoch_d_length),
        ("stoch_k_length", resolved.stoch_k_length),
        ("sma_length", resolved.sma_length),
    ];
    for (name, value) in periods {
        if value == 0 {
            return Err(InsyncIndexError::InvalidPeriod { name, value });
        }
    }

    if !resolved.bb_multiplier.is_finite() || resolved.bb_multiplier <= 0.0 {
        return Err(InsyncIndexError::InvalidFloat {
            name: "bb_multiplier",
            value: resolved.bb_multiplier,
        });
    }

    Ok(resolved)
}

#[inline(always)]
fn normalize_kernel(_kernel: Kernel) -> Kernel {
    Kernel::Scalar
}

#[inline(always)]
fn valid_bar(high: f64, low: f64, close: f64, volume: f64) -> bool {
    high.is_finite()
        && low.is_finite()
        && close.is_finite()
        && volume.is_finite()
        && volume > 0.0
        && high >= low
}

#[inline(always)]
fn has_any_valid_bar(high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> bool {
    (0..close.len()).any(|i| valid_bar(high[i], low[i], close[i], volume[i]))
}

#[inline(always)]
fn prepare_input<'a>(
    input: &'a InsyncIndexInput<'a>,
) -> Result<(&'a [f64], &'a [f64], &'a [f64], &'a [f64], ResolvedParams), InsyncIndexError> {
    let (high, low, close, volume) = input.as_refs();
    let len = close.len();
    if len == 0 {
        return Err(InsyncIndexError::EmptyInputData);
    }
    if high.len() != len || low.len() != len || volume.len() != len {
        return Err(InsyncIndexError::DataLengthMismatch);
    }
    if !has_any_valid_bar(high, low, close, volume) {
        return Err(InsyncIndexError::AllValuesNaN);
    }
    Ok((high, low, close, volume, resolve_params(&input.params)?))
}

#[derive(Clone, Debug)]
struct RollingSmaState {
    period: usize,
    buf: Vec<f64>,
    head: usize,
    len: usize,
    sum: f64,
}

impl RollingSmaState {
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
        self.buf.fill(0.0);
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
            self.sum += value;
            self.len += 1;
            if self.len == self.period {
                return Some(self.sum / self.period as f64);
            }
            return None;
        }
        let old = self.buf[self.head];
        self.buf[self.head] = value;
        self.sum += value - old;
        self.head += 1;
        if self.head == self.period {
            self.head = 0;
        }
        Some(self.sum / self.period as f64)
    }
}

#[derive(Clone, Debug)]
struct RollingVarianceState {
    period: usize,
    buf: Vec<f64>,
    head: usize,
    len: usize,
    sum: f64,
    sumsq: f64,
}

impl RollingVarianceState {
    #[inline(always)]
    fn new(period: usize) -> Self {
        Self {
            period,
            buf: vec![0.0; period.max(1)],
            head: 0,
            len: 0,
            sum: 0.0,
            sumsq: 0.0,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.buf.fill(0.0);
        self.head = 0;
        self.len = 0;
        self.sum = 0.0;
        self.sumsq = 0.0;
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        if self.len < self.period {
            self.buf[self.len] = value;
            self.sum += value;
            self.sumsq += value * value;
            self.len += 1;
            if self.len < self.period {
                return None;
            }
        } else {
            let old = self.buf[self.head];
            self.buf[self.head] = value;
            self.sum += value - old;
            self.sumsq += value.mul_add(value, -(old * old));
            self.head += 1;
            if self.head == self.period {
                self.head = 0;
            }
        }
        let mean = self.sum / self.period as f64;
        let variance = (self.sumsq / self.period as f64 - mean * mean).max(0.0);
        Some((mean, variance.sqrt()))
    }
}

#[derive(Clone, Debug)]
struct RollingCciState {
    period: usize,
    buf: Vec<f64>,
    head: usize,
    len: usize,
    sum: f64,
}

impl RollingCciState {
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
        self.buf.fill(0.0);
        self.head = 0;
        self.len = 0;
        self.sum = 0.0;
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<f64> {
        if self.len < self.period {
            self.buf[self.len] = value;
            self.sum += value;
            self.len += 1;
            if self.len < self.period {
                return None;
            }
        } else {
            let old = self.buf[self.head];
            self.buf[self.head] = value;
            self.sum += value - old;
            self.head += 1;
            if self.head == self.period {
                self.head = 0;
            }
        }
        let mean = self.sum / self.period as f64;
        let mad = self
            .buf
            .iter()
            .take(self.period)
            .map(|x| (*x - mean).abs())
            .sum::<f64>()
            / self.period as f64;
        if mad == 0.0 || !mad.is_finite() {
            return None;
        }
        Some((value - mean) / (0.015 * mad))
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
    fn update(&mut self, value: f64) -> f64 {
        let next = match self.value {
            Some(prev) => self.alpha.mul_add(value, (1.0 - self.alpha) * prev),
            None => value,
        };
        self.value = Some(next);
        next
    }
}

#[derive(Clone, Debug)]
struct WilderRsiState {
    period: usize,
    prev: Option<f64>,
    gains: f64,
    losses: f64,
    count: usize,
    avg_gain: Option<f64>,
    avg_loss: Option<f64>,
}

impl WilderRsiState {
    #[inline(always)]
    fn new(period: usize) -> Self {
        Self {
            period,
            prev: None,
            gains: 0.0,
            losses: 0.0,
            count: 0,
            avg_gain: None,
            avg_loss: None,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.prev = None;
        self.gains = 0.0;
        self.losses = 0.0;
        self.count = 0;
        self.avg_gain = None;
        self.avg_loss = None;
    }

    #[inline(always)]
    fn rsi_from_avgs(avg_gain: f64, avg_loss: f64) -> f64 {
        if avg_gain == 0.0 && avg_loss == 0.0 {
            50.0
        } else if avg_loss == 0.0 {
            100.0
        } else if avg_gain == 0.0 {
            0.0
        } else {
            let rs = avg_gain / avg_loss;
            100.0 - 100.0 / (1.0 + rs)
        }
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<f64> {
        let prev = match self.prev {
            Some(prev) => prev,
            None => {
                self.prev = Some(value);
                return None;
            }
        };
        let change = value - prev;
        let gain = change.max(0.0);
        let loss = (-change).max(0.0);
        self.prev = Some(value);

        if self.avg_gain.is_none() || self.avg_loss.is_none() {
            self.gains += gain;
            self.losses += loss;
            self.count += 1;
            if self.count < self.period {
                return None;
            }
            let avg_gain = self.gains / self.period as f64;
            let avg_loss = self.losses / self.period as f64;
            self.avg_gain = Some(avg_gain);
            self.avg_loss = Some(avg_loss);
            return Some(Self::rsi_from_avgs(avg_gain, avg_loss));
        }

        let period_f = self.period as f64;
        let avg_gain = ((self.avg_gain.unwrap_or(0.0) * (period_f - 1.0)) + gain) / period_f;
        let avg_loss = ((self.avg_loss.unwrap_or(0.0) * (period_f - 1.0)) + loss) / period_f;
        self.avg_gain = Some(avg_gain);
        self.avg_loss = Some(avg_loss);
        Some(Self::rsi_from_avgs(avg_gain, avg_loss))
    }
}

#[derive(Clone, Debug)]
struct RocState {
    period: usize,
    buf: Vec<f64>,
    head: usize,
    len: usize,
}

impl RocState {
    #[inline(always)]
    fn new(period: usize) -> Self {
        Self {
            period,
            buf: vec![0.0; period.max(1)],
            head: 0,
            len: 0,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.buf.fill(0.0);
        self.head = 0;
        self.len = 0;
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<f64> {
        if self.len < self.period {
            self.buf[self.len] = value;
            self.len += 1;
            return None;
        }
        let prev = self.buf[self.head];
        self.buf[self.head] = value;
        self.head += 1;
        if self.head == self.period {
            self.head = 0;
        }
        if prev == 0.0 || !prev.is_finite() {
            return None;
        }
        Some(100.0 * (value - prev) / prev)
    }
}

#[derive(Clone, Debug)]
struct DpoState {
    close_sma: RollingSmaState,
    dpo_sma: RollingSmaState,
    barsback: usize,
    sma_history: VecDeque<Option<f64>>,
    delayed_components: VecDeque<i32>,
}

impl DpoState {
    #[inline(always)]
    fn new(period: usize, sma_length: usize) -> Self {
        Self {
            close_sma: RollingSmaState::new(period),
            dpo_sma: RollingSmaState::new(sma_length),
            barsback: period / 2 + 1,
            sma_history: VecDeque::with_capacity(period / 2 + 3),
            delayed_components: VecDeque::with_capacity(DPO_DELAY + 2),
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.close_sma.reset();
        self.dpo_sma.reset();
        self.sma_history.clear();
        self.delayed_components.clear();
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> i32 {
        let sma_now = self.close_sma.update(value);
        self.sma_history.push_back(sma_now);

        let mut component = 0;
        if self.sma_history.len() > self.barsback {
            let past_sma = self.sma_history.pop_front().unwrap_or(None);
            if let Some(past_sma) = past_sma {
                let dpo = value - past_sma;
                if let Some(avg) = self.dpo_sma.update(dpo) {
                    let diff = dpo - avg;
                    if diff < 0.0 && avg < 0.0 {
                        component = -5;
                    } else if diff > 0.0 && avg > 0.0 {
                        component = 5;
                    }
                }
            }
        }

        self.delayed_components.push_back(component);
        if self.delayed_components.len() <= DPO_DELAY {
            0
        } else {
            self.delayed_components.pop_front().unwrap_or(0)
        }
    }
}

#[derive(Clone, Debug)]
struct MfiState {
    period: usize,
    prev_tp: Option<f64>,
    pos_buf: Vec<f64>,
    neg_buf: Vec<f64>,
    head: usize,
    len: usize,
    pos_sum: f64,
    neg_sum: f64,
}

impl MfiState {
    #[inline(always)]
    fn new(period: usize) -> Self {
        Self {
            period,
            prev_tp: None,
            pos_buf: vec![0.0; period.max(1)],
            neg_buf: vec![0.0; period.max(1)],
            head: 0,
            len: 0,
            pos_sum: 0.0,
            neg_sum: 0.0,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.prev_tp = None;
        self.pos_buf.fill(0.0);
        self.neg_buf.fill(0.0);
        self.head = 0;
        self.len = 0;
        self.pos_sum = 0.0;
        self.neg_sum = 0.0;
    }

    #[inline(always)]
    fn update(&mut self, tp: f64, volume: f64) -> Option<f64> {
        let prev_tp = match self.prev_tp {
            Some(prev) => prev,
            None => {
                self.prev_tp = Some(tp);
                return None;
            }
        };
        let mut pos = 0.0;
        let mut neg = 0.0;
        if tp > prev_tp {
            pos = volume * tp;
        } else if tp < prev_tp {
            neg = volume * tp;
        }
        self.prev_tp = Some(tp);

        if self.len < self.period {
            self.pos_buf[self.len] = pos;
            self.neg_buf[self.len] = neg;
            self.pos_sum += pos;
            self.neg_sum += neg;
            self.len += 1;
            if self.len < self.period {
                return None;
            }
        } else {
            let old_pos = self.pos_buf[self.head];
            let old_neg = self.neg_buf[self.head];
            self.pos_buf[self.head] = pos;
            self.neg_buf[self.head] = neg;
            self.pos_sum += pos - old_pos;
            self.neg_sum += neg - old_neg;
            self.head += 1;
            if self.head == self.period {
                self.head = 0;
            }
        }

        if self.pos_sum == 0.0 && self.neg_sum == 0.0 {
            return Some(50.0);
        }
        if self.neg_sum == 0.0 {
            return Some(100.0);
        }
        if self.pos_sum == 0.0 {
            return Some(0.0);
        }
        let rs = self.pos_sum / self.neg_sum;
        Some(100.0 - 100.0 / (1.0 + rs))
    }
}

#[derive(Clone, Debug)]
struct StochState {
    length: usize,
    index: usize,
    highs: VecDeque<(usize, f64)>,
    lows: VecDeque<(usize, f64)>,
    k_sma: RollingSmaState,
    d_sma: RollingSmaState,
}

impl StochState {
    #[inline(always)]
    fn new(length: usize, smooth_d: usize, smooth_k: usize) -> Self {
        Self {
            length,
            index: 0,
            highs: VecDeque::with_capacity(length.max(1)),
            lows: VecDeque::with_capacity(length.max(1)),
            k_sma: RollingSmaState::new(smooth_k),
            d_sma: RollingSmaState::new(smooth_d),
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.index = 0;
        self.highs.clear();
        self.lows.clear();
        self.k_sma.reset();
        self.d_sma.reset();
    }

    #[inline(always)]
    fn update(&mut self, high: f64, low: f64, close: f64) -> (Option<f64>, Option<f64>) {
        let idx = self.index;
        self.index += 1;

        while let Some(&(_, value)) = self.highs.back() {
            if value <= high {
                self.highs.pop_back();
            } else {
                break;
            }
        }
        self.highs.push_back((idx, high));

        while let Some(&(_, value)) = self.lows.back() {
            if value >= low {
                self.lows.pop_back();
            } else {
                break;
            }
        }
        self.lows.push_back((idx, low));

        let expire_before = idx.saturating_add(1).saturating_sub(self.length);
        while let Some(&(window_idx, _)) = self.highs.front() {
            if window_idx < expire_before {
                self.highs.pop_front();
            } else {
                break;
            }
        }
        while let Some(&(window_idx, _)) = self.lows.front() {
            if window_idx < expire_before {
                self.lows.pop_front();
            } else {
                break;
            }
        }

        if idx + 1 < self.length {
            return (None, None);
        }

        let highest = self.highs.front().map(|&(_, value)| value).unwrap_or(high);
        let lowest = self.lows.front().map(|&(_, value)| value).unwrap_or(low);
        let denom = highest - lowest;
        if denom <= 0.0 || !denom.is_finite() {
            return (None, None);
        }
        let fast = 100.0 * (close - lowest) / denom;
        let k = self.k_sma.update(fast);
        let d = match k {
            Some(k_value) => self.d_sma.update(k_value),
            None => None,
        };
        (k, d)
    }
}

#[derive(Clone, Debug)]
struct EmoSignalState {
    divisor: f64,
    prev_hl2: Option<f64>,
    emo_sma: RollingSmaState,
    emo_avg_sma: RollingSmaState,
}

impl EmoSignalState {
    #[inline(always)]
    fn new(divisor: usize, emo_length: usize, sma_length: usize) -> Self {
        Self {
            divisor: divisor as f64,
            prev_hl2: None,
            emo_sma: RollingSmaState::new(emo_length),
            emo_avg_sma: RollingSmaState::new(sma_length),
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.prev_hl2 = None;
        self.emo_sma.reset();
        self.emo_avg_sma.reset();
    }

    #[inline(always)]
    fn update(&mut self, high: f64, low: f64, volume: f64) -> i32 {
        let hl2 = 0.5 * (high + low);
        let prev_hl2 = match self.prev_hl2 {
            Some(prev) => prev,
            None => {
                self.prev_hl2 = Some(hl2);
                return 0;
            }
        };
        self.prev_hl2 = Some(hl2);

        let raw = self.divisor * (hl2 - prev_hl2) * (high - low) / volume;
        let emo = match self.emo_sma.update(raw) {
            Some(value) => value,
            None => return 0,
        };
        let emo_avg = match self.emo_avg_sma.update(emo) {
            Some(value) => value,
            None => return 0,
        };
        let diff = emo - emo_avg;
        if diff < 0.0 && emo_avg < 0.0 {
            -5
        } else if diff > 0.0 && emo_avg > 0.0 {
            5
        } else {
            0
        }
    }
}

#[derive(Clone, Debug)]
struct MacdSignalState {
    fast: EmaState,
    slow: EmaState,
    trend_sma: RollingSmaState,
}

impl MacdSignalState {
    #[inline(always)]
    fn new(fast_length: usize, slow_length: usize, sma_length: usize) -> Self {
        Self {
            fast: EmaState::new(fast_length),
            slow: EmaState::new(slow_length),
            trend_sma: RollingSmaState::new(sma_length),
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.fast.reset();
        self.slow.reset();
        self.trend_sma.reset();
    }

    #[inline(always)]
    fn update(&mut self, close: f64) -> i32 {
        let macd = self.fast.update(close) - self.slow.update(close);
        let macd_avg = match self.trend_sma.update(macd) {
            Some(value) => value,
            None => return 0,
        };
        let diff = macd - macd_avg;
        if diff < 0.0 && macd_avg < 0.0 {
            -5
        } else if diff > 0.0 && macd_avg > 0.0 {
            5
        } else {
            0
        }
    }
}

#[derive(Clone, Debug)]
struct RocSignalState {
    roc: RocState,
    roc_sma: RollingSmaState,
}

impl RocSignalState {
    #[inline(always)]
    fn new(roc_length: usize, sma_length: usize) -> Self {
        Self {
            roc: RocState::new(roc_length),
            roc_sma: RollingSmaState::new(sma_length),
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.roc.reset();
        self.roc_sma.reset();
    }

    #[inline(always)]
    fn update(&mut self, close: f64) -> i32 {
        let roc = match self.roc.update(close) {
            Some(value) => value,
            None => return 0,
        };
        let roc_avg = match self.roc_sma.update(roc) {
            Some(value) => value,
            None => return 0,
        };
        let diff = roc - roc_avg;
        if diff < 0.0 && roc_avg < 0.0 {
            -5
        } else if diff > 0.0 && roc_avg > 0.0 {
            5
        } else {
            0
        }
    }
}

#[derive(Clone, Debug)]
pub struct InsyncIndexStream {
    params: ResolvedParams,
    bb: RollingVarianceState,
    cci: RollingCciState,
    emo: EmoSignalState,
    macd: MacdSignalState,
    mfi: MfiState,
    dpo: DpoState,
    roc: RocSignalState,
    rsi: WilderRsiState,
    stoch: StochState,
}

impl InsyncIndexStream {
    #[inline(always)]
    fn from_resolved(params: ResolvedParams) -> Self {
        Self {
            bb: RollingVarianceState::new(params.bb_length),
            cci: RollingCciState::new(params.cci_length),
            emo: EmoSignalState::new(params.emo_divisor, params.emo_length, params.sma_length),
            macd: MacdSignalState::new(params.fast_length, params.slow_length, params.sma_length),
            mfi: MfiState::new(params.mfi_length),
            dpo: DpoState::new(params.dpo_length, params.sma_length),
            roc: RocSignalState::new(params.roc_length, params.sma_length),
            rsi: WilderRsiState::new(params.rsi_length),
            stoch: StochState::new(
                params.stoch_length,
                params.stoch_d_length,
                params.stoch_k_length,
            ),
            params,
        }
    }

    #[inline]
    pub fn try_new(params: InsyncIndexParams) -> Result<Self, InsyncIndexError> {
        Ok(Self::from_resolved(resolve_params(&params)?))
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.bb.reset();
        self.cci.reset();
        self.emo.reset();
        self.macd.reset();
        self.mfi.reset();
        self.dpo.reset();
        self.roc.reset();
        self.rsi.reset();
        self.stoch.reset();
    }

    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> Option<f64> {
        self.update_reset_on_nan(high, low, close, volume)
    }

    #[inline(always)]
    pub fn update_reset_on_nan(
        &mut self,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Option<f64> {
        if !valid_bar(high, low, close, volume) {
            self.reset();
            return None;
        }

        let mut score = 50.0;

        if let Some((mean, stddev)) = self.bb.update(close) {
            let lower = mean - self.params.bb_multiplier * stddev;
            let upper = mean + self.params.bb_multiplier * stddev;
            let denom = upper - lower;
            if denom > 0.0 {
                let position = (close - lower) / denom;
                if position < 0.05 {
                    score -= 5.0;
                } else if position > 0.95 {
                    score += 5.0;
                }
            }
        }

        if let Some(cci) = self.cci.update(close) {
            if cci > 100.0 {
                score += 5.0;
            } else if cci < -100.0 {
                score -= 5.0;
            }
        }

        score += self.emo.update(high, low, volume) as f64;
        score += self.macd.update(close) as f64;

        let typical = (high + low + close) / 3.0;
        if let Some(mfi) = self.mfi.update(typical, volume) {
            if mfi > 80.0 {
                score += 5.0;
            } else if mfi < 20.0 {
                score -= 5.0;
            }
        }

        score += self.dpo.update(close) as f64;
        score += self.roc.update(close) as f64;

        if let Some(rsi) = self.rsi.update(close) {
            if rsi > 70.0 {
                score += 5.0;
            } else if rsi < 30.0 {
                score -= 5.0;
            }
        }

        let (k, d) = self.stoch.update(high, low, close);
        if let Some(k) = k {
            if k > 80.0 {
                score += 5.0;
            } else if k < 20.0 {
                score -= 5.0;
            }
        }
        if let Some(d) = d {
            if d > 80.0 {
                score += 5.0;
            } else if d < 20.0 {
                score -= 5.0;
            }
        }

        Some(score)
    }
}

#[inline(always)]
fn insync_index_compute_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    params: ResolvedParams,
    _kernel: Kernel,
    out: &mut [f64],
) {
    let mut stream = InsyncIndexStream::from_resolved(params);
    for i in 0..close.len() {
        out[i] = stream
            .update_reset_on_nan(high[i], low[i], close[i], volume[i])
            .unwrap_or(f64::NAN);
    }
}

#[inline]
pub fn insync_index(input: &InsyncIndexInput) -> Result<InsyncIndexOutput, InsyncIndexError> {
    insync_index_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn insync_index_with_kernel(
    input: &InsyncIndexInput,
    kernel: Kernel,
) -> Result<InsyncIndexOutput, InsyncIndexError> {
    let (high, low, close, volume, params) = prepare_input(input)?;
    let mut values = alloc_with_nan_prefix(close.len(), close.len());
    insync_index_compute_into(
        high,
        low,
        close,
        volume,
        params,
        normalize_kernel(kernel),
        &mut values,
    );
    Ok(InsyncIndexOutput { values })
}

#[inline]
pub fn insync_index_into_slice(
    out: &mut [f64],
    input: &InsyncIndexInput,
    kernel: Kernel,
) -> Result<(), InsyncIndexError> {
    let (high, low, close, volume, params) = prepare_input(input)?;
    if out.len() != close.len() {
        return Err(InsyncIndexError::OutputLengthMismatch {
            expected: close.len(),
            got: out.len(),
        });
    }
    insync_index_compute_into(
        high,
        low,
        close,
        volume,
        params,
        normalize_kernel(kernel),
        out,
    );
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn insync_index_into(
    input: &InsyncIndexInput,
    out: &mut [f64],
) -> Result<(), InsyncIndexError> {
    insync_index_into_slice(out, input, Kernel::Auto)
}

#[derive(Clone, Debug)]
pub struct InsyncIndexBatchRange {
    pub emo_divisor: (usize, usize, usize),
    pub emo_length: (usize, usize, usize),
    pub fast_length: (usize, usize, usize),
    pub slow_length: (usize, usize, usize),
    pub mfi_length: (usize, usize, usize),
    pub bb_length: (usize, usize, usize),
    pub bb_multiplier: (f64, f64, f64),
    pub cci_length: (usize, usize, usize),
    pub dpo_length: (usize, usize, usize),
    pub roc_length: (usize, usize, usize),
    pub rsi_length: (usize, usize, usize),
    pub stoch_length: (usize, usize, usize),
    pub stoch_d_length: (usize, usize, usize),
    pub stoch_k_length: (usize, usize, usize),
    pub sma_length: (usize, usize, usize),
}

impl Default for InsyncIndexBatchRange {
    fn default() -> Self {
        Self {
            emo_divisor: (DEFAULT_EMO_DIVISOR, DEFAULT_EMO_DIVISOR, 0),
            emo_length: (DEFAULT_EMO_LENGTH, DEFAULT_EMO_LENGTH, 0),
            fast_length: (DEFAULT_FAST_LENGTH, DEFAULT_FAST_LENGTH, 0),
            slow_length: (DEFAULT_SLOW_LENGTH, DEFAULT_SLOW_LENGTH, 0),
            mfi_length: (DEFAULT_MFI_LENGTH, DEFAULT_MFI_LENGTH, 0),
            bb_length: (DEFAULT_BB_LENGTH, DEFAULT_BB_LENGTH, 0),
            bb_multiplier: (DEFAULT_BB_MULTIPLIER, DEFAULT_BB_MULTIPLIER, 0.0),
            cci_length: (DEFAULT_CCI_LENGTH, DEFAULT_CCI_LENGTH, 0),
            dpo_length: (DEFAULT_DPO_LENGTH, DEFAULT_DPO_LENGTH, 0),
            roc_length: (DEFAULT_ROC_LENGTH, DEFAULT_ROC_LENGTH, 0),
            rsi_length: (DEFAULT_RSI_LENGTH, DEFAULT_RSI_LENGTH, 0),
            stoch_length: (DEFAULT_STOCH_LENGTH, DEFAULT_STOCH_LENGTH, 0),
            stoch_d_length: (DEFAULT_STOCH_D_LENGTH, DEFAULT_STOCH_D_LENGTH, 0),
            stoch_k_length: (DEFAULT_STOCH_K_LENGTH, DEFAULT_STOCH_K_LENGTH, 0),
            sma_length: (DEFAULT_SMA_LENGTH, DEFAULT_SMA_LENGTH, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct InsyncIndexBatchBuilder {
    range: InsyncIndexBatchRange,
    kernel: Kernel,
}

impl InsyncIndexBatchBuilder {
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
    pub fn emo_divisor_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.emo_divisor = (start, end, step);
        self
    }

    #[inline]
    pub fn emo_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.emo_length = (start, end, step);
        self
    }

    #[inline]
    pub fn fast_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.fast_length = (start, end, step);
        self
    }

    #[inline]
    pub fn slow_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.slow_length = (start, end, step);
        self
    }

    #[inline]
    pub fn mfi_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.mfi_length = (start, end, step);
        self
    }

    #[inline]
    pub fn bb_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.bb_length = (start, end, step);
        self
    }

    #[inline]
    pub fn bb_multiplier_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.bb_multiplier = (start, end, step);
        self
    }

    #[inline]
    pub fn cci_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.cci_length = (start, end, step);
        self
    }

    #[inline]
    pub fn dpo_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.dpo_length = (start, end, step);
        self
    }

    #[inline]
    pub fn roc_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.roc_length = (start, end, step);
        self
    }

    #[inline]
    pub fn rsi_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.rsi_length = (start, end, step);
        self
    }

    #[inline]
    pub fn stoch_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.stoch_length = (start, end, step);
        self
    }

    #[inline]
    pub fn stoch_d_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.stoch_d_length = (start, end, step);
        self
    }

    #[inline]
    pub fn stoch_k_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.stoch_k_length = (start, end, step);
        self
    }

    #[inline]
    pub fn sma_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.sma_length = (start, end, step);
        self
    }

    #[inline]
    pub fn emo_divisor_static(mut self, value: usize) -> Self {
        self.range.emo_divisor = (value, value, 0);
        self
    }

    #[inline]
    pub fn emo_length_static(mut self, value: usize) -> Self {
        self.range.emo_length = (value, value, 0);
        self
    }

    #[inline]
    pub fn fast_length_static(mut self, value: usize) -> Self {
        self.range.fast_length = (value, value, 0);
        self
    }

    #[inline]
    pub fn slow_length_static(mut self, value: usize) -> Self {
        self.range.slow_length = (value, value, 0);
        self
    }

    #[inline]
    pub fn mfi_length_static(mut self, value: usize) -> Self {
        self.range.mfi_length = (value, value, 0);
        self
    }

    #[inline]
    pub fn bb_length_static(mut self, value: usize) -> Self {
        self.range.bb_length = (value, value, 0);
        self
    }

    #[inline]
    pub fn bb_multiplier_static(mut self, value: f64) -> Self {
        self.range.bb_multiplier = (value, value, 0.0);
        self
    }

    #[inline]
    pub fn cci_length_static(mut self, value: usize) -> Self {
        self.range.cci_length = (value, value, 0);
        self
    }

    #[inline]
    pub fn dpo_length_static(mut self, value: usize) -> Self {
        self.range.dpo_length = (value, value, 0);
        self
    }

    #[inline]
    pub fn roc_length_static(mut self, value: usize) -> Self {
        self.range.roc_length = (value, value, 0);
        self
    }

    #[inline]
    pub fn rsi_length_static(mut self, value: usize) -> Self {
        self.range.rsi_length = (value, value, 0);
        self
    }

    #[inline]
    pub fn stoch_length_static(mut self, value: usize) -> Self {
        self.range.stoch_length = (value, value, 0);
        self
    }

    #[inline]
    pub fn stoch_d_length_static(mut self, value: usize) -> Self {
        self.range.stoch_d_length = (value, value, 0);
        self
    }

    #[inline]
    pub fn stoch_k_length_static(mut self, value: usize) -> Self {
        self.range.stoch_k_length = (value, value, 0);
        self
    }

    #[inline]
    pub fn sma_length_static(mut self, value: usize) -> Self {
        self.range.sma_length = (value, value, 0);
        self
    }

    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Result<InsyncIndexBatchOutput, InsyncIndexError> {
        insync_index_batch_with_kernel(high, low, close, volume, &self.range, self.kernel)
    }

    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<InsyncIndexBatchOutput, InsyncIndexError> {
        self.apply_slices(
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
            candles.volume.as_slice(),
        )
    }
}

#[derive(Clone, Debug)]
pub struct InsyncIndexBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<InsyncIndexParams>,
    pub rows: usize,
    pub cols: usize,
}

impl InsyncIndexBatchOutput {
    pub fn row_for_params(&self, params: &InsyncIndexParams) -> Option<usize> {
        let target = resolve_params(params).ok()?;
        self.combos.iter().position(|combo| {
            resolve_params(combo)
                .map(|resolved| {
                    resolved.emo_divisor == target.emo_divisor
                        && resolved.emo_length == target.emo_length
                        && resolved.fast_length == target.fast_length
                        && resolved.slow_length == target.slow_length
                        && resolved.mfi_length == target.mfi_length
                        && resolved.bb_length == target.bb_length
                        && (resolved.bb_multiplier - target.bb_multiplier).abs() <= 1e-12
                        && resolved.cci_length == target.cci_length
                        && resolved.dpo_length == target.dpo_length
                        && resolved.roc_length == target.roc_length
                        && resolved.rsi_length == target.rsi_length
                        && resolved.stoch_length == target.stoch_length
                        && resolved.stoch_d_length == target.stoch_d_length
                        && resolved.stoch_k_length == target.stoch_k_length
                        && resolved.sma_length == target.sma_length
                })
                .unwrap_or(false)
        })
    }
}

fn axis_usize(
    name: &'static str,
    range: (usize, usize, usize),
) -> Result<Vec<usize>, InsyncIndexError> {
    let (start, end, step) = range;
    if start == 0 || end == 0 {
        return Err(InsyncIndexError::InvalidRange {
            name,
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
        return Err(InsyncIndexError::InvalidRange {
            name,
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

fn axis_f64(name: &'static str, range: (f64, f64, f64)) -> Result<Vec<f64>, InsyncIndexError> {
    let (start, end, step) = range;
    if !start.is_finite() || !end.is_finite() || start <= 0.0 || end <= 0.0 {
        return Err(InsyncIndexError::InvalidRange {
            name,
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if step == 0.0 || (start - end).abs() <= f64::EPSILON {
        return Ok(vec![start]);
    }
    if !step.is_finite() || step < 0.0 {
        return Err(InsyncIndexError::InvalidRange {
            name,
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    let mut out = Vec::new();
    if start < end {
        let mut value = start;
        while value <= end + 1e-12 {
            out.push(value);
            value += step;
            if step <= 0.0 {
                break;
            }
        }
    } else {
        let mut value = start;
        while value >= end - 1e-12 {
            out.push(value);
            value -= step;
            if step <= 0.0 {
                break;
            }
        }
    }
    if out.is_empty() {
        return Err(InsyncIndexError::InvalidRange {
            name,
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

pub fn expand_grid_insync_index(
    sweep: &InsyncIndexBatchRange,
) -> Result<Vec<InsyncIndexParams>, InsyncIndexError> {
    let emo_divisors = axis_usize("emo_divisor", sweep.emo_divisor)?;
    let emo_lengths = axis_usize("emo_length", sweep.emo_length)?;
    let fast_lengths = axis_usize("fast_length", sweep.fast_length)?;
    let slow_lengths = axis_usize("slow_length", sweep.slow_length)?;
    let mfi_lengths = axis_usize("mfi_length", sweep.mfi_length)?;
    let bb_lengths = axis_usize("bb_length", sweep.bb_length)?;
    let bb_multipliers = axis_f64("bb_multiplier", sweep.bb_multiplier)?;
    let cci_lengths = axis_usize("cci_length", sweep.cci_length)?;
    let dpo_lengths = axis_usize("dpo_length", sweep.dpo_length)?;
    let roc_lengths = axis_usize("roc_length", sweep.roc_length)?;
    let rsi_lengths = axis_usize("rsi_length", sweep.rsi_length)?;
    let stoch_lengths = axis_usize("stoch_length", sweep.stoch_length)?;
    let stoch_d_lengths = axis_usize("stoch_d_length", sweep.stoch_d_length)?;
    let stoch_k_lengths = axis_usize("stoch_k_length", sweep.stoch_k_length)?;
    let sma_lengths = axis_usize("sma_length", sweep.sma_length)?;

    let mut out = Vec::new();
    for emo_divisor in emo_divisors {
        for &emo_length in &emo_lengths {
            for &fast_length in &fast_lengths {
                for &slow_length in &slow_lengths {
                    for &mfi_length in &mfi_lengths {
                        for &bb_length in &bb_lengths {
                            for &bb_multiplier in &bb_multipliers {
                                for &cci_length in &cci_lengths {
                                    for &dpo_length in &dpo_lengths {
                                        for &roc_length in &roc_lengths {
                                            for &rsi_length in &rsi_lengths {
                                                for &stoch_length in &stoch_lengths {
                                                    for &stoch_d_length in &stoch_d_lengths {
                                                        for &stoch_k_length in &stoch_k_lengths {
                                                            for &sma_length in &sma_lengths {
                                                                out.push(InsyncIndexParams {
                                                                    emo_divisor: Some(emo_divisor),
                                                                    emo_length: Some(emo_length),
                                                                    fast_length: Some(fast_length),
                                                                    slow_length: Some(slow_length),
                                                                    mfi_length: Some(mfi_length),
                                                                    bb_length: Some(bb_length),
                                                                    bb_multiplier: Some(
                                                                        bb_multiplier,
                                                                    ),
                                                                    cci_length: Some(cci_length),
                                                                    dpo_length: Some(dpo_length),
                                                                    roc_length: Some(roc_length),
                                                                    rsi_length: Some(rsi_length),
                                                                    stoch_length: Some(
                                                                        stoch_length,
                                                                    ),
                                                                    stoch_d_length: Some(
                                                                        stoch_d_length,
                                                                    ),
                                                                    stoch_k_length: Some(
                                                                        stoch_k_length,
                                                                    ),
                                                                    sma_length: Some(sma_length),
                                                                });
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

pub fn insync_index_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &InsyncIndexBatchRange,
    kernel: Kernel,
) -> Result<InsyncIndexBatchOutput, InsyncIndexError> {
    let batch_kernel = match kernel {
        Kernel::Auto => Kernel::ScalarBatch,
        other if other.is_batch() => other,
        other => return Err(InsyncIndexError::InvalidKernelForBatch(other)),
    };
    insync_index_batch_impl(
        high,
        low,
        close,
        volume,
        sweep,
        batch_kernel.to_non_batch(),
        true,
    )
}

pub fn insync_index_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &InsyncIndexBatchRange,
) -> Result<InsyncIndexBatchOutput, InsyncIndexError> {
    insync_index_batch_impl(high, low, close, volume, sweep, Kernel::Scalar, false)
}

pub fn insync_index_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &InsyncIndexBatchRange,
) -> Result<InsyncIndexBatchOutput, InsyncIndexError> {
    insync_index_batch_impl(high, low, close, volume, sweep, Kernel::Scalar, true)
}

fn insync_index_batch_impl(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &InsyncIndexBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<InsyncIndexBatchOutput, InsyncIndexError> {
    let combos = expand_grid_insync_index(sweep)?;
    let rows = combos.len();
    let cols = close.len();

    if cols == 0 {
        return Err(InsyncIndexError::EmptyInputData);
    }
    if high.len() != cols || low.len() != cols || volume.len() != cols {
        return Err(InsyncIndexError::DataLengthMismatch);
    }
    if !has_any_valid_bar(high, low, close, volume) {
        return Err(InsyncIndexError::AllValuesNaN);
    }
    for params in &combos {
        resolve_params(params)?;
    }

    let matrix = make_uninit_matrix(rows, cols);
    let mut guard = ManuallyDrop::new(matrix);
    let values_mu: &mut [MaybeUninit<f64>] =
        unsafe { std::slice::from_raw_parts_mut(guard.as_mut_ptr(), guard.len()) };

    let do_row = |row: usize, row_mu: &mut [MaybeUninit<f64>]| {
        let params = resolve_params(&combos[row]).expect("validated params");
        let dst = unsafe {
            std::slice::from_raw_parts_mut(row_mu.as_mut_ptr() as *mut f64, row_mu.len())
        };
        insync_index_compute_into(high, low, close, volume, params, kernel, dst);
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            values_mu
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, row_mu)| do_row(row, row_mu));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, row_mu) in values_mu.chunks_mut(cols).enumerate() {
                do_row(row, row_mu);
            }
        }
    } else {
        for (row, row_mu) in values_mu.chunks_mut(cols).enumerate() {
            do_row(row, row_mu);
        }
    }

    let values =
        unsafe { Vec::from_raw_parts(guard.as_mut_ptr() as *mut f64, rows * cols, rows * cols) };
    Ok(InsyncIndexBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[cfg(feature = "python")]
#[pyfunction(name = "insync_index")]
#[pyo3(signature = (
    high,
    low,
    close,
    volume,
    emo_divisor=DEFAULT_EMO_DIVISOR,
    emo_length=DEFAULT_EMO_LENGTH,
    fast_length=DEFAULT_FAST_LENGTH,
    slow_length=DEFAULT_SLOW_LENGTH,
    mfi_length=DEFAULT_MFI_LENGTH,
    bb_length=DEFAULT_BB_LENGTH,
    bb_multiplier=DEFAULT_BB_MULTIPLIER,
    cci_length=DEFAULT_CCI_LENGTH,
    dpo_length=DEFAULT_DPO_LENGTH,
    roc_length=DEFAULT_ROC_LENGTH,
    rsi_length=DEFAULT_RSI_LENGTH,
    stoch_length=DEFAULT_STOCH_LENGTH,
    stoch_d_length=DEFAULT_STOCH_D_LENGTH,
    stoch_k_length=DEFAULT_STOCH_K_LENGTH,
    sma_length=DEFAULT_SMA_LENGTH,
    kernel=None
))]
pub fn insync_index_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    emo_divisor: usize,
    emo_length: usize,
    fast_length: usize,
    slow_length: usize,
    mfi_length: usize,
    bb_length: usize,
    bb_multiplier: f64,
    cci_length: usize,
    dpo_length: usize,
    roc_length: usize,
    rsi_length: usize,
    stoch_length: usize,
    stoch_d_length: usize,
    stoch_k_length: usize,
    sma_length: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let volume = volume.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = InsyncIndexInput::from_slices(
        high,
        low,
        close,
        volume,
        InsyncIndexParams {
            emo_divisor: Some(emo_divisor),
            emo_length: Some(emo_length),
            fast_length: Some(fast_length),
            slow_length: Some(slow_length),
            mfi_length: Some(mfi_length),
            bb_length: Some(bb_length),
            bb_multiplier: Some(bb_multiplier),
            cci_length: Some(cci_length),
            dpo_length: Some(dpo_length),
            roc_length: Some(roc_length),
            rsi_length: Some(rsi_length),
            stoch_length: Some(stoch_length),
            stoch_d_length: Some(stoch_d_length),
            stoch_k_length: Some(stoch_k_length),
            sma_length: Some(sma_length),
        },
    );
    let output = py
        .allow_threads(|| insync_index_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(output.values.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "InsyncIndexStream")]
pub struct InsyncIndexStreamPy {
    stream: InsyncIndexStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl InsyncIndexStreamPy {
    #[new]
    #[pyo3(signature = (
        emo_divisor=DEFAULT_EMO_DIVISOR,
        emo_length=DEFAULT_EMO_LENGTH,
        fast_length=DEFAULT_FAST_LENGTH,
        slow_length=DEFAULT_SLOW_LENGTH,
        mfi_length=DEFAULT_MFI_LENGTH,
        bb_length=DEFAULT_BB_LENGTH,
        bb_multiplier=DEFAULT_BB_MULTIPLIER,
        cci_length=DEFAULT_CCI_LENGTH,
        dpo_length=DEFAULT_DPO_LENGTH,
        roc_length=DEFAULT_ROC_LENGTH,
        rsi_length=DEFAULT_RSI_LENGTH,
        stoch_length=DEFAULT_STOCH_LENGTH,
        stoch_d_length=DEFAULT_STOCH_D_LENGTH,
        stoch_k_length=DEFAULT_STOCH_K_LENGTH,
        sma_length=DEFAULT_SMA_LENGTH
    ))]
    fn new(
        emo_divisor: usize,
        emo_length: usize,
        fast_length: usize,
        slow_length: usize,
        mfi_length: usize,
        bb_length: usize,
        bb_multiplier: f64,
        cci_length: usize,
        dpo_length: usize,
        roc_length: usize,
        rsi_length: usize,
        stoch_length: usize,
        stoch_d_length: usize,
        stoch_k_length: usize,
        sma_length: usize,
    ) -> PyResult<Self> {
        let stream = InsyncIndexStream::try_new(InsyncIndexParams {
            emo_divisor: Some(emo_divisor),
            emo_length: Some(emo_length),
            fast_length: Some(fast_length),
            slow_length: Some(slow_length),
            mfi_length: Some(mfi_length),
            bb_length: Some(bb_length),
            bb_multiplier: Some(bb_multiplier),
            cci_length: Some(cci_length),
            dpo_length: Some(dpo_length),
            roc_length: Some(roc_length),
            rsi_length: Some(rsi_length),
            stoch_length: Some(stoch_length),
            stoch_d_length: Some(stoch_d_length),
            stoch_k_length: Some(stoch_k_length),
            sma_length: Some(sma_length),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> Option<f64> {
        self.stream.update_reset_on_nan(high, low, close, volume)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "insync_index_batch")]
#[pyo3(signature = (
    high,
    low,
    close,
    volume,
    emo_divisor_range=(DEFAULT_EMO_DIVISOR, DEFAULT_EMO_DIVISOR, 0),
    emo_length_range=(DEFAULT_EMO_LENGTH, DEFAULT_EMO_LENGTH, 0),
    fast_length_range=(DEFAULT_FAST_LENGTH, DEFAULT_FAST_LENGTH, 0),
    slow_length_range=(DEFAULT_SLOW_LENGTH, DEFAULT_SLOW_LENGTH, 0),
    mfi_length_range=(DEFAULT_MFI_LENGTH, DEFAULT_MFI_LENGTH, 0),
    bb_length_range=(DEFAULT_BB_LENGTH, DEFAULT_BB_LENGTH, 0),
    bb_multiplier_range=(DEFAULT_BB_MULTIPLIER, DEFAULT_BB_MULTIPLIER, 0.0),
    cci_length_range=(DEFAULT_CCI_LENGTH, DEFAULT_CCI_LENGTH, 0),
    dpo_length_range=(DEFAULT_DPO_LENGTH, DEFAULT_DPO_LENGTH, 0),
    roc_length_range=(DEFAULT_ROC_LENGTH, DEFAULT_ROC_LENGTH, 0),
    rsi_length_range=(DEFAULT_RSI_LENGTH, DEFAULT_RSI_LENGTH, 0),
    stoch_length_range=(DEFAULT_STOCH_LENGTH, DEFAULT_STOCH_LENGTH, 0),
    stoch_d_length_range=(DEFAULT_STOCH_D_LENGTH, DEFAULT_STOCH_D_LENGTH, 0),
    stoch_k_length_range=(DEFAULT_STOCH_K_LENGTH, DEFAULT_STOCH_K_LENGTH, 0),
    sma_length_range=(DEFAULT_SMA_LENGTH, DEFAULT_SMA_LENGTH, 0),
    kernel=None
))]
pub fn insync_index_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    emo_divisor_range: (usize, usize, usize),
    emo_length_range: (usize, usize, usize),
    fast_length_range: (usize, usize, usize),
    slow_length_range: (usize, usize, usize),
    mfi_length_range: (usize, usize, usize),
    bb_length_range: (usize, usize, usize),
    bb_multiplier_range: (f64, f64, f64),
    cci_length_range: (usize, usize, usize),
    dpo_length_range: (usize, usize, usize),
    roc_length_range: (usize, usize, usize),
    rsi_length_range: (usize, usize, usize),
    stoch_length_range: (usize, usize, usize),
    stoch_d_length_range: (usize, usize, usize),
    stoch_k_length_range: (usize, usize, usize),
    sma_length_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let volume = volume.as_slice()?;
    let kernel = validate_kernel(kernel, true)?;
    let sweep = InsyncIndexBatchRange {
        emo_divisor: emo_divisor_range,
        emo_length: emo_length_range,
        fast_length: fast_length_range,
        slow_length: slow_length_range,
        mfi_length: mfi_length_range,
        bb_length: bb_length_range,
        bb_multiplier: bb_multiplier_range,
        cci_length: cci_length_range,
        dpo_length: dpo_length_range,
        roc_length: roc_length_range,
        rsi_length: rsi_length_range,
        stoch_length: stoch_length_range,
        stoch_d_length: stoch_d_length_range,
        stoch_k_length: stoch_k_length_range,
        sma_length: sma_length_range,
    };
    let output = py
        .allow_threads(|| insync_index_batch_with_kernel(high, low, close, volume, &sweep, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let values = output
        .values
        .into_pyarray(py)
        .reshape((output.rows, output.cols))?;
    let dict = PyDict::new(py);
    dict.set_item("values", values)?;
    dict.set_item("rows", output.rows)?;
    dict.set_item("cols", output.cols)?;
    dict.set_item(
        "emo_divisors",
        output
            .combos
            .iter()
            .map(|p| p.emo_divisor.unwrap_or(DEFAULT_EMO_DIVISOR))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "emo_lengths",
        output
            .combos
            .iter()
            .map(|p| p.emo_length.unwrap_or(DEFAULT_EMO_LENGTH))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "fast_lengths",
        output
            .combos
            .iter()
            .map(|p| p.fast_length.unwrap_or(DEFAULT_FAST_LENGTH))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "slow_lengths",
        output
            .combos
            .iter()
            .map(|p| p.slow_length.unwrap_or(DEFAULT_SLOW_LENGTH))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "mfi_lengths",
        output
            .combos
            .iter()
            .map(|p| p.mfi_length.unwrap_or(DEFAULT_MFI_LENGTH))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "bb_lengths",
        output
            .combos
            .iter()
            .map(|p| p.bb_length.unwrap_or(DEFAULT_BB_LENGTH))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "bb_multipliers",
        output
            .combos
            .iter()
            .map(|p| p.bb_multiplier.unwrap_or(DEFAULT_BB_MULTIPLIER))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "cci_lengths",
        output
            .combos
            .iter()
            .map(|p| p.cci_length.unwrap_or(DEFAULT_CCI_LENGTH))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "dpo_lengths",
        output
            .combos
            .iter()
            .map(|p| p.dpo_length.unwrap_or(DEFAULT_DPO_LENGTH))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "roc_lengths",
        output
            .combos
            .iter()
            .map(|p| p.roc_length.unwrap_or(DEFAULT_ROC_LENGTH))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "rsi_lengths",
        output
            .combos
            .iter()
            .map(|p| p.rsi_length.unwrap_or(DEFAULT_RSI_LENGTH))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "stoch_lengths",
        output
            .combos
            .iter()
            .map(|p| p.stoch_length.unwrap_or(DEFAULT_STOCH_LENGTH))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "stoch_d_lengths",
        output
            .combos
            .iter()
            .map(|p| p.stoch_d_length.unwrap_or(DEFAULT_STOCH_D_LENGTH))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "stoch_k_lengths",
        output
            .combos
            .iter()
            .map(|p| p.stoch_k_length.unwrap_or(DEFAULT_STOCH_K_LENGTH))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "sma_lengths",
        output
            .combos
            .iter()
            .map(|p| p.sma_length.unwrap_or(DEFAULT_SMA_LENGTH))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_insync_index_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(insync_index_py, m)?)?;
    m.add_function(wrap_pyfunction!(insync_index_batch_py, m)?)?;
    m.add_class::<InsyncIndexStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct InsyncIndexBatchConfig {
    emo_divisor_range: Option<Vec<usize>>,
    emo_length_range: Option<Vec<usize>>,
    fast_length_range: Option<Vec<usize>>,
    slow_length_range: Option<Vec<usize>>,
    mfi_length_range: Option<Vec<usize>>,
    bb_length_range: Option<Vec<usize>>,
    bb_multiplier_range: Option<Vec<f64>>,
    cci_length_range: Option<Vec<usize>>,
    dpo_length_range: Option<Vec<usize>>,
    roc_length_range: Option<Vec<usize>>,
    rsi_length_range: Option<Vec<usize>>,
    stoch_length_range: Option<Vec<usize>>,
    stoch_d_length_range: Option<Vec<usize>>,
    stoch_k_length_range: Option<Vec<usize>>,
    sma_length_range: Option<Vec<usize>>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct InsyncIndexBatchJsOutput {
    values: Vec<f64>,
    rows: usize,
    cols: usize,
    combos: Vec<InsyncIndexParams>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
fn parse_usize_js_range(
    range: Option<Vec<usize>>,
    default_value: usize,
    name: &'static str,
) -> Result<(usize, usize, usize), JsValue> {
    match range {
        Some(values) => {
            if values.len() != 3 {
                return Err(JsValue::from_str(&format!(
                    "Invalid config: {name} must have exactly 3 elements [start, end, step]"
                )));
            }
            Ok((values[0], values[1], values[2]))
        }
        None => Ok((default_value, default_value, 0)),
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
fn parse_f64_js_range(
    range: Option<Vec<f64>>,
    default_value: f64,
    name: &'static str,
) -> Result<(f64, f64, f64), JsValue> {
    match range {
        Some(values) => {
            if values.len() != 3 {
                return Err(JsValue::from_str(&format!(
                    "Invalid config: {name} must have exactly 3 elements [start, end, step]"
                )));
            }
            Ok((values[0], values[1], values[2]))
        }
        None => Ok((default_value, default_value, 0.0)),
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
fn sweep_from_js_config(config: InsyncIndexBatchConfig) -> Result<InsyncIndexBatchRange, JsValue> {
    Ok(InsyncIndexBatchRange {
        emo_divisor: parse_usize_js_range(
            config.emo_divisor_range,
            DEFAULT_EMO_DIVISOR,
            "emo_divisor_range",
        )?,
        emo_length: parse_usize_js_range(
            config.emo_length_range,
            DEFAULT_EMO_LENGTH,
            "emo_length_range",
        )?,
        fast_length: parse_usize_js_range(
            config.fast_length_range,
            DEFAULT_FAST_LENGTH,
            "fast_length_range",
        )?,
        slow_length: parse_usize_js_range(
            config.slow_length_range,
            DEFAULT_SLOW_LENGTH,
            "slow_length_range",
        )?,
        mfi_length: parse_usize_js_range(
            config.mfi_length_range,
            DEFAULT_MFI_LENGTH,
            "mfi_length_range",
        )?,
        bb_length: parse_usize_js_range(
            config.bb_length_range,
            DEFAULT_BB_LENGTH,
            "bb_length_range",
        )?,
        bb_multiplier: parse_f64_js_range(
            config.bb_multiplier_range,
            DEFAULT_BB_MULTIPLIER,
            "bb_multiplier_range",
        )?,
        cci_length: parse_usize_js_range(
            config.cci_length_range,
            DEFAULT_CCI_LENGTH,
            "cci_length_range",
        )?,
        dpo_length: parse_usize_js_range(
            config.dpo_length_range,
            DEFAULT_DPO_LENGTH,
            "dpo_length_range",
        )?,
        roc_length: parse_usize_js_range(
            config.roc_length_range,
            DEFAULT_ROC_LENGTH,
            "roc_length_range",
        )?,
        rsi_length: parse_usize_js_range(
            config.rsi_length_range,
            DEFAULT_RSI_LENGTH,
            "rsi_length_range",
        )?,
        stoch_length: parse_usize_js_range(
            config.stoch_length_range,
            DEFAULT_STOCH_LENGTH,
            "stoch_length_range",
        )?,
        stoch_d_length: parse_usize_js_range(
            config.stoch_d_length_range,
            DEFAULT_STOCH_D_LENGTH,
            "stoch_d_length_range",
        )?,
        stoch_k_length: parse_usize_js_range(
            config.stoch_k_length_range,
            DEFAULT_STOCH_K_LENGTH,
            "stoch_k_length_range",
        )?,
        sma_length: parse_usize_js_range(
            config.sma_length_range,
            DEFAULT_SMA_LENGTH,
            "sma_length_range",
        )?,
    })
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "insync_index_js")]
pub fn insync_index_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    emo_divisor: usize,
    emo_length: usize,
    fast_length: usize,
    slow_length: usize,
    mfi_length: usize,
    bb_length: usize,
    bb_multiplier: f64,
    cci_length: usize,
    dpo_length: usize,
    roc_length: usize,
    rsi_length: usize,
    stoch_length: usize,
    stoch_d_length: usize,
    stoch_k_length: usize,
    sma_length: usize,
) -> Result<Vec<f64>, JsValue> {
    let input = InsyncIndexInput::from_slices(
        high,
        low,
        close,
        volume,
        InsyncIndexParams {
            emo_divisor: Some(emo_divisor),
            emo_length: Some(emo_length),
            fast_length: Some(fast_length),
            slow_length: Some(slow_length),
            mfi_length: Some(mfi_length),
            bb_length: Some(bb_length),
            bb_multiplier: Some(bb_multiplier),
            cci_length: Some(cci_length),
            dpo_length: Some(dpo_length),
            roc_length: Some(roc_length),
            rsi_length: Some(rsi_length),
            stoch_length: Some(stoch_length),
            stoch_d_length: Some(stoch_d_length),
            stoch_k_length: Some(stoch_k_length),
            sma_length: Some(sma_length),
        },
    );
    insync_index(&input)
        .map(|out| out.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "insync_index_batch_js")]
pub fn insync_index_batch_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: InsyncIndexBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let sweep = sweep_from_js_config(config)?;
    let batch = insync_index_batch_slice(high, low, close, volume, &sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&InsyncIndexBatchJsOutput {
        values: batch.values,
        rows: batch.rows,
        cols: batch.cols,
        combos: batch.combos,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn insync_index_alloc(len: usize) -> *mut f64 {
    let mut buf = vec![0.0_f64; len];
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn insync_index_free(ptr: *mut f64, len: usize) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let _ = Vec::from_raw_parts(ptr, 0, len);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn insync_index_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    volume_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    emo_divisor: usize,
    emo_length: usize,
    fast_length: usize,
    slow_length: usize,
    mfi_length: usize,
    bb_length: usize,
    bb_multiplier: f64,
    cci_length: usize,
    dpo_length: usize,
    roc_length: usize,
    rsi_length: usize,
    stoch_length: usize,
    stoch_d_length: usize,
    stoch_k_length: usize,
    sma_length: usize,
) -> Result<(), JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || volume_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to insync_index_into",
        ));
    }
    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len);
        let input = InsyncIndexInput::from_slices(
            high,
            low,
            close,
            volume,
            InsyncIndexParams {
                emo_divisor: Some(emo_divisor),
                emo_length: Some(emo_length),
                fast_length: Some(fast_length),
                slow_length: Some(slow_length),
                mfi_length: Some(mfi_length),
                bb_length: Some(bb_length),
                bb_multiplier: Some(bb_multiplier),
                cci_length: Some(cci_length),
                dpo_length: Some(dpo_length),
                roc_length: Some(roc_length),
                rsi_length: Some(rsi_length),
                stoch_length: Some(stoch_length),
                stoch_d_length: Some(stoch_d_length),
                stoch_k_length: Some(stoch_k_length),
                sma_length: Some(sma_length),
            },
        );
        insync_index_into_slice(out, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "insync_index_into_host")]
pub fn insync_index_into_host(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    out_ptr: *mut f64,
    emo_divisor: usize,
    emo_length: usize,
    fast_length: usize,
    slow_length: usize,
    mfi_length: usize,
    bb_length: usize,
    bb_multiplier: f64,
    cci_length: usize,
    dpo_length: usize,
    roc_length: usize,
    rsi_length: usize,
    stoch_length: usize,
    stoch_d_length: usize,
    stoch_k_length: usize,
    sma_length: usize,
) -> Result<(), JsValue> {
    if out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to insync_index_into_host",
        ));
    }
    unsafe {
        let out = std::slice::from_raw_parts_mut(out_ptr, close.len());
        let input = InsyncIndexInput::from_slices(
            high,
            low,
            close,
            volume,
            InsyncIndexParams {
                emo_divisor: Some(emo_divisor),
                emo_length: Some(emo_length),
                fast_length: Some(fast_length),
                slow_length: Some(slow_length),
                mfi_length: Some(mfi_length),
                bb_length: Some(bb_length),
                bb_multiplier: Some(bb_multiplier),
                cci_length: Some(cci_length),
                dpo_length: Some(dpo_length),
                roc_length: Some(roc_length),
                rsi_length: Some(rsi_length),
                stoch_length: Some(stoch_length),
                stoch_d_length: Some(stoch_d_length),
                stoch_k_length: Some(stoch_k_length),
                sma_length: Some(sma_length),
            },
        );
        insync_index_into_slice(out, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn insync_index_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    volume_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    config: JsValue,
) -> Result<usize, JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || volume_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to insync_index_batch_into",
        ));
    }
    let config: InsyncIndexBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let sweep = sweep_from_js_config(config)?;
    let combos = expand_grid_insync_index(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, rows * len);
        let batch = insync_index_batch_slice(high, low, close, volume, &sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        out.copy_from_slice(&batch.values);
    }
    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn insync_index_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    emo_divisor: usize,
    emo_length: usize,
    fast_length: usize,
    slow_length: usize,
    mfi_length: usize,
    bb_length: usize,
    bb_multiplier: f64,
    cci_length: usize,
    dpo_length: usize,
    roc_length: usize,
    rsi_length: usize,
    stoch_length: usize,
    stoch_d_length: usize,
    stoch_k_length: usize,
    sma_length: usize,
    out: &js_sys::Float64Array,
) -> Result<usize, JsValue> {
    let values = insync_index_js(
        high,
        low,
        close,
        volume,
        emo_divisor,
        emo_length,
        fast_length,
        slow_length,
        mfi_length,
        bb_length,
        bb_multiplier,
        cci_length,
        dpo_length,
        roc_length,
        rsi_length,
        stoch_length,
        stoch_d_length,
        stoch_k_length,
        sma_length,
    )?;
    crate::write_wasm_f64_output("insync_index_output_into_js", &values, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn insync_index_batch_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = insync_index_batch_js(high, low, close, volume, config)?;
    crate::write_wasm_selected_object_f64_outputs("insync_index_batch_output_into_js", &value, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_ohlcv(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut high = Vec::with_capacity(len);
        let mut low = Vec::with_capacity(len);
        let mut close = Vec::with_capacity(len);
        let mut volume = Vec::with_capacity(len);
        for i in 0..len {
            let base = 100.0 + (i as f64) * 0.2 + ((i as f64) * 0.07).sin();
            let spread = 1.0 + ((i as f64) * 0.03).cos().abs();
            low.push(base - spread);
            high.push(base + spread);
            close.push(base + ((i as f64) * 0.11).sin() * 0.35);
            volume.push(1_000.0 + (i as f64) * 3.0);
        }
        (high, low, close, volume)
    }

    #[test]
    fn short_input_returns_baseline_not_error() {
        let high = [11.0, 12.0, 13.0];
        let low = [9.0, 10.0, 11.0];
        let close = [10.0, 11.0, 12.0];
        let volume = [100.0, 100.0, 100.0];
        let input = InsyncIndexInput::from_slices(
            &high,
            &low,
            &close,
            &volume,
            InsyncIndexParams::default(),
        );
        let out = insync_index(&input).unwrap();
        assert_eq!(out.values.len(), close.len());
        assert!(out.values[0].is_finite());
        assert_eq!(out.values[0], 50.0);
    }

    #[test]
    fn stream_matches_batch_default() {
        let (high, low, close, volume) = sample_ohlcv(256);
        let input = InsyncIndexInput::from_slices(
            &high,
            &low,
            &close,
            &volume,
            InsyncIndexParams::default(),
        );
        let batch = insync_index(&input).unwrap();

        let mut stream = InsyncIndexStream::try_new(InsyncIndexParams::default()).unwrap();
        let streamed: Vec<f64> = high
            .iter()
            .zip(&low)
            .zip(&close)
            .zip(&volume)
            .map(|(((h, l), c), v)| stream.update(*h, *l, *c, *v).unwrap_or(f64::NAN))
            .collect();

        assert_eq!(batch.values.len(), streamed.len());
        for (a, b) in batch.values.iter().zip(streamed.iter()) {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert!((a - b).abs() <= 1e-12, "batch {a} != stream {b}");
        }
    }

    #[test]
    fn batch_single_param_matches_single() {
        let (high, low, close, volume) = sample_ohlcv(192);
        let input = InsyncIndexInput::from_slices(
            &high,
            &low,
            &close,
            &volume,
            InsyncIndexParams::default(),
        );
        let single = insync_index(&input).unwrap();
        let sweep = InsyncIndexBatchRange::default();
        let batch = insync_index_batch_slice(&high, &low, &close, &volume, &sweep).unwrap();
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, close.len());
        for (a, b) in batch.values[..close.len()].iter().zip(single.values.iter()) {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert!((a - b).abs() <= 1e-12, "batch {a} != single {b}");
        }
    }

    #[test]
    fn invalid_bb_multiplier_rejected() {
        let err = InsyncIndexStream::try_new(InsyncIndexParams {
            bb_multiplier: Some(0.0),
            ..InsyncIndexParams::default()
        })
        .unwrap_err();
        assert!(matches!(
            err,
            InsyncIndexError::InvalidFloat {
                name: "bb_multiplier",
                ..
            }
        ));
    }
}
