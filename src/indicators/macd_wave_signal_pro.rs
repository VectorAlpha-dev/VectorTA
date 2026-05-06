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
use std::mem::ManuallyDrop;
use thiserror::Error;

const DIFF_FAST_PERIOD: usize = 12;
const DIFF_SLOW_PERIOD: usize = 26;
const DEA_PERIOD: usize = 9;
const LINE_LONG_PERIOD: usize = 40;
const LINE_SHORT_START: usize = 6;
const LINE_SHORT_END: usize = 20;
const LINE_SHORT_COUNT: usize = LINE_SHORT_END - LINE_SHORT_START + 1;
const LINE_SHORT_AVG_INV: f64 = 1.0 / LINE_SHORT_COUNT as f64;
const MID_CLOSE_WEIGHT: f64 = 7.0;
const MID_DIVISOR: f64 = 10.0;
const MACD_SCALE: f64 = 2.0;

#[derive(Debug, Clone)]
pub enum MacdWaveSignalProData<'a> {
    Candles(&'a Candles),
    Slices {
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct MacdWaveSignalProOutput {
    pub diff: Vec<f64>,
    pub dea: Vec<f64>,
    pub macd_histogram: Vec<f64>,
    pub line_convergence: Vec<f64>,
    pub buy_signal: Vec<f64>,
    pub sell_signal: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
pub struct MacdWaveSignalProPoint {
    pub diff: f64,
    pub dea: f64,
    pub macd_histogram: f64,
    pub line_convergence: f64,
    pub buy_signal: f64,
    pub sell_signal: f64,
}

impl MacdWaveSignalProPoint {
    #[inline(always)]
    fn nan() -> Self {
        Self {
            diff: f64::NAN,
            dea: f64::NAN,
            macd_histogram: f64::NAN,
            line_convergence: f64::NAN,
            buy_signal: f64::NAN,
            sell_signal: f64::NAN,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct MacdWaveSignalProParams;

#[derive(Debug, Clone)]
pub struct MacdWaveSignalProInput<'a> {
    pub data: MacdWaveSignalProData<'a>,
    pub params: MacdWaveSignalProParams,
}

impl<'a> MacdWaveSignalProInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: MacdWaveSignalProParams) -> Self {
        Self {
            data: MacdWaveSignalProData::Candles(candles),
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: MacdWaveSignalProParams,
    ) -> Self {
        Self {
            data: MacdWaveSignalProData::Slices {
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
        Self::from_candles(candles, MacdWaveSignalProParams)
    }

    #[inline]
    pub fn as_slices(&self) -> (&'a [f64], &'a [f64], &'a [f64], &'a [f64]) {
        match &self.data {
            MacdWaveSignalProData::Candles(candles) => {
                (&candles.open, &candles.high, &candles.low, &candles.close)
            }
            MacdWaveSignalProData::Slices {
                open,
                high,
                low,
                close,
            } => (open, high, low, close),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MacdWaveSignalProBuilder {
    kernel: Kernel,
}

impl MacdWaveSignalProBuilder {
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
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<MacdWaveSignalProOutput, MacdWaveSignalProError> {
        let input = MacdWaveSignalProInput::from_candles(candles, MacdWaveSignalProParams);
        macd_wave_signal_pro_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn apply_slices(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<MacdWaveSignalProOutput, MacdWaveSignalProError> {
        let input =
            MacdWaveSignalProInput::from_slices(open, high, low, close, MacdWaveSignalProParams);
        macd_wave_signal_pro_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn into_stream(self) -> Result<MacdWaveSignalProStream, MacdWaveSignalProError> {
        let _ = self.kernel;
        MacdWaveSignalProStream::try_new(MacdWaveSignalProParams)
    }
}

#[derive(Debug, Error)]
pub enum MacdWaveSignalProError {
    #[error("macd_wave_signal_pro: Input data slice is empty.")]
    EmptyInputData,
    #[error("macd_wave_signal_pro: All values are NaN.")]
    AllValuesNaN,
    #[error(
        "macd_wave_signal_pro: Inconsistent slice lengths: open={open_len}, high={high_len}, low={low_len}, close={close_len}"
    )]
    InconsistentSliceLengths {
        open_len: usize,
        high_len: usize,
        low_len: usize,
        close_len: usize,
    },
    #[error("macd_wave_signal_pro: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "macd_wave_signal_pro: Output length mismatch: expected = {expected}, diff = {diff_got}, dea = {dea_got}, macd_histogram = {macd_histogram_got}, line_convergence = {line_convergence_got}, buy_signal = {buy_signal_got}, sell_signal = {sell_signal_got}"
    )]
    OutputLengthMismatch {
        expected: usize,
        diff_got: usize,
        dea_got: usize,
        macd_histogram_got: usize,
        line_convergence_got: usize,
        buy_signal_got: usize,
        sell_signal_got: usize,
    },
    #[error("macd_wave_signal_pro: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("macd_wave_signal_pro: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Clone, Debug)]
struct RollingSma {
    period: usize,
    values: Vec<f64>,
    index: usize,
    count: usize,
    sum: f64,
}

impl RollingSma {
    #[inline]
    fn new(period: usize) -> Self {
        Self {
            period,
            values: vec![0.0; period],
            index: 0,
            count: 0,
            sum: 0.0,
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.index = 0;
        self.count = 0;
        self.sum = 0.0;
    }

    #[inline]
    fn update(&mut self, value: f64) -> Option<f64> {
        if self.count < self.period {
            self.values[self.index] = value;
            self.sum += value;
            self.index += 1;
            if self.index == self.period {
                self.index = 0;
            }
            self.count += 1;
            if self.count == self.period {
                Some(self.sum / self.period as f64)
            } else {
                None
            }
        } else {
            let old = self.values[self.index];
            self.values[self.index] = value;
            self.sum += value - old;
            self.index += 1;
            if self.index == self.period {
                self.index = 0;
            }
            Some(self.sum / self.period as f64)
        }
    }
}

#[derive(Clone, Debug)]
struct RollingShortSmas {
    values: [[f64; LINE_SHORT_END]; LINE_SHORT_COUNT],
    index: [usize; LINE_SHORT_COUNT],
    count: [usize; LINE_SHORT_COUNT],
    sum: [f64; LINE_SHORT_COUNT],
}

impl RollingShortSmas {
    #[inline]
    fn new() -> Self {
        Self {
            values: [[0.0; LINE_SHORT_END]; LINE_SHORT_COUNT],
            index: [0; LINE_SHORT_COUNT],
            count: [0; LINE_SHORT_COUNT],
            sum: [0.0; LINE_SHORT_COUNT],
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.index = [0; LINE_SHORT_COUNT];
        self.count = [0; LINE_SHORT_COUNT];
        self.sum = [0.0; LINE_SHORT_COUNT];
    }

    #[inline]
    fn update(&mut self, value: f64) -> Option<f64> {
        let mut short_sum = 0.0;
        let mut short_ready = 0usize;
        let mut i = 0usize;
        while i < LINE_SHORT_COUNT {
            let period = LINE_SHORT_START + i;
            let index = self.index[i];
            if self.count[i] < period {
                self.values[i][index] = value;
                self.sum[i] += value;
                let mut next = index + 1;
                if next == period {
                    next = 0;
                }
                self.index[i] = next;
                self.count[i] += 1;
                if self.count[i] == period {
                    short_sum += self.sum[i] / period as f64;
                    short_ready += 1;
                }
            } else {
                let old = self.values[i][index];
                self.values[i][index] = value;
                self.sum[i] += value - old;
                let mut next = index + 1;
                if next == period {
                    next = 0;
                }
                self.index[i] = next;
                short_sum += self.sum[i] / period as f64;
                short_ready += 1;
            }
            i += 1;
        }

        if short_ready == LINE_SHORT_COUNT {
            Some(short_sum * LINE_SHORT_AVG_INV)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
struct SeededEma {
    period: usize,
    alpha: f64,
    beta: f64,
    count: usize,
    sum: f64,
    value: f64,
    ready: bool,
}

impl SeededEma {
    #[inline]
    fn new(period: usize) -> Self {
        let alpha = 2.0 / (period as f64 + 1.0);
        Self {
            period,
            alpha,
            beta: 1.0 - alpha,
            count: 0,
            sum: 0.0,
            value: f64::NAN,
            ready: false,
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.count = 0;
        self.sum = 0.0;
        self.value = f64::NAN;
        self.ready = false;
    }

    #[inline]
    fn update(&mut self, value: f64) -> Option<f64> {
        if self.count < self.period {
            self.count += 1;
            self.sum += value;
            if self.count == self.period {
                self.value = self.sum / self.period as f64;
                self.ready = true;
                Some(self.value)
            } else {
                None
            }
        } else {
            self.value = value.mul_add(self.alpha, self.beta * self.value);
            Some(self.value)
        }
    }
}

#[derive(Clone, Debug)]
struct CoreState {
    ema_fast: SeededEma,
    ema_slow: SeededEma,
    ema_dea: SeededEma,
    line_short: RollingShortSmas,
    line_long: RollingSma,
    prev_diff: f64,
    prev_dea: f64,
}

impl CoreState {
    #[inline]
    fn new() -> Self {
        Self {
            ema_fast: SeededEma::new(DIFF_FAST_PERIOD),
            ema_slow: SeededEma::new(DIFF_SLOW_PERIOD),
            ema_dea: SeededEma::new(DEA_PERIOD),
            line_short: RollingShortSmas::new(),
            line_long: RollingSma::new(LINE_LONG_PERIOD),
            prev_diff: f64::NAN,
            prev_dea: f64::NAN,
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.ema_fast.reset();
        self.ema_slow.reset();
        self.ema_dea.reset();
        self.line_short.reset();
        self.line_long.reset();
        self.prev_diff = f64::NAN;
        self.prev_dea = f64::NAN;
    }

    #[inline]
    fn update(&mut self, open: f64, high: f64, low: f64, close: f64) -> MacdWaveSignalProPoint {
        let mut point = MacdWaveSignalProPoint::nan();

        let fast = self.ema_fast.update(close);
        let slow = self.ema_slow.update(close);
        if let (Some(fast), Some(slow)) = (fast, slow) {
            let diff = fast - slow;
            point.diff = diff;

            if let Some(dea) = self.ema_dea.update(diff) {
                point.dea = dea;
                point.macd_histogram = MACD_SCALE * (diff - dea);
                if self.prev_diff.is_finite() && self.prev_dea.is_finite() {
                    point.buy_signal = if diff > dea && self.prev_diff <= self.prev_dea {
                        1.0
                    } else {
                        0.0
                    };
                    point.sell_signal = if diff < dea && self.prev_diff >= self.prev_dea {
                        1.0
                    } else {
                        0.0
                    };
                } else {
                    point.buy_signal = 0.0;
                    point.sell_signal = 0.0;
                }
            }

            self.prev_diff = diff;
            self.prev_dea = point.dea;
        }

        let mid = (MID_CLOSE_WEIGHT.mul_add(close, open + high + low)) / MID_DIVISOR;
        let short_avg = self.line_short.update(mid);
        if let Some(long) = self.line_long.update(mid) {
            if let Some(short_avg) = short_avg {
                point.line_convergence = short_avg - long;
            }
        }

        point
    }
}

#[derive(Debug, Clone)]
pub struct MacdWaveSignalProStream {
    state: CoreState,
}

impl MacdWaveSignalProStream {
    #[inline]
    pub fn try_new(_params: MacdWaveSignalProParams) -> Result<Self, MacdWaveSignalProError> {
        Ok(Self {
            state: CoreState::new(),
        })
    }

    #[inline]
    pub fn update(
        &mut self,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
    ) -> Option<MacdWaveSignalProPoint> {
        if !valid_ohlc_bar(open, high, low, close) {
            self.state.reset();
            return None;
        }
        Some(self.state.update(open, high, low, close))
    }

    #[inline]
    pub fn reset(&mut self) {
        self.state.reset();
    }

    #[inline]
    pub fn get_warmup_period(&self) -> usize {
        LINE_LONG_PERIOD.saturating_sub(1)
    }
}

#[inline(always)]
fn valid_ohlc_bar(open: f64, high: f64, low: f64, close: f64) -> bool {
    open.is_finite() && high.is_finite() && low.is_finite() && close.is_finite()
}

#[inline(always)]
fn first_valid_ohlc(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> usize {
    let len = close.len();
    let mut i = 0usize;
    while i < len {
        if valid_ohlc_bar(open[i], high[i], low[i], close[i]) {
            return i;
        }
        i += 1;
    }
    len
}

#[inline(always)]
fn count_valid_ohlc(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> usize {
    let mut count = 0usize;
    let len = close.len();
    let mut i = 0usize;
    while i < len {
        if valid_ohlc_bar(open[i], high[i], low[i], close[i]) {
            count += 1;
        }
        i += 1;
    }
    count
}

#[inline(always)]
fn valid_ohlc_summary(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> (usize, usize) {
    let len = close.len();
    let mut first = len;
    let mut count = 0usize;
    let mut i = 0usize;
    while i < len {
        if valid_ohlc_bar(open[i], high[i], low[i], close[i]) {
            if first == len {
                first = i;
            }
            count += 1;
        }
        i += 1;
    }
    (first, count)
}

#[inline(always)]
fn output_warmups(first: usize, len: usize) -> [usize; 6] {
    [
        first
            .saturating_add(DIFF_SLOW_PERIOD)
            .saturating_sub(1)
            .min(len),
        first
            .saturating_add(DIFF_SLOW_PERIOD + DEA_PERIOD)
            .saturating_sub(2)
            .min(len),
        first
            .saturating_add(DIFF_SLOW_PERIOD + DEA_PERIOD)
            .saturating_sub(2)
            .min(len),
        first
            .saturating_add(LINE_LONG_PERIOD)
            .saturating_sub(1)
            .min(len),
        first
            .saturating_add(DIFF_SLOW_PERIOD + DEA_PERIOD)
            .saturating_sub(2)
            .min(len),
        first
            .saturating_add(DIFF_SLOW_PERIOD + DEA_PERIOD)
            .saturating_sub(2)
            .min(len),
    ]
}

#[inline(always)]
fn max_required_valid() -> usize {
    LINE_LONG_PERIOD.max(DIFF_SLOW_PERIOD + DEA_PERIOD - 1)
}

#[inline(always)]
fn prepare<'a>(
    input: &'a MacdWaveSignalProInput,
    kernel: Kernel,
) -> Result<
    (
        (&'a [f64], &'a [f64], &'a [f64], &'a [f64]),
        usize,
        usize,
        Kernel,
    ),
    MacdWaveSignalProError,
> {
    let (open, high, low, close) = input.as_slices();
    if open.is_empty() || high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(MacdWaveSignalProError::EmptyInputData);
    }
    if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
        return Err(MacdWaveSignalProError::InconsistentSliceLengths {
            open_len: open.len(),
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }
    let (first, valid) = valid_ohlc_summary(open, high, low, close);
    if first >= close.len() {
        return Err(MacdWaveSignalProError::AllValuesNaN);
    }
    let needed = max_required_valid();
    if valid < needed {
        return Err(MacdWaveSignalProError::NotEnoughValidData { needed, valid });
    }
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other.to_non_batch(),
    };
    Ok(((open, high, low, close), first, valid, chosen))
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn macd_wave_signal_pro_row_from_slices(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    diff: &mut [f64],
    dea: &mut [f64],
    macd_histogram: &mut [f64],
    line_convergence: &mut [f64],
    buy_signal: &mut [f64],
    sell_signal: &mut [f64],
) {
    let len = close.len();
    debug_assert_eq!(open.len(), len);
    debug_assert_eq!(high.len(), len);
    debug_assert_eq!(low.len(), len);
    debug_assert_eq!(diff.len(), len);
    debug_assert_eq!(dea.len(), len);
    debug_assert_eq!(macd_histogram.len(), len);
    debug_assert_eq!(line_convergence.len(), len);
    debug_assert_eq!(buy_signal.len(), len);
    debug_assert_eq!(sell_signal.len(), len);

    let mut state = CoreState::new();
    let mut i = 0usize;
    while i < len {
        if valid_ohlc_bar(open[i], high[i], low[i], close[i]) {
            let point = state.update(open[i], high[i], low[i], close[i]);
            diff[i] = point.diff;
            dea[i] = point.dea;
            macd_histogram[i] = point.macd_histogram;
            line_convergence[i] = point.line_convergence;
            buy_signal[i] = point.buy_signal;
            sell_signal[i] = point.sell_signal;
        } else {
            state.reset();
            diff[i] = f64::NAN;
            dea[i] = f64::NAN;
            macd_histogram[i] = f64::NAN;
            line_convergence[i] = f64::NAN;
            buy_signal[i] = f64::NAN;
            sell_signal[i] = f64::NAN;
        }
        i += 1;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn macd_wave_signal_pro_row_clean_from_slices(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    diff: &mut [f64],
    dea: &mut [f64],
    macd_histogram: &mut [f64],
    line_convergence: &mut [f64],
    buy_signal: &mut [f64],
    sell_signal: &mut [f64],
) {
    if first > 0 {
        for dst in [
            &mut *diff,
            &mut *dea,
            &mut *macd_histogram,
            &mut *line_convergence,
            &mut *buy_signal,
            &mut *sell_signal,
        ] {
            for v in &mut dst[..first] {
                *v = f64::NAN;
            }
        }
    }

    let len = close.len();
    let mut state = CoreState::new();
    let mut i = first;
    unsafe {
        let open_ptr = open.as_ptr();
        let high_ptr = high.as_ptr();
        let low_ptr = low.as_ptr();
        let close_ptr = close.as_ptr();
        let diff_ptr = diff.as_mut_ptr();
        let dea_ptr = dea.as_mut_ptr();
        let hist_ptr = macd_histogram.as_mut_ptr();
        let line_ptr = line_convergence.as_mut_ptr();
        let buy_ptr = buy_signal.as_mut_ptr();
        let sell_ptr = sell_signal.as_mut_ptr();
        while i < len {
            let point = state.update(
                *open_ptr.add(i),
                *high_ptr.add(i),
                *low_ptr.add(i),
                *close_ptr.add(i),
            );
            *diff_ptr.add(i) = point.diff;
            *dea_ptr.add(i) = point.dea;
            *hist_ptr.add(i) = point.macd_histogram;
            *line_ptr.add(i) = point.line_convergence;
            *buy_ptr.add(i) = point.buy_signal;
            *sell_ptr.add(i) = point.sell_signal;
            i += 1;
        }
    }
}

#[inline]
pub fn macd_wave_signal_pro(
    input: &MacdWaveSignalProInput,
) -> Result<MacdWaveSignalProOutput, MacdWaveSignalProError> {
    macd_wave_signal_pro_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn macd_wave_signal_pro_with_kernel(
    input: &MacdWaveSignalProInput,
    kernel: Kernel,
) -> Result<MacdWaveSignalProOutput, MacdWaveSignalProError> {
    let ((open, high, low, close), first, valid, _chosen) = prepare(input, kernel)?;
    let len = close.len();
    let warmups = output_warmups(first, len);
    let mut diff = alloc_with_nan_prefix(len, warmups[0]);
    let mut dea = alloc_with_nan_prefix(len, warmups[1]);
    let mut macd_histogram = alloc_with_nan_prefix(len, warmups[2]);
    let mut line_convergence = alloc_with_nan_prefix(len, warmups[3]);
    let mut buy_signal = alloc_with_nan_prefix(len, warmups[4]);
    let mut sell_signal = alloc_with_nan_prefix(len, warmups[5]);
    if valid == len - first {
        macd_wave_signal_pro_row_clean_from_slices(
            open,
            high,
            low,
            close,
            first,
            &mut diff,
            &mut dea,
            &mut macd_histogram,
            &mut line_convergence,
            &mut buy_signal,
            &mut sell_signal,
        );
    } else {
        macd_wave_signal_pro_row_from_slices(
            open,
            high,
            low,
            close,
            &mut diff,
            &mut dea,
            &mut macd_histogram,
            &mut line_convergence,
            &mut buy_signal,
            &mut sell_signal,
        );
    }
    Ok(MacdWaveSignalProOutput {
        diff,
        dea,
        macd_histogram,
        line_convergence,
        buy_signal,
        sell_signal,
    })
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub fn macd_wave_signal_pro_into_slices(
    diff_out: &mut [f64],
    dea_out: &mut [f64],
    macd_histogram_out: &mut [f64],
    line_convergence_out: &mut [f64],
    buy_signal_out: &mut [f64],
    sell_signal_out: &mut [f64],
    input: &MacdWaveSignalProInput,
    kernel: Kernel,
) -> Result<(), MacdWaveSignalProError> {
    let ((open, high, low, close), first, valid, _chosen) = prepare(input, kernel)?;
    let len = close.len();
    if diff_out.len() != len
        || dea_out.len() != len
        || macd_histogram_out.len() != len
        || line_convergence_out.len() != len
        || buy_signal_out.len() != len
        || sell_signal_out.len() != len
    {
        return Err(MacdWaveSignalProError::OutputLengthMismatch {
            expected: len,
            diff_got: diff_out.len(),
            dea_got: dea_out.len(),
            macd_histogram_got: macd_histogram_out.len(),
            line_convergence_got: line_convergence_out.len(),
            buy_signal_got: buy_signal_out.len(),
            sell_signal_got: sell_signal_out.len(),
        });
    }

    if valid == len - first {
        macd_wave_signal_pro_row_clean_from_slices(
            open,
            high,
            low,
            close,
            first,
            diff_out,
            dea_out,
            macd_histogram_out,
            line_convergence_out,
            buy_signal_out,
            sell_signal_out,
        );
    } else {
        macd_wave_signal_pro_row_from_slices(
            open,
            high,
            low,
            close,
            diff_out,
            dea_out,
            macd_histogram_out,
            line_convergence_out,
            buy_signal_out,
            sell_signal_out,
        );
    }
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn macd_wave_signal_pro_into(
    input: &MacdWaveSignalProInput,
    diff_out: &mut [f64],
    dea_out: &mut [f64],
    macd_histogram_out: &mut [f64],
    line_convergence_out: &mut [f64],
    buy_signal_out: &mut [f64],
    sell_signal_out: &mut [f64],
) -> Result<(), MacdWaveSignalProError> {
    macd_wave_signal_pro_into_slices(
        diff_out,
        dea_out,
        macd_histogram_out,
        line_convergence_out,
        buy_signal_out,
        sell_signal_out,
        input,
        Kernel::Auto,
    )
}

#[derive(Clone, Debug, Default)]
pub struct MacdWaveSignalProBatchRange;

#[derive(Clone, Debug, Default)]
pub struct MacdWaveSignalProBatchBuilder {
    kernel: Kernel,
}

impl MacdWaveSignalProBatchBuilder {
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
    pub fn apply_slices(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<MacdWaveSignalProBatchOutput, MacdWaveSignalProError> {
        macd_wave_signal_pro_batch_with_kernel(
            open,
            high,
            low,
            close,
            &MacdWaveSignalProBatchRange,
            self.kernel,
        )
    }

    #[inline]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<MacdWaveSignalProBatchOutput, MacdWaveSignalProError> {
        self.apply_slices(&candles.open, &candles.high, &candles.low, &candles.close)
    }
}

#[derive(Clone, Debug)]
pub struct MacdWaveSignalProBatchOutput {
    pub diff: Vec<f64>,
    pub dea: Vec<f64>,
    pub macd_histogram: Vec<f64>,
    pub line_convergence: Vec<f64>,
    pub buy_signal: Vec<f64>,
    pub sell_signal: Vec<f64>,
    pub combos: Vec<MacdWaveSignalProParams>,
    pub rows: usize,
    pub cols: usize,
}

impl MacdWaveSignalProBatchOutput {
    #[inline]
    pub fn row_for_params(&self, _params: &MacdWaveSignalProParams) -> Option<usize> {
        if self.rows == 0 {
            None
        } else {
            Some(0)
        }
    }

    #[inline]
    pub fn values_for(
        &self,
        _params: &MacdWaveSignalProParams,
    ) -> Option<(&[f64], &[f64], &[f64], &[f64], &[f64], &[f64])> {
        if self.rows == 0 {
            None
        } else {
            Some((
                &self.diff[..self.cols],
                &self.dea[..self.cols],
                &self.macd_histogram[..self.cols],
                &self.line_convergence[..self.cols],
                &self.buy_signal[..self.cols],
                &self.sell_signal[..self.cols],
            ))
        }
    }
}

#[inline(always)]
fn expand_grid_macd_wave_signal_pro(
    _range: &MacdWaveSignalProBatchRange,
) -> Result<Vec<MacdWaveSignalProParams>, MacdWaveSignalProError> {
    Ok(vec![MacdWaveSignalProParams])
}

#[inline]
pub fn macd_wave_signal_pro_batch_with_kernel(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &MacdWaveSignalProBatchRange,
    kernel: Kernel,
) -> Result<MacdWaveSignalProBatchOutput, MacdWaveSignalProError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(MacdWaveSignalProError::InvalidKernelForBatch(other)),
    };
    macd_wave_signal_pro_batch_par_slice(open, high, low, close, sweep, batch_kernel.to_non_batch())
}

#[inline]
pub fn macd_wave_signal_pro_batch_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &MacdWaveSignalProBatchRange,
    kernel: Kernel,
) -> Result<MacdWaveSignalProBatchOutput, MacdWaveSignalProError> {
    macd_wave_signal_pro_batch_inner(open, high, low, close, sweep, kernel, false)
}

#[inline]
pub fn macd_wave_signal_pro_batch_par_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &MacdWaveSignalProBatchRange,
    kernel: Kernel,
) -> Result<MacdWaveSignalProBatchOutput, MacdWaveSignalProError> {
    macd_wave_signal_pro_batch_inner(open, high, low, close, sweep, kernel, true)
}

#[allow(clippy::too_many_arguments)]
fn macd_wave_signal_pro_batch_inner(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &MacdWaveSignalProBatchRange,
    _kernel: Kernel,
    parallel: bool,
) -> Result<MacdWaveSignalProBatchOutput, MacdWaveSignalProError> {
    if open.is_empty() || high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(MacdWaveSignalProError::EmptyInputData);
    }
    if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
        return Err(MacdWaveSignalProError::InconsistentSliceLengths {
            open_len: open.len(),
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }
    let first = first_valid_ohlc(open, high, low, close);
    if first >= close.len() {
        return Err(MacdWaveSignalProError::AllValuesNaN);
    }
    let valid = count_valid_ohlc(open, high, low, close);
    let needed = max_required_valid();
    if valid < needed {
        return Err(MacdWaveSignalProError::NotEnoughValidData { needed, valid });
    }

    let combos = expand_grid_macd_wave_signal_pro(sweep)?;
    let rows = combos.len();
    let cols = close.len();

    let mut diff_mu = make_uninit_matrix(rows, cols);
    let mut dea_mu = make_uninit_matrix(rows, cols);
    let mut macd_mu = make_uninit_matrix(rows, cols);
    let mut line_mu = make_uninit_matrix(rows, cols);
    let mut buy_mu = make_uninit_matrix(rows, cols);
    let mut sell_mu = make_uninit_matrix(rows, cols);
    let warmups = output_warmups(first, cols);
    init_matrix_prefixes(&mut diff_mu, cols, &[warmups[0]]);
    init_matrix_prefixes(&mut dea_mu, cols, &[warmups[1]]);
    init_matrix_prefixes(&mut macd_mu, cols, &[warmups[2]]);
    init_matrix_prefixes(&mut line_mu, cols, &[warmups[3]]);
    init_matrix_prefixes(&mut buy_mu, cols, &[warmups[4]]);
    init_matrix_prefixes(&mut sell_mu, cols, &[warmups[5]]);

    let mut diff_guard = ManuallyDrop::new(diff_mu);
    let mut dea_guard = ManuallyDrop::new(dea_mu);
    let mut macd_guard = ManuallyDrop::new(macd_mu);
    let mut line_guard = ManuallyDrop::new(line_mu);
    let mut buy_guard = ManuallyDrop::new(buy_mu);
    let mut sell_guard = ManuallyDrop::new(sell_mu);

    let diff_out = unsafe {
        std::slice::from_raw_parts_mut(diff_guard.as_mut_ptr() as *mut f64, diff_guard.len())
    };
    let dea_out = unsafe {
        std::slice::from_raw_parts_mut(dea_guard.as_mut_ptr() as *mut f64, dea_guard.len())
    };
    let macd_out = unsafe {
        std::slice::from_raw_parts_mut(macd_guard.as_mut_ptr() as *mut f64, macd_guard.len())
    };
    let line_out = unsafe {
        std::slice::from_raw_parts_mut(line_guard.as_mut_ptr() as *mut f64, line_guard.len())
    };
    let buy_out = unsafe {
        std::slice::from_raw_parts_mut(buy_guard.as_mut_ptr() as *mut f64, buy_guard.len())
    };
    let sell_out = unsafe {
        std::slice::from_raw_parts_mut(sell_guard.as_mut_ptr() as *mut f64, sell_guard.len())
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;

            diff_out
                .par_chunks_mut(cols)
                .zip(dea_out.par_chunks_mut(cols))
                .zip(macd_out.par_chunks_mut(cols))
                .zip(line_out.par_chunks_mut(cols))
                .zip(buy_out.par_chunks_mut(cols))
                .zip(sell_out.par_chunks_mut(cols))
                .for_each(
                    |(((((dst_diff, dst_dea), dst_macd), dst_line), dst_buy), dst_sell)| {
                        macd_wave_signal_pro_row_from_slices(
                            open, high, low, close, dst_diff, dst_dea, dst_macd, dst_line, dst_buy,
                            dst_sell,
                        );
                    },
                );
        }

        #[cfg(target_arch = "wasm32")]
        {
            macd_wave_signal_pro_row_from_slices(
                open,
                high,
                low,
                close,
                &mut diff_out[..cols],
                &mut dea_out[..cols],
                &mut macd_out[..cols],
                &mut line_out[..cols],
                &mut buy_out[..cols],
                &mut sell_out[..cols],
            );
        }
    } else {
        macd_wave_signal_pro_row_from_slices(
            open,
            high,
            low,
            close,
            &mut diff_out[..cols],
            &mut dea_out[..cols],
            &mut macd_out[..cols],
            &mut line_out[..cols],
            &mut buy_out[..cols],
            &mut sell_out[..cols],
        );
    }

    let diff = unsafe {
        Vec::from_raw_parts(
            diff_guard.as_mut_ptr() as *mut f64,
            diff_guard.len(),
            diff_guard.capacity(),
        )
    };
    let dea = unsafe {
        Vec::from_raw_parts(
            dea_guard.as_mut_ptr() as *mut f64,
            dea_guard.len(),
            dea_guard.capacity(),
        )
    };
    let macd_histogram = unsafe {
        Vec::from_raw_parts(
            macd_guard.as_mut_ptr() as *mut f64,
            macd_guard.len(),
            macd_guard.capacity(),
        )
    };
    let line_convergence = unsafe {
        Vec::from_raw_parts(
            line_guard.as_mut_ptr() as *mut f64,
            line_guard.len(),
            line_guard.capacity(),
        )
    };
    let buy_signal = unsafe {
        Vec::from_raw_parts(
            buy_guard.as_mut_ptr() as *mut f64,
            buy_guard.len(),
            buy_guard.capacity(),
        )
    };
    let sell_signal = unsafe {
        Vec::from_raw_parts(
            sell_guard.as_mut_ptr() as *mut f64,
            sell_guard.len(),
            sell_guard.capacity(),
        )
    };

    Ok(MacdWaveSignalProBatchOutput {
        diff,
        dea,
        macd_histogram,
        line_convergence,
        buy_signal,
        sell_signal,
        combos,
        rows,
        cols,
    })
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub fn macd_wave_signal_pro_batch_inner_into(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &MacdWaveSignalProBatchRange,
    kernel: Kernel,
    parallel: bool,
    diff_out: &mut [f64],
    dea_out: &mut [f64],
    macd_histogram_out: &mut [f64],
    line_convergence_out: &mut [f64],
    buy_signal_out: &mut [f64],
    sell_signal_out: &mut [f64],
) -> Result<Vec<MacdWaveSignalProParams>, MacdWaveSignalProError> {
    let out = macd_wave_signal_pro_batch_inner(open, high, low, close, sweep, kernel, parallel)?;
    let total = out.rows * out.cols;
    if diff_out.len() != total
        || dea_out.len() != total
        || macd_histogram_out.len() != total
        || line_convergence_out.len() != total
        || buy_signal_out.len() != total
        || sell_signal_out.len() != total
    {
        return Err(MacdWaveSignalProError::OutputLengthMismatch {
            expected: total,
            diff_got: diff_out.len(),
            dea_got: dea_out.len(),
            macd_histogram_got: macd_histogram_out.len(),
            line_convergence_got: line_convergence_out.len(),
            buy_signal_got: buy_signal_out.len(),
            sell_signal_got: sell_signal_out.len(),
        });
    }

    diff_out.copy_from_slice(&out.diff);
    dea_out.copy_from_slice(&out.dea);
    macd_histogram_out.copy_from_slice(&out.macd_histogram);
    line_convergence_out.copy_from_slice(&out.line_convergence);
    buy_signal_out.copy_from_slice(&out.buy_signal);
    sell_signal_out.copy_from_slice(&out.sell_signal);
    Ok(out.combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "macd_wave_signal_pro")]
#[pyo3(signature = (open, high, low, close, kernel=None))]
pub fn macd_wave_signal_pro_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let open = open.as_slice()?;
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input =
        MacdWaveSignalProInput::from_slices(open, high, low, close, MacdWaveSignalProParams);
    let out = py
        .allow_threads(|| macd_wave_signal_pro_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        out.diff.into_pyarray(py),
        out.dea.into_pyarray(py),
        out.macd_histogram.into_pyarray(py),
        out.line_convergence.into_pyarray(py),
        out.buy_signal.into_pyarray(py),
        out.sell_signal.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyclass(name = "MacdWaveSignalProStream")]
pub struct MacdWaveSignalProStreamPy {
    stream: MacdWaveSignalProStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl MacdWaveSignalProStreamPy {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(Self {
            stream: MacdWaveSignalProStream::try_new(MacdWaveSignalProParams)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        })
    }

    fn update(
        &mut self,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
    ) -> Option<(f64, f64, f64, f64, f64, f64)> {
        self.stream.update(open, high, low, close).map(|point| {
            (
                point.diff,
                point.dea,
                point.macd_histogram,
                point.line_convergence,
                point.buy_signal,
                point.sell_signal,
            )
        })
    }

    fn reset(&mut self) {
        self.stream.reset();
    }

    #[getter]
    fn warmup_period(&self) -> usize {
        self.stream.get_warmup_period()
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "macd_wave_signal_pro_batch")]
#[pyo3(signature = (open, high, low, close, kernel=None))]
pub fn macd_wave_signal_pro_batch_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let open = open.as_slice()?;
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let kernel = validate_kernel(kernel, true)?;

    let rows = 1usize;
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let diff_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let dea_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let macd_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let line_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let buy_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let sell_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };

    let diff_slice = unsafe { diff_arr.as_slice_mut()? };
    let dea_slice = unsafe { dea_arr.as_slice_mut()? };
    let macd_slice = unsafe { macd_arr.as_slice_mut()? };
    let line_slice = unsafe { line_arr.as_slice_mut()? };
    let buy_slice = unsafe { buy_arr.as_slice_mut()? };
    let sell_slice = unsafe { sell_arr.as_slice_mut()? };

    py.allow_threads(|| {
        let kernel = match kernel {
            Kernel::Auto => detect_best_batch_kernel(),
            other => other,
        };
        macd_wave_signal_pro_batch_inner_into(
            open,
            high,
            low,
            close,
            &MacdWaveSignalProBatchRange,
            kernel.to_non_batch(),
            true,
            diff_slice,
            dea_slice,
            macd_slice,
            line_slice,
            buy_slice,
            sell_slice,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("diff", diff_arr.reshape((rows, cols))?)?;
    dict.set_item("dea", dea_arr.reshape((rows, cols))?)?;
    dict.set_item("macd_histogram", macd_arr.reshape((rows, cols))?)?;
    dict.set_item("line_convergence", line_arr.reshape((rows, cols))?)?;
    dict.set_item("buy_signal", buy_arr.reshape((rows, cols))?)?;
    dict.set_item("sell_signal", sell_arr.reshape((rows, cols))?)?;
    dict.set_item("params", Vec::<f64>::new().into_pyarray(py))?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_macd_wave_signal_pro_module(
    module: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(macd_wave_signal_pro_py, module)?)?;
    module.add_function(wrap_pyfunction!(macd_wave_signal_pro_batch_py, module)?)?;
    module.add_class::<MacdWaveSignalProStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct MacdWaveSignalProJsOutput {
    pub diff: Vec<f64>,
    pub dea: Vec<f64>,
    pub macd_histogram: Vec<f64>,
    pub line_convergence: Vec<f64>,
    pub buy_signal: Vec<f64>,
    pub sell_signal: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "macd_wave_signal_pro_js")]
pub fn macd_wave_signal_pro_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> Result<JsValue, JsValue> {
    let input =
        MacdWaveSignalProInput::from_slices(open, high, low, close, MacdWaveSignalProParams);
    let out = macd_wave_signal_pro(&input).map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&MacdWaveSignalProJsOutput {
        diff: out.diff,
        dea: out.dea,
        macd_histogram: out.macd_histogram,
        line_convergence: out.line_convergence,
        buy_signal: out.buy_signal,
        sell_signal: out.sell_signal,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn macd_wave_signal_pro_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn macd_wave_signal_pro_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
fn has_duplicate_ptrs(ptrs: &[usize]) -> bool {
    for i in 0..ptrs.len() {
        for j in (i + 1)..ptrs.len() {
            if ptrs[i] == ptrs[j] {
                return true;
            }
        }
    }
    false
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[allow(clippy::too_many_arguments)]
unsafe fn macd_wave_signal_pro_into_raw(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    diff_ptr: *mut f64,
    dea_ptr: *mut f64,
    macd_histogram_ptr: *mut f64,
    line_convergence_ptr: *mut f64,
    buy_signal_ptr: *mut f64,
    sell_signal_ptr: *mut f64,
    len: usize,
    kernel: Kernel,
) -> Result<(), JsValue> {
    let open = std::slice::from_raw_parts(open_ptr, len);
    let high = std::slice::from_raw_parts(high_ptr, len);
    let low = std::slice::from_raw_parts(low_ptr, len);
    let close = std::slice::from_raw_parts(close_ptr, len);
    let input =
        MacdWaveSignalProInput::from_slices(open, high, low, close, MacdWaveSignalProParams);

    let output_ptrs = [
        diff_ptr as usize,
        dea_ptr as usize,
        macd_histogram_ptr as usize,
        line_convergence_ptr as usize,
        buy_signal_ptr as usize,
        sell_signal_ptr as usize,
    ];
    let need_temp = output_ptrs.iter().any(|&ptr| {
        ptr == open_ptr as usize
            || ptr == high_ptr as usize
            || ptr == low_ptr as usize
            || ptr == close_ptr as usize
    }) || has_duplicate_ptrs(&output_ptrs);

    if need_temp {
        let mut diff = vec![0.0; len];
        let mut dea = vec![0.0; len];
        let mut macd_histogram = vec![0.0; len];
        let mut line_convergence = vec![0.0; len];
        let mut buy_signal = vec![0.0; len];
        let mut sell_signal = vec![0.0; len];
        macd_wave_signal_pro_into_slices(
            &mut diff,
            &mut dea,
            &mut macd_histogram,
            &mut line_convergence,
            &mut buy_signal,
            &mut sell_signal,
            &input,
            kernel,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        std::slice::from_raw_parts_mut(diff_ptr, len).copy_from_slice(&diff);
        std::slice::from_raw_parts_mut(dea_ptr, len).copy_from_slice(&dea);
        std::slice::from_raw_parts_mut(macd_histogram_ptr, len).copy_from_slice(&macd_histogram);
        std::slice::from_raw_parts_mut(line_convergence_ptr, len)
            .copy_from_slice(&line_convergence);
        std::slice::from_raw_parts_mut(buy_signal_ptr, len).copy_from_slice(&buy_signal);
        std::slice::from_raw_parts_mut(sell_signal_ptr, len).copy_from_slice(&sell_signal);
    } else {
        macd_wave_signal_pro_into_slices(
            std::slice::from_raw_parts_mut(diff_ptr, len),
            std::slice::from_raw_parts_mut(dea_ptr, len),
            std::slice::from_raw_parts_mut(macd_histogram_ptr, len),
            std::slice::from_raw_parts_mut(line_convergence_ptr, len),
            std::slice::from_raw_parts_mut(buy_signal_ptr, len),
            std::slice::from_raw_parts_mut(sell_signal_ptr, len),
            &input,
            kernel,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }

    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub struct MacdWaveSignalProContext {
    kernel: Kernel,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
impl MacdWaveSignalProContext {
    #[wasm_bindgen(constructor)]
    pub fn new() -> MacdWaveSignalProContext {
        MacdWaveSignalProContext {
            kernel: detect_best_kernel(),
        }
    }

    #[wasm_bindgen]
    #[allow(clippy::too_many_arguments)]
    pub fn update_into(
        &self,
        open_ptr: *const f64,
        high_ptr: *const f64,
        low_ptr: *const f64,
        close_ptr: *const f64,
        diff_ptr: *mut f64,
        dea_ptr: *mut f64,
        macd_histogram_ptr: *mut f64,
        line_convergence_ptr: *mut f64,
        buy_signal_ptr: *mut f64,
        sell_signal_ptr: *mut f64,
        len: usize,
    ) -> Result<(), JsValue> {
        if open_ptr.is_null()
            || high_ptr.is_null()
            || low_ptr.is_null()
            || close_ptr.is_null()
            || diff_ptr.is_null()
            || dea_ptr.is_null()
            || macd_histogram_ptr.is_null()
            || line_convergence_ptr.is_null()
            || buy_signal_ptr.is_null()
            || sell_signal_ptr.is_null()
        {
            return Err(JsValue::from_str("Null pointer provided"));
        }

        unsafe {
            macd_wave_signal_pro_into_raw(
                open_ptr,
                high_ptr,
                low_ptr,
                close_ptr,
                diff_ptr,
                dea_ptr,
                macd_histogram_ptr,
                line_convergence_ptr,
                buy_signal_ptr,
                sell_signal_ptr,
                len,
                self.kernel,
            )?;
        }
        Ok(())
    }

    pub fn get_warmup_period(&self) -> usize {
        LINE_LONG_PERIOD - 1
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn macd_wave_signal_pro_into(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    diff_ptr: *mut f64,
    dea_ptr: *mut f64,
    macd_histogram_ptr: *mut f64,
    line_convergence_ptr: *mut f64,
    buy_signal_ptr: *mut f64,
    sell_signal_ptr: *mut f64,
    len: usize,
) -> Result<(), JsValue> {
    if open_ptr.is_null()
        || high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || diff_ptr.is_null()
        || dea_ptr.is_null()
        || macd_histogram_ptr.is_null()
        || line_convergence_ptr.is_null()
        || buy_signal_ptr.is_null()
        || sell_signal_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        macd_wave_signal_pro_into_raw(
            open_ptr,
            high_ptr,
            low_ptr,
            close_ptr,
            diff_ptr,
            dea_ptr,
            macd_histogram_ptr,
            line_convergence_ptr,
            buy_signal_ptr,
            sell_signal_ptr,
            len,
            Kernel::Auto,
        )?;
    }

    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct MacdWaveSignalProBatchConfig {}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct MacdWaveSignalProBatchJsOutput {
    pub diff: Vec<f64>,
    pub dea: Vec<f64>,
    pub macd_histogram: Vec<f64>,
    pub line_convergence: Vec<f64>,
    pub buy_signal: Vec<f64>,
    pub sell_signal: Vec<f64>,
    pub combos: Vec<MacdWaveSignalProParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "macd_wave_signal_pro_batch_js")]
pub fn macd_wave_signal_pro_batch_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let _: MacdWaveSignalProBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let out = macd_wave_signal_pro_batch_with_kernel(
        open,
        high,
        low,
        close,
        &MacdWaveSignalProBatchRange,
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&MacdWaveSignalProBatchJsOutput {
        diff: out.diff,
        dea: out.dea,
        macd_histogram: out.macd_histogram,
        line_convergence: out.line_convergence,
        buy_signal: out.buy_signal,
        sell_signal: out.sell_signal,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn macd_wave_signal_pro_batch_into(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    diff_ptr: *mut f64,
    dea_ptr: *mut f64,
    macd_histogram_ptr: *mut f64,
    line_convergence_ptr: *mut f64,
    buy_signal_ptr: *mut f64,
    sell_signal_ptr: *mut f64,
    len: usize,
) -> Result<usize, JsValue> {
    if open_ptr.is_null()
        || high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || diff_ptr.is_null()
        || dea_ptr.is_null()
        || macd_histogram_ptr.is_null()
        || line_convergence_ptr.is_null()
        || buy_signal_ptr.is_null()
        || sell_signal_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        macd_wave_signal_pro_batch_inner_into(
            open,
            high,
            low,
            close,
            &MacdWaveSignalProBatchRange,
            Kernel::Auto,
            false,
            std::slice::from_raw_parts_mut(diff_ptr, len),
            std::slice::from_raw_parts_mut(dea_ptr, len),
            std::slice::from_raw_parts_mut(macd_histogram_ptr, len),
            std::slice::from_raw_parts_mut(line_convergence_ptr, len),
            std::slice::from_raw_parts_mut(buy_signal_ptr, len),
            std::slice::from_raw_parts_mut(sell_signal_ptr, len),
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }
    Ok(1)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn macd_wave_signal_pro_output_into_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = macd_wave_signal_pro_js(open, high, low, close)?;
    crate::write_wasm_object_f64_outputs("macd_wave_signal_pro_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn macd_wave_signal_pro_batch_output_into_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = macd_wave_signal_pro_batch_js(open, high, low, close, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "macd_wave_signal_pro_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    fn assert_series_eq(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (&a, &b) in actual.iter().zip(expected.iter()) {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert!(
                (a - b).abs() <= 1e-12,
                "series mismatch: expected {b}, got {a}"
            );
        }
    }

    fn sample_ohlc(length: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut open = Vec::with_capacity(length);
        let mut high = Vec::with_capacity(length);
        let mut low = Vec::with_capacity(length);
        let mut close = Vec::with_capacity(length);
        for i in 0..length {
            let x = i as f64;
            let o = 100.0 + x * 0.09 + (x * 0.07).sin() * 0.8;
            let c = o + (x * 0.11).cos() * 1.1;
            let h = o.max(c) + 0.6 + (x * 0.03).sin().abs() * 0.2;
            let l = o.min(c) - 0.6 - (x * 0.05).cos().abs() * 0.2;
            open.push(o);
            high.push(h);
            low.push(l);
            close.push(c);
        }
        (open, high, low, close)
    }

    #[test]
    fn macd_wave_signal_pro_output_contract() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close) = sample_ohlc(256);
        let input = MacdWaveSignalProInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            MacdWaveSignalProParams,
        );
        let out = macd_wave_signal_pro_with_kernel(&input, Kernel::Scalar)?;
        assert_eq!(out.diff.len(), close.len());
        assert_eq!(out.dea.len(), close.len());
        assert_eq!(out.macd_histogram.len(), close.len());
        assert_eq!(out.line_convergence.len(), close.len());
        assert_eq!(out.buy_signal.len(), close.len());
        assert_eq!(out.sell_signal.len(), close.len());
        for value in out.buy_signal.iter().copied().filter(|v| v.is_finite()) {
            assert!(value == 0.0 || value == 1.0);
        }
        for value in out.sell_signal.iter().copied().filter(|v| v.is_finite()) {
            assert!(value == 0.0 || value == 1.0);
        }
        Ok(())
    }

    #[test]
    fn macd_wave_signal_pro_into_matches_api() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close) = sample_ohlc(192);
        let input = MacdWaveSignalProInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            MacdWaveSignalProParams,
        );
        let out = macd_wave_signal_pro(&input)?;
        let len = close.len();
        let mut diff = vec![0.0; len];
        let mut dea = vec![0.0; len];
        let mut macd = vec![0.0; len];
        let mut line = vec![0.0; len];
        let mut buy = vec![0.0; len];
        let mut sell = vec![0.0; len];
        macd_wave_signal_pro_into(
            &input, &mut diff, &mut dea, &mut macd, &mut line, &mut buy, &mut sell,
        )?;
        assert_series_eq(&diff, &out.diff);
        assert_series_eq(&dea, &out.dea);
        assert_series_eq(&macd, &out.macd_histogram);
        assert_series_eq(&line, &out.line_convergence);
        assert_series_eq(&buy, &out.buy_signal);
        assert_series_eq(&sell, &out.sell_signal);
        Ok(())
    }

    #[test]
    fn macd_wave_signal_pro_stream_matches_batch_with_reset() -> Result<(), Box<dyn Error>> {
        let (mut open, mut high, mut low, mut close) = sample_ohlc(220);
        open[90] = f64::NAN;
        high[90] = f64::NAN;
        low[90] = f64::NAN;
        close[90] = f64::NAN;

        let input = MacdWaveSignalProInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            MacdWaveSignalProParams,
        );
        let batch = macd_wave_signal_pro(&input)?;
        let mut stream = MacdWaveSignalProStream::try_new(MacdWaveSignalProParams)?;

        let mut diff = Vec::with_capacity(close.len());
        let mut dea = Vec::with_capacity(close.len());
        let mut macd = Vec::with_capacity(close.len());
        let mut line = Vec::with_capacity(close.len());
        let mut buy = Vec::with_capacity(close.len());
        let mut sell = Vec::with_capacity(close.len());

        for i in 0..close.len() {
            if let Some(point) = stream.update(open[i], high[i], low[i], close[i]) {
                diff.push(point.diff);
                dea.push(point.dea);
                macd.push(point.macd_histogram);
                line.push(point.line_convergence);
                buy.push(point.buy_signal);
                sell.push(point.sell_signal);
            } else {
                diff.push(f64::NAN);
                dea.push(f64::NAN);
                macd.push(f64::NAN);
                line.push(f64::NAN);
                buy.push(f64::NAN);
                sell.push(f64::NAN);
            }
        }

        assert_series_eq(&diff, &batch.diff);
        assert_series_eq(&dea, &batch.dea);
        assert_series_eq(&macd, &batch.macd_histogram);
        assert_series_eq(&line, &batch.line_convergence);
        assert_series_eq(&buy, &batch.buy_signal);
        assert_series_eq(&sell, &batch.sell_signal);
        Ok(())
    }

    #[test]
    fn macd_wave_signal_pro_batch_matches_single() -> Result<(), Box<dyn Error>> {
        let (open, high, low, close) = sample_ohlc(160);
        let batch = macd_wave_signal_pro_batch_with_kernel(
            &open,
            &high,
            &low,
            &close,
            &MacdWaveSignalProBatchRange,
            Kernel::ScalarBatch,
        )?;
        let single = macd_wave_signal_pro(&MacdWaveSignalProInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            MacdWaveSignalProParams,
        ))?;
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, close.len());
        assert_series_eq(&batch.diff, &single.diff);
        assert_series_eq(&batch.dea, &single.dea);
        assert_series_eq(&batch.macd_histogram, &single.macd_histogram);
        assert_series_eq(&batch.line_convergence, &single.line_convergence);
        assert_series_eq(&batch.buy_signal, &single.buy_signal);
        assert_series_eq(&batch.sell_signal, &single.sell_signal);
        Ok(())
    }

    #[test]
    fn macd_wave_signal_pro_rejects_short_valid_history() {
        let (open, high, low, close) = sample_ohlc(32);
        let input = MacdWaveSignalProInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            MacdWaveSignalProParams,
        );
        let err = macd_wave_signal_pro(&input).unwrap_err();
        match err {
            MacdWaveSignalProError::NotEnoughValidData { needed, valid } => {
                assert_eq!(needed, 40);
                assert_eq!(valid, 32);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
