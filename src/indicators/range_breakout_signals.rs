#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray1};
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
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn range_breakout_signals_output_into_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    range_length: usize,
    confirmation_length: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = range_breakout_signals_js(
        open,
        high,
        low,
        close,
        volume,
        range_length,
        confirmation_length,
    )?;
    crate::write_wasm_object_f64_outputs("range_breakout_signals_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn range_breakout_signals_batch_unified_output_into_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = range_breakout_signals_batch_unified_js(open, high, low, close, volume, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "range_breakout_signals_batch_unified_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
use std::error::Error as StdError;
use std::mem::ManuallyDrop;
use thiserror::Error;

const DEFAULT_RANGE_LENGTH: usize = 20;
const DEFAULT_CONFIRMATION_LENGTH: usize = 5;
const ATR_LENGTH: usize = 14;
const ATR_MULTIPLIER: f64 = 1.2;
const VOLATILITY_THRESHOLD: f64 = 1.2;
const BULLISH_LOCATION_WEIGHT: f64 = 0.15;
const BEARISH_LOCATION_WEIGHT: f64 = 0.85;

#[inline(always)]
fn open_source(candles: &Candles) -> &[f64] {
    &candles.open
}

#[inline(always)]
fn high_source(candles: &Candles) -> &[f64] {
    &candles.high
}

#[inline(always)]
fn low_source(candles: &Candles) -> &[f64] {
    &candles.low
}

#[inline(always)]
fn close_source(candles: &Candles) -> &[f64] {
    &candles.close
}

#[inline(always)]
fn volume_source(candles: &Candles) -> &[f64] {
    &candles.volume
}

#[derive(Debug, Clone)]
pub enum RangeBreakoutSignalsData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct RangeBreakoutSignalsOutput {
    pub range_top: Vec<f64>,
    pub range_bottom: Vec<f64>,
    pub bullish: Vec<f64>,
    pub extra_bullish: Vec<f64>,
    pub bearish: Vec<f64>,
    pub extra_bearish: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct RangeBreakoutSignalsParams {
    pub range_length: Option<usize>,
    pub confirmation_length: Option<usize>,
}

impl Default for RangeBreakoutSignalsParams {
    fn default() -> Self {
        Self {
            range_length: Some(DEFAULT_RANGE_LENGTH),
            confirmation_length: Some(DEFAULT_CONFIRMATION_LENGTH),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RangeBreakoutSignalsInput<'a> {
    pub data: RangeBreakoutSignalsData<'a>,
    pub params: RangeBreakoutSignalsParams,
}

impl<'a> RangeBreakoutSignalsInput<'a> {
    #[inline(always)]
    pub fn from_candles(candles: &'a Candles, params: RangeBreakoutSignalsParams) -> Self {
        Self {
            data: RangeBreakoutSignalsData::Candles { candles },
            params,
        }
    }

    #[inline(always)]
    pub fn from_slices(
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
        params: RangeBreakoutSignalsParams,
    ) -> Self {
        Self {
            data: RangeBreakoutSignalsData::Slices {
                open,
                high,
                low,
                close,
                volume,
            },
            params,
        }
    }

    #[inline(always)]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, RangeBreakoutSignalsParams::default())
    }

    #[inline(always)]
    pub fn get_range_length(&self) -> usize {
        self.params.range_length.unwrap_or(DEFAULT_RANGE_LENGTH)
    }

    #[inline(always)]
    pub fn get_confirmation_length(&self) -> usize {
        self.params
            .confirmation_length
            .unwrap_or(DEFAULT_CONFIRMATION_LENGTH)
    }

    #[inline(always)]
    fn as_ohlcv(&self) -> (&'a [f64], &'a [f64], &'a [f64], &'a [f64], &'a [f64]) {
        match &self.data {
            RangeBreakoutSignalsData::Candles { candles } => (
                open_source(candles),
                high_source(candles),
                low_source(candles),
                close_source(candles),
                volume_source(candles),
            ),
            RangeBreakoutSignalsData::Slices {
                open,
                high,
                low,
                close,
                volume,
            } => (*open, *high, *low, *close, *volume),
        }
    }
}

impl<'a> AsRef<[f64]> for RangeBreakoutSignalsInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        self.as_ohlcv().3
    }
}

#[derive(Clone, Debug)]
pub struct RangeBreakoutSignalsBuilder {
    range_length: Option<usize>,
    confirmation_length: Option<usize>,
    kernel: Kernel,
}

impl Default for RangeBreakoutSignalsBuilder {
    fn default() -> Self {
        Self {
            range_length: None,
            confirmation_length: None,
            kernel: Kernel::Auto,
        }
    }
}

impl RangeBreakoutSignalsBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn range_length(mut self, value: usize) -> Self {
        self.range_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn confirmation_length(mut self, value: usize) -> Self {
        self.confirmation_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline(always)]
    fn params(self) -> RangeBreakoutSignalsParams {
        RangeBreakoutSignalsParams {
            range_length: self.range_length,
            confirmation_length: self.confirmation_length,
        }
    }

    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<RangeBreakoutSignalsOutput, RangeBreakoutSignalsError> {
        let kernel = self.kernel;
        let input = RangeBreakoutSignalsInput::from_candles(candles, self.params());
        range_breakout_signals_with_kernel(&input, kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Result<RangeBreakoutSignalsOutput, RangeBreakoutSignalsError> {
        let kernel = self.kernel;
        let input =
            RangeBreakoutSignalsInput::from_slices(open, high, low, close, volume, self.params());
        range_breakout_signals_with_kernel(&input, kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<RangeBreakoutSignalsStream, RangeBreakoutSignalsError> {
        RangeBreakoutSignalsStream::try_new(self.params())
    }
}

#[derive(Debug, Error)]
pub enum RangeBreakoutSignalsError {
    #[error("range_breakout_signals: input data slice is empty.")]
    EmptyInputData,
    #[error("range_breakout_signals: all values are NaN.")]
    AllValuesNaN,
    #[error(
        "range_breakout_signals: inconsistent slice lengths: open={open_len}, high={high_len}, low={low_len}, close={close_len}, volume={volume_len}"
    )]
    InconsistentSliceLengths {
        open_len: usize,
        high_len: usize,
        low_len: usize,
        close_len: usize,
        volume_len: usize,
    },
    #[error(
        "range_breakout_signals: invalid range_length: range_length = {range_length}, data length = {data_len}"
    )]
    InvalidRangeLength {
        range_length: usize,
        data_len: usize,
    },
    #[error(
        "range_breakout_signals: invalid confirmation_length: confirmation_length = {confirmation_length}"
    )]
    InvalidConfirmationLength { confirmation_length: usize },
    #[error("range_breakout_signals: not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("range_breakout_signals: output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("range_breakout_signals: invalid range for {axis}: start = {start}, end = {end}, step = {step}")]
    InvalidRange {
        axis: &'static str,
        start: String,
        end: String,
        step: String,
    },
    #[error("range_breakout_signals: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Clone, Copy, Debug)]
struct PreparedRangeBreakoutSignals<'a> {
    open: &'a [f64],
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    volume: &'a [f64],
    range_length: usize,
    confirmation_length: usize,
    warmup: usize,
}

#[derive(Clone, Copy, Debug)]
struct ActiveRange {
    top: f64,
    bottom: f64,
}

#[derive(Clone, Debug)]
struct MedianSmaWindow {
    len: usize,
    ring: Vec<f64>,
    sorted: Vec<f64>,
    head: usize,
    count: usize,
    sum: f64,
}

impl MedianSmaWindow {
    #[inline(always)]
    fn new(len: usize) -> Self {
        Self {
            len,
            ring: vec![0.0; len],
            sorted: Vec::with_capacity(len),
            head: 0,
            count: 0,
            sum: 0.0,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.sorted.clear();
        self.head = 0;
        self.count = 0;
        self.sum = 0.0;
    }

    #[inline(always)]
    fn push(&mut self, value: f64) -> Option<(f64, f64)> {
        if self.count == self.len {
            let old = self.ring[self.head];
            self.sum -= old;
            if let Some(index) = self.sorted.iter().position(|probe| *probe == old) {
                self.sorted.remove(index);
            }
            self.ring[self.head] = value;
            self.head += 1;
            if self.head == self.len {
                self.head = 0;
            }
        } else {
            self.ring[self.count] = value;
            self.count += 1;
            if self.count == self.len {
                self.head = 0;
            }
        }

        self.sum += value;
        let index = self.sorted.partition_point(|probe| *probe <= value);
        self.sorted.insert(index, value);

        if self.count < self.len {
            return None;
        }

        let median = if self.len & 1 == 1 {
            self.sorted[self.len >> 1]
        } else {
            let upper = self.len >> 1;
            (self.sorted[upper - 1] + self.sorted[upper]) * 0.5
        };
        Some((median, self.sum / self.len as f64))
    }
}

#[derive(Clone, Debug)]
struct AtrState {
    len: usize,
    count: usize,
    sum: f64,
    value: f64,
    prev_close: f64,
    have_prev_close: bool,
}

impl AtrState {
    #[inline(always)]
    fn new(len: usize) -> Self {
        Self {
            len,
            count: 0,
            sum: 0.0,
            value: f64::NAN,
            prev_close: f64::NAN,
            have_prev_close: false,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.count = 0;
        self.sum = 0.0;
        self.value = f64::NAN;
        self.prev_close = f64::NAN;
        self.have_prev_close = false;
    }

    #[inline(always)]
    fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        let prev_close = if self.have_prev_close {
            self.prev_close
        } else {
            close
        };
        let tr = (high - low)
            .max((high - prev_close).abs())
            .max((low - prev_close).abs());
        self.prev_close = close;
        self.have_prev_close = true;

        if self.count < self.len {
            self.count += 1;
            self.sum += tr;
            if self.count < self.len {
                return None;
            }
            self.value = self.sum / self.len as f64;
            return Some(self.value);
        }

        self.value = ((self.value * (self.len - 1) as f64) + tr) / self.len as f64;
        Some(self.value)
    }
}

#[derive(Clone, Debug)]
struct VolumeWindow {
    len: usize,
    up_ring: Vec<f64>,
    down_ring: Vec<f64>,
    head: usize,
    count: usize,
    up_sum: f64,
    down_sum: f64,
}

impl VolumeWindow {
    #[inline(always)]
    fn new(len: usize) -> Self {
        Self {
            len,
            up_ring: vec![0.0; len],
            down_ring: vec![0.0; len],
            head: 0,
            count: 0,
            up_sum: 0.0,
            down_sum: 0.0,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.head = 0;
        self.count = 0;
        self.up_sum = 0.0;
        self.down_sum = 0.0;
    }

    #[inline(always)]
    fn push(&mut self, up: f64, down: f64) {
        if self.count == self.len {
            self.up_sum -= self.up_ring[self.head];
            self.down_sum -= self.down_ring[self.head];
            self.up_ring[self.head] = up;
            self.down_ring[self.head] = down;
            self.head += 1;
            if self.head == self.len {
                self.head = 0;
            }
        } else {
            self.up_ring[self.count] = up;
            self.down_ring[self.count] = down;
            self.count += 1;
            if self.count == self.len {
                self.head = 0;
            }
        }
        self.up_sum += up;
        self.down_sum += down;
    }

    #[inline(always)]
    fn is_full(&self) -> bool {
        self.count == self.len
    }
}

#[derive(Clone, Debug)]
struct BoolWindow {
    ring: Vec<bool>,
    head: usize,
    count: usize,
}

impl BoolWindow {
    #[inline(always)]
    fn new(len: usize) -> Self {
        Self {
            ring: vec![false; len],
            head: 0,
            count: 0,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.head = 0;
        self.count = 0;
    }

    #[inline(always)]
    fn push(&mut self, value: bool) {
        if self.count == self.ring.len() {
            self.ring[self.head] = value;
            self.head += 1;
            if self.head == self.ring.len() {
                self.head = 0;
            }
        } else {
            self.ring[self.count] = value;
            self.count += 1;
            if self.count == self.ring.len() {
                self.head = 0;
            }
        }
    }

    #[inline(always)]
    fn get_ago(&self, ago: usize) -> Option<bool> {
        if ago >= self.count {
            return None;
        }
        if self.count < self.ring.len() {
            return Some(self.ring[self.count - 1 - ago]);
        }
        let len = self.ring.len();
        let latest = if self.head == 0 {
            len - 1
        } else {
            self.head - 1
        };
        let index = if latest >= ago {
            latest - ago
        } else {
            latest + len - ago
        };
        Some(self.ring[index])
    }

    #[inline(always)]
    fn is_full(&self) -> bool {
        self.count == self.ring.len()
    }
}

#[derive(Clone, Debug)]
struct MedianSmaWindow20 {
    ring: [f64; DEFAULT_RANGE_LENGTH],
    sorted: [f64; DEFAULT_RANGE_LENGTH],
    head: usize,
    count: usize,
    sum: f64,
}

impl MedianSmaWindow20 {
    #[inline(always)]
    fn new() -> Self {
        Self {
            ring: [0.0; DEFAULT_RANGE_LENGTH],
            sorted: [0.0; DEFAULT_RANGE_LENGTH],
            head: 0,
            count: 0,
            sum: 0.0,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.head = 0;
        self.count = 0;
        self.sum = 0.0;
    }

    #[inline(always)]
    fn push(&mut self, value: f64) -> Option<(f64, f64)> {
        let sorted_len;
        if self.count == DEFAULT_RANGE_LENGTH {
            let old = self.ring[self.head];
            self.sum -= old;
            let mut index = 0usize;
            while self.sorted[index] != old {
                index += 1;
            }
            while index + 1 < DEFAULT_RANGE_LENGTH {
                self.sorted[index] = self.sorted[index + 1];
                index += 1;
            }
            self.ring[self.head] = value;
            self.head += 1;
            if self.head == DEFAULT_RANGE_LENGTH {
                self.head = 0;
            }
            sorted_len = DEFAULT_RANGE_LENGTH - 1;
        } else {
            self.ring[self.count] = value;
            self.count += 1;
            if self.count == DEFAULT_RANGE_LENGTH {
                self.head = 0;
            }
            sorted_len = self.count - 1;
        }

        self.sum += value;
        let mut index = 0usize;
        while index < sorted_len && self.sorted[index] <= value {
            index += 1;
        }
        let mut j = sorted_len;
        while j > index {
            self.sorted[j] = self.sorted[j - 1];
            j -= 1;
        }
        self.sorted[index] = value;

        if self.count < DEFAULT_RANGE_LENGTH {
            return None;
        }

        Some((
            (self.sorted[9] + self.sorted[10]) * 0.5,
            self.sum / DEFAULT_RANGE_LENGTH as f64,
        ))
    }
}

#[derive(Clone, Debug)]
struct VolumeWindow6 {
    up_ring: [f64; DEFAULT_CONFIRMATION_LENGTH + 1],
    down_ring: [f64; DEFAULT_CONFIRMATION_LENGTH + 1],
    head: usize,
    count: usize,
    up_sum: f64,
    down_sum: f64,
}

impl VolumeWindow6 {
    #[inline(always)]
    fn new() -> Self {
        Self {
            up_ring: [0.0; DEFAULT_CONFIRMATION_LENGTH + 1],
            down_ring: [0.0; DEFAULT_CONFIRMATION_LENGTH + 1],
            head: 0,
            count: 0,
            up_sum: 0.0,
            down_sum: 0.0,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.head = 0;
        self.count = 0;
        self.up_sum = 0.0;
        self.down_sum = 0.0;
    }

    #[inline(always)]
    fn push(&mut self, up: f64, down: f64) {
        if self.count == DEFAULT_CONFIRMATION_LENGTH + 1 {
            self.up_sum -= self.up_ring[self.head];
            self.down_sum -= self.down_ring[self.head];
            self.up_ring[self.head] = up;
            self.down_ring[self.head] = down;
            self.head += 1;
            if self.head == DEFAULT_CONFIRMATION_LENGTH + 1 {
                self.head = 0;
            }
        } else {
            self.up_ring[self.count] = up;
            self.down_ring[self.count] = down;
            self.count += 1;
            if self.count == DEFAULT_CONFIRMATION_LENGTH + 1 {
                self.head = 0;
            }
        }
        self.up_sum += up;
        self.down_sum += down;
    }

    #[inline(always)]
    fn is_full(&self) -> bool {
        self.count == DEFAULT_CONFIRMATION_LENGTH + 1
    }
}

#[derive(Clone, Debug)]
struct BoolWindow6 {
    ring: [bool; DEFAULT_CONFIRMATION_LENGTH + 1],
    head: usize,
    count: usize,
}

impl BoolWindow6 {
    #[inline(always)]
    fn new() -> Self {
        Self {
            ring: [false; DEFAULT_CONFIRMATION_LENGTH + 1],
            head: 0,
            count: 0,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.head = 0;
        self.count = 0;
    }

    #[inline(always)]
    fn push(&mut self, value: bool) {
        if self.count == DEFAULT_CONFIRMATION_LENGTH + 1 {
            self.ring[self.head] = value;
            self.head += 1;
            if self.head == DEFAULT_CONFIRMATION_LENGTH + 1 {
                self.head = 0;
            }
        } else {
            self.ring[self.count] = value;
            self.count += 1;
            if self.count == DEFAULT_CONFIRMATION_LENGTH + 1 {
                self.head = 0;
            }
        }
    }

    #[inline(always)]
    fn get_ago(&self, ago: usize) -> Option<bool> {
        if ago >= self.count {
            return None;
        }
        if self.count < DEFAULT_CONFIRMATION_LENGTH + 1 {
            return Some(self.ring[self.count - 1 - ago]);
        }
        let latest = if self.head == 0 {
            DEFAULT_CONFIRMATION_LENGTH
        } else {
            self.head - 1
        };
        let index = if latest >= ago {
            latest - ago
        } else {
            latest + DEFAULT_CONFIRMATION_LENGTH + 1 - ago
        };
        Some(self.ring[index])
    }

    #[inline(always)]
    fn is_full(&self) -> bool {
        self.count == DEFAULT_CONFIRMATION_LENGTH + 1
    }
}

#[derive(Clone, Debug)]
struct RangeBreakoutSignalsDefaultState {
    dist_window: MedianSmaWindow20,
    atr_state: AtrState,
    volume_window: VolumeWindow6,
    under_window: BoolWindow6,
    prev_volatility: f64,
    active_range: Option<ActiveRange>,
}

impl RangeBreakoutSignalsDefaultState {
    #[inline(always)]
    fn new() -> Self {
        Self {
            dist_window: MedianSmaWindow20::new(),
            atr_state: AtrState::new(ATR_LENGTH),
            volume_window: VolumeWindow6::new(),
            under_window: BoolWindow6::new(),
            prev_volatility: f64::NAN,
            active_range: None,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.dist_window.reset();
        self.atr_state.reset();
        self.volume_window.reset();
        self.under_window.reset();
        self.prev_volatility = f64::NAN;
        self.active_range = None;
    }

    #[inline(always)]
    fn update(
        &mut self,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Option<(f64, f64, f64, f64, f64, f64)> {
        if !open.is_finite()
            || !high.is_finite()
            || !low.is_finite()
            || !close.is_finite()
            || !volume.is_finite()
        {
            self.reset();
            return None;
        }

        let previous_volatility = self.prev_volatility;
        let atr = self.atr_state.update(high, low, close);
        let volatility = self
            .dist_window
            .push((close - open).abs())
            .and_then(|(median, mean)| (median > 0.0).then_some(mean / median));

        let current_isunder = volatility.is_some_and(|value| value < VOLATILITY_THRESHOLD);
        let (up_volume, down_volume) = RangeBreakoutSignalsState::split_volume(open, close, volume);
        self.volume_window.push(up_volume, down_volume);
        self.under_window.push(current_isunder);

        let ready = volatility.is_some()
            && atr.is_some()
            && previous_volatility.is_finite()
            && self.volume_window.is_full()
            && self.under_window.is_full();

        if ready {
            let under_ago = self
                .under_window
                .get_ago(DEFAULT_CONFIRMATION_LENGTH)
                .unwrap_or(false);
            let current_volatility = volatility.unwrap_or(f64::NAN);
            let crossed_under = previous_volatility >= VOLATILITY_THRESHOLD
                && current_volatility < VOLATILITY_THRESHOLD;
            if self.active_range.is_none() && crossed_under && current_isunder && under_ago {
                let offset = atr.unwrap_or(f64::NAN) * ATR_MULTIPLIER;
                self.active_range = Some(ActiveRange {
                    top: close + offset,
                    bottom: close - offset,
                });
            }
        }

        let mut range_top = f64::NAN;
        let mut range_bottom = f64::NAN;
        let mut bullish = f64::NAN;
        let mut extra_bullish = f64::NAN;
        let mut bearish = f64::NAN;
        let mut extra_bearish = f64::NAN;

        if let Some(range) = self.active_range {
            range_top = range.top;
            range_bottom = range.bottom;

            if close > range.top || close < range.bottom {
                let bullish_break = close > range.top;
                let location = RangeBreakoutSignalsState::location(range, bullish_break);
                let bullish_volume = self.volume_window.up_sum > self.volume_window.down_sum;

                if bullish_break {
                    bullish = location;
                    if bullish_volume {
                        extra_bullish = location;
                    }
                } else {
                    bearish = location;
                    if !bullish_volume {
                        extra_bearish = location;
                    }
                }

                self.active_range = None;
            }
        }

        self.prev_volatility = volatility.unwrap_or(f64::NAN);

        ready.then_some((
            range_top,
            range_bottom,
            bullish,
            extra_bullish,
            bearish,
            extra_bearish,
        ))
    }
}

#[derive(Clone, Debug)]
struct RangeBreakoutSignalsState {
    confirmation_length: usize,
    dist_window: MedianSmaWindow,
    atr_state: AtrState,
    volume_window: VolumeWindow,
    under_window: BoolWindow,
    prev_volatility: f64,
    active_range: Option<ActiveRange>,
}

impl RangeBreakoutSignalsState {
    #[inline(always)]
    fn new(range_length: usize, confirmation_length: usize) -> Self {
        Self {
            confirmation_length,
            dist_window: MedianSmaWindow::new(range_length),
            atr_state: AtrState::new(ATR_LENGTH),
            volume_window: VolumeWindow::new(confirmation_length + 1),
            under_window: BoolWindow::new(confirmation_length + 1),
            prev_volatility: f64::NAN,
            active_range: None,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.dist_window.reset();
        self.atr_state.reset();
        self.volume_window.reset();
        self.under_window.reset();
        self.prev_volatility = f64::NAN;
        self.active_range = None;
    }

    #[inline(always)]
    fn split_volume(open: f64, close: f64, volume: f64) -> (f64, f64) {
        if close > open {
            (volume, 0.0)
        } else if close < open {
            (0.0, volume)
        } else {
            let half = volume * 0.5;
            (half, half)
        }
    }

    #[inline(always)]
    fn location(range: ActiveRange, bullish: bool) -> f64 {
        let span = range.top - range.bottom;
        let weight = if bullish {
            BULLISH_LOCATION_WEIGHT
        } else {
            BEARISH_LOCATION_WEIGHT
        };
        range.bottom + span * weight
    }

    #[inline(always)]
    fn update(
        &mut self,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Option<(f64, f64, f64, f64, f64, f64)> {
        if !open.is_finite()
            || !high.is_finite()
            || !low.is_finite()
            || !close.is_finite()
            || !volume.is_finite()
        {
            self.reset();
            return None;
        }

        self.update_finite(open, high, low, close, volume)
    }

    #[inline(always)]
    fn update_finite(
        &mut self,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Option<(f64, f64, f64, f64, f64, f64)> {
        let previous_volatility = self.prev_volatility;
        let atr = self.atr_state.update(high, low, close);
        let volatility = self
            .dist_window
            .push((close - open).abs())
            .and_then(|(median, mean)| (median > 0.0).then_some(mean / median));

        let current_isunder = volatility.is_some_and(|value| value < VOLATILITY_THRESHOLD);
        let (up_volume, down_volume) = Self::split_volume(open, close, volume);
        self.volume_window.push(up_volume, down_volume);
        self.under_window.push(current_isunder);

        let ready = volatility.is_some()
            && atr.is_some()
            && previous_volatility.is_finite()
            && self.volume_window.is_full()
            && self.under_window.is_full();

        if ready {
            let under_ago = self
                .under_window
                .get_ago(self.confirmation_length)
                .unwrap_or(false);
            let current_volatility = volatility.unwrap_or(f64::NAN);
            let crossed_under = previous_volatility >= VOLATILITY_THRESHOLD
                && current_volatility < VOLATILITY_THRESHOLD;
            if self.active_range.is_none() && crossed_under && current_isunder && under_ago {
                let offset = atr.unwrap_or(f64::NAN) * ATR_MULTIPLIER;
                self.active_range = Some(ActiveRange {
                    top: close + offset,
                    bottom: close - offset,
                });
            }
        }

        let mut range_top = f64::NAN;
        let mut range_bottom = f64::NAN;
        let mut bullish = f64::NAN;
        let mut extra_bullish = f64::NAN;
        let mut bearish = f64::NAN;
        let mut extra_bearish = f64::NAN;

        if let Some(range) = self.active_range {
            range_top = range.top;
            range_bottom = range.bottom;

            if close > range.top || close < range.bottom {
                let bullish_break = close > range.top;
                let location = Self::location(range, bullish_break);
                let bullish_volume = self.volume_window.up_sum > self.volume_window.down_sum;

                if bullish_break {
                    bullish = location;
                    if bullish_volume {
                        extra_bullish = location;
                    }
                } else {
                    bearish = location;
                    if !bullish_volume {
                        extra_bearish = location;
                    }
                }

                self.active_range = None;
            }
        }

        self.prev_volatility = volatility.unwrap_or(f64::NAN);

        ready.then_some((
            range_top,
            range_bottom,
            bullish,
            extra_bullish,
            bearish,
            extra_bearish,
        ))
    }
}

#[derive(Clone, Debug)]
pub struct RangeBreakoutSignalsStream {
    params: RangeBreakoutSignalsParams,
    state: RangeBreakoutSignalsState,
}

impl RangeBreakoutSignalsStream {
    #[inline(always)]
    pub fn try_new(params: RangeBreakoutSignalsParams) -> Result<Self, RangeBreakoutSignalsError> {
        let range_length = params.range_length.unwrap_or(DEFAULT_RANGE_LENGTH);
        let confirmation_length = params
            .confirmation_length
            .unwrap_or(DEFAULT_CONFIRMATION_LENGTH);
        validate_params(range_length, confirmation_length, usize::MAX)?;
        Ok(Self {
            params,
            state: RangeBreakoutSignalsState::new(range_length, confirmation_length),
        })
    }

    #[inline(always)]
    pub fn update(
        &mut self,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Option<(f64, f64, f64, f64, f64, f64)> {
        self.state.update(open, high, low, close, volume)
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.state = RangeBreakoutSignalsState::new(
            self.params.range_length.unwrap_or(DEFAULT_RANGE_LENGTH),
            self.params
                .confirmation_length
                .unwrap_or(DEFAULT_CONFIRMATION_LENGTH),
        );
    }
}

#[inline(always)]
fn required_valid_bars(range_length: usize, confirmation_length: usize) -> usize {
    (range_length + 1)
        .max(ATR_LENGTH)
        .max(confirmation_length + 1)
}

#[inline(always)]
fn validate_params(
    range_length: usize,
    confirmation_length: usize,
    data_len: usize,
) -> Result<(), RangeBreakoutSignalsError> {
    if range_length == 0 {
        return Err(RangeBreakoutSignalsError::InvalidRangeLength {
            range_length,
            data_len,
        });
    }
    if confirmation_length == 0 {
        return Err(RangeBreakoutSignalsError::InvalidConfirmationLength {
            confirmation_length,
        });
    }
    if data_len != usize::MAX && range_length > data_len {
        return Err(RangeBreakoutSignalsError::InvalidRangeLength {
            range_length,
            data_len,
        });
    }
    Ok(())
}

fn analyze_valid_segments(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
) -> Result<(usize, usize), RangeBreakoutSignalsError> {
    if open.is_empty() {
        return Err(RangeBreakoutSignalsError::EmptyInputData);
    }
    if open.len() != high.len()
        || open.len() != low.len()
        || open.len() != close.len()
        || open.len() != volume.len()
    {
        return Err(RangeBreakoutSignalsError::InconsistentSliceLengths {
            open_len: open.len(),
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
            volume_len: volume.len(),
        });
    }

    let mut run = 0usize;
    let mut max_run = 0usize;
    let mut valid = 0usize;
    for i in 0..open.len() {
        if open[i].is_finite()
            && high[i].is_finite()
            && low[i].is_finite()
            && close[i].is_finite()
            && volume[i].is_finite()
        {
            valid += 1;
            run += 1;
            max_run = max_run.max(run);
        } else {
            run = 0;
        }
    }

    if valid == 0 {
        return Err(RangeBreakoutSignalsError::AllValuesNaN);
    }

    Ok((valid, max_run))
}

fn prepare_input<'a>(
    input: &'a RangeBreakoutSignalsInput<'a>,
    kernel: Kernel,
) -> Result<PreparedRangeBreakoutSignals<'a>, RangeBreakoutSignalsError> {
    if matches!(kernel, Kernel::Auto) {
        let _ = detect_best_kernel();
    }

    let (open, high, low, close, volume) = input.as_ohlcv();
    let range_length = input.get_range_length();
    let confirmation_length = input.get_confirmation_length();
    validate_params(range_length, confirmation_length, close.len())?;

    let (_, max_run) = analyze_valid_segments(open, high, low, close, volume)?;
    let needed = required_valid_bars(range_length, confirmation_length);
    if max_run < needed {
        return Err(RangeBreakoutSignalsError::NotEnoughValidData {
            needed,
            valid: max_run,
        });
    }

    Ok(PreparedRangeBreakoutSignals {
        open,
        high,
        low,
        close,
        volume,
        range_length,
        confirmation_length,
        warmup: needed - 1,
    })
}

fn compute_row(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    range_length: usize,
    confirmation_length: usize,
    range_top_out: &mut [f64],
    range_bottom_out: &mut [f64],
    bullish_out: &mut [f64],
    extra_bullish_out: &mut [f64],
    bearish_out: &mut [f64],
    extra_bearish_out: &mut [f64],
) -> Result<(), RangeBreakoutSignalsError> {
    let expected = close.len();
    for out in [
        &mut *range_top_out,
        &mut *range_bottom_out,
        &mut *bullish_out,
        &mut *extra_bullish_out,
        &mut *bearish_out,
        &mut *extra_bearish_out,
    ] {
        if out.len() != expected {
            return Err(RangeBreakoutSignalsError::OutputLengthMismatch {
                expected,
                got: out.len(),
            });
        }
    }

    if range_length == DEFAULT_RANGE_LENGTH && confirmation_length == DEFAULT_CONFIRMATION_LENGTH {
        let mut state = RangeBreakoutSignalsDefaultState::new();
        for i in 0..expected {
            if let Some((rt, rb, b, eb, br, ebr)) =
                state.update(open[i], high[i], low[i], close[i], volume[i])
            {
                range_top_out[i] = rt;
                range_bottom_out[i] = rb;
                bullish_out[i] = b;
                extra_bullish_out[i] = eb;
                bearish_out[i] = br;
                extra_bearish_out[i] = ebr;
            } else {
                range_top_out[i] = f64::NAN;
                range_bottom_out[i] = f64::NAN;
                bullish_out[i] = f64::NAN;
                extra_bullish_out[i] = f64::NAN;
                bearish_out[i] = f64::NAN;
                extra_bearish_out[i] = f64::NAN;
            }
        }
        return Ok(());
    }

    let mut state = RangeBreakoutSignalsState::new(range_length, confirmation_length);
    for i in 0..expected {
        if let Some((rt, rb, b, eb, br, ebr)) =
            state.update(open[i], high[i], low[i], close[i], volume[i])
        {
            range_top_out[i] = rt;
            range_bottom_out[i] = rb;
            bullish_out[i] = b;
            extra_bullish_out[i] = eb;
            bearish_out[i] = br;
            extra_bearish_out[i] = ebr;
        } else {
            range_top_out[i] = f64::NAN;
            range_bottom_out[i] = f64::NAN;
            bullish_out[i] = f64::NAN;
            extra_bullish_out[i] = f64::NAN;
            bearish_out[i] = f64::NAN;
            extra_bearish_out[i] = f64::NAN;
        }
    }

    Ok(())
}

#[inline]
pub fn range_breakout_signals(
    input: &RangeBreakoutSignalsInput,
) -> Result<RangeBreakoutSignalsOutput, RangeBreakoutSignalsError> {
    range_breakout_signals_with_kernel(input, Kernel::Auto)
}

pub fn range_breakout_signals_with_kernel(
    input: &RangeBreakoutSignalsInput,
    kernel: Kernel,
) -> Result<RangeBreakoutSignalsOutput, RangeBreakoutSignalsError> {
    let prepared = prepare_input(input, kernel)?;
    let len = prepared.close.len();
    let warmup = prepared.warmup;
    let mut range_top = alloc_with_nan_prefix(len, warmup);
    let mut range_bottom = alloc_with_nan_prefix(len, warmup);
    let mut bullish = alloc_with_nan_prefix(len, warmup);
    let mut extra_bullish = alloc_with_nan_prefix(len, warmup);
    let mut bearish = alloc_with_nan_prefix(len, warmup);
    let mut extra_bearish = alloc_with_nan_prefix(len, warmup);

    compute_row(
        prepared.open,
        prepared.high,
        prepared.low,
        prepared.close,
        prepared.volume,
        prepared.range_length,
        prepared.confirmation_length,
        &mut range_top,
        &mut range_bottom,
        &mut bullish,
        &mut extra_bullish,
        &mut bearish,
        &mut extra_bearish,
    )?;

    Ok(RangeBreakoutSignalsOutput {
        range_top,
        range_bottom,
        bullish,
        extra_bullish,
        bearish,
        extra_bearish,
    })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn range_breakout_signals_into(
    range_top_out: &mut [f64],
    range_bottom_out: &mut [f64],
    bullish_out: &mut [f64],
    extra_bullish_out: &mut [f64],
    bearish_out: &mut [f64],
    extra_bearish_out: &mut [f64],
    input: &RangeBreakoutSignalsInput,
) -> Result<(), RangeBreakoutSignalsError> {
    range_breakout_signals_into_slice(
        range_top_out,
        range_bottom_out,
        bullish_out,
        extra_bullish_out,
        bearish_out,
        extra_bearish_out,
        input,
        Kernel::Auto,
    )
}

pub fn range_breakout_signals_into_slice(
    range_top_out: &mut [f64],
    range_bottom_out: &mut [f64],
    bullish_out: &mut [f64],
    extra_bullish_out: &mut [f64],
    bearish_out: &mut [f64],
    extra_bearish_out: &mut [f64],
    input: &RangeBreakoutSignalsInput,
    kernel: Kernel,
) -> Result<(), RangeBreakoutSignalsError> {
    let prepared = prepare_input(input, kernel)?;
    compute_row(
        prepared.open,
        prepared.high,
        prepared.low,
        prepared.close,
        prepared.volume,
        prepared.range_length,
        prepared.confirmation_length,
        range_top_out,
        range_bottom_out,
        bullish_out,
        extra_bullish_out,
        bearish_out,
        extra_bearish_out,
    )
}

#[derive(Clone, Debug)]
pub struct RangeBreakoutSignalsBatchRange {
    pub range_length: (usize, usize, usize),
    pub confirmation_length: (usize, usize, usize),
}

impl Default for RangeBreakoutSignalsBatchRange {
    fn default() -> Self {
        Self {
            range_length: (DEFAULT_RANGE_LENGTH, DEFAULT_RANGE_LENGTH, 0),
            confirmation_length: (DEFAULT_CONFIRMATION_LENGTH, DEFAULT_CONFIRMATION_LENGTH, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct RangeBreakoutSignalsBatchBuilder {
    range: RangeBreakoutSignalsBatchRange,
    kernel: Kernel,
}

impl RangeBreakoutSignalsBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline(always)]
    pub fn range_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.range_length = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn confirmation_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.confirmation_length = (start, end, step);
        self
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct RangeBreakoutSignalsBatchOutput {
    pub range_top: Vec<f64>,
    pub range_bottom: Vec<f64>,
    pub bullish: Vec<f64>,
    pub extra_bullish: Vec<f64>,
    pub bearish: Vec<f64>,
    pub extra_bearish: Vec<f64>,
    pub combos: Vec<RangeBreakoutSignalsParams>,
    pub rows: usize,
    pub cols: usize,
}

#[inline(always)]
fn axis_usize(
    axis: &'static str,
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, RangeBreakoutSignalsError> {
    if start == end || step == 0 {
        return Ok(vec![start]);
    }
    let mut out = Vec::new();
    if start < end {
        let mut value = start;
        while value <= end {
            out.push(value);
            value =
                value
                    .checked_add(step)
                    .ok_or_else(|| RangeBreakoutSignalsError::InvalidRange {
                        axis,
                        start: start.to_string(),
                        end: end.to_string(),
                        step: step.to_string(),
                    })?;
        }
    } else {
        let mut value = start;
        while value >= end {
            out.push(value);
            value =
                value
                    .checked_sub(step)
                    .ok_or_else(|| RangeBreakoutSignalsError::InvalidRange {
                        axis,
                        start: start.to_string(),
                        end: end.to_string(),
                        step: step.to_string(),
                    })?;
        }
    }
    if out.is_empty() || *out.last().unwrap() != end {
        return Err(RangeBreakoutSignalsError::InvalidRange {
            axis,
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

pub fn expand_grid_range_breakout_signals(
    sweep: &RangeBreakoutSignalsBatchRange,
) -> Result<Vec<RangeBreakoutSignalsParams>, RangeBreakoutSignalsError> {
    let range_lengths = axis_usize("range_length", sweep.range_length)?;
    let confirmation_lengths = axis_usize("confirmation_length", sweep.confirmation_length)?;
    let mut out = Vec::with_capacity(range_lengths.len() * confirmation_lengths.len());
    for &range_length in &range_lengths {
        for &confirmation_length in &confirmation_lengths {
            out.push(RangeBreakoutSignalsParams {
                range_length: Some(range_length),
                confirmation_length: Some(confirmation_length),
            });
        }
    }
    Ok(out)
}

fn batch_inner_into(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &RangeBreakoutSignalsBatchRange,
    parallel: bool,
    range_top_out: &mut [f64],
    range_bottom_out: &mut [f64],
    bullish_out: &mut [f64],
    extra_bullish_out: &mut [f64],
    bearish_out: &mut [f64],
    extra_bearish_out: &mut [f64],
) -> Result<Vec<RangeBreakoutSignalsParams>, RangeBreakoutSignalsError> {
    let (_, max_run) = analyze_valid_segments(open, high, low, close, volume)?;
    let combos = expand_grid_range_breakout_signals(sweep)?;
    let rows = combos.len();
    let cols = close.len();
    let expected = rows * cols;

    for out in [
        &mut *range_top_out,
        &mut *range_bottom_out,
        &mut *bullish_out,
        &mut *extra_bullish_out,
        &mut *bearish_out,
        &mut *extra_bearish_out,
    ] {
        if out.len() != expected {
            return Err(RangeBreakoutSignalsError::OutputLengthMismatch {
                expected,
                got: out.len(),
            });
        }
    }

    for params in &combos {
        let needed = required_valid_bars(
            params.range_length.unwrap_or(DEFAULT_RANGE_LENGTH),
            params
                .confirmation_length
                .unwrap_or(DEFAULT_CONFIRMATION_LENGTH),
        );
        if max_run < needed {
            return Err(RangeBreakoutSignalsError::NotEnoughValidData {
                needed,
                valid: max_run,
            });
        }
    }

    let do_row = |row: usize,
                  range_top_row: &mut [f64],
                  range_bottom_row: &mut [f64],
                  bullish_row: &mut [f64],
                  extra_bullish_row: &mut [f64],
                  bearish_row: &mut [f64],
                  extra_bearish_row: &mut [f64]| {
        let params = &combos[row];
        compute_row(
            open,
            high,
            low,
            close,
            volume,
            params.range_length.unwrap_or(DEFAULT_RANGE_LENGTH),
            params
                .confirmation_length
                .unwrap_or(DEFAULT_CONFIRMATION_LENGTH),
            range_top_row,
            range_bottom_row,
            bullish_row,
            extra_bullish_row,
            bearish_row,
            extra_bearish_row,
        )
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            range_top_out
                .par_chunks_mut(cols)
                .zip(range_bottom_out.par_chunks_mut(cols))
                .zip(bullish_out.par_chunks_mut(cols))
                .zip(extra_bullish_out.par_chunks_mut(cols))
                .zip(bearish_out.par_chunks_mut(cols))
                .zip(extra_bearish_out.par_chunks_mut(cols))
                .enumerate()
                .try_for_each(
                    |(
                        row,
                        (
                            (
                                (
                                    ((range_top_row, range_bottom_row), bullish_row),
                                    extra_bullish_row,
                                ),
                                bearish_row,
                            ),
                            extra_bearish_row,
                        ),
                    )| {
                        do_row(
                            row,
                            range_top_row,
                            range_bottom_row,
                            bullish_row,
                            extra_bullish_row,
                            bearish_row,
                            extra_bearish_row,
                        )
                    },
                )?;
        }
        #[cfg(target_arch = "wasm32")]
        {
            for row in 0..rows {
                let start = row * cols;
                let end = start + cols;
                do_row(
                    row,
                    &mut range_top_out[start..end],
                    &mut range_bottom_out[start..end],
                    &mut bullish_out[start..end],
                    &mut extra_bullish_out[start..end],
                    &mut bearish_out[start..end],
                    &mut extra_bearish_out[start..end],
                )?;
            }
        }
    } else {
        for row in 0..rows {
            let start = row * cols;
            let end = start + cols;
            do_row(
                row,
                &mut range_top_out[start..end],
                &mut range_bottom_out[start..end],
                &mut bullish_out[start..end],
                &mut extra_bullish_out[start..end],
                &mut bearish_out[start..end],
                &mut extra_bearish_out[start..end],
            )?;
        }
    }

    Ok(combos)
}

pub fn range_breakout_signals_batch_with_kernel(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &RangeBreakoutSignalsBatchRange,
    kernel: Kernel,
) -> Result<RangeBreakoutSignalsBatchOutput, RangeBreakoutSignalsError> {
    match kernel {
        Kernel::Auto => {
            let _ = detect_best_batch_kernel();
        }
        k if !k.is_batch() => return Err(RangeBreakoutSignalsError::InvalidKernelForBatch(k)),
        _ => {}
    }

    let rows = expand_grid_range_breakout_signals(sweep)?.len();
    let cols = close.len();
    let mut top_guard = ManuallyDrop::new(make_uninit_matrix(rows, cols));
    let mut bottom_guard = ManuallyDrop::new(make_uninit_matrix(rows, cols));
    let mut bullish_guard = ManuallyDrop::new(make_uninit_matrix(rows, cols));
    let mut extra_bullish_guard = ManuallyDrop::new(make_uninit_matrix(rows, cols));
    let mut bearish_guard = ManuallyDrop::new(make_uninit_matrix(rows, cols));
    let mut extra_bearish_guard = ManuallyDrop::new(make_uninit_matrix(rows, cols));

    let top: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(top_guard.as_mut_ptr() as *mut f64, top_guard.len())
    };
    let bottom: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(bottom_guard.as_mut_ptr() as *mut f64, bottom_guard.len())
    };
    let bullish: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(bullish_guard.as_mut_ptr() as *mut f64, bullish_guard.len())
    };
    let extra_bullish: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(
            extra_bullish_guard.as_mut_ptr() as *mut f64,
            extra_bullish_guard.len(),
        )
    };
    let bearish: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(bearish_guard.as_mut_ptr() as *mut f64, bearish_guard.len())
    };
    let extra_bearish: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(
            extra_bearish_guard.as_mut_ptr() as *mut f64,
            extra_bearish_guard.len(),
        )
    };

    let combos = batch_inner_into(
        open,
        high,
        low,
        close,
        volume,
        sweep,
        !cfg!(target_arch = "wasm32"),
        top,
        bottom,
        bullish,
        extra_bullish,
        bearish,
        extra_bearish,
    )?;

    Ok(RangeBreakoutSignalsBatchOutput {
        range_top: unsafe {
            Vec::from_raw_parts(
                top_guard.as_mut_ptr() as *mut f64,
                top_guard.len(),
                top_guard.capacity(),
            )
        },
        range_bottom: unsafe {
            Vec::from_raw_parts(
                bottom_guard.as_mut_ptr() as *mut f64,
                bottom_guard.len(),
                bottom_guard.capacity(),
            )
        },
        bullish: unsafe {
            Vec::from_raw_parts(
                bullish_guard.as_mut_ptr() as *mut f64,
                bullish_guard.len(),
                bullish_guard.capacity(),
            )
        },
        extra_bullish: unsafe {
            Vec::from_raw_parts(
                extra_bullish_guard.as_mut_ptr() as *mut f64,
                extra_bullish_guard.len(),
                extra_bullish_guard.capacity(),
            )
        },
        bearish: unsafe {
            Vec::from_raw_parts(
                bearish_guard.as_mut_ptr() as *mut f64,
                bearish_guard.len(),
                bearish_guard.capacity(),
            )
        },
        extra_bearish: unsafe {
            Vec::from_raw_parts(
                extra_bearish_guard.as_mut_ptr() as *mut f64,
                extra_bearish_guard.len(),
                extra_bearish_guard.capacity(),
            )
        },
        combos,
        rows,
        cols,
    })
}

pub fn range_breakout_signals_batch_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &RangeBreakoutSignalsBatchRange,
    kernel: Kernel,
) -> Result<RangeBreakoutSignalsBatchOutput, RangeBreakoutSignalsError> {
    range_breakout_signals_batch_with_kernel(open, high, low, close, volume, sweep, kernel)
}

pub fn range_breakout_signals_batch_par_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &RangeBreakoutSignalsBatchRange,
    kernel: Kernel,
) -> Result<RangeBreakoutSignalsBatchOutput, RangeBreakoutSignalsError> {
    range_breakout_signals_batch_with_kernel(open, high, low, close, volume, sweep, kernel)
}

#[cfg(feature = "python")]
#[pyfunction(name = "range_breakout_signals")]
#[pyo3(signature = (open, high, low, close, volume, range_length=DEFAULT_RANGE_LENGTH, confirmation_length=DEFAULT_CONFIRMATION_LENGTH, kernel=None))]
pub fn range_breakout_signals_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    range_length: usize,
    confirmation_length: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let kernel = validate_kernel(kernel, false)?;
    let input = RangeBreakoutSignalsInput::from_slices(
        open.as_slice()?,
        high.as_slice()?,
        low.as_slice()?,
        close.as_slice()?,
        volume.as_slice()?,
        RangeBreakoutSignalsParams {
            range_length: Some(range_length),
            confirmation_length: Some(confirmation_length),
        },
    );
    let out = py
        .allow_threads(|| range_breakout_signals_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dict = PyDict::new(py);
    dict.set_item("range_top", out.range_top.into_pyarray(py))?;
    dict.set_item("range_bottom", out.range_bottom.into_pyarray(py))?;
    dict.set_item("bullish", out.bullish.into_pyarray(py))?;
    dict.set_item("extra_bullish", out.extra_bullish.into_pyarray(py))?;
    dict.set_item("bearish", out.bearish.into_pyarray(py))?;
    dict.set_item("extra_bearish", out.extra_bearish.into_pyarray(py))?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyfunction(name = "range_breakout_signals_batch")]
#[pyo3(signature = (open, high, low, close, volume, range_length_range=(DEFAULT_RANGE_LENGTH, DEFAULT_RANGE_LENGTH, 0), confirmation_length_range=(DEFAULT_CONFIRMATION_LENGTH, DEFAULT_CONFIRMATION_LENGTH, 0), kernel=None))]
pub fn range_breakout_signals_batch_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    range_length_range: (usize, usize, usize),
    confirmation_length_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let kernel = validate_kernel(kernel, true)?;
    let out = range_breakout_signals_batch_with_kernel(
        open.as_slice()?,
        high.as_slice()?,
        low.as_slice()?,
        close.as_slice()?,
        volume.as_slice()?,
        &RangeBreakoutSignalsBatchRange {
            range_length: range_length_range,
            confirmation_length: confirmation_length_range,
        },
        kernel,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dict = PyDict::new(py);
    dict.set_item(
        "range_top",
        out.range_top
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "range_bottom",
        out.range_bottom
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "bullish",
        out.bullish.into_pyarray(py).reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "extra_bullish",
        out.extra_bullish
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "bearish",
        out.bearish.into_pyarray(py).reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "extra_bearish",
        out.extra_bearish
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "range_lengths",
        out.combos
            .iter()
            .map(|combo| combo.range_length.unwrap_or(DEFAULT_RANGE_LENGTH))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "confirmation_lengths",
        out.combos
            .iter()
            .map(|combo| {
                combo
                    .confirmation_length
                    .unwrap_or(DEFAULT_CONFIRMATION_LENGTH)
            })
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", out.rows)?;
    dict.set_item("cols", out.cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "RangeBreakoutSignalsStream")]
pub struct RangeBreakoutSignalsStreamPy {
    inner: RangeBreakoutSignalsStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl RangeBreakoutSignalsStreamPy {
    #[new]
    #[pyo3(signature = (range_length=None, confirmation_length=None))]
    pub fn new(range_length: Option<usize>, confirmation_length: Option<usize>) -> PyResult<Self> {
        let inner = RangeBreakoutSignalsStream::try_new(RangeBreakoutSignalsParams {
            range_length,
            confirmation_length,
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    pub fn update(
        &mut self,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Option<(f64, f64, f64, f64, f64, f64)> {
        self.inner.update(open, high, low, close, volume)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct RangeBreakoutSignalsBatchConfig {
    pub range_length_range: (usize, usize, usize),
    pub confirmation_length_range: (usize, usize, usize),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn range_breakout_signals_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    range_length: usize,
    confirmation_length: usize,
) -> Result<JsValue, JsValue> {
    let input = RangeBreakoutSignalsInput::from_slices(
        open,
        high,
        low,
        close,
        volume,
        RangeBreakoutSignalsParams {
            range_length: Some(range_length),
            confirmation_length: Some(confirmation_length),
        },
    );
    let out = range_breakout_signals_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&out).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn range_breakout_signals_alloc(len: usize) -> *mut f64 {
    let mut values = Vec::<f64>::with_capacity(len);
    let ptr = values.as_mut_ptr();
    std::mem::forget(values);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn range_breakout_signals_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn range_breakout_signals_into(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    volume_ptr: *const f64,
    range_top_ptr: *mut f64,
    range_bottom_ptr: *mut f64,
    bullish_ptr: *mut f64,
    extra_bullish_ptr: *mut f64,
    bearish_ptr: *mut f64,
    extra_bearish_ptr: *mut f64,
    len: usize,
    range_length: usize,
    confirmation_length: usize,
) -> Result<(), JsValue> {
    if open_ptr.is_null()
        || high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || volume_ptr.is_null()
        || range_top_ptr.is_null()
        || range_bottom_ptr.is_null()
        || bullish_ptr.is_null()
        || extra_bullish_ptr.is_null()
        || bearish_ptr.is_null()
        || extra_bearish_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let input = RangeBreakoutSignalsInput::from_slices(
            std::slice::from_raw_parts(open_ptr, len),
            std::slice::from_raw_parts(high_ptr, len),
            std::slice::from_raw_parts(low_ptr, len),
            std::slice::from_raw_parts(close_ptr, len),
            std::slice::from_raw_parts(volume_ptr, len),
            RangeBreakoutSignalsParams {
                range_length: Some(range_length),
                confirmation_length: Some(confirmation_length),
            },
        );
        let out = range_breakout_signals_with_kernel(&input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        std::slice::from_raw_parts_mut(range_top_ptr, len).copy_from_slice(&out.range_top);
        std::slice::from_raw_parts_mut(range_bottom_ptr, len).copy_from_slice(&out.range_bottom);
        std::slice::from_raw_parts_mut(bullish_ptr, len).copy_from_slice(&out.bullish);
        std::slice::from_raw_parts_mut(extra_bullish_ptr, len).copy_from_slice(&out.extra_bullish);
        std::slice::from_raw_parts_mut(bearish_ptr, len).copy_from_slice(&out.bearish);
        std::slice::from_raw_parts_mut(extra_bearish_ptr, len).copy_from_slice(&out.extra_bearish);
    }
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = range_breakout_signals_batch)]
pub fn range_breakout_signals_batch_unified_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: RangeBreakoutSignalsBatchConfig =
        serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let out = range_breakout_signals_batch_with_kernel(
        open,
        high,
        low,
        close,
        volume,
        &RangeBreakoutSignalsBatchRange {
            range_length: config.range_length_range,
            confirmation_length: config.confirmation_length_range,
        },
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&out).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = range_breakout_signals_batch_into)]
pub fn range_breakout_signals_batch_into(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    volume_ptr: *const f64,
    range_top_ptr: *mut f64,
    range_bottom_ptr: *mut f64,
    bullish_ptr: *mut f64,
    extra_bullish_ptr: *mut f64,
    bearish_ptr: *mut f64,
    extra_bearish_ptr: *mut f64,
    len: usize,
    range_length_start: usize,
    range_length_end: usize,
    range_length_step: usize,
    confirmation_length_start: usize,
    confirmation_length_end: usize,
    confirmation_length_step: usize,
) -> Result<usize, JsValue> {
    let sweep = RangeBreakoutSignalsBatchRange {
        range_length: (range_length_start, range_length_end, range_length_step),
        confirmation_length: (
            confirmation_length_start,
            confirmation_length_end,
            confirmation_length_step,
        ),
    };
    let rows = expand_grid_range_breakout_signals(&sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?
        .len();
    let total = rows
        .checked_mul(len)
        .ok_or_else(|| JsValue::from_str("rows * cols overflow"))?;

    unsafe {
        let out = range_breakout_signals_batch_with_kernel(
            std::slice::from_raw_parts(open_ptr, len),
            std::slice::from_raw_parts(high_ptr, len),
            std::slice::from_raw_parts(low_ptr, len),
            std::slice::from_raw_parts(close_ptr, len),
            std::slice::from_raw_parts(volume_ptr, len),
            &sweep,
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        std::slice::from_raw_parts_mut(range_top_ptr, total).copy_from_slice(&out.range_top);
        std::slice::from_raw_parts_mut(range_bottom_ptr, total).copy_from_slice(&out.range_bottom);
        std::slice::from_raw_parts_mut(bullish_ptr, total).copy_from_slice(&out.bullish);
        std::slice::from_raw_parts_mut(extra_bullish_ptr, total)
            .copy_from_slice(&out.extra_bullish);
        std::slice::from_raw_parts_mut(bearish_ptr, total).copy_from_slice(&out.bearish);
        std::slice::from_raw_parts_mut(extra_bearish_ptr, total)
            .copy_from_slice(&out.extra_bearish);
    }

    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub struct RangeBreakoutSignalsStreamWasm {
    inner: RangeBreakoutSignalsStream,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
impl RangeBreakoutSignalsStreamWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(
        range_length: Option<usize>,
        confirmation_length: Option<usize>,
    ) -> Result<RangeBreakoutSignalsStreamWasm, JsValue> {
        let inner = RangeBreakoutSignalsStream::try_new(RangeBreakoutSignalsParams {
            range_length,
            confirmation_length,
        })
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Self { inner })
    }

    pub fn update(
        &mut self,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.inner.update(open, high, low, close, volume))
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_ohlcv() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut open = Vec::with_capacity(96);
        let mut high = Vec::with_capacity(96);
        let mut low = Vec::with_capacity(96);
        let mut close = Vec::with_capacity(96);
        let mut volume = Vec::with_capacity(96);

        for i in 0..24 {
            let base = 100.0 + i as f64 * 0.35;
            let o = base - 1.4;
            let c = base + if i & 1 == 0 { 1.7 } else { -1.6 };
            open.push(o);
            close.push(c);
            high.push(o.max(c) + 1.1);
            low.push(o.min(c) - 1.0);
            volume.push(900.0 + (i as f64) * 8.0);
        }

        for i in 24..36 {
            let base = 108.5 + ((i - 24) as f64) * 0.03;
            open.push(base - 0.03);
            close.push(base + 0.03);
            high.push(base + 0.18);
            low.push(base - 0.18);
            volume.push(1600.0 + (i as f64) * 6.0);
        }

        open.push(109.2);
        high.push(112.4);
        low.push(109.0);
        close.push(112.1);
        volume.push(2200.0);

        for i in 37..60 {
            let base = 111.8 - ((i - 37) as f64) * 0.08;
            let o = base + 0.8;
            let c = base - 0.9;
            open.push(o);
            close.push(c);
            high.push(o.max(c) + 0.9);
            low.push(o.min(c) - 0.8);
            volume.push(1100.0 + (i as f64) * 5.0);
        }

        for i in 60..72 {
            let base = 104.7 - ((i - 60) as f64) * 0.02;
            open.push(base + 0.02);
            close.push(base - 0.02);
            high.push(base + 0.16);
            low.push(base - 0.16);
            volume.push(1750.0 + (i as f64) * 5.0);
        }

        open.push(103.9);
        high.push(104.0);
        low.push(100.6);
        close.push(100.9);
        volume.push(2400.0);

        for i in 73..96 {
            let base = 101.4 + ((i - 73) as f64) * 0.06;
            let o = base - 0.3;
            let c = base + 0.25;
            open.push(o);
            close.push(c);
            high.push(o.max(c) + 0.45);
            low.push(o.min(c) - 0.45);
            volume.push(1200.0 + (i as f64) * 3.0);
        }

        (open, high, low, close, volume)
    }

    #[test]
    fn range_breakout_signals_outputs_present() -> Result<(), Box<dyn StdError>> {
        let (open, high, low, close, volume) = sample_ohlcv();
        let out = range_breakout_signals(&RangeBreakoutSignalsInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            &volume,
            RangeBreakoutSignalsParams::default(),
        ))?;
        assert!(out.range_top.iter().any(|value| value.is_finite()));
        assert!(out.range_bottom.iter().any(|value| value.is_finite()));
        Ok(())
    }

    fn assert_same(lhs: &[f64], rhs: &[f64]) {
        assert_eq!(lhs.len(), rhs.len());
        for (idx, (&a, &b)) in lhs.iter().zip(rhs.iter()).enumerate() {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert_eq!(a, b, "mismatch at {idx}: {a} vs {b}");
        }
    }

    #[test]
    fn default_fixed_state_matches_stream() -> Result<(), Box<dyn StdError>> {
        let (open, high, low, close, volume) = sample_ohlcv();
        let out = range_breakout_signals(&RangeBreakoutSignalsInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            &volume,
            RangeBreakoutSignalsParams::default(),
        ))?;
        let mut stream =
            RangeBreakoutSignalsStream::try_new(RangeBreakoutSignalsParams::default())?;
        let mut range_top = Vec::with_capacity(close.len());
        let mut range_bottom = Vec::with_capacity(close.len());
        let mut bullish = Vec::with_capacity(close.len());
        let mut extra_bullish = Vec::with_capacity(close.len());
        let mut bearish = Vec::with_capacity(close.len());
        let mut extra_bearish = Vec::with_capacity(close.len());

        for i in 0..close.len() {
            if let Some((rt, rb, b, eb, br, ebr)) =
                stream.update(open[i], high[i], low[i], close[i], volume[i])
            {
                range_top.push(rt);
                range_bottom.push(rb);
                bullish.push(b);
                extra_bullish.push(eb);
                bearish.push(br);
                extra_bearish.push(ebr);
            } else {
                range_top.push(f64::NAN);
                range_bottom.push(f64::NAN);
                bullish.push(f64::NAN);
                extra_bullish.push(f64::NAN);
                bearish.push(f64::NAN);
                extra_bearish.push(f64::NAN);
            }
        }

        assert_same(&out.range_top, &range_top);
        assert_same(&out.range_bottom, &range_bottom);
        assert_same(&out.bullish, &bullish);
        assert_same(&out.extra_bullish, &extra_bullish);
        assert_same(&out.bearish, &bearish);
        assert_same(&out.extra_bearish, &extra_bearish);
        Ok(())
    }
}
