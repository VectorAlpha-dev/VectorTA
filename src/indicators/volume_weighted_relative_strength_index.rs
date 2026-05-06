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

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_uninit_f64, detect_best_batch_kernel, init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::collections::VecDeque;
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

const DEFAULT_SOURCE: &str = "close";
const DEFAULT_RSI_LENGTH: usize = 14;
const DEFAULT_RANGE_LENGTH: usize = 10;
const DEFAULT_MA_LENGTH: usize = 14;
const DEFAULT_MA_TYPE: &str = "EMA";
const EPS: f64 = 1e-12;

#[derive(Debug, Clone)]
pub enum VolumeWeightedRelativeStrengthIndexData<'a> {
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
pub struct VolumeWeightedRelativeStrengthIndexOutput {
    pub rsi: Vec<f64>,
    pub consolidation_strength: Vec<f64>,
    pub rsi_ma: Vec<f64>,
    pub bearish_tp: Vec<f64>,
    pub bullish_tp: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct VolumeWeightedRelativeStrengthIndexParams {
    pub rsi_length: Option<usize>,
    pub range_length: Option<usize>,
    pub ma_length: Option<usize>,
    pub ma_type: Option<String>,
}

impl Default for VolumeWeightedRelativeStrengthIndexParams {
    fn default() -> Self {
        Self {
            rsi_length: Some(DEFAULT_RSI_LENGTH),
            range_length: Some(DEFAULT_RANGE_LENGTH),
            ma_length: Some(DEFAULT_MA_LENGTH),
            ma_type: Some(DEFAULT_MA_TYPE.to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VolumeWeightedRelativeStrengthIndexInput<'a> {
    pub data: VolumeWeightedRelativeStrengthIndexData<'a>,
    pub params: VolumeWeightedRelativeStrengthIndexParams,
}

impl<'a> VolumeWeightedRelativeStrengthIndexInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: VolumeWeightedRelativeStrengthIndexParams,
    ) -> Self {
        Self {
            data: VolumeWeightedRelativeStrengthIndexData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        source: &'a [f64],
        volume: &'a [f64],
        params: VolumeWeightedRelativeStrengthIndexParams,
    ) -> Self {
        Self {
            data: VolumeWeightedRelativeStrengthIndexData::Slices { source, volume },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(
            candles,
            DEFAULT_SOURCE,
            VolumeWeightedRelativeStrengthIndexParams::default(),
        )
    }
}

#[derive(Debug, Clone, Copy)]
enum VolumeWeightedRelativeStrengthIndexMaKind {
    Ema,
    Sma,
    Hma,
    Rma,
    Wma,
    Vwma,
}

#[derive(Debug, Clone, Copy)]
struct ResolvedParams {
    rsi_length: usize,
    range_length: usize,
    ma_length: usize,
    ma_kind: VolumeWeightedRelativeStrengthIndexMaKind,
}

#[derive(Debug, Clone)]
struct PreparedInput<'a> {
    source: &'a [f64],
    volume: &'a [f64],
    len: usize,
    params: ResolvedParams,
    warmup: usize,
}

#[derive(Debug, Clone)]
pub struct VolumeWeightedRelativeStrengthIndexBuilder {
    source: Option<&'static str>,
    rsi_length: Option<usize>,
    range_length: Option<usize>,
    ma_length: Option<usize>,
    ma_type: Option<String>,
    kernel: Kernel,
}

impl Default for VolumeWeightedRelativeStrengthIndexBuilder {
    fn default() -> Self {
        Self {
            source: None,
            rsi_length: None,
            range_length: None,
            ma_length: None,
            ma_type: None,
            kernel: Kernel::Auto,
        }
    }
}

impl VolumeWeightedRelativeStrengthIndexBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn source(mut self, value: &'static str) -> Self {
        self.source = Some(value);
        self
    }

    #[inline(always)]
    pub fn rsi_length(mut self, value: usize) -> Self {
        self.rsi_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn range_length(mut self, value: usize) -> Self {
        self.range_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn ma_length(mut self, value: usize) -> Self {
        self.ma_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn ma_type<S: Into<String>>(mut self, value: S) -> Self {
        self.ma_type = Some(value.into());
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }
}

#[derive(Debug, Error)]
pub enum VolumeWeightedRelativeStrengthIndexError {
    #[error("volume_weighted_relative_strength_index: input data slice is empty")]
    EmptyInputData,
    #[error("volume_weighted_relative_strength_index: data length mismatch: source={source_len}, volume={volume_len}")]
    DataLengthMismatch {
        source_len: usize,
        volume_len: usize,
    },
    #[error("volume_weighted_relative_strength_index: all values are NaN")]
    AllValuesNaN,
    #[error("volume_weighted_relative_strength_index: invalid rsi_length: rsi_length = {rsi_length}, data length = {data_len}")]
    InvalidRsiLength { rsi_length: usize, data_len: usize },
    #[error("volume_weighted_relative_strength_index: invalid range_length: range_length = {range_length}, data length = {data_len}")]
    InvalidRangeLength {
        range_length: usize,
        data_len: usize,
    },
    #[error("volume_weighted_relative_strength_index: invalid ma_length: ma_length = {ma_length}, data length = {data_len}")]
    InvalidMaLength { ma_length: usize, data_len: usize },
    #[error("volume_weighted_relative_strength_index: invalid ma_type: {ma_type}")]
    InvalidMaType { ma_type: String },
    #[error("volume_weighted_relative_strength_index: not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("volume_weighted_relative_strength_index: output length mismatch: expected {expected}, got {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("volume_weighted_relative_strength_index: invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("volume_weighted_relative_strength_index: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Debug, Clone, Copy)]
struct VolumeWeightedRelativeStrengthIndexPoint {
    rsi: f64,
    consolidation_strength: f64,
    rsi_ma: f64,
    bearish_tp: f64,
    bullish_tp: f64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum VolumeWeightedRelativeStrengthIndexOutputField {
    Rsi,
    ConsolidationStrength,
    RsiMa,
    BearishTp,
    BullishTp,
}

#[derive(Debug, Clone, Copy)]
pub struct VolumeWeightedRelativeStrengthIndexStreamOutput {
    pub rsi: f64,
    pub consolidation_strength: f64,
    pub rsi_ma: f64,
    pub bearish_tp: f64,
    pub bullish_tp: f64,
}

#[derive(Debug, Clone)]
pub struct VolumeWeightedRelativeStrengthIndexStream {
    core: VolumeWeightedRelativeStrengthIndexCore,
}

#[derive(Debug, Clone)]
struct RmaState {
    period: usize,
    count: usize,
    sum: f64,
    value: Option<f64>,
}

impl RmaState {
    #[inline(always)]
    fn new(period: usize) -> Self {
        Self {
            period,
            count: 0,
            sum: 0.0,
            value: None,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.count = 0;
        self.sum = 0.0;
        self.value = None;
    }

    #[inline(always)]
    fn update(&mut self, input: f64) -> Option<f64> {
        if let Some(prev) = self.value {
            let next = ((prev * (self.period as f64 - 1.0)) + input) / self.period as f64;
            self.value = Some(next);
            return Some(next);
        }
        self.count += 1;
        self.sum += input;
        if self.count == self.period {
            let seeded = self.sum / self.period as f64;
            self.value = Some(seeded);
            Some(seeded)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
struct RollingSma {
    period: usize,
    buffer: VecDeque<f64>,
    sum: f64,
    nan_count: usize,
}

impl RollingSma {
    #[inline(always)]
    fn new(period: usize) -> Self {
        Self {
            period,
            buffer: VecDeque::with_capacity(period),
            sum: 0.0,
            nan_count: 0,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.buffer.clear();
        self.sum = 0.0;
        self.nan_count = 0;
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<f64> {
        if self.buffer.len() == self.period {
            if let Some(old) = self.buffer.pop_front() {
                if old.is_nan() {
                    self.nan_count = self.nan_count.saturating_sub(1);
                } else {
                    self.sum -= old;
                }
            }
        }
        self.buffer.push_back(value);
        if value.is_nan() {
            self.nan_count += 1;
        } else {
            self.sum += value;
        }
        if self.buffer.len() < self.period {
            return None;
        }
        if self.nan_count > 0 {
            Some(f64::NAN)
        } else {
            Some(self.sum / self.period as f64)
        }
    }
}

#[derive(Debug, Clone)]
struct LinWma {
    period: usize,
    inv_norm: f64,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,
    count: usize,
    sum: f64,
    wsum: f64,
    nan_count: usize,
    dirty: bool,
}

impl LinWma {
    #[inline(always)]
    fn new(period: usize) -> Self {
        let norm = (period as f64) * ((period as f64) + 1.0) * 0.5;
        Self {
            period,
            inv_norm: 1.0 / norm,
            buffer: vec![f64::NAN; period],
            head: 0,
            filled: false,
            count: 0,
            sum: 0.0,
            wsum: 0.0,
            nan_count: 0,
            dirty: false,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.buffer.fill(f64::NAN);
        self.head = 0;
        self.filled = false;
        self.count = 0;
        self.sum = 0.0;
        self.wsum = 0.0;
        self.nan_count = 0;
        self.dirty = false;
    }

    #[inline(always)]
    fn rebuild(&mut self) {
        self.sum = 0.0;
        self.wsum = 0.0;
        self.nan_count = 0;

        let mut idx = self.head;
        for i in 0..self.period {
            let v = self.buffer[idx];
            if v.is_nan() {
                self.nan_count += 1;
            } else {
                self.sum += v;
                self.wsum = (i as f64 + 1.0).mul_add(v, self.wsum);
            }
            idx = if idx + 1 == self.period { 0 } else { idx + 1 };
        }
        self.dirty = self.nan_count != 0;
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<f64> {
        let n = self.period as f64;
        let old = self.buffer[self.head];
        self.buffer[self.head] = value;
        self.head = if self.head + 1 == self.period {
            0
        } else {
            self.head + 1
        };

        if !self.filled {
            self.count += 1;
            if value.is_nan() {
                self.nan_count += 1;
                self.dirty = true;
            } else {
                self.sum += value;
                self.wsum = (self.count as f64).mul_add(value, self.wsum);
            }
            if self.count == self.period {
                self.filled = true;
                return Some(if self.nan_count > 0 {
                    f64::NAN
                } else {
                    self.wsum * self.inv_norm
                });
            }
            return None;
        }

        if old.is_nan() {
            self.nan_count = self.nan_count.saturating_sub(1);
        }
        if value.is_nan() {
            self.nan_count += 1;
        }

        if self.nan_count > 0 {
            self.dirty = true;
            return Some(f64::NAN);
        }

        if self.dirty {
            self.rebuild();
            self.dirty = false;
            return Some(self.wsum * self.inv_norm);
        }

        let prev_sum = self.sum;
        self.sum = prev_sum + value - old;
        self.wsum = n.mul_add(value, self.wsum - prev_sum);
        Some(self.wsum * self.inv_norm)
    }
}

#[derive(Debug, Clone)]
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
    fn update(&mut self, value: f64) -> Option<f64> {
        let next = match self.value {
            Some(prev) => self.alpha.mul_add(value, (1.0 - self.alpha) * prev),
            None => value,
        };
        self.value = Some(next);
        Some(next)
    }
}

#[derive(Debug, Clone)]
struct HmaState {
    wma_half: LinWma,
    wma_full: LinWma,
    wma_sqrt: LinWma,
}

impl HmaState {
    #[inline(always)]
    fn new(period: usize) -> Self {
        let half = (period / 2).max(1);
        let sqrt_len = ((period as f64).sqrt().floor() as usize).max(1);
        Self {
            wma_half: LinWma::new(half),
            wma_full: LinWma::new(period.max(1)),
            wma_sqrt: LinWma::new(sqrt_len),
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.wma_half.reset();
        self.wma_full.reset();
        self.wma_sqrt.reset();
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<f64> {
        let half = self.wma_half.update(value);
        let full = self.wma_full.update(value);
        if let (Some(h), Some(f)) = (half, full) {
            self.wma_sqrt.update(2.0f64.mul_add(h, -f))
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
struct VwmaState {
    period: usize,
    pv: VecDeque<f64>,
    vols: VecDeque<f64>,
    pv_sum: f64,
    vol_sum: f64,
    nan_count: usize,
}

impl VwmaState {
    #[inline(always)]
    fn new(period: usize) -> Self {
        Self {
            period,
            pv: VecDeque::with_capacity(period),
            vols: VecDeque::with_capacity(period),
            pv_sum: 0.0,
            vol_sum: 0.0,
            nan_count: 0,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.pv.clear();
        self.vols.clear();
        self.pv_sum = 0.0;
        self.vol_sum = 0.0;
        self.nan_count = 0;
    }

    #[inline(always)]
    fn update(&mut self, value: f64, volume: f64) -> Option<f64> {
        if self.pv.len() == self.period {
            let old_pv = self.pv.pop_front().unwrap_or(f64::NAN);
            let old_vol = self.vols.pop_front().unwrap_or(f64::NAN);
            if old_pv.is_nan() || old_vol.is_nan() {
                self.nan_count = self.nan_count.saturating_sub(1);
            } else {
                self.pv_sum -= old_pv;
                self.vol_sum -= old_vol;
            }
        }
        let pv = value * volume;
        self.pv.push_back(pv);
        self.vols.push_back(volume);
        if pv.is_nan() || volume.is_nan() {
            self.nan_count += 1;
        } else {
            self.pv_sum += pv;
            self.vol_sum += volume;
        }
        if self.pv.len() < self.period {
            return None;
        }
        if self.nan_count > 0 || self.vol_sum.abs() <= EPS {
            Some(f64::NAN)
        } else {
            Some(self.pv_sum / self.vol_sum)
        }
    }
}

#[derive(Debug, Clone)]
enum MaState {
    Ema(EmaState),
    Sma(RollingSma),
    Hma(HmaState),
    Rma(RmaState),
    Wma(LinWma),
    Vwma(VwmaState),
}

impl MaState {
    #[inline(always)]
    fn new(kind: VolumeWeightedRelativeStrengthIndexMaKind, period: usize) -> Self {
        match kind {
            VolumeWeightedRelativeStrengthIndexMaKind::Ema => Self::Ema(EmaState::new(period)),
            VolumeWeightedRelativeStrengthIndexMaKind::Sma => Self::Sma(RollingSma::new(period)),
            VolumeWeightedRelativeStrengthIndexMaKind::Hma => Self::Hma(HmaState::new(period)),
            VolumeWeightedRelativeStrengthIndexMaKind::Rma => Self::Rma(RmaState::new(period)),
            VolumeWeightedRelativeStrengthIndexMaKind::Wma => Self::Wma(LinWma::new(period)),
            VolumeWeightedRelativeStrengthIndexMaKind::Vwma => Self::Vwma(VwmaState::new(period)),
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        match self {
            Self::Ema(state) => state.reset(),
            Self::Sma(state) => state.reset(),
            Self::Hma(state) => state.reset(),
            Self::Rma(state) => state.reset(),
            Self::Wma(state) => state.reset(),
            Self::Vwma(state) => state.reset(),
        }
    }

    #[inline(always)]
    fn update(&mut self, value: f64, volume: f64) -> Option<f64> {
        match self {
            Self::Ema(state) => state.update(value),
            Self::Sma(state) => state.update(value),
            Self::Hma(state) => state.update(value),
            Self::Rma(state) => state.update(value),
            Self::Wma(state) => state.update(value),
            Self::Vwma(state) => state.update(value, volume),
        }
    }
}

#[derive(Debug, Clone)]
struct VolumeWeightedRelativeStrengthIndexCore {
    params: ResolvedParams,
    prev_source: Option<f64>,
    prev_rsi: Option<f64>,
    prev_dir: Option<i8>,
    valid_rsi_count: usize,
    up_rma: RmaState,
    down_rma: RmaState,
    volume_rma: RmaState,
    ma_state: MaState,
    range_sma: RollingSma,
    range_hma_short: HmaState,
    range_hma_long: HmaState,
}

impl VolumeWeightedRelativeStrengthIndexCore {
    #[inline(always)]
    fn new(params: ResolvedParams) -> Self {
        let short = (params.range_length / 2).max(1);
        Self {
            params,
            prev_source: None,
            prev_rsi: None,
            prev_dir: None,
            valid_rsi_count: 0,
            up_rma: RmaState::new(params.rsi_length),
            down_rma: RmaState::new(params.rsi_length),
            volume_rma: RmaState::new(params.rsi_length),
            ma_state: MaState::new(params.ma_kind, params.ma_length),
            range_sma: RollingSma::new(params.range_length),
            range_hma_short: HmaState::new(short),
            range_hma_long: HmaState::new(params.range_length.saturating_mul(2).max(1)),
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.prev_source = None;
        self.prev_rsi = None;
        self.prev_dir = None;
        self.valid_rsi_count = 0;
        self.up_rma.reset();
        self.down_rma.reset();
        self.volume_rma.reset();
        self.ma_state.reset();
        self.range_sma.reset();
        self.range_hma_short.reset();
        self.range_hma_long.reset();
    }

    #[inline(always)]
    fn update(
        &mut self,
        source: f64,
        volume: f64,
    ) -> Option<VolumeWeightedRelativeStrengthIndexPoint> {
        if !source.is_finite() || !volume.is_finite() {
            self.reset();
            return None;
        }

        let prev_source = match self.prev_source {
            Some(value) => value,
            None => {
                self.prev_source = Some(source);
                return None;
            }
        };
        self.prev_source = Some(source);

        let delta = source - prev_source;
        let gain = delta.max(0.0) * volume;
        let loss = (-delta).max(0.0) * volume;
        let up_num = self.up_rma.update(gain);
        let down_num = self.down_rma.update(loss);
        let vol_avg = self.volume_rma.update(volume);
        let (Some(up_num), Some(down_num), Some(vol_avg)) = (up_num, down_num, vol_avg) else {
            return None;
        };
        if vol_avg.abs() <= EPS {
            return None;
        }

        let up = up_num / vol_avg;
        let down = down_num / vol_avg;
        let rsi = if down.abs() <= EPS {
            100.0
        } else if up.abs() <= EPS {
            0.0
        } else {
            100.0 - (100.0 / (1.0 + up / down))
        };

        let rsi_ma = self.ma_state.update(rsi, volume).unwrap_or(f64::NAN);
        let bearish_tp = if let Some(prev) = self.prev_rsi {
            if prev >= 80.0 && rsi < 80.0 {
                95.0
            } else {
                f64::NAN
            }
        } else {
            f64::NAN
        };
        let bullish_tp = if let Some(prev) = self.prev_rsi {
            if prev <= 20.0 && rsi > 20.0 {
                5.0
            } else {
                f64::NAN
            }
        } else {
            f64::NAN
        };

        self.valid_rsi_count += 1;
        let dir = if rsi > 50.0 { 1 } else { -1 };
        let transition = if self.prev_dir.is_some_and(|prev| prev + dir == 0) {
            1.0
        } else {
            0.0
        };
        let denom = self.valid_rsi_count.min(self.params.range_length) as f64;
        let x = transition / denom;
        let p = self.range_sma.update(x).unwrap_or(f64::NAN);
        let scale = ((self.params.range_length / 2).max(1) as f64) * (5.0 / 3.0);
        let short = if p.is_finite() {
            self.range_hma_short.update(p).unwrap_or(f64::NAN)
        } else {
            f64::NAN
        };
        let f = if short.is_finite() {
            short * scale
        } else {
            f64::NAN
        };
        let consolidation_strength = if f.is_finite() {
            self.range_hma_long
                .update(f)
                .map(|value| value.max(0.0))
                .unwrap_or(f64::NAN)
        } else {
            f64::NAN
        };

        self.prev_dir = Some(dir);
        self.prev_rsi = Some(rsi);

        Some(VolumeWeightedRelativeStrengthIndexPoint {
            rsi,
            consolidation_strength,
            rsi_ma,
            bearish_tp,
            bullish_tp,
        })
    }
}

#[inline(always)]
fn normalize_ma_type(
    ma_type: &str,
) -> Result<VolumeWeightedRelativeStrengthIndexMaKind, VolumeWeightedRelativeStrengthIndexError> {
    let value = ma_type.trim();
    if value.eq_ignore_ascii_case("ema") {
        Ok(VolumeWeightedRelativeStrengthIndexMaKind::Ema)
    } else if value.eq_ignore_ascii_case("sma") {
        Ok(VolumeWeightedRelativeStrengthIndexMaKind::Sma)
    } else if value.eq_ignore_ascii_case("hma") {
        Ok(VolumeWeightedRelativeStrengthIndexMaKind::Hma)
    } else if value.eq_ignore_ascii_case("smma (rma)")
        || value.eq_ignore_ascii_case("smma")
        || value.eq_ignore_ascii_case("rma")
    {
        Ok(VolumeWeightedRelativeStrengthIndexMaKind::Rma)
    } else if value.eq_ignore_ascii_case("wma") {
        Ok(VolumeWeightedRelativeStrengthIndexMaKind::Wma)
    } else if value.eq_ignore_ascii_case("vwma") {
        Ok(VolumeWeightedRelativeStrengthIndexMaKind::Vwma)
    } else {
        Err(VolumeWeightedRelativeStrengthIndexError::InvalidMaType {
            ma_type: ma_type.to_string(),
        })
    }
}

#[inline(always)]
fn extract_source_volume<'a>(
    input: &'a VolumeWeightedRelativeStrengthIndexInput<'a>,
) -> Result<(&'a [f64], &'a [f64]), VolumeWeightedRelativeStrengthIndexError> {
    let (source, volume) = match &input.data {
        VolumeWeightedRelativeStrengthIndexData::Candles { candles, source } => (
            if source.eq_ignore_ascii_case(DEFAULT_SOURCE) {
                candles.close.as_slice()
            } else {
                source_type(candles, source)
            },
            candles.volume.as_slice(),
        ),
        VolumeWeightedRelativeStrengthIndexData::Slices { source, volume } => (*source, *volume),
    };
    if source.is_empty() || volume.is_empty() {
        return Err(VolumeWeightedRelativeStrengthIndexError::EmptyInputData);
    }
    if source.len() != volume.len() {
        return Err(
            VolumeWeightedRelativeStrengthIndexError::DataLengthMismatch {
                source_len: source.len(),
                volume_len: volume.len(),
            },
        );
    }
    Ok((source, volume))
}

fn prepare_input<'a>(
    input: &'a VolumeWeightedRelativeStrengthIndexInput<'a>,
) -> Result<PreparedInput<'a>, VolumeWeightedRelativeStrengthIndexError> {
    let (source, volume) = extract_source_volume(input)?;
    let len = source.len();
    let first = (0..len)
        .find(|&i| source[i].is_finite() && volume[i].is_finite())
        .ok_or(VolumeWeightedRelativeStrengthIndexError::AllValuesNaN)?;

    let rsi_length = input.params.rsi_length.unwrap_or(DEFAULT_RSI_LENGTH);
    let range_length = input.params.range_length.unwrap_or(DEFAULT_RANGE_LENGTH);
    let ma_length = input.params.ma_length.unwrap_or(DEFAULT_MA_LENGTH);
    let ma_type = input.params.ma_type.as_deref().unwrap_or(DEFAULT_MA_TYPE);
    let ma_kind = normalize_ma_type(ma_type)?;

    if rsi_length == 0 || rsi_length > len {
        return Err(VolumeWeightedRelativeStrengthIndexError::InvalidRsiLength {
            rsi_length,
            data_len: len,
        });
    }
    if range_length == 0 || range_length > len {
        return Err(
            VolumeWeightedRelativeStrengthIndexError::InvalidRangeLength {
                range_length,
                data_len: len,
            },
        );
    }
    if ma_length == 0 || ma_length > len {
        return Err(VolumeWeightedRelativeStrengthIndexError::InvalidMaLength {
            ma_length,
            data_len: len,
        });
    }

    let valid = len - first;
    if valid < rsi_length + 1 {
        return Err(
            VolumeWeightedRelativeStrengthIndexError::NotEnoughValidData {
                needed: rsi_length + 1,
                valid,
            },
        );
    }

    let short = (range_length / 2).max(1);
    let short_hma_warm = short + ((short as f64).sqrt().floor() as usize).max(1) - 1;
    let long_period = range_length.saturating_mul(2).max(1);
    let long_hma_warm = long_period + ((long_period as f64).sqrt().floor() as usize).max(1) - 1;
    let ma_warm = match ma_kind {
        VolumeWeightedRelativeStrengthIndexMaKind::Ema => 1,
        VolumeWeightedRelativeStrengthIndexMaKind::Sma
        | VolumeWeightedRelativeStrengthIndexMaKind::Rma
        | VolumeWeightedRelativeStrengthIndexMaKind::Wma
        | VolumeWeightedRelativeStrengthIndexMaKind::Vwma => ma_length,
        VolumeWeightedRelativeStrengthIndexMaKind::Hma => {
            ma_length + ((ma_length as f64).sqrt().floor() as usize).max(1) - 1
        }
    };
    let warmup = first
        + (rsi_length + 1)
            .max(rsi_length + ma_warm)
            .max(rsi_length + range_length + short_hma_warm + long_hma_warm);

    Ok(PreparedInput {
        source,
        volume,
        len,
        params: ResolvedParams {
            rsi_length,
            range_length,
            ma_length,
            ma_kind,
        },
        warmup: warmup.min(len),
    })
}

#[inline]
pub fn volume_weighted_relative_strength_index(
    input: &VolumeWeightedRelativeStrengthIndexInput,
) -> Result<VolumeWeightedRelativeStrengthIndexOutput, VolumeWeightedRelativeStrengthIndexError> {
    volume_weighted_relative_strength_index_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn volume_weighted_relative_strength_index_with_kernel(
    input: &VolumeWeightedRelativeStrengthIndexInput,
    _kernel: Kernel,
) -> Result<VolumeWeightedRelativeStrengthIndexOutput, VolumeWeightedRelativeStrengthIndexError> {
    let prepared = prepare_input(input)?;
    let mut rsi = alloc_uninit_f64(prepared.len);
    let mut consolidation_strength = alloc_uninit_f64(prepared.len);
    let mut rsi_ma = alloc_uninit_f64(prepared.len);
    let mut bearish_tp = alloc_uninit_f64(prepared.len);
    let mut bullish_tp = alloc_uninit_f64(prepared.len);
    volume_weighted_relative_strength_index_write_prepared(
        &prepared,
        &mut rsi,
        &mut consolidation_strength,
        &mut rsi_ma,
        &mut bearish_tp,
        &mut bullish_tp,
    );
    Ok(VolumeWeightedRelativeStrengthIndexOutput {
        rsi,
        consolidation_strength,
        rsi_ma,
        bearish_tp,
        bullish_tp,
    })
}

#[inline]
pub fn volume_weighted_relative_strength_index_into(
    input: &VolumeWeightedRelativeStrengthIndexInput,
    kernel: Kernel,
    output: &mut VolumeWeightedRelativeStrengthIndexOutput,
) -> Result<(), VolumeWeightedRelativeStrengthIndexError> {
    volume_weighted_relative_strength_index_into_slices(
        input,
        kernel,
        &mut output.rsi,
        &mut output.consolidation_strength,
        &mut output.rsi_ma,
        &mut output.bearish_tp,
        &mut output.bullish_tp,
    )
}

pub fn volume_weighted_relative_strength_index_into_slices(
    input: &VolumeWeightedRelativeStrengthIndexInput,
    _kernel: Kernel,
    dst_rsi: &mut [f64],
    dst_consolidation_strength: &mut [f64],
    dst_rsi_ma: &mut [f64],
    dst_bearish_tp: &mut [f64],
    dst_bullish_tp: &mut [f64],
) -> Result<(), VolumeWeightedRelativeStrengthIndexError> {
    let prepared = prepare_input(input)?;
    let got = *[
        dst_rsi.len(),
        dst_consolidation_strength.len(),
        dst_rsi_ma.len(),
        dst_bearish_tp.len(),
        dst_bullish_tp.len(),
    ]
    .iter()
    .min()
    .unwrap_or(&0);
    if dst_rsi.len() != prepared.len
        || dst_consolidation_strength.len() != prepared.len
        || dst_rsi_ma.len() != prepared.len
        || dst_bearish_tp.len() != prepared.len
        || dst_bullish_tp.len() != prepared.len
    {
        return Err(
            VolumeWeightedRelativeStrengthIndexError::OutputLengthMismatch {
                expected: prepared.len,
                got,
            },
        );
    }

    volume_weighted_relative_strength_index_write_prepared(
        &prepared,
        dst_rsi,
        dst_consolidation_strength,
        dst_rsi_ma,
        dst_bearish_tp,
        dst_bullish_tp,
    );

    Ok(())
}

#[inline(always)]
fn volume_weighted_relative_strength_index_write_prepared(
    prepared: &PreparedInput<'_>,
    dst_rsi: &mut [f64],
    dst_consolidation_strength: &mut [f64],
    dst_rsi_ma: &mut [f64],
    dst_bearish_tp: &mut [f64],
    dst_bullish_tp: &mut [f64],
) {
    let mut core = VolumeWeightedRelativeStrengthIndexCore::new(prepared.params);
    for i in 0..prepared.len {
        if let Some(point) = core.update(prepared.source[i], prepared.volume[i]) {
            dst_rsi[i] = point.rsi;
            dst_consolidation_strength[i] = point.consolidation_strength;
            dst_rsi_ma[i] = point.rsi_ma;
            dst_bearish_tp[i] = point.bearish_tp;
            dst_bullish_tp[i] = point.bullish_tp;
        } else {
            dst_rsi[i] = f64::NAN;
            dst_consolidation_strength[i] = f64::NAN;
            dst_rsi_ma[i] = f64::NAN;
            dst_bearish_tp[i] = f64::NAN;
            dst_bullish_tp[i] = f64::NAN;
        }
    }
}

#[inline]
pub(crate) fn volume_weighted_relative_strength_index_output_into_slice(
    input: &VolumeWeightedRelativeStrengthIndexInput,
    _kernel: Kernel,
    field: VolumeWeightedRelativeStrengthIndexOutputField,
    dst: &mut [f64],
) -> Result<(), VolumeWeightedRelativeStrengthIndexError> {
    let prepared = prepare_input(input)?;
    if dst.len() != prepared.len {
        return Err(
            VolumeWeightedRelativeStrengthIndexError::OutputLengthMismatch {
                expected: prepared.len,
                got: dst.len(),
            },
        );
    }

    if matches!(field, VolumeWeightedRelativeStrengthIndexOutputField::Rsi) {
        volume_weighted_relative_strength_index_rsi_only_into(&prepared, dst);
        return Ok(());
    }

    let mut core = VolumeWeightedRelativeStrengthIndexCore::new(prepared.params);
    for i in 0..prepared.len {
        let value = if let Some(point) = core.update(prepared.source[i], prepared.volume[i]) {
            match field {
                VolumeWeightedRelativeStrengthIndexOutputField::Rsi => point.rsi,
                VolumeWeightedRelativeStrengthIndexOutputField::ConsolidationStrength => {
                    point.consolidation_strength
                }
                VolumeWeightedRelativeStrengthIndexOutputField::RsiMa => point.rsi_ma,
                VolumeWeightedRelativeStrengthIndexOutputField::BearishTp => point.bearish_tp,
                VolumeWeightedRelativeStrengthIndexOutputField::BullishTp => point.bullish_tp,
            }
        } else {
            f64::NAN
        };
        dst[i] = value;
    }

    Ok(())
}

#[inline(always)]
fn volume_weighted_relative_strength_index_rsi_only_into(
    prepared: &PreparedInput<'_>,
    dst: &mut [f64],
) {
    let mut prev_source = None;
    let mut up_rma = RmaState::new(prepared.params.rsi_length);
    let mut down_rma = RmaState::new(prepared.params.rsi_length);
    let mut volume_rma = RmaState::new(prepared.params.rsi_length);

    for (i, out) in dst.iter_mut().enumerate().take(prepared.len) {
        let source = prepared.source[i];
        let volume = prepared.volume[i];
        if !source.is_finite() || !volume.is_finite() {
            prev_source = None;
            up_rma.reset();
            down_rma.reset();
            volume_rma.reset();
            *out = f64::NAN;
            continue;
        }

        let Some(prev) = prev_source else {
            prev_source = Some(source);
            *out = f64::NAN;
            continue;
        };
        prev_source = Some(source);

        let delta = source - prev;
        let gain = delta.max(0.0) * volume;
        let loss = (-delta).max(0.0) * volume;
        let up_num = up_rma.update(gain);
        let down_num = down_rma.update(loss);
        let vol_avg = volume_rma.update(volume);
        let (Some(up_num), Some(down_num), Some(vol_avg)) = (up_num, down_num, vol_avg) else {
            *out = f64::NAN;
            continue;
        };
        if vol_avg.abs() <= EPS {
            *out = f64::NAN;
            continue;
        }

        let up = up_num / vol_avg;
        let down = down_num / vol_avg;
        *out = if down.abs() <= EPS {
            100.0
        } else if up.abs() <= EPS {
            0.0
        } else {
            100.0 - (100.0 / (1.0 + up / down))
        };
    }
}

impl VolumeWeightedRelativeStrengthIndexBuilder {
    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<VolumeWeightedRelativeStrengthIndexOutput, VolumeWeightedRelativeStrengthIndexError>
    {
        let input = VolumeWeightedRelativeStrengthIndexInput::from_candles(
            candles,
            self.source.unwrap_or(DEFAULT_SOURCE),
            VolumeWeightedRelativeStrengthIndexParams {
                rsi_length: self.rsi_length,
                range_length: self.range_length,
                ma_length: self.ma_length,
                ma_type: self.ma_type,
            },
        );
        volume_weighted_relative_strength_index_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        source: &[f64],
        volume: &[f64],
    ) -> Result<VolumeWeightedRelativeStrengthIndexOutput, VolumeWeightedRelativeStrengthIndexError>
    {
        let input = VolumeWeightedRelativeStrengthIndexInput::from_slices(
            source,
            volume,
            VolumeWeightedRelativeStrengthIndexParams {
                rsi_length: self.rsi_length,
                range_length: self.range_length,
                ma_length: self.ma_length,
                ma_type: self.ma_type,
            },
        );
        volume_weighted_relative_strength_index_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<VolumeWeightedRelativeStrengthIndexStream, VolumeWeightedRelativeStrengthIndexError>
    {
        VolumeWeightedRelativeStrengthIndexStream::try_new(
            VolumeWeightedRelativeStrengthIndexParams {
                rsi_length: self.rsi_length,
                range_length: self.range_length,
                ma_length: self.ma_length,
                ma_type: self.ma_type,
            },
        )
    }
}

impl VolumeWeightedRelativeStrengthIndexStream {
    pub fn try_new(
        params: VolumeWeightedRelativeStrengthIndexParams,
    ) -> Result<Self, VolumeWeightedRelativeStrengthIndexError> {
        let rsi_length = params.rsi_length.unwrap_or(DEFAULT_RSI_LENGTH);
        let range_length = params.range_length.unwrap_or(DEFAULT_RANGE_LENGTH);
        let ma_length = params.ma_length.unwrap_or(DEFAULT_MA_LENGTH);
        let ma_type = params
            .ma_type
            .unwrap_or_else(|| DEFAULT_MA_TYPE.to_string());
        let ma_kind = normalize_ma_type(&ma_type)?;
        if rsi_length == 0 {
            return Err(VolumeWeightedRelativeStrengthIndexError::InvalidRsiLength {
                rsi_length,
                data_len: 0,
            });
        }
        if range_length == 0 {
            return Err(
                VolumeWeightedRelativeStrengthIndexError::InvalidRangeLength {
                    range_length,
                    data_len: 0,
                },
            );
        }
        if ma_length == 0 {
            return Err(VolumeWeightedRelativeStrengthIndexError::InvalidMaLength {
                ma_length,
                data_len: 0,
            });
        }
        Ok(Self {
            core: VolumeWeightedRelativeStrengthIndexCore::new(ResolvedParams {
                rsi_length,
                range_length,
                ma_length,
                ma_kind,
            }),
        })
    }

    #[inline(always)]
    pub fn update(
        &mut self,
        source: f64,
        volume: f64,
    ) -> Option<VolumeWeightedRelativeStrengthIndexStreamOutput> {
        self.core.update(source, volume).map(|point| {
            VolumeWeightedRelativeStrengthIndexStreamOutput {
                rsi: point.rsi,
                consolidation_strength: point.consolidation_strength,
                rsi_ma: point.rsi_ma,
                bearish_tp: point.bearish_tp,
                bullish_tp: point.bullish_tp,
            }
        })
    }
}

#[derive(Clone, Debug)]
pub struct VolumeWeightedRelativeStrengthIndexBatchRange {
    pub rsi_length: (usize, usize, usize),
    pub range_length: (usize, usize, usize),
    pub ma_length: (usize, usize, usize),
    pub ma_type: String,
}

impl Default for VolumeWeightedRelativeStrengthIndexBatchRange {
    fn default() -> Self {
        Self {
            rsi_length: (DEFAULT_RSI_LENGTH, DEFAULT_RSI_LENGTH, 0),
            range_length: (DEFAULT_RANGE_LENGTH, DEFAULT_RANGE_LENGTH, 0),
            ma_length: (DEFAULT_MA_LENGTH, DEFAULT_MA_LENGTH, 0),
            ma_type: DEFAULT_MA_TYPE.to_string(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct VolumeWeightedRelativeStrengthIndexBatchOutput {
    pub rsi: Vec<f64>,
    pub consolidation_strength: Vec<f64>,
    pub rsi_ma: Vec<f64>,
    pub bearish_tp: Vec<f64>,
    pub bullish_tp: Vec<f64>,
    pub combos: Vec<VolumeWeightedRelativeStrengthIndexParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Debug)]
pub struct VolumeWeightedRelativeStrengthIndexBatchBuilder {
    source: Option<&'static str>,
    range: VolumeWeightedRelativeStrengthIndexBatchRange,
    kernel: Kernel,
}

impl Default for VolumeWeightedRelativeStrengthIndexBatchBuilder {
    fn default() -> Self {
        Self {
            source: None,
            range: VolumeWeightedRelativeStrengthIndexBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl VolumeWeightedRelativeStrengthIndexBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn source(mut self, value: &'static str) -> Self {
        self.source = Some(value);
        self
    }

    #[inline(always)]
    pub fn range(mut self, value: VolumeWeightedRelativeStrengthIndexBatchRange) -> Self {
        self.range = value;
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
    ) -> Result<
        VolumeWeightedRelativeStrengthIndexBatchOutput,
        VolumeWeightedRelativeStrengthIndexError,
    > {
        let source = source_type(candles, self.source.unwrap_or(DEFAULT_SOURCE));
        self.apply_slices(source, candles.volume.as_slice())
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        source: &[f64],
        volume: &[f64],
    ) -> Result<
        VolumeWeightedRelativeStrengthIndexBatchOutput,
        VolumeWeightedRelativeStrengthIndexError,
    > {
        volume_weighted_relative_strength_index_batch_with_kernel(
            source,
            volume,
            &self.range,
            self.kernel,
        )
    }
}

fn axis_usize(
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, VolumeWeightedRelativeStrengthIndexError> {
    if step == 0 || start == end {
        return Ok(vec![start]);
    }
    let mut out = Vec::new();
    if start <= end {
        let mut current = start;
        while current <= end {
            out.push(current);
            match current.checked_add(step) {
                Some(next) => current = next,
                None => break,
            }
        }
    } else {
        let mut current = start;
        while current >= end {
            out.push(current);
            match current.checked_sub(step) {
                Some(next) => current = next,
                None => break,
            }
            if current < end {
                break;
            }
        }
    }
    if out.is_empty() {
        return Err(VolumeWeightedRelativeStrengthIndexError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

fn expand_grid(
    range: &VolumeWeightedRelativeStrengthIndexBatchRange,
) -> Result<Vec<VolumeWeightedRelativeStrengthIndexParams>, VolumeWeightedRelativeStrengthIndexError>
{
    normalize_ma_type(&range.ma_type)?;
    let rsi_lengths = axis_usize(range.rsi_length)?;
    let range_lengths = axis_usize(range.range_length)?;
    let ma_lengths = axis_usize(range.ma_length)?;
    let total = rsi_lengths
        .len()
        .checked_mul(range_lengths.len())
        .and_then(|value| value.checked_mul(ma_lengths.len()))
        .ok_or_else(|| VolumeWeightedRelativeStrengthIndexError::InvalidRange {
            start: range.rsi_length.0.to_string(),
            end: range.rsi_length.1.to_string(),
            step: range.rsi_length.2.to_string(),
        })?;

    let mut out = Vec::with_capacity(total);
    for &rsi_length in &rsi_lengths {
        for &range_length in &range_lengths {
            for &ma_length in &ma_lengths {
                out.push(VolumeWeightedRelativeStrengthIndexParams {
                    rsi_length: Some(rsi_length),
                    range_length: Some(range_length),
                    ma_length: Some(ma_length),
                    ma_type: Some(range.ma_type.clone()),
                });
            }
        }
    }
    Ok(out)
}

#[inline]
pub fn volume_weighted_relative_strength_index_batch_with_kernel(
    source: &[f64],
    volume: &[f64],
    range: &VolumeWeightedRelativeStrengthIndexBatchRange,
    kernel: Kernel,
) -> Result<VolumeWeightedRelativeStrengthIndexBatchOutput, VolumeWeightedRelativeStrengthIndexError>
{
    if source.is_empty() || volume.is_empty() {
        return Err(VolumeWeightedRelativeStrengthIndexError::EmptyInputData);
    }
    if source.len() != volume.len() {
        return Err(
            VolumeWeightedRelativeStrengthIndexError::DataLengthMismatch {
                source_len: source.len(),
                volume_len: volume.len(),
            },
        );
    }

    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        value if value.is_batch() => value,
        _ => return Err(VolumeWeightedRelativeStrengthIndexError::InvalidKernelForBatch(kernel)),
    };
    let single_kernel = batch_kernel.to_non_batch();
    let combos = expand_grid(range)?;
    let rows = combos.len();
    let cols = source.len();

    let first = (0..cols)
        .find(|&i| source[i].is_finite() && volume[i].is_finite())
        .ok_or(VolumeWeightedRelativeStrengthIndexError::AllValuesNaN)?;
    let warmups: Vec<usize> = combos
        .iter()
        .map(|combo| {
            let rsi_length = combo.rsi_length.unwrap_or(DEFAULT_RSI_LENGTH);
            let range_length = combo.range_length.unwrap_or(DEFAULT_RANGE_LENGTH);
            let ma_length = combo.ma_length.unwrap_or(DEFAULT_MA_LENGTH);
            let ma_kind = normalize_ma_type(combo.ma_type.as_deref().unwrap_or(DEFAULT_MA_TYPE))
                .unwrap_or(VolumeWeightedRelativeStrengthIndexMaKind::Ema);
            let short = (range_length / 2).max(1);
            let short_hma_warm = short + ((short as f64).sqrt().floor() as usize).max(1) - 1;
            let long = range_length.saturating_mul(2).max(1);
            let long_hma_warm = long + ((long as f64).sqrt().floor() as usize).max(1) - 1;
            let ma_warm = match ma_kind {
                VolumeWeightedRelativeStrengthIndexMaKind::Ema => 1,
                VolumeWeightedRelativeStrengthIndexMaKind::Sma
                | VolumeWeightedRelativeStrengthIndexMaKind::Rma
                | VolumeWeightedRelativeStrengthIndexMaKind::Wma
                | VolumeWeightedRelativeStrengthIndexMaKind::Vwma => ma_length,
                VolumeWeightedRelativeStrengthIndexMaKind::Hma => {
                    ma_length + ((ma_length as f64).sqrt().floor() as usize).max(1) - 1
                }
            };
            (first
                + (rsi_length + 1)
                    .max(rsi_length + ma_warm)
                    .max(rsi_length + range_length + short_hma_warm + long_hma_warm))
            .min(cols)
        })
        .collect();

    let mut rsi_mu = make_uninit_matrix(rows, cols);
    let mut consolidation_mu = make_uninit_matrix(rows, cols);
    let mut ma_mu = make_uninit_matrix(rows, cols);
    let mut bearish_mu = make_uninit_matrix(rows, cols);
    let mut bullish_mu = make_uninit_matrix(rows, cols);

    init_matrix_prefixes(&mut rsi_mu, cols, &warmups);
    init_matrix_prefixes(&mut consolidation_mu, cols, &warmups);
    init_matrix_prefixes(&mut ma_mu, cols, &warmups);
    init_matrix_prefixes(&mut bearish_mu, cols, &warmups);
    init_matrix_prefixes(&mut bullish_mu, cols, &warmups);

    let mut rsi_guard = ManuallyDrop::new(rsi_mu);
    let mut consolidation_guard = ManuallyDrop::new(consolidation_mu);
    let mut ma_guard = ManuallyDrop::new(ma_mu);
    let mut bearish_guard = ManuallyDrop::new(bearish_mu);
    let mut bullish_guard = ManuallyDrop::new(bullish_mu);

    let rsi_all = unsafe { mu_slice_as_f64_slice_mut(&mut rsi_guard) };
    let consolidation_all = unsafe { mu_slice_as_f64_slice_mut(&mut consolidation_guard) };
    let ma_all = unsafe { mu_slice_as_f64_slice_mut(&mut ma_guard) };
    let bearish_all = unsafe { mu_slice_as_f64_slice_mut(&mut bearish_guard) };
    let bullish_all = unsafe { mu_slice_as_f64_slice_mut(&mut bullish_guard) };

    let run_row = |row: usize,
                   rsi_row: &mut [f64],
                   consolidation_row: &mut [f64],
                   ma_row: &mut [f64],
                   bearish_row: &mut [f64],
                   bullish_row: &mut [f64]|
     -> Result<(), VolumeWeightedRelativeStrengthIndexError> {
        let input = VolumeWeightedRelativeStrengthIndexInput::from_slices(
            source,
            volume,
            combos[row].clone(),
        );
        volume_weighted_relative_strength_index_into_slices(
            &input,
            single_kernel,
            rsi_row,
            consolidation_row,
            ma_row,
            bearish_row,
            bullish_row,
        )
    };

    #[cfg(not(target_arch = "wasm32"))]
    {
        rsi_all
            .par_chunks_mut(cols)
            .zip(consolidation_all.par_chunks_mut(cols))
            .zip(ma_all.par_chunks_mut(cols))
            .zip(bearish_all.par_chunks_mut(cols))
            .zip(bullish_all.par_chunks_mut(cols))
            .enumerate()
            .try_for_each(
                |(row, ((((rsi_row, consolidation_row), ma_row), bearish_row), bullish_row))| {
                    run_row(
                        row,
                        rsi_row,
                        consolidation_row,
                        ma_row,
                        bearish_row,
                        bullish_row,
                    )
                },
            )?;
    }

    #[cfg(target_arch = "wasm32")]
    {
        for row in 0..rows {
            let start = row * cols;
            let end = start + cols;
            run_row(
                row,
                &mut rsi_all[start..end],
                &mut consolidation_all[start..end],
                &mut ma_all[start..end],
                &mut bearish_all[start..end],
                &mut bullish_all[start..end],
            )?;
        }
    }

    Ok(VolumeWeightedRelativeStrengthIndexBatchOutput {
        rsi: unsafe { vec_f64_from_mu_guard(rsi_guard) },
        consolidation_strength: unsafe { vec_f64_from_mu_guard(consolidation_guard) },
        rsi_ma: unsafe { vec_f64_from_mu_guard(ma_guard) },
        bearish_tp: unsafe { vec_f64_from_mu_guard(bearish_guard) },
        bullish_tp: unsafe { vec_f64_from_mu_guard(bullish_guard) },
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn mu_slice_as_f64_slice_mut(buf: &mut ManuallyDrop<Vec<MaybeUninit<f64>>>) -> &mut [f64] {
    core::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut f64, buf.len())
}

#[inline(always)]
unsafe fn vec_f64_from_mu_guard(buf: ManuallyDrop<Vec<MaybeUninit<f64>>>) -> Vec<f64> {
    let mut buf = buf;
    Vec::from_raw_parts(buf.as_mut_ptr() as *mut f64, buf.len(), buf.capacity())
}

#[cfg(feature = "python")]
#[pyfunction(name = "volume_weighted_relative_strength_index")]
#[pyo3(signature = (
    source,
    volume,
    rsi_length=DEFAULT_RSI_LENGTH,
    range_length=DEFAULT_RANGE_LENGTH,
    ma_length=DEFAULT_MA_LENGTH,
    ma_type=DEFAULT_MA_TYPE,
    kernel=None
))]
pub fn volume_weighted_relative_strength_index_py<'py>(
    py: Python<'py>,
    source: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    rsi_length: usize,
    range_length: usize,
    ma_length: usize,
    ma_type: &str,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let source = source.as_slice()?;
    let volume = volume.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = VolumeWeightedRelativeStrengthIndexInput::from_slices(
        source,
        volume,
        VolumeWeightedRelativeStrengthIndexParams {
            rsi_length: Some(rsi_length),
            range_length: Some(range_length),
            ma_length: Some(ma_length),
            ma_type: Some(ma_type.to_string()),
        },
    );
    let output = py
        .allow_threads(|| volume_weighted_relative_strength_index_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dict = PyDict::new(py);
    dict.set_item("rsi", output.rsi.into_pyarray(py))?;
    dict.set_item(
        "consolidation_strength",
        output.consolidation_strength.into_pyarray(py),
    )?;
    dict.set_item("rsi_ma", output.rsi_ma.into_pyarray(py))?;
    dict.set_item("bearish_tp", output.bearish_tp.into_pyarray(py))?;
    dict.set_item("bullish_tp", output.bullish_tp.into_pyarray(py))?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "VolumeWeightedRelativeStrengthIndexStream")]
pub struct VolumeWeightedRelativeStrengthIndexStreamPy {
    stream: VolumeWeightedRelativeStrengthIndexStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl VolumeWeightedRelativeStrengthIndexStreamPy {
    #[new]
    #[pyo3(signature = (
        rsi_length=DEFAULT_RSI_LENGTH,
        range_length=DEFAULT_RANGE_LENGTH,
        ma_length=DEFAULT_MA_LENGTH,
        ma_type=DEFAULT_MA_TYPE
    ))]
    fn new(
        rsi_length: usize,
        range_length: usize,
        ma_length: usize,
        ma_type: &str,
    ) -> PyResult<Self> {
        let stream = VolumeWeightedRelativeStrengthIndexStream::try_new(
            VolumeWeightedRelativeStrengthIndexParams {
                rsi_length: Some(rsi_length),
                range_length: Some(range_length),
                ma_length: Some(ma_length),
                ma_type: Some(ma_type.to_string()),
            },
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, source: f64, volume: f64) -> Option<(f64, f64, f64, f64, f64)> {
        self.stream.update(source, volume).map(|output| {
            (
                output.rsi,
                output.consolidation_strength,
                output.rsi_ma,
                output.bearish_tp,
                output.bullish_tp,
            )
        })
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "volume_weighted_relative_strength_index_batch")]
#[pyo3(signature = (
    source,
    volume,
    rsi_length_range=(DEFAULT_RSI_LENGTH, DEFAULT_RSI_LENGTH, 0),
    range_length_range=(DEFAULT_RANGE_LENGTH, DEFAULT_RANGE_LENGTH, 0),
    ma_length_range=(DEFAULT_MA_LENGTH, DEFAULT_MA_LENGTH, 0),
    ma_type=DEFAULT_MA_TYPE,
    kernel=None
))]
pub fn volume_weighted_relative_strength_index_batch_py<'py>(
    py: Python<'py>,
    source: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    rsi_length_range: (usize, usize, usize),
    range_length_range: (usize, usize, usize),
    ma_length_range: (usize, usize, usize),
    ma_type: &str,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let source = source.as_slice()?;
    let volume = volume.as_slice()?;
    let kernel = validate_kernel(kernel, true)?;
    let output = py
        .allow_threads(|| {
            volume_weighted_relative_strength_index_batch_with_kernel(
                source,
                volume,
                &VolumeWeightedRelativeStrengthIndexBatchRange {
                    rsi_length: rsi_length_range,
                    range_length: range_length_range,
                    ma_length: ma_length_range,
                    ma_type: ma_type.to_string(),
                },
                kernel,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let total = output.rows * output.cols;
    let out_rsi = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_consolidation = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_ma = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_bearish = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_bullish = unsafe { PyArray1::<f64>::new(py, [total], false) };
    unsafe { out_rsi.as_slice_mut()? }.copy_from_slice(&output.rsi);
    unsafe { out_consolidation.as_slice_mut()? }.copy_from_slice(&output.consolidation_strength);
    unsafe { out_ma.as_slice_mut()? }.copy_from_slice(&output.rsi_ma);
    unsafe { out_bearish.as_slice_mut()? }.copy_from_slice(&output.bearish_tp);
    unsafe { out_bullish.as_slice_mut()? }.copy_from_slice(&output.bullish_tp);

    let dict = PyDict::new(py);
    dict.set_item("rsi", out_rsi.reshape((output.rows, output.cols))?)?;
    dict.set_item(
        "consolidation_strength",
        out_consolidation.reshape((output.rows, output.cols))?,
    )?;
    dict.set_item("rsi_ma", out_ma.reshape((output.rows, output.cols))?)?;
    dict.set_item(
        "bearish_tp",
        out_bearish.reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "bullish_tp",
        out_bullish.reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "rsi_lengths",
        output
            .combos
            .iter()
            .map(|combo| combo.rsi_length.unwrap_or(DEFAULT_RSI_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "range_lengths",
        output
            .combos
            .iter()
            .map(|combo| combo.range_length.unwrap_or(DEFAULT_RANGE_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "ma_lengths",
        output
            .combos
            .iter()
            .map(|combo| combo.ma_length.unwrap_or(DEFAULT_MA_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "ma_types",
        output
            .combos
            .iter()
            .map(|combo| {
                combo
                    .ma_type
                    .clone()
                    .unwrap_or_else(|| DEFAULT_MA_TYPE.to_string())
            })
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("rows", output.rows)?;
    dict.set_item("cols", output.cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_volume_weighted_relative_strength_index_module(
    m: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        volume_weighted_relative_strength_index_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        volume_weighted_relative_strength_index_batch_py,
        m
    )?)?;
    m.add_class::<VolumeWeightedRelativeStrengthIndexStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct VolumeWeightedRelativeStrengthIndexJsOutput {
    pub rsi: Vec<f64>,
    pub consolidation_strength: Vec<f64>,
    pub rsi_ma: Vec<f64>,
    pub bearish_tp: Vec<f64>,
    pub bullish_tp: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = volume_weighted_relative_strength_index_js)]
pub fn volume_weighted_relative_strength_index_js(
    source: &[f64],
    volume: &[f64],
    rsi_length: usize,
    range_length: usize,
    ma_length: usize,
    ma_type: &str,
) -> Result<JsValue, JsValue> {
    let input = VolumeWeightedRelativeStrengthIndexInput::from_slices(
        source,
        volume,
        VolumeWeightedRelativeStrengthIndexParams {
            rsi_length: Some(rsi_length),
            range_length: Some(range_length),
            ma_length: Some(ma_length),
            ma_type: Some(ma_type.to_string()),
        },
    );
    let output = volume_weighted_relative_strength_index_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&VolumeWeightedRelativeStrengthIndexJsOutput {
        rsi: output.rsi,
        consolidation_strength: output.consolidation_strength,
        rsi_ma: output.rsi_ma,
        bearish_tp: output.bearish_tp,
        bullish_tp: output.bullish_tp,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct VolumeWeightedRelativeStrengthIndexBatchConfig {
    pub rsi_length_range: (usize, usize, usize),
    pub range_length_range: (usize, usize, usize),
    pub ma_length_range: (usize, usize, usize),
    pub ma_type: String,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct VolumeWeightedRelativeStrengthIndexBatchJsOutput {
    pub rsi: Vec<f64>,
    pub consolidation_strength: Vec<f64>,
    pub rsi_ma: Vec<f64>,
    pub bearish_tp: Vec<f64>,
    pub bullish_tp: Vec<f64>,
    pub rsi_lengths: Vec<usize>,
    pub range_lengths: Vec<usize>,
    pub ma_lengths: Vec<usize>,
    pub ma_types: Vec<String>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = volume_weighted_relative_strength_index_batch)]
pub fn volume_weighted_relative_strength_index_batch_js(
    source: &[f64],
    volume: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let cfg: VolumeWeightedRelativeStrengthIndexBatchConfig =
        serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let output = volume_weighted_relative_strength_index_batch_with_kernel(
        source,
        volume,
        &VolumeWeightedRelativeStrengthIndexBatchRange {
            rsi_length: cfg.rsi_length_range,
            range_length: cfg.range_length_range,
            ma_length: cfg.ma_length_range,
            ma_type: cfg.ma_type,
        },
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    serde_wasm_bindgen::to_value(&VolumeWeightedRelativeStrengthIndexBatchJsOutput {
        rsi: output.rsi,
        consolidation_strength: output.consolidation_strength,
        rsi_ma: output.rsi_ma,
        bearish_tp: output.bearish_tp,
        bullish_tp: output.bullish_tp,
        rsi_lengths: output
            .combos
            .iter()
            .map(|combo| combo.rsi_length.unwrap_or(DEFAULT_RSI_LENGTH))
            .collect(),
        range_lengths: output
            .combos
            .iter()
            .map(|combo| combo.range_length.unwrap_or(DEFAULT_RANGE_LENGTH))
            .collect(),
        ma_lengths: output
            .combos
            .iter()
            .map(|combo| combo.ma_length.unwrap_or(DEFAULT_MA_LENGTH))
            .collect(),
        ma_types: output
            .combos
            .iter()
            .map(|combo| {
                combo
                    .ma_type
                    .clone()
                    .unwrap_or_else(|| DEFAULT_MA_TYPE.to_string())
            })
            .collect(),
        rows: output.rows,
        cols: output.cols,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn volume_weighted_relative_strength_index_output_into_js(
    source: &[f64],
    volume: &[f64],
    rsi_length: usize,
    range_length: usize,
    ma_length: usize,
    ma_type: &str,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = volume_weighted_relative_strength_index_js(
        source,
        volume,
        rsi_length,
        range_length,
        ma_length,
        ma_type,
    )?;
    crate::write_wasm_object_f64_outputs(
        "volume_weighted_relative_strength_index_output_into_js",
        &value,
        out,
    )
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn volume_weighted_relative_strength_index_batch_output_into_js(
    source: &[f64],
    volume: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = volume_weighted_relative_strength_index_batch_js(source, volume, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "volume_weighted_relative_strength_index_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_close_volume() -> (Vec<f64>, Vec<f64>) {
        let mut close = Vec::with_capacity(320);
        let mut volume = Vec::with_capacity(320);
        for i in 0..320 {
            let c = 100.0 + i as f64 * 0.12 + (i as f64 * 0.17).sin() * 2.1;
            let v = 1_000.0 + (i as f64 * 0.11).cos().abs() * 400.0 + i as f64 * 1.5;
            close.push(c);
            volume.push(v);
        }
        (close, volume)
    }

    #[test]
    fn volume_weighted_relative_strength_index_into_matches_single() {
        let (source, volume) = sample_close_volume();
        let input = VolumeWeightedRelativeStrengthIndexInput::from_slices(
            &source,
            &volume,
            VolumeWeightedRelativeStrengthIndexParams::default(),
        );
        let out = volume_weighted_relative_strength_index(&input).expect("single");
        let mut rsi = vec![0.0; source.len()];
        let mut consolidation = vec![0.0; source.len()];
        let mut rsi_ma = vec![0.0; source.len()];
        let mut bearish = vec![0.0; source.len()];
        let mut bullish = vec![0.0; source.len()];
        volume_weighted_relative_strength_index_into_slices(
            &input,
            Kernel::Scalar,
            &mut rsi,
            &mut consolidation,
            &mut rsi_ma,
            &mut bearish,
            &mut bullish,
        )
        .expect("into");

        for i in 0..source.len() {
            for (lhs, rhs) in [
                (out.rsi[i], rsi[i]),
                (out.consolidation_strength[i], consolidation[i]),
                (out.rsi_ma[i], rsi_ma[i]),
                (out.bearish_tp[i], bearish[i]),
                (out.bullish_tp[i], bullish[i]),
            ] {
                if lhs.is_nan() {
                    assert!(rhs.is_nan());
                } else {
                    assert!((lhs - rhs).abs() <= 1e-12);
                }
            }
        }
    }

    #[test]
    fn volume_weighted_relative_strength_index_stream_matches_batch() {
        let (source, volume) = sample_close_volume();
        let batch = volume_weighted_relative_strength_index(
            &VolumeWeightedRelativeStrengthIndexInput::from_slices(
                &source,
                &volume,
                VolumeWeightedRelativeStrengthIndexParams::default(),
            ),
        )
        .expect("batch");
        let mut stream = VolumeWeightedRelativeStrengthIndexStream::try_new(
            VolumeWeightedRelativeStrengthIndexParams::default(),
        )
        .expect("stream");
        let mut collected = Vec::with_capacity(source.len());
        for i in 0..source.len() {
            collected.push(stream.update(source[i], volume[i]));
        }
        for i in 0..source.len() {
            let Some(point) = collected[i] else {
                assert!(batch.rsi[i].is_nan());
                continue;
            };
            for (lhs, rhs) in [
                (point.rsi, batch.rsi[i]),
                (
                    point.consolidation_strength,
                    batch.consolidation_strength[i],
                ),
                (point.rsi_ma, batch.rsi_ma[i]),
                (point.bearish_tp, batch.bearish_tp[i]),
                (point.bullish_tp, batch.bullish_tp[i]),
            ] {
                if rhs.is_nan() {
                    assert!(lhs.is_nan());
                } else {
                    assert!((lhs - rhs).abs() <= 1e-12);
                }
            }
        }
    }

    #[test]
    fn volume_weighted_relative_strength_index_batch_first_row_matches_single() {
        let (source, volume) = sample_close_volume();
        let single = volume_weighted_relative_strength_index(
            &VolumeWeightedRelativeStrengthIndexInput::from_slices(
                &source,
                &volume,
                VolumeWeightedRelativeStrengthIndexParams::default(),
            ),
        )
        .expect("single");
        let batch = volume_weighted_relative_strength_index_batch_with_kernel(
            &source,
            &volume,
            &VolumeWeightedRelativeStrengthIndexBatchRange {
                rsi_length: (14, 16, 2),
                range_length: (10, 10, 0),
                ma_length: (14, 14, 0),
                ma_type: "EMA".to_string(),
            },
            Kernel::ScalarBatch,
        )
        .expect("batch");
        assert_eq!(batch.rows, 2);
        assert_eq!(batch.cols, source.len());
        for i in 0..source.len() {
            let idx = i;
            for (lhs, rhs) in [
                (single.rsi[i], batch.rsi[idx]),
                (
                    single.consolidation_strength[i],
                    batch.consolidation_strength[idx],
                ),
                (single.rsi_ma[i], batch.rsi_ma[idx]),
                (single.bearish_tp[i], batch.bearish_tp[idx]),
                (single.bullish_tp[i], batch.bullish_tp[idx]),
            ] {
                if lhs.is_nan() {
                    assert!(rhs.is_nan());
                } else {
                    assert!((lhs - rhs).abs() <= 1e-12);
                }
            }
        }
    }

    #[test]
    fn volume_weighted_relative_strength_index_rejects_invalid_params() {
        let (source, volume) = sample_close_volume();
        let err = volume_weighted_relative_strength_index(
            &VolumeWeightedRelativeStrengthIndexInput::from_slices(
                &source,
                &volume,
                VolumeWeightedRelativeStrengthIndexParams {
                    rsi_length: Some(0),
                    range_length: Some(10),
                    ma_length: Some(14),
                    ma_type: Some("EMA".to_string()),
                },
            ),
        )
        .expect_err("invalid rsi_length");
        assert!(err.to_string().contains("invalid rsi_length"));

        let err = VolumeWeightedRelativeStrengthIndexStream::try_new(
            VolumeWeightedRelativeStrengthIndexParams {
                rsi_length: Some(14),
                range_length: Some(10),
                ma_length: Some(14),
                ma_type: Some("BAD".to_string()),
            },
        )
        .expect_err("invalid ma_type");
        assert!(err.to_string().contains("invalid ma_type"));
    }
}
