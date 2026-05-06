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
use std::collections::VecDeque;
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

const DEFAULT_SWING_SIZE: usize = 10;
const DEFAULT_BOS_CONFIRMATION: &str = "Candle Close";
const DEFAULT_BASIS_LENGTH: usize = 100;
const DEFAULT_ATR_LENGTH: usize = 14;
const DEFAULT_ATR_SMOOTH: usize = 21;
const DEFAULT_VOL_MULT: f64 = 2.0;
const EPS: f64 = 1e-12;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketStructureConfluenceBosConfirmation {
    CandleClose,
    Wicks,
}

impl MarketStructureConfluenceBosConfirmation {
    #[inline(always)]
    fn parse(value: &str) -> Option<Self> {
        match value {
            "Candle Close" | "candle_close" | "candle close" => Some(Self::CandleClose),
            "Wicks" | "wicks" => Some(Self::Wicks),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum MarketStructureConfluenceData<'a> {
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
pub struct MarketStructureConfluenceOutput {
    pub basis: Vec<f64>,
    pub upper_band: Vec<f64>,
    pub lower_band: Vec<f64>,
    pub structure_direction: Vec<f64>,
    pub bullish_arrow: Vec<f64>,
    pub bearish_arrow: Vec<f64>,
    pub bullish_change: Vec<f64>,
    pub bearish_change: Vec<f64>,
    pub hh: Vec<f64>,
    pub lh: Vec<f64>,
    pub hl: Vec<f64>,
    pub ll: Vec<f64>,
    pub bullish_bos: Vec<f64>,
    pub bullish_choch: Vec<f64>,
    pub bearish_bos: Vec<f64>,
    pub bearish_choch: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct MarketStructureConfluenceParams {
    pub swing_size: Option<usize>,
    pub bos_confirmation: Option<String>,
    pub basis_length: Option<usize>,
    pub atr_length: Option<usize>,
    pub atr_smooth: Option<usize>,
    pub vol_mult: Option<f64>,
}

impl Default for MarketStructureConfluenceParams {
    fn default() -> Self {
        Self {
            swing_size: Some(DEFAULT_SWING_SIZE),
            bos_confirmation: Some(DEFAULT_BOS_CONFIRMATION.to_string()),
            basis_length: Some(DEFAULT_BASIS_LENGTH),
            atr_length: Some(DEFAULT_ATR_LENGTH),
            atr_smooth: Some(DEFAULT_ATR_SMOOTH),
            vol_mult: Some(DEFAULT_VOL_MULT),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MarketStructureConfluenceInput<'a> {
    pub data: MarketStructureConfluenceData<'a>,
    pub params: MarketStructureConfluenceParams,
}

impl<'a> MarketStructureConfluenceInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: MarketStructureConfluenceParams) -> Self {
        Self {
            data: MarketStructureConfluenceData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: MarketStructureConfluenceParams,
    ) -> Self {
        Self {
            data: MarketStructureConfluenceData::Slices { high, low, close },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, MarketStructureConfluenceParams::default())
    }
}

#[derive(Clone, Debug)]
pub struct MarketStructureConfluenceBuilder {
    swing_size: Option<usize>,
    bos_confirmation: Option<String>,
    basis_length: Option<usize>,
    atr_length: Option<usize>,
    atr_smooth: Option<usize>,
    vol_mult: Option<f64>,
    kernel: Kernel,
}

impl Default for MarketStructureConfluenceBuilder {
    fn default() -> Self {
        Self {
            swing_size: None,
            bos_confirmation: None,
            basis_length: None,
            atr_length: None,
            atr_smooth: None,
            vol_mult: None,
            kernel: Kernel::Auto,
        }
    }
}

impl MarketStructureConfluenceBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn swing_size(mut self, value: usize) -> Self {
        self.swing_size = Some(value);
        self
    }

    #[inline(always)]
    pub fn bos_confirmation<S: Into<String>>(mut self, value: S) -> Self {
        self.bos_confirmation = Some(value.into());
        self
    }

    #[inline(always)]
    pub fn basis_length(mut self, value: usize) -> Self {
        self.basis_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn atr_length(mut self, value: usize) -> Self {
        self.atr_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn atr_smooth(mut self, value: usize) -> Self {
        self.atr_smooth = Some(value);
        self
    }

    #[inline(always)]
    pub fn vol_mult(mut self, value: f64) -> Self {
        self.vol_mult = Some(value);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }
}

#[derive(Debug, Error)]
pub enum MarketStructureConfluenceError {
    #[error("market_structure_confluence: input data slice is empty")]
    EmptyInputData,
    #[error(
        "market_structure_confluence: data length mismatch: high={high}, low={low}, close={close}"
    )]
    DataLengthMismatch {
        high: usize,
        low: usize,
        close: usize,
    },
    #[error("market_structure_confluence: all values are NaN")]
    AllValuesNaN,
    #[error("market_structure_confluence: invalid swing_size: swing_size = {swing_size}, data length = {data_len}")]
    InvalidSwingSize { swing_size: usize, data_len: usize },
    #[error("market_structure_confluence: invalid bos_confirmation: {bos_confirmation}")]
    InvalidBosConfirmation { bos_confirmation: String },
    #[error("market_structure_confluence: invalid basis_length: basis_length = {basis_length}, data length = {data_len}")]
    InvalidBasisLength {
        basis_length: usize,
        data_len: usize,
    },
    #[error("market_structure_confluence: invalid atr_length: atr_length = {atr_length}, data length = {data_len}")]
    InvalidAtrLength { atr_length: usize, data_len: usize },
    #[error("market_structure_confluence: invalid atr_smooth: atr_smooth = {atr_smooth}, data length = {data_len}")]
    InvalidAtrSmooth { atr_smooth: usize, data_len: usize },
    #[error("market_structure_confluence: invalid vol_mult: {vol_mult}")]
    InvalidVolMult { vol_mult: f64 },
    #[error(
        "market_structure_confluence: not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("market_structure_confluence: output length mismatch: expected {expected}, got {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("market_structure_confluence: invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("market_structure_confluence: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Clone, Copy, Debug)]
struct ResolvedParams {
    swing_size: usize,
    bos_confirmation: MarketStructureConfluenceBosConfirmation,
    basis_length: usize,
    atr_length: usize,
    atr_smooth: usize,
    vol_mult: f64,
}

#[derive(Clone, Debug)]
struct PreparedInput<'a> {
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    len: usize,
    params: ResolvedParams,
    first: usize,
    valid: usize,
    warmup: usize,
}

#[derive(Clone, Copy, Debug)]
struct MarketStructureConfluencePoint {
    basis: f64,
    upper_band: f64,
    lower_band: f64,
    structure_direction: f64,
    bullish_arrow: f64,
    bearish_arrow: f64,
    bullish_change: f64,
    bearish_change: f64,
    hh: f64,
    lh: f64,
    hl: f64,
    ll: f64,
    bullish_bos: f64,
    bullish_choch: f64,
    bearish_bos: f64,
    bearish_choch: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MarketStructureConfluenceStreamOutput {
    pub basis: f64,
    pub upper_band: f64,
    pub lower_band: f64,
    pub structure_direction: f64,
    pub bullish_arrow: f64,
    pub bearish_arrow: f64,
    pub bullish_change: f64,
    pub bearish_change: f64,
    pub hh: f64,
    pub lh: f64,
    pub hl: f64,
    pub ll: f64,
    pub bullish_bos: f64,
    pub bullish_choch: f64,
    pub bearish_bos: f64,
    pub bearish_choch: f64,
}

#[derive(Clone, Debug)]
struct AtrState {
    period: usize,
    count: usize,
    sum: f64,
    value: Option<f64>,
    prev_close: Option<f64>,
}

impl AtrState {
    #[inline(always)]
    fn new(period: usize) -> Self {
        Self {
            period,
            count: 0,
            sum: 0.0,
            value: None,
            prev_close: None,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.count = 0;
        self.sum = 0.0;
        self.value = None;
        self.prev_close = None;
    }

    #[inline(always)]
    fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        let tr = if let Some(prev_close) = self.prev_close {
            let hl = high - low;
            let hc = (high - prev_close).abs();
            let lc = (low - prev_close).abs();
            hl.max(hc).max(lc)
        } else {
            high - low
        };
        self.prev_close = Some(close);
        if let Some(prev) = self.value {
            let next = ((prev * (self.period as f64 - 1.0)) + tr) / self.period as f64;
            self.value = Some(next);
            Some(next)
        } else {
            self.count += 1;
            self.sum += tr;
            if self.count == self.period {
                let seeded = self.sum / self.period as f64;
                self.value = Some(seeded);
                Some(seeded)
            } else {
                None
            }
        }
    }
}

#[derive(Clone, Debug)]
struct RollingSma {
    period: usize,
    buffer: Vec<f64>,
    head: usize,
    len: usize,
    sum: f64,
}

impl RollingSma {
    #[inline(always)]
    fn new(period: usize) -> Self {
        Self {
            period,
            buffer: vec![0.0; period],
            head: 0,
            len: 0,
            sum: 0.0,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.head = 0;
        self.len = 0;
        self.sum = 0.0;
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<f64> {
        if self.len < self.period {
            self.buffer[self.len] = value;
            self.len += 1;
            self.sum += value;
            if self.len < self.period {
                None
            } else {
                Some(self.sum / self.period as f64)
            }
        } else {
            let old = self.buffer[self.head];
            self.buffer[self.head] = value;
            self.sum += value - old;
            self.head += 1;
            if self.head == self.period {
                self.head = 0;
            }
            Some(self.sum / self.period as f64)
        }
    }
}

#[derive(Clone, Debug)]
struct WmaState {
    period: usize,
    buffer: Vec<f64>,
    pos: usize,
    len: usize,
    sum: f64,
    weighted_sum: f64,
    divisor: f64,
}

impl WmaState {
    #[inline(always)]
    fn new(period: usize) -> Self {
        Self {
            period,
            buffer: vec![0.0; period],
            pos: 0,
            len: 0,
            sum: 0.0,
            weighted_sum: 0.0,
            divisor: (period as f64) * (period as f64 + 1.0) * 0.5,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.pos = 0;
        self.len = 0;
        self.sum = 0.0;
        self.weighted_sum = 0.0;
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<f64> {
        if self.len < self.period {
            self.buffer[self.pos] = value;
            self.pos = (self.pos + 1) % self.period;
            self.len += 1;
            self.sum += value;
            self.weighted_sum += self.len as f64 * value;
            if self.len == self.period {
                Some(self.weighted_sum / self.divisor)
            } else {
                None
            }
        } else {
            let old = self.buffer[self.pos];
            let old_sum = self.sum;
            self.buffer[self.pos] = value;
            self.pos = (self.pos + 1) % self.period;
            self.weighted_sum = self.weighted_sum - old_sum + self.period as f64 * value;
            self.sum = old_sum - old + value;
            Some(self.weighted_sum / self.divisor)
        }
    }
}

#[derive(Clone, Debug)]
struct PivotDetector {
    period: usize,
    values: VecDeque<(f64, usize)>,
    is_high: bool,
}

impl PivotDetector {
    #[inline(always)]
    fn new(period: usize, is_high: bool) -> Self {
        Self {
            period,
            values: VecDeque::with_capacity(period * 2 + 1),
            is_high,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.values.clear();
    }

    #[inline(always)]
    fn update(&mut self, value: f64, index: usize) -> Option<(f64, usize)> {
        self.values.push_back((value, index));
        let needed = self.period * 2 + 1;
        if self.values.len() < needed {
            return None;
        }
        let (center_value, center_index) = self.values[self.period];
        let mut ok = center_value.is_finite();
        if ok {
            for (i, (other, _)) in self.values.iter().enumerate() {
                if i == self.period {
                    continue;
                }
                if !other.is_finite() {
                    ok = false;
                    break;
                }
                if self.is_high {
                    if *other > center_value {
                        ok = false;
                        break;
                    }
                } else if *other < center_value {
                    ok = false;
                    break;
                }
            }
        }
        self.values.pop_front();
        if ok {
            Some((center_value, center_index))
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
struct MarketStructureConfluenceCore {
    params: ResolvedParams,
    basis_state: WmaState,
    atr_state: AtrState,
    svol_state: RollingSma,
    piv_high: PivotDetector,
    piv_low: PivotDetector,
    index: usize,
    prev_high: Option<f64>,
    prev_low: Option<f64>,
    prev_high_idx: Option<usize>,
    prev_low_idx: Option<usize>,
    high_active: bool,
    low_active: bool,
    prev_break_dir: i32,
}

impl MarketStructureConfluenceCore {
    #[inline(always)]
    fn new(params: ResolvedParams) -> Self {
        Self {
            basis_state: WmaState::new(params.basis_length),
            atr_state: AtrState::new(params.atr_length),
            svol_state: RollingSma::new(params.atr_smooth),
            piv_high: PivotDetector::new(params.swing_size, true),
            piv_low: PivotDetector::new(params.swing_size, false),
            params,
            index: 0,
            prev_high: None,
            prev_low: None,
            prev_high_idx: None,
            prev_low_idx: None,
            high_active: false,
            low_active: false,
            prev_break_dir: 0,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.basis_state.reset();
        self.atr_state.reset();
        self.svol_state.reset();
        self.piv_high.reset();
        self.piv_low.reset();
        self.index = 0;
        self.prev_high = None;
        self.prev_low = None;
        self.prev_high_idx = None;
        self.prev_low_idx = None;
        self.high_active = false;
        self.low_active = false;
        self.prev_break_dir = 0;
    }

    #[inline(always)]
    fn update(
        &mut self,
        high: f64,
        low: f64,
        close: f64,
    ) -> Option<MarketStructureConfluencePoint> {
        let basis = self.basis_state.update(close);
        let svol = self
            .atr_state
            .update(high, low, close)
            .and_then(|atr| self.svol_state.update(atr));

        let mut hh = 0.0;
        let mut lh = 0.0;
        let mut hl = 0.0;
        let mut ll = 0.0;

        if let Some((pivot_high, pivot_idx)) = self.piv_high.update(high, self.index) {
            let is_hh = self
                .prev_high
                .map(|value| pivot_high >= value)
                .unwrap_or(true);
            if is_hh {
                hh = 1.0;
            } else {
                lh = 1.0;
            }
            self.prev_high = Some(pivot_high);
            self.prev_high_idx = Some(pivot_idx);
            self.high_active = true;
        }

        if let Some((pivot_low, pivot_idx)) = self.piv_low.update(low, self.index) {
            let is_hl = self
                .prev_low
                .map(|value| pivot_low >= value)
                .unwrap_or(true);
            if is_hl {
                hl = 1.0;
            } else {
                ll = 1.0;
            }
            self.prev_low = Some(pivot_low);
            self.prev_low_idx = Some(pivot_idx);
            self.low_active = true;
        }

        let high_src = match self.params.bos_confirmation {
            MarketStructureConfluenceBosConfirmation::CandleClose => close,
            MarketStructureConfluenceBosConfirmation::Wicks => high,
        };
        let low_src = match self.params.bos_confirmation {
            MarketStructureConfluenceBosConfirmation::CandleClose => close,
            MarketStructureConfluenceBosConfirmation::Wicks => low,
        };

        let mut high_broken = false;
        let mut low_broken = false;
        if self.high_active {
            if let Some(prev_high) = self.prev_high {
                if high_src > prev_high {
                    high_broken = true;
                    self.high_active = false;
                }
            }
        }
        if self.low_active {
            if let Some(prev_low) = self.prev_low {
                if low_src < prev_low {
                    low_broken = true;
                    self.low_active = false;
                }
            }
        }

        let mut bullish_change = 0.0;
        let mut bearish_change = 0.0;
        let mut bullish_bos = 0.0;
        let mut bullish_choch = 0.0;
        let mut bearish_bos = 0.0;
        let mut bearish_choch = 0.0;

        if high_broken {
            let last_break_dir = self.prev_break_dir;
            if last_break_dir == -1 {
                bullish_choch = 1.0;
            } else {
                bullish_bos = 1.0;
            }
            if last_break_dir == -1 || last_break_dir == 0 {
                bullish_change = 1.0;
            }
            self.prev_break_dir = 1;
        }

        if low_broken {
            let last_break_dir = self.prev_break_dir;
            if last_break_dir == 1 {
                bearish_choch = 1.0;
            } else {
                bearish_bos = 1.0;
            }
            if last_break_dir == 1 || last_break_dir == 0 {
                bearish_change = 1.0;
            }
            self.prev_break_dir = -1;
        }

        self.index += 1;

        let (basis, svol) = match (basis, svol) {
            (Some(basis), Some(svol)) => (basis, svol),
            _ => return None,
        };

        let upper_band = basis + self.params.vol_mult * svol;
        let lower_band = basis - self.params.vol_mult * svol;
        let structure_direction = self.prev_break_dir as f64;
        let bullish_arrow = if self.prev_break_dir == 1 && low < lower_band && high > lower_band {
            1.0
        } else {
            0.0
        };
        let bearish_arrow = if self.prev_break_dir == -1 && low < upper_band && high > upper_band {
            1.0
        } else {
            0.0
        };

        Some(MarketStructureConfluencePoint {
            basis,
            upper_band,
            lower_band,
            structure_direction,
            bullish_arrow,
            bearish_arrow,
            bullish_change,
            bearish_change,
            hh,
            lh,
            hl,
            ll,
            bullish_bos,
            bullish_choch,
            bearish_bos,
            bearish_choch,
        })
    }
}

#[derive(Clone, Debug)]
pub struct MarketStructureConfluenceStream {
    core: MarketStructureConfluenceCore,
}

impl MarketStructureConfluenceStream {
    #[inline]
    pub fn try_new(
        params: MarketStructureConfluenceParams,
    ) -> Result<Self, MarketStructureConfluenceError> {
        let resolved = resolve_params(params, usize::MAX)?;
        Ok(Self {
            core: MarketStructureConfluenceCore::new(resolved),
        })
    }

    #[inline(always)]
    pub fn update(
        &mut self,
        high: f64,
        low: f64,
        close: f64,
    ) -> Option<MarketStructureConfluenceStreamOutput> {
        self.core
            .update(high, low, close)
            .map(|point| MarketStructureConfluenceStreamOutput {
                basis: point.basis,
                upper_band: point.upper_band,
                lower_band: point.lower_band,
                structure_direction: point.structure_direction,
                bullish_arrow: point.bullish_arrow,
                bearish_arrow: point.bearish_arrow,
                bullish_change: point.bullish_change,
                bearish_change: point.bearish_change,
                hh: point.hh,
                lh: point.lh,
                hl: point.hl,
                ll: point.ll,
                bullish_bos: point.bullish_bos,
                bullish_choch: point.bullish_choch,
                bearish_bos: point.bearish_bos,
                bearish_choch: point.bearish_choch,
            })
    }
}

#[inline]
pub fn market_structure_confluence(
    input: &MarketStructureConfluenceInput<'_>,
) -> Result<MarketStructureConfluenceOutput, MarketStructureConfluenceError> {
    market_structure_confluence_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn market_structure_confluence_with_kernel(
    input: &MarketStructureConfluenceInput<'_>,
    kernel: Kernel,
) -> Result<MarketStructureConfluenceOutput, MarketStructureConfluenceError> {
    let prepared = prepare_input(input, kernel)?;
    let mut basis = alloc_with_nan_prefix(prepared.len, prepared.warmup);
    let mut upper_band = alloc_with_nan_prefix(prepared.len, prepared.warmup);
    let mut lower_band = alloc_with_nan_prefix(prepared.len, prepared.warmup);
    let mut structure_direction = alloc_with_nan_prefix(prepared.len, prepared.warmup);
    let mut bullish_arrow = alloc_with_nan_prefix(prepared.len, prepared.warmup);
    let mut bearish_arrow = alloc_with_nan_prefix(prepared.len, prepared.warmup);
    let mut bullish_change = alloc_with_nan_prefix(prepared.len, prepared.warmup);
    let mut bearish_change = alloc_with_nan_prefix(prepared.len, prepared.warmup);
    let mut hh = alloc_with_nan_prefix(prepared.len, prepared.warmup);
    let mut lh = alloc_with_nan_prefix(prepared.len, prepared.warmup);
    let mut hl = alloc_with_nan_prefix(prepared.len, prepared.warmup);
    let mut ll = alloc_with_nan_prefix(prepared.len, prepared.warmup);
    let mut bullish_bos = alloc_with_nan_prefix(prepared.len, prepared.warmup);
    let mut bullish_choch = alloc_with_nan_prefix(prepared.len, prepared.warmup);
    let mut bearish_bos = alloc_with_nan_prefix(prepared.len, prepared.warmup);
    let mut bearish_choch = alloc_with_nan_prefix(prepared.len, prepared.warmup);

    compute_into_slices(
        &prepared,
        &mut basis,
        &mut upper_band,
        &mut lower_band,
        &mut structure_direction,
        &mut bullish_arrow,
        &mut bearish_arrow,
        &mut bullish_change,
        &mut bearish_change,
        &mut hh,
        &mut lh,
        &mut hl,
        &mut ll,
        &mut bullish_bos,
        &mut bullish_choch,
        &mut bearish_bos,
        &mut bearish_choch,
    )?;

    Ok(MarketStructureConfluenceOutput {
        basis,
        upper_band,
        lower_band,
        structure_direction,
        bullish_arrow,
        bearish_arrow,
        bullish_change,
        bearish_change,
        hh,
        lh,
        hl,
        ll,
        bullish_bos,
        bullish_choch,
        bearish_bos,
        bearish_choch,
    })
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub fn market_structure_confluence_into(
    input: &MarketStructureConfluenceInput<'_>,
    basis: &mut [f64],
    upper_band: &mut [f64],
    lower_band: &mut [f64],
    structure_direction: &mut [f64],
    bullish_arrow: &mut [f64],
    bearish_arrow: &mut [f64],
    bullish_change: &mut [f64],
    bearish_change: &mut [f64],
    hh: &mut [f64],
    lh: &mut [f64],
    hl: &mut [f64],
    ll: &mut [f64],
    bullish_bos: &mut [f64],
    bullish_choch: &mut [f64],
    bearish_bos: &mut [f64],
    bearish_choch: &mut [f64],
) -> Result<(), MarketStructureConfluenceError> {
    market_structure_confluence_into_slices(
        input,
        Kernel::Auto,
        basis,
        upper_band,
        lower_band,
        structure_direction,
        bullish_arrow,
        bearish_arrow,
        bullish_change,
        bearish_change,
        hh,
        lh,
        hl,
        ll,
        bullish_bos,
        bullish_choch,
        bearish_bos,
        bearish_choch,
    )
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub fn market_structure_confluence_into_slices(
    input: &MarketStructureConfluenceInput<'_>,
    kernel: Kernel,
    basis: &mut [f64],
    upper_band: &mut [f64],
    lower_band: &mut [f64],
    structure_direction: &mut [f64],
    bullish_arrow: &mut [f64],
    bearish_arrow: &mut [f64],
    bullish_change: &mut [f64],
    bearish_change: &mut [f64],
    hh: &mut [f64],
    lh: &mut [f64],
    hl: &mut [f64],
    ll: &mut [f64],
    bullish_bos: &mut [f64],
    bullish_choch: &mut [f64],
    bearish_bos: &mut [f64],
    bearish_choch: &mut [f64],
) -> Result<(), MarketStructureConfluenceError> {
    let prepared = prepare_input(input, kernel)?;
    let got = *[
        basis.len(),
        upper_band.len(),
        lower_band.len(),
        structure_direction.len(),
        bullish_arrow.len(),
        bearish_arrow.len(),
        bullish_change.len(),
        bearish_change.len(),
        hh.len(),
        lh.len(),
        hl.len(),
        ll.len(),
        bullish_bos.len(),
        bullish_choch.len(),
        bearish_bos.len(),
        bearish_choch.len(),
    ]
    .iter()
    .min()
    .unwrap_or(&0);
    if basis.len() != prepared.len
        || upper_band.len() != prepared.len
        || lower_band.len() != prepared.len
        || structure_direction.len() != prepared.len
        || bullish_arrow.len() != prepared.len
        || bearish_arrow.len() != prepared.len
        || bullish_change.len() != prepared.len
        || bearish_change.len() != prepared.len
        || hh.len() != prepared.len
        || lh.len() != prepared.len
        || hl.len() != prepared.len
        || ll.len() != prepared.len
        || bullish_bos.len() != prepared.len
        || bullish_choch.len() != prepared.len
        || bearish_bos.len() != prepared.len
        || bearish_choch.len() != prepared.len
    {
        return Err(MarketStructureConfluenceError::OutputLengthMismatch {
            expected: prepared.len,
            got,
        });
    }

    compute_into_slices(
        &prepared,
        basis,
        upper_band,
        lower_band,
        structure_direction,
        bullish_arrow,
        bearish_arrow,
        bullish_change,
        bearish_change,
        hh,
        lh,
        hl,
        ll,
        bullish_bos,
        bullish_choch,
        bearish_bos,
        bearish_choch,
    )
}

#[inline]
fn resolve_data<'a>(
    input: &'a MarketStructureConfluenceInput<'a>,
) -> Result<(&'a [f64], &'a [f64], &'a [f64]), MarketStructureConfluenceError> {
    match &input.data {
        MarketStructureConfluenceData::Candles { candles } => Ok((
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
        )),
        MarketStructureConfluenceData::Slices { high, low, close } => {
            if high.len() != low.len() || high.len() != close.len() {
                return Err(MarketStructureConfluenceError::DataLengthMismatch {
                    high: high.len(),
                    low: low.len(),
                    close: close.len(),
                });
            }
            Ok((high, low, close))
        }
    }
}

#[inline]
fn resolve_params(
    params: MarketStructureConfluenceParams,
    data_len: usize,
) -> Result<ResolvedParams, MarketStructureConfluenceError> {
    let swing_size = params.swing_size.unwrap_or(DEFAULT_SWING_SIZE);
    let bos_confirmation_raw = params
        .bos_confirmation
        .unwrap_or_else(|| DEFAULT_BOS_CONFIRMATION.to_string());
    let bos_confirmation = MarketStructureConfluenceBosConfirmation::parse(&bos_confirmation_raw)
        .ok_or(MarketStructureConfluenceError::InvalidBosConfirmation {
        bos_confirmation: bos_confirmation_raw.clone(),
    })?;
    let basis_length = params.basis_length.unwrap_or(DEFAULT_BASIS_LENGTH);
    let atr_length = params.atr_length.unwrap_or(DEFAULT_ATR_LENGTH);
    let atr_smooth = params.atr_smooth.unwrap_or(DEFAULT_ATR_SMOOTH);
    let vol_mult = params.vol_mult.unwrap_or(DEFAULT_VOL_MULT);

    if swing_size < 2 || (data_len != usize::MAX && swing_size * 2 + 1 > data_len) {
        return Err(MarketStructureConfluenceError::InvalidSwingSize {
            swing_size,
            data_len,
        });
    }
    if basis_length == 0 || (data_len != usize::MAX && basis_length > data_len) {
        return Err(MarketStructureConfluenceError::InvalidBasisLength {
            basis_length,
            data_len,
        });
    }
    if atr_length == 0 || (data_len != usize::MAX && atr_length > data_len) {
        return Err(MarketStructureConfluenceError::InvalidAtrLength {
            atr_length,
            data_len,
        });
    }
    if atr_smooth == 0 || (data_len != usize::MAX && atr_smooth > data_len) {
        return Err(MarketStructureConfluenceError::InvalidAtrSmooth {
            atr_smooth,
            data_len,
        });
    }
    if !vol_mult.is_finite() || vol_mult < 0.0 {
        return Err(MarketStructureConfluenceError::InvalidVolMult { vol_mult });
    }

    Ok(ResolvedParams {
        swing_size,
        bos_confirmation,
        basis_length,
        atr_length,
        atr_smooth,
        vol_mult,
    })
}

#[inline]
fn prepare_input<'a>(
    input: &'a MarketStructureConfluenceInput<'a>,
    kernel: Kernel,
) -> Result<PreparedInput<'a>, MarketStructureConfluenceError> {
    let (high, low, close) = resolve_data(input)?;
    let len = close.len();
    if len == 0 {
        return Err(MarketStructureConfluenceError::EmptyInputData);
    }
    let mut first = len;
    let mut valid = 0usize;
    for i in 0..len {
        if high[i].is_finite() && low[i].is_finite() && close[i].is_finite() {
            if first == len {
                first = i;
            }
            valid += 1;
        }
    }
    if first == len {
        return Err(MarketStructureConfluenceError::AllValuesNaN);
    }
    let params = resolve_params(input.params.clone(), len)?;
    let needed = (params.swing_size * 2 + 1)
        .max(params.basis_length)
        .max(params.atr_length + params.atr_smooth - 1);
    if valid < needed {
        return Err(MarketStructureConfluenceError::NotEnoughValidData { needed, valid });
    }
    let _chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        value => value,
    };
    Ok(PreparedInput {
        high,
        low,
        close,
        len,
        params,
        first,
        valid,
        warmup: first
            + (params.swing_size * 2)
                .max(params.basis_length.saturating_sub(1))
                .max(params.atr_length + params.atr_smooth - 2),
    })
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn compute_into_slices(
    prepared: &PreparedInput<'_>,
    dst_basis: &mut [f64],
    dst_upper_band: &mut [f64],
    dst_lower_band: &mut [f64],
    dst_structure_direction: &mut [f64],
    dst_bullish_arrow: &mut [f64],
    dst_bearish_arrow: &mut [f64],
    dst_bullish_change: &mut [f64],
    dst_bearish_change: &mut [f64],
    dst_hh: &mut [f64],
    dst_lh: &mut [f64],
    dst_hl: &mut [f64],
    dst_ll: &mut [f64],
    dst_bullish_bos: &mut [f64],
    dst_bullish_choch: &mut [f64],
    dst_bearish_bos: &mut [f64],
    dst_bearish_choch: &mut [f64],
) -> Result<(), MarketStructureConfluenceError> {
    let clean = prepared.first == 0 && prepared.valid == prepared.len;
    if clean {
        let warmup = prepared.warmup.min(prepared.len);
        for dst in [
            &mut *dst_basis,
            &mut *dst_upper_band,
            &mut *dst_lower_band,
            &mut *dst_structure_direction,
            &mut *dst_bullish_arrow,
            &mut *dst_bearish_arrow,
            &mut *dst_bullish_change,
            &mut *dst_bearish_change,
            &mut *dst_hh,
            &mut *dst_lh,
            &mut *dst_hl,
            &mut *dst_ll,
            &mut *dst_bullish_bos,
            &mut *dst_bullish_choch,
            &mut *dst_bearish_bos,
            &mut *dst_bearish_choch,
        ] {
            for value in &mut dst[..warmup] {
                *value = f64::NAN;
            }
        }
    } else {
        dst_basis.fill(f64::NAN);
        dst_upper_band.fill(f64::NAN);
        dst_lower_band.fill(f64::NAN);
        dst_structure_direction.fill(f64::NAN);
        dst_bullish_arrow.fill(f64::NAN);
        dst_bearish_arrow.fill(f64::NAN);
        dst_bullish_change.fill(f64::NAN);
        dst_bearish_change.fill(f64::NAN);
        dst_hh.fill(f64::NAN);
        dst_lh.fill(f64::NAN);
        dst_hl.fill(f64::NAN);
        dst_ll.fill(f64::NAN);
        dst_bullish_bos.fill(f64::NAN);
        dst_bullish_choch.fill(f64::NAN);
        dst_bearish_bos.fill(f64::NAN);
        dst_bearish_choch.fill(f64::NAN);
    }

    let mut core = MarketStructureConfluenceCore::new(prepared.params);
    for i in 0..prepared.len {
        let Some(point) = core.update(prepared.high[i], prepared.low[i], prepared.close[i]) else {
            continue;
        };
        dst_basis[i] = point.basis;
        dst_upper_band[i] = point.upper_band;
        dst_lower_band[i] = point.lower_band;
        dst_structure_direction[i] = point.structure_direction;
        dst_bullish_arrow[i] = point.bullish_arrow;
        dst_bearish_arrow[i] = point.bearish_arrow;
        dst_bullish_change[i] = point.bullish_change;
        dst_bearish_change[i] = point.bearish_change;
        dst_hh[i] = point.hh;
        dst_lh[i] = point.lh;
        dst_hl[i] = point.hl;
        dst_ll[i] = point.ll;
        dst_bullish_bos[i] = point.bullish_bos;
        dst_bullish_choch[i] = point.bullish_choch;
        dst_bearish_bos[i] = point.bearish_bos;
        dst_bearish_choch[i] = point.bearish_choch;
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct MarketStructureConfluenceBatchRange {
    pub swing_size: (usize, usize, usize),
    pub bos_confirmation: Vec<String>,
    pub basis_length: (usize, usize, usize),
    pub atr_length: (usize, usize, usize),
    pub atr_smooth: (usize, usize, usize),
    pub vol_mult: (f64, f64, f64),
}

impl Default for MarketStructureConfluenceBatchRange {
    fn default() -> Self {
        Self {
            swing_size: (DEFAULT_SWING_SIZE, DEFAULT_SWING_SIZE, 0),
            bos_confirmation: vec![DEFAULT_BOS_CONFIRMATION.to_string()],
            basis_length: (DEFAULT_BASIS_LENGTH, DEFAULT_BASIS_LENGTH, 0),
            atr_length: (DEFAULT_ATR_LENGTH, DEFAULT_ATR_LENGTH, 0),
            atr_smooth: (DEFAULT_ATR_SMOOTH, DEFAULT_ATR_SMOOTH, 0),
            vol_mult: (DEFAULT_VOL_MULT, DEFAULT_VOL_MULT, 0.0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct MarketStructureConfluenceBatchOutput {
    pub basis: Vec<f64>,
    pub upper_band: Vec<f64>,
    pub lower_band: Vec<f64>,
    pub structure_direction: Vec<f64>,
    pub bullish_arrow: Vec<f64>,
    pub bearish_arrow: Vec<f64>,
    pub bullish_change: Vec<f64>,
    pub bearish_change: Vec<f64>,
    pub hh: Vec<f64>,
    pub lh: Vec<f64>,
    pub hl: Vec<f64>,
    pub ll: Vec<f64>,
    pub bullish_bos: Vec<f64>,
    pub bullish_choch: Vec<f64>,
    pub bearish_bos: Vec<f64>,
    pub bearish_choch: Vec<f64>,
    pub combos: Vec<MarketStructureConfluenceParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Debug)]
pub struct MarketStructureConfluenceBatchBuilder {
    range: MarketStructureConfluenceBatchRange,
    kernel: Kernel,
}

impl Default for MarketStructureConfluenceBatchBuilder {
    fn default() -> Self {
        Self {
            range: MarketStructureConfluenceBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl MarketStructureConfluenceBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn range(mut self, value: MarketStructureConfluenceBatchRange) -> Self {
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
    ) -> Result<MarketStructureConfluenceBatchOutput, MarketStructureConfluenceError> {
        self.apply_slices(
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
        )
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<MarketStructureConfluenceBatchOutput, MarketStructureConfluenceError> {
        market_structure_confluence_batch_with_kernel(high, low, close, &self.range, self.kernel)
    }
}

fn axis_usize(
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, MarketStructureConfluenceError> {
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
        return Err(MarketStructureConfluenceError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

fn axis_f64(
    (start, end, step): (f64, f64, f64),
) -> Result<Vec<f64>, MarketStructureConfluenceError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(MarketStructureConfluenceError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if step.abs() < EPS || (start - end).abs() < EPS {
        return Ok(vec![start]);
    }
    let dir = if end >= start { 1.0 } else { -1.0 };
    let step_eff = dir * step.abs();
    let mut current = start;
    let mut out = Vec::new();
    if dir > 0.0 {
        while current <= end + EPS {
            out.push(current);
            current += step_eff;
        }
    } else {
        while current >= end - EPS {
            out.push(current);
            current += step_eff;
        }
    }
    if out.is_empty() {
        return Err(MarketStructureConfluenceError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

fn expand_grid(
    range: &MarketStructureConfluenceBatchRange,
) -> Result<Vec<MarketStructureConfluenceParams>, MarketStructureConfluenceError> {
    let swing_sizes = axis_usize(range.swing_size)?;
    let bos_confirmations = if range.bos_confirmation.is_empty() {
        vec![DEFAULT_BOS_CONFIRMATION.to_string()]
    } else {
        range.bos_confirmation.clone()
    };
    let basis_lengths = axis_usize(range.basis_length)?;
    let atr_lengths = axis_usize(range.atr_length)?;
    let atr_smooths = axis_usize(range.atr_smooth)?;
    let vol_mults = axis_f64(range.vol_mult)?;

    let total = swing_sizes
        .len()
        .checked_mul(bos_confirmations.len())
        .and_then(|n| n.checked_mul(basis_lengths.len()))
        .and_then(|n| n.checked_mul(atr_lengths.len()))
        .and_then(|n| n.checked_mul(atr_smooths.len()))
        .and_then(|n| n.checked_mul(vol_mults.len()))
        .ok_or_else(|| MarketStructureConfluenceError::InvalidRange {
            start: range.swing_size.0.to_string(),
            end: range.swing_size.1.to_string(),
            step: range.swing_size.2.to_string(),
        })?;

    let mut out = Vec::with_capacity(total);
    for &swing_size in &swing_sizes {
        for bos_confirmation in &bos_confirmations {
            for &basis_length in &basis_lengths {
                for &atr_length in &atr_lengths {
                    for &atr_smooth in &atr_smooths {
                        for &vol_mult in &vol_mults {
                            out.push(MarketStructureConfluenceParams {
                                swing_size: Some(swing_size),
                                bos_confirmation: Some(bos_confirmation.clone()),
                                basis_length: Some(basis_length),
                                atr_length: Some(atr_length),
                                atr_smooth: Some(atr_smooth),
                                vol_mult: Some(vol_mult),
                            });
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub fn market_structure_confluence_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    range: &MarketStructureConfluenceBatchRange,
    kernel: Kernel,
) -> Result<MarketStructureConfluenceBatchOutput, MarketStructureConfluenceError> {
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(MarketStructureConfluenceError::EmptyInputData);
    }
    if high.len() != low.len() || high.len() != close.len() {
        return Err(MarketStructureConfluenceError::DataLengthMismatch {
            high: high.len(),
            low: low.len(),
            close: close.len(),
        });
    }

    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        value if value.is_batch() => value,
        _ => {
            return Err(MarketStructureConfluenceError::InvalidKernelForBatch(
                kernel,
            ))
        }
    };
    let single_kernel = batch_kernel.to_non_batch();
    let combos = expand_grid(range)?;
    let rows = combos.len();
    let cols = close.len();
    let first = (0..cols)
        .find(|&i| high[i].is_finite() && low[i].is_finite() && close[i].is_finite())
        .ok_or(MarketStructureConfluenceError::AllValuesNaN)?;
    let warmups: Vec<usize> = combos
        .iter()
        .map(|combo| {
            let swing_size = combo.swing_size.unwrap_or(DEFAULT_SWING_SIZE);
            let basis_length = combo.basis_length.unwrap_or(DEFAULT_BASIS_LENGTH);
            let atr_length = combo.atr_length.unwrap_or(DEFAULT_ATR_LENGTH);
            let atr_smooth = combo.atr_smooth.unwrap_or(DEFAULT_ATR_SMOOTH);
            first
                + (swing_size * 2)
                    .max(basis_length.saturating_sub(1))
                    .max(atr_length + atr_smooth - 2)
        })
        .collect();

    let mut basis_mu = make_uninit_matrix(rows, cols);
    let mut upper_band_mu = make_uninit_matrix(rows, cols);
    let mut lower_band_mu = make_uninit_matrix(rows, cols);
    let mut structure_direction_mu = make_uninit_matrix(rows, cols);
    let mut bullish_arrow_mu = make_uninit_matrix(rows, cols);
    let mut bearish_arrow_mu = make_uninit_matrix(rows, cols);
    let mut bullish_change_mu = make_uninit_matrix(rows, cols);
    let mut bearish_change_mu = make_uninit_matrix(rows, cols);
    let mut hh_mu = make_uninit_matrix(rows, cols);
    let mut lh_mu = make_uninit_matrix(rows, cols);
    let mut hl_mu = make_uninit_matrix(rows, cols);
    let mut ll_mu = make_uninit_matrix(rows, cols);
    let mut bullish_bos_mu = make_uninit_matrix(rows, cols);
    let mut bullish_choch_mu = make_uninit_matrix(rows, cols);
    let mut bearish_bos_mu = make_uninit_matrix(rows, cols);
    let mut bearish_choch_mu = make_uninit_matrix(rows, cols);

    init_matrix_prefixes(&mut basis_mu, cols, &warmups);
    init_matrix_prefixes(&mut upper_band_mu, cols, &warmups);
    init_matrix_prefixes(&mut lower_band_mu, cols, &warmups);
    init_matrix_prefixes(&mut structure_direction_mu, cols, &warmups);
    init_matrix_prefixes(&mut bullish_arrow_mu, cols, &warmups);
    init_matrix_prefixes(&mut bearish_arrow_mu, cols, &warmups);
    init_matrix_prefixes(&mut bullish_change_mu, cols, &warmups);
    init_matrix_prefixes(&mut bearish_change_mu, cols, &warmups);
    init_matrix_prefixes(&mut hh_mu, cols, &warmups);
    init_matrix_prefixes(&mut lh_mu, cols, &warmups);
    init_matrix_prefixes(&mut hl_mu, cols, &warmups);
    init_matrix_prefixes(&mut ll_mu, cols, &warmups);
    init_matrix_prefixes(&mut bullish_bos_mu, cols, &warmups);
    init_matrix_prefixes(&mut bullish_choch_mu, cols, &warmups);
    init_matrix_prefixes(&mut bearish_bos_mu, cols, &warmups);
    init_matrix_prefixes(&mut bearish_choch_mu, cols, &warmups);

    let mut basis_guard = ManuallyDrop::new(basis_mu);
    let mut upper_band_guard = ManuallyDrop::new(upper_band_mu);
    let mut lower_band_guard = ManuallyDrop::new(lower_band_mu);
    let mut structure_direction_guard = ManuallyDrop::new(structure_direction_mu);
    let mut bullish_arrow_guard = ManuallyDrop::new(bullish_arrow_mu);
    let mut bearish_arrow_guard = ManuallyDrop::new(bearish_arrow_mu);
    let mut bullish_change_guard = ManuallyDrop::new(bullish_change_mu);
    let mut bearish_change_guard = ManuallyDrop::new(bearish_change_mu);
    let mut hh_guard = ManuallyDrop::new(hh_mu);
    let mut lh_guard = ManuallyDrop::new(lh_mu);
    let mut hl_guard = ManuallyDrop::new(hl_mu);
    let mut ll_guard = ManuallyDrop::new(ll_mu);
    let mut bullish_bos_guard = ManuallyDrop::new(bullish_bos_mu);
    let mut bullish_choch_guard = ManuallyDrop::new(bullish_choch_mu);
    let mut bearish_bos_guard = ManuallyDrop::new(bearish_bos_mu);
    let mut bearish_choch_guard = ManuallyDrop::new(bearish_choch_mu);

    let basis_all = unsafe { mu_slice_as_f64_slice_mut(&mut basis_guard) };
    let upper_band_all = unsafe { mu_slice_as_f64_slice_mut(&mut upper_band_guard) };
    let lower_band_all = unsafe { mu_slice_as_f64_slice_mut(&mut lower_band_guard) };
    let structure_direction_all =
        unsafe { mu_slice_as_f64_slice_mut(&mut structure_direction_guard) };
    let bullish_arrow_all = unsafe { mu_slice_as_f64_slice_mut(&mut bullish_arrow_guard) };
    let bearish_arrow_all = unsafe { mu_slice_as_f64_slice_mut(&mut bearish_arrow_guard) };
    let bullish_change_all = unsafe { mu_slice_as_f64_slice_mut(&mut bullish_change_guard) };
    let bearish_change_all = unsafe { mu_slice_as_f64_slice_mut(&mut bearish_change_guard) };
    let hh_all = unsafe { mu_slice_as_f64_slice_mut(&mut hh_guard) };
    let lh_all = unsafe { mu_slice_as_f64_slice_mut(&mut lh_guard) };
    let hl_all = unsafe { mu_slice_as_f64_slice_mut(&mut hl_guard) };
    let ll_all = unsafe { mu_slice_as_f64_slice_mut(&mut ll_guard) };
    let bullish_bos_all = unsafe { mu_slice_as_f64_slice_mut(&mut bullish_bos_guard) };
    let bullish_choch_all = unsafe { mu_slice_as_f64_slice_mut(&mut bullish_choch_guard) };
    let bearish_bos_all = unsafe { mu_slice_as_f64_slice_mut(&mut bearish_bos_guard) };
    let bearish_choch_all = unsafe { mu_slice_as_f64_slice_mut(&mut bearish_choch_guard) };

    let run_row = |row: usize,
                   basis_row: &mut [f64],
                   upper_band_row: &mut [f64],
                   lower_band_row: &mut [f64],
                   structure_direction_row: &mut [f64],
                   bullish_arrow_row: &mut [f64],
                   bearish_arrow_row: &mut [f64],
                   bullish_change_row: &mut [f64],
                   bearish_change_row: &mut [f64],
                   hh_row: &mut [f64],
                   lh_row: &mut [f64],
                   hl_row: &mut [f64],
                   ll_row: &mut [f64],
                   bullish_bos_row: &mut [f64],
                   bullish_choch_row: &mut [f64],
                   bearish_bos_row: &mut [f64],
                   bearish_choch_row: &mut [f64]|
     -> Result<(), MarketStructureConfluenceError> {
        let input =
            MarketStructureConfluenceInput::from_slices(high, low, close, combos[row].clone());
        market_structure_confluence_into_slices(
            &input,
            single_kernel,
            basis_row,
            upper_band_row,
            lower_band_row,
            structure_direction_row,
            bullish_arrow_row,
            bearish_arrow_row,
            bullish_change_row,
            bearish_change_row,
            hh_row,
            lh_row,
            hl_row,
            ll_row,
            bullish_bos_row,
            bullish_choch_row,
            bearish_bos_row,
            bearish_choch_row,
        )
    };

    #[cfg(not(target_arch = "wasm32"))]
    {
        basis_all
            .par_chunks_mut(cols)
            .zip(upper_band_all.par_chunks_mut(cols))
            .zip(lower_band_all.par_chunks_mut(cols))
            .zip(structure_direction_all.par_chunks_mut(cols))
            .zip(bullish_arrow_all.par_chunks_mut(cols))
            .zip(bearish_arrow_all.par_chunks_mut(cols))
            .zip(bullish_change_all.par_chunks_mut(cols))
            .zip(bearish_change_all.par_chunks_mut(cols))
            .zip(hh_all.par_chunks_mut(cols))
            .zip(lh_all.par_chunks_mut(cols))
            .zip(hl_all.par_chunks_mut(cols))
            .zip(ll_all.par_chunks_mut(cols))
            .zip(bullish_bos_all.par_chunks_mut(cols))
            .zip(bullish_choch_all.par_chunks_mut(cols))
            .zip(bearish_bos_all.par_chunks_mut(cols))
            .zip(bearish_choch_all.par_chunks_mut(cols))
            .enumerate()
            .try_for_each(
                |(
                    row,
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        (
                                                            (
                                                                (
                                                                    (
                                                                        (
                                                                            (
                                                                                basis_row,
                                                                                upper_band_row,
                                                                            ),
                                                                            lower_band_row,
                                                                        ),
                                                                        structure_direction_row,
                                                                    ),
                                                                    bullish_arrow_row,
                                                                ),
                                                                bearish_arrow_row,
                                                            ),
                                                            bullish_change_row,
                                                        ),
                                                        bearish_change_row,
                                                    ),
                                                    hh_row,
                                                ),
                                                lh_row,
                                            ),
                                            hl_row,
                                        ),
                                        ll_row,
                                    ),
                                    bullish_bos_row,
                                ),
                                bullish_choch_row,
                            ),
                            bearish_bos_row,
                        ),
                        bearish_choch_row,
                    ),
                )| {
                    run_row(
                        row,
                        basis_row,
                        upper_band_row,
                        lower_band_row,
                        structure_direction_row,
                        bullish_arrow_row,
                        bearish_arrow_row,
                        bullish_change_row,
                        bearish_change_row,
                        hh_row,
                        lh_row,
                        hl_row,
                        ll_row,
                        bullish_bos_row,
                        bullish_choch_row,
                        bearish_bos_row,
                        bearish_choch_row,
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
                &mut basis_all[start..end],
                &mut upper_band_all[start..end],
                &mut lower_band_all[start..end],
                &mut structure_direction_all[start..end],
                &mut bullish_arrow_all[start..end],
                &mut bearish_arrow_all[start..end],
                &mut bullish_change_all[start..end],
                &mut bearish_change_all[start..end],
                &mut hh_all[start..end],
                &mut lh_all[start..end],
                &mut hl_all[start..end],
                &mut ll_all[start..end],
                &mut bullish_bos_all[start..end],
                &mut bullish_choch_all[start..end],
                &mut bearish_bos_all[start..end],
                &mut bearish_choch_all[start..end],
            )?;
        }
    }

    let basis = unsafe { assume_init_vec(basis_guard) };
    let upper_band = unsafe { assume_init_vec(upper_band_guard) };
    let lower_band = unsafe { assume_init_vec(lower_band_guard) };
    let structure_direction = unsafe { assume_init_vec(structure_direction_guard) };
    let bullish_arrow = unsafe { assume_init_vec(bullish_arrow_guard) };
    let bearish_arrow = unsafe { assume_init_vec(bearish_arrow_guard) };
    let bullish_change = unsafe { assume_init_vec(bullish_change_guard) };
    let bearish_change = unsafe { assume_init_vec(bearish_change_guard) };
    let hh = unsafe { assume_init_vec(hh_guard) };
    let lh = unsafe { assume_init_vec(lh_guard) };
    let hl = unsafe { assume_init_vec(hl_guard) };
    let ll = unsafe { assume_init_vec(ll_guard) };
    let bullish_bos = unsafe { assume_init_vec(bullish_bos_guard) };
    let bullish_choch = unsafe { assume_init_vec(bullish_choch_guard) };
    let bearish_bos = unsafe { assume_init_vec(bearish_bos_guard) };
    let bearish_choch = unsafe { assume_init_vec(bearish_choch_guard) };

    Ok(MarketStructureConfluenceBatchOutput {
        basis,
        upper_band,
        lower_band,
        structure_direction,
        bullish_arrow,
        bearish_arrow,
        bullish_change,
        bearish_change,
        hh,
        lh,
        hl,
        ll,
        bullish_bos,
        bullish_choch,
        bearish_bos,
        bearish_choch,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn mu_slice_as_f64_slice_mut(buf: &mut ManuallyDrop<Vec<MaybeUninit<f64>>>) -> &mut [f64] {
    std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut f64, buf.len())
}

#[inline(always)]
unsafe fn assume_init_vec(buf: ManuallyDrop<Vec<MaybeUninit<f64>>>) -> Vec<f64> {
    let mut buf = buf;
    Vec::from_raw_parts(buf.as_mut_ptr() as *mut f64, buf.len(), buf.capacity())
}

#[cfg(feature = "python")]
#[pyfunction(name = "market_structure_confluence")]
#[pyo3(signature = (high, low, close, swing_size=DEFAULT_SWING_SIZE, bos_confirmation=DEFAULT_BOS_CONFIRMATION, basis_length=DEFAULT_BASIS_LENGTH, atr_length=DEFAULT_ATR_LENGTH, atr_smooth=DEFAULT_ATR_SMOOTH, vol_mult=DEFAULT_VOL_MULT, kernel=None))]
pub fn market_structure_confluence_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    swing_size: usize,
    bos_confirmation: &str,
    basis_length: usize,
    atr_length: usize,
    atr_smooth: usize,
    vol_mult: f64,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = MarketStructureConfluenceInput::from_slices(
        high,
        low,
        close,
        MarketStructureConfluenceParams {
            swing_size: Some(swing_size),
            bos_confirmation: Some(bos_confirmation.to_string()),
            basis_length: Some(basis_length),
            atr_length: Some(atr_length),
            atr_smooth: Some(atr_smooth),
            vol_mult: Some(vol_mult),
        },
    );
    let output = py
        .allow_threads(|| market_structure_confluence_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dict = PyDict::new(py);
    dict.set_item("basis", output.basis.into_pyarray(py))?;
    dict.set_item("upper_band", output.upper_band.into_pyarray(py))?;
    dict.set_item("lower_band", output.lower_band.into_pyarray(py))?;
    dict.set_item(
        "structure_direction",
        output.structure_direction.into_pyarray(py),
    )?;
    dict.set_item("bullish_arrow", output.bullish_arrow.into_pyarray(py))?;
    dict.set_item("bearish_arrow", output.bearish_arrow.into_pyarray(py))?;
    dict.set_item("bullish_change", output.bullish_change.into_pyarray(py))?;
    dict.set_item("bearish_change", output.bearish_change.into_pyarray(py))?;
    dict.set_item("hh", output.hh.into_pyarray(py))?;
    dict.set_item("lh", output.lh.into_pyarray(py))?;
    dict.set_item("hl", output.hl.into_pyarray(py))?;
    dict.set_item("ll", output.ll.into_pyarray(py))?;
    dict.set_item("bullish_bos", output.bullish_bos.into_pyarray(py))?;
    dict.set_item("bullish_choch", output.bullish_choch.into_pyarray(py))?;
    dict.set_item("bearish_bos", output.bearish_bos.into_pyarray(py))?;
    dict.set_item("bearish_choch", output.bearish_choch.into_pyarray(py))?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyfunction(name = "market_structure_confluence_batch")]
#[pyo3(signature = (high, low, close, swing_size_range=(DEFAULT_SWING_SIZE, DEFAULT_SWING_SIZE, 0), bos_confirmation_options=vec![DEFAULT_BOS_CONFIRMATION.to_string()], basis_length_range=(DEFAULT_BASIS_LENGTH, DEFAULT_BASIS_LENGTH, 0), atr_length_range=(DEFAULT_ATR_LENGTH, DEFAULT_ATR_LENGTH, 0), atr_smooth_range=(DEFAULT_ATR_SMOOTH, DEFAULT_ATR_SMOOTH, 0), vol_mult_range=(DEFAULT_VOL_MULT, DEFAULT_VOL_MULT, 0.0), kernel=None))]
pub fn market_structure_confluence_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    swing_size_range: (usize, usize, usize),
    bos_confirmation_options: Vec<String>,
    basis_length_range: (usize, usize, usize),
    atr_length_range: (usize, usize, usize),
    atr_smooth_range: (usize, usize, usize),
    vol_mult_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let kernel = validate_kernel(kernel, true)?;
    let output = py
        .allow_threads(|| {
            market_structure_confluence_batch_with_kernel(
                high,
                low,
                close,
                &MarketStructureConfluenceBatchRange {
                    swing_size: swing_size_range,
                    bos_confirmation: bos_confirmation_options,
                    basis_length: basis_length_range,
                    atr_length: atr_length_range,
                    atr_smooth: atr_smooth_range,
                    vol_mult: vol_mult_range,
                },
                kernel,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let total = output.rows * output.cols;
    let arrays = [
        unsafe { PyArray1::<f64>::new(py, [total], false) },
        unsafe { PyArray1::<f64>::new(py, [total], false) },
        unsafe { PyArray1::<f64>::new(py, [total], false) },
        unsafe { PyArray1::<f64>::new(py, [total], false) },
        unsafe { PyArray1::<f64>::new(py, [total], false) },
        unsafe { PyArray1::<f64>::new(py, [total], false) },
        unsafe { PyArray1::<f64>::new(py, [total], false) },
        unsafe { PyArray1::<f64>::new(py, [total], false) },
        unsafe { PyArray1::<f64>::new(py, [total], false) },
        unsafe { PyArray1::<f64>::new(py, [total], false) },
        unsafe { PyArray1::<f64>::new(py, [total], false) },
        unsafe { PyArray1::<f64>::new(py, [total], false) },
        unsafe { PyArray1::<f64>::new(py, [total], false) },
        unsafe { PyArray1::<f64>::new(py, [total], false) },
        unsafe { PyArray1::<f64>::new(py, [total], false) },
        unsafe { PyArray1::<f64>::new(py, [total], false) },
    ];
    unsafe { arrays[0].as_slice_mut()? }.copy_from_slice(&output.basis);
    unsafe { arrays[1].as_slice_mut()? }.copy_from_slice(&output.upper_band);
    unsafe { arrays[2].as_slice_mut()? }.copy_from_slice(&output.lower_band);
    unsafe { arrays[3].as_slice_mut()? }.copy_from_slice(&output.structure_direction);
    unsafe { arrays[4].as_slice_mut()? }.copy_from_slice(&output.bullish_arrow);
    unsafe { arrays[5].as_slice_mut()? }.copy_from_slice(&output.bearish_arrow);
    unsafe { arrays[6].as_slice_mut()? }.copy_from_slice(&output.bullish_change);
    unsafe { arrays[7].as_slice_mut()? }.copy_from_slice(&output.bearish_change);
    unsafe { arrays[8].as_slice_mut()? }.copy_from_slice(&output.hh);
    unsafe { arrays[9].as_slice_mut()? }.copy_from_slice(&output.lh);
    unsafe { arrays[10].as_slice_mut()? }.copy_from_slice(&output.hl);
    unsafe { arrays[11].as_slice_mut()? }.copy_from_slice(&output.ll);
    unsafe { arrays[12].as_slice_mut()? }.copy_from_slice(&output.bullish_bos);
    unsafe { arrays[13].as_slice_mut()? }.copy_from_slice(&output.bullish_choch);
    unsafe { arrays[14].as_slice_mut()? }.copy_from_slice(&output.bearish_bos);
    unsafe { arrays[15].as_slice_mut()? }.copy_from_slice(&output.bearish_choch);

    let dict = PyDict::new(py);
    dict.set_item("basis", arrays[0].reshape((output.rows, output.cols))?)?;
    dict.set_item("upper_band", arrays[1].reshape((output.rows, output.cols))?)?;
    dict.set_item("lower_band", arrays[2].reshape((output.rows, output.cols))?)?;
    dict.set_item(
        "structure_direction",
        arrays[3].reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "bullish_arrow",
        arrays[4].reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "bearish_arrow",
        arrays[5].reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "bullish_change",
        arrays[6].reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "bearish_change",
        arrays[7].reshape((output.rows, output.cols))?,
    )?;
    dict.set_item("hh", arrays[8].reshape((output.rows, output.cols))?)?;
    dict.set_item("lh", arrays[9].reshape((output.rows, output.cols))?)?;
    dict.set_item("hl", arrays[10].reshape((output.rows, output.cols))?)?;
    dict.set_item("ll", arrays[11].reshape((output.rows, output.cols))?)?;
    dict.set_item(
        "bullish_bos",
        arrays[12].reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "bullish_choch",
        arrays[13].reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "bearish_bos",
        arrays[14].reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "bearish_choch",
        arrays[15].reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "swing_sizes",
        output
            .combos
            .iter()
            .map(|combo| combo.swing_size.unwrap_or(DEFAULT_SWING_SIZE) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "bos_confirmations",
        output
            .combos
            .iter()
            .map(|combo| {
                combo
                    .bos_confirmation
                    .clone()
                    .unwrap_or_else(|| DEFAULT_BOS_CONFIRMATION.to_string())
            })
            .collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "basis_lengths",
        output
            .combos
            .iter()
            .map(|combo| combo.basis_length.unwrap_or(DEFAULT_BASIS_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "atr_lengths",
        output
            .combos
            .iter()
            .map(|combo| combo.atr_length.unwrap_or(DEFAULT_ATR_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "atr_smooths",
        output
            .combos
            .iter()
            .map(|combo| combo.atr_smooth.unwrap_or(DEFAULT_ATR_SMOOTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "vol_mults",
        output
            .combos
            .iter()
            .map(|combo| combo.vol_mult.unwrap_or(DEFAULT_VOL_MULT))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", output.rows)?;
    dict.set_item("cols", output.cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "MarketStructureConfluenceStream")]
pub struct MarketStructureConfluenceStreamPy {
    stream: MarketStructureConfluenceStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl MarketStructureConfluenceStreamPy {
    #[new]
    #[pyo3(signature = (swing_size=DEFAULT_SWING_SIZE, bos_confirmation=DEFAULT_BOS_CONFIRMATION, basis_length=DEFAULT_BASIS_LENGTH, atr_length=DEFAULT_ATR_LENGTH, atr_smooth=DEFAULT_ATR_SMOOTH, vol_mult=DEFAULT_VOL_MULT))]
    fn new(
        swing_size: usize,
        bos_confirmation: &str,
        basis_length: usize,
        atr_length: usize,
        atr_smooth: usize,
        vol_mult: f64,
    ) -> PyResult<Self> {
        let stream = MarketStructureConfluenceStream::try_new(MarketStructureConfluenceParams {
            swing_size: Some(swing_size),
            bos_confirmation: Some(bos_confirmation.to_string()),
            basis_length: Some(basis_length),
            atr_length: Some(atr_length),
            atr_smooth: Some(atr_smooth),
            vol_mult: Some(vol_mult),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, high: f64, low: f64, close: f64) -> Option<Vec<f64>> {
        self.stream.update(high, low, close).map(|output| {
            vec![
                output.basis,
                output.upper_band,
                output.lower_band,
                output.structure_direction,
                output.bullish_arrow,
                output.bearish_arrow,
                output.bullish_change,
                output.bearish_change,
                output.hh,
                output.lh,
                output.hl,
                output.ll,
                output.bullish_bos,
                output.bullish_choch,
                output.bearish_bos,
                output.bearish_choch,
            ]
        })
    }
}

#[cfg(feature = "python")]
pub fn register_market_structure_confluence_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(market_structure_confluence_py, m)?)?;
    m.add_function(wrap_pyfunction!(market_structure_confluence_batch_py, m)?)?;
    m.add_class::<MarketStructureConfluenceStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct MarketStructureConfluenceJsOutput {
    pub basis: Vec<f64>,
    pub upper_band: Vec<f64>,
    pub lower_band: Vec<f64>,
    pub structure_direction: Vec<f64>,
    pub bullish_arrow: Vec<f64>,
    pub bearish_arrow: Vec<f64>,
    pub bullish_change: Vec<f64>,
    pub bearish_change: Vec<f64>,
    pub hh: Vec<f64>,
    pub lh: Vec<f64>,
    pub hl: Vec<f64>,
    pub ll: Vec<f64>,
    pub bullish_bos: Vec<f64>,
    pub bullish_choch: Vec<f64>,
    pub bearish_bos: Vec<f64>,
    pub bearish_choch: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = market_structure_confluence_js)]
pub fn market_structure_confluence_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    swing_size: usize,
    bos_confirmation: String,
    basis_length: usize,
    atr_length: usize,
    atr_smooth: usize,
    vol_mult: f64,
) -> Result<JsValue, JsValue> {
    let input = MarketStructureConfluenceInput::from_slices(
        high,
        low,
        close,
        MarketStructureConfluenceParams {
            swing_size: Some(swing_size),
            bos_confirmation: Some(bos_confirmation),
            basis_length: Some(basis_length),
            atr_length: Some(atr_length),
            atr_smooth: Some(atr_smooth),
            vol_mult: Some(vol_mult),
        },
    );
    let output = market_structure_confluence_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&MarketStructureConfluenceJsOutput {
        basis: output.basis,
        upper_band: output.upper_band,
        lower_band: output.lower_band,
        structure_direction: output.structure_direction,
        bullish_arrow: output.bullish_arrow,
        bearish_arrow: output.bearish_arrow,
        bullish_change: output.bullish_change,
        bearish_change: output.bearish_change,
        hh: output.hh,
        lh: output.lh,
        hl: output.hl,
        ll: output.ll,
        bullish_bos: output.bullish_bos,
        bullish_choch: output.bullish_choch,
        bearish_bos: output.bearish_bos,
        bearish_choch: output.bearish_choch,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct MarketStructureConfluenceBatchConfig {
    pub swing_size_range: (usize, usize, usize),
    pub bos_confirmation_options: Vec<String>,
    pub basis_length_range: (usize, usize, usize),
    pub atr_length_range: (usize, usize, usize),
    pub atr_smooth_range: (usize, usize, usize),
    pub vol_mult_range: (f64, f64, f64),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct MarketStructureConfluenceBatchJsOutput {
    pub basis: Vec<f64>,
    pub upper_band: Vec<f64>,
    pub lower_band: Vec<f64>,
    pub structure_direction: Vec<f64>,
    pub bullish_arrow: Vec<f64>,
    pub bearish_arrow: Vec<f64>,
    pub bullish_change: Vec<f64>,
    pub bearish_change: Vec<f64>,
    pub hh: Vec<f64>,
    pub lh: Vec<f64>,
    pub hl: Vec<f64>,
    pub ll: Vec<f64>,
    pub bullish_bos: Vec<f64>,
    pub bullish_choch: Vec<f64>,
    pub bearish_bos: Vec<f64>,
    pub bearish_choch: Vec<f64>,
    pub swing_sizes: Vec<usize>,
    pub bos_confirmations: Vec<String>,
    pub basis_lengths: Vec<usize>,
    pub atr_lengths: Vec<usize>,
    pub atr_smooths: Vec<usize>,
    pub vol_mults: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = market_structure_confluence_batch)]
pub fn market_structure_confluence_batch_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let cfg: MarketStructureConfluenceBatchConfig =
        serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let output = market_structure_confluence_batch_with_kernel(
        high,
        low,
        close,
        &MarketStructureConfluenceBatchRange {
            swing_size: cfg.swing_size_range,
            bos_confirmation: cfg.bos_confirmation_options,
            basis_length: cfg.basis_length_range,
            atr_length: cfg.atr_length_range,
            atr_smooth: cfg.atr_smooth_range,
            vol_mult: cfg.vol_mult_range,
        },
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    serde_wasm_bindgen::to_value(&MarketStructureConfluenceBatchJsOutput {
        basis: output.basis,
        upper_band: output.upper_band,
        lower_band: output.lower_band,
        structure_direction: output.structure_direction,
        bullish_arrow: output.bullish_arrow,
        bearish_arrow: output.bearish_arrow,
        bullish_change: output.bullish_change,
        bearish_change: output.bearish_change,
        hh: output.hh,
        lh: output.lh,
        hl: output.hl,
        ll: output.ll,
        bullish_bos: output.bullish_bos,
        bullish_choch: output.bullish_choch,
        bearish_bos: output.bearish_bos,
        bearish_choch: output.bearish_choch,
        swing_sizes: output
            .combos
            .iter()
            .map(|combo| combo.swing_size.unwrap_or(DEFAULT_SWING_SIZE))
            .collect(),
        bos_confirmations: output
            .combos
            .iter()
            .map(|combo| {
                combo
                    .bos_confirmation
                    .clone()
                    .unwrap_or_else(|| DEFAULT_BOS_CONFIRMATION.to_string())
            })
            .collect(),
        basis_lengths: output
            .combos
            .iter()
            .map(|combo| combo.basis_length.unwrap_or(DEFAULT_BASIS_LENGTH))
            .collect(),
        atr_lengths: output
            .combos
            .iter()
            .map(|combo| combo.atr_length.unwrap_or(DEFAULT_ATR_LENGTH))
            .collect(),
        atr_smooths: output
            .combos
            .iter()
            .map(|combo| combo.atr_smooth.unwrap_or(DEFAULT_ATR_SMOOTH))
            .collect(),
        vol_mults: output
            .combos
            .iter()
            .map(|combo| combo.vol_mult.unwrap_or(DEFAULT_VOL_MULT))
            .collect(),
        rows: output.rows,
        cols: output.cols,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn market_structure_confluence_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    swing_size: usize,
    bos_confirmation: String,
    basis_length: usize,
    atr_length: usize,
    atr_smooth: usize,
    vol_mult: f64,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = market_structure_confluence_js(
        high,
        low,
        close,
        swing_size,
        bos_confirmation,
        basis_length,
        atr_length,
        atr_smooth,
        vol_mult,
    )?;
    crate::write_wasm_object_f64_outputs("market_structure_confluence_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn market_structure_confluence_batch_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = market_structure_confluence_batch_js(high, low, close, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "market_structure_confluence_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_ohlc() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut high = Vec::with_capacity(420);
        let mut low = Vec::with_capacity(420);
        let mut close = Vec::with_capacity(420);
        for i in 0..420 {
            let base = 100.0 + i as f64 * 0.08 + (i as f64 * 0.11).sin() * 1.7;
            let c = base + (i as f64 * 0.07).cos() * 0.45;
            let h = c + 0.8 + (i as f64 * 0.09).sin().abs() * 0.4;
            let l = c - 0.8 - (i as f64 * 0.13).cos().abs() * 0.35;
            high.push(h);
            low.push(l);
            close.push(c);
        }
        (high, low, close)
    }

    #[test]
    fn market_structure_confluence_into_matches_single() {
        let (high, low, close) = sample_ohlc();
        let input = MarketStructureConfluenceInput::from_slices(
            &high,
            &low,
            &close,
            MarketStructureConfluenceParams::default(),
        );
        let out = market_structure_confluence_with_kernel(&input, Kernel::Scalar).expect("single");
        let mut basis = vec![0.0; close.len()];
        let mut upper_band = vec![0.0; close.len()];
        let mut lower_band = vec![0.0; close.len()];
        let mut structure_direction = vec![0.0; close.len()];
        let mut bullish_arrow = vec![0.0; close.len()];
        let mut bearish_arrow = vec![0.0; close.len()];
        let mut bullish_change = vec![0.0; close.len()];
        let mut bearish_change = vec![0.0; close.len()];
        let mut hh = vec![0.0; close.len()];
        let mut lh = vec![0.0; close.len()];
        let mut hl = vec![0.0; close.len()];
        let mut ll = vec![0.0; close.len()];
        let mut bullish_bos = vec![0.0; close.len()];
        let mut bullish_choch = vec![0.0; close.len()];
        let mut bearish_bos = vec![0.0; close.len()];
        let mut bearish_choch = vec![0.0; close.len()];

        market_structure_confluence_into_slices(
            &input,
            Kernel::Scalar,
            &mut basis,
            &mut upper_band,
            &mut lower_band,
            &mut structure_direction,
            &mut bullish_arrow,
            &mut bearish_arrow,
            &mut bullish_change,
            &mut bearish_change,
            &mut hh,
            &mut lh,
            &mut hl,
            &mut ll,
            &mut bullish_bos,
            &mut bullish_choch,
            &mut bearish_bos,
            &mut bearish_choch,
        )
        .expect("into");

        for i in 0..close.len() {
            for (lhs, rhs) in [
                (out.basis[i], basis[i]),
                (out.upper_band[i], upper_band[i]),
                (out.lower_band[i], lower_band[i]),
                (out.structure_direction[i], structure_direction[i]),
                (out.bullish_arrow[i], bullish_arrow[i]),
                (out.bearish_arrow[i], bearish_arrow[i]),
                (out.bullish_change[i], bullish_change[i]),
                (out.bearish_change[i], bearish_change[i]),
                (out.hh[i], hh[i]),
                (out.lh[i], lh[i]),
                (out.hl[i], hl[i]),
                (out.ll[i], ll[i]),
                (out.bullish_bos[i], bullish_bos[i]),
                (out.bullish_choch[i], bullish_choch[i]),
                (out.bearish_bos[i], bearish_bos[i]),
                (out.bearish_choch[i], bearish_choch[i]),
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
    fn market_structure_confluence_stream_matches_batch() {
        let (high, low, close) = sample_ohlc();
        let input = MarketStructureConfluenceInput::from_slices(
            &high,
            &low,
            &close,
            MarketStructureConfluenceParams::default(),
        );
        let out = market_structure_confluence(&input).expect("batch");
        let mut stream =
            MarketStructureConfluenceStream::try_new(MarketStructureConfluenceParams::default())
                .expect("stream");
        let mut collected = Vec::with_capacity(close.len());
        for i in 0..close.len() {
            collected.push(stream.update(high[i], low[i], close[i]));
        }
        for i in 0..close.len() {
            let Some(point) = collected[i] else {
                assert!(out.basis[i].is_nan());
                continue;
            };
            for (lhs, rhs) in [
                (point.basis, out.basis[i]),
                (point.upper_band, out.upper_band[i]),
                (point.lower_band, out.lower_band[i]),
                (point.structure_direction, out.structure_direction[i]),
                (point.bullish_arrow, out.bullish_arrow[i]),
                (point.bearish_arrow, out.bearish_arrow[i]),
                (point.bullish_change, out.bullish_change[i]),
                (point.bearish_change, out.bearish_change[i]),
                (point.hh, out.hh[i]),
                (point.lh, out.lh[i]),
                (point.hl, out.hl[i]),
                (point.ll, out.ll[i]),
                (point.bullish_bos, out.bullish_bos[i]),
                (point.bullish_choch, out.bullish_choch[i]),
                (point.bearish_bos, out.bearish_bos[i]),
                (point.bearish_choch, out.bearish_choch[i]),
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
    fn market_structure_confluence_batch_first_row_matches_single() {
        let (high, low, close) = sample_ohlc();
        let single = market_structure_confluence(&MarketStructureConfluenceInput::from_slices(
            &high,
            &low,
            &close,
            MarketStructureConfluenceParams::default(),
        ))
        .expect("single");
        let batch = market_structure_confluence_batch_with_kernel(
            &high,
            &low,
            &close,
            &MarketStructureConfluenceBatchRange {
                swing_size: (10, 12, 2),
                bos_confirmation: vec!["Candle Close".to_string()],
                basis_length: (100, 100, 0),
                atr_length: (14, 14, 0),
                atr_smooth: (21, 21, 0),
                vol_mult: (2.0, 2.0, 0.0),
            },
            Kernel::ScalarBatch,
        )
        .expect("batch");
        assert_eq!(batch.rows, 2);
        assert_eq!(batch.cols, close.len());
        for i in 0..close.len() {
            let idx = i;
            for (lhs, rhs) in [
                (single.basis[i], batch.basis[idx]),
                (single.upper_band[i], batch.upper_band[idx]),
                (single.lower_band[i], batch.lower_band[idx]),
                (
                    single.structure_direction[i],
                    batch.structure_direction[idx],
                ),
                (single.bullish_arrow[i], batch.bullish_arrow[idx]),
                (single.bearish_arrow[i], batch.bearish_arrow[idx]),
                (single.bullish_change[i], batch.bullish_change[idx]),
                (single.bearish_change[i], batch.bearish_change[idx]),
                (single.hh[i], batch.hh[idx]),
                (single.lh[i], batch.lh[idx]),
                (single.hl[i], batch.hl[idx]),
                (single.ll[i], batch.ll[idx]),
                (single.bullish_bos[i], batch.bullish_bos[idx]),
                (single.bullish_choch[i], batch.bullish_choch[idx]),
                (single.bearish_bos[i], batch.bearish_bos[idx]),
                (single.bearish_choch[i], batch.bearish_choch[idx]),
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
    fn market_structure_confluence_rejects_invalid_params() {
        let (high, low, close) = sample_ohlc();
        let err = market_structure_confluence(&MarketStructureConfluenceInput::from_slices(
            &high,
            &low,
            &close,
            MarketStructureConfluenceParams {
                swing_size: Some(1),
                ..MarketStructureConfluenceParams::default()
            },
        ))
        .expect_err("invalid swing size");
        assert!(err.to_string().contains("invalid swing_size"));

        let err = MarketStructureConfluenceStream::try_new(MarketStructureConfluenceParams {
            bos_confirmation: Some("bad".to_string()),
            ..MarketStructureConfluenceParams::default()
        })
        .expect_err("invalid confirmation");
        assert!(err.to_string().contains("invalid bos_confirmation"));
    }
}
