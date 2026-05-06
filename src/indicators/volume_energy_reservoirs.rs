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
    alloc_with_nan_prefix, detect_best_batch_kernel, init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::collections::VecDeque;
use std::mem::ManuallyDrop;
use thiserror::Error;

const DEFAULT_LENGTH: usize = 20;
const DEFAULT_SENSITIVITY: f64 = 1.5;
const MIN_LENGTH: usize = 5;
const VOLUME_STDEV_LENGTH: usize = 100;
const MOMENTUM_EMA_LENGTH: usize = 5;
const MOMENTUM_EMA_ALPHA: f64 = 2.0 / (MOMENTUM_EMA_LENGTH as f64 + 1.0);
const RESERVOIR_CAP: f64 = 10.0;
const RESERVOIR_SQUEEZE_THRESHOLD: f64 = 5.0;
const STABILITY_THRESHOLD: f64 = 0.2;
const FLOAT_TOL: f64 = 1e-12;

#[derive(Debug, Clone)]
pub enum VolumeEnergyReservoirsData<'a> {
    Candles(&'a Candles),
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct VolumeEnergyReservoirsOutput {
    pub momentum: Vec<f64>,
    pub reservoir: Vec<f64>,
    pub squeeze_active: Vec<f64>,
    pub squeeze_start: Vec<f64>,
    pub range_high: Vec<f64>,
    pub range_low: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
pub struct VolumeEnergyReservoirsPoint {
    pub momentum: f64,
    pub reservoir: f64,
    pub squeeze_active: f64,
    pub squeeze_start: f64,
    pub range_high: f64,
    pub range_low: f64,
}

impl VolumeEnergyReservoirsPoint {
    #[inline(always)]
    fn nan() -> Self {
        Self {
            momentum: f64::NAN,
            reservoir: f64::NAN,
            squeeze_active: f64::NAN,
            squeeze_start: f64::NAN,
            range_high: f64::NAN,
            range_low: f64::NAN,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum VolumeEnergyReservoirsOutputField {
    Momentum,
    Reservoir,
    SqueezeActive,
    SqueezeStart,
    RangeHigh,
    RangeLow,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct VolumeEnergyReservoirsParams {
    pub length: Option<usize>,
    pub sensitivity: Option<f64>,
}

impl Default for VolumeEnergyReservoirsParams {
    fn default() -> Self {
        Self {
            length: Some(DEFAULT_LENGTH),
            sensitivity: Some(DEFAULT_SENSITIVITY),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VolumeEnergyReservoirsInput<'a> {
    pub data: VolumeEnergyReservoirsData<'a>,
    pub params: VolumeEnergyReservoirsParams,
}

impl<'a> VolumeEnergyReservoirsInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: VolumeEnergyReservoirsParams) -> Self {
        Self {
            data: VolumeEnergyReservoirsData::Candles(candles),
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
        params: VolumeEnergyReservoirsParams,
    ) -> Self {
        Self {
            data: VolumeEnergyReservoirsData::Slices {
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
        Self::from_candles(candles, VolumeEnergyReservoirsParams::default())
    }

    #[inline]
    pub fn as_slices(&self) -> (&'a [f64], &'a [f64], &'a [f64], &'a [f64]) {
        match &self.data {
            VolumeEnergyReservoirsData::Candles(candles) => {
                (&candles.high, &candles.low, &candles.close, &candles.volume)
            }
            VolumeEnergyReservoirsData::Slices {
                high,
                low,
                close,
                volume,
            } => (high, low, close, volume),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct VolumeEnergyReservoirsBuilder {
    length: Option<usize>,
    sensitivity: Option<f64>,
    kernel: Kernel,
}

impl VolumeEnergyReservoirsBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn length(mut self, value: usize) -> Self {
        self.length = Some(value);
        self
    }

    #[inline]
    pub fn sensitivity(mut self, value: f64) -> Self {
        self.sensitivity = Some(value);
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
    ) -> Result<VolumeEnergyReservoirsOutput, VolumeEnergyReservoirsError> {
        let input = VolumeEnergyReservoirsInput::from_candles(
            candles,
            VolumeEnergyReservoirsParams {
                length: self.length,
                sensitivity: self.sensitivity,
            },
        );
        volume_energy_reservoirs_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Result<VolumeEnergyReservoirsOutput, VolumeEnergyReservoirsError> {
        let input = VolumeEnergyReservoirsInput::from_slices(
            high,
            low,
            close,
            volume,
            VolumeEnergyReservoirsParams {
                length: self.length,
                sensitivity: self.sensitivity,
            },
        );
        volume_energy_reservoirs_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn into_stream(self) -> Result<VolumeEnergyReservoirsStream, VolumeEnergyReservoirsError> {
        VolumeEnergyReservoirsStream::try_new(VolumeEnergyReservoirsParams {
            length: self.length,
            sensitivity: self.sensitivity,
        })
    }
}

#[derive(Debug, Error)]
pub enum VolumeEnergyReservoirsError {
    #[error("volume_energy_reservoirs: Input data slice is empty.")]
    EmptyInputData,
    #[error("volume_energy_reservoirs: All values are NaN.")]
    AllValuesNaN,
    #[error(
        "volume_energy_reservoirs: Inconsistent slice lengths - high={high_len}, low={low_len}, close={close_len}, volume={volume_len}"
    )]
    MismatchedInputLengths {
        high_len: usize,
        low_len: usize,
        close_len: usize,
        volume_len: usize,
    },
    #[error(
        "volume_energy_reservoirs: Invalid length: length = {length}, data length = {data_len}"
    )]
    InvalidLength { length: usize, data_len: usize },
    #[error("volume_energy_reservoirs: Invalid sensitivity: {sensitivity}")]
    InvalidSensitivity { sensitivity: f64 },
    #[error("volume_energy_reservoirs: Output length mismatch: expected = {expected}")]
    OutputLengthMismatch { expected: usize },
    #[error("volume_energy_reservoirs: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("volume_energy_reservoirs: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Clone, Copy, Debug)]
struct ResolvedParams {
    length: usize,
    sensitivity: f64,
}

#[inline(always)]
fn first_valid_ohlcv(high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> usize {
    let len = high.len();
    let mut i = 0usize;
    while i < len {
        if high[i].is_finite()
            && low[i].is_finite()
            && close[i].is_finite()
            && volume[i].is_finite()
        {
            return i;
        }
        i += 1;
    }
    len
}

#[inline(always)]
fn resolve_params(
    params: &VolumeEnergyReservoirsParams,
    data_len: Option<usize>,
) -> Result<ResolvedParams, VolumeEnergyReservoirsError> {
    let length = params.length.unwrap_or(DEFAULT_LENGTH);
    let sensitivity = params.sensitivity.unwrap_or(DEFAULT_SENSITIVITY);
    let data_len = data_len.unwrap_or(0);

    if length < MIN_LENGTH {
        return Err(VolumeEnergyReservoirsError::InvalidLength { length, data_len });
    }
    if !sensitivity.is_finite() || sensitivity < 0.5 {
        return Err(VolumeEnergyReservoirsError::InvalidSensitivity { sensitivity });
    }

    Ok(ResolvedParams {
        length,
        sensitivity,
    })
}

#[derive(Clone, Debug)]
struct ReservoirCoreState {
    params: ResolvedParams,
    segment_index: usize,
    volume_ring: [f64; VOLUME_STDEV_LENGTH],
    volume_head: usize,
    volume_count: usize,
    volume_sum: f64,
    volume_sum_sq: f64,
    high_window: VecDeque<(usize, f64)>,
    low_window: VecDeque<(usize, f64)>,
    reservoir: f64,
    ema: f64,
    ema_ready: bool,
    prev_squeeze_active: bool,
    current_high: f64,
    current_low: f64,
    has_range: bool,
    is_extending: bool,
}

impl ReservoirCoreState {
    #[inline]
    fn new(params: ResolvedParams) -> Self {
        Self {
            params,
            segment_index: 0,
            volume_ring: [0.0; VOLUME_STDEV_LENGTH],
            volume_head: 0,
            volume_count: 0,
            volume_sum: 0.0,
            volume_sum_sq: 0.0,
            high_window: VecDeque::with_capacity(params.length.max(4)),
            low_window: VecDeque::with_capacity(params.length.max(4)),
            reservoir: 0.0,
            ema: 0.0,
            ema_ready: false,
            prev_squeeze_active: false,
            current_high: f64::NAN,
            current_low: f64::NAN,
            has_range: false,
            is_extending: false,
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.segment_index = 0;
        self.volume_ring = [0.0; VOLUME_STDEV_LENGTH];
        self.volume_head = 0;
        self.volume_count = 0;
        self.volume_sum = 0.0;
        self.volume_sum_sq = 0.0;
        self.high_window.clear();
        self.low_window.clear();
        self.reservoir = 0.0;
        self.ema = 0.0;
        self.ema_ready = false;
        self.prev_squeeze_active = false;
        self.current_high = f64::NAN;
        self.current_low = f64::NAN;
        self.has_range = false;
        self.is_extending = false;
    }

    #[inline(always)]
    fn update(
        &mut self,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> VolumeEnergyReservoirsPoint {
        let idx = self.segment_index;
        self.segment_index += 1;

        self.push_volume(volume);
        self.push_high(idx, high);
        self.push_low(idx, low);

        let hi = self
            .high_window
            .front()
            .map(|entry| entry.1)
            .unwrap_or(high);
        let lo = self.low_window.front().map(|entry| entry.1).unwrap_or(low);
        let mid_price = 0.5 * (hi + lo);
        let price_range = hi - lo;
        let hl2 = 0.5 * (high + low);
        let price_rel = if price_range.abs() <= FLOAT_TOL {
            0.0
        } else {
            (hl2 - mid_price) / price_range
        };

        let norm_vol = self.normalized_volume(volume);
        if norm_vol < 1.0 && price_rel.abs() < STABILITY_THRESHOLD {
            self.reservoir += 0.5;
        } else if norm_vol > self.params.sensitivity {
            self.reservoir *= 0.7;
        } else {
            self.reservoir = (self.reservoir - 0.1).max(0.0);
        }
        self.reservoir = self.reservoir.min(RESERVOIR_CAP);

        let momentum = price_rel * norm_vol * 20.0;
        if !self.ema_ready {
            self.ema = momentum;
            self.ema_ready = true;
        } else {
            self.ema += MOMENTUM_EMA_ALPHA * (momentum - self.ema);
        }

        let squeeze_active = self.reservoir > RESERVOIR_SQUEEZE_THRESHOLD;
        let squeeze_start = squeeze_active && !self.prev_squeeze_active;
        let squeeze_end = !squeeze_active && self.prev_squeeze_active;

        if squeeze_start {
            self.current_high = high;
            self.current_low = low;
            self.has_range = true;
            self.is_extending = false;
        }

        if squeeze_active && self.has_range {
            self.current_high = self.current_high.max(high);
            self.current_low = self.current_low.min(low);
        }

        let mut range_visible = squeeze_active || self.is_extending;
        if squeeze_end && self.has_range {
            self.is_extending = true;
            range_visible = true;
        }
        if self.is_extending && self.has_range {
            range_visible = true;
            if close > self.current_high || close < self.current_low {
                self.is_extending = false;
            }
        }

        self.prev_squeeze_active = squeeze_active;

        VolumeEnergyReservoirsPoint {
            momentum: self.ema,
            reservoir: self.reservoir,
            squeeze_active: if squeeze_active { 1.0 } else { 0.0 },
            squeeze_start: if squeeze_start { 1.0 } else { 0.0 },
            range_high: if range_visible && self.has_range {
                self.current_high
            } else {
                f64::NAN
            },
            range_low: if range_visible && self.has_range {
                self.current_low
            } else {
                f64::NAN
            },
        }
    }

    #[inline(always)]
    fn push_volume(&mut self, value: f64) {
        if self.volume_count == VOLUME_STDEV_LENGTH {
            let old = self.volume_ring[self.volume_head];
            self.volume_sum -= old;
            self.volume_sum_sq -= old * old;
        } else {
            self.volume_count += 1;
        }
        self.volume_ring[self.volume_head] = value;
        self.volume_head = (self.volume_head + 1) % VOLUME_STDEV_LENGTH;
        self.volume_sum += value;
        self.volume_sum_sq += value * value;
    }

    #[inline(always)]
    fn normalized_volume(&self, volume: f64) -> f64 {
        if self.volume_count < VOLUME_STDEV_LENGTH {
            return 0.0;
        }
        let mean = self.volume_sum / VOLUME_STDEV_LENGTH as f64;
        let variance = (self.volume_sum_sq / VOLUME_STDEV_LENGTH as f64 - mean * mean).max(0.0);
        let stdev = variance.sqrt();
        if stdev.abs() <= FLOAT_TOL {
            1.0
        } else {
            volume / stdev
        }
    }

    #[inline(always)]
    fn push_high(&mut self, idx: usize, value: f64) {
        while let Some((_, tail)) = self.high_window.back().copied() {
            if tail <= value {
                self.high_window.pop_back();
            } else {
                break;
            }
        }
        self.high_window.push_back((idx, value));
        while let Some((front_idx, _)) = self.high_window.front().copied() {
            if front_idx + self.params.length <= idx {
                self.high_window.pop_front();
            } else {
                break;
            }
        }
    }

    #[inline(always)]
    fn push_low(&mut self, idx: usize, value: f64) {
        while let Some((_, tail)) = self.low_window.back().copied() {
            if tail >= value {
                self.low_window.pop_back();
            } else {
                break;
            }
        }
        self.low_window.push_back((idx, value));
        while let Some((front_idx, _)) = self.low_window.front().copied() {
            if front_idx + self.params.length <= idx {
                self.low_window.pop_front();
            } else {
                break;
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct VolumeEnergyReservoirsStream {
    state: ReservoirCoreState,
}

impl VolumeEnergyReservoirsStream {
    #[inline]
    pub fn try_new(
        params: VolumeEnergyReservoirsParams,
    ) -> Result<Self, VolumeEnergyReservoirsError> {
        let params = resolve_params(&params, None)?;
        Ok(Self {
            state: ReservoirCoreState::new(params),
        })
    }

    #[inline]
    pub fn update(
        &mut self,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Option<VolumeEnergyReservoirsPoint> {
        if !high.is_finite() || !low.is_finite() || !close.is_finite() || !volume.is_finite() {
            self.state.reset();
            return None;
        }
        Some(self.state.update(high, low, close, volume))
    }

    #[inline]
    pub fn reset(&mut self) {
        self.state.reset();
    }

    #[inline]
    pub fn get_warmup_period(&self) -> usize {
        0
    }
}

#[inline(always)]
fn volume_energy_reservoirs_row_from_slices(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    params: ResolvedParams,
    momentum: &mut [f64],
    reservoir: &mut [f64],
    squeeze_active: &mut [f64],
    squeeze_start: &mut [f64],
    range_high: &mut [f64],
    range_low: &mut [f64],
) {
    let len = high.len();
    debug_assert_eq!(low.len(), len);
    debug_assert_eq!(close.len(), len);
    debug_assert_eq!(volume.len(), len);
    debug_assert_eq!(momentum.len(), len);
    debug_assert_eq!(reservoir.len(), len);
    debug_assert_eq!(squeeze_active.len(), len);
    debug_assert_eq!(squeeze_start.len(), len);
    debug_assert_eq!(range_high.len(), len);
    debug_assert_eq!(range_low.len(), len);

    let mut state = ReservoirCoreState::new(params);
    let mut i = 0usize;
    while i < len {
        let h = high[i];
        let l = low[i];
        let c = close[i];
        let v = volume[i];
        let point = if h.is_finite() && l.is_finite() && c.is_finite() && v.is_finite() {
            state.update(h, l, c, v)
        } else {
            state.reset();
            VolumeEnergyReservoirsPoint::nan()
        };
        momentum[i] = point.momentum;
        reservoir[i] = point.reservoir;
        squeeze_active[i] = point.squeeze_active;
        squeeze_start[i] = point.squeeze_start;
        range_high[i] = point.range_high;
        range_low[i] = point.range_low;
        i += 1;
    }
}

#[inline(always)]
fn volume_energy_reservoirs_selected_row_from_slices(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    params: ResolvedParams,
    field: VolumeEnergyReservoirsOutputField,
    out: &mut [f64],
) {
    let len = high.len();
    debug_assert_eq!(low.len(), len);
    debug_assert_eq!(close.len(), len);
    debug_assert_eq!(volume.len(), len);
    debug_assert_eq!(out.len(), len);

    macro_rules! write_field {
        ($member:ident) => {{
            let mut state = ReservoirCoreState::new(params);
            let mut i = 0usize;
            while i < len {
                let h = high[i];
                let l = low[i];
                let c = close[i];
                let v = volume[i];
                let point = if h.is_finite() && l.is_finite() && c.is_finite() && v.is_finite() {
                    state.update(h, l, c, v)
                } else {
                    state.reset();
                    VolumeEnergyReservoirsPoint::nan()
                };
                out[i] = point.$member;
                i += 1;
            }
        }};
    }

    match field {
        VolumeEnergyReservoirsOutputField::Momentum => write_field!(momentum),
        VolumeEnergyReservoirsOutputField::Reservoir => write_field!(reservoir),
        VolumeEnergyReservoirsOutputField::SqueezeActive => write_field!(squeeze_active),
        VolumeEnergyReservoirsOutputField::SqueezeStart => write_field!(squeeze_start),
        VolumeEnergyReservoirsOutputField::RangeHigh => write_field!(range_high),
        VolumeEnergyReservoirsOutputField::RangeLow => write_field!(range_low),
    }
}

#[inline]
pub fn volume_energy_reservoirs(
    input: &VolumeEnergyReservoirsInput,
) -> Result<VolumeEnergyReservoirsOutput, VolumeEnergyReservoirsError> {
    volume_energy_reservoirs_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn volume_energy_reservoirs_with_kernel(
    input: &VolumeEnergyReservoirsInput,
    _kernel: Kernel,
) -> Result<VolumeEnergyReservoirsOutput, VolumeEnergyReservoirsError> {
    let (high, low, close, volume) = input.as_slices();
    if high.is_empty() || low.is_empty() || close.is_empty() || volume.is_empty() {
        return Err(VolumeEnergyReservoirsError::EmptyInputData);
    }
    if high.len() != low.len() || high.len() != close.len() || high.len() != volume.len() {
        return Err(VolumeEnergyReservoirsError::MismatchedInputLengths {
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
            volume_len: volume.len(),
        });
    }
    if first_valid_ohlcv(high, low, close, volume) >= high.len() {
        return Err(VolumeEnergyReservoirsError::AllValuesNaN);
    }

    let params = resolve_params(&input.params, Some(high.len()))?;
    let len = high.len();
    let mut momentum = alloc_with_nan_prefix(len, 0);
    let mut reservoir = alloc_with_nan_prefix(len, 0);
    let mut squeeze_active = alloc_with_nan_prefix(len, 0);
    let mut squeeze_start = alloc_with_nan_prefix(len, 0);
    let mut range_high = alloc_with_nan_prefix(len, 0);
    let mut range_low = alloc_with_nan_prefix(len, 0);
    volume_energy_reservoirs_row_from_slices(
        high,
        low,
        close,
        volume,
        params,
        &mut momentum,
        &mut reservoir,
        &mut squeeze_active,
        &mut squeeze_start,
        &mut range_high,
        &mut range_low,
    );
    Ok(VolumeEnergyReservoirsOutput {
        momentum,
        reservoir,
        squeeze_active,
        squeeze_start,
        range_high,
        range_low,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn volume_energy_reservoirs_into_slices(
    momentum_out: &mut [f64],
    reservoir_out: &mut [f64],
    squeeze_active_out: &mut [f64],
    squeeze_start_out: &mut [f64],
    range_high_out: &mut [f64],
    range_low_out: &mut [f64],
    input: &VolumeEnergyReservoirsInput,
    _kernel: Kernel,
) -> Result<(), VolumeEnergyReservoirsError> {
    let (high, low, close, volume) = input.as_slices();
    if high.is_empty() || low.is_empty() || close.is_empty() || volume.is_empty() {
        return Err(VolumeEnergyReservoirsError::EmptyInputData);
    }
    if high.len() != low.len() || high.len() != close.len() || high.len() != volume.len() {
        return Err(VolumeEnergyReservoirsError::MismatchedInputLengths {
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
            volume_len: volume.len(),
        });
    }
    let expected = high.len();
    if momentum_out.len() != expected
        || reservoir_out.len() != expected
        || squeeze_active_out.len() != expected
        || squeeze_start_out.len() != expected
        || range_high_out.len() != expected
        || range_low_out.len() != expected
    {
        return Err(VolumeEnergyReservoirsError::OutputLengthMismatch { expected });
    }
    if first_valid_ohlcv(high, low, close, volume) >= high.len() {
        return Err(VolumeEnergyReservoirsError::AllValuesNaN);
    }
    let params = resolve_params(&input.params, Some(high.len()))?;
    volume_energy_reservoirs_row_from_slices(
        high,
        low,
        close,
        volume,
        params,
        momentum_out,
        reservoir_out,
        squeeze_active_out,
        squeeze_start_out,
        range_high_out,
        range_low_out,
    );
    Ok(())
}

#[inline]
pub(crate) fn volume_energy_reservoirs_output_into_slice(
    out: &mut [f64],
    input: &VolumeEnergyReservoirsInput,
    _kernel: Kernel,
    field: VolumeEnergyReservoirsOutputField,
) -> Result<(), VolumeEnergyReservoirsError> {
    let (high, low, close, volume) = input.as_slices();
    if high.is_empty() || low.is_empty() || close.is_empty() || volume.is_empty() {
        return Err(VolumeEnergyReservoirsError::EmptyInputData);
    }
    if high.len() != low.len() || high.len() != close.len() || high.len() != volume.len() {
        return Err(VolumeEnergyReservoirsError::MismatchedInputLengths {
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
            volume_len: volume.len(),
        });
    }
    let expected = high.len();
    if out.len() != expected {
        return Err(VolumeEnergyReservoirsError::OutputLengthMismatch { expected });
    }
    if first_valid_ohlcv(high, low, close, volume) >= high.len() {
        return Err(VolumeEnergyReservoirsError::AllValuesNaN);
    }
    let params = resolve_params(&input.params, Some(high.len()))?;
    volume_energy_reservoirs_selected_row_from_slices(high, low, close, volume, params, field, out);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn volume_energy_reservoirs_into(
    momentum_out: &mut [f64],
    reservoir_out: &mut [f64],
    squeeze_active_out: &mut [f64],
    squeeze_start_out: &mut [f64],
    range_high_out: &mut [f64],
    range_low_out: &mut [f64],
    input: &VolumeEnergyReservoirsInput,
) -> Result<(), VolumeEnergyReservoirsError> {
    volume_energy_reservoirs_into_slices(
        momentum_out,
        reservoir_out,
        squeeze_active_out,
        squeeze_start_out,
        range_high_out,
        range_low_out,
        input,
        Kernel::Auto,
    )
}

#[derive(Debug, Clone, PartialEq)]
pub struct VolumeEnergyReservoirsBatchRange {
    pub length: (usize, usize, usize),
    pub sensitivity: (f64, f64, f64),
}

impl Default for VolumeEnergyReservoirsBatchRange {
    fn default() -> Self {
        Self {
            length: (DEFAULT_LENGTH, DEFAULT_LENGTH, 0),
            sensitivity: (DEFAULT_SENSITIVITY, DEFAULT_SENSITIVITY, 0.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VolumeEnergyReservoirsBatchOutput {
    pub momentum: Vec<f64>,
    pub reservoir: Vec<f64>,
    pub squeeze_active: Vec<f64>,
    pub squeeze_start: Vec<f64>,
    pub range_high: Vec<f64>,
    pub range_low: Vec<f64>,
    pub combos: Vec<VolumeEnergyReservoirsParams>,
    pub rows: usize,
    pub cols: usize,
}

impl VolumeEnergyReservoirsBatchOutput {
    #[inline]
    pub fn params_for(&self, row: usize) -> Option<&VolumeEnergyReservoirsParams> {
        self.combos.get(row)
    }

    #[inline]
    pub fn row_slices(
        &self,
        row: usize,
    ) -> Option<(&[f64], &[f64], &[f64], &[f64], &[f64], &[f64])> {
        if row >= self.rows {
            return None;
        }
        let start = row * self.cols;
        let end = start + self.cols;
        Some((
            &self.momentum[start..end],
            &self.reservoir[start..end],
            &self.squeeze_active[start..end],
            &self.squeeze_start[start..end],
            &self.range_high[start..end],
            &self.range_low[start..end],
        ))
    }
}

#[derive(Clone, Debug, Default)]
pub struct VolumeEnergyReservoirsBatchBuilder {
    range: VolumeEnergyReservoirsBatchRange,
    kernel: Kernel,
}

impl VolumeEnergyReservoirsBatchBuilder {
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
    pub fn sensitivity_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.sensitivity = (start, end, step);
        self
    }

    #[inline]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Result<VolumeEnergyReservoirsBatchOutput, VolumeEnergyReservoirsError> {
        volume_energy_reservoirs_batch_with_kernel(
            high,
            low,
            close,
            volume,
            &self.range,
            self.kernel,
        )
    }

    #[inline]
    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<VolumeEnergyReservoirsBatchOutput, VolumeEnergyReservoirsError> {
        self.apply_slices(&candles.high, &candles.low, &candles.close, &candles.volume)
    }
}

#[inline(always)]
fn expand_axis_usize(
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, VolumeEnergyReservoirsError> {
    if step == 0 || start == end {
        return Ok(vec![start]);
    }
    let mut out = Vec::new();
    if start < end {
        let mut value = start;
        while value <= end {
            out.push(value);
            let next = value.saturating_add(step);
            if next == value {
                break;
            }
            value = next;
        }
    } else {
        let mut value = start;
        loop {
            out.push(value);
            if value == end {
                break;
            }
            let next = value.saturating_sub(step);
            if next == value || next < end {
                break;
            }
            value = next;
        }
    }
    if out.is_empty() {
        return Err(VolumeEnergyReservoirsError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

#[inline(always)]
fn expand_axis_f64(
    start: f64,
    end: f64,
    step: f64,
) -> Result<Vec<f64>, VolumeEnergyReservoirsError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() || start > end {
        return Err(VolumeEnergyReservoirsError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if (start - end).abs() < FLOAT_TOL {
        if step.abs() > FLOAT_TOL {
            return Err(VolumeEnergyReservoirsError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        return Ok(vec![start]);
    }
    if step <= 0.0 {
        return Err(VolumeEnergyReservoirsError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    let mut out = Vec::new();
    let mut value = start;
    while value <= end + FLOAT_TOL {
        out.push(value.min(end));
        value += step;
    }
    if (out.last().copied().unwrap_or(start) - end).abs() > 1e-9 {
        return Err(VolumeEnergyReservoirsError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

fn expand_grid_volume_energy_reservoirs(
    sweep: &VolumeEnergyReservoirsBatchRange,
) -> Result<Vec<VolumeEnergyReservoirsParams>, VolumeEnergyReservoirsError> {
    let lengths = expand_axis_usize(sweep.length)?;
    let sensitivities = expand_axis_f64(
        sweep.sensitivity.0,
        sweep.sensitivity.1,
        sweep.sensitivity.2,
    )?;
    let mut combos = Vec::with_capacity(lengths.len().saturating_mul(sensitivities.len()));
    for length in lengths {
        for &sensitivity in &sensitivities {
            let params = VolumeEnergyReservoirsParams {
                length: Some(length),
                sensitivity: Some(sensitivity),
            };
            let _ = resolve_params(&params, None)?;
            combos.push(params);
        }
    }
    Ok(combos)
}

#[inline]
pub fn volume_energy_reservoirs_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &VolumeEnergyReservoirsBatchRange,
    kernel: Kernel,
) -> Result<VolumeEnergyReservoirsBatchOutput, VolumeEnergyReservoirsError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(VolumeEnergyReservoirsError::InvalidKernelForBatch(other)),
    };
    volume_energy_reservoirs_batch_par_slices(
        high,
        low,
        close,
        volume,
        sweep,
        batch_kernel.to_non_batch(),
    )
}

#[inline]
pub fn volume_energy_reservoirs_batch_slices(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &VolumeEnergyReservoirsBatchRange,
    kernel: Kernel,
) -> Result<VolumeEnergyReservoirsBatchOutput, VolumeEnergyReservoirsError> {
    volume_energy_reservoirs_batch_inner(high, low, close, volume, sweep, kernel, false)
}

#[inline]
pub fn volume_energy_reservoirs_batch_par_slices(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &VolumeEnergyReservoirsBatchRange,
    kernel: Kernel,
) -> Result<VolumeEnergyReservoirsBatchOutput, VolumeEnergyReservoirsError> {
    volume_energy_reservoirs_batch_inner(high, low, close, volume, sweep, kernel, true)
}

#[allow(clippy::too_many_lines)]
pub fn volume_energy_reservoirs_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &VolumeEnergyReservoirsBatchRange,
    _kernel: Kernel,
    parallel: bool,
) -> Result<VolumeEnergyReservoirsBatchOutput, VolumeEnergyReservoirsError> {
    if high.is_empty() || low.is_empty() || close.is_empty() || volume.is_empty() {
        return Err(VolumeEnergyReservoirsError::EmptyInputData);
    }
    if high.len() != low.len() || high.len() != close.len() || high.len() != volume.len() {
        return Err(VolumeEnergyReservoirsError::MismatchedInputLengths {
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
            volume_len: volume.len(),
        });
    }
    if first_valid_ohlcv(high, low, close, volume) >= high.len() {
        return Err(VolumeEnergyReservoirsError::AllValuesNaN);
    }

    let combos = expand_grid_volume_energy_reservoirs(sweep)?;
    let resolved = combos
        .iter()
        .map(|params| resolve_params(params, Some(high.len())))
        .collect::<Result<Vec<_>, _>>()?;
    let rows = combos.len();
    let cols = high.len();
    let total =
        rows.checked_mul(cols)
            .ok_or(VolumeEnergyReservoirsError::OutputLengthMismatch {
                expected: usize::MAX,
            })?;
    let zero_prefixes = vec![0usize; rows];

    let mut momentum_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut momentum_mu, cols, &zero_prefixes);
    let mut momentum_guard = ManuallyDrop::new(momentum_mu);
    let momentum_out =
        unsafe { std::slice::from_raw_parts_mut(momentum_guard.as_mut_ptr() as *mut f64, total) };

    let mut reservoir_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut reservoir_mu, cols, &zero_prefixes);
    let mut reservoir_guard = ManuallyDrop::new(reservoir_mu);
    let reservoir_out =
        unsafe { std::slice::from_raw_parts_mut(reservoir_guard.as_mut_ptr() as *mut f64, total) };

    let mut squeeze_active_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut squeeze_active_mu, cols, &zero_prefixes);
    let mut squeeze_active_guard = ManuallyDrop::new(squeeze_active_mu);
    let squeeze_active_out = unsafe {
        std::slice::from_raw_parts_mut(squeeze_active_guard.as_mut_ptr() as *mut f64, total)
    };

    let mut squeeze_start_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut squeeze_start_mu, cols, &zero_prefixes);
    let mut squeeze_start_guard = ManuallyDrop::new(squeeze_start_mu);
    let squeeze_start_out = unsafe {
        std::slice::from_raw_parts_mut(squeeze_start_guard.as_mut_ptr() as *mut f64, total)
    };

    let mut range_high_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut range_high_mu, cols, &zero_prefixes);
    let mut range_high_guard = ManuallyDrop::new(range_high_mu);
    let range_high_out =
        unsafe { std::slice::from_raw_parts_mut(range_high_guard.as_mut_ptr() as *mut f64, total) };

    let mut range_low_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut range_low_mu, cols, &zero_prefixes);
    let mut range_low_guard = ManuallyDrop::new(range_low_mu);
    let range_low_out =
        unsafe { std::slice::from_raw_parts_mut(range_low_guard.as_mut_ptr() as *mut f64, total) };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let momentum_ptr = momentum_out.as_mut_ptr() as usize;
            let reservoir_ptr = reservoir_out.as_mut_ptr() as usize;
            let squeeze_active_ptr = squeeze_active_out.as_mut_ptr() as usize;
            let squeeze_start_ptr = squeeze_start_out.as_mut_ptr() as usize;
            let range_high_ptr = range_high_out.as_mut_ptr() as usize;
            let range_low_ptr = range_low_out.as_mut_ptr() as usize;

            resolved
                .par_iter()
                .enumerate()
                .for_each(|(row, params)| unsafe {
                    let start = row * cols;
                    volume_energy_reservoirs_row_from_slices(
                        high,
                        low,
                        close,
                        volume,
                        *params,
                        std::slice::from_raw_parts_mut((momentum_ptr as *mut f64).add(start), cols),
                        std::slice::from_raw_parts_mut(
                            (reservoir_ptr as *mut f64).add(start),
                            cols,
                        ),
                        std::slice::from_raw_parts_mut(
                            (squeeze_active_ptr as *mut f64).add(start),
                            cols,
                        ),
                        std::slice::from_raw_parts_mut(
                            (squeeze_start_ptr as *mut f64).add(start),
                            cols,
                        ),
                        std::slice::from_raw_parts_mut(
                            (range_high_ptr as *mut f64).add(start),
                            cols,
                        ),
                        std::slice::from_raw_parts_mut(
                            (range_low_ptr as *mut f64).add(start),
                            cols,
                        ),
                    );
                });
        }

        #[cfg(target_arch = "wasm32")]
        for (row, params) in resolved.iter().enumerate() {
            let start = row * cols;
            let end = start + cols;
            volume_energy_reservoirs_row_from_slices(
                high,
                low,
                close,
                volume,
                *params,
                &mut momentum_out[start..end],
                &mut reservoir_out[start..end],
                &mut squeeze_active_out[start..end],
                &mut squeeze_start_out[start..end],
                &mut range_high_out[start..end],
                &mut range_low_out[start..end],
            );
        }
    } else {
        for (row, params) in resolved.iter().enumerate() {
            let start = row * cols;
            let end = start + cols;
            volume_energy_reservoirs_row_from_slices(
                high,
                low,
                close,
                volume,
                *params,
                &mut momentum_out[start..end],
                &mut reservoir_out[start..end],
                &mut squeeze_active_out[start..end],
                &mut squeeze_start_out[start..end],
                &mut range_high_out[start..end],
                &mut range_low_out[start..end],
            );
        }
    }

    let momentum = unsafe {
        Vec::from_raw_parts(
            momentum_guard.as_mut_ptr() as *mut f64,
            momentum_guard.len(),
            momentum_guard.capacity(),
        )
    };
    let reservoir = unsafe {
        Vec::from_raw_parts(
            reservoir_guard.as_mut_ptr() as *mut f64,
            reservoir_guard.len(),
            reservoir_guard.capacity(),
        )
    };
    let squeeze_active = unsafe {
        Vec::from_raw_parts(
            squeeze_active_guard.as_mut_ptr() as *mut f64,
            squeeze_active_guard.len(),
            squeeze_active_guard.capacity(),
        )
    };
    let squeeze_start = unsafe {
        Vec::from_raw_parts(
            squeeze_start_guard.as_mut_ptr() as *mut f64,
            squeeze_start_guard.len(),
            squeeze_start_guard.capacity(),
        )
    };
    let range_high = unsafe {
        Vec::from_raw_parts(
            range_high_guard.as_mut_ptr() as *mut f64,
            range_high_guard.len(),
            range_high_guard.capacity(),
        )
    };
    let range_low = unsafe {
        Vec::from_raw_parts(
            range_low_guard.as_mut_ptr() as *mut f64,
            range_low_guard.len(),
            range_low_guard.capacity(),
        )
    };
    core::mem::forget(momentum_guard);
    core::mem::forget(reservoir_guard);
    core::mem::forget(squeeze_active_guard);
    core::mem::forget(squeeze_start_guard);
    core::mem::forget(range_high_guard);
    core::mem::forget(range_low_guard);

    Ok(VolumeEnergyReservoirsBatchOutput {
        momentum,
        reservoir,
        squeeze_active,
        squeeze_start,
        range_high,
        range_low,
        combos,
        rows,
        cols,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn volume_energy_reservoirs_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &VolumeEnergyReservoirsBatchRange,
    kernel: Kernel,
    parallel: bool,
    momentum: &mut [f64],
    reservoir: &mut [f64],
    squeeze_active: &mut [f64],
    squeeze_start: &mut [f64],
    range_high: &mut [f64],
    range_low: &mut [f64],
) -> Result<Vec<VolumeEnergyReservoirsParams>, VolumeEnergyReservoirsError> {
    let out =
        volume_energy_reservoirs_batch_inner(high, low, close, volume, sweep, kernel, parallel)?;
    let total = out.rows * out.cols;
    if momentum.len() != total
        || reservoir.len() != total
        || squeeze_active.len() != total
        || squeeze_start.len() != total
        || range_high.len() != total
        || range_low.len() != total
    {
        return Err(VolumeEnergyReservoirsError::OutputLengthMismatch { expected: total });
    }
    momentum.copy_from_slice(&out.momentum);
    reservoir.copy_from_slice(&out.reservoir);
    squeeze_active.copy_from_slice(&out.squeeze_active);
    squeeze_start.copy_from_slice(&out.squeeze_start);
    range_high.copy_from_slice(&out.range_high);
    range_low.copy_from_slice(&out.range_low);
    Ok(out.combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "volume_energy_reservoirs")]
#[pyo3(signature = (
    high,
    low,
    close,
    volume,
    length=DEFAULT_LENGTH,
    sensitivity=DEFAULT_SENSITIVITY,
    kernel=None
))]
pub fn volume_energy_reservoirs_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    length: usize,
    sensitivity: f64,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let volume = volume.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = VolumeEnergyReservoirsInput::from_slices(
        high,
        low,
        close,
        volume,
        VolumeEnergyReservoirsParams {
            length: Some(length),
            sensitivity: Some(sensitivity),
        },
    );
    let out = py
        .allow_threads(|| volume_energy_reservoirs_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        out.momentum.into_pyarray(py),
        out.reservoir.into_pyarray(py),
        out.squeeze_active.into_pyarray(py),
        out.squeeze_start.into_pyarray(py),
        out.range_high.into_pyarray(py),
        out.range_low.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyclass(name = "VolumeEnergyReservoirsStream")]
pub struct VolumeEnergyReservoirsStreamPy {
    stream: VolumeEnergyReservoirsStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl VolumeEnergyReservoirsStreamPy {
    #[new]
    #[pyo3(signature = (length=DEFAULT_LENGTH, sensitivity=DEFAULT_SENSITIVITY))]
    fn new(length: usize, sensitivity: f64) -> PyResult<Self> {
        let stream = VolumeEnergyReservoirsStream::try_new(VolumeEnergyReservoirsParams {
            length: Some(length),
            sensitivity: Some(sensitivity),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(
        &mut self,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Option<(f64, f64, f64, f64, f64, f64)> {
        self.stream.update(high, low, close, volume).map(|point| {
            (
                point.momentum,
                point.reservoir,
                point.squeeze_active,
                point.squeeze_start,
                point.range_high,
                point.range_low,
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
#[pyfunction(name = "volume_energy_reservoirs_batch")]
#[pyo3(signature = (
    high,
    low,
    close,
    volume,
    length_range=(DEFAULT_LENGTH, DEFAULT_LENGTH, 0),
    sensitivity_range=(DEFAULT_SENSITIVITY, DEFAULT_SENSITIVITY, 0.0),
    kernel=None
))]
pub fn volume_energy_reservoirs_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    sensitivity_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let volume = volume.as_slice()?;
    let kernel = validate_kernel(kernel, true)?;
    let sweep = VolumeEnergyReservoirsBatchRange {
        length: length_range,
        sensitivity: sensitivity_range,
    };
    let combos = expand_grid_volume_energy_reservoirs(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let momentum_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let reservoir_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let squeeze_active_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let squeeze_start_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let range_high_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let range_low_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };

    let momentum_slice = unsafe { momentum_arr.as_slice_mut()? };
    let reservoir_slice = unsafe { reservoir_arr.as_slice_mut()? };
    let squeeze_active_slice = unsafe { squeeze_active_arr.as_slice_mut()? };
    let squeeze_start_slice = unsafe { squeeze_start_arr.as_slice_mut()? };
    let range_high_slice = unsafe { range_high_arr.as_slice_mut()? };
    let range_low_slice = unsafe { range_low_arr.as_slice_mut()? };

    let combos = py
        .allow_threads(|| {
            let batch_kernel = match kernel {
                Kernel::Auto => detect_best_batch_kernel(),
                other => other,
            };
            volume_energy_reservoirs_batch_inner_into(
                high,
                low,
                close,
                volume,
                &sweep,
                batch_kernel.to_non_batch(),
                true,
                momentum_slice,
                reservoir_slice,
                squeeze_active_slice,
                squeeze_start_slice,
                range_high_slice,
                range_low_slice,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("momentum", momentum_arr.reshape((rows, cols))?)?;
    dict.set_item("reservoir", reservoir_arr.reshape((rows, cols))?)?;
    dict.set_item("squeeze_active", squeeze_active_arr.reshape((rows, cols))?)?;
    dict.set_item("squeeze_start", squeeze_start_arr.reshape((rows, cols))?)?;
    dict.set_item("range_high", range_high_arr.reshape((rows, cols))?)?;
    dict.set_item("range_low", range_low_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "lengths",
        combos
            .iter()
            .map(|combo| combo.length.unwrap_or(DEFAULT_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "sensitivities",
        combos
            .iter()
            .map(|combo| combo.sensitivity.unwrap_or(DEFAULT_SENSITIVITY))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_volume_energy_reservoirs_module(
    module: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(volume_energy_reservoirs_py, module)?)?;
    module.add_function(wrap_pyfunction!(volume_energy_reservoirs_batch_py, module)?)?;
    module.add_class::<VolumeEnergyReservoirsStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct VolumeEnergyReservoirsJsOutput {
    pub momentum: Vec<f64>,
    pub reservoir: Vec<f64>,
    pub squeeze_active: Vec<f64>,
    pub squeeze_start: Vec<f64>,
    pub range_high: Vec<f64>,
    pub range_low: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "volume_energy_reservoirs_js")]
pub fn volume_energy_reservoirs_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    length: usize,
    sensitivity: f64,
) -> Result<JsValue, JsValue> {
    let input = VolumeEnergyReservoirsInput::from_slices(
        high,
        low,
        close,
        volume,
        VolumeEnergyReservoirsParams {
            length: Some(length),
            sensitivity: Some(sensitivity),
        },
    );
    let out = volume_energy_reservoirs(&input).map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&VolumeEnergyReservoirsJsOutput {
        momentum: out.momentum,
        reservoir: out.reservoir,
        squeeze_active: out.squeeze_active,
        squeeze_start: out.squeeze_start,
        range_high: out.range_high,
        range_low: out.range_low,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn volume_energy_reservoirs_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn volume_energy_reservoirs_free(ptr: *mut f64, len: usize) {
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
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn volume_energy_reservoirs_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    volume_ptr: *const f64,
    momentum_ptr: *mut f64,
    reservoir_ptr: *mut f64,
    squeeze_active_ptr: *mut f64,
    squeeze_start_ptr: *mut f64,
    range_high_ptr: *mut f64,
    range_low_ptr: *mut f64,
    len: usize,
    length: usize,
    sensitivity: f64,
) -> Result<(), JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || volume_ptr.is_null()
        || momentum_ptr.is_null()
        || reservoir_ptr.is_null()
        || squeeze_active_ptr.is_null()
        || squeeze_start_ptr.is_null()
        || range_high_ptr.is_null()
        || range_low_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);
        let input = VolumeEnergyReservoirsInput::from_slices(
            high,
            low,
            close,
            volume,
            VolumeEnergyReservoirsParams {
                length: Some(length),
                sensitivity: Some(sensitivity),
            },
        );

        let output_ptrs = [
            momentum_ptr as usize,
            reservoir_ptr as usize,
            squeeze_active_ptr as usize,
            squeeze_start_ptr as usize,
            range_high_ptr as usize,
            range_low_ptr as usize,
        ];
        let need_temp = output_ptrs.iter().any(|&ptr| {
            ptr == high_ptr as usize
                || ptr == low_ptr as usize
                || ptr == close_ptr as usize
                || ptr == volume_ptr as usize
        }) || has_duplicate_ptrs(&output_ptrs);

        if need_temp {
            let mut momentum = vec![0.0; len];
            let mut reservoir = vec![0.0; len];
            let mut squeeze_active = vec![0.0; len];
            let mut squeeze_start = vec![0.0; len];
            let mut range_high = vec![0.0; len];
            let mut range_low = vec![0.0; len];
            volume_energy_reservoirs_into_slices(
                &mut momentum,
                &mut reservoir,
                &mut squeeze_active,
                &mut squeeze_start,
                &mut range_high,
                &mut range_low,
                &input,
                Kernel::Auto,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
            std::slice::from_raw_parts_mut(momentum_ptr, len).copy_from_slice(&momentum);
            std::slice::from_raw_parts_mut(reservoir_ptr, len).copy_from_slice(&reservoir);
            std::slice::from_raw_parts_mut(squeeze_active_ptr, len)
                .copy_from_slice(&squeeze_active);
            std::slice::from_raw_parts_mut(squeeze_start_ptr, len).copy_from_slice(&squeeze_start);
            std::slice::from_raw_parts_mut(range_high_ptr, len).copy_from_slice(&range_high);
            std::slice::from_raw_parts_mut(range_low_ptr, len).copy_from_slice(&range_low);
        } else {
            volume_energy_reservoirs_into_slices(
                std::slice::from_raw_parts_mut(momentum_ptr, len),
                std::slice::from_raw_parts_mut(reservoir_ptr, len),
                std::slice::from_raw_parts_mut(squeeze_active_ptr, len),
                std::slice::from_raw_parts_mut(squeeze_start_ptr, len),
                std::slice::from_raw_parts_mut(range_high_ptr, len),
                std::slice::from_raw_parts_mut(range_low_ptr, len),
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
pub struct VolumeEnergyReservoirsBatchJsConfig {
    pub length_range: Option<(usize, usize, usize)>,
    pub sensitivity_range: Option<(f64, f64, f64)>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct VolumeEnergyReservoirsBatchJsOutput {
    pub momentum: Vec<f64>,
    pub reservoir: Vec<f64>,
    pub squeeze_active: Vec<f64>,
    pub squeeze_start: Vec<f64>,
    pub range_high: Vec<f64>,
    pub range_low: Vec<f64>,
    pub combos: Vec<VolumeEnergyReservoirsParams>,
    pub lengths: Vec<usize>,
    pub sensitivities: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "volume_energy_reservoirs_batch_js")]
pub fn volume_energy_reservoirs_batch_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: VolumeEnergyReservoirsBatchJsConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let sweep = VolumeEnergyReservoirsBatchRange {
        length: config
            .length_range
            .unwrap_or((DEFAULT_LENGTH, DEFAULT_LENGTH, 0)),
        sensitivity: config.sensitivity_range.unwrap_or((
            DEFAULT_SENSITIVITY,
            DEFAULT_SENSITIVITY,
            0.0,
        )),
    };
    let out =
        volume_energy_reservoirs_batch_with_kernel(high, low, close, volume, &sweep, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&VolumeEnergyReservoirsBatchJsOutput {
        lengths: out
            .combos
            .iter()
            .map(|combo| combo.length.unwrap_or(DEFAULT_LENGTH))
            .collect(),
        sensitivities: out
            .combos
            .iter()
            .map(|combo| combo.sensitivity.unwrap_or(DEFAULT_SENSITIVITY))
            .collect(),
        momentum: out.momentum,
        reservoir: out.reservoir,
        squeeze_active: out.squeeze_active,
        squeeze_start: out.squeeze_start,
        range_high: out.range_high,
        range_low: out.range_low,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn volume_energy_reservoirs_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    volume_ptr: *const f64,
    momentum_ptr: *mut f64,
    reservoir_ptr: *mut f64,
    squeeze_active_ptr: *mut f64,
    squeeze_start_ptr: *mut f64,
    range_high_ptr: *mut f64,
    range_low_ptr: *mut f64,
    len: usize,
    length_start: usize,
    length_end: usize,
    length_step: usize,
    sensitivity_start: f64,
    sensitivity_end: f64,
    sensitivity_step: f64,
) -> Result<usize, JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || volume_ptr.is_null()
        || momentum_ptr.is_null()
        || reservoir_ptr.is_null()
        || squeeze_active_ptr.is_null()
        || squeeze_start_ptr.is_null()
        || range_high_ptr.is_null()
        || range_low_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    let sweep = VolumeEnergyReservoirsBatchRange {
        length: (length_start, length_end, length_step),
        sensitivity: (sensitivity_start, sensitivity_end, sensitivity_step),
    };

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);
        let combos = expand_grid_volume_energy_reservoirs(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let total = rows
            .checked_mul(len)
            .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;

        let output_ptrs = [
            momentum_ptr as usize,
            reservoir_ptr as usize,
            squeeze_active_ptr as usize,
            squeeze_start_ptr as usize,
            range_high_ptr as usize,
            range_low_ptr as usize,
        ];
        let need_temp = output_ptrs.iter().any(|&ptr| {
            ptr == high_ptr as usize
                || ptr == low_ptr as usize
                || ptr == close_ptr as usize
                || ptr == volume_ptr as usize
        }) || has_duplicate_ptrs(&output_ptrs);

        if need_temp {
            let mut momentum = vec![0.0; total];
            let mut reservoir = vec![0.0; total];
            let mut squeeze_active = vec![0.0; total];
            let mut squeeze_start = vec![0.0; total];
            let mut range_high = vec![0.0; total];
            let mut range_low = vec![0.0; total];
            let rows = volume_energy_reservoirs_batch_inner_into(
                high,
                low,
                close,
                volume,
                &sweep,
                Kernel::Auto,
                false,
                &mut momentum,
                &mut reservoir,
                &mut squeeze_active,
                &mut squeeze_start,
                &mut range_high,
                &mut range_low,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?
            .len();
            std::slice::from_raw_parts_mut(momentum_ptr, total).copy_from_slice(&momentum);
            std::slice::from_raw_parts_mut(reservoir_ptr, total).copy_from_slice(&reservoir);
            std::slice::from_raw_parts_mut(squeeze_active_ptr, total)
                .copy_from_slice(&squeeze_active);
            std::slice::from_raw_parts_mut(squeeze_start_ptr, total)
                .copy_from_slice(&squeeze_start);
            std::slice::from_raw_parts_mut(range_high_ptr, total).copy_from_slice(&range_high);
            std::slice::from_raw_parts_mut(range_low_ptr, total).copy_from_slice(&range_low);
            Ok(rows)
        } else {
            let rows = volume_energy_reservoirs_batch_inner_into(
                high,
                low,
                close,
                volume,
                &sweep,
                Kernel::Auto,
                false,
                std::slice::from_raw_parts_mut(momentum_ptr, total),
                std::slice::from_raw_parts_mut(reservoir_ptr, total),
                std::slice::from_raw_parts_mut(squeeze_active_ptr, total),
                std::slice::from_raw_parts_mut(squeeze_start_ptr, total),
                std::slice::from_raw_parts_mut(range_high_ptr, total),
                std::slice::from_raw_parts_mut(range_low_ptr, total),
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?
            .len();
            Ok(rows)
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn volume_energy_reservoirs_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    length: usize,
    sensitivity: f64,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = volume_energy_reservoirs_js(high, low, close, volume, length, sensitivity)?;
    crate::write_wasm_object_f64_outputs("volume_energy_reservoirs_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn volume_energy_reservoirs_batch_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = volume_energy_reservoirs_batch_js(high, low, close, volume, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "volume_energy_reservoirs_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    fn sample_candles(length: usize) -> Candles {
        let close = (0..length)
            .map(|i| {
                let x = i as f64;
                100.0 + x * 0.04 + (x * 0.13).sin() * 2.4 + (x * 0.03).cos() * 0.8
            })
            .collect::<Vec<_>>();
        let open = close.iter().map(|value| value - 0.2).collect::<Vec<_>>();
        let high = close
            .iter()
            .enumerate()
            .map(|(i, value)| value + 0.7 + (i as f64 * 0.05).cos().abs() * 0.3)
            .collect::<Vec<_>>();
        let low = close
            .iter()
            .enumerate()
            .map(|(i, value)| value - 0.8 - (i as f64 * 0.07).sin().abs() * 0.25)
            .collect::<Vec<_>>();
        let volume = (0..length)
            .map(|i| {
                let x = i as f64;
                1_000.0 + x * 3.0 + (x * 0.11).sin() * 220.0 + (x * 0.021).cos() * 50.0
            })
            .collect::<Vec<_>>();
        Candles::new((0..length as i64).collect(), open, high, low, close, volume)
    }

    fn assert_series_eq(left: &[f64], right: &[f64], tol: f64) {
        assert_eq!(left.len(), right.len());
        for (&lhs, &rhs) in left.iter().zip(right.iter()) {
            assert!(
                (lhs.is_nan() && rhs.is_nan()) || (lhs - rhs).abs() <= tol,
                "series mismatch: left={lhs:?}, right={rhs:?}"
            );
        }
    }

    #[test]
    fn volume_energy_reservoirs_output_contract() -> Result<(), Box<dyn Error>> {
        let candles = sample_candles(320);
        let out =
            volume_energy_reservoirs(&VolumeEnergyReservoirsInput::with_default_candles(&candles))?;
        assert_eq!(out.momentum.len(), candles.close.len());
        assert_eq!(out.reservoir.len(), candles.close.len());
        assert_eq!(out.squeeze_active.len(), candles.close.len());
        assert_eq!(out.squeeze_start.len(), candles.close.len());
        assert_eq!(out.range_high.len(), candles.close.len());
        assert_eq!(out.range_low.len(), candles.close.len());
        for &value in &out.reservoir {
            if value.is_finite() {
                assert!((0.0..=10.0).contains(&value));
            }
        }
        Ok(())
    }

    #[test]
    fn volume_energy_reservoirs_rejects_invalid_params() {
        let candles = sample_candles(64);
        let bad_length = volume_energy_reservoirs(&VolumeEnergyReservoirsInput::from_candles(
            &candles,
            VolumeEnergyReservoirsParams {
                length: Some(4),
                sensitivity: Some(DEFAULT_SENSITIVITY),
            },
        ));
        assert!(matches!(
            bad_length,
            Err(VolumeEnergyReservoirsError::InvalidLength { .. })
        ));

        let bad_sensitivity = volume_energy_reservoirs(&VolumeEnergyReservoirsInput::from_candles(
            &candles,
            VolumeEnergyReservoirsParams {
                length: Some(DEFAULT_LENGTH),
                sensitivity: Some(f64::NAN),
            },
        ));
        assert!(matches!(
            bad_sensitivity,
            Err(VolumeEnergyReservoirsError::InvalidSensitivity { .. })
        ));
    }

    #[test]
    fn volume_energy_reservoirs_builder_matches_direct() -> Result<(), Box<dyn Error>> {
        let candles = sample_candles(240);
        let direct = volume_energy_reservoirs(&VolumeEnergyReservoirsInput::from_candles(
            &candles,
            VolumeEnergyReservoirsParams {
                length: Some(24),
                sensitivity: Some(1.8),
            },
        ))?;
        let built = VolumeEnergyReservoirsBuilder::new()
            .length(24)
            .sensitivity(1.8)
            .apply(&candles)?;
        assert_series_eq(&direct.momentum, &built.momentum, 1e-12);
        assert_series_eq(&direct.reservoir, &built.reservoir, 1e-12);
        assert_series_eq(&direct.squeeze_active, &built.squeeze_active, 1e-12);
        assert_series_eq(&direct.squeeze_start, &built.squeeze_start, 1e-12);
        assert_series_eq(&direct.range_high, &built.range_high, 1e-12);
        assert_series_eq(&direct.range_low, &built.range_low, 1e-12);
        Ok(())
    }

    #[test]
    fn volume_energy_reservoirs_stream_matches_batch_with_reset() -> Result<(), Box<dyn Error>> {
        let candles = sample_candles(220);
        let mut high = candles.high.clone();
        let mut low = candles.low.clone();
        let mut close = candles.close.clone();
        let mut volume = candles.volume.clone();
        high[110] = f64::NAN;
        low[110] = f64::NAN;
        close[110] = f64::NAN;
        volume[110] = f64::NAN;

        let batch = volume_energy_reservoirs(&VolumeEnergyReservoirsInput::from_slices(
            &high,
            &low,
            &close,
            &volume,
            VolumeEnergyReservoirsParams {
                length: Some(18),
                sensitivity: Some(1.7),
            },
        ))?;

        let mut stream = VolumeEnergyReservoirsStream::try_new(VolumeEnergyReservoirsParams {
            length: Some(18),
            sensitivity: Some(1.7),
        })?;
        let mut streamed = VolumeEnergyReservoirsOutput {
            momentum: Vec::with_capacity(high.len()),
            reservoir: Vec::with_capacity(high.len()),
            squeeze_active: Vec::with_capacity(high.len()),
            squeeze_start: Vec::with_capacity(high.len()),
            range_high: Vec::with_capacity(high.len()),
            range_low: Vec::with_capacity(high.len()),
        };
        for i in 0..high.len() {
            if let Some(point) = stream.update(high[i], low[i], close[i], volume[i]) {
                streamed.momentum.push(point.momentum);
                streamed.reservoir.push(point.reservoir);
                streamed.squeeze_active.push(point.squeeze_active);
                streamed.squeeze_start.push(point.squeeze_start);
                streamed.range_high.push(point.range_high);
                streamed.range_low.push(point.range_low);
            } else {
                streamed.momentum.push(f64::NAN);
                streamed.reservoir.push(f64::NAN);
                streamed.squeeze_active.push(f64::NAN);
                streamed.squeeze_start.push(f64::NAN);
                streamed.range_high.push(f64::NAN);
                streamed.range_low.push(f64::NAN);
            }
        }

        assert_eq!(stream.get_warmup_period(), 0);
        assert_series_eq(&batch.momentum, &streamed.momentum, 1e-12);
        assert_series_eq(&batch.reservoir, &streamed.reservoir, 1e-12);
        assert_series_eq(&batch.squeeze_active, &streamed.squeeze_active, 1e-12);
        assert_series_eq(&batch.squeeze_start, &streamed.squeeze_start, 1e-12);
        assert_series_eq(&batch.range_high, &streamed.range_high, 1e-12);
        assert_series_eq(&batch.range_low, &streamed.range_low, 1e-12);
        Ok(())
    }

    #[test]
    fn volume_energy_reservoirs_into_matches_api() -> Result<(), Box<dyn Error>> {
        let candles = sample_candles(160);
        let input = VolumeEnergyReservoirsInput::with_default_candles(&candles);
        let baseline = volume_energy_reservoirs(&input)?;

        let mut momentum = vec![f64::NAN; candles.close.len()];
        let mut reservoir = vec![f64::NAN; candles.close.len()];
        let mut squeeze_active = vec![f64::NAN; candles.close.len()];
        let mut squeeze_start = vec![f64::NAN; candles.close.len()];
        let mut range_high = vec![f64::NAN; candles.close.len()];
        let mut range_low = vec![f64::NAN; candles.close.len()];
        volume_energy_reservoirs_into(
            &mut momentum,
            &mut reservoir,
            &mut squeeze_active,
            &mut squeeze_start,
            &mut range_high,
            &mut range_low,
            &input,
        )?;

        assert_series_eq(&baseline.momentum, &momentum, 1e-12);
        assert_series_eq(&baseline.reservoir, &reservoir, 1e-12);
        assert_series_eq(&baseline.squeeze_active, &squeeze_active, 1e-12);
        assert_series_eq(&baseline.squeeze_start, &squeeze_start, 1e-12);
        assert_series_eq(&baseline.range_high, &range_high, 1e-12);
        assert_series_eq(&baseline.range_low, &range_low, 1e-12);
        Ok(())
    }

    #[test]
    fn volume_energy_reservoirs_batch_single_param_matches_single() -> Result<(), Box<dyn Error>> {
        let candles = sample_candles(144);
        let batch = volume_energy_reservoirs_batch_with_kernel(
            &candles.high,
            &candles.low,
            &candles.close,
            &candles.volume,
            &VolumeEnergyReservoirsBatchRange {
                length: (18, 18, 0),
                sensitivity: (1.7, 1.7, 0.0),
            },
            Kernel::Auto,
        )?;
        let single = volume_energy_reservoirs(&VolumeEnergyReservoirsInput::from_candles(
            &candles,
            VolumeEnergyReservoirsParams {
                length: Some(18),
                sensitivity: Some(1.7),
            },
        ))?;

        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, candles.close.len());
        let row = batch.row_slices(0).unwrap();
        assert_series_eq(row.0, &single.momentum, 1e-12);
        assert_series_eq(row.1, &single.reservoir, 1e-12);
        assert_series_eq(row.2, &single.squeeze_active, 1e-12);
        assert_series_eq(row.3, &single.squeeze_start, 1e-12);
        assert_series_eq(row.4, &single.range_high, 1e-12);
        assert_series_eq(row.5, &single.range_low, 1e-12);
        Ok(())
    }

    #[test]
    fn volume_energy_reservoirs_batch_metadata() -> Result<(), Box<dyn Error>> {
        let candles = sample_candles(96);
        let batch = volume_energy_reservoirs_batch_with_kernel(
            &candles.high,
            &candles.low,
            &candles.close,
            &candles.volume,
            &VolumeEnergyReservoirsBatchRange {
                length: (16, 20, 4),
                sensitivity: (1.5, 2.0, 0.5),
            },
            Kernel::Auto,
        )?;
        assert_eq!(batch.rows, 4);
        assert_eq!(batch.cols, candles.close.len());
        assert_eq!(batch.combos[0].length, Some(16));
        assert_eq!(batch.combos[0].sensitivity, Some(1.5));
        assert_eq!(batch.combos[3].length, Some(20));
        assert_eq!(batch.combos[3].sensitivity, Some(2.0));
        Ok(())
    }
}
