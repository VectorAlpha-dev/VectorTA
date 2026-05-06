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
use serde_wasm_bindgen;
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

use crate::indicators::moving_averages::ema::{EmaParams, EmaStream};
use crate::indicators::moving_averages::hma::{HmaParams, HmaStream};
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

const DEFAULT_ALPHA_LENGTH: usize = 33;
const DEFAULT_ALPHA_MULTIPLIER: f64 = 3.3;
const DEFAULT_MFI_LENGTH: usize = 14;
const MFI_HMA_LENGTH: usize = 7;
const MFI_HMA_SQRT: usize = 2;

type TrendFlowTrailRow = (
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
);

#[derive(Debug, Clone)]
pub enum TrendFlowTrailData<'a> {
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
pub struct TrendFlowTrailOutput {
    pub alpha_trail: Vec<f64>,
    pub alpha_trail_bullish: Vec<f64>,
    pub alpha_trail_bearish: Vec<f64>,
    pub alpha_dir: Vec<f64>,
    pub mfi: Vec<f64>,
    pub tp_upper: Vec<f64>,
    pub tp_lower: Vec<f64>,
    pub alpha_trail_bullish_switch: Vec<f64>,
    pub alpha_trail_bearish_switch: Vec<f64>,
    pub mfi_overbought: Vec<f64>,
    pub mfi_oversold: Vec<f64>,
    pub mfi_cross_up_mid: Vec<f64>,
    pub mfi_cross_down_mid: Vec<f64>,
    pub price_cross_alpha_trail_up: Vec<f64>,
    pub price_cross_alpha_trail_down: Vec<f64>,
    pub mfi_above_90: Vec<f64>,
    pub mfi_below_10: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct TrendFlowTrailParams {
    pub alpha_length: Option<usize>,
    pub alpha_multiplier: Option<f64>,
    pub mfi_length: Option<usize>,
}

impl Default for TrendFlowTrailParams {
    fn default() -> Self {
        Self {
            alpha_length: Some(DEFAULT_ALPHA_LENGTH),
            alpha_multiplier: Some(DEFAULT_ALPHA_MULTIPLIER),
            mfi_length: Some(DEFAULT_MFI_LENGTH),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrendFlowTrailInput<'a> {
    pub data: TrendFlowTrailData<'a>,
    pub params: TrendFlowTrailParams,
}

impl<'a> TrendFlowTrailInput<'a> {
    #[inline(always)]
    pub fn from_candles(candles: &'a Candles, params: TrendFlowTrailParams) -> Self {
        Self {
            data: TrendFlowTrailData::Candles { candles },
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
        params: TrendFlowTrailParams,
    ) -> Self {
        Self {
            data: TrendFlowTrailData::Slices {
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
        Self::from_candles(candles, TrendFlowTrailParams::default())
    }

    #[inline(always)]
    pub fn get_alpha_length(&self) -> usize {
        self.params.alpha_length.unwrap_or(DEFAULT_ALPHA_LENGTH)
    }

    #[inline(always)]
    pub fn get_alpha_multiplier(&self) -> f64 {
        self.params
            .alpha_multiplier
            .unwrap_or(DEFAULT_ALPHA_MULTIPLIER)
    }

    #[inline(always)]
    pub fn get_mfi_length(&self) -> usize {
        self.params.mfi_length.unwrap_or(DEFAULT_MFI_LENGTH)
    }

    #[inline(always)]
    fn as_ohlcv(&self) -> (&'a [f64], &'a [f64], &'a [f64], &'a [f64], &'a [f64]) {
        match &self.data {
            TrendFlowTrailData::Candles { candles } => (
                source_type(candles, "open"),
                source_type(candles, "high"),
                source_type(candles, "low"),
                source_type(candles, "close"),
                source_type(candles, "volume"),
            ),
            TrendFlowTrailData::Slices {
                open,
                high,
                low,
                close,
                volume,
            } => (*open, *high, *low, *close, *volume),
        }
    }
}

impl<'a> AsRef<[f64]> for TrendFlowTrailInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        self.as_ohlcv().3
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TrendFlowTrailBuilder {
    alpha_length: Option<usize>,
    alpha_multiplier: Option<f64>,
    mfi_length: Option<usize>,
    kernel: Kernel,
}

impl Default for TrendFlowTrailBuilder {
    fn default() -> Self {
        Self {
            alpha_length: None,
            alpha_multiplier: None,
            mfi_length: None,
            kernel: Kernel::Auto,
        }
    }
}

impl TrendFlowTrailBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn alpha_length(mut self, value: usize) -> Self {
        self.alpha_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn alpha_multiplier(mut self, value: f64) -> Self {
        self.alpha_multiplier = Some(value);
        self
    }

    #[inline(always)]
    pub fn mfi_length(mut self, value: usize) -> Self {
        self.mfi_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline(always)]
    fn params(self) -> TrendFlowTrailParams {
        TrendFlowTrailParams {
            alpha_length: self.alpha_length,
            alpha_multiplier: self.alpha_multiplier,
            mfi_length: self.mfi_length,
        }
    }

    #[inline(always)]
    pub fn apply(self, candles: &Candles) -> Result<TrendFlowTrailOutput, TrendFlowTrailError> {
        let kernel = self.kernel;
        trend_flow_trail_with_kernel(
            &TrendFlowTrailInput::from_candles(candles, self.params()),
            kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Result<TrendFlowTrailOutput, TrendFlowTrailError> {
        let kernel = self.kernel;
        trend_flow_trail_with_kernel(
            &TrendFlowTrailInput::from_slices(open, high, low, close, volume, self.params()),
            kernel,
        )
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<TrendFlowTrailStream, TrendFlowTrailError> {
        TrendFlowTrailStream::try_new(self.params())
    }
}

#[derive(Debug, Error)]
pub enum TrendFlowTrailError {
    #[error("trend_flow_trail: input data slice is empty.")]
    EmptyInputData,
    #[error("trend_flow_trail: all values are NaN.")]
    AllValuesNaN,
    #[error("trend_flow_trail: inconsistent slice lengths: open={open_len}, high={high_len}, low={low_len}, close={close_len}, volume={volume_len}")]
    InconsistentSliceLengths {
        open_len: usize,
        high_len: usize,
        low_len: usize,
        close_len: usize,
        volume_len: usize,
    },
    #[error("trend_flow_trail: invalid alpha_length: {alpha_length}")]
    InvalidAlphaLength { alpha_length: usize },
    #[error("trend_flow_trail: invalid alpha_multiplier: {alpha_multiplier}")]
    InvalidAlphaMultiplier { alpha_multiplier: f64 },
    #[error("trend_flow_trail: invalid mfi_length: {mfi_length}")]
    InvalidMfiLength { mfi_length: usize },
    #[error("trend_flow_trail: not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("trend_flow_trail: output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error(
        "trend_flow_trail: invalid range for {axis}: start = {start}, end = {end}, step = {step}"
    )]
    InvalidRange {
        axis: &'static str,
        start: String,
        end: String,
        step: String,
    },
    #[error("trend_flow_trail: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Clone, Copy, Debug)]
struct PreparedTrendFlowTrail<'a> {
    open: &'a [f64],
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    volume: &'a [f64],
    alpha_length: usize,
    alpha_multiplier: f64,
    mfi_length: usize,
    warmup: usize,
}

#[derive(Clone, Debug)]
struct HmaLikeStream {
    period: usize,
    inner: Option<HmaStream>,
}

impl HmaLikeStream {
    fn try_new_alpha(period: usize) -> Result<Self, TrendFlowTrailError> {
        if period == 0 {
            return Err(TrendFlowTrailError::InvalidAlphaLength {
                alpha_length: period,
            });
        }
        Ok(Self {
            period,
            inner: if period == 1 {
                None
            } else {
                Some(
                    HmaStream::try_new(HmaParams {
                        period: Some(period),
                    })
                    .map_err(|_| TrendFlowTrailError::InvalidAlphaLength {
                        alpha_length: period,
                    })?,
                )
            },
        })
    }

    fn try_new_fixed(period: usize) -> Result<Self, TrendFlowTrailError> {
        Ok(Self {
            period,
            inner: Some(
                HmaStream::try_new(HmaParams {
                    period: Some(period),
                })
                .map_err(|_| TrendFlowTrailError::InvalidMfiLength { mfi_length: period })?,
            ),
        })
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<f64> {
        if self.period == 1 {
            Some(value)
        } else {
            self.inner.as_mut().and_then(|inner| inner.update(value))
        }
    }
}

#[derive(Clone, Debug)]
struct MoneyFlowRawState {
    len: usize,
    pos: Vec<f64>,
    neg: Vec<f64>,
    head: usize,
    count: usize,
    pos_sum: f64,
    neg_sum: f64,
    prev_src: Option<f64>,
}

impl MoneyFlowRawState {
    fn new(len: usize) -> Result<Self, TrendFlowTrailError> {
        if len == 0 {
            return Err(TrendFlowTrailError::InvalidMfiLength { mfi_length: len });
        }
        Ok(Self {
            len,
            pos: vec![0.0; len],
            neg: vec![0.0; len],
            head: 0,
            count: 0,
            pos_sum: 0.0,
            neg_sum: 0.0,
            prev_src: None,
        })
    }

    #[inline(always)]
    fn update(&mut self, src: f64, volume: f64) -> Option<f64> {
        let delta = self.prev_src.map(|prev| src - prev).unwrap_or(0.0);
        self.prev_src = Some(src);
        let pos_flow = if delta > 0.0 { volume * src } else { 0.0 };
        let neg_flow = if delta < 0.0 { volume * src } else { 0.0 };
        if self.count == self.len {
            self.pos_sum -= self.pos[self.head];
            self.neg_sum -= self.neg[self.head];
        } else {
            self.count += 1;
        }
        self.pos[self.head] = pos_flow;
        self.neg[self.head] = neg_flow;
        self.pos_sum += pos_flow;
        self.neg_sum += neg_flow;
        self.head = (self.head + 1) % self.len;
        if self.count < self.len {
            return None;
        }
        let ratio = self.pos_sum / self.neg_sum;
        Some(100.0 - (100.0 / (1.0 + ratio)))
    }
}

#[derive(Clone, Debug)]
struct TrendFlowTrailState {
    alpha_length: usize,
    alpha_multiplier: f64,
    mfi_length: usize,
    basis_stream: HmaLikeStream,
    spread_stream: EmaStream,
    money_flow_raw: MoneyFlowRawState,
    mfi_stream: HmaLikeStream,
    prev_upper: Option<f64>,
    prev_lower: Option<f64>,
    prev_trail: Option<f64>,
    prev_alpha_dir: Option<f64>,
    prev_close: Option<f64>,
    prev_mfi: Option<f64>,
}

impl TrendFlowTrailState {
    fn try_new(
        alpha_length: usize,
        alpha_multiplier: f64,
        mfi_length: usize,
    ) -> Result<Self, TrendFlowTrailError> {
        validate_params(alpha_length, alpha_multiplier, mfi_length, usize::MAX)?;
        Ok(Self {
            alpha_length,
            alpha_multiplier,
            mfi_length,
            basis_stream: HmaLikeStream::try_new_alpha(alpha_length)?,
            spread_stream: EmaStream::try_new(EmaParams {
                period: Some(alpha_length.max(1)),
            })
            .map_err(|_| TrendFlowTrailError::InvalidAlphaLength { alpha_length })?,
            money_flow_raw: MoneyFlowRawState::new(mfi_length)?,
            mfi_stream: HmaLikeStream::try_new_fixed(MFI_HMA_LENGTH)?,
            prev_upper: None,
            prev_lower: None,
            prev_trail: None,
            prev_alpha_dir: None,
            prev_close: None,
            prev_mfi: None,
        })
    }

    #[inline(always)]
    fn reset(&mut self) {
        *self = Self::try_new(self.alpha_length, self.alpha_multiplier, self.mfi_length)
            .expect("trend_flow_trail params already validated");
    }

    fn update(
        &mut self,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Option<TrendFlowTrailRow> {
        if !(open.is_finite()
            && high.is_finite()
            && low.is_finite()
            && close.is_finite()
            && volume.is_finite())
        {
            self.reset();
            return None;
        }

        let basis = self.basis_stream.update(close);
        let spread = self
            .spread_stream
            .update((high - low).abs())
            .map(|x| x * self.alpha_multiplier);
        let hlc3 = (high + low + close) / 3.0;
        let mfi = self
            .money_flow_raw
            .update(hlc3, volume)
            .and_then(|raw| self.mfi_stream.update(raw));

        let prev_close = self.prev_close;
        self.prev_close = Some(close);
        let prev_alpha_dir = self.prev_alpha_dir;
        let prev_trail = self.prev_trail;
        let prev_mfi = self.prev_mfi;

        let (basis, spread, mfi) = match (basis, spread, mfi) {
            (Some(basis), Some(spread), Some(mfi)) if mfi.is_finite() => (basis, spread, mfi),
            _ => return None,
        };

        let mut upper = basis + spread;
        let mut lower = basis - spread;
        let prev_upper = self.prev_upper.unwrap_or(0.0);
        let prev_lower = self.prev_lower.unwrap_or(0.0);
        let prev_close_value = prev_close.unwrap_or(0.0);

        lower = if lower > prev_lower || prev_close_value < prev_lower {
            lower
        } else {
            prev_lower
        };
        upper = if upper < prev_upper || prev_close_value > prev_upper {
            upper
        } else {
            prev_upper
        };

        let alpha_dir = if prev_trail.is_none() {
            1.0
        } else if prev_trail == Some(prev_upper) {
            if close > upper {
                -1.0
            } else {
                1.0
            }
        } else if close < lower {
            1.0
        } else {
            -1.0
        };

        let alpha_trail = if alpha_dir < 0.0 { lower } else { upper };
        self.prev_upper = Some(upper);
        self.prev_lower = Some(lower);
        self.prev_trail = Some(alpha_trail);
        self.prev_alpha_dir = Some(alpha_dir);
        self.prev_mfi = Some(mfi);

        Some((
            alpha_trail,
            if alpha_dir < 0.0 {
                alpha_trail
            } else {
                f64::NAN
            },
            if alpha_dir > 0.0 {
                alpha_trail
            } else {
                f64::NAN
            },
            alpha_dir,
            mfi,
            if crossover(prev_mfi, mfi, 80.0) && alpha_dir == -1.0 {
                1.0
            } else {
                f64::NAN
            },
            if crossunder(prev_mfi, mfi, 20.0) && alpha_dir == 1.0 {
                1.0
            } else {
                f64::NAN
            },
            if crossover(prev_alpha_dir, alpha_dir, 0.0) {
                1.0
            } else {
                f64::NAN
            },
            if crossunder(prev_alpha_dir, alpha_dir, 0.0) {
                1.0
            } else {
                f64::NAN
            },
            if crossover(prev_mfi, mfi, 80.0) {
                1.0
            } else {
                f64::NAN
            },
            if crossunder(prev_mfi, mfi, 20.0) {
                1.0
            } else {
                f64::NAN
            },
            if crossover(prev_mfi, mfi, 50.0) {
                1.0
            } else {
                f64::NAN
            },
            if crossunder(prev_mfi, mfi, 50.0) {
                1.0
            } else {
                f64::NAN
            },
            if cross_pair(prev_close, close, prev_trail, alpha_trail) {
                1.0
            } else {
                f64::NAN
            },
            if crossunder_pair(prev_close, close, prev_trail, alpha_trail) {
                1.0
            } else {
                f64::NAN
            },
            if crossover(prev_mfi, mfi, 90.0) {
                1.0
            } else {
                f64::NAN
            },
            if crossunder(prev_mfi, mfi, 10.0) {
                1.0
            } else {
                f64::NAN
            },
        ))
    }
}

#[derive(Clone, Debug)]
pub struct TrendFlowTrailStream {
    state: TrendFlowTrailState,
}

impl TrendFlowTrailStream {
    #[inline(always)]
    pub fn try_new(params: TrendFlowTrailParams) -> Result<Self, TrendFlowTrailError> {
        Ok(Self {
            state: TrendFlowTrailState::try_new(
                params.alpha_length.unwrap_or(DEFAULT_ALPHA_LENGTH),
                params.alpha_multiplier.unwrap_or(DEFAULT_ALPHA_MULTIPLIER),
                params.mfi_length.unwrap_or(DEFAULT_MFI_LENGTH),
            )?,
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
    ) -> Option<TrendFlowTrailRow> {
        self.state.update(open, high, low, close, volume)
    }
}

#[inline(always)]
fn crossover(prev_left: Option<f64>, left: f64, right: f64) -> bool {
    prev_left
        .map(|p| p.is_finite() && p <= right && left.is_finite() && left > right)
        .unwrap_or(false)
}

#[inline(always)]
fn crossunder(prev_left: Option<f64>, left: f64, right: f64) -> bool {
    prev_left
        .map(|p| p.is_finite() && p >= right && left.is_finite() && left < right)
        .unwrap_or(false)
}

#[inline(always)]
fn cross_pair(prev_left: Option<f64>, left: f64, prev_right: Option<f64>, right: f64) -> bool {
    match (prev_left, prev_right) {
        (Some(pl), Some(pr))
            if pl.is_finite() && pr.is_finite() && left.is_finite() && right.is_finite() =>
        {
            pl <= pr && left > right
        }
        _ => false,
    }
}

#[inline(always)]
fn crossunder_pair(prev_left: Option<f64>, left: f64, prev_right: Option<f64>, right: f64) -> bool {
    match (prev_left, prev_right) {
        (Some(pl), Some(pr))
            if pl.is_finite() && pr.is_finite() && left.is_finite() && right.is_finite() =>
        {
            pl >= pr && left < right
        }
        _ => false,
    }
}

#[inline(always)]
fn alpha_required_bars(alpha_length: usize) -> usize {
    if alpha_length <= 1 {
        1
    } else {
        alpha_length + (alpha_length as f64).sqrt().floor() as usize - 1
    }
}

#[inline(always)]
fn required_valid_bars(alpha_length: usize, mfi_length: usize) -> usize {
    alpha_required_bars(alpha_length).max(mfi_length + MFI_HMA_LENGTH + MFI_HMA_SQRT - 1)
}

fn validate_params(
    alpha_length: usize,
    alpha_multiplier: f64,
    mfi_length: usize,
    data_len: usize,
) -> Result<(), TrendFlowTrailError> {
    if alpha_length == 0 {
        return Err(TrendFlowTrailError::InvalidAlphaLength { alpha_length });
    }
    if !alpha_multiplier.is_finite() || alpha_multiplier < 0.1 {
        return Err(TrendFlowTrailError::InvalidAlphaMultiplier { alpha_multiplier });
    }
    if mfi_length == 0 {
        return Err(TrendFlowTrailError::InvalidMfiLength { mfi_length });
    }
    if data_len != usize::MAX {
        let needed = required_valid_bars(alpha_length, mfi_length);
        if data_len < needed {
            return Err(TrendFlowTrailError::NotEnoughValidData {
                needed,
                valid: data_len,
            });
        }
    }
    Ok(())
}

fn analyze_valid_segments(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
) -> Result<(usize, usize), TrendFlowTrailError> {
    if open.is_empty() {
        return Err(TrendFlowTrailError::EmptyInputData);
    }
    if open.len() != high.len()
        || open.len() != low.len()
        || open.len() != close.len()
        || open.len() != volume.len()
    {
        return Err(TrendFlowTrailError::InconsistentSliceLengths {
            open_len: open.len(),
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
            volume_len: volume.len(),
        });
    }
    let mut valid = 0usize;
    let mut run = 0usize;
    let mut max_run = 0usize;
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
        return Err(TrendFlowTrailError::AllValuesNaN);
    }
    Ok((valid, max_run))
}

fn prepare_input<'a>(
    input: &'a TrendFlowTrailInput<'a>,
    kernel: Kernel,
) -> Result<PreparedTrendFlowTrail<'a>, TrendFlowTrailError> {
    if matches!(kernel, Kernel::Auto) {
        let _ = detect_best_kernel();
    }
    let (open, high, low, close, volume) = input.as_ohlcv();
    let alpha_length = input.get_alpha_length();
    let alpha_multiplier = input.get_alpha_multiplier();
    let mfi_length = input.get_mfi_length();
    validate_params(alpha_length, alpha_multiplier, mfi_length, close.len())?;
    let (_, max_run) = analyze_valid_segments(open, high, low, close, volume)?;
    let needed = required_valid_bars(alpha_length, mfi_length);
    if max_run < needed {
        return Err(TrendFlowTrailError::NotEnoughValidData {
            needed,
            valid: max_run,
        });
    }
    Ok(PreparedTrendFlowTrail {
        open,
        high,
        low,
        close,
        volume,
        alpha_length,
        alpha_multiplier,
        mfi_length,
        warmup: needed - 1,
    })
}

#[allow(clippy::too_many_arguments)]
fn compute_row(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    alpha_length: usize,
    alpha_multiplier: f64,
    mfi_length: usize,
    alpha_trail_out: &mut [f64],
    alpha_trail_bullish_out: &mut [f64],
    alpha_trail_bearish_out: &mut [f64],
    alpha_dir_out: &mut [f64],
    mfi_out: &mut [f64],
    tp_upper_out: &mut [f64],
    tp_lower_out: &mut [f64],
    alpha_trail_bullish_switch_out: &mut [f64],
    alpha_trail_bearish_switch_out: &mut [f64],
    mfi_overbought_out: &mut [f64],
    mfi_oversold_out: &mut [f64],
    mfi_cross_up_mid_out: &mut [f64],
    mfi_cross_down_mid_out: &mut [f64],
    price_cross_alpha_trail_up_out: &mut [f64],
    price_cross_alpha_trail_down_out: &mut [f64],
    mfi_above_90_out: &mut [f64],
    mfi_below_10_out: &mut [f64],
) -> Result<(), TrendFlowTrailError> {
    let expected = close.len();
    for out in [
        &*alpha_trail_out,
        &*alpha_trail_bullish_out,
        &*alpha_trail_bearish_out,
        &*alpha_dir_out,
        &*mfi_out,
        &*tp_upper_out,
        &*tp_lower_out,
        &*alpha_trail_bullish_switch_out,
        &*alpha_trail_bearish_switch_out,
        &*mfi_overbought_out,
        &*mfi_oversold_out,
        &*mfi_cross_up_mid_out,
        &*mfi_cross_down_mid_out,
        &*price_cross_alpha_trail_up_out,
        &*price_cross_alpha_trail_down_out,
        &*mfi_above_90_out,
        &*mfi_below_10_out,
    ] {
        if out.len() != expected {
            return Err(TrendFlowTrailError::OutputLengthMismatch {
                expected,
                got: out.len(),
            });
        }
    }
    let mut state = TrendFlowTrailState::try_new(alpha_length, alpha_multiplier, mfi_length)?;
    for i in 0..expected {
        match state.update(open[i], high[i], low[i], close[i], volume[i]) {
            Some((a, ab, ar, d, m, tu, tl, bs, rs, ob, os, mu, md, pu, pd, a90, b10)) => {
                alpha_trail_out[i] = a;
                alpha_trail_bullish_out[i] = ab;
                alpha_trail_bearish_out[i] = ar;
                alpha_dir_out[i] = d;
                mfi_out[i] = m;
                tp_upper_out[i] = tu;
                tp_lower_out[i] = tl;
                alpha_trail_bullish_switch_out[i] = bs;
                alpha_trail_bearish_switch_out[i] = rs;
                mfi_overbought_out[i] = ob;
                mfi_oversold_out[i] = os;
                mfi_cross_up_mid_out[i] = mu;
                mfi_cross_down_mid_out[i] = md;
                price_cross_alpha_trail_up_out[i] = pu;
                price_cross_alpha_trail_down_out[i] = pd;
                mfi_above_90_out[i] = a90;
                mfi_below_10_out[i] = b10;
            }
            None => {
                alpha_trail_out[i] = f64::NAN;
                alpha_trail_bullish_out[i] = f64::NAN;
                alpha_trail_bearish_out[i] = f64::NAN;
                alpha_dir_out[i] = f64::NAN;
                mfi_out[i] = f64::NAN;
                tp_upper_out[i] = f64::NAN;
                tp_lower_out[i] = f64::NAN;
                alpha_trail_bullish_switch_out[i] = f64::NAN;
                alpha_trail_bearish_switch_out[i] = f64::NAN;
                mfi_overbought_out[i] = f64::NAN;
                mfi_oversold_out[i] = f64::NAN;
                mfi_cross_up_mid_out[i] = f64::NAN;
                mfi_cross_down_mid_out[i] = f64::NAN;
                price_cross_alpha_trail_up_out[i] = f64::NAN;
                price_cross_alpha_trail_down_out[i] = f64::NAN;
                mfi_above_90_out[i] = f64::NAN;
                mfi_below_10_out[i] = f64::NAN;
            }
        }
    }
    Ok(())
}

#[inline]
pub fn trend_flow_trail(
    input: &TrendFlowTrailInput,
) -> Result<TrendFlowTrailOutput, TrendFlowTrailError> {
    trend_flow_trail_with_kernel(input, Kernel::Auto)
}

pub fn trend_flow_trail_with_kernel(
    input: &TrendFlowTrailInput,
    kernel: Kernel,
) -> Result<TrendFlowTrailOutput, TrendFlowTrailError> {
    let prepared = prepare_input(input, kernel)?;
    let len = prepared.close.len();
    let warmup = prepared.warmup;
    let mut alpha_trail = alloc_with_nan_prefix(len, warmup);
    let mut alpha_trail_bullish = alloc_with_nan_prefix(len, warmup);
    let mut alpha_trail_bearish = alloc_with_nan_prefix(len, warmup);
    let mut alpha_dir = alloc_with_nan_prefix(len, warmup);
    let mut mfi = alloc_with_nan_prefix(len, warmup);
    let mut tp_upper = alloc_with_nan_prefix(len, warmup);
    let mut tp_lower = alloc_with_nan_prefix(len, warmup);
    let mut alpha_trail_bullish_switch = alloc_with_nan_prefix(len, warmup);
    let mut alpha_trail_bearish_switch = alloc_with_nan_prefix(len, warmup);
    let mut mfi_overbought = alloc_with_nan_prefix(len, warmup);
    let mut mfi_oversold = alloc_with_nan_prefix(len, warmup);
    let mut mfi_cross_up_mid = alloc_with_nan_prefix(len, warmup);
    let mut mfi_cross_down_mid = alloc_with_nan_prefix(len, warmup);
    let mut price_cross_alpha_trail_up = alloc_with_nan_prefix(len, warmup);
    let mut price_cross_alpha_trail_down = alloc_with_nan_prefix(len, warmup);
    let mut mfi_above_90 = alloc_with_nan_prefix(len, warmup);
    let mut mfi_below_10 = alloc_with_nan_prefix(len, warmup);
    compute_row(
        prepared.open,
        prepared.high,
        prepared.low,
        prepared.close,
        prepared.volume,
        prepared.alpha_length,
        prepared.alpha_multiplier,
        prepared.mfi_length,
        &mut alpha_trail,
        &mut alpha_trail_bullish,
        &mut alpha_trail_bearish,
        &mut alpha_dir,
        &mut mfi,
        &mut tp_upper,
        &mut tp_lower,
        &mut alpha_trail_bullish_switch,
        &mut alpha_trail_bearish_switch,
        &mut mfi_overbought,
        &mut mfi_oversold,
        &mut mfi_cross_up_mid,
        &mut mfi_cross_down_mid,
        &mut price_cross_alpha_trail_up,
        &mut price_cross_alpha_trail_down,
        &mut mfi_above_90,
        &mut mfi_below_10,
    )?;
    Ok(TrendFlowTrailOutput {
        alpha_trail,
        alpha_trail_bullish,
        alpha_trail_bearish,
        alpha_dir,
        mfi,
        tp_upper,
        tp_lower,
        alpha_trail_bullish_switch,
        alpha_trail_bearish_switch,
        mfi_overbought,
        mfi_oversold,
        mfi_cross_up_mid,
        mfi_cross_down_mid,
        price_cross_alpha_trail_up,
        price_cross_alpha_trail_down,
        mfi_above_90,
        mfi_below_10,
    })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[allow(clippy::too_many_arguments)]
pub fn trend_flow_trail_into(
    alpha_trail_out: &mut [f64],
    alpha_trail_bullish_out: &mut [f64],
    alpha_trail_bearish_out: &mut [f64],
    alpha_dir_out: &mut [f64],
    mfi_out: &mut [f64],
    tp_upper_out: &mut [f64],
    tp_lower_out: &mut [f64],
    alpha_trail_bullish_switch_out: &mut [f64],
    alpha_trail_bearish_switch_out: &mut [f64],
    mfi_overbought_out: &mut [f64],
    mfi_oversold_out: &mut [f64],
    mfi_cross_up_mid_out: &mut [f64],
    mfi_cross_down_mid_out: &mut [f64],
    price_cross_alpha_trail_up_out: &mut [f64],
    price_cross_alpha_trail_down_out: &mut [f64],
    mfi_above_90_out: &mut [f64],
    mfi_below_10_out: &mut [f64],
    input: &TrendFlowTrailInput,
) -> Result<(), TrendFlowTrailError> {
    trend_flow_trail_into_slice(
        alpha_trail_out,
        alpha_trail_bullish_out,
        alpha_trail_bearish_out,
        alpha_dir_out,
        mfi_out,
        tp_upper_out,
        tp_lower_out,
        alpha_trail_bullish_switch_out,
        alpha_trail_bearish_switch_out,
        mfi_overbought_out,
        mfi_oversold_out,
        mfi_cross_up_mid_out,
        mfi_cross_down_mid_out,
        price_cross_alpha_trail_up_out,
        price_cross_alpha_trail_down_out,
        mfi_above_90_out,
        mfi_below_10_out,
        input,
        Kernel::Auto,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn trend_flow_trail_into_slice(
    alpha_trail_out: &mut [f64],
    alpha_trail_bullish_out: &mut [f64],
    alpha_trail_bearish_out: &mut [f64],
    alpha_dir_out: &mut [f64],
    mfi_out: &mut [f64],
    tp_upper_out: &mut [f64],
    tp_lower_out: &mut [f64],
    alpha_trail_bullish_switch_out: &mut [f64],
    alpha_trail_bearish_switch_out: &mut [f64],
    mfi_overbought_out: &mut [f64],
    mfi_oversold_out: &mut [f64],
    mfi_cross_up_mid_out: &mut [f64],
    mfi_cross_down_mid_out: &mut [f64],
    price_cross_alpha_trail_up_out: &mut [f64],
    price_cross_alpha_trail_down_out: &mut [f64],
    mfi_above_90_out: &mut [f64],
    mfi_below_10_out: &mut [f64],
    input: &TrendFlowTrailInput,
    kernel: Kernel,
) -> Result<(), TrendFlowTrailError> {
    let prepared = prepare_input(input, kernel)?;
    compute_row(
        prepared.open,
        prepared.high,
        prepared.low,
        prepared.close,
        prepared.volume,
        prepared.alpha_length,
        prepared.alpha_multiplier,
        prepared.mfi_length,
        alpha_trail_out,
        alpha_trail_bullish_out,
        alpha_trail_bearish_out,
        alpha_dir_out,
        mfi_out,
        tp_upper_out,
        tp_lower_out,
        alpha_trail_bullish_switch_out,
        alpha_trail_bearish_switch_out,
        mfi_overbought_out,
        mfi_oversold_out,
        mfi_cross_up_mid_out,
        mfi_cross_down_mid_out,
        price_cross_alpha_trail_up_out,
        price_cross_alpha_trail_down_out,
        mfi_above_90_out,
        mfi_below_10_out,
    )
}

#[derive(Clone, Debug)]
pub struct TrendFlowTrailBatchRange {
    pub alpha_length: (usize, usize, usize),
    pub alpha_multiplier: (f64, f64, f64),
    pub mfi_length: (usize, usize, usize),
}

impl Default for TrendFlowTrailBatchRange {
    fn default() -> Self {
        Self {
            alpha_length: (DEFAULT_ALPHA_LENGTH, DEFAULT_ALPHA_LENGTH, 0),
            alpha_multiplier: (DEFAULT_ALPHA_MULTIPLIER, DEFAULT_ALPHA_MULTIPLIER, 0.0),
            mfi_length: (DEFAULT_MFI_LENGTH, DEFAULT_MFI_LENGTH, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TrendFlowTrailBatchBuilder {
    pub range: TrendFlowTrailBatchRange,
    pub kernel: Kernel,
}

impl TrendFlowTrailBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn alpha_length(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.alpha_length = (start, end, step);
        self
    }
    pub fn alpha_multiplier(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.alpha_multiplier = (start, end, step);
        self
    }
    pub fn mfi_length(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.mfi_length = (start, end, step);
        self
    }
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }
    pub fn apply_slices(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Result<TrendFlowTrailBatchOutput, TrendFlowTrailError> {
        trend_flow_trail_batch_with_kernel(open, high, low, close, volume, &self.range, self.kernel)
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct TrendFlowTrailBatchOutput {
    pub rows: usize,
    pub cols: usize,
    pub alpha_trail: Vec<f64>,
    pub alpha_trail_bullish: Vec<f64>,
    pub alpha_trail_bearish: Vec<f64>,
    pub alpha_dir: Vec<f64>,
    pub mfi: Vec<f64>,
    pub tp_upper: Vec<f64>,
    pub tp_lower: Vec<f64>,
    pub alpha_trail_bullish_switch: Vec<f64>,
    pub alpha_trail_bearish_switch: Vec<f64>,
    pub mfi_overbought: Vec<f64>,
    pub mfi_oversold: Vec<f64>,
    pub mfi_cross_up_mid: Vec<f64>,
    pub mfi_cross_down_mid: Vec<f64>,
    pub price_cross_alpha_trail_up: Vec<f64>,
    pub price_cross_alpha_trail_down: Vec<f64>,
    pub mfi_above_90: Vec<f64>,
    pub mfi_below_10: Vec<f64>,
}

fn axis_usize(
    axis: &'static str,
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, TrendFlowTrailError> {
    if start == end || step == 0 {
        return Ok(vec![start]);
    }
    let mut out = Vec::new();
    if start < end {
        let mut value = start;
        while value <= end {
            out.push(value);
            value = value.saturating_add(step);
            if step == 0 {
                break;
            }
        }
    } else {
        let mut value = start;
        while value >= end {
            out.push(value);
            if value < step {
                break;
            }
            value -= step;
            if step == 0 {
                break;
            }
        }
    }
    if out.is_empty() || out.last().copied() != Some(end) {
        return Err(TrendFlowTrailError::InvalidRange {
            axis,
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

fn axis_f64(
    axis: &'static str,
    (start, end, step): (f64, f64, f64),
) -> Result<Vec<f64>, TrendFlowTrailError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() || step < 0.0 {
        return Err(TrendFlowTrailError::InvalidRange {
            axis,
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if (start - end).abs() <= f64::EPSILON || step == 0.0 {
        return Ok(vec![start]);
    }
    let mut out = Vec::new();
    let eps = step.abs() * 1e-9 + 1e-12;
    if start < end {
        let mut value = start;
        while value <= end + eps {
            out.push(value.min(end));
            value += step;
        }
    } else {
        let mut value = start;
        while value >= end - eps {
            out.push(value.max(end));
            value -= step;
        }
    }
    if out.is_empty() || (out.last().copied().unwrap_or(start) - end).abs() > eps {
        return Err(TrendFlowTrailError::InvalidRange {
            axis,
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

pub fn expand_grid_trend_flow_trail(
    sweep: &TrendFlowTrailBatchRange,
) -> Result<Vec<TrendFlowTrailParams>, TrendFlowTrailError> {
    let alpha_lengths = axis_usize("alpha_length", sweep.alpha_length)?;
    let alpha_multipliers = axis_f64("alpha_multiplier", sweep.alpha_multiplier)?;
    let mfi_lengths = axis_usize("mfi_length", sweep.mfi_length)?;
    let mut out =
        Vec::with_capacity(alpha_lengths.len() * alpha_multipliers.len() * mfi_lengths.len());
    for &alpha_length in &alpha_lengths {
        for &alpha_multiplier in &alpha_multipliers {
            for &mfi_length in &mfi_lengths {
                out.push(TrendFlowTrailParams {
                    alpha_length: Some(alpha_length),
                    alpha_multiplier: Some(alpha_multiplier),
                    mfi_length: Some(mfi_length),
                });
            }
        }
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn trend_flow_trail_batch_with_kernel(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &TrendFlowTrailBatchRange,
    kernel: Kernel,
) -> Result<TrendFlowTrailBatchOutput, TrendFlowTrailError> {
    match kernel {
        Kernel::Auto => {
            let _ = detect_best_batch_kernel();
        }
        k if !k.is_batch() => return Err(TrendFlowTrailError::InvalidKernelForBatch(k)),
        _ => {}
    }
    let (_, max_run) = analyze_valid_segments(open, high, low, close, volume)?;
    let combos = expand_grid_trend_flow_trail(sweep)?;
    for params in &combos {
        let needed = required_valid_bars(
            params.alpha_length.unwrap_or(DEFAULT_ALPHA_LENGTH),
            params.mfi_length.unwrap_or(DEFAULT_MFI_LENGTH),
        );
        if max_run < needed {
            return Err(TrendFlowTrailError::NotEnoughValidData {
                needed,
                valid: max_run,
            });
        }
    }
    let rows = combos.len();
    let cols = close.len();
    let total = rows * cols;
    let warmups: Vec<usize> = combos
        .iter()
        .map(|p| {
            required_valid_bars(
                p.alpha_length.unwrap_or(DEFAULT_ALPHA_LENGTH),
                p.mfi_length.unwrap_or(DEFAULT_MFI_LENGTH),
            ) - 1
        })
        .collect();

    let mut alpha_trail_mu = make_uninit_matrix(rows, cols);
    let mut alpha_trail_bullish_mu = make_uninit_matrix(rows, cols);
    let mut alpha_trail_bearish_mu = make_uninit_matrix(rows, cols);
    let mut alpha_dir_mu = make_uninit_matrix(rows, cols);
    let mut mfi_mu = make_uninit_matrix(rows, cols);
    let mut tp_upper_mu = make_uninit_matrix(rows, cols);
    let mut tp_lower_mu = make_uninit_matrix(rows, cols);
    let mut alpha_trail_bullish_switch_mu = make_uninit_matrix(rows, cols);
    let mut alpha_trail_bearish_switch_mu = make_uninit_matrix(rows, cols);
    let mut mfi_overbought_mu = make_uninit_matrix(rows, cols);
    let mut mfi_oversold_mu = make_uninit_matrix(rows, cols);
    let mut mfi_cross_up_mid_mu = make_uninit_matrix(rows, cols);
    let mut mfi_cross_down_mid_mu = make_uninit_matrix(rows, cols);
    let mut price_cross_alpha_trail_up_mu = make_uninit_matrix(rows, cols);
    let mut price_cross_alpha_trail_down_mu = make_uninit_matrix(rows, cols);
    let mut mfi_above_90_mu = make_uninit_matrix(rows, cols);
    let mut mfi_below_10_mu = make_uninit_matrix(rows, cols);

    for buf in [
        &mut alpha_trail_mu,
        &mut alpha_trail_bullish_mu,
        &mut alpha_trail_bearish_mu,
        &mut alpha_dir_mu,
        &mut mfi_mu,
        &mut tp_upper_mu,
        &mut tp_lower_mu,
        &mut alpha_trail_bullish_switch_mu,
        &mut alpha_trail_bearish_switch_mu,
        &mut mfi_overbought_mu,
        &mut mfi_oversold_mu,
        &mut mfi_cross_up_mid_mu,
        &mut mfi_cross_down_mid_mu,
        &mut price_cross_alpha_trail_up_mu,
        &mut price_cross_alpha_trail_down_mu,
        &mut mfi_above_90_mu,
        &mut mfi_below_10_mu,
    ] {
        init_matrix_prefixes(buf, cols, &warmups);
    }

    let alpha_trail =
        unsafe { std::slice::from_raw_parts_mut(alpha_trail_mu.as_mut_ptr() as *mut f64, total) };
    let alpha_trail_bullish = unsafe {
        std::slice::from_raw_parts_mut(alpha_trail_bullish_mu.as_mut_ptr() as *mut f64, total)
    };
    let alpha_trail_bearish = unsafe {
        std::slice::from_raw_parts_mut(alpha_trail_bearish_mu.as_mut_ptr() as *mut f64, total)
    };
    let alpha_dir =
        unsafe { std::slice::from_raw_parts_mut(alpha_dir_mu.as_mut_ptr() as *mut f64, total) };
    let mfi = unsafe { std::slice::from_raw_parts_mut(mfi_mu.as_mut_ptr() as *mut f64, total) };
    let tp_upper =
        unsafe { std::slice::from_raw_parts_mut(tp_upper_mu.as_mut_ptr() as *mut f64, total) };
    let tp_lower =
        unsafe { std::slice::from_raw_parts_mut(tp_lower_mu.as_mut_ptr() as *mut f64, total) };
    let alpha_trail_bullish_switch = unsafe {
        std::slice::from_raw_parts_mut(
            alpha_trail_bullish_switch_mu.as_mut_ptr() as *mut f64,
            total,
        )
    };
    let alpha_trail_bearish_switch = unsafe {
        std::slice::from_raw_parts_mut(
            alpha_trail_bearish_switch_mu.as_mut_ptr() as *mut f64,
            total,
        )
    };
    let mfi_overbought = unsafe {
        std::slice::from_raw_parts_mut(mfi_overbought_mu.as_mut_ptr() as *mut f64, total)
    };
    let mfi_oversold =
        unsafe { std::slice::from_raw_parts_mut(mfi_oversold_mu.as_mut_ptr() as *mut f64, total) };
    let mfi_cross_up_mid = unsafe {
        std::slice::from_raw_parts_mut(mfi_cross_up_mid_mu.as_mut_ptr() as *mut f64, total)
    };
    let mfi_cross_down_mid = unsafe {
        std::slice::from_raw_parts_mut(mfi_cross_down_mid_mu.as_mut_ptr() as *mut f64, total)
    };
    let price_cross_alpha_trail_up = unsafe {
        std::slice::from_raw_parts_mut(
            price_cross_alpha_trail_up_mu.as_mut_ptr() as *mut f64,
            total,
        )
    };
    let price_cross_alpha_trail_down = unsafe {
        std::slice::from_raw_parts_mut(
            price_cross_alpha_trail_down_mu.as_mut_ptr() as *mut f64,
            total,
        )
    };
    let mfi_above_90 =
        unsafe { std::slice::from_raw_parts_mut(mfi_above_90_mu.as_mut_ptr() as *mut f64, total) };
    let mfi_below_10 =
        unsafe { std::slice::from_raw_parts_mut(mfi_below_10_mu.as_mut_ptr() as *mut f64, total) };

    for (row, params) in combos.iter().enumerate() {
        let offset = row * cols;
        compute_row(
            open,
            high,
            low,
            close,
            volume,
            params.alpha_length.unwrap_or(DEFAULT_ALPHA_LENGTH),
            params.alpha_multiplier.unwrap_or(DEFAULT_ALPHA_MULTIPLIER),
            params.mfi_length.unwrap_or(DEFAULT_MFI_LENGTH),
            &mut alpha_trail[offset..offset + cols],
            &mut alpha_trail_bullish[offset..offset + cols],
            &mut alpha_trail_bearish[offset..offset + cols],
            &mut alpha_dir[offset..offset + cols],
            &mut mfi[offset..offset + cols],
            &mut tp_upper[offset..offset + cols],
            &mut tp_lower[offset..offset + cols],
            &mut alpha_trail_bullish_switch[offset..offset + cols],
            &mut alpha_trail_bearish_switch[offset..offset + cols],
            &mut mfi_overbought[offset..offset + cols],
            &mut mfi_oversold[offset..offset + cols],
            &mut mfi_cross_up_mid[offset..offset + cols],
            &mut mfi_cross_down_mid[offset..offset + cols],
            &mut price_cross_alpha_trail_up[offset..offset + cols],
            &mut price_cross_alpha_trail_down[offset..offset + cols],
            &mut mfi_above_90[offset..offset + cols],
            &mut mfi_below_10[offset..offset + cols],
        )?;
    }

    Ok(TrendFlowTrailBatchOutput {
        rows,
        cols,
        alpha_trail: alpha_trail.to_vec(),
        alpha_trail_bullish: alpha_trail_bullish.to_vec(),
        alpha_trail_bearish: alpha_trail_bearish.to_vec(),
        alpha_dir: alpha_dir.to_vec(),
        mfi: mfi.to_vec(),
        tp_upper: tp_upper.to_vec(),
        tp_lower: tp_lower.to_vec(),
        alpha_trail_bullish_switch: alpha_trail_bullish_switch.to_vec(),
        alpha_trail_bearish_switch: alpha_trail_bearish_switch.to_vec(),
        mfi_overbought: mfi_overbought.to_vec(),
        mfi_oversold: mfi_oversold.to_vec(),
        mfi_cross_up_mid: mfi_cross_up_mid.to_vec(),
        mfi_cross_down_mid: mfi_cross_down_mid.to_vec(),
        price_cross_alpha_trail_up: price_cross_alpha_trail_up.to_vec(),
        price_cross_alpha_trail_down: price_cross_alpha_trail_down.to_vec(),
        mfi_above_90: mfi_above_90.to_vec(),
        mfi_below_10: mfi_below_10.to_vec(),
    })
}

#[allow(clippy::too_many_arguments)]
pub fn trend_flow_trail_batch_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &TrendFlowTrailBatchRange,
    kernel: Kernel,
) -> Result<TrendFlowTrailBatchOutput, TrendFlowTrailError> {
    trend_flow_trail_batch_with_kernel(open, high, low, close, volume, sweep, kernel)
}

#[allow(clippy::too_many_arguments)]
pub fn trend_flow_trail_batch_par_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &TrendFlowTrailBatchRange,
    kernel: Kernel,
) -> Result<TrendFlowTrailBatchOutput, TrendFlowTrailError> {
    trend_flow_trail_batch_with_kernel(open, high, low, close, volume, sweep, kernel)
}

#[cfg(feature = "python")]
#[pyfunction(name = "trend_flow_trail")]
#[pyo3(signature = (open, high, low, close, volume, alpha_length=DEFAULT_ALPHA_LENGTH, alpha_multiplier=DEFAULT_ALPHA_MULTIPLIER, mfi_length=DEFAULT_MFI_LENGTH, kernel=None))]
pub fn trend_flow_trail_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    alpha_length: usize,
    alpha_multiplier: f64,
    mfi_length: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let kernel = validate_kernel(kernel, false)?;
    let open = open.as_slice()?;
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let volume = volume.as_slice()?;
    let input = TrendFlowTrailInput::from_slices(
        open,
        high,
        low,
        close,
        volume,
        TrendFlowTrailParams {
            alpha_length: Some(alpha_length),
            alpha_multiplier: Some(alpha_multiplier),
            mfi_length: Some(mfi_length),
        },
    );
    let out = py
        .allow_threads(|| trend_flow_trail_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dict = PyDict::new(py);
    dict.set_item("alpha_trail", out.alpha_trail.into_pyarray(py))?;
    dict.set_item(
        "alpha_trail_bullish",
        out.alpha_trail_bullish.into_pyarray(py),
    )?;
    dict.set_item(
        "alpha_trail_bearish",
        out.alpha_trail_bearish.into_pyarray(py),
    )?;
    dict.set_item("alpha_dir", out.alpha_dir.into_pyarray(py))?;
    dict.set_item("mfi", out.mfi.into_pyarray(py))?;
    dict.set_item("tp_upper", out.tp_upper.into_pyarray(py))?;
    dict.set_item("tp_lower", out.tp_lower.into_pyarray(py))?;
    dict.set_item(
        "alpha_trail_bullish_switch",
        out.alpha_trail_bullish_switch.into_pyarray(py),
    )?;
    dict.set_item(
        "alpha_trail_bearish_switch",
        out.alpha_trail_bearish_switch.into_pyarray(py),
    )?;
    dict.set_item("mfi_overbought", out.mfi_overbought.into_pyarray(py))?;
    dict.set_item("mfi_oversold", out.mfi_oversold.into_pyarray(py))?;
    dict.set_item("mfi_cross_up_mid", out.mfi_cross_up_mid.into_pyarray(py))?;
    dict.set_item(
        "mfi_cross_down_mid",
        out.mfi_cross_down_mid.into_pyarray(py),
    )?;
    dict.set_item(
        "price_cross_alpha_trail_up",
        out.price_cross_alpha_trail_up.into_pyarray(py),
    )?;
    dict.set_item(
        "price_cross_alpha_trail_down",
        out.price_cross_alpha_trail_down.into_pyarray(py),
    )?;
    dict.set_item("mfi_above_90", out.mfi_above_90.into_pyarray(py))?;
    dict.set_item("mfi_below_10", out.mfi_below_10.into_pyarray(py))?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyfunction(name = "trend_flow_trail_batch")]
#[pyo3(signature = (open, high, low, close, volume, alpha_length_range=(DEFAULT_ALPHA_LENGTH, DEFAULT_ALPHA_LENGTH, 0), alpha_multiplier_range=(DEFAULT_ALPHA_MULTIPLIER, DEFAULT_ALPHA_MULTIPLIER, 0.0), mfi_length_range=(DEFAULT_MFI_LENGTH, DEFAULT_MFI_LENGTH, 0), kernel=None))]
pub fn trend_flow_trail_batch_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    alpha_length_range: (usize, usize, usize),
    alpha_multiplier_range: (f64, f64, f64),
    mfi_length_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let kernel = validate_kernel(kernel, true)?;
    let out = trend_flow_trail_batch_with_kernel(
        open.as_slice()?,
        high.as_slice()?,
        low.as_slice()?,
        close.as_slice()?,
        volume.as_slice()?,
        &TrendFlowTrailBatchRange {
            alpha_length: alpha_length_range,
            alpha_multiplier: alpha_multiplier_range,
            mfi_length: mfi_length_range,
        },
        kernel,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dict = PyDict::new(py);
    dict.set_item("rows", out.rows)?;
    dict.set_item("cols", out.cols)?;
    dict.set_item(
        "alpha_trail",
        out.alpha_trail
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "alpha_trail_bullish",
        out.alpha_trail_bullish
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "alpha_trail_bearish",
        out.alpha_trail_bearish
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "alpha_dir",
        out.alpha_dir
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "mfi",
        out.mfi.into_pyarray(py).reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "tp_upper",
        out.tp_upper
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "tp_lower",
        out.tp_lower
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "alpha_trail_bullish_switch",
        out.alpha_trail_bullish_switch
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "alpha_trail_bearish_switch",
        out.alpha_trail_bearish_switch
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "mfi_overbought",
        out.mfi_overbought
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "mfi_oversold",
        out.mfi_oversold
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "mfi_cross_up_mid",
        out.mfi_cross_up_mid
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "mfi_cross_down_mid",
        out.mfi_cross_down_mid
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "price_cross_alpha_trail_up",
        out.price_cross_alpha_trail_up
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "price_cross_alpha_trail_down",
        out.price_cross_alpha_trail_down
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "mfi_above_90",
        out.mfi_above_90
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "mfi_below_10",
        out.mfi_below_10
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "TrendFlowTrailStream")]
pub struct TrendFlowTrailStreamPy {
    inner: TrendFlowTrailStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl TrendFlowTrailStreamPy {
    #[new]
    #[pyo3(signature = (alpha_length=None, alpha_multiplier=None, mfi_length=None))]
    pub fn new(
        alpha_length: Option<usize>,
        alpha_multiplier: Option<f64>,
        mfi_length: Option<usize>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: TrendFlowTrailStream::try_new(TrendFlowTrailParams {
                alpha_length,
                alpha_multiplier,
                mfi_length,
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?,
        })
    }

    pub fn update(
        &mut self,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Option<Vec<f64>> {
        self.inner
            .update(open, high, low, close, volume)
            .map(|row| {
                vec![
                    row.0, row.1, row.2, row.3, row.4, row.5, row.6, row.7, row.8, row.9, row.10,
                    row.11, row.12, row.13, row.14, row.15, row.16,
                ]
            })
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn trend_flow_trail_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    alpha_length: usize,
    alpha_multiplier: f64,
    mfi_length: usize,
) -> Result<JsValue, JsValue> {
    let out = trend_flow_trail_with_kernel(
        &TrendFlowTrailInput::from_slices(
            open,
            high,
            low,
            close,
            volume,
            TrendFlowTrailParams {
                alpha_length: Some(alpha_length),
                alpha_multiplier: Some(alpha_multiplier),
                mfi_length: Some(mfi_length),
            },
        ),
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&out).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn trend_flow_trail_alloc(len: usize) -> *mut f64 {
    let mut values = Vec::<f64>::with_capacity(len);
    let ptr = values.as_mut_ptr();
    std::mem::forget(values);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn trend_flow_trail_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn trend_flow_trail_into(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    volume_ptr: *const f64,
    alpha_trail_ptr: *mut f64,
    alpha_trail_bullish_ptr: *mut f64,
    alpha_trail_bearish_ptr: *mut f64,
    alpha_dir_ptr: *mut f64,
    mfi_ptr: *mut f64,
    tp_upper_ptr: *mut f64,
    tp_lower_ptr: *mut f64,
    alpha_trail_bullish_switch_ptr: *mut f64,
    alpha_trail_bearish_switch_ptr: *mut f64,
    mfi_overbought_ptr: *mut f64,
    mfi_oversold_ptr: *mut f64,
    mfi_cross_up_mid_ptr: *mut f64,
    mfi_cross_down_mid_ptr: *mut f64,
    price_cross_alpha_trail_up_ptr: *mut f64,
    price_cross_alpha_trail_down_ptr: *mut f64,
    mfi_above_90_ptr: *mut f64,
    mfi_below_10_ptr: *mut f64,
    len: usize,
    alpha_length: usize,
    alpha_multiplier: f64,
    mfi_length: usize,
) -> Result<(), JsValue> {
    unsafe {
        trend_flow_trail_into_slice(
            std::slice::from_raw_parts_mut(alpha_trail_ptr, len),
            std::slice::from_raw_parts_mut(alpha_trail_bullish_ptr, len),
            std::slice::from_raw_parts_mut(alpha_trail_bearish_ptr, len),
            std::slice::from_raw_parts_mut(alpha_dir_ptr, len),
            std::slice::from_raw_parts_mut(mfi_ptr, len),
            std::slice::from_raw_parts_mut(tp_upper_ptr, len),
            std::slice::from_raw_parts_mut(tp_lower_ptr, len),
            std::slice::from_raw_parts_mut(alpha_trail_bullish_switch_ptr, len),
            std::slice::from_raw_parts_mut(alpha_trail_bearish_switch_ptr, len),
            std::slice::from_raw_parts_mut(mfi_overbought_ptr, len),
            std::slice::from_raw_parts_mut(mfi_oversold_ptr, len),
            std::slice::from_raw_parts_mut(mfi_cross_up_mid_ptr, len),
            std::slice::from_raw_parts_mut(mfi_cross_down_mid_ptr, len),
            std::slice::from_raw_parts_mut(price_cross_alpha_trail_up_ptr, len),
            std::slice::from_raw_parts_mut(price_cross_alpha_trail_down_ptr, len),
            std::slice::from_raw_parts_mut(mfi_above_90_ptr, len),
            std::slice::from_raw_parts_mut(mfi_below_10_ptr, len),
            &TrendFlowTrailInput::from_slices(
                std::slice::from_raw_parts(open_ptr, len),
                std::slice::from_raw_parts(high_ptr, len),
                std::slice::from_raw_parts(low_ptr, len),
                std::slice::from_raw_parts(close_ptr, len),
                std::slice::from_raw_parts(volume_ptr, len),
                TrendFlowTrailParams {
                    alpha_length: Some(alpha_length),
                    alpha_multiplier: Some(alpha_multiplier),
                    mfi_length: Some(mfi_length),
                },
            ),
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct TrendFlowTrailBatchConfig {
    pub alpha_length_range: (usize, usize, usize),
    pub alpha_multiplier_range: (f64, f64, f64),
    pub mfi_length_range: (usize, usize, usize),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = trend_flow_trail_batch)]
pub fn trend_flow_trail_batch_unified_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: TrendFlowTrailBatchConfig =
        serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let out = trend_flow_trail_batch_with_kernel(
        open,
        high,
        low,
        close,
        volume,
        &TrendFlowTrailBatchRange {
            alpha_length: config.alpha_length_range,
            alpha_multiplier: config.alpha_multiplier_range,
            mfi_length: config.mfi_length_range,
        },
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&out).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub struct TrendFlowTrailStreamWasm {
    inner: TrendFlowTrailStream,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
impl TrendFlowTrailStreamWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(
        alpha_length: Option<usize>,
        alpha_multiplier: Option<f64>,
        mfi_length: Option<usize>,
    ) -> Result<Self, JsValue> {
        Ok(Self {
            inner: TrendFlowTrailStream::try_new(TrendFlowTrailParams {
                alpha_length,
                alpha_multiplier,
                mfi_length,
            })
            .map_err(|e| JsValue::from_str(&e.to_string()))?,
        })
    }

    pub fn update(
        &mut self,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Result<JsValue, JsValue> {
        let value = self
            .inner
            .update(open, high, low, close, volume)
            .map(|row| {
                vec![
                    row.0, row.1, row.2, row.3, row.4, row.5, row.6, row.7, row.8, row.9, row.10,
                    row.11, row.12, row.13, row.14, row.15, row.16,
                ]
            });
        serde_wasm_bindgen::to_value(&value).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn trend_flow_trail_output_into_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    alpha_length: usize,
    alpha_multiplier: f64,
    mfi_length: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = trend_flow_trail_js(
        open,
        high,
        low,
        close,
        volume,
        alpha_length,
        alpha_multiplier,
        mfi_length,
    )?;
    crate::write_wasm_object_f64_outputs("trend_flow_trail_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn trend_flow_trail_batch_unified_output_into_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = trend_flow_trail_batch_unified_js(open, high, low, close, volume, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "trend_flow_trail_batch_unified_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_ohlcv() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut open = Vec::with_capacity(180);
        let mut high = Vec::with_capacity(180);
        let mut low = Vec::with_capacity(180);
        let mut close = Vec::with_capacity(180);
        let mut volume = Vec::with_capacity(180);
        for i in 0..180 {
            let base = if i < 60 {
                100.0 + i as f64 * 0.35
            } else if i < 120 {
                121.0 - (i - 60) as f64 * 0.22
            } else {
                108.0 + (i - 120) as f64 * 0.42
            };
            let wiggle = ((i % 7) as f64 - 3.0) * 0.18;
            let c = base + wiggle;
            let o = c - ((i % 5) as f64 - 2.0) * 0.11;
            open.push(o);
            close.push(c);
            high.push(c.max(o) + 1.2 + (i % 3) as f64 * 0.07);
            low.push(c.min(o) - 1.1 - (i % 4) as f64 * 0.05);
            volume.push(1000.0 + (i % 9) as f64 * 37.0 + i as f64 * 4.0);
        }
        (open, high, low, close, volume)
    }

    fn assert_series_eq(left: &[f64], right: &[f64]) {
        assert_eq!(left.len(), right.len());
        for (lhs, rhs) in left.iter().zip(right.iter()) {
            assert!(lhs == rhs || (lhs.is_nan() && rhs.is_nan()));
        }
    }

    #[test]
    fn trend_flow_trail_outputs_present() -> Result<(), TrendFlowTrailError> {
        let (open, high, low, close, volume) = sample_ohlcv();
        let out = trend_flow_trail(&TrendFlowTrailInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            &volume,
            TrendFlowTrailParams::default(),
        ))?;
        assert!(out.alpha_trail.iter().any(|v| v.is_finite()));
        assert!(out.mfi.iter().any(|v| v.is_finite()));
        Ok(())
    }

    #[test]
    fn trend_flow_trail_stream_matches_api() -> Result<(), TrendFlowTrailError> {
        let (open, high, low, close, volume) = sample_ohlcv();
        let params = TrendFlowTrailParams::default();
        let out = trend_flow_trail(&TrendFlowTrailInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            &volume,
            params.clone(),
        ))?;
        let mut stream = TrendFlowTrailStream::try_new(params)?;
        let mut alpha_trail = vec![f64::NAN; close.len()];
        for i in 0..close.len() {
            if let Some((trail, ..)) = stream.update(open[i], high[i], low[i], close[i], volume[i])
            {
                alpha_trail[i] = trail;
            }
        }
        assert_series_eq(&alpha_trail, &out.alpha_trail);
        Ok(())
    }
}
