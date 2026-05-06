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

use crate::indicators::moving_averages::ema::{EmaParams, EmaStream};
use crate::indicators::moving_averages::hma::{HmaParams, HmaStream};
use crate::indicators::moving_averages::sma::{SmaParams, SmaStream};
use crate::indicators::moving_averages::vwma::{VwmaParams, VwmaStream};
use crate::indicators::moving_averages::wma::{WmaParams, WmaStream};
use crate::indicators::moving_averages::zlema::{ZlemaParams, ZlemaStream};
use crate::indicators::tsf::{TsfParams, TsfStream};
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_uninit_f64, alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::error::Error;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum KairiRelativeIndexData<'a> {
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
pub struct KairiRelativeIndexOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct KairiRelativeIndexParams {
    pub length: Option<usize>,
    pub ma_type: Option<String>,
}

impl Default for KairiRelativeIndexParams {
    fn default() -> Self {
        Self {
            length: Some(50),
            ma_type: Some("SMA".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct KairiRelativeIndexInput<'a> {
    pub data: KairiRelativeIndexData<'a>,
    pub params: KairiRelativeIndexParams,
}

impl<'a> KairiRelativeIndexInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: KairiRelativeIndexParams,
    ) -> Self {
        Self {
            data: KairiRelativeIndexData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        source: &'a [f64],
        volume: &'a [f64],
        params: KairiRelativeIndexParams,
    ) -> Self {
        Self {
            data: KairiRelativeIndexData::Slices { source, volume },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", KairiRelativeIndexParams::default())
    }

    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(50)
    }

    #[inline]
    pub fn get_ma_type(&self) -> String {
        self.params
            .ma_type
            .clone()
            .unwrap_or_else(|| "SMA".to_string())
    }
}

#[derive(Copy, Clone, Debug)]
pub struct KairiRelativeIndexBuilder {
    length: Option<usize>,
    ma_type: Option<&'static str>,
    kernel: Kernel,
}

impl Default for KairiRelativeIndexBuilder {
    fn default() -> Self {
        Self {
            length: None,
            ma_type: None,
            kernel: Kernel::Auto,
        }
    }
}

impl KairiRelativeIndexBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn length(mut self, value: usize) -> Self {
        self.length = Some(value);
        self
    }

    #[inline(always)]
    pub fn ma_type(mut self, value: &'static str) -> Self {
        self.ma_type = Some(value);
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
    ) -> Result<KairiRelativeIndexOutput, KairiRelativeIndexError> {
        let params = KairiRelativeIndexParams {
            length: self.length,
            ma_type: self.ma_type.map(str::to_string),
        };
        kairi_relative_index_with_kernel(
            &KairiRelativeIndexInput::from_candles(candles, "close", params),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<KairiRelativeIndexOutput, KairiRelativeIndexError> {
        let params = KairiRelativeIndexParams {
            length: self.length,
            ma_type: self.ma_type.map(str::to_string),
        };
        kairi_relative_index_with_kernel(
            &KairiRelativeIndexInput::from_candles(candles, source, params),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        source: &[f64],
        volume: &[f64],
    ) -> Result<KairiRelativeIndexOutput, KairiRelativeIndexError> {
        let params = KairiRelativeIndexParams {
            length: self.length,
            ma_type: self.ma_type.map(str::to_string),
        };
        kairi_relative_index_with_kernel(
            &KairiRelativeIndexInput::from_slices(source, volume, params),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<KairiRelativeIndexStream, KairiRelativeIndexError> {
        KairiRelativeIndexStream::try_new(KairiRelativeIndexParams {
            length: self.length,
            ma_type: self.ma_type.map(str::to_string),
        })
    }
}

#[derive(Debug, Error)]
pub enum KairiRelativeIndexError {
    #[error("kairi_relative_index: Input data slice is empty.")]
    EmptyInputData,
    #[error(
        "kairi_relative_index: Input length mismatch: source = {source_len}, volume = {volume_len}"
    )]
    InputLengthMismatch {
        source_len: usize,
        volume_len: usize,
    },
    #[error("kairi_relative_index: All values are NaN.")]
    AllValuesNaN,
    #[error("kairi_relative_index: Invalid length: length = {length}, data length = {data_len}")]
    InvalidLength { length: usize, data_len: usize },
    #[error("kairi_relative_index: Invalid moving average type: {ma_type}")]
    InvalidMaType { ma_type: String },
    #[error("kairi_relative_index: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("kairi_relative_index: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("kairi_relative_index: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("kairi_relative_index: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error(
        "kairi_relative_index: Output length mismatch: dst = {dst_len}, expected = {expected_len}"
    )]
    MismatchedOutputLen { dst_len: usize, expected_len: usize },
    #[error("kairi_relative_index: Invalid input: {msg}")]
    InvalidInput { msg: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KairiMaKind {
    Sma,
    Ema,
    Wma,
    Tma,
    Vidya,
    Wwma,
    Zlema,
    Tsf,
    Hma,
    Vwma,
}

impl KairiMaKind {
    #[inline(always)]
    fn parse(value: &str) -> Result<Self, KairiRelativeIndexError> {
        match value.trim().to_ascii_uppercase().as_str() {
            "SMA" => Ok(Self::Sma),
            "EMA" => Ok(Self::Ema),
            "WMA" => Ok(Self::Wma),
            "TMA" => Ok(Self::Tma),
            "VIDYA" => Ok(Self::Vidya),
            "WWMA" => Ok(Self::Wwma),
            "ZLEMA" => Ok(Self::Zlema),
            "TSF" => Ok(Self::Tsf),
            "HMA" | "HULL" => Ok(Self::Hma),
            "VWMA" => Ok(Self::Vwma),
            other => Err(KairiRelativeIndexError::InvalidMaType {
                ma_type: other.to_string(),
            }),
        }
    }

    #[inline(always)]
    fn canonical(self) -> &'static str {
        match self {
            Self::Sma => "SMA",
            Self::Ema => "EMA",
            Self::Wma => "WMA",
            Self::Tma => "TMA",
            Self::Vidya => "VIDYA",
            Self::Wwma => "WWMA",
            Self::Zlema => "ZLEMA",
            Self::Tsf => "TSF",
            Self::Hma => "HMA",
            Self::Vwma => "VWMA",
        }
    }

    #[inline(always)]
    fn needs_volume(self) -> bool {
        matches!(self, Self::Vwma)
    }

    #[inline(always)]
    fn required_samples(self, length: usize) -> usize {
        match self {
            Self::Wwma | Self::Vidya => 1,
            Self::Hma => length + (length as f64).sqrt().floor() as usize - 1,
            _ => length,
        }
    }
}

#[derive(Debug, Clone)]
struct ValidatedKairiRelativeIndexParams {
    length: usize,
    ma_kind: KairiMaKind,
    ma_type: String,
}

#[derive(Debug, Clone)]
struct TmaStreamLocal {
    stage1: SmaStream,
    stage2: SmaStream,
}

impl TmaStreamLocal {
    #[inline(always)]
    fn try_new(length: usize) -> Result<Self, KairiRelativeIndexError> {
        let p1 = (length + 1) / 2;
        let p2 = length / 2 + 1;
        Ok(Self {
            stage1: SmaStream::try_new(SmaParams { period: Some(p1) })
                .map_err(|e| KairiRelativeIndexError::InvalidInput { msg: e.to_string() })?,
            stage2: SmaStream::try_new(SmaParams { period: Some(p2) })
                .map_err(|e| KairiRelativeIndexError::InvalidInput { msg: e.to_string() })?,
        })
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<f64> {
        self.stage1
            .update(value)
            .and_then(|v| self.stage2.update(v))
    }
}

#[derive(Debug, Clone)]
struct WwmaStreamLocal {
    alpha: f64,
    state: f64,
    initialized: bool,
}

impl WwmaStreamLocal {
    #[inline(always)]
    fn try_new(length: usize) -> Result<Self, KairiRelativeIndexError> {
        Ok(Self {
            alpha: 1.0 / length as f64,
            state: f64::NAN,
            initialized: false,
        })
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<f64> {
        if !self.initialized {
            self.state = value;
            self.initialized = true;
        } else {
            self.state = self.alpha.mul_add(value, (1.0 - self.alpha) * self.state);
        }
        Some(self.state)
    }
}

#[derive(Debug, Clone)]
struct VidyaStreamLocal {
    alpha: f64,
    prev: f64,
    have_prev: bool,
    state: f64,
    initialized: bool,
    ring_up: [f64; 9],
    ring_down: [f64; 9],
    head: usize,
    count: usize,
    sum_up: f64,
    sum_down: f64,
}

impl VidyaStreamLocal {
    #[inline(always)]
    fn try_new(length: usize) -> Result<Self, KairiRelativeIndexError> {
        Ok(Self {
            alpha: 2.0 / (length as f64 + 1.0),
            prev: f64::NAN,
            have_prev: false,
            state: f64::NAN,
            initialized: false,
            ring_up: [0.0; 9],
            ring_down: [0.0; 9],
            head: 0,
            count: 0,
            sum_up: 0.0,
            sum_down: 0.0,
        })
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<f64> {
        if !self.have_prev {
            self.prev = value;
            self.have_prev = true;
            self.state = value;
            self.initialized = true;
            return Some(self.state);
        }

        let diff = value - self.prev;
        self.prev = value;

        let up = diff.max(0.0);
        let down = (-diff).max(0.0);

        if self.count == 9 {
            self.sum_up -= self.ring_up[self.head];
            self.sum_down -= self.ring_down[self.head];
        } else {
            self.count += 1;
        }

        self.ring_up[self.head] = up;
        self.ring_down[self.head] = down;
        self.sum_up += up;
        self.sum_down += down;
        self.head += 1;
        if self.head == 9 {
            self.head = 0;
        }

        let denom = self.sum_up + self.sum_down;
        let cmo_abs = if denom > 0.0 {
            ((self.sum_up - self.sum_down) / denom).abs()
        } else {
            0.0
        };
        let adaptive_alpha = self.alpha * cmo_abs;

        if !self.initialized {
            self.state = value;
            self.initialized = true;
        } else {
            self.state = adaptive_alpha.mul_add(value, (1.0 - adaptive_alpha) * self.state);
        }
        Some(self.state)
    }
}

#[derive(Debug, Clone)]
enum KairiMaState {
    Sma(SmaStream),
    Ema(EmaStream),
    Wma(WmaStream),
    Tma(TmaStreamLocal),
    Vidya(VidyaStreamLocal),
    Wwma(WwmaStreamLocal),
    Zlema(ZlemaStream),
    Tsf(TsfStream),
    Hma(HmaStream),
    Vwma(VwmaStream),
}

#[derive(Debug, Clone)]
pub struct KairiRelativeIndexStream {
    params: KairiRelativeIndexParams,
    ma_kind: KairiMaKind,
    state: KairiMaState,
}

impl KairiRelativeIndexStream {
    #[inline(always)]
    pub fn try_new(params: KairiRelativeIndexParams) -> Result<Self, KairiRelativeIndexError> {
        let length = params.length.unwrap_or(50);
        if length < 2 {
            return Err(KairiRelativeIndexError::InvalidLength {
                length,
                data_len: 0,
            });
        }
        let ma_type = params.ma_type.clone().unwrap_or_else(|| "SMA".to_string());
        let ma_kind = KairiMaKind::parse(&ma_type)?;
        let state = build_state(ma_kind, length)?;
        Ok(Self {
            params,
            ma_kind,
            state,
        })
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        *self = Self::try_new(self.params.clone())
            .expect("KairiRelativeIndexStream::reset: params should remain valid");
    }

    #[inline(always)]
    pub fn update(&mut self, source: f64, volume: f64) -> Option<f64> {
        if !source.is_finite() || (self.ma_kind.needs_volume() && !volume.is_finite()) {
            self.reset();
            return None;
        }

        let ma = match &mut self.state {
            KairiMaState::Sma(state) => state.update(source),
            KairiMaState::Ema(state) => state.update(source),
            KairiMaState::Wma(state) => state.update(source),
            KairiMaState::Tma(state) => state.update(source),
            KairiMaState::Vidya(state) => state.update(source),
            KairiMaState::Wwma(state) => state.update(source),
            KairiMaState::Zlema(state) => state.update(source),
            KairiMaState::Tsf(state) => state.update(source),
            KairiMaState::Hma(state) => state.update(source),
            KairiMaState::Vwma(state) => state.update(source, volume),
        }?;

        if !ma.is_finite() {
            return None;
        }

        if ma == 0.0 {
            return if source == 0.0 { Some(0.0) } else { None };
        }

        Some((source - ma) * 100.0 / ma)
    }

    #[inline(always)]
    pub fn get_warmup_period(&self) -> usize {
        self.ma_kind
            .required_samples(self.params.length.unwrap_or(50))
            .saturating_sub(1)
    }
}

#[inline(always)]
fn build_state(
    ma_kind: KairiMaKind,
    length: usize,
) -> Result<KairiMaState, KairiRelativeIndexError> {
    Ok(match ma_kind {
        KairiMaKind::Sma => KairiMaState::Sma(
            SmaStream::try_new(SmaParams {
                period: Some(length),
            })
            .map_err(|e| KairiRelativeIndexError::InvalidInput { msg: e.to_string() })?,
        ),
        KairiMaKind::Ema => KairiMaState::Ema(
            EmaStream::try_new(EmaParams {
                period: Some(length),
            })
            .map_err(|e| KairiRelativeIndexError::InvalidInput { msg: e.to_string() })?,
        ),
        KairiMaKind::Wma => KairiMaState::Wma(
            WmaStream::try_new(WmaParams {
                period: Some(length),
            })
            .map_err(|e| KairiRelativeIndexError::InvalidInput { msg: e.to_string() })?,
        ),
        KairiMaKind::Tma => KairiMaState::Tma(TmaStreamLocal::try_new(length)?),
        KairiMaKind::Vidya => KairiMaState::Vidya(VidyaStreamLocal::try_new(length)?),
        KairiMaKind::Wwma => KairiMaState::Wwma(WwmaStreamLocal::try_new(length)?),
        KairiMaKind::Zlema => KairiMaState::Zlema(
            ZlemaStream::try_new(ZlemaParams {
                period: Some(length),
            })
            .map_err(|e| KairiRelativeIndexError::InvalidInput { msg: e.to_string() })?,
        ),
        KairiMaKind::Tsf => KairiMaState::Tsf(
            TsfStream::try_new(TsfParams {
                period: Some(length),
            })
            .map_err(|e| KairiRelativeIndexError::InvalidInput { msg: e.to_string() })?,
        ),
        KairiMaKind::Hma => KairiMaState::Hma(
            HmaStream::try_new(HmaParams {
                period: Some(length),
            })
            .map_err(|e| KairiRelativeIndexError::InvalidInput { msg: e.to_string() })?,
        ),
        KairiMaKind::Vwma => KairiMaState::Vwma(
            VwmaStream::try_new(VwmaParams {
                period: Some(length),
            })
            .map_err(|e| KairiRelativeIndexError::InvalidInput { msg: e.to_string() })?,
        ),
    })
}

#[inline(always)]
fn input_slices<'a>(
    input: &'a KairiRelativeIndexInput<'a>,
) -> Result<(&'a [f64], &'a [f64]), KairiRelativeIndexError> {
    match &input.data {
        KairiRelativeIndexData::Candles { candles, source } => {
            let source = match *source {
                "open" => &candles.open,
                "high" => &candles.high,
                "low" => &candles.low,
                "close" => &candles.close,
                "volume" => &candles.volume,
                "hl2" => &candles.hl2,
                "hlc3" => &candles.hlc3,
                "ohlc4" => &candles.ohlc4,
                "hlcc4" | "hlcc" => &candles.hlcc4,
                _ => source_type(candles, source),
            };
            Ok((source, candles.volume.as_slice()))
        }
        KairiRelativeIndexData::Slices { source, volume } => Ok((*source, *volume)),
    }
}

#[inline(always)]
fn longest_valid_run(source: &[f64], volume: &[f64], ma_kind: KairiMaKind) -> usize {
    let mut best = 0usize;
    let mut cur = 0usize;
    for i in 0..source.len() {
        let valid = source[i].is_finite() && (!ma_kind.needs_volume() || volume[i].is_finite());
        if valid {
            cur += 1;
            if cur > best {
                best = cur;
            }
        } else {
            cur = 0;
        }
    }
    best
}

#[inline(always)]
fn validate_input<'a>(
    input: &'a KairiRelativeIndexInput<'a>,
    kernel: Kernel,
) -> Result<
    (
        &'a [f64],
        &'a [f64],
        ValidatedKairiRelativeIndexParams,
        Kernel,
    ),
    KairiRelativeIndexError,
> {
    let (source, volume) = input_slices(input)?;
    if source.is_empty() {
        return Err(KairiRelativeIndexError::EmptyInputData);
    }
    if source.len() != volume.len() {
        return Err(KairiRelativeIndexError::InputLengthMismatch {
            source_len: source.len(),
            volume_len: volume.len(),
        });
    }

    let ma_type = input.get_ma_type();
    let ma_kind = KairiMaKind::parse(&ma_type)?;
    if !source.iter().any(|v| v.is_finite()) {
        return Err(KairiRelativeIndexError::AllValuesNaN);
    }

    let length = input.get_length();
    if length < 2 || length > source.len() {
        return Err(KairiRelativeIndexError::InvalidLength {
            length,
            data_len: source.len(),
        });
    }

    let valid = longest_valid_run(source, volume, ma_kind);
    let needed = ma_kind.required_samples(length);
    if valid < needed {
        return Err(KairiRelativeIndexError::NotEnoughValidData { needed, valid });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other if other.is_batch() => other.to_non_batch(),
        other => other,
    };

    Ok((
        source,
        volume,
        ValidatedKairiRelativeIndexParams {
            length,
            ma_kind,
            ma_type: ma_kind.canonical().to_string(),
        },
        chosen,
    ))
}

#[inline(always)]
fn compute_into(
    source: &[f64],
    volume: &[f64],
    params: &ValidatedKairiRelativeIndexParams,
    out: &mut [f64],
) -> Result<(), KairiRelativeIndexError> {
    let mut stream = KairiRelativeIndexStream::try_new(KairiRelativeIndexParams {
        length: Some(params.length),
        ma_type: Some(params.ma_type.clone()),
    })?;
    for (dst, (&src, &vol)) in out.iter_mut().zip(source.iter().zip(volume.iter())) {
        *dst = stream.update(src, vol).unwrap_or(f64::NAN);
    }
    Ok(())
}

#[inline(always)]
fn is_default_kairi_params(params: &ValidatedKairiRelativeIndexParams) -> bool {
    params.length == 50 && matches!(params.ma_kind, KairiMaKind::Sma)
}

#[inline]
fn compute_default_sma50_into(source: &[f64], out: &mut [f64]) {
    const PERIOD: usize = 50;
    const SCALE: f64 = 100.0;
    const RCP: f64 = 1.0 / PERIOD as f64;

    out.fill(f64::NAN);
    let mut sum = 0.0;
    let mut valid_count = 0usize;
    let mut values = [0.0f64; PERIOD];
    let mut valid = [0u8; PERIOD];
    let mut head = 0usize;
    let mut count = 0usize;

    for i in 0..source.len() {
        if count == PERIOD {
            if valid[head] != 0 {
                sum -= values[head];
                valid_count -= 1;
            }
        } else {
            count += 1;
        }

        let src = source[i];
        if src.is_finite() {
            values[head] = src;
            valid[head] = 1;
            sum += src;
            valid_count += 1;
        } else {
            values[head] = 0.0;
            valid[head] = 0;
        }

        head += 1;
        if head == PERIOD {
            head = 0;
        }

        if count == PERIOD && valid_count == PERIOD {
            let ma = sum * RCP;
            if ma != 0.0 {
                out[i] = (src - ma) * SCALE / ma;
            } else if src == 0.0 {
                out[i] = 0.0;
            }
        }
    }
}

#[inline]
pub fn kairi_relative_index(
    input: &KairiRelativeIndexInput,
) -> Result<KairiRelativeIndexOutput, KairiRelativeIndexError> {
    kairi_relative_index_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn kairi_relative_index_with_kernel(
    input: &KairiRelativeIndexInput,
    kernel: Kernel,
) -> Result<KairiRelativeIndexOutput, KairiRelativeIndexError> {
    let (source, volume, params, _chosen) = validate_input(input, kernel)?;
    let mut out = if is_default_kairi_params(&params) {
        alloc_uninit_f64(source.len())
    } else {
        alloc_with_nan_prefix(source.len(), source.len())
    };
    if is_default_kairi_params(&params) {
        compute_default_sma50_into(source, &mut out);
    } else {
        compute_into(source, volume, &params, &mut out)?;
    }
    Ok(KairiRelativeIndexOutput { values: out })
}

#[inline]
pub fn kairi_relative_index_into_slice(
    dst: &mut [f64],
    input: &KairiRelativeIndexInput,
    kernel: Kernel,
) -> Result<(), KairiRelativeIndexError> {
    let (source, volume, params, _chosen) = validate_input(input, kernel)?;
    if dst.len() != source.len() {
        return Err(KairiRelativeIndexError::OutputLengthMismatch {
            expected: source.len(),
            got: dst.len(),
        });
    }
    if is_default_kairi_params(&params) {
        compute_default_sma50_into(source, dst);
    } else {
        for v in dst.iter_mut() {
            *v = f64::NAN;
        }
        compute_into(source, volume, &params, dst)?;
    }
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn kairi_relative_index_into(
    input: &KairiRelativeIndexInput,
    out: &mut [f64],
) -> Result<(), KairiRelativeIndexError> {
    kairi_relative_index_into_slice(out, input, Kernel::Auto)
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct KairiRelativeIndexBatchRange {
    pub length: (usize, usize, usize),
    pub ma_type: String,
}

impl Default for KairiRelativeIndexBatchRange {
    fn default() -> Self {
        Self {
            length: (50, 258, 1),
            ma_type: "SMA".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct KairiRelativeIndexBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<KairiRelativeIndexParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Debug)]
pub struct KairiRelativeIndexBatchBuilder {
    range: KairiRelativeIndexBatchRange,
    kernel: Kernel,
}

impl Default for KairiRelativeIndexBatchBuilder {
    fn default() -> Self {
        Self {
            range: KairiRelativeIndexBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl KairiRelativeIndexBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline(always)]
    pub fn length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.length = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn length_static(mut self, length: usize) -> Self {
        self.range.length = (length, length, 0);
        self
    }

    #[inline(always)]
    pub fn ma_type(mut self, ma_type: impl Into<String>) -> Self {
        self.range.ma_type = ma_type.into();
        self
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        source: &[f64],
        volume: &[f64],
    ) -> Result<KairiRelativeIndexBatchOutput, KairiRelativeIndexError> {
        kairi_relative_index_batch_with_kernel(source, volume, &self.range, self.kernel)
    }

    #[inline(always)]
    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<KairiRelativeIndexBatchOutput, KairiRelativeIndexError> {
        kairi_relative_index_batch_with_kernel(
            source_type(candles, source),
            candles.volume.as_slice(),
            &self.range,
            self.kernel,
        )
    }
}

#[inline(always)]
pub fn expand_grid_kairi_relative_index(
    range: &KairiRelativeIndexBatchRange,
) -> Result<Vec<KairiRelativeIndexParams>, KairiRelativeIndexError> {
    let (start, end, step) = range.length;
    let lengths = if step == 0 || start == end {
        vec![start]
    } else if start < end {
        (start..=end).step_by(step.max(1)).collect()
    } else {
        let mut values = Vec::new();
        let mut current = start;
        let stride = step.max(1);
        loop {
            values.push(current);
            if current <= end {
                break;
            }
            let next = current.saturating_sub(stride);
            if next == current {
                break;
            }
            current = next;
        }
        values
    };

    if lengths.is_empty() {
        return Err(KairiRelativeIndexError::InvalidRange { start, end, step });
    }

    let mut combos = Vec::with_capacity(lengths.len());
    for length in lengths {
        combos.push(KairiRelativeIndexParams {
            length: Some(length),
            ma_type: Some(range.ma_type.clone()),
        });
    }
    Ok(combos)
}

#[inline(always)]
fn to_batch_kernel(kernel: Kernel) -> Result<Kernel, KairiRelativeIndexError> {
    Ok(match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(KairiRelativeIndexError::InvalidKernelForBatch(other)),
    })
}

pub fn kairi_relative_index_batch_with_kernel(
    source: &[f64],
    volume: &[f64],
    sweep: &KairiRelativeIndexBatchRange,
    kernel: Kernel,
) -> Result<KairiRelativeIndexBatchOutput, KairiRelativeIndexError> {
    let batch_kernel = to_batch_kernel(kernel)?;
    let combos = expand_grid_kairi_relative_index(sweep)?;
    let rows = combos.len();
    let cols = source.len();
    if cols == 0 {
        return Err(KairiRelativeIndexError::EmptyInputData);
    }
    if cols != volume.len() {
        return Err(KairiRelativeIndexError::InputLengthMismatch {
            source_len: cols,
            volume_len: volume.len(),
        });
    }

    let mut values = alloc_with_nan_prefix(rows * cols, rows * cols);
    kairi_relative_index_batch_inner_into(
        source,
        volume,
        &combos,
        batch_kernel,
        true,
        &mut values,
    )?;

    Ok(KairiRelativeIndexBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline]
pub fn kairi_relative_index_batch_slice(
    source: &[f64],
    volume: &[f64],
    sweep: &KairiRelativeIndexBatchRange,
    kernel: Kernel,
) -> Result<KairiRelativeIndexBatchOutput, KairiRelativeIndexError> {
    let batch_kernel = to_batch_kernel(kernel)?;
    let combos = expand_grid_kairi_relative_index(sweep)?;
    let rows = combos.len();
    let cols = source.len();
    let mut values = alloc_with_nan_prefix(rows * cols, rows * cols);
    kairi_relative_index_batch_inner_into(
        source,
        volume,
        &combos,
        batch_kernel,
        false,
        &mut values,
    )?;
    Ok(KairiRelativeIndexBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline]
pub fn kairi_relative_index_batch_par_slice(
    source: &[f64],
    volume: &[f64],
    sweep: &KairiRelativeIndexBatchRange,
    kernel: Kernel,
) -> Result<KairiRelativeIndexBatchOutput, KairiRelativeIndexError> {
    kairi_relative_index_batch_with_kernel(source, volume, sweep, kernel)
}

fn kairi_relative_index_batch_inner_into(
    source: &[f64],
    volume: &[f64],
    combos: &[KairiRelativeIndexParams],
    _kernel: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<(), KairiRelativeIndexError> {
    if source.is_empty() {
        return Err(KairiRelativeIndexError::EmptyInputData);
    }
    if source.len() != volume.len() {
        return Err(KairiRelativeIndexError::InputLengthMismatch {
            source_len: source.len(),
            volume_len: volume.len(),
        });
    }
    let expected_len = combos.len() * source.len();
    if out.len() != expected_len {
        return Err(KairiRelativeIndexError::MismatchedOutputLen {
            dst_len: out.len(),
            expected_len,
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
    if parallel {
        out.par_chunks_mut(source.len())
            .zip(combos.par_iter())
            .try_for_each(|(row, params)| {
                let input = KairiRelativeIndexInput::from_slices(source, volume, params.clone());
                kairi_relative_index_into_slice(row, &input, Kernel::Auto)
            })?;
        return Ok(());
    }

    for (row, params) in out.chunks_mut(source.len()).zip(combos.iter()) {
        let input = KairiRelativeIndexInput::from_slices(source, volume, params.clone());
        kairi_relative_index_into_slice(row, &input, Kernel::Auto)?;
    }

    Ok(())
}

#[cfg(feature = "python")]
#[pyfunction(name = "kairi_relative_index")]
#[pyo3(signature = (source, volume, length=50, ma_type="SMA", kernel=None))]
pub fn kairi_relative_index_py<'py>(
    py: Python<'py>,
    source: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    length: usize,
    ma_type: &str,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let source = source.as_slice()?;
    let volume = volume.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let input = KairiRelativeIndexInput::from_slices(
        source,
        volume,
        KairiRelativeIndexParams {
            length: Some(length),
            ma_type: Some(ma_type.to_string()),
        },
    );
    let out = py
        .allow_threads(|| kairi_relative_index_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(out.values.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "KairiRelativeIndexStream")]
pub struct KairiRelativeIndexStreamPy {
    stream: KairiRelativeIndexStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl KairiRelativeIndexStreamPy {
    #[new]
    #[pyo3(signature = (length=50, ma_type="SMA"))]
    fn new(length: usize, ma_type: &str) -> PyResult<Self> {
        let stream = KairiRelativeIndexStream::try_new(KairiRelativeIndexParams {
            length: Some(length),
            ma_type: Some(ma_type.to_string()),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, source: f64, volume: f64) -> Option<f64> {
        self.stream.update(source, volume)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "kairi_relative_index_batch")]
#[pyo3(signature = (source, volume, length_range=(50,50,0), ma_type="SMA", kernel=None))]
pub fn kairi_relative_index_batch_py<'py>(
    py: Python<'py>,
    source: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    ma_type: &str,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let source = source.as_slice()?;
    let volume = volume.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let output = py
        .allow_threads(|| {
            kairi_relative_index_batch_with_kernel(
                source,
                volume,
                &KairiRelativeIndexBatchRange {
                    length: length_range,
                    ma_type: ma_type.to_string(),
                },
                kern,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let rows = output.rows;
    let cols = output.cols;
    let dict = PyDict::new(py);
    dict.set_item(
        "values",
        output.values.into_pyarray(py).reshape((rows, cols))?,
    )?;
    dict.set_item(
        "lengths",
        output
            .combos
            .iter()
            .map(|params| params.length.unwrap_or(50) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    dict.set_item("ma_type", ma_type)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_kairi_relative_index_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(kairi_relative_index_py, m)?)?;
    m.add_function(wrap_pyfunction!(kairi_relative_index_batch_py, m)?)?;
    m.add_class::<KairiRelativeIndexStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KairiRelativeIndexBatchConfig {
    pub length_range: Vec<usize>,
    pub ma_type: Option<String>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = kairi_relative_index_js)]
pub fn kairi_relative_index_js(
    source: &[f64],
    volume: &[f64],
    length: usize,
    ma_type: &str,
) -> Result<JsValue, JsValue> {
    let input = KairiRelativeIndexInput::from_slices(
        source,
        volume,
        KairiRelativeIndexParams {
            length: Some(length),
            ma_type: Some(ma_type.to_string()),
        },
    );
    let out = kairi_relative_index_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&out.values).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = kairi_relative_index_batch_js)]
pub fn kairi_relative_index_batch_js(
    source: &[f64],
    volume: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: KairiRelativeIndexBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.length_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: length_range must have exactly 3 elements [start, end, step]",
        ));
    }

    let out = kairi_relative_index_batch_with_kernel(
        source,
        volume,
        &KairiRelativeIndexBatchRange {
            length: (
                config.length_range[0],
                config.length_range[1],
                config.length_range[2],
            ),
            ma_type: config.ma_type.unwrap_or_else(|| "SMA".to_string()),
        },
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("values"),
        &serde_wasm_bindgen::to_value(&out.values).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("rows"),
        &JsValue::from_f64(out.rows as f64),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("cols"),
        &JsValue::from_f64(out.cols as f64),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("combos"),
        &serde_wasm_bindgen::to_value(&out.combos).unwrap(),
    )?;
    Ok(obj.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn kairi_relative_index_alloc(len: usize) -> *mut f64 {
    let mut buf = Vec::<f64>::with_capacity(len);
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn kairi_relative_index_free(ptr: *mut f64, len: usize) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let _ = Vec::from_raw_parts(ptr, 0, len);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn kairi_relative_index_into(
    source_ptr: *const f64,
    volume_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length: usize,
    ma_type: &str,
) -> Result<(), JsValue> {
    if source_ptr.is_null() || volume_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer"));
    }
    let source = unsafe { std::slice::from_raw_parts(source_ptr, len) };
    let volume = unsafe { std::slice::from_raw_parts(volume_ptr, len) };
    let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, len) };
    let input = KairiRelativeIndexInput::from_slices(
        source,
        volume,
        KairiRelativeIndexParams {
            length: Some(length),
            ma_type: Some(ma_type.to_string()),
        },
    );
    kairi_relative_index_into_slice(out, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn kairi_relative_index_batch_into(
    source_ptr: *const f64,
    volume_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    start: usize,
    end: usize,
    step: usize,
    ma_type: &str,
) -> Result<usize, JsValue> {
    if source_ptr.is_null() || volume_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer"));
    }
    let source = unsafe { std::slice::from_raw_parts(source_ptr, len) };
    let volume = unsafe { std::slice::from_raw_parts(volume_ptr, len) };
    let sweep = KairiRelativeIndexBatchRange {
        length: (start, end, step),
        ma_type: ma_type.to_string(),
    };
    let combos =
        expand_grid_kairi_relative_index(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, rows * len) };
    kairi_relative_index_batch_inner_into(source, volume, &combos, Kernel::Auto, false, out)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn kairi_relative_index_output_into_js(
    source: &[f64],
    volume: &[f64],
    length: usize,
    ma_type: &str,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = kairi_relative_index_js(source, volume, length, ma_type)?;
    crate::write_wasm_object_f64_outputs("kairi_relative_index_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn kairi_relative_index_batch_output_into_js(
    source: &[f64],
    volume: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = kairi_relative_index_batch_js(source, volume, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "kairi_relative_index_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::dispatch::{
        compute_cpu, IndicatorComputeRequest, IndicatorDataRef, ParamKV, ParamValue,
    };

    fn sample_source_volume(len: usize) -> (Vec<f64>, Vec<f64>) {
        let mut source = Vec::with_capacity(len);
        let mut volume = Vec::with_capacity(len);
        for i in 0..len {
            source.push(100.0 + (i as f64 * 0.35) + ((i % 7) as f64 - 3.0) * 0.4);
            volume.push(1000.0 + ((i * 13) % 17) as f64 * 15.0);
        }
        (source, volume)
    }

    fn manual_sma(series: &[f64], length: usize) -> Vec<f64> {
        let mut out = vec![f64::NAN; series.len()];
        let mut sum = 0.0;
        for i in 0..series.len() {
            sum += series[i];
            if i >= length {
                sum -= series[i - length];
            }
            if i + 1 >= length {
                let ma = sum / length as f64;
                out[i] = (series[i] - ma) * 100.0 / ma;
            }
        }
        out
    }

    fn manual_vwma(series: &[f64], volume: &[f64], length: usize) -> Vec<f64> {
        let mut out = vec![f64::NAN; series.len()];
        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..series.len() {
            num += series[i] * volume[i];
            den += volume[i];
            if i >= length {
                num -= series[i - length] * volume[i - length];
                den -= volume[i - length];
            }
            if i + 1 >= length {
                let ma = num / den;
                out[i] = (series[i] - ma) * 100.0 / ma;
            }
        }
        out
    }

    #[test]
    fn kairi_relative_index_sma_matches_manual() -> Result<(), Box<dyn Error>> {
        let (source, volume) = sample_source_volume(128);
        let input = KairiRelativeIndexInput::from_slices(
            &source,
            &volume,
            KairiRelativeIndexParams {
                length: Some(10),
                ma_type: Some("SMA".to_string()),
            },
        );
        let out = kairi_relative_index(&input)?;
        let expected = manual_sma(&source, 10);
        for (a, b) in out.values.iter().zip(expected.iter()) {
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan());
            } else {
                assert!((a - b).abs() < 1e-12);
            }
        }
        Ok(())
    }

    #[test]
    fn kairi_relative_index_vwma_matches_manual() -> Result<(), Box<dyn Error>> {
        let source = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let volume = vec![1.0, 5.0, 1.0, 5.0, 1.0, 5.0];
        let input = KairiRelativeIndexInput::from_slices(
            &source,
            &volume,
            KairiRelativeIndexParams {
                length: Some(3),
                ma_type: Some("VWMA".to_string()),
            },
        );
        let out = kairi_relative_index(&input)?;
        let expected = manual_vwma(&source, &volume, 3);
        for (a, b) in out.values.iter().zip(expected.iter()) {
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan());
            } else {
                assert!((a - b).abs() < 1e-12);
            }
        }
        Ok(())
    }

    #[test]
    fn kairi_relative_index_into_matches_api() -> Result<(), Box<dyn Error>> {
        let (source, volume) = sample_source_volume(200);
        let input = KairiRelativeIndexInput::from_slices(
            &source,
            &volume,
            KairiRelativeIndexParams {
                length: Some(20),
                ma_type: Some("EMA".to_string()),
            },
        );
        let base = kairi_relative_index(&input)?;
        let mut out = vec![0.0; source.len()];
        kairi_relative_index_into_slice(&mut out, &input, Kernel::Auto)?;
        for (a, b) in out.iter().zip(base.values.iter()) {
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan());
            } else {
                assert!((a - b).abs() < 1e-12);
            }
        }
        Ok(())
    }

    #[test]
    fn kairi_relative_index_stream_matches_batch() -> Result<(), Box<dyn Error>> {
        let (source, volume) = sample_source_volume(220);
        let input = KairiRelativeIndexInput::from_slices(
            &source,
            &volume,
            KairiRelativeIndexParams {
                length: Some(14),
                ma_type: Some("TMA".to_string()),
            },
        );
        let batch = kairi_relative_index(&input)?;
        let mut stream = KairiRelativeIndexStream::try_new(KairiRelativeIndexParams {
            length: Some(14),
            ma_type: Some("TMA".to_string()),
        })?;
        let mut values = Vec::with_capacity(source.len());
        for (&s, &v) in source.iter().zip(volume.iter()) {
            values.push(stream.update(s, v).unwrap_or(f64::NAN));
        }
        for (a, b) in values.iter().zip(batch.values.iter()) {
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan());
            } else {
                assert!((a - b).abs() < 1e-12);
            }
        }
        Ok(())
    }

    #[test]
    fn kairi_relative_index_batch_single_matches_single() -> Result<(), Box<dyn Error>> {
        let (source, volume) = sample_source_volume(128);
        let single = kairi_relative_index(&KairiRelativeIndexInput::from_slices(
            &source,
            &volume,
            KairiRelativeIndexParams {
                length: Some(25),
                ma_type: Some("HMA".to_string()),
            },
        ))?;
        let batch = kairi_relative_index_batch_with_kernel(
            &source,
            &volume,
            &KairiRelativeIndexBatchRange {
                length: (25, 25, 0),
                ma_type: "HMA".to_string(),
            },
            Kernel::Auto,
        )?;
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, source.len());
        for (a, b) in batch.values.iter().zip(single.values.iter()) {
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan());
            } else {
                assert!((a - b).abs() < 1e-12);
            }
        }
        Ok(())
    }

    #[test]
    fn kairi_relative_index_rejects_invalid_params() {
        let (source, volume) = sample_source_volume(32);
        let err = kairi_relative_index(&KairiRelativeIndexInput::from_slices(
            &source,
            &volume,
            KairiRelativeIndexParams {
                length: Some(1),
                ma_type: Some("SMA".to_string()),
            },
        ))
        .unwrap_err();
        assert!(matches!(err, KairiRelativeIndexError::InvalidLength { .. }));
    }

    #[test]
    fn kairi_relative_index_dispatch_compute_returns_value() -> Result<(), Box<dyn Error>> {
        let (source, volume) = sample_source_volume(128);
        let params = [
            ParamKV {
                key: "length",
                value: ParamValue::Int(20),
            },
            ParamKV {
                key: "ma_type",
                value: ParamValue::EnumString("VWMA"),
            },
        ];
        let out = compute_cpu(IndicatorComputeRequest {
            indicator_id: "kairi_relative_index",
            output_id: Some("value"),
            data: IndicatorDataRef::CloseVolume {
                close: &source,
                volume: &volume,
            },
            params: &params,
            kernel: Kernel::Auto,
        })?;
        assert_eq!(out.output_id, "value");
        assert_eq!(out.cols, source.len());
        Ok(())
    }
}
