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
    alloc_uninit_f64, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::mem::ManuallyDrop;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum DemandIndexData<'a> {
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
pub struct DemandIndexOutput {
    pub demand_index: Vec<f64>,
    pub signal: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct DemandIndexParams {
    pub len_bs: Option<usize>,
    pub len_bs_ma: Option<usize>,
    pub len_di_ma: Option<usize>,
    pub ma_type: Option<String>,
}

impl Default for DemandIndexParams {
    fn default() -> Self {
        Self {
            len_bs: Some(19),
            len_bs_ma: Some(19),
            len_di_ma: Some(19),
            ma_type: Some("ema".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DemandIndexInput<'a> {
    pub data: DemandIndexData<'a>,
    pub params: DemandIndexParams,
}

impl<'a> DemandIndexInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: DemandIndexParams) -> Self {
        Self {
            data: DemandIndexData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
        params: DemandIndexParams,
    ) -> Self {
        Self {
            data: DemandIndexData::Slices {
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
        Self::from_candles(candles, DemandIndexParams::default())
    }

    #[inline]
    pub fn get_len_bs(&self) -> usize {
        self.params.len_bs.unwrap_or(19)
    }

    #[inline]
    pub fn get_len_bs_ma(&self) -> usize {
        self.params.len_bs_ma.unwrap_or(19)
    }

    #[inline]
    pub fn get_len_di_ma(&self) -> usize {
        self.params.len_di_ma.unwrap_or(19)
    }

    #[inline]
    pub fn get_ma_type(&self) -> &str {
        self.params.ma_type.as_deref().unwrap_or("ema")
    }
}

#[derive(Copy, Clone, Debug)]
pub struct DemandIndexBuilder {
    len_bs: Option<usize>,
    len_bs_ma: Option<usize>,
    len_di_ma: Option<usize>,
    ma_type: Option<MaType>,
    kernel: Kernel,
}

impl Default for DemandIndexBuilder {
    fn default() -> Self {
        Self {
            len_bs: None,
            len_bs_ma: None,
            len_di_ma: None,
            ma_type: None,
            kernel: Kernel::Auto,
        }
    }
}

impl DemandIndexBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn len_bs(mut self, len_bs: usize) -> Self {
        self.len_bs = Some(len_bs);
        self
    }

    #[inline]
    pub fn len_bs_ma(mut self, len_bs_ma: usize) -> Self {
        self.len_bs_ma = Some(len_bs_ma);
        self
    }

    #[inline]
    pub fn len_di_ma(mut self, len_di_ma: usize) -> Self {
        self.len_di_ma = Some(len_di_ma);
        self
    }

    #[inline]
    pub fn ma_type(mut self, ma_type: &str) -> Result<Self, DemandIndexError> {
        self.ma_type = Some(MaType::parse(ma_type)?);
        Ok(self)
    }

    #[inline]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline]
    pub fn apply(self, candles: &Candles) -> Result<DemandIndexOutput, DemandIndexError> {
        let input = DemandIndexInput::from_candles(
            candles,
            DemandIndexParams {
                len_bs: self.len_bs,
                len_bs_ma: self.len_bs_ma,
                len_di_ma: self.len_di_ma,
                ma_type: Some(self.ma_type.unwrap_or(MaType::Ema).as_str().to_string()),
            },
        );
        demand_index_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Result<DemandIndexOutput, DemandIndexError> {
        let input = DemandIndexInput::from_slices(
            high,
            low,
            close,
            volume,
            DemandIndexParams {
                len_bs: self.len_bs,
                len_bs_ma: self.len_bs_ma,
                len_di_ma: self.len_di_ma,
                ma_type: Some(self.ma_type.unwrap_or(MaType::Ema).as_str().to_string()),
            },
        );
        demand_index_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn into_stream(self) -> Result<DemandIndexStream, DemandIndexError> {
        DemandIndexStream::try_new(DemandIndexParams {
            len_bs: self.len_bs,
            len_bs_ma: self.len_bs_ma,
            len_di_ma: self.len_di_ma,
            ma_type: Some(self.ma_type.unwrap_or(MaType::Ema).as_str().to_string()),
        })
    }
}

#[derive(Debug, Error)]
pub enum DemandIndexError {
    #[error("demand_index: Input data slice is empty.")]
    EmptyInputData,
    #[error("demand_index: All values are NaN.")]
    AllValuesNaN,
    #[error("demand_index: Invalid len_bs: len_bs = {len_bs}, data length = {data_len}")]
    InvalidLenBs { len_bs: usize, data_len: usize },
    #[error("demand_index: Invalid len_bs_ma: len_bs_ma = {len_bs_ma}, data length = {data_len}")]
    InvalidLenBsMa { len_bs_ma: usize, data_len: usize },
    #[error("demand_index: Invalid len_di_ma: len_di_ma = {len_di_ma}, data length = {data_len}")]
    InvalidLenDiMa { len_di_ma: usize, data_len: usize },
    #[error("demand_index: Invalid ma_type: {ma_type}")]
    InvalidMaType { ma_type: String },
    #[error(
        "demand_index: Inconsistent slice lengths: high={high_len}, low={low_len}, close={close_len}, volume={volume_len}"
    )]
    InconsistentSliceLengths {
        high_len: usize,
        low_len: usize,
        close_len: usize,
        volume_len: usize,
    },
    #[error("demand_index: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "demand_index: Output length mismatch: expected = {expected}, demand_index = {demand_index_got}, signal = {signal_got}"
    )]
    OutputLengthMismatch {
        expected: usize,
        demand_index_got: usize,
        signal_got: usize,
    },
    #[error("demand_index: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("demand_index: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum MaType {
    Ema,
    Sma,
    Wma,
    Rma,
}

impl MaType {
    #[inline]
    fn parse(value: &str) -> Result<Self, DemandIndexError> {
        if value.eq_ignore_ascii_case("ema") {
            return Ok(Self::Ema);
        }
        if value.eq_ignore_ascii_case("sma") {
            return Ok(Self::Sma);
        }
        if value.eq_ignore_ascii_case("wma") {
            return Ok(Self::Wma);
        }
        if value.eq_ignore_ascii_case("rma") || value.eq_ignore_ascii_case("wilders") {
            return Ok(Self::Rma);
        }
        Err(DemandIndexError::InvalidMaType {
            ma_type: value.to_string(),
        })
    }

    #[inline]
    fn as_str(self) -> &'static str {
        match self {
            Self::Ema => "ema",
            Self::Sma => "sma",
            Self::Wma => "wma",
            Self::Rma => "rma",
        }
    }
}

#[derive(Debug, Clone)]
struct SmaState {
    period: usize,
    values: Vec<f64>,
    valid: Vec<u8>,
    idx: usize,
    count: usize,
    valid_count: usize,
    sum: f64,
}

impl SmaState {
    #[inline]
    fn new(period: usize) -> Self {
        Self {
            period,
            values: vec![0.0; period],
            valid: vec![0u8; period],
            idx: 0,
            count: 0,
            valid_count: 0,
            sum: 0.0,
        }
    }

    #[inline]
    fn update(&mut self, value: Option<f64>) -> Option<f64> {
        if self.count >= self.period {
            let old_idx = self.idx;
            if self.valid[old_idx] != 0 {
                self.valid_count = self.valid_count.saturating_sub(1);
                self.sum -= self.values[old_idx];
            }
        } else {
            self.count += 1;
        }

        match value {
            Some(v) if v.is_finite() => {
                self.values[self.idx] = v;
                self.valid[self.idx] = 1;
                self.valid_count += 1;
                self.sum += v;
            }
            _ => {
                self.values[self.idx] = 0.0;
                self.valid[self.idx] = 0;
            }
        }

        self.idx += 1;
        if self.idx == self.period {
            self.idx = 0;
        }

        if self.count < self.period {
            None
        } else if self.valid_count == self.period {
            Some(self.sum / self.period as f64)
        } else {
            Some(f64::NAN)
        }
    }
}

#[derive(Debug, Clone)]
struct WmaState {
    period: usize,
    denom: f64,
    values: Vec<f64>,
    valid: Vec<u8>,
    idx: usize,
    count: usize,
    valid_count: usize,
}

impl WmaState {
    #[inline]
    fn new(period: usize) -> Self {
        Self {
            period,
            denom: (period * (period + 1) / 2) as f64,
            values: vec![0.0; period],
            valid: vec![0u8; period],
            idx: 0,
            count: 0,
            valid_count: 0,
        }
    }

    #[inline]
    fn update(&mut self, value: Option<f64>) -> Option<f64> {
        if self.count >= self.period {
            let old_idx = self.idx;
            if self.valid[old_idx] != 0 {
                self.valid_count = self.valid_count.saturating_sub(1);
            }
        } else {
            self.count += 1;
        }

        match value {
            Some(v) if v.is_finite() => {
                self.values[self.idx] = v;
                self.valid[self.idx] = 1;
                self.valid_count += 1;
            }
            _ => {
                self.values[self.idx] = 0.0;
                self.valid[self.idx] = 0;
            }
        }

        self.idx += 1;
        if self.idx == self.period {
            self.idx = 0;
        }

        if self.count < self.period {
            return None;
        }
        if self.valid_count != self.period {
            return Some(f64::NAN);
        }

        let mut weighted = 0.0;
        let mut weight = 1.0;
        let mut pos = self.idx;
        for _ in 0..self.period {
            weighted += self.values[pos] * weight;
            weight += 1.0;
            pos += 1;
            if pos == self.period {
                pos = 0;
            }
        }
        Some(weighted / self.denom)
    }
}

#[derive(Debug, Clone)]
struct ExpState {
    alpha: f64,
    value: f64,
    initialized: bool,
}

impl ExpState {
    #[inline]
    fn ema(period: usize) -> Self {
        Self {
            alpha: 2.0 / (period as f64 + 1.0),
            value: f64::NAN,
            initialized: false,
        }
    }

    #[inline]
    fn rma(period: usize) -> Self {
        Self {
            alpha: 1.0 / period as f64,
            value: f64::NAN,
            initialized: false,
        }
    }

    #[inline]
    fn update(&mut self, value: Option<f64>) -> Option<f64> {
        match value {
            Some(v) if v.is_finite() => {
                if self.initialized {
                    self.value += self.alpha * (v - self.value);
                } else {
                    self.value = v;
                    self.initialized = true;
                }
                Some(self.value)
            }
            _ => {
                self.value = f64::NAN;
                self.initialized = false;
                Some(f64::NAN)
            }
        }
    }
}

#[derive(Debug, Clone)]
enum MaState {
    Sma(SmaState),
    Wma(WmaState),
    Ema(ExpState),
    Rma(ExpState),
}

impl MaState {
    #[inline]
    fn new(kind: MaType, period: usize) -> Self {
        match kind {
            MaType::Sma => Self::Sma(SmaState::new(period)),
            MaType::Wma => Self::Wma(WmaState::new(period)),
            MaType::Ema => Self::Ema(ExpState::ema(period)),
            MaType::Rma => Self::Rma(ExpState::rma(period)),
        }
    }

    #[inline]
    fn update(&mut self, value: Option<f64>) -> Option<f64> {
        match self {
            Self::Sma(state) => state.update(value),
            Self::Wma(state) => state.update(value),
            Self::Ema(state) => state.update(value),
            Self::Rma(state) => state.update(value),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DemandIndexStream {
    len_bs: usize,
    len_bs_ma: usize,
    len_di_ma: usize,
    ma_type: MaType,
    h0: f64,
    l0: f64,
    has_h0_l0: bool,
    prev_p: f64,
    has_prev_p: bool,
    volume_avg: MaState,
    bp_avg: MaState,
    sp_avg: MaState,
    signal_avg: SmaState,
}

impl DemandIndexStream {
    pub fn try_new(params: DemandIndexParams) -> Result<Self, DemandIndexError> {
        let len_bs = params.len_bs.unwrap_or(19);
        let len_bs_ma = params.len_bs_ma.unwrap_or(19);
        let len_di_ma = params.len_di_ma.unwrap_or(19);
        let ma_type = MaType::parse(params.ma_type.as_deref().unwrap_or("ema"))?;
        validate_lengths(len_bs, len_bs_ma, len_di_ma, 0)?;

        Ok(Self {
            len_bs,
            len_bs_ma,
            len_di_ma,
            ma_type,
            h0: f64::NAN,
            l0: f64::NAN,
            has_h0_l0: false,
            prev_p: f64::NAN,
            has_prev_p: false,
            volume_avg: MaState::new(ma_type, len_bs),
            bp_avg: MaState::new(ma_type, len_bs_ma),
            sp_avg: MaState::new(ma_type, len_bs_ma),
            signal_avg: SmaState::new(len_di_ma),
        })
    }

    #[inline]
    pub fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> Option<(f64, f64)> {
        if !valid_ohlcv_bar(high, low, close, volume) {
            self.volume_avg.update(None);
            let di = demand_index_finalize(None, None, &mut self.bp_avg, &mut self.sp_avg);
            let signal = update_signal(&mut self.signal_avg, di);
            self.has_prev_p = false;
            return if di.is_none() && signal.is_none() {
                None
            } else {
                Some((di.unwrap_or(f64::NAN), signal.unwrap_or(f64::NAN)))
            };
        }

        if !self.has_h0_l0 {
            self.h0 = high;
            self.l0 = low;
            self.has_h0_l0 = true;
        }

        let volume_avg = self.volume_avg.update(Some(volume));
        let p = high + low + 2.0 * close;

        if !self.has_prev_p {
            let di = demand_index_finalize(None, None, &mut self.bp_avg, &mut self.sp_avg);
            let signal = update_signal(&mut self.signal_avg, di);
            self.prev_p = p;
            self.has_prev_p = true;
            return if di.is_none() && signal.is_none() {
                None
            } else {
                Some((di.unwrap_or(f64::NAN), signal.unwrap_or(f64::NAN)))
            };
        }

        let bp_sp = compute_bp_sp(volume_avg, volume, p, self.prev_p, self.h0, self.l0);
        let di = demand_index_finalize(
            bp_sp.map(|x| x.0),
            bp_sp.map(|x| x.1),
            &mut self.bp_avg,
            &mut self.sp_avg,
        );
        let signal = update_signal(&mut self.signal_avg, di);

        self.prev_p = p;
        self.has_prev_p = true;
        Some((di.unwrap_or(f64::NAN), signal.unwrap_or(f64::NAN)))
    }

    #[inline]
    pub fn get_warmup_period(&self) -> usize {
        signal_warmup(self.ma_type, self.len_bs, self.len_bs_ma, self.len_di_ma)
    }
}

#[inline]
pub fn demand_index(input: &DemandIndexInput) -> Result<DemandIndexOutput, DemandIndexError> {
    demand_index_with_kernel(input, Kernel::Auto)
}

#[inline]
fn validate_lengths(
    len_bs: usize,
    len_bs_ma: usize,
    len_di_ma: usize,
    data_len: usize,
) -> Result<(), DemandIndexError> {
    if len_bs == 0 {
        return Err(DemandIndexError::InvalidLenBs { len_bs, data_len });
    }
    if len_bs_ma == 0 {
        return Err(DemandIndexError::InvalidLenBsMa {
            len_bs_ma,
            data_len,
        });
    }
    if len_di_ma == 0 {
        return Err(DemandIndexError::InvalidLenDiMa {
            len_di_ma,
            data_len,
        });
    }
    Ok(())
}

#[inline]
fn valid_ohlcv_bar(high: f64, low: f64, close: f64, volume: f64) -> bool {
    high.is_finite() && low.is_finite() && close.is_finite() && volume.is_finite()
}

#[inline]
fn extract_input<'a>(
    input: &'a DemandIndexInput<'a>,
) -> Result<(&'a [f64], &'a [f64], &'a [f64], &'a [f64]), DemandIndexError> {
    let (high, low, close, volume) = match &input.data {
        DemandIndexData::Candles { candles } => (
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
            candles.volume.as_slice(),
        ),
        DemandIndexData::Slices {
            high,
            low,
            close,
            volume,
        } => (*high, *low, *close, *volume),
    };

    if high.is_empty() || low.is_empty() || close.is_empty() || volume.is_empty() {
        return Err(DemandIndexError::EmptyInputData);
    }
    if high.len() != low.len() || high.len() != close.len() || high.len() != volume.len() {
        return Err(DemandIndexError::InconsistentSliceLengths {
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
            volume_len: volume.len(),
        });
    }
    Ok((high, low, close, volume))
}

#[inline]
fn first_valid_ohlcv(high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Option<usize> {
    let mut i = 0usize;
    while i < high.len() {
        if valid_ohlcv_bar(high[i], low[i], close[i], volume[i]) {
            return Some(i);
        }
        i += 1;
    }
    None
}

#[inline]
fn count_valid_ohlcv(high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> usize {
    let mut count = 0usize;
    for i in 0..high.len() {
        if valid_ohlcv_bar(high[i], low[i], close[i], volume[i]) {
            count += 1;
        }
    }
    count
}

#[inline]
fn scan_valid_ohlcv(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
) -> Option<(usize, usize)> {
    let mut first = usize::MAX;
    let mut count = 0usize;
    for i in 0..high.len() {
        if valid_ohlcv_bar(high[i], low[i], close[i], volume[i]) {
            if count == 0 {
                first = i;
            }
            count += 1;
        }
    }
    if count == 0 {
        None
    } else {
        Some((first, count))
    }
}

#[inline]
fn di_warmup(ma_type: MaType, len_bs: usize, len_bs_ma: usize) -> usize {
    match ma_type {
        MaType::Ema | MaType::Rma => 1,
        MaType::Sma | MaType::Wma => len_bs.saturating_sub(1).max(1) + len_bs_ma.saturating_sub(1),
    }
}

#[inline]
fn signal_warmup(ma_type: MaType, len_bs: usize, len_bs_ma: usize, len_di_ma: usize) -> usize {
    di_warmup(ma_type, len_bs, len_bs_ma) + len_di_ma.saturating_sub(1)
}

#[inline]
fn needed_valid_bars(ma_type: MaType, len_bs: usize, len_bs_ma: usize, len_di_ma: usize) -> usize {
    signal_warmup(ma_type, len_bs, len_bs_ma, len_di_ma) + 1
}

#[inline]
fn normalize_h0_l0(h0: f64, l0: f64) -> f64 {
    (h0 - l0).abs().max(1e-12)
}

#[inline]
fn compute_bp_sp(
    volume_avg: Option<f64>,
    volume: f64,
    p: f64,
    prev_p: f64,
    h0: f64,
    l0: f64,
) -> Option<(f64, f64)> {
    let v_avg = volume_avg?;
    if !v_avg.is_finite() || !p.is_finite() || !prev_p.is_finite() {
        return None;
    }

    let v_ratio = if v_avg == 0.0 { 1.0 } else { volume / v_avg };
    if !v_ratio.is_finite() {
        return None;
    }

    let denom = normalize_h0_l0(h0, l0);
    let k = 0.375;

    if p < prev_p {
        if p == 0.0 {
            return None;
        }
        let exponent = (k * (p + prev_p) / denom) * ((prev_p - p) / p);
        return Some((v_ratio / exponent.exp(), v_ratio));
    }
    if p > prev_p {
        if prev_p == 0.0 {
            return None;
        }
        let exponent = (k * (p + prev_p) / denom) * ((p - prev_p) / prev_p);
        return Some((v_ratio, v_ratio / exponent.exp()));
    }
    Some((v_ratio, v_ratio))
}

#[inline]
fn finalize_di(bp_ma: Option<f64>, sp_ma: Option<f64>) -> Option<f64> {
    let bp = bp_ma?;
    let sp = sp_ma?;
    if !bp.is_finite() || !sp.is_finite() {
        return None;
    }

    if bp > sp {
        if bp == 0.0 {
            return Some(100.0);
        }
        return Some(100.0 * (1.0 - sp / bp));
    }
    if bp < sp {
        if sp == 0.0 {
            return Some(-100.0);
        }
        return Some(100.0 * (bp / sp - 1.0));
    }
    Some(0.0)
}

#[inline]
fn demand_index_finalize(
    bp: Option<f64>,
    sp: Option<f64>,
    bp_avg: &mut MaState,
    sp_avg: &mut MaState,
) -> Option<f64> {
    let bp_ma = bp_avg.update(bp.filter(|v| v.is_finite()));
    let sp_ma = sp_avg.update(sp.filter(|v| v.is_finite()));
    finalize_di(bp_ma, sp_ma)
}

#[inline]
fn demand_index_finalize_exp(
    bp: Option<f64>,
    sp: Option<f64>,
    bp_avg: &mut ExpState,
    sp_avg: &mut ExpState,
) -> Option<f64> {
    let bp_ma = bp_avg.update(bp.filter(|v| v.is_finite()));
    let sp_ma = sp_avg.update(sp.filter(|v| v.is_finite()));
    finalize_di(bp_ma, sp_ma)
}

#[inline]
fn update_signal(signal_avg: &mut SmaState, di: Option<f64>) -> Option<f64> {
    signal_avg.update(di.filter(|v| v.is_finite()))
}

#[inline]
fn demand_index_compute_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    len_bs: usize,
    len_bs_ma: usize,
    len_di_ma: usize,
    ma_type: MaType,
    demand_index_out: &mut [f64],
    signal_out: &mut [f64],
) {
    if ma_type == MaType::Ema {
        demand_index_compute_ema_into(
            high,
            low,
            close,
            volume,
            len_bs,
            len_bs_ma,
            len_di_ma,
            demand_index_out,
            signal_out,
        );
        return;
    }

    let mut volume_avg = MaState::new(ma_type, len_bs);
    let mut bp_avg = MaState::new(ma_type, len_bs_ma);
    let mut sp_avg = MaState::new(ma_type, len_bs_ma);
    let mut signal_avg = SmaState::new(len_di_ma);
    let mut h0 = f64::NAN;
    let mut l0 = f64::NAN;
    let mut has_h0_l0 = false;
    let mut prev_p = f64::NAN;
    let mut has_prev_p = false;

    for i in 0..high.len() {
        let h = high[i];
        let l = low[i];
        let c = close[i];
        let v = volume[i];

        if !valid_ohlcv_bar(h, l, c, v) {
            volume_avg.update(None);
            let di = demand_index_finalize(None, None, &mut bp_avg, &mut sp_avg);
            let signal = update_signal(&mut signal_avg, di);
            demand_index_out[i] = di.unwrap_or(f64::NAN);
            signal_out[i] = signal.unwrap_or(f64::NAN);
            has_prev_p = false;
            continue;
        }

        if !has_h0_l0 {
            h0 = h;
            l0 = l;
            has_h0_l0 = true;
        }

        let volume_avg_now = volume_avg.update(Some(v));
        let p = h + l + 2.0 * c;

        if !has_prev_p {
            let di = demand_index_finalize(None, None, &mut bp_avg, &mut sp_avg);
            let signal = update_signal(&mut signal_avg, di);
            demand_index_out[i] = di.unwrap_or(f64::NAN);
            signal_out[i] = signal.unwrap_or(f64::NAN);
            prev_p = p;
            has_prev_p = true;
            continue;
        }

        let bp_sp = compute_bp_sp(volume_avg_now, v, p, prev_p, h0, l0);
        let di = demand_index_finalize(
            bp_sp.map(|x| x.0),
            bp_sp.map(|x| x.1),
            &mut bp_avg,
            &mut sp_avg,
        );
        let signal = update_signal(&mut signal_avg, di);
        demand_index_out[i] = di.unwrap_or(f64::NAN);
        signal_out[i] = signal.unwrap_or(f64::NAN);
        prev_p = p;
        has_prev_p = true;
    }
}

#[inline]
fn demand_index_compute_ema_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    len_bs: usize,
    len_bs_ma: usize,
    len_di_ma: usize,
    demand_index_out: &mut [f64],
    signal_out: &mut [f64],
) {
    let mut volume_avg = ExpState::ema(len_bs);
    let mut bp_avg = ExpState::ema(len_bs_ma);
    let mut sp_avg = ExpState::ema(len_bs_ma);
    let mut signal_avg = SmaState::new(len_di_ma);
    let mut h0 = f64::NAN;
    let mut l0 = f64::NAN;
    let mut has_h0_l0 = false;
    let mut prev_p = f64::NAN;
    let mut has_prev_p = false;

    for i in 0..high.len() {
        let h = high[i];
        let l = low[i];
        let c = close[i];
        let v = volume[i];

        if !valid_ohlcv_bar(h, l, c, v) {
            volume_avg.update(None);
            let di = demand_index_finalize_exp(None, None, &mut bp_avg, &mut sp_avg);
            let signal = update_signal(&mut signal_avg, di);
            demand_index_out[i] = di.unwrap_or(f64::NAN);
            signal_out[i] = signal.unwrap_or(f64::NAN);
            has_prev_p = false;
            continue;
        }

        if !has_h0_l0 {
            h0 = h;
            l0 = l;
            has_h0_l0 = true;
        }

        let volume_avg_now = volume_avg.update(Some(v));
        let p = h + l + 2.0 * c;

        if !has_prev_p {
            let di = demand_index_finalize_exp(None, None, &mut bp_avg, &mut sp_avg);
            let signal = update_signal(&mut signal_avg, di);
            demand_index_out[i] = di.unwrap_or(f64::NAN);
            signal_out[i] = signal.unwrap_or(f64::NAN);
            prev_p = p;
            has_prev_p = true;
            continue;
        }

        let bp_sp = compute_bp_sp(volume_avg_now, v, p, prev_p, h0, l0);
        let (bp, sp) = match bp_sp {
            Some((bp, sp)) => (Some(bp), Some(sp)),
            None => (None, None),
        };
        let di = demand_index_finalize_exp(bp, sp, &mut bp_avg, &mut sp_avg);
        let signal = update_signal(&mut signal_avg, di);
        demand_index_out[i] = di.unwrap_or(f64::NAN);
        signal_out[i] = signal.unwrap_or(f64::NAN);
        prev_p = p;
        has_prev_p = true;
    }
}

#[inline]
pub fn demand_index_with_kernel(
    input: &DemandIndexInput,
    kernel: Kernel,
) -> Result<DemandIndexOutput, DemandIndexError> {
    let (high, low, close, volume) = extract_input(input)?;
    let len = high.len();
    let len_bs = input.get_len_bs();
    let len_bs_ma = input.get_len_bs_ma();
    let len_di_ma = input.get_len_di_ma();
    validate_lengths(len_bs, len_bs_ma, len_di_ma, len)?;
    let ma_type = MaType::parse(input.get_ma_type())?;

    if len_bs > len {
        return Err(DemandIndexError::InvalidLenBs {
            len_bs,
            data_len: len,
        });
    }
    if len_bs_ma > len {
        return Err(DemandIndexError::InvalidLenBsMa {
            len_bs_ma,
            data_len: len,
        });
    }
    if len_di_ma > len {
        return Err(DemandIndexError::InvalidLenDiMa {
            len_di_ma,
            data_len: len,
        });
    }

    let (_, valid) =
        scan_valid_ohlcv(high, low, close, volume).ok_or(DemandIndexError::AllValuesNaN)?;
    let needed = needed_valid_bars(ma_type, len_bs, len_bs_ma, len_di_ma);
    if valid < needed {
        return Err(DemandIndexError::NotEnoughValidData { needed, valid });
    }

    let _ = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other.to_non_batch(),
    };

    let mut demand_index_out = alloc_uninit_f64(len);
    let mut signal_out = alloc_uninit_f64(len);
    demand_index_compute_into(
        high,
        low,
        close,
        volume,
        len_bs,
        len_bs_ma,
        len_di_ma,
        ma_type,
        &mut demand_index_out,
        &mut signal_out,
    );

    Ok(DemandIndexOutput {
        demand_index: demand_index_out,
        signal: signal_out,
    })
}

#[inline]
pub fn demand_index_into_slices(
    demand_index_out: &mut [f64],
    signal_out: &mut [f64],
    input: &DemandIndexInput,
    kernel: Kernel,
) -> Result<(), DemandIndexError> {
    let (high, low, close, volume) = extract_input(input)?;
    let len = high.len();
    if demand_index_out.len() != len || signal_out.len() != len {
        return Err(DemandIndexError::OutputLengthMismatch {
            expected: len,
            demand_index_got: demand_index_out.len(),
            signal_got: signal_out.len(),
        });
    }

    let len_bs = input.get_len_bs();
    let len_bs_ma = input.get_len_bs_ma();
    let len_di_ma = input.get_len_di_ma();
    validate_lengths(len_bs, len_bs_ma, len_di_ma, len)?;
    let ma_type = MaType::parse(input.get_ma_type())?;
    let valid = count_valid_ohlcv(high, low, close, volume);
    let needed = needed_valid_bars(ma_type, len_bs, len_bs_ma, len_di_ma);
    if valid < needed {
        return Err(DemandIndexError::NotEnoughValidData { needed, valid });
    }

    let _ = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other.to_non_batch(),
    };

    demand_index_compute_into(
        high,
        low,
        close,
        volume,
        len_bs,
        len_bs_ma,
        len_di_ma,
        ma_type,
        demand_index_out,
        signal_out,
    );
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn demand_index_into(
    input: &DemandIndexInput,
    demand_index_out: &mut [f64],
    signal_out: &mut [f64],
) -> Result<(), DemandIndexError> {
    demand_index_into_slices(demand_index_out, signal_out, input, Kernel::Auto)
}

#[derive(Clone, Debug)]
pub struct DemandIndexBatchRange {
    pub len_bs: (usize, usize, usize),
    pub len_bs_ma: (usize, usize, usize),
    pub len_di_ma: (usize, usize, usize),
    pub ma_type: Option<String>,
}

impl Default for DemandIndexBatchRange {
    fn default() -> Self {
        Self {
            len_bs: (19, 19, 0),
            len_bs_ma: (19, 19, 0),
            len_di_ma: (19, 19, 0),
            ma_type: Some("ema".to_string()),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct DemandIndexBatchBuilder {
    range: DemandIndexBatchRange,
    kernel: Kernel,
}

impl DemandIndexBatchBuilder {
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
    pub fn len_bs_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.len_bs = (start, end, step);
        self
    }

    #[inline]
    pub fn len_bs_ma_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.len_bs_ma = (start, end, step);
        self
    }

    #[inline]
    pub fn len_di_ma_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.len_di_ma = (start, end, step);
        self
    }

    #[inline]
    pub fn ma_type(mut self, ma_type: &str) -> Result<Self, DemandIndexError> {
        self.range.ma_type = Some(MaType::parse(ma_type)?.as_str().to_string());
        Ok(self)
    }

    #[inline]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Result<DemandIndexBatchOutput, DemandIndexError> {
        demand_index_batch_with_kernel(high, low, close, volume, &self.range, self.kernel)
    }

    #[inline]
    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<DemandIndexBatchOutput, DemandIndexError> {
        self.apply_slices(&candles.high, &candles.low, &candles.close, &candles.volume)
    }
}

#[derive(Clone, Debug)]
pub struct DemandIndexBatchOutput {
    pub demand_index: Vec<f64>,
    pub signal: Vec<f64>,
    pub combos: Vec<DemandIndexParams>,
    pub rows: usize,
    pub cols: usize,
}

impl DemandIndexBatchOutput {
    pub fn row_for_params(&self, params: &DemandIndexParams) -> Option<usize> {
        self.combos.iter().position(|combo| {
            combo.len_bs.unwrap_or(19) == params.len_bs.unwrap_or(19)
                && combo.len_bs_ma.unwrap_or(19) == params.len_bs_ma.unwrap_or(19)
                && combo.len_di_ma.unwrap_or(19) == params.len_di_ma.unwrap_or(19)
                && combo
                    .ma_type
                    .as_deref()
                    .unwrap_or("ema")
                    .eq_ignore_ascii_case(params.ma_type.as_deref().unwrap_or("ema"))
        })
    }
}

#[inline]
fn expand_axis_usize(
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, DemandIndexError> {
    if step == 0 || start == end {
        return Ok(vec![start]);
    }

    let mut out = Vec::new();
    if start < end {
        let mut x = start;
        while x <= end {
            out.push(x);
            let next = x.saturating_add(step);
            if next == x {
                break;
            }
            x = next;
        }
    } else {
        let mut x = start;
        loop {
            out.push(x);
            if x == end {
                break;
            }
            let next = x.saturating_sub(step);
            if next == x || next < end {
                break;
            }
            x = next;
        }
    }

    if out.is_empty() {
        return Err(DemandIndexError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

#[inline]
fn expand_grid_demand_index(
    range: &DemandIndexBatchRange,
) -> Result<Vec<DemandIndexParams>, DemandIndexError> {
    let len_bs_values = expand_axis_usize(range.len_bs)?;
    let len_bs_ma_values = expand_axis_usize(range.len_bs_ma)?;
    let len_di_ma_values = expand_axis_usize(range.len_di_ma)?;
    let ma_type = MaType::parse(range.ma_type.as_deref().unwrap_or("ema"))?;

    let mut out =
        Vec::with_capacity(len_bs_values.len() * len_bs_ma_values.len() * len_di_ma_values.len());
    for &len_bs in &len_bs_values {
        for &len_bs_ma in &len_bs_ma_values {
            for &len_di_ma in &len_di_ma_values {
                validate_lengths(len_bs, len_bs_ma, len_di_ma, 0)?;
                out.push(DemandIndexParams {
                    len_bs: Some(len_bs),
                    len_bs_ma: Some(len_bs_ma),
                    len_di_ma: Some(len_di_ma),
                    ma_type: Some(ma_type.as_str().to_string()),
                });
            }
        }
    }
    Ok(out)
}

#[inline]
pub fn demand_index_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &DemandIndexBatchRange,
    kernel: Kernel,
) -> Result<DemandIndexBatchOutput, DemandIndexError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(DemandIndexError::InvalidKernelForBatch(other)),
    };
    demand_index_batch_par_slice(high, low, close, volume, sweep, batch_kernel.to_non_batch())
}

#[inline]
pub fn demand_index_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &DemandIndexBatchRange,
    kernel: Kernel,
) -> Result<DemandIndexBatchOutput, DemandIndexError> {
    demand_index_batch_inner(high, low, close, volume, sweep, kernel, false)
}

#[inline]
pub fn demand_index_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &DemandIndexBatchRange,
    kernel: Kernel,
) -> Result<DemandIndexBatchOutput, DemandIndexError> {
    demand_index_batch_inner(high, low, close, volume, sweep, kernel, true)
}

fn demand_index_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &DemandIndexBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<DemandIndexBatchOutput, DemandIndexError> {
    if high.is_empty() || low.is_empty() || close.is_empty() || volume.is_empty() {
        return Err(DemandIndexError::EmptyInputData);
    }
    if high.len() != low.len() || high.len() != close.len() || high.len() != volume.len() {
        return Err(DemandIndexError::InconsistentSliceLengths {
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
            volume_len: volume.len(),
        });
    }

    let combos = expand_grid_demand_index(sweep)?;
    let rows = combos.len();
    let cols = high.len();
    let (first, valid) =
        scan_valid_ohlcv(high, low, close, volume).ok_or(DemandIndexError::AllValuesNaN)?;

    let mut di_mu = make_uninit_matrix(rows, cols);
    let mut signal_mu = make_uninit_matrix(rows, cols);
    let mut di_warms = Vec::with_capacity(rows);
    let mut signal_warms = Vec::with_capacity(rows);

    for combo in &combos {
        let len_bs = combo.len_bs.unwrap_or(19);
        let len_bs_ma = combo.len_bs_ma.unwrap_or(19);
        let len_di_ma = combo.len_di_ma.unwrap_or(19);
        let ma_type = MaType::parse(combo.ma_type.as_deref().unwrap_or("ema"))?;
        let needed = needed_valid_bars(ma_type, len_bs, len_bs_ma, len_di_ma);
        if valid < needed {
            return Err(DemandIndexError::NotEnoughValidData { needed, valid });
        }
        di_warms.push((first + di_warmup(ma_type, len_bs, len_bs_ma)).min(cols));
        signal_warms.push((first + signal_warmup(ma_type, len_bs, len_bs_ma, len_di_ma)).min(cols));
    }

    init_matrix_prefixes(&mut di_mu, cols, &di_warms);
    init_matrix_prefixes(&mut signal_mu, cols, &signal_warms);

    let mut di_guard = ManuallyDrop::new(di_mu);
    let mut signal_guard = ManuallyDrop::new(signal_mu);
    let di_out = unsafe {
        std::slice::from_raw_parts_mut(di_guard.as_mut_ptr() as *mut f64, di_guard.len())
    };
    let signal_out = unsafe {
        std::slice::from_raw_parts_mut(signal_guard.as_mut_ptr() as *mut f64, signal_guard.len())
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            di_out
                .par_chunks_mut(cols)
                .zip(signal_out.par_chunks_mut(cols))
                .zip(combos.par_iter())
                .for_each(|((dst_di, dst_signal), combo)| {
                    let ma_type = MaType::parse(combo.ma_type.as_deref().unwrap_or("ema"))
                        .unwrap_or(MaType::Ema);
                    demand_index_compute_into(
                        high,
                        low,
                        close,
                        volume,
                        combo.len_bs.unwrap_or(19),
                        combo.len_bs_ma.unwrap_or(19),
                        combo.len_di_ma.unwrap_or(19),
                        ma_type,
                        dst_di,
                        dst_signal,
                    );
                });
        }
    } else {
        let _ = kernel;
        for (row, combo) in combos.iter().enumerate() {
            let start = row * cols;
            let end = start + cols;
            demand_index_compute_into(
                high,
                low,
                close,
                volume,
                combo.len_bs.unwrap_or(19),
                combo.len_bs_ma.unwrap_or(19),
                combo.len_di_ma.unwrap_or(19),
                MaType::parse(combo.ma_type.as_deref().unwrap_or("ema"))?,
                &mut di_out[start..end],
                &mut signal_out[start..end],
            );
        }
    }

    let demand_index = unsafe {
        Vec::from_raw_parts(
            di_guard.as_mut_ptr() as *mut f64,
            di_guard.len(),
            di_guard.capacity(),
        )
    };
    let signal = unsafe {
        Vec::from_raw_parts(
            signal_guard.as_mut_ptr() as *mut f64,
            signal_guard.len(),
            signal_guard.capacity(),
        )
    };
    core::mem::forget(di_guard);
    core::mem::forget(signal_guard);

    Ok(DemandIndexBatchOutput {
        demand_index,
        signal,
        combos,
        rows,
        cols,
    })
}

pub fn demand_index_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    sweep: &DemandIndexBatchRange,
    kernel: Kernel,
    demand_index_out: &mut [f64],
    signal_out: &mut [f64],
) -> Result<Vec<DemandIndexParams>, DemandIndexError> {
    let out = demand_index_batch_inner(high, low, close, volume, sweep, kernel, false)?;
    let total = out.rows * out.cols;
    if demand_index_out.len() != total || signal_out.len() != total {
        return Err(DemandIndexError::OutputLengthMismatch {
            expected: total,
            demand_index_got: demand_index_out.len(),
            signal_got: signal_out.len(),
        });
    }
    demand_index_out.copy_from_slice(&out.demand_index);
    signal_out.copy_from_slice(&out.signal);
    Ok(out.combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "demand_index")]
#[pyo3(signature = (high, low, close, volume, len_bs=None, len_bs_ma=None, len_di_ma=None, ma_type=None, kernel=None))]
pub fn demand_index_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    len_bs: Option<usize>,
    len_bs_ma: Option<usize>,
    len_di_ma: Option<usize>,
    ma_type: Option<&str>,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let volume = volume.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let input = DemandIndexInput::from_slices(
        high,
        low,
        close,
        volume,
        DemandIndexParams {
            len_bs,
            len_bs_ma,
            len_di_ma,
            ma_type: ma_type.map(|s| s.to_string()),
        },
    );
    let out = py
        .allow_threads(|| demand_index_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        out.demand_index.into_pyarray(py),
        out.signal.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyclass(name = "DemandIndexStream")]
pub struct DemandIndexStreamPy {
    stream: DemandIndexStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl DemandIndexStreamPy {
    #[new]
    #[pyo3(signature = (len_bs=19, len_bs_ma=19, len_di_ma=19, ma_type="ema"))]
    fn new(len_bs: usize, len_bs_ma: usize, len_di_ma: usize, ma_type: &str) -> PyResult<Self> {
        let stream = DemandIndexStream::try_new(DemandIndexParams {
            len_bs: Some(len_bs),
            len_bs_ma: Some(len_bs_ma),
            len_di_ma: Some(len_di_ma),
            ma_type: Some(ma_type.to_string()),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> Option<(f64, f64)> {
        self.stream.update(high, low, close, volume)
    }

    #[getter]
    fn warmup_period(&self) -> usize {
        self.stream.get_warmup_period()
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "demand_index_batch")]
#[pyo3(signature = (high, low, close, volume, len_bs_range, len_bs_ma_range, len_di_ma_range, ma_type=None, kernel=None))]
pub fn demand_index_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    len_bs_range: (usize, usize, usize),
    len_bs_ma_range: (usize, usize, usize),
    len_di_ma_range: (usize, usize, usize),
    ma_type: Option<&str>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let volume = volume.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let sweep = DemandIndexBatchRange {
        len_bs: len_bs_range,
        len_bs_ma: len_bs_ma_range,
        len_di_ma: len_di_ma_range,
        ma_type: Some(ma_type.unwrap_or("ema").to_string()),
    };
    let out = py
        .allow_threads(|| demand_index_batch_with_kernel(high, low, close, volume, &sweep, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item(
        "demand_index",
        out.demand_index
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "signal",
        out.signal.into_pyarray(py).reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "len_bs",
        out.combos
            .iter()
            .map(|p| p.len_bs.unwrap_or(19) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "len_bs_ma",
        out.combos
            .iter()
            .map(|p| p.len_bs_ma.unwrap_or(19) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "len_di_ma",
        out.combos
            .iter()
            .map(|p| p.len_di_ma.unwrap_or(19) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "ma_types",
        out.combos
            .iter()
            .map(|p| p.ma_type.as_deref().unwrap_or("ema").to_string())
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("rows", out.rows)?;
    dict.set_item("cols", out.cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_demand_index_module(module: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(demand_index_py, module)?)?;
    module.add_function(wrap_pyfunction!(demand_index_batch_py, module)?)?;
    module.add_class::<DemandIndexStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "demand_index_js")]
pub fn demand_index_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    len_bs: usize,
    len_bs_ma: usize,
    len_di_ma: usize,
    ma_type: &str,
) -> Result<js_sys::Object, JsValue> {
    let input = DemandIndexInput::from_slices(
        high,
        low,
        close,
        volume,
        DemandIndexParams {
            len_bs: Some(len_bs),
            len_bs_ma: Some(len_bs_ma),
            len_di_ma: Some(len_di_ma),
            ma_type: Some(ma_type.to_string()),
        },
    );
    let out = demand_index(&input).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let result = js_sys::Object::new();
    let di_array = js_sys::Float64Array::new_with_length(out.demand_index.len() as u32);
    di_array.copy_from(&out.demand_index);
    js_sys::Reflect::set(&result, &JsValue::from_str("demand_index"), &di_array)?;
    let signal_array = js_sys::Float64Array::new_with_length(out.signal.len() as u32);
    signal_array.copy_from(&out.signal);
    js_sys::Reflect::set(&result, &JsValue::from_str("signal"), &signal_array)?;
    Ok(result)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn demand_index_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn demand_index_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn demand_index_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    volume_ptr: *const f64,
    demand_index_ptr: *mut f64,
    signal_ptr: *mut f64,
    len: usize,
    len_bs: usize,
    len_bs_ma: usize,
    len_di_ma: usize,
    ma_type: &str,
) -> Result<(), JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || volume_ptr.is_null()
        || demand_index_ptr.is_null()
        || signal_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);
        let input = DemandIndexInput::from_slices(
            high,
            low,
            close,
            volume,
            DemandIndexParams {
                len_bs: Some(len_bs),
                len_bs_ma: Some(len_bs_ma),
                len_di_ma: Some(len_di_ma),
                ma_type: Some(ma_type.to_string()),
            },
        );
        let di_out = std::slice::from_raw_parts_mut(demand_index_ptr, len);
        let signal_out = std::slice::from_raw_parts_mut(signal_ptr, len);
        demand_index_into_slices(di_out, signal_out, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct DemandIndexBatchConfig {
    pub len_bs_range: (usize, usize, usize),
    pub len_bs_ma_range: (usize, usize, usize),
    pub len_di_ma_range: (usize, usize, usize),
    pub ma_type: Option<String>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct DemandIndexBatchJsOutput {
    pub demand_index: Vec<f64>,
    pub signal: Vec<f64>,
    pub combos: Vec<DemandIndexParams>,
    pub len_bs: Vec<usize>,
    pub len_bs_ma: Vec<usize>,
    pub len_di_ma: Vec<usize>,
    pub ma_types: Vec<String>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "demand_index_batch_js")]
pub fn demand_index_batch_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: DemandIndexBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let sweep = DemandIndexBatchRange {
        len_bs: config.len_bs_range,
        len_bs_ma: config.len_bs_ma_range,
        len_di_ma: config.len_di_ma_range,
        ma_type: Some(config.ma_type.unwrap_or_else(|| "ema".to_string())),
    };
    let out = demand_index_batch_inner(
        high,
        low,
        close,
        volume,
        &sweep,
        detect_best_kernel(),
        false,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&DemandIndexBatchJsOutput {
        len_bs: out.combos.iter().map(|p| p.len_bs.unwrap_or(19)).collect(),
        len_bs_ma: out
            .combos
            .iter()
            .map(|p| p.len_bs_ma.unwrap_or(19))
            .collect(),
        len_di_ma: out
            .combos
            .iter()
            .map(|p| p.len_di_ma.unwrap_or(19))
            .collect(),
        ma_types: out
            .combos
            .iter()
            .map(|p| p.ma_type.as_deref().unwrap_or("ema").to_string())
            .collect(),
        demand_index: out.demand_index,
        signal: out.signal,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn demand_index_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    volume_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    len_bs_start: usize,
    len_bs_end: usize,
    len_bs_step: usize,
    len_bs_ma_start: usize,
    len_bs_ma_end: usize,
    len_bs_ma_step: usize,
    len_di_ma_start: usize,
    len_di_ma_end: usize,
    len_di_ma_step: usize,
    ma_type: &str,
) -> Result<usize, JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || volume_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    let sweep = DemandIndexBatchRange {
        len_bs: (len_bs_start, len_bs_end, len_bs_step),
        len_bs_ma: (len_bs_ma_start, len_bs_ma_end, len_bs_ma_step),
        len_di_ma: (len_di_ma_start, len_di_ma_end, len_di_ma_step),
        ma_type: Some(ma_type.to_string()),
    };
    let combos = expand_grid_demand_index(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let total = rows
        .checked_mul(len)
        .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        let batch = demand_index_batch_inner(
            high,
            low,
            close,
            volume,
            &sweep,
            detect_best_kernel(),
            false,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        out.copy_from_slice(&batch.demand_index);
    }
    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn demand_index_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    len_bs: usize,
    len_bs_ma: usize,
    len_di_ma: usize,
    ma_type: &str,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let result = demand_index_js(
        high, low, close, volume, len_bs, len_bs_ma, len_di_ma, ma_type,
    )?;
    let value = JsValue::from(result);
    crate::write_wasm_object_f64_outputs("demand_index_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn demand_index_batch_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = demand_index_batch_js(high, low, close, volume, config)?;
    crate::write_wasm_selected_object_f64_outputs("demand_index_batch_output_into_js", &value, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use std::error::Error;

    fn load_ohlcv() -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>), Box<dyn Error>> {
        let candles = read_candles_from_csv("src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv")?;
        Ok((candles.high, candles.low, candles.close, candles.volume))
    }

    fn assert_series_eq(left: &[f64], right: &[f64]) {
        assert_eq!(left.len(), right.len());
        for (idx, (&a, &b)) in left.iter().zip(right.iter()).enumerate() {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert!(
                (a - b).abs() <= 1e-10,
                "mismatch at {}: left={}, right={}",
                idx,
                a,
                b
            );
        }
    }

    #[test]
    fn demand_index_output_contract() -> Result<(), Box<dyn Error>> {
        let (high, low, close, volume) = load_ohlcv()?;
        let input = DemandIndexInput::from_slices(
            &high,
            &low,
            &close,
            &volume,
            DemandIndexParams::default(),
        );
        let out = demand_index_with_kernel(&input, Kernel::Scalar)?;
        assert_eq!(out.demand_index.len(), close.len());
        assert_eq!(out.signal.len(), close.len());
        assert!(out.demand_index.iter().any(|v| v.is_finite()));
        assert!(out.signal.iter().any(|v| v.is_finite()));
        for &value in out.demand_index.iter().filter(|v| v.is_finite()).take(64) {
            assert!((-100.0..=100.0).contains(&value));
        }
        Ok(())
    }

    #[test]
    fn demand_index_into_matches_api() -> Result<(), Box<dyn Error>> {
        let (high, low, close, volume) = load_ohlcv()?;
        let input = DemandIndexInput::from_slices(
            &high,
            &low,
            &close,
            &volume,
            DemandIndexParams::default(),
        );
        let expected = demand_index(&input)?;
        let mut di = vec![f64::NAN; close.len()];
        let mut signal = vec![f64::NAN; close.len()];
        demand_index_into_slices(&mut di, &mut signal, &input, Kernel::Auto)?;
        assert_series_eq(&di, &expected.demand_index);
        assert_series_eq(&signal, &expected.signal);
        Ok(())
    }

    #[test]
    fn demand_index_kernel_parity() -> Result<(), Box<dyn Error>> {
        let (high, low, close, volume) = load_ohlcv()?;
        let input = DemandIndexInput::from_slices(
            &high,
            &low,
            &close,
            &volume,
            DemandIndexParams::default(),
        );
        let auto = demand_index_with_kernel(&input, Kernel::Auto)?;
        let scalar = demand_index_with_kernel(&input, Kernel::Scalar)?;
        assert_series_eq(&auto.demand_index, &scalar.demand_index);
        assert_series_eq(&auto.signal, &scalar.signal);
        Ok(())
    }

    #[test]
    fn demand_index_invalid_len_bs() {
        let high = [1.0, 2.0, 3.0];
        let low = [0.5, 1.5, 2.5];
        let close = [0.75, 1.75, 2.75];
        let volume = [10.0, 11.0, 12.0];
        let input = DemandIndexInput::from_slices(
            &high,
            &low,
            &close,
            &volume,
            DemandIndexParams {
                len_bs: Some(0),
                len_bs_ma: Some(19),
                len_di_ma: Some(19),
                ma_type: Some("ema".to_string()),
            },
        );
        assert!(matches!(
            demand_index(&input),
            Err(DemandIndexError::InvalidLenBs { .. })
        ));
    }

    #[test]
    fn demand_index_stream_matches_batch() -> Result<(), Box<dyn Error>> {
        let (high, low, close, volume) = load_ohlcv()?;
        let params = DemandIndexParams {
            len_bs: Some(10),
            len_bs_ma: Some(10),
            len_di_ma: Some(8),
            ma_type: Some("ema".to_string()),
        };
        let batch = demand_index(&DemandIndexInput::from_slices(
            &high,
            &low,
            &close,
            &volume,
            params.clone(),
        ))?;
        let mut stream = DemandIndexStream::try_new(params)?;
        let mut di = Vec::with_capacity(close.len());
        let mut signal = Vec::with_capacity(close.len());
        for i in 0..close.len() {
            if let Some((di_value, signal_value)) =
                stream.update(high[i], low[i], close[i], volume[i])
            {
                di.push(di_value);
                signal.push(signal_value);
            } else {
                di.push(f64::NAN);
                signal.push(f64::NAN);
            }
        }
        assert_series_eq(&di, &batch.demand_index);
        assert_series_eq(&signal, &batch.signal);
        Ok(())
    }

    #[test]
    fn demand_index_batch_single_matches_single() -> Result<(), Box<dyn Error>> {
        let (high, low, close, volume) = load_ohlcv()?;
        let sweep = DemandIndexBatchRange::default();
        let batch =
            demand_index_batch_with_kernel(&high, &low, &close, &volume, &sweep, Kernel::Auto)?;
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, close.len());
        let single = demand_index(&DemandIndexInput::from_slices(
            &high,
            &low,
            &close,
            &volume,
            DemandIndexParams::default(),
        ))?;
        assert_series_eq(&batch.demand_index[..close.len()], &single.demand_index);
        assert_series_eq(&batch.signal[..close.len()], &single.signal);
        Ok(())
    }

    #[test]
    fn demand_index_invalid_window_recovers() -> Result<(), Box<dyn Error>> {
        let (mut high, mut low, mut close, mut volume) = load_ohlcv()?;
        high[80] = f64::NAN;
        low[80] = f64::NAN;
        close[80] = f64::NAN;
        volume[80] = f64::NAN;

        let out = demand_index(&DemandIndexInput::from_slices(
            &high,
            &low,
            &close,
            &volume,
            DemandIndexParams::default(),
        ))?;
        assert!(out.demand_index[80].is_nan());
        assert!(out.signal[80].is_nan());
        assert!(out.demand_index.iter().skip(140).any(|v| v.is_finite()));
        assert!(out.signal.iter().skip(160).any(|v| v.is_finite()));
        Ok(())
    }
}
