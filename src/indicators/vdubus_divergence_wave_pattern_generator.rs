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
use crate::utilities::helpers::{alloc_uninit_f64, detect_best_batch_kernel, make_uninit_matrix};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn vdubus_divergence_wave_pattern_generator_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    fast_depth: usize,
    slow_depth: usize,
    fast_length: usize,
    slow_length: usize,
    signal_length: usize,
    lookback: usize,
    err_tol: f64,
    show_standard: bool,
    show_climax: bool,
    show_rounded: bool,
    show_predator: bool,
    show_gartley: bool,
    show_bat: bool,
    show_butterfly: bool,
    show_crab: bool,
    show_deep: bool,
    show_hs: bool,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = vdubus_divergence_wave_pattern_generator_js(
        high,
        low,
        close,
        fast_depth,
        slow_depth,
        fast_length,
        slow_length,
        signal_length,
        lookback,
        err_tol,
        show_standard,
        show_climax,
        show_rounded,
        show_predator,
        show_gartley,
        show_bat,
        show_butterfly,
        show_crab,
        show_deep,
        show_hs,
    )?;
    crate::write_wasm_object_f64_outputs(
        "vdubus_divergence_wave_pattern_generator_output_into_js",
        &value,
        out,
    )
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn vdubus_divergence_wave_pattern_generator_batch_unified_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value =
        vdubus_divergence_wave_pattern_generator_batch_unified_js(high, low, close, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "vdubus_divergence_wave_pattern_generator_batch_unified_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
use std::error::Error as StdError;
use std::mem::ManuallyDrop;
use thiserror::Error;

const DEFAULT_FAST_DEPTH: usize = 9;
const DEFAULT_SLOW_DEPTH: usize = 24;
const DEFAULT_FAST_LENGTH: usize = 21;
const DEFAULT_SLOW_LENGTH: usize = 34;
const DEFAULT_SIGNAL_LENGTH: usize = 5;
const DEFAULT_LOOKBACK: usize = 3;
const DEFAULT_ERR_TOL: f64 = 0.15;

const DEFAULT_SHOW_STANDARD: bool = true;
const DEFAULT_SHOW_CLIMAX: bool = true;
const DEFAULT_SHOW_ROUNDED: bool = true;
const DEFAULT_SHOW_PREDATOR: bool = true;
const DEFAULT_SHOW_GARTLEY: bool = false;
const DEFAULT_SHOW_BAT: bool = false;
const DEFAULT_SHOW_BUTTERFLY: bool = false;
const DEFAULT_SHOW_CRAB: bool = false;
const DEFAULT_SHOW_DEEP: bool = false;
const DEFAULT_SHOW_HS: bool = true;

const FAMILY_NONE: f64 = 0.0;
const FAMILY_RETRACEMENT: f64 = 1.0;
const FAMILY_GARTLEY: f64 = 2.0;
const FAMILY_BAT: f64 = 3.0;
const FAMILY_BUTTERFLY: f64 = 4.0;
const FAMILY_CRAB: f64 = 5.0;
const FAMILY_DEEP: f64 = 6.0;
const FAMILY_HEAD_SHOULDERS: f64 = 7.0;

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

#[derive(Debug, Clone)]
pub enum VdubusDivergenceWavePatternGeneratorData<'a> {
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
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct VdubusDivergenceWavePatternGeneratorOutput {
    pub fast_standard: Vec<f64>,
    pub fast_climax: Vec<f64>,
    pub fast_rounded: Vec<f64>,
    pub fast_predator: Vec<f64>,
    pub slow_standard: Vec<f64>,
    pub slow_climax: Vec<f64>,
    pub slow_rounded: Vec<f64>,
    pub slow_predator: Vec<f64>,
    pub opposing_force: Vec<f64>,
    pub macd: Vec<f64>,
    pub signal: Vec<f64>,
    pub hist: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct VdubusDivergenceWavePatternGeneratorParams {
    pub fast_depth: Option<usize>,
    pub slow_depth: Option<usize>,
    pub fast_length: Option<usize>,
    pub slow_length: Option<usize>,
    pub signal_length: Option<usize>,
    pub lookback: Option<usize>,
    pub err_tol: Option<f64>,
    pub show_standard: Option<bool>,
    pub show_climax: Option<bool>,
    pub show_rounded: Option<bool>,
    pub show_predator: Option<bool>,
    pub show_gartley: Option<bool>,
    pub show_bat: Option<bool>,
    pub show_butterfly: Option<bool>,
    pub show_crab: Option<bool>,
    pub show_deep: Option<bool>,
    pub show_hs: Option<bool>,
}

impl Default for VdubusDivergenceWavePatternGeneratorParams {
    fn default() -> Self {
        Self {
            fast_depth: Some(DEFAULT_FAST_DEPTH),
            slow_depth: Some(DEFAULT_SLOW_DEPTH),
            fast_length: Some(DEFAULT_FAST_LENGTH),
            slow_length: Some(DEFAULT_SLOW_LENGTH),
            signal_length: Some(DEFAULT_SIGNAL_LENGTH),
            lookback: Some(DEFAULT_LOOKBACK),
            err_tol: Some(DEFAULT_ERR_TOL),
            show_standard: Some(DEFAULT_SHOW_STANDARD),
            show_climax: Some(DEFAULT_SHOW_CLIMAX),
            show_rounded: Some(DEFAULT_SHOW_ROUNDED),
            show_predator: Some(DEFAULT_SHOW_PREDATOR),
            show_gartley: Some(DEFAULT_SHOW_GARTLEY),
            show_bat: Some(DEFAULT_SHOW_BAT),
            show_butterfly: Some(DEFAULT_SHOW_BUTTERFLY),
            show_crab: Some(DEFAULT_SHOW_CRAB),
            show_deep: Some(DEFAULT_SHOW_DEEP),
            show_hs: Some(DEFAULT_SHOW_HS),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VdubusDivergenceWavePatternGeneratorInput<'a> {
    pub data: VdubusDivergenceWavePatternGeneratorData<'a>,
    pub params: VdubusDivergenceWavePatternGeneratorParams,
}

impl<'a> VdubusDivergenceWavePatternGeneratorInput<'a> {
    #[inline(always)]
    pub fn from_candles(
        candles: &'a Candles,
        params: VdubusDivergenceWavePatternGeneratorParams,
    ) -> Self {
        Self {
            data: VdubusDivergenceWavePatternGeneratorData::Candles { candles },
            params,
        }
    }

    #[inline(always)]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: VdubusDivergenceWavePatternGeneratorParams,
    ) -> Self {
        Self {
            data: VdubusDivergenceWavePatternGeneratorData::Slices { high, low, close },
            params,
        }
    }

    #[inline(always)]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(
            candles,
            VdubusDivergenceWavePatternGeneratorParams::default(),
        )
    }

    #[inline(always)]
    pub fn get_fast_depth(&self) -> usize {
        self.params.fast_depth.unwrap_or(DEFAULT_FAST_DEPTH)
    }

    #[inline(always)]
    pub fn get_slow_depth(&self) -> usize {
        self.params.slow_depth.unwrap_or(DEFAULT_SLOW_DEPTH)
    }

    #[inline(always)]
    pub fn get_fast_length(&self) -> usize {
        self.params.fast_length.unwrap_or(DEFAULT_FAST_LENGTH)
    }

    #[inline(always)]
    pub fn get_slow_length(&self) -> usize {
        self.params.slow_length.unwrap_or(DEFAULT_SLOW_LENGTH)
    }

    #[inline(always)]
    pub fn get_signal_length(&self) -> usize {
        self.params.signal_length.unwrap_or(DEFAULT_SIGNAL_LENGTH)
    }

    #[inline(always)]
    pub fn get_lookback(&self) -> usize {
        self.params.lookback.unwrap_or(DEFAULT_LOOKBACK)
    }

    #[inline(always)]
    pub fn get_err_tol(&self) -> f64 {
        self.params.err_tol.unwrap_or(DEFAULT_ERR_TOL)
    }

    #[inline(always)]
    pub fn get_show_standard(&self) -> bool {
        self.params.show_standard.unwrap_or(DEFAULT_SHOW_STANDARD)
    }

    #[inline(always)]
    pub fn get_show_climax(&self) -> bool {
        self.params.show_climax.unwrap_or(DEFAULT_SHOW_CLIMAX)
    }

    #[inline(always)]
    pub fn get_show_rounded(&self) -> bool {
        self.params.show_rounded.unwrap_or(DEFAULT_SHOW_ROUNDED)
    }

    #[inline(always)]
    pub fn get_show_predator(&self) -> bool {
        self.params.show_predator.unwrap_or(DEFAULT_SHOW_PREDATOR)
    }

    #[inline(always)]
    pub fn get_show_gartley(&self) -> bool {
        self.params.show_gartley.unwrap_or(DEFAULT_SHOW_GARTLEY)
    }

    #[inline(always)]
    pub fn get_show_bat(&self) -> bool {
        self.params.show_bat.unwrap_or(DEFAULT_SHOW_BAT)
    }

    #[inline(always)]
    pub fn get_show_butterfly(&self) -> bool {
        self.params.show_butterfly.unwrap_or(DEFAULT_SHOW_BUTTERFLY)
    }

    #[inline(always)]
    pub fn get_show_crab(&self) -> bool {
        self.params.show_crab.unwrap_or(DEFAULT_SHOW_CRAB)
    }

    #[inline(always)]
    pub fn get_show_deep(&self) -> bool {
        self.params.show_deep.unwrap_or(DEFAULT_SHOW_DEEP)
    }

    #[inline(always)]
    pub fn get_show_hs(&self) -> bool {
        self.params.show_hs.unwrap_or(DEFAULT_SHOW_HS)
    }

    #[inline(always)]
    fn as_hlc(&self) -> (&'a [f64], &'a [f64], &'a [f64]) {
        match &self.data {
            VdubusDivergenceWavePatternGeneratorData::Candles { candles } => (
                high_source(candles),
                low_source(candles),
                close_source(candles),
            ),
            VdubusDivergenceWavePatternGeneratorData::Slices { high, low, close } => {
                (*high, *low, *close)
            }
        }
    }
}

impl<'a> AsRef<[f64]> for VdubusDivergenceWavePatternGeneratorInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        self.as_hlc().2
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ResolvedParams {
    fast_depth: usize,
    slow_depth: usize,
    fast_length: usize,
    slow_length: usize,
    signal_length: usize,
    lookback: usize,
    err_tol: f64,
    show_standard: bool,
    show_climax: bool,
    show_rounded: bool,
    show_predator: bool,
    show_gartley: bool,
    show_bat: bool,
    show_butterfly: bool,
    show_crab: bool,
    show_deep: bool,
    show_hs: bool,
}

impl From<&VdubusDivergenceWavePatternGeneratorInput<'_>> for ResolvedParams {
    fn from(value: &VdubusDivergenceWavePatternGeneratorInput<'_>) -> Self {
        Self {
            fast_depth: value.get_fast_depth(),
            slow_depth: value.get_slow_depth(),
            fast_length: value.get_fast_length(),
            slow_length: value.get_slow_length(),
            signal_length: value.get_signal_length(),
            lookback: value.get_lookback(),
            err_tol: value.get_err_tol(),
            show_standard: value.get_show_standard(),
            show_climax: value.get_show_climax(),
            show_rounded: value.get_show_rounded(),
            show_predator: value.get_show_predator(),
            show_gartley: value.get_show_gartley(),
            show_bat: value.get_show_bat(),
            show_butterfly: value.get_show_butterfly(),
            show_crab: value.get_show_crab(),
            show_deep: value.get_show_deep(),
            show_hs: value.get_show_hs(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct VdubusDivergenceWavePatternGeneratorBuilder {
    fast_depth: Option<usize>,
    slow_depth: Option<usize>,
    fast_length: Option<usize>,
    slow_length: Option<usize>,
    signal_length: Option<usize>,
    lookback: Option<usize>,
    err_tol: Option<f64>,
    show_standard: Option<bool>,
    show_climax: Option<bool>,
    show_rounded: Option<bool>,
    show_predator: Option<bool>,
    show_gartley: Option<bool>,
    show_bat: Option<bool>,
    show_butterfly: Option<bool>,
    show_crab: Option<bool>,
    show_deep: Option<bool>,
    show_hs: Option<bool>,
    kernel: Kernel,
}

impl Default for VdubusDivergenceWavePatternGeneratorBuilder {
    fn default() -> Self {
        Self {
            fast_depth: None,
            slow_depth: None,
            fast_length: None,
            slow_length: None,
            signal_length: None,
            lookback: None,
            err_tol: None,
            show_standard: None,
            show_climax: None,
            show_rounded: None,
            show_predator: None,
            show_gartley: None,
            show_bat: None,
            show_butterfly: None,
            show_crab: None,
            show_deep: None,
            show_hs: None,
            kernel: Kernel::Auto,
        }
    }
}

impl VdubusDivergenceWavePatternGeneratorBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn fast_depth(mut self, value: usize) -> Self {
        self.fast_depth = Some(value);
        self
    }

    #[inline(always)]
    pub fn slow_depth(mut self, value: usize) -> Self {
        self.slow_depth = Some(value);
        self
    }

    #[inline(always)]
    pub fn fast_length(mut self, value: usize) -> Self {
        self.fast_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn slow_length(mut self, value: usize) -> Self {
        self.slow_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn signal_length(mut self, value: usize) -> Self {
        self.signal_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn lookback(mut self, value: usize) -> Self {
        self.lookback = Some(value);
        self
    }

    #[inline(always)]
    pub fn err_tol(mut self, value: f64) -> Self {
        self.err_tol = Some(value);
        self
    }

    #[inline(always)]
    pub fn show_standard(mut self, value: bool) -> Self {
        self.show_standard = Some(value);
        self
    }

    #[inline(always)]
    pub fn show_climax(mut self, value: bool) -> Self {
        self.show_climax = Some(value);
        self
    }

    #[inline(always)]
    pub fn show_rounded(mut self, value: bool) -> Self {
        self.show_rounded = Some(value);
        self
    }

    #[inline(always)]
    pub fn show_predator(mut self, value: bool) -> Self {
        self.show_predator = Some(value);
        self
    }

    #[inline(always)]
    pub fn show_gartley(mut self, value: bool) -> Self {
        self.show_gartley = Some(value);
        self
    }

    #[inline(always)]
    pub fn show_bat(mut self, value: bool) -> Self {
        self.show_bat = Some(value);
        self
    }

    #[inline(always)]
    pub fn show_butterfly(mut self, value: bool) -> Self {
        self.show_butterfly = Some(value);
        self
    }

    #[inline(always)]
    pub fn show_crab(mut self, value: bool) -> Self {
        self.show_crab = Some(value);
        self
    }

    #[inline(always)]
    pub fn show_deep(mut self, value: bool) -> Self {
        self.show_deep = Some(value);
        self
    }

    #[inline(always)]
    pub fn show_hs(mut self, value: bool) -> Self {
        self.show_hs = Some(value);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline(always)]
    fn params(self) -> VdubusDivergenceWavePatternGeneratorParams {
        VdubusDivergenceWavePatternGeneratorParams {
            fast_depth: self.fast_depth,
            slow_depth: self.slow_depth,
            fast_length: self.fast_length,
            slow_length: self.slow_length,
            signal_length: self.signal_length,
            lookback: self.lookback,
            err_tol: self.err_tol,
            show_standard: self.show_standard,
            show_climax: self.show_climax,
            show_rounded: self.show_rounded,
            show_predator: self.show_predator,
            show_gartley: self.show_gartley,
            show_bat: self.show_bat,
            show_butterfly: self.show_butterfly,
            show_crab: self.show_crab,
            show_deep: self.show_deep,
            show_hs: self.show_hs,
        }
    }
}

#[derive(Debug, Error)]
pub enum VdubusDivergenceWavePatternGeneratorError {
    #[error("vdubus_divergence_wave_pattern_generator: input data slice is empty.")]
    EmptyInputData,
    #[error("vdubus_divergence_wave_pattern_generator: all values are NaN.")]
    AllValuesNaN,
    #[error(
        "vdubus_divergence_wave_pattern_generator: inconsistent data lengths - high = {high_len}, low = {low_len}, close = {close_len}"
    )]
    DataLengthMismatch {
        high_len: usize,
        low_len: usize,
        close_len: usize,
    },
    #[error(
        "vdubus_divergence_wave_pattern_generator: invalid depth/lookback: fast_depth = {fast_depth}, slow_depth = {slow_depth}, lookback = {lookback}"
    )]
    InvalidDepth {
        fast_depth: usize,
        slow_depth: usize,
        lookback: usize,
    },
    #[error(
        "vdubus_divergence_wave_pattern_generator: invalid periods: fast_length = {fast_length}, slow_length = {slow_length}, signal_length = {signal_length}, data length = {data_len}"
    )]
    InvalidPeriods {
        fast_length: usize,
        slow_length: usize,
        signal_length: usize,
        data_len: usize,
    },
    #[error("vdubus_divergence_wave_pattern_generator: invalid err_tol: {err_tol}")]
    InvalidTolerance { err_tol: f64 },
    #[error(
        "vdubus_divergence_wave_pattern_generator: not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "vdubus_divergence_wave_pattern_generator: output length mismatch: expected = {expected}, got = {got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error(
        "vdubus_divergence_wave_pattern_generator: invalid range for {axis}: start = {start}, end = {end}, step = {step}"
    )]
    InvalidRange {
        axis: &'static str,
        start: String,
        end: String,
        step: String,
    },
    #[error("vdubus_divergence_wave_pattern_generator: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Clone, Copy, Debug)]
struct PreparedInput<'a> {
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    params: ResolvedParams,
    warmup: usize,
}

#[inline(always)]
fn normalize_single_kernel(_kernel: Kernel) -> Kernel {
    Kernel::Scalar
}

#[inline(always)]
fn required_valid_bars(slow_length: usize, signal_length: usize) -> usize {
    slow_length + signal_length - 1
}

#[inline(always)]
fn validate_params(
    params: ResolvedParams,
    data_len: usize,
) -> Result<(), VdubusDivergenceWavePatternGeneratorError> {
    if params.fast_depth == 0 || params.slow_depth == 0 || params.lookback == 0 {
        return Err(VdubusDivergenceWavePatternGeneratorError::InvalidDepth {
            fast_depth: params.fast_depth,
            slow_depth: params.slow_depth,
            lookback: params.lookback,
        });
    }
    if params.fast_length == 0
        || params.slow_length == 0
        || params.signal_length == 0
        || params.fast_length > data_len
        || params.slow_length > data_len
        || params.signal_length > data_len
    {
        return Err(VdubusDivergenceWavePatternGeneratorError::InvalidPeriods {
            fast_length: params.fast_length,
            slow_length: params.slow_length,
            signal_length: params.signal_length,
            data_len,
        });
    }
    if !params.err_tol.is_finite() || params.err_tol <= 0.0 || params.err_tol > 0.5 {
        return Err(
            VdubusDivergenceWavePatternGeneratorError::InvalidTolerance {
                err_tol: params.err_tol,
            },
        );
    }
    Ok(())
}

#[inline(always)]
fn analyze_valid_segments(
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> Result<(usize, usize), VdubusDivergenceWavePatternGeneratorError> {
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(VdubusDivergenceWavePatternGeneratorError::EmptyInputData);
    }
    if high.len() != low.len() || high.len() != close.len() {
        return Err(
            VdubusDivergenceWavePatternGeneratorError::DataLengthMismatch {
                high_len: high.len(),
                low_len: low.len(),
                close_len: close.len(),
            },
        );
    }

    let mut first_valid = None;
    let mut max_run = 0usize;
    let mut run = 0usize;

    for i in 0..close.len() {
        let valid = high[i].is_finite() && low[i].is_finite() && close[i].is_finite();
        if valid {
            if first_valid.is_none() {
                first_valid = Some(i);
            }
            run += 1;
            max_run = max_run.max(run);
        } else {
            run = 0;
        }
    }

    match first_valid {
        Some(first) => Ok((first, max_run)),
        None => Err(VdubusDivergenceWavePatternGeneratorError::AllValuesNaN),
    }
}

#[inline(always)]
fn prepare_input<'a>(
    input: &'a VdubusDivergenceWavePatternGeneratorInput<'a>,
    kernel: Kernel,
) -> Result<PreparedInput<'a>, VdubusDivergenceWavePatternGeneratorError> {
    let _chosen = normalize_single_kernel(kernel);
    let (high, low, close) = input.as_hlc();
    let params = ResolvedParams::from(input);
    validate_params(params, close.len())?;
    let (first_valid, max_run) = analyze_valid_segments(high, low, close)?;
    let needed = required_valid_bars(params.slow_length, params.signal_length);
    if max_run < needed {
        return Err(
            VdubusDivergenceWavePatternGeneratorError::NotEnoughValidData {
                needed,
                valid: max_run,
            },
        );
    }
    Ok(PreparedInput {
        high,
        low,
        close,
        params,
        warmup: first_valid + needed - 1,
    })
}

#[derive(Clone, Debug)]
struct EmaState {
    length: usize,
    alpha: f64,
    count: usize,
    sum: f64,
    value: f64,
    started: bool,
}

impl EmaState {
    #[inline(always)]
    fn new(length: usize) -> Self {
        Self {
            length,
            alpha: 2.0 / (length as f64 + 1.0),
            count: 0,
            sum: 0.0,
            value: f64::NAN,
            started: false,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.count = 0;
        self.sum = 0.0;
        self.value = f64::NAN;
        self.started = false;
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<f64> {
        if !value.is_finite() {
            self.reset();
            return None;
        }
        if !self.started {
            self.sum += value;
            self.count += 1;
            if self.count == self.length {
                self.value = self.sum / self.length as f64;
                self.started = true;
                Some(self.value)
            } else {
                None
            }
        } else {
            self.value = self.alpha.mul_add(value, (1.0 - self.alpha) * self.value);
            Some(self.value)
        }
    }
}

#[derive(Clone, Debug)]
struct MacdState {
    fast: EmaState,
    slow: EmaState,
    signal: EmaState,
}

impl MacdState {
    #[inline(always)]
    fn new(fast_length: usize, slow_length: usize, signal_length: usize) -> Self {
        Self {
            fast: EmaState::new(fast_length),
            slow: EmaState::new(slow_length),
            signal: EmaState::new(signal_length),
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.fast.reset();
        self.slow.reset();
        self.signal.reset();
    }

    #[inline(always)]
    fn update(&mut self, close: f64) -> Option<(f64, f64, f64)> {
        let fast = self.fast.update(close)?;
        let slow = self.slow.update(close)?;
        let macd = fast - slow;
        let signal = self.signal.update(macd)?;
        Some((macd, signal, macd - signal))
    }
}

#[derive(Clone, Copy, Debug)]
enum PivotKind {
    High,
    Low,
}

#[derive(Clone, Debug)]
struct PivotDetector {
    kind: PivotKind,
    span: usize,
    window: Vec<f64>,
    head: usize,
    count: usize,
}

impl PivotDetector {
    #[inline(always)]
    fn new(kind: PivotKind, span: usize) -> Self {
        Self {
            kind,
            span,
            window: vec![f64::NAN; span * 2 + 1],
            head: 0,
            count: 0,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.head = 0;
        self.count = 0;
        self.window.fill(f64::NAN);
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<f64> {
        if !value.is_finite() {
            self.reset();
            return None;
        }

        self.window[self.head] = value;
        self.head += 1;
        if self.head == self.window.len() {
            self.head = 0;
        }
        if self.count < self.window.len() {
            self.count += 1;
        }
        if self.count < self.window.len() {
            return None;
        }

        let start = self.head;
        let center_idx = (start + self.span) % self.window.len();
        let center = self.window[center_idx];
        if !center.is_finite() {
            return None;
        }

        for j in 0..self.window.len() {
            if j == self.span {
                continue;
            }
            let current = self.window[(start + j) % self.window.len()];
            if !current.is_finite() {
                return None;
            }
            match self.kind {
                PivotKind::High => {
                    if center <= current {
                        return None;
                    }
                }
                PivotKind::Low => {
                    if center >= current {
                        return None;
                    }
                }
            }
        }
        Some(center)
    }
}

#[inline(always)]
fn push_front_cap(values: &mut Vec<f64>, value: f64) {
    values.insert(0, value);
    if values.len() > 10 {
        values.pop();
    }
}

#[derive(Clone, Debug)]
struct MomentumState {
    high_detector: PivotDetector,
    low_detector: PivotDetector,
    wave_highs: Vec<f64>,
    wave_lows: Vec<f64>,
}

impl MomentumState {
    #[inline(always)]
    fn new(lookback: usize) -> Self {
        Self {
            high_detector: PivotDetector::new(PivotKind::High, lookback),
            low_detector: PivotDetector::new(PivotKind::Low, lookback),
            wave_highs: Vec::with_capacity(10),
            wave_lows: Vec::with_capacity(10),
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.high_detector.reset();
        self.low_detector.reset();
        self.wave_highs.clear();
        self.wave_lows.clear();
    }

    #[inline(always)]
    fn update(&mut self, hist: f64) {
        if let Some(value) = self.high_detector.update(hist) {
            push_front_cap(&mut self.wave_highs, value);
        }
        if let Some(value) = self.low_detector.update(hist) {
            push_front_cap(&mut self.wave_lows, value);
        }
    }

    #[inline(always)]
    fn opposing_force(&self) -> f64 {
        let bull = self.wave_lows.first().copied().unwrap_or(0.0).abs();
        let bear = self.wave_highs.first().copied().unwrap_or(0.0).abs();
        if bull > bear {
            1.0
        } else if bear > bull {
            -1.0
        } else {
            0.0
        }
    }

    #[inline(always)]
    fn standard_bearish(&self) -> bool {
        self.wave_highs.len() >= 3
            && self.wave_highs[1] < self.wave_highs[2]
            && self.wave_highs[0] <= self.wave_highs[1]
    }

    #[inline(always)]
    fn standard_bullish(&self) -> bool {
        self.wave_lows.len() >= 3
            && self.wave_lows[1] > self.wave_lows[2]
            && self.wave_lows[0] >= self.wave_lows[1]
    }

    #[inline(always)]
    fn climax_bearish(&self) -> bool {
        self.wave_highs.len() >= 3
            && self.wave_highs[1] >= self.wave_highs[2]
            && self.wave_highs[0] < self.wave_highs[1]
    }

    #[inline(always)]
    fn climax_bullish(&self) -> bool {
        self.wave_lows.len() >= 3
            && self.wave_lows[1] <= self.wave_lows[2]
            && self.wave_lows[0] > self.wave_lows[1]
    }

    #[inline(always)]
    fn rounded_bearish(&self) -> bool {
        self.wave_highs.len() >= 4
            && self.wave_highs[3] > self.wave_highs[2]
            && self.wave_highs[2] > self.wave_highs[1]
            && self.wave_highs[1] > self.wave_highs[0]
    }

    #[inline(always)]
    fn rounded_bullish(&self) -> bool {
        self.wave_lows.len() >= 4
            && self.wave_lows[3] < self.wave_lows[2]
            && self.wave_lows[2] < self.wave_lows[1]
            && self.wave_lows[1] < self.wave_lows[0]
    }

    #[inline(always)]
    fn bearish_predator(&self) -> bool {
        self.wave_highs.len() >= 2 && self.wave_highs[0] > self.wave_highs[1]
    }

    #[inline(always)]
    fn bullish_predator(&self) -> bool {
        self.wave_lows.len() >= 2 && self.wave_lows[0] < self.wave_lows[1]
    }
}

#[inline(always)]
fn harmonic_family_code(xb_ratio: f64, xd_ratio: f64, err_tol: f64) -> f64 {
    if (xb_ratio - 0.618).abs() < err_tol && (xd_ratio - 0.786).abs() < err_tol {
        FAMILY_GARTLEY
    } else if xb_ratio >= 0.382 - err_tol
        && xb_ratio <= 0.5 + err_tol
        && (xd_ratio - 0.886).abs() < err_tol
    {
        FAMILY_BAT
    } else if (xb_ratio - 0.786).abs() < err_tol
        && xd_ratio >= 1.27 - err_tol
        && xd_ratio <= 1.618 + err_tol
    {
        FAMILY_BUTTERFLY
    } else if xb_ratio >= 0.382 - err_tol
        && xb_ratio <= 0.618 + err_tol
        && (xd_ratio - 1.618).abs() < err_tol
    {
        FAMILY_CRAB
    } else if xd_ratio > 1.0 {
        FAMILY_DEEP
    } else {
        FAMILY_RETRACEMENT
    }
}

#[inline(always)]
fn standard_family_from_filters(params: &ResolvedParams, family: f64, is_hs: bool) -> f64 {
    if is_hs {
        if params.show_hs {
            return FAMILY_HEAD_SHOULDERS;
        }
        return FAMILY_NONE;
    }

    match family as i32 {
        1 => FAMILY_RETRACEMENT,
        2 if params.show_gartley => FAMILY_GARTLEY,
        3 if params.show_bat => FAMILY_BAT,
        4 if params.show_butterfly => FAMILY_BUTTERFLY,
        5 if params.show_crab => FAMILY_CRAB,
        6 if params.show_deep => FAMILY_DEEP,
        _ => FAMILY_NONE,
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct EngineSignals {
    standard: f64,
    climax: f64,
    rounded: f64,
    predator: f64,
}

#[derive(Clone, Debug)]
struct StructureEngine {
    high_detector: PivotDetector,
    low_detector: PivotDetector,
    pivots: Vec<f64>,
}

impl StructureEngine {
    #[inline(always)]
    fn new(depth: usize) -> Self {
        Self {
            high_detector: PivotDetector::new(PivotKind::High, depth),
            low_detector: PivotDetector::new(PivotKind::Low, depth),
            pivots: Vec::with_capacity(10),
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.high_detector.reset();
        self.low_detector.reset();
        self.pivots.clear();
    }

    #[inline(always)]
    fn push_pivot(&mut self, value: f64) {
        push_front_cap(&mut self.pivots, value);
    }

    #[inline(always)]
    fn evaluate_bearish(&self, momentum: &MomentumState, params: &ResolvedParams) -> EngineSignals {
        if self.pivots.len() < 5 {
            return EngineSignals::default();
        }

        let y_d = self.pivots[0];
        let y_b = self.pivots[2];
        let y_a = self.pivots[3];
        let y_x = self.pivots[4];
        let xa_len = (y_a - y_x).abs();
        let ab_len = (y_b - y_a).abs();
        let xb_ratio = if xa_len != 0.0 { ab_len / xa_len } else { 0.0 };
        let xd_ratio = if xa_len != 0.0 {
            (y_d - y_x).abs() / xa_len
        } else {
            0.0
        };
        let raw_family = harmonic_family_code(xb_ratio, xd_ratio, params.err_tol);
        let is_hs = params.show_hs && y_b > y_x && y_b > y_d;
        let standard_momentum = momentum.standard_bearish();

        let mut out = EngineSignals::default();
        if params.show_standard && standard_momentum {
            let family = standard_family_from_filters(params, raw_family, is_hs);
            if family != FAMILY_NONE {
                out.standard = -family;
            }
        }
        if params.show_climax && momentum.climax_bearish() {
            out.climax = -1.0;
        }
        if params.show_rounded && momentum.rounded_bearish() {
            out.rounded = -1.0;
        }
        if params.show_predator && !standard_momentum && y_d < y_x && momentum.bearish_predator() {
            out.predator = -1.0;
        }
        out
    }

    #[inline(always)]
    fn evaluate_bullish(&self, momentum: &MomentumState, params: &ResolvedParams) -> EngineSignals {
        if self.pivots.len() < 5 {
            return EngineSignals::default();
        }

        let y_d = self.pivots[0];
        let y_b = self.pivots[2];
        let y_a = self.pivots[3];
        let y_x = self.pivots[4];
        let xa_len = (y_a - y_x).abs();
        let ab_len = (y_b - y_a).abs();
        let xb_ratio = if xa_len != 0.0 { ab_len / xa_len } else { 0.0 };
        let xd_ratio = if xa_len != 0.0 {
            (y_d - y_x).abs() / xa_len
        } else {
            0.0
        };
        let raw_family = harmonic_family_code(xb_ratio, xd_ratio, params.err_tol);
        let is_inverse_hs = params.show_hs && y_b < y_x && y_b < y_d;
        let standard_momentum = momentum.standard_bullish();

        let mut out = EngineSignals::default();
        if params.show_standard && standard_momentum {
            let family = standard_family_from_filters(params, raw_family, is_inverse_hs);
            if family != FAMILY_NONE {
                out.standard = family;
            }
        }
        if params.show_climax && momentum.climax_bullish() {
            out.climax = 1.0;
        }
        if params.show_rounded && momentum.rounded_bullish() {
            out.rounded = 1.0;
        }
        if params.show_predator && !standard_momentum && y_d > y_x && momentum.bullish_predator() {
            out.predator = 1.0;
        }
        out
    }

    #[inline(always)]
    fn update(
        &mut self,
        high: f64,
        low: f64,
        momentum: &MomentumState,
        params: &ResolvedParams,
    ) -> EngineSignals {
        let mut out = EngineSignals::default();
        if let Some(value) = self.high_detector.update(high) {
            self.push_pivot(value);
            out = self.evaluate_bearish(momentum, params);
        }
        if let Some(value) = self.low_detector.update(low) {
            self.push_pivot(value);
            out = self.evaluate_bullish(momentum, params);
        }
        out
    }
}

#[derive(Clone, Debug)]
struct VdubusDivergenceWavePatternGeneratorState {
    params: ResolvedParams,
    macd: MacdState,
    momentum: MomentumState,
    fast_engine: StructureEngine,
    slow_engine: StructureEngine,
}

impl VdubusDivergenceWavePatternGeneratorState {
    #[inline(always)]
    fn new(params: ResolvedParams) -> Self {
        Self {
            macd: MacdState::new(params.fast_length, params.slow_length, params.signal_length),
            momentum: MomentumState::new(params.lookback),
            fast_engine: StructureEngine::new(params.fast_depth),
            slow_engine: StructureEngine::new(params.slow_depth),
            params,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.macd.reset();
        self.momentum.reset();
        self.fast_engine.reset();
        self.slow_engine.reset();
    }

    #[inline(always)]
    fn update(
        &mut self,
        high: f64,
        low: f64,
        close: f64,
    ) -> Option<(f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64)> {
        if !high.is_finite() || !low.is_finite() || !close.is_finite() {
            self.reset();
            return None;
        }

        let macd_out = self.macd.update(close);
        if let Some((_, _, hist)) = macd_out {
            self.momentum.update(hist);
        }

        let fast = self
            .fast_engine
            .update(high, low, &self.momentum, &self.params);
        let slow = self
            .slow_engine
            .update(high, low, &self.momentum, &self.params);
        let (macd, signal, hist) = macd_out?;
        Some((
            fast.standard,
            fast.climax,
            fast.rounded,
            fast.predator,
            slow.standard,
            slow.climax,
            slow.rounded,
            slow.predator,
            self.momentum.opposing_force(),
            macd,
            signal,
            hist,
        ))
    }
}

#[derive(Clone, Debug)]
pub struct VdubusDivergenceWavePatternGeneratorStream {
    params: VdubusDivergenceWavePatternGeneratorParams,
    state: VdubusDivergenceWavePatternGeneratorState,
}

impl VdubusDivergenceWavePatternGeneratorStream {
    #[inline(always)]
    pub fn try_new(
        params: VdubusDivergenceWavePatternGeneratorParams,
    ) -> Result<Self, VdubusDivergenceWavePatternGeneratorError> {
        let dummy = VdubusDivergenceWavePatternGeneratorInput::from_slices(
            &[1.0],
            &[1.0],
            &[1.0],
            params.clone(),
        );
        let resolved = ResolvedParams::from(&dummy);
        validate_params(resolved, usize::MAX)?;
        Ok(Self {
            state: VdubusDivergenceWavePatternGeneratorState::new(resolved),
            params,
        })
    }

    #[inline(always)]
    pub fn update(
        &mut self,
        high: f64,
        low: f64,
        close: f64,
    ) -> Option<(f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64)> {
        self.state.update(high, low, close)
    }

    #[inline(always)]
    pub fn params(&self) -> &VdubusDivergenceWavePatternGeneratorParams {
        &self.params
    }
}

#[derive(Clone, Debug)]
pub struct VdubusDivergenceWavePatternGeneratorBatchRange {
    pub fast_depth: (usize, usize, usize),
    pub slow_depth: (usize, usize, usize),
    pub fast_length: (usize, usize, usize),
    pub slow_length: (usize, usize, usize),
    pub signal_length: (usize, usize, usize),
    pub lookback: (usize, usize, usize),
    pub err_tol: (f64, f64, f64),
    pub show_standard: bool,
    pub show_climax: bool,
    pub show_rounded: bool,
    pub show_predator: bool,
    pub show_gartley: bool,
    pub show_bat: bool,
    pub show_butterfly: bool,
    pub show_crab: bool,
    pub show_deep: bool,
    pub show_hs: bool,
}

impl Default for VdubusDivergenceWavePatternGeneratorBatchRange {
    fn default() -> Self {
        Self {
            fast_depth: (DEFAULT_FAST_DEPTH, DEFAULT_FAST_DEPTH, 0),
            slow_depth: (DEFAULT_SLOW_DEPTH, DEFAULT_SLOW_DEPTH, 0),
            fast_length: (DEFAULT_FAST_LENGTH, DEFAULT_FAST_LENGTH, 0),
            slow_length: (DEFAULT_SLOW_LENGTH, DEFAULT_SLOW_LENGTH, 0),
            signal_length: (DEFAULT_SIGNAL_LENGTH, DEFAULT_SIGNAL_LENGTH, 0),
            lookback: (DEFAULT_LOOKBACK, DEFAULT_LOOKBACK, 0),
            err_tol: (DEFAULT_ERR_TOL, DEFAULT_ERR_TOL, 0.0),
            show_standard: DEFAULT_SHOW_STANDARD,
            show_climax: DEFAULT_SHOW_CLIMAX,
            show_rounded: DEFAULT_SHOW_ROUNDED,
            show_predator: DEFAULT_SHOW_PREDATOR,
            show_gartley: DEFAULT_SHOW_GARTLEY,
            show_bat: DEFAULT_SHOW_BAT,
            show_butterfly: DEFAULT_SHOW_BUTTERFLY,
            show_crab: DEFAULT_SHOW_CRAB,
            show_deep: DEFAULT_SHOW_DEEP,
            show_hs: DEFAULT_SHOW_HS,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct VdubusDivergenceWavePatternGeneratorBatchBuilder {
    range: VdubusDivergenceWavePatternGeneratorBatchRange,
    kernel: Kernel,
}

#[derive(Clone, Debug)]
pub struct VdubusDivergenceWavePatternGeneratorBatchOutput {
    pub fast_standard: Vec<f64>,
    pub fast_climax: Vec<f64>,
    pub fast_rounded: Vec<f64>,
    pub fast_predator: Vec<f64>,
    pub slow_standard: Vec<f64>,
    pub slow_climax: Vec<f64>,
    pub slow_rounded: Vec<f64>,
    pub slow_predator: Vec<f64>,
    pub opposing_force: Vec<f64>,
    pub macd: Vec<f64>,
    pub signal: Vec<f64>,
    pub hist: Vec<f64>,
    pub combos: Vec<VdubusDivergenceWavePatternGeneratorParams>,
    pub rows: usize,
    pub cols: usize,
}

impl VdubusDivergenceWavePatternGeneratorBatchBuilder {
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
    pub fn fast_depth_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.fast_depth = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn slow_depth_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.slow_depth = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn fast_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.fast_length = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn slow_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.slow_length = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn signal_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.signal_length = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn lookback_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.lookback = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn err_tol_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.err_tol = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn show_standard(mut self, value: bool) -> Self {
        self.range.show_standard = value;
        self
    }

    #[inline(always)]
    pub fn show_climax(mut self, value: bool) -> Self {
        self.range.show_climax = value;
        self
    }

    #[inline(always)]
    pub fn show_rounded(mut self, value: bool) -> Self {
        self.range.show_rounded = value;
        self
    }

    #[inline(always)]
    pub fn show_predator(mut self, value: bool) -> Self {
        self.range.show_predator = value;
        self
    }

    #[inline(always)]
    pub fn show_gartley(mut self, value: bool) -> Self {
        self.range.show_gartley = value;
        self
    }

    #[inline(always)]
    pub fn show_bat(mut self, value: bool) -> Self {
        self.range.show_bat = value;
        self
    }

    #[inline(always)]
    pub fn show_butterfly(mut self, value: bool) -> Self {
        self.range.show_butterfly = value;
        self
    }

    #[inline(always)]
    pub fn show_crab(mut self, value: bool) -> Self {
        self.range.show_crab = value;
        self
    }

    #[inline(always)]
    pub fn show_deep(mut self, value: bool) -> Self {
        self.range.show_deep = value;
        self
    }

    #[inline(always)]
    pub fn show_hs(mut self, value: bool) -> Self {
        self.range.show_hs = value;
        self
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<
        VdubusDivergenceWavePatternGeneratorBatchOutput,
        VdubusDivergenceWavePatternGeneratorError,
    > {
        vdubus_divergence_wave_pattern_generator_batch_with_kernel(
            high,
            low,
            close,
            &self.range,
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<
        VdubusDivergenceWavePatternGeneratorBatchOutput,
        VdubusDivergenceWavePatternGeneratorError,
    > {
        self.apply_slices(&candles.high, &candles.low, &candles.close)
    }
}

#[inline(always)]
fn axis_usize(
    axis: &'static str,
    (start, end, step): (usize, usize, usize),
) -> Result<Vec<usize>, VdubusDivergenceWavePatternGeneratorError> {
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
            if value == end {
                break;
            }
            match value.checked_sub(step) {
                Some(next) if next < value => value = next,
                _ => break,
            }
        }
    }

    if out.is_empty() || !out.last().is_some_and(|value| *value == end) {
        return Err(VdubusDivergenceWavePatternGeneratorError::InvalidRange {
            axis,
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

#[inline(always)]
fn axis_float(
    axis: &'static str,
    (start, end, step): (f64, f64, f64),
) -> Result<Vec<f64>, VdubusDivergenceWavePatternGeneratorError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() || step < 0.0 {
        return Err(VdubusDivergenceWavePatternGeneratorError::InvalidRange {
            axis,
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if step == 0.0 || start == end {
        return Ok(vec![start]);
    }

    let eps = step.abs() * 1e-9 + 1e-12;
    let mut out = Vec::new();
    if start < end {
        let mut value = start;
        while value <= end + eps {
            out.push(value);
            value += step;
        }
    } else {
        let mut value = start;
        while value + eps >= end {
            out.push(value);
            value -= step;
        }
    }

    if out.is_empty() {
        return Err(VdubusDivergenceWavePatternGeneratorError::InvalidRange {
            axis,
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

pub fn expand_grid_vdubus_divergence_wave_pattern_generator(
    sweep: &VdubusDivergenceWavePatternGeneratorBatchRange,
) -> Result<
    Vec<VdubusDivergenceWavePatternGeneratorParams>,
    VdubusDivergenceWavePatternGeneratorError,
> {
    let fast_depths = axis_usize("fast_depth", sweep.fast_depth)?;
    let slow_depths = axis_usize("slow_depth", sweep.slow_depth)?;
    let fast_lengths = axis_usize("fast_length", sweep.fast_length)?;
    let slow_lengths = axis_usize("slow_length", sweep.slow_length)?;
    let signal_lengths = axis_usize("signal_length", sweep.signal_length)?;
    let lookbacks = axis_usize("lookback", sweep.lookback)?;
    let err_tols = axis_float("err_tol", sweep.err_tol)?;

    let mut out = Vec::new();
    for &fast_depth in &fast_depths {
        for &slow_depth in &slow_depths {
            for &fast_length in &fast_lengths {
                for &slow_length in &slow_lengths {
                    for &signal_length in &signal_lengths {
                        for &lookback in &lookbacks {
                            for &err_tol in &err_tols {
                                out.push(VdubusDivergenceWavePatternGeneratorParams {
                                    fast_depth: Some(fast_depth),
                                    slow_depth: Some(slow_depth),
                                    fast_length: Some(fast_length),
                                    slow_length: Some(slow_length),
                                    signal_length: Some(signal_length),
                                    lookback: Some(lookback),
                                    err_tol: Some(err_tol),
                                    show_standard: Some(sweep.show_standard),
                                    show_climax: Some(sweep.show_climax),
                                    show_rounded: Some(sweep.show_rounded),
                                    show_predator: Some(sweep.show_predator),
                                    show_gartley: Some(sweep.show_gartley),
                                    show_bat: Some(sweep.show_bat),
                                    show_butterfly: Some(sweep.show_butterfly),
                                    show_crab: Some(sweep.show_crab),
                                    show_deep: Some(sweep.show_deep),
                                    show_hs: Some(sweep.show_hs),
                                });
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

fn compute_row(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    params: ResolvedParams,
    fast_standard_out: &mut [f64],
    fast_climax_out: &mut [f64],
    fast_rounded_out: &mut [f64],
    fast_predator_out: &mut [f64],
    slow_standard_out: &mut [f64],
    slow_climax_out: &mut [f64],
    slow_rounded_out: &mut [f64],
    slow_predator_out: &mut [f64],
    opposing_force_out: &mut [f64],
    macd_out: &mut [f64],
    signal_out: &mut [f64],
    hist_out: &mut [f64],
) -> Result<(), VdubusDivergenceWavePatternGeneratorError> {
    let len = close.len();
    if fast_standard_out.len() != len
        || fast_climax_out.len() != len
        || fast_rounded_out.len() != len
        || fast_predator_out.len() != len
        || slow_standard_out.len() != len
        || slow_climax_out.len() != len
        || slow_rounded_out.len() != len
        || slow_predator_out.len() != len
        || opposing_force_out.len() != len
        || macd_out.len() != len
        || signal_out.len() != len
        || hist_out.len() != len
    {
        return Err(
            VdubusDivergenceWavePatternGeneratorError::OutputLengthMismatch {
                expected: len,
                got: fast_standard_out
                    .len()
                    .max(fast_climax_out.len())
                    .max(fast_rounded_out.len())
                    .max(fast_predator_out.len())
                    .max(slow_standard_out.len())
                    .max(slow_climax_out.len())
                    .max(slow_rounded_out.len())
                    .max(slow_predator_out.len())
                    .max(opposing_force_out.len())
                    .max(macd_out.len())
                    .max(signal_out.len())
                    .max(hist_out.len()),
            },
        );
    }

    fast_standard_out.fill(f64::NAN);
    fast_climax_out.fill(f64::NAN);
    fast_rounded_out.fill(f64::NAN);
    fast_predator_out.fill(f64::NAN);
    slow_standard_out.fill(f64::NAN);
    slow_climax_out.fill(f64::NAN);
    slow_rounded_out.fill(f64::NAN);
    slow_predator_out.fill(f64::NAN);
    opposing_force_out.fill(f64::NAN);
    macd_out.fill(f64::NAN);
    signal_out.fill(f64::NAN);
    hist_out.fill(f64::NAN);

    let mut state = VdubusDivergenceWavePatternGeneratorState::new(params);
    for i in 0..len {
        if let Some((
            fast_standard,
            fast_climax,
            fast_rounded,
            fast_predator,
            slow_standard,
            slow_climax,
            slow_rounded,
            slow_predator,
            opposing_force,
            macd,
            signal,
            hist,
        )) = state.update(high[i], low[i], close[i])
        {
            fast_standard_out[i] = fast_standard;
            fast_climax_out[i] = fast_climax;
            fast_rounded_out[i] = fast_rounded;
            fast_predator_out[i] = fast_predator;
            slow_standard_out[i] = slow_standard;
            slow_climax_out[i] = slow_climax;
            slow_rounded_out[i] = slow_rounded;
            slow_predator_out[i] = slow_predator;
            opposing_force_out[i] = opposing_force;
            macd_out[i] = macd;
            signal_out[i] = signal;
            hist_out[i] = hist;
        }
    }
    Ok(())
}

#[inline]
pub fn vdubus_divergence_wave_pattern_generator(
    input: &VdubusDivergenceWavePatternGeneratorInput,
) -> Result<VdubusDivergenceWavePatternGeneratorOutput, VdubusDivergenceWavePatternGeneratorError> {
    vdubus_divergence_wave_pattern_generator_with_kernel(input, Kernel::Auto)
}

pub fn vdubus_divergence_wave_pattern_generator_with_kernel(
    input: &VdubusDivergenceWavePatternGeneratorInput,
    kernel: Kernel,
) -> Result<VdubusDivergenceWavePatternGeneratorOutput, VdubusDivergenceWavePatternGeneratorError> {
    let prepared = prepare_input(input, kernel)?;
    let len = prepared.close.len();
    let mut output = VdubusDivergenceWavePatternGeneratorOutput {
        fast_standard: alloc_uninit_f64(len),
        fast_climax: alloc_uninit_f64(len),
        fast_rounded: alloc_uninit_f64(len),
        fast_predator: alloc_uninit_f64(len),
        slow_standard: alloc_uninit_f64(len),
        slow_climax: alloc_uninit_f64(len),
        slow_rounded: alloc_uninit_f64(len),
        slow_predator: alloc_uninit_f64(len),
        opposing_force: alloc_uninit_f64(len),
        macd: alloc_uninit_f64(len),
        signal: alloc_uninit_f64(len),
        hist: alloc_uninit_f64(len),
    };
    let _ = prepared.warmup;
    compute_row(
        prepared.high,
        prepared.low,
        prepared.close,
        prepared.params,
        &mut output.fast_standard,
        &mut output.fast_climax,
        &mut output.fast_rounded,
        &mut output.fast_predator,
        &mut output.slow_standard,
        &mut output.slow_climax,
        &mut output.slow_rounded,
        &mut output.slow_predator,
        &mut output.opposing_force,
        &mut output.macd,
        &mut output.signal,
        &mut output.hist,
    )?;
    Ok(output)
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn vdubus_divergence_wave_pattern_generator_into(
    fast_standard_out: &mut [f64],
    fast_climax_out: &mut [f64],
    fast_rounded_out: &mut [f64],
    fast_predator_out: &mut [f64],
    slow_standard_out: &mut [f64],
    slow_climax_out: &mut [f64],
    slow_rounded_out: &mut [f64],
    slow_predator_out: &mut [f64],
    opposing_force_out: &mut [f64],
    macd_out: &mut [f64],
    signal_out: &mut [f64],
    hist_out: &mut [f64],
    input: &VdubusDivergenceWavePatternGeneratorInput,
) -> Result<(), VdubusDivergenceWavePatternGeneratorError> {
    vdubus_divergence_wave_pattern_generator_into_slice(
        fast_standard_out,
        fast_climax_out,
        fast_rounded_out,
        fast_predator_out,
        slow_standard_out,
        slow_climax_out,
        slow_rounded_out,
        slow_predator_out,
        opposing_force_out,
        macd_out,
        signal_out,
        hist_out,
        input,
        Kernel::Auto,
    )
}

pub fn vdubus_divergence_wave_pattern_generator_into_slice(
    fast_standard_out: &mut [f64],
    fast_climax_out: &mut [f64],
    fast_rounded_out: &mut [f64],
    fast_predator_out: &mut [f64],
    slow_standard_out: &mut [f64],
    slow_climax_out: &mut [f64],
    slow_rounded_out: &mut [f64],
    slow_predator_out: &mut [f64],
    opposing_force_out: &mut [f64],
    macd_out: &mut [f64],
    signal_out: &mut [f64],
    hist_out: &mut [f64],
    input: &VdubusDivergenceWavePatternGeneratorInput,
    kernel: Kernel,
) -> Result<(), VdubusDivergenceWavePatternGeneratorError> {
    let prepared = prepare_input(input, kernel)?;
    compute_row(
        prepared.high,
        prepared.low,
        prepared.close,
        prepared.params,
        fast_standard_out,
        fast_climax_out,
        fast_rounded_out,
        fast_predator_out,
        slow_standard_out,
        slow_climax_out,
        slow_rounded_out,
        slow_predator_out,
        opposing_force_out,
        macd_out,
        signal_out,
        hist_out,
    )
}

fn vdubus_divergence_wave_pattern_generator_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &VdubusDivergenceWavePatternGeneratorBatchRange,
    parallel: bool,
    fast_standard_out: &mut [f64],
    fast_climax_out: &mut [f64],
    fast_rounded_out: &mut [f64],
    fast_predator_out: &mut [f64],
    slow_standard_out: &mut [f64],
    slow_climax_out: &mut [f64],
    slow_rounded_out: &mut [f64],
    slow_predator_out: &mut [f64],
    opposing_force_out: &mut [f64],
    macd_out: &mut [f64],
    signal_out: &mut [f64],
    hist_out: &mut [f64],
) -> Result<
    Vec<VdubusDivergenceWavePatternGeneratorParams>,
    VdubusDivergenceWavePatternGeneratorError,
> {
    let (_, max_run) = analyze_valid_segments(high, low, close)?;
    let combos = expand_grid_vdubus_divergence_wave_pattern_generator(sweep)?;
    let rows = combos.len();
    let cols = close.len();
    let expected = rows.checked_mul(cols).ok_or(
        VdubusDivergenceWavePatternGeneratorError::OutputLengthMismatch {
            expected: usize::MAX,
            got: fast_standard_out.len(),
        },
    )?;
    if fast_standard_out.len() != expected
        || fast_climax_out.len() != expected
        || fast_rounded_out.len() != expected
        || fast_predator_out.len() != expected
        || slow_standard_out.len() != expected
        || slow_climax_out.len() != expected
        || slow_rounded_out.len() != expected
        || slow_predator_out.len() != expected
        || opposing_force_out.len() != expected
        || macd_out.len() != expected
        || signal_out.len() != expected
        || hist_out.len() != expected
    {
        return Err(
            VdubusDivergenceWavePatternGeneratorError::OutputLengthMismatch {
                expected,
                got: fast_standard_out
                    .len()
                    .max(fast_climax_out.len())
                    .max(fast_rounded_out.len())
                    .max(fast_predator_out.len())
                    .max(slow_standard_out.len())
                    .max(slow_climax_out.len())
                    .max(slow_rounded_out.len())
                    .max(slow_predator_out.len())
                    .max(opposing_force_out.len())
                    .max(macd_out.len())
                    .max(signal_out.len())
                    .max(hist_out.len()),
            },
        );
    }

    for combo in &combos {
        let input =
            VdubusDivergenceWavePatternGeneratorInput::from_slices(high, low, close, combo.clone());
        let params = ResolvedParams::from(&input);
        validate_params(params, cols)?;
        let needed = required_valid_bars(params.slow_length, params.signal_length);
        if max_run < needed {
            return Err(
                VdubusDivergenceWavePatternGeneratorError::NotEnoughValidData {
                    needed,
                    valid: max_run,
                },
            );
        }
    }

    let do_row = |row: usize,
                  fast_standard_row: &mut [f64],
                  fast_climax_row: &mut [f64],
                  fast_rounded_row: &mut [f64],
                  fast_predator_row: &mut [f64],
                  slow_standard_row: &mut [f64],
                  slow_climax_row: &mut [f64],
                  slow_rounded_row: &mut [f64],
                  slow_predator_row: &mut [f64],
                  opposing_force_row: &mut [f64],
                  macd_row: &mut [f64],
                  signal_row: &mut [f64],
                  hist_row: &mut [f64]| {
        let params = ResolvedParams::from(&VdubusDivergenceWavePatternGeneratorInput::from_slices(
            high,
            low,
            close,
            combos[row].clone(),
        ));
        compute_row(
            high,
            low,
            close,
            params,
            fast_standard_row,
            fast_climax_row,
            fast_rounded_row,
            fast_predator_row,
            slow_standard_row,
            slow_climax_row,
            slow_rounded_row,
            slow_predator_row,
            opposing_force_row,
            macd_row,
            signal_row,
            hist_row,
        )
    };

    let run_row = |row: usize,
                   fast_standard_row: &mut [f64],
                   fast_climax_row: &mut [f64],
                   fast_rounded_row: &mut [f64],
                   fast_predator_row: &mut [f64],
                   slow_standard_row: &mut [f64],
                   slow_climax_row: &mut [f64],
                   slow_rounded_row: &mut [f64],
                   slow_predator_row: &mut [f64],
                   opposing_force_row: &mut [f64],
                   macd_row: &mut [f64],
                   signal_row: &mut [f64],
                   hist_row: &mut [f64]| {
        do_row(
            row,
            fast_standard_row,
            fast_climax_row,
            fast_rounded_row,
            fast_predator_row,
            slow_standard_row,
            slow_climax_row,
            slow_rounded_row,
            slow_predator_row,
            opposing_force_row,
            macd_row,
            signal_row,
            hist_row,
        )
    };

    let _ = parallel;
    for row in 0..rows {
        let start = row * cols;
        run_row(
            row,
            &mut fast_standard_out[start..start + cols],
            &mut fast_climax_out[start..start + cols],
            &mut fast_rounded_out[start..start + cols],
            &mut fast_predator_out[start..start + cols],
            &mut slow_standard_out[start..start + cols],
            &mut slow_climax_out[start..start + cols],
            &mut slow_rounded_out[start..start + cols],
            &mut slow_predator_out[start..start + cols],
            &mut opposing_force_out[start..start + cols],
            &mut macd_out[start..start + cols],
            &mut signal_out[start..start + cols],
            &mut hist_out[start..start + cols],
        )?;
    }

    Ok(combos)
}

pub fn vdubus_divergence_wave_pattern_generator_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &VdubusDivergenceWavePatternGeneratorBatchRange,
    kernel: Kernel,
) -> Result<
    VdubusDivergenceWavePatternGeneratorBatchOutput,
    VdubusDivergenceWavePatternGeneratorError,
> {
    match kernel {
        Kernel::Auto => {
            let _ = detect_best_batch_kernel();
        }
        k if !k.is_batch() => {
            return Err(VdubusDivergenceWavePatternGeneratorError::InvalidKernelForBatch(k));
        }
        _ => {}
    }
    vdubus_divergence_wave_pattern_generator_batch_slice(
        high,
        low,
        close,
        sweep,
        Kernel::ScalarBatch,
    )
}

pub fn vdubus_divergence_wave_pattern_generator_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &VdubusDivergenceWavePatternGeneratorBatchRange,
    _kernel: Kernel,
) -> Result<
    VdubusDivergenceWavePatternGeneratorBatchOutput,
    VdubusDivergenceWavePatternGeneratorError,
> {
    vdubus_divergence_wave_pattern_generator_batch_impl(high, low, close, sweep, false)
}

pub fn vdubus_divergence_wave_pattern_generator_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &VdubusDivergenceWavePatternGeneratorBatchRange,
    _kernel: Kernel,
) -> Result<
    VdubusDivergenceWavePatternGeneratorBatchOutput,
    VdubusDivergenceWavePatternGeneratorError,
> {
    vdubus_divergence_wave_pattern_generator_batch_impl(high, low, close, sweep, false)
}

fn vdubus_divergence_wave_pattern_generator_batch_impl(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &VdubusDivergenceWavePatternGeneratorBatchRange,
    parallel: bool,
) -> Result<
    VdubusDivergenceWavePatternGeneratorBatchOutput,
    VdubusDivergenceWavePatternGeneratorError,
> {
    let rows = expand_grid_vdubus_divergence_wave_pattern_generator(sweep)?.len();
    let cols = close.len();

    let mut fast_standard_guard = ManuallyDrop::new(make_uninit_matrix(rows, cols));
    let mut fast_climax_guard = ManuallyDrop::new(make_uninit_matrix(rows, cols));
    let mut fast_rounded_guard = ManuallyDrop::new(make_uninit_matrix(rows, cols));
    let mut fast_predator_guard = ManuallyDrop::new(make_uninit_matrix(rows, cols));
    let mut slow_standard_guard = ManuallyDrop::new(make_uninit_matrix(rows, cols));
    let mut slow_climax_guard = ManuallyDrop::new(make_uninit_matrix(rows, cols));
    let mut slow_rounded_guard = ManuallyDrop::new(make_uninit_matrix(rows, cols));
    let mut slow_predator_guard = ManuallyDrop::new(make_uninit_matrix(rows, cols));
    let mut opposing_force_guard = ManuallyDrop::new(make_uninit_matrix(rows, cols));
    let mut macd_guard = ManuallyDrop::new(make_uninit_matrix(rows, cols));
    let mut signal_guard = ManuallyDrop::new(make_uninit_matrix(rows, cols));
    let mut hist_guard = ManuallyDrop::new(make_uninit_matrix(rows, cols));

    let fast_standard_out = unsafe {
        core::slice::from_raw_parts_mut(
            fast_standard_guard.as_mut_ptr() as *mut f64,
            fast_standard_guard.len(),
        )
    };
    let fast_climax_out = unsafe {
        core::slice::from_raw_parts_mut(
            fast_climax_guard.as_mut_ptr() as *mut f64,
            fast_climax_guard.len(),
        )
    };
    let fast_rounded_out = unsafe {
        core::slice::from_raw_parts_mut(
            fast_rounded_guard.as_mut_ptr() as *mut f64,
            fast_rounded_guard.len(),
        )
    };
    let fast_predator_out = unsafe {
        core::slice::from_raw_parts_mut(
            fast_predator_guard.as_mut_ptr() as *mut f64,
            fast_predator_guard.len(),
        )
    };
    let slow_standard_out = unsafe {
        core::slice::from_raw_parts_mut(
            slow_standard_guard.as_mut_ptr() as *mut f64,
            slow_standard_guard.len(),
        )
    };
    let slow_climax_out = unsafe {
        core::slice::from_raw_parts_mut(
            slow_climax_guard.as_mut_ptr() as *mut f64,
            slow_climax_guard.len(),
        )
    };
    let slow_rounded_out = unsafe {
        core::slice::from_raw_parts_mut(
            slow_rounded_guard.as_mut_ptr() as *mut f64,
            slow_rounded_guard.len(),
        )
    };
    let slow_predator_out = unsafe {
        core::slice::from_raw_parts_mut(
            slow_predator_guard.as_mut_ptr() as *mut f64,
            slow_predator_guard.len(),
        )
    };
    let opposing_force_out = unsafe {
        core::slice::from_raw_parts_mut(
            opposing_force_guard.as_mut_ptr() as *mut f64,
            opposing_force_guard.len(),
        )
    };
    let macd_out = unsafe {
        core::slice::from_raw_parts_mut(macd_guard.as_mut_ptr() as *mut f64, macd_guard.len())
    };
    let signal_out = unsafe {
        core::slice::from_raw_parts_mut(signal_guard.as_mut_ptr() as *mut f64, signal_guard.len())
    };
    let hist_out = unsafe {
        core::slice::from_raw_parts_mut(hist_guard.as_mut_ptr() as *mut f64, hist_guard.len())
    };

    let combos = vdubus_divergence_wave_pattern_generator_batch_inner_into(
        high,
        low,
        close,
        sweep,
        parallel,
        fast_standard_out,
        fast_climax_out,
        fast_rounded_out,
        fast_predator_out,
        slow_standard_out,
        slow_climax_out,
        slow_rounded_out,
        slow_predator_out,
        opposing_force_out,
        macd_out,
        signal_out,
        hist_out,
    )?;

    let fast_standard = unsafe {
        Vec::from_raw_parts(
            fast_standard_guard.as_mut_ptr() as *mut f64,
            fast_standard_guard.len(),
            fast_standard_guard.capacity(),
        )
    };
    let fast_climax = unsafe {
        Vec::from_raw_parts(
            fast_climax_guard.as_mut_ptr() as *mut f64,
            fast_climax_guard.len(),
            fast_climax_guard.capacity(),
        )
    };
    let fast_rounded = unsafe {
        Vec::from_raw_parts(
            fast_rounded_guard.as_mut_ptr() as *mut f64,
            fast_rounded_guard.len(),
            fast_rounded_guard.capacity(),
        )
    };
    let fast_predator = unsafe {
        Vec::from_raw_parts(
            fast_predator_guard.as_mut_ptr() as *mut f64,
            fast_predator_guard.len(),
            fast_predator_guard.capacity(),
        )
    };
    let slow_standard = unsafe {
        Vec::from_raw_parts(
            slow_standard_guard.as_mut_ptr() as *mut f64,
            slow_standard_guard.len(),
            slow_standard_guard.capacity(),
        )
    };
    let slow_climax = unsafe {
        Vec::from_raw_parts(
            slow_climax_guard.as_mut_ptr() as *mut f64,
            slow_climax_guard.len(),
            slow_climax_guard.capacity(),
        )
    };
    let slow_rounded = unsafe {
        Vec::from_raw_parts(
            slow_rounded_guard.as_mut_ptr() as *mut f64,
            slow_rounded_guard.len(),
            slow_rounded_guard.capacity(),
        )
    };
    let slow_predator = unsafe {
        Vec::from_raw_parts(
            slow_predator_guard.as_mut_ptr() as *mut f64,
            slow_predator_guard.len(),
            slow_predator_guard.capacity(),
        )
    };
    let opposing_force = unsafe {
        Vec::from_raw_parts(
            opposing_force_guard.as_mut_ptr() as *mut f64,
            opposing_force_guard.len(),
            opposing_force_guard.capacity(),
        )
    };
    let macd = unsafe {
        Vec::from_raw_parts(
            macd_guard.as_mut_ptr() as *mut f64,
            macd_guard.len(),
            macd_guard.capacity(),
        )
    };
    let signal = unsafe {
        Vec::from_raw_parts(
            signal_guard.as_mut_ptr() as *mut f64,
            signal_guard.len(),
            signal_guard.capacity(),
        )
    };
    let hist = unsafe {
        Vec::from_raw_parts(
            hist_guard.as_mut_ptr() as *mut f64,
            hist_guard.len(),
            hist_guard.capacity(),
        )
    };

    Ok(VdubusDivergenceWavePatternGeneratorBatchOutput {
        fast_standard,
        fast_climax,
        fast_rounded,
        fast_predator,
        slow_standard,
        slow_climax,
        slow_rounded,
        slow_predator,
        opposing_force,
        macd,
        signal,
        hist,
        combos,
        rows,
        cols,
    })
}

#[cfg(feature = "python")]
fn output_to_pydict<'py>(
    py: Python<'py>,
    output: VdubusDivergenceWavePatternGeneratorOutput,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("fast_standard", output.fast_standard.into_pyarray(py))?;
    dict.set_item("fast_climax", output.fast_climax.into_pyarray(py))?;
    dict.set_item("fast_rounded", output.fast_rounded.into_pyarray(py))?;
    dict.set_item("fast_predator", output.fast_predator.into_pyarray(py))?;
    dict.set_item("slow_standard", output.slow_standard.into_pyarray(py))?;
    dict.set_item("slow_climax", output.slow_climax.into_pyarray(py))?;
    dict.set_item("slow_rounded", output.slow_rounded.into_pyarray(py))?;
    dict.set_item("slow_predator", output.slow_predator.into_pyarray(py))?;
    dict.set_item("opposing_force", output.opposing_force.into_pyarray(py))?;
    dict.set_item("macd", output.macd.into_pyarray(py))?;
    dict.set_item("signal", output.signal.into_pyarray(py))?;
    dict.set_item("hist", output.hist.into_pyarray(py))?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyfunction(name = "vdubus_divergence_wave_pattern_generator")]
#[pyo3(signature = (high, low, close, fast_depth=DEFAULT_FAST_DEPTH, slow_depth=DEFAULT_SLOW_DEPTH, fast_length=DEFAULT_FAST_LENGTH, slow_length=DEFAULT_SLOW_LENGTH, signal_length=DEFAULT_SIGNAL_LENGTH, lookback=DEFAULT_LOOKBACK, err_tol=DEFAULT_ERR_TOL, show_standard=DEFAULT_SHOW_STANDARD, show_climax=DEFAULT_SHOW_CLIMAX, show_rounded=DEFAULT_SHOW_ROUNDED, show_predator=DEFAULT_SHOW_PREDATOR, show_gartley=DEFAULT_SHOW_GARTLEY, show_bat=DEFAULT_SHOW_BAT, show_butterfly=DEFAULT_SHOW_BUTTERFLY, show_crab=DEFAULT_SHOW_CRAB, show_deep=DEFAULT_SHOW_DEEP, show_hs=DEFAULT_SHOW_HS, kernel=None))]
pub fn vdubus_divergence_wave_pattern_generator_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    fast_depth: usize,
    slow_depth: usize,
    fast_length: usize,
    slow_length: usize,
    signal_length: usize,
    lookback: usize,
    err_tol: f64,
    show_standard: bool,
    show_climax: bool,
    show_rounded: bool,
    show_predator: bool,
    show_gartley: bool,
    show_bat: bool,
    show_butterfly: bool,
    show_crab: bool,
    show_deep: bool,
    show_hs: bool,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let input = VdubusDivergenceWavePatternGeneratorInput::from_slices(
        high.as_slice()?,
        low.as_slice()?,
        close.as_slice()?,
        VdubusDivergenceWavePatternGeneratorParams {
            fast_depth: Some(fast_depth),
            slow_depth: Some(slow_depth),
            fast_length: Some(fast_length),
            slow_length: Some(slow_length),
            signal_length: Some(signal_length),
            lookback: Some(lookback),
            err_tol: Some(err_tol),
            show_standard: Some(show_standard),
            show_climax: Some(show_climax),
            show_rounded: Some(show_rounded),
            show_predator: Some(show_predator),
            show_gartley: Some(show_gartley),
            show_bat: Some(show_bat),
            show_butterfly: Some(show_butterfly),
            show_crab: Some(show_crab),
            show_deep: Some(show_deep),
            show_hs: Some(show_hs),
        },
    );
    let kernel = validate_kernel(kernel, false)?;
    let output = py
        .allow_threads(|| vdubus_divergence_wave_pattern_generator_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    output_to_pydict(py, output)
}

#[cfg(feature = "python")]
#[pyfunction(name = "vdubus_divergence_wave_pattern_generator_batch")]
#[pyo3(signature = (high, low, close, fast_depth_range=(DEFAULT_FAST_DEPTH, DEFAULT_FAST_DEPTH, 0), slow_depth_range=(DEFAULT_SLOW_DEPTH, DEFAULT_SLOW_DEPTH, 0), fast_length_range=(DEFAULT_FAST_LENGTH, DEFAULT_FAST_LENGTH, 0), slow_length_range=(DEFAULT_SLOW_LENGTH, DEFAULT_SLOW_LENGTH, 0), signal_length_range=(DEFAULT_SIGNAL_LENGTH, DEFAULT_SIGNAL_LENGTH, 0), lookback_range=(DEFAULT_LOOKBACK, DEFAULT_LOOKBACK, 0), err_tol_range=(DEFAULT_ERR_TOL, DEFAULT_ERR_TOL, 0.0), show_standard=DEFAULT_SHOW_STANDARD, show_climax=DEFAULT_SHOW_CLIMAX, show_rounded=DEFAULT_SHOW_ROUNDED, show_predator=DEFAULT_SHOW_PREDATOR, show_gartley=DEFAULT_SHOW_GARTLEY, show_bat=DEFAULT_SHOW_BAT, show_butterfly=DEFAULT_SHOW_BUTTERFLY, show_crab=DEFAULT_SHOW_CRAB, show_deep=DEFAULT_SHOW_DEEP, show_hs=DEFAULT_SHOW_HS, kernel=None))]
pub fn vdubus_divergence_wave_pattern_generator_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    fast_depth_range: (usize, usize, usize),
    slow_depth_range: (usize, usize, usize),
    fast_length_range: (usize, usize, usize),
    slow_length_range: (usize, usize, usize),
    signal_length_range: (usize, usize, usize),
    lookback_range: (usize, usize, usize),
    err_tol_range: (f64, f64, f64),
    show_standard: bool,
    show_climax: bool,
    show_rounded: bool,
    show_predator: bool,
    show_gartley: bool,
    show_bat: bool,
    show_butterfly: bool,
    show_crab: bool,
    show_deep: bool,
    show_hs: bool,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let sweep = VdubusDivergenceWavePatternGeneratorBatchRange {
        fast_depth: fast_depth_range,
        slow_depth: slow_depth_range,
        fast_length: fast_length_range,
        slow_length: slow_length_range,
        signal_length: signal_length_range,
        lookback: lookback_range,
        err_tol: err_tol_range,
        show_standard,
        show_climax,
        show_rounded,
        show_predator,
        show_gartley,
        show_bat,
        show_butterfly,
        show_crab,
        show_deep,
        show_hs,
    };
    let _ = validate_kernel(kernel, true)?;
    let output = py
        .allow_threads(|| {
            vdubus_divergence_wave_pattern_generator_batch_with_kernel(
                high_slice,
                low_slice,
                close_slice,
                &sweep,
                Kernel::Auto,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dict = PyDict::new(py);
    dict.set_item(
        "fast_standard",
        PyArray1::from_vec(py, output.fast_standard).reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "fast_climax",
        PyArray1::from_vec(py, output.fast_climax).reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "fast_rounded",
        PyArray1::from_vec(py, output.fast_rounded).reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "fast_predator",
        PyArray1::from_vec(py, output.fast_predator).reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "slow_standard",
        PyArray1::from_vec(py, output.slow_standard).reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "slow_climax",
        PyArray1::from_vec(py, output.slow_climax).reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "slow_rounded",
        PyArray1::from_vec(py, output.slow_rounded).reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "slow_predator",
        PyArray1::from_vec(py, output.slow_predator).reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "opposing_force",
        PyArray1::from_vec(py, output.opposing_force).reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "macd",
        PyArray1::from_vec(py, output.macd).reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "signal",
        PyArray1::from_vec(py, output.signal).reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "hist",
        PyArray1::from_vec(py, output.hist).reshape((output.rows, output.cols))?,
    )?;
    dict.set_item("rows", output.rows)?;
    dict.set_item("cols", output.cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "VdubusDivergenceWavePatternGeneratorStream")]
pub struct VdubusDivergenceWavePatternGeneratorStreamPy {
    stream: VdubusDivergenceWavePatternGeneratorStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl VdubusDivergenceWavePatternGeneratorStreamPy {
    #[new]
    #[pyo3(signature = (fast_depth=DEFAULT_FAST_DEPTH, slow_depth=DEFAULT_SLOW_DEPTH, fast_length=DEFAULT_FAST_LENGTH, slow_length=DEFAULT_SLOW_LENGTH, signal_length=DEFAULT_SIGNAL_LENGTH, lookback=DEFAULT_LOOKBACK, err_tol=DEFAULT_ERR_TOL, show_standard=DEFAULT_SHOW_STANDARD, show_climax=DEFAULT_SHOW_CLIMAX, show_rounded=DEFAULT_SHOW_ROUNDED, show_predator=DEFAULT_SHOW_PREDATOR, show_gartley=DEFAULT_SHOW_GARTLEY, show_bat=DEFAULT_SHOW_BAT, show_butterfly=DEFAULT_SHOW_BUTTERFLY, show_crab=DEFAULT_SHOW_CRAB, show_deep=DEFAULT_SHOW_DEEP, show_hs=DEFAULT_SHOW_HS))]
    fn new(
        fast_depth: usize,
        slow_depth: usize,
        fast_length: usize,
        slow_length: usize,
        signal_length: usize,
        lookback: usize,
        err_tol: f64,
        show_standard: bool,
        show_climax: bool,
        show_rounded: bool,
        show_predator: bool,
        show_gartley: bool,
        show_bat: bool,
        show_butterfly: bool,
        show_crab: bool,
        show_deep: bool,
        show_hs: bool,
    ) -> PyResult<Self> {
        let stream = VdubusDivergenceWavePatternGeneratorStream::try_new(
            VdubusDivergenceWavePatternGeneratorParams {
                fast_depth: Some(fast_depth),
                slow_depth: Some(slow_depth),
                fast_length: Some(fast_length),
                slow_length: Some(slow_length),
                signal_length: Some(signal_length),
                lookback: Some(lookback),
                err_tol: Some(err_tol),
                show_standard: Some(show_standard),
                show_climax: Some(show_climax),
                show_rounded: Some(show_rounded),
                show_predator: Some(show_predator),
                show_gartley: Some(show_gartley),
                show_bat: Some(show_bat),
                show_butterfly: Some(show_butterfly),
                show_crab: Some(show_crab),
                show_deep: Some(show_deep),
                show_hs: Some(show_hs),
            },
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(
        &mut self,
        high: f64,
        low: f64,
        close: f64,
    ) -> Option<(f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64)> {
        self.stream.update(high, low, close)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct VdubusDivergenceWavePatternGeneratorBatchConfig {
    pub fast_depth_range: (usize, usize, usize),
    pub slow_depth_range: (usize, usize, usize),
    pub fast_length_range: (usize, usize, usize),
    pub slow_length_range: (usize, usize, usize),
    pub signal_length_range: (usize, usize, usize),
    pub lookback_range: (usize, usize, usize),
    pub err_tol_range: (f64, f64, f64),
    pub show_standard: bool,
    pub show_climax: bool,
    pub show_rounded: bool,
    pub show_predator: bool,
    pub show_gartley: bool,
    pub show_bat: bool,
    pub show_butterfly: bool,
    pub show_crab: bool,
    pub show_deep: bool,
    pub show_hs: bool,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct VdubusDivergenceWavePatternGeneratorBatchJsOutput {
    pub fast_standard: Vec<f64>,
    pub fast_climax: Vec<f64>,
    pub fast_rounded: Vec<f64>,
    pub fast_predator: Vec<f64>,
    pub slow_standard: Vec<f64>,
    pub slow_climax: Vec<f64>,
    pub slow_rounded: Vec<f64>,
    pub slow_predator: Vec<f64>,
    pub opposing_force: Vec<f64>,
    pub macd: Vec<f64>,
    pub signal: Vec<f64>,
    pub hist: Vec<f64>,
    pub combos: Vec<VdubusDivergenceWavePatternGeneratorParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn vdubus_divergence_wave_pattern_generator_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    fast_depth: usize,
    slow_depth: usize,
    fast_length: usize,
    slow_length: usize,
    signal_length: usize,
    lookback: usize,
    err_tol: f64,
    show_standard: bool,
    show_climax: bool,
    show_rounded: bool,
    show_predator: bool,
    show_gartley: bool,
    show_bat: bool,
    show_butterfly: bool,
    show_crab: bool,
    show_deep: bool,
    show_hs: bool,
) -> Result<JsValue, JsValue> {
    let input = VdubusDivergenceWavePatternGeneratorInput::from_slices(
        high,
        low,
        close,
        VdubusDivergenceWavePatternGeneratorParams {
            fast_depth: Some(fast_depth),
            slow_depth: Some(slow_depth),
            fast_length: Some(fast_length),
            slow_length: Some(slow_length),
            signal_length: Some(signal_length),
            lookback: Some(lookback),
            err_tol: Some(err_tol),
            show_standard: Some(show_standard),
            show_climax: Some(show_climax),
            show_rounded: Some(show_rounded),
            show_predator: Some(show_predator),
            show_gartley: Some(show_gartley),
            show_bat: Some(show_bat),
            show_butterfly: Some(show_butterfly),
            show_crab: Some(show_crab),
            show_deep: Some(show_deep),
            show_hs: Some(show_hs),
        },
    );
    let output = vdubus_divergence_wave_pattern_generator_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn vdubus_divergence_wave_pattern_generator_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn vdubus_divergence_wave_pattern_generator_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn vdubus_divergence_wave_pattern_generator_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    fast_standard_ptr: *mut f64,
    fast_climax_ptr: *mut f64,
    fast_rounded_ptr: *mut f64,
    fast_predator_ptr: *mut f64,
    slow_standard_ptr: *mut f64,
    slow_climax_ptr: *mut f64,
    slow_rounded_ptr: *mut f64,
    slow_predator_ptr: *mut f64,
    opposing_force_ptr: *mut f64,
    macd_ptr: *mut f64,
    signal_ptr: *mut f64,
    hist_ptr: *mut f64,
    len: usize,
    fast_depth: usize,
    slow_depth: usize,
    fast_length: usize,
    slow_length: usize,
    signal_length: usize,
    lookback: usize,
    err_tol: f64,
    show_standard: bool,
    show_climax: bool,
    show_rounded: bool,
    show_predator: bool,
    show_gartley: bool,
    show_bat: bool,
    show_butterfly: bool,
    show_crab: bool,
    show_deep: bool,
    show_hs: bool,
) -> Result<(), JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || fast_standard_ptr.is_null()
        || fast_climax_ptr.is_null()
        || fast_rounded_ptr.is_null()
        || fast_predator_ptr.is_null()
        || slow_standard_ptr.is_null()
        || slow_climax_ptr.is_null()
        || slow_rounded_ptr.is_null()
        || slow_predator_ptr.is_null()
        || opposing_force_ptr.is_null()
        || macd_ptr.is_null()
        || signal_ptr.is_null()
        || hist_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let input = VdubusDivergenceWavePatternGeneratorInput::from_slices(
            std::slice::from_raw_parts(high_ptr, len),
            std::slice::from_raw_parts(low_ptr, len),
            std::slice::from_raw_parts(close_ptr, len),
            VdubusDivergenceWavePatternGeneratorParams {
                fast_depth: Some(fast_depth),
                slow_depth: Some(slow_depth),
                fast_length: Some(fast_length),
                slow_length: Some(slow_length),
                signal_length: Some(signal_length),
                lookback: Some(lookback),
                err_tol: Some(err_tol),
                show_standard: Some(show_standard),
                show_climax: Some(show_climax),
                show_rounded: Some(show_rounded),
                show_predator: Some(show_predator),
                show_gartley: Some(show_gartley),
                show_bat: Some(show_bat),
                show_butterfly: Some(show_butterfly),
                show_crab: Some(show_crab),
                show_deep: Some(show_deep),
                show_hs: Some(show_hs),
            },
        );
        vdubus_divergence_wave_pattern_generator_into_slice(
            std::slice::from_raw_parts_mut(fast_standard_ptr, len),
            std::slice::from_raw_parts_mut(fast_climax_ptr, len),
            std::slice::from_raw_parts_mut(fast_rounded_ptr, len),
            std::slice::from_raw_parts_mut(fast_predator_ptr, len),
            std::slice::from_raw_parts_mut(slow_standard_ptr, len),
            std::slice::from_raw_parts_mut(slow_climax_ptr, len),
            std::slice::from_raw_parts_mut(slow_rounded_ptr, len),
            std::slice::from_raw_parts_mut(slow_predator_ptr, len),
            std::slice::from_raw_parts_mut(opposing_force_ptr, len),
            std::slice::from_raw_parts_mut(macd_ptr, len),
            std::slice::from_raw_parts_mut(signal_ptr, len),
            std::slice::from_raw_parts_mut(hist_ptr, len),
            &input,
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = vdubus_divergence_wave_pattern_generator_batch)]
pub fn vdubus_divergence_wave_pattern_generator_batch_unified_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: VdubusDivergenceWavePatternGeneratorBatchConfig =
        serde_wasm_bindgen::from_value(config)
            .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
    let sweep = VdubusDivergenceWavePatternGeneratorBatchRange {
        fast_depth: config.fast_depth_range,
        slow_depth: config.slow_depth_range,
        fast_length: config.fast_length_range,
        slow_length: config.slow_length_range,
        signal_length: config.signal_length_range,
        lookback: config.lookback_range,
        err_tol: config.err_tol_range,
        show_standard: config.show_standard,
        show_climax: config.show_climax,
        show_rounded: config.show_rounded,
        show_predator: config.show_predator,
        show_gartley: config.show_gartley,
        show_bat: config.show_bat,
        show_butterfly: config.show_butterfly,
        show_crab: config.show_crab,
        show_deep: config.show_deep,
        show_hs: config.show_hs,
    };
    let output = vdubus_divergence_wave_pattern_generator_batch_with_kernel(
        high,
        low,
        close,
        &sweep,
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&VdubusDivergenceWavePatternGeneratorBatchJsOutput {
        fast_standard: output.fast_standard,
        fast_climax: output.fast_climax,
        fast_rounded: output.fast_rounded,
        fast_predator: output.fast_predator,
        slow_standard: output.slow_standard,
        slow_climax: output.slow_climax,
        slow_rounded: output.slow_rounded,
        slow_predator: output.slow_predator,
        opposing_force: output.opposing_force,
        macd: output.macd,
        signal: output.signal,
        hist: output.hist,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn vdubus_divergence_wave_pattern_generator_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    fast_standard_ptr: *mut f64,
    fast_climax_ptr: *mut f64,
    fast_rounded_ptr: *mut f64,
    fast_predator_ptr: *mut f64,
    slow_standard_ptr: *mut f64,
    slow_climax_ptr: *mut f64,
    slow_rounded_ptr: *mut f64,
    slow_predator_ptr: *mut f64,
    opposing_force_ptr: *mut f64,
    macd_ptr: *mut f64,
    signal_ptr: *mut f64,
    hist_ptr: *mut f64,
    len: usize,
    fast_depth_start: usize,
    fast_depth_end: usize,
    fast_depth_step: usize,
    slow_depth_start: usize,
    slow_depth_end: usize,
    slow_depth_step: usize,
    fast_length_start: usize,
    fast_length_end: usize,
    fast_length_step: usize,
    slow_length_start: usize,
    slow_length_end: usize,
    slow_length_step: usize,
    signal_length_start: usize,
    signal_length_end: usize,
    signal_length_step: usize,
    lookback_start: usize,
    lookback_end: usize,
    lookback_step: usize,
    err_tol_start: f64,
    err_tol_end: f64,
    err_tol_step: f64,
    show_standard: bool,
    show_climax: bool,
    show_rounded: bool,
    show_predator: bool,
    show_gartley: bool,
    show_bat: bool,
    show_butterfly: bool,
    show_crab: bool,
    show_deep: bool,
    show_hs: bool,
) -> Result<usize, JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || fast_standard_ptr.is_null()
        || fast_climax_ptr.is_null()
        || fast_rounded_ptr.is_null()
        || fast_predator_ptr.is_null()
        || slow_standard_ptr.is_null()
        || slow_climax_ptr.is_null()
        || slow_rounded_ptr.is_null()
        || slow_predator_ptr.is_null()
        || opposing_force_ptr.is_null()
        || macd_ptr.is_null()
        || signal_ptr.is_null()
        || hist_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }
    let sweep = VdubusDivergenceWavePatternGeneratorBatchRange {
        fast_depth: (fast_depth_start, fast_depth_end, fast_depth_step),
        slow_depth: (slow_depth_start, slow_depth_end, slow_depth_step),
        fast_length: (fast_length_start, fast_length_end, fast_length_step),
        slow_length: (slow_length_start, slow_length_end, slow_length_step),
        signal_length: (signal_length_start, signal_length_end, signal_length_step),
        lookback: (lookback_start, lookback_end, lookback_step),
        err_tol: (err_tol_start, err_tol_end, err_tol_step),
        show_standard,
        show_climax,
        show_rounded,
        show_predator,
        show_gartley,
        show_bat,
        show_butterfly,
        show_crab,
        show_deep,
        show_hs,
    };
    let rows = expand_grid_vdubus_divergence_wave_pattern_generator(&sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?
        .len();
    unsafe {
        vdubus_divergence_wave_pattern_generator_batch_inner_into(
            std::slice::from_raw_parts(high_ptr, len),
            std::slice::from_raw_parts(low_ptr, len),
            std::slice::from_raw_parts(close_ptr, len),
            &sweep,
            false,
            std::slice::from_raw_parts_mut(fast_standard_ptr, rows * len),
            std::slice::from_raw_parts_mut(fast_climax_ptr, rows * len),
            std::slice::from_raw_parts_mut(fast_rounded_ptr, rows * len),
            std::slice::from_raw_parts_mut(fast_predator_ptr, rows * len),
            std::slice::from_raw_parts_mut(slow_standard_ptr, rows * len),
            std::slice::from_raw_parts_mut(slow_climax_ptr, rows * len),
            std::slice::from_raw_parts_mut(slow_rounded_ptr, rows * len),
            std::slice::from_raw_parts_mut(slow_predator_ptr, rows * len),
            std::slice::from_raw_parts_mut(opposing_force_ptr, rows * len),
            std::slice::from_raw_parts_mut(macd_ptr, rows * len),
            std::slice::from_raw_parts_mut(signal_ptr, rows * len),
            std::slice::from_raw_parts_mut(hist_ptr, rows * len),
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn oscillating_data(size: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut high = Vec::with_capacity(size);
        let mut low = Vec::with_capacity(size);
        let mut close = Vec::with_capacity(size);
        for i in 0..size {
            let x = i as f64;
            let c = 100.0 + (x * 0.19).sin() * 12.0 + (x * 0.043).cos() * 5.0 + 0.03 * x;
            close.push(c);
            high.push(c + 1.4 + ((i % 5) as f64) * 0.07);
            low.push(c - 1.3 - ((i % 3) as f64) * 0.05);
        }
        (high, low, close)
    }

    #[test]
    fn harmonic_ratio_fix_enables_named_patterns() {
        assert_eq!(harmonic_family_code(0.5, 1.618, 0.15), FAMILY_CRAB);
        assert_eq!(harmonic_family_code(0.786, 1.27, 0.15), FAMILY_BUTTERFLY);
    }

    #[test]
    fn vdubus_outputs_finite_momentum() -> Result<(), Box<dyn StdError>> {
        let (high, low, close) = oscillating_data(320);
        let input = VdubusDivergenceWavePatternGeneratorInput::from_slices(
            &high,
            &low,
            &close,
            VdubusDivergenceWavePatternGeneratorParams::default(),
        );
        let output = vdubus_divergence_wave_pattern_generator(&input)?;
        assert!(output.hist.iter().any(|v| v.is_finite()));
        assert!(output.opposing_force.iter().any(|v| v.is_finite()));
        Ok(())
    }

    #[test]
    fn vdubus_nan_gap_restart() -> Result<(), Box<dyn StdError>> {
        let (mut high, mut low, mut close) = oscillating_data(260);
        high[140] = f64::NAN;
        low[140] = f64::NAN;
        close[140] = f64::NAN;
        let input = VdubusDivergenceWavePatternGeneratorInput::from_slices(
            &high,
            &low,
            &close,
            VdubusDivergenceWavePatternGeneratorParams::default(),
        );
        let output = vdubus_divergence_wave_pattern_generator(&input)?;
        let restart_end = (140 + required_valid_bars(DEFAULT_SLOW_LENGTH, DEFAULT_SIGNAL_LENGTH))
            .min(output.hist.len());
        for i in 140..restart_end {
            assert!(output.hist[i].is_nan());
        }
        Ok(())
    }

    #[test]
    fn vdubus_stream_matches_batch() -> Result<(), Box<dyn StdError>> {
        let (high, low, close) = oscillating_data(240);
        let params = VdubusDivergenceWavePatternGeneratorParams::default();
        let batch = vdubus_divergence_wave_pattern_generator(
            &VdubusDivergenceWavePatternGeneratorInput::from_slices(
                &high,
                &low,
                &close,
                params.clone(),
            ),
        )?;
        let mut stream = VdubusDivergenceWavePatternGeneratorStream::try_new(params)?;
        for i in 0..close.len() {
            let streamed = stream.update(high[i], low[i], close[i]);
            if let Some(values) = streamed {
                assert_eq!(values.9, batch.macd[i]);
                assert_eq!(values.10, batch.signal[i]);
                assert_eq!(values.11, batch.hist[i]);
            } else {
                assert!(batch.hist[i].is_nan());
            }
        }
        Ok(())
    }

    #[test]
    fn vdubus_invalid_period_errors() {
        let (high, low, close) = oscillating_data(32);
        let input = VdubusDivergenceWavePatternGeneratorInput::from_slices(
            &high,
            &low,
            &close,
            VdubusDivergenceWavePatternGeneratorParams {
                slow_length: Some(0),
                ..Default::default()
            },
        );
        assert!(matches!(
            vdubus_divergence_wave_pattern_generator(&input),
            Err(VdubusDivergenceWavePatternGeneratorError::InvalidPeriods { .. })
        ));
    }
}
