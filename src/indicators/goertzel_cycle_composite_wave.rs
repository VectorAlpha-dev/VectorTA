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

use crate::utilities::data_loader::{source_type, Candles};
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
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

const DEFAULT_MAX_PERIOD: usize = 120;
const DEFAULT_START_AT_CYCLE: usize = 1;
const DEFAULT_USE_TOP_CYCLES: usize = 2;
const DEFAULT_BAR_TO_CALCULATE: usize = 1;
const DEFAULT_DT_ZL_PER1: usize = 10;
const DEFAULT_DT_ZL_PER2: usize = 40;
const DEFAULT_DT_HP_PER1: usize = 20;
const DEFAULT_DT_HP_PER2: usize = 80;
const DEFAULT_DT_REG_ZL_SMOOTH_PER: usize = 5;
const DEFAULT_HP_SMOOTH_PER: usize = 20;
const DEFAULT_ZLMA_SMOOTH_PER: usize = 10;
const DEFAULT_BART_NO_CYCLES: usize = 5;
const DEFAULT_BART_SMOOTH_PER: usize = 2;
const DEFAULT_BART_SIG_LIMIT: usize = 50;

impl<'a> AsRef<[f64]> for GoertzelCycleCompositeWaveInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            GoertzelCycleCompositeWaveData::Slice(slice) => slice,
            GoertzelCycleCompositeWaveData::Candles { candles, source } => {
                source_type(candles, source)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum GoertzelCycleCompositeWaveData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct GoertzelCycleCompositeWaveOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    serde(rename_all = "snake_case")
)]
pub enum GoertzelDetrendMode {
    None,
    HodrickPrescottSmoothing,
    ZeroLagSmoothing,
    HodrickPrescottDetrending,
    ZeroLagDetrending,
    LogZeroLagRegressionDetrending,
}

impl Default for GoertzelDetrendMode {
    fn default() -> Self {
        Self::HodrickPrescottDetrending
    }
}

impl GoertzelDetrendMode {
    #[inline(always)]
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "none" => Some(Self::None),
            "hodrick_prescott_smoothing" | "hp_smoothing" | "hpsmth" => {
                Some(Self::HodrickPrescottSmoothing)
            }
            "zero_lag_smoothing" | "zl_smoothing" | "zlagsmth" => Some(Self::ZeroLagSmoothing),
            "hodrick_prescott_detrending" | "hp_detrending" | "hpsmthdt" => {
                Some(Self::HodrickPrescottDetrending)
            }
            "zero_lag_detrending" | "zl_detrending" | "zlagsmthdt" => Some(Self::ZeroLagDetrending),
            "log_zero_lag_regression_detrending" | "log_zl_regression" | "logzlagregression" => {
                Some(Self::LogZeroLagRegressionDetrending)
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct GoertzelCycleCompositeWaveParams {
    pub max_period: Option<usize>,
    pub start_at_cycle: Option<usize>,
    pub use_top_cycles: Option<usize>,
    pub bar_to_calculate: Option<usize>,
    pub detrend_mode: Option<GoertzelDetrendMode>,
    pub dt_zl_per1: Option<usize>,
    pub dt_zl_per2: Option<usize>,
    pub dt_hp_per1: Option<usize>,
    pub dt_hp_per2: Option<usize>,
    pub dt_reg_zl_smooth_per: Option<usize>,
    pub hp_smooth_per: Option<usize>,
    pub zlma_smooth_per: Option<usize>,
    pub filter_bartels: Option<bool>,
    pub bart_no_cycles: Option<usize>,
    pub bart_smooth_per: Option<usize>,
    pub bart_sig_limit: Option<usize>,
    pub sort_bartels: Option<bool>,
    pub squared_amp: Option<bool>,
    pub use_cosine: Option<bool>,
    pub subtract_noise: Option<bool>,
    pub use_cycle_strength: Option<bool>,
}

impl Default for GoertzelCycleCompositeWaveParams {
    fn default() -> Self {
        Self {
            max_period: Some(DEFAULT_MAX_PERIOD),
            start_at_cycle: Some(DEFAULT_START_AT_CYCLE),
            use_top_cycles: Some(DEFAULT_USE_TOP_CYCLES),
            bar_to_calculate: Some(DEFAULT_BAR_TO_CALCULATE),
            detrend_mode: Some(GoertzelDetrendMode::HodrickPrescottDetrending),
            dt_zl_per1: Some(DEFAULT_DT_ZL_PER1),
            dt_zl_per2: Some(DEFAULT_DT_ZL_PER2),
            dt_hp_per1: Some(DEFAULT_DT_HP_PER1),
            dt_hp_per2: Some(DEFAULT_DT_HP_PER2),
            dt_reg_zl_smooth_per: Some(DEFAULT_DT_REG_ZL_SMOOTH_PER),
            hp_smooth_per: Some(DEFAULT_HP_SMOOTH_PER),
            zlma_smooth_per: Some(DEFAULT_ZLMA_SMOOTH_PER),
            filter_bartels: Some(false),
            bart_no_cycles: Some(DEFAULT_BART_NO_CYCLES),
            bart_smooth_per: Some(DEFAULT_BART_SMOOTH_PER),
            bart_sig_limit: Some(DEFAULT_BART_SIG_LIMIT),
            sort_bartels: Some(false),
            squared_amp: Some(true),
            use_cosine: Some(true),
            subtract_noise: Some(false),
            use_cycle_strength: Some(true),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GoertzelCycleCompositeWaveInput<'a> {
    pub data: GoertzelCycleCompositeWaveData<'a>,
    pub params: GoertzelCycleCompositeWaveParams,
}

impl<'a> GoertzelCycleCompositeWaveInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: GoertzelCycleCompositeWaveParams,
    ) -> Self {
        Self {
            data: GoertzelCycleCompositeWaveData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(data: &'a [f64], params: GoertzelCycleCompositeWaveParams) -> Self {
        Self {
            data: GoertzelCycleCompositeWaveData::Slice(data),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(
            candles,
            "close",
            GoertzelCycleCompositeWaveParams::default(),
        )
    }
}

#[derive(Copy, Clone, Debug)]
pub struct GoertzelCycleCompositeWaveBuilder {
    params: GoertzelCycleCompositeWaveParams,
    kernel: Kernel,
}

impl Default for GoertzelCycleCompositeWaveBuilder {
    fn default() -> Self {
        Self {
            params: GoertzelCycleCompositeWaveParams::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl GoertzelCycleCompositeWaveBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn params(mut self, value: GoertzelCycleCompositeWaveParams) -> Self {
        self.params = value;
        self
    }

    #[inline(always)]
    pub fn max_period(mut self, value: usize) -> Self {
        self.params.max_period = Some(value);
        self
    }

    #[inline(always)]
    pub fn start_at_cycle(mut self, value: usize) -> Self {
        self.params.start_at_cycle = Some(value);
        self
    }

    #[inline(always)]
    pub fn use_top_cycles(mut self, value: usize) -> Self {
        self.params.use_top_cycles = Some(value);
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
    ) -> Result<GoertzelCycleCompositeWaveOutput, GoertzelCycleCompositeWaveError> {
        goertzel_cycle_composite_wave_with_kernel(
            &GoertzelCycleCompositeWaveInput::from_candles(candles, "close", self.params),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<GoertzelCycleCompositeWaveOutput, GoertzelCycleCompositeWaveError> {
        goertzel_cycle_composite_wave_with_kernel(
            &GoertzelCycleCompositeWaveInput::from_slice(data, self.params),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<GoertzelCycleCompositeWaveStream, GoertzelCycleCompositeWaveError> {
        GoertzelCycleCompositeWaveStream::try_new(self.params)
    }
}

#[derive(Debug, Error)]
pub enum GoertzelCycleCompositeWaveError {
    #[error("goertzel_cycle_composite_wave: Input data slice is empty.")]
    EmptyInputData,
    #[error("goertzel_cycle_composite_wave: All values are NaN.")]
    AllValuesNaN,
    #[error("goertzel_cycle_composite_wave: Invalid parameter {name}: {value}")]
    InvalidParameter { name: &'static str, value: usize },
    #[error(
        "goertzel_cycle_composite_wave: Not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "goertzel_cycle_composite_wave: Output length mismatch: expected = {expected}, got = {got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("goertzel_cycle_composite_wave: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("goertzel_cycle_composite_wave: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error(
        "goertzel_cycle_composite_wave: Output length mismatch: dst = {dst_len}, expected = {expected_len}"
    )]
    MismatchedOutputLen { dst_len: usize, expected_len: usize },
    #[error("goertzel_cycle_composite_wave: Invalid input: {msg}")]
    InvalidInput { msg: String },
    #[error("goertzel_cycle_composite_wave: Invalid detrend mode: {0}")]
    InvalidDetrendMode(String),
}

#[derive(Debug, Clone, Copy)]
struct CycleInfo {
    cycle: usize,
    amplitude: f64,
    phase: f64,
    bartels: f64,
}

#[inline(always)]
fn sample_size_for_params(params: &GoertzelCycleCompositeWaveParams) -> usize {
    let max_period = params.max_period.unwrap_or(DEFAULT_MAX_PERIOD);
    let bar_to_calculate = params.bar_to_calculate.unwrap_or(DEFAULT_BAR_TO_CALCULATE);
    let bart_no_cycles = params.bart_no_cycles.unwrap_or(DEFAULT_BART_NO_CYCLES);
    let cycle_span = (2 * max_period).max(bart_no_cycles.saturating_mul(max_period));
    cycle_span.saturating_add(bar_to_calculate)
}

#[inline(always)]
fn longest_valid_run(data: &[f64]) -> usize {
    let mut best = 0usize;
    let mut cur = 0usize;
    for &value in data {
        if value.is_finite() {
            cur += 1;
            best = best.max(cur);
        } else {
            cur = 0;
        }
    }
    best
}

#[inline(always)]
fn validate_positive(
    name: &'static str,
    value: usize,
) -> Result<(), GoertzelCycleCompositeWaveError> {
    if value == 0 {
        return Err(GoertzelCycleCompositeWaveError::InvalidParameter { name, value });
    }
    Ok(())
}

#[inline(always)]
fn validate_hp_period(
    name: &'static str,
    value: usize,
) -> Result<(), GoertzelCycleCompositeWaveError> {
    if value < 2 {
        return Err(GoertzelCycleCompositeWaveError::InvalidParameter { name, value });
    }
    Ok(())
}

#[inline(always)]
fn validate_params(
    params: &GoertzelCycleCompositeWaveParams,
) -> Result<(), GoertzelCycleCompositeWaveError> {
    validate_hp_period(
        "max_period",
        params.max_period.unwrap_or(DEFAULT_MAX_PERIOD),
    )?;
    validate_positive(
        "start_at_cycle",
        params.start_at_cycle.unwrap_or(DEFAULT_START_AT_CYCLE),
    )?;
    validate_positive(
        "use_top_cycles",
        params.use_top_cycles.unwrap_or(DEFAULT_USE_TOP_CYCLES),
    )?;
    validate_positive(
        "dt_zl_per1",
        params.dt_zl_per1.unwrap_or(DEFAULT_DT_ZL_PER1),
    )?;
    validate_positive(
        "dt_zl_per2",
        params.dt_zl_per2.unwrap_or(DEFAULT_DT_ZL_PER2),
    )?;
    validate_hp_period(
        "dt_hp_per1",
        params.dt_hp_per1.unwrap_or(DEFAULT_DT_HP_PER1),
    )?;
    validate_hp_period(
        "dt_hp_per2",
        params.dt_hp_per2.unwrap_or(DEFAULT_DT_HP_PER2),
    )?;
    validate_positive(
        "dt_reg_zl_smooth_per",
        params
            .dt_reg_zl_smooth_per
            .unwrap_or(DEFAULT_DT_REG_ZL_SMOOTH_PER),
    )?;
    validate_hp_period(
        "hp_smooth_per",
        params.hp_smooth_per.unwrap_or(DEFAULT_HP_SMOOTH_PER),
    )?;
    validate_positive(
        "zlma_smooth_per",
        params.zlma_smooth_per.unwrap_or(DEFAULT_ZLMA_SMOOTH_PER),
    )?;
    validate_positive(
        "bart_no_cycles",
        params.bart_no_cycles.unwrap_or(DEFAULT_BART_NO_CYCLES),
    )?;
    validate_positive(
        "bart_smooth_per",
        params.bart_smooth_per.unwrap_or(DEFAULT_BART_SMOOTH_PER),
    )?;
    Ok(())
}

#[inline(always)]
fn validate_common(
    data: &[f64],
    params: &GoertzelCycleCompositeWaveParams,
) -> Result<usize, GoertzelCycleCompositeWaveError> {
    if data.is_empty() {
        return Err(GoertzelCycleCompositeWaveError::EmptyInputData);
    }
    validate_params(params)?;
    let max_run = longest_valid_run(data);
    if max_run == 0 {
        return Err(GoertzelCycleCompositeWaveError::AllValuesNaN);
    }
    let needed = sample_size_for_params(params);
    if max_run < needed {
        return Err(GoertzelCycleCompositeWaveError::NotEnoughValidData {
            needed,
            valid: max_run,
        });
    }
    Ok(needed)
}

#[inline(always)]
fn hp_lambda(period: usize) -> f64 {
    0.0625 / (std::f64::consts::PI / period as f64).sin().powi(4)
}

fn zero_lag_ma(src: &[f64], smooth_per: usize) -> Vec<f64> {
    let bars_taken = src.len();
    let mut lwma1 = vec![0.0; bars_taken];
    let mut output = vec![0.0; bars_taken];

    for i in (0..bars_taken).rev() {
        let mut sum = 0.0;
        let mut sumw = 0.0;
        for k in 0..smooth_per {
            let idx = i + k;
            if idx < bars_taken {
                let weight = (smooth_per - k) as f64;
                sumw += weight;
                sum += weight * src[idx];
            }
        }
        lwma1[i] = if sumw != 0.0 { sum / sumw } else { 0.0 };
    }

    for i in 0..bars_taken {
        let mut sum = 0.0;
        let mut sumw = 0.0;
        for k in 0..smooth_per {
            if i >= k {
                let weight = (smooth_per - k) as f64;
                sumw += weight;
                sum += weight * lwma1[i - k];
            }
        }
        output[i] = if sumw != 0.0 { sum / sumw } else { 0.0 };
    }

    output
}

fn hodrick_prescott_filter(src: &[f64], lambda: f64) -> Vec<f64> {
    let per = src.len();
    let mut a = vec![0.0; per];
    let mut b = vec![0.0; per];
    let mut c = vec![0.0; per];
    let mut output = src.to_vec();

    if per == 0 {
        return output;
    }

    a[0] = 1.0 + lambda;
    b[0] = -2.0 * lambda;
    c[0] = lambda;
    for i in 1..per.saturating_sub(2) {
        a[i] = 6.0 * lambda + 1.0;
        b[i] = -4.0 * lambda;
        c[i] = lambda;
    }
    if per > 1 {
        a[1] = 5.0 * lambda + 1.0;
        a[per - 2] = 5.0 * lambda + 1.0;
        a[per - 1] = 1.0 + lambda;
        b[per - 2] = -2.0 * lambda;
    }

    let mut h1 = 0.0;
    let mut h2 = 0.0;
    let mut h3 = 0.0;
    let mut h4 = 0.0;
    let mut h5 = 0.0;
    let mut hh1 = 0.0;
    let mut hh2 = 0.0;
    let mut hh3 = 0.0;
    let mut hh5 = 0.0;

    for i in 0..per {
        let z = a[i] - h4 * h1 - hh5 * h2;
        if z.abs() <= f64::EPSILON {
            break;
        }
        let hb = b[i];
        hh1 = h1;
        h1 = (hb - h4 * h2) / z;
        b[i] = h1;
        let hc = c[i];
        hh2 = h2;
        h2 = hc / z;
        c[i] = h2;
        a[i] = (src[i] - hh3 * hh5 - h3 * h4) / z;
        hh3 = h3;
        h3 = a[i];
        h4 = hb - h5 * hh1;
        hh5 = h5;
        h5 = hc;
    }

    let mut h1b = a[per - 1];
    let mut h2b = 0.0;
    output[per - 1] = h1b;
    for i in (0..per.saturating_sub(1)).rev() {
        output[i] = a[i] - b[i] * h1b - c[i] * h2b;
        h2b = h1b;
        h1b = output[i];
    }

    output
}

fn detrend_ln_zero_lag_regression(src: &[f64], smooth_per: usize) -> Option<Vec<f64>> {
    let mut calc_values = zero_lag_ma(src, smooth_per);
    for value in &mut calc_values {
        if *value <= 0.0 || !value.is_finite() {
            return None;
        }
        *value = value.ln() * 100.0;
    }

    let bars_taken = calc_values.len();
    let mut sumy = 0.0;
    let mut sumx = 0.0;
    let mut sumxy = 0.0;
    let mut sumx2 = 0.0;
    for (i, &value) in calc_values.iter().enumerate() {
        let x = i as f64;
        sumy += value;
        sumx += x;
        sumxy += x * value;
        sumx2 += x * x;
    }

    let denom = sumx2 * bars_taken as f64 - sumx * sumx;
    if denom.abs() <= f64::EPSILON {
        return None;
    }
    let slope = (sumxy * bars_taken as f64 - sumx * sumy) / denom;
    let intercept = (sumy - sumx * slope) / bars_taken as f64;

    let mut output = vec![0.0; bars_taken];
    for i in 0..bars_taken {
        output[i] = calc_values[i] - (intercept + slope * i as f64);
    }
    Some(output)
}

fn apply_detrend_mode(
    src_rev: &[f64],
    params: &GoertzelCycleCompositeWaveParams,
) -> Option<Vec<f64>> {
    let mode = params
        .detrend_mode
        .unwrap_or(GoertzelDetrendMode::HodrickPrescottDetrending);
    let out = match mode {
        GoertzelDetrendMode::None => src_rev.to_vec(),
        GoertzelDetrendMode::HodrickPrescottSmoothing => {
            hodrick_prescott_filter(src_rev, hp_lambda(params.hp_smooth_per.unwrap_or(20)))
        }
        GoertzelDetrendMode::ZeroLagSmoothing => {
            zero_lag_ma(src_rev, params.zlma_smooth_per.unwrap_or(10))
        }
        GoertzelDetrendMode::HodrickPrescottDetrending => {
            let fast = hodrick_prescott_filter(src_rev, hp_lambda(params.dt_hp_per1.unwrap_or(20)));
            let slow = hodrick_prescott_filter(src_rev, hp_lambda(params.dt_hp_per2.unwrap_or(80)));
            fast.iter().zip(slow.iter()).map(|(a, b)| a - b).collect()
        }
        GoertzelDetrendMode::ZeroLagDetrending => {
            let fast = zero_lag_ma(src_rev, params.dt_zl_per1.unwrap_or(10));
            let slow = zero_lag_ma(src_rev, params.dt_zl_per2.unwrap_or(40));
            fast.iter().zip(slow.iter()).map(|(a, b)| a - b).collect()
        }
        GoertzelDetrendMode::LogZeroLagRegressionDetrending => {
            detrend_ln_zero_lag_regression(src_rev, params.dt_reg_zl_smooth_per.unwrap_or(5))?
        }
    };
    if out.iter().all(|v| v.is_finite()) {
        Some(out)
    } else {
        None
    }
}

fn bartels_prob(n: usize, cycle_count: usize, values: &[f64]) -> f64 {
    if n == 0 || cycle_count == 0 || values.len() < n * cycle_count {
        return 1.0;
    }

    let mut avg_coeff_a = 0.0;
    let mut avg_coeff_b = 0.0;
    let mut avg_ind_amplit = 0.0;
    let mut vsin = vec![0.0; n];
    let mut vcos = vec![0.0; n];

    for i in 0..n {
        let theta = (i + 1) as f64 / n as f64 * 2.0 * std::f64::consts::PI;
        vsin[i] = theta.sin();
        vcos[i] = theta.cos();
    }

    for t in 0..cycle_count {
        let mut coeff_a = 0.0;
        let mut coeff_b = 0.0;
        let base = t * n;
        for i in 0..n {
            let value = values[base + i];
            coeff_a += vsin[i] * value;
            coeff_b += vcos[i] * value;
        }
        avg_coeff_a += coeff_a;
        avg_coeff_b += coeff_b;
        avg_ind_amplit += coeff_a * coeff_a + coeff_b * coeff_b;
    }

    avg_coeff_a /= cycle_count as f64;
    avg_coeff_b /= cycle_count as f64;
    let avg_ampl = (avg_coeff_a * avg_coeff_a + avg_coeff_b * avg_coeff_b).sqrt();
    let avg_ind_amplit = (avg_ind_amplit / cycle_count as f64).sqrt();
    let expected_ampl = avg_ind_amplit / (cycle_count as f64).sqrt();
    if expected_ampl <= f64::EPSILON {
        return 1.0;
    }
    let a_ratio = avg_ampl / expected_ampl;
    (-a_ratio * a_ratio).exp()
}

fn apply_bartels(
    src_rev: &[f64],
    cycles: &mut Vec<CycleInfo>,
    params: &GoertzelCycleCompositeWaveParams,
) {
    let bart_smooth_per = params.bart_smooth_per.unwrap_or(DEFAULT_BART_SMOOTH_PER);
    let bart_no_cycles = params.bart_no_cycles.unwrap_or(DEFAULT_BART_NO_CYCLES);
    let bart_sig_limit = params.bart_sig_limit.unwrap_or(DEFAULT_BART_SIG_LIMIT) as f64;

    for cycle in cycles.iter_mut() {
        let bars_taken = cycle.cycle.saturating_mul(bart_no_cycles);
        if bars_taken == 0 || bars_taken > src_rev.len() {
            cycle.bartels = 0.0;
            continue;
        }
        cycle.bartels = detrend_ln_zero_lag_regression(&src_rev[..bars_taken], bart_smooth_per)
            .map(|values| (1.0 - bartels_prob(cycle.cycle, bart_no_cycles, &values)) * 100.0)
            .unwrap_or(0.0);
    }

    cycles.retain(|cycle| cycle.bartels > bart_sig_limit);
    if params.sort_bartels.unwrap_or(false) {
        cycles.sort_by(|a, b| {
            b.bartels
                .partial_cmp(&a.bartels)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
}

fn extract_cycles(src_rev: &[f64], params: &GoertzelCycleCompositeWaveParams) -> Vec<CycleInfo> {
    let per = params.max_period.unwrap_or(DEFAULT_MAX_PERIOD);
    let for_bar = params.bar_to_calculate.unwrap_or(DEFAULT_BAR_TO_CALCULATE);
    let sample = 2 * per;
    if src_rev.len() < for_bar + sample || sample < 2 {
        return Vec::new();
    }

    let mut amp_work = vec![0.0; sample + 1];
    let mut phase_work = vec![0.0; sample + 1];
    let mut mark_work = vec![0.0; sample + 1];
    let mut detrended = vec![0.0; sample + 1];

    let temp1 = src_rev[for_bar + sample - 1];
    let trend_slope = (src_rev[for_bar] - temp1) / (sample as f64 - 1.0);
    for k in (1..sample).rev() {
        detrended[k] = src_rev[for_bar + k - 1] - (temp1 + trend_slope * (sample - k) as f64);
    }

    for k in 2..=per {
        let z = 1.0 / k as f64;
        let coeff = 2.0 * (2.0 * std::f64::consts::PI * z).cos();
        let mut w = 0.0;
        let mut x = 0.0;
        let mut y = 0.0;
        for i in (1..=sample).rev() {
            w = coeff * x - y + detrended[i];
            y = x;
            x = w;
        }
        let mut real = x - y * coeff / 2.0;
        if real.abs() <= f64::EPSILON {
            real = 1e-7;
        }
        let imag = y * (2.0 * std::f64::consts::PI * z).sin();
        let amplitude = if params.squared_amp.unwrap_or(true) {
            real * real + imag * imag
        } else {
            (real * real + imag * imag).sqrt()
        };
        amp_work[k] = if params.use_cycle_strength.unwrap_or(true) {
            amplitude / k as f64
        } else {
            amplitude
        };
        let mut phase = (imag / real).atan();
        if real < 0.0 {
            phase += std::f64::consts::PI;
        } else if imag < 0.0 {
            phase += 2.0 * std::f64::consts::PI;
        }
        phase_work[k] = phase;
    }

    for k in 3..per {
        if amp_work[k] > amp_work[k - 1] && amp_work[k] > amp_work[k + 1] {
            mark_work[k] = k as f64 * 1e-4;
        }
    }

    let mut cycles = Vec::new();
    for i in 0..=per + 1 {
        if i < mark_work.len() && mark_work[i] > 0.0 {
            cycles.push(CycleInfo {
                cycle: (10000.0 * mark_work[i]).round() as usize,
                amplitude: amp_work[i],
                phase: phase_work[i],
                bartels: 0.0,
            });
        }
    }

    cycles.sort_by(|a, b| {
        b.amplitude
            .partial_cmp(&a.amplitude)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if params.filter_bartels.unwrap_or(false) {
        apply_bartels(src_rev, &mut cycles, params);
    }
    cycles
}

fn current_wave_from_cycles(
    cycles: &[CycleInfo],
    params: &GoertzelCycleCompositeWaveParams,
) -> f64 {
    if cycles.is_empty() {
        return 0.0;
    }
    let start = params
        .start_at_cycle
        .unwrap_or(DEFAULT_START_AT_CYCLE)
        .saturating_sub(1);
    if start >= cycles.len() {
        return 0.0;
    }
    let count = params.use_top_cycles.unwrap_or(DEFAULT_USE_TOP_CYCLES);
    let end = (start + count).min(cycles.len());
    let trig = |cycle: &CycleInfo| {
        if params.use_cosine.unwrap_or(true) {
            cycle.amplitude * cycle.phase.cos()
        } else {
            cycle.amplitude * cycle.phase.sin()
        }
    };
    let mut out = cycles[start..end].iter().map(trig).sum::<f64>();
    if params.subtract_noise.unwrap_or(false) && end < cycles.len() {
        out -= cycles[end..].iter().map(trig).sum::<f64>();
    }
    out
}

#[inline(always)]
fn compute_window_wave(src_rev: &[f64], params: &GoertzelCycleCompositeWaveParams) -> Option<f64> {
    let processed = apply_detrend_mode(src_rev, params)?;
    let cycles = extract_cycles(&processed, params);
    Some(current_wave_from_cycles(&cycles, params))
}

#[derive(Debug, Clone)]
pub struct GoertzelCycleCompositeWaveStream {
    params: GoertzelCycleCompositeWaveParams,
    sample_size: usize,
    window: VecDeque<f64>,
}

impl GoertzelCycleCompositeWaveStream {
    #[inline(always)]
    pub fn try_new(
        params: GoertzelCycleCompositeWaveParams,
    ) -> Result<Self, GoertzelCycleCompositeWaveError> {
        validate_params(&params)?;
        let sample_size = sample_size_for_params(&params);
        Ok(Self {
            params,
            sample_size,
            window: VecDeque::with_capacity(sample_size.max(1)),
        })
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.window.clear();
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        if !value.is_finite() {
            self.reset();
            return None;
        }
        self.window.push_back(value);
        if self.window.len() > self.sample_size {
            self.window.pop_front();
        }
        if self.window.len() < self.sample_size {
            return None;
        }

        let mut src_rev = Vec::with_capacity(self.sample_size);
        for &value in self.window.iter().rev() {
            src_rev.push(value);
        }
        compute_window_wave(&src_rev, &self.params)
    }

    #[inline(always)]
    pub fn get_warmup_period(&self) -> usize {
        self.sample_size.saturating_sub(1)
    }
}

fn compute_row(data: &[f64], params: &GoertzelCycleCompositeWaveParams, out: &mut [f64]) {
    out.fill(f64::NAN);
    let sample_size = sample_size_for_params(params);
    let mut src_rev = Vec::with_capacity(sample_size);

    for end in sample_size.saturating_sub(1)..data.len() {
        let window = &data[end + 1 - sample_size..=end];
        if window.iter().any(|v| !v.is_finite()) {
            continue;
        }
        src_rev.clear();
        src_rev.extend(window.iter().rev().copied());
        if let Some(value) = compute_window_wave(&src_rev, params) {
            out[end] = value;
        }
    }
}

pub fn goertzel_cycle_composite_wave(
    input: &GoertzelCycleCompositeWaveInput,
) -> Result<GoertzelCycleCompositeWaveOutput, GoertzelCycleCompositeWaveError> {
    goertzel_cycle_composite_wave_with_kernel(input, Kernel::Auto)
}

pub fn goertzel_cycle_composite_wave_with_kernel(
    input: &GoertzelCycleCompositeWaveInput,
    kernel: Kernel,
) -> Result<GoertzelCycleCompositeWaveOutput, GoertzelCycleCompositeWaveError> {
    let data = input.as_ref();
    let needed = validate_common(data, &input.params)?;
    let _chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let mut values = alloc_with_nan_prefix(data.len(), needed.saturating_sub(1));
    compute_row(data, &input.params, &mut values);
    Ok(GoertzelCycleCompositeWaveOutput { values })
}

pub fn goertzel_cycle_composite_wave_into_slice(
    dst: &mut [f64],
    input: &GoertzelCycleCompositeWaveInput,
    kernel: Kernel,
) -> Result<(), GoertzelCycleCompositeWaveError> {
    let data = input.as_ref();
    validate_common(data, &input.params)?;
    if dst.len() != data.len() {
        return Err(GoertzelCycleCompositeWaveError::OutputLengthMismatch {
            expected: data.len(),
            got: dst.len(),
        });
    }

    let _chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    compute_row(data, &input.params, dst);
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn goertzel_cycle_composite_wave_into(
    input: &GoertzelCycleCompositeWaveInput,
    dst: &mut [f64],
) -> Result<(), GoertzelCycleCompositeWaveError> {
    goertzel_cycle_composite_wave_into_slice(dst, input, Kernel::Auto)
}

#[derive(Debug, Clone, Copy)]
pub struct GoertzelCycleCompositeWaveBatchRange {
    pub max_period: (usize, usize, usize),
    pub start_at_cycle: (usize, usize, usize),
    pub use_top_cycles: (usize, usize, usize),
    pub base_params: GoertzelCycleCompositeWaveParams,
}

impl Default for GoertzelCycleCompositeWaveBatchRange {
    fn default() -> Self {
        Self {
            max_period: (DEFAULT_MAX_PERIOD, DEFAULT_MAX_PERIOD, 0),
            start_at_cycle: (DEFAULT_START_AT_CYCLE, DEFAULT_START_AT_CYCLE, 0),
            use_top_cycles: (DEFAULT_USE_TOP_CYCLES, DEFAULT_USE_TOP_CYCLES, 0),
            base_params: GoertzelCycleCompositeWaveParams::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GoertzelCycleCompositeWaveBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<GoertzelCycleCompositeWaveParams>,
    pub rows: usize,
    pub cols: usize,
}

impl GoertzelCycleCompositeWaveBatchOutput {
    pub fn values_for(&self, params: &GoertzelCycleCompositeWaveParams) -> Option<&[f64]> {
        self.combos
            .iter()
            .position(|combo| combo == params)
            .and_then(|row| {
                let start = row.checked_mul(self.cols)?;
                self.values.get(start..start + self.cols)
            })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GoertzelCycleCompositeWaveBatchBuilder {
    range: GoertzelCycleCompositeWaveBatchRange,
    kernel: Kernel,
}

impl Default for GoertzelCycleCompositeWaveBatchBuilder {
    fn default() -> Self {
        Self {
            range: GoertzelCycleCompositeWaveBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl GoertzelCycleCompositeWaveBatchBuilder {
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
    pub fn max_period_range(mut self, value: (usize, usize, usize)) -> Self {
        self.range.max_period = value;
        self
    }

    #[inline(always)]
    pub fn start_at_cycle_range(mut self, value: (usize, usize, usize)) -> Self {
        self.range.start_at_cycle = value;
        self
    }

    #[inline(always)]
    pub fn use_top_cycles_range(mut self, value: (usize, usize, usize)) -> Self {
        self.range.use_top_cycles = value;
        self
    }

    #[inline(always)]
    pub fn params(mut self, value: GoertzelCycleCompositeWaveParams) -> Self {
        self.range.base_params = value;
        self
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<GoertzelCycleCompositeWaveBatchOutput, GoertzelCycleCompositeWaveError> {
        goertzel_cycle_composite_wave_batch_with_kernel(data, &self.range, self.kernel)
    }
}

#[inline(always)]
fn expand_axis(
    range: (usize, usize, usize),
) -> Result<Vec<usize>, GoertzelCycleCompositeWaveError> {
    let (start, end, step) = range;
    if start == 0 {
        return Err(GoertzelCycleCompositeWaveError::InvalidRange { start, end, step });
    }
    if step == 0 {
        return Ok(vec![start]);
    }
    if start > end {
        return Err(GoertzelCycleCompositeWaveError::InvalidRange { start, end, step });
    }
    let mut out = Vec::new();
    let mut cur = start;
    loop {
        out.push(cur);
        if cur >= end {
            break;
        }
        let next =
            cur.checked_add(step)
                .ok_or_else(|| GoertzelCycleCompositeWaveError::InvalidInput {
                    msg: "goertzel_cycle_composite_wave: range step overflow".to_string(),
                })?;
        if next <= cur {
            return Err(GoertzelCycleCompositeWaveError::InvalidRange { start, end, step });
        }
        cur = next.min(end);
    }
    Ok(out)
}

#[inline(always)]
fn expand_grid_checked(
    range: &GoertzelCycleCompositeWaveBatchRange,
) -> Result<Vec<GoertzelCycleCompositeWaveParams>, GoertzelCycleCompositeWaveError> {
    let max_periods = expand_axis(range.max_period)?;
    let start_cycles = expand_axis(range.start_at_cycle)?;
    let top_cycles = expand_axis(range.use_top_cycles)?;
    let mut combos = Vec::with_capacity(max_periods.len() * start_cycles.len() * top_cycles.len());
    for max_period in max_periods {
        for start_at_cycle in &start_cycles {
            for use_top_cycles in &top_cycles {
                let mut params = range.base_params;
                params.max_period = Some(max_period);
                params.start_at_cycle = Some(*start_at_cycle);
                params.use_top_cycles = Some(*use_top_cycles);
                combos.push(params);
            }
        }
    }
    Ok(combos)
}

pub fn expand_grid_goertzel_cycle_composite_wave(
    range: &GoertzelCycleCompositeWaveBatchRange,
) -> Vec<GoertzelCycleCompositeWaveParams> {
    expand_grid_checked(range).unwrap_or_default()
}

pub fn goertzel_cycle_composite_wave_batch_with_kernel(
    data: &[f64],
    sweep: &GoertzelCycleCompositeWaveBatchRange,
    kernel: Kernel,
) -> Result<GoertzelCycleCompositeWaveBatchOutput, GoertzelCycleCompositeWaveError> {
    goertzel_cycle_composite_wave_batch_inner(data, sweep, kernel, true)
}

pub fn goertzel_cycle_composite_wave_batch_slice(
    data: &[f64],
    sweep: &GoertzelCycleCompositeWaveBatchRange,
    kernel: Kernel,
) -> Result<GoertzelCycleCompositeWaveBatchOutput, GoertzelCycleCompositeWaveError> {
    goertzel_cycle_composite_wave_batch_inner(data, sweep, kernel, false)
}

pub fn goertzel_cycle_composite_wave_batch_par_slice(
    data: &[f64],
    sweep: &GoertzelCycleCompositeWaveBatchRange,
    kernel: Kernel,
) -> Result<GoertzelCycleCompositeWaveBatchOutput, GoertzelCycleCompositeWaveError> {
    goertzel_cycle_composite_wave_batch_inner(data, sweep, kernel, true)
}

fn goertzel_cycle_composite_wave_batch_inner(
    data: &[f64],
    sweep: &GoertzelCycleCompositeWaveBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<GoertzelCycleCompositeWaveBatchOutput, GoertzelCycleCompositeWaveError> {
    let combos = expand_grid_checked(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    let total =
        rows.checked_mul(cols)
            .ok_or_else(|| GoertzelCycleCompositeWaveError::InvalidInput {
                msg: "goertzel_cycle_composite_wave: rows*cols overflow in batch".to_string(),
            })?;

    if data.is_empty() {
        return Err(GoertzelCycleCompositeWaveError::EmptyInputData);
    }
    let max_run = longest_valid_run(data);
    if max_run == 0 {
        return Err(GoertzelCycleCompositeWaveError::AllValuesNaN);
    }
    let mut max_needed = 0usize;
    let warmups: Vec<usize> = combos
        .iter()
        .map(|params| {
            let needed = sample_size_for_params(params);
            max_needed = max_needed.max(needed);
            needed.saturating_sub(1)
        })
        .collect();
    if max_run < max_needed {
        return Err(GoertzelCycleCompositeWaveError::NotEnoughValidData {
            needed: max_needed,
            valid: max_run,
        });
    }

    let mut values_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut values_mu, cols, &warmups);
    let mut values = unsafe {
        Vec::from_raw_parts(
            values_mu.as_mut_ptr() as *mut f64,
            values_mu.len(),
            values_mu.capacity(),
        )
    };
    std::mem::forget(values_mu);
    debug_assert_eq!(values.len(), total);

    goertzel_cycle_composite_wave_batch_inner_into(data, sweep, kernel, parallel, &mut values)?;

    Ok(GoertzelCycleCompositeWaveBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

fn goertzel_cycle_composite_wave_batch_inner_into(
    data: &[f64],
    sweep: &GoertzelCycleCompositeWaveBatchRange,
    kernel: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<GoertzelCycleCompositeWaveParams>, GoertzelCycleCompositeWaveError> {
    match kernel {
        Kernel::Auto
        | Kernel::Scalar
        | Kernel::ScalarBatch
        | Kernel::Avx2
        | Kernel::Avx2Batch
        | Kernel::Avx512
        | Kernel::Avx512Batch => {}
        other => {
            return Err(GoertzelCycleCompositeWaveError::InvalidKernelForBatch(
                other,
            ))
        }
    }

    let combos = expand_grid_checked(sweep)?;
    let len = data.len();
    if len == 0 {
        return Err(GoertzelCycleCompositeWaveError::EmptyInputData);
    }
    let total = combos.len().checked_mul(len).ok_or_else(|| {
        GoertzelCycleCompositeWaveError::InvalidInput {
            msg: "goertzel_cycle_composite_wave: rows*cols overflow in batch_into".to_string(),
        }
    })?;
    if out.len() != total {
        return Err(GoertzelCycleCompositeWaveError::MismatchedOutputLen {
            dst_len: out.len(),
            expected_len: total,
        });
    }

    let max_run = longest_valid_run(data);
    if max_run == 0 {
        return Err(GoertzelCycleCompositeWaveError::AllValuesNaN);
    }
    let max_needed = combos.iter().map(sample_size_for_params).max().unwrap_or(0);
    if max_run < max_needed {
        return Err(GoertzelCycleCompositeWaveError::NotEnoughValidData {
            needed: max_needed,
            valid: max_run,
        });
    }

    let _chosen = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other => other,
    };

    let worker = |row: usize, dst: &mut [f64]| compute_row(data, &combos[row], dst);

    if parallel && combos.len() > 1 {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out.par_chunks_mut(len)
                .enumerate()
                .for_each(|(row, dst)| worker(row, dst));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, dst) in out.chunks_mut(len).enumerate() {
                worker(row, dst);
            }
        }
    } else {
        for (row, dst) in out.chunks_mut(len).enumerate() {
            worker(row, dst);
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
fn parse_mode_py(value: &str) -> PyResult<GoertzelDetrendMode> {
    GoertzelDetrendMode::parse(value)
        .ok_or_else(|| PyValueError::new_err(format!("Invalid detrend mode: {value}")))
}

#[cfg(feature = "python")]
fn build_params_py(
    max_period: usize,
    start_at_cycle: usize,
    use_top_cycles: usize,
    bar_to_calculate: usize,
    detrend_mode: &str,
    dt_zl_per1: usize,
    dt_zl_per2: usize,
    dt_hp_per1: usize,
    dt_hp_per2: usize,
    dt_reg_zl_smooth_per: usize,
    hp_smooth_per: usize,
    zlma_smooth_per: usize,
    filter_bartels: bool,
    bart_no_cycles: usize,
    bart_smooth_per: usize,
    bart_sig_limit: usize,
    sort_bartels: bool,
    squared_amp: bool,
    use_cosine: bool,
    subtract_noise: bool,
    use_cycle_strength: bool,
) -> PyResult<GoertzelCycleCompositeWaveParams> {
    Ok(GoertzelCycleCompositeWaveParams {
        max_period: Some(max_period),
        start_at_cycle: Some(start_at_cycle),
        use_top_cycles: Some(use_top_cycles),
        bar_to_calculate: Some(bar_to_calculate),
        detrend_mode: Some(parse_mode_py(detrend_mode)?),
        dt_zl_per1: Some(dt_zl_per1),
        dt_zl_per2: Some(dt_zl_per2),
        dt_hp_per1: Some(dt_hp_per1),
        dt_hp_per2: Some(dt_hp_per2),
        dt_reg_zl_smooth_per: Some(dt_reg_zl_smooth_per),
        hp_smooth_per: Some(hp_smooth_per),
        zlma_smooth_per: Some(zlma_smooth_per),
        filter_bartels: Some(filter_bartels),
        bart_no_cycles: Some(bart_no_cycles),
        bart_smooth_per: Some(bart_smooth_per),
        bart_sig_limit: Some(bart_sig_limit),
        sort_bartels: Some(sort_bartels),
        squared_amp: Some(squared_amp),
        use_cosine: Some(use_cosine),
        subtract_noise: Some(subtract_noise),
        use_cycle_strength: Some(use_cycle_strength),
    })
}

#[cfg(feature = "python")]
#[pyfunction(name = "goertzel_cycle_composite_wave")]
#[pyo3(signature = (
    data,
    max_period=DEFAULT_MAX_PERIOD,
    start_at_cycle=DEFAULT_START_AT_CYCLE,
    use_top_cycles=DEFAULT_USE_TOP_CYCLES,
    bar_to_calculate=DEFAULT_BAR_TO_CALCULATE,
    detrend_mode="hodrick_prescott_detrending",
    dt_zl_per1=DEFAULT_DT_ZL_PER1,
    dt_zl_per2=DEFAULT_DT_ZL_PER2,
    dt_hp_per1=DEFAULT_DT_HP_PER1,
    dt_hp_per2=DEFAULT_DT_HP_PER2,
    dt_reg_zl_smooth_per=DEFAULT_DT_REG_ZL_SMOOTH_PER,
    hp_smooth_per=DEFAULT_HP_SMOOTH_PER,
    zlma_smooth_per=DEFAULT_ZLMA_SMOOTH_PER,
    filter_bartels=false,
    bart_no_cycles=DEFAULT_BART_NO_CYCLES,
    bart_smooth_per=DEFAULT_BART_SMOOTH_PER,
    bart_sig_limit=DEFAULT_BART_SIG_LIMIT,
    sort_bartels=false,
    squared_amp=true,
    use_cosine=true,
    subtract_noise=false,
    use_cycle_strength=true,
    kernel=None
))]
pub fn goertzel_cycle_composite_wave_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    max_period: usize,
    start_at_cycle: usize,
    use_top_cycles: usize,
    bar_to_calculate: usize,
    detrend_mode: &str,
    dt_zl_per1: usize,
    dt_zl_per2: usize,
    dt_hp_per1: usize,
    dt_hp_per2: usize,
    dt_reg_zl_smooth_per: usize,
    hp_smooth_per: usize,
    zlma_smooth_per: usize,
    filter_bartels: bool,
    bart_no_cycles: usize,
    bart_smooth_per: usize,
    bart_sig_limit: usize,
    sort_bartels: bool,
    squared_amp: bool,
    use_cosine: bool,
    subtract_noise: bool,
    use_cycle_strength: bool,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let params = build_params_py(
        max_period,
        start_at_cycle,
        use_top_cycles,
        bar_to_calculate,
        detrend_mode,
        dt_zl_per1,
        dt_zl_per2,
        dt_hp_per1,
        dt_hp_per2,
        dt_reg_zl_smooth_per,
        hp_smooth_per,
        zlma_smooth_per,
        filter_bartels,
        bart_no_cycles,
        bart_smooth_per,
        bart_sig_limit,
        sort_bartels,
        squared_amp,
        use_cosine,
        subtract_noise,
        use_cycle_strength,
    )?;
    let input = GoertzelCycleCompositeWaveInput::from_slice(data, params);
    let out = py
        .allow_threads(|| goertzel_cycle_composite_wave_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(out.values.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "GoertzelCycleCompositeWaveStream")]
pub struct GoertzelCycleCompositeWaveStreamPy {
    stream: GoertzelCycleCompositeWaveStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl GoertzelCycleCompositeWaveStreamPy {
    #[new]
    #[pyo3(signature = (
        max_period=DEFAULT_MAX_PERIOD,
        start_at_cycle=DEFAULT_START_AT_CYCLE,
        use_top_cycles=DEFAULT_USE_TOP_CYCLES,
        bar_to_calculate=DEFAULT_BAR_TO_CALCULATE,
        detrend_mode="hodrick_prescott_detrending",
        dt_zl_per1=DEFAULT_DT_ZL_PER1,
        dt_zl_per2=DEFAULT_DT_ZL_PER2,
        dt_hp_per1=DEFAULT_DT_HP_PER1,
        dt_hp_per2=DEFAULT_DT_HP_PER2,
        dt_reg_zl_smooth_per=DEFAULT_DT_REG_ZL_SMOOTH_PER,
        hp_smooth_per=DEFAULT_HP_SMOOTH_PER,
        zlma_smooth_per=DEFAULT_ZLMA_SMOOTH_PER,
        filter_bartels=false,
        bart_no_cycles=DEFAULT_BART_NO_CYCLES,
        bart_smooth_per=DEFAULT_BART_SMOOTH_PER,
        bart_sig_limit=DEFAULT_BART_SIG_LIMIT,
        sort_bartels=false,
        squared_amp=true,
        use_cosine=true,
        subtract_noise=false,
        use_cycle_strength=true
    ))]
    fn new(
        max_period: usize,
        start_at_cycle: usize,
        use_top_cycles: usize,
        bar_to_calculate: usize,
        detrend_mode: &str,
        dt_zl_per1: usize,
        dt_zl_per2: usize,
        dt_hp_per1: usize,
        dt_hp_per2: usize,
        dt_reg_zl_smooth_per: usize,
        hp_smooth_per: usize,
        zlma_smooth_per: usize,
        filter_bartels: bool,
        bart_no_cycles: usize,
        bart_smooth_per: usize,
        bart_sig_limit: usize,
        sort_bartels: bool,
        squared_amp: bool,
        use_cosine: bool,
        subtract_noise: bool,
        use_cycle_strength: bool,
    ) -> PyResult<Self> {
        let params = build_params_py(
            max_period,
            start_at_cycle,
            use_top_cycles,
            bar_to_calculate,
            detrend_mode,
            dt_zl_per1,
            dt_zl_per2,
            dt_hp_per1,
            dt_hp_per2,
            dt_reg_zl_smooth_per,
            hp_smooth_per,
            zlma_smooth_per,
            filter_bartels,
            bart_no_cycles,
            bart_smooth_per,
            bart_sig_limit,
            sort_bartels,
            squared_amp,
            use_cosine,
            subtract_noise,
            use_cycle_strength,
        )?;
        let stream = GoertzelCycleCompositeWaveStream::try_new(params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }

    fn reset(&mut self) {
        self.stream.reset();
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "goertzel_cycle_composite_wave_batch")]
#[pyo3(signature = (
    data,
    max_period_range=(DEFAULT_MAX_PERIOD, DEFAULT_MAX_PERIOD, 0),
    start_at_cycle_range=(DEFAULT_START_AT_CYCLE, DEFAULT_START_AT_CYCLE, 0),
    use_top_cycles_range=(DEFAULT_USE_TOP_CYCLES, DEFAULT_USE_TOP_CYCLES, 0),
    bar_to_calculate=DEFAULT_BAR_TO_CALCULATE,
    detrend_mode="hodrick_prescott_detrending",
    dt_zl_per1=DEFAULT_DT_ZL_PER1,
    dt_zl_per2=DEFAULT_DT_ZL_PER2,
    dt_hp_per1=DEFAULT_DT_HP_PER1,
    dt_hp_per2=DEFAULT_DT_HP_PER2,
    dt_reg_zl_smooth_per=DEFAULT_DT_REG_ZL_SMOOTH_PER,
    hp_smooth_per=DEFAULT_HP_SMOOTH_PER,
    zlma_smooth_per=DEFAULT_ZLMA_SMOOTH_PER,
    filter_bartels=false,
    bart_no_cycles=DEFAULT_BART_NO_CYCLES,
    bart_smooth_per=DEFAULT_BART_SMOOTH_PER,
    bart_sig_limit=DEFAULT_BART_SIG_LIMIT,
    sort_bartels=false,
    squared_amp=true,
    use_cosine=true,
    subtract_noise=false,
    use_cycle_strength=true,
    kernel=None
))]
pub fn goertzel_cycle_composite_wave_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    max_period_range: (usize, usize, usize),
    start_at_cycle_range: (usize, usize, usize),
    use_top_cycles_range: (usize, usize, usize),
    bar_to_calculate: usize,
    detrend_mode: &str,
    dt_zl_per1: usize,
    dt_zl_per2: usize,
    dt_hp_per1: usize,
    dt_hp_per2: usize,
    dt_reg_zl_smooth_per: usize,
    hp_smooth_per: usize,
    zlma_smooth_per: usize,
    filter_bartels: bool,
    bart_no_cycles: usize,
    bart_smooth_per: usize,
    bart_sig_limit: usize,
    sort_bartels: bool,
    squared_amp: bool,
    use_cosine: bool,
    subtract_noise: bool,
    use_cycle_strength: bool,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let base_params = build_params_py(
        DEFAULT_MAX_PERIOD,
        DEFAULT_START_AT_CYCLE,
        DEFAULT_USE_TOP_CYCLES,
        bar_to_calculate,
        detrend_mode,
        dt_zl_per1,
        dt_zl_per2,
        dt_hp_per1,
        dt_hp_per2,
        dt_reg_zl_smooth_per,
        hp_smooth_per,
        zlma_smooth_per,
        filter_bartels,
        bart_no_cycles,
        bart_smooth_per,
        bart_sig_limit,
        sort_bartels,
        squared_amp,
        use_cosine,
        subtract_noise,
        use_cycle_strength,
    )?;
    let output = py
        .allow_threads(|| {
            goertzel_cycle_composite_wave_batch_with_kernel(
                data,
                &GoertzelCycleCompositeWaveBatchRange {
                    max_period: max_period_range,
                    start_at_cycle: start_at_cycle_range,
                    use_top_cycles: use_top_cycles_range,
                    base_params,
                },
                kern,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item(
        "values",
        output
            .values
            .into_pyarray(py)
            .reshape((output.rows, output.cols))?,
    )?;
    dict.set_item(
        "max_periods",
        output
            .combos
            .iter()
            .map(|params| params.max_period.unwrap_or(DEFAULT_MAX_PERIOD) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "start_at_cycles",
        output
            .combos
            .iter()
            .map(|params| params.start_at_cycle.unwrap_or(DEFAULT_START_AT_CYCLE) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "use_top_cycles",
        output
            .combos
            .iter()
            .map(|params| params.use_top_cycles.unwrap_or(DEFAULT_USE_TOP_CYCLES) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", output.rows)?;
    dict.set_item("cols", output.cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_goertzel_cycle_composite_wave_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(goertzel_cycle_composite_wave_py, m)?)?;
    m.add_function(wrap_pyfunction!(goertzel_cycle_composite_wave_batch_py, m)?)?;
    m.add_class::<GoertzelCycleCompositeWaveStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoertzelCycleCompositeWaveJsConfig {
    pub max_period: Option<usize>,
    pub start_at_cycle: Option<usize>,
    pub use_top_cycles: Option<usize>,
    pub bar_to_calculate: Option<usize>,
    pub detrend_mode: Option<GoertzelDetrendMode>,
    pub dt_zl_per1: Option<usize>,
    pub dt_zl_per2: Option<usize>,
    pub dt_hp_per1: Option<usize>,
    pub dt_hp_per2: Option<usize>,
    pub dt_reg_zl_smooth_per: Option<usize>,
    pub hp_smooth_per: Option<usize>,
    pub zlma_smooth_per: Option<usize>,
    pub filter_bartels: Option<bool>,
    pub bart_no_cycles: Option<usize>,
    pub bart_smooth_per: Option<usize>,
    pub bart_sig_limit: Option<usize>,
    pub sort_bartels: Option<bool>,
    pub squared_amp: Option<bool>,
    pub use_cosine: Option<bool>,
    pub subtract_noise: Option<bool>,
    pub use_cycle_strength: Option<bool>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
impl Default for GoertzelCycleCompositeWaveJsConfig {
    fn default() -> Self {
        let params = GoertzelCycleCompositeWaveParams::default();
        Self {
            max_period: params.max_period,
            start_at_cycle: params.start_at_cycle,
            use_top_cycles: params.use_top_cycles,
            bar_to_calculate: params.bar_to_calculate,
            detrend_mode: params.detrend_mode,
            dt_zl_per1: params.dt_zl_per1,
            dt_zl_per2: params.dt_zl_per2,
            dt_hp_per1: params.dt_hp_per1,
            dt_hp_per2: params.dt_hp_per2,
            dt_reg_zl_smooth_per: params.dt_reg_zl_smooth_per,
            hp_smooth_per: params.hp_smooth_per,
            zlma_smooth_per: params.zlma_smooth_per,
            filter_bartels: params.filter_bartels,
            bart_no_cycles: params.bart_no_cycles,
            bart_smooth_per: params.bart_smooth_per,
            bart_sig_limit: params.bart_sig_limit,
            sort_bartels: params.sort_bartels,
            squared_amp: params.squared_amp,
            use_cosine: params.use_cosine,
            subtract_noise: params.subtract_noise,
            use_cycle_strength: params.use_cycle_strength,
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
impl From<GoertzelCycleCompositeWaveJsConfig> for GoertzelCycleCompositeWaveParams {
    fn from(value: GoertzelCycleCompositeWaveJsConfig) -> Self {
        Self {
            max_period: value.max_period,
            start_at_cycle: value.start_at_cycle,
            use_top_cycles: value.use_top_cycles,
            bar_to_calculate: value.bar_to_calculate,
            detrend_mode: value.detrend_mode,
            dt_zl_per1: value.dt_zl_per1,
            dt_zl_per2: value.dt_zl_per2,
            dt_hp_per1: value.dt_hp_per1,
            dt_hp_per2: value.dt_hp_per2,
            dt_reg_zl_smooth_per: value.dt_reg_zl_smooth_per,
            hp_smooth_per: value.hp_smooth_per,
            zlma_smooth_per: value.zlma_smooth_per,
            filter_bartels: value.filter_bartels,
            bart_no_cycles: value.bart_no_cycles,
            bart_smooth_per: value.bart_smooth_per,
            bart_sig_limit: value.bart_sig_limit,
            sort_bartels: value.sort_bartels,
            squared_amp: value.squared_amp,
            use_cosine: value.use_cosine,
            subtract_noise: value.subtract_noise,
            use_cycle_strength: value.use_cycle_strength,
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoertzelCycleCompositeWaveBatchConfig {
    pub max_period_range: Vec<usize>,
    pub start_at_cycle_range: Vec<usize>,
    pub use_top_cycles_range: Vec<usize>,
    #[serde(flatten)]
    pub base: GoertzelCycleCompositeWaveJsConfig,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoertzelCycleCompositeWaveBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<GoertzelCycleCompositeWaveParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn goertzel_cycle_composite_wave_js(
    data: &[f64],
    config: JsValue,
) -> Result<Vec<f64>, JsValue> {
    let config: GoertzelCycleCompositeWaveJsConfig = if config.is_undefined() || config.is_null() {
        GoertzelCycleCompositeWaveJsConfig::default()
    } else {
        serde_wasm_bindgen::from_value(config)
            .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?
    };
    let input = GoertzelCycleCompositeWaveInput::from_slice(data, config.into());
    let out = goertzel_cycle_composite_wave_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(out.values)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn goertzel_cycle_composite_wave_batch_js(
    data: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: GoertzelCycleCompositeWaveBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.max_period_range.len() != 3
        || config.start_at_cycle_range.len() != 3
        || config.use_top_cycles_range.len() != 3
    {
        return Err(JsValue::from_str(
            "Invalid config: ranges must have exactly 3 elements [start, end, step]",
        ));
    }
    let out = goertzel_cycle_composite_wave_batch_with_kernel(
        data,
        &GoertzelCycleCompositeWaveBatchRange {
            max_period: (
                config.max_period_range[0],
                config.max_period_range[1],
                config.max_period_range[2],
            ),
            start_at_cycle: (
                config.start_at_cycle_range[0],
                config.start_at_cycle_range[1],
                config.start_at_cycle_range[2],
            ),
            use_top_cycles: (
                config.use_top_cycles_range[0],
                config.use_top_cycles_range[1],
                config.use_top_cycles_range[2],
            ),
            base_params: config.base.into(),
        },
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    serde_wasm_bindgen::to_value(&GoertzelCycleCompositeWaveBatchJsOutput {
        values: out.values,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn goertzel_cycle_composite_wave_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn goertzel_cycle_composite_wave_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn goertzel_cycle_composite_wave_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    max_period: usize,
    start_at_cycle: usize,
    use_top_cycles: usize,
    bar_to_calculate: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to goertzel_cycle_composite_wave_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let input = GoertzelCycleCompositeWaveInput::from_slice(
            data,
            GoertzelCycleCompositeWaveParams {
                max_period: Some(max_period),
                start_at_cycle: Some(start_at_cycle),
                use_top_cycles: Some(use_top_cycles),
                bar_to_calculate: Some(bar_to_calculate),
                ..GoertzelCycleCompositeWaveParams::default()
            },
        );
        goertzel_cycle_composite_wave_into_slice(
            std::slice::from_raw_parts_mut(out_ptr, len),
            &input,
            Kernel::Scalar,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn goertzel_cycle_composite_wave_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    max_period_start: usize,
    max_period_end: usize,
    max_period_step: usize,
    start_at_cycle_start: usize,
    start_at_cycle_end: usize,
    start_at_cycle_step: usize,
    use_top_cycles_start: usize,
    use_top_cycles_end: usize,
    use_top_cycles_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to goertzel_cycle_composite_wave_batch_into",
        ));
    }
    let sweep = GoertzelCycleCompositeWaveBatchRange {
        max_period: (max_period_start, max_period_end, max_period_step),
        start_at_cycle: (
            start_at_cycle_start,
            start_at_cycle_end,
            start_at_cycle_step,
        ),
        use_top_cycles: (
            use_top_cycles_start,
            use_top_cycles_end,
            use_top_cycles_step,
        ),
        base_params: GoertzelCycleCompositeWaveParams::default(),
    };
    let combos = expand_grid_checked(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        goertzel_cycle_composite_wave_batch_inner_into(
            data,
            &sweep,
            Kernel::ScalarBatch,
            false,
            std::slice::from_raw_parts_mut(out_ptr, rows * len),
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }
    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn goertzel_cycle_composite_wave_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Float64Array,
) -> Result<usize, JsValue> {
    let values = goertzel_cycle_composite_wave_js(data, config)?;
    crate::write_wasm_f64_output("goertzel_cycle_composite_wave_output_into_js", &values, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn goertzel_cycle_composite_wave_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = goertzel_cycle_composite_wave_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "goertzel_cycle_composite_wave_batch_output_into_js",
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

    fn sample_data(len: usize) -> Vec<f64> {
        (0..len)
            .map(|i| {
                let x = i as f64;
                100.0
                    + (2.0 * std::f64::consts::PI * x / 28.0).sin() * 2.0
                    + (2.0 * std::f64::consts::PI * x / 14.0).cos() * 0.8
                    + x * 0.01
            })
            .collect()
    }

    #[test]
    fn goertzel_cycle_composite_wave_produces_finite_tail() -> Result<(), Box<dyn Error>> {
        let data = sample_data(320);
        let out = goertzel_cycle_composite_wave(&GoertzelCycleCompositeWaveInput::from_slice(
            &data,
            GoertzelCycleCompositeWaveParams {
                max_period: Some(32),
                ..GoertzelCycleCompositeWaveParams::default()
            },
        ))?;
        assert!(out.values.iter().filter(|v| v.is_finite()).count() > 16);
        Ok(())
    }

    #[test]
    fn goertzel_cycle_composite_wave_into_matches_api() -> Result<(), Box<dyn Error>> {
        let data = sample_data(420);
        let params = GoertzelCycleCompositeWaveParams {
            max_period: Some(32),
            ..GoertzelCycleCompositeWaveParams::default()
        };
        let input = GoertzelCycleCompositeWaveInput::from_slice(&data, params);
        let baseline = goertzel_cycle_composite_wave(&input)?.values;
        let mut out = vec![0.0; data.len()];
        goertzel_cycle_composite_wave_into_slice(&mut out, &input, Kernel::Auto)?;
        for (a, b) in baseline.iter().zip(out.iter()) {
            assert!((a.is_nan() && b.is_nan()) || (*a - *b).abs() <= 1e-12);
        }
        Ok(())
    }

    #[test]
    fn goertzel_cycle_composite_wave_stream_matches_batch() -> Result<(), Box<dyn Error>> {
        let data = sample_data(420);
        let params = GoertzelCycleCompositeWaveParams {
            max_period: Some(32),
            ..GoertzelCycleCompositeWaveParams::default()
        };
        let batch = goertzel_cycle_composite_wave(&GoertzelCycleCompositeWaveInput::from_slice(
            &data, params,
        ))?
        .values;
        let mut stream = GoertzelCycleCompositeWaveStream::try_new(params)?;
        let mut stream_values = Vec::with_capacity(data.len());
        for value in data {
            stream_values.push(stream.update(value).unwrap_or(f64::NAN));
        }
        for (a, b) in batch.iter().zip(stream_values.iter()) {
            assert!((a.is_nan() && b.is_nan()) || (*a - *b).abs() <= 1e-12);
        }
        Ok(())
    }

    #[test]
    fn goertzel_cycle_composite_wave_batch_single_matches_single() -> Result<(), Box<dyn Error>> {
        let data = sample_data(420);
        let params = GoertzelCycleCompositeWaveParams {
            max_period: Some(32),
            ..GoertzelCycleCompositeWaveParams::default()
        };
        let single = goertzel_cycle_composite_wave(&GoertzelCycleCompositeWaveInput::from_slice(
            &data, params,
        ))?
        .values;
        let batch = goertzel_cycle_composite_wave_batch_with_kernel(
            &data,
            &GoertzelCycleCompositeWaveBatchRange {
                max_period: (32, 32, 0),
                start_at_cycle: (DEFAULT_START_AT_CYCLE, DEFAULT_START_AT_CYCLE, 0),
                use_top_cycles: (DEFAULT_USE_TOP_CYCLES, DEFAULT_USE_TOP_CYCLES, 0),
                base_params: params,
            },
            Kernel::Auto,
        )?;
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, data.len());
        let row = batch.values_for(&params).unwrap();
        for (a, b) in row.iter().zip(single.iter()) {
            assert!((a.is_nan() && b.is_nan()) || (*a - *b).abs() <= 1e-12);
        }
        Ok(())
    }

    #[test]
    fn goertzel_cycle_composite_wave_rejects_invalid_params() {
        let data = sample_data(128);
        let err = goertzel_cycle_composite_wave(&GoertzelCycleCompositeWaveInput::from_slice(
            &data,
            GoertzelCycleCompositeWaveParams {
                max_period: Some(1),
                ..GoertzelCycleCompositeWaveParams::default()
            },
        ))
        .unwrap_err();
        assert!(matches!(
            err,
            GoertzelCycleCompositeWaveError::InvalidParameter {
                name: "max_period",
                ..
            }
        ));
    }

    #[test]
    fn goertzel_cycle_composite_wave_dispatch_compute_returns_value() -> Result<(), Box<dyn Error>>
    {
        let data = sample_data(420);
        let params = [ParamKV {
            key: "max_period",
            value: ParamValue::Int(32),
        }];
        let out = compute_cpu(IndicatorComputeRequest {
            indicator_id: "goertzel_cycle_composite_wave",
            data: IndicatorDataRef::Slice { values: &data },
            params: &params,
            output_id: Some("wave"),
            kernel: Kernel::Auto,
        })?;
        assert_eq!(out.output_id, "wave");
        assert_eq!(out.rows, 1);
        assert_eq!(out.cols, data.len());
        Ok(())
    }
}
