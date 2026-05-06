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
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_adaptive_cg_output_into_js(
    data: &[f64],
    alpha: Option<f64>,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = ehlers_adaptive_cg_js(data, alpha)?;
    crate::write_wasm_object_f64_outputs("ehlers_adaptive_cg_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_adaptive_cg_batch_unified_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = ehlers_adaptive_cg_batch_unified_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "ehlers_adaptive_cg_batch_unified_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
use std::error::Error as StdError;
use std::mem::ManuallyDrop;
use thiserror::Error;

const DEFAULT_ALPHA: f64 = 0.07;
const MIN_VALID_LEN: usize = 14;
const MAX_WINDOW: usize = 100;

impl<'a> AsRef<[f64]> for EhlersAdaptiveCgInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            EhlersAdaptiveCgData::Slice(slice) => slice,
            EhlersAdaptiveCgData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum EhlersAdaptiveCgData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct EhlersAdaptiveCgOutput {
    pub cg: Vec<f64>,
    pub trigger: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct EhlersAdaptiveCgParams {
    pub alpha: Option<f64>,
}

impl Default for EhlersAdaptiveCgParams {
    fn default() -> Self {
        Self {
            alpha: Some(DEFAULT_ALPHA),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EhlersAdaptiveCgInput<'a> {
    pub data: EhlersAdaptiveCgData<'a>,
    pub params: EhlersAdaptiveCgParams,
}

impl<'a> EhlersAdaptiveCgInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: EhlersAdaptiveCgParams,
    ) -> Self {
        Self {
            data: EhlersAdaptiveCgData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: EhlersAdaptiveCgParams) -> Self {
        Self {
            data: EhlersAdaptiveCgData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "hl2", EhlersAdaptiveCgParams::default())
    }

    #[inline]
    pub fn get_alpha(&self) -> f64 {
        self.params.alpha.unwrap_or(DEFAULT_ALPHA)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct EhlersAdaptiveCgBuilder {
    alpha: Option<f64>,
    kernel: Kernel,
}

impl Default for EhlersAdaptiveCgBuilder {
    fn default() -> Self {
        Self {
            alpha: None,
            kernel: Kernel::Auto,
        }
    }
}

impl EhlersAdaptiveCgBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = Some(alpha);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline(always)]
    pub fn apply(self, candles: &Candles) -> Result<EhlersAdaptiveCgOutput, EhlersAdaptiveCgError> {
        let input = EhlersAdaptiveCgInput::from_candles(
            candles,
            "hl2",
            EhlersAdaptiveCgParams { alpha: self.alpha },
        );
        ehlers_adaptive_cg_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<EhlersAdaptiveCgOutput, EhlersAdaptiveCgError> {
        let input =
            EhlersAdaptiveCgInput::from_slice(data, EhlersAdaptiveCgParams { alpha: self.alpha });
        ehlers_adaptive_cg_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<EhlersAdaptiveCgStream, EhlersAdaptiveCgError> {
        EhlersAdaptiveCgStream::try_new(EhlersAdaptiveCgParams { alpha: self.alpha })
    }
}

#[derive(Debug, Error)]
pub enum EhlersAdaptiveCgError {
    #[error("ehlers_adaptive_cg: input data slice is empty.")]
    EmptyInputData,
    #[error("ehlers_adaptive_cg: all values are NaN.")]
    AllValuesNaN,
    #[error("ehlers_adaptive_cg: invalid alpha: {alpha}")]
    InvalidAlpha { alpha: f64 },
    #[error("ehlers_adaptive_cg: not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("ehlers_adaptive_cg: output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("ehlers_adaptive_cg: invalid range: start={start}, end={end}, step={step}")]
    InvalidRange { start: f64, end: f64, step: f64 },
    #[error("ehlers_adaptive_cg: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Copy, Clone, Debug)]
struct PreparedInput<'a> {
    data: &'a [f64],
    len: usize,
    first_valid: usize,
    alpha: f64,
}

#[derive(Copy, Clone, Debug)]
struct AlphaCoeffs {
    alpha_half_sq: f64,
    one_minus_alpha: f64,
    one_minus_alpha_sq: f64,
}

impl AlphaCoeffs {
    #[inline(always)]
    fn new(alpha: f64) -> Self {
        let alpha_half = 1.0 - 0.5 * alpha;
        let one_minus_alpha = 1.0 - alpha;
        Self {
            alpha_half_sq: alpha_half * alpha_half,
            one_minus_alpha,
            one_minus_alpha_sq: one_minus_alpha * one_minus_alpha,
        }
    }
}

#[derive(Debug, Clone)]
struct AdaptiveBuffers {
    smooth: Vec<f64>,
    cycle: Vec<f64>,
    q1: Vec<f64>,
    dp: Vec<f64>,
    ip: Vec<f64>,
    p: Vec<f64>,
}

impl AdaptiveBuffers {
    fn new(len: usize) -> Self {
        Self {
            smooth: vec![0.0; len],
            cycle: vec![0.0; len],
            q1: vec![0.0; len],
            dp: vec![0.1; len],
            ip: vec![0.0; len],
            p: vec![0.0; len],
        }
    }

    fn push_default(&mut self) {
        self.smooth.push(0.0);
        self.cycle.push(0.0);
        self.q1.push(0.0);
        self.dp.push(0.1);
        self.ip.push(0.0);
        self.p.push(0.0);
    }
}

#[inline(always)]
fn validate_alpha(alpha: f64) -> Result<f64, EhlersAdaptiveCgError> {
    if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
        return Err(EhlersAdaptiveCgError::InvalidAlpha { alpha });
    }
    Ok(alpha)
}

#[inline(always)]
fn normalize_single_kernel(kernel: Kernel) -> Kernel {
    match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    }
}

#[inline(always)]
fn normalize_single_kernel_to_scalar(kernel: Kernel) -> Kernel {
    match normalize_single_kernel(kernel) {
        Kernel::Auto
        | Kernel::Scalar
        | Kernel::ScalarBatch
        | Kernel::Avx2
        | Kernel::Avx2Batch
        | Kernel::Avx512
        | Kernel::Avx512Batch => Kernel::Scalar,
    }
}

#[inline(always)]
fn prepare_input<'a>(
    input: &'a EhlersAdaptiveCgInput<'a>,
) -> Result<PreparedInput<'a>, EhlersAdaptiveCgError> {
    let data = input.as_ref();
    if data.is_empty() {
        return Err(EhlersAdaptiveCgError::EmptyInputData);
    }

    let first_valid = data
        .iter()
        .position(|value| !value.is_nan())
        .ok_or(EhlersAdaptiveCgError::AllValuesNaN)?;
    let valid = data.len() - first_valid;
    if valid < MIN_VALID_LEN {
        return Err(EhlersAdaptiveCgError::NotEnoughValidData {
            needed: MIN_VALID_LEN,
            valid,
        });
    }

    Ok(PreparedInput {
        data,
        len: data.len(),
        first_valid,
        alpha: validate_alpha(input.get_alpha())?,
    })
}

#[inline(always)]
fn median3(a: f64, b: f64, c: f64) -> f64 {
    (a + b + c) - a.min(b.min(c)) - a.max(b.max(c))
}

#[inline(always)]
fn history_or(data: &[f64], idx: usize, lag: usize, fallback: f64) -> f64 {
    idx.checked_sub(lag)
        .and_then(|j| data.get(j).copied())
        .unwrap_or(fallback)
}

#[inline(always)]
fn append_step(
    data: &[f64],
    first_valid: usize,
    coeffs: AlphaCoeffs,
    idx: usize,
    buffers: &mut AdaptiveBuffers,
    cg_out: &mut [f64],
) {
    if idx < first_valid {
        return;
    }

    let x0 = data[idx];
    if x0.is_nan() {
        return;
    }

    let x1 = history_or(data, idx, 1, x0);
    let x2 = history_or(data, idx, 2, x1);
    let x3 = history_or(data, idx, 3, x2);

    buffers.smooth[idx] = (x0 + 2.0 * x1 + 2.0 * x2 + x3) / 6.0;

    if idx < first_valid + 7 {
        buffers.cycle[idx] = (x0 - 2.0 * x1 + x2) * 0.25;
    } else {
        let s0 = buffers.smooth[idx];
        let s1 = buffers.smooth[idx - 1];
        let s2 = buffers.smooth[idx - 2];
        buffers.cycle[idx] = coeffs.alpha_half_sq * (s0 - 2.0 * s1 + s2)
            + 2.0 * coeffs.one_minus_alpha * buffers.cycle[idx - 1]
            - coeffs.one_minus_alpha_sq * buffers.cycle[idx - 2];
    }

    if idx >= first_valid + 6 {
        buffers.q1[idx] = (0.0962 * buffers.cycle[idx] + 0.5769 * buffers.cycle[idx - 2]
            - 0.5769 * buffers.cycle[idx - 4]
            - 0.0962 * buffers.cycle[idx - 6])
            * (0.5 + 0.08 * history_or(&buffers.ip, idx, 1, 0.0));
    }

    if idx >= first_valid + 7 {
        let i1 = buffers.cycle[idx - 3];
        let prev_i1 = buffers.cycle[idx - 4];
        let q = buffers.q1[idx];
        let prev_q = buffers.q1[idx - 1];
        let raw = if q.abs() > f64::EPSILON && prev_q.abs() > f64::EPSILON {
            (i1 / q - prev_i1 / prev_q) / (1.0 + i1 * prev_i1 / (q * prev_q))
        } else {
            0.0
        };
        buffers.dp[idx] = raw.clamp(0.1, 1.1);
    } else {
        buffers.dp[idx] = 0.1;
    }

    let md = if idx >= first_valid + 4 {
        median3(
            buffers.dp[idx],
            buffers.dp[idx - 1],
            median3(
                buffers.dp[idx - 2],
                buffers.dp[idx - 3],
                buffers.dp[idx - 4],
            ),
        )
    } else {
        0.1
    };
    let dc = std::f64::consts::TAU / md + 0.5;

    buffers.ip[idx] = if idx == first_valid {
        dc
    } else {
        0.33 * dc + 0.67 * buffers.ip[idx - 1]
    };
    buffers.p[idx] = if idx == first_valid {
        buffers.ip[idx]
    } else {
        0.15 * buffers.ip[idx] + 0.85 * buffers.p[idx - 1]
    };

    let window = ((buffers.p[idx] * 0.5).round() as isize).clamp(1, MAX_WINDOW as isize) as usize;
    if idx + 1 < first_valid + window || idx + 1 < window {
        return;
    }

    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for lag in 0..window {
        let value = data[idx - lag];
        if value.is_nan() {
            return;
        }
        numerator += (lag as f64 + 1.0) * value;
        denominator += value;
    }

    cg_out[idx] = if denominator.abs() > f64::EPSILON {
        -numerator / denominator + (window as f64 + 1.0) * 0.5
    } else {
        0.0
    };
}

fn compute_series_into(prepared: PreparedInput<'_>, cg_out: &mut [f64], trigger_out: &mut [f64]) {
    let mut buffers = AdaptiveBuffers::new(prepared.len);
    let coeffs = AlphaCoeffs::new(prepared.alpha);
    for idx in prepared.first_valid..prepared.len {
        append_step(
            prepared.data,
            prepared.first_valid,
            coeffs,
            idx,
            &mut buffers,
            cg_out,
        );
    }

    for idx in prepared.first_valid.saturating_add(1)..prepared.len {
        trigger_out[idx] = cg_out[idx - 1];
    }
}

#[inline]
pub fn ehlers_adaptive_cg(
    input: &EhlersAdaptiveCgInput,
) -> Result<EhlersAdaptiveCgOutput, EhlersAdaptiveCgError> {
    ehlers_adaptive_cg_with_kernel(input, Kernel::Auto)
}

pub fn ehlers_adaptive_cg_with_kernel(
    input: &EhlersAdaptiveCgInput,
    kernel: Kernel,
) -> Result<EhlersAdaptiveCgOutput, EhlersAdaptiveCgError> {
    let prepared = prepare_input(input)?;
    let _kernel = normalize_single_kernel_to_scalar(kernel);
    let mut cg = alloc_with_nan_prefix(prepared.len, prepared.len);
    let mut trigger = alloc_with_nan_prefix(prepared.len, prepared.len);
    compute_series_into(prepared, &mut cg, &mut trigger);
    Ok(EhlersAdaptiveCgOutput { cg, trigger })
}

pub fn ehlers_adaptive_cg_into_slice(
    cg_out: &mut [f64],
    trigger_out: &mut [f64],
    input: &EhlersAdaptiveCgInput,
    kernel: Kernel,
) -> Result<(), EhlersAdaptiveCgError> {
    let prepared = prepare_input(input)?;
    let _kernel = normalize_single_kernel_to_scalar(kernel);
    if cg_out.len() != prepared.len {
        return Err(EhlersAdaptiveCgError::OutputLengthMismatch {
            expected: prepared.len,
            got: cg_out.len(),
        });
    }
    if trigger_out.len() != prepared.len {
        return Err(EhlersAdaptiveCgError::OutputLengthMismatch {
            expected: prepared.len,
            got: trigger_out.len(),
        });
    }
    cg_out.fill(f64::NAN);
    trigger_out.fill(f64::NAN);
    compute_series_into(prepared, cg_out, trigger_out);
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn ehlers_adaptive_cg_into(
    input: &EhlersAdaptiveCgInput,
    cg_out: &mut [f64],
    trigger_out: &mut [f64],
) -> Result<(), EhlersAdaptiveCgError> {
    ehlers_adaptive_cg_into_slice(cg_out, trigger_out, input, Kernel::Auto)
}

#[derive(Debug, Clone)]
pub struct EhlersAdaptiveCgStream {
    alpha: f64,
    coeffs: AlphaCoeffs,
    first_valid: Option<usize>,
    data: Vec<f64>,
    buffers: AdaptiveBuffers,
    cg: Vec<f64>,
}

impl EhlersAdaptiveCgStream {
    pub fn try_new(params: EhlersAdaptiveCgParams) -> Result<Self, EhlersAdaptiveCgError> {
        let alpha = validate_alpha(params.alpha.unwrap_or(DEFAULT_ALPHA))?;
        Ok(Self {
            alpha,
            coeffs: AlphaCoeffs::new(alpha),
            first_valid: None,
            data: Vec::new(),
            buffers: AdaptiveBuffers::new(0),
            cg: Vec::new(),
        })
    }

    pub fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        let idx = self.data.len();
        self.data.push(value);
        self.buffers.push_default();
        self.cg.push(f64::NAN);

        if self.first_valid.is_none() && !value.is_nan() {
            self.first_valid = Some(idx);
        }
        let first_valid = self.first_valid?;
        append_step(
            &self.data,
            first_valid,
            self.coeffs,
            idx,
            &mut self.buffers,
            &mut self.cg,
        );

        let current = self.cg[idx];
        if current.is_nan() {
            return None;
        }

        let trigger = if idx > first_valid {
            self.cg[idx - 1]
        } else {
            f64::NAN
        };
        Some((current, trigger))
    }

    pub fn reset(&mut self) {
        self.first_valid = None;
        self.data.clear();
        self.buffers = AdaptiveBuffers::new(0);
        self.cg.clear();
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EhlersAdaptiveCgBatchRange {
    pub alpha: (f64, f64, f64),
}

impl Default for EhlersAdaptiveCgBatchRange {
    fn default() -> Self {
        Self {
            alpha: (DEFAULT_ALPHA, DEFAULT_ALPHA, 0.0),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EhlersAdaptiveCgBatchBuilder {
    range: EhlersAdaptiveCgBatchRange,
    kernel: Kernel,
}

impl Default for EhlersAdaptiveCgBatchBuilder {
    fn default() -> Self {
        Self {
            range: EhlersAdaptiveCgBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl EhlersAdaptiveCgBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn alpha_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.alpha = (start, end, step);
        self
    }

    pub fn alpha_static(mut self, alpha: f64) -> Self {
        self.range.alpha = (alpha, alpha, 0.0);
        self
    }

    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<EhlersAdaptiveCgBatchOutput, EhlersAdaptiveCgError> {
        ehlers_adaptive_cg_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<EhlersAdaptiveCgBatchOutput, EhlersAdaptiveCgError> {
        self.apply_slice(source_type(candles, "hl2"))
    }

    pub fn apply_candles_source(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<EhlersAdaptiveCgBatchOutput, EhlersAdaptiveCgError> {
        self.apply_slice(source_type(candles, source))
    }
}

#[derive(Debug, Clone)]
pub struct EhlersAdaptiveCgBatchOutput {
    pub cg: Vec<f64>,
    pub trigger: Vec<f64>,
    pub combos: Vec<EhlersAdaptiveCgParams>,
    pub rows: usize,
    pub cols: usize,
}

impl EhlersAdaptiveCgBatchOutput {
    pub fn row_for_params(&self, params: &EhlersAdaptiveCgParams) -> Option<usize> {
        let alpha = params.alpha.unwrap_or(DEFAULT_ALPHA);
        self.combos
            .iter()
            .position(|combo| (combo.alpha.unwrap_or(DEFAULT_ALPHA) - alpha).abs() <= 1e-12)
    }

    pub fn cg_for(&self, params: &EhlersAdaptiveCgParams) -> Option<&[f64]> {
        let row = self.row_for_params(params)?;
        let start = row.checked_mul(self.cols)?;
        let end = start.checked_add(self.cols)?;
        self.cg.get(start..end)
    }

    pub fn trigger_for(&self, params: &EhlersAdaptiveCgParams) -> Option<&[f64]> {
        let row = self.row_for_params(params)?;
        let start = row.checked_mul(self.cols)?;
        let end = start.checked_add(self.cols)?;
        self.trigger.get(start..end)
    }
}

fn expand_grid(
    range: &EhlersAdaptiveCgBatchRange,
) -> Result<Vec<EhlersAdaptiveCgParams>, EhlersAdaptiveCgError> {
    fn axis((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, EhlersAdaptiveCgError> {
        if !start.is_finite() || !end.is_finite() || !step.is_finite() {
            return Err(EhlersAdaptiveCgError::InvalidRange { start, end, step });
        }
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return Ok(vec![start]);
        }

        let step_abs = step.abs();
        let mut values = Vec::new();
        if start < end {
            let mut current = start;
            while current <= end + 1e-12 {
                values.push(current);
                current += step_abs;
            }
        } else {
            let mut current = start;
            while current + 1e-12 >= end {
                values.push(current);
                current -= step_abs;
            }
        }

        if values.is_empty() {
            return Err(EhlersAdaptiveCgError::InvalidRange { start, end, step });
        }
        Ok(values)
    }

    axis(range.alpha)?
        .into_iter()
        .map(|alpha| {
            validate_alpha(alpha)?;
            Ok(EhlersAdaptiveCgParams { alpha: Some(alpha) })
        })
        .collect()
}

#[inline(always)]
pub fn ehlers_adaptive_cg_batch_slice(
    data: &[f64],
    sweep: &EhlersAdaptiveCgBatchRange,
) -> Result<EhlersAdaptiveCgBatchOutput, EhlersAdaptiveCgError> {
    ehlers_adaptive_cg_batch_inner(data, sweep, Kernel::Scalar, false)
}

#[inline(always)]
pub fn ehlers_adaptive_cg_batch_par_slice(
    data: &[f64],
    sweep: &EhlersAdaptiveCgBatchRange,
) -> Result<EhlersAdaptiveCgBatchOutput, EhlersAdaptiveCgError> {
    ehlers_adaptive_cg_batch_inner(data, sweep, Kernel::Scalar, true)
}

pub fn ehlers_adaptive_cg_batch_with_kernel(
    data: &[f64],
    sweep: &EhlersAdaptiveCgBatchRange,
    kernel: Kernel,
) -> Result<EhlersAdaptiveCgBatchOutput, EhlersAdaptiveCgError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(EhlersAdaptiveCgError::InvalidKernelForBatch(other)),
    };
    let scalar_kernel = match batch_kernel {
        Kernel::ScalarBatch => Kernel::Scalar,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::Avx512Batch => Kernel::Avx512,
        _ => unreachable!(),
    };
    ehlers_adaptive_cg_batch_inner(
        data,
        sweep,
        scalar_kernel,
        !matches!(batch_kernel, Kernel::ScalarBatch),
    )
}

fn ehlers_adaptive_cg_batch_inner(
    data: &[f64],
    sweep: &EhlersAdaptiveCgBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<EhlersAdaptiveCgBatchOutput, EhlersAdaptiveCgError> {
    let _kernel = normalize_single_kernel_to_scalar(kernel);
    let combos = expand_grid(sweep)?;
    if data.is_empty() {
        return Err(EhlersAdaptiveCgError::EmptyInputData);
    }
    let first_valid = data
        .iter()
        .position(|value| !value.is_nan())
        .ok_or(EhlersAdaptiveCgError::AllValuesNaN)?;
    let valid = data.len() - first_valid;
    if valid < MIN_VALID_LEN {
        return Err(EhlersAdaptiveCgError::NotEnoughValidData {
            needed: MIN_VALID_LEN,
            valid,
        });
    }

    let rows = combos.len();
    let cols = data.len();

    let cg_mu = make_uninit_matrix(rows, cols);
    let trigger_mu = make_uninit_matrix(rows, cols);
    let mut cg_guard = ManuallyDrop::new(cg_mu);
    let mut trigger_guard = ManuallyDrop::new(trigger_mu);

    let cg_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(cg_guard.as_mut_ptr() as *mut f64, cg_guard.len())
    };
    let trigger_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(trigger_guard.as_mut_ptr() as *mut f64, trigger_guard.len())
    };

    let do_row = |row: usize, cg_row: &mut [f64], trigger_row: &mut [f64]| {
        let prepared = PreparedInput {
            data,
            len: cols,
            first_valid,
            alpha: combos[row].alpha.unwrap_or(DEFAULT_ALPHA),
        };
        cg_row.fill(f64::NAN);
        trigger_row.fill(f64::NAN);
        compute_series_into(prepared, cg_row, trigger_row);
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            cg_out
                .par_chunks_mut(cols)
                .zip(trigger_out.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, (cg_row, trigger_row))| do_row(row, cg_row, trigger_row));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, (cg_row, trigger_row)) in cg_out
                .chunks_mut(cols)
                .zip(trigger_out.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, cg_row, trigger_row);
            }
        }
    } else {
        for (row, (cg_row, trigger_row)) in cg_out
            .chunks_mut(cols)
            .zip(trigger_out.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, cg_row, trigger_row);
        }
    }

    let cg = unsafe {
        Vec::from_raw_parts(
            cg_guard.as_mut_ptr() as *mut f64,
            cg_guard.len(),
            cg_guard.capacity(),
        )
    };
    let trigger = unsafe {
        Vec::from_raw_parts(
            trigger_guard.as_mut_ptr() as *mut f64,
            trigger_guard.len(),
            trigger_guard.capacity(),
        )
    };

    Ok(EhlersAdaptiveCgBatchOutput {
        cg,
        trigger,
        combos,
        rows,
        cols,
    })
}

#[cfg(feature = "python")]
#[pyfunction(name = "ehlers_adaptive_cg")]
#[pyo3(signature = (data, alpha=None, *, kernel=None))]
pub fn ehlers_adaptive_cg_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    alpha: Option<f64>,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let slice = data.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = EhlersAdaptiveCgInput::from_slice(slice, EhlersAdaptiveCgParams { alpha });
    let out = py
        .allow_threads(|| ehlers_adaptive_cg_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((out.cg.into_pyarray(py), out.trigger.into_pyarray(py)))
}

#[cfg(feature = "python")]
#[pyclass(name = "EhlersAdaptiveCgStream")]
pub struct EhlersAdaptiveCgStreamPy {
    inner: EhlersAdaptiveCgStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl EhlersAdaptiveCgStreamPy {
    #[new]
    pub fn new(alpha: Option<f64>) -> PyResult<Self> {
        let inner = EhlersAdaptiveCgStream::try_new(EhlersAdaptiveCgParams { alpha })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    pub fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        self.inner.update(value)
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "ehlers_adaptive_cg_batch")]
#[pyo3(signature = (data, alpha_range, kernel=None))]
pub fn ehlers_adaptive_cg_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    alpha_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let slice = data.as_slice()?;
    let kernel = validate_kernel(kernel, true)?;
    let out = py
        .allow_threads(|| {
            ehlers_adaptive_cg_batch_with_kernel(
                slice,
                &EhlersAdaptiveCgBatchRange { alpha: alpha_range },
                kernel,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("cg", out.cg.into_pyarray(py).reshape((out.rows, out.cols))?)?;
    dict.set_item(
        "trigger",
        out.trigger.into_pyarray(py).reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "alphas",
        out.combos
            .iter()
            .map(|combo| combo.alpha.unwrap_or(DEFAULT_ALPHA))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", out.rows)?;
    dict.set_item("cols", out.cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_ehlers_adaptive_cg_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ehlers_adaptive_cg_py, m)?)?;
    m.add_function(wrap_pyfunction!(ehlers_adaptive_cg_batch_py, m)?)?;
    m.add_class::<EhlersAdaptiveCgStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
struct EhlersAdaptiveCgJsOutput {
    cg: Vec<f64>,
    trigger: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
struct EhlersAdaptiveCgStreamJsOutput {
    cg: f64,
    trigger: f64,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct EhlersAdaptiveCgBatchConfig {
    pub alpha_range: (f64, f64, f64),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct EhlersAdaptiveCgBatchJsOutput {
    pub cg: Vec<f64>,
    pub trigger: Vec<f64>,
    pub combos: Vec<EhlersAdaptiveCgParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = ehlers_adaptive_cg_js)]
pub fn ehlers_adaptive_cg_js(data: &[f64], alpha: Option<f64>) -> Result<JsValue, JsValue> {
    let out = ehlers_adaptive_cg(&EhlersAdaptiveCgInput::from_slice(
        data,
        EhlersAdaptiveCgParams { alpha },
    ))
    .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&EhlersAdaptiveCgJsOutput {
        cg: out.cg,
        trigger: out.trigger,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = ehlers_adaptive_cg_batch)]
pub fn ehlers_adaptive_cg_batch_unified_js(
    data: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: EhlersAdaptiveCgBatchConfig =
        serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let out = ehlers_adaptive_cg_batch_with_kernel(
        data,
        &EhlersAdaptiveCgBatchRange {
            alpha: config.alpha_range,
        },
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&EhlersAdaptiveCgBatchJsOutput {
        cg: out.cg,
        trigger: out.trigger,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = ehlers_adaptive_cg_alloc)]
pub fn ehlers_adaptive_cg_alloc(len: usize) -> *mut f64 {
    let mut values = vec![0.0; len];
    let ptr = values.as_mut_ptr();
    std::mem::forget(values);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = ehlers_adaptive_cg_free)]
pub fn ehlers_adaptive_cg_free(ptr: *mut f64, len: usize) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        drop(Vec::from_raw_parts(ptr, 0, len));
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = ehlers_adaptive_cg_into)]
pub fn ehlers_adaptive_cg_into(
    data_ptr: *const f64,
    cg_ptr: *mut f64,
    trigger_ptr: *mut f64,
    len: usize,
    alpha: Option<f64>,
) -> Result<(), JsValue> {
    if data_ptr.is_null() || cg_ptr.is_null() || trigger_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to ehlers_adaptive_cg_into",
        ));
    }

    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let input = EhlersAdaptiveCgInput::from_slice(data, EhlersAdaptiveCgParams { alpha });

        let alias_input = data_ptr == cg_ptr as *const f64 || data_ptr == trigger_ptr as *const f64;
        let alias_outputs = cg_ptr == trigger_ptr;
        if alias_input || alias_outputs {
            let mut cg = vec![0.0; len];
            let mut trigger = vec![0.0; len];
            ehlers_adaptive_cg_into_slice(&mut cg, &mut trigger, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            std::slice::from_raw_parts_mut(cg_ptr, len).copy_from_slice(&cg);
            std::slice::from_raw_parts_mut(trigger_ptr, len).copy_from_slice(&trigger);
            return Ok(());
        }

        let cg = std::slice::from_raw_parts_mut(cg_ptr, len);
        let trigger = std::slice::from_raw_parts_mut(trigger_ptr, len);
        ehlers_adaptive_cg_into_slice(cg, trigger, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub struct EhlersAdaptiveCgStreamWasm {
    inner: EhlersAdaptiveCgStream,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
impl EhlersAdaptiveCgStreamWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(alpha: Option<f64>) -> Result<EhlersAdaptiveCgStreamWasm, JsValue> {
        Ok(Self {
            inner: EhlersAdaptiveCgStream::try_new(EhlersAdaptiveCgParams { alpha })
                .map_err(|e| JsValue::from_str(&e.to_string()))?,
        })
    }

    pub fn update(&mut self, value: f64) -> Result<JsValue, JsValue> {
        match self.inner.update(value) {
            Some((cg, trigger)) => {
                serde_wasm_bindgen::to_value(&EhlersAdaptiveCgStreamJsOutput { cg, trigger })
                    .map_err(|e| JsValue::from_str(&e.to_string()))
            }
            None => Ok(JsValue::NULL),
        }
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn sample_data() -> Vec<f64> {
        (0..256)
            .map(|idx| {
                let x = idx as f64;
                100.0 + (x * 0.13).sin() * 2.5 + (x * 0.047).cos() * 1.75 + x * 0.02
            })
            .collect()
    }

    #[test]
    fn ehlers_adaptive_cg_into_matches_api() -> Result<(), Box<dyn StdError>> {
        let data = sample_data();
        let input = EhlersAdaptiveCgInput::from_slice(&data, EhlersAdaptiveCgParams::default());
        let out = ehlers_adaptive_cg(&input)?;

        let mut cg = vec![0.0; data.len()];
        let mut trigger = vec![0.0; data.len()];
        ehlers_adaptive_cg_into(&input, &mut cg, &mut trigger)?;

        assert_eq!(out.cg.len(), cg.len());
        assert_eq!(out.trigger.len(), trigger.len());
        for idx in 0..data.len() {
            let a = out.cg[idx];
            let b = cg[idx];
            assert!((a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12);
            let a = out.trigger[idx];
            let b = trigger[idx];
            assert!((a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12);
        }
        Ok(())
    }

    #[test]
    fn ehlers_adaptive_cg_stream_matches_batch() -> Result<(), Box<dyn StdError>> {
        let data = sample_data();
        let batch = ehlers_adaptive_cg(&EhlersAdaptiveCgInput::from_slice(
            &data,
            EhlersAdaptiveCgParams::default(),
        ))?;

        let mut stream = EhlersAdaptiveCgStream::try_new(EhlersAdaptiveCgParams::default())?;
        let mut cg = vec![f64::NAN; data.len()];
        let mut trigger = vec![f64::NAN; data.len()];
        for (idx, value) in data.iter().copied().enumerate() {
            if let Some((cg_value, trigger_value)) = stream.update(value) {
                cg[idx] = cg_value;
                trigger[idx] = trigger_value;
            }
        }

        for idx in 0..data.len() {
            let a = batch.cg[idx];
            let b = cg[idx];
            assert!((a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12);
            let a = batch.trigger[idx];
            let b = trigger[idx];
            assert!((a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12);
        }
        Ok(())
    }

    #[test]
    fn ehlers_adaptive_cg_batch_matches_single() -> Result<(), Box<dyn StdError>> {
        let data = sample_data();
        let batch = ehlers_adaptive_cg_batch_with_kernel(
            &data,
            &EhlersAdaptiveCgBatchRange {
                alpha: (0.07, 0.09, 0.02),
            },
            Kernel::ScalarBatch,
        )?;

        assert_eq!(batch.rows, 2);
        assert_eq!(batch.cols, data.len());

        for (row, params) in batch.combos.iter().enumerate() {
            let single =
                ehlers_adaptive_cg(&EhlersAdaptiveCgInput::from_slice(&data, params.clone()))?;
            let start = row * batch.cols;
            for idx in 0..batch.cols {
                let a = single.cg[idx];
                let b = batch.cg[start + idx];
                assert!((a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12);
                let a = single.trigger[idx];
                let b = batch.trigger[start + idx];
                assert!((a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12);
            }
        }

        Ok(())
    }

    #[test]
    fn ehlers_adaptive_cg_fixture_has_values() -> Result<(), Box<dyn StdError>> {
        let candles = read_candles_from_csv("src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv")?;
        let out = ehlers_adaptive_cg(&EhlersAdaptiveCgInput::with_default_candles(&candles))?;
        assert_eq!(out.cg.len(), candles.close.len());
        assert_eq!(out.trigger.len(), candles.close.len());
        assert!(out.cg.iter().skip(64).any(|value| value.is_finite()));
        assert!(out.trigger.iter().skip(64).any(|value| value.is_finite()));
        Ok(())
    }
}
