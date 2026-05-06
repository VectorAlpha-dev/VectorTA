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

use crate::indicators::rsi::{rsi_into_slice, RsiInput, RsiParams};
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
use std::error::Error;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum EhlersDataSamplingRelativeStrengthIndicatorData<'a> {
    Candles { candles: &'a Candles },
    Slices { open: &'a [f64], close: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct EhlersDataSamplingRelativeStrengthIndicatorOutput {
    pub ds_rsi: Vec<f64>,
    pub original_rsi: Vec<f64>,
    pub signal: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct EhlersDataSamplingRelativeStrengthIndicatorParams {
    pub length: Option<usize>,
}

impl Default for EhlersDataSamplingRelativeStrengthIndicatorParams {
    fn default() -> Self {
        Self { length: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct EhlersDataSamplingRelativeStrengthIndicatorInput<'a> {
    pub data: EhlersDataSamplingRelativeStrengthIndicatorData<'a>,
    pub params: EhlersDataSamplingRelativeStrengthIndicatorParams,
}

impl<'a> EhlersDataSamplingRelativeStrengthIndicatorInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        params: EhlersDataSamplingRelativeStrengthIndicatorParams,
    ) -> Self {
        Self {
            data: EhlersDataSamplingRelativeStrengthIndicatorData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        open: &'a [f64],
        close: &'a [f64],
        params: EhlersDataSamplingRelativeStrengthIndicatorParams,
    ) -> Self {
        Self {
            data: EhlersDataSamplingRelativeStrengthIndicatorData::Slices { open, close },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(
            candles,
            EhlersDataSamplingRelativeStrengthIndicatorParams::default(),
        )
    }

    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(14)
    }

    #[inline]
    pub fn get_open_close(&self) -> (&'a [f64], &'a [f64]) {
        match &self.data {
            EhlersDataSamplingRelativeStrengthIndicatorData::Candles { candles } => {
                (candles.open.as_slice(), candles.close.as_slice())
            }
            EhlersDataSamplingRelativeStrengthIndicatorData::Slices { open, close } => {
                (*open, *close)
            }
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct EhlersDataSamplingRelativeStrengthIndicatorBuilder {
    length: Option<usize>,
    kernel: Kernel,
}

impl Default for EhlersDataSamplingRelativeStrengthIndicatorBuilder {
    fn default() -> Self {
        Self {
            length: None,
            kernel: Kernel::Auto,
        }
    }
}

impl EhlersDataSamplingRelativeStrengthIndicatorBuilder {
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
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<
        EhlersDataSamplingRelativeStrengthIndicatorOutput,
        EhlersDataSamplingRelativeStrengthIndicatorError,
    > {
        let params = EhlersDataSamplingRelativeStrengthIndicatorParams {
            length: self.length,
        };
        ehlers_data_sampling_relative_strength_indicator_with_kernel(
            &EhlersDataSamplingRelativeStrengthIndicatorInput::from_candles(candles, params),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        open: &[f64],
        close: &[f64],
    ) -> Result<
        EhlersDataSamplingRelativeStrengthIndicatorOutput,
        EhlersDataSamplingRelativeStrengthIndicatorError,
    > {
        let params = EhlersDataSamplingRelativeStrengthIndicatorParams {
            length: self.length,
        };
        ehlers_data_sampling_relative_strength_indicator_with_kernel(
            &EhlersDataSamplingRelativeStrengthIndicatorInput::from_slices(open, close, params),
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<
        EhlersDataSamplingRelativeStrengthIndicatorStream,
        EhlersDataSamplingRelativeStrengthIndicatorError,
    > {
        EhlersDataSamplingRelativeStrengthIndicatorStream::try_new(
            EhlersDataSamplingRelativeStrengthIndicatorParams {
                length: self.length,
            },
        )
    }
}

#[derive(Debug, Error)]
pub enum EhlersDataSamplingRelativeStrengthIndicatorError {
    #[error("ehlers_data_sampling_relative_strength_indicator: Input data slice is empty.")]
    EmptyInputData,
    #[error(
        "ehlers_data_sampling_relative_strength_indicator: Input length mismatch: open = {open_len}, close = {close_len}"
    )]
    InputLengthMismatch { open_len: usize, close_len: usize },
    #[error("ehlers_data_sampling_relative_strength_indicator: All values are NaN.")]
    AllValuesNaN,
    #[error(
        "ehlers_data_sampling_relative_strength_indicator: Invalid length: length = {length}, data length = {data_len}"
    )]
    InvalidLength { length: usize, data_len: usize },
    #[error(
        "ehlers_data_sampling_relative_strength_indicator: Not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "ehlers_data_sampling_relative_strength_indicator: Output length mismatch: expected = {expected}, got = {got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error(
        "ehlers_data_sampling_relative_strength_indicator: Invalid range: start={start}, end={end}, step={step}"
    )]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("ehlers_data_sampling_relative_strength_indicator: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error(
        "ehlers_data_sampling_relative_strength_indicator: Output length mismatch: dst = {dst_len}, expected = {expected_len}"
    )]
    MismatchedOutputLen { dst_len: usize, expected_len: usize },
    #[error("ehlers_data_sampling_relative_strength_indicator: Invalid input: {msg}")]
    InvalidInput { msg: String },
}

#[derive(Debug, Clone)]
pub struct EhlersDataSamplingRelativeStrengthIndicatorStream {
    length: usize,
    open: Vec<f64>,
    close: Vec<f64>,
}

impl EhlersDataSamplingRelativeStrengthIndicatorStream {
    #[inline(always)]
    pub fn try_new(
        params: EhlersDataSamplingRelativeStrengthIndicatorParams,
    ) -> Result<Self, EhlersDataSamplingRelativeStrengthIndicatorError> {
        let length = params.length.unwrap_or(14);
        validate_length(length, usize::MAX)?;
        Ok(Self {
            length,
            open: Vec::new(),
            close: Vec::new(),
        })
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.open.clear();
        self.close.clear();
    }

    #[inline(always)]
    pub fn update(&mut self, open: f64, close: f64) -> Option<(f64, f64, f64)> {
        if !open.is_finite() || !close.is_finite() {
            self.reset();
            return None;
        }
        self.open.push(open);
        self.close.push(close);
        let out = ehlers_data_sampling_relative_strength_indicator(
            &EhlersDataSamplingRelativeStrengthIndicatorInput::from_slices(
                &self.open,
                &self.close,
                EhlersDataSamplingRelativeStrengthIndicatorParams {
                    length: Some(self.length),
                },
            ),
        )
        .ok()?;
        let idx = self.close.len().saturating_sub(1);
        let ds_rsi = out.ds_rsi[idx];
        let original_rsi = out.original_rsi[idx];
        let signal = out.signal[idx];
        if ds_rsi.is_finite() && original_rsi.is_finite() && signal.is_finite() {
            Some((ds_rsi, original_rsi, signal))
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn get_warmup_period(&self) -> usize {
        self.length
    }
}

#[inline(always)]
fn validate_length(
    length: usize,
    data_len: usize,
) -> Result<(), EhlersDataSamplingRelativeStrengthIndicatorError> {
    if length == 0 || (data_len != usize::MAX && length > data_len) {
        return Err(
            EhlersDataSamplingRelativeStrengthIndicatorError::InvalidLength { length, data_len },
        );
    }
    Ok(())
}

#[inline(always)]
fn longest_valid_run_pair(open: &[f64], close: &[f64]) -> usize {
    let mut best = 0usize;
    let mut cur = 0usize;
    for (&o, &c) in open.iter().zip(close.iter()) {
        if o.is_finite() && c.is_finite() {
            cur += 1;
            best = best.max(cur);
        } else {
            cur = 0;
        }
    }
    best
}

#[inline(always)]
fn validate_common(
    open: &[f64],
    close: &[f64],
    length: usize,
) -> Result<(), EhlersDataSamplingRelativeStrengthIndicatorError> {
    if open.is_empty() || close.is_empty() {
        return Err(EhlersDataSamplingRelativeStrengthIndicatorError::EmptyInputData);
    }
    if open.len() != close.len() {
        return Err(
            EhlersDataSamplingRelativeStrengthIndicatorError::InputLengthMismatch {
                open_len: open.len(),
                close_len: close.len(),
            },
        );
    }
    validate_length(length, close.len())?;
    let valid = longest_valid_run_pair(open, close);
    if valid == 0 {
        return Err(EhlersDataSamplingRelativeStrengthIndicatorError::AllValuesNaN);
    }
    if valid < length {
        return Err(
            EhlersDataSamplingRelativeStrengthIndicatorError::NotEnoughValidData {
                needed: length,
                valid,
            },
        );
    }
    Ok(())
}

#[inline(always)]
fn midpoint_series(open: &[f64], close: &[f64]) -> Vec<f64> {
    open.iter()
        .zip(close.iter())
        .map(|(&o, &c)| {
            if o.is_finite() && c.is_finite() {
                (o + c) * 0.5
            } else {
                f64::NAN
            }
        })
        .collect()
}

#[inline(always)]
fn classify_signal(slo: f64, prev_slo_nz: f64) -> f64 {
    if slo > 0.0 {
        if slo > prev_slo_nz {
            2.0
        } else {
            1.0
        }
    } else if slo < 0.0 {
        if slo < prev_slo_nz {
            -2.0
        } else {
            -1.0
        }
    } else {
        0.0
    }
}

#[inline(always)]
fn fill_signal(ds_rsi: &[f64], signal: &mut [f64]) {
    signal.fill(f64::NAN);
    let mut prev_slo = f64::NAN;
    for i in 0..ds_rsi.len() {
        let ds = ds_rsi[i];
        if !ds.is_finite() {
            prev_slo = f64::NAN;
            continue;
        }
        let prev_ds_nz = if i > 0 && ds_rsi[i - 1].is_finite() {
            ds_rsi[i - 1]
        } else {
            0.0
        };
        let slo = ds - prev_ds_nz;
        signal[i] = classify_signal(slo, if prev_slo.is_finite() { prev_slo } else { 0.0 });
        prev_slo = slo;
    }
}

#[inline(always)]
fn map_rsi_error(
    err: crate::indicators::rsi::RsiError,
) -> EhlersDataSamplingRelativeStrengthIndicatorError {
    match err {
        crate::indicators::rsi::RsiError::EmptyInputData => {
            EhlersDataSamplingRelativeStrengthIndicatorError::EmptyInputData
        }
        crate::indicators::rsi::RsiError::AllValuesNaN => {
            EhlersDataSamplingRelativeStrengthIndicatorError::AllValuesNaN
        }
        crate::indicators::rsi::RsiError::InvalidPeriod { period, data_len } => {
            EhlersDataSamplingRelativeStrengthIndicatorError::InvalidLength {
                length: period,
                data_len,
            }
        }
        crate::indicators::rsi::RsiError::NotEnoughValidData { needed, valid } => {
            EhlersDataSamplingRelativeStrengthIndicatorError::NotEnoughValidData { needed, valid }
        }
        crate::indicators::rsi::RsiError::OutputLengthMismatch { expected, got } => {
            EhlersDataSamplingRelativeStrengthIndicatorError::OutputLengthMismatch { expected, got }
        }
        crate::indicators::rsi::RsiError::InvalidRange { start, end, step } => {
            EhlersDataSamplingRelativeStrengthIndicatorError::InvalidRange { start, end, step }
        }
        crate::indicators::rsi::RsiError::InvalidKernelForBatch(kernel) => {
            EhlersDataSamplingRelativeStrengthIndicatorError::InvalidKernelForBatch(kernel)
        }
    }
}

#[inline(always)]
fn compute_into_outputs(
    open: &[f64],
    close: &[f64],
    length: usize,
    kernel: Kernel,
    ds_rsi: &mut [f64],
    original_rsi: &mut [f64],
    signal: &mut [f64],
) -> Result<(), EhlersDataSamplingRelativeStrengthIndicatorError> {
    validate_common(open, close, length)?;
    if ds_rsi.len() != close.len()
        || original_rsi.len() != close.len()
        || signal.len() != close.len()
    {
        return Err(
            EhlersDataSamplingRelativeStrengthIndicatorError::OutputLengthMismatch {
                expected: close.len(),
                got: ds_rsi.len().min(original_rsi.len()).min(signal.len()),
            },
        );
    }

    let midpoint = midpoint_series(open, close);
    let params = RsiParams {
        period: Some(length),
    };
    let original_input = RsiInput::from_slice(close, params.clone());
    let ds_input = RsiInput::from_slice(&midpoint, params);
    rsi_into_slice(original_rsi, &original_input, kernel).map_err(map_rsi_error)?;
    rsi_into_slice(ds_rsi, &ds_input, kernel).map_err(map_rsi_error)?;
    fill_signal(ds_rsi, signal);
    Ok(())
}

#[inline]
pub fn ehlers_data_sampling_relative_strength_indicator(
    input: &EhlersDataSamplingRelativeStrengthIndicatorInput,
) -> Result<
    EhlersDataSamplingRelativeStrengthIndicatorOutput,
    EhlersDataSamplingRelativeStrengthIndicatorError,
> {
    ehlers_data_sampling_relative_strength_indicator_with_kernel(input, Kernel::Auto)
}

pub fn ehlers_data_sampling_relative_strength_indicator_with_kernel(
    input: &EhlersDataSamplingRelativeStrengthIndicatorInput,
    kernel: Kernel,
) -> Result<
    EhlersDataSamplingRelativeStrengthIndicatorOutput,
    EhlersDataSamplingRelativeStrengthIndicatorError,
> {
    let (open, close) = input.get_open_close();
    let length = input.get_length();
    validate_common(open, close, length)?;

    let _chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let mut ds_rsi = alloc_uninit_f64(close.len());
    let mut original_rsi = alloc_uninit_f64(close.len());
    let mut signal = alloc_uninit_f64(close.len());

    compute_into_outputs(
        open,
        close,
        length,
        kernel,
        &mut ds_rsi,
        &mut original_rsi,
        &mut signal,
    )?;

    Ok(EhlersDataSamplingRelativeStrengthIndicatorOutput {
        ds_rsi,
        original_rsi,
        signal,
    })
}

pub fn ehlers_data_sampling_relative_strength_indicator_into_slice(
    dst_ds_rsi: &mut [f64],
    dst_original_rsi: &mut [f64],
    dst_signal: &mut [f64],
    input: &EhlersDataSamplingRelativeStrengthIndicatorInput,
    kernel: Kernel,
) -> Result<(), EhlersDataSamplingRelativeStrengthIndicatorError> {
    let (open, close) = input.get_open_close();
    let length = input.get_length();
    validate_common(open, close, length)?;
    if dst_ds_rsi.len() != close.len()
        || dst_original_rsi.len() != close.len()
        || dst_signal.len() != close.len()
    {
        return Err(
            EhlersDataSamplingRelativeStrengthIndicatorError::OutputLengthMismatch {
                expected: close.len(),
                got: dst_ds_rsi
                    .len()
                    .min(dst_original_rsi.len())
                    .min(dst_signal.len()),
            },
        );
    }

    let _chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    compute_into_outputs(
        open,
        close,
        length,
        kernel,
        dst_ds_rsi,
        dst_original_rsi,
        dst_signal,
    )
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn ehlers_data_sampling_relative_strength_indicator_into(
    input: &EhlersDataSamplingRelativeStrengthIndicatorInput,
    dst_ds_rsi: &mut [f64],
    dst_original_rsi: &mut [f64],
    dst_signal: &mut [f64],
) -> Result<(), EhlersDataSamplingRelativeStrengthIndicatorError> {
    ehlers_data_sampling_relative_strength_indicator_into_slice(
        dst_ds_rsi,
        dst_original_rsi,
        dst_signal,
        input,
        Kernel::Auto,
    )
}

#[derive(Debug, Clone, Copy)]
pub struct EhlersDataSamplingRelativeStrengthIndicatorBatchRange {
    pub length: (usize, usize, usize),
}

impl Default for EhlersDataSamplingRelativeStrengthIndicatorBatchRange {
    fn default() -> Self {
        Self {
            length: (14, 14, 0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EhlersDataSamplingRelativeStrengthIndicatorBatchOutput {
    pub ds_rsi: Vec<f64>,
    pub original_rsi: Vec<f64>,
    pub signal: Vec<f64>,
    pub combos: Vec<EhlersDataSamplingRelativeStrengthIndicatorParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct EhlersDataSamplingRelativeStrengthIndicatorBatchBuilder {
    range: EhlersDataSamplingRelativeStrengthIndicatorBatchRange,
    kernel: Kernel,
}

impl Default for EhlersDataSamplingRelativeStrengthIndicatorBatchBuilder {
    fn default() -> Self {
        Self {
            range: EhlersDataSamplingRelativeStrengthIndicatorBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl EhlersDataSamplingRelativeStrengthIndicatorBatchBuilder {
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
    pub fn length_static(mut self, value: usize) -> Self {
        self.range.length = (value, value, 0);
        self
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        open: &[f64],
        close: &[f64],
    ) -> Result<
        EhlersDataSamplingRelativeStrengthIndicatorBatchOutput,
        EhlersDataSamplingRelativeStrengthIndicatorError,
    > {
        ehlers_data_sampling_relative_strength_indicator_batch_with_kernel(
            open,
            close,
            &self.range,
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<
        EhlersDataSamplingRelativeStrengthIndicatorBatchOutput,
        EhlersDataSamplingRelativeStrengthIndicatorError,
    > {
        ehlers_data_sampling_relative_strength_indicator_batch_with_kernel(
            candles.open.as_slice(),
            candles.close.as_slice(),
            &self.range,
            self.kernel,
        )
    }
}

#[inline(always)]
fn expand_grid_checked(
    range: &EhlersDataSamplingRelativeStrengthIndicatorBatchRange,
) -> Result<
    Vec<EhlersDataSamplingRelativeStrengthIndicatorParams>,
    EhlersDataSamplingRelativeStrengthIndicatorError,
> {
    let (start, end, step) = range.length;
    if start == 0 || end == 0 {
        return Err(
            EhlersDataSamplingRelativeStrengthIndicatorError::InvalidRange { start, end, step },
        );
    }
    if step == 0 {
        return Ok(vec![EhlersDataSamplingRelativeStrengthIndicatorParams {
            length: Some(start),
        }]);
    }
    if start > end {
        return Err(
            EhlersDataSamplingRelativeStrengthIndicatorError::InvalidRange { start, end, step },
        );
    }

    let mut out = Vec::new();
    let mut cur = start;
    loop {
        out.push(EhlersDataSamplingRelativeStrengthIndicatorParams { length: Some(cur) });
        if cur >= end {
            break;
        }
        let next = cur.saturating_add(step);
        if next <= cur {
            return Err(
                EhlersDataSamplingRelativeStrengthIndicatorError::InvalidRange { start, end, step },
            );
        }
        cur = next.min(end);
        if cur == out.last().and_then(|p| p.length).unwrap_or(cur) {
            break;
        }
    }
    Ok(out)
}

#[inline(always)]
pub fn expand_grid_ehlers_data_sampling_relative_strength_indicator(
    range: &EhlersDataSamplingRelativeStrengthIndicatorBatchRange,
) -> Vec<EhlersDataSamplingRelativeStrengthIndicatorParams> {
    expand_grid_checked(range).unwrap_or_default()
}

pub fn ehlers_data_sampling_relative_strength_indicator_batch_with_kernel(
    open: &[f64],
    close: &[f64],
    sweep: &EhlersDataSamplingRelativeStrengthIndicatorBatchRange,
    kernel: Kernel,
) -> Result<
    EhlersDataSamplingRelativeStrengthIndicatorBatchOutput,
    EhlersDataSamplingRelativeStrengthIndicatorError,
> {
    match kernel {
        Kernel::Auto
        | Kernel::Scalar
        | Kernel::ScalarBatch
        | Kernel::Avx2
        | Kernel::Avx2Batch
        | Kernel::Avx512
        | Kernel::Avx512Batch => {}
        other => {
            return Err(
                EhlersDataSamplingRelativeStrengthIndicatorError::InvalidKernelForBatch(other),
            )
        }
    }

    let combos = expand_grid_checked(sweep)?;
    let max_length = combos
        .iter()
        .map(|params| params.length.unwrap_or(14))
        .max()
        .unwrap_or(0);
    validate_common(open, close, max_length)?;

    let rows = combos.len();
    let cols = close.len();
    let mut ds_mu = make_uninit_matrix(rows, cols);
    let mut orig_mu = make_uninit_matrix(rows, cols);
    let mut sig_mu = make_uninit_matrix(rows, cols);
    let warmups: Vec<usize> = combos
        .iter()
        .map(|params| params.length.unwrap_or(14))
        .collect();
    init_matrix_prefixes(&mut ds_mu, cols, &warmups);
    init_matrix_prefixes(&mut orig_mu, cols, &warmups);
    init_matrix_prefixes(&mut sig_mu, cols, &warmups);

    let mut ds_rsi = unsafe {
        Vec::from_raw_parts(
            ds_mu.as_mut_ptr() as *mut f64,
            ds_mu.len(),
            ds_mu.capacity(),
        )
    };
    let mut original_rsi = unsafe {
        Vec::from_raw_parts(
            orig_mu.as_mut_ptr() as *mut f64,
            orig_mu.len(),
            orig_mu.capacity(),
        )
    };
    let mut signal = unsafe {
        Vec::from_raw_parts(
            sig_mu.as_mut_ptr() as *mut f64,
            sig_mu.len(),
            sig_mu.capacity(),
        )
    };
    std::mem::forget(ds_mu);
    std::mem::forget(orig_mu);
    std::mem::forget(sig_mu);

    ehlers_data_sampling_relative_strength_indicator_batch_inner_into(
        open,
        close,
        sweep,
        kernel,
        true,
        &mut ds_rsi,
        &mut original_rsi,
        &mut signal,
    )?;

    Ok(EhlersDataSamplingRelativeStrengthIndicatorBatchOutput {
        ds_rsi,
        original_rsi,
        signal,
        combos,
        rows,
        cols,
    })
}

pub fn ehlers_data_sampling_relative_strength_indicator_batch_slice(
    open: &[f64],
    close: &[f64],
    sweep: &EhlersDataSamplingRelativeStrengthIndicatorBatchRange,
    kernel: Kernel,
) -> Result<
    EhlersDataSamplingRelativeStrengthIndicatorBatchOutput,
    EhlersDataSamplingRelativeStrengthIndicatorError,
> {
    ehlers_data_sampling_relative_strength_indicator_batch_inner(open, close, sweep, kernel, false)
}

pub fn ehlers_data_sampling_relative_strength_indicator_batch_par_slice(
    open: &[f64],
    close: &[f64],
    sweep: &EhlersDataSamplingRelativeStrengthIndicatorBatchRange,
    kernel: Kernel,
) -> Result<
    EhlersDataSamplingRelativeStrengthIndicatorBatchOutput,
    EhlersDataSamplingRelativeStrengthIndicatorError,
> {
    ehlers_data_sampling_relative_strength_indicator_batch_inner(open, close, sweep, kernel, true)
}

fn ehlers_data_sampling_relative_strength_indicator_batch_inner(
    open: &[f64],
    close: &[f64],
    sweep: &EhlersDataSamplingRelativeStrengthIndicatorBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<
    EhlersDataSamplingRelativeStrengthIndicatorBatchOutput,
    EhlersDataSamplingRelativeStrengthIndicatorError,
> {
    let combos = expand_grid_checked(sweep)?;
    let rows = combos.len();
    let cols = close.len();
    let total = rows.checked_mul(cols).ok_or_else(|| {
        EhlersDataSamplingRelativeStrengthIndicatorError::InvalidInput {
            msg: "ehlers_data_sampling_relative_strength_indicator: rows*cols overflow in batch"
                .to_string(),
        }
    })?;

    let mut ds_mu = make_uninit_matrix(rows, cols);
    let mut orig_mu = make_uninit_matrix(rows, cols);
    let mut sig_mu = make_uninit_matrix(rows, cols);
    let warmups: Vec<usize> = combos
        .iter()
        .map(|params| params.length.unwrap_or(14))
        .collect();
    init_matrix_prefixes(&mut ds_mu, cols, &warmups);
    init_matrix_prefixes(&mut orig_mu, cols, &warmups);
    init_matrix_prefixes(&mut sig_mu, cols, &warmups);

    let mut ds_rsi = unsafe {
        Vec::from_raw_parts(
            ds_mu.as_mut_ptr() as *mut f64,
            ds_mu.len(),
            ds_mu.capacity(),
        )
    };
    let mut original_rsi = unsafe {
        Vec::from_raw_parts(
            orig_mu.as_mut_ptr() as *mut f64,
            orig_mu.len(),
            orig_mu.capacity(),
        )
    };
    let mut signal = unsafe {
        Vec::from_raw_parts(
            sig_mu.as_mut_ptr() as *mut f64,
            sig_mu.len(),
            sig_mu.capacity(),
        )
    };
    std::mem::forget(ds_mu);
    std::mem::forget(orig_mu);
    std::mem::forget(sig_mu);

    debug_assert_eq!(ds_rsi.len(), total);
    debug_assert_eq!(original_rsi.len(), total);
    debug_assert_eq!(signal.len(), total);

    ehlers_data_sampling_relative_strength_indicator_batch_inner_into(
        open,
        close,
        sweep,
        kernel,
        parallel,
        &mut ds_rsi,
        &mut original_rsi,
        &mut signal,
    )?;

    Ok(EhlersDataSamplingRelativeStrengthIndicatorBatchOutput {
        ds_rsi,
        original_rsi,
        signal,
        combos,
        rows,
        cols,
    })
}

fn ehlers_data_sampling_relative_strength_indicator_batch_inner_into(
    open: &[f64],
    close: &[f64],
    sweep: &EhlersDataSamplingRelativeStrengthIndicatorBatchRange,
    kernel: Kernel,
    parallel: bool,
    out_ds_rsi: &mut [f64],
    out_original_rsi: &mut [f64],
    out_signal: &mut [f64],
) -> Result<
    Vec<EhlersDataSamplingRelativeStrengthIndicatorParams>,
    EhlersDataSamplingRelativeStrengthIndicatorError,
> {
    match kernel {
        Kernel::Auto
        | Kernel::Scalar
        | Kernel::ScalarBatch
        | Kernel::Avx2
        | Kernel::Avx2Batch
        | Kernel::Avx512
        | Kernel::Avx512Batch => {}
        other => {
            return Err(
                EhlersDataSamplingRelativeStrengthIndicatorError::InvalidKernelForBatch(other),
            )
        }
    }

    let combos = expand_grid_checked(sweep)?;
    let len = close.len();
    if len == 0 || open.is_empty() {
        return Err(EhlersDataSamplingRelativeStrengthIndicatorError::EmptyInputData);
    }
    if open.len() != close.len() {
        return Err(
            EhlersDataSamplingRelativeStrengthIndicatorError::InputLengthMismatch {
                open_len: open.len(),
                close_len: close.len(),
            },
        );
    }

    let total = combos.len().checked_mul(len).ok_or_else(|| {
        EhlersDataSamplingRelativeStrengthIndicatorError::InvalidInput {
            msg:
                "ehlers_data_sampling_relative_strength_indicator: rows*cols overflow in batch_into"
                    .to_string(),
        }
    })?;
    if out_ds_rsi.len() != total || out_original_rsi.len() != total || out_signal.len() != total {
        return Err(
            EhlersDataSamplingRelativeStrengthIndicatorError::MismatchedOutputLen {
                dst_len: out_ds_rsi
                    .len()
                    .min(out_original_rsi.len())
                    .min(out_signal.len()),
                expected_len: total,
            },
        );
    }

    let max_length = combos
        .iter()
        .map(|params| params.length.unwrap_or(14))
        .max()
        .unwrap_or(0);
    validate_common(open, close, max_length)?;

    let _chosen = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other => other,
    };

    let worker = |row: usize, dst_ds: &mut [f64], dst_orig: &mut [f64], dst_sig: &mut [f64]| {
        dst_ds.fill(f64::NAN);
        dst_orig.fill(f64::NAN);
        dst_sig.fill(f64::NAN);
        let length = combos[row].length.unwrap_or(14);
        let _ = compute_into_outputs(
            open,
            close,
            length,
            kernel.to_non_batch(),
            dst_ds,
            dst_orig,
            dst_sig,
        );
    };

    if parallel && combos.len() > 1 {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_ds_rsi
                .par_chunks_mut(len)
                .zip(out_original_rsi.par_chunks_mut(len))
                .zip(out_signal.par_chunks_mut(len))
                .enumerate()
                .for_each(|(row, ((dst_ds, dst_orig), dst_sig))| {
                    worker(row, dst_ds, dst_orig, dst_sig)
                });
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, ((dst_ds, dst_orig), dst_sig)) in out_ds_rsi
                .chunks_mut(len)
                .zip(out_original_rsi.chunks_mut(len))
                .zip(out_signal.chunks_mut(len))
                .enumerate()
            {
                worker(row, dst_ds, dst_orig, dst_sig);
            }
        }
    } else {
        for (row, ((dst_ds, dst_orig), dst_sig)) in out_ds_rsi
            .chunks_mut(len)
            .zip(out_original_rsi.chunks_mut(len))
            .zip(out_signal.chunks_mut(len))
            .enumerate()
        {
            worker(row, dst_ds, dst_orig, dst_sig);
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "ehlers_data_sampling_relative_strength_indicator")]
#[pyo3(signature = (open_, close, length=14, kernel=None))]
pub fn ehlers_data_sampling_relative_strength_indicator_py<'py>(
    py: Python<'py>,
    open_: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    length: usize,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let open_ = open_.as_slice()?;
    let close = close.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let input = EhlersDataSamplingRelativeStrengthIndicatorInput::from_slices(
        open_,
        close,
        EhlersDataSamplingRelativeStrengthIndicatorParams {
            length: Some(length),
        },
    );
    let out = py
        .allow_threads(|| {
            ehlers_data_sampling_relative_strength_indicator_with_kernel(&input, kern)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        out.ds_rsi.into_pyarray(py),
        out.original_rsi.into_pyarray(py),
        out.signal.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyclass(name = "EhlersDataSamplingRelativeStrengthIndicatorStream")]
pub struct EhlersDataSamplingRelativeStrengthIndicatorStreamPy {
    stream: EhlersDataSamplingRelativeStrengthIndicatorStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl EhlersDataSamplingRelativeStrengthIndicatorStreamPy {
    #[new]
    #[pyo3(signature = (length=14))]
    fn new(length: usize) -> PyResult<Self> {
        let stream = EhlersDataSamplingRelativeStrengthIndicatorStream::try_new(
            EhlersDataSamplingRelativeStrengthIndicatorParams {
                length: Some(length),
            },
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, open_: f64, close: f64) -> Option<(f64, f64, f64)> {
        self.stream.update(open_, close)
    }

    fn reset(&mut self) {
        self.stream.reset();
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "ehlers_data_sampling_relative_strength_indicator_batch")]
#[pyo3(signature = (open_, close, length_range=(14,14,0), kernel=None))]
pub fn ehlers_data_sampling_relative_strength_indicator_batch_py<'py>(
    py: Python<'py>,
    open_: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let open_ = open_.as_slice()?;
    let close = close.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let output = py
        .allow_threads(|| {
            ehlers_data_sampling_relative_strength_indicator_batch_with_kernel(
                open_,
                close,
                &EhlersDataSamplingRelativeStrengthIndicatorBatchRange {
                    length: length_range,
                },
                kern,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let rows = output.rows;
    let cols = output.cols;
    let dict = PyDict::new(py);
    dict.set_item(
        "ds_rsi",
        output.ds_rsi.into_pyarray(py).reshape((rows, cols))?,
    )?;
    dict.set_item(
        "original_rsi",
        output.original_rsi.into_pyarray(py).reshape((rows, cols))?,
    )?;
    dict.set_item(
        "signal",
        output.signal.into_pyarray(py).reshape((rows, cols))?,
    )?;
    dict.set_item(
        "lengths",
        output
            .combos
            .iter()
            .map(|params| params.length.unwrap_or(14) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_ehlers_data_sampling_relative_strength_indicator_module(
    m: &Bound<'_, PyModule>,
) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        ehlers_data_sampling_relative_strength_indicator_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        ehlers_data_sampling_relative_strength_indicator_batch_py,
        m
    )?)?;
    m.add_class::<EhlersDataSamplingRelativeStrengthIndicatorStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EhlersDataSamplingRelativeStrengthIndicatorBatchConfig {
    pub length_range: Vec<usize>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = ehlers_data_sampling_relative_strength_indicator_js)]
pub fn ehlers_data_sampling_relative_strength_indicator_js(
    open_: &[f64],
    close: &[f64],
    length: usize,
) -> Result<JsValue, JsValue> {
    let input = EhlersDataSamplingRelativeStrengthIndicatorInput::from_slices(
        open_,
        close,
        EhlersDataSamplingRelativeStrengthIndicatorParams {
            length: Some(length),
        },
    );
    let out = ehlers_data_sampling_relative_strength_indicator_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("ds_rsi"),
        &serde_wasm_bindgen::to_value(&out.ds_rsi).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("original_rsi"),
        &serde_wasm_bindgen::to_value(&out.original_rsi).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("signal"),
        &serde_wasm_bindgen::to_value(&out.signal).unwrap(),
    )?;
    Ok(obj.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = ehlers_data_sampling_relative_strength_indicator_batch_js)]
pub fn ehlers_data_sampling_relative_strength_indicator_batch_js(
    open_: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: EhlersDataSamplingRelativeStrengthIndicatorBatchConfig =
        serde_wasm_bindgen::from_value(config)
            .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.length_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: length_range must have exactly 3 elements [start, end, step]",
        ));
    }
    let out = ehlers_data_sampling_relative_strength_indicator_batch_with_kernel(
        open_,
        close,
        &EhlersDataSamplingRelativeStrengthIndicatorBatchRange {
            length: (
                config.length_range[0],
                config.length_range[1],
                config.length_range[2],
            ),
        },
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("ds_rsi"),
        &serde_wasm_bindgen::to_value(&out.ds_rsi).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("original_rsi"),
        &serde_wasm_bindgen::to_value(&out.original_rsi).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("signal"),
        &serde_wasm_bindgen::to_value(&out.signal).unwrap(),
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
pub fn ehlers_data_sampling_relative_strength_indicator_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(3 * len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_data_sampling_relative_strength_indicator_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, 3 * len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_data_sampling_relative_strength_indicator_into(
    open_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length: usize,
) -> Result<(), JsValue> {
    if open_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to ehlers_data_sampling_relative_strength_indicator_into",
        ));
    }
    unsafe {
        let open_ = std::slice::from_raw_parts(open_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, 3 * len);
        let (dst_ds_rsi, tail) = out.split_at_mut(len);
        let (dst_original_rsi, dst_signal) = tail.split_at_mut(len);
        let input = EhlersDataSamplingRelativeStrengthIndicatorInput::from_slices(
            open_,
            close,
            EhlersDataSamplingRelativeStrengthIndicatorParams {
                length: Some(length),
            },
        );
        ehlers_data_sampling_relative_strength_indicator_into_slice(
            dst_ds_rsi,
            dst_original_rsi,
            dst_signal,
            &input,
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_data_sampling_relative_strength_indicator_batch_into(
    open_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length_start: usize,
    length_end: usize,
    length_step: usize,
) -> Result<usize, JsValue> {
    if open_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to ehlers_data_sampling_relative_strength_indicator_batch_into",
        ));
    }
    let sweep = EhlersDataSamplingRelativeStrengthIndicatorBatchRange {
        length: (length_start, length_end, length_step),
    };
    let combos = expand_grid_checked(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let total = rows
        .checked_mul(len)
        .and_then(|v| v.checked_mul(3))
        .ok_or_else(|| {
            JsValue::from_str(
                "rows*cols overflow in ehlers_data_sampling_relative_strength_indicator_batch_into",
            )
        })?;
    unsafe {
        let open_ = std::slice::from_raw_parts(open_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        let split = rows * len;
        let (dst_ds_rsi, tail) = out.split_at_mut(split);
        let (dst_original_rsi, dst_signal) = tail.split_at_mut(split);
        ehlers_data_sampling_relative_strength_indicator_batch_inner_into(
            open_,
            close,
            &sweep,
            Kernel::Auto,
            false,
            dst_ds_rsi,
            dst_original_rsi,
            dst_signal,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }
    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_data_sampling_relative_strength_indicator_output_into_js(
    open_: &[f64],
    close: &[f64],
    length: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = ehlers_data_sampling_relative_strength_indicator_js(open_, close, length)?;
    crate::write_wasm_object_f64_outputs(
        "ehlers_data_sampling_relative_strength_indicator_output_into_js",
        &value,
        out,
    )
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_data_sampling_relative_strength_indicator_batch_output_into_js(
    open_: &[f64],
    close: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = ehlers_data_sampling_relative_strength_indicator_batch_js(open_, close, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "ehlers_data_sampling_relative_strength_indicator_batch_output_into_js",
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

    fn sample_open_close(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let close = (0..len)
            .map(|i| {
                let x = i as f64;
                100.0 + x * 0.12 + (x * 0.17).sin() * 2.1 + (x * 0.031).cos() * 0.8
            })
            .collect::<Vec<_>>();
        let open = close
            .iter()
            .enumerate()
            .map(|(i, &c)| c - (i as f64 * 0.11).sin() * 0.9 - 0.15)
            .collect::<Vec<_>>();
        let high = open
            .iter()
            .zip(close.iter())
            .map(|(&o, &c)| o.max(c) + 0.5)
            .collect::<Vec<_>>();
        let low = open
            .iter()
            .zip(close.iter())
            .map(|(&o, &c)| o.min(c) - 0.5)
            .collect::<Vec<_>>();
        (open, high, low, close)
    }

    fn assert_series_close(left: &[f64], right: &[f64], tol: f64) {
        assert_eq!(left.len(), right.len());
        for (&a, &b) in left.iter().zip(right.iter()) {
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan());
            } else {
                assert!((a - b).abs() <= tol, "left={a} right={b}");
            }
        }
    }

    #[test]
    fn ehlers_data_sampling_relative_strength_indicator_output_contract(
    ) -> Result<(), Box<dyn Error>> {
        let (open, _high, _low, close) = sample_open_close(256);
        let input = EhlersDataSamplingRelativeStrengthIndicatorInput::from_slices(
            &open,
            &close,
            EhlersDataSamplingRelativeStrengthIndicatorParams { length: Some(14) },
        );
        let out = ehlers_data_sampling_relative_strength_indicator(&input)?;
        assert_eq!(out.ds_rsi.len(), close.len());
        assert_eq!(out.original_rsi.len(), close.len());
        assert_eq!(out.signal.len(), close.len());
        assert_eq!(out.ds_rsi.iter().position(|v| v.is_finite()), Some(14));
        assert_eq!(
            out.original_rsi.iter().position(|v| v.is_finite()),
            Some(14)
        );
        assert_eq!(out.signal.iter().position(|v| v.is_finite()), Some(14));
        Ok(())
    }

    #[test]
    fn ehlers_data_sampling_relative_strength_indicator_into_matches_api(
    ) -> Result<(), Box<dyn Error>> {
        let (open, _high, _low, close) = sample_open_close(220);
        let input = EhlersDataSamplingRelativeStrengthIndicatorInput::from_slices(
            &open,
            &close,
            EhlersDataSamplingRelativeStrengthIndicatorParams { length: Some(14) },
        );
        let base = ehlers_data_sampling_relative_strength_indicator(&input)?;
        let mut ds = vec![f64::NAN; close.len()];
        let mut orig = vec![f64::NAN; close.len()];
        let mut sig = vec![f64::NAN; close.len()];
        ehlers_data_sampling_relative_strength_indicator_into_slice(
            &mut ds,
            &mut orig,
            &mut sig,
            &input,
            Kernel::Auto,
        )?;
        assert_series_close(&base.ds_rsi, &ds, 1e-12);
        assert_series_close(&base.original_rsi, &orig, 1e-12);
        assert_series_close(&base.signal, &sig, 1e-12);
        Ok(())
    }

    #[test]
    fn ehlers_data_sampling_relative_strength_indicator_stream_matches_batch(
    ) -> Result<(), Box<dyn Error>> {
        let (open, _high, _low, close) = sample_open_close(256);
        let batch = ehlers_data_sampling_relative_strength_indicator(
            &EhlersDataSamplingRelativeStrengthIndicatorInput::from_slices(
                &open,
                &close,
                EhlersDataSamplingRelativeStrengthIndicatorParams { length: Some(14) },
            ),
        )?;

        let mut stream = EhlersDataSamplingRelativeStrengthIndicatorStream::try_new(
            EhlersDataSamplingRelativeStrengthIndicatorParams { length: Some(14) },
        )?;
        let mut ds = Vec::with_capacity(close.len());
        let mut orig = Vec::with_capacity(close.len());
        let mut sig = Vec::with_capacity(close.len());
        for (&o, &c) in open.iter().zip(close.iter()) {
            if let Some((a, b, d)) = stream.update(o, c) {
                ds.push(a);
                orig.push(b);
                sig.push(d);
            } else {
                ds.push(f64::NAN);
                orig.push(f64::NAN);
                sig.push(f64::NAN);
            }
        }
        assert_series_close(&batch.ds_rsi, &ds, 1e-12);
        assert_series_close(&batch.original_rsi, &orig, 1e-12);
        assert_series_close(&batch.signal, &sig, 1e-12);
        Ok(())
    }

    #[test]
    fn ehlers_data_sampling_relative_strength_indicator_batch_single_matches_single(
    ) -> Result<(), Box<dyn Error>> {
        let (open, _high, _low, close) = sample_open_close(256);
        let single = ehlers_data_sampling_relative_strength_indicator(
            &EhlersDataSamplingRelativeStrengthIndicatorInput::from_slices(
                &open,
                &close,
                EhlersDataSamplingRelativeStrengthIndicatorParams { length: Some(14) },
            ),
        )?;
        let batch = ehlers_data_sampling_relative_strength_indicator_batch_with_kernel(
            &open,
            &close,
            &EhlersDataSamplingRelativeStrengthIndicatorBatchRange {
                length: (14, 14, 0),
            },
            Kernel::Auto,
        )?;
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, close.len());
        assert_series_close(&single.ds_rsi, &batch.ds_rsi, 1e-12);
        assert_series_close(&single.original_rsi, &batch.original_rsi, 1e-12);
        assert_series_close(&single.signal, &batch.signal, 1e-12);
        Ok(())
    }

    #[test]
    fn ehlers_data_sampling_relative_strength_indicator_rejects_invalid_params() {
        let (open, _high, _low, close) = sample_open_close(64);
        let err = ehlers_data_sampling_relative_strength_indicator(
            &EhlersDataSamplingRelativeStrengthIndicatorInput::from_slices(
                &open,
                &close,
                EhlersDataSamplingRelativeStrengthIndicatorParams { length: Some(0) },
            ),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            EhlersDataSamplingRelativeStrengthIndicatorError::InvalidLength { .. }
        ));
    }

    #[test]
    fn ehlers_data_sampling_relative_strength_indicator_dispatch_compute_returns_outputs(
    ) -> Result<(), Box<dyn Error>> {
        let (open, high, low, close) = sample_open_close(192);
        for output_id in ["ds_rsi", "original_rsi", "signal"] {
            let req = IndicatorComputeRequest {
                indicator_id: "ehlers_data_sampling_relative_strength_indicator",
                output_id: Some(output_id),
                data: IndicatorDataRef::Ohlc {
                    open: &open,
                    high: &high,
                    low: &low,
                    close: &close,
                },
                params: &[ParamKV {
                    key: "length",
                    value: ParamValue::Int(14),
                }],
                kernel: Kernel::Auto,
            };
            let out = compute_cpu(req)?;
            assert_eq!(out.output_id, output_id);
            assert_eq!(out.rows, 1);
            assert_eq!(out.cols, close.len());
        }
        Ok(())
    }
}
